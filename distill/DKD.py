import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller


# 输出：DKD 损失。
def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    # 通过 _get_gt_mask 和 _get_other_mask 函数获取两个掩码，用于区分目标类和其他类。
    gt_mask = _get_gt_mask(logits_student, target)
    logits_teacher = logits_teacher.to(gt_mask.device)
    logits_student = logits_student.to(gt_mask.device)
    other_mask = _get_other_mask(logits_student, target)
    # 对学生模型和教师模型的输出进行 softmax 操作，并根据掩码合并结果。
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    # 计算交叉熵损失（F.kl_div）
    # 分别计算两个部分的损失（tckd_loss 和 nckd_loss）。
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    # 返回加权和的损失。
    return alpha * tckd_loss + beta * nckd_loss


# 辅助函数 _get_gt_mask 和 _get_other_mask
# _get_gt_mask：生成一个与目标标签对应的 one-hot 掩码。
def _get_gt_mask(logits, target):
    logits = logits.to(target.device)
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    # zeros = torch.zeros_like(logits)
    # print("\nStep 1 - Zeros:")
    # print(zeros)
    #
    # # Step 2: 将 target 进行维度扩展
    # target_unsqueezed = target.unsqueeze(1)
    # print("\nStep 2 - Target Unsqueezed:")
    # print(target_unsqueezed)
    #
    # # Step 3: 使用 scatter_ 方法将目标类别的位置标记为 1
    # scattered = zeros.scatter_(1, target_unsqueezed, 1)
    # print("\nStep 3 - Scattered:")
    # print(scattered)
    #
    # # Step 4: 将张量转换为布尔类型
    # mask = scattered.bool()
    # print("\nStep 4 - Mask:")
    return mask


# # _get_other_mask：生成一个与目标标签对应位置相反的 one-hot 掩码。
def _get_other_mask(logits, target):
    logits = logits.to(target.device)
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


# 输入：t（一个 tensor）、mask1 和 mask2（两个掩码）。
# 输出：根据两个掩码对 tensor 进行合并的结果。
def cat_mask(t, mask1, mask2):
    t = t.to(mask1.device)
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt


# 继承自 Distiller 类，该类可能是一个用于模型蒸馏的基类。
# 构造函数接受学生模型、教师模型和配置信息。
# forward_train 方法用于在训练阶段计算 DKD 损失：
class DKD(Distiller):
    """Decoupled Knowledge Distillation(CVPR 2022)"""

    def __init__(self, student, teacher, cfg):
        super(DKD, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.DKD.CE_WEIGHT
        self.alpha = cfg.DKD.ALPHA
        self.beta = cfg.DKD.BETA
        self.temperature = cfg.DKD.T
        self.warmup = cfg.DKD.WARMUP

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # 计算交叉熵损失（loss_ce）。
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        # 计算 DKD 损失（loss_dkd）。
        # 根据训练的 epoch 数量动态调整 DKD 损失的权重。
        loss_dkd = min(kwargs["epoch"] / self.warmup, 1.0) * dkd_loss(
            logits_student,
            logits_teacher,
            target,
            self.alpha,
            self.beta,
            self.temperature,
        )
        # print('loss_dkd', loss_dkd)
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_dkd,
        }
        # 返回学生模型的 logits 和损失字典。
        return logits_student, losses_dict
