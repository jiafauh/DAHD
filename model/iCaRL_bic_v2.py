import torch.nn.parallel
import torch
import torch.nn as nn
from .consistency_loss import get_robust_loss
from .temporalShiftModule.ops.models import TSN
from .temporalShiftModule.ops.utils import AverageMeter, accuracy
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
import os
from torch.cuda.amp import autocast


class BiasLayer(nn.Module):
    def __init__(self):
        super(BiasLayer, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, requires_grad=True))
        self.beta = nn.Parameter(torch.zeros(1, requires_grad=True))

    def forward(self, x):
        return self.alpha * x + self.beta

    def printParam(self, i):
        print(i, self.alpha.item(), self.beta.item())


class iCaRL_BIC(nn.Module):
    # 初始化模型的各种参数，并创建一个基于TSN模型的特征编码器。
    def __init__(self, conf_model, num_class, conf_checkpoint):
        super(iCaRL_BIC, self).__init__()

        self.conf_checkpoint = conf_checkpoint
        self.conf_model = conf_model
        num_segments = conf_model['num_segments']
        modality = conf_model['modality']
        arch = conf_model['arch']
        consensus_type = conf_model['consensus_type']
        dropout = conf_model['dropout']
        img_feature_dim = conf_model['img_feature_dim']
        no_partialbn = conf_model['no_partialbn']
        pretrain = conf_model['pretrain']
        shift = conf_model['shift']
        shift_div = conf_model['shift_div']
        shift_place = conf_model['shift_place']
        self.fc_lr5 = conf_model['fc_lr5']
        temporal_pool = conf_model['temporal_pool']
        non_local = conf_model['non_local']
        self.feature_encoder = TSN(num_class, num_segments, modality,
                                   base_model=arch,
                                   consensus_type=consensus_type,
                                   dropout=dropout,
                                   img_feature_dim=img_feature_dim,
                                   partial_bn=not no_partialbn,
                                   pretrain=pretrain,
                                   is_shift=shift, shift_div=shift_div, shift_place=shift_place,
                                   fc_lr5=self.fc_lr5,
                                   temporal_pool=temporal_pool,
                                   non_local=non_local)

        # feature_dim = self.feature_encoder.new_fc.in_features
        # self.new_fc = nn.Linear(feature_dim, num_class)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.crop_size = self.feature_encoder.crop_size
        self.scale_size = self.feature_encoder.scale_size
        self.input_mean = self.feature_encoder.input_mean
        self.input_std = self.feature_encoder.input_std
        self.is_activityNet = conf_model['is_activityNet']

        if torch.cuda.device_count() > 1:
            self.feature_encoder = nn.DataParallel(self.feature_encoder)

        print("Let's use", torch.cuda.device_count(), "GPUs!", flush=True)
        self.feature_encoder.to(self.device)
        self.list_bias_layers = []
        self.list_splits = []
        #         self.bias_layer = BiasLayer()
        #         self.bias_layer.to(self.device)

        self.n_classes = num_class
        self.n_known = 0

        # self.compute_means = True
        self.exemplar_means = []
        self.memory = {}
        self.list_val_acc_ii = {'val': [], 'test': []}

        self.adv_lambda = conf_model['adv_lambda']
        print('adv_lambda: ', self.adv_lambda)

    # 返回模型的优化策略，用于设置优化器的参数。
    def get_optim_policies(self):
        if torch.cuda.device_count() > 1:
            return self.feature_encoder.module.get_optim_policies()
        else:
            return self.feature_encoder.get_optim_policies()

    # 该方法用于获取数据增强操作，例如随机裁剪、随机翻转等。
    def get_augmentation(self, flip=True):
        if torch.cuda.device_count() > 1:
            return self.feature_encoder.module.get_augmentation(flip)
        else:
            return self.feature_encoder.get_augmentation(flip)

    # 用于扩展模型的分类器以适应新类别的加入。
    def augment_classification(self, num_new_classes, device):
        # 增加模型的类别数，通过添加新类别的数量
        self.n_classes += num_new_classes

        # 增加模型的类别数，通过添加新类别的数量
        if torch.cuda.device_count() > 1:
            return self.feature_encoder.module.augment_classification(num_new_classes, device)
        else:
            return self.feature_encoder.augment_classification(num_new_classes, device)

        # self.bias_layer = BiasLayer()

    def set_losses(self, cls_loss, dist_loss):
        self.cls_loss = cls_loss
        self.dist_loss = dist_loss

    # 初始化模型的各种参数，并创建一个基于TSN模型的特征编码器。
    def forward(self, x):
        x = self.feature_encoder(x, get_emb=False)
        if self.n_known > 0:
            list_out = []
            init_val = 0
            final_val = 0
            for i, val_lim in enumerate(self.list_splits):
                x_old_classes = x[:, init_val:val_lim]
                init_val = val_lim
                x_old_classes = self.list_bias_layers[i](x_old_classes)
                list_out.append(x_old_classes)
            x = torch.cat(list_out, dim=1)
        return x

    # 用于将样本添加到模型的记忆中，这些样本将用作老类别的样本。
    def add_samples_to_mem(self, cilsettask, data, m):
        for class_id, videos in data.items():
            data_class = {class_id: videos}
            class_loader = cilsettask.get_dataloader(data_class, sample_frame=True)
            features = []
            video_names = []
            for _, video_name, video, _, _ in class_loader:
                video = video.to(self.device)
                feature = self.feature_encoder(video, get_emb=True).data.cpu().numpy()
                feature = feature / np.linalg.norm(feature)
                features.append(feature[0])
                video_names.append(video_name)

            features = np.array(features)
            class_mean = np.mean(features, axis=0)
            class_mean = class_mean / np.linalg.norm(class_mean)  # Normalize

            exemplar_set = []
            exemplar_features = []  # list of Variables of shape (feature_size,)
            list_selected_idx = []
            for k in range(m):
                S = np.sum(exemplar_features, axis=0)
                phi = features
                mu = class_mean
                mu_p = 1.0 / (k + 1) * (phi + S)
                mu_p = mu_p / np.linalg.norm(mu_p)
                #                     i = np.argmin(np.sqrt(np.sum((mu - mu_p) ** 2, axis=1)))
                dist = np.sqrt(np.sum((mu - mu_p) ** 2, axis=1))
                if k <= len(dist) - 2:
                    list_idx = np.argpartition(dist, k)[:k + 1]
                elif k < len(dist):
                    fixed_k = len(dist) - 2
                    list_idx = np.argpartition(dist, fixed_k)[:fixed_k + 2]
                else:
                    break

                for idx in list_idx:
                    if idx not in list_selected_idx:
                        list_selected_idx.append(idx)
                        exemplar_set.append(video_names[idx][0])
                        exemplar_features.append(features[idx])
                        break

                """
                print "Selected example", i
                print "|exemplar_mean - class_mean|:",
                print np.linalg.norm((np.mean(exemplar_features, axis=0) - class_mean))
                #features = np.delete(features, i, axis=0)
                """

            if self.is_activityNet:
                new_exemplar_set = []
                for video_name in exemplar_set:
                    idx = video_name.split('_')[-1]
                    for vid in videos:
                        if vid['id'] == int(idx):
                            new_exemplar_set.append(vid)
                exemplar_set = new_exemplar_set

            self.memory[class_id] = exemplar_set

        self.memory = {class_id: videos[:m] for class_id, videos in self.memory.items()}
        for class_id, videos in self.memory.items():
            print('Memory... Class: {}, num videos: {}'.format(class_id, len(videos)))

    def load_best_checkpoint(self, path_model, current_task):
        path_best_model = path_model.format('Best_Model')
        if os.path.exists(path_best_model):
            checkpoint_dict = torch.load(path_best_model)
            task_to_load = checkpoint_dict['current_task']
            if task_to_load == current_task:
                self.feature_encoder.load_state_dict(checkpoint_dict['state_dict'])
                self.list_splits = checkpoint_dict['list_splits']
                self.list_bias_layers = checkpoint_dict['bias_list']

    def save_checkpoint(self, dict_to_save, path_model, is_best):
        if is_best:
            print('Saving ... ')
            best_model = path_model.format('Best_Model')
            torch.save(dict_to_save, best_model)
            print("Save Best Networks for task: {}, epoch: {}".format(dict_to_save['current_task'] + 1,
                                                                      dict_to_save['current_epoch'] + 1), flush=True)

    # 用于在验证集上评估模型的性能，并记录准确率和损失等指标。
    def validate(self, val_cilDatasetList, current_task_id):
        top1 = AverageMeter()
        total_acc = AverageMeter()
        val_loaders_list = val_cilDatasetList.get_valSet_by_taskNum(current_task_id + 1)

        # switch to evaluate mode
        self.feature_encoder.eval()
        for bias_layer in self.list_bias_layers:
            bias_layer.eval()

        with torch.no_grad():
            for n_task, (val_loader, num_classes) in enumerate(val_loaders_list):
                for _, _, videos, _, target in val_loader:
                    target = target.to(self.device)
                    videos = videos.to(self.device)
                    # compute output
                    output = self.forward(videos)

                    # measure accuracy and record loss
                    acc_val = accuracy(output.data, target, topk=(1,))[0]

                    top1.update(acc_val.item(), videos.size(0))

                total_acc.update(top1.avg, num_classes)
                print('Train... task : {}, acc with classifier: {}'.format(n_task, top1.avg))
                top1.reset()
        output = ('Pre Testing Results: Pre_Acc {total_acc.avg:.3f}'
                  .format(total_acc=total_acc))
        print(output)
        return total_acc.avg

    # 用于在验证集上评估模型的性能，并记录准确率和损失等指标。
    def final_validate(self, val_cilDatasetList, current_task_id, experiment, type_val='val'):
        top1 = AverageMeter()
        total_acc = AverageMeter()
        val_loaders_list = val_cilDatasetList.get_valSet_by_taskNum(current_task_id + 1)
        BWF = AverageMeter()

        # switch to evaluate mode
        self.feature_encoder.eval()
        for bias_layer in self.list_bias_layers:
            bias_layer.eval()

        with torch.no_grad():
            for n_task, (val_loader, num_classes) in enumerate(val_loaders_list):
                for _, _, videos, _, target in val_loader:
                    target = target.to(self.device)
                    videos = videos.to(self.device)
                    # compute output
                    output = self.forward(videos)

                    # check the accuracy function
                    acc_val = accuracy(output.data, target, topk=(1,))[0]

                    # top1.update(acc_val.item(), videos.size(0))
                    top1.update(acc_val, videos.size(0))

                experiment.log_metric("Acc_task_{}".format(n_task + 1), top1.avg, step=current_task_id + 1)
                if n_task == current_task_id:
                    self.list_val_acc_ii[type_val].append(top1.avg)
                elif n_task < current_task_id:
                    forgetting = self.list_val_acc_ii[type_val][n_task] - top1.avg
                    BWF.update(forgetting, num_classes)
                total_acc.update(top1.avg, num_classes)
                top1.reset()

            output = ('Testing Results: Acc {total_acc.avg:.3f}'
                      .format(total_acc=total_acc))
            print(output)
            experiment.log_metric("Total_Acc_Per_task", total_acc.avg, step=current_task_id + 1)
            experiment.log_metric("Total_BWF_Per_task", BWF.avg, step=current_task_id + 1)
        return total_acc.avg

    # 定义了模型的训练过程，包括前向传播、计算损失、反向传播和更新参数等步骤。
    def train_step(self, len_data, dataloader_cil, num_epochs, optimizer, stage_id, best_acc_val, experiment, task_id,
                   val_cilDatasetList):

        # 设置部分 Batch Normalization 层，根据配置文件决定是否启用部分 BN。
        self.set_partialBN()

        eval_freq = self.conf_checkpoint['eval_freq']
        path_model = self.conf_checkpoint['path_model']
        # 初始化一个温度参数 T 为 2。
        T = 2
        n_classes = self.feature_encoder.new_fc.out_features
        # 计算当前已知类别的样本的 softmax 概率值并存储在变量 q 中。这个 softmax 概率值在训练过程中用于计算蒸馏损失。
        with torch.no_grad():
            q = torch.zeros(len_data, self.n_known).to(self.device)
            for indices, _, videos, _, labels in dataloader_cil:
                videos = videos.to(self.device)
                indices = indices.to(self.device)
                #             g = F.sigmoid(self.forward(videos))
                pre_p = self.forward(videos)
                g = F.softmax(pre_p[:, :self.n_known] / T, dim=1)
                q[indices] = g.data
            q = Variable(q).to(self.device)

        for epoch in range(num_epochs):

            # 设置部分 BN 层。
            self.set_partialBN()
            # 根据阶段（初始化阶段或微调阶段）决定是否设置模型和偏置层为训练模式或评估模式。
            if stage_id == 0:
                self.feature_encoder.train()
                for bias_layer in self.list_bias_layers:
                    bias_layer.eval()
            else:
                self.feature_encoder.eval()
                for bias_layer in self.list_bias_layers:
                    bias_layer.train()

            # 初始化两个 AverageMeter 对象，用于计算准确率和损失的平均值。
            acc_Avg = AverageMeter()
            loss_Avg = AverageMeter()
            # 遍历训练数据加载器，对每个批次执行以下步骤：
            for i, (indices, _, videos, videos_aug, labels) in enumerate(dataloader_cil):
                # 将视频和标签移动到设备上。
                videos = videos.to(self.device)
                labels = labels.to(self.device)
                indices = indices.to(self.device)
                videos_aug = videos_aug.to(self.device)

                # 将优化器的梯度清零。
                optimizer.zero_grad()
                with autocast():
                    # 通过前向传播计算模型的输出。
                    g = self.forward(videos)
                    g_aug = self.forward(videos_aug)

                    # Classification loss for new classes
                    #                     loss = self.cls_loss(g, labels)
                    # 计算分类损失和训练集上的准确率。对抗性损失
                    loss = get_robust_loss(self.cls_loss, g, g_aug, labels, adv_lambda=self.adv_lambda,
                                           cr_lambda=0, l_outputs=None, l_outputs_aug=None)
                    acc_train = accuracy(g.data, labels, topk=(1,))[0]

                    # Distilation loss for old classes
                    # 如果存在已知类别并且处于初始化阶段，则计算蒸馏损失。
                    if self.n_known > 0 and stage_id == 0:
                        print('distill loss')
                        alpha = self.n_known / n_classes
                        # g = F.sigmoid(g)
                        q_i = q[indices]
                        #                         dist_loss = sum(self.dist_loss(g[:,y], q_i[:,y])\
                        #                                 for y in range(self.n_known))

                        logp = F.log_softmax(g[:, :self.n_known] / T, dim=1)
                        loss_soft_target = -torch.mean(torch.sum(q_i * logp, dim=1))
                        loss = loss_soft_target * alpha + (1 - alpha) * loss

                # 反向传播并更新模型参数。
                loss.backward()

                clip_gradient = self.conf_model['clip_gradient']
                if clip_gradient is not None:
                    total_norm = clip_grad_norm_(self.feature_encoder.parameters(), clip_gradient)

                optimizer.step()

                # 记录训练集上的准确率和损失。
                experiment.log_metric("Acc_task_{}_stage_{}".format(task_id + 1, stage_id), acc_train.item())
                experiment.log_metric("Loss_task_{}_stage_{}".format(task_id + 1, stage_id), loss.item())
                loss_Avg.update(loss.item(), videos.size(0))
                acc_Avg.update(acc_train.item(), videos.size(0))

                # 如果达到记录频率，则输出当前阶段、当前周期、当前损失。
                if (i + 1) % 2 == 0:
                    print('Stage: %d, Epoch [%d/%d], Loss: %.4f'
                          % (stage_id, epoch + 1, num_epochs, loss.item()))

            # 记录每个周期的平均准确率和损失。
            experiment.log_metric("Epoch_Acc_task_{}_stage_{}".format(task_id + 1, stage_id), acc_Avg.avg)
            experiment.log_metric("Epoch_Loss_task_{}_stage_{}".format(task_id + 1, stage_id), loss_Avg.avg)

            # 如果达到评估频率或者当前周期是最后一个周期，则在验证集上进行验证，记录最佳准确率，并保存模型参数。
            if (epoch + 1) % eval_freq == 0 or epoch == num_epochs - 1:
                acc_val = self.validate(val_cilDatasetList, task_id)
                is_best = acc_val >= best_acc_val
                best_acc_val = max(acc_val, best_acc_val)
                output_best = 'Best Pre Acc Val@1: %.3f\n' % best_acc_val
                print(output_best)
                dict_to_save = {'state_dict': self.feature_encoder.state_dict(), 'bias_list': self.list_bias_layers,
                                'accuracy': acc_val, 'current_epoch': epoch, 'current_task': task_id,
                                'list_splits': self.list_splits}

                self.save_checkpoint(dict_to_save, path_model, is_best)
        # 返回最佳验证集准确率。
        return best_acc_val

    # 根据配置参数设置是否使用部分批量归一化。
    def set_partialBN(self):
        no_partialbn = self.conf_model['no_partialbn']
        if no_partialbn:
            if torch.cuda.device_count() > 1:
                self.feature_encoder.module.partialBN(False)
            else:
                self.feature_encoder.partialBN(False)
        else:
            if torch.cuda.device_count() > 1:
                self.feature_encoder.module.partialBN(True)
            else:
                self.feature_encoder.partialBN(True)

    def train(self, train_train_dataloader_cil, train_val_dataloader_cil, len_train_train_data, len_train_val_data,
              optimizer, num_epochs, experiment, task_id, val_cilDatasetList):

        # self.compute_means = True
        best_acc_val = 0
        path_model = self.conf_checkpoint['path_model']

        with experiment.train():

            # 如果没有提供验证数据集（train_val_dataloader_cil is None），则直接在训练数据集上训练模型，并将所有参数设置为可训练。
            if train_val_dataloader_cil is None:
                for param in self.feature_encoder.parameters():
                    param.requires_grad = True
                bias_layer = BiasLayer()
                bias_layer.to(self.device)
                for param in bias_layer.parameters():
                    param.requires_grad = False
                self.list_bias_layers.append(bias_layer)
                self.list_splits.append(self.n_classes)
                self.train_step(len_train_train_data, train_train_dataloader_cil, num_epochs, optimizer, 0,
                                best_acc_val, experiment, task_id, val_cilDatasetList)
            # 如果提供了验证数据集，则进行两个阶段的训练：初始化阶段和微调阶段。
            else:
                list_dataloader = [train_train_dataloader_cil, train_val_dataloader_cil]

                bias_layer = BiasLayer()
                bias_layer.to(self.device)
                for param in bias_layer.parameters():
                    param.requires_grad = False

                self.list_bias_layers.append(bias_layer)
                self.list_splits.append(self.n_classes)

                for id_phase, dataloader_cil in enumerate(list_dataloader):

                    # 在初始化阶段（id_phase == 0），所有模型参数都被设置为可训练，然后在训练数据集上进行训练。
                    if id_phase == 0:
                        print('Init phase: ', id_phase + 1)
                        for param in self.feature_encoder.parameters():
                            param.requires_grad = True

                        best_acc_val = self.train_step(len_train_train_data, dataloader_cil, num_epochs, optimizer,
                                                       id_phase, best_acc_val, experiment, task_id, val_cilDatasetList)
                        print('finish phase: {}, acc: {}'.format(id_phase + 1, best_acc_val))
                    # 在微调阶段（id_phase == 1），加载最佳的模型参数（来自初始化阶段的最佳模型），然后将模型的所有参数设置为不可训练，
                    # 除了最后一个添加的偏置层的参数。这个偏置层用于增加模型的适应性以及适应新类别的特性。然后在验证数据集上微调模型。
                    else:
                        print('Init phase: ', id_phase + 1)
                        self.load_best_checkpoint(path_model, task_id)

                        for param in self.feature_encoder.parameters():
                            param.requires_grad = False

                        bias_layer = self.list_bias_layers[-1]
                        for param in bias_layer.parameters():
                            param.requires_grad = True
                        bias_optimizer = torch.optim.SGD(bias_layer.parameters(), lr=0.001)

                        for i, bias_layer in enumerate(self.list_bias_layers):
                            bias_layer.printParam(i)

                        best_acc_val = self.train_step(len_train_val_data, dataloader_cil, num_epochs, bias_optimizer,
                                                       id_phase, best_acc_val, experiment, task_id, val_cilDatasetList)
                        print('finish phase: {}, acc: {}'.format(id_phase + 1, best_acc_val))
                        for i, bias_layer in enumerate(self.list_bias_layers):
                            bias_layer.printParam(i)
                        for param in self.list_bias_layers[-1].parameters():
                            param.requires_grad = False

            self.load_best_checkpoint(path_model, task_id)
