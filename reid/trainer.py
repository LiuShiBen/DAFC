
import PIL.Image as Image
import time
from .loss import TripletLoss, CrossEntropyLabelSmooth, SoftTripletLoss
import torch
import torch.nn as nn
from .utils.meters import AverageMeter
from .utils.my_tools import *
import numpy as np
from torch.nn import functional as F
from scipy.linalg import sqrtm

class Trainer:
    def __init__(self, args, model, optimizer, num_classes,
                 data_loader_train, training_phase, add_num=0, margin=0.0,
                 ):
        self.num_task_experts = args.num_task_experts  #5
        self.model = model
        self.model.cuda()
        self.data_loader_train = data_loader_train
        self.training_phase = training_phase
        self.add_num = add_num
        self.gamma = 0.5
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes).cuda()
        self.criterion_triple = SoftTripletLoss(margin=margin).cuda()
        self.trip_hard = TripletLoss(margin=margin).cuda()
        self.T = 2
        self.consistency = 0.2
        self.consistency_rampup = 100.0
        self.train_iters = len(self.data_loader_train)
        self.optimizer = optimizer


    def train(self, epoch):

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_base = AverageMeter()
        losses_KL = AverageMeter()


        end = time.time()
        self.model.train()
        #print("training_step:", training_stepping)
        if self.training_phase == 1:
            for i in range(len(self.data_loader_train)):
                train_inputs = self.data_loader_train.next()
                data_time.update(time.time() - end)
                imgs, targets, cids, domains = self._parse_data(train_inputs)
                # print("imgs:", imgs.shape, targets.shape)
                targets += self.add_num
                #Current network output
                cls_out, _, feat_out, _, _= self.model(imgs, self.training_phase)
                loss_ce = self.CE_loss(cls_out, targets)
                loss_tp = self.Hard_loss(feat_out, targets)

                loss = loss_ce + loss_tp# + (loss_i2t + loss_t2i) / 2
                losses_base.update(loss)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                batch_time.update(time.time() - end)
                end = time.time()
                if (i + 1) == self.train_iters or (i + 1) % (self.train_iters // 4) == 0:
                    print('Epoch: [{}][{}/{}]\t'
                          'Time {:.3f} ({:.3f})\t'
                          'Loss_base {:.3f} ({:.3f})\t'
                          .format(epoch, i + 1, self.train_iters,
                                  batch_time.val, batch_time.avg,
                                  losses_base.val, losses_base.avg))
        if 1 < self.training_phase <= 5:
            if epoch < 30:
                for i in range(len(self.data_loader_train)):
                    train_inputs = self.data_loader_train.next()
                    data_time.update(time.time() - end)
                    imgs, targets, cids, domains = self._parse_data(train_inputs)
                    targets += self.add_num
                    # Current network output
                    _, cls_adapt, _, feat_adapt, _= self.model(imgs, self.training_phase)

                    loss_ce = self.CE_loss(cls_adapt, targets)
                    loss_tp = self.Hard_loss(feat_adapt, targets)
                    loss = loss_ce + loss_tp
                    #print("loss:", loss)
                    losses_base.update(loss)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    batch_time.update(time.time() - end)
                    end = time.time()
                    if (i + 1) == self.train_iters or (i + 1) % (self.train_iters // 4) == 0:
                        print('Epoch: [{}][{}/{}]\t'
                              'Time {:.3f} ({:.3f})\t'
                              'Loss_base {:.3f} ({:.3f})\t'
                              .format(epoch, i + 1, self.train_iters,
                                      batch_time.val, batch_time.avg,
                                      losses_base.val, losses_base.avg))
            if 30 <=epoch < 60:
                for i in range(len(self.data_loader_train)):
                    train_inputs = self.data_loader_train.next()
                    data_time.update(time.time() - end)
                    imgs, targets, cids, domains = self._parse_data(train_inputs)
                    # print("imgs:", imgs.shape, targets.shape)
                    targets += self.add_num
                    # Current network output
                    cls_out, _, feat_out, _, expert_out= self.model(imgs, self.training_phase)
                    # corss-entroy loss of new samples
                    loss_ce = self.CE_loss(cls_out, targets)
                    # triplet loss of new samples
                    loss_tp = self.Hard_loss(feat_out, targets)

                    loss_kl = self.kl_loss(expert_out, self.training_phase, self.num_task_experts)
                    #loss_kl = self.fid_loss(expert_out, self.training_phase, self.num_expert)
                    # loss_i2t = self.SupConLoss(img_proj, text_proj, targets, targets)
                    # loss_t2i = self.SupConLoss(text_proj, img_proj, targets, targets)
                    loss = loss_ce + loss_tp + loss_kl  # + (loss_i2t + loss_t2i) / 2
                    losses_KL.update(loss_kl)
                    losses_base.update(loss)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    batch_time.update(time.time() - end)
                    end = time.time()
                    if (i + 1) == self.train_iters or (i + 1) % (self.train_iters // 4) == 0:
                        print('Epoch: [{}][{}/{}]\t'
                              'Time {:.3f} ({:.3f})\t'
                              'Loss_base {:.3f} ({:.3f})\t'
                              'Loss_kL {:.3f} ({:.3f})\t'
                              .format(epoch, i + 1, self.train_iters,
                                      batch_time.val, batch_time.avg,
                                      losses_base.val, losses_base.avg,
                                      losses_KL.val, losses_KL.avg))
            else:
                pass
        #print("sucessful.....", self.training_phase)

    def cal_statistics(self, feature):
        B,C,K = feature.shape
        #print(B,C,K)
        features = feature[:,:,0]
        for i in range(1, K):
            #print(features.shape, feature[:,:,i].shape, i)
            features = torch.cat((features, feature[:,:,i]), dim=1)
        features = features.permute(1, 0).cpu()
        mean = torch.mean(features, dim = 0)
        sigma = torch.cov(features.T)
        return mean, sigma

    def cal_fid(self, mu1, sigmal1, mu2, sigmal2):
        diff = mu1-mu2
        #print("diff:", diff.shape)
        sigmal1_n = sigmal1.detach().numpy()
        sigmal2_n = sigmal2.detach().numpy()
        covmean, _ = sqrtm(sigmal1_n @ sigmal2_n, disp = False)
        #print("covmean:", covmean.shape)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid = diff @ diff + torch.trace(sigmal1 + sigmal2 - 2 * torch.from_numpy(covmean))
        return fid

    def fid_loss(self, expert_feat, training_step, num_expert):
        loss = []
        if training_step==2:
            expert1_feat = expert_feat[:,:, num_expert * (training_step-2):num_expert*(training_step-1)]
            expert2_feat = expert_feat[:, :, num_expert * (training_step - 1):num_expert * training_step]
            mu1, cov1 = self. cal_statistics(expert1_feat)
            mu2, cov2 = self.cal_statistics(expert2_feat)
            loss = self.cal_fid(mu1, cov1, mu2, cov2)
            #loss = torch.mean(torch.stack(loss))
        elif training_step==3:
            expert1_feat = expert_feat[:, :, num_expert * (training_step - 3) : num_expert * (training_step - 2)]
            expert2_feat = expert_feat[:, :, num_expert * (training_step - 2) : num_expert * (training_step - 1)]
            expert3_feat = expert_feat[:, :, num_expert * (training_step - 1) : num_expert * training_step]
            mu1, cov1 = self.cal_statistics(expert1_feat)
            mu2, cov2 = self.cal_statistics(expert2_feat)
            mu3, cov3 = self.cal_statistics(expert3_feat)
            loss12 = self.cal_fid(mu1, cov1, mu2, cov2)
            loss23 = self.cal_fid(mu2, cov2, mu3, cov3)
            loss = (loss12 + loss23) / 2
        elif training_step==4:
            expert1_feat = expert_feat[:, :, num_expert * (training_step - 4) : num_expert * (training_step - 3)]
            expert2_feat = expert_feat[:, :, num_expert * (training_step - 3) : num_expert * (training_step - 2)]
            expert3_feat = expert_feat[:, :, num_expert * (training_step - 2) : num_expert * (training_step - 1)]
            expert4_feat = expert_feat[:, :, num_expert * (training_step - 1) : num_expert * training_step]
            mu1, cov1 = self.cal_statistics(expert1_feat)
            mu2, cov2 = self.cal_statistics(expert2_feat)
            mu3, cov3 = self.cal_statistics(expert3_feat)
            mu4, cov4 = self.cal_statistics(expert4_feat)
            loss12 = self.cal_fid(mu1, cov1, mu2, cov2)
            loss23 = self.cal_fid(mu2, cov2, mu3, cov3)
            loss34 = self.cal_fid(mu3, cov3, mu4, cov4)
            loss = (loss12 + loss23 + loss34) / 3
        elif training_step==5:
            expert1_feat = expert_feat[:, :, num_expert * (training_step - 5) : num_expert * (training_step - 4)]
            expert2_feat = expert_feat[:, :, num_expert * (training_step - 4) : num_expert * (training_step - 3)]
            expert3_feat = expert_feat[:, :, num_expert * (training_step - 3) : num_expert * (training_step - 2)]
            expert4_feat = expert_feat[:, :, num_expert * (training_step - 2) : num_expert * (training_step - 1)]
            expert5_feat = expert_feat[:, :, num_expert * (training_step - 1) : num_expert * training_step]
            mu1, cov1 = self.cal_statistics(expert1_feat)
            mu2, cov2 = self.cal_statistics(expert2_feat)
            mu3, cov3 = self.cal_statistics(expert3_feat)
            mu4, cov4 = self.cal_statistics(expert4_feat)
            mu5, cov5 = self.cal_statistics(expert5_feat)
            loss12 = self.cal_fid(mu1, cov1, mu2, cov2)
            loss23 = self.cal_fid(mu2, cov2, mu3, cov3)
            loss34 = self.cal_fid(mu3, cov3, mu4, cov4)
            loss45 = self.cal_fid(mu4, cov4, mu5, cov5)
            loss = (loss12 + loss23 + loss34 + loss45) / 4
        else:
           pass
        return loss
        print("FID_LOSS", loss)



    def kl_loss(self, expert_feat, training_step, num_expert):
        loss = []
        if training_step==2:
            expert1_feat = expert_feat[:,:, num_expert * (training_step-2):num_expert*(training_step-1)]
            expert2_feat = expert_feat[:, :, num_expert * (training_step - 1):num_expert * training_step]
            for i in range(num_expert):
                distance_matrix_21 = self.cosine_distance(expert2_feat[:, :, i], expert1_feat[:, :, i])
                #print("distance_matrix:", distance_matrix.shape)
                loss.append(torch.mean(distance_matrix_21))
            loss = torch.mean(torch.stack(loss))
        elif training_step==3:
            expert1_feat = expert_feat[:, :, num_expert * (training_step - 3) : num_expert * (training_step - 2)]
            expert2_feat = expert_feat[:, :, num_expert * (training_step - 2) : num_expert * (training_step - 1)]
            expert3_feat = expert_feat[:, :, num_expert * (training_step - 1) : num_expert * training_step]
            for i in range(num_expert):
                distance_matrix_31 = self.cosine_distance(expert3_feat[:, :, i], expert1_feat[:, :, i])
                distance_matrix_32 = self.cosine_distance(expert3_feat[:, :, i], expert2_feat[:, :, i])
                #print("distance_matrix:", distance_matrix.shape)
                #loss.append(torch.mean(distance_matrix_31) * 1/2 + torch.mean(distance_matrix_32)*1/3)
                loss.append((torch.mean(distance_matrix_31) + torch.mean(distance_matrix_32)) / 2.0)
            loss = torch.mean(torch.stack(loss))
        elif training_step==4:
            expert1_feat = expert_feat[:, :, num_expert * (training_step - 4) : num_expert * (training_step - 3)]
            expert2_feat = expert_feat[:, :, num_expert * (training_step - 3) : num_expert * (training_step - 2)]
            expert3_feat = expert_feat[:, :, num_expert * (training_step - 2) : num_expert * (training_step - 1)]
            expert4_feat = expert_feat[:, :, num_expert * (training_step - 1) : num_expert * training_step]
            for i in range(num_expert):
                distance_matrix_41 = self.cosine_distance(expert4_feat[:, :, i], expert1_feat[:, :, i])
                distance_matrix_42 = self.cosine_distance(expert4_feat[:, :, i], expert2_feat[:, :, i])
                distance_matrix_43 = self.cosine_distance(expert4_feat[:, :, i], expert3_feat[:, :, i])
                #print("distance_matrix:", distance_matrix.shape)
                #loss.append(torch.mean(distance_matrix_41) * 1/2 + torch.mean(distance_matrix_42) * 1/3 + torch.mean(distance_matrix_43) * 1/4)
                loss.append((torch.mean(distance_matrix_41) + torch.mean(distance_matrix_42) + torch.mean(distance_matrix_43))/3.0)
            loss = torch.mean(torch.stack(loss))
        elif training_step==5:
            expert1_feat = expert_feat[:, :, num_expert * (training_step - 5) : num_expert * (training_step - 4)]
            expert2_feat = expert_feat[:, :, num_expert * (training_step - 4) : num_expert * (training_step - 3)]
            expert3_feat = expert_feat[:, :, num_expert * (training_step - 3) : num_expert * (training_step - 2)]
            expert4_feat = expert_feat[:, :, num_expert * (training_step - 2) : num_expert * (training_step - 1)]
            expert5_feat = expert_feat[:, :, num_expert * (training_step - 1) : num_expert * training_step]
            for i in range(num_expert):
                distance_matrix_51 = self.cosine_distance(expert5_feat[:, :, i], expert1_feat[:, :, i])
                distance_matrix_52 = self.cosine_distance(expert5_feat[:, :, i], expert2_feat[:, :, i])
                distance_matrix_53 = self.cosine_distance(expert5_feat[:, :, i], expert3_feat[:, :, i])
                distance_matrix_54 = self.cosine_distance(expert5_feat[:, :, i], expert4_feat[:, :, i])
                #print("distance_matrix:", distance_matrix.shape)
                #loss.append(torch.mean(distance_matrix_51) * 1/2 + torch.mean(distance_matrix_52) * 1/3 + torch.mean(distance_matrix_53) * 1/4 + torch.mean(distance_matrix_54) * 1/5)
                loss.append((torch.mean(distance_matrix_51) + torch.mean(distance_matrix_52) + torch.mean(distance_matrix_53) + torch.mean(distance_matrix_54)) / 4.0)
            loss = torch.mean(torch.stack(loss))
        else:
           pass
        return loss
        print("KL_LOSS", loss)

    def get_current_consistency_weight(self, epoch):
        return self.consistency * self.sigmoid_rampup(epoch, self.consistency_rampup)

    def sigmoid_rampup(self, current, rampup_length):
        # Exponential rampup
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))
    def _parse_data(self, inputs):
        imgs, _, pids, cids, domains = inputs
        inputs = imgs.cuda()

        targets = pids.cuda()
        return inputs, targets, cids, domains

    def CE_loss(self, s_outputs, targets):
        loss_ce = self.criterion_ce(s_outputs, targets)  #ID loss
        return loss_ce

    def Tri_loss(self, s_features, targets):
        fea_loss = []
        for i in range(len(s_features)):
            loss_tr = self.criterion_triple(s_features[i], s_features[i], targets) #tri loss
            fea_loss.append(loss_tr)
        loss_tr = sum(fea_loss)# / len(fea_loss)
        return loss_tr

    def Hard_loss(self, s_features, targets):
        fea_loss = []
        for i in range(0, len(s_features)):
            loss_tr = self.trip_hard(s_features[i], targets)[0]
            fea_loss.append(loss_tr)
        loss_tr = sum(fea_loss)# / len(fea_loss)
        return loss_tr

    def cosine_distance(sself, input1, input2):
        """Computes cosine distance.
        Args:
            input1 (torch.Tensor): 2-D feature matrix.
            input2 (torch.Tensor): 2-D feature matrix.
        Returns:
            torch.Tensor: distance matrix.
        """
        input1_normed = F.normalize(input1, p=2, dim=1)
        input2_normed = F.normalize(input2, p=2, dim=1)
        distmat = 1 - torch.mm(input1_normed, input2_normed.t())
        return distmat

    def loss_kd_js(self, old_logit, new_logit):
        old_logits = old_logit.detach()
        new_logits = new_logit
        #print("new_logits:", new_logits.shape, old_logits.shape)
        p_s = F.log_softmax((new_logits + old_logits)/(2*self.T), dim=1)
        p_t = F.softmax(old_logits/self.T, dim=1)
        p_t2 = F.softmax(new_logits/self.T, dim=1)
        loss = 0.5*F.kl_div(p_s, p_t, reduction='batchmean')*(self.T**2) + 0.5*F.kl_div(p_s, p_t2, reduction='batchmean')*(self.T**2)
        return loss

    def Dissimilar(self, feat2, feat3):
        feat23 = torch.cat((feat2.unsqueeze(1), feat3.unsqueeze(1)), 1)
        B, N, C = feat23.shape
        dist_mat = self.cosine_dist(feat23, feat23)
        top_triu = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
        _dist = dist_mat[:, top_triu]
        dist = torch.mean(_dist, dim=(0, 1))
        return dist

    def SupConLoss(self, text_features, image_features, t_label, i_targets):
            batch_size = text_features.shape[0]
            batch_size_N = image_features.shape[0]
            mask = torch.eq(t_label.unsqueeze(1).expand(batch_size, batch_size_N), \
                            i_targets.unsqueeze(0).expand(batch_size, batch_size_N)).float().cuda()

            logits = torch.div(torch.matmul(text_features, image_features.T), 1)
            # for numerical stability
            logits_max, _ = torch.max(logits, dim=1, keepdim=True)
            logits = logits - logits_max.detach()
            exp_logits = torch.exp(logits)
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
            loss = - mean_log_prob_pos.mean()

            return loss