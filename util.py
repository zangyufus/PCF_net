#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: util
@Time: 4/5/19 3:47 PM
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


def cal_loss(pred, gold, weight, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)
    n_class = pred.size(1)
    if smoothing:
        # eps = 0.1
        # one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        # one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        # gt3 = one_hot[:,3]
        # log_prb = F.softmax(pred, dim=1)
        #
        # class3 = log_prb[:,3]
        # mask = class3 < 0.5
        # class3_lt_5 = torch.masked_select(class3, mask)
        # gt3_1 = torch.masked_select(gt3, mask)
        # log_class3_lt_5 = torch.log(class3_lt_5)
        # loss3_1 = (1-class3_lt_5) * gt3_1 * log_class3_lt_5
        #
        # class3_ge_5 = torch.masked_select(class3, ~mask)
        # gt3_2 = torch.masked_select(gt3, ~mask)
        # log_class3_ge_5 = torch.log(class3_ge_5)
        # loss3_2 = gt3_2 * log_class3_ge_5
        # loss3 = loss3_1.sum(dim=0) + loss3_2.sum(dim=0)
        #
        # log_pred = torch.log(log_prb[:,:3])
        # gt = one_hot[:,:3]
        # loss = (log_pred * gt).sum(dim=0)
        # loss = torch.cat((-loss.reshape(-1,1),-loss3.reshape(-1,1)),dim=0) / gold.size(0)
        # weight_loss = weight.view(-1,1) * loss
        # loss = weight_loss.sum(dim=0)


        ###原始loss
        eps = 0.1
        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)
        loss_per_class = -(one_hot * log_prb).sum(dim=0) / gold.size(0)
        weight_loss = weight * loss_per_class
        loss = weight_loss.sum(dim=0)
        # loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='none')
        loss_per_class = torch.zeros(n_class, device=pred.device)
        for c in range(n_class):
            class_mask = gold == c
            class_loss = loss[class_mask].mean()
            loss_per_class[c] = class_loss

    return loss

class WeightedCrossEntropy(nn.Module):
    def __init__(self, class_weights=None):
        super(WeightedCrossEntropy,self).__init__()
        self.class_weights = class_weights

    def forward(self, pred, gold):
        loss = nn.CrossEntropyLoss(weight=self.class_weights)(pred, gold)

        return loss


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, num_classes=3, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """
        super(FocalLoss, self).__init__()
        self.size_average = size_average
        if alpha is None:
            self.alpha = torch.ones(num_classes)
        elif isinstance(alpha, np.ndarray):
            assert len(alpha) == num_classes  # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1  # 如果α为一个常数,则降低第一类的影响,在目标检测中第一类为背景类
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)  # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

        self.gamma = gamma

        print('Focal Loss:')
        print('    Alpha = {}'.format(self.alpha))
        print('    Gamma = {}'.format(self.gamma))

    def forward(self, preds, labels):
            """
            focal_loss损失计算
            :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
            :param labels:  实际类别. size:[B,N] or [B]
            :return:
            """

            # assert preds.dim()==2 and labels.dim()==1
            preds = preds.view(-1, preds.size(-1))
            alpha = self.alpha.to(preds.device)
            preds_logsoft = F.log_softmax(preds, dim=1)  # log_softmax
            preds_softmax = torch.exp(preds_logsoft)  # softmax

            preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))  # 这部分实现nll_loss ( crossempty = log_softmax + nll )
            preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
            alpha = alpha.gather(0, labels.view(-1))
            loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),
                              preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

            loss = torch.mul(alpha, loss.t())
            if self.size_average:
                loss = loss.mean()
            else:
                loss = loss.sum()
            return loss


class MultiFocalLoss(nn.Module):
    """
    Focal_Loss= -1*alpha*((1-pt)**gamma)*log(pt)
    Args:
        num_class: number of classes
        alpha: class balance factor shape=[num_class, ]
        gamma: hyper-parameter
        reduction: reduction type
    """

    def __init__(self, num_class, alpha=None, gamma=2, reduction='mean'):
        super(MultiFocalLoss, self).__init__()
        self.num_class = num_class
        self.gamma = gamma
        self.reduction = reduction
        self.smooth = 1e-4
        self.alpha = alpha
        if alpha is None:
            self.alpha = torch.ones(num_class, ) - 0.5
        elif isinstance(alpha, (int, float)):
            self.alpha = torch.as_tensor([alpha] * num_class)
        elif isinstance(alpha, (list, np.ndarray)):
            self.alpha = torch.as_tensor(alpha)
        if self.alpha.shape[0] != num_class:
            raise RuntimeError('the length not equal to number of class')

    def forward(self, logit, target):
        # assert isinstance(self.alpha,torch.Tensor)\
        alpha = self.alpha.to(logit.device)
        prob = F.softmax(logit, dim=1)

        if prob.dim() > 2:
            # used for 3d-conv:  N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            N, C = logit.shape[:2]
            prob = prob.view(N, C, -1)
            prob = prob.transpose(1, 2).contiguous()  # [N,C,d1*d2..] -> [N,d1*d2..,C]
            prob = prob.view(-1, prob.size(-1))  # [N,d1*d2..,C]-> [N*d1*d2..,C]

        ori_shp = target.shape
        target = target.view(-1, 1)

        prob = prob.gather(1, target).view(-1) + self.smooth  # avoid nan
        logpt = torch.log(prob)
        # alpha_class = alpha.gather(0, target.squeeze(-1))
        alpha_weight = alpha[target.squeeze().long()]
        loss = -alpha_weight * torch.pow(torch.sub(1.0, prob), self.gamma) * logpt

        if self.reduction == 'mean':
            loss = (loss*10).mean()
        elif self.reduction == 'none':
            loss = loss.view(ori_shp)
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss