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
