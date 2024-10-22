import argparse
import os
import parser
import sys
import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class CAA_Module(nn.Module):
    """ Channel-wise Affinity Attention module"""

    def __init__(self, in_dim, in_pts):
        super(CAA_Module, self).__init__()

        self.bn1 = nn.BatchNorm1d(in_pts // 8)
        self.bn2 = nn.BatchNorm1d(in_pts // 8)
        self.bn3 = nn.BatchNorm1d(in_dim)

        self.query_conv = nn.Sequential(
            nn.Conv1d(in_channels=in_pts, out_channels=in_pts // 8, kernel_size=1, bias=False),
            self.bn1,
            nn.ReLU())
        self.key_conv = nn.Sequential(
            nn.Conv1d(in_channels=in_pts, out_channels=in_pts // 8, kernel_size=1, bias=False),
            self.bn2,
            nn.ReLU())
        self.value_conv = nn.Sequential(nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1, bias=False),
                                        self.bn3,
                                        nn.ReLU())

        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X N )
            returns :
                out : output feature maps( B X C X N )
        """

        # Compact Channel-wise Comparator block
        x_hat = x.permute(0, 2, 1)
        proj_query = self.query_conv(x_hat)
        proj_key = self.key_conv(x_hat).permute(0, 2, 1)
        similarity_mat = torch.bmm(proj_key, proj_query)

        # Channel Affinity Estimator block
        affinity_mat = torch.max(similarity_mat, -1, keepdim=True)[0].expand_as(similarity_mat) - similarity_mat
        affinity_mat = self.softmax(affinity_mat)

        proj_value = self.value_conv(x)
        out = torch.bmm(affinity_mat, proj_value)
        # residual connection with a learnable weight
        out = self.alpha * out + x
        return out

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)   # (batch_size, num_points, k)
        else:
            idx = knn(x[:, :3], k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature      # (batch_size, 2*num_dims, num_points, k)

class Transform_Net(nn.Module):
    def __init__(self, args):
        super(Transform_Net, self).__init__()
        self.args = args
        self.k = 3

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(18, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn4 = nn.BatchNorm1d(256)

        self.transform = nn.Linear(256, 9*9)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(9, 9))

    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = self.conv3(x)                       # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 1024, num_points) -> (batch_size, 1024)

        x = F.leaky_relu(self.bn3(self.linear1(x)), negative_slope=0.2)     # (batch_size, 1024) -> (batch_size, 512)
        x = F.leaky_relu(self.bn4(self.linear2(x)), negative_slope=0.2)     # (batch_size, 512) -> (batch_size, 256)

        x = self.transform(x)                   # (batch_size, 256) -> (batch_size, 3*3)
        x = x.view(batch_size, 9, 9)            # (batch_size, 3*3) -> (batch_size, 3, 3)

        return x

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        self.k = args.k
        self.outdim = args.num_classes
        self.transform_net = Transform_Net(args)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(args.emb_dims)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(128)

        # self.conv1 = nn.Sequential(nn.Conv2d(14,  64, kernel_size=1, bias=False),
        #                            self.bn1,
        #                            nn.LeakyReLU(negative_slope=0.2))
        self.conv1 = nn.Sequential(nn.Conv2d(18, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(1216, 512, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv9 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv10 = nn.Conv1d(128, self.outdim, kernel_size=1, bias=False)


    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)

        x = get_graph_feature(x, k=self.k, dim9=True)  # (batch_size, 9, num_points) -> (batch_size, 9*2, num_points, k)
        x = self.conv1(x)                              # (batch_size, 9*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                              # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]           # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)   

        x = get_graph_feature(x1, k=self.k)            # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                              # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                              # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]           # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)    

        x = get_graph_feature(x2, k=self.k)            # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                              # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv(x)                               # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]           # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)    

        x = torch.cat((x1, x2, x3), dim=1)             # (batch_size, 64*3, num_points)

        x = self.conv6(x)                              # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]             # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)    

        x = x.repeat(1, 1, num_points)                 # (batch_size, 1024, num_points)
        x = torch.cat((x, x1, x2, x3), dim=1)          # (batch_size, 1024+64*3, num_points)

        x = self.conv7(x)                              # (batch_size, 1024+64*3, num_points) -> (batch_size, 512, num_points)
        x = self.conv8(x)                              # (batch_size, 512, num_points) -> (batch_size, 256, num_points)    
        x = self.dp1(x)
        x1 = self.conv9(x)                               # (batch_size, 256, num_points) -> (batch_size, 128, num_points)
        x = self.conv10(x1)                              # (batch_size, 256, num_points) -> (batch_size, arg.nums, num_points)

        return x,x1


class PCF_net(nn.Module):
    def __init__(self, args):
        super(PCF_net, self).__init__()
        self.args = args
        self.k = args.k
        self.atten = CAA_Module(128*3, args.num_points)
        self.outdim = args.num_classes
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(args.emb_dims)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(128)

        self.conv1 = nn.Sequential(nn.Conv2d(12, 32, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv1d = nn.Sequential(nn.Conv1d(3, 32, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(32),
                                   nn.ReLU())
        self.conv2d = nn.Sequential(nn.Conv1d(32, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.ReLU())

        self.conv3 = nn.Sequential(nn.Conv2d((64+6)*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3d = nn.Sequential(nn.Conv1d(64+3, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.ReLU())
        self.conv4d = nn.Sequential(nn.Conv1d(64, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.ReLU())

        self.conv5 = nn.Sequential(nn.Conv2d((64+64+6)*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5d = nn.Sequential(nn.Conv1d(64+64+3, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.ReLU())
        self.conv6d = nn.Sequential(nn.Conv1d(64, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.ReLU())

        self.conv6 = nn.Sequential(nn.Conv1d(128*3, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(1408, 512, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv9 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv10 = nn.Conv1d(128, self.outdim, kernel_size=1, bias=False)
    def segment(self, x):##
        x = self.conv7(x)  # (batch_size, 1024+64*3, num_points) -> (batch_size, 512, num_points)
        x = self.conv8(x)  # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        # x = self.conv9(x)  # (batch_size, 256, num_points) -> (batch_size, arg.nums, num_points)
        x = self.conv10(x)  # (batch_size, 256, num_points) -> (batch_size, arg.nums, num_points)
        return x

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)
        RGB = x[:,3:6,:] ##B,3,N
        COR = torch.cat([x[:,:3,:], x[:, 6:9, :]], dim=1) ##B,6,N

        cor = get_graph_feature(COR, k=self.k, dim9=True)  # (batch_size, 9, num_points) -> (batch_size, 9*2, num_points, k)
        cor = self.conv1(cor)                              # (batch_size, 9*2, num_points, k) -> (batch_size, 64, num_points, k)
        cor = self.conv2(cor)                              # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        cor1 = cor.max(dim=-1, keepdim=False)[0]           # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        rgb = self.conv1d(RGB)                             #B,3,N - B,32,N
        rgb1 = self.conv2d(rgb)                             #B,32,N - B,64,N
        inf1 = torch.cat((cor1, rgb1), dim=1)                #B,128,N

        cor = torch.cat((COR, cor1), dim=1)               #B,64+6,N
        cor = get_graph_feature(cor, k=self.k)            # (batch_size, 64+6, num_points) -> (batch_size, 64+6*2, num_points, k)
        cor = self.conv3(cor)                              # (batch_size, 140, num_points, k) -> (batch_size, 64, num_points, k)
        cor = self.conv4(cor)                              # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        cor2 = cor.max(dim=-1, keepdim=False)[0]           # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        rgb = torch.cat((RGB,rgb1), dim=1)                  #B,64+3,N
        rgb = self.conv3d(rgb)                             #B,64,N
        rgb2 = self.conv4d(rgb)                             #B,64,N
        inf2 = torch.cat((rgb2,cor2), dim=1)                #B,128,N

        cor = torch.cat((COR, cor2, cor1), dim=1)               # B,64+64+6,N
        cor = get_graph_feature(cor, k=self.k)            # (batch_size, 64+64+6, num_points) -> (batch_size, 64+64+6*2, num_points, k)
        cor = self.conv5(cor)                              # (batch_size, 64+64+6*2, num_points, k) -> (batch_size, 64, num_points, k)
        cor = self.conv(cor)                               # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        cor3 = cor.max(dim=-1, keepdim=False)[0]           # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        rgb = torch.cat((RGB,rgb2,rgb1), dim=1)            #B,64+64+3,N
        rgb = self.conv5d(rgb)                             #B,64,N
        rgb3 = self.conv6d(rgb)                             #B,64,N
        inf3 = torch.cat((rgb3,cor3), dim=1)                #B,128,N

        x = torch.cat((inf1, inf2, inf3), dim=1)       # (batch_size, 128*3, num_points)添加attention
        # x = self.atten(x)
        x = self.conv6(x)                              # (batch_size, 128*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]              # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        x = x.repeat(1, 1, num_points)                 # (batch_size, 1024, num_points)
        x = torch.cat((x, inf1, inf2, inf3), dim=1)    # (batch_size, 1024+64*3, num_points)
        x = self.atten(x)
        
        x = self.segment(x)

        return x
