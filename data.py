#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: data.py
@Time: 2018/10/13 6:21 PM

Modified by 
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@Time: 2020/2/27 9:32 PM
"""


import os
import sys
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
PRE_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prepare_data')
sys.path.append(PRE_DATA_DIR)
from prepare_data import indoor3d_util


def download_modelnet40():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def download_shapenetpart():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data')):
        www = 'https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data')))
        os.system('rm %s' % (zipfile))

# 1
def download_S3DIS():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'indoor3d_sem_seg_hdf5_data')):
        www = 'https://shapenet.cs.stanford.edu/media/indoor3d_sem_seg_hdf5_data.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))
    if not os.path.exists(os.path.join(DATA_DIR, 'Stanford3dDataset_v1.2_Aligned_Version')):
        if not os.path.exists(os.path.join(DATA_DIR, 'Stanford3dDataset_v1.2_Aligned_Version.zip')):
            print('Please download Stanford3dDataset_v1.2_Aligned_Version.zip \
                from https://goo.gl/forms/4SoGp4KtH1jfRqEj2 and place it under data/')
            sys.exit(0)
        else:
            zippath = os.path.join(DATA_DIR, 'Stanford3dDataset_v1.2_Aligned_Version.zip')
            os.system('unzip %s' % (zippath))
            os.system('rm %s' % (zippath))


def load_data_cls(partition):
    download_modelnet40()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40*hdf5_2048', '*%s*.h5'%partition)):
        f = h5py.File(h5_name, 'r+')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def load_data_partseg(partition):
    download_shapenetpart()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    all_seg = []
    if partition == 'trainval':
        file = glob.glob(os.path.join(DATA_DIR, 'shapenet*hdf5*', '*train*.h5')) \
               + glob.glob(os.path.join(DATA_DIR, 'shapenet*hdf5*', '*val*.h5'))
    else:
        file = glob.glob(os.path.join(DATA_DIR, 'shapenet*hdf5*', '*%s*.h5'%partition))
    for h5_name in file:
        f = h5py.File(h5_name, 'r+')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        seg = f['pid'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
        all_seg.append(seg)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    all_seg = np.concatenate(all_seg, axis=0)
    return all_data, all_label, all_seg

# 2
def prepare_training_data_semseg():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    # if not os.path.exists(os.path.join(DATA_DIR, 'stanford_indoor3d')):
    if not os.path.exists(os.path.join(DATA_DIR, 'tud_facade')):
        os.system('python prepare_data/collect_indoor3d_data.py')
    if not os.path.exists(os.path.join(DATA_DIR, 'facade_sem_seg_hdf5_data_train')):
        os.system('python prepare_data/gen_indoor3d_h5_train.py')

# 3
def prepare_test_data_semseg():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    # if not os.path.exists(os.path.join(DATA_DIR, 'stanford_indoor3d')):
    if not os.path.exists(os.path.join(DATA_DIR, 'tud_facade')):
        os.system('python prepare_data/collect_indoor3d_data.py')
    # if not os.path.exists(os.path.join(DATA_DIR, 'indoor3d_sem_seg_hdf5_data_test')):
    if not os.path.exists(os.path.join(DATA_DIR, 'facade_sem_seg_hdf5_data_test')):
        os.system('python prepare_data/gen_indoor3d_h5_test.py')

# 4
def load_data_semseg(partition, test_area):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    # download_S3DIS()
    if not os.path.exists(os.path.join(DATA_DIR, 'facade_sem_seg_hdf5_data_train')):
        prepare_training_data_semseg()
    if not os.path.exists(os.path.join(DATA_DIR, 'facade_sem_seg_hdf5_data_test')):
        prepare_test_data_semseg()
    prepare_training_data_semseg()
    prepare_test_data_semseg()
    if partition == 'train':
        # data_dir = os.path.join(DATA_DIR, 'indoor3d_sem_seg_hdf5_data_train')
        data_dir = os.path.join(DATA_DIR, 'facade_sem_seg_hdf5_data_train')
    else:
        # data_dir = os.path.join(DATA_DIR, 'indoor3d_sem_seg_hdf5_data_test')
        data_dir = os.path.join(DATA_DIR, 'facade_sem_seg_hdf5_data_test')
    with open(os.path.join(data_dir, "all_files.txt")) as f:
        all_files = [line.rstrip() for line in f]
    with open(os.path.join(data_dir, "room_filelist.txt")) as f:
        room_filelist = [line.rstrip() for line in f]
    data_batchlist, label_batchlist = [], []
    for f in all_files:
        file = h5py.File(os.path.join(DATA_DIR, f), 'r+')
        data = file["data"][:]
        label = file["label"][:]
        data_batchlist.append(data)
        label_batchlist.append(label)
    data_batches = np.concatenate(data_batchlist, 0)
    seg_batches = np.concatenate(label_batchlist, 0)
    test_area_name = "Area_" + test_area
    train_idxs, test_idxs = [], []
    for i, room_name in enumerate(room_filelist):
        if test_area_name in room_name:
            test_idxs.append(i)
        else:
            train_idxs.append(i)
    if partition == 'train':
        all_data = data_batches[train_idxs, ...]
        all_seg = seg_batches[train_idxs, ...]
    else:
        all_data = data_batches[test_idxs, ...]
        all_seg = seg_batches[test_idxs, ...]
    return all_data, all_seg


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


def rotate_pointcloud(pointcloud):
    theta = np.pi*2 * np.random.uniform()
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    pointcloud[:,[0,2]] = pointcloud[:,[0,2]].dot(rotation_matrix) # random rotation (x,z)
    return pointcloud


class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_data_cls(partition)
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


class ShapeNetPart(Dataset):
    def __init__(self, num_points, partition='train', class_choice=None):
        self.data, self.label, self.seg = load_data_partseg(partition)
        self.cat2id = {'airplane': 0, 'bag': 1, 'cap': 2, 'car': 3, 'chair': 4, 
                       'earphone': 5, 'guitar': 6, 'knife': 7, 'lamp': 8, 'laptop': 9, 
                       'motor': 10, 'mug': 11, 'pistol': 12, 'rocket': 13, 'skateboard': 14, 'table': 15}
        self.seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
        self.index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]
        self.num_points = num_points
        self.partition = partition        
        self.class_choice = class_choice

        if self.class_choice != None:
            id_choice = self.cat2id[self.class_choice]
            indices = (self.label == id_choice).squeeze()
            self.data = self.data[indices]
            self.label = self.label[indices]
            self.seg = self.seg[indices]
            self.seg_num_all = self.seg_num[id_choice]
            self.seg_start_index = self.index_start[id_choice]
        else:
            self.seg_num_all = 50
            self.seg_start_index = 0

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        seg = self.seg[item][:self.num_points]
        if self.partition == 'trainval':
            # pointcloud = translate_pointcloud(pointcloud)
            indices = list(range(pointcloud.shape[0]))
            np.random.shuffle(indices)
            pointcloud = pointcloud[indices]
            seg = seg[indices]
        return pointcloud, label, seg

    def __len__(self):
        return self.data.shape[0]

# 5
class S3DIS(Dataset):
    def __init__(self, num_points=4096, partition='train', test_area='1'):
        self.data, self.seg = load_data_semseg(partition, test_area)
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        seg = self.seg[item][:self.num_points]
        if self.partition == 'train':
            indices = list(range(pointcloud.shape[0]))
            np.random.shuffle(indices)
            pointcloud = pointcloud[indices]
            seg = seg[indices]
        seg = torch.LongTensor(seg)
        return pointcloud, seg

    def __len__(self):
        return self.data.shape[0]


def farthest_point_sampling_numpy(points, k):
    """
    使用最远点采样算法选择k个点
    :param points: 数据点的数组，每一行代表一个数据点
    :param k: 要选择的点的数量
    :return: 选择的k个点的索引
    """
    points = points[:, :3]
    n = len(points)
    selected_indices = [0]  # 从第一个点开始选择
    distances = np.linalg.norm(points - points[0], axis=1)  # 计算到第一个点的距离

    for _ in range(1, k):
        # 选择下一个距离最远的点
        farthest_index = np.argmax(distances)
        selected_indices.append(farthest_index)

        # 更新距离数组，选择的点到所有点的距离
        new_distances = np.linalg.norm(points - points[farthest_index], axis=1)
        distances = np.minimum(distances, new_distances)

    return selected_indices


def farthest_point_sampling(point_cloud, num_points):
    point_cloud = point_cloud[:,:3]
    device = torch.device("cuda:0")
    point_cloud = torch.Tensor(point_cloud).to(device)
    point_cloud = point_cloud.unsqueeze(0)
    B, N, C = point_cloud.shape
    centroids = torch.zeros(B, num_points, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    # while len(sampled_indices) < num_points:
    for i in range(num_points):
        centroids[:, i] = farthest
        centroid = point_cloud[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((point_cloud - centroid) ** 2, -1)
        mask = dist < distance
        distance = distance.type(torch.cuda.FloatTensor)
        dist = dist.type(torch.cuda.FloatTensor)

        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    centroids = centroids.squeeze().cpu().detach().numpy()
    return centroids

class S3DISDataset_eval(Dataset):  # load data block by block, without using h5 files
    def __init__(self, split='train', data_root='trainval_fullarea', num_point=4096, test_area='5', block_size=1.0, stride=1.0, num_class=20, use_all_points=False, num_thre = 1024):
        super().__init__()
        self.num_point = num_point
        self.block_size = block_size
        self.use_all_points = use_all_points
        self.stride = stride
        self.num_thre = num_thre
        rooms = sorted(os.listdir(data_root))
        rooms = [room for room in rooms if 'Area_' in room]
        if split == 'train':
            rooms_split = [room for room in rooms if not 'Area_{}'.format(test_area) in room]
        else:
            if test_area == 'all':
              rooms_split = [room for room in rooms if not 'Area_{}'.format(test_area) in room]
            else:
              rooms_split = [room for room in rooms if 'Area_{}'.format(test_area) in room]
        self.room_points, self.room_labels = [], []
        self.room_coord_min, self.room_coord_max = [], []
        # print(rooms)
        # print(rooms_split)

        room_idxs = []
        for index, room_name in enumerate(rooms_split):
            room_path = os.path.join(data_root, room_name)
            room_data = np.load(room_path)
            points, labels = room_data[:, 0:6], room_data[:, 6]
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)
            block_points, block_labels = indoor3d_util.room2blocks(points, labels, self.num_point, block_size=self.block_size,
                                                       stride=self.stride, random_sample=False, sample_num=None, use_all_points=self.use_all_points)
            room_idxs.extend([index] * int(block_points.shape[0]))  # extend with number of blocks in a room
            self.room_points.append(block_points), self.room_labels.append(block_labels)
        self.room_points = np.concatenate(self.room_points)
        self.room_labels = np.concatenate(self.room_labels)

        self.room_idxs = np.array(room_idxs)
        print("Totally {} samples in {} set.".format(len(self.room_idxs), split))

    def __getitem__(self, idx):  # get items in one block
        room_idx = self.room_idxs[idx]
        selected_points = self.room_points[idx]   # num_point * 6
        current_labels = self.room_labels[idx]   # num_point
        center = np.mean(selected_points, axis=0)
        N_points = selected_points.shape[0]

        # add normalized xyz
        current_points = np.zeros((N_points, 9))  # num_point * 9
        current_points[:, 6] = selected_points[:, 0] / self.room_coord_max[room_idx][0]
        current_points[:, 7] = selected_points[:, 1] / self.room_coord_max[room_idx][1]
        current_points[:, 8] = selected_points[:, 2] / self.room_coord_max[room_idx][2]
        selected_points[:, 0] = selected_points[:, 0] - center[0]
        selected_points[:, 1] = selected_points[:, 1] - center[1]
        selected_points[:, 3:6] /= 255.0
        current_points[:, 0:6] = selected_points

        return current_points, current_labels

    def __len__(self):
        return len(self.room_idxs)

if __name__ == '__main__':
    train = ModelNet40(1024)
    test = ModelNet40(1024, 'test')
    data, label = train[0]
    print(data.shape)
    print(label.shape)

    trainval = ShapeNetPart(2048, 'trainval')
    test = ShapeNetPart(2048, 'test')
    data, label, seg = trainval[0]
    print(data.shape)
    print(label.shape)
    print(seg.shape)

    train = S3DISDataset(split='train', data_root='data/rgb_train', num_point=4096, test_area='all',
                 block_size=2,sample_rate=5, num_class=4)

    data, seg = train[0]
    print(data.shape)
    print(seg.shape)
