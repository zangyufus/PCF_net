import numpy as np
import glob
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
import indoor3d_util

# # -----------------------------------------------------------------------------
# # CONSTANTS
# # -----------------------------------------------------------------------------

# # .rstrip()方法，删除字符串末尾的指定字符（默认为空格）
# # enumerate()函数，用于将一个可遍历的数据对象（如列表）组合为一个索引序列，同时列出数据
# #和数据下标 eg: seasons = ['Spring', 'Summer', 'Fall', 'Winter']
# #list(enumerate(seasons))----[(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]
# DATA_PATH = os.path.join(ROOT_DIR, 'data', 'Facade_raw_data')
# g_classes = [x.rstrip() for x in open(os.path.join(BASE_DIR, 'meta/class_names.txt'))]
# g_class2label = {cls: i for i,cls in enumerate(g_classes)}
# g_class2color = {'window':       [100,100,255],
#          'wall':                 [0,255,255],
#          'door':                 [200,200,100],
#          'airconditioner':       [255,0,0],
#          'clutter':              [50,50,50]} 
# g_easy_view_labels = [3,4]
# # g_label2color = {g_classes.index(cls): g_class2color[cls] for cls in g_classes}
# print(g_classes)
# print(g_class2label)

# def calculate_sem_IoU(pred_np, seg_np):
#     I_all = np.zeros(CLS_NUM)
#     U_all = np.zeros(CLS_NUM)
#     for sem_idx in range(seg_np.shape[0]):
#         for sem in range(CLS_NUM):
#             I = np.sum(np.logical_and(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
#             U = np.sum(np.logical_or(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
#             I_all[sem] += I
#             U_all[sem] += U
#     return I_all, U_all, I_all / U_all


# train_ious = calculate_sem_IoU(train_pred_seg, train_true_seg)
#         outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f, train iou: %.6f' % (epoch, 
#                                                                                                   train_loss*1.0/count,
#                                                                                                   train_acc,
#                                                                                                   avg_per_class_acc,
#                                                                                                   np.mean(train_ious))

# I_all_test, U_all_test, test_ious = calculate_sem_IoU(test_pred_seg, test_true_seg)
#         outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % (epoch,
#                                                                                               test_loss*1.0/count,
#                                                                                               test_acc,
#                                                                                               avg_per_class_acc,
#       
#                                                                                        np.mean(test_ious))
split = 'test'
test_area = 1
data_root = 'C:\\Users\\Lenovo\\Desktop\\dgcnn\\data\\Facade_as_S3DIS_test_NPY'
num_point = 4096
block_size = 1.0
stride = 1.0
use_all_points = True
rooms = sorted(os.listdir(data_root))
rooms = [room for room in rooms if 'Area_' in room]
if split == 'train':
    rooms_split = [room for room in rooms if not 'Area_{}'.format(test_area) in room]
else:
    rooms_split = [room for room in rooms if 'Area_{}'.format(test_area) in room]
room_points, room_labels = [], []
room_coord_min, room_coord_max = [], []
print(rooms)
print(rooms_split)

room_idxs = []
for index, room_name in enumerate(rooms_split):
    room_path = os.path.join(data_root, room_name)
    room_data = np.load(room_path)
    points, labels = room_data[:, 0:6], room_data[:, 6]
    print(labels)
    coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
    room_coord_min.append(coord_min), room_coord_max.append(coord_max)
    block_points, block_labels = indoor3d_util.room2blocks(points, labels, num_point, block_size=block_size,
                                                stride=stride, random_sample=False, sample_num=None, use_all_points=use_all_points)
    room_idxs.extend([index] * int(block_points.shape[0]))  # extend with number of blocks in a room
    room_points.append(block_points), room_labels.append(block_labels)
room_points = np.concatenate(room_points)
room_labels = np.concatenate(room_labels)

room_idxs = np.array(room_idxs)
print("Totally {} samples in {} set.".format(len(room_idxs), split))