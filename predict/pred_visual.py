import os
import re
import sys
import numpy as np
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = os.path.dirname(BASE_DIR)
# sys.path.append(BASE_DIR)
# produce pred results
# folder_path = r'E:\python_code\WangShuYe\dgcnn_new\predict\new'
# # save_folder = 'merge_ky'
# file_list = os.listdir(folder_path)
# txt_file_list = [file_name for file_name in file_list if file_name.endswith('.txt')]
# data_list = []
# for file_name in txt_file_list:
    # 读取点云数据
file_path = 'Area_all_room_21_pred_gt.txt'
# point_cloud = np.loadtxt(file_path)
f = open(file_path, "r+")
# f = open('E:\python_code\WangShuYe\dgcnn_new\predict\Area_48- Cloud.txt', "r+")
elements = file_path.split(('.txt'))
# elments = re.split('[/\\\\]', file_name)
out_path = elements[0]
pred = open((out_path + "_pred.txt"), "w+")
# pred = open('E:\python_code\WangShuYe\dgcnn_new\predict\Area_48_pred- Cloud.txt', "w+")
for line in f.readlines():
    line = line.strip().split( )
    # line = line.strip().split()
    if (line[6] == "0"):
        str = ' 100 100 255'+'\n'
        # str = " 100 100 255" + "\n"
    elif (line[6] == '1'):
        str = ' 100 255 100' + '\n'
    elif (line[6] == '2'):
        str = ' 0 0 0' + '\n'
    else:
        # str = ' 100 100 255'+'\n'
        str = " 100 100 100" + "\n"
    pred.write(line[0] + " " + line[1] + " " + line[2] + str)
f.close()
pred.close()

f = open(file_path, "r+")
# f = open('E:\python_code\WangShuYe\dgcnn_new\predict\Area_48- Cloud.txt', "r+")
gt = open((out_path + "_gt.txt"), "w+")
# gt = open('E:\python_code\WangShuYe\dgcnn_new\predict\Area_48_gt- Cloud.txt', "w+")
for line in f.readlines():
    line = line.strip().split( )
    # line = line.strip().split()
    if (line[7] == "0"):
        str = ' 100 100 255'+'\n'
        # str = " 100 100 255" + "\n"##蓝色窗户
    elif (line[7] == '1'):
        str = ' 100 255 100' + '\n'##黄绿色墙
    elif (line[7] == '2'):
        str = ' 0 0 0' + '\n'##阳台
    else:
        # str = ' 100 100 255'+'\n'
        str = " 100 100 100" + "\n"##门
    gt.write(line[0] + " " + line[1] + " " + line[2] + str)
f.close()
gt.close()
