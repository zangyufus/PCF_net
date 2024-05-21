from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from data import S3DISDataset_eval
from model import PCF_net
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream, FocalLoss
import sklearn.metrics as metrics
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import provider
import time
from sklearn.metrics import precision_recall_fscore_support
from LovaszSoftmax import lovasz_softmax_flat



CLS_NUM = 4

def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp test.py checkpoints'+'/'+args.exp_name+'/'+'test.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')

def calculate_sem_IoU(pred_np, seg_np ,num_classes):
    I_all = np.zeros(num_classes)
    U_all = np.zeros(num_classes)
    # original code: for sem_idx in range(seg_np.shape[0])
    for sem_idx in range(len(seg_np)):
        for sem in range(num_classes):
            I = np.sum(np.logical_and(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            U = np.sum(np.logical_or(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            I_all[sem] += I
            U_all[sem] += U
    return I_all / U_all

def test(args, io):
    DUMP_DIR = args.test_visu_dir
    all_true_cls = []
    all_pred_cls = []
    all_true_seg = []
    all_pred_seg = []
    dataset = S3DISDataset_eval(split='test', data_root=args.data_dir, num_point=args.num_points, test_area=args.test_area,
                                block_size=args.block_size, num_class=args.num_classes,sample_rate=1.5)
    test_loader = DataLoader(dataset,num_workers=4, batch_size=args.test_batch_size, shuffle=False, drop_last=True)

    room_idx = np.array(dataset.room_idxs)
    num_blocks = len(room_idx)

    fout_data_label = []
    for room_id in np.unique(room_idx):
        out_data_label_filename = 'Area_%s_room_%d_pred_gt.txt' % (args.test_area, room_id)
        out_data_label_filename = os.path.join(DUMP_DIR, out_data_label_filename)
        fout_data_label.append(open(out_data_label_filename, 'w+'))

    device = torch.device("cuda" if args.cuda else "cpu")

    io.cprint('Start overall evaluation...')

    # Try to load models
    if args.model == 'pcf_net':
        model = PCF_net(args).to(device)
    else:
        raise Exception("Not implemented")

    model = nn.DataParallel(model)
    checkpoint = torch.load(os.path.join(args.model_root, 'best_model.t7'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    total_params += sum(p.numel() for p in model.buffers())
    print(f'{total_params:,} total parameters.')
    print(f'{total_params / (1024 * 1024):.2f}M total parameters.')
    io.cprint('model_all.t7 restored.')

    test_acc = 0.0
    count = 0.0
    test_true_cls = []
    test_pred_cls = []
    test_true_seg = []
    test_pred_seg = []

    io.cprint('Start testing ...')
    num_batch = 0
    for data, seg in tqdm(test_loader):
        data, seg = data.to(device), seg.to(device)
        data = data.permute(0, 2, 1).float()
        batch_size = data.size()[0]
        seg_pred,features = model(data)
        seg_pred = seg_pred.permute(0, 2, 1).contiguous()
        pred = seg_pred.max(dim=2)[1]
        seg_np = seg.cpu().numpy()
        pred_np = pred.detach().cpu().numpy()
        test_true_cls.append(seg_np.reshape(-1))
        test_pred_cls.append(pred_np.reshape(-1))
        test_true_seg.append(seg_np)
        test_pred_seg.append(pred_np)
        print(np.array(seg_np).shape)

        # write prediction results
        for batch_id in range(batch_size):

            feature = features.detach().cpu().numpy()[batch_id, :, :]
            heatmap = np.sum(feature, axis=0)
            heatmap /= np.max(heatmap)
            pts = data[batch_id, :, :]

            pts = pts.permute(1, 0).float()
            l = seg[batch_id, :]
            pts[:, 3:6] *= 65535.0
            pred_ = pred[batch_id, :]
            logits = seg_pred[batch_id, :, :]
            # compute room_id
            room_id = room_idx[num_batch + batch_id]
            for i in range(pts.shape[0]):
                fout_data_label[room_id].write('%f %f %f %d %d %d %d %d %f\n' % (
                    pts[i, 6] * dataset.room_coord_max[room_id][0],
                    pts[i, 7] * dataset.room_coord_max[room_id][1],
                    pts[i, 8] * dataset.room_coord_max[room_id][2],
                    pts[i, 3], pts[i, 4], pts[i, 5], pred_[i], l[i], heatmap[i]))  # xyzRGB pred gt
        num_batch += batch_size
        torch.cuda.empty_cache()

    for room_id in np.unique(room_idx):
        fout_data_label[room_id].close()

    test_true_cls = np.concatenate(test_true_cls)
    test_pred_cls = np.concatenate(test_pred_cls)
    test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
    test_true_seg = np.concatenate(test_true_seg, axis=0)
    test_pred_seg = np.concatenate(test_pred_seg, axis=0)
    test_ious = calculate_sem_IoU(test_pred_cls, test_true_cls ,args.num_classes)
    outstr = 'Test :: test area: %s, test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % (args.test_area,
                                                                                            test_acc,
                                                                                            avg_per_class_acc,
                                                                                            np.mean(test_ious))
    io.cprint(outstr)


    precision, recall, f1, support = precision_recall_fscore_support(test_true_cls, test_pred_cls,average=None)
    io.cprint('precision:')
    io.cprint(str(precision))
    io.cprint('recall:')
    io.cprint(str(recall))
    io.cprint('F1:')
    io.cprint(str(f1))
    io.cprint('support:')
    io.cprint(str(support))


    # calculate confusion matrix
    conf_mat = metrics.confusion_matrix(test_true_cls, test_pred_cls)
    io.cprint('Confusion matrix:')
    io.cprint(str(conf_mat))
    ##calculate iou
    iou_list = []
    for class_id in range(conf_mat.shape[0]):
        TP = conf_mat[class_id, class_id]
        FP = np.sum(conf_mat[:, class_id]) - TP
        FN = np.sum(conf_mat[class_id, :]) - TP
        IoU = TP / (TP + FP + FN)
        iou_list.append(IoU)
    for class_id, iou in enumerate(iou_list):
        print(f'Class {class_id} IoU: {iou:.4f}')
    print(f'mIoU {np.mean(iou_list)}')

    all_true_cls.append(test_true_cls)
    all_pred_cls.append(test_pred_cls)
    all_true_seg.append(test_true_seg)
    all_pred_seg.append(test_pred_seg)

    if args.test_area == 'all':
        all_true_cls = np.concatenate(all_true_cls)
        all_pred_cls = np.concatenate(all_pred_cls)
        all_acc = metrics.accuracy_score(all_true_cls, all_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(all_true_cls, all_pred_cls)
        all_true_seg = np.concatenate(all_true_seg, axis=0)
        all_pred_seg = np.concatenate(all_pred_seg, axis=0)
        all_ious = calculate_sem_IoU(all_pred_seg, all_true_seg, args.num_classes)
        outstr = 'Overall Test :: test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % (all_acc,
                                                                                         avg_per_class_acc,
                                                                                         np.mean(all_ious))
        io.cprint(outstr)


if __name__ == "__main__":

    # # Testing settings
    parser = argparse.ArgumentParser(description='Point Cloud Part Segmentation')
    # data_dir
    parser.add_argument('--data_dir', type=str, default='data/val',
                        help='Directory of data')
    parser.add_argument('--tb_dir', type=str, default='log_tensorboard',
                        help='Directory of tensorboard logs')
    # exp_name
    parser.add_argument('--exp_name', type=str, default='semseg_eval_1', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='pcf_net', metavar='N',
                        choices=['pcf_net'],
                        help='Model to use, [dgcnn]')
    parser.add_argument('--dataset', type=str, default='S3DIS', metavar='N',
                        choices=['S3DIS'])
    parser.add_argument('--block_size', type=float, default=4.0,
                        help='size of one block')
    parser.add_argument('--num_classes', type=int, default=4,
                        help='number of classes in the dataset')
    # test_area
    parser.add_argument('--test_area', type=str, default='all', metavar='N',)
    # batch_size
    parser.add_argument('--batch_size', type=int, default=4, metavar='batch_size',
                        help='Size of batch)')
    # test_batch_size
    parser.add_argument('--test_batch_size', type=int, default=4, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                        choices=['cos', 'step'],
                        help='Scheduler to use, [cos, step]')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # eval
    parser.add_argument('--eval', type=bool,  default=True,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=4096,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    # model_root
    parser.add_argument('--model_root', type=str, default='checkpoints/exp/models/', metavar='N',
                        help='Pretrained model root')
    parser.add_argument('--test_visu_dir', default='predict',
                        help='Directory of test visualization files.')


    args = parser.parse_args()

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if args.eval:
        test(args, io)
