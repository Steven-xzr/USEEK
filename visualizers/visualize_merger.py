# -*- coding: utf-8 -*-
import os
import random
import argparse
import contextlib
import torch
import torch.optim as optim
from merger.data_flower import all_h5
from merger.merger_net import Net
from merger.composed_chamfer import composed_sqrt_chamfer

import open3d as o3d
import copy

arg_parser = argparse.ArgumentParser(description="Training Skeleton Merger. Valid .h5 files must contain a 'data' array of shape (N, n, 3) and a 'label' array of shape (N, 1).", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
arg_parser.add_argument('-t', '--train-data-dir', type=str, default='./point_cloud/train',
                        help='Directory that contains training .h5 files.')
arg_parser.add_argument('-v', '--val-data-dir', type=str, default='./point_cloud/test',
                        help='Directory that contains validation .h5 files.')
arg_parser.add_argument('-c', '--subclass', type=int, default=0,
                        help='Subclass label ID to train on.')  # 14 is `chair` class. 0 is airplane. 38 is mug. 45 is rifle. 13 is car. 18 is table. 27 is guitar. 30 is knife.
# arg_parser.add_argument('-c', '--subclass', type=str, default='02691156',
#                        help='Subclass label ID to train on.')
arg_parser.add_argument('-m', '--checkpoint', type=str, default='./saved_models/airplane_merger.pt',
                        help='Model checkpoint file path for saving.')
arg_parser.add_argument('-k', '--n-keypoint', type=int, default=4,
                        help='Requested number of keypoints to detect.')
arg_parser.add_argument('-d', '--device', type=str, default='cpu',
                        help='Pytorch device for training.')
arg_parser.add_argument('--idx', type=int, default=20)
args = arg_parser.parse_args()


kp_idx = [0, 2, 3]


def pcd_normalization(b_pcd: torch.Tensor, centralize=True):
    """
    Normalize the point cloud to [-1, 1]
    :return: b_pcd: torch.Tensor of shape (b, n_points, 3)
    """

    if not centralize:
        data = copy.deepcopy(b_pcd)    # [B, N, 3]
        dmin = data.min(dim=1, keepdim=True)[0].min(dim=-1, keepdim=True)[0]
        dmax = data.max(dim=1, keepdim=True)[0].max(dim=-1, keepdim=True)[0]
        data = (data - dmin) / (dmax - dmin)
        return 2.0 * (data - 0.5)
    else:
        data = copy.deepcopy(b_pcd)    # [B, N, 3]
        data -= data.mean(-2, keepdim=True)
        data /= torch.max(torch.norm(data, dim=-1, keepdim=True), dim=-2, keepdim=True)[0]
        return data


if __name__ == '__main__':
    DATASET = args.train_data_dir
    TESTSET = args.val_data_dir
    # x, xl = all_h5(DATASET, True, True, subclasses=(args.subclass,), sample=None)  # n x 2048 x 3
    x_test, xl_test = all_h5(TESTSET, True, True, subclasses=(args.subclass,), sample=None)

    pcd = x_test[args.idx]
    # pcd = pcd_normalization(torch.from_numpy(pcd).unsqueeze(0).to(args.device))
    # pcd = torch.from_numpy(pcd).unsqueeze(0).to(args.device)
    #
    # net = Net(2048, args.n_keypoint, 'PointNet2').to(args.device)
    # net.eval()
    # # net.load_state_dict(torch.load(args.checkpoint))
    # net.load_state_dict(torch.load(args.checkpoint, map_location=torch.device('cpu'))['model_state_dict'])
    #
    # with torch.no_grad():
    #     _, keypoints, _, _, _ = net(pcd)
    #
    # keypoints = keypoints.cpu().numpy().squeeze()
    # pcd = pcd.cpu().numpy().squeeze()

    pcd_vis = o3d.geometry.PointCloud()
    pcd_vis.points = o3d.utility.Vector3dVector(pcd.reshape(-1, 3))
    pcd_vis.paint_uniform_color([0, 0, 0])

    # kp_vis = o3d.geometry.PointCloud()
    # kp_vis.points = o3d.utility.Vector3dVector(keypoints[kp_idx, :].reshape(-1, 3))
    # kp_vis.paint_uniform_color([0, 1, 0])

    # o3d.visualization.draw_geometries([pcd_vis, kp_vis])

    o3d.visualization.draw_geometries([pcd_vis])
