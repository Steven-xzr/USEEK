import os
import numpy as np
import h5py
import argparse
import open3d as o3d


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--prediction_dir', type=str, default='./prediction/knife_merger.npz')
arg_parser.add_argument('--idx', type=int, default=0)       # 3, 4, 5, 6, 8:four engines
# arg_parser.add_argument('--top', type=int, default=250)
arg_parser.add_argument('--kp_idx', type=int, default=0)
args = arg_parser.parse_args()


if __name__ == '__main__':
    pred_dataset = np.load(args.prediction_dir)

    pcd = pred_dataset['pcd'][args.idx]
    gt = pred_dataset['gt'][args.idx]
    pred = pred_dataset['pred'][args.idx][args.kp_idx]
    # pred = pred_dataset['pred'][args.idx]

    pcd_vis = o3d.geometry.PointCloud()
    pcd_vis.points = o3d.utility.Vector3dVector(pcd.reshape(-1, 3))
    pcd_vis.paint_uniform_color([0, 0, 0])

    gt_vis = o3d.geometry.PointCloud()
    gt_vis.points = o3d.utility.Vector3dVector(gt.reshape(-1, 3))
    gt_vis.paint_uniform_color([0, 1, 0])

    pred_vis = o3d.geometry.PointCloud()
    pred_vis.points = o3d.utility.Vector3dVector(pred.reshape(-1, 3))
    pred_vis.paint_uniform_color([1, 0, 0])

    o3d.visualization.draw_geometries([pcd_vis, gt_vis])
