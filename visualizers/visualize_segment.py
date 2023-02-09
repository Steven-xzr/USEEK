import os
import numpy as np
import h5py
import argparse
import open3d as o3d


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--segment_dataset', type=str, default='./saved_segments/knife.npz')
arg_parser.add_argument('--idx', type=int, default=0)       # 3, 4, 5, 6, 8:four engines
# arg_parser.add_argument('--top', type=int, default=250)
arg_parser.add_argument('--kp_idx', type=int, default=0)
args = arg_parser.parse_args()


"""
=====================
chair_example.pt
kp_idx, semantic
0, back-cushion, right
1, back, right top
    2, leg, right back
3, back, left top
    4, cushion, front left
    5, leg, right front
    6, leg, left front
7, back-cushion, left
8, cushion, front right
9, cushion, front left
=====================
"""

if __name__ == '__main__':
    segment_dataset = np.load(args.segment_dataset)

    pcd = segment_dataset['train_pcds'][args.idx]
    mask = segment_dataset['train_segments'][args.idx]

    pcd_vis = o3d.geometry.PointCloud()
    pcd_vis.points = o3d.utility.Vector3dVector(pcd.reshape(-1, 3))
    pcd_vis.paint_uniform_color([0, 0, 0])

    # pcd_mask = []
    # # for k in range(10):
    # for k in [0, 1, 2, 4]:       # 0: left; 1: nose; 2: right; 3: middle; 4: tail
    #     pcd_mask.append(np.take(pcd, np.where(mask[:, k] == 1), axis=0).squeeze())
    # pcd_mask = np.concatenate(np.asarray(pcd_mask))
    
    mask_0 = o3d.geometry.PointCloud()
    mask_0.points = o3d.utility.Vector3dVector(np.take(pcd, np.where(mask[:, 0] == 1), axis=0).squeeze().reshape(-1, 3))
    mask_0.paint_uniform_color([0.8, 0, 0])

    mask_1 = o3d.geometry.PointCloud()
    mask_1.points = o3d.utility.Vector3dVector(np.take(pcd, np.where(mask[:, 1] == 1), axis=0).squeeze().reshape(-1, 3))
    mask_1.paint_uniform_color([0, 0.8, 0.8])

    mask_2 = o3d.geometry.PointCloud()
    mask_2.points = o3d.utility.Vector3dVector(np.take(pcd, np.where(mask[:, 2] == 1), axis=0).squeeze().reshape(-1, 3))
    mask_2.paint_uniform_color([0.8, 0, 0.8])

    mask_4 = o3d.geometry.PointCloud()
    mask_4.points = o3d.utility.Vector3dVector(np.take(pcd, np.where(mask[:, 4] == 1), axis=0).squeeze().reshape(-1, 3))
    mask_4.paint_uniform_color([0, 0.8, 0])

    all_geo = [pcd_vis, mask_0, mask_1, mask_2, mask_4]
    for geo in all_geo:
        geo.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([0.7, -0.2, -0.8])), np.array([0, 0, 0]))
    # o3d.visualization.draw_geometries([pcd_vis, mask_0, mask_1, mask_2, mask_4])

    o3d.visualization.draw_geometries([pcd_vis])
