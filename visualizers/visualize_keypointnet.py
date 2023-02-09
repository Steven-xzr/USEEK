import os
import numpy as np
import json
import argparse
import open3d as o3d

from utils import naive_read_pcd
import torch
import merger.merger_net as merger_net


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--annotation', type=str, default='./keypointnet/annotations/knife.json')
arg_parser.add_argument('--pcd-path', type=str, default='./keypointnet/pcds',
                        help='Point cloud file folder path from KeypointNet dataset.')
arg_parser.add_argument('--obj_idx', type=int, default=1)   # 80
arg_parser.add_argument('--kp_idx', type=int, default=2)
arg_parser.add_argument('--pred_idx', type=int, default=1)
arg_parser.add_argument('-m', '--checkpoint-path', '--model-path', type=str, default='saved_models/knife_merger_k4.pt',
                        help='Model checkpoint file path to load.')
arg_parser.add_argument('-k', '--n-keypoint', type=int, default=4,
                        help='Requested number of keypoints to detect.')
arg_parser.add_argument('--max-points', type=int, default=2048,
                        help='Indicates maximum points in each input point cloud.')
arg_parser.add_argument('--encoder-type', type=str, default='PointNet2', choices=['PointNet2', 'DGCNN', 'EQCNN', 'SPRIN'])
args = arg_parser.parse_args()


"""
=====================
airplane
semantic_id, semantic
0, nose
2, tail
3, left_wing
5, right_wing
6, body (front link of the right wing)
=====================
chair
semantic_id, semantic
0, back, left top
1, back, right top
2, back-cushion, left
3, back-cushion, right
4, cushion, front left
5, cushion, front right
6, left arm, front
7, left arm, back
8, right arm, front
9, right arm, back
10~14, base support, in a circle
15, cushion-base, center
16, cushion-base, bottom
17~20, four legs
    19, left back
20, left front
=====================
guitar
semantic_id, semantic
0, top
1, neck-body
    2, body, top left
    3, body, top right
4, body, bottom left
5, body, bottom right
    6, body, bottom center
=====================
=====================
knife
semantic_id, semantic
0, top
1, handle
2, neck (middle)
=====================
"""


if __name__ == '__main__':
    net = merger_net.Net(args.max_points, args.n_keypoint, args.encoder_type).to('cpu')
    net.load_state_dict(torch.load(args.checkpoint_path, map_location=torch.device('cpu'))['model_state_dict'])
    net.eval()

    kpn_ds = json.load(open(args.annotation))
    entry = kpn_ds[args.obj_idx]
    cid = entry['class_id']
    mid = entry['model_id']
    pcd = naive_read_pcd(r'{}/{}/{}.pcd'.format(args.pcd_path, cid, mid))

    kp = np.array(entry['keypoints'][args.kp_idx]['xyz'])
    print('semantic_id=', entry['keypoints'][args.kp_idx]['semantic_id'])

    # normalized
    pcmax = pcd.max()
    pcmin = pcd.min()
    pcd_n = (pcd - pcmin) / (pcmax - pcmin)
    pcd_n = 2.0 * (pcd_n - 0.5)

    kp_n = (kp - pcmin) / (pcmax - pcmin)
    kp_n = 2.0 * (kp_n - 0.5)

    with torch.no_grad():
        _, pred_n, _, _, _ = net(torch.Tensor(pcd_n).unsqueeze(0))

    pcd_vis = o3d.geometry.PointCloud()
    pcd_vis.points = o3d.utility.Vector3dVector(pcd_n.reshape(-1, 3))
    pcd_vis.paint_uniform_color([0, 0, 0])

    kp_vis = o3d.geometry.PointCloud()
    kp_vis.points = o3d.utility.Vector3dVector(kp_n.reshape(-1, 3))
    kp_vis.paint_uniform_color([1, 0, 0])

    pred_vis = o3d.geometry.PointCloud()
    pred_vis.points = o3d.utility.Vector3dVector(pred_n[:, :, :].numpy().reshape(-1, 3))
    pred_vis.paint_uniform_color([0, 1, 0])

    o3d.visualization.draw_geometries([pcd_vis, kp_vis])
