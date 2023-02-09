# -*- coding: utf-8 -*-
import os
import random
import argparse
import torch
import numpy as np
from merger.data_flower import all_h5
from merger.merger_net import Net
from tqdm import tqdm

from torch.utils.data import DataLoader, Dataset


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-t', '--train-data-dir', type=str, default='./point_cloud/train',
                        help='Directory that contains training .h5 files.')
arg_parser.add_argument('-v', '--val-data-dir', type=str, default='./point_cloud/test',
                        help='Directory that contains validation .h5 files.')
arg_parser.add_argument('-c', '--subclass', type=int, default=30,
                        help='Subclass label ID to train on.')  # 14 is `chair` class. 0 is airplane. 38 is mug. 45 is rifle. 13 is car. 18 is table. 27 is guitar. 44 is remote. 8 is bus.
# arg_parser.add_argument('-c', '--subclass', type=str, default='02691156',
#                        help='Subclass label ID to train on.')
arg_parser.add_argument('-m', '--merger-pretrain', '--saved-model-path', type=str, default='./saved_models/knife_merger_k4.pt',
                        help='Model checkpoint file path for saving.')
arg_parser.add_argument('-k', '--n-keypoint', type=int, default=4,
                        help='Requested number of keypoints to detect.')
arg_parser.add_argument('-d', '--device', type=str, default='cuda:1',
                        help='Pytorch device for training.')
arg_parser.add_argument('-b', '--batch', type=int, default=16,
                        help='Batch size.')
arg_parser.add_argument('--max-points', type=int, default=2048,
                        help='Indicates maximum points in each input point cloud.')

# args for generating segmentations
arg_parser.add_argument('--dist_threshold', type=float, default=0.10, help='w.r.t. the normalized distance')
arg_parser.add_argument('--expand', type=float, default=1.2, help='the coordinates of the keypoints are expanded so they are stretched to the edges')
arg_parser.add_argument('--segment-output', type=str, default='./saved_segments/knife_expand.npz')
args = arg_parser.parse_args()


keypoint_select_dict = {
    'guitar': [1, 2, 3, 5],
    'knife': [0, 2, 3]
}


def generate_keypoints(dataloader):
    with torch.no_grad():
        skeleton_merger.eval()
        for batch_id, batch_pcd in enumerate(tqdm(dataloader)):
            _, batch_keypoints, _, _, _ = skeleton_merger(batch_pcd.to(args.device))
            if batch_id == 0:
                keypoints = batch_keypoints.cpu().numpy()
            else:
                keypoints = np.concatenate((keypoints, batch_keypoints.cpu().numpy()), axis=0)  # [D, N, K]
    return keypoints


def generate_segments(pcds, keypoints):
    def generate_segment_mask(keypoint, pcd):
        """
        :argument
        keypoint: K x 3
        pcd: N x 3
        :return
        mask: N x K
        """
        distances = np.sqrt(np.sum(np.square(pcd.reshape(-1, 1, 3) - keypoint.reshape(1, -1, 3)), axis=-1))  # [N, K]
        # Normalized distance
        distances = (distances - np.min(distances, axis=0)) / (np.max(distances, axis=0) - np.min(distances, axis=0))
        # Fill the mask
        mask = np.where(distances < args.dist_threshold, 1., 0.)
        return mask

    segment_masks = []
    for pcd, keypoint in zip(pcds, keypoints):
        segment_mask = generate_segment_mask(keypoint[keypoint_select_dict['knife'], :] * args.expand, pcd)  # [N, K]
        segment_masks.append(segment_mask)
    return np.asarray(segment_masks)


if __name__ == '__main__':
    x, xl = all_h5(args.train_data_dir, True, True, subclasses=(args.subclass,), sample=None)  # n x 2048 x 3
    x_test, xl_test = all_h5(args.val_data_dir, True, True, subclasses=(args.subclass,), sample=None)
    train_dataloader = DataLoader(x, batch_size=args.batch, shuffle=False)
    test_dataloader = DataLoader(x_test, batch_size=args.batch, shuffle=False)
    skeleton_merger = Net(2048, args.n_keypoint, 'PointNet2')
    skeleton_merger.load_state_dict(torch.load(args.merger_pretrain, map_location=torch.device('cpu'))['model_state_dict'])
    skeleton_merger.to(args.device)

    train_keypoints = generate_keypoints(train_dataloader)
    test_keypoints = generate_keypoints(test_dataloader)

    train_segments = generate_segments(x, train_keypoints)
    test_segments = generate_segments(x_test, test_keypoints)
    np.savez(args.segment_output, train_pcds=x, test_pcds=x_test, train_segments=train_segments, test_segments=test_segments)
