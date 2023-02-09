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

from torch.utils.data import DataLoader, Dataset
from pytorch3d.transforms import RotateAxisAngle, Rotate, random_rotations

import copy

arg_parser = argparse.ArgumentParser(description="Training Skeleton Merger. Valid .h5 files must contain a 'data' array of shape (N, n, 3) and a 'label' array of shape (N, 1).", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
arg_parser.add_argument('-t', '--train-data-dir', type=str, default='./point_cloud/train',
                        help='Directory that contains training .h5 files.')
arg_parser.add_argument('-v', '--val-data-dir', type=str, default='./point_cloud/test',
                        help='Directory that contains validation .h5 files.')
arg_parser.add_argument('-c', '--subclass', type=int, default=30,
                        help='Subclass label ID to train on.')  # 14 is `chair` class. 0 is airplane. 38 is mug. 45 is rifle. 13 is car. 18 is table. 27 is guitar. 30 is knife.
# arg_parser.add_argument('-c', '--subclass', type=str, default='02691156',
#                        help='Subclass label ID to train on.')
arg_parser.add_argument('-m', '--checkpoint', '--saved-model-path', type=str, default='./saved_models/knife_merger_da.pt',
                        help='Model checkpoint file path for saving.')
arg_parser.add_argument('-k', '--n-keypoint', type=int, default=3,
                        help='Requested number of keypoints to detect.')
arg_parser.add_argument('-d', '--device', type=str, default='cuda:1',
                        help='Pytorch device for training.')
arg_parser.add_argument('-b', '--batch', type=int, default=16,
                        help='Batch size.')
arg_parser.add_argument('-e', '--epochs', type=int, default=15,
                        help='Number of epochs to train.')
arg_parser.add_argument('--max-points', type=int, default=2048,
                        help='Indicates maximum points in each input point cloud.')

arg_parser.add_argument('--encoder-type', type=str, default='PointNet2', choices=['PointNet2', 'DGCNN', 'EQCNN', 'EQPointNet', 'SPRIN'])
arg_parser.add_argument('--data-augmentation', type=str, default='so3', choices=['aligned', 'z', 'so3'])
arg_parser.add_argument('--sparse', type=bool, default=False)
arg_parser.add_argument('--correlation', type=bool, default=False)


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


def L2(embed):
    return 0.01 * (torch.sum(embed ** 2))


class Data_aug(Dataset):
    def __init__(self, data, aug=None):
        self.data = data
        if aug == 'z':
            self.transform = RotateAxisAngle(angle=torch.rand(1) * 360, axis="Z", degrees=True)
        elif aug == 'so3':
            self.transform = Rotate(R=random_rotations(1))
        else:
            self.transform = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pcd = self.data[idx]
        if self.transform:
            pcd = self.transform.transform_points(torch.tensor(pcd))
        return pcd


def train_loop(dataloader, model, optimizer, epoch):
    size = len(dataloader.dataset)
    for batch, x in enumerate(dataloader):
        x = pcd_normalization(x).float().to(args.device)
        RPCD, KPCD, KPA, LF, MA = model(x)
        loss_recon = composed_sqrt_chamfer(x, RPCD, MA)
        loss_div = L2(LF)

        loss = loss_recon + loss_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(x)
            print(f"recon: {loss_recon:>7f} div: {loss_div:>7f}  [{current:>5d}/{size:>5d}]")


def eval_loop(dataloader, model, epoch):
    num_batches = len(dataloader)
    test_recon = 0
    test_div = 0

    with torch.no_grad():
        for x in dataloader:
            x = pcd_normalization(x).float().to(args.device)
            RPCD, KPCD, KPA, LF, MA = model(x)
            loss_recon = composed_sqrt_chamfer(x, RPCD, MA)
            loss_div = L2(LF)
            test_recon += loss_recon
            test_div += loss_div

    test_recon /= num_batches
    test_div /= num_batches
    print(f"Test Error: \n Avg recon: {test_recon:>8f} Avg div: {test_div:>8f}\n")
    return test_recon


if __name__ == '__main__':
    args = arg_parser.parse_args()
    DATASET = args.train_data_dir
    TESTSET = args.val_data_dir
    batch = args.batch
    x, xl = all_h5(DATASET, True, True, subclasses=(args.subclass,), sample=None)  # n x 2048 x 3
    x_test, xl_test = all_h5(TESTSET, True, True, subclasses=(args.subclass,), sample=None)
    dataset = Data_aug(x, args.data_augmentation)
    dataset_test = Data_aug(x_test, args.data_augmentation)
    train_dataloader = DataLoader(dataset, batch_size=args.batch)
    test_dataloader = DataLoader(dataset_test, batch_size=args.batch)
    net = Net(args.max_points, args.n_keypoint, args.encoder_type).to(args.device)
    optimizer = optim.Adam(net.parameters())

    eval_loss = 1e8
    for t in range(args.epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, net, optimizer, t)
        new_eval_loss = eval_loop(test_dataloader, net, t)
        if new_eval_loss < eval_loss:
            eval_loss = new_eval_loss
            torch.save({
                'epoch': t,
                'model_state_dict': net.state_dict(),
            }, args.checkpoint)
