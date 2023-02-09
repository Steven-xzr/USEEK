import os
import numpy as np
import h5py
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import random
from dgl.geometry import farthest_point_sampler
import copy

from merger.sprin.model import SPRINSeg


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--segment_dataset', type=str, default='./saved_segments/knife_expand.npz')
arg_parser.add_argument('--checkpoint', type=str, default='./saved_models/knife_segment_k3_expand.pt',
                        help='Model checkpoint file path for saving.')
arg_parser.add_argument('--device', type=str, default='cuda:1',
                        help='Pytorch device for training.')
arg_parser.add_argument('--batch', type=int, default=8,
                        help='Batch size.')
arg_parser.add_argument('--lr', type=float, default=1e-4)
arg_parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs to train.')
arg_parser.add_argument('-k', '--n-keypoints', type=int, default=3,
                        help='Requested number of keypoints to detect.')
arg_parser.add_argument('--corr_factor', type=float, default=0)
arg_parser.add_argument('--do_negative_sampling', action='store_true', default=True)
arg_parser.add_argument('--do_fps', action='store_true', default=False)
arg_parser.add_argument('--negative_sampling_factor', type=int, default=3)
args = arg_parser.parse_args()


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


class SegmentMask(Dataset):
    def __init__(self, pcds, masks):
        self.pcds = pcds
        self.masks = masks

    def __len__(self):
        return self.pcds.shape[0]

    def __getitem__(self, idx):
        return self.pcds[idx], self.masks[idx]


def loss(predicts, labels, pcds):   # [B, N, K]
    corr_matrix = torch.einsum('bij,bjk->bik', predicts, predicts.permute(0, 2, 1))  # [B, K, K]
    trace = torch.einsum('bii->b', corr_matrix)  # b
    corr = (torch.sum(corr_matrix) - torch.sum(trace)) * 0.5 / predicts.numel()

    if not args.do_negative_sampling:
        bce = F.binary_cross_entropy(predicts, labels)
    else:
        bce = 0
        for predict, label, pcd in zip(predicts, labels, pcds):     # [N, K]
            predict = predict.permute(1, 0)     # [K, N]
            label = label.permute(1, 0)     # [K, N]
            for ii, label_ in enumerate(label):     # [N,]
                positive_mask = torch.where(label_ == 1)[0]
                negative_mask = torch.where(label_ == 0)[0]
                if args.do_fps:
                    negative_pcd = pcd[negative_mask]   # [m, 3]
                    sampled_index = farthest_point_sampler(negative_pcd.unsqueeze(0), min(len(negative_mask), len(positive_mask) * args.negative_sampling_factor)).squeeze()
                else:
                    sampled_index = torch.LongTensor(random.sample(range(len(negative_mask)), min(len(negative_mask), len(positive_mask) * args.negative_sampling_factor))).to(args.device)
                mask = torch.concat((positive_mask, negative_mask[sampled_index]))
                bce += F.binary_cross_entropy(predict[ii, mask], label_[mask])
    bce /= predicts.shape[0] * predicts.shape[2]

    return bce, corr


def train_loop(dataloader, model, loss_fn, optimizer, epoch):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X = pcd_normalization(X)
        pred, _ = model(X.float().to(args.device))
        bce, corr = loss_fn(torch.sigmoid(pred), y.float().to(args.device), X)
        loss = bce + corr * args.corr_factor

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"BCE: {bce:>7f} Corr: {corr:>7f} [{current:>5d}/{size:>5d}]")


def eval_loop(dataloader, model, loss_fn, epoch):
    num_batches = len(dataloader)
    test_bce = 0
    test_corr = 0

    with torch.no_grad():
        for X, y in dataloader:
            X = pcd_normalization(X)
            pred, _ = model(X.float().to(args.device))
            bce, corr = loss_fn(torch.sigmoid(pred), y.float().to(args.device), X)
            test_bce += bce
            test_corr += corr

    test_bce /= num_batches
    test_corr /= num_batches
    print(f"Test Error: \n Avg bce: {test_bce:>8f} Avg corr: {test_corr:>8f}\n")
    return test_bce


if __name__ == '__main__':
    segment_dataset = np.load(args.segment_dataset)
    train_dataset = SegmentMask(segment_dataset['train_pcds'], segment_dataset['train_segments'])
    test_dataset = SegmentMask(segment_dataset['test_pcds'], segment_dataset['test_segments'])
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch, shuffle=True)

    model = SPRINSeg(args.n_keypoints).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    eval_loss = 1e8

    for t in range(args.epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss, optimizer, t)
        new_eval_loss = eval_loop(test_dataloader, model, loss, t)
        if new_eval_loss < eval_loss:
            eval_loss = new_eval_loss
            torch.save({
                'epoch': t,
                'model_state_dict': model.state_dict(),
            }, args.checkpoint)
