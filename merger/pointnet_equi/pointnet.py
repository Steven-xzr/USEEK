import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from .layers import *


class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x

class STNkd(nn.Module):
    def __init__(self, pooling, d=64):
        super(STNkd, self).__init__()
        self.pooling = pooling
        
        self.conv1 = VNLinearLeakyReLU(d, 64//3, dim=4, negative_slope=0.0)
        self.conv2 = VNLinearLeakyReLU(64//3, 128//3, dim=4, negative_slope=0.0)
        self.conv3 = VNLinearLeakyReLU(128//3, 1024//3, dim=4, negative_slope=0.0)

        self.fc1 = VNLinearLeakyReLU(1024//3, 512//3, dim=3, negative_slope=0.0)
        self.fc2 = VNLinearLeakyReLU(512//3, 256//3, dim=3, negative_slope=0.0)
        
        if self.pooling == 'max':
            self.pool = VNMaxPool(1024//3)
        elif self.pooling == 'mean':
            self.pool = mean_pool
        
        self.fc3 = VNLinear(256//3, d)
        self.d = d

    def forward(self, x):
        batchsize = x.size()[0]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, args, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        self.args = args
        self.n_knn = 20
        
        self.conv_pos = VNLinearLeakyReLU(3, 64//3, dim=5, negative_slope=0.0)
        self.conv1 = VNLinearLeakyReLU(64//3, 64//3, dim=4, negative_slope=0.0)
        self.conv2 = VNLinearLeakyReLU(64//3*2, 128//3, dim=4, negative_slope=0.0)
        
        self.conv3 = VNLinear(128//3, 1024//3)
        self.bn3 = VNBatchNorm(1024//3, dim=4)
        
        self.std_feature = VNStdFeature(1024//3*2, dim=4, normalize_frame=False, negative_slope=0.0)
        
        if args.pooling == 'max':
            self.pool = VNMaxPool(64//3)
        elif args.pooling == 'mean':
            self.pool = mean_pool
        
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        
        if self.feature_transform:
            self.fstn = STNkd(args, d=64//3)

    def forward(self, x):
        B, D, N = x.size()
        
        x = x.unsqueeze(1)
        feat = get_graph_feature_cross(x, k=self.n_knn)
        x = self.conv_pos(feat)     # batch_size x channel x 3 x n_samples x n_knn
        x = self.pool(x)       # batch_size x channel x 3 x n_samples
        x = self.conv1(x)
        
        if self.feature_transform:
            x_global = self.fstn(x).unsqueeze(-1).repeat(1,1,1,N)
            x = torch.cat((x, x_global), 1)
        
        pointfeat = x
        x = self.conv2(x)
        x = self.bn3(self.conv3(x))
        
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x, trans = self.std_feature(x)
        x = x.view(B, -1, N)
        
        x = torch.max(x, -1, keepdim=False)[0]
        
        trans_feat = None
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat

def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1) - I), dim=(1, 2)))
    return loss
