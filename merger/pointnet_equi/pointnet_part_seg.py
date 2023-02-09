import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from .layers import *
from .pointnet import STN3d, STNkd, feature_transform_reguliarzer


class get_model(nn.Module):
    def __init__(self, part_num=50, normal_channel=False, k=20):
        super(get_model, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.part_num = part_num
        self.pooling = 'mean'
        self.k = k
        
        self.conv_pos = VNLinearLeakyReLU(3, 64//3, dim=5, negative_slope=0.0)
        self.conv1 = VNLinearLeakyReLU(64//3, 64//3, dim=4, negative_slope=0.0)
        self.conv2 = VNLinearLeakyReLU(64//3, 128//3, dim=4, negative_slope=0.0)
        self.conv3 = VNLinearLeakyReLU(128//3, 128//3, dim=4, negative_slope=0.0)
        self.conv4 = VNLinearLeakyReLU(128//3*2, 512//3, dim=4, negative_slope=0.0)
        
        self.conv5 = VNLinear(512//3, 2048//3)
        self.bn5 = VNBatchNorm(2048//3, dim=4)
        
        self.std_feature = VNStdFeature(2048//3*2, dim=4,
                                        normalize_frame=False, negative_slope=0.0)
        
        if self.pooling == 'max':
            self.pool = VNMaxPool(64//3)
        elif self.pooling == 'mean':
            self.pool = mean_pool
        
        self.fstn = STNkd(self.pooling, d=128//3)
        
        self.convs1 = torch.nn.Conv1d(9025-16, 256, 1)
        self.convs2 = torch.nn.Conv1d(256, 256, 1)
        self.convs3 = torch.nn.Conv1d(256, 128, 1)
        self.convs4 = torch.nn.Conv1d(128, part_num, 1)
        self.bns1 = nn.BatchNorm1d(256)
        self.bns2 = nn.BatchNorm1d(256)
        self.bns3 = nn.BatchNorm1d(128)

    def forward(self, point_cloud):
        B, D, N = point_cloud.size()
        
        point_cloud = point_cloud.unsqueeze(1)
        feat = get_graph_feature_cross(point_cloud, k=self.k)
        point_cloud = self.conv_pos(feat)
        point_cloud = self.pool(point_cloud)

        out1 = self.conv1(point_cloud)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        
        trans_feat = None
        net_global = self.fstn(out3).unsqueeze(-1).repeat(1,1,1,N)
        net_transformed = torch.cat((out3, net_global), 1)

        out4 = self.conv4(net_transformed)
        out5 = self.bn5(self.conv5(out4))
        
        out5_mean = out5.mean(dim=-1, keepdim=True).expand(out5.size())
        out5 = torch.cat((out5, out5_mean), 1)
        out5, trans = self.std_feature(out5)
        out5 = out5.view(B, -1, N)
        
        out_max = torch.max(out5, -1, keepdim=False)[0]     # [batch, 2048//3*6] ?

        #out_max = torch.cat([out_max,label.squeeze(1)],1)
        expand = out_max.view(-1, 2048//3*6, 1).repeat(1, 1, N)
        
        out1234 = torch.cat((out1, out2, out3, out4), dim=1)
        out1234 = torch.einsum('bijm,bjkm->bikm', out1234, trans).view(B, -1, N)
        
        concat = torch.cat([expand, out1234, out5], 1)
        
        net = F.relu(self.bns1(self.convs1(concat)))
        net = F.relu(self.bns2(self.convs2(net)))
        net = F.relu(self.bns3(self.convs3(net)))
        net = self.convs4(net)
        net = net.transpose(2, 1).contiguous()
        net = F.log_softmax(net.view(-1, self.part_num), dim=-1)
        net = net.view(B, N, self.part_num) # [B, N, 50]

        return net, out_max


class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        #self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target)
        #mat_diff_loss = feature_transform_reguliarzer(trans_feat)
        total_loss = loss #+ mat_diff_loss * self.mat_diff_loss_scale
        return total_loss
