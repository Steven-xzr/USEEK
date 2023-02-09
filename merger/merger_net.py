# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from merger.pointnetpp.pointnet2_sem_seg_msg import get_model as PointNetPP
from merger.pointnet_equi.pointnet_part_seg import get_model as EQPointNet
from merger.vn_dgcnn.model import DGCNN_partseg as DGCNN
from merger.vn_dgcnn.model_equi import EQCNN_partseg as EQCNN
from merger.sprin.model import SPRINSeg as SPRIN


torch.square = lambda x: x ** 2


class PBlock(nn.Module):  # MLP Block
    def __init__(self, iu, *units, should_perm):
        super().__init__()
        self.sublayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.should_perm = should_perm
        ux = iu
        for uy in units:
            self.sublayers.append(nn.Linear(ux, uy))
            self.batch_norms.append(nn.BatchNorm1d(uy))
            ux = uy

    def forward(self, input_x):
        x = input_x
        for sublayer, batch_norm in zip(self.sublayers, self.batch_norms):
            x = sublayer(x)
            if self.should_perm:
                x = x.permute(0, 2, 1)
            x = batch_norm(x)
            if self.should_perm:
                x = x.permute(0, 2, 1)
            x = F.relu(x)
        return x


class Head(nn.Module):  # Decoder unit, one per line
    def __init__(self):
        super().__init__()
        self.emb = nn.Parameter(torch.randn((200, 3)) * 0.002)

    def forward(self, KPA, KPB):
        dist = torch.mean(torch.sqrt(1e-3 + (torch.sum(torch.square(KPA - KPB), dim=-1))))
        count = min(200, max(15, int((dist / 0.01).item())))
        device = dist.device
        self.f_interp = torch.linspace(0.0, 1.0, count).unsqueeze(0).unsqueeze(-1).to(device)
        self.b_interp = 1.0 - self.f_interp
        # KPA: N x 3, KPB: N x 3
        # Interpolated: N x count x 3
        K = KPA.unsqueeze(-2) * self.f_interp + KPB.unsqueeze(-2) * self.b_interp
        R = self.emb[:count, :].unsqueeze(0) + K  # N x count x 3
        return R.reshape((-1, count, 3)), self.emb


class Net(nn.Module):  # Skeleton Merger structure
    def __init__(self, npt, k, enc_type):
        super().__init__()
        self.npt = npt
        self.k = k
        self.enc_type = enc_type

        # Modifications here
        if enc_type == 'PointNet2':
            self.PTW = PointNetPP(k)
        elif enc_type == 'DGCNN':
            self.PTW = DGCNN(k)
        elif enc_type == 'EQCNN':
            self.PTW = EQCNN(k)
        elif enc_type == 'EQPointNet':
            self.PTW = EQPointNet(part_num=self.k)
        elif enc_type == 'SPRIN':
            self.PTW = SPRIN(n_classes=self.k)
        else:
            self.PTW = None

        self.PT_L = nn.Linear(k, k)
        self.MA_EMB = nn.Parameter(torch.randn([k * (k - 1) // 2]))
        if enc_type == 'EQCNN':
            self.MA = PBlock(2046, 512, 256, should_perm=False)    #1024//3*3 or 2046
        elif enc_type == 'EQPointNet':
            self.MA = PBlock(2048//3*6, 512, 256, should_perm=False)
        elif enc_type == 'SPRIN':
            self.MA = PBlock(256+32, 512, 256, should_perm=False)
        else:
            self.MA = PBlock(1024, 512, 256, should_perm=False)
        self.MA_L = nn.Linear(256, k * (k - 1) // 2)
        self.DEC = nn.ModuleList()
        for i in range(k):
            DECN = nn.ModuleList()
            for j in range(i):
                DECN.append(Head())
            self.DEC.append(DECN)

    def forward(self, input_x):
        if self.enc_type == 'PointNet2':
            APP_PT = torch.cat([input_x, input_x, input_x], -1)         # input:[batch, n, 3]
            KP, GF = self.PTW(APP_PT.permute(0, 2, 1))                  # PTW:[batch, 3*3, n] -> KP:[batch, n, k] & GF:[batch, emb_dim(1024), 16]
        elif self.enc_type == 'SPRIN':
            KP, GF = self.PTW(input_x)
        else:
            KP, GF = self.PTW(input_x.permute(0, 2, 1))                 # KP:[batch, n, k] & GF:[batch, emb_dim(1024)]

        KPL = self.PT_L(KP)                                             # KPL:[batch, n, k]
        KPA = F.softmax(KPL.permute(0, 2, 1), -1)                       # KPL:[batch, k, n]
        '''
        if np.random.rand() < 0.01:
            print(KPA[np.random.randint(8)])'''

        KPCD = KPA.bmm(input_x)                                         # KPCD:[batch, k, 3]    ### keypoint proposals
        RP = []                                                         # RP: len == k*(k-1)/2
        L = []
        for i in range(self.k):
            for j in range(i):
                R, EM = self.DEC[i][j](KPCD[:, i, :], KPCD[:, j, :])    # per skeleton:
                RP.append(R)                                            # R:[batch, ki, 3]
                L.append(EM)                                            # EM:[200, 3]       ### offset
        if self.enc_type == 'PointNet2':
            GF = F.max_pool1d(GF, 16).squeeze(-1)                            # GF:[batch, 1024]
        MA = F.sigmoid(self.MA_L(self.MA(GF)))                         # MA:[batch, k*(k-1)/2]
        # MA = torch.sigmoid(self.MA_EMB).expand(input_x.shape[0], -1)
        LF = torch.cat(L, dim=1)  # P x 72 x 3
        return RP, KPCD, KPA, LF, MA


class StudentNet(nn.Module):
    def __init__(self, npt, k, enc_type, strategy='keypoint'):
        super().__init__()
        self.npt = npt
        self.k = k
        self.enc_type = enc_type

        # Modifications here
        if enc_type == 'PointNet2':
            self.PTW = PointNetPP(k)
        elif enc_type == 'DGCNN':
            self.PTW = DGCNN(k)
        elif enc_type == 'EQCNN':
            self.PTW = EQCNN(k)
        elif enc_type == 'EQPointNet':
            self.PTW = EQPointNet(part_num=self.k)
        elif enc_type == 'SPRIN':
            self.PTW = SPRIN(n_classes=self.k)
        else:
            self.PTW = None

        self.PT_L = nn.Linear(k, k)
        self.MA_EMB = nn.Parameter(torch.randn([k * (k - 1) // 2]))
        if enc_type == 'EQCNN':
            self.MA = PBlock(2046, 512, 256, should_perm=False)    #1024//3*3 or 2046
        elif enc_type == 'EQPointNet':
            self.MA = PBlock(2048//3*6, 512, 256, should_perm=False)
        elif enc_type == 'SPRIN':
            self.MA = PBlock(256+32, 512, 256, should_perm=False)
        else:
            self.MA = PBlock(1024, 512, 256, should_perm=False)
        self.MA_L = nn.Linear(256, k * (k - 1) // 2)
        self.DEC = nn.ModuleList()
        for i in range(k):
            DECN = nn.ModuleList()
            for j in range(i):
                DECN.append(Head())
            self.DEC.append(DECN)

    def forward(self, input_x):
        if self.enc_type == 'PointNet2':
            APP_PT = torch.cat([input_x, input_x, input_x], -1)         # input:[batch, n, 3]
            KP, GF = self.PTW(APP_PT.permute(0, 2, 1))                  # PTW:[batch, 3*3, n] -> KP:[batch, n, k] & GF:[batch, emb_dim(1024), 16]
        elif self.enc_type == 'SPRIN':
            KP, GF = self.PTW(input_x)
        else:
            KP, GF = self.PTW(input_x.permute(0, 2, 1))                 # KP:[batch, n, k] & GF:[batch, emb_dim(1024)]

        KPL = self.PT_L(KP)                                             # KPL:[batch, n, k]
        KPA = F.softmax(KPL.permute(0, 2, 1), -1)                       # KPL:[batch, k, n]     activation strength
        '''
        if np.random.rand() < 0.01:
            print(KPA[np.random.randint(8)])'''

        KPCD = KPA.bmm(input_x)                                         # KPCD:[batch, k, 3]    ### keypoint proposals
        RP = []                                                         # RP: len == k*(k-1)/2
        L = []
        for i in range(self.k):
            for j in range(i):
                R, EM = self.DEC[i][j](KPCD[:, i, :], KPCD[:, j, :])    # per skeleton:
                RP.append(R)                                            # R:[batch, ki, 3]
                L.append(EM)                                            # EM:[200, 3]       ### offset
        if self.enc_type == 'PointNet2':
            GF = F.max_pool1d(GF, 16).squeeze(-1)                            # GF:[batch, 1024]
        MA = F.sigmoid(self.MA_L(self.MA(GF)))                         # MA:[batch, k*(k-1)/2]
        # MA = torch.sigmoid(self.MA_EMB).expand(input_x.shape[0], -1)
        LF = torch.cat(L, dim=1)  # P x 72 x 3
        return RP, KPCD, KPA, LF, MA
