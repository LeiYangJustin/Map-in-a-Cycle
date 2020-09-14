## This file is partly adapted from PointNet++ pytorch implementation from this github repo https://github.com/yanx27/Pointnet_Pointnet2_pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
from base import BaseModel
from pointnet3.arch.yanx27_pointnet_util import PointNetSetAbstractionMsg,PointNetSetAbstraction,PointNetFeaturePropagation



class Pointnet2MSG_yanx27_vanilla(BaseModel):
    def __init__(self, output_channels, additional_channels=0):
        super(Pointnet2MSG_yanx27_vanilla, self).__init__()
        self.additional_channels = additional_channels
        self.sa1 = PointNetSetAbstractionMsg(
            npoint=512,
            radius_list=[0.1, 0.2, 0.4], 
            nsample_list=[32, 64, 128], 
            in_channel=additional_channels, # PSAMsg [prechannel]
            mlp_list=[[32, 32, 64], [64, 64, 128], [64, 96, 128]]
            )
        self.sa2 = PointNetSetAbstractionMsg(
            npoint=128,
            radius_list=[0.4, 0.8], 
            nsample_list=[64, 128], 
            in_channel=64+128+128,  # PSAMsg [prechannel]
            mlp_list=[[128, 128, 256], [128, 196, 256]]
            )
        self.sa3 = PointNetSetAbstraction(
            npoint=None,
            radius=None, 
            nsample=None, 
            in_channel=256+256+3, # PSA [prechannel+3]
            mlp=[256, 512, 1024],
            group_all=True
            )
        self.fp3 = PointNetFeaturePropagation(
            in_channel=1024+256+256, 
            mlp=[512, 256])
        self.fp2 = PointNetFeaturePropagation(
            in_channel=256+128+128+64, 
            mlp=[256, 256])
        self.fp1 = PointNetFeaturePropagation(
            in_channel=256+additional_channels, 
            mlp=[128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, output_channels, 1)
        self.conv2 = nn.Conv1d(128, output_channels, 1)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            # pc[..., 3:].transpose(1, 2).contiguous()
            pc[..., 3:].contiguous()
            if pc.size(-1) > 3 else None
        )
        return xyz, features

    def forward(self, xyz):
        """
        xyz: B, N, C(+D)
        """
        # Set Abstraction layers
        if len(xyz.shape) == 4:
            B0, b, N, C = xyz.shape
            xyz = xyz.reshape(-1, N, C)
        
        l0_xyz, l0_points = self._break_up_pc(xyz)
        l0_xyz = l0_xyz.permute(0, 2, 1)
        
        if self.additional_channels == 0:
            l0_points = None
        else:
            assert l0_points is not None, "lo_points shall not be None!"
            l0_points = l0_points[..., 0:self.additional_channels]
            l0_points = l0_points.permute(0,2,1)

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)

        # FC layers
        if False:
            feat = F.relu(self.bn1(self.conv1(l0_points)))
            x = self.conv2(self.drop1(feat))
        else:
            x = self.conv2(l0_points)
        x = x.permute(0, 2, 1) ## OUTPUT: B, N, D
        return x
