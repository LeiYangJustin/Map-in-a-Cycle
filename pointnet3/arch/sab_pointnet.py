import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
from base import BaseModel
from pointnet3.arch.yanx27_pointnet_util import PointNetSetAbstractionMsg,PointNetSetAbstraction,PointNetFeaturePropagation
from pointnet3.arch.sab_modules import SAB

class Pointnet2MSG_yanx27_sab_partseg(BaseModel):
    def __init__(self, output_channels, additional_channels=0):
        super(Pointnet2MSG_yanx27_sab_partseg, self).__init__()
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

        ## Self-attention modules
        num_heads = 4
        # for set abstraction
        out_channel_sa1, out_channel_sa2, out_channel_sa3=128+128+64, 256+256, 1024
        self.att_sa1 = SAB(out_channel_sa1, out_channel_sa1, num_heads, ln=True)
        self.att_sa2 = SAB(out_channel_sa2, out_channel_sa2, num_heads, ln=True)
        self.att_sa3 = SAB(out_channel_sa3, out_channel_sa3, num_heads, ln=True)
        # for feature propagation
        out_channel_fp1, out_channel_fp2, out_channel_fp3=128, 256, 256
        self.att_fp3 = SAB(out_channel_fp3, out_channel_fp3, num_heads, ln=True)
        self.att_fp2 = SAB(out_channel_fp2, out_channel_fp2, num_heads, ln=True)
        self.att_fp1 = SAB(out_channel_fp1, out_channel_fp1, num_heads, ln=True)

        # self.activate_att = [True, True, True, True, True, True]

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
            assert l0_points is not None, "l0_points shall not be None!"
            l0_points = l0_points[..., 0:self.additional_channels]
            l0_points = l0_points.permute(0,2,1)
        
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l1_points = self.att_sa1(l1_points.permute(0,2,1)).permute(0,2,1)
        
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l2_points = self.att_sa2(l2_points.permute(0,2,1)).permute(0,2,1)
        
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l3_points = self.att_sa3(l3_points.permute(0,2,1)).permute(0,2,1)
        
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l2_points = self.att_fp3(l2_points.permute(0,2,1)).permute(0,2,1)
        
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l1_points = self.att_fp2(l1_points.permute(0,2,1)).permute(0,2,1)

        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)
        l0_points = self.att_fp1(l0_points.permute(0,2,1)).permute(0,2,1)

        # FC layers
        if False:
            feat = F.relu(self.bn1(self.conv1(l0_points)))
            x = self.conv2(self.drop1(feat))
        else:
            x = self.conv2(l0_points)
        x = x.permute(0, 2, 1) ## OUTPUT: B, N, D
        return x

# class Pointnet2MSG_transformer(BaseModel):
#     def __init__(self, output_channels, additional_channels=0):
#         super(Pointnet2MSG_transformer, self).__init__()
#         self.additional_channels = additional_channels
#         num_heads = 4 # Self-attention modules

#         ## Set abstraction for local features
#         self.sa = PointNetSetAbstractionMsg(
#             npoint=512,
#             radius_list=[0.1, 0.2, 0.4], 
#             nsample_list=[32, 64, 128], 
#             in_channel=additional_channels, # PSAMsg [prechannel]
#             mlp_list=[[32, 32, 64], [64, 64, 128], [64, 96, 128]]
#             )
#         ## Self attention for non-local features
#         self.transformer = nn.Sequential(
#             SAB(64+128+128, 512, 8, ln=True),
#             SAB(512, 512, 8, ln=True),
#             SAB(512, 256, num_heads, ln=True),
#             SAB(256, 256, num_heads, ln=True),
#             SAB(256, 128, num_heads, ln=True),
#         )
#         self.fp = PointNetFeaturePropagation(
#             in_channel=128+additional_channels, 
#             mlp=[128, 128, 128])
#         self.conv = nn.Conv1d(128, output_channels, 1)

#     def _break_up_pc(self, pc):
#         xyz = pc[..., 0:3].contiguous()
#         features = (
#             # pc[..., 3:].transpose(1, 2).contiguous()
#             pc[..., 3:].contiguous()
#             if pc.size(-1) > 3 else None
#         )
#         return xyz, features

#     def forward(self, xyz):
#         """
#         xyz: B, N, C(+D)
#         """
#         # Set Abstraction layers
#         if len(xyz.shape) == 4:
#             B0, b, N, C = xyz.shape
#             xyz = xyz.reshape(-1, N, C)
        
#         l0_xyz, l0_points = self._break_up_pc(xyz)
#         l0_xyz = l0_xyz.permute(0, 2, 1)
        
#         if self.additional_channels == 0:
#             l0_points = None
#         else:
#             assert l0_points is not None, "l0_points shall not be None!"
#             l0_points = l0_points[..., 0:self.additional_channels]
#             l0_points = l0_points.permute(0,2,1)
        
#         ## network process
#         l1_xyz, l1_points = self.sa(l0_xyz, l0_points)
#         feat = self.transformer(l1_points.permute(0,2,1)).permute(0,2,1)
#         l0_points = self.fp(l0_xyz, l1_xyz, l0_points, feat)
#         # FC layers
#         x = self.conv(l0_points)
#         x = x.permute(0, 2, 1) ## OUTPUT: B, N, D
#         return x