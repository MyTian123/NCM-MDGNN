from torch import nn
from ops import MeshConv, MeshConv_transpose, ResBlock
import torch.nn.functional as F
import os
import torch
import pickle
import numpy as np


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, level, mesh_folder, bias=True):
        """
        use mesh_file for the mesh of one-level up
        """
        super().__init__()
        mesh_file = os.path.join(mesh_folder, "icosphere_{}.pkl".format(level))
        half_in = int(in_ch/2)
        self.up = MeshConv_transpose(half_in, half_in, mesh_file, stride=2)
        self.conv = ResBlock(in_ch, out_ch, out_ch, level, False, mesh_folder)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, level, mesh_folder, bias=True):
        """
        use mesh_file for the mesh of one-level down
        """
        super().__init__()
        self.conv = ResBlock(in_ch, in_ch, out_ch, level+1, True, mesh_folder)

    def forward(self, x):
        x = self.conv(x)
        return x


class SphericalUNet(nn.Module):
    def __init__(self, mesh_folder, in_ch, out_ch, max_level, min_level=0, fdim=64):
        super().__init__()
        self.mesh_folder = mesh_folder
        self.fdim = fdim
        self.max_level = max_level
        self.min_level = min_level
        self.levels = max_level - min_level
        self.down = []
        self.up = []
        self.in_conv = MeshConv(in_ch, fdim, self.__meshfile(max_level), stride=1)
        self.out_conv = MeshConv(fdim, out_ch, self.__meshfile(max_level), stride=1)
        # Downward path
        for i in range(self.levels-1):
            idx = i
            self.down.append(Down(fdim*(2**idx), fdim*(2**(idx+1)), max_level-i-1, mesh_folder))
        self.down.append(Down(fdim*(2**(self.levels-1)), fdim*(2**(self.levels-1)), min_level, mesh_folder))
        # Upward path
        for i in range(self.levels-1):
            self.up.append(Up(fdim*(2**(self.levels-i)), fdim*(2**(self.levels-i-2)), min_level+i+1, mesh_folder))
        self.up.append(Up(fdim*2, fdim, max_level, mesh_folder))
        self.down = nn.ModuleList(self.down)
        self.up = nn.ModuleList(self.up)

    def forward(self, x):
        x_ = [self.in_conv(x)]

        for i in range(self.levels):
            x_.append(self.down[i](x_[-1]))

        x = self.up[0](x_[-1], x_[-2])

        for i in range(self.levels-1):
            x = self.up[i+1](x, x_[-3-i])

        x = self.out_conv(x)

        return x 

    def __meshfile(self, i):
        return os.path.join(self.mesh_folder, "icosphere_{}.pkl".format(i))


class LinearRegressionModel(nn.Module):
    def __init__(self, shape, out, hs=64):
        super(LinearRegressionModel, self).__init__()
        self.linout = nn.Sequential(
            nn.Linear(shape, hs),
            nn.ReLU(inplace=True),
            nn.Linear(hs, hs),
            nn.ReLU(inplace=True),
            nn.Linear(hs, out),
        )

    def forward(self, x):
        out = self.linout(x) 
        return out


class Multi_uunet(nn.Module):
    def __init__(self, bs, mesh_folder, in_ch, out_ch, max_level, min_level, fdim=48):
        super(Multi_uunet, self).__init__()
        self.mesh_folder = mesh_folder
        self.fdim = fdim
        self.max_level = max_level
        self.min_level = min_level
        self.bs = bs
        self.gnn_layer = 2
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.relu = nn.GELU()
        self.hidden_ = 128
        self.vertex = 10*4**max_level+2
        self.spheric_aluNet2 = SphericalUNet(self.mesh_folder, self.hidden_, out_ch, self.max_level, self.min_level, self.fdim)
        self.Lmodel1 = LinearRegressionModel(out_ch, out_ch)
        self.Lmodel2 = LinearRegressionModel(in_ch, self.hidden_)

        self.Lmodel_north = nn.Linear(self.vertex, 1)
        self.Lmodel_res = nn.Linear(1, self.vertex)
        self.drop = 0
        self.dropout = nn.Dropout(0.2)

        # ── BN_1_3 固化统计量 (register_buffer 会随 state_dict 保存/加载) ──
        # 初始化为0，加载固化权重后被覆盖。若std仍为0则退化为动态计算（向后兼容）
        self.register_buffer('bn1_mean', torch.zeros(in_ch))
        self.register_buffer('bn1_std',  torch.zeros(in_ch))
        self.register_buffer('bn2_mean', torch.zeros(self.vertex))
        self.register_buffer('bn2_std',  torch.zeros(self.vertex))
        self.register_buffer('bn3_mean', torch.zeros(out_ch))
        self.register_buffer('bn3_std',  torch.zeros(out_ch))
        self.register_buffer('bn4_mean', torch.zeros(out_ch))
        self.register_buffer('bn4_std',  torch.zeros(out_ch))

    def forward(self, x):
        # 每次前向传播重置 BN 调用计数器
        self._bn_call_idx = 0

        x = x.unsqueeze(-1)
        x = self.Lmodel_res(x)
        x = self.BN_1_3(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)
        x = self.Lmodel2(x)
        x = self.BN_1_3(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)
        x = self.spheric_aluNet2(x)
        x = self.BN_1_3(x)
        x = self.relu(x)
        x = self.Lmodel_north(x) 
        x = self.BN_1_3(x)
        x = self.relu(x)
        x = x.squeeze(-1)
        x = self.Lmodel1(x)

        return x
    
    def BN_1_4(self, x):
        pre_x = x.permute(1, 0, 2, 3)
        pre_x = torch.flatten(pre_x, 1, -1)
        mean = pre_x.mean(dim=1)
        std = pre_x.std(dim=1)
        x = (x-mean.unsqueeze(-1).unsqueeze(-1).unsqueeze(0))/(std.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)+1e-5)
        return x
    
    def BN_1_3(self, x):
        """带固化统计量的 BN_1_3：若已加载固化权重则使用固定 mean/std，否则动态计算（向后兼容）"""
        idx = self._bn_call_idx % 4
        self._bn_call_idx += 1

        bn_means = [self.bn1_mean, self.bn2_mean, self.bn3_mean, self.bn4_mean]
        bn_stds  = [self.bn1_std,  self.bn2_std,  self.bn3_std,  self.bn4_std]
        m = bn_means[idx]
        s = bn_stds[idx]

        # std 全零 → 动态计算（训练时或加载旧权重时）
        if s.abs().sum() == 0:
            pre_x = x.permute(1, 0, 2)
            pre_x = torch.flatten(pre_x, 1, -1)
            m = pre_x.mean(dim=1)
            s = pre_x.std(dim=1)

        return (x - m.unsqueeze(-1).unsqueeze(0)) / (s.unsqueeze(-1).unsqueeze(0) + 1e-5)


class vertical_down(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(vertical_down, self).__init__()
        self.in_ch = in_channel
        self.out_ch = out_channel
        self.conv1ddown1 = nn.Conv1d(self.in_ch, 256, 1, stride=1)
        self.conv1ddown2 = nn.Conv1d(256, self.out_ch, 1, stride=1)
    def forward(self, x):
        x = self.conv1ddown1(x)
        x = self.conv1ddown2(x)
        return x


class vertical_up(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(vertical_up, self).__init__()
        self.in_ch = in_channel
        self.out_ch = out_channel
        self.conv1dup1 = nn.ConvTranspose1d(self.in_ch, 256, 1, stride=1)
        self.conv1dup2 = nn.ConvTranspose1d(256, self.out_ch, 1, stride=1)
    def forward(self, x):
        x = self.conv1dup1(x)
        x = self.conv1dup2(x)
        return x


class vertical_down3d(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(vertical_down3d, self).__init__()
        self.in_ch = in_channel
        self.out_ch = out_channel
        self.conv2ddown1 = nn.Conv2d(self.in_ch, 256, (1, 1))
        self.conv2ddown2 = nn.Conv2d(256, self.out_ch, (1, 1))
    def forward(self, x):
        out = self.conv2ddown1(x)
        out = self.conv2ddown2(out)
        return out


class vertical_up2d(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(vertical_up2d, self).__init__()
        self.in_ch = in_channel
        self.out_ch = out_channel
        self.conv2dup1 = nn.ConvTranspose2d(self.in_ch, 256, (1, 1))
        self.conv2dup2 = nn.ConvTranspose2d(256, self.out_ch, (1, 1))
    def forward(self, x):
        out = self.conv2dup1(x)
        out = self.conv2dup2(out)
        return out
