from torch import nn
from ops import MeshConv, MeshConv_transpose, ResBlock
import torch.nn.functional as F
import os
import torch
import pickle
import numpy as np
from GCNmodel import GraphGNN, GCNLayer
from torch_geometric.nn import GCNConv
from utils import SPmm
from EncoderDecoder import Grid2Mesh_Encoder,Mesh2Grid_Decoder, Grid2Mesh
import os
import time
from Unet import UNet
# from DDDUnet import UNet3d
from AttUnet import attUNet

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
        #     self.down.append(Down(fdim*(2**idx), fdim*(2**(idx+1)), max_level-i-1, mesh_folder))
        # self.down.append(Down(fdim*(2**(self.levels-1)), fdim*(2**(self.levels-1)), min_level, mesh_folder))
        # Upward path
        for i in range(self.levels-1):
            self.up.append(Up(fdim*(2**(self.levels-i)), fdim*(2**(self.levels-i-2)), min_level+i+1, mesh_folder))
        self.up.append(Up(fdim*2, fdim, max_level, mesh_folder))
        #     self.up.append(Up(fdim*(2**(self.levels-i)), fdim*(2**(self.levels-i-2)), min_level+i+1, mesh_folder))
        # self.up.append(Up(fdim*2, fdim, max_level, mesh_folder))
        self.down = nn.ModuleList(self.down)
        self.up = nn.ModuleList(self.up)

    def forward(self, x):

        #x初始是[bs, 16, 10242]
        x_ = [self.in_conv(x)] #[bs, 185, 10242]

        for i in range(self.levels): #不同程度的icos循环down
            x_.append(self.down[i](x_[-1]))

        x = self.up[0](x_[-1], x_[-2]) #整体一起up #[bs, 64, 42]

        for i in range(self.levels-1): # -3-i不是很能理解
            x = self.up[i+1](x, x_[-3-i]) #([bs, 8, 10242])等，第二三个数字变化

        x = self.out_conv(x) #[bs, 3, 10242]

        return x 

    def __meshfile(self, i):
        return os.path.join(self.mesh_folder, "icosphere_{}.pkl".format(i))

class Multi_uunet(nn.Module):
    # fl=args.forecast_len, mesh_folder=args.mesh_folder, in_ch=5, out_ch=5, 
    #     max_level=args.max_level, min_level=args.min_level, fdim=args.feat
    def __init__(self, bs, mesh_folder, in_ch, out_ch, max_level, min_level, fdim=48): #12 42 162 642 2562 10242 40962
        super(Multi_uunet, self).__init__()
        self.mesh_folder = mesh_folder
        self.fdim = fdim
        self.max_level = max_level
        self.min_level = min_level
        self.bs = bs
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.relu = nn.GELU()
        self.hidden_ = 32
        self.vertex = 10*4**max_level+2
        # out_ch = 64
        self.spheric_aluNet2 = SphericalUNet(self.mesh_folder, self.hidden_, out_ch, self.max_level, self.min_level, self.fdim)
        # self.Lmodel1 = LinearRegressionModel(in_ch*plevel*2, self.uu_level0)
        self.Lmodel1 = LinearRegressionModel(out_ch,out_ch)
        self.Lmodel2 = AdvancedMLP(in_ch, self.hidden_)  #*2是为了残差

        self.Lmodel_north = AdvancedMLP(self.vertex, 1) # LinearRegressionModel(self.vertex, 1)
        self.Lmodel_res = AdvancedMLP(1, self.vertex) # LinearRegressionModel(1, self.vertex)
        self.drop = 0
        self.dropout = nn.Dropout(0.2) #nn.functional.dropout(0.6)

    def forward(self,x):
        x = x.unsqueeze(-1)
        # x = x.permute(0,2,1)
        x = self.Lmodel_res(x)
        x = self.BN_1_3(x)
        x = self.relu(x)
        x = x.permute(0,2,1) #bs,vertex,channels
        # print(x.shape)
        x = self.Lmodel2(x)
        x = self.BN_1_3(x)
        x = self.relu(x)
        x = x.permute(0,2,1) #[bs,C,Vertex]
        # print(x.shape)
        x = self.spheric_aluNet2(x) #[bs,C,Vertex]
        x = self.BN_1_3(x)
        x = self.relu(x) #[bs,C,Vertex]
        # x = torch.cat((x,x_res),1) #bs,64,40962
        # x = x.permute(0,2,1)
        x = self.Lmodel_north(x)
        x = self.BN_1_3(x)
        x = self.relu(x)
        x = x.squeeze(-1)
        # # x = x.permute(0,2,1) #bs,channels
        x = self.Lmodel1(x)

        return x
    
    def BN_1_4(self,x):
        pre_x = x.permute(1,0,2,3)
        pre_x = torch.flatten(pre_x,1,-1)
        mean = pre_x.mean(dim=1)
        std = pre_x.std(dim=1)
        x = (x-mean.unsqueeze(-1).unsqueeze(-1).unsqueeze(0))/(std.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)+1e-5)
        return x
    
    def BN_1_3(self,x):
        pre_x = x.permute(1,0,2)
        pre_x = torch.flatten(pre_x,1,-1)
        mean = pre_x.mean(dim=1)
        std = pre_x.std(dim=1)
        x = (x-mean.unsqueeze(-1).unsqueeze(0))/(std.unsqueeze(-1).unsqueeze(0)+1e-5)
        return x

class LinearRegressionModel(nn.Module):
    def __init__(self,shape,out,hs=64):
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
        # out = self.linear(x)
        # out = self.linear1(out)

        return out
    
class LinearRegressionModel2(nn.Module):
    def __init__(self,shape,out,hs=32):
        super(LinearRegressionModel2, self).__init__()
        self.linout = nn.Sequential(
            nn.Linear(shape, hs),
            nn.ReLU(inplace=True),
            nn.Linear(hs, hs),
            nn.ReLU(inplace=True),
            nn.Linear(hs, out),
        )

    def forward(self, x):
        out = self.linout(x) 
        # out = self.linear(x)
        # out = self.linear1(out)

        return out

class vertical_down(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(vertical_down, self).__init__()
        self.in_ch = in_channel
        self.out_ch = out_channel
        self.conv1ddown1 = nn.Conv1d(self.in_ch, 256, 1, stride=1) # (input_channels, output_channels, kernel_size, stride)
        self.conv1ddown2 = nn.Conv1d(256, self.out_ch, 1, stride=1)
    def forward(self, x): #x (batch_size,channels,length)
        x = self.conv1ddown1(x)
        x = self.conv1ddown2(x)
        return x

class vertical_up(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(vertical_up, self).__init__()
        self.in_ch = in_channel
        self.out_ch = out_channel
        self.conv1dup1 = nn.ConvTranspose1d(self.in_ch, 256, 1, stride=1) # (input_channels, output_channels, kernel_size, stride)
        self.conv1dup2 = nn.ConvTranspose1d(256, self.out_ch, 1, stride=1)
    def forward(self, x): #x (batch_size,channels,length)
        x = self.conv1dup1(x)
        x = self.conv1dup2(x)
        return x

class vertical_down3d(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(vertical_down3d, self).__init__()
        self.in_ch = in_channel
        self.out_ch = out_channel
        # self.kernel_size1 = kernel_1
        # self.kernel_size2 = kernel_2
        self.conv2ddown1 = nn.Conv2d(self.in_ch, 256, (1, 1)) # Conv2d[ channels, output, height_2, width_2 ]
        self.conv2ddown2 = nn.Conv2d(256, self.out_ch, (1, 1))#只能用于37层的大气层
    def forward(self, x): #x (batch_size,channels,length)
        out = self.conv2ddown1(x)
        out = self.conv2ddown2(out)
        return out

class vertical_up2d(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(vertical_up2d, self).__init__()
        self.in_ch = in_channel
        self.out_ch = out_channel
        # self.kernel_size1 = kernel_1
        # self.kernel_size2 = kernel_2
        self.conv2dup1 = nn.ConvTranspose2d(self.in_ch, 256, (1, 1)) # Conv2d[ channels, output, height_2, width_2 ]
        self.conv2dup2 = nn.ConvTranspose2d(256, self.out_ch, (1, 1))
    def forward(self, x): #x (batch_size,channels,length)
        out = self.conv2dup1(x)
        out = self.conv2dup2(out)
        return out
    
class AdvancedMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AdvancedMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim)
        )
        
    def forward(self, x):
        return self.network(x)