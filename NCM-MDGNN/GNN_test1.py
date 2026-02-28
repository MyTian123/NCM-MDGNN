import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import numpy as np

class MolecularGNN(nn.Module):
    def __init__(self, num_node_types=10, hidden_dim=64, output_dim=15):
        super(MolecularGNN, self).__init__()
        
        # 节点嵌入层（基于节点类型或度）
        self.node_embedding = nn.Embedding(num_node_types, hidden_dim)
        
        # GNN层
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        # 掺杂浓度处理
        self.conc_embedding = nn.Linear(1, hidden_dim)
        
        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x, edge_index, batch, concentration):
        # x: 节点特征（可以是节点类型、度等）
        # edge_index: 图的边结构
        # batch: 图的batch索引
        # concentration: 掺杂浓度
        
        # 节点嵌入
        x = self.node_embedding(x)
        
        # GNN消息传递
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        
        # 图级池化
        graph_embedding = global_mean_pool(x, batch)
        
        # 浓度嵌入
        conc_embedding = self.conc_embedding(concentration.unsqueeze(1))
        
        # 合并图嵌入和浓度信息
        combined = torch.cat([graph_embedding, conc_embedding], dim=1)
        
        # 预测15种属性
        output = self.fc(combined)
        
        return output