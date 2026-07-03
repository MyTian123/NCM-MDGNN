"""计算 BN_1_3 固化统计量：使用全量训练样本预计算 mean/std 并注册为 buffer"""
import os, sys, numpy as np, torch

DEPLOY = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DEPLOY)
from model_def import Multi_uunet
data_all  = np.load(os.path.join(DEPLOY, '..', 'train_real.npy'))
label_all = np.load(os.path.join(DEPLOY, '..', 'label_real.npy'))

fin  = torch.tensor(data_all, dtype=torch.float32)
fout = torch.tensor(label_all, dtype=torch.float32)
mX = fin.mean(0); sX = fin.std(0) + 1e-8
mY = fout.mean(0); sY = fout.std(0) + 1e-8

# 创建模型（加载后模型自带 buffer 初始化为0）
model = Multi_uunet(1, os.path.join(DEPLOY, 'meshcnn'), 20, 2,
                     max_level=4, min_level=0, fdim=2)
model.load_state_dict(torch.load(os.path.join(DEPLOY, 'mdgnn_state_dict_frozenBN.pt'),
                                 map_location='cpu', weights_only=True))
model.eval()

# 收集4次BN_1_3的统计量
stats = [[], [], [], []]
def capture_bn(x, idx):
    pre_x = x.permute(1, 0, 2)
    pre_x_flat = torch.flatten(pre_x, 1, -1)
    mean = pre_x_flat.mean(dim=1)
    std  = pre_x_flat.std(dim=1)
    stats[idx].append((mean.detach().clone(), std.detach().clone()))
    return (x - mean.unsqueeze(-1).unsqueeze(0)) / (std.unsqueeze(-1).unsqueeze(0) + 1e-5)

def traced_forward(x):
    x = x.unsqueeze(-1)
    x = model.Lmodel_res(x);       x = capture_bn(x, 0)
    x = model.relu(x);             x = x.permute(0, 2, 1)
    x = model.Lmodel2(x);          x = capture_bn(x, 1)
    x = model.relu(x);             x = x.permute(0, 2, 1)
    x = model.spheric_aluNet2(x);  x = capture_bn(x, 2)
    x = model.relu(x)
    x = model.Lmodel_north(x);     x = capture_bn(x, 3)
    x = model.relu(x);             x = x.squeeze(-1)
    x = model.Lmodel1(x)
    return x

with torch.no_grad():
    _ = traced_forward((fin - mX) / sX)

# 注册为 buffer 并保存
for i in range(4):
    mean = stats[i][0][0]
    std  = stats[i][0][1]
    model.register_buffer(f'bn{i+1}_mean', mean)
    model.register_buffer(f'bn{i+1}_std',  std)
    print(f'bn{i+1}: shape={mean.shape}, mean=[{mean.min():.6f}, {mean.max():.6f}], std=[{std.min():.6f}, {std.max():.6f}]')

new_path = os.path.join(DEPLOY, 'mdgnn_state_dict_frozenBN.pt')
torch.save(model.state_dict(), new_path)
print(f'\n已保存: {new_path}')
print(f'文件大小: {os.path.getsize(new_path)/1024:.1f} KB')
