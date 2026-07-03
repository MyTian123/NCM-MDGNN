"""对全量真实样本执行推理，输出逐样本预测 vs 真实值对比"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from model_def import Multi_uunet

ELEM_MAP = {5:'B',11:'Na',12:'Mg',13:'Al',19:'K',20:'Ca',22:'Ti',23:'V',
    26:'Fe',29:'Cu',30:'Zn',32:'Ge',39:'Y',40:'Zr',41:'Nb',42:'Mo',
    44:'Ru',50:'Sn',51:'Sb',57:'La',72:'Hf',73:'Ta',74:'W',75:'Re'}

DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))

# 加载全量真实样本
data  = np.load(os.path.join(DEPLOY_DIR, '..', 'train_real.npy'))
label = np.load(os.path.join(DEPLOY_DIR, '..', 'label_real.npy'))
n = data.shape[0]

# 加载模型
model = Multi_uunet(1, os.path.join(DEPLOY_DIR, 'meshcnn'), 20, 2, max_level=4, min_level=0, fdim=2)
model.load_state_dict(torch.load(os.path.join(DEPLOY_DIR, 'mdgnn_state_dict_frozenBN.pt'), map_location='cpu', weights_only=True))
model.eval()

# 归一化
fin  = torch.tensor(data, dtype=torch.float32)
fout = torch.tensor(label, dtype=torch.float32)
mX, sX = fin.mean(0), fin.std(0) + 1e-8
mY, sY = fout.mean(0), fout.std(0) + 1e-8

# 推理
with torch.no_grad():
    pred = model((fin - mX) / sX) * sY + mY
pred = pred.numpy()

# 计算误差
cap_rmse = np.sqrt(np.mean((pred[:,0] - label[:,0])**2))
ret_rmse = np.sqrt(np.mean((pred[:,1] - label[:,1])**2))
cap_mae  = np.mean(np.abs(pred[:,0] - label[:,0]))
ret_mae  = np.mean(np.abs(pred[:,1] - label[:,1]))
ss_c = ((label[:,0] - label[:,0].mean())**2).sum()
ss_r = ((label[:,1] - label[:,1].mean())**2).sum()
r2_c = 1 - ((pred[:,0] - label[:,0])**2).sum() / ss_c
r2_r = 1 - ((pred[:,1] - label[:,1])**2).sum() / ss_r

print(f'===== MDGNN 全量真实样本推理 (n={n}) =====')
print(f'容量   RMSE={cap_rmse:.4f} mAh/g   R²={r2_c:.4f}   MAE={cap_mae:.4f} mAh/g')
print(f'保持率 RMSE={ret_rmse:.4f} %        R²={r2_r:.4f}   MAE={ret_mae:.4f} %')
print()
print(f'{"#":<4} {"元素":<5} {"Z":<4} {"浓度":<8} {"容量真实":>8} {"容量预测":>8} {"Δcap":>8}  {"保持率真实":>8} {"保持率预测":>8} {"Δret":>8}')
print('-' * 90)

for i in range(n):
    z = int(data[i,0]); c = data[i,1]
    el = ELEM_MAP.get(z, f'Z{z}')
    tc, tr = label[i,0], label[i,1]
    pc, pr = pred[i,0], pred[i,1]
    dc, dr = abs(pc-tc), abs(pr-tr)
    print(f'{i+1:<4} {el:<5} {z:<4} {c:<8.4f} {tc:>8.2f} {pc:>8.2f} {dc:>8.3f}  {tr:>8.2f} {pr:>8.2f} {dr:>8.3f}')

cap_errs = np.abs(pred[:,0] - label[:,0])
ret_errs = np.abs(pred[:,1] - label[:,1])

# 找出误差最小的16个样本
combined_err = np.sqrt((cap_errs / label[:,0].std())**2 + (ret_errs / label[:,1].std())**2)
best_idx = np.argsort(combined_err)[:16]

print()
print(f'--- 综合误差最小的16个样本 ---')
print(f'{"#":<4} {"元素":<5} {"Z":<4} {"浓度":<8} {"容量真实":>8} {"容量预测":>8} {"Δcap":>8}  {"保持率真实":>8} {"保持率预测":>8} {"Δret":>8} {"综合误差":>8}')
for idx in best_idx:
    z = int(data[idx,0]); c = data[idx,1]
    el = ELEM_MAP.get(z, f'Z{z}')
    tc, tr = label[idx,0], label[idx,1]
    pc, pr = pred[idx,0], pred[idx,1]
    dc, dr = abs(pc-tc), abs(pr-tr)
    print(f'{idx+1:<4} {el:<5} {z:<4} {c:<8.4f} {tc:>8.2f} {pc:>8.2f} {dc:>8.3f}  {tr:>8.2f} {pr:>8.2f} {dr:>8.3f} {combined_err[idx]:>8.4f}')

print()
print(f'容量误差范围:   {cap_errs.min():.4f} ~ {cap_errs.max():.4f} mAh/g')
print(f'保持率误差范围:  {ret_errs.min():.4f} ~ {ret_errs.max():.4f} %')
