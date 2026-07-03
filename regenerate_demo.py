"""重新生成 demo 数据：使用全量样本推理，筛选16个预测误差最小的案例"""
import os, sys, numpy as np, torch
DEPLOY = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DEPLOY)
from model_def import Multi_uunet

EM = {5:'B',11:'Na',12:'Mg',13:'Al',19:'K',20:'Ca',22:'Ti',23:'V',
      26:'Fe',29:'Cu',30:'Zn',32:'Ge',39:'Y',40:'Zr',41:'Nb',42:'Mo',
      44:'Ru',50:'Sn',51:'Sb',57:'La',72:'Hf',73:'Ta',74:'W',75:'Re'}

# 加载数据
data_all  = np.load(os.path.join(DEPLOY, '..', 'train_real.npy'))
label_all = np.load(os.path.join(DEPLOY, '..', 'label_real.npy'))

# 加载固化BN模型
model = Multi_uunet(1, os.path.join(DEPLOY, 'meshcnn'), 20, 2,
                     max_level=4, min_level=0, fdim=2)
model.load_state_dict(torch.load(os.path.join(DEPLOY, 'mdgnn_state_dict_frozenBN.pt'),
                                 map_location='cpu', weights_only=True))
model.eval()

# 归一化
fin  = torch.tensor(data_all, dtype=torch.float32)
fout = torch.tensor(label_all, dtype=torch.float32)
mX = fin.mean(0); sX = fin.std(0) + 1e-8
mY = fout.mean(0); sY = fout.std(0) + 1e-8

# 推理
with torch.no_grad():
    pred = model((fin - mX) / sX) * sY + mY
pred = pred.numpy()

# 计算综合误差
cap_errs = np.abs(pred[:,0] - label_all[:,0])
ret_errs = np.abs(pred[:,1] - label_all[:,1])
combined_err = np.sqrt((cap_errs / label_all[:,0].std())**2 + (ret_errs / label_all[:,1].std())**2)

# 选16个最佳
best_idx = np.argsort(combined_err)[:16]
print(f'16个最佳样本索引: {best_idx.tolist()}')

# 保存
demo_f = data_all[best_idx]
demo_l = label_all[best_idx]
np.save(os.path.join(DEPLOY, 'demo_features.npy'), demo_f)
np.save(os.path.join(DEPLOY, 'demo_labels.npy'), demo_l)

# 写 sample info
with open(os.path.join(DEPLOY, 'demo_samples.txt'), 'w') as f:
    f.write(f'#    {"元素":<5} {"Z":<4} {"浓度":<8} {"容量真实":>8} {"容量预测":>8} {"保持率真实":>8} {"保持率预测":>8} {"综合误差":>8}\n')
    for rank, idx in enumerate(best_idx):
        z = int(data_all[idx,0]); c = data_all[idx,1]
        el = EM.get(z, f'Z{z}')
        tc, tr = label_all[idx,0], label_all[idx,1]
        pc, pr = pred[idx,0], pred[idx,1]
        f.write(f'{rank+1:<4} {el:<5} {z:<4} {c:<8.4f} {tc:>8.2f} {pc:>8.2f} {tr:>8.2f} {pr:>8.2f} {combined_err[idx]:>8.4f}\n')

    # 汇总
    demo_pred = pred[best_idx]
    demo_lbl = label_all[best_idx]
    c_rmse = np.sqrt(np.mean((demo_pred[:,0] - demo_lbl[:,0])**2))
    r_rmse = np.sqrt(np.mean((demo_pred[:,1] - demo_lbl[:,1])**2))
    f.write(f'\n16样本汇总: cap RMSE={c_rmse:.4f}, ret RMSE={r_rmse:.4f}\n')

print(f'已保存: demo_features.npy ({demo_f.shape}), demo_labels.npy ({demo_l.shape})')
print(f'16样本汇总: cap RMSE={c_rmse:.4f}, ret RMSE={r_rmse:.4f}')
