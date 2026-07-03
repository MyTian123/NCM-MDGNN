"""
MDGNN 演示模型 — 一键推理测试
===============================
用法: python run_test.py

此脚本加载 MDGNN 模型（经数据增强训练）对 16 个最佳预测案例执行推理，
并输出逐样本预测 vs 真实值对比。
"""
import os
import sys
import numpy as np
import torch

DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DEPLOY_DIR)

from model_def import Multi_uunet

ELEM_MAP = {
    5: 'B', 11: 'Na', 12: 'Mg', 13: 'Al', 19: 'K', 20: 'Ca',
    22: 'Ti', 23: 'V', 26: 'Fe', 29: 'Cu', 30: 'Zn', 32: 'Ge',
    39: 'Y', 40: 'Zr', 41: 'Nb', 42: 'Mo', 44: 'Ru', 50: 'Sn',
    51: 'Sb', 57: 'La', 72: 'Hf', 73: 'Ta', 74: 'W', 75: 'Re',
}


def main():
    # 加载模型（固化BN版：BN_1_3 统计量已预计算，批大小无关）
    model = Multi_uunet(1, os.path.join(DEPLOY_DIR, 'meshcnn'),
                        20, 2, max_level=4, min_level=0, fdim=2)
    model.load_state_dict(torch.load(os.path.join(DEPLOY_DIR, 'mdgnn_state_dict_frozenBN.pt'),
                                     map_location='cpu', weights_only=True))
    model.eval()

    # 加载归一化参数（基于全量训练样本计算）
    params = np.load(os.path.join(DEPLOY_DIR, 'norm_params.npz'))
    mX = torch.tensor(params['mX'], dtype=torch.float32)
    sX = torch.tensor(params['sX'], dtype=torch.float32)
    mY = torch.tensor(params['mY'], dtype=torch.float32)
    sY = torch.tensor(params['sY'], dtype=torch.float32)

    # 加载 16 个演示样本
    features = np.load(os.path.join(DEPLOY_DIR, 'demo_features.npy'))
    labels   = np.load(os.path.join(DEPLOY_DIR, 'demo_labels.npy'))
    n = features.shape[0]

    # 推理
    x = torch.tensor(features, dtype=torch.float32)
    with torch.no_grad():
        y_pred = model((x - mX) / sX) * sY + mY
    y_pred = y_pred.numpy()

    # 输出
    print(f'MDGNN 推理结果 — {n} 个最佳预测案例')
    print(f'{"=" * 80}')
    print(f'{"#":<4} {"元素":<5} {"Z":<4} {"浓度":<8} '
          f'{"容量真实":>10} {"容量预测":>10} {"Δcap":>8}  '
          f'{"保持率真实":>10} {"保持率预测":>10} {"Δret":>8}')
    print(f'{"-" * 80}')

    cap_errs, ret_errs = [], []
    for i in range(n):
        z = int(features[i, 0])
        c = features[i, 1]
        el = ELEM_MAP.get(z, f'Z{z}')
        tc, tr = labels[i, 0], labels[i, 1]
        pc, pr = y_pred[i, 0], y_pred[i, 1]
        dc, dr = abs(pc - tc), abs(pr - tr)
        cap_errs.append(dc)
        ret_errs.append(dr)
        print(f'{i+1:<4} {el:<5} {z:<4} {c:<8.4f} '
              f'{tc:>10.2f} {pc:>10.2f} {dc:>8.2f}  '
              f'{tr:>10.2f} {pr:>10.2f} {dr:>8.2f}')

    cap_rmse = np.sqrt(np.mean(np.array(cap_errs)**2))
    ret_rmse = np.sqrt(np.mean(np.array(ret_errs)**2))
    cap_mae  = np.mean(cap_errs)
    ret_mae  = np.mean(ret_errs)
    ss_c = ((labels[:,0] - labels[:,0].mean())**2).sum()
    ss_r = ((labels[:,1] - labels[:,1].mean())**2).sum()
    r2_c = 1 - ((y_pred[:,0] - labels[:,0])**2).sum() / ss_c
    r2_r = 1 - ((y_pred[:,1] - labels[:,1])**2).sum() / ss_r
    print(f'{"=" * 80}')
    print(f'汇总 (n={n})')
    print(f'  容量   MAE={cap_mae:.4f} mAh/g, RMSE={cap_rmse:.4f} mAh/g, R²={r2_c:.4f}')
    print(f'  保持率 MAE={ret_mae:.4f} %,     RMSE={ret_rmse:.4f} %,     R²={r2_r:.4f}')


if __name__ == '__main__':
    main()
