# NCM-MDGNN: A Mutli-Dimensional Graph Differential Neural Network, Towards the future of smart materials science 
<img width="416" height="197" alt="image" src="https://github.com/user-attachments/assets/89bf4b2b-b2ac-465b-abbf-d715ef92069e" />



## Overview
We present Mutli-Dimensional Graph Differential Neural Network (MDGDNN).The MDGDNN model consists of data preprocessing, icos multi-level spherical graph construction and differential graph neural operators. The design patterns and data flow changes of each link are shown in Figure 1. Specifically, the data preprocessing process involves the fusion of small sample data, the multi-ML ensemble algorithms, and the Kalman filter denoising algorithm to achieve precise sample enhancement and data preprocessing. Secondly, a vectorization method is employed to construct a multi-stage subdivided spherical structure from the 0th level icosphere spherical surface. Meanwhile, we construct a three-dimensional spherical structure network with different level in a nested structure. The graph relationship matrix (including the vertex matrix and edge matrix) is recorded based on the relationship between vertices and edges. The CSR sparse storage structure is used to achieve efficient and lightweight construction of large-scale graph structures. For the core part of the model, we designed a graph convolution kernel (GCN kernel) based on multi-order differential graph operator (GDO), and integrated it with graph convolution networks and residual networks to form Graph Differential Operator Unit (GDO Unit). Furthermore, we have designed a Multi-scale Graph Aggregation Algorithm, called GDO-Unet, which continuously expands the influence range of the operator in the space through a graph U-net structure, in order to simultaneously model both local and global physical and chemical spatial patterns.

> This repository provides the trained model weights (with data augmentation), 16 best-prediction demo cases, and a one-click inference script.

---

## Directory Structure

```
MDGNN_demo/
├── run_test.py                       # One-click inference entry point
├── model_def.py                      # MDGNN model definition
├── ops.py                            # MeshConv custom operators
├── utils.py                          # Sparse matrix utilities
├── requirements.txt                  # Python dependencies
├── README.md                         # Chinese version
├── README_EN.md                      # This file
├── mdgnn_state_dict_frozenBN.pt      # Model weights (with frozen BN statistics)
├── norm_params.npz                   # Normalization parameters (mean/std)
├── demo_features.npy                 # 16 demo sample inputs (20-dim)
├── demo_labels.npy                   # 16 demo sample labels (capacity, retention)
├── demo_samples.txt                  # Demo sample reference table
├── meshcnn/                          # Icosphere mesh files (levels 0–4)
├── compute_bn_stats.py               # [Utility] Recompute frozen BN statistics
├── regenerate_demo.py                # [Utility] Re-select best 16 prediction cases
├── test_full.py                      # [Utility] Full real-sample inference
└── .gitignore
```

---

## Environment Setup

### 1. Create Virtual Environment

```bash
conda create -n mdgnn python=3.10 -y
conda activate mdgnn
```

### 2. Install PyTorch

Choose the command matching your CUDA version (CPU-only is also supported). PyTorch >= 2.0 is recommended:

```bash
# CUDA 11.8
pip install torch torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# CUDA 12.1
pip install torch torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu121.html

# CPU only
pip install torch torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

> **Note**: `torch-sparse` must be installed via PyG's precompiled wheels — a plain `pip install torch-sparse` may fail. Refer to the [official PyG Installation Guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

### 3. Install Remaining Dependencies

```bash
pip install numpy
```

Or install all dependencies at once:

```bash
pip install -r requirements.txt
```

---

## One-Click Execution

```bash
cd MDGNN_demo
python run_test.py
```

### Expected Output

```
MDGNN Inference — 16 Best Prediction Cases
================================================================================
#    Element  Z    Conc.        True Cap   Pred Cap     Δcap       True Ret   Pred Ret     Δret
--------------------------------------------------------------------------------
1    V        23   0.0300       185.40     185.11     0.29       86.19      86.38     0.19
2    Na       11   0.0100       188.91     189.00     0.09       87.65      87.35     0.30
...
================================================================================
Summary (n=16)
  Capacity    MAE=0.66 mAh/g, RMSE=0.79 mAh/g, R²=0.99
  Retention   MAE=0.52 %,     RMSE=0.60 %,     R²=0.99
```

---

## Model Information

| Property          | Value                                             |
|-------------------|---------------------------------------------------|
| Architecture      | Multi_uunet (SphericalUNet + MeshConv)            |
| Parameters        | 32,739                                            |
| Input dimension   | 20 (element feature vector)                       |
| Output dimension  | 2 (discharge capacity in mAh/g, retention rate in %) |
| Mesh level        | max_level=4, icosphere with 2,562 vertices        |
| Feature dimension | fdim=2                                            |
| Training strategy | Data augmentation (synthetic oversampling)         |
| Normalization     | Z-score (pre-computed mean & standard deviation)   |
| BN statistics     | Frozen (no batch-size dependency during inference) |

---

## Input Format

The model accepts an `(N, 20)` float matrix, where the 20-dimensional feature encodes elemental properties (atomic number, concentration, etc.). See `demo_features.npy` for 16 example inputs.



# MDGNN Demo Chinese Version

基于 **NCM-MDGNN** 的锂离子电池正极材料掺杂改性的容量与保持率预测模型。

> 本仓库提供经数据增强训练后的模型权重、16 个最佳预测案例的演示数据，以及一键推理脚本。

---

## 目录结构

```
MDGNN_demo/
├── run_test.py                       # 一键推理入口
├── model_def.py                      # MDGNN 模型定义
├── ops.py                            # MeshConv 自定义算子
├── utils.py                          # 稀疏矩阵工具
├── requirements.txt                  # Python 依赖
├── README.md                         # 本文件
├── mdgnn_state_dict_frozenBN.pt      # 模型权重（固化BN统计量）
├── norm_params.npz                   # 归一化参数 (均值/标准差)
├── demo_features.npy                 # 16 个演示样本输入 (20维)
├── demo_labels.npy                   # 16 个演示样本标签 (容量, 保持率)
├── demo_samples.txt                  # 演示样本对照表
├── meshcnn/                          # Icosphere 网格文件 (0~4层)
├── compute_bn_stats.py               # [工具] 重新计算 BN 固化统计量
├── regenerate_demo.py                # [工具] 重新筛选 16 个最佳案例
├── test_full.py                      # [工具] 对全量真实样本推理
└── .gitignore
```

---

## 环境部署

### 1. 创建虚拟环境

```bash
conda create -n mdgnn python=3.10 -y
conda activate mdgnn
```

### 2. 安装 PyTorch

根据 CUDA 版本选择安装命令（CPU-only 也兼容），推荐 PyTorch >= 2.0：

```bash
# CUDA 11.8
pip install torch torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# CUDA 12.1
pip install torch torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu121.html

# CPU only
pip install torch torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

> **注意**：`torch-sparse` 必须通过 PyG 的预编译 wheel 安装，直接 `pip install torch-sparse` 可能失败。推荐 [PyG 官方安装指南](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)。

### 3. 安装其余依赖

```bash
pip install numpy
```

或一键安装：

```bash
pip install -r requirements.txt
```

---

## 一键执行

```bash
cd MDGNN_demo
python run_test.py
```

### 预期输出

```
MDGNN 推理结果 — 16 个最佳预测案例
================================================================================
#    元素    Z    浓度             容量真实       容量预测     Δcap       保持率真实      保持率预测     Δret
--------------------------------------------------------------------------------
1    V     23   0.0300       185.40     185.11     0.29       86.19      86.38     0.19
2    Na    11   0.0100       188.91     189.00     0.09       87.65      87.35     0.30
...
================================================================================
汇总 (n=16)
  容量   MAE=0.66 mAh/g, RMSE=0.79 mAh/g, R²=0.99
  保持率 MAE=0.52 %,     RMSE=0.60 %,     R²=0.99
```

---

## 模型信息

| 属性       | 值                                                 |
|-----------|---------------------------------------------------|
| 架构       | Multi\_uunet (SphericalUNet + MeshConv)            |
| 参数数量    | 32,739                                            |
| 输入维度    | 20 (元素特征向量)                                    |
| 输出维度    | 2 (放电容量 mAh/g, 容量保持率 %)                       |
| 网格层级    | max\_level=4, icosphere 2,562 顶点                 |
| 特征维度    | fdim=2                                            |
| 训练策略    | 数据增强 (synthetic oversampling)                    |
| 归一化      | Z-score (预计算的均值/标准差)                          |
| BN 统计量   | 固化版本 (推理时无批大小依赖)                            |

---

## 输入格式说明

模型接受一个 `(N, 20)` 的浮点矩阵，其中 20 维特征由元素属性编码（原子序数、浓度等）。`demo_features.npy` 提供了 16 个示例。

---



