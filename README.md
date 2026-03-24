# UAV Detection YOLO11

<div align="center">

**面向低空经济的复杂场景空中障碍物实时检测算法**

基于 YOLO11 的轻量化无人机障碍物检测模型

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## 项目简介

本项目聚焦低空经济中无人机（UAV）飞行的核心安全问题，开发了一种**轻量化、实时**的空中障碍物检测算法。基于 YOLO11s 模型，通过多源数据融合训练与网络结构改进，实现对复杂场景下动态与静态障碍物的快速准确识别。

### 核心特点

- **多源数据融合**：整合 UAV1、UAV2、Airborne、AOD4 等公开数据集，缓解小样本训练不稳定问题
- **轻量化设计**：参数量仅增加 19%，适合部署于嵌入式或移动平台
- **实时检测**：推理速度达 80+ FPS，满足无人机自主避障实时性要求
- **高精度**：mAP@50 从基线 0.25 提升至 0.875+

---

## 模型架构

<div align="center">
<img src="https://github.com/Future-Elite/uav-detection-yolo11/raw/main/docs/model_architecture.png" width="800">
</div>

### 主要改进模块

| 模块 | 功能描述 |
|------|----------|
| **CSPPC** | 跨阶段部分连接，增强特征融合能力 |
| **ECA** | 高效通道注意力机制，提升特征表达能力 |
| **SPPELAN** | 空间金字塔池化扩展层，多尺度特征融合 |
| **Enhanced P3** | P3 层小目标特征增强（DWConv + 空洞卷积 + ECA） |

---

## 性能对比

### mAP@50 渐进提升

<div align="center">
<img width="1403" height="1175" alt="performance_comparison" src="https://github.com/user-attachments/assets/3f027446-0498-40c9-bcc6-4e3555950f26" />
</div>

| 配置 | mAP@50 | 相对基线提升 |
|------|--------|--------------|
| YOLO11s (基线) | 0.804 | - |
| +SIoU | 0.821 | +2% |
| +CSPPC | 0.855 | +6% |
| +ECA | 0.857 | +7% |
| +EnhancedP3 | 0.863 | +7% |
| **完整改进 (最佳)** | **0.875** | **+9%** |

### 多数据集性能

<div align="center">
<img src="https://github.com/Future-Elite/uav-detection-yolo11/raw/main/docs/performance_detailed.png" width="800">
</div>

| 数据集 | 基线 mAP@50 | 改进模型 | 提升幅度 |
|--------|-------------|----------|----------|
| AOD4 | 0.25 | **0.88** | +252% |
| Anti2 | 0.15 | **0.47** | +213% |
| UAV1 | 0.35 | **0.82** | +134% |
| UAV2 | 0.30 | **0.78** | +160% |

### 各类别检测性能

| 类别 | 基线 AP@50 | 改进模型 | 提升 |
|------|------------|----------|------|
| plane | 0.32 | **0.91** | +0.59 |
| bird | 0.18 | **0.76** | +0.58 |
| drone | 0.28 | **0.89** | +0.61 |
| helicopter | 0.22 | **0.91** | +0.69 |

### 速度-精度权衡

| 模型 | FPS | mAP@50 |
|------|-----|--------|
| YOLO11s (基线) | 85 | 0.26 |
| YOLO11m | 145 | 0.65 |
| YOLO11l | 35 | 0.90 |
| **YOLO11s +改进** | **80** | **0.88** |

> 改进模型在几乎不牺牲速度的前提下，将精度从 0.26 提升至 0.88，是**最优工作点**。

---

## 项目结构

```
uav-detection-yolo11/
├── train.py                 # 模型训练脚本
├── predict.py               # 模型预测与验证脚本
├── enhance.py               # 数据增强脚本
├── merge_datasets.py        # 多源数据集合并
├── vis.py                   # 可视化工具
├── configs/                 # 训练配置文件
│   └── merged-config.yaml
├── refined-models/          # 改进模型架构定义
│   ├── yolo11-CSPPC-ECA-SPPELAN.yaml
│   ├── yolo11-CSPPC-ECA-enhancedP3.yaml
│   └── ...
└── ultralytics/             # Ultralytics YOLO 框架
```

---

## 快速开始

### 环境配置

```bash
# 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate  # Windows

# 安装依赖
pip install torch torchvision
pip install ultralytics
pip install opencv-python numpy pandas matplotlib
```

### 训练模型

```bash
python train.py
```

### 模型验证

```bash
python predict.py
```

---

## 数据集

### 支持的目标类别

| 类别 | 中文 | 描述 |
|------|------|------|
| plane | 飞机 | 固定翼飞行器 |
| bird | 鸟类 | 飞鸟等自然障碍物 |
| drone | 无人机 | 多旋翼/固定翼无人机 |
| helicopter | 直升机 | 旋翼飞行器 |

### 数据来源

- **UAV1/UAV2**: 无人机检测数据集
- **Airborne**: 空中目标数据集
- **AOD4**: 空中障碍物检测测试集
- **Anti2**: 困难测试集（极小目标、恶劣光照）

---

## 技术亮点

### 数据策略

多源数据融合有效缓解了小样本训练不稳定问题，解决了模型"能不能学"的问题。

### 结构改进

ECA + SPPELAN + Enhanced P3 协同作用显著提升检测性能，解决了模型"能不能学得更好"的问题。

### 消融实验结论

- **P2 层问题**: 在有限数据集下，P2 层会增加噪声正样本比例，导致回归不稳定
- **最优方案**: 通过增强 P3 特征表达与解耦头设计来提升小目标检测性能

---

## 引用

如果本项目对你的研究有帮助，欢迎引用：

```bibtex
@misc{uav-detection-yolo11,
  title={UAV Detection YOLO11: Lightweight Aerial Obstacle Detection},
  author={Future-Elite},
  year={2024},
  howpublished={\url{https://github.com/Future-Elite/uav-detection-yolo11}}
}
```

---

## License

MIT License

---

<div align="center">

**Star** ⭐ 本项目以支持开发！

</div>
