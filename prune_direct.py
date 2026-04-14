"""
模型剪枝脚本 - 对已训练好的模型进行剪枝

策略说明：
由于YOLO框架限制，直接修改模型结构进行剪枝较为复杂。
本脚本采用"软剪枝"策略：
1. 分析BN层gamma参数，识别不重要的通道
2. 通过稀疏训练（L1正则化）让这些通道的gamma趋近于0
3. 微调恢复精度
4. 后续可通过导出ONNX并移除零通道实现真正的压缩

使用方法：
    python prune_direct.py
"""

import warnings
from pathlib import Path
import json

import torch
import torch.nn as nn
import numpy as np

from ultralytics import YOLO
from ultralytics.utils import LOGGER

warnings.filterwarnings("ignore")


# ==================== 配置区域 ====================
# 模型路径 - mAP@50 0.929 的最佳模型
MODEL_PATH = "./runs/detect/merged/refined-enhanced3/weights/best.pt"

# 数据集路径
DATA_PATH = "../datasets/merged_dataset_5_enhanced/data.yaml"
VAL_DATA_PATH = "../datasets/Airborne/data.yaml"

# 剪枝配置
PRUNE_RATIO = 0.3          # 目标剪枝比例

# 稀疏训练配置
SPARSE_EPOCHS = 1         # 稀疏训练轮次
SPARSE_LR = 0.0001         # 稀疏训练学习率
L1_LAMBDA = 0.001          # L1正则化系数

# 微调配置
FINETUNE_EPOCHS = 1       # 微调轮次
FINETUNE_LR = 0.00005      # 微调学习率

# 通用配置
BATCH_SIZE = 8
IMG_SIZE = 640
DEVICE = 0

# 输出配置
PROJECT = "runs/detect/prune"
NAME = "pruned_soft"
# =================================================


def gather_bn_info(model):
    """收集所有BN层信息"""
    bn_info = {}
    
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            gamma = module.weight.data.abs().cpu().numpy()
            bn_info[name] = {
                'num_channels': len(gamma),
                'gamma_mean': float(gamma.mean()),
                'gamma_std': float(gamma.std()),
                'gamma_min': float(gamma.min()),
                'gamma_max': float(gamma.max()),
                'near_zero_count': int(np.sum(gamma < 0.01)),
            }
    
    return bn_info


def gather_gamma_values(model):
    """收集所有gamma值"""
    gamma_list = []
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            gamma_list.append(module.weight.data.abs().cpu().numpy())
    return np.concatenate(gamma_list) if gamma_list else np.array([])


def count_parameters(model):
    """计算参数量"""
    return sum(p.numel() for p in model.parameters())


def plot_gamma_distribution(gamma_values, save_path, threshold=None, title=""):
    """绘制gamma分布图"""
    try:
        import matplotlib.pyplot as plt
        
        # 检查空数组
        if len(gamma_values) == 0:
            LOGGER.warning("gamma_values为空，跳过绘图")
            return
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 左图：分布直方图
        axes[0].hist(gamma_values, bins=100, alpha=0.7, color='blue', edgecolor='black')
        if threshold is not None:
            axes[0].axvline(x=threshold, color='red', linestyle='--', linewidth=2, 
                           label=f'阈值={threshold:.4f}')
        axes[0].set_xlabel('|Gamma|', fontsize=12)
        axes[0].set_ylabel('数量', fontsize=12)
        axes[0].set_title(f'{title} - Gamma分布', fontsize=14)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 右图：统计信息
        near_zero = np.sum(gamma_values < 0.01)
        very_small = np.sum(gamma_values < 0.001)
        total = len(gamma_values)
        
        stats_text = f"""统计信息:
总通道数: {total}
接近零 (<0.01): {near_zero} ({near_zero/total:.1%})
极小值 (<0.001): {very_small} ({very_small/total:.1%})
均值: {gamma_values.mean():.4f}
标准差: {gamma_values.std():.4f}
最小值: {gamma_values.min():.6f}
最大值: {gamma_values.max():.4f}"""
        
        axes[1].text(0.1, 0.5, stats_text, fontsize=12, transform=axes[1].transAxes,
                    verticalalignment='center', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1].set_title('统计信息', fontsize=14)
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        LOGGER.info(f"分布图保存至: {save_path}")
    except ImportError:
        LOGGER.warning("matplotlib未安装，跳过绘图")


def compute_prune_threshold(gamma_values, prune_ratio):
    """计算剪枝阈值"""
    if len(gamma_values) == 0:
        return 1e-5
    sorted_gamma = np.sort(gamma_values)
    idx = int(len(sorted_gamma) * prune_ratio)
    threshold = sorted_gamma[idx] if idx < len(sorted_gamma) else sorted_gamma[-1]
    return max(threshold, 1e-5)


def add_l1_regularization(model, lambda_l1):
    """
    为BN层添加L1正则化梯度
    返回一个回调函数，在训练中使用
    """
    def on_train_batch_end(trainer):
        """在backward后、optimizer.step前修改梯度"""
        with torch.no_grad():
            for module in trainer.model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    if module.weight.grad is not None:
                        # L1正则化梯度: λ * sign(weight)
                        module.weight.grad.add_(
                            lambda_l1 * torch.sign(module.weight.data)
                        )
    return on_train_batch_end


def main():
    """主函数"""
    save_dir = Path(PROJECT) / NAME
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # ===== 1. 加载并分析原始模型 =====
    LOGGER.info("=" * 60)
    LOGGER.info("步骤1: 加载并分析原始模型")
    LOGGER.info("=" * 60)
    
    model = YOLO(MODEL_PATH)
    original_params = count_parameters(model.model)
    original_gamma = gather_gamma_values(model.model)
    
    LOGGER.info(f"原始模型参数量: {original_params:,}")
    LOGGER.info(f"BN层通道数: {len(original_gamma)}")
    LOGGER.info(f"接近零通道 (<0.01): {np.sum(original_gamma < 0.01)}")
    
    # 计算目标阈值
    target_threshold = compute_prune_threshold(original_gamma, PRUNE_RATIO)
    LOGGER.info(f"目标剪枝阈值: {target_threshold:.6f}")
    
    # 绘制原始分布
    plot_gamma_distribution(
        original_gamma,
        str(save_dir / "1_gamma_original.png"),
        threshold=target_threshold,
        title="原始模型"
    )
    
    # ===== 2. 验证原始模型性能 =====
    LOGGER.info("=" * 60)
    LOGGER.info("步骤2: 验证原始模型性能")
    LOGGER.info("=" * 60)
    
    original_results = model.val(
        data=VAL_DATA_PATH,
        split='test',
        device=DEVICE,
        batch=16,
    )
    original_map50 = original_results.box.map50
    original_map = original_results.box.map
    LOGGER.info(f"原始模型 mAP@50: {original_map50:.4f}, mAP@50-95: {original_map:.4f}")
    
    # ===== 3. 稀疏训练 =====
    LOGGER.info("=" * 60)
    LOGGER.info("步骤3: 稀疏训练 (L1正则化)")
    LOGGER.info("=" * 60)
    LOGGER.info(f"稀疏训练轮次: {SPARSE_EPOCHS}")
    LOGGER.info(f"L1正则化系数: {L1_LAMBDA}")
    
    sparse_model = YOLO(MODEL_PATH)
    
    # 添加L1正则化回调
    sparse_model.add_callback(
        "on_train_batch_end",
        add_l1_regularization(sparse_model, L1_LAMBDA)
    )
    
    # Epoch结束时记录gamma分布
    def on_epoch_end(trainer):
        epoch = trainer.epoch
        if (epoch + 1) % 10 == 0:
            gamma = gather_gamma_values(trainer.model)
            near_zero = np.sum(gamma < 0.01)
            LOGGER.info(f"Epoch {epoch+1}: 接近零通道数 {near_zero} ({near_zero/len(gamma):.1%})")
    
    sparse_model.add_callback("on_train_epoch_end", on_epoch_end)
    
    sparse_results = sparse_model.train(
        data=DATA_PATH,
        epochs=SPARSE_EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        lr0=SPARSE_LR,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0001,
        warmup_epochs=3,
        device=DEVICE,
        project=PROJECT,
        name=f"{NAME}_sparse",
        plots=True,
        save=True,
        val=True,
        patience=15,
        workers=4,
        exist_ok=True,
    )
    
    # 检查稀疏训练后的gamma分布
    sparse_model_path = Path(PROJECT) / f"{NAME}_sparse" / "weights" / "best.pt"
    if not sparse_model_path.exists():
        LOGGER.warning(f"稀疏训练模型不存在: {sparse_model_path}")
        LOGGER.warning("跳过微调，使用原始模型")
        sparse_model_path = Path(MODEL_PATH)
    
    sparse_model = YOLO(str(sparse_model_path))
    sparse_gamma = gather_gamma_values(sparse_model.model)
    
    if len(sparse_gamma) > 0:
        plot_gamma_distribution(
            sparse_gamma,
            str(save_dir / "2_gamma_after_sparse.png"),
            threshold=target_threshold,
            title="稀疏训练后"
        )
        
        sparse_near_zero = np.sum(sparse_gamma < 0.01)
        LOGGER.info(f"稀疏训练后接近零通道: {sparse_near_zero} ({sparse_near_zero/len(sparse_gamma):.1%})")
    
    # ===== 4. 微调恢复精度 =====
    LOGGER.info("=" * 60)
    LOGGER.info("步骤4: 微调恢复精度")
    LOGGER.info("=" * 60)
    LOGGER.info(f"微调轮次: {FINETUNE_EPOCHS}")
    
    # 使用更小的L1系数继续微调
    finetune_model = YOLO(str(sparse_model_path))
    finetune_model.add_callback(
        "on_train_batch_end",
        add_l1_regularization(finetune_model, L1_LAMBDA * 0.1)  # 减小L1系数
    )
    
    finetune_results = finetune_model.train(
        data=DATA_PATH,
        epochs=FINETUNE_EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        lr0=FINETUNE_LR,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0001,
        warmup_epochs=2,
        device=DEVICE,
        project=PROJECT,
        name=f"{NAME}_finetune",
        plots=True,
        save=True,
        val=True,
        patience=10,
        workers=4,
        exist_ok=True,
    )
    
    # ===== 5. 最终验证 =====
    LOGGER.info("=" * 60)
    LOGGER.info("步骤5: 最终验证")
    LOGGER.info("=" * 60)
    
    final_model_path = Path(PROJECT) / f"{NAME}_finetune" / "weights" / "best.pt"
    if not final_model_path.exists():
        final_model_path = Path(PROJECT) / f"{NAME}_finetune" / "weights" / "last.pt"
    if not final_model_path.exists():
        LOGGER.warning(f"微调模型不存在，使用稀疏训练模型")
        final_model_path = sparse_model_path
    if not final_model_path.exists():
        LOGGER.warning(f"使用原始模型")
        final_model_path = Path(MODEL_PATH)
    
    LOGGER.info(f"最终模型路径: {final_model_path}")
    
    final_model = YOLO(str(final_model_path))
    final_results = final_model.val(
        data=VAL_DATA_PATH,
        split='test',
        device=DEVICE,
        batch=16,
    )
    
    final_gamma = gather_gamma_values(final_model.model)
    final_near_zero = np.sum(final_gamma < 0.01) if len(final_gamma) > 0 else 0
    final_very_small = np.sum(final_gamma < 0.001) if len(final_gamma) > 0 else 0
    
    plot_gamma_distribution(
        final_gamma,
        str(save_dir / "3_gamma_final.png"),
        threshold=target_threshold,
        title="最终模型"
    )
    
    # ===== 6. 保存结果 =====
    summary = {
        "原始模型": {
            "路径": MODEL_PATH,
            "参数量": original_params,
            "mAP@50": float(original_map50),
            "mAP@50-95": float(original_map),
            "接近零通道": int(np.sum(original_gamma < 0.01)),
        },
        "剪枝后模型": {
            "路径": str(final_model_path),
            "参数量": count_parameters(final_model.model),
            "mAP@50": float(final_results.box.map50),
            "mAP@50-95": float(final_results.box.map),
            "接近零通道 (<0.01)": int(final_near_zero),
            "极小值通道 (<0.001)": int(final_very_small),
        },
        "精度变化": {
            "ΔmAP@50": float(final_results.box.map50 - original_map50),
            "ΔmAP@50-95": float(final_results.box.map - original_map),
        },
        "训练配置": {
            "目标剪枝比例": f"{PRUNE_RATIO:.0%}",
            "稀疏训练轮次": SPARSE_EPOCHS,
            "L1系数": L1_LAMBDA,
            "微调轮次": FINETUNE_EPOCHS,
        }
    }
    
    with open(save_dir / "prune_summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # 打印结果
    LOGGER.info("=" * 60)
    LOGGER.info("剪枝完成!")
    LOGGER.info("=" * 60)
    LOGGER.info(f"原始模型: mAP@50 = {original_map50:.4f}")
    LOGGER.info(f"剪枝后模型: mAP@50 = {final_results.box.map50:.4f}")
    LOGGER.info(f"精度变化: ΔmAP@50 = {final_results.box.map50 - original_map50:.4f}")
    LOGGER.info(f"接近零通道: {final_near_zero} ({final_near_zero/len(final_gamma):.1%})")
    LOGGER.info(f"结果保存至: {save_dir}")
    
    # 说明后续步骤
    LOGGER.info("")
    LOGGER.info("=" * 60)
    LOGGER.info("后续步骤: 实现真正的模型压缩")
    LOGGER.info("=" * 60)
    LOGGER.info("软剪枝已使不重要通道的gamma接近零。")
    LOGGER.info("要实现真正的参数减少，可以:")
    LOGGER.info("1. 导出ONNX模型，使用ONNX优化工具移除零通道")
    LOGGER.info("2. 使用TensorRT等推理框架进行部署优化")
    LOGGER.info("3. 使用torch.prune进行结构化剪枝（需重建模型）")


if __name__ == "__main__":
    main()
