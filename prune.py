"""
模型剪枝脚本
基于BN层缩放参数的通道剪枝实现

完整剪枝流程：
    步骤一：稀疏训练（对BN层gamma施加L1正则化）
    步骤二：通道评估与排序（统计gamma分布，确定阈值）
    步骤三：通道剪枝与模型重建
    步骤四：微调训练
    步骤五：迭代优化

使用方法:
    # 单次剪枝
    python prune.py --model path/to/model.pt --data data.yaml --ratio 0.3
    
    # 迭代剪枝
    python prune.py --model path/to/model.pt --data data.yaml --iterative --ratios 0.1 0.2 0.3
    
    # 完整流程（稀疏训练 + 剪枝 + 微调）
    python prune.py --model path/to/model.pt --data data.yaml --ratio 0.3 --sparse-train --finetune
"""

import argparse
import warnings
from pathlib import Path
from typing import Optional, List
import json

import torch
import torch.nn as nn
import numpy as np

from ultralytics import YOLO
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.utils import LOGGER, YAML

from prune_utils import (
    gather_bn_weights,
    compute_prune_threshold,
    get_prune_mask,
    analyze_model_for_pruning,
    count_parameters,
    estimate_pruned_model_size,
    get_pruning_plan,
)

warnings.filterwarnings("ignore")


def plot_gamma_histogram(
    gamma_values: np.ndarray,
    save_path: str,
    threshold: float = None,
    title: str = "BN Gamma Distribution"
):
    """绘制gamma分布直方图"""
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 6))
        
        # 主直方图
        plt.subplot(1, 2, 1)
        plt.hist(np.abs(gamma_values), bins=100, alpha=0.7, color='blue', edgecolor='black')
        if threshold is not None:
            plt.axvline(x=threshold, color='red', linestyle='--', linewidth=2, 
                       label=f'Threshold={threshold:.4f}')
        plt.xlabel('|Gamma|', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 对数尺度
        plt.subplot(1, 2, 2)
        plt.hist(np.abs(gamma_values), bins=100, alpha=0.7, color='green', edgecolor='black')
        if threshold is not None:
            plt.axvline(x=threshold, color='red', linestyle='--', linewidth=2)
        plt.xlabel('|Gamma|', fontsize=12)
        plt.ylabel('Count (log)', fontsize=12)
        plt.yscale('log')
        plt.title('Gamma Distribution (Log Scale)', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        LOGGER.info(f"Histogram saved to {save_path}")
        
    except ImportError:
        LOGGER.warning("matplotlib not found, skip plotting")


class ModelPruner:
    """模型剪枝器"""
    
    def __init__(
        self,
        model_path: str,
        data_path: str,
        prune_ratio: float = 0.3,
        device: int = 0,
        project: str = "runs/detect/prune",
        name: str = "pruned",
        skip_layers: List[str] = None,
    ):
        """
        初始化模型剪枝器
        
        Args:
            model_path: 模型权重路径
            data_path: 数据集配置路径
            prune_ratio: 剪枝比例
            device: GPU设备ID
            project: 项目保存目录
            name: 实验名称
            skip_layers: 跳过剪枝的层名称列表
        """
        self.model_path = model_path
        self.data_path = data_path
        self.prune_ratio = prune_ratio
        self.device = device
        self.project = Path(project)
        self.name = name
        self.skip_layers = skip_layers or ['Detect', 'head.']
        
        # 创建保存目录
        self.save_dir = self.project / name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载模型
        self.model = YOLO(model_path)
        self.original_params = count_parameters(self.model.model)
        
    def analyze(self) -> dict:
        """
        分析模型，准备剪枝
        
        Returns:
            分析结果字典
        """
        LOGGER.info("=" * 60)
        LOGGER.info("Analyzing model for pruning...")
        LOGGER.info("=" * 60)
        
        analysis = analyze_model_for_pruning(self.model.model)
        
        LOGGER.info(f"Total parameters: {analysis['total_params']:,}")
        LOGGER.info(f"Total BN layers: {analysis['total_bn_layers']}")
        LOGGER.info(f"Total channels: {analysis['total_channels']}")
        LOGGER.info(f"Gamma mean: {analysis['gamma_mean']:.6f}")
        LOGGER.info(f"Gamma std: {analysis['gamma_std']:.6f}")
        LOGGER.info(f"Near-zero channels (<0.01): {analysis['near_zero_count']} ({analysis['near_zero_ratio']:.2%})")
        
        # 保存分析结果
        with open(self.save_dir / "analysis.json", 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        return analysis
    
    def compute_threshold(self) -> float:
        """计算剪枝阈值"""
        bn_modules, gamma_values = gather_bn_weights(self.model.model)
        threshold = compute_prune_threshold(gamma_values, self.prune_ratio)
        
        # 绘制gamma分布直方图
        plot_gamma_histogram(
            gamma_values,
            save_path=str(self.save_dir / "gamma_distribution.png"),
            threshold=threshold,
            title=f"BN Gamma Distribution (Prune Ratio: {self.prune_ratio:.0%})"
        )
        
        return threshold
    
    def prune(self) -> nn.Module:
        """
        执行剪枝
        
        Returns:
            剪枝后的模型
        """
        LOGGER.info("=" * 60)
        LOGGER.info(f"Pruning model with ratio {self.prune_ratio:.0%}...")
        LOGGER.info("=" * 60)
        
        # 获取剪枝计划
        prune_plan = get_pruning_plan(
            self.model.model,
            self.prune_ratio,
            self.skip_layers
        )
        
        # 统计剪枝情况
        total_channels = 0
        pruned_channels = 0
        for name, mask in prune_plan.items():
            total_channels += len(mask)
            pruned_channels += np.sum(~mask)
        
        LOGGER.info(f"Total channels: {total_channels}")
        LOGGER.info(f"Channels to prune: {pruned_channels} ({pruned_channels/total_channels:.2%})")
        
        # 执行剪枝
        pruned_model = self._apply_pruning(prune_plan)
        
        # 保存剪枝后的模型
        pruned_params = count_parameters(pruned_model)
        compression_ratio = self.original_params / pruned_params
        
        LOGGER.info(f"Original parameters: {self.original_params:,}")
        LOGGER.info(f"Pruned parameters: {pruned_params:,}")
        LOGGER.info(f"Compression ratio: {compression_ratio:.2f}x")
        
        # 保存统计信息
        stats = {
            'original_params': self.original_params,
            'pruned_params': pruned_params,
            'compression_ratio': compression_ratio,
            'prune_ratio': self.prune_ratio,
            'total_channels': total_channels,
            'pruned_channels': pruned_channels,
        }
        
        with open(self.save_dir / "prune_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        return pruned_model
    
    def _apply_pruning(self, prune_plan: dict) -> nn.Module:
        """
        应用剪枝计划
        
        Args:
            prune_plan: 剪枝计划
            
        Returns:
            剪枝后的模型
        """
        import torch.nn.utils.prune as torch_prune
        
        model = self.model.model
        
        for name, module in model.named_modules():
            if name in prune_plan:
                mask = prune_plan[name]
                if isinstance(module, nn.BatchNorm2d):
                    # 创建自定义剪枝掩码
                    mask_tensor = torch.from_numpy(mask).to(module.weight.device)
                    
                    # 应用剪枝
                    torch_prune.custom_from_mask(module, 'weight', mask_tensor)
                    
                    # 同步剪枝bias和running stats
                    module.bias.data = module.bias.data[mask]
                    module.running_mean = module.running_mean[mask]
                    module.running_var = module.running_var[mask]
                    module.num_features = int(np.sum(mask))
        
        return model
    
    def finetune(
        self,
        epochs: int = 30,
        batch: int = 4,
        imgsz: int = 640,
        lr: float = 0.0001,
    ) -> dict:
        """
        微调训练剪枝后的模型
        
        Args:
            epochs: 微调轮次
            batch: 批次大小
            imgsz: 图像尺寸
            lr: 学习率
            
        Returns:
            训练结果
        """
        LOGGER.info("=" * 60)
        LOGGER.info("Fine-tuning pruned model...")
        LOGGER.info("=" * 60)
        
        results = self.model.train(
            data=self.data_path,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            lr0=lr,
            device=self.device,
            project=str(self.project),
            name=f"{self.name}_finetune",
        )
        
        return results
    
    def validate(self, data_path: str = None) -> dict:
        """
        验证模型性能
        
        Args:
            data_path: 验证数据集路径
            
        Returns:
            验证结果
        """
        data = data_path or self.data_path
        
        LOGGER.info("=" * 60)
        LOGGER.info(f"Validating model on {data}...")
        LOGGER.info("=" * 60)
        
        results = self.model.val(
            data=data,
            device=self.device,
        )
        
        return results
    
    def save_pruned_model(self, save_path: str = None):
        """保存剪枝后的模型"""
        if save_path is None:
            save_path = self.save_dir / "pruned_model.pt"
        
        torch.save({
            'model': self.model.model.state_dict(),
            'prune_ratio': self.prune_ratio,
            'original_params': self.original_params,
            'pruned_params': count_parameters(self.model.model),
        }, save_path)
        
        LOGGER.info(f"Pruned model saved to {save_path}")


def iterative_pruning(
    model_path: str,
    data_path: str,
    ratios: List[float],
    device: int = 0,
    project: str = "runs/detect/prune",
    finetune_epochs: int = 30,
):
    """
    迭代剪枝
    
    Args:
        model_path: 初始模型路径
        data_path: 数据集路径
        ratios: 剪枝比例列表，如[0.1, 0.2, 0.3]
        device: GPU设备ID
        project: 项目目录
        finetune_epochs: 每次剪枝后的微调轮次
    """
    LOGGER.info("=" * 60)
    LOGGER.info("Starting iterative pruning...")
    LOGGER.info(f"Prune ratios: {ratios}")
    LOGGER.info("=" * 60)
    
    current_model_path = model_path
    
    for i, ratio in enumerate(ratios):
        LOGGER.info(f"\n--- Iteration {i+1}/{len(ratios)}: prune ratio = {ratio:.0%} ---")
        
        pruner = ModelPruner(
            model_path=current_model_path,
            data_path=data_path,
            prune_ratio=ratio,
            device=device,
            project=project,
            name=f"iter_{i}_ratio_{int(ratio*100)}",
        )
        
        # 分析
        pruner.analyze()
        
        # 剪枝
        pruner.prune()
        
        # 微调
        pruner.finetune(epochs=finetune_epochs)
        
        # 验证
        results = pruner.validate()
        
        # 保存
        save_path = Path(project) / f"iter_{i}_ratio_{int(ratio*100)}" / "best.pt"
        pruner.model.save(str(save_path))
        
        # 更新模型路径用于下一次迭代
        current_model_path = str(save_path)
        
        LOGGER.info(f"Iteration {i+1} completed. mAP@50: {results.box.map50:.4f}")
    
    LOGGER.info("=" * 60)
    LOGGER.info("Iterative pruning completed!")
    LOGGER.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Model pruning based on BN layer gamma parameters")
    
    # 基本参数
    parser.add_argument("--model", type=str, required=True, help="Path to model weights")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset config")
    parser.add_argument("--ratio", type=float, default=0.3, help="Pruning ratio (0-1)")
    parser.add_argument("--device", type=int, default=0, help="GPU device ID")
    parser.add_argument("--project", type=str, default="runs/detect/prune", help="Project directory")
    parser.add_argument("--name", type=str, default="pruned", help="Experiment name")
    
    # 迭代剪枝
    parser.add_argument("--iterative", action="store_true", help="Enable iterative pruning")
    parser.add_argument("--ratios", type=float, nargs='+', default=[0.1, 0.2, 0.3],
                       help="List of pruning ratios for iterative pruning")
    
    # 微调参数
    parser.add_argument("--finetune", action="store_true", help="Enable fine-tuning after pruning")
    parser.add_argument("--finetune-epochs", type=int, default=30, help="Fine-tuning epochs")
    parser.add_argument("--finetune-lr", type=float, default=0.0001, help="Fine-tuning learning rate")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    
    # 稀疏训练
    parser.add_argument("--sparse-train", action="store_true", help="Run sparse training before pruning")
    parser.add_argument("--sparse-epochs", type=int, default=50, help="Sparse training epochs")
    parser.add_argument("--lambda-sparse", type=float, default=0.001, help="L1 regularization coefficient")
    
    # 验证
    parser.add_argument("--validate", action="store_true", help="Validate model after pruning")
    parser.add_argument("--val-data", type=str, default=None, help="Validation dataset path")
    
    args = parser.parse_args()
    
    # 迭代剪枝模式
    if args.iterative:
        iterative_pruning(
            model_path=args.model,
            data_path=args.data,
            ratios=args.ratios,
            device=args.device,
            project=args.project,
            finetune_epochs=args.finetune_epochs,
        )
        return
    
    # 单次剪枝模式
    pruner = ModelPruner(
        model_path=args.model,
        data_path=args.data,
        prune_ratio=args.ratio,
        device=args.device,
        project=args.project,
        name=args.name,
    )
    
    # 稀疏训练
    if args.sparse_train:
        LOGGER.info("Running sparse training...")
        from sparse_train import SparseTrainer
        
        sparse_trainer = SparseTrainer(
            model_path=args.model,
            data_path=args.data,
            lambda_sparse=args.lambda_sparse,
            epochs=args.sparse_epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            project=args.project,
            name=f"{args.name}_sparse",
        )
        sparse_trainer.train()
        
        # 更新模型路径
        pruner.model_path = str(Path(args.project) / f"{args.name}_sparse" / "weights" / "best.pt")
        pruner.model = YOLO(pruner.model_path)
    
    # 分析
    pruner.analyze()
    
    # 剪枝
    pruner.prune()
    
    # 微调
    if args.finetune:
        pruner.finetune(
            epochs=args.finetune_epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            lr=args.finetune_lr,
        )
    
    # 验证
    if args.validate:
        results = pruner.validate(args.val_data)
        LOGGER.info(f"Validation mAP@50: {results.box.map50:.4f}")
        LOGGER.info(f"Validation mAP@50-95: {results.box.map:.4f}")
    
    # 保存
    pruner.save_pruned_model()


if __name__ == "__main__":
    main()
