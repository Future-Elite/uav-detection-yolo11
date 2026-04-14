"""
稀疏训练脚本 - 对BN层gamma参数施加L1正则化
用于模型剪枝前的预处理，使不重要的通道gamma值趋近于0

使用方法:
    python sparse_train.py --model path/to/model.pt --data data.yaml --epochs 50 --lambda 0.001
"""

import argparse
import warnings
from pathlib import Path
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from ultralytics import YOLO
from ultralytics.utils import LOGGER, YAML

warnings.filterwarnings("ignore")


def compute_bn_l1_penalty(model: nn.Module) -> torch.Tensor:
    """
    计算所有BN层gamma参数的L1范数
    
    Args:
        model: 神经网络模型
        
    Returns:
        L1范数值（标量张量）
    """
    l1_penalty = torch.tensor(0.0, device=next(model.parameters()).device)
    
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            # BN层的weight参数即为gamma
            l1_penalty += torch.sum(torch.abs(module.weight))
    
    return l1_penalty


def gather_bn_weights(model: nn.Module) -> dict:
    """
    收集所有BN层的gamma参数信息
    
    Args:
        model: 神经网络模型
        
    Returns:
        包含每层BN参数信息的字典
    """
    bn_info = {}
    idx = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            gamma = module.weight.data.cpu().numpy()
            bn_info[name] = {
                'idx': idx,
                'num_channels': len(gamma),
                'gamma_mean': float(gamma.mean()),
                'gamma_std': float(gamma.std()),
                'gamma_min': float(gamma.min()),
                'gamma_max': float(gamma.max()),
            }
            idx += 1
    
    return bn_info


def plot_gamma_distribution(model: nn.Module, save_path: str = None):
    """
    绘制BN层gamma参数分布直方图
    
    Args:
        model: 神经网络模型
        save_path: 图片保存路径
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        gamma_values = []
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                gamma_values.extend(module.weight.data.cpu().numpy().tolist())
        
        gamma_values = np.array(gamma_values)
        
        plt.figure(figsize=(10, 6))
        plt.hist(gamma_values, bins=100, alpha=0.7, color='blue', edgecolor='black')
        plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='y=0')
        plt.xlabel('Gamma Value', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title('BN Layer Gamma Distribution (Sparse Training)', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 统计信息
        near_zero_ratio = np.sum(np.abs(gamma_values) < 0.01) / len(gamma_values)
        plt.text(0.02, 0.98, f'Near-zero ratio (<0.01): {near_zero_ratio:.2%}',
                transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            LOGGER.info(f"Gamma distribution plot saved to {save_path}")
        
        plt.close()
        
    except ImportError:
        LOGGER.warning("matplotlib not found, skip plotting")


class SparseTrainer:
    """稀疏训练器"""
    
    def __init__(
        self,
        model_path: str,
        data_path: str,
        lambda_sparse: float = 0.001,
        epochs: int = 50,
        batch: int = 4,
        imgsz: int = 640,
        device: int = 0,
        project: str = "runs/detect/sparse",
        name: str = "sparse_train",
        **kwargs
    ):
        """
        初始化稀疏训练器
        
        Args:
            model_path: 模型权重路径
            data_path: 数据集配置路径
            lambda_sparse: L1稀疏正则化系数
            epochs: 训练轮次
            batch: 批次大小
            imgsz: 图像尺寸
            device: GPU设备ID
            project: 项目保存目录
            name: 实验名称
        """
        self.model_path = model_path
        self.data_path = data_path
        self.lambda_sparse = lambda_sparse
        self.epochs = epochs
        self.batch = batch
        self.imgsz = imgsz
        self.device = device
        self.project = project
        self.name = name
        self.kwargs = kwargs
        
        # 加载模型
        self.model = YOLO(model_path)
        
    def train(self):
        """执行稀疏训练"""
        import tempfile
        import os
        
        # 创建回调函数，正确添加L1正则化
        # L1正则化的梯度 = lambda * sign(weight)
        # 在backward之后，optimizer.step之前修改梯度
        
        def on_train_batch_end(trainer):
            """
            在每个batch训练结束后添加L1正则化梯度
            正确方式：直接修改梯度，而不是修改损失值
            """
            with torch.no_grad():
                for module in trainer.model.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        if module.weight.grad is not None:
                            # L1正则化梯度: lambda * sign(weight)
                            module.weight.grad.add_(
                                self.lambda_sparse * torch.sign(module.weight.data)
                            )
        
        def on_train_epoch_start(trainer):
            """每个epoch开始时记录gamma分布"""
            epoch = trainer.epoch
            if epoch % 10 == 0:  # 每10个epoch记录一次
                save_dir = Path(trainer.save_dir)
                plot_gamma_distribution(
                    trainer.model,
                    save_path=str(save_dir / f"gamma_dist_epoch_{epoch}.png")
                )
                
                # 打印BN层信息
                bn_info = gather_bn_weights(trainer.model)
                total_channels = sum(info['num_channels'] for info in bn_info.values())
                near_zero = sum(
                    sum(1 for g in trainer.model.get_submodule(name).weight.data.cpu().numpy() if abs(g) < 0.01)
                    for name in bn_info.keys()
                )
                LOGGER.info(f"Epoch {epoch}: {len(bn_info)} BN layers, {total_channels} channels, "
                          f"{near_zero} near-zero ({near_zero/total_channels:.2%})")
        
        # 添加回调
        self.model.add_callback("on_train_batch_end", on_train_batch_end)
        self.model.add_callback("on_train_epoch_start", on_train_epoch_start)
        
        # 开始训练
        LOGGER.info(f"Starting sparse training with λ={self.lambda_sparse}")
        LOGGER.info(f"Model: {self.model_path}")
        LOGGER.info(f"Data: {self.data_path}")
        LOGGER.info(f"Epochs: {self.epochs}")
        
        results = self.model.train(
            data=self.data_path,
            epochs=self.epochs,
            batch=self.batch,
            imgsz=self.imgsz,
            device=self.device,
            project=self.project,
            name=self.name,
            **self.kwargs
        )
        
        # 训练结束后绘制最终的gamma分布
        final_save_dir = Path(self.project) / self.name
        plot_gamma_distribution(
            self.model.model,
            save_path=str(final_save_dir / "gamma_dist_final.png")
        )
        
        LOGGER.info(f"Sparse training completed. Model saved to {final_save_dir / 'weights' / 'best.pt'}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Sparse training for model pruning")
    parser.add_argument("--model", type=str, required=True, help="Path to model weights")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset config")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--lambda", dest="lambda_sparse", type=float, default=0.001, 
                       help="L1 sparse regularization coefficient")
    parser.add_argument("--device", type=int, default=0, help="GPU device ID")
    parser.add_argument("--project", type=str, default="runs/detect/sparse", help="Project directory")
    parser.add_argument("--name", type=str, default="sparse_train", help="Experiment name")
    parser.add_argument("--lr0", type=float, default=0.001, help="Initial learning rate")
    
    args = parser.parse_args()
    
    trainer = SparseTrainer(
        model_path=args.model,
        data_path=args.data,
        lambda_sparse=args.lambda_sparse,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        lr0=args.lr0,
    )
    
    trainer.train()


if __name__ == "__main__":
    main()
