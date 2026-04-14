"""
模型剪枝工具模块
基于BN层缩放参数的通道剪枝实现

主要功能：
1. BN层gamma参数统计与分析
2. 剪枝阈值计算
3. 通道剪枝实现
4. 模型重建与参数调整
"""

import warnings
from copy import deepcopy
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

from ultralytics.nn.modules import Conv, C2f, C3k2, Bottleneck
from ultralytics.nn.tasks import nn
from ultralytics.utils import LOGGER

warnings.filterwarnings("ignore")


def gather_bn_weights(model: nn.Module) -> Tuple[Dict[str, nn.BatchNorm2d], np.ndarray]:
    """
    收集模型中所有BN层的gamma参数
    
    Args:
        model: 神经网络模型
        
    Returns:
        bn_modules: BN层名称到模块的映射
        gamma_values: 所有gamma参数拼接成的数组
    """
    bn_modules = {}
    gamma_list = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            bn_modules[name] = module
            gamma_list.append(module.weight.data.cpu().numpy())
    
    gamma_values = np.concatenate(gamma_list) if gamma_list else np.array([])
    
    return bn_modules, gamma_values


def compute_prune_threshold(
    gamma_values: np.ndarray,
    prune_ratio: float = 0.3,
    min_threshold: float = 1e-4
) -> float:
    """
    计算剪枝阈值
    
    根据全局剪枝比例确定gamma阈值，使得比例为prune_ratio的通道被剪枝
    
    Args:
        gamma_values: 所有gamma参数值
        prune_ratio: 剪枝比例 (0, 1)
        min_threshold: 最小阈值，避免误剪重要通道
        
    Returns:
        剪枝阈值tau
    """
    if len(gamma_values) == 0:
        return min_threshold
    
    # 取gamma的绝对值
    abs_gamma = np.abs(gamma_values)
    
    # 计算阈值，使得prune_ratio比例的通道gamma值小于阈值
    sorted_gamma = np.sort(abs_gamma)
    threshold_idx = int(len(sorted_gamma) * prune_ratio)
    threshold = sorted_gamma[threshold_idx] if threshold_idx < len(sorted_gamma) else sorted_gamma[-1]
    
    # 确保阈值不小于最小值
    threshold = max(threshold, min_threshold)
    
    LOGGER.info(f"Prune threshold: {threshold:.6f} (ratio: {prune_ratio:.2%})")
    LOGGER.info(f"Gamma statistics: min={abs_gamma.min():.6f}, max={abs_gamma.max():.6f}, "
               f"mean={abs_gamma.mean():.6f}, std={abs_gamma.std():.6f}")
    
    return threshold


def get_prune_mask(
    bn_module: nn.BatchNorm2d,
    threshold: float
) -> np.ndarray:
    """
    获取单个BN层的剪枝掩码
    
    Args:
        bn_module: BN层模块
        threshold: 剪枝阈值
        
    Returns:
        布尔掩码数组，True表示保留该通道
    """
    gamma = bn_module.weight.data.cpu().numpy()
    mask = np.abs(gamma) > threshold
    
    # 确保至少保留一个通道
    if not np.any(mask):
        mask[np.argmax(np.abs(gamma))] = True
    
    return mask


def count_parameters(model: nn.Module) -> int:
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters())


def count_bn_layers(model: nn.Module) -> int:
    """统计BN层数量"""
    return sum(1 for m in model.modules() if isinstance(m, nn.BatchNorm2d))


def analyze_model_for_pruning(model: nn.Module) -> Dict:
    """
    分析模型结构，为剪枝做准备
    
    Args:
        model: 神经网络模型
        
    Returns:
        分析结果字典
    """
    bn_modules, gamma_values = gather_bn_weights(model)
    
    analysis = {
        'total_params': count_parameters(model),
        'total_bn_layers': len(bn_modules),
        'total_channels': len(gamma_values),
        'gamma_mean': float(np.mean(np.abs(gamma_values))) if len(gamma_values) > 0 else 0,
        'gamma_std': float(np.std(np.abs(gamma_values))) if len(gamma_values) > 0 else 0,
        'gamma_min': float(np.min(np.abs(gamma_values))) if len(gamma_values) > 0 else 0,
        'gamma_max': float(np.max(np.abs(gamma_values))) if len(gamma_values) > 0 else 0,
        'near_zero_count': int(np.sum(np.abs(gamma_values) < 0.01)) if len(gamma_values) > 0 else 0,
    }
    
    analysis['near_zero_ratio'] = analysis['near_zero_count'] / analysis['total_channels'] if analysis['total_channels'] > 0 else 0
    
    # 分层统计
    layer_stats = {}
    for name, module in bn_modules.items():
        gamma = module.weight.data.cpu().numpy()
        layer_stats[name] = {
            'num_channels': len(gamma),
            'mean': float(np.mean(np.abs(gamma))),
            'std': float(np.std(np.abs(gamma))),
            'min': float(np.min(np.abs(gamma))),
            'max': float(np.max(np.abs(gamma))),
            'near_zero': int(np.sum(np.abs(gamma) < 0.01)),
        }
    
    analysis['layer_stats'] = layer_stats
    
    return analysis


def prune_conv_bn_pair(
    conv: nn.Conv2d,
    bn: nn.BatchNorm2d,
    in_mask: np.ndarray,
    out_mask: np.ndarray,
) -> Tuple[nn.Conv2d, nn.BatchNorm2d]:
    """
    剪枝Conv-BN层对
    
    Args:
        conv: 卷积层
        bn: BN层
        in_mask: 输入通道掩码
        out_mask: 输出通道掩码
        
    Returns:
        剪枝后的Conv和BN层
    """
    # 剪枝输入通道
    if in_mask is not None and np.sum(in_mask) < conv.in_channels:
        conv.weight.data = conv.weight.data[:, in_mask, :, :]
        conv.in_channels = np.sum(in_mask)
        if conv.bias is not None:
            conv.bias.data = conv.bias.data[in_mask]
    
    # 剪枝输出通道
    if out_mask is not None and np.sum(out_mask) < conv.out_channels:
        conv.weight.data = conv.weight.data[out_mask, :, :, :]
        conv.out_channels = np.sum(out_mask)
        if conv.bias is not None:
            conv.bias.data = conv.bias.data[out_mask]
        
        # 剪枝BN层
        bn.weight.data = bn.weight.data[out_mask]
        bn.bias.data = bn.bias.data[out_mask]
        bn.running_mean = bn.running_mean[out_mask]
        bn.running_var = bn.running_var[out_mask]
        bn.num_features = np.sum(out_mask)
    
    return conv, bn


def get_pruning_plan(
    model: nn.Module,
    prune_ratio: float = 0.3,
    skip_layers: List[str] = None,
) -> Dict[str, np.ndarray]:
    """
    生成剪枝计划
    
    Args:
        model: 神经网络模型
        prune_ratio: 全局剪枝比例
        skip_layers: 跳过剪枝的层名称列表（如检测头）
        
    Returns:
        每个BN层的剪枝掩码字典
    """
    if skip_layers is None:
        skip_layers = ['Detect', 'head']  # 默认跳过检测头
    
    bn_modules, gamma_values = gather_bn_weights(model)
    threshold = compute_prune_threshold(gamma_values, prune_ratio)
    
    prune_plan = {}
    for name, bn_module in bn_modules.items():
        # 检查是否跳过该层
        should_skip = any(skip_key in name for skip_key in skip_layers)
        
        if should_skip:
            # 保留所有通道
            mask = np.ones(bn_module.num_features, dtype=bool)
        else:
            mask = get_prune_mask(bn_module, threshold)
        
        prune_plan[name] = mask
        
        # 记录剪枝情况
        prune_count = np.sum(~mask)
        if prune_count > 0:
            LOGGER.info(f"Layer {name}: prune {prune_count}/{len(mask)} channels "
                       f"({prune_count/len(mask):.1%})")
    
    return prune_plan


def apply_pruning(
    model: nn.Module,
    prune_plan: Dict[str, np.ndarray],
) -> nn.Module:
    """
    应用剪枝计划到模型
    
    Args:
        model: 神经网络模型
        prune_plan: 剪枝计划（BN层名称到掩码的映射）
        
    Returns:
        剪枝后的模型
    """
    model = deepcopy(model)
    
    for name, module in model.named_modules():
        if isinstance(module, Conv) and hasattr(module, 'bn'):
            bn_name = f"{name}.bn"
            if bn_name in prune_plan:
                mask = prune_plan[bn_name]
                num_pruned = np.sum(~mask)
                
                if num_pruned > 0:
                    # 剪枝权重
                    module.conv.weight.data = module.conv.weight.data[mask, :, :, :]
                    module.bn.weight.data = module.bn.weight.data[mask]
                    module.bn.bias.data = module.bn.bias.data[mask]
                    module.bn.running_mean = module.bn.running_mean[mask]
                    module.bn.running_var = module.bn.running_var[mask]
                    
                    # 更新通道数
                    module.conv.out_channels = np.sum(mask)
                    module.bn.num_features = np.sum(mask)
    
    return model


def estimate_pruned_model_size(
    model: nn.Module,
    prune_ratio: float = 0.3,
) -> Dict:
    """
    估计剪枝后的模型大小
    
    Args:
        model: 神经网络模型
        prune_ratio: 剪枝比例
        
    Returns:
        估计结果字典
    """
    original_params = count_parameters(model)
    
    # 简单估计：假设参数量减少与剪枝比例平方成正比（因为卷积权重是输入输出通道的乘积）
    estimated_params = int(original_params * (1 - prune_ratio) ** 2)
    
    return {
        'original_params': original_params,
        'estimated_params': estimated_params,
        'compression_ratio': original_params / estimated_params if estimated_params > 0 else 1,
        'estimated_size_mb': estimated_params * 4 / (1024 * 1024),  # FP32
    }


class BNStatisticsHook:
    """BN层统计钩子，用于记录训练过程中的gamma分布"""
    
    def __init__(self):
        self.gamma_history = []
        self.handles = []
    
    def register(self, model: nn.Module):
        """注册钩子"""
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                handle = module.register_forward_hook(self._hook_fn)
                self.handles.append(handle)
    
    def _hook_fn(self, module, input, output):
        """前向传播钩子"""
        if isinstance(module, nn.BatchNorm2d):
            gamma = module.weight.data.cpu().numpy()
            self.gamma_history.append({
                'mean': float(np.mean(np.abs(gamma))),
                'std': float(np.std(gamma)),
                'near_zero_ratio': float(np.sum(np.abs(gamma) < 0.01) / len(gamma)),
            })
    
    def remove(self):
        """移除钩子"""
        for handle in self.handles:
            handle.remove()
        self.handles = []
    
    def get_statistics(self) -> List[Dict]:
        """获取统计历史"""
        return self.gamma_history
