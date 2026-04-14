"""
模型量化脚本 - 将浮点模型转换为低精度模型并评估对比

量化方法说明：
    1. PTQ (Post-Training Quantization): 训练后量化
       - 不需要重新训练，直接对训练好的模型进行量化
       - 速度快，实现简单
       - 精度有一定损失（通常1-3%）
    
    2. QAT (Quantization-Aware Training): 量化感知训练
       - 在训练过程中模拟量化效果
       - 精度损失更小
       - 需要重新训练

量化精度：
    - FP16: 半精度浮点，2倍压缩，精度损失极小
    - INT8: 8位整数，4倍压缩，精度损失1-3%
    - INT4: 4位整数，8倍压缩，精度损失较大

使用方法：
    python quantize.py                    # 全量量化+评估+绘图
    python quantize.py --method eval      # 仅评估已有模型
    python quantize.py --method fp16      # 仅FP16量化
    python quantize.py --method onnx      # 仅导出ONNX
"""

import warnings
from pathlib import Path
import time
import json

import torch
import numpy as np

from ultralytics import YOLO
from ultralytics.utils import LOGGER

warnings.filterwarnings("ignore")


# ==================== 配置区域 ====================
# 模型路径
MODEL_PATH = "./runs/detect/merged/refined-enhanced/weights/best.pt"

# 验证数据集
VAL_DATA_PATH = "../datasets/Airborne/data.yaml"

# 量化配置
DEVICE = 0
IMG_SIZE = 640

# 输出目录
OUTPUT_DIR = Path("./runs/detect/quantized")
# =================================================


def get_model_size(model_path):
    """获取模型文件/文件夹大小（MB）"""
    path = Path(model_path)
    if not path.exists():
        return 0
    
    if path.is_file():
        size_bytes = path.stat().st_size
    else:
        # 文件夹：计算所有文件大小
        size_bytes = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
    
    return size_bytes / (1024 * 1024)


def count_parameters(model):
    """计算参数量"""
    return sum(p.numel() for p in model.parameters())


def evaluate_model(model_path, name, data_path, device=0):
    """
    评估模型性能
    
    Returns:
        dict: 包含mAP、推理速度、模型大小等指标
    """
    LOGGER.info(f"评估模型: {name}")
    
    result = {
        'name': name,
        'path': str(model_path),
        'size_mb': get_model_size(model_path),
        'map50': 0,
        'map50_95': 0,
        'inference_ms': 0,
        'fps': 0,
        'status': 'success',
    }
    
    try:
        # 加载模型
        model = YOLO(model_path)
        
        # 根据模型类型选择设备
        # ONNX在Windows上可能不支持CUDA，使用CPU
        eval_device = device
        if 'ONNX' in name or 'onnx' in str(model_path).lower():
            # 尝试CUDA，失败则回退CPU
            try:
                test_model = YOLO(model_path)
                test_model.predict(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE), device=device, verbose=False)
            except Exception:
                LOGGER.info(f"  ONNX CUDA不支持，使用CPU评估")
                eval_device = 'cpu'
        
        # NCNN使用CPU
        if 'NCNN' in name or 'ncnn' in str(model_path).lower():
            eval_device = 'cpu'
        
        # 验证获取mAP
        val_results = model.val(
            data=data_path,
            split='test',
            device=eval_device,
            batch=16,
            verbose=False,
        )
        result['map50'] = float(val_results.box.map50)
        result['map50_95'] = float(val_results.box.map)
        
        # 测试推理速度
        if eval_device != 'cpu' and torch.cuda.is_available():
            dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(f'cuda:{device}')
            
            # 预热
            for _ in range(3):
                model.predict(dummy_input, verbose=False)
            
            # 测速
            torch.cuda.synchronize()
            start = time.time()
            num_runs = 20
            for _ in range(num_runs):
                model.predict(dummy_input, verbose=False)
            torch.cuda.synchronize()
            end = time.time()
            
            avg_time = (end - start) / num_runs * 1000
            result['inference_ms'] = round(avg_time, 2)
            result['fps'] = round(1000 / avg_time, 1)
        else:
            # CPU测速
            dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
            for _ in range(3):
                model.predict(dummy_input, verbose=False)
            
            start = time.time()
            num_runs = 10
            for _ in range(num_runs):
                model.predict(dummy_input, verbose=False)
            end = time.time()
            
            avg_time = (end - start) / num_runs * 1000
            result['inference_ms'] = round(avg_time, 2)
            result['fps'] = round(1000 / avg_time, 1)
        
        LOGGER.info(f"  mAP@50: {result['map50']:.4f}, mAP@50-95: {result['map50_95']:.4f}")
        LOGGER.info(f"  推理速度: {result['inference_ms']:.2f} ms ({result['fps']:.1f} FPS)")
        LOGGER.info(f"  模型大小: {result['size_mb']:.2f} MB")
        
    except Exception as e:
        LOGGER.warning(f"  评估失败: {e}")
        result['status'] = f'failed: {str(e)[:80]}'
    
    return result


def export_onnx():
    """导出ONNX格式"""
    LOGGER.info("=" * 60)
    LOGGER.info("导出ONNX格式")
    LOGGER.info("=" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    model = YOLO(MODEL_PATH)
    
    try:
        onnx_path = model.export(
            format='onnx',
            imgsz=IMG_SIZE,
            simplify=True,
            opset=12,
            device=DEVICE,
        )
        LOGGER.info(f"ONNX导出成功: {onnx_path}")
        return str(onnx_path)
    except Exception as e:
        LOGGER.warning(f"ONNX导出失败: {e}")
        return None


def export_tensorrt():
    """导出TensorRT格式"""
    LOGGER.info("=" * 60)
    LOGGER.info("导出TensorRT (FP16)")
    LOGGER.info("=" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    model = YOLO(MODEL_PATH)
    
    try:
        trt_path = model.export(
            format='engine',
            imgsz=IMG_SIZE,
            device=DEVICE,
            half=True,
        )
        LOGGER.info(f"TensorRT导出成功: {trt_path}")
        return str(trt_path)
    except Exception as e:
        LOGGER.warning(f"TensorRT导出失败: {e}")
        return None


def export_openvino():
    """导出OpenVINO格式"""
    LOGGER.info("=" * 60)
    LOGGER.info("导出OpenVINO格式")
    LOGGER.info("=" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    model = YOLO(MODEL_PATH)
    
    try:
        openvino_path = model.export(
            format='openvino',
            imgsz=IMG_SIZE,
        )
        LOGGER.info(f"OpenVINO导出成功: {openvino_path}")
        return str(openvino_path)
    except Exception as e:
        LOGGER.warning(f"OpenVINO导出失败: {e}")
        return None


def export_ncnn():
    """导出NCNN格式"""
    LOGGER.info("=" * 60)
    LOGGER.info("导出NCNN格式")
    LOGGER.info("=" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    model = YOLO(MODEL_PATH)
    
    try:
        ncnn_path = model.export(
            format='ncnn',
            imgsz=IMG_SIZE,
            simplify=True,
        )
        LOGGER.info(f"NCNN导出成功: {ncnn_path}")
        return str(ncnn_path)
    except Exception as e:
        LOGGER.warning(f"NCNN导出失败: {e}")
        return None


def export_torchscript_fp16():
    """导出TorchScript FP16格式"""
    LOGGER.info("=" * 60)
    LOGGER.info("导出TorchScript (FP16)")
    LOGGER.info("=" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    model = YOLO(MODEL_PATH)
    
    try:
        # 导出TorchScript格式（支持FP16）
        ts_path = model.export(
            format='torchscript',
            imgsz=IMG_SIZE,
            device=DEVICE,
            half=True,  # FP16
        )
        LOGGER.info(f"TorchScript FP16导出成功: {ts_path}")
        return str(ts_path)
    except Exception as e:
        LOGGER.warning(f"TorchScript导出失败: {e}")
        return None


def export_fp16():
    """FP16半精度量化（使用原始模型在推理时启用FP16）"""
    LOGGER.info("=" * 60)
    LOGGER.info("FP16 半精度量化")
    LOGGER.info("=" * 60)
    
    # FP16不需要单独导出文件，直接使用原始模型
    # 在评估时会测试FP16推理速度
    LOGGER.info("FP16使用原始模型，推理时启用half()模式")
    return MODEL_PATH  # 返回原始模型路径，评估时使用FP16


def plot_comparison(results, output_dir):
    """
    绘制对比图
    
    Args:
        results: 评估结果列表
        output_dir: 输出目录
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 过滤有效结果
        valid_results = [r for r in results if r['status'] == 'success']
        if len(valid_results) < 2:
            LOGGER.warning("有效结果不足，跳过绘图")
            return
        
        names = [r['name'] for r in valid_results]
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. mAP@50 对比
        ax1 = axes[0, 0]
        map50_values = [r['map50'] for r in valid_results]
        colors = ['#2ecc71' if v == max(map50_values) else '#3498db' for v in map50_values]
        bars1 = ax1.bar(names, map50_values, color=colors, edgecolor='black', linewidth=1.2)
        ax1.set_ylabel('mAP@50', fontsize=12)
        ax1.set_title('检测精度对比 (mAP@50)', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, max(map50_values) * 1.15)
        ax1.tick_params(axis='x', rotation=15)
        for bar, val in zip(bars1, map50_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 2. 模型大小对比
        ax2 = axes[0, 1]
        size_values = [r['size_mb'] for r in valid_results]
        colors2 = ['#e74c3c' if v == min(size_values) else '#f39c12' for v in size_values]
        bars2 = ax2.bar(names, size_values, color=colors2, edgecolor='black', linewidth=1.2)
        ax2.set_ylabel('模型大小 (MB)', fontsize=12)
        ax2.set_title('模型大小对比', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=15)
        for bar, val in zip(bars2, size_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{val:.1f}', ha='center', va='bottom', fontsize=10)
        
        # 3. 推理速度对比
        ax3 = axes[1, 0]
        fps_values = [r['fps'] for r in valid_results if r['fps'] > 0]
        fps_names = [r['name'] for r in valid_results if r['fps'] > 0]
        if fps_values:
            colors3 = ['#9b59b6' if v == max(fps_values) else '#8e44ad' for v in fps_values]
            bars3 = ax3.bar(fps_names, fps_values, color=colors3, edgecolor='black', linewidth=1.2)
            ax3.set_ylabel('FPS', fontsize=12)
            ax3.set_title('推理速度对比 (FPS)', fontsize=14, fontweight='bold')
            ax3.tick_params(axis='x', rotation=15)
            for bar, val in zip(bars3, fps_values):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                        f'{val:.1f}', ha='center', va='bottom', fontsize=10)
        
        # 4. 综合指标雷达图
        ax4 = axes[1, 1]
        
        # 归一化指标
        original = valid_results[0]  # 原始模型作为基准
        
        # 计算相对指标 (相对于原始模型)
        relative_map = [r['map50'] / original['map50'] * 100 for r in valid_results]
        relative_size = [original['size_mb'] / r['size_mb'] * 100 if r['size_mb'] > 0 else 0 for r in valid_results]
        relative_fps = [r['fps'] / original['fps'] * 100 if original['fps'] > 0 and r['fps'] > 0 else 0 for r in valid_results]
        
        x = np.arange(len(names))
        width = 0.25
        
        ax4.bar(x - width, relative_map, width, label='相对精度', color='#3498db')
        ax4.bar(x, relative_size, width, label='压缩比', color='#e74c3c')
        ax4.bar(x + width, relative_fps, width, label='加速比', color='#2ecc71')
        
        ax4.set_ylabel('相对值 (%)', fontsize=12)
        ax4.set_title('综合指标对比 (相对于原始模型)', fontsize=14, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(names, rotation=15)
        ax4.axhline(y=100, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        ax4.legend(loc='upper right')
        
        plt.tight_layout()
        
        # 保存图表
        output_path = output_dir / "quantize_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        LOGGER.info(f"对比图保存至: {output_path}")
        
        # 额外绘制：精度-大小权衡图
        plot_precision_size_tradeoff(valid_results, output_dir)
        
    except ImportError:
        LOGGER.warning("matplotlib未安装，跳过绘图")
    except Exception as e:
        LOGGER.warning(f"绘图失败: {e}")


def plot_precision_size_tradeoff(results, output_dir):
    """
    绘制精度-大小权衡图
    
    帮助选择最优的量化方案
    """
    try:
        import matplotlib.pyplot as plt
        
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # 绘制散点
        for r in results:
            if r['status'] != 'success':
                continue
            
            size = r['size_mb']
            map50 = r['map50']
            name = r['name']
            
            # 根据类型选择颜色和标记
            if '原始' in name or 'PyTorch' in name:
                color, marker = '#e74c3c', 's'  # 红色方形
            elif 'ONNX' in name:
                color, marker = '#3498db', 'o'  # 蓝色圆形
            elif 'TensorRT' in name or 'TRT' in name:
                color, marker = '#2ecc71', '^'  # 绿色三角
            elif 'FP16' in name:
                color, marker = '#9b59b6', 'D'  # 紫色菱形
            else:
                color, marker = '#f39c12', 'p'  # 橙色五边形
            
            ax.scatter(size, map50, c=color, marker=marker, s=200, edgecolors='black', linewidths=1.5, zorder=3)
            ax.annotate(name, (size, map50), xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
        
        # 设置坐标轴
        ax.set_xlabel('模型大小 (MB)', fontsize=13)
        ax.set_ylabel('mAP@50', fontsize=13)
        ax.set_title('精度-大小权衡图\n(右上角为理想区域)', fontsize=14, fontweight='bold')
        
        # 添加网格
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 添加理想区域标注
        ax.axhline(y=max(r['map50'] for r in results if r['status']=='success') * 0.95, 
                   color='green', linestyle=':', alpha=0.5, label='95%精度线')
        
        plt.tight_layout()
        
        output_path = output_dir / "precision_size_tradeoff.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        LOGGER.info(f"权衡图保存至: {output_path}")
        
    except Exception as e:
        LOGGER.warning(f"权衡图绘制失败: {e}")


def full_evaluation():
    """
    完整的量化+评估流程
    
    1. 导出各种格式
    2. 评估所有模型
    3. 绘制对比图
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    LOGGER.info("=" * 60)
    LOGGER.info("开始完整的量化评估流程")
    LOGGER.info("=" * 60)
    
    results = []
    
    # 1. 评估原始PyTorch模型
    LOGGER.info("\n[1/6] 评估原始PyTorch模型...")
    original_result = evaluate_model(MODEL_PATH, '原始PyTorch', VAL_DATA_PATH, DEVICE)
    results.append(original_result)
    
    # 2. 导出并评估ONNX
    LOGGER.info("\n[2/6] 导出ONNX...")
    onnx_path = export_onnx()
    if onnx_path:
        onnx_result = evaluate_model(onnx_path, 'ONNX', VAL_DATA_PATH, DEVICE)
        results.append(onnx_result)
    
    # 3. 导出并评估TorchScript (FP16)
    LOGGER.info("\n[3/6] 导出TorchScript (FP16)...")
    ts_path = export_torchscript_fp16()
    if ts_path:
        ts_result = evaluate_model(ts_path, 'TorchScript-FP16', VAL_DATA_PATH, DEVICE)
        results.append(ts_result)
    
    # 4. 导出并评估TensorRT
    LOGGER.info("\n[4/6] 导出TensorRT...")
    trt_path = export_tensorrt()
    if trt_path:
        trt_result = evaluate_model(trt_path, 'TensorRT', VAL_DATA_PATH, DEVICE)
        results.append(trt_result)
    
    # 5. 导出并评估OpenVINO
    LOGGER.info("\n[5/6] 导出OpenVINO...")
    openvino_path = export_openvino()
    if openvino_path:
        openvino_result = evaluate_model(openvino_path, 'OpenVINO', VAL_DATA_PATH, DEVICE)
        results.append(openvino_result)
    
    # 6. 导出并评估NCNN
    LOGGER.info("\n[6/6] 导出NCNN...")
    ncnn_path = export_ncnn()
    if ncnn_path:
        ncnn_result = evaluate_model(ncnn_path, 'NCNN', VAL_DATA_PATH, DEVICE)
        results.append(ncnn_result)
    
    # 保存结果
    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("保存评估结果")
    LOGGER.info("=" * 60)
    
    output_file = OUTPUT_DIR / "quantize_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    LOGGER.info(f"结果保存至: {output_file}")
    
    # 绘制对比图
    LOGGER.info("\n绘制对比图...")
    plot_comparison(results, OUTPUT_DIR)
    
    # 打印总结
    print_summary(results)
    
    return results


def print_summary(results):
    """打印评估总结"""
    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("评估总结")
    LOGGER.info("=" * 60)
    
    # 表头
    print(f"\n{'模型':<15} {'mAP@50':<10} {'mAP@50-95':<10} {'大小(MB)':<10} {'推理(ms)':<10} {'FPS':<10}")
    print("-" * 65)
    
    for r in results:
        if r['status'] != 'success':
            print(f"{r['name']:<15} {'FAILED':<10}")
            continue
        
        print(f"{r['name']:<15} {r['map50']:<10.4f} {r['map50_95']:<10.4f} "
              f"{r['size_mb']:<10.2f} {r['inference_ms']:<10.2f} {r['fps']:<10.1f}")
    
    print("-" * 65)
    
    # 计算压缩比和加速比
    if len(results) >= 2 and results[0]['status'] == 'success':
        original = results[0]
        print(f"\n相对于原始模型 ({original['name']}):")
        for r in results[1:]:
            if r['status'] != 'success':
                continue
            
            # 安全计算压缩比
            if r['size_mb'] > 0 and original['size_mb'] > 0:
                compression = original['size_mb'] / r['size_mb']
            else:
                compression = 0
            
            # 安全计算加速比
            if original['fps'] > 0 and r['fps'] > 0:
                speedup = r['fps'] / original['fps']
            else:
                speedup = 0
            
            # 安全计算精度保留率
            if original['map50'] > 0:
                precision_retention = r['map50'] / original['map50'] * 100
            else:
                precision_retention = 0
            
            comp_str = f"{compression:.2f}x" if compression > 0 else "N/A"
            speed_str = f"{speedup:.2f}x" if speedup > 0 else "N/A"
            print(f"  {r['name']}: 压缩{comp_str}, 加速{speed_str}, 精度保留{precision_retention:.1f}%")


def evaluate_existing():
    """评估已导出的模型"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    LOGGER.info("评估已存在的量化模型...")
    
    results = []
    
    # 查找已存在的模型
    model_files = {
        '原始PyTorch': MODEL_PATH,
        'ONNX': str(OUTPUT_DIR.parent.parent / MODEL_PATH.split('/')[-1].replace('.pt', '.onnx')),
    }
    
    # 检查OUTPUT_DIR下的模型
    for f in OUTPUT_DIR.iterdir():
        if f.is_file():
            name = f.stem.replace('model_', '')
            model_files[name] = str(f)
    
    # 查找导出的模型
    for pattern, name in [('*_best.onnx', 'ONNX'), ('*.engine', 'TensorRT'), ('*_openvino_model', 'OpenVINO')]:
        files = list(OUTPUT_DIR.parent.glob(pattern))
        if files:
            model_files[name] = str(files[0])
    
    for name, path in model_files.items():
        if Path(path).exists():
            result = evaluate_model(path, name, VAL_DATA_PATH, DEVICE)
            results.append(result)
    
    if results:
        plot_comparison(results, OUTPUT_DIR)
        print_summary(results)
    
    return results


def main():
    """主入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='模型量化与评估')
    parser.add_argument('--method', type=str, default='all',
                        choices=['all', 'eval', 'torchscript', 'onnx', 'tensorrt', 
                                 'openvino', 'ncnn', 'compare'],
                        help='量化方法: all(全量), eval(仅评估)')
    args = parser.parse_args()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.method == 'all':
        full_evaluation()
    elif args.method == 'eval':
        evaluate_existing()
    elif args.method == 'torchscript':
        path = export_torchscript_fp16()
        if path:
            result = evaluate_model(path, 'TorchScript-FP16', VAL_DATA_PATH, DEVICE)
            print_summary([result])
    elif args.method == 'onnx':
        path = export_onnx()
        if path:
            result = evaluate_model(path, 'ONNX', VAL_DATA_PATH, DEVICE)
            print_summary([result])
    elif args.method == 'tensorrt':
        path = export_tensorrt()
        if path:
            result = evaluate_model(path, 'TensorRT', VAL_DATA_PATH, DEVICE)
            print_summary([result])
    elif args.method == 'openvino':
        path = export_openvino()
        if path:
            result = evaluate_model(path, 'OpenVINO', VAL_DATA_PATH, DEVICE)
            print_summary([result])
    elif args.method == 'ncnn':
        path = export_ncnn()
        if path:
            result = evaluate_model(path, 'NCNN', VAL_DATA_PATH, DEVICE)
            print_summary([result])


if __name__ == "__main__":
    main()
