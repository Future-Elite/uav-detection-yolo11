"""
模型性能测试脚本

使用方法：
    1. 在下方 MODELS 列表中配置要测试的模型
    2. 运行: python benchmark_model.py
"""

import time
from pathlib import Path

import torch
from ultralytics import YOLO

# ==================== 配置 ====================
VAL_DATA = "../datasets/Airborne/data.yaml"
IMG_SIZE = 640
DEVICE = 0

# 在此配置要测试的模型：(路径, 名称)
MODELS = [
    ("./runs/detect/merged/refined-enhanced3/weights/best.pt", "FP32"),
    ("runs/detect/quantized/model_fp16.pt", "FP16"),
    ("runs/detect/quantized_onnx/model_int8_ptq.onnx", "INT8-PTQ"),
    # 添加更多模型...
]
# ===============================================


def get_model_size(path):
    """获取模型大小（MB）"""
    p = Path(path)
    return p.stat().st_size / (1024 * 1024)


def benchmark_torchscript(model_path, name):
    """测试 TorchScript 模型（只测推理速度）"""
    print(f"\n{'='*50}")
    print(f" 测试模型: {name} (TorchScript)")
    print(f"{'='*50}")
    
    size_mb = get_model_size(model_path)
    print(f"模型大小: {size_mb:.2f} MB")
    
    # 加载模型
    print("加载模型...")
    device = f'cuda:{DEVICE}' if torch.cuda.is_available() else 'cpu'
    model = torch.jit.load(str(model_path), map_location=device)
    model.eval()
    
    # 测速
    print("测速中...")
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
    
    # 预热
    with torch.no_grad():
        for _ in range(5):
            model(dummy)
    
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    
    # 测速
    start = time.time()
    with torch.no_grad():
        for _ in range(30):
            model(dummy)
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    inference_ms = (time.time() - start) / 30 * 1000
    
    fps = 1000 / inference_ms
    print(f"推理时间: {inference_ms:.2f} ms")
    print(f"FPS: {fps:.1f}")
    
    return {
        'name': name,
        'map50': -1,
        'map50_95': -1,
        'size_mb': size_mb,
        'inference_ms': inference_ms,
        'fps': fps,
    }


def benchmark_onnx(model_path, name):
    """测试 ONNX 模型（只测推理速度）"""
    print(f"\n{'='*50}")
    print(f" 测试模型: {name} (ONNX)")
    print(f"{'='*50}")
    
    size_mb = get_model_size(model_path)
    print(f"模型大小: {size_mb:.2f} MB")
    
    import onnxruntime as ort
    import numpy as np
    
    # 检查可用的执行提供者
    providers = ort.get_available_providers()
    print(f"可用提供者: {providers}")
    
    # 优先使用 CUDA
    if 'CUDAExecutionProvider' in providers:
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(str(model_path), sess_options, providers=['CUDAExecutionProvider'])
        device_name = 'CUDA'
    else:
        session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
        device_name = 'CPU'
    
    print(f"使用设备: {device_name}")
    
    # 获取输入信息
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    print(f"输入: {input_name}, 形状: {input_shape}")
    
    # 测速
    print("测速中...")
    dummy = np.random.randn(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)
    
    # 预热
    for _ in range(5):
        session.run(None, {input_name: dummy})
    
    # 测速
    start = time.time()
    for _ in range(30):
        session.run(None, {input_name: dummy})
    inference_ms = (time.time() - start) / 30 * 1000
    
    fps = 1000 / inference_ms
    print(f"推理时间: {inference_ms:.2f} ms")
    print(f"FPS: {fps:.1f}")
    
    return {
        'name': name,
        'map50': -1,
        'map50_95': -1,
        'size_mb': size_mb,
        'inference_ms': inference_ms,
        'fps': fps,
    }


def benchmark_yolo(model_path, name, val_data):
    """测试 YOLO PyTorch 模型"""
    print(f"\n{'='*50}")
    print(f" 测试模型: {name}")
    print(f"{'='*50}")
    
    size_mb = get_model_size(model_path)
    print(f"模型大小: {size_mb:.2f} MB")
    
    # 加载模型
    print("加载模型...")
    model = YOLO(str(model_path))
    
    # 验证
    print("验证中...")
    device = DEVICE
    val_res = model.val(data=val_data, split='test', device=device, batch=16, verbose=False)
    map50 = float(val_res.box.map50)
    map50_95 = float(val_res.box.map)
    print(f"mAP@50: {map50:.4f}")
    print(f"mAP@50-95: {map50_95:.4f}")
    
    # 测速
    print("测速中...")
    if torch.cuda.is_available():
        dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(f'cuda:{device}')
        for _ in range(5):
            model.predict(dummy, verbose=False)
        torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(30):
            model.predict(dummy, verbose=False)
        torch.cuda.synchronize()
        inference_ms = (time.time() - start) / 30 * 1000
    else:
        dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        for _ in range(3):
            model.predict(dummy, verbose=False)
        
        start = time.time()
        for _ in range(10):
            model.predict(dummy, verbose=False)
        inference_ms = (time.time() - start) / 10 * 1000
    
    fps = 1000 / inference_ms
    print(f"推理时间: {inference_ms:.2f} ms")
    print(f"FPS: {fps:.1f}")
    
    return {
        'name': name,
        'map50': map50,
        'map50_95': map50_95,
        'size_mb': size_mb,
        'inference_ms': inference_ms,
        'fps': fps,
    }


def benchmark(model_path, name=None, val_data=VAL_DATA):
    """测试单个模型（自动识别格式）"""
    model_path = Path(model_path)
    name = name or model_path.stem
    suffix = model_path.suffix.lower()
    
    if suffix == '.pt':
        return benchmark_yolo(model_path, name, val_data)
    elif suffix == '.torchscript':
        return benchmark_torchscript(model_path, name)
    elif suffix == '.onnx':
        return benchmark_onnx(model_path, name)
    else:
        # 尝试用 YOLO 加载
        try:
            return benchmark_yolo(model_path, name, val_data)
        except Exception as e:
            print(f"不支持的模型格式: {suffix}, 错误: {e}")
            return None


def print_table(results):
    """打印结果表格"""
    print(f"\n{'='*80}")
    print(" 测试结果汇总")
    print(f"{'='*80}")
    print(f"{'模型':<20} {'mAP@50':<10} {'mAP@50-95':<10} {'大小(MB)':<10} {'推理(ms)':<10} {'FPS':<10}")
    print("-" * 70)
    
    for r in results:
        map50_str = f"{r['map50']:.4f}" if r['map50'] >= 0 else "N/A"
        map50_95_str = f"{r['map50_95']:.4f}" if r['map50_95'] >= 0 else "N/A"
        print(f"{r['name']:<20} {map50_str:<10} {map50_95_str:<10} "
              f"{r['size_mb']:<10.2f} {r['inference_ms']:<10.2f} {r['fps']:<10.1f}")
    print("-" * 70)


def main():
    results = []
    
    for path, name in MODELS:
        if Path(path).exists():
            result = benchmark(path, name, VAL_DATA)
            if result:
                results.append(result)
        else:
            print(f"模型不存在: {path}")
    
    if results:
        print_table(results)


if __name__ == "__main__":
    main()
