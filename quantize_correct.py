"""
YOLO模型量化脚本 — FP32 / FP16 / INT8 对比评估

量化方式：
    FP32  : 原始全精度 ONNX（基线）
    FP16  : 半精度 ONNX
    INT8  : 训练后动态 INT8 量化 ONNX（onnxruntime）

使用方法：
    1. 安装依赖: pip install onnxruntime onnx matplotlib torchvision
    2. 在下方配置模型和数据集路径
    3. 运行:
       python quantize_correct.py                # 全流程：量化 + 评估 + 绘图
       python quantize_correct.py --method quant # 仅量化
       python quantize_correct.py --method eval  # 仅评估已有模型
       python quantize_correct.py --method plot  # 仅绘图（需先完成评估）
"""

import os
import sys
import warnings
import json
import argparse
import time
from pathlib import Path

import torch
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from ultralytics import YOLO

warnings.filterwarnings("ignore")

# ==================== 配置 ====================
MODEL_PATH = "./runs/detect/merged/refined-enhanced3/weights/best.pt"
DATA_PATH = "../datasets/merged_dataset_5_enhanced/data.yaml"
VAL_DATA_PATH = "../datasets/Airborne/data.yaml"

OUTPUT_DIR = Path("./runs/detect/quantized_correct")
IMG_SIZE = 640
DEVICE = "0"
BATCH_SIZE = 16
NUM_CALIB_BATCH = 32
# ==============================================


def get_model_size(path):
    p = Path(path)
    if not p.exists():
        return 0
    if p.is_file():
        return p.stat().st_size / (1024 * 1024)
    return sum(f.stat().st_size for f in p.rglob('*') if f.is_file()) / (1024 * 1024)


# ==================== 量化 ====================

def quantize_fp32():
    print("\n" + "=" * 60)
    print(" [1/3] 导出 FP32 ONNX")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(MODEL_PATH)
    fp32_path = model.export(format='onnx', imgsz=IMG_SIZE, simplify=True, device='cpu', half=False)
    print(f"FP32 ONNX: {fp32_path} ({get_model_size(fp32_path):.2f} MB)")
    return str(fp32_path)


def quantize_fp16():
    print("\n" + "=" * 60)
    print(" [2/3] 导出 FP16 ONNX")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(MODEL_PATH)
    fp16_path = model.export(format='onnx', imgsz=IMG_SIZE, simplify=True, device=DEVICE, half=True)
    print(f"FP16 ONNX: {fp16_path} ({get_model_size(fp16_path):.2f} MB)")
    return str(fp16_path)


def quantize_int8(fp32_path):
    print("\n" + "=" * 60)
    print(" [3/3] INT8 动态量化")
    print("=" * 60)

    from onnxruntime.quantization import quantize_dynamic, QuantType

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    int8_path = str(OUTPUT_DIR / "model_int8_dynamic.onnx")

    try:
        quantize_dynamic(
            fp32_path,
            int8_path,
            weight_type=QuantType.QInt8,
            op_types_to_quantize=['Conv', 'MatMul'],
        )
        print(f"INT8 ONNX: {int8_path} ({get_model_size(int8_path):.2f} MB)")
    except Exception as e:
        print(f"动态量化失败 (Cast节点冲突): {e}")
        print("尝试使用 per_channel=False + 权重量化方式...")
        try:
            import onnx
            from onnxruntime.quantization.shape_inference import quant_pre_process
            preprocessed = str(OUTPUT_DIR / "model_int8_prep.onnx")
            quant_pre_process(fp32_path, preprocessed)
            quantize_dynamic(
                preprocessed,
                int8_path,
                weight_type=QuantType.QInt8,
                op_types_to_quantize=['Conv', 'MatMul'],
            )
            Path(preprocessed).unlink(missing_ok=True)
            print(f"INT8 ONNX (fallback): {int8_path} ({get_model_size(int8_path):.2f} MB)")
        except Exception as e2:
            print(f"INT8 量化完全失败: {e2}")
            int8_path = None

    return int8_path


def run_quantization():
    fp32_path = quantize_fp32()
    fp16_path = quantize_fp16()
    int8_path = quantize_int8(fp32_path)
    return {
        'FP32': fp32_path,
        'FP16': fp16_path,
        'INT8': int8_path,
    }


# ==================== 评估 ====================

def evaluate_pt_model(model_path, name):
    result = {
        'name': name,
        'path': str(model_path),
        'size_mb': get_model_size(model_path),
        'map50': -1,
        'map50_95': -1,
        'inference_ms': 0,
        'fps': 0,
        'status': 'success',
    }

    try:
        model = YOLO(model_path)
        val_res = model.val(data=VAL_DATA_PATH, split='test', device=DEVICE, batch=BATCH_SIZE, verbose=False)
        result['map50'] = float(val_res.box.map50)
        result['map50_95'] = float(val_res.box.map)

        dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(f'cuda:{DEVICE}')
        for _ in range(5):
            model.predict(dummy, verbose=False)
        torch.cuda.synchronize()

        start = time.time()
        for _ in range(30):
            model.predict(dummy, verbose=False)
        torch.cuda.synchronize()
        result['inference_ms'] = (time.time() - start) / 30 * 1000
        result['fps'] = 1000 / result['inference_ms']

        print(f"  {name}: mAP@50={result['map50']:.4f}, mAP@50-95={result['map50_95']:.4f}, "
              f"{result['inference_ms']:.2f}ms, {result['fps']:.1f}FPS, {result['size_mb']:.2f}MB")
    except Exception as e:
        print(f"  {name} 评估失败: {e}")
        result['status'] = f'failed: {e}'

    return result


def evaluate_onnx_model(model_path, name):
    import onnxruntime as ort

    result = {
        'name': name,
        'path': str(model_path),
        'size_mb': get_model_size(model_path),
        'map50': -1,
        'map50_95': -1,
        'inference_ms': 0,
        'fps': 0,
        'status': 'success',
    }

    try:
        providers = ort.get_available_providers()
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        if 'CUDAExecutionProvider' in providers:
            session = ort.InferenceSession(model_path, sess_opts, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        else:
            session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

        input_name = session.get_inputs()[0].name
        input_type = session.get_inputs()[0].type

        if 'float16' in input_type:
            dummy = np.random.randn(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float16)
        else:
            dummy = np.random.randn(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)

        for _ in range(5):
            session.run(None, {input_name: dummy})

        start = time.time()
        for _ in range(30):
            session.run(None, {input_name: dummy})
        result['inference_ms'] = (time.time() - start) / 30 * 1000
        result['fps'] = 1000 / result['inference_ms']

        print(f"  {name}: {result['inference_ms']:.2f}ms, {result['fps']:.1f}FPS, {result['size_mb']:.2f}MB")
    except Exception as e:
        print(f"  {name} 评估失败: {e}")
        result['status'] = f'failed: {e}'

    return result


def evaluate_all(model_paths):
    print("\n" + "=" * 60)
    print(" 模型评估")
    print("=" * 60)

    results = []
    for name, path in model_paths.items():
        if path is None or not Path(path).exists():
            print(f"\n{name}: 文件不存在，跳过")
            continue

        print(f"\n评估 {name}...")
        if str(path).endswith('.pt'):
            results.append(evaluate_pt_model(path, name))
        else:
            results.append(evaluate_onnx_model(path, name))

    return results


# ==================== 绘图 ====================

def plot_comparison(results, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    valid = [r for r in results if r['status'] == 'success']
    if len(valid) < 2:
        print("有效结果不足2个，跳过绘图")
        return

    names = [r['name'] for r in valid]
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6', '#1abc9c']

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('模型量化性能对比', fontsize=18, fontweight='bold', y=0.98)

    # ---- 1. mAP@50 ----
    ax1 = fig.add_subplot(2, 3, 1)
    map50 = [r['map50'] for r in valid if r['map50'] >= 0]
    map50_names = [r['name'] for r in valid if r['map50'] >= 0]
    if map50:
        bars = ax1.bar(map50_names, map50, color=colors[:len(map50)], edgecolor='black', linewidth=1.5)
        ax1.set_ylabel('mAP@50', fontsize=12, fontweight='bold')
        ax1.set_title('检测精度对比 (mAP@50)', fontsize=14, fontweight='bold')
        ax1.set_ylim(min(map50) * 0.95, max(map50) * 1.03)
        for bar, val in zip(bars, map50):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                     f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax1.tick_params(axis='x', rotation=15)
    ax1.grid(axis='y', alpha=0.3)

    # ---- 2. mAP@50-95 ----
    ax2 = fig.add_subplot(2, 3, 2)
    map50_95 = [r['map50_95'] for r in valid if r['map50_95'] >= 0]
    map50_95_names = [r['name'] for r in valid if r['map50_95'] >= 0]
    if map50_95:
        bars = ax2.bar(map50_95_names, map50_95, color=colors[:len(map50_95)], edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('mAP@50-95', fontsize=12, fontweight='bold')
        ax2.set_title('检测精度对比 (mAP@50-95)', fontsize=14, fontweight='bold')
        ax2.set_ylim(min(map50_95) * 0.95, max(map50_95) * 1.03)
        for bar, val in zip(bars, map50_95):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                     f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2.tick_params(axis='x', rotation=15)
    ax2.grid(axis='y', alpha=0.3)

    # ---- 3. 模型大小 ----
    ax3 = fig.add_subplot(2, 3, 3)
    sizes = [r['size_mb'] for r in valid]
    bars = ax3.bar(names, sizes, color=colors[:len(valid)], edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('模型大小 (MB)', fontsize=12, fontweight='bold')
    ax3.set_title('存储空间对比', fontsize=14, fontweight='bold')
    for bar, val in zip(bars, sizes):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax3.tick_params(axis='x', rotation=15)
    ax3.grid(axis='y', alpha=0.3)

    # ---- 4. 推理速度 ----
    ax4 = fig.add_subplot(2, 3, 4)
    fps_vals = [r['fps'] for r in valid if r['fps'] > 0]
    fps_names = [r['name'] for r in valid if r['fps'] > 0]
    if fps_vals:
        bars = ax4.bar(fps_names, fps_vals, color=colors[:len(fps_vals)], edgecolor='black', linewidth=1.5)
        ax4.set_ylabel('FPS', fontsize=12, fontweight='bold')
        ax4.set_title('推理速度对比 (FPS)', fontsize=14, fontweight='bold')
        for bar, val in zip(bars, fps_vals):
            ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax4.tick_params(axis='x', rotation=15)
    ax4.grid(axis='y', alpha=0.3)

    # ---- 5. 压缩比 + 精度保留率 ----
    ax5 = fig.add_subplot(2, 3, 5)
    base_size = valid[0]['size_mb']
    base_map = valid[0]['map50'] if valid[0]['map50'] >= 0 else 1

    compression = [base_size / r['size_mb'] if r['size_mb'] > 0 else 0 for r in valid]
    retention = [r['map50'] / base_map * 100 if r['map50'] >= 0 and base_map > 0 else 0 for r in valid]

    x = np.arange(len(names))
    width = 0.35
    bars_c = ax5.bar(x - width / 2, compression, width, label='压缩比', color='#e74c3c', edgecolor='black', linewidth=1)
    bars_r = ax5.bar(x + width / 2, retention, width, label='精度保留率(%)', color='#2ecc71', edgecolor='black', linewidth=1)

    for bar, val in zip(bars_c, compression):
        ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f'{val:.2f}x', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar, val in zip(bars_r, retention):
        if val > 0:
            ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax5.set_xticks(x)
    ax5.set_xticklabels(names, rotation=15)
    ax5.set_title('压缩比 & 精度保留率', fontsize=14, fontweight='bold')
    ax5.legend(loc='upper right')
    ax5.grid(axis='y', alpha=0.3)

    # ---- 6. 精度-大小权衡散点 ----
    ax6 = fig.add_subplot(2, 3, 6)
    for i, r in enumerate(valid):
        if r['map50'] >= 0:
            ax6.scatter(r['size_mb'], r['map50'], c=colors[i % len(colors)], s=250,
                        edgecolors='black', linewidths=2, label=r['name'], zorder=3)
            ax6.annotate(r['name'], (r['size_mb'], r['map50']),
                         xytext=(8, 8), textcoords='offset points', fontsize=10, fontweight='bold')
    ax6.set_xlabel('模型大小 (MB)', fontsize=12, fontweight='bold')
    ax6.set_ylabel('mAP@50', fontsize=12, fontweight='bold')
    ax6.set_title('精度-大小权衡图\n(左上角最优)', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3, linestyle='--')
    ax6.legend(loc='lower right', fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = output_dir / "quantize_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"对比图保存至: {out_path}")


def plot_radar(results, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    valid = [r for r in results if r['status'] == 'success']
    if len(valid) < 2:
        return

    base = valid[0]
    metrics = ['精度', '压缩', '速度', '效率']

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6', '#1abc9c']

    for i, r in enumerate(valid):
        precision = r['map50'] / base['map50'] * 100 if base['map50'] > 0 else 0
        compression = min(base['size_mb'] / r['size_mb'] * 50, 100) if r['size_mb'] > 0 else 0
        speed = min(r['fps'] / base['fps'] * 50, 100) if base['fps'] > 0 and r['fps'] > 0 else 0
        efficiency = (precision + compression + speed) / 3

        values = [precision, compression, speed, efficiency]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, color=colors[i % len(colors)], label=r['name'])
        ax.fill(angles, values, alpha=0.15, color=colors[i % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1))
    ax.set_title('量化综合性能雷达图', fontsize=14, fontweight='bold', pad=20)

    out_path = output_dir / "quantize_radar.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"雷达图保存至: {out_path}")


def plot_speed_accuracy(results, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    valid = [r for r in results if r['status'] == 'success' and r['map50'] >= 0]
    if len(valid) < 2:
        return

    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6', '#1abc9c']

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, r in enumerate(valid):
        ax.scatter(r['inference_ms'], r['map50'], c=colors[i % len(colors)], s=300,
                   edgecolors='black', linewidths=2, label=r['name'], zorder=3)
        ax.annotate(r['name'], (r['inference_ms'], r['map50']),
                    xytext=(8, 8), textcoords='offset points', fontsize=11, fontweight='bold')

    ax.set_xlabel('推理时间 (ms)', fontsize=14, fontweight='bold')
    ax.set_ylabel('mAP@50', fontsize=14, fontweight='bold')
    ax.set_title('精度-速度权衡图\n(左上角最优: 高精度+低延迟)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower left', fontsize=10)

    plt.tight_layout()
    out_path = output_dir / "speed_accuracy_tradeoff.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"精度-速度图保存至: {out_path}")


def generate_plots(results):
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_comparison(results, output_dir)
    plot_radar(results, output_dir)
    plot_speed_accuracy(results, output_dir)


# ==================== 汇总 ====================

def print_summary(results):
    print("\n" + "=" * 80)
    print(" 量化结果汇总")
    print("=" * 80)
    print(f"{'模型':<15} {'mAP@50':<10} {'mAP@50-95':<10} {'大小(MB)':<10} {'推理(ms)':<10} {'FPS':<10}")
    print("-" * 65)

    for r in results:
        if r['status'] != 'success':
            print(f"{r['name']:<15} FAILED")
            continue
        map50_str = f"{r['map50']:.4f}" if r['map50'] >= 0 else "N/A"
        map50_95_str = f"{r['map50_95']:.4f}" if r['map50_95'] >= 0 else "N/A"
        print(f"{r['name']:<15} {map50_str:<10} {map50_95_str:<10} "
              f"{r['size_mb']:<10.2f} {r['inference_ms']:<10.2f} {r['fps']:<10.1f}")
    print("-" * 65)

    valid = [r for r in results if r['status'] == 'success']
    if len(valid) >= 2:
        base = valid[0]
        print(f"\n相对于 {base['name']}:")
        for r in valid[1:]:
            comp = base['size_mb'] / r['size_mb'] if r['size_mb'] > 0 else 0
            speed = r['fps'] / base['fps'] if base['fps'] > 0 and r['fps'] > 0 else 0
            prec = r['map50'] / base['map50'] * 100 if base['map50'] > 0 and r['map50'] >= 0 else 0
            print(f"  {r['name']:<15}: 压缩 {comp:.2f}x, 加速 {speed:.2f}x, 精度保留 {prec:.1f}%")

    output_file = OUTPUT_DIR / "quantize_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存至: {output_file}")


# ==================== 主流程 ====================

def main():
    parser = argparse.ArgumentParser(description='YOLO模型量化 (FP32/FP16/INT8)')
    parser.add_argument('--method', type=str, default='all',
                        choices=['all', 'quant', 'eval', 'plot'],
                        help='all=量化+评估+绘图, quant=仅量化, eval=仅评估, plot=仅绘图')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results_json = OUTPUT_DIR / "quantize_results.json"

    if args.method in ['all', 'quant']:
        model_paths = run_quantization()
        model_paths['FP32-PyTorch'] = MODEL_PATH

        if args.method == 'quant':
            return

        results = evaluate_all(model_paths)
        print_summary(results)
        generate_plots(results)

    elif args.method == 'eval':
        if not results_json.exists():
            print(f"未找到评估结果文件: {results_json}，请先运行量化流程")
            return

        with open(results_json, 'r', encoding='utf-8') as f:
            saved = json.load(f)

        model_paths = {}
        for r in saved:
            if Path(r['path']).exists():
                model_paths[r['name']] = r['path']

        model_paths.setdefault('FP32-PyTorch', MODEL_PATH)
        results = evaluate_all(model_paths)
        print_summary(results)
        generate_plots(results)

    elif args.method == 'plot':
        if not results_json.exists():
            print(f"未找到评估结果文件: {results_json}，请先运行评估流程")
            return

        with open(results_json, 'r', encoding='utf-8') as f:
            results = json.load(f)

        generate_plots(results)


if __name__ == "__main__":
    main()
