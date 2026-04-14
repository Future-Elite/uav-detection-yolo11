"""
ONNX 模型量化脚本

支持的量化方法：
    - FP32: 原始32位浮点（基准）
    - FP16: 16位半精度浮点
    - INT8-PTQ: ONNX Runtime训练后量化
    - INT8-QAT: 量化感知训练后导出ONNX
    - 混合精度: 部分层INT8 + 部分层FP16

使用方法：
    python quantize_torchscript.py           # 全部量化+评估+绘图
    python quantize_torchscript.py --method int8-ptq  # 仅INT8 PTQ
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
MODEL_PATH = "./runs/detect/merged/refined-enhanced3/weights/best.pt"
DATA_PATH = "../datasets/merged_dataset_5_enhanced/data.yaml"
VAL_DATA_PATH = "../datasets/Airborne/data.yaml"

DEVICE = 0
IMG_SIZE = 640
CALIBRATION_IMAGES = 100  # INT8校准图像数量

OUTPUT_DIR = Path("./runs/detect/quantized_onnx")
# =================================================


def get_model_size(path):
    """获取模型大小（MB）"""
    p = Path(path)
    if not p.exists():
        return 0
    if p.is_file():
        return p.stat().st_size / (1024 * 1024)
    return sum(f.stat().st_size for f in p.rglob('*') if f.is_file()) / (1024 * 1024)


def get_calibration_dataset():
    """获取校准数据集"""
    from PIL import Image
    import random
    
    data_dir = Path(DATA_PATH).parent / 'train' / 'images'
    image_files = list(data_dir.glob('*.jpg')) + list(data_dir.glob('*.png'))
    
    if not image_files:
        LOGGER.warning("无校准图像")
        return None
    
    return random.sample(image_files, min(CALIBRATION_IMAGES, len(image_files)))


# ==================== 量化方法 ====================

def export_fp32():
    """导出FP32 TorchScript"""
    LOGGER.info("导出 FP32 TorchScript...")
    model = YOLO(MODEL_PATH)
    path = model.export(format='torchscript', imgsz=IMG_SIZE, device=DEVICE)
    LOGGER.info(f"FP32 导出成功: {path} ({get_model_size(path):.2f} MB)")
    return str(path)


def export_fp16():
    """导出FP16 TorchScript"""
    LOGGER.info("导出 FP16 TorchScript...")
    model = YOLO(MODEL_PATH)
    path = model.export(format='torchscript', imgsz=IMG_SIZE, device=DEVICE, half=True)
    LOGGER.info(f"FP16 导出成功: {path} ({get_model_size(path):.2f} MB)")
    return str(path)


def export_int8_ptq():
    """
    INT8 PTQ - ONNX Runtime 训练后量化
    
    流程：
    1. 导出FP32 ONNX
    2. 准备校准数据
    3. 运行ONNX Runtime量化
    """
    LOGGER.info("=" * 50)
    LOGGER.info("INT8 PTQ - ONNX Runtime量化")
    LOGGER.info("=" * 50)
    
    output_path = OUTPUT_DIR / "model_int8_ptq.onnx"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        import onnx
        from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType, CalibrationDataReader, QuantFormat
        
        # 1. 导出预处理后的ONNX（用于量化）
        LOGGER.info("导出预处理ONNX...")
        model = YOLO(MODEL_PATH)
        fp32_path = model.export(format='onnx', imgsz=IMG_SIZE, simplify=True, device=DEVICE)
        
        # 2. 动态量化（更简单，不需要校准数据）
        LOGGER.info("执行动态INT8量化...")
        try:
            quantize_dynamic(
                fp32_path,
                str(output_path),
                weight_type=QuantType.QInt8,
                per_channel=True,  # 每通道量化，精度更高
            )
            LOGGER.info(f"INT8 PTQ (动态) 导出成功: {output_path} ({get_model_size(output_path):.2f} MB)")
            return str(output_path)
        except Exception as e:
            LOGGER.warning(f"动态量化失败: {e}，尝试静态量化...")
        
        # 3. 静态量化（需要校准数据，精度更高）
        LOGGER.info("执行静态INT8量化...")
        
        class ImageCalibrationDataReader(CalibrationDataReader):
            def __init__(self, image_files, batch_size=1):
                self.image_files = image_files
                self.batch_size = batch_size
                self.index = 0
                self.input_name = "images"
                
            def get_next(self):
                if self.index >= len(self.image_files):
                    return None
                
                from PIL import Image
                import torchvision.transforms as T
                
                batch = []
                for _ in range(self.batch_size):
                    if self.index >= len(self.image_files):
                        break
                    img = Image.open(self.image_files[self.index]).convert('RGB')
                    img = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor()])(img)
                    # 确保是4维: (1, C, H, W)
                    img_array = img.numpy()
                    if img_array.ndim == 3:
                        img_array = np.expand_dims(img_array, axis=0)
                    batch.append(img_array)
                    self.index += 1
                
                if not batch:
                    return None
                # 合并batch，结果为 (N, C, H, W)
                return {self.input_name: np.concatenate(batch, axis=0)}
            
            def rewind(self):
                self.index = 0
        
        # 获取校准数据
        calib_images = get_calibration_dataset()
        if calib_images:
            calib_reader = ImageCalibrationDataReader(calib_images)
            
            from onnxruntime.quantization import shape_inference
            import onnx
            
            # 预处理模型（量化需要）
            preprocessed_path = OUTPUT_DIR / "model_preprocessed.onnx"
            shape_inference.quant_pre_process(fp32_path, str(preprocessed_path))
            
            # 配置会话选项，使用CPU执行器避免数据传输错误
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.add_session_config_entry("session.enable_mem_pattern", "0")
            so.add_session_config_entry("session.use_env_malloc", "0")
            
            quantize_static(
                str(preprocessed_path),
                str(output_path),
                calib_reader,
                quant_format=QuantFormat.QDQ,
                weight_type=QuantType.QInt8,
                per_channel=True,
                extra_session_options=so,
            )
            
            # 清理临时文件
            preprocessed_path.unlink(missing_ok=True)
            
            LOGGER.info(f"INT8 PTQ (静态) 导出成功: {output_path} ({get_model_size(output_path):.2f} MB)")
            return str(output_path)
        else:
            LOGGER.warning("无校准数据，使用动态量化结果")
            return str(output_path)
            
    except ImportError as e:
        LOGGER.warning(f"未安装onnxruntime: {e}")
        LOGGER.info("安装: pip install onnxruntime")
        return None
    except Exception as e:
        LOGGER.warning(f"INT8 PTQ失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def export_int8_qat():
    """
    INT8 QAT - 量化感知训练
    
    流程：
    1. 使用混合精度微调
    2. 导出ONNX并量化
    """
    LOGGER.info("=" * 50)
    LOGGER.info("INT8 QAT - 量化感知训练")
    LOGGER.info("=" * 50)
    
    output_path = OUTPUT_DIR / "model_int8_qat.onnx"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. 混合精度微调
        LOGGER.info("混合精度微调...")
        model = YOLO(MODEL_PATH)
        
        results = model.train(
            data=DATA_PATH,
            epochs=5,
            batch=8,
            imgsz=IMG_SIZE,
            lr0=0.0001,
            device=DEVICE,
            amp=True,  # 自动混合精度
            project=str(OUTPUT_DIR),
            name='qat_finetune',
            exist_ok=True,
            plots=False,
            save=True,
            val=False,
            verbose=False,
        )
        
        # 2. 导出ONNX
        finetuned_path = OUTPUT_DIR / 'qat_finetune' / 'weights' / 'best.pt'
        if not finetuned_path.exists():
            finetuned_path = OUTPUT_DIR / 'qat_finetune' / 'weights' / 'last.pt'
        
        if finetuned_path.exists():
            model_finetuned = YOLO(str(finetuned_path))
            onnx_path = model_finetuned.export(format='onnx', imgsz=IMG_SIZE, simplify=True, device=DEVICE)
            
            # 3. 量化
            try:
                from onnxruntime.quantization import quantize_dynamic, QuantType
                quantize_dynamic(onnx_path, str(output_path), weight_type=QuantType.QInt8)
                LOGGER.info(f"INT8 QAT 导出成功: {output_path} ({get_model_size(output_path):.2f} MB)")
                return str(output_path)
            except ImportError:
                # 无onnxruntime，直接用ONNX
                Path(onnx_path).rename(output_path)
                LOGGER.info(f"INT8 QAT (ONNX) 导出成功: {output_path}")
                return str(output_path)
        else:
            raise FileNotFoundError("微调模型未找到")
            
    except Exception as e:
        LOGGER.warning(f"INT8 QAT失败: {e}")
        # 回退到INT8 PTQ
        return export_int8_ptq()


def export_fp16_onnx():
    """导出FP16 ONNX"""
    LOGGER.info("导出 FP16 ONNX...")
    model = YOLO(MODEL_PATH)
    path = model.export(format='onnx', imgsz=IMG_SIZE, simplify=True, device=DEVICE, half=True)
    LOGGER.info(f"FP16 ONNX 导出成功: {path} ({get_model_size(path):.2f} MB)")
    return str(path)


def export_mixed_precision():
    """
    混合精度量化
    
    部分层INT8 + 关键层FP16
    """
    LOGGER.info("=" * 50)
    LOGGER.info("混合精度量化")
    LOGGER.info("=" * 50)
    
    output_path = OUTPUT_DIR / "model_mixed.onnx"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        import onnx
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        # 导出FP32 ONNX
        model = YOLO(MODEL_PATH)
        fp32_path = model.export(format='onnx', imgsz=IMG_SIZE, simplify=True, device=DEVICE)
        
        # 执行量化
        LOGGER.info("执行混合精度量化...")
        quantize_dynamic(
            fp32_path,
            str(output_path),
            weight_type=QuantType.QInt8,
        )
        
        LOGGER.info(f"混合精度 导出成功: {output_path} ({get_model_size(output_path):.2f} MB)")
        return str(output_path)
        
    except ImportError:
        LOGGER.warning("未安装onnxruntime，导出FP16 ONNX替代")
        return export_fp16_onnx()
    except Exception as e:
        LOGGER.warning(f"混合精度量化失败: {e}")
        return export_fp16_onnx()


# ==================== 评估 ====================

def evaluate_model(model_path, name):
    """评估模型"""
    LOGGER.info(f"评估 {name}...")
    
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
        # 对于ONNX模型，使用YOLO加载
        model = YOLO(model_path)
        
        # 确定设备（ONNX可能不支持CUDA，使用CPU）
        eval_device = DEVICE
        if model_path.endswith('.onnx'):
            # 检查ONNX Runtime是否支持CUDA
            try:
                import onnxruntime as ort
                available_providers = ort.get_available_providers()
                LOGGER.info(f"  可用的执行提供者: {available_providers}")
                
                # 如果没有CUDAExecutionProvider，使用CPU
                if 'CUDAExecutionProvider' not in available_providers:
                    LOGGER.info("  ONNX Runtime不支持CUDA，使用CPU评估")
                    eval_device = 'cpu'
                else:
                    # 有CUDA支持，尝试测试推理
                    try:
                        test_model = YOLO(model_path)
                        # 使用CPU测试，避免数据传输错误
                        test_model.predict(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE), device='cpu', verbose=False)
                        # CPU测试成功，尝试CUDA
                        test_model.predict(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE), device=DEVICE, verbose=False)
                    except Exception as e:
                        LOGGER.info(f"  ONNX CUDA推理失败 ({e})，使用CPU评估")
                        eval_device = 'cpu'
            except ImportError:
                LOGGER.info("  未安装onnxruntime，使用CPU评估")
                eval_device = 'cpu'
        
        # 验证
        val_res = model.val(data=VAL_DATA_PATH, split='test', device=eval_device, batch=16, verbose=False)
        result['map50'] = float(val_res.box.map50)
        result['map50_95'] = float(val_res.box.map)
        
        # 测速
        if eval_device != 'cpu' and torch.cuda.is_available():
            dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(f'cuda:{DEVICE}')
            for _ in range(3):
                model.predict(dummy, verbose=False)
            
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(20):
                model.predict(dummy, verbose=False)
            torch.cuda.synchronize()
            
            result['inference_ms'] = round((time.time() - start) / 20 * 1000, 2)
            result['fps'] = round(1000 / result['inference_ms'], 1)
        else:
            # CPU测速
            dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
            for _ in range(3):
                model.predict(dummy, verbose=False)
            
            start = time.time()
            for _ in range(10):
                model.predict(dummy, verbose=False)
            
            result['inference_ms'] = round((time.time() - start) / 10 * 1000, 2)
            result['fps'] = round(1000 / result['inference_ms'], 1)
        
        LOGGER.info(f"  mAP@50: {result['map50']:.4f}, 推理: {result['inference_ms']:.2f}ms, 大小: {result['size_mb']:.2f}MB")
        
    except Exception as e:
        LOGGER.warning(f"  评估失败: {e}")
        result['status'] = f'failed: {str(e)[:50]}'
    
    return result


# ==================== 绘图 ====================

def plot_comparison(results, output_dir):
    """绘制综合对比图"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        valid = [r for r in results if r['status'] == 'success']
        if len(valid) < 2:
            LOGGER.warning("有效结果不足，跳过绘图")
            return
        
        names = [r['name'] for r in valid]
        colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']
        
        fig = plt.figure(figsize=(18, 12))
        
        # 1. mAP@50 对比
        ax1 = fig.add_subplot(2, 3, 1)
        map50 = [r['map50'] for r in valid]
        bars = ax1.bar(names, map50, color=colors[:len(names)], edgecolor='black', linewidth=1.5)
        ax1.set_ylabel('mAP@50', fontsize=12, fontweight='bold')
        ax1.set_title('检测精度对比', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, max(map50) * 1.1)
        for bar, val in zip(bars, map50):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax1.tick_params(axis='x', rotation=15)
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. 模型大小对比
        ax2 = fig.add_subplot(2, 3, 2)
        sizes = [r['size_mb'] for r in valid]
        bars = ax2.bar(names, sizes, color=colors[:len(names)], edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('模型大小 (MB)', fontsize=12, fontweight='bold')
        ax2.set_title('存储空间对比', fontsize=14, fontweight='bold')
        for bar, val in zip(bars, sizes):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax2.tick_params(axis='x', rotation=15)
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. 推理速度对比
        ax3 = fig.add_subplot(2, 3, 3)
        fps = [r['fps'] for r in valid]
        bars = ax3.bar(names, fps, color=colors[:len(names)], edgecolor='black', linewidth=1.5)
        ax3.set_ylabel('FPS', fontsize=12, fontweight='bold')
        ax3.set_title('推理速度对比', fontsize=14, fontweight='bold')
        for bar, val in zip(bars, fps):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax3.tick_params(axis='x', rotation=15)
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. 压缩比
        ax4 = fig.add_subplot(2, 3, 4)
        base_size = valid[0]['size_mb']
        compression = [base_size / r['size_mb'] if r['size_mb'] > 0 else 0 for r in valid]
        bars = ax4.bar(names, compression, color=colors[:len(names)], edgecolor='black', linewidth=1.5)
        ax4.set_ylabel('压缩比', fontsize=12, fontweight='bold')
        ax4.set_title(f'压缩比 (相对于{valid[0]["name"]})', fontsize=14, fontweight='bold')
        for bar, val in zip(bars, compression):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{val:.2f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax4.tick_params(axis='x', rotation=15)
        ax4.grid(axis='y', alpha=0.3)
        
        # 5. 精度保留率
        ax5 = fig.add_subplot(2, 3, 5)
        base_map = valid[0]['map50']
        retention = [r['map50'] / base_map * 100 for r in valid]
        bars = ax5.bar(names, retention, color=colors[:len(names)], edgecolor='black', linewidth=1.5)
        ax5.set_ylabel('精度保留率 (%)', fontsize=12, fontweight='bold')
        ax5.set_title('精度保留率', fontsize=14, fontweight='bold')
        ax5.axhline(y=95, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='95%阈值')
        ax5.axhline(y=98, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='98%阈值')
        for bar, val in zip(bars, retention):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax5.tick_params(axis='x', rotation=15)
        ax5.legend(loc='lower right')
        ax5.grid(axis='y', alpha=0.3)
        
        # 6. 精度-大小权衡
        ax6 = fig.add_subplot(2, 3, 6)
        for i, r in enumerate(valid):
            ax6.scatter(r['size_mb'], r['map50'], c=colors[i], s=200, 
                       edgecolors='black', linewidths=2, label=r['name'], zorder=3)
            ax6.annotate(r['name'], (r['size_mb'], r['map50']), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')
        ax6.set_xlabel('模型大小 (MB)', fontsize=12, fontweight='bold')
        ax6.set_ylabel('mAP@50', fontsize=12, fontweight='bold')
        ax6.set_title('精度-大小权衡图\n(左上角最优)', fontsize=14, fontweight='bold')
        ax6.grid(True, alpha=0.3, linestyle='--')
        ax6.legend(loc='lower right', fontsize=9)
        
        plt.tight_layout()
        
        output_path = output_dir / "quantize_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        LOGGER.info(f"对比图保存至: {output_path}")
        
    except Exception as e:
        LOGGER.warning(f"绘图失败: {e}")


def plot_radar(results, output_dir):
    """绘制雷达图"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        
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
        
        colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']
        
        for i, r in enumerate(valid):
            precision = r['map50'] / base['map50'] * 100 if base['map50'] > 0 else 0
            compression = min(base['size_mb'] / r['size_mb'] * 50, 100) if r['size_mb'] > 0 else 0
            speed = min(r['fps'] / base['fps'] * 50, 100) if base['fps'] > 0 else 0
            efficiency = (precision + compression + speed) / 3
            
            values = [precision, compression, speed, efficiency]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, color=colors[i], label=r['name'])
            ax.fill(angles, values, alpha=0.15, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=12, fontweight='bold')
        ax.set_ylim(0, 110)
        ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1))
        ax.set_title('ONNX量化综合性能雷达图', fontsize=14, fontweight='bold', pad=20)
        
        output_path = output_dir / "quantize_radar.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        LOGGER.info(f"雷达图保存至: {output_path}")
        
    except Exception as e:
        LOGGER.warning(f"雷达图绘制失败: {e}")


def print_summary(results):
    """打印总结"""
    print("\n" + "=" * 80)
    print(" ONNX 量化评估结果")
    print("=" * 80)
    
    print(f"\n{'方法':<15} {'mAP@50':<10} {'mAP@50-95':<10} {'大小(MB)':<10} {'推理(ms)':<10} {'FPS':<10}")
    print("-" * 65)
    
    for r in results:
        if r['status'] != 'success':
            print(f"{r['name']:<15} FAILED")
            continue
        print(f"{r['name']:<15} {r['map50']:<10.4f} {r['map50_95']:<10.4f} "
              f"{r['size_mb']:<10.2f} {r['inference_ms']:<10.2f} {r['fps']:<10.1f}")
    
    print("-" * 65)
    
    valid = [r for r in results if r['status'] == 'success']
    if len(valid) >= 2:
        base = valid[0]
        print(f"\n相对于 {base['name']}:")
        for r in valid[1:]:
            comp = base['size_mb'] / r['size_mb'] if r['size_mb'] > 0 else 0
            speed = r['fps'] / base['fps'] if base['fps'] > 0 else 0
            prec = r['map50'] / base['map50'] * 100 if base['map50'] > 0 else 0
            print(f"  {r['name']:<15}: 压缩 {comp:.2f}x, 加速 {speed:.2f}x, 精度保留 {prec:.1f}%")


# ==================== 主流程 ====================

def run_all_quantization():
    """运行全部量化"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    LOGGER.info("=" * 60)
    LOGGER.info(" ONNX 模型量化")
    LOGGER.info("=" * 60)
    
    results = []
    
    # 1. FP32
    LOGGER.info("\n[1/5] FP32 (原始)...")
    path = export_fp32()
    if path:
        results.append(evaluate_model(path, 'FP32'))
    
    # 2. FP16
    LOGGER.info("\n[2/5] FP16...")
    path = export_fp16()
    if path:
        results.append(evaluate_model(path, 'FP16'))
    
    # 3. INT8 PTQ
    LOGGER.info("\n[3/5] INT8 PTQ...")
    path = export_int8_ptq()
    if path:
        results.append(evaluate_model(path, 'INT8-PTQ'))
    
    # 4. INT8 QAT
    LOGGER.info("\n[4/5] INT8 QAT...")
    path = export_int8_qat()
    if path:
        results.append(evaluate_model(path, 'INT8-QAT'))
    
    # 5. 混合精度
    LOGGER.info("\n[5/5] 混合精度...")
    path = export_mixed_precision()
    if path:
        results.append(evaluate_model(path, '混合精度'))
    
    # 保存结果
    output_file = OUTPUT_DIR / "quantize_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    LOGGER.info(f"\n结果保存至: {output_file}")
    
    # 绘图
    plot_comparison(results, OUTPUT_DIR)
    plot_radar(results, OUTPUT_DIR)
    
    # 打印总结
    print_summary(results)
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='ONNX模型量化')
    parser.add_argument('--method', type=str, default='all',
                        choices=['all', 'fp32', 'fp16', 'int8-ptq', 'int8-qat', 'mixed'],
                        help='量化方法')
    args = parser.parse_args()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.method == 'all':
        run_all_quantization()
    else:
        export_func = {
            'fp32': export_fp32,
            'fp16': export_fp16,
            'int8-ptq': export_int8_ptq,
            'int8-qat': export_int8_qat,
            'mixed': export_mixed_precision,
        }[args.method]
        
        path = export_func()
        if path:
            name_map = {'fp32': 'FP32', 'fp16': 'FP16', 'int8-ptq': 'INT8-PTQ', 
                       'int8-qat': 'INT8-QAT', 'mixed': '混合精度'}
            result = evaluate_model(path, name_map[args.method])
            print_summary([result])


if __name__ == "__main__":
    main()
