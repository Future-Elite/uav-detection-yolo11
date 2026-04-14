"""
ONNX模型预测脚本

使用方法：
    python predict_onnx.py

支持：
    - 单张图像预测
    - 目录批量预测
    - 性能基准测试
"""

import os
import sys
import warnings
import time
import json
from pathlib import Path

import numpy as np
import cv2

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

warnings.filterwarnings("ignore")

# ==================== 配置 ====================
# 使用 FP32 ONNX 模型（INT8 量化模型不支持 ConvInteger 操作）
MODEL_PATH = "./runs/detect/merged/refined-enhanced3/weights/best.onnx"
DATA_PATH = "../datasets/Airborne/test/images"
OUTPUT_DIR = Path("./runs/detect/onnx_predictions")

IMG_SIZE = 640
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# 类别名称
CLASS_NAMES = ['plane', 'bird', 'drone', 'helicopter']

# 颜色列表（BGR格式）
COLORS = [
    (56, 56, 255),    # plane - 红色
    (255, 178, 102),  # bird - 天蓝色
    (102, 255, 102),  # drone - 绿色
    (255, 102, 255),  # helicopter - 粉色
]
# =============================================


def get_model_size(path):
    """获取模型大小（MB）"""
    p = Path(path)
    return p.stat().st_size / (1024 * 1024)


# ==================== ONNX预测器 ====================

class ONNXPredictor:
    """ONNX模型预测器"""
    
    def __init__(self, model_path):
        import onnxruntime as ort
        
        self.model_path = model_path
        self.img_size = IMG_SIZE
        self.class_names = CLASS_NAMES
        self.colors = COLORS
        
        # 创建ONNX会话（使用CPU）
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        print(f"可用执行提供器: {ort.get_available_providers()}")
        
        self.session = ort.InferenceSession(model_path, sess_options, providers=['CPUExecutionProvider'])
        print("使用 CPU 推理")
        
        # 获取输入输出信息
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        print(f"输入: {self.input_name}")
        print(f"输出: {self.output_names}")
    
    def preprocess(self, image):
        """图像预处理"""
        self.orig_shape = image.shape[:2]
        
        h, w = image.shape[:2]
        r = min(self.img_size / h, self.img_size / w)
        new_h, new_w = int(h * r), int(w * r)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        pad_h = (self.img_size - new_h) // 2
        pad_w = (self.img_size - new_w) // 2
        
        padded = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
        
        input_tensor = padded[:, :, ::-1].transpose(2, 0, 1)
        input_tensor = input_tensor.astype(np.float32) / 255.0
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        return input_tensor, r, (pad_w, pad_h)
    
    def postprocess(self, outputs, ratio, pad, conf_thres, iou_thres):
        """后处理：解码输出，应用NMS"""
        predictions = outputs[0]
        
        if predictions.shape[1] > predictions.shape[2]:
            predictions = predictions.transpose(0, 2, 1)
        
        predictions = predictions[0]
        
        # 过滤低置信度
        scores = predictions[:, 4] * predictions[:, 5:].max(axis=1)
        mask = scores > conf_thres
        predictions = predictions[mask]
        
        if len(predictions) == 0:
            return []
        
        boxes = predictions[:, :4]
        objectness = predictions[:, 4]
        class_probs = predictions[:, 5:]
        
        class_ids = np.argmax(class_probs, axis=1)
        confidences = objectness * class_probs[np.arange(len(class_ids)), class_ids]
        
        # 转换边界框
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        
        pad_w, pad_h = pad
        x1 = (x1 - pad_w) / ratio
        y1 = (y1 - pad_h) / ratio
        x2 = (x2 - pad_w) / ratio
        y2 = (y2 - pad_h) / ratio
        
        orig_h, orig_w = self.orig_shape
        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)
        
        # NMS
        boxes_nms = np.stack([x1, y1, x2, y2], axis=1)
        indices = nms(boxes_nms, confidences, iou_thres)
        
        detections = []
        for i in indices:
            detections.append({
                'bbox': [float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])],
                'confidence': float(confidences[i]),
                'class_id': int(class_ids[i]),
                'class_name': self.class_names[class_ids[i]]
            })
        
        return detections
    
    def predict(self, image, conf_thres=CONF_THRESHOLD, iou_thres=IOU_THRESHOLD):
        """预测单张图像"""
        input_tensor, ratio, pad = self.preprocess(image)
        
        start = time.time()
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        inference_time = (time.time() - start) * 1000
        
        detections = self.postprocess(outputs, ratio, pad, conf_thres, iou_thres)
        
        return detections, inference_time
    
    def draw_detections(self, image, detections):
        """绘制检测结果"""
        annotated = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            conf = det['confidence']
            class_id = det['class_id']
            class_name = det['class_name']
            
            color = self.colors[class_id % len(self.colors)]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            label = f"{class_name} {conf:.2f}"
            (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(annotated, (x1, y1 - label_h - baseline - 5), (x1 + label_w, y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return annotated


def nms(boxes, scores, iou_threshold):
    """非极大值抑制"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return keep


# ==================== 预测功能 ====================

def predict_directory(source_dir, model_path=None, output_dir=None, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD):
    """预测目录中所有图像"""
    from tqdm import tqdm
    
    model_path = model_path or MODEL_PATH
    source_dir = Path(source_dir)
    output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(" 目录批量预测")
    print("=" * 60)
    print(f"模型: {model_path}")
    print(f"源目录: {source_dir}")
    print(f"输出目录: {output_dir}")
    
    # 查找图像
    exts = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    image_files = []
    for ext in exts:
        image_files.extend(source_dir.glob(f'*{ext}'))
        image_files.extend(source_dir.glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"未找到图像文件: {source_dir}")
        return
    
    print(f"找到 {len(image_files)} 张图像\n")
    
    # 创建预测器
    predictor = ONNXPredictor(model_path)
    
    # 统计
    total_time = 0
    total_detections = 0
    results = []
    
    for img_path in tqdm(image_files, desc="预测中"):
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        detections, inference_time = predictor.predict(image, conf, iou)
        total_time += inference_time
        total_detections += len(detections)
        
        # 绘制并保存
        annotated = predictor.draw_detections(image, detections)
        cv2.putText(annotated, f"Inference: {inference_time:.1f}ms | Detections: {len(detections)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        output_path = output_dir / f"pred_{img_path.name}"
        cv2.imwrite(str(output_path), annotated)
        
        results.append({
            'image': str(img_path),
            'detections': len(detections),
            'inference_ms': inference_time
        })
    
    # 统计信息
    print(f"\n统计信息:")
    print(f"  总图像数: {len(image_files)}")
    print(f"  总检测数: {total_detections}")
    print(f"  平均推理时间: {total_time / len(image_files):.2f} ms")
    print(f"  平均FPS: {1000 * len(image_files) / total_time:.1f}")
    
    # 保存结果
    results_file = output_dir / "predictions.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {results_file}")


def benchmark(model_path=None, num_runs=100):
    """性能基准测试"""
    model_path = model_path or MODEL_PATH
    
    print("=" * 60)
    print(" 性能基准测试")
    print("=" * 60)
    print(f"模型: {model_path}")
    print(f"测试次数: {num_runs}")
    print(f"模型大小: {get_model_size(model_path):.2f} MB\n")
    
    predictor = ONNXPredictor(model_path)
    
    # 随机输入
    input_tensor = np.random.randn(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)
    
    # 预热
    for _ in range(10):
        predictor.session.run(predictor.output_names, {predictor.input_name: input_tensor})
    
    # 测速
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        predictor.session.run(predictor.output_names, {predictor.input_name: input_tensor})
        times.append((time.perf_counter() - start) * 1000)
    
    times = np.array(times)
    print(f"\n平均推理时间: {times.mean():.2f} ms")
    print(f"标准差: {times.std():.2f} ms")
    print(f"最小值: {times.min():.2f} ms")
    print(f"最大值: {times.max():.2f} ms")
    print(f"FPS: {1000 / times.mean():.1f}")
    
    return {
        'model': model_path,
        'size_mb': get_model_size(model_path),
        'avg_ms': times.mean(),
        'std_ms': times.std(),
        'min_ms': times.min(),
        'max_ms': times.max(),
        'fps': 1000 / times.mean()
    }


# ==================== 主流程 ====================

def main():
    """主函数：不传参数时运行benchmark"""
    benchmark()


if __name__ == "__main__":
    main()
