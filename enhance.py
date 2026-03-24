"""
数据增强脚本 - 针对无人机检测数据集的域差距问题
====================================================

问题分析：
- 训练集（merged_dataset_3）: 中等尺度目标、良好光照、简单背景
- AOD4测试集 (mAP50=0.85): 与训练集相似度较高
- Anti2测试集 (mAP50=0.47): 极小目标、恶劣光照、复杂背景

增强策略：
1. 针对极小目标: 随机小尺度缩放、Mosaic、Copy-Paste小目标
2. 针对低对比度/光照: 亮度对比度调整、颜色抖动、雾化模拟
3. 针对模糊: 运动模糊、高斯模糊模拟
4. 针对复杂背景: Mosaic/Mixup、随机遮挡
"""

import cv2
import numpy as np
import random
import shutil
from pathlib import Path
from typing import List, Tuple, Optional
import yaml
from tqdm import tqdm


# =========================
# 配置区域
# =========================

class Config:
    # 源数据集路径
    SOURCE_DATASET = Path("../datasets/merged_dataset_3")
    
    # 输出数据集路径
    OUTPUT_DATASET = Path("../datasets/merged_dataset_3_enhanced_2")
    
    # 增强概率（每个样本应用增强的概率，保持数据集大小不变）
    AUGMENT_PROBABILITY = 1.0           # 1.0 表示所有样本都增强
    
    # 是否启用各类增强
    ENABLE_SMALL_TARGET_AUG = True      # 小目标增强
    ENABLE_LOW_LIGHT_AUG = True         # 低光照增强
    ENABLE_BLUR_AUG = True              # 模糊增强
    ENABLE_COPY_PASTE_SMALL = True      # 小目标复制粘贴
    
    # 小目标增强参数
    SMALL_SCALE_RANGE = (0.3, 0.7)      # 小尺度缩放范围
    SMALL_TARGET_COPY_TIMES = 3         # 小目标复制次数
    
    # 低光照增强参数
    BRIGHTNESS_RANGE = (-0.3, 0.1)      # 亮度调整范围
    CONTRAST_RANGE = (0.5, 1.0)         # 对比度调整范围
    FOG_PROBABILITY = 0.3               # 雾化概率
    FOG_INTENSITY_RANGE = (0.1, 0.4)    # 雾化强度范围
    
    # 模糊增强参数
    MOTION_BLUR_PROB = 0.3              # 运动模糊概率
    MOTION_BLUR_RANGE = (3, 10)         # 运动模糊核大小范围
    GAUSSIAN_BLUR_PROB = 0.2            # 高斯模糊概率
    GAUSSIAN_BLUR_RANGE = (3, 7)        # 高斯模糊核大小范围
    
    # 噪声参数
    GAUSSIAN_NOISE_PROB = 0.3           # 高斯噪声概率
    GAUSSIAN_NOISE_RANGE = (5, 20)      # 高斯噪声标准差范围
    
    # 小目标阈值（占图像面积比例）
    SMALL_TARGET_THRESHOLD = 0.01       # 1%以下为小目标
    
    # 随机种子
    RANDOM_SEED = 42
    
    # 图像扩展名
    IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# =========================
# 基础增强函数
# =========================

def adjust_brightness_contrast(
    image: np.ndarray,
    brightness: float = 0.0,
    contrast: float = 1.0
) -> np.ndarray:
    """调整亮度和对比度"""
    # brightness: -1.0 ~ 1.0
    # contrast: 0.0 ~ 2.0
    img_float = image.astype(np.float32)
    
    # 调整亮度
    img_float = img_float + brightness * 255
    
    # 调整对比度
    mean = np.mean(img_float)
    img_float = (img_float - mean) * contrast + mean
    
    # 裁剪并转换回uint8
    img_float = np.clip(img_float, 0, 255)
    return img_float.astype(np.uint8)


def add_fog(
    image: np.ndarray,
    intensity: float = 0.3
) -> np.ndarray:
    """
    添加雾化效果
    intensity: 雾化强度 (0.0 ~ 1.0)
    """
    # 创建雾化层（灰白色）
    fog_color = np.array([200, 200, 200], dtype=np.float32)
    
    # 生成随机雾化模式
    fog_layer = np.ones_like(image, dtype=np.float32) * fog_color
    
    # 添加一些随机变化模拟真实雾气
    noise = np.random.randn(*image.shape[:2]) * 20
    noise = np.stack([noise] * 3, axis=-1)
    fog_layer = fog_layer + noise
    fog_layer = np.clip(fog_layer, 0, 255)
    
    # 混合原图和雾化层
    img_float = image.astype(np.float32)
    result = img_float * (1 - intensity) + fog_layer * intensity
    
    return np.clip(result, 0, 255).astype(np.uint8)


def add_motion_blur(
    image: np.ndarray,
    kernel_size: int = 7,
    angle: Optional[float] = None
) -> np.ndarray:
    """
    添加运动模糊效果
    kernel_size: 模糊核大小
    angle: 运动方向角度（度）
    """
    if angle is None:
        angle = random.uniform(0, 360)
    
    # 创建运动模糊核
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2
    
    # 根据角度生成核
    angle_rad = np.deg2rad(angle)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    
    for i in range(kernel_size):
        offset = i - center
        x = int(center + offset * cos_a)
        y = int(center + offset * sin_a)
        if 0 <= x < kernel_size and 0 <= y < kernel_size:
            kernel[y, x] = 1
    
    kernel = kernel / kernel.sum() if kernel.sum() > 0 else kernel
    
    # 应用模糊
    return cv2.filter2D(image, -1, kernel)


def add_gaussian_blur(
    image: np.ndarray,
    kernel_size: int = 5
) -> np.ndarray:
    """添加高斯模糊"""
    # 确保核大小为奇数
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def add_gaussian_noise(
    image: np.ndarray,
    sigma: float = 10
) -> np.ndarray:
    """添加高斯噪声"""
    noise = np.random.randn(*image.shape) * sigma
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def color_jitter(
    image: np.ndarray,
    hue_shift: float = 0.0,
    sat_shift: float = 0.0,
    val_shift: float = 0.0
) -> np.ndarray:
    """
    颜色抖动 (HSV空间)
    hue_shift: 色调偏移 (-0.5 ~ 0.5)
    sat_shift: 饱和度偏移 (-0.5 ~ 0.5)
    val_shift: 明度偏移 (-0.5 ~ 0.5)
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    
    # HSV的范围是 H:0-179, S:0-255, V:0-255
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift * 179) % 179
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + sat_shift * 255, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + val_shift * 255, 0, 255)
    
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


# =========================
# 小目标增强函数
# =========================

def read_label_file(label_path: Path) -> List[List[float]]:
    """读取YOLO格式标签文件，只取前5个值 (class, x, y, w, h)"""
    labels = []
    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    # 只取前5个值，忽略分割点等额外数据
                    labels.append([float(x) for x in parts[:5]])
    return labels


def write_label_file(label_path: Path, labels: List[List[float]]):
    """写入YOLO格式标签文件"""
    with open(label_path, 'w') as f:
        for label in labels:
            f.write(' '.join([str(x) for x in label]) + '\n')


def get_small_targets(
    labels: List[List[float]],
    img_width: int,
    img_height: int,
    threshold: float = 0.01
) -> List[Tuple[int, List[float]]]:
    """
    获取小目标索引
    返回: [(索引, 标签), ...]
    """
    small_targets = []
    for i, label in enumerate(labels):
        # label格式: [class, x_center, y_center, width, height]
        w = label[3] * img_width
        h = label[4] * img_height
        area_ratio = (w * h) / (img_width * img_height)
        
        if area_ratio < threshold:
            small_targets.append((i, label))
    
    return small_targets


def copy_paste_small_targets(
    image: np.ndarray,
    labels: List[List[float]],
    copy_times: int = 3,
    threshold: float = 0.01
) -> Tuple[np.ndarray, List[List[float]]]:
    """
    复制粘贴小目标增强
    在图像中复制小目标到其他位置
    """
    h, w = image.shape[:2]
    new_labels = labels.copy()
    new_image = image.copy()
    
    small_targets = get_small_targets(labels, w, h, threshold)
    
    if not small_targets:
        return new_image, new_labels
    
    for _ in range(copy_times):
        # 随机选择一个小目标
        idx, label = random.choice(small_targets)
        
        # 计算目标在图像中的位置
        cx, cy = int(label[1] * w), int(label[2] * h)
        tw, th = int(label[3] * w), int(label[4] * h)
        
        # 提取目标区域（带一些背景）
        margin = max(5, int(min(tw, th) * 0.2))
        x1 = max(0, cx - tw // 2 - margin)
        y1 = max(0, cy - th // 2 - margin)
        x2 = min(w, cx + tw // 2 + margin)
        y2 = min(h, cy + th // 2 + margin)
        
        target_crop = image[y1:y2, x1:x2].copy()
        
        # 随机新位置
        new_cx = random.randint(tw // 2 + margin, w - tw // 2 - margin)
        new_cy = random.randint(th // 2 + margin, h - th // 2 - margin)
        
        # 检查是否与现有目标重叠
        overlap = False
        for other_label in new_labels:
            other_cx, other_cy = other_label[1] * w, other_label[2] * h
            other_tw, other_th = other_label[3] * w, other_label[4] * h
            
            if abs(new_cx - other_cx) < (tw + other_tw) / 2 and \
               abs(new_cy - other_cy) < (th + other_th) / 2:
                overlap = True
                break
        
        if overlap:
            continue
        
        # 粘贴到新位置
        new_x1 = max(0, new_cx - tw // 2 - margin)
        new_y1 = max(0, new_cy - th // 2 - margin)
        new_x2 = min(w, new_cx + tw // 2 + margin)
        new_y2 = min(h, new_cy + th // 2 + margin)
        
        # 调整裁剪区域大小以匹配目标区域
        crop_h, crop_w = target_crop.shape[:2]
        target_h, target_w = new_y2 - new_y1, new_x2 - new_x1
        
        if crop_h > 0 and crop_w > 0 and target_h > 0 and target_w > 0:
            # 使用简单的alpha混合
            alpha = 0.7
            resized_crop = cv2.resize(target_crop, (target_w, target_h))
            
            # 创建简单的边缘羽化
            mask = np.ones((target_h, target_w), dtype=np.float32)
            blur_size = min(target_h, target_w) // 4
            if blur_size > 0 and blur_size % 2 == 1:
                mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
                mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-6)
            
            mask = np.stack([mask] * 3, axis=-1)
            
            roi = new_image[new_y1:new_y2, new_x1:new_x2]
            blended = (roi * (1 - alpha * mask) + resized_crop * alpha * mask).astype(np.uint8)
            new_image[new_y1:new_y2, new_x1:new_x2] = blended
            
            # 添加新标签
            new_label = label.copy()
            new_label[1] = new_cx / w
            new_label[2] = new_cy / h
            new_labels.append(new_label)
    
    return new_image, new_labels


def small_scale_augment(
    image: np.ndarray,
    labels: List[List[float]],
    scale_range: Tuple[float, float] = (0.3, 0.7)
) -> Tuple[np.ndarray, List[List[float]]]:
    """
    小尺度缩放增强
    将图像缩小后再放大，模拟远距离目标
    """
    h, w = image.shape[:2]
    
    # 随机缩放比例
    scale = random.uniform(*scale_range)
    
    # 先缩小
    small_h, small_w = int(h * scale), int(w * scale)
    small_image = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)
    
    # 再放大回原始尺寸（会产生模糊）
    result = cv2.resize(small_image, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # 标签保持不变（因为是缩放后再放大，目标位置比例不变）
    return result, labels


# =========================
# Mosaic增强
# =========================

def mosaic_augment(
    images: List[np.ndarray],
    labels_list: List[List[List[float]]],
    target_size: int = 640
) -> Tuple[np.ndarray, List[List[float]]]:
    """
    Mosaic数据增强
    将4张图像拼接成一张
    """
    assert len(images) == 4, "Mosaic需要4张图像"
    
    # 创建输出画布
    mosaic_img = np.zeros((target_size * 2, target_size * 2, 3), dtype=np.uint8)
    mosaic_labels = []
    
    # 随机选择中心点
    cx = random.randint(target_size // 2, target_size * 3 // 2)
    cy = random.randint(target_size // 2, target_size * 3 // 2)
    
    for i, (img, labels) in enumerate(zip(images, labels_list)):
        h, w = img.shape[:2]
        
        # 确定放置位置
        if i == 0:  # 左上
            x1a, y1a = max(cx - w, 0), max(cy - h, 0)
            x2a, y2a = cx, cy
            x1b, y1b = w - (x2a - x1a), h - (y2a - y1a)
            x2b, y2b = w, h
        elif i == 1:  # 右上
            x1a, y1a = cx, max(cy - h, 0)
            x2a, y2a = min(cx + w, target_size * 2), cy
            x1b, y1b = 0, h - (y2a - y1a)
            x2b, y2b = min(w, x2a - x1a), h
        elif i == 2:  # 左下
            x1a, y1a = max(cx - w, 0), cy
            x2a, y2a = cx, min(target_size * 2, cy + h)
            x1b, y1b = w - (x2a - x1a), 0
            x2b, y2b = w, min(y2a - y1a, h)
        else:  # 右下
            x1a, y1a = cx, cy
            x2a, y2a = min(target_size * 2, cx + w), min(target_size * 2, cy + h)
            x1b, y1b = 0, 0
            x2b, y2b = min(w, x2a - x1a), min(h, y2a - y1a)
        
        # 放置图像
        mosaic_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
        
        # 调整标签坐标
        for label in labels:
            if len(label) < 5:
                continue
            cls, xc, yc, bw, bh = label[:5]
            
            # 原始像素坐标
            px = xc * w
            py = yc * h
            pw = bw * w
            ph = bh * h
            
            # 转换到裁剪区域
            px_new = px - x1b + x1a
            py_new = py - y1b + y1a
            
            # 转换为归一化坐标
            new_xc = px_new / (target_size * 2)
            new_yc = py_new / (target_size * 2)
            new_bw = pw / (target_size * 2)
            new_bh = ph / (target_size * 2)
            
            # 检查目标是否在图像内
            if 0 < new_xc < 1 and 0 < new_yc < 1:
                mosaic_labels.append([cls, new_xc, new_yc, new_bw, new_bh])
    
    # 裁剪到目标大小
    final_img = mosaic_img[cy - target_size // 2:cy + target_size // 2,
                           cx - target_size // 2:cx + target_size // 2]
    
    # 调整标签
    final_labels = []
    for label in mosaic_labels:
        if len(label) < 5:
            continue
        cls, xc, yc, bw, bh = label[:5]
        # 调整到裁剪后的坐标
        new_xc = (xc * target_size * 2 - (cx - target_size // 2)) / target_size
        new_yc = (yc * target_size * 2 - (cy - target_size // 2)) / target_size
        
        if 0 < new_xc < 1 and 0 < new_yc < 1:
            final_labels.append([cls, new_xc, new_yc, bw, bh])
    
    return final_img, final_labels


# =========================
# 综合增强函数
# =========================

def apply_augmentation(
    image: np.ndarray,
    labels: List[List[float]],
    config: Config
) -> Tuple[np.ndarray, List[List[float]]]:
    """应用综合增强"""
    
    # 1. 小目标增强
    if config.ENABLE_SMALL_TARGET_AUG and random.random() < 0.5:
        image, labels = copy_paste_small_targets(
            image, labels,
            copy_times=config.SMALL_TARGET_COPY_TIMES,
            threshold=config.SMALL_TARGET_THRESHOLD
        )
    
    # 2. 小尺度缩放（模拟远距离）
    if config.ENABLE_SMALL_TARGET_AUG and random.random() < 0.3:
        image, labels = small_scale_augment(
            image, labels,
            scale_range=config.SMALL_SCALE_RANGE
        )
    
    # 3. 低光照/对比度增强
    if config.ENABLE_LOW_LIGHT_AUG:
        # 亮度对比度调整
        brightness = random.uniform(*config.BRIGHTNESS_RANGE)
        contrast = random.uniform(*config.CONTRAST_RANGE)
        image = adjust_brightness_contrast(image, brightness, contrast)
        
        # 雾化效果
        if random.random() < config.FOG_PROBABILITY:
            fog_intensity = random.uniform(*config.FOG_INTENSITY_RANGE)
            image = add_fog(image, fog_intensity)
        
        # 颜色抖动
        if random.random() < 0.3:
            hue = random.uniform(-0.1, 0.1)
            sat = random.uniform(-0.3, 0.1)
            val = random.uniform(-0.2, 0.1)
            image = color_jitter(image, hue, sat, val)
    
    # 4. 模糊增强
    if config.ENABLE_BLUR_AUG:
        # 运动模糊
        if random.random() < config.MOTION_BLUR_PROB:
            kernel_size = random.randint(*config.MOTION_BLUR_RANGE)
            image = add_motion_blur(image, kernel_size)
        
        # 高斯模糊
        if random.random() < config.GAUSSIAN_BLUR_PROB:
            kernel_size = random.choice(range(
                config.GAUSSIAN_BLUR_RANGE[0],
                config.GAUSSIAN_BLUR_RANGE[1] + 1, 2
            ))
            image = add_gaussian_blur(image, kernel_size)
    
    # 5. 高斯噪声
    if random.random() < config.GAUSSIAN_NOISE_PROB:
        sigma = random.uniform(*config.GAUSSIAN_NOISE_RANGE)
        image = add_gaussian_noise(image, sigma)
    
    return image, labels


# =========================
# 主处理流程
# =========================

def collect_samples(dataset_root: Path) -> List[Tuple[Path, Path]]:
    """收集图像-标签对"""
    samples = []
    for split in ["train", "valid", "test"]:
        img_dir = dataset_root / split / "images"
        lbl_dir = dataset_root / split / "labels"
        
        if not img_dir.exists():
            continue
        
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in Config.IMAGE_SUFFIXES:
                continue
            
            label_path = lbl_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                samples.append((img_path, label_path, split))
    
    return samples


def prepare_output_dirs(config: Config):
    """创建输出目录"""
    for split in ["train", "valid", "test"]:
        (config.OUTPUT_DATASET / split / "images").mkdir(parents=True, exist_ok=True)
        (config.OUTPUT_DATASET / split / "labels").mkdir(parents=True, exist_ok=True)


def process_dataset(config: Config):
    """处理整个数据集（保持数据集大小不变）"""
    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    
    print("[INFO] 收集样本...")
    samples = collect_samples(config.SOURCE_DATASET)
    print(f"[INFO] 共收集 {len(samples)} 个样本")
    
    print("[INFO] 创建输出目录...")
    prepare_output_dirs(config)
    
    print("[INFO] 开始处理...")
    
    # 按split分组处理
    splits = {"train": [], "valid": [], "test": []}
    for sample in samples:
        img_path, label_path, split = sample
        splits[split].append((img_path, label_path))
    
    # 处理每个split
    for split, split_samples in splits.items():
        print(f"\n[INFO] 处理 {split} 集 ({len(split_samples)} 样本)...")
        
        for idx, (img_path, label_path) in enumerate(tqdm(split_samples)):
            # 读取图像和标签
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"[WARN] 无法读取图像: {img_path}")
                continue
            
            labels = read_label_file(label_path)
            
            # 根据概率决定是否应用增强
            if random.random() < config.AUGMENT_PROBABILITY:
                aug_image, aug_labels = apply_augmentation(
                    image.copy(), labels.copy(), config
                )
            else:
                aug_image, aug_labels = image, labels
            
            # 保存增强后的图像（保持原始文件名）
            output_name = f"{idx:06d}"
            cv2.imwrite(
                str(config.OUTPUT_DATASET / split / "images" / f"{output_name}.jpg"),
                aug_image
            )
            write_label_file(
                config.OUTPUT_DATASET / split / "labels" / f"{output_name}.txt",
                aug_labels
            )
    
    print("\n[INFO] 数据增强完成！")


def write_data_yaml(config: Config):
    """生成data.yaml配置文件"""
    # 读取原始data.yaml获取类别信息
    with open(config.SOURCE_DATASET / "data.yaml", 'r', encoding='utf-8') as f:
        source_data = yaml.safe_load(f)
    
    data = {
        "path": str(config.OUTPUT_DATASET),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": source_data.get("nc", 4),
        "names": source_data.get("names", {0: "plane", 1: "bird", 2: "drone", 3: "helicopter"})
    }
    
    with open(config.OUTPUT_DATASET / "data.yaml", 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    
    print(f"[INFO] 已生成 {config.OUTPUT_DATASET / 'data.yaml'}")


# =========================
# 入口
# =========================

if __name__ == "__main__":
    config = Config()
    
    print("=" * 60)
    print("数据增强配置")
    print("=" * 60)
    print(f"源数据集: {config.SOURCE_DATASET}")
    print(f"输出数据集: {config.OUTPUT_DATASET}")
    print(f"小目标增强: {config.ENABLE_SMALL_TARGET_AUG}")
    print(f"低光照增强: {config.ENABLE_LOW_LIGHT_AUG}")
    print(f"模糊增强: {config.ENABLE_BLUR_AUG}")
    print("=" * 60)
    
    process_dataset(config)
    write_data_yaml(config)
    
    print("\n[DONE] 所有处理完成！")
    print(f"增强后的数据集保存在: {config.OUTPUT_DATASET}")
