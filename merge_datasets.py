from __future__ import annotations

import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

# =========================
# 1. 配置区域
# =========================

@dataclass(frozen=True)
class SourceDataset:
    name: str
    root: Path


SOURCE_DATASETS = [
    SourceDataset("UAV1", Path("../datasets/UAV DATASET")),
    SourceDataset("UAV2", Path("../datasets/DRONE DETECTION")),
    SourceDataset("Airborne", Path("../datasets/Airborne")),
]

TARGET_DATASET = Path("../datasets/merged_dataset_dynamic")

# 输出 split 结构保持不变，总样本量默认与原始样本量一致
SPLITS = ("train", "valid", "test")
OUTPUT_SPLIT_RATIO = {
    "train": 0.8,
    "valid": 0.2,
    "test": 0.0,
}

RANDOM_SEED = 42
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 动态加权采样超参数
DATASET_SIZE_POWER = 0.5
CLASS_BALANCE_POWER = 1.0
BACKGROUND_SAMPLE_FACTOR = 0.25
MIN_SAMPLE_WEIGHT = 1e-6

# True: 使用源数据已有 split，分别在 train/valid/test 内独立重采样
# False: 先汇总后按 OUTPUT_SPLIT_RATIO 重新划分
PRESERVE_SOURCE_SPLITS = True

# None 表示输出样本总量与输入相同；也可手动指定总样本数
TARGET_TOTAL_SAMPLES = None

CLASS_NAMES = {
    0: "plane",
    1: "bird",
    2: "drone",
    3: "helicopter",
}


# =========================
# 2. 数据结构与工具函数
# =========================

@dataclass(frozen=True)
class Sample:
    dataset_name: str
    source_split: str
    img_path: Path
    label_path: Path
    class_hist: Counter

    @property
    def instance_count(self) -> int:
        return sum(self.class_hist.values())


def parse_label_hist(label_path: Path) -> Counter:
    """统计单张图像标签中的类别实例数。"""
    hist = Counter()
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            hist[int(float(parts[0]))] += 1
    return hist


def collect_samples(dataset_root: Path) -> list[Sample]:
    """收集数据集中的 image-label 对及其类别统计。"""
    samples: list[Sample] = []
    for split in SPLITS:
        img_dir = dataset_root / split / "images"
        lbl_dir = dataset_root / split / "labels"
        if not img_dir.exists() or not lbl_dir.exists():
            continue

        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue

            label_path = lbl_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue

            samples.append(
                Sample(
                    dataset_name=dataset_root.name,
                    source_split=split,
                    img_path=img_path,
                    label_path=label_path,
                    class_hist=parse_label_hist(label_path),
                )
            )

    return samples


def validate_source_datasets():
    missing = [f"{source.name} -> {source.root}" for source in SOURCE_DATASETS if not source.root.exists()]
    if missing:
        raise FileNotFoundError("以下源数据集目录不存在:\n" + "\n".join(missing))


def prepare_dirs():
    for split in SPLITS:
        (TARGET_DATASET / split / "images").mkdir(parents=True, exist_ok=True)
        (TARGET_DATASET / split / "labels").mkdir(parents=True, exist_ok=True)


def summarize_class_counts(samples: list[Sample]) -> Counter:
    counts = Counter()
    for sample in samples:
        counts.update(sample.class_hist)
    return counts


def summarize_dataset_sizes(samples: list[Sample]) -> dict[str, int]:
    sizes = defaultdict(int)
    for sample in samples:
        sizes[sample.dataset_name] += 1
    return dict(sizes)


def pretty_class_counts(class_counts: Counter) -> str:
    return ", ".join(
        f"{CLASS_NAMES.get(cls_id, cls_id)}={class_counts.get(cls_id, 0)}"
        for cls_id in sorted(CLASS_NAMES)
    )


def build_sample_weights(samples: list[Sample]) -> list[float]:
    """
    根据数据集规模和类别分布为每张图像计算动态权重。

    权重由两部分组成：
    1. 数据集规模因子：小数据集获得更高基础权重
    2. 类别稀有度因子：包含稀有类别的图像被更频繁采样
    """
    if not samples:
        return []

    dataset_sizes = summarize_dataset_sizes(samples)
    class_counts = summarize_class_counts(samples)

    avg_dataset_size = sum(dataset_sizes.values()) / max(len(dataset_sizes), 1)
    non_zero_class_counts = [count for count in class_counts.values() if count > 0]
    avg_class_count = sum(non_zero_class_counts) / max(len(non_zero_class_counts), 1)

    dataset_weights = {
        dataset_name: max((avg_dataset_size / size) ** DATASET_SIZE_POWER, MIN_SAMPLE_WEIGHT)
        for dataset_name, size in dataset_sizes.items()
        if size > 0
    }
    class_weights = {
        class_id: max((avg_class_count / count) ** CLASS_BALANCE_POWER, MIN_SAMPLE_WEIGHT)
        for class_id, count in class_counts.items()
        if count > 0
    }

    sample_weights: list[float] = []
    for sample in samples:
        dataset_factor = dataset_weights.get(sample.dataset_name, 1.0)

        if sample.instance_count == 0:
            class_factor = BACKGROUND_SAMPLE_FACTOR
        else:
            weighted_sum = sum(
                class_weights.get(class_id, 1.0) * instance_count
                for class_id, instance_count in sample.class_hist.items()
            )
            class_factor = weighted_sum / sample.instance_count

        sample_weights.append(max(dataset_factor * class_factor, MIN_SAMPLE_WEIGHT))

    return sample_weights


def split_samples(samples: list[Sample], rng: random.Random) -> dict[str, list[Sample]]:
    """在不保留源 split 时，按比例重新划分原始样本。"""
    shuffled = samples[:]
    rng.shuffle(shuffled)

    total = len(shuffled)
    target_total = TARGET_TOTAL_SAMPLES or total
    target_total = min(target_total, total)

    shuffled = shuffled[:target_total]
    n_train = int(len(shuffled) * OUTPUT_SPLIT_RATIO["train"])
    n_valid = int(len(shuffled) * OUTPUT_SPLIT_RATIO["valid"])

    return {
        "train": shuffled[:n_train],
        "valid": shuffled[n_train:n_train + n_valid],
        "test": shuffled[n_train + n_valid:],
    }


def resample_split(samples: list[Sample], rng: random.Random, target_count: int) -> list[Sample]:
    """按动态权重对某个 split 做有放回重采样。"""
    if not samples or target_count <= 0:
        return []

    weights = build_sample_weights(samples)
    return rng.choices(samples, weights=weights, k=target_count)


def copy_samples(split_name: str, samples: list[Sample]):
    for idx, sample in enumerate(samples):
        new_name = f"{sample.dataset_name}_{idx:06d}"
        shutil.copy(
            sample.img_path,
            TARGET_DATASET / split_name / "images" / f"{new_name}{sample.img_path.suffix}",
        )
        shutil.copy(
            sample.label_path,
            TARGET_DATASET / split_name / "labels" / f"{new_name}.txt",
        )


def print_sampling_report(split_name: str, original: list[Sample], resampled: list[Sample]):
    print(f"\n[INFO] Split: {split_name}")
    print(f"       原始样本数: {len(original)}")
    print(f"       重采样后样本数: {len(resampled)}")
    print(f"       原始类别分布: {pretty_class_counts(summarize_class_counts(original))}")
    print(f"       重采样后分布: {pretty_class_counts(summarize_class_counts(resampled))}")


# =========================
# 3. 主合并逻辑
# =========================

def merge():
    rng = random.Random(RANDOM_SEED)
    validate_source_datasets()
    prepare_dirs()

    all_samples: list[Sample] = []
    for source in SOURCE_DATASETS:
        dataset_samples = collect_samples(source.root)
        dataset_samples = [
            Sample(
                dataset_name=source.name,
                source_split=sample.source_split,
                img_path=sample.img_path,
                label_path=sample.label_path,
                class_hist=sample.class_hist,
            )
            for sample in dataset_samples
        ]
        all_samples.extend(dataset_samples)
        print(
            f"[INFO] {source.name}: {len(dataset_samples)} samples, "
            f"{pretty_class_counts(summarize_class_counts(dataset_samples))}"
        )

    if PRESERVE_SOURCE_SPLITS:
        original_split_map = {split: [] for split in SPLITS}
        for sample in all_samples:
            if sample.source_split in original_split_map:
                original_split_map[sample.source_split].append(sample)
    else:
        original_split_map = split_samples(all_samples, rng)

    resampled_split_map: dict[str, list[Sample]] = {}
    for split in SPLITS:
        original = original_split_map.get(split, [])
        resampled = resample_split(original, rng, len(original))
        resampled_split_map[split] = resampled
        print_sampling_report(split, original, resampled)

    for split, samples in resampled_split_map.items():
        copy_samples(split, samples)

    total_original = sum(len(samples) for samples in original_split_map.values())
    total_resampled = sum(len(samples) for samples in resampled_split_map.values())
    print(f"\n[DONE] Dynamic weighted dataset merged successfully!")
    print(f"[INFO] Total samples: {total_original} -> {total_resampled}")


# =========================
# 4. 生成 data.yaml
# =========================

def write_data_yaml():
    names_block = "\n".join(f"  {idx}: {name}" for idx, name in CLASS_NAMES.items())
    content = (
        f"path: {TARGET_DATASET}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        f"nc: {len(CLASS_NAMES)}\n"
        "names:\n"
        f"{names_block}\n"
    )
    with open(TARGET_DATASET / "data.yaml", "w", encoding="utf-8") as f:
        f.write(content)


if __name__ == "__main__":
    merge()
    write_data_yaml()
