import random
import shutil
from pathlib import Path
import yaml

# =========================
# 1. 配置区域
# =========================

SOURCE_DATASETS = [
    Path("../datasets/UAV1"),
    Path("../datasets/UAV2"),
    Path("../datasets/Airborne"),
]

TARGET_DATASET = Path("../datasets/merged_dataset_5")

# 每个数据集在最终数据中所占比例
DATASET_WEIGHTS = {
    "UAV1": 0.14,
    "UAV2": 0.175,
    "Airborne": 0.385,
}

# 最终 train / val / test 比例
SPLIT_RATIO = {
    "train": 0.8,
    "valid": 0.2,
    "test": 0.0,
}

RANDOM_SEED = 42
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# =========================
# 2. 工具函数
# =========================

def collect_pairs(dataset_root: Path):
    """收集 image-label 对"""
    pairs = []
    for split in ["train", "valid", "test"]:
        img_dir = dataset_root / split / "images"
        lbl_dir = dataset_root / split / "labels"
        if not img_dir.exists():
            continue

        for img in img_dir.iterdir():
            if img.suffix.lower() not in IMAGE_SUFFIXES:
                continue  # 忽略 .npy 等非图片文件

            label = lbl_dir / f"{img.stem}.txt"
            if label.exists():
                pairs.append((img, label))

    return pairs


def prepare_dirs():
    for split in ["train", "valid", "test"]:
        (TARGET_DATASET / split / "images").mkdir(parents=True, exist_ok=True)
        (TARGET_DATASET / split / "labels").mkdir(parents=True, exist_ok=True)


# =========================
# 3. 主合并逻辑
# =========================

def merge():
    random.seed(RANDOM_SEED)
    prepare_dirs()

    all_selected = []

    # 按数据集比例抽样
    for ds in SOURCE_DATASETS:
        pairs = collect_pairs(ds)
        random.shuffle(pairs)

        take_num = int(len(pairs) * DATASET_WEIGHTS[ds.name])
        selected = pairs[:take_num]

        print(f"[INFO] {ds.name}: {take_num}/{len(pairs)} samples selected")
        all_selected.extend([(ds.name, p) for p in selected])

    # 再打乱一次
    random.shuffle(all_selected)

    # 按 train / val / test 划分
    total = len(all_selected)
    n_train = int(total * SPLIT_RATIO["train"])
    n_val = int(total * SPLIT_RATIO["valid"])

    split_map = {
        "train": all_selected[:n_train],
        "valid": all_selected[n_train:n_train + n_val],
        "test": all_selected[n_train + n_val:]
    }

    # 拷贝文件
    for split, items in split_map.items():
        for idx, (ds_name, (img, lbl)) in enumerate(items):
            new_name = f"{ds_name}_{idx:06d}"

            shutil.copy(
                img,
                TARGET_DATASET / split / "images" / f"{new_name}{img.suffix}"
            )
            shutil.copy(
                lbl,
                TARGET_DATASET / split / "labels" / f"{new_name}.txt"
            )

    print("[DONE] Dataset merged successfully!")


# =========================
# 4. 生成 data.yaml
# =========================

def write_data_yaml():
    data = {
        "path": str(TARGET_DATASET),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 4,
        "names": {
            0: "plane",
            1: "bird",
            2: "drone",
            3: "helicopter"
        }
    }

    with open(TARGET_DATASET / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


if __name__ == "__main__":
    merge()
    write_data_yaml()
