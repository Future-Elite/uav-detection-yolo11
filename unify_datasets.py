import yaml
from pathlib import Path

# ======================
# 1. 统一类别定义
# ======================
UNIFIED_NAMES = {
    0: "plane",
    1: "bird",
    2: "drone",
    3: "helicopter"
}

# 原始类别名 → 统一类别名
NAME_ALIAS = {
    "airplane": "plane",
    "plane": "plane",
    "bird": "bird",
    "birds": "bird",
    "drone": "drone",
    "helicopter": "helicopter"
}


def unify_one_dataset(dataset_root: Path):
    data_yaml = dataset_root / "data.yaml"
    assert data_yaml.exists(), f"{data_yaml} 不存在"

    # 读取 data.yaml
    with open(data_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    old_names = data["names"]

    # 建立 old_id → new_id 映射
    id_map = {}
    for old_id, old_name in old_names.items():
        unified_name = NAME_ALIAS[old_name]
        new_id = list(UNIFIED_NAMES.values()).index(unified_name)
        id_map[int(old_id)] = new_id

    print(f"[INFO] {dataset_root.name} 类别映射: {id_map}")

    # 遍历 train / val / test
    for split in ["train", "valid", "test"]:
        labels_dir = dataset_root / split / "labels"
        if not labels_dir.exists():
            continue

        for label_file in labels_dir.rglob("*.txt"):
            new_lines = []
            with open(label_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    old_cls = int(parts[0])
                    parts[0] = str(id_map[old_cls])
                    new_lines.append(" ".join(parts))

            with open(label_file, "w") as f:
                f.write("\n".join(new_lines))

    # 重写 data.yaml
    data["names"] = UNIFIED_NAMES
    data["nc"] = len(UNIFIED_NAMES)

    with open(data_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)

    print(f"[DONE] {dataset_root.name} 处理完成\n")


if __name__ == "__main__":
    datasets = [
        Path("../datasets/UAV1"),
        Path("../datasets/UAV2"),
        Path("../datasets/Airborne"),
    ]

    for ds in datasets:
        unify_one_dataset(ds)
