import os
import glob


def polygon_to_bbox(parts):
    cls = int(parts[0])
    coords = list(map(float, parts[1:]))

    xs = coords[0::2]
    ys = coords[1::2]

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2
    w = xmax - xmin
    h = ymax - ymin

    return f"{cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n"


def process_label_file(txt_path):
    new_lines = []
    with open(txt_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) == 5:
            # 已是 bbox
            # print(f"[INFO] 已是 bbox 格式，跳过: {txt_path} -> {line}")
            new_lines.append(line)
        elif len(parts) > 5 and len(parts[1:]) % 2 == 0:
            # polygon
            new_lines.append(polygon_to_bbox(parts))
            print(f"[INFO] 转换 polygon 为 bbox: {txt_path} -> {line.strip()} to {new_lines[-1].strip()}")
        else:
            print(f"[WARN] 跳过异常行: {txt_path} -> {line}")

    with open(txt_path, "w") as f:
        f.writelines(new_lines)


def convert_dataset(label_dir):
    txt_files = glob.glob(os.path.join(label_dir, "**/*.txt"), recursive=True)
    print(f"Found {len(txt_files)} label files")

    for txt in txt_files:
        process_label_file(txt)


if __name__ == "__main__":
    label_dir = "../datasets/UAV2/test/labels"
    convert_dataset(label_dir)
