from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_CLASS_NAMES = {
    0: "plane",
    1: "bird",
    2: "drone",
    3: "helicopter",
}
DEFAULT_COLORS = {
    0: (255, 80, 80),
    1: (80, 220, 120),
    2: (60, 150, 255),
    3: (60, 220, 220),
}


@dataclass
class BoxRecord:
    cls_id: int
    conf: float
    box: tuple[int, int, int, int]


@dataclass
class ImageCase:
    image_path: Path
    predictions: list[BoxRecord]
    gt_count: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    avg_tp_conf: float
    display_score: float
    strict_correct: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="生成类似论文展示图的测试样例拼图，默认从测试集中随机抽取样本。"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("runs/detect/ablation-5/weights/best.pt"),
        help="模型权重路径",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("../datasets/Airborne/data.yaml"),
        help="数据集 data.yaml 路径",
    )
    parser.add_argument("--split", type=str, default="test", choices=("train", "val", "valid", "test"), help="要展示的数据集划分")
    parser.add_argument("--images", type=Path, default=None, help="手动指定图片目录，优先级高于 --data")
    parser.add_argument("--labels", type=Path, default=None, help="手动指定标签目录，优先级高于 --data")
    parser.add_argument("--imgsz", type=int, default=640, help="推理尺寸")
    parser.add_argument("--conf", type=float, default=0.25, help="预测置信度阈值")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU 阈值")
    parser.add_argument("--match-iou", type=float, default=0.5, help="预测与真值匹配的 IoU 阈值")
    parser.add_argument("--batch", type=int, default=8, help="批量推理大小")
    parser.add_argument("--rows", type=int, default=4, help="拼图行数")
    parser.add_argument("--cols", type=int, default=4, help="拼图列数")
    parser.add_argument("--tile-size", type=int, default=320, help="单张子图边长")
    parser.add_argument("--max-det", type=int, default=20, help="每张图最多保留的预测框")
    parser.add_argument("--device", type=str, default=0, help="推理设备")
    parser.add_argument("--seed", type=int, default=42, help="随机选图种子，固定后可复现同一批样例")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plots/test_examples/test_example_grid.jpg"),
        help="拼图输出路径",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path("plots/test_examples/test_example_grid.csv"),
        help="入选样例统计表输出路径",
    )
    return parser.parse_args()


def resolve_candidate_path(*candidates: Path) -> Path | None:
    for candidate in candidates:
        if candidate is None:
            continue
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    return None


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def normalize_class_names(names: dict | list | None) -> dict[int, str]:
    if isinstance(names, list):
        return {idx: str(name) for idx, name in enumerate(names)}
    if isinstance(names, dict):
        return {int(idx): str(name) for idx, name in names.items()}
    return DEFAULT_CLASS_NAMES.copy()


def resolve_dataset_dirs(
    data_path: Path,
    split: str,
    images_override: Path | None,
    labels_override: Path | None,
) -> tuple[Path, Path, dict[int, str]]:
    data_cfg = load_yaml(data_path)
    class_names = normalize_class_names(data_cfg.get("names"))

    if images_override is not None:
        image_dir = images_override.resolve()
    else:
        split_key = "val" if split == "valid" and "valid" not in data_cfg else split
        split_rel = data_cfg.get(split_key)
        if split_rel is None:
            raise FileNotFoundError(f"{data_path} 中未找到 split={split_key} 的配置")

        dataset_root_hint = Path(str(data_cfg.get("path", ".")))
        image_dir = resolve_candidate_path(
            data_path.parent / dataset_root_hint / split_rel,
            Path.cwd() / dataset_root_hint / split_rel,
            data_path.parent / split_rel,
        )
        if image_dir is None:
            raise FileNotFoundError(f"无法解析图片目录: data={data_path}, split={split_key}")

    if labels_override is not None:
        label_dir = labels_override.resolve()
    else:
        label_dir = image_dir.parent / "labels"

    if not image_dir.exists():
        raise FileNotFoundError(f"图片目录不存在: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"标签目录不存在: {label_dir}")

    return image_dir, label_dir, class_names


def list_images(image_dir: Path) -> list[Path]:
    return sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


def clamp_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))
    return x1, y1, x2, y2


def yolo_to_xyxy(
    xc: float,
    yc: float,
    bw: float,
    bh: float,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1 = int(round((xc - bw / 2.0) * width))
    y1 = int(round((yc - bh / 2.0) * height))
    x2 = int(round((xc + bw / 2.0) * width))
    y2 = int(round((yc + bh / 2.0) * height))
    return clamp_box((x1, y1, x2, y2), width, height)


def load_ground_truth(label_path: Path, width: int, height: int) -> list[BoxRecord]:
    if not label_path.exists():
        return []

    records: list[BoxRecord] = []
    with label_path.open("r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(float(parts[0]))
            xc, yc, bw, bh = map(float, parts[1:5])
            records.append(
                BoxRecord(
                    cls_id=cls_id,
                    conf=1.0,
                    box=yolo_to_xyxy(xc, yc, bw, bh, width, height),
                )
            )
    return records


def extract_predictions(result, width: int, height: int) -> list[BoxRecord]:
    if result.boxes is None or len(result.boxes) == 0:
        return []

    xyxy = result.boxes.xyxy.detach().cpu().numpy()
    confs = result.boxes.conf.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy()
    if not (len(xyxy) == len(confs) == len(classes)):
        raise RuntimeError("预测结果字段长度不一致，无法解析检测框。")

    records: list[BoxRecord] = []
    for box, conf, cls_id in zip(xyxy, confs, classes):
        x1, y1, x2, y2 = [int(round(value)) for value in box.tolist()]
        records.append(
            BoxRecord(
                cls_id=int(cls_id),
                conf=float(conf),
                box=clamp_box((x1, y1, x2, y2), width, height),
            )
        )
    records.sort(key=lambda item: item.conf, reverse=True)
    return records


def compute_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def match_predictions(
    gt_boxes: list[BoxRecord],
    pred_boxes: list[BoxRecord],
    match_iou: float,
) -> tuple[int, list[float]]:
    used_gt: set[int] = set()
    tp_confidences: list[float] = []

    for pred in pred_boxes:
        best_gt_index = -1
        best_iou = 0.0
        for gt_index, gt in enumerate(gt_boxes):
            if gt_index in used_gt or gt.cls_id != pred.cls_id:
                continue
            iou = compute_iou(gt.box, pred.box)
            if iou > best_iou:
                best_iou = iou
                best_gt_index = gt_index

        if best_gt_index >= 0 and best_iou >= match_iou:
            used_gt.add(best_gt_index)
            tp_confidences.append(pred.conf)

    return len(tp_confidences), tp_confidences


def evaluate_case(image_path: Path, label_dir: Path, result, match_iou: float) -> ImageCase | None:
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    height, width = image.shape[:2]
    label_path = label_dir / f"{image_path.stem}.txt"
    gt_boxes = load_ground_truth(label_path, width, height)
    pred_boxes = extract_predictions(result, width, height)

    if not gt_boxes and not pred_boxes:
        return None

    tp, tp_confidences = match_predictions(gt_boxes, pred_boxes, match_iou)
    fp = max(0, len(pred_boxes) - tp)
    fn = max(0, len(gt_boxes) - tp)

    if len(pred_boxes) == 0:
        precision = 0.0
    else:
        precision = tp / len(pred_boxes)

    recall = 0.0 if len(gt_boxes) == 0 else tp / len(gt_boxes)
    avg_tp_conf = float(sum(tp_confidences) / len(tp_confidences)) if tp_confidences else 0.0

    display_score = 0.45 * precision + 0.35 * recall + 0.20 * avg_tp_conf
    strict_correct = bool(gt_boxes) and tp == len(gt_boxes) and fp == 0

    return ImageCase(
        image_path=image_path,
        predictions=pred_boxes,
        gt_count=len(gt_boxes),
        tp=tp,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        avg_tp_conf=avg_tp_conf,
        display_score=display_score,
        strict_correct=strict_correct,
    )


def batch_items(items: list[Path], batch_size: int) -> list[list[Path]]:
    return [items[idx:idx + batch_size] for idx in range(0, len(items), batch_size)]


def select_cases(cases: list[ImageCase], required: int, args: argparse.Namespace) -> list[ImageCase]:
    candidates = [case for case in cases if case.gt_count > 0]
    if not candidates:
        candidates = cases[:]

    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    return candidates[:required]


def draw_label(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    bg_color: tuple[int, int, int],
    text_color: tuple[int, int, int] = (255, 255, 255),
    scale: float = 0.6,
    thickness: int = 2,
) -> None:
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(image, (x, y - text_h - baseline - 6), (x + text_w + 10, y), bg_color, thickness=-1)
    cv2.putText(image, text, (x + 5, y - baseline - 2), font, scale, text_color, thickness, cv2.LINE_AA)


def annotate_image(
    case: ImageCase,
    class_names: dict[int, str],
) -> np.ndarray:
    image = cv2.imread(str(case.image_path))
    if image is None:
        raise FileNotFoundError(f"无法读取图片: {case.image_path}")

    line_width = max(2, int(round(min(image.shape[:2]) / 220)))
    text_scale = max(0.55, min(image.shape[:2]) / 900.0)

    for prediction in case.predictions:
        color = DEFAULT_COLORS.get(prediction.cls_id, (180, 180, 180))
        x1, y1, x2, y2 = prediction.box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, line_width)
        label = f"{class_names.get(prediction.cls_id, str(prediction.cls_id))} {prediction.conf:.2f}"
        text_y = max(y1, 24)
        draw_label(image, label, (x1, text_y), color, scale=text_scale, thickness=max(1, line_width - 1))

    header = (
        f"{case.image_path.name} | P {case.precision:.2f}  "
        f"R {case.recall:.2f}  C {case.avg_tp_conf:.2f}"
    )
    draw_label(image, header, (10, max(28, int(28 * text_scale * 1.5))), (30, 30, 30), scale=text_scale, thickness=2)
    return image


def resize_with_padding(image: np.ndarray, target_size: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(target_size / max(width, 1), target_size / max(height, 1))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)

    canvas = np.full((target_size, target_size, 3), 20, dtype=np.uint8)
    offset_x = (target_size - new_width) // 2
    offset_y = (target_size - new_height) // 2
    canvas[offset_y:offset_y + new_height, offset_x:offset_x + new_width] = resized
    return canvas


def build_grid(cases: list[ImageCase], class_names: dict[int, str], rows: int, cols: int, tile_size: int) -> np.ndarray:
    gap = 6
    canvas_h = rows * tile_size + (rows + 1) * gap
    canvas_w = cols * tile_size + (cols + 1) * gap
    canvas = np.full((canvas_h, canvas_w, 3), 32, dtype=np.uint8)

    for index in range(rows * cols):
        row = index // cols
        col = index % cols
        x = gap + col * (tile_size + gap)
        y = gap + row * (tile_size + gap)

        if index < len(cases):
            tile = resize_with_padding(annotate_image(cases[index], class_names), tile_size)
        else:
            tile = np.full((tile_size, tile_size, 3), 45, dtype=np.uint8)
            draw_label(tile, "No sample", (18, 40), (80, 80, 80), scale=0.9, thickness=2)

        canvas[y:y + tile_size, x:x + tile_size] = tile

    return canvas


def save_case_table(cases: list[ImageCase], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["image", "tp", "fp", "fn", "precision", "recall", "avg_tp_conf", "display_score", "strict_correct"])
        for case in cases:
            writer.writerow([
                case.image_path.name,
                case.tp,
                case.fp,
                case.fn,
                f"{case.precision:.4f}",
                f"{case.recall:.4f}",
                f"{case.avg_tp_conf:.4f}",
                f"{case.display_score:.4f}",
                int(case.strict_correct),
            ])


def main() -> None:
    args = parse_args()

    image_dir, label_dir, class_names = resolve_dataset_dirs(args.data, args.split, args.images, args.labels)
    image_paths = list_images(image_dir)
    if not image_paths:
        raise FileNotFoundError(f"图片目录为空: {image_dir}")

    model = YOLO(str(args.model))
    predict_kwargs = {
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "max_det": args.max_det,
        "verbose": False,
    }
    if args.device:
        predict_kwargs["device"] = args.device

    cases: list[ImageCase] = []
    for image_batch in batch_items(image_paths, max(1, args.batch)):
        results = model.predict(source=[str(path) for path in image_batch], **predict_kwargs)
        if len(results) != len(image_batch):
            raise RuntimeError(
                f"模型返回结果数量与输入图片数量不一致: images={len(image_batch)}, results={len(results)}"
            )
        for image_path, result in zip(image_batch, results):
            case = evaluate_case(image_path, label_dir, result, args.match_iou)
            if case is not None:
                cases.append(case)

    if not cases:
        raise RuntimeError("没有得到可用样例，请检查标签、模型或阈值配置。")

    required = args.rows * args.cols
    selected_cases = select_cases(cases, required, args)
    if not selected_cases:
        raise RuntimeError("没有筛选出可展示的样例，请适当降低阈值。")

    grid_image = build_grid(selected_cases, class_names, args.rows, args.cols, args.tile_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), grid_image)
    save_case_table(selected_cases, args.csv_output)

    strict_count = sum(1 for case in selected_cases if case.strict_correct)
    avg_precision = sum(case.precision for case in selected_cases) / len(selected_cases)
    avg_recall = sum(case.recall for case in selected_cases) / len(selected_cases)
    avg_conf = sum(case.avg_tp_conf for case in selected_cases) / len(selected_cases)

    print(f"已保存拼图: {args.output.resolve()}")
    print(f"已保存统计表: {args.csv_output.resolve()}")
    print(
        "展示样例统计: "
        f"selected={len(selected_cases)}, "
        f"strict_correct={strict_count}, "
        f"avg_precision={avg_precision:.3f}, "
        f"avg_recall={avg_recall:.3f}, "
        f"avg_tp_conf={avg_conf:.3f}"
    )
    print(f"说明: 当前样例为随机抽取，随机种子 seed={args.seed}。")


if __name__ == "__main__":
    main()
