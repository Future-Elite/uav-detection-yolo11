"""
自动筛选并生成典型失败样例分析图

默认目标:
    (a) 强伪装漏检
    (b) 眩光误检
    (c) 模糊定位偏差

默认使用方式:
    D:\\Workspace\\Thesis\\.venv\\Scripts\\python.exe auto_failure_case_analysis.py ^
        --data ..\\datasets\\Airborne\\data.yaml ^
        --model runs\\detect\\ablation-5\\weights\\best.pt

输出:
    1. 自动筛选的 case JSON
    2. 最终论文图 PNG/SVG
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

import plot_failure_case_analysis as failure_plot


def resolve_existing_path(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def load_dataset_paths(data_yaml: Path, split: str) -> tuple[Path, Path, dict[int, str]]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root_text = data.get("path", ".")
    root = Path(root_text)
    if root.is_absolute():
        resolved_root = root
    else:
        resolved_root = resolve_existing_path(
            [
                (Path.cwd() / root),
                (data_yaml.parent / root),
                data_yaml.parent,
            ]
        )
    image_dir = (resolved_root / data[split]).resolve()
    labels_dir = Path(str(image_dir).replace("\\images", "\\labels").replace("/images", "/labels")).resolve()
    names = data["names"]
    if isinstance(names, list):
        names = {i: name for i, name in enumerate(names)}
    else:
        names = {int(k): v for k, v in names.items()}
    return image_dir, labels_dir, names


def yolo_to_xyxy(box: list[float], width: int, height: int) -> list[int]:
    _, cx, cy, bw, bh = box
    x1 = int((cx - bw / 2) * width)
    y1 = int((cy - bh / 2) * height)
    x2 = int((cx + bw / 2) * width)
    y2 = int((cy + bh / 2) * height)
    return [max(0, x1), max(0, y1), min(width - 1, x2), min(height - 1, y2)]


def read_gt_labels(label_path: Path, image_shape: tuple[int, int], names: dict[int, str]) -> list[dict]:
    height, width = image_shape[:2]
    if not label_path.exists():
        return []
    items = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = [float(item) for item in line.split()]
        cls_id = int(parts[0])
        items.append(
            {
                "cls": cls_id,
                "name": names.get(cls_id, str(cls_id)),
                "box": yolo_to_xyxy(parts, width, height),
            }
        )
    return items


def box_iou_xyxy(box1: list[int], box2: list[int]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area1 = max(1, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(1, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def box_center(box: list[int]) -> tuple[float, float]:
    return (0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3]))


def center_distance_ratio(box1: list[int], box2: list[int]) -> float:
    c1 = box_center(box1)
    c2 = box_center(box2)
    dist = math.hypot(c1[0] - c2[0], c1[1] - c2[1])
    diag = math.hypot(max(1, box1[2] - box1[0]), max(1, box1[3] - box1[1]))
    return dist / diag


def expand_box(box: list[int], image_shape: tuple[int, int], scale: float = 1.8) -> list[int]:
    h, w = image_shape[:2]
    cx, cy = box_center(box)
    bw = (box[2] - box[0]) * scale
    bh = (box[3] - box[1]) * scale
    x1 = max(0, int(cx - bw / 2))
    y1 = max(0, int(cy - bh / 2))
    x2 = min(w - 1, int(cx + bw / 2))
    y2 = min(h - 1, int(cy + bh / 2))
    return [x1, y1, x2, y2]


def crop_image(image: np.ndarray, box: list[int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return image[0:1, 0:1]
    return image[y1:y2, x1:x2]


def crop_stats(image: np.ndarray, box: list[int]) -> dict[str, float]:
    crop = crop_image(image, box)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mean_v = float(hsv[:, :, 2].mean()) / 255.0
    mean_s = float(hsv[:, :, 1].mean()) / 255.0
    std_gray = float(gray.std()) / 255.0
    bright_ratio = float((hsv[:, :, 2] > 220).mean())
    white_ratio = float(((hsv[:, :, 2] > 180) & (hsv[:, :, 1] < 40)).mean())
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return {
        "mean_v": mean_v,
        "mean_s": mean_s,
        "std_gray": std_gray,
        "bright_ratio": bright_ratio,
        "white_ratio": white_ratio,
        "lap_var": lap_var,
    }


def match_boxes(gt_boxes: list[dict], pred_boxes: list[dict], iou_threshold: float = 0.5, same_class: bool = True) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    pairs = []
    for i, gt in enumerate(gt_boxes):
        for j, pred in enumerate(pred_boxes):
            if same_class and gt["cls"] != pred["cls"]:
                continue
            iou = box_iou_xyxy(gt["box"], pred["box"])
            if iou > 0:
                pairs.append((i, j, iou))
    pairs.sort(key=lambda item: item[2], reverse=True)

    matched_gt = set()
    matched_pred = set()
    matches: list[tuple[int, int, float]] = []
    for i, j, iou in pairs:
        if iou < iou_threshold or i in matched_gt or j in matched_pred:
            continue
        matched_gt.add(i)
        matched_pred.add(j)
        matches.append((i, j, iou))

    unmatched_gt = [i for i in range(len(gt_boxes)) if i not in matched_gt]
    unmatched_pred = [j for j in range(len(pred_boxes)) if j not in matched_pred]
    return matches, unmatched_gt, unmatched_pred


def nearest_same_class_pred(gt_box: dict, pred_boxes: list[dict]) -> tuple[int | None, float]:
    best_idx = None
    best_iou = 0.0
    for idx, pred in enumerate(pred_boxes):
        if pred["cls"] != gt_box["cls"]:
            continue
        iou = box_iou_xyxy(gt_box["box"], pred["box"])
        if iou > best_iou:
            best_iou = iou
            best_idx = idx
    return best_idx, best_iou


def make_annotation_text_xy(box: list[int], image_shape: tuple[int, int], side: str = "left") -> tuple[list[int], list[int]]:
    x1, y1, x2, y2 = box
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    h, w = image_shape[:2]
    if side == "left":
        return [cx, cy], [max(20, x1 - int(0.28 * w)), max(25, y1 - 25)]
    if side == "right":
        return [cx, cy], [min(w - 120, x2 + 20), max(25, y1 - 25)]
    return [cx, cy], [cx, max(25, y1 - 30)]


def build_pred_boxes(result) -> list[dict]:
    boxes = []
    if result.boxes is None or len(result.boxes) == 0:
        return boxes
    xyxy = result.boxes.xyxy.cpu().numpy().astype(int)
    conf = result.boxes.conf.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy().astype(int)
    for i in range(len(xyxy)):
        boxes.append(
            {
                "box": xyxy[i].tolist(),
                "conf": float(conf[i]),
                "cls": int(cls[i]),
                "name": result.names[int(cls[i])],
            }
        )
    return boxes


def rank_camouflage_candidate(image: np.ndarray, gt_box: dict) -> float:
    box = gt_box["box"]
    crop = crop_stats(image, box)
    context = crop_stats(image, expand_box(box, image.shape, scale=2.2))
    contrast_score = 1.0 - min(crop["std_gray"] / 0.18, 1.0)
    similarity = 1.0 - min(abs(crop["mean_v"] - context["mean_v"]) / 0.25, 1.0)
    return 0.35 * crop["white_ratio"] + 0.25 * crop["bright_ratio"] + 0.20 * contrast_score + 0.20 * similarity


def rank_glare_candidate(image: np.ndarray, pred_box: dict) -> float:
    box = pred_box["box"]
    crop = crop_stats(image, box)
    h, _ = image.shape[:2]
    _, cy = box_center(box)
    top_score = 1.0 - min(cy / max(h, 1), 1.0)
    contrast_score = 1.0 - min(crop["std_gray"] / 0.22, 1.0)
    return 0.40 * crop["bright_ratio"] + 0.25 * crop["white_ratio"] + 0.20 * top_score + 0.15 * contrast_score


def rank_blur_candidate(image: np.ndarray, gt_box: dict, pred_box: dict, iou: float) -> float:
    crop = crop_stats(image, gt_box["box"])
    blur_score = 1.0 - min(math.log1p(crop["lap_var"]) / 6.0, 1.0)
    offset_score = min(center_distance_ratio(gt_box["box"], pred_box["box"]) / 0.8, 1.0)
    iou_penalty = 1.0 - min(iou / 0.6, 1.0)
    return 0.45 * blur_score + 0.35 * offset_score + 0.20 * iou_penalty


def select_distinct_cases(candidate_lists: dict[str, list[dict]]) -> dict[str, dict]:
    chosen: dict[str, dict] = {}
    used_images: set[str] = set()
    for case_key in ("camouflage_miss", "glare_fp", "blur_bias"):
        for item in candidate_lists.get(case_key, []):
            image_key = str(item["image_path"])
            if image_key not in used_images:
                chosen[case_key] = item
                used_images.add(image_key)
                break
        if case_key not in chosen and candidate_lists.get(case_key):
            chosen[case_key] = candidate_lists[case_key][0]
    return chosen


def build_cases_config(selected: dict[str, dict]) -> dict:
    cases = []

    miss = selected["camouflage_miss"]
    xy, xytext = make_annotation_text_xy(miss["target_box"], miss["image_shape"], "left")
    xy2, xytext2 = make_annotation_text_xy(miss["target_box"], miss["image_shape"], "right")
    cases.append(
        {
            "tag": "a",
            "title": "强伪装漏检",
            "description": "目标几乎隐身在白云背景中，模型未输出有效检测框。",
            "original_image": str(miss["image_path"]),
            "result_image": str(miss["image_path"]),
            "original_boxes": [{"label": "GT", "box": miss["target_box"], "color": failure_plot.DEFAULT_COLORS["gt"], "linewidth": 2.0, "linestyle": "--"}],
            "result_boxes": [],
            "annotations": [
                {"panel": "original", "text": "目标与明亮背景高度混淆", "xy": xy, "xytext": xytext, "color": "#2C3E50"},
                {"panel": "result", "text": "未检测到目标", "xy": xy2, "xytext": xytext2, "color": failure_plot.DEFAULT_COLORS["pred"]},
            ],
        }
    )

    glare = selected["glare_fp"]
    xy, xytext = make_annotation_text_xy(glare["fp_box"], glare["image_shape"], "left")
    cases.append(
        {
            "tag": "b",
            "title": "眩光误检",
            "description": "逆光条件下，模型将太阳周围光晕误判为障碍物目标。",
            "original_image": str(glare["image_path"]),
            "result_image": str(glare["image_path"]),
            "original_boxes": [],
            "result_boxes": [{"label": "误检", "box": glare["fp_box"], "color": failure_plot.DEFAULT_COLORS["fp"], "linewidth": 2.2, "linestyle": "-"}],
            "annotations": [
                {
                    "panel": "result",
                    "text": "高亮光晕区域被误检",
                    "xy": xy,
                    "xytext": xytext,
                    "color": failure_plot.DEFAULT_COLORS["fp"],
                }
            ],
        }
    )

    blur = selected["blur_bias"]
    xy, xytext = make_annotation_text_xy(blur["pred_box"], blur["image_shape"], "left")
    cases.append(
        {
            "tag": "c",
            "title": "模糊定位偏差",
            "description": "高速运动导致目标模糊，预测框（红）明显偏离真实框（绿）的中心。",
            "original_image": str(blur["image_path"]),
            "result_image": str(blur["image_path"]),
            "original_boxes": [{"label": "GT", "box": blur["gt_box"], "color": failure_plot.DEFAULT_COLORS["gt"], "linewidth": 2.0, "linestyle": "--"}],
            "result_boxes": [
                {"label": "GT", "box": blur["gt_box"], "color": failure_plot.DEFAULT_COLORS["gt"], "linewidth": 2.2, "linestyle": "-"},
                {"label": "Pred", "box": blur["pred_box"], "color": failure_plot.DEFAULT_COLORS["pred"], "linewidth": 2.2, "linestyle": "-"},
            ],
            "annotations": [
                {
                    "panel": "result",
                    "text": "预测框（红）与真实框（绿）中心偏移明显",
                    "xy": xy,
                    "xytext": xytext,
                    "color": failure_plot.DEFAULT_COLORS["pred"],
                }
            ],
        }
    )

    return {"figure_title": "典型失败样例分析图", "cases": cases}


def write_json(payload: dict, output_path: Path) -> None:
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="自动筛选并生成典型失败样例分析图")
    parser.add_argument("--data", type=Path, default=Path("../datasets/Airborne/data.yaml"), help="数据集 data.yaml")
    parser.add_argument("--split", default="test", help="数据集 split，默认 test")
    parser.add_argument("--model", type=Path, default=Path("runs/detect/ablation-5/weights/best.pt"), help="用于分析的模型权重")
    parser.add_argument("--device", default="0", help="设备，如 0 或 cpu")
    parser.add_argument("--imgsz", type=int, default=640, help="推理尺寸")
    parser.add_argument("--conf", type=float, default=0.001, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU 阈值")
    parser.add_argument("--batch", type=int, default=16, help="推理 batch")
    parser.add_argument("--max-images", type=int, default=None, help="最多扫描多少张图，默认扫描全部")
    parser.add_argument("--layout", choices=["rows", "cols"], default="rows", help="最终论文图布局")
    parser.add_argument("--output-dir", type=Path, default=Path("plots/failure_case_analysis_auto"), help="输出目录")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.data.exists():
        raise FileNotFoundError(f"数据集配置不存在: {args.data.resolve()}")
    if not args.model.exists():
        raise FileNotFoundError(f"模型权重不存在: {args.model.resolve()}")

    image_dir, labels_dir, names = load_dataset_paths(args.data, args.split)
    image_paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
    if args.max_images:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise FileNotFoundError(f"未在 {image_dir} 中找到图片")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.model))

    candidate_lists = {
        "camouflage_miss": [],
        "glare_fp": [],
        "blur_bias": [],
    }
    fallback_lists = {
        "camouflage_miss": [],
        "glare_fp": [],
        "blur_bias": [],
    }

    print(f"扫描图片数: {len(image_paths)}")
    results = model.predict(
        source=[str(p) for p in image_paths],
        stream=True,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        batch=args.batch,
        device=args.device,
        verbose=False,
    )

    for image_path, result in zip(image_paths, results):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        gt_boxes = read_gt_labels(labels_dir / f"{image_path.stem}.txt", image.shape, names)
        pred_boxes = build_pred_boxes(result)

        for gt_box in gt_boxes:
            pred_idx, best_iou = nearest_same_class_pred(gt_box, pred_boxes)
            camouflage_score = rank_camouflage_candidate(image, gt_box) + 0.25 * (1.0 - min(best_iou / 0.5, 1.0))
            fallback_lists["camouflage_miss"].append(
                {
                    "score": camouflage_score,
                    "image_path": image_path,
                    "image_shape": image.shape,
                    "target_box": gt_box["box"],
                }
            )
            if pred_idx is None or best_iou < 0.5:
                candidate_lists["camouflage_miss"].append(
                    {
                        "score": camouflage_score,
                        "image_path": image_path,
                        "image_shape": image.shape,
                        "target_box": gt_box["box"],
                    }
                )

        for pred in pred_boxes:
            best_iou_any = 0.0
            for gt_box in gt_boxes:
                if gt_box["cls"] != pred["cls"]:
                    continue
                best_iou_any = max(best_iou_any, box_iou_xyxy(gt_box["box"], pred["box"]))
            glare_score = rank_glare_candidate(image, pred) + 0.20 * (1.0 - min(best_iou_any / 0.3, 1.0))
            fallback_lists["glare_fp"].append(
                {
                    "score": glare_score,
                    "image_path": image_path,
                    "image_shape": image.shape,
                    "fp_box": pred["box"],
                }
            )
            if not gt_boxes or best_iou_any < 0.3:
                candidate_lists["glare_fp"].append(
                    {
                        "score": glare_score,
                        "image_path": image_path,
                        "image_shape": image.shape,
                        "fp_box": pred["box"],
                    }
                )

        for gt_box in gt_boxes:
            pred_idx, iou = nearest_same_class_pred(gt_box, pred_boxes)
            if pred_idx is None:
                continue
            pred_box = pred_boxes[pred_idx]
            blur_score = rank_blur_candidate(image, gt_box, pred_box, iou)
            fallback_lists["blur_bias"].append(
                {
                    "score": blur_score,
                    "image_path": image_path,
                    "image_shape": image.shape,
                    "gt_box": gt_box["box"],
                    "pred_box": pred_box["box"],
                }
            )
            if 0.05 <= iou < 0.7 and center_distance_ratio(gt_box["box"], pred_box["box"]) >= 0.10:
                candidate_lists["blur_bias"].append(
                    {
                        "score": blur_score,
                        "image_path": image_path,
                        "image_shape": image.shape,
                        "gt_box": gt_box["box"],
                        "pred_box": pred_box["box"],
                    }
                )

    for key in candidate_lists:
        candidate_lists[key].sort(key=lambda item: item["score"], reverse=True)
        fallback_lists[key].sort(key=lambda item: item["score"], reverse=True)
        if not candidate_lists[key]:
            candidate_lists[key] = fallback_lists[key]
        if not candidate_lists[key]:
            raise RuntimeError(f"未自动找到 {key} 候选样例，可扩大扫描范围")

    selected = select_distinct_cases(candidate_lists)
    config = build_cases_config(selected)
    config_path = args.output_dir / "auto_selected_failure_cases.json"
    figure_path = args.output_dir / "failure_case_analysis_auto.png"
    write_json(config, config_path)

    if args.layout == "rows":
        failure_plot.plot_rows(config["cases"], config_path.parent, figure_path, config["figure_title"])
    else:
        failure_plot.plot_cols(config["cases"], config_path.parent, figure_path, config["figure_title"])

    print(f"已保存配置: {config_path.resolve()}")
    print(f"已生成图片: {figure_path.resolve()}")
    print(f"已生成图片: {figure_path.with_suffix('.svg').resolve()}")


if __name__ == "__main__":
    main()
