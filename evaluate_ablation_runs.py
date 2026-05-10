"""
基于真实测试集的消融实验评估与可视化脚本

功能:
    1. 扫描 runs/detect/ablation-* 下的 best.pt
    2. 在指定数据集上执行 model.val(split='test')
    3. 汇总 P、R、F1、mAP@50、mAP@50-95
    4. 输出 CSV、JSON 与对比图

示例:
    D:\\Workspace\\Thesis\\.venv\\Scripts\\python.exe evaluate_ablation_runs.py
    D:\\Workspace\\Thesis\\.venv\\Scripts\\python.exe evaluate_ablation_runs.py --include baseline
    D:\\Workspace\\Thesis\\.venv\\Scripts\\python.exe evaluate_ablation_runs.py --device cpu
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

matplotlib.use("Agg")

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150


DISPLAY_NAMES = {
    "precision": "P",
    "recall": "R",
    "f1": "F1",
    "map50": "mAP@50",
    "map50_95": "mAP@50-95",
}

EXPERIMENT_NAME_MAP = {
    "ablation-1": "基线：YOLO11n",
    "ablation-2": "基线+CSPPC",
    "ablation-3": "基线+CSPPC+ECA",
    "ablation-4": "基线+CSPPC+ECA+SPPELAN",
    "ablation-5": "基线+CSPPC+ECA+SPPELAN+Enhanced P3+SIoU",
}

EXPERIMENT_PLOT_LABEL_MAP = {
    "ablation-1": "（1）基线\nYOLO11n",
    "ablation-2": "（2）基线\n+CSPPC",
    "ablation-3": "（3）基线\n+CSPPC+ECA",
    "ablation-4": "（4）基线\n+CSPPC+ECA\n+SPPELAN",
    "ablation-5": "（5）基线\n+CSPPC+ECA\n+SPPELAN\n+Enhanced P3+SIoU",
}

COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
]


def natural_sort_key(text: str) -> list[object]:
    return [int(item) if item.isdigit() else item.lower() for item in re.split(r"(\d+)", text)]


def get_experiment_display_name(experiment: str) -> str:
    return EXPERIMENT_NAME_MAP.get(experiment, experiment)


def get_experiment_plot_label(experiment: str) -> str:
    return EXPERIMENT_PLOT_LABEL_MAP.get(experiment, get_experiment_display_name(experiment))


def collect_experiment_dirs(root: Path, patterns: list[str], experiments: list[str] | None) -> list[Path]:
    if experiments:
        dirs = [root / name for name in experiments]
    else:
        dirs = []
        for pattern in patterns:
            dirs.extend(root.glob(pattern))

    valid_dirs: list[Path] = []
    seen: set[Path] = set()
    for exp_dir in dirs:
        resolved = exp_dir.resolve()
        if resolved in seen:
            continue
        if exp_dir.is_dir() and (exp_dir / "weights" / "best.pt").exists():
            valid_dirs.append(exp_dir)
            seen.add(resolved)

    return sorted(valid_dirs, key=lambda path: natural_sort_key(path.name))


def evaluate_one_experiment(
    exp_dir: Path,
    data: Path,
    imgsz: int,
    batch: int,
    conf: float,
    iou: float,
    device: str,
    workers: int,
    eval_project: Path,
) -> dict[str, float | int | str]:
    model_path = exp_dir / "weights" / "best.pt"
    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(data),
        split="test",
        imgsz=imgsz,
        batch=batch,
        conf=conf,
        iou=iou,
        device=device,
        workers=workers,
        plots=False,
        save_json=False,
        project=str(eval_project),
        name=exp_dir.name,
        exist_ok=True,
        verbose=False,
    )

    results_dict = metrics.results_dict
    precision = float(results_dict["metrics/precision(B)"])
    recall = float(results_dict["metrics/recall(B)"])
    map50 = float(results_dict["metrics/mAP50(B)"])
    map50_95 = float(results_dict["metrics/mAP50-95(B)"])
    f1 = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)

    speed = getattr(metrics, "speed", {}) or {}
    return {
        "experiment": exp_dir.name,
        "display_name": get_experiment_display_name(exp_dir.name),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map50": map50,
        "map50_95": map50_95,
        "preprocess_ms": float(speed.get("preprocess", 0.0)),
        "inference_ms": float(speed.get("inference", 0.0)),
        "postprocess_ms": float(speed.get("postprocess", 0.0)),
        "model_path": str(model_path.resolve()),
    }


def save_summary_csv(records: list[dict[str, float | int | str]], output_path: Path) -> None:
    fieldnames = [
        "experiment",
        "display_name",
        "precision",
        "recall",
        "f1",
        "map50",
        "map50_95",
        "preprocess_ms",
        "inference_ms",
        "postprocess_ms",
        "model_path",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow({key: row[key] for key in fieldnames})


def save_summary_json(records: list[dict[str, float | int | str]], output_path: Path) -> None:
    output_path.write_text(json.dumps({"records": records}, ensure_ascii=False, indent=2), encoding="utf-8")


def save_figure_dual(fig: plt.Figure, output_path: Path) -> None:
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")


def annotate_bars(ax: plt.Axes, bars, values: list[float], fmt: str = "{:.2f}") -> None:
    offset = max(values) * 0.015 if values else 0.0
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )


def annotate_grouped_bar_with_gain(
    ax: plt.Axes,
    bars,
    values: list[float],
    baseline_value: float,
) -> None:
    top_offset = max(values) * 0.012 if values else 0.0
    for idx, (bar, value) in enumerate(zip(bars, values)):
        gain = 0.0 if baseline_value == 0 else (value - baseline_value) / baseline_value * 100.0
        gain_text = "基线" if idx == 0 else f"{gain:+.2f}%"
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()
        ax.text(
            x,
            y * 0.55,
            gain_text,
            ha="center",
            va="center",
            fontsize=7.2,
            fontweight="bold",
            color="white" if y >= 35 else "black",
        )
        ax.text(
            x,
            y + top_offset,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=7.8,
            fontweight="bold",
        )


def plot_grouped_bar(records: list[dict[str, float | int | str]], output_path: Path) -> None:
    names = [get_experiment_plot_label(str(item["experiment"])) for item in records]
    metric_order = ["precision", "recall", "f1", "map50", "map50_95"]
    x = np.arange(len(names))
    width = 0.15

    fig, ax = plt.subplots(figsize=(12, 6.5))
    for idx, metric in enumerate(metric_order):
        values = [float(item[metric]) * 100 for item in records]
        positions = x + (idx - 2) * width
        bars = ax.bar(
            positions,
            values,
            width=width,
            label=DISPLAY_NAMES[metric],
            color=COLORS[idx % len(COLORS)],
            edgecolor="black",
            linewidth=0.8,
        )
        annotate_grouped_bar_with_gain(ax, bars, values, baseline_value=values[0])

    # ax.set_title("测试集多指标分组柱状图", fontsize=15, fontweight="bold")
    ax.set_ylabel("指标值 (%)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=0, ha="center", fontsize=8.5)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    save_figure_dual(fig, output_path)
    plt.close(fig)


def plot_metric_line(records: list[dict[str, float | int | str]], output_path: Path) -> None:
    names = [get_experiment_plot_label(str(item["experiment"])) for item in records]
    x = np.arange(len(names))
    top_metrics = ["precision", "recall", "f1"]
    bottom_metrics = ["map50", "map50_95"]

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for idx, metric in enumerate(top_metrics):
        values = [float(item[metric]) * 100 for item in records]
        axes[0].plot(
            x,
            values,
            marker="o",
            linewidth=2.2,
            markersize=6,
            color=COLORS[idx % len(COLORS)],
            label=DISPLAY_NAMES[metric],
        )

    for idx, metric in enumerate(bottom_metrics, start=3):
        values = [float(item[metric]) * 100 for item in records]
        axes[1].plot(
            x,
            values,
            marker="s",
            linewidth=2.2,
            markersize=6,
            color=COLORS[idx % len(COLORS)],
            label=DISPLAY_NAMES[metric],
        )

    # axes[0].set_title("测试集关键指标折线图", fontsize=15, fontweight="bold")
    axes[0].set_ylabel("P / R / F1 (%)", fontsize=12, fontweight="bold")
    axes[0].set_ylim(80, 100)
    axes[0].grid(True, linestyle="--", alpha=0.3)
    axes[0].legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.20))

    axes[1].set_ylabel("mAP (%)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("实验", fontsize=12, fontweight="bold")
    axes[1].set_ylim(50, 100)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=0, ha="center", fontsize=8.5)
    axes[1].grid(True, linestyle="--", alpha=0.3)
    axes[1].legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.18))

    fig.tight_layout()
    save_figure_dual(fig, output_path)
    plt.close(fig)


def plot_bar_line_combined(records: list[dict[str, float | int | str]], output_path: Path) -> None:
    names = [get_experiment_plot_label(str(item["experiment"])) for item in records]
    metric_order = ["precision", "recall", "f1", "map50", "map50_95"]
    x = np.arange(len(names))
    width = 0.15

    fig, axes = plt.subplots(1, 2, figsize=(18, 6.8))

    bar_ax = axes[0]
    for idx, metric in enumerate(metric_order):
        values = [float(item[metric]) * 100 for item in records]
        positions = x + (idx - 2) * width
        bars = bar_ax.bar(
            positions,
            values,
            width=width,
            label=DISPLAY_NAMES[metric],
            color=COLORS[idx % len(COLORS)],
            edgecolor="black",
            linewidth=0.8,
        )
        annotate_grouped_bar_with_gain(bar_ax, bars, values, baseline_value=values[0])

    bar_ax.set_title("（a）多指标分组柱状图", fontsize=14, fontweight="bold")
    bar_ax.set_ylabel("指标值 (%)", fontsize=12, fontweight="bold")
    bar_ax.set_xticks(x)
    bar_ax.set_xticklabels(names, rotation=0, ha="center", fontsize=8.5)
    bar_ax.set_ylim(0, 105)
    bar_ax.grid(axis="y", linestyle="--", alpha=0.3)
    bar_ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.14), fontsize=9)

    line_ax = axes[1]
    for idx, metric in enumerate(metric_order):
        values = [float(item[metric]) * 100 for item in records]
        marker = "o" if idx < 3 else "s"
        line_ax.plot(
            x,
            values,
            marker=marker,
            linewidth=2.2,
            markersize=6,
            color=COLORS[idx % len(COLORS)],
            label=DISPLAY_NAMES[metric],
        )

    line_ax.set_title("（b）关键指标折线图", fontsize=14, fontweight="bold")
    line_ax.set_ylabel("指标值 (%)", fontsize=12, fontweight="bold")
    line_ax.set_xticks(x)
    line_ax.set_xticklabels(names, rotation=0, ha="center", fontsize=8.5)
    line_ax.set_ylim(50, 100)
    line_ax.grid(True, linestyle="--", alpha=0.3)
    line_ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.14), fontsize=9)

    fig.tight_layout()
    save_figure_dual(fig, output_path)
    plt.close(fig)


def plot_metric_radar(records: list[dict[str, float | int | str]], output_path: Path, radar_base: float = 80.0) -> None:
    metric_order = ["precision", "recall", "f1", "map50"]
    labels = [DISPLAY_NAMES[metric] for metric in metric_order]
    angles = np.linspace(0, 2 * np.pi, len(metric_order), endpoint=False).tolist()
    angles += angles[:1]
    all_values = [float(item[metric]) * 100 for item in records for metric in metric_order]
    min_value = min(all_values)
    radar_floor = float(radar_base)
    if min_value < radar_floor:
        radar_floor = float(np.floor(min_value / 5.0) * 5.0)
    radar_ticks = list(np.arange(radar_floor, 101, 10))

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": "polar"})
    for idx, item in enumerate(records):
        values = [max(float(item[metric]) * 100 - radar_floor, 0.0) for metric in metric_order]
        values += values[:1]
        color = COLORS[idx % len(COLORS)]
        ax.plot(angles, values, linewidth=2.0, color=color, label=get_experiment_plot_label(str(item["experiment"])))
        ax.fill(angles, values, color=color, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 100 - radar_floor)
    ax.set_yticks([tick - radar_floor for tick in radar_ticks])
    ax.set_yticklabels([f"{int(tick)}" for tick in radar_ticks], fontsize=9)
    ax.grid(alpha=0.35)
    # ax.set_title("测试集指标雷达图", fontsize=15, fontweight="bold", pad=22)
    ax.legend(loc="upper right", bbox_to_anchor=(1.34, 1.12), fontsize=8.5)
    fig.tight_layout()
    save_figure_dual(fig, output_path)
    plt.close(fig)


def plot_metric_heatmap(records: list[dict[str, float | int | str]], output_path: Path) -> None:
    metric_order = ["precision", "recall", "f1", "map50", "map50_95"]
    metric_labels = [DISPLAY_NAMES[metric] for metric in metric_order]
    exp_names = [get_experiment_plot_label(str(item["experiment"])) for item in records]
    values = np.array([[float(item[metric]) * 100 for item in records] for metric in metric_order], dtype=float)
    color_values = np.zeros_like(values)
    for i, row in enumerate(values):
        row_min = float(np.min(row))
        row_max = float(np.max(row))
        if row_max - row_min < 1e-6:
            color_values[i, :] = 0.5
        else:
            color_values[i, :] = (row - row_min) / (row_max - row_min)

    fig, ax = plt.subplots(figsize=(12, 5.6))
    image = ax.imshow(color_values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    # ax.set_title("测试集指标热力图", fontsize=15, fontweight="bold")
    ax.set_xticks(np.arange(len(exp_names)))
    ax.set_xticklabels(exp_names, rotation=0, ha="center", fontsize=8.5)
    ax.set_yticks(np.arange(len(metric_labels)))
    ax.set_yticklabels(metric_labels, fontsize=9)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            text_color = "black"
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", color=text_color, fontsize=9)

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("行内相对水平", fontsize=11)
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.set_ticklabels(["低", "中", "高"])
    fig.tight_layout()
    save_figure_dual(fig, output_path)
    plt.close(fig)


def print_summary(records: list[dict[str, float | int | str]]) -> None:
    print("\n" + "=" * 132)
    print("Airborne 测试集消融实验评估结果")
    print("=" * 132)
    print(
        f"{'Experiment':<52}{'Dir':<16}{'P':<10}{'R':<10}{'F1':<10}"
        f"{'mAP50':<10}{'mAP50-95':<12}{'Infer(ms)':<10}"
    )
    print("-" * 132)
    for item in records:
        print(
            f"{str(item.get('display_name', item['experiment'])):<52}"
            f"{str(item['experiment']):<16}"
            f"{float(item['precision']):<10.4f}{float(item['recall']):<10.4f}"
            f"{float(item['f1']):<10.4f}{float(item['map50']):<10.4f}"
            f"{float(item['map50_95']):<12.4f}"
            f"{float(item['inference_ms']):<10.2f}"
        )
    print("-" * 132)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="基于测试集的消融实验综合评估")
    parser.add_argument("--root", type=Path, default=Path("runs/detect"), help="实验目录根路径")
    parser.add_argument("--data", type=Path, default=Path("../datasets/Airborne/data.yaml"), help="数据集 YAML")
    parser.add_argument("--pattern", dest="patterns", action="append", default=None, help="实验目录匹配模式，可重复传入")
    parser.add_argument("--experiments", nargs="+", default=None, help="显式指定实验名")
    parser.add_argument("--include", nargs="+", default=None, help="默认 ablation-* 之外额外加入的实验名")
    parser.add_argument("--imgsz", type=int, default=640, help="输入尺寸")
    parser.add_argument("--batch", type=int, default=16, help="验证 batch")
    parser.add_argument("--conf", type=float, default=0.001, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU 阈值")
    parser.add_argument("--device", default="cpu", help="设备，如 0 或 cpu")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers，沙箱里建议 0")
    parser.add_argument("--radar-base", type=float, default=80.0, help="雷达图基底，默认 80；若真实数据低于该值会自动下调")
    parser.add_argument("--output", type=Path, default=Path("plots/ablation_test_analysis"), help="结果输出目录")
    parser.add_argument("--eval-project", type=Path, default=Path("runs/ablation_test_eval"), help="Ultralytics 评估缓存目录")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    patterns = args.patterns or ["ablation-*"]

    experiments = args.experiments[:] if args.experiments else None
    if experiments is None and args.include:
        experiments = []
        for pattern in patterns:
            experiments.extend([path.name for path in args.root.glob(pattern) if path.is_dir()])
        experiments.extend(args.include)

    exp_dirs = collect_experiment_dirs(args.root, patterns, experiments)
    if not exp_dirs:
        raise FileNotFoundError(f"未找到可评估实验目录: {args.root.resolve()}")
    if not args.data.exists():
        raise FileNotFoundError(f"数据集配置不存在: {args.data.resolve()}")

    args.output.mkdir(parents=True, exist_ok=True)
    args.eval_project.mkdir(parents=True, exist_ok=True)

    records = []
    for exp_dir in exp_dirs:
        print(f"\n[Eval] {exp_dir.name} on {args.data}")
        record = evaluate_one_experiment(
            exp_dir=exp_dir,
            data=args.data,
            imgsz=args.imgsz,
            batch=args.batch,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            workers=args.workers,
            eval_project=args.eval_project,
        )
        records.append(record)

    save_summary_csv(records, args.output / "ablation_test_summary.csv")
    save_summary_json(records, args.output / "ablation_test_summary.json")
    plot_grouped_bar(records, args.output / "ablation_test_grouped_bar.png")
    plot_metric_line(records, args.output / "ablation_test_metric_line.png")
    plot_bar_line_combined(records, args.output / "ablation_test_bar_line_combined.png")
    plot_metric_radar(records, args.output / "ablation_test_metric_radar.png", radar_base=args.radar_base)
    plot_metric_heatmap(records, args.output / "ablation_test_metric_heatmap.png")
    print_summary(records)
    print(f"\n结果目录: {args.output.resolve()}")


if __name__ == "__main__":
    main()
