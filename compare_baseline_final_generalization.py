"""
基线模型与最终模型的主数据集/泛化数据集对比实验脚本

功能:
    1. 在主数据集和 AOD4 泛化数据集上分别测试基线模型与最终模型
    2. 汇总 P、R、F1、mAP@50、mAP@50-95
    3. 输出 4 个类别分别的 mAP@50 指标对比与提升
    4. 生成对比图与 CSV/JSON 汇总文件

默认配置:
    基线模型: runs/detect/ablation-1/weights/best.pt
    最终模型: runs/detect/ablation-5/weights/best.pt
    主数据集: ../datasets/Airborne/data.yaml
    泛化数据集: ../datasets/AOD4/data.yaml

示例:
    D:\\Workspace\\Thesis\\.venv\\Scripts\\python.exe compare_baseline_final_generalization.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

matplotlib.use("Agg")

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

METRIC_ORDER = ["precision", "recall", "f1", "map50", "map50_95"]
METRIC_LABELS = {
    "precision": "P",
    "recall": "R",
    "f1": "F1",
    "map50": "mAP@50",
    "map50_95": "mAP@50-95",
}

MODEL_LABELS = {
    "baseline": "基线模型",
    "final": "最终模型",
}

DATASET_LABELS = {
    "main": "主数据集",
    "aod4": "AOD4泛化数据集",
}

MODEL_COLORS = {
    "baseline": "#4C78A8",
    "final": "#E45756",
}

IMPROVEMENT_COLORS = {
    "main": "#F58518",
    "aod4": "#72B7B2",
}


def evaluate_model(
    model_path: Path,
    data_path: Path,
    dataset_key: str,
    model_key: str,
    device: str,
    imgsz: int,
    batch: int,
    conf: float,
    iou: float,
    workers: int,
    eval_project: Path,
) -> dict[str, object]:
    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(data_path),
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
        name=f"{dataset_key}_{model_key}",
        exist_ok=True,
        verbose=False,
    )

    results_dict = metrics.results_dict
    precision = float(results_dict["metrics/precision(B)"])
    recall = float(results_dict["metrics/recall(B)"])
    map50 = float(results_dict["metrics/mAP50(B)"])
    map50_95 = float(results_dict["metrics/mAP50-95(B)"])
    f1 = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)

    summary_rows = metrics.summary()
    per_class_map50 = {row["Class"]: float(row["mAP50"]) for row in summary_rows}
    per_class_map50_95 = {row["Class"]: float(row["mAP50-95"]) for row in summary_rows}

    speed = getattr(metrics, "speed", {}) or {}
    class_names = [metrics.names[i] for i in sorted(metrics.names)]

    return {
        "dataset_key": dataset_key,
        "dataset_name": DATASET_LABELS.get(dataset_key, dataset_key),
        "model_key": model_key,
        "model_name": MODEL_LABELS.get(model_key, model_key),
        "model_path": str(model_path.resolve()),
        "data_path": str(data_path.resolve()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map50": map50,
        "map50_95": map50_95,
        "inference_ms": float(speed.get("inference", 0.0)),
        "class_names": class_names,
        "per_class_map50": per_class_map50,
        "per_class_map50_95": per_class_map50_95,
    }


def save_overall_csv(records: list[dict[str, object]], output_path: Path) -> None:
    fieldnames = [
        "dataset_key",
        "dataset_name",
        "model_key",
        "model_name",
        "precision",
        "recall",
        "f1",
        "map50",
        "map50_95",
        "inference_ms",
        "model_path",
        "data_path",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow({key: row[key] for key in fieldnames})


def save_per_class_csv(records: list[dict[str, object]], output_path: Path) -> None:
    fieldnames = ["dataset_key", "dataset_name", "model_key", "model_name", "class_name", "map50", "map50_95"]
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            class_names = row["class_names"]
            map50 = row["per_class_map50"]
            map50_95 = row["per_class_map50_95"]
            for class_name in class_names:
                writer.writerow(
                    {
                        "dataset_key": row["dataset_key"],
                        "dataset_name": row["dataset_name"],
                        "model_key": row["model_key"],
                        "model_name": row["model_name"],
                        "class_name": class_name,
                        "map50": map50.get(class_name, 0.0),
                        "map50_95": map50_95.get(class_name, 0.0),
                    }
                )


def save_json(records: list[dict[str, object]], output_path: Path) -> None:
    output_path.write_text(json.dumps({"records": records}, ensure_ascii=False, indent=2), encoding="utf-8")


def plot_overall_metrics(records: list[dict[str, object]], output_path: Path) -> None:
    datasets = ["main", "aod4"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    width = 0.32

    for ax, dataset_key in zip(axes, datasets):
        dataset_rows = [row for row in records if row["dataset_key"] == dataset_key]
        dataset_rows.sort(key=lambda row: 0 if row["model_key"] == "baseline" else 1)
        x = np.arange(len(METRIC_ORDER))

        for idx, row in enumerate(dataset_rows):
            values = [float(row[metric]) * 100 for metric in METRIC_ORDER]
            positions = x + (idx - 0.5) * width
            bars = ax.bar(
                positions,
                values,
                width=width,
                color=MODEL_COLORS[str(row["model_key"])],
                edgecolor="black",
                linewidth=0.8,
                label=str(row["model_name"]),
            )
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.8,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )

        baseline_row = next(row for row in dataset_rows if row["model_key"] == "baseline")
        final_row = next(row for row in dataset_rows if row["model_key"] == "final")
        for i, metric in enumerate(METRIC_ORDER):
            base_value = float(baseline_row[metric]) * 100
            final_value = float(final_row[metric]) * 100
            gain = 0.0 if base_value == 0 else (final_value - base_value) / base_value * 100.0
            ax.text(
                i + width / 2,
                final_value * 0.55,
                f"{gain:+.2f}%",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if final_value >= 35 else "black",
                fontweight="bold",
            )

        ax.set_title(str(DATASET_LABELS[dataset_key]), fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([METRIC_LABELS[metric] for metric in METRIC_ORDER], fontsize=10)
        ax.set_ylim(0, 105)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    axes[0].set_ylabel("指标值 (%)", fontsize=12, fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[:2], labels[:2], loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_classwise_map50(records: list[dict[str, object]], output_path: Path) -> None:
    datasets = ["main", "aod4"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    width = 0.35

    for ax, dataset_key in zip(axes, datasets):
        dataset_rows = [row for row in records if row["dataset_key"] == dataset_key]
        baseline_row = next(row for row in dataset_rows if row["model_key"] == "baseline")
        final_row = next(row for row in dataset_rows if row["model_key"] == "final")
        class_names = list(baseline_row["class_names"])
        x = np.arange(len(class_names))
        baseline_values = [float(baseline_row["per_class_map50"].get(name, 0.0)) * 100 for name in class_names]
        final_values = [float(final_row["per_class_map50"].get(name, 0.0)) * 100 for name in class_names]

        baseline_bars = ax.bar(
            x - width / 2,
            baseline_values,
            width=width,
            color=MODEL_COLORS["baseline"],
            edgecolor="black",
            linewidth=0.8,
            label="基线模型",
        )
        final_bars = ax.bar(
            x + width / 2,
            final_values,
            width=width,
            color=MODEL_COLORS["final"],
            edgecolor="black",
            linewidth=0.8,
            label="最终模型",
        )

        for bar, value in zip(baseline_bars, baseline_values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8, f"{value:.2f}", ha="center", va="bottom", fontsize=8)

        for bar, base_value, final_value in zip(final_bars, baseline_values, final_values):
            gain = 0.0 if base_value == 0 else (final_value - base_value) / base_value * 100.0
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8, f"{final_value:.2f}", ha="center", va="bottom", fontsize=8)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 0.55,
                f"{gain:+.2f}%",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if final_value >= 35 else "black",
                fontweight="bold",
            )

        ax.set_title(f"{DATASET_LABELS[dataset_key]}: 4类 mAP@50 对比", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, fontsize=10)
        ax.set_ylim(0, 105)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    axes[0].set_ylabel("mAP@50 (%)", fontsize=12, fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[:2], labels[:2], loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_classwise_improvement(records: list[dict[str, object]], output_path: Path) -> None:
    baseline_main = next(row for row in records if row["dataset_key"] == "main" and row["model_key"] == "baseline")
    final_main = next(row for row in records if row["dataset_key"] == "main" and row["model_key"] == "final")
    baseline_aod4 = next(row for row in records if row["dataset_key"] == "aod4" and row["model_key"] == "baseline")
    final_aod4 = next(row for row in records if row["dataset_key"] == "aod4" and row["model_key"] == "final")

    class_names = list(baseline_main["class_names"])
    main_gain = []
    aod4_gain = []
    for class_name in class_names:
        main_base = float(baseline_main["per_class_map50"].get(class_name, 0.0)) * 100
        main_final = float(final_main["per_class_map50"].get(class_name, 0.0)) * 100
        aod4_base = float(baseline_aod4["per_class_map50"].get(class_name, 0.0)) * 100
        aod4_final = float(final_aod4["per_class_map50"].get(class_name, 0.0)) * 100
        main_gain.append(main_final - main_base)
        aod4_gain.append(aod4_final - aod4_base)

    x = np.arange(len(class_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, main_gain, width=width, color=IMPROVEMENT_COLORS["main"], edgecolor="black", linewidth=0.8, label=DATASET_LABELS["main"])
    bars2 = ax.bar(x + width / 2, aod4_gain, width=width, color=IMPROVEMENT_COLORS["aod4"], edgecolor="black", linewidth=0.8, label=DATASET_LABELS["aod4"])

    for bars in (bars1, bars2):
        for bar in bars:
            value = bar.get_height()
            va = "bottom" if value >= 0 else "top"
            offset = 0.5 if value >= 0 else -0.5
            ax.text(bar.get_x() + bar.get_width() / 2, value + offset, f"{value:+.2f}", ha="center", va=va, fontsize=8, fontweight="bold")

    ax.axhline(0, color="black", linewidth=1.0)
    ax.set_title("各类别 mAP@50 提升幅度对比", fontsize=15, fontweight="bold")
    ax.set_ylabel("最终模型 - 基线模型 (百分点)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def print_summary(records: list[dict[str, object]]) -> None:
    print("\n" + "=" * 120)
    print("主数据集 / AOD4 泛化数据集 对比实验结果")
    print("=" * 120)
    print(f"{'Dataset':<18}{'Model':<12}{'P':<10}{'R':<10}{'F1':<10}{'mAP50':<10}{'mAP50-95':<12}{'Infer(ms)':<10}")
    print("-" * 120)
    for row in records:
        print(
            f"{str(row['dataset_name']):<18}{str(row['model_name']):<12}"
            f"{float(row['precision']):<10.4f}{float(row['recall']):<10.4f}{float(row['f1']):<10.4f}"
            f"{float(row['map50']):<10.4f}{float(row['map50_95']):<12.4f}{float(row['inference_ms']):<10.2f}"
        )
    print("-" * 120)

    for dataset_key in ("main", "aod4"):
        baseline_row = next(row for row in records if row["dataset_key"] == dataset_key and row["model_key"] == "baseline")
        final_row = next(row for row in records if row["dataset_key"] == dataset_key and row["model_key"] == "final")
        print(f"\n{DATASET_LABELS[dataset_key]} 相对基线提升:")
        for metric in METRIC_ORDER:
            base_value = float(baseline_row[metric]) * 100
            final_value = float(final_row[metric]) * 100
            gain = 0.0 if base_value == 0 else (final_value - base_value) / base_value * 100.0
            print(f"  {METRIC_LABELS[metric]:<10}: {base_value:.2f} -> {final_value:.2f} ({gain:+.2f}%)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="基线模型与最终模型在主数据集/AOD4上的对比实验")
    parser.add_argument("--baseline-model", type=Path, default=Path("runs/detect/ablation-1/weights/best.pt"), help="基线模型权重")
    parser.add_argument("--final-model", type=Path, default=Path("runs/detect/ablation-5/weights/best.pt"), help="最终模型权重")
    parser.add_argument("--main-data", type=Path, default=Path("../datasets/Airborne/data.yaml"), help="主数据集 data.yaml")
    parser.add_argument("--aod4-data", type=Path, default=Path("../datasets/AOD4/data.yaml"), help="AOD4 data.yaml")
    parser.add_argument("--device", default="0", help="设备，如 0 或 cpu")
    parser.add_argument("--imgsz", type=int, default=640, help="验证图像尺寸")
    parser.add_argument("--batch", type=int, default=16, help="验证 batch")
    parser.add_argument("--conf", type=float, default=0.001, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU 阈值")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers，沙箱里建议 0")
    parser.add_argument("--output", type=Path, default=Path("plots/baseline_final_generalization"), help="结果输出目录")
    parser.add_argument("--eval-project", type=Path, default=Path("runs/baseline_final_generalization_eval"), help="Ultralytics 缓存目录")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for required_path in (args.baseline_model, args.final_model, args.main_data, args.aod4_data):
        if not required_path.exists():
            raise FileNotFoundError(f"路径不存在: {required_path.resolve()}")

    args.output.mkdir(parents=True, exist_ok=True)
    args.eval_project.mkdir(parents=True, exist_ok=True)

    tasks = [
        ("main", "baseline", args.main_data, args.baseline_model),
        ("main", "final", args.main_data, args.final_model),
        ("aod4", "baseline", args.aod4_data, args.baseline_model),
        ("aod4", "final", args.aod4_data, args.final_model),
    ]

    records: list[dict[str, object]] = []
    for dataset_key, model_key, data_path, model_path in tasks:
        print(f"\n[Eval] {MODEL_LABELS[model_key]} on {DATASET_LABELS[dataset_key]}")
        record = evaluate_model(
            model_path=model_path,
            data_path=data_path,
            dataset_key=dataset_key,
            model_key=model_key,
            device=args.device,
            imgsz=args.imgsz,
            batch=args.batch,
            conf=args.conf,
            iou=args.iou,
            workers=args.workers,
            eval_project=args.eval_project,
        )
        records.append(record)

    save_overall_csv(records, args.output / "baseline_final_overall.csv")
    save_per_class_csv(records, args.output / "baseline_final_per_class.csv")
    save_json(records, args.output / "baseline_final_compare.json")
    plot_overall_metrics(records, args.output / "baseline_final_overall_metrics.png")
    plot_classwise_map50(records, args.output / "baseline_final_classwise_map50.png")
    plot_classwise_improvement(records, args.output / "baseline_final_classwise_improvement.png")
    print_summary(records)
    print(f"\n结果目录: {args.output.resolve()}")


if __name__ == "__main__":
    main()
