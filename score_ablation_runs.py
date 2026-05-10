"""
消融实验综合评分与可视化脚本

功能:
    1. 自动扫描 runs/detect 下的消融实验目录
    2. 从每个 results.csv 中提取最佳 epoch 指标
    3. 计算 P、R、F1、mAP@50、mAP@50-95 的综合得分
    4. 输出排名表与汇总 CSV/JSON
    5. 绘制雷达图、折线图、综合评分柱状图

默认使用:
    python score_ablation_runs.py

常用示例:
    python score_ablation_runs.py --include baseline
    python score_ablation_runs.py --experiments ablation-1 ablation-2 ablation-3
    python score_ablation_runs.py --criterion "metrics/mAP50-95(B)"
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

matplotlib.use("Agg")

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150


FIELD_MAP = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "map50": "metrics/mAP50(B)",
    "map50_95": "metrics/mAP50-95(B)",
}

DISPLAY_NAMES = {
    "precision": "P",
    "recall": "R",
    "f1": "F1",
    "map50": "mAP@50",
    "map50_95": "mAP@50-95",
}

DEFAULT_WEIGHTS = {
    "precision": 0.15,
    "recall": 0.15,
    "f1": 0.20,
    "map50": 0.30,
    "map50_95": 0.20,
}

DEFAULT_COLORS = [
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


def parse_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return default
    return float(value)


def parse_weights(weight_args: list[str] | None) -> dict[str, float]:
    weights = DEFAULT_WEIGHTS.copy()
    if not weight_args:
        return weights

    for item in weight_args:
        if "=" not in item:
            raise ValueError(f"权重参数格式错误: {item}，应为 key=value")
        key, raw_value = item.split("=", 1)
        key = key.strip().lower()
        if key not in weights:
            valid = ", ".join(weights.keys())
            raise ValueError(f"未知权重字段: {key}，可选字段: {valid}")
        weights[key] = float(raw_value)

    total = sum(weights.values())
    if total <= 0:
        raise ValueError("权重和必须大于 0")
    return {key: value / total for key, value in weights.items()}


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
        if exp_dir.is_dir() and (exp_dir / "results.csv").exists():
            valid_dirs.append(exp_dir)
            seen.add(resolved)

    return sorted(valid_dirs, key=lambda path: natural_sort_key(path.name))


def read_results_csv(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def compute_row_score(row: dict[str, str], weights: dict[str, float]) -> float:
    precision = parse_float(row, FIELD_MAP["precision"])
    recall = parse_float(row, FIELD_MAP["recall"])
    map50 = parse_float(row, FIELD_MAP["map50"])
    map50_95 = parse_float(row, FIELD_MAP["map50_95"])
    f1 = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)
    return (
        weights["precision"] * precision
        + weights["recall"] * recall
        + weights["f1"] * f1
        + weights["map50"] * map50
        + weights["map50_95"] * map50_95
    ) * 100.0


def choose_best_row(rows: list[dict[str, str]], criterion: str, weights: dict[str, float]) -> dict[str, str]:
    if not rows:
        raise ValueError("results.csv 为空")
    if criterion.lower() == "score":
        return max(rows, key=lambda row: compute_row_score(row, weights))
    if criterion not in rows[0]:
        return rows[-1]
    return max(rows, key=lambda row: parse_float(row, criterion, float("-inf")))


def build_record(exp_dir: Path, criterion: str, weights: dict[str, float]) -> dict[str, float | int | str]:
    rows = read_results_csv(exp_dir / "results.csv")
    best_row = choose_best_row(rows, criterion, weights)

    precision = parse_float(best_row, FIELD_MAP["precision"])
    recall = parse_float(best_row, FIELD_MAP["recall"])
    map50 = parse_float(best_row, FIELD_MAP["map50"])
    map50_95 = parse_float(best_row, FIELD_MAP["map50_95"])
    f1 = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)
    score = compute_row_score(best_row, weights)

    return {
        "experiment": exp_dir.name,
        "epoch": int(parse_float(best_row, "epoch", -1)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map50": map50,
        "map50_95": map50_95,
        "score": score,
        "criterion_value": score if criterion.lower() == "score" else parse_float(best_row, criterion),
        "criterion": criterion,
        "results_csv": str((exp_dir / "results.csv").resolve()),
    }


def save_summary_csv(records: list[dict[str, float | int | str]], output_path: Path) -> None:
    fieldnames = [
        "rank",
        "experiment",
        "epoch",
        "precision",
        "recall",
        "f1",
        "map50",
        "map50_95",
        "score",
        "criterion",
        "criterion_value",
        "results_csv",
    ]

    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow(row)


def save_summary_json(records: list[dict[str, float | int | str]], weights: dict[str, float], output_path: Path) -> None:
    payload = {
        "weights": weights,
        "records": records,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def annotate_bars(ax: plt.Axes, bars, values: list[float], fmt: str = "{:.2f}") -> None:
    top_margin = max(values) * 0.015 if values else 0.0
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + top_margin,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )


def plot_score_bar(records: list[dict[str, float | int | str]], output_path: Path) -> None:
    sorted_records = sorted(records, key=lambda item: float(item["score"]), reverse=True)
    names = [str(item["experiment"]) for item in sorted_records]
    scores = [float(item["score"]) for item in sorted_records]
    colors = DEFAULT_COLORS[: len(names)]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, scores, color=colors, edgecolor="black", linewidth=1.2)
    ax.set_title("消融实验综合评分对比", fontsize=15, fontweight="bold")
    ax.set_ylabel("综合评分", fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(scores) * 1.12 if scores else 1)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.tick_params(axis="x", rotation=15)
    annotate_bars(ax, bars, scores)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_metric_line(records: list[dict[str, float | int | str]], output_path: Path) -> None:
    names = [str(item["experiment"]) for item in records]
    x = np.arange(len(names))
    metric_order = ["precision", "recall", "f1", "map50", "map50_95"]

    fig, ax = plt.subplots(figsize=(11, 6))
    for idx, metric in enumerate(metric_order):
        values = [float(item[metric]) * 100 for item in records]
        ax.plot(
            x,
            values,
            marker="o",
            linewidth=2.2,
            markersize=6,
            color=DEFAULT_COLORS[idx % len(DEFAULT_COLORS)],
            label=DISPLAY_NAMES[metric],
        )
        for xi, yi in zip(x, values):
            ax.text(xi, yi + 0.6, f"{yi:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_title("消融实验关键指标折线图", fontsize=15, fontweight="bold")
    ax.set_ylabel("指标值 (%)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_metric_radar(records: list[dict[str, float | int | str]], output_path: Path) -> None:
    metric_order = ["precision", "recall", "f1", "map50", "map50_95"]
    labels = [DISPLAY_NAMES[metric] for metric in metric_order]
    angles = np.linspace(0, 2 * np.pi, len(metric_order), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 8), subplot_kw={"projection": "polar"})
    for idx, item in enumerate(records):
        values = [float(item[metric]) * 100 for metric in metric_order]
        values += values[:1]
        color = DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]
        ax.plot(angles, values, linewidth=2.0, color=color, label=str(item["experiment"]))
        ax.fill(angles, values, color=color, alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"])
    ax.set_title("消融实验指标雷达图", fontsize=15, fontweight="bold", pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.10))
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def print_summary(records: list[dict[str, float | int | str]], weights: dict[str, float]) -> None:
    weight_desc = ", ".join(f"{DISPLAY_NAMES.get(key, key)}={value:.2f}" for key, value in weights.items())
    print("\n" + "=" * 88)
    print("消融实验综合评分结果")
    print("=" * 88)
    print(f"综合评分权重: {weight_desc}")
    print(
        f"{'Rank':<6}{'Experiment':<16}{'Epoch':<8}{'P':<10}{'R':<10}"
        f"{'F1':<10}{'mAP50':<10}{'mAP50-95':<12}{'Score':<10}"
    )
    print("-" * 88)
    for item in records:
        print(
            f"{int(item['rank']):<6}{str(item['experiment']):<16}{int(item['epoch']):<8}"
            f"{float(item['precision']):<10.4f}{float(item['recall']):<10.4f}"
            f"{float(item['f1']):<10.4f}{float(item['map50']):<10.4f}"
            f"{float(item['map50_95']):<12.4f}{float(item['score']):<10.2f}"
        )
    print("-" * 88)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="消融实验综合评分与可视化")
    parser.add_argument("--root", type=Path, default=Path("runs/detect"), help="实验根目录")
    parser.add_argument(
        "--pattern",
        dest="patterns",
        action="append",
        default=None,
        help="实验匹配模式，可重复传入，默认仅匹配 ablation-*",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        help="显式指定实验目录名，例如 ablation-1 ablation-2 baseline",
    )
    parser.add_argument(
        "--include",
        nargs="+",
        default=None,
        help="在默认 ablation-* 之外附加目录名，例如 --include baseline",
    )
    parser.add_argument(
        "--criterion",
        default="score",
        help='选择最佳 epoch 的依据，默认 "score"，也可传 metrics/mAP50(B) 等原始列名',
    )
    parser.add_argument(
        "--weight",
        action="append",
        default=None,
        help="自定义综合评分权重，例如 --weight map50=0.35 --weight f1=0.25",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plots/ablation_analysis"),
        help="输出目录",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    weights = parse_weights(args.weight)
    patterns = args.patterns or ["ablation-*"]

    experiments = args.experiments[:] if args.experiments else None
    if experiments is None and args.include:
        experiments = []
        for pattern in patterns:
            experiments.extend([path.name for path in args.root.glob(pattern) if path.is_dir()])
        experiments.extend(args.include)

    exp_dirs = collect_experiment_dirs(args.root, patterns, experiments)
    if not exp_dirs:
        raise FileNotFoundError(f"未在 {args.root.resolve()} 下找到可用实验目录")

    ordered_records = [build_record(exp_dir, args.criterion, weights) for exp_dir in exp_dirs]
    ranked_records = sorted(ordered_records, key=lambda item: float(item["score"]), reverse=True)
    for rank, item in enumerate(ranked_records, start=1):
        item["rank"] = rank

    args.output.mkdir(parents=True, exist_ok=True)
    save_summary_csv(ranked_records, args.output / "ablation_summary.csv")
    save_summary_json(ranked_records, weights, args.output / "ablation_summary.json")
    plot_score_bar(ranked_records, args.output / "ablation_score_bar.png")
    plot_metric_line(ordered_records, args.output / "ablation_metric_line.png")
    plot_metric_radar(ordered_records, args.output / "ablation_metric_radar.png")
    print_summary(ranked_records, weights)
    print(f"\n结果已输出到: {args.output.resolve()}")


if __name__ == "__main__":
    main()
