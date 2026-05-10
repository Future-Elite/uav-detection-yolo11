"""
典型失败样例分析图生成脚本

支持两种排版:
    1. rows: 3行×2列，每行一个失败类型（默认）
    2. cols: 2行×3列，每列一个失败类型

每个失败类型包含:
    - original_image: 原图
    - result_image: 结果图
    - original_boxes/result_boxes: 可选框
    - annotations: 可选箭头说明

使用示例:
    D:\\Workspace\\Thesis\\.venv\\Scripts\\python.exe plot_failure_case_analysis.py ^
        --config configs\\failure_case_analysis_template.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

matplotlib.use("Agg")

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 180

DEFAULT_COLORS = {
    "pred": "#E74C3C",
    "gt": "#2ECC71",
    "fp": "#F39C12",
    "note": "#2C3E50",
}


def load_config(config_path: Path) -> dict:
    return json.loads(config_path.read_text(encoding="utf-8"))


def resolve_path(path_text: str, config_dir: Path) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = (config_dir / path).resolve()
    return path


def draw_boxes(ax: plt.Axes, boxes: list[dict]) -> None:
    for item in boxes:
        x1, y1, x2, y2 = item["box"]
        color = item.get("color", DEFAULT_COLORS["pred"])
        linewidth = float(item.get("linewidth", 2.0))
        linestyle = item.get("linestyle", "-")

        rect = mpatches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            edgecolor=color,
            linewidth=linewidth,
            linestyle=linestyle,
        )
        ax.add_patch(rect)

        label = item.get("label")
        if label:
            ax.text(
                x1,
                max(y1 - 6, 8),
                label,
                fontsize=9,
                fontweight="bold",
                color=color,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5),
            )


def draw_annotations(ax: plt.Axes, annotations: list[dict]) -> None:
    for item in annotations:
        xy = item["xy"]
        xytext = item["xytext"]
        text = item["text"]
        color = item.get("color", DEFAULT_COLORS["note"])
        ax.annotate(
            text,
            xy=xy,
            xytext=xytext,
            textcoords="data",
            fontsize=9,
            fontweight="bold",
            color=color,
            arrowprops=dict(arrowstyle="->", color=color, linewidth=1.8),
            bbox=dict(facecolor="white", edgecolor=color, alpha=0.88, pad=2.0),
        )


def render_panel(ax: plt.Axes, image_path: Path, title: str, boxes: list[dict], annotations: list[dict]) -> None:
    image = mpimg.imread(image_path)
    ax.imshow(image)
    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=6)
    ax.axis("off")
    if boxes:
        draw_boxes(ax, boxes)
    if annotations:
        draw_annotations(ax, annotations)


def plot_rows(cases: list[dict], config_dir: Path, output_path: Path, figure_title: str) -> None:
    fig, axes = plt.subplots(len(cases), 2, figsize=(12, 15))

    for idx, case in enumerate(cases):
        original_path = resolve_path(case["original_image"], config_dir)
        result_path = resolve_path(case["result_image"], config_dir)
        left_title = f"({case['tag']}) {case['title']} - 原图"
        right_title = f"({case['tag']}) {case['title']} - 检测结果"

        render_panel(
            axes[idx, 0],
            original_path,
            left_title,
            case.get("original_boxes", []),
            [ann for ann in case.get("annotations", []) if ann.get("panel", "result") == "original"],
        )
        render_panel(
            axes[idx, 1],
            result_path,
            right_title,
            case.get("result_boxes", []),
            [ann for ann in case.get("annotations", []) if ann.get("panel", "result") == "result"],
        )

        description = case.get("description", "")
        if description:
            axes[idx, 0].text(
                0.0,
                -0.08,
                description,
                transform=axes[idx, 0].transAxes,
                fontsize=9.2,
                color="#333333",
                va="top",
            )

    fig.suptitle(figure_title, fontsize=16, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_cols(cases: list[dict], config_dir: Path, output_path: Path, figure_title: str) -> None:
    fig, axes = plt.subplots(2, len(cases), figsize=(16, 8.6))

    for idx, case in enumerate(cases):
        original_path = resolve_path(case["original_image"], config_dir)
        result_path = resolve_path(case["result_image"], config_dir)
        top_title = f"({case['tag']}) {case['title']} - 原图"
        bottom_title = f"({case['tag']}) {case['title']} - 检测结果"

        render_panel(
            axes[0, idx],
            original_path,
            top_title,
            case.get("original_boxes", []),
            [ann for ann in case.get("annotations", []) if ann.get("panel", "result") == "original"],
        )
        render_panel(
            axes[1, idx],
            result_path,
            bottom_title,
            case.get("result_boxes", []),
            [ann for ann in case.get("annotations", []) if ann.get("panel", "result") == "result"],
        )

        description = case.get("description", "")
        if description:
            axes[1, idx].text(
                0.0,
                -0.10,
                description,
                transform=axes[1, idx].transAxes,
                fontsize=8.8,
                color="#333333",
                va="top",
            )

    fig.suptitle(figure_title, fontsize=16, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成典型失败样例分析图")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/failure_case_analysis_template.json"),
        help="JSON 配置文件路径",
    )
    parser.add_argument(
        "--layout",
        choices=["rows", "cols"],
        default="rows",
        help="rows=3行2列，cols=2行3列",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plots/failure_case_analysis/failure_case_analysis.png"),
        help="输出 PNG 路径",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.config.exists():
        raise FileNotFoundError(f"配置文件不存在: {args.config.resolve()}")

    config = load_config(args.config)
    cases = config.get("cases", [])
    if len(cases) != 3:
        raise ValueError("配置文件必须包含 3 个失败样例 case")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure_title = config.get("figure_title", "典型失败样例分析图")

    if args.layout == "rows":
        plot_rows(cases, args.config.parent, args.output, figure_title)
    else:
        plot_cols(cases, args.config.parent, args.output, figure_title)

    print(f"已生成: {args.output.resolve()}")
    print(f"已生成: {args.output.with_suffix('.svg').resolve()}")


if __name__ == "__main__":
    main()
