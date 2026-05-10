from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


ROOT = Path(__file__).resolve().parent
RUN_DIR = ROOT / "runs" / "detect" / "baseline"
CSV_PATH = RUN_DIR / "results.csv"
OUTPUT_DIR = ROOT / "plots" / "baseline_training_curves"
OUTPUT_PNG = OUTPUT_DIR / "baseline_training_curves.png"
OUTPUT_SVG = OUTPUT_DIR / "baseline_training_curves.svg"
TITLE_FONT = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=12, weight="bold")

def load_results(csv_path: Path) -> dict[str, list[float]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"未找到训练日志: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"训练日志为空: {csv_path}")

    data: dict[str, list[float]] = {}
    for field in reader.fieldnames or []:
        data[field] = []
        for row in rows:
            value = row[field].strip()
            data[field].append(float(value))
    return data


def add_metric_annotation(ax: plt.Axes, xs: list[float], ys: list[float], color: str) -> None:
    final_x = xs[-1]
    final_y = ys[-1]
    ax.axhline(final_y, color=color, linestyle=":", linewidth=1.2, alpha=0.65)
    ax.text(
        final_x * 0.7,
        final_y + 0.015,
        f"{final_y:.4f}",
        color=color,
        fontsize=9,
        fontweight="bold",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_results(CSV_PATH)

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 180

    epochs = data["epoch"]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.5))

    ax = axes[0, 0]
    map50 = data["metrics/mAP50(B)"]
    map50_95 = data["metrics/mAP50-95(B)"]
    ax.plot(epochs, map50, color="#1b9e9c", linewidth=1.8, label="mAP@50")
    ax.plot(epochs, map50_95, color="#8c1d2d", linewidth=1.6, label="mAP@50-95")
    add_metric_annotation(ax, epochs, map50, "#1b9e9c")
    add_metric_annotation(ax, epochs, map50_95, "#8c1d2d")
    ax.set_title("(a) mAP曲线", fontproperties=TITLE_FONT)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP")
    ax.set_xlim(1, epochs[-1])
    ax.set_ylim(0, 1.0)
    ax.legend(loc="lower right", frameon=True)

    ax = axes[0, 1]
    precision = data["metrics/precision(B)"]
    recall = data["metrics/recall(B)"]
    ax.plot(epochs, precision, color="#ff8c00", linewidth=1.7, label="Precision")
    ax.plot(epochs, recall, color="#5b2c6f", linewidth=1.6, label="Recall")
    ax.set_title("(b) Precision与Recall曲线", fontproperties=TITLE_FONT)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.set_xlim(1, epochs[-1])
    ax.set_ylim(0, 1.0)
    ax.legend(loc="lower right", frameon=True)

    ax = axes[1, 0]
    train_box = data["train/box_loss"]
    val_box = data["val/box_loss"]
    ax.plot(epochs, train_box, color="#4ea3d8", linewidth=1.5, label="Train")
    ax.plot(epochs, val_box, color="#ef6f6c", linewidth=1.5, linestyle="--", label="Val")
    ax.set_title("(c) 边框损失曲线", fontproperties=TITLE_FONT)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Box Loss")
    ax.set_xlim(1, epochs[-1])
    ax.legend(loc="upper right", frameon=True)

    ax = axes[1, 1]
    train_cls = data["train/cls_loss"]
    val_cls = data["val/cls_loss"]
    ax.plot(epochs, train_cls, color="#4ea3d8", linewidth=1.5, label="Train")
    ax.plot(epochs, val_cls, color="#ef6f6c", linewidth=1.5, linestyle="--", label="Val")
    ax.set_title("(d) 分类损失曲线", fontproperties=TITLE_FONT)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Class Loss")
    ax.set_xlim(1, epochs[-1])
    ax.legend(loc="upper right", frameon=True)

    for ax in axes.flat:
        ax.grid(True, linestyle="--", alpha=0.45)

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_SVG, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"已保存: {OUTPUT_PNG}")
    print(f"已保存: {OUTPUT_SVG}")


if __name__ == "__main__":
    main()
