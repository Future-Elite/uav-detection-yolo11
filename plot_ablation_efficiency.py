from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 600


OUTPUT_DIR = Path("plots/ablation_test_analysis")
OUTPUT_PATH = OUTPUT_DIR / "ablation_efficiency_speed_params_fps.png"

EXPERIMENT_LABELS = [
    "（1）基线\nYOLO11n",
    "（2）基线\n+CSPPC",
    "（3）基线\n+CSPPC+ECA",
    "（4）基线\n+CSPPC+ECA\n+SPPELAN",
    "（5）基线\n+CSPPC+ECA\n+SPPELAN\n+Enhanced P3+SIoU",
]

INFERENCE_MS = [2.75, 2.70, 2.72, 2.69, 4.66]
PARAMS_M = [2.59, 2.66, 2.66, 2.59, 4.09]
FPS = [363.2, 370.2, 367.9, 372.1, 214.4]


def annotate_bars(ax: plt.Axes, bars, values: list[float]) -> None:
    offset = max(values) * 0.018
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )


def plot_efficiency(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(EXPERIMENT_LABELS))
    width = 0.34

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(16, 9.5),
        gridspec_kw={"height_ratios": [1, 1.05], "hspace": 0.18},
    )

    ax_delay = axes[0]
    ax_params = ax_delay.twinx()

    delay_bars = ax_delay.bar(
        x - width / 2,
        INFERENCE_MS,
        width=width,
        label="推理延迟",
        color="#1f77b4",
        edgecolor="black",
        linewidth=1.2,
    )
    params_bars = ax_params.bar(
        x + width / 2,
        PARAMS_M,
        width=width,
        label="参数量",
        color="#ff7f0e",
        edgecolor="black",
        linewidth=1.2,
    )

    annotate_bars(ax_delay, delay_bars, INFERENCE_MS)
    annotate_bars(ax_params, params_bars, PARAMS_M)

    ax_delay.set_ylabel("推理延迟 (ms)", fontsize=14, fontweight="bold")
    ax_params.set_ylabel("参数量 (M)", fontsize=14, fontweight="bold")
    ax_delay.set_ylim(0, 5.7)
    ax_params.set_ylim(0, 5.0)
    ax_delay.set_xticks(x)
    ax_delay.set_xticklabels([])
    ax_delay.grid(axis="y", linestyle="--", alpha=0.28)
    ax_delay.tick_params(axis="y", labelsize=12)
    ax_params.tick_params(axis="y", labelsize=12)

    handles = [delay_bars, params_bars]
    labels = ["推理延迟", "参数量"]
    ax_delay.legend(
        handles,
        labels,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        fontsize=12,
        frameon=True,
    )

    ax_fps = axes[1]
    ax_fps.plot(
        x,
        FPS,
        marker="o",
        markersize=7,
        linewidth=2.6,
        color="#2ca02c",
        label="FPS",
    )
    for xi, yi in zip(x, FPS):
        ax_fps.text(
            xi,
            yi + 4.0,
            f"{yi:.1f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax_fps.set_ylabel("FPS", fontsize=14, fontweight="bold")
    ax_fps.set_xticks(x)
    ax_fps.set_xticklabels(EXPERIMENT_LABELS, fontsize=12, fontweight="bold")
    ax_fps.set_ylim(195, 400)
    ax_fps.grid(True, linestyle="--", alpha=0.28)
    ax_fps.tick_params(axis="y", labelsize=12)
    ax_fps.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14), fontsize=12, frameon=True)

    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    plot_efficiency(OUTPUT_PATH)
    print(f"Saved: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
