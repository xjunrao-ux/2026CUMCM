"""绘制问题四参数下恒定半径与收缩半径的对比图。

输入：result/A_q4_radius_comparison 中的正式 CSV/JSON。
输出：picture/A_q4_radius_comparison/A_q4_radius_comparison.png。

运行：python code/a4_fixed_radius_comparison_visualization.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FIXED = "#4C78A8"
SHRINK = "#E07A5F"
ACCENT = "#2A9D8F"
GRAY = "#6B7280"
LIGHT = "#E5E7EB"


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Arial",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "font.size": 7.2,
            "axes.titlesize": 8.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.7,
            "axes.linewidth": 0.75,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def interpolate_row_at_time(frame: pd.DataFrame, time_s: float, column: str) -> float:
    times = frame["time_s"].to_numpy(dtype=float)
    values = frame[column].to_numpy(dtype=float)
    valid = np.isfinite(times) & np.isfinite(values)
    if not np.any(valid):
        return float("nan")
    sample_times = times[valid]
    sample_values = values[valid]
    if not np.all(np.diff(sample_times) > 0.0):
        raise ValueError(f"{column} 的插值时间必须严格递增")
    return float(np.interp(time_s, sample_times, sample_values))


def normalized_profile(
    frame: pd.DataFrame, prefix: str, time_h: float, radius_cm: float
) -> tuple[np.ndarray, np.ndarray]:
    time_s = time_h * 3600.0
    xi = np.linspace(0.0, 1.0, 101)
    physical_columns = [f"{prefix}_r_{r:.1f}_cm" for r in np.arange(0.0, 2.01, 0.1)]
    radii = []
    values = []
    for radius, column in zip(np.arange(0.0, 2.01, 0.1), physical_columns):
        value = interpolate_row_at_time(frame, time_s, column)
        if np.isfinite(value) and radius <= radius_cm + 1.0e-10:
            radii.append(float(radius))
            values.append(value)
    surface = interpolate_row_at_time(frame, time_s, f"{prefix}_surface")
    if not radii or radii[-1] < radius_cm - 1.0e-10:
        radii.append(radius_cm)
        values.append(surface)
    else:
        values[-1] = surface
    radii_array = np.asarray(radii)
    values_array = np.asarray(values)
    if not np.all(np.diff(radii_array) > 0.0):
        raise ValueError("径向插值坐标必须严格递增")
    return xi, np.interp(xi * radius_cm, radii_array, values_array)


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.13,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--figure-dir", type=Path)
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    repo_root = args.repo_root.resolve() if args.repo_root else script_path.parents[1]
    result_dir = (
        args.result_dir.resolve()
        if args.result_dir
        else repo_root / "result" / "A_q4_radius_comparison"
    )
    figure_dir = (
        args.figure_dir.resolve()
        if args.figure_dir
        else repo_root / "picture" / "A_q4_radius_comparison"
    )
    figure_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads(
        (result_dir / "A_q4_radius_comparison_summary.json").read_text(
            encoding="utf-8"
        )
    )
    frame = pd.read_csv(result_dir / "A_q4_radius_comparison_full_60s.csv")
    setup_style()

    fixed_h = summary["constant_radius"]["drying_time_h"]
    shrink_h = summary["shrinking_radius"]["drying_time_h"]
    saved_h = summary["shrinkage_impact"]["time_saved_h"]
    reduction = summary["shrinkage_impact"]["time_reduction_pct"]
    common = frame[frame["time_s"] <= summary["shrinking_radius"]["drying_time_s"]].copy()

    fig = plt.figure(figsize=(7.09, 5.35), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.12, 1.0], width_ratios=[1.35, 1.0])

    ax = fig.add_subplot(grid[0, 0])
    ax.plot(frame["time_h"], frame["fixed_r_0.0_cm"], color=FIXED, lw=1.8, label="恒定半径")
    ax.plot(frame["time_h"], frame["shrink_r_0.0_cm"], color=SHRINK, lw=1.8, label="实测收缩半径")
    ax.axhline(0.15, color=GRAY, lw=0.9, ls="--")
    ax.axvspan(shrink_h, fixed_h, color=ACCENT, alpha=0.10, lw=0)
    ax.text(
        (shrink_h + fixed_h) / 2,
        0.62,
        f"节省 {saved_h:.1f} h",
        color=ACCENT,
        ha="center",
        va="center",
        fontsize=7.5,
        fontweight="bold",
    )
    ax.scatter([shrink_h, fixed_h], [0.15, 0.15], c=[SHRINK, FIXED], s=18, zorder=5)
    ax.set(xlabel="时间 / h", ylabel="中心水分浓度 / (kg/kg)", xlim=(0, fixed_h * 1.03))
    ax.set_ylim(0.0, 2.62)
    ax.grid(axis="y", color=LIGHT, lw=0.6)
    ax.legend(loc="upper right")
    ax.set_title("收缩使中心更早跨越干燥阈值", loc="left", fontweight="bold")
    panel_label(ax, "a")

    ax = fig.add_subplot(grid[0, 1])
    bars = ax.bar(
        [0, 1],
        [fixed_h, shrink_h],
        width=0.58,
        color=[FIXED, SHRINK],
        edgecolor="white",
        linewidth=0.7,
    )
    for bar, value in zip(bars, [fixed_h, shrink_h]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 3.0,
            f"{value:.2f} h",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    ax.annotate(
        f"缩短 {reduction:.1f}%",
        xy=(1, shrink_h),
        xytext=(0.52, 102),
        arrowprops={"arrowstyle": "->", "color": ACCENT, "lw": 1.0},
        color=ACCENT,
        ha="center",
        fontweight="bold",
    )
    ax.set_xticks([0, 1], ["恒定半径", "收缩半径"])
    ax.set_ylabel("烘干时间 / h")
    ax.set_ylim(0, 145)
    ax.grid(axis="y", color=LIGHT, lw=0.6)
    ax.set_title("半径收缩显著降低总时长", loc="left", fontweight="bold")
    panel_label(ax, "b")

    ax = fig.add_subplot(grid[1, 0])
    gap = common["fixed_r_0.0_cm"] - common["shrink_r_0.0_cm"]
    ax.plot(common["time_h"], gap, color=ACCENT, lw=1.7, label="中心水分差")
    ax.axhline(0.0, color=GRAY, lw=0.7)
    ax.fill_between(common["time_h"], 0.0, gap, where=gap >= 0, color=ACCENT, alpha=0.14)
    ax.set_xlabel("时间 / h")
    ax.set_ylabel("C恒定 - C收缩 / (kg/kg)", color=ACCENT)
    ax.tick_params(axis="y", colors=ACCENT)
    ax.grid(axis="y", color=LIGHT, lw=0.6)
    ax2 = ax.twinx()
    ax2.plot(common["time_h"], common["shrink_radius_cm"] / 2.0, color=SHRINK, lw=1.25, ls="--")
    ax2.set_ylabel("半径比 R/R0", color=SHRINK)
    ax2.tick_params(axis="y", colors=SHRINK)
    ax2.set_ylim(0.55, 1.02)
    ax2.spines["top"].set_visible(False)
    ax.set_title("半径缩小累积放大内部脱水差异", loc="left", fontweight="bold")
    panel_label(ax, "c")

    ax = fig.add_subplot(grid[1, 1])
    for time_h, alpha in [(24.0, 0.55), (48.0, 1.0)]:
        shrink_radius = interpolate_row_at_time(frame, time_h * 3600.0, "shrink_radius_cm")
        xi, fixed_profile = normalized_profile(frame, "fixed", time_h, 2.0)
        _, shrink_profile = normalized_profile(frame, "shrink", time_h, shrink_radius)
        ax.plot(
            xi,
            fixed_profile,
            color=FIXED,
            lw=1.5,
            alpha=alpha,
            ls="-",
            label=f"恒定 {time_h:.0f} h",
        )
        ax.plot(
            xi,
            shrink_profile,
            color=SHRINK,
            lw=1.5,
            alpha=alpha,
            ls="--",
            label=f"收缩 {time_h:.0f} h",
        )
    ax.axhline(0.15, color=GRAY, lw=0.8, ls=":")
    ax.set(xlabel="归一化半径 xi=r/R(t)", ylabel="水分浓度 / (kg/kg)", xlim=(0, 1))
    ax.grid(axis="y", color=LIGHT, lw=0.6)
    ax.legend(loc="upper right", ncol=2, columnspacing=0.8, handlelength=1.8)
    ax.set_title("同一材料位置的径向分布整体下移", loc="left", fontweight="bold")
    panel_label(ax, "d")

    fig.suptitle(
        "问题四参数下半径收缩对烘干结果的影响",
        fontsize=10.5,
        fontweight="bold",
        x=0.01,
        ha="left",
    )
    output = figure_dir / "A_q4_radius_comparison.png"
    fig.savefig(output, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"图片：{output}")


if __name__ == "__main__":
    main()
