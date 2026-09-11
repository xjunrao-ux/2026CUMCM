from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator


COLORS = {
    "temperature": "#C6534C",
    "moisture": "#2D6A9F",
    "radius": "#3F7463",
    "accent": "#B7791F",
    "neutral": "#667085",
    "grid": "#D9DEE5",
    "preheat": "#F4EBDD",
    "plateau": "#E8F0EC",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "Arial", "SimHei", "DejaVu Sans"],
        "font.size": 7.5,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制A题清洗数据的科研级时间序列图。")
    parser.add_argument("--project-dir", type=Path, required=True, help="A题目录。")
    parser.add_argument("--output-dir", type=Path, required=True, help="图件输出目录。")
    return parser.parse_args()


def load_clean_data(project_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean_dir = project_dir / "数据清洗"
    clean_env = clean_dir / "附件1_清洗后.xlsx"
    clean_radius = clean_dir / "附件2_清洗后.xlsx"
    if clean_env.exists() and clean_radius.exists():
        env = pd.read_excel(clean_env, sheet_name="清洗数据")
        radius = pd.read_excel(clean_radius, sheet_name="清洗数据")
    else:
        source_dir = project_dir / "附件"
        env = pd.read_excel(source_dir / "附件1.xlsx", sheet_name="Sheet1")
        radius = pd.read_excel(source_dir / "附件2.xlsx", sheet_name="Sheet1")
    env.columns = ["time_s", "temperature_c", "moisture_kgkg"]
    radius.columns = ["time_s", "radius_cm"]

    for frame in (env, radius):
        if frame.isna().any().any():
            raise ValueError("清洗数据包含缺失值，停止绘图。")
        if not frame["time_s"].is_monotonic_increasing:
            raise ValueError("时间列并非单调递增，停止绘图。")

    if len(env) != 241 or len(radius) != 145:
        raise ValueError(f"记录数异常：附件1={len(env)}，附件2={len(radius)}。")

    env["time_h"] = env["time_s"] / 3600.0
    env["temperature_k"] = env["temperature_c"] + 273.15
    radius["time_h"] = radius["time_s"] / 3600.0
    return env, radius


def finish_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.55, alpha=0.7)
    ax.tick_params(direction="out", colors="#344054")
    ax.spines["left"].set_color("#667085")
    ax.spines["bottom"].set_color("#667085")


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.07, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top", ha="left", color="#101828")


def export_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.tiff", dpi=600, bbox_inches="tight")


def make_environment_figure(env: pd.DataFrame, output_dir: Path) -> None:
    tail_start_s = float(env["time_s"].max() - 3600)
    temp_tail = float(env.loc[env["time_s"] >= tail_start_s, "temperature_k"].mean())
    moisture_tail = float(env.loc[env["time_s"] >= tail_start_s, "moisture_kgkg"].mean())
    preheat_end_h = 1800 / 3600

    env_time_h = env["time_h"].to_numpy()
    assert np.all(np.diff(env_time_h) > 0), "插值时间轴必须严格递增。"
    fig, axes = plt.subplots(2, 1, figsize=(7.0079, 4.7244), sharex=True, constrained_layout=False)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.12, top=0.88, hspace=0.34)

    ax = axes[0]
    ax.axvspan(0, preheat_end_h, color=COLORS["preheat"], zorder=0)
    ax.plot(env["time_h"], env["temperature_k"], color=COLORS["temperature"], lw=1.55, zorder=3)
    ax.scatter(env["time_h"], env["temperature_k"], color=COLORS["temperature"], s=4.2, alpha=0.45, linewidths=0, zorder=2)
    ax.axhline(temp_tail, color=COLORS["neutral"], lw=0.7, ls=(0, (3, 3)), alpha=0.65, zorder=1)
    ax.text(0.25, 0.96, "预热平衡阶段", transform=ax.get_xaxis_transform(), fontsize=6.5, color="#7A5D2B", ha="center", va="top")
    ax.text(2.25, 0.96, "恒温干燥阶段", transform=ax.get_xaxis_transform(), fontsize=6.5, color=COLORS["neutral"], ha="center", va="top")
    ax.text(3.95, temp_tail + 0.25, f"末1 h均值 {temp_tail:.2f} K", ha="right", va="bottom", fontsize=6.5, color=COLORS["neutral"])
    ax.set(xlim=(0, 4), ylim=(299, 326), ylabel="烘房温度 (K)", title="温度响应")
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    finish_axis(ax)
    panel_label(ax, "a")

    ax = axes[1]
    ax.axvspan(0, preheat_end_h, color=COLORS["preheat"], zorder=0)
    ax.plot(env["time_h"], env["moisture_kgkg"], color=COLORS["moisture"], lw=1.55, zorder=3)
    ax.scatter(env["time_h"], env["moisture_kgkg"], color=COLORS["moisture"], s=4.2, alpha=0.45, linewidths=0, zorder=2)
    ax.axhline(moisture_tail, color=COLORS["neutral"], lw=0.7, ls=(0, (3, 3)), alpha=0.65, zorder=1)
    ax.text(3.95, moisture_tail + 0.0007, f"末1 h均值 {moisture_tail:.5f}", ha="right", va="bottom", fontsize=6.5, color=COLORS["neutral"])
    ax.set(xlim=(0, 4), ylim=(0.017, 0.053), xlabel="时间 (h)", ylabel="烘房水分浓度 (kg/kg)", title="水分浓度响应")
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(0.01))
    finish_axis(ax)
    panel_label(ax, "b")

    fig.suptitle("烘房环境随时间变化", x=0.09, y=0.965, ha="left", fontsize=10.5, fontweight="bold", color="#101828")
    export_figure(fig, output_dir, "图1_烘房环境随时间变化")
    plt.close(fig)


def make_radius_figure(radius: pd.DataFrame, output_dir: Path) -> None:
    time_h = radius["time_h"].to_numpy()
    radius_cm = radius["radius_cm"].to_numpy()
    assert np.all(np.diff(time_h) > 0), "插值时间轴必须严格递增。"
    milestones = [(2.5, "50%"), (10.0, "90%"), (22.0, "99%")]

    fig, ax = plt.subplots(figsize=(5.9055, 3.3858), constrained_layout=False)
    fig.subplots_adjust(left=0.12, right=0.975, bottom=0.16, top=0.83)

    ax.fill_between(time_h, radius_cm, radius_cm[-1], color=COLORS["radius"], alpha=0.10, zorder=1)
    ax.plot(time_h, radius_cm, color=COLORS["radius"], lw=1.7, zorder=3)
    ax.scatter(time_h, radius_cm, color=COLORS["radius"], s=5.0, alpha=0.55, linewidths=0, zorder=2)

    offsets = {2.5: (6.2, 1.78), 10.0: (14.0, 1.48), 22.0: (27.0, 1.31)}
    for milestone_h, label in milestones:
        value = float(np.interp(milestone_h, time_h, radius_cm))
        text_x, text_y = offsets[milestone_h]
        ax.scatter([milestone_h], [value], s=22, facecolor="white", edgecolor=COLORS["accent"], linewidth=1.0, zorder=5)
        ax.annotate(f"完成总收缩{label}\n{milestone_h:g} h, {value:.3f} cm", xy=(milestone_h, value), xytext=(text_x, text_y), fontsize=6.5, color="#694A13", arrowprops={"arrowstyle": "-", "lw": 0.7, "color": COLORS["accent"]})

    ax.text(71.5, radius_cm[-1] + 0.025, f"接近稳定值 {radius_cm[-1]:.3f} cm", ha="right", va="bottom", fontsize=6.5, color=COLORS["radius"])
    ax.set(xlim=(0, 72), ylim=(1.14, 2.04), xlabel="烘干时间 (h)", ylabel="药材半径 (cm)")
    ax.xaxis.set_major_locator(MultipleLocator(12))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    finish_axis(ax)
    fig.suptitle("药材半径随时间变化", x=0.12, y=0.955, ha="left", fontsize=10.5, fontweight="bold", color="#101828")
    export_figure(fig, output_dir, "图2_药材半径随时间变化")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    project_dir = args.project_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    env, radius = load_clean_data(project_dir)
    make_environment_figure(env, output_dir)
    make_radius_figure(radius, output_dir)
    shutil.copy2(Path(__file__), output_dir / "绘图代码.py")


if __name__ == "__main__":
    main()
