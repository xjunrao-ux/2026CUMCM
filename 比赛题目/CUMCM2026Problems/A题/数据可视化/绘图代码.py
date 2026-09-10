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
    temp_tail = float(env.loc[env["time_s"] >= 12600, "temperature_c"].mean())
    moisture_tail = float(env.loc[env["time_s"] >= 12600, "moisture_kgkg"].mean())
    temp_95_h = 4740 / 3600
    moisture_95_h = 6180 / 3600

    env_time_h = env["time_h"].to_numpy()
    assert np.all(np.diff(env_time_h) > 0), "插值时间轴必须严格递增。"
    fig, axes = plt.subplots(1, 2, figsize=(7.0079, 3.0709), constrained_layout=False)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.22, top=0.84, wspace=0.30)

    ax = axes[0]
    ax.axvspan(0, 0.5, color=COLORS["preheat"], zorder=0)
    ax.axhspan(temp_tail - 1.0, temp_tail + 1.0, color=COLORS["temperature"], alpha=0.08, zorder=0)
    ax.plot(env["time_h"], env["temperature_c"], color=COLORS["temperature"], lw=1.55, zorder=3)
    ax.scatter(env["time_h"], env["temperature_c"], color=COLORS["temperature"], s=4.2, alpha=0.45, linewidths=0, zorder=2)
    ax.axhline(temp_tail, color=COLORS["neutral"], lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.axvline(temp_95_h, color=COLORS["accent"], lw=0.9, ls=(0, (2, 2)), zorder=1)
    ax.text(0.08, 0.95, "预热区间", transform=ax.transAxes, fontsize=6.5, color="#7A5D2B", va="top")
    temp_at_95 = float(np.interp(temp_95_h, env_time_h, env["temperature_c"]))
    ax.annotate("达到最终升温幅度95%\n1.32 h", xy=(temp_95_h, temp_at_95), xytext=(1.70, 39.0), fontsize=6.5, color="#7A4E0D", arrowprops={"arrowstyle": "-", "lw": 0.7, "color": COLORS["accent"]})
    ax.text(3.95, temp_tail + 0.25, f"末30 min均值 {temp_tail:.2f} °C", ha="right", va="bottom", fontsize=6.5, color=COLORS["neutral"])
    ax.set(xlim=(0, 4), ylim=(26, 52), xlabel="时间 (h)", ylabel="烘房温度 (°C)", title="温度响应")
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    finish_axis(ax)
    panel_label(ax, "a")

    ax = axes[1]
    ax.axvspan(0, 0.5, color=COLORS["preheat"], zorder=0)
    ax.axhspan(moisture_tail - 0.002, moisture_tail + 0.002, color=COLORS["moisture"], alpha=0.08, zorder=0)
    ax.plot(env["time_h"], env["moisture_kgkg"], color=COLORS["moisture"], lw=1.55, zorder=3)
    ax.scatter(env["time_h"], env["moisture_kgkg"], color=COLORS["moisture"], s=4.2, alpha=0.45, linewidths=0, zorder=2)
    ax.axhline(moisture_tail, color=COLORS["neutral"], lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.axvline(moisture_95_h, color=COLORS["accent"], lw=0.9, ls=(0, (2, 2)), zorder=1)
    ax.text(0.08, 0.95, "预热区间", transform=ax.transAxes, fontsize=6.5, color="#7A5D2B", va="top")
    moisture_at_95 = float(np.interp(moisture_95_h, env_time_h, env["moisture_kgkg"]))
    ax.annotate("达到最终增幅95%\n1.72 h", xy=(moisture_95_h, moisture_at_95), xytext=(2.00, 0.036), fontsize=6.5, color="#7A4E0D", arrowprops={"arrowstyle": "-", "lw": 0.7, "color": COLORS["accent"]})
    ax.text(3.95, moisture_tail + 0.0007, f"末30 min均值 {moisture_tail:.5f}", ha="right", va="bottom", fontsize=6.5, color=COLORS["neutral"])
    ax.set(xlim=(0, 4), ylim=(0.017, 0.053), xlabel="时间 (h)", ylabel="烘房水分浓度 (kg/kg)", title="水分浓度响应")
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(0.01))
    finish_axis(ax)
    panel_label(ax, "b")

    fig.suptitle("烘房环境在约1.7 h内逐步趋于稳定", x=0.09, y=0.965, ha="left", fontsize=10.5, fontweight="bold", color="#101828")
    fig.text(0.09, 0.075, "全部241个观测点，采样间隔60 s；实线连接原始清洗数据，未进行平滑。浅色横带为稳定带，米色竖带为预热区间。", fontsize=6.5, color="#667085")
    export_figure(fig, output_dir, "图1_烘房环境时间演化")
    plt.close(fig)


def make_radius_figure(radius: pd.DataFrame, output_dir: Path) -> None:
    time_h = radius["time_h"].to_numpy()
    radius_cm = radius["radius_cm"].to_numpy()
    assert np.all(np.diff(time_h) > 0), "插值时间轴必须严格递增。"
    milestones = [(2.5, "50%"), (10.0, "90%"), (22.0, "99%")]

    fig, ax = plt.subplots(figsize=(5.9055, 3.3858), constrained_layout=False)
    fig.subplots_adjust(left=0.12, right=0.975, bottom=0.22, top=0.83)

    ax.axvspan(35, 72, color=COLORS["plateau"], zorder=0)
    ax.fill_between(time_h, radius_cm, radius_cm[-1], color=COLORS["radius"], alpha=0.10, zorder=1)
    ax.plot(time_h, radius_cm, color=COLORS["radius"], lw=1.7, zorder=3)
    ax.scatter(time_h, radius_cm, color=COLORS["radius"], s=5.0, alpha=0.55, linewidths=0, zorder=2)

    offsets = {2.5: (6.2, 1.78), 10.0: (14.0, 1.48), 22.0: (27.0, 1.31)}
    for milestone_h, label in milestones:
        value = float(np.interp(milestone_h, time_h, radius_cm))
        text_x, text_y = offsets[milestone_h]
        ax.scatter([milestone_h], [value], s=22, facecolor="white", edgecolor=COLORS["accent"], linewidth=1.0, zorder=5)
        ax.annotate(f"完成总收缩{label}\n{milestone_h:g} h, {value:.3f} cm", xy=(milestone_h, value), xytext=(text_x, text_y), fontsize=6.5, color="#694A13", arrowprops={"arrowstyle": "-", "lw": 0.7, "color": COLORS["accent"]})

    ax.axvline(35, color=COLORS["neutral"], lw=0.8, ls=(0, (3, 3)), zorder=1)
    ax.text(53.5, 1.93, "近稳定区间\nr ≈ 1.20 cm", ha="center", va="top", fontsize=7, color=COLORS["radius"])
    ax.text(71.5, radius_cm[-1] - 0.025, f"72 h: {radius_cm[-1]:.3f} cm", ha="right", va="top", fontsize=6.5, color=COLORS["radius"])
    ax.set(xlim=(0, 72), ylim=(1.14, 2.04), xlabel="烘干时间 (h)", ylabel="药材半径 (cm)")
    ax.xaxis.set_major_locator(MultipleLocator(12))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    finish_axis(ax)
    fig.suptitle("药材半径收缩集中于烘干早期并逐渐进入平台", x=0.12, y=0.955, ha="left", fontsize=10.5, fontweight="bold", color="#101828")
    fig.text(0.12, 0.075, "全部145个观测点，采样间隔0.5 h；实线连接原始清洗数据，未进行平滑。收缩完成度以2.000–1.198 cm为总变化范围。", fontsize=6.5, color="#667085")
    export_figure(fig, output_dir, "图2_药材半径收缩")
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
