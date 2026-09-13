# -*- coding: utf-8 -*-
"""第三问二维轴对称模型的结果分析图（只读正式结果，不重新求解）。

数据来源：
- results/A_problem3_dimension2/moisture_2d_60s_r0p1cm_z0p5cm.csv  二维水分场（60 s 采样）
- result/A_problem3_2d_exposed/comparison_1d_2d_midplane.csv        同网格同时步的一维/二维中截面对照

输出（600 dpi PNG，写到本脚本所在目录）：
- a3d2_field_snapshots.png     二维水分场快照（6/24/42 h）+ 轴线轴向剖面
- a3d2_1d_vs_2d_midplane.png   中截面径向水分：一维 vs 二维，及二者之差

配色沿用仓库已校验的调色板（dataviz 参考调色板：类目槽位 1-5、蓝色顺序色带）。
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

plt.rcParams.update(
    {
        "font.sans-serif": ["Microsoft YaHei", "SimHei"],
        "axes.unicode_minus": False,
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "text.color": "#0b0b0b",
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": "#0b0b0b",
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
    }
)

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
BLUE_RAMP = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
    "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
    "#184f95", "#104281", "#0d366b",
]
MUTED = "#52514e"
GRIDLINE = "#e1e0d9"
REF = "#767676"

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
NEW_2D_CSV = REPO_ROOT / "results" / "A_problem3_dimension2" / "moisture_2d_60s_r0p1cm_z0p5cm.csv"
COMPARE_CSV = REPO_ROOT / "result" / "A_problem3_2d_exposed" / "comparison_1d_2d_midplane.csv"

BLUE_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list("repo_blue", BLUE_RAMP)


def load_2d_field_csv(path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """读取二维水分长表，返回按 z 升序的宽表与 r/cm、z/m 坐标。"""
    frame = pd.read_csv(path, encoding="utf-8-sig")
    radius_labels = [col for col in frame.columns if col.startswith("r_")]
    radii_cm = np.array(
        [float(col.split("_")[1].replace("cm", "")) for col in radius_labels]
    )
    z_m = np.sort(frame["z_m"].unique())
    wide = frame.pivot_table(
        index="time_s", columns="z_m", values=radius_labels, sort=True
    )
    # 每行一个时刻：n_z 个 z 层 × n_r 个径向点
    return wide, radii_cm, z_m


def field_at_time(wide: pd.DataFrame, z_m: np.ndarray, time_s: int) -> np.ndarray:
    values = wide.loc[float(time_s)].to_numpy(dtype=float)
    return values.reshape(len(z_m), -1)


def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, color=GRIDLINE, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(direction="out", length=3)


def figure_field_snapshots(wide: pd.DataFrame, radii_cm: np.ndarray, z_m: np.ndarray) -> None:
    """图一：二维水分场快照 + 轴线轴向剖面。"""
    snapshot_times = [21600, 86400, 151200]  # 6 / 24 / 42 h
    profile_times = [6.0, 12.0, 18.0, 24.0, 42.0]
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.4))
    r_cm = radii_cm
    z_cm = z_m * 100.0

    for ax, time_s, letter in zip(axes.flat[:3], snapshot_times, "abc"):
        field = field_at_time(wide, z_m, time_s)
        mesh = ax.pcolormesh(r_cm, z_cm, field, cmap=BLUE_CMAP, shading="auto")
        if field.min() < 0.15 < field.max():
            ax.contour(r_cm, z_cm, field, levels=[0.15], colors="white", linewidths=0.8)
        colorbar = fig.colorbar(mesh, ax=ax, fraction=0.045, pad=0.02)
        colorbar.ax.tick_params(labelsize=8)
        colorbar.set_label("水分含量 (kg/kg)", fontsize=9)
        ax.set_title(f"({letter}) {time_s / 3600.0:g} h 水分场", fontsize=11)
        ax.set_aspect("auto")
        style_axes(ax)

    axes[1, 0].set_xlabel("径向位置 r/cm")
    axes[1, 1].set_xlabel("轴向位置 z/cm")
    axes[0, 0].set_ylabel("轴向位置 z/cm")
    axes[1, 0].set_ylabel("轴向位置 z/cm")
    axes[0, 1].set_ylabel("轴向位置 z/cm")

    ax = axes[1, 1]
    for color, time_h in zip(CAT, profile_times):
        field = field_at_time(wide, z_m, int(round(time_h * 3600.0)))
        ax.plot(z_cm, field[:, 0], color=color, linewidth=2.0, label=f"{time_h:g} h")
    ax.axvline(12.5, color=REF, linewidth=0.8, linestyle="--")
    ax.text(12.5, 1.02, "端面 z=L/2", color=REF, fontsize=8.5, ha="right")
    ax.set_title("(d) 轴线 (r=0) 轴向水分剖面", fontsize=11)
    ax.set_ylabel("水分含量 (kg/kg)")
    ax.set_ylim(0.05, 1.10)
    ax.set_xlim(0.0, 14.6)
    style_axes(ax)
    label_x = {6.0: 12.55, 12.0: 12.55, 18.0: 13.05, 24.0: 13.55, 42.0: 14.05}
    for color, time_h in zip(CAT, profile_times):
        field = field_at_time(wide, z_m, int(round(time_h * 3600.0)))
        ax.text(label_x[time_h], field[-1, 0], f"{time_h:g} h",
                color=MUTED, fontsize=8.5, va="center")

    fig.suptitle(
        "第三问二维模型水分场：端面效应仅在端面附近形成局部干燥楔形区",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(SCRIPT_DIR / "a3d2_field_snapshots.png", dpi=600)
    plt.close(fig)


def figure_1d_vs_2d_midplane() -> None:
    """图二：中截面（z=0）径向水分的一维/二维对照与差值。"""
    frame = pd.read_csv(COMPARE_CSV, encoding="utf-8-sig")
    compare_times = [6.0, 24.0, 42.0, 54.0]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4))

    ax = axes[0]
    for color, time_h in zip(CAT, compare_times):
        rows = frame[frame["time_h"] == time_h]
        ax.plot(
            rows["radius_cm"], rows["moisture_1d_kg_kg"],
            color=color, linewidth=2.0, marker="o", markersize=4,
        )
        ax.plot(
            rows["radius_cm"], rows["moisture_2d_midplane_kg_kg"],
            color=color, linewidth=2.0, linestyle="--", marker="s", markersize=3.5,
        )
    ax.axhline(0.15, color=REF, linewidth=0.8, linestyle=":")
    ax.text(0.05, 0.158, "烘干阈值 0.15 kg/kg", color=REF, fontsize=8.5)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], color=MUTED, linewidth=2.0, label="一维模型（实线）"),
        Line2D([], [], color=MUTED, linewidth=2.0, linestyle="--", label="二维模型（虚线）"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=9)
    label_dy = {6.0: 0.0, 24.0: 0.0, 42.0: 0.016, 54.0: -0.016}
    for color, time_h in zip(CAT, compare_times):
        rows = frame[frame["time_h"] == time_h]
        ax.text(2.02, float(rows["moisture_1d_kg_kg"].iloc[-1]) + label_dy[time_h],
                f"{time_h:g} h", color=MUTED, fontsize=8.5, va="center")
    ax.set_xlabel("径向位置 r/cm")
    ax.set_ylabel("中截面水分含量 (kg/kg)")
    ax.set_title("(a) 中截面径向水分分布（同网格同时步对照）", fontsize=11)
    ax.set_xlim(-0.15, 3.2)
    ax.set_ylim(0.02, 1.12)
    style_axes(ax)

    ax = axes[1]
    label_dy = {6.0: 0.0, 24.0: 0.0, 42.0: 0.10, 54.0: -0.10}
    for color, time_h in zip(CAT, compare_times):
        rows = frame[frame["time_h"] == time_h]
        diff = (rows["moisture_2d_midplane_kg_kg"] - rows["moisture_1d_kg_kg"]) * 1.0e5
        ax.plot(rows["radius_cm"], diff, color=color, linewidth=2.0,
                marker="o", markersize=4)
        ax.text(2.02, float(diff.iloc[-1]) + label_dy[time_h], f"{time_h:g} h",
                color=MUTED, fontsize=8.5, va="center")
    ax.axhline(0.0, color=REF, linewidth=0.8, linestyle="--")
    ax.set_xlabel("径向位置 r/cm")
    ax.set_ylabel(r"$\Delta C$ (×10$^{-5}$ kg/kg)")
    ax.set_title("(b) 二维 − 一维（端面效应的净影响）", fontsize=11)
    ax.set_xlim(-0.15, 3.2)
    style_axes(ax)

    fig.tight_layout()
    fig.savefig(SCRIPT_DIR / "a3d2_1d_vs_2d_midplane.png", dpi=600)
    plt.close(fig)


def print_summary_numbers() -> None:
    """把分析引用到的关键数字打印出来，便于核对论文表述。"""
    new = json.loads(
        (REPO_ROOT / "results" / "A_problem3_dimension2" / "validation_summary.json")
        .read_text(encoding="utf-8")
    )
    old = json.loads(
        (REPO_ROOT / "result" / "A_problem3_2d_exposed" / "summary.json")
        .read_text(encoding="utf-8")
    )
    frame = pd.read_csv(COMPARE_CSV, encoding="utf-8-sig")
    diff = frame["difference_2d_minus_1d"].abs()
    print("=== 烘干时间对照（s）===")
    print("1D 精细 (1/30/1 s)        :", new["one_dimensional_reference"]["original_production_drying_time_s"])
    print("1D 匹配步长 (30/60/1 s)   :", new["one_dimensional_reference"]["matched_time_step_drying_time_s"])
    print("2D 新版 (64x40, 30/60/1 s):", new["drying_time"]["production_s"])
    print("2D 新版粗网格 (32x20)     :", new["drying_time"]["coarse_s"])
    print("2D 旧版 (80x40, 30/120/1):", old["production"]["drying_time_s"])
    print("2D 旧版匹配 1D            :", old["comparison_with_1d"]["matched_1d_drying_time_s"])
    print("2D 旧版粗网格 (40x20)     :", old["coarse_validation"]["drying_time_s"])
    print("=== 中截面 1D vs 2D 最大差 ===")
    print(f"max |2D-1D| = {diff.max():.3e} kg/kg @ "
          f"{frame.loc[diff.idxmax(), 'time_h']:g} h, r={frame.loc[diff.idxmax(), 'radius_cm']:g} cm")
    print("=== 端面贡献（旧版）===")
    print("侧面出口份额:", f"{old['half_domain_moisture_balance']['side_outflow_fraction']:.4f}",
          "端面出口份额:", f"{old['half_domain_moisture_balance']['end_outflow_fraction']:.4f}")
    print("=== 守恒与验证 ===")
    print("新版 水分相对残差:", new["whole_product_balances"]["moisture_relative_residual"],
          "热量相对残差:", new["whole_product_balances"]["heat_relative_residual"])
    print("绝缘端面退化检验: T 误差", old["checks"]["insulated_end_reduction"]["temperature_max_abs_error_c"],
          "C 误差", old["checks"]["insulated_end_reduction"]["moisture_max_abs_error_kg_kg"])
    print("Picard 最大迭代: 新版", new["checks"]["maximum_picard_iterations"],
          "旧版", old["production"]["maximum_picard_iterations"])


def main() -> None:
    wide, radii_cm, z_m = load_2d_field_csv(NEW_2D_CSV)
    figure_field_snapshots(wide, radii_cm, z_m)
    figure_1d_vs_2d_midplane()
    print_summary_numbers()
    print("Figures ->", SCRIPT_DIR)


if __name__ == "__main__":
    main()
