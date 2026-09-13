"""第四问移动半径水热耦合结果的论文级可视化。

输入
----
result/A_q4_shrinkage/ 下由 ``a4_shrinkage_fvm.py`` 生成的完整场 CSV
与 validation_summary.json。

输出
----
picture/A_q4_shrinkage/ 下 3 张 600 dpi PNG；
result/A_q4_shrinkage/A_q4_visualization_metrics.json。

运行
----
python code/a4_visualization.py

脚本只读取正式数值结果，不重新求解、不平滑原数据。为使移动边界热力图
边缘连续，仅在每个时刻的有效固定距离节点与精确表面值之间做分段线性插值；
当前半径之外始终掩膜为域外区域。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 保留可编辑矢量文字所需配置；本仓库按协作规范只导出 PNG。
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = [
    "Arial",
    "Microsoft YaHei",
    "SimHei",
    "DejaVu Sans",
]
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams.update({'svg.fonttype': 'none', 'pdf.fonttype': 42})

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


INITIAL_RADIUS_CM = 2.0
INITIAL_TEMPERATURE_C = 28.0
INITIAL_MOISTURE_KG_KG = 2.55
DRYING_THRESHOLD_KG_KG = 0.15
MEASURED_ENVIRONMENT_END_H = 4.0

PALETTE = {
    "blue": "#0F4D92",
    "blue_2": "#3775BA",
    "teal": "#42949E",
    "violet": "#7C6CCF",
    "red": "#B64342",
    "orange": "#E28E2C",
    "gray": "#767676",
    "dark": "#272727",
    "outside": "#E7E9ED",
    "grid": "#D8D8D8",
}


def apply_style() -> None:
    """统一图形字体、线宽与背景。"""
    plt.rcParams.update(
        {
            # 中文标签使用实际存在的微软雅黑；Arial 继续作为拉丁字符回退。
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei",
                "Arial",
                "SimHei",
                "DejaVu Sans",
            ],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def parse_radius(column: str) -> float:
    """将 r_0.1_cm 一类字段解析为物理半径（cm）。"""
    if not column.startswith("r_") or not column.endswith("_cm"):
        raise ValueError(f"无法识别径向字段：{column}")
    return float(column.removeprefix("r_").removesuffix("_cm"))


def read_solution(
    data_dir: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict,
]:
    """读取并校验第四问完整时空解，温度统一转换为摄氏度。"""
    moisture_path = data_dir / "moisture_full_60s_0p1cm.csv"
    temperature_path = data_dir / "temperature_auxiliary_60s_0p1cm_K.csv"
    validation_path = data_dir / "validation_summary.json"
    required = [moisture_path, temperature_path, validation_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "缺少第四问完整场结果，请先运行 code/a4_shrinkage_fvm.py：\n"
            + "\n".join(missing)
        )

    moisture_frame = pd.read_csv(moisture_path)
    temperature_frame = pd.read_csv(temperature_path)
    validation = json.loads(validation_path.read_text(encoding="utf-8"))

    expected_prefix = ["time_s", "radius_cm"]
    if list(moisture_frame.columns[:2]) != expected_prefix:
        raise ValueError("水分结果缺少 time_s、radius_cm 字段。")
    if list(temperature_frame.columns[:2]) != expected_prefix:
        raise ValueError("温度结果缺少 time_s、radius_cm 字段。")
    if list(moisture_frame.columns[2:-1]) != list(temperature_frame.columns[2:-1]):
        raise ValueError("温度与水分结果的固定距离网格不一致。")

    time_s = moisture_frame["time_s"].to_numpy(dtype=float)
    radii_cm = moisture_frame["radius_cm"].to_numpy(dtype=float)
    if not np.array_equal(time_s, temperature_frame["time_s"].to_numpy(dtype=float)):
        raise ValueError("温度与水分结果的时间网格不一致。")
    if not np.allclose(
        radii_cm, temperature_frame["radius_cm"].to_numpy(dtype=float)
    ):
        raise ValueError("温度与水分结果的移动半径序列不一致。")

    radial_columns = list(moisture_frame.columns[2:-1])
    fixed_radii_cm = np.asarray([parse_radius(name) for name in radial_columns])
    moisture = moisture_frame[radial_columns].to_numpy(dtype=float)
    temperature_c = (
        temperature_frame[radial_columns].to_numpy(dtype=float) - 273.15
    )
    surface_moisture = moisture_frame.iloc[:, -1].to_numpy(dtype=float)
    surface_temperature_c = temperature_frame.iloc[:, -1].to_numpy(dtype=float) - 273.15

    if time_s.size < 2 or np.any(np.diff(time_s) <= 0.0):
        raise ValueError("时间网格必须严格递增且至少包含两个时刻。")
    if np.any(np.diff(fixed_radii_cm) <= 0.0):
        raise ValueError("固定距离网格必须严格递增。")
    if np.any(np.diff(radii_cm) > 1.0e-10):
        raise ValueError("药材半径结果不是单调不增序列。")
    if not np.all(np.isfinite(surface_moisture)) or not np.all(
        np.isfinite(surface_temperature_c)
    ):
        raise ValueError("移动表面结果中含有 NaN 或 Inf。")

    inside = fixed_radii_cm[None, :] <= radii_cm[:, None] + 1.0e-9
    if np.any(~np.isfinite(moisture[inside])) or np.any(
        ~np.isfinite(temperature_c[inside])
    ):
        raise ValueError("当前半径以内存在缺失的温度或水分值。")
    if np.any(np.isfinite(moisture[~inside])) or np.any(
        np.isfinite(temperature_c[~inside])
    ):
        raise ValueError("当前半径以外存在未掩膜的温度或水分值。")

    # 正式文件从 60 s 起记录；补入题设给出的均匀初值，使图从 t=0 开始。
    if time_s[0] > 0.0:
        time_s = np.insert(time_s, 0, 0.0)
        radii_cm = np.insert(radii_cm, 0, INITIAL_RADIUS_CM)
        moisture = np.vstack(
            [np.full(fixed_radii_cm.size, INITIAL_MOISTURE_KG_KG), moisture]
        )
        temperature_c = np.vstack(
            [np.full(fixed_radii_cm.size, INITIAL_TEMPERATURE_C), temperature_c]
        )
        surface_moisture = np.insert(
            surface_moisture, 0, INITIAL_MOISTURE_KG_KG
        )
        surface_temperature_c = np.insert(
            surface_temperature_c, 0, INITIAL_TEMPERATURE_C
        )

    return (
        time_s,
        radii_cm,
        fixed_radii_cm,
        temperature_c,
        moisture,
        surface_temperature_c,
        surface_moisture,
        validation,
    )


def interpolate_moving_field(
    fixed_radii_cm: np.ndarray,
    current_radii_cm: np.ndarray,
    field: np.ndarray,
    surface_field: np.ndarray,
    dense_radii_cm: np.ndarray,
) -> np.ndarray:
    """在有效域内分段线性插值；移动表面之外保留 NaN。"""
    dense = np.full((field.shape[0], dense_radii_cm.size), np.nan, dtype=float)
    for row, current_radius in enumerate(current_radii_cm):
        valid = np.isfinite(field[row]) & (
            fixed_radii_cm <= current_radius + 1.0e-9
        )
        xp = fixed_radii_cm[valid]
        fp = field[row, valid]
        if xp.size == 0:
            raise ValueError(f"第 {row} 个时刻没有有效的固定距离节点。")

        # 精确表面点可能与最后一个 0.1 cm 固定节点重合，重合时直接替换。
        if math.isclose(xp[-1], current_radius, abs_tol=1.0e-9):
            fp = fp.copy()
            fp[-1] = surface_field[row]
        else:
            xp = np.append(xp, current_radius)
            fp = np.append(fp, surface_field[row])
        if np.any(np.diff(xp) <= 0.0):
            raise ValueError("移动域插值节点不是严格递增序列。")

        query = dense_radii_cm <= current_radius + 1.0e-9
        dense[row, query] = np.interp(dense_radii_cm[query], xp, fp)
    return dense


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.075,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def add_time_guides(
    ax: plt.Axes,
    drying_h: float,
    *,
    heatmap: bool = False,
) -> None:
    """标出 4 h 环境边界切换时刻和最终干燥时刻。"""
    color = "white" if heatmap else PALETTE["gray"]
    ax.axvline(
        MEASURED_ENVIRONMENT_END_H,
        color=color,
        lw=0.9,
        ls="--",
        alpha=0.9,
        zorder=4,
    )
    ax.axvline(drying_h, color=PALETTE["red"], lw=1.0, ls=":", zorder=4)

    # 在常规 10 h 刻度之外显式加入 4 h，避免虚线含义只靠正文猜测。
    x_min, x_max = ax.get_xlim()
    ticks = [
        float(tick)
        for tick in ax.get_xticks()
        if x_min - 1.0e-9 <= tick <= x_max + 1.0e-9
    ]
    ticks.append(MEASURED_ENVIRONMENT_END_H)
    ax.set_xticks(sorted(set(round(tick, 8) for tick in ticks)))


def save_png(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def mark_outside_domain(
    ax: plt.Axes,
    time_h: np.ndarray,
    current_radii_cm: np.ndarray,
    *,
    show_label: bool,
) -> None:
    """以浅灰阴影明确标出因收缩而不存在的空间区域。"""
    ax.fill_between(
        time_h,
        current_radii_cm,
        INITIAL_RADIUS_CM,
        facecolor=PALETTE["outside"],
        edgecolor="#C8CBD0",
        hatch="////",
        linewidth=0.0,
        alpha=0.78,
        zorder=2,
    )
    ax.plot(
        time_h,
        current_radii_cm,
        color=PALETTE["dark"],
        lw=1.35,
        zorder=3,
    )
    if show_label:
        ax.text(
            0.77 * time_h[-1],
            1.72,
            "已收缩域外",
            color=PALETTE["gray"],
            fontsize=7,
            ha="center",
            va="center",
            zorder=5,
        )


def plot_overview(
    time_h: np.ndarray,
    current_radii_cm: np.ndarray,
    dense_radii_cm: np.ndarray,
    temperature_dense_c: np.ndarray,
    moisture_dense: np.ndarray,
    drying_h: float,
    output_dir: Path,
) -> None:
    """主图：温度和水分的移动边界时空场。"""
    fig, (ax_temp, ax_moisture) = plt.subplots(
        2,
        1,
        figsize=(7.2, 5.8),
        sharex=True,
        constrained_layout=True,
    )
    fig.get_layout_engine().set(hspace=0.16, h_pad=0.05)

    temp_masked = np.ma.masked_invalid(temperature_dense_c.T)
    temp_mesh = ax_temp.pcolormesh(
        time_h,
        dense_radii_cm,
        temp_masked,
        shading="auto",
        cmap="magma",
        rasterized=True,
    )
    mark_outside_domain(ax_temp, time_h, current_radii_cm, show_label=True)
    add_time_guides(ax_temp, drying_h, heatmap=True)
    ax_temp.set(
        xlabel="时间 t / h",
        ylabel="到中心距离 r / cm",
        title="温度场：热量由移动表面向中心传递",
        ylim=(0.0, INITIAL_RADIUS_CM),
    )
    ax_temp.tick_params(axis="x", labelbottom=True, pad=2)
    temp_bar = fig.colorbar(temp_mesh, ax=ax_temp, pad=0.012, aspect=26)
    temp_bar.set_label("温度 T / °C")

    positive_moisture = moisture_dense[np.isfinite(moisture_dense)]
    if np.any(positive_moisture <= 0.0):
        raise ValueError("对数色标要求全部有效水分值为正。")
    moisture_mesh = ax_moisture.pcolormesh(
        time_h,
        dense_radii_cm,
        np.ma.masked_invalid(moisture_dense.T),
        shading="auto",
        cmap="YlGnBu",
        norm=mcolors.LogNorm(
            vmin=float(np.min(positive_moisture)),
            vmax=float(np.max(positive_moisture)),
        ),
        rasterized=True,
    )
    threshold_contour = ax_moisture.contour(
        time_h,
        dense_radii_cm,
        moisture_dense.T,
        levels=[DRYING_THRESHOLD_KG_KG],
        colors=[PALETTE["red"]],
        linewidths=1.1,
        zorder=4,
    )
    ax_moisture.clabel(
        threshold_contour,
        fmt={DRYING_THRESHOLD_KG_KG: "C = 0.15"},
        fontsize=6.5,
        inline=True,
    )
    mark_outside_domain(ax_moisture, time_h, current_radii_cm, show_label=False)
    add_time_guides(ax_moisture, drying_h, heatmap=True)
    ax_moisture.set(
        xlabel="时间 t / h",
        ylabel="到中心距离 r / cm",
        title="水分场：低含水率前沿由表面推进至中心（对数色标）",
        xlim=(0.0, time_h[-1]),
        ylim=(0.0, INITIAL_RADIUS_CM),
    )
    moisture_bar = fig.colorbar(moisture_mesh, ax=ax_moisture, pad=0.012, aspect=26)
    moisture_bar.set_label("水分浓度 C / (kg/kg)")

    legend_handles = [
        Line2D([0], [0], color=PALETTE["dark"], lw=1.35, label="当前药材表面 R(t)"),
        Patch(
            facecolor=PALETTE["outside"],
            edgecolor="#C8CBD0",
            hatch="////",
            label="已收缩域外",
        ),
        Line2D(
            [0],
            [0],
            color=PALETTE["red"],
            lw=1.1,
            label="干燥阈值 C = 0.15",
        ),
    ]
    ax_moisture.legend(
        handles=legend_handles,
        loc="upper right",
        ncol=3,
        columnspacing=0.9,
        handlelength=1.7,
    )

    for label, ax in zip("ab", [ax_temp, ax_moisture]):
        add_panel_label(ax, label)
    save_png(fig, output_dir / "A_q4_moving_boundary_heatmaps.png")


def profile_at_index(
    index: int,
    fixed_radii_cm: np.ndarray,
    current_radii_cm: np.ndarray,
    field: np.ndarray,
    surface_field: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(field[index]) & (
        fixed_radii_cm <= current_radii_cm[index] + 1.0e-9
    )
    x = fixed_radii_cm[valid]
    y = field[index, valid]
    if math.isclose(x[-1], current_radii_cm[index], abs_tol=1.0e-9):
        y = y.copy()
        y[-1] = surface_field[index]
    else:
        x = np.append(x, current_radii_cm[index])
        y = np.append(y, surface_field[index])
    return x, y


def plot_radial_profiles(
    time_h: np.ndarray,
    current_radii_cm: np.ndarray,
    fixed_radii_cm: np.ndarray,
    temperature_c: np.ndarray,
    moisture: np.ndarray,
    surface_temperature_c: np.ndarray,
    surface_moisture: np.ndarray,
    drying_h: float,
    output_dir: Path,
) -> None:
    """补充图：选定时刻的物理径向剖面及端点收缩。"""
    targets_h = [0.0, 1.0, 2.0, 4.0, 12.0, 30.0, drying_h]
    indices = [int(np.argmin(np.abs(time_h - target))) for target in targets_h]
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(indices)))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), constrained_layout=True)

    # 全时段移动表面端点轨迹：横坐标为 R(t)，纵坐标为表面温度/水分。
    # 灰色虚线只提示各径向剖面的移动端点，不把任何区域标成最终域外。
    axes[0].plot(
        current_radii_cm,
        surface_temperature_c,
        color=PALETTE["gray"],
        lw=0.9,
        ls="--",
        alpha=0.8,
        label="移动表面轨迹",
        zorder=1,
    )
    axes[1].plot(
        current_radii_cm,
        surface_moisture,
        color=PALETTE["gray"],
        lw=0.9,
        ls="--",
        alpha=0.8,
        label="移动表面轨迹",
        zorder=1,
    )

    for index, target, color in zip(indices, targets_h, colors):
        label = "结束" if math.isclose(target, drying_h) else f"{target:g} h"
        r_temp, temp_profile = profile_at_index(
            index,
            fixed_radii_cm,
            current_radii_cm,
            temperature_c,
            surface_temperature_c,
        )
        r_moisture, moisture_profile = profile_at_index(
            index,
            fixed_radii_cm,
            current_radii_cm,
            moisture,
            surface_moisture,
        )
        axes[0].plot(r_temp, temp_profile, color=color, lw=1.35, label=label)
        axes[1].plot(
            r_moisture,
            moisture_profile,
            color=color,
            lw=1.35,
            label=label,
        )

    axes[1].axhline(
        DRYING_THRESHOLD_KG_KG,
        color=PALETTE["red"],
        lw=1.0,
        ls="--",
    )
    axes[0].set(title="径向温度剖面", ylabel="温度 T / °C")
    axes[1].set(title="径向水分剖面", ylabel="水分浓度 C / (kg/kg)")
    for ax in axes:
        ax.set_xlabel("到中心距离 r / cm")
        ax.set_xlim(0.0, INITIAL_RADIUS_CM)
        ax.grid(color=PALETTE["grid"], lw=0.6, alpha=0.7)
        ax.legend(ncol=2, loc="best", columnspacing=0.75, handlelength=1.6)
    add_panel_label(axes[0], "a")
    add_panel_label(axes[1], "b")
    save_png(fig, output_dir / "A_q4_radial_profiles.png")


def plot_time_series(
    time_h: np.ndarray,
    fixed_radii_cm: np.ndarray,
    temperature_c: np.ndarray,
    moisture: np.ndarray,
    surface_temperature_c: np.ndarray,
    surface_moisture: np.ndarray,
    drying_h: float,
    output_dir: Path,
) -> None:
    """补充图：中心与移动表面随时间的响应。"""
    centre_index = int(np.argmin(np.abs(fixed_radii_cm)))
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.2, 5.2),
        sharex=True,
        constrained_layout=True,
    )
    fig.get_layout_engine().set(hspace=0.16, h_pad=0.05)

    centre_temperature_c = temperature_c[:, centre_index]
    axes[0].fill_between(
        time_h,
        centre_temperature_c,
        surface_temperature_c,
        color=PALETTE["blue_2"],
        alpha=0.22,
        linewidth=0.0,
        label="中心—表面温差",
        zorder=1,
    )
    axes[1].fill_between(
        time_h,
        moisture[:, centre_index],
        surface_moisture,
        color=PALETTE["blue_2"],
        alpha=0.18,
        linewidth=0.0,
        label="中心—表面水分差",
        zorder=1,
    )

    axes[0].plot(
        time_h,
        centre_temperature_c,
        color=PALETTE["blue"],
        lw=1.5,
        label="中心",
        zorder=2,
    )
    axes[0].plot(
        time_h,
        surface_temperature_c,
        color=PALETTE["orange"],
        lw=1.5,
        label="移动表面",
        zorder=2,
    )
    axes[1].plot(
        time_h,
        moisture[:, centre_index],
        color=PALETTE["blue"],
        lw=1.5,
        label="中心",
        zorder=2,
    )
    axes[1].plot(
        time_h,
        surface_moisture,
        color=PALETTE["orange"],
        lw=1.5,
        label="移动表面",
        zorder=2,
    )
    axes[1].axhline(
        DRYING_THRESHOLD_KG_KG,
        color=PALETTE["red"],
        lw=1.0,
        ls="--",
        label="干燥阈值",
    )
    for ax in axes:
        ax.set_xlim(0.0, time_h[-1])
        add_time_guides(ax, drying_h)
        ax.grid(axis="y", color=PALETTE["grid"], lw=0.6, alpha=0.7)
    axes[0].legend(ncol=3, loc="best")
    axes[1].legend(ncol=4, loc="best")
    axes[0].set(
        title="中心与移动表面的温度响应",
        xlabel="时间 t / h",
        ylabel="温度 T / °C",
    )
    axes[0].tick_params(axis="x", labelbottom=True, pad=2)
    axes[1].set(
        title="中心与移动表面的水分响应",
        xlabel="时间 t / h",
        ylabel="水分浓度 C / (kg/kg)",
    )
    add_panel_label(axes[0], "a")
    add_panel_label(axes[1], "b")
    save_png(fig, output_dir / "A_q4_center_surface_time_series.png")


def build_metrics(
    time_h: np.ndarray,
    current_radii_cm: np.ndarray,
    fixed_radii_cm: np.ndarray,
    temperature_c: np.ndarray,
    moisture: np.ndarray,
    surface_temperature_c: np.ndarray,
    surface_moisture: np.ndarray,
    validation: dict,
) -> dict:
    centre_index = int(np.argmin(np.abs(fixed_radii_cm)))
    temperature_difference = np.abs(
        surface_temperature_c - temperature_c[:, centre_index]
    )
    moisture_difference = moisture[:, centre_index] - surface_moisture
    radius_change = np.abs(current_radii_cm - current_radii_cm[-1])
    plateau_candidates = np.flatnonzero(radius_change <= 1.0e-8)
    plateau_h = (
        float(time_h[plateau_candidates[0]]) if plateau_candidates.size else None
    )
    return {
        "source": {
            "time_records_including_initial": int(time_h.size),
            "fixed_distance_nodes": int(fixed_radii_cm.size),
            "fixed_distance_spacing_cm": float(np.median(np.diff(fixed_radii_cm))),
            "visualization_interpolation": (
                "piecewise linear within each current radius; exact moving-surface "
                "value included; outside domain remains masked"
            ),
        },
        "drying_time_h": float(validation["drying_time"]["production_h"]),
        "radius": {
            "initial_cm": float(current_radii_cm[0]),
            "final_cm": float(current_radii_cm[-1]),
            "linear_shrinkage_percent": float(
                100.0 * (1.0 - current_radii_cm[-1] / current_radii_cm[0])
            ),
            "cross_section_area_reduction_percent": float(
                100.0
                * (1.0 - (current_radii_cm[-1] / current_radii_cm[0]) ** 2)
            ),
            "final_radius_first_reached_h": plateau_h,
        },
        "maximum_center_surface_difference": {
            "temperature_C": float(np.max(temperature_difference)),
            "temperature_time_h": float(time_h[int(np.argmax(temperature_difference))]),
            "moisture_kg_kg": float(np.max(moisture_difference)),
            "moisture_time_h": float(time_h[int(np.argmax(moisture_difference))]),
        },
        "final_state": {
            "centre_temperature_C": float(temperature_c[-1, centre_index]),
            "surface_temperature_C": float(surface_temperature_c[-1]),
            "centre_moisture_kg_kg": float(moisture[-1, centre_index]),
            "surface_moisture_kg_kg": float(surface_moisture[-1]),
        },
        "data_integrity": validation["checks"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=repo_root / "result" / "A_q4_shrinkage",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "picture" / "A_q4_shrinkage",
    )
    args = parser.parse_args()

    apply_style()
    (
        time_s,
        current_radii_cm,
        fixed_radii_cm,
        temperature_c,
        moisture,
        surface_temperature_c,
        surface_moisture,
        validation,
    ) = read_solution(args.data_dir)
    time_h = time_s / 3600.0
    drying_h = float(validation["drying_time"]["production_h"])
    if not math.isclose(time_h[-1], drying_h, abs_tol=1.0 / 3600.0):
        raise ValueError("完整场末时刻与验证文件中的干燥时间不一致。")

    dense_radii_cm = np.linspace(0.0, INITIAL_RADIUS_CM, 201)
    temperature_dense_c = interpolate_moving_field(
        fixed_radii_cm,
        current_radii_cm,
        temperature_c,
        surface_temperature_c,
        dense_radii_cm,
    )
    moisture_dense = interpolate_moving_field(
        fixed_radii_cm,
        current_radii_cm,
        moisture,
        surface_moisture,
        dense_radii_cm,
    )

    plot_overview(
        time_h,
        current_radii_cm,
        dense_radii_cm,
        temperature_dense_c,
        moisture_dense,
        drying_h,
        args.output_dir,
    )
    plot_radial_profiles(
        time_h,
        current_radii_cm,
        fixed_radii_cm,
        temperature_c,
        moisture,
        surface_temperature_c,
        surface_moisture,
        drying_h,
        args.output_dir,
    )
    plot_time_series(
        time_h,
        fixed_radii_cm,
        temperature_c,
        moisture,
        surface_temperature_c,
        surface_moisture,
        drying_h,
        args.output_dir,
    )

    metrics = build_metrics(
        time_h,
        current_radii_cm,
        fixed_radii_cm,
        temperature_c,
        moisture,
        surface_temperature_c,
        surface_moisture,
        validation,
    )
    metrics_path = args.data_dir / "A_q4_visualization_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Input records (including t=0): {time_h.size}")
    print(f"Drying time: {drying_h:.6f} h")
    print(
        f"Radius: {current_radii_cm[0]:.3f} -> "
        f"{current_radii_cm[-1]:.3f} cm"
    )
    print(f"Figures: {args.output_dir}")
    for path in sorted(args.output_dir.glob("A_q4_*.png")):
        print(path.name)
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
