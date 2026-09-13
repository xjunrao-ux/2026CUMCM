"""A题第三问：蒸发潜热模型与常规模型的定量对比可视化。

本脚本只读取 ``results`` 下已经计算完成的正式结果，不重新求解模型，
也不对曲线做平滑或抽样。输出主对比图、时空差值图以及可复核的指标表。
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


# 可编辑矢量文字与最终尺寸字号设置。
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = [
    "Microsoft YaHei",
    "Arial",
    "SimHei",
    "DejaVu Sans",
    "sans-serif",
]
plt.rcParams.update({'svg.fonttype': 'none', 'pdf.fonttype': 42})
plt.rcParams["axes.unicode_minus"] = False


ROOT = Path(__file__).resolve().parent.parent
REGULAR_DIR = ROOT / "results" / "A_problem3_drying_time"
LATENT_DIR = ROOT / "results" / "A_problem3_drying_time_steam"
FIGURE_DIR = ROOT / "picture" / "A_problem3_latent_heat_comparison"
SUMMARY_DIR = ROOT / "results" / "A_problem3_latent_heat_comparison"

INITIAL_TEMPERATURE_C = 28.0
INITIAL_MOISTURE_KG_KG = 2.55
DRYING_THRESHOLD_KG_KG = 0.15

# 颜色同时用线型区分，保证灰度打印与常见色觉缺陷下仍可识别。
C_REGULAR = "#2A6FBB"
C_LATENT = "#D95F02"
C_CONDUCTION = "#198754"
C_REFERENCE = "#6F6F6F"
C_GRID = "#DDDCD5"
C_AXIS = "#B9B8B0"
C_TEXT = "#252525"
C_MUTED = "#74736E"


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 7.4,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.75,
            "axes.edgecolor": C_AXIS,
            "axes.titlecolor": C_TEXT,
            "axes.labelcolor": C_TEXT,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "xtick.color": C_MUTED,
            "ytick.color": C_MUTED,
            "legend.fontsize": 6.8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "grid.color": C_GRID,
            "grid.linewidth": 0.65,
            "grid.alpha": 0.85,
        }
    )


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def radial_columns(frame: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    columns = list(frame.columns[1:])
    radii = []
    for column in columns:
        match = re.fullmatch(r"r_([0-9.]+)_cm", column)
        if match is None:
            raise ValueError(f"无法识别径向列名：{column}")
        radii.append(float(match.group(1)))
    radii_array = np.asarray(radii, dtype=float)
    if not np.all(np.diff(radii_array) > 0):
        raise ValueError("径向网格必须严格递增。")
    return columns, radii_array


def read_field(data_dir: Path) -> dict[str, object]:
    moisture = pd.read_csv(data_dir / "moisture_full_60s_0p1cm.csv")
    temperature = pd.read_csv(data_dir / "temperature_auxiliary_60s_0p1cm.csv")
    columns, radii_cm = radial_columns(moisture)
    if list(temperature.columns) != list(moisture.columns):
        raise ValueError("温度场与含水率场的列结构不一致。")
    if not np.array_equal(moisture["time_s"], temperature["time_s"]):
        raise ValueError("同一模型的温度场与含水率场时刻不一致。")
    if not np.all(np.diff(moisture["time_s"].to_numpy(dtype=float)) > 0):
        raise ValueError("时间序列必须严格递增。")
    if moisture.isna().any().any() or temperature.isna().any().any():
        raise ValueError("完整场结果中存在缺失值。")

    time_s = np.insert(moisture["time_s"].to_numpy(dtype=float), 0, 0.0)
    moisture_values = np.vstack(
        [np.full((1, len(columns)), INITIAL_MOISTURE_KG_KG), moisture[columns].to_numpy(dtype=float)]
    )
    temperature_values = np.vstack(
        [np.full((1, len(columns)), INITIAL_TEMPERATURE_C), temperature[columns].to_numpy(dtype=float)]
    )
    if not np.isfinite(moisture_values).all() or not np.isfinite(temperature_values).all():
        raise ValueError("完整场结果中存在非有限数值。")
    return {
        "raw_moisture": moisture,
        "raw_temperature": temperature,
        "time_s": time_s,
        "time_h": time_s / 3600.0,
        "radii_cm": radii_cm,
        "moisture": moisture_values,
        "temperature": temperature_values,
    }


def align_common_grid(regular: dict[str, object], latent: dict[str, object]) -> dict[str, np.ndarray]:
    reg_m = regular["raw_moisture"].set_index("time_s")
    lat_m = latent["raw_moisture"].set_index("time_s")
    reg_t = regular["raw_temperature"].set_index("time_s")
    lat_t = latent["raw_temperature"].set_index("time_s")
    common_s = np.intersect1d(reg_m.index.to_numpy(dtype=float), lat_m.index.to_numpy(dtype=float))
    # 两个精确终止时刻不同，差值图仅使用真正共享的60 s记录点；加入共同初值。
    common_s = np.insert(common_s, 0, 0.0)
    zero_m = np.full((1, reg_m.shape[1]), INITIAL_MOISTURE_KG_KG)
    zero_t = np.full((1, reg_t.shape[1]), INITIAL_TEMPERATURE_C)
    reg_m_values = np.vstack([zero_m, reg_m.loc[common_s[1:]].to_numpy(dtype=float)])
    lat_m_values = np.vstack([zero_m, lat_m.loc[common_s[1:]].to_numpy(dtype=float)])
    reg_t_values = np.vstack([zero_t, reg_t.loc[common_s[1:]].to_numpy(dtype=float)])
    lat_t_values = np.vstack([zero_t, lat_t.loc[common_s[1:]].to_numpy(dtype=float)])
    return {
        "time_s": common_s,
        "time_h": common_s / 3600.0,
        "delta_moisture": lat_m_values - reg_m_values,
        "delta_temperature": lat_t_values - reg_t_values,
    }


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=9.0,
        fontweight="bold",
        color=C_TEXT,
        ha="left",
        va="bottom",
    )


def finish_axes(ax: plt.Axes, *, grid_axis: str = "both") -> None:
    ax.grid(True, axis=grid_axis, zorder=0)
    ax.tick_params(length=3.0, width=0.7)


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def compute_metrics(
    regular: dict[str, object],
    latent: dict[str, object],
    aligned: dict[str, np.ndarray],
    regular_validation: dict,
    latent_validation: dict,
    energy: pd.DataFrame,
) -> dict:
    regular_h = float(regular_validation["drying_time"]["production_h"])
    latent_h = float(latent_validation["drying_time"]["production_h"])
    delta_h = latent_h - regular_h

    delta_t = aligned["delta_temperature"]
    delta_c = aligned["delta_moisture"]
    i_t = np.unravel_index(int(np.argmin(delta_t)), delta_t.shape)
    i_c = np.unravel_index(int(np.argmax(np.abs(delta_c))), delta_c.shape)
    radii_cm = np.asarray(regular["radii_cm"], dtype=float)

    latent_surface_t = np.asarray(latent["temperature"], dtype=float)[:, -1]
    i_surface_min = int(np.argmin(latent_surface_t))
    common_time_s = aligned["time_s"]
    six_hour_i = int(np.flatnonzero(np.isclose(common_time_s, 6.0 * 3600.0))[0])

    balance = latent_validation["energy_balance_per_unit_length"]
    convective_total = float(balance["integrated_convective_input_J_m"])
    latent_share = 100.0 * float(balance["integrated_latent_heat_J_m"]) / convective_total
    sensible_share = 100.0 * float(balance["integrated_conductive_heat_into_solid_J_m"]) / convective_total

    conductive = energy["conductive_heat_flux_W_m2"].to_numpy(dtype=float)
    cooling = conductive < 0.0
    cooling_times = energy.loc[cooling, "time_s"].to_numpy(dtype=float) / 3600.0
    max_flux_i = int(energy["latent_heat_flux_W_m2"].to_numpy(dtype=float).argmax())

    return {
        "drying_time": {
            "regular_h": regular_h,
            "latent_heat_h": latent_h,
            "difference_h": delta_h,
            "relative_change_pct": 100.0 * delta_h / regular_h,
            "controlling_radius_cm": float(latent_validation["drying_time"]["controlling_radius_cm"]),
        },
        "temperature_effect": {
            "latent_surface_min_C": float(latent_surface_t[i_surface_min]),
            "latent_surface_min_time_h": float(np.asarray(latent["time_h"])[i_surface_min]),
            "maximum_cooling_difference_C": float(delta_t[i_t]),
            "maximum_cooling_time_h": float(aligned["time_h"][i_t[0]]),
            "maximum_cooling_radius_cm": float(radii_cm[i_t[1]]),
        },
        "moisture_effect": {
            "maximum_absolute_difference_kg_kg": float(delta_c[i_c]),
            "maximum_absolute_difference_time_h": float(aligned["time_h"][i_c[0]]),
            "maximum_absolute_difference_radius_cm": float(radii_cm[i_c[1]]),
            "center_difference_at_6h_kg_kg": float(delta_c[six_hour_i, 0]),
            "delta_range_kg_kg": [float(np.min(delta_c)), float(np.max(delta_c))],
        },
        "energy": {
            "integrated_convective_input_J_m": convective_total,
            "integrated_latent_heat_J_m": float(balance["integrated_latent_heat_J_m"]),
            "integrated_sensible_heat_J_m": float(balance["integrated_conductive_heat_into_solid_J_m"]),
            "latent_share_pct": latent_share,
            "sensible_share_pct": sensible_share,
            "peak_latent_heat_flux_W_m2": float(energy["latent_heat_flux_W_m2"].iloc[max_flux_i]),
            "peak_latent_heat_flux_time_h": float(energy["time_s"].iloc[max_flux_i] / 3600.0),
            "conductive_cooling_interval_h": [
                float(cooling_times.min()),
                float(cooling_times.max()),
            ],
            "relative_surface_energy_residual": float(balance["surface_integrated_relative_residual"]),
        },
        "data_integrity": {
            "regular_saved_rows": int(len(regular["raw_moisture"])),
            "latent_saved_rows": int(len(latent["raw_moisture"])),
            "common_60s_rows_plus_initial": int(len(aligned["time_s"])),
            "radial_positions": int(len(radii_cm)),
            "excluded_observations": 0,
            "difference_grid_rule": "exact inner join on shared time_s values; no interpolation or smoothing",
            "regular_moisture_decimal_places": 4,
            "latent_moisture_decimal_places": 8,
        },
    }


def plot_main_comparison(
    regular: dict[str, object],
    latent: dict[str, object],
    aligned: dict[str, np.ndarray],
    metrics: dict,
    energy: pd.DataFrame,
) -> None:
    fig = plt.figure(figsize=(7.2, 7.35), layout="constrained")
    outer = fig.add_gridspec(3, 2, height_ratios=[1.05, 1.05, 0.72])
    fig.suptitle(
        "第三问：蒸发潜热通过早期蒸发冷却延缓全域干燥",
        fontsize=10.2,
        fontweight="bold",
        color=C_TEXT,
    )

    time_regular = np.asarray(regular["time_h"], dtype=float)
    time_latent = np.asarray(latent["time_h"], dtype=float)
    temp_regular = np.asarray(regular["temperature"], dtype=float)
    temp_latent = np.asarray(latent["temperature"], dtype=float)
    moisture_regular = np.asarray(regular["moisture"], dtype=float)
    moisture_latent = np.asarray(latent["moisture"], dtype=float)

    # a — 机理：早期表面与中心蒸发冷却。
    ax_a = fig.add_subplot(outer[0, 0])
    add_panel_label(ax_a, "a")
    ax_a.set_title("前 6 h 温度响应")
    ax_a.plot(time_regular, temp_regular[:, -1], color=C_REGULAR, lw=1.55, label="常规—表面")
    ax_a.plot(time_latent, temp_latent[:, -1], color=C_LATENT, lw=1.55, ls="--", label="潜热—表面")
    ax_a.plot(time_regular, temp_regular[:, 0], color=C_REGULAR, lw=1.15, ls=":", label="常规—中心")
    ax_a.plot(time_latent, temp_latent[:, 0], color=C_LATENT, lw=1.15, ls="-.", label="潜热—中心")
    oven_6h = energy[energy["time_s"] <= 6.0 * 3600.0]
    ax_a.plot(
        oven_6h["time_s"] / 3600.0,
        oven_6h["oven_temperature_C"],
        color=C_REFERENCE,
        lw=0.95,
        alpha=0.8,
        label="烘房边界",
    )
    ax_a.set_xlim(0, 6)
    ax_a.set_ylim(8, 54)
    ax_a.set_xlabel("时间 (h)")
    ax_a.set_ylabel("温度 (℃)")
    finish_axes(ax_a)
    ax_a.legend(ncol=2, loc="lower right", handlelength=2.2, columnspacing=0.9)
    cooling = metrics["temperature_effect"]
    cooling_time = cooling["maximum_cooling_time_h"]
    cooling_y = np.interp(cooling_time, time_latent, temp_latent[:, -1])
    ax_a.annotate(
        f"最大表面降温 {abs(cooling['maximum_cooling_difference_C']):.2f} ℃",
        xy=(cooling_time, cooling_y),
        xytext=(1.15, 12.0),
        fontsize=7.0,
        color=C_TEXT,
        arrowprops={"arrowstyle": "-", "color": C_MUTED, "lw": 0.7},
    )

    # b — 决定性结果：控制点（中心）水分下降与终止时刻。
    ax_b = fig.add_subplot(outer[0, 1])
    add_panel_label(ax_b, "b")
    ax_b.set_title("控制点含水率与烘干终点")
    ax_b.plot(time_regular, moisture_regular[:, 0], color=C_REGULAR, lw=1.7, label="常规模型")
    ax_b.plot(time_latent, moisture_latent[:, 0], color=C_LATENT, lw=1.7, ls="--", label="蒸发潜热模型")
    common_h = aligned["time_h"]
    common_reg_center = moisture_regular[
        np.searchsorted(np.asarray(regular["time_s"]), aligned["time_s"]), 0
    ]
    common_lat_center = moisture_latent[
        np.searchsorted(np.asarray(latent["time_s"]), aligned["time_s"]), 0
    ]
    ax_b.fill_between(
        common_h,
        common_reg_center,
        common_lat_center,
        color=C_LATENT,
        alpha=0.12,
        linewidth=0,
        label="潜热导致的水分滞留",
    )
    ax_b.axhline(DRYING_THRESHOLD_KG_KG, color=C_REFERENCE, lw=0.9, ls=(0, (4, 3)))
    regular_h = metrics["drying_time"]["regular_h"]
    latent_h = metrics["drying_time"]["latent_heat_h"]
    ax_b.axvline(regular_h, color=C_REGULAR, lw=0.9, ls=":")
    ax_b.axvline(latent_h, color=C_LATENT, lw=0.9, ls=":")
    ax_b.set_xlim(0, 62.5)
    ax_b.set_ylim(0, 2.7)
    ax_b.set_xlabel("时间 (h)")
    ax_b.set_ylabel("中心含水率 (kg/kg)")
    finish_axes(ax_b)
    ax_b.legend(loc="upper right", handlelength=2.2)
    ax_b.text(32, 0.19, "烘干阈值 0.15", fontsize=6.8, color=C_MUTED)
    ax_b.annotate(
        f"6 h 多保留 {metrics['moisture_effect']['center_difference_at_6h_kg_kg']:.3f} kg/kg",
        xy=(6.0, np.interp(6.0, time_latent, moisture_latent[:, 0])),
        xytext=(13.0, 1.55),
        fontsize=7.0,
        color=C_TEXT,
        arrowprops={"arrowstyle": "-", "color": C_MUTED, "lw": 0.7},
    )

    # c — 机制闭合：表面热流分配。
    ax_c = fig.add_subplot(outer[1, :])
    add_panel_label(ax_c, "c")
    ax_c.set_title("潜热模型表面能量平衡（前 6 h）")
    early = energy[energy["time_s"] <= 6.0 * 3600.0]
    early_h = early["time_s"].to_numpy(dtype=float) / 3600.0
    ax_c.plot(early_h, early["convective_heat_flux_W_m2"], color=C_REGULAR, lw=1.55, label="对流输入")
    ax_c.plot(early_h, early["latent_heat_flux_W_m2"], color=C_LATENT, lw=1.55, label="蒸发潜热")
    ax_c.plot(early_h, early["conductive_heat_flux_W_m2"], color=C_CONDUCTION, lw=1.35, label="传导进入药材")
    ax_c.axhline(0.0, color=C_REFERENCE, lw=0.75)
    ax_c.set_yscale("symlog", linthresh=5.0, linscale=0.8)
    ax_c.set_yticks([-1000, -100, -10, 0, 10, 100, 1000])
    ax_c.set_yticklabels(["−1000", "−100", "−10", "0", "10", "100", "1000"])
    ax_c.set_xlim(0, 6)
    ax_c.set_xlabel("时间 (h)")
    ax_c.set_ylabel("表面热流密度 (W/m²)")
    finish_axes(ax_c)
    ax_c.legend(loc="upper right", ncol=3, handlelength=2.0, columnspacing=1.2)
    interval = metrics["energy"]["conductive_cooling_interval_h"]
    ax_c.axvspan(interval[0], interval[1], color=C_LATENT, alpha=0.10, lw=0)
    ax_c.text(
        0.55,
        -550,
        f"传导为负：{interval[0]:.3f}–{interval[1]:.3f} h\n内部显热向表面补偿蒸发",
        fontsize=6.8,
        color=C_TEXT,
    )

    # d/e — 终点效应与累计能量去向，共同构成定量摘要。
    ax_d = fig.add_subplot(outer[2, 0])
    add_panel_label(ax_d, "d")
    ax_d.set_title("全域烘干时间")
    ax_d.barh([1, 0], [regular_h, latent_h], color=[C_REGULAR, C_LATENT], height=0.52, zorder=3)
    ax_d.set_yticks([1, 0], ["常规模型", "蒸发潜热模型"])
    ax_d.set_xlim(0, 66)
    ax_d.set_ylim(-0.58, 1.76)
    ax_d.set_xlabel("时间 (h)")
    finish_axes(ax_d, grid_axis="x")
    ax_d.text(regular_h + 0.8, 1, f"{regular_h:.2f} h", va="center", fontsize=7.4, color=C_TEXT)
    ax_d.text(latent_h + 0.8, 0, f"{latent_h:.2f} h", va="center", fontsize=7.4, color=C_TEXT)
    ax_d.text(
        64.5,
        1.58,
        f"+{metrics['drying_time']['difference_h']:.2f} h（+{metrics['drying_time']['relative_change_pct']:.2f}%）",
        ha="right",
        va="center",
        fontsize=7.4,
        fontweight="bold",
        color=C_TEXT,
    )

    ax_e = fig.add_subplot(outer[2, 1])
    add_panel_label(ax_e, "e")
    ax_e.set_title("潜热模型累计对流热量去向")
    latent_share = metrics["energy"]["latent_share_pct"]
    sensible_share = metrics["energy"]["sensible_share_pct"]
    ax_e.barh([0], [latent_share], color=C_LATENT, height=0.46, zorder=3)
    ax_e.barh([0], [sensible_share], left=[latent_share], color=C_CONDUCTION, height=0.46, zorder=3)
    ax_e.set_xlim(0, 100)
    ax_e.set_ylim(-0.48, 0.70)
    ax_e.set_yticks([])
    ax_e.set_xlabel("占对流输入热量比例 (%)")
    finish_axes(ax_e, grid_axis="x")
    ax_e.text(latent_share / 2.0, 0, f"蒸发潜热 {latent_share:.2f}%", ha="center", va="center", color="white", fontsize=7.2, fontweight="bold")
    ax_e.annotate(
        f"显热 {sensible_share:.2f}%",
        xy=(latent_share + sensible_share / 2.0, 0.0),
        xytext=(82, 0.48),
        ha="center",
        fontsize=6.8,
        color=C_TEXT,
        arrowprops={"arrowstyle": "-", "color": C_MUTED, "lw": 0.7},
    )

    save_figure(fig, FIGURE_DIR / "q3_latent_heat_model_comparison")


def plot_spatiotemporal_difference(
    regular: dict[str, object],
    aligned: dict[str, np.ndarray],
    metrics: dict,
) -> None:
    time_h = aligned["time_h"]
    radii_cm = np.asarray(regular["radii_cm"], dtype=float)
    delta_t = aligned["delta_temperature"].T
    delta_c = aligned["delta_moisture"].T

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), layout="constrained")
    fig.suptitle(
        "蒸发潜热相对常规模型的时空影响（潜热模型 − 常规模型）",
        fontsize=9.8,
        fontweight="bold",
        color=C_TEXT,
    )

    # 温差允许正负，以0为物理中点。
    t_vmin = min(float(np.nanmin(delta_t)), -0.5)
    t_vmax = max(float(np.nanmax(delta_t)), 0.5)
    t_norm = TwoSlopeNorm(vmin=t_vmin, vcenter=0.0, vmax=t_vmax)
    mesh_t = axes[0].pcolormesh(
        time_h,
        radii_cm,
        delta_t,
        shading="auto",
        cmap="RdBu_r",
        norm=t_norm,
        rasterized=True,
    )
    add_panel_label(axes[0], "a")
    axes[0].set_title("温度差 ΔT")
    axes[0].set_xlabel("时间 (h)")
    axes[0].set_ylabel("距中心半径 (cm)")
    axes[0].axvline(4.0, color="white", lw=0.8, ls=(0, (4, 3)), alpha=0.9)
    axes[0].text(4.35, 1.83, "4 h", color="white", fontsize=6.6, va="top")
    cbar_t = fig.colorbar(mesh_t, ax=axes[0], pad=0.025, fraction=0.05)
    cbar_t.set_label("ΔT (℃)")
    cbar_t.ax.tick_params(labelsize=6.6)
    t_effect = metrics["temperature_effect"]
    axes[0].plot(t_effect["maximum_cooling_time_h"], t_effect["maximum_cooling_radius_cm"], marker="o", ms=3.8, mfc="none", mec="white", mew=0.9)

    # 含水率差可能含有由4位小数源文件带来的极小负值，仍原样显示，不截断。
    c_vmin = min(float(np.nanmin(delta_c)), -0.002)
    c_vmax = max(float(np.nanmax(delta_c)), 0.002)
    c_norm = TwoSlopeNorm(vmin=c_vmin, vcenter=0.0, vmax=c_vmax)
    mesh_c = axes[1].pcolormesh(
        time_h,
        radii_cm,
        delta_c,
        shading="auto",
        cmap="PuOr_r",
        norm=c_norm,
        rasterized=True,
    )
    add_panel_label(axes[1], "b")
    axes[1].set_title("含水率差 ΔC")
    axes[1].set_xlabel("时间 (h)")
    axes[1].set_ylabel("距中心半径 (cm)")
    axes[1].axvline(4.0, color=C_TEXT, lw=0.8, ls=(0, (4, 3)), alpha=0.75)
    axes[1].text(4.35, 1.83, "4 h", color=C_TEXT, fontsize=6.6, va="top")
    cbar_c = fig.colorbar(mesh_c, ax=axes[1], pad=0.025, fraction=0.05)
    cbar_c.set_label("ΔC (kg/kg)")
    cbar_c.ax.tick_params(labelsize=6.6)
    c_effect = metrics["moisture_effect"]
    axes[1].plot(c_effect["maximum_absolute_difference_time_h"], c_effect["maximum_absolute_difference_radius_cm"], marker="o", ms=3.8, mfc="none", mec=C_TEXT, mew=0.9)

    for ax in axes:
        ax.set_xlim(0, float(time_h[-1]))
        ax.set_ylim(0, 2.0)
        ax.tick_params(length=3.0, width=0.7)

    save_figure(fig, FIGURE_DIR / "q3_latent_heat_spatiotemporal_difference")


def write_summary(metrics: dict) -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    (SUMMARY_DIR / "comparison_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    rows = [
        ("常规模型烘干时间", metrics["drying_time"]["regular_h"], "h"),
        ("蒸发潜热模型烘干时间", metrics["drying_time"]["latent_heat_h"], "h"),
        ("烘干时间增加", metrics["drying_time"]["difference_h"], "h"),
        ("烘干时间相对增加", metrics["drying_time"]["relative_change_pct"], "%"),
        ("最大温差（潜热-常规）", metrics["temperature_effect"]["maximum_cooling_difference_C"], "℃"),
        ("潜热模型最低表面温度", metrics["temperature_effect"]["latent_surface_min_C"], "℃"),
        ("6 h中心含水率差（潜热-常规）", metrics["moisture_effect"]["center_difference_at_6h_kg_kg"], "kg/kg"),
        ("最大含水率差（潜热-常规）", metrics["moisture_effect"]["maximum_absolute_difference_kg_kg"], "kg/kg"),
        ("累计热量用于蒸发", metrics["energy"]["latent_share_pct"], "%"),
        ("累计热量用于显热", metrics["energy"]["sensible_share_pct"], "%"),
        ("蒸发潜热通量峰值", metrics["energy"]["peak_latent_heat_flux_W_m2"], "W/m²"),
    ]
    pd.DataFrame(rows, columns=["指标", "数值", "单位"]).to_csv(
        SUMMARY_DIR / "comparison_key_metrics.csv", index=False, encoding="utf-8-sig"
    )

    d = metrics["drying_time"]
    t = metrics["temperature_effect"]
    c = metrics["moisture_effect"]
    e = metrics["energy"]
    text = f"""# 第三问蒸发潜热模型可视化分析

## 核心结论

将表面蒸发潜热项纳入能量边界后，模型出现显著的早期蒸发冷却，温度降低进一步抑制温度相关水分扩散，使中心区域在前中期保留更多水分，最终把全域烘干时间从 {d['regular_h']:.3f} h 延长到 {d['latent_heat_h']:.3f} h，增加 {d['difference_h']:.3f} h（{d['relative_change_pct']:.3f}%）。

## 定量证据

- 最大温差为 {t['maximum_cooling_difference_C']:.3f} ℃，出现在 {t['maximum_cooling_time_h']:.3f} h、半径 {t['maximum_cooling_radius_cm']:.1f} cm（药材表面）。
- 潜热模型表面最低温度为 {t['latent_surface_min_C']:.3f} ℃，出现在 {t['latent_surface_min_time_h']:.3f} h。
- 早期表层 ΔC 为负而内部 ΔC 为正，说明潜热模型产生了更强的表里水分分层；中心水分扩散受低温抑制并成为最终控制位置。
- 6 h 时中心含水率比常规模型高 {c['center_difference_at_6h_kg_kg']:.4f} kg/kg；全时空最大正差值为 {c['maximum_absolute_difference_kg_kg']:.4f} kg/kg。
- 潜热模型累计对流输入热量中，{e['latent_share_pct']:.3f}% 用于水分蒸发，仅 {e['sensible_share_pct']:.3f}% 转化为药材显热。
- 表面传导热流在 {e['conductive_cooling_interval_h'][0]:.3f}–{e['conductive_cooling_interval_h'][1]:.3f} h 为负，说明早期蒸发需求一度超过外界对流供热，需要药材内部显热向表面补偿。
- 表面能量积分残差相对值为 {e['relative_surface_energy_residual']:.3e}，能量收支数值闭合良好。

## 可比性与限制

两模型采用相同的一维固定半径几何、环境边界、传质系数和全域 C<0.15 kg/kg 终止判据，差异来自表面蒸发潜热项。差值场使用两模型共同的 60 s 时刻精确合并，不做插值、平滑或抽样；21 个径向位置全部保留。常规模型含水率场仅保存 4 位小数，而潜热模型保存 8 位小数，因此接近零的小差值受输出舍入精度限制，不宜作高精度局部解释。本结果为确定性数值模拟，没有重复样本或统计不确定度。
"""
    (SUMMARY_DIR / "analysis_summary.md").write_text(text, encoding="utf-8")

    qa = """# 图形质量与数据完整性记录

- 核心结论：蒸发潜热引发早期冷却并延长全域烘干时间。
- 结果问题：潜热项通过何种能量路径、在何处何时改变温湿场，并使终止时刻改变多少？
- 图形类型：定量多面板图。
- 后端：Python/matplotlib；所有绘图与导出均使用同一后端。
- 最终尺寸：主图 182.9 mm × 186.7 mm；差值图 182.9 mm × 82.6 mm。
- 输出：SVG（可编辑文字）、PDF、600 dpi PNG 与 LZW 压缩 TIFF。
- 数据：完整使用两套60 s时序及21个径向位置；差值场按共同时间戳精确内连接，无插值、平滑或抽样。
- 不确定度：确定性数值模拟，无重复样本，故不绘制误差条或置信区间。
- 审稿风险：两套含水率CSV保存精度不同（4位与8位小数），近零差值仅作定性判断；最大效应和烘干时长差远高于舍入量级。
"""
    (SUMMARY_DIR / "qa_notes.md").write_text(qa, encoding="utf-8")


def main() -> None:
    apply_style()
    regular = read_field(REGULAR_DIR)
    latent = read_field(LATENT_DIR)
    if not np.array_equal(regular["radii_cm"], latent["radii_cm"]):
        raise ValueError("两模型径向网格不一致，不能直接作差。")
    aligned = align_common_grid(regular, latent)
    regular_validation = read_json(REGULAR_DIR / "validation_summary.json")
    latent_validation = read_json(LATENT_DIR / "validation_summary.json")
    energy = pd.read_csv(LATENT_DIR / "surface_energy_60s.csv")
    if energy.isna().any().any():
        raise ValueError("表面能量数据中存在缺失值。")
    metrics = compute_metrics(
        regular,
        latent,
        aligned,
        regular_validation,
        latent_validation,
        energy,
    )
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plot_main_comparison(regular, latent, aligned, metrics, energy)
    plot_spatiotemporal_difference(regular, aligned, metrics)
    write_summary(metrics)
    print(f"figures: {FIGURE_DIR}")
    print(f"summary: {SUMMARY_DIR}")


if __name__ == "__main__":
    main()
