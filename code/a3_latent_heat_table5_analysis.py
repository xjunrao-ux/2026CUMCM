"""A题第三问：蒸发潜热模型对表5正式交付结果的影响与滞后时间闭合分析。

本脚本只读取 ``results`` 下两套已计算完成的正式结果（常规模型与蒸发
潜热模型），不重新求解。定量回答两个问题：
  1. 蒸发项使表5各行、各半径改变多少？影响在何时、何处最大？
  2. 为什么累计热量中 97.5% 被蒸发消耗，但烘干时间只延长 5.3%？
     以等含水率水平的滞后时间曲线闭合：早期蒸发冷却造成 ~2 h 的时间
     滞后，此后两模型动力学趋同，终末滞后收敛到烘干时间差本身。

输出：
  * table5_comparison.csv   表5逐行逐半径对比（差值、相对偏差）
  * lag_analysis.csv        等中心含水率水平的滞后时间曲线
  * table5_impact_metrics.json
  * q3_latent_heat_table5_impact.{svg,pdf,png,tiff}
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd


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
SUMMARY_DIR = ROOT / "results" / "A_problem3_latent_heat_comparison"
FIGURE_DIR = ROOT / "picture" / "A_problem3_latent_heat_comparison"

INITIAL_MOISTURE_KG_KG = 2.55
DRYING_THRESHOLD_KG_KG = 0.15
DRY_BULK_DENSITY_KG_M3 = (650.0 + 128.0 * INITIAL_MOISTURE_KG_KG) / (
    1.0 + INITIAL_MOISTURE_KG_KG
)

# 与第四问对比图一致的验证过调色板：#2a78d6 常规、#eb6834 潜热，
# #767676 仅作参考线；滞后曲线为派生量，用墨色。
C_REGULAR = "#2a78d6"
C_LATENT = "#eb6834"
C_REFERENCE = "#767676"
C_LAG = "#252525"
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


def read_table5(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return (row labels, radii cm, values) from a table5_moisture.csv."""
    frame = pd.read_csv(path, encoding="utf-8-sig")
    labels = [str(value) for value in frame.iloc[:, 0]]
    radii_cm = np.asarray(
        [float(column.split()[0]) for column in frame.columns[1:]], dtype=float
    )
    values = frame.iloc[:, 1:].to_numpy(dtype=float)
    return labels, radii_cm, values


def read_center_series(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (time s, center moisture) from the 60 s full CSV."""
    frame = pd.read_csv(
        data_dir / "moisture_full_60s_0p1cm.csv", encoding="utf-8-sig"
    )
    time_s = frame["time_s"].to_numpy(dtype=float)
    center = frame["r_0.0_cm"].to_numpy(dtype=float)
    return time_s, center


def retained_water_difference(
    data_dir_regular: Path, data_dir_latent: Path, query_time_s: float
) -> float:
    """Extra water retained by the latent model per unit length, kg/m.

    Delta M = rho_d * 2*pi * integral (C_lat - C_reg) r dr over the disk.
    """
    regular = pd.read_csv(
        data_dir_regular / "moisture_full_60s_0p1cm.csv", encoding="utf-8-sig"
    )
    latent = pd.read_csv(
        data_dir_latent / "moisture_full_60s_0p1cm.csv", encoding="utf-8-sig"
    )
    radii_cm = np.asarray(
        [float(column.split("_")[1]) for column in regular.columns[1:]],
        dtype=float,
    )
    row_regular = regular.loc[
        np.isclose(regular["time_s"], query_time_s), regular.columns[1:]
    ].iloc[0]
    row_latent = latent.loc[
        np.isclose(latent["time_s"], query_time_s), latent.columns[1:]
    ].iloc[0]
    if row_regular.empty or row_latent.empty:
        raise ValueError(f"两模型在 {query_time_s:g} s 处没有共同记录。")
    radii_m = radii_cm / 100.0
    delta = row_latent.to_numpy(dtype=float) - row_regular.to_numpy(dtype=float)
    return float(
        2.0
        * np.pi
        * DRY_BULK_DENSITY_KG_M3
        * np.trapezoid(delta * radii_m, radii_m)
    )


def lag_at_level(
    time_regular_s: np.ndarray,
    center_regular: np.ndarray,
    time_latent_s: np.ndarray,
    center_latent: np.ndarray,
    level: float,
) -> tuple[float, float]:
    """Return (t_regular, t_latent) at which the center moisture equals level."""
    t_regular = float(
        np.interp(level, center_regular[::-1], time_regular_s[::-1])
    )
    t_latent = float(np.interp(level, center_latent[::-1], time_latent_s[::-1]))
    return t_regular, t_latent


def build_table5_comparison(
    regular: tuple[list[str], np.ndarray, np.ndarray],
    latent: tuple[list[str], np.ndarray, np.ndarray],
) -> pd.DataFrame:
    regular_labels, regular_radii, regular_values = regular
    latent_labels, latent_radii, latent_values = latent
    if not np.array_equal(regular_radii, latent_radii):
        raise ValueError("两模型表5的半径列不一致。")
    latent_by_label = {
        label: values for label, values in zip(latent_labels, latent_values)
    }
    rows: list[dict[str, object]] = []
    for label, values in zip(regular_labels, regular_values):
        counterpart = latent_by_label.get(label)
        if counterpart is None:
            continue
        for radius, regular_value, latent_value in zip(
            regular_radii, values, counterpart
        ):
            rows.append(
                {
                    "表5行": label,
                    "半径cm": radius,
                    "常规模型C": regular_value,
                    "潜热模型C": latent_value,
                    "差值ΔC": latent_value - regular_value,
                    "相对偏差%": 100.0 * (latent_value / regular_value - 1.0),
                }
            )
    # 常规模型已于57.5 h终止，60 h行只有潜热模型有值。
    for label, values in zip(latent_labels, latent_values):
        if label in regular_labels:
            continue
        for radius, latent_value in zip(latent_radii, values):
            rows.append(
                {
                    "表5行": label,
                    "半径cm": radius,
                    "常规模型C": np.nan,
                    "潜热模型C": latent_value,
                    "差值ΔC": np.nan,
                    "相对偏差%": np.nan,
                }
            )
    return pd.DataFrame(rows)


def compute_metrics(
    regular: tuple[list[str], np.ndarray, np.ndarray],
    latent: tuple[list[str], np.ndarray, np.ndarray],
    comparison: pd.DataFrame,
    regular_drying_h: float,
    latent_drying_h: float,
    regular_coarse_h: float,
    latent_coarse_h: float,
) -> dict:
    shared = comparison[comparison["相对偏差%"].notna()]
    max_row = shared.loc[shared["相对偏差%"].idxmax()]
    at_6h = shared[shared["表5行"] == "6"]
    at_12h = shared[shared["表5行"] == "12"]
    end_rows = shared[shared["表5行"] == "烘干结束时间"]
    end_identical = bool(
        np.allclose(
            end_rows["差值ΔC"].to_numpy(dtype=float), 0.0, atol=5.0e-5
        )
    )

    time_regular, center_regular = read_center_series(REGULAR_DIR)
    time_latent, center_latent = read_center_series(LATENT_DIR)

    # 以潜热模型在6 h、12 h的中心值为参照水平，计算等水平滞后。
    levels = [
        float(np.interp(6.0 * 3600.0, time_latent, center_latent)),
        float(np.interp(12.0 * 3600.0, time_latent, center_latent)),
    ]
    lag_milestones = {}
    for label, level in zip(["6h", "12h"], levels):
        t_regular, t_latent = lag_at_level(
            time_regular, center_regular, time_latent, center_latent, level
        )
        lag_milestones[label] = {
            "level_kg_kg": level,
            "regular_time_h": t_regular / 3600.0,
            "latent_time_h": t_latent / 3600.0,
            "lag_h": (t_latent - t_regular) / 3600.0,
        }
    terminal_lag_h = latent_drying_h - regular_drying_h
    lag_milestones["terminal"] = {
        "level_kg_kg": DRYING_THRESHOLD_KG_KG,
        "regular_time_h": regular_drying_h,
        "latent_time_h": latent_drying_h,
        "lag_h": terminal_lag_h,
    }

    energy = pd.read_csv(LATENT_DIR / "surface_energy_60s.csv", encoding="utf-8-sig")
    latent_flux = energy["latent_heat_flux_W_m2"].to_numpy(dtype=float)
    peak_flux = float(np.max(latent_flux))
    active = energy["time_s"].to_numpy(dtype=float)[latent_flux > 0.05 * peak_flux]
    surface_deficit = (
        energy["oven_temperature_C"] - energy["surface_temperature_C"]
    ).to_numpy(dtype=float)
    deficit_times = energy["time_s"].to_numpy(dtype=float)
    deficit_milestones = {}
    for threshold in (5.0, 2.0, 1.0):
        window = deficit_times[surface_deficit > threshold]
        deficit_milestones[str(threshold)] = float(window.max() / 3600.0)

    return {
        "drying_time": {
            "regular_h": regular_drying_h,
            "latent_heat_h": latent_drying_h,
            "difference_h": terminal_lag_h,
            "relative_change_pct": 100.0 * terminal_lag_h / regular_drying_h,
            "coarse_grid_difference_h": latent_coarse_h - regular_coarse_h,
            "coarse_fine_impact_agreement_h": abs(
                (latent_coarse_h - regular_coarse_h) - terminal_lag_h
            ),
        },
        "table5": {
            "shared_row_count": int(shared["表5行"].nunique()),
            "maximum_relative_change_pct": float(max_row["相对偏差%"]),
            "maximum_relative_change_row": str(max_row["表5行"]),
            "maximum_relative_change_radius_cm": float(max_row["半径cm"]),
            "center_6h_regular": float(at_6h.loc[at_6h["半径cm"] == 0.0, "常规模型C"].iloc[0]),
            "center_6h_latent": float(at_6h.loc[at_6h["半径cm"] == 0.0, "潜热模型C"].iloc[0]),
            "center_6h_relative_change_pct": float(
                at_6h.loc[at_6h["半径cm"] == 0.0, "相对偏差%"].iloc[0]
            ),
            "center_12h_relative_change_pct": float(
                at_12h.loc[at_12h["半径cm"] == 0.0, "相对偏差%"].iloc[0]
            ),
            "end_row_identical_to_4_decimals": end_identical,
            "steam_only_row": "60 h（常规模型已于57.5 h终止）",
        },
        "retained_water_per_unit_length_kg_m": {
            "6h": retained_water_difference(
                REGULAR_DIR, LATENT_DIR, 6.0 * 3600.0
            ),
            "12h": retained_water_difference(
                REGULAR_DIR, LATENT_DIR, 12.0 * 3600.0
            ),
            "24h": retained_water_difference(
                REGULAR_DIR, LATENT_DIR, 24.0 * 3600.0
            ),
            "dry_bulk_density_kg_m3": DRY_BULK_DENSITY_KG_M3,
        },
        "lag_closure": {
            "milestones_h": lag_milestones,
            "terminal_lag_h": terminal_lag_h,
            "share_set_by_6h_pct": 100.0
            * lag_milestones["6h"]["lag_h"]
            / terminal_lag_h,
            "share_set_by_12h_pct": 100.0
            * lag_milestones["12h"]["lag_h"]
            / terminal_lag_h,
        },
        "evaporation_window": {
            "peak_latent_heat_flux_W_m2": peak_flux,
            "last_time_above_5pct_of_peak_h": float(active.max() / 3600.0),
            "surface_deficit_last_time_h": {
                "5C": deficit_milestones["5.0"],
                "2C": deficit_milestones["2.0"],
                "1C": deficit_milestones["1.0"],
            },
        },
    }


def plot_figure(
    regular: tuple[list[str], np.ndarray, np.ndarray],
    latent: tuple[list[str], np.ndarray, np.ndarray],
    metrics: dict,
) -> None:
    regular_labels, regular_radii, regular_values = regular
    latent_labels, latent_radii, latent_values = latent

    fig = plt.figure(figsize=(7.2, 3.35), layout="constrained")
    outer = fig.add_gridspec(1, 3, width_ratios=[1.28, 0.92, 1.0])
    fig.suptitle(
        "第三问：蒸发潜热对表5的影响——早期大、终态钉住，滞后约3 h",
        fontsize=9.6,
        fontweight="bold",
        color=C_TEXT,
    )

    # a — 表5中心行随时间：两条答案曲线与终点。
    time_regular, center_regular = read_center_series(REGULAR_DIR)
    time_latent, center_latent = read_center_series(LATENT_DIR)
    ax_a = fig.add_subplot(outer[0, 0])
    add_panel_label(ax_a, "a")
    ax_a.set_title("表5中心含水率轨迹")
    ax_a.plot(
        time_regular / 3600.0,
        center_regular,
        color=C_REGULAR,
        lw=1.6,
        label="常规模型",
    )
    ax_a.plot(
        time_latent / 3600.0,
        center_latent,
        color=C_LATENT,
        lw=1.6,
        ls="--",
        label="蒸发潜热模型",
    )
    regular_h = metrics["drying_time"]["regular_h"]
    latent_h = metrics["drying_time"]["latent_heat_h"]
    ax_a.axhline(
        DRYING_THRESHOLD_KG_KG, color=C_REFERENCE, lw=0.85, ls=(0, (4, 3))
    )
    ax_a.axvline(regular_h, color=C_REGULAR, lw=0.8, ls=":")
    ax_a.axvline(latent_h, color=C_LATENT, lw=0.8, ls=":")
    t5 = metrics["table5"]
    ax_a.annotate(
        f"6 h：+{t5['center_6h_latent'] - t5['center_6h_regular']:.3f} kg/kg"
        f"（+{t5['center_6h_relative_change_pct']:.1f}%）",
        xy=(6.0, t5["center_6h_latent"]),
        xytext=(11.2, 1.92),
        fontsize=7.0,
        color=C_TEXT,
        arrowprops={"arrowstyle": "-", "color": C_MUTED, "lw": 0.7},
    )
    ax_a.annotate(
        f"终点 +{metrics['drying_time']['difference_h']:.2f} h"
        f"（+{metrics['drying_time']['relative_change_pct']:.1f}%）",
        xy=(latent_h, DRYING_THRESHOLD_KG_KG),
        xytext=(38.5, 0.62),
        fontsize=7.0,
        color=C_TEXT,
        arrowprops={"arrowstyle": "-", "color": C_MUTED, "lw": 0.7},
    )
    ax_a.text(30.5, 0.185, "烘干阈值 0.15", fontsize=6.6, color=C_MUTED)
    ax_a.set_xlim(0, 62.5)
    ax_a.set_ylim(0, 2.7)
    ax_a.set_xlabel("时间 (h)")
    ax_a.set_ylabel("中心含水率 (kg/kg)")
    finish_axes(ax_a)
    ax_a.legend(loc="upper right", handlelength=2.2)

    # b — 表5相对偏差热图：行=表5时刻，列=半径。
    comparison = pd.read_csv(
        SUMMARY_DIR / "table5_comparison.csv", encoding="utf-8-sig"
    )
    shared = comparison[comparison["相对偏差%"].notna()].copy()
    shared = shared[shared["表5行"] != "60"]
    ordered_labels = []
    for label in regular_labels:
        if label in shared["表5行"].to_numpy():
            ordered_labels.append(label)
    grid = np.full(
        (len(ordered_labels), len(regular_radii)), np.nan, dtype=float
    )
    for i, label in enumerate(ordered_labels):
        rows = shared[shared["表5行"] == label]
        for j, radius in enumerate(regular_radii):
            value = rows.loc[rows["半径cm"] == radius, "相对偏差%"]
            if not value.empty:
                grid[i, j] = float(value.iloc[0])
    maximum = float(np.nanmax(grid))
    cmap = LinearSegmentedColormap.from_list(
        "orange_seq",
        ["#fdf3ec", "#f6c39f", "#eb6834", "#8f3412"],
    )
    ax_b = fig.add_subplot(outer[0, 1])
    add_panel_label(ax_b, "b")
    ax_b.set_title("表5相对偏差（潜热 − 常规）/ 常规")
    mesh = ax_b.pcolormesh(
        np.arange(len(regular_radii) + 1) - 0.5,
        np.arange(len(ordered_labels) + 1) - 0.5,
        grid,
        cmap=cmap,
        vmin=0.0,
        vmax=max(maximum, 1.0),
        rasterized=True,
    )
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if np.isnan(grid[i, j]):
                continue
            dark = grid[i, j] / maximum > 0.55
            ax_b.text(
                j,
                i,
                f"{grid[i, j]:.1f}",
                ha="center",
                va="center",
                fontsize=6.1,
                color="white" if dark else C_TEXT,
            )
    row_labels = [label if label != "烘干结束时间" else "终点" for label in ordered_labels]
    ax_b.set_xticks(np.arange(len(regular_radii)))
    ax_b.set_xticklabels([f"{radius:g}" for radius in regular_radii])
    ax_b.set_yticks(np.arange(len(ordered_labels)))
    ax_b.set_yticklabels(row_labels)
    ax_b.set_xlabel("半径 (cm)")
    ax_b.set_ylabel("表5行 (h)")
    ax_b.set_xticks(np.arange(len(regular_radii)))
    cbar = fig.colorbar(mesh, ax=ax_b, pad=0.03, fraction=0.05)
    cbar.set_label("相对偏差 (%)")
    cbar.ax.tick_params(labelsize=6.4)
    ax_b.tick_params(length=2.6, width=0.7)

    # c — 等水平滞后时间曲线与蒸发活跃期。
    time_regular_s, center_regular = read_center_series(REGULAR_DIR)
    time_latent_s, center_latent = read_center_series(LATENT_DIR)
    levels = np.linspace(2.5, 0.155, 48)
    lag_curve = []
    for level in levels:
        t_regular, t_latent = lag_at_level(
            time_regular_s, center_regular, time_latent_s, center_latent, level
        )
        lag_curve.append((t_regular / 3600.0, (t_latent - t_regular) / 3600.0))
    lag_curve = np.asarray(lag_curve, dtype=float)
    ax_c = fig.add_subplot(outer[0, 2])
    add_panel_label(ax_c, "c")
    ax_c.set_title("等中心含水率水平的滞后时间")
    ax_c.plot(
        lag_curve[:, 0],
        lag_curve[:, 1],
        color=C_LAG,
        lw=1.6,
        label="滞后 t_潜热 − t_常规",
    )
    terminal_lag = metrics["drying_time"]["difference_h"]
    ax_c.axhline(
        terminal_lag, color=C_REFERENCE, lw=0.85, ls=(0, (4, 3))
    )
    ax_c.text(
        47.0,
        terminal_lag + 0.07,
        f"终末滞后 {terminal_lag:.2f} h",
        fontsize=6.8,
        color=C_MUTED,
        ha="right",
    )
    milestones = metrics["lag_closure"]["milestones_h"]
    for label, style in zip(["6h", "12h"], ["o", "s"]):
        ax_c.plot(
            milestones[label]["regular_time_h"],
            milestones[label]["lag_h"],
            marker=style,
            ms=4.2,
            mfc="white",
            mec=C_LAG,
            mew=1.0,
            zorder=4,
        )
        ax_c.annotate(
            f"{label}：{milestones[label]['lag_h']:.2f} h",
            xy=(
                milestones[label]["regular_time_h"],
                milestones[label]["lag_h"],
            ),
            xytext=(2.6, milestones[label]["lag_h"] + 0.24),
            fontsize=6.8,
            color=C_TEXT,
            arrowprops={"arrowstyle": "-", "color": C_MUTED, "lw": 0.7},
        )
    window_end = metrics["evaporation_window"][
        "surface_deficit_last_time_h"
    ]["2C"]
    ax_c.axvspan(0.0, window_end, color=C_LATENT, alpha=0.10, lw=0)
    ax_c.text(
        0.35,
        0.30,
        f"表面降温 >2 ℃ 持续至 {window_end:.1f} h",
        fontsize=6.6,
        color=C_TEXT,
    )
    ax_c.set_xlim(0, 62.5)
    ax_c.set_ylim(0, 3.5)
    ax_c.set_xlabel("常规模型达到该水平的时间 (h)")
    ax_c.set_ylabel("滞后时间 (h)")
    finish_axes(ax_c)
    ax_c.legend(loc="upper left", handlelength=2.0)

    save_figure(fig, FIGURE_DIR / "q3_latent_heat_table5_impact")


def main() -> None:
    apply_style()
    regular = read_table5(REGULAR_DIR / "table5_moisture.csv")
    latent = read_table5(LATENT_DIR / "table5_moisture.csv")
    regular_validation = json.loads(
        (REGULAR_DIR / "validation_summary.json").read_text(encoding="utf-8")
    )
    latent_validation = json.loads(
        (LATENT_DIR / "validation_summary.json").read_text(encoding="utf-8")
    )
    comparison = build_table5_comparison(regular, latent)
    metrics = compute_metrics(
        regular,
        latent,
        comparison,
        float(regular_validation["drying_time"]["production_h"]),
        float(latent_validation["drying_time"]["production_h"]),
        float(regular_validation["drying_time"]["coarse_h"]),
        float(latent_validation["drying_time"]["coarse_h"]),
    )

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(
        SUMMARY_DIR / "table5_comparison.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.4f",
    )
    time_regular_s, center_regular = read_center_series(REGULAR_DIR)
    time_latent_s, center_latent = read_center_series(LATENT_DIR)
    levels = np.linspace(2.5, 0.155, 48)
    lag_rows = []
    for level in levels:
        t_regular, t_latent = lag_at_level(
            time_regular_s, center_regular, time_latent_s, center_latent, level
        )
        lag_rows.append(
            {
                "中心含水率水平kg/kg": level,
                "常规模型达到时刻h": t_regular / 3600.0,
                "潜热模型达到时刻h": t_latent / 3600.0,
                "滞后h": (t_latent - t_regular) / 3600.0,
            }
        )
    pd.DataFrame(lag_rows).to_csv(
        SUMMARY_DIR / "lag_analysis.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )
    (SUMMARY_DIR / "table5_impact_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plot_figure(regular, latent, metrics)
    print(f"figures: {FIGURE_DIR}")
    print(f"summary: {SUMMARY_DIR}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
