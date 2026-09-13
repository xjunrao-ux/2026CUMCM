"""Visualize the full-process temperature and moisture solution for Question 3.

The script reads the 60 s radial solution, reconstructs the known initial state,
and exports publication-ready PNG figures only.  It does not modify the model
solution or smooth the source data.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Required editable-text configuration; PNG is retained as the requested output.
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['pdf.fonttype'] = 42

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd


PALETTE = {
    "blue": "#0F4D92",
    "blue_2": "#3775BA",
    "teal": "#42949E",
    "violet": "#7C6CCF",
    "red": "#B64342",
    "orange": "#E28E2C",
    "gray": "#767676",
    "dark": "#272727",
    "light": "#E7E9ED",
}

RADIUS_CM = 2.0
DRYING_THRESHOLD = 0.15
MEASURED_END_H = 4.0


def apply_style() -> None:
    """Apply one restrained visual system to all figures."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Arial",
                "DejaVu Sans",
            ],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def parse_radius(column: str) -> float:
    if not column.startswith("r_") or not column.endswith("_cm"):
        raise ValueError(f"Unexpected radial column name: {column}")
    return float(column.removeprefix("r_").removesuffix("_cm"))


def read_solution(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    moisture = pd.read_csv(data_dir / "moisture_full_60s_0p1cm.csv")
    temperature = pd.read_csv(data_dir / "temperature_auxiliary_60s_0p1cm.csv")
    validation = json.loads((data_dir / "validation_summary.json").read_text(encoding="utf-8"))

    if list(moisture.columns) != list(temperature.columns):
        raise ValueError("Temperature and moisture files have different columns.")
    if not np.array_equal(moisture["time_s"].to_numpy(), temperature["time_s"].to_numpy()):
        raise ValueError("Temperature and moisture files have different time grids.")

    radii = np.asarray([parse_radius(name) for name in moisture.columns[1:]], dtype=float)
    time_s = moisture["time_s"].to_numpy(dtype=float)
    c = moisture.iloc[:, 1:].to_numpy(dtype=float)
    temp = temperature.iloc[:, 1:].to_numpy(dtype=float)

    if radii.size != 21 or not np.allclose(radii, np.arange(0.0, 2.01, 0.1)):
        raise ValueError("Expected a 0-2 cm radial grid at 0.1 cm spacing.")
    if np.any(np.diff(time_s) <= 0) or not np.all(np.isfinite(c)) or not np.all(np.isfinite(temp)):
        raise ValueError("The solution contains an invalid time grid or non-finite values.")

    # The result files start at 60 s; the known uniform initial condition is added
    # solely so every plot begins at t=0.
    time_s = np.insert(time_s, 0, 0.0)
    c = np.vstack([np.full(radii.size, 2.55), c])
    temp = np.vstack([np.full(radii.size, 28.0), temp])
    return time_s, radii, temp, c, validation


def add_stage_guides(ax: plt.Axes, drying_h: float, *, show_label: bool = True) -> None:
    ax.axvspan(0, MEASURED_END_H, color=PALETTE["light"], alpha=0.55, zorder=0)
    ax.axvline(MEASURED_END_H, color=PALETTE["gray"], lw=1.0, ls="--", zorder=2)
    ax.axvline(drying_h, color=PALETTE["red"], lw=1.1, ls=":", zorder=2)
    if show_label:
        ax.text(
            MEASURED_END_H,
            1.01,
            "4 h：转为末1 h均值边界",
            transform=ax.get_xaxis_transform(),
            ha="left",
            va="bottom",
            color=PALETTE["gray"],
            fontsize=7.5,
        )


def save_png(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def plot_full_process(
    time_h: np.ndarray,
    radii: np.ndarray,
    temp: np.ndarray,
    c: np.ndarray,
    drying_h: float,
    output_dir: Path,
) -> None:
    selected = [0, 5, 10, 15, 20]
    colors = ["#0F4D92", "#3775BA", "#42949E", "#7C6CCF", "#B64342"]
    fig, axes = plt.subplots(2, 1, figsize=(7.1, 5.4), sharex=True)

    for idx, color in zip(selected, colors):
        label = f"r = {radii[idx]:g} cm"
        axes[0].plot(time_h, temp[:, idx], color=color, lw=1.6, label=label)
        axes[1].plot(time_h, c[:, idx], color=color, lw=1.6, label=label)

    add_stage_guides(axes[0], drying_h)
    add_stage_guides(axes[1], drying_h, show_label=False)
    axes[1].axhline(
        DRYING_THRESHOLD,
        color=PALETTE["red"],
        lw=1.1,
        ls="--",
        label="干燥阈值 0.15 kg/kg",
    )
    axes[0].set_ylabel("温度 / °C")
    axes[1].set_ylabel("水分浓度 / (kg/kg)")
    axes[1].set_xlabel("时间 / h")
    axes[0].set_title("全时段温度响应：温度场先达到近似均匀")
    axes[1].set_title("全时段水分响应：中心存在显著的长尾干燥")
    axes[0].set_ylim(27, 52)
    axes[1].set_ylim(0, 2.65)
    axes[1].set_xlim(0, drying_h * 1.015)
    axes[0].grid(axis="y", color="#D8D8D8", lw=0.6, alpha=0.65)
    axes[1].grid(axis="y", color="#D8D8D8", lw=0.6, alpha=0.65)
    axes[0].legend(ncol=5, loc="lower right", columnspacing=0.9, handlelength=1.8)
    axes[1].legend(ncol=3, loc="upper right", columnspacing=1.0, handlelength=1.8)
    axes[0].text(-0.07, 1.04, "a", transform=axes[0].transAxes, fontweight="bold", fontsize=11)
    axes[1].text(-0.07, 1.04, "b", transform=axes[1].transAxes, fontweight="bold", fontsize=11)
    fig.subplots_adjust(hspace=0.36)
    save_png(fig, output_dir / "q3_full_process_curves.png")


def plot_heatmaps(
    time_h: np.ndarray,
    radii: np.ndarray,
    temp: np.ndarray,
    c: np.ndarray,
    drying_h: float,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(7.1, 5.4), sharex=True)
    temp_mesh = axes[0].pcolormesh(
        time_h,
        radii,
        temp.T,
        shading="auto",
        cmap="RdYlBu_r",
        vmin=float(np.min(temp)),
        vmax=float(np.max(temp)),
        rasterized=True,
    )
    moisture_mesh = axes[1].pcolormesh(
        time_h,
        radii,
        c.T,
        shading="auto",
        cmap="YlGnBu",
        norm=mcolors.LogNorm(vmin=max(float(np.min(c)), 0.045), vmax=float(np.max(c))),
        rasterized=True,
    )
    contour = axes[1].contour(
        time_h,
        radii,
        c.T,
        levels=[DRYING_THRESHOLD],
        colors=[PALETTE["red"]],
        linewidths=1.25,
    )
    axes[1].clabel(contour, fmt={DRYING_THRESHOLD: "C = 0.15"}, fontsize=7, inline=True)

    for ax in axes:
        ax.axvline(MEASURED_END_H, color="white", lw=1.0, ls="--")
        ax.axvline(drying_h, color=PALETTE["red"], lw=1.0, ls=":")
        ax.set_ylabel("到中心距离 r / cm")
        ax.set_ylim(0, RADIUS_CM)
    axes[1].set_xlabel("时间 / h")
    axes[0].set_title("温度时空演化")
    axes[1].set_title("水分时空演化（对数色标；红线为干燥前沿）")
    cbar_t = fig.colorbar(temp_mesh, ax=axes[0], pad=0.018, aspect=32)
    cbar_t.set_label("温度 / °C")
    cbar_c = fig.colorbar(moisture_mesh, ax=axes[1], pad=0.018, aspect=32)
    cbar_c.set_label("水分浓度 / (kg/kg)")
    axes[0].text(-0.07, 1.04, "a", transform=axes[0].transAxes, fontweight="bold", fontsize=11)
    axes[1].text(-0.07, 1.04, "b", transform=axes[1].transAxes, fontweight="bold", fontsize=11)
    fig.subplots_adjust(hspace=0.34)
    save_png(fig, output_dir / "q3_spatiotemporal_heatmaps.png")


def nearest_indices(time_h: np.ndarray, targets_h: list[float]) -> list[int]:
    return [int(np.argmin(np.abs(time_h - target))) for target in targets_h]


def plot_radial_profiles(
    time_h: np.ndarray,
    radii: np.ndarray,
    temp: np.ndarray,
    c: np.ndarray,
    drying_h: float,
    output_dir: Path,
) -> None:
    temp_targets = [0.5, 1.0, 2.0, 4.0, 12.0, drying_h]
    moisture_targets = [4.0, 12.0, 24.0, 36.0, 48.0, drying_h]
    cmap = plt.get_cmap("viridis")
    colors = [cmap(value) for value in np.linspace(0.1, 0.9, 6)]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.1))

    for idx, target, color in zip(nearest_indices(time_h, temp_targets), temp_targets, colors):
        label = "结束" if math.isclose(target, drying_h) else f"{target:g} h"
        axes[0].plot(radii, temp[idx], color=color, lw=1.6, marker="o", ms=2.4, label=label)
    for idx, target, color in zip(nearest_indices(time_h, moisture_targets), moisture_targets, colors):
        label = "结束" if math.isclose(target, drying_h) else f"{target:g} h"
        axes[1].plot(radii, c[idx], color=color, lw=1.6, marker="o", ms=2.4, label=label)

    axes[1].axhline(DRYING_THRESHOLD, color=PALETTE["red"], lw=1.0, ls="--")
    axes[0].set_title("径向温度剖面")
    axes[1].set_title("径向水分剖面")
    axes[0].set_ylabel("温度 / °C")
    axes[1].set_ylabel("水分浓度 / (kg/kg)")
    for ax in axes:
        ax.set_xlabel("到中心距离 r / cm")
        ax.set_xlim(0, RADIUS_CM)
        ax.grid(axis="y", color="#D8D8D8", lw=0.6, alpha=0.65)
        ax.legend(ncol=2, loc="best", columnspacing=0.8, handlelength=1.5)
    axes[0].text(-0.14, 1.04, "a", transform=axes[0].transAxes, fontweight="bold", fontsize=11)
    axes[1].text(-0.14, 1.04, "b", transform=axes[1].transAxes, fontweight="bold", fontsize=11)
    fig.subplots_adjust(wspace=0.34)
    save_png(fig, output_dir / "q3_radial_profiles.png")


def crossing_times(
    time_h: np.ndarray, c: np.ndarray, validation: dict
) -> np.ndarray:
    """Return approximate first threshold times on the saved 60 s grid."""
    result = np.full(c.shape[1], np.nan)
    for col in range(c.shape[1]):
        indices = np.flatnonzero(c[:, col] < DRYING_THRESHOLD)
        if indices.size:
            result[col] = time_h[indices[0]]

    # The exact centre crossing is retained in the validation record because the
    # required four-decimal spreadsheet rounds its terminal value to 0.1500.
    result[0] = float(validation["drying_time"]["production_h"])
    return result


def radial_average(values: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Cross-sectional average for an axisymmetric cylinder."""
    return 2.0 * np.trapezoid(values * radii[None, :], radii, axis=1) / RADIUS_CM**2


def plot_drying_metrics(
    time_h: np.ndarray,
    radii: np.ndarray,
    temp: np.ndarray,
    c: np.ndarray,
    validation: dict,
    output_dir: Path,
) -> dict:
    drying_h = float(validation["drying_time"]["production_h"])
    c_avg = radial_average(c, radii)
    c_spread = c[:, 0] - c[:, -1]
    t_spread = temp[:, -1] - temp[:, 0]
    dry_times = crossing_times(time_h, c, validation)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.1, 3.15))

    ax_a.plot(time_h, c[:, 0], color=PALETTE["blue"], lw=1.3, label="中心")
    ax_a.plot(time_h, c_avg, color=PALETTE["teal"], lw=1.5, label="截面平均")
    ax_a.plot(time_h, c[:, -1], color=PALETTE["orange"], lw=1.3, label="表面")
    ax_a.axhline(DRYING_THRESHOLD, color=PALETTE["red"], lw=1.0, ls="--")
    add_stage_guides(ax_a, drying_h, show_label=False)
    ax_a.set(xlabel="时间 / h", ylabel="水分浓度 / (kg/kg)", title="整体脱水与极值包络")
    ax_a.set_xlim(0, drying_h * 1.015)
    ax_a.legend(ncol=3, columnspacing=0.8, handlelength=1.5)

    ax_b.plot(time_h, c_spread, color=PALETTE["violet"], lw=1.5, label="中心−表面水分差")
    ax_b.set(xlabel="时间 / h", ylabel="ΔC / (kg/kg)", title="径向水分不均匀性")
    ax_b.set_xlim(0, drying_h * 1.015)
    ax_b.grid(axis="y", color="#D8D8D8", lw=0.6, alpha=0.65)
    peak_idx = int(np.argmax(c_spread))
    ax_b.scatter(time_h[peak_idx], c_spread[peak_idx], s=22, color=PALETTE["red"], zorder=4)
    ax_b.annotate(
        f"最大差值 {c_spread[peak_idx]:.3f}\n@ {time_h[peak_idx]:.2f} h",
        xy=(time_h[peak_idx], c_spread[peak_idx]),
        xytext=(8, -28),
        textcoords="offset points",
        fontsize=7.5,
        arrowprops={"arrowstyle": "-", "color": PALETTE["gray"], "lw": 0.8},
    )

    for label, ax in zip("ab", [ax_a, ax_b]):
        ax.text(-0.14, 1.05, label, transform=ax.transAxes, fontweight="bold", fontsize=11)

    fig.subplots_adjust(wspace=0.34)
    save_png(fig, output_dir / "q3_drying_metrics.png")

    # A separate time-radius map makes the advancing drying front read in the
    # same direction as the spatiotemporal heatmap: time from left to right,
    # centre at the bottom, and surface at the top.
    fig_front, ax_front = plt.subplots(figsize=(6.4, 3.8))
    ax_front.fill_betweenx(
        radii,
        dry_times,
        drying_h,
        color=PALETTE["blue_2"],
        alpha=0.16,
        label="该径向位置已经达标",
    )
    ax_front.plot(dry_times, radii, color=PALETTE["blue"], lw=2.0, marker="o", ms=4.2)
    ax_front.axvline(MEASURED_END_H, color=PALETTE["gray"], lw=1.0, ls="--")
    ax_front.scatter(
        [dry_times[-1], dry_times[0]],
        [radii[-1], radii[0]],
        s=[36, 42],
        color=[PALETTE["orange"], PALETTE["red"]],
        zorder=4,
    )
    ax_front.annotate(
        f"表面首先达标：{dry_times[-1]:.2f} h",
        xy=(dry_times[-1], radii[-1]),
        xytext=(8, -20),
        textcoords="offset points",
        fontsize=8,
        arrowprops={"arrowstyle": "-", "color": PALETTE["gray"], "lw": 0.8},
    )
    ax_front.annotate(
        f"中心最后达标：{dry_times[0]:.2f} h",
        xy=(dry_times[0], radii[0]),
        xytext=(-118, 16),
        textcoords="offset points",
        fontsize=8,
        arrowprops={"arrowstyle": "-", "color": PALETTE["gray"], "lw": 0.8},
    )
    ax_front.text(
        MEASURED_END_H,
        1.01,
        "4 h",
        transform=ax_front.get_xaxis_transform(),
        ha="center",
        va="bottom",
        color=PALETTE["gray"],
        fontsize=8,
    )
    ax_front.set(
        xlabel="首次达到 C < 0.15 的时间 / h",
        ylabel="到中心距离 r / cm",
        title="干燥前沿由药材表面逐步推进至中心",
    )
    ax_front.set_xlim(0, drying_h + 2.2)
    ax_front.set_ylim(0, RADIUS_CM)
    ax_front.set_yticks(np.arange(0, 2.01, 0.25))
    ax_front.grid(color="#D8D8D8", lw=0.6, alpha=0.65)
    ax_front.legend(loc="lower left")
    save_png(fig_front, output_dir / "q3_threshold_front.png")
    switch_idx = int(np.argmin(np.abs(time_h - MEASURED_END_H)))
    peak_t_idx = int(np.argmax(t_spread))
    uniform_candidates = np.flatnonzero(
        (np.arange(time_h.size) > peak_t_idx) & (t_spread <= 0.1)
    )
    uniform_time_h = float(time_h[uniform_candidates[0]]) if uniform_candidates.size else None
    return {
        "drying_time_h": drying_h,
        "state_at_4h": {
            "centre_temperature_C": float(temp[switch_idx, 0]),
            "surface_temperature_C": float(temp[switch_idx, -1]),
            "centre_moisture_kg_kg": float(c[switch_idx, 0]),
            "surface_moisture_kg_kg": float(c[switch_idx, -1]),
        },
        "peak_moisture_difference_kg_kg": float(c_spread[peak_idx]),
        "peak_moisture_difference_time_h": float(time_h[peak_idx]),
        "maximum_temperature_difference_C": float(np.max(t_spread)),
        "maximum_temperature_difference_time_h": float(time_h[peak_t_idx]),
        "temperature_spatial_difference_below_0p1C_after_h": uniform_time_h,
        "final_cross_section_mean_moisture_kg_kg": float(c_avg[-1]),
        "radius_cm": radii.tolist(),
        "threshold_crossing_time_h": dry_times.tolist(),
        "note": "Threshold times use the saved 60 s grid; the centre uses the exact terminal record.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=repo_root / "results" / "A_problem3_drying_time",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "picture" / "A_problem3_drying_time",
    )
    args = parser.parse_args()

    apply_style()
    time_s, radii, temp, c, validation = read_solution(args.data_dir)
    time_h = time_s / 3600.0
    drying_h = float(validation["drying_time"]["production_h"])

    plot_full_process(time_h, radii, temp, c, drying_h, args.output_dir)
    plot_heatmaps(time_h, radii, temp, c, drying_h, args.output_dir)
    plot_radial_profiles(time_h, radii, temp, c, drying_h, args.output_dir)
    metrics = plot_drying_metrics(time_h, radii, temp, c, validation, args.output_dir)
    (args.output_dir / "q3_visualization_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Input records (including t=0): {time_h.size}")
    print(f"Drying time: {drying_h:.6f} h")
    print(f"Figures written to: {args.output_dir}")
    for path in sorted(args.output_dir.glob("*.png")):
        print(path.name)


if __name__ == "__main__":
    main()
