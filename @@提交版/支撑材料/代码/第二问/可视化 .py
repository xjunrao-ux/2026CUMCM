"""Visualize A-problem question 2 coupled temperature and moisture results.

All quantitative panels use the complete 0-10800 s result grid.  Outputs are
PNG only, following the user's established project preference.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np


TABLE_TIMES_S = np.array([1800, 3600, 5400, 7200, 9000, 10800], dtype=int)
RADIUS_CM = np.round(np.arange(0.0, 2.0 + 0.05, 0.1), 1)
HEAT_TRANSFER_COEFFICIENT_W_M2_K = 25.0
MASS_TRANSFER_COEFFICIENT_M_S = 8.0e-7
PALETTE = {
    "air": "#606060",
    "center": "#0F4D92",
    "surface_temperature": "#D55E00",
    "surface_moisture": "#009E73",
    "average_temperature": "#B64342",
    "average_moisture": "#3775BA",
    "grid": "#D8D8D8",
}


def configure_style() -> None:
    available = {font.name for font in fm.fontManager.ttflist}
    candidates = ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"]
    selected = [name for name in candidates if name in available]
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": selected or ["DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8.0,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "axes.unicode_minus": False,
        }
    )


def load_numeric_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    if data.ndim != 2 or data.shape != (10801, 22):
        raise ValueError(f"Expected a 10801 x 22 result table: {path}")
    if not np.all(np.isfinite(data)):
        raise ValueError(f"Non-finite result value in {path}")
    return data[:, 0], data[:, 1:]


def load_environment(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    if data.ndim != 2 or data.shape != (10801, 4):
        raise ValueError(f"Expected a 10801 x 4 environment table: {path}")
    if not np.all(np.isfinite(data)):
        raise ValueError(f"Non-finite environment value in {path}")
    return data[:, 0], data[:, 1], data[:, 3]


def validate_data(
    temperature_times: np.ndarray,
    moisture_times: np.ndarray,
    environment_times: np.ndarray,
) -> None:
    expected_times = np.arange(0.0, 10801.0, 1.0)
    for label, times in [
        ("temperature", temperature_times),
        ("moisture", moisture_times),
        ("environment", environment_times),
    ]:
        if not np.array_equal(times, expected_times):
            raise ValueError(f"{label} time grid is not the required 0-10800 s grid.")
        if not np.all(np.diff(times) > 0.0):
            raise ValueError(f"{label} time grid is not strictly increasing.")


def cylinder_average(field: np.ndarray) -> np.ndarray:
    """Cross-sectional mean: 2/R^2 times the integral of field*r dr."""
    radius_m = RADIUS_CM / 100.0
    radius_outer = radius_m[-1]
    return 2.0 * np.trapezoid(field * radius_m[None, :], radius_m, axis=1) / (
        radius_outer**2
    )


def style_axis(axis: plt.Axes) -> None:
    axis.grid(True, color=PALETTE["grid"], linewidth=0.45, alpha=0.75)
    axis.set_axisbelow(True)


def save_png(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def render_heatmap(
    path: Path,
    times_s: np.ndarray,
    field: np.ndarray,
    title: str,
    colorbar_label: str,
    cmap: str,
    contour_levels: list[float],
) -> None:
    figure, axis = plt.subplots(figsize=(7.09, 3.95), constrained_layout=True)
    hours = times_s / 3600.0
    image = axis.pcolormesh(
        hours,
        RADIUS_CM,
        field.T,
        shading="auto",
        cmap=cmap,
        rasterized=True,
    )
    contours = axis.contour(
        hours,
        RADIUS_CM,
        field.T,
        levels=contour_levels,
        colors="white",
        linewidths=0.55,
        alpha=0.82,
    )
    axis.clabel(contours, inline=True, fontsize=6.5, fmt="%g")
    colorbar = figure.colorbar(image, ax=axis, pad=0.025, aspect=28)
    colorbar.set_label(colorbar_label)
    colorbar.outline.set_linewidth(0.6)
    axis.set(
        title=title,
        xlabel="时间 / h",
        ylabel="到药材中心的距离 / cm",
        xlim=(0.0, 3.0),
        ylim=(0.0, 2.0),
    )
    axis.set_xticks(np.arange(0.0, 3.01, 0.5))
    axis.set_yticks(np.arange(0.0, 2.01, 0.5))
    save_png(figure, path)


def render_radial_profiles(
    path: Path, temperature: np.ndarray, moisture: np.ndarray
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(7.09, 3.45))
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, TABLE_TIMES_S.size))
    for time_s, color in zip(TABLE_TIMES_S, colors):
        label = f"{time_s / 3600:g} h"
        axes[0].plot(
            RADIUS_CM,
            temperature[time_s],
            color=color,
            linewidth=1.35,
            label=label,
        )
        axes[1].plot(
            RADIUS_CM,
            moisture[time_s],
            color=color,
            linewidth=1.35,
            label=label,
        )
    axes[0].set(
        title="(a) 温度径向分布",
        xlabel="到药材中心的距离 / cm",
        ylabel="温度 / °C",
        xlim=(0.0, 2.0),
    )
    axes[1].set(
        title="(b) 水分浓度径向分布",
        xlabel="到药材中心的距离 / cm",
        ylabel="水分浓度 / (kg/kg)",
        xlim=(0.0, 2.0),
    )
    for axis in axes:
        style_axis(axis)
        axis.set_xticks(np.arange(0.0, 2.01, 0.5))
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=6,
        handlelength=2.5,
        columnspacing=1.0,
    )
    figure.subplots_adjust(left=0.10, right=0.98, bottom=0.16, top=0.78, wspace=0.28)
    save_png(figure, path)


def render_center_surface(
    path: Path,
    times_s: np.ndarray,
    temperature: np.ndarray,
    moisture: np.ndarray,
    oven_temperature: np.ndarray,
    oven_moisture: np.ndarray,
) -> None:
    hours = times_s / 3600.0
    figure, axes = plt.subplots(
        1, 2, figsize=(7.09, 3.25), constrained_layout=True
    )
    axes[0].plot(
        hours,
        oven_temperature,
        color=PALETTE["air"],
        linestyle="--",
        linewidth=1.15,
        label="烘房空气",
    )
    axes[0].plot(
        hours,
        temperature[:, -1],
        color=PALETTE["surface_temperature"],
        linewidth=1.45,
        label="药材表面",
    )
    axes[0].plot(
        hours,
        temperature[:, 0],
        color=PALETTE["center"],
        linewidth=1.45,
        label="药材中心",
    )
    axes[0].set(
        title="(a) 中心与表面温度响应",
        xlabel="时间 / h",
        ylabel="温度 / °C",
        xlim=(0.0, 3.0),
    )
    axes[1].plot(
        hours,
        oven_moisture,
        color=PALETTE["air"],
        linestyle="--",
        linewidth=1.15,
        label="烘房空气",
    )
    axes[1].plot(
        hours,
        moisture[:, -1],
        color=PALETTE["surface_moisture"],
        linewidth=1.45,
        label="药材表面",
    )
    axes[1].plot(
        hours,
        moisture[:, 0],
        color=PALETTE["center"],
        linewidth=1.45,
        label="药材中心",
    )
    axes[1].set(
        title="(b) 中心与表面水分响应",
        xlabel="时间 / h",
        ylabel="水分浓度 / (kg/kg)",
        xlim=(0.0, 3.0),
    )
    for axis in axes:
        style_axis(axis)
        axis.set_xticks(np.arange(0.0, 3.01, 0.5))
        axis.legend(loc="best")
    save_png(figure, path)


def render_average_evolution(
    path: Path,
    times_s: np.ndarray,
    temperature: np.ndarray,
    moisture: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    hours = times_s / 3600.0
    average_temperature = cylinder_average(temperature)
    average_moisture = cylinder_average(moisture)
    figure, axes = plt.subplots(
        1, 2, figsize=(7.09, 3.25), constrained_layout=True
    )
    axes[0].plot(
        hours,
        average_temperature,
        color=PALETTE["average_temperature"],
        linewidth=1.55,
    )
    axes[0].fill_between(
        hours,
        temperature[:, 0],
        temperature[:, -1],
        color=PALETTE["surface_temperature"],
        alpha=0.13,
        linewidth=0,
        label="中心—表面范围",
    )
    axes[0].set(
        title="(a) 体积平均温度",
        xlabel="时间 / h",
        ylabel="温度 / °C",
        xlim=(0.0, 3.0),
    )
    axes[0].legend(loc="lower right")
    axes[1].plot(
        hours,
        average_moisture,
        color=PALETTE["average_moisture"],
        linewidth=1.55,
    )
    axes[1].fill_between(
        hours,
        moisture[:, -1],
        moisture[:, 0],
        color=PALETTE["average_moisture"],
        alpha=0.13,
        linewidth=0,
        label="表面—中心范围",
    )
    axes[1].set(
        title="(b) 体积平均水分浓度",
        xlabel="时间 / h",
        ylabel="水分浓度 / (kg/kg)",
        xlim=(0.0, 3.0),
    )
    axes[1].legend(loc="upper right")
    for axis in axes:
        style_axis(axis)
        axis.set_xticks(np.arange(0.0, 3.01, 0.5))
    save_png(figure, path)
    return average_temperature, average_moisture


def render_phase_trajectory(
    path: Path,
    times_s: np.ndarray,
    average_temperature: np.ndarray,
    average_moisture: np.ndarray,
) -> None:
    figure, axis = plt.subplots(figsize=(4.25, 3.55), constrained_layout=True)
    hours = times_s / 3600.0
    points = axis.scatter(
        average_temperature[::10],
        average_moisture[::10],
        c=hours[::10],
        cmap="plasma",
        s=8,
        linewidths=0,
        rasterized=True,
    )
    axis.plot(
        average_temperature,
        average_moisture,
        color="#5B5B5B",
        linewidth=0.65,
        alpha=0.55,
        zorder=0,
    )
    for time_s in TABLE_TIMES_S:
        axis.scatter(
            average_temperature[time_s],
            average_moisture[time_s],
            facecolor="white",
            edgecolor="#272727",
            s=23,
            linewidth=0.7,
            zorder=3,
        )
        axis.annotate(
            f"{time_s / 3600:g} h",
            (average_temperature[time_s], average_moisture[time_s]),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=6.5,
        )
    colorbar = figure.colorbar(points, ax=axis, pad=0.025, aspect=25)
    colorbar.set_label("时间 / h")
    colorbar.outline.set_linewidth(0.6)
    axis.set(
        title="药材平均温度—水分演化轨迹",
        xlabel="体积平均温度 / °C",
        ylabel="体积平均水分浓度 / (kg/kg)",
    )
    style_axis(axis)
    save_png(figure, path)


def centered_moving_average(values: np.ndarray, window: int = 60) -> np.ndarray:
    """Edge-preserving moving average used only to clarify the plotted trend."""
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def render_surface_flux(
    path: Path,
    times_s: np.ndarray,
    temperature: np.ndarray,
    moisture: np.ndarray,
    oven_temperature: np.ndarray,
    oven_moisture: np.ndarray,
) -> dict[str, float]:
    """Plot convective heat and moisture-potential fluxes at the surface."""
    hours = times_s / 3600.0
    heat_flux = HEAT_TRANSFER_COEFFICIENT_W_M2_K * (
        oven_temperature - temperature[:, -1]
    )
    moisture_flux = MASS_TRANSFER_COEFFICIENT_M_S * (
        moisture[:, -1] - oven_moisture
    )
    moisture_flux_scaled = moisture_flux * 1.0e6
    heat_trend = centered_moving_average(heat_flux)
    moisture_trend = centered_moving_average(moisture_flux_scaled)
    duration = times_s[-1] - times_s[0]
    heat_peak_index = int(np.argmax(np.abs(heat_flux)))
    moisture_peak_index = int(np.argmax(np.abs(moisture_flux)))
    summary = {
        "heat_peak_w_m2": float(heat_flux[heat_peak_index]),
        "heat_peak_time_h": float(hours[heat_peak_index]),
        "heat_mean_abs_w_m2": float(
            np.trapezoid(np.abs(heat_flux), times_s) / duration
        ),
        "heat_3h_w_m2": float(heat_flux[-1]),
        "moisture_peak_1e6": float(moisture_flux_scaled[moisture_peak_index]),
        "moisture_peak_time_h": float(hours[moisture_peak_index]),
        "moisture_mean_abs_1e6": float(
            np.trapezoid(np.abs(moisture_flux_scaled), times_s) / duration
        ),
        "moisture_3h_1e6": float(moisture_flux_scaled[-1]),
    }

    figure, axes = plt.subplots(
        2, 1, figsize=(7.09, 4.65), sharex=True, constrained_layout=True
    )
    axes[0].plot(
        hours,
        heat_flux,
        color="#E7A06A",
        linewidth=0.55,
        alpha=0.55,
        label="Raw（逐秒通量）",
    )
    axes[0].plot(
        hours,
        heat_trend,
        color=PALETTE["surface_temperature"],
        linewidth=1.45,
        label="Mean（60 s 移动平均）",
    )
    axes[0].axhline(0.0, color="#707070", linewidth=0.65)
    axes[0].set(
        title="(a) 表面对流热通量（正值：空气 → 药材）",
        ylabel="热通量 / (W/m²)",
        xlim=(0.0, 3.0),
    )
    axes[0].text(
        0.985,
        0.72,
        "峰值：{:.2f}（{:.2f} h）\n平均绝对值：{:.2f}\n3 h：{:.2f}".format(
            summary["heat_peak_w_m2"],
            summary["heat_peak_time_h"],
            summary["heat_mean_abs_w_m2"],
            summary["heat_3h_w_m2"],
        ),
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=7.0,
        color="#4A4A4A",
    )

    axes[1].plot(
        hours,
        moisture_flux_scaled,
        color="#75C8B5",
        linewidth=0.55,
        alpha=0.58,
        label="Raw（逐秒通量）",
    )
    axes[1].plot(
        hours,
        moisture_trend,
        color=PALETTE["surface_moisture"],
        linewidth=1.45,
        label="Mean（60 s 移动平均）",
    )
    axes[1].axhline(0.0, color="#707070", linewidth=0.65)
    axes[1].set(
        title="(b) 表面水分通量（正值：药材 → 空气）",
        xlabel="时间 / h",
        ylabel="水分通量 / [10^-6 (kg/kg)·m/s]",
        xlim=(0.0, 3.0),
    )
    axes[1].text(
        0.985,
        0.72,
        "峰值：{:.3f}（{:.2f} h）\n平均绝对值：{:.3f}\n3 h：{:.3f}".format(
            summary["moisture_peak_1e6"],
            summary["moisture_peak_time_h"],
            summary["moisture_mean_abs_1e6"],
            summary["moisture_3h_1e6"],
        ),
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=7.0,
        color="#4A4A4A",
    )

    for axis in axes:
        style_axis(axis)
        axis.legend(loc="upper right")
    axes[1].set_xticks(np.arange(0.0, 3.01, 0.5))
    save_png(figure, path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    result_dir = repo_root / "results" / "A_problem2_coupled"
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else repo_root / "picture" / "A_problem2_coupled"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    temperature_times, temperature = load_numeric_csv(
        result_dir / "temperature_full_1s_0p1cm.csv"
    )
    moisture_times, moisture = load_numeric_csv(
        result_dir / "moisture_full_1s_0p1cm.csv"
    )
    environment_times, oven_temperature, oven_moisture = load_environment(
        result_dir / "environment_0_3h_1s_kelvin.csv"
    )
    validate_data(temperature_times, moisture_times, environment_times)
    configure_style()

    render_heatmap(
        output_dir / "temperature_heatmap.png",
        temperature_times,
        temperature,
        "3小时内药材温度的时空变化",
        "温度 / °C",
        "magma",
        [30, 35, 40, 45, 49],
    )
    render_heatmap(
        output_dir / "moisture_heatmap.png",
        moisture_times,
        moisture,
        "3小时内药材水分浓度的时空变化",
        "水分浓度 / (kg/kg)",
        "viridis",
        [1.2, 1.5, 1.8, 2.1, 2.4],
    )
    render_radial_profiles(
        output_dir / "radial_profiles.png", temperature, moisture
    )
    render_center_surface(
        output_dir / "center_surface_evolution.png",
        temperature_times,
        temperature,
        moisture,
        oven_temperature,
        oven_moisture,
    )
    average_temperature, average_moisture = render_average_evolution(
        output_dir / "volume_average_evolution.png",
        temperature_times,
        temperature,
        moisture,
    )
    render_phase_trajectory(
        output_dir / "temperature_moisture_trajectory.png",
        temperature_times,
        average_temperature,
        average_moisture,
    )
    flux_summary = render_surface_flux(
        output_dir / "surface_heat_moisture_flux.png",
        temperature_times,
        temperature,
        moisture,
        oven_temperature,
        oven_moisture,
    )

    print(f"Generated 7 PNG figures in: {output_dir}")
    print(
        "3 h volume averages: "
        f"T={average_temperature[-1]:.4f} C, "
        f"C={average_moisture[-1]:.4f} kg/kg"
    )
    print("Surface flux summary:")
    for name, value in flux_summary.items():
        print(f"  {name}={value:.6g}")


if __name__ == "__main__":
    main()
