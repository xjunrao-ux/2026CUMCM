"""A题问题1：有限体积结果可视化。

本脚本只读取fvm_model.py输出的无舍入模型状态和表面通量，不重新求解
微分方程。每幅图的代码起点均用“图N”分隔注释标明。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

import fvm_model as model


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def configure_fonts() -> None:
    """沿用原版中文字体和竞赛论文图件设置。"""
    available = {font.name for font in fm.fontManager.ttflist}
    candidates = ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"]
    selected = [name for name in candidates if name in available]
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": selected or ["DejaVu Sans"],
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.7,
        }
    )


def style_axis(axis: plt.Axes) -> None:
    axis.grid(True, color="#D9D9D9", linewidth=0.45, alpha=0.75)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def save_figure(figure: plt.Figure, png_path: Path, png_dpi: int = 300) -> None:
    """按原分辨率设置仅输出PNG图件。"""
    figure.savefig(png_path, dpi=png_dpi, facecolor="white")


def load_model_state(path: Path) -> tuple[model.TemperatureResult, model.MoistureResult]:
    """从模型脚本保存的无舍入NPZ中恢复温度场和水分场。"""
    with np.load(path) as state:
        temperature = model.TemperatureResult(
            times_s=state["times_s"],
            radii_m=state["temperature_radii_m"],
            temperature_c=state["temperature_c"],
            capacities_j_k=state["temperature_capacities_j_k"],
            boundary_heat_input_j_m=float(state["temperature_boundary_heat_input_j_m"]),
        )
        moisture = model.MoistureResult(
            times_s=state["times_s"],
            radii_m=state["moisture_radii_m"],
            moisture_kg_kg=state["moisture_kg_kg"],
            surface_moisture_kg_kg=state["surface_moisture_kg_kg"],
            volumes_m3_m=state["moisture_volumes_m3_m"],
            boundary_outflow_m3_kg_kg_m=float(
                state["moisture_boundary_outflow_m3_kg_kg_m"]
            ),
            maximum_picard_iterations=int(state["maximum_picard_iterations"]),
        )
    return temperature, moisture


def load_surface_fluxes(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("表面通量文件为空。")
    return {
        name: np.asarray([float(row[name]) for row in rows], dtype=float)
        for name in rows[0]
    }


# ===========================================================================
# 图1开始：七个报告时刻的温度/水分径向分布图
# ===========================================================================
def plot_radial_profiles(
    path: Path,
    temperature: model.TemperatureResult,
    moisture: model.MoistureResult,
) -> None:
    visual_radii_cm = np.linspace(0.0, 2.0, 201)
    temperature_profiles = model.sample_field(
        temperature.times_s,
        temperature.radii_m,
        temperature.temperature_c,
        model.TABLE_TIMES_S,
        visual_radii_cm,
    )
    moisture_profiles = model.sample_moisture_field(
        moisture, model.TABLE_TIMES_S, visual_radii_cm
    )
    figure, axes = plt.subplots(
        1, 2, figsize=(7.09, 3.25), constrained_layout=True
    )
    colors = plt.get_cmap("viridis")(
        np.linspace(0.08, 0.92, model.TABLE_TIMES_S.size)
    )
    for index, (time_s, color) in enumerate(zip(model.TABLE_TIMES_S, colors)):
        label = f"{time_s:g} s" if time_s % 60 else f"{time_s / 60:g} min"
        axes[0].plot(
            visual_radii_cm,
            temperature_profiles[index],
            color=color,
            linewidth=1.35,
            label=label,
        )
        axes[1].plot(
            visual_radii_cm,
            moisture_profiles[index],
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
    axes[0].legend(frameon=False, ncol=2, loc="upper left")
    axes[1].legend(frameon=False, ncol=2, loc="lower left")
    save_figure(figure, path)
    plt.close(figure)


# ===========================================================================
# 图2和图3开始：温度场热力图、水分场热力图
# ===========================================================================
def plot_heatmap(
    path: Path,
    times_s: np.ndarray,
    radii_cm: np.ndarray,
    values: np.ndarray,
    title: str,
    colorbar_label: str,
    cmap: str,
) -> None:
    figure, axis = plt.subplots(figsize=(7.09, 3.95), constrained_layout=True)
    image = axis.pcolormesh(
        times_s / 60.0,
        radii_cm,
        values.T,
        shading="auto",
        cmap=cmap,
        rasterized=True,
    )
    colorbar = figure.colorbar(image, ax=axis, pad=0.025, aspect=28)
    colorbar.set_label(colorbar_label)
    colorbar.outline.set_linewidth(0.6)
    axis.set(
        xlabel="时间 / min",
        ylabel="到药材中心的距离 / cm",
        title=title,
        xlim=(0.0, model.END_TIME_S / 60.0),
        ylim=(0.0, model.RADIUS_M * 100.0),
    )
    axis.set_xticks(np.arange(0.0, 31.0, 5.0))
    axis.set_yticks(np.arange(0.0, 2.01, 0.5))
    save_figure(figure, path)
    plt.close(figure)


# ===========================================================================
# 图4开始：烘房、药材中心和药材表面的温度/水分响应图
# ===========================================================================
def plot_center_surface_response(
    path: Path,
    environment: model.EnvironmentData,
    temperature: model.TemperatureResult,
    moisture: model.MoistureResult,
) -> None:
    times_s = temperature.times_s
    time_minutes = times_s / 60.0
    if not np.all(np.diff(environment.times_s) > 0.0):
        raise ValueError("环境数据时间必须严格递增，才能进行线性插值。")
    oven_temperature = np.interp(
        times_s, environment.times_s, environment.temperature_c
    )
    oven_moisture = np.interp(
        times_s, environment.times_s, environment.moisture_kg_kg
    )
    temperature_center_surface = model.sample_field(
        times_s,
        temperature.radii_m,
        temperature.temperature_c,
        times_s,
        np.array([0.0, 2.0]),
    )
    moisture_center_surface = model.sample_moisture_field(
        moisture, times_s, np.array([0.0, 2.0])
    )
    figure, axes = plt.subplots(
        1, 2, figsize=(7.09, 3.25), constrained_layout=True
    )
    axes[0].plot(
        time_minutes,
        oven_temperature,
        color="#555555",
        linestyle="--",
        linewidth=1.2,
        label="烘房空气",
    )
    axes[0].plot(
        time_minutes,
        temperature_center_surface[:, 1],
        color="#D55E00",
        linewidth=1.5,
        label="药材表面",
    )
    axes[0].plot(
        time_minutes,
        temperature_center_surface[:, 0],
        color="#0072B2",
        linewidth=1.5,
        label="药材中心",
    )
    axes[0].set(
        title="(a) 中心与表面温度响应",
        xlabel="时间 / min",
        ylabel="温度 / °C",
        xlim=(0.0, 30.0),
    )
    axes[1].plot(
        time_minutes,
        oven_moisture,
        color="#555555",
        linestyle="--",
        linewidth=1.2,
        label="烘房空气",
    )
    axes[1].plot(
        time_minutes,
        moisture_center_surface[:, 1],
        color="#009E73",
        linewidth=1.5,
        label="药材表面",
    )
    axes[1].plot(
        time_minutes,
        moisture_center_surface[:, 0],
        color="#0072B2",
        linewidth=1.5,
        label="药材中心",
    )
    axes[1].set(
        title="(b) 中心与表面水分响应",
        xlabel="时间 / min",
        ylabel="水分浓度 / (kg/kg)",
        xlim=(0.0, 30.0),
    )
    for axis in axes:
        style_axis(axis)
        axis.set_xticks(np.arange(0.0, 31.0, 5.0))
        axis.legend(frameon=False, loc="best")
    save_figure(figure, path)
    plt.close(figure)


# ===========================================================================
# 图5开始：药材表面水分通量与热流密度图
# ===========================================================================
def plot_surface_flux(path: Path, fluxes: dict[str, np.ndarray]) -> None:
    time_minutes = fluxes["time_s"] / 60.0
    heat_flux = fluxes["heat_flux_into_herb_w_m2"]
    moisture_flux_g_m2_s = 1000.0 * fluxes["moisture_mass_flux_out_kg_m2_s"]
    figure, axes = plt.subplots(
        1, 2, figsize=(7.09, 3.25), constrained_layout=True
    )
    axes[0].plot(time_minutes, heat_flux, color="#D55E00", linewidth=1.5)
    axes[0].fill_between(
        time_minutes, 0.0, heat_flux, color="#E69F00", alpha=0.16, linewidth=0
    )
    axes[0].set(
        title="(a) 进入药材的表面热流密度",
        xlabel="时间 / min",
        ylabel="热流密度 / (W/m²)",
        xlim=(0.0, model.END_TIME_S / 60.0),
    )
    axes[1].plot(
        time_minutes, moisture_flux_g_m2_s, color="#009E73", linewidth=1.5
    )
    axes[1].fill_between(
        time_minutes,
        0.0,
        moisture_flux_g_m2_s,
        color="#009E73",
        alpha=0.14,
        linewidth=0,
    )
    axes[1].set(
        title="(b) 离开药材的表面水分通量",
        xlabel="时间 / min",
        ylabel="水分质量通量 / [g/(m²·s)]",
        xlim=(0.0, model.END_TIME_S / 60.0),
    )
    for axis in axes:
        style_axis(axis)
        axis.axhline(0.0, color="#555555", linewidth=0.65)
        axis.set_xticks(np.arange(0.0, 31.0, 5.0))
    save_figure(figure, path)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    project_root = default_project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-data",
        type=Path,
        default=(
            project_root
            / "results"
            / "A_problem1_modular"
            / "input"
            / "attachment1_linear_1s.csv"
        ),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=project_root / "results" / "A_problem1_modular" / "model",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "picture" / "A_problem1_modular",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.resolve()
    state_path = model_dir / "model_state.npz"
    if not state_path.exists():
        raise FileNotFoundError(f"找不到模型状态：{state_path}\n请先运行fvm_model.py。")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_fonts()

    environment = model.load_environment_csv(args.input_data.resolve())
    temperature, moisture = load_model_state(state_path)
    fluxes = load_surface_fluxes(model_dir / "surface_fluxes_1s.csv")

    plot_radial_profiles(output_dir / "radial_profiles.png", temperature, moisture)

    visual_radii_cm = np.linspace(0.0, 2.0, 201)
    temperature_visual = model.sample_field(
        temperature.times_s,
        temperature.radii_m,
        temperature.temperature_c,
        temperature.times_s,
        visual_radii_cm,
    )
    moisture_visual = model.sample_moisture_field(
        moisture, moisture.times_s, visual_radii_cm
    )
    plot_heatmap(
        output_dir / "temperature_heatmap.png",
        temperature.times_s,
        visual_radii_cm,
        temperature_visual,
        "30分钟内药材温度的时空变化",
        "温度 / °C",
        "inferno",
    )
    plot_heatmap(
        output_dir / "moisture_heatmap.png",
        moisture.times_s,
        visual_radii_cm,
        moisture_visual,
        "30分钟内药材水分浓度的时空变化",
        "水分浓度 / (kg/kg)",
        "viridis",
    )
    plot_center_surface_response(
        output_dir / "center_surface_evolution.png",
        environment,
        temperature,
        moisture,
    )
    plot_surface_flux(
        output_dir / "surface_heat_moisture_flux.png", fluxes
    )
    print(f"图件目录：{output_dir}")


if __name__ == "__main__":
    main()
