"""A problem, question 1: temperature and moisture fields by finite volumes.

The herb is approximated as a long axisymmetric cylinder.  Under Appendix 2,
the heat and moisture equations share the same geometry, time grid, and
measured oven environment, but the supplied constitutive laws contain no
temperature-moisture cross term.  Therefore this script performs a synchronized
co-simulation without inventing an unprovided latent-heat parameter.

Outputs are restricted to CSV/JSON data and PNG figures as requested.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from openpyxl import load_workbook

import a1_temperature_fvm as heat


RADIUS_M = 0.02
INITIAL_MOISTURE_KG_KG = 2.55
MASS_TRANSFER_COEFFICIENT_M_S = 8.0e-7
END_TIME_S = 1800.0

TABLE_TIMES_S = np.array([100, 300, 600, 900, 1200, 1500, 1800], dtype=float)
TABLE_RADII_CM = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=float)
OUTPUT_RADII_CM = np.round(np.arange(0.0, 2.0 + 0.05, 0.1), 1)


@dataclass
class EnvironmentData:
    times_s: np.ndarray
    temperature_c: np.ndarray
    moisture_kg_kg: np.ndarray


@dataclass
class MoistureResult:
    times_s: np.ndarray
    radii_m: np.ndarray
    moisture_kg_kg: np.ndarray
    surface_moisture_kg_kg: np.ndarray
    volumes_m3_m: np.ndarray
    boundary_outflow_m3_kg_kg_m: float
    maximum_picard_iterations: int


def configure_fonts() -> None:
    """Choose a local CJK-capable font and competition-paper styling."""
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


def find_default_data_path(repo_root: Path) -> Path:
    path = (
        repo_root
        / "比赛题目"
        / "CUMCM2026Problems"
        / "A题"
        / "附件"
        / "附件1.xlsx"
    )
    if not path.exists():
        raise FileNotFoundError(f"Cannot find Attachment 1: {path}")
    return path


def load_environment(path: Path) -> EnvironmentData:
    """Read time, oven temperature, and oven moisture without editing the file."""
    workbook = load_workbook(path, read_only=True, data_only=True)
    sheet = workbook.active
    rows = list(sheet.iter_rows(min_row=2, values_only=True))
    workbook.close()

    data = [row[:3] for row in rows if row[0] is not None]
    times = np.asarray([float(row[0]) for row in data], dtype=float)
    temperatures = np.asarray([float(row[1]) for row in data], dtype=float)
    moistures = np.asarray([float(row[2]) for row in data], dtype=float)

    if not np.all(np.diff(times) > 0.0):
        raise ValueError("Attachment 1 time values must be strictly increasing.")
    if times[0] > 0.0 or times[-1] < END_TIME_S:
        raise ValueError("Attachment 1 does not cover the required 0-1800 s interval.")
    if not (np.all(np.isfinite(temperatures)) and np.all(np.isfinite(moistures))):
        raise ValueError("Attachment 1 contains a non-finite environmental value.")

    return EnvironmentData(times, temperatures, moistures)


def moisture_diffusivity(moisture_kg_kg: np.ndarray) -> np.ndarray:
    """Appendix 2 empirical diffusivity D=7e-9 exp(-0.89/C), in m2/s."""
    moisture = np.asarray(moisture_kg_kg, dtype=float)
    if np.any(moisture <= 0.0):
        raise ValueError("Moisture concentration must remain positive in D(C).")
    return 7.0e-9 * np.exp(-0.89 / moisture)


def factor_tridiagonal(
    lower: np.ndarray, diagonal: np.ndarray, upper: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Factor a tridiagonal matrix for Thomas solves."""
    n = diagonal.size
    denominators = np.empty(n, dtype=float)
    upper_prime = np.empty(n - 1, dtype=float)
    denominators[0] = diagonal[0]
    if denominators[0] <= 0.0:
        raise ValueError("Invalid tridiagonal system.")
    upper_prime[0] = upper[0] / denominators[0]
    for i in range(1, n - 1):
        denominators[i] = diagonal[i] - lower[i - 1] * upper_prime[i - 1]
        if denominators[i] <= 0.0:
            raise ValueError("Invalid tridiagonal system.")
        upper_prime[i] = upper[i] / denominators[i]
    denominators[-1] = diagonal[-1] - lower[-1] * upper_prime[-1]
    if denominators[-1] <= 0.0:
        raise ValueError("Invalid tridiagonal system.")
    return lower.copy(), upper_prime, denominators


def solve_factored_tridiagonal(
    factor: tuple[np.ndarray, np.ndarray, np.ndarray], rhs: np.ndarray
) -> np.ndarray:
    """Solve a factored tridiagonal linear system."""
    lower, upper_prime, denominators = factor
    n = rhs.size
    rhs_prime = np.empty(n, dtype=float)
    rhs_prime[0] = rhs[0] / denominators[0]
    for i in range(1, n):
        rhs_prime[i] = (
            rhs[i] - lower[i - 1] * rhs_prime[i - 1]
        ) / denominators[i]
    solution = np.empty(n, dtype=float)
    solution[-1] = rhs_prime[-1]
    for i in range(n - 2, -1, -1):
        solution[i] = rhs_prime[i] - upper_prime[i] * solution[i + 1]
    return solution


def build_moisture_system(
    radii_m: np.ndarray,
    west_faces_m: np.ndarray,
    east_faces_m: np.ndarray,
    volumes_m3_m: np.ndarray,
    moisture_iterate: np.ndarray,
    time_step_s: float,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], float, float]:
    """Build one Picard-linearized backward-Euler moisture system."""
    n = radii_m.size
    diffusivity_nodes = moisture_diffusivity(moisture_iterate)
    lower = np.zeros(n - 1, dtype=float)
    diagonal = np.ones(n, dtype=float)
    upper = np.zeros(n - 1, dtype=float)

    for i in range(n):
        if i > 0:
            west_resistance = (
                (west_faces_m[i] - radii_m[i - 1]) / diffusivity_nodes[i - 1]
                + (radii_m[i] - west_faces_m[i]) / diffusivity_nodes[i]
            )
            conductance_w = 2.0 * math.pi * west_faces_m[i] / west_resistance
            coefficient_w = conductance_w / volumes_m3_m[i]
            diagonal[i] += time_step_s * coefficient_w
            lower[i - 1] = -time_step_s * coefficient_w
        if i < n - 1:
            east_resistance = (
                (east_faces_m[i] - radii_m[i]) / diffusivity_nodes[i]
                + (radii_m[i + 1] - east_faces_m[i]) / diffusivity_nodes[i + 1]
            )
            conductance_e = 2.0 * math.pi * east_faces_m[i] / east_resistance
            coefficient_e = conductance_e / volumes_m3_m[i]
            diagonal[i] += time_step_s * coefficient_e
            upper[i] = -time_step_s * coefficient_e

    surface_area_per_length = 2.0 * math.pi * RADIUS_M
    # The outer unknown is at the centre of the last control volume.  The
    # half-cell diffusion resistance and air-side convection resistance are
    # placed in series, avoiding a first-order surface-node approximation.
    boundary_conductance = surface_area_per_length / (
        (RADIUS_M - radii_m[-1]) / diffusivity_nodes[-1]
        + 1.0 / MASS_TRANSFER_COEFFICIENT_M_S
    )
    boundary_coefficient = boundary_conductance / volumes_m3_m[-1]
    diagonal[-1] += time_step_s * boundary_coefficient
    return (
        factor_tridiagonal(lower, diagonal, upper),
        boundary_coefficient,
        boundary_conductance,
    )


def simulate_moisture(
    source_times_s: np.ndarray,
    source_moisture_kg_kg: np.ndarray,
    radial_step_cm: float,
    time_step_s: float,
    picard_tolerance: float = 1.0e-10,
    maximum_picard_iterations: int = 50,
) -> MoistureResult:
    """Solve nonlinear radial moisture diffusion with a convective boundary."""
    radial_step_m = radial_step_cm / 100.0
    intervals = int(round(RADIUS_M / radial_step_m))
    steps = int(round(END_TIME_S / time_step_s))
    if not math.isclose(intervals * radial_step_m, RADIUS_M, abs_tol=1e-12):
        raise ValueError("The radial step must divide the 0.02 m radius exactly.")
    if not math.isclose(steps * time_step_s, END_TIME_S, abs_tol=1e-12):
        raise ValueError("The time step must divide 1800 s exactly.")
    record_stride = int(round(1.0 / time_step_s))
    if not math.isclose(record_stride * time_step_s, 1.0, abs_tol=1e-12):
        raise ValueError("The time step must divide the 1 s output interval exactly.")

    # A quadratic face mapping clusters cells near the drying surface, where an
    # initially very thin moisture boundary layer forms.  The interior remains
    # coarser because it is nearly uniform during the first 30 minutes.
    logical_faces = np.linspace(0.0, 1.0, intervals + 1)
    all_faces = RADIUS_M * (1.0 - (1.0 - logical_faces) ** 2)
    west_faces = all_faces[:-1]
    east_faces = all_faces[1:]
    radii = 0.5 * (west_faces + east_faces)
    volumes = math.pi * (east_faces**2 - west_faces**2)

    output_times = np.arange(0.0, END_TIME_S + 1.0, 1.0)
    moisture = np.full(radii.size, INITIAL_MOISTURE_KG_KG, dtype=float)
    records = np.empty((output_times.size, radii.size), dtype=float)
    records[0] = moisture
    surface_records = np.empty(output_times.size, dtype=float)
    surface_records[0] = INITIAL_MOISTURE_KG_KG
    boundary_outflow = 0.0
    surface_area_per_length = 2.0 * math.pi * RADIUS_M
    record_index = 1
    maximum_iterations_used = 0

    for step in range(1, steps + 1):
        time_now = step * time_step_s
        oven_moisture = float(
            np.interp(time_now, source_times_s, source_moisture_kg_kg)
        )
        previous = moisture.copy()
        iterate = previous.copy()

        for iteration in range(1, maximum_picard_iterations + 1):
            factor, boundary_coefficient, boundary_conductance = build_moisture_system(
                radii,
                west_faces,
                east_faces,
                volumes,
                iterate,
                time_step_s,
            )
            rhs = previous.copy()
            rhs[-1] += time_step_s * boundary_coefficient * oven_moisture
            updated = solve_factored_tridiagonal(factor, rhs)
            if np.max(np.abs(updated - iterate)) < picard_tolerance:
                iterate = updated
                break
            iterate = updated
        else:
            raise RuntimeError(f"Moisture Picard iteration failed at t={time_now:g} s.")

        maximum_iterations_used = max(maximum_iterations_used, iteration)
        moisture = iterate
        surface_diffusivity = float(moisture_diffusivity(moisture[-1:])[0])
        surface_distance = RADIUS_M - radii[-1]
        surface_moisture = (
            (surface_diffusivity / surface_distance) * moisture[-1]
            + MASS_TRANSFER_COEFFICIENT_M_S * oven_moisture
        ) / (
            surface_diffusivity / surface_distance
            + MASS_TRANSFER_COEFFICIENT_M_S
        )
        boundary_outflow += (
            time_step_s
            * boundary_conductance
            * (moisture[-1] - oven_moisture)
        )

        if step % record_stride == 0:
            records[record_index] = moisture
            surface_records[record_index] = surface_moisture
            record_index += 1

    return MoistureResult(
        times_s=output_times,
        radii_m=radii,
        moisture_kg_kg=records,
        surface_moisture_kg_kg=surface_records,
        volumes_m3_m=volumes,
        boundary_outflow_m3_kg_kg_m=boundary_outflow,
        maximum_picard_iterations=maximum_iterations_used,
    )


def sample_field(
    times_s: np.ndarray,
    radii_m: np.ndarray,
    field: np.ndarray,
    query_times_s: np.ndarray,
    query_radii_cm: np.ndarray,
) -> np.ndarray:
    """Sample a recorded field at exact integer seconds and requested radii."""
    time_indices = np.rint(query_times_s).astype(int)
    if not np.allclose(times_s[time_indices], query_times_s):
        raise ValueError("Requested times are not available in the 1 s records.")
    sampled = np.empty((query_times_s.size, query_radii_cm.size), dtype=float)
    query_radii_m = query_radii_cm / 100.0
    for j, time_index in enumerate(time_indices):
        sampled[j] = np.interp(query_radii_m, radii_m, field[time_index])
    return sampled


def sample_moisture_field(
    result: MoistureResult,
    query_times_s: np.ndarray,
    query_radii_cm: np.ndarray,
) -> np.ndarray:
    """Sample cell-centred moisture, reconstructing centre and surface values."""
    time_indices = np.rint(query_times_s).astype(int)
    if not np.allclose(result.times_s[time_indices], query_times_s):
        raise ValueError("Requested times are not available in the 1 s records.")
    sampled = np.empty((query_times_s.size, query_radii_cm.size), dtype=float)
    extended_radii = np.concatenate(([0.0], result.radii_m, [RADIUS_M]))
    query_radii_m = query_radii_cm / 100.0
    for j, time_index in enumerate(time_indices):
        nodes = result.moisture_kg_kg[time_index]
        # Even symmetry gives C(r)=a+b*r^2+..., hence this second-order centre
        # reconstruction from values at dr/2 and 3dr/2.
        r0_squared = result.radii_m[0] ** 2
        r1_squared = result.radii_m[1] ** 2
        center_value = (
            nodes[0] * r1_squared - nodes[1] * r0_squared
        ) / (r1_squared - r0_squared)
        extended_values = np.concatenate(
            (
                [center_value],
                nodes,
                [result.surface_moisture_kg_kg[time_index]],
            )
        )
        sampled[j] = np.interp(query_radii_m, extended_radii, extended_values)
    return sampled


def write_field_csv(
    path: Path, times_s: np.ndarray, radii_cm: np.ndarray, values: np.ndarray
) -> None:
    """Write the official 1 s by 0.1 cm rectangular result."""
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_s", *[f"r_{radius:.1f}_cm" for radius in radii_cm]])
        for time_s, row in zip(times_s, values):
            writer.writerow([f"{time_s:.0f}", *[f"{value:.4f}" for value in row]])


def write_table_csv(
    path: Path, times_s: np.ndarray, radii_cm: np.ndarray, values: np.ndarray
) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["时间/s", *[f"{radius:g} cm" for radius in radii_cm]])
        for time_s, row in zip(times_s, values):
            writer.writerow([f"{time_s:.0f}", *[f"{value:.4f}" for value in row]])


def render_table_png(
    path: Path,
    table_number: int,
    title: str,
    unit_label: str,
    values: np.ndarray,
) -> None:
    """Render a clean PNG version of one paper table."""
    configure_fonts()
    figure, axis = plt.subplots(figsize=(11.7, 5.6))
    axis.axis("off")
    axis.set_title(f"表{table_number}  {title}", fontsize=18, fontweight="bold", pad=34)
    axis.text(
        0.5,
        1.01,
        "到药材中心的距离 / cm",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=13,
    )
    cell_text = [[f"{value:.4f}" for value in row] for row in values]
    row_labels = [f"{time_s:.0f}" for time_s in TABLE_TIMES_S]
    column_labels = [f"{radius:g}" for radius in TABLE_RADII_CM]
    table = axis.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=column_labels,
        cellLoc="center",
        rowLoc="center",
        bbox=[0.06, 0.04, 0.91, 0.80],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#7A7A7A")
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor("#D9E4F0")
            cell.set_text_props(fontweight="bold")
        if column == -1:
            cell.set_facecolor("#F3F4F6")
            cell.set_text_props(fontweight="bold")
    axis.text(
        -0.015,
        0.815,
        "时间 / s",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=12,
    )
    axis.text(
        0.97,
        -0.015,
        unit_label,
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#555555",
    )
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def style_axis(axis: plt.Axes) -> None:
    """Apply restrained paper-ready axis styling."""
    axis.grid(True, color="#D9D9D9", linewidth=0.45, alpha=0.75)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def render_heatmap_png(
    path: Path,
    times_s: np.ndarray,
    radii_cm: np.ndarray,
    values: np.ndarray,
    title: str,
    colorbar_label: str,
    cmap: str,
) -> None:
    """Render a continuous time-radius field as a 300-dpi PNG heatmap."""
    configure_fonts()
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
        xlim=(0.0, END_TIME_S / 60.0),
        ylim=(0.0, RADIUS_M * 100.0),
    )
    axis.set_xticks(np.arange(0.0, 31.0, 5.0))
    axis.set_yticks(np.arange(0.0, 2.01, 0.5))
    figure.savefig(path, dpi=300, facecolor="white")
    plt.close(figure)


def render_radial_profiles_png(
    path: Path,
    radii_cm: np.ndarray,
    temperature_profiles: np.ndarray,
    moisture_profiles: np.ndarray,
) -> None:
    """Compare radial profiles at the seven reporting times."""
    configure_fonts()
    figure, axes = plt.subplots(
        1, 2, figsize=(7.09, 3.25), constrained_layout=True
    )
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, TABLE_TIMES_S.size))
    for index, (time_s, color) in enumerate(zip(TABLE_TIMES_S, colors)):
        label = f"{time_s:g} s" if time_s % 60 else f"{time_s / 60:g} min"
        axes[0].plot(
            radii_cm,
            temperature_profiles[index],
            color=color,
            linewidth=1.35,
            label=label,
        )
        axes[1].plot(
            radii_cm,
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
    figure.savefig(path, dpi=300, facecolor="white")
    plt.close(figure)


def render_center_surface_evolution_png(
    path: Path,
    environment: EnvironmentData,
    times_s: np.ndarray,
    temperature_center_surface: np.ndarray,
    moisture_center_surface: np.ndarray,
) -> None:
    """Show boundary forcing and the delayed responses at centre and surface."""
    configure_fonts()
    time_minutes = times_s / 60.0
    oven_temperature = np.interp(
        times_s, environment.times_s, environment.temperature_c
    )
    oven_moisture = np.interp(
        times_s, environment.times_s, environment.moisture_kg_kg
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
    figure.savefig(path, dpi=300, facecolor="white")
    plt.close(figure)


def build_validation(
    environment: EnvironmentData,
    temperature: heat.SimulationResult,
    moisture_coarse: MoistureResult,
    moisture: MoistureResult,
) -> dict:
    coarse_on_output = sample_moisture_field(
        moisture_coarse,
        moisture_coarse.times_s,
        OUTPUT_RADII_CM,
    )
    production_on_output = sample_moisture_field(
        moisture,
        moisture.times_s,
        OUTPUT_RADII_CM,
    )
    refinement_change = np.abs(coarse_on_output - production_on_output)
    coarse_table = sample_moisture_field(
        moisture_coarse,
        TABLE_TIMES_S,
        TABLE_RADII_CM,
    )
    production_table = sample_moisture_field(
        moisture,
        TABLE_TIMES_S,
        TABLE_RADII_CM,
    )
    table_refinement_change = np.abs(coarse_table - production_table)

    stored_change = float(
        np.sum(
            moisture.volumes_m3_m
            * (moisture.moisture_kg_kg[-1] - moisture.moisture_kg_kg[0])
        )
    )
    balance_residual = stored_change + moisture.boundary_outflow_m3_kg_kg_m
    balance_scale = max(abs(stored_change), abs(moisture.boundary_outflow_m3_kg_kg_m), 1e-30)

    radial_violations = int(
        np.count_nonzero(np.diff(moisture.moisture_kg_kg, axis=1) > 1e-10)
        + np.count_nonzero(
            moisture.surface_moisture_kg_kg
            - moisture.moisture_kg_kg[:, -1]
            > 1e-10
        )
    )
    time_violations = int(
        np.count_nonzero(np.diff(moisture.moisture_kg_kg, axis=0) > 1e-10)
        + np.count_nonzero(np.diff(moisture.surface_moisture_kg_kg) > 1e-10)
    )
    below_environment_violations = 0
    for i, time_s in enumerate(moisture.times_s):
        oven_moisture = float(
            np.interp(time_s, environment.times_s, environment.moisture_kg_kg)
        )
        below_environment_violations += int(
            np.count_nonzero(moisture.moisture_kg_kg[i] < oven_moisture - 1e-10)
        )
        below_environment_violations += int(
            moisture.surface_moisture_kg_kg[i] < oven_moisture - 1e-10
        )

    all_moisture_values = np.concatenate(
        (moisture.moisture_kg_kg.ravel(), moisture.surface_moisture_kg_kg)
    )
    diffusivities = moisture_diffusivity(all_moisture_values)
    temperature_table = sample_field(
        temperature.times_s,
        temperature.radii_m,
        temperature.temperature_c,
        TABLE_TIMES_S,
        TABLE_RADII_CM,
    )
    moisture_table = sample_moisture_field(
        moisture,
        TABLE_TIMES_S,
        TABLE_RADII_CM,
    )

    return {
        "model_scope": {
            "geometry": "one-dimensional axisymmetric cylinder",
            "temperature_equation": "transient conduction with convective boundary",
            "moisture_equation": "nonlinear Fick diffusion with convective mass boundary",
            "coupling": (
                "synchronized co-simulation; Appendix 2 supplies no latent-heat or "
                "temperature-dependent moisture term"
            ),
            "production_nominal_radial_step_cm": 0.0125,
            "production_time_step_s": 0.125,
            "coarse_nominal_radial_step_cm": 0.025,
            "coarse_time_step_s": 0.25,
            "moisture_radial_grid": "quadratically clustered toward the surface",
        },
        "moisture_grid_refinement": {
            "coarse_to_production_maximum_change_kg_kg": float(
                np.max(refinement_change)
            ),
            "coarse_to_production_mean_change_kg_kg": float(
                np.mean(refinement_change)
            ),
            "table2_maximum_change_kg_kg": float(
                np.max(table_refinement_change)
            ),
            "table2_mean_change_kg_kg": float(
                np.mean(table_refinement_change)
            ),
        },
        "moisture_balance_per_unit_length": {
            "stored_change_integral": stored_change,
            "integrated_boundary_outflow": moisture.boundary_outflow_m3_kg_kg_m,
            "residual": balance_residual,
            "relative_residual": abs(balance_residual) / balance_scale,
        },
        "moisture_checks": {
            "minimum_kg_kg": float(np.min(all_moisture_values)),
            "maximum_kg_kg": float(np.max(all_moisture_values)),
            "center_1800_s_kg_kg": float(moisture_table[-1, 0]),
            "surface_1800_s_kg_kg": float(moisture_table[-1, -1]),
            "volume_average_1800_s_kg_kg": float(
                np.sum(moisture.volumes_m3_m * moisture.moisture_kg_kg[-1])
                / np.sum(moisture.volumes_m3_m)
            ),
            "minimum_diffusivity_m2_s": float(np.min(diffusivities)),
            "maximum_diffusivity_m2_s": float(np.max(diffusivities)),
            "radial_order_violation_count": radial_violations,
            "time_monotonicity_violation_count": time_violations,
            "below_environment_violation_count": below_environment_violations,
            "maximum_picard_iterations": moisture.maximum_picard_iterations,
        },
        "temperature_selected_outputs": {
            "center_1800_s_C": float(temperature_table[-1, 0]),
            "surface_1800_s_C": float(temperature_table[-1, -1]),
            "oven_1800_s_C": float(
                np.interp(1800.0, environment.times_s, environment.temperature_c)
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="Path to Attachment 1 xlsx")
    parser.add_argument("--repo-root", type=Path, help="Repository root")
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    repo_root = args.repo_root.resolve() if args.repo_root else script_path.parents[1]
    data_path = args.data.resolve() if args.data else find_default_data_path(repo_root)
    result_dir = repo_root / "results" / "A_problem1_coupled"
    picture_dir = repo_root / "picture" / "A_problem1_coupled"
    result_dir.mkdir(parents=True, exist_ok=True)
    picture_dir.mkdir(parents=True, exist_ok=True)

    environment = load_environment(data_path)

    temperature = heat.simulate_temperature(
        environment.times_s,
        environment.temperature_c,
        radial_step_cm=0.025,
        time_step_s=0.25,
    )
    moisture_coarse = simulate_moisture(
        environment.times_s,
        environment.moisture_kg_kg,
        radial_step_cm=0.025,
        time_step_s=0.25,
    )
    moisture = simulate_moisture(
        environment.times_s,
        environment.moisture_kg_kg,
        radial_step_cm=0.0125,
        time_step_s=0.125,
    )

    temperature_full = sample_field(
        temperature.times_s,
        temperature.radii_m,
        temperature.temperature_c,
        temperature.times_s,
        OUTPUT_RADII_CM,
    )
    moisture_full = sample_moisture_field(
        moisture,
        moisture.times_s,
        OUTPUT_RADII_CM,
    )
    temperature_table = sample_field(
        temperature.times_s,
        temperature.radii_m,
        temperature.temperature_c,
        TABLE_TIMES_S,
        TABLE_RADII_CM,
    )
    moisture_table = sample_moisture_field(
        moisture,
        TABLE_TIMES_S,
        TABLE_RADII_CM,
    )

    visual_radii_cm = np.linspace(0.0, 2.0, 201)
    temperature_visual = sample_field(
        temperature.times_s,
        temperature.radii_m,
        temperature.temperature_c,
        temperature.times_s,
        visual_radii_cm,
    )
    moisture_visual = sample_moisture_field(
        moisture,
        moisture.times_s,
        visual_radii_cm,
    )
    temperature_profiles = sample_field(
        temperature.times_s,
        temperature.radii_m,
        temperature.temperature_c,
        TABLE_TIMES_S,
        visual_radii_cm,
    )
    moisture_profiles = sample_moisture_field(
        moisture,
        TABLE_TIMES_S,
        visual_radii_cm,
    )
    temperature_center_surface = sample_field(
        temperature.times_s,
        temperature.radii_m,
        temperature.temperature_c,
        temperature.times_s,
        np.array([0.0, 2.0]),
    )
    moisture_center_surface = sample_moisture_field(
        moisture,
        moisture.times_s,
        np.array([0.0, 2.0]),
    )

    write_field_csv(
        result_dir / "temperature_full_1s_0p1cm.csv",
        temperature.times_s,
        OUTPUT_RADII_CM,
        temperature_full,
    )
    write_field_csv(
        result_dir / "moisture_full_1s_0p1cm.csv",
        moisture.times_s,
        OUTPUT_RADII_CM,
        moisture_full,
    )
    write_table_csv(
        result_dir / "table1_temperature.csv",
        TABLE_TIMES_S,
        TABLE_RADII_CM,
        temperature_table,
    )
    write_table_csv(
        result_dir / "table2_moisture.csv",
        TABLE_TIMES_S,
        TABLE_RADII_CM,
        moisture_table,
    )
    render_table_png(
        picture_dir / "table1_temperature.png",
        1,
        "30分钟内药材的温度",
        "单位：°C",
        temperature_table,
    )
    render_table_png(
        picture_dir / "table2_moisture.png",
        2,
        "30分钟内药材的水分浓度",
        "单位：kg/kg",
        moisture_table,
    )
    render_heatmap_png(
        picture_dir / "temperature_heatmap.png",
        temperature.times_s,
        visual_radii_cm,
        temperature_visual,
        "30分钟内药材温度的时空变化",
        "温度 / °C",
        "inferno",
    )
    render_heatmap_png(
        picture_dir / "moisture_heatmap.png",
        moisture.times_s,
        visual_radii_cm,
        moisture_visual,
        "30分钟内药材水分浓度的时空变化",
        "水分浓度 / (kg/kg)",
        "viridis",
    )
    render_radial_profiles_png(
        picture_dir / "radial_profiles.png",
        visual_radii_cm,
        temperature_profiles,
        moisture_profiles,
    )
    render_center_surface_evolution_png(
        picture_dir / "center_surface_evolution.png",
        environment,
        temperature.times_s,
        temperature_center_surface,
        moisture_center_surface,
    )

    validation = build_validation(
        environment, temperature, moisture_coarse, moisture
    )
    with (result_dir / "validation_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(validation, stream, ensure_ascii=False, indent=2)

    print("Table 1: temperature (deg C)")
    print(np.array2string(temperature_table, precision=4, suppress_small=False))
    print("\nTable 2: moisture concentration (kg/kg)")
    print(np.array2string(moisture_table, precision=4, suppress_small=False))
    print("\nValidation summary")
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    print(f"\nResults: {result_dir}")
    print(f"PNG figures: {picture_dir}")


if __name__ == "__main__":
    main()
