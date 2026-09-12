"""A题问题四：径向收缩圆柱的二维轴对称热湿耦合有限体积模型。

本程序严格延续 ``a4_shrinkage_fvm.py`` 的问题四架构：半径 R(t) 由附件2
分段线性插值，轴向长度保持 0.25 m；径向使用材料坐标 xi=r/R(t)，轴向
计算半域 0<=z<=0.125 m；材料性质完全采用附件4经验公式，温度始终以 K
迭代。模型不引入蒸发潜热、蒸发体积源或额外拟合参数。

r=0、z=0 是对称边界，移动侧壁 r=R(t) 与端面 z=0.125 m 同时采用
原问题四的对流换热和传质边界。空间离散为单元中心二维轴对称有限体积
五点格式，时间离散为后向欧拉，非线性耦合使用 Picard 迭代。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import spsolve

import a3_dimension2 as q3_2d
import a4_shrinkage_fvm as q4


HALF_LENGTH_M = 0.125
FULL_LENGTH_M = 2.0 * HALF_LENGTH_M
OUTPUT_RADII_CM = q4.OUTPUT_RADII_CM.copy()
OUTPUT_Z_M = np.round(np.arange(0.0, HALF_LENGTH_M + 0.0025, 0.005), 3)
WORKBOOK_Z_M = np.array([0.0, 0.025, 0.050, 0.075, 0.100, 0.125])
TABLE_RADII_CM = q4.TABLE_RADII_CM.copy()


@dataclass(frozen=True)
class MaterialGrid2D:
    nr: int
    nz: int
    xi_faces: np.ndarray
    xi_centers: np.ndarray
    z_faces_m: np.ndarray
    z_centers_m: np.ndarray


@dataclass
class StepResult2D:
    temperature_k: np.ndarray
    moisture_kg_kg: np.ndarray
    side_temperature_k: np.ndarray
    end_temperature_k: np.ndarray
    side_moisture_kg_kg: np.ndarray
    end_moisture_kg_kg: np.ndarray
    moisture_outflow_rate_half_m3_s: float
    moisture_storage_change_half_m3: float
    heat_inflow_rate_half_w: float
    heat_storage_change_half_j: float
    minimum_diffusivity_m2_s: float
    maximum_diffusivity_m2_s: float
    iterations: int


@dataclass
class ShrinkageResult2D:
    times_s: np.ndarray
    radii_cm: np.ndarray
    sampled_temperature_k: np.ndarray
    sampled_moisture_kg_kg: np.ndarray
    sampled_surface_temperature_k: np.ndarray
    sampled_surface_moisture_kg_kg: np.ndarray
    drying_time_s: float
    previous_time_s: float
    previous_maximum_moisture: float
    final_maximum_moisture: float
    controlling_radius_m: float
    controlling_z_m: float
    final_temperature_k: np.ndarray
    final_moisture_kg_kg: np.ndarray
    final_side_temperature_k: np.ndarray
    final_end_temperature_k: np.ndarray
    final_side_moisture_kg_kg: np.ndarray
    final_end_moisture_kg_kg: np.ndarray
    material_grid: MaterialGrid2D
    final_grid: q3_2d.Grid2D
    discrete_moisture_storage_change_whole_m3: float
    integrated_moisture_outflow_whole_m3: float
    integrated_heat_storage_whole_j: float
    integrated_heat_inflow_whole_j: float
    minimum_diffusivity_m2_s: float
    maximum_diffusivity_m2_s: float
    maximum_picard_iterations: int
    time_step_counts: dict[str, int]
    elapsed_seconds: float


def make_material_grid(nr: int, nz: int) -> MaterialGrid2D:
    """Quadratically refine the mesh toward the moving side and fixed end."""
    if nr < 2 or nz < 2:
        raise ValueError("nr and nz must both be at least 2.")
    logical_r = np.linspace(0.0, 1.0, nr + 1)
    logical_z = np.linspace(0.0, 1.0, nz + 1)
    xi_faces = 1.0 - (1.0 - logical_r) ** 2
    z_faces = HALF_LENGTH_M * (1.0 - (1.0 - logical_z) ** 2)
    return MaterialGrid2D(
        nr=nr,
        nz=nz,
        xi_faces=xi_faces,
        xi_centers=0.5 * (xi_faces[:-1] + xi_faces[1:]),
        z_faces_m=z_faces,
        z_centers_m=0.5 * (z_faces[:-1] + z_faces[1:]),
    )


def physical_grid(material_grid: MaterialGrid2D, radius_m: float) -> q3_2d.Grid2D:
    """Rebuild physical FVM volumes and face areas at the current radius."""
    r_faces = radius_m * material_grid.xi_faces
    r_centers = radius_m * material_grid.xi_centers
    z_faces = material_grid.z_faces_m
    z_centers = material_grid.z_centers_m
    axial_widths = np.diff(z_faces)
    annular_areas = math.pi * (r_faces[1:] ** 2 - r_faces[:-1] ** 2)
    volumes = axial_widths[:, None] * annular_areas[None, :]
    radial_areas = 2.0 * math.pi * axial_widths[:, None] * r_faces[None, :]
    return q3_2d.Grid2D(
        nr=material_grid.nr,
        nz=material_grid.nz,
        r_faces_m=r_faces,
        r_centers_m=r_centers,
        z_faces_m=z_faces,
        z_centers_m=z_centers,
        cell_volumes_m3=volumes,
        radial_face_areas_m2=radial_areas,
        axial_face_areas_m2=annular_areas,
    )


def boundary_conductances(
    grid: q3_2d.Grid2D,
    coefficient: np.ndarray,
    transfer_coefficient: float,
) -> tuple[np.ndarray, np.ndarray]:
    side = q3_2d.robin_conductance(
        grid.radial_face_areas_m2[:, -1],
        coefficient[:, -1],
        grid.r_faces_m[-1] - grid.r_centers_m[-1],
        transfer_coefficient,
    )
    end = q3_2d.robin_conductance(
        grid.axial_face_areas_m2,
        coefficient[-1, :],
        grid.z_faces_m[-1] - grid.z_centers_m[-1],
        transfer_coefficient,
    )
    return side, end


def advance_coupled_step(
    material_grid: MaterialGrid2D,
    radius_m: float,
    temperature_previous_k: np.ndarray,
    moisture_previous: np.ndarray,
    oven_temperature_k: float,
    oven_moisture: float,
    time_step_s: float,
    temperature_tolerance_k: float = 1.0e-8,
    moisture_tolerance: float = 1.0e-10,
    maximum_iterations: int = 30,
) -> StepResult2D:
    """Advance one fully implicit nonlinear step without a latent-heat term."""
    grid = physical_grid(material_grid, radius_m)
    temperature_iterate = temperature_previous_k.copy()
    moisture_iterate = moisture_previous.copy()
    for iteration in range(1, maximum_iterations + 1):
        conductivity = q4.conductivity_w_m_k(moisture_iterate)
        heat_storage = q4.density_kg_m3(
            moisture_iterate
        ) * q4.heat_capacity_j_kg_k(moisture_iterate)
        heat_matrix, heat_rhs, _, _ = q3_2d.assemble_diffusion_system(
            grid,
            temperature_previous_k,
            conductivity,
            heat_storage,
            q4.HEAT_TRANSFER_COEFFICIENT_W_M2_K,
            oven_temperature_k,
            time_step_s,
        )
        temperature_updated = np.asarray(spsolve(heat_matrix, heat_rhs)).reshape(
            material_grid.nz, material_grid.nr
        )

        diffusivity = q4.moisture_diffusivity_m2_s(
            moisture_iterate, temperature_updated
        )
        moisture_matrix, moisture_rhs, _, _ = q3_2d.assemble_diffusion_system(
            grid,
            moisture_previous,
            diffusivity,
            np.ones_like(moisture_iterate),
            q4.MASS_TRANSFER_COEFFICIENT_M_S,
            oven_moisture,
            time_step_s,
        )
        moisture_updated = np.asarray(
            spsolve(moisture_matrix, moisture_rhs)
        ).reshape(material_grid.nz, material_grid.nr)

        temperature_error = float(
            np.max(np.abs(temperature_updated - temperature_iterate))
        )
        moisture_error = float(
            np.max(np.abs(moisture_updated - moisture_iterate))
        )
        temperature_iterate = temperature_updated
        moisture_iterate = moisture_updated
        if (
            temperature_error < temperature_tolerance_k
            and moisture_error < moisture_tolerance
        ):
            break
    else:
        raise RuntimeError("The 2D question-4 Picard iteration did not converge.")

    final_conductivity = q4.conductivity_w_m_k(moisture_iterate)
    final_heat_storage = q4.density_kg_m3(
        moisture_iterate
    ) * q4.heat_capacity_j_kg_k(moisture_iterate)
    final_diffusivity = q4.moisture_diffusivity_m2_s(
        moisture_iterate, temperature_iterate
    )
    heat_side_g, heat_end_g = boundary_conductances(
        grid, final_conductivity, q4.HEAT_TRANSFER_COEFFICIENT_W_M2_K
    )
    moisture_side_g, moisture_end_g = boundary_conductances(
        grid, final_diffusivity, q4.MASS_TRANSFER_COEFFICIENT_M_S
    )

    side_temperature = q3_2d.reconstruct_robin_surface(
        temperature_iterate[:, -1],
        oven_temperature_k,
        final_conductivity[:, -1],
        q4.HEAT_TRANSFER_COEFFICIENT_W_M2_K,
        grid.r_faces_m[-1] - grid.r_centers_m[-1],
    )
    end_temperature = q3_2d.reconstruct_robin_surface(
        temperature_iterate[-1, :],
        oven_temperature_k,
        final_conductivity[-1, :],
        q4.HEAT_TRANSFER_COEFFICIENT_W_M2_K,
        grid.z_faces_m[-1] - grid.z_centers_m[-1],
    )
    side_moisture = q3_2d.reconstruct_robin_surface(
        moisture_iterate[:, -1],
        oven_moisture,
        final_diffusivity[:, -1],
        q4.MASS_TRANSFER_COEFFICIENT_M_S,
        grid.r_faces_m[-1] - grid.r_centers_m[-1],
    )
    end_moisture = q3_2d.reconstruct_robin_surface(
        moisture_iterate[-1, :],
        oven_moisture,
        final_diffusivity[-1, :],
        q4.MASS_TRANSFER_COEFFICIENT_M_S,
        grid.z_faces_m[-1] - grid.z_centers_m[-1],
    )
    moisture_outflow_rate = float(
        np.sum(moisture_side_g * (moisture_iterate[:, -1] - oven_moisture))
        + np.sum(moisture_end_g * (moisture_iterate[-1, :] - oven_moisture))
    )
    heat_inflow_rate = float(
        np.sum(heat_side_g * (oven_temperature_k - temperature_iterate[:, -1]))
        + np.sum(heat_end_g * (oven_temperature_k - temperature_iterate[-1, :]))
    )
    return StepResult2D(
        temperature_k=temperature_iterate,
        moisture_kg_kg=moisture_iterate,
        side_temperature_k=side_temperature,
        end_temperature_k=end_temperature,
        side_moisture_kg_kg=side_moisture,
        end_moisture_kg_kg=end_moisture,
        moisture_outflow_rate_half_m3_s=moisture_outflow_rate,
        moisture_storage_change_half_m3=float(
            np.sum(grid.cell_volumes_m3 * (moisture_iterate - moisture_previous))
        ),
        heat_inflow_rate_half_w=heat_inflow_rate,
        heat_storage_change_half_j=float(
            np.sum(
                final_heat_storage
                * grid.cell_volumes_m3
                * (temperature_iterate - temperature_previous_k)
            )
        ),
        minimum_diffusivity_m2_s=float(np.min(final_diffusivity)),
        maximum_diffusivity_m2_s=float(np.max(final_diffusivity)),
        iterations=iteration,
    )


def sample_field(
    grid: q3_2d.Grid2D,
    values: np.ndarray,
    side_values: np.ndarray,
    end_values: np.ndarray,
    query_radii_cm: np.ndarray = OUTPUT_RADII_CM,
    query_z_m: np.ndarray = OUTPUT_Z_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample fixed physical radii and the moving surface over requested z."""
    query_r_m = np.asarray(query_radii_cm, dtype=float) / 100.0
    query_z = np.asarray(query_z_m, dtype=float)
    valid_r = query_r_m <= grid.r_faces_m[-1] + 1.0e-12
    radial_samples = np.full((grid.nz, query_r_m.size), np.nan, dtype=float)
    center_r = q3_2d.reconstruct_even_zero(grid.r_centers_m, values, axis=1)
    extended_r = np.concatenate(([0.0], grid.r_centers_m, [grid.r_faces_m[-1]]))
    for j in range(grid.nz):
        extended_values = np.concatenate(
            ([center_r[j]], values[j, :], [side_values[j]])
        )
        radial_samples[j, valid_r] = np.interp(
            np.minimum(query_r_m[valid_r], grid.r_faces_m[-1]),
            extended_r,
            extended_values,
        )

    sampled = np.full((query_z.size, query_r_m.size), np.nan, dtype=float)
    middle_samples = q3_2d.reconstruct_even_zero(
        grid.z_centers_m, radial_samples[:, valid_r], axis=0
    )
    end_center = float(
        q3_2d.reconstruct_even_zero(grid.r_centers_m, end_values, axis=0)
    )
    corner = 0.5 * (float(side_values[-1]) + float(end_values[-1]))
    end_samples = np.interp(
        query_r_m[valid_r],
        extended_r,
        np.concatenate(([end_center], end_values, [corner])),
    )
    extended_z = np.concatenate(([0.0], grid.z_centers_m, [grid.z_faces_m[-1]]))
    valid_indices = np.flatnonzero(valid_r)
    for local_i, global_i in enumerate(valid_indices):
        sampled[:, global_i] = np.interp(
            query_z,
            extended_z,
            np.concatenate(
                (
                    [middle_samples[local_i]],
                    radial_samples[:, global_i],
                    [end_samples[local_i]],
                )
            ),
        )

    side_middle = float(
        q3_2d.reconstruct_even_zero(grid.z_centers_m, side_values, axis=0)
    )
    sampled_surface = np.interp(
        query_z,
        extended_z,
        np.concatenate(([side_middle], side_values, [corner])),
    )
    return sampled, sampled_surface


def choose_time_step(
    time_s: float,
    maximum_moisture: float,
    early_step_s: float,
    late_step_s: float,
    final_step_s: float,
) -> tuple[float, str]:
    if time_s < q4.MEASURED_ENVIRONMENT_END_S:
        return early_step_s, "measured_environment"
    if maximum_moisture < q4.DRYING_THRESHOLD_KG_KG + 0.002:
        return final_step_s, "threshold_refinement"
    return late_step_s, "constant_environment"


def simulate_until_dry(
    boundary: q4.BoundaryProgram,
    radius_program: q4.RadiusProgram,
    nr: int,
    nz: int,
    early_step_s: float,
    late_step_s: float,
    final_step_s: float,
    maximum_time_s: float = q4.MAXIMUM_SIMULATION_TIME_S,
    progress_label: str = "2D-Q4",
) -> ShrinkageResult2D:
    material_grid = make_material_grid(nr, nz)
    temperature = np.full((nz, nr), q4.INITIAL_TEMPERATURE_K, dtype=float)
    moisture = np.full((nz, nr), q4.INITIAL_MOISTURE_KG_KG, dtype=float)
    side_temperature = np.full(nz, q4.INITIAL_TEMPERATURE_K)
    end_temperature = np.full(nr, q4.INITIAL_TEMPERATURE_K)
    side_moisture = np.full(nz, q4.INITIAL_MOISTURE_KG_KG)
    end_moisture = np.full(nr, q4.INITIAL_MOISTURE_KG_KG)

    time_s = 0.0
    next_output_s = q4.OUTPUT_INTERVAL_S
    maximum_moisture = q4.INITIAL_MOISTURE_KG_KG
    controlling_r = 0.0
    controlling_z = 0.0
    previous_time = 0.0
    previous_maximum = maximum_moisture
    forced_final_resolution = False

    record_times: list[float] = []
    record_radii: list[float] = []
    temperature_records: list[np.ndarray] = []
    moisture_records: list[np.ndarray] = []
    surface_temperature_records: list[np.ndarray] = []
    surface_moisture_records: list[np.ndarray] = []
    time_step_counts = {
        "measured_environment": 0,
        "constant_environment": 0,
        "threshold_refinement": 0,
        "crossing_recomputed": 0,
    }
    discrete_moisture_storage_change = 0.0
    integrated_moisture_outflow = 0.0
    integrated_heat_storage = 0.0
    integrated_heat_inflow = 0.0
    minimum_diffusivity = math.inf
    maximum_diffusivity = 0.0
    maximum_iterations = 0
    started = time.perf_counter()
    next_progress_hour = 12.0

    while time_s < maximum_time_s:
        if forced_final_resolution:
            proposed_step, step_label = final_step_s, "threshold_refinement"
        else:
            proposed_step, step_label = choose_time_step(
                time_s,
                maximum_moisture,
                early_step_s,
                late_step_s,
                final_step_s,
            )
        time_step_s = proposed_step
        for event_time in (
            q4.MEASURED_ENVIRONMENT_END_S,
            next_output_s,
            maximum_time_s,
        ):
            if time_s < event_time < time_s + time_step_s:
                time_step_s = event_time - time_s
        if time_step_s <= 0.0:
            raise RuntimeError("A non-positive time step was generated.")

        new_time = time_s + time_step_s
        radius_m = radius_program.radius_m(new_time)
        oven_temperature_k, oven_moisture = boundary.values(new_time)
        step = advance_coupled_step(
            material_grid,
            radius_m,
            temperature,
            moisture,
            oven_temperature_k,
            oven_moisture,
            time_step_s,
        )
        current_grid = physical_grid(material_grid, radius_m)
        new_maximum, new_r, new_z = q3_2d.field_maximum_location(
            current_grid, step.moisture_kg_kg
        )
        if new_maximum < q4.DRYING_THRESHOLD_KG_KG and time_step_s > final_step_s:
            forced_final_resolution = True
            time_step_counts["crossing_recomputed"] += 1
            continue

        # The grid is a half cylinder; multiply storage and boundary rates by two.
        discrete_moisture_storage_change += 2.0 * step.moisture_storage_change_half_m3
        integrated_moisture_outflow += (
            2.0 * time_step_s * step.moisture_outflow_rate_half_m3_s
        )
        integrated_heat_storage += 2.0 * step.heat_storage_change_half_j
        integrated_heat_inflow += 2.0 * time_step_s * step.heat_inflow_rate_half_w

        temperature = step.temperature_k
        moisture = step.moisture_kg_kg
        side_temperature = step.side_temperature_k
        end_temperature = step.end_temperature_k
        side_moisture = step.side_moisture_kg_kg
        end_moisture = step.end_moisture_kg_kg
        time_s = new_time
        maximum_moisture = new_maximum
        controlling_r = new_r
        controlling_z = new_z
        minimum_diffusivity = min(minimum_diffusivity, step.minimum_diffusivity_m2_s)
        maximum_diffusivity = max(maximum_diffusivity, step.maximum_diffusivity_m2_s)
        maximum_iterations = max(maximum_iterations, step.iterations)
        time_step_counts[step_label] += 1

        is_output_time = math.isclose(time_s, next_output_s, abs_tol=1.0e-8)
        if is_output_time:
            sampled_temperature, sampled_surface_temperature = sample_field(
                current_grid, temperature, side_temperature, end_temperature
            )
            sampled_moisture, sampled_surface_moisture = sample_field(
                current_grid, moisture, side_moisture, end_moisture
            )
            record_times.append(time_s)
            record_radii.append(radius_m * 100.0)
            temperature_records.append(sampled_temperature)
            moisture_records.append(sampled_moisture)
            surface_temperature_records.append(sampled_surface_temperature)
            surface_moisture_records.append(sampled_surface_moisture)
            next_output_s += q4.OUTPUT_INTERVAL_S

        if time_s / 3600.0 >= next_progress_hour:
            print(
                f"[{progress_label}] t={time_s / 3600.0:.1f} h, "
                f"R={radius_m * 100.0:.4f} cm, max C={maximum_moisture:.6f}, "
                f"Picard={step.iterations}",
                flush=True,
            )
            next_progress_hour += 12.0

        if maximum_moisture < q4.DRYING_THRESHOLD_KG_KG:
            if not is_output_time:
                sampled_temperature, sampled_surface_temperature = sample_field(
                    current_grid, temperature, side_temperature, end_temperature
                )
                sampled_moisture, sampled_surface_moisture = sample_field(
                    current_grid, moisture, side_moisture, end_moisture
                )
                record_times.append(time_s)
                record_radii.append(radius_m * 100.0)
                temperature_records.append(sampled_temperature)
                moisture_records.append(sampled_moisture)
                surface_temperature_records.append(sampled_surface_temperature)
                surface_moisture_records.append(sampled_surface_moisture)
            break
        previous_time = time_s
        previous_maximum = maximum_moisture
    else:
        raise RuntimeError("Maximum simulation time reached before drying.")

    return ShrinkageResult2D(
        times_s=np.asarray(record_times),
        radii_cm=np.asarray(record_radii),
        sampled_temperature_k=np.stack(temperature_records),
        sampled_moisture_kg_kg=np.stack(moisture_records),
        sampled_surface_temperature_k=np.stack(surface_temperature_records),
        sampled_surface_moisture_kg_kg=np.stack(surface_moisture_records),
        drying_time_s=time_s,
        previous_time_s=previous_time,
        previous_maximum_moisture=previous_maximum,
        final_maximum_moisture=maximum_moisture,
        controlling_radius_m=controlling_r,
        controlling_z_m=controlling_z,
        final_temperature_k=temperature,
        final_moisture_kg_kg=moisture,
        final_side_temperature_k=side_temperature,
        final_end_temperature_k=end_temperature,
        final_side_moisture_kg_kg=side_moisture,
        final_end_moisture_kg_kg=end_moisture,
        material_grid=material_grid,
        final_grid=current_grid,
        discrete_moisture_storage_change_whole_m3=discrete_moisture_storage_change,
        integrated_moisture_outflow_whole_m3=integrated_moisture_outflow,
        integrated_heat_storage_whole_j=integrated_heat_storage,
        integrated_heat_inflow_whole_j=integrated_heat_inflow,
        minimum_diffusivity_m2_s=minimum_diffusivity,
        maximum_diffusivity_m2_s=maximum_diffusivity,
        maximum_picard_iterations=maximum_iterations,
        time_step_counts=time_step_counts,
        elapsed_seconds=time.perf_counter() - started,
    )


def build_table_times(drying_time_s: float) -> np.ndarray:
    regular = np.arange(
        q4.TABLE_INTERVAL_S,
        drying_time_s - 1.0e-9,
        q4.TABLE_INTERVAL_S,
        dtype=float,
    )
    return np.append(regular, drying_time_s)


def record_indices_at_times(
    result: ShrinkageResult2D, query_times_s: np.ndarray
) -> np.ndarray:
    lookup = {round(float(value), 8): index for index, value in enumerate(result.times_s)}
    indices = []
    for value in query_times_s:
        key = round(float(value), 8)
        if key not in lookup:
            raise ValueError(f"Requested time {value:g} s was not recorded.")
        indices.append(lookup[key])
    return np.asarray(indices, dtype=int)


def write_table6_csv(path: Path, result: ShrinkageResult2D, times_s: np.ndarray) -> None:
    indices = record_indices_at_times(result, times_s)
    radial_indices = [
        int(np.where(np.isclose(OUTPUT_RADII_CM, radius))[0][0])
        for radius in TABLE_RADII_CM
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["时间/h（z=0中截面）", *[f"{radius:g} cm" for radius in TABLE_RADII_CM], "药材表面"]
        )
        for time_s, index in zip(times_s, indices):
            label = (
                "烘干结束时间"
                if math.isclose(time_s, result.drying_time_s, abs_tol=1.0e-8)
                else f"{time_s / 3600.0:.0f}"
            )
            values = result.sampled_moisture_kg_kg[index, 0, radial_indices]
            writer.writerow(
                [
                    label,
                    *["" if np.isnan(value) else f"{value:.10f}" for value in values],
                    f"{result.sampled_surface_moisture_kg_kg[index, 0]:.10f}",
                ]
            )


def write_midplane_csv(path: Path, result: ShrinkageResult2D, field: str) -> None:
    if field == "moisture":
        values = result.sampled_moisture_kg_kg[:, 0, :]
        surfaces = result.sampled_surface_moisture_kg_kg[:, 0]
        digits = 10
    else:
        values = result.sampled_temperature_k[:, 0, :]
        surfaces = result.sampled_surface_temperature_k[:, 0]
        digits = 6
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["time_s", "radius_cm", *[f"r_{r:.1f}_cm" for r in OUTPUT_RADII_CM], "surface"]
        )
        for time_s, radius_cm, row, surface in zip(
            result.times_s, result.radii_cm, values, surfaces
        ):
            writer.writerow(
                [
                    f"{time_s:.0f}",
                    f"{radius_cm:.8f}",
                    *["" if np.isnan(value) else f"{value:.{digits}f}" for value in row],
                    f"{surface:.{digits}f}",
                ]
            )


def write_2d_long_csv(path: Path, result: ShrinkageResult2D, field: str) -> None:
    if field == "moisture":
        values = result.sampled_moisture_kg_kg
        surfaces = result.sampled_surface_moisture_kg_kg
        digits = 10
    else:
        values = result.sampled_temperature_k
        surfaces = result.sampled_surface_temperature_k
        digits = 6
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "time_s", "time_h", "radius_cm", "z_m",
                *[f"r_{r:.1f}_cm" for r in OUTPUT_RADII_CM], "surface",
            ]
        )
        for time_s, radius_cm, field_at_time, surface_at_time in zip(
            result.times_s, result.radii_cm, values, surfaces
        ):
            for z_m, row, surface in zip(OUTPUT_Z_M, field_at_time, surface_at_time):
                writer.writerow(
                    [
                        f"{time_s:.0f}",
                        f"{time_s / 3600.0:.8f}",
                        f"{radius_cm:.8f}",
                        f"{z_m:.3f}",
                        *["" if np.isnan(value) else f"{value:.{digits}f}" for value in row],
                        f"{surface:.{digits}f}",
                    ]
                )


def write_final_cell_field(path: Path, result: ShrinkageResult2D) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["xi_center", "z_center_m", "r_center_m", "temperature_K", "moisture_kg_kg"]
        )
        for j, z_m in enumerate(result.final_grid.z_centers_m):
            for i, (xi, r_m) in enumerate(
                zip(result.material_grid.xi_centers, result.final_grid.r_centers_m)
            ):
                writer.writerow(
                    [
                        f"{xi:.10f}", f"{z_m:.10f}", f"{r_m:.10f}",
                        f"{result.final_temperature_k[j, i]:.8f}",
                        f"{result.final_moisture_kg_kg[j, i]:.10f}",
                    ]
                )


def write_inputs_csv(
    path: Path,
    boundary: q4.BoundaryProgram,
    radius_program: q4.RadiusProgram,
    end_time_s: float,
) -> None:
    times = np.arange(0.0, math.floor(end_time_s / 60.0) * 60.0 + 1.0, 60.0)
    if not math.isclose(times[-1], end_time_s, abs_tol=1.0e-8):
        times = np.append(times, end_time_s)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["time_s", "time_h", "radius_cm", "oven_temperature_K", "oven_moisture_kg_kg"]
        )
        for time_s in times:
            temperature_k, moisture = boundary.values(float(time_s))
            writer.writerow(
                [
                    f"{time_s:.0f}", f"{time_s / 3600.0:.8f}",
                    f"{radius_program.radius_cm(float(time_s)):.8f}",
                    f"{temperature_k:.8f}", f"{moisture:.10f}",
                ]
            )


def common_refinement_difference(
    production: ShrinkageResult2D, coarse: ShrinkageResult2D
) -> tuple[int, float, float]:
    coarse_lookup = {round(float(t), 8): i for i, t in enumerate(coarse.times_s)}
    differences: list[np.ndarray] = []
    common_rows = 0
    for index, time_s in enumerate(production.times_s):
        key = round(float(time_s), 8)
        if key not in coarse_lookup:
            continue
        coarse_index = coarse_lookup[key]
        a = production.sampled_moisture_kg_kg[index]
        b = coarse.sampled_moisture_kg_kg[coarse_index]
        mask = np.isfinite(a) & np.isfinite(b)
        if np.any(mask):
            differences.append(np.abs(a[mask] - b[mask]))
            differences.append(
                np.abs(
                    production.sampled_surface_moisture_kg_kg[index]
                    - coarse.sampled_surface_moisture_kg_kg[coarse_index]
                )
            )
            common_rows += 1
    combined = np.concatenate(differences)
    return common_rows, float(np.max(combined)), float(np.mean(combined))


def build_validation(
    boundary: q4.BoundaryProgram,
    radius_program: q4.RadiusProgram,
    production: ShrinkageResult2D,
    coarse: ShrinkageResult2D,
    matched_one_dimensional: q4.ShrinkageResult,
    original_validation_path: Path,
    time_steps_s: list[float],
) -> dict:
    common_rows, refinement_maximum, refinement_mean = common_refinement_difference(
        production, coarse
    )
    moisture_residual = (
        production.discrete_moisture_storage_change_whole_m3
        + production.integrated_moisture_outflow_whole_m3
    )
    moisture_scale = max(
        abs(production.discrete_moisture_storage_change_whole_m3),
        abs(production.integrated_moisture_outflow_whole_m3),
        1.0e-30,
    )
    heat_residual = (
        production.integrated_heat_storage_whole_j
        - production.integrated_heat_inflow_whole_j
    )
    heat_scale = max(
        abs(production.integrated_heat_storage_whole_j),
        abs(production.integrated_heat_inflow_whole_j),
        1.0e-30,
    )
    original_time = None
    if original_validation_path.exists():
        try:
            with original_validation_path.open("r", encoding="utf-8") as stream:
                original_time = float(json.load(stream)["drying_time"]["production_s"])
        except (KeyError, ValueError, TypeError, json.JSONDecodeError):
            original_time = None

    moisture_values = production.sampled_moisture_kg_kg
    temperature_values = production.sampled_temperature_k
    invalid_outside = 0
    missing_inside = 0
    for row_index, radius_cm in enumerate(production.radii_cm):
        inside = OUTPUT_RADII_CM <= radius_cm + 1.0e-10
        invalid_outside += int(
            np.count_nonzero(np.isfinite(moisture_values[row_index, :, ~inside]))
        )
        missing_inside += int(
            np.count_nonzero(~np.isfinite(moisture_values[row_index, :, inside]))
        )
    finite_moisture = moisture_values[np.isfinite(moisture_values)]
    finite_temperature = temperature_values[np.isfinite(temperature_values)]
    source_radius_error = float(
        np.max(
            np.abs(
                np.asarray(
                    [radius_program.radius_cm(t) for t in radius_program.source_times_s]
                )
                - radius_program.source_radii_cm
            )
        )
    )
    production_dr = q4.INITIAL_RADIUS_M / production.material_grid.nr
    production_dz = HALF_LENGTH_M / production.material_grid.nz
    return {
        "model_scope": {
            "geometry": "2D axisymmetric cylinder with moving radius and fixed length",
            "domain": "0<=xi=r/R(t)<=1; 0<=z<=0.125 m",
            "radial_shrinkage": "Attachment 2 piecewise-linear R(t), identical to main Q4",
            "axial_shrinkage": "excluded, identical to main Q4 assumption",
            "constitutive_laws": "Appendix 4 rho(C), cp(C), k(C), D(C,T) without modification",
            "temperature_state_unit": "K",
            "latent_heat": "excluded",
            "evaporation_source": "excluded",
            "boundary_conditions": "symmetry at xi=0,z=0; original convection on side and end",
            "drying_threshold": "strict reconstructed max over full 2D product <0.15 kg/kg",
            "environment_after_4h": "time-weighted mean over Attachment 1 from 3 h to 4 h",
            "solver": "cell-centred FVM, backward Euler, sparse five-point matrix, Picard coupling",
        },
        "grid_and_time": {
            "production": {
                "nr": production.material_grid.nr,
                "nz": production.material_grid.nz,
                "nominal_initial_dr_m": production_dr,
                "nominal_dz_m": production_dz,
                "elapsed_s": production.elapsed_seconds,
            },
            "coarse": {
                "nr": coarse.material_grid.nr,
                "nz": coarse.material_grid.nz,
                "nominal_initial_dr_m": q4.INITIAL_RADIUS_M / coarse.material_grid.nr,
                "nominal_dz_m": HALF_LENGTH_M / coarse.material_grid.nz,
                "elapsed_s": coarse.elapsed_seconds,
            },
            "time_steps_s": time_steps_s,
            "saved_interval_s": q4.OUTPUT_INTERVAL_S,
            "saved_radii_cm": OUTPUT_RADII_CM.tolist(),
            "saved_z_m": OUTPUT_Z_M.tolist(),
            "workbook_z_m": WORKBOOK_Z_M.tolist(),
        },
        "constant_boundary_after_4h": {
            "temperature_K": boundary.plateau_temperature_k,
            "temperature_C_for_reference": boundary.plateau_temperature_k - 273.15,
            "moisture_kg_kg": boundary.plateau_moisture_kg_kg,
        },
        "radius": {
            "attachment_interval_s": [
                float(radius_program.source_times_s[0]),
                float(radius_program.source_times_s[-1]),
            ],
            "initial_cm": float(radius_program.source_radii_cm[0]),
            "drying_time_cm": float(production.radii_cm[-1]),
            "source_node_reconstruction_max_error_cm": source_radius_error,
            "recorded_nonincreasing": bool(np.all(np.diff(production.radii_cm) <= 1.0e-12)),
        },
        "drying_time": {
            "production_s": production.drying_time_s,
            "production_h": production.drying_time_s / 3600.0,
            "coarse_s": coarse.drying_time_s,
            "coarse_h": coarse.drying_time_s / 3600.0,
            "coarse_production_difference_s": abs(production.drying_time_s - coarse.drying_time_s),
            "previous_time_s": production.previous_time_s,
            "previous_maximum_moisture": production.previous_maximum_moisture,
            "final_maximum_moisture": production.final_maximum_moisture,
            "controlling_radius_m": production.controlling_radius_m,
            "controlling_z_m": production.controlling_z_m,
        },
        "one_dimensional_comparison": {
            "original_main_model_s": original_time,
            "matched_grid_time_step_s": matched_one_dimensional.drying_time_s,
            "two_dimensional_minus_matched_one_dimensional_s": production.drying_time_s - matched_one_dimensional.drying_time_s,
            "matched_initial_radial_cells": production.material_grid.nr,
            "interpretation": "Negative means the added end convection accelerates drying on matched radial/time discretization.",
        },
        "grid_refinement": {
            "common_record_count": common_rows,
            "maximum_sampled_moisture_change_kg_kg": refinement_maximum,
            "mean_sampled_moisture_change_kg_kg": refinement_mean,
        },
        "whole_product_balances": {
            "discrete_moisture_storage_change": production.discrete_moisture_storage_change_whole_m3,
            "integrated_moisture_outflow": production.integrated_moisture_outflow_whole_m3,
            "moisture_residual": moisture_residual,
            "moisture_relative_residual": abs(moisture_residual) / moisture_scale,
            "integrated_heat_storage_J": production.integrated_heat_storage_whole_j,
            "integrated_heat_inflow_J": production.integrated_heat_inflow_whole_j,
            "heat_residual_J": heat_residual,
            "heat_relative_residual": abs(heat_residual) / heat_scale,
        },
        "physical_ranges": {
            "temperature_min_K": float(np.min(finite_temperature)),
            "temperature_max_K": float(np.max(finite_temperature)),
            "moisture_min_kg_kg": float(np.min(finite_moisture)),
            "moisture_max_kg_kg": float(np.max(finite_moisture)),
            "diffusivity_min_m2_s": production.minimum_diffusivity_m2_s,
            "diffusivity_max_m2_s": production.maximum_diffusivity_m2_s,
        },
        "checks": {
            "previous_state_not_dry": bool(production.previous_maximum_moisture >= q4.DRYING_THRESHOLD_KG_KG),
            "final_state_all_dry": bool(production.final_maximum_moisture < q4.DRYING_THRESHOLD_KG_KG),
            "invalid_values_outside_current_radius": invalid_outside,
            "missing_values_inside_current_radius": missing_inside,
            "radial_order_violation_count": int(np.count_nonzero(np.diff(moisture_values, axis=2) > 1.0e-8)),
            "axial_order_violation_count": int(np.count_nonzero(np.diff(moisture_values, axis=1) > 1.0e-8)),
            "time_monotonicity_violation_count": int(np.count_nonzero(np.diff(moisture_values, axis=0) > 1.0e-8)),
            "maximum_picard_iterations": production.maximum_picard_iterations,
            "time_step_counts": production.time_step_counts,
        },
    }


def run_workbook_builder(repo_root: Path, result_dir: Path) -> None:
    node = Path(
        r"C:\Users\user\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe"
    )
    builder = repo_root / "code" / "build_a4_dimension2_workbook.cjs"
    command = [
        str(node), str(builder),
        str(result_dir / "table6_midplane_moisture.csv"),
        str(result_dir / "midplane_moisture_60s_r0p1cm.csv"),
        str(result_dir / "moisture_2d_60s_r0p1cm_z0p5cm.csv"),
        str(result_dir / "temperature_2d_60s_r0p1cm_z0p5cm_K.csv"),
        str(result_dir / "validation_summary.json"),
        str(result_dir / "result4_dimension2.xlsx"),
        str(result_dir / "workbook_preview"),
    ]
    environment = os.environ.copy()
    environment["NODE_PATH"] = str(
        Path(
            r"C:\Users\user\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\node_modules"
        )
    )
    subprocess.run(command, check=True, env=environment)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attachment1", type=Path)
    parser.add_argument("--attachment2", type=Path)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--nr", type=int, default=64)
    parser.add_argument("--nz", type=int, default=40)
    parser.add_argument("--coarse-nr", type=int, default=32)
    parser.add_argument("--coarse-nz", type=int, default=20)
    parser.add_argument("--early-dt", type=float, default=1.0)
    parser.add_argument("--late-dt", type=float, default=30.0)
    parser.add_argument("--final-dt", type=float, default=1.0)
    parser.add_argument("--maximum-time", type=float, default=q4.MAXIMUM_SIMULATION_TIME_S)
    parser.add_argument("--skip-coarse", action="store_true")
    parser.add_argument("--skip-xlsx", action="store_true")
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    repo_root = args.repo_root.resolve() if args.repo_root else script_path.parents[1]
    attachment1 = args.attachment1.resolve() if args.attachment1 else q4.find_attachment(repo_root, "附件1.xlsx")
    attachment2 = args.attachment2.resolve() if args.attachment2 else q4.find_attachment(repo_root, "附件2.xlsx")
    result_dir = args.result_dir.resolve() if args.result_dir else repo_root / "results" / "A_problem4_dimension2"
    result_dir.mkdir(parents=True, exist_ok=True)

    environment = q4.q2.load_and_preprocess_environment(attachment1)
    boundary = q4.build_boundary_program(environment)
    radius_program = q4.load_radius_program(attachment2)
    production = simulate_until_dry(
        boundary, radius_program, args.nr, args.nz,
        args.early_dt, args.late_dt, args.final_dt,
        maximum_time_s=args.maximum_time,
        progress_label=f"production {args.nr}x{args.nz}",
    )
    coarse = production if args.skip_coarse else simulate_until_dry(
        boundary, radius_program, args.coarse_nr, args.coarse_nz,
        args.early_dt, args.late_dt, args.final_dt,
        maximum_time_s=args.maximum_time,
        progress_label=f"coarse {args.coarse_nr}x{args.coarse_nz}",
    )
    matched_one_dimensional = q4.simulate_until_dry(
        boundary,
        radius_program,
        nominal_initial_step_cm=2.0 / args.nr,
        early_step_s=args.early_dt,
        late_step_s=args.late_dt,
        final_step_s=args.final_dt,
    )

    table_times = build_table_times(production.drying_time_s)
    write_table6_csv(result_dir / "table6_midplane_moisture.csv", production, table_times)
    write_midplane_csv(result_dir / "midplane_moisture_60s_r0p1cm.csv", production, "moisture")
    write_midplane_csv(result_dir / "midplane_temperature_60s_r0p1cm_K.csv", production, "temperature")
    write_2d_long_csv(result_dir / "moisture_2d_60s_r0p1cm_z0p5cm.csv", production, "moisture")
    write_2d_long_csv(result_dir / "temperature_2d_60s_r0p1cm_z0p5cm_K.csv", production, "temperature")
    write_final_cell_field(result_dir / "final_cell_field_2d.csv", production)
    write_inputs_csv(result_dir / "environment_radius_60s.csv", boundary, radius_program, production.drying_time_s)

    validation = build_validation(
        boundary,
        radius_program,
        production,
        coarse,
        matched_one_dimensional,
        repo_root / "results" / "A_problem4_shrinkage" / "validation_summary.json",
        [args.early_dt, args.late_dt, args.final_dt],
    )
    with (result_dir / "validation_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(validation, stream, ensure_ascii=False, indent=2)
    if not args.skip_xlsx:
        run_workbook_builder(repo_root, result_dir)

    print(
        f"2D Q4 drying time: {production.drying_time_s:.0f} s = "
        f"{production.drying_time_s / 3600.0:.6f} h"
    )
    print(
        f"Threshold bracket: {production.previous_time_s:.0f} s "
        f"({production.previous_maximum_moisture:.10f}) -> "
        f"{production.drying_time_s:.0f} s ({production.final_maximum_moisture:.10f})"
    )
    print(
        f"Controlling point: r={production.controlling_radius_m:.8f} m, "
        f"z={production.controlling_z_m:.8f} m; "
        f"R(t_end)={production.radii_cm[-1]:.6f} cm"
    )
    print("\nTable 6-compatible midplane output")
    with (result_dir / "table6_midplane_moisture.csv").open("r", encoding="utf-8-sig") as stream:
        print(stream.read())
    print(f"Results: {result_dir}")


if __name__ == "__main__":
    main()
