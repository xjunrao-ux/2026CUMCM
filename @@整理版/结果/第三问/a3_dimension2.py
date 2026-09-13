"""A题问题三：固定几何圆柱的二维轴对称热湿耦合有限体积模型。

计算域取圆柱的一半：0 <= r <= 0.02 m，0 <= z <= 0.125 m。
r=0 与 z=0 为对称边界；r=R 的侧壁和 z=L/2 的端面均采用题设对流
换热/传质边界。模型只把原问题三的一维模型推广到二维，不引入收缩和
蒸发潜热。温度场以摄氏度存储，但水分扩散系数中的温度始终转换为 K。

离散采用单元中心有限体积法、全隐式后向欧拉和 Picard 热湿耦合迭代；
每个场形成二维五点稀疏线性方程组，由 scipy.sparse.linalg.spsolve 求解。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

import a2_coupled_fvm as q2
import a3_drying_time_fvm as q3_1d


RADIUS_M = q2.RADIUS_M
HALF_LENGTH_M = 0.125
FULL_LENGTH_M = 2.0 * HALF_LENGTH_M
DRYING_THRESHOLD_KG_KG = q3_1d.DRYING_THRESHOLD_KG_KG
MEASURED_END_TIME_S = q3_1d.MEASURED_END_TIME_S
OUTPUT_INTERVAL_S = 60.0
TABLE_INTERVAL_S = 6.0 * 3600.0
MAXIMUM_SIMULATION_TIME_S = 7.0 * 24.0 * 3600.0

OUTPUT_RADII_CM = np.round(np.arange(0.0, 2.0 + 0.05, 0.1), 1)
OUTPUT_Z_M = np.round(np.arange(0.0, HALF_LENGTH_M + 0.0025, 0.005), 3)
TABLE_RADII_CM = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=float)
WORKBOOK_Z_M = np.array([0.0, 0.025, 0.050, 0.075, 0.100, 0.125])


@dataclass(frozen=True)
class Grid2D:
    nr: int
    nz: int
    r_faces_m: np.ndarray
    r_centers_m: np.ndarray
    z_faces_m: np.ndarray
    z_centers_m: np.ndarray
    cell_volumes_m3: np.ndarray
    radial_face_areas_m2: np.ndarray
    axial_face_areas_m2: np.ndarray

    @property
    def dr_m(self) -> float:
        return float(RADIUS_M / self.nr)

    @property
    def dz_m(self) -> float:
        return float(HALF_LENGTH_M / self.nz)

    @property
    def radial_cell_width_range_m(self) -> tuple[float, float]:
        widths = np.diff(self.r_faces_m)
        return float(np.min(widths)), float(np.max(widths))

    @property
    def axial_cell_width_range_m(self) -> tuple[float, float]:
        widths = np.diff(self.z_faces_m)
        return float(np.min(widths)), float(np.max(widths))


@dataclass
class StepResult2D:
    temperature_c: np.ndarray
    moisture_kg_kg: np.ndarray
    side_temperature_c: np.ndarray
    end_temperature_c: np.ndarray
    side_moisture_kg_kg: np.ndarray
    end_moisture_kg_kg: np.ndarray
    moisture_outflow_rate_half_m3_s: float
    heat_inflow_rate_half_w: float
    heat_storage_change_half_j: float
    iterations: int
    minimum_diffusivity_m2_s: float
    maximum_diffusivity_m2_s: float


@dataclass
class DryingResult2D:
    times_s: np.ndarray
    sampled_temperature_c: np.ndarray
    sampled_moisture_kg_kg: np.ndarray
    drying_time_s: float
    last_not_dry_time_s: float
    last_not_dry_maximum_moisture: float
    final_maximum_moisture: float
    controlling_radius_m: float
    controlling_z_m: float
    final_temperature_c: np.ndarray
    final_moisture_kg_kg: np.ndarray
    final_side_temperature_c: np.ndarray
    final_end_temperature_c: np.ndarray
    final_side_moisture_kg_kg: np.ndarray
    final_end_moisture_kg_kg: np.ndarray
    grid: Grid2D
    integrated_moisture_outflow_whole_m3: float
    integrated_heat_inflow_whole_j: float
    integrated_heat_storage_whole_j: float
    maximum_picard_iterations: int
    time_step_counts: dict[str, int]
    elapsed_seconds: float
    minimum_diffusivity_m2_s: float
    maximum_diffusivity_m2_s: float


def make_grid(nr: int, nz: int) -> Grid2D:
    """Create a boundary-refined cell-centred mesh on the half cylinder.

    The quadratic face mapping is identical to the radial mapping of the
    accepted one-dimensional Q3 code.  It concentrates cells next to the
    exposed side/end boundary layers without wasting cells at symmetry planes.
    """
    if nr < 2 or nz < 2:
        raise ValueError("nr and nz must both be at least 2.")
    logical_r = np.linspace(0.0, 1.0, nr + 1)
    logical_z = np.linspace(0.0, 1.0, nz + 1)
    r_faces = RADIUS_M * (1.0 - (1.0 - logical_r) ** 2)
    z_faces = HALF_LENGTH_M * (1.0 - (1.0 - logical_z) ** 2)
    r_centers = 0.5 * (r_faces[:-1] + r_faces[1:])
    z_centers = 0.5 * (z_faces[:-1] + z_faces[1:])
    annular_areas = math.pi * (r_faces[1:] ** 2 - r_faces[:-1] ** 2)
    cell_volumes = np.repeat(
        (annular_areas * (z_faces[1] - z_faces[0]))[None, :], nz, axis=0
    )
    radial_face_areas = 2.0 * math.pi * r_faces * (z_faces[1] - z_faces[0])
    axial_face_areas = annular_areas
    return Grid2D(
        nr=nr,
        nz=nz,
        r_faces_m=r_faces,
        r_centers_m=r_centers,
        z_faces_m=z_faces,
        z_centers_m=z_centers,
        cell_volumes_m3=cell_volumes,
        radial_face_areas_m2=radial_face_areas,
        axial_face_areas_m2=axial_face_areas,
    )


def moisture_diffusivity_kelvin(
    moisture_kg_kg: np.ndarray, temperature_c: np.ndarray
) -> np.ndarray:
    """Appendix 3 diffusivity; the Arrhenius temperature is explicitly Kelvin."""
    moisture = np.asarray(moisture_kg_kg, dtype=float)
    temperature_k = np.asarray(temperature_c, dtype=float) + 273.15
    if np.any(moisture <= 0.0) or np.any(temperature_k <= 0.0):
        raise ValueError("Moisture and absolute temperature must be positive.")
    return 2.4e-3 * np.exp(-0.45 / moisture) * np.exp(-3850.0 / temperature_k)


def robin_conductance(
    area_m2: np.ndarray,
    cell_coefficient: np.ndarray,
    half_cell_distance_m: float,
    transfer_coefficient: float,
) -> np.ndarray:
    """Series resistance of a half cell and the external convection film."""
    return area_m2 / (
        half_cell_distance_m / cell_coefficient + 1.0 / transfer_coefficient
    )


def assemble_diffusion_system(
    grid: Grid2D,
    previous: np.ndarray,
    coefficient: np.ndarray,
    storage_coefficient: np.ndarray,
    transfer_coefficient: float,
    ambient_value: float,
    time_step_s: float,
):
    """Assemble the backward-Euler axisymmetric five-point FVM matrix."""
    if previous.shape != (grid.nz, grid.nr):
        raise ValueError("Field shape does not match the two-dimensional grid.")
    n = grid.nr * grid.nz
    flat_index = np.arange(n, dtype=np.int64).reshape(grid.nz, grid.nr)
    diagonal = (
        storage_coefficient * grid.cell_volumes_m3 / time_step_s
    ).ravel().copy()
    rhs = diagonal * previous.ravel()
    rows: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    data: list[np.ndarray] = []

    # Internal radial faces.  The face area already contains the cylindrical 2*pi*r factor.
    radial_resistance = (
        (grid.r_faces_m[1:-1][None, :] - grid.r_centers_m[:-1][None, :])
        / coefficient[:, :-1]
        + (grid.r_centers_m[1:][None, :] - grid.r_faces_m[1:-1][None, :])
        / coefficient[:, 1:]
    )
    radial_g = grid.radial_face_areas_m2[1:-1][None, :] / radial_resistance
    west = flat_index[:, :-1].ravel()
    east = flat_index[:, 1:].ravel()
    radial_flat = radial_g.ravel()
    np.add.at(diagonal, west, radial_flat)
    np.add.at(diagonal, east, radial_flat)
    rows.extend([west, east])
    columns.extend([east, west])
    data.extend([-radial_flat, -radial_flat])

    # Internal axial faces.
    axial_resistance = (
        (grid.z_faces_m[1:-1, None] - grid.z_centers_m[:-1, None])
        / coefficient[:-1, :]
        + (grid.z_centers_m[1:, None] - grid.z_faces_m[1:-1, None])
        / coefficient[1:, :]
    )
    axial_g = grid.axial_face_areas_m2[None, :] / axial_resistance
    south = flat_index[:-1, :].ravel()
    north = flat_index[1:, :].ravel()
    axial_flat = axial_g.ravel()
    np.add.at(diagonal, south, axial_flat)
    np.add.at(diagonal, north, axial_flat)
    rows.extend([south, north])
    columns.extend([north, south])
    data.extend([-axial_flat, -axial_flat])

    # Exposed side r=R.
    side_g = robin_conductance(
        np.full(grid.nz, grid.radial_face_areas_m2[-1]),
        coefficient[:, -1],
        RADIUS_M - grid.r_centers_m[-1],
        transfer_coefficient,
    )
    side_cells = flat_index[:, -1]
    np.add.at(diagonal, side_cells, side_g)
    np.add.at(rhs, side_cells, side_g * ambient_value)

    # Exposed end z=L/2.  Doubling the half-domain later represents both physical ends.
    end_g = robin_conductance(
        grid.axial_face_areas_m2,
        coefficient[-1, :],
        HALF_LENGTH_M - grid.z_centers_m[-1],
        transfer_coefficient,
    )
    end_cells = flat_index[-1, :]
    np.add.at(diagonal, end_cells, end_g)
    np.add.at(rhs, end_cells, end_g * ambient_value)

    diagonal_index = np.arange(n, dtype=np.int64)
    rows.append(diagonal_index)
    columns.append(diagonal_index)
    data.append(diagonal)
    matrix = coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(columns))),
        shape=(n, n),
    ).tocsr()
    return matrix, rhs, side_g, end_g


def reconstruct_robin_surface(
    cell_value: np.ndarray,
    ambient_value: float,
    cell_coefficient: np.ndarray,
    transfer_coefficient: float,
    half_cell_distance_m: float,
) -> np.ndarray:
    cell_side = cell_coefficient / half_cell_distance_m
    return (
        cell_side * cell_value + transfer_coefficient * ambient_value
    ) / (cell_side + transfer_coefficient)


def advance_coupled_step(
    grid: Grid2D,
    temperature_previous_c: np.ndarray,
    moisture_previous_kg_kg: np.ndarray,
    oven_temperature_c: float,
    oven_moisture_kg_kg: float,
    time_step_s: float,
    temperature_tolerance_c: float = 1.0e-8,
    moisture_tolerance: float = 1.0e-10,
    maximum_iterations: int = 30,
) -> StepResult2D:
    """Advance one nonlinear time level with sequential Picard coupling."""
    temperature_iterate = temperature_previous_c.copy()
    moisture_iterate = moisture_previous_kg_kg.copy()
    for iteration in range(1, maximum_iterations + 1):
        conductivity = q2.conductivity_w_m_k(moisture_iterate)
        volumetric_heat_capacity = q2.density_kg_m3(
            moisture_iterate
        ) * q2.heat_capacity_j_kg_k(moisture_iterate)
        heat_matrix, heat_rhs, _, _ = assemble_diffusion_system(
            grid,
            temperature_previous_c,
            conductivity,
            volumetric_heat_capacity,
            q2.HEAT_TRANSFER_COEFFICIENT_W_M2_K,
            oven_temperature_c,
            time_step_s,
        )
        temperature_updated = np.asarray(spsolve(heat_matrix, heat_rhs)).reshape(
            grid.nz, grid.nr
        )

        diffusivity = moisture_diffusivity_kelvin(
            moisture_iterate, temperature_updated
        )
        moisture_matrix, moisture_rhs, _, _ = assemble_diffusion_system(
            grid,
            moisture_previous_kg_kg,
            diffusivity,
            np.ones_like(moisture_iterate),
            q2.MASS_TRANSFER_COEFFICIENT_M_S,
            oven_moisture_kg_kg,
            time_step_s,
        )
        moisture_updated = np.asarray(
            spsolve(moisture_matrix, moisture_rhs)
        ).reshape(grid.nz, grid.nr)

        temperature_error = float(
            np.max(np.abs(temperature_updated - temperature_iterate))
        )
        moisture_error = float(
            np.max(np.abs(moisture_updated - moisture_iterate))
        )
        temperature_iterate = temperature_updated
        moisture_iterate = moisture_updated
        if (
            temperature_error < temperature_tolerance_c
            and moisture_error < moisture_tolerance
        ):
            break
    else:
        raise RuntimeError("The two-dimensional Picard iteration did not converge.")

    final_conductivity = q2.conductivity_w_m_k(moisture_iterate)
    final_heat_capacity = q2.density_kg_m3(
        moisture_iterate
    ) * q2.heat_capacity_j_kg_k(moisture_iterate)
    final_diffusivity = moisture_diffusivity_kelvin(
        moisture_iterate, temperature_iterate
    )
    _, _, heat_side_g, heat_end_g = assemble_diffusion_system(
        grid,
        temperature_previous_c,
        final_conductivity,
        final_heat_capacity,
        q2.HEAT_TRANSFER_COEFFICIENT_W_M2_K,
        oven_temperature_c,
        time_step_s,
    )
    _, _, moisture_side_g, moisture_end_g = assemble_diffusion_system(
        grid,
        moisture_previous_kg_kg,
        final_diffusivity,
        np.ones_like(moisture_iterate),
        q2.MASS_TRANSFER_COEFFICIENT_M_S,
        oven_moisture_kg_kg,
        time_step_s,
    )
    side_temperature = reconstruct_robin_surface(
        temperature_iterate[:, -1],
        oven_temperature_c,
        final_conductivity[:, -1],
        q2.HEAT_TRANSFER_COEFFICIENT_W_M2_K,
        RADIUS_M - grid.r_centers_m[-1],
    )
    end_temperature = reconstruct_robin_surface(
        temperature_iterate[-1, :],
        oven_temperature_c,
        final_conductivity[-1, :],
        q2.HEAT_TRANSFER_COEFFICIENT_W_M2_K,
        HALF_LENGTH_M - grid.z_centers_m[-1],
    )
    side_moisture = reconstruct_robin_surface(
        moisture_iterate[:, -1],
        oven_moisture_kg_kg,
        final_diffusivity[:, -1],
        q2.MASS_TRANSFER_COEFFICIENT_M_S,
        RADIUS_M - grid.r_centers_m[-1],
    )
    end_moisture = reconstruct_robin_surface(
        moisture_iterate[-1, :],
        oven_moisture_kg_kg,
        final_diffusivity[-1, :],
        q2.MASS_TRANSFER_COEFFICIENT_M_S,
        HALF_LENGTH_M - grid.z_centers_m[-1],
    )
    moisture_outflow_rate = float(
        np.sum(moisture_side_g * (moisture_iterate[:, -1] - oven_moisture_kg_kg))
        + np.sum(moisture_end_g * (moisture_iterate[-1, :] - oven_moisture_kg_kg))
    )
    heat_inflow_rate = float(
        np.sum(heat_side_g * (oven_temperature_c - temperature_iterate[:, -1]))
        + np.sum(heat_end_g * (oven_temperature_c - temperature_iterate[-1, :]))
    )
    heat_storage_change = float(
        np.sum(
            final_heat_capacity
            * grid.cell_volumes_m3
            * (temperature_iterate - temperature_previous_c)
        )
    )
    return StepResult2D(
        temperature_c=temperature_iterate,
        moisture_kg_kg=moisture_iterate,
        side_temperature_c=side_temperature,
        end_temperature_c=end_temperature,
        side_moisture_kg_kg=side_moisture,
        end_moisture_kg_kg=end_moisture,
        moisture_outflow_rate_half_m3_s=moisture_outflow_rate,
        heat_inflow_rate_half_w=heat_inflow_rate,
        heat_storage_change_half_j=heat_storage_change,
        iterations=iteration,
        minimum_diffusivity_m2_s=float(np.min(final_diffusivity)),
        maximum_diffusivity_m2_s=float(np.max(final_diffusivity)),
    )


def reconstruct_even_zero(
    coordinate_centers: np.ndarray, values: np.ndarray, axis: int
) -> np.ndarray:
    """Second-order reconstruction at a symmetry plane from f=f0+a*x^2."""
    x0_sq = float(coordinate_centers[0] ** 2)
    x1_sq = float(coordinate_centers[1] ** 2)
    first = np.take(values, 0, axis=axis)
    second = np.take(values, 1, axis=axis)
    return (first * x1_sq - second * x0_sq) / (x1_sq - x0_sq)


def sample_field(
    grid: Grid2D,
    values: np.ndarray,
    side_values: np.ndarray,
    end_values: np.ndarray,
    query_radii_cm: np.ndarray = OUTPUT_RADII_CM,
    query_z_m: np.ndarray = OUTPUT_Z_M,
) -> np.ndarray:
    """Sample a cell-centred field at requested (z,r), including all boundaries."""
    query_r_m = np.asarray(query_radii_cm, dtype=float) / 100.0
    query_z = np.asarray(query_z_m, dtype=float)
    center_r = reconstruct_even_zero(grid.r_centers_m, values, axis=1)
    radial_samples = np.empty((grid.nz, query_r_m.size), dtype=float)
    extended_r = np.concatenate(([0.0], grid.r_centers_m, [RADIUS_M]))
    for j in range(grid.nz):
        extended_values = np.concatenate(
            ([center_r[j]], values[j, :], [side_values[j]])
        )
        radial_samples[j, :] = np.interp(query_r_m, extended_r, extended_values)

    middle_samples = reconstruct_even_zero(
        grid.z_centers_m, radial_samples, axis=0
    )
    end_center = float(reconstruct_even_zero(grid.r_centers_m, end_values, axis=0))
    corner_value = 0.5 * (float(side_values[-1]) + float(end_values[-1]))
    end_samples = np.interp(
        query_r_m,
        extended_r,
        np.concatenate(([end_center], end_values, [corner_value])),
    )
    sampled = np.empty((query_z.size, query_r_m.size), dtype=float)
    extended_z = np.concatenate(([0.0], grid.z_centers_m, [HALF_LENGTH_M]))
    for i in range(query_r_m.size):
        extended_values = np.concatenate(
            ([middle_samples[i]], radial_samples[:, i], [end_samples[i]])
        )
        sampled[:, i] = np.interp(query_z, extended_z, extended_values)
    return sampled


def field_maximum_location(grid: Grid2D, moisture: np.ndarray) -> tuple[float, float, float]:
    """Include symmetry-plane reconstructions in the strict all-domain maximum."""
    flat_index = int(np.argmax(moisture))
    j_cell, i_cell = np.unravel_index(flat_index, moisture.shape)
    candidates: list[tuple[float, float, float]] = [
        (
            float(moisture[j_cell, i_cell]),
            float(grid.r_centers_m[i_cell]),
            float(grid.z_centers_m[j_cell]),
        )
    ]
    center_r = reconstruct_even_zero(grid.r_centers_m, moisture, axis=1)
    j_center = int(np.argmax(center_r))
    candidates.append((float(center_r[j_center]), 0.0, float(grid.z_centers_m[j_center])))
    middle_z = reconstruct_even_zero(grid.z_centers_m, moisture, axis=0)
    i_middle = int(np.argmax(middle_z))
    candidates.append((float(middle_z[i_middle]), float(grid.r_centers_m[i_middle]), 0.0))
    center = float(reconstruct_even_zero(grid.z_centers_m, center_r, axis=0))
    candidates.append((center, 0.0, 0.0))
    return max(candidates, key=lambda item: item[0])


def choose_time_step(
    time_s: float,
    maximum_moisture: float,
    early_time_step_s: float,
    late_time_step_s: float,
    final_time_step_s: float,
) -> tuple[float, str]:
    if time_s < MEASURED_END_TIME_S:
        return early_time_step_s, "measured_environment"
    if maximum_moisture < DRYING_THRESHOLD_KG_KG + 0.002:
        return final_time_step_s, "threshold_refinement"
    return late_time_step_s, "constant_environment"


def simulate_until_dry(
    boundary: q3_1d.BoundaryProgram,
    nr: int,
    nz: int,
    early_time_step_s: float,
    late_time_step_s: float,
    final_time_step_s: float,
    maximum_time_s: float = MAXIMUM_SIMULATION_TIME_S,
    progress_label: str = "2D",
) -> DryingResult2D:
    grid = make_grid(nr, nz)
    temperature = np.full(
        (nz, nr), q2.INITIAL_TEMPERATURE_C, dtype=float
    )
    moisture = np.full((nz, nr), q2.INITIAL_MOISTURE_KG_KG, dtype=float)
    side_temperature = np.full(nz, q2.INITIAL_TEMPERATURE_C)
    end_temperature = np.full(nr, q2.INITIAL_TEMPERATURE_C)
    side_moisture = np.full(nz, q2.INITIAL_MOISTURE_KG_KG)
    end_moisture = np.full(nr, q2.INITIAL_MOISTURE_KG_KG)
    time_s = 0.0
    next_output_s = OUTPUT_INTERVAL_S
    record_times: list[float] = []
    temperature_records: list[np.ndarray] = []
    moisture_records: list[np.ndarray] = []
    integrated_moisture_outflow = 0.0
    integrated_heat_inflow = 0.0
    integrated_heat_storage = 0.0
    maximum_iterations = 0
    minimum_diffusivity = math.inf
    maximum_diffusivity = 0.0
    time_step_counts = {
        "measured_environment": 0,
        "constant_environment": 0,
        "threshold_refinement": 0,
    }
    maximum_moisture = q2.INITIAL_MOISTURE_KG_KG
    controlling_r = 0.0
    controlling_z = 0.0
    last_not_dry_time = 0.0
    last_not_dry_maximum = maximum_moisture
    started = time.perf_counter()
    next_progress_hour = 12.0

    while time_s < maximum_time_s:
        proposed_step, step_label = choose_time_step(
            time_s,
            maximum_moisture,
            early_time_step_s,
            late_time_step_s,
            final_time_step_s,
        )
        time_step_s = proposed_step
        for event_time in (MEASURED_END_TIME_S, next_output_s, maximum_time_s):
            if time_s < event_time < time_s + time_step_s:
                time_step_s = event_time - time_s
        if time_step_s <= 0.0:
            raise RuntimeError("A non-positive time step was generated.")

        new_time = time_s + time_step_s
        oven_temperature, oven_moisture = boundary.values(new_time)
        step = advance_coupled_step(
            grid,
            temperature,
            moisture,
            oven_temperature,
            oven_moisture,
            time_step_s,
        )
        # Grid represents half the physical cylinder; factor 2 gives whole-product balances.
        integrated_moisture_outflow += (
            2.0 * time_step_s * step.moisture_outflow_rate_half_m3_s
        )
        integrated_heat_inflow += 2.0 * time_step_s * step.heat_inflow_rate_half_w
        integrated_heat_storage += 2.0 * step.heat_storage_change_half_j
        temperature = step.temperature_c
        moisture = step.moisture_kg_kg
        side_temperature = step.side_temperature_c
        end_temperature = step.end_temperature_c
        side_moisture = step.side_moisture_kg_kg
        end_moisture = step.end_moisture_kg_kg
        time_s = new_time
        time_step_counts[step_label] += 1
        maximum_iterations = max(maximum_iterations, step.iterations)
        minimum_diffusivity = min(minimum_diffusivity, step.minimum_diffusivity_m2_s)
        maximum_diffusivity = max(maximum_diffusivity, step.maximum_diffusivity_m2_s)

        maximum_moisture, controlling_r, controlling_z = field_maximum_location(
            grid, moisture
        )
        is_output_time = math.isclose(time_s, next_output_s, abs_tol=1.0e-8)
        if is_output_time:
            record_times.append(time_s)
            temperature_records.append(
                sample_field(grid, temperature, side_temperature, end_temperature)
            )
            moisture_records.append(
                sample_field(grid, moisture, side_moisture, end_moisture)
            )
            next_output_s += OUTPUT_INTERVAL_S

        if time_s / 3600.0 >= next_progress_hour:
            print(
                f"[{progress_label}] t={time_s / 3600.0:.1f} h, "
                f"max C={maximum_moisture:.6f}, Picard={step.iterations}",
                flush=True,
            )
            next_progress_hour += 12.0

        if maximum_moisture < DRYING_THRESHOLD_KG_KG:
            if not is_output_time:
                record_times.append(time_s)
                temperature_records.append(
                    sample_field(grid, temperature, side_temperature, end_temperature)
                )
                moisture_records.append(
                    sample_field(grid, moisture, side_moisture, end_moisture)
                )
            break
        last_not_dry_time = time_s
        last_not_dry_maximum = maximum_moisture
    else:
        raise RuntimeError("Maximum simulation time reached before the product dried.")

    return DryingResult2D(
        times_s=np.asarray(record_times),
        sampled_temperature_c=np.stack(temperature_records),
        sampled_moisture_kg_kg=np.stack(moisture_records),
        drying_time_s=time_s,
        last_not_dry_time_s=last_not_dry_time,
        last_not_dry_maximum_moisture=last_not_dry_maximum,
        final_maximum_moisture=maximum_moisture,
        controlling_radius_m=controlling_r,
        controlling_z_m=controlling_z,
        final_temperature_c=temperature,
        final_moisture_kg_kg=moisture,
        final_side_temperature_c=side_temperature,
        final_end_temperature_c=end_temperature,
        final_side_moisture_kg_kg=side_moisture,
        final_end_moisture_kg_kg=end_moisture,
        grid=grid,
        integrated_moisture_outflow_whole_m3=integrated_moisture_outflow,
        integrated_heat_inflow_whole_j=integrated_heat_inflow,
        integrated_heat_storage_whole_j=integrated_heat_storage,
        maximum_picard_iterations=maximum_iterations,
        time_step_counts=time_step_counts,
        elapsed_seconds=time.perf_counter() - started,
        minimum_diffusivity_m2_s=minimum_diffusivity,
        maximum_diffusivity_m2_s=maximum_diffusivity,
    )


def build_table_times(drying_time_s: float) -> np.ndarray:
    regular = np.arange(
        TABLE_INTERVAL_S,
        drying_time_s - 1.0e-9,
        TABLE_INTERVAL_S,
        dtype=float,
    )
    return np.append(regular, drying_time_s)


def recorded_rows_at_times(result: DryingResult2D, times_s: np.ndarray) -> np.ndarray:
    lookup = {round(float(t), 8): i for i, t in enumerate(result.times_s)}
    indices = []
    for value in times_s:
        key = round(float(value), 8)
        if key not in lookup:
            raise ValueError(f"Requested time {value:g} s was not recorded.")
        indices.append(lookup[key])
    return np.asarray(indices, dtype=int)


def table5_values(result: DryingResult2D, table_times_s: np.ndarray) -> np.ndarray:
    time_indices = recorded_rows_at_times(result, table_times_s)
    radius_indices = [
        int(np.where(np.isclose(OUTPUT_RADII_CM, radius))[0][0])
        for radius in TABLE_RADII_CM
    ]
    return result.sampled_moisture_kg_kg[time_indices, 0, :][:, radius_indices]


def write_table5_csv(
    path: Path, times_s: np.ndarray, values: np.ndarray, drying_time_s: float
) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["时间/h（z=0中截面）", *[f"{r:g} cm" for r in TABLE_RADII_CM]])
        for time_s, row in zip(times_s, values):
            label = (
                "烘干结束时间"
                if math.isclose(time_s, drying_time_s, abs_tol=1.0e-8)
                else f"{time_s / 3600.0:.0f}"
            )
            writer.writerow([label, *[f"{value:.6f}" for value in row]])


def write_midplane_csv(path: Path, result: DryingResult2D, field: str) -> None:
    values = (
        result.sampled_moisture_kg_kg
        if field == "moisture"
        else result.sampled_temperature_c
    )
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_s", *[f"r_{r:.1f}_cm" for r in OUTPUT_RADII_CM]])
        for time_s, row in zip(result.times_s, values[:, 0, :]):
            writer.writerow([f"{time_s:.0f}", *[f"{value:.6f}" for value in row]])


def write_2d_long_csv(path: Path, result: DryingResult2D, field: str) -> None:
    values = (
        result.sampled_moisture_kg_kg
        if field == "moisture"
        else result.sampled_temperature_c
    )
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["time_s", "time_h", "z_m", *[f"r_{r:.1f}_cm" for r in OUTPUT_RADII_CM]]
        )
        for time_s, field_at_time in zip(result.times_s, values):
            for z_m, row in zip(OUTPUT_Z_M, field_at_time):
                writer.writerow(
                    [
                        f"{time_s:.0f}",
                        f"{time_s / 3600.0:.8f}",
                        f"{z_m:.3f}",
                        *[f"{value:.6f}" for value in row],
                    ]
                )


def write_final_cell_field_csv(path: Path, result: DryingResult2D) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["z_center_m", "r_center_m", "temperature_C", "temperature_K", "moisture_kg_kg"]
        )
        for j, z_m in enumerate(result.grid.z_centers_m):
            for i, r_m in enumerate(result.grid.r_centers_m):
                temperature_c = result.final_temperature_c[j, i]
                writer.writerow(
                    [
                        f"{z_m:.9f}",
                        f"{r_m:.9f}",
                        f"{temperature_c:.8f}",
                        f"{temperature_c + 273.15:.8f}",
                        f"{result.final_moisture_kg_kg[j, i]:.10f}",
                    ]
                )


def write_boundary_csv(path: Path, boundary: q3_1d.BoundaryProgram, end_time_s: float) -> None:
    times = np.arange(0.0, math.floor(end_time_s / 60.0) * 60.0 + 1.0, 60.0)
    if not math.isclose(times[-1], end_time_s, abs_tol=1.0e-8):
        times = np.append(times, end_time_s)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_s", "time_h", "oven_temperature_C", "oven_temperature_K", "oven_moisture_kg_kg"])
        for time_s in times:
            temperature_c, moisture = boundary.values(float(time_s))
            writer.writerow(
                [
                    f"{time_s:.0f}",
                    f"{time_s / 3600.0:.8f}",
                    f"{temperature_c:.8f}",
                    f"{temperature_c + 273.15:.8f}",
                    f"{moisture:.10f}",
                ]
            )


def build_validation(
    boundary: q3_1d.BoundaryProgram,
    production: DryingResult2D,
    coarse: DryingResult2D,
    table_times_s: np.ndarray,
    table_values: np.ndarray,
    one_dimensional_reference_path: Path,
) -> dict:
    common_end = min(production.drying_time_s, coarse.drying_time_s)
    production_lookup = {
        round(float(t), 8): i for i, t in enumerate(production.times_s)
    }
    coarse_lookup = {round(float(t), 8): i for i, t in enumerate(coarse.times_s)}
    common_keys = sorted(
        set(production_lookup).intersection(coarse_lookup), key=float
    )
    common_keys = [key for key in common_keys if float(key) <= common_end]
    production_common = np.stack(
        [production.sampled_moisture_kg_kg[production_lookup[key]] for key in common_keys]
    )
    coarse_common = np.stack(
        [coarse.sampled_moisture_kg_kg[coarse_lookup[key]] for key in common_keys]
    )
    refinement_change = np.abs(production_common - coarse_common)

    stored_change = float(
        2.0
        * np.sum(
            production.grid.cell_volumes_m3
            * (production.final_moisture_kg_kg - q2.INITIAL_MOISTURE_KG_KG)
        )
    )
    moisture_residual = stored_change + production.integrated_moisture_outflow_whole_m3
    moisture_scale = max(
        abs(stored_change), abs(production.integrated_moisture_outflow_whole_m3), 1.0e-30
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

    moisture_values = production.sampled_moisture_kg_kg
    one_dimensional_reference = None
    if one_dimensional_reference_path.exists():
        try:
            with one_dimensional_reference_path.open("r", encoding="utf-8") as stream:
                previous = json.load(stream)
            one_d_time = float(previous["drying_time"]["production_s"])
            one_dimensional_reference = {
                "source": str(one_dimensional_reference_path),
                "drying_time_s": one_d_time,
                "two_dimensional_minus_one_dimensional_s": production.drying_time_s - one_d_time,
                "two_dimensional_minus_one_dimensional_h": (production.drying_time_s - one_d_time) / 3600.0,
            }
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            one_dimensional_reference = {"source": str(one_dimensional_reference_path), "status": "unreadable"}

    dry_volume_m3 = math.pi * RADIUS_M**2 * FULL_LENGTH_M
    dry_density = 275.042253521
    return {
        "model_scope": {
            "geometry": "fixed full cylinder represented by axisymmetric half-domain",
            "domain_m": {"r": [0.0, RADIUS_M], "z": [0.0, HALF_LENGTH_M]},
            "symmetry_boundaries": ["r=0", "z=0 midplane"],
            "convective_boundaries": ["r=R side", "z=L/2 end"],
            "latent_heat": "excluded",
            "shrinkage": "excluded in question 3",
            "dry_density_role": "constant physical mass-balance reference; cancels from normalized Fick equation",
            "dry_density_kg_m3": dry_density,
            "constant_dry_mass_kg": dry_density * dry_volume_m3,
            "drying_threshold": "strict max over all reconstructed 2D positions < 0.15 kg/kg",
            "post_4h_boundary": "time-weighted mean over Attachment 1 from 3 h to 4 h",
            "temperature_storage_unit": "degree Celsius",
            "diffusivity_temperature_unit": "kelvin after adding 273.15",
            "solver": "cell-centred FVM, backward Euler, five-point sparse matrix, Picard coupling",
        },
        "grid_and_time": {
            "production": {
                "nr": production.grid.nr,
                "nz": production.grid.nz,
                "nominal_dr_m": production.grid.dr_m,
                "nominal_dz_m": production.grid.dz_m,
                "radial_cell_width_range_m": list(production.grid.radial_cell_width_range_m),
                "axial_cell_width_range_m": list(production.grid.axial_cell_width_range_m),
                "elapsed_s": production.elapsed_seconds,
            },
            "coarse": {
                "nr": coarse.grid.nr,
                "nz": coarse.grid.nz,
                "nominal_dr_m": coarse.grid.dr_m,
                "nominal_dz_m": coarse.grid.dz_m,
                "radial_cell_width_range_m": list(coarse.grid.radial_cell_width_range_m),
                "axial_cell_width_range_m": list(coarse.grid.axial_cell_width_range_m),
                "elapsed_s": coarse.elapsed_seconds,
            },
            "saved_interval_s": OUTPUT_INTERVAL_S,
            "saved_radii_cm": OUTPUT_RADII_CM.tolist(),
            "saved_z_m": OUTPUT_Z_M.tolist(),
            "workbook_z_m": WORKBOOK_Z_M.tolist(),
            "production_time_step_counts": production.time_step_counts,
        },
        "constant_boundary_after_4h": {
            "temperature_C": boundary.plateau_temperature_c,
            "temperature_K": boundary.plateau_temperature_c + 273.15,
            "moisture_kg_kg": boundary.plateau_moisture_kg_kg,
        },
        "drying_time": {
            "production_s": production.drying_time_s,
            "production_h": production.drying_time_s / 3600.0,
            "coarse_s": coarse.drying_time_s,
            "coarse_h": coarse.drying_time_s / 3600.0,
            "coarse_production_difference_s": abs(production.drying_time_s - coarse.drying_time_s),
            "previous_time_s": production.last_not_dry_time_s,
            "previous_maximum_moisture": production.last_not_dry_maximum_moisture,
            "final_maximum_moisture": production.final_maximum_moisture,
            "controlling_radius_m": production.controlling_radius_m,
            "controlling_z_m": production.controlling_z_m,
        },
        "grid_refinement": {
            "common_record_count": len(common_keys),
            "maximum_sampled_moisture_change_kg_kg": float(np.max(refinement_change)),
            "mean_sampled_moisture_change_kg_kg": float(np.mean(refinement_change)),
        },
        "whole_product_balances": {
            "moisture_stored_change_m3_kg_kg": stored_change,
            "integrated_moisture_outflow_m3_kg_kg": production.integrated_moisture_outflow_whole_m3,
            "moisture_residual": moisture_residual,
            "moisture_relative_residual": abs(moisture_residual) / moisture_scale,
            "integrated_heat_storage_J": production.integrated_heat_storage_whole_j,
            "integrated_heat_inflow_J": production.integrated_heat_inflow_whole_j,
            "heat_residual_J": heat_residual,
            "heat_relative_residual": abs(heat_residual) / heat_scale,
        },
        "checks": {
            "previous_state_not_dry": bool(production.last_not_dry_maximum_moisture >= DRYING_THRESHOLD_KG_KG),
            "final_state_all_dry": bool(production.final_maximum_moisture < DRYING_THRESHOLD_KG_KG),
            "minimum_saved_moisture": float(np.min(moisture_values)),
            "radial_order_violation_count": int(np.count_nonzero(np.diff(moisture_values, axis=2) > 1.0e-8)),
            "axial_order_violation_count": int(np.count_nonzero(np.diff(moisture_values, axis=1) > 1.0e-8)),
            "time_monotonicity_violation_count": int(np.count_nonzero(np.diff(moisture_values, axis=0) > 1.0e-8)),
            "maximum_picard_iterations": production.maximum_picard_iterations,
            "minimum_diffusivity_m2_s": production.minimum_diffusivity_m2_s,
            "maximum_diffusivity_m2_s": production.maximum_diffusivity_m2_s,
        },
        "one_dimensional_reference": one_dimensional_reference,
        "table5_midplane": {
            "row_count": int(table_times_s.size),
            "last_time_h": float(table_times_s[-1] / 3600.0),
            "last_row": [float(value) for value in table_values[-1]],
        },
    }


def run_workbook_builder(repo_root: Path, result_dir: Path) -> None:
    """Build and verify result3_dimension2.xlsx with the repository Node runtime."""
    import subprocess

    node_executable = Path(
        r"C:\Users\user\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe"
    )
    builder = repo_root / "code" / "build_a3_dimension2_workbook.cjs"
    preview_dir = result_dir / "workbook_preview"
    command = [
        str(node_executable),
        str(builder),
        str(result_dir / "table5_midplane_moisture.csv"),
        str(result_dir / "midplane_moisture_60s_r0p1cm.csv"),
        str(result_dir / "moisture_2d_60s_r0p1cm_z0p5cm.csv"),
        str(result_dir / "temperature_2d_60s_r0p1cm_z0p5cm.csv"),
        str(result_dir / "validation_summary.json"),
        str(result_dir / "result3_dimension2.xlsx"),
        str(preview_dir),
    ]
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="附件1.xlsx的路径")
    parser.add_argument("--repo-root", type=Path, help="项目根目录")
    parser.add_argument("--nr", type=int, default=32, help="生产网格径向单元数")
    parser.add_argument("--nz", type=int, default=80, help="生产网格半轴向单元数")
    parser.add_argument("--coarse-nr", type=int, default=16, help="粗网格径向单元数")
    parser.add_argument("--coarse-nz", type=int, default=40, help="粗网格半轴向单元数")
    parser.add_argument("--early-dt", type=float, default=30.0)
    parser.add_argument("--late-dt", type=float, default=60.0)
    parser.add_argument("--final-dt", type=float, default=1.0)
    parser.add_argument("--maximum-time", type=float, default=MAXIMUM_SIMULATION_TIME_S)
    parser.add_argument("--skip-coarse", action="store_true", help="调试时让粗网格复用生产结果")
    parser.add_argument("--skip-xlsx", action="store_true", help="只生成CSV和JSON")
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    repo_root = args.repo_root.resolve() if args.repo_root else script_path.parents[1]
    data_path = args.data.resolve() if args.data else q2.find_default_data_path(repo_root)
    result_dir = repo_root / "results" / "A_problem3_dimension2"
    result_dir.mkdir(parents=True, exist_ok=True)

    environment = q2.load_and_preprocess_environment(data_path)
    boundary = q3_1d.build_boundary_program(environment)
    production = simulate_until_dry(
        boundary,
        nr=args.nr,
        nz=args.nz,
        early_time_step_s=args.early_dt,
        late_time_step_s=args.late_dt,
        final_time_step_s=args.final_dt,
        maximum_time_s=args.maximum_time,
        progress_label=f"production {args.nr}x{args.nz}",
    )
    coarse = production if args.skip_coarse else simulate_until_dry(
        boundary,
        nr=args.coarse_nr,
        nz=args.coarse_nz,
        early_time_step_s=args.early_dt,
        late_time_step_s=args.late_dt,
        final_time_step_s=args.final_dt,
        maximum_time_s=args.maximum_time,
        progress_label=f"coarse {args.coarse_nr}x{args.coarse_nz}",
    )

    table_times = build_table_times(production.drying_time_s)
    table = table5_values(production, table_times)
    write_table5_csv(result_dir / "table5_midplane_moisture.csv", table_times, table, production.drying_time_s)
    write_midplane_csv(result_dir / "midplane_moisture_60s_r0p1cm.csv", production, "moisture")
    write_midplane_csv(result_dir / "midplane_temperature_60s_r0p1cm.csv", production, "temperature")
    write_2d_long_csv(result_dir / "moisture_2d_60s_r0p1cm_z0p5cm.csv", production, "moisture")
    write_2d_long_csv(result_dir / "temperature_2d_60s_r0p1cm_z0p5cm.csv", production, "temperature")
    write_final_cell_field_csv(result_dir / "final_cell_field_2d.csv", production)
    write_boundary_csv(result_dir / "boundary_environment_60s.csv", boundary, production.drying_time_s)

    validation = build_validation(
        boundary,
        production,
        coarse,
        table_times,
        table,
        repo_root / "results" / "A_problem3_drying_time" / "validation_summary.json",
    )
    with (result_dir / "validation_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(validation, stream, ensure_ascii=False, indent=2)

    if not args.skip_xlsx:
        run_workbook_builder(repo_root, result_dir)

    print(
        f"Post-4 h boundary: {boundary.plateau_temperature_c:.6f} deg C, "
        f"{boundary.plateau_moisture_kg_kg:.8f} kg/kg"
    )
    print(
        f"2D drying time: {production.drying_time_s:.0f} s = "
        f"{production.drying_time_s / 3600.0:.6f} h"
    )
    print(
        f"Threshold bracket: {production.last_not_dry_time_s:.0f} s "
        f"({production.last_not_dry_maximum_moisture:.10f}) -> "
        f"{production.drying_time_s:.0f} s ({production.final_maximum_moisture:.10f})"
    )
    print(
        f"Controlling point: r={production.controlling_radius_m:.6f} m, "
        f"z={production.controlling_z_m:.6f} m"
    )
    print("\nTable 5-compatible midplane moisture (kg/kg)")
    for row_time, row in zip(table_times, table):
        label = "end" if math.isclose(row_time, production.drying_time_s) else f"{row_time / 3600.0:.0f} h"
        print(label, np.array2string(row, precision=6))
    print(f"\nResults: {result_dir}")


if __name__ == "__main__":
    main()
