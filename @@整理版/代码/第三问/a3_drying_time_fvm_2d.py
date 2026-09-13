"""A题问题3：考虑两个端面暴露的轴对称二维水热耦合模型。

求解半长圆柱 ``0 <= r <= R, 0 <= z <= L/2``。其中 ``z=0`` 为
中截面对称面，``z=L/2`` 为暴露端面；完整圆柱的另一半由对称性得到。
侧面与端面均使用第三类换热、传质边界，物性关系和环境边界程序与原
一维第三问保持一致，以便单独评价有限长度与端面效应。

运行示例（从项目根目录执行）：

    python code/a3_drying_time_fvm_2d.py

正式结果写入 ``result/A_problem3_2d_exposed/``。可用 ``--quick`` 执行
小网格短时测试，或用 ``--skip-validation`` 跳过粗网格对照。
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
from scipy import sparse
from scipy.sparse.linalg import spsolve

import a2_coupled_fvm as q2
import a3_drying_time_fvm as q3


LENGTH_M = 0.25
HALF_LENGTH_M = LENGTH_M / 2.0
DRYING_THRESHOLD_KG_KG = q3.DRYING_THRESHOLD_KG_KG
OUTPUT_INTERVAL_S = 60.0
TABLE_INTERVAL_S = 6.0 * 3600.0
OUTPUT_RADII_CM = np.round(np.arange(0.0, 2.0 + 0.05, 0.1), 1)
TABLE_RADII_CM = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
AXIAL_OUTPUT_CM = np.round(np.arange(0.0, 12.5 + 0.25, 0.5), 1)


@dataclass(frozen=True)
class Grid2D:
    radial_faces_m: np.ndarray
    radii_m: np.ndarray
    axial_faces_m: np.ndarray
    axial_positions_m: np.ndarray
    annulus_areas_m2: np.ndarray
    axial_widths_m: np.ndarray
    volumes_m3: np.ndarray
    radial_edge_indices: np.ndarray

    @property
    def nr(self) -> int:
        return self.radii_m.size

    @property
    def nz(self) -> int:
        return self.axial_positions_m.size


@dataclass
class StepResult2D:
    temperature_c: np.ndarray
    moisture_kg_kg: np.ndarray
    side_temperature_c: np.ndarray
    side_moisture_kg_kg: np.ndarray
    end_temperature_c: np.ndarray
    end_moisture_kg_kg: np.ndarray
    side_moisture_conductance: np.ndarray
    end_moisture_conductance: np.ndarray
    iterations: int


@dataclass
class SimulationResult2D:
    grid: Grid2D
    times_s: np.ndarray
    midplane_temperature_c: np.ndarray
    midplane_moisture_kg_kg: np.ndarray
    snapshot_times_s: np.ndarray
    temperature_snapshots_c: np.ndarray
    moisture_snapshots_kg_kg: np.ndarray
    drying_time_s: float
    previous_time_s: float
    previous_maximum_moisture: float
    final_maximum_moisture: float
    controlling_radius_cm: float
    controlling_axial_cm: float
    controlling_location: str
    integrated_side_outflow: float
    integrated_end_outflow: float
    storage_change: float
    balance_residual: float
    balance_relative_residual: float
    maximum_picard_iterations: int
    time_step_counts: dict[str, int]
    runtime_s: float


def make_grid(nr: int, nz: int, radial_power: float = 2.0, axial_power: float = 2.0) -> Grid2D:
    """建立侧面、端面加密的半长轴对称有限体积网格。"""
    if nr < 3 or nz < 3:
        raise ValueError("nr and nz must both be at least 3.")
    radial_logical = np.linspace(0.0, 1.0, nr + 1)
    radial_faces = q2.RADIUS_M * (
        1.0 - (1.0 - radial_logical) ** radial_power
    )
    axial_logical = np.linspace(0.0, 1.0, nz + 1)
    axial_faces = HALF_LENGTH_M * (
        1.0 - (1.0 - axial_logical) ** axial_power
    )
    radii = 0.5 * (radial_faces[:-1] + radial_faces[1:])
    axial_positions = 0.5 * (axial_faces[:-1] + axial_faces[1:])
    annulus_areas = math.pi * (radial_faces[1:] ** 2 - radial_faces[:-1] ** 2)
    axial_widths = np.diff(axial_faces)
    volumes = axial_widths[:, None] * annulus_areas[None, :]
    radial_edge_indices = (
        np.arange(nz, dtype=int)[:, None] * nr
        + np.arange(nr - 1, dtype=int)[None, :]
    ).ravel()
    return Grid2D(
        radial_faces_m=radial_faces,
        radii_m=radii,
        axial_faces_m=axial_faces,
        axial_positions_m=axial_positions,
        annulus_areas_m2=annulus_areas,
        axial_widths_m=axial_widths,
        volumes_m3=volumes,
        radial_edge_indices=radial_edge_indices,
    )


def _internal_conductances(
    grid: Grid2D, coefficient: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """按两侧半单元串联扩散阻力计算径向和轴向导通系数。"""
    radial_face = grid.radial_faces_m[1:-1][None, :]
    radial_resistance = (
        (radial_face - grid.radii_m[:-1][None, :]) / coefficient[:, :-1]
        + (grid.radii_m[1:][None, :] - radial_face) / coefficient[:, 1:]
    )
    radial_area = 2.0 * math.pi * radial_face * grid.axial_widths_m[:, None]
    radial_conductance = radial_area / radial_resistance

    axial_face = grid.axial_faces_m[1:-1][:, None]
    axial_resistance = (
        (axial_face - grid.axial_positions_m[:-1, None]) / coefficient[:-1, :]
        + (grid.axial_positions_m[1:, None] - axial_face) / coefficient[1:, :]
    )
    axial_conductance = grid.annulus_areas_m2[None, :] / axial_resistance
    return radial_conductance, axial_conductance


def _boundary_conductance(
    area: np.ndarray,
    distance_m: float | np.ndarray,
    cell_coefficient: np.ndarray,
    transfer_coefficient: float,
) -> np.ndarray:
    if transfer_coefficient <= 0.0:
        return np.zeros_like(np.asarray(cell_coefficient, dtype=float))
    return area / (distance_m / cell_coefficient + 1.0 / transfer_coefficient)


def build_diffusion_system_2d(
    grid: Grid2D,
    field_previous: np.ndarray,
    diffusion_coefficient: np.ndarray,
    storage_coefficient: np.ndarray,
    side_transfer_coefficient: float,
    end_transfer_coefficient: float,
    boundary_value: float,
    time_step_s: float,
) -> tuple[sparse.csc_matrix, np.ndarray, np.ndarray, np.ndarray]:
    """组装后向欧拉二维轴对称扩散方程的稀疏五点系统。"""
    nr, nz = grid.nr, grid.nz
    if field_previous.shape != (nz, nr):
        raise ValueError("Field shape does not match the two-dimensional grid.")
    radial_g, axial_g = _internal_conductances(grid, diffusion_coefficient)
    storage = storage_coefficient * grid.volumes_m3 / time_step_s
    diagonal = storage.copy()
    diagonal[:, :-1] += radial_g
    diagonal[:, 1:] += radial_g
    diagonal[:-1, :] += axial_g
    diagonal[1:, :] += axial_g

    side_area = 2.0 * math.pi * q2.RADIUS_M * grid.axial_widths_m
    side_g = _boundary_conductance(
        side_area,
        q2.RADIUS_M - grid.radii_m[-1],
        diffusion_coefficient[:, -1],
        side_transfer_coefficient,
    )
    end_g = _boundary_conductance(
        grid.annulus_areas_m2,
        HALF_LENGTH_M - grid.axial_positions_m[-1],
        diffusion_coefficient[-1, :],
        end_transfer_coefficient,
    )
    diagonal[:, -1] += side_g
    diagonal[-1, :] += end_g

    rhs = storage * field_previous
    rhs[:, -1] += side_g * boundary_value
    rhs[-1, :] += end_g * boundary_value

    cell_count = nr * nz
    offset_one = np.zeros(cell_count - 1)
    offset_one[grid.radial_edge_indices] = -radial_g.ravel()
    offset_axial = -axial_g.ravel()
    matrix = sparse.diags(
        [offset_axial, offset_one, diagonal.ravel(), offset_one, offset_axial],
        offsets=[-nr, -1, 0, 1, nr],
        shape=(cell_count, cell_count),
        format="csc",
    )
    return matrix, rhs.ravel(), side_g, end_g


def _surface_values(
    cell_values: np.ndarray,
    boundary_value: float,
    cell_coefficient: np.ndarray,
    transfer_coefficient: float,
    distance_m: float,
) -> np.ndarray:
    if transfer_coefficient <= 0.0:
        return np.asarray(cell_values, dtype=float).copy()
    cell_side = cell_coefficient / distance_m
    return (cell_side * cell_values + transfer_coefficient * boundary_value) / (
        cell_side + transfer_coefficient
    )


def advance_coupled_step_2d(
    grid: Grid2D,
    temperature_previous: np.ndarray,
    moisture_previous: np.ndarray,
    oven_temperature_c: float,
    oven_moisture_kg_kg: float,
    time_step_s: float,
    temperature_tolerance_c: float = 1.0e-8,
    moisture_tolerance: float = 1.0e-10,
    maximum_iterations: int = 30,
    end_heat_transfer_coefficient: float = q2.HEAT_TRANSFER_COEFFICIENT_W_M2_K,
    end_mass_transfer_coefficient: float = q2.MASS_TRANSFER_COEFFICIENT_M_S,
) -> StepResult2D:
    """用Picard迭代推进一个全隐式水热耦合时间步。"""
    temperature_iterate = temperature_previous.copy()
    moisture_iterate = moisture_previous.copy()
    for iteration in range(1, maximum_iterations + 1):
        conductivity = q2.conductivity_w_m_k(moisture_iterate)
        heat_storage = q2.density_kg_m3(moisture_iterate) * q2.heat_capacity_j_kg_k(
            moisture_iterate
        )
        heat_matrix, heat_rhs, _, _ = build_diffusion_system_2d(
            grid,
            temperature_previous,
            conductivity,
            heat_storage,
            q2.HEAT_TRANSFER_COEFFICIENT_W_M2_K,
            end_heat_transfer_coefficient,
            oven_temperature_c,
            time_step_s,
        )
        temperature_updated = np.asarray(spsolve(heat_matrix, heat_rhs)).reshape(
            grid.nz, grid.nr
        )

        diffusivity = q2.moisture_diffusivity_m2_s(
            moisture_iterate, temperature_updated
        )
        moisture_matrix, moisture_rhs, side_moisture_g, end_moisture_g = (
            build_diffusion_system_2d(
                grid,
                moisture_previous,
                diffusivity,
                np.ones_like(moisture_iterate),
                q2.MASS_TRANSFER_COEFFICIENT_M_S,
                end_mass_transfer_coefficient,
                oven_moisture_kg_kg,
                time_step_s,
            )
        )
        moisture_updated = np.asarray(
            spsolve(moisture_matrix, moisture_rhs)
        ).reshape(grid.nz, grid.nr)

        temperature_error = float(
            np.max(np.abs(temperature_updated - temperature_iterate))
        )
        moisture_error = float(np.max(np.abs(moisture_updated - moisture_iterate)))
        temperature_iterate = temperature_updated
        moisture_iterate = moisture_updated
        if (
            temperature_error < temperature_tolerance_c
            and moisture_error < moisture_tolerance
        ):
            break
    else:
        raise RuntimeError("Two-dimensional coupled Picard iteration did not converge.")

    final_conductivity = q2.conductivity_w_m_k(moisture_iterate)
    final_diffusivity = q2.moisture_diffusivity_m2_s(
        moisture_iterate, temperature_iterate
    )
    side_distance = q2.RADIUS_M - grid.radii_m[-1]
    end_distance = HALF_LENGTH_M - grid.axial_positions_m[-1]
    side_temperature = _surface_values(
        temperature_iterate[:, -1],
        oven_temperature_c,
        final_conductivity[:, -1],
        q2.HEAT_TRANSFER_COEFFICIENT_W_M2_K,
        side_distance,
    )
    side_moisture = _surface_values(
        moisture_iterate[:, -1],
        oven_moisture_kg_kg,
        final_diffusivity[:, -1],
        q2.MASS_TRANSFER_COEFFICIENT_M_S,
        side_distance,
    )
    end_temperature = _surface_values(
        temperature_iterate[-1, :],
        oven_temperature_c,
        final_conductivity[-1, :],
        end_heat_transfer_coefficient,
        end_distance,
    )
    end_moisture = _surface_values(
        moisture_iterate[-1, :],
        oven_moisture_kg_kg,
        final_diffusivity[-1, :],
        end_mass_transfer_coefficient,
        end_distance,
    )
    return StepResult2D(
        temperature_c=temperature_iterate,
        moisture_kg_kg=moisture_iterate,
        side_temperature_c=side_temperature,
        side_moisture_kg_kg=side_moisture,
        end_temperature_c=end_temperature,
        end_moisture_kg_kg=end_moisture,
        side_moisture_conductance=side_moisture_g,
        end_moisture_conductance=end_moisture_g,
        iterations=iteration,
    )


def _even_boundary_value(
    coordinate_0: float,
    coordinate_1: float,
    value_0: np.ndarray | float,
    value_1: np.ndarray | float,
) -> np.ndarray | float:
    """由偶对称二次函数重构坐标零点处的值。"""
    x0_sq = coordinate_0**2
    x1_sq = coordinate_1**2
    return (value_0 * x1_sq - value_1 * x0_sq) / (x1_sq - x0_sq)


def reconstruct_midplane(grid: Grid2D, field: np.ndarray) -> np.ndarray:
    return np.asarray(
        _even_boundary_value(
            grid.axial_positions_m[0],
            grid.axial_positions_m[1],
            field[0, :],
            field[1, :],
        )
    )


def reconstruct_axis(grid: Grid2D, field: np.ndarray) -> np.ndarray:
    return np.asarray(
        _even_boundary_value(
            grid.radii_m[0],
            grid.radii_m[1],
            field[:, 0],
            field[:, 1],
        )
    )


def reconstruct_center_point(grid: Grid2D, field: np.ndarray) -> float:
    axis = reconstruct_axis(grid, field)
    return float(
        _even_boundary_value(
            grid.axial_positions_m[0],
            grid.axial_positions_m[1],
            axis[0],
            axis[1],
        )
    )


def sample_midplane_radial(
    grid: Grid2D,
    field: np.ndarray,
    side_surface_values: np.ndarray,
    query_radii_cm: np.ndarray,
) -> np.ndarray:
    midplane = reconstruct_midplane(grid, field)
    center = float(
        _even_boundary_value(
            grid.radii_m[0], grid.radii_m[1], midplane[0], midplane[1]
        )
    )
    side_midplane = float(
        _even_boundary_value(
            grid.axial_positions_m[0],
            grid.axial_positions_m[1],
            side_surface_values[0],
            side_surface_values[1],
        )
    )
    coordinates = np.concatenate(([0.0], grid.radii_m, [q2.RADIUS_M]))
    values = np.concatenate(([center], midplane, [side_midplane]))
    return np.interp(query_radii_cm / 100.0, coordinates, values)


def sample_axis_axial(
    grid: Grid2D,
    field: np.ndarray,
    end_surface_values: np.ndarray,
    query_axial_cm: np.ndarray,
) -> np.ndarray:
    axis = reconstruct_axis(grid, field)
    center = float(
        _even_boundary_value(
            grid.axial_positions_m[0],
            grid.axial_positions_m[1],
            axis[0],
            axis[1],
        )
    )
    end_axis = float(
        _even_boundary_value(
            grid.radii_m[0],
            grid.radii_m[1],
            end_surface_values[0],
            end_surface_values[1],
        )
    )
    coordinates = np.concatenate(
        ([0.0], grid.axial_positions_m, [HALF_LENGTH_M])
    )
    values = np.concatenate(([center], axis, [end_axis]))
    return np.interp(query_axial_cm / 100.0, coordinates, values)


def maximum_moisture(
    grid: Grid2D,
    moisture: np.ndarray,
    side_surface: np.ndarray,
    end_surface: np.ndarray,
) -> tuple[float, float, float, str]:
    """在内部、对称边界和物理边界候选点中寻找二维最大水分。"""
    flat_index = int(np.argmax(moisture))
    j, i = np.unravel_index(flat_index, moisture.shape)
    candidates: list[tuple[float, float, float, str]] = [
        (
            float(moisture[j, i]),
            float(grid.radii_m[i] * 100.0),
            float(grid.axial_positions_m[j] * 100.0),
            "cell_center",
        )
    ]
    axis = reconstruct_axis(grid, moisture)
    j_axis = int(np.argmax(axis))
    candidates.append(
        (
            float(axis[j_axis]),
            0.0,
            float(grid.axial_positions_m[j_axis] * 100.0),
            "symmetry_axis",
        )
    )
    midplane = reconstruct_midplane(grid, moisture)
    i_mid = int(np.argmax(midplane))
    candidates.append(
        (
            float(midplane[i_mid]),
            float(grid.radii_m[i_mid] * 100.0),
            0.0,
            "midplane",
        )
    )
    candidates.append((reconstruct_center_point(grid, moisture), 0.0, 0.0, "center"))
    j_side = int(np.argmax(side_surface))
    candidates.append(
        (
            float(side_surface[j_side]),
            q2.RADIUS_M * 100.0,
            float(grid.axial_positions_m[j_side] * 100.0),
            "side_surface",
        )
    )
    i_end = int(np.argmax(end_surface))
    candidates.append(
        (
            float(end_surface[i_end]),
            float(grid.radii_m[i_end] * 100.0),
            HALF_LENGTH_M * 100.0,
            "end_surface",
        )
    )
    return max(candidates, key=lambda item: item[0])


def choose_time_step(
    time_s: float,
    maximum_value: float,
    early_step_s: float,
    late_step_s: float,
    final_step_s: float,
    refinement_margin: float,
) -> tuple[float, str]:
    if time_s < q3.MEASURED_END_TIME_S:
        return early_step_s, "measured_environment"
    if maximum_value < DRYING_THRESHOLD_KG_KG + refinement_margin:
        return final_step_s, "threshold_refinement"
    return late_step_s, "constant_environment"


def simulate_until_dry_2d(
    boundary: q3.BoundaryProgram,
    nr: int,
    nz: int,
    early_step_s: float,
    late_step_s: float,
    final_step_s: float,
    refinement_margin: float,
    maximum_time_s: float = q3.MAXIMUM_SIMULATION_TIME_S,
) -> SimulationResult2D:
    """求解至二维全域水分严格低于阈值。"""
    started = time.perf_counter()
    grid = make_grid(nr, nz)
    temperature = np.full((nz, nr), q2.INITIAL_TEMPERATURE_C)
    moisture = np.full((nz, nr), q2.INITIAL_MOISTURE_KG_KG)
    side_temperature = np.full(nz, q2.INITIAL_TEMPERATURE_C)
    side_moisture = np.full(nz, q2.INITIAL_MOISTURE_KG_KG)
    end_temperature = np.full(nr, q2.INITIAL_TEMPERATURE_C)
    end_moisture = np.full(nr, q2.INITIAL_MOISTURE_KG_KG)
    initial_storage = float(np.sum(grid.volumes_m3 * moisture))

    record_times: list[float] = []
    midplane_temperature_records: list[np.ndarray] = []
    midplane_moisture_records: list[np.ndarray] = []
    snapshot_times: list[float] = []
    temperature_snapshots: list[np.ndarray] = []
    moisture_snapshots: list[np.ndarray] = []
    time_s = 0.0
    next_output_s = OUTPUT_INTERVAL_S
    next_snapshot_s = TABLE_INTERVAL_S
    max_value = q2.INITIAL_MOISTURE_KG_KG
    previous_time = 0.0
    previous_maximum = max_value
    controlling_r = 0.0
    controlling_z = 0.0
    controlling_location = "initial_uniform"
    side_outflow = 0.0
    end_outflow = 0.0
    maximum_iterations_used = 0
    time_step_counts = {
        "measured_environment": 0,
        "constant_environment": 0,
        "threshold_refinement": 0,
    }

    while time_s < maximum_time_s:
        proposed_step, step_label = choose_time_step(
            time_s,
            max_value,
            early_step_s,
            late_step_s,
            final_step_s,
            refinement_margin,
        )
        time_step_s = proposed_step
        event_times = [maximum_time_s, next_snapshot_s]
        if time_s < q3.MEASURED_END_TIME_S:
            event_times.append(q3.MEASURED_END_TIME_S)
        future_events = [event for event in event_times if event > time_s + 1.0e-10]
        if future_events:
            time_step_s = min(time_step_s, min(future_events) - time_s)
        if time_step_s <= 0.0:
            raise RuntimeError("Non-positive two-dimensional time step encountered.")

        old_time = time_s
        temperature_previous = temperature
        moisture_previous = moisture
        side_temperature_previous = side_temperature
        side_moisture_previous = side_moisture
        new_time = time_s + time_step_s
        oven_temperature, oven_moisture = boundary.values(new_time)
        step = advance_coupled_step_2d(
            grid,
            temperature,
            moisture,
            oven_temperature,
            oven_moisture,
            time_step_s,
        )
        side_outflow += time_step_s * float(
            np.sum(
                step.side_moisture_conductance
                * (step.moisture_kg_kg[:, -1] - oven_moisture)
            )
        )
        end_outflow += time_step_s * float(
            np.sum(
                step.end_moisture_conductance
                * (step.moisture_kg_kg[-1, :] - oven_moisture)
            )
        )
        temperature = step.temperature_c
        moisture = step.moisture_kg_kg
        side_temperature = step.side_temperature_c
        side_moisture = step.side_moisture_kg_kg
        end_temperature = step.end_temperature_c
        end_moisture = step.end_moisture_kg_kg
        time_s = new_time
        time_step_counts[step_label] += 1
        maximum_iterations_used = max(maximum_iterations_used, step.iterations)
        max_value, controlling_r, controlling_z, controlling_location = maximum_moisture(
            grid, moisture, side_moisture, end_moisture
        )

        recorded_final_step_time = False
        while next_output_s <= time_s + 1.0e-7:
            fraction = (next_output_s - old_time) / time_step_s
            output_temperature = temperature_previous + fraction * (
                temperature - temperature_previous
            )
            output_moisture = moisture_previous + fraction * (
                moisture - moisture_previous
            )
            output_side_temperature = side_temperature_previous + fraction * (
                side_temperature - side_temperature_previous
            )
            output_side_moisture = side_moisture_previous + fraction * (
                side_moisture - side_moisture_previous
            )
            record_times.append(next_output_s)
            midplane_temperature_records.append(
                sample_midplane_radial(
                    grid,
                    output_temperature,
                    output_side_temperature,
                    OUTPUT_RADII_CM,
                )
            )
            midplane_moisture_records.append(
                sample_midplane_radial(
                    grid, output_moisture, output_side_moisture, OUTPUT_RADII_CM
                )
            )
            recorded_final_step_time = math.isclose(
                next_output_s, time_s, abs_tol=1.0e-7
            )
            next_output_s += OUTPUT_INTERVAL_S

        is_snapshot = math.isclose(time_s, next_snapshot_s, abs_tol=1.0e-7)
        if is_snapshot:
            snapshot_times.append(time_s)
            temperature_snapshots.append(temperature.copy())
            moisture_snapshots.append(moisture.copy())
            next_snapshot_s += TABLE_INTERVAL_S

        if max_value < DRYING_THRESHOLD_KG_KG:
            if not recorded_final_step_time:
                record_times.append(time_s)
                midplane_temperature_records.append(
                    sample_midplane_radial(
                        grid, temperature, side_temperature, OUTPUT_RADII_CM
                    )
                )
                midplane_moisture_records.append(
                    sample_midplane_radial(
                        grid, moisture, side_moisture, OUTPUT_RADII_CM
                    )
                )
            if not is_snapshot:
                snapshot_times.append(time_s)
                temperature_snapshots.append(temperature.copy())
                moisture_snapshots.append(moisture.copy())
            break

        previous_time = time_s
        previous_maximum = max_value
    else:
        raise RuntimeError("Maximum time reached before the two-dimensional model dried.")

    final_storage = float(np.sum(grid.volumes_m3 * moisture))
    storage_change = final_storage - initial_storage
    residual = storage_change + side_outflow + end_outflow
    scale = max(abs(storage_change), abs(side_outflow + end_outflow), 1.0e-30)
    return SimulationResult2D(
        grid=grid,
        times_s=np.asarray(record_times),
        midplane_temperature_c=np.vstack(midplane_temperature_records),
        midplane_moisture_kg_kg=np.vstack(midplane_moisture_records),
        snapshot_times_s=np.asarray(snapshot_times),
        temperature_snapshots_c=np.stack(temperature_snapshots),
        moisture_snapshots_kg_kg=np.stack(moisture_snapshots),
        drying_time_s=time_s,
        previous_time_s=previous_time,
        previous_maximum_moisture=previous_maximum,
        final_maximum_moisture=max_value,
        controlling_radius_cm=controlling_r,
        controlling_axial_cm=controlling_z,
        controlling_location=controlling_location,
        integrated_side_outflow=side_outflow,
        integrated_end_outflow=end_outflow,
        storage_change=storage_change,
        balance_residual=residual,
        balance_relative_residual=abs(residual) / scale,
        maximum_picard_iterations=maximum_iterations_used,
        time_step_counts=time_step_counts,
        runtime_s=time.perf_counter() - started,
    )


def _exact_record_rows(result: SimulationResult2D, query_times_s: np.ndarray) -> np.ndarray:
    lookup = {round(float(t), 7): i for i, t in enumerate(result.times_s)}
    return np.vstack(
        [result.midplane_moisture_kg_kg[lookup[round(float(t), 7)]] for t in query_times_s]
    )


def table_times(drying_time_s: float) -> np.ndarray:
    regular = np.arange(TABLE_INTERVAL_S, drying_time_s - 1.0e-9, TABLE_INTERVAL_S)
    return np.concatenate((regular, [drying_time_s]))


def write_matrix_csv(
    path: Path, times_s: np.ndarray, values: np.ndarray, radii_cm: np.ndarray
) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_s", *[f"r_{r:.1f}_cm" for r in radii_cm]])
        for t, row in zip(times_s, values, strict=True):
            writer.writerow([f"{t:.6f}", *[f"{value:.10f}" for value in row]])


def write_table5(path: Path, result: SimulationResult2D) -> tuple[np.ndarray, np.ndarray]:
    times = table_times(result.drying_time_s)
    full_rows = _exact_record_rows(result, times)
    radius_indices = [int(np.argmin(np.abs(OUTPUT_RADII_CM - r))) for r in TABLE_RADII_CM]
    rows = full_rows[:, radius_indices]
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["时间/h", *[f"{r:g} cm" for r in TABLE_RADII_CM]])
        for index, (t, row) in enumerate(zip(times, rows, strict=True)):
            label = "烘干结束时间" if index == len(times) - 1 else f"{t / 3600.0:g}"
            writer.writerow([label, *[f"{value:.4f}" for value in row]])
    return times, rows


def write_axial_table(path: Path, result: SimulationResult2D) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_h", *[f"z_{z:.1f}_cm" for z in AXIAL_OUTPUT_CM]])
        for t, field in zip(
            result.snapshot_times_s, result.moisture_snapshots_kg_kg, strict=True
        ):
            # 端面边界值按当前快照的环境和局部扩散系数重构。
            _, oven_moisture = _GLOBAL_BOUNDARY.values(float(t))
            temperature = result.temperature_snapshots_c[
                int(np.where(result.snapshot_times_s == t)[0][0])
            ]
            diffusivity = q2.moisture_diffusivity_m2_s(field, temperature)
            end_surface = _surface_values(
                field[-1, :],
                oven_moisture,
                diffusivity[-1, :],
                q2.MASS_TRANSFER_COEFFICIENT_M_S,
                HALF_LENGTH_M - result.grid.axial_positions_m[-1],
            )
            row = sample_axis_axial(result.grid, field, end_surface, AXIAL_OUTPUT_CM)
            writer.writerow([f"{t / 3600.0:.6f}", *[f"{value:.10f}" for value in row]])


def load_one_dimensional_reference(repo_root: Path) -> tuple[float | None, dict[float, np.ndarray]]:
    validation_path = repo_root / "results" / "A_problem3_drying_time" / "validation_summary.json"
    table_path = repo_root / "results" / "A_problem3_drying_time" / "table5_moisture.csv"
    drying_time = None
    if validation_path.exists():
        drying_time = float(json.loads(validation_path.read_text(encoding="utf-8"))["drying_time"]["production_s"])
    rows: dict[float, np.ndarray] = {}
    if table_path.exists():
        with table_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            next(reader)
            for row in reader:
                try:
                    time_h = float(row[0])
                except ValueError:
                    continue
                rows[time_h] = np.asarray([float(value) for value in row[1:]], dtype=float)
    return drying_time, rows


def matching_one_dimensional_rows(
    result: q3.LongDryingResult,
) -> dict[float, np.ndarray]:
    """提取同网格、同时间步一维基线的6 h径向结果。"""
    radius_indices = [
        int(np.argmin(np.abs(q3.OUTPUT_RADII_CM - radius)))
        for radius in TABLE_RADII_CM
    ]
    rows: dict[float, np.ndarray] = {}
    for time_s, row in zip(result.times_s, result.moisture_kg_kg, strict=True):
        time_h = time_s / 3600.0
        if math.isclose(time_h / 6.0, round(time_h / 6.0), abs_tol=1.0e-10):
            rows[time_h] = row[radius_indices]
    return rows


def run_matching_one_dimensional(
    boundary: q3.BoundaryProgram,
    nr: int,
    early_step_s: float,
    late_step_s: float,
    final_step_s: float,
    refinement_margin: float,
) -> q3.LongDryingResult:
    """运行与二维模型具有相同推进步长的一维基线。"""
    if not math.isclose(
        TABLE_INTERVAL_S / late_step_s,
        round(TABLE_INTERVAL_S / late_step_s),
        abs_tol=1.0e-10,
    ):
        raise ValueError("late-step must divide the 6 h comparison interval.")
    original_output_interval = q3.OUTPUT_INTERVAL_S
    try:
        # 原一维程序为满足60 s输出会强制把大步长截成60 s；匹配基线只需
        # 6 h对照值，故令记录间隔等于二维后期步长，确保时间离散完全相同。
        q3.OUTPUT_INTERVAL_S = late_step_s
        return q3.simulate_until_dry(
            boundary,
            nominal_radial_step_cm=2.0 / nr,
            early_time_step_s=early_step_s,
            late_time_step_s=late_step_s,
            final_time_step_s=final_step_s,
            refinement_margin=refinement_margin,
        )
    finally:
        q3.OUTPUT_INTERVAL_S = original_output_interval


def refresh_existing_comparison(
    output_dir: Path,
    repo_root: Path,
    matching_one_d: q3.LongDryingResult,
    coarse_matching_one_d: q3.LongDryingResult | None = None,
) -> dict:
    """不重跑二维场，仅刷新同时间离散的一维比较结果。"""
    summary_path = output_dir / "summary.json"
    table_path = output_dir / "table5_midplane_moisture.csv"
    if not summary_path.exists() or not table_path.exists():
        raise FileNotFoundError("Existing 2D summary/table files are required.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    two_d_time = float(summary["production"]["drying_time_s"])
    comparison = summary["comparison_with_1d"]
    comparison.update(
        {
            "matched_1d_drying_time_s": matching_one_d.drying_time_s,
            "matched_1d_drying_time_h": matching_one_d.drying_time_s / 3600.0,
            "matched_time_reduction_s": matching_one_d.drying_time_s - two_d_time,
            "matched_time_reduction_h": (
                matching_one_d.drying_time_s - two_d_time
            )
            / 3600.0,
            "matched_time_reduction_percent": 100.0
            * (matching_one_d.drying_time_s - two_d_time)
            / matching_one_d.drying_time_s,
        }
    )
    historical_time = comparison.get("historical_fine_1d_drying_time_s")
    geometry_delta = two_d_time - matching_one_d.drying_time_s
    if historical_time is not None:
        comparison.update(
            {
                "bias_corrected_2d_drying_time_s": historical_time
                + geometry_delta,
                "bias_corrected_2d_drying_time_h": (
                    historical_time + geometry_delta
                )
                / 3600.0,
                "bias_corrected_reduction_vs_fine_1d_s": -geometry_delta,
                "bias_corrected_reduction_vs_fine_1d_percent": 100.0
                * (-geometry_delta)
                / historical_time,
                "bias_correction_method": "fine 1D result plus matched-grid (2D minus 1D) geometry increment",
            }
        )
    if coarse_matching_one_d is not None and "coarse_validation" in summary:
        coarse_2d_time = float(summary["coarse_validation"]["drying_time_s"])
        coarse_reduction = coarse_matching_one_d.drying_time_s - coarse_2d_time
        production_reduction = matching_one_d.drying_time_s - two_d_time
        summary["coarse_validation"].update(
            {
                "matched_1d_drying_time_s": coarse_matching_one_d.drying_time_s,
                "matched_geometry_time_reduction_s": coarse_reduction,
                "geometry_reduction_difference_vs_production_s": coarse_reduction
                - production_reduction,
            }
        )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    times: list[float] = []
    rows: list[np.ndarray] = []
    full_path = output_dir / "midplane_moisture_60s_0p1cm.csv"
    with full_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        next(reader)
        for row in reader:
            time_s = float(row[0])
            if time_s < two_d_time and math.isclose(
                time_s / TABLE_INTERVAL_S,
                round(time_s / TABLE_INTERVAL_S),
                abs_tol=1.0e-10,
            ):
                full_row = np.asarray([float(value) for value in row[1:]], dtype=float)
                radius_indices = [
                    int(np.argmin(np.abs(OUTPUT_RADII_CM - radius)))
                    for radius in TABLE_RADII_CM
                ]
                times.append(time_s)
                rows.append(full_row[radius_indices])
    write_comparison_csv(
        output_dir / "comparison_1d_2d_midplane.csv",
        np.asarray([*times, two_d_time]),
        np.vstack([*rows, rows[-1]]),
        matching_one_dimensional_rows(matching_one_d),
    )
    return summary


def insulated_end_reduction_check(
    boundary: q3.BoundaryProgram, nr: int, nz: int
) -> dict[str, float]:
    """端面零通量时，二维单步应退化为同网格一维解。"""
    grid_2d = make_grid(nr, nz)
    grid_1d = q2.make_grid(2.0 / nr)
    temperature_2d = np.full((nz, nr), q2.INITIAL_TEMPERATURE_C)
    moisture_2d = np.full((nz, nr), q2.INITIAL_MOISTURE_KG_KG)
    oven_temperature, oven_moisture = boundary.values(60.0)
    step_2d = advance_coupled_step_2d(
        grid_2d,
        temperature_2d,
        moisture_2d,
        oven_temperature,
        oven_moisture,
        60.0,
        end_heat_transfer_coefficient=0.0,
        end_mass_transfer_coefficient=0.0,
    )
    step_1d = q3.advance_coupled_step(
        grid_1d,
        np.full(nr, q2.INITIAL_TEMPERATURE_C),
        np.full(nr, q2.INITIAL_MOISTURE_KG_KG),
        oven_temperature,
        oven_moisture,
        60.0,
    )
    return {
        "temperature_max_abs_error_c": float(
            np.max(np.abs(step_2d.temperature_c - step_1d.temperature_c[None, :]))
        ),
        "moisture_max_abs_error_kg_kg": float(
            np.max(np.abs(step_2d.moisture_kg_kg - step_1d.moisture_kg_kg[None, :]))
        ),
    }


def write_comparison_csv(
    path: Path,
    table_times_s: np.ndarray,
    table_rows: np.ndarray,
    one_d_rows: dict[float, np.ndarray],
) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "time_h",
                "radius_cm",
                "moisture_1d_kg_kg",
                "moisture_2d_midplane_kg_kg",
                "difference_2d_minus_1d",
            ]
        )
        for t, row_2d in zip(table_times_s[:-1], table_rows[:-1], strict=True):
            time_h = t / 3600.0
            if time_h not in one_d_rows:
                continue
            for radius, value_1d, value_2d in zip(
                TABLE_RADII_CM, one_d_rows[time_h], row_2d, strict=True
            ):
                writer.writerow(
                    [
                        f"{time_h:g}",
                        f"{radius:g}",
                        f"{value_1d:.10f}",
                        f"{value_2d:.10f}",
                        f"{value_2d - value_1d:.10f}",
                    ]
                )


def build_summary(
    result: SimulationResult2D,
    coarse: SimulationResult2D | None,
    boundary: q3.BoundaryProgram,
    one_d_drying_time_s: float | None,
    matching_one_d: q3.LongDryingResult,
    coarse_matching_one_d: q3.LongDryingResult | None,
    reduction_check: dict[str, float],
) -> dict:
    side_fraction = result.integrated_side_outflow / (
        result.integrated_side_outflow + result.integrated_end_outflow
    )
    summary: dict = {
        "model_scope": {
            "geometry": "fixed finite cylinder, axisymmetric 2D half-length domain",
            "radius_m": q2.RADIUS_M,
            "length_m": LENGTH_M,
            "end_faces": "both fully exposed; half-domain uses midplane symmetry",
            "side_heat_transfer_w_m2_k": q2.HEAT_TRANSFER_COEFFICIENT_W_M2_K,
            "end_heat_transfer_w_m2_k": q2.HEAT_TRANSFER_COEFFICIENT_W_M2_K,
            "side_mass_transfer_m_s": q2.MASS_TRANSFER_COEFFICIENT_M_S,
            "end_mass_transfer_m_s": q2.MASS_TRANSFER_COEFFICIENT_M_S,
            "latent_heat": "excluded",
            "shrinkage": "excluded in question 3",
            "drying_threshold_kg_kg": DRYING_THRESHOLD_KG_KG,
            "official_radial_output": "midplane z=0, the conservative cross-section",
        },
        "constant_boundary": {
            "temperature_c": boundary.plateau_temperature_c,
            "moisture_kg_kg": boundary.plateau_moisture_kg_kg,
        },
        "production": {
            "nr": result.grid.nr,
            "nz_half_length": result.grid.nz,
            "cell_count": result.grid.nr * result.grid.nz,
            "drying_time_s": result.drying_time_s,
            "drying_time_h": result.drying_time_s / 3600.0,
            "previous_time_s": result.previous_time_s,
            "previous_maximum_moisture": result.previous_maximum_moisture,
            "final_maximum_moisture": result.final_maximum_moisture,
            "controlling_radius_cm": result.controlling_radius_cm,
            "controlling_axial_distance_from_midplane_cm": result.controlling_axial_cm,
            "controlling_location": result.controlling_location,
            "maximum_picard_iterations": result.maximum_picard_iterations,
            "time_step_counts": result.time_step_counts,
            "runtime_s": result.runtime_s,
        },
        "comparison_with_1d": {
            "historical_fine_1d_drying_time_s": one_d_drying_time_s,
            "historical_fine_1d_drying_time_h": None
            if one_d_drying_time_s is None
            else one_d_drying_time_s / 3600.0,
            "matched_1d_drying_time_s": matching_one_d.drying_time_s,
            "matched_1d_drying_time_h": matching_one_d.drying_time_s / 3600.0,
            "matched_time_reduction_s": matching_one_d.drying_time_s
            - result.drying_time_s,
            "matched_time_reduction_h": (
                matching_one_d.drying_time_s - result.drying_time_s
            )
            / 3600.0,
            "matched_time_reduction_percent": 100.0
            * (matching_one_d.drying_time_s - result.drying_time_s)
            / matching_one_d.drying_time_s,
        },
        "half_domain_moisture_balance": {
            "storage_change": result.storage_change,
            "integrated_side_outflow": result.integrated_side_outflow,
            "integrated_end_outflow": result.integrated_end_outflow,
            "side_outflow_fraction": side_fraction,
            "end_outflow_fraction": 1.0 - side_fraction,
            "residual": result.balance_residual,
            "relative_residual": result.balance_relative_residual,
        },
        "checks": {
            "previous_state_not_dry": result.previous_maximum_moisture
            >= DRYING_THRESHOLD_KG_KG,
            "final_state_all_dry": result.final_maximum_moisture
            < DRYING_THRESHOLD_KG_KG,
            "controlling_point_is_midplane_axis": abs(result.controlling_radius_cm) < 1.0e-12
            and abs(result.controlling_axial_cm) < 1.0e-12,
            "saved_midplane_time_monotonicity_violation_count": int(
                np.sum(np.diff(result.midplane_moisture_kg_kg, axis=0) > 1.0e-9)
            ),
            "minimum_saved_moisture": float(np.min(result.midplane_moisture_kg_kg)),
            "insulated_end_reduction": reduction_check,
        },
    }
    if coarse is not None:
        summary["coarse_validation"] = {
            "nr": coarse.grid.nr,
            "nz_half_length": coarse.grid.nz,
            "drying_time_s": coarse.drying_time_s,
            "drying_time_h": coarse.drying_time_s / 3600.0,
            "coarse_minus_production_s": coarse.drying_time_s - result.drying_time_s,
            "coarse_minus_production_percent": 100.0
            * (coarse.drying_time_s - result.drying_time_s)
            / result.drying_time_s,
            "relative_moisture_balance_residual": coarse.balance_relative_residual,
        }
        if coarse_matching_one_d is not None:
            coarse_reduction = coarse_matching_one_d.drying_time_s - coarse.drying_time_s
            production_reduction = matching_one_d.drying_time_s - result.drying_time_s
            summary["coarse_validation"].update(
                {
                    "matched_1d_drying_time_s": coarse_matching_one_d.drying_time_s,
                    "matched_geometry_time_reduction_s": coarse_reduction,
                    "geometry_reduction_difference_vs_production_s": coarse_reduction
                    - production_reduction,
                }
            )
    geometry_delta = result.drying_time_s - matching_one_d.drying_time_s
    if one_d_drying_time_s is not None:
        summary["comparison_with_1d"].update(
            {
                "bias_corrected_2d_drying_time_s": one_d_drying_time_s
                + geometry_delta,
                "bias_corrected_2d_drying_time_h": (
                    one_d_drying_time_s + geometry_delta
                )
                / 3600.0,
                "bias_corrected_reduction_vs_fine_1d_s": -geometry_delta,
                "bias_corrected_reduction_vs_fine_1d_percent": 100.0
                * (-geometry_delta)
                / one_d_drying_time_s,
                "bias_correction_method": "fine 1D result plus matched-grid (2D minus 1D) geometry increment",
            }
        )
    return summary


def write_outputs(
    output_dir: Path,
    result: SimulationResult2D,
    coarse: SimulationResult2D | None,
    boundary: q3.BoundaryProgram,
    repo_root: Path,
    matching_one_d: q3.LongDryingResult,
    coarse_matching_one_d: q3.LongDryingResult | None,
    reduction_check: dict[str, float],
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_matrix_csv(
        output_dir / "midplane_moisture_60s_0p1cm.csv",
        result.times_s,
        result.midplane_moisture_kg_kg,
        OUTPUT_RADII_CM,
    )
    write_matrix_csv(
        output_dir / "midplane_temperature_60s_0p1cm.csv",
        result.times_s,
        result.midplane_temperature_c,
        OUTPUT_RADII_CM,
    )
    table_time_values, table_rows = write_table5(
        output_dir / "table5_midplane_moisture.csv", result
    )
    write_axial_table(output_dir / "axis_axial_moisture_6h.csv", result)
    np.savez_compressed(
        output_dir / "fields_2d_6h_and_final.npz",
        times_s=result.snapshot_times_s,
        radii_m=result.grid.radii_m,
        axial_positions_m=result.grid.axial_positions_m,
        temperature_c=result.temperature_snapshots_c,
        moisture_kg_kg=result.moisture_snapshots_kg_kg,
    )
    one_d_time, _ = load_one_dimensional_reference(repo_root)
    one_d_rows = matching_one_dimensional_rows(matching_one_d)
    write_comparison_csv(
        output_dir / "comparison_1d_2d_midplane.csv",
        table_time_values,
        table_rows,
        one_d_rows,
    )
    summary = build_summary(
        result,
        coarse,
        boundary,
        one_d_time,
        matching_one_d,
        coarse_matching_one_d,
        reduction_check,
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


_GLOBAL_BOUNDARY: q3.BoundaryProgram


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nr", type=int, default=80)
    parser.add_argument("--nz", type=int, default=40)
    parser.add_argument("--early-step", type=float, default=30.0)
    parser.add_argument("--late-step", type=float, default=120.0)
    parser.add_argument("--final-step", type=float, default=1.0)
    parser.add_argument("--refinement-margin", type=float, default=5.0e-5)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--refresh-comparison-only", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    environment = q2.load_and_preprocess_environment(q2.find_default_data_path(repo_root))
    boundary = q3.build_boundary_program(environment)
    global _GLOBAL_BOUNDARY
    _GLOBAL_BOUNDARY = boundary

    if args.refresh_comparison_only:
        output_dir = args.output_dir or repo_root / "result" / "A_problem3_2d_exposed"
        matching_one_d = run_matching_one_dimensional(
            boundary,
            nr=args.nr,
            early_step_s=args.early_step,
            late_step_s=args.late_step,
            final_step_s=args.final_step,
            refinement_margin=args.refinement_margin,
        )
        existing_summary = json.loads(
            (output_dir / "summary.json").read_text(encoding="utf-8")
        )
        coarse_matching_one_d = None
        if "coarse_validation" in existing_summary:
            coarse_matching_one_d = run_matching_one_dimensional(
                boundary,
                nr=int(existing_summary["coarse_validation"]["nr"]),
                early_step_s=2.0 * args.early_step,
                late_step_s=2.0 * args.late_step,
                final_step_s=2.0 * args.final_step,
                refinement_margin=2.0 * args.refinement_margin,
            )
        summary = refresh_existing_comparison(
            output_dir, repo_root, matching_one_d, coarse_matching_one_d
        )
        print(json.dumps(summary["comparison_with_1d"], ensure_ascii=False, indent=2))
        return

    if args.quick:
        result = simulate_until_dry_2d(
            boundary,
            nr=12,
            nz=10,
            early_step_s=120.0,
            late_step_s=600.0,
            final_step_s=10.0,
            refinement_margin=5.0e-4,
            maximum_time_s=q3.MAXIMUM_SIMULATION_TIME_S,
        )
        print(json.dumps({"quick_drying_time_s": result.drying_time_s}, indent=2))
        return

    result = simulate_until_dry_2d(
        boundary,
        nr=args.nr,
        nz=args.nz,
        early_step_s=args.early_step,
        late_step_s=args.late_step,
        final_step_s=args.final_step,
        refinement_margin=args.refinement_margin,
    )
    matching_one_d = run_matching_one_dimensional(
        boundary,
        nr=args.nr,
        early_step_s=args.early_step,
        late_step_s=args.late_step,
        final_step_s=args.final_step,
        refinement_margin=args.refinement_margin,
    )
    reduction_check = insulated_end_reduction_check(
        boundary, nr=min(args.nr, 40), nz=min(args.nz, 20)
    )
    coarse = None
    coarse_matching_one_d = None
    if not args.skip_validation:
        coarse = simulate_until_dry_2d(
            boundary,
            nr=max(20, args.nr // 2),
            nz=max(16, args.nz // 2),
            early_step_s=2.0 * args.early_step,
            late_step_s=2.0 * args.late_step,
            final_step_s=2.0 * args.final_step,
            refinement_margin=2.0 * args.refinement_margin,
        )
        coarse_matching_one_d = run_matching_one_dimensional(
            boundary,
            nr=max(20, args.nr // 2),
            early_step_s=2.0 * args.early_step,
            late_step_s=2.0 * args.late_step,
            final_step_s=2.0 * args.final_step,
            refinement_margin=2.0 * args.refinement_margin,
        )
    output_dir = args.output_dir or repo_root / "result" / "A_problem3_2d_exposed"
    summary = write_outputs(
        output_dir,
        result,
        coarse,
        boundary,
        repo_root,
        matching_one_d,
        coarse_matching_one_d,
        reduction_check,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
