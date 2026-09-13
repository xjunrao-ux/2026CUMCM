"""A题问题4：计入收缩和表面蒸发潜热的移动半径有限体积模型。

本程序以 ``a4_shrinkage_fvm.py`` 为基准，保留附件2半径的分段线性插值、
材料坐标 xi=r/R(t)、固定物理距离输出规则和严格干燥判据。热物性与水分
扩散系数全部采用A题附录4（而不是问题3的附录3）：

    rho = 760 + 90 C
    cp  = 1850 + 2150 C/(C+1)
    k   = 0.12 + 0.20 C/(C+1)
    D   = 4.2e-4 exp(-0.30/C) exp(-3850/T), T in K

题给 h_m=8e-7 m/s 保持固定。模型只在移动表面的能量平衡中加入
L_v*j_w，不再添加体积潜热源；其中 rho_d=(760+90*C_s)/(1+C_s)
在每次 Picard 迭代中更新。4 h后烘房温度和水分浓度仍取附件1中
3--4 h的时间加权均值。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import a2_coupled_fvm as q2
import a4_shrinkage_fvm as base


LATENT_HEAT_INTERCEPT_KJ_KG = 2500.8
LATENT_HEAT_SLOPE_KJ_KG_K = 2.36

# 初态值只用于结果核对；迭代中必须按当前表面含水率更新 rho_d。
INITIAL_DRY_BULK_DENSITY_KG_M3 = (
    760.0 + 90.0 * base.INITIAL_MOISTURE_KG_KG
) / (1.0 + base.INITIAL_MOISTURE_KG_KG)


@dataclass
class SteamStepResult:
    temperature_k: np.ndarray
    moisture_kg_kg: np.ndarray
    surface_temperature_k: float
    surface_moisture_kg_kg: float
    moisture_boundary_conductance: float
    moisture_storage_change: float
    minimum_diffusivity_m2_s: float
    maximum_diffusivity_m2_s: float
    surface_dry_bulk_density_kg_m3: float
    water_mass_flux_kg_m2_s: float
    latent_heat_j_kg: float
    latent_heat_flux_w_m2: float
    convective_heat_flux_w_m2: float
    conductive_heat_flux_w_m2: float
    surface_energy_residual_w_m2: float
    sensible_energy_change_j_m: float
    storage_balance_residual_j_m: float
    iterations: int


@dataclass
class SteamShrinkageResult(base.ShrinkageResult):
    oven_temperature_k: np.ndarray
    oven_moisture_kg_kg: np.ndarray
    surface_dry_bulk_density_kg_m3: np.ndarray
    water_mass_flux_kg_m2_s: np.ndarray
    latent_heat_j_kg: np.ndarray
    latent_heat_flux_w_m2: np.ndarray
    convective_heat_flux_w_m2: np.ndarray
    conductive_heat_flux_w_m2: np.ndarray
    surface_energy_residual_w_m2: np.ndarray
    integrated_convective_heat_j_m: float
    integrated_conductive_heat_j_m: float
    integrated_latent_heat_j_m: float
    accumulated_sensible_energy_change_j_m: float
    maximum_storage_balance_residual_j_m: float
    maximum_surface_energy_residual_w_m2: float
    integrated_physical_water_outflow_kg_m: float


def dry_bulk_density_kg_m3(moisture_kg_kg: float | np.ndarray) -> float | np.ndarray:
    """Appendix 4 local dry-bulk density rho_d=rho(C)/(1+C)."""
    moisture = np.asarray(moisture_kg_kg, dtype=float)
    if np.any(moisture <= -1.0):
        raise ValueError("Moisture must be greater than -1 kg/kg.")
    density = base.density_kg_m3(moisture) / (1.0 + moisture)
    if density.ndim == 0:
        return float(density)
    return density


def latent_heat_of_vaporization_j_kg(surface_temperature_k: float) -> float:
    surface_temperature_c = surface_temperature_k - 273.15
    value = 1000.0 * (
        LATENT_HEAT_INTERCEPT_KJ_KG
        - LATENT_HEAT_SLOPE_KJ_KG_K * surface_temperature_c
    )
    if value <= 0.0:
        raise ValueError("The latent heat correlation returned a non-positive value.")
    return value


def surface_temperature_with_evaporation(
    outer_cell_temperature_k: float,
    oven_temperature_k: float,
    surface_conductivity_w_m_k: float,
    surface_distance_m: float,
    latent_heat_flux_w_m2: float,
) -> float:
    """Solve k/delta*(T_s-T_P)=h_T*(T_inf-T_s)-q_lat."""
    cell_side = surface_conductivity_w_m_k / surface_distance_m
    return (
        cell_side * outer_cell_temperature_k
        + base.HEAT_TRANSFER_COEFFICIENT_W_M2_K * oven_temperature_k
        - latent_heat_flux_w_m2
    ) / (cell_side + base.HEAT_TRANSFER_COEFFICIENT_W_M2_K)


def advance_coupled_step(
    material_grid: base.MaterialGrid,
    radius_m: float,
    temperature_previous_k: np.ndarray,
    moisture_previous: np.ndarray,
    surface_temperature_previous_k: float,
    oven_temperature_k: float,
    oven_moisture: float,
    time_step_s: float,
    temperature_tolerance_k: float = 1.0e-8,
    moisture_tolerance: float = 1.0e-10,
    surface_temperature_tolerance_k: float = 1.0e-9,
    maximum_iterations: int = 50,
) -> SteamStepResult:
    """Advance one moving-grid step with Appendix-4 properties and latent heat."""
    grid = base.physical_grid(material_grid, radius_m)
    temperature_iterate = temperature_previous_k.copy()
    moisture_iterate = moisture_previous.copy()
    surface_temperature_iterate = float(surface_temperature_previous_k)
    surface_distance = radius_m - grid.radii_m[-1]
    surface_area_per_length = 2.0 * math.pi * radius_m

    for iteration in range(1, maximum_iterations + 1):
        # 先按附录4的D(C,T)求水分场及表面水分，再计算蒸发潜热通量。
        diffusivity = base.moisture_diffusivity_m2_s(
            moisture_iterate, temperature_iterate
        )
        moisture_system = base.build_dynamic_diffusion_system(
            grid,
            radius_m,
            moisture_previous,
            diffusivity,
            np.ones_like(moisture_iterate),
            base.MASS_TRANSFER_COEFFICIENT_M_S,
            oven_moisture,
            time_step_s,
        )
        moisture_updated = q2.solve_tridiagonal(*moisture_system[:4])
        surface_moisture = q2.reconstruct_surface_value(
            float(moisture_updated[-1]),
            oven_moisture,
            float(diffusivity[-1]),
            base.MASS_TRANSFER_COEFFICIENT_M_S,
            surface_distance,
        )
        surface_dry_bulk_density = dry_bulk_density_kg_m3(surface_moisture)
        water_mass_flux = (
            surface_dry_bulk_density
            * base.MASS_TRANSFER_COEFFICIENT_M_S
            * (surface_moisture - oven_moisture)
        )
        latent_heat = latent_heat_of_vaporization_j_kg(
            surface_temperature_iterate
        )
        latent_heat_flux = latent_heat * water_mass_flux

        # 热方程严格采用附录4的rho、cp和k。
        conductivity = base.conductivity_w_m_k(moisture_updated)
        heat_storage = base.density_kg_m3(
            moisture_updated
        ) * base.heat_capacity_j_kg_k(moisture_updated)
        lower, diagonal, upper, rhs, _ = base.build_dynamic_diffusion_system(
            grid,
            radius_m,
            temperature_previous_k,
            conductivity,
            heat_storage,
            base.HEAT_TRANSFER_COEFFICIENT_W_M2_K,
            oven_temperature_k,
            time_step_s,
        )
        cell_side = float(conductivity[-1]) / surface_distance
        latent_transfer_fraction = cell_side / (
            cell_side + base.HEAT_TRANSFER_COEFFICIENT_W_M2_K
        )
        rhs[-1] -= (
            surface_area_per_length
            * latent_transfer_fraction
            * latent_heat_flux
        )
        temperature_updated = q2.solve_tridiagonal(
            lower, diagonal, upper, rhs
        )
        surface_temperature_updated = surface_temperature_with_evaporation(
            float(temperature_updated[-1]),
            oven_temperature_k,
            float(conductivity[-1]),
            surface_distance,
            latent_heat_flux,
        )

        temperature_error = float(
            np.max(np.abs(temperature_updated - temperature_iterate))
        )
        moisture_error = float(
            np.max(np.abs(moisture_updated - moisture_iterate))
        )
        surface_temperature_error = abs(
            surface_temperature_updated - surface_temperature_iterate
        )
        temperature_iterate = temperature_updated
        moisture_iterate = moisture_updated
        surface_temperature_iterate = surface_temperature_updated
        if (
            temperature_error < temperature_tolerance_k
            and moisture_error < moisture_tolerance
            and surface_temperature_error < surface_temperature_tolerance_k
        ):
            break
    else:
        raise RuntimeError("Latent-heat Picard iteration did not converge.")

    conductive_heat_flux = cell_side * (
        surface_temperature_iterate - float(temperature_iterate[-1])
    )
    convective_heat_flux = base.HEAT_TRANSFER_COEFFICIENT_W_M2_K * (
        oven_temperature_k - surface_temperature_iterate
    )
    surface_energy_residual = (
        convective_heat_flux - conductive_heat_flux - latent_heat_flux
    )
    sensible_energy_change = float(
        np.sum(
            heat_storage
            * grid.volumes_m3_m
            * (temperature_iterate - temperature_previous_k)
        )
    )
    conductive_heat_rate_per_length = (
        surface_area_per_length * conductive_heat_flux
    )
    storage_balance_residual = (
        sensible_energy_change
        - time_step_s * conductive_heat_rate_per_length
    )
    moisture_storage_change = float(
        np.sum(
            grid.volumes_m3_m
            * (moisture_iterate - moisture_previous)
        )
    )

    return SteamStepResult(
        temperature_k=temperature_iterate,
        moisture_kg_kg=moisture_iterate,
        surface_temperature_k=surface_temperature_iterate,
        surface_moisture_kg_kg=float(surface_moisture),
        moisture_boundary_conductance=float(moisture_system[4]),
        moisture_storage_change=moisture_storage_change,
        minimum_diffusivity_m2_s=float(np.min(diffusivity)),
        maximum_diffusivity_m2_s=float(np.max(diffusivity)),
        surface_dry_bulk_density_kg_m3=float(surface_dry_bulk_density),
        water_mass_flux_kg_m2_s=float(water_mass_flux),
        latent_heat_j_kg=float(latent_heat),
        latent_heat_flux_w_m2=float(latent_heat_flux),
        convective_heat_flux_w_m2=float(convective_heat_flux),
        conductive_heat_flux_w_m2=float(conductive_heat_flux),
        surface_energy_residual_w_m2=float(surface_energy_residual),
        sensible_energy_change_j_m=sensible_energy_change,
        storage_balance_residual_j_m=float(storage_balance_residual),
        iterations=iteration,
    )


def simulate_until_dry(
    boundary: base.BoundaryProgram,
    radius_program: base.RadiusProgram,
    nominal_initial_step_cm: float,
    early_step_s: float,
    late_step_s: float,
    final_step_s: float,
) -> SteamShrinkageResult:
    material_grid = base.make_material_grid(nominal_initial_step_cm)
    n = material_grid.centers_xi.size
    temperature = np.full(n, base.INITIAL_TEMPERATURE_K, dtype=float)
    moisture = np.full(n, base.INITIAL_MOISTURE_KG_KG, dtype=float)
    surface_temperature = base.INITIAL_TEMPERATURE_K
    surface_moisture = base.INITIAL_MOISTURE_KG_KG
    time_s = 0.0
    next_output_s = base.OUTPUT_INTERVAL_S

    record_times: list[float] = []
    record_radii: list[float] = []
    temperature_records: list[np.ndarray] = []
    moisture_records: list[np.ndarray] = []
    surface_temperature_records: list[float] = []
    surface_moisture_records: list[float] = []
    oven_temperature_records: list[float] = []
    oven_moisture_records: list[float] = []
    surface_dry_bulk_density_records: list[float] = []
    water_mass_flux_records: list[float] = []
    latent_heat_records: list[float] = []
    latent_heat_flux_records: list[float] = []
    convective_heat_flux_records: list[float] = []
    conductive_heat_flux_records: list[float] = []
    surface_energy_residual_records: list[float] = []

    maximum_moisture = base.INITIAL_MOISTURE_KG_KG
    controlling_radius_cm = 0.0
    previous_time_s = 0.0
    previous_maximum = maximum_moisture
    maximum_picard_iterations = 0
    time_step_counts = {
        "measured_environment": 0,
        "constant_environment": 0,
        "threshold_refinement": 0,
        "crossing_recomputed": 0,
    }
    discrete_storage_change = 0.0
    integrated_outflow = 0.0
    minimum_diffusivity = math.inf
    maximum_diffusivity = 0.0
    integrated_convective_heat = 0.0
    integrated_conductive_heat = 0.0
    integrated_latent_heat = 0.0
    accumulated_sensible_energy_change = 0.0
    maximum_storage_balance_residual = 0.0
    maximum_surface_energy_residual = 0.0
    integrated_physical_water_outflow = 0.0
    forced_final_resolution = False

    while time_s < base.MAXIMUM_SIMULATION_TIME_S:
        if forced_final_resolution:
            proposed_step = final_step_s
            step_label = "threshold_refinement"
        else:
            proposed_step, step_label = base.choose_time_step(
                time_s,
                maximum_moisture,
                early_step_s,
                late_step_s,
                final_step_s,
            )
        time_step_s = proposed_step
        if (
            time_s < base.MEASURED_ENVIRONMENT_END_S
            < time_s + time_step_s
        ):
            time_step_s = base.MEASURED_ENVIRONMENT_END_S - time_s
        if time_s < next_output_s < time_s + time_step_s:
            time_step_s = next_output_s - time_s
        if time_s + time_step_s > base.MAXIMUM_SIMULATION_TIME_S:
            time_step_s = base.MAXIMUM_SIMULATION_TIME_S - time_s
        if time_step_s <= 0.0:
            raise RuntimeError("Non-positive adaptive time step encountered.")

        new_time_s = time_s + time_step_s
        new_radius_m = radius_program.radius_m(new_time_s)
        oven_temperature_k, oven_moisture = boundary.values(new_time_s)
        step = advance_coupled_step(
            material_grid,
            new_radius_m,
            temperature,
            moisture,
            surface_temperature,
            oven_temperature_k,
            oven_moisture,
            time_step_s,
        )
        new_maximum, new_controlling_radius = base.maximum_with_location(
            material_grid,
            new_radius_m,
            step.moisture_kg_kg,
            step.surface_moisture_kg_kg,
        )
        if new_maximum < base.DRYING_THRESHOLD_KG_KG and time_step_s > final_step_s:
            forced_final_resolution = True
            time_step_counts["crossing_recomputed"] += 1
            continue

        surface_area_per_length = 2.0 * math.pi * new_radius_m
        discrete_storage_change += step.moisture_storage_change
        integrated_outflow += (
            time_step_s
            * step.moisture_boundary_conductance
            * (step.moisture_kg_kg[-1] - oven_moisture)
        )
        integrated_physical_water_outflow += (
            time_step_s
            * surface_area_per_length
            * step.water_mass_flux_kg_m2_s
        )
        integrated_convective_heat += (
            time_step_s
            * surface_area_per_length
            * step.convective_heat_flux_w_m2
        )
        integrated_conductive_heat += (
            time_step_s
            * surface_area_per_length
            * step.conductive_heat_flux_w_m2
        )
        integrated_latent_heat += (
            time_step_s
            * surface_area_per_length
            * step.latent_heat_flux_w_m2
        )
        accumulated_sensible_energy_change += step.sensible_energy_change_j_m
        maximum_storage_balance_residual = max(
            maximum_storage_balance_residual,
            abs(step.storage_balance_residual_j_m),
        )
        maximum_surface_energy_residual = max(
            maximum_surface_energy_residual,
            abs(step.surface_energy_residual_w_m2),
        )

        temperature = step.temperature_k
        moisture = step.moisture_kg_kg
        surface_temperature = step.surface_temperature_k
        surface_moisture = step.surface_moisture_kg_kg
        time_s = new_time_s
        maximum_moisture = new_maximum
        controlling_radius_cm = new_controlling_radius
        maximum_picard_iterations = max(
            maximum_picard_iterations, step.iterations
        )
        minimum_diffusivity = min(
            minimum_diffusivity, step.minimum_diffusivity_m2_s
        )
        maximum_diffusivity = max(
            maximum_diffusivity, step.maximum_diffusivity_m2_s
        )
        time_step_counts[step_label] += 1

        is_output_time = math.isclose(time_s, next_output_s, abs_tol=1.0e-8)
        is_dry = maximum_moisture < base.DRYING_THRESHOLD_KG_KG
        if is_output_time or is_dry:
            if not record_times or not math.isclose(
                record_times[-1], time_s, abs_tol=1.0e-8
            ):
                current_radius_cm = new_radius_m * 100.0
                record_times.append(time_s)
                record_radii.append(current_radius_cm)
                temperature_records.append(
                    base.sample_physical_field(
                        material_grid,
                        new_radius_m,
                        temperature,
                        surface_temperature,
                        base.OUTPUT_RADII_CM,
                    )
                )
                moisture_records.append(
                    base.sample_physical_field(
                        material_grid,
                        new_radius_m,
                        moisture,
                        surface_moisture,
                        base.OUTPUT_RADII_CM,
                    )
                )
                surface_temperature_records.append(surface_temperature)
                surface_moisture_records.append(surface_moisture)
                oven_temperature_records.append(oven_temperature_k)
                oven_moisture_records.append(oven_moisture)
                surface_dry_bulk_density_records.append(
                    step.surface_dry_bulk_density_kg_m3
                )
                water_mass_flux_records.append(step.water_mass_flux_kg_m2_s)
                latent_heat_records.append(step.latent_heat_j_kg)
                latent_heat_flux_records.append(step.latent_heat_flux_w_m2)
                convective_heat_flux_records.append(
                    step.convective_heat_flux_w_m2
                )
                conductive_heat_flux_records.append(
                    step.conductive_heat_flux_w_m2
                )
                surface_energy_residual_records.append(
                    step.surface_energy_residual_w_m2
                )
        if is_output_time:
            next_output_s += base.OUTPUT_INTERVAL_S

        if is_dry:
            break
        previous_time_s = time_s
        previous_maximum = maximum_moisture
    else:
        raise RuntimeError("Maximum simulation time reached before drying.")

    return SteamShrinkageResult(
        times_s=np.asarray(record_times, dtype=float),
        radii_cm=np.asarray(record_radii, dtype=float),
        temperature_k=np.vstack(temperature_records),
        moisture_kg_kg=np.vstack(moisture_records),
        surface_temperature_k=np.asarray(surface_temperature_records, dtype=float),
        surface_moisture_kg_kg=np.asarray(surface_moisture_records, dtype=float),
        drying_time_s=time_s,
        previous_time_s=previous_time_s,
        previous_maximum_moisture=previous_maximum,
        final_maximum_moisture=maximum_moisture,
        controlling_radius_cm=controlling_radius_cm,
        final_internal_temperature_k=temperature,
        final_internal_moisture_kg_kg=moisture,
        maximum_picard_iterations=maximum_picard_iterations,
        time_step_counts=time_step_counts,
        discrete_moisture_storage_change=discrete_storage_change,
        integrated_moisture_outflow=integrated_outflow,
        minimum_diffusivity_m2_s=minimum_diffusivity,
        maximum_diffusivity_m2_s=maximum_diffusivity,
        oven_temperature_k=np.asarray(oven_temperature_records, dtype=float),
        oven_moisture_kg_kg=np.asarray(oven_moisture_records, dtype=float),
        surface_dry_bulk_density_kg_m3=np.asarray(
            surface_dry_bulk_density_records, dtype=float
        ),
        water_mass_flux_kg_m2_s=np.asarray(water_mass_flux_records, dtype=float),
        latent_heat_j_kg=np.asarray(latent_heat_records, dtype=float),
        latent_heat_flux_w_m2=np.asarray(latent_heat_flux_records, dtype=float),
        convective_heat_flux_w_m2=np.asarray(
            convective_heat_flux_records, dtype=float
        ),
        conductive_heat_flux_w_m2=np.asarray(
            conductive_heat_flux_records, dtype=float
        ),
        surface_energy_residual_w_m2=np.asarray(
            surface_energy_residual_records, dtype=float
        ),
        integrated_convective_heat_j_m=integrated_convective_heat,
        integrated_conductive_heat_j_m=integrated_conductive_heat,
        integrated_latent_heat_j_m=integrated_latent_heat,
        accumulated_sensible_energy_change_j_m=(
            accumulated_sensible_energy_change
        ),
        maximum_storage_balance_residual_j_m=(
            maximum_storage_balance_residual
        ),
        maximum_surface_energy_residual_w_m2=(
            maximum_surface_energy_residual
        ),
        integrated_physical_water_outflow_kg_m=(
            integrated_physical_water_outflow
        ),
    )


def write_full_csv(path: Path, result: SteamShrinkageResult) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "time_s",
                "radius_cm",
                *[f"r_{radius:.1f}_cm" for radius in base.OUTPUT_RADII_CM],
                "surface",
            ]
        )
        for time_s, radius_cm, values, surface in zip(
            result.times_s,
            result.radii_cm,
            result.moisture_kg_kg,
            result.surface_moisture_kg_kg,
        ):
            writer.writerow(
                [
                    f"{time_s:.0f}",
                    f"{radius_cm:.8f}",
                    *[
                        "" if np.isnan(value) else f"{value:.8f}"
                        for value in values
                    ],
                    f"{surface:.8f}",
                ]
            )


def write_temperature_csv(path: Path, result: SteamShrinkageResult) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "time_s",
                "radius_cm",
                *[f"r_{radius:.1f}_cm" for radius in base.OUTPUT_RADII_CM],
                "surface_temperature_K",
            ]
        )
        for time_s, radius_cm, values, surface in zip(
            result.times_s,
            result.radii_cm,
            result.temperature_k,
            result.surface_temperature_k,
        ):
            writer.writerow(
                [
                    f"{time_s:.0f}",
                    f"{radius_cm:.8f}",
                    *[
                        "" if np.isnan(value) else f"{value:.6f}"
                        for value in values
                    ],
                    f"{surface:.6f}",
                ]
            )


def write_workbook_payload(path: Path, result: SteamShrinkageResult) -> None:
    rows: list[list[float | int | None]] = []
    for time_s, values, surface in zip(
        result.times_s, result.moisture_kg_kg, result.surface_moisture_kg_kg
    ):
        rows.append(
            [
                int(round(float(time_s))),
                *[
                    None if np.isnan(value) else round(float(value), 8)
                    for value in values
                ],
                round(float(surface), 8),
            ]
        )
    payload = {
        "headers": [
            "时间\\到药材中心的距离",
            *[round(float(radius), 1) for radius in base.OUTPUT_RADII_CM],
            "药材表面",
        ],
        "rows": rows,
        "drying_time_s": result.drying_time_s,
    }
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"))


def write_surface_energy_csv(path: Path, result: SteamShrinkageResult) -> None:
    headers = [
        "time_s",
        "radius_cm",
        "oven_temperature_C",
        "oven_temperature_K",
        "oven_moisture_kg_kg",
        "surface_temperature_C",
        "surface_temperature_K",
        "surface_moisture_kg_kg",
        "surface_dry_bulk_density_kg_m3",
        "water_mass_flux_kg_m2_s",
        "latent_heat_J_kg",
        "latent_heat_flux_W_m2",
        "convective_heat_flux_W_m2",
        "conductive_heat_flux_W_m2",
        "surface_energy_residual_W_m2",
    ]
    columns = [
        result.times_s,
        result.radii_cm,
        result.oven_temperature_k - 273.15,
        result.oven_temperature_k,
        result.oven_moisture_kg_kg,
        result.surface_temperature_k - 273.15,
        result.surface_temperature_k,
        result.surface_moisture_kg_kg,
        result.surface_dry_bulk_density_kg_m3,
        result.water_mass_flux_kg_m2_s,
        result.latent_heat_j_kg,
        result.latent_heat_flux_w_m2,
        result.convective_heat_flux_w_m2,
        result.conductive_heat_flux_w_m2,
        result.surface_energy_residual_w_m2,
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        for row in zip(*columns):
            writer.writerow(
                [f"{row[0]:.0f}", *[f"{value:.10g}" for value in row[1:]]]
            )


def write_interpolated_inputs_csv(
    path: Path,
    radius_program: base.RadiusProgram,
    boundary: base.BoundaryProgram,
) -> None:
    """Write auditable 4--72 h minute inputs and each radius segment."""
    times = np.arange(
        base.MEASURED_ENVIRONMENT_END_S,
        72.0 * 3600.0 + base.OUTPUT_INTERVAL_S,
        base.OUTPUT_INTERVAL_S,
        dtype=float,
    )
    source_times = radius_program.source_times_s
    source_radii = radius_program.source_radii_cm
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "time_s",
                "time_h",
                "model_radius_cm",
                "oven_temperature_C",
                "oven_temperature_K",
                "oven_moisture_kg_kg",
                "segment_start_s",
                "segment_end_s",
                "segment_start_radius_cm",
                "segment_end_radius_cm",
                "segment_slope_cm_s",
                "data_type",
            ]
        )
        for time_s in times:
            index = int(np.searchsorted(source_times, time_s, side="right") - 1)
            index = min(max(index, 0), source_times.size - 2)
            start_time = float(source_times[index])
            end_time = float(source_times[index + 1])
            start_radius = float(source_radii[index])
            end_radius = float(source_radii[index + 1])
            slope = (end_radius - start_radius) / (end_time - start_time)
            radius = radius_program.radius_cm(time_s)
            oven_temperature_k, oven_moisture = boundary.values(time_s)
            is_source_node = bool(
                np.any(np.isclose(source_times, time_s, atol=1.0e-9))
            )
            writer.writerow(
                [
                    f"{time_s:.0f}",
                    f"{time_s / 3600.0:.6f}",
                    f"{radius:.10f}",
                    f"{oven_temperature_k - 273.15:.10f}",
                    f"{oven_temperature_k:.10f}",
                    f"{oven_moisture:.12f}",
                    f"{start_time:.0f}",
                    f"{end_time:.0f}",
                    f"{start_radius:.10f}",
                    f"{end_radius:.10f}",
                    f"{slope:.14g}",
                    "附件2节点" if is_source_node else "分段线性插值",
                ]
            )


def _reference_constant_mean_drying_time(repo_root: Path) -> float | None:
    path = (
        repo_root
        / "results"
        / "A_problem4_shrinkage"
        / "validation_summary.json"
    )
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    value = payload.get("drying_time", {}).get("production_s")
    return float(value) if value is not None else None


def build_validation(
    repo_root: Path,
    boundary: base.BoundaryProgram,
    radius_program: base.RadiusProgram,
    production: SteamShrinkageResult,
    coarse: SteamShrinkageResult,
) -> dict:
    validation = base.build_validation(
        boundary, radius_program, production, coarse
    )
    validation["model_scope"].update(
        {
            "constitutive_laws": (
                "Appendix 4 only: rho=760+90C; cp=1850+2150C/(C+1); "
                "k=0.12+0.20C/(C+1); "
                "D=4.2e-4*exp(-0.30/C)*exp(-3850/T)"
            ),
            "latent_heat": (
                "included once as the moving-surface sink L_v*j_w; "
                "no volumetric latent-heat source"
            ),
            "surface_energy_balance": (
                "h_T(T_inf-T_s)=k*dT/dr|R(t)+L_v*j_w"
            ),
            "radius_interpolation_change": (
                "none; the original piecewise-linear Attachment 2 algorithm is retained"
            ),
            "surface_dry_bulk_density": (
                "dynamic Appendix-4 rho_d,s=(760+90*C_s)/(1+C_s), "
                "updated in every Picard iteration"
            ),
            "diffusivity_temperature_unit": (
                "absolute temperature in kelvin for exp(-3850/T_K)"
            ),
        }
    )
    validation["fixed_parameters"] = {
        "heat_transfer_coefficient_W_m2_K": (
            base.HEAT_TRANSFER_COEFFICIENT_W_M2_K
        ),
        "mass_transfer_coefficient_m_s": (
            base.MASS_TRANSFER_COEFFICIENT_M_S
        ),
        "initial_dry_bulk_density_kg_m3_for_reference": (
            INITIAL_DRY_BULK_DENSITY_KG_M3
        ),
        "latent_heat_correlation": (
            "L_v=[2500.8-2.36*T_s(C)]*1000 J/kg"
        ),
    }

    production_grid = base.make_material_grid(0.00625)
    final_radius_m = radius_program.radius_m(production.drying_time_s)
    final_grid = base.physical_grid(production_grid, final_radius_m)
    empirical_dry_inventory_initial = (
        INITIAL_DRY_BULK_DENSITY_KG_M3
        * math.pi
        * base.INITIAL_RADIUS_M**2
    )
    empirical_dry_inventory_final = float(
        np.sum(
            dry_bulk_density_kg_m3(production.final_internal_moisture_kg_kg)
            * final_grid.volumes_m3_m
        )
    )
    validation["dynamic_surface_density"] = {
        "formula": "rho_d,s=(760+90*C_s)/(1+C_s) kg/m3",
        "initial_reference_kg_m3": INITIAL_DRY_BULK_DENSITY_KG_M3,
        "minimum_saved_surface_kg_m3": float(
            np.min(production.surface_dry_bulk_density_kg_m3)
        ),
        "maximum_saved_surface_kg_m3": float(
            np.max(production.surface_dry_bulk_density_kg_m3)
        ),
        "integrated_surface_water_outflow_kg_m": (
            production.integrated_physical_water_outflow_kg_m
        ),
        "empirical_dry_mass_inventory_initial_kg_m": (
            empirical_dry_inventory_initial
        ),
        "empirical_dry_mass_inventory_final_kg_m": empirical_dry_inventory_final,
        "empirical_inventory_relative_change": (
            empirical_dry_inventory_final / empirical_dry_inventory_initial - 1.0
        ),
        "interpretation": (
            "rho_d is the requested Appendix-4 local conversion in the latent flux; "
            "the normalized moving-grid Fick balance remains the governing "
            "moisture-conservation check."
        ),
    }

    surface_integrated_residual = (
        production.integrated_convective_heat_j_m
        - production.integrated_conductive_heat_j_m
        - production.integrated_latent_heat_j_m
    )
    storage_integrated_residual = (
        production.accumulated_sensible_energy_change_j_m
        - production.integrated_conductive_heat_j_m
    )
    surface_scale = max(
        abs(production.integrated_convective_heat_j_m),
        abs(production.integrated_conductive_heat_j_m),
        abs(production.integrated_latent_heat_j_m),
        1.0e-30,
    )
    storage_scale = max(
        abs(production.accumulated_sensible_energy_change_j_m),
        abs(production.integrated_conductive_heat_j_m),
        1.0e-30,
    )
    validation["energy_balance_per_unit_length"] = {
        "integrated_convective_input_J_m": (
            production.integrated_convective_heat_j_m
        ),
        "integrated_latent_heat_J_m": production.integrated_latent_heat_j_m,
        "integrated_conductive_heat_into_solid_J_m": (
            production.integrated_conductive_heat_j_m
        ),
        "accumulated_sensible_energy_change_J_m": (
            production.accumulated_sensible_energy_change_j_m
        ),
        "surface_integrated_residual_J_m": surface_integrated_residual,
        "surface_integrated_relative_residual": (
            abs(surface_integrated_residual) / surface_scale
        ),
        "storage_integrated_residual_J_m": storage_integrated_residual,
        "storage_integrated_relative_residual": (
            abs(storage_integrated_residual) / storage_scale
        ),
        "maximum_step_surface_residual_W_m2": (
            production.maximum_surface_energy_residual_w_m2
        ),
        "maximum_step_storage_residual_J_m": (
            production.maximum_storage_balance_residual_j_m
        ),
    }
    validation["physical_ranges"].update(
        {
            "surface_temperature_min_K": float(
                np.min(production.surface_temperature_k)
            ),
            "surface_temperature_max_K": float(
                np.max(production.surface_temperature_k)
            ),
            "surface_temperature_min_C": float(
                np.min(production.surface_temperature_k) - 273.15
            ),
            "surface_temperature_max_C": float(
                np.max(production.surface_temperature_k) - 273.15
            ),
            "latent_heat_flux_min_W_m2": float(
                np.min(production.latent_heat_flux_w_m2)
            ),
            "latent_heat_flux_max_W_m2": float(
                np.max(production.latent_heat_flux_w_m2)
            ),
            "surface_dry_bulk_density_min_kg_m3": float(
                np.min(production.surface_dry_bulk_density_kg_m3)
            ),
            "surface_dry_bulk_density_max_kg_m3": float(
                np.max(production.surface_dry_bulk_density_kg_m3)
            ),
        }
    )
    reference_time = _reference_constant_mean_drying_time(repo_root)
    if reference_time is not None:
        validation["comparison_with_original_no_latent_heat_model"] = {
            "reference_drying_time_s": reference_time,
            "steam_model_drying_time_s": production.drying_time_s,
            "difference_s": production.drying_time_s - reference_time,
            "difference_h": (
                production.drying_time_s - reference_time
            ) / 3600.0,
            "relative_change_pct": 100.0
            * (production.drying_time_s / reference_time - 1.0),
        }
    return validation


def _find_node_runtime() -> tuple[str, str | None]:
    bundled_root = (
        Path.home()
        / ".cache"
        / "codex-runtimes"
        / "codex-primary-runtime"
        / "dependencies"
        / "node"
    )
    bundled_node = bundled_root / "bin" / "node.exe"
    executable = str(bundled_node) if bundled_node.exists() else shutil.which("node")
    if not executable:
        raise FileNotFoundError(
            "Node.js was not found. Use --skip-xlsx or install a Node runtime."
        )
    bundled_modules = bundled_root / "node_modules"
    return (
        executable,
        str(bundled_modules) if bundled_modules.exists() else None,
    )


def build_workbooks(
    script_path: Path, repo_root: Path, result_dir: Path
) -> tuple[Path, Path]:
    builder = script_path.with_name("a4_steam_build_workbooks.cjs")
    template = (
        repo_root
        / "比赛题目"
        / "CUMCM2026Problems"
        / "A题"
        / "附件"
        / "附件3"
        / "result4.xlsx"
    )
    result4_path = result_dir / "result4_steam.xlsx"
    inputs_path = result_dir / "q4_inputs_4h_72h.xlsx"
    preview_dir = repo_root / "tmp" / "A_problem4_steam_verify"
    if not builder.exists() or not template.exists():
        raise FileNotFoundError("Cannot find the question-4 builder or template.")
    node, module_path = _find_node_runtime()
    environment = os.environ.copy()
    if module_path:
        current = environment.get("NODE_PATH")
        environment["NODE_PATH"] = (
            module_path if not current else module_path + os.pathsep + current
        )
    completed = subprocess.run(
        [
            node,
            str(builder),
            str(template),
            str(result_dir / "result4_payload.json"),
            str(result_dir / "q4_inputs_4h_72h_60s.csv"),
            str(find_attachment_path(repo_root, "附件2.xlsx")),
            str(result_dir / "validation_summary.json"),
            str(result4_path),
            str(inputs_path),
            str(preview_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environment,
    )
    if completed.stdout.strip():
        print(completed.stdout.strip(), flush=True)
    return result4_path, inputs_path


def find_attachment_path(repo_root: Path, filename: str) -> Path:
    return base.find_attachment(repo_root, filename)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attachment1", type=Path, help="附件1路径")
    parser.add_argument("--attachment2", type=Path, help="附件2路径")
    parser.add_argument("--repo-root", type=Path, help="项目根目录")
    parser.add_argument("--output-dir", type=Path, help="结果输出目录")
    parser.add_argument(
        "--skip-xlsx",
        action="store_true",
        help="只生成CSV/JSON，不构建两个Excel工作簿",
    )
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    repo_root = args.repo_root.resolve() if args.repo_root else script_path.parents[1]
    attachment1 = (
        args.attachment1.resolve()
        if args.attachment1
        else find_attachment_path(repo_root, "附件1.xlsx")
    )
    attachment2 = (
        args.attachment2.resolve()
        if args.attachment2
        else find_attachment_path(repo_root, "附件2.xlsx")
    )
    result_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else repo_root / "results" / "A_problem4_shrinkage_steam"
    )
    result_dir.mkdir(parents=True, exist_ok=True)

    environment = q2.load_and_preprocess_environment(attachment1)
    boundary = base.build_boundary_program(environment)
    radius_program = base.load_radius_program(attachment2)

    coarse = simulate_until_dry(
        boundary,
        radius_program,
        nominal_initial_step_cm=0.0125,
        early_step_s=1.0,
        late_step_s=30.0,
        final_step_s=1.0,
    )
    production = simulate_until_dry(
        boundary,
        radius_program,
        nominal_initial_step_cm=0.00625,
        early_step_s=1.0,
        late_step_s=30.0,
        final_step_s=1.0,
    )

    table_times = base.build_table_times(production.drying_time_s)
    write_full_csv(result_dir / "moisture_full_60s_0p1cm.csv", production)
    write_temperature_csv(
        result_dir / "temperature_auxiliary_60s_0p1cm_K.csv", production
    )
    base.write_table_csv(
        result_dir / "table6_moisture.csv", production, table_times
    )
    write_workbook_payload(result_dir / "result4_payload.json", production)
    write_surface_energy_csv(result_dir / "surface_energy_60s.csv", production)
    write_interpolated_inputs_csv(
        result_dir / "q4_inputs_4h_72h_60s.csv",
        radius_program,
        boundary,
    )
    validation = build_validation(
        repo_root, boundary, radius_program, production, coarse
    )
    with (result_dir / "validation_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(validation, stream, ensure_ascii=False, indent=2)

    result4_path = None
    inputs_path = None
    if not args.skip_xlsx:
        result4_path, inputs_path = build_workbooks(
            script_path, repo_root, result_dir
        )

    print(
        f"Appendix-4 fixed h_m: {base.MASS_TRANSFER_COEFFICIENT_M_S:.3e} m/s; "
        "dynamic rho_d=(760+90*C_s)/(1+C_s); diffusivity T in K"
    )
    print(
        f"Drying time: {production.drying_time_s:.0f} s "
        f"= {production.drying_time_s / 3600.0:.6f} h"
    )
    print(
        f"Threshold bracket: {production.previous_time_s:.0f} s -> "
        f"{production.drying_time_s:.0f} s; "
        f"{production.previous_maximum_moisture:.10f} -> "
        f"{production.final_maximum_moisture:.10f} kg/kg"
    )
    print("\nTable 6: moisture concentration (kg/kg)")
    with (result_dir / "table6_moisture.csv").open(
        "r", encoding="utf-8-sig"
    ) as stream:
        print(stream.read().strip())
    print("\nValidation summary")
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    if result4_path is not None:
        print(f"\nresult4 workbook: {result4_path}")
    if inputs_path is not None:
        print(f"4--72 h inputs workbook: {inputs_path}")
    print(f"Results: {result_dir}")


if __name__ == "__main__":
    main()
