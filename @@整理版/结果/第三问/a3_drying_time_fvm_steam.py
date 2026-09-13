"""A题问题3：计入表面蒸发潜热的全程干燥有限体积模型。

本程序以 ``a3_drying_time_fvm.py`` 为基准：0--4 h 使用附件1实测烘房
环境，4 h 后使用3--4 h时间加权均值；药材内部仍采用附录3的导热方程、
Fick扩散方程和物性关系。唯一的物理升级是在外表面能量平衡中加入水分
蒸发潜热 ``L_v j_w``。题目给定的传质系数 h_m=8e-7 m/s 保持固定。问题3
半径和长度不变，且干物质质量守恒，因此 rho_d=m_d/V 固定为初态值；模型
不计收缩，也不再叠加体积潜热源，以免重复计算蒸发耗热。

默认输出到 ``results/A_problem3_drying_time_steam``：
  * table5_moisture.csv：题目表5；
  * result3_steam.xlsx：每60 s、每0.1 cm的含水率结果；
  * moisture_full_60s_0p1cm.csv：上述工作簿的数据源；
  * surface_energy_60s.csv：表面蒸发和能量通量诊断；
  * validation_summary.json：阈值、网格、质量和能量守恒验证。
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
import a3_drying_time_fvm as base


# 水的汽化潜热经验式：L_v=[2500.8-2.36 T_s(°C)] kJ/kg。
LATENT_HEAT_INTERCEPT_KJ_KG = 2500.8
LATENT_HEAT_SLOPE_KJ_KG_K = 2.36

# 问题3体积固定且干物质质量不变，因此干基体积密度全程保持初态值。
DRY_BULK_DENSITY_KG_M3 = (
    650.0 + 128.0 * q2.INITIAL_MOISTURE_KG_KG
) / (1.0 + q2.INITIAL_MOISTURE_KG_KG)


@dataclass
class SteamStepResult:
    temperature_c: np.ndarray
    moisture_kg_kg: np.ndarray
    surface_temperature_c: float
    surface_moisture_kg_kg: float
    heat_boundary_conductance: float
    moisture_boundary_conductance: float
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
class SteamDryingResult:
    # 与base.LongDryingResult同名的字段用于复用表5和网格验证逻辑。
    times_s: np.ndarray
    temperature_c: np.ndarray
    moisture_kg_kg: np.ndarray
    drying_time_s: float
    last_not_dry_time_s: float
    last_not_dry_maximum_moisture: float
    final_maximum_moisture: float
    controlling_radius_cm: float
    final_internal_temperature_c: np.ndarray
    final_internal_moisture_kg_kg: np.ndarray
    grid: q2.Grid
    integrated_moisture_outflow: float
    maximum_picard_iterations: int
    time_step_counts: dict[str, int]

    # 蒸发潜热模型的逐分钟诊断量。
    oven_temperature_c: np.ndarray
    oven_moisture_kg_kg: np.ndarray
    surface_temperature_c: np.ndarray
    surface_moisture_kg_kg: np.ndarray
    surface_dry_bulk_density_kg_m3: np.ndarray
    water_mass_flux_kg_m2_s: np.ndarray
    latent_heat_j_kg: np.ndarray
    latent_heat_flux_w_m2: np.ndarray
    convective_heat_flux_w_m2: np.ndarray
    conductive_heat_flux_w_m2: np.ndarray
    surface_energy_residual_w_m2: np.ndarray

    # 全过程积分守恒量（均按单位圆柱长度计）。
    integrated_convective_heat_j_m: float
    integrated_conductive_heat_j_m: float
    integrated_latent_heat_j_m: float
    integrated_physical_water_outflow_kg_m: float
    accumulated_sensible_energy_change_j_m: float
    maximum_storage_balance_residual_j_m: float
    maximum_surface_energy_residual_w_m2: float


def latent_heat_of_vaporization_j_kg(surface_temperature_c: float) -> float:
    """Return water latent heat at the current product-surface temperature."""
    value = 1000.0 * (
        LATENT_HEAT_INTERCEPT_KJ_KG
        - LATENT_HEAT_SLOPE_KJ_KG_K * surface_temperature_c
    )
    if value <= 0.0:
        raise ValueError("The latent heat correlation returned a non-positive value.")
    return value


def surface_temperature_with_evaporation(
    outer_cell_temperature_c: float,
    oven_temperature_c: float,
    surface_conductivity_w_m_k: float,
    surface_distance_m: float,
    latent_heat_flux_w_m2: float,
) -> float:
    """Solve the algebraic surface energy balance for T_s.

    k/delta*(T_s-T_P) = h_T*(T_inf-T_s) - q_lat.
    """
    cell_side = surface_conductivity_w_m_k / surface_distance_m
    heat_transfer = q2.HEAT_TRANSFER_COEFFICIENT_W_M2_K
    return (
        cell_side * outer_cell_temperature_c
        + heat_transfer * oven_temperature_c
        - latent_heat_flux_w_m2
    ) / (cell_side + heat_transfer)


def advance_coupled_step(
    grid: q2.Grid,
    temperature_previous: np.ndarray,
    moisture_previous: np.ndarray,
    surface_temperature_previous_c: float,
    oven_temperature_c: float,
    oven_moisture_kg_kg: float,
    time_step_s: float,
    temperature_tolerance_c: float = 1.0e-8,
    moisture_tolerance: float = 1.0e-10,
    surface_temperature_tolerance_c: float = 1.0e-9,
    maximum_iterations: int = 50,
) -> SteamStepResult:
    """Advance one nonlinear implicit step with surface latent-heat coupling."""
    temperature_iterate = temperature_previous.copy()
    moisture_iterate = moisture_previous.copy()
    surface_temperature_iterate = float(surface_temperature_previous_c)
    surface_distance = q2.RADIUS_M - grid.radii_m[-1]
    surface_area_per_length = 2.0 * math.pi * q2.RADIUS_M
    heat_transfer = q2.HEAT_TRANSFER_COEFFICIENT_W_M2_K
    mass_transfer = q2.MASS_TRANSFER_COEFFICIENT_M_S

    for iteration in range(1, maximum_iterations + 1):
        # Fick方程仍用题给固定h_m；表面值由膜阻与末单元扩散阻串联重构。
        diffusivity = q2.moisture_diffusivity_m2_s(
            moisture_iterate, temperature_iterate
        )
        moisture_system = q2.build_diffusion_system(
            grid,
            moisture_previous,
            diffusivity,
            np.ones_like(moisture_iterate),
            mass_transfer,
            oven_moisture_kg_kg,
            time_step_s,
        )
        moisture_updated = q2.solve_tridiagonal(*moisture_system[:4])
        surface_moisture = q2.reconstruct_surface_value(
            moisture_updated[-1],
            oven_moisture_kg_kg,
            float(diffusivity[-1]),
            mass_transfer,
            surface_distance,
        )

        # 问题3中m_d和V均不变，故rho_d=m_d/V固定为初态值。
        surface_dry_bulk_density = DRY_BULK_DENSITY_KG_M3
        # j_w=rho_d*h_m*(C_s-C_inf)，正值表示水分由药材流向烘房。
        water_mass_flux = (
            surface_dry_bulk_density
            * mass_transfer
            * (surface_moisture - oven_moisture_kg_kg)
        )
        latent_heat = latent_heat_of_vaporization_j_kg(
            surface_temperature_iterate
        )
        latent_heat_flux = latent_heat * water_mass_flux

        conductivity = q2.conductivity_w_m_k(moisture_updated)
        volumetric_heat_capacity = (
            q2.density_kg_m3(moisture_updated)
            * q2.heat_capacity_j_kg_k(moisture_updated)
        )
        lower, diagonal, upper, rhs, heat_boundary_conductance = (
            q2.build_diffusion_system(
                grid,
                temperature_previous,
                conductivity,
                volumetric_heat_capacity,
                heat_transfer,
                oven_temperature_c,
                time_step_s,
            )
        )

        # 消去表面温度后，潜热负载传到末单元的比例为
        # alpha=(k/delta)/(k/delta+h_T)。标准对流矩阵保持不变，只改右端项。
        cell_side = float(conductivity[-1]) / surface_distance
        latent_transfer_fraction = cell_side / (cell_side + heat_transfer)
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
            oven_temperature_c,
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
            temperature_error < temperature_tolerance_c
            and moisture_error < moisture_tolerance
            and surface_temperature_error < surface_temperature_tolerance_c
        ):
            break
    else:
        raise RuntimeError("Latent-heat Picard iteration did not converge.")

    # 以下两个等式分别检查表面能量平衡和有限体积总能量平衡。
    conductive_heat_flux = cell_side * (
        surface_temperature_iterate - float(temperature_iterate[-1])
    )
    convective_heat_flux = heat_transfer * (
        oven_temperature_c - surface_temperature_iterate
    )
    surface_energy_residual = (
        convective_heat_flux - conductive_heat_flux - latent_heat_flux
    )
    sensible_energy_change = float(
        np.sum(
            volumetric_heat_capacity
            * grid.volumes_m3_m
            * (temperature_iterate - temperature_previous)
        )
    )
    conductive_heat_rate_per_length = (
        surface_area_per_length * conductive_heat_flux
    )
    storage_balance_residual = (
        sensible_energy_change
        - time_step_s * conductive_heat_rate_per_length
    )

    return SteamStepResult(
        temperature_c=temperature_iterate,
        moisture_kg_kg=moisture_iterate,
        surface_temperature_c=surface_temperature_iterate,
        surface_moisture_kg_kg=float(surface_moisture),
        heat_boundary_conductance=float(heat_boundary_conductance),
        moisture_boundary_conductance=float(moisture_system[4]),
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
    nominal_radial_step_cm: float,
    early_time_step_s: float,
    late_time_step_s: float,
    final_time_step_s: float,
) -> SteamDryingResult:
    """Run the latent-heat model until max_r C(r,t)<0.15 kg/kg."""
    grid = q2.make_grid(nominal_radial_step_cm)
    temperature = np.full(
        grid.radii_m.size, q2.INITIAL_TEMPERATURE_C, dtype=float
    )
    moisture = np.full(
        grid.radii_m.size, q2.INITIAL_MOISTURE_KG_KG, dtype=float
    )
    surface_temperature = q2.INITIAL_TEMPERATURE_C
    surface_moisture = q2.INITIAL_MOISTURE_KG_KG
    time_s = 0.0
    next_output_time_s = base.OUTPUT_INTERVAL_S

    record_times: list[float] = []
    temperature_records: list[np.ndarray] = []
    moisture_records: list[np.ndarray] = []
    oven_temperature_records: list[float] = []
    oven_moisture_records: list[float] = []
    surface_temperature_records: list[float] = []
    surface_moisture_records: list[float] = []
    surface_dry_bulk_density_records: list[float] = []
    water_mass_flux_records: list[float] = []
    latent_heat_records: list[float] = []
    latent_heat_flux_records: list[float] = []
    convective_heat_flux_records: list[float] = []
    conductive_heat_flux_records: list[float] = []
    surface_energy_residual_records: list[float] = []

    integrated_moisture_outflow = 0.0
    integrated_convective_heat = 0.0
    integrated_conductive_heat = 0.0
    integrated_latent_heat = 0.0
    integrated_physical_water_outflow = 0.0
    accumulated_sensible_energy_change = 0.0
    maximum_storage_balance_residual = 0.0
    maximum_surface_energy_residual = 0.0
    maximum_picard_iterations = 0
    surface_area_per_length = 2.0 * math.pi * q2.RADIUS_M
    time_step_counts = {
        "measured_environment": 0,
        "constant_environment": 0,
        "threshold_refinement": 0,
    }

    maximum_moisture = q2.INITIAL_MOISTURE_KG_KG
    last_not_dry_time = 0.0
    last_not_dry_maximum = maximum_moisture
    controlling_radius_cm = 0.0

    while time_s < base.MAXIMUM_SIMULATION_TIME_S:
        proposed_step, step_label = base.choose_time_step(
            time_s,
            maximum_moisture,
            early_time_step_s,
            late_time_step_s,
            final_time_step_s,
        )
        time_step_s = proposed_step
        if time_s < base.MEASURED_END_TIME_S < time_s + time_step_s:
            time_step_s = base.MEASURED_END_TIME_S - time_s
        if time_s < next_output_time_s < time_s + time_step_s:
            time_step_s = next_output_time_s - time_s
        if time_s + time_step_s > base.MAXIMUM_SIMULATION_TIME_S:
            time_step_s = base.MAXIMUM_SIMULATION_TIME_S - time_s
        if time_step_s <= 0.0:
            raise RuntimeError("Non-positive adaptive time step encountered.")

        new_time = time_s + time_step_s
        oven_temperature, oven_moisture = boundary.values(new_time)
        step = advance_coupled_step(
            grid,
            temperature,
            moisture,
            surface_temperature,
            oven_temperature,
            oven_moisture,
            time_step_s,
        )
        integrated_moisture_outflow += (
            time_step_s
            * step.moisture_boundary_conductance
            * (step.moisture_kg_kg[-1] - oven_moisture)
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
        integrated_physical_water_outflow += (
            time_step_s
            * surface_area_per_length
            * step.water_mass_flux_kg_m2_s
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

        temperature = step.temperature_c
        moisture = step.moisture_kg_kg
        surface_temperature = step.surface_temperature_c
        surface_moisture = step.surface_moisture_kg_kg
        time_s = new_time
        time_step_counts[step_label] += 1
        maximum_picard_iterations = max(
            maximum_picard_iterations, step.iterations
        )

        center_moisture = base.reconstruct_center(grid, moisture)
        extended_moisture = np.concatenate(
            ([center_moisture], moisture, [surface_moisture])
        )
        maximum_index = int(np.argmax(extended_moisture))
        maximum_moisture = float(extended_moisture[maximum_index])
        extended_radii_cm = np.concatenate(
            ([0.0], grid.radii_m * 100.0, [q2.RADIUS_M * 100.0])
        )
        controlling_radius_cm = float(extended_radii_cm[maximum_index])

        is_output_time = math.isclose(
            time_s, next_output_time_s, abs_tol=1.0e-8
        )
        is_dry = maximum_moisture < base.DRYING_THRESHOLD_KG_KG
        if is_output_time or is_dry:
            # 若终点恰好落在整分钟，只记录一次。
            if not record_times or not math.isclose(
                record_times[-1], time_s, abs_tol=1.0e-8
            ):
                record_times.append(time_s)
                temperature_records.append(
                    base.sample_radial_field(
                        grid,
                        temperature,
                        surface_temperature,
                        base.OUTPUT_RADII_CM,
                    )
                )
                moisture_records.append(
                    base.sample_radial_field(
                        grid,
                        moisture,
                        surface_moisture,
                        base.OUTPUT_RADII_CM,
                    )
                )
                oven_temperature_records.append(oven_temperature)
                oven_moisture_records.append(oven_moisture)
                surface_temperature_records.append(surface_temperature)
                surface_moisture_records.append(surface_moisture)
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
            next_output_time_s += base.OUTPUT_INTERVAL_S

        if is_dry:
            break
        last_not_dry_time = time_s
        last_not_dry_maximum = maximum_moisture
    else:
        raise RuntimeError("Maximum simulation time reached before drying.")

    return SteamDryingResult(
        times_s=np.asarray(record_times, dtype=float),
        temperature_c=np.vstack(temperature_records),
        moisture_kg_kg=np.vstack(moisture_records),
        drying_time_s=time_s,
        last_not_dry_time_s=last_not_dry_time,
        last_not_dry_maximum_moisture=last_not_dry_maximum,
        final_maximum_moisture=maximum_moisture,
        controlling_radius_cm=controlling_radius_cm,
        final_internal_temperature_c=temperature,
        final_internal_moisture_kg_kg=moisture,
        grid=grid,
        integrated_moisture_outflow=integrated_moisture_outflow,
        maximum_picard_iterations=maximum_picard_iterations,
        time_step_counts=time_step_counts,
        oven_temperature_c=np.asarray(oven_temperature_records),
        oven_moisture_kg_kg=np.asarray(oven_moisture_records),
        surface_temperature_c=np.asarray(surface_temperature_records),
        surface_moisture_kg_kg=np.asarray(surface_moisture_records),
        surface_dry_bulk_density_kg_m3=np.asarray(
            surface_dry_bulk_density_records
        ),
        water_mass_flux_kg_m2_s=np.asarray(water_mass_flux_records),
        latent_heat_j_kg=np.asarray(latent_heat_records),
        latent_heat_flux_w_m2=np.asarray(latent_heat_flux_records),
        convective_heat_flux_w_m2=np.asarray(convective_heat_flux_records),
        conductive_heat_flux_w_m2=np.asarray(conductive_heat_flux_records),
        surface_energy_residual_w_m2=np.asarray(
            surface_energy_residual_records
        ),
        integrated_convective_heat_j_m=integrated_convective_heat,
        integrated_conductive_heat_j_m=integrated_conductive_heat,
        integrated_latent_heat_j_m=integrated_latent_heat,
        integrated_physical_water_outflow_kg_m=(
            integrated_physical_water_outflow
        ),
        accumulated_sensible_energy_change_j_m=(
            accumulated_sensible_energy_change
        ),
        maximum_storage_balance_residual_j_m=(
            maximum_storage_balance_residual
        ),
        maximum_surface_energy_residual_w_m2=(
            maximum_surface_energy_residual
        ),
    )


def write_full_csv(path: Path, result: SteamDryingResult) -> None:
    """Write high-precision moisture data used by result3_steam.xlsx."""
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["time_s", *[f"r_{r:.1f}_cm" for r in base.OUTPUT_RADII_CM]]
        )
        for time_s, row in zip(result.times_s, result.moisture_kg_kg):
            writer.writerow(
                [f"{time_s:.0f}", *[f"{value:.8f}" for value in row]]
            )


def write_temperature_csv(path: Path, result: SteamDryingResult) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["time_s", *[f"r_{r:.1f}_cm" for r in base.OUTPUT_RADII_CM]]
        )
        for time_s, row in zip(result.times_s, result.temperature_c):
            writer.writerow(
                [f"{time_s:.0f}", *[f"{value:.6f}" for value in row]]
            )


def write_surface_energy_csv(path: Path, result: SteamDryingResult) -> None:
    headers = [
        "time_s",
        "oven_temperature_C",
        "oven_moisture_kg_kg",
        "surface_temperature_C",
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
        result.oven_temperature_c,
        result.oven_moisture_kg_kg,
        result.surface_temperature_c,
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


def _reference_constant_mean_drying_time(repo_root: Path) -> float | None:
    path = (
        repo_root
        / "results"
        / "A_problem3_drying_time"
        / "validation_summary.json"
    )
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    value = payload.get("drying_time", {}).get("production_s")
    return float(value) if value is not None else None


def build_validation(
    boundary: base.BoundaryProgram,
    production: SteamDryingResult,
    coarse: SteamDryingResult,
    table_times_s: np.ndarray,
    production_table: np.ndarray,
    repo_root: Path,
) -> dict:
    validation = base.build_validation(
        boundary, production, coarse, table_times_s, production_table
    )
    validation["model_scope"].update(
        {
            "latent_heat": (
                "included once as the surface sink L_v*j_w; "
                "no volumetric latent-heat source"
            ),
            "surface_energy_balance": (
                "h_T(T_inf-T_s)=k*dT/dr|R+L_v*j_w"
            ),
            "surface_dry_bulk_density": (
                "constant rho_d=m_d/V because both dry mass and the fixed "
                "question-3 cylinder volume are conserved"
            ),
        }
    )
    validation["fixed_parameters"] = {
        "heat_transfer_coefficient_W_m2_K": (
            q2.HEAT_TRANSFER_COEFFICIENT_W_M2_K
        ),
        "mass_transfer_coefficient_m_s": q2.MASS_TRANSFER_COEFFICIENT_M_S,
        "dry_bulk_density_kg_m3": DRY_BULK_DENSITY_KG_M3,
        "dry_bulk_density_derivation": (
            "rho_d=rho(C0)/(1+C0), C0=2.55 kg/kg; fixed in question 3"
        ),
        "latent_heat_correlation": "L_v=[2500.8-2.36*T_s(C)]*1000 J/kg",
    }

    physical_stored_water_change = float(
        DRY_BULK_DENSITY_KG_M3
        * np.sum(
            production.grid.volumes_m3_m
            * (
                production.final_internal_moisture_kg_kg
                - q2.INITIAL_MOISTURE_KG_KG
            )
        )
    )
    physical_surface_outflow = production.integrated_physical_water_outflow_kg_m
    physical_water_residual = (
        physical_stored_water_change + physical_surface_outflow
    )
    physical_water_scale = max(
        abs(physical_stored_water_change),
        abs(physical_surface_outflow),
        1.0e-30,
    )
    validation["water_mass_balance_per_unit_length"] = {
        "constant_dry_bulk_density_kg_m3": DRY_BULK_DENSITY_KG_M3,
        "stored_water_change_kg_m": physical_stored_water_change,
        "integrated_surface_outflow_kg_m": physical_surface_outflow,
        "residual_kg_m": physical_water_residual,
        "relative_residual": abs(physical_water_residual) / physical_water_scale,
        "dry_mass_interpretation": (
            "Question 3 fixes cylinder volume; constant dry mass therefore "
            "implies constant dry-bulk density."
        ),
    }

    integrated_surface_residual = (
        production.integrated_convective_heat_j_m
        - production.integrated_conductive_heat_j_m
        - production.integrated_latent_heat_j_m
    )
    integrated_storage_residual = (
        production.accumulated_sensible_energy_change_j_m
        - production.integrated_conductive_heat_j_m
    )
    energy_scale = max(
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
        "surface_integrated_residual_J_m": integrated_surface_residual,
        "surface_integrated_relative_residual": (
            abs(integrated_surface_residual) / energy_scale
        ),
        "storage_integrated_residual_J_m": integrated_storage_residual,
        "storage_integrated_relative_residual": (
            abs(integrated_storage_residual) / storage_scale
        ),
        "maximum_step_surface_residual_W_m2": (
            production.maximum_surface_energy_residual_w_m2
        ),
        "maximum_step_storage_residual_J_m": (
            production.maximum_storage_balance_residual_j_m
        ),
    }
    validation["thermal_ranges"] = {
        "minimum_saved_internal_temperature_C": float(
            np.min(production.temperature_c)
        ),
        "maximum_saved_internal_temperature_C": float(
            np.max(production.temperature_c)
        ),
        "minimum_saved_surface_temperature_C": float(
            np.min(production.surface_temperature_c)
        ),
        "maximum_saved_surface_temperature_C": float(
            np.max(production.surface_temperature_c)
        ),
        "minimum_latent_heat_flux_W_m2": float(
            np.min(production.latent_heat_flux_w_m2)
        ),
        "maximum_latent_heat_flux_W_m2": float(
            np.max(production.latent_heat_flux_w_m2)
        ),
        "surface_dry_bulk_density_kg_m3": DRY_BULK_DENSITY_KG_M3,
    }

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


def build_result3_workbook(
    script_path: Path,
    repo_root: Path,
    result_dir: Path,
    output_filename: str = "result3_steam.xlsx",
) -> Path:
    builder = script_path.with_name("build_result3_steam_from_csv.cjs")
    template = (
        repo_root
        / "比赛题目"
        / "CUMCM2026Problems"
        / "A题"
        / "附件"
        / "附件3"
        / "result3.xlsx"
    )
    source_csv = result_dir / "moisture_full_60s_0p1cm.csv"
    output = result_dir / output_filename
    preview_dir = repo_root / "tmp" / "A_problem3_steam_fixed_rho_verify"
    if not builder.exists() or not template.exists():
        raise FileNotFoundError("Cannot find the result3 builder or template workbook.")
    node, module_path = _find_node_runtime()
    environment = os.environ.copy()
    if module_path:
        current = environment.get("NODE_PATH")
        environment["NODE_PATH"] = (
            module_path if not current else module_path + os.pathsep + current
        )
    try:
        completed = subprocess.run(
            [
                node,
                str(builder),
                str(template),
                str(source_csv),
                str(output),
                str(preview_dir),
            ],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=environment,
        )
    except subprocess.CalledProcessError as error:
        if error.stdout:
            print(error.stdout, flush=True)
        if error.stderr:
            print(error.stderr, flush=True)
        raise
    if completed.stdout.strip():
        print(completed.stdout.strip(), flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="附件1路径")
    parser.add_argument("--repo-root", type=Path, help="项目根目录")
    parser.add_argument("--output-dir", type=Path, help="结果输出目录")
    parser.add_argument(
        "--skip-xlsx",
        action="store_true",
        help="只生成CSV/JSON，不构建result3_steam.xlsx",
    )
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    repo_root = args.repo_root.resolve() if args.repo_root else script_path.parents[1]
    data_path = args.data.resolve() if args.data else q2.find_default_data_path(repo_root)
    result_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else repo_root / "results" / "A_problem3_drying_time_steam"
    )
    result_dir.mkdir(parents=True, exist_ok=True)

    environment = q2.load_and_preprocess_environment(data_path)
    boundary = base.build_boundary_program(environment)
    coarse = simulate_until_dry(
        boundary,
        nominal_radial_step_cm=0.025,
        early_time_step_s=2.0,
        late_time_step_s=60.0,
        final_time_step_s=2.0,
    )
    production = simulate_until_dry(
        boundary,
        nominal_radial_step_cm=0.0125,
        early_time_step_s=1.0,
        late_time_step_s=30.0,
        final_time_step_s=1.0,
    )

    table_times = base.build_table_times(production.drying_time_s)
    table = base.table_values(production, table_times)
    write_full_csv(result_dir / "moisture_full_60s_0p1cm.csv", production)
    write_temperature_csv(
        result_dir / "temperature_auxiliary_60s_0p1cm.csv", production
    )
    base.write_table_csv(
        result_dir / "table5_moisture.csv",
        table_times,
        table,
        production.drying_time_s,
    )
    write_surface_energy_csv(result_dir / "surface_energy_60s.csv", production)
    validation = build_validation(
        boundary, production, coarse, table_times, table, repo_root
    )
    with (result_dir / "validation_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(validation, stream, ensure_ascii=False, indent=2)

    workbook_path = None
    if not args.skip_xlsx:
        workbook_path = build_result3_workbook(script_path, repo_root, result_dir)

    print(
        f"Plateau boundary: {boundary.plateau_temperature_c:.6f} deg C, "
        f"{boundary.plateau_moisture_kg_kg:.8f} kg/kg"
    )
    print(
        f"Fixed h_m: {q2.MASS_TRANSFER_COEFFICIENT_M_S:.3e} m/s; "
        f"fixed rho_d={DRY_BULK_DENSITY_KG_M3:.9f} kg/m3"
    )
    print(
        f"Drying time: {production.drying_time_s:.0f} s "
        f"= {production.drying_time_s / 3600.0:.6f} h"
    )
    print(
        f"Threshold bracket: {production.last_not_dry_time_s:.0f} s -> "
        f"{production.drying_time_s:.0f} s; "
        f"{production.last_not_dry_maximum_moisture:.10f} -> "
        f"{production.final_maximum_moisture:.10f} kg/kg"
    )
    print("\nTable 5: moisture concentration (kg/kg)")
    for time_value, row in zip(table_times, table):
        label = (
            "end"
            if math.isclose(time_value, production.drying_time_s)
            else f"{time_value / 3600:.0f} h"
        )
        print(label, np.array2string(row, precision=4, suppress_small=False))
    print("\nValidation summary")
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    if workbook_path is not None:
        print(f"\nWorkbook: {workbook_path}")
    print(f"Results: {result_dir}")


if __name__ == "__main__":
    main()
