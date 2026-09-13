"""A题问题1：圆柱药材温度场和水分场有限体积模型。

输入为data_preprocessing.py生成的1 s分段线性环境数据；本文件只负责
有限体积计算、表格/场数据输出和数值验证，不包含作图代码。

计算设置与原版保持一致：
* 温度：节点中心有限体积、后向欧拉、Thomas三对角求解；
* 水分：表面加密的单元中心有限体积、后向欧拉、Picard非线性迭代；
* 温度正式网格0.025 cm / 0.25 s；
* 水分正式网格0.0125 cm / 0.125 s，并保留0.025 cm / 0.25 s网格验证。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


RADIUS_M = 0.02
INITIAL_TEMPERATURE_C = 28.0
INITIAL_MOISTURE_KG_KG = 2.55
DENSITY_KG_M3 = 820.0
SPECIFIC_HEAT_J_KG_K = 2600.0
THERMAL_CONDUCTIVITY_W_M_K = 0.36
CONVECTION_COEFFICIENT_W_M2_K = 25.0
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
class TemperatureResult:
    times_s: np.ndarray
    radii_m: np.ndarray
    temperature_c: np.ndarray
    capacities_j_k: np.ndarray
    boundary_heat_input_j_m: float


@dataclass
class MoistureResult:
    times_s: np.ndarray
    radii_m: np.ndarray
    moisture_kg_kg: np.ndarray
    surface_moisture_kg_kg: np.ndarray
    volumes_m3_m: np.ndarray
    boundary_outflow_m3_kg_kg_m: float
    maximum_picard_iterations: int


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_environment_csv(path: Path) -> EnvironmentData:
    """读取预处理脚本输出的统一环境边界数据。"""
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("预处理环境数据为空。")
    required = {"time_s", "temperature_c", "moisture_kg_kg"}
    if not required.issubset(rows[0]):
        raise ValueError(f"预处理环境数据缺少字段：{sorted(required)}")

    times = np.asarray([float(row["time_s"]) for row in rows], dtype=float)
    temperatures = np.asarray(
        [float(row["temperature_c"]) for row in rows], dtype=float
    )
    moistures = np.asarray(
        [float(row["moisture_kg_kg"]) for row in rows], dtype=float
    )
    if not np.all(np.diff(times) > 0.0):
        raise ValueError("预处理数据的时间必须严格递增。")
    if times[0] > 0.0 or times[-1] < END_TIME_S:
        raise ValueError("预处理数据没有完整覆盖0--1800 s。")
    if not (
        np.all(np.isfinite(temperatures)) and np.all(np.isfinite(moistures))
    ):
        raise ValueError("预处理数据包含缺失值或非有限数值。")
    return EnvironmentData(times, temperatures, moistures)


def factor_tridiagonal(
    lower: np.ndarray, diagonal: np.ndarray, upper: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """预分解三对角矩阵，供Thomas算法重复使用。"""
    n = diagonal.size
    denominators = np.empty(n, dtype=float)
    upper_prime = np.empty(n - 1, dtype=float)
    denominators[0] = diagonal[0]
    if abs(denominators[0]) < 1e-15:
        raise np.linalg.LinAlgError("三对角分解出现零主元。")
    upper_prime[0] = upper[0] / denominators[0]
    for i in range(1, n - 1):
        denominators[i] = diagonal[i] - lower[i - 1] * upper_prime[i - 1]
        if abs(denominators[i]) < 1e-15:
            raise np.linalg.LinAlgError("三对角分解出现零主元。")
        upper_prime[i] = upper[i] / denominators[i]
    denominators[-1] = diagonal[-1] - lower[-1] * upper_prime[-1]
    if abs(denominators[-1]) < 1e-15:
        raise np.linalg.LinAlgError("三对角分解出现零主元。")
    return lower.copy(), denominators, upper_prime


def solve_factored_tridiagonal(
    factor: tuple[np.ndarray, np.ndarray, np.ndarray], rhs: np.ndarray
) -> np.ndarray:
    """用已分解的Thomas因子求解三对角线性方程。"""
    lower, denominators, upper_prime = factor
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


# ---------------------------------------------------------------------------
# 温度有限体积模型：以下函数来自原a1_temperature_fvm.py的计算部分
# ---------------------------------------------------------------------------


def build_temperature_system(
    radial_step_m: float, time_step_s: float
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray], float]:
    """建立圆柱坐标下后向欧拉温度有限体积方程。"""
    intervals = int(round(RADIUS_M / radial_step_m))
    if not math.isclose(intervals * radial_step_m, RADIUS_M, abs_tol=1e-12):
        raise ValueError("径向步长必须整除0.02 m半径。")

    radii = np.linspace(0.0, RADIUS_M, intervals + 1)
    n = radii.size
    west_faces = np.maximum(0.0, radii - radial_step_m / 2.0)
    east_faces = np.minimum(RADIUS_M, radii + radial_step_m / 2.0)
    volumes = math.pi * (east_faces**2 - west_faces**2)
    capacities = DENSITY_KG_M3 * SPECIFIC_HEAT_J_KG_K * volumes

    lower = np.zeros(n - 1, dtype=float)
    diagonal = np.ones(n, dtype=float)
    upper = np.zeros(n - 1, dtype=float)
    for i in range(n):
        if i > 0:
            conductance_w = (
                THERMAL_CONDUCTIVITY_W_M_K
                * 2.0
                * math.pi
                * west_faces[i]
                / radial_step_m
            )
            coefficient_w = conductance_w / capacities[i]
            diagonal[i] += time_step_s * coefficient_w
            lower[i - 1] = -time_step_s * coefficient_w
        if i < n - 1:
            conductance_e = (
                THERMAL_CONDUCTIVITY_W_M_K
                * 2.0
                * math.pi
                * east_faces[i]
                / radial_step_m
            )
            coefficient_e = conductance_e / capacities[i]
            diagonal[i] += time_step_s * coefficient_e
            upper[i] = -time_step_s * coefficient_e

    surface_area_per_length = 2.0 * math.pi * RADIUS_M
    convection_coefficient = (
        CONVECTION_COEFFICIENT_W_M2_K
        * surface_area_per_length
        / capacities[-1]
    )
    diagonal[-1] += time_step_s * convection_coefficient
    return radii, factor_tridiagonal(lower, diagonal, upper), convection_coefficient


def simulate_temperature(
    source_times_s: np.ndarray,
    source_temperature_c: np.ndarray,
    radial_step_cm: float,
    time_step_s: float,
) -> TemperatureResult:
    """按原版设置求解0--1800 s径向温度场。"""
    radial_step_m = radial_step_cm / 100.0
    steps = int(round(END_TIME_S / time_step_s))
    if not math.isclose(steps * time_step_s, END_TIME_S, abs_tol=1e-12):
        raise ValueError("时间步长必须整除1800 s。")
    record_stride = int(round(1.0 / time_step_s))
    if not math.isclose(record_stride * time_step_s, 1.0, abs_tol=1e-12):
        raise ValueError("时间步长必须整除1 s输出间隔。")

    radii, factor, convection_coefficient = build_temperature_system(
        radial_step_m, time_step_s
    )
    west_faces = np.maximum(0.0, radii - radial_step_m / 2.0)
    east_faces = np.minimum(RADIUS_M, radii + radial_step_m / 2.0)
    volumes = math.pi * (east_faces**2 - west_faces**2)
    capacities = DENSITY_KG_M3 * SPECIFIC_HEAT_J_KG_K * volumes

    output_times = np.arange(0.0, END_TIME_S + 1.0, 1.0)
    temperature = np.full(radii.size, INITIAL_TEMPERATURE_C, dtype=float)
    records = np.empty((output_times.size, radii.size), dtype=float)
    records[0] = temperature
    boundary_heat_input = 0.0
    surface_area_per_length = 2.0 * math.pi * RADIUS_M
    record_index = 1

    for step in range(1, steps + 1):
        time_now = step * time_step_s
        oven_temperature = float(
            np.interp(time_now, source_times_s, source_temperature_c)
        )
        rhs = temperature.copy()
        rhs[-1] += time_step_s * convection_coefficient * oven_temperature
        temperature = solve_factored_tridiagonal(factor, rhs)
        boundary_heat_input += (
            time_step_s
            * CONVECTION_COEFFICIENT_W_M2_K
            * surface_area_per_length
            * (oven_temperature - temperature[-1])
        )
        if step % record_stride == 0:
            records[record_index] = temperature
            record_index += 1

    if record_index != output_times.size:
        raise RuntimeError("温度场保存时刻数量不一致。")
    return TemperatureResult(
        output_times,
        radii,
        records,
        capacities,
        boundary_heat_input,
    )


# ---------------------------------------------------------------------------
# 水分有限体积模型：以下函数来自原a1_coupled_fvm.py的计算部分
# ---------------------------------------------------------------------------


def moisture_diffusivity(moisture_kg_kg: np.ndarray) -> np.ndarray:
    """附件2经验关系D(C)=7e-9 exp(-0.89/C)，单位m²/s。"""
    moisture = np.asarray(moisture_kg_kg, dtype=float)
    if np.any(moisture <= 0.0):
        raise ValueError("计算D(C)时水分浓度必须保持为正。")
    return 7.0e-9 * np.exp(-0.89 / moisture)


def build_moisture_system(
    radii_m: np.ndarray,
    west_faces_m: np.ndarray,
    east_faces_m: np.ndarray,
    volumes_m3_m: np.ndarray,
    moisture_iterate: np.ndarray,
    time_step_s: float,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], float, float]:
    """建立一次Picard线性化后的水分后向欧拉方程。"""
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
    """求解非线性径向水分扩散及表面对流传质。"""
    radial_step_m = radial_step_cm / 100.0
    intervals = int(round(RADIUS_M / radial_step_m))
    steps = int(round(END_TIME_S / time_step_s))
    if not math.isclose(intervals * radial_step_m, RADIUS_M, abs_tol=1e-12):
        raise ValueError("径向步长必须整除0.02 m半径。")
    if not math.isclose(steps * time_step_s, END_TIME_S, abs_tol=1e-12):
        raise ValueError("时间步长必须整除1800 s。")
    record_stride = int(round(1.0 / time_step_s))
    if not math.isclose(record_stride * time_step_s, 1.0, abs_tol=1e-12):
        raise ValueError("时间步长必须整除1 s输出间隔。")

    # 二次映射在干燥表面聚集控制体，保持原程序的非均匀网格。
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
            factor, boundary_coefficient, boundary_conductance = (
                build_moisture_system(
                    radii,
                    west_faces,
                    east_faces,
                    volumes,
                    iterate,
                    time_step_s,
                )
            )
            rhs = previous.copy()
            rhs[-1] += time_step_s * boundary_coefficient * oven_moisture
            updated = solve_factored_tridiagonal(factor, rhs)
            if np.max(np.abs(updated - iterate)) < picard_tolerance:
                iterate = updated
                break
            iterate = updated
        else:
            raise RuntimeError(f"水分Picard迭代在t={time_now:g} s未收敛。")

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
        output_times,
        radii,
        records,
        surface_records,
        volumes,
        boundary_outflow,
        maximum_iterations_used,
    )


# ---------------------------------------------------------------------------
# 结果采样、表格输出与验证
# ---------------------------------------------------------------------------


def sample_field(
    times_s: np.ndarray,
    radii_m: np.ndarray,
    field: np.ndarray,
    query_times_s: np.ndarray,
    query_radii_cm: np.ndarray,
) -> np.ndarray:
    """在1 s记录和指定物理半径上采样普通径向场。"""
    time_indices = np.rint(query_times_s).astype(int)
    if not np.allclose(times_s[time_indices], query_times_s):
        raise ValueError("请求的时间不在1 s记录中。")
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
    """采样单元中心水分场，并重建圆心与真实表面数值。"""
    time_indices = np.rint(query_times_s).astype(int)
    if not np.allclose(result.times_s[time_indices], query_times_s):
        raise ValueError("请求的时间不在1 s记录中。")
    sampled = np.empty((query_times_s.size, query_radii_cm.size), dtype=float)
    extended_radii = np.concatenate(([0.0], result.radii_m, [RADIUS_M]))
    query_radii_m = query_radii_cm / 100.0
    for j, time_index in enumerate(time_indices):
        nodes = result.moisture_kg_kg[time_index]
        r0_squared = result.radii_m[0] ** 2
        r1_squared = result.radii_m[1] ** 2
        center_value = (
            nodes[0] * r1_squared - nodes[1] * r0_squared
        ) / (r1_squared - r0_squared)
        extended_values = np.concatenate(
            ([center_value], nodes, [result.surface_moisture_kg_kg[time_index]])
        )
        sampled[j] = np.interp(query_radii_m, extended_radii, extended_values)
    return sampled


def calculate_surface_fluxes(
    environment: EnvironmentData,
    temperature: TemperatureResult,
    moisture: MoistureResult,
) -> dict[str, np.ndarray]:
    """计算表面热流密度和向外水分通量，符号约定与原版一致。"""
    if not np.array_equal(temperature.times_s, moisture.times_s):
        raise ValueError("温度和水分结果必须具有相同时间网格。")
    times_s = temperature.times_s
    oven_temperature = np.interp(
        times_s, environment.times_s, environment.temperature_c
    )
    oven_moisture = np.interp(
        times_s, environment.times_s, environment.moisture_kg_kg
    )
    surface_temperature = temperature.temperature_c[:, -1]
    surface_moisture = moisture.surface_moisture_kg_kg
    heat_flux = CONVECTION_COEFFICIENT_W_M2_K * (
        oven_temperature - surface_temperature
    )
    moisture_ratio_flux = MASS_TRANSFER_COEFFICIENT_M_S * (
        surface_moisture - oven_moisture
    )
    moisture_mass_flux = DENSITY_KG_M3 * moisture_ratio_flux
    return {
        "time_s": times_s,
        "oven_temperature_c": oven_temperature,
        "surface_temperature_c": surface_temperature,
        "heat_flux_into_herb_w_m2": heat_flux,
        "oven_moisture_kg_kg": oven_moisture,
        "surface_moisture_kg_kg": surface_moisture,
        "moisture_ratio_flux_out_m_s": moisture_ratio_flux,
        "moisture_mass_flux_out_kg_m2_s": moisture_mass_flux,
    }


def write_field_csv(
    path: Path, times_s: np.ndarray, radii_cm: np.ndarray, values: np.ndarray
) -> None:
    """按原版四位小数输出1 s×0.1 cm完整场。"""
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_s", *[f"r_{r:.1f}_cm" for r in radii_cm]])
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


def write_surface_flux_csv(path: Path, fluxes: dict[str, np.ndarray]) -> None:
    headers = list(fluxes)
    arrays = [fluxes[name] for name in headers]
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        for row in zip(*arrays):
            writer.writerow([f"{row[0]:.0f}", *[f"{value:.10g}" for value in row[1:]]])


def build_validation(
    environment: EnvironmentData,
    temperature: TemperatureResult,
    moisture_coarse: MoistureResult,
    moisture: MoistureResult,
) -> dict:
    """保持原版的网格、水分守恒、单调性和物理范围检查。"""
    coarse_on_output = sample_moisture_field(
        moisture_coarse, moisture_coarse.times_s, OUTPUT_RADII_CM
    )
    production_on_output = sample_moisture_field(
        moisture, moisture.times_s, OUTPUT_RADII_CM
    )
    refinement_change = np.abs(coarse_on_output - production_on_output)
    coarse_table = sample_moisture_field(
        moisture_coarse, TABLE_TIMES_S, TABLE_RADII_CM
    )
    production_table = sample_moisture_field(
        moisture, TABLE_TIMES_S, TABLE_RADII_CM
    )
    table_refinement_change = np.abs(coarse_table - production_table)

    stored_change = float(
        np.sum(
            moisture.volumes_m3_m
            * (moisture.moisture_kg_kg[-1] - moisture.moisture_kg_kg[0])
        )
    )
    balance_residual = stored_change + moisture.boundary_outflow_m3_kg_kg_m
    balance_scale = max(
        abs(stored_change), abs(moisture.boundary_outflow_m3_kg_kg_m), 1e-30
    )
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
        moisture, TABLE_TIMES_S, TABLE_RADII_CM
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
            "coarse_to_production_maximum_change_kg_kg": float(np.max(refinement_change)),
            "coarse_to_production_mean_change_kg_kg": float(np.mean(refinement_change)),
            "table2_maximum_change_kg_kg": float(np.max(table_refinement_change)),
            "table2_mean_change_kg_kg": float(np.mean(table_refinement_change)),
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


def run_production_model(
    environment: EnvironmentData,
) -> tuple[TemperatureResult, MoistureResult, MoistureResult]:
    """集中保留原主程序的三次正式/验证计算调用。"""
    temperature = simulate_temperature(
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
    return temperature, moisture_coarse, moisture


def save_model_state(
    path: Path,
    temperature: TemperatureResult,
    moisture: MoistureResult,
) -> None:
    """保存无舍入内部场，供可视化脚本使用，不参与重新计算。"""
    np.savez_compressed(
        path,
        times_s=temperature.times_s,
        temperature_radii_m=temperature.radii_m,
        temperature_c=temperature.temperature_c,
        temperature_capacities_j_k=temperature.capacities_j_k,
        temperature_boundary_heat_input_j_m=np.asarray(
            temperature.boundary_heat_input_j_m
        ),
        moisture_radii_m=moisture.radii_m,
        moisture_kg_kg=moisture.moisture_kg_kg,
        surface_moisture_kg_kg=moisture.surface_moisture_kg_kg,
        moisture_volumes_m3_m=moisture.volumes_m3_m,
        moisture_boundary_outflow_m3_kg_kg_m=np.asarray(
            moisture.boundary_outflow_m3_kg_kg_m
        ),
        maximum_picard_iterations=np.asarray(moisture.maximum_picard_iterations),
    )


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
        help="data_preprocessing.py生成的1 s线性插值CSV。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "results" / "A_problem1_modular" / "model",
        help="有限体积结果输出目录。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input_data.resolve()
    if not input_path.exists():
        raise FileNotFoundError(
            f"找不到预处理数据：{input_path}\n请先运行data_preprocessing.py。"
        )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = load_environment_csv(input_path)
    temperature, moisture_coarse, moisture = run_production_model(environment)

    temperature_full = sample_field(
        temperature.times_s,
        temperature.radii_m,
        temperature.temperature_c,
        temperature.times_s,
        OUTPUT_RADII_CM,
    )
    moisture_full = sample_moisture_field(
        moisture, moisture.times_s, OUTPUT_RADII_CM
    )
    temperature_table = sample_field(
        temperature.times_s,
        temperature.radii_m,
        temperature.temperature_c,
        TABLE_TIMES_S,
        TABLE_RADII_CM,
    )
    moisture_table = sample_moisture_field(
        moisture, TABLE_TIMES_S, TABLE_RADII_CM
    )
    fluxes = calculate_surface_fluxes(environment, temperature, moisture)

    write_field_csv(
        output_dir / "temperature_full_1s_0p1cm.csv",
        temperature.times_s,
        OUTPUT_RADII_CM,
        temperature_full,
    )
    write_field_csv(
        output_dir / "moisture_full_1s_0p1cm.csv",
        moisture.times_s,
        OUTPUT_RADII_CM,
        moisture_full,
    )
    write_table_csv(
        output_dir / "table1_temperature.csv",
        TABLE_TIMES_S,
        TABLE_RADII_CM,
        temperature_table,
    )
    write_table_csv(
        output_dir / "table2_moisture.csv",
        TABLE_TIMES_S,
        TABLE_RADII_CM,
        moisture_table,
    )
    write_surface_flux_csv(output_dir / "surface_fluxes_1s.csv", fluxes)
    save_model_state(output_dir / "model_state.npz", temperature, moisture)

    validation = build_validation(
        environment, temperature, moisture_coarse, moisture
    )
    (output_dir / "validation_summary.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("表1：温度 / °C")
    print(np.array2string(temperature_table, precision=4, suppress_small=False))
    print("\n表2：水分浓度 / (kg/kg)")
    print(np.array2string(moisture_table, precision=4, suppress_small=False))
    print(f"\n结果目录：{output_dir}")


if __name__ == "__main__":
    main()
