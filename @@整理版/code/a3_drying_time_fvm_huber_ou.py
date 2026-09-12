"""A题问题3：Huber稳健标定与OU随机边界驱动的全程干燥模型。

0--4 h仍使用附件1的逐分钟实测烘房温度和水分浓度。对3--4 h稳定段分别
进行Huber位置/尺度标定，并把等间隔OU过程写成AR(1)精确转移形式。4--72 h
使用固定随机种子生成逐分钟温度和水分浓度路径；两个变量的同期扰动相关性
由稳健截尾后的OU创新估计。药材内部传热、传质、物性关系和有限体积迭代均
直接复用a3_drying_time_fvm.py，因而与原恒值边界模型仅有环境输入不同。

默认结果写入 results/A_problem3_drying_time_huber_ou，包括：
  * table5_moisture.csv：与原问题3相同的表5格式；
  * result3.xlsx：与附件3模板相同的逐分钟、0.1 cm径向间隔格式；
  * ou_boundary_4h_72h_60s.csv：4--72 h逐分钟OU环境路径；
  * ou_parameters.json与validation_summary.json：标定参数和数值验证。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import secrets
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

import a2_coupled_fvm as q2
import a3_drying_time_fvm as base


FIT_START_TIME_S = 3.0 * 3600.0
MEASURED_END_TIME_S = 4.0 * 3600.0
OU_END_TIME_S = 72.0 * 3600.0
OU_STEP_S = 60.0
HUBER_C = 1.345
MINIMUM_PHI = 1.0e-6
MAXIMUM_PHI = 1.0 - 1.0e-6
DEFAULT_SEED = 20260912


@dataclass(frozen=True)
class RobustOUFit:
    variable: str
    unit: str
    sample_count: int
    transition_count: int
    sampling_interval_s: float
    huber_c: float
    equilibrium_mean: float
    discrete_phi: float
    kappa_per_s: float
    relaxation_time_s: float
    half_life_s: float
    innovation_scale: float
    stationary_scale: float
    ordinary_mean: float
    ordinary_standard_deviation: float
    ordinary_lag1_correlation: float
    phi_at_lower_bound: bool


@dataclass
class OUBoundaryProgram:
    source_times_s: np.ndarray
    source_temperature_c: np.ndarray
    source_moisture_kg_kg: np.ndarray
    ou_times_s: np.ndarray
    ou_temperature_c: np.ndarray
    ou_moisture_kg_kg: np.ndarray
    temperature_fit: RobustOUFit
    moisture_fit: RobustOUFit
    innovation_correlation: float
    seed: int
    sigma_multiplier: float

    @property
    def plateau_temperature_c(self) -> float:
        """Compatibility name used by the original validation routine."""
        return self.temperature_fit.equilibrium_mean

    @property
    def plateau_moisture_kg_kg(self) -> float:
        """Compatibility name used by the original validation routine."""
        return self.moisture_fit.equilibrium_mean

    def values(self, time_s: float) -> tuple[float, float]:
        if time_s <= MEASURED_END_TIME_S:
            return (
                float(
                    np.interp(
                        time_s,
                        self.source_times_s,
                        self.source_temperature_c,
                    )
                ),
                float(
                    np.interp(
                        time_s,
                        self.source_times_s,
                        self.source_moisture_kg_kg,
                    )
                ),
            )
        if time_s > self.ou_times_s[-1]:
            raise ValueError(
                f"OU boundary is only defined through {self.ou_times_s[-1] / 3600:g} h."
            )
        # 与0--4 h实测边界相同，有限体积子步在相邻逐分钟记录之间线性插值。
        return (
            float(np.interp(time_s, self.ou_times_s, self.ou_temperature_c)),
            float(np.interp(time_s, self.ou_times_s, self.ou_moisture_kg_kg)),
        )


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _huber_scale_consistency(huber_c: float) -> float:
    """E[min(Z^2,c^2)] for a standard normal Z."""
    density = math.exp(-0.5 * huber_c**2) / math.sqrt(2.0 * math.pi)
    upper_tail = 1.0 - _normal_cdf(huber_c)
    return (
        1.0
        - 2.0 * huber_c * density
        + 2.0 * (huber_c**2 - 1.0) * upper_tail
    )


def _initial_scale(values: np.ndarray) -> float:
    median = float(np.median(values))
    mad_scale = 1.4826 * float(np.median(np.abs(values - median)))
    ordinary_scale = (
        float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    )
    return max(mad_scale, 0.25 * ordinary_scale, np.finfo(float).eps)


def _huber_scale_about_zero(
    residuals: np.ndarray,
    initial_scale: float,
    huber_c: float,
    maximum_iterations: int = 200,
) -> float:
    beta = _huber_scale_consistency(huber_c)
    scale = max(float(initial_scale), np.finfo(float).eps)
    for _ in range(maximum_iterations):
        clipped_squared = np.minimum(residuals**2, (huber_c * scale) ** 2)
        updated = math.sqrt(float(np.mean(clipped_squared)) / beta)
        updated = max(updated, np.finfo(float).eps)
        if abs(updated - scale) <= 1.0e-10 * max(scale, 1.0e-12):
            return updated
        scale = updated
    raise RuntimeError("Huber scale iteration did not converge.")


def huber_location_scale(
    values: np.ndarray,
    huber_c: float = HUBER_C,
    maximum_iterations: int = 200,
) -> tuple[float, float]:
    """Huber Proposal-2 location and normal-consistent scale."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 3 or not np.all(np.isfinite(values)):
        raise ValueError("Huber calibration requires at least three finite values.")
    location = float(np.median(values))
    scale = _initial_scale(values)
    for _ in range(maximum_iterations):
        standardized = (values - location) / scale
        absolute = np.abs(standardized)
        weights = np.ones_like(absolute)
        outside = absolute > huber_c
        weights[outside] = huber_c / absolute[outside]
        updated_location = float(np.sum(weights * values) / np.sum(weights))
        updated_scale = _huber_scale_about_zero(
            values - updated_location,
            scale,
            huber_c,
        )
        location_converged = abs(updated_location - location) <= (
            1.0e-11 * max(abs(location), scale, 1.0)
        )
        scale_converged = abs(updated_scale - scale) <= (
            1.0e-10 * max(scale, 1.0e-12)
        )
        location, scale = updated_location, updated_scale
        if location_converged and scale_converged:
            return location, scale
    raise RuntimeError("Huber location-scale iteration did not converge.")


def fit_robust_ou(
    times_s: np.ndarray,
    values: np.ndarray,
    variable: str,
    unit: str,
    huber_c: float = HUBER_C,
) -> tuple[RobustOUFit, np.ndarray]:
    """Fit the exact-discrete OU/AR(1) transition with bounded Huber IRLS."""
    times_s = np.asarray(times_s, dtype=float)
    values = np.asarray(values, dtype=float)
    if times_s.ndim != 1 or values.ndim != 1 or times_s.size != values.size:
        raise ValueError("OU fit requires equally sized one-dimensional arrays.")
    if values.size < 20:
        raise ValueError("At least 20 stable-stage records are required for OU fitting.")
    intervals = np.diff(times_s)
    sampling_interval_s = float(np.median(intervals))
    if not np.allclose(intervals, sampling_interval_s, rtol=0.0, atol=1.0e-9):
        raise ValueError("OU calibration records must be equally spaced.")

    equilibrium_mean, level_scale = huber_location_scale(values, huber_c)
    x = values[:-1] - equilibrium_mean
    y = values[1:] - equilibrium_mean
    denominator = float(np.dot(x, x))
    if denominator <= np.finfo(float).eps:
        raise ValueError(f"{variable} has no measurable stable-stage variation.")

    raw_phi = float(np.dot(x, y) / denominator)
    phi = float(np.clip(raw_phi, MINIMUM_PHI, MAXIMUM_PHI))
    innovation_scale = _initial_scale(y - phi * x)
    for _ in range(200):
        residuals = y - phi * x
        standardized = residuals / innovation_scale
        absolute = np.abs(standardized)
        weights = np.ones_like(absolute)
        outside = absolute > huber_c
        weights[outside] = huber_c / absolute[outside]
        weighted_denominator = float(np.sum(weights * x * x))
        if weighted_denominator <= np.finfo(float).eps:
            raise RuntimeError(f"Cannot identify the OU coefficient for {variable}.")
        updated_phi = float(
            np.clip(
                np.sum(weights * x * y) / weighted_denominator,
                MINIMUM_PHI,
                MAXIMUM_PHI,
            )
        )
        updated_residuals = y - updated_phi * x
        updated_scale = _huber_scale_about_zero(
            updated_residuals,
            innovation_scale,
            huber_c,
        )
        if (
            abs(updated_phi - phi) <= 1.0e-11
            and abs(updated_scale - innovation_scale)
            <= 1.0e-10 * max(innovation_scale, 1.0e-12)
        ):
            phi, innovation_scale = updated_phi, updated_scale
            break
        phi, innovation_scale = updated_phi, updated_scale
    else:
        raise RuntimeError(f"Huber OU iteration did not converge for {variable}.")

    residuals = y - phi * x
    kappa_per_s = -math.log(phi) / sampling_interval_s
    stationary_scale = innovation_scale / math.sqrt(1.0 - phi**2)
    lag1 = float(np.corrcoef(values[:-1], values[1:])[0, 1])
    fit = RobustOUFit(
        variable=variable,
        unit=unit,
        sample_count=int(values.size),
        transition_count=int(values.size - 1),
        sampling_interval_s=sampling_interval_s,
        huber_c=huber_c,
        equilibrium_mean=equilibrium_mean,
        discrete_phi=phi,
        kappa_per_s=kappa_per_s,
        relaxation_time_s=1.0 / kappa_per_s,
        half_life_s=math.log(2.0) / kappa_per_s,
        innovation_scale=innovation_scale,
        stationary_scale=stationary_scale,
        ordinary_mean=float(np.mean(values)),
        ordinary_standard_deviation=float(np.std(values, ddof=1)),
        ordinary_lag1_correlation=lag1,
        phi_at_lower_bound=bool(phi <= 1.01 * MINIMUM_PHI),
    )
    return fit, residuals


def robust_innovation_correlation(
    temperature_residuals: np.ndarray,
    moisture_residuals: np.ndarray,
    temperature_scale: float,
    moisture_scale: float,
    huber_c: float = HUBER_C,
) -> float:
    """Estimate contemporaneous coupling after Huber winsorization."""
    temperature_scores = np.clip(
        temperature_residuals / temperature_scale,
        -huber_c,
        huber_c,
    )
    moisture_scores = np.clip(
        moisture_residuals / moisture_scale,
        -huber_c,
        huber_c,
    )
    correlation = float(np.corrcoef(temperature_scores, moisture_scores)[0, 1])
    if not math.isfinite(correlation):
        return 0.0
    return float(np.clip(correlation, -0.95, 0.95))


def simulate_joint_ou_path(
    temperature_fit: RobustOUFit,
    moisture_fit: RobustOUFit,
    start_temperature_c: float,
    start_moisture_kg_kg: float,
    innovation_correlation: float,
    seed: int,
    sigma_multiplier: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate one reproducible exact-discrete, jointly correlated OU path."""
    if sigma_multiplier < 0.0:
        raise ValueError("sigma_multiplier must be non-negative.")
    times_s = np.arange(
        MEASURED_END_TIME_S,
        OU_END_TIME_S + 0.5 * OU_STEP_S,
        OU_STEP_S,
        dtype=float,
    )
    temperature = np.empty(times_s.size, dtype=float)
    moisture = np.empty(times_s.size, dtype=float)
    temperature[0] = start_temperature_c
    moisture[0] = start_moisture_kg_kg

    rng = np.random.default_rng(seed)
    independent = rng.standard_normal((times_s.size - 1, 2))
    correlated_temperature = independent[:, 0]
    correlated_moisture = (
        innovation_correlation * independent[:, 0]
        + math.sqrt(1.0 - innovation_correlation**2) * independent[:, 1]
    )
    for index in range(1, times_s.size):
        temperature[index] = (
            temperature_fit.equilibrium_mean
            + temperature_fit.discrete_phi
            * (temperature[index - 1] - temperature_fit.equilibrium_mean)
            + sigma_multiplier
            * temperature_fit.innovation_scale
            * correlated_temperature[index - 1]
        )
        moisture[index] = (
            moisture_fit.equilibrium_mean
            + moisture_fit.discrete_phi
            * (moisture[index - 1] - moisture_fit.equilibrium_mean)
            + sigma_multiplier
            * moisture_fit.innovation_scale
            * correlated_moisture[index - 1]
        )

    if np.any(temperature <= -273.15) or np.any(moisture <= 0.0):
        raise ValueError(
            "The simulated OU boundary contains a non-physical value; "
            "reduce --sigma-multiplier or change the random seed."
        )
    return times_s, temperature, moisture


def build_huber_ou_boundary(
    environment: q2.EnvironmentData,
    seed: int,
    sigma_multiplier: float,
) -> OUBoundaryProgram:
    mask = (
        (environment.source_times_s >= FIT_START_TIME_S)
        & (environment.source_times_s <= MEASURED_END_TIME_S)
    )
    fit_times = environment.source_times_s[mask]
    fit_temperature = environment.source_temperature_c[mask]
    fit_moisture = environment.source_moisture_kg_kg[mask]
    if (
        fit_times.size < 2
        or not math.isclose(float(fit_times[0]), FIT_START_TIME_S)
        or not math.isclose(float(fit_times[-1]), MEASURED_END_TIME_S)
    ):
        raise ValueError("Attachment 1 must contain the complete 3--4 h interval.")

    temperature_fit, temperature_residuals = fit_robust_ou(
        fit_times,
        fit_temperature,
        "oven_temperature",
        "degC",
    )
    moisture_fit, moisture_residuals = fit_robust_ou(
        fit_times,
        fit_moisture,
        "oven_moisture_concentration",
        "kg/kg",
    )
    correlation = robust_innovation_correlation(
        temperature_residuals,
        moisture_residuals,
        temperature_fit.innovation_scale,
        moisture_fit.innovation_scale,
    )
    start_temperature = float(
        np.interp(
            MEASURED_END_TIME_S,
            environment.source_times_s,
            environment.source_temperature_c,
        )
    )
    start_moisture = float(
        np.interp(
            MEASURED_END_TIME_S,
            environment.source_times_s,
            environment.source_moisture_kg_kg,
        )
    )
    ou_times, ou_temperature, ou_moisture = simulate_joint_ou_path(
        temperature_fit,
        moisture_fit,
        start_temperature,
        start_moisture,
        correlation,
        seed,
        sigma_multiplier,
    )
    return OUBoundaryProgram(
        source_times_s=environment.source_times_s,
        source_temperature_c=environment.source_temperature_c,
        source_moisture_kg_kg=environment.source_moisture_kg_kg,
        ou_times_s=ou_times,
        ou_temperature_c=ou_temperature,
        ou_moisture_kg_kg=ou_moisture,
        temperature_fit=temperature_fit,
        moisture_fit=moisture_fit,
        innovation_correlation=correlation,
        seed=seed,
        sigma_multiplier=sigma_multiplier,
    )


def write_ou_boundary_csv(path: Path, boundary: OUBoundaryProgram) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "time_s",
                "time_h",
                "oven_temperature_c",
                "oven_moisture_kg_kg",
            ]
        )
        for time_s, temperature, moisture in zip(
            boundary.ou_times_s,
            boundary.ou_temperature_c,
            boundary.ou_moisture_kg_kg,
        ):
            writer.writerow(
                [
                    f"{time_s:.0f}",
                    f"{time_s / 3600.0:.6f}",
                    f"{temperature:.8f}",
                    f"{moisture:.10f}",
                ]
            )


def _realized_summary(values: np.ndarray) -> dict[str, float]:
    # 首行是4 h实测衔接值，统计只使用4 h后的OU生成值。
    generated = values[1:]
    return {
        "mean": float(np.mean(generated)),
        "standard_deviation": float(np.std(generated, ddof=1)),
        "minimum": float(np.min(generated)),
        "maximum": float(np.max(generated)),
        "lag1_correlation": float(
            np.corrcoef(generated[:-1], generated[1:])[0, 1]
        ),
    }


def load_constant_mean_reference(repo_root: Path) -> dict | None:
    path = repo_root / "results" / "A_problem3_drying_time" / "validation_summary.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    return {
        "source": str(path),
        "drying_time_s": float(payload["drying_time"]["production_s"]),
        "drying_time_h": float(payload["drying_time"]["production_h"]),
    }


def build_validation(
    repo_root: Path,
    boundary: OUBoundaryProgram,
    production: base.LongDryingResult,
    coarse: base.LongDryingResult,
    table_times_s: np.ndarray,
    production_table: np.ndarray,
) -> dict:
    validation = base.build_validation(
        boundary,
        production,
        coarse,
        table_times_s,
        production_table,
    )
    validation["model_scope"]["post_4h_boundary"] = (
        "Huber-calibrated exact-discrete joint OU path at 60 s intervals"
    )
    validation["model_scope"]["ou_path_interval_s"] = OU_STEP_S
    validation["model_scope"]["ou_path_end_time_h"] = OU_END_TIME_S / 3600.0
    validation["model_scope"]["random_seed"] = boundary.seed
    validation["model_scope"]["sigma_multiplier"] = boundary.sigma_multiplier
    time_step_counts = validation["checks"]["time_step_counts"]
    time_step_counts["ou_environment"] = time_step_counts.pop(
        "constant_environment"
    )
    saved_increments = np.diff(production.moisture_kg_kg, axis=0)
    increase_count = int(np.count_nonzero(saved_increments > 1.0e-8))
    maximum_increase = float(max(np.max(saved_increments), 0.0))
    validation["checks"].pop("time_monotonicity_violation_count")
    validation["checks"]["saved_pointwise_moisture_increase_count"] = increase_count
    validation["checks"]["maximum_saved_pointwise_increase_kg_kg"] = maximum_increase
    validation["checks"]["pointwise_monotonicity_note"] = (
        "Small local increases are admissible under a stochastic ambient-moisture "
        "boundary and are not treated as numerical monotonicity violations."
    )
    validation["ou_equilibrium"] = validation.pop("constant_boundary")
    validation["ou_calibration"] = {
        "fit_interval_s": [FIT_START_TIME_S, MEASURED_END_TIME_S],
        "temperature": asdict(boundary.temperature_fit),
        "moisture": asdict(boundary.moisture_fit),
        "robust_innovation_correlation": boundary.innovation_correlation,
        "interpretation": (
            "phi at its lower bound means correlation decays below the "
            "one-minute observation resolution; the discrete path is then "
            "practically white noise around the Huber equilibrium."
        ),
    }
    validation["realized_ou_boundary_4h_72h"] = {
        "temperature_c": _realized_summary(boundary.ou_temperature_c),
        "moisture_kg_kg": _realized_summary(boundary.ou_moisture_kg_kg),
    }
    reference = load_constant_mean_reference(repo_root)
    if reference is not None:
        reference["delta_ou_minus_mean_s"] = (
            production.drying_time_s - reference["drying_time_s"]
        )
        reference["delta_ou_minus_mean_min"] = (
            production.drying_time_s - reference["drying_time_s"]
        ) / 60.0
        reference["relative_change_pct"] = 100.0 * (
            production.drying_time_s / reference["drying_time_s"] - 1.0
        )
    validation["comparison_with_constant_mean_model"] = reference
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
    module_path = str(bundled_modules) if bundled_modules.exists() else None
    return executable, module_path


def build_result3_workbook(
    script_path: Path,
    repo_root: Path,
    result_dir: Path,
) -> Path:
    builder = script_path.with_name("build_result3_from_csv.cjs")
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
    output = result_dir / "result3.xlsx"
    preview_dir = repo_root / "tmp" / "A_problem3_huber_ou_verify"
    if not builder.exists() or not template.exists():
        raise FileNotFoundError("Cannot find the result3 builder or template workbook.")
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
    if completed.stdout.strip():
        print(completed.stdout.strip(), flush=True)
    return output


def update_ou_history_workbook(
    script_path: Path,
    result_dir: Path,
) -> Path:
    """Create or append the current OU run to a persistent Excel workbook."""
    builder = script_path.with_name("update_ou_boundary_workbook.cjs")
    boundary_csv = result_dir / "ou_boundary_4h_72h_60s.csv"
    parameters_json = result_dir / "ou_parameters.json"
    validation_json = result_dir / "validation_summary.json"
    output = result_dir / "ou_boundary_runs.xlsx"
    preview_dir = result_dir.parent.parent / "tmp" / "A_problem3_ou_history_verify"
    if not builder.exists():
        raise FileNotFoundError(f"Cannot find OU history workbook builder: {builder}")
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
            str(boundary_csv),
            str(parameters_json),
            str(validation_json),
            str(output),
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
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="附件1路径")
    parser.add_argument("--repo-root", type=Path, help="项目根目录")
    parser.add_argument("--output-dir", type=Path, help="结果输出目录")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--randomize-seed",
        action="store_true",
        help="每次从系统熵生成并记录一个新随机种子，用于观察路径间差异",
    )
    parser.add_argument(
        "--sigma-multiplier",
        type=float,
        default=1.0,
        help="OU创新标准差倍率，默认1.0",
    )
    parser.add_argument(
        "--skip-xlsx",
        action="store_true",
        help="只生成CSV/JSON，不构建result3.xlsx",
    )
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    repo_root = args.repo_root.resolve() if args.repo_root else script_path.parents[1]
    data_path = args.data.resolve() if args.data else q2.find_default_data_path(repo_root)
    result_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else repo_root / "results" / "A_problem3_drying_time_huber_ou"
    )
    result_dir.mkdir(parents=True, exist_ok=True)

    actual_seed = secrets.randbits(32) if args.randomize_seed else args.seed
    environment = q2.load_and_preprocess_environment(data_path)
    boundary = build_huber_ou_boundary(
        environment,
        seed=actual_seed,
        sigma_multiplier=args.sigma_multiplier,
    )
    # 粗、细网格使用同一条OU路径，避免把随机路径差异误当作网格误差。
    coarse = base.simulate_until_dry(
        boundary,
        nominal_radial_step_cm=0.025,
        early_time_step_s=2.0,
        late_time_step_s=60.0,
        final_time_step_s=2.0,
    )
    production = base.simulate_until_dry(
        boundary,
        nominal_radial_step_cm=0.0125,
        early_time_step_s=1.0,
        late_time_step_s=30.0,
        final_time_step_s=1.0,
    )

    table_times = base.build_table_times(production.drying_time_s)
    table = base.table_values(production, table_times)
    base.write_full_csv(result_dir / "moisture_full_60s_0p1cm.csv", production)
    base.write_temperature_csv(
        result_dir / "temperature_auxiliary_60s_0p1cm.csv",
        production,
    )
    base.write_table_csv(
        result_dir / "table5_moisture.csv",
        table_times,
        table,
        production.drying_time_s,
    )
    write_ou_boundary_csv(result_dir / "ou_boundary_4h_72h_60s.csv", boundary)

    parameters = {
        "method": (
            "Huber Proposal-2 equilibrium and scale; bounded Huber IRLS for "
            "the exact-discrete OU/AR(1) coefficient"
        ),
        "fit_interval_s": [FIT_START_TIME_S, MEASURED_END_TIME_S],
        "path_interval_s": OU_STEP_S,
        "path_end_time_s": OU_END_TIME_S,
        "random_seed": boundary.seed,
        "sigma_multiplier": boundary.sigma_multiplier,
        "temperature": asdict(boundary.temperature_fit),
        "moisture": asdict(boundary.moisture_fit),
        "robust_innovation_correlation": boundary.innovation_correlation,
    }
    with (result_dir / "ou_parameters.json").open("w", encoding="utf-8") as stream:
        json.dump(parameters, stream, ensure_ascii=False, indent=2)

    validation = build_validation(
        repo_root,
        boundary,
        production,
        coarse,
        table_times,
        table,
    )
    with (result_dir / "validation_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(validation, stream, ensure_ascii=False, indent=2)

    workbook_path = None
    boundary_history_path = None
    if not args.skip_xlsx:
        workbook_path = build_result3_workbook(script_path, repo_root, result_dir)
        boundary_history_path = update_ou_history_workbook(script_path, result_dir)

    print(
        "Huber-OU equilibrium: "
        f"{boundary.temperature_fit.equilibrium_mean:.8f} deg C, "
        f"{boundary.moisture_fit.equilibrium_mean:.10f} kg/kg",
        flush=True,
    )
    print(
        "Discrete phi: "
        f"temperature={boundary.temperature_fit.discrete_phi:.8g}, "
        f"moisture={boundary.moisture_fit.discrete_phi:.8g}; "
        f"innovation correlation={boundary.innovation_correlation:.6f}",
        flush=True,
    )
    print(
        f"Drying time: {production.drying_time_s:.0f} s "
        f"= {production.drying_time_s / 3600.0:.6f} h",
        flush=True,
    )
    print("\nTable 5: moisture concentration (kg/kg)", flush=True)
    for time_s, row in zip(table_times, table):
        label = (
            "end"
            if math.isclose(time_s, production.drying_time_s)
            else f"{time_s / 3600.0:.0f} h"
        )
        print(label, np.array2string(row, precision=4), flush=True)
    if workbook_path is not None:
        print(f"result3 workbook: {workbook_path}", flush=True)
    if boundary_history_path is not None:
        print(f"OU boundary history workbook: {boundary_history_path}", flush=True)
    print(f"Results: {result_dir}", flush=True)


if __name__ == "__main__":
    main()
