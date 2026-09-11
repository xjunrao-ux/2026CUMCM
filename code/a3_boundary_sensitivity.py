"""A题问题3：烘房恒值边界的拟合与灵敏度检验。

对照两类恒温干燥边界参数：
1. 原模型：附件1中3--4 h观测值的时间加权均值；
2. 鲁棒渐近模型：y(t)=y_inf+A exp[-(t-t0)/tau]，使用Huber型损失，
   并以3--4 h为留出集，从候选起点中选择预测误差最小者。

脚本还把边界拟合置信区间、实际控制扰动和恒温阶段切换时刻传入第三问
有限体积模型。筛选计算使用问题3已有的粗网格；推荐替代参数另用生产网格
复算，以免把参数灵敏度和离散误差混在一起。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

import a2_coupled_fvm as q2
import a3_drying_time_fvm as q3


FIT_START_CANDIDATES_H = (0.5, 1.0, 1.5, 2.0)
HOLDOUT_START_H = 3.0
TAIL_START_H = 3.5
STABILITY_SWITCH_TIME_S = 100.0 * 60.0
TEMPERATURE_PERTURBATION_C = 0.5
MOISTURE_PERTURBATION_KG_KG = 0.001
BOOTSTRAP_BLOCK_LENGTH = 5
BOOTSTRAP_SEED = 2026


@dataclass
class AsymptoticFit:
    selected_start_h: float
    plateau: float
    amplitude: float
    time_constant_h: float
    fit_rmse: float
    holdout_rmse: float
    bootstrap_standard_error: float
    bootstrap_ci95_lower: float
    bootstrap_ci95_upper: float
    holdout_candidates: list[dict[str, float]]


@dataclass
class ConfigurableBoundary:
    source_times_s: np.ndarray
    source_temperature_c: np.ndarray
    source_moisture_kg_kg: np.ndarray
    plateau_temperature_c: float
    plateau_moisture_kg_kg: float
    switch_time_s: float

    def values(self, time_s: float) -> tuple[float, float]:
        if time_s <= self.switch_time_s:
            return (
                float(np.interp(time_s, self.source_times_s, self.source_temperature_c)),
                float(np.interp(time_s, self.source_times_s, self.source_moisture_kg_kg)),
            )
        return self.plateau_temperature_c, self.plateau_moisture_kg_kg


def _tail_scale(values: np.ndarray) -> float:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    robust_sigma = 1.4826 * mad
    ordinary_sigma = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    return max(robust_sigma, 0.25 * ordinary_sigma, np.finfo(float).eps)


def fit_first_order(
    times_h: np.ndarray,
    values: np.ndarray,
    start_h: float,
    end_h: float,
) -> tuple[np.ndarray, float]:
    mask = (times_h >= start_h) & (times_h <= end_h)
    times = times_h[mask]
    observations = values[mask]
    if observations.size < 20:
        raise ValueError("At least 20 records are required for asymptotic fitting.")

    local_times = times - start_h
    tail = observations[times >= max(start_h, end_h - 0.5)]
    plateau0 = float(np.median(tail))
    amplitude0 = min(float(observations[0] - plateau0), -np.finfo(float).eps)
    data_range = max(float(np.ptp(observations)), _tail_scale(tail))
    lower = np.array(
        [float(np.min(observations) - data_range), -3.0 * data_range, 1.0 / 60.0]
    )
    upper = np.array(
        [float(np.max(observations) + data_range), 0.0, 12.0]
    )
    result = least_squares(
        lambda p: p[0] + p[1] * np.exp(-local_times / p[2]) - observations,
        x0=np.array([plateau0, amplitude0, 0.5]),
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=_tail_scale(tail),
        max_nfev=20_000,
    )
    if not result.success:
        raise RuntimeError(f"Robust asymptotic fit failed: {result.message}")
    prediction = result.x[0] + result.x[1] * np.exp(-local_times / result.x[2])
    rmse = float(np.sqrt(np.mean((prediction - observations) ** 2)))
    return result.x, rmse


def select_fit_start(
    times_h: np.ndarray, values: np.ndarray
) -> tuple[float, float, list[dict[str, float]]]:
    holdout = times_h > HOLDOUT_START_H
    candidate_rows: list[dict[str, float]] = []
    for start_h in FIT_START_CANDIDATES_H:
        parameters, training_rmse = fit_first_order(
            times_h, values, start_h, HOLDOUT_START_H
        )
        local_holdout_times = times_h[holdout] - start_h
        prediction = parameters[0] + parameters[1] * np.exp(
            -local_holdout_times / parameters[2]
        )
        holdout_rmse = float(np.sqrt(np.mean((prediction - values[holdout]) ** 2)))
        candidate_rows.append(
            {
                "start_h": float(start_h),
                "training_rmse": training_rmse,
                "holdout_rmse": holdout_rmse,
                "training_plateau": float(parameters[0]),
            }
        )
    selected = min(candidate_rows, key=lambda row: row["holdout_rmse"])
    return float(selected["start_h"]), float(selected["holdout_rmse"]), candidate_rows


def bootstrap_plateau(
    times_h: np.ndarray,
    values: np.ndarray,
    start_h: float,
    fitted_parameters: np.ndarray,
    replications: int,
    seed: int,
) -> tuple[float, float, float]:
    mask = times_h >= start_h
    times = times_h[mask]
    observations = values[mask]
    local_times = times - start_h
    prediction = fitted_parameters[0] + fitted_parameters[1] * np.exp(
        -local_times / fitted_parameters[2]
    )
    residuals = observations - prediction
    residuals -= float(np.mean(residuals))
    rng = np.random.default_rng(seed)
    block_count = int(math.ceil(residuals.size / BOOTSTRAP_BLOCK_LENGTH))
    maximum_start = residuals.size - BOOTSTRAP_BLOCK_LENGTH + 1
    estimates: list[float] = []
    for _ in range(replications):
        starts = rng.integers(0, maximum_start, size=block_count)
        sampled = np.concatenate(
            [
                residuals[index : index + BOOTSTRAP_BLOCK_LENGTH]
                for index in starts
            ]
        )[: residuals.size]
        synthetic = prediction + sampled
        try:
            parameters, _ = fit_first_order(
                times,
                synthetic,
                float(times[0]),
                float(times[-1]),
            )
        except (RuntimeError, ValueError):
            continue
        estimates.append(float(parameters[0]))
    if len(estimates) < max(50, int(0.9 * replications)):
        raise RuntimeError("Too many bootstrap fits failed.")
    estimate_array = np.asarray(estimates)
    lower, upper = np.quantile(estimate_array, [0.025, 0.975])
    return float(np.std(estimate_array, ddof=1)), float(lower), float(upper)


def fit_asymptote(
    times_h: np.ndarray,
    values: np.ndarray,
    bootstrap_replications: int,
    seed: int,
) -> AsymptoticFit:
    selected_start, holdout_rmse, candidates = select_fit_start(times_h, values)
    parameters, fit_rmse = fit_first_order(
        times_h, values, selected_start, float(times_h[-1])
    )
    standard_error, ci_lower, ci_upper = bootstrap_plateau(
        times_h,
        values,
        selected_start,
        parameters,
        bootstrap_replications,
        seed,
    )
    return AsymptoticFit(
        selected_start_h=selected_start,
        plateau=float(parameters[0]),
        amplitude=float(parameters[1]),
        time_constant_h=float(parameters[2]),
        fit_rmse=fit_rmse,
        holdout_rmse=holdout_rmse,
        bootstrap_standard_error=standard_error,
        bootstrap_ci95_lower=ci_lower,
        bootstrap_ci95_upper=ci_upper,
        holdout_candidates=candidates,
    )


def time_weighted_mean(
    times_s: np.ndarray, values: np.ndarray, start_s: float, end_s: float
) -> float:
    mask = (times_s >= start_s) & (times_s <= end_s)
    selected_times = times_s[mask]
    selected_values = values[mask]
    if selected_times.size < 2 or not math.isclose(selected_times[0], start_s):
        raise ValueError("The requested averaging interval is incomplete.")
    if not math.isclose(selected_times[-1], end_s):
        raise ValueError("The requested averaging interval is incomplete.")
    return float(np.trapezoid(selected_values, selected_times) / (end_s - start_s))


def linear_slope_per_hour(
    times_s: np.ndarray, values: np.ndarray, start_s: float
) -> float:
    mask = times_s >= start_s
    slope_per_second = float(np.polyfit(times_s[mask], values[mask], deg=1)[0])
    return slope_per_second * 3600.0


def make_boundary(
    environment: q2.EnvironmentData,
    temperature_c: float,
    moisture_kg_kg: float,
    switch_time_s: float,
) -> ConfigurableBoundary:
    return ConfigurableBoundary(
        source_times_s=environment.source_times_s,
        source_temperature_c=environment.source_temperature_c,
        source_moisture_kg_kg=environment.source_moisture_kg_kg,
        plateau_temperature_c=temperature_c,
        plateau_moisture_kg_kg=moisture_kg_kg,
        switch_time_s=switch_time_s,
    )


def run_screening_case(payload: dict) -> dict:
    data_path = Path(payload["data_path"])
    environment = q2.load_and_preprocess_environment(data_path)
    boundary = make_boundary(
        environment,
        float(payload["temperature_c"]),
        float(payload["moisture_kg_kg"]),
        float(payload["switch_time_s"]),
    )
    result = q3.simulate_until_dry(
        boundary,
        nominal_radial_step_cm=0.025,
        early_time_step_s=2.0,
        late_time_step_s=60.0,
        final_time_step_s=2.0,
    )
    return {
        **payload,
        "drying_time_s": float(result.drying_time_s),
        "drying_time_h": float(result.drying_time_s / 3600.0),
        "controlling_radius_cm": float(result.controlling_radius_cm),
    }


def run_production_case(
    environment: q2.EnvironmentData,
    temperature_c: float,
    moisture_kg_kg: float,
    switch_time_s: float,
) -> dict[str, float]:
    boundary = make_boundary(
        environment, temperature_c, moisture_kg_kg, switch_time_s
    )
    result = q3.simulate_until_dry(
        boundary,
        nominal_radial_step_cm=0.0125,
        early_time_step_s=1.0,
        late_time_step_s=30.0,
        final_time_step_s=1.0,
    )
    return {
        "drying_time_s": float(result.drying_time_s),
        "drying_time_h": float(result.drying_time_s / 3600.0),
        "controlling_radius_cm": float(result.controlling_radius_cm),
    }


def load_baseline_production_time(repo_root: Path) -> float | None:
    path = repo_root / "results" / "A_problem3_drying_time" / "validation_summary.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    return float(payload["drying_time"]["production_s"])


def compare_cleaned_copy(
    repo_root: Path, original: q2.EnvironmentData
) -> tuple[str | None, bool | None]:
    cleaned_path = repo_root / "数据清洗" / "附件1_清洗后.xlsx"
    if not cleaned_path.exists():
        return None, None
    cleaned = q2.load_and_preprocess_environment(cleaned_path)
    identical = all(
        np.array_equal(left, right)
        for left, right in [
            (original.source_times_s, cleaned.source_times_s),
            (original.source_temperature_c, cleaned.source_temperature_c),
            (original.source_moisture_kg_kg, cleaned.source_moisture_kg_kg),
        ]
    )
    return str(cleaned_path), bool(identical)


def write_csv(path: Path, rows: list[dict]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--skip-production", action="store_true")
    args = parser.parse_args()
    if args.bootstrap < 100:
        raise ValueError("Use at least 100 bootstrap replications.")

    script_path = Path(__file__).resolve()
    repo_root = args.repo_root.resolve() if args.repo_root else script_path.parents[1]
    data_path = args.data.resolve() if args.data else q2.find_default_data_path(repo_root)
    result_dir = repo_root / "results" / "A_problem3_boundary_sensitivity"
    result_dir.mkdir(parents=True, exist_ok=True)

    environment = q2.load_and_preprocess_environment(data_path)
    cleaned_path, cleaned_identical = compare_cleaned_copy(repo_root, environment)
    times_h = environment.source_times_s / 3600.0
    baseline_temperature = time_weighted_mean(
        environment.source_times_s,
        environment.source_temperature_c,
        q3.PLATEAU_START_TIME_S,
        q3.MEASURED_END_TIME_S,
    )
    baseline_moisture = time_weighted_mean(
        environment.source_times_s,
        environment.source_moisture_kg_kg,
        q3.PLATEAU_START_TIME_S,
        q3.MEASURED_END_TIME_S,
    )
    tail_mask = environment.source_times_s >= TAIL_START_H * 3600.0
    tail_temperature = float(np.mean(environment.source_temperature_c[tail_mask]))
    tail_moisture = float(np.mean(environment.source_moisture_kg_kg[tail_mask]))

    temperature_fit = fit_asymptote(
        times_h,
        environment.source_temperature_c,
        args.bootstrap,
        BOOTSTRAP_SEED,
    )
    moisture_fit = fit_asymptote(
        times_h,
        environment.source_moisture_kg_kg,
        args.bootstrap,
        BOOTSTRAP_SEED + 1,
    )

    def case(
        name: str,
        temperature_c: float,
        moisture_kg_kg: float,
        switch_time_s: float = q3.MEASURED_END_TIME_S,
        description: str = "",
    ) -> dict:
        return {
            "case": name,
            "description": description,
            "data_path": str(data_path),
            "temperature_c": float(temperature_c),
            "moisture_kg_kg": float(moisture_kg_kg),
            "switch_time_s": float(switch_time_s),
        }

    cases = [
        case(
            "baseline_mean_3_4h",
            baseline_temperature,
            baseline_moisture,
            description="原模型：3--4 h时间加权均值，4 h后切换恒值边界",
        ),
        case(
            "robust_asymptote",
            temperature_fit.plateau,
            moisture_fit.plateau,
            description="留出检验选窗的Huber鲁棒一阶渐近拟合，4 h后切换",
        ),
        case(
            "tail30_mean_switch_100min",
            tail_temperature,
            tail_moisture,
            STABILITY_SWITCH_TIME_S,
            "补充清洗记录口径：末30 min均值，100 min后切换恒值边界",
        ),
        case(
            "switch_100min_only",
            baseline_temperature,
            baseline_moisture,
            STABILITY_SWITCH_TIME_S,
            "仅提前恒值边界切换时刻，恒值仍取原模型均值",
        ),
        case("temperature_minus", baseline_temperature - TEMPERATURE_PERTURBATION_C, baseline_moisture),
        case("temperature_plus", baseline_temperature + TEMPERATURE_PERTURBATION_C, baseline_moisture),
        case("moisture_minus", baseline_temperature, baseline_moisture - MOISTURE_PERTURBATION_KG_KG),
        case("moisture_plus", baseline_temperature, baseline_moisture + MOISTURE_PERTURBATION_KG_KG),
        case(
            "fit_ci_fast_corner",
            temperature_fit.bootstrap_ci95_upper,
            moisture_fit.bootstrap_ci95_lower,
            description="拟合95%区间中较快干燥的边界角点",
        ),
        case(
            "fit_ci_slow_corner",
            temperature_fit.bootstrap_ci95_lower,
            moisture_fit.bootstrap_ci95_upper,
            description="拟合95%区间中较慢干燥的边界角点",
        ),
    ]

    screening_rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(run_screening_case, row): row["case"] for row in cases}
        for future in as_completed(futures):
            result = future.result()
            screening_rows.append(result)
            print(
                f"screening {result['case']}: {result['drying_time_h']:.6f} h",
                flush=True,
            )
    case_order = {row["case"]: index for index, row in enumerate(cases)}
    screening_rows.sort(key=lambda row: case_order[row["case"]])
    baseline_screening_s = next(
        row["drying_time_s"]
        for row in screening_rows
        if row["case"] == "baseline_mean_3_4h"
    )
    for row in screening_rows:
        row["delta_vs_baseline_min"] = (
            row["drying_time_s"] - baseline_screening_s
        ) / 60.0
        row.pop("data_path")

    row_by_case = {row["case"]: row for row in screening_rows}
    temperature_derivative = (
        row_by_case["temperature_plus"]["drying_time_h"]
        - row_by_case["temperature_minus"]["drying_time_h"]
    ) / (2.0 * TEMPERATURE_PERTURBATION_C)
    moisture_derivative = (
        row_by_case["moisture_plus"]["drying_time_h"]
        - row_by_case["moisture_minus"]["drying_time_h"]
    ) / (2.0 * MOISTURE_PERTURBATION_KG_KG)
    baseline_screening_h = baseline_screening_s / 3600.0
    temperature_elasticity_kelvin = (
        temperature_derivative
        * (baseline_temperature + 273.15)
        / baseline_screening_h
    )
    moisture_elasticity = (
        moisture_derivative * baseline_moisture / baseline_screening_h
    )

    production_baseline_s = load_baseline_production_time(repo_root)
    production_alternative = None
    if not args.skip_production:
        print("production robust_asymptote: running", flush=True)
        production_alternative = run_production_case(
            environment,
            temperature_fit.plateau,
            moisture_fit.plateau,
            q3.MEASURED_END_TIME_S,
        )
        if production_baseline_s is not None:
            production_alternative["delta_vs_existing_baseline_min"] = (
                production_alternative["drying_time_s"] - production_baseline_s
            ) / 60.0
            production_alternative["relative_change_pct"] = 100.0 * (
                production_alternative["drying_time_s"] / production_baseline_s - 1.0
            )
        print(
            "production robust_asymptote: "
            f"{production_alternative['drying_time_h']:.6f} h",
            flush=True,
        )

    report = {
        "data_integrity": {
            "record_count": int(environment.source_times_s.size),
            "sampling_interval_s": float(np.median(np.diff(environment.source_times_s))),
            "cleaned_copy_path": cleaned_path,
            "cleaned_and_original_numeric_values_identical": cleaned_identical,
            "smoothing_used": False,
        },
        "baseline_time_weighted_3_4h": {
            "temperature_c": baseline_temperature,
            "moisture_kg_kg": baseline_moisture,
            "switch_time_s": q3.MEASURED_END_TIME_S,
        },
        "tail_30min_arithmetic_mean": {
            "temperature_c": tail_temperature,
            "moisture_kg_kg": tail_moisture,
        },
        "plateau_diagnostics_3_4h": {
            "temperature_slope_c_per_h": linear_slope_per_hour(
                environment.source_times_s,
                environment.source_temperature_c,
                q3.PLATEAU_START_TIME_S,
            ),
            "moisture_slope_kg_kg_per_h": linear_slope_per_hour(
                environment.source_times_s,
                environment.source_moisture_kg_kg,
                q3.PLATEAU_START_TIME_S,
            ),
        },
        "robust_asymptotic_fit": {
            "model": "y(t)=y_inf+A*exp(-(t-t0)/tau), A<=0; soft-L1 robust loss",
            "selection": "fit through 3 h; choose t0 by minimum 3--4 h holdout RMSE; refit through 4 h",
            "bootstrap": {
                "method": "moving-block residual bootstrap",
                "replications": args.bootstrap,
                "block_length_records": BOOTSTRAP_BLOCK_LENGTH,
            },
            "temperature": asdict(temperature_fit),
            "moisture": asdict(moisture_fit),
        },
        "screening_grid": {
            "nominal_radial_step_cm": 0.025,
            "time_steps_s": [2.0, 60.0, 2.0],
            "cases": screening_rows,
        },
        "local_sensitivity": {
            "temperature_step_c": TEMPERATURE_PERTURBATION_C,
            "moisture_step_kg_kg": MOISTURE_PERTURBATION_KG_KG,
            "d_drying_time_h_per_temperature_c": temperature_derivative,
            "d_drying_time_h_per_moisture_kg_kg": moisture_derivative,
            "temperature_elasticity_using_kelvin": temperature_elasticity_kelvin,
            "moisture_elasticity": moisture_elasticity,
        },
        "production_comparison": {
            "existing_baseline_drying_time_s": production_baseline_s,
            "existing_baseline_drying_time_h": (
                None if production_baseline_s is None else production_baseline_s / 3600.0
            ),
            "robust_asymptote": production_alternative,
        },
        "interpretation": {
            "primary_conclusion": (
                "Use the robust asymptotic fit as a sensitivity comparator, not as proof that the "
                "time-weighted mean is wrong; the observed final hour is already stationary."
            ),
            "parameter_priority": (
                "Compare normalized elasticities and scenario deltas; Celsius percentages are not used."
            ),
        },
    }
    write_csv(result_dir / "boundary_sensitivity_cases.csv", screening_rows)
    with (result_dir / "boundary_fit_sensitivity_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(report, ensure_ascii=True, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
