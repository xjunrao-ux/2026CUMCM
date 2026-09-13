"""A题问题3：以OU随机过程为依据的稳健性分析（恒值均值做法 vs OU随机做法）。

两种做法对4 h后的烘房边界作出不同假设，其余模型（附件3水热耦合有限体积、
药材物性、数值格式）完全一致：

  * 做法一（基准，a3_drying_time_fvm.py）：4 h后温度、水分浓度固定为附件1
    3--4 h的时间加权均值（确定性恒值边界）；
  * 做法二（a3_drying_time_fvm_huber_ou.py）：4 h后按3--4 h稳定段Huber稳健
    标定的精确离散OU/AR(1)过程逐分钟生成联合随机边界。

稳健性分析全部在已验证的粗网格（0.025 cm，2/60/2 s）上进行，并统一施加
细网格基准的粗-细修正。设计：

  * sigma=1.0（标定噪声水平）做N=60的主蒙特卡洛，得到干燥时间与内部水分
    轨迹的抽样分布；
  * sigma in {0, 0.5, 1.5, 2.0} 做噪声水平扫描（sigma=0为OU的确定性极限）；
  * 另加恒定Huber均衡边界，把"标定口径差异"与"随机波动差异"分解开。

输出：
  results/A_problem3_ou_robustness/  检查点CSV、轨迹npz、汇总JSON与说明MD；
  picture/A_problem3_ou_robustness/ 4张稳健性分析图（PNG/PDF/SVG）。

参考：Che Taib & Darus (2025), doi:10.11113/matematika.v41.n1.1610。
该文以OU过程描述温度偏离，并进一步令均值回复速度随机变化。本题稳定段仅有
61个逐分钟观测，无法可靠辨识嵌套的Lévy驱动随机回复速度，故采用可辨识、可复现
的一阶OU/AR(1)边界，并以蒙特卡洛传播边界不确定性。
"""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import numpy as np
import pandas as pd

import a2_coupled_fvm as q2
import a3_drying_time_fvm as q3
import a3_drying_time_fvm_huber_ou as oum


DEFAULT_SEED = 20260912
SEED_STEP = 1000
SIGMA_MAIN = 1.0
SEEDS_MAIN = 60
SEEDS_SWEEP = 24
SIGMA_SWEEP_VALUES = [0.5, 1.5, 2.0]
SAMPLE_PATH_SEEDS = 12
TRAJECTORY_SEEDS = 12

PALETTE = {
    "blue": "#0F4D92",
    "blue_2": "#3775BA",
    "teal": "#42949E",
    "orange": "#E28E2C",
    "red": "#B64342",
    "violet": "#7C6CCF",
    "gray": "#767676",
    "light": "#E7E9ED",
}

# Chinese display labels have no capitalization distinction; named constants keep
# that language-specific fact separate from generic legend-case linting.
LABEL_MEASURED = "附件1实测 (3--4 h)"
LABEL_DRYING_THRESHOLD = "干燥阈值 0.15 kg/kg"
LABEL_RANGE = "极差 (max−min)"


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Arial",
                "DejaVu Sans",
            ],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def save_png(fig: plt.Figure, path: Path) -> None:
    """Export a high-resolution preview plus editable vector formats."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.06)
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.06)
    fig.savefig(
        path.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pad_inches=0.06,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def interp_monotone(
    x: float | np.ndarray,
    xp: np.ndarray,
    fp: np.ndarray,
) -> float | np.ndarray:
    """Linear interpolation with an explicit strictly increasing-grid guard."""
    xp = np.asarray(xp, dtype=float)
    fp = np.asarray(fp, dtype=float)
    if xp.ndim != 1 or fp.ndim != 1 or xp.size != fp.size:
        raise ValueError("Interpolation grids must be equally sized one-dimensional arrays.")
    if not np.all(np.diff(xp) > 0.0):
        raise ValueError("Interpolation grid must be strictly increasing.")
    return np.interp(x, xp, fp)


@dataclass
class MCBoundary:
    """4 h前用附件1实测插值，4 h后用逐分钟OU路径线性插值（与做法二相同）。"""

    source_times_s: np.ndarray
    source_temperature_c: np.ndarray
    source_moisture_kg_kg: np.ndarray
    ou_times_s: np.ndarray
    ou_temperature_c: np.ndarray
    ou_moisture_kg_kg: np.ndarray

    def values(self, time_s: float) -> tuple[float, float]:
        if time_s <= oum.MEASURED_END_TIME_S:
            return (
                float(
                    interp_monotone(time_s, self.source_times_s, self.source_temperature_c)
                ),
                float(
                    interp_monotone(time_s, self.source_times_s, self.source_moisture_kg_kg)
                ),
            )
        return (
            float(interp_monotone(time_s, self.ou_times_s, self.ou_temperature_c)),
            float(interp_monotone(time_s, self.ou_times_s, self.ou_moisture_kg_kg)),
        )


def seed_for(sigma: float, index: int) -> int:
    return DEFAULT_SEED + int(round(10.0 * sigma)) * SEED_STEP + index


def fit_ou(environment: q2.EnvironmentData) -> dict:
    mask = (
        (environment.source_times_s >= oum.FIT_START_TIME_S)
        & (environment.source_times_s <= oum.MEASURED_END_TIME_S)
    )
    fit_times = environment.source_times_s[mask]
    temperature_fit, temperature_residuals = oum.fit_robust_ou(
        fit_times, environment.source_temperature_c[mask], "oven_temperature", "degC"
    )
    moisture_fit, moisture_residuals = oum.fit_robust_ou(
        fit_times,
        environment.source_moisture_kg_kg[mask],
        "oven_moisture_concentration",
        "kg/kg",
    )
    correlation = oum.robust_innovation_correlation(
        temperature_residuals,
        moisture_residuals,
        temperature_fit.innovation_scale,
        moisture_fit.innovation_scale,
    )
    start_temperature = float(
        interp_monotone(
            oum.MEASURED_END_TIME_S,
            environment.source_times_s,
            environment.source_temperature_c,
        )
    )
    start_moisture = float(
        interp_monotone(
            oum.MEASURED_END_TIME_S,
            environment.source_times_s,
            environment.source_moisture_kg_kg,
        )
    )
    return {
        "temperature_fit": temperature_fit,
        "moisture_fit": moisture_fit,
        "temperature_residuals": temperature_residuals,
        "moisture_residuals": moisture_residuals,
        "innovation_correlation": correlation,
        "start_temperature_c": start_temperature,
        "start_moisture_kg_kg": start_moisture,
    }


def build_payloads(
    environment: q2.EnvironmentData,
    fit: dict,
    record_trajectory: bool,
) -> list[dict]:
    source = {
        "source_times_s": environment.source_times_s,
        "source_temperature_c": environment.source_temperature_c,
        "source_moisture_kg_kg": environment.source_moisture_kg_kg,
    }
    baseline = q3.build_boundary_program(environment)
    temperature_fit = fit["temperature_fit"]
    moisture_fit = fit["moisture_fit"]
    payloads: list[dict] = []

    # 做法一基准：3--4 h时间加权均值恒值边界。
    payloads.append(
        {
            "case_key": "baseline_mean_3_4h",
            "kind": "constant",
            "sigma_multiplier": 0.0,
            "seed": 0,
            "plateau_temperature_c": baseline.plateau_temperature_c,
            "plateau_moisture_kg_kg": baseline.plateau_moisture_kg_kg,
            "record_trajectory": record_trajectory,
            **source,
        }
    )

    def ou_payload(key: str, sigma: float, seed: int, record: bool) -> dict:
        times, temperature, moisture = oum.simulate_joint_ou_path(
            temperature_fit,
            moisture_fit,
            fit["start_temperature_c"],
            fit["start_moisture_kg_kg"],
            fit["innovation_correlation"],
            seed,
            sigma,
        )
        return {
            "case_key": key,
            "kind": "ou",
            "sigma_multiplier": sigma,
            "seed": seed,
            "ou_times_s": times,
            "ou_temperature_c": temperature,
            "ou_moisture_kg_kg": moisture,
            "record_trajectory": record,
            **source,
        }

    # 做法二的确定性极限（sigma=0，无噪声OU路径），不记轨迹以免混入sigma=1样本带。
    payloads.append(
        ou_payload("ou_s0_limit", 0.0, DEFAULT_SEED, False)
    )
    # 恒定Huber均衡边界：把标定口径差异（时间加权均值 vs Huber均衡）分离出来。
    payloads.append(
        {
            "case_key": "constant_huber_equilibrium",
            "kind": "constant",
            "sigma_multiplier": 0.0,
            "seed": 0,
            "plateau_temperature_c": temperature_fit.equilibrium_mean,
            "plateau_moisture_kg_kg": moisture_fit.equilibrium_mean,
            "record_trajectory": False,
            **source,
        }
    )

    for index in range(SEEDS_MAIN):
        payloads.append(
            ou_payload(
                f"ou_s{SIGMA_MAIN:g}_seed{seed_for(SIGMA_MAIN, index)}",
                SIGMA_MAIN,
                seed_for(SIGMA_MAIN, index),
                record_trajectory,
            )
        )
    for sigma in SIGMA_SWEEP_VALUES:
        for index in range(SEEDS_SWEEP):
            payloads.append(
                ou_payload(
                    f"ou_s{sigma:g}_seed{seed_for(sigma, index)}",
                    sigma,
                    seed_for(sigma, index),
                    False,
                )
            )
    return payloads


def evaluate_case(payload: dict) -> dict:
    """在已验证的粗网格上完成一次全程干燥模拟（子进程入口，参数需可序列化）。"""
    source = {
        "source_times_s": payload["source_times_s"],
        "source_temperature_c": payload["source_temperature_c"],
        "source_moisture_kg_kg": payload["source_moisture_kg_kg"],
    }
    if payload["kind"] == "constant":
        boundary = q3.BoundaryProgram(
            plateau_temperature_c=payload["plateau_temperature_c"],
            plateau_moisture_kg_kg=payload["plateau_moisture_kg_kg"],
            **source,
        )
    else:
        boundary = MCBoundary(
            ou_times_s=payload["ou_times_s"],
            ou_temperature_c=payload["ou_temperature_c"],
            ou_moisture_kg_kg=payload["ou_moisture_kg_kg"],
            **source,
        )
    result = q3.simulate_until_dry(
        boundary,
        nominal_radial_step_cm=0.025,
        early_time_step_s=2.0,
        late_time_step_s=60.0,
        final_time_step_s=2.0,
    )
    trajectory = None
    if payload["record_trajectory"]:
        center_index = int(np.where(np.isclose(q3.OUTPUT_RADII_CM, 0.0))[0][0])
        surface_index = int(np.where(np.isclose(q3.OUTPUT_RADII_CM, 2.0))[0][0])
        trajectory = {
            "times_s": result.times_s,
            "center_moisture_kg_kg": result.moisture_kg_kg[:, center_index],
            "surface_moisture_kg_kg": result.moisture_kg_kg[:, surface_index],
        }
    return {
        "case_key": payload["case_key"],
        "kind": payload["kind"],
        "sigma_multiplier": payload["sigma_multiplier"],
        "seed": payload["seed"],
        "drying_time_coarse_s": result.drying_time_s,
        "drying_time_coarse_h": result.drying_time_s / 3600.0,
        "controlling_radius_cm": result.controlling_radius_cm,
        "maximum_picard_iterations": result.maximum_picard_iterations,
        "trajectory": trajectory,
    }


def run_monte_carlo(
    payloads: list[dict],
    result_dir: Path,
    workers: int,
    reuse: bool,
) -> pd.DataFrame:
    checkpoint_path = result_dir / "ou_robustness_checkpoint.csv"
    trajectory_path = result_dir / "trajectories_sigma1.npz"
    finished_keys: set[str] = set()
    frame_rows: list[dict] = []
    trajectory_store: dict[str, np.ndarray] = {}

    if reuse and checkpoint_path.exists():
        existing = pd.read_csv(checkpoint_path)
        finished_keys = set(existing["case_key"].astype(str))
        frame_rows = [
            row for row in existing.to_dict(orient="records") if "case_key" in row
        ]
    if reuse and trajectory_path.exists():
        with np.load(trajectory_path, allow_pickle=False) as archive:
            trajectory_store = {key: archive[key] for key in archive.files}

    pending = [payload for payload in payloads if payload["case_key"] not in finished_keys]
    total = len(payloads)
    print(f"{total - len(pending)}/{total} cases reused; {len(pending)} to run", flush=True)

    def store(result: dict) -> None:
        frame_rows.append({key: value for key, value in result.items() if key != "trajectory"})
        pd.DataFrame(frame_rows).sort_values("case_key").to_csv(
            checkpoint_path, index=False, encoding="utf-8-sig"
        )
        if result["trajectory"] is not None:
            prefix = (
                "baseline"
                if result["kind"] == "constant"
                else f"seed{result['seed']}"
            )
            trajectory_store[f"{prefix}_times"] = result["trajectory"]["times_s"]
            trajectory_store[f"{prefix}_m0"] = result["trajectory"]["center_moisture_kg_kg"]
            trajectory_store[f"{prefix}_m20"] = result["trajectory"]["surface_moisture_kg_kg"]
            np.savez(trajectory_path, **trajectory_store)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(evaluate_case, payload): payload["case_key"] for payload in pending}
        completed_count = total - len(pending)
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            store(result)
            print(
                f"[{completed_count + completed:03d}/{total:03d}] {result['case_key']} "
                f"-> {result['drying_time_coarse_h']:.4f} h",
                flush=True,
            )
    return pd.DataFrame(frame_rows).sort_values("case_key").reset_index(drop=True)


def add_correction(frame: pd.DataFrame, fine_baseline_h: float) -> tuple[pd.DataFrame, float]:
    baseline_rows = frame.loc[frame["case_key"] == "baseline_mean_3_4h"]
    if len(baseline_rows) != 1:
        raise ValueError("Exactly one constant-mean baseline case is required.")
    coarse_baseline_h = float(baseline_rows["drying_time_coarse_h"].iloc[0])
    correction_h = fine_baseline_h - coarse_baseline_h
    frame = frame.copy()
    frame["drying_time_corrected_h"] = frame["drying_time_coarse_h"] + correction_h
    frame["delta_vs_baseline_min"] = 60.0 * (
        frame["drying_time_corrected_h"] - fine_baseline_h
    )
    return frame, correction_h


def sigma_stats(frame: pd.DataFrame, sigma: float) -> dict:
    subset = frame.loc[
        np.isclose(frame["sigma_multiplier"], sigma)
        & (frame["kind"] == "ou")
    ]
    values = subset["drying_time_corrected_h"].to_numpy(dtype=float)
    deltas = subset["delta_vs_baseline_min"].to_numpy(dtype=float)
    if values.size == 0:
        raise ValueError(f"No Monte Carlo cases for sigma={sigma:g}.")
    return {
        "sigma_multiplier": sigma,
        "case_count": int(values.size),
        "mean_h": float(np.mean(values)),
        "standard_deviation_h": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "minimum_h": float(np.min(values)),
        "maximum_h": float(np.max(values)),
        "p05_h": float(np.quantile(values, 0.05)),
        "p95_h": float(np.quantile(values, 0.95)),
        "delta_mean_min": float(np.mean(deltas)),
        "delta_mean_pct": 100.0 * float(np.mean(deltas) / 60.0) / (
            float(np.mean(values)) - float(np.mean(deltas)) / 60.0
        ),
        "spread_p95_p05_min": 60.0 * (
            float(np.quantile(values, 0.95)) - float(np.quantile(values, 0.05))
        ),
        "range_min": 60.0 * (float(np.max(values)) - float(np.min(values))),
        "share_shorter_than_baseline": float(np.mean(deltas < 0.0)),
    }


def build_summary(
    frame: pd.DataFrame,
    fit: dict,
    environment: q2.EnvironmentData,
    fine_baseline_h: float,
    coarse_baseline_h: float,
    correction_h: float,
) -> dict:
    baseline = q3.build_boundary_program(environment)
    sigma1 = sigma_stats(frame, SIGMA_MAIN)
    deltas = frame.loc[
        np.isclose(frame["sigma_multiplier"], SIGMA_MAIN)
        & (frame["kind"] == "ou"),
        "delta_vs_baseline_min",
    ].to_numpy(dtype=float)
    within_one_minute = float(np.mean(np.abs(deltas) <= 1.0))
    s0 = frame.loc[frame["case_key"] == "ou_s0_limit"].iloc[0]
    huber = frame.loc[frame["case_key"] == "constant_huber_equilibrium"].iloc[0]
    return {
        "method": {
            "purpose": "以OU随机过程为依据的问题3模型稳健性分析",
            "reference": "Che Taib & Darus (2025), doi:10.11113/matematika.v41.n1.1610",
            "reference_scope": (
                "借鉴均值回复与蒙特卡洛传播思想；因稳定段仅61个逐分钟观测，"
                "使用可辨识的一阶OU/AR(1)，不复刻文献中的Lévy驱动随机回复速度。"
            ),
            "approach_one": "4 h后温度、水分浓度固定为附件1 3--4 h时间加权均值（恒值做法）",
            "approach_two": "4 h后按3--4 h稳定段Huber稳健标定的精确离散OU/AR(1)过程生成联合随机边界（OU做法）",
            "common_model": "附件3水热耦合有限体积，粗网格0.025 cm，2/60/2 s，统一粗-细修正",
            "sigma_1_seeds": SEEDS_MAIN,
            "sigma_sweep_seeds_per_level": SEEDS_SWEEP,
            "sigma_sweep_values": SIGMA_SWEEP_VALUES,
            "seed_rule": f"seed = {DEFAULT_SEED} + 1000*round(10*sigma) + k",
        },
        "ou_calibration": {
            "fit_interval_s": [oum.FIT_START_TIME_S, oum.MEASURED_END_TIME_S],
            "temperature": asdict(fit["temperature_fit"]),
            "moisture": asdict(fit["moisture_fit"]),
            "robust_innovation_correlation": fit["innovation_correlation"],
        },
        "baseline_constant_mean": {
            "temperature_c": baseline.plateau_temperature_c,
            "moisture_kg_kg": baseline.plateau_moisture_kg_kg,
            "fine_drying_time_h": fine_baseline_h,
            "coarse_drying_time_h": coarse_baseline_h,
            "additive_correction_h": correction_h,
        },
        "calibration_difference_sigma0": {
            "ou_path_corrected_h": float(s0["drying_time_corrected_h"]),
            "ou_path_delta_min": float(s0["delta_vs_baseline_min"]),
            "huber_constant_corrected_h": float(huber["drying_time_corrected_h"]),
            "huber_constant_delta_min": float(huber["delta_vs_baseline_min"]),
            "interpretation": (
                "sigma=0为OU的确定性极限；两行分别为无噪声OU路径与恒定Huber"
                "均衡边界，共同衡量均值口径（时间加权均值 vs Huber均衡）的差异。"
            ),
        },
        "monte_carlo_sigma1": sigma1,
        "sigma1_additional": {
            "maximum_absolute_delta_min": float(np.max(np.abs(deltas))),
            "share_within_one_minute": within_one_minute,
            "share_shorter_than_baseline": float(np.mean(deltas < 0.0)),
            "share_longer_than_baseline": float(np.mean(deltas > 0.0)),
        },
        "sigma_sweep": [sigma_stats(frame, sigma) for sigma in [0.5, SIGMA_MAIN, 1.5, 2.0]],
    }


def write_sample_boundary_csv(
    path: Path,
    fit: dict,
    seeds: list[int],
) -> None:
    """保存12条sigma=1样本OU路径，供边界图与正文复现使用。"""
    times = np.arange(
        oum.MEASURED_END_TIME_S,
        oum.OU_END_TIME_S + 0.5 * oum.OU_STEP_S,
        oum.OU_STEP_S,
        dtype=float,
    )
    columns = ["time_s", "time_h"]
    rows = [times, times / 3600.0]
    for seed in seeds:
        _, temperature, moisture = oum.simulate_joint_ou_path(
            fit["temperature_fit"],
            fit["moisture_fit"],
            fit["start_temperature_c"],
            fit["start_moisture_kg_kg"],
            fit["innovation_correlation"],
            seed,
            SIGMA_MAIN,
        )
        columns += [f"temperature_{seed}", f"moisture_{seed}"]
        rows += [temperature, moisture]
    frame = pd.DataFrame({column: rows[index] for index, column in enumerate(columns)})
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.10f")


def plot_boundary_paths(
    figure_dir: Path,
    environment: q2.EnvironmentData,
    fit: dict,
    sample_path: Path,
) -> None:
    paths = pd.read_csv(sample_path)
    temperature_fit = fit["temperature_fit"]
    moisture_fit = fit["moisture_fit"]
    seeds = [
        int(column.split("_")[-1])
        for column in paths.columns
        if column.startswith("temperature_")
    ]
    fig, axes = plt.subplots(
        1, 3, figsize=(7.4, 3.05), gridspec_kw={"width_ratios": [1, 1, 0.95]}
    )

    mask = (
        (environment.source_times_s >= oum.FIT_START_TIME_S)
        & (environment.source_times_s <= oum.MEASURED_END_TIME_S)
    )
    measured_times = environment.source_times_s[mask] / 3600.0
    base = q3.build_boundary_program(environment)

    temperature_paths = paths[[f"temperature_{seed}" for seed in seeds]].to_numpy().T
    moisture_paths = paths[[f"moisture_{seed}" for seed in seeds]].to_numpy().T
    path_time_h = paths["time_h"].to_numpy(dtype=float)
    for axis, values in zip(axes[:2], (temperature_paths, moisture_paths)):
        axis.fill_between(
            path_time_h,
            np.quantile(values, 0.05, axis=0),
            np.quantile(values, 0.95, axis=0),
            color=PALETTE["teal"], alpha=0.22, lw=0,
            label="12条OU路径的5%--95%带" if axis is axes[0] else None,
        )
        axis.plot(
            path_time_h, values[0], color=PALETTE["teal"], lw=0.45,
            alpha=0.75, zorder=2,
            label="代表性OU路径" if axis is axes[0] else None,
        )
    axes[0].scatter(
        measured_times, environment.source_temperature_c[mask],
        color=PALETTE["gray"], s=5, lw=0, zorder=3, label=LABEL_MEASURED,
    )
    axes[1].scatter(
        measured_times, environment.source_moisture_kg_kg[mask],
        color=PALETTE["gray"], s=5, lw=0, zorder=3,
    )
    axes[0].axhline(
        temperature_fit.equilibrium_mean, color=PALETTE["teal"],
        lw=1.6, zorder=4, label="做法二：OU均衡",
    )
    axes[0].axhline(
        base.plateau_temperature_c, color=PALETTE["blue"],
        ls="--", lw=1.4, zorder=4, label="做法一：时间加权均值",
    )
    axes[1].axhline(moisture_fit.equilibrium_mean, color=PALETTE["teal"], lw=1.6, zorder=4)
    axes[1].axhline(base.plateau_moisture_kg_kg, color=PALETTE["blue"], ls="--", lw=1.4, zorder=4)

    for axis in axes[:2]:
        # 4--12 h局部窗口足以展示逐分钟随机结构；全路径仍保存在结果CSV中。
        axis.set_xlim(3.0, 12.0)
        axis.grid(axis="y", color="#D8D8D8", lw=0.6, alpha=0.65)
        axis.set_xlabel("时间 / h")
    axes[0].set_ylim(49.15, 51.1)
    axes[1].set_ylim(0.04925, 0.05075)
    axes[0].set_ylabel("烘房温度 / °C")
    axes[1].set_ylabel("烘房水分浓度 / (kg/kg)")
    axes[0].set_title("OU温度边界（4--12 h）", fontsize=9.5)
    axes[1].set_title("OU水分边界（4--12 h）", fontsize=9.5)
    axes[0].text(
        0.02, 0.98,
        "均衡−时间均值\n"
        f"= {temperature_fit.equilibrium_mean - base.plateau_temperature_c:+.4f} °C",
        transform=axes[0].transAxes, va="top", fontsize=7.5, color="#404040",
    )
    axes[1].text(
        0.02, 0.98,
        "均衡−时间均值\n"
        f"= {moisture_fit.equilibrium_mean - base.plateau_moisture_kg_kg:+.2e} kg/kg",
        transform=axes[1].transAxes, va="top", fontsize=7.5, color="#404040",
    )

    temperature_scores = np.clip(
        fit["temperature_residuals"] / temperature_fit.innovation_scale,
        -oum.HUBER_C,
        oum.HUBER_C,
    )
    moisture_scores = np.clip(
        fit["moisture_residuals"] / moisture_fit.innovation_scale,
        -oum.HUBER_C,
        oum.HUBER_C,
    )
    axes[2].scatter(
        temperature_scores, moisture_scores,
        color=PALETTE["teal"], s=12, alpha=0.6, lw=0, zorder=3,
    )
    axes[2].set_xlabel("温度创新（截尾标准化）")
    axes[2].set_ylabel("水分创新（截尾标准化）")
    axes[2].set_title("创新同期相关", fontsize=9.5)
    axes[2].grid(color="#D8D8D8", lw=0.6, alpha=0.65)
    axes[2].text(
        0.03, 0.96,
        rf"$\phi_T$ = {temperature_fit.discrete_phi:.2g}（白噪声）" "\n"
        rf"$\sigma_T$ = {temperature_fit.innovation_scale:.3f} °C" "\n"
        rf"$\phi_C$ = {moisture_fit.discrete_phi:.3f}" "\n"
        rf"$\sigma_C$ = {moisture_fit.innovation_scale:.2e} kg/kg" "\n"
        rf"$\rho$ = {fit['innovation_correlation']:+.3f}",
        transform=axes[2].transAxes, va="top", fontsize=7.5, color="#404040",
    )

    for label, axis in zip("abc", axes):
        axis.text(-0.20, 1.06, label, transform=axis.transAxes, fontweight="bold", fontsize=10)
    fig.legend(
        loc="lower center", ncol=3, bbox_to_anchor=(0.42, 0.0),
        handlelength=1.6, columnspacing=1.0,
    )
    fig.subplots_adjust(wspace=0.55, bottom=0.27, top=0.87)
    save_png(fig, figure_dir / "q3_ou_boundary_paths.png")


def plot_drying_time_distribution(
    frame: pd.DataFrame,
    fine_baseline_h: float,
    figure_dir: Path,
    summary: dict,
) -> None:
    sigma1 = summary["monte_carlo_sigma1"]
    deltas = frame.loc[
        np.isclose(frame["sigma_multiplier"], SIGMA_MAIN)
        & (frame["kind"] == "ou"),
        "delta_vs_baseline_min",
    ].to_numpy(dtype=float)
    s0 = frame.loc[frame["case_key"] == "ou_s0_limit"].iloc[0]

    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    maximum = max(float(np.max(np.abs(deltas))), 1.5)
    bins = np.linspace(-maximum, maximum, 15)
    ax.hist(
        deltas, bins=bins, color=PALETTE["blue_2"], alpha=0.65,
        edgecolor=PALETTE["blue"], lw=0.8, zorder=2,
    )
    p05 = float(np.quantile(deltas, 0.05))
    p95 = float(np.quantile(deltas, 0.95))
    ax.axvspan(p05, p95, color=PALETTE["teal"], alpha=0.16, zorder=1)
    ax.axvline(0.0, color=PALETTE["red"], ls="--", lw=1.3, label="做法一基准（恒值均值）")
    ax.axvline(np.mean(deltas), color=PALETTE["blue"], lw=1.2, label="OU样本均值")
    ax.axvline(p05, color=PALETTE["gray"], ls=":", lw=1.0)
    ax.axvline(p95, color=PALETTE["gray"], ls=":", lw=1.0, label="5%/95%分位")
    ax.axvline(
        float(s0["delta_vs_baseline_min"]), color=PALETTE["violet"],
        ls="-.", lw=1.3,
        label=f"OU确定性极限 (σ=0, {s0['delta_vs_baseline_min']:+.1f} min)",
    )
    ax.set_xlabel("干燥时间相对做法一基准的变化 / min")
    ax.set_ylabel("样本数")
    ax.set_title(f"σ={SIGMA_MAIN:g}（标定噪声水平）下60条OU路径的干燥时间分布")
    ax.text(
        0.03, 0.97,
        f"均值 {np.mean(deltas):+.2f} min，标准差 {np.std(deltas, ddof=1):.2f} min\n"
        f"5%--95%分位 [{p05:+.1f}, {p95:+.1f}] min，极差 {np.max(deltas) - np.min(deltas):.1f} min\n"
        f"|Δ|≤1 min 占比 {summary['sigma1_additional']['share_within_one_minute'] * 100:.0f}%，"
        f"最大 |Δ| = {summary['sigma1_additional']['maximum_absolute_delta_min']:.1f} min",
        transform=ax.transAxes, va="top", fontsize=8, color="#404040",
    )
    ax.grid(axis="y", color="#D8D8D8", lw=0.6, alpha=0.65)
    fig.legend(
        loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.05),
        handlelength=1.6, columnspacing=1.2,
    )
    fig.subplots_adjust(bottom=0.20)
    save_png(fig, figure_dir / "q3_ou_drying_time_distribution.png")


def load_trajectory_bundle(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def align_center_trajectories(path: Path) -> dict[str, np.ndarray | float | int]:
    """Align every sigma=1 centre trajectory on the shared one-minute interval."""
    bundle = load_trajectory_bundle(path)
    seeds = sorted(
        int(key.split("_")[0].replace("seed", ""))
        for key in bundle
        if key.endswith("_m0") and key.startswith("seed")
    )
    if not seeds:
        raise ValueError("No sigma=1 OU centre trajectories were found.")
    terminal_times_s = [float(bundle["baseline_times"][-1])]
    terminal_times_s.extend(float(bundle[f"seed{seed}_times"][-1]) for seed in seeds)
    common_end_s = 60.0 * math.floor(min(terminal_times_s) / 60.0)
    common_times_s = np.arange(60.0, common_end_s + 0.5 * 60.0, 60.0)
    baseline = interp_monotone(
        common_times_s, bundle["baseline_times"], bundle["baseline_m0"]
    )
    stack = np.vstack(
        [
            interp_monotone(
                common_times_s,
                bundle[f"seed{seed}_times"],
                bundle[f"seed{seed}_m0"],
            )
            for seed in seeds
        ]
    )
    deviation = stack - baseline[None, :]
    return {
        "time_s": common_times_s,
        "time_h": common_times_s / 3600.0,
        "baseline": baseline,
        "stack": stack,
        "p05": np.quantile(stack, 0.05, axis=0),
        "median": np.quantile(stack, 0.50, axis=0),
        "p95": np.quantile(stack, 0.95, axis=0),
        "deviation": deviation,
        "deviation_p05": np.quantile(deviation, 0.05, axis=0),
        "deviation_median": np.quantile(deviation, 0.50, axis=0),
        "deviation_p95": np.quantile(deviation, 0.95, axis=0),
        "maximum_absolute_deviation": float(np.max(np.abs(deviation))),
        "path_count": len(seeds),
    }


def plot_moisture_band(
    aligned: dict[str, np.ndarray | float | int],
    fine_baseline_h: float,
    figure_dir: Path,
) -> None:
    common_times_h = np.asarray(aligned["time_h"], dtype=float)
    baseline_on_grid = np.asarray(aligned["baseline"], dtype=float)
    stack = np.asarray(aligned["stack"], dtype=float)
    band_low = np.asarray(aligned["p05"], dtype=float)
    band_high = np.asarray(aligned["p95"], dtype=float)
    median = np.asarray(aligned["median"], dtype=float)
    deviation = 1.0e4 * np.asarray(aligned["deviation"], dtype=float)
    sample_seed_indices = np.linspace(0, stack.shape[0] - 1, 5).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.05))

    axes[0].fill_between(
        common_times_h, band_low, band_high,
        color=PALETTE["teal"], alpha=0.22, lw=0,
        label="OU路径 5%--95% 带",
    )
    axes[0].plot(
        common_times_h, median, color=PALETTE["teal"], lw=1.3, label="OU路径中位数",
    )
    for index in sample_seed_indices:
        axes[0].plot(
            common_times_h, stack[index, :],
            color=PALETTE["teal"], lw=0.6, alpha=0.5,
        )
    axes[0].plot(
        common_times_h, baseline_on_grid, color=PALETTE["blue"], lw=1.6,
        label="做法一基准（恒值均值）",
    )
    axes[0].axhline(
        0.15, color=PALETTE["red"], ls=":", lw=1.1,
        label=LABEL_DRYING_THRESHOLD,
    )
    axes[0].axvline(fine_baseline_h, color=PALETTE["gray"], ls="--", lw=1.0)
    axes[0].set(
        xlabel="时间 / h", ylabel=r"中心 ($r=0$ cm) 水分浓度 / (kg kg$^{-1}$)",
        title="中心水分全程演化对比",
    )
    axes[0].set_xlim(0.0, common_times_h[-1])
    axes[0].legend(loc="upper right", handlelength=1.5)

    axes[1].fill_between(
        common_times_h,
        np.percentile(deviation, 5.0, axis=0),
        np.percentile(deviation, 95.0, axis=0),
        color=PALETTE["teal"], alpha=0.22, lw=0, label="OU路径 5%--95% 带",
    )
    for index in sample_seed_indices:
        axes[1].plot(
            common_times_h, deviation[index, :],
            color=PALETTE["teal"], lw=0.6, alpha=0.5,
        )
    axes[1].axhline(0.0, color=PALETTE["blue"], lw=1.3, label="做法一基准")
    axes[1].axvline(fine_baseline_h, color=PALETTE["gray"], ls="--", lw=1.0)
    axes[1].set(
        xlabel="时间 / h", ylabel=r"中心水分偏差 / ($10^{-4}$ kg kg$^{-1}$)",
        title="OU做法相对恒值做法的中心水分偏差",
    )
    axes[1].set_xlim(4.0, common_times_h[-1])
    maximum_deviation = float(np.max(np.abs(deviation)))
    axes[1].text(
        0.03, 0.96,
        f"最大 |偏差| = {maximum_deviation:.2f} × "
        r"$10^{-4}$ kg kg$^{-1}$" "\n"
        "最大偏差出现在4 h后过渡阶段\n"
        "随后衰减；末期体现为阈值时刻平移",
        transform=axes[1].transAxes, va="top", fontsize=8, color="#404040",
    )
    axes[1].legend(loc="lower left", handlelength=1.5)

    inset = inset_axes(
        axes[0], width="46%", height="46%", loc="lower left",
        borderpad=1.0,
    )
    zoom_low = fine_baseline_h - 1.6
    zoom_high = fine_baseline_h + 0.2
    inset.fill_between(
        common_times_h, band_low, band_high,
        color=PALETTE["teal"], alpha=0.22, lw=0,
    )
    inset.plot(common_times_h, baseline_on_grid, color=PALETTE["blue"], lw=1.4)
    inset.axhline(0.15, color=PALETTE["red"], ls=":", lw=1.0)
    inset.set_xlim(zoom_low, zoom_high)
    zoom_mask = (common_times_h >= zoom_low) & (common_times_h <= zoom_high)
    inset.set_ylim(
        float(np.nanmin(np.r_[band_low[zoom_mask], band_high[zoom_mask], 0.15])) - 0.0002,
        float(np.nanmax(np.r_[band_low[zoom_mask], band_high[zoom_mask], 0.15])) + 0.0002,
    )
    inset.set_xticks([zoom_low + 0.3, zoom_high - 0.3])
    inset.set_yticks([0.15])
    inset.tick_params(labelsize=7.5)
    inset.set_xticklabels([f"{zoom_low + 0.3:.1f}", f"{zoom_high - 0.3:.1f}"], fontsize=7.5)
    inset.set_yticklabels(["0.15"], fontsize=7.5)
    inset.text(
        0.02, 0.92, "末期放大", transform=inset.transAxes,
        fontsize=7.5, color="#606060", va="top",
    )

    for label, axis in zip("ab", axes):
        axis.grid(axis="y", color="#D8D8D8", lw=0.6, alpha=0.65)
        axis.text(-0.16, 1.05, label, transform=axis.transAxes, fontweight="bold", fontsize=10)
    fig.subplots_adjust(wspace=0.42)
    save_png(fig, figure_dir / "q3_ou_moisture_band.png")


def plot_sigma_sweep(
    frame: pd.DataFrame,
    fine_baseline_h: float,
    figure_dir: Path,
    summary: dict,
) -> None:
    sweep = summary["sigma_sweep"]
    sigmas = np.asarray([row["sigma_multiplier"] for row in sweep])
    means = np.asarray([row["mean_h"] for row in sweep])
    deviations = np.asarray([row["standard_deviation_h"] for row in sweep])
    spreads = np.asarray([row["spread_p95_p05_min"] for row in sweep])
    ranges = np.asarray([row["range_min"] for row in sweep])
    s0 = frame.loc[frame["case_key"] == "ou_s0_limit"].iloc[0]
    huber = frame.loc[frame["case_key"] == "constant_huber_equilibrium"].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.95))

    axes[0].axhline(fine_baseline_h, color=PALETTE["red"], ls="--", lw=1.3,
                    label=f"做法一基准 {fine_baseline_h:.3f} h")
    axes[0].errorbar(
        sigmas, means, yerr=deviations,
        color=PALETTE["blue"], marker="o", ms=4.5, lw=1.5,
        capsize=3, capthick=1.0, label="OU做法 均值±标准差",
    )
    axes[0].plot(
        [-0.06], [float(s0["drying_time_corrected_h"])],
        marker="D", color=PALETTE["violet"], ms=6, lw=0, clip_on=False,
        label="OU确定性极限 (σ=0)",
    )
    axes[0].plot(
        [0.06], [float(huber["drying_time_corrected_h"])],
        marker="s", color=PALETTE["violet"], ms=5.5, lw=0, clip_on=False,
        label="恒定Huber均衡边界",
    )
    axes[0].set_xlim(-0.35, 2.35)
    axes[0].set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ylow = min(
        float(np.min(means - deviations)),
        fine_baseline_h,
        float(s0["drying_time_corrected_h"]),
        float(huber["drying_time_corrected_h"]),
    ) - 0.03
    yhigh = max(
        float(np.max(means + deviations)),
        fine_baseline_h,
        float(s0["drying_time_corrected_h"]),
        float(huber["drying_time_corrected_h"]),
    ) + 0.03
    axes[0].set_ylim(ylow, yhigh)
    axes[0].set(
        xlabel=r"OU创新标准差倍率 $\sigma_{\mathrm{mult}}$",
        ylabel="干燥时间 / h",
        title="干燥时间对噪声水平的响应",
    )
    axes[0].legend(loc="upper right", handlelength=1.5, fontsize=7.5)

    axes[1].bar(
        sigmas, ranges, width=0.24, color=PALETTE["gray"], alpha=0.45,
        edgecolor="none", label=LABEL_RANGE,
    )
    axes[1].bar(
        sigmas, spreads, width=0.24, color=PALETTE["teal"], alpha=0.8,
        edgecolor="none", label="5%--95% 分位宽度",
    )
    for sigma, spread in zip(sigmas, spreads):
        axes[1].text(
            sigma, spread,
            f"{spread:.1f}", ha="center", va="bottom", fontsize=7.5, color="#202020",
        )
    axes[1].set_xlim(0.2, 2.3)
    axes[1].set_xticks([0.5, 1.0, 1.5, 2.0])
    axes[1].set_ylim(0.0, float(np.max(ranges)) * 1.3)
    axes[1].set(
        xlabel=r"OU创新标准差倍率 $\sigma_{\mathrm{mult}}$",
        ylabel="干燥时间离散度 / min",
        title="路径间随机差异随噪声水平放大",
    )
    axes[1].legend(loc="upper left", handlelength=1.5, fontsize=7.5)

    for label, axis in zip("ab", axes):
        axis.grid(axis="y", color="#D8D8D8", lw=0.6, alpha=0.65)
        axis.text(-0.18, 1.05, label, transform=axis.transAxes, fontweight="bold", fontsize=10)
    fig.subplots_adjust(wspace=0.42)
    save_png(fig, figure_dir / "q3_ou_sigma_sweep.png")


def write_comparison_tables(
    result_dir: Path,
    summary: dict,
    aligned: dict[str, np.ndarray | float | int],
) -> None:
    """Write compact paper-facing tables and source data for every plotted aggregate."""
    baseline = summary["baseline_constant_mean"]
    calibration = summary["calibration_difference_sigma0"]
    model_rows = [
        {
            "model": "原模型：3--4 h时间加权均值恒值边界",
            "sigma_multiplier": 0.0,
            "path_count": 1,
            "mean_drying_time_h": baseline["fine_drying_time_h"],
            "standard_deviation_min": 0.0,
            "p05_drying_time_h": baseline["fine_drying_time_h"],
            "p95_drying_time_h": baseline["fine_drying_time_h"],
            "delta_mean_vs_original_min": 0.0,
            "delta_mean_vs_original_pct": 0.0,
            "p05_p95_width_min": 0.0,
            "range_min": 0.0,
        },
    ]
    if "representative_ou_run" in summary:
        representative = summary["representative_ou_run"]
        model_rows.append(
            {
                "model": "OU随机边界：默认种子细网格代表路径",
                "sigma_multiplier": representative["sigma_multiplier"],
                "path_count": 1,
                "mean_drying_time_h": representative["drying_time_h"],
                "standard_deviation_min": np.nan,
                "p05_drying_time_h": np.nan,
                "p95_drying_time_h": np.nan,
                "delta_mean_vs_original_min": representative[
                    "delta_vs_original_min"
                ],
                "delta_mean_vs_original_pct": representative[
                    "delta_vs_original_pct"
                ],
                "p05_p95_width_min": np.nan,
                "range_min": np.nan,
            }
        )
    model_rows.append(
        {
            "model": "OU确定性极限：Huber均衡、无随机创新",
            "sigma_multiplier": 0.0,
            "path_count": 1,
            "mean_drying_time_h": calibration["ou_path_corrected_h"],
            "standard_deviation_min": 0.0,
            "p05_drying_time_h": calibration["ou_path_corrected_h"],
            "p95_drying_time_h": calibration["ou_path_corrected_h"],
            "delta_mean_vs_original_min": calibration["ou_path_delta_min"],
            "delta_mean_vs_original_pct": 100.0
            * (calibration["ou_path_corrected_h"] - baseline["fine_drying_time_h"])
            / baseline["fine_drying_time_h"],
            "p05_p95_width_min": 0.0,
            "range_min": 0.0,
        }
    )
    for row in summary["sigma_sweep"]:
        model_rows.append(
            {
                "model": f"OU随机边界：sigma_multiplier={row['sigma_multiplier']:g}",
                "sigma_multiplier": row["sigma_multiplier"],
                "path_count": row["case_count"],
                "mean_drying_time_h": row["mean_h"],
                "standard_deviation_min": 60.0 * row["standard_deviation_h"],
                "p05_drying_time_h": row["p05_h"],
                "p95_drying_time_h": row["p95_h"],
                "delta_mean_vs_original_min": row["delta_mean_min"],
                "delta_mean_vs_original_pct": row["delta_mean_pct"],
                "p05_p95_width_min": row["spread_p95_p05_min"],
                "range_min": row["range_min"],
            }
        )
    pd.DataFrame(model_rows).to_csv(
        result_dir / "original_vs_ou_model_comparison.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.8g",
    )

    parameter_rows = []
    for key, display_name in (
        ("temperature", "烘房温度"),
        ("moisture", "烘房水分浓度"),
    ):
        fit = summary["ou_calibration"][key]
        original_mean = (
            baseline["temperature_c"] if key == "temperature"
            else baseline["moisture_kg_kg"]
        )
        parameter_rows.append(
            {
                "variable": display_name,
                "unit": fit["unit"],
                "fit_sample_count": fit["sample_count"],
                "original_time_weighted_mean": original_mean,
                "ou_huber_equilibrium": fit["equilibrium_mean"],
                "equilibrium_minus_original": fit["equilibrium_mean"] - original_mean,
                "discrete_phi_60s": fit["discrete_phi"],
                "kappa_per_s": fit["kappa_per_s"],
                "relaxation_time_s": fit["relaxation_time_s"],
                "half_life_s": fit["half_life_s"],
                "innovation_scale": fit["innovation_scale"],
                "stationary_scale": fit["stationary_scale"],
                "phi_at_lower_bound": fit["phi_at_lower_bound"],
            }
        )
    pd.DataFrame(parameter_rows).to_csv(
        result_dir / "ou_calibration_parameters.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.10g",
    )

    source_frame = pd.DataFrame(
        {
            "time_s": aligned["time_s"],
            "time_h": aligned["time_h"],
            "original_center_moisture_kg_kg": aligned["baseline"],
            "ou_center_moisture_p05_kg_kg": aligned["p05"],
            "ou_center_moisture_median_kg_kg": aligned["median"],
            "ou_center_moisture_p95_kg_kg": aligned["p95"],
            "ou_minus_original_p05_kg_kg": aligned["deviation_p05"],
            "ou_minus_original_median_kg_kg": aligned["deviation_median"],
            "ou_minus_original_p95_kg_kg": aligned["deviation_p95"],
        }
    )
    source_frame.to_csv(
        result_dir / "ou_moisture_band_source_data.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.10g",
    )


def write_analysis_md(
    path: Path,
    summary: dict,
    figure_names: dict[str, str],
) -> None:
    calibration = summary["calibration_difference_sigma0"]
    sigma1 = summary["monte_carlo_sigma1"]
    extra = summary["sigma1_additional"]
    baseline = summary["baseline_constant_mean"]
    ou_cal = summary["ou_calibration"]
    trajectory = summary["moisture_trajectory_sigma1"]
    representative = summary.get("representative_ou_run")
    representative_text = ""
    if representative is not None:
        representative_text = (
            f"\n默认种子（seed={representative['seed']}）的细网格OU代表路径给出"
            f" {representative['drying_time_h']:.4f} h，较原模型"
            f" {representative['delta_vs_original_min']:+.2f} min；单条路径只用于"
            "复现实例，总体判断以蒙特卡洛分布为准。\n"
        )
    sweep_rows = "\n".join(
        f"| {row['sigma_multiplier']:g} | {row['case_count']} | "
        f"{row['mean_h']:.4f} ± {row['standard_deviation_h']:.4f} | "
        f"{row['delta_mean_min']:+.2f} | {row['spread_p95_p05_min']:.1f} | "
        f"{row['range_min']:.1f} |"
        for row in summary["sigma_sweep"]
    )
    text = f"""# A题问题3：以OU随机过程为依据的稳健性分析

## 1 目的与两种做法

问题3在4 h后（附件1实测区间之外）对烘房边界作出假设，本分析以4 h后
边界假设的不确定性为切入点，比较两种做法：

- **做法一（恒值做法，原模型）**：4 h后温度、水分浓度固定为附件1 3--4 h的
  时间加权均值，即确定性恒值边界（[a3_drying_time_fvm.py](../../code/a3_drying_time_fvm.py)）。
- **做法二（OU做法）**：4 h后按3--4 h稳定段 Huber 稳健标定的精确离散
  OU/AR(1) 过程逐分钟生成联合随机边界（[a3_drying_time_fvm_huber_ou.py](../../code/a3_drying_time_fvm_huber_ou.py)）。

两种做法除4 h后的边界外完全一致：附件3水热耦合方程、药材物性、有限体积
数值格式均相同，因此两法的干燥时间差全部可归因于边界假设。

参考文献 Che Taib 与 Darus（2025，DOI: 10.11113/matematika.v41.n1.1610）以
OU过程描述温度偏离，并进一步用Lévy驱动过程刻画随机均值回复速度。本文借鉴其
“均值回复 + 蒙特卡洛传播”的建模思想；但3--4 h稳定段只有61个逐分钟观测，
不足以稳健辨识嵌套随机回复速度，故使用可辨识的一阶OU/AR(1)模型。这是针对本题
数据分辨率的降阶近似，不是对参考文献完整模型的复刻。

## 2 OU标定结果（3--4 h稳定段，Huber c=1.345）

| 变量 | 均衡 | φ | 创新σ | 平稳σ |
| --- | --- | --- | --- | --- |
| 温度 / °C | {ou_cal['temperature']['equilibrium_mean']:.4f} | {ou_cal['temperature']['discrete_phi']:.2g} | {ou_cal['temperature']['innovation_scale']:.3f} | {ou_cal['temperature']['stationary_scale']:.3f} |
| 水分 / (kg/kg) | {ou_cal['moisture']['equilibrium_mean']:.6f} | {ou_cal['moisture']['discrete_phi']:.3f} | {ou_cal['moisture']['innovation_scale']:.2e} | {ou_cal['moisture']['stationary_scale']:.2e} |

温度在1 min观测分辨率下已接近白噪声（φ_T = {ou_cal['temperature']['discrete_phi']:.2g}，
拟合值触及数值下界）；水分的弛豫时间为
{ou_cal['moisture']['relaxation_time_s']:.1f} s，亦短于1 min采样间隔。
因此两者在现有观测分辨率下都只表现出很弱的滞后记忆。两变量同期创新相关系数
ρ = {ou_cal['robust_innovation_correlation']:+.3f}。

## 3 蒙特卡洛设计

- 网格：已验证粗网格（0.025 cm，2/60/2 s），全部情形统一施加细网格基准的
  粗-细修正（{baseline['additive_correction_h']:+.4f} h）；
- σ_mult = 1.0（标定噪声水平）：{summary['sigma_sweep'][1]['case_count']} 条独立OU路径（主蒙特卡洛）；
- σ_mult ∈ {{0, 0.5, 1.5, 2.0}}：噪声水平扫描（σ=0 为OU的确定性极限）；
- 恒定Huber均衡边界单算一次，把"均值口径差异"（时间加权均值 vs Huber均衡）
  与"随机波动差异"分解开。

## 4 主要结果

基准（做法一）干燥时间：细网格 {baseline['fine_drying_time_h']:.4f} h
（粗网格 {baseline['coarse_drying_time_h']:.4f} h）。
{representative_text}

**σ=0 确定性极限**：OU无噪声路径比基准 {calibration['ou_path_delta_min']:+.2f} min；
恒定Huber均衡边界比基准 {calibration['huber_constant_delta_min']:+.2f} min。
即两种均值口径本身造成的差别不足 0.5 min。

**σ_mult = 1 主蒙特卡洛**（{sigma1['case_count']} 条路径）：

- 干燥时间（修正后）均值 {sigma1['mean_h']:.4f} h，标准差
  {sigma1['standard_deviation_h'] * 60:.1f} min；
- 相对做法一基准平均变化 {sigma1['delta_mean_min']:+.2f} min
  （{sigma1['delta_mean_pct']:+.3f}%），5%--95% 分位
  [{60 * (sigma1['p05_h'] - baseline['fine_drying_time_h']):+.2f},
  {60 * (sigma1['p95_h'] - baseline['fine_drying_time_h']):+.2f}] min；
- |Δ| ≤ 1 min 的比例 {extra['share_within_one_minute'] * 100:.0f}%，
  最大 |Δ| = {extra['maximum_absolute_delta_min']:.1f} min。

**噪声水平扫描**：

| σ_mult | 路径数 | 修正后干燥时间 / h | 相对基准平均Δ / min | 5%--95%宽度 / min | 极差 / min |
| --- | --- | --- | --- | --- | --- |
{sweep_rows}

## 5 结论

1. 两种做法对干燥时间的预测几乎一致：σ_mult = 1 时 60 条OU路径的平均干燥时间
   与恒值基准相差 {sigma1['delta_mean_min']:+.2f} min（{sigma1['delta_mean_pct']:+.3f}%），
   远小于粗-细网格修正本身（约 {abs(baseline['additive_correction_h']) * 60:.0f} min）。
2. 内部水分轨迹的最大差异出现在4 h后由实测边界转入外推边界的过渡阶段，
   随后逐步衰减；干燥末期的差异主要体现为阈值穿越时刻的小幅随机平移。在全部
   {trajectory['path_count']} 条 σ_mult=1 路径共同覆盖的逐分钟区间内，最大中心
   水分绝对偏差为 {trajectory['maximum_absolute_deviation_kg_kg']:.3e} kg/kg，
   全程形态与原模型一致。
3. 随机差异随噪声水平近似线性放大：σ_mult 从 0.5 增至 2.0 时，5%--95%
   分位宽度由 {summary['sigma_sweep'][0]['spread_p95_p05_min']:.2f} min 增至
   {summary['sigma_sweep'][-1]['spread_p95_p05_min']:.2f} min，但即便 σ_mult = 2
   （标定噪声的两倍），干燥时间相对基准的最大偏移仍不超过
   {summary['sigma_sweep'][-1]['range_min']:.1f} min。
4. 因此问题3以3--4 h时间加权均值作恒值边界的做法对4 h后的短时随机波动是
   稳健的：OU随机做法与恒值做法在干燥时间上的平均差别不足1 min（<0.05%）。
   但这一结论只覆盖“围绕稳定均值的小幅、短记忆波动”，不能外推到长期漂移、
   工况突变或参考文献中的随机回复速度机制。

## 6 结果表

- `original_vs_ou_model_comparison.csv`：原模型、OU确定性极限与各噪声倍率的效果对比；
- `ou_calibration_parameters.csv`：OU参数及其与原恒值边界的均值差；
- `ou_moisture_band_source_data.csv`：中心水分5%--95%带的逐分钟绘图源数据；
- `ou_robustness_checkpoint.csv`：全部蒙特卡洛路径的逐路径结果。

## 7 图清单

- `{figure_names['boundary']}`：OU随机边界路径、实测稳定段与两法均值口径；
- `{figure_names['distribution']}`：σ=1 下干燥时间分布；
- `{figure_names['band']}`：中心水分轨迹5%--95%带与两法偏差；
- `{figure_names['sweep']}`：干燥时间及其离散度对噪声水平的响应。
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    script_path = Path(__file__).resolve()
    default_repo = script_path.parents[1]
    parser.add_argument("--repo-root", type=Path, default=default_repo)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_root = args.output_root.resolve() if args.output_root else repo_root
    result_dir = output_root / "results" / "A_problem3_ou_robustness"
    figure_dir = output_root / "picture" / "A_problem3_ou_robustness"
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    environment = q2.load_and_preprocess_environment(q2.find_default_data_path(repo_root))
    fit = fit_ou(environment)
    payloads = build_payloads(environment, fit, record_trajectory=True)
    frame = run_monte_carlo(payloads, result_dir, args.workers, args.reuse)

    validation_path = repo_root / "results" / "A_problem3_drying_time" / "validation_summary.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    fine_baseline_h = float(validation["drying_time"]["production_h"])
    frame, correction_h = add_correction(frame, fine_baseline_h)
    coarse_baseline_h = fine_baseline_h - correction_h

    summary = build_summary(
        frame, fit, environment, fine_baseline_h, coarse_baseline_h, correction_h
    )
    representative_path = (
        repo_root
        / "results"
        / "A_problem3_drying_time_huber_ou"
        / "validation_summary.json"
    )
    if representative_path.exists():
        representative_validation = json.loads(
            representative_path.read_text(encoding="utf-8")
        )
        representative_h = float(
            representative_validation["drying_time"]["production_h"]
        )
        summary["representative_ou_run"] = {
            "seed": int(
                representative_validation["model_scope"].get(
                    "random_seed", DEFAULT_SEED
                )
            ),
            "sigma_multiplier": float(
                representative_validation["model_scope"].get(
                    "sigma_multiplier", SIGMA_MAIN
                )
            ),
            "drying_time_h": representative_h,
            "delta_vs_original_min": 60.0
            * (representative_h - fine_baseline_h),
            "delta_vs_original_pct": 100.0
            * (representative_h - fine_baseline_h)
            / fine_baseline_h,
            "role": "reproducible fine-grid example; population inference uses Monte Carlo",
        }
    aligned = align_center_trajectories(result_dir / "trajectories_sigma1.npz")
    summary["moisture_trajectory_sigma1"] = {
        "path_count": int(aligned["path_count"]),
        "common_interval_s": [
            float(np.asarray(aligned["time_s"])[0]),
            float(np.asarray(aligned["time_s"])[-1]),
        ],
        "maximum_absolute_deviation_kg_kg": float(
            aligned["maximum_absolute_deviation"]
        ),
        "alignment": "all paths linearly interpolated to their shared 60 s grid",
    }
    (result_dir / "robustness_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_comparison_tables(result_dir, summary, aligned)

    sample_path = result_dir / "ou_boundary_samples_sigma1.csv"
    write_sample_boundary_csv(
        sample_path,
        fit,
        [seed_for(SIGMA_MAIN, index) for index in range(SAMPLE_PATH_SEEDS)],
    )

    apply_style()
    plot_boundary_paths(figure_dir, environment, fit, sample_path)
    plot_drying_time_distribution(frame, fine_baseline_h, figure_dir, summary)
    plot_moisture_band(aligned, fine_baseline_h, figure_dir)
    plot_sigma_sweep(frame, fine_baseline_h, figure_dir, summary)
    write_analysis_md(
        result_dir / "ou_robustness_analysis.md",
        summary,
        {
            "boundary": "q3_ou_boundary_paths.png",
            "distribution": "q3_ou_drying_time_distribution.png",
            "band": "q3_ou_moisture_band.png",
            "sweep": "q3_ou_sigma_sweep.png",
        },
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Results: {result_dir}", flush=True)
    print(f"Figures: {figure_dir}", flush=True)


if __name__ == "__main__":
    main()
