"""Sensitivity analysis for the post-4 h boundary assumptions in Question 3.

The analysis perturbs the constant oven temperature and oven moisture used
after 4 h, and also changes the averaging-window length used to construct that
constant boundary.  The verified coarse grid is used for the parameter sweep;
all scenario times receive the same coarse-to-production baseline correction.
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

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['pdf.fonttype'] = 42

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

import a2_coupled_fvm as q2
import a3_drying_time_fvm as q3


TEMPERATURE_DELTAS_C = np.array([-4.0, -2.0, 0.0, 2.0, 4.0])
MOISTURE_FACTORS = np.array([0.70, 0.85, 1.00, 1.15, 1.30])
AVERAGING_WINDOWS_H = np.array([0.25, 0.50, 1.00, 1.50, 2.00])

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


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    group: str
    plateau_temperature_c: float
    plateau_moisture_kg_kg: float
    temperature_delta_c: float | None = None
    moisture_factor: float | None = None
    averaging_window_h: float | None = None


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
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def time_average(
    source_times_s: np.ndarray,
    source_values: np.ndarray,
    start_s: float,
    end_s: float,
) -> float:
    """Trapezoidal time average with interpolated interval endpoints."""
    if source_times_s.ndim != 1 or source_values.ndim != 1:
        raise ValueError("Time-average inputs must be one-dimensional.")
    if source_times_s.size != source_values.size:
        raise ValueError("Time and value arrays must have the same length.")
    if not np.all(np.diff(source_times_s) > 0.0):
        raise ValueError("Source times must be strictly increasing for interpolation.")
    if not source_times_s[0] <= start_s < end_s <= source_times_s[-1]:
        raise ValueError("The averaging interval is outside the source data.")
    inside = (source_times_s > start_s) & (source_times_s < end_s)
    times = np.concatenate(([start_s], source_times_s[inside], [end_s]))
    values = np.interp(times, source_times_s, source_values)
    return float(np.trapezoid(values, times) / (end_s - start_s))


def build_scenarios(environment: q2.EnvironmentData) -> tuple[list[Scenario], float, float]:
    baseline = q3.build_boundary_program(environment)
    base_t = baseline.plateau_temperature_c
    base_c = baseline.plateau_moisture_kg_kg
    scenarios: list[Scenario] = []

    for delta_t in TEMPERATURE_DELTAS_C:
        for moisture_factor in MOISTURE_FACTORS:
            scenarios.append(
                Scenario(
                    scenario_id=f"joint_T{delta_t:+.0f}_C{moisture_factor:.2f}",
                    group="joint",
                    plateau_temperature_c=base_t + float(delta_t),
                    plateau_moisture_kg_kg=base_c * float(moisture_factor),
                    temperature_delta_c=float(delta_t),
                    moisture_factor=float(moisture_factor),
                )
            )

    for window_h in AVERAGING_WINDOWS_H:
        end_s = q3.MEASURED_END_TIME_S
        start_s = end_s - float(window_h) * 3600.0
        mean_t = time_average(
            environment.source_times_s,
            environment.source_temperature_c,
            start_s,
            end_s,
        )
        mean_c = time_average(
            environment.source_times_s,
            environment.source_moisture_kg_kg,
            start_s,
            end_s,
        )
        scenarios.append(
            Scenario(
                scenario_id=f"window_{window_h:.2f}h",
                group="window",
                plateau_temperature_c=mean_t,
                plateau_moisture_kg_kg=mean_c,
                averaging_window_h=float(window_h),
            )
        )
    return scenarios, base_t, base_c


def case_key(temperature_c: float, moisture: float) -> str:
    return f"{temperature_c:.12f}|{moisture:.12f}"


def evaluate_case(payload: tuple[str, float, float, np.ndarray, np.ndarray, np.ndarray]) -> dict:
    key, plateau_t, plateau_c, source_times, source_t, source_c = payload
    boundary = q3.BoundaryProgram(
        source_times_s=source_times,
        source_temperature_c=source_t,
        source_moisture_kg_kg=source_c,
        plateau_temperature_c=plateau_t,
        plateau_moisture_kg_kg=plateau_c,
    )
    result = q3.simulate_until_dry(
        boundary,
        nominal_radial_step_cm=0.025,
        early_time_step_s=2.0,
        late_time_step_s=60.0,
        final_time_step_s=2.0,
    )
    return {
        "case_key": key,
        "drying_time_coarse_s": result.drying_time_s,
        "drying_time_coarse_h": result.drying_time_s / 3600.0,
        "controlling_radius_cm": result.controlling_radius_cm,
        "maximum_picard_iterations": result.maximum_picard_iterations,
    }


def run_parameter_sweep(
    scenarios: list[Scenario],
    environment: q2.EnvironmentData,
    output_dir: Path,
    workers: int,
) -> pd.DataFrame:
    unique_cases: dict[str, tuple[float, float]] = {}
    for scenario in scenarios:
        key = case_key(scenario.plateau_temperature_c, scenario.plateau_moisture_kg_kg)
        unique_cases[key] = (
            scenario.plateau_temperature_c,
            scenario.plateau_moisture_kg_kg,
        )

    payloads = [
        (
            key,
            values[0],
            values[1],
            environment.source_times_s,
            environment.source_temperature_c,
            environment.source_moisture_kg_kg,
        )
        for key, values in unique_cases.items()
    ]
    results: list[dict] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(evaluate_case, payload): payload[0] for payload in payloads}
        total = len(futures)
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            pd.DataFrame(results).sort_values("case_key").to_csv(
                output_dir / "sensitivity_case_checkpoint.csv", index=False, encoding="utf-8-sig"
            )
            print(
                f"[{completed:02d}/{total:02d}] {result['case_key']} -> "
                f"{result['drying_time_coarse_h']:.4f} h",
                flush=True,
            )

    case_frame = pd.DataFrame(results)
    rows = []
    lookup = case_frame.set_index("case_key")
    for scenario in scenarios:
        row = asdict(scenario)
        key = case_key(scenario.plateau_temperature_c, scenario.plateau_moisture_kg_kg)
        row.update(lookup.loc[key].to_dict())
        rows.append(row)
    return pd.DataFrame(rows)


def add_corrected_times(
    frame: pd.DataFrame,
    base_t: float,
    base_c: float,
    fine_baseline_h: float,
) -> tuple[pd.DataFrame, float, float]:
    is_baseline = (
        (frame["group"] == "joint")
        & np.isclose(frame["plateau_temperature_c"], base_t)
        & np.isclose(frame["plateau_moisture_kg_kg"], base_c)
    )
    if is_baseline.sum() != 1:
        raise ValueError("Exactly one joint-grid baseline is required.")
    coarse_baseline_h = float(frame.loc[is_baseline, "drying_time_coarse_h"].iloc[0])
    correction_h = fine_baseline_h - coarse_baseline_h
    frame = frame.copy()
    frame["drying_time_corrected_h"] = frame["drying_time_coarse_h"] + correction_h
    frame["change_from_baseline_h"] = frame["drying_time_corrected_h"] - fine_baseline_h
    frame["change_from_baseline_pct"] = (
        100.0 * frame["change_from_baseline_h"] / fine_baseline_h
    )
    return frame, coarse_baseline_h, correction_h


def joint_subset(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame["group"] == "joint"].copy()


def one_factor_subsets(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    joint = joint_subset(frame)
    by_t = joint.loc[np.isclose(joint["moisture_factor"], 1.0)].sort_values(
        "temperature_delta_c"
    )
    by_c = joint.loc[np.isclose(joint["temperature_delta_c"], 0.0)].sort_values(
        "moisture_factor"
    )
    return by_t, by_c


def local_sensitivities(
    frame: pd.DataFrame, base_t: float, base_c: float, baseline_h: float
) -> dict:
    by_t, by_c = one_factor_subsets(frame)
    t_minus = float(by_t.loc[np.isclose(by_t["temperature_delta_c"], -2.0), "drying_time_corrected_h"].iloc[0])
    t_plus = float(by_t.loc[np.isclose(by_t["temperature_delta_c"], 2.0), "drying_time_corrected_h"].iloc[0])
    c_minus = float(by_c.loc[np.isclose(by_c["moisture_factor"], 0.85), "drying_time_corrected_h"].iloc[0])
    c_plus = float(by_c.loc[np.isclose(by_c["moisture_factor"], 1.15), "drying_time_corrected_h"].iloc[0])
    dtd_temperature = (t_plus - t_minus) / 4.0
    dtd_moisture = (c_plus - c_minus) / (0.30 * base_c)
    return {
        "temperature_hours_per_C": dtd_temperature,
        "moisture_hours_per_0p01_kg_kg": dtd_moisture * 0.01,
        "temperature_normalized_sensitivity": dtd_temperature * base_t / baseline_h,
        "moisture_normalized_sensitivity": dtd_moisture * base_c / baseline_h,
    }


def plot_one_factor(
    frame: pd.DataFrame,
    base_t: float,
    base_c: float,
    baseline_h: float,
    sensitivities: dict,
    figure_dir: Path,
) -> None:
    by_t, by_c = one_factor_subsets(frame)
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.65), gridspec_kw={"width_ratios": [1, 1, 0.82]})

    axes[0].plot(
        by_t["plateau_temperature_c"],
        by_t["drying_time_corrected_h"],
        color=PALETTE["blue"],
        marker="o",
        lw=1.8,
    )
    axes[0].axvline(base_t, color=PALETTE["gray"], ls="--", lw=1.0)
    axes[0].axhline(baseline_h, color=PALETTE["red"], ls=":", lw=1.0)
    axes[0].set(
        xlabel="4 h后平均温度 / °C",
        ylabel="干燥时间 / h",
        title="平均温度单因素响应",
    )

    axes[1].plot(
        by_c["plateau_moisture_kg_kg"],
        by_c["drying_time_corrected_h"],
        color=PALETTE["teal"],
        marker="o",
        lw=1.8,
    )
    axes[1].axvline(base_c, color=PALETTE["gray"], ls="--", lw=1.0)
    axes[1].axhline(baseline_h, color=PALETTE["red"], ls=":", lw=1.0)
    axes[1].set(
        xlabel="4 h后平均水分 / (kg/kg)",
        ylabel="干燥时间 / h",
        title="平均水分单因素响应",
    )

    names = ["平均温度", "平均水分"]
    values = [
        sensitivities["temperature_normalized_sensitivity"],
        sensitivities["moisture_normalized_sensitivity"],
    ]
    colors = [PALETTE["blue"] if value < 0 else PALETTE["red"] for value in values]
    axes[2].barh(names, values, color=colors, height=0.52)
    axes[2].axvline(0, color="#444444", lw=0.8)
    for y, value in enumerate(values):
        if value < 0:
            axes[2].text(
                value / 2.0, y, f"{value:+.3f}", ha="center", va="center",
                fontsize=8, color="white", fontweight="bold"
            )
        else:
            axes[2].text(
                value + 0.05, y, f"{value:+.3f}", ha="left", va="center",
                fontsize=8, color="#202020"
            )
    axes[2].set(xlabel="归一化灵敏度", title="局部灵敏度比较")
    bound = max(abs(np.asarray(values))) * 1.35
    axes[2].set_xlim(-bound, bound)

    for label, ax in zip("abc", axes):
        ax.grid(axis="y", color="#D8D8D8", lw=0.6, alpha=0.65)
        ax.text(-0.18, 1.05, label, transform=ax.transAxes, fontweight="bold", fontsize=10)
    fig.subplots_adjust(wspace=0.46)
    save_png(fig, figure_dir / "q3_sensitivity_one_factor.png")


def plot_joint_surface(frame: pd.DataFrame, figure_dir: Path) -> None:
    joint = joint_subset(frame)
    delta = joint.pivot(
        index="moisture_factor", columns="temperature_delta_c", values="change_from_baseline_h"
    ).sort_index().sort_index(axis=1)
    absolute = joint.pivot(
        index="moisture_factor", columns="temperature_delta_c", values="drying_time_corrected_h"
    ).reindex(index=delta.index, columns=delta.columns)
    matrix = delta.to_numpy(dtype=float)
    maximum = float(np.max(np.abs(matrix)))
    norm = mcolors.TwoSlopeNorm(vmin=-maximum, vcenter=0.0, vmax=maximum)

    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    image = ax.imshow(matrix, cmap="RdBu_r", norm=norm, origin="lower", aspect="auto")
    ax.set_xticks(np.arange(delta.shape[1]), [f"{value:+.0f}" for value in delta.columns])
    ax.set_yticks(np.arange(delta.shape[0]), [f"{value:.2f}" for value in delta.index])
    ax.set(
        xlabel="4 h后平均温度偏差 ΔT / °C",
        ylabel="4 h后平均水分相对基准的倍率",
        title="温度—水分联合扰动对干燥时间的影响",
    )
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            rgba = image.cmap(image.norm(matrix[row, col]))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            text_color = "white" if luminance < 0.48 else "#202020"
            ax.text(
                col,
                row,
                f"{absolute.iloc[row, col]:.2f} h\n({matrix[row, col]:+.2f})",
                ha="center",
                va="center",
                fontsize=7.5,
                color=text_color,
            )
    cbar = fig.colorbar(image, ax=ax, pad=0.025, aspect=28)
    cbar.set_label("相对基准的干燥时间变化 / h")
    save_png(fig, figure_dir / "q3_sensitivity_joint_heatmap.png")


def plot_window_sensitivity(
    frame: pd.DataFrame, baseline_h: float, figure_dir: Path
) -> None:
    window = frame.loc[frame["group"] == "window"].sort_values("averaging_window_h")
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.65))

    axes[0].plot(
        window["averaging_window_h"],
        window["plateau_temperature_c"],
        color=PALETTE["blue"],
        marker="o",
        lw=1.7,
    )
    axes[0].axvline(1.0, color=PALETTE["gray"], ls="--", lw=1.0)
    axes[0].set(
        xlabel="平均时间窗 / h",
        ylabel="平均温度 / °C",
        title="时间窗与平均温度",
    )

    axes[1].plot(
        window["averaging_window_h"],
        window["plateau_moisture_kg_kg"],
        color=PALETTE["orange"],
        marker="s",
        lw=1.7,
    )
    axes[1].axvline(1.0, color=PALETTE["gray"], ls="--", lw=1.0)
    axes[1].set(
        xlabel="平均时间窗 / h",
        ylabel="平均水分 / (kg/kg)",
        title="时间窗与平均水分",
    )

    axes[2].plot(
        window["averaging_window_h"],
        window["drying_time_corrected_h"],
        color=PALETTE["violet"],
        marker="o",
        lw=1.8,
    )
    axes[2].axvline(1.0, color=PALETTE["gray"], ls="--", lw=1.0)
    axes[2].axhline(baseline_h, color=PALETTE["red"], ls=":", lw=1.0)
    axes[2].set(
        xlabel="平均时间窗 / h",
        ylabel="干燥时间 / h",
        title="时间窗与干燥时间",
    )
    for label, ax in zip("abc", axes):
        ax.grid(axis="y", color="#D8D8D8", lw=0.6, alpha=0.65)
        ax.text(-0.20, 1.05, label, transform=ax.transAxes, fontweight="bold", fontsize=10)
    fig.subplots_adjust(wspace=0.55)
    save_png(fig, figure_dir / "q3_sensitivity_averaging_window.png")


def summarize(
    frame: pd.DataFrame,
    base_t: float,
    base_c: float,
    fine_baseline_h: float,
    coarse_baseline_h: float,
    correction_h: float,
    sensitivities: dict,
) -> dict:
    joint = joint_subset(frame)
    window = frame.loc[frame["group"] == "window"].copy()
    minimum = joint.loc[joint["drying_time_corrected_h"].idxmin()]
    maximum = joint.loc[joint["drying_time_corrected_h"].idxmax()]
    return {
        "method": {
            "parameter_sweep_grid": "verified coarse grid: 0.025 cm; 2/60/2 s",
            "correction": "fine baseline plus coarse scenario change from coarse baseline",
            "temperature_deltas_C": TEMPERATURE_DELTAS_C.tolist(),
            "moisture_factors": MOISTURE_FACTORS.tolist(),
            "averaging_windows_h": AVERAGING_WINDOWS_H.tolist(),
        },
        "baseline": {
            "plateau_temperature_C": base_t,
            "plateau_moisture_kg_kg": base_c,
            "fine_drying_time_h": fine_baseline_h,
            "coarse_drying_time_h": coarse_baseline_h,
            "additive_correction_h": correction_h,
        },
        "local_sensitivity": sensitivities,
        "joint_range": {
            "minimum_corrected_drying_time_h": float(minimum["drying_time_corrected_h"]),
            "minimum_temperature_C": float(minimum["plateau_temperature_c"]),
            "minimum_moisture_kg_kg": float(minimum["plateau_moisture_kg_kg"]),
            "maximum_corrected_drying_time_h": float(maximum["drying_time_corrected_h"]),
            "maximum_temperature_C": float(maximum["plateau_temperature_c"]),
            "maximum_moisture_kg_kg": float(maximum["plateau_moisture_kg_kg"]),
        },
        "averaging_window": {
            "minimum_drying_time_h": float(window["drying_time_corrected_h"].min()),
            "maximum_drying_time_h": float(window["drying_time_corrected_h"].max()),
            "range_h": float(window["drying_time_corrected_h"].max() - window["drying_time_corrected_h"].min()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    script_path = Path(__file__).resolve()
    default_repo = script_path.parents[1]
    parser.add_argument("--repo-root", type=Path, default=default_repo)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_root = args.output_root.resolve() if args.output_root else repo_root
    result_dir = output_root / "results" / "A_problem3_sensitivity"
    figure_dir = output_root / "picture" / "A_problem3_sensitivity"
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    environment = q2.load_and_preprocess_environment(q2.find_default_data_path(repo_root))
    scenarios, base_t, base_c = build_scenarios(environment)
    final_table_path = result_dir / "sensitivity_scenarios.csv"
    if args.reuse and final_table_path.exists():
        frame = pd.read_csv(final_table_path)
    else:
        frame = run_parameter_sweep(scenarios, environment, result_dir, args.workers)

    validation_path = repo_root / "results" / "A_problem3_drying_time" / "validation_summary.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    fine_baseline_h = float(validation["drying_time"]["production_h"])
    frame, coarse_baseline_h, correction_h = add_corrected_times(
        frame, base_t, base_c, fine_baseline_h
    )
    frame.to_csv(final_table_path, index=False, encoding="utf-8-sig", float_format="%.10f")

    sensitivities = local_sensitivities(frame, base_t, base_c, fine_baseline_h)
    summary = summarize(
        frame,
        base_t,
        base_c,
        fine_baseline_h,
        coarse_baseline_h,
        correction_h,
        sensitivities,
    )
    (result_dir / "sensitivity_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    apply_style()
    plot_one_factor(frame, base_t, base_c, fine_baseline_h, sensitivities, figure_dir)
    plot_joint_surface(frame, figure_dir)
    plot_window_sensitivity(frame, fine_baseline_h, figure_dir)

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Results: {result_dir}", flush=True)
    print(f"Figures: {figure_dir}", flush=True)


if __name__ == "__main__":
    main()
