"""Local one-at-a-time sensitivity analysis for A-problem question 2.

The sweep perturbs four uncertain transport quantities around the baseline:
surface heat-transfer coefficient h, surface mass-transfer coefficient km,
thermal conductivity lambda, and moisture diffusivity D.  Each parameter is
evaluated at 0.8, 0.9, 1.0, 1.1, and 1.2 times its baseline value while all
other quantities remain fixed.  The ±20% interval is a diagnostic perturbation
range, not a statistical confidence interval.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

import a2_coupled_fvm as model


FACTORS = np.array([0.8, 0.9, 1.0, 1.1, 1.2], dtype=float)
TRAJECTORY_FACTORS = (0.8, 1.0, 1.2)
TRAJECTORY_TIMES_S = np.arange(0.0, model.END_TIME_S + 60.0, 60.0)
PARAMETERS = {
    "h": {
        "label": "换热系数 h",
        "short": "h",
        "baseline": model.HEAT_TRANSFER_COEFFICIENT_W_M2_K,
        "unit": "W/(m²·K)",
    },
    "km": {
        "label": "传质系数 km",
        "short": "km",
        "baseline": model.MASS_TRANSFER_COEFFICIENT_M_S,
        "unit": "m/s",
    },
    "lambda": {
        "label": "导热系数 λ",
        "short": "λ",
        "baseline": 1.0,
        "unit": "基准倍率",
    },
    "D": {
        "label": "扩散系数 D",
        "short": "D",
        "baseline": 1.0,
        "unit": "基准倍率",
    },
}

METRIC_LABELS = {
    "heating_gain_c": "3 h平均升温量",
    "water_removed_kg_kg": "3 h平均脱水量",
    "temperature_gradient_c": "3 h表面—中心温差",
    "moisture_gradient_kg_kg": "3 h中心—表面水分差",
}

PALETTE = {
    0.8: "#3B5B92",
    1.0: "#555555",
    1.2: "#D55E00",
    "minus": "#4C78A8",
    "plus": "#E07B39",
    "grid": "#D8D8D8",
}


@dataclass
class ScenarioResult:
    scenario_id: str
    parameter: str
    factor: float
    parameter_value: float
    elapsed_s: float
    metrics: dict[str, float]
    trajectory: dict[str, np.ndarray]


def configure_style() -> None:
    available = {font.name for font in fm.fontManager.ttflist}
    candidates = ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"]
    selected = [name for name in candidates if name in available]
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": selected or ["DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8.0,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.4,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "axes.unicode_minus": False,
        }
    )


def style_axis(axis: plt.Axes) -> None:
    axis.grid(True, color=PALETTE["grid"], linewidth=0.45, alpha=0.75)
    axis.set_axisbelow(True)


def save_png(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def scenario_kwargs(parameter: str, factor: float) -> dict[str, float]:
    kwargs = {
        "heat_transfer_coefficient_w_m2_k": model.HEAT_TRANSFER_COEFFICIENT_W_M2_K,
        "mass_transfer_coefficient_m_s": model.MASS_TRANSFER_COEFFICIENT_M_S,
        "conductivity_multiplier": 1.0,
        "diffusivity_multiplier": 1.0,
    }
    if parameter == "h":
        kwargs["heat_transfer_coefficient_w_m2_k"] *= factor
    elif parameter == "km":
        kwargs["mass_transfer_coefficient_m_s"] *= factor
    elif parameter == "lambda":
        kwargs["conductivity_multiplier"] = factor
    elif parameter == "D":
        kwargs["diffusivity_multiplier"] = factor
    elif parameter != "baseline":
        raise ValueError(f"Unknown sensitivity parameter: {parameter}")
    return kwargs


def weighted_average(field: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    return np.sum(field * volumes[None, :], axis=1) / np.sum(volumes)


def run_scenario(
    environment: model.EnvironmentData,
    parameter: str,
    factor: float,
) -> ScenarioResult:
    scenario_id = "baseline" if parameter == "baseline" else f"{parameter}_{factor:.1f}"
    started = time.perf_counter()
    result = model.simulate_coupled(
        environment,
        nominal_radial_step_cm=0.025,
        time_step_s=1.0,
        **scenario_kwargs(parameter, factor),
    )
    elapsed = time.perf_counter() - started

    avg_temperature = weighted_average(result.temperature_c, result.grid.volumes_m3_m)
    avg_moisture = weighted_average(result.moisture_kg_kg, result.grid.volumes_m3_m)
    center_temperature = model.sample_result(
        result, "temperature", TRAJECTORY_TIMES_S, np.array([0.0])
    )[:, 0]
    center_moisture = model.sample_result(
        result, "moisture", TRAJECTORY_TIMES_S, np.array([0.0])
    )[:, 0]
    indices = TRAJECTORY_TIMES_S.astype(int)
    surface_temperature = result.surface_temperature_c[indices]
    surface_moisture = result.surface_moisture_kg_kg[indices]

    final_center_temperature = float(center_temperature[-1])
    final_center_moisture = float(center_moisture[-1])
    final_surface_temperature = float(surface_temperature[-1])
    final_surface_moisture = float(surface_moisture[-1])
    final_average_temperature = float(avg_temperature[-1])
    final_average_moisture = float(avg_moisture[-1])

    metrics = {
        "average_temperature_c": final_average_temperature,
        "average_moisture_kg_kg": final_average_moisture,
        "center_temperature_c": final_center_temperature,
        "surface_temperature_c": final_surface_temperature,
        "center_moisture_kg_kg": final_center_moisture,
        "surface_moisture_kg_kg": final_surface_moisture,
        "heating_gain_c": final_average_temperature - model.INITIAL_TEMPERATURE_C,
        "water_removed_kg_kg": model.INITIAL_MOISTURE_KG_KG - final_average_moisture,
        "temperature_gradient_c": final_surface_temperature - final_center_temperature,
        "moisture_gradient_kg_kg": final_center_moisture - final_surface_moisture,
        "maximum_picard_iterations": float(result.maximum_picard_iterations),
    }
    trajectory = {
        "time_s": TRAJECTORY_TIMES_S.copy(),
        "average_temperature_c": avg_temperature[indices],
        "average_moisture_kg_kg": avg_moisture[indices],
        "center_temperature_c": center_temperature,
        "surface_temperature_c": surface_temperature,
        "center_moisture_kg_kg": center_moisture,
        "surface_moisture_kg_kg": surface_moisture,
    }
    baseline_value = 1.0 if parameter == "baseline" else PARAMETERS[parameter]["baseline"]
    return ScenarioResult(
        scenario_id=scenario_id,
        parameter=parameter,
        factor=factor,
        parameter_value=float(baseline_value * factor),
        elapsed_s=elapsed,
        metrics=metrics,
        trajectory=trajectory,
    )


def build_scenarios(environment: model.EnvironmentData) -> list[ScenarioResult]:
    cases = [("baseline", 1.0)]
    cases.extend(
        (parameter, float(factor))
        for parameter in PARAMETERS
        for factor in FACTORS
        if not math.isclose(float(factor), 1.0)
    )
    results: list[ScenarioResult] = []
    total = len(cases)
    for index, (parameter, factor) in enumerate(cases, start=1):
        print(
            f"[{index:02d}/{total:02d}] running {parameter} x {factor:.1f}",
            flush=True,
        )
        scenario = run_scenario(environment, parameter, factor)
        results.append(scenario)
        print(
            "  completed in {:.2f} s; Tavg={:.4f} C, Cavg={:.4f} kg/kg".format(
                scenario.elapsed_s,
                scenario.metrics["average_temperature_c"],
                scenario.metrics["average_moisture_kg_kg"],
            ),
            flush=True,
        )
    return results


def scenario_lookup(
    scenarios: list[ScenarioResult], parameter: str, factor: float
) -> ScenarioResult:
    if math.isclose(factor, 1.0):
        parameter = "baseline"
    for scenario in scenarios:
        if scenario.parameter == parameter and math.isclose(scenario.factor, factor):
            return scenario
    raise KeyError((parameter, factor))


def calculate_elasticities(
    scenarios: list[ScenarioResult],
) -> dict[str, dict[str, float]]:
    baseline = scenario_lookup(scenarios, "baseline", 1.0)
    elasticities: dict[str, dict[str, float]] = {}
    for parameter in PARAMETERS:
        low = scenario_lookup(scenarios, parameter, 0.9)
        high = scenario_lookup(scenarios, parameter, 1.1)
        elasticities[parameter] = {}
        for metric in METRIC_LABELS:
            y0 = baseline.metrics[metric]
            elasticity = (high.metrics[metric] - low.metrics[metric]) / (0.2 * y0)
            elasticities[parameter][metric] = float(elasticity)
    return elasticities


def write_metrics_csv(path: Path, scenarios: list[ScenarioResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_names = list(next(iter(scenarios)).metrics)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "scenario_id",
                "parameter",
                "factor",
                "parameter_value",
                "elapsed_s",
                *metric_names,
            ]
        )
        for scenario in scenarios:
            writer.writerow(
                [
                    scenario.scenario_id,
                    scenario.parameter,
                    f"{scenario.factor:.1f}",
                    f"{scenario.parameter_value:.10g}",
                    f"{scenario.elapsed_s:.4f}",
                    *[f"{scenario.metrics[name]:.10g}" for name in metric_names],
                ]
            )


def write_trajectory_csv(path: Path, scenarios: list[ScenarioResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    trajectory_names = [
        name for name in next(iter(scenarios)).trajectory if name != "time_s"
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["scenario_id", "parameter", "factor", "time_s", *trajectory_names]
        )
        for scenario in scenarios:
            for row_index, time_s in enumerate(scenario.trajectory["time_s"]):
                writer.writerow(
                    [
                        scenario.scenario_id,
                        scenario.parameter,
                        f"{scenario.factor:.1f}",
                        f"{time_s:.0f}",
                        *[
                            f"{scenario.trajectory[name][row_index]:.10g}"
                            for name in trajectory_names
                        ],
                    ]
                )


def write_elasticity_csv(
    path: Path, elasticities: dict[str, dict[str, float]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["parameter", *METRIC_LABELS])
        for parameter in PARAMETERS:
            writer.writerow(
                [
                    parameter,
                    *[f"{elasticities[parameter][metric]:.8f}" for metric in METRIC_LABELS],
                ]
            )


def render_trajectory_figure(
    path: Path, scenarios: list[ScenarioResult]
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(7.09, 5.25), constrained_layout=True)
    panels = [
        (axes[0, 0], "h", "average_temperature_c", "(a) 换热系数对平均温度的影响", "温度 / ℃"),
        (axes[0, 1], "lambda", "average_temperature_c", "(b) 导热系数对平均温度的影响", "温度 / ℃"),
        (axes[1, 0], "km", "average_moisture_kg_kg", "(c) 传质系数对平均水分的影响", "水分浓度 / (kg/kg)"),
        (axes[1, 1], "D", "average_moisture_kg_kg", "(d) 扩散系数对平均水分的影响", "水分浓度 / (kg/kg)"),
    ]
    for axis, parameter, field, title, ylabel in panels:
        for factor in TRAJECTORY_FACTORS:
            scenario = scenario_lookup(scenarios, parameter, factor)
            axis.plot(
                scenario.trajectory["time_s"] / 3600.0,
                scenario.trajectory[field],
                color=PALETTE[factor],
                linewidth=1.35 if factor == 1.0 else 1.15,
                label=f"{factor:.1f} 倍",
            )
        axis.set(title=title, xlabel="时间 / h", ylabel=ylabel, xlim=(0.0, 3.0))
        axis.set_xticks(np.arange(0.0, 3.01, 0.5))
        style_axis(axis)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        handlelength=2.7,
        columnspacing=1.4,
    )
    save_png(figure, path)


def percent_change(value: float, baseline: float) -> float:
    return 100.0 * (value / baseline - 1.0)


def render_tornado_figure(path: Path, scenarios: list[ScenarioResult]) -> None:
    baseline = scenario_lookup(scenarios, "baseline", 1.0)
    parameter_order = ["D", "km", "lambda", "h"]
    figure, axes = plt.subplots(1, 2, figsize=(7.09, 3.45), constrained_layout=True)
    panels = [
        (axes[0], "heating_gain_c", "(a) 3 h平均升温量相对变化"),
        (axes[1], "water_removed_kg_kg", "(b) 3 h平均脱水量相对变化"),
    ]
    y = np.arange(len(parameter_order))
    for axis, metric, title in panels:
        minus = np.array(
            [
                percent_change(
                    scenario_lookup(scenarios, parameter, 0.8).metrics[metric],
                    baseline.metrics[metric],
                )
                for parameter in parameter_order
            ]
        )
        plus = np.array(
            [
                percent_change(
                    scenario_lookup(scenarios, parameter, 1.2).metrics[metric],
                    baseline.metrics[metric],
                )
                for parameter in parameter_order
            ]
        )
        axis.barh(
            y - 0.17,
            minus,
            height=0.3,
            color=PALETTE["minus"],
            label="参数 -20%",
        )
        axis.barh(
            y + 0.17,
            plus,
            height=0.3,
            color=PALETTE["plus"],
            label="参数 +20%",
        )
        axis.axvline(0.0, color="#555555", linewidth=0.75)
        axis.set(
            title=title,
            xlabel="相对基准变化 / %",
            yticks=y,
            yticklabels=[PARAMETERS[p]["label"] for p in parameter_order],
        )
        style_axis(axis)
        axis.grid(axis="y", visible=False)
    axes[0].legend(loc="best")
    save_png(figure, path)


def render_elasticity_heatmap(
    path: Path, elasticities: dict[str, dict[str, float]]
) -> None:
    parameter_order = list(PARAMETERS)
    metric_order = list(METRIC_LABELS)
    matrix = np.array(
        [
            [elasticities[parameter][metric] for metric in metric_order]
            for parameter in parameter_order
        ]
    )
    limit = max(0.05, float(np.max(np.abs(matrix))))
    figure, axis = plt.subplots(figsize=(7.09, 3.15), constrained_layout=True)
    image = axis.imshow(
        matrix,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        aspect="auto",
    )
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            text_color = "white" if abs(value) > 0.58 * limit else "#222222"
            axis.text(
                column,
                row,
                f"{value:+.3f}",
                ha="center",
                va="center",
                fontsize=7.4,
                color=text_color,
            )
    axis.set(
        title="±10%局部归一化灵敏度系数",
        xticks=np.arange(len(metric_order)),
        xticklabels=[METRIC_LABELS[metric] for metric in metric_order],
        yticks=np.arange(len(parameter_order)),
        yticklabels=[PARAMETERS[parameter]["label"] for parameter in parameter_order],
    )
    colorbar = figure.colorbar(image, ax=axis, pad=0.025, aspect=24)
    colorbar.set_label("归一化灵敏度系数")
    colorbar.outline.set_linewidth(0.6)
    save_png(figure, path)


def build_summary(
    scenarios: list[ScenarioResult],
    elasticities: dict[str, dict[str, float]],
) -> dict:
    baseline = scenario_lookup(scenarios, "baseline", 1.0)
    ranking = {}
    for metric in METRIC_LABELS:
        ranking[metric] = sorted(
            (
                {
                    "parameter": parameter,
                    "elasticity": elasticities[parameter][metric],
                    "absolute_elasticity": abs(elasticities[parameter][metric]),
                }
                for parameter in PARAMETERS
            ),
            key=lambda row: row["absolute_elasticity"],
            reverse=True,
        )
    changes_20_percent = {}
    for parameter in PARAMETERS:
        changes_20_percent[parameter] = {}
        for factor in (0.8, 1.2):
            scenario = scenario_lookup(scenarios, parameter, factor)
            changes_20_percent[parameter][f"factor_{factor:.1f}"] = {
                metric: percent_change(scenario.metrics[metric], baseline.metrics[metric])
                for metric in METRIC_LABELS
            }
    return {
        "method": {
            "type": "one-at-a-time local sensitivity analysis",
            "factors": FACTORS.tolist(),
            "local_elasticity_factors": [0.9, 1.1],
            "spatial_discretization": "80 surface-refined finite volumes",
            "time_step_s": 1.0,
            "trajectory_output_interval_s": 60.0,
            "note": "The ±20% range is a diagnostic perturbation, not a confidence interval.",
        },
        "parameters": PARAMETERS,
        "baseline_metrics": baseline.metrics,
        "elasticities": elasticities,
        "absolute_elasticity_ranking": ranking,
        "relative_changes_at_20_percent": changes_20_percent,
        "total_runtime_s": float(sum(scenario.elapsed_s for scenario in scenarios)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    repo_root = args.repo_root.resolve() if args.repo_root else script_path.parents[1]
    output_root = args.output_root.resolve() if args.output_root else repo_root
    result_dir = output_root / "results" / "A_problem2_sensitivity"
    picture_dir = output_root / "picture" / "A_problem2_sensitivity"
    result_dir.mkdir(parents=True, exist_ok=True)
    picture_dir.mkdir(parents=True, exist_ok=True)

    environment = model.load_and_preprocess_environment(model.find_default_data_path(repo_root))
    configure_style()
    scenarios = build_scenarios(environment)
    elasticities = calculate_elasticities(scenarios)

    write_metrics_csv(result_dir / "sensitivity_metrics.csv", scenarios)
    write_trajectory_csv(result_dir / "sensitivity_trajectories_60s.csv", scenarios)
    write_elasticity_csv(result_dir / "local_elasticity.csv", elasticities)
    summary = build_summary(scenarios, elasticities)
    with (result_dir / "sensitivity_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)

    render_trajectory_figure(picture_dir / "sensitivity_trajectories.png", scenarios)
    render_tornado_figure(picture_dir / "sensitivity_tornado.png", scenarios)
    render_elasticity_heatmap(
        picture_dir / "sensitivity_elasticity_heatmap.png", elasticities
    )

    print(f"Results: {result_dir}")
    print(f"Figures: {picture_dir}")
    print(json.dumps(summary["absolute_elasticity_ranking"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
