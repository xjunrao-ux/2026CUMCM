"""A题问题1：有限体积模型的单因素参数灵敏度分析。

保持附件1边界输入和其余参数不变，分别改变一个物性/边界参数。
响应采用30 min温升与失水量的相对变化，避免摄氏温度和含水量基值
对百分比产生误导。图件仅输出PNG，数据输出CSV。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

import a1_coupled_fvm as model
import a1_temperature_fvm as heat


BASELINE = {
    "rho": 820.0,
    "cp": 2600.0,
    "k": 0.36,
    "h": 25.0,
    "hm": 8.0e-7,
    "D0": 7.0e-9,
    "beta": 0.89,
}

PARAMETERS = {
    "rho": ("密度 ρ", "kg/m³", "temperature"),
    "cp": ("比热容 cp", "J/(kg·K)", "temperature"),
    "k": ("导热系数 k", "W/(m·K)", "temperature"),
    "h": ("对流换热系数 h", "W/(m²·K)", "temperature"),
    "hm": ("对流传质系数 hm", "m/s", "moisture"),
    "D0": ("扩散前因子 D0", "m²/s", "moisture"),
    "beta": ("扩散指数系数 β", "—", "moisture"),
}

FACTOR_GRID = np.round(np.arange(0.50, 1.5001, 0.05), 2)
INITIAL_T = 28.0
INITIAL_C = 2.55


@dataclass
class RunMetrics:
    center_temperature_c: float = math.nan
    surface_temperature_c: float = math.nan
    average_temperature_c: float = math.nan
    center_moisture_kg_kg: float = math.nan
    surface_moisture_kg_kg: float = math.nan
    average_moisture_kg_kg: float = math.nan


def configure_plot() -> None:
    available = {font.name for font in fm.fontManager.ttflist}
    fonts = [x for x in ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"] if x in available]
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": fonts or ["DejaVu Sans"],
        "axes.unicode_minus": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "axes.titlesize": 9.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 6.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def reset_parameters() -> None:
    heat.DENSITY_KG_M3 = BASELINE["rho"]
    heat.SPECIFIC_HEAT_J_KG_K = BASELINE["cp"]
    heat.THERMAL_CONDUCTIVITY_W_M_K = BASELINE["k"]
    heat.CONVECTION_COEFFICIENT_W_M2_K = BASELINE["h"]
    model.MASS_TRANSFER_COEFFICIENT_M_S = BASELINE["hm"]
    set_diffusivity(BASELINE["D0"], BASELINE["beta"])


def set_diffusivity(d0: float, beta: float) -> None:
    def diffusivity(moisture_kg_kg: np.ndarray) -> np.ndarray:
        moisture = np.asarray(moisture_kg_kg, dtype=float)
        if np.any(moisture <= 0.0):
            raise ValueError("Moisture concentration must remain positive in D(C).")
        return d0 * np.exp(-beta / moisture)

    model.moisture_diffusivity = diffusivity


def run_temperature(environment: model.EnvironmentData) -> RunMetrics:
    result = heat.simulate_temperature(
        environment.times_s, environment.temperature_c,
        radial_step_cm=0.05, time_step_s=1.0,
    )
    weights = result.capacities_j_k / np.sum(result.capacities_j_k)
    return RunMetrics(
        center_temperature_c=float(result.temperature_c[-1, 0]),
        surface_temperature_c=float(result.temperature_c[-1, -1]),
        average_temperature_c=float(np.sum(weights * result.temperature_c[-1])),
    )


def run_moisture(environment: model.EnvironmentData) -> RunMetrics:
    result = model.simulate_moisture(
        environment.times_s, environment.moisture_kg_kg,
        radial_step_cm=0.05, time_step_s=1.0,
        picard_tolerance=1.0e-9,
    )
    return RunMetrics(
        center_moisture_kg_kg=float(result.moisture_kg_kg[-1, 0]),
        surface_moisture_kg_kg=float(result.surface_moisture_kg_kg[-1]),
        average_moisture_kg_kg=float(
            np.sum(result.volumes_m3_m * result.moisture_kg_kg[-1])
            / np.sum(result.volumes_m3_m)
        ),
    )


def set_parameter(key: str, value: float) -> None:
    if key == "rho":
        heat.DENSITY_KG_M3 = value
    elif key == "cp":
        heat.SPECIFIC_HEAT_J_KG_K = value
    elif key == "k":
        heat.THERMAL_CONDUCTIVITY_W_M_K = value
    elif key == "h":
        heat.CONVECTION_COEFFICIENT_W_M2_K = value
    elif key == "hm":
        model.MASS_TRANSFER_COEFFICIENT_M_S = value
    elif key == "D0":
        set_diffusivity(value, BASELINE["beta"])
    elif key == "beta":
        set_diffusivity(BASELINE["D0"], value)
    else:
        raise KeyError(key)


def response_changes(metrics: RunMetrics, baseline: RunMetrics, domain: str) -> dict[str, float]:
    if domain == "temperature":
        values = {
            "center_temperature_c": metrics.center_temperature_c,
            "surface_temperature_c": metrics.surface_temperature_c,
            "average_temperature_c": metrics.average_temperature_c,
            "center_temperature_rise_c": metrics.center_temperature_c - INITIAL_T,
            "surface_temperature_rise_c": metrics.surface_temperature_c - INITIAL_T,
            "average_temperature_rise_c": metrics.average_temperature_c - INITIAL_T,
        }
        for point in ["center", "surface", "average"]:
            denominator = getattr(baseline, f"{point}_temperature_c") - INITIAL_T
            values[f"{point}_response_change_pct"] = 100.0 * (
                values[f"{point}_temperature_rise_c"] / denominator - 1.0
            )
    else:
        values = {
            "center_moisture_kg_kg": metrics.center_moisture_kg_kg,
            "surface_moisture_kg_kg": metrics.surface_moisture_kg_kg,
            "average_moisture_kg_kg": metrics.average_moisture_kg_kg,
            "center_moisture_loss_kg_kg": INITIAL_C - metrics.center_moisture_kg_kg,
            "surface_moisture_loss_kg_kg": INITIAL_C - metrics.surface_moisture_kg_kg,
            "average_moisture_loss_kg_kg": INITIAL_C - metrics.average_moisture_kg_kg,
        }
        for point in ["center", "surface", "average"]:
            denominator = INITIAL_C - getattr(baseline, f"{point}_moisture_kg_kg")
            # 中心30 min几乎没有失水，分母过小，不作为阈值判断依据。
            values[f"{point}_response_change_pct"] = (
                math.nan if abs(denominator) < 1.0e-6
                else 100.0 * (values[f"{point}_moisture_loss_kg_kg"] / denominator - 1.0)
            )
    return values


def max_relevant_change(row: dict[str, float], domain: str) -> float:
    points = ["center", "surface", "average"] if domain == "temperature" else ["surface", "average"]
    return max(abs(row[f"{point}_response_change_pct"]) for point in points)


def crossing_factor(rows: list[dict[str, float]], target: float, side: str) -> float | None:
    ordered = sorted(rows, key=lambda r: r["factor"])
    base_index = next(i for i, row in enumerate(ordered) if math.isclose(row["factor"], 1.0))
    indices = range(base_index, -1, -1) if side == "lower" else range(base_index, len(ordered))
    previous = ordered[base_index]
    for index in indices:
        current = ordered[index]
        y0, y1 = previous["max_abs_response_change_pct"], current["max_abs_response_change_pct"]
        if y1 >= target and not math.isclose(current["factor"], 1.0):
            if math.isclose(y1, y0):
                return float(current["factor"])
            fraction = (target - y0) / (y1 - y0)
            return float(previous["factor"] + fraction * (current["factor"] - previous["factor"]))
        previous = current
    return None


def interpolate_change(rows: list[dict[str, float]], factor: float) -> float:
    factors = np.asarray([r["factor"] for r in rows])
    changes = np.asarray([r["max_abs_response_change_pct"] for r in rows])
    assert np.all(np.diff(factors) > 0.0)
    return float(np.interp(factor, factors, changes))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # 温度域与水分域具有不同响应列；取字段并集，空白留空。
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def fmt_factor(value: float | None, side: str) -> str:
    if value is None:
        # 扫描边界处仍未越过阈值，只能断言稳定区间至少延伸至该边界。
        return "≤0.50" if side == "lower" else "≥1.50"
    return f"{value:.3f}"


def plot_response_curves(path: Path, grouped: dict[str, list[dict]]) -> None:
    configure_plot()
    fig, axes = plt.subplots(2, 2, figsize=(7.20, 5.30), constrained_layout=True)
    panels = [
        (axes[0, 0], "temperature", "center", "(a) 中心温升"),
        (axes[0, 1], "temperature", "surface", "(b) 表面温升"),
        (axes[1, 0], "moisture", "average", "(c) 体积平均失水量"),
        (axes[1, 1], "moisture", "surface", "(d) 表面失水量"),
    ]
    palette = {
        "rho": "#0072B2", "cp": "#56B4E9", "k": "#009E73", "h": "#E69F00",
        "hm": "#CC79A7", "D0": "#D55E00", "beta": "#6A3D9A",
    }
    markers = {"rho": "o", "cp": "s", "k": "^", "h": "D", "hm": "o", "D0": "^", "beta": "s"}
    for ax, domain, point, title in panels:
        for key, rows in grouped.items():
            if PARAMETERS[key][2] != domain:
                continue
            x = np.asarray([r["factor"] for r in rows])
            y = np.asarray([r[f"{point}_response_change_pct"] for r in rows])
            ax.plot(x, y, color=palette[key], lw=1.35, marker=markers[key], ms=2.7,
                    markevery=2, label=PARAMETERS[key][0])
        ax.axhspan(-5, 5, color="#D9EAD3", alpha=0.35, zorder=0)
        ax.axhline(10, color="#666666", ls="--", lw=0.8)
        ax.axhline(-10, color="#666666", ls="--", lw=0.8)
        ax.axvline(1.0, color="#333333", lw=0.7)
        ax.set(title=title, xlabel="参数倍率（相对基准）", ylabel="响应相对变化 / %", xlim=(0.5, 1.5))
        ax.grid(True, color="#E2E2E2", lw=0.45)
        ax.legend(frameon=False, ncol=2, loc="best")
    fig.suptitle("A题问题1参数灵敏度：30 min温升与失水响应", fontsize=10.0, fontweight="bold")
    fig.savefig(path, dpi=600, facecolor="white")
    plt.close(fig)


def plot_tornado(path: Path, grouped: dict[str, list[dict]]) -> None:
    configure_plot()
    labels, lower, upper, domains = [], [], [], []
    for key, rows in grouped.items():
        labels.append(PARAMETERS[key][0])
        lower.append(interpolate_change(rows, 0.8))
        upper.append(interpolate_change(rows, 1.2))
        domains.append(PARAMETERS[key][2])
    order = np.argsort(np.maximum(lower, upper))
    labels = [labels[i] for i in order]
    lower = np.asarray(lower)[order]
    upper = np.asarray(upper)[order]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.20, 3.65), constrained_layout=True)
    ax.barh(y - 0.18, lower, height=0.34, color="#56B4E9", label="参数降至0.8倍")
    ax.barh(y + 0.18, upper, height=0.34, color="#D55E00", label="参数增至1.2倍")
    ax.axvline(5, color="#888888", ls=":", lw=0.9)
    ax.axvline(10, color="#555555", ls="--", lw=0.9)
    ax.set(yticks=y, yticklabels=labels, xlabel="最大响应变化 / %",
           title="参数±20%扰动的影响强度（取相关响应中的最大值）")
    ax.grid(axis="x", color="#E2E2E2", lw=0.45)
    ax.legend(frameon=False, ncol=2, loc="lower right")
    fig.savefig(path, dpi=600, facecolor="white")
    plt.close(fig)


def plot_threshold_table(path: Path, thresholds: list[dict]) -> None:
    configure_plot()
    headers = ["参数", "基准值", "5%预警稳定倍率区间", "10%显著稳定倍率区间", "±20%最大响应"]
    cells = []
    for row in thresholds:
        unit = row["unit"]
        baseline = f'{row["baseline_value"]:.3g} {unit}' if unit != "—" else f'{row["baseline_value"]:.3g}'
        cells.append([
            row["parameter_name"], baseline,
            f'{row["lower_factor_5pct_display"]} ～ {row["upper_factor_5pct_display"]}',
            f'{row["lower_factor_10pct_display"]} ～ {row["upper_factor_10pct_display"]}',
            f'{row["max_change_at_minus20_pct"]:.2f}% / {row["max_change_at_plus20_pct"]:.2f}%',
        ])
    fig, ax = plt.subplots(figsize=(7.20, 3.25), constrained_layout=True)
    ax.axis("off")
    table = ax.table(cellText=cells, colLabels=headers, cellLoc="center", loc="center",
                     colWidths=[0.19, 0.16, 0.23, 0.23, 0.19])
    table.auto_set_font_size(False)
    table.set_fontsize(6.6)
    table.scale(1.0, 1.45)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#D0D0D0")
        cell.set_linewidth(0.45)
        if r == 0:
            cell.set_facecolor("#1F4E79")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        elif r % 2 == 0:
            cell.set_facecolor("#F3F6F8")
    ax.set_title("灵敏度阈值汇总（区间内最大响应变化低于对应阈值）", fontsize=9.5, fontweight="bold", pad=7)
    fig.savefig(path, dpi=600, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result_dir = args.output_root / "results" / "A_problem1_sensitivity"
    figure_dir = args.output_root / "picture" / "A_problem1_sensitivity"
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    environment = model.load_environment(args.data)
    reset_parameters()
    baseline_temperature = run_temperature(environment)
    baseline_moisture = run_moisture(environment)
    baseline_by_domain = {"temperature": baseline_temperature, "moisture": baseline_moisture}

    rows: list[dict] = []
    grouped: dict[str, list[dict]] = {}
    for key, (name, unit, domain) in PARAMETERS.items():
        parameter_rows = []
        for factor in FACTOR_GRID:
            reset_parameters()
            actual = BASELINE[key] * float(factor)
            set_parameter(key, actual)
            metrics = run_temperature(environment) if domain == "temperature" else run_moisture(environment)
            values = response_changes(metrics, baseline_by_domain[domain], domain)
            row = {
                "parameter_key": key,
                "parameter_name": name,
                "domain": domain,
                "unit": unit,
                "baseline_value": BASELINE[key],
                "factor": float(factor),
                "parameter_value": actual,
                **values,
            }
            row["max_abs_response_change_pct"] = max_relevant_change(row, domain)
            rows.append(row)
            parameter_rows.append(row)
        grouped[key] = parameter_rows

    thresholds: list[dict] = []
    for key, parameter_rows in grouped.items():
        lower5 = crossing_factor(parameter_rows, 5.0, "lower")
        upper5 = crossing_factor(parameter_rows, 5.0, "upper")
        lower10 = crossing_factor(parameter_rows, 10.0, "lower")
        upper10 = crossing_factor(parameter_rows, 10.0, "upper")
        name, unit, domain = PARAMETERS[key]
        thresholds.append({
            "parameter_key": key,
            "parameter_name": name,
            "domain": domain,
            "unit": unit,
            "baseline_value": BASELINE[key],
            "lower_factor_5pct": lower5,
            "upper_factor_5pct": upper5,
            "lower_factor_10pct": lower10,
            "upper_factor_10pct": upper10,
            "lower_factor_5pct_display": fmt_factor(lower5, "lower"),
            "upper_factor_5pct_display": fmt_factor(upper5, "upper"),
            "lower_factor_10pct_display": fmt_factor(lower10, "lower"),
            "upper_factor_10pct_display": fmt_factor(upper10, "upper"),
            "lower_value_10pct": None if lower10 is None else BASELINE[key] * lower10,
            "upper_value_10pct": None if upper10 is None else BASELINE[key] * upper10,
            "max_change_at_minus20_pct": interpolate_change(parameter_rows, 0.8),
            "max_change_at_plus20_pct": interpolate_change(parameter_rows, 1.2),
        })

    write_csv(result_dir / "parameter_sensitivity_sweep.csv", rows)
    write_csv(result_dir / "parameter_sensitivity_thresholds.csv", thresholds)
    plot_response_curves(figure_dir / "sensitivity_response_curves.png", grouped)
    plot_tornado(figure_dir / "sensitivity_tornado.png", grouped)
    plot_threshold_table(figure_dir / "sensitivity_threshold_table.png", thresholds)

    rho_rows, cp_rows = grouped["rho"], grouped["cp"]
    rho_cp_difference = max(abs(a["max_abs_response_change_pct"] - b["max_abs_response_change_pct"])
                            for a, b in zip(rho_rows, cp_rows))
    report = {
        "method": "one-at-a-time factor sweep",
        "factor_grid": FACTOR_GRID.tolist(),
        "large_difference_definition": "at least 10% change in temperature rise or moisture loss",
        "warning_definition": "at least 5% change in temperature rise or moisture loss",
        "screening_grid": {"radial_step_cm": 0.05, "time_step_s": 1.0},
        "baseline_screening": {
            "center_temperature_c": baseline_temperature.center_temperature_c,
            "surface_temperature_c": baseline_temperature.surface_temperature_c,
            "average_temperature_c": baseline_temperature.average_temperature_c,
            "center_moisture_kg_kg": baseline_moisture.center_moisture_kg_kg,
            "surface_moisture_kg_kg": baseline_moisture.surface_moisture_kg_kg,
            "average_moisture_kg_kg": baseline_moisture.average_moisture_kg_kg,
        },
        "rho_cp_curve_max_difference_pct_point": rho_cp_difference,
        "thresholds": thresholds,
    }
    (result_dir / "sensitivity_validation_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    reset_parameters()
    # Windows控制台可能使用GBK；转义非ASCII字符可确保脚本正常结束。
    print(json.dumps(report, ensure_ascii=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
