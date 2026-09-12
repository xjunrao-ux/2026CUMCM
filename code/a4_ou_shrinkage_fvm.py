"""A题问题4：Huber-OU随机烘房边界驱动的收缩有限体积模型。

本程序直接复用 ``a4_shrinkage_fvm.py`` 的半径收缩、温度/水分有限体积离散、
物性关系、Picard迭代、终止判据和表6/result4输出规则。唯一的模型改动是：

* 0--4 h仍使用附件1清洗后的逐分钟实测烘房温度和水分浓度；
* 3--4 h稳定段沿用问题3的Huber稳健标定与精确离散联合OU模型；
* 4--72 h用固定随机种子生成逐分钟环境路径，有限体积子步在线性插值后取值；
* OU温度由摄氏度显式换算为开尔文后传给问题4求解器。

默认结果写入 ``results/A_problem4_shrinkage_huber_ou``，不会覆盖恒均值模型。
"""

from __future__ import annotations

import argparse
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
import a3_drying_time_fvm_huber_ou as q3_ou
import a4_shrinkage_fvm as base


@dataclass(frozen=True)
class KelvinOUBoundaryProgram:
    """Adapt the question-3 OU path (degC) to the question-4 solver (K)."""

    ou: q3_ou.OUBoundaryProgram

    @property
    def plateau_temperature_k(self) -> float:
        return self.ou.temperature_fit.equilibrium_mean + 273.15

    @property
    def plateau_moisture_kg_kg(self) -> float:
        return self.ou.moisture_fit.equilibrium_mean

    def values(self, time_s: float) -> tuple[float, float]:
        temperature_c, moisture = self.ou.values(time_s)
        return temperature_c + 273.15, moisture


def _realized_summary(values: np.ndarray) -> dict[str, float]:
    """Summarize generated values after, but not including, the 4 h join row."""
    generated = np.asarray(values[1:], dtype=float)
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
    path = repo_root / "results" / "A_problem4_shrinkage" / "validation_summary.json"
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
    boundary: KelvinOUBoundaryProgram,
    radius_program: base.RadiusProgram,
    production: base.ShrinkageResult,
    coarse: base.ShrinkageResult,
) -> dict:
    validation = base.build_validation(
        boundary,
        radius_program,
        production,
        coarse,
    )
    path = boundary.ou
    scope = validation["model_scope"]
    scope["environment_after_4h"] = (
        "Huber-calibrated exact-discrete joint OU path at 60 s intervals"
    )
    scope["ou_fit_interval_s"] = [q3_ou.FIT_START_TIME_S, q3_ou.MEASURED_END_TIME_S]
    scope["ou_path_interval_s"] = q3_ou.OU_STEP_S
    scope["ou_path_end_time_h"] = q3_ou.OU_END_TIME_S / 3600.0
    scope["random_seed"] = path.seed
    scope["sigma_multiplier"] = path.sigma_multiplier
    scope["temperature_conversion"] = "T_K = T_degC + 273.15"

    validation["ou_equilibrium"] = validation.pop("constant_boundary")
    validation["ou_calibration"] = {
        "method": (
            "Huber Proposal-2 equilibrium and scale; bounded Huber IRLS for "
            "the exact-discrete OU/AR(1) coefficient"
        ),
        "temperature": asdict(path.temperature_fit),
        "moisture": asdict(path.moisture_fit),
        "robust_innovation_correlation": path.innovation_correlation,
        "interpretation": (
            "A phi value at its lower bound means that persistence is shorter "
            "than the one-minute observation interval; the simulated series is "
            "then practically white noise around the Huber equilibrium."
        ),
    }
    validation["realized_ou_boundary_4h_72h"] = {
        "temperature_c": _realized_summary(path.ou_temperature_c),
        "moisture_kg_kg": _realized_summary(path.ou_moisture_kg_kg),
    }

    time_step_counts = validation["checks"]["time_step_counts"]
    time_step_counts["ou_environment"] = time_step_counts.pop(
        "constant_environment"
    )
    internal_increments = np.diff(production.moisture_kg_kg, axis=0)
    surface_increments = np.diff(production.surface_moisture_kg_kg)
    finite_internal = internal_increments[np.isfinite(internal_increments)]
    all_increments = np.concatenate((finite_internal, surface_increments))
    checks = validation["checks"]
    checks.pop("time_monotonicity_violation_count")
    checks["saved_pointwise_moisture_increase_count"] = int(
        np.count_nonzero(all_increments > 1.0e-8)
    )
    checks["maximum_saved_pointwise_increase_kg_kg"] = float(
        max(float(np.max(all_increments)), 0.0)
    )
    checks["pointwise_monotonicity_note"] = (
        "Small local increases are physically admissible under a stochastic "
        "ambient-moisture boundary and are not numerical monotonicity failures."
    )

    reference = load_constant_mean_reference(repo_root)
    if reference is not None:
        delta_s = production.drying_time_s - reference["drying_time_s"]
        reference["delta_ou_minus_mean_s"] = delta_s
        reference["delta_ou_minus_mean_min"] = delta_s / 60.0
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


def build_workbooks(script_path: Path, repo_root: Path, result_dir: Path) -> tuple[Path, Path]:
    builder = script_path.with_name("a4_ou_build_workbooks.cjs")
    template = (
        repo_root
        / "比赛题目"
        / "CUMCM2026Problems"
        / "A题"
        / "附件"
        / "附件3"
        / "result4.xlsx"
    )
    if not builder.exists() or not template.exists():
        raise FileNotFoundError("Cannot find the question-4 workbook builder or template.")

    result4 = result_dir / "result4.xlsx"
    environment_workbook = result_dir / "ou_environment_parameters.xlsx"
    preview_dir = repo_root / "tmp" / "A_problem4_huber_ou_verify"
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
            str(result_dir / "ou_boundary_4h_72h_60s.csv"),
            str(result_dir / "ou_parameters.json"),
            str(result_dir / "validation_summary.json"),
            str(result4),
            str(environment_workbook),
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
    return result4, environment_workbook


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attachment1", type=Path, help="附件1路径")
    parser.add_argument("--attachment2", type=Path, help="附件2路径")
    parser.add_argument("--repo-root", type=Path, help="项目根目录")
    parser.add_argument("--output-dir", type=Path, help="结果输出目录")
    parser.add_argument("--seed", type=int, default=q3_ou.DEFAULT_SEED)
    parser.add_argument(
        "--randomize-seed",
        action="store_true",
        help="每次从系统熵生成并记录新种子，用于重复蒙特卡洛路径",
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
        help="只生成CSV/JSON，不构建两个Excel工作簿",
    )
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    repo_root = args.repo_root.resolve() if args.repo_root else script_path.parents[1]
    attachment1 = (
        args.attachment1.resolve()
        if args.attachment1
        else base.find_attachment(repo_root, "附件1.xlsx")
    )
    attachment2 = (
        args.attachment2.resolve()
        if args.attachment2
        else base.find_attachment(repo_root, "附件2.xlsx")
    )
    result_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else repo_root / "results" / "A_problem4_shrinkage_huber_ou"
    )
    result_dir.mkdir(parents=True, exist_ok=True)

    actual_seed = secrets.randbits(32) if args.randomize_seed else args.seed
    environment = q2.load_and_preprocess_environment(attachment1)
    ou_boundary = q3_ou.build_huber_ou_boundary(
        environment,
        seed=actual_seed,
        sigma_multiplier=args.sigma_multiplier,
    )
    boundary = KelvinOUBoundaryProgram(ou_boundary)
    radius_program = base.load_radius_program(attachment2)

    # 粗、细网格必须共享同一条OU路径，随机差异不能混入网格收敛误差。
    coarse = base.simulate_until_dry(
        boundary,
        radius_program,
        nominal_initial_step_cm=0.0125,
        early_step_s=1.0,
        late_step_s=30.0,
        final_step_s=1.0,
    )
    production = base.simulate_until_dry(
        boundary,
        radius_program,
        nominal_initial_step_cm=0.00625,
        early_step_s=1.0,
        late_step_s=30.0,
        final_step_s=1.0,
    )

    table_times = base.build_table_times(production.drying_time_s)
    base.write_full_csv(result_dir / "moisture_full_60s_0p1cm.csv", production)
    base.write_temperature_csv(
        result_dir / "temperature_auxiliary_60s_0p1cm_K.csv",
        production,
    )
    base.write_table_csv(
        result_dir / "table6_moisture.csv",
        production,
        table_times,
    )
    base.write_workbook_payload(result_dir / "result4_payload.json", production)
    q3_ou.write_ou_boundary_csv(
        result_dir / "ou_boundary_4h_72h_60s.csv",
        ou_boundary,
    )

    parameters = {
        "method": (
            "Huber Proposal-2 equilibrium and scale; bounded Huber IRLS for "
            "the exact-discrete OU/AR(1) coefficient"
        ),
        "fit_interval_s": [q3_ou.FIT_START_TIME_S, q3_ou.MEASURED_END_TIME_S],
        "path_interval_s": q3_ou.OU_STEP_S,
        "path_end_time_s": q3_ou.OU_END_TIME_S,
        "random_seed": ou_boundary.seed,
        "sigma_multiplier": ou_boundary.sigma_multiplier,
        "temperature": asdict(ou_boundary.temperature_fit),
        "moisture": asdict(ou_boundary.moisture_fit),
        "robust_innovation_correlation": ou_boundary.innovation_correlation,
        "question4_temperature_input_unit": "K",
        "temperature_conversion": "T_K = T_degC + 273.15",
    }
    with (result_dir / "ou_parameters.json").open("w", encoding="utf-8") as stream:
        json.dump(parameters, stream, ensure_ascii=False, indent=2)

    validation = build_validation(
        repo_root,
        boundary,
        radius_program,
        production,
        coarse,
    )
    with (result_dir / "validation_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(validation, stream, ensure_ascii=False, indent=2)

    result4_path = None
    environment_path = None
    if not args.skip_xlsx:
        result4_path, environment_path = build_workbooks(
            script_path,
            repo_root,
            result_dir,
        )

    print(
        "Huber-OU equilibrium: "
        f"{ou_boundary.temperature_fit.equilibrium_mean:.8f} deg C, "
        f"{ou_boundary.moisture_fit.equilibrium_mean:.10f} kg/kg",
        flush=True,
    )
    print(
        f"Random seed: {ou_boundary.seed}; sigma multiplier: "
        f"{ou_boundary.sigma_multiplier:g}",
        flush=True,
    )
    print(
        f"Drying time: {production.drying_time_s:.0f} s "
        f"= {production.drying_time_s / 3600.0:.6f} h",
        flush=True,
    )
    if result4_path is not None:
        print(f"result4 workbook: {result4_path}", flush=True)
    if environment_path is not None:
        print(f"OU environment workbook: {environment_path}", flush=True)
    print(f"Results: {result_dir}", flush=True)


if __name__ == "__main__":
    main()
