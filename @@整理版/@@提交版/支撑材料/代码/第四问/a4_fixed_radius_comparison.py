"""问题四参数下的恒定半径与实测收缩半径对照计算。

用途：保持附录 4 物性、环境边界、初值和换热/传质条件不变，仅比较
R(t)=2 cm 与附件 2 实测 R(t) 两种几何假设，从而分离半径收缩的影响。

输入：比赛题目中的附件 1、附件 2，以及正式问题四收缩结果。
输出：result/A_q4_radius_comparison 下的完整 60 s 对照表、6 h 汇总表、
影响指标和数值检验结果。

运行：python code/a4_fixed_radius_comparison.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_solver(repo_root: Path):
    code_dir = repo_root / "code"
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    import a2_coupled_fvm as q2
    import a4_shrinkage_fvm as q4

    return q2, q4


def make_constant_radius_program(q4, measured_radius):
    """沿用附件 2 的时间节点，但把所有半径固定为初始 2 cm。"""
    return q4.RadiusProgram(
        measured_radius.source_times_s.copy(),
        np.full_like(measured_radius.source_radii_cm, 2.0, dtype=float),
    )


def prefixed_output(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    renamed = {}
    for column in frame.columns:
        if column == "time_s":
            continue
        renamed[column] = f"{prefix}_{column}"
    return frame.rename(columns=renamed)


def value_at_time(frame: pd.DataFrame, column: str, time_s: float) -> float:
    valid = frame[["time_s", column]].dropna()
    return float(np.interp(time_s, valid["time_s"], valid[column]))


def scenario_table(q4, result, scenario: str) -> pd.DataFrame:
    times = q4.build_table_times(result.drying_time_s)
    indices = q4.record_indices_at_times(result, times)
    output_indices = [
        int(np.flatnonzero(np.isclose(q4.OUTPUT_RADII_CM, radius))[0])
        for radius in q4.TABLE_RADII_CM
    ]
    rows = []
    for time_s, index in zip(times, indices):
        row = {
            "scenario": scenario,
            "time_s": float(time_s),
            "time_h": float(time_s / 3600.0),
            "is_drying_endpoint": bool(
                math.isclose(time_s, result.drying_time_s, abs_tol=1.0e-8)
            ),
            "radius_cm": float(result.radii_cm[index]),
        }
        for radius, output_index in zip(q4.TABLE_RADII_CM, output_indices):
            value = result.moisture_kg_kg[index, output_index]
            row[f"C_r_{radius:g}_cm_kg_kg"] = (
                None if np.isnan(value) else float(value)
            )
        row["C_surface_kg_kg"] = float(result.surface_moisture_kg_kg[index])
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(path: Path, summary: dict, table_6h: pd.DataFrame) -> None:
    impact = summary["shrinkage_impact"]
    fixed = summary["constant_radius"]
    shrinking = summary["shrinking_radius"]
    table_display = table_6h.copy()
    table_display["scenario"] = table_display["scenario"].map(
        {"constant_radius": "恒定半径", "shrinking_radius": "实测收缩半径"}
    )
    table_display["time_h"] = table_display["time_h"].map(lambda x: f"{x:.4f}")
    table_display["radius_cm"] = table_display["radius_cm"].map(lambda x: f"{x:.4f}")
    for column in [c for c in table_display if c.startswith("C_")]:
        table_display[column] = table_display[column].map(
            lambda x: "—" if pd.isna(x) else f"{x:.4f}"
        )
    table_display = table_display.drop(columns=["time_s", "is_drying_endpoint"])
    table_display = table_display.rename(
        columns={
            "scenario": "情景",
            "time_h": "时间/h",
            "radius_cm": "当前半径/cm",
            "C_r_0_cm_kg_kg": "0 cm",
            "C_r_0.5_cm_kg_kg": "0.5 cm",
            "C_r_1_cm_kg_kg": "1.0 cm",
            "C_r_1.5_cm_kg_kg": "1.5 cm",
            "C_r_2_cm_kg_kg": "2.0 cm",
            "C_surface_kg_kg": "表面",
        }
    )
    headers = [str(column) for column in table_display.columns]
    markdown_lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|",
    ]
    for row in table_display.itertuples(index=False, name=None):
        markdown_lines.append("| " + " | ".join(str(value) for value in row) + " |")
    markdown = "\n".join(markdown_lines)
    text = f"""# 问题四参数下半径收缩影响对比

## 对比口径

两组计算均采用附录 4 的密度、比热容、导热系数和水分扩散系数，采用相同的附件 1 环境边界、初值、对流换热/传质系数、有限体积离散和严格全域含水率阈值。唯一差异是：恒定尺寸情景令半径始终为 2 cm；收缩情景采用附件 2 的分段线性半径轨迹。

## 主要结果

| 指标 | 恒定半径 | 实测收缩半径 | 收缩影响 |
|---|---:|---:|---:|
| 烘干时间/h | {fixed['drying_time_h']:.6f} | {shrinking['drying_time_h']:.6f} | 缩短 {impact['time_saved_h']:.6f} h |
| 烘干时间/d | {fixed['drying_time_d']:.6f} | {shrinking['drying_time_d']:.6f} | 缩短 {impact['time_reduction_pct']:.2f}% |
| 结束半径/cm | {fixed['radius_at_endpoint_cm']:.4f} | {shrinking['radius_at_endpoint_cm']:.4f} | 半径减小 {impact['radius_reduction_pct_at_shrinking_endpoint']:.2f}% |
| 相对速度 | 1.0000 | {impact['drying_speed_factor']:.4f} | 收缩情景达到阈值所需时间为基线的 {impact['shrinking_to_fixed_time_ratio']:.2%} |

在收缩情景结束时，恒定半径情景的中心水分浓度仍为 {impact['fixed_center_at_shrinking_endpoint_kg_kg']:.4f} kg/kg，比阈值高 {impact['fixed_center_excess_over_threshold_at_shrinking_endpoint_kg_kg']:.4f} kg/kg。结果表明，若忽略尺寸收缩，会把问题四参数下的烘干时间由 {shrinking['drying_time_h']:.2f} h 高估到 {fixed['drying_time_h']:.2f} h；半径收缩是本工况中决定干燥时长的强影响因素。

## 6 h 间隔完整对照表

表中“—”表示该固定物理位置已经位于收缩后的药材表面之外；每个情景最后一行是该情景首次满足全域水分浓度严格小于 0.15 kg/kg 的时刻。

{markdown}

## 数值说明

- 两组结果都由同一问题四求解器计算，半径是唯一被替换的输入函数。
- 恒定半径正式网格与粗网格的干燥时间差为 {fixed['coarse_fine_difference_s']:.0f} s。
- 收缩情景正式网格与粗网格的干燥时间差为 {shrinking['coarse_fine_difference_s']:.0f} s。
- 完整 60 s 对照表保留所有 0.1 cm 固定物理位置；收缩情景中超出当前半径的位置保持为空值，不做外推。
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--result-dir", type=Path)
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    repo_root = args.repo_root.resolve() if args.repo_root else script_path.parents[1]
    result_dir = (
        args.result_dir.resolve()
        if args.result_dir
        else repo_root / "result" / "A_q4_radius_comparison"
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    q2, q4 = load_solver(repo_root)

    attachment1 = q4.find_attachment(repo_root, "附件1.xlsx")
    attachment2 = q4.find_attachment(repo_root, "附件2.xlsx")
    environment = q2.load_and_preprocess_environment(attachment1)
    boundary = q4.build_boundary_program(environment)
    measured_radius = q4.load_radius_program(attachment2)
    constant_radius = make_constant_radius_program(q4, measured_radius)

    print("[1/2] 求解问题四参数下的恒定半径正式网格...", flush=True)
    fixed_production = q4.simulate_until_dry(
        boundary,
        constant_radius,
        nominal_initial_step_cm=0.00625,
        early_step_s=1.0,
        late_step_s=30.0,
        final_step_s=1.0,
    )
    print("[2/2] 求解恒定半径粗网格用于收敛核对...", flush=True)
    fixed_coarse = q4.simulate_until_dry(
        boundary,
        constant_radius,
        nominal_initial_step_cm=0.0125,
        early_step_s=1.0,
        late_step_s=30.0,
        final_step_s=1.0,
    )

    q4.write_full_csv(result_dir / "A_q4_constant_radius_full_60s.csv", fixed_production)
    q4.write_temperature_csv(
        result_dir / "A_q4_constant_radius_temperature_60s_K.csv", fixed_production
    )
    fixed_validation = q4.build_validation(
        boundary, constant_radius, fixed_production, fixed_coarse
    )
    (result_dir / "A_q4_constant_radius_validation.json").write_text(
        json.dumps(fixed_validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    shrinking_dir = repo_root / "result" / "A_q4_shrinkage"
    shrinking_validation = json.loads(
        (shrinking_dir / "validation_summary.json").read_text(encoding="utf-8")
    )
    shrinking_full = pd.read_csv(
        shrinking_dir / "moisture_full_60s_0p1cm.csv", encoding="utf-8-sig"
    )
    fixed_full = pd.read_csv(
        result_dir / "A_q4_constant_radius_full_60s.csv", encoding="utf-8-sig"
    )
    fixed_prefixed = prefixed_output(fixed_full, "fixed")
    shrinking_prefixed = prefixed_output(shrinking_full, "shrink")
    full_comparison = fixed_prefixed.merge(
        shrinking_prefixed, on="time_s", how="outer", validate="one_to_one"
    ).sort_values("time_s")
    full_comparison.insert(1, "time_h", full_comparison["time_s"] / 3600.0)
    for radius in q4.OUTPUT_RADII_CM:
        label = f"r_{radius:.1f}_cm"
        full_comparison[f"delta_fixed_minus_shrink_{label}"] = (
            full_comparison[f"fixed_{label}"] - full_comparison[f"shrink_{label}"]
        )
    full_comparison["delta_fixed_minus_shrink_surface"] = (
        full_comparison["fixed_surface"] - full_comparison["shrink_surface"]
    )
    full_comparison.to_csv(
        result_dir / "A_q4_radius_comparison_full_60s.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.10g",
    )

    fixed_table = scenario_table(q4, fixed_production, "constant_radius")
    # 用正式求解结果的完整记录重建同一字段口径，避免依赖展示用四位小数表。
    # 6 h 收缩表由正式 CSV 在精确记录时刻提取；末端使用验证 JSON 的未舍入时刻。
    shrink_times = np.arange(
        q4.TABLE_INTERVAL_S,
        shrinking_validation["drying_time"]["production_s"] - 1.0e-9,
        q4.TABLE_INTERVAL_S,
    )
    shrink_times = np.append(
        shrink_times, shrinking_validation["drying_time"]["production_s"]
    )
    shrink_rows = []
    for time_s in shrink_times:
        row = {
            "scenario": "shrinking_radius",
            "time_s": float(time_s),
            "time_h": float(time_s / 3600.0),
            "is_drying_endpoint": bool(
                math.isclose(
                    time_s,
                    shrinking_validation["drying_time"]["production_s"],
                    abs_tol=1.0e-8,
                )
            ),
            "radius_cm": value_at_time(shrinking_full, "radius_cm", time_s),
        }
        for radius in q4.TABLE_RADII_CM:
            column = f"r_{radius:.1f}_cm"
            row[f"C_r_{radius:g}_cm_kg_kg"] = (
                value_at_time(shrinking_full, column, time_s)
                if shrinking_full[column].notna().any()
                and radius <= row["radius_cm"] + 1.0e-10
                else None
            )
        row["C_surface_kg_kg"] = value_at_time(shrinking_full, "surface", time_s)
        shrink_rows.append(row)
    shrinking_table = pd.DataFrame(shrink_rows)
    table_6h = pd.concat([fixed_table, shrinking_table], ignore_index=True)
    table_6h.to_csv(
        result_dir / "A_q4_radius_comparison_table_6h.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.10g",
    )

    fixed_time_h = fixed_production.drying_time_s / 3600.0
    shrinking_time_s = float(shrinking_validation["drying_time"]["production_s"])
    shrinking_time_h = shrinking_time_s / 3600.0
    shrinking_endpoint_radius = float(shrinking_validation["radius"]["drying_time_cm"])
    fixed_center_at_shrink_end = value_at_time(
        fixed_full, "r_0.0_cm", shrinking_time_s
    )
    time_saved_h = fixed_time_h - shrinking_time_h
    summary = {
        "comparison_definition": {
            "shared": (
                "Appendix 4 properties, Attachment 1 boundary, initial state, "
                "heat/mass transfer coefficients, FVM solver and threshold"
            ),
            "only_difference": "R(t)=2 cm versus Attachment 2 measured R(t)",
            "threshold_kg_kg": q4.DRYING_THRESHOLD_KG_KG,
        },
        "constant_radius": {
            "drying_time_s": float(fixed_production.drying_time_s),
            "drying_time_h": float(fixed_time_h),
            "drying_time_d": float(fixed_time_h / 24.0),
            "radius_at_endpoint_cm": 2.0,
            "previous_maximum_moisture_kg_kg": float(
                fixed_production.previous_maximum_moisture
            ),
            "final_maximum_moisture_kg_kg": float(
                fixed_production.final_maximum_moisture
            ),
            "controlling_radius_cm": float(fixed_production.controlling_radius_cm),
            "coarse_fine_difference_s": float(
                abs(fixed_production.drying_time_s - fixed_coarse.drying_time_s)
            ),
        },
        "shrinking_radius": {
            "drying_time_s": shrinking_time_s,
            "drying_time_h": shrinking_time_h,
            "drying_time_d": shrinking_time_h / 24.0,
            "radius_at_endpoint_cm": shrinking_endpoint_radius,
            "previous_maximum_moisture_kg_kg": float(
                shrinking_validation["drying_time"]["previous_maximum_moisture"]
            ),
            "final_maximum_moisture_kg_kg": float(
                shrinking_validation["drying_time"]["final_maximum_moisture"]
            ),
            "controlling_radius_cm": float(
                shrinking_validation["drying_time"]["controlling_radius_cm"]
            ),
            "coarse_fine_difference_s": float(
                shrinking_validation["drying_time"]["coarse_production_difference_s"]
            ),
        },
        "shrinkage_impact": {
            "time_saved_s": float(fixed_production.drying_time_s - shrinking_time_s),
            "time_saved_h": float(time_saved_h),
            "time_reduction_pct": float(100.0 * time_saved_h / fixed_time_h),
            "shrinking_to_fixed_time_ratio": float(shrinking_time_h / fixed_time_h),
            "drying_speed_factor": float(fixed_time_h / shrinking_time_h),
            "radius_reduction_pct_at_shrinking_endpoint": float(
                100.0 * (1.0 - shrinking_endpoint_radius / 2.0)
            ),
            "cross_section_reduction_pct_at_shrinking_endpoint": float(
                100.0 * (1.0 - (shrinking_endpoint_radius / 2.0) ** 2)
            ),
            "fixed_center_at_shrinking_endpoint_kg_kg": fixed_center_at_shrink_end,
            "fixed_center_excess_over_threshold_at_shrinking_endpoint_kg_kg": float(
                fixed_center_at_shrink_end - q4.DRYING_THRESHOLD_KG_KG
            ),
        },
        "data_counts": {
            "fixed_records": int(fixed_full.shape[0]),
            "shrinking_records": int(shrinking_full.shape[0]),
            "combined_time_records": int(full_comparison.shape[0]),
            "output_physical_radii": int(q4.OUTPUT_RADII_CM.size),
        },
    }
    (result_dir / "A_q4_radius_comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary_rows = [
        {
            "scenario": "constant_radius",
            "radius_rule": "R(t)=2 cm",
            "drying_time_s": fixed_production.drying_time_s,
            "drying_time_h": fixed_time_h,
            "drying_time_d": fixed_time_h / 24.0,
            "endpoint_radius_cm": 2.0,
            "time_change_vs_fixed_h": 0.0,
            "time_change_vs_fixed_pct": 0.0,
            "speed_factor_vs_fixed": 1.0,
        },
        {
            "scenario": "shrinking_radius",
            "radius_rule": "Attachment 2 R(t)",
            "drying_time_s": shrinking_time_s,
            "drying_time_h": shrinking_time_h,
            "drying_time_d": shrinking_time_h / 24.0,
            "endpoint_radius_cm": shrinking_endpoint_radius,
            "time_change_vs_fixed_h": -time_saved_h,
            "time_change_vs_fixed_pct": -100.0 * time_saved_h / fixed_time_h,
            "speed_factor_vs_fixed": fixed_time_h / shrinking_time_h,
        },
    ]
    pd.DataFrame(summary_rows).to_csv(
        result_dir / "A_q4_radius_comparison_summary.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.10g",
    )
    write_report(
        result_dir / "A_q4_radius_comparison_report.md", summary, table_6h
    )
    print(json.dumps(summary["shrinkage_impact"], ensure_ascii=False, indent=2))
    print(f"结果目录：{result_dir}", flush=True)


if __name__ == "__main__":
    main()
