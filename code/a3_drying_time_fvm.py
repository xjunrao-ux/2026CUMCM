"""A problem, question 3: full-process drying time by coupled finite volumes.

The model continues the Appendix 3 heat-moisture equations until every radial
position has a dry-basis moisture concentration below 0.15 kg/kg.  Attachment
1 drives the boundary from 0 to 4 h.  Thereafter, the oven temperature and
moisture are held at their time-weighted averages over the final measured hour
(3-4 h).  Shrinkage and evaporation latent heat are excluded in question 3.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import a2_coupled_fvm as q2


MEASURED_END_TIME_S = 4.0 * 3600.0
PLATEAU_START_TIME_S = 3.0 * 3600.0
DRYING_THRESHOLD_KG_KG = 0.15
OUTPUT_INTERVAL_S = 60.0
TABLE_INTERVAL_S = 6.0 * 3600.0
MAXIMUM_SIMULATION_TIME_S = 7.0 * 24.0 * 3600.0
OUTPUT_RADII_CM = np.round(np.arange(0.0, 2.0 + 0.05, 0.1), 1)
TABLE_RADII_CM = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=float)


@dataclass
class BoundaryProgram:
    source_times_s: np.ndarray
    source_temperature_c: np.ndarray
    source_moisture_kg_kg: np.ndarray
    plateau_temperature_c: float
    plateau_moisture_kg_kg: float

    def values(self, time_s: float) -> tuple[float, float]:
        if time_s <= MEASURED_END_TIME_S:
            return (
                float(
                    np.interp(
                        time_s, self.source_times_s, self.source_temperature_c
                    )
                ),
                float(
                    np.interp(
                        time_s, self.source_times_s, self.source_moisture_kg_kg
                    )
                ),
            )
        return self.plateau_temperature_c, self.plateau_moisture_kg_kg


@dataclass
class LongDryingResult:
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


@dataclass
class StepResult:
    temperature_c: np.ndarray
    moisture_kg_kg: np.ndarray
    surface_temperature_c: float
    surface_moisture_kg_kg: float
    heat_boundary_conductance: float
    moisture_boundary_conductance: float
    iterations: int


def build_boundary_program(environment: q2.EnvironmentData) -> BoundaryProgram:
    """Use a time-weighted 3-4 h mean as the post-measurement plateau."""
    mask = (
        (environment.source_times_s >= PLATEAU_START_TIME_S)
        & (environment.source_times_s <= MEASURED_END_TIME_S)
    )
    times = environment.source_times_s[mask]
    temperature = environment.source_temperature_c[mask]
    moisture = environment.source_moisture_kg_kg[mask]
    if times.size < 2 or times[0] != PLATEAU_START_TIME_S or times[-1] != MEASURED_END_TIME_S:
        raise ValueError("Attachment 1 must contain the complete 3-4 h plateau interval.")
    duration = MEASURED_END_TIME_S - PLATEAU_START_TIME_S
    plateau_temperature = float(np.trapezoid(temperature, times) / duration)
    plateau_moisture = float(np.trapezoid(moisture, times) / duration)
    return BoundaryProgram(
        source_times_s=environment.source_times_s,
        source_temperature_c=environment.source_temperature_c,
        source_moisture_kg_kg=environment.source_moisture_kg_kg,
        plateau_temperature_c=plateau_temperature,
        plateau_moisture_kg_kg=plateau_moisture,
    )


def boundary_conductance(
    grid: q2.Grid, cell_coefficient: float, transfer_coefficient: float
) -> float:
    surface_area_per_length = 2.0 * math.pi * q2.RADIUS_M
    return surface_area_per_length / (
        (q2.RADIUS_M - grid.radii_m[-1]) / cell_coefficient
        + 1.0 / transfer_coefficient
    )


def advance_coupled_step(
    grid: q2.Grid,
    temperature_previous: np.ndarray,
    moisture_previous: np.ndarray,
    oven_temperature_c: float,
    oven_moisture_kg_kg: float,
    time_step_s: float,
    temperature_tolerance_c: float = 1.0e-8,
    moisture_tolerance: float = 1.0e-10,
    maximum_iterations: int = 30,
) -> StepResult:
    """Advance one fully coupled implicit time step by Picard iteration."""
    temperature_iterate = temperature_previous.copy()
    moisture_iterate = moisture_previous.copy()
    for iteration in range(1, maximum_iterations + 1):
        conductivity = q2.conductivity_w_m_k(moisture_iterate)
        volumetric_heat_capacity = (
            q2.density_kg_m3(moisture_iterate)
            * q2.heat_capacity_j_kg_k(moisture_iterate)
        )
        heat_system = q2.build_diffusion_system(
            grid,
            temperature_previous,
            conductivity,
            volumetric_heat_capacity,
            q2.HEAT_TRANSFER_COEFFICIENT_W_M2_K,
            oven_temperature_c,
            time_step_s,
        )
        temperature_updated = q2.solve_tridiagonal(*heat_system[:4])

        diffusivity = q2.moisture_diffusivity_m2_s(
            moisture_iterate, temperature_updated
        )
        moisture_system = q2.build_diffusion_system(
            grid,
            moisture_previous,
            diffusivity,
            np.ones_like(moisture_iterate),
            q2.MASS_TRANSFER_COEFFICIENT_M_S,
            oven_moisture_kg_kg,
            time_step_s,
        )
        moisture_updated = q2.solve_tridiagonal(*moisture_system[:4])

        temperature_error = float(
            np.max(np.abs(temperature_updated - temperature_iterate))
        )
        moisture_error = float(np.max(np.abs(moisture_updated - moisture_iterate)))
        temperature_iterate = temperature_updated
        moisture_iterate = moisture_updated
        if (
            temperature_error < temperature_tolerance_c
            and moisture_error < moisture_tolerance
        ):
            break
    else:
        raise RuntimeError("Coupled Picard iteration did not converge.")

    final_conductivity = q2.conductivity_w_m_k(moisture_iterate)
    final_diffusivity = q2.moisture_diffusivity_m2_s(
        moisture_iterate, temperature_iterate
    )
    surface_distance = q2.RADIUS_M - grid.radii_m[-1]
    surface_temperature = q2.reconstruct_surface_value(
        temperature_iterate[-1],
        oven_temperature_c,
        final_conductivity[-1],
        q2.HEAT_TRANSFER_COEFFICIENT_W_M2_K,
        surface_distance,
    )
    surface_moisture = q2.reconstruct_surface_value(
        moisture_iterate[-1],
        oven_moisture_kg_kg,
        final_diffusivity[-1],
        q2.MASS_TRANSFER_COEFFICIENT_M_S,
        surface_distance,
    )
    return StepResult(
        temperature_c=temperature_iterate,
        moisture_kg_kg=moisture_iterate,
        surface_temperature_c=surface_temperature,
        surface_moisture_kg_kg=surface_moisture,
        heat_boundary_conductance=boundary_conductance(
            grid,
            float(final_conductivity[-1]),
            q2.HEAT_TRANSFER_COEFFICIENT_W_M2_K,
        ),
        moisture_boundary_conductance=boundary_conductance(
            grid,
            float(final_diffusivity[-1]),
            q2.MASS_TRANSFER_COEFFICIENT_M_S,
        ),
        iterations=iteration,
    )


def reconstruct_center(grid: q2.Grid, values: np.ndarray) -> float:
    """Second-order even-symmetry reconstruction at r=0."""
    r0_squared = grid.radii_m[0] ** 2
    r1_squared = grid.radii_m[1] ** 2
    return float(
        (values[0] * r1_squared - values[1] * r0_squared)
        / (r1_squared - r0_squared)
    )


def sample_radial_field(
    grid: q2.Grid,
    values: np.ndarray,
    surface_value: float,
    query_radii_cm: np.ndarray,
) -> np.ndarray:
    extended_radii = np.concatenate(([0.0], grid.radii_m, [q2.RADIUS_M]))
    extended_values = np.concatenate(
        ([reconstruct_center(grid, values)], values, [surface_value])
    )
    return np.interp(query_radii_cm / 100.0, extended_radii, extended_values)


def choose_time_step(
    time_s: float,
    maximum_moisture: float,
    early_time_step_s: float,
    late_time_step_s: float,
    final_time_step_s: float,
    refinement_margin: float = 0.002,
) -> tuple[float, str]:
    if time_s < MEASURED_END_TIME_S:
        return early_time_step_s, "measured_environment"
    if maximum_moisture < DRYING_THRESHOLD_KG_KG + refinement_margin:
        return final_time_step_s, "threshold_refinement"
    return late_time_step_s, "constant_environment"


def simulate_until_dry(
    boundary: BoundaryProgram,
    nominal_radial_step_cm: float,
    early_time_step_s: float,
    late_time_step_s: float,
    final_time_step_s: float,
    refinement_margin: float = 0.002,
) -> LongDryingResult:
    """Run the full coupled model until all radial positions are dry."""
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
    next_output_time_s = OUTPUT_INTERVAL_S
    record_times: list[float] = []
    temperature_records: list[np.ndarray] = []
    moisture_records: list[np.ndarray] = []
    integrated_moisture_outflow = 0.0
    maximum_picard_iterations = 0
    time_step_counts = {
        "measured_environment": 0,
        "constant_environment": 0,
        "threshold_refinement": 0,
    }

    maximum_moisture = q2.INITIAL_MOISTURE_KG_KG
    last_not_dry_time = 0.0
    last_not_dry_maximum = maximum_moisture
    controlling_radius_cm = 0.0

    while time_s < MAXIMUM_SIMULATION_TIME_S:
        proposed_step, step_label = choose_time_step(
            time_s,
            maximum_moisture,
            early_time_step_s,
            late_time_step_s,
            final_time_step_s,
            refinement_margin,
        )
        time_step_s = proposed_step
        if time_s < MEASURED_END_TIME_S < time_s + time_step_s:
            time_step_s = MEASURED_END_TIME_S - time_s
        if time_s < next_output_time_s < time_s + time_step_s:
            time_step_s = next_output_time_s - time_s
        if time_s + time_step_s > MAXIMUM_SIMULATION_TIME_S:
            time_step_s = MAXIMUM_SIMULATION_TIME_S - time_s
        if time_step_s <= 0.0:
            raise RuntimeError("Non-positive adaptive time step encountered.")

        new_time = time_s + time_step_s
        oven_temperature, oven_moisture = boundary.values(new_time)
        step = advance_coupled_step(
            grid,
            temperature,
            moisture,
            oven_temperature,
            oven_moisture,
            time_step_s,
        )
        integrated_moisture_outflow += (
            time_step_s
            * step.moisture_boundary_conductance
            * (step.moisture_kg_kg[-1] - oven_moisture)
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

        center_moisture = reconstruct_center(grid, moisture)
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
        if is_output_time:
            record_times.append(time_s)
            temperature_records.append(
                sample_radial_field(
                    grid,
                    temperature,
                    surface_temperature,
                    OUTPUT_RADII_CM,
                )
            )
            moisture_records.append(
                sample_radial_field(
                    grid,
                    moisture,
                    surface_moisture,
                    OUTPUT_RADII_CM,
                )
            )
            next_output_time_s += OUTPUT_INTERVAL_S

        if maximum_moisture < DRYING_THRESHOLD_KG_KG:
            if not is_output_time:
                record_times.append(time_s)
                temperature_records.append(
                    sample_radial_field(
                        grid,
                        temperature,
                        surface_temperature,
                        OUTPUT_RADII_CM,
                    )
                )
                moisture_records.append(
                    sample_radial_field(
                        grid,
                        moisture,
                        surface_moisture,
                        OUTPUT_RADII_CM,
                    )
                )
            break

        last_not_dry_time = time_s
        last_not_dry_maximum = maximum_moisture
    else:
        raise RuntimeError("Maximum simulation time reached before drying.")

    return LongDryingResult(
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
    )


def values_at_times(result: LongDryingResult, query_times_s: np.ndarray) -> np.ndarray:
    """Return recorded 0.1 cm moisture rows at exact requested times."""
    result_lookup = {round(float(time), 8): index for index, time in enumerate(result.times_s)}
    rows = []
    for time in query_times_s:
        key = round(float(time), 8)
        if key not in result_lookup:
            raise ValueError(f"Requested time {time:g} s was not recorded.")
        rows.append(result.moisture_kg_kg[result_lookup[key]])
    return np.vstack(rows)


def build_table_times(drying_time_s: float) -> np.ndarray:
    regular = np.arange(
        TABLE_INTERVAL_S,
        drying_time_s - 1.0e-9,
        TABLE_INTERVAL_S,
        dtype=float,
    )
    if regular.size and math.isclose(regular[-1], drying_time_s, abs_tol=1.0e-8):
        return regular
    return np.append(regular, drying_time_s)


def table_values(result: LongDryingResult, table_times_s: np.ndarray) -> np.ndarray:
    output_values = values_at_times(result, table_times_s)
    column_indices = [
        int(np.where(np.isclose(OUTPUT_RADII_CM, radius))[0][0])
        for radius in TABLE_RADII_CM
    ]
    return output_values[:, column_indices]


def write_full_csv(path: Path, result: LongDryingResult) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_s", *[f"r_{r:.1f}_cm" for r in OUTPUT_RADII_CM]])
        for time_s, row in zip(result.times_s, result.moisture_kg_kg):
            writer.writerow([f"{time_s:.0f}", *[f"{value:.4f}" for value in row]])


def write_temperature_csv(path: Path, result: LongDryingResult) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_s", *[f"r_{r:.1f}_cm" for r in OUTPUT_RADII_CM]])
        for time_s, row in zip(result.times_s, result.temperature_c):
            writer.writerow([f"{time_s:.0f}", *[f"{value:.4f}" for value in row]])


def write_table_csv(
    path: Path, table_times_s: np.ndarray, values: np.ndarray, drying_time_s: float
) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["时间/h", *[f"{radius:g} cm" for radius in TABLE_RADII_CM]])
        for time_s, row in zip(table_times_s, values):
            label = (
                "烘干结束时间"
                if math.isclose(time_s, drying_time_s, abs_tol=1.0e-8)
                else f"{time_s / 3600:.0f}"
            )
            writer.writerow([label, *[f"{value:.4f}" for value in row]])


def build_validation(
    boundary: BoundaryProgram,
    production: LongDryingResult,
    coarse: LongDryingResult,
    table_times_s: np.ndarray,
    production_table: np.ndarray,
) -> dict:
    common_end = min(production.drying_time_s, coarse.drying_time_s)
    common_times = production.times_s[
        (production.times_s <= common_end)
        & np.isclose(np.mod(production.times_s, OUTPUT_INTERVAL_S), 0.0)
    ]
    coarse_lookup = {round(float(t), 8): i for i, t in enumerate(coarse.times_s)}
    common_times = np.asarray(
        [t for t in common_times if round(float(t), 8) in coarse_lookup],
        dtype=float,
    )
    production_common = values_at_times(production, common_times)
    coarse_common = values_at_times(coarse, common_times)
    refinement_change = np.abs(production_common - coarse_common)

    stored_change = float(
        np.sum(
            production.grid.volumes_m3_m
            * (
                production.final_internal_moisture_kg_kg
                - q2.INITIAL_MOISTURE_KG_KG
            )
        )
    )
    balance_residual = stored_change + production.integrated_moisture_outflow
    balance_scale = max(
        abs(stored_change), abs(production.integrated_moisture_outflow), 1.0e-30
    )
    radial_violations = int(
        np.count_nonzero(np.diff(production.moisture_kg_kg, axis=1) > 1.0e-8)
    )
    time_violations = int(
        np.count_nonzero(np.diff(production.moisture_kg_kg, axis=0) > 1.0e-8)
    )
    final_row = production.moisture_kg_kg[-1]
    final_max_output = float(np.max(final_row))

    return {
        "model_scope": {
            "geometry": "fixed-radius one-dimensional axisymmetric cylinder",
            "constitutive_laws": "Appendix 3 used over the full process",
            "latent_heat": "excluded",
            "shrinkage": "excluded in question 3",
            "drying_threshold_kg_kg": DRYING_THRESHOLD_KG_KG,
            "measured_boundary_interval_s": [0, int(MEASURED_END_TIME_S)],
            "post_4h_boundary": "time-weighted mean over Attachment 1 final hour",
            "production_nominal_radial_step_cm": 0.0125,
            "production_time_steps_s": [1.0, 30.0, 1.0],
            "coarse_nominal_radial_step_cm": 0.025,
            "coarse_time_steps_s": [2.0, 60.0, 2.0],
            "saved_interval_s": OUTPUT_INTERVAL_S,
        },
        "constant_boundary": {
            "temperature_C": boundary.plateau_temperature_c,
            "temperature_K": boundary.plateau_temperature_c + 273.15,
            "moisture_kg_kg": boundary.plateau_moisture_kg_kg,
        },
        "drying_time": {
            "production_s": production.drying_time_s,
            "production_h": production.drying_time_s / 3600.0,
            "coarse_s": coarse.drying_time_s,
            "coarse_h": coarse.drying_time_s / 3600.0,
            "coarse_production_difference_s": abs(
                production.drying_time_s - coarse.drying_time_s
            ),
            "controlling_radius_cm": production.controlling_radius_cm,
            "previous_time_s": production.last_not_dry_time_s,
            "previous_maximum_moisture": production.last_not_dry_maximum_moisture,
            "final_maximum_moisture": production.final_maximum_moisture,
            "final_output_grid_maximum_moisture": final_max_output,
        },
        "grid_refinement": {
            "common_record_count": int(common_times.size),
            "maximum_moisture_change_kg_kg": float(np.max(refinement_change)),
            "mean_moisture_change_kg_kg": float(np.mean(refinement_change)),
        },
        "moisture_balance_per_unit_length": {
            "stored_change_integral": stored_change,
            "integrated_boundary_outflow": production.integrated_moisture_outflow,
            "residual": balance_residual,
            "relative_residual": abs(balance_residual) / balance_scale,
        },
        "checks": {
            "previous_state_not_dry": bool(
                production.last_not_dry_maximum_moisture
                >= DRYING_THRESHOLD_KG_KG
            ),
            "final_state_all_dry": bool(
                production.final_maximum_moisture < DRYING_THRESHOLD_KG_KG
            ),
            "radial_order_violation_count": radial_violations,
            "time_monotonicity_violation_count": time_violations,
            "minimum_saved_moisture_kg_kg": float(
                np.min(production.moisture_kg_kg)
            ),
            "maximum_picard_iterations": production.maximum_picard_iterations,
            "time_step_counts": production.time_step_counts,
        },
        "table5": {
            "row_count": int(table_times_s.size),
            "last_time_h": float(table_times_s[-1] / 3600.0),
            "last_row": [float(value) for value in production_table[-1]],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="Path to Attachment 1 xlsx")
    parser.add_argument("--repo-root", type=Path, help="Repository root")
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    repo_root = args.repo_root.resolve() if args.repo_root else script_path.parents[1]
    data_path = args.data.resolve() if args.data else q2.find_default_data_path(repo_root)
    result_dir = repo_root / "results" / "A_problem3_drying_time"
    result_dir.mkdir(parents=True, exist_ok=True)

    environment = q2.load_and_preprocess_environment(data_path)
    boundary = build_boundary_program(environment)
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

    table_times = build_table_times(production.drying_time_s)
    table = table_values(production, table_times)
    write_full_csv(result_dir / "moisture_full_60s_0p1cm.csv", production)
    write_temperature_csv(result_dir / "temperature_auxiliary_60s_0p1cm.csv", production)
    write_table_csv(
        result_dir / "table5_moisture.csv",
        table_times,
        table,
        production.drying_time_s,
    )
    validation = build_validation(
        boundary, production, coarse, table_times, table
    )
    with (result_dir / "validation_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(validation, stream, ensure_ascii=False, indent=2)

    print(
        f"Plateau boundary: {boundary.plateau_temperature_c:.6f} deg C, "
        f"{boundary.plateau_moisture_kg_kg:.8f} kg/kg"
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
    for time_s, row in zip(table_times, table):
        label = "end" if math.isclose(time_s, production.drying_time_s) else f"{time_s / 3600:.0f} h"
        print(label, np.array2string(row, precision=4, suppress_small=False))
    print("\nValidation summary")
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    print(f"\nResults: {result_dir}")


if __name__ == "__main__":
    main()
