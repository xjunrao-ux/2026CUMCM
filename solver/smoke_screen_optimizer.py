#!/usr/bin/env python3
"""2025 CUMCM A: smoke-screen deployment optimizer.

Only NumPy is required.  The program can solve Problems 1--5, write CSV files,
and fill the official result1/result2/result3 XLSX templates using only the
Python standard library.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
import xml.etree.ElementTree as ET

import numpy as np


NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
ET.register_namespace("", NS)


@dataclass
class Physics:
    missile_speed: float = 300.0
    drone_speed_min: float = 70.0
    drone_speed_max: float = 140.0
    gravity: float = 9.8
    cloud_radius: float = 10.0
    cloud_sink_speed: float = 3.0
    cloud_lifetime: float = 20.0
    min_drop_interval: float = 1.0


@dataclass
class BombPlan:
    drone: str
    number: int
    heading: float
    speed: float
    drop_time: float
    fuse: float
    missile: str = "M1"


@dataclass
class ScenarioSpec:
    drones: List[str]
    bomb_counts: Dict[str, int]
    missiles: List[str]
    dynamic_assignment: bool = False


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["_config_dir"] = str(path.resolve().parent)
    return cfg


class SmokeModel:
    def __init__(self, cfg: dict, dt: float = 0.10):
        self.cfg = cfg
        self.physics = Physics(**cfg.get("physics", {}))
        self.false_target = np.asarray(cfg.get("false_target", [0, 0, 0]), dtype=float)
        self.drones = {k: np.asarray(v, dtype=float) for k, v in cfg["drones"].items()}
        self.missiles = {k: np.asarray(v, dtype=float) for k, v in cfg["missiles"].items()}
        target = cfg["target"]
        self.target_base = np.asarray(target["base_center"], dtype=float)
        self.target_radius = float(target["radius"])
        self.target_height = float(target["height"])
        self.visibility_mode = cfg.get("visibility_mode", "cylinder")
        self.azimuth_samples = int(cfg.get("cylinder_azimuth_samples", 12))
        self.dt = float(dt)
        self.target_points = self._make_target_points()
        self.impact_times = {
            k: float(np.linalg.norm(v - self.false_target) / self.physics.missile_speed)
            for k, v in self.missiles.items()
        }
        self.horizon = max(self.impact_times.values())
        self.times = np.arange(0.0, self.horizon + self.dt * 0.5, self.dt)
        self.missile_tracks = {k: self._missile_positions(k, self.times) for k in self.missiles}

    def _make_target_points(self) -> np.ndarray:
        center = self.target_base + np.array([0.0, 0.0, self.target_height / 2.0])
        if self.visibility_mode == "center":
            return center[None, :]
        if self.visibility_mode != "cylinder":
            raise ValueError("visibility_mode must be 'center' or 'cylinder'")
        pts = [self.target_base.copy(), self.target_base + [0, 0, self.target_height], center]
        for z in (0.0, self.target_height):
            for a in np.linspace(0.0, 2.0 * math.pi, self.azimuth_samples, endpoint=False):
                pts.append(self.target_base + [self.target_radius * math.cos(a),
                                               self.target_radius * math.sin(a), z])
        return np.asarray(pts, dtype=float)

    def _missile_positions(self, name: str, times: np.ndarray) -> np.ndarray:
        p0 = self.missiles[name]
        direction = (self.false_target - p0) / np.linalg.norm(self.false_target - p0)
        return p0[None, :] + self.physics.missile_speed * times[:, None] * direction[None, :]

    @staticmethod
    def direction(heading: float) -> np.ndarray:
        return np.array([math.cos(heading), math.sin(heading), 0.0])

    def drop_point(self, p: BombPlan) -> np.ndarray:
        return self.drones[p.drone] + p.speed * self.direction(p.heading) * p.drop_time

    def burst_time(self, p: BombPlan) -> float:
        return p.drop_time + p.fuse

    def burst_point(self, p: BombPlan) -> np.ndarray:
        point = self.drones[p.drone] + p.speed * self.direction(p.heading) * self.burst_time(p)
        point[2] -= 0.5 * self.physics.gravity * p.fuse ** 2
        return point

    def valid_plan(self, p: BombPlan) -> bool:
        bp = self.burst_point(p)
        return p.drop_time >= 0 and p.fuse >= 0 and bp[2] >= 0

    def _max_sightline_distance(self, cloud: np.ndarray, missile: np.ndarray) -> np.ndarray:
        # Arrays are [time, xyz].  Evaluate distance to every sampled target ray.
        worst = np.zeros(len(cloud), dtype=float)
        for q in self.target_points:
            mq = q[None, :] - missile
            mc = cloud - missile
            denom = np.einsum("ij,ij->i", mq, mq)
            lam = np.clip(np.einsum("ij,ij->i", mc, mq) / np.maximum(denom, 1e-12), 0.0, 1.0)
            closest = missile + lam[:, None] * mq
            dist = np.linalg.norm(cloud - closest, axis=1)
            worst = np.maximum(worst, dist)
        return worst

    def effectiveness_mask(self, p: BombPlan, missile_name: str) -> np.ndarray:
        mask = np.zeros(len(self.times), dtype=bool)
        if not self.valid_plan(p):
            return mask
        tb = self.burst_time(p)
        burst = self.burst_point(p)
        active = ((self.times >= tb) &
                  (self.times <= tb + self.physics.cloud_lifetime) &
                  (self.times <= self.impact_times[missile_name]))
        if not np.any(active):
            return mask
        t = self.times[active]
        cloud = np.repeat(burst[None, :], len(t), axis=0)
        cloud[:, 2] -= self.physics.cloud_sink_speed * (t - tb)
        above_ground = cloud[:, 2] >= 0
        missile = self.missile_tracks[missile_name][active]
        distance = self._max_sightline_distance(cloud, missile)
        mask[active] = above_ground & (distance <= self.physics.cloud_radius)
        return mask

    def duration(self, mask: np.ndarray) -> float:
        # Counting samples times dt is stable for optimization; interval extraction
        # below uses half-step boundaries for reporting.
        return float(mask.sum() * self.dt)

    def intervals(self, mask: np.ndarray) -> List[Tuple[float, float]]:
        padded = np.r_[False, mask, False].astype(np.int8)
        changes = np.diff(padded)
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0] - 1
        ans = []
        for s, e in zip(starts, ends):
            left = max(0.0, self.times[s] - self.dt / 2)
            right = min(self.horizon, self.times[e] + self.dt / 2)
            ans.append((float(left), float(right)))
        return ans


def scenario_spec(number: int, cfg: dict) -> ScenarioSpec:
    available_drones = list(cfg["drones"])
    available_missiles = list(cfg["missiles"])
    if number == 2:
        return ScenarioSpec(["FY1"], {"FY1": 1}, ["M1"])
    if number == 3:
        return ScenarioSpec(["FY1"], {"FY1": 3}, ["M1"])
    if number == 4:
        names = [x for x in ["FY1", "FY2", "FY3"] if x in cfg["drones"]]
        return ScenarioSpec(names, {x: 1 for x in names}, ["M1"])
    if number == 5:
        return ScenarioSpec(available_drones, {x: 3 for x in available_drones},
                            available_missiles, dynamic_assignment=True)
    raise ValueError("Optimization scenarios are 2, 3, 4, or 5")


class PlanCodec:
    """Map a bounded vector to feasible headings, speeds and drop schedules."""

    def __init__(self, model: SmokeModel, spec: ScenarioSpec):
        self.model = model
        self.spec = spec
        self.bounds: List[Tuple[float, float]] = []
        self.layout = []
        min_gap = model.physics.min_drop_interval
        for drone in spec.drones:
            n = spec.bomb_counts[drone]
            start = len(self.bounds)
            self.bounds += [(-math.pi, math.pi),
                            (model.physics.drone_speed_min, model.physics.drone_speed_max),
                            (0.0, model.horizon)]
            # Extra gap is added to the mandatory min_gap.
            self.bounds += [(0.0, model.horizon) for _ in range(n - 1)]
            max_fuse = math.sqrt(2.0 * model.drones[drone][2] / model.physics.gravity)
            self.bounds += [(0.0, max_fuse) for _ in range(n)]
            self.layout.append((drone, n, start, min_gap))

    def decode(self, x: np.ndarray) -> Tuple[List[BombPlan], float]:
        plans: List[BombPlan] = []
        penalty = 0.0
        for drone, n, start, min_gap in self.layout:
            pos = start
            heading, speed, first = map(float, x[pos:pos + 3])
            pos += 3
            gaps = x[pos:pos + n - 1]
            pos += n - 1
            fuses = x[pos:pos + n]
            drops = [first]
            for gap in gaps:
                drops.append(drops[-1] + min_gap + float(gap))
            for k, (drop, fuse) in enumerate(zip(drops, fuses), start=1):
                plan = BombPlan(drone, k, heading, speed, float(drop), float(fuse))
                plans.append(plan)
                burst = model_time = drop + float(fuse)
                if drop > self.model.horizon:
                    penalty += 10.0 + 2.0 * (drop - self.model.horizon)
                if model_time > self.model.horizon:
                    penalty += 0.2 * (model_time - self.model.horizon)
                if self.model.burst_point(plan)[2] < 0:
                    penalty += 100.0 + abs(self.model.burst_point(plan)[2])
        return plans, penalty


def greedy_assignment(model: SmokeModel, plans: List[BombPlan], missiles: Sequence[str],
                      trials: int, rng: np.random.Generator) -> Tuple[float, Dict[int, str], Dict[int, np.ndarray]]:
    masks = {(i, m): model.effectiveness_mask(p, m)
             for i, p in enumerate(plans) for m in missiles}
    best_score = -1.0
    best_assign: Dict[int, str] = {}
    indices = np.arange(len(plans))
    for trial in range(max(1, trials)):
        order = indices if trial == 0 else rng.permutation(indices)
        unions = {m: np.zeros(len(model.times), dtype=bool) for m in missiles}
        assign = {}
        for i in order:
            gains = []
            for m in missiles:
                gain = np.count_nonzero(masks[(int(i), m)] & ~unions[m])
                gains.append(gain)
            chosen = missiles[int(np.argmax(gains))]
            assign[int(i)] = chosen
            unions[chosen] |= masks[(int(i), chosen)]
        score = sum(model.duration(v) for v in unions.values())
        if score > best_score:
            best_score, best_assign = score, assign
    assigned_masks = {i: masks[(i, best_assign[i])] for i in range(len(plans))}
    return best_score, best_assign, assigned_masks


def evaluate(model: SmokeModel, codec: PlanCodec, spec: ScenarioSpec, x: np.ndarray,
             rng: np.random.Generator, assignment_trials: int = 1):
    plans, penalty = codec.decode(x)
    if penalty >= 100.0:
        return -penalty, plans, {}, {}
    if spec.dynamic_assignment:
        score, assignment, masks = greedy_assignment(
            model, plans, spec.missiles, assignment_trials, rng)
    else:
        missile = spec.missiles[0]
        masks = {i: model.effectiveness_mask(p, missile) for i, p in enumerate(plans)}
        union = np.zeros(len(model.times), dtype=bool)
        for mask in masks.values():
            union |= mask
        score = model.duration(union)
        assignment = {i: missile for i in range(len(plans))}
    return score - penalty, plans, assignment, masks


def differential_evolution(objective, bounds: Sequence[Tuple[float, float]], seed: int,
                           generations: int, population: int) -> Tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    lo = np.asarray([b[0] for b in bounds], dtype=float)
    hi = np.asarray([b[1] for b in bounds], dtype=float)
    dim = len(bounds)
    pop_n = max(8, int(population))
    pop = rng.uniform(lo, hi, size=(pop_n, dim))
    values = np.asarray([objective(v) for v in pop])
    f, cr = 0.75, 0.85
    for gen in range(generations):
        for i in range(pop_n):
            candidates = np.delete(np.arange(pop_n), i)
            replace = len(candidates) < 3
            a, b, c = rng.choice(candidates, 3, replace=replace)
            mutant = np.clip(pop[a] + f * (pop[b] - pop[c]), lo, hi)
            cross = rng.random(dim) < cr
            cross[rng.integers(dim)] = True
            trial = np.where(cross, mutant, pop[i])
            val = objective(trial)
            if val > values[i]:
                pop[i], values[i] = trial, val
        if (gen + 1) % max(1, generations // 10) == 0:
            print(f"  generation {gen + 1:4d}/{generations}: best={values.max():.3f} s", flush=True)
    best = int(np.argmax(values))
    return pop[best].copy(), float(values[best])


def local_polish(objective, x: np.ndarray, bounds: Sequence[Tuple[float, float]], seed: int,
                 steps: int = 300) -> Tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed + 991)
    lo = np.asarray([b[0] for b in bounds])
    hi = np.asarray([b[1] for b in bounds])
    span = hi - lo
    best = x.copy()
    value = objective(best)
    for k in range(steps):
        scale = 0.08 * (1.0 - k / max(steps, 1)) + 0.002
        trial = np.clip(best + rng.normal(0.0, scale, len(best)) * span, lo, hi)
        score = objective(trial)
        if score > value:
            best, value = trial, score
    return best, float(value)


def solve_scenario(model: SmokeModel, number: int, seed: int, generations: int,
                   population: int, polish_steps: int) -> Tuple[List[BombPlan], dict]:
    spec = scenario_spec(number, model.cfg)
    codec = PlanCodec(model, spec)
    eval_rng = np.random.default_rng(seed + 12345)

    def objective(x):
        return evaluate(model, codec, spec, x, eval_rng, assignment_trials=1)[0]

    print(f"Problem {number}: {len(codec.bounds)} variables, population={population}, generations={generations}")
    best_x, _ = differential_evolution(objective, codec.bounds, seed, generations, population)
    best_x, _ = local_polish(objective, best_x, codec.bounds, seed, polish_steps)
    score, plans, assignment, masks = evaluate(
        model, codec, spec, best_x, np.random.default_rng(seed + 54321),
        assignment_trials=30 if spec.dynamic_assignment else 1)
    for i, p in enumerate(plans):
        p.missile = assignment[i]
    unions = {m: np.zeros(len(model.times), dtype=bool) for m in spec.missiles}
    for i, p in enumerate(plans):
        unions[p.missile] |= model.effectiveness_mask(p, p.missile)
    details = {
        "objective_seconds": sum(model.duration(v) for v in unions.values()),
        "missile_union_seconds": {m: model.duration(mask) for m, mask in unions.items()},
        "missile_union_intervals": {m: model.intervals(mask) for m, mask in unions.items()},
        "individual_masks": {i: model.effectiveness_mask(p, p.missile) for i, p in enumerate(plans)},
    }
    print(f"Problem {number} final union objective: {details['objective_seconds']:.3f} s")
    return plans, details


def problem1(model: SmokeModel) -> Tuple[List[BombPlan], dict]:
    # "toward the false target" means the horizontal projection toward (0, 0).
    start = model.drones["FY1"]
    delta = model.false_target[:2] - start[:2]
    heading = math.atan2(delta[1], delta[0])
    p = BombPlan("FY1", 1, heading, 120.0, 1.5, 3.6, "M1")
    mask = model.effectiveness_mask(p, "M1")
    return [p], {
        "objective_seconds": model.duration(mask),
        "missile_union_seconds": {"M1": model.duration(mask)},
        "missile_union_intervals": {"M1": model.intervals(mask)},
        "individual_masks": {0: mask},
    }


HEADERS = ["无人机编号", "无人机运动方向(度)", "无人机运动速度(m/s)", "烟幕干扰弹编号",
           "投放点x(m)", "投放点y(m)", "投放点z(m)", "起爆点x(m)", "起爆点y(m)",
           "起爆点z(m)", "有效干扰时长(s)", "干扰的导弹编号", "投放时刻(s)", "引信时长(s)",
           "起爆时刻(s)", "有效区间"]


def plan_records(model: SmokeModel, plans: List[BombPlan], details: dict) -> List[dict]:
    rows = []
    for i, p in enumerate(plans):
        drop, burst = model.drop_point(p), model.burst_point(p)
        mask = details["individual_masks"][i]
        intervals = model.intervals(mask)
        rows.append({
            HEADERS[0]: p.drone,
            HEADERS[1]: math.degrees(p.heading) % 360.0,
            HEADERS[2]: p.speed,
            HEADERS[3]: p.number,
            HEADERS[4]: drop[0], HEADERS[5]: drop[1], HEADERS[6]: drop[2],
            HEADERS[7]: burst[0], HEADERS[8]: burst[1], HEADERS[9]: burst[2],
            HEADERS[10]: model.duration(mask), HEADERS[11]: p.missile,
            HEADERS[12]: p.drop_time, HEADERS[13]: p.fuse, HEADERS[14]: model.burst_time(p),
            HEADERS[15]: "; ".join(f"[{a:.2f}, {b:.2f}]" for a, b in intervals),
        })
    return rows


def write_csv(path: Path, rows: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(rows)


def col_name(index: int) -> str:
    ans = ""
    while index:
        index, rem = divmod(index - 1, 26)
        ans = chr(65 + rem) + ans
    return ans


def set_cell(row: ET.Element, address: str, value):
    cells = list(row.findall(f"{{{NS}}}c"))
    cell = next((c for c in cells if c.get("r") == address), None)
    if cell is None:
        cell = ET.Element(f"{{{NS}}}c", {"r": address})
        row.append(cell)
    for child in list(cell):
        cell.remove(child)
    if isinstance(value, str):
        cell.set("t", "inlineStr")
        inline = ET.SubElement(cell, f"{{{NS}}}is")
        ET.SubElement(inline, f"{{{NS}}}t").text = value
    else:
        cell.attrib.pop("t", None)
        ET.SubElement(cell, f"{{{NS}}}v").text = f"{float(value):.8f}".rstrip("0").rstrip(".")


def fill_xlsx_template(template: Path, output: Path, table_rows: List[List[object]]):
    """Fill Sheet1 data rows while preserving the official workbook formatting."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(template, "r") as zin:
        sheet_name = "xl/worksheets/sheet1.xml"
        root = ET.fromstring(zin.read(sheet_name))
        sheet_data = root.find(f"{{{NS}}}sheetData")
        assert sheet_data is not None
        existing = {int(r.get("r")): r for r in sheet_data.findall(f"{{{NS}}}row")}
        for ridx, values in enumerate(table_rows, start=2):
            row = existing.get(ridx)
            if row is None:
                row = ET.SubElement(sheet_data, f"{{{NS}}}row", {"r": str(ridx)})
            for cidx, value in enumerate(values, start=1):
                set_cell(row, f"{col_name(cidx)}{ridx}", value)
        xml = ET.tostring(root, encoding="utf-8", xml_declaration=True)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx", dir=output.parent) as tmp:
            temp_path = Path(tmp.name)
        try:
            with zipfile.ZipFile(temp_path, "w", zipfile.ZIP_DEFLATED) as zout:
                for item in zin.infolist():
                    data = xml if item.filename == sheet_name else zin.read(item.filename)
                    zout.writestr(item, data)
            shutil.move(str(temp_path), output)
        finally:
            temp_path.unlink(missing_ok=True)


def template_rows(number: int, rows: List[dict]) -> List[List[object]]:
    if number == 3:
        return [[r[HEADERS[1]], r[HEADERS[2]], r[HEADERS[3]], r[HEADERS[4]], r[HEADERS[5]],
                 r[HEADERS[6]], r[HEADERS[7]], r[HEADERS[8]], r[HEADERS[9]], r[HEADERS[10]]]
                for r in rows]
    if number == 4:
        return [[r[HEADERS[0]], r[HEADERS[1]], r[HEADERS[2]], r[HEADERS[4]], r[HEADERS[5]],
                 r[HEADERS[6]], r[HEADERS[7]], r[HEADERS[8]], r[HEADERS[9]], r[HEADERS[10]]]
                for r in rows]
    if number == 5:
        return [[r[HEADERS[0]], r[HEADERS[1]], r[HEADERS[2]], r[HEADERS[3]], r[HEADERS[4]],
                 r[HEADERS[5]], r[HEADERS[6]], r[HEADERS[7]], r[HEADERS[8]], r[HEADERS[9]],
                 r[HEADERS[10]], r[HEADERS[11]]] for r in rows]
    raise ValueError("Only Problems 3--5 have official XLSX templates")


def resolve_template(cfg: dict, key: str) -> Path:
    raw = Path(cfg["templates"][key])
    return raw if raw.is_absolute() else Path(cfg["_config_dir"]) / raw


def write_summary(path: Path, number: int, model: SmokeModel, details: dict):
    payload = {
        "problem": number,
        "visibility_mode": model.visibility_mode,
        "time_step_seconds": model.dt,
        "total_union_seconds": details["objective_seconds"],
        "missile_union_seconds": details["missile_union_seconds"],
        "missile_union_intervals": details["missile_union_intervals"],
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="烟幕干扰弹投放策略优化求解器")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config_example.json"))
    parser.add_argument("--scenario", default="2", choices=["1", "2", "3", "4", "5", "all"])
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--dt", type=float, default=0.10, help="optimization/reporting time step in seconds")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--generations", type=int, default=80)
    parser.add_argument("--population", type=int, default=80)
    parser.add_argument("--polish-steps", type=int, default=300)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model = SmokeModel(cfg, args.dt)
    args.output.mkdir(parents=True, exist_ok=True)
    numbers = [1, 2, 3, 4, 5] if args.scenario == "all" else [int(args.scenario)]
    for number in numbers:
        if number == 1:
            plans, details = problem1(model)
            print(f"Problem 1 effective duration: {details['objective_seconds']:.3f} s")
        else:
            plans, details = solve_scenario(model, number, args.seed + number,
                                            args.generations, args.population, args.polish_steps)
        rows = plan_records(model, plans, details)
        write_csv(args.output / f"problem{number}_details.csv", rows)
        write_summary(args.output / f"problem{number}_summary.json", number, model, details)
        if number in (3, 4, 5):
            key = f"result{number - 2}"
            template = resolve_template(cfg, key)
            fill_xlsx_template(template, args.output / f"result{number - 2}.xlsx",
                               template_rows(number, rows))
    print(f"Results written to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
