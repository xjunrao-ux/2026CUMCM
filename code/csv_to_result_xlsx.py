# -*- coding: utf-8 -*-
"""将各问题的完整结果 CSV 转换为符合题目格式要求（附件3模板）的 result1-4.xlsx。

用途
----
按 2026 CUMCM A 题附件3 的模板格式，把 results/ 下各问题输出的
``*_full_*.csv`` 详细结果转换为最终提交用的四个工作簿：

- result1.xlsx：问题1，"温度"、"水分浓度" 两个工作表，时间 1..1800 s（每 1 s），
  到药材中心距离 0..2 cm（每 0.1 cm，共 21 列）；
- result2.xlsx：问题2，同 result1 的表结构，时间 1..10800 s（每 1 s）；
- result3.xlsx：问题3，单个 "Sheet1" 工作表，时间 60 s 起每隔 60 s，
  末行附加烘干结束时刻（每 60 s 间隔之外允许一个精确结束时刻行）；
- result4.xlsx：问题4，单个 "Sheet1" 工作表，时间同 result3，
  列 = 距离 0..2 cm（每 0.1 cm）+ 末列 "药材表面"；
  超出当前表面半径的距离列留空（药材收缩后该空间位置已不存在）。

所有数值统一保留四位小数（写入时四舍五入），并设置单元格数字格式
"0.0000"；时间列（s）为整数，格式 "0"。

输入（相对项目根目录）
----------------------
- results/A_problem1_coupled/{temperature,moisture}_full_1s_0p1cm.csv
- results/A_problem2_coupled/{temperature,moisture}_full_1s_0p1cm.csv
- results/A_problem3_drying_time/moisture_full_60s_0p1cm.csv
- results/A_problem4_shrinkage/moisture_full_60s_0p1cm.csv

输出
----
- results/submission/result1.xlsx ... result4.xlsx（目录不存在时自动创建）

运行方式
--------
    python code/csv_to_result_xlsx.py                # 默认：严格按模板，从 t=1 s 起
    python code/csv_to_result_xlsx.py --include-t0   # result1/2 额外保留 t=0 初始行
    python code/csv_to_result_xlsx.py --outdir results/xxx

关键参数
--------
- CONFIGS：每题源 CSV、时间步长、工作表布局的唯一配置来源；
- INCLUDE_T0 / --include-t0：是否在 result1/result2 中保留 t=0 初始状态行
  （附件3 模板首行数据为 t=1 s，默认不保留）。

依赖
----
openpyxl（pip install openpyxl）。仅标准库 + openpyxl，无其他依赖。
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter

# ---------------------------------------------------------------------------
# 配置：每题对应的源 CSV、时间步长与工作表布局
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 距离网格：0..2 cm 每 0.1 cm，共 21 列（与附件3模板第 1 行一致）
DIST_CM = [round(i / 10, 1) for i in range(21)]

# result1/result2 的工作表定义：工作表名 -> CSV 文件名
TEMP_SHEET = "温度"
MOIST_SHEET = "水分浓度"

CONFIGS = {
    "result1.xlsx": {
        "sheets": [
            (TEMP_SHEET, "results/A_problem1_coupled/temperature_full_1s_0p1cm.csv"),
            (MOIST_SHEET, "results/A_problem1_coupled/moisture_full_1s_0p1cm.csv"),
        ],
        "time_step": 1,        # 时间步长（s）
        "first_time": 0,       # 源 CSV 起始时刻（s），模板要求从 1 s 起输出
        "last_time": 1800,     # 源 CSV 结束时刻（s）
        "with_surface_col": False,
    },
    "result2.xlsx": {
        "sheets": [
            (TEMP_SHEET, "results/A_problem2_coupled/temperature_full_1s_0p1cm.csv"),
            (MOIST_SHEET, "results/A_problem2_coupled/moisture_full_1s_0p1cm.csv"),
        ],
        "time_step": 1,
        "first_time": 0,
        "last_time": 10800,
        "with_surface_col": False,
    },
    "result3.xlsx": {
        "sheets": [
            ("Sheet1", "results/A_problem3_drying_time/moisture_full_60s_0p1cm.csv"),
        ],
        "time_step": 60,
        "first_time": 60,
        "last_time": None,     # 不校验固定结束时刻：允许末行附加烘干结束时刻
        "with_surface_col": False,
    },
    "result4.xlsx": {
        "sheets": [
            ("Sheet1", "results/A_problem4_shrinkage/moisture_full_60s_0p1cm.csv"),
        ],
        "time_step": 60,
        "first_time": 60,
        "last_time": None,
        "with_surface_col": True,   # 首行末列 "药材表面"，CSV 含 radius_cm 与 surface 列
    },
}

# 单元格数字格式：时间列整数，数据列保留四位小数
TIME_NUMFMT = "0"
VALUE_NUMFMT = "0.0000"
# 列宽：时间列 24，其余 11
TIME_COL_WIDTH = 24
VALUE_COL_WIDTH = 11

EPS = 1e-9  # 浮点比较容差


# ---------------------------------------------------------------------------
# CSV 读取与校验
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> list[list[str]]:
    """读取 UTF-8（含 BOM）CSV，返回含表头在内的全部行。"""
    if not path.is_file():
        raise FileNotFoundError(f"找不到输入文件: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))
    if len(rows) < 2:
        raise ValueError(f"{path} 为空或缺少数据行")
    return rows


def validate_grid_csv(rows: list[list[str]], path: Path, time_step: int,
                      first_time: int, last_time: int | None) -> None:
    """校验普通（问题1-3）完整结果 CSV：列数、表头、时间序列与数值有限性。

    - 列数必须为 22（时间 + 21 个距离列）；
    - 时间列严格按 time_step 递增，末行允许为附加的结束时刻（与前一行的
      间隔不必等于 time_step，但必须大于 0）；
    - 所有数据单元格必须可解析为有限浮点数。
    """
    n_expected = 1 + len(DIST_CM)
    if len(rows[0]) != n_expected:
        raise ValueError(
            f"{path}: 期望 {n_expected} 列，实际 {len(rows[0])} 列")
    if rows[0][0] != "time_s":
        raise ValueError(f"{path}: 表头首列应为 time_s，实际 {rows[0][0]!r}")
    times: list[int] = []
    for i, row in enumerate(rows[1:], start=1):
        if len(row) != n_expected:
            raise ValueError(f"{path}: 第 {i + 1} 行列数 {len(row)} != {n_expected}")
        times.append(int(row[0]))
        for cell in row[1:]:
            try:
                value = float(cell)
            except ValueError:
                raise ValueError(f"{path}: 第 {i + 1} 行存在非数值单元格 {cell!r}") from None
            if not math.isfinite(value):
                raise ValueError(f"{path}: 第 {i + 1} 行存在非有限值 {cell!r}")
    if times[0] != first_time:
        raise ValueError(f"{path}: 首行时刻 {times[0]} != 期望 {first_time}")
    for a, b in zip(times, times[1:]):
        if b - a != time_step:
            # 仅允许末行为附加的烘干结束时刻
            if b != times[-1] or b - a <= 0:
                raise ValueError(
                    f"{path}: 时间序列在 {a} -> {b} 处不符合步长 {time_step} s")
    if last_time is not None and times[-2] + time_step != last_time:
        # 当末行是附加结束时刻时，倒数第二行应恰为固定步长序列的终点
        raise ValueError(
            f"{path}: 倒数第二行时刻 {times[-2]} 与期望 {last_time} 不符")
    if last_time is not None and times[-1] != last_time:
        raise ValueError(f"{path}: 末行时刻 {times[-1]} != 期望 {last_time}")


def validate_shrinkage_csv(rows: list[list[str]], path: Path,
                           time_step: int, first_time: int) -> None:
    """校验问题4收缩模型完整结果 CSV。

    列布局：time_s, radius_cm, r_0.0_cm .. r_2.0_cm（21 列）, surface，
    共 24 列。要求：
    - 时间序列同 validate_grid_csv（末行允许附加结束时刻）；
    - radius 非增（收缩）、且与时间同向一致；
    - 距离列中 r > radius(t) 的格必须为空（收缩后该位置在药材之外），
      r <= radius(t) 的格必须为有限数值；
    - surface 列每行必须有有限数值。
    """
    n_expected = 2 + len(DIST_CM) + 1
    if len(rows[0]) != n_expected:
        raise ValueError(
            f"{path}: 期望 {n_expected} 列，实际 {len(rows[0])} 列")
    if rows[0][0] != "time_s" or rows[0][-1] != "surface":
        raise ValueError(f"{path}: 表头首列应为 time_s、末列应为 surface")
    times: list[int] = []
    prev_radius: float | None = None
    for i, row in enumerate(rows[1:], start=1):
        if len(row) != n_expected:
            raise ValueError(f"{path}: 第 {i + 1} 行列数 {len(row)} != {n_expected}")
        times.append(int(row[0]))
        radius = float(row[1])
        if not math.isfinite(radius):
            raise ValueError(f"{path}: 第 {i + 1} 行 radius 非有限值")
        if prev_radius is not None and radius > prev_radius + EPS:
            raise ValueError(
                f"{path}: 第 {i + 1} 行 radius={radius} 大于上一行 {prev_radius}（应收缩）")
        prev_radius = radius
        for dist, cell in zip(DIST_CM, row[2:-1]):
            if cell == "":
                if dist <= radius + EPS:
                    raise ValueError(
                        f"{path}: 第 {i + 1} 行距离 {dist} cm 位于表面内却为空")
            else:
                value = float(cell)
                if not math.isfinite(value):
                    raise ValueError(f"{path}: 第 {i + 1} 行存在非有限值 {cell!r}")
                if dist > radius + EPS:
                    raise ValueError(
                        f"{path}: 第 {i + 1} 行距离 {dist} cm 超出表面 {radius} cm 却非空")
        if row[-1] == "":
            raise ValueError(f"{path}: 第 {i + 1} 行 surface 列为空")
        surface = float(row[-1])
        if not math.isfinite(surface):
            raise ValueError(f"{path}: 第 {i + 1} 行 surface 非有限值")
    if times[0] != first_time:
        raise ValueError(f"{path}: 首行时刻 {times[0]} != 期望 {first_time}")
    for a, b in zip(times, times[1:]):
        if b - a != time_step:
            if b is not times[-1] or b - a <= 0:
                raise ValueError(
                    f"{path}: 时间序列在 {a} -> {b} 处不符合步长 {time_step} s")


# ---------------------------------------------------------------------------
# 工作簿写出
# ---------------------------------------------------------------------------

def write_sheet(ws, rows: list[list[str]], with_surface_col: bool) -> None:
    """把 CSV 行（含表头）写入工作表，并套用模板格式。

    - 第 1 行：A1 = "时间\\到药材中心的距离"，随后为 0..2 cm 距离表头，
      问题4 末列再加 "药材表面"；
    - 时间列写整数，数据列 round(value, 4) 并设格式 "0.0000"；
    - 空字符串 -> 空白单元格；
    - 表头加粗居中，冻结首行首列，设定列宽。
    """
    header = ["时间\\到药材中心的距离", *DIST_CM]
    if with_surface_col:
        header.append("药材表面")
    ws.append(header)

    for row in rows[1:]:
        # 问题4的 CSV 第二列为 radius_cm（收缩半径），模板无此列，需跳过
        data_cells = row[2:-1] if with_surface_col else row[1:]
        values = [int(row[0])]
        for cell in data_cells:
            values.append(round(float(cell), 4) if cell != "" else None)
        if with_surface_col:
            values.append(round(float(row[-1]), 4))
        ws.append(values)

    n_cols = len(header)
    # 表头样式：加粗、居中
    for col in range(1, n_cols + 1):
        cell = ws.cell(row=1, column=col)
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal="center", vertical="center")
    # 数字格式与对齐
    ws.cell(row=1, column=1).number_format = "@"
    for col in range(2, n_cols + 1):
        for row_idx in range(2, ws.max_row + 1):
            cell = ws.cell(row=row_idx, column=col)
            cell.number_format = VALUE_NUMFMT
            cell.alignment = Alignment(horizontal="right")
    for row_idx in range(2, ws.max_row + 1):
        cell = ws.cell(row=row_idx, column=1)
        cell.number_format = TIME_NUMFMT
        cell.alignment = Alignment(horizontal="right")
    # 列宽与冻结
    ws.column_dimensions["A"].width = TIME_COL_WIDTH
    for col in range(2, n_cols + 1):
        ws.column_dimensions[get_column_letter(col)].width = VALUE_COL_WIDTH
    ws.freeze_panes = "B2"


def build_workbook(cfg: dict, include_t0: bool) -> tuple[Workbook, dict]:
    """按配置读取并校验源 CSV，返回 (工作簿, 摘要信息)。"""
    wb = Workbook()
    summary: dict = {}
    for idx, (sheet_name, rel_csv) in enumerate(cfg["sheets"]):
        path = PROJECT_ROOT / rel_csv
        rows = load_csv(path)
        time_step = cfg["time_step"]
        if cfg["with_surface_col"]:
            validate_shrinkage_csv(rows, path, time_step, cfg["first_time"])
        else:
            validate_grid_csv(rows, path, time_step, cfg["first_time"], cfg["last_time"])

        # 默认严格按模板：result1/2 从 t=1 s 起（模板首行数据为 1），
        # 可选 --include-t0 保留 t=0 初始行；result3/4 源 CSV 本身从 60 s 起。
        if not include_t0 and cfg["first_time"] == 0:
            rows = [rows[0], *[r for r in rows[1:] if int(r[0]) >= 1]]

        ws = wb.active if idx == 0 else wb.create_sheet()
        ws.title = sheet_name
        write_sheet(ws, rows, cfg["with_surface_col"])
        summary[sheet_name] = {"rows": ws.max_row - 1, "cols": ws.max_column,
                               "first_time": ws.cell(row=2, column=1).value,
                               "last_time": ws.cell(row=ws.max_row, column=1).value}
    return wb, summary


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="将各问题完整结果 CSV 转换为符合附件3模板格式的 result1-4.xlsx")
    parser.add_argument("--outdir", default="results/submission",
                        help="输出目录（相对项目根目录），默认 results/submission")
    parser.add_argument("--include-t0", action="store_true",
                        help="result1/result2 保留 t=0 初始状态行（模板默认从 t=1 s 起）")
    args = parser.parse_args()

    outdir = PROJECT_ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    for filename, cfg in CONFIGS.items():
        wb, summary = build_workbook(cfg, include_t0=args.include_t0)
        out_path = outdir / filename
        wb.save(out_path)
        detail = "；".join(
            f"{name}: {info['rows']} 行 x {info['cols']} 列, "
            f"t={info['first_time']}..{info['last_time']} s"
            for name, info in summary.items())
        print(f"已生成 {out_path.relative_to(PROJECT_ROOT)}  [{detail}]")


if __name__ == "__main__":
    main()
