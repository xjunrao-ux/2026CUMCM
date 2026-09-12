"""A题问题4：考虑蒸发潜热模型与常规模型的对比可视化。

输入
----
results/A_problem4_shrinkage/        常规模型（无潜热）完整场结果；
results/A_problem4_shrinkage_steam/  考虑蒸发潜热模型完整场结果。

输出
----
picture/A_q4_steam_comparison/ 下 4 张 600 dpi PNG：
  A_q4_table6_comparison.png       表6水分浓度对比（2×2 小倍数）；
  A_q4_temperature_comparison.png  表面/中心温度对比；
  A_q4_surface_energy_balance.png  表面能量平衡分解与全流程能量去向；
  A_q4_drying_time_comparison.png  烘干时间对比；
result/A_q4_steam_comparison/ 下 comparison_metrics.json 与
table6_comparison.csv（两模型表6数值对照与相对差）。

运行
----
python code/a4_steam_comparison_visualization.py

脚本只读取正式数值结果，不重新求解、不平滑原数据。曲线直接取自
60 s/0.1 cm 的完整场文件；表6取样点直接取自 table6_moisture.csv。

配色（在白底上通过 dataviz 六项校验的固定顺序分类色）：
  常规模型        #2a78d6 蓝（实线）
  蒸发潜热模型    #eb6834 橙（虚线）
  传导进入药材    #1baf7a 青
  参考线/边界     #767676 灰（不作为分类系列）
文字一律使用墨色，不用系列色；系列身份由线色+线型双重编码。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = [
    "Microsoft YaHei",
    "Arial",
    "SimHei",
    "DejaVu Sans",
]
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42

ROOT = Path(__file__).resolve().parent.parent
REGULAR_DIR = ROOT / "results" / "A_problem4_shrinkage"
STEAM_DIR = ROOT / "results" / "A_problem4_shrinkage_steam"
PICTURE_DIR = ROOT / "picture" / "A_q4_steam_comparison"
METRICS_DIR = ROOT / "result" / "A_q4_steam_comparison"

INITIAL_TEMPERATURE_C = 28.0
INITIAL_MOISTURE_KG_KG = 2.55
DRYING_THRESHOLD_KG_KG = 0.15

# 固定顺序分类色（白底校验通过）；参考线灰不作为分类系列。
C_REGULAR = "#2a78d6"   # 常规模型
C_STEAM = "#eb6834"     # 蒸发潜热模型 / 蒸发潜热通量
C_AQUA = "#1baf7a"      # 传导进入药材
C_REF = "#767676"       # 烘房温度等参考线
C_GRID = "#e1e0d9"
C_AXIS = "#c3c2b7"
INK = "#272727"
MUTED = "#898781"


def apply_style() -> None:
    """统一图形字体、线宽与背景。"""
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "axes.edgecolor": C_AXIS,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "grid.color": C_GRID,
            "grid.linewidth": 0.7,
        }
    )


def read_validation(tag: str) -> dict:
    path = (REGULAR_DIR if tag == "regular" else STEAM_DIR) / "validation_summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def read_field(tag: str) -> dict:
    """读取 60 s/0.1 cm 完整场，返回时间(h)、中心与表面序列（温度转℃）。"""
    data_dir = REGULAR_DIR if tag == "regular" else STEAM_DIR
    moisture = pd.read_csv(data_dir / "moisture_full_60s_0p1cm.csv")
    temperature = pd.read_csv(data_dir / "temperature_auxiliary_60s_0p1cm_K.csv")
    time_h = np.insert(moisture["time_s"].to_numpy(dtype=float) / 3600.0, 0, 0.0)
    center_moisture = np.insert(
        moisture.iloc[:, 2].to_numpy(dtype=float), 0, INITIAL_MOISTURE_KG_KG
    )
    surface_moisture = np.insert(
        moisture.iloc[:, -1].to_numpy(dtype=float), 0, INITIAL_MOISTURE_KG_KG
    )
    center_temperature = np.insert(
        temperature.iloc[:, 2].to_numpy(dtype=float) - 273.15, 0,
        INITIAL_TEMPERATURE_C,
    )
    surface_temperature = np.insert(
        temperature.iloc[:, -1].to_numpy(dtype=float) - 273.15, 0,
        INITIAL_TEMPERATURE_C,
    )
    return {
        "time_h": time_h,
        "center_moisture": center_moisture,
        "surface_moisture": surface_moisture,
        "center_temperature": center_temperature,
        "surface_temperature": surface_temperature,
    }


def read_table6(tag: str) -> dict:
    """读取表6水分浓度（时间/h 或 烘干结束时间 标记 + 各距离列）。"""
    data_dir = REGULAR_DIR if tag == "regular" else STEAM_DIR
    frame = pd.read_csv(data_dir / "table6_moisture.csv")
    time_labels = frame.iloc[:, 0].astype(str).to_list()
    # 表6列序：时间, 0 cm, 0.5 cm, 1 cm, 1.5 cm(空), 2 cm(空), 药材表面
    distances = {"0cm": 1, "0.5cm": 2, "1cm": 3, "surface": 6}
    values = {key: frame.iloc[:, idx].to_numpy(dtype=float) for key, idx in distances.items()}
    return {"time_labels": time_labels, "values": values}


def plot_series(ax, time_h, values, color, line_style, marker, label, end_label=None):
    """按统一规格绘制一条曲线：2 pt 线、末端实心圆点、可选末端数值标注。"""
    (line,) = ax.plot(
        time_h,
        values,
        color=color,
        linestyle=line_style,
        linewidth=1.9,
        solid_capstyle="round",
        label=label,
        zorder=3,
    )
    ax.plot(
        time_h[-1],
        values[-1],
        marker="o",
        markersize=5.5,
        markerfacecolor=color,
        markeredgecolor="white",
        markeredgewidth=1.0,
        zorder=4,
    )
    if end_label is not None:
        ax.annotate(
            end_label,
            xy=(time_h[-1], values[-1]),
            xytext=(7, 0),
            textcoords="offset points",
            color=INK,
            fontsize=8.5,
            va="center",
            zorder=5,
        )
    return line


def make_fig_table6(regular: dict, steam: dict, reg_v: dict, steam_v: dict) -> None:
    """图1：表6水分浓度对比（2×2 小倍数，空心取样点为表6数值）。"""
    reg_table = read_table6("regular")
    stm_table = read_table6("steam")

    def mid_series(tag: str, column: str):
        """0.5 cm / 1 cm 固定距离列的时间曲线（60 s 分辨率）。"""
        frame = REGULAR_DIR if tag == "regular" else STEAM_DIR
        moisture = pd.read_csv(frame / "moisture_full_60s_0p1cm.csv")
        time_h = np.insert(moisture["time_s"].to_numpy(dtype=float) / 3600.0, 0, 0.0)
        values = np.insert(
            moisture.iloc[:, column].to_numpy(dtype=float), 0,
            INITIAL_MOISTURE_KG_KG,
        )
        return time_h, values

    fig, axes = plt.subplots(
        2, 2, figsize=(7.4, 5.8), sharex=True, gridspec_kw={"hspace": 0.42, "wspace": 0.12}
    )
    fig.suptitle(
        "图 1  表 6 水分浓度对比：常规模型 vs 蒸发潜热模型",
        fontsize=11, color=INK, y=0.965,
    )
    drying_h = {
        "regular": reg_v["drying_time"]["production_h"],
        "steam": steam_v["drying_time"]["production_h"],
    }

    for row, (col_key, table_key, title, ylim, mid_col) in enumerate(
        [
            ("center_moisture", "0cm", "距中心 0 cm（药材中心）", (0, 2.7), None),
            (None, "0.5cm", "距中心 0.5 cm", (0, 2.7), 7),
            (None, "1cm", "距中心 1 cm", (0, 2.7), 12),
            ("surface_moisture", "surface", "药材表面（移动边界）", (0, 0.6), None),
        ]
    ):
        ax = axes.flat[row]
        ax.set_title(title, fontsize=9.5, color=INK, pad=4)
        ax.set_ylim(*ylim)
        ax.grid(True, which="major", linewidth=0.7)
        ax.axhline(
            DRYING_THRESHOLD_KG_KG, color=C_REF, linewidth=0.9, linestyle=(0, (5, 3)),
            zorder=2,
        )
        # 完整时间曲线（60 s 分辨率）
        for tag, color, ls in [("regular", C_REGULAR, "-"), ("steam", C_STEAM, "--")]:
            if col_key is not None:
                time_h = regular["time_h"] if tag == "regular" else steam["time_h"]
                values = regular[col_key] if tag == "regular" else steam[col_key]
            else:
                time_h, values = mid_series(tag, mid_col)
            ax.plot(time_h, values, color=color, linestyle=ls, linewidth=1.9,
                    solid_capstyle="round", zorder=3)
        # 表6取样点（每 6 h 一行 + 烘干结束时间行）
        for tag, table, color, marker in [
            ("regular", reg_table, C_REGULAR, "o"),
            ("steam", stm_table, C_STEAM, "s"),
        ]:
            values = table["values"][table_key]
            labels = table["time_labels"]
            xs, ys = [], []
            for label, value in zip(labels, values):
                xs.append(
                    drying_h[tag] if label.startswith("烘干") else float(label)
                )
                ys.append(value)
            ax.plot(
                xs, ys, linestyle="none", marker=marker, markersize=4.6,
                markerfacecolor="none", markeredgecolor=color,
                markeredgewidth=1.3, zorder=5,
            )
        if row == 0:
            ax.annotate(
                "2.33（6 h）", xy=(6, 2.3272), xytext=(1.6, 2.38),
                fontsize=8, color=INK, zorder=6,
            )
            ax.annotate(
                "1.72（6 h）", xy=(6, 1.7198), xytext=(7.6, 1.55),
                fontsize=8, color=INK, zorder=6,
            )
        if row == 1:
            ax.text(
                0.985, 0.92, "0.15 烘干阈值", transform=ax.transAxes,
                ha="right", fontsize=7.5, color=MUTED,
            )
        if row in (2, 3):
            ax.set_xlabel("时间 (h)", fontsize=9, color=INK)
        if row in (0, 2):
            ax.set_ylabel("水分浓度 (kg/kg)", fontsize=9, color=INK)
        ax.set_xticks([0, 12, 24, 36, 48, 60])

    handles = [
        plt.Line2D([], [], color=C_REGULAR, linewidth=1.9, label="常规模型（实线）"),
        plt.Line2D([], [], color=C_STEAM, linewidth=1.9, linestyle="--",
                   label="蒸发潜热模型（虚线）"),
        plt.Line2D([], [], linestyle="none", marker="o", markersize=5,
                   markerfacecolor="none", markeredgecolor=INK, label="表6取样点"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.01), handlelength=1.6, columnspacing=1.4)
    path = PICTURE_DIR / "A_q4_table6_comparison.png"
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"written {path}")


def make_fig_temperature(reg: dict, stm: dict) -> None:
    """图2：表面与中心温度对比（烘房温度作为共同参考）。"""
    energy = pd.read_csv(STEAM_DIR / "surface_energy_60s.csv")
    oven_time_h = energy["time_s"].to_numpy(dtype=float) / 3600.0
    oven_temperature = energy["oven_temperature_C"].to_numpy(dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.3), sharey=True,
                             gridspec_kw={"wspace": 0.10})
    fig.suptitle("图 2  温度对比：蒸发潜热模型出现明显蒸发冷却", fontsize=11,
                 color=INK, y=1.02)
    titles = ["(a) 药材表面温度", "(b) 药材中心温度"]
    for ax, key, title in zip(axes, ["surface_temperature", "center_temperature"],
                              titles):
        ax.set_title(title, fontsize=9.5, color=INK, pad=4)
        ax.plot(oven_time_h, oven_temperature, color=C_REF, linewidth=1.2,
                linestyle=":", solid_capstyle="round", zorder=2)
        plot_series(ax, reg["time_h"], reg[key], C_REGULAR, "-", "o", None)
        plot_series(ax, stm["time_h"], stm[key], C_STEAM, "--", "s", None)
        ax.set_xlim(0, 60)
        ax.set_ylim(0, 60)
        ax.grid(True, linewidth=0.7)
        ax.set_xlabel("时间 (h)", fontsize=9, color=INK)
        ax.set_xticks([0, 12, 24, 36, 48, 60])

    axes[0].set_ylabel("温度 (℃)", fontsize=9, color=INK)
    axes[0].annotate("烘房温度（边界）", xy=(30, 50.0), xytext=(30, 44.2),
                     fontsize=8, color=MUTED, ha="center")
    axes[0].annotate(
        "表面最低 10.0 ℃\n（约 11 min）",
        xy=(0.18, 10.02), xytext=(9.5, 7.5),
        fontsize=8, color=INK,
        arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8,
                        shrinkA=2, shrinkB=2),
    )
    axes[0].annotate(
        "常规模型表面约 4 h 即达 50 ℃",
        xy=(4, 50.0), xytext=(14.5, 54.5),
        fontsize=8, color=INK,
        arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8,
                        shrinkA=2, shrinkB=2),
    )
    axes[1].annotate(
        "中心最低 13.4 ℃\n（约 0.65 h）",
        xy=(0.65, 13.43), xytext=(11.5, 5.5),
        fontsize=8, color=INK,
        arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8,
                        shrinkA=2, shrinkB=2),
    )
    handles = [
        plt.Line2D([], [], color=C_REGULAR, linewidth=1.9, label="常规模型"),
        plt.Line2D([], [], color=C_STEAM, linewidth=1.9, linestyle="--",
                   label="蒸发潜热模型"),
        plt.Line2D([], [], color=C_REF, linewidth=1.2, linestyle=":",
                   label="烘房温度"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.045), handlelength=1.6, columnspacing=1.4)
    path = PICTURE_DIR / "A_q4_temperature_comparison.png"
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"written {path}")


def make_fig_energy(steam_v: dict) -> None:
    """图3：表面能量平衡分解与全流程能量去向。"""
    energy = pd.read_csv(STEAM_DIR / "surface_energy_60s.csv")
    time_h = energy["time_s"].to_numpy(dtype=float) / 3600.0
    latent = energy["latent_heat_flux_W_m2"].to_numpy(dtype=float)
    convective = energy["convective_heat_flux_W_m2"].to_numpy(dtype=float)
    conductive = energy["conductive_heat_flux_W_m2"].to_numpy(dtype=float)
    cooling = conductive < 0.0
    cooling_span = (time_h[cooling].min(), time_h[cooling].max())

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4), gridspec_kw={"wspace": 0.32})
    fig.suptitle(
        "图 3  蒸发潜热模型的表面能量平衡（表面处 h_T(T_∞−T_s) = k·∂T/∂r + L_v·j_w）",
        fontsize=10.5, color=INK, y=1.03,
    )
    ax = axes[0]
    ax.set_title("(a) 表面热流密度随时间的演化", fontsize=9.5, color=INK, pad=4)
    ax.axvspan(cooling_span[0], cooling_span[1], color=C_STEAM, alpha=0.07, zorder=1)
    ax.plot(time_h, convective, color=C_REGULAR, linewidth=1.9, zorder=3)
    ax.plot(time_h, latent, color=C_STEAM, linewidth=1.9, zorder=3)
    ax.plot(time_h, np.abs(conductive), color=C_AQUA, linewidth=1.9, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(1.0, 2500.0)
    ax.set_xlim(0, 60)
    ax.set_yticks([1, 10, 100, 1000])
    ax.grid(True, linewidth=0.7)
    ax.set_xlabel("时间 (h)", fontsize=9, color=INK)
    ax.set_ylabel("热流密度 (W/m²)", fontsize=9, color=INK)
    ax.annotate(
        "冷却期 0.017–0.467 h：潜热需求超过对流传入，\n药材内部显热向外供出（传导为负）",
        xy=(0.24, 700), xytext=(3.1, 780), fontsize=7.5, color=INK,
    )
    ax.annotate("1218 W/m²（60 s，潜热峰值）", xy=(1 / 60, 1218.3),
                xytext=(2.4, 1350), fontsize=7.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8,
                                shrinkA=2, shrinkB=2))
    ax.annotate("267 W/m²（60 s，对流）", xy=(1 / 60, 267.4),
                xytext=(5.2, 208), fontsize=7.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8,
                                shrinkA=2, shrinkB=2))
    ax.annotate("951 W/m²（60 s，传导）", xy=(1 / 60, 950.9),
                xytext=(5.2, 700), fontsize=7.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8,
                                shrinkA=2, shrinkB=2))
    handles = [
        plt.Line2D([], [], color=C_REGULAR, linewidth=1.9, label="对流传入 h_T(T_∞−T_s)"),
        plt.Line2D([], [], color=C_STEAM, linewidth=1.9, label="蒸发潜热 L_v·j_w（|·|）"),
        plt.Line2D([], [], color=C_AQUA, linewidth=1.9, label="传导进入药材 k·∂T/∂r（|·|）"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=7.5,
              borderaxespad=0.2)

    ax = axes[1]
    ax.set_title("(b) 全流程对流传入热量的去向", fontsize=9.5, color=INK, pad=4)
    balance = steam_v["energy_balance_per_unit_length"]
    total = balance["integrated_convective_input_J_m"]
    latent_share = balance["integrated_latent_heat_J_m"] / total * 100.0
    sensible_share = balance["integrated_conductive_heat_into_solid_J_m"] / total * 100.0
    ax.barh(
        [0], [latent_share], color=C_STEAM, height=0.52,
        label="蒸发潜热",
    )
    ax.barh(
        [0], [sensible_share], left=[latent_share], color=C_AQUA, height=0.52,
        label="药材显热增温",
    )
    ax.text(
        latent_share / 2, 0, f"蒸发潜热 {latent_share:.2f} %", ha="center",
        va="center", color="white", fontsize=8.5, fontweight="bold",
    )
    ax.annotate(
        f"显热增温 {sensible_share:.2f} %（1.35×$10^4$ J/m）",
        xy=(latent_share + sensible_share, 0), xytext=(63, 0.34),
        fontsize=8, color=INK, va="center",
        arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8,
                        shrinkA=2, shrinkB=2),
    )
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.62, 0.85)
    ax.set_yticks([])
    ax.set_xlabel("占对流传入热量的比例 (%)", fontsize=9, color=INK)
    ax.grid(True, axis="x", linewidth=0.7)
    ax.text(
        0.5, -0.56,
        "总对流输入 2.168×$10^6$ J/m，能量守恒残差相对值 < 6×$10^{-11}$",
        transform=ax.transAxes, ha="center", fontsize=7.5, color=MUTED,
    )
    path = PICTURE_DIR / "A_q4_surface_energy_balance.png"
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"written {path}")


def make_fig_drying_time(reg_v: dict, steam_v: dict) -> None:
    """图4：烘干时间对比（时长条 + 差值标注）。"""
    regular_h = reg_v["drying_time"]["production_h"]
    steam_h = steam_v["drying_time"]["production_h"]
    difference_h = steam_v["comparison_with_original_no_latent_heat_model"]["difference_h"]
    relative_pct = steam_v["comparison_with_original_no_latent_heat_model"]["relative_change_pct"]

    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    fig.suptitle("图 4  烘干时间对比（干燥判据：全域 C ≤ 0.15 kg/kg）",
                 fontsize=11, color=INK, y=1.0)
    ax.barh([1], [regular_h], color=C_REGULAR, height=0.52, zorder=3)
    ax.barh([0], [steam_h], color=C_STEAM, height=0.52, zorder=3)
    ax.axvline(4.0, color=C_REF, linewidth=0.9, linestyle=(0, (5, 3)), zorder=2)
    ax.annotate("附件1环境结束（4 h）", xy=(4.0, 1.62), xytext=(4.8, 1.66),
                fontsize=7.5, color=MUTED)
    ax.text(
        regular_h + 0.7, 1, f"{regular_h:.2f} h", va="center", fontsize=12,
        fontweight="bold", color=INK,
    )
    ax.text(
        steam_h + 0.7, 0, f"{steam_h:.2f} h", va="center", fontsize=12,
        fontweight="bold", color=INK,
    )
    ax.annotate(
        "", xy=(regular_h, 1.66), xytext=(steam_h, 1.66),
        arrowprops=dict(arrowstyle="<->", color=INK, linewidth=0.9),
    )
    ax.text(
        (regular_h + steam_h) / 2, 1.82, f"+{difference_h:.2f} h（+{relative_pct:.2f} %）",
        ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=INK,
    )
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["蒸发潜热模型", "常规模型"], fontsize=9.5, color=INK)
    ax.set_xlim(0, 64)
    ax.set_ylim(-0.66, 2.15)
    ax.set_xlabel("烘干时长 (h)", fontsize=9, color=INK)
    ax.set_xticks([0, 12, 24, 36, 48, 60])
    ax.grid(True, axis="x", linewidth=0.7)
    ax.text(
        0.5, -0.30,
        "两模型采用相同的附件2半径收缩（2.0 → 1.2 cm，线收缩 40.0%），差异仅来自表面蒸发潜热项",
        transform=ax.transAxes, ha="center", fontsize=7.5, color=MUTED,
    )
    path = PICTURE_DIR / "A_q4_drying_time_comparison.png"
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"written {path}")


def write_metrics(reg_v: dict, steam_v: dict) -> None:
    """汇总对比指标与表6数值对照。"""
    reg = read_field("regular")
    stm = read_field("steam")
    reg_table = read_table6("regular")
    stm_table = read_table6("steam")

    rows = []
    reg_drying = reg_v["drying_time"]["production_h"]
    steam_drying = steam_v["drying_time"]["production_h"]
    for label_reg, label_stm in zip(reg_table["time_labels"], stm_table["time_labels"]):
        time_value = steam_drying if label_stm.startswith("烘干") else float(label_stm)
        for key, name in [("0cm", "0 cm"), ("0.5cm", "0.5 cm"), ("1cm", "1 cm"),
                          ("surface", "表面")]:
            rv = reg_table["values"][key][reg_table["time_labels"].index(label_reg)]
            sv = stm_table["values"][key][stm_table["time_labels"].index(label_stm)]
            rows.append({
                "time": time_value,
                "row": label_stm,
                "location": name,
                "regular": rv,
                "steam": sv,
                "difference": sv - rv,
                "relative_pct": (sv / rv - 1.0) * 100.0 if rv else None,
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(METRICS_DIR / "table6_comparison.csv", index=False,
                 encoding="utf-8-sig")

    metrics = {
        "drying_time": {
            "regular_h": reg_v["drying_time"]["production_h"],
            "steam_h": steam_v["drying_time"]["production_h"],
            "difference_h": steam_v["comparison_with_original_no_latent_heat_model"]["difference_h"],
            "relative_pct": steam_v["comparison_with_original_no_latent_heat_model"]["relative_change_pct"],
        },
        "surface_temperature_c": {
            "regular_min": float(reg["surface_temperature"].min()),
            "regular_max": float(reg["surface_temperature"].max()),
            "steam_min": float(stm["surface_temperature"].min()),
            "steam_min_time_h": float(stm["time_h"][stm["surface_temperature"].argmin()]),
            "steam_max": float(stm["surface_temperature"].max()),
        },
        "center_temperature_c": {
            "steam_min": float(stm["center_temperature"].min()),
            "steam_min_time_h": float(stm["time_h"][stm["center_temperature"].argmin()]),
        },
        "energy_balance_per_unit_length": steam_v["energy_balance_per_unit_length"],
        "latent_heat_flux": {
            "max_w_m2": steam_v["physical_ranges"]["latent_heat_flux_max_W_m2"],
            "min_w_m2": steam_v["physical_ranges"]["latent_heat_flux_min_W_m2"],
        },
        "radius_shrinkage": "2.0 -> 1.2 cm (identical for both models, Attachment 2)",
    }
    (METRICS_DIR / "comparison_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"written {METRICS_DIR / 'comparison_metrics.json'}")
    print(f"written {METRICS_DIR / 'table6_comparison.csv'}")


def main() -> None:
    apply_style()
    PICTURE_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    reg_v = read_validation("regular")
    steam_v = read_validation("steam")
    reg = read_field("regular")
    stm = read_field("steam")
    make_fig_table6(reg, stm, reg_v, steam_v)
    make_fig_temperature(reg, stm)
    make_fig_energy(steam_v)
    make_fig_drying_time(reg_v, steam_v)
    write_metrics(reg_v, steam_v)


if __name__ == "__main__":
    main()
