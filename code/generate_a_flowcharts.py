"""Generate first-version draw.io flowcharts and PNG previews for 2026 CUMCM A.

The draw.io files are the editable sources.  PNG previews are rendered from the
same explicit geometry so they can be reviewed even when the draw.io desktop
CLI is unavailable.
"""

from __future__ import annotations

import html
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


CANVAS_W = 1600
CANVAS_H = 1200
FONT_REGULAR = Path(r"C:\Windows\Fonts\msyh.ttc")
FONT_BOLD = Path(r"C:\Windows\Fonts\msyhbd.ttc")

GREY_FILL = "#F7F8FA"
GREY_STROKE = "#4E5969"
TEXT = "#1F2937"
ORANGE_FILL = "#FFF7EE"
ORANGE_HEAD = "#FCE6D1"
ORANGE = "#F28C28"
BLUE_FILL = "#F3F8FF"
BLUE_HEAD = "#DDEEFF"
BLUE = "#2F80ED"
PURPLE_FILL = "#F8F4FF"
PURPLE_HEAD = "#EDE3FF"
PURPLE = "#8B5CF6"
GREEN_FILL = "#F2FBF7"
GREEN_HEAD = "#DDF4E8"
GREEN = "#2F9E69"
CONTROL = "#596773"


@dataclass
class Node:
    id: str
    text: str
    x: int
    y: int
    w: int
    h: int
    kind: str = "box"
    fill: str = "#FFFFFF"
    stroke: str = GREY_STROKE
    font_size: int = 18
    bold: bool = False
    stroke_width: int = 2


@dataclass
class Edge:
    id: str
    points: list[tuple[int, int]]
    color: str = CONTROL
    width: int = 3
    arrow: bool = True
    label: str = ""
    label_pos: tuple[int, int] | None = None


@dataclass
class Container:
    id: str
    title: str
    x: int
    y: int
    w: int
    h: int
    fill: str
    header: str
    stroke: str


@dataclass
class Diagram:
    slug: str
    title: str
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    containers: list[Container] = field(default_factory=list)


def n(
    id_: str,
    text: str,
    x: int,
    y: int,
    w: int,
    h: int,
    *,
    kind: str = "box",
    fill: str = "#FFFFFF",
    stroke: str = GREY_STROKE,
    font_size: int = 18,
    bold: bool = False,
    stroke_width: int = 2,
) -> Node:
    return Node(id_, text, x, y, w, h, kind, fill, stroke, font_size, bold, stroke_width)


def e(
    id_: str,
    *points: tuple[int, int],
    color: str = CONTROL,
    width: int = 3,
    arrow: bool = True,
    label: str = "",
    label_pos: tuple[int, int] | None = None,
) -> Edge:
    return Edge(id_, list(points), color, width, arrow, label, label_pos)


def c(
    id_: str,
    title: str,
    x: int,
    y: int,
    w: int,
    h: int,
    fill: str,
    header: str,
    stroke: str,
) -> Container:
    return Container(id_, title, x, y, w, h, fill, header, stroke)


def q1_diagram() -> Diagram:
    d = Diagram("A_q1_flowchart_v1", "问题一：固定物性温度—水分场同步求解流程")
    d.nodes += [
        n("q1_title", d.title, 200, 12, 1200, 42, kind="label", font_size=26, bold=True),
        n("q1_input", "附件1环境数据、附录2物性参数\n药材尺寸与初始状态", 420, 65, 760, 65, bold=True),
        n("q1_prep", "检查时间序列与数据完整性\n对 T∞(t)、C∞(t) 作分段线性插值", 420, 150, 760, 65),
        n("q1_init", "T(r,0)=28 ℃，C(r,0)=2.55 kg/kg\n温度均匀网格；水分表面加密网格", 390, 235, 820, 72),
        n("q1_next", "进入当前时间步并更新环境边界", 500, 325, 600, 52, fill=GREY_FILL),
    ]
    d.containers += [
        c("q1_heat_group", "温度场求解", 160, 400, 600, 560, ORANGE_FILL, ORANGE_HEAD, ORANGE),
        c("q1_moist_group", "水分场求解", 840, 400, 600, 560, BLUE_FILL, BLUE_HEAD, BLUE),
    ]
    d.nodes += [
        n("q1_t_env", "读取当前 T∞(t)", 205, 465, 510, 60, stroke=ORANGE),
        n("q1_t_eq", "建立圆柱径向导热\n有限体积方程", 205, 550, 510, 65, stroke=ORANGE),
        n("q1_t_disc", "后向 Euler：Δt=0.25 s\n均匀网格：Δr=0.025 cm", 205, 640, 510, 68, stroke=ORANGE),
        n("q1_t_solve", "Thomas 算法求解\n三对角方程组", 205, 735, 510, 65, stroke=ORANGE),
        n("q1_t_out", "得到 T^(n+1)", 205, 830, 510, 60, fill="#FFFDF9", stroke=ORANGE, bold=True),
        n("q1_c_env", "读取当前 C∞(t)\n以 Cⁿ 作为 Picard 初值", 885, 455, 510, 68, stroke=BLUE),
        n("q1_c_coef", "计算 D(C)\n并更新界面传质系数", 885, 542, 510, 65, stroke=BLUE),
        n("q1_c_eq", "建立非线性 Fick 扩散方程\n表面内外阻力串联", 885, 625, 510, 68, stroke=BLUE),
        n("q1_c_solve", "后向 Euler：Δt=0.125 s\nThomas 算法求解", 885, 710, 510, 68, stroke=BLUE),
        n("q1_c_conv", "max|C^(m+1)−C^m| < 10^(-10)？", 875, 800, 530, 82, kind="decision", fill="#FFFFFF", stroke=BLUE, font_size=17),
        n("q1_c_out", "得到 C^(n+1) 并重构表面值", 885, 895, 510, 54, fill="#FAFDFF", stroke=BLUE, bold=True),
        n("q1_end", "记录整数秒结果；t = 1800 s？", 550, 980, 500, 88, kind="decision", fill="#FFFFFF"),
        n("q1_output", "插值到 0.1 cm 输出网格，提取表1、表2\n生成 result1.xlsx，并进行网格、守恒与物理合理性检查", 380, 1090, 840, 78, fill=GREY_FILL, bold=True),
    ]
    d.edges += [
        e("q1_e1", (800, 130), (800, 149)),
        e("q1_e2", (800, 215), (800, 234)),
        e("q1_e3", (800, 307), (800, 324)),
        e("q1_split_l", (799, 377), (799, 390), (460, 390), (460, 464), color=ORANGE),
        e("q1_split_r", (801, 377), (801, 390), (1140, 390), (1140, 454), color=BLUE),
        e("q1_t1", (460, 525), (460, 549), color=ORANGE),
        e("q1_t2", (460, 615), (460, 639), color=ORANGE),
        e("q1_t3", (460, 708), (460, 734), color=ORANGE),
        e("q1_t4", (460, 800), (460, 829), color=ORANGE),
        e("q1_c1", (1140, 523), (1140, 541), color=BLUE),
        e("q1_c2", (1140, 607), (1140, 624), color=BLUE),
        e("q1_c3", (1140, 693), (1140, 709), color=BLUE),
        e("q1_c4", (1140, 778), (1140, 799), color=BLUE),
        e("q1_picard_no", (1406, 841), (1490, 841), (1490, 574), (1396, 574), color=BLUE, label="否", label_pos=(1448, 823)),
        e("q1_picard_yes", (1140, 882), (1140, 894), color=BLUE, label="是", label_pos=(1156, 884)),
        e("q1_merge_l", (460, 890), (460, 960), (710, 960), (710, 979), color=ORANGE),
        e("q1_merge_r", (1140, 949), (1140, 960), (890, 960), (890, 979), color=BLUE),
        e("q1_outer_no", (549, 1024), (80, 1024), (80, 350), (499, 350), label="否", label_pos=(100, 995)),
        e("q1_outer_yes", (800, 1068), (800, 1089), label="是", label_pos=(818, 1070)),
    ]
    return d


def q2_diagram() -> Diagram:
    d = Diagram("A_q2_flowchart_v1", "问题二：变物性水热双向耦合求解流程")
    d.nodes += [
        n("q2_title", d.title, 200, 12, 1200, 42, kind="label", font_size=26, bold=True),
        n("q2_input", "附件1环境数据、附录3经验公式\n固定半径 R=2 cm 与初始状态", 420, 65, 760, 65, bold=True),
        n("q2_prep", "检查数据并线性插值\n生成 0～3 h 的逐秒环境边界", 420, 150, 760, 65),
        n("q2_init", "建立160个向表面加密的径向控制体\n初始化 T⁰=28 ℃、C⁰=2.55 kg/kg，Δt=0.5 s", 370, 235, 860, 72),
    ]
    d.containers.append(c("q2_core", "联合 Picard 水热耦合迭代", 220, 335, 1160, 605, "#FBFCFE", "#E9EEF5", GREY_STROKE))
    d.nodes += [
        n("q2_step", "读取当前 T∞(t)、C∞(t)\n令 T^0=T^n，C^0=C^n", 500, 400, 600, 68, fill=GREY_FILL),
        n("q2_heat_prop", "由 C^m 更新\nρ(C)、cp(C)、k(C)", 300, 510, 450, 75, fill=ORANGE_FILL, stroke=ORANGE),
        n("q2_heat_solve", "组装温度有限体积方程\nThomas 求 T^(m+1)", 850, 510, 450, 75, fill=ORANGE_FILL, stroke=ORANGE),
        n("q2_moist_prop", "由 T^(m+1)、C^m\n更新 D(T,C)", 300, 640, 450, 75, fill=BLUE_FILL, stroke=BLUE),
        n("q2_moist_solve", "组装水分有限体积方程\nThomas 求 C^(m+1)", 850, 640, 450, 75, fill=BLUE_FILL, stroke=BLUE),
        n("q2_conv", "||ΔT||∞<10^(-8) 且 ||ΔC||∞<10^(-10)？", 515, 770, 570, 92, kind="decision", fill="#FFFFFF"),
        n("q2_accept", "接受 T^(n+1)、C^(n+1)\n重构中心值与真实表面值", 500, 875, 600, 62, fill="#FFFFFF", bold=True),
        n("q2_end", "每1 s保存一次；t = 10800 s？", 550, 975, 500, 88, kind="decision", fill="#FFFFFF"),
        n("q2_output", "插值到 0.1 cm 网格，生成表3、表4与 result2.xlsx\n进行综合加密、水分守恒、范围及单调性验证", 360, 1090, 880, 78, fill=GREY_FILL, bold=True),
    ]
    d.edges += [
        e("q2_e1", (800, 130), (800, 149)),
        e("q2_e2", (800, 215), (800, 234)),
        e("q2_e3", (800, 307), (800, 399)),
        e("q2_e4", (800, 468), (800, 490), (525, 490), (525, 509), color=ORANGE),
        e("q2_e5", (750, 547), (849, 547), color=ORANGE),
        e("q2_e6", (1075, 585), (1075, 610), (525, 610), (525, 639), color=BLUE),
        e("q2_e7", (750, 677), (849, 677), color=BLUE),
        e("q2_e8", (1075, 715), (1075, 745), (800, 745), (800, 769)),
        e("q2_no", (1086, 816), (1330, 816), (1330, 490), (525, 490), (525, 509), label="否", label_pos=(1240, 795)),
        e("q2_yes", (800, 862), (800, 874), label="是", label_pos=(818, 864)),
        e("q2_e9", (800, 937), (800, 974)),
        e("q2_outer", (549, 1019), (90, 1019), (90, 432), (499, 432), label="否", label_pos=(110, 990)),
        e("q2_finish", (800, 1063), (800, 1089), label="是", label_pos=(818, 1067)),
    ]
    return d


def q3_diagram() -> Diagram:
    d = Diagram("A_q3_flowchart_v1", "问题三：全烘干过程自适应推进与终止判定流程")
    d.nodes += [
        n("q3_title", d.title, 180, 12, 1240, 42, kind="label", font_size=26, bold=True),
        n("q3_input", "附件1环境数据、附录3经验公式\n沿用第二问固定半径水热耦合模型", 400, 65, 800, 65, bold=True),
        n("q3_prep", "0～4 h采用分段线性边界\n4 h后采用3～4 h环境参数时间加权平均值", 400, 150, 800, 65),
        n("q3_init", "生成表面加密有限体积网格\n初始化温度场、水分场及 Cmax=2.55 kg/kg", 400, 235, 800, 72),
    ]
    d.containers += [
        c("q3_boundary_group", "边界条件与自适应时间步", 120, 345, 580, 585, PURPLE_FILL, PURPLE_HEAD, PURPLE),
        c("q3_solver_group", "单时间步水热耦合求解", 760, 345, 720, 585, "#FBFCFE", "#E9EEF5", GREY_STROKE),
    ]
    d.nodes += [
        n("q3_time_branch", "当前时间 t ≤ 4 h？", 250, 415, 320, 82, kind="decision", fill="#FFFFFF", stroke=PURPLE),
        n("q3_measured", "是：使用附件1\n插值环境边界", 155, 535, 230, 68, fill="#FFFFFF", stroke=PURPLE),
        n("q3_plateau", "否：使用3～4 h\n平均恒定边界", 435, 535, 230, 68, fill="#FFFFFF", stroke=PURPLE),
        n("q3_dt", "选择 Δt\n前4 h：1 s；稳定期：30 s\nCmax<0.152：1 s", 205, 650, 410, 90, fill="#FFFFFF", stroke=PURPLE),
        n("q3_align", "对齐4 h切换点\n及每个60 s输出时刻", 205, 790, 410, 72, fill="#FFFFFF", stroke=PURPLE),
        n("q3_picard", "以 Tⁿ、Cⁿ 作为 Picard 初值", 830, 415, 580, 60, fill=GREY_FILL),
        n("q3_heat", "由 C 更新热物性\n隐式求解温度场 T^(m+1)", 830, 505, 580, 70, fill=ORANGE_FILL, stroke=ORANGE),
        n("q3_moist", "由 T、C 更新 D(T,C)\n隐式求解水分场 C^(m+1)", 830, 605, 580, 70, fill=BLUE_FILL, stroke=BLUE),
        n("q3_conv", "温度和水分是否同时收敛？", 865, 710, 510, 82, kind="decision", fill="#FFFFFF"),
        n("q3_reconstruct", "重构中心、内部节点与表面值\n计算全域最大水分 Cmax", 830, 825, 580, 68, fill="#FFFFFF", bold=True),
        n("q3_dry", "max C(r,t) < 0.15 kg/kg？", 550, 975, 500, 88, kind="decision", fill="#FFFFFF"),
        n("q3_output", "记录首次达标时刻；每60 s输出0.1 cm结果\n生成表5、result3.xlsx，并进行粗细计算与守恒验证", 350, 1090, 900, 78, fill=GREY_FILL, bold=True),
    ]
    d.edges += [
        e("q3_e1", (800, 130), (800, 149)),
        e("q3_e2", (800, 215), (800, 234)),
        e("q3_e3", (800, 307), (800, 325), (410, 325), (410, 414), color=PURPLE),
        e("q3_yes_bound", (330, 497), (330, 534), color=PURPLE, label="是", label_pos=(342, 506)),
        e("q3_no_bound", (490, 497), (550, 515), (550, 534), color=PURPLE, label="否", label_pos=(518, 500)),
        e("q3_merge_bound1", (270, 603), (270, 630), (410, 630), (410, 649), color=PURPLE),
        e("q3_merge_bound2", (550, 603), (550, 630), (410, 630), (410, 649), color=PURPLE),
        e("q3_dt_align", (410, 740), (410, 789), color=PURPLE),
        e("q3_to_solver", (615, 826), (730, 826), (730, 445), (829, 445)),
        e("q3_s1", (1120, 475), (1120, 504), color=ORANGE),
        e("q3_s2", (1120, 575), (1120, 604), color=BLUE),
        e("q3_s3", (1120, 675), (1120, 709)),
        e("q3_picard_no", (1376, 751), (1440, 751), (1440, 540), (1411, 540), color=CONTROL, label="否", label_pos=(1400, 732)),
        e("q3_picard_yes", (1120, 792), (1120, 824), label="是", label_pos=(1138, 798)),
        e("q3_to_dry", (1120, 893), (1120, 950), (800, 950), (800, 974)),
        e("q3_not_dry", (549, 1019), (75, 1019), (75, 456), (249, 456), label="否", label_pos=(95, 990)),
        e("q3_dry_yes", (800, 1063), (800, 1089), label="是", label_pos=(818, 1067)),
    ]
    return d


def q4_diagram() -> Diagram:
    d = Diagram("A_q4_flowchart_v1", "问题四：考虑径向收缩的移动网格水热耦合流程")
    d.nodes += [
        n("q4_title", d.title, 180, 12, 1240, 42, kind="label", font_size=26, bold=True),
        n("q4_env", "附件1：烘房温度与水分\n构造4 h前后环境边界", 250, 70, 500, 70, fill=GREY_FILL, bold=True),
        n("q4_radius", "附件2：时间—药材半径\n检查半径为正且非增", 850, 70, 500, 70, fill=GREY_FILL, bold=True),
        n("q4_env_prep", "0～4 h线性插值\n4 h后采用3～4 h平均值", 250, 165, 500, 65),
        n("q4_radius_prep", "分段线性插值得到 R(t)\nR(0)=2 cm", 850, 165, 500, 65),
        n("q4_init", "附录4物性公式；引入材料坐标 ξ=r/R(t)\n在 ξ∈[0,1] 上建立固定的表面加密网格", 380, 260, 840, 72, bold=True),
    ]
    d.containers += [
        c("q4_grid_group", "移动半径与物理网格更新", 100, 370, 520, 560, PURPLE_FILL, PURPLE_HEAD, PURPLE),
        c("q4_solver_group", "当前收缩网格上的水热耦合求解", 680, 370, 820, 560, "#FBFCFE", "#E9EEF5", GREY_STROKE),
    ]
    d.nodes += [
        n("q4_boundary", "更新环境边界\n并选择自适应时间步", 155, 445, 410, 72, fill="#FFFFFF", stroke=PURPLE),
        n("q4_r", "读取 R(t^(n+1))\n更新当前药材表面位置", 155, 555, 410, 72, fill="#FFFFFF", stroke=PURPLE),
        n("q4_grid", "映射 r=R(t^(n+1))ξ\n重算控制体体积与界面面积", 155, 665, 410, 75, fill="#FFFFFF", stroke=PURPLE),
        n("q4_resist", "更新表面扩散—对流串联阻力\n并对齐60 s输出时刻", 155, 780, 410, 75, fill="#FFFFFF", stroke=PURPLE),
        n("q4_picard", "以 Tⁿ、Cⁿ 作为 Picard 初值", 800, 435, 580, 60, fill=GREY_FILL),
        n("q4_heat", "按附录4由 C 更新 ρ、cp、k\n在动态控制体上隐式求温度", 800, 525, 580, 72, fill=ORANGE_FILL, stroke=ORANGE),
        n("q4_moist", "由 T、C 更新 D(T,C)\n在动态控制体上隐式求水分", 800, 630, 580, 72, fill=BLUE_FILL, stroke=BLUE),
        n("q4_conv", "温度和水分是否同时收敛？", 835, 735, 510, 82, kind="decision", fill="#FFFFFF"),
        n("q4_reconstruct", "重构中心值与移动表面值\n计算全域最大水分 Cmax", 800, 840, 580, 65, fill="#FFFFFF", bold=True),
        n("q4_dry", "max C(r,t) < 0.15 kg/kg？", 550, 975, 500, 88, kind="decision", fill="#FFFFFF"),
        n("q4_output", "每60 s采样固定物理位置并单列移动表面；域外位置留空\n生成表6、result4.xlsx，并验证半径、守恒、完整性与收敛性", 310, 1090, 980, 78, fill=GREY_FILL, bold=True),
    ]
    d.edges += [
        e("q4_env1", (500, 140), (500, 164)),
        e("q4_rad1", (1100, 140), (1100, 164)),
        e("q4_merge1", (500, 230), (500, 245), (800, 245), (800, 259)),
        e("q4_merge2", (1100, 230), (1100, 245), (800, 245), (800, 259)),
        e("q4_to_grid", (800, 332), (800, 350), (360, 350), (360, 444), color=PURPLE),
        e("q4_g1", (360, 517), (360, 554), color=PURPLE),
        e("q4_g2", (360, 627), (360, 664), color=PURPLE),
        e("q4_g3", (360, 740), (360, 779), color=PURPLE),
        e("q4_to_solver", (565, 817), (650, 817), (650, 465), (799, 465)),
        e("q4_s1", (1090, 495), (1090, 524), color=ORANGE),
        e("q4_s2", (1090, 597), (1090, 629), color=BLUE),
        e("q4_s3", (1090, 702), (1090, 734)),
        e("q4_picard_no", (1346, 776), (1445, 776), (1445, 561), (1381, 561), label="否", label_pos=(1395, 758)),
        e("q4_picard_yes", (1090, 817), (1090, 839), label="是", label_pos=(1108, 820)),
        e("q4_to_dry", (1090, 905), (1090, 950), (800, 950), (800, 974)),
        e("q4_not_dry", (549, 1019), (70, 1019), (70, 480), (154, 480), label="否", label_pos=(90, 990)),
        e("q4_dry_yes", (800, 1063), (800, 1089), label="是", label_pos=(818, 1067)),
    ]
    return d


def add_vertex(root: ET.Element, node: Node) -> None:
    text = html.escape(node.text, quote=True).replace("\n", "<br>")
    if node.kind == "label":
        style = (
            "text;html=1;strokeColor=none;fillColor=none;whiteSpace=wrap;"
            f"fontSize={node.font_size};fontStyle={1 if node.bold else 0};"
            f"fontColor={TEXT};fontFamily=Microsoft YaHei;align=center;verticalAlign=middle;"
        )
    elif node.kind == "decision":
        style = (
            "rhombus;whiteSpace=wrap;html=1;align=center;verticalAlign=middle;"
            f"fillColor={node.fill};strokeColor={node.stroke};strokeWidth={node.stroke_width};"
            f"fontSize={node.font_size};fontStyle={1 if node.bold else 0};"
            f"fontColor={TEXT};fontFamily=Microsoft YaHei;"
        )
    else:
        style = (
            "rounded=1;arcSize=12;whiteSpace=wrap;html=1;align=center;verticalAlign=middle;"
            f"fillColor={node.fill};strokeColor={node.stroke};strokeWidth={node.stroke_width};"
            f"fontSize={node.font_size};fontStyle={1 if node.bold else 0};"
            f"fontColor={TEXT};fontFamily=Microsoft YaHei;spacing=4;"
        )
    cell = ET.SubElement(root, "mxCell", id=node.id, value=text, style=style, vertex="1", parent="1")
    ET.SubElement(cell, "mxGeometry", x=str(node.x), y=str(node.y), width=str(node.w), height=str(node.h), **{"as": "geometry"})


def add_edge(root: ET.Element, edge: Edge) -> None:
    value = html.escape(edge.label, quote=True)
    style = (
        "edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;"
        f"strokeColor={edge.color};strokeWidth={edge.width};"
        f"endArrow={'block' if edge.arrow else 'none'};endFill=1;endSize=7;"
        "fontSize=17;fontStyle=1;fontColor=#374151;fontFamily=Microsoft YaHei;"
    )
    points = nudged_points(edge)
    cell = ET.SubElement(root, "mxCell", id=edge.id, value=value, style=style, edge="1", parent="1")
    geom = ET.SubElement(cell, "mxGeometry", relative="1", **{"as": "geometry"})
    ET.SubElement(geom, "mxPoint", x=str(points[0][0]), y=str(points[0][1]), **{"as": "sourcePoint"})
    ET.SubElement(geom, "mxPoint", x=str(points[-1][0]), y=str(points[-1][1]), **{"as": "targetPoint"})
    if len(points) > 2:
        arr = ET.SubElement(geom, "Array", **{"as": "points"})
        for x, y in points[1:-1]:
            ET.SubElement(arr, "mxPoint", x=str(x), y=str(y))


def add_container(root: ET.Element, group: Container) -> None:
    bg_style = f"rounded=1;arcSize=10;html=1;fillColor={group.fill};strokeColor=none;"
    bg = ET.SubElement(root, "mxCell", id=f"{group.id}_bg", value="", style=bg_style, vertex="1", parent="1")
    ET.SubElement(bg, "mxGeometry", x=str(group.x), y=str(group.y), width=str(group.w), height=str(group.h), **{"as": "geometry"})

    header_style = f"rounded=1;arcSize=10;html=1;fillColor={group.header};strokeColor=none;"
    head = ET.SubElement(root, "mxCell", id=f"{group.id}_header", value="", style=header_style, vertex="1", parent="1")
    ET.SubElement(head, "mxGeometry", x=str(group.x), y=str(group.y), width=str(group.w), height="52", **{"as": "geometry"})

    add_vertex(root, n(f"{group.id}_title", group.title, group.x + 20, group.y + 6, group.w - 40, 40, kind="label", font_size=22, bold=True))
    border_points = [
        ((group.x, group.y + group.h), (group.x, group.y), (group.x + group.w, group.y)),
        ((group.x + group.w, group.y), (group.x + group.w, group.y + group.h), (group.x, group.y + group.h)),
    ]
    for idx, pts in enumerate(border_points, 1):
        add_edge(root, e(f"{group.id}_border_{idx}", *pts, color=group.stroke, width=3, arrow=False))


def write_drawio(diagram: Diagram, path: Path) -> None:
    mxfile = ET.Element("mxfile", host="app.diagrams.net", agent="Codex", version="24.7.17", pages="1")
    page = ET.SubElement(mxfile, "diagram", id=diagram.slug, name="流程图")
    model = ET.SubElement(
        page,
        "mxGraphModel",
        dx=str(CANVAS_W),
        dy=str(CANVAS_H),
        grid="0",
        gridSize="10",
        guides="1",
        tooltips="1",
        connect="1",
        arrows="1",
        fold="1",
        page="1",
        pageScale="1",
        pageWidth=str(CANVAS_W),
        pageHeight=str(CANVAS_H),
        math="0",
        shadow="0",
        background="#FFFFFF",
    )
    root = ET.SubElement(model, "root")
    ET.SubElement(root, "mxCell", id="0")
    ET.SubElement(root, "mxCell", id="1", parent="0")
    for group in diagram.containers:
        add_container(root, group)
    for edge in diagram.edges:
        add_edge(root, edge)
    for node in diagram.nodes:
        add_vertex(root, node)
    tree = ET.ElementTree(mxfile)
    ET.indent(tree, space="  ")
    path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def pil_font(size: int, bold: bool, scale: int) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold and FONT_BOLD.exists() else FONT_REGULAR
    return ImageFont.truetype(str(path), size * scale)


def nudged_points(edge: Edge) -> list[tuple[int, int]]:
    """Keep arrow endpoints one pixel away from box borders."""
    points = list(edge.points)
    if not edge.arrow or len(points) < 2:
        return points

    def sign(value: int) -> int:
        return 0 if value == 0 else (1 if value > 0 else -1)

    x0, y0 = points[0]
    x1, y1 = points[1]
    points[0] = (x0 + sign(x1 - x0), y0 + sign(y1 - y0))
    xp, yp = points[-2]
    xt, yt = points[-1]
    points[-1] = (xt - sign(xt - xp), yt - sign(yt - yp))
    return points


def draw_centered_text(draw: ImageDraw.ImageDraw, node: Node, scale: int) -> None:
    font = pil_font(node.font_size, node.bold, scale)
    box = (node.x * scale, node.y * scale, (node.x + node.w) * scale, (node.y + node.h) * scale)
    spacing = 5 * scale
    bbox = draw.multiline_textbbox((0, 0), node.text, font=font, spacing=spacing, align="center")
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx = (box[0] + box[2] - tw) / 2
    ty = (box[1] + box[3] - th) / 2 - bbox[1]
    draw.multiline_text((tx, ty), node.text, font=font, fill=rgb(TEXT), spacing=spacing, align="center")


def draw_arrow(draw: ImageDraw.ImageDraw, edge: Edge, scale: int) -> None:
    pts = [(x * scale, y * scale) for x, y in nudged_points(edge)]
    draw.line(pts, fill=rgb(edge.color), width=edge.width * scale, joint="curve")
    if edge.arrow:
        x2, y2 = pts[-1]
        x1, y1 = pts[-2]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length > 0:
            ux, uy = dx / length, dy / length
            px, py = -uy, ux
            arrow_len = 13 * scale
            arrow_half = 6 * scale
            base_x, base_y = x2 - ux * arrow_len, y2 - uy * arrow_len
            draw.polygon(
                [(x2, y2), (base_x + px * arrow_half, base_y + py * arrow_half), (base_x - px * arrow_half, base_y - py * arrow_half)],
                fill=rgb(edge.color),
            )
    if edge.label and edge.label_pos:
        font = pil_font(17, True, scale)
        lx, ly = edge.label_pos[0] * scale, edge.label_pos[1] * scale
        bbox = draw.textbbox((lx, ly), edge.label, font=font)
        pad = 3 * scale
        draw.rounded_rectangle((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), radius=3 * scale, fill="white")
        draw.text((lx, ly), edge.label, font=font, fill=rgb(TEXT))


def render_png(diagram: Diagram, path: Path) -> None:
    scale = 2
    image = Image.new("RGB", (CANVAS_W * scale, CANVAS_H * scale), "white")
    draw = ImageDraw.Draw(image)

    for group in diagram.containers:
        xy = (group.x * scale, group.y * scale, (group.x + group.w) * scale, (group.y + group.h) * scale)
        draw.rounded_rectangle(xy, radius=16 * scale, fill=rgb(group.fill), outline=rgb(group.stroke), width=3 * scale)
        head_xy = (group.x * scale, group.y * scale, (group.x + group.w) * scale, (group.y + 52) * scale)
        draw.rounded_rectangle(head_xy, radius=16 * scale, fill=rgb(group.header))
        draw.rectangle((group.x * scale, (group.y + 32) * scale, (group.x + group.w) * scale, (group.y + 52) * scale), fill=rgb(group.header))

    for edge in diagram.edges:
        draw_arrow(draw, edge, scale)

    for group in diagram.containers:
        title_node = n("", group.title, group.x + 20, group.y + 6, group.w - 40, 40, kind="label", font_size=22, bold=True)
        draw_centered_text(draw, title_node, scale)

    for node in diagram.nodes:
        if node.kind == "box":
            xy = (node.x * scale, node.y * scale, (node.x + node.w) * scale, (node.y + node.h) * scale)
            draw.rounded_rectangle(xy, radius=14 * scale, fill=rgb(node.fill), outline=rgb(node.stroke), width=node.stroke_width * scale)
        elif node.kind == "decision":
            cx = (node.x + node.w / 2) * scale
            cy = (node.y + node.h / 2) * scale
            pts = [
                (cx, node.y * scale),
                ((node.x + node.w) * scale, cy),
                (cx, (node.y + node.h) * scale),
                (node.x * scale, cy),
            ]
            draw.polygon(pts, fill=rgb(node.fill), outline=rgb(node.stroke))
            draw.line(pts + [pts[0]], fill=rgb(node.stroke), width=node.stroke_width * scale, joint="curve")
        draw_centered_text(draw, node, scale)

    image = image.resize((CANVAS_W, CANVAS_H), Image.Resampling.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, dpi=(300, 300), optimize=True)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    drawio_dir = repo_root / "docs" / "flowcharts"
    png_dir = repo_root / "picture" / "A_flowcharts"
    diagrams = [q1_diagram(), q2_diagram(), q3_diagram(), q4_diagram()]
    for diagram in diagrams:
        drawio_path = drawio_dir / f"{diagram.slug}.drawio"
        png_path = png_dir / f"{diagram.slug}.png"
        write_drawio(diagram, drawio_path)
        render_png(diagram, png_path)
        print(drawio_path)
        print(png_path)


if __name__ == "__main__":
    main()
