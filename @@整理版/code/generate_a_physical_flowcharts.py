"""生成 A 题四问的黑白物理模型流程图。

用途
----
将四问的物理对象、控制方程、边界条件、耦合机制、终止判据和关键结论
组织为适合论文审阅的黑白流程图。数值算法仅作为物理模型之后的支撑层。

输入
----
脚本不读取原始附件；图中公式与数值来自已经核对的模型详稿。

输出
----
默认在 ``picture/A_flowcharts_physical_bw/`` 下生成 4 张 600 dpi PNG。

运行
----
从项目根目录执行：

    python code/generate_a_physical_flowcharts.py

也可只生成某一问：

    python code/generate_a_physical_flowcharts.py --only q4
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle


PAGE_WIDTH_MM = 180.0
PAGE_HEIGHT_MM = 250.0
DEFAULT_DPI = 600
X_MAX = 100.0
Y_MAX = 142.0

BLACK = "#111111"
MID_GREY = "#555555"
WHITE = "#FFFFFF"


def _find_chinese_font() -> Path | None:
    """按常见安装位置选择中文字体，不依赖个人用户目录。"""

    candidates = (
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
    )
    return next((path for path in candidates if path.exists()), None)


FONT_PATH = _find_chinese_font()
FONT_REGULAR = FontProperties(fname=str(FONT_PATH)) if FONT_PATH else FontProperties(family="sans-serif")
FONT_BOLD = FONT_REGULAR.copy()
FONT_BOLD.set_weight("bold")

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 7.2,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "figure.facecolor": WHITE,
        "savefig.facecolor": WHITE,
    }
)


@dataclass(frozen=True)
class Node:
    """一个流程图节点，坐标使用统一的版面逻辑坐标。"""

    key: str
    x: float
    y: float
    w: float
    h: float
    text: str
    kind: str = "box"
    fontsize: float = 7.4
    bold: bool = False
    align: str = "center"
    linewidth: float = 1.15
    linestyle: str = "-"
    double_border: bool = False

    @property
    def left(self) -> float:
        return self.x

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def bottom(self) -> float:
        return self.y

    @property
    def top(self) -> float:
        return self.y + self.h

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2


@dataclass(frozen=True)
class Edge:
    """折线箭头；最后一段带箭头。"""

    points: tuple[tuple[float, float], ...]
    label: str = ""
    label_xy: tuple[float, float] | None = None
    linewidth: float = 1.05
    dashed: bool = False


@dataclass
class FlowFigure:
    """一张流程图的全部声明式元素。"""

    slug: str
    title: str
    subtitle: str
    nodes: list[Node]
    edges: list[Edge]
    section_headers: list[tuple[float, str]]
    decorators: list[Callable[[Axes], None]] = field(default_factory=list)


def top(node: Node) -> tuple[float, float]:
    return node.cx, node.top


def bottom(node: Node) -> tuple[float, float]:
    return node.cx, node.bottom


def left(node: Node) -> tuple[float, float]:
    return node.left, node.cy


def right(node: Node) -> tuple[float, float]:
    return node.right, node.cy


def _section_header(ax: Axes, y: float, text: str) -> None:
    ax.text(
        4.0,
        y,
        text,
        ha="left",
        va="center",
        fontsize=8.3,
        fontproperties=FONT_BOLD,
        color=BLACK,
    )
    ax.plot([39.0, 96.0], [y, y], color=BLACK, lw=0.75, solid_capstyle="butt")


def _draw_node(ax: Axes, node: Node) -> tuple[object, object]:
    if node.kind == "decision":
        vertices = [
            (node.cx, node.top),
            (node.right, node.cy),
            (node.cx, node.bottom),
            (node.left, node.cy),
        ]
        patch = Polygon(
            vertices,
            closed=True,
            facecolor=WHITE,
            edgecolor=BLACK,
            linewidth=node.linewidth,
            linestyle=node.linestyle,
            joinstyle="miter",
            zorder=3,
        )
        text_width = 0.64 * node.w
        text_height = 0.52 * node.h
    else:
        patch = Rectangle(
            (node.x, node.y),
            node.w,
            node.h,
            facecolor=WHITE,
            edgecolor=BLACK,
            linewidth=node.linewidth,
            linestyle=node.linestyle,
            joinstyle="miter",
            zorder=3,
        )
        text_width = node.w - 2.0
        text_height = node.h - 1.2
    ax.add_patch(patch)

    if node.double_border and node.kind != "decision":
        inset = 0.65
        ax.add_patch(
            Rectangle(
                (node.x + inset, node.y + inset),
                node.w - 2 * inset,
                node.h - 2 * inset,
                facecolor="none",
                edgecolor=BLACK,
                linewidth=0.65,
                zorder=3.1,
            )
        )

    if node.align == "left":
        tx = node.x + 1.5
        ha = "left"
    else:
        tx = node.cx
        ha = "center"
    font = FONT_BOLD if node.bold else FONT_REGULAR
    artist = ax.text(
        tx,
        node.cy,
        node.text,
        ha=ha,
        va="center",
        multialignment=node.align,
        linespacing=1.18,
        fontsize=node.fontsize,
        fontproperties=font,
        color=BLACK,
        zorder=4,
    )
    artist._flow_allowed_width = text_width  # type: ignore[attr-defined]
    artist._flow_allowed_height = text_height  # type: ignore[attr-defined]
    artist._flow_node_key = node.key  # type: ignore[attr-defined]
    return patch, artist


def _draw_edge(ax: Axes, edge: Edge) -> None:
    if len(edge.points) < 2:
        raise ValueError("Edge 至少需要两个点。")

    style = "--" if edge.dashed else "-"
    for (x0, y0), (x1, y1) in zip(edge.points[:-2], edge.points[1:-1]):
        ax.plot(
            [x0, x1],
            [y0, y1],
            color=BLACK,
            lw=edge.linewidth,
            ls=style,
            solid_capstyle="butt",
            zorder=1,
        )

    start = edge.points[-2]
    end = edge.points[-1]
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=9.5,
        linewidth=edge.linewidth,
        linestyle=style,
        color=BLACK,
        shrinkA=0,
        shrinkB=0,
        zorder=2,
    )
    ax.add_patch(arrow)

    if edge.label and edge.label_xy:
        ax.text(
            *edge.label_xy,
            edge.label,
            ha="center",
            va="center",
            fontsize=7.0,
            fontproperties=FONT_BOLD,
            color=BLACK,
            bbox={"facecolor": WHITE, "edgecolor": "none", "pad": 0.35},
            zorder=5,
        )


def _fit_text_to_nodes(fig: Figure, ax: Axes, node_artists: Sequence[tuple[Node, object]]) -> None:
    """逐节点检查文本包围盒，必要时小幅缩小但不低于 6 pt。"""

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for node, artist in node_artists:
        allowed_w = node.w * ax.bbox.width / X_MAX
        allowed_h = node.h * ax.bbox.height / Y_MAX
        if node.kind == "decision":
            allowed_w *= 0.64
            allowed_h *= 0.52
        else:
            allowed_w -= 8.0
            allowed_h -= 6.0

        current = node.fontsize
        bbox = artist.get_window_extent(renderer=renderer)  # type: ignore[attr-defined]
        while (bbox.width > allowed_w or bbox.height > allowed_h) and current > 6.0:
            current = max(6.0, current - 0.2)
            artist.set_fontsize(current)  # type: ignore[attr-defined]
            fig.canvas.draw()
            bbox = artist.get_window_extent(renderer=renderer)  # type: ignore[attr-defined]
        if bbox.width > allowed_w + 1 or bbox.height > allowed_h + 1:
            raise RuntimeError(f"节点 {node.key} 的文本在 6 pt 时仍超出图框，请调整换行或布局。")


def _check_node_overlap(nodes: Iterable[Node]) -> None:
    """检查普通节点外框是否发生几何重叠。"""

    node_list = list(nodes)
    for i, a in enumerate(node_list):
        for b in node_list[i + 1 :]:
            overlap_x = min(a.right, b.right) - max(a.left, b.left)
            overlap_y = min(a.top, b.top) - max(a.bottom, b.bottom)
            if overlap_x > 0.05 and overlap_y > 0.05:
                raise RuntimeError(f"节点 {a.key} 与 {b.key} 的外框重叠。")


def _draw_cylinder_icon(ax: Axes, node: Node, moving: bool = False) -> None:
    """在几何节点内画一个黑白圆柱截面示意。"""

    cx = node.x + node.w * 0.34
    cy = node.cy - 0.2
    radius = min(node.w * 0.18, node.h * 0.32)
    ax.add_patch(Circle((cx, cy), radius, fill=False, edgecolor=BLACK, linewidth=1.0, zorder=5))
    ax.add_patch(Circle((cx, cy), 0.18, facecolor=BLACK, edgecolor=BLACK, zorder=5))
    arrow_end = (cx + 0.82 * radius, cy + 0.38 * radius)
    ax.add_patch(
        FancyArrowPatch(
            (cx, cy),
            arrow_end,
            arrowstyle="-|>",
            mutation_scale=7.0,
            linewidth=0.9,
            color=BLACK,
            zorder=5,
        )
    )
    ax.text(
        cx + 0.48 * radius,
        cy + 0.75 * radius,
        "R(t)" if moving else "R0",
        fontsize=6.4,
        fontproperties=FONT_REGULAR,
        ha="center",
        va="center",
        color=BLACK,
        zorder=6,
    )
    if moving:
        ax.add_patch(
            Circle(
                (cx, cy),
                radius * 1.28,
                fill=False,
                edgecolor=MID_GREY,
                linewidth=0.75,
                linestyle="--",
                zorder=4,
            )
        )
        ax.add_patch(
            FancyArrowPatch(
                (cx + radius * 1.35, cy),
                (cx + radius * 1.02, cy),
                arrowstyle="-|>",
                mutation_scale=6.8,
                linewidth=0.8,
                color=BLACK,
                zorder=5,
            )
        )


def _geometry_text(ax: Axes, node: Node, lines: str) -> None:
    ax.text(
        node.x + node.w * 0.72,
        node.cy,
        lines,
        ha="center",
        va="center",
        multialignment="center",
        linespacing=1.18,
        fontsize=7.1,
        fontproperties=FONT_REGULAR,
        color=BLACK,
        zorder=6,
    )


def _q1() -> FlowFigure:
    geom = Node("q1_geometry", 4, 114, 28, 12, "", linewidth=1.2)
    assumptions = Node(
        "q1_assumptions",
        36,
        114,
        60,
        12,
        "长圆柱近似：T=T(r,t)，C=C(r,t)\nL/(2R0)=6.25，侧/端面积比=12.5\n固定半径 R0=0.02 m；忽略轴向、周向传递",
        fontsize=7.2,
    )
    conditions = Node(
        "q1_conditions",
        8,
        97,
        84,
        12,
        "外界驱动、初始状态与中心边界\n附件1分段线性插值得 T∞(t)、C∞(t)，0≤t≤1800 s\nT(r,0)=28 ℃，C(r,0)=2.55 kg/kg；r=0：∂T/∂r=∂C/∂r=0",
        fontsize=7.0,
    )
    heat = Node(
        "q1_heat",
        4,
        70,
        44,
        19,
        "能量守恒 + Fourier 定律\nρcp ∂T/∂t = (k/r)∂/∂r(r∂T/∂r)\nρ=820 kg/m³，cp=2600 J/(kg·K)\nk=0.36 W/(m·K)，α=1.6886×10^(-7) m²/s\n表面：k∂T/∂r=h(T∞−Ts)\nBi_h=1.3889>0.1 → 内部导热热阻不可忽略",
        fontsize=6.75,
        align="left",
        linewidth=1.35,
    )
    moisture = Node(
        "q1_moisture",
        52,
        70,
        44,
        19,
        "水分守恒 + Fick 定律\n∂C/∂t = (1/r)∂/∂r[rD(C)∂C/∂r]\nD(C)=7.0×10^(-9) exp(−0.89/C) m²/s\n表面：−D∂C/∂r=km(Cs−C∞)\nkm=8.0×10^(-7) m/s\nBi_m≈3.24>0.1 → 表层先失水、中心滞后",
        fontsize=6.75,
        align="left",
        linewidth=1.35,
    )
    relation = Node(
        "q1_relation",
        18,
        58,
        64,
        7,
        "同一几何、环境与输出时刻下同步推进；方程中无 T、C 双向反馈，属于同步传递而非强耦合",
        fontsize=7.0,
        bold=True,
        linestyle="--",
    )
    numeric = Node(
        "q1_numeric",
        8,
        40,
        84,
        12,
        "数值支撑层（不改变物理机理）\n径向有限体积 + Robin 串联阻力边界 + 后向 Euler + Thomas 三对角求解\n温度 Δr=0.025 cm、Δt=0.25 s；水分 160 个表面加密控制体、Δt=0.125 s，Picard 收敛",
        fontsize=6.9,
    )
    output = Node(
        "q1_output",
        8,
        21,
        84,
        13,
        "输出与检验\n得到 0～30 min 径向 T、C 场；30 min：T中心/表面=33.5756/36.7857 ℃，\nC中心/表面=2.5500/1.5102 kg/kg；守恒、网格加密与物理单调性检验通过",
        fontsize=7.0,
    )
    conclusion = Node(
        "q1_conclusion",
        14,
        6,
        72,
        9,
        "物理结论：预热已向内部明显传播，但失水仍集中在表层；热阻与质阻均不可忽略",
        fontsize=7.4,
        bold=True,
        double_border=True,
        linewidth=1.35,
    )

    nodes = [geom, assumptions, conditions, heat, moisture, relation, numeric, output, conclusion]
    edges = [
        Edge((right(geom), left(assumptions))),
        Edge((bottom(assumptions), (66, 111), (50, 111), top(conditions))),
        Edge((bottom(conditions), (50, 92), (26, 92), top(heat))),
        Edge(((50, 92), (74, 92), top(moisture))),
        Edge((bottom(heat), (26, 67), (50, 67), top(relation))),
        Edge((bottom(moisture), (74, 67), (50, 67), top(relation))),
        Edge((bottom(relation), top(numeric))),
        Edge((bottom(numeric), top(output))),
        Edge((bottom(output), top(conclusion))),
    ]

    return FlowFigure(
        slug="A_q1_physical_flowchart_bw_v2",
        title="问题一｜固定物性下的径向水热同步传递",
        subtitle="核心：从长圆柱内的能量/水分守恒出发，解释“内部热阻与质阻为何不能忽略”",
        nodes=nodes,
        edges=edges,
        section_headers=[
            (130, "01  物理对象、初值与外界边界"),
            (93, "02  两条守恒主线：热传导与水分扩散"),
            (55, "03  数值实现、结果与物理解释"),
        ],
        decorators=[
            lambda ax: _draw_cylinder_icon(ax, geom, moving=False),
            lambda ax: _geometry_text(ax, geom, "一维径向\n0≤r≤R0"),
        ],
    )


def _q2() -> FlowFigure:
    foundation = Node(
        "q2_foundation",
        4,
        114,
        44,
        12,
        "固定半径长圆柱，0≤r≤R0\nT(r,0)=28 ℃，C(r,0)=2.55 kg/kg\n中心对称；表面采用换热/传质 Robin 边界",
        fontsize=7.0,
    )
    environment = Node(
        "q2_environment",
        52,
        114,
        44,
        12,
        "附件1环境边界分段线性插值\nT∞(t)、C∞(t)，0≤t≤3 h\nh=25 W/(m²·K)，km=8.0×10^(-7) m/s",
        fontsize=7.0,
    )
    state = Node(
        "q2_state",
        35,
        101,
        30,
        7,
        "当前迭代状态 C^(m)、T^(m)",
        fontsize=7.2,
        bold=True,
    )
    properties = Node(
        "q2_properties",
        4,
        83,
        43,
        14,
        "水分改变热物性\nρ(C)=650+128C\ncp(C)=1450+2736C/(C+1)\nk(C)=0.21+0.38C/(C+1)",
        fontsize=6.9,
        align="left",
        linewidth=1.35,
    )
    heat = Node(
        "q2_heat",
        53,
        83,
        43,
        14,
        "热量守恒决定温度场\nρ(C)cp(C)∂T/∂t\n= (1/r)∂/∂r[rk(C)∂T/∂r]\n固定热物性后隐式求 T^(m+1)",
        fontsize=6.9,
        align="left",
        linewidth=1.35,
    )
    diffusivity = Node(
        "q2_diffusivity",
        53,
        64,
        43,
        14,
        "温度和水分共同改变扩散率\nD(C,TK)=2.4×10^(-3) exp(−0.45/C)\n           × exp(−3850/TK)\nTK=T+273.15（Arrhenius 温度用 K）",
        fontsize=6.65,
        align="left",
        linewidth=1.35,
    )
    moisture = Node(
        "q2_moisture",
        4,
        64,
        43,
        14,
        "水分守恒反馈水分场\n∂C/∂t = (1/r)∂/∂r\n          [rD(C,TK)∂C/∂r]\n固定 D 后隐式求 C^(m+1)",
        fontsize=6.9,
        align="left",
        linewidth=1.35,
    )
    coupling = Node(
        "q2_coupling",
        19,
        54,
        62,
        6,
        "双向物性耦合闭环：C → (ρ,cp,k) → T；(T,C) → D → C（未计蒸发潜热）",
        fontsize=6.9,
        bold=True,
        linestyle="--",
    )
    numeric = Node(
        "q2_numeric",
        8,
        38,
        84,
        11,
        "每个时间步采用分块 Gauss–Seidel 型 Picard 迭代\n160 个表面加密控制体，Δt=0.5 s；后向 Euler + Thomas\n‖ΔT‖∞<10^(-8) 且 ‖ΔC‖∞<10^(-10) 后接受新时刻",
        fontsize=6.9,
    )
    output = Node(
        "q2_output",
        8,
        21,
        84,
        12,
        "3 h 输出与检验\nT中心/表面=49.8495/49.9664 ℃（温差仅 0.117 ℃）；\nC中心/表面=1.7662/1.0081 kg/kg；守恒、网格与耦合收敛检验通过",
        fontsize=7.0,
    )
    conclusion = Node(
        "q2_conclusion",
        13,
        6,
        74,
        9,
        "物理结论：α≈10D，温度比水分更快均匀；干燥后期逐渐转由内部水分扩散控制",
        fontsize=7.35,
        bold=True,
        double_border=True,
        linewidth=1.35,
    )

    nodes = [foundation, environment, state, properties, heat, diffusivity, moisture, coupling, numeric, output, conclusion]
    edges = [
        Edge((bottom(foundation), (26, 111), (50, 111), top(state))),
        Edge((bottom(environment), (74, 111), (50, 111), top(state))),
        Edge((bottom(state), (50, 99), (25.5, 99), top(properties)), label="C^(m)", label_xy=(30, 99)),
        Edge((right(properties), left(heat))),
        Edge((bottom(heat), top(diffusivity)), label="T^(m+1)", label_xy=(77, 80.5)),
        Edge((left(diffusivity), right(moisture))),
        Edge(((25.5, 64), (25.5, 61.5), (12, 61.5), (12, 104.5), (35, 104.5)), label="C^(m+1)", label_xy=(15, 80.5)),
        Edge((bottom(moisture), (25.5, 62), (50, 62), top(coupling))),
        Edge((bottom(diffusivity), (74.5, 62), (50, 62), top(coupling))),
        Edge((bottom(coupling), top(numeric))),
        Edge((bottom(numeric), top(output))),
        Edge((bottom(output), top(conclusion))),
    ]

    return FlowFigure(
        slug="A_q2_physical_flowchart_bw_v2",
        title="问题二｜温湿相关物性驱动的非线性水热耦合",
        subtitle="核心：水分先改变热物性，温度与水分再共同改变扩散率，形成可解释的反馈闭环",
        nodes=nodes,
        edges=edges,
        section_headers=[
            (130, "01  继承的几何、初值与 Robin 边界"),
            (111, "02  物性反馈构成的双向耦合闭环"),
            (51, "03  非线性求解、结果与限制环节"),
        ],
    )


def _q3() -> FlowFigure:
    physics = Node(
        "q3_physics",
        4,
        114,
        44,
        12,
        "固定半径 R0=0.02 m\n完整继承问题二的物性耦合方程\nρcp∂T/∂t=(1/r)∂/∂r(rk∂T/∂r)\n∂C/∂t=(1/r)∂/∂r(rD∂C/∂r)",
        fontsize=6.8,
        align="left",
    )
    boundary = Node(
        "q3_boundary",
        52,
        114,
        44,
        12,
        "长期环境边界\n0～4 h：附件1分段线性插值\nt>4 h：T∞=49.9959 ℃，C∞=0.049990\n（取 3～4 h 时间加权平均）",
        fontsize=6.8,
        align="left",
    )
    step = Node(
        "q3_step",
        15,
        96,
        70,
        8,
        "从 t=0 重新推进完整干燥过程，并读取当前 T∞(t)、C∞(t)",
        fontsize=7.15,
        bold=True,
    )
    coupled = Node(
        "q3_coupled",
        10,
        80,
        80,
        11,
        "在固定物理域 0≤r≤R0 上求解单时间步\nC→(ρ,cp,k)→T；(T,C)→D→C\n中心零通量、表面换热/传质，Picard 同时收敛后接受状态",
        fontsize=6.95,
    )
    reconstruct = Node(
        "q3_reconstruct",
        15,
        66,
        70,
        8,
        "重构中心、内部控制体与真实表面值；计算 Cmax(t)=max[0≤r≤R0] C(r,t)",
        fontsize=6.95,
    )
    decision = Node(
        "q3_decision",
        24,
        49,
        52,
        11,
        "是否首次满足 Cmax(t)<0.15 kg/kg？",
        kind="decision",
        fontsize=7.1,
        bold=True,
        linewidth=1.35,
    )
    adapt = Node(
        "q3_adapt",
        4,
        30,
        42,
        12,
        "未达标：继续推进\nΔt=1 s（前4 h）\nΔt=30 s（平台段）\nCmax<0.152 后恢复 1 s",
        fontsize=6.9,
    )
    bracket = Node(
        "q3_bracket",
        54,
        30,
        42,
        12,
        "达标：锁定首次严格越阈时刻\nCmax(207016 s)=0.1500001537\nCmax(207017 s)=0.1499998625\n最终控制位置：r=0",
        fontsize=6.65,
    )
    output = Node(
        "q3_output",
        14,
        13,
        72,
        11,
        "固定半径完整干燥时间\ntd=207017 s=57.5047 h\n每60 s输出径向场；网格加密时间差 103 s，长期边界窗灵敏度最大约 0.127%",
        fontsize=6.9,
        bold=True,
        double_border=True,
        linewidth=1.35,
    )
    note = Node(
        "q3_note",
        10,
        2,
        80,
        6,
        "判据检查全域未舍入值，而非表面值、平均值或表格中的四舍五入值",
        fontsize=7.0,
        bold=True,
        linestyle="--",
    )

    nodes = [physics, boundary, step, coupled, reconstruct, decision, adapt, bracket, output, note]
    edges = [
        Edge((bottom(physics), (26, 109), (50, 109), top(step))),
        Edge((bottom(boundary), (74, 109), (50, 109), top(step))),
        Edge((bottom(step), top(coupled))),
        Edge((bottom(coupled), top(reconstruct))),
        Edge((bottom(reconstruct), top(decision))),
        Edge((left(decision), (10, 54.5), (10, 42), top(adapt)), label="否", label_xy=(12.5, 53.5)),
        Edge((right(decision), (90, 54.5), (90, 42), top(bracket)), label="是", label_xy=(87.5, 53.5)),
        Edge(((25, 30), (25, 27), (2, 27), (2, 100), left(step))),
        Edge((bottom(bracket), (75, 27), (50, 27), top(output))),
        Edge((bottom(output), top(note))),
    ]

    return FlowFigure(
        slug="A_q3_physical_flowchart_bw_v2",
        title="问题三｜固定半径下的完整干燥过程与终止事件",
        subtitle="核心：延拓长期环境边界，并以“全域首次严格低于阈值”定义干燥终点",
        nodes=nodes,
        edges=edges,
        section_headers=[
            (130, "01  继承的物理模型与长期环境边界"),
            (108, "02  全程推进与全域水分终止事件"),
            (46, "03  阈值定位、输出与稳健性"),
        ],
    )


def _q4() -> FlowFigure:
    moving_geom = Node("q4_moving_geom", 4, 114, 28, 12, "", linewidth=1.2)
    radius = Node(
        "q4_radius",
        36,
        114,
        60,
        12,
        "移动物理域：0≤r≤R(t)；附件2分段线性插值得 R(t)\n半径由 2.000 cm 收缩至约 1.200 cm；材料点保持 r/R(t) 不变\n环境沿用问题三：0～4 h 插值，之后取末 1 h 时间加权平均",
        fontsize=6.8,
    )
    mapping = Node(
        "q4_mapping",
        8,
        98,
        84,
        11,
        "固定材料坐标：ξ=r/R(t)，0≤ξ≤1\n∂/∂r=[1/R(t)]∂/∂ξ；材料速度 u_r=[(dR/dt)/R(t)]r\n固定 ξ 推进即随材料运动，无需另加由网格收缩产生的对流项",
        fontsize=6.95,
        bold=True,
    )
    heat = Node(
        "q4_heat",
        4,
        73,
        44,
        17,
        "变换后的能量方程\nρ(C)cp(C)∂T(ξ,t)/∂t\n= {1/[R²(t)ξ]} ∂/∂ξ[ξk(C)∂T/∂ξ]\nξ=0：∂T/∂ξ=0\nξ=1：(ks/R)∂T/∂ξ=h[T∞−T(1,t)]",
        fontsize=6.75,
        align="left",
        linewidth=1.35,
    )
    moisture = Node(
        "q4_moisture",
        52,
        73,
        44,
        17,
        "变换后的水分方程\n∂C(ξ,t)/∂t = {1/[R²(t)ξ]}\n          ×∂/∂ξ[ξD(C,T)∂C/∂ξ]\nξ=0：∂C/∂ξ=0\nξ=1：−(Ds/R)∂C/∂ξ=km[C(1,t)−C∞]",
        fontsize=6.75,
        align="left",
        linewidth=1.35,
    )
    properties = Node(
        "q4_properties",
        8,
        60,
        84,
        8,
        "附录4物性：ρ=760+90C；cp=1850+2150C/(C+1)；k=0.12+0.20C/(C+1)\nD=4.2×10^(-4) exp(−0.30/C)exp(−3850/TK)，TK 使用 K",
        fontsize=6.8,
    )
    mechanism = Node(
        "q4_mechanism",
        12,
        50,
        76,
        6,
        "1/R²(t) 随收缩增大 → 内部扩散路径缩短、场量更快均匀化；表面积/体积比也随之改变",
        fontsize=6.95,
        bold=True,
        linestyle="--",
    )
    numeric = Node(
        "q4_numeric",
        8,
        36,
        84,
        9,
        "每步更新 R(t)、物理网格、控制体体积及表面阻力；320 个材料控制体\n在当前收缩网格上作后向 Euler–Picard–Thomas 耦合求解，并重构移动表面值",
        fontsize=6.9,
    )
    decision = Node(
        "q4_decision",
        24,
        21,
        52,
        10,
        "Cmax(t)=max[0≤r≤R(t)] C(r,t)<0.15？",
        kind="decision",
        fontsize=7.05,
        bold=True,
        linewidth=1.35,
    )
    output = Node(
        "q4_output",
        10,
        5,
        80,
        11,
        "移动边界干燥时间 td=184022 s=51.1172 h\n固定距离点落到 r>R(t) 时留空，另列真实移动表面；相对问题三缩短 11.11%\n该差异是“收缩几何 + 附录4物性”的综合影响，不能全部归因于收缩",
        fontsize=6.85,
        bold=True,
        double_border=True,
        linewidth=1.35,
    )

    nodes = [moving_geom, radius, mapping, heat, moisture, properties, mechanism, numeric, decision, output]
    edges = [
        Edge((right(moving_geom), left(radius))),
        Edge(((50, 114), top(mapping))),
        Edge((bottom(mapping), (50, 94), (26, 94), top(heat))),
        Edge(((50, 94), (74, 94), top(moisture))),
        Edge((bottom(heat), (26, 70), (50, 70), top(properties))),
        Edge((bottom(moisture), (74, 70), (50, 70), top(properties))),
        Edge((bottom(properties), top(mechanism))),
        Edge((bottom(mechanism), top(numeric))),
        Edge((bottom(numeric), top(decision))),
        Edge((left(decision), (5, 26), (5, 40.5), left(numeric)), label="否", label_xy=(8, 25.5)),
        Edge((bottom(decision), top(output)), label="是", label_xy=(53, 18.7)),
    ]

    return FlowFigure(
        slug="A_q4_physical_flowchart_bw_v2",
        title="问题四｜收缩药材的移动边界水热耦合模型",
        subtitle="核心：用材料坐标固定移动区域，显式展示半径收缩如何进入控制方程与边界条件",
        nodes=nodes,
        edges=edges,
        section_headers=[
            (130, "01  时变半径、移动物理域与材料坐标"),
            (94, "02  收缩坐标下的控制方程、边界与物性"),
            (47, "03  动态网格求解、阈值事件与结果解释"),
        ],
        decorators=[
            lambda ax: _draw_cylinder_icon(ax, moving_geom, moving=True),
            lambda ax: _geometry_text(ax, moving_geom, "均匀径向\n收缩"),
        ],
    )


def build_flowcharts() -> dict[str, FlowFigure]:
    return {"q1": _q1(), "q2": _q2(), "q3": _q3(), "q4": _q4()}


def render_flowchart(spec: FlowFigure, output_path: Path, dpi: int) -> None:
    """使用 Matplotlib 渲染单张 PNG，并执行节点重叠与文本越界检查。"""

    _check_node_overlap(spec.nodes)
    fig = plt.figure(figsize=(PAGE_WIDTH_MM / 25.4, PAGE_HEIGHT_MM / 25.4), dpi=dpi)
    ax = fig.add_axes([0.025, 0.018, 0.95, 0.965])
    ax.set_xlim(0, X_MAX)
    ax.set_ylim(0, Y_MAX)
    ax.set_aspect("auto")
    ax.axis("off")

    ax.text(
        50,
        138.6,
        spec.title,
        ha="center",
        va="center",
        fontsize=14.0,
        fontproperties=FONT_BOLD,
        color=BLACK,
    )
    ax.text(
        50,
        134.6,
        spec.subtitle,
        ha="center",
        va="center",
        fontsize=7.2,
        fontproperties=FONT_REGULAR,
        color=MID_GREY,
    )
    ax.plot([4, 96], [132.4, 132.4], color=BLACK, lw=1.1)

    for y, text in spec.section_headers:
        _section_header(ax, y, text)

    for edge in spec.edges:
        _draw_edge(ax, edge)

    node_artists: list[tuple[Node, object]] = []
    for node in spec.nodes:
        _, artist = _draw_node(ax, node)
        node_artists.append((node, artist))

    for decorator in spec.decorators:
        decorator(ax)

    _fit_text_to_nodes(fig, ax, node_artists)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, facecolor=WHITE, edgecolor="none", bbox_inches=None)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=("q1", "q2", "q3", "q4"),
        help="仅生成指定问题；省略时一次生成四问。",
    )
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="PNG 分辨率，默认 600 dpi。")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="输出目录；相对路径按项目根目录解析。",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.dpi < 150:
        raise ValueError("流程图用于论文审阅，dpi 不应低于 150。")

    repo_root = Path(__file__).resolve().parents[1]
    if args.output_dir is None:
        output_dir = repo_root / "picture" / "A_flowcharts_physical_bw"
    elif args.output_dir.is_absolute():
        output_dir = args.output_dir
    else:
        output_dir = repo_root / args.output_dir

    specs = build_flowcharts()
    selected = [args.only] if args.only else ["q1", "q2", "q3", "q4"]
    for key in selected:
        spec = specs[key]
        output_path = output_dir / f"{spec.slug}.png"
        render_flowchart(spec, output_path, args.dpi)
        print(output_path)


if __name__ == "__main__":
    main()
