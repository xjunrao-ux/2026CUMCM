# 药材烘干过程的水热传递与收缩移动边界模型

## 1 模型建立的总体思路

题目要求依次描述药材在预热平衡阶段、完整烘干阶段以及发生尺寸收缩时的温度场和水分场。四问并不是四套彼此独立的模型，而是在同一传递机理上的逐层扩展：问题一采用附录 2 的常物性参数，分别求解非稳态导热方程和非线性水分扩散方程；问题二采用附录 3 的温湿相关物性，使温度场和水分场通过材料参数产生双向反馈；问题三保持问题二的控制方程不变，将环境边界延拓至长期干燥阶段，并以药材全域水分均低于阈值作为终止条件；问题四进一步引入附件 2 给出的时变半径，通过材料坐标把移动区域映射到固定计算区间，并采用附录 4 的物性关系重新计算干燥时间。

整体模型链条可以概括为

$$
\text{环境数据}\longrightarrow\text{表面对流边界}
\longrightarrow\begin{cases}
\text{圆柱径向热传导},\\
\text{圆柱径向水分扩散},
\end{cases}
\longrightarrow\text{温度场与水分场},
$$

其中问题二至问题四还包含

$$
C\longrightarrow \rho(C),\ c_p(C),\ k(C)\longrightarrow T,
$$

$$
(T,C)\longrightarrow D(T,C)\longrightarrow C.
$$

应当指出，现有求解代码没有引入蒸发潜热项。因此，问题二至问题四属于“由物性参数实现的非线性水热耦合模型”，而不是包含相变潜热的完全水热耦合模型；问题一中温度与水分方程只共享几何、时间范围和环境数据，本质上是同步计算而不是强耦合。

## 2 几何简化与模型假设

药材长度为 $L=0.25\ \mathrm m$，初始半径为 $R_0=0.02\ \mathrm m$，长径比为 $L/(2R_0)=6.25$。烘干时侧表面积与两端面积之比为

$$
\frac{2\pi R_0L}{2\pi R_0^2}=\frac{L}{R_0}=12.5,
$$

即侧表面是端面总面积的 12.5 倍。故端部效应对总体传递的影响相对较小，可将药材近似为长圆柱，并只保留径向变化。基于题目给出的数据范围和现有代码，作如下假设。

1. 药材内部在同一径向位置处的温度和水分均匀，忽略轴向和周向梯度，即 $T=T(r,t)$、$C=C(r,t)$。
2. 药材视为连续、均匀且各向同性的多孔介质，内部不存在宏观液体对流，水分迁移用等效 Fick 扩散描述。
3. 药材内部无体积热源，忽略热辐射；现有模型不考虑蒸发潜热对温度场的直接反馈。
4. 药材表面与烘房空气之间分别服从对流换热和对流传质规律，换热系数 $h=25\ \mathrm{W/(m^2\cdot K)}$，传质系数 $k_m=8.0\times10^{-7}\ \mathrm{m/s}$。
5. 附件 1 中的空气水分按与药材水分相容的“等效外界水分势”处理，从而可用于 Robin 传质边界。该处理属于经验性等效假设。
6. 问题一至问题三中药材半径保持 $R=R_0$；问题四中半径为附件 2 确定的 $R(t)$，并假设收缩在径向上均匀，各材料点保持相对位置 $r/R(t)$ 不变。
7. 初始时药材内部温度和水分均匀，分别为 $28\ ^\circ\mathrm C$ 和 $2.55\ \mathrm{kg/kg}$。
8. 附录给出的经验物性关系在相应问题的计算区间内成立；温差可用摄氏度计算，但 Arrhenius 指数中的温度必须使用开尔文温度。

## 3 主要符号

| 符号 | 含义 | 单位 |
|---|---|---|
| $r$ | 到圆柱轴线的当前物理距离 | m |
| $R$、$R(t)$ | 固定半径、时变半径 | m |
| $\xi=r/R(t)$ | 问题四使用的无量纲材料坐标 | 1 |
| $t$ | 时间 | s |
| $T(r,t)$ | 药材温度 | $^\circ\mathrm C$ 或 K |
| $C(r,t)$ | 药材干基水分浓度 | kg/kg |
| $T_\infty(t)$ | 烘房空气温度 | $^\circ\mathrm C$ 或 K |
| $C_\infty(t)$ | 烘房等效水分浓度 | kg/kg |
| $\rho$ | 药材密度 | kg/m$^3$ |
| $c_p$ | 药材比热容 | J/(kg$\cdot$K) |
| $k$ | 药材导热系数 | W/(m$\cdot$K) |
| $D$ | 水分有效扩散系数 | m$^2$/s |
| $h$ | 表面对流换热系数 | W/(m$^2\cdot$K) |
| $k_m$ | 表面对流传质系数 | m/s |
| $q_r$ | 径向导热通量，向外为正 | W/m$^2$ |
| $J_r$ | 径向水分通量，向外为正 | 与 $C$ 的等效单位一致 |
| $T_s$、$C_s$ | 药材真实表面温度和水分 | $^\circ\mathrm C$、kg/kg |
| $V_i$ | 单位轴向长度下第 $i$ 个控制体体积 | m$^3$/m |
| $G_f$ | 控制体界面等效导通系数 | 随方程而定 |

## 4 烘房环境边界的构造

附件 1 只在离散时刻 $t_j$ 给出烘房温度和水分。为使每个数值时间步都有确定的边界输入，在相邻观测点间采用分段线性插值。对任一环境变量 $y\in\{T_\infty,C_\infty\}$，当 $t_j\le t\le t_{j+1}$ 时，定义

$$
y(t)=y_j+\frac{y_{j+1}-y_j}{t_{j+1}-t_j}(t-t_j).
$$

该构造严格通过原始观测节点，不引入过冲，且与代码中的 `numpy.interp` 一致。问题一取 $0\sim1800\ \mathrm s$，问题二取 $0\sim10800\ \mathrm s$；问题三和问题四在 $0\sim4\ \mathrm h$ 内使用附件 1 的插值结果。

附件 1 在 4 h 后没有数据。考虑到最后一小时已进入相对稳定的平台段，问题三和问题四将 $3\sim4\ \mathrm h$ 的时间加权平均作为后续恒定环境边界：

$$
\bar T_\infty=\frac{1}{3600}
\int_{10800}^{14400}T_\infty(t)\,\mathrm dt,
\qquad
\bar C_\infty=\frac{1}{3600}
\int_{10800}^{14400}C_\infty(t)\,\mathrm dt.
$$

离散数据的积分采用梯形公式

$$
\int_{t_0}^{t_n}y(t)\,\mathrm dt
\approx\sum_{j=0}^{n-1}\frac{y_j+y_{j+1}}{2}(t_{j+1}-t_j).
$$

由附件数据得到

$$
\bar T_\infty=49.9959167\ ^\circ\mathrm C,
\qquad
\bar C_\infty=0.04999042\ \mathrm{kg/kg}.
$$

因此长期边界写成

$$
(T_\infty,C_\infty)=
\begin{cases}
(T_{\mathrm{att}}(t),C_{\mathrm{att}}(t)),&0\le t\le14400\ \mathrm s,\\
(\bar T_\infty,\bar C_\infty),&t>14400\ \mathrm s.
\end{cases}
$$

已有灵敏度计算表明，把平均时间窗改为最后 0.25、0.5、1.5 或 2 h 时，干燥时间最大变化约 4.37 min，仅占基准结果的 0.127%，因而最后 1 h 平均边界在本数据范围内具有较好的稳健性。

## 5 固定半径圆柱的控制方程

### 5.1 温度方程的守恒推导

在半径 $r$ 处取厚度为 $\mathrm dr$、轴向长度为 1 的同心圆环微元。其体积和径向传热面积分别为

$$
\mathrm dV=\pi[(r+\mathrm dr)^2-r^2]
=2\pi r\,\mathrm dr+O(\mathrm dr^2),
$$

$$
A(r)=2\pi r.
$$

根据傅里叶定律，沿半径增大方向的热通量为

$$
q_r(r,t)=-k\frac{\partial T}{\partial r}.
$$

单位时间内由内侧流入微元的热量为 $q_r(r)A(r)$，由外侧流出的热量为 $q_r(r+\mathrm dr)A(r+\mathrm dr)$。由“显热积累率等于净导热流入率”得

$$
\rho c_p\frac{\partial T}{\partial t}\,2\pi r\,\mathrm dr
=q_r(r)2\pi r-q_r(r+\mathrm dr)2\pi(r+\mathrm dr).
$$

右端作一阶展开并除以 $2\pi r\,\mathrm dr$，有

$$
\rho c_p\frac{\partial T}{\partial t}
=-\frac{1}{r}\frac{\partial(rq_r)}{\partial r}.
$$

代入傅里叶定律，得到圆柱坐标下的一维非稳态导热方程

$$
\boxed{
\rho(C)c_p(C)\frac{\partial T}{\partial t}
=\frac{1}{r}\frac{\partial}{\partial r}
\left[rk(C)\frac{\partial T}{\partial r}\right]
}.
$$

若 $\rho$、$c_p$ 和 $k$ 均为常数，则方程化为

$$
\frac{\partial T}{\partial t}
=\alpha\left(\frac{\partial^2T}{\partial r^2}
+\frac{1}{r}\frac{\partial T}{\partial r}\right),
\qquad
\alpha=\frac{k}{\rho c_p}.
$$

问题一中

$$
\alpha=\frac{0.36}{820\times2600}
=1.6886\times10^{-7}\ \mathrm{m^2/s}.
$$

其 Biot 数为

$$
Bi_h=\frac{hR}{k}
=\frac{25\times0.02}{0.36}=1.3889>0.1.
$$

这说明药材内部导热热阻不可忽略，不能用整体温度相同的集中参数模型代替空间分布模型。至 $t=1800\ \mathrm s$ 时的 Fourier 数为

$$
Fo=\frac{\alpha t}{R^2}=0.7598,
$$

表明预热已向内部明显传播，但尚不能预先假定温度完全均匀。

### 5.2 水分方程的守恒推导

采用等效扩散理论，把题目给出的干基水分 $C$ 作为扩散状态量。根据 Fick 第一定律，向外的径向水分通量为

$$
J_r=-D(C,T)\frac{\partial C}{\partial r}.
$$

对同一圆环微元应用水分守恒，有

$$
\frac{\partial C}{\partial t}\,2\pi r\,\mathrm dr
=J_r(r)2\pi r-J_r(r+\mathrm dr)2\pi(r+\mathrm dr).
$$

令 $\mathrm dr\to0$，得到

$$
\frac{\partial C}{\partial t}
=-\frac{1}{r}\frac{\partial(rJ_r)}{\partial r}.
$$

代入 Fick 定律，得到非线性水分扩散方程

$$
\boxed{
\frac{\partial C}{\partial t}
=\frac{1}{r}\frac{\partial}{\partial r}
\left[rD(C,T)\frac{\partial C}{\partial r}\right]
}.
$$

由于 $D$ 随状态变化，通常不能把 $D$ 直接移到微分算子外。现有代码的方程左端储存系数取 1，因此严格守恒的是等效积分量 $\int_VC\,\mathrm dV$，不能直接把它解释为真实水质量。若需要建立严格的干基水质量守恒，应进一步引入干固体密度 $\rho_d$，写成

$$
\frac{\partial(\rho_dC)}{\partial t}
=\frac{1}{r}\frac{\partial}{\partial r}
\left(r\rho_dD\frac{\partial C}{\partial r}\right).
$$

### 5.3 初始条件

四问均从相同初始状态开始：

$$
\boxed{T(r,0)=28\ ^\circ\mathrm C},
\qquad
\boxed{C(r,0)=2.55\ \mathrm{kg/kg}}.
$$

### 5.4 中心对称边界

圆柱轴线处不存在净径向通量，故

$$
\boxed{
\left.\frac{\partial T}{\partial r}\right|_{r=0}=0
},
\qquad
\boxed{
\left.\frac{\partial C}{\partial r}\right|_{r=0}=0
}.
$$

该条件同时消除了控制方程在 $r=0$ 处形式上的 $1/r$ 奇异性。

### 5.5 表面对流换热边界

药材表面由空气获得热量。若仍以径向向外为正，则固体向外导热通量为 $-k\partial T/\partial r$，空气对药材的净供热通量为 $h(T_\infty-T_s)$。界面通量连续给出

$$
\boxed{
k_s\left.\frac{\partial T}{\partial r}\right|_{r=R}
=h[T_\infty(t)-T_s(t)]
},
$$

等价地也可写成

$$
-k_s\left.\frac{\partial T}{\partial r}\right|_{r=R}
=h[T_s(t)-T_\infty(t)].
$$

论文中两种写法只能选择一种，并应保持通量正方向一致。

### 5.6 表面对流传质边界

以水分由药材向空气排出为正，表面对流传质通量为

$$
J_s=k_m[C_s(t)-C_\infty(t)].
$$

与内部扩散通量连续可得

$$
\boxed{
-D_s\left.\frac{\partial C}{\partial r}\right|_{r=R}
=k_m[C_s(t)-C_\infty(t)]
}.
$$

在问题一的初始状态下，

$$
D(C_0)=7\times10^{-9}\exp\left(-\frac{0.89}{2.55}\right)
=4.94\times10^{-9}\ \mathrm{m^2/s},
$$

相应传质 Biot 数约为

$$
Bi_m=\frac{k_mR}{D(C_0)}\approx3.24>0.1.
$$

因此内部扩散阻力同样不可忽略，这与计算中表层先失水、中心长期保持高含水率的现象一致。

## 6 问题一的常物性热湿同步传递模型

### 6.1 温度模型

问题一采用附录 2 的常数

$$
\rho=820\ \mathrm{kg/m^3},\qquad
c_p=2600\ \mathrm{J/(kg\cdot K)},\qquad
k=0.36\ \mathrm{W/(m\cdot K)}.
$$

故温度方程为

$$
\boxed{
\rho c_p\frac{\partial T}{\partial t}
=\frac{k}{r}\frac{\partial}{\partial r}
\left(r\frac{\partial T}{\partial r}\right)
},
\qquad 0<r<R_0, 0<t\le1800\ \mathrm s.
$$

配合中心绝热边界和表面对流换热边界即可得到预热阶段的径向温度场。

### 6.2 水分模型

附录 2 给出水分扩散系数

$$
\boxed{
D(C)=7.0\times10^{-9}
\exp\left(-\frac{0.89}{C}\right)
}.
$$

于是水分模型为

$$
\boxed{
\frac{\partial C}{\partial t}
=\frac{1}{r}\frac{\partial}{\partial r}
\left[rD(C)\frac{\partial C}{\partial r}\right]
},
\qquad 0<r<R_0, 0<t\le1800\ \mathrm s.
$$

温度方程中没有 $C$，水分扩散系数中也没有 $T$，故两方程在数学上可分别求解。将二者在同一环境数据和输出时刻下同步推进，只是为了统一获得温度—水分时空结果，不应在论文中将其表述为强耦合模型。

## 7 问题二的非线性水热物性耦合模型

问题二使用附录 3 的经验公式：

$$
\boxed{\rho(C)=650+128C},
$$

$$
\boxed{c_p(C)=1450+2736\frac{C}{C+1}},
$$

$$
\boxed{k(C)=0.21+0.38\frac{C}{C+1}},
$$

$$
\boxed{
D(C,T_K)=2.4\times10^{-3}
\exp\left(-\frac{0.45}{C}\right)
\exp\left(-\frac{3850}{T_K}\right)
},
\qquad T_K=T+273.15.
$$

代入共同控制方程，得到

$$
\boxed{
\rho(C)c_p(C)\frac{\partial T}{\partial t}
=\frac{1}{r}\frac{\partial}{\partial r}
\left[rk(C)\frac{\partial T}{\partial r}\right]
},
$$

$$
\boxed{
\frac{\partial C}{\partial t}
=\frac{1}{r}\frac{\partial}{\partial r}
\left[rD(C,T_K)\frac{\partial C}{\partial r}\right]
}.
$$

当水分下降时，$\rho$、$c_p$ 和 $k$ 随之变化，从而改变温度场；温度和水分又共同改变 $D$，进而反馈到水分场。这构成了非线性双向物性耦合。现有计算中水分扩散系数范围为

$$
5.55\times10^{-9}\le D\le1.25\times10^{-8}\ \mathrm{m^2/s},
$$

热扩散率

$$
\alpha(C)=\frac{k(C)}{\rho(C)c_p(C)}
$$

的范围约为

$$
1.45\times10^{-7}\le\alpha\le1.82\times10^{-7}\ \mathrm{m^2/s}.
$$

热扩散率比水分扩散系数高约一个数量级，因而温度场比水分场更快趋于均匀；这也为后续“干燥后期主要受内部水分迁移限制”的结果解释提供了机理依据。

## 8 问题三的长期干燥模型与终止事件

问题三沿用问题二的控制方程、物性关系、初始条件和边界条件，并从 $t=0$ 重新计算完整过程。区别只在于：4 h 后使用第 4 节构造的恒定环境边界，并持续推进直至所有位置均满足水分阈值。

定义全域最大水分

$$
C_{\max}(t)=\max_{0\le r\le R_0}C(r,t).
$$

烘干结束时间定义为首次严格达标时刻

$$
\boxed{
t_d=\inf\left\{t>0:C_{\max}(t)<0.15\right\}
}.
$$

该判据检查重构中心值、全部内部控制体中心值和真实表面值，而不是只检查表面水分或体积平均水分。由于中心位置的水分最高，最终控制位置为 $r=0$。程序得到阈值两侧状态

$$
C_{\max}(207016\ \mathrm s)=0.1500001537>0.15,
$$

$$
C_{\max}(207017\ \mathrm s)=0.1499998625<0.15,
$$

故问题三的干燥时间为

$$
\boxed{t_d=207017\ \mathrm s=57.5047\ \mathrm h}.
$$

结果表按四位小数输出时末时刻中心水分显示为 0.1500，但程序的事件判断使用未舍入值，因此仍满足严格小于 0.15 的要求。

## 9 问题四的收缩移动边界模型

### 9.1 时变半径的构造

设附件 2 在时刻 $t_j$ 给出的药材半径为 $R_j$。相邻数据点之间采用分段线性插值：

$$
\boxed{
R(t)=R_j+\frac{R_{j+1}-R_j}{t_{j+1}-t_j}(t-t_j),
\quad t_j\le t\le t_{j+1}
}.
$$

附件 2 中半径从 $2.000\ \mathrm{cm}$ 逐步减小并在主要干燥阶段稳定于约 $1.200\ \mathrm{cm}$。问题四的物理区域随时间变化为

$$
0\le r\le R(t).
$$

### 9.2 材料坐标变换

直接在随时间变化的区域上求解会导致网格节点不断移动。为固定计算区间，引入无量纲材料坐标

$$
\boxed{\xi=\frac{r}{R(t)}},
\qquad 0\le\xi\le1,
$$

并定义

$$
\widehat T(\xi,t)=T(\xi R(t),t),
\qquad
\widehat C(\xi,t)=C(\xi R(t),t).
$$

空间导数满足

$$
\frac{\partial}{\partial r}
=\frac{1}{R(t)}\frac{\partial}{\partial\xi}.
$$

对任意场变量 $\phi$，固定物理位置和固定材料坐标下的时间导数关系为

$$
\left.\frac{\partial\phi}{\partial t}\right|_r
=\left.\frac{\partial\widehat\phi}{\partial t}\right|_\xi
-\frac{\dot R(t)}{R(t)}\xi
\frac{\partial\widehat\phi}{\partial\xi}.
$$

均匀径向收缩时材料点速度为

$$
u_r(r,t)=\frac{\dot R(t)}{R(t)}r=\xi\dot R(t).
$$

因此材料导数为

$$
\frac{\mathrm D\phi}{\mathrm Dt}
=\left.\frac{\partial\phi}{\partial t}\right|_r
+u_r\frac{\partial\phi}{\partial r}
=\left.\frac{\partial\widehat\phi}{\partial t}\right|_\xi.
$$

这说明在固定 $\xi$ 的网格上推进状态量，已经随材料运动，无需再显式加入由网格收缩产生的对流项。

### 9.3 变换后的控制方程

将 $r=\xi R(t)$ 和 $\partial/\partial r=R^{-1}\partial/\partial\xi$ 代入圆柱扩散算子：

$$
\frac{1}{r}\frac{\partial}{\partial r}
\left(r\Gamma\frac{\partial\phi}{\partial r}\right)
=\frac{1}{R^2(t)\xi}
\frac{\partial}{\partial\xi}
\left(\xi\Gamma\frac{\partial\widehat\phi}{\partial\xi}\right).
$$

于是问题四的材料坐标方程为

$$
\boxed{
\rho(\widehat C)c_p(\widehat C)
\frac{\partial\widehat T}{\partial t}
=\frac{1}{R^2(t)\xi}
\frac{\partial}{\partial\xi}
\left[\xi k(\widehat C)
\frac{\partial\widehat T}{\partial\xi}\right]
},
$$

$$
\boxed{
\frac{\partial\widehat C}{\partial t}
=\frac{1}{R^2(t)\xi}
\frac{\partial}{\partial\xi}
\left[\xi D(\widehat C,\widehat T)
\frac{\partial\widehat C}{\partial\xi}\right]
}.
$$

式中的 $R^{-2}(t)$ 表明，在其他条件相同时，半径减小会缩短内部扩散路径并加快场量均匀化，这是收缩模型给出更短干燥时间的重要原因之一。

在 $\xi=0$ 处仍有对称条件

$$
\left.\frac{\partial\widehat T}{\partial\xi}\right|_{\xi=0}=0,
\qquad
\left.\frac{\partial\widehat C}{\partial\xi}\right|_{\xi=0}=0.
$$

在 $\xi=1$ 处，表面边界变为

$$
\boxed{
\frac{k_s}{R(t)}
\left.\frac{\partial\widehat T}{\partial\xi}\right|_{\xi=1}
=h(T_\infty-\widehat T_s)
},
$$

$$
\boxed{
-\frac{D_s}{R(t)}
\left.\frac{\partial\widehat C}{\partial\xi}\right|_{\xi=1}
=k_m(\widehat C_s-C_\infty)
}.
$$

### 9.4 问题四的物性关系

问题四采用附录 4：

$$
\boxed{\rho(C)=760+90C},
$$

$$
\boxed{c_p(C)=1850+2150\frac{C}{C+1}},
$$

$$
\boxed{k(C)=0.12+0.20\frac{C}{C+1}},
$$

$$
\boxed{
D(C,T_K)=4.2\times10^{-4}
\exp\left(-\frac{0.30}{C}\right)
\exp\left(-\frac{3850}{T_K}\right)
}.
$$

温度在程序内部直接用 K 保存，从而避免 Arrhenius 公式中摄氏温度与绝对温度混用。

### 9.5 问题四的终止条件与输出区域

在当前物理区域内定义

$$
C_{\max}(t)=\max_{0\le r\le R(t)}C(r,t),
$$

并仍以 $C_{\max}(t)<0.15$ 的首次发生时刻为干燥结束时间。计算得到

$$
\boxed{t_d=184022\ \mathrm s=51.1172\ \mathrm h}.
$$

此时前一步和终止步的最大水分分别为

$$
C_{\max}(184021\ \mathrm s)=0.1500003574,
$$

$$
C_{\max}(184022\ \mathrm s)=0.1499998440.
$$

问题四的完整结果按固定物理距离 $0,0.1,\ldots,2.0\ \mathrm{cm}$ 输出，并另外设置“药材表面”列。若某固定距离满足 $r>R(t)$，该位置已经位于收缩后的药材外部，结果留空；表面列始终记录 $r=R(t)$ 处的真实边界值。

现有移动边界程序与问题一至问题三保持同一等效扩散口径，即推进材料点上的 $C$ 并用当前几何计算扩散通量。其离散守恒检验针对模型中的等效储存增量，而不是包含固相骨架压缩、干物质密度变化和体积雅可比变化的完整真实水质量守恒。论文宜将其明确表述为“均匀收缩下的等效移动边界模型”。

## 10 有限体积离散

### 10.1 径向控制体与表面加密网格

除问题一的温度子模型采用均匀节点中心网格外，其余模型均使用向表面加密的单元中心有限体积网格。令逻辑坐标面

$$
\eta_j=\frac{j}{N},\qquad j=0,1,\ldots,N,
$$

固定半径问题中的物理界面为

$$
\boxed{
r_j^f=R\left[1-(1-\eta_j)^2\right]
}.
$$

问题四则先在材料坐标中定义

$$
\xi_j^f=1-(1-\eta_j)^2,
$$

再由

$$
r_j^f(t)=R(t)\xi_j^f
$$

得到当前物理网格。第 $i$ 个控制体中心和单位轴向长度下的体积为

$$
r_i=\frac{r_{i-1}^f+r_i^f}{2},
$$

$$
\boxed{
V_i=\pi\left[(r_i^f)^2-(r_{i-1}^f)^2\right]
}.
$$

二次映射使外层控制体更薄，可提高对表面温度和水分陡峭梯度的分辨能力。“名义径向步长”只用于确定控制体数量，并不等于所有单元的实际宽度。

### 10.2 统一方程形式

将温度和水分方程统一记为

$$
S\frac{\partial\phi}{\partial t}
=\frac{1}{r}\frac{\partial}{\partial r}
\left(r\Gamma\frac{\partial\phi}{\partial r}\right),
$$

其中

$$
(\phi,\Gamma,S)=
\begin{cases}
(T,k,\rho c_p),&\text{温度方程},\\
(C,D,1),&\text{水分方程}.
\end{cases}
$$

对第 $i$ 个控制体积分，可得

$$
S_iV_i\frac{\mathrm d\phi_i}{\mathrm dt}
=F_{w,i}+F_{e,i},
$$

其中 $F_{w,i}$ 和 $F_{e,i}$ 分别表示通过西、东界面流入控制体的传递量。

### 10.3 内部界面导通系数

设相邻控制体中心为 $r_i$、$r_{i+1}$，公共界面为 $r_{i+1/2}$。界面两侧物性分别为 $\Gamma_i$ 和 $\Gamma_{i+1}$。从中心 $i$ 到界面的阻力为

$$
\mathcal R_i=
\frac{r_{i+1/2}-r_i}{2\pi r_{i+1/2}\Gamma_i},
$$

从界面到中心 $i+1$ 的阻力为

$$
\mathcal R_{i+1}=
\frac{r_{i+1}-r_{i+1/2}}{2\pi r_{i+1/2}\Gamma_{i+1}}.
$$

两段阻力串联，故界面导通系数为

$$
\boxed{
G_{i+1/2}
=\frac{1}{\mathcal R_i+\mathcal R_{i+1}}
=\frac{2\pi r_{i+1/2}}
{\dfrac{r_{i+1/2}-r_i}{\Gamma_i}
+\dfrac{r_{i+1}-r_{i+1/2}}{\Gamma_{i+1}}}
}.
$$

于是由右侧相邻控制体流入第 $i$ 个控制体的传递量为

$$
F_{i+1/2}=G_{i+1/2}(\phi_{i+1}-\phi_i).
$$

该表达式相当于在非均匀网格上使用按扩散阻力加权的调和平均，能够保证界面两侧通量连续。

### 10.4 Robin 边界的串联阻力

最外层控制体中心 $r_N$ 与真实表面 $R$ 之间存在半单元内部传导或扩散阻力，表面与空气之间存在对流阻力。令

$$
\delta_s=R-r_N,
\qquad A_R=2\pi R.
$$

内部阻力和外部阻力分别为

$$
\mathcal R_{\mathrm{in}}=\frac{\delta_s}{A_R\Gamma_N},
\qquad
\mathcal R_{\mathrm{out}}=\frac{1}{A_R\beta},
$$

其中温度方程取 $\beta=h$，水分方程取 $\beta=k_m$。二者串联后得到等效边界导通系数

$$
\boxed{
G_b=\frac{1}{\mathcal R_{\mathrm{in}}+\mathcal R_{\mathrm{out}}}
=\frac{A_R}{\dfrac{\delta_s}{\Gamma_N}+\dfrac{1}{\beta}}
}.
$$

由环境向最外层控制体的净传递量为

$$
F_b=G_b(\phi_\infty-\phi_N).
$$

上述串联阻力形式用于问题一的水分子模型以及问题二至问题四的单元中心模型。问题一的温度子模型采用节点中心网格，并直接把最外侧未知量布置在 $r=R$ 的物理表面，因此该子模型的边界导通系数直接取 $hA_R$，不再附加内部半单元阻力。问题四则把 $R$、$r_N$、$A_R$ 和 $V_i$ 更新为当前时刻的 $R(t)$ 和物理网格值，其余离散结构保持不变。

### 10.5 后向 Euler 时间离散

采用全隐式后向 Euler 格式

$$
\left.\frac{\partial\phi}{\partial t}\right|_i^{n+1}
\approx\frac{\phi_i^{n+1}-\phi_i^n}{\Delta t}.
$$

对内部控制体，有

$$
S_iV_i\frac{\phi_i^{n+1}-\phi_i^n}{\Delta t}
=G_{i-1/2}(\phi_{i-1}^{n+1}-\phi_i^{n+1})
+G_{i+1/2}(\phi_{i+1}^{n+1}-\phi_i^{n+1}).
$$

整理为

$$
-G_{i-1/2}\phi_{i-1}^{n+1}
+\left(\frac{S_iV_i}{\Delta t}+G_{i-1/2}+G_{i+1/2}\right)
\phi_i^{n+1}
-G_{i+1/2}\phi_{i+1}^{n+1}
=\frac{S_iV_i}{\Delta t}\phi_i^n.
$$

最外层控制体的方程为

$$
-G_{N-1/2}\phi_{N-1}^{n+1}
+\left(\frac{S_NV_N}{\Delta t}+G_{N-1/2}+G_b\right)
\phi_N^{n+1}
=\frac{S_NV_N}{\Delta t}\phi_N^n+G_b\phi_\infty^{n+1}.
$$

离散矩阵仅包含主对角线和相邻两条对角线，因而形成三对角方程组，可用 Thomas 追赶法在 $O(N)$ 时间内求解。全隐式格式对扩散方程具有良好稳定性，时间精度为一阶。

## 11 非线性耦合迭代

问题一水分方程以及问题二至问题四的两个控制方程均含状态相关系数。每个时间步采用 Picard 迭代将非线性方程线性化。令

$$
T^{(0)}=T^n,qquad C^{(0)}=C^n.
$$

第 $m$ 次迭代依次执行：

1. 根据 $C^{(m)}$ 计算 $\rho^{(m)}$、$c_p^{(m)}$ 和 $k^{(m)}$；
2. 固定上述系数，隐式求解温度方程，得到 $T^{(m+1)}$；
3. 根据 $C^{(m)}$ 和 $T^{(m+1)}$ 计算 $D^{(m)}$；
4. 固定 $D^{(m)}$，隐式求解水分方程，得到 $C^{(m+1)}$；
5. 检查无穷范数误差

$$
\varepsilon_T=
\left\|T^{(m+1)}-T^{(m)}\right\|_\infty,
$$

$$
\varepsilon_C=
\left\|C^{(m+1)}-C^{(m)}\right\|_\infty.
$$

当

$$
\boxed{\varepsilon_T<10^{-8}},
\qquad
\boxed{\varepsilon_C<10^{-10}}
$$

同时成立时接受该时间步。问题二和问题三最多允许 30 次 Picard 迭代；实际最大迭代次数分别为 4 次和 11 次，问题四为 8 次，所有时间步均正常收敛。这一更新顺序属于分块 Gauss-Seidel 型 Picard 迭代，因为新温度立即用于当前轮次的扩散系数计算。

## 12 中心值与真实表面值重构

### 12.1 中心值重构

单元中心有限体积法的首个未知量不在 $r=0$。由中心对称性，中心附近的场变量可展开为偶函数

$$
\phi(r)=\phi(0)+ar^2+O(r^4).
$$

利用最靠近中心的两个控制体值 $\phi_0$、$\phi_1$，可消去系数 $a$，得到二阶中心重构公式

$$
\boxed{
\phi(0)=
\frac{\phi_0r_1^2-\phi_1r_0^2}{r_1^2-r_0^2}
}.
$$

### 12.2 表面值重构

对于单元中心网格，最外层控制体中心值 $\phi_N$ 不等于真实表面值 $\phi_s$。由内部传递与表面对流通量连续，有

$$
\frac{\Gamma_N}{\delta_s}(\phi_s-\phi_N)
=\beta(\phi_\infty-\phi_s).
$$

整理得

$$
\boxed{
\phi_s=
\frac{\dfrac{\Gamma_N}{\delta_s}\phi_N+\beta\phi_\infty}
{\dfrac{\Gamma_N}{\delta_s}+\beta}
}.
$$

问题一的水分场以及问题二至问题四的温度场和水分场均使用该式重构真实表面值。对问题四，只需令 $\delta_s=R(t)-r_N(t)$。题目要求的其他径向位置则在“中心重构值—内部控制体值—表面重构值”组成的节点上作分段线性插值。问题一的温度网格已经在 $r=0$ 和 $r=R$ 布置节点，因而中心值和表面值可直接取对应节点未知量。

## 13 时间步、网格与求解参数

| 问题 | 空间离散 | 正式时间步 | 输出间隔 | 备注 |
|---|---|---|---|---|
| 问题一温度 | 均匀节点中心网格，$\Delta r=0.025$ cm | 0.25 s | 1 s | 常系数矩阵预分解 |
| 问题一水分 | 160 个表面加密控制体，名义步长 0.0125 cm | 0.125 s | 1 s | Picard 迭代 |
| 问题二 | 160 个表面加密控制体，名义步长 0.0125 cm | 0.5 s | 1 s | 温湿物性耦合 |
| 问题三 | 160 个表面加密控制体，名义步长 0.0125 cm | 1 s、30 s、1 s | 60 s | 环境实测段、平台段、阈值附近 |
| 问题四 | 320 个材料控制体，初始名义步长 0.00625 cm | 1 s、30 s、1 s | 60 s | 当前半径物理网格 |

问题三和问题四采用分段时间步

$$
\Delta t=
\begin{cases}
1\ \mathrm s,&t<4\ \mathrm h,\\
30\ \mathrm s,&t\ge4\ \mathrm h\text{ 且 }C_{\max}\ge0.152,\\
1\ \mathrm s,&C_{\max}<0.152.
\end{cases}
$$

程序会自动缩短单个时间步，使时间网格精确经过 4 h 边界切换点和每个 60 s 输出时刻；当接近阈值时恢复 1 s 步长，以准确定位首次严格达标时刻。

## 14 数值可靠性检验

### 14.1 守恒检验

固定半径模型中，水分的离散储存变化为

$$
\Delta M_C=\sum_iV_i(C_i^{\mathrm{end}}-C_i^0),
$$

累计表面流出量为

$$
M_{\mathrm{out}}
=\sum_n\Delta t_nG_b^n(C_N^{n+1}-C_\infty^{n+1}).
$$

理论上应满足

$$
\boxed{\Delta M_C+M_{\mathrm{out}}=0}.
$$

问题一、问题二和问题三的相对水分守恒残差分别约为

$$
5.67\times10^{-14},\qquad
3.96\times10^{-13},\qquad
3.63\times10^{-12}.
$$

问题一温度模型还满足

$$
\sum_i\rho c_pV_i(T_i^{\mathrm{end}}-T_i^0)
-\sum_n\Delta t_nhA_R(T_\infty^{n+1}-T_s^{n+1})=0,
$$

其相对能量残差为 $6.73\times10^{-13}$。

问题四在移动物理网格上的等效离散储存增量与累计边界流出量之和的相对残差为 $2.20\times10^{-13}$。该指标验证的是代码所离散方程的代数守恒性，而非包含固相压缩的真实质量守恒。

### 14.2 网格与时间步加密

问题一温度场的正式网格与更细参考网格相比，最大绝对差为 $2.51\times10^{-4}\ ^\circ\mathrm C$；问题一水分场粗细计算的最大差为 $2.61\times10^{-4}\ \mathrm{kg/kg}$。

问题二将名义径向步长由 0.025 cm 加密至 0.0125 cm，同时将时间步由 1.0 s 缩小至 0.5 s，完整温度场和水分场最大差分别为

$$
1.064\times10^{-3}\ ^\circ\mathrm C,
\qquad
1.431\times10^{-3}\ \mathrm{kg/kg}.
$$

问题三粗细方案得到的干燥时间分别为 207120 s 和 207017 s，相差 103 s；问题四粗细方案得到的干燥时间分别为 184026 s 和 184022 s，仅相差 4 s。上述结果表明关键结论对离散尺度不敏感。

### 14.3 物理合理性检查

所有计算均检查以下条件：温度和水分保持有限且水分为正；水分随时间无反常增加；从中心到表面的水分总体非增；终止前一步不满足阈值而终止步全域满足阈值；问题四中当前半径之外没有伪造数值、当前半径之内没有缺失值。各项检查均通过。

## 15 四问之间的递进关系与主要结论

问题一建立了预热平衡阶段的基础传递框架。30 min 时药材中心和表面温度分别为 $33.5756\ ^\circ\mathrm C$ 和 $36.7857\ ^\circ\mathrm C$，中心和表面水分分别为 $2.5500\ \mathrm{kg/kg}$ 和 $1.5102\ \mathrm{kg/kg}$，说明热量已经明显向内部传播，而失水仍主要集中在表层。

问题二引入温湿相关物性后，3 h 时中心和表面温度分别为 $49.8495\ ^\circ\mathrm C$ 和 $49.9664\ ^\circ\mathrm C$，温差仅约 $0.117\ ^\circ\mathrm C$；同期中心和表面水分分别为 $1.7662\ \mathrm{kg/kg}$ 和 $1.0081\ \mathrm{kg/kg}$，仍存在明显梯度。因此，后期干燥的主要限制环节已由外部供热逐渐转为内部水分扩散。

问题三在固定半径下得到干燥时间 $57.5047\ \mathrm h$。问题四考虑收缩后得到 $51.1172\ \mathrm h$，比固定半径模型缩短约

$$
57.5047-51.1172=6.3875\ \mathrm h,
$$

相对缩短

$$
\frac{6.3875}{57.5047}\times100\%\approx11.11\%.
$$

问题四相对问题三同时改变了几何描述和经验物性体系，因此上述 $11.11\%$ 是“收缩移动边界与附录 4 参数”的综合影响，不能全部归因于尺寸收缩。就机理而言，半径减小会缩短内部扩散路径并改变表面积与体积比，是问题四干燥加快的重要因素；若要单独量化收缩效应，还应在相同物性参数下增设固定半径对照组。

## 16 论文表述中需要保持的边界

1. 问题一宜称“常物性热湿同步传递模型”，不要称为强耦合模型。
2. 问题二至问题四宜称“非线性水热物性耦合模型”，并明确未计蒸发潜热。
3. $T$ 在热传导温差中可用摄氏度，但在 $\exp(-3850/T)$ 中必须使用 K。
4. 表面边界是第三类 Robin 边界，真实表面值由内部半单元阻力和外部对流阻力共同决定，不能直接令 $T_s=T_\infty$ 或 $C_s=C_\infty$。
5. 问题三和问题四的干燥终止条件是“所有位置严格小于 0.15”，判断时必须使用未舍入数值。
6. 问题四的固定距离输出点在收缩后可能落到药材之外，此时应留空，不能外推。
7. 现有模型中的水分守恒是对等效扩散变量的守恒。若论文需要讨论真实失水质量或能耗，应另行引入干物质密度、蒸发潜热及严格焓守恒方程，不能直接由当前积分量替代。
8. 温度方程采用表观体积热容形式 $\rho(C)c_p(C)\,\partial T/\partial t$，而不是严格焓形式 $\partial[\rho(C)c_p(C)T]/\partial t$，因此没有包含 $\rho$、$c_p$ 随水分变化产生的附加时间导数项。
9. 问题四相对问题三的时间差是几何与物性同时变化的综合结果。若要解释为纯粹的“收缩贡献”，还需在同一套附录 4 物性下增加固定半径对照计算。
