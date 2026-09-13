# A题第二问：水热耦合模型的建立与数值求解

## 1. 文档说明

本文依据第二问实际计算程序 [`a2_coupled_fvm.py`](../code/a2_coupled_fvm.py) 整理，目的是使论文中的物理假设、控制方程、初边值条件、数值离散和程序实现保持一致。

当前模型属于**固定半径圆柱的一维径向非线性水热物性耦合模型**。第二问暂不考虑药材收缩、蒸发潜热、热辐射和内部宏观对流，水分与温度主要通过热物性和有效扩散系数相互影响。

模型中的耦合关系为

$$
C\longrightarrow \rho(C),\;c_p(C),\;\lambda(C)\longrightarrow T,
$$

$$
(T,C)\longrightarrow D(T,C)\longrightarrow C.
$$

因此，该模型是双向物性耦合模型，但不是包含蒸发潜热源项的完全水热耦合模型。

---

## 2. 符号与参数

| 符号 | 含义 | 单位 |
|---|---|---|
| $r$ | 到圆柱轴线的径向距离 | m |
| $R$ | 药材半径，$R=0.02$ | m |
| $t$ | 干燥时间 | s |
| $T(r,t)$ | 药材内部温度 | ℃ |
| $T_K(r,t)$ | 药材内部绝对温度，$T_K=T+273.15$ | K |
| $C(r,t)$ | 药材内部水分浓度或含水率 | kg/kg |
| $T_a(t)$ | 烘房空气温度 | ℃ |
| $C_a(t)$ | 烘房空气水分 | kg/kg |
| $\rho(C)$ | 药材密度 | kg/m³ |
| $c_p(C)$ | 药材比热容 | J/(kg·K) |
| $\lambda(C)$ | 药材导热系数 | W/(m·K) |
| $D(C,T)$ | 水分有效扩散系数 | m²/s |
| $h$ | 表面对流换热系数，$h=25$ | W/(m²·K) |
| $k_m$ | 表面对流传质系数，$k_m=8.0\times10^{-7}$ | m/s |
| $T_s(t)$ | 药材真实表面温度 | ℃ |
| $C_s(t)$ | 药材真实表面水分 | kg/kg |

计算区域为

$$
0\le r\le R,\qquad 0\le t\le10800\ \mathrm{s}.
$$

---

## 3. 模型基本假设

1. 将药材视为均匀、各向同性的长圆柱体。
2. 圆柱轴向长度相对半径足够大，只考虑径向传热与传质，忽略轴向和周向变化。
3. 第二问中半径固定为 $2\ \mathrm{cm}$，不考虑干燥收缩和移动边界。
4. 药材内部无宏观水分对流，仅考虑由水分梯度引起的有效扩散。
5. 药材内部无体积热源。
6. 表面与空气之间分别采用对流换热和对流传质边界。
7. 换热系数 $h$ 和传质系数 $k_m$ 在全过程保持不变。
8. 不考虑热辐射。
9. 按第二问当前计算范围，不考虑水分蒸发潜热。
10. 烘房空气温度和空气水分由附件1给出的离散数据随时间线性插值得到。
11. 药材密度、比热容和导热系数随本地水分变化，有效水分扩散系数同时随本地温度和水分变化。

---

## 4. 温度控制方程

### 4.1 圆柱微元能量守恒

在半径 $r$ 处选取厚度为 $\mathrm{d}r$、轴向长度为1的圆环微元，其体积为

$$
\mathrm{d}V=2\pi r\,\mathrm{d}r.
$$

微元内显热积累率写为

$$
\rho(C)c_p(C)\frac{\partial T}{\partial t}
2\pi r\,\mathrm{d}r.
$$

根据傅里叶定律，沿半径增大方向的热通量为

$$
q_r=-\lambda(C)\frac{\partial T}{\partial r}.
$$

由“微元内能增加率等于流入热量减去流出热量”，有

$$
\rho(C)c_p(C)\frac{\partial T}{\partial t}
2\pi r\,\mathrm{d}r
=-
\frac{\partial}{\partial r}
\left(2\pi r q_r\right)\mathrm{d}r.
$$

代入傅里叶定律并约去 $2\pi\,\mathrm{d}r$，得到圆柱坐标下的变物性热传导方程

$$
\boxed{
\rho(C)c_p(C)\frac{\partial T}{\partial t}
=
\frac{1}{r}
\frac{\partial}{\partial r}
\left[
r\lambda(C)\frac{\partial T}{\partial r}
\right]
}.
$$

由于 $\lambda$ 随水分变化，不能将其直接移到微分算子外部。

---

## 5. 水分控制方程

### 5.1 圆柱微元水分守恒

采用有效扩散理论描述药材内部水分迁移。根据Fick定律，径向水分通量为

$$
J_r=-D(C,T)\frac{\partial C}{\partial r}.
$$

对单位轴向长度的圆环微元进行守恒分析：

$$
\frac{\partial C}{\partial t}2\pi r\,\mathrm{d}r
=-
\frac{\partial}{\partial r}
\left(2\pi rJ_r\right)\mathrm{d}r.
$$

代入Fick定律，得到

$$
\boxed{
\frac{\partial C}{\partial t}
=
\frac{1}{r}
\frac{\partial}{\partial r}
\left[
rD(C,T)\frac{\partial C}{\partial r}
\right]
}.
$$

当前代码在方程左端没有乘干物质体积密度，因此数值上严格守恒的是 $\int_V C\,\mathrm{d}V$。若需要计算真实失水质量，应进一步引入干物质密度。

---

## 6. 物性关系与耦合机理

### 6.1 密度

$$
\boxed{\rho(C)=650+128C}.
$$

### 6.2 比热容

$$
\boxed{
c_p(C)=1450+\frac{2736C}{C+1}
}.
$$

含水率越高，药材比热容越大，升高相同温度所需的显热越多。

### 6.3 导热系数

$$
\boxed{
\lambda(C)=0.21+\frac{0.38C}{C+1}
}.
$$

水分变化会改变药材的导热能力，从而使温度方程受到水分场的反馈影响。

### 6.4 水分有效扩散系数

$$
\boxed{
D(C,T_K)
=2.4\times10^{-3}
\exp\left(-\frac{0.45}{C}\right)
\exp\left(-\frac{3850}{T_K}\right)
}.
$$

其中

$$
T_K=T+273.15.
$$

温度越高，Arrhenius项越大，水分扩散速度越快；水分浓度变化也会改变扩散系数。因此水分方程是同时依赖 $T$ 和 $C$ 的非线性扩散方程。

本次计算得到

$$
5.55\times10^{-9}
\le D\le
1.25\times10^{-8}\ \mathrm{m^2/s}.
$$

相应热扩散率

$$
\alpha=\frac{\lambda}{\rho c_p}
$$

的范围约为

$$
1.45\times10^{-7}
\le\alpha\le
1.82\times10^{-7}\ \mathrm{m^2/s}.
$$

热扩散率总体比水分扩散系数高一个数量级，因此温度场比水分场更快趋于均匀。

---

## 7. 初始条件

假定干燥开始时药材内部温度和水分均匀：

$$
\boxed{T(r,0)=28\ ^\circ\mathrm{C}},
\qquad 0\le r\le R,
$$

$$
\boxed{C(r,0)=2.55\ \mathrm{kg/kg}},
\qquad 0\le r\le R.
$$

---

## 8. 边界条件

### 8.1 圆柱中心对称边界

在 $r=0$ 处不存在穿过圆柱中心的净通量，因此

$$
\boxed{
\left.\frac{\partial T}{\partial r}\right|_{r=0}=0
},
$$

$$
\boxed{
\left.\frac{\partial C}{\partial r}\right|_{r=0}=0
}.
$$

有限体积离散时，中心控制面的面积为零，所以该对称条件能够自然满足。

### 8.2 表面对流换热边界

以空气向药材供热为正方向，表面对流热通量为

$$
q_h=h[T_a(t)-T_s(t)].
$$

由表面导热通量与对流换热通量连续，得到

$$
\boxed{
\lambda_s
\left.\frac{\partial T}{\partial r}\right|_{r=R}
=h[T_a(t)-T_s(t)]
}.
$$

### 8.3 表面对流传质边界

以药材向空气排湿为正方向，表面水分通量为

$$
J_C=k_m[C_s(t)-C_a(t)].
$$

由内部扩散通量和表面对流传质通量连续，得到

$$
\boxed{
-D_s
\left.\frac{\partial C}{\partial r}\right|_{r=R}
=k_m[C_s(t)-C_a(t)]
}.
$$

温度和水分的表面条件均为第三类Robin边界条件，不是给定表面温度或给定表面水分。

---

## 9. 烘房环境数据预处理

代码读取附件1的时间、空气温度和空气水分，并线性插值到每一秒。

当 $t_j\le t\le t_{j+1}$ 时，空气温度为

$$
T_a(t)=T_{a,j}
+\frac{T_{a,j+1}-T_{a,j}}{t_{j+1}-t_j}(t-t_j).
$$

空气水分为

$$
C_a(t)=C_{a,j}
+\frac{C_{a,j+1}-C_{a,j}}{t_{j+1}-t_j}(t-t_j).
$$

温度插值结果同时转换为开尔文温度，但开尔文温度只用于有效水分扩散系数中的Arrhenius项。热传导方程中的温差继续使用摄氏温度，因为摄氏温差与开尔文温差数值相同。

---

## 10. 径向有限体积网格

### 10.1 非均匀网格映射

程序没有直接使用均匀径向网格，而是向药材表面加密。令

$$
\xi_j=\frac{j}{N},\qquad j=0,1,\ldots,N,
$$

控制体界面坐标为

$$
\boxed{
r_j^{f}=R\left[1-(1-\xi_j)^2\right]
}.
$$

第 $i$ 个控制体中心为

$$
r_i=\frac{r_{i-1}^{f}+r_i^{f}}{2}.
$$

单位轴向长度的控制体体积为

$$
\boxed{
V_i=\pi\left[(r_i^{f})^2-(r_{i-1}^{f})^2\right]
}.
$$

半径为 $2\ \mathrm{cm}$，生产计算的名义径向尺度为 $0.0125\ \mathrm{cm}$，因此控制体数量为

$$
N=\frac{2}{0.0125}=160.
$$

“名义径向尺度”仅用于确定控制体数量，实际网格间距并不相等。输出文件中的 $0.1\ \mathrm{cm}$ 间隔也是后处理采样间隔，不是求解网格间距。

---

## 11. 内部界面通量离散

将温度和水分方程统一写成

$$
S\frac{\partial\phi}{\partial t}
=\frac{1}{r}\frac{\partial}{\partial r}
\left(r\Gamma\frac{\partial\phi}{\partial r}\right),
$$

其中

$$
\phi=
\begin{cases}
T,&\text{温度方程},\\
C,&\text{水分方程},
\end{cases}
$$

$$
\Gamma=
\begin{cases}
\lambda,&\text{温度方程},\\
D,&\text{水分方程},
\end{cases}
$$

$$
S=
\begin{cases}
\rho c_p,&\text{温度方程},\\
1,&\text{水分方程}.
\end{cases}
$$

相邻控制体 $i$ 和 $i+1$ 之间的界面导通系数为

$$
\boxed{
G_{i+1/2}
=
\frac{2\pi r_{i+1/2}}
{
\dfrac{r_{i+1/2}-r_i}{\Gamma_i}
+
\dfrac{r_{i+1}-r_{i+1/2}}{\Gamma_{i+1}}
}
}.
$$

该表达式将界面两侧的扩散阻力串联起来，相当于采用与非均匀网格相匹配的调和平均，可以保证界面通量连续。

相应界面传输量为

$$
F_{i+1/2}=G_{i+1/2}(\phi_{i+1}-\phi_i).
$$

---

## 12. 时间离散与代数方程

采用后向Euler全隐式时间格式：

$$
\frac{\partial\phi}{\partial t}
\approx
\frac{\phi_i^{n+1}-\phi_i^n}{\Delta t}.
$$

生产计算取

$$
\boxed{\Delta t=0.5\ \mathrm{s}}.
$$

对内部控制体积分后得到

$$
S_iV_i
\frac{\phi_i^{n+1}-\phi_i^n}{\Delta t}
=G_{i-1/2}(\phi_{i-1}^{n+1}-\phi_i^{n+1})
+G_{i+1/2}(\phi_{i+1}^{n+1}-\phi_i^{n+1}).
$$

整理为

$$
a_{P,i}\phi_i^{n+1}
-G_{i-1/2}\phi_{i-1}^{n+1}
-G_{i+1/2}\phi_{i+1}^{n+1}
=\frac{S_iV_i}{\Delta t}\phi_i^n,
$$

其中

$$
\boxed{
a_{P,i}
=\frac{S_iV_i}{\Delta t}
+G_{i-1/2}+G_{i+1/2}
}.
$$

有限体积离散后形成严格三对角线性方程组。

---

## 13. 表面串联阻力与边界离散

最外侧控制体中心到药材真实表面存在内部传导或扩散阻力，真实表面到空气之间存在外部对流阻力。

设最外侧控制体中心到表面的距离为

$$
\delta=R-r_N,
$$

表面积为

$$
A_R=2\pi R.
$$

将内部阻力和外部对流阻力串联，可得等效边界导通系数

$$
\boxed{
G_b=
\frac{A_R}
{
\dfrac{\delta}{\Gamma_N}
+\dfrac{1}{\beta}
}
},
$$

其中

$$
\beta=
\begin{cases}
h,&\text{温度方程},\\
k_m,&\text{水分方程}.
\end{cases}
$$

在最外侧控制体的离散方程中，主对角线增加 $G_b$，右端项增加 $G_b\phi_a$。

---

## 14. 真实表面值重构

最外侧控制体中心值不能直接视为真实表面值。由内部通量和表面对流通量连续可得

$$
\frac{\Gamma_N}{\delta}(\phi_s-\phi_N)
=\beta(\phi_a-\phi_s).
$$

整理得到

$$
\boxed{
\phi_s
=
\frac{
\dfrac{\Gamma_N}{\delta}\phi_N+\beta\phi_a
}
{
\dfrac{\Gamma_N}{\delta}+\beta
}
}.
$$

温度和水分均使用这一公式重构 $r=R$ 处的真实表面值。该处理避免了把最外侧控制体中心错误当作药材表面的常见问题。

---

## 15. 非线性Picard迭代

由于物性系数依赖未知的 $T$ 和 $C$，每个时间步需要进行非线性迭代。

在第 $n+1$ 个时间步，首先取

$$
T^{(0)}=T^n,\qquad C^{(0)}=C^n.
$$

第 $k$ 次Picard迭代依次执行：

1. 根据 $C^{(k)}$ 计算

   $$
   \rho^{(k)}=\rho(C^{(k)}),\qquad
   c_p^{(k)}=c_p(C^{(k)}),\qquad
   \lambda^{(k)}=\lambda(C^{(k)}).
   $$

2. 固定上述物性，隐式求解温度方程，得到 $T^{(k+1)}$。

3. 根据当前水分和新温度计算

   $$
   D^{(k)}=D(C^{(k)},T^{(k+1)}).
   $$

4. 固定扩散系数，隐式求解水分方程，得到 $C^{(k+1)}$。

5. 计算迭代误差

   $$
   \varepsilon_T
   =\max_i|T_i^{(k+1)}-T_i^{(k)}|,
   $$

   $$
   \varepsilon_C
   =\max_i|C_i^{(k+1)}-C_i^{(k)}|.
   $$

当

$$
\varepsilon_T<10^{-8}\ ^\circ\mathrm{C}
$$

且

$$
\varepsilon_C<10^{-10}\ \mathrm{kg/kg}
$$

时认为收敛。每个时间步最多迭代30次，本次计算实际最多使用4次。

该迭代顺序属于分块Gauss–Seidel型Picard迭代：先更新温度，再利用新温度更新水分，下一次迭代再将新水分反馈给温度方程。

---

## 16. 三对角方程求解

每个控制体只与左右相邻控制体发生交换，因此温度和水分离散方程均可写为

$$
a_{W,i}\phi_{i-1}+a_{P,i}\phi_i+a_{E,i}\phi_{i+1}=b_i.
$$

程序采用Thomas追赶法求解，计算复杂度约为 $O(N)$。全隐式格式具有较好的数值稳定性，但时间精度为一阶。

---

## 17. 中心点与指定位置结果重构

有限体积未知量位于控制体中心，圆柱轴线 $r=0$ 处没有直接未知量。由中心对称性，中心附近可写成偶函数展开

$$
\phi(r)=\phi(0)+ar^2+O(r^4).
$$

利用靠近中心的两个控制体值可得

$$
\boxed{
\phi(0)
=
\frac{
\phi_0r_1^2-\phi_1r_0^2
}
{r_1^2-r_0^2}
}.
$$

对于内部指定半径，程序采用相邻控制体中心之间的线性插值；对于 $r=R$，采用Robin边界公式重构表面值。

生产模型每 $0.5\ \mathrm{s}$ 推进一次，每 $1\ \mathrm{s}$ 保存一次，并最终输出 $0,0.1,\ldots,2.0\ \mathrm{cm}$ 共21个径向位置的数据。

---

## 18. 数值计算流程

```text
读取附件1
    ↓
检查时间序列单调性、覆盖范围和有限性
    ↓
将空气温度与空气水分线性插值到每1秒
    ↓
建立160个向药材表面加密的径向控制体
    ↓
初始化 T=28 ℃、C=2.55 kg/kg
    ↓
以0.5秒为时间步推进
    ↓
根据当前C更新ρ、cp、λ
    ↓
组装并隐式求解温度方程
    ↓
根据新T和当前C更新D
    ↓
组装并隐式求解水分方程
    ↓
检查Picard迭代收敛
    ↓
重构真实中心值与真实表面值
    ↓
每1秒保存一次结果
    ↓
输出完整时空数据与题目指定时刻表格
    ↓
进行综合加密、守恒、物理范围和单调性检查
```

---

## 19. 结果输出

程序输出：

- [`environment_0_3h_1s_kelvin.csv`](../results/A_problem2_coupled/environment_0_3h_1s_kelvin.csv)：预处理后的烘房环境数据；
- [`temperature_full_1s_0p1cm.csv`](../results/A_problem2_coupled/temperature_full_1s_0p1cm.csv)：温度完整时空结果；
- [`moisture_full_1s_0p1cm.csv`](../results/A_problem2_coupled/moisture_full_1s_0p1cm.csv)：水分完整时空结果；
- [`table3_temperature.csv`](../results/A_problem2_coupled/table3_temperature.csv)：题目要求时刻和半径处的温度；
- [`table4_moisture.csv`](../results/A_problem2_coupled/table4_moisture.csv)：题目要求时刻和半径处的水分；
- [`validation_summary.json`](../results/A_problem2_coupled/validation_summary.json)：数值验证结果；
- [`result2.xlsx`](../results/A_problem2_coupled/result2.xlsx)：第二问汇总工作簿。

---

## 20. 数值验证

### 20.1 综合加密验证

程序使用两套离散尺度：

| 方案 | 控制体数量 | 名义径向尺度 | 时间步长 |
|---|---:|---:|---:|
| 粗计算 | 80 | 0.025 cm | 1.0 s |
| 生产计算 | 160 | 0.0125 cm | 0.5 s |

两套计算结果的最大差异为：

| 指标 | 最大差异 |
|---|---:|
| 完整温度场 | 0.001064 ℃ |
| 完整水分场 | 0.001431 kg/kg |
| 表3温度 | 0.000837 ℃ |
| 表4水分 | 0.000136 kg/kg |

由于空间网格和时间步长被同时加密，该检验应称为“综合加密验证”，不能分别视为独立的网格无关性检验和时间步长无关性检验。

### 20.2 水分守恒

单位轴向长度上的水分守恒关系为

$$
\sum_iV_i(C_i^{\mathrm{end}}-C_i^0)
+\int_0^{t_{\mathrm{end}}}
G_b(C_N-C_a)\,\mathrm{d}t=0.
$$

计算得到相对守恒残差

$$
3.96\times10^{-13},
$$

说明有限体积离散在数学意义上严格保持水分守恒。

### 20.3 数值范围与单调性

计算过程中：

$$
28.0000\le T\le49.9664\ ^\circ\mathrm{C},
$$

$$
1.0081\le C\le2.5500\ \mathrm{kg/kg}.
$$

未出现负水分浓度、温度异常或显著径向次序错误。附件1平台阶段的微小温度波动会使局部径向温差短暂达到千分之几摄氏度，因此代码对温度径向次序检查使用 $0.01\ ^\circ\mathrm{C}$ 的物理容差。

---

## 21. 3小时结果摘要

在 $t=10800\ \mathrm{s}$ 时：

| 指标 | 数值 |
|---|---:|
| 烘房空气温度 | 50.1950 ℃ |
| 药材中心温度 | 49.8495 ℃ |
| 药材表面温度 | 49.9664 ℃ |
| 烘房空气水分 | 0.04977 kg/kg |
| 药材中心水分 | 1.7662 kg/kg |
| 药材表面水分 | 1.0081 kg/kg |
| 有限体积加权平均水分 | 1.3825 kg/kg |

温度中心—表面差已经很小，而水分中心—表面差仍然显著，说明传热过程比水分扩散更快达到近似均匀状态，后期干燥主要受内部水分迁移限制。

---

## 22. 当前模型的局限性

### 22.1 未考虑蒸发潜热

温度方程中没有蒸发吸热项，例如

$$
-L_v\dot m_v.
$$

因此在蒸发较强的阶段，模型可能高估药材温度。该项是后续增强模型中最重要的补充之一。

### 22.2 空气水分与药材水分被直接比较

代码使用

$$
J_C=k_m(C_s-C_a).
$$

如果附件1中的空气水分表示“kg水/kg干空气”，而药材水分表示“kg水/kg干药材”，则两者虽然形式上都是kg/kg，却不是相同的热力学状态变量。

更严格的处理方式是通过相对湿度、水活度或吸附等温线获得平衡含水率：

$$
C_{\mathrm{eq}}=f(T_a,\mathrm{RH}),
$$

再建立

$$
J_C=k_m(C_s-C_{\mathrm{eq}}).
$$

因此，当前 $C_a$ 应解释为经验性的外部水分势边界。

### 22.3 水分守恒量并非严格的真实水质量

当前水分方程守恒的是 $\int C\,\mathrm{d}V$。若需要计算失水质量，应考虑干物质密度：

$$
\frac{\partial(\rho_dC)}{\partial t}
=\nabla\cdot(\rho_dD\nabla C).
$$

### 22.4 采用表观体积热容形式

程序采用

$$
\rho(C)c_p(C)\frac{\partial T}{\partial t},
$$

没有直接采用严格的焓守恒形式

$$
\frac{\partial[\rho(C)c_p(C)T]}{\partial t}.
$$

因此忽略了密度和比热随水分变化而产生的附加时间导数项。

### 22.5 外部传递系数固定

换热系数 $h$ 和传质系数 $k_m$ 在程序中均为常数。实际情况下，它们可能随空气速度、温度、湿度和药材表面状态变化。

### 22.6 未考虑半径收缩

第二问使用固定半径网格。尺寸变化和移动边界将在后续考虑收缩的问题中单独处理，不能直接沿用本问固定区域的离散坐标。

### 22.7 缺少独立热量守恒检查

当前程序对水分进行了守恒验证，但没有建立完整热量收支。后续加入潜热后，应检查

$$
Q_{\mathrm{conv}}
=\Delta H_{\mathrm{solid}}+Q_{\mathrm{evap}}.
$$

---

## 23. 论文中的模型定位建议

可在论文中将第二问模型概括为：

> 将药材近似为固定半径的长圆柱体，建立一维径向非线性水热耦合模型。温度场满足水分相关热物性下的变系数热传导方程，水分场满足温度—水分相关有效扩散系数下的Fick扩散方程；药材表面采用随烘房环境变化的对流换热与对流传质边界。模型使用向表面加密的有限体积网格、后向Euler时间格式和Picard迭代求解。第二问暂不考虑药材收缩、蒸发潜热及辐射换热，因此水热耦合主要通过材料热物性和有效水分扩散系数实现。

---

## 24. 代码与公式对应关系

| 建模环节 | 对应程序位置 |
|---|---|
| 基本参数和初值 | [`a2_coupled_fvm.py`](../code/a2_coupled_fvm.py#L23) |
| 附件1读取与线性插值 | [`load_and_preprocess_environment`](../code/a2_coupled_fvm.py#L81) |
| 密度、比热容、导热系数 | [`density_kg_m3` 等函数](../code/a2_coupled_fvm.py#L116) |
| 有效水分扩散系数 | [`moisture_diffusivity_m2_s`](../code/a2_coupled_fvm.py#L128) |
| 非均匀径向网格 | [`make_grid`](../code/a2_coupled_fvm.py#L143) |
| 三对角追赶法 | [`solve_tridiagonal`](../code/a2_coupled_fvm.py#L158) |
| 内部界面导通系数 | [`internal_conductances`](../code/a2_coupled_fvm.py#L182) |
| 有限体积方程组装 | [`build_diffusion_system`](../code/a2_coupled_fvm.py#L192) |
| 真实表面值重构 | [`reconstruct_surface_value`](../code/a2_coupled_fvm.py#L222) |
| 时间推进和Picard迭代 | [`simulate_coupled`](../code/a2_coupled_fvm.py#L235) |
| 中心值和指定半径插值 | [`sample_result`](../code/a2_coupled_fvm.py#L392) |
| 数值验证 | [`build_validation`](../code/a2_coupled_fvm.py#L470) |
| 粗细两套计算与结果输出 | [`main`](../code/a2_coupled_fvm.py#L620) |

