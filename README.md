# 2026CUMCM

2026 年高教社杯全国大学生数学建模竞赛 **A 题：药材的烘干问题** 项目。

题目背景：干燥是决定中药材成品品质的关键工序，热风烘干包含预热平衡与恒温干燥两个阶段。本项目通过数理分析与数值仿真（有限体积法 FVM 求解圆柱内热湿耦合 PDE）建立干燥规律模型。

**四问脉络**：

1. **问题1** — 预热平衡阶段（30 分钟）药材温度场与水分浓度场
2. **问题2** — 整个烘干过程（2–3 天），给出前 3 小时结果
3. **问题3** — 全程干燥时间：4 小时后的边界假设、Huber/OU 稳健性、蒸发潜热
4. **问题4** — 药材收缩：移动边界 + 表面蒸发潜热

---

## 目录架构

```
2026CUMCM/
├── A题.pdf                 # 题目原文
├── AGENT.md                # AI 协作工作说明
├── 比赛题目/CUMCM2026Problems/  # A–E 题及附件（附件1=烘房环境数据、附件2=物性参数、附件3=result1-4.xlsx 提交模板）
├── 规则文件/               # 竞赛通知、论文格式规范、AI 使用规定等 PDF
├── code/                   # 全部建模与计算代码
├── results/                # 各题各版本的完整数值输出（按问题分子目录）
├── result/                 # 较新一批的最终输出（问题3二维、问题4收缩/蒸汽对比）
├── picture/                # 所有可视化图片（按问题分子目录 + 数据可视化/论文图）
├── docs/                   # 建模过程文档 + draw.io 流程图
├── article/                # 论文详稿（A题论文_模型建立详稿.md）
├── 数据清洗/               # 清洗后的附件1/附件2 xlsx + 环境数据 CSV
├── @@整理版/               # 整理后的"提交版"：结果/代码、可视化、数据清洗、@@提交版/支撑材料/submission（最终提交的4个 xlsx）
├── 参考文献/               # 水蛭烘干工艺论文等
├── 参考资料文档/           # AI 工具提示词、往届论文等
├── 支撑材料/               # 支撑材料
├── processing/             # 处理中间产物
└── tmp/                    # 调试/验证用的临时文件（校验输出、node_modules、草稿等）
```

---

## code/ 文件说明

### 问题1（预热平衡阶段）

| 文件 | 作用 |
|---|---|
| [a1_temperature_fvm.py](code/a1_temperature_fvm.py) | 问题1径向温度场 FVM 求解 |
| [a1_coupled_fvm.py](code/a1_coupled_fvm.py) | 问题1温度+水分浓度耦合 FVM，输出 result1.xlsx |
| [a1_parameter_sensitivity.py](code/a1_parameter_sensitivity.py) | 问题1单因素参数灵敏度分析 |
| [attachment1_interpolation_compare.py](code/attachment1_interpolation_compare.py) | 附件1数据的1秒分段线性 vs PCHIP 插值对比 |
| [attachment1_three_fit_compare.py](code/attachment1_three_fit_compare.py) | 附件1前1800s三种边界曲线构造方法对比 |
| [A_problem1_modular/](code/A_problem1_modular/) | 问题1的**模块化重构版**：data_preprocessing（数据预处理）、fvm_model（模型）、robustness_check（稳健性）、sensitivity_analysis（灵敏度）、visualization（可视化），另附建模流程图.md |

### 问题2（全程烘干）

| 文件 | 作用 |
|---|---|
| [a2_coupled_fvm.py](code/a2_coupled_fvm.py) | 问题2全程非线性热湿耦合 FVM，输出 result2.xlsx |
| [a2_visualization.py](code/a2_visualization.py) | 问题2结果可视化 |

### 问题3（干燥时间）

| 文件 | 作用 |
|---|---|
| [a3_drying_time_fvm.py](code/a3_drying_time_fvm.py) | 问题3全程干燥时间耦合 FVM（基础1D版） |
| [a3_drying_time_fvm_2d.py](code/a3_drying_time_fvm_2d.py) | 考虑**两个端面暴露**的轴对称2D水热耦合模型 |
| [a3_dimension2.py](code/a3_dimension2.py) | 固定几何圆柱的2D轴对称热湿耦合 FVM |
| [a3_drying_time_fvm_huber_ou.py](code/a3_drying_time_fvm_huber_ou.py) | Huber 稳健标定 + OU 随机边界驱动的干燥模型 |
| [a3_drying_time_fvm_steam.py](code/a3_drying_time_fvm_steam.py) | 计入表面蒸发潜热的干燥模型 |
| [a3_ou_robustness.py](code/a3_ou_robustness.py) | OU 随机过程稳健性分析（恒值均值 vs OU 随机做法） |
| [a3_sensitivity_analysis.py](code/a3_sensitivity_analysis.py) | 4小时后边界假设的灵敏度分析 |
| [a3_visualization.py](code/a3_visualization.py) | 问题3全程温度/水分可视化 |

### 问题4（收缩）

| 文件 | 作用 |
|---|---|
| [a4_shrinkage_fvm.py](code/a4_shrinkage_fvm.py) | 问题4移动半径（收缩）热湿耦合 FVM，输出 result4.xlsx |
| [a4_shrinkage_fvm_steam.py](code/a4_shrinkage_fvm_steam.py) | 收缩 + 蒸发潜热的移动半径 FVM |
| [a4_ou_shrinkage_fvm.py](code/a4_ou_shrinkage_fvm.py) | Huber-OU 随机烘房边界驱动的收缩 FVM |
| [a4_visualization.py](code/a4_visualization.py) | 问题4论文级可视化 |
| [a4_steam_comparison_visualization.py](code/a4_steam_comparison_visualization.py) | 蒸发潜热模型 vs 常规模型的对比可视化 |

### 工具类脚本

| 文件 | 作用 |
|---|---|
| [generate_a_flowcharts.py](code/generate_a_flowcharts.py) | 生成第一版 draw.io 流程图及 PNG 预览 |
| [generate_a_physical_flowcharts.py](code/generate_a_physical_flowcharts.py) | 生成四问黑白物理模型流程图 |
| [csv_to_result_xlsx.py](code/csv_to_result_xlsx.py) | 把各题结果 CSV 转成符合附件3模板格式的 result1-4.xlsx |
| [绘图代码.py](code/绘图代码.py) | 绘图代码合集 |
| `*.cjs`（6个） | Node.js 脚本，用 `@oai/artifact-tool` 库把 CSV 构建成带格式的 xlsx 工作簿：build_a3_dimension2_workbook、build_result3_from_csv、build_result3_steam_from_csv、a4_ou_build_workbooks、a4_steam_build_workbooks、update_ou_boundary_workbook |

---

## result/ 与 results/ 说明

两个目录都是模型输出：**results/ 是按问题分目录的完整历史输出**，**result/ 是最新一批关键输出**。

### result/（3个子目录）

- **A_problem3_2d_exposed/** — 问题3二维端面暴露模型：2D场数据（npz）、中平面温度/水分 CSV、table5、`result3_2d_exposed.xlsx`、summary.json
- **A_q4_shrinkage/** — 问题4收缩模型：table6 水分、`result4_payload.json`、验证摘要
- **A_q4_steam_comparison/** — 问题4蒸汽对比：comparison_metrics.json、table6_comparison.csv

### results/（按问题划分）

每个问题子目录的典型构成：**完整时间序列 CSV（水分/温度）+ 论文表格 CSV（table1-6）+ `resultN.xlsx`（提交格式）+ `validation_summary.json`（模型自检验证报告）**。

| 子目录 | 内容 |
|---|---|
| A_problem1_coupled / A_problem1_temperature | 问题1耦合版/纯温度版：table1温度、table2水分、result1.xlsx |
| A_problem1_interpolation | 附件1三种插值方法对比指标与参数 |
| A_problem1_modular | 模块化重构版输出 |
| A_problem1_sensitivity | 参数灵敏度扫描与阈值表 |
| A_problem2_coupled | 问题2：table3温度、table4水分、result2.xlsx、烘房环境CSV |
| A_problem3_boundary_sensitivity | 4h后边界拟合灵敏度案例 |
| A_problem3_dimension2 | 二维模型：中平面/轴向场 CSV、table5 |
| A_problem3_drying_time | 问题3基础版：table5水分、result3.xlsx |
| A_problem3_drying_time_huber_ou | OU边界版：ou_boundary_runs.xlsx、ou_parameters.json、result3.xlsx |
| A_problem3_drying_time_steam | 蒸发潜热版：result3_steam_fixed_rho.xlsx、表面能量CSV |
| A_problem3_ou_robustness | OU稳健性：分析md、轨迹npz、summary |
| A_problem3_sensitivity | 问题3灵敏度：分析md、场景与摘要 |
| A_problem4_shrinkage | 问题4基础版：table6水分、result4.xlsx |
| A_problem4_shrinkage_huber_ou | OU版问题4：ou_parameters、result4.xlsx |
| A_problem4_shrinkage_steam | 蒸发潜热版：result4_steam.xlsx、q4输入输出CSV |
| submission/ | **最终提交的 result1-4.xlsx 四个文件** |

---

## 交付说明

`@@整理版/` 是整理好的交付目录：

- `@@整理版/结果/` — 按"第一问~第四问"整理的代码
- `@@整理版/可视化/` — 各问可视化输出
- `@@整理版/数据清洗/` — 数据预处理结果
- `@@整理版/@@提交版/支撑材料/submission/` — 最终提交的 4 个 xlsx

整体上，每个问题都开发了基础版、稳健性（OU）版、蒸发潜热（steam）版等多个模型变体，每个变体留有独立的输出目录与验证报告，最后通过 submission 汇总提交。
