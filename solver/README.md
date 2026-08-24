# 使用说明

## 1. 环境

程序只要求 Python 3.9+ 和 NumPy：

```powershell
pip install numpy
```

无需安装 SciPy、pandas、openpyxl；官方 Excel 模板由标准库直接填写。

## 2. 输入坐标

复制并修改 `config_example.json`：

- `drones`：无人机编号及三维初始坐标；
- `missiles`：导弹编号及三维初始坐标；
- `false_target`：导弹飞向的假目标坐标；
- `target.base_center`：真目标下底面中心；
- `target.radius`、`target.height`：真目标圆柱半径和高度；
- `visibility_mode`：`cylinder` 为严格圆柱遮蔽，`center` 为目标中心视线遮蔽。

题目原始坐标已经全部写入示例配置，因此不修改配置即可直接求题。

## 3. 运行

在项目根目录执行：

```powershell
python solver/smoke_screen_optimizer.py --scenario 1 --dt 0.01
python solver/smoke_screen_optimizer.py --scenario 2 --generations 200 --population 120
python solver/smoke_screen_optimizer.py --scenario 3 --generations 300 --population 160
python solver/smoke_screen_optimizer.py --scenario 4 --generations 400 --population 200
python solver/smoke_screen_optimizer.py --scenario 5 --generations 800 --population 300
```

一次求全部问题可使用：

```powershell
python solver/smoke_screen_optimizer.py --scenario all --output outputs/final
```

常用参数：

- `--config`：自定义 JSON 输入文件；
- `--output`：输出目录；
- `--dt`：时间离散步长，默认 0.10 s；最终计算建议用 0.02--0.05 s 复核；
- `--seed`：随机种子；
- `--generations`、`--population`：差分进化迭代数和种群数；
- `--polish-steps`：全局搜索后的局部精修次数。

问题 5 的变量最多，推荐使用不同 `--seed` 独立运行 5--10 次，比较 `problem5_summary.json` 中的 `total_union_seconds`，保留最好结果。

## 4. 输出

每个问题均输出：

- `problemN_details.csv`：包含航向、速度、投放点、起爆点、投放/起爆时刻、单弹遮蔽时长和有效区间；
- `problemN_summary.json`：包含各导弹遮蔽区间的并集及总时长。

问题 3--5 还会生成与附件格式一致的 `result1.xlsx`、`result2.xlsx`、`result3.xlsx`。

航向角以 $x$ 轴正方向为 0 度，逆时针为正，输出范围为 0--360 度。

## 5. 快速校验

题目问题 1 使用 `center` 判据、`--dt 0.001` 时，程序计算约 1.435 s；使用默认的严格 `cylinder` 判据时约 1.39 s。两者差异来自是否要求整个圆柱目标均被遮挡，论文中应明确采用的判据。
