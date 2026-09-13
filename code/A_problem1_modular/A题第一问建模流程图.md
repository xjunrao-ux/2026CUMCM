# A题第一问建模流程图

下面的 Mermaid 源码可直接在支持 Mermaid 的 Markdown 编辑器、Typora、Obsidian、GitHub 或 Mermaid Live Editor 中继续编辑。

```mermaid
flowchart TB
    START(["开始：求解前30 min药材内部温度场与水分场"])

    subgraph DATA[阶段一：附件1数据预处理]
        D1["读取附件1原始观测数据<br/>时间、烘房温度、烘房水分浓度"]
        D2{"数据有效性检查"}
        D3["截取0～1800 s<br/>保留31个观测节点"]
        D4["采用分段线性插值 numpy.interp<br/>生成1 s间隔环境边界"]
        D5[("输出公共输入数据<br/>attachment1_linear_1s.csv")]
        D1 --> D2
        D2 -->|时间严格递增、无缺失、覆盖完整| D3
        D2 -->|不满足| ERR["报错并停止"]
        D3 --> D4 --> D5
    end

    subgraph HEAT[阶段二A：径向温度场有限体积模型]
        T1["建立一维轴对称圆柱导热模型<br/>半径R=0.02 m，初温28 ℃"]
        T2["控制方程<br/>ρcp·∂T/∂t = 1/r·∂/∂r (kr·∂T/∂r)"]
        T3["边界条件<br/>中心对称；表面对流换热"]
        T4["节点中心有限体积离散<br/>均匀径向网格 Δr=0.025 cm"]
        T5["后向欧拉时间离散<br/>Δt=0.25 s"]
        T6["预分解三对角矩阵<br/>Thomas算法逐步求解"]
        T7[("每1 s保存无舍入温度场")]
        T1 --> T2 --> T3 --> T4 --> T5 --> T6 --> T7
    end

    subgraph MOISTURE[阶段二B：径向水分场有限体积模型]
        M1["建立一维轴对称非线性Fick扩散模型<br/>初始水分浓度C₀=2.55 kg/kg"]
        M2["非线性扩散系数<br/>D(C)=7×10⁻⁹ exp(-0.89/C)"]
        M3["边界条件<br/>中心对称；表面对流传质"]
        M4["二次映射生成表面加密网格<br/>名义Δr=0.0125 cm"]
        M5["半控制体扩散阻力<br/>与空气侧传质阻力串联"]
        M6["后向欧拉时间离散<br/>Δt=0.125 s"]
        M7["每个时间步进行Picard迭代<br/>更新D(C)并用Thomas算法求解"]
        M8{"最大变化是否小于<br/>10⁻¹⁰?"}
        M9["重建真实表面水分浓度<br/>并累计表面流出量"]
        M10[("每1 s保存无舍入水分场")]
        M1 --> M2 --> M3 --> M4 --> M5 --> M6 --> M7 --> M8
        M8 -->|否，继续迭代| M7
        M8 -->|是| M9 --> M10
        M8 -->|超过50次| ERR
    end

    subgraph MERGE[阶段三：同步汇流、采样与物理量计算]
        C1["温度场与水分场同步联合计算<br/>共享几何、时间范围和环境数据"]
        C2["当前题设未给出潜热或温度相关扩散项<br/>因此两方程无直接交叉反馈"]
        C3["插值到统一输出网格<br/>时间1 s、半径0.1 cm"]
        C4["提取题目指定位置和时刻<br/>r=0、0.5、1.0、1.5、2.0 cm<br/>t=100、300、600、900、1200、1500、1800 s"]
        C5["计算表面热流密度<br/>q=h·(T∞−Ts)"]
        C6["计算向外水分通量<br/>J=hm·(Cs−C∞)"]
        C7[("输出完整温度/水分场、表1、表2<br/>表面通量与无舍入模型状态")]
        C1 --> C2 --> C3
        C3 --> C4 --> C7
        C3 --> C5 --> C7
        C3 --> C6 --> C7
    end

    subgraph VALIDATION[阶段四：模型验证]
        V1["水分粗网格复算<br/>Δr=0.025 cm，Δt=0.25 s"]
        V2["粗细网格结果对比<br/>检查网格收敛性"]
        V3["守恒检查<br/>温度能量平衡与水分收支平衡"]
        V4["物理合理性检查<br/>范围、径向次序、时间单调性、环境边界"]
        V5{"所有检查是否通过?"}
        V6[("输出 validation_summary.json")]
        V1 --> V2 --> V3 --> V4 --> V5
        V5 -->|通过| V6
        V5 -->|未通过| REVIEW["检查网格、步长、参数与边界数据"]
    end

    subgraph VISUAL[阶段五：结果可视化]
        P1["径向温度与水分分布图"]
        P2["温度时空热力图"]
        P3["水分时空热力图"]
        P4["中心—表面—烘房响应图"]
        P5["表面热流密度与水分通量图"]
        POUT[("PNG图件")]
        P1 --> POUT
        P2 --> POUT
        P3 --> POUT
        P4 --> POUT
        P5 --> POUT
    end

    subgraph ANALYSIS[阶段六：可靠性与扩展分析]
        S1["单因素灵敏度分析 OAT"]
        S2["7个参数分别取基准值0.50～1.50倍<br/>步长0.05，其余参数保持不变"]
        S3["计算30 min中心、表面和平均响应<br/>以温升与失水量衡量相对变化"]
        S4["确定5%预警区间、10%显著区间<br/>并绘制响应曲线与龙卷风图"]

        R1["边界构造稳健性检验"]
        R2["对比分段线性、PCHIP<br/>与端点约束拉伸指数"]
        R3["比较节点误差、留一预测误差<br/>单调违例、局部过冲与斜率连续性"]
        R4["输出三方法对比数据、指标表<br/>拟合参数和可视化图片"]

        S1 --> S2 --> S3 --> S4
        R1 --> R2 --> R3 --> R4
    end

    END(["形成第一问的数值结果、验证证据<br/>敏感性结论与稳健性结论"])

    START --> D1
    D5 --> T1
    D5 --> M1
    T7 --> C1
    M10 --> C1
    M10 --> V1
    C7 --> V3
    C7 --> P1
    C7 --> P2
    C7 --> P3
    C7 --> P4
    C7 --> P5
    D5 --> S1
    D3 --> R1
    V6 --> END
    POUT --> END
    S4 --> END
    R4 --> END

    classDef input fill:#EAF2F8,stroke:#2E6F9E,color:#17324D,stroke-width:1.2px;
    classDef heat fill:#FDEDEC,stroke:#C6534C,color:#5C2420,stroke-width:1.2px;
    classDef moisture fill:#E8F5EF,stroke:#3F7463,color:#1F493C,stroke-width:1.2px;
    classDef process fill:#F5F5F5,stroke:#667085,color:#20242B,stroke-width:1px;
    classDef output fill:#FFF4E5,stroke:#B7791F,color:#5C3B0B,stroke-width:1.2px;
    classDef decision fill:#FFF9DB,stroke:#9B7B00,color:#4D3E00,stroke-width:1.2px;
    classDef warning fill:#FCE8E6,stroke:#B3261E,color:#6D1611,stroke-width:1.2px;

    class D1,D3,D4,D5 input;
    class T1,T2,T3,T4,T5,T6,T7 heat;
    class M1,M2,M3,M4,M5,M6,M7,M9,M10 moisture;
    class C1,C2,C3,C4,C5,C6,V1,V2,V3,V4,P1,P2,P3,P4,P5,S1,S2,S3,S4,R1,R2,R3,R4 process;
    class C7,V6,POUT,END output;
    class D2,M8,V5 decision;
    class ERR,REVIEW warning;
```

## 阅读顺序

1. 附件1观测数据先经过完整性检查，再转换成统一的1秒分段线性边界。
2. 温度场和水分场分别求解；两者共享环境数据，但当前题设下没有直接交叉反馈项。
3. 两个场汇合后，统一提取题目表格、计算表面热流和水分通量。
4. 网格收敛性、守恒性和物理单调性共同构成模型验证。
5. 主结果进入可视化；同一边界数据还进入单因素灵敏度分析。
6. 原始观测节点进入三种拟合方法对比，用于评价边界构造方法的稳健性。

## 与五个程序的对应关系

| 流程阶段 | 对应程序 |
|---|---|
| 阶段一：数据预处理 | `data_preprocessing.py` |
| 阶段二至四：有限体积求解、结果采样与验证 | `fvm_model.py` |
| 阶段五：结果可视化 | `visualization.py` |
| 阶段六：参数灵敏度 | `sensitivity_analysis.py` |
| 阶段六：边界构造稳健性 | `robustness_check.py` |

> 注：代码与图中使用的标准算法名称为 PCHIP（Piecewise Cubic Hermite Interpolating Polynomial）。
