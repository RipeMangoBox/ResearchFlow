---
title: 'Modeling Sparse and Bursty Vulnerability Sightings: Forecasting Under Data Constraints'
type: paper
paper_level: B
venue: arXiv
year: 2026
paper_link: https://arxiv.org/abs/2604.16038
aliases:
- 稀疏突发漏洞观测计数预测：泊松回归与指数衰减
- MSBVSF
- 漏洞sighting数据的稀疏突发特性从根本上违反了SARIMAX的平
---

# Modeling Sparse and Bursty Vulnerability Sightings: Forecasting Under Data Constraints

[Paper](https://arxiv.org/abs/2604.16038)

**Topics**: [[T__Time_Series_Forecasting]] | **Datasets**: vulnerability sighting forecasting

> [!tip] 核心洞察
> 漏洞sighting数据的稀疏突发特性从根本上违反了SARIMAX的平稳性假设，强行套用只会放大噪声。真正的改变在于将问题重新定义为离散计数事件建模（Poisson回归），而非连续时序预测——这与数据的生成机制更匹配。周聚合进一步缓解了零值主导的稀疏性。VLAI严重性分数作为语义侧信号的引入，是将上游文本理解模型与下游时序预测打通的尝试，但实际增益有限，说明严重性与sighting量之间的关联比预期弱。

| 中文题名 | 稀疏突发漏洞观测计数预测：泊松回归与指数衰减 |
| 英文题名 | Modeling Sparse and Bursty Vulnerability Sightings: Forecasting Under Data Constraints |
| 会议/期刊 | arXiv (Cornell University) (预印本) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.16038) · [Code] · [Project] |
| 主要任务 | vulnerability sighting forecasting, time-series forecasting, vulnerability intelligence |
| 主要 baseline | SARIMAX, SARIMAX with log(x+1), SARIMAX with VLAI severity exogenous variables |

> [!abstract] 因为「漏洞观测数据具有极端稀疏性、突发爆发性和短历史序列三大特征，传统SARIMAX产生负值预测与不现实置信区间」，作者在「SARIMAX」基础上改了「以泊松回归替换高斯似然、引入VLAI严重度外生变量、新增指数衰减非参数替代」，在「vulnerability sighting forecasting benchmark」上取得「消除负值预测、更合理的计数不确定性量化、无需长历史序列的短期预测能力」

- **预测稳定性**：泊松回归消除SARIMAX的负值预测问题（Figure 2, Figure 8对比）
- **数据需求**：指数衰减函数 y(t) = a·e^(-bt) + c 无需长期历史序列即可运营部署
- **语义增强**：VLAI RoBERTa严重度分数作为外生变量提升解释力

## 背景与动机

在网络安全运营中，安全团队需要预测特定CVE（Common Vulnerabilities and Exposures）在未来几周内会收到多少公开关注或利用迹象——这些被称为"vulnerability sightings"。例如，CVE-2025-61932可能在披露首周仅有2条观测记录，第三周突然激增至47条，随后数周归零。这种**极端稀疏（大量零值）、突发爆发（短期激增）、历史极短（往往不足10周）**的数据形态，使传统时间序列方法陷入困境。

现有方法如何应对？**SARIMAX**（Seasonal AutoRegressive Integrated Moving Average with eXogenous variables）作为经典时序基线，假设高斯误差分布，通过差分和季节性参数捕捉趋势；**SARIMAX with log(x+1)** 尝试通过对数变换将计数映射到实数域，但核心仍维持高斯似然；**VLAI**（同作者前置工作）利用RoBERTa从漏洞描述文本提取严重度分数，但仅用于静态分类而非动态预测。

这些方法为何不足？SARIMAX的**高斯假设与离散计数数据的本质矛盾**：当观测值接近零时，模型被迫允许负值预测（Figure 2, Figure 8中可见不现实的置信区间下探至负值区域），且季节性参数需要长历史序列才能可靠估计——而多数CVE在披露后数周内即失去关注，根本不具备"长历史"。log变换虽缓解右偏，却未改变似然函数的根本性质，且对零值膨胀（zero-inflation）处理粗糙。VLAI的严重度信息则未被有效整合进预测流程。

本文的核心动机即在于：**为数据约束下的离散事件预测，设计适配计数本质且能利用文本语义信息的轻量方法族**。

## 核心创新

核心洞察：**泊松似然天然约束预测空间于非负整数，因为计数数据的生成机制是事件到达过程而非高斯噪声叠加，从而使SARIMAX的负值预测问题从根本上消除成为可能**；同时，**指数衰减的三参数结构仅需初始观测即可拟合，因为运营场景中"历史长度"本身就是稀缺资源，从而使无长期序列条件下的短期活动估计成为可能**。

| 维度 | Baseline (SARIMAX) | 本文 |
|:---|:---|:---|
| **概率假设** | Gaussian likelihood: y ~ N(μ, σ²) | Poisson likelihood: y ~ Pois(λ)，λ = exp(Xβ) |
| **预测空间** | 实数域 ℝ（允许负值） | 非负整数 ℕ⁺（物理可解释） |
| **数据需求** | 需长历史序列拟合季节性参数 | 指数衰减仅需初始计数与 timestamps |
| **语义利用** | 无文本信息 | VLAI严重度分数作为外生解释变量 |
| **不确定性量化** | 高斯置信区间在稀疏区不可靠 | Poisson固有方差-均值关系更适配计数波动 |

## 整体框架


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/78fa5a20-2708-402a-b34a-bb7a6beaeea6/figures/Figure_2.png)
*Figure 2: Figure 1: Observed sightings over time for CVE-2025-61932*



数据流从原始CVE观测记录开始，经四个核心模块处理：

1. **周聚合模块（weekly aggregation）**：输入为原始漏洞 sightings（带时间戳的离散事件），输出为按CVE-ID聚合的周计数时间序列。此模块为标准预处理，无新颖性。

2. **VLAI严重度评分模块（VLAI severity scoring）**：输入为漏洞描述文本，经预训练RoBERTa模型推理，输出[0,1]区间的严重度分数。该模型为同作者前置工作VLAI，本文将其从"分类器"重新定位为"特征提取器"。

3. **泊松回归预测器（Poisson regression forecaster）**：输入为周计数序列与VLAI严重度外生变量，通过广义线性模型（GLM）以Poisson似然进行参数估计，输出带置信区间的非负计数预测。**此模块替换SARIMAX的核心推理策略**。

4. **指数衰减预测器（exponential decay forecaster）**：输入为初始观测计数与对应时间戳，直接拟合三参数函数 y(t) = a·e^(-bt) + c，输出短期活动水平估计。**此模块为无历史序列场景的全新替代方案**。

```
原始CVE观测记录
    ↓
[周聚合计数] ─────────────────────────┐
    ↓                                  │
[VLAI文本分析] → [严重度分数 s ∈ [0,1]] │
    ↓                                  ↓
┌─────────────────────────────────────────────┐
│  多模型并行预测框架                           │
│  ├─ SARIMAX（高斯误差，基线对照）              │
│  ├─ SARIMAX + log(x+1)（对数变换基线）        │
│  ├─ SARIMAX + VLAI外生变量（语义增强基线）     │
│  ├─ ★ 泊松回归 + VLAI外生变量（核心创新）      │
│  └─ ★ 指数衰减 y(t)=a·e^(-bt)+c（无历史替代） │
└─────────────────────────────────────────────┘
    ↓
预测结果对比与不确定性量化
```

## 核心模块与公式推导

### 模块 1: log1p 归一化（数据预处理层）

**直觉**：计数数据大量零值、右偏分布，需映射到实数域以适配传统模型，同时保持零值语义不变。

**Baseline 公式** (SARIMAX with log(x+1) transformation):
$$x' = \log(x + 1)$$
符号: $x$ = 原始周计数（非负整数），$x'$ = 变换后实数值。关键性质：$\log(0+1) = 0$，零值保持不变；$x > 0$ 时压缩右尾。

**变化点**：此变换仅为数据管道修饰，未触及SARIMAX的高斯似然核心——当预测值经指数回传 $x = \exp(x') - 1$ 时，中间步骤的线性预测仍可能产生负值 $x'$。

**本文处理**：泊松回归直接建模原始计数，无需此变换；指数衰减同样直接拟合原始计数时序。

---

### 模块 2: 泊松回归预测器（核心替换模块，对应框架图位置 3）

**直觉**：漏洞 sightings 是事件计数，其生成机制为泊松过程——单位时间内事件到达数服从Poisson分布，均值由外生因素（如严重度）解释。

**Baseline 公式** (SARIMAX):
$$y_t \sim \mathcal{N}(\mu_t, \sigma^2), \quad \mu_t = c + \sum_{i=1}^{p}\phi_i y_{t-i} + \sum_{j=1}^{q}\theta_j \epsilon_{t-j} + \sum_{k=1}^{K}\gamma_k x_{k,t}$$
符号: $\phi_i$ = AR系数，$\theta_j$ = MA系数，$\gamma_k$ = 外生变量系数，$x_{k,t}$ = VLAI严重度等外生变量。

**变化点**：高斯假设导致① 支持域为全体实数，负值预测无物理意义；② 方差恒定假设与计数的"均值=方差"特性矛盾；③ 在稀疏区（$y_t \approx 0$），高斯近似极差。

**本文公式（推导）**:
$$\text{Step 1}: \lambda_t = \exp\left(\beta_0 + \sum_{k=1}^{K}\beta_k x_{k,t} + \beta_{VLAI} \cdot s_{CVE}\right) \quad \text{将对数均值链接到线性预测器，确保 } \lambda_t > 0$$
$$\text{Step 2}: y_t \sim \text{Pois}(\lambda_t), \quad P(Y=y_t) = \frac{\lambda_t^{y_t} e^{-\lambda_t}}{y_t!} \quad \text{Poisson似然天然约束非负整数支持域}$$
$$\text{最终}: \mathcal{L}(\boldsymbol{\beta}) = \sum_{t=1}^{T}\left[y_t \log\lambda_t - \lambda_t - \log(y_t!)\right] \quad \text{对数似然用于参数估计}$$

**对应消融**：移除Poisson似然回退至高斯 → 负值预测与不现实置信区间重现（实验部分定性验证）。

---

### 模块 3: 指数衰减预测器（无历史替代模块，对应框架图位置 4）

**直觉**：许多CVE在披露后呈现"首周高峰、随后快速衰减"模式，三参数指数函数可捕捉此规律，无需历史序列估计季节性。

**Baseline 公式**：SARIMAX需满足 $T \geq 2(p+q+P+Q)$ 的最小样本要求，对短序列不可行。

**变化点**：运营场景中"无足够历史"是常态而非例外，需要完全脱离时间序列依赖结构的替代方案。

**本文公式（推导）**:
$$\text{Step 1}: y(t) = a \cdot e^{-bt} + c \quad \text{假设活动水平随时间指数衰减至基线}$$
$$\text{Step 2}: \text{给定观测 } \{(t_i, y_i)\}_{i=1}^{n} \text{（通常 } n \leq 5\text{），非线性最小二乘拟合 } (a, b, c)$$
$$\text{最终}: \hat{y}(t_{new}) = \hat{a} \cdot e^{-\hat{b} \cdot t_{new}} + \hat{c}, \quad t_{new} > \max(t_i)$$

符号: $a$ = 初始活动幅度（$a+c$ 为 $t=0$ 外推值），$b$ = 衰减速率（$b>0$ 确保衰减），$c$ = 渐近基线水平（长期稳态关注度）。

**替代形式**（逻辑斯蒂增长，用于"先增后饱和"模式）:
$$y(t) = \frac{L}{1 + e^{-k(t - t_0)}}$$
其中 $L$ = 承载容量上限，$k$ = 增长率，$t_0$ = 增长中点位置。

**对应消融**：移除指数衰减选项 → 丧失无历史数据时的短期预测能力（实验部分定性验证）。

## 实验与分析

**主结果对比**（基于论文Figure 2-Figure 13的定性可视化评估，**缺乏定量数值指标**）：

| 方法 | 预测稳定性 | 可解释性 | 数据需求 | 核心缺陷 |
|:---|:---|:---|:---|:---|
| SARIMAX | 负值预测、不现实置信区间（Figure 2, Figure 8） | 稀疏条件下差 | 需长历史序列 | Gaussian假设与计数本质矛盾 |
| SARIMAX + log(x+1) | 略改善但仍不稳定 | 中等 | 需长历史序列 | 未改变似然函数根本 |
| SARIMAX + VLAI外生 | 置信区间略收窄 | 严重度信息增强解释力 | 需长历史序列 | 仍受高斯假设制约 |
| **Poisson + VLAI外生** | **稳定非负预测**（Figure 3, Figure 9, Figure 10, Figure 13） | **与离散事件驱动对齐** | 中等长度序列 | 长期预测未评估 |
| **指数衰减** | **短期估计合理**（Figure 4, Figure 11） | 参数物理意义明确 | **仅需初始观测** | 仅适用单调衰减模式 |


![Figure 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/78fa5a20-2708-402a-b34a-bb7a6beaeea6/figures/Figure_4.png)
*Figure 4: Figure 3: Poisson regression*



**核心证据分析**：
- **Figure 2**（CVE-2025-61932的SARIMAX预测）：展示高斯置信区间在稀疏区的不可靠性——区间过度扩张且下探负值区域
- **Figure 3**（同CVE的Poisson回归）：置信区间更紧凑，预测值始终非负，与观测计数形态更吻合
- **Figure 8**（CVE-2025-59287的SARIMAX）：另一案例验证SARIMAX负值预测问题的普遍性
- **Figure 4, Figure 11**（指数衰减）：验证三参数拟合在运营场景的有效性

**消融实验**（定性结论）：
- 移除VLAI外生变量 → 漏洞影响潜力解释力下降
- 泊松→高斯似然回退 → 负值预测重现
- 移除指数衰减 → 短序列场景无替代方案


![Figure 8](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/78fa5a20-2708-402a-b34a-bb7a6beaeea6/figures/Figure_8.png)
*Figure 8: Figure 7: Observed sightings over time for CVE-2025-59287*



**公平性检验**：
- **基线强度**：❌ **不足**。SARIMAX为1970s方法，未与N-BEATS、TFT、PatchTST等现代深度概率预测方法对比；未测试零膨胀模型（ZIP/ZINB）等计数专用方案
- **定量指标**：❌ **缺失**。无MAE、RMSE、CRPS等数值指标，结论依赖视觉 inspection
- **确认偏差风险**：⚠️ VLAI为同作者前置工作，严重度分数效用可能过拟合
- **计算成本**：✓ 轻量。泊松回归与指数衰减均为低计算开销方法，适配运营部署
- **失败案例**：逻辑斯蒂增长（Figure 6）作为指数衰减的替代形式，适用场景有限；长期预测性能未评估

## 方法谱系与知识库定位

**方法家族**：经典时间序列预测 → 计数数据适配改良分支

**父方法**：SARIMAX（Seasonal AutoRegressive Integrated Moving Average with eXogenous variables）
- **替换 slot**: inference_strategy — Poisson likelihood 替代 Gaussian likelihood，预测空间从 ℝ⁺ 约束至 ℕ⁺
- **扩展 slot**: data_pipeline — VLAI严重度分数作为外生变量注入
- **新增 slot**: inference_strategy — 指数衰减函数作为无历史序列的完全替代方案

**直接基线对比**：
| 基线 | 与本文差异 |
|:---|:---|
| SARIMAX | 高斯假设 vs. 泊松似然；需长历史 vs. 指数衰减免历史 |
| SARIMAX + log(x+1) | 数据变换修饰 vs. 根本改变概率生成模型 |
| SARIMAX + VLAI外生 | 同外生变量、不同似然函数；验证严重度信息增益的消融对照 |
| VLAI（组件来源） | 静态文本分类器 vs. 本文将其转化为动态预测外生变量 |

**后续方向**：
1. 引入神经计数模型（DeepAR with Negative Binomial, N-BEATS-I）作为更强基线，构建定量评估协议
2. 探索层次化预测：融合CVE层级结构（CWE类别、厂商、产品）提升稀疏数据共享强度
3. 跨领域迁移验证：公共卫生突发事件计数、设备故障报告等同类稀疏突发场景

**知识库标签**：
- **modality**: time-series + text（跨模态：NLP语义提取→结构化预测）
- **paradigm**: supervised, statistical forecasting（非深度学习的轻量统计方法）
- **scenario**: data-constrained operational deployment（数据稀缺运营场景）
- **mechanism**: count-based likelihood, exponential decay, exogenous variable augmentation
- **constraint**: sparse data, short time horizons, zero-inflated counts, bursty events

