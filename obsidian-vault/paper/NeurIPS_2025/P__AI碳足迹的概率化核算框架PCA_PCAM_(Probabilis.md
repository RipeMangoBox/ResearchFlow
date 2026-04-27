---
title: Unveiling the Uncertainty in Embodied and Operational Carbon of Large AI Models through a Probabilistic Carbon Accounting Model
type: paper
paper_level: A
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- AI碳足迹的概率化核算框架PCAM
- PCAM (Probabilis
- PCAM (Probabilistic Carbon Accounting Model)
- PCAM (Probabilistic Carbon Accounti
acceptance: Poster
method: PCAM (Probabilistic Carbon Accounting Model)
modalities:
- Text
paradigm: unsupervised
baselines:
- LLM端到端碳足迹建模框架_LLMCarbon
---

# Unveiling the Uncertainty in Embodied and Operational Carbon of Large AI Models through a Probabilistic Carbon Accounting Model

**Topics**: [[T__Benchmark_-_Evaluation]] | **Method**: [[M__PCAM]] | **Datasets**: Embodied carbon accounting, Operational carbon accounting, XLM embodied carbon detailed, Embodied carbon accounting of XLM, Operational carbon accounting of large AI models

> [!tip] 核心洞察
> PCAM (Probabilistic Carbon Accounting Model) quantifies uncertainties in AI model carbon accounting through parameter-specific distributions, achieving dramatically lower error (≤7.44%) compared to deterministic methods like LLMCarbon (≤108.51%).

| 中文题名 | AI碳足迹的概率化核算框架PCAM |
| 英文题名 | Unveiling the Uncertainty in Embodied and Operational Carbon of Large AI Models through a Probabilistic Carbon Accounting Model |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/) · [Code](https://github.com/) · [Project](https://) |
| 主要任务 | AI碳核算 / 基准测试与评估 |
| 主要 baseline | LLMCarbon, STEC, Carbontracker, ACT |

> [!abstract] 因为「现有AI碳核算方法均为确定性模型，无法量化制造和运营中的不确定性」，作者在「LLMCarbon」基础上改了「将系统级平均参数替换为组件级概率分布，并引入KDE非参数密度估计」，在「隐含碳核算对比基准」上取得「误差从≤108.51%降至≤7.44%」

- **隐含碳误差**: PCAM ≤7.44% vs. LLMCarbon ≤108.51%，最坏情况误差降低14.6倍
- **数据覆盖**: 聚合2021-2023年韩国、中国、台湾、美国、日本五地电网小时级碳强度数据
- **组件粒度**: 处理器、内存(DRAM)、存储(SSD)分别建模，替代系统级粗粒度平均

## 背景与动机

训练一个GPT-4级别的模型究竟排放多少二氧化碳？当前答案是一个确定的数字——但这个数字忽略了关键事实：台积电在不同工艺节点的良率波动、同一地区不同季节的电网碳强度变化、以及不同供应商DRAM制造能耗的差异。现有方法如LLMCarbon [1] 和基于LCA的确定性模型将制造碳强度(CI)、硬件良率(Y)、单位面积能耗(EPS)等参数视为单一确定值，导致估算结果与实际存在数量级偏差。

现有处理方法各有局限。**LLMCarbon** [1] 采用系统级平均隐含碳值(EC_system)，将整块GPU或TPU的制造排放简化为一个固定系数，无法反映同一型号硬件因产地、批次不同导致的差异。**STEC** [13] 虽引入时空维度，但仍使用确定性时空值而非概率分布，无法表达参数本身的置信区间。**ACT** [12] 提供组件级架构工具，但依赖LCA报告的静态数据，未考虑技术演进带来的参数漂移。

这些方法的共同短板在于：**将本质上具有地理异质性、时间动态性和供应链不确定性的制造参数，压缩为单一确定值**。例如，Figure 1 显示2021-2023年中国十大省级电网的小时级碳强度分布呈显著多峰特征，简单取均值会抹除极端值风险；Figure 2 进一步展示不同工艺节点(<10nm vs. 28+nm)处理器和存储器的制造碳强度核密度估计形态迥异，系统级平均无法捕捉这种结构差异。



本文提出PCAM，首次将概率建模引入AI碳核算领域，通过组件级参数分布和KDE密度估计量化不确定性。

## 核心创新

核心洞察：**制造和运营碳排放的参数本身具有可观测的统计分布**，因为同一工艺节点在不同代工厂、同一地区在不同季节的实测数据天然离散，从而使非参数概率建模替代点估计成为可能。

| 维度 | Baseline (LLMCarbon) | 本文 (PCAM) |
|:---|:---|:---|
| 数据来源 | 单一LCA报告，产品类别级平均 | ESG报告、技术报告、电网运营商数据聚合 |
| 参数表征 | 确定值 (如 EC_system = 固定常数) | KDE概率密度分布 (如 Ỹ, CĨm, ẼPS) |
| 计算粒度 | 系统级 (整块GPU/TPU) | 组件级 (处理器/内存/存储分别建模) |
| 输出形式 | 单一点估计 | 完整概率分布 + 置信区间 |
| 不确定性处理 | 忽略 | 通过蒙特卡洛采样传播 |

这一范式转换使碳核算从"报告一个数字"升级为"量化数字的可信范围"，为AI系统的气候风险评估提供决策依据。

## 整体框架



PCAM的处理流程包含五个串行模块，从多源异构数据到最终的概率化碳足迹输出：

1. **多源数据收集与聚合**：输入为ESG报告(如TSMC [22]、SK Hynix [23])、技术报告(如NVIDIA A100规格 [10]、Google TPU [11])、电网运营商小时级碳强度数据、以及Seagate LCA报告 [24]；输出为稀疏的离散参数样本集，涵盖良率Y、制造碳强度CIm、单位面积耗电EPS、气体排放GPS、材料排放MPS等。

2. **KDE分布估计**：输入为稀疏样本，通过Parzen-Rosenblatt核密度估计 [15] 生成连续概率密度函数；输出为每个参数的概率分布，带宽h通过数据驱动方式选择。这是将确定性参数转化为不确定性表征的核心枢纽。

3. **组件级隐含碳概率建模**：输入为硬件配置(芯片尺寸DieSize、操作时间t、寿命T)和参数分布；对处理器、DRAM、SSD分别建立概率化方程；输出为各组件隐含碳的独立概率分布。

4. **运营碳概率建模**：输入为实际耗电量E和时变碳强度CI分布；输出为运营碳OC的概率分布，捕获用电时间、地理位置带来的不确定性。

5. **蒙特卡洛聚合**：从各参数分布中联合采样，通过大数定律收敛得到总碳排放C_model = EC_model + OC_model的完整概率分布，提供均值、中位数、置信区间等多维输出。

```
[多源数据] → {KDE估计} → [参数分布 Ỹ, CĨm, ẼPS, ...]
                                    ↓
[硬件配置] → [处理器模型] → [EC_p分布] ─┐
[硬件配置] → [内存模型]   → [EC_m分布] ─┼→ {蒙特卡洛} → [C_model概率分布]
[硬件配置] → [存储模型]   → [EC_s分布] ─┘         ↑
[E, CI(t)] → [运营碳模型] → [OC分布] ──────────────┘
```

## 核心模块与公式推导

### 模块 1: KDE非参数密度估计（对应框架图"KDE分布估计"层）

**直觉**: 制造参数的真实分布未知且可能多峰，参数化假设（如正态分布）会引入模型偏差，需用数据本身说话。

**Baseline 公式** (确定性赋值): 
$$x = \bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$$
符号: $x_i$ = 第i个观测样本, $\bar{x}$ = 样本算术平均, $n$ = 样本数

**变化点**: 简单平均将不同地区、不同时间的异质性信息压缩为一点，丢失极端值信息；且当样本稀疏时(n小)，均值估计方差大。

**本文公式（推导）**:
$$\text{Step 1}: K(u) = \frac{1}{\sqrt{2\pi}}e^{-u^2/2} \quad \text{选择高斯核函数，保证平滑性和可微性}$$
$$\text{Step 2}: \hat{f}_h(x) = \frac{1}{nh}\sum_{i=1}^{n}K\left(\frac{x-x_i}{h}\right) \quad \text{Parzen-Rosenblatt核估计，带宽}h\text{控制偏差-方差权衡}$$
$$\text{最终}: \tilde{X} \sim \hat{f}_h(x), \quad x \in \mathbb{R} \quad \text{参数}X\text{的概率化表示，波浪号表示随机变量}$$

**对应消融**: 未显式提供KDE vs. 参数化分布的消融，但Figure 3展示变异系数(CV)的KDE结果与直方图高度吻合，验证非参数选择的合理性。

### 模块 2: 处理器隐含碳概率模型（对应框架图"组件级概率建模"层）

**直觉**: 处理器是AI硬件中隐含碳最高的组件，其良率、制造能耗和碳强度的波动直接决定整体估算精度。

**Baseline 公式** (LLMCarbon [1]):
$$EC_{model} = \frac{t}{T} \cdot EC_{system}$$
符号: $t$ = 实际使用时间, $T$ = 硬件寿命, $EC_{system}$ = 系统级平均隐含碳（查表常数）

**变化点**: $EC_{system}$ 将不同工艺节点、不同代工厂的处理器混为一谈；忽略芯片尺寸DieSize、良率Y、单位面积耗电EPS等物理参数的实际分布。

**本文公式（推导）**:
$$\text{Step 1}: EC_{model}^p = \frac{t_p \cdot DieSize}{T_p \cdot \tilde{Y}} \cdot (\tilde{CI}_m \cdot \tilde{EPS} + GPS + MPS) \quad \text{将系统级}EC_{system}\text{分解为物理参数乘积，关键参数概率化}$$
$$\text{Step 2}: \tilde{Y} \sim \hat{f}_h^Y(y), \quad \tilde{CI}_m \sim \hat{f}_h^{CI}(c), \quad \tilde{EPS} \sim \hat{f}_h^{EPS}(e) \quad \text{各参数独立采样，波浪号为KDE分布}$$
$$\text{最终}: EC_{model}^p = \frac{t_p \cdot DieSize}{T_p} \cdot \frac{\tilde{CI}_m \cdot \tilde{EPS} + GPS + MPS}{\tilde{Y}}$$
符号: $t_p$ = 处理器使用时间, $T_p$ = 处理器寿命, $DieSize$ = 芯片面积(mm²), $\tilde{Y}$ = 良率概率模型, $\tilde{CI}_m$ = 制造碳强度(gCO₂e/kWh), $\tilde{EPS}$ = 单位面积耗电(kWh/mm²), $GPS$ = 气体工艺排放, $MPS$ = 材料工艺排放

**对应消融**: Table 4显示PCAM对多种AI模型(XLM、GPT-3等)的隐含碳累积估算误差≤7.44%，而LLMCarbon高达≤108.51%。

### 模块 3: 运营碳概率模型与蒙特卡洛聚合（对应框架图"运营碳建模"与"蒙特卡洛聚合"层）

**直觉**: 同一模型在北京冬季(煤电为主) vs. 四川夏季(水电为主)训练，运营碳可差数倍，时间-空间耦合的不确定性必须联合传播。

**Baseline 公式**:
$$OC_{model} = E \cdot CI$$
符号: $E$ = 总耗电量(kWh), $CI$ = 电网平均碳强度(gCO₂e/kWh)

**变化点**: 使用单一$CI$平均值忽略用电时段和地域的碳强度波动；且未与隐含碳的不确定性联合考虑。

**本文公式（推导）**:
$$\text{Step 1}: \tilde{CI}(t, loc) \sim \hat{f}_h^{CI(t,loc)} \quad \text{时变、地变的碳强度分布，从电网运营商小时级数据学习}$$
$$\text{Step 2}: \tilde{OC}_{model} = E \cdot \tilde{CI}(t, loc) \quad \text{运营碳成为随机变量}$$
$$\text{Step 3}: \tilde{C}_{model} = \tilde{EC}_{model}^p + \tilde{EC}_{model}^m + \tilde{EC}_{model}^s + \tilde{OC}_{model} \quad \text{组件级与运营碳联合}$$
$$\text{最终}: \tilde{C}_{model}^{(k)} = \sum_{j=1}^{4}\tilde{C}_j^{(k)}, \quad k=1,...,N_{MC} \quad \text{蒙特卡洛采样}N_{MC}\text{次，输出经验分布}$$

**对应消融**: Table 5显示PCAM在运营碳累积估算上相对LLMCarbon显著改进，Figure 5进一步展示PCAM估算结果与美国EPA标准的对比一致性。

## 实验与分析


![Table 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/864810c1-4c48-48e7-8ade-128cc3d90af4/figures/Table_4.png)
*Table 4 (comparison): The comparison between LLMCarbon and PCAM on embodied carbon accumulation for different AI models*



本文在隐含碳和运营碳两个维度上系统对比PCAM与LLMCarbon的核算精度。Table 4展示了对XLM、GPT-3等多种大模型的隐含碳累积估算结果：PCAM的误差上界为≤7.44%，而LLMCarbon的误差上界高达≤108.51%，最坏情况下精度提升14.6倍。这一差距源于LLMCarbon使用粗粒度的系统级平均隐含碳值，当面对新型号硬件或不同产地时产生系统性偏差；PCAM通过组件级参数分布捕捉了这种异质性。


![Table 5](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/864810c1-4c48-48e7-8ade-128cc3d90af4/figures/Table_5.png)
*Table 5 (comparison): The comparison between LLMCarbon and PCAM on different AI models for operational carbon accumulation*



Table 5进一步展示运营碳累积估算的对比。虽然具体数值未在摘要中完整披露，但PCAM同样显著优于LLMCarbon，这得益于时变碳强度分布对用电时段和地域的精确建模。Figure 5将PCAM的运营碳估算结果与美国EPA标准进行交叉验证，显示PCAM的分布估计与官方标准具有良好的一致性。


![Figure 5](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/864810c1-4c48-48e7-8ade-128cc3d90af4/figures/Figure_5.png)
*Figure 5 (comparison): The comparison between LLMCarbon and PCAM on operational carbon accounting to the U.S. EPA standard*



Figure 3和Figure 4从数据层面支撑方法有效性：Figure 3展示变异系数(CV)的核密度估计，揭示不同参数的不确定性幅度差异；Figure 4展示制造碳排放的KDE分布，验证多源聚合数据的统计合理性。这些分布形态直接决定了蒙特卡洛采样的收敛行为。

**公平性检验**：本文的比较存在若干局限。首先，主要定量对比仅限于LLMCarbon一个基线，虽提及Carbontracker [26] 和STEC [13]，但未在主要结果中呈现全面对比——STEC仅在XLM单一模型上比较，Carbontracker缺乏定量结果。其次，缺少对KDE带宽选择、蒙特卡洛采样次数的敏感性分析，也未提供各独立组件（如仅概率化良率vs.仅概率化碳强度）的消融实验。作者声明"任何计算机均可复现"但未给出实际计算时间或资源消耗。数据集聚合自多种来源，可能存在未量化的采集偏差。尽管如此，≤7.44% vs. ≤108.51%的误差差距幅度巨大，且物理机理清晰，核心结论具有稳健性。

## 方法谱系与知识库定位

**方法家族**: AI碳核算 → 概率化不确定性量化

**直接继承**: LLMCarbon [1] 是PCAM的直系父方法。PCAM保留了"组件使用时间/寿命比例 × 单位排放 + 用电量 × 碳强度"的分解结构，但将四个关键slot彻底改造：数据管道从单源LCA查表替换为多源聚合+KDE估计；架构从系统级常数修改为组件级概率方程；推理策略从点估计输出替换为蒙特卡洛分布采样；优化目标从最小化单点排放改为量化不确定性范围。

**旁系关联**: 
- **STEC** [13]: 时空隐含碳模型，PCAM概率化扩展其时空维度但替换确定性值为分布
- **ACT** [12]: 架构碳建模工具，PCAM继承其组件级思想但加入概率传播
- **Carbontracker** [26]: 运营碳追踪工具，PCAM在引用但未深度对比

**后续方向**: (1) 将PCAM的概率框架扩展至神经网络架构搜索(CE-NAS [6] 的碳效率优化可结合不确定性约束)；(2) 建立在线更新机制，随新硬件发布自动刷新KDE分布；(3) 开发轻量化版本，在边缘设备上实时估算碳足迹分布。

**标签**: 模态[文本/通用AI系统] | 范式[无监督密度估计+蒙特卡洛推断] | 场景[AI可持续性评估/碳足迹审计] | 机制[核密度估计/概率传播/组件级分解] | 约束[多源异构数据/稀疏样本/时空动态]

## 引用网络

### 直接 baseline（本文基于）

- [[P__LLM端到端碳足迹建模框架_LLMCarbon]] _(直接 baseline)_: LLMCarbon is the most directly comparable prior work on end-to-end carbon footpr

