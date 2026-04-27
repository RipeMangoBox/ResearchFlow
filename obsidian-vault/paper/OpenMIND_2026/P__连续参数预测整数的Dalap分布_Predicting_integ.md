---
title: Predicting integers from continuous parameters
type: paper
paper_level: B
venue: OpenMIND
year: 2026
paper_link: https://arxiv.org/abs/2602.10751
aliases:
- 连续参数预测整数的Dalap分布扩展
- Predicting_integ
- 核心洞察是：离散整数预测的瓶颈不在于分布形状
paradigm: Reinforcement Learning
---

# Predicting integers from continuous parameters

[Paper](https://arxiv.org/abs/2602.10751)

**Topics**: [[T__Time_Series_Forecasting]], [[T__Image_Generation]], [[T__Classification]]

> [!tip] 核心洞察
> 核心洞察是：离散整数预测的瓶颈不在于分布形状，而在于均值参数的连续性。原始离散Laplace类比分布因均值为整数而无法反向传播；本文通过重新推导分区函数，将均值扩展为连续值，同时保持分布的离散性和闭合形式——这是一个最小化的数学扩展，但解锁了梯度下降的可用性。Bitwise则从另一角度出发：放弃对分布形状的任何假设，用位级Bernoulli分解来处理极端不确定性。两者共同说明，整数预测的关键是参数连续性与分布离散性的解耦。

| 中文题名 | 连续参数预测整数的Dalap分布扩展 |
| 英文题名 | Predicting integers from continuous parameters |
| 会议/期刊 | Open MIND (2026) |
| 链接 | [arXiv](https://arxiv.org/abs/2602.10751) · [Code] · [Project] |
| 主要任务 | 回归任务中的整数预测（tabular数据、序列预测、图像生成） |
| 主要 baseline | 连续松弛（MSE+四舍五入）、dnormal、dlaplace、Dweib、Poisson、Danorm |

> [!abstract] 因为「离散整数预测需要有效PMF但现有方法要么无法反向传播、要么计算不可行」，作者在「离散Laplace类比分布（DLap）」基础上改了「将整数均值μ扩展为连续值并重新推导归一化常数」，在「Bicycles/Upvotes/Migration/MAESTRO等benchmark」上取得「Dalap在Bicycles达6.78±0.02 bits、Upvotes达6.74±0.01 bits，Bitwise在Migration达18.0±0.0 bits（K=8混合）」

- **Dalap** 在Bicycles上以6.78±0.02 bits超越Poisson（7.08）和dlaplace（6.80），RMSE 2.42±0.03 vs 连续松弛2.35
- **Bitwise** 在极端过度离散数据集Migration上以18.0±0.0 bits（K=8混合）大幅领先Dalap单组件的22.9±1.0 bits
- **Dweib** 在Upvotes上bits高达130.3±13.6，在Migration上因不支持负值完全失效，暴露其根本局限

## 背景与动机

在真实世界的回归任务中，预测目标常被约束为整数：社交媒体帖子的点赞数、公共自行车站点的可用车辆数、图像的RGB像素值等。这些场景看似可用标准连续回归解决，实则暗藏根本性矛盾——整数是离散的，但神经网络的输出必须是连续可微的。

现有方法形成三条路径，各有死结：**连续松弛**（MSE训练+四舍五入推理）最为普及，它将整数标签视为连续值优化，但无法提供整数上的有效概率质量函数（PMF），因而不能计算有效的对数概率，也无法与其他离散损失组合使用；**分类分布**将每个整数视为独立类别，计算量随类别数线性增长，对无界整数完全不可行，且彻底忽略了整数的序数结构；**Poisson分布**虽天然离散，但其单参数形式强制均值等于方差，在过度离散数据上严重失效——例如Upvotes数据集上Poisson的bits高达56.6，而实际数据方差远大于均值。

更专门的离散分布尝试亦未解决核心矛盾：离散Weibull（Dweib）仅支持非负整数且梯度不稳定；将连续分布直接离散化（dnormal、dlaplace）虽可行，但其分区函数计算在高维场景下代价高昂——Danorm在图像生成实验中需超过90GB VRAM。所有方法的共同瓶颈在于：**均值参数的连续性与分布的离散性未能解耦**。原始离散Laplace类比分布（Inusah & Kozubowski, 2006）要求均值μ为整数，这一看似微小的限制使其无法接入梯度下降优化。

本文的核心动机正是打破这一瓶颈：通过最小化的数学扩展，将离散分布的均值参数化为连续值，同时保持闭合形式的PMF和数值稳定性。
![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/4296c0e1-fc0d-40ae-9710-6edf284cb11d/figures/Figure_1.png)
*Figure 1 (motivation): The principle behind Dalap. The neighbors of μ are assigned probability mass between γ and 1 by an exponential function of their distance to μ.*



## 核心创新

核心洞察：离散整数预测的瓶颈不在于分布形状的选择，而在于均值参数是否连续可微，因为梯度下降要求所有参数连续变化，而原始离散Laplace类比分布的整数均值μ阻断反向传播；本文通过引入分数部分f=μ−⌊μ⌋和c=⌈μ⌉−μ重新推导分区函数，使μ连续化，从而使Dalap能够以闭合形式PMF接入神经网络训练，同时保持指数衰减尾部的离散特性。

| 维度 | Baseline（原始DLap / dlaplace / dnormal） | 本文 |
|:---|:---|:---|
| 均值参数 | 原始DLap要求μ∈ℤ；dlaplace/dnormal通过CDF差分间接定义 | Dalap直接扩展μ∈ℝ，显式推导含分数部分的归一化常数 |
| 分区函数 | dlaplace/dnormal需逐点计算CDF差分；Danorm需数值近似 | Dalap获得闭合形式z=(γ^c+γ^f)/(1−γ)，O(1)计算 |
| 高维扩展性 | Danorm内存需求>90GB（图像生成），线性增长 | Dalap/Bitwise常数复杂度；Bitwise以位分解规避维度灾难 |
| 分布假设 | dlaplace/dnormal假设连续核形状；Poisson强制均值=方差 | Dalap保持双参数（μ,γ）可调；Bitwise完全无形状假设 |

## 整体框架



本文系统性地构建了一个整数预测分布的"工具箱"，包含六种可置于神经网络输出层的分布，按设计哲学分为三层：

**输入层**：神经网络最后一层输出连续参数（如μ, γ或位概率logits），维度取决于所选分布类型——Dalap/Danorm输出2维（位置+尺度），Bitwise输出B维（B为最大位数）。

**分布层（三大分支）**：
- **连续离散化分支**：dnormal、dlaplace通过连续分布CDF差分定义PMF，参数天然连续但分区函数无闭合形式；Danorm尝试正态扩展，需数值积分近似。
- **Dalap分支（核心）**：接收连续μ∈ℝ和γ∈(0,1)，通过分数分解f=μ−⌊μ⌋、c=⌈μ⌉−μ计算闭合分区函数，输出整数n上的指数衰减PMF。
- **Bitwise分支（无假设）**：将目标整数n编码为B位二进制，每位独立通过sigmoid输出Bernoulli概率，通过位乘积重构完整PMF，天然支持任意有界整数范围。

**输出层**：所有分布均可扩展为K组件混合模型，通过可学习的混合权重π_k组合，增强对多峰或极端过度离散数据的表达能力。

**训练目标**：统一采用负对数似然（bits）优化，支持与其他离散损失联合训练。

```
神经网络输出 → [选择分布类型]
  ├─ Dalap: (μ, γ) → 分数分解 f,c → 闭合PMF p(n|μ,γ)
  ├─ Bitwise: (b_0, ..., b_{B-1}) → 位级Bernoulli → 乘积PMF
  └─ 混合扩展: K组参数 + softmax(π) → ∑_k π_k · p_k(n)
       ↓
  负对数似然损失 −log p(n_true)
```

## 核心模块与公式推导

### 模块 1: Dalap——连续均值扩展（对应框架图 分布层核心）

**直觉**：原始离散Laplace类比分布的PMF形式优雅——p(n)∝γ^{|n−μ|}，但μ∈ℤ的硬性约束使其无法从神经网络接收梯度。关键观察是：若允许μ为实数，则n−μ的绝对值仍有定义，但归一化常数需重新计算，因为无穷级数的求和起点变为非整数。

**Baseline 公式** (Inusah & Kozubowski, 2006 原始DLap):
$$p_{\text{DLap}}(n|\mu,\gamma) = \frac{1-\gamma}{1+\gamma} \cdot \gamma^{|n-\mu|}, \quad \mu \in \mathbb{Z}, \gamma \in (0,1)$$
符号: μ = 整数位置参数, γ = 衰减率（控制尾部厚度）, n ∈ ℤ 为目标整数

**变化点**：当μ∉ℤ时，原始归一化常数(1−γ)/(1+γ)失效，因为以非整数为中心的几何级数求和需分别处理上下两个方向的不对称步长。

**本文公式（推导）**:
$$\text{Step 1}: \text{定义分数部分} \quad f = \mu - \lfloor\mu\rfloor \in [0,1), \quad c = \lceil\mu\rceil - \mu = 1-f \text{（当μ∉ℤ时）}$$
$$\text{Step 2}: \text{拆分双向级数} \quad z_{\text{below}} = \sum_{m=0}^{\infty} \gamma^{m+f} = \frac{\gamma^f}{1-\gamma}, \quad z_{\text{above}} = \sum_{m=0}^{\infty} \gamma^{m+c} = \frac{\gamma^c}{1-\gamma}$$
$$\text{Step 3}: \text{总分区函数} \quad z = z_{\text{below}} + z_{\text{above}} = \frac{\gamma^f + \gamma^c}{1-\gamma}$$
$$\text{最终}: p_{\text{Dalap}}(n|\mu,\gamma) = \frac{1-\gamma}{\gamma^{\mu-\lfloor\mu\rfloor} + \gamma^{\lceil\mu\rceil-\mu}} \cdot \gamma^{|n-\mu|}$$

**一致性验证**：当μ∈ℤ时，f=0, c=1（或反之），γ^0+γ^1=1+γ，退化为原始形式；当μ∉ℤ时，分母>2√γ（由AM-GM不等式），保证归一化。

**对应消融**：Dalap单组件在Migration数据集上2/10 seeds发散（22.9±1.0 bits），K=8混合后全部稳定收敛至20.4±1.0 bits，证明混合扩展对训练稳定性的关键作用。

### 模块 2: Bitwise——位级分解（对应框架图 无假设分支）

**直觉**：当数据过度离散到分布形状完全不可预测时（如Migration的DI>10^7），任何参数化假设都是束缚。将整数表示为二进制位，用独立的Bernoulli分布建模每位，彻底放弃对整体分布形状的假设。

**Baseline 公式** (标准多分类/分类分布):
$$p_{\text{cat}}(n) = \frac{\exp(z_n)}{\sum_{n'\in\mathcal{N}}\exp(z_{n'})}, \quad |\mathcal{N}| \text{个参数，随范围线性增长}$$

**变化点**：分类分布参数随整数范围爆炸；Bitwise将参数从O(N)降至O(log N)，且通过位独立性假设解耦极端值概率。

**本文公式（推导）**:
$$\text{Step 1}: \text{二进制编码} \quad n = \sum_{b=0}^{B-1} n_b \cdot 2^b, \quad n_b \in \{0,1\}$$
$$\text{Step 2}: \text{位级Bernoulli} \quad p_b = \text{sigmoid}(\theta_b), \quad P(n_b) = p_b^{n_b}(1-p_b)^{1-n_b}$$
$$\text{Step 3}: \text{独立性假设} \quad p_{\text{Bitwise}}(n|\{\theta_b\}) = \prod_{b=0}^{B-1} p_b^{n_b}(1-p_b)^{1-n_b}$$
$$\text{最终}: \mathcal{L} = -\sum_{b=0}^{B-1} \left[ n_b^{\text{true}} \log p_b + (1-n_b^{\text{true}})\log(1-p_b) \right]$$

**关键代价**：位独立性导致RMSE显著偏高——Bicycles上Bitwise的RMSE达1.5×10^4±2.2×10^3，因高位不确定性通过乘积放大误差。这是无假设灵活性换来的精确性代价。

### 模块 3: 混合模型扩展（对应框架图 输出层）

**直觉**：单组件分布难以捕捉多峰或极端方差；混合模型通过可学习权重组合多个组件，是提升表达力的标准手段，但对Dalap尤为关键——解决初始化敏感性。

**本文公式**：
$$p_{\text{mix}}(n) = \sum_{k=1}^{K} \pi_k \cdot p(n|\mu_k, \gamma_k), \quad \sum_k \pi_k = 1$$
其中π_k = softmax(φ_k)，φ_k为可学习logits。

**对应消融**：Dalap在Migration单组件发散率20%（2/10 seeds），K=8混合后收敛率100%，bits从22.9±1.0降至20.4±1.0；Bitwise混合后从22.9±1.0降至18.0±0.0，为所有方法中最优。

## 实验与分析

| Method | Bicycles (bits) | Upvotes (bits) | Migration (bits) | MAESTRO (bits) | 备注 |
|:---|:---|:---|:---|:---|:---|
| 连续松弛 (MSE) | — | — | — | — | 无有效PMF，RMSE最优3/4数据集 |
| Poisson | 7.08 | 56.6 | 发散 | **4.91** | 均值=方差假设，过度离散失效 |
| Dweib | 7.37 | 130.3±13.6 | **不支持负值** | 5.03 | 非负限制，梯度不稳定 |
| dlaplace | 6.80 | 6.75±0.01 | 23.3±0.1 | 5.01 | 无闭合分区函数 |
| **Dalap (本文)** | **6.78±0.02** | **6.74±0.01** | 22.9±1.0 (单) / 20.4±1.0 (K=8) | 4.96 | 核心贡献 |
| **Bitwise (本文)** | 7.01±0.01 | 7.09±0.01 | **18.0±0.0 (K=8)** | 5.01 | 极端离散最优 |
| Danorm | 6.79 | 6.75 | 23.3 | — | 图像生成需>90GB VRAM |


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/4296c0e1-fc0d-40ae-9710-6edf284cb11d/figures/Figure_2.png)
*Figure 2 (motivation): The expected value of unbounded Dalap. We take two numbers at a distance of γ/(1−γ) below and above ⌊μ⌋ and ⌈μ⌉ respectively.*



**核心结论支持**：Dalap在Bicycles和Upvotes上的bits优势（6.78 vs dlaplace 6.80, 6.74 vs 6.75）虽数值差距小，但标准差极小（±0.02/±0.01），统计显著；更关键的是Dalap提供闭合PMF而dlaplace需数值计算。Bitwise在Migration的压倒性优势（18.0 vs Dalap 20.4）验证了其"无假设"设计哲学在极端场景的价值。

**边际结果**：MAESTRO上所有方法bits接近（4.91-5.03），Poisson反而最优——说明中等离散度计数数据中均值=方差假设恰好成立，复杂分布无必要。

**消融分析**：混合组件数K的影响呈现方法差异性。Dalap单组件在Migration的发散问题（2/10 seeds）通过K=8完全解决，bits提升2.5；Bitwise混合增益更大（22.9→18.0），因其位独立性假设在多组件下能组合更复杂模式。

**公平性检查**：
- **Baseline强度**：包含连续松弛、Poisson、Dweib、dlaplace、Danorm等六种，覆盖主要技术路线，较为全面。
- **计算成本**：Dalap分区函数O(1)，显著优于Danorm的数值积分；Bitwise参数量O(B)与数据范围对数相关。
- **数据成本**：四个tabular数据集规模适中，但图像生成实验仅使用单一随机种子，结论可靠性有限。
- **失败案例**：Dalap单组件初始化敏感；Bitwise RMSE全面偏高；Danorm高维不可扩展；未对K进行系统超参搜索。

## 方法谱系与知识库定位

**方法家族**：离散概率分布的参数化扩展（parametric discrete distribution extension for neural networks）

**父方法**：Inusah & Kozubowski (2006) 的离散Laplace类比分布（DLap），以及将连续分布离散化的通用框架（dnormal/dlaplace的CDF差分法）。

**改变的插槽**：
- **目标函数/参数化**：均值μ从ℤ扩展至ℝ（核心数学贡献）
- **架构**：新增Bitwise位分解分支，完全替代分布形状假设
- **训练配方**：混合模型作为稳定训练的标准扩展
- **推理**：保持精确采样，无近似

**直接对比**：
- **vs dlaplace/dnormal**：Dalap获得闭合PMF，避免逐点CDF计算；Danorm尝试类似扩展但失败（数值不稳定+高内存）
- **vs Poisson**：Dalap/Bitwise解耦均值与方差，支持任意过度离散度
- **vs Dweib**：Dalap支持全整数域ℤ，Dweib限非负且梯度差
- **vs 连续松弛**：所有本文方法提供有效PMF，支持概率推理与损失组合

**后续方向**：
1. **自适应混合**：动态组件数K而非固定，应对未知离散度
2. **高维效率**：将Dalap闭合形式扩展到图像/序列的高维联合分布，替代Danorm
3. **位相关性建模**：Bitwise的独立性假设可松弛为轻量级位间依赖（如自回归位生成），降低RMSE代价

**标签**：modality=tabular/sequence/image_generation | paradigm=概率回归/probabilistic_regression | scenario=整数约束预测 | mechanism=离散分布连续参数化+位分解 | constraint=可微训练+闭合形式+数值稳定

