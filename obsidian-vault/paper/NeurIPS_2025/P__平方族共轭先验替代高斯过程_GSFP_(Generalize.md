---
title: Squared families are useful conjugate priors
type: paper
paper_level: A
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 平方族共轭先验替代高斯过程
- GSFP (Generalize
- GSFP (Generalized Squared Family Prior)
- Squared families of probability dis
acceptance: Poster
method: GSFP (Generalized Squared Family Prior)
modalities:
- tabular
paradigm: supervised
---

# Squared families are useful conjugate priors

**Topics**: [[T__Few-Shot_Learning]], [[T__Time_Series_Forecasting]] | **Method**: [[M__GSFP]] | **Datasets**: UCI Regression

> [!tip] 核心洞察
> Squared families of probability distributions can serve as conjugate priors that enable closed-form tractable Bayesian inference while providing rich multi-modal alternatives to Gaussian processes with neural network features.

| 中文题名 | 平方族共轭先验替代高斯过程 |
| 英文题名 | Squared families are useful conjugate priors |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2505.xxxxx) · [Code](待补充) · [Project](待补充) |
| 主要任务 | Few-Shot Learning, Bayesian Regression, Time Series Forecasting |
| 主要 baseline | Deep Kernel Learning / GP, DKT, NGGP, Normalizing flows |

> [!abstract] 因为「高斯过程先验在深度核学习中只能建模单峰分布，且非高斯似然下需要近似推断」，作者在「Deep Kernel Learning / GP with neural network features」基础上改了「将GP先验替换为Generalized Squared Family (GSF) 共轭先验，实现闭式精确后验更新与边缘似然计算」，在「UCI Regression (9 datasets)」上取得「Boston NLL 0.27±0.00 vs DKT 0.30±0.00, Energy NLL 0.55±0.00 vs DKT 0.70±0.00, Protein NLL 70.83±2.72 vs DKT 118.11±0.02」

- **Boston**: Test NLL 0.27±0.00，相比 DKT (0.30±0.00) 降低 10%，相比 RBF (7.09±0.53) 降低 96%
- **Protein**: Test NLL 70.83±2.72，相比 DKT (118.11±0.02) 降低 40%，相比 RBF (297.26±14.70) 降低 76%
- **Energy**: Test NLL 0.55±0.00，相比 DKT (0.70±0.00) 降低 21%，相比 RBF (8.20±0.82) 降低 94%

## 背景与动机

在贝叶斯深度学习中，一个核心问题是如何对神经网络提取的特征进行不确定性建模。当前主流范式——Deep Kernel Learning (DKL)——将高斯过程（Gaussian Process, GP）先验与神经网络特征提取器结合：神经网络 φ 将输入 x 映射到特征空间，GP 在这些特征上定义先验 p(f) = GP(0, k(·,·))。这一框架在回归任务中表现优异，但存在根本性局限：GP 先验与 Gaussian 似然共轭，导致后验始终是单峰的（unimodal）。当真实后验本应具有多个模态时——例如周期性数据中存在多个相位解释、或分层模型中参数有多个竞争取值——GP 会错误地将概率质量摊平在模态之间，产生过度不确定性。

现有方法如何应对这一问题？DKT (Deep Kernel Transfer) [12] 沿用 GP 先验，通过元学习快速适应新任务，但未解决单峰局限；NGGP (Neural Gaussian Process) [17] 尝试结合神经网络与 GP 推断，仍受限于 Gaussian 共轭；Normalizing flows 虽能建模多模态分布，但需大量样本且缺乏闭式后验更新，必须依赖变分推断（VI）或 MCMC 近似。这些近似方法引入额外超参数、增加计算开销，且无法保证后验质量。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/de4a951b-31e0-4cce-98b8-c36ebf6e1167/figures/Figure_1.png)
*Figure 1 (example): (Left) Empirical Bayesian estimation of the rate parameter λ of a Poisson likelihood in a hierarchical model. (Right) Prior predictive distributions for the Gaussian model evaluated with respect to the prior parameter and sample size N, comparing unimodal and heavier-tailed predictions.*



关键缺口在于：**没有一种先验既能保持与 GP 类似的函数空间灵活性，又能提供多模态建模能力，同时维持闭式精确推断**。本文提出 Generalized Squared Family (GSF) 先验，恰好填补这一空白：通过将先验密度定义为希尔伯特空间中函数 L2 范数的平方 q(·|f) ∝ ‖f(·)‖₂²，GSF 与"平方族"似然形成共轭关系，实现无需近似的精确贝叶斯更新。

## 核心创新

核心洞察：平方族分布（squared families）可以充当共轭先验，因为平方结构使先验密度与似然的乘积仍保持可处理的积分形式，从而使闭式精确后验更新与边缘似然计算成为可能——这是 GP 先验仅在 Gaussian 似然下才能享受的特权，如今被推广到更广泛的似然族与多模态场景。

| 维度 | Baseline (Deep Kernel Learning / GP) | 本文 (GSFP) |
|:---|:---|:---|
| 先验分布 | Gaussian process p(f) = GP(0, k(·,·))，单峰 | Generalized Squared Family q(·|f) ∝ ‖f(·)‖₂²，可多峰 |
| 推断策略 | Gaussian 似然下解析后验；非 Gaussian 需 VI/MCMC 近似 | 与平方族似然严格共轭，闭式精确后验 ν(dω) = p(U\|ω)μ(dω) |
| 优化目标 | GP 边缘似然或 ELBO（近似下界） | 精确边缘似然 p(U) = ∫ p(U\|ω)μ(dω)，支持经验贝叶斯 |
| 后验形态 | 始终单峰 | 可呈现多峰，捕捉复杂不确定性结构 |

## 整体框架


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/de4a951b-31e0-4cce-98b8-c36ebf6e1167/figures/Figure_2.png)
*Figure 2 (architecture): Given base measure α and feature ψ, a parameter M>0 induces a prior G(·|M,α,ψ) and likelihood; M+N induces the posterior. The graphical model corresponds to exponential family or dispersion distributions, with linear regression in feature space defining a GP prior over likelihood mean functions.*



GSFP (Generalized Squared Family Prior) 框架将深度核学习中的 GP 先验完整替换为平方族先验，同时保持端到端可学习性。数据流如下：

1. **Neural feature extractor**（输入：原始特征 x；输出：学习表示 φ(x)）：标准神经网络，负责将输入映射到适合核方法的空间。此模块与 DKL 相同，非本文创新。

2. **GSF prior specification**（输入：特征映射 f ∈ H；输出：平方族先验密度 q(·|f) ∝ ‖f(·)‖₂²）：核心替换模块。将 GP 先验替换为基于 L2 范数平方的 GSF 先验，通过选择基测度 μ 和特征映射 ψ 控制先验性质。

3. **Closed-form posterior update**（输入：先验测度 μ，似然 p(U|ω)；输出：后验测度 ν(dω) = p(U|ω)μ(dω)）：利用 GSF 与平方族似然的共轭性，后验为先验测度与似然的简单乘积，无需任何近似。

4. **Marginal likelihood optimization**（输入：训练数据，模型超参数；输出：经经验贝叶斯优化的超参数）：精确边缘似然使端到端超参数学习成为可能，替代 ELBO 优化或交叉验证。

5. **Predictive distribution**（输入：测试点，后验分布；输出：预测均值与不确定性）：基于闭式后验计算预测分布，输出点估计与不确定性量化。

```
Input x → [Neural φ] → φ(x) → [GSF Prior q(·|f) ∝ ‖f‖₂²] → 
Observe Data U → [Posterior ν(dω) = p(U|ω)μ(dω)] → 
[Marginal Likelihood p(U)] → [Predictive Dist.] → Output
```

## 核心模块与公式推导

### 模块 1: 平方族先验密度（对应框架图位置 2，替换 GP 先验）

**直觉**：将函数空间中"能量"的平方作为概率密度，使先验天然支持多峰结构——不同"能量谷"对应不同后验解释。

**Baseline 公式** (Deep Kernel Learning / GP):
$$f \sim \mathcal{GP}(0, k(\cdot, \cdot))$$
符号: $f$ = 随机函数, $k(\cdot, \cdot)$ = 核函数, 先验为 Gaussian 过程，后验始终单峰。

**变化点**：GP 先验的 Gaussian 结构限制其仅能建模单峰后验；且仅与 Gaussian 似然共轭。本文将先验重新参数化为函数 L2 范数的平方，突破 Gaussian 限制。

**本文公式（推导）**:
$$\text{Step 1}: \quad q(\cdot|f) \propto \|f(\cdot)\|_2^2 \quad \text{将指数族的自然参数-充分统计量结构替换为 L2 范数平方}$$
$$\text{Step 2}: \quad f \in \mathcal{H} \text{（希尔伯特空间）}, \quad q(\cdot|f) = \frac{\|f(\cdot)\|_2^2}{Z(f)} \quad \text{重归一化以保证概率密度积分为 1}$$
$$\text{最终}: \quad q(x|f) = \frac{\|f(x)\|_2^2}{\int \|f(\omega)\|_2^2 \mu(d\omega)}$$
符号: $f$ = 希尔伯特空间 $\mathcal{H}$ 中的函数, $\|\cdot\|_2$ = L2 范数, $\mu$ = 基测度, $Z(f)$ = 归一化常数。

**对应消融**：Table 1 显示在 9 个 UCI 数据集上，GSFP 相比 RBF/Spectral/NN Linear 等简化基线取得大幅降低的 NLL；移除神经网络特征提取（退化为线性核）导致性能崩溃。

---

### 模块 2: 共轭后验更新（对应框架图位置 3，核心推断机制）

**直觉**：平方族的代数结构使先验与似然的乘积仍属于同一测度族，后验更新退化为测度的简单乘法——这是共轭性的极致体现。

**Baseline 公式** (VI/MCMC 近似推断):
$$q^*(f) = \text{arg}\min_{q \in \mathcal{Q}} \mathrm{KL}(q(f) \| p(f|U)) \quad \text{或 MCMC 采样}$$
符号: $q^*$ = 最优变分分布, $\mathcal{Q}$ = 变分族, $p(f|U)$ = 真实后验。需要迭代优化或采样，无闭式解。

**变化点**：变分推断引入近似误差，MCMC 计算昂贵；两者均无法提供精确后验。本文证明平方族与平方族似然严格共轭，后验有闭式表达式。

**本文公式（推导）**:
$$\text{Step 1}: \quad \text{设先验测度 } \mu(d\omega), \text{ 似然 } p(U|\omega) \propto \|g(U, \omega)\|_2^2 \quad \text{（平方族似然）}$$
$$\text{Step 2}: \quad \nu(d\omega) = p(U|\omega)\mu(d\omega) \quad \text{后验测度 = 先验测度 × 似然，无需重参数化}$$
$$\text{Step 3}: \quad p(\omega|U) = \frac{p(U|\omega)\mu(d\omega)}{\int p(U|\omega')\mu(d\omega')} = \frac{\nu(d\omega)}{p(U)} \quad \text{归一化得到标准后验密度}$$
$$\text{最终}: \quad \text{boxed}{\nu(d\omega) = p(U|\omega)\mu(d\omega)}$$
符号: $\nu$ = 后验测度, $\omega$ = 隐变量, $U$ = 观测数据。关键：后验与先验同属平方族，仅基测度由 $\mu$ 更新为 $\nu$。

**对应消融**：Table 1 中 DKT/NGGP 使用近似 GP 推断，在多个数据集上 NLL 高于 GSFP；例如 Energy 数据集 DKT 0.70±0.00 vs GSFP 0.55±0.00，差距 21%。

---

### 模块 3: 精确边缘似然（对应框架图位置 4，支持经验贝叶斯）

**直觉**：共轭结构不仅简化后验，还使边缘似然（model evidence）可精确计算，从而可用梯度下降优化超参数，替代交叉验证。

**Baseline 公式** (ELBO 或 GP 边缘似然):
$$\mathcal{L}_{\mathrm{ELBO}} = \mathbb{E}_{q(f)}[\log p(U|f)] - \mathrm{KL}(q(f) \| p(f))$$
符号: ELBO = Evidence Lower BOund, 是真实边缘似然的下界，引入近似误差。

**变化点**：ELBO 的紧度依赖变分族 $\mathcal{Q}$ 的选择；GP 边缘似然仅适用于 Gaussian 似然。本文的精确边缘似然对任意平方族似然成立。

**本文公式（推导）**:
$$\text{Step 1}: \quad p(U) = \int p(U|\omega) \mu(d\omega) \quad \text{（边缘似然定义）}$$
$$\text{Step 2}: \quad = \int \|g(U, \omega)\|_2^2 \mu(d\omega) \quad \text{（代入平方族似然）}$$
$$\text{Step 3}: \quad = \left\|\int g(U, \omega) \sqrt{\mu(d\omega)}\right\|_2^2 \quad \text{（利用平方结构，积分与范数交换）}$$
$$\text{最终}: \quad \text{boxed}{p(U) = \int \|g(U, \omega)\|_2^2 \mu(d\omega) = \text{闭式可计算}}$$
符号: $g(U, \omega)$ = 似然核函数, $p(U)$ = 边缘似然（model evidence）。该闭式使经验贝叶斯成为可能：$\hat{\lambda} = \text{arg}\max_\lambda p(U|\lambda)$，如图 1 左图所示对 Poisson 率参数 $\lambda$ 的估计。

**对应消融**：Figure 1（左）展示随着观测数 N 增加，GSF 后验对 Poisson 率参数 $\lambda$ 的估计快速集中，验证边缘似然驱动的经验贝叶斯有效性。

## 实验与分析


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/de4a951b-31e0-4cce-98b8-c36ebf6e1167/figures/Table_1.png)
*Table 1 (quantitative): Benchmark results showing test NLL for regression tasks across 9 data domains, comparing Bayesian last layer (BLL), standard last layer (SLL), and standard full model (SFM) methods with mean ± std over 5-fold cross-validation.*



本文在 9 个 UCI 回归数据集（Boston, Concrete, Energy, Kin8nm, Naval, Power Plant, Protein, Wine, Yacht）上评估 GSFP，以 Test NLL（Negative Log-Likelihood，越低越好）为主要指标。 展示了完整对比结果。总体而言，GSFP 在 8/9 数据集上优于或持平于最强基线 DKT，在大型数据集上优势尤为显著：Protein 上 NLL 70.83±2.72 vs DKT 118.11±0.02（降低 40%），Naval 上 42.08±3.60 vs 67.41±2.61（降低 38%），Power Plant 上 27.32±2.16 vs 42.51±0.56（降低 36%）。这些结果说明 GSF 先验在复杂高维回归问题上能有效捕捉数据不确定性，而 GP 基线的单峰假设成为瓶颈。

然而结果并非全线占优：Concrete 数据集上 GSFP 的 NLL 为 1.55±0.17，反而差于 DKT 的 1.10±0.00（劣化 41%）。作者未明确解释此异常，推测可能与该数据集特定的噪声结构或 GSF 超参数敏感性有关。与简单基线（RBF, Spectral, NN Linear）相比，GSFP 优势巨大，例如 Boston 上相比 RBF 的 7.09±0.53 降低 96%，Yacht 上相比 NN Linear 的 5.24±1.66 降低 97%——但这主要反映神经网络特征提取的价值，而非 GSF 先验本身的独特贡献。


![Figure 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/de4a951b-31e0-4cce-98b8-c36ebf6e1167/figures/Figure_4.png)
*Figure 4 (qualitative): When is a multimodal prior helpful? GP using deep kernel learning with periodic activations: (Top) unimodal prior exhibits uncertainty between clusters; (Bottom) inverse-Gamma prior over RBF lengthscale induces a heavy-tailed, multimodal prior over functions, allowing selection of correct lengthscale modes.*



Figure 4 定性展示了多模态先验的实际价值：在具有周期性激活的深度核学习中，单峰 GP 先验在数据稀疏区域呈现过度不确定性（uncertainty between modes），而 GSF 先验能正确识别多个可能的模式，减少不必要的方差。

关于计算成本，实验使用单张 RTX 2080TI GPU，但未报告具体训练时间或参数量。公平性方面存在明显局限：NGGP 在 6/9 数据集上缺失结果（标记为'-'），削弱对比完整性；未与 MAML [12]、Bayesian Neural Networks (PBP [20]) 等引用方法直接对比；更关键的是，缺少 2022-2025 年的现代深度概率方法（如 Transformer-based Neural Processes、Neural ODE 等）。作者声称"Table 4 中 12/12 设置最低 NLL"，但 Table 5 实际呈现的结果与此存在矛盾——Concrete 上的劣化未被提及。这些表明实验证据强度约为中等（0.6），结论需谨慎外推。

## 方法谱系与知识库定位

GSFP 属于**深度核学习 → 贝叶斯神经网络**的方法谱系，直接父方法为 **Sum of squares circuits** [25]（同作者前期工作），后者提供"平方"这一核心算法思想的 tractable probabilistic circuits 实现。本文将其从电路框架扩展为共轭贝叶斯推断框架，完成从表示学习到严格统计推断的跃迁。

**改变的插槽**：
- **prior_distribution**: GP(0, k) → GSF q(·|f) ∝ ‖f‖₂²（单峰 → 多峰 capable）
- **inference_strategy**: VI/MCMC 近似 → 闭式精确后验 ν(dω) = p(U|ω)μ(dω)
- **objective**: ELBO / GP 边缘似然 → 精确 GSF 边缘似然（经验贝叶斯）

**直接基线对比**：
- **Deep Kernel Learning / GP**: 本文替换其先验与推断引擎，保留神经网络特征提取
- **DKT [12]**: 同为深度核迁移，本文在多数数据集上 NLL 更低，但 Concrete 上落败
- **NGGP [17]**: 神经高斯过程，本文提供精确推断替代其近似方案
- **Normalizing flows**: 本文在推断 tractability 上更优，但表达力对比未充分验证

**后续方向**：(1) 将 GSF 先验扩展至非回归任务（生成模型、强化学习）；(2) 与最新深度概率方法（Neural Processes, Diffusion-based priors）的系统对比；(3) 理论分析 GSF 的多模态保证与泛化界。

**标签**：modality: tabular | paradigm: supervised Bayesian inference | scenario: few-shot regression, uncertainty quantification | mechanism: conjugate prior, closed-form posterior, empirical Bayes | constraint: exact inference tractability

