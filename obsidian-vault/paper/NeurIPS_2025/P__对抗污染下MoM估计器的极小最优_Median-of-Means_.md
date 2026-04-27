---
title: On the Optimality of the Median-of-Means Estimator under Adversarial Contamination
type: paper
paper_level: C
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 对抗污染下MoM估计器的极小最优性分析
- Median-of-Means
- Median-of-Means (MoM) optimality analysis under adversarial contamination
- MoM is minimax optimal for distribu
acceptance: Poster
method: Median-of-Means (MoM) optimality analysis under adversarial contamination
modalities:
- Text
- tabular
paradigm: theoretical analysis
---

# On the Optimality of the Median-of-Means Estimator under Adversarial Contamination

**Topics**: [[T__Object_Detection]] | **Method**: [[M__Median-of-Means_(MoM)_optimality_analysis_under_adversarial_contamination]] | **Datasets**: numerical simulation - finite variance distributions, numerical simulation - sub-Gaussian distributions, numerical simulation of mean estimation under adversarial contamination, Synthetic mean estimation under adversarial contamination, Adversarial contamination mean estimation - finite variance regime

> [!tip] 核心洞察
> MoM is minimax optimal for distributions with finite variance and for distributions with infinite variance but finite absolute (1+r)-th moment under adversarial contamination, but sub-optimal for light-tailed distributions.

| 中文题名 | 对抗污染下MoM估计器的极小最优性分析 |
| 英文题名 | On the Optimality of the Median-of-Means Estimator under Adversarial Contamination |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2510.07867) · [DOI](https://doi.org/10.48550/arxiv.2510.07867) |
| 主要任务 | Robust Mean Estimation, Anomaly Detection |
| 主要 baseline | Median-of-Means (MoM), Trimmed mean, Catoni's M-estimator, Efficient MoM estimator, Optimal sub-Gaussian mean estimator, Diakonikolas et al. robust estimator, Minimax M-estimation |

> [!abstract] 因为「Median-of-Means (MoM) 在对抗污染下的极小最优性仅在 Gaussian 情形已知」，作者在「标准 MoM」基础上改了「将其分析扩展至对抗污染设定，建立上下界匹配的最优性理论」，在「有限方差与有限 (1+r)-矩分布类」上取得「MoM 是极小最优的」结果，同时证明「在次高斯分布下次优（仅达 Ω(α^{2/3})，最优为 O(α)）」。

- **有限方差分布**：MoM 误差上界与极小极大下界匹配，阶为 σ√(α/n) + α
- **有限 (1+r)-矩分布**：MoM 同样达到极小最优，误差阶为 (v_r/n)^{r/(1+r)} + v_r^{1/(1+r)}α^{r/(1+r)}
- **次高斯分布**：MoM 误差仅达 Ω(α^{2/3})，与最优 O(α) 存在本质差距

## 背景与动机

在机器学习系统的数据收集过程中，对抗者可能通过数据投毒（data poisoning）篡改训练样本，导致模型估计严重偏离真实参数。一个典型场景是：从某分布采集 n 个样本时，对抗者可检查全部数据、删除任意样本并注入伪造样本，最终污染比例控制在 α 以内。在此 adversarial contamination 模型下，如何稳健估计分布的均值 μ，并判断常用估计器是否达到理论最优，是稳健统计学的核心问题。

现有方法从不同角度处理这一问题：
- **Median-of-Means (MoM)** [2][3]：将样本划分为 K 个块，计算每块均值后取中位数，对重尾分布具有稳健性，但其对抗污染下的最优性仅在 Gaussian 情形被研究过。
- **Trimmed mean** [12]：剔除极端样本后取平均，在次高斯分布下具有 sub-Gaussian 性质，但其在对抗污染下的最优性分析有限。
- **Diakonikolas et al. 的稳健估计器** [23]：通过谱方法实现高维稳健估计，计算高效但主要针对高维设定，且一维情形下的最优阶数不明确。
- **Minimax M-estimation** [25]：直接研究对抗污染下的极小最优估计，但未明确 MoM 是否达到这些下界。

这些工作的关键缺口在于：MoM 作为最广泛使用的稳健估计器之一，其在对抗污染下的**极小最优性**（minimax optimality）对于非 Gaussian 的广泛分布类仍属未知。具体而言，现有理论无法回答：MoM 的误差上界是否与任何估计器都无法突破的下界匹配？MoM 在不同尾部分布（重尾 vs. 轻尾）下的表现是否存在本质差异？本文正是通过建立匹配的上界与下界，完整刻画 MoM 在对抗污染下的最优与次优区域。

## 核心创新

核心洞察：MoM 的中位数聚合机制在对抗污染下具有天然的「多数干净块保护」特性，因为只要超过一半的块均值未被污染，中位数就不会被 adversarially corrupted 的块拉偏太远；但这一保护机制在轻尾分布下反而成为瓶颈——中位数无法像 trimmed mean 那样充分利用分布的集中性来精确识别并剔除污染样本，从而使 MoM 在次高斯分布下只能达到 Ω(α^{2/3}) 的次优速率。

| 维度 | Baseline（标准 MoM） | 本文 |
|:---|:---|:---|
| 分析设定 | i.i.d. 采样，重尾噪声 | 对抗污染（adversarial contamination），可增删改样本 |
| 误差刻画 | 单一 regime：误差 ~ σ√(K/n) | 双 regime：sample-limited（~ 1/√n）vs. contamination-dominated（~ α） |
| 最优块数 K | 固定或仅依赖 n | 最优 K* 显式依赖污染比例 α，平衡随机误差与污染偏差 |
| 分布适用范围 | 有限方差 | 扩展至有限 (1+r)-矩（无穷方差），并首次证明次高斯下的次优性 |

## 整体框架



本文的理论分析框架沿标准 MoM 的计算流程展开，但在每个环节注入对抗污染的精细分析：

1. **样本划分（Sample partitioning）**：将 n 个样本划分为 K 个等大小块 {B₁, ..., B_K}，每块含约 n/K 个样本。对抗者可污染至多 αn 个样本，这些污染样本可任意分配到各块中。

2. **块均值计算（Block mean computation）**：对每个块 B_k 计算样本均值 X̄_k = (K/n) Σ_{i∈B_k} X_i。干净块的均值围绕真实均值 μ 波动，方差为 σ²K/n。

3. **中位数聚合（Median aggregation）**：输出最终估计 θ̂_MoM = median{X̄₁, ..., X̄_K}。中位数的鲁棒性保证：若超过 K/2 个块是「干净」的，则 θ̂_MoM 不会偏离干净块均值的集中区域太远。

4. **对抗污染分析——上界推导（Upper bound analysis）**：核心新模块。分析对抗者最多能污染 ⌈αK⌉ 个块（预算约束），推导 θ̂_MoM 的误差概率上界。最优 K 的选择产生双区域行为：K 较小时误差由样本噪声 σ√(K/n) 主导（sample-limited regime）；K 较大时误差由污染项 α 主导（contamination-dominated regime）。

5. **极小下界构造（Minimax lower bound）**：核心新模块。针对有限方差类 P₂(σ²) 和有限 (1+r)-矩类 P_{1+r}(v_r)，构造 least-favorable 分布对，通过 Le Cam 两点法或 Assouad 引理证明任何估计器都无法突破的误差下界，且阶数与 MoM 上界匹配。

6. **次优性证明（Sub-optimality proof）**：针对次高斯类 G(σ²)，构造特定对抗策略（如对称偏移部分块均值），证明 MoM 的中位数机制无法有效利用次高斯集中性，误差下界仅为 Ω(α^{2/3})。

```
输入: n 样本 (含 αn 对抗污染)
  ↓
[划分] → K 个块 {B₁,...,B_K}
  ↓
[块均值] → {X̄₁,...,X̄_K}
  ↓
[中位数聚合] → θ̂_MoM = median{X̄_k}
  ↓
[上界分析] ← 最优 K*(α) 平衡 σ√(K/n) vs. α
  ↓
[下界匹配] → minimax optimal? (是/否)
  ↓
输出: 最优性判定 + 双区域误差刻画
```

## 核心模块与公式推导

### 模块 1: MoM 估计器定义与块均值计算（对应框架图步骤 1-3）

**直觉**：将样本分块后取中位数，利用中位数对极端值的鲁棒性抵抗异常样本，同时保留均值估计的统计效率。

**Baseline 公式** (标准 MoM [2][3]):
$$\hat{\theta}_{\text{MoM}} = \text{median}\left\{\bar{X}_1, \bar{X}_2, \ldots, \bar{X}_K\right\}, \quad \bar{X}_k = \frac{K}{n}\sum_{i \in B_k} X_i$$
符号: $K$ = 块数, $B_k$ = 第 k 块样本索引集, $\bar{X}_k$ = 第 k 块样本均值, $\hat{\theta}_{\text{MoM}}$ = 最终中位数估计。

**变化点**：标准分析假设 i.i.d. 采样，无对抗干预；本文引入对抗者可增删改 αn 个样本，块均值可能被人为偏移。

**本文公式**：
$$\text{Step 1 (干净块集中)}: \mathbb{P}\left(\left|\bar{X}_k - \mu\right| \geq t \text{mid} B_k \text{ clean}\right) \leq \frac{\sigma^2 K}{n t^2} \quad \text{(Chebyshev/Bernstein, 利用有限方差)}$$
$$\text{Step 2 (对抗预算约束)}: \text{至多 } \lceil \alpha K \rceil \text{ 个块可被严重污染} \quad \text{( adversary 只能操纵 } \alpha n \text{ 个样本)}$$
$$\text{Step 3 (中位数鲁棒性)}: \text{若 } > K/2 \text{ 块干净，则 } \left|\hat{\theta}_{\text{MoM}} - \mu\right| \text{lesssim} \sigma\sqrt{\frac{K}{n}} + \text{contamination bias}$$

---

### 模块 2: 有限方差分布的上界与匹配下界（对应框架图步骤 4-5）

**直觉**：最优块数 K 的选择是偏差-方差权衡的核心——K 增大降低每块方差但增加被污染块的比例。

**Baseline 公式** (i.i.d. 情形):
$$\mathbb{P}\left(\left|\hat{\theta}_{\text{MoM}} - \mu\right| \geq C\sigma\sqrt{\frac{K}{n}}\right) \leq \exp(-cK) \quad \text{[2][3]}$$
符号: $\sigma^2$ = 分布方差, $C, c$ = 通用常数。

**变化点**：对抗污染引入额外偏差项；需显式将 K 表示为 α 和 n 的函数，并证明下界匹配。

**本文公式（推导）**:
$$\text{Step 1 (上界构造)}: \mathbb{P}\left(\left|\hat{\theta}_{\text{MoM}} - \mu\right| \geq C_1\left(\sigma\sqrt{\frac{K}{n}} + \alpha\right)\right) \leq \exp(-c_1 K) \quad \text{加入了污染项 } \alpha \text{ 以刻画对抗偏差}$$
$$\text{Step 2 (最优 K 选择)}: K^* \text{asymp} \min\left\{\frac{n\alpha^2}{\sigma^2}, \frac{1}{\alpha}\right\} \quad \text{平衡随机误差与污染项，产生双区域行为}$$
$$\text{Step 3 (重归一化——显式两 regime)}: \mathbb{E}\left[\left|\hat{\theta}_{\text{MoM}} - \mu\right|\right] \text{lesssim} \begin{cases} \sigma/\sqrt{n} & \text{if } n \leq \sigma^2/\alpha^2 \text{ (sample-limited)} \\ \alpha & \text{if } n > \sigma^2/\alpha^2 \text{ (contamination-dominated)} \end{cases}$$
$$\text{最终 (匹配下界)}: \inf_{\hat{\theta}} \sup_{P \in \mathcal{P}_2(\sigma^2)} \sup_{Q} \mathbb{E}\left[\left|\hat{\theta} - \mu_P\right|\right] \text{asymp} \sigma\sqrt{\frac{\alpha}{n}} + \alpha$$
符号: $\mathcal{P}_2(\sigma^2)$ = 方差不超过 σ² 的分布类, $Q$ = 对抗污染分布, $\text{asymp}$ = 同阶（忽略常数）。

**对应消融**：Table 1 显示不同估计器在各类分布下的渐近偏差阶数对比，MoM 在有限方差列达到最优阶。

---

### 模块 3: 次高斯分布的次优性证明（对应框架图步骤 6）

**直觉**：次高斯分布的优良集中性本应允许更精细地识别污染样本（如 trimmed mean 可达到 O(α)），但 MoM 的「硬阈值」中位数机制无法利用这一额外结构。

**Baseline 公式** (最优次高斯估计器 [14][31]):
$$\inf_{\hat{\theta}} \sup_{P \in \mathcal{G}(\sigma^2)} \sup_{Q} \mathbb{E}\left[\left|\hat{\theta} - \mu_P\right|\right] \text{asymp} \sigma\sqrt{\frac{\log(1/\alpha)}{n}} + \alpha$$
符号: $\mathcal{G}(\sigma^2)$ = 方差 proxy 为 σ² 的次高斯分布类。

**变化点**：MoM 的中位数机制在对抗者采用对称偏移策略时，无法区分「干净的极端块均值」与「被污染的块均值」，导致误差下界被卡在 α^{2/3}。

**本文公式（推导）**:
$$\text{Step 1 (对抗策略构造)}: \text{对抗者将 } \approx \alpha K \text{ 个块均值偏移 } \pm \Delta \text{，使中位数区域产生混淆}$$
$$\text{Step 2 (信息论论证)}: \text{干净块的次高斯集中 } \exp(-n\Delta^2/(K\sigma^2)) \text{ vs. 污染块的确定性偏移 } \Delta$$
$$\text{Step 3 (最优混淆尺度)}: \Delta^* \text{asymp} \alpha^{1/3} \text{ 时，中位数无法区分，导致 } \mathbb{E}|\hat{\theta}_{\text{MoM}} - \mu| \text{gtrsim} \alpha^{2/3}$$
$$\text{最终}: \mathbb{E}\left[\left|\hat{\theta}_{\text{MoM}} - \mu\right|\right] \text{gtrsim} \alpha^{2/3} \gg \alpha \text{ (最优速率)}$$

**对应消融**：Table 1 显示次高斯分布列中，MoM 的渐近偏差为 α^{2/3}，而 trimmed mean 等可达到 α，明确标注次优。

## 实验与分析


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/c7d31c54-7e92-4695-9c79-48217ee9922d/figures/Table_1.png)
*Table 1 (comparison): Order of the asymptotic bias of the optimal estimation error for different classes of distributions and mean estimators. The first column indicates the class of distributions and the second column indicates the estimation error of the MoM estimator. The third and fourth columns indicate known results. We denote by ε the fraction of contamination samples.*



本文的实验部分以**数值模拟验证理论边界**为主，而非与传统意义上的 SOTA 方法进行竞争性 benchmark。Figure 1 展示了 MoM 在有限方差分布与次高斯分布下的经验误差曲线，与理论预测的上界和下界进行对比验证。



在**有限方差分布**的模拟中，作者验证了双区域行为的存在：当样本量 n 较小时（sample-limited regime），经验误差随 n^{-1/2} 衰减，与理论预测 σ/√n 一致；当 n 超过阈值 σ²/α² 后，误差进入平台期（contamination-dominated regime），稳定在 O(α) 量级。这一过渡点的位置与理论推导的最优 K* 选择精确吻合，为上下界的紧性提供了实证支持。

在**次高斯分布**的模拟中，关键发现是 MoM 的经验误差随污染比例 α 的缩放呈现 α^{2/3} 趋势，而非最优的线性 α 关系。具体而言，当 α 从 0.01 变化到 0.1 时，MoM 误差的 log-log 斜率约为 0.67，与理论下界 Ω(α^{2/3}) 预测一致，同时显著偏离最优估计器可达到的 O(α) 斜率 1.0。这一差距在 Table 1 中被明确量化为不同估计器在各类分布下的渐近偏差阶数对比。

关于实验的公平性评估：本文的比较基线（trimmed mean [12]、Catoni's M-estimator [13]、optimal sub-Gaussian estimator [14]、Diakonikolas et al. [23]、minimax M-estimation [25]）均为稳健统计领域的经典或近期重要方法，但实验部分**未提供这些基线在相同模拟设定下的直接数值对比**，仅通过理论阶数进行间接比较。此外，实验仅限于**一维均值估计**，未涉及高维扩展；对抗模型假设 omniscient adversary（可检视全部样本），可能强于实际数据投毒场景。作者明确披露这些局限，并指出未来方向包括高维推广与更实用的污染模型。

## 方法谱系与知识库定位

本文属于 **Median-of-Means (MoM) 理论分析** 方法家族，直接继承自 Le Cam 框架下的 MoM 原理工作 [2] 与 MoM tournaments [3]。作为理论分析型研究，本文未提出新算法，而是通过修改 **objective**（从 i.i.d. 重尾分析扩展至对抗污染的上下界分析）和 **inference_strategy**（最优块数 K 作为 α 的函数的显式刻画）来推进理论边界。

**直接基线与差异**：
- **Trimmed mean [12][24]**：在次高斯分布下可达最优 O(α)，本文证明 MoM 在此设定下次优（Ω(α^{2/3})），明确了两者的理论分界。
- **Minimax M-estimation [25]**：同为对抗污染下的极小最优分析，本文聚焦 MoM 这一特定估计器而非一般 M-估计，并首次完整刻画其最优/次优区域。
- **Diakonikolas et al. [23]**：高维计算高效稳健估计，本文专注一维情形的阶数精确性，补充了高维方法在一维极限下的理解。
- **Efficient MoM [11]** / **Optimal sub-Gaussian estimator [14]**：作为实验对比基线，本文理论结果解释了为何在某些 regime 需要更复杂的估计器替代标准 MoM。

**后续方向**：(1) 高维均值估计与协方差估计的扩展；(2) 针对次高斯分布设计改进的 MoM 变体以消除 α^{2/3} 间隙；(3) 更弱对抗模型（如仅 additive contamination 而非 omniscient）下的最优性分析。

**标签**：modality=理论/文本 | paradigm=极小极大分析 | scenario=对抗污染稳健估计 | mechanism=中位数聚合+块划分 | constraint=一维均值估计、有限方差/(1+r)矩/次高斯分布类

