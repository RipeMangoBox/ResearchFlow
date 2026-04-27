---
title: Pseudo-Labeling for Kernel Ridge Regression under Covariate Shift
type: paper
paper_level: A
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 协变量偏移下KRR的伪标签选择方法
- Pseudo-Labeling
- Pseudo-Labeling for KRR under Covariate Shift
- A pseudo-labeling approach that spl
acceptance: Poster
cited_by: 18
code_url: https://github.com/kw2934/KRR
method: Pseudo-Labeling for KRR under Covariate Shift
modalities:
- tabular
paradigm: supervised
---

# Pseudo-Labeling for Kernel Ridge Regression under Covariate Shift

[Code](https://github.com/kw2934/KRR)

**Topics**: [[T__Domain_Adaptation]] | **Method**: [[M__Pseudo-Labeling_for_KRR_under_Covariate_Shift]] | **Datasets**: Synthetic covariate shift problem

> [!tip] 核心洞察
> A pseudo-labeling approach that splits source data for training candidate models and an imputation model, then selects among candidates using pseudo-labels on target data, achieves minimax optimal excess risk up to a polylogarithmic factor.

| 中文题名 | 协变量偏移下KRR的伪标签选择方法 |
| 英文题名 | Pseudo-Labeling for Kernel Ridge Regression under Covariate Shift |
| 会议/期刊 | NeurIPS 2025 (Poster) · Annals of Statistics 2025 |
| 链接 | [arXiv](https://arxiv.org/abs/2302.10160) · [Code](https://github.com/kw2934/KRR) · [DOI](https://doi.org/10.1214/25-aos2566) |
| 主要任务 | Domain Adaptation（域适应），具体为协变量偏移(covariate shift)下的核岭回归 |
| 主要 baseline | Standard Kernel Ridge Regression [Hoerl and Kennard, 1970]、Feng et al. (2024) 统一核方法分析、Lepski-type adaptive methods [Blanchard et al., 2019] |

> [!abstract] 因为「协变量偏移下标准KRR无法利用无标签目标数据，且源分布上的模型选择无法适应目标分布」，作者在「Standard KRR」基础上改了「数据分割+插补模型生成伪标签+伪标签验证选择」的两阶段流程，在「合成协变量偏移问题(P与Q分布)」上取得「minimax最优超额风险，仅差polylogarithmic因子」

- 关键性能：理论保证超额风险达到minimax最优速率，有效样本量自动适应目标分布结构
- 关键性能：插补模型惩罚参数 λ̃ = O(n⁻¹)，专为模型选择精度优化而非预测精度
- 关键性能：候选惩罚集合 Λ 仅含 O(log n) 个几何序列参数，计算高效

## 背景与动机

在机器学习的实际应用中，训练数据与测试数据往往来自不同分布——这被称为**协变量偏移(covariate shift)**。例如，医疗预测模型在某医院(source)训练，却需部署到另一人群特征不同的医院(target)；由于特征分布 P(X) ≠ Q(X)，直接在源数据上训练的模型在目标域表现可能严重退化。

现有方法如何处理这一问题？**重要性加权(importance weighting)** [Cortes et al., 2010] 通过密度比调整源样本权重，但密度比估计本身困难且方差大。**Feng et al. (2024)** 提供了核方法在协变量偏移下的统一理论分析，但仅利用源数据，未使用无标签目标数据。**Lepski-type方法** [Blanchard et al., 2019] 自适应选择正则化参数，但同样局限于源分布验证，无法感知目标分布结构。

这些方法的共同短板在于：**无法有效利用无标签目标数据来指导模型选择**。标准KRR在源数据上用交叉验证选出的模型，面对目标分布可能过平滑或过粗糙；而目标域缺乏标签，传统验证无从进行。伪标签技术[Lee, 2013; Kumar et al., 2020]在深度学习中有成功应用，但如何适配到核方法、如何保证理论最优性，仍是开放问题。

本文的核心动机正是：**通过数据分割和专门优化的插补模型，生成适合模型选择（而非预测）的伪标签，从而在核方法框架下实现协变量偏移的自适应适应**。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/4bb0d4ea-584a-4f81-a72c-9c010d502918/figures/Figure_1.png)
*Figure 1 (motivation): Covariate shift and its adaptation*



## 核心创新

核心洞察：**插补模型的惩罚参数应优化模型选择精度（伪标签向量的偏差-方差权衡），而非传统预测精度**，因为伪标签验证只需"相对排序正确"而非"绝对值准确"，从而使不完美的伪标签仍能支持minimax最优的模型选择成为可能。

| 维度 | Baseline (Standard KRR) | 本文 |
|:---|:---|:---|
| 数据使用 | 全部源数据训练单一模型，目标数据完全不用 | 源数据分割为D1/D2，目标无标签数据用于验证选择 |
| 模型选择 | 交叉验证或理论公式在源分布上选惩罚参数 | 多候选模型 + 插补模型生成伪标签 + 目标域伪标签验证 |
| 插补模型目标 | 无插补模型 | λ̃ = O(n⁻¹)，优化模型选择精度而非预测MSE |
| 理论保证 | 源分布最优，目标分布可能次优 | 目标分布上minimax最优，自适应有效样本量 |

## 整体框架



本文提出两阶段伪标签KRR框架，数据流如下：

**输入**：带标签源数据 {(xᵢ, yᵢ)}ᵢ₌₁ⁿ ~ P，无标签目标数据 {x'ᵢ}ᵢ₌₁ⁿ⁰ ~ Q

**Step 1 — 数据分割**：将源数据随机划分为 D1（大小 n₁，用于候选模型）和 D2（大小 n₂ = n−n₁，用于插补模型）。这是关键设计：D2必须"牺牲"给插补模型，而非全部用于候选训练。

**Step 2 — 候选模型训练**：在 D1 上训练一族KRR模型 {f̂_λ}λ∈Λ，其中 Λ 为 O(log n) 个几何序列惩罚参数，范围从 O(n⁻¹) 到 O(1)。这些候选覆盖从欠平滑到过平滑的模型谱。

**Step 3 — 插补模型训练**：在 D2 上训练单一KRR模型 ẽf，惩罚参数 λ̃ = O(n⁻¹)。该模型不追求预测精度，而是优化生成伪标签的"模型选择友好性"。

**Step 4 — 伪标签生成与模型选择**：用 ẽf 为目标数据生成伪标签 ỹ'ᵢ = ẽf(x'ᵢ)，然后通过最小化伪标签验证损失选择最优候选：λ̂ ∈ argminλ (1/n₀)Σ|f̂_λ(x'ᵢ) − ỹ'ᵢ|²。

**输出**：选定的模型 f̂_λ̂，直接用于目标域预测。

```
源数据 {(xᵢ,yᵢ)} ~ P
    │
    ▼
[随机分割] ──→ D1 (n₁) ──→ [KRR多惩罚训练] ──→ {f̂_λ}λ∈Λ
         └──→ D2 (n₂) ──→ [KRR单惩罚训练, λ̃=O(n⁻¹)] ──→ ẽf
                                                    │
目标数据 {x'ᵢ} ~ Q ─────────────────────────────────┘
    │
    ▼
[伪标签生成] ỹ'ᵢ = ẽf(x'ᵢ)
    │
    ▼
[验证选择] λ̂ = argminλ (1/n₀)Σ|f̂_λ(x'ᵢ)−ỹ'ᵢ|²
    │
    ▼
输出: f̂_λ̂
```

## 核心模块与公式推导

### 模块 1: 候选KRR模型训练（对应框架图左侧D1分支）

**直觉**：生成覆盖不同平滑程度的模型谱，让后续选择有"候选可挑"。

**Baseline 公式** (Standard KRR):
$$\hat{f} = \text{arg}\min_{f \in \mathcal{F}} \left\{ \frac{1}{n} \sum_{i=1}^{n} |f(x_i) - y_i|^2 + \lambda \|f\|_{\mathcal{F}}^2 \right\}$$
符号: $\mathcal{F}$ = RKHS, $\lambda$ = 惩罚参数, $\|f\|_{\mathcal{F}}$ = RKHS范数

**变化点**：标准KRR用全部数据训练单一模型；本文将数据限制在D1，并扩展为**多候选集合**，惩罚参数从单点变为几何序列。

**本文公式**:
$$\hat{f}_\lambda = \text{arg}\min_{f \in \mathcal{F}} \left\{ \frac{1}{|D_1|} \sum_{(x,y) \in D_1} |f(x) - y|^2 + \lambda \|f\|_{\mathcal{F}}^2 \right\}, \quad \forall \lambda \in \Lambda$$
其中 $\Lambda = \{\lambda_j = \lambda_0 \cdot r^j : j = 0,1,...,O(\log n)\}$，从 $O(n^{-1})$ 到 $O(1)$。

---

### 模块 2: 插补模型与伪标签生成（对应框架图D2分支+目标数据）

**直觉**：用D2训练一个"专门用来生成验证标签"的模型，其优化目标与传统预测不同。

**Baseline 公式** (Standard validation with true labels):
$$\hat{\lambda} \in \text{arg}\min_{\lambda} \frac{1}{n_{val}} \sum_{(x,y) \in D_{val}} |\hat{f}_\lambda(x) - y|^2$$

**变化点**：目标域无真实标签，传统验证不可行。需用插补模型 ẽf 生成伪标签 ỹ'，但**关键洞察**是：λ̃ 的选择不应最小化预测MSE，而应最小化对模型选择的影响。

**本文公式（推导）**:

$$\text{Step 1 (插补模型训练)}: \quad \tilde{f} = \text{arg}\min_{f \in \mathcal{F}} \left\{ \frac{1}{|D_2|} \sum_{(x,y) \in D_2} |f(x) - y|^2 + \tilde{\lambda} \|f\|_{\mathcal{F}}^2 \right\} \quad \text{加入}\tilde{\lambda}=O(n^{-1})\text{以控制方差}$$

$$\text{Step 2 (伪标签生成)}: \quad \tilde{y}'_i = \tilde{f}(x'_i), \quad i \in [n_0] \quad \text{用插补输出替代真实目标标签}$$

$$\text{Step 3 (伪标签验证选择)}: \quad \hat{\lambda} \in \text{arg}\min_{\lambda \in \Lambda} \left\{ \frac{1}{n_0} \sum_{i=1}^{n_0} |\hat{f}_\lambda(x'_i) - \tilde{y}'_i|^2 \right\} \quad \text{重归一化保证选择一致性}$$

**理论关键**：模型选择精度取决于伪标签向量 $\tilde{y}'$ 的**偏差**和**方差代理量(variance proxy)**，而非单个点的MSE。λ̃ = O(n⁻¹) 使偏差足够小、方差适中，从而保证相对排序正确。

**对应消融**：Table 1 显示三个候选（欠平滑/过平滑/最优）在P和Q上的超额风险差异，验证伪标签选择能识别接近最优的候选。

## 实验与分析


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/4bb0d4ea-584a-4f81-a72c-9c010d502918/figures/Table_1.png)
*Table 1 (comparison): Error rate of three candidates on P and Q*



本文在**合成协变量偏移问题**上评估，设定明确的源分布P和目标分布Q。如Table 1所示，三个候选模型（Candidate 1欠平滑、Candidate 2过平滑、Candidate 3接近最优）在源分布P和目标分布Q上表现出显著不同的超额风险。核心发现是：**在源分布P上表现最优的模型，在目标分布Q上未必最优**，这直接验证了协变量偏移下源验证的失效。本文的伪标签选择机制能够识别出在目标分布Q上接近最优的候选，其理论保证为minimax最优速率，仅差polylogarithmic因子。



从理论角度，超额风险的收敛速率自适应于**有效样本量(effective sample size)**，该量由目标分布Q与源分布P之间的协变量偏移程度决定。当偏移较小时，方法自动获得更快收敛；当偏移较大时，速率相应调整，这是标准KRR无法实现的适应性。

**计算效率**：每个KRR有闭式解，复杂度O(n³)；本文需两次KRR求解（候选族+插补模型）及O(n₀ log n)的验证选择，总体仍保持多项式时间。存储方面，O(log n)个候选各需O(n)系数，总计O(n log n)。

**公平性检验**：实验部分存在一定局限。Table 1仅展示三个候选的对比，未明确显示"本文方法"与[Feng et al., 2024]等强基线的直接数值比较；缺乏与重要性加权[Cortes et al., 2010]、标签传播[Cai et al., 2021]等方法的实验对比。理论中的λ̃ = O(n⁻¹)指导是渐近的，有限样本下可能需要校准。此外，实验仅限于合成设定，未在真实域适应benchmark（如Office-31等）上验证。

## 方法谱系与知识库定位

本文属于**核方法(kernel methods)** + **域适应(domain adaptation)**的方法族，直接父方法为**Kernel Ridge Regression** [Hoerl and Kennard, 1970]。

**改变的slots**：
- **data_pipeline**：源数据全用 → 分割为D1/D2，目标数据从不用 → 用于伪标签验证
- **training_recipe**：单模型单惩罚 → 多候选几何序列 + 专用插补模型
- **objective**：预测精度导向 → 模型选择精度导向的偏差-方差权衡
- **inference_strategy**：直接应用 → 伪标签验证后选择

**直接基线与差异**：
- **[Feng et al., 2024]**：统一核方法协变量偏移分析；本文扩展其引入无标签目标数据的伪标签利用
- **[Blanchard et al., 2019] Lepski-type方法**：自适应选择正则化，但仅在源分布；本文将适应性延伸到目标分布
- **[Lee, 2013] / [Kumar et al., 2020]**：深度学习中的伪标签；本文首次系统引入核方法并给出minimax理论保证
- **[Huang et al., 2006]**：用无标签数据修正选择偏差；本文聚焦KRR的模型选择而非密度比估计

**后续方向**：(1) 扩展到其他核方法（如核SVM、高斯过程）；(2) 设计有限样本下λ̃的自适应选择规则，替代渐近O(n⁻¹)指导；(3) 结合深度特征的核方法，在真实高维域适应数据上验证。

**标签**：modality=tabular | paradigm=supervised + semi-supervised proxy | scenario=domain adaptation under covariate shift | mechanism=pseudo-labeling + data splitting + imputation model | constraint=theoretical guarantee, minimax optimality, closed-form computation

