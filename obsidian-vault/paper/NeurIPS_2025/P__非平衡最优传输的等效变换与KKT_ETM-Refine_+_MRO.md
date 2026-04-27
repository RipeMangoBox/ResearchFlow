---
title: Solving Discrete (Semi) Unbalanced Optimal Transport with Equivalent Transformation Mechanism and KKT-Multiplier Regularization
type: paper
paper_level: A
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 非平衡最优传输的等效变换与KKT乘子正则化
- ETM-Refine + MRO
- ETM-Refine + MROT
- The Equivalent Transformation Mecha
acceptance: Poster
method: ETM-Refine + MROT
modalities:
- tabular
paradigm: optimization-based
---

# Solving Discrete (Semi) Unbalanced Optimal Transport with Equivalent Transformation Mechanism and KKT-Multiplier Regularization

**Topics**: [[T__Domain_Adaptation]] | **Method**: [[M__ETM-Refine_+_MROT]] | **Datasets**: Treatment Effect Estimation - ACIC, Treatment Effect Estimation - IHDP

> [!tip] 核心洞察
> The Equivalent Transformation Mechanism (ETM) can exactly determine marginal probabilities for SemiUOT and UOT with KL divergence, transforming them into classic OT problems, and when combined with KKT-Multiplier Regularization (MROT), achieves more accurate matching results than existing entropy-regularized approaches.

| 中文题名 | 非平衡最优传输的等效变换与KKT乘子正则化 |
| 英文题名 | Solving Discrete (Semi) Unbalanced Optimal Transport with Equivalent Transformation Mechanism and KKT-Multiplier Regularization |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/) · [Code](https://github.com/) · [Project](https://) |
| 主要任务 | Domain Adaptation, Treatment Effect Estimation |
| 主要 baseline | Ent-UOT / Ent-SemiUOT, Sinkhorn algorithm, ESCFR, UniOT |

> [!abstract] 因为「熵正则化非平衡最优传输产生密集且不准确的匹配解」，作者在「Chizat et al. 的熵正则化UOT框架」基础上改了「用ETM精确确定边缘分布并替换熵正则为KKT乘子正则化」，在「Office-Home universal domain adaptation」上取得「H-score 92.72%，相比UniOT提升+1.59」

- Office-Home UDA: H-score 92.72%，超越UniOT 91.13%（+1.59）和MROT-Ent变体92.31%（+0.41）
- IHDP处理效应估计: PEHE Out-Sample 1.146，优于ESCFR 1.282（-0.136）和MROT-Ent 1.275（-0.129）
- ACIC处理效应估计: PEHE In-Sample 2.104，优于ESCFR 2.252（-0.148）和MROT-Ent 2.327（-0.223）

## 背景与动机

最优传输（Optimal Transport, OT）是衡量两个概率分布之间差异的核心数学工具，在域适应、生成模型和因果推断中有广泛应用。然而，经典OT要求源分布和目标分布的总质量严格相等，这在实际应用中往往不成立——例如域适应中源域和目标域的样本数通常不同，处理效应估计中处理组和对照组的规模也可能不平衡。为此，研究者提出了非平衡最优传输（Unbalanced OT, UOT）和半非平衡最优传输（SemiUOT），通过KL散度松弛边缘约束来允许质量差异。

现有方法主要依赖熵正则化（Entropic Regularization）来求解UOT/SemiUOT问题。Chizat et al. [20] 的Scaling algorithms将Sinkhorn迭代扩展到非平衡场景，通过添加τ·KL(π1|a) + τ·KL(πᵀ1|b) + ε·H(π)的惩罚项来获得光滑的优化目标。这类方法计算高效，但存在根本性缺陷：熵正则化H(π) = -∑π_ij log π_ij 强制运输计划π保持正值，导致解极度密集（dense），大量微小的非零元素模糊了真实的对应关系，使得匹配结果在需要精确对齐的任务中表现不佳。此外，当松弛参数τ→∞时，现有方法无法精确确定边缘分布，只能得到近似解。

本文的核心动机正是解决这一"近似困境"：能否在不牺牲计算效率的前提下，精确确定UOT/SemiUOT的边缘分布，并设计一种不依赖熵正则化的替代机制来获得稀疏准确的匹配？作者通过分析KL散度松弛在极限情况下的行为，发现可以通过简单的比例缩放因子ωL精确确定边缘概率，进而将问题转化为经典OT；同时引入KKT乘子信息作为新的正则化项，最终实现了从"近似密集解"到"精确稀疏解"的跨越。

## 核心创新

核心洞察：UOT/SemiUOT中KL散度松弛的边缘约束在极限情况下（τ→∞）具有确定性的缩放行为，因为总质量比值的闭式表达使得精确边缘确定成为可能，从而使经典OT求解器可直接应用于非平衡场景，同时KKT乘子信息替代熵正则化进一步提升了匹配精度。

| 维度 | Baseline (Ent-UOT/Ent-SemiUOT) | 本文 (ETM-Refine + MROT) |
|:---|:---|:---|
| 边缘确定 | 迭代Sinkhorn近似，τ有限时边缘约束松弛 | ETM闭式计算ωL，τ→∞时精确确定π1 = ωL·a |
| 正则化机制 | 熵正则H(π) = -∑π_ij log π_ij，强制解密集 | KKT乘子正则G(π,s) = ⟨π,s⟩，利用对偶信息 |
| 求解策略 | 直接在非平衡问题上迭代，解空间为UOT | 两阶段：ETM转化为经典OT → MROT求解 |
| 解的性质 | 密集（dense）、不准确、含大量微小非零元 | 稀疏、精确匹配、边缘约束严格满足 |

## 整体框架


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/8fd6c3cf-ee4a-4127-9ee9-d40cf061b886/figures/Figure_1.png)
*Figure 1 (result): SemiEOT matching achieved in 2D data*



ETM-Refine + MROT采用清晰的两阶段流水线，将非平衡最优传输问题转化为可精确求解的经典OT问题：

**输入**：SemiUOT或UOT问题实例，包含成本矩阵C ∈ ℝ^{M×N}，源质量向量a ∈ ℝ^M_+，目标质量向量b ∈ ℝ^N_+，以及松弛参数τ（或τ_a, τ_b）。

**第一阶段：ETM边缘确定（ETM Marginal Determination）**
- 输入：原始非平衡问题 (C, a, b, τ)
- 核心计算：根据问题类型计算ωL缩放因子——SemiUOT用ωL = ⟨b,1_N⟩/⟨a,1_M⟩，UOT用ωL = √(⟨b,1⟩/⟨a,1⟩)
- 输出：精确边缘分布ā和b̄，使得π1_N = ā, π^⊤1_M = b̄严格成立，将原问题转化为经典OT

**第二阶段：MROT求解器（MROT Solver）**
- 输入：转化后的经典OT问题 (C, ā, b̄) 及KKT乘子s
- 核心操作：在目标函数中加入KKT乘子正则项η_G·⟨π,s⟩，替代传统熵正则项
- 变体选择：MROT-Ent（保留轻微熵正则）或MROT-Norm（纯KKT正则，更精确）
- 输出：稀疏准确的运输计划π

```
原始UOT/SemiUOT问题 (C,a,b,τ)
    ↓
[ETM阶段] 计算ωL缩放因子
    ├─ SemiUOT: ωL = ⟨b,1⟩/⟨a,1⟩,  ā = ωL·a
    └─ UOT: ωL = √(⟨b,1⟩/⟨a,1⟩),  ā = √ωL·a, b̄ = b/√ωL
    ↓
经典OT问题 (C, ā, b̄)
    ↓
[MROT阶段] 加入KKT乘子正则 η_G·⟨π,s⟩
    ├─ MROT-Ent: 保留轻微熵正则
    └─ MROT-Norm: 纯KKT乘子正则（推荐，更精确）
    ↓
稀疏运输计划 π
```

## 核心模块与公式推导

### 模块 1: ETM边缘确定机制（对应框架图 第一阶段）

**直觉**：当KL散度松弛参数τ→∞时，惩罚项主导目标函数，边缘约束的行为趋于确定；通过分析这一极限，可得到边缘分布的闭式表达，无需迭代近似。

**Baseline 公式** (Ent-UOT/Ent-SemiUOT [20]):
$$\min_{\pi \geq 0} \langle C, \pi \rangle + \tau \cdot \text{KL}(\pi\mathbf{1}_N | a) + \tau \cdot \text{KL}(\pi^\text{top}\mathbf{1}_M | b) + \varepsilon \cdot H(\pi)$$
符号: π ∈ ℝ^{M×N}_+ 为运输计划, C 为成本矩阵, a,b 为源/目标质量, τ 为KL松弛强度, ε 为熵正则系数, H(π) = -∑π_{ij}log π_{ij}

**变化点**：Baseline在有限τ下求解，边缘约束仅近似满足（π1 ≈ a, π^⊤1 ≈ b），且熵正则强制π密集。本文发现τ→∞时存在精确极限行为，可通过ωL缩放直接确定边缘。

**本文公式（推导）**:
$$\text{Step 1 (SemiUOT)}: \quad \omega_L = \frac{\langle b, \mathbf{1}_N \rangle}{\langle a, \mathbf{1}_M \rangle} \quad \text{（总质量比值，源质量不足时放大源边缘）}$$
$$\text{Step 2 (SemiUOT)}: \quad \bar{a} = \omega_L \cdot a, \quad \text{使得 } \pi\mathbf{1}_N = \bar{a} \text{ 精确成立}$$
$$\text{Step 1 (UOT)}: \quad \omega_L = \sqrt{\frac{\langle b, \mathbf{1} \rangle}{\langle a, \mathbf{1} \rangle}} \quad \text{（几何平均保持质量守恒）}$$
$$\text{Step 2 (UOT)}: \quad \bar{a} = \sqrt{\omega_L} \cdot a, \quad \bar{b} = \frac{b}{\sqrt{\omega_L}}, \quad \text{使得 } \pi\mathbf{1}_N = \bar{a}, \pi^\text{top}\mathbf{1}_M = \bar{b}$$
$$\text{最终}: \text{转化为经典OT: } \min_{\pi \geq 0} \langle C, \pi \rangle \text{ s.t. } \pi\mathbf{1} = \bar{a}, \pi^\text{top}\mathbf{1} = \bar{b}$$

**对应消融**：Figure 6(a)-(b)显示，使用精确ETM边缘确定相比近似方法显著降低绝对误差。

---

### 模块 2: KKT乘子正则化 MROT（对应框架图 第二阶段）

**直觉**：经典OT的对偶问题产生KKT乘子（对偶变量），这些乘子蕴含了约束优化的结构信息；将其显式引入目标函数，可引导运输计划学习更优的稀疏结构，避免熵正则的"过度平滑"。

**Baseline 公式** (Entropic OT [24]):
$$\min_{\pi \geq 0} \langle C, \pi \rangle + \varepsilon H(\pi) \text{ s.t. } \pi\mathbf{1} = \bar{a}, \pi^\text{top}\mathbf{1} = \bar{b}$$
符号: ε 为熵正则系数, H(π) 强制π>0以保证可微性

**变化点**：熵正则H(π)虽使问题光滑易解，但导致π所有元素非零，解密集且偏离真实最优。本文完全替换为正则化机制，利用KKT条件中的乘子信息s。

**本文公式（推导）**:
$$\text{Step 1 (对偶问题与KKT条件)}: \quad \mathcal{L}(\pi, f, g) = \langle C, \pi \rangle - \langle f, \pi\mathbf{1} - \bar{a} \rangle - \langle g, \pi^\text{top}\mathbf{1} - \bar{b} \rangle$$
$$\text{Step 2 (提取乘子)}: \quad s_{ij} = f_i + g_j - C_{ij} \quad \text{（互补松弛相关的KKT乘子信息）}$$
$$\text{Step 3 (加入MROT正则)}: \quad G(\pi, s) = \langle \pi, s \rangle = \sum_{ij} \pi_{ij} s_{ij}$$
$$\text{最终 (MROT目标)}: \quad \min_{\pi \geq 0} \langle C, \pi \rangle + \eta_G \cdot \langle \pi, s \rangle + \mathbb{1}_{\text{Ent}} \cdot \varepsilon H(\pi) \text{ s.t. 边缘约束}$$
其中 η_G 为KKT乘子正则系数，\mathbb{1}_{\text{Ent}} 为MROT-Ent变体的指示函数（MROT-Norm时取0）。

**对应消融**：Table 7显示，ESCFR + UOT(ETM-Refine + MROT-Norm)在ACIC AUUC In-Sample上达到0.883，而MROT-Ent变体仅0.839，差距0.044；在IHDP AUUC In-Sample上0.798 vs 0.769，差距0.029。Figure 5直观展示了η_G=100时KKT乘子正则化获得清晰匹配，而η_G∈{0,1}时效果差。

## 实验与分析


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/8fd6c3cf-ee4a-4127-9ee9-d40cf061b886/figures/Table_1.png)
*Table 1 (quantitative): Classification accuracy (%) on Office-31*



本文在三大任务上验证ETM-Refine + MROT的有效性：处理效应估计（ACIC/IHDP）、通用域适应（Office-Home）以及合成数据匹配。核心结果集中在Table 7（实验主表）中呈现。

**处理效应估计**：在ACIC基准上，ESCFR + UOT(ETM-Refine + MROT-Norm)取得PEHE In-Sample 2.104、Out-Sample 2.216，相比原始ESCFR（2.252 / 2.316）分别降低0.148和0.100；相比MROT-Ent变体（2.327 / 2.261）降低0.223和0.045。AUUC指标上优势更显著：ACIC In-Sample AUUC 0.883，超越ESCFR 0.796（+0.087）和MROT-Ent 0.839（+0.044）。IHDP基准呈现相似模式：PEHE Out-Sample 1.146，优于ESCFR 1.282（-0.136）和MROT-Ent 1.275（-0.129）；AUUC Out-Sample 0.802，超越ESCFR 0.719（+0.083）和MROT-Ent 0.763（+0.039）。值得注意的是，TARNet在ACIC In-Sample AUUC上略高（0.886 vs 0.883），但本文方法在Out-of-Sample泛化上更稳健。


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/8fd6c3cf-ee4a-4127-9ee9-d40cf061b886/figures/Figure_2.png)
*Figure 2 (result): Results of our method on synthetic data*



**通用域适应**：Office-Home基准上，UniOT + UOT(ETM-Refine + MROT-Norm)取得平均H-score 92.72%，超越原始UniOT 91.13%（+1.59），也优于MROT-Ent变体92.31%（+0.41），达到该基准上的最佳报告结果。


![Table 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/8fd6c3cf-ee4a-4127-9ee9-d40cf061b886/figures/Table_2.png)
*Table 2 (quantitative): Classification accuracy (%) on Office-31 and Image-CLEF for UDA*



**消融分析**：关键消融验证了各组件的必要性。（1）MROT-Norm vs MROT-Ent：非熵变体在多数指标上更优，ACIC AUUC In-Sample差距达0.044，证明去除残余熵正则的收益；（2）η_G超参数：Figure 5显示η_G=100时获得清晰匹配，η_G∈{0,1}时匹配模糊，说明足够的KKT乘子权重至关重要；（3）ϵ超参数：Figure 6(c)在Office-Home上显示较小ϵ提供更精确近似。

**公平性审视**：本文比较的基线ESCFR [17]和UniOT [16]是各自领域的代表性方法，但未直接与Chizat et al. [20]的scaling algorithm在相同任务上对比，也未纳入[21]的semi-dual UOT等近期求解器。计算复杂度分析虽有提供，但缺乏大规模问题上的实际运行时间对比。作者坦承方法限于KL散度松弛，ℓ1/ℓ2范数变体未处理；且η_G和ϵ的调参敏感性需要用户注意。

## 方法谱系与知识库定位

ETM-Refine + MROT属于**最优传输算法**谱系，直接继承自Chizat et al. [20]的熵正则化UOT scaling算法，但在三个核心slot上进行了根本性替换：objective（KL+熵正则 → KKT乘子正则）、inference_strategy（迭代Sinkhorn → 两阶段精确边缘确定+经典OT求解）、reward_design（H(π) → G(π,s)=⟨π,s⟩）。

**直接基线对比**：
- **Ent-UOT/Ent-SemiUOT [20]**：本文替换其熵正则化和近似边缘确定，保留KL散度的问题建模思想
- **Sinkhorn algorithm [24]**：本文继承其矩阵缩放的高效计算哲学，但完全绕过迭代过程，改用闭式变换
- **ESCFR [17]**：本文方法作为插件替换其内部的UOT求解器，实验设置保持一致
- **UniOT [16]**：同理，ETM-Refine + MROT嵌入其框架替代原有的OT模块

**后续方向**：（1）将ETM推广至ℓ1/ℓ2范数松弛的非平衡OT问题，突破KL散度的限制；（2）开发η_G的自适应选择策略，降低超参数敏感性；（3）结合神经网络参数化，将MROT扩展至连续测度的大规模场景。

**知识库标签**：modality=tabular | paradigm=optimization-based | scenario=domain_adaptation, causal_inference | mechanism=dual_regularization, exact_marginal_transformation | constraint=unbalanced_mass, discrete_setting

