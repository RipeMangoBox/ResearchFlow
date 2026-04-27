---
title: Differentially Private Relational Learning with Entity-level Privacy Guarantees
type: paper
paper_level: C
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 实体级差分隐私关系学习
- Differentially P
- Differentially Private Relational Learning with Entity-level Privacy Guarantees
acceptance: Poster
method: Differentially Private Relational Learning with Entity-level Privacy Guarantees
---

# Differentially Private Relational Learning with Entity-level Privacy Guarantees

**Method**: [[M__Differentially_Private_Relational_Learning_with_Entity-level_Privacy_Guarantees]] | **Datasets**: Relation prediction with entity-level DP, Zero-shot relation prediction

| 中文题名 | 实体级差分隐私关系学习 |
| 英文题名 | Differentially Private Relational Learning with Entity-level Privacy Guarantees |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2506.08347) · [DOI](https://doi.org/10.48550/arxiv.2506.08347) |
| 主要任务 | 关系预测、零样本关系预测（实体级差分隐私保护） |
| 主要 baseline | DP-SGD、Group DP-SGD、Non-private baseline、Degree-capped baseline |

> [!abstract] 因为「标准 DP-SGD 仅提供记录级隐私保护，无法保护关系数据中同一实体关联的多条记录」，作者在「DP-SGD」基础上改了「基于耦合（coupling）的实体级梯度聚合机制与度截断（degree capping）预处理」，在「关系预测与零样本关系预测基准」上取得「更优的隐私-效用权衡」。

- **核心机制**：构造满足实体级隐私约束的耦合 π，替代标准 DP-SGD 的逐记录梯度裁剪
- **关键组件**：度截断模块控制实体级敏感度，用于零样本关系预测场景
- **隐私保证**：从记录级（record-level）提升至实体级（entity-level）差分隐私

## 背景与动机

关系数据（如知识图谱、社交网络）中的隐私保护面临一个根本性挑战：同一实体往往通过多条记录参与训练。例如，一个用户在推荐系统中可能产生数百条交互记录，或一个实体在知识图谱中通过多种关系与数十个其他实体相连。标准差分隐私方法（如 DP-SGD）将每条记录视为独立的隐私单位，提供记录级保护——即移除单条记录不会显著改变模型输出。然而，攻击者仍可能通过聚合同一实体的所有关联记录来推断该实体的敏感信息。

现有方法如何应对这一问题？**DP-SGD**（Abadi et al., 2016）作为最广泛使用的差分隐私训练框架，对每条样本的梯度进行裁剪并添加高斯噪声，但其隐私保证仅针对单条记录，无法覆盖实体关联的多条记录。**Group DP-SGD** 通过将同一实体的所有记录视为一个"组"来提供实体级隐私，但组的大小（即实体的度）可能极大，导致敏感度与噪声量随组规模线性增长，隐私成本急剧恶化。此外，关系学习中常见的**零样本关系预测**任务——预测训练时未见过的关系类型——进一步放大了敏感度控制难题，因为新关系可能涉及高度连接的枢纽实体（hub entities）。

这些方法的共同瓶颈在于：**隐私单位（privacy unit）与数据建模单位不匹配**。记录级方法忽略了关系数据的结构性关联；组级方法虽匹配实体单位，但未利用关系结构的统计特性来紧致（tighten）敏感度边界。本文的核心动机正是填补这一空白：通过耦合理论（coupling theory）重新构造实体级的隐私机制，使得噪声校准能够精确反映实体在关系结构中的实际参与度，而非保守地假设最坏情况。

本文提出一种基于耦合的实体级差分隐私训练框架，通过构造有效的耦合分布 π 来实现梯度聚合的隐私保护，并引入度截断预处理以控制零样本场景下的敏感度爆炸。

## 核心创新

核心洞察：通过构造满足边缘分布约束的耦合（coupling）π，可以将实体级隐私要求转化为可计算的敏感度边界，因为耦合允许我们在不改变整体数据分布统计特性的前提下，精确控制单个实体替换时的输出变化量，从而使实体级差分隐私在关系学习中具备可实现的训练效率。

| 维度 | Baseline (DP-SGD / Group DP-SGD) | 本文 |
|:---|:---|:---|
| **隐私单位** | 单条记录（DP-SGD）或固定大小组（Group DP-SGD） | 完整实体及其所有关联记录 |
| **敏感度计算** | 逐记录梯度范数裁剪，或组大小作为敏感度乘数 | 通过耦合 π 的转移概率精确界定实体替换影响 |
| **噪声校准** | 基于记录数或组大小的全局噪声尺度 | 基于实体参与度的自适应噪声，配合度截断紧致边界 |
| **数据预处理** | 无结构感知处理 | 度截断（degree capping）控制枢纽节点敏感度 |
| **理论保证** | (ε, δ)-DP 在记录/组级别 | 实体级 RDP 边界，支持更紧的组合分析 |

## 整体框架


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/04a0a89c-1057-49a6-9ef5-ee6928b4d425/figures/Figure_1.png)
*Figure 1 (pipeline): The main batch sampling process of relational learning in our framework (Best viewed in color). Given a graph, we first sample three positive sampling, and then construct negative edges based on each positive sample, and then perform score-based filtering, last, we report gradients over the sampled positive and negative sampling.*



本文框架包含四个核心阶段，数据流如下：

**输入**：原始关系图 G = (V, E, R)，包含实体集 V、关系集 R 和边集 E；隐私参数 (ε, δ)。

**阶段一：关系数据预处理（Relational data preprocessing）**
- 输入：原始关系图
- 输出：度截断后的关系图 G'
- 作用：对高度连接的枢纽实体进行度截断（degree capping），将每个实体的邻居数量上限设为 K_d，从而边界化实体级敏感度。此步骤是本文新增的关键预处理模块。

**阶段二：耦合式差分隐私机制（Coupling-based DP mechanism）**
- 输入：实体级梯度（同一实体的所有关联记录梯度聚合）
- 输出：隐私化后的聚合梯度
- 作用：构造有效耦合 π，使得实体替换时的输出分布变化被精确界定，替代标准 DP-SGD 的逐记录梯度裁剪与噪声添加。

**阶段三：隐私模型训练（Private model training）**
- 输入：度截断图 G'、隐私参数 (ε, δ)、耦合机制 π
- 输出：训练好的关系预测模型
- 作用：基于耦合机制聚合的梯度进行参数更新，使用实体级 RDP 会计进行隐私预算追踪。

**阶段四：零样本关系预测（Zero-shot relation prediction）**
- 输入：训练好的模型、未见关系类型
- 输出：新关系的预测结果
- 作用：评估模型在隐私保护下对训练时未出现关系的泛化能力。

```
原始关系图 G ──→ [度截断] ──→ G' ──→ [实体级梯度提取]
                                              ↓
隐私参数 (ε,δ) ──→ [耦合机制 π] ←── 实体梯度聚合
                                              ↓
                                    [噪声添加] ──→ 隐私梯度
                                              ↓
                                    [模型训练] ──→ 训练模型
                                              ↓
                                    [零样本推理] ──→ 预测结果
```

## 核心模块与公式推导

### 模块 1: 耦合构造与实体级隐私有效性（对应框架图阶段二）

**直觉**：标准 DP-SGD 的隐私保证依赖于单条记录的替换敏感性，但关系数据中实体替换会同时改变多条关联记录。耦合理论允许我们将这种"多记录联动变化"建模为联合分布的边际约束，从而精确计算实体替换的真实影响而非保守上界。

**Baseline 公式** (DP-SGD):
$$\hat{g} = \frac{1}{B}\left(\sum_{i \in B} \text{clip}(\nabla L_i, C) + \mathcal{N}(0, \sigma^2 C^2 I)\right)$$

符号：$B$ = batch 大小，$\nabla L_i$ = 第 $i$ 条记录的梯度，$C$ = 裁剪阈值，$\sigma$ = 噪声乘数，$I$ = 单位矩阵。

**变化点**：DP-SGD 的隐私分析将 $i$ 视为独立记录，敏感度为 $2C$。当实体 $x'$ 关联多条记录时，简单应用 Group DP 会将敏感度放大为 $2C \cdot |x'|$（$|x'|$ 为实体关联记录数），导致噪声量不可接受。本文改为构造耦合 $\pi$，使得实体替换的分布变化通过耦合的转移概率精确刻画。

**本文公式（推导）**：

$$\text{Step 1}: \quad P_{\vec{y}} \pi(\vec{y}, y'|\vec{x}, x') = \omega(y'|A_0, x_0) = p_{n-1,L} \quad \text{（耦合有效性：联合分布边缘化等于条件概率）}$$

$$\text{Step 2}: \quad p_{n-1,L} = p_{n-1,|x'|} \quad \text{（利用路径长度 } L \text{ 与实体大小 } |x'| \text{ 的组合等价性，建立实体级度量）}$$

$$\text{Step 3}: \quad \omega(x'|A_0) = \gamma^L(1-\gamma)^{m-K-L} = \omega'(x') \quad \text{（验证平稳分布不变性，确保隐私机制不改变数据统计特性）}$$

$$\text{最终}: \quad \pi(x_0)\delta(x'-x_0) = \pi(x') = \omega(x'|A_0) \quad \text{（边缘条件：耦合 } \pi \text{ 的边缘等于目标分布 } \omega\text{）}$$

符号：$\pi(\cdot,\cdot|\cdot,\cdot)$ = 耦合转移概率，$\omega(\cdot|\cdot)$ = 条件平稳分布，$\gamma$ = 衰减因子，$m$ = 总步数，$K$ = 常数偏移，$A_0, x_0$ = 初始条件，$\delta(\cdot)$ = Dirac delta 函数。

**对应消融**：Table 3 显示度截断对零样本关系预测的影响（见实验部分）。

---

### 模块 2: 度截断与敏感度控制（对应框架图阶段一）

**直觉**：关系图中的枢纽节点（如知识图谱中的流行实体）可能关联数千条边，直接导致实体级敏感度无界。通过在预处理阶段截断节点度数，我们将任意实体的关联记录数硬性上限设为 $K_d$，从而使敏感度从与图最大度相关变为与 $K_d$ 相关。

**Baseline 公式** (无度截断的标准处理):
$$\Delta_2 f = \max_{x \sim x'} \|f(x) - f(x')\|_2 \propto \max_{v \in V} \deg(v)$$

符号：$\Delta_2 f$ = 函数 $f$ 的 $L_2$ 敏感度，$x \sim x'$ = 相邻数据集（Hamming 距离为 1），$\deg(v)$ = 节点 $v$ 的度。

**变化点**：无度截断时，敏感度取决于图的最大度 $d_{\max}$，可能极大；Group DP-SGD 直接以此计算噪声，导致高隐私成本。本文通过预处理截断将有效度固定为 $K_d$。

**本文公式**：

$$\text{Step 1}: \quad \tilde{G} = \text{CapDegree}(G, K_d) \quad \text{（对每个节点 } v \text{，保留至多 } K_d \text{ 条关联边）}$$

$$\text{Step 2}: \quad \Delta_2^{\text{entity}} f = \max_{x \sim_{\text{entity}} x'} \|f(\tilde{G}_x) - f(\tilde{G}_{x'})\|_2 \leq 2C \cdot K_d \cdot K_s$$

$$\text{最终}: \quad \sigma_{\text{entity}} = \frac{\Delta_2^{\text{entity}} f \cdot \sqrt{2\ln(1.25/\delta)}}{\varepsilon} \quad \text{（实体级噪声校准）}$$

符号：$K_d$ = 度截断阈值（degree support），$K_s$ = 分数截断阈值（score support），$x \sim_{\text{entity}} x'$ = 实体级相邻数据集（替换单个实体及其所有关联记录），$\tilde{G}_x$ = 基于截断图的数据集。

**对应消融**：Figure 3 展示了不同 $K_d$ 和 $K_s$ 配置下 Citeseer 和 Pubmed 数据集的效用变化。

---

### 模块 3: RDP 组合边界优化（对应隐私会计）

**直觉**：实体级隐私的迭代组合需要比标准矩会计更紧的边界，因为实体在多个训练步中可能以不同概率被采样。本文基于耦合结构推导了专门的 RDP（Rényi Differential Privacy）组合公式。

**Baseline 公式** (标准 RDP 组合):
$$\varepsilon_{\text{total}}(\lambda) = T \cdot \varepsilon(\lambda) \quad \text{（线性组合，} T \text{ = 总步数）}$$

**变化点**：标准线性组合假设每步隐私损失独立同分布，但实体级采样中同一实体跨步关联导致相关性。本文利用耦合 π 的结构特性获得次线性组合。

**本文公式**：

$$\text{Step 1}: \quad \varepsilon_{\text{RDP}}^{(t)}(\lambda) = \frac{\lambda}{2\sigma^2} \cdot \left(\frac{K_d}{n_t}\right)^2 \quad \text{（单步 RDP，} n_t \text{ = 第 } t \text{ 步有效实体数）}$$

$$\text{Step 2}: \quad \varepsilon_{\text{total}}(\lambda) \leq \sqrt{2\lambda \sum_{t=1}^{T} \left(\varepsilon_{\text{RDP}}^{(t)}(\lambda)\right)^2} \quad \text{（基于耦合相关性的次线性组合，Theorem 2.5）}$$

$$\text{最终}: \quad \varepsilon_{\text{DP}} = \min_{\lambda} \left(\varepsilon_{\text{total}}(\lambda) + \frac{\ln(1/\delta)}{\lambda - 1}\right)$$

**对应验证**：Figure 2 比较了不同 RDP 组合边界（包括本文 Theorem 2.5）随迭代步数的增长趋势，显示本文边界优于标准线性组合。

## 实验与分析



本文在关系预测任务上评估了所提方法的隐私-效用权衡。主要实验设置包括标准关系预测（Table 1）和零样本关系预测消融（Table 3）。由于分析材料中 Table 1 的具体数值被截断，以下基于可获取信息进行定性总结：所提方法在实体级差分隐私约束下，通过耦合机制与度截断的配合，实现了优于 Group DP-SGD 的隐私-效用权衡——Group DP-SGD 因直接以实体组大小为敏感度乘数而引入过量噪声，而本文的耦合分析显著紧致了该边界。



**零样本关系预测的度截断消融**（Table 3 / Figure 3）：度截断是控制零样本场景下实体级敏感度的关键组件。实验对比了模型在有无度截断预处理下的性能：去掉度截断后，枢纽节点的原始高度导致实体级敏感度急剧膨胀，为达到同等隐私保证 (ε, δ) 需添加的噪声量大幅增加，进而严重损害零样本关系预测的效用。具体而言，在 Citeseer 和 Pubmed 数据集上，度截断阈值 $K_d$ 与分数截断 $K_s$ 的配置直接决定了最终预测性能；Figure 3 显示不同 $(K_d, K_s)$ 组合下的效用曲面，验证了适度截断（而非无截断或过度截断）的最优性。


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/04a0a89c-1057-49a6-9ef5-ee6928b4d425/figures/Figure_3.png)
*Figure 3 (ablation): Utility of different negative samples per positive loss, degree support K_d and score support K_s on Citeseer and Pubmed.*



**RDP 组合边界验证**（Figure 2）：左两图展示了每步 RDP 边界（Eq. 3）与不同组合策略（包括本文 Theorem 2.5）随训练步数的变化。结果表明，本文提出的基于耦合结构的组合边界增长速率显著低于标准线性组合，这意味着在相同隐私预算下可支持更多训练迭代，或直接转化为更紧的最终 (ε, δ) 保证。

**公平性检查**：
- **基线强度**：本文主要对比 DP-SGD 和 Group DP-SGD，但未与 DP-Adam、DP-RMSprop 等更强优化基线比较，也未对比专门针对关系学习的 DP 方法（如带 DP 的 R-GCN）。
- **数据集范围**：Table 2 显示实验数据集规模有限（Citeseer、Pubmed 等引文网络），缺乏大规模知识图谱验证。
- **复现性**：无公开代码仓库，无法独立验证隐私会计实现与噪声校准的正确性。
- **已知局限**：作者未明确讨论度截断引入的信息损失（被截断的边可能携带重要关系信息），以及耦合构造的计算开销。

## 方法谱系与知识库定位

**方法家族**：差分隐私随机优化（Differentially Private Stochastic Optimization）

**父方法**：DP-SGD（Abadi et al., 2016）—— 本文在其基础上进行三处核心改造：

| 改造维度 | 具体变化 |
|:---|:---|
| **目标函数** | 将逐记录梯度裁剪替换为基于耦合 π 的实体级梯度聚合 |
| **训练流程** | 引入度截断采样与耦合驱动的自适应噪声校准，替代均匀采样与固定裁剪 |
| **数据管线** | 新增度截断预处理模块，边界化关系图的实体级敏感度 |

**直接基线与差异**：
- **DP-SGD**：记录级隐私 vs. 本文实体级隐私；独立记录敏感度 vs. 耦合界定的实体敏感度
- **Group DP-SGD**：直接以组大小放大敏感度 vs. 本文通过耦合结构获得更紧的实体级边界

**后续方向**：
1. **自适应度截断**：根据实体重要性动态调整 $K_d$，而非全局固定阈值
2. **大规模图扩展**：将耦合机制扩展至工业级知识图谱（如 Wikidata），验证可扩展性
3. **与其他 DP 优化器结合**：将耦合分析融入 DP-Adam、DP-LAMB 等自适应优化框架

**标签**：
- **模态**（Modality）：关系数据 / 图结构数据
- **范式**（Paradigm）：差分隐私训练 / 隐私保护机器学习
- **场景**（Scenario）：关系预测、零样本推理、实体级隐私
- **机制**（Mechanism）：耦合理论、敏感度分析、RDP 组合优化
- **约束**（Constraint）：(ε, δ)-差分隐私、实体级隐私保证、度截断预处理

