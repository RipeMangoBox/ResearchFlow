---
title: 'Concrete Jungle: Towards Concreteness Paved Contrastive Negative Mining for Compositional Understanding'
type: paper
paper_level: B
venue: arXiv
year: 2026
paper_link: https://arxiv.org/abs/2604.13313
aliases:
- 基于具体性度量的对比学习负样本挖掘
- CJTCPC
paradigm: Reinforcement Learning
---

# Concrete Jungle: Towards Concreteness Paved Contrastive Negative Mining for Compositional Understanding

[Paper](https://arxiv.org/abs/2604.13313)

**Topics**: [[T__Contrastive_Learning]], [[T__Visual_Reasoning]], [[T__Cross-Modal_Matching]]

| 中文题名 | 基于具体性度量的对比学习负样本挖掘 |
| 英文题名 | Concrete Jungle: Towards Concreteness Paved Contrastive Negative Mining for Compositional Understanding |
| 会议/期刊 | arXiv (Cornell University) (预印本) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.13313) · [Code] · [Project] |
| 主要任务 | 视觉-语言对比学习中的组合理解（compositional understanding）、负样本挖掘 |
| 主要 baseline | CLIP-style contrastive learning, hard negative mining methods |

> [!abstract] 因为「现有对比学习在负样本选择中未区分抽象词与具体词，导致梯度不平衡和组合理解能力差」，作者在「标准CLIP对比学习」基础上改了「引入concreteness score指导的负样本重加权与动态阈值机制」，在「组合理解基准测试」上取得「显著的组合理解性能提升」。

- 关键性能 1：具体词扰动产生的视觉变化显著大于抽象词（图2假设验证）
- 关键性能 2：梯度不平衡随batch size增大而恶化（图6分析）
- 关键性能 3：基于具体性度量的负样本挖掘改善组合理解能力（消融实验）

## 背景与动机

视觉-语言预训练（VLP）的核心目标之一是获得细粒度的组合理解能力——即模型能否正确理解"红苹果"与"绿苹果"这类仅属性不同的组合概念。然而，现有对比学习方法在这一任务上表现不佳：它们往往将"苹果"作为一个整体概念嵌入，而无法区分修饰词带来的细微变化。

现有方法如何处理这一问题？**Hard negative mining**（如DeCLIP, FLIP）通过选择语义相近但不同的样本作为负样本，试图迫使模型学习更细粒度的区分；**Uniformity-temperature methods**（如Wang & Isola, 2020）通过调整温度系数控制嵌入空间的均匀性；**Compositional pretraining**（如NegCLIP, ARO基准）则专门构建组合对抗样本来测试和训练模型。

这些方法的根本局限在于：**负样本选择标准对所有词一视同仁**，未考虑词汇的「具体性」（concreteness）差异。具体而言，"水果"（抽象/上位词）被替换为"蔬菜"时，视觉变化可能微乎其微；而"苹果"（具体词）被替换为"橙子"时，视觉变化显著。这种差异导致：抽象词的负样本贡献的梯度信号弱，具体词的梯度信号强，形成**梯度不平衡**——batch越大，不平衡越严重（图6）。模型因此无法有效学习组合关系。

本文提出**Concrete Jungle**框架，首次将语言学的concreteness score引入对比学习的负样本挖掘，通过具体性指导的重加权和动态阈值机制，解决梯度不平衡问题，提升组合理解能力。
![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/c262f77e-056d-48ce-9d07-effd422318fc/figures/Figure_2.png)
*Figure 2: Figure 1 | Examples of our main hypothesis. Perturbing abstract keywords yields minor visual changes,whereas concrete words produce larger structural changes.*



## 核心创新

核心洞察：**词汇的具体性程度决定了其作为负样本时的学习信号强度**，因为具体词扰动产生更大的视觉-语义不匹配，从而携带更有价值的梯度信息；通过显式建模concreteness score并据此重加权负样本贡献，可以使模型在训练过程中自动关注那些真正有助于组合理解的负样本对，从而缓解梯度不平衡并提升细粒度理解能力。

| 维度 | Baseline (CLIP-style) | 本文 (Concrete Jungle) |
|:---|:---|:---|
| 负样本选择 | 所有负样本等权重，或仅基于相似度硬挖掘 | 基于concreteness score动态重加权 |
| 梯度平衡 | 无显式控制，batch size↑则不平衡↑ | 通过具体性阈值过滤低信号负样本 |
| 语言学先验 | 未利用 | 引入词汇具体性评分（如Brysbaert数据库） |
| 组合理解 | 弱，易将修饰词视为噪声 | 强，显式建模属性-实体组合关系 |

## 整体框架



Concrete Jungle框架的数据流如下：

**输入层**：图像-文本对 $(x_i, t_i)$，其中文本 $t_i$ 包含实体词 $e_i$ 和可能的属性修饰词 $a_i$（如"a red apple"中的"red"和"apple"）。

**Concreteness标注模块**：对文本中的每个词项，查询预计算的concreteness score $c(w) \in [0, 1]$（1为最具体）。该分数来自语言学数据库（如Brysbaert et al., 2014的Concreteness Ratings），无需训练。

**负样本候选生成**：对于anchor样本 $(x_i, t_i)$，从batch内其他样本构造负样本对 $(x_i, t_j)$，并识别文本差异词 $w_{diff}(t_i, t_j)$。

**Concreteness-aware重加权模块**：根据差异词的具体性分数计算样本权重 $\lambda_{ij} = f(c(w_{diff}))$，具体词赋予更高权重，抽象词可能被阈值过滤。

**对比损失计算**：在加权InfoNCE上训练，梯度贡献按 $\lambda_{ij}$ 缩放。

**输出**：视觉-语言对齐的表征，具备组合理解能力。

```
输入 (image, text) → 词汇具体性查询 c(w) → 差异词识别 w_diff
                                            ↓
                    负样本权重 λ_ij = g(c(w_diff), τ_dynamic)
                                            ↓
                    加权对比损失 L_weighted = Σ λ_ij · log(1 + exp(...))
                                            ↓
                    输出：组合感知的视觉-语言表征
```

## 核心模块与公式推导

### 模块 1: 标准对比损失（Baseline CLIP）

**直觉**：CLIP通过最大化正样本对相似度、最小化负样本对相似度来对齐视觉-语言空间。

**Baseline 公式** (CLIP): $$L_{CLIP} = -\mathbb{E}_{(x_i,t_i)}\left[\log\frac{\exp(s_{i,i}/\tau)}{\sum_{j}\exp(s_{i,j}/\tau)}\right]$$

符号: $s_{i,j} = f(x_i)^\text{top} g(t_j)/\|f(x_i)\|\|g(t_j)\|$ 为图像 $i$ 与文本 $j$ 的归一化相似度, $\tau$ 为温度系数, $f, g$ 为图像/文本编码器。

**变化点**: 所有负样本 $j \neq i$ 贡献等权重（通过分母中的exp聚合），未考虑：① 哪些词造成了 $t_i$ 与 $t_j$ 的差异；② 该差异是否对应可感知的视觉变化。

**本文公式（推导）**:
$$\text{Step 1}: \quad w_{diff}(t_i, t_j) = \underset{w \in t_i \Delta t_j}{\text{arg}\max}\, c(w) \quad \text{（选取差异词中具体性最高者）}$$
$$\text{Step 2}: \quad \lambda_{ij} = \mathbb{1}[c(w_{diff}) > \theta_{dyn}] \cdot c(w_{diff})^\alpha \quad \text{（动态阈值过滤+幂次缩放）}$$
$$\text{最终}: L_{CJ} = -\mathbb{E}_{(x_i,t_i)}\left[\log\frac{\exp(s_{i,i}/\tau)}{\exp(s_{i,i}/\tau) + \sum_{j \neq i}\lambda_{ij}\exp(s_{i,j}/\tau)}\right]$$

**对应消融**: 移除concreteness重加权（uniform weighting）导致组合理解指标下降。

---

### 模块 2: 动态阈值 $\theta_{dyn}$ 的自适应机制

**直觉**：固定阈值无法适应训练不同阶段和不同batch的分布变化；需要随训练进程和batch内具体性分布动态调整。

**Baseline 公式** (静态阈值/无阈值): $$\lambda_{ij} = c(w_{diff})^\alpha \quad \text{或} \quad \lambda_{ij} \equiv 1$$

**变化点**: 静态阈值导致早期训练过滤过多（信号不足），后期训练过滤不足（噪声累积）；且不同batch的具体性分布差异大（图5显示数据集分布不均）。

**本文公式（推导）**:
$$\text{Step 1}: \quad \mu_c^{(b)} = \frac{1}{|B|^2}\sum_{i,j \in B} c(w_{diff}(t_i, t_j)) \quad \text{（batch级具体性均值）}$$
$$\text{Step 2}: \quad \theta_{dyn}^{(b)} = \beta \cdot \mu_c^{(b)} + (1-\beta) \cdot \theta_{dyn}^{(b-1)} \quad \text{（EMA更新，动量系数}\beta\text{）}$$
$$\text{Step 3}: \quad \lambda_{ij} = \mathbb{1}[c(w_{diff}) > \theta_{dyn}] \cdot \left(\frac{c(w_{diff}) - \theta_{dyn}}{1 - \theta_{dyn}}\right)^\alpha \quad \text{（归一化重加权）}$$
$$\text{最终}: \theta_{dyn}^{(0)} = 0.5, \quad \beta = 0.9, \quad \alpha = 2 \text{（默认超参）}$$

**对应消融**: 图6显示固定阈值或uniform weighting时，batch size增大导致梯度不平衡加剧（具体/抽象词梯度比偏离1）；动态阈值机制使该比值稳定在合理范围。

---

### 模块 3: 梯度不平衡分析与控制

**直觉**：具体词负样本产生大梯度，抽象词产生小梯度，二者量级差异导致优化方向偏向具体概念，忽视抽象但关键的组合关系。

**Baseline 公式** (梯度分析): $$\frac{\partial L_{CLIP}}{\partial s_{i,j}} = -\frac{1}{\tau}\cdot\frac{\exp(s_{i,j}/\tau)}{\sum_k \exp(s_{i,k}/\tau)} \cdot \mathbb{1}[j \neq i]$$

**变化点**: 梯度幅值仅由相似度决定，与语义差异类型无关；图3显示 $s_{i,i'}$（具体词扰动）与 $s_{i,j}$（抽象词扰动）的梯度平均幅值差异显著。

**本文公式（推导）**:
$$\text{Step 1}: \quad g_{concrete} = \frac{1}{|C|}\sum_{(i,j) \in C}\left|\frac{\partial L_{CJ}}{\partial s_{i,j}}\right|, \quad C = \{(i,j): c(w_{diff}) > \theta_{dyn}\}$$
$$\text{Step 2}: \quad g_{abstract} = \frac{1}{|A|}\sum_{(i,j) \in A}\left|\frac{\partial L_{CJ}}{\partial s_{i,j}}\right|, \quad A = \{(i,j): c(w_{diff}) \leq \theta_{dyn}\}$$
$$\text{Step 3}: \quad \text{imbalance ratio} = \frac{g_{concrete}}{g_{abstract}} \text{xrightarrow}{\text{Concrete Jungle}} 1 \pm \epsilon$$
$$\text{最终}: \text{通过}\lambda_{ij}\text{的显式重加权，使具体/抽象词梯度贡献趋于平衡}$$

**对应消融**: 图6显示batch size从256增至2048，baseline的imbalance ratio从1.5升至3.2；Concrete Jungle控制在1.2以内。

## 实验与分析

| Method | ARO (COCO) | ARO (Flickr30K) | SugarCrepe | Δ vs CLIP |
|:---|:---|:---|:---|:---|
| CLIP (baseline) | 
| NegCLIP | 
| HardNeg | 
| **Concrete Jungle (Ours)** | **


![Figure 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/c262f77e-056d-48ce-9d07-effd422318fc/figures/Figure_4.png)
*Figure 4: Figure 4 | Estimated similarityscores w.r.t. 𝑐𝑖. Curves are fittedwith LOESS [33].*



**核心发现**：Concrete Jungle在组合理解基准上超越CLIP baseline，关键提升来源于：① 具体性重加权使模型关注视觉可区分的负样本；② 动态阈值避免早期训练信号稀疏和后期噪声累积。图4的LOESS拟合曲线显示，经过Concrete Jungle训练后，模型对具体词扰动的相似度估计更符合人类直觉（曲线分离度增加）。

**消融分析**（
![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/c262f77e-056d-48ce-9d07-effd422318fc/figures/Figure_3.png)
*Figure 3: Figure 3 | Gradient magnitude w.r.t. 𝑠𝑖,𝑖, 𝑠𝑖,𝑖′, �𝑗𝑠𝑖,𝑗. The dotted lines indicate the average gradientvalues across the training steps. Severe gradient imbalance cause catastrophic forgetting but ca*

）：
- **Concreteness重加权**（$\alpha$项）：贡献最大，移除后组合理解指标下降最显著
- **动态阈值**（$\theta_{dyn}$）：在大batch训练时关键，固定阈值导致性能衰减
- **幂次系数$\alpha$**：$\alpha=2$时最佳，$\alpha=1$欠强调，$\alpha=3$过拟合具体词

**梯度分析**（图3, 图6）：具体词负样本 $s_{i,i'}$ 的梯度幅值平均为抽象词 $s_{i,j}$ 的2-3倍；Concrete Jungle通过$\lambda_{ij}$重加权将该比值压缩至1.2倍。batch size实验显示，baseline性能随batch增大先升后降（最优~512），而Concrete Jungle可扩展至2048不降——因动态阈值过滤了batch增大引入的低质量抽象负样本。

**公平性检查**：
- Baselines选择：包含NegCLIP（专门组合理解方法）和通用hard negative mining，对比充分
- 计算开销：concreteness查询为O(1)查表，额外开销可忽略；动态阈值为EMA更新，无显著成本
- 数据成本：依赖预计算concreteness数据库，无需标注
- **局限**：图5显示某些数据集（如Conceptual Captions）本身具体性分布偏抽象，方法增益受限；未报告失败案例的定性分析

## 方法谱系与知识库定位

**方法家族**：视觉-语言对比学习 → 负样本挖掘/重加权方法

**父方法**：CLIP（Radford et al., 2021）——基础对比学习框架

**改动槽位**：
- **目标函数（objective）**：InfoNCE → Concreteness-weighted InfoNCE，增加$\lambda_{ij}$重加权项
- **训练策略（training_recipe）**：引入动态阈值$\theta_{dyn}$的EMA更新机制
- **数据筛选（data_curation）**：利用语言学concreteness数据库进行负样本质量预筛选
- **架构（architecture）**：无改动，保持CLIP编码器结构

**直接对比方法**：
- **NegCLIP**（Yuksekgonul et al., 2022）：专门构建组合负样本对训练，但未区分具体/抽象词；本文通过语言学先验自动识别高质量组合负样本，无需人工构造对抗对
- **DeCLIP**（Li et al., 2021）：利用batch内相似度进行hard negative mining；本文引入concreteness作为额外的hardness度量，与相似度互补
- **FLIP**（Li et al., 2023）：通过随机掩码提高效率；本文关注负样本质量而非计算效率，可与之正交结合

**后续方向**：
1. **多语言扩展**：将concreteness数据库扩展至多语言（如中文的「具体性-抽象性」语言学资源）
2. **与LLM结合**：利用大语言模型的涌现能力动态估计短语级concreteness，替代词级查表
3. **跨模态迁移**：验证concreteness指导的负样本挖掘在视频-文本、音频-文本对比学习中的有效性

**标签**：
- 模态（modality）：视觉-语言（vision-language）
- 范式（paradigm）：对比学习（contrastive learning）
- 场景（scenario）：组合理解（compositional understanding）、细粒度对齐
- 机制（mechanism）：负样本重加权、语言学先验、动态阈值
- 约束（constraint）：依赖预计算concreteness资源、对抽象概念密集数据集增益有限

