---
title: A Hard-to-Beat Baseline for Training-free CLIP-based Adaptation
type: paper
paper_level: C
venue: ICLR
year: 2024
paper_link: null
aliases:
- CLIP免训练自适应的极简集成基线
- Alpha-CLIP (trai
- Alpha-CLIP (training-free CLIP adaptation with classifier ensemble)
acceptance: Poster
cited_by: 50
code_url: https://zhengbo.wang/ICLR24
method: Alpha-CLIP (training-free CLIP adaptation with classifier ensemble)
---

# A Hard-to-Beat Baseline for Training-free CLIP-based Adaptation

[Code](https://zhengbo.wang/ICLR24)

**Topics**: [[T__Few-Shot_Learning]], [[T__Domain_Adaptation]], [[T__OOD_Detection]] | **Method**: [[M__Alpha-CLIP]] | **Datasets**: Few-shot

| 中文题名 | CLIP免训练自适应的极简集成基线 |
| 英文题名 | A Hard-to-Beat Baseline for Training-free CLIP-based Adaptation |
| 会议/期刊 | ICLR 2024 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2402.04087) · [Code](https://zhengbo.wang/ICLR24) · [Project] |
| 主要任务 | 少样本图像分类 (Few-shot Classification)、基础到新类泛化 (Base-to-New Generalization)、长尾学习 (Imbalanced Learning) |
| 主要 baseline | Zero-Shot CLIP, CoOp, Tip-Adapter, CoCoOp, KgCoOp |

> [!abstract]
> 因为「现有CLIP自适应方法要么需要梯度训练（CoOp/CoCoOp），要么依赖复杂的cache模型（Tip-Adapter）」，作者在「Tip-Adapter」基础上改了「用KNN分类器替换attention-based cache，并仅通过搜索单个超参数alpha实现零样本与KNN预测的线性集成」，在「11个数据集少样本分类基准」上取得「ResNet-50 16-shot平均准确率76.05%，超越CoOp +2.63%」

- **关键性能**: ResNet-50 16-shot 11数据集平均准确率 76.05% vs CoOp 73.42% vs Tip-Adapter 70.32%
- **关键性能**: ViT-B/16 16-shot 平均准确率 81.85% vs CoOp 79.71% vs Tip-Adapter 76.81%
- **关键性能**: 完全免训练，无额外参数，仅搜索超参数alpha ∈ [0.0001, 100.0]

## 背景与动机

视觉-语言预训练模型CLIP展现了强大的零样本泛化能力，但在下游任务的少样本场景中，其性能仍有较大提升空间。具体而言，当每个类别仅有16张标注图像时，Zero-Shot CLIP在11个标准数据集上的平均准确率仅为58.77%，远不能满足实际应用需求。

现有方法主要从两个方向解决这一问题。**CoOp** 学习连续的prompt向量，通过梯度优化使文本编码器适应下游任务，但需要端到端训练且泛化到新类时性能下降。**Tip-Adapter** 提出免训练方案，构建key-value cache模型并通过attention机制融合视觉特征，避免了梯度计算，但引入了复杂的cache结构和额外的attention计算。**CoCoOp** 进一步扩展为实例条件化的prompt学习，同样需要训练且参数量增加。

这些方法的核心局限在于：**需要训练的方法**（CoOp/CoCoOp）存在训练开销和过拟合风险；**免训练的方法**（Tip-Adapter）虽然省去了训练，但其attention-based cache模型设计复杂，且未充分利用支持集视觉特征的直接结构信息。作者观察到，Tip-Adapter的cache机制本质上是在模拟近邻关系，却绕过了更直接的KNN途径。一个自然的问题由此产生：是否可以设计一种更简单、更直接、且性能更强的免训练方案？

本文的核心动机正是验证"极简设计"的有效性——用标准的KNN分类器替代复杂的cache-attention结构，仅通过线性插值融合零样本预测与KNN预测，并以单个超参数alpha在验证集上搜索最优平衡。这种设计不仅大幅简化了方法，还在多个基准上取得了超越复杂训练方法的效果。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/c9aedefa-4af8-4022-b052-eaa326e15430/figures/Figure_1.png)
*Figure 1 (pipeline): Overview of our training-free method*



## 核心创新

核心洞察：零样本CLIP分类器与KNN分类器的线性集成即可达到甚至超越复杂训练方法的性能，因为KNN直接在视觉特征空间捕捉了支持集的结构信息，而alpha搜索自动找到了最优的融合平衡点，从而使免训练CLIP自适应不再需要复杂的attention cache或梯度优化成为可能。

| 维度 | Baseline (Tip-Adapter) | 本文 (Alpha-CLIP) |
|:---|:---|:---|
| 自适应机制 | Key-Value Cache + Attention | K-Nearest Neighbors (k=64) |
| 融合方式 | Attention-based residual connection | 线性插值 p = α·p_knn + (1-α)·p_zs |
| 可学习参数 | Cache keys/values (不可训练但存储密集) | 仅超参数α（验证集搜索） |
| 训练需求 | 免训练，但结构复杂 | 免训练，极简结构 |
| 新类泛化 | 需重新构建cache | KNN天然支持新类 |

## 整体框架



方法整体采用双分支并行结构，输入图像同时通过两个独立的分类分支，最终通过alpha权重融合预测结果。数据流如下：

**输入**: 测试图像 x，以及支持集（每类k-shot图像）

**分支一：Zero-Shot CLIP Classifier**
- 输入：类别文本提示（如"a photo of a [CLASS]"）
- 处理：文本编码器提取文本特征，与图像特征计算余弦相似度
- 输出：零样本概率分布 p_zs ∈ ℝ^C（C为类别数）

**分支二：KNN Classifier**
- 输入：支持集图像经视觉编码器提取的特征库，以及测试图像特征
- 处理：在视觉特征空间查找k=64个最近邻，基于邻域标签投票
- 输出：KNN概率分布 p_knn ∈ ℝ^C

**Alpha Ensemble模块**
- 输入：p_zs, p_knn, 以及搜索得到的超参数α
- 处理：线性插值融合两个预测
- 输出：最终预测 p = α · p_knn + (1-α) · p_zs

**超参数搜索**
- 在验证集上对α ∈ [0.0001, 100.0]进行网格搜索，选取最优值

```
[测试图像 x] ──┬──→ [CLIP Visual Encoder] ──→ [Zero-Shot CLIP] ──→ p_zs
              │                                        ↑
              └──→ [CLIP Visual Encoder] ──→ [KNN (k=64)] ───────→ p_knn
                                                        ↓
                                              [Alpha Ensemble: p = α·p_knn + (1-α)·p_zs] ──→ 最终预测
```

## 核心模块与公式推导

### 模块 1: Zero-Shot CLIP 分类器（框架图左侧分支）

**直觉**: 利用CLIP预训练的视觉-语言对齐能力，通过文本提示直接获得类别先验，无需任何下游适应。

**Baseline 公式** (Zero-Shot CLIP):
$$p_{\text{zs}}^{(i)} = \frac{\exp(\cos(E_{\text{img}}(x), E_{\text{text}}(t_i)) / \tau)}{\sum_{j=1}^{C} \exp(\cos(E_{\text{img}}(x), E_{\text{text}}(t_j)) / \tau)}$$

符号: $E_{\text{img}}$ = CLIP图像编码器, $E_{\text{text}}$ = CLIP文本编码器, $t_i$ = 第i类的文本提示, $\tau$ = 温度系数, $\cos(\cdot,\cdot)$ = 余弦相似度

**变化点**: Zero-Shot CLIP完全依赖预训练知识，在下游分布偏移时性能受限。本文保留此分支作为稳定的先验基线，但引入KNN分支进行自适应修正。

**本文使用**: 与baseline完全一致，p_zs 作为集成的一个固定分支。

---

### 模块 2: KNN 分类器（框架图右侧分支）

**直觉**: 直接在视觉特征空间利用支持集的局部结构，通过近邻投票获得数据驱动的类别后验，无需任何参数学习。

**Baseline 公式** (Tip-Adapter attention-based cache):
$$p_{\text{tip}} = \text{Attention}(Q= E_{\text{img}}(x), K=\{k_i\}_{i=1}^{N}, V=\{v_i\}_{i=1}^{N}) + \text{CLIP}(x)$$

符号: $Q$ = 查询图像特征, $K,V$ = cache中的key-value对, $N$ = 支持集大小

**变化点**: Tip-Adapter的attention cache需要存储所有支持集特征作为key-value对，计算attention权重，结构冗余。本文发现直接用KNN即可达到类似效果：

**本文公式（推导）**:
$$\text{Step 1}: \quad \mathcal{N}_k(x) = \text{arg}\text{top}_k_{x' \in \mathcal{S}} \cos(E_{\text{img}}(x), E_{\text{img}}(x')) \quad \text{[在支持集S中找k个最近邻]}$$

$$\text{Step 2}: \quad p_{\text{knn}}^{(i)} = \frac{|\{x' \in \mathcal{N}_k(x) : y' = i\}|}{k} \quad \text{[邻域标签频率归一化为概率]}$$

$$\text{最终}: \quad p_{\text{knn}} = \text{softmax}(\log p_{\text{knn}} / \tau') \quad \text{[可选温度缩放，与CLIP输出对齐]}$$

其中k=64为固定超参数。KNN分类器完全无参数，直接利用支持集特征空间的拓扑结构。

**对应消融**: 

---

### 模块 3: Alpha 集成机制（框架图中央融合模块）

**直觉**: 零样本预测提供稳定的预训练先验，KNN预测提供数据驱动的自适应修正，线性插值以alpha为杠杆自动平衡二者。

**Baseline 公式** (Tip-Adapter ensemble):
$$p_{\text{tip-final}} = \text{Residue}(p_{\text{cache}}, p_{\text{zs}}) = p_{\text{zs}} + \lambda \cdot \text{Attention}(Q,K,V)$$

符号: $\lambda$ = 固定或学习的残差权重

**变化点**: Tip-Adapter采用加法残差融合，需要学习或调整残差权重，且cache输出与CLIP输出维度需对齐。本文改为更简洁的凸组合形式，alpha作为全局插值系数：

**本文公式（推导）**:
$$\text{Step 1}: \quad \alpha^* = \text{arg}\max_{\alpha \in [0.0001, 100.0]} \text{Acc}_{\text{val}}(\alpha \cdot p_{\text{knn}} + (1-\alpha) \cdot p_{\text{zs}}) \quad \text{[验证集网格搜索]}$$

$$\text{Step 2}: \quad p_{\text{final}} = \alpha^* \cdot p_{\text{knn}} + (1 - \alpha^*) \cdot p_{\text{zs}} \quad \text{[测试时固定alpha推理]}$$

**关键性质**: 当$\alpha^* = 0$时退化为纯Zero-Shot CLIP；当$\alpha^* = 1$时退化为纯KNN；中间值实现自适应平衡。作者发现最优alpha通常在1-10范围内，说明KNN预测需要适当放大才能与零样本预测有效互补。

**对应消融**: Table 11（附录C）显示，固定$\alpha=1$时11数据集平均准确率降至73.45%，相比最优alpha的76.05%下降-2.6%；在Flowers数据集上差距更大，$\alpha=1$仅89.32% vs 最优95.72%，差距-6.4%。

## 实验与分析


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/c9aedefa-4af8-4022-b052-eaa326e15430/figures/Table_1.png)
*Table 1 (quantitative): Results of few-shot classification on 11 datasets*



本文在少样本分类、基础到新类泛化、长尾学习三个场景下评估方法。核心结果来自11个标准数据集（包括ImageNet、Caltech101、OxfordPets、StanfordCars等）的少样本分类基准。


![Table 8](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/c9aedefa-4af8-4022-b052-eaa326e15430/figures/Table_8.png)
*Table 8 (comparison): Base-to-novel generalization*



在16-shot设置下，本文方法（Alpha-CLIP）使用ResNet-50 backbone达到76.05%的平均准确率，相比需要梯度训练的CoOp（73.42%）提升+2.63%，相比同样免训练的Tip-Adapter（70.32%）提升+5.73%。这一结果尤为显著，因为CoOp需要训练连续prompt向量，而本文完全免训练。当换用更强的ViT-B/16 backbone时，优势保持：Alpha-CLIP达81.85%，超越CoOp（79.71%）+2.14%，超越Tip-Adapter（76.81%）+5.04%。Figure 2进一步展示了随shot数增加的性能曲线趋势，本文方法在各shot设置下均保持稳定领先。


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/c9aedefa-4af8-4022-b052-eaa326e15430/figures/Figure_2.png)
*Figure 2 (result): Results of few-shot classification on the 11 datasets*



在基础到新类泛化任务中（Table 8），本文方法在ImageNet上训练后泛化到新类，与CoCoOp、KgCoOp等专门优化泛化的方法相比具有竞争力。Figure 4展示了跨数据集迁移能力：在ImageNet-1K上搜索alpha后，直接应用于另外10个数据集仍取得强劲性能，验证了方法的迁移鲁棒性。


![Table 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/c9aedefa-4af8-4022-b052-eaa326e15430/figures/Table_2.png)
*Table 2 (quantitative): Results of imbalanced learning on ImageNet-LT and Places-LT datasets*



消融实验聚焦于alpha的敏感性。Table 11显示，alpha的选取对性能影响显著：11数据集平均而言，最优alpha（约1-10范围）达76.05%，而固定$\alpha=1$降至73.45%（-2.6%），固定$\alpha=0.0001$接近纯Zero-Shot性能。特别地，在Flowers102这类细粒度数据集上，alpha的影响被放大：最优95.72% vs $\alpha=1$时89.32%，差距达-6.4%。这表明KNN预测需要适当放大（alpha>1）才能有效补偿零样本CLIP在细粒度类别上的不足。

**公平性检查**: 本文比较的训练方法（CoOp）并非当时最强，未与CoCoOp、KgCoOp在少样本主表直接对比，也未包含CLIP-Adapter、ProGrad、PromptSRC等后续方法。方法极简（单超参数）既是优势也可能被视为创新不足——标题自我定位为"Hard-to-Beat Baseline"而非突破性方法。作者坦诚KNN是标准技术，核心贡献在于验证其作为强基线的有效性。推理时KNN需要遍历支持集，大规模支持集下延迟高于纯前向传播，但Figure 5显示相比Tip-Adapter的cache机制仍有速度优势。

## 方法谱系与知识库定位

方法家族：**免训练CLIP自适应 (Training-free CLIP Adaptation)**

**父方法**: Tip-Adapter (Zhang et al., 2022) — 首个提出免训练CLIP自适应，使用key-value cache + attention机制。本文直接继承其"免训练"核心思想，但将cache-attention替换为更简洁的KNN+线性插值。

**改动插槽**:
- **inference_strategy**: Tip-Adapter的attention-based cache → KNN分类器 + alpha线性集成
- **training_recipe**: 同为免训练，但Tip-Adapter需构建cache，本文仅需搜索单超参数alpha
- **data_pipeline**: Tip-Adapter的cache存储所有支持集特征 → KNN直接使用特征空间近邻关系

**直接基线对比**:
- **Zero-Shot CLIP**: 无自适应，本文通过KNN分支+alpha集成引入数据驱动修正
- **CoOp/CoCoOp**: 需梯度训练prompt，本文完全免训练但性能超越
- **Tip-Adapter**: 同为免训练，本文更简单（无cache结构）且性能更强（+5.73%）

**后续方向**:
1. 将alpha扩展为实例自适应权重（如基于测试样本不确定性动态调整），而非全局固定值
2. 结合更先进的视觉特征增强（如DINOv2特征）替代CLIP视觉编码器，进一步提升KNN判别力
3. 探索KNN与prompt学习的联合优化，在保持低训练成本的同时获得更精细的适应

**标签**: 模态=图像+文本 / 范式=免训练自适应 / 场景=少样本分类 / 机制=集成学习+KNN / 约束=零额外参数+零梯度计算

