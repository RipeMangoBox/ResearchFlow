---
title: Vision-and-Language Training Helps Deploy Taxonomic Knowledge but Does Not Fundamentally Alter It
type: paper
paper_level: C
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- VL训练助部署分类知识但不改变其本质
- TaxonomiGQA
- VL training does not substantially
acceptance: Poster
cited_by: 6
code_url: https://taxonomigqa.github.io
method: TaxonomiGQA
modalities:
- Text
- Image
paradigm: supervised
---

# Vision-and-Language Training Helps Deploy Taxonomic Knowledge but Does Not Fundamentally Alter It

[Code](https://taxonomigqa.github.io)

**Topics**: [[T__Visual_Question_Answering]], [[T__Reasoning]] | **Method**: [[M__TaxonomiGQA]] | **Datasets**: TaxonomiGQA

> [!tip] 核心洞察
> VL training does not substantially alter the underlying taxonomic knowledge of language models, but improves their ability to deploy this knowledge in task contexts, with performance gains linked to visual similarity and cohesion of taxonomic categories.

| 中文题名 | VL训练助部署分类知识但不改变其本质 |
| 英文题名 | Vision-and-Language Training Helps Deploy Taxonomic Knowledge but Does Not Fundamentally Alter It |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2507.13328) · [Code](https://taxonomigqa.github.io) · [Project](https://taxonomigqa.github.io) |
| 主要任务 | Visual Question Answering, Taxonomic Reasoning |
| 主要 baseline | GQA, Qwen2.5-I (text-only LM), Qwen2.5-VL-I (vision-language model) |

> [!abstract] 因为「vision-language training 是否从根本上改变语言模型的分类知识表征」这一问题尚不明确，作者在「GQA」基础上构建了「TaxonomiGQA」这一纯文本分类推理基准，通过「最小对比较（minimal pair comparison）」设计，在「TaxonomiGQA」上发现「VLM 相对 LM 的条件准确率优势可由视觉相似度显著预测（b=0.52, p<.01），但两者的底层分类知识表征无显著差异」。

- **关键性能 1**: VLM（Qwen2.5-VL-I）的条件准确率受视觉相似度显著预测，固定效应斜率 b=0.52（SE=0.19, p<.01），而对应文本 LM（Qwen2.5-I）的预测不显著（b=0.23, SE=0.17, p=0.18）
- **关键性能 2**: 表示相似性分析（RSA）显示 VLM 与 LM 的转换后 unembedding 矩阵在分类层次结构上无显著差异
- **关键性能 3**: 视觉凝聚度（visual cohesion）与相似度效应大小呈正相关，高凝聚度类别（如 band）中 VL 训练增益更大

## 背景与动机

Vision-language models（VLMs）在多模态任务上表现优异，但一个根本问题悬而未决：视觉-语言训练是**从根本上重塑**了语言模型内部的分类知识（taxonomic knowledge），还是仅仅**帮助模型更好地调用**已有的语言知识？例如，当 VLM 正确回答「斑马是哺乳动物」时，这一能力是来自 VL 训练对「斑马-哺乳动物」概念边界的重新编码，还是来自预训练语言模型已有的词汇语义知识，只是 VL 训练让模型更擅长在任务中「激活」这些知识？

现有研究从不同侧面触及这一问题，但均未直接回答。**GQA** 作为经典的视觉问答数据集，提供了丰富的场景图结构和组合式问题，但其评估需要图像输入，无法分离视觉与语言成分的独立贡献。**Emergent visual-semantic hierarchies in image-text representations** [2] 发现图像-文本表征中涌现出类别的层次结构，但未对比 VL 训练前后同一语言模型的变化。**Analyzing BERT's knowledge of hypernymy via prompting** [14] 专注于纯语言模型的上位词知识，缺乏对多模态训练的考察。**Language Guided Visual Question Answering** [11] 和 **Modality-aware integration with large language models for knowledge-based visual question answering** [9] 通过外部知识增强 VQA 性能，但关注的是知识注入而非 VL 训练对内部表征的本体性影响。

这些工作的共同局限在于：**没有控制语言输入恒定，直接对比同一模型族在 VL 训练前后的分类知识**。因此，无法区分「表征改变」与「部署改善」两种假说。本文构建 TaxonomiGQA，以纯文本形式呈现分类问题，通过最小对比较隔离 VL 训练效应，直接检验这一核心问题。

## 核心创新

核心洞察：VL 训练创造的是**任务执行层面的「部署优势」而非表征层面的「知识重构」**，因为视觉相似度可以系统性地预测 VLM 何时优于 LM，而两者的底层词汇-概念映射结构保持不变，从而使「通过行为指标反推内部知识变化」的传统范式受到根本挑战。

| 维度 | Baseline (GQA / 标准 VQA) | 本文 (TaxonomiGQA) |
|:---|:---|:---|
| 输入模态 | 图像 + 文本问题 | **纯文本问题**（无图像输入） |
| 评估对象 | 模型整体 VQA 准确率 | **VLM-LM 条件准确率差**（conditional accuracy） |
| 对比设计 | 跨模型或跨条件比较 | **最小对比较**：同一模型族的 VLM 与 LM 在**完全相同**的文本问题上作答 |
| 解释变量 | 无系统性视觉因素分析 | **视觉相似度**（viz_sim）+ **视觉凝聚度**（cohesion）作为预测变量 |
| 统计方法 | 标准准确率 / F1 | **线性混合效应模型**（hypernym-level random effects）|

## 整体框架


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/27ab5b6a-f3ec-4985-9b23-387f45125375/figures/Figure_1.png)
*Figure 1 (pipeline): The three-step pipeline to create TaxonomyQA*



TaxonomiGQA 的整体框架包含五个串行模块，从 GQA 场景图出发，最终输出统计推断结果：

1. **Taxonomy Construction（分类体系构建）**：输入 GQA 场景图结构，提取层次化的上位词-下位词关系，覆盖 9 种问题类型（如「Is X a Y?」「What type of Y is X?」等）。输出为结构化的分类关系集合。

2. **Synthetic Question Generation（合成问题生成）**：输入分类关系，自动生成纯文本格式的问答对，保留 GQA 的组合式推理特点但完全移除图像依赖。输出为 TaxonomiGQA 基准数据集。

3. **Minimal Pair Evaluation（最小对评估）**：输入完全相同的文本问题，并行送入 Qwen2.5-I（文本 LM）和 Qwen2.5-VL-I（VLM）。输出为两模型在各问题上的准确率。

4. **Visual Similarity Computation（视觉相似度计算）**：输入 THINGS 数据集中的自然物体图像，通过 Qwen2.5-VL-I 的视觉编码器提取图像表征，计算下位词与上位词成员间的余弦相似度。输出为每对概念的 viz_sim 分数。

5. **Linear Mixed-Effects Modeling（线性混合效应建模）**：输入条件准确率（cond_acc）、视觉相似度（viz_sim）及上位词分组（hypernym），拟合 `cond_acc ~ viz_sim + (1 + viz_sim | hypernym)`。输出为固定效应估计和上位词特异性随机斜率。

```
GQA Scene Graph → [Taxonomy Construction] → Hierarchical Relations
                                               ↓
THINGS Images → [Visual Similarity Computation] → viz_sim scores
                                               ↓
[Synthetic Question Generation] → Text-only QA Pairs
                                               ↓
[Minimal Pair Evaluation] → LM Accuracy + VLM Accuracy
                                               ↓
                                    Conditional Accuracy
                                               ↓
[Linear Mixed-Effects Modeling] → Statistical Inference
```

## 核心模块与公式推导

### 模块 1: 视觉相似度计算（对应框架图第 4 步）

**直觉**: 若 VL 训练增益源于视觉经验对概念边界的塑造，则下位词与其上位词成员间的视觉相似度应能预测 VLM 的相对优势；使用独立数据集 THINGS 可避免 GQA 图像分布带来的混淆。

**Baseline 公式** (标准图像表征提取): 
$$\mathbf{v}_i = \text{Encoder}_{\text{vision}}(I_i), \quad I_i \sim \text{Dataset}$$
符号: $\mathbf{v}_i$ = 图像 $I_i$ 的视觉编码器输出表征，通常取自预训练 CNN 或 Transformer 的 global pooling 层。

**变化点**: 标准 VQA 使用 GQA 自身图像，但 GQA 图像与问题构造存在系统性关联；本文改用**独立数据集 THINGS** 的自然物体图像，且明确从**目标 VLM 自身的视觉编码器**（Qwen2.5-VL-I）提取表征，确保相似度估计反映模型实际「所见」而非研究者预设的视觉特征。

**本文公式（推导）**:
$$\text{Step 1}: \mathbf{v}_i = \text{Encoder}_{\text{vision}}^{\text{Qwen2.5-VL-I}}(I_i), \quad I_i \sim \text{THINGS} \quad \text{（从目标 VLM 编码器提取，非独立视觉模型）}$$
$$\text{Step 2}: \text{viz\_sim}(o_i, o_j) = \frac{\mathbf{v}_i \cdot \mathbf{v}_j}{\|\mathbf{v}_i\| \|\mathbf{v}_j\|} \quad \text{（叶节点对象与上位词内其他成员的余弦相似度）}$$
$$\text{最终}: \overline{\text{viz\_sim}}(o_i, \text{hypernym}_k) = \frac{1}{|\text{hypernym}_k| - 1} \sum_{o_j \in \text{hypernym}_k \text{setminus} \{o_i\}} \text{viz\_sim}(o_i, o_j)$$

**对应消融**: 文中未直接报告移除独立 THINGS 数据源的消融，但强调使用 GQA 自身图像会导致相似度估计与问题构造混淆（Appendix 讨论）。

---

### 模块 2: 条件准确率与线性混合效应模型（对应框架图第 5 步）

**直觉**: 简单的 VLM 准确率比较无法区分「知识增长」与「部署改善」；条件准确率（VLM 减 LM 的准确率差）隔离了 VL 训练的净效应，而混合效应模型捕捉不同上位词类别的异质性。

**Baseline 公式** (标准逻辑回归):
$$\text{logit}(P(\text{correct}=1)) = \beta_0 + \beta_1 \cdot \text{viz\_sim} + \epsilon$$
符号: $\beta_0$ = 截距，$\beta_1$ = 视觉相似度的固定效应，$\epsilon$ = 独立同分布误差。

**变化点**: 标准逻辑回归假设观测独立且效应同质，但同一上位词（如 animal）下的多个下位词（dog, cat, horse）存在聚类相关性，且不同上位词的相似度-准确率关系斜率可能不同（如 furniture 可能斜率平缓，band 可能陡峭）。忽略此结构会导致标准误估计偏误和错误推断。

**本文公式（推导）**:
$$\text{Step 1}: \text{cond\_acc}_i = \text{Accuracy}_{\text{VLM},i} - \text{Accuracy}_{\text{LM},i} \quad \text{（定义条件准确率为 VLM-LM 差距）}$$
$$\text{Step 2}: \text{cond\_acc}_{ij} = (\gamma_{00} + u_{0j}) + (\gamma_{10} + u_{1j}) \cdot \text{viz\_sim}_{ij} + \epsilon_{ij} \quad \text{（加入上位词层级随机截距和随机斜率）}$$
$$\text{其中}: \begin{pmatrix} u_{0j} \\ u_{1j} \end{pmatrix} \sim \mathcal{N}\left(\mathbf{0}, \begin{pmatrix} \tau_{00}^2 & \tau_{01} \\ \tau_{01} & \tau_{10}^2 \end{pmatrix}\right) \quad \text{（随机效应的协方差结构）}$$
$$\text{最终}: \text{cond\_acc} \sim \text{viz\_sim} + (1 + \text{viz\_sim} \text{mid} \text{hypernym})$$

符号: $\gamma_{00}$ = 平均截距（固定效应），$\gamma_{10}$ = 平均斜率（核心关注：viz_sim 对 cond_acc 的预测力），$u_{0j}, u_{1j}$ = 第 $j$ 个上位词的随机截距和随机斜率，$\tau$ = 随机效应方差-协方差参数。

**对应消融**: 文中报告固定效应估计 VLM: b=0.52 (SE=0.19, p<.01)，LM: b=0.23 (SE=0.17, p=0.18)。若将模型简化为普通最小二乘（忽略随机斜率），标准误估计将过于乐观，可能错误宣称 LM 也存在显著效应。

## 实验与分析


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/27ab5b6a-f3ec-4985-9b23-387f45125375/figures/Table_1.png)
*Table 1 (quantitative): Left: RSA comparison of inter-model (cross-model) representational similarity for the target concepts in our taxonomy. Right: Rank Embeddings*




![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/27ab5b6a-f3ec-4985-9b23-387f45125375/figures/Figure_3.png)
*Figure 3 (result): Counterfactual representational analysis of Qwen2.5-VL and Qwen2.5-VL-Instruct*



本文在自建的 TaxonomiGQA 基准上评估 Qwen2.5-VL-I 与 Qwen2.5-I 的最小对表现。核心发现是：**VLM 在纯文本分类问题上的准确率系统性地高于其文本 LM  counterpart，但这一优势完全可由视觉相似度解释，而非底层知识表征的改变**。

具体而言，线性混合效应模型显示，视觉相似度对 VLM 条件准确率的固定效应斜率为 **b=0.52（SE=0.19, p<.01）**，而对 LM 的对应预测不显著（b=0.23, SE=0.17, p=0.18）。这意味着：当「斑马」与「哺乳动物」类别中其他成员（狮子、鲸鱼等）在视觉编码空间中更相似时，VLM 相比 LM 的优势更大；而对于视觉分散的类别（如「动物」包含昆虫、鱼类、鸟类），VL 训练几乎不提供额外增益。这一模式直接支持「部署假说」而非「重构假说」——若 VL 训练真正改变了分类知识的内部结构，其效应不应如此紧密地绑定于视觉相似度这一外部变量。



表示层面，RSA（Representational Similarity Analysis）比较了 VLM 与 LM 的转换后 unembedding 矩阵（Table 1）。结果显示，两者在分类层次结构上的表征相似性无显著差异，进一步证实 VL 训练未重塑词汇-概念映射的拓扑结构。Figure 3 的反事实表征分析（counterfactual representational analysis）显示，即使对 Qwen2.5-VL 进行指令微调得到 Qwen2.5-VL-Instruct，其内部表征相对于基础模型的变化也局限于任务适配层面。


![Figure 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/27ab5b6a-f3ec-4985-9b23-387f45125375/figures/Figure_4.png)
*Figure 4 (result): Hypothesis-specific conflicts of image similarity on predicting VLM accuracy on TaxonomyQA*



Figure 4 展示了各上位词类别的随机斜率与视觉凝聚度的关系。高凝聚度类别（如 band，其成员 guitar/drum/piano 在视觉上高度一致）呈现陡峭的正相似度效应；低凝聚度类别（如 animal）则斜率趋零。这一异质性模式是混合效应模型相比标准回归的关键增值——若仅报告平均效应，将掩盖「VL 训练帮助有限」的重要类别。

**公平性检验**：本文的比较在最小对设计内是公平的（同一架构、相同参数规模、相同文本输入），但存在明显局限。首先，**仅使用单一模型对**（Qwen2.5-VL-I vs Qwen2.5-I），未覆盖 GPT-4V、Claude、Gemini 等其他 VLM 家族，也未测试 Llama 3 等不同 LM 基座。其次，视觉相似度估计依赖于 Qwen2.5-VL-I 自身的视觉编码器，使用其他编码器（如 CLIP、DINOv2）可能得到不同模式。第三，TaxonomiGQA 为合成数据集，其问题构造的系统性模式（如固定的句法模板）可能无法完全推广到自然语言中的分类推理。作者明确承认这些局限，并将跨模型族验证列为未来工作。

## 方法谱系与知识库定位

TaxonomiGQA 属于 **VQA 基准构造** 方法族，直接父方法为 **GQA**（Hudson & Manning, 2019）。核心改动 slots 为：

- **data_curation**: 从 GQA 的场景图结构合成纯文本分类问题，剥离图像输入但保留组合式推理结构
- **evaluation_strategy**: 用条件准确率（conditional accuracy）替代标准 VQA 准确率，并引入视觉相似度作为系统性解释变量
- **inference_strategy**: 新增最小对比较（minimal pair comparison），强制 VLM 与 LM 在相同文本输入下作答以隔离 VL 训练效应

**直接 baselines 与差异**：
- **GQA**: 原始多模态 VQA 基准；TaxonomiGQA 是其纯文本、分类聚焦的变体，用于控制模态混淆
- **Qwen2.5-I / Qwen2.5-VL-I**: 评估用的模型对；本文贡献非模型本身，而是揭示两者关系的诊断框架
- **THINGS** [15]: 独立图像数据库；本文借用于视觉相似度计算，原用途为认知心理学中的概念表征研究

**后续方向**：（1）扩展至多个 VLM-LM 模型族（GPT-4V、Gemini、Molmo 等）验证结论的普适性；（2）从分类知识推广至其他知识类型（因果知识、空间知识、社会知识），检验「部署假说」的边界；（3）开发干预方法——若 VL 训练仅帮助部署，则可通过视觉相似度预测「何时需要额外训练」或「如何设计更有效的跨模态迁移策略」。

**标签**: 模态(text+image) / 范式(诊断性基准) / 场景(概念推理与知识评估) / 机制(最小对比较+混合效应建模) / 约束(合成数据、单模型对、视觉编码器依赖)

