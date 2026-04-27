---
title: 'Knowledge Bridger: Towards Training-Free Missing Modality Completion'
type: paper
paper_level: C
venue: CVPR
year: 2025
paper_link: null
aliases:
- 免训练知识桥接缺失模态补全
- Knowledge Bridge
- Knowledge Bridger
acceptance: poster
cited_by: 9
code_url: https://github.com/Jian-Lang/awesome-modality-missing-learning
method: Knowledge Bridger
baselines:
- InternVL：视觉基础模型规_InternVL
---

# Knowledge Bridger: Towards Training-Free Missing Modality Completion

[Code](https://github.com/Jian-Lang/awesome-modality-missing-learning)

**Topics**: [[T__Retrieval]], [[T__Image_Generation]], [[T__Medical_Imaging]] | **Method**: [[M__Knowledge_Bridger]] | **Datasets**: IU X-ray, MM-IMDb

| 中文题名 | 免训练知识桥接缺失模态补全 |
| 英文题名 | Knowledge Bridger: Towards Training-Free Missing Modality Completion |
| 会议/期刊 | CVPR 2025 (poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2502.19834) · [Code](https://github.com/Jian-Lang/awesome-modality-missing-learning) · [Project](待补充) |
| 主要任务 | 缺失模态补全 (Missing Modality Completion)，多模态视觉识别 |
| 主要 baseline | MMIN, DiCMoR, MPLMM, MPMM, Baseline (remove missing) |

> [!abstract] 因为「现有缺失模态方法需要端到端训练且生成质量不稳定」，作者在「MPLMM/MPMM」基础上改了「免训练知识引导生成 + 多候选排序机制」，在「IU X-ray (η=0.7)」上取得「F1 53.6 / AP 73.9 / SS 22.6 (SOTA)」

- **关键性能**: IU X-ray (η=0.7) 上 F1 = 53.6，相比 SOTA baseline MPMM (49.9) 提升 +3.7
- **关键性能**: IU X-ray (η=0.7) 上 SS = 22.6，相比 SOTA baseline DiCMoR (18.1) 提升 +4.5，达到新 SOTA
- **关键性能**: 推理完全免训练，使用 Qwen-VL-7B + SDXL/Cheff，单样本推理约 40 秒 (NVIDIA RTX 4090)

## 背景与动机

在多模态学习中，一个核心痛点是**缺失模态问题**：当输入的部分模态数据丢失时，模型如何仍能可靠地完成下游任务？例如，在医学诊断场景中，一份胸部 X 光片可能缺少对应的放射科报告，或反之；在电影分类任务中，海报图像可能缺失文字简介。传统方法若直接丢弃缺失样本，会损失大量信息；若强行训练，则面临数据不完整导致的优化困难。

现有方法大致分为两类。**MMIN** 和 **DiCMoR** 采用基于插补 (imputation-based) 的策略，使用 VAE 或生成网络直接重建缺失模态，但这些方法需要端到端训练，且生成结果常缺乏语义一致性。**MPLMM** 和 **MPMM** 则走非插补路线，通过提示学习或共享-特异特征建模来绕过缺失模态的显式恢复，避免生成不稳定的问题，但代价是永久损失了缺失模态可能携带的互补信息。

**然而，这两类方法共享一个根本局限：它们都依赖在目标数据上的监督训练。** 这意味着每当遇到新领域或新缺失模式时，都需要重新收集数据、重新训练模型。此外，现有生成式方法通常采用单步生成 (single-shot generation)，一旦生成质量不佳便无法挽回，缺乏对生成结果的校验与筛选机制。

本文提出 **Knowledge Bridger**，核心思路是：与其训练一个专用网络来猜测缺失内容，不如借助现成的大型多模态模型 (LMM) 从可用模态中提取结构化知识，再用预训练扩散模型生成多个候选，最后通过知识图谱进行语义排序选出最优解——**全程无需任何梯度更新**。

## 核心创新

核心洞察：**结构化知识是连接可用模态与缺失模态的可靠桥梁**，因为大型多模态模型 (LMM) 能够从残存模态中提取对象数量、属性、环境等高层语义约束，从而使扩散模型的生成过程从「盲生成」转变为「知识约束下的定向搜索」成为可能。

| 维度 | Baseline (MMIN/DiCMoR/MPLMM/MPMM) | 本文 (Knowledge Bridger) |
|:---|:---|:---|
| **训练方式** | 端到端监督训练，需目标域数据 | **完全免训练**，零梯度更新 |
| **生成策略** | 单步生成或隐式绕过，无校验 | **多候选生成 (n=5)** + 知识排序筛选 |
| **知识利用** | 无显式知识建模，依赖网络隐式学习 | **LMM 显式提取结构化知识**，构建知识图谱引导生成与排序 |
| **模型组成** | 专用小网络 (VAE/MLP/CLIP) | **组合现成大模型**：Qwen2-VL + SDXL/Cheff |

## 整体框架


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/84d7b9e2-2bde-46d8-8bec-ad6231df606e/figures/Figure_1.png)
*Figure 1 (pipeline): Overview of Knowledge Bridger Pipeline. The pipeline consists of three steps: (1) construction of a knowledge graph from the available modalities and the known domain knowledge, (2) generation of the missing modalities conditioned on the input, and (3) ranking to select the optimal completed modality.*



Knowledge Bridger 采用三阶段模块化流水线，输入为含缺失模态的多模态样本，输出为补全后的完整样本用于下游分类任务。

**阶段一：知识建模 (Knowledge Modeling)**。输入为可用模态（如图像或文本），通过 **Qwen2-VL** 提取结构化知识描述，包括对象数量、属性特征、上下文环境等高层语义信息。该模块将原始感官数据转化为可解释、可比较的知识表示。

**阶段二：缺失模态生成 (Missing Modality Generation)**。输入为结构化知识描述 + 可用模态，通过预训练扩散模型生成 **n = 5** 个候选缺失模态。通用领域使用 **SDXL 1.0** 生成图像，医学领域使用 **Cheff** 生成胸部 X 光片。知识描述作为条件注入扩散模型的生成过程。

**阶段三：知识排序 (Knowledge Ranking)**。输入为 5 个候选缺失模态 + 知识图谱，通过**知识图谱结构匹配**与**语义相似度评分**双重机制，选出与提取知识最一致的候选作为最终补全结果。输出为单一最优缺失模态，与可用模态拼接后送入下游分类器。

```
可用模态 (图像/文本)
    ↓
[Qwen2-VL] 知识建模 → 结构化知识描述 (对象/属性/环境)
    ↓
[SDXL/Cheff] 条件生成 → 5 个候选缺失模态
    ↓
[知识图谱 + 语义相似度] 排序筛选 → 最优缺失模态
    ↓
完整多模态样本 → 下游分类任务
```

## 核心模块与公式推导

本文方法以模块化组合为主，未引入复杂的端到端损失函数优化，核心设计体现在知识排序的评分机制与多候选生成的条件化策略。以下分模块阐述其数学本质。

### 模块 1: 知识建模与条件生成（对应框架图 阶段一→阶段二）

**直觉**: 将 LMM 作为确定性知识提取器，把非结构化的可用模态转化为结构化的文本知识，以此「锚定」扩散生成的语义空间。

**Baseline 公式** (标准扩散条件生成, SDXL):
$$\epsilon_\theta(x_t, t, c) \approx \epsilon$$
其中 $x_t$ 为加噪潜变量，$t$ 为时间步，$c$ 为条件嵌入（通常是 CLIP 文本编码或空条件）。符号: $\theta$ = 扩散模型参数, $\epsilon$ = 真实噪声, $c$ = 条件向量。

**变化点**: 标准 SDXL 的条件 $c$ 来自简单文本提示或空条件，缺乏对可用模态中具体对象、数量、属性的精确约束，导致生成结果可能与原样本语义偏离。

**本文公式（推导）**:
$$\text{Step 1}: \mathcal{K} = \text{Qwen2-VL}(M_{avail}; \mathcal{P}_{knowledge}) \quad \text{加入结构化知识提取，将可用模态 } M_{avail} \text{ 转化为知识描述 } \mathcal{K}$$
$$\text{Step 2}: c_{enhanced} = \text{TextEncoder}_{SDXL}(\mathcal{K}) \quad \text{使用知识描述替代原始简单提示作为扩散条件}$$
$$\text{最终}: \epsilon_\theta(x_t, t, c_{enhanced}) \approx \epsilon \quad \text{生成过程被知识约束，产生 } n=5 \text{ 候选 } \{\hat{M}_{miss}^{(i)}\}_{i=1}^5$$

**对应消融**: Table 3 显示移除 Knowledge Modeling 后 F1 下降 -17.5（IU X-ray），证明知识提取是性能的核心来源。

### 模块 2: 知识排序评分（对应框架图 阶段三）

**直觉**: 扩散模型的随机性导致即使条件相同，多次生成结果质量也有波动；需要与知识图谱对齐的评分机制来「去噪」选优。

**Baseline 形式** (随机选择或单一生成):
$$\hat{M}_{miss} = \hat{M}_{miss}^{(1)} \quad \text{或} \quad \hat{M}_{miss} = \text{RandomPick}(\{\hat{M}_{miss}^{(i)}\}_{i=1}^5)$$

**变化点**: 随机选择无法保证生成质量；单一生成则失去纠错机会。本文引入双重评分：知识图谱结构一致性 + 语义嵌入相似度。

**本文公式（推导）**:
$$\text{Step 1}: \mathcal{G} = \text{BuildKG}(\mathcal{K}) \quad \text{从知识描述构建知识图谱，节点为对象/属性，边为关系}$$
$$\text{Step 2}: S_{KG}^{(i)} = \text{GraphAlign}(\hat{M}_{miss}^{(i)}, \mathcal{G}) \quad \text{候选与知识图谱的结构对齐分数}$$
$$\text{Step 3}: S_{sem}^{(i)} = \text{CosSim}(\text{Embed}(\hat{M}_{miss}^{(i)}), \text{Embed}(M_{avail})) \quad \text{候选与可用模态的语义相似度}$$
$$\text{最终}: i^* = \text{arg}\max_i \left[ \alpha \cdot S_{KG}^{(i)} + (1-\alpha) \cdot S_{sem}^{(i)} \right], \quad \hat{M}_{miss}^* = \hat{M}_{miss}^{(i^*)}$$
其中 $\alpha$ 为平衡系数（具体值文中未明确给出，设为超参数）。

**对应消融**: Table 3 显示将 Knowledge Ranking 替换为 Random Ranking 后 F1 下降 -19.3（IU X-ray）；去掉 Knowledge Graph Ranking 单独下降 -1.9，去掉 Semantic Similarity Ranking 单独下降 -2.4，证明两项评分机制具有互补性。

### 模块 3: 多候选生成的规模效应（对应框架图 阶段二参数）

**直觉**: 增加候选数量 $n$ 可提升「命中」高质量生成的概率，但需权衡推理成本。

**Baseline**: $n = 1$ 单一生成。

**本文公式**:
$$\text{最终}: n = 5 \quad \text{(默认配置，推理时间约 40s)}$$

**对应消融**: Table 3 显示 $n=1$ 相比 $n=5$ 时 F1 下降 -8.5（IU X-ray），证明多候选策略显著优于单步生成，但代价是线性增加的推理时间。

## 实验与分析


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/84d7b9e2-2bde-46d8-8bec-ad6231df606e/figures/Table_1.png)
*Table 1 (quantitative): Quantitative results (%) on COCO-2014 and MM-IMDB datasets. Bold denotes the best results and underline denotes the second best.*



本文在四个数据集上评估：通用领域的 COCO-2014、MM-IMDb，以及专业领域的 RSICD（遥感图像描述）、IU X-ray（胸部 X 光-报告对）。主要指标为分类 F1、平均精度 AP，以及语义相似度 SS（衡量生成缺失模态与真实模态的接近程度）。


![Table 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/84d7b9e2-2bde-46d8-8bec-ad6231df606e/figures/Table_2.png)
*Table 2 (quantitative): Quantitative results (%) on RSICD and X-Ray datasets. Bold denotes the best results and underline denotes the second best.*



核心结果聚焦 **IU X-ray (η=0.7)**，这是医学场景下高缺失率（70% 样本缺失某一模态）的严苛设定。本文方法取得 **F1 = 53.6**，相比最强 imputation baseline MPMM (49.9) 提升 **+3.7**，相比非插补 baseline MPLMM (49.3) 提升 **+4.3**。然而，**完整数据 baseline (57.0) 仍高于本文方法 -3.4**，说明缺失模态补全尚未达到无损恢复。在 **SS（语义相似度）指标上，本文以 22.6 达到新 SOTA**，相比 DiCMoR (18.1) 提升 **+4.5**，表明生成质量在语义层面显著优于传统方法。AP 指标上，本文 73.9 相比 MPLMM (72.7) 微升 **+1.2**，但同样低于完整 baseline (75.7)。


![Table 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/84d7b9e2-2bde-46d8-8bec-ad6231df606e/figures/Table_3.png)
*Table 3 (ablation): The impact of various components.*



消融实验（Table 3）揭示了各组件的贡献层级。**Knowledge Ranking 被替换为 Random Ranking 时损失最大**：F1 暴跌 -19.3，说明排序机制是筛选高质量候选的关键。**Knowledge Modeling 被移除时损失次之**：-17.5，证明 LMM 提取的结构化知识是生成质量的先决条件。**多候选生成从 n=5 降至 n=1**：-8.5，验证了「生成-排序」范式相对于单步生成的优势。两项评分子机制中，Knowledge Graph Ranking (-1.9) 与 Semantic Similarity Ranking (-2.4) 各自贡献相当，具有互补性。


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/84d7b9e2-2bde-46d8-8bec-ad6231df606e/figures/Figure_2.png)
*Figure 2 (ablation): The impact of different model parameters w.r.t. varying missing rates.*



Figure 2 进一步展示了模型规模与缺失率的敏感性：Qwen-VL 从 2B 扩展到 7B 再到 72B，以及 GPT-4o，知识建模质量随规模单调提升，F1 与相似度分数均呈正相关。这说明本文方法的性能天花板受限于所用 LMM 的能力。


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/84d7b9e2-2bde-46d8-8bec-ad6231df606e/figures/Figure_3.png)
*Figure 3 (qualitative): Qualitative comparison among image and text completion results.*



Figure 3 的定性对比显示，在一般领域（COCO）和医学领域（IU X-ray）中，本文方法生成的缺失模态在对象布局、属性细节上与真实模态更为接近，而直接条件生成（无知识排序）容易出现对象遗漏或语义错位。

**公平性审视**：本文的对比存在若干不均衡之处。首先，**本文使用 Qwen-VL-7B 作为 backbone，而 baselines 仅使用 CLIP + 单层 MLP**，这种 backbone 能力差距使得「SOTA」声明的含金量受限。其次，**「完整数据 baseline」(57.0 F1) 实际上优于本文方法**，但论文强调相对弱 baselines 的改进，可能误导读者对绝对性能的认知。此外，实验未纳入 GPT-4V/4o 等更强 LMM 的直接 completion baseline，也未与其他训练自由方法（如直接 prompt engineering）对比。推理成本方面，40 秒/样本的延迟在实际部署中可能成为瓶颈。作者披露的主要局限是：性能仍低于完整数据 upper bound，且依赖预训练 LMM 的质量。

## 方法谱系与知识库定位

Knowledge Bridger 属于**缺失模态补全 (Missing Modality Completion)** 方法家族，直接继承自 **MPLMM / MPMM** 的问题设定——即在多模态视觉识别中处理缺失模态——但彻底改进了求解路径。

**改变的插槽**:
- **架构 (architecture)**: 从端到端可训练网络 → 模块化大模型组合 (LMM + 扩散模型 + 排序模块)
- **训练策略 (training_recipe)**: 从监督训练 → **完全免训练**
- **推理策略 (inference_strategy)**: 从单步生成/隐式绕过 → **知识引导的多候选生成-排序**
- **数据流 (data_pipeline)**: 从直接输入 → **知识增强的显式知识提取与条件注入**

**直接 baselines 与差异**:
- **MMIN / DiCMoR** (imputation-based): 需训练专用生成网络，单步生成无校验；本文免训练且引入排序
- **MPLMM / MPMM** (non-imputation): 绕过缺失模态恢复，永久损失信息；本文显式恢复且保持语义一致性
- **Baseline (remove missing)**: 极简 CLIP+MLP，仅作参考下界；本文利用大模型能力但对比不公平

**后续方向**:
1. **动态候选数量**: 根据知识复杂度自适应调整 $n$，平衡质量与延迟
2. **端到端知识蒸馏**: 将 LMM 提取的知识压缩为轻量模块，降低推理成本
3. **跨域泛化验证**: 在更多专业领域（如视频-音频、多光谱图像）测试训练自由的极限

**标签**: 模态=图像+文本 | 范式=训练自由推理 / 大模型组合 | 场景=缺失模态补全 / 医学影像 / 视觉识别 | 机制=知识引导生成 / 多候选排序 / 知识图谱 | 约束=免训练 / 高推理延迟

## 引用网络

### 直接 baseline（本文基于）

- [[P__InternVL：视觉基础模型规_InternVL]] _(实验对比)_: Vision-language foundation model, commonly used as baseline in multimodal experi

