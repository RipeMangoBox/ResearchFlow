---
title: Dual-View Training for Instruction-Following Information Retrieval
type: paper
paper_level: B
venue: Unknown
year: 2026
paper_link: https://arxiv.org/abs/2604.18845
aliases:
- 指令遵循检索的双视角训练框架
- DTIFIR
---

# Dual-View Training for Instruction-Following Information Retrieval

[Paper](https://arxiv.org/abs/2604.18845)

**Topics**: [[T__Retrieval]], [[T__Classification]], [[T__Benchmark_-_Evaluation]]

| 中文题名 | 指令遵循检索的双视角训练框架 |
| 英文题名 | Dual-View Training for Instruction-Following Information Retrieval |
| 会议/期刊 | arXiv 2026 (preprint) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.18845) · [Code](待补充) · [Project](待补充) |
| 主要任务 | Instruction-Following Information Retrieval (指令遵循信息检索) |
| 主要 baseline | BGE-M3, RetroMAE, GTE, E5-mistral, BGE-large-en-v1.5, Contriever |

> [!abstract] 因为「现有检索模型难以遵循复杂指令（如排除特定来源、限制时间范围、否定性条件），且缺乏大规模指令遵循训练数据」，作者在「标准对比学习框架（in-batch negatives + hard negatives）」基础上改了「引入双视角训练策略：通过反转文档相关性极性合成新指令，构建正负样本对进行联合训练」，在「MIRAGE、LoTTE、BEIR 等指令遵循检索 benchmark」上取得「BGE-M3 在 MIRAGE 上提升 5.3 nDCG@10，RetroMAE 提升 4.7 nDCG@10」

- **关键性能 1**: BGE-M3 + Dual-View 在 MIRAGE benchmark 上 nDCG@10 从 32.1 提升至 37.4（+5.3），超越 E5-mistral (35.8)
- **关键性能 2**: RetroMAE + Dual-View 在 MIRAGE 上 nDCG@10 从 28.4 提升至 33.1（+4.7），轻量模型接近大模型性能
- **关键性能 3**: 在 LoTTE 测试集上，Dual-View 训练使 BGE-M3 的 Recall@100 从 71.2 提升至 76.5（+5.3）

## 背景与动机

现代信息检索系统面临的核心挑战是：用户查询不再仅仅是关键词匹配，而是包含复杂约束的自然语言指令。例如，用户可能要求"查找关于气候变化的研究，但排除 IPCC 报告"，或"找到 2020 年前发表的、不支持远程工作生产力的论文"。这类**指令遵循检索**（Instruction-Following IR）要求模型理解并执行指令中的逻辑约束——包括否定、时间限制、来源排除等。

现有方法如何处理这一问题？**稠密检索器**（如 Contriever、GTE）通过对比学习将查询和文档映射到同一向量空间，但训练时仅使用简单查询-文档对，缺乏指令理解能力。**指令微调模型**（如 E5-mistral）利用大语言模型生成合成数据，但主要关注查询改写而非复杂约束遵循。**BGE-M3** 虽支持多粒度检索，其训练数据中的指令仍以相关性判断为主，缺少对"反向约束"（即明确要求不相关）的显式建模。

这些方法的共同短板在于：**训练数据缺乏对相关性极性的显式控制**。具体而言，现有数据只告诉模型"这些文档相关"，却很少明确训练"这些文档因违反某指令而不相关"。当遇到否定指令（如"不要包含..."）时，模型因缺乏此类训练信号而频繁失效。此外，构建大规模带复杂指令标注的数据成本极高，限制了监督学习的可行性。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/1bf5cc71-beab-4fcd-9151-ef0085143bae/figures/Figure_1.png)
*Figure 1 (motivation): Figure 1: We synthesize new instructions that reverse the relevance polarity of existing documents, creating challenging samples that sharpen the retriever's sensitivity to instructional nuances.*



本文的核心动机正是解决这一数据瓶颈：作者提出通过**自动反转现有标注的相关性极性**来合成高质量训练样本，无需人工标注即可让模型学习"什么是不相关"的细粒度推理。

## 核心创新

核心洞察：**文档相关性的二元判断可以双向利用**——同一文档对既可作为某指令的正例，也可通过极性反转成为另一合成指令的负例，因为现有检索数据中的相关性标签蕴含了可逆的语义约束，从而使低成本大规模合成指令遵循训练数据成为可能。

| 维度 | Baseline（标准对比学习） | 本文（Dual-View Training） |
|:---|:---|:---|
| 训练样本来源 | 人工标注或 LLM 生成的查询-文档对 | 基于现有标注自动反转极性合成新指令 |
| 负样本策略 | In-batch negatives / BM25 hard negatives | **显式构造语义对立指令**作为负样本 |
| 学习目标 | 拉近正例、推开随机负例 | **双向对齐**：正例指令→相关文档，反义指令→原正例变为负例 |
| 数据成本 | 需 LLM 生成或人工标注复杂指令 | **零额外标注**，从现有数据挖掘 |
| 指令覆盖 | 主要覆盖肯定性相关判断 | **显式覆盖否定、排除、限制类指令** |

与基于 LLM 合成数据的方法（如 E5、InPars）相比，Dual-View 不依赖大模型生成能力，而是利用标注结构的对称性；与基于规则的数据增强相比，它能保持语义连贯性而非简单替换关键词。

## 整体框架



Dual-View Training 框架包含三个核心阶段，形成从数据合成到联合训练的完整流水线：

**阶段一：相关性极性反转（Polarity Reversal）**
- 输入：现有检索数据集中的查询-文档对 $(q, d)$ 及其相关性标签 $r \in \{0,1\}$
- 处理：对正例对 $(q, d, r=1)$，自动生成"反义指令" $q^{\text{neg}}$，使原相关文档 $d$ 在新指令下变为明确不相关
- 输出：合成负样本三元组 $(q^{\text{neg}}, d, r=0)$，保留原正样本 $(q, d, r=1)$

**阶段二：双视角对比学习（Dual-View Contrastive Learning）**
- 输入：原始正样本 + 合成负样本构成的 batch
- 处理：联合编码原始指令和反义指令，在共享向量空间中执行对比学习
- 输出：更新后的 encoder 参数，使模型能区分语义对立但表面相似的指令

**阶段三：指令遵循检索推断**
- 输入：用户复杂指令（含否定、限制等约束）
- 处理：编码指令为查询向量，与文档库进行相似度排序
- 输出：满足指令约束的排序文档列表

数据流示意：
```
原始数据: (q, d+) ──→ 极性反转模块 ──→ (q^neg, d+, r=0) 合成负样本
                │                              │
                └────────→ [双视角对比学习] ←───┘
                              ↓
                    联合损失 L_dual = L_pos + L_neg
                              ↓
                        指令遵循检索模型
```

关键设计：反义指令生成不是简单的关键词否定（如加 NOT），而是基于文档内容的语义级反转，确保合成样本的合理性和难度。

## 核心模块与公式推导

### 模块 1: 相关性极性反转（Polarity Reversal Synthesis）

**直觉**: 若文档 $d$ 因满足属性集合 $A$ 而与查询 $q$ 相关，则构造要求"不满足 $A$ 中关键属性"的反义指令，可使 $d$ 成为明确负例。

**Baseline（标准数据增强）**: 无显式公式，通常采用随机负采样或 BM25 挖掘 hard negatives：
$$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(q, d^+)/\tau)}{\exp(\text{sim}(q, d^+)/\tau) + \sum_{d^- \in \mathcal{N}} \exp(\text{sim}(q, d^-)/\tau)}$$
符号: $q$ = 查询编码, $d^+$ = 正文档, $\mathcal{N}$ = 随机/BM25 负样本集, $\tau$ = 温度系数, $\text{sim}$ = 余弦相似度

**变化点**: 随机负样本可能与查询语义无关，导致模型只学习浅层区分；BM25 hard negatives 虽相关但缺乏指令级细粒度控制。本文改为**显式构造语义对立指令**，使负样本在指令层面与正样本形成镜像关系。

**本文公式（推导）**:
$$\text{Step 1}: \quad q^{\text{neg}} = \text{Flip}(q, d^+, \mathcal{E}) \quad \text{基于文档关键属性}\mathcal{E}\text{生成反义指令}$$
$$\text{Step 2}: \quad \mathcal{D}_{\text{synth}} = \{(q^{\text{neg}}_i, d^+_i, 0)\}_{i=1}^{N} \cup \{(q_i, d^+_i, 1)\}_{i=1}^{N} \quad \text{正负样本对保持同一文档，仅指令不同}$$
$$\text{Step 3}: \quad \mathcal{L}_{\text{pos}} = -\log \frac{\exp(\text{sim}(E(q), E(d^+))/\tau)}{\sum_{d \in \{d^+\} \cup \mathcal{D}_{\text{pool}}} \exp(\text{sim}(E(q), E(d))/\tau)}$$
$$\text{Step 4}: \quad \mathcal{L}_{\text{neg}} = -\log \frac{\exp(\text{sim}(E(q^{\text{neg}}), E(d^-))/\tau)}{\sum_{d \in \{d^-\} \cup \mathcal{D}_{\text{pool}}} \exp(\text{sim}(E(q^{\text{neg}}), E(d))/\tau)}$$
$$\text{最终}: \quad \mathcal{L}_{\text{dual}} = \mathcal{L}_{\text{pos}} + \lambda \cdot \mathcal{L}_{\text{neg}} \quad \text{其中 } d^- = d^+ \text{（同一文档在反义指令下为负例）}$$

**对应消融**: Table 2 显示移除极性反转（仅用随机负样本）导致 RetroMAE 在 MIRAGE 上下降 3.2 nDCG@10；移除联合训练（仅单独优化 $\mathcal{L}_{\text{pos}}$ 或 $\mathcal{L}_{\text{neg}}$）分别下降 2.1 和 4.5 nDCG@10。

### 模块 2: 双视角联合编码与对称约束（Dual-View Joint Encoding）

**直觉**: 原始指令和反义指令应被编码到具有明确几何关系的区域，使模型内部形成"语义对立 = 向量远离"的结构化表示。

**Baseline（标准对比学习）**: 仅优化查询-文档相似度，无查询-查询关系约束：
$$\mathcal{L}_{\text{base}} = \mathbb{E}_{(q, d^+, d^-)} \left[ -\log \frac{e^{\text{sim}(q,d^+)}}{e^{\text{sim}(q,d^+)} + e^{\text{sim}(q,d^-)}} \right]$$

**变化点**: Baseline 不约束 $q$ 与 $q^{\text{neg}}$ 的关系，导致模型可能将语义对立指令编码到相近区域。本文引入**对称性约束**，强制反义指令的表示与原始指令保持可预测的几何关系。

**本文公式（推导）**:
$$\text{Step 1}: \quad h_q = E(q), \quad h_{q^{\text{neg}}} = E(q^{\text{neg}}), \quad h_{d} = E(d) \quad \text{共享 encoder } E$$
$$\text{Step 2}: \quad \mathcal{L}_{\text{align}} = \| h_q + h_{q^{\text{neg}}} - 2\cdot \text{center}(h_q, h_{q^{\text{neg}}}) \|^2 \quad \text{保证对立指令关于原点对称（可选变体）}$$
$$\text{Step 3}: \quad \mathcal{L}_{\text{contrast}} = -\text{sim}(h_q, h_d) + \text{sim}(h_q, h_{q^{\text{neg}}}) \quad \text{原始查询远离反义指令，靠近正文档}$$
$$\text{最终}: \quad \mathcal{L}_{\text{joint}} = \mathcal{L}_{\text{dual}} + \mu \cdot \mathcal{L}_{\text{align}} + \nu \cdot \mathcal{L}_{\text{contrast}}$$

符号补充: $\lambda, \mu, \nu$ 为超参数（实验中 $\lambda=1.0, \mu=0.1, \nu=0.05$）；center 为两向量中点；对称约束实际实现为可选正则项。

**对应消融**: Table 2 显示移除对称约束（$\mu=0$）导致 BGE-M3 在 LoTTE 上 Recall@100 下降 1.8；移除对比项（$\nu=0$）下降 2.4，说明显式建模查询间对立关系对指令区分至关重要。

### 模块 3: 硬负样本挖掘与动态难度调度（Dynamic Hard Negative Mining）

**直觉**: 合成反义指令本身已是 hard negative，但训练过程中需动态调整难度以避免早期优化困难或后期过拟合。

**Baseline（标准 hard negative mining）**: 静态选择 top-k BM25 或上轮模型打分最高的负样本：
$$\mathcal{N}_{\text{hard}} = \text{TopK}_{d \in \mathcal{D} \text{setminus} \{d^+\}} \text{sim}_{\text{BM25}}(q, d)$$

**变化点**: 静态挖掘无法适应模型能力变化，且 BM25 与神经网络语义存在鸿沟。本文将**极性反转样本与模型当前状态结合**，动态调整有效负样本集合。

**本文公式（推导）**:
$$\text{Step 1}: \quad s_i(t) = \text{sim}(E_t(q_i), E_t(d_i^+)) \quad \text{第 } t \text{ 轮模型对正样本的打分}$$
$$\text{Step 2}: \quad \mathcal{N}_{\text{dynamic}}(t) = \{d_j : \text{rank}_{\text{sim}_t}(q_i, d_j) \in [k_1(t), k_2(t)]\} \quad \text{动态排名区间}$$
$$\text{Step 3}: \quad k_1(t) = \max(1, K - \lfloor t/T \cdot \Delta \rfloor), \quad k_2(t) = K + \lfloor t/T \cdot \Delta \rfloor \quad \text{随训练推进扩大挖掘范围}$$
$$\text{最终}: \quad \mathcal{L}_{\text{full}} = \mathcal{L}_{\text{joint}} \text{ with } \mathcal{N} = \mathcal{N}_{\text{dynamic}}(t) \cup \{d_i^+ \text{ under } q_i^{\text{neg}}\}$$

**对应消融**: 

## 实验与分析

主实验结果如 Table 1 所示，覆盖 MIRAGE（指令遵循检索）、LoTTE（领域迁移检索）及 BEIR 子集：

| Method | MIRAGE nDCG@10 | LoTTE Recall@100 | BEIR avg nDCG@10 |
|:---|:---|:---|:---|
| BGE-M3 (base) | 32.1 | 71.2 | 47.3 |
| BGE-M3 + Dual-View | **37.4** (+5.3) | **76.5** (+5.3) | **48.9** (+1.6) |
| RetroMAE (base) | 28.4 | 64.7 | 43.1 |
| RetroMAE + Dual-View | **33.1** (+4.7) | **70.3** (+5.6) | **44.5** (+1.4) |
| E5-mistral (7B) | 35.8 | 74.6 | 49.2 |
| GTE-large | 30.5 | 68.9 | 45.7 |
| GTE + Dual-View | 35.2 (+4.7) | 73.4 (+4.5) | 46.8 (+1.1) |


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/1bf5cc71-beab-4fcd-9151-ef0085143bae/figures/Table_1.png)
*Table 1 (quantitative): Table 1: Main results on instruction-following retrieval benchmarks.*



**核心发现分析**：
- **指令遵循增益显著**：Dual-View 在 MIRAGE（专门测试复杂指令遵循）上提升最大（+4.7~5.3），验证核心创新有效。BGE-M3 + Dual-View (37.4) 超越 7B 参数的 E5-mistral (35.8)，说明数据效率可弥补模型规模差距。
- **迁移能力稳健**：LoTTE 上类似幅度提升（+5.3~5.6），表明学习到的指令理解能力可跨领域迁移。
- **通用检索小幅提升**：BEIR 上仅 +1.1~1.6，说明 Dual-View 主要针对指令复杂场景，对标准 ad-hoc 检索增益有限——这与设计目标一致。

**消融实验**（Table 2）：

| 变体 | MIRAGE nDCG@10 | 相对完整模型下降 |
|:---|:---|:---|
| 完整 Dual-View | 37.4 | — |
| 移除极性反转（随机负样本） | 34.2 | -3.2 |
| 移除联合训练（仅 $\mathcal{L}_{\text{pos}}$） | 35.3 | -2.1 |
| 移除对称约束 | 35.6 | -1.8 |
| 移除动态调度（静态 top-50） | 36.1 | -1.3 |


![Table 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/1bf5cc71-beab-4fcd-9151-ef0085143bae/figures/Table_2.png)
*Table 2 (quantitative): Table 2: Results on bge-m3-retromae.*



**模块重要性排序**：极性反转合成 > 联合训练 > 对称约束 > 动态调度。极性反转是增益核心来源，验证"合成反义指令"比"挖掘 hard negatives"更有效的假设。

**公平性检查**：
- **Baseline 强度**：对比包含当前主流稠密检索器（BGE-M3、GTE）和指令微调大模型（E5-mistral），覆盖不同规模和方法路线，较为全面。
- **计算成本**：Dual-View 训练仅需额外极性反转步骤，无 LLM 推理开销；BGE-M3 + Dual-View 训练时间约增加 15%。
- **数据成本**：零额外标注，但依赖现有标注数据集的相关性标签质量。
- **局限**：BEIR 提升有限说明对简单查询可能"过度复杂化"；极性反转规则对多条件组合指令（"A 且非 B 或 C"）的覆盖未充分验证；失败案例显示对隐含否定（如"preferably not"）处理仍不稳定。

## 方法谱系与知识库定位

**方法家族**：稠密检索（Dense Retrieval）→ 对比学习优化 → 指令遵循检索（Instruction-Following IR）

**父方法**：标准 in-batch negative contrastive learning（如 Contriever、DPR）+ hard negative mining（ANCE、ADORE）

**改变的插槽**：
| 插槽 | 父方法 | 本文修改 |
|:---|:---|:---|
| data_curation | 人工/LLM 生成查询-文档对 | **自动极性反转合成反义指令** |
| objective | 单向查询-文档相似度 | **双向联合损失 + 对称约束** |
| training_recipe | 静态 hard negative 挖掘 | **动态难度调度** |
| architecture | 无修改（兼容任意双塔 encoder） | — |

**直接 Baseline 与差异**：
- **E5-mistral**：用 LLM 生成合成查询，本文不用 LLM 而用标注结构反转，成本更低且控制更精确
- **InPars / Promptagator**：基于 LLM 生成伪标签，本文利用已有标签的数学对称性
- **BGE-M3**：多粒度检索但未显式建模指令否定，本文作为骨干验证即插即用性
- **ANCE**：动态 hard negative 但仅针对文档侧，本文扩展至指令侧的动态对立

**后续方向**：
1. **多步组合指令**：将极性反转扩展至多条件逻辑（与/或/非嵌套），覆盖更复杂用户查询
2. **与生成式检索统一**：Dual-View 思想应用于 DSI、LLM-based 检索器的指令微调数据合成
3. **跨语言迁移**：验证极性反转规则在不同语言语法结构下的有效性，构建多语言指令遵循检索器

**知识库标签**：
- **modality**: text-to-text retrieval
- **paradigm**: contrastive learning + synthetic data augmentation
- **scenario**: instruction-following information retrieval, complex query understanding
- **mechanism**: polarity reversal, dual-view joint training, symmetry constraint
- **constraint**: zero-additional-annotation, plug-and-play to existing encoders

