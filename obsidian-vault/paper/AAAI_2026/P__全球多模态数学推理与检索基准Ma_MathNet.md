---
title: 'MathNet: a Global Multimodal Benchmark for Mathematical Reasoning and Retrieval'
type: paper
paper_level: B
venue: AAAI
year: 2026
paper_link: https://arxiv.org/abs/2604.18584
aliases:
- 全球多模态数学推理与检索基准MathNet
- MathNet
- 数学等价性是结构性属性而非语义属性——两个问题可以用完全不同的符号、语
code_url: https://github.com/ShadeAlsha/MathNet
method: MathNet
paradigm: Reinforcement Learning
---

# MathNet: a Global Multimodal Benchmark for Mathematical Reasoning and Retrieval

[Paper](https://arxiv.org/abs/2604.18584) | [Code](https://github.com/ShadeAlsha/MathNet)

**Topics**: [[T__Math_Reasoning]], [[T__Retrieval]], [[T__Benchmark_-_Evaluation]] | **Method**: [[M__MathNet]]

> [!tip] 核心洞察
> 数学等价性是结构性属性而非语义属性——两个问题可以用完全不同的符号、语言和表述形式表达同一数学对象，而现有基于词向量的嵌入模型对此天然盲目。MathNet 的核心直觉是：要暴露这一盲点，必须构建专门包含「表面相似但数学不等价」硬负例的检索基准，而不能依赖通用语义检索评估集。同时，通过 Zero Shot / Embed-RAG / Expert-RAG 三段式解耦设计，可以精确定位 RAG 系统的性能瓶颈来自检索器还是推理模型，为后续改进提供明确方向。

| 中文题名 | 全球多模态数学推理与检索基准MathNet |
| 英文题名 | MathNet: a Global Multimodal Benchmark for Mathematical Reasoning and Retrieval |
| 会议/期刊 | ICLR 2026 (会议论文) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.18584) · [Code](https://github.com/ShadeAlsha/MathNet) · [Project](https://github.com/ShadeAlsha/MathNet) |
| 主要任务 | 数学推理评估（MathNet-Solve）、数学感知检索评估（MathNet-Retrieve）、RAG系统解耦分析（MathNet-RAG） |
| 主要 baseline | OlympiadBench、MATH、GSM8K、通用语义检索基准（MS MARCO等） |

> [!abstract] 因为「现有数学基准规模有限、语言单一、且完全忽视数学感知检索能力」，作者在「OlympiadBench等现有基准」基础上改了「构建三层次评估生态系统（30K+全球竞赛题+硬负例检索集+RAG解耦协议）」，在「MathNet-Solve/Retrieve/RAG」上取得「gemini-3.1-pro 76.3%整体准确率、嵌入检索器暴露为RAG主要瓶颈」

- **关键性能 1**: MathNet-Solve 覆盖 30,676 道题，规模达 OlympiadBench（6,142题）的 5 倍
- **关键性能 2**: gemini-3.1-pro 整体准确率 76.3%，GPT-5 为 68.1%；几何最难，GPT-5 仅 56.3%
- **关键性能 3**: DeepSeek-V3.2-Speciale 在 Expert-RAG 较 Zero Shot 提升最高达 12%，但 Embed-RAG 与 Expert-RAG 差距显著，嵌入检索器为 RAG 瓶颈

## 背景与动机

现有数学推理基准存在三大结构性缺陷，严重制约了大语言模型和多模态模型的真实能力评估。以当前最大的奥林匹克级基准 OlympiadBench 为例，其仅含 6,142 道英文与中文题目，规模有限且语言覆盖单一；主流基准如 MATH 和 GSM8K 更是仅聚焦生成式解题能力，任务类型单一。这导致研究社区无法回答一个关键问题：当模型面对来自47个国家、17种语言、跨越40年历史的真实竞赛题时，其推理边界究竟在哪里？


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/860f89fb-d182-4215-8b08-d56401831604/figures/Figure_1.png)
*Figure 1: Figure 1: Overview of MATHNET. MATHNET contains 30K+ Olympiad-level problems across 47countries, 17 languages, and 143 competitions over 40 years with expert-authored solutions. Weevaluate several lea*



更深层的问题在于**数学感知检索（Math-Aware Retrieval）**的评估空白。现有嵌入模型基于语义相似度训练，对数学等价性这一结构性属性天然盲目：例如 $x^2+y^2=1$ 与 $a^2+b^2=1$ 数学等价（仅变量重命名），而 $x+y=1$ 虽表面词汇重叠却数学不等价，但通用检索系统往往将后者排在更高位置。这一缺陷直接制约了检索增强生成（RAG）在数学场景中的实际效果——研究者无法判断 RAG 失败究竟源于检索器找不到相关题目，还是推理模型无法利用检索到的上下文。

现有方法的处理方式及其局限：
- **OlympiadBench** 构建了多语言竞赛题集合，但仅评估端到端解题，未涉及检索环节；
- **MATH** 建立了难度分级的解题基准，但规模有限（12,500题）且全为英文，无多模态与跨语言设计；
- **通用检索基准（MS MARCO等）** 评估语义检索能力，但缺乏数学等价性判断所需的硬负例构造，无法暴露嵌入模型的结构性盲区。

这些局限的共性在于：将数学推理简化为单一生成任务，忽视了「检索-推理」耦合系统中检索质量的基础性作用。本文的核心动机正是填补这一空白——通过构建全球规模最大、语言最多、且首次系统评估数学感知检索与RAG解耦分析的基准，为社区提供定位瓶颈的精确工具。

## 核心创新

**核心洞察：数学等价性是结构性属性而非语义属性**，因为同一数学对象可用完全不同的符号系统、自然语言和表述形式表达（如几何图形的坐标法与综合法、同一方程的变量重命名），从而使「基于硬负例的检索评估 + RAG三段式解耦协议」成为可能——这是现有语义检索基准无法提供的诊断能力。

| 维度 | Baseline（OlympiadBench / MS MARCO） | 本文 MathNet |
|:---|:---|:---|
| 规模与覆盖 | 6,142题，2种语言 | 30,676题，17种语言，47国，143项竞赛，40年跨度 |
| 任务类型 | 仅生成式解题 | 解题（Solve）+ 检索（Retrieve）+ RAG解耦（RAG）三层次 |
| 检索评估 | 无数学专用检索基准 | 40K合成问题，含等价正例与硬负例，专门暴露嵌入模型对结构性等价的盲区 |
| RAG诊断 | 端到端黑箱评估 | Zero Shot / Embed-RAG / Expert-RAG 三段式协议，精确定位瓶颈来源 |
| 评估粒度 | 二元正确/错误 | GPT-5裁判0-7分制，区分真正推理与偶然答对；68种题型本体分类 |

## 整体框架


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/860f89fb-d182-4215-8b08-d56401831604/figures/Figure_3.png)
*Figure 3: Figure 3: Overview of MATHNET problem–solution extraction pipeline. The curation pipelineconsists of three stages: (1) document ingestion and problem segmentation, (2) problem and solutionextraction w*



MathNet 的三层次评估生态系统按数据流展开如下：

**输入层：原始竞赛文档**
→ 收集全球47个国家、143项数学竞赛、40年历史的PDF/扫描件/网页，涵盖17种语言。

**模块一：文档摄取与预处理（Document Ingestion & Preprocessing）**
→ 输入：原始多模态文档（含LaTeX、手写扫描、几何图形）
→ 处理：OCR提取、LaTeX解析、自然语言与公式对齐
→ 输出：结构化原始题目-解答对

**模块二：专家验证与本体标注（Expert Verification & Ontology Annotation）**
→ 输入：结构化原始数据
→ 处理：系统性人工验证确保LaTeX与自然语言描述对齐；按68种题型本体分类（代数、几何、数论、组合等子领域）
→ 输出：MathNet-Solve 主语料库（30,676道验证题，配官方解答）

**模块三：检索数据集合成（Retrieval Set Synthesis）**
→ 输入：10K锚点问题
→ 处理：合成生成40K额外问题，每个锚点配1个数学等价正例（变量重命名、等价表述）和3个表面相似但数学不等价的硬负例
→ 输出：MathNet-Retrieve（50K问题对，专门测试数学感知检索）

**模块四：RAG解耦基准构建（RAG Decoupling Benchmark）**
→ 输入：70道IMO级精选问题
→ 处理：人工专家配对结构相似的相关问题及官方解答
→ 输出：MathNet-RAG（支持三种推理设置对比）

**评估协议层**
→ MathNet-Solve：GPT-5裁判0-7分，≥6分正确
→ MathNet-Retrieve：Recall@k + 余弦相似度分布分析
→ MathNet-RAG：Zero Shot vs Embed-RAG vs Expert-RAG 三段对比

```
原始文档 → [摄取预处理] → 结构化数据 → [专家验证] → MathNet-Solve (30K+题)
                                    ↓
                              [合成生成] → MathNet-Retrieve (50K对, 硬负例)
                                    ↓
                              [人工配对] → MathNet-RAG (70题, 三段式RAG)
```

## 核心模块与公式推导

### 模块 1: 解题评估协议——GPT-5裁判评分（对应框架图 MathNet-Solve 评估层）

**直觉**: 数学推理的正确性不能简单用字符串匹配判断，需要细粒度评分区分「完全正确」「思路正确但计算错误」「完全错误」等层次。

**Baseline 公式** (OlympiadBench / MATH 等): 
$$\text{Acc}_{\text{base}} = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}[\text{pred}_i = \text{answer}_i]$$
符号: $\text{pred}_i$ = 模型生成答案, $\text{answer}_i$ = 标准答案, $\mathbb{1}[\cdot]$ = 指示函数

**变化点**: 二元正确/错误无法捕捉「推理过程部分正确」的情况；且开放式证明题无唯一答案字符串。本文改为**细粒度裁判评分 + 阈值判定**。

**本文公式（推导）**:
$$\text{Step 1}: \quad s_i = \text{GPT-5}\_\text{Judge}(q_i, \text{official\_sol}_i, \text{pred\_sol}_i) \in \{0,1,2,3,4,5,6,7\} \quad \text{（0-7分制评分，加入官方解答作为参考标准）}$$
$$\text{Step 2}: \quad \hat{y}_i = \mathbb{1}[s_i \geq 6] \quad \text{（阈值6分视为正确，保证与人工判断高一致性）}$$
$$\text{最终}: \quad \text{Acc}_{\text{MathNet}} = \frac{1}{N}\sum_{i=1}^{N} \hat{y}_i, \quad \text{同时报告} \bar{s} = \frac{1}{N}\sum_{i=1}^{N} s_i \text{（平均得分）}$$

**对应消融**: 

---

### 模块 2: 数学感知检索评估——硬负例构造与Recall@k（对应框架图 MathNet-Retrieve 层）

**直觉**: 通用检索的负例随机采样过于简单，无法测试模型是否真正理解数学结构；必须构造「表面相似、数学不等价」的硬负例才能暴露嵌入模型的结构性盲区。

**Baseline 公式** (MS MARCO / 通用检索):
$$\text{Recall@}k_{\text{base}} = \frac{1}{|Q|}\sum_{q \in Q} \frac{|\{d \in D_q^+ : \text{rank}(d) \leq k\}|}{|D_q^+|}$$
其中 $D_q^+$ 为相关文档集，负例 $D_q^-$ 通常随机采样或基于BM25。

**变化点**: 随机负例无法区分「语义相似但数学不等价」的情况；本文引入**结构化硬负例合成**，将数学等价性判断显式纳入评估。

**本文公式（推导）**:
$$\text{Step 1}: \quad \text{等价正例生成}: d_q^{+} = \text{Transform}_{\text{eq}}(q), \quad \text{其中 Transform}_{\text{eq}} \in \{\text{变量重命名}, \text{等价表述}, \text{坐标法↔综合法转换}\}$$
$$\text{Step 2}: \quad \text{硬负例生成}: d_{q,j}^{-} = \text{Transform}_{\text{hard}}(q), \quad j=1,2,3$$
$$\text{其中 Transform}_{\text{hard}} \text{ 保持表面特征（词汇重叠、公式结构相似）但破坏数学等价性}$$
$$\text{Step 3}: \quad \text{候选池构造}: C_q = \{d_q^{+}\} \cup \{d_{q,j}^{-}\}_{j=1}^{3} \cup D_{\text{irrelevant}}$$
$$\text{最终}: \quad \text{Recall@}k_{\text{MathNet}} = \frac{1}{|Q|}\sum_{q \in Q} \mathbb{1}[\text{rank}(d_q^{+}) \leq k]$$
$$\text{附加分析}: \quad \Delta_{\text{sim}} = \mathbb{E}[\cos(E(q), E(d^{+}))] - \mathbb{E}[\cos(E(q), E(d^{-}_{\text{hard}}))]$$
（$\Delta_{\text{sim}}$ 揭示嵌入模型区分等价与硬负例的能力，理想情况应显著大于0）

**对应消融**: 

---

### 模块 3: RAG解耦分析协议（对应框架图 MathNet-RAG 层）

**直觉**: RAG系统失败时，无法判断是「检索器没找到」还是「找到了但推理器没用好」；需要人为控制检索质量来解耦两个组件的贡献。

**Baseline 公式** (标准RAG评估):
$$\text{Acc}_{\text{RAG}} = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}[\text{LLM}(q_i, \text{Retrieve}(q_i)) = \text{correct}]$$
黑箱端到端，无法分解 $\text{Retrieve}$ 与 $\text{LLM}$ 的独立贡献。

**变化点**: 引入**人工专家检索作为上界**，通过三段式对比实现解耦：Zero Shot（无检索，测纯推理能力）→ Embed-RAG（真实检索器，测实际系统）→ Expert-RAG（人工精选，测检索质量上界时的推理能力）。

**本文公式（推导）**:
$$\text{Step 1}: \quad \text{Zero Shot}: \quad A_0 = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}[\text{LLM}(q_i | \emptyset) = \text{correct}] \quad \text{（纯推理基线，无上下文）}$$
$$\text{Step 2}: \quad \text{Embed-RAG}: \quad A_E = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}[\text{LLM}(q_i | \text{EmbedRetr}(q_i)) = \text{correct}] \quad \text{（真实嵌入检索器提供的上下文）}$$
$$\text{Step 3}: \quad \text{Expert-RAG}: \quad A_X = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}[\text{LLM}(q_i | \text{Expert}(q_i)) = \text{correct}] \quad \text{（人工专家精选的相关问题+解答）}$$
$$\text{最终解耦指标}:$$
$$\Delta_{\text{retrieval\_value}} = A_E - A_0 \quad \text{（嵌入检索带来的实际增益，衡量检索器价值）}$$
$$\Delta_{\text{retrieval\_gap}} = A_X - A_E \quad \text{（检索错误造成的残余损耗，衡量当前检索器距最优的差距）}$$
$$\Delta_{\text{reasoning\_ceiling}} = A_X - A_0 \quad \text{（理想检索下的最大推理增益，衡量推理模型利用相关上下文的上界）}$$

**对应消融**: 实验显示 DeepSeek-V3.2-Speciale 的 $\Delta_{\text{retrieval\_value}}$ 最高达12%，但多数模型的 $\Delta_{\text{retrieval\_gap}} > 0$ 且显著，证明嵌入检索器是当前RAG流水线的主要瓶颈。

## 实验与分析

**主结果：MathNet-Solve 解题准确率（27模型对比）**

| 模型 | 整体准确率 | 代数 | 几何 | 数论 | 组合/离散 | 备注 |
|:---|:---|:---|:---|:---|:---|:---|
| gemini-3.1-pro | 76.3% | — | — | — | — | 最高整体 |
| GPT-5 | 68.1% | 82.9% | 56.3% | — | 64.1% | 裁判模型自身 |
| （其他25个模型） |  | — | — | — | — | — |


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/860f89fb-d182-4215-8b08-d56401831604/figures/Figure_2.png)
*Figure 2: Figure 2: Overview of MathNet-Solve. The dataset spans national, regional, TST, and interna-tional competitions, with varying solution lengths. It has grown since the early 2000s and includestextual a*



**关键发现分析**：
- **支持核心 claim 的数字**：gemini-3.1-pro 的 76.3% 与 GPT-5 的 68.1% 差距显著，说明模型间存在真实能力分化；几何 56.3% vs 代数 82.9% 的巨大跨度（26.6个百分点）验证了MathNet题型分类能有效暴露领域特异性弱点。
- **边际/存疑数字**：摘要报告 gemini-3.1-pro 78.4%、GPT-5 69.3%，与正文 76.3%、68.1% 不一致，削弱结果可信度，应以正文为准。

**RAG解耦实验结果**：

| 设置 | DeepSeek-V3.2-Speciale | 典型模型范围 |
|:---|:---|:---|
| Zero Shot | 基线 | 基线 |
| Embed-RAG | +若干% |  |
| Expert-RAG | 较Zero Shot最高+12% | — |
| $\Delta_{\text{retrieval\_gap}}$ (Expert-Embed) | 显著 > 0 | 普遍显著 |



**消融与瓶颈定位**：
- Embed-RAG 与 Expert-RAG 之间的显著差距（$\Delta_{\text{retrieval\_gap}} > 0$）是本文最关键的发现，直接证明**当前嵌入检索器是RAG系统的主要瓶颈**，而非推理模型无法利用相关上下文。
- 70道IMO级题目的MathNet-RAG样本量极小，RAG结论的统计显著性和泛化性有限。

**公平性检查**：
- **Baseline强度**：未纳入 Tangent-CFT、MathDowsers 等专用数学公式检索系统，可能低估专用系统的竞争力；嵌入检索器仅使用通用语义模型。
- **计算/数据成本**：30,676道题的专家验证成本高昂，可扩展性受限；GPT-5作为裁判引入模型依赖偏差。
- **失败案例**：几何领域普遍低分（GPT-5仅56.3%），暗示多模态图形理解仍是未解决难题；硬负例检索实验中，嵌入模型对变量重命名后的等价问题识别率。

## 方法谱系与知识库定位

**方法家族**：数学推理基准测试 → 多模态/多语言数据工程 → 检索增强生成（RAG）诊断工具

**Parent Method**：OlympiadBench（2023）——首个大规模多语言奥林匹克数学基准，提供竞赛题收集与评估的基础范式。MathNet 直接继承其「竞赛题-官方解答」的数据结构，但在规模（5×）、语言（17 vs 2种）、任务维度（解题+检索+RAG vs 仅解题）上全面扩展。

**直接 Baselines 与差异**：
- **OlympiadBench**：MathNet 扩展至5倍规模、8.5倍语言覆盖，新增检索与RAG评估维度
- **MATH / GSM8K**：MathNet 聚焦奥林匹克级难度而非K-12，且首次系统评估检索能力
- **MS MARCO / 通用检索基准**：MathNet-Retrieve 引入数学专用硬负例构造，暴露结构性等价判断盲区
- **MMLU-STEM / SciEval**：MathNet 专精数学领域，提供68种题型本体而非宽泛科学分类

**Slots 改变**：
- **data_curation**：从单一英文/中文扩展至17语言47国；从随机负例扩展至数学硬负例合成
- **evaluation_recipe**：从二元正确/错误扩展至0-7分GPT-5裁判；从端到端RAG扩展至三段式解耦协议
- **architecture / objective / inference**：无模型架构或训练方法创新（纯基准工作）

**Follow-up 方向**：
1. **专用数学嵌入模型**：基于MathNet-Retrieve硬负例训练，提升结构性等价判断能力
2. **多模态几何理解**：针对56.3%低分领域，开发图形-文本联合推理模型
3. **自动化硬负例生成**：当前合成方法描述不充分，需更系统的数学等价/不等价变换算法

**知识库标签**：
- modality: `text+latex+geometry_figures`（多模态）
- paradigm: `benchmark/evaluation`（基准评估）
- scenario: `mathematical_reasoning`, `olympiad_level`, `retrieval_augmented_generation`
- mechanism: `hard_negative_mining`, `decoupled_evaluation`, `expert_in_the_loop`
- constraint: `multilingual_17`, `cross_lingual`, `scalable_curation`

