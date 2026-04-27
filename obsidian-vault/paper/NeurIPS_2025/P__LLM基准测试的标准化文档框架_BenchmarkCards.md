---
title: 'BenchmarkCards: Standardized Documentation for Large Language Model Benchmarks'
type: paper
paper_level: A
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- LLM基准测试的标准化文档框架
- BenchmarkCards
acceptance: Poster
cited_by: 3
code_url: https://github.com/SokolAnn/BenchmarkCards/
method: BenchmarkCards
modalities:
- Text
paradigm:
- supervised
- N/A - not a training paper
baselines:
- ML就绪数据集的Croissan_Croissant
---

# BenchmarkCards: Standardized Documentation for Large Language Model Benchmarks

[Code](https://github.com/SokolAnn/BenchmarkCards/)

**Topics**: [[T__Benchmark_-_Evaluation]] | **Method**: [[M__BenchmarkCards]] | **Datasets**: User Study - Benchmark Author Survey, User Study - Benchmark User Interviews, Case Study - BBQ vs RealToxicityPrompts, User Study

> [!tip] 核心洞察
> BenchmarkCards, a validated documentation framework standardizing critical benchmark attributes, simplifies benchmark selection and enhances transparency for informed LLM evaluation.

| 中文题名 | LLM基准测试的标准化文档框架 |
| 英文题名 | BenchmarkCards: Standardized Documentation for Large Language Model Benchmarks |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2410.12974) · [Code](https://github.com/SokolAnn/BenchmarkCards/) · [Project](https://github.com/SokolAnn/BenchmarkCards/) |
| 主要任务 | Benchmark / Evaluation |
| 主要 baseline | Model Cards, Datasheets, FactSheets, Risk Cards, Croissant |

> [!abstract] 因为「LLM 基准测试缺乏标准化文档框架，导致难以比较、选择和正确解读结果」，作者在「Model Cards / Datasheets / FactSheets / Risk Cards」基础上扩展了「Cards」文档范式至基准测试领域，设计了「BenchmarkCards 结构化模板」，通过「两组用户研究（benchmark authors 与 users, n=10）」验证了「框架的正确性、完整性与实用性」。

- 核心性能：用户研究显示 BenchmarkCards 显著简化了基准测试选择流程，提升了透明度（qualitative validation）
- 覆盖范围：模板涵盖 Purpose, Methodology, Data, Metrics, Targeted Risks, Demographic Analysis, Ethical & Legal Considerations, Limitations 等 8+ 个专用字段
- 自动化支持：使用 gpt-4o-mini 从现有 benchmark 论文中自动抽取信息生成初始卡片草稿

## 背景与动机

当前 LLM 领域存在数千个基准测试，但研究者面临一个根本性困境：当需要评估模型在特定场景下的表现时，如何快速判断哪个 benchmark 真正适用？例如，一个关注法律领域模型公平性的研究者，可能需要在 LexGLUE、BBQ、WinoQueer 等众多基准中筛选，但现有 benchmark 的原始论文往往分散描述其目的、数据构成、评估指标和潜在风险，缺乏统一结构。

现有文档框架各自覆盖不同层面，但均未直接解决 benchmark 级别的标准化文档需求：
- **Model Cards** [2]：针对单个 ML 模型的文档化，记录预期用途、性能表现与限制，但不涉及评估该模型所用的 benchmark 本身
- **Datasheets** [7] 与 **Croissant** [2]：专注于数据集层面的元数据，描述数据动机、组成、收集过程，但未涵盖 metrics、pre/post-processing、evaluation methodology 等 benchmark 核心组件
- **FactSheets** [3]：面向 AI 系统整体的服务商合规声明，粒度较粗，缺乏 benchmark 特有的方法论细节
- **Risk Cards** [15]：最接近的先驱，结构化评估语言模型部署风险，但目标实体是模型部署而非 benchmark 本身

这些框架的共同局限在于：**它们将 benchmark 视为黑箱工具，而非需要独立透明化文档的实体**。Benchmark 包含数据集、指标、评估流程、预处理规则、人口统计学考量、伦理风险等多维信息，这些信息分散在论文不同位置，导致 benchmark 选择依赖研究者的经验判断，增加了误用和误读风险——例如将面向通用能力的 benchmark 用于评估专业领域模型，或忽视 benchmark 已知的人口统计学偏见。

本文提出 BenchmarkCards，首次将 "Cards" 文档范式系统性地扩展至 LLM benchmark 领域，建立专门面向基准测试的标准化文档结构。

## 核心创新

核心洞察：Benchmark 是一个**复合实体**（数据集 + 指标 + 预处理/后处理 + 评估方法论 + 风险声明），而非单纯的数据集或模型附属品，因此需要独立于 Model Cards 和 Datasheets 的文档框架；因为现有 "Cards" 范式已成功验证于模型与系统层面，将其适配至 benchmark 这一新实体类型，从而使 benchmark 的透明比较、负责任选择和结果正确解读成为可能。

| 维度 | Baseline (Model Cards / Datasheets / Risk Cards) | 本文 (BenchmarkCards) |
|:---|:---|:---|
| **目标实体** | 单个 ML 模型 / 数据集 / 模型部署风险 | **Benchmark（评估工具本身）** |
| **核心字段** | 模型性能、训练数据、预期用途；或数据集组成、收集过程 | **Purpose, Methodology, Data, Metrics, Targeted Risks, Demographic Analysis, Ethical & Legal Considerations, Limitations** |
| **信息范围** | 模型行为描述；或纯数据描述；或风险评估 | **数据集 + 指标 + 预处理/后处理 + 评估流程 + 伦理考量 + 限制** 的完整闭环 |

## 整体框架


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/eaf2bc1c-9c44-46f1-afbe-b3d4e98bb137/figures/Table_1.png)
*Table 1: BenchmarkCard Template*



BenchmarkCards 的整体流程包含四个核心阶段，形成从原始论文到可验证文档的完整 pipeline：

1. **输入源 (Benchmark Paper)**：以已发表的 benchmark 论文为信息源，包含分散在论文各章节的方法描述、数据说明、评估设置等

2. **自动化信息抽取 (gpt-4o-mini Extraction)**：使用 gpt-4o-mini 从非结构化论文文本中提取结构化信息，生成初始 BenchmarkCard 草稿。该模块将自然语言描述映射到预定义的模板字段

3. **作者验证 (Author Verification)**：邀请 benchmark 原作者审查生成的卡片草稿，验证正确性与完整性，进行修正补充，形成权威版本

4. **用户效用评估 (User Utility Assessment)**：独立开展用户研究，让 benchmark 使用者（如模型评估研究者）基于 BenchmarkCards 进行基准选择决策，收集关于实用性、可理解性、全面性的反馈

最终输出为经过验证的 **Final BenchmarkCard**，作为社区共享的标准化文档工件，托管于公开仓库。

```
Benchmark Paper
    ↓
gpt-4o-mini Extraction ──→ Initial Card Draft
    ↓
Author Verification ──→ Verified/Corrected Card
    ↓
User Utility Assessment (parallel validation)
    ↓
Final BenchmarkCard → Public Repository
```

## 核心模块与公式推导

本文作为概念性框架论文，核心贡献在于结构化模板设计与验证方法论，而非数学公式推导。以下详述三个核心模块的设计逻辑与操作化定义：

### 模块 1: BenchmarkCard 模板结构（对应 Table 1）

**直觉**：Benchmark 的透明使用需要覆盖其全生命周期信息，从设计动机到使用限制，每个字段对应一个常见的误用风险点。

**Baseline 模板** (Model Cards [2] / Datasheets [7] / Risk Cards [15]):
- Model Cards: Intended Use, Factors, Metrics, Evaluation Data, Training Data, Quantitative Analyses, Ethical Considerations, Caveats and Recommendations
- Datasheets: Motivation, Composition, Collection Process, Preprocessing/Cleaning, Uses, Distribution, Maintenance
- Risk Cards: Risk Category, Risk Description, Assessment Method, Mitigation Strategy, Residual Risk

**变化点**：上述模板要么聚焦模型（Model Cards）、要么聚焦纯数据（Datasheets）、要么聚焦风险评估（Risk Cards），均缺少 benchmark 特有的 **evaluation methodology**、**metrics 与 pre/post-processing 的绑定关系**、**targeted risks 与 demographic analysis 的显式关联**。

**本文模板字段（操作化定义）**：
- **Purpose**：benchmark 的设计目标与适用场景，解决"这个 benchmark 测什么"
- **Methodology**：评估流程、实验设置、打分规则，解决"怎么测"
- **Data**：数据来源、规模、预处理、后处理，解决"用什么测"
- **Metrics**：具体指标定义与计算方式，解决"如何量化"
- **Targeted Risks**：benchmark 旨在揭示的特定风险类型
- **Demographic Analysis**：数据或评估中涉及的人口统计学维度
- **Potential Harm**：已知或潜在的负面应用后果
- **Ethical & Legal Considerations**：伦理审查、法律合规要求
- **Limitations**：benchmark 本身的边界条件与已知缺陷

**对应验证**：Table 1 展示完整模板结构；Table 2 通过 BBQ 与 RealToxicityPrompts 的对比示例，说明不同 benchmark 在 Targeted Risks 和 Demographic Analysis 字段的差异化填写如何帮助用户区分其适用场景。

### 模块 2: 自动化信息抽取流水线（gpt-4o-mini Extraction）

**直觉**：手动为数百个现有 benchmark 创建卡片成本极高，需要自动化启动机制，但 LLM 抽取的准确性需要人工校验兜底。

**Baseline 做法**：纯人工撰写 benchmark 文档（如各 benchmark 原始论文的方法章节），或依赖 Croissant 等半自动元数据格式（仅覆盖数据集层面）。

**变化点**：引入 gpt-4o-mini 作为**草稿生成器**而非最终输出，将非结构化论文文本映射到 BenchmarkCard 的严格字段结构；与 Croissant 不同，抽取范围扩展至 methodology、metrics、risks 等非数据字段。

**操作流程**：
$$
\text{Step 1: } \text{Paper} \text{xrightarrow}{\text{chunking}} \text{Sectioned Text (Purpose, Method, Data, Results, Limitations)}
$$
$$
\text{Step 2: } \text{Sectioned Text} \text{xrightarrow}{\text{gpt-4o-mini + prompt engineering}} \text{Draft Card Fields}
$$
$$
\text{Step 3: } \text{Draft Card} \text{xrightarrow}{\text{human author verification}} \text{Verified BenchmarkCard}
$$

**关键设计**：抽取提示词（prompt）需显式定义每个字段的边界条件，例如区分 "Data"（benchmark 使用的数据集）与 "Training Data"（被测模型的训练数据），避免 Model Cards 常见的信息混淆。

**对应验证**：用户研究中指出 gpt-4o-mini 生成的草稿需要人工修正，但未报告具体错误率；作者验证环节的设计即为此兜底机制。

### 模块 3: 双轨验证机制（Author + User Studies）

**直觉**：文档框架的有效性需同时满足"生产者正确"（作者认可）和"消费者有用"（用户能据此做出更好决策）两个标准。

**Baseline 做法**：多数文档框架（如 Model Cards）仅提供模板规范，缺乏系统性验证；或仅做单一视角评估（如仅专家审查或仅用户调研）。

**变化点**：设计**双轨并行验证**——作者侧验证正确性与完整性，用户侧验证实用性与可理解性，两者反馈迭代优化模板。

**验证流程**：
$$
\text{Track A (Author Study): } \text{Generated Card} \rightarrow \text{Authors rate correctness & completeness} \rightarrow \text{Template refinement}
$$
$$
\text{Track B (User Study): } n=10 \text{ participants use Cards for benchmark selection} \rightarrow \text{Assess utility, understandability, comprehensiveness}
$$

**关键发现**：用户研究（n=10）表明 BenchmarkCards 解决了 benchmark 选择中的关键痛点——参与者能够基于结构化字段快速排除不相关 benchmark，并识别出原始论文中分散描述的关键限制信息。

**对应局限**：样本量较小（n=10），且缺乏定量指标（如选择准确率、时间效率的数值对比），验证强度为定性级别。

## 实验与分析


![Table 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/eaf2bc1c-9c44-46f1-afbe-b3d4e98bb137/figures/Table_2.png)
*Table 2 (comparison): Comparison of BHQ and RealToxicityPrompts Benchmarks*



本文的验证体系围绕两组互补的用户研究展开，而非传统 benchmark 上的数值指标比较。实验设计的核心问题是：BenchmarkCards 是否能同时满足 benchmark 生产者（作者）的准确性要求与消费者（用户）的实用性需求？

**基准测试文档对比验证**：Table 2 展示了 BBQ（Bias Benchmark for Question Answering）与 RealToxicityPrompts 两个 benchmark 的 BenchmarkCard 对比实例。通过结构化字段的并置，读者可以清晰识别两者的关键差异：BBQ 的 Targeted Risks 聚焦于社会群体偏见（如种族、性别、年龄），Demographic Analysis 明确标注了受保护群体的覆盖范围；而 RealToxicityPrompts 的 Targeted Risks 侧重于毒性内容生成，Demographic Analysis 维度不同。这种结构化对比在原始论文中需要跨多篇文献手动整合，BenchmarkCards 将其压缩为单页可扫描的格式。



**作者验证研究（Author Study）**：邀请 benchmark 原作者审查 gpt-4o-mini 生成的卡片草稿。验证维度包括正确性（信息是否准确反映 benchmark 设计）和完整性（关键信息是否遗漏）。该研究确认模板字段能够覆盖 benchmark 的核心元信息，同时识别出自动化抽取的边界案例——例如某些 benchmark 的 pre-processing 规则隐含在代码库而非论文正文中，需要作者补充。

**用户效用研究（User Study, n=10）**：10 名具有 LLM 评估经验的研究者参与，任务为基于 BenchmarkCards 从候选集合中选择适合特定场景的 benchmark。定性反馈表明：参与者认为 BenchmarkCards 显著简化了选择流程，特别是 Targeted Risks 和 Limitations 字段帮助快速排除不适用选项；参与者能够识别出原本需要深入阅读原始论文才能发现的 benchmark 限制（如特定人口群体的覆盖不足）。然而，研究未报告选择准确率、时间节省比例等定量指标，也未设置对照组（如无 BenchmarkCards 的基线选择流程）。

**公平性检验**：本文的比较基线主要为概念层面的文档框架（Model Cards, Datasheets, FactSheets, Risk Cards），而非直接的 benchmark 文档竞品。缺失的对比包括：与 Croissant 在 benchmark 元数据覆盖上的定量比较；与 Unitxt（灵活评估框架）在 benchmark 处理流程上的对比；以及自动化卡片生成质量的客观指标（如字段填充准确率、与人工撰写的 BLEU/ROUGE 分数）。此外，gpt-4o-mini 抽取引入的误差范围未量化，且框架的大规模社区采纳效果有待验证——当前公开仓库的 star 数与贡献者规模尚不明确。

## 方法谱系与知识库定位

**方法家族**：AI 透明化文档（"Cards" 范式）

**直接父方法**：Risk Cards [15] —— 将 "Cards" 结构化文档从模型部署风险评估扩展至 benchmark 文档化，继承其字段化设计哲学，但将目标实体从"模型部署"切换为"评估工具本身"。

**谱系关系**：
- **Model Cards** → BenchmarkCards：目标实体从"模型"变为"benchmark"，新增 methodology、metrics、pre/post-processing 字段
- **Datasheets** → BenchmarkCards：从纯"数据集描述"扩展至"数据集 + 评估流程 + 风险声明"的完整闭环
- **FactSheets** → BenchmarkCards：从"AI 系统级合规声明"细化至"benchmark 级技术文档"
- **Croissant** → BenchmarkCards：从"ML-ready 数据集元数据格式"扩展至包含评估方法论和风险考量的 benchmark 专用格式

**改动槽位**：
- **architecture**：通用模板 → benchmark 专用 9 字段模板
- **data_pipeline**：模型/数据集/系统文档 → benchmark 复合实体文档（数据集 + 指标 + 流程 + 风险）
- **training_recipe**：无训练，但引入 gpt-4o-mini 自动化抽取作为生成流程
- **inference**：无推理，但用户研究验证作为效用评估机制

**后续方向**：
1. **规模化自动化**：将 gpt-4o-mini 抽取升级为更精确的专用模型，建立字段级准确率评估基准
2. **社区生态建设**：与 Hugging Face、Papers With Code 等平台集成，推动 BenchmarkCards 成为 benchmark 提交的标准配套
3. **动态更新机制**：设计版本控制与更新协议，解决 benchmark 迭代（如 HumanEval → HumanEval+）时的文档同步问题

**标签**：modality=text | paradigm=documentation_framework | scenario=LLM_evaluation | mechanism=structured_metadata + automated_extraction + human_verification | constraint=community_adoption_dependent

## 引用网络

### 直接 baseline（本文基于）

- [[P__ML就绪数据集的Croissan_Croissant]] _(方法来源)_: Croissant is a closely related metadata format for datasets; BenchmarkCards like

