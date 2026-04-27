---
title: 'Rigor in AI: Doing Rigorous AI Work Requires a Broader, Responsible AI-Informed Conception of Rigor'
type: paper
paper_level: A
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- AI研究六维度严谨性框架
- Six-Facet Rigor
- Six-Facet Rigor Framework for AI
- Rigorous AI work requires expanding
acceptance: Poster
cited_by: 3
method: Six-Facet Rigor Framework for AI
modalities:
- Text
---

# Rigor in AI: Doing Rigorous AI Work Requires a Broader, Responsible AI-Informed Conception of Rigor

**Topics**: [[T__Benchmark_-_Evaluation]] | **Method**: [[M__Six-Facet_Rigor_Framework_for_AI]]

> [!tip] 核心洞察
> Rigorous AI work requires expanding beyond methodological rigor to include six facets: epistemic, normative, conceptual, methodological, reporting, and interpretative rigor.

| 中文题名 | AI研究六维度严谨性框架 |
| 英文题名 | Rigor in AI: Doing Rigorous AI Work Requires a Broader, Responsible AI-Informed Conception of Rigor |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2506.14652) · [DOI](https://doi.org/10.48550/arxiv.2506.14652) |
| 主要任务 | AI研究质量评估 / 负责任AI |
| 主要 baseline | 传统方法论严谨性（Methodological rigor only）|

> [!abstract] 因为「AI领域对严谨性的理解过于狭窄，仅聚焦于数学/统计/计算正确性，导致夸大AI能力与忽视意外后果」，作者在「传统方法论严谨性」基础上扩展了「六维度严谨性框架（epistemic, normative, conceptual, methodological, reporting, interpretative）」，在「AI研究评估范式」上提出「系统性重构严谨性概念的理论框架」。

- 框架将严谨性从单一维度扩展为六个相互依赖的维度
- 识别出上游维度（认识论、概念性）对下游维度（方法论、报告、解释）的决定性影响
- 强调规范性考量（normative rigor）贯穿所有维度的交叉作用

## 背景与动机

当前AI研究存在一种系统性偏见：当研究者谈论"严谨性"（rigor）时，几乎总是指向方法论层面的技术正确性——代码是否可复现、实验是否对照、benchmark分数是否领先。然而，这种狭窄的严谨性观念本身已成为问题。例如，大量NLP研究在未充分审视训练数据的社会偏见（epistemic缺口）或模型"理解"能力的理论定义（conceptual模糊）的情况下，便基于benchmark表现做出强认知声明，最终导致"随机鹦鹉"式的过度承诺与可重复性危机。

现有工作如何回应这一问题？**传统方法论严谨性**（methodological rigor）要求数学证明、统计显著性检验与计算可扩展性，但完全不追问"为何选择这个问题"或"这些假设从何而来"。**Data Statements for NLP** [17] 尝试通过文档化数据收集过程来提升透明度，但仅覆盖报告维度，未触及上游的认识论与概念性选择。**Values in ML Research** [22] 揭示了ML研究中嵌入的价值判断，却未将其整合为一个系统的评估框架。**NeurIPS Broader Impact Statements** [13] 强制研究者思考社会影响，但作为一种行政要求，缺乏与严谨性概念的理论整合，常流于形式。

这些工作的共同局限在于：它们各自解决了严谨性的某个碎片，却未将AI研究视为一个从"问题选择→理论建构→方法执行→结果报告→认知推断"的完整流程。更关键的是，传统观念将方法论置于核心，仿佛只要方法正确，上游的假设与下游的声明便自动可靠——这一预设本身正是众多AI夸大宣传的根源。作者因此提出：严谨性必须被重新构想为一个六维度的、具有内在依赖结构的系统性概念。
![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/7ff3d36d-4791-48e3-b05d-93eb7d964add/figures/Figure_1.png)
*Figure 1: Simplified overview of the objects of concern for each facet of rigor and of common interdependencies across them. Solid arrows from one facet to another indicate that the object of concern of one facet can build, while dashed arrows indicate that the object of concern of one facet can inform. Dependencies are illustrated through both arrows as well as dotted boxes.*



## 核心创新

核心洞察：AI研究的严谨性不能仅由方法论正确性定义，因为研究的上游选择（认识论假设、概念操作化、价值立场）与下游后果（报告透明度、推断适度性）共同决定了工作的科学 integrity，从而使六维度相互依赖的框架成为评估AI研究质量的必要语言。

| 维度 | Baseline（传统严谨性） | 本文 |
|:---|:---|:---|
| 评估范围 | 技术方法及其应用 | 完整研究流程：从背景知识到认知推断 |
| 核心关切 | 数学/统计/计算正确性、benchmark、可扩展性 | 六维度：认识论、规范性、概念性、方法论、报告、解释 |
| 维度关系 | 单一维度，无结构 | 上游→下游依赖链，规范性贯穿所有维度 |
| 目标受众 | 技术同行评审 | 研究者、政策制定者、记者、利益相关者的跨学科对话 |

## 整体框架



六维度严谨性框架的数据流呈现为从上游到下游的依赖结构，同时受到规范性维度的交叉调节：

**Epistemic Rigor（认识论严谨性）** → 输入：学科背景知识、先前文献、领域假设；输出：经审慎评估的问题选择与基础假设。这是整个链条的起点，决定"我们为何研究此问题"以及"我们以何种知识状态出发"。

**Conceptual Rigor（概念严谨性）** → 输入：理论构念、操作化选择；输出：明确定义的可测量变量与可检验假设。它承接认识论输出，将抽象问题转化为可研究的形式，但本身不承诺任何特定方法。

**Normative Rigor（规范性严谨性）** → 输入：价值观、伦理考量、利益相关者利益；输出：显式陈述且经辩护的规范性承诺。此维度不处于线性链中，而是以交叉方式渗透并调节所有其他五个维度。

**Methodological Rigor（方法论严谨性）** → 输入：研究问题、构念、可用方法；输出：有效且可靠的研究发现。这是传统严谨性的全部内容，在框架中仅作为核心环节之一，其质量受上游认识论与概念性选择的约束。

**Reporting Rigor（报告严谨性）** → 输入：研究发现、数据、分析结果；输出：完整、准确、透明的结果沟通。它决定外部世界能从研究中获取什么信息，直接影响下游推断的可能性。

**Interpretative Rigor（解释严谨性）** → 输入：研究发现、局限性、情境语境；输出：有根据的、范围适当、避免过度泛化的声明。这是链条终点，其失败（如从有限实验推断"通用智能"）是AI领域最受诟病的现象之一。

```
[Epistemic] ──→ [Conceptual] ──→ [Methodological] ──→ [Reporting] ──→ [Interpretative]
      ↑___________↑_________________↑____________________↑_________________↑
                    [Normative]（交叉调节所有维度）
```

## 核心模块与公式推导

本文为一篇概念性/立场论文（position paper），不涉及数学公式、损失函数或优化目标的推导。以下以命题式逻辑呈现三个核心模块的理论结构：

### 模块 1: Epistemic Rigor（认识论严谨性）（对应框架图上游位置）

**直觉**: 研究者在动手之前，必须先审视自己站在谁的肩膀上——背景知识是否可靠、是否被恰当引用、是否适用于当前问题。

**Baseline（传统AI研究）**: 隐含假设为 $K_{bg} \Rightarrow Q_{valid}$，即背景知识 $K_{bg}$ 自动保证问题选择 $Q$ 的有效性，无需显式审查。

**变化点**: 传统做法将背景知识视为黑箱；本文要求显式评估 $K_{bg}$ 的来源、时效性、领域适用性与潜在偏见。

**本文命题结构**:
$$\text{Step 1}: \quad K_{bg} \text{xrightarrow}{\text{scrutinize}} K_{bg}^{*} \quad \text{（对背景知识进行批判性审查）}$$
$$\text{Step 2}: \quad K_{bg}^{*} \wedge C_{domain} \text{vdash} Q_{justified} \quad \text{（结合领域条件推出经辩护的问题选择）}$$
$$\text{Step 3}: \quad \neg(K_{bg}^{*} \wedge C_{domain}) \Rightarrow \neg Q_{valid} \quad \text{（不满足认识论条件则问题无效）}$$

**评估标准**: 背景知识是否清晰沟通、是否适当、是否有充分依据、是否被恰当应用（Table 1）。

### 模块 2: Conceptual Rigor（概念严谨性）（对应框架图上游→核心过渡位置）

**直觉**: 如果一个核心概念（如"公平性""理解""偏见"）未被明确定义，后续所有方法论精巧都建立在流沙之上。

**Baseline（传统AI研究）**: 操作化定义 $O$ 直接替代理论构念 $T$，即 $T \approx O$，忽略二者之间的概念差距。

**变化点**: 传统做法常将可测量代理（如F1分数、困惑度）与理论目标混为一谈；本文要求显式论证 $T \to O$ 的映射合理性。

**本文命题结构**:
$$\text{Step 1}: \quad T \text{xrightarrow}{\text{define}} T_{formal} \quad \text{（理论构念的形式化定义）}$$
$$\text{Step 2}: \quad T_{formal} \text{xrightarrow}{\text{operationalize}} O_{measurable} \quad \text{（操作化为可测量变量）}$$
$$\text{Step 3}: \quad \text{justify}(T_{formal} \sim O_{measurable}) \quad \text{（显式辩护映射的保真度）}$$

**关键符号**: $T$ = 理论构念（如"语言理解"）；$O$ = 操作化测量（如GLUE分数）；$\sim$ = 概念-操作映射关系。

### 模块 3: Interpretative Rigor（解释严谨性）（对应框架图下游位置）

**直觉**: 研究发现与公开声明之间的差距是AI夸大宣传的核心机制，必须系统性地约束推断范围。

**Baseline（传统AI研究）**: 从发现到声明的推理是直觉式的，$F_{findings} \Rightarrow C_{claims}$，缺乏显式边界标记。

**变化点**: 传统做法倾向于最大化声明的普遍性与影响；本文要求每个声明附带显式的适用范围限定与反事实条件。

**本文命题结构**:
$$\text{Step 1}: \quad F_{findings} \wedge L_{limitations} \wedge C_{context} \text{xrightarrow}{\text{warrant}} C_{qualified} \quad \text{（有限发现+局限性+语境推出限定声明）}$$
$$\text{Step 2}: \quad \forall C_{claims}: \quad \text{scope}(C) \subseteq \text{scope}(F) \quad \text{（声明范围不超过发现支持范围）}$$
$$\text{Step 3}: \quad \neg(\exists C_{overgeneralized}) \quad \text{（禁止过度泛化声明的存在）}$$

**对应消融**: 作者未提供定量消融，但指出去掉interpretative rigor的代价——"narrow methodological focus obscures upstream and downstream factors affecting research integrity"，即方法论严谨性的孤立追求会系统性遮蔽影响研究完整性的上下游因素。

## 实验与分析


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/7ff3d36d-4791-48e3-b05d-93eb7d964add/figures/Table_1.png)
*Table 1: Overview of the six facets of rigor in AI research and practice. For each facet, we highlight one object-level and one meta-level work example, as representative of the kinds of rigor that each facet mandates for AI work.*



本文作为概念性立场论文，不提供传统意义上的实验结果、benchmark分数或消融实验。其核心"证据"在于理论论证的完整性与框架对现有问题的解释力。Table 1 展示了六维度的系统对比：每个维度的"关注对象"（object of concern）与"评估标准"（evaluative criteria）被明确定义，并配以促进该维度严谨性的机制示例。例如，epistemic rigor要求审视"哪些背景知识被用于问题选择"，其评估标准包括该知识是否"清晰沟通、适当、有充分依据、恰当应用"；对应的促进机制包括文献综述的透明度与跨学科知识整合。



Figure 1 以嵌套方框与箭头可视化六维度的依赖结构：epistemic rigor 包含 conceptual rigor，后者包含 methodological rigor，形成上游到下游的层级；reporting rigor 与 interpretative rigor 作为方法论产出的下游处理；normative rigor 以贯穿所有方框的方式表示其交叉调节作用。实线箭头表示维度间的直接影响，反映了作者的核心论点——上游维度的失败会级联传播至下游。

**框架的局限性（作者自陈）**: 
- 缺乏实证验证：作者明确承认这是一个"position paper without empirical validation of the framework's effectiveness"
- 未提供具体可操作化指标：每个维度的评估标准以定性描述呈现，未转化为可量化的检查清单或评分 rubric
- 维度边界模糊："acknowledges difficulty in drawing clear boundaries between facets"，尤其是 normative rigor 与其他维度的交叉可能引发分类争议

**公平性检查**: 本文的比较基准是传统"methodological rigor only"观念，而非与其他已有多维度框架的直接对比。缺失的基线包括：对框架实际应用效果的案例研究、与其他领域（如医学研究、社会科学）严谨性框架的系统性比较、以及定量证据表明更广泛的严谨性概念能改善研究产出质量。作为NeurIPS poster，其贡献定位为理论框架与社区对话催化剂，而非经验性验证研究。

## 方法谱系与知识库定位

**方法谱系**: 负责任AI（Responsible AI）→ AI研究质量评估 → **Six-Facet Rigor Framework**

**直接基线与差异**:
- **Methodological rigor (traditional)**: 本文将其从唯一维度降格为六分之一，强调其受上游维度约束
- **Data Statements for NLP [17]**: 仅覆盖 reporting rigor 的一个子集（数据文档化），本文将其整合入更广泛的报告维度并建立与上游维度的联系
- **Values in ML Research [22]**: 聚焦 normative 分析但缺乏系统性框架，本文将其提升为跨维度调节机制
- **NeurIPS Broader Impact Statements [13]**: 行政要求的合规性实践，本文提供理论基础说明为何此类反思属于严谨性而非附加项

**改变的插槽**:
- `evaluation_framework`: 从单一方法论严谨性 → 六维度相互依赖框架
- `assessment_scope`: 从技术方法 → 完整研究流程（背景知识→认知推断）

**后续方向**:
1. **操作化与工具化**: 将六维度转化为可实际使用的审稿检查清单、研究设计模板或自动评估工具
2. **领域特化**: 在计算机视觉、强化学习、生成式AI等子领域验证框架的适用性与调整维度权重
3. **实证验证**: 通过历史案例分析或前瞻性实验，检验框架是否能有效预测研究的长期影响力与可重复性

**标签**: 文本 / 概念框架 / AI研究评估 / 负责任AI / 科学哲学 / 无训练

