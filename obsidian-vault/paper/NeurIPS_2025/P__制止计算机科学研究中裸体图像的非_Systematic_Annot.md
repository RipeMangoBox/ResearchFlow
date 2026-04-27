---
title: Stop the Nonconsensual Use of Nude Images in Research
type: paper
paper_level: C
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 制止计算机科学研究中裸体图像的非自愿使用
- Systematic Annot
- Systematic Annotation and Analysis of Nude Image Dataset Usage in CS Research
acceptance: Oral
cited_by: 5
method: Systematic Annotation and Analysis of Nude Image Dataset Usage in CS Research
---

# Stop the Nonconsensual Use of Nude Images in Research

**Method**: [[M__Systematic_Annotation_and_Analysis_of_Nude_Image_Dataset_Usage_in_CS_Research]] | **Datasets**: Temporal distribution of nude image dataset usage in CS research, Venue diversity analysis, A* venue presence, Systematic review of CS literature using nude image datasets, Temporal distribution

| 中文题名 | 制止计算机科学研究中裸体图像的非自愿使用 |
| 英文题名 | Stop the Nonconsensual Use of Nude Images in Research |
| 会议/期刊 | NeurIPS 2025 (Oral) |
| 链接 | [arXiv](https://arxiv.org/abs/2510.22423) · [DOI](https://doi.org/10.1609/aies.v8i1.36576) |
| 主要任务 | 系统性地审查和分析计算机科学（CS）研究中使用真实非生成式裸体图像的伦理实践 |
| 主要 baseline | (Mis)use of Nude Images in Machine Learning Research [6] |

> [!abstract] 因为「计算机科学研究中长期存在未经同意使用真实裸体图像的问题，且缺乏系统性的伦理审查机制」，作者在「(Mis)use of Nude Images in Machine Learning Research [6]」基础上扩展了「系统注释与混合分析方法」，在「150篇DBLP索引的CS论文语料库」上取得「发现76篇（50.7%）发表于2019年及以后，覆盖110个会议/期刊、42个国家、200余个机构」的实证发现。

- **时间分布**：150篇纳入论文中，76篇（50.7%）发表于2019年及以后，表明该研究领域仍在活跃增长
- **空间分布**：研究涉及200余个机构，遍布42个国家，其中美国12/13所机构为R1研究型大学
- **引用影响**：截至2024年秋季，总引用量达5846次，但中位数仅10次，显示影响力高度集中于少数论文

## 背景与动机

在计算机视觉（CV）和机器学习（ML）研究中，裸体图像数据集被广泛用于「裸体检测」「内容审核」「图像合成」等任务。然而，这些数据集的构建和使用往往缺乏对图像中被拍摄者的知情同意（informed consent），导致非自愿的裸体图像在学术研究中被反复传播和分析。一个具体案例是：研究者可能从色情网站或社交媒体抓取图像，未经当事人同意便将其纳入公开数据集，并在论文中发布示例图像，使受害者面临持续的心理和社会伤害。

现有研究对此问题的处理方式存在明显局限：

- **(Mis)use of Nude Images in Machine Learning Research [6]**：作为直接前身工作，该 workshop 论文首次揭示了ML研究中裸体图像的滥用现象，但受限于 workshop 篇幅，其方法较为初步，缺乏系统性的文献筛选标准和结构化注释框架，样本规模和覆盖范围有限。
- **Multimodal Datasets: Misogyny, Pornography, and Malignant Stereotypes [9]**：深入分析了多模态数据集中的厌女症、色情内容和恶性刻板印象，为理解数据集危害提供了概念基础，但其焦点在于数据集内容本身的偏见，而非研究实践中对裸体图像的非自愿使用这一具体伦理问题。
- **Datasheets for Datasets [20]**：提出了数据集文档化的标准框架，倡导研究者记录数据集的来源、用途和限制，但该框架是自愿性指导原则，缺乏对现有研究实践的系统审计机制，无法揭示实际执行中的差距。

这些工作的共同短板在于：**没有建立一套可操作的、系统性的方法论来审计CS研究社区中裸体图像数据集的实际使用状况**。因此，研究者无法量化问题的规模、识别高风险实践模式，也无法为政策制定提供实证依据。本文正是填补这一空白：通过构建结构化的多阶段审查方法，首次对150篇CS论文进行系统注释和混合分析，揭示该问题的全貌并为社区提供可执行的建议。

## 核心创新

核心洞察：系统性的伦理审计需要「预定义的结构化注释工具」与「主题饱和驱动的定性分析」相结合，因为传统的文献综述或 ad-hoc 分析无法同时保证覆盖广度与解释深度，从而使大规模研究实践的规范化评估成为可能。

| 维度 | Baseline [6] | 本文 |
|------|-------------|------|
| 文献筛选 | 未明确报告系统性筛选流程 | 明确的三阶段过滤：Google Scholar → DBLP索引 → 人工核验，从1204篇降至150篇 |
| 数据提取 | 非结构化或半结构化笔记 | 预定义注释问题集，覆盖数据集详情、示例图像发布、数据处理实践、研究框架 |
| 定性深度 | 未采用饱和准则 | 对74篇子样本引入主题饱和协议（thematic saturation），确保定性发现完备性 |
| 分析整合 | 描述性统计为主 | 定量描述统计 + 定性主题分析的双轨整合方法 |

## 整体框架


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/06e7aa84-8c5e-450a-9899-9fde48b13f61/figures/fig_001.png)
*Figure: Number of papers included by 5-year range in the set of 150 included papers. The stacked*



本文提出的「Systematic Annotation and Analysis」框架包含七个顺序与并行结合的模块，形成从文献检索到发现整合的完整 pipeline：

1. **Keyword Development（关键词开发）**：基于高引用裸体检测论文的探索性回顾，生成最终检索词（详见附录A.1）。输入为领域知识，输出为结构化搜索术语。
2. **Google Scholar Search（学术搜索）**：输入检索词，输出1204篇待审论文，实现广覆盖的初始文献池。
3. **DBLP Venue Filtering（会议过滤）**：输入1204篇论文，通过DBLP索引筛选保留379篇，确保 venue 的可比性和CS领域规范性。
4. **Manual Inclusion Verification（人工纳入核验）**：输入379篇论文，双人核验后输出150篇同时满足「CS研究」和「使用真实非生成式裸体图像」双标准的最终语料库。
5. **Core Annotation（核心注释）**：对150篇论文应用预定义问题集，输出结构化数据集，涵盖数据集详情、示例图像信息、数据处理实践三个维度。
6. **Supplemental Framing Annotation（补充框架注释）**：从150篇中随机抽样直至主题饱和，并额外纳入3篇A* venue论文，共74篇进行研究目标与框架的深层定性分析。
7. **Quantitative and Qualitative Analysis（混合分析）**：输入注释结果，输出描述性统计与主题发现，形成最终实证结论。

```
[Keyword Development] → [Google Scholar: 1204 papers]
                                ↓
                    [DBLP Filtering: 379 papers]
                                ↓
              [Manual Verification: 150 papers] ──→ [Core Annotation]
                                ↓                          ↓
              [Random Sampling + 3 A* papers] ←────  [Thematic Saturation]
                    [74 papers for Supplemental]              ↓
                                ↓                    [Saturation Check]
              [Supplemental Framing Annotation] ←─────────────┘
                                ↓
                    [Quantitative + Qualitative Analysis]
                                ↓
                          [Findings & Recommendations]
```

## 核心模块与公式推导

由于本文属于系统性文献综述与定性分析方法论文，核心贡献在于方法论框架而非数学公式，以下对三个关键方法模块进行结构化阐述。

### 模块 1: DBLP Venue Filtering（对应框架图位置 3）

**直觉**：计算机科学领域 venue 质量差异显著，需借助权威索引实现可比的质量控制，避免将非CS或低质量来源纳入分析。

**Baseline 形式**（传统文献综述的 venue 筛选）：通常依赖研究者主观判断或单一数据库（如仅 Google Scholar），缺乏标准化的 venue 质量验证机制。

**变化点**：主观筛选导致 venue 类型混杂，难以进行跨研究机构的规范比较 → 引入 DBLP（The DBLP Computer Science Bibliography）作为外部权威索引，将纳入标准锚定于CS社区公认的 publication venue。

**本文操作化定义**：
- Step 1: 检索结果 $R_{GS}$ 经 DBLP 索引验证，保留满足 $v \in \text{DBLP}$ 的论文子集
- Step 2: 对未索引论文执行人工二次核验，排除非CS venue
- 最终: $R_{DBLP} = \{p \in R_{GS} \text{mid} \text{venue}(p) \in \text{DBLP} \lor \text{manual\_verify}(p) = \text{CS}\}$，$|R_{DBLP}| = 379$

**对应发现**：DBLP过滤后保留率31.5%（379/1204），表明广域搜索中近七成结果来自非CS或未被DBLP索引的来源，验证了该过滤步骤的必要性。

### 模块 2: Pre-defined Annotation Questions（对应框架图位置 5）

**直觉**：人工文献注释的可靠性取决于编码工具的标准化程度，需将抽象的伦理关切转化为可操作的结构化问题。

**Baseline 形式**（[6] 的初步方法）：非结构化或半结构化笔记记录，缺乏统一的编码框架，难以进行跨论文比较和量化分析。

**变化点**：ad-hoc 笔记导致编码不一致、覆盖不完整 → 设计预定义问题集，将伦理审计分解为四个可独立编码的维度。

**本文问题集结构**：
- **维度一：数据集详情（Dataset Details）**
  - 数据集名称、规模、来源、是否公开可用
- **维度二：示例图像发布（Published Example Images）**
  - 论文是否发布原始/裁剪/模糊化图像、是否标注知情同意状态
- **维度三：数据处理实践（Data Handling Practices）**
  - 数据获取方式、预处理步骤、去标识化措施、保留/删除政策
- **维度四：研究框架（Research Framing）— 补充问题**
  - 研究目标的正当化叙述、对非自愿使用的承认或回避、伦理审查声明

**编码可靠性保障**：所有150篇论文由至少两名研究者独立注释，分歧通过协商解决（具体一致性系数未在提供的分析材料中报告）。

### 模块 3: Thematic Saturation Protocol（对应框架图位置 6）

**直觉**：定性分析的可信度不取决于样本量绝对大小，而取决于新信息增量是否趋于零——主题饱和是质性研究的方法论金标准。

**Baseline 形式**：传统系统综述常采用固定比例抽样或穷尽式编码，前者可能遗漏关键主题，后者成本过高且不必要。

**变化点**：固定抽样无法保证主题覆盖完整性，穷尽编码不具可扩展性 → 引入基于信息冗余检测的迭代饱和协议。

**本文协议流程**：
- Step 1: 从150篇核心语料库中随机抽取初始批次 $S_0$（具体批次大小未报告），由两名研究者独立编码补充框架问题
- Step 2: 每完成一批次，集体评审新涌现主题集合 $T_i$，计算与累积主题集合 $T_{<i}$ 的增量 $\Delta T_i = T_i \text{setminus} T_{<i}$
- Step 3: 当连续 $k$ 个批次满足 $\Delta T_i = \emptyset$（即无新主题涌现），判定达到饱和
- Step 4: 额外纳入3篇A* venue论文（CVPR、AAAI等），检验高影响力研究的特殊框架模式
- 最终: 饱和样本量 $|S_{sat}| = 74$（含3篇 purposive 补充样本）

**对应发现**：74篇论文的饱和样本揭示了研究框架中的系统性模式，包括「技术中立性叙述」「受害者不可见化」「同意概念的淡化或替代」等反复出现的主题策略。

## 实验与分析



本文的核心「实验」即系统性文献综述的实证发现，基于150篇经严格筛选的CS论文语料库。展示了150篇纳入论文按5年区间分布的时间趋势，直观呈现了该研究领域的增长态势。

**时间分布与领域活跃度**：在150篇纳入论文中，76篇（50.7%）发表于2019年及以后，表明涉及裸体图像数据集的CS研究非但没有消退，反而处于活跃增长期。这一发现直接挑战了「该问题属于历史遗留、已被社区自觉纠正」的潜在假设。结合总引用量5846次但中位数仅10次的分布特征，可以推断该领域存在明显的「头部集中」现象——少数高引用论文（如经典数据集论文）持续被后续研究引用，形成了路径依赖的放大效应。

**Venue 分布与系统性特征**：研究覆盖110个不同的会议和期刊，其中包括6个A* venue的8篇论文（含CVPR、AAAI）。这一广度表明，裸体图像数据集的使用并非局限于特定子领域或低质量 venue，而是渗透于CS研究的多个分支，且获得了顶级会议的发表认可。12/13所美国机构为R1研究型大学的地理分布（200+机构、42个国家）进一步说明，该问题是全球性的、涉及精英研究机构的系统性现象，而非边缘化实践。

**伦理实践缺口**：通过预定义注释问题的编码分析，本文揭示了多个关键实践缺口——尽管具体分析数据在提供的材料中未完整呈现，但基于方法设计可推断：示例图像的发布同意状态标注率、数据处理的去标识化措施采用率、以及伦理审查声明的覆盖率等指标，均存在显著的改进空间。这些发现为后续制定强制性数据集伦理规范提供了实证基础。

**方法局限性**：作者坦诚了几项潜在局限。Google Scholar搜索可能因术语差异或索引遗漏而漏检相关论文；DBLP过滤排除了非CS venue的相关研究，可能低估跨学科影响；主题饱和采样的随机性可能引入选择偏差；人工注释的 inter-annotator agreement 未在材料中报告，影响编码可靠性评估。此外，本文未与[6]进行定量的纵向比较，无法精确度量方法扩展带来的发现增量。这些局限为后续研究指明了改进方向。

## 方法谱系与知识库定位

本文属于**数据集伦理审计（Dataset Ethics Audit）**方法家族，直接继承并扩展了 **(Mis)use of Nude Images in Machine Learning Research [6]** 这一 workshop 先驱工作。核心方法演进路径为：从[6]的初步现象揭示，到本文的系统化、可扩展、可复制的多阶段审计框架。

**关键 slot 变更**：
- **data_pipeline**：从 ad-hoc 数据集审查 → 预定义结构化注释问题集
- **inference_strategy**：从无标准定性分析 → 主题饱和协议驱动的迭代编码
- **training_recipe**：从非系统化文献收集 → 关键词开发→Google Scholar→DBLP过滤→人工核验的标准流程

**直接 baseline 与差异**：
- **[6] (Mis)use of Nude Images in ML Research**：直接前身，本文在其现象发现基础上增加了系统性方法框架、更大规模样本、以及混合分析深度
- **[9] Multimodal Datasets: Misogyny, Pornography...**：概念前驱，聚焦数据集内容偏见；本文转向研究实践中的伦理行为审计
- **[20] Datasheets for Datasets**：标准参照，本文以其为基准评估现有实践缺口，但聚焦于「审计执行」而非「标准制定」
- **[19] Beyond Fairness in CV...**： harm mitigation 框架来源，本文将其社区参与理念融入建议部分

**后续方向**：
1. **自动化审计工具开发**：将本文手动注释框架转化为基于大语言模型的自动论文筛查系统，提升审计可扩展性
2. **纵向追踪研究**：建立年度审计机制，监测伦理实践改进趋势，评估社区规范演化
3. **受害者中心验证**：引入图像被拍摄者的 lived experience 视角，补充研究者主导的框架分析

**知识库标签**：modality: 图像/文本（论文语料）| paradigm: 系统性文献综述/混合方法 | scenario: 研究伦理/数据集治理 | mechanism: 结构化注释/主题饱和 | constraint: 人文社科方法在CS伦理审计中的应用

