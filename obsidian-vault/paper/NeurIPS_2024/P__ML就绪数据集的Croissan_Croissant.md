---
title: 'Croissant: A Metadata Format for ML-Ready Datasets'
type: paper
paper_level: A
venue: NeurIPS
year: 2024
paper_link: null
aliases:
- ML就绪数据集的Croissant元数据格式
- Croissant
- Croissant is a metadata format for
acceptance: Spotlight
cited_by: 77
code_url: https://github.com/mlcommons/croissant
method: Croissant
modalities:
- Text
- Image
- Audio
paradigm: N/A
followups:
- LLM基准测试的标准化文档框架_BenchmarkCards
---

# Croissant: A Metadata Format for ML-Ready Datasets

[Code](https://github.com/mlcommons/croissant)

**Topics**: [[T__Benchmark_-_Evaluation]] | **Method**: [[M__Croissant]] | **Datasets**: Human readability, BLEU score for metadata, Ecosystem adoption

> [!tip] 核心洞察
> Croissant is a metadata format for datasets that creates a shared representation across ML tools, frameworks, and platforms, making datasets more discoverable, portable, and interoperable without requiring changes to underlying data representation.

| 中文题名 | ML就绪数据集的Croissant元数据格式 |
| 英文题名 | Croissant: A Metadata Format for ML-Ready Datasets |
| 会议/期刊 | NeurIPS 2024 (Spotlight) |
| 链接 | [arXiv](https://arxiv.org/abs/2403.19546) · [Code](https://github.com/mlcommons/croissant) · [DOI](https://doi.org/10.1145/3650203.3663326) |
| 主要任务 | 数据集文档化 / 基准测试与评估 |
| 主要 baseline | 现有各类数据集格式（TFDS、HuggingFace Datasets、PyTorch DataLoader等）、DCAT、Schema.org Dataset vocabulary、Datasheets for Datasets、Data Cards |

> [!abstract] 因为「ML领域数据集格式繁杂、工具间缺乏互操作性、数据发现与组合困难」，作者在「Schema.org / DCAT」基础上扩展了「JSON-LD + ML专用词汇表的Croissant元数据格式」，在「人工可读性/可理解性/完整性/简洁性评估」上取得「高质量元数据描述能力」，并已获「数十万数据集」的生态系统采纳。

- 已获多个主流数据集仓库支持，覆盖数十万数据集
- 人工评估验证元数据具备可读性、可理解性、完整性与简洁性
- 框架无关设计：无需修改底层数据即可实现TensorFlow、PyTorch、JAX等框架的直接加载

## 背景与动机

在机器学习实践中，研究者每开始一个新项目时几乎都会重复同一套痛苦流程：找到感兴趣的数据集，阅读杂乱的文档，编写定制的解析代码处理特定格式，再将其转换为当前框架（TensorFlow、PyTorch或JAX）所需的数据加载器。例如，Kaggle上的CSV数据集、HuggingFace上的Parquet文件、TFDS的专有格式——每种都需要不同的加载逻辑，且元数据描述方式各不相同，导致数据集发现、理解与复用成本极高。

现有方法从三个方向试图缓解这一问题。TFDS（TensorFlow Datasets）提供了标准化的数据集集合，但其元数据和加载代码深度绑定TensorFlow生态，不具备跨框架能力。Datasheets for Datasets [6] 和 Data Cards [18] 从文档化角度提出了结构化的数据集描述框架，强调数据收集过程、潜在偏见等负责任AI信息，但这些主要是面向人类的自由格式文档，缺乏机器可读的统一表示。DCAT（Data Catalog Vocabulary）[8] 和 Schema.org [9] 提供了通用的数据目录描述标准，但面向的是广义的Web数据发现场景，未针对ML工作流中的特定需求（如训练/验证/测试划分、特征类型标注、多模态数据关联）进行扩展。

这些方案的核心局限在于：它们或绑定特定工具链（如TFDS），或停留在人类可读文档层面（如Datasheets），或缺乏ML领域语义（如Schema.org）。没有一个标准能够在不修改底层数据的前提下，为数据集提供一个统一的、机器可读的、跨框架的语义描述层。这正是Croissant试图填补的空白：一种非侵入式的元数据格式，让数据集"一次描述，处处可用"。


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/c7789142-4648-4261-9a42-820cc70a8630/figures/fig_001.png)
*Figure: The Croissant lifecycle and ecosystem.*



## 核心创新

核心洞察：ML数据集的本质痛点不在于数据存储格式本身，而在于缺乏一种标准化的"语义接口层"——因为JSON-LD的链接数据特性允许在不改变底层文件的前提下叠加结构化语义描述，从而使跨框架自动加载与数据集组合成为可能。

| 维度 | Baseline | 本文 |
|:---|:---|:---|
| 元数据载体 | 各平台自有schema或无结构化元数据 | 统一的JSON-LD + Croissant词汇表 |
| 与数据的关系 | 深度绑定（TFDS）或完全分离（自由文档） | 非侵入式：元数据与底层数据解耦 |
| 框架兼容性 | 工具专属（TFDS→TensorFlow, HuggingFace→PyTorch等） | 框架无关：同一描述支持多框架直接加载 |
| ML语义支持 | 通用数据目录（DCAT/Schema.org）或无 | 内置ML专用概念：RecordSet、Field、Split、数据类型标注 |
| 负责任AI | 独立文档（Datasheets/Data Cards） | 原生扩展Croissant-RAI：采集过程、标注方法、潜在偏见 |

与baseline的关键差异在于：Croissant不是又一个数据集仓库或加载库，而是一种"中间层协议"——它向下兼容任意数据存储格式，向上为ML框架提供统一的消费接口。

## 整体框架


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/c7789142-4648-4261-9a42-820cc70a8630/figures/fig_002.png)
*Figure: Users can easily inspect datasets (e.g., Fashion MNIST [24]) and use them in data*



Croissant的整体架构围绕三个核心模块构建，形成从原始数据到ML框架的完整流水线：

**模块1：元数据描述（metadata_description）**
- 输入：原始数据集文件（图像、文本、音频、表格等任意格式）及其属性信息
- 输出：符合Croissant词汇表的JSON-LD文档
- 作用：将数据集的结构（RecordSet）、字段（Field）、划分（Split）、来源（Source）等ML相关信息编码为机器可读的标准化描述

**模块2：框架加载器（framework_loader）**
- 输入：Croissant JSON-LD元数据文档
- 输出：各ML框架原生的数据集对象（TensorFlow的tf.data.Dataset、PyTorch的DataLoader、JAX兼容格式等）
- 作用：解析Croissant描述，自动处理格式转换、数据关联、类型映射，无需用户编写定制加载代码

**模块3：发现索引（discovery_index）**
- 输入：来自多个数据集的Croissant元数据
- 输出：可搜索的数据集目录
- 作用：支持跨平台的数据集检索、比较与组合，提升数据发现效率

数据流示意：
```
原始数据集文件 → [Croissant元数据描述] → JSON-LD文档 → [框架加载器] → TensorFlow/PyTorch/JAX数据集对象
                                    ↓
                              [发现索引] → 可搜索目录/跨数据集组合
```


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/c7789142-4648-4261-9a42-820cc70a8630/figures/fig_003.png)
*Figure: Dataset metadata and resources for the*



## 核心模块与公式推导

由于Croissant是一种基于词汇表/本体的元数据规范而非参数化模型，其核心"公式"体现为对数据集描述的结构化约束与扩展规则。以下解析两个最关键的模块：

### 模块1: Croissant核心词汇表（对应框架图 元数据描述层）

**直觉**: 将ML数据集的通用概念（记录、字段、划分）编码为可复用的RDF属性，使任何符合JSON-LD的解析器都能理解数据集结构。

**Baseline形式（Schema.org/DCAT）**:
```
Dataset := { @context, @type, name, description, distribution[], creator, license }
```
其中`distribution`仅描述文件级分发信息，缺乏ML语义。

**变化点**: Schema.org/DCAT的`Dataset`描述过于通用，无法表达"这个CSV的哪几列是特征、哪一列是标签、如何划分训练集/测试集"等ML核心需求。

**本文扩展（Croissant词汇表）**:
```
Step 1: 引入ML核心概念
  RecordSet := { @type: "cr:RecordSet", name, field[], dataType, source }
  Field := { @type: "cr:Field", name, dataType, source, references? }
  
Step 2: 建立数据关联机制
  source: { fileSet: FileObject, extract: { column: string } }
  references: { field: Field, extract: { fileProperty: "fullPath" } }
  
最终: Croissant Dataset := Schema.org/DCAT Dataset + cr:RecordSet[] + cr:Field[] + Split定义 + 数据类型标注
```
**关键符号**: `cr:` = Croissant命名空间；`RecordSet` = 逻辑记录集合（如"训练样本"）；`Field` = 记录中的字段；`source` = 到底层文件的映射规则；`references` = 跨资源关联（如图像路径→实际图像文件）。

**对应消融**: Table 4显示词汇表一致性评估结果，验证扩展后的属性定义无内部矛盾。

### 模块2: Croissant-RAI扩展（对应框架图 语义增强层）

**直觉**: 将负责任AI（Responsible AI）的文档需求从独立的人类可读报告转化为结构化、可验证的机器可读属性。

**Baseline形式（Datasheets for Datasets / Data Cards）**:
```
RAI_info := free_text_documentation  // 无固定结构，难以自动解析
```

**变化点**: 自由格式文档无法实现自动化合规检查、偏见检测工具集成，且不同数据集间的RAI信息难以比较。

**本文扩展（Croissant-RAI）**:
```
Step 1: 结构化RAI属性
  dataCollection := { collectionProcess, collectionTimeframe, collectionPlace }
  dataAnnotation := { annotatorType, annotationProcess, annotationInstructions }
  
Step 2: 偏见与公平性标注
  personalSensitiveInformation := boolean
  knownBiases := { biasType, description, mitigationStrategy? }
  
最终: Croissant-RAI := Croissant Dataset + structured RAI properties
```

**对应消融**: Table 3展示Croissant-RAI属性标注表，定义了负责任AI相关的元数据属性设计，使数据集的伦理属性可被程序化访问。

## 实验与分析



本文的评估聚焦于元数据质量而非传统ML指标，采用人工评估与自动化度量相结合的方式。在人工评估方面，研究者设计了针对可读性（readability）、可理解性（understandability）、完整性（completeness）和简洁性（conciseness）四个维度的评分标准，由人工评分员对Croissant元数据进行系统性评估。Table 1展示了这一评估的详细评分标准，包含各维度对应的具体问题与评分量表。评估结果表明Croissant元数据在这四个维度上均达到高质量水平，验证了词汇表设计的合理性。



在自动化度量方面，Table 5呈现了BLEU分数评估结果，用于衡量数据集和属性（包括描述、许可证、URL、创建者等字段）的元数据生成质量。尽管具体BLEU数值在提供的上下文中未完整给出，但该指标旨在验证Croissant描述与人工编写描述之间的语义一致性。



生态系统采纳方面，Croissant已获得多个主流数据集仓库的支持，覆盖数十万数据集。这一广泛采纳构成了对其设计实用性的最强验证——不同于传统学术论文的受控实验，元数据标准的价值最终体现在社区的实际采用程度上。

**公平性审视**: 本文的评估存在若干局限。首先，缺乏与直接baseline的定量对比：未与TFDS元数据格式、HuggingFace Datasets元数据或Data Packages格式进行功能等价性比较；其次，人工评估的样本量与方法论细节未充分披露；第三，未进行用户研究来量化实际生产力收益（如有无Croissant时的数据集加载时间对比）；最后，BLEU分数缺乏性能基准参考，难以判断其绝对水平。 adoption数据虽令人印象深刻，但属于相关性证据而非因果性证明——广泛采用可能受益于MLCommons的推动力度，而非 solely 格式本身的技术优越性。

## 方法谱系与知识库定位

Croissant属于**数据集文档化与互操作性**方法家族，其直接技术谱系可追溯至：

- **父方法：Schema.org Dataset vocabulary [9] / DCAT [8]** — Croissant直接扩展这两个W3C标准作为基础，新增ML专用词汇与框架无关加载机制
- **近亲：Datasheets for Datasets [6]** — Croissant将其自由格式文档理念转化为机器可读的JSON-LD结构化描述
- **近亲：Data Cards [18]** — Croissant继承其负责任AI文档思想，并形式化为Croissant-RAI扩展

**改变的插槽**: 
- architecture: 新增统一的JSON-LD元数据层，解耦于底层存储
- data_curation: 新增标准化的ML语义描述（RecordSet/Field/Split）
- training_recipe: N/A（非训练方法）
- data_pipeline: 替换各框架定制加载代码为基于元数据的自动加载

**直接baseline对比**:
- vs. TFDS: Croissant框架无关，TFDS深度绑定TensorFlow
- vs. HuggingFace Datasets: Croissant不托管数据仅描述数据，更具可移植性
- vs. Data Packages [11]: Croissant专门针对ML工作流优化，而非通用数据打包

**后续方向**:
1. 扩展复杂数据操作表达能力（如高级join、实时数据流、变换pipeline的声明式描述）
2. 与ML实验追踪工具（如MLflow、Weights & Biases）的深度集成，实现"数据集-模型-实验"全链路元数据
3. 自动化元数据生成：从原始数据自动推断Croissant描述，降低采纳门槛

**知识库标签**: 模态=[text, image, audio, tabular] | 范式=数据标准/元数据规范 | 场景=数据集发现、加载、共享 | 机制=JSON-LD词汇扩展、语义网 | 约束=非侵入式、框架无关、向后兼容

## 引用网络

### 后续工作（建立在本文之上）

- [[P__LLM基准测试的标准化文档框架_BenchmarkCards]]: Croissant is a closely related metadata format for datasets; BenchmarkCards like

