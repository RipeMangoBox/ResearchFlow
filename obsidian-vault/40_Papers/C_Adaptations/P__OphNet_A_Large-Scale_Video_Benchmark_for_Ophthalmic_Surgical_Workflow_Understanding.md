---
title: 'OphNet: A Large-Scale Video Benchmark for Ophthalmic Surgical Workflow Understanding'
type: paper
paper_level: C
venue: ECCV
year: 2024
acceptance: accepted
cited_by: 14
core_operator: OphNet 的核心直觉是：眼科手术工作流理解的瓶颈不在于算法，而在于缺乏领域专属的大规模标注数据。通过构建三层层次化标注体系（类型→阶段→步骤），OphNet 将手术工作流的语义结构显式编码进数据集设计中，使模型能够在多粒度上学习手术知识。其有效性的根本逻辑是：高质量、大规模、细粒度的领域数据是推动专科医疗AI进步的必要基础设施，而非可以通过算法技巧绕过的障碍。
paper_link: https://www.semanticscholar.org/paper/80ae23fba0fd41a42934107664d63e8a63362f61
structurality_score: 0.15
---

# OphNet: A Large-Scale Video Benchmark for Ophthalmic Surgical Workflow Understanding

## Links

- Mechanism: [[C__surgical_workflow_benchmark_construction]]

> OphNet 的核心直觉是：眼科手术工作流理解的瓶颈不在于算法，而在于缺乏领域专属的大规模标注数据。通过构建三层层次化标注体系（类型→阶段→步骤），OphNet 将手术工作流的语义结构显式编码进数据集设计中，使模型能够在多粒度上学习手术知识。其有效性的根本逻辑是：高质量、大规模、细粒度的领域数据是推动专科医疗AI进步的必要基础设施，而非可以通过算法技巧绕过的障碍。

> **适配/插件型**。可快速浏览，看改了哪个 slot 和效果。

## 核心公式

$$
\text{mAP} = \frac{1}{|C|} \sum_{c \in C} AP_c
$$

> 用于衡量多类别手术阶段与步骤识别的平均精度，是基准评测的核心指标。
> *Slot*: surgical phase/step recognition evaluation

$$
\text{IoU} = \frac{|P \cap G|}{|P \cup G|}
$$

> 时序动作定位任务中用于衡量预测片段与真实片段重叠程度的标准指标。
> *Slot*: temporal action localization evaluation

## 关键图表

**Table 1**
: Comparison of OphNet with existing surgical workflow datasets across scale, annotation granularity, and task coverage
> 证据支持: OphNet 是目前规模最大、标注最细粒度的眼科手术工作流视频基准，覆盖手术类型、阶段、步骤等多层次标注。

**Table 2**
: Statistics of OphNet dataset including number of videos, phases, steps, and surgical types
> 证据支持: OphNet 包含66种眼科手术类型、102个手术阶段、150个手术步骤，共14,000余个视频片段，支撑其大规模声称。

**Table 3 / Table 4**
: Benchmark results of multiple baseline models on surgical phase recognition and step recognition tasks
> 证据支持: 现有主流时序理解模型在OphNet上的性能普遍偏低，验证了该基准的挑战性。

**Figure 2**
: Hierarchical annotation structure of OphNet showing surgical type → phase → step taxonomy
> 证据支持: OphNet 采用三层层次化标注体系（手术类型→阶段→步骤），是其区别于现有数据集的核心设计。

## 详细分析

# OphNet: A Large-Scale Video Benchmark for Ophthalmic Surgical Workflow Understanding

## Part I：问题与挑战

眼科手术工作流理解是计算机辅助手术（CAS）领域的重要研究方向，但长期面临数据资源匮乏的核心瓶颈。现有手术视频数据集（如 Cholec80、CholecT50）主要聚焦于腹腔镜胆囊切除术等普通外科手术，眼科手术领域几乎没有可用的大规模标注视频基准。眼科手术具有高度独特的视觉特征：显微镜拍摄视角、器械极为细小、操作精度要求极高、手术类型繁多（白内障、青光眼、视网膜手术等），这使得通用手术数据集上训练的模型无法直接迁移。此外，现有少数眼科手术数据集规模极小、标注粒度单一（仅有阶段级别标注），无法支撑细粒度工作流理解任务。研究者面临的挑战是：缺乏一个覆盖多种眼科手术类型、具备层次化细粒度标注、支持多类下游任务评测的标准化基准，导致该领域算法研究进展缓慢、不同方法之间缺乏可比性。OphNet 正是为填补这一空白而构建的。

## Part II：方法与洞察

OphNet 的核心贡献是构建了一个大规模、多层次、多任务的眼科手术工作流视频基准数据集，而非提出新的算法模型。其方法论创新体现在以下几个维度：

**1. 数据规模扩展**：OphNet 收集了超过14,000个视频片段，涵盖66种眼科手术类型，是目前规模最大的眼科手术工作流视频基准。相比现有数据集，规模提升幅度显著（见 Table 1 对比）。

**2. 三层层次化标注体系**：这是 OphNet 最核心的设计决策。标注体系定义为「手术类型 → 手术阶段 → 手术步骤」三个层次，共涵盖66种手术类型、102个手术阶段、150个手术步骤。这种层次化设计允许模型在不同粒度上学习手术工作流的语义结构，而非仅依赖单一粒度标签。标注过程由经验丰富的眼科医生参与审核，以保证医学准确性和标注一致性。

**3. 多任务评测框架**：OphNet 同时支持三类下游任务：（a）手术阶段识别（surgical phase recognition），使用 mAP 作为核心评测指标；（b）手术步骤识别（surgical step recognition），同样采用 mAP 评测；（c）时序动作定位（temporal action localization），采用 IoU 相关指标评测。这种多任务设计使 OphNet 能够作为统一基准支撑不同研究方向。

**4. 基线模型评测**：论文在 OphNet 上系统评测了多个主流视频理解模型（VideoSwin、TimeSformer、MS-TCN 等），建立了标准化的评测协议和性能基线，为后续研究提供可比较的参考点。

**5. 领域迁移分析**：论文还进行了从通用手术数据集（如 Cholec80）到 OphNet 的迁移实验，定量验证了眼科手术视频的领域特殊性，为领域适应研究提供了实验依据。

总体而言，OphNet 的「方法」本质上是一套数据工程与基准设计方案，其价值在于为领域提供标准化的评测平台，而非算法层面的创新。

### 核心直觉

OphNet 的核心直觉是：眼科手术工作流理解的瓶颈不在于算法，而在于缺乏领域专属的大规模标注数据。通过构建三层层次化标注体系（类型→阶段→步骤），OphNet 将手术工作流的语义结构显式编码进数据集设计中，使模型能够在多粒度上学习手术知识。其有效性的根本逻辑是：高质量、大规模、细粒度的领域数据是推动专科医疗AI进步的必要基础设施，而非可以通过算法技巧绕过的障碍。

## Part III：证据与局限

核心证据来自三个层面：（1）数据集规模对比（Table 1）直接支撑「最大规模」声称，OphNet 在视频数量、手术类型数、标注层次数等维度均显著超越现有数据集；（2）基线实验结果（Table 3/4）显示主流模型在手术阶段识别任务上 mAP 普遍低于50%，支撑「高挑战性」声称；（3）迁移实验（Section 4.3）定量验证了眼科手术视频的领域特殊性。

主要局限性：首先，「最大规模」声称依赖于与特定数据集集合的横向比较，若存在未被纳入比较的更大数据集则结论可能动摇；其次，基线性能偏低部分可能源于领域迁移问题（通用预训练模型未针对医疗视频优化），而非任务本身难度，两者难以完全解耦；第三，标注质量声称缺乏具体量化的一致性指标（如 Cohen's Kappa），主要依赖文字描述；第四，手术工作流专用方法（如 TeCNO、OperA）是否全部纳入基线比较尚不明确，可能存在强基线缺失问题；第五，OphNet 仅覆盖眼科手术，其结论不能直接推广至其他外科领域。
