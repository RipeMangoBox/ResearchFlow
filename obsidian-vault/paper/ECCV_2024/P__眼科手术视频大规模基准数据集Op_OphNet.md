---
title: 'OphNet: A Large-Scale Video Benchmark for Ophthalmic Surgical Workflow Understanding'
type: paper
paper_level: C
venue: ECCV
year: 2024
paper_link: https://www.semanticscholar.org/paper/80ae23fba0fd41a42934107664d63e8a63362f61
aliases:
- 眼科手术视频大规模基准数据集OphNet
- OphNet
- OphNet 的核心直觉是：眼科手术工作流理解的瓶颈不在于算法
acceptance: accepted
cited_by: 14
method: OphNet
---

# OphNet: A Large-Scale Video Benchmark for Ophthalmic Surgical Workflow Understanding

[Paper](https://www.semanticscholar.org/paper/80ae23fba0fd41a42934107664d63e8a63362f61)

**Topics**: [[T__Benchmark_-_Evaluation]], [[T__Video_Understanding]] | **Method**: [[M__OphNet]]

> [!tip] 核心洞察
> OphNet 的核心直觉是：眼科手术工作流理解的瓶颈不在于算法，而在于缺乏领域专属的大规模标注数据。通过构建三层层次化标注体系（类型→阶段→步骤），OphNet 将手术工作流的语义结构显式编码进数据集设计中，使模型能够在多粒度上学习手术知识。其有效性的根本逻辑是：高质量、大规模、细粒度的领域数据是推动专科医疗AI进步的必要基础设施，而非可以通过算法技巧绕过的障碍。

| 中文题名 | 眼科手术视频大规模基准数据集OphNet |
| 英文题名 | OphNet: A Large-Scale Video Benchmark for Ophthalmic Surgical Workflow Understanding |
| 会议/期刊 | ECCV 2024 (accepted) |
| 链接 | [Semantic Scholar](https://www.semanticscholar.org/paper/80ae23fba0fd41a42934107664d63e8a63362f61) · Code & Project:  |
| 主要任务 | 手术阶段识别 (surgical phase recognition)、手术步骤识别 (surgical step recognition)、时序动作定位 (temporal action localization) |
| 主要 baseline | VideoSwin、TimeSformer、MS-TCN、TeCNO、OperA |

> [!abstract] 因为「眼科手术工作流理解领域缺乏大规模标注视频基准，现有通用手术数据集无法迁移」，作者「构建了三层层次化标注体系」，在「OphNet 基准」上取得「14,000+视频片段、66种手术类型、150个手术步骤的多任务评测框架」

- **规模**: 14,000+ 视频片段，66 种眼科手术类型，超越现有所有眼科手术数据集
- **标注粒度**: 三层层次化标注（类型→阶段→步骤），共 102 个阶段、150 个步骤
- **基线性能**: 主流模型在手术阶段识别任务上 mAP 普遍低于 50%，验证任务高挑战性

## 背景与动机

眼科手术工作流理解是计算机辅助手术（Computer-Assisted Surgery, CAS）的核心研究方向，旨在让机器自动识别手术视频中的关键阶段与精细操作，从而辅助医生培训、实时导航与术后评估。然而，该领域长期面临一个根本性瓶颈：数据荒漠。以白内障手术为例，医生在显微镜下使用极细器械进行毫米级操作，视频具有独特的视觉特征——高倍放大视角、器械微小、光照变化剧烈、组织形变微妙。这些特征使得在腹腔镜胆囊切除术视频（如 Cholec80、CholecT50）上训练的模型完全失效。

现有方法如何应对这一困境？**Cholec80** 等通用手术数据集提供了阶段级别标注，但仅覆盖普通外科，且手术类型单一；**CholecT50** 扩展了工具识别与细粒度标注，然而其视觉域与眼科显微镜视频差异巨大；少数眼科专用数据集（如特定白内障数据集）规模极小，通常仅有数十个视频，且标注粒度单一（仅有阶段级别），无法支撑步骤级别的精细理解。这些方法的共同局限在于：数据规模不足、领域不匹配、标注层次缺失，导致模型无法学习眼科手术特有的多粒度语义结构。

更深层的问题是，由于缺乏标准化基准，研究者各自使用私有数据或极小规模数据集，方法之间缺乏可比性，领域进展缓慢。OphNet 正是为填补这一空白而构建：通过系统性的数据工程与标注设计，为眼科手术工作流理解提供一个大规模、多层次、多任务的公共评测平台。

## 核心创新

核心洞察：眼科手术工作流理解的瓶颈不在于算法创新，而在于缺乏领域专属的大规模层次化标注数据，因为眼科手术的视觉特征（显微镜视角、微器械操作）与通用外科差异极大，从而使「将手术语义结构显式编码进数据设计」成为可能。

| 维度 | Baseline（现有手术数据集） | 本文（OphNet） |
|:---|:---|:---|
| 数据规模 | 数十至数百视频，单一/少数手术类型 | 14,000+ 视频片段，66 种眼科手术类型 |
| 标注粒度 | 单一层级（仅阶段或仅工具） | 三层层次化：手术类型 → 手术阶段（102个） → 手术步骤（150个） |
| 任务支持 | 单一任务（如阶段识别） | 多任务统一评测：阶段识别、步骤识别、时序动作定位 |
| 领域覆盖 | 普通外科（腹腔镜等） | 眼科专属（白内障、青光眼、视网膜等） |
| 评测协议 | 缺乏标准化，跨研究不可比 | 系统化基线评测（VideoSwin/TimeSformer/MS-TCN），建立可比参考点 |

## 整体框架



OphNet 的整体框架是一个「数据构建 → 层次化标注 → 多任务评测」的完整基准工程流水线，而非传统意义上的模型架构。数据流如下：

**原始视频采集** → **视频预处理与片段化** → **三层层次化标注** → **质量审核与一致性校验** → **标准化数据划分** → **多任务基准评测**

各核心模块的功能定义：

- **原始视频采集模块**：输入为多家医疗机构的眼科手术录像，输出为原始长视频流。覆盖白内障、青光眼、视网膜手术等 66 种手术类型，确保领域多样性。

- **视频预处理与片段化模块**：输入为原始长视频，输出为 14,000+ 标准化视频片段。执行去标识化、分辨率统一、时长短片切分等操作，适配深度学习模型的输入要求。

- **三层层次化标注模块**（核心创新）：输入为视频片段，输出为三层结构化标签。第一层「手术类型」标注 66 种手术类别；第二层「手术阶段」细分出 102 个语义阶段（如切口制作、晶状体乳化等）；第三层「手术步骤」进一步细化为 150 个原子操作步骤。标注由经验丰富的眼科医生执行并审核。

- **质量审核模块**：输入为初步标注结果，输出为医学准确性校验后的最终标注。依赖专家交叉审核机制，（具体一致性量化指标如 Cohen's Kappa 。

- **标准化评测协议模块**：输入为标注完成的数据集，输出为三类任务的训练/验证/测试划分与评测脚本。定义手术阶段识别（mAP）、手术步骤识别（mAP）、时序动作定位（IoU-based metrics）的标准化评测流程。

- **基线评测模块**：输入为 OphNet 数据，输出为多个主流视频理解模型的性能基准。覆盖 2D-CNN、3D-CNN、Transformer、时序建模等多类架构。

```
[原始手术录像] ──→ [预处理/片段化] ──→ [类型级标注: 66类]
                                              ↓
[专家审核] ←──── [步骤级标注: 150步] ←── [阶段级标注: 102阶段]
                                              ↓
                                    [标准化数据划分]
                                              ↓
              ┌─────────────────┬─────────────┴─────────────┐
              ↓                 ↓                           ↓
        [阶段识别]         [步骤识别]                  [时序动作定位]
        (mAP评测)          (mAP评测)                   (IoU评测)
              ↓                 ↓                           ↓
        [VideoSwin]      [TimeSformer]                [MS-TCN等]
        [基线结果]         [基线结果]                   [基线结果]
```

## 核心模块与公式推导

OphNet 作为基准数据集论文，其核心「模块」体现为数据工程设计与评测协议定义，而非传统可训练模块。以下解析三个最关键的设计决策及其形式化定义：

### 模块 1: 三层层次化标注体系（对应框架图 核心设计层）

**直觉**: 手术工作流具有固有的层次化语义结构，单一层级标注会损失不同粒度间的依赖关系，模型难以学习从宏观类型到微观步骤的推理链条。

**Baseline 形式**（现有数据集，如 Cholec80 的单层标注）：
$$\mathcal{D}_{\text{base}} = \{(v_i, y_i)\}_{i=1}^{N}, \quad y_i \in \mathcal{C}_{\text{phase}}$$
其中 $v_i$ 为视频片段，$y_i$ 为单一阶段标签，$\mathcal{C}_{\text{phase}}$ 为阶段类别集合（Cholec80 中 $|\mathcal{C}_{\text{phase}}|=7$）。

**变化点**: 单层标注无法表达「手术类型决定合法阶段集合，阶段决定合法步骤集合」的层次约束，导致模型在语义不一致的标签空间上学习。

**本文形式化定义**:
$$\mathcal{D}_{\text{OphNet}} = \{(v_i, y_i^{\text{type}}, y_i^{\text{phase}}, y_i^{\text{step}})\}_{i=1}^{N}$$
$$y_i^{\text{type}} \in \mathcal{T}, \quad |\mathcal{T}| = 66$$
$$y_i^{\text{phase}} \in \mathcal{P}(y_i^{\text{type}}), \quad |\mathcal{P}| = 102 \text{（全局）}$$
$$y_i^{\text{step}} \in \mathcal{S}(y_i^{\text{phase}}), \quad |\mathcal{S}| = 150 \text{（全局）}$$
其中 $\mathcal{P}(t)$ 和 $\mathcal{S}(p)$ 为类型 $t$ 和阶段 $p$ 对应的合法子集，形成树状层次约束：$\mathcal{T} \text{succ} \mathcal{P} \text{succ} \mathcal{S}$。

**对应消融**: Table 1 显示 OphNet 在视频数量（14,000+）、手术类型数（66）、标注层次数（3层）均显著超越现有数据集（Cholec80: 80视频/1类型/1层；CholecT50: 50视频/1类型/2层）。

### 模块 2: 多任务统一评测协议（对应框架图 评测层）

**直觉**: 不同任务需要适配的评测指标与数据划分，缺乏统一协议会导致结果不可比。

**Baseline 形式**（分散的评测实践）：
$$\text{Task}_A: \text{Acc}@1 = \frac{1}{N}\sum_i \mathbb{1}[\hat{y}_i = y_i]$$
$$\text{Task}_B: \text{random split per paper}$$
各研究使用不同指标与划分，无法横向比较。

**变化点**: 任务定义不一致、划分方式不固定、指标选择不统一。

**本文形式化定义**:

**手术阶段/步骤识别任务**（帧级分类，mAP 评测）：
$$\text{mAP} = \frac{1}{|\mathcal{C}|}\sum_{c \in \mathcal{C}} \text{AP}_c, \quad \text{AP}_c = \int_0^1 p_c(r) \, dr$$
其中 $p_c(r)$ 为类别 $c$ 的 precision-recall 曲线，$\mathcal{C} \in \{\mathcal{P}, \mathcal{S}\}$。

**时序动作定位任务**（片段检测，IoU 评测）：
$$\text{mAP}_{\text{TAL}} = \frac{1}{|\mathcal{C}|}\sum_{c}\frac{1}{|\mathcal{I}|}\sum_{\tau \in \mathcal{I}} \text{AP}_{c,\tau}$$
其中 $\mathcal{I} = \{0.5, 0.75, 0.95\}$ 为 IoU 阈值集合，动作提议 $(t_s, t_e, c)$ 与真值的时序 IoU 定义为：
$$\text{IoU} = \frac{|\hat{\mathcal{T}} \cap \mathcal{T}^*|}{|\hat{\mathcal{T}} \cup \mathcal{T}^*|}, \quad \hat{\mathcal{T}} = [\hat{t}_s, \hat{t}_e]$$

**对应消融**: Table 3/4 显示在统一协议下，VideoSwin 阶段识别 mAP 为（具体数值待补充），TimeSformer 为（具体数值待补充），建立可比基线。

### 模块 3: 领域迁移分析框架（对应框架图 分析层）

**直觉**: 验证眼科手术视频的领域特殊性，为领域适应研究提供定量依据。

**Baseline 形式**（直接迁移，无适应）：
$$\theta^* = \text{arg}\min_\theta \mathcal{L}(f_\theta; \mathcal{D}_{\text{source}}), \quad \text{eval on } \mathcal{D}_{\text{target}}$$

**变化点**: 源域（Cholec80 等通用外科）与目标域（OphNet 眼科）视觉分布差异大，直接迁移性能崩溃。

**本文形式化定义**: 定义迁移增益/损失指标
$$\Delta_{\text{transfer}} = \text{mAP}(f_{\text{pretrain}}; \mathcal{D}_{\text{OphNet}}) - \text{mAP}(f_{\text{scratch}}; \mathcal{D}_{\text{OphNet}})$$
$$\Delta_{\text{domain\_gap}} = \text{mAP}(f_{\text{source-only}}; \mathcal{D}_{\text{OphNet}}) - \text{mAP}(f_{\text{target-only}}; \mathcal{D}_{\text{OphNet}})$$

实验显示（Section 4.3）从 Cholec80 预训练迁移至 OphNet 的 $\Delta_{\text{domain\_gap}}$ 为显著负值，定量验证领域特殊性。

## 实验与分析



### 主实验结果

| Method | 手术阶段识别 mAP | 手术步骤识别 mAP | 时序动作定位 mAP@0.5 |
|:---|:---|:---|:---|
| 2D-CNN baseline |  |  |  |
| VideoSwin |  |  |  |
| TimeSformer |  |  |  |
| MS-TCN |  |  | — |
| 最优结果 | < 50%（阶段识别） |  |  |

**核心发现分析**：

1. **高挑战性验证**：主流视频理解模型在手术阶段识别任务上 mAP 普遍低于 50%（Table 3/4），即使强如 VideoSwin、TimeSformer 等 Transformer 架构也未能突破此瓶颈。这一结果支撑了 OphNet 作为「高难度基准」的定位，说明眼科手术视频的细粒度理解远超当前模型能力。

2. **架构对比**：时序建模方法（MS-TCN）在帧级分类任务上表现优于纯空间方法，但引入时空 Transformer（VideoSwin、TimeSformer）并未带来预期的大幅提升，暗示眼科手术视频中**长程时序依赖的建模仍有瓶颈**，或预训练域差异削弱了迁移效果。



3. **领域迁移实验**（关键支撑证据）：从 Cholec80 预训练迁移至 OphNet 的性能显著低于直接在 OphNet 上训练，定量验证了眼科手术视频的领域特殊性。这一结果具有双重意义：一方面说明通用手术数据集不足以支撑眼科应用；另一方面为未来的**领域适应（domain adaptation）**与**自监督预训练**研究指明了方向。

### 公平性检查与局限

- **基线充分性**：论文评测了 VideoSwin、TimeSformer、MS-TCN 等主流架构，但专用手术工作流方法（如 TeCNO、OperA）是否全部纳入比较尚不明确，可能存在强基线缺失。
- **计算/数据成本**：14,000+ 视频片段的标注成本极高，依赖专业眼科医生参与，复现或扩展难度较大。
- **失败案例**：低 mAP 可能部分源于通用预训练模型未针对医疗视频优化（领域迁移问题），而非任务本身绝对难度，两者难以完全解耦。
- **标注质量透明度**：缺乏具体量化的一致性指标（如 Cohen's Kappa），主要依赖文字描述的专家审核流程。

## 方法谱系与知识库定位

**方法家族**：医疗视频理解基准数据集 / 手术工作流分析数据基础设施

**父方法/直接先驱**：
- **Cholec80**（2016）：首个大规模手术阶段识别基准，7 阶段单任务，奠定手术工作流理解的数据范式。OphNet 继承其「阶段识别」任务定义，但扩展至三层粒度与眼科领域。
- **CholecT50**（2020）：引入工具识别与多任务，50 视频。OphNet 继承其「多任务评测」思想，但规模扩展 280 倍且新增步骤级标注。
- **HeiChole / AutoLaparo** 等：其他专科手术数据集。OphNet 与其平行，但首次覆盖眼科领域。

**直接 Baseline 差异**：
| Baseline | 与 OphNet 的核心差异 |
|:---|:---|
| Cholec80 / CholecT50 | 普通外科（腹腔镜）vs 眼科（显微镜）；单层/双层标注 vs 三层层次化；80/50视频 vs 14,000+ |
| 特定白内障小数据集 | 单一手术类型、数十视频、无标准化评测 vs 66类型大规模多任务基准 |
| 通用视频数据集（Kinetics等） | 无手术语义标签，无法直接用于工作流理解评测 |

**后续方向**：
1. **眼科专用预训练模型**：基于 OphNet 大规模数据，开发自监督预训练策略，缓解领域迁移瓶颈；
2. **层次化模型架构**：显式利用类型→阶段→步骤的层次约束设计结构化预测模型；
3. **跨专科扩展**：将 OphNet 的数据工程范式推广至神经外科、骨科等其他显微手术领域。

**知识库标签**：
- **模态 (modality)**：视频（手术显微镜录像）
- **范式 (paradigm)**：基准数据集构建 / 数据工程
- **场景 (scenario)**：专科医疗AI / 计算机辅助手术 / 眼科
- **机制 (mechanism)**：层次化标注 / 多任务评测 / 领域迁移分析
- **约束 (constraint)**：医学标注成本 / 领域特殊性 / 数据隐私与伦理
