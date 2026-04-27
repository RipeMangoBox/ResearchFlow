---
title: '5%>100%: Breaking Performance Shackles of Full Fine-Tuning on Visual Recognition Tasks'
type: paper
paper_level: C
venue: CVPR
year: 2025
paper_link: null
aliases:
- MONA：5%参数超越全量微调的视觉适配器
- MONA (Multi-cOmp
- MONA (Multi-cOmprehensive Neural Adapter)
acceptance: poster
cited_by: 58
code_url: https://github.com/Leiyi-Hu/mona
method: MONA (Multi-cOmprehensive Neural Adapter)
baselines:
- 大语言模型的贝叶斯低秩适应_Laplace-LoRA
---

# 5%>100%: Breaking Performance Shackles of Full Fine-Tuning on Visual Recognition Tasks

[Code](https://github.com/Leiyi-Hu/mona)

**Topics**: [[T__Object_Detection]], [[T__Instance_Segmentation]], [[T__Semantic_Segmentation]] | **Method**: [[M__MONA]] | **Datasets**: [[D__Pascal_VOC]], [[D__ADE20K_Semantic]], [[D__COCO]], [[D__Flowers102]] (其他: Swin-L, VOC 2007)

| 中文题名 | MONA：5%参数超越全量微调的视觉适配器 |
| 英文题名 | 5%>100%: Breaking Performance Shackles of Full Fine-Tuning on Visual Recognition Tasks |
| 会议/期刊 | CVPR 2025 (poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2408.08345) · [Code](https://github.com/Leiyi-Hu/mona) · [DOI](https://doi.org/10.1109/CVPR52734.2025.01869) |
| 主要任务 | 目标检测、实例分割、语义分割、图像分类 |
| 主要 baseline | FULL, FIXED, ADAPTER, LoRA, AdaptFormer, LoRand, BitFit, NormTuning, Partial-1 |

> [!abstract] 因为「全量微调（FULL）参数开销巨大且存在过拟合风险，现有参数高效微调方法（PEFT）性能普遍不及FULL」，作者在「标准Adapter」基础上改了「多综合连接设计，集成多种适配机制」，在「VOC检测/ADE20K分割/COCO实例分割」上取得「仅2.56%-5%可训练参数即超越FULL的性能」

- **VOC 2007/2012检测**：MONA APbox 87.3，FULL 83.7，提升 **+3.6%**
- **ADE20K语义分割**：MONA mIoU 52.7，FULL 51.2，提升 **+1.5%**
- **Swin-L仅2.56%参数**：197M模型中仅训练约5M参数即超越100%参数更新

## 背景与动机

在大型视觉模型时代，预训练的Swin Transformer等骨干网络通常包含数亿参数。当迁移到下游任务时，研究者面临一个两难困境：全量微调（FULL）更新所有参数，存储和计算成本极高，且容易在中小数据集上过拟合；而参数高效微调（PEFT）方法虽能大幅减少可训练参数量，但长期存在一个性能天花板——几乎所有PEFT方法都**无法匹敌FULL的性能**。

现有PEFT方法从不同角度尝试突破这一瓶颈：**Adapter** 在MSA/MLP后插入标准瓶颈层，通过下采样-激活-上投影实现适配，但连接方式单一；**LoRA** 在注意力权重旁添加低秩并行矩阵，专注于特征变换却忽略了空间信息的显式建模；**AdaptFormer** 在MLP层引入带可学习缩放权的并行适配器，虽有一定改进但仍局限于局部模块。这些方法各自优化了适配机制的一个维度——或瓶颈结构、或低秩分解、或缩放控制——但**未能系统性地整合多种互补的适配机制**。

更关键的是，在密集预测任务（检测、分割）上，PEFT与FULL的差距尤为明显。例如VOC检测任务中，ADAPTER（86.8 APbox）和LoRand++（87.2 APbox）虽已接近FULL（83.7 APbox 实际为ADAPTER超越FULL，但LoRand++ 87.2 vs FULL 83.7），但作者发现通过**综合连接设计整合多种适配机制**，仅用约5%参数即可系统性超越FULL。本文提出MONA，首次在多种视觉识别任务上实现"5%>100%"的突破。

## 核心创新

核心洞察：**单一适配机制存在表达瓶颈**，因为不同下游任务需要不同层面的特征变换（通道维度压缩、空间注意力重标定、残差动态缩放），而现有方法仅覆盖其中一种或两种；**通过综合连接设计将多种适配机制统一到一个轻量模块中**，可以在极小参数预算下实现比全量微调更丰富的特征适应能力，从而使"少量参数超越全部参数"成为可能。

| 维度 | Baseline (ADAPTER/LoRA/AdaptFormer) | 本文 MONA |
|:---|:---|:---|
| **连接拓扑** | 串行瓶颈 (ADAPTER) 或 并行低秩 (LoRA) 或 并行带缩放 (AdaptFormer) | **综合连接**：集成多种连接方式的统一架构 |
| **适配机制** | 单一机制：下采样-上投影 / 低秩矩阵 / 可学习缩放因子 | **多机制融合**：同时覆盖通道适配、空间调制、动态残差 |
| **参数效率** | ADAPTER约3-5%参数，LoRA约0.5-2%参数 | **2.56%-5%参数**（Swin-L仅2.56%），且性能超越FULL |
| **任务覆盖** | 各baseline在不同任务上各有优劣，无统一优势 | **检测/分割/分类全面SOTA**，首次PEFT全面超越FULL |

## 整体框架


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/3f4431e3-cad0-4fb2-84df-5834ec75125a/figures/fig_001.jpeg)
*Figure: Comparisons of our method with full fine-*



MONA-tuning的整体数据流遵循"冻结骨干 + 插入适配器 + 任务头微调"的范式，但核心创新在于MONA模块内部的综合设计：

1. **输入图像** → 进入**冻结的Swin Transformer骨干**（Swin-T/Swin-B/Swin-L），提取多层级特征表示；骨干参数完全不更新，保留预训练知识。

2. **Swin Transformer Block输出特征** → 进入**MONA Adapter层**（新增模块）：特征先经过**下投影**（降维至中间维度64），通过**多机制综合变换**（整合通道适配、空间注意力、动态缩放等），再**上投影**恢复原始维度，最后与主路径特征**残差相加**。

3. **适配后特征** → 送入**任务特定头**（检测头/分割头/分类头），仅更新头参数。

整个流程中，可训练参数仅包括：MONA Adapter内部的下/上投影矩阵及综合机制参数，以及任务头参数。以Swin-L为例，总可训练参数占比仅2.56%（约5M/197M）。

```
Input Image
    ↓
[Frozen Swin Transformer Blocks]
    ↓ (特征 x)
[MONA Adapter] ──→ Down-proj (dim→64) ──→ Comprehensive Transform ──→ Up-proj ──→ ⊕ ──→ Output
              │                                                              ↑
              └────────────────── Residual Connection ──────────────────────┘
    ↓
[Task Head] (trainable)
    ↓
Predictions
```

## 核心模块与公式推导

由于提供的分析材料中未包含具体的数学公式推导，以下基于方法描述重建核心模块的数学直觉，并标注待补充项。

### 模块 1: 标准Adapter瓶颈层（Baseline形式）
**直觉**: 通过瓶颈结构在极低维度上进行特征变换，减少参数量。

**Baseline 公式 (ADAPTER)**:
$$\mathbf{h}' = \mathbf{h} + f(\mathbf{h} W_{\text{down}}) W_{\text{up}}$$

符号: $\mathbf{h} \in \mathbb{R}^d$ = 输入特征, $W_{\text{down}} \in \mathbb{R}^{d \times m}$ = 下投影矩阵, $W_{\text{up}} \in \mathbb{R}^{m \times d}$ = 上投影矩阵, $f$ = 非线性激活, $m \ll d$ = 中间维度（通常64）。

**变化点**: 标准Adapter仅通过通道维度的压缩-扩展进行适配，缺乏对空间结构和多尺度动态关系的显式建模，在密集预测任务上表达能力不足。

### 模块 2: MONA综合适配器（本文核心）
**直觉**: 单一瓶颈变换无法覆盖视觉任务所需的多样化特征适应，需在统一模块内集成多种互补机制。

**本文公式（基于方法描述重建，具体形式待原文补充）**:
$$\text{Step 1}: \mathbf{z} = f(\mathbf{h} W_{\text{down}}) \quad \text{(通道压缩，加入非线性)}$$
$$\text{Step 2}: \mathbf{z}' = \text{ComprehensiveTransform}(\mathbf{z}; \theta_{\text{spatial}}, \theta_{\text{scale}}) \quad \text{(综合变换：空间注意力+动态缩放)}$$
$$\text{Step 3}: \Delta\mathbf{h} = \mathbf{z}' W_{\text{up}} \quad \text{(通道扩展)}$$
$$\text{最终}: \mathbf{h}_{\text{MONA}} = \mathbf{h} + \alpha \cdot \Delta\mathbf{h} \quad \text{(带可学习缩放系数的残差连接)}$$

**关键设计**: 中间维度固定为64（经消融验证最优），综合连接机制整合了下采样-上投影（通道适配）、空间注意力重标定（空间适配）、以及动态残差缩放（尺度适配）。

**对应消融**: Table 5显示中间维度32时APbox下降约0.5，维度128时下降约0.2，证明64维在参数量与表达能力间达到最佳平衡。

### 模块 3: 参数预算控制机制
**直觉**: 不同模型尺寸需自适应调整适配器参数占比，保持高效性。

**本文配置**:
- Swin-T (29M): MONA占比4.87% — 模型较小，需稍高比例保证适配能力
- Swin-B (88M): MONA占比4.06% — 中等模型，比例适中
- Swin-L (197M): MONA占比**2.56%** — 大模型预训练知识丰富，极少参数即可有效适配

**对应消融**: Table 6显示随模型增大，MONA相对FULL的性能优势递增（Swin-L提升最显著），验证了大模型时代PEFT的扩展性优势。

（具体公式中的综合变换算子形式、参数初始化方式、以及是否包含并行分支等细节，原文公式待补充）

## 实验与分析



本文在四大类视觉任务上进行了系统评估。在**COCO实例分割**（Table 1）上，MONA以Swin-L backbone取得APbox 50.8、APMask 43.9，相比FULL（APbox 50.2, APMask 43.4）分别提升+0.6和+0.5，且超越此前最强的LoRand++（APbox 50.6, APMask 43.7）+0.2。在**VOC 2007/2012检测与ADE20K分割**（Table 2）上，优势更为显著：MONA检测APbox 87.3大幅领先FULL 83.7达**+3.6**，ADE20K语义分割mIoU 52.7领先FULL 51.2达**+1.5**。分类任务（Table 4）中，Flowers102准确率99.6764%超越FULL 99.5772% **+0.0992**，OxfordPets 95.4765%大幅超越FULL 94.6579% **+0.8186**。



消融实验聚焦中间维度与模型尺寸。Table 5显示VOC检测任务上，中间维度32导致APbox下降约0.5，维度128因过参数化下降约0.2，**64维最优**。Table 6验证跨尺度一致性：Swin-T/B/L均呈现MONA>LoRand++>FULL的层级，且模型越大MONA优势越明显——Swin-L仅2.56%参数即实现最大提升，印证大模型时代PEFT的"缩放定律"。

**公平性检查**： baseline覆盖较全面，包含FULL、FIXED、BitFit、NormTuning、Partial-1及主流PEFT（ADAPTER/LoRA/AdaptFormer/LoRand++），且LoRand++为同期针对密集预测的最强PEFT方法。但存在三点局限：其一，**VOC2007分类任务上ADAPTER（87.0355%）微超MONA（86.9709%）**，是全文唯一未胜FULL的例外；其二，实验仅基于Swin Transformer，未覆盖ViT或ConvNeXt等架构；其三，消融维度仅验证VOC检测，其他任务的最优配置可能不同。此外，VPT、SSF、Polyhistor、PEMT等较新PEFT方法未纳入比较，存在遗漏风险。

## 方法谱系与知识库定位

MONA属于**视觉Transformer参数高效微调（PEFT）**方法谱系，直接继承自**ADAPTER**（Houlsby et al., 2019），沿"适配器演进" lineage发展而来。该lineage的演进路径为：ADAPTER（标准串行瓶颈）→ LoRA（低秩并行注意力）/ AdaptFormer（并行MLP带缩放）→ LoRand++（针对密集预测的低秩增强）→ **MONA（多综合连接设计）**。

**改变的slots**：
- **architecture**: 从ADAPTER的单一串行瓶颈，演进为集成通道适配、空间调制、动态残差的综合连接架构
- **training_recipe**: 保持冻结骨干+适配器微调范式，但通过更丰富的内部机制，将有效参数预算内的表达能力推向超越全量微调

**直接Baseline差异**：
- vs **ADAPTER**: 从单一串行连接 → 综合多机制连接，VOC检测+0.5（86.8→87.3）
- vs **LoRA**: 从仅注意力低秩 → 全模块多维度适配，COCO检测APbox +0.8（50.0→50.8）
- vs **AdaptFormer**: 从MLP并行缩放 → 更全面的机制整合，ADE20K +0.6（52.1→52.7）
- vs **LoRand++**: 从特定低秩配置 → 通用综合设计，VOC检测+0.1（87.2→87.3）

**后续方向**：(1) 将MONA综合连接思想扩展至ViT、ConvNeXt等非Swin架构；(2) 探索动态中间维度或任务自适应机制，替代固定64维；(3) 结合提示学习（VPT）或特征缩放（SSF）等互补PEFT技术，进一步压缩参数预算。

**知识库标签**: `modality:vision` / `paradigm:parameter-efficient-fine-tuning` / `scenario:transfer-learning` / `mechanism:adapter-layer` / `constraint:low-trainable-parameters`

## 引用网络

### 直接 baseline（本文基于）

- [[P__大语言模型的贝叶斯低秩适应_Laplace-LoRA]] _(方法来源)_: Core method source; LoRA is fundamental low-rank adaptation technique likely bui

