---
title: 'SGC-Net: Stratified Granular Comparison Network for Open-Vocabulary HOI Detection'
type: paper
paper_level: C
venue: CVPR
year: 2025
paper_link: null
aliases:
- 分层粒度比较网络SGC-Net用于开放词汇HOI检测
- SGC-Net
acceptance: poster
cited_by: 2
code_url: https://github.com/Phil0212/SGC-Net
method: SGC-Net
---

# SGC-Net: Stratified Granular Comparison Network for Open-Vocabulary HOI Detection

[Code](https://github.com/Phil0212/SGC-Net)

**Topics**: [[T__Object_Detection]], [[T__Few-Shot_Learning]] | **Method**: [[M__SGC-Net]] | **Datasets**: [[D__HICO-DET]], [[D__SWIG-HOI]]

| 中文题名 | 分层粒度比较网络SGC-Net用于开放词汇HOI检测 |
| 英文题名 | SGC-Net: Stratified Granular Comparison Network for Open-Vocabulary HOI Detection |
| 会议/期刊 | CVPR 2025 (poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2503.00414) · [Code](https://github.com/Phil0212/SGC-Net) · [DOI](https://doi.org/10.1109/cvpr52734.2025.00428) |
| 主要任务 | Open-Vocabulary Human-Object Interaction (HOI) Detection |
| 主要 baseline | CMD-SE [24], GEN-VLKT [31], HOICLIP [39], OpenCat [58], MP-HOI [53] |

> [!abstract] 因为「开放词汇HOI检测中，现有方法对细粒度语义相似交互的判别能力不足，且直接匹配视觉-文本特征难以泛化到未见类别」，作者在「CMD-SE」基础上改了「引入层次粒度比较(HGC)、动态高斯加权(DGW)和粒度分层聚合(GSA)三大模块，并用LLM生成层次化语义描述」，在「HICO-DET Unseen」上取得「+6.57% mAP」

- **HICO-DET Unseen**: 相比 CMD-SE 提升 **+6.57% mAP**，Seen +4.39%，Full +4.87%
- **SWIG-HOI**: Rare 16.55 vs CMD-SE 14.79 (+1.76%)，Unseen 12.46 vs 10.55 (+1.91%)，Full 17.2（超越所有依赖预训练检测器的OV-HOI方法）
- **消融验证**: 移除 GSA 导致 SWIG-HOI Full mAP **-5.04**，移除 HGC **-2.39**，验证核心模块必要性

## 背景与动机

开放词汇人机交互检测（Open-Vocabulary HOI Detection）旨在识别训练时未见过的交互类别，例如模型只见过"ride bicycle"，但需要检测"ride horse"。现有方法普遍依赖 CLIP 等视觉-语言预训练模型，将图像区域特征与文本嵌入直接匹配以分类交互。然而，这种"单点匹配"范式面临两个根本性困难：一是**细粒度混淆**——语义相近的交互（如"hold knife"与"cut with knife"）在特征空间中难以区分；二是**未见类别迁移差**——直接相似度计算无法利用层次化语义结构来泛化到新类别。

CMD-SE [24] 作为当前无需预训练检测器的代表性 baseline，采用 CLIP ViT-B/16 提取最后一层视觉特征，配合 12 个可学习 token 进行检测，虽简化了流程但存在明显局限：仅使用单层特征丢失了中间层蕴含的多粒度信息；文本端仅用固定类别标签编码，缺乏对交互语义的层次化刻画；推理时直接计算视觉-文本点积，无法根据匹配置信度动态选择描述粒度。GEN-VLKT [31]、HOICLIP [39] 等方法虽引入 CLIP 文本嵌入，但仍停留在单层特征与固定描述的框架内。MP-HOI [53] 等依赖预训练检测器的方法在 SWIG-HOI 上表现不佳，且需额外标注成本。

核心瓶颈在于：**视觉特征提取缺乏层次化粒度融合，文本语义缺乏结构化层次描述，而两者的匹配机制又过于粗粒度**。本文提出 SGC-Net，通过"分层粒度比较"同时改造视觉端、文本端与匹配策略，实现从粗到细、动态门控的交互识别。

## 核心创新

核心洞察：**人类理解交互是层次化的**——先判断"是否接触物体"，再确认"如何使用"，最后细化"具体动作"；因为 CLIP 不同层编码了从低级视觉模式到高级语义的不同粒度信息，且 LLM 可自动生成对应层次的文本描述，从而使"逐层递归、阈值门控的粒度比较"成为可能。

| 维度 | Baseline (CMD-SE) | 本文 (SGC-Net) |
|:---|:---|:---|
| 视觉特征 | 单层（最后一层）ViT 特征 | 分块 {6-8}, {9-11}, {12} 多粒度特征，DGW 距离感知聚合 |
| 文本描述 | 固定类别标签 | GPT-3.5 生成层次化粒度描述，分组阈值 N=6 平衡粒度与效率 |
| 匹配机制 | 直接点积相似度 $p_i = \mathbf{D}_i \cdot \mathbf{x}^T$ | GSA 递归分数：仅当细粒度层级置信度显著高于粗粒度时才激活，阈值 τ 控制 |
| 推理策略 | 单级匹配 | 粗粒度基础分 + 细粒度递归分的 λ=0.5 插值融合 |

## 整体框架


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/de4107dc-2cd3-44a7-af24-fea5a2a836bf/figures/fig_001.png)
*Figure: (a) The last layer capture high-level global semantics*



SGC-Net 的完整数据流可分为视觉分支与文本分支，最终在 GSA 模块中融合推理：

**视觉分支**：输入图像 → CLIP ViT-B/16（12层）逐层提取特征 → **Layer Splitting** 将 12 层划分为三个语义块 {6-8}（中低级视觉）、{9-11}（高级语义）、{12}（最抽象全局）→ **DGW 模块** 按各层到目标层的距离计算高斯权重，聚合为最终视觉特征 $\mathbf{Z}$ → 与查询 $\mathbf{Q}$ 一并送入检测解码器输出人体/物体框。

**文本分支**：类别标签 → **GPT-3.5 生成层次化描述**（分组阈值 N=6 控制层次数 $M_i$）→ 前缀/连接 token 增强 → CLIP 文本编码器 → 多层级文本嵌入 $\{\mathbf{x}^j\}_{j=1}^{M_i}$。

**匹配分支（GSA）**：图像区域表示 $\mathbf{D}_i^j$ 与各层级文本嵌入计算相似度 $p_i^j$ → 阈值门控判断 $\mathbf{u}_i^k$ 决定是否激活更细粒度 → 递归聚合得 $r(\mathbf{x}, i)$ → 与基础相似度插值得最终交互分数 $s(\mathbf{x}, i)$ → 经物体置信度校准后输出。

```
图像 ──→ CLIP ViT-B/16 ──→ Layer Splitting ──→ DGW Aggregation ──→ Decoder ──→ 框+交互预测
                                              ↑
LLM层次描述 ──→ Text Encoder ──→ GSA (递归门控匹配) ──→ 分数校准 ──→ 最终分数
```

## 核心模块与公式推导

### 模块 1: Dynamic Gaussian Weighting (DGW)（对应框架图 视觉特征聚合部分）

**直觉**: CLIP 预训练的视觉-语言对齐主要建立在特定层上，直接拼接或求和各层会破坏这种对齐；应按"距离目标层的远近"衰减权重，保留有效信息同时抑制干扰。

**Baseline 公式** (CMD-SE 的直接聚合): $$\mathbf{Z} = \text{Concat}(\mathbf{F}_l) \text{ 或 } \sum \mathbf{F}_l$$
符号: $\mathbf{F}_l$ = 第 $l$ 层视觉特征, $l \in [1, 12]$（CLIP ViT-B/16 的 12 层）

**变化点**: 均匀聚合或简单拼接导致不同层的视觉-语言关联相互干扰；本文假设各层对最终语义贡献应服从以目标层 $d$ 为中心的高斯分布。

**本文公式（推导）**:
$$\text{Step 1}: \alpha_l^s = \exp\left(-\frac{1}{2} \frac{(d - l)^2}{\sigma^2}\right), \quad l \in [1, d] \quad \text{（加入距离感知高斯权重，保护预训练对齐）}$$
$$\text{Step 2}: \mathbf{Z} = \sum_{s=1}^{S} \alpha_s \left(\sum_{l=1}^{d}\alpha_{l}^s \mathbf{F}_{l}\right) \quad \text{（块内按层权重聚合，块间按可学习权重 } \alpha_s \text{ 融合）}$$
**最终**: 聚合特征 $\mathbf{Z}$ 输入检测解码器，配合查询 $\mathbf{Q}$ 得 $\mathbf{X} = \text{Dec}(\mathbf{Q}, \mathbf{Z})$

**对应消融**: 将 DGW 替换为 Self-attention 聚合，SWIG-HOI Full mAP **-3.3**；替换为 Concat/Sum 同样显著下降。引入可学习块权重 $\alpha_s$（DGW*）相比固定权重再提升 **+0.27** mAP。

---

### 模块 2: Granular Stratified Aggregation (GSA)（对应框架图 推理匹配部分）

**直觉**: 人类识别新交互时会"由粗到细"验证——若"使用工具"都不匹配，则无需判断"切/削"；通过阈值门控实现这种"早停"机制，避免细粒度描述的噪声干扰。

**Baseline 公式** (CMD-SE 直接相似度): $$p_i = \mathbf{D}_i \cdot \mathbf{x}^T, \quad s(\mathbf{x}, i) = p_i$$
符号: $\mathbf{D}_i$ = 图像区域表示, $\mathbf{x}$ = 文本嵌入, $p_i$ = 单级匹配分数

**变化点**: 单层匹配无法利用层次化语义结构；对于未见类别，粗粒度描述（如"人与物体接触"）比细粒度（如"用刀切割"）更可靠，但需要机制判断是否值得"深入"更细粒度。

**本文公式（推导）**:
$$\text{Step 1}: p_i^j = \mathbf{D}_i^j \cdot \mathbf{x}^T \quad \text{（计算第 } j \text{ 个粒度层级的相似度，} j \in [1, M_i] \text{）}$$
$$\text{Step 2}: \mathbf{u}_i^k = \mathbb{I}\left(p_i^{k+1} > p_i^k + \tau\right) \quad \text{（加入阈值门控：仅当细粒度显著优于粗粒度 } \tau \text{ 时才激活）}$$
$$\text{Step 3}: r(\mathbf{x}, i) = \frac{p_i^1 + \sum_{j=2}^{M_i} p_i^j \prod_{k=1}^{j-1} \mathbf{u}_i^k}{1 + \sum_{j=2}^{M_i} \prod_{k=1}^{j-1} \mathbf{u}_i^k} \quad \text{（递归加权平均：分子为层级分数的累积加权和，权重为前置门控的累积乘积；分母归一化）}$$
$$\text{最终}: s(\mathbf{x}, i) = (1 - \lambda) (\mathbf{p}_i^1 + \mathbf{t} \cdot \mathbf{x}^T) + \lambda \cdot r(\mathbf{x}, i), \quad \lambda = 0.5$$

**对应消融**: 移除 GSA（仅用单层匹配），SWIG-HOI Full mAP **-5.04**，为所有消融中最大降幅，验证递归层次聚合对开放词汇泛化的核心作用。

---

### 模块 3: Hierarchical Granular Comparison (HGC)（对应框架图 层划分部分）

**直觉**: CLIP 不同层关注不同模式——浅层关注边缘纹理，中层关注部件组合，深层关注完整语义；将相邻层分组为"语义块"可在保留连续性的同时获取多粒度判别信号。

**Baseline 公式**: 单层提取 $\mathbf{F}_{12}$，无分块概念。

**变化点**: 单层丢失中间粒度信息；但随意分块（如均匀切分）会破坏层间连续性。实验发现 {6-8}, {9-11}, {12} 的非均匀划分最优——将最抽象的第 12 层独立保留其全局语义。

**本文公式（推导）**:
$$\text{Step 1}: \text{Block}_1 = \{F_6, F_7, F_8\}, \text{Block}_2 = \{F_9, F_{10}, F_{11}\}, \text{Block}_3 = \{F_{12}\} \quad \text{（经验性最优划分，独立保留顶层）}$$
$$\text{Step 2}: \text{各 Block 经 DGW 内部聚合后，再跨块融合} \quad \text{（重归一化保证多粒度信息不淹没）}$$
**最终**: 三粒度特征共同输入检测头，增强对"hold knife" vs "cut with knife" 等细粒度交互的判别。

**对应消融**: 改为均匀划分 {4-6}, {7-9}, {10-12}，SWIG-HOI Full mAP **-0.85**，证明将第 12 层独立分离的关键性。

## 实验与分析



本文在 HICO-DET 和 SWIG-HOI 两个标准 benchmark 上评估开放词汇 HOI 检测性能。在 **HICO-DET** 上，SGC-Net 相比直接 baseline CMD-SE 实现全面提升：Unseen 类别 mAP **+6.57%**（核心指标，衡量对新交互的泛化能力），Seen 类别 **+4.39%**，Full **+4.87%**。这一差距表明层次化粒度比较对开放词汇场景尤为关键——未见类别缺乏训练样本，必须依赖语义层次的迁移推理。

在更大规模的 **SWIG-HOI** 上，SGC-Net 取得 Rare 16.55（vs CMD-SE 14.79，+1.76）、Unseen 12.46（vs 10.55，+1.91）、Full 17.2 的绝对数值。特别地，Full 17.2 超越了所有依赖预训练检测器的 OV-HOI 方法（如 MP-HOI），而 SGC-Net 无需此类预训练，简化了部署流程。



消融实验进一步量化各模块贡献。除前述 GSA (-5.04)、HGC (-2.39)、DGW 替换 (-3.3) 外，LLM 层次描述的分组阈值 N 经实验确定为 **N=6**：N=4 层次过细导致冗余，N=8 层次过粗丧失判别力。图 3 展示了该权衡曲线。定性分析（图 4）显示，层次化描述使模型能区分"wash cup"与"drink from cup"——粗粒度层均匹配"手持杯子"，但细粒度层通过"水流/饮用动作"实现正确判别。

**公平性审视**: 作者坦诚 HICO-DET 上与 DETR-based 方法（如 GEN-VLKT）的比较"不完全公平"，因后者经 COCO 预训练存在数据泄漏风险。此外，本文未报告推理延迟、参数量或训练耗时，与 HOICLIP、OpenCat 等的直接数值对比亦不完整（仅 CMD-SE 有完整相对提升）。SWIG-HOI 的绝对数值虽清晰，但部分 baseline 绝对值未在提取文本中显式给出。

## 方法谱系与知识库定位

**方法家族**: 基于 CLIP 的开放词汇检测 → 无需预训练检测器的端到端 OV-HOI

**父方法**: CMD-SE [24]（"Exploring the Potential of Large Foundation Models for Open-Vocabulary HOI Detection"）——首个移除预训练检测器依赖、纯 CLIP 驱动的 OV-HOI 框架。

**改动槽位**: 
- **Architecture**: 单层特征 → HGC 分块 + DGW 高斯加权聚合
- **Inference strategy**: 直接相似度匹配 → GSA 递归门控层次聚合  
- **Data curation**: 固定标签 → GPT-3.5 生成层次化粒度描述（分组阈值 N=6）
- **Training recipe**: 损失权重 λ_b=5, λ_cls=2, λ_iou=5，DGW 可学习块权重

**直接 baselines 差异**:
- **GEN-VLKT [31]**: DETR 架构 + CLIP 文本嵌入，需 COCO 预训练检测器
- **HOICLIP [39]**: 知识蒸馏加速，仍单层特征固定描述
- **MP-HOI [53]**: 多提示学习，依赖预训练检测器，SWIG-HOI 上显著落后

**后续方向**: (1) 扩展至更多视觉编码器（如 SigLIP、EVA-CLIP）验证 DGW 通用性；(2) 层次描述的自动化优化替代人工设定阈值 N 和 τ；(3) 视频 HOI 的时序粒度扩展。

**标签**: 视觉-语言融合 / 开放词汇检测 / 人机交互检测 / 层次化推理 / 递归门控机制 / 无需预训练检测器

