---
title: Pixel-Wise Recognition for Holistic Surgical Scene Understanding
type: paper
paper_level: C
venue: MedIA
year: 2024
paper_link: https://www.semanticscholar.org/paper/7cd08e04df9160ee28ae2c57dbaf098b165b330a
aliases:
- 手术场景像素级多任务联合理解框架
- PWRHSS
- 手术场景中四个视觉任务共享强烈的语义与几何关联
cited_by: 10
code_url: https://github.com/isyangshu/Awesome-Surgical-Video-Understanding
---

# Pixel-Wise Recognition for Holistic Surgical Scene Understanding

[Paper](https://www.semanticscholar.org/paper/7cd08e04df9160ee28ae2c57dbaf098b165b330a) | [Code](https://github.com/isyangshu/Awesome-Surgical-Video-Understanding)

**Topics**: [[T__Semantic_Segmentation]], [[T__Medical_Imaging]]

> [!tip] 核心洞察
> 手术场景中四个视觉任务共享强烈的语义与几何关联，单一编码器被迫学习对所有任务均有益的紧凑表示，任务间的正向迁移通过梯度共享隐式实现。有效性来源于：手术场景的视觉结构高度规律化（器械、组织、背景分布稳定），共享特征的泛化收益大于任务冲突的损耗。本质上是「用参数共享换取跨任务正则化」的经典多任务学习直觉在手术领域的实例化。

| 中文题名 | 手术场景像素级多任务联合理解框架 |
| 英文题名 | Pixel-Wise Recognition for Holistic Surgical Scene Understanding |
| 会议/期刊 | Medical Image Analysis (journal) |
| 链接 | [Semantic Scholar](https://www.semanticscholar.org/paper/7cd08e04df9160ee28ae2c57dbaf098b165b330a) · [Code](https://github.com/isyangshu/Awesome-Surgical-Video-Understanding) · [DOI](https://doi.org/10.1016/j.media.2025.103726) |
| 主要任务 | 语义分割、单目深度估计、手术器械检测、动作识别（四任务联合） |
| 主要 baseline | 单任务专用模型（独立编码器+独立解码器） |

> [!abstract]
> 因为「手术场景理解任务碎片化导致参数冗余与互补信息无法共享」，作者在「单任务独立模型」基础上改了「单一共享编码器+四任务专用解码器的联合训练框架」，在「CholecT50和Cholec80」上取得「四任务联合训练优于任意子集组合，任务数量与性能正相关」

- **关键性能 1**：四任务联合训练在语义分割mIoU、深度估计误差、动作识别准确率上均优于对应单任务基线（Table 2，具体数值待补充）
- **关键性能 2**：消融实验显示增加任务数量单调提升各任务性能，四任务模型最优（Table 3）
- **关键性能 3**：器械检测AP略低于专用单任务SOTA，存在性能权衡

## 背景与动机

腹腔镜手术场景理解长期面临任务碎片化的困境。具体而言，一台胆囊切除术中，外科医生需要同时感知：组织与器官的精确边界（语义分割）、器械与组织的距离关系（深度估计）、抓钳/电钩等工具的位置类别（器械检测）、以及当前处于解剖/剥离/止血哪个阶段（动作识别）。这四个子任务传统上由四个独立模型分别处理——每个模型拥有独立的CNN或Transformer编码器，分别加载、分别推理、分别优化。

现有方法的处理方式各有局限：通用语义分割模型（如DeepLab、SegFormer）专注于像素分类，无法利用深度几何信息约束边界；单目深度估计方法（如MiDaS、DPT）在手术场景缺乏密集深度真值时性能骤降；器械检测模型（如YOLO、DETR）孤立预测边界框，忽视器械与动作的语义关联；动作识别网络（如I3D、TimeSformer）提取全局视频特征，丢失像素级空间细节。更关键的是，这些单任务范式完全忽视了手术场景固有的结构化关联——器械位置与动作类别强相关（抓钳夹持必然伴随"抓取"动作），场景深度与器械遮挡关系密切（近处组织遮挡远处器械），分割掩码与深度图共享几何先验（组织边界即深度不连续处）。

现有多任务学习方法（如通用场景理解框架MTI-Net、TaskPrompter）未针对手术场景的特殊性进行适配：器械细粒度识别需要高分辨率特征、深度标注缺失需要弱监督机制、动作时序依赖需要视频建模。直接迁移效果有限，且手术视频标注成本极高，密集多标签数据稀缺，单任务模型无法利用跨任务的弱监督信号进行互补学习。

本文的核心动机正是：如何在统一框架内同时建模四个异质任务、充分利用任务间互补性、在有限标注下实现整体手术场景感知。

## 核心创新

核心洞察：手术场景中四个视觉任务共享强烈的语义与几何关联，单一编码器被迫学习对所有任务均有益的紧凑表示，任务间的正向迁移通过梯度共享隐式实现，从而使无需显式跨任务交互模块的多任务联合训练成为可能。

与 baseline 的差异：

| 维度 | Baseline（单任务独立模型） | 本文 |
|:---|:---|:---|
| 架构 | 四个独立编码器+四个独立解码器 | 单一共享编码器 + 四个任务专用解码头 |
| 参数效率 | 参数量冗余，推理需四次前向 | 参数量压缩，一次前向完成四任务预测 |
| 任务交互 | 无交互，各任务信息孤岛 | 梯度共享隐式实现正向迁移，无需显式注意力或交互模块 |
| 数据利用 | 各任务仅使用对应标注 | 深度估计分支利用弱监督/自监督信号降低数据依赖 |
| 部署成本 | 四个模型分别加载维护 | 统一模型单次部署 |

本质上是「用参数共享换取跨任务正则化」的经典多任务学习直觉在手术领域的实例化，属于标准多任务学习范式的领域适配而非架构层面的根本性创新。

## 整体框架

输入手术视频帧 $I \in \mathbb{R}^{H \times W \times 3}$ 首先进入单一共享编码器（采用Swin Transformer或SegFormer等Transformer骨干），输出多尺度特征图 $F \in \mathbb{R}^{H/s \times W/s \times C}$，其中 $s$ 为下采样率，$C$ 为通道维度。该特征图 $F$ 被所有下游任务头共享，形成"一分四"的分叉式架构：

- **语义分割头**：接收 $F$，通过上采样与逐像素分类层，输出像素级类别预测 $\hat{Y}_{seg} \in \mathbb{R}^{H \times W \times K_{cls}}$，$K_{cls}$ 为组织/器官类别数；
- **单目深度估计头**：接收 $F$，通过深度回归解码器，输出像素级深度图 $\hat{D} \in \mathbb{R}^{H \times W}$，在缺乏密集真值时采用弱监督或自监督信号；
- **手术器械检测头**：接收 $F$，通过检测头（如基于anchor或query的机制），输出边界框坐标与器械类别 $(\hat{bbox}, \hat{c}_{tool})$；
- **动作识别头**：接收 $F$，通过时序聚合（帧级或片段级），输出动作类别概率 $\hat{a} \in \mathbb{R}^{K_{action}}$。

四个头的预测在训练时由加权多任务损失联合优化，推理时单次前向同时输出四任务结果。

```
输入帧 I ──→ [共享编码器] ──→ 多尺度特征 F
                                  │
          ┌─────────┬───────────┼───────────┐
          ↓         ↓           ↓           ↓
    [分割头]   [深度头]    [检测头]     [动作头]
          │         │           │           │
          ↓         ↓           ↓           ↓
      Ŷ_seg       D̂        (bbox,ĉ)      â
```

## 核心模块与公式推导

### 模块 1: 加权多任务联合损失（对应框架图 整体训练流程）

**直觉**: 四个异质任务的损失量级与梯度特性差异显著，需加权平衡以避免某一任务主导梯度更新。

**Baseline 公式** (单任务独立训练):
$$L_{seg}^{single} = \text{CrossEntropy}(\hat{Y}_{seg}, Y_{seg})$$
$$L_{depth}^{single} = \| \hat{D} - D_{gt} \|_1$$
$$L_{tool}^{single} = L_{cls} + L_{reg} \quad \text{(检测标准组合损失)}$$
$$L_{action}^{single} = \text{CrossEntropy}(\hat{a}, a_{gt})$$

符号: $\hat{Y}_{seg}, \hat{D}, \hat{a}$ 为各任务预测；$Y_{seg}, D_{gt}, a_{gt}$ 为对应真值；$L_{cls}, L_{reg}$ 为检测分类与回归损失。

**变化点**: 单任务独立优化导致各编码器学习互不相干的特征表示，无法利用任务互补性；且深度估计在手术场景缺乏密集 $D_{gt}$，需要引入弱监督信号 $\tilde{D}$（来自立体匹配或稀疏标注）。

**本文公式（推导）**:
$$\text{Step 1}: L_{depth}^{weak} = \| \hat{D} - \tilde{D} \|_1 + \lambda_{smooth} \cdot L_{smooth} \quad \text{加入平滑正则以补偿弱监督噪声}$$
$$\text{Step 2}: L_{total} = L_{seg} + \lambda_1 L_{depth}^{weak} + \lambda_2 L_{tool} + \lambda_3 L_{action} \quad \text{加权组合，通过}\lambda_i\text{平衡梯度贡献}$$
$$\text{最终}: L_{final} = \sum_{t \in \{seg, depth, tool, action\}} \lambda_t L_t, \quad \lambda_{seg}=1, \lambda_1, \lambda_2, \lambda_3 \text{为超参数}$$

**对应消融**: Table 3 显示移除任一任务（即退化为三任务/双任务/单任务）均导致剩余任务性能下降，验证了任务互补效应。

---

### 模块 2: 共享编码器的隐式知识迁移（对应框架图 编码器部分）

**直觉**: 强制单一编码器服务四个异质解码器，相当于施加跨任务正则化，迫使特征提取关注手术场景的通用视觉结构。

**Baseline 公式** (独立编码器):
$$F_{seg} = E_{seg}(I; \theta_{seg}), \quad F_{depth} = E_{depth}(I; \theta_{depth}), \quad \ldots$$
各编码器参数独立，$\theta_{seg} \cap \theta_{depth} = \emptyset$。

**变化点**: 独立编码器参数量随任务数线性增长，且无法利用其他任务的梯度信号更新自身表示；手术场景视觉结构高度规律化（器械、组织、背景空间分布稳定），共享特征的泛化收益应大于任务冲突的损耗。

**本文公式（推导）**:
$$\text{Step 1}: F = E_{shared}(I; \theta_{shared}), \quad \theta_{shared} \text{唯一编码器参数} \quad \text{参数共享，压缩模型规模}$$
$$\text{Step 2}: \frac{\partial L_{total}}{\partial \theta_{shared}} = \sum_t \lambda_t \frac{\partial L_t}{\partial F} \cdot \frac{\partial F}{\partial \theta_{shared}} \quad \text{多任务梯度聚合，隐式迁移}$$
$$\text{最终}: \theta_{shared}^{*} = \text{arg}\min_{\theta_{shared}} \mathbb{E}_{(I, \{Y_t\})} \left[ \sum_t \lambda_t L_t(E_{shared}(I; \theta_{shared})) \right]$$

**对应消融**: Table 3 显示四任务联合训练的共享编码器在分割mIoU、深度误差、动作准确率上均优于该编码器单独训练任一任务，证明正向迁移真实存在。

---

### 模块 3: 深度估计弱监督适配（对应框架图 深度头部分）

**直觉**: 手术场景密集深度真值标注成本极高，需利用立体视频的几何约束或稀疏深度采样作为替代监督信号。

**Baseline 公式** (全监督深度估计):
$$L_{depth}^{full} = \frac{1}{|M|} \sum_{(i,j) \in M} | \hat{D}_{ij} - D_{ij}^{gt} |, \quad M \text{为全部像素掩码}$$

**变化点**: 手术数据集中 $|M_{gt}| \ll |M|$，密集真值 $D^{gt}$ 不可得；需改用稀疏标注 $D^{sparse}$ 或立体匹配生成的伪标签 $\tilde{D}^{stereo}$。

**本文公式（推导）**:
$$\text{Step 1}: \tilde{D}_{ij} = \begin{cases} D_{ij}^{sparse} & (i,j) \in M_{sparse} \\ f_{stereo}(I_L, I_R)_{ij} & \text{otherwise} \end{cases} \quad \text{混合监督：稀疏真值+立体伪标签}$$
$$\text{Step 2}: L_{depth} = \frac{1}{|M_{valid}|} \sum_{(i,j) \in M_{valid}} | \hat{D}_{ij} - \tilde{D}_{ij} | + \lambda_{grad} \cdot \| \nabla \hat{D} - \nabla \tilde{D} \|_1 \quad \text{加入梯度一致性约束边缘对齐}$$
$$\text{最终}: L_{depth}^{final} = L_{depth} \cdot \mathbb{1}[|M_{valid}| > \tau] \quad \text{有效像素数阈值过滤，避免低质量伪标签主导}$$

**对应消融**: 论文未明确报告深度弱监督信号质量的消融，为潜在局限。

## 实验与分析

| Method | 语义分割 mIoU ↑ | 深度估计误差 ↓ | 器械检测 AP ↑ | 动作识别 Acc ↑ |
|:---|:---|:---|:---|:---|
| 单任务基线（独立编码器） |  |  |  |  |
| 本文四任务联合 | 优于单任务 | 优于单任务 | 略低于专用SOTA | 优于单任务 |
| Δ (联合 vs. 单任务) | + | - | - | + |

核心实验证据支持多任务互补效应：Table 2 显示联合训练在语义分割、深度估计、动作识别三个任务上均优于对应单任务基线，说明共享表示有益于整体理解。Table 3 的消融实验是关键证据——增加任务数量单调提升各任务性能，四任务模型最优，验证了「任务数量-性能正相关」的核心主张。

然而需审慎解读：
- **器械检测为明显短板**：AP略低于专用单任务SOTA，体现多任务学习经典的性能权衡（negative transfer或优化冲突），作者未深入分析此现象成因；
- **基线公平性存疑**：单任务基线的超参数调优程度未说明，若基线未经充分调优则优势可能被夸大；
- **缺失关键比较**：未与近年强力多任务学习方法 MTI-Net、TaskPrompter 等直接比较，无法定位本文在通用多任务学习谱系中的位置；
- **部署性能未明**：未报告推理速度（FPS），实时手术导航场景下的部署可行性不明；
- **泛化性有限**：消融仅在单一数据集上进行，跨数据集（CholecT50 vs. Cholec80）及跨手术类型（机器人手术 vs. 腹腔镜）的迁移能力未测试；
- **深度弱监督风险**：伪标签质量及其对共享特征的潜在负面影响未充分讨论，可能引入误差传播。

## 方法谱系与知识库定位

**方法家族**: 多任务学习（Multi-Task Learning, MTL）→ 计算机视觉通用场景理解 → 手术视频领域适配

**父方法**: 标准硬参数共享多任务学习（Hard Parameter Sharing, Caruana 1997），具体实现借鉴 SegFormer/Swin Transformer 作为共享编码器，各任务头为对应领域的标准解码器设计。

**改动槽位**:
- **架构**: 单一共享Transformer编码器 + 四任务专用头（分割/深度/检测/动作）
- **目标**: 加权多任务损失，深度分支引入弱监督适配
- **训练_recipe**: 联合端到端训练，梯度共享隐式迁移
- **数据_curation**: 利用立体视频几何约束生成深度伪标签
- **推理**: 单次前向四任务并行输出

**直接基线与差异**:
- **单任务专用模型**（如独立SegFormer分割、独立DPT深度）：本文以参数共享压缩模型，以联合训练实现迁移；
- **通用多任务框架 MTI-Net/TaskPrompter**: 本文未显式比较，推测差异在于未引入跨任务注意力/提示机制，更轻量但交互能力更弱；
- **手术视频专用方法**（如CholecNet系列）：本文首次将四任务统一于单一框架，而非针对单一任务优化。

**后续方向**:
1. 引入显式跨任务交互模块（如任务注意力、特征蒸馏）缓解检测任务性能权衡；
2. 探索任务动态加权（如GradNorm、Uncertainty Weighting）替代固定超参数 $\lambda$；
3. 向机器人手术（如da Vinci数据集）迁移验证跨术式泛化性。

**标签**: 视频理解 / 多任务学习 / 手术场景 / 硬参数共享 / 弱监督深度估计 / 资源受限标注
