---
title: Towards 3D Objectness Learning in an Open World
type: paper
paper_level: A
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 开放世界3D对象性学习的无提示检测
- OP3Det
- OP3Det achieves class-agnostic open
acceptance: Poster
code_url: https://op3det.github.io/
method: OP3Det
modalities:
- 3D point cloud
- RGB image
paradigm: supervised
baselines:
- 无监督视觉特征学习的DINOv2_DINOv2
---

# Towards 3D Objectness Learning in an Open World

[Code](https://op3det.github.io/)

**Topics**: [[T__Object_Detection]], [[T__Few-Shot_Learning]] | **Method**: [[M__OP3Det]] | **Datasets**: Open-world 3D object, 2D open-world

> [!tip] 核心洞察
> OP3Det achieves class-agnostic open-world 3D object detection without hand-crafted text prompts by leveraging 2D foundation model priors and a cross-modal mixture of experts that dynamically routes uni-modal and multi-modal features.

| 中文题名 | 开放世界3D对象性学习的无提示检测 |
| 英文题名 | Towards 3D Objectness Learning in an Open World |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2510.17686) · [Code](https://op3det.github.io/) · [Project](https://op3det.github.io/) |
| 主要任务 | 开放世界3D目标检测 (Open-world 3D Object Detection)、类无关检测 (Class-agnostic Detection) |
| 主要 baseline | SAM3D、SAM、LDET、OLN、SOS、Mask R-CNN、VoteNet、FCAF3D、CenterPoint |

> [!abstract] 因为「传统3D检测器只能识别预定义类别，开放词汇方法依赖文本提示且受限于语义重叠」，作者在「SAM3D等2D-3D投影方法」基础上改了「引入多尺度点采样结合LDET整体对象性过滤，并设计跨模态混合专家网络动态路由点云与RGB特征」，在「SUN RGB-D / ScanNet开放世界3D检测基准」上取得「AP 13.9（超SOS +5.0）、F1 21.0（超SOS +6.5）」

- **AP 13.9 / AR100 42.9 / F1 21.0**：在SUN RGB-D/ScanNet上全面超越现有开放世界3D检测方法，F1较SOS提升+6.5
- **多尺度点采样**：相比单尺度策略AP提升7.6%，有效解决单一阈值τ的噪声-保留权衡
- **跨模态MoE**：动态路由uni-modal与multi-modal特征路径，较固定融合机制显著提升泛化性

## 背景与动机

现有3D目标检测系统面临一个根本困境：封闭集检测器（如VoteNet、PointRCNN系列）只能识别训练时预定义的类别，遇到未见过的新对象便束手无策；而开放词汇方法（如PointCLIP、OV-Uni3DETR）虽能扩展类别，却严重依赖人工设计的文本提示，且受限于语义重叠问题——当新类别与基类语义相近时，模型容易混淆。更关键的是，这些方法本质上仍是"类特定"的，需要知道要检测什么。

以室内场景为例：一个服务机器人进入新环境，可能遇到训练时从未见过的家具、装饰品或工具。理想情况下，它应能无提示地发现"任何看起来像物体的区域"，而非等待用户输入"请找一下那个奇怪的金属支架"。这正是**3D对象性学习（3D Objectness Learning）**的核心诉求：学习通用的"物体性"先验，而非特定类别的语义映射。

现有方法如何处理这一问题？**SAM3D**将SAM的2D掩码投影到3D空间，但依赖时序融合过滤噪声，无法应用于单帧检测场景；**OW3DET**尝试开放世界3D检测，但仍需类别信息驱动；**CODA**通过跨模态对齐发现新框，却受限于文本提示的词汇扩展。它们的共同短板在于：要么需要预定义类别或文本提示（非真正"无提示"），要么单帧场景下噪声过滤能力不足（SAM产生大量碎片化掩码）。

本文的核心动机正是填补这一空白：构建首个**类无关、无提示、单帧**的开放世界3D检测器，让模型从2D基础模型的丰富语义先验中学习通用的3D对象性，无需任何人工语义监督。
![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/d7d765d3-af00-449c-87a0-6d135841b7e8/figures/Figure_1.png)



## 核心创新

核心洞察：**2D基础模型的语义先验与整体对象性理解可以迁移为3D空间中的通用对象性信号**，因为SAM的密集掩码提供了丰富的候选区域，LDET的全局评分补充了SAM局部分割的盲区，从而使单帧、无提示、类无关的3D对象发现成为可能。

| 维度 | Baseline (SAM3D/传统开放词汇) | 本文 (OP3Det) |
|:---|:---|:---|
| **推理策略** | 需文本提示或预定义类别；SAM3D需时序融合 | 完全无提示、类无关，单帧即完成检测 |
| **2D-3D转换** | 单尺度点采样，固定阈值τ，噪声与漏检权衡困难 | 多尺度点采样聚合多阈值结果，联合LDET整体过滤 |
| **特征融合** | 早期/晚期固定融合或单模态处理 | 跨模态MoE动态路由，自适应选择uni/multi-modal路径 |

这一设计使OP3Det成为首个不依赖任何语义标签或文本提示的3D开放世界检测器，将"对象性"本身作为可学习的跨模态表示。

## 整体框架


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/d7d765d3-af00-449c-87a0-6d135841b7e8/figures/Figure_2.png)
*Figure 2: Figure 2: The overview of OP3Det. We apply SAM to introduce abundant 2D semantic knowledgefor 3D object discovery. Multi-scale point sampling is utilized in this process. The cross-modal MoEis then em*



OP3Det的完整数据流遵循"2D先验提取 → 3D提议生成 → 跨模态融合 → 类无关检测"的四阶段范式：

**阶段一：2D语义先验提取**。输入RGB图像，经SAM生成密集2D分割掩码集合M。SAM的"分割一切"能力提供类无关的候选区域，但掩码质量参差不齐，包含大量碎片化噪声。

**阶段二：多尺度3D提议生成（核心创新模块）**。对SAM掩码执行**多尺度点采样**：在多个置信度阈值{τ₁, τ₂, ..., τₖ}下分别采样3D点，聚合为候选提议集合。此步骤较单尺度策略显著扩展对象覆盖。随后引入**LDET整体过滤**：利用LDET 2D检测器的 holistic objectness score 对提议进行质量重排序，滤除SAM过度分割产生的伪对象，保留真正有整体性的3D实体。

**阶段三：跨模态特征融合（核心创新模块）**。并行的点云分支提取几何特征，RGB分支提取外观特征，二者输入**Cross-Modal Mixture of Experts (MoE)**。门控网络根据输入动态计算各专家权重，自适应选择仅点云、仅RGB或融合特征路径，输出最优的跨模态表示。

**阶段四：类无关检测头**。基于融合特征直接回归3D边界框，全程无分类层、无文本编码器，实现真正的prompt-free检测。

```
RGB Image ──→ SAM ──→ Multi-scale Point Sampling ──→ LDET Filtering ──┐
                                                                      ├──→ Cross-Modal MoE ──→ 3D Box Head
Point Cloud ──────────────────────────────────────────────────────────┘
         (geometric features)                    (appearance features)
```

## 核心模块与公式推导

### 模块一：多尺度点采样（对应框架图阶段二）

**直觉**：单一阈值τ的点采样必然面临噪声-保留权衡——高τ过滤噪声但漏掉低置信度真对象，低τ保留对象但引入大量碎片。多尺度聚合可兼得二者之长。

**Baseline公式** (SAM3D单尺度采样):
$$\mathcal{P}_{single} = \text{SamplePoints}(\mathcal{M}, \tau)$$
符号: $\mathcal{M}$ = SAM生成的2D掩码集合, $\tau$ = 固定置信度阈值, $\mathcal{P}$ = 3D点云提议集合

**变化点**：单尺度策略在实验中显示"无论τ如何选择，AP均下降"——固定阈值无法适应场景中对象的多样性。本文改为多阈值集成，并引入LDET整体对象性评分补充几何过滤。

**本文公式（推导）**:
$$\text{Step 1}: \mathcal{P}_{multi} = \text{bigcup}_{i=1}^{k} \text{SamplePoints}(\mathcal{M}, \tau_i) \quad \text{在} k \text{个阈值下分别采样并聚合}$$
$$\text{Step 2}: s_{holistic} = \text{LDET}(I, mask_j) \quad \text{用LDET计算每个掩码的整体对象性分数}$$
$$\text{Step 3}: \mathcal{P}_{refined} = \text{TopK}(\mathcal{P}_{multi}, \alpha \cdot s_{SAM} + \beta \cdot s_{LDET} + \gamma \cdot s_{geo}) \quad \text{加权融合后筛选}$$
$$\text{最终}: \mathcal{P}_{final} = \{p \in \mathcal{P}_{refined} \text{mid} \text{score}(p) > \delta\}$$

**对应消融**：Table 5显示，去掉多尺度点采样（改用单尺度）AP下降7.6%；去掉LDET后AR显著低于直接SAM，AP亦受损。

### 模块二：跨模态混合专家网络（对应框架图阶段三）

**直觉**：点云与RGB的互补性随场景动态变化——纹理丰富区域RGB更可靠，几何结构复杂区域点云更稳定。固定融合权重无法适应这种变化，需动态路由。

**Baseline公式** (传统早期/晚期融合):
$$\mathbf{z} = f_{fusion}(\mathbf{x}_{PC}, \mathbf{x}_{RGB})$$
符号: $\mathbf{x}_{PC}$ = 点云特征, $\mathbf{x}_{RGB}$ = RGB图像特征, $f_{fusion}$ = 固定融合函数（如拼接后MLP或注意力）

**变化点**：固定融合假设所有样本的最优融合策略相同，忽视了模态间的动态互补性。本文引入MoE结构，让门控网络根据输入内容自适应选择专家组合。

**本文公式（推导）**:
$$\text{Step 1}: \mathbf{x} = [\mathbf{x}_{PC}; \mathbf{x}_{RGB}] \quad \text{拼接多模态特征作为门控输入}$$
$$\text{Step 2}: g_i(\mathbf{x}) = \frac{\exp(\mathbf{W}_g^i \cdot \mathbf{x})}{\sum_{j=1}^{N} \exp(\mathbf{W}_g^j \cdot \mathbf{x})} \quad \text{softmax门控产生归一化路由权重}$$
$$\text{Step 3}: \mathbf{e}_i(\mathbf{x}_i) = \begin{cases} \text{MLP}_{PC}(\mathbf{x}_{PC}) & i \in \mathcal{E}_{PC} \\ \text{MLP}_{RGB}(\mathbf{x}_{RGB}) & i \in \mathcal{E}_{RGB} \\ \text{MLP}_{multi}([\mathbf{x}_{PC}; \mathbf{x}_{RGB}]) & i \in \mathcal{E}_{multi} \end{cases} \quad \text{三类专家分别处理不同模态}$$
$$\text{最终}: \mathbf{z} = \sum_{i=1}^{N} g_i(\mathbf{x}) \cdot \mathbf{e}_i(\mathbf{x}_i) \quad \text{加权聚合专家输出}$$

**对应消融**：Table 6显示，移除跨模态MoE（改用单一固定融合策略）导致性能下降；仅使用点云或仅使用RGB的uni-modal路径均劣于动态路由的完整模型。

## 实验与分析


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/d7d765d3-af00-449c-87a0-6d135841b7e8/figures/Table_1.png)
*Table 1: Table 1: The cross-category performance of OP3Det on the SUN RGB-D and ScanNet dataset.Closed-world 3D detection methods are trained on 3D point clouds with only seen categoriesannotated. Open-vocabul*



本文在SUN RGB-D、ScanNet、KITTI三个主流3D数据集上评估OP3Det。核心基准测试为SUN RGB-D/ScanNet上的类无关开放世界3D检测：模型仅在基类上训练，测试时需同时检测基类和新类，且无任何文本提示。
![Table 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/d7d765d3-af00-449c-87a0-6d135841b7e8/figures/Table_4.png)
*Table 4: Table 4: Comparison with 3D open-vocabulary methods on the SUN RGB-D and ScanNetdataset for open-vocabulary 3D object detection (class-specific). The experimental setting istotally the same as CoDA, a*



Table 1/2的综合结果显示，OP3Det取得AP 13.9、AR100 42.9、F1 21.0的成绩。其中F1 21.0较此前最优方法SOS（F1 14.5）提升+6.5，较GGN（F1 8.4）提升+12.6；AP 13.9较SOS（AP 8.9）提升+5.0，较Mask R-CNN（AP 1.0）提升+12.9。值得注意的是，SAM单独使用时AR100高达48.1（超过OP3Det的42.9），但AP仅3.6、F1仅6.7，说明SAM产生大量低质量提议导致精确率崩溃；OP3Det通过LDET过滤和多尺度策略，以略降的召回换取大幅提升的精确率，最终F1最优。在KITTI跨数据集测试（Table 3）中，模型仅在"car"和"pedestrian"上训练，能泛化到 cyclist、van 等户外新类别，验证了开放世界能力。


![Table 5](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/d7d765d3-af00-449c-87a0-6d135841b7e8/figures/Table_5.png)
*Table 5: Table 5: Ablation Study on the SUN RGB-D dataset. SAM: utilizing SAM for object discovery.PS: multi-scale point sampling in 3D novel object discovery. CM-MoE: cross-modal MoE.*



消融实验（Table 5/6）揭示了各组件的贡献：去掉多尺度点采样改用最优单尺度τ，AP下降7.6%；去掉LDET后AR显著低于原始SAM，说明LDET的整体对象性评分对维持召回至关重要；Table 6的跨模态MoE消融显示，仅使用点云(PC)或仅使用图像(Img)的uni-modal路径均劣于完整MoE，动态路由机制有效利用了模态互补性。

公平性审视：对比基线中，SAM3D需重新实现以提取3D框，可能非其原始设计目标；SAM的高AR/低AP特性表明AR100并非完美指标。本文未与CODA、OV-Uni3DETR等最新开放词汇3D方法直接对比（这些需文本提示，任务设定不同）。训练使用NVIDIA A100 GPU，具体数量与时间在补充材料中。作者披露的失败模式包括：非刚性低对比度区域（如白色窗帘）检测失败，以及复杂场景中细小物体的漏检——这些恰是SAM掩码分割的固有难点向3D的传递。
![Figure 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/d7d765d3-af00-449c-87a0-6d135841b7e8/figures/Figure_4.png)
*Figure 4: Figure 4: The visualized results of OP3Det on the SUN RGB-D (the first row) and ScanNet (thesecond row) dataset. The red boxes are base classes and blue boxes are novel classes.*



## 方法谱系与知识库定位

**方法家族**：2D基础模型驱动的3D开放世界检测 → 父方法为SAM3D（SAM掩码的3D投影）。

**改动槽位**：
- **推理策略**：替换为无提示、类无关、单帧检测
- **数据流水线**：修改为多尺度点采样 + LDET整体过滤
- **架构**：新增跨模态MoE动态路由

**直接基线对比**：
- **SAM3D**：同样使用SAM，但依赖时序融合且做实例分割而非检测；本文改为单帧+检测框+多尺度采样
- **OW3DET [4]**：开放世界3D检测先驱，但需类别驱动；本文完全去除类别依赖
- **CODA [7]**：协同新框发现与跨模态对齐，需文本提示；本文无需任何语义输入
- **OV-Uni3DETR [9]**：循环模态传播的开放词汇检测，文本驱动；本文走"对象性"而非"语义"路线

**后续方向**：(1) 单模态鲁棒性——当前需RGB+点云双输入，探索纯点云的3D对象性学习；(2) 时序扩展——将动态路由MoE与SAM3D的时序融合结合，提升视频场景性能；(3) 与开放词汇方法的桥接——学习通用对象性后，如何高效接入文本提示实现"对象性+语义"的灵活切换。

**知识库标签**：`modality: RGB-D点云` / `paradigm: 无监督预训练+监督微调` / `scenario: 开放世界/新类发现` / `mechanism: 混合专家网络/动态路由/2D-3D知识迁移` / `constraint: 无文本提示/类无关/单帧推理`

## 引用网络

### 直接 baseline（本文基于）

- [[P__无监督视觉特征学习的DINOv2_DINOv2]] _(方法来源)_: DINOv2 provides self-supervised visual features. Likely used as backbone or feat

