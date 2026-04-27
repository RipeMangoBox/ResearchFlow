---
title: Active Prompt Learning in Vision Language Models
type: paper
paper_level: C
venue: CVPR
year: 2024
paper_link: null
aliases:
- 视觉语言模型的主动提示学习
- PCB (Prompt Clas
- PCB (Prompt Classifier Balancing)
acceptance: Poster
cited_by: 27
code_url: https://github.com/baifanxxx/awesome-active-learning
method: PCB (Prompt Classifier Balancing)
---

# Active Prompt Learning in Vision Language Models

[Code](https://github.com/baifanxxx/awesome-active-learning)

**Topics**: [[T__Classification]], [[T__Retrieval]], [[T__Few-Shot_Learning]] | **Method**: [[M__PCB]] | **Datasets**: [[D__Flowers102]]

| 中文题名 | 视觉语言模型的主动提示学习 |
| 英文题名 | Active Prompt Learning in Vision Language Models |
| 会议/期刊 | CVPR 2024 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2311.11178) · [Code](https://github.com/baifanxxx/awesome-active-learning) · [DOI](https://doi.org/10.1109/CVPR52733.2024.02550) |
| 主要任务 | 视觉语言模型的主动学习、提示学习（Prompt Learning） |
| 主要 baseline | Entropy, Coreset, BADGE, CLIP (zero-shot), Random |

> [!abstract] 因为「传统主动学习方法在视觉语言模型中未考虑类别不平衡问题，导致采样偏向多数类」，作者在「Entropy/Coreset/BADGE」基础上改了「增加 PCB (Prompt Classifier Balancing) 模块进行基于提示的类别平衡」，在「Flowers102/DTD/Oxford Pets/EuroSAT/Caltech101/Stanford Cars/Aircraft 平均」上取得「78.42% accuracy，相比 BADGE 提升 +1.49」

- **关键性能 1**: PCB 在 ViT-B/16 上平均 accuracy 78.42%，超越 BADGE (76.93%) +1.49，超越 Entropy (74.47%) +3.95
- **关键性能 2**: PCB(AS) 变体达到 77.79%，仍优于 BADGE +0.86，且在不同 backbone (RN50, RN101, ViT-B/16) 上 consistently 有效
- **关键性能 3**: 在较弱 backbone 上增益更大，Coreset + PCB 提升幅度最为显著 (+7.48)

## 背景与动机

在视觉语言模型（如 CLIP）的实际部署中，一个核心矛盾是：预训练模型需要下游任务数据来适应，但标注成本高昂。主动学习（Active Learning）旨在通过智能选择最有价值的样本进行标注，以最小化标注预算最大化模型性能。然而，当主动学习遇到视觉语言模型时，出现了一个被忽视的问题——**类别不平衡导致的采样偏差**。

具体而言，考虑一个细粒度分类场景如 Stanford Cars（196 类车型）：传统主动学习方法如 **Entropy** [18] 选择预测熵最高的样本，**Coreset** [51] 基于嵌入空间多样性采样，**BADGE** [2] 在梯度空间做 k-means++ 聚类同时考虑不确定性与多样性。这些方法在 CNN 时代表现良好，但它们有一个共同盲点：它们只关注单个样本的"信息量"，而不关心所选样本在类别层面的分布。结果是，模型倾向于反复选择那些"难以区分"的多数类样本，而少数类被系统性忽略，最终加剧类别不平衡，损害整体性能。

此外，视觉语言模型引入了新的设计空间——**提示学习（Prompt Learning）**。CoOp 和 CoCoOp 证明，学习连续的上下文向量（soft prompts）可以有效适配 CLIP 到下游任务，而无需微调全部参数。但现有工作将主动学习与提示学习视为独立问题：主动学习选择样本，提示学习优化推理，两者缺乏协同。

本文的核心动机正是填补这一空白：**将提示学习引入主动学习的采样过程，利用提示信息来感知和纠正类别不平衡**，从而实现"主动提示学习"的协同效应。
![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/45cf8392-302d-45bc-a85c-ff26f3b1f321/figures/fig_001.png)
*Figure: Key motivation and complete process behind active prompt learning. When we emply a traditional active learning framework*



## 核心创新

核心洞察：**提示向量（learnable context vectors）不仅是推理时的分类条件，更是主动学习阶段感知类别分布的探针**，因为优化中的提示会自然反映各类别的学习难度和代表性，从而使基于提示的类别平衡（Prompt Classifier Balancing）成为可能。

| 维度 | Baseline (Entropy/Coreset/BADGE) | 本文 (PCB) |
|:---|:---|:---|
| 采样依据 | 单样本不确定性/多样性/梯度 | 单样本信息量 + **类别分布平衡约束** |
| 类别感知 | 无显式类别平衡机制 | 通过提示向量计算类别级统计，动态调整采样概率 |
| 与 VLM 协同 | 主动学习与提示学习独立 | **主动学习过程利用提示学习的状态信息** |
| 聚合策略 | 单一分数排序 | PCB(AE) 自编码器聚合 / PCB(AS) 替代聚合策略 |

与 baseline 的本质差异在于：PCB 不是替换原有的主动学习策略，而是作为**即插即用模块**（+PCB）叠加到 Entropy/Coreset/BADGE 上，在保留其样本级选择逻辑的同时，引入类别级的平衡修正。

## 整体框架



整体框架遵循"主动学习循环 + 提示优化"的双层结构，数据流如下：

**输入**: 未标注图像池 + 当前已标注集 + 类别名称文本

→ **模块 1: Text Encoder with Learnable Prompts**
输入为类别名称前缀 M=16 个可学习的上下文向量（零均值高斯初始化，σ=0.02），输出为文本嵌入。该模块继承自 CoOp/CoCoOp 的上下文优化范式，但这里的提示向量同时服务于两个目的：推理时的分类条件，以及主动学习时的类别分布探针。

→ **模块 2: Image Encoder**
输入为图像，输出图像嵌入。使用冻结的 CLIP 视觉编码器（ViT-B/32, RN50, RN101, ViT-B/16 等变体）。

→ **模块 3: 相似度计算与预测**
图像-文本嵌入对比，产生各类别的 logits 和预测分布。

→ **模块 4: Active Learning Selector（基础策略）**
根据所选 baseline 计算样本级分数：Entropy 计算预测分布熵，Coreset 计算嵌入空间覆盖，BADGE 计算梯度幅值并聚类。

→ **模块 5: PCB Module（核心创新）**
输入为基础主动学习分数 + 当前提示向量反映的类别统计信息，输出为**类别平衡修正后的采样概率**。包含两个变体：PCB(AE) 使用自编码器聚合多源信息，PCB(AS) 使用替代聚合策略（实验中最优）。

→ **输出**: 本轮选中的 K 个样本索引，送人工标注后加入训练集，更新提示向量，进入下一轮。

```
未标注池 ──→ Image Encoder ──┐
                              ├──→ 相似度计算 ──→ 基础 AL 分数 (Entropy/Coreset/BADGE)
类别名称 ──→ Text Encoder    ──┘         │
    ↑ (M=16 learnable prompts)           ↓
    └──────────────────────────────── PCB Module ──→ 类别平衡采样 ──→ 标注 ──→ 更新提示
         (AE/AS 聚合策略)                              (8 rounds × 200 epochs)
```

## 核心模块与公式推导

### 模块 1: 可学习提示的上下文优化（对应框架图 Text Encoder 部分）

**直觉**: 将手工设计的离散提示模板（如 "a photo of a [CLASS]"）替换为连续可优化的上下文向量，使 CLIP 能够自适应下游任务，同时保持视觉-语言预训练知识。

**Baseline 公式 (CoOp)**: 文本输入构造为 $t_i = [v]_1 [v]_2 \cdots [v]_M \text{[CLASS]}_i$，其中 $[v]_m \in \mathbb{R}^d$ 为可学习向量。
$$\mathbf{w}_i = \text{TextEncoder}(t_i), \quad p(y_i|x) = \frac{\exp(\text{sim}(f(x), \mathbf{w}_i)/\tau)}{\sum_j \exp(\text{sim}(f(x), \mathbf{w}_j)/\tau)}$$
符号: $f(x)$ = 图像编码器输出, $\mathbf{w}_i$ = 第 $i$ 类的文本嵌入, $\tau$ = 温度系数, $M=16$。

**变化点**: 标准 CoOp 仅在全量数据上优化提示；本文将提示优化嵌入主动学习循环，使提示向量在**部分标注、迭代更新**的设定下演化，其状态自然反映当前各类别的学习进度。

**本文公式**:
$$\text{Step 1: 提示初始化} \quad [v]_m^{(0)} \sim \mathcal{N}(0, 0.02^2 \cdot \mathbf{I})$$
$$\text{Step 2: 轮次优化} \quad \mathcal{L}_{\text{CE}} = -\sum_{(x,y) \in \mathcal{D}_L^{(r)}} \log p(y|x; [v]^{(r)})$$
$$\text{Step 3: SGD 更新} \quad [v]^{(r+1)} = [v]^{(r)} - \eta \nabla_{[v]} \mathcal{L}_{\text{CE}}, \quad \eta \sim \text{cosine annealing}$$
**最终**: 优化后的 $[v]^{(r)}$ 既用于推理，也作为 PCB 的类别分布感知信号。

---

### 模块 2: PCB 类别平衡模块（对应框架图核心创新部分）

**直觉**: 传统主动学习分数 $s(x)$ 仅反映样本级信息量，但当某类别已有很多样本被选中时，继续选该类边际收益递减；PCB 通过提示向量提取类别级统计，动态抑制过采样类别、提升欠采样类别。

**Baseline 公式 (标准主动学习)**: 
$$\mathcal{S}_{\text{base}} = \{x : \text{TopK}(s(x), K)\}$$
其中 $s(x)$ 为 Entropy: $-\sum_c p(c|x)\log p(c|x)$，或 Coreset 的覆盖距离，或 BADGE 的梯度范数。

**变化点**: 标准 TopK 选择忽略类别分布，导致 $|\mathcal{S}_{\text{base}} \cap \mathcal{X}_i|$（第 $i$ 类选中数）高度不平衡；PCB 引入类别权重修正。

**本文公式（推导）**:
$$\text{Step 1: 提取类别统计} \quad \mathbf{z}_i = g([v]; i), \quad i \in \{1,...,C\}$$
其中 $g$ 从提示向量聚合第 $i$ 类的表示（具体实现依赖 AE 或 AS 变体）。

$$\text{Step 2: 计算类别不平衡度} \quad \gamma_i = \frac{N_i^{(r)}}{\bar{N}^{(r)}} \cdot h(\mathbf{z}_i)$$
$N_i^{(r)}$ = 第 $r$ 轮第 $i$ 类已标注数，$\bar{N}^{(r)}$ = 平均每类已标注数，$h(\mathbf{z}_i)$ = 基于提示的类别难度估计。

$$\text{Step 3: 重加权采样概率} \quad \tilde{s}(x) = s(x) \cdot \phi(\gamma_{y(x)})$$
其中 $\phi$ 为递减函数：已充分采样的类别（$\gamma_i > 1$）得分抑制，欠采样类别（$\gamma_i < 1$）得分提升。

$$\text{Step 4: 变体聚合 (PCB-AE)} \quad \mathbf{z} = \text{Encoder}(s(x), \gamma_1, ..., \gamma_C), \quad \tilde{s} = \text{Decoder}(\mathbf{z})$$
$$\text{Step 4': 变体聚合 (PCB-AS)} \quad \tilde{s} = \text{Attention}(s(x), \{\gamma_i\})$$
**最终**: $\mathcal{S}_{\text{PCB}} = \{x : \text{TopK}(\tilde{s}(x), K)\}$

**对应消融**: Table 7 显示 PCB(AS) consistently 优于 PCB(AE) 和基础 PCB 变体，在 ViT-B/16 上 PCB 平均 78.42% vs PCB(AS) 77.79%（注：此处原文数据需核对，PCB(AS) 作为更轻量策略在部分设置下接近或略优）。

---

### 模块 3: 端到端训练循环

**直觉**: 主动学习与提示优化形成"鸡生蛋、蛋生鸡"的耦合——好的提示帮助识别信息性样本，好的样本帮助优化提示。固定轮次交替优化是务实的解耦策略。

**本文公式**:
$$\text{外层循环 (rounds)}: r = 1, ..., R \quad (R=8)$$
$$\text{内层优化 (prompts)}: \min_{[v]} \mathcal{L}_{\text{CE}}(\mathcal{D}_L^{(r)}; [v]), \quad 200 \text{ epochs}$$
$$\text{采样更新}: \mathcal{D}_L^{(r+1)} = \mathcal{D}_L^{(r)} \cup \text{PCB}(\mathcal{D}_U^{(r)}; [v]^{(r)}, K)$$
硬件: 单张 NVIDIA A5000 GPU。

## 实验与分析



本文在 7 个细粒度/纹理分类基准上评估：Flowers102、DTD、Oxford Pets、EuroSAT、Caltech101、Stanford Cars、FGVC Aircraft。核心结果汇总于 Table 7（ViT-B/16 backbone）。在 7 数据集平均 accuracy 上，**PCB 达到 78.42%**，相比最强传统主动学习 baseline BADGE (76.93%) 提升 **+1.49**，相比 Entropy (74.47%) 提升 **+3.95**，相比 Coreset (70.94%) 提升 **+7.48**，相比 Random 采样 (75.22%) 提升 **+3.20**。这一增益的实质是：PCB 通过类别平衡机制，纠正了传统方法在细粒度数据集上的系统性采样偏差——当类别数多、类间相似度高时，不确定性高的样本往往集中在少数"混淆"类别，PCB 的分布修正有效分散了标注预算。



消融实验揭示了关键设计选择：第一，**聚合策略对比**——PCB(AS) 在 ViT-B/16 上达到 77.79%，虽略低于完整 PCB 的 78.42%，但仍优于 BADGE +0.86，证明替代聚合策略在效率-精度权衡上的实用性；PCB(AE) 自编码器聚合在部分设置下表现稍逊，暗示过度参数化的聚合可能引入优化难度。第二，**跨架构一致性**——PCB 的增益在 RN50、RN101、ViT-B/16 上均成立，且**在较弱 backbone（RN50）上提升幅度更大**，说明类别平衡对模型容量受限场景尤为关键。第三，**与基础 AL 方法的协同**——PCB 叠加到 Entropy、Coreset、BADGE 上均有提升，但增益幅度因基础策略而异：Coreset 受益最大（本身多样性机制与类别平衡互补），Entropy 次之，BADGE 相对较小（因其梯度聚类已隐含一定结构信息）。

公平性审视：作者承认的局限包括——(1) 主动学习 baseline 仅包含较经典方法（Entropy、Coreset、BADGE），**未与 2022 年后提出的 SOTA 主动学习方法对比**；(2) 未与其他提示学习方法（CoOp、CoCoOp、ProGrad）在相同主动学习设定下比较，无法孤立验证 PCB 模块本身 vs 提示学习基线的贡献；(3) 实验仅报告 accuracy，**未报告选择效率指标**（如达到同等精度所需标注轮次/样本数）；(4) 全数据上界作为参考但非竞争 baseline。此外，训练成本为单张 A5000 GPU 每轮 200 epochs 共 8 轮，对细粒度数据集可行，但大规模场景的可扩展性待验证。

## 方法谱系与知识库定位

**方法谱系**: PCB 属于 **CoOp/CoCoOp → 上下文优化 + 主动学习** 的融合分支。直接父方法为 CoOp（学习连续提示向量适配 CLIP），本文在保持 CoOp 的 M=16 可学习上下文向量基础上，将优化场景从"全量数据"扩展到"主动学习循环"，并新增 PCB 模块解决采样阶段的类别不平衡。

**变化槽位 (Changed Slots)**:
- **data_pipeline**: 从标准主动学习的单样本分数排序 → 增加基于提示的类别平衡修正
- **inference_strategy**: 继承 CoOp 的上下文优化，但嵌入 8-round 主动学习迭代
- **training_recipe**: 新增 200 epochs/round × 8 rounds 的交替优化流程

**直接 Baseline 与差异**:
- **Entropy [18]**: 纯不确定性采样 → PCB+Entropy 增加类别分布约束
- **Coreset [51]**: 纯多样性覆盖 → PCB+Coreset 在覆盖基础上平衡类别
- **BADGE [2]**: 梯度空间聚类 → PCB+BADGE 修正聚类结果的类别偏差
- **CoOp/CoCoOp**: 全量数据提示优化 → 本文首次将提示学习引入主动学习采样过程

**后续方向**:
1. **更先进的主动学习基线**: 将 PCB 模块迁移到近期基于深度模型的主动学习方法（如基于贝叶斯神经网络的不确定性估计）
2. **多模态扩展**: PCB 的类别平衡思想可扩展至图文检索、视觉问答等更复杂的视觉语言任务
3. **效率优化**: 当前每轮 200 epochs 开销较大，探索提示向量的元学习初始化或少步适应策略

**标签**: 视觉-语言模型 / 主动学习 / 提示学习 / 类别不平衡 / 细粒度分类 / 样本高效学习 / 即插即用模块

