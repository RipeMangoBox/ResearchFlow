---
title: 'Linear Differential Vision Transformer: Learning Visual Contrasts via Pairwise Differentials'
type: paper
paper_level: C
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 线性差分视觉Transformer：通过成对差分学习视觉对比
- Visual-Contrast
- Visual-Contrast Attention (VCA)
- Visual-Contrast Attention (VCA) is
acceptance: Poster
cited_by: 1
method: Visual-Contrast Attention (VCA)
modalities:
- Image
paradigm: supervised
baselines:
- InLine：可注入性与局部建模_InLine
- 代理注意力：融合Softmax与_Agent_Attention
---

# Linear Differential Vision Transformer: Learning Visual Contrasts via Pairwise Differentials

**Topics**: [[T__Classification]], [[T__Image_Generation]] | **Method**: [[M__Visual-Contrast_Attention]] | **Datasets**: [[D__ImageNet-1K]]

> [!tip] 核心洞察
> Visual-Contrast Attention (VCA) is a drop-in replacement for MHSA that reduces complexity from O(N²C) to O(NnC) by distilling visual-contrast tokens and using pairwise differential interactions, improving both recognition accuracy and generation quality with negligible overhead.

| 中文题名 | 线性差分视觉Transformer：通过成对差分学习视觉对比 |
| 英文题名 | Linear Differential Vision Transformer: Learning Visual Contrasts via Pairwise Differentials |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2511.00833) · [Code](待补充) · [Project](待补充) |
| 主要任务 | Image Classification, Image Generation |
| 主要 baseline | DeiT, DiT, SiT, PVT, Swin Transformer, CSwin Transformer, Differential Attention [88] |

> [!abstract] 因为「Vision Transformer的MHSA计算二次复杂度的全序列相似度矩阵，浪费计算于弱相关token且无法显式建模判别性差异」，作者在「Multi-Head Self-Attention (MHSA)」基础上改了「用两阶段Visual-Contrast Attention (VCA)替代：Stage I池化生成n个视觉对比token降复杂度至O(NnC)，Stage II引入正负双流差分交互显式放大判别信号」，在「ImageNet-1K分类与生成」上取得「DeiT-Tiny Top-1 72.2% → 75.6% (+3.3)，DiT-S/2 FID 64.6 → 62.3 (↓2.3)」

- **DeiT-Tiny + VCA**: ImageNet-1K Top-1 Accuracy 75.6%，相比vanilla DeiT-Tiny提升 +3.3 percentage points，仅增加 <0.3M 参数、无额外FLOPs
- **DiT-S/2 + VCA**: ImageNet-1K 256×256生成 FID-50K 62.3，相比DiT-S/2 + Stage I only的64.6下降 ↓2.3；相比Differential Attention [88] both stages的63.9下降 ↓1.6
- **SiT-S/2 + VCA**: FID-50K 53.0，相比vanilla SiT-S/2的57.3下降 ↓4.3

## 背景与动机

Vision Transformer (ViT) 的核心组件 Multi-Head Self-Attention (MHSA) 在计算每个token时，需要与全部N个token做query-key交互，形成N×N的相似度矩阵。这种二次复杂度O(N²C)不仅计算昂贵，更严重的是：一张图片中大量背景token或冗余区域之间的相关性被同等对待，而真正区分"猫耳朵"与"狗耳朵"的判别性差异却被淹没在均匀的softmax归一化中。例如，在ImageNet分类中，模型本应聚焦于动物头部特征，但标准注意力可能将过多权重分配给无关的背景纹理。

现有方法如何尝试解决这一问题？**Performer** [4] 通过核技巧将注意力近似为线性复杂度，但牺牲了精确的结构信息；**Differential Attention** [88] 提出正负双流的差分思想，然而直接在完整序列上操作，未解决序列长度本身的瓶颈；**FLatten Transformer** [19] 采用聚焦线性注意力降低复杂度，却缺乏显式的对比推理机制。这些方法或侧重效率、或侧重判别性，未能同时兼顾。

其根本局限在于：**没有一种机制能同时实现序列压缩（降复杂度）与显式对比学习（增强判别性）**。MHSA的均匀归一化本质上是对所有相关性做"平均主义"，而线性注意力方法虽快却丢失了哪些token对真正值得对比的信息。此外，双流设计若缺乏空间感知的结构编码，对比推理将沦为无的放矢。

本文提出Visual-Contrast Attention (VCA)，通过两阶段设计将全局对比token生成与patch级差分注意力解耦，以线性复杂度实现显式视觉对比学习，并作为drop-in模块验证于分类（DeiT/PVT/Swin/CSwin）与生成（DiT/SiT）两类架构。

## 核心创新

核心洞察：视觉判别性信息可以通过「少量全局对比token + 正负双流差分」来显式建模，因为空间池化后的n个对比token已能捕获区域级语义差异，而双流的减法交互天然放大差异信号、抑制冗余，从而使线性复杂度下的显式对比推理成为可能。

| 维度 | Baseline (MHSA / Differential Attention [88]) | 本文 (VCA) |
|:---|:---|:---|
| **复杂度** | O(N²C) 二次；或O(N²C) 仍用全序列 [88] | O(NnC) 线性，n ≪ N，Stage I池化降序列长度 |
| **对比机制** | 无显式对比；或双流但无序列压缩 [88] | 全局对比token + patch级正负差分，两阶段叠加 |
| **位置编码** | 单一套标准位置编码 | 双位置编码 Pos. Str./Neg. Str.，分别编码正负结构 |
| **架构侵入性** | MHSA需完全替换；[88]需修改注意力核 | Drop-in替换，<0.3M额外参数，零额外FLOPs |

## 整体框架



VCA作为MHSA的直接替代模块，保持外部接口不变，内部采用两阶段流水线：

**输入**: 密集查询特征场 Q ∈ ℝ^(N×C)，其中N为原始序列长度（如14×14=196个patch token），C为通道维度。

**Stage I: Global Contrast Token Generation（全局对比token生成）**
- 输入: Q ∈ ℝ^(N×C)
- 操作: 空间池化（spatial pooling）将N个token压缩为n个visual-contrast token，其中n ≪ N（例如n=49或更少）
- 输出: 压缩后的查询表示 Q̃ ∈ ℝ^(n×C)
- 作用: 将后续注意力的序列长度从N降至n，实现线性复杂度O(NnC)的理论基础

**Stage II: Patch-wise Differential Attention（patch级差分注意力）**
- 输入: Stage I输出的visual-contrast token，以及原始空间结构信息
- 操作: 将token分裂为**Positive Stream (Q⁺, K⁺, V⁺)** 和 **Negative Stream (Q⁻, K⁻, V⁻)**，各自配备独立可学习的dual positional embeddings（Pos. Str.与Neg. Str.）
- 交互: 双流通过差分操作融合——注意力输出 = f(Q⁺,K⁺,V⁺) − g(Q⁻,K⁻,V⁻)，显式放大判别性差异、抑制冗余背景
- 输出: 精炼后的注意力特征，增强区域分离能力

**整体数据流**:
```
Input Tokens (N×C)
    ↓
[Stage I] Spatial Pooling → n Visual-Contrast Tokens (n≪N)
    ↓
[Stage II] Split to Q⁺/Q⁻ + Dual Positional Embeddings (Pos.Str./Neg.Str.)
    ↓
    ├── Positive Stream Attention: amplify discriminative features
    └── Negative Stream Attention: suppress redundant correlations
    ↓
Differential Interaction (⊕ positive ⊖ negative)
    ↓
Output: Refined attention with enhanced visual contrasts
```

两阶段设计的关键解耦：Stage I负责「看哪里」（全局语义区域选择），Stage II负责「怎么看」（局部判别性差异计算）。这种解耦使得VCA既能像线性注意力一样高效，又能像对比学习一样具有判别性。

## 核心模块与公式推导

### 模块 1: Visual-Contrast Token Generation（Stage I，对应框架图左侧）

**直觉**: 并非所有N个patch token都值得精细交互，空间池化后的少量token已足以承载全局对比语义，这是复杂度从二次降为线性的关键。

**Baseline 公式** (Standard MHSA):
$$\text{Attn}_{\text{MHSA}} = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V, \quad Q,K,V \in \mathbb{R}^{N \times C}$$
符号: $Q$ = 查询矩阵, $K$ = 键矩阵, $V$ = 值矩阵, $N$ = 序列长度, $d$ = 头维度。复杂度为O(N²C)。

**变化点**: MHSA直接对全序列N做QK^T，大量计算浪费于背景-背景、前景-背景的弱相关对。本文假设：全局对比信息可被压缩到n ≪ N个代表性token中。

**本文公式（推导）**:
$$\text{Step 1}: \tilde{Q} = \text{Pool}(Q), \quad \tilde{Q} \in \mathbb{R}^{n \times C} \quad \text{（空间池化降序列长度，n ≪ N）}$$
$$\text{Step 2}: \text{Attn}_{\text{Stage I}} = f(\tilde{Q}, K, V) \quad \text{（在压缩后的序列上计算注意力，复杂度O(NnC)）}$$
$$\text{最终}: \text{Global Contrast Tokens} = \tilde{Q} \text{ 及其注意力输出}$$

**对应消融**: Table 3显示，仅保留Stage I时DeiT-Tiny达75.4%（vs 75.6%完整），DiT-S/2 FID 64.6（vs 62.3完整），证明全局对比token本身已贡献主要增益，但需Stage II精修。

---

### 模块 2: Positive/Negative Stream Splitting with Differential Interaction（Stage II，对应框架图右侧）

**直觉**: 人类视觉通过" figure-ground "分离来识别物体，VCA用可学习的正负双流模拟这一过程：正流抓取判别性特征，负流抓取冗余特征，差分输出即对比结果。

**Baseline 公式** (Differential Attention [88]):
$$\text{Attn}_{\text{Diff}} = f(Q^+, K^+, V^+) - g(Q^-, K^-, V^-)$$
符号: $Q^+, K^+, V^+$ = 正流查询/键/值, $Q^-, K^-, V^-$ = 负流查询/键/值, $f, g$ = 注意力函数。该公式已具备双流差分思想，但直接在原始序列上操作，无序列压缩，且缺乏空间结构感知的编码。

**变化点**: [88]的局限在于（1）未降低序列长度，复杂度仍为二次；（2）正负流共享相同位置编码，无法分别编码"判别性结构"与"背景结构"的空间分布。本文引入dual positional embeddings，使正负流各自拥有空间先验。

**本文公式（推导）**:
$$\text{Step 1}: Q^+ = \tilde{Q} + E_{\text{pos}}, \quad Q^- = \tilde{Q} + E_{\text{neg}} \quad \text{（加入双位置编码Pos.Str./Neg.Str.以区分空间结构）}$$
$$\text{Step 2}: A^+ = \text{softmax}\left(\frac{Q^+K^{+T}}{\sqrt{d}}\right)V^+, \quad A^- = \text{softmax}\left(\frac{Q^-K^{-T}}{\sqrt{d}}\right)V^- \quad \text{（独立计算正负流注意力）}$$
$$\text{Step 3}: \text{Out}_{\text{Stage II}} = \text{LN}(A^+ - \lambda \cdot A^-) \quad \text{（差分融合，λ为可学习平衡系数；LayerNorm稳定训练）}$$
$$\text{最终 VCA 输出}: \text{VCA}(Q) = \text{StageI}_{\text{global}}(\tilde{Q}) \circ \text{StageII}_{\text{patch}}(Q^+, Q^-)$$
符号: $E_{\text{pos}}, E_{\text{neg}}$ = 正/负结构位置编码, $\lambda$ = 负流抑制强度（可学习）, $\circ$ = 两阶段组合操作（实验显示近似线性叠加）。

**对应消融**: Table 4显示，去掉Emb. Pos. Str.（仅用Pool）时性能降至75.1% / FID 63.7；而Pool+Emb.配置为最优（75.6% / 62.3），证明dual positional embeddings对对比推理不可或缺。Table 3进一步显示，将VCA替换为Diff[88] both stages时降至75.1% / 63.9，验证两阶段设计优于原始差分注意力。

---

### 模块 3: Two-Stage Composition（两阶段组合，对应完整VCA模块）

**直觉**: 全局对比提供"粗定位"，patch差分提供"精判别"，两者线性叠加即可实现1+1>2的效果。

**Baseline**: 单阶段MHSA或单阶段Diff Attention均无法同时获得效率与判别性。

**本文公式**:
$$\text{VCA} = \underbrace{\text{Pool}(Q) \rightarrow \tilde{Q}}_{\text{Stage I: 全局对比，降复杂度}} \rightarrow \underbrace{\text{Split}(\tilde{Q}) \text{xrightarrow}{E_{\text{pos}}/E_{\text{neg}}} (Q^+, Q^-) \rightarrow A^+ - \lambda A^-}_{\text{Stage II: patch差分，增强判别性}}$$

**对应消融**: Table 3中Stage I only (75.4% / 64.6) + Stage II only (75.5% / 64.3) ≈ 完整VCA (75.6% / 62.3)，显示两阶段贡献近似正交且叠加，完整组合显著优于任一单阶段。

## 实验与分析


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/5b0ba9be-b6c0-4abe-889a-eee862a9b392/figures/Table_1.png)
*Table 1 (result): Image classification results on ImageNet.*



本文在ImageNet-1K上评估了VCA的两类任务：**图像分类**（Table 1）与**类条件图像生成**（Table 2）。分类实验覆盖plain ViT（DeiT）、hierarchical ViT（PVT、Swin、CSwin）四类架构；生成实验覆盖diffusion模型DiT与flow模型SiT的多尺寸配置（S/8, S/4, S/2, B/8, B/4, B/2）。

**分类任务 headline**: VCA-DeiT-Tiny在ImageNet-1K上达到Top-1 Accuracy 75.6%，相比vanilla DeiT-Tiny的72.2%提升+3.3 percentage points，且仅增加0.3M参数（5.7M → 6.0M）、零额外FLOPs。在hierarchical架构上，PVT-Tiny增益最大（+3.1），Swin-T与CSwin-T增益为+0.4~+1.0，显示VCA对轻量plain ViT的提升更为显著。

**生成任务 headline**: VCA在DiT-S/2上将FID-50K从64.6（Stage I only）降至62.3，在SiT-S/2上从57.3降至53.0（↓4.3），在SiT-B/2上从35.3降至32.7（↓2.6）。这些改进在256×256与512×512分辨率、不同模型尺寸上保持一致趋势，证明VCA对diffusion与flow两类生成范式均有效。


![Table 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/5b0ba9be-b6c0-4abe-889a-eee862a9b392/figures/Table_3.png)
*Table 3 (ablation): Ablations on detailed model architecture across image classification and generation tasks.*



**消融分析**（Table 3与Table 4）揭示了关键组件的贡献：
- **Stage II（patch差分）** 的独立贡献：DeiT-Tiny 75.5%（仅Stage II）vs 75.6%（完整），DiT-S/2 FID 64.3 vs 62.3；单独Stage II已接近完整性能，但与Stage I组合后生成质量显著提升（-2.0 FID），说明两阶段存在协同效应。
- **Dual positional embeddings** 的关键作用：Table 4显示，去掉Pos. Str. embedding（仅用Pool）导致分类降至75.1%、生成FID恶化至63.7；而最优配置Pool+Emb.（Pos. Str.）达到75.6%/62.3。尝试将Neg. Str.也改为Pool或Pool+Emb.的变体均不如本文设计，验证了正负流需不对称的空间编码策略。
- **与Differential Attention [88]的对比**：将VCA两阶段均替换为[88]的原始差分注意力，结果为75.1%/63.9，显著低于VCA的75.6%/62.3，证明Stage I的token池化与Stage II的精修设计均不可或缺。

**公平性检查**: 本文比较了DeiT、PVT、Swin、CSwin等识别基线与DiT、SiT等生成基线，但未包含更新的高效注意力机制如FlashAttention-2、局部注意力变体，也未与SDXL、PixArt等更强生成模型对比。作者坦承的局限包括：缺乏wall-clock时间测量验证O(NnC)的实际加速、Base模型上生成增益存在衰减（diminishing returns）、仅ImageNet-1K评估未验证跨域泛化。此外，FID改进幅度（2-5点）在感知显著性上可能有限，且生成训练成本高昂难以复现。

## 方法谱系与知识库定位

**方法谱系**: VCA属于**线性注意力/高效Transformer**方法族，直接父方法为**Differential Attention [88]**。谱系演进路径为：Standard MHSA → Linear Attention variants (Performer [4], FLatten [19]) → Differential Attention [88] → **Visual-Contrast Attention (VCA)**。VCA在[88]的正负双流思想上，新增了Stage I序列压缩与dual positional embeddings的空间对比编码，实现了从"二次复杂度差分"到"线性复杂度显式对比"的跨越。

**直接基线与差异**:
- **MHSA**: VCA完全替换其注意力核，保留外部接口，复杂度O(N²C) → O(NnC)
- **Differential Attention [88]**: VCA继承双流差分思想，但新增Stage I全局token池化与Stage II精修，实验显示75.6% vs 75.1%、FID 62.3 vs 63.9
- **FLatten Transformer [19]**: 同为线性注意力，VCA额外引入显式对比机制与双位置编码
- **Agent Attention [22]**: 亦尝试整合softmax与线性注意力，VCA则通过正负流差分实现更结构化的对比推理

**后续方向**:
1. **跨模态扩展**: 将visual-contrast token思想迁移至视频（时序对比token）或多模态（图文对比token）
2. **硬件协同验证**: 结合FlashAttention-2等IO-aware实现，验证O(NnC)的wall-clock加速
3. **自监督预训练**: 利用VCA的显式对比特性设计对比学习预训练目标，替代随机掩码

**标签**: 模态=image | 范式=supervised learning, diffusion models, flow models | 场景=image classification, class-conditional generation | 机制=linear attention, contrastive learning, dual-stream architecture, spatial pooling | 约束=low parameter overhead (<0.3M), zero extra FLOPs, architecture-agnostic

## 引用网络

### 直接 baseline（本文基于）

- [[P__InLine：可注入性与局部建模_InLine]] _(直接 baseline)_: Reconciles softmax and linear attention; core related work on linear attention m
- [[P__代理注意力：融合Softmax与_Agent_Attention]] _(直接 baseline)_: Integrates softmax and linear attention; directly relevant attention mechanism t

