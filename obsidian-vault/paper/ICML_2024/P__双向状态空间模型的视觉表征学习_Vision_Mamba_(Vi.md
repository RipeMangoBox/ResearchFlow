---
title: 'Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model'
type: paper
paper_level: C
venue: ICML
year: 2024
paper_link: null
aliases:
- 双向状态空间模型的视觉表征学习
- Vision Mamba (Vi
- Vision Mamba (Vim)
acceptance: Poster
cited_by: 1695
code_url: https://github.com/hustvl/Vim
method: Vision Mamba (Vim)
---

# Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model

[Code](https://github.com/hustvl/Vim)

**Topics**: [[T__Classification]], [[T__Object_Detection]], [[T__Semantic_Segmentation]] | **Method**: [[M__Vision_Mamba]] | **Datasets**: [[D__ImageNet-1K]]

| 中文题名 | 双向状态空间模型的视觉表征学习 |
| 英文题名 | Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model |
| 会议/期刊 | ICML 2024 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2401.09417) · [Code](https://github.com/hustvl/Vim) · [Project](https://github.com/hustvl/Vim) |
| 主要任务 | ImageNet-1K 图像分类、语义分割 |
| 主要 baseline | Mamba, Swin Transformer, DeiT, FocalTransformer |

> [!abstract] 因为「视觉任务需要双向上下文理解但标准 Mamba 仅支持单向序列建模」，作者在「Mamba」基础上改了「增加反向 SSM 分支与中间类别令牌，并扩展为层次化架构 Hier-Vim」，在「ImageNet-1K」上取得「Hier-Vim-T 82.5% top-1 accuracy，相比 Swin-T 提升 +1.3%」

- **Hier-Vim-T** 在 ImageNet-1K 上达到 **82.5%** top-1 accuracy，超越 Swin-T (81.2%) **+1.3%**，与 CVT-21 持平
- **Hier-Vim-S** 达到 **83.4%**，超越 Swin-S (83.2%) **+0.2%**，但略低于 FocalTransformer-S (83.5%) **-0.1%**
- **Hier-Vim-B** 达到 **83.9%**，超越 Swin-B (83.5%) **+0.4%**，超越 FocalTransformer-B (83.8%) **+0.1%**

## 背景与动机

视觉表征学习的核心挑战在于：如何高效地建模图像中的全局依赖关系，同时避免自注意力机制的二次计算复杂度。以 ViT 为代表的视觉 Transformer 将图像切分为 patch 序列，通过自注意力实现全局交互，但计算量随序列长度平方增长；以 Swin Transformer 为代表的层次化方法通过局部窗口注意力缓解这一问题，但仍受限于注意力机制的本质开销。

Mamba (Gu & Dao, 2023) 提出了一种突破性的替代方案：基于选择性状态空间模型（Selective State Space Model, SSM）的线性时间序列建模方法。其核心思想是通过隐状态递推 $h_t = \overline{A} h_{t-1} + \overline{B} x_t$ 替代注意力机制，将计算复杂度从 $O(N^2)$ 降至 $O(N)$，同时保持对长序列的建模能力。然而，Mamba 最初为语言建模设计，仅支持**单向前向处理**——这对于 NLP 任务（如自回归生成）是自然的选择，却对视觉任务造成了根本性的限制：图像中的物体没有固有的"阅读顺序"，像素或 patch 之间的空间关系是**双向对称**的。例如，一张猫的图片中，猫头与猫尾的语义关联需要同时从前向后和从后向前两个方向来捕捉，单向处理会丢失关键的上下文信息。

现有将 Mamba 适配到视觉领域的尝试（如直接采用单向 Mamba block）面临两个具体问题：其一，单向 SSM 无法有效编码图像的双向空间上下文，导致密集预测任务（如分割）性能受限；其二，标准的类别令牌放置策略（序列头部，如 DeiT）与 SSM 的循环特性不匹配——SSM 的隐状态在序列末端才充分累积全局信息，将分类令牌置于开头无法利用这一特性。

本文提出 Vision Mamba (Vim)，通过**双向 SSM 设计**和**中间类别令牌策略**，将 Mamba 的线性复杂度优势迁移到视觉领域，并进一步扩展为层次化架构 Hier-Vim，在保持计算效率的同时实现与主流视觉 Transformer 竞争的性能。

## 核心创新

核心洞察：**视觉序列需要双向状态更新**，因为图像空间不存在固有的因果方向，从而使线性复杂度的状态空间模型能够匹敌甚至超越窗口化注意力的表征能力成为可能。

与 baseline 的差异：

| 维度 | Baseline (Mamba / DeiT) | 本文 (Vim / Hier-Vim) |
|:---|:---|:---|
| 序列处理方向 | 单向前向 SSM | **双向 SSM**：前向 + 反向两个独立分支 |
| 卷积预处理 | 仅前向 Conv1d | **双向 Conv1d**：前向与反向各含独立 1D 卷积 |
| 类别令牌位置 | 序列头部 (DeiT-style) 或均值池化 | **序列中间位置**，利用 SSM 循环累积特性与中心物体先验 |
| 整体架构 | Plain（单尺度） | **Hierarchical 4-stage**（Hier-Vim），多尺度特征金字塔 |

关键设计选择：反向分支并非简单地将整个序列反转后复用同一 SSM，而是为反向处理配备**独立的 Conv1d 和 SSM 参数**，确保两个方向的学习不会互相干扰；中间类别令牌位于 $\lfloor n/2 \rfloor$ 位置，使前向和反向 SSM 的隐状态在分类点处均得到充分传播。

## 整体框架


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/261a26ad-070d-4cd1-a4c2-73d52cb00bcd/figures/fig_001.png)
*Figure: Performance and efficiency comparisons between DeiT (Touvron et al., 2021a) and our Vim model. Results show*



Vim 的整体数据流遵循"图像 patch 化 → 双向序列处理 → 中间位置分类"的范式：

1. **Visual Sequence Embedding（视觉序列嵌入）**：输入图像被切分为不重叠的 patch（如 $16 \times 16$），经线性投影展平为视觉令牌序列 $x \in \mathbb{R}^{n \times d}$，其中 $n$ 为序列长度，$d$ 为嵌入维度。

2. **Bidirectional Vim Block（双向 Vim 块）**：核心处理单元，替换 Mamba 中的单向 block。每个 block 接收视觉序列，并行通过前向和反向两个分支处理。

3. **Forward SSM + Conv1d（前向分支）**：标准 Mamba 组件，对输入序列按 $t = 1, 2, ..., n$ 顺序执行状态递推，输出前向特征 $y^{\text{fwd}}$。

4. **Backward Conv1d + SSM（反向分支）**：核心创新组件。输入序列先被反转 $x^{\text{rev}}$，再经独立的 Backward Conv1d 预处理，最后通过独立的 Backward SSM 按 $t = n, n-1, ..., 1$ 递推，输出反向特征 $y^{\text{bwd}}$。

5. **Fusion/Concatenation（双向特征融合）**：将 $y^{\text{fwd}}$ 与 $y^{\text{bwd}}$ 融合（具体方式如拼接或逐元素操作），经残差连接输出。

6. **Middle Class Token Aggregation（中间类别令牌聚合）**：从最终序列的中间位置 $\lfloor n/2 \rfloor$ 提取特征，送入分类头进行预测，替代传统的头部类别令牌或全局池化。

层次化变体 Hier-Vim 将上述 plain 架构扩展为 4-stage 金字塔：各 stage 具有不同的通道数（channels）和 block 数，逐步下采样并增加通道维度，以适配多尺度视觉表征需求。

```
Input Image
    ↓ [Patch Embedding]
Visual Token Sequence x ∈ R^{n×d}
    ↓ [Stack of Bidirectional Vim Blocks]
    ├─→ Forward Branch:  Conv1d_fwd → SSM_fwd → y^{fwd}
    └─→ Backward Branch: Conv1d_bwd → SSM_bwd → y^{bwd}
              ↑ (input reversed)
    ↓ [Fusion: y = f(y^{fwd}, y^{bwd}) + Residual]
Processed Sequence
    ↓ [Extract x_{⌊n/2⌋}]
Classification Head → Output Logits
```

## 核心模块与公式推导

### 模块 1: 双向状态空间模型（Bidirectional SSM）

**直觉**：图像空间关系无方向性，单向递推会丢失来自"未来"位置的上下文；通过反转序列并引入独立反向分支，使每个位置都能聚合双向信息。

**Baseline 公式** (Mamba, 单向前向 SSM)：
$$h_t = \overline{A} h_{t-1} + \overline{B} x_t$$

符号：$h_t \in \mathbb{R}^d$ 为时刻 $t$ 的隐状态；$\overline{A}, \overline{B}$ 为离散化后的系统矩阵；$x_t$ 为当前输入。

**变化点**：Mamba 的前向公式仅能从左到右累积信息，对于位置 $t$ 无法访问 $t+1, ..., n$ 的内容。视觉任务中，物体部件的空间分布无方向偏好，单向假设导致感受野不对称。

**本文公式（推导）**：
$$\text{Step 1 (序列反转)}: \quad x^{\text{rev}}_t = x_{n-t+1}$$
$$\text{Step 2 (反向递推)}: \quad h^{\text{bwd}}_t = \overline{A} h^{\text{bwd}}_{t+1} + \overline{B} x^{\text{rev}}_t \quad \text{（从末端向开头传播）}$$
$$\text{Step 3 (位置对齐)}: \quad y^{\text{bwd}}_t = C h^{\text{bwd}}_{n-t+1} + D x_t \quad \text{（将反向输出映射回原序列顺序）}$$
$$\text{最终 (双向融合)}: \quad y_t = \text{Fusion}\left(y^{\text{fwd}}_t, y^{\text{bwd}}_t\right)$$

**对应消融**：去掉 Backward Conv1d 仅保留 Bidirectional SSM，分类准确率从 73.9% 降至 72.8%（-1.1%），语义分割 mIoU 下降 -2.7，验证了反向卷积预处理的关键作用。

### 模块 2: 中间类别令牌（Middle Class Token）

**直觉**：SSM 的隐状态具有循环累积特性，序列末端的隐状态包含最丰富的全局信息；将分类令牌置于中间位置，使其同时受益于前向 SSM 从头至中的累积以及反向 SSM 从尾至中的累积。

**Baseline 公式** (DeiT, 头部类别令牌 / 均值池化)：
$$\hat{x}_{\text{class}} = x_0 \quad \text{(head token)} \quad \text{或} \quad \hat{x}_{\text{class}} = \frac{1}{n}\sum_{i=1}^n x_i \quad \text{(mean pool)}$$

**变化点**：头部令牌 $x_0$ 在单向 SSM 中仅接收来自位置 $1, ..., 0$ 的信息（实际为初始状态），几乎无全局上下文；均值池化虽聚合全局信息但破坏了 SSM 的递推结构优势。

**本文公式**：
$$\text{Step 1 (位置选择)}: \quad k = \left\lfloor \frac{n}{2} \right\rfloor$$
$$\text{Step 2 (特征提取)}: \quad \hat{x}_{\text{class}} = x_k$$
$$\text{Step 3 (分类)}: \quad p = \text{softmax}(W \cdot \text{LN}(\hat{x}_{\text{class}}) + b)$$

**对应消融**：使用均值池化替代中间类别令牌，准确率从 76.1% 暴跌至 73.9%（-2.2%）；使用头部类别令牌（DeiT-style）降至 75.2%（-0.9%），验证了中间位置对 SSM 循环特性的充分利用。

### 模块 3: 双向特征融合策略

**直觉**：前向与反向 SSM 输出需有效整合，但简单的逐元素相加可能导致信息淹没；本文通过消融对比了不同融合策略的优劣。

**Baseline 形式** (朴素双向扩展)：
$$y_t = y^{\text{fwd}}_t + y^{\text{bwd}}_t \quad \text{(Bidirectional Layer)}$$

**变化点**：消融实验表明，Bidirectional Layer（逐层双向融合）严重损害分类性能（-2.7%），Bidirectional Block（块级双向，即本文采用的独立分支后融合）虽然分类下降更多（-3.0% vs 单向基线），但对分割任务略有帮助。最终 Vim 采用**块级独立分支 + 末端融合**策略，配合独立的 Backward Conv1d，在分类和密集预测间取得平衡。

**本文实现**：
$$\text{Forward path}: \quad \tilde{x} = \text{LayerNorm}(x), \quad u = \text{Linear}_{\text{proj}}(\tilde{x})$$
$$u^{\text{fwd}} = \text{SSM}_{\text{fwd}}(\text{Conv1d}_{\text{fwd}}(u)), \quad u^{\text{bwd}} = \text{SSM}_{\text{bwd}}(\text{Conv1d}_{\text{bwd}}(\text{Reverse}(u)))$$
$$\text{Fusion}: \quad y = \text{Linear}_{\text{out}}(\text{Concat}[u^{\text{fwd}}, u^{\text{bwd}}]) + x \quad \text{(残差连接)}$$

## 实验与分析



本文在 ImageNet-1K 分类基准上评估了 Hier-Vim 的三个尺度变体（Tiny/Small/Base）。如 Table 7 所示，Hier-Vim-T 取得 82.5% 的 top-1 accuracy，相比同规模的 Swin-T（81.2%）提升 +1.3%，与 CVT-21（82.5%）持平但参数量更具优势；Hier-Vim-S 达到 83.4%，以 +0.2% 优势超过 Swin-S（83.2%），但略低于 FocalTransformer-S（83.5%）0.1%；Hier-Vim-B 取得 83.9%，超越 Swin-B（83.5%）+0.4%，并以 +0.1% 微弱优势超过 FocalTransformer-B（83.8%）。总体而言，Hier-Vim 在 Tiny 和 Base 尺度上确立了相比 Swin Transformer 的明确优势，但在 Small 尺度上未能全面超越 FocalTransformer 系列。



消融实验（Table 4 与 Table 5）揭示了各设计组件的贡献。在双向设计方面，从完整 Bidirectional SSM + Conv1d 退化为纯单向处理，分类准确率下降 0.7%（73.9% vs 73.2%），且语义分割 mIoU 出现显著滑坡；若采用 Bidirectional Block（块级独立参数）替代逐层共享的双向设计，分类暴跌 -3.0%（70.9% vs 73.9%），但分割略有改善，印证了独立参数双向分支的必要性。在分类设计方面，中间类别令牌是最关键的单项创新：替换为均值池化导致 -2.2%（73.9% vs 76.1%），替换为头部类别令牌（DeiT-style）仍有 -0.9% 差距（75.2% vs 76.1%），证明中间位置对 SSM 循环累积特性的不可替代性。



效率方面，Figure 1 展示了 Vim 相比 DeiT 在性能-效率权衡上的优势：在相近准确率下，Vim 具有更低的计算复杂度和内存占用，这得益于 SSM 的线性时间特性替代了自注意力的二次复杂度。

公平性检验：本文的比较存在一定局限。首先，Table 7 中 Hier-Vim-T 与 CVT-21 持平而非超越，Hier-Vim-S 甚至低于 FocalTransformer-S，未能在所有尺度上实现一致领先；其次，缺失与 DeiT、ViT、ConvNeXt、EfficientNet 等主流架构的直接对比，也未纳入 MaxViT、CoAtNet 等更新进的基线；第三，分割实验仅采用简单的 2-layer Segmenter head，可能未能充分释放 Hier-Vim 的多尺度潜力；最后，论文未报告具体的训练时间、GPU 类型及推理延迟数据，效率优势的理论分析（线性复杂度）与实际硬件表现之间的 gap 有待验证。作者披露的中间类别令牌在极端非中心构图图像上可能失效，这是该方法的一个潜在 failure mode。

## 方法谱系与知识库定位

**方法家族**：线性复杂度序列建模 → 状态空间模型 (S4/S5) → 选择性 SSM (Mamba) → **视觉适配 (Vision Mamba)**

**父方法**：Mamba (Gu & Dao, 2023)。Vim 直接继承 Mamba block 的核心结构，但在两个关键 slot 上进行改造：
- **architecture**：单向 SSM → 双向 SSM + 双向 Conv1d
- **inference_strategy**：标准序列处理 → 中间类别令牌提取

**直接基线与差异**：
- **Mamba**：仅前向 SSM，为语言因果建模设计；Vim 增加反向分支与独立卷积，打破单向限制
- **Swin Transformer**：层次化窗口注意力，$O(N)$ 复杂度但非序列模型；Vim 以纯序列建模实现可比的层次化架构（Hier-Vim），保持 SSM 的循环特性
- **DeiT**：提供头部类别令牌训练策略；Vim 将其作为消融对照，证明中间位置更适配 SSM

**后续方向**：(1) 将双向 SSM 扩展至视频时序建模，利用时间维度的双向性；(2) 探索更复杂的双向融合机制（如门控、注意力加权）替代简单拼接；(3) 在检测、分割等密集预测任务上设计专用的双向特征金字塔，替代当前简单的 Segmenter head。

**标签**：`modality: vision` | `paradigm: state_space_model` | `scenario: image_classification` | `mechanism: bidirectional_sequence_modeling` | `constraint: linear_time_complexity`

