---
title: 'Demystify Mamba in Vision: A Linear Attention Perspective'
type: paper
paper_level: A
venue: NeurIPS
year: 2024
paper_link: null
aliases:
- Mamba的线性注意力视角解构
- MILA (Mamba-Insp
- MILA (Mamba-Inspired Linear Attention)
- The forget gate and modified block
acceptance: Poster
cited_by: 9
code_url: https://github.com/LeapLabTHU/MLLA
method: MILA (Mamba-Inspired Linear Attention)
modalities:
- Image
paradigm: supervised
baselines:
- InLine：可注入性与局部建模_InLine
---

# Demystify Mamba in Vision: A Linear Attention Perspective

[Code](https://github.com/LeapLabTHU/MLLA)

**Topics**: [[T__Classification]], [[T__Semantic_Segmentation]] | **Method**: [[M__MILA]] | **Datasets**: [[D__ImageNet-1K]]

> [!tip] 核心洞察
> The forget gate and modified block design are the core contributors to Mamba's success, and incorporating these two designs into linear attention yields a Mamba-Inspired Linear Attention (MILA) model that outperforms vision Mamba models while maintaining parallelizable computation.

| 中文题名 | Mamba的线性注意力视角解构 |
| 英文题名 | Demystify Mamba in Vision: A Linear Attention Perspective |
| 会议/期刊 | NeurIPS 2024 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2405.16605) · [Code](https://github.com/LeapLabTHU/MLLA) · [DOI](https://doi.org/10.52202/079017-4039) |
| 主要任务 | Image Classification, Semantic Segmentation |
| 主要 baseline | Linear Attention Transformer, Mamba/SSM, Vim, VMamba, LocalVMamba, FLatten Transformer |

> [!abstract] 因为「Mamba 与 linear attention 数学形式相似但性能差距显著」，作者在「Linear Attention Transformer」基础上改了「引入 forget gate 和 Mamba-style block design」，在「ImageNet-1K」上取得「MILA-T 83.5%，超越 VMamba-T 82.5% 和 LocalVMamba-T 82.7%」

- **ImageNet-1K Top-1**: MILA-T 83.5% vs. LocalVMamba-T 82.7% (+0.8%), MILA-S 84.3% vs. LocalVMamba-S 83.7% (+0.6%), MILA-B 85.3% vs. VMamba-B 83.9% (+1.4%)
- **无 MESA 仍领先**: MILA-T 83.3% 仍超 LocalVMamba-T 82.7%（+0.6%），证明核心设计本身有效
- **参数量**: T/S/B 三档分别为 25M / 43M / 96M，保持并行计算与高速推理

## 背景与动机

Mamba 及其视觉变体（Vim、VMamba、LocalVMamba）在图像分类等任务上表现优异，但其成功根源始终模糊。一个令人困惑的现象是：Mamba 的选择性状态空间模型（selective SSM）与 linear attention Transformer 在数学上都具有线性复杂度，核心运算都涉及累积的键值乘积，但 Mamba 显著优于 linear attention。例如，FLatten Transformer 作为先进的 linear attention 方法在 ImageNet-1K 上仅得 82.1%，而 VMamba-T 达到 82.5%，这 0.4% 的差距背后究竟是什么机制在起作用？

现有工作从不同角度尝试解释。Katharopoulos 等人提出的 Linear Attention Transformer 将 softmax attention 核化为线性形式，通过递归计算 $S_i = S_{i-1} + K_i^T V_i$ 实现 $O(n)$ 复杂度，但保留了归一化分母 $z_i$。近期「The hidden attention of mamba models」等研究开始关注 Mamba 的注意力特性，但未系统拆解其与普通 linear attention 的结构性差异。另一路线如「Bridging the divide: Reconsidering softmax and linear attention」重新审视了两种注意力的关系，却未触及 Mamba 特有的门控设计。

这些工作的共同盲区在于：没有将 Mamba 严格嵌入 linear attention 的数学框架进行逐组件对比，因而无法回答——Mamba 的输入门 $\Delta_i$、遗忘门 $e^{A_i}$、shortcut $D \odot x_i$、无归一化、单头设计、以及独特的 block 结构——这六个区别中哪些真正关键？本文的核心动机正是通过统一数学表述，系统消融这六个因素，最终识别出真正驱动性能的设计选择。
![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/3dc5de15-63c6-460e-873f-8015eb2af816/figures/fig_001.jpeg)
*Figure: Illustration of selective SSM in Mamba (eq. (11)) and single head linear attention (eq. (12)).*



## 核心创新

核心洞察：Mamba 本质上是一种带有指数衰减遗忘门且无归一化的单头 linear attention，因为将 SSM 的递归状态展开为显式累积和后，其有效键 $\tilde{K}_j = e^{\sum A_k} B_j$ 天然编码了位置相关的衰减权重，从而使 linear attention 框架能够精确吸收 Mamba 的核心机制而不失并行性。

| 维度 | Baseline (Linear Attention) | 本文 (MILA) |
|:---|:---|:---|
| 注意力核 | $y_i = Q_i S_i / (Q_i z_i)$，含归一化分母 $z_i$ | $y_i = Q_i S_i$，移除归一化，引入 $e^{A_i}$ 衰减 |
| 记忆衰减 | 无显式衰减，所有历史 token 等权累积 | 数据依赖的指数遗忘门 $e^{A_i}$，实现位置相关衰减 |
| Block 结构 | 标准 pre-norm，attention 与 MLP 常规排列 | Mamba-style 路由门控结构，特定残差连接方式 |
| 并行性 | 天然可并行（RNN 形式） | 保持完全并行，避免 Mamba 的序列扫描限制 |

## 整体框架


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/3dc5de15-63c6-460e-873f-8015eb2af816/figures/fig_002.jpeg)
*Figure: Illustration of selective state space model (eq. (8)) and its equivalent form (eq. (9)).*



MILA 采用标准的 4 阶段层次化视觉 backbone，整体数据流如下：

**输入**：$H \times W \times 3$ 图像

**Stem**：下采样 4 倍，输出 $H/4 \times W/4 \times C_1$ 特征图，初始嵌入与空间压缩

**Stage 1（$L_1$ 个 MILA Blocks）**：在 $H/4 \times W/4$ 分辨率处理低级特征，每个 block 包含 Mamba-style 门控路由与带遗忘门的 linear attention

**Downsampling 1→2**：空间减半、通道扩展，进入 $H/8 \times W/8 \times C_2$

**Stage 2（$L_2$ 个 MILA Blocks）**：中级特征处理，继续堆叠 MILA block

**Downsampling 2→3**：至 $H/16 \times W/16 \times C_3$

**Stage 3（$L_3$ 个 MILA Blocks）**：高级语义特征聚合

**Downsampling 3→4**：至 $H/32 \times W/32 \times C_4$

**Stage 4（$L_4$ 个 MILA Blocks）**：最终语义表征，接全局平均池化与分类头输出预测

```
Image (H×W×3)
  → Stem ─────────────────────────► H/4 × W/4 × C₁
  → [MILA Block]×L₁ ─────────────► H/4 × W/4 × C₁
  → Downsample ──────────────────► H/8 × W/8 × C₂
  → [MILA Block]×L₂ ─────────────► H/8 × W/8 × C₂
  → Downsample ──────────────────► H/16 × W/16 × C₃
  → [MILA Block]×L₃ ─────────────► H/16 × W/16 × C₃
  → Downsample ──────────────────► H/32 × W/32 × C₄
  → [MILA Block]×L₄ ─────────────► H/32 × W/32 × C₄
  → GAP + Classifier ────────────► Logits
```

关键设计：MILA block 内部保留 Mamba 的宏观拓扑（输入投影、门控分支、SSM/attention 主路径、输出融合），但将核心 SSM 替换为带遗忘门的 linear attention 核，实现并行训练与推理。

## 核心模块与公式推导

### 模块 1: 统一视角——Mamba 即 Linear Attention（对应框架图 attention 核心）

**直觉**：将 Mamba 的递归状态方程显式展开，可发现其结构与 linear attention 的累积和完全同构，仅多了指数衰减因子与 shortcut。

**Baseline 公式** (Linear Attention Transformer [26]):
$$y_i = \frac{Q_i S_i}{Q_i z_i}, \quad S_i = S_{i-1} + K_i^T V_i, \quad z_i = z_{i-1} + K_i^T$$

符号: $Q_i, K_i, V_i \in \mathbb{R}^{1 \times d}$ 为 query/key/value；$S_i \in \mathbb{R}^{d \times d}$ 为累积的 KV 状态；$z_i \in \mathbb{R}^{1 \times d}$ 为归一化分母（累积的 key 和）。

**变化点**：Mamba 的 selective SSM 使用输入依赖的参数 $\bar{A}_i, \bar{B}_i$，且没有显式的 $z_i$ 归一化。作者通过连续化重参数化，将离散 SSM 转化为显式门控形式。

**本文公式（推导）**:
$$\text{Step 1}: h_i = \bar{A}_i h_{i-1} + \bar{B}_i x_i \quad \text{(标准离散 SSM)}$$
$$\text{Step 2}: h_i = e^{A_i} h_{i-1} + B_i x_i \quad \text{(连续化重参数化，显式分离遗忘门 $e^{A_i}$)}$$
$$\text{Step 3}: h_i = \sum_{j=1}^{i} \left(e^{\sum_{k=j+1}^{i} A_k} B_j x_j\right) \quad \text{(递归展开为显式累积和)}$$
$$\text{Step 4}: y_i = Q_i \sum_{j=1}^{i} \tilde{K}_j^T \tilde{V}_j + D \odot x_i \quad \text{(映射到 linear attention: } \tilde{K}_j = e^{\sum A_k} B_j, \tilde{V}_j = x_j\text{)}$$

**最终**: Mamba = Linear Attention + 指数衰减遗忘门 + 无归一化 + shortcut $D \odot x_i$

**对应消融**：Table 1 显示，将六个区别逐一移除，遗忘门与无归一化的组合影响最为显著。

---

### 模块 2: MILA Attention——带遗忘门的 Linear Attention（对应框架图 MILA Block 内部）

**直觉**：既然 Mamba 的核心增益来自遗忘门 $e^{A_i}$ 而非完整 SSM，可直接将该门嵌入 linear attention 的并行框架，避免 Mamba 的硬件不友好扫描。

**Baseline 公式** (标准 Linear Attention):
$$y_i = \frac{Q_i \sum_{j=1}^{i} K_j^T V_j}{Q_i \sum_{j=1}^{i} K_j^T}$$

**变化点**：(1) 移除归一化分母，避免数值不稳定与性能损失；(2) 引入数据依赖的指数衰减 $e^{A_i}$ 替代等权累积，使不同位置的 historical token 具有差异化的贡献权重；(3) 保留 shortcut $D \odot x_i$ 维持梯度流动。

**本文公式（推导）**:
$$\text{Step 1}: \tilde{K}_j = e^{\sum_{k=j+1}^{i} A_k} \cdot B_j \quad \text{(有效 key 带位置衰减，数据依赖)}$$
$$\text{Step 2}: S_i = e^{A_i} S_{i-1} + B_i^T x_i \quad \text{(递归状态更新，加入指数衰减因子)}$$
$$\text{Step 3}: y_i = Q_i S_i + D \odot x_i \quad \text{(线性投影输出，无归一化，保留 shortcut)}$$

**最终**: $y_i = Q_i \underbrace{\sum_{j=1}^{i} \left(e^{\sum_{k=j+1}^{i} A_k} B_j^T x_j\right)}_{\text{带衰减的累积状态}} + D \odot x_i$

**对应消融**：Table 2 对比遗忘门的替代方案（如固定衰减、可学习衰减等），指数衰减 $e^{A_i}$ 显著优于其他设计。

---

### 模块 3: Mamba-Style Block Design（对应框架图 block 拓扑）

**直觉**：Mamba 的宏观 block 结构（输入投影、门控分支、主路径、残差融合）本身构成一种优于标准 Transformer block 的归纳偏置，与 attention 机制的选择解耦。

**Baseline 公式** (标准 Transformer Block):
$$x' = \text{LN}(x) + \text{Attention}(\text{LN}(x))$$
$$x'' = \text{LN}(x') + \text{MLP}(\text{LN}(x'))$$

**变化点**：Mamba block 采用 (1) 输入依赖的线性投影生成多个门控分支；(2) 主路径与门控支路的逐元素乘积调制；(3) 特定的残差连接顺序与 MLP 融合方式。

**本文公式（推导）**:
$$\text{Step 1}: [x_1, x_2, g] = \text{Linear}_{\text{split}}(x) \quad \text{(输入投影为多支路)}$$
$$\text{Step 2}: h = \text{MILA-Attention}(x_1) \odot \sigma(g) \quad \text{(主路径经门控调制)}$$
$$\text{Step 3}: y = x_2 + \text{MLP}(\text{LN}(h)) \quad \text{(残差融合与 MLP)}$$

**最终**: 整体 block 输出 = 输入分支 + MLP(门控调制的 attention 输出)

**对应消融**：Table 1 中替换为标准 linear attention block 设计导致显著性能下降，验证了 block topology 的独立重要性。

## 实验与分析



本文在 ImageNet-1K 上进行了系统评估，涵盖 T/S/B 三种模型尺度。 展示了 MILA 与 vision Mamba 及 linear attention 方法的完整对比。核心结果显示：MILA-T 达到 83.5% Top-1 准确率，相比最强的 vision Mamba 基线 LocalVMamba-T（82.7%）提升 +0.8%，相比最佳 linear attention 方法 FLatten Transformer（82.1%）提升 +1.3%。这一差距在更大模型上持续扩大：MILA-S 84.3% vs. LocalVMamba-S 83.7%（+0.6%），MILA-B 85.3% vs. VMamba-B 83.9%（+1.4%），表明 MILA 的设计优势具有尺度一致性。

值得注意的是，即使移除 MESA 训练策略，MILA-T 仍取得 83.3%，依然领先 LocalVMamba-T 的 82.7%（+0.6%），说明性能增益主要源于架构创新而非训练技巧。速度测试方面，Figure 6 显示 MILA 在 RTX3090 上保持与 linear attention 相当的并行推理效率，避免了 Mamba 序列扫描带来的硬件效率损失。



消融实验进一步验证了核心洞察。 中，作者对六个区别逐一进行消融：移除遗忘门或替换为简单替代方案导致性能显著下降；替换为标准 block 设计同样造成明显退化。相比之下，输入门、shortcut、单头设计、无归一化这四个因素的单独移除影响较小，与核心主张一致。具体而言，遗忘门的替代方案实验中，固定衰减或线性衰减均不及数据依赖的指数衰减 $e^{A_i}$。

公平性方面，本文对比的 vision Mamba 基线（Vim、VMamba、LocalVMamba）代表了该方向的最新进展，但缺少与 DeiT、Swin Transformer、ConvNeXt 等最强通用视觉 backbone 的直接对比。此外，MESA 训练策略虽仅带来 0.1-0.3 的边际增益，但未在所有基线上统一使用，可能存在轻微的不对等。作者也坦承分析聚焦于视觉任务，向 NLP 等领域的泛化有待验证。

## 方法谱系与知识库定位

MILA 隶属于 **Linear Attention → Vision Efficient Transformer** 方法谱系，直接父方法为 Katharopoulos 等人的 Linear Attention Transformer [26]。核心继承关系：保留线性复杂度的并行 attention 框架；关键变异发生在两个 slot——(1) attention_mechanism：引入 Mamba 的指数衰减遗忘门 $e^{A_i}$，移除归一化分母 $z_i$；(2) block_design：整体替换为 Mamba-style 的宏观拓扑结构。

**直接基线与差异**：
- **Linear Attention Transformer [26]**：MILA 添加遗忘门与 Mamba block，解决其无显式位置衰减、block 设计简单的问题
- **Mamba [14]**：MILA 将其 SSM 核替换为 parallelizable linear attention，解除序列扫描的硬件约束
- **VMamba / LocalVMamba [25]**：MILA 在性能更优的同时保持完全并行，避免窗口化扫描的复杂性
- **FLatten Transformer [15]**：同属于 linear attention 家族，MILA 通过 Mamba 门控设计取得 +1.3 的显著优势

**后续方向**：(1) 将遗忘门机制推广到 NLP 长序列建模，验证跨模态通用性；(2) 与 Swin、ConvNeXt 等强基线进行严格对等比较；(3) 探索遗忘门与 softmax attention 的混合形式，结合两者优势。

**标签**：modality=image | paradigm=supervised classification | scenario=efficient vision backbone | mechanism=linear attention with exponential decay forget gate | constraint=parallel computation, linear complexity

## 引用网络

### 直接 baseline（本文基于）

- [[P__InLine：可注入性与局部建模_InLine]] _(方法来源)_: Directly on softmax vs linear attention; core to this paper's theoretical perspe

