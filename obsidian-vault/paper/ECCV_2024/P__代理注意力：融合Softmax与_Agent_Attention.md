---
title: 'Agent Attention: On the Integration of Softmax and Linear Attention'
type: paper
paper_level: C
venue: ECCV
year: 2024
paper_link: null
aliases:
- 代理注意力：融合Softmax与线性注意力
- Agent Attention
- Agent attention integrates the expr
acceptance: Poster
cited_by: 241
code_url: https://github.com/LeapLabTHU/Agent-Attention
method: Agent Attention
modalities:
- Image
paradigm: supervised
followups:
- 线性差分视觉Transforme_Visual-Contrast_
- InLine：可注入性与局部建模_InLine
---

# Agent Attention: On the Integration of Softmax and Linear Attention

[Code](https://github.com/LeapLabTHU/Agent-Attention)

**Topics**: [[T__Classification]], [[T__Semantic_Segmentation]], [[T__Object_Detection]] | **Method**: [[M__Agent_Attention]] | **Datasets**: [[D__ImageNet-1K]], [[D__ADE20K_Semantic]], [[D__MS-COCO]]

> [!tip] 核心洞察
> Agent attention integrates the expressiveness of Softmax attention with the linear complexity of linear attention by using a small set of agent tokens to aggregate global information and broadcast it to queries.

| 中文题名 | 代理注意力：融合Softmax与线性注意力 |
| 英文题名 | Agent Attention: On the Integration of Softmax and Linear Attention |
| 会议/期刊 | ECCV 2024 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2312.08874) · [Code](https://github.com/LeapLabTHU/Agent-Attention) · [DOI](https://doi.org/10.1007/978-3-031-72973-7_8) |
| 主要任务 | ImageNet-1K 图像分类、COCO 目标检测、ADE20K 语义分割、Stable Diffusion 图像生成 |
| 主要 baseline | Softmax Attention、Linear Attention、Swin Transformer、PVT、DeiT、Flatten Transformer、Performer |

> [!abstract] 因为「Softmax 注意力具有二次复杂度而线性注意力损失表达能力」，作者在「Linear Attention」基础上改了「引入代理令牌做两步聚合-广播」，在「ImageNet-1K / ADE20K / COCO」上取得「Agent-Swin-T 82.6% top-1 accuracy（+1.3 over Swin-T）、Agent-PVT-L 46.52 mIoU（+3.03 over PVT-L）」

- **ImageNet-1K**: Agent-Swin-T 达到 **82.6%** top-1 accuracy，相比 Swin-T **81.3%** 提升 **+1.3**
- **ADE20K 语义分割**: Agent-PVT-L 达到 **46.52 mIoU** / **58.5 mAcc**，相比 PVT-L **43.49 mIoU** / **54.62 mAcc** 提升 **+3.03** / **+3.88**
- **参数量降低**: Agent-PVT-T 仅 **15M**（vs PVT-T **17M**），Agent-PVT-S **24M**（vs PVT-S **28M**）

## 背景与动机

Vision Transformer 的核心瓶颈在于注意力计算的二次复杂度：当处理高分辨率图像时，token 数量 n 急剧增长，Softmax 注意力的 O(n²) 计算成为不可承受的负担。例如一张 1024×1024 的图像若按 16×16 patch 划分，将产生 4096 个 token，其注意力矩阵高达 4096² ≈ 1680 万项。

现有方法主要从两个方向缓解这一问题：
- **Softmax Attention**（标准多头注意力）：通过 Softmax(QK^T/√d)V 计算，具有全局感受野和强表达能力，但复杂度为 O(n²)，难以扩展至高分辨率。
- **Linear Attention**（如 Performer、Flatten Transformer）：利用特征映射 φ 将 Softmax 核函数近似为内积形式 φ(Q)φ(K)^T V，将复杂度降至 O(n)，但实证表明其注意力分布与 Softmax 差异显著，表达能力不足，导致性能下降。
- **Swin Transformer** 等局部注意力方法：通过 shifted window 限制注意力范围，虽降低复杂度但牺牲了全局建模能力。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/61af2222-3da8-4a08-a347-6ab88086aaf5/figures/Figure_1.png)
*Figure 1 (motivation): An illustration of the motivation of agent attention.*



这些方法的困境在于：**效率与表达能力不可兼得**——要么保留 Softmax 的全局建模但承受二次复杂度，要么追求线性复杂度但损失注意力分布质量。更关键的是，线性注意力的特征映射往往无法生成类似 Softmax 的尖锐、有选择性的注意力分布，导致模型难以聚焦关键区域。

本文提出 Agent Attention，核心思路是：**用少量代理令牌（agent tokens）作为"信息枢纽"，先通过 Softmax 注意力聚合全局信息，再通过线性注意力高效广播给所有查询**，从而同时获得 Softmax 的表达能力和线性注意力的计算效率。

## 核心创新

核心洞察：**引入中间代理令牌作为信息瓶颈**，因为代理令牌数量 m << n 使得两步计算均可控，从而使「Softmax 的表达能力 + 线性复杂度的效率」成为可能。

与 baseline 的差异：

| 维度 | Baseline (Softmax / Linear) | 本文 Agent Attention |
|------|---------------------------|-------------------|
| 计算路径 | 查询直接 attend 所有键（Softmax）或 查询通过特征映射直接 attend 所有键（Linear） | 查询 → 代理令牌（线性）← 全局信息（Softmax 聚合） |
| 复杂度 | O(n²) / O(n) | **O(nm)**，其中 m << n 为代理令牌数 |
| 注意力分布 | Softmax 尖锐可选择 / Linear 平坦无区分 | **接近 Softmax 的分布**，通过代理聚合保留 |
| 即插即用性 | 原生模块 | 可替换任意 Transformer 的注意力块，无需修改整体架构 |

## 整体框架


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/61af2222-3da8-4a08-a347-6ab88086aaf5/figures/Figure_2.png)
*Figure 2 (architecture): Differences between Softmax attention, Linear attention and Agent attention.*



Agent Attention 模块的数据流如下：

1. **输入特征** X ∈ R^(n×d)：来自前一层 Transformer 的 n 个 d 维 token。
2. **代理令牌生成（Agent Token Generation）**：从 X 动态生成 m 个代理令牌 Q_a ∈ R^(m×d)，其中 m << n。实现方式包括 spatial pooling（空间池化）或 depthwise convolution（DWC），保证代理令牌具有代表性且计算轻量。
3. **代理聚合（Agent Aggregation）**：代理令牌 Q_a 作为"查询"，通过 **Softmax 注意力** 从所有键 K 和值 V 中聚合全局信息，输出聚合特征 A ∈ R^(m×d)。此步骤保留 Softmax 的表达能力，但仅计算 m×n 的注意力矩阵。
4. **代理广播（Agent Broadcast）**：原始查询 Q 通过 **线性注意力** 从聚合特征 A 中检索信息，输出 O ∈ R^(n×d)。利用特征映射 φ 将计算降至线性复杂度。

整体流程可概括为：

```
输入 X ──→ [生成 Q_a] ──→ SoftmaxAttn(Q_a, K, V) ──→ A ──→ LinearAttn(Q, K_a, A) ──→ 输出 O
         (pooling/DWC)      (agent aggregation)         (agent broadcast)
```


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/61af2222-3da8-4a08-a347-6ab88086aaf5/figures/Figure_3.png)
*Figure 3 (architecture): An illustration of our agent attention and agent attention modules.*



该模块可即插即用替换 DeiT、Swin、PVT 等现有架构中的标准注意力块，形成 Agent-DeiT、Agent-Swin、Agent-PVT 等变体。

## 核心模块与公式推导

### 模块 1: 代理聚合（Agent Aggregation）（对应框架图 Stage 2）

**直觉**: 用少量代理令牌作为"信息摘要员"，通过保留 Softmax 的尖锐注意力分布来捕获全局关键信息，避免线性注意力的表达能力损失。

**Baseline 公式** (Softmax Attention):
$$\text{Attention}(Q,K,V) = \text{Softmax}(QK^T/\sqrt{d})V$$
符号: $Q, K, V \in \mathbb{R}^{n \times d}$ 为查询/键/值矩阵，$d$ 为 head 维度，$n$ 为 token 数。

**变化点**: 标准 Softmax 中每个查询直接 attend n 个键，复杂度 O(n²)。本文将查询替换为 m 个代理令牌 Q_a（m << n），大幅降低第一步计算量，同时保留 Softmax 的非线性归一化特性。

**本文公式（推导）**:
$$\text{Step 1}: Q_a = \text{Pool}(X) \text{ 或 } \text{DWC}(X) \quad \text{（动态生成 m 个代理令牌，m << n）}$$
$$\text{Step 2}: A = \text{Softmax}(Q_a K^T / \sqrt{d}) V \quad \text{（代理令牌聚合全局信息，复杂度 O(mn)）}$$
$$\text{最终}: A \in \mathbb{R}^{m \times d} \text{ 为聚合后的代理特征}$$

**对应消融**: Table 5 显示将动态代理令牌替换为静态可学习参数，ImageNet-1K accuracy 从 **82.6% 降至 82.2%（-0.4）**，验证动态生成的必要性。

---

### 模块 2: 代理广播（Agent Broadcast）（对应框架图 Stage 3）

**直觉**: 原始查询数量庞大，但代理特征 A 已浓缩全局信息，故用线性注意力的高效计算从 A 中检索，实现复杂度从 O(n²) 到 O(nm) 的跃迁。

**Baseline 公式** (Linear Attention [22]):
$$\text{LinearAttn}(Q,K,V) = \phi(Q)(\phi(K)^T V)$$
符号: $\phi(\cdot)$ 为特征映射（如 ReLU 或 elu+1），将输入映射到核空间以近似 Softmax 内积。

**变化点**: 标准线性注意力让查询直接 attend 所有键，虽为 O(n) 但分布平坦。本文让查询 attend 代理键 K_a 和聚合特征 A，既保持线性复杂度，又通过 A 的 Softmax 聚合历史获得更丰富的信息内容。

**本文公式（推导）**:
$$\text{Step 1}: K_a = W_k Q_a \quad \text{（代理令牌的键，共享投影或独立）}$$
$$\text{Step 2}: O = \phi(Q)(\phi(K_a)^T A) \quad \text{（查询从代理特征线性检索，复杂度 O(nm)）}$$
$$\text{最终}: O \in \mathbb{R}^{n \times d} \text{ 为输出特征，等价于 } \text{LinearAttn}(Q, K_a, A)$$

**对应消融**: Table 5 中移除代理广播（即退化为纯 Softmax 或纯 Linear）导致性能显著下降；Figure 10 可视化显示 Agent Attention 的注意力分布与 Softmax 高度相似，而 Linear Attention 分布平坦。

---

### 模块 3: 完整 Agent Attention 组合

**直觉**: 将聚合与广播组合为统一模块，实现"Softmax 的表达 + Linear 的效率"的有机融合。

**本文公式（最终组合）**:
$$\text{AgentAttn}(Q,K,V) = \text{LinearAttn}(Q, K_a, \text{SoftmaxAttn}(Q_a, K, V))$$

或展开为：
$$\text{AgentAttn}(Q,K,V) = \phi(Q) \big(\phi(K_a)^T \cdot \text{Softmax}(Q_a K^T / \sqrt{d}) V\big)$$

**复杂度分析**: 整体复杂度为 O(nm) + O(mn) = **O(nm)**，其中 m << n（如 m = n/49 通过 7×7 pooling）。相比 Softmax 的 O(n²)，高分辨率下节省显著；相比 Linear 的 O(n)，仅增加与 m 成正比的常数因子。

**对应消融**: Table 7 显示在 Stage 4（最终阶段）替换为 Agent Attention 会 **marginal decrease (-0.1)**，提示深层语义聚合仍部分依赖原始 Softmax；而在 Stages 1-3 替换获 **+1.3** 增益，但仅替换早期阶段反致 **-0.9 ~ -0.8**，说明各阶段需协同使用。

## 实验与分析


![Table 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/61af2222-3da8-4a08-a347-6ab88086aaf5/figures/Table_3.png)
*Table 3 (quantitative): Result of semantic segmentation on ADE20K.*



本文在 ImageNet-1K 分类、COCO 检测、ADE20K 分割及 Stable Diffusion 生成任务上进行了系统验证。核心 headline 为：Agent-Swin-T 在 ImageNet-1K 上达到 **82.6% top-1 accuracy**，相比 Swin-T 的 **81.3%** 提升 **+1.3**；Agent-PVT-L 在 ADE20K 语义分割上达到 **46.52 mIoU** 和 **58.5 mAcc**，相比 PVT-L 的 **43.49 mIoU** / **54.62 mAcc** 分别提升 **+3.03** 和 **+3.88**。这些增益在降低参数量的同时实现（Agent-PVT-T 15M vs PVT-T 17M），表明效率与精度双赢。


![Figure 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/61af2222-3da8-4a08-a347-6ab88086aaf5/figures/Figure_4.png)
*Figure 4 (result): Comparisons with SOTA on ImageNet.*



ImageNet-C 鲁棒性测试（Figure 5）进一步验证 Agent Attention 的分布质量：代理聚合保留的 Softmax 特性使模型对 corruption 更具鲁棒性。Figure 4 显示 Agent-Swin 系列在多种配置下 consistently 优于原版 Swin。Figure 7 的 FLOPs 对比表明，Agent Attention 在相近计算预算下优于纯 Linear Attention 方案。


![Table 5](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/61af2222-3da8-4a08-a347-6ab88086aaf5/figures/Table_5.png)
*Table 5 (ablation): Ablation for our attention modules.*



消融实验（Table 5）揭示关键设计选择：将动态代理令牌替换为静态可学习参数导致 accuracy **82.6% → 82.2%（-0.4）**，证明动态生成的必要性；而 pooling 与 DWC 两种动态方式效果持平（均为 82.6%），deformed points 略升至 82.7%（+0.1）。Table 7 的阶段替换实验显示：全阶段替换获最佳收益，但 Stage 4 单独替换微降 0.1，暗示最深层仍需原始 Softmax 的精细聚合。


![Figure 7](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/61af2222-3da8-4a08-a347-6ab88086aaf5/figures/Figure_7.png)
*Figure 7 (comparison): Comparisons of FLOPs between linear and our Agent Attention.*



公平性检查：对比 baseline 覆盖较全面，包括 Swin、PVT、DeiT、ConvNeXt、CSWin、MViTv2 等主流架构，以及 Flatten Transformer、SOFT、Hydra Attention 等高效注意力方案。但未与 Flash Attention、Sparse Attention 等更 recent 的高效方案对比；代理令牌数量 m 的选择缺乏深入理论分析；扩散模型实验（Figure 6 的 AgentSD）展示定性效果但定量细节有限。整体证据强度较高，主要结论经多任务、多架构交叉验证。

## 方法谱系与知识库定位

Agent Attention 属于 **高效注意力机制** 方法族，直接父方法为 **Linear Attention**（Katharopoulos et al., "Transformers are RNNs" [22]）。本文在 Linear Attention 的线性计算框架上，新增了关键的 Softmax 代理聚合步骤，形成"先聚合后广播"的两阶段结构。

**改变的 slots**：
- **attention_mechanism**: 从单步 QK^T 计算 → 两步 factorized 计算（代理聚合 + 代理广播）
- **architecture**: 新增显式的代理令牌生成模块（pooling/DWC）
- **inference_strategy**: 从直接计算完整 n×n 矩阵 → 通过 m 个代理令牌分解计算

**直接 baseline 及差异**：
- **Softmax Attention**: 表达能力相同但复杂度 O(n²)；Agent Attention 通过代理令牌降至 O(nm)
- **Linear Attention (Performer [7], Flatten [14])**: 复杂度同为线性但分布平坦；Agent Attention 通过 Softmax 聚合保留尖锐分布
- **Swin Transformer [29]**: 局部窗口注意力；Agent Attention 保持全局感受野
- **PVT/DeiT**: 标准 Transformer 变体；Agent Attention 即插即用替换其注意力块

**后续方向**：
1. 代理令牌数量的自适应选择：当前 m 为固定超参，探索输入依赖的动态 m 调整
2. 向多模态扩展：LLM 中的长序列场景、视频时序建模等 n 极大的场景
3. 理论刻画：代理令牌最优表示查询多样性的条件，以及 Stage 4 替换降效的深层原因

**标签**: 模态=图像 | 范式=监督学习 | 场景=高效视觉Transformer | 机制=注意力分解/代理令牌 | 约束=低复杂度+高表达能力

## 引用网络

### 后续工作（建立在本文之上）

- [[P__线性差分视觉Transforme_Visual-Contrast_]]: Integrates softmax and linear attention; directly relevant attention mechanism t
- [[P__InLine：可注入性与局部建模_InLine]]: Agent Attention directly integrates softmax and linear attention — very closely 

