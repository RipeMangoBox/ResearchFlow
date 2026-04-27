---
title: On the Surprising Effectiveness of Large Learning Rates under Standard Width Scaling
type: paper
paper_level: C
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 标准参数化下大学习率的宽度缩放理论
- SP-full-align (S
- SP-full-align (Standard Parameterization with Full Alignment)
acceptance: Spotlight
cited_by: 5
code_url: github.com/tml-tuebingen/torch-module-monitor
method: SP-full-align (Standard Parameterization with Full Alignment)
---

# On the Surprising Effectiveness of Large Learning Rates under Standard Width Scaling

[Code](github.com/tml-tuebingen/torch-module-monitor)

**Method**: [[M__SP-full-align]] | **Datasets**: Width scaling for 8-layer MLPs, 1.4B GPT model

| 中文题名 | 标准参数化下大学习率的宽度缩放理论 |
| 英文题名 | On the Surprising Effectiveness of Large Learning Rates under Standard Width Scaling |
| 会议/期刊 | NeurIPS 2025 (Spotlight) |
| 链接 | [arXiv](https://arxiv.org/abs/2505.22491) · [Code](https://github.com/tml-tuebingen/torch-module-monitor) · [Project](https://arxiv.org/abs/2505.22491) |
| 主要任务 | 宽度无关的神经网络训练（width-independent neural network training） |
| 主要 baseline | μP (Maximal Update Parameterization)、SP (Standard Parameterization)、Adam |

> [!abstract] 因为「μP 需要非标准参数化才能保持宽度无关的训练动态」，作者在「μP」基础上改了「保留标准参数化（SP），通过补偿梯度缩放来调整学习率缩放」，在「8层 MLP 宽度至 16384 及 1.4B GPT-2」上取得「与 μP 相当的宽度无关训练效果，且无需修改参数化方案」

- 8层 MLP 宽度扩展至 16384 仍保持宽度无关的训练动态
- 1.4B 参数 GPT-2（宽度 4096）在单张 A100 上 24 小时内成功训练
- 标准参数化（SP）配合调整后的学习率缩放即可替代 μP 的非标准参数化方案

## 背景与动机

神经网络宽度缩放是深度学习理论中的核心问题：当网络隐藏层维度 n 增大时，如何保持训练动态稳定且可预测？一个具体例子是，将 MLP 的隐藏层从 512 扩至 16384 时，若学习率设置不当，梯度更新会爆炸或消失，导致训练崩溃。

现有方法主要从三个方向处理此问题：

**μP (Maximal Update Parameterization)** 是当前宽度无关训练的标准理论，通过特殊的初始化方案（如调整权重方差）和学习率分层缩放（隐藏层 η_l = Θ(n^{-1})，最后一层 η_{L+1} = Θ(1)）来保证宽度无关性。然而，μP 要求修改参数化方案本身，无法直接应用于已有的标准实现。

**Standard Parameterization (SP)** 即最常见的参数化方式（如 PyTorch 默认初始化），配合朴素的学习率缩放（如所有层统一缩放）在大宽度下会失效——梯度缩放行为改变导致特征更新幅度随宽度剧烈波动。

**NTK parameterization** 等方案则走向另一极端，让网络在无限宽度极限下表现为核方法，但牺牲了特征学习能力，不适用于实际深度训练。

这些方法的共同短板在于：μP 虽有效但需要"改造"网络本身；SP 虽通用但缺乏正确的学习率缩放理论；而核方法极限则丢失了深度学习的核心优势。特别地，一个被忽视的关键现象是：在标准参数化下，反向传播梯度 ∂f/∂x^L 的缩放行为并非 μP 假设的 Θ(n^{-1})，而是 Θ(n^{-min(b_{L+1},c_{L+1})})，这一差异直接导致朴素学习率缩放失效。

本文的核心动机即：能否在完全保留标准参数化的前提下，仅通过调整学习率缩放，实现与 μP 等价的宽度无关训练？


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/9b496f52-eb7e-44b4-9855-c1211497741d/figures/Figure_3.png)
*Figure 3: Learning rate regimes for SGD in a quadratic bowl. In blue and shaded area: update divergence; in orange: convergent...*



## 核心创新

核心洞察：标准参数化下梯度缩放的改变可以被精确刻画并通过学习率补偿，因为梯度与激活之间的对齐指数（alignment exponents）是梯度训练的涌现性质而非人为设计，从而使纯学习率调整替代非标准参数化成为可能。

| 维度 | Baseline (μP) | 本文 (SP-full-align) |
|:---|:---|:---|
| 参数化方案 | 非标准：需修改初始化方差、学习率分层规则 | 标准：PyTorch 默认初始化，仅调整学习率 |
| 梯度缩放假设 | ∂f/∂x^L = Θ(n^{-1})（由参数化强制保证） | ∂f/∂x^L = Θ(n^{-min(b_{L+1},c_{L+1})})（自然涌现） |
| 学习率缩放 | 隐藏层 η_l = Θ(n^{-1})，最后一层 Θ(1) | 隐藏层 η_l = Θ(n^{-min(b_{L+1},c_{L+1})})，补偿梯度变化 |
| 对齐指数来源 | 理论预设 p_l=1, q_l=1/2 | 训练涌现 p_{1:L+1}=1, q_{1:L}=1/2, q_{L+1}=1 |
| 实现成本 | 需重写模型初始化逻辑 | 仅修改优化器学习率参数 |

## 整体框架


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/9b496f52-eb7e-44b4-9855-c1211497741d/figures/Figure_2.png)
*Figure 2 (result): Alignment loss minimised with μ-dependence. Alignment loss between uncentered weight update ΔW and incoming activation x...*



SP-full-align 的训练流程包含四个核心阶段，完全基于标准参数化实现：

**输入**：标准初始化的网络权重 W、训练数据 x、Adam 优化器（含小 ε 设置）

**阶段 1 — 前向传播（Forward pass with standard parameterization）**：使用标准 PyTorch 初始化（如 Kaiming uniform）执行常规前向计算，输出各层激活 h^l 和最终 logits f。无需任何参数化修改。

**阶段 2 — 反向传播（Backward pass with gradient computation）**：计算损失梯度并反向传播，得到各层参数梯度 ∇_{W^l} L。此步骤与标准训练完全一致。

**阶段 3 — Adam ε-归一化（Adam with ε-normalization）**：使用 Adam 优化器，关键配置为足够小的 ε 值。小 ε 使得 Adam 的更新近似于逐层梯度归一化，这是保证对齐行为正确的关键。输出为归一化后的梯度更新方向。

**阶段 4 — SP-full-align 学习率缩放（SP-full-align learning rate scaling）**：核心创新模块。根据层索引 l、网络宽度 n 和理论推导的梯度缩放指数，为每层计算补偿后的学习率 η_l = Θ(n^{-min(b_{L+1},c_{L+1})})。该缩放抵消标准参数化下梯度缩放的改变，确保特征更新的宽度无关性。

**输出**：宽度无关的权重更新，使得无论网络宽度如何，有效特征学习幅度保持稳定。

```
Input (x, W_standard) 
  → Forward: h^l, f  [标准参数化]
  → Backward: ∇_{W^l} L  [标准梯度计算]
  → Adam ε-norm: 归一化梯度  [小ε Adam]
  → SP-full-align LR: η_l = Θ(n^{-min(b_{L+1},c_{L+1})})  [补偿缩放]
  → Update: W_{t+1} = W_t - η_l · update_direction
  → Output: 宽度无关的训练动态
```

## 核心模块与公式推导

### 模块 1: 梯度缩放分析（对应框架图 阶段 2→4）

**直觉**：标准参数化下，最后一层反向传播梯度的宽度缩放行为与 μP 不同，必须精确刻画才能设计补偿机制。

**Baseline 公式 (μP)**:
$$\frac{\partial f}{\partial x^L} = \Theta(n^{-1}) \quad \text{(under μP)}$$
符号: $f$ = 网络输出 logits, $x^L$ = 最后一层隐藏激活, $n$ = 隐藏层宽度, $\Theta(\cdot)$ = 渐进紧界

**变化点**：μP 通过特殊参数化强制梯度按 $n^{-1}$ 缩放；但标准参数化下，权重 $W^{L+1}_t$ 的演化不受此约束，其缩放指数由初始化超参数 $b_{L+1}, c_{L+1}$（控制权重初始方差和学习率）共同决定。

**本文公式（推导）**:
$$\text{Step 1}: \frac{\partial f}{\partial x^L} = W^{L+1}_t \quad \text{(精确表达式，权重矩阵本身)}$$
$$\text{Step 2}: W^{L+1}_t = W^{L+1}_0 - \eta \sum_{s=0}^{t-1} \text{(updates)} \Rightarrow \Theta(n^{-\min(b_{L+1},c_{L+1})}) \quad \text{(考虑初始化与训练演化)}$$
$$\text{最终}: \frac{\partial f}{\partial x^L} = \Theta(n^{-\min(b_{L+1},c_{L+1})})$$

**对应消融**：Figure 4 显示，未补偿的标准 SP 在大学习率下出现 sharp divergence（训练损失突增），而应用补偿后恢复稳定学习。

### 模块 2: 学习率缩放补偿（对应框架图 阶段 4）

**直觉**：既然梯度缩放了 $n^{-\min(b_{L+1},c_{L+1})}$，学习率需同等补偿以保持有效更新幅度。

**Baseline 公式 (μP)**:
$$\eta_l^{\mu P} = \Theta(n^{-1}) \text{ (hidden layers)}, \quad \eta_{L+1}^{\mu P} = \Theta(1) \text{ (last layer)}$$
符号: $\eta_l$ = 第 $l$ 层学习率, $l \in \{1,...,L+1\}$

**变化点**：μP 的 $n^{-1}$ 缩放基于其强制参数化；SP-full-align 需匹配实际涌现的梯度缩放 $\Theta(n^{-\min(b_{L+1},c_{L+1})})$，且该指数可能小于 1（即需要更激进的学习率衰减）。

**本文公式（推导）**:
$$\text{Step 1}: \Delta W^l \propto \eta_l \cdot \nabla_{W^l}L \propto \eta_l \cdot \frac{\partial f}{\partial h^l} \cdot (x^{l-1})^\text{top} \quad \text{(梯度-激活外积结构)}$$
$$\text{Step 2}: \text{要求 } \Delta W^l \cdot x^{l-1} = \Theta(1) \text{ (宽度无关的特征更新)}$$
$$\text{Step 3}: \eta_l \cdot \Theta(n^{-\min(b_{L+1},c_{L+1})}) \cdot \Theta(n^{1/2}) \cdot \Theta(n^{1/2}) = \Theta(1) \Rightarrow \eta_l = \Theta(n^{-\min(b_{L+1},c_{L+1})})$$
$$\text{最终}: \eta_l^{\text{SP-full-align}} = \Theta(n^{-\min(b_{L+1},c_{L+1})}) \text{ (hidden layers)}$$

**对应消融**：Figure 5 显示，基于该缩放的预测与 CIFAR-10 上经验最优学习率高度吻合；去掉补偿（即使用朴素 SP）则宽度扩展时训练失败。

### 模块 3: 对齐指数理论（对应框架图 理论支撑）

**直觉**：梯度与激活的对齐程度并非固定，而是梯度优化过程的涌现性质，这解释了为何小 ε Adam 能自然产生正确的对齐行为。

**Baseline 公式 (infinite-width theory)**:
$$p_{1:L+1} = 1, \quad q_{1:L} = \frac{1}{2}, \quad q_{L+1} = 1 \quad \text{(理论预设，与优化器无关)}$$
符号: $p_l$ = 梯度 $\partial f/\partial h^l$ 的缩放指数, $q_l$ = 激活 $x^l$ 的缩放指数

**变化点**：传统理论假设这些指数为固定常数；本文发现它们是梯度训练的动态结果，且 Adam 的 ε-归一化通过影响梯度统计量间接调控这些指数。

**本文公式（推导）**:
$$\text{Step 1}: \nabla_{W^l} L = \chi \frac{\partial f}{\partial h^l} (x^{l-1})^\text{top} \quad \text{(损失梯度分解，χ 为全局梯度)}$$
$$\text{Step 2}: \text{Adam update} \approx \frac{\nabla_{W^l}L}{\sqrt{\mathbb{E}[g^2]} + \epsilon} \approx \frac{\nabla_{W^l}L}{||\nabla_{W^l}L||_F/\sqrt{nd}} \text{ (小ε时逐层归一化)}$$
$$\text{Step 3}: \text{Central limit effect} \Rightarrow p_l = \frac{1}{2}, q_l = \frac{1}{2} \text{ (有限宽度下的涌现行为)}$$
$$\text{最终}: p_{1:L+1} = 1, \quad q_{1:L} = \frac{1}{2}, \quad q_{L+1} = 1 \text{ (SP-full-align 实际观测值)}$$

**对应消融**：Figure 2 显示 alignment loss 在 μP 依赖下被最小化，验证了指数选择的正确性；Figure 6 显示宽网络与标准 CNN/ResNet 在控制发散边缘处保持 LR 可分离性。

## 实验与分析



本文在三个层面验证 SP-full-align 的有效性：MLP 宽度缩放、大规模语言模型训练、以及学习率转移预测。

**核心结果**：在 8 层 MLP 上，SP-full-align 实现宽度至 16384 的完全宽度无关训练，与 μP 性能相当但无需修改参数化。Figure 1 显示最优学习率在 MNIST、CIFAR-10 和 GPT-2 上均呈现与宽度反比缩放的趋势——这与传统认知（大宽度需更小学习率）一致，但 SP-full-align 揭示了该缩放的确切指数应为 $-\min(b_{L+1},c_{L+1})$ 而非简单的 $-1$。Figure 4 进一步展示关键现象：在 sharp divergence 边缘之后，SP-full-align 仍能恢复 hidden-features learning，训练损失下降且 weight alignment 持续改善，这是朴素 SP 无法实现的。


![Figure 6](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/9b496f52-eb7e-44b4-9855-c1211497741d/figures/Figure_6.png)
*Figure 6 (result): LR separable at the edge of controlled divergence. Train/test loss and alignment of wide and standard CNNs and ResNets...*



**大规模验证**：1.4B 参数 GPT-2（宽度 4096）在单张 A100 上 24 小时内成功训练（Figure 5 及实验描述），证明方法可扩展至实际大模型场景。



**消融分析**：Figure 5 展示学习率转移预测——基于理论推导的缩放规则，从窄网络（宽度 256）预测宽网络（宽度 2048）的最优学习率，在 CIFAR-10 上预测值与经验最优值吻合。这等价于最强消融：若缩放指数错误，预测将系统性偏离。Figure 6 的 LR 可分离性实验显示，宽网络与标准 CNN/ResNet 在发散边缘附近仍能保持训练/测试损失和 alignment 的稳定分离，验证补偿机制的有效性。Figure 7 的比较表明，μP 使其他损失函数（如 focal loss、label smoothing）具备竞争力——SP-full-align 继承此优势但无需参数化改造。

**公平性检查**：主要对比 baseline μP 是宽度缩放领域的标准理论，对比公平；但缺少与 NTK parameterization、Edge of Stability (EoS) 理论等近期工作的直接比较。计算资源披露较简略（单 A100 / RTX 2080 Ti，每运行 <24 小时），未报告总项目计算量。作者未明确讨论失败模式，但 Figure 3 暗示存在明确的收敛/发散边界（蓝色区域为 update divergence，橙色为 convergent），实际应用需避免进入蓝色区域。

## 方法谱系与知识库定位

**方法家族**：Neural Network Width Scaling / Mean-Field Theory

**父方法**：μP (Maximal Update Parameterization, Yang et al.) — SP-full-align 直接继承其"宽度无关训练"目标，但通过替换核心机制实现。

**关键改动槽位**：
- **Architecture**：μP 的非标准参数化 → SP 的标准参数化（保留）
- **Training recipe**：μP 的固定 $n^{-1}$ 学习率缩放 → 补偿式 $n^{-\min(b_{L+1},c_{L+1})}$ 缩放（创新）
- **Objective dynamics**：理论预设对齐指数 → 梯度训练涌现对齐指数（创新）

**直接 baselines**：
- **μP**：SP-full-align 达到等价宽度无关性，但 μP 需修改参数化，SP-full-align 仅调学习率
- **Standard SP + naive LR**：大宽度下训练失败，SP-full-align 通过补偿解决
- **Adam (小 ε)**：作为组件被采用，其逐层归一化特性是对齐指数涌现的关键

**后续方向**：
1. 将补偿理论扩展至其他优化器（如 Lion、Shampoo），验证对齐指数的优化器无关性
2. 结合 Edge of Stability (EoS) 理论，解释大学习率下的隐式正则化效应
3. 应用于混合专家（MoE）模型，处理动态宽度变化场景

**知识库标签**：
- **Modality**: 通用（MLP / CNN / Transformer / GPT）
- **Paradigm**: 监督学习 / 预训练
- **Scenario**: 大宽度网络训练、超参数迁移
- **Mechanism**: 梯度缩放补偿、对齐指数分析、学习率缩放理论
- **Constraint**: 标准参数化兼容、单卡可训练（1.4B 模型 24h A100）

