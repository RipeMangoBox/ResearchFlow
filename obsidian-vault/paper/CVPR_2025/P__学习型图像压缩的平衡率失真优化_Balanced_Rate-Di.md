---
title: Balanced Rate-Distortion Optimization in Learned Image Compression
type: paper
paper_level: C
venue: CVPR
year: 2025
paper_link: null
aliases:
- 学习型图像压缩的平衡率失真优化
- Balanced Rate-Di
- Balanced Rate-Distortion Optimization (Two Solutions)
acceptance: poster
cited_by: 3
code_url: https://github.com/ppingzhang/Deep-Learning-Based-Image-Compression
method: Balanced Rate-Distortion Optimization (Two Solutions)
---

# Balanced Rate-Distortion Optimization in Learned Image Compression

[Code](https://github.com/ppingzhang/Deep-Learning-Based-Image-Compression)

**Topics**: [[T__Image_Generation]], [[T__Compression]] | **Method**: [[M__Balanced_Rate-Distortion_Optimization]] | **Datasets**: Kodak, Tecnick, CLIC 2022

| 中文题名 | 学习型图像压缩的平衡率失真优化 |
| 英文题名 | Balanced Rate-Distortion Optimization in Learned Image Compression |
| 会议/期刊 | CVPR 2025 (poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2502.20161) · [Code](https://github.com/ppingzhang/Deep-Learning-Based-Image-Compression) · [DOI](https://doi.org/10.1109/cvpr52734.2025.00232) |
| 主要任务 | 学习型图像压缩（Learned Image Compression）中的多目标率失真优化 |
| 主要 baseline | M&S Hyperprior [50]、ELIC [27]、TCM-S [43]、标准 R-D 优化（固定 λ） |

> [!abstract] 因为「标准率失真优化使用固定拉格朗日乘子 λ，无法自适应平衡多个目标」，作者在「Tchebycheff 标量化」基础上改了「训练目标为自适应梯度权重（Solution 1）和约束二次规划解析解（Solution 2）」，在「Kodak / Tecnick / CLIC 2022」上取得「ELIC BD-Rate 最高 -3.00%」

- **ELIC + Solution 1** 在 CLIC 2022 上 BD-Rate **-3.00%**，Tecnick 上 **-2.63%**，Kodak 上 **-2.68%**
- **训练开销**：Solution 1 每 epoch 增加 **+20-23%** 时间，Solution 2 增加 **+47-50%** 时间
- **模型参数量与推理延迟**：与 baseline 架构完全一致，零额外开销

## 背景与动机

学习型图像压缩（Learned Image Compression, LIC）的核心挑战在于同时优化两个相互竞争的目标：压缩率（Rate, R）和重建质量（Distortion, D）。现有方法通常采用标准率失真优化框架，即最小化加权损失 $L = R + \lambda D$，其中 λ 是固定的拉格朗日乘子。然而，这种固定权重的策略存在根本性缺陷：不同训练阶段、不同图像内容、不同码率点对 R 和 D 的敏感度差异巨大，固定 λ 无法自适应地平衡这些动态变化的需求。

现有方法如何处理这一问题？**M&S Hyperprior [50]** 作为经典基线，采用均值-尺度超先验架构，但完全依赖固定 λ 的优化；**ELIC [27]** 通过不均匀分组的空间-通道上下文自适应编码提升效率，其训练目标仍是标准加权求和；**TCM-S [43]** 引入 Transformer 空间上下文模型增强表达能力，但同样未改变固定权重的优化范式。此外，多目标优化领域已有 **Tchebycheff 标量化 [4]** 及其平滑变体 **[6]**，以及 **FAMO [15]**、**Direction-oriented Multi-objective Learning [18]** 等自适应优化方法，但这些方法尚未被系统性地应用于图像压缩的率失真优化中。

这些方法的共同短板在于：**固定 λ 导致优化轨迹偏向单一目标**，在高码率区域可能过度追求质量而浪费比特，在低码率区域可能过度压缩而牺牲视觉质量；同时，多目标优化方法虽能自适应平衡权重，但直接迁移到 LIC 面临数值不稳定性、收敛困难等挑战。本文提出两种互补解决方案，分别针对「从头训练」和「微调预训练模型」两种场景，实现真正意义上的平衡率失真优化。

## 核心创新

核心洞察：**将率失真优化重新建模为多目标优化问题，通过自适应权重机制替代固定 λ，从而使训练过程能够动态响应 R 和 D 的梯度竞争关系，实现更优的帕累托前沿。**

具体而言，本文发现标准 R-D 优化的本质缺陷在于权重空间的「刚性」——一旦 λ 固定，优化轨迹即被锁定。而多目标优化中的 Tchebycheff 标量化虽能处理竞争目标，但直接应用于 LIC 存在两大障碍：（1）梯度权重在训练末期因损失值变小而产生数值不稳定；（2）从头训练与微调预训练模型需要不同的优化动力学。

| 维度 | Baseline（标准 R-D 优化） | 本文 |
|:---|:---|:---|
| 权重形式 | 固定标量 λ | 自适应向量 $w_t$，逐迭代更新 |
| 优化目标 | 单目标加权求和 $L = R + \lambda D$ | 多目标标量化 $L(w_t)$，平衡多个 R-D 点 |
| 求解方式 | 闭式梯度下降 | Solution 1: 粗到精梯度下降 + softmax logits；Solution 2: 约束 QP 解析解 |
| 训练场景 | 统一策略 | Solution 1 专用于从头训练，Solution 2 专用于微调预训练模型 |
| 数值稳定 | 无特殊处理 | 权重重归一化（Weight Renormalization）+ 权重衰减（γ=0.001）|

## 整体框架

本文框架的核心设计是**解耦优化目标与网络架构**——不修改任何编码器/解码器结构，仅通过替换训练目标函数实现性能提升。整体数据流如下：

```
输入图像 → [标准 LIC 编码器/解码器] → 重建图像 + 码率估计
                    ↑
            训练阶段：自适应权重优化模块
            （Solution 1 或 Solution 2）
```

**模块分解：**

1. **标准 R-D 优化模型（Standard R-D Optimized Model）**：输入训练数据，输出预训练模型。这是现有 LIC 架构（M&S Hyperprior / ELIC / TCM-S）的标准训练结果，作为 Solution 2 的输入起点。

2. **Solution 1：自适应梯度加权训练（Adaptive Gradient Weighting）**：输入训练数据（从头训练场景），输出具有自适应梯度权重的模型。核心机制是维护一组无约束 softmax logits $w_{i,t}$，通过粗到精的梯度下降迭代优化，并配合权重衰减 γ=0.001 防止非收敛。

3. **Solution 2：约束 QP 微调（Constrained QP Fine-tuning）**：输入预训练标准模型（微调场景），输出 QP 优化权重后的微调模型。通过求解梯度最优性条件 $\nabla_{w_t} L = Qw_t - \lambda \mathbf{1} = 0$ 直接得到解析解，无需迭代优化权重。

4. **权重重归一化（Weight Renormalization）**：输入权重 $w_{i,t}$ 和损失 $L_{i,t}$，输出稳定化后的重归一化权重。应用于两种 Solution 的训练末期，解决损失值趋小时权重更新的数值不稳定问题。

两种 Solution 的选用策略经交叉验证确定：**Solution 1 专用于从头训练**（200 epochs for λ=0.0018，150 epochs fine-tuning for others），**Solution 2 专用于微调预训练模型**（50 epochs, lr=5e-5）。级联使用（S1+S2）仅带来边际提升，故不推荐。

## 核心模块与公式推导

### 模块 1: Solution 2 — 约束二次规划（Constrained QP）

**直觉**：将多目标权重的优化转化为带约束的二次规划问题，利用 LIC 损失函数的二次结构直接求解最优权重，避免迭代优化的不稳定性和计算开销。

**Baseline 公式**（标准 R-D 优化）：
$$L = R + \lambda D$$
符号：$R$ = 码率（bit-rate），$D$ = 失真（如 MSE 或 MS-SSIM），$\lambda$ = 固定拉格朗日乘子。

**变化点**：固定 λ 无法适应不同训练阶段 R 和 D 梯度的动态竞争关系。本文将问题重新参数化为权重向量 $w_t$ 的优化，并假设损失函数关于权重具有二次结构。

**本文公式（推导）**：
$$\text{Step 1}: \quad L(w_t) \text{ 建模为关于 } w_t \text{ 的二次型} \quad \text{（利用 Tchebycheff 标量化框架）}$$
$$\text{Step 2}: \quad \nabla_{w_t} L = Qw_t - \lambda \mathbf{1} = 0 \quad \text{（令梯度为零，得到 QP 最优性条件）}$$
$$\text{Step 3}: \quad w_t = Q^{-1}(\lambda \mathbf{1}) \quad \text{（解析求解，直接得到最优权重分配）}$$

符号：$Q$ = Hessian 相关矩阵（由 R 和 D 的梯度结构构成），$\lambda$ = 拉格朗日乘子，$\mathbf{1}$ = 全1向量。关键优势在于**无需迭代优化权重**，一次矩阵求逆即可得到解析精确解，特别适合微调场景的快速收敛。

**对应消融**：Figure 3(c) 的交叉验证显示，Solution 2 用于微调时显著优于 Solution 1 用于微调；反之 Solution 1 从头训练与 Solution 2 从头训练效果相当但收敛更快。

---

### 模块 2: Solution 1 — 自适应梯度加权（Adaptive Gradient Weighting）

**直觉**：从头训练时损失 landscape 复杂，解析假设难以成立，因此采用迭代式自适应加权，通过 softmax 参数化权重并结合权重衰减，实现粗到精的稳定优化。

**Baseline 公式**（标准多任务优化）：
$$L = \sum_i w_i L_i, \quad \sum_i w_i = 1, \quad w_i \geq 0$$

**变化点**：直接硬约束 $w_i \geq 0$ 导致优化困难；瞬时梯度更新易引起权重震荡和非收敛。本文改用**无约束 softmax logits** 参数化，并引入权重衰减稳定训练末期行为。

**本文公式（推导）**：
$$\text{Step 1}: \quad \tilde{w}_{i,t} = \text{softmax}(\phi_{i,t}) \quad \text{（logits } \phi_{i,t} \text{ 无约束，通过 softmax 保证正性和归一化）}$$
$$\text{Step 2}: \quad \phi_{i,t+1} = \phi_{i,t} - \eta \nabla_{\phi} L - \gamma \phi_{i,t} \quad \text{（加入权重衰减项 } \gamma=0.001 \text{，缓解瞬时梯度导致的非收敛）}$$
$$\text{Step 3}: \quad w_{i,t} = \frac{\tilde{w}_{i,t} L_{i,t}}{\sum_j \tilde{w}_{j,t} L_{j,t}} \quad \text{（训练末期权重重归一化，解决损失值趋小时的数值不稳定）}$$

**最终更新**：梯度权重 $w_{i,t}$ 用于加权各目标（不同 λ 对应的 R-D 点）的梯度，实现多目标联合优化。

**对应消融**：Figure 3(b) 显示 γ=0 时完全不收敛（结果无法绘制），γ=0.001 时性能最优，其他值 {0.01, 0.015, 0.005, 0.0005} 均更差。

---

### 模块 3: 权重重归一化（Weight Renormalization）

**直觉**：训练末期各目标损失值趋于很小，直接梯度加权会导致权重更新被数值误差主导，需要显式归一化维持稳定。

**Baseline**：无此机制，标准多目标优化方法（如 FAMO [15]）未针对 LIC 的损失尺度特性做特殊处理。

**本文公式**：
$$w_{i,t}^{\text{renorm}} = \frac{w_{i,t} L_{i,t}}{\sum_j w_{j,t} L_{j,t}}$$

**对应消融**：Figure 3(a) 显示去掉重归一化后，Solution 1 和 Solution 2 的最终收敛性均变差（PSNR/bpp trade-off 恶化）。

## 实验与分析

本文在三个标准图像压缩基准数据集上进行评估：Kodak（24 张经典测试图）、Tecnick 1200×1200（高分辨率图像集）、以及 CLIC 2022（学习型图像压缩挑战赛数据集）。评估指标为 BD-Rate（负值表示相对于 anchor 的码率节省），anchor 为各架构重新训练的标准 R-D 优化模型（固定 λ）。



Table 1 汇总了主要结果。以 ELIC 架构为例，Solution 1 在三个数据集上分别取得 BD-Rate **-2.68%**（Kodak）、**-2.63%**（Tecnick）、**-3.00%**（CLIC 2022）；Solution 2 作为微调方案，相应结果为 **-1.81%**、**-2.03%**、**-2.39%**。M&S Hyperprior 和 TCM-S 架构上两种 Solution 也均有稳定提升。这一提升的实质意义在于：**在相同重建质量（PSNR）下，本文方法可节省约 2-3% 的码率**，且优势在高 bpp（高码率）区域尤为明显——这与固定 λ 优化在高码率时过度追求质量、未能有效分配比特的缺陷直接对应。



Figure 2 的 R-D 曲线直观展示了这一规律：提出的 S1/S2 曲线系统性地位于标准方法左上方，且 ELIC 架构上的 gap 最大，说明本文优化方法与现代高效架构的兼容性更佳。



消融实验（Figure 3 系列）揭示了关键设计决策：
- **权重重归一化**（Figure 3a）：移除后两种 Solution 的最终收敛均变差，验证了其必要性；
- **权重衰减系数 γ**（Figure 3b）：γ=0 导致完全不收敛，γ=0.001 最优，敏感度显著；
- **交叉验证**（Figure 3c）：Solution 1 用于微调劣于 Solution 2，Solution 2 用于从头训练与 S1 相当但更慢，验证了「S1 从头训练、S2 微调」的策略分工。

**公平性检验**：本文比较的是重新训练的「Standard」模型而非官方 checkpoint，可能引入实现差异；未与同为多目标优化的 [7]（Variable-Rate LIC with Multi-Objective Optimization）、[12]（R-D-Computation Frontier）、[20]（R-D-Complexity Optimization）直接对比，这些是方法谱系上最直接的参照。训练开销方面，Solution 2 的 +47-50% epoch 时间对于微调场景尚可接受，但 Solution 1 的 +20-23% 增量在从头训练时具有较好的性价比。作者未报告失败模式或 bad case 分析。

## 方法谱系与知识库定位

**方法家族**：多目标优化 → Tchebycheff 标量化 / 平滑 Tchebycheff 标量化 → 学习型图像压缩中的自适应率失真优化

**父方法**：**Smooth Tchebycheff Scalarization [6]** 是最直接的谱系源头，本文将其从通用多目标优化框架适配到 LIC 领域，并针对图像压缩的特殊需求（损失尺度变化大、从头训练 vs 微调场景差异）做了两项关键扩展：

| 改动槽位 | 父方法 | 本文改动 |
|:---|:---|:---|
| objective | 通用多目标标量化 | 专用于 R-D 优化的自适应权重机制 |
| training_recipe | 统一优化过程 | Solution 1（从头训练）与 Solution 2（微调）分工 |
| application_domain | 通用优化 | 学习型图像压缩，加入权重重归一化应对损失尺度问题 |

**直接基线对比**：
- **vs 标准 R-D 优化（固定 λ）**：替换目标函数，权重自适应动态调整
- **vs FAMO [15]**：同为自适应多任务优化，但本文针对 LIC 引入 QP 解析解和权重重归一化
- **vs Direction-oriented Multi-objective Learning [18]**：借鉴可证明收敛的随机算法思想，但本文聚焦 R-D 特定的数值稳定性

**后续方向**：
1. 将框架扩展到 **R-D-复杂度三目标优化**（直接回应 [12][20] 的方向）；
2. 探索 **Variable-Rate** 场景下的动态权重调整（与 [7] 的未竟对话）；
3. 权重自适应机制向 **视频压缩** 和 **3D 场景表示压缩** 的迁移。

**标签**：modality=图像 / paradigm=端到端学习型压缩 / scenario=有损压缩 / mechanism=多目标优化+自适应权重+二次规划解析解 / constraint=保持基线架构零修改

