---
title: Flow Equivariant Recurrent Neural Networks
type: paper
paper_level: A
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 流等变循环神经网络FERNN
- FERNN (Flow Equi
- FERNN (Flow Equivariant Recurrent Neural Network)
- Flow equivariant recurrent neural n
acceptance: Spotlight
cited_by: 4
code_url: https://kempnerinstitute.harvard.edu/research/deeper-learning/flow-equivariant-recurrent-neural-networks/
method: FERNN (Flow Equivariant Recurrent Neural Network)
modalities:
- Video
- Image
paradigm: supervised
---

# Flow Equivariant Recurrent Neural Networks

[Code](https://kempnerinstitute.harvard.edu/research/deeper-learning/flow-equivariant-recurrent-neural-networks/)

**Topics**: [[T__Video_Understanding]], [[T__Time_Series_Forecasting]] | **Method**: [[M__FERNN]] | **Datasets**: Flowing MNIST, Moving KTH

> [!tip] 核心洞察
> Flow equivariant recurrent neural networks (FERNNs) that encode one-parameter Lie subgroup symmetries over time achieve superior training speed, length generalization, and velocity generalization compared to non-equivariant RNNs on sequence prediction and classification tasks.

| 中文题名 | 流等变循环神经网络FERNN |
| 英文题名 | Flow Equivariant Recurrent Neural Networks |
| 会议/期刊 | NeurIPS 2025 (Spotlight) |
| 链接 | [arXiv](https://arxiv.org/abs/2507.14793) · [Code](https://kempnerinstitute.harvard.edu/research/deeper-learning/flow-equivariant-recurrent-neural-networks/) · [Project](https://kempnerinstitute.harvard.edu/research/deeper-learning/flow-equivariant-recurrent-neural-networks/) |
| 主要任务 | Video Understanding, Time Series Forecasting |
| 主要 baseline | G-RNN (Group-equivariant RNN), standard RNN, escnn library |

> [!abstract] 因为「传统等变网络只能处理静态空间变换，无法利用时间参数化的连续流对称性」，作者在「G-RNN」基础上改了「将隐藏状态定义在流生成元与空间群的乘积空间 V×G 上，并引入流卷积和流位移算子」，在「Flowing MNIST 和 Moving KTH」上取得「Test MSE 相比 G-RNN 提升 54×，且实现长度泛化和速度泛化」。

- **Flowing MNIST (V^T_2)**: Test MSE 1.5e-4 ± 2e-5，相比 G-RNN (8.1e-3 ± 6e-4) 提升 **54×**
- **Flowing MNIST (V^R_4)**: Test MSE 6.1e-4 ± 3e-5，相比 G-RNN (4.0e-3 ± 5e-4) 提升 **6.6×**
- **参数效率**: FERNN 与 G-RNN 参数量完全相同，通过 V 维度参数共享实现

## 背景与动机

自然视频中的运动并非离散跳跃，而是由连续的时间参数化变换所支配——例如相机平移、物体旋转等。这些变换在数学上对应 Lie 群的单参数子群（one-parameter Lie subgroups），即"流"（flow）。然而，现有的等变神经网络仅针对静态空间对称性设计：给定一帧图像，网络保证输出在输入旋转或平移时相应变换；但当面对序列数据时，模型无法利用"速度"这一关键信息——不同速度的同一运动在隐藏状态中被同等对待，导致模型必须从零学习动态规律。

现有方法如何处理这一问题？**G-RNN**（Group-equivariant RNN）将群卷积引入循环结构，隐藏状态定义在空间群 G 上，保证了单帧的空间等变性，但其递推关系 h_{t+1} = σ(h_t ⋆_G W + f_t ⋆_G U) 完全忽略了时间演化中的流信息。**标准 RNN** 和 **ConvLSTM** 类方法虽能建模时序，但缺乏任何等变性保证，面对未见过的速度或更长序列时泛化能力有限。**Equivariant deep dynamical model** [2] 虽关注运动预测，但采用前馈架构而非循环结构，无法处理任意长度的在线序列。

这些方法的共同短板在于：**将"空间等变"与"时间动态"割裂处理**。G-RNN 的隐藏状态 h_t(g) 仅依赖空间位置 g ∈ G，不包含速度 ν；当输入序列的速度改变时，隐藏状态的演化路径无法自适应调整。作者通过理论反例证明（Figure 2）：即使所有子组件都是等变的，标准 RNN 和 G-RNN 整体上仍不是流等变的——输入流的变化不会引起隐藏状态的相应几何变换。这一根本局限促使作者提出将隐藏状态扩展到流生成元空间 V 与空间群 G 的乘积空间，使网络天生"知道"速度如何影响未来状态。
![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/b4eb803b-4f5a-40ab-a42b-3b42f14ee847/figures/fig_001.png)
*Figure: Static vs. Flow*



本文的核心思路是：让循环网络的隐藏状态本身成为流的函数，从而将时间参数化的对称性嵌入架构归纳偏置。

## 核心创新

核心洞察：**序列模型的隐藏状态应当定义在流生成元与空间群的联合空间上**，因为自然运动由 Lie 代数元素（速度）参数化，从而使网络对任意速度的时间演化具有内置等变性，无需从数据中学习。

| 维度 | Baseline (G-RNN) | 本文 (FERNN) |
|:---|:---|:---|
| 隐藏状态空间 | h_t(g)，仅空间群 G | h_t(ν, g)，流生成元 V × 空间群 G |
| 卷积操作 | 群卷积 ⋆_G，仅混合空间位置 | 流卷积 ⋆_{V×G}，同时混合速度与空间位置 |
| 递推动力学 | 无显式流演化，速度信息丢失 | 流位移 ψ_1(ν)·g，隐藏状态随速度自适应偏移 |
| 速度泛化 | 需重新学习不同速度 | 内置等变，零样本泛化到新速度 |
| 参数量 | N | 相同 N（V 维度参数共享 + max-pool）|

## 整体框架


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/b4eb803b-4f5a-40ab-a42b-3b42f14ee847/figures/fig_002.png)
*Figure: G-RNNs are not generally flow equivariant. We*



FERNN 的数据流遵循"提升 → 流演化 → 池化 → 解码"的四阶段范式：

1. **输入 Lifting 卷积**（Input lifting convolution）：接收当前帧 f_t ∈ X（像素空间），通过卷积核 U 将特征映射到 V × G 空间。在 trivial lift 变体中，输出沿 ν 维度恒定；在非平凡 lift 中，输出随 ν 流动。

2. **流卷积层**（Flow convolution layer）：对隐藏状态 h_t(ν, g) 执行 ⋆_{V×G} 操作，同时在速度维度 V 和空间维度 G 上混合信息。这是标准群卷积向乘积空间的自然扩展，实现为高效的多维卷积。

3. **FERNN 递推与流位移**（FERNN recurrence with flow shift）：核心更新模块。Trivial lift 下，隐藏状态经流卷积后施加单步流变换 ψ_1(ν)·g，再与 lifting 后的输入相加；非平凡 lift 下流已融入 lifting，递推形式更简洁。

4. **V-MaxPooling**（V-maxpool）：在 ν 维度取最大值，将 h_{t+1}(ν, g) 压缩为 h_{t+1}(g)，确保与 G-RNN 相同的解码前维度，实现参数共享。

5. **CNN 解码器**（CNN decoder）：将池化后的隐藏状态映射为预测帧 f̂_{t+1}。

```
f_t → [Lift: f_t ⋆̂_{V×G} U] → h_t(ν,g) on V×G
                              ↓
                         [Flow Conv ⋆_{V×G}]
                              ↓
                    [Flow Shift ψ_1(ν)·g] (trivial)
                              ↓
                         h_{t+1}(ν,g)
                              ↓
                    [Max over ν] → h_{t+1}(g)
                              ↓
                         [CNN Decoder]
                              ↓
                         f̂_{t+1}
```

## 核心模块与公式推导

### 模块 1: 流卷积 Flow Convolution（框架图步骤 2）

**直觉**: 标准群卷积仅在空间位置间混合信息，但运动预测需要同时考虑"以什么速度移动"和"移动到何处"。

**Baseline 公式** (G-RNN [12]): 
$$[h_t \star_G W](g) = \sum_{g' \in G} h_t(g') W(g^{-1} \cdot g')$$
符号: $g, g' \in G$ 为空间群元素（如平移/旋转），$\cdot$ 为群作用，$W$ 为共享卷积核。

**变化点**: G-RNN 的隐藏状态不含速度信息，同一物体的快慢运动被同等处理。需将卷积定义域扩展到流生成元空间 V。

**本文公式（推导）**:
$$\text{Step 1}: \text{定义乘积空间索引} \quad (\nu, g) \in V \times G$$
$$\text{Step 2}: \text{引入生成元组合} \odot \text{与群作用} \cdot \text{的联合操作}$$
$$\text{最终}: [h_t \star_{V \times G} W](\nu, g) = \sum_{\nu' \in V} \sum_{g' \in G} h_t(\nu', g') W(\nu^{-1} \odot \nu', g^{-1} \cdot g')$$
其中 $\odot$ 为流生成元的组合运算（如速度相加），$\cdot$ 为群作用。该卷积实现为 N 维卷积，计算复杂度随 $|V|$ 线性增长。

**对应消融**: Figure 11 显示运行时和内存随 $|V|$ 线性缩放；Table 1 显示 FERNN-V^T_1（$|V|=1$）vs FERNN-V^T_2（$|V|=2$），后者 MSE 从 5.3e-4 降至 1.5e-4，提升 3.5×。

---

### 模块 2: FERNN 递推 — Trivial Lift（框架图步骤 3）

**直觉**: 隐藏状态不仅要"记录当前速度"，还要"按该速度演化一步"，才能对未来状态预测具备等变性。

**Baseline 公式** (G-RNN):
$$h_{t+1} = \sigma(h_t \star_G W + f_t \star_G U)$$
符号: $\sigma$ 为激活函数，$f_t$ 为输入帧，$U$ 为输入卷积核。

**变化点**: G-RNN 的递推无速度相关的状态偏移。当输入序列整体加速时，隐藏状态的演化轨迹不会相应拉伸，破坏流等变性。需引入流位移算子 $\psi_1(\nu)$，表示由生成元 $\nu$ 诱导的单步流变换。

**本文公式（推导）**:
$$\text{Step 1}: \text{隐藏状态扩展} \quad h_t \rightarrow h_t(\nu', g') \text{ on } V \times G$$
$$\text{Step 2}: \text{流卷积后施加流位移} \quad [h_t \star_{V \times G} W](\psi_1(\nu) \cdot g)$$
$$\text{Step 3}: \text{输入 lifting 保持与 } \nu \text{ 无关（trivial）} \quad [f_t \hat{\star}_{V \times G} U](\nu, g)$$
$$\text{最终}: h_{t+1}(\nu, g) = \sigma\left(\left[h_t \star_{V \times G} W\right](\psi_1(\nu) \cdot g) + \left[f_t \hat{\star}_{V \times G} U\right](\nu, g)\right)$$
关键：$\psi_1(\nu) \cdot g$ 将群元 $g$ 沿流 $\nu$ 推进一步，使隐藏状态的时空演化与输入速度耦合。

**对应消融**: Table 1 显示去掉流位移（退化为 G-RNN）导致 Flowing MNIST (V^T_2) 上 MSE 从 1.5e-4 恶化至 8.1e-3，差距 **54×**。

---

### 模块 3: 非平凡 Lifting 变体（附录理论扩展）

**直觉**: 若将流变换直接融入输入卷积，可简化递推形式，同时保持等变性。

**Baseline 公式**: Trivial lift 的输入卷积 $[f_t \hat{\star}_{V \times G} U](\nu, g)$ 输出沿 $\nu$ 恒定。

**变化点**: Trivial lift 要求递推中的显式流位移；非平凡 lift 将时间参数化融入卷积核本身，使 lifting 输出天然随 $\nu$ 流动。

**本文公式（推导）**:
$$\text{Step 1}: \text{时间参数化核} \quad U_i^k(g^{-1} \cdot \psi_t(\nu)^{-1} \cdot x)$$
$$\text{Step 2}: \text{ lifting 输出随流演化} \quad [f_t \hat{\star}_{V \times G} U_i](\nu, g) = \sum_{x \in X} \sum_{k=1}^{K} f_k(x) U_i^k(g^{-1} \cdot \psi_t(\nu)^{-1} \cdot x)$$
$$\text{Step 3}: \text{证明等变性简化为 V 维度平移} \quad (\psi(\hat{\nu}) \cdot h[f])_t(\nu, g) = h_t[f](\nu - \hat{\nu}, g)$$
$$\text{最终递推}: h_{t+1}(\nu, g) = \sigma\left(\left[h_t \star_{V \times G} W\right](\nu, g) + \left[f_t \hat{\star}_{V \times G} U\right](\nu, g)\right)$$
此时流作用仅引起 $\nu$ 维度的平移（置换），无需额外的 $g$ 维度位移，递推形式与 G-RNN 更为相似但等变性更强。该变体在附录 Equations 99-104 中严格证明，但未在主实验中进行实证比较。

## 实验与分析



本文在 **Flowing MNIST** 和 **Moving KTH** 两个基准上评估 FERNN。Flowing MNIST 通过 SE(2) 群作用生成：数字以恒定速度平移（V^T_N）或旋转（V^R_N），测试模型对合成运动的预测能力。Moving KTH 则在原始 KTH 动作识别数据集上施加相机运动增强（Figure 10），模拟真实视频中的平移流。所有实验使用 H100 GPU，训练损失为第 11-20 时间步的 MSE 平均：$L = \frac{1}{10} \sum_{t=11}^{20} \|\hat{f}_{t+1} - f_{t+1}\|^2$。


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/b4eb803b-4f5a-40ab-a42b-3b42f14ee847/figures/fig_003.png)
*Figure: Increased flow equivariance increases training speed*



核心数值来自 Table 1：在 Flowing MNIST 平移流（V^T_2，两个速度生成元）上，FERNN 达到 Test MSE **1.5e-4 ± 2e-5**，相比 G-RNN 的 8.1e-3 ± 6e-4 提升 **54×**，相比部分等变的 FERNN-V^T_1（仅一个生成元，5.3e-4 ± 8e-5）提升 **3.5×**。旋转流（V^R_4）上，FERNN 的 6.1e-4 ± 3e-5 相比 G-RNN 的 4.0e-3 ± 5e-4 提升 **6.6×**。这些差距并非来自参数量优势——FERNN 与 G-RNN 参数量完全相同，优势纯粹源于架构归纳偏置。

长度泛化方面（Figure 4），FERNN 在训练时最长 20 步的序列上训练，测试时外推至 **70 步**，MSE 仍保持稳定；G-RNN 在超过训练长度后迅速发散。速度泛化方面（Figure 5/6），FERNN 对训练时未见过的速度（ν ∈ V 的插值或外推）仍保持低 MSE，而 G-RNN 需针对每个速度重新训练。



消融实验揭示关键组件贡献：Figure 3 显示验证损失收敛速度随流等变程度（$|V|$ 增大）显著提升；V-mixing（不同生成元间的信息混合）对弹性碰撞等多物体交互场景至关重要（Table 4）。Figure 11 证实运行时和内存随 $|V|$ 线性增长，当前 PyTorch 实现比 G-RNN 慢约 **30×**，主要因 naive for loop 而非优化扫描操作。

公平性检验：基线选择存在局限。未与 **ConvLSTM**、**PredRNN** 等现代非等变循环架构比较，也未与 **Video Swin Transformer**、**TimeSformer** 等 Transformer 视频模型对比。此外，FERNN 的 naive 实现使其训练时间比较处于劣势——若采用 CUDA kernel 优化，实际 wall-clock 收敛速度可能优于 G-RNN（FERNN 迭代次数更少）。数据集局限于合成运动（Flowing MNIST）和简单动作识别（KTH），复杂真实视频基准尚未验证。

## 方法谱系与知识库定位

FERNN 属于 **等变神经网络 → 群等变循环网络** 的方法谱系，直接父方法为 **G-RNN**（基于 Cohen & Welling 的 Group equivariant convolutional networks [12] 和 Weiler et al. 的 Steerable CNNs [13] 构建的循环扩展）。

**改动槽位**: 
- **架构**: 隐藏状态从 G 扩展到 V×G 乘积空间
- **目标函数**: 不变（仍为 MSE），但归纳偏置改变优化景观
- **训练策略**: 不变（监督学习），但收敛速度因等变性提升
- **数据策划**: 需预定义流生成元集 V
- **推理**: 增加 V-maxpool 步骤

**直接基线差异**:
- **G-RNN [12]**: 隐藏状态仅定义在空间群 G，无流信息；FERNN 扩展至 V×G 并引入流位移
- **Standard RNN**: 无任何等变性；FERNN 保持空间等变同时获得流等变
- **Recurrent Vision Transformer [19]**: 非等变 Transformer 循环架构；FERNN 以卷积归纳偏置实现参数高效的速度泛化

**后续方向**:
1. **高效实现**: 将 naive for loop 替换为 CUDA scan/kernel，消除 30× 训练速度劣势
2. **高维流扩展**: 当前线性缩放限制 $|V|$ 规模；需探索稀疏或连续流生成元表示
3. **复杂真实视频**: 从 Flowing MNIST/KTH 扩展至 Kinetics、Something-Something 等基准，验证对真实世界非刚性运动的建模能力

**标签**: modality=video/image | paradigm=supervised sequence prediction | scenario=motion forecasting, action recognition | mechanism=Lie group equivariance, recurrent neural network, flow symmetry | constraint=linear scaling with generator set size, limited to one-parameter subgroups

## 引用网络

### 直接 baseline（本文基于）

- Task-Optimized Convolutional Recurrent Networks Align with Tactile Processing in the Rodent Brain _(NeurIPS 2025, 直接 baseline, 未深度分析)_: Very recent (2025) and closely related CRNN work, likely compared against as bas

