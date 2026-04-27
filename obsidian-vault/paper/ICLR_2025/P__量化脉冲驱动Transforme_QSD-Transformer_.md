---
title: Quantized Spike-driven Transformer
type: paper
paper_level: C
venue: ICLR
year: 2025
paper_link: null
aliases:
- 量化脉冲驱动Transformer高效视觉识别
- QSD-Transformer
- QSD-Transformer (Quantized Spike-driven Transformer)
acceptance: Poster
cited_by: 18
code_url: https://github.com/bollossom/QSD-Transformer
method: QSD-Transformer (Quantized Spike-driven Transformer)
---

# Quantized Spike-driven Transformer

[Code](https://github.com/bollossom/QSD-Transformer)

**Topics**: [[T__Compression]], [[T__Classification]] (其他: Quantization) | **Method**: [[M__QSD-Transformer]] | **Datasets**: [[D__ImageNet-1K]]

| 中文题名 | 量化脉冲驱动Transformer高效视觉识别 |
| 英文题名 | Quantized Spike-driven Transformer |
| 会议/期刊 | ICLR 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2501.13492) · [Code](https://github.com/bollossom/QSD-Transformer) · [DOI](https://doi.org/10.52202/075280-2798) |
| 主要任务 | ImageNet-1K 图像分类（脉冲神经网络 SNN） |
| 主要 baseline | Spikformer, SD-Transformer v2, SpikingResformer |

> [!abstract] 因为「直接量化 Spikformer 导致准确率暴跌 6.14%」，作者在「Spikformer」基础上改了「将标准 LIF 神经元替换为 IE-LIF 多比特脉冲神经元，并引入 FGD 细粒度蒸馏与低比特权重量化」，在「ImageNet-1K」上取得「80.3% Top-1 准确率，仅用 6.8M 参数和 8.7mJ 功耗，优于 SD-Transformer v2 的 79.7% / 55.4M / 52.4mJ」

- **效率突破**：相比 SD-Transformer v2，参数减少 87.58%（6.8M vs. 55.4M），功耗降低 83.40%（8.7mJ vs. 52.4mJ），准确率仍提升 +0.6%（80.3% vs. 79.7%）
- **量化恢复**：直接量化 Spikformer 仅 69.36%，加入 IE-LIF 后提升至 75.36%，再加 FGD 达到 75.5%，累计恢复 +6.14%
- **极端量化**：2-bit 权重量化下仍保持 73.1% 准确率

## 背景与动机

脉冲神经网络（Spiking Neural Network, SNN）以其事件驱动的稀疏计算特性，被视为实现超低功耗神经形态计算的核心路径。然而，当 SNN 与 Transformer 架构结合时，一个根本矛盾浮现：标准脉冲神经元仅输出 0/1 的二值脉冲，信息容量极其有限，一旦引入权重量化（如 2-bit、4-bit）以压缩模型规模，注意力机制中的 query/key/value 分布会高度集中于低数值区域，导致信息严重损失。例如，直接对 Spikformer 进行 4-bit 权重量化，ImageNet 准确率从浮点模型的 75.5% 暴跌至 69.36%，几乎丧失实用价值。

现有方法如何应对这一问题？Spikformer 首次将标准 Transformer 移植到 SNN 领域，采用 LIF（Leaky Integrate-and-Fire）神经元和 Spike-Driven Self-Attention（SDSA），但其设计假设浮点权重，未考虑量化场景。SD-Transformer v2 通过改进架构提升了 SNN 的效率边界，但在极端压缩下仍需要 55.4M 参数和 52.4mJ 功耗，且未解决量化带来的信息瓶颈。SpikingResformer 引入残差连接增强训练稳定性，然而其 60.4M 参数规模和 9.7mJ 功耗在边缘部署中仍显沉重。

这些方法的共同短板在于：**将脉冲活动的二值性视为不可更改的物理约束**，从而在量化后陷入"低信息量 → 量化误差放大 → 准确率崩溃"的死锁。具体而言，标准 LIF 的 1-bit 活动输出使 Q-SDSA（量化脉冲驱动自注意力）中的信息分布高度集中于低值（如 query 均值仅 0.31，key 均值仅 0.22），量化后有效信息几乎被抹平。

本文的核心动机正是打破这一死锁：通过赋予脉冲神经元多比特表达能力，从根本上提升量化场景下的信息容量，并配合针对性的蒸馏策略实现高效恢复。

## 核心创新

核心洞察：**脉冲活动的比特数是可配置的信息瓶颈**，因为标准 LIF 的 1-bit 输出人为限制了神经元的表达维度，从而使量化误差在注意力机制中被逐级放大；若将活动比特从 1 提升至 b（如 b=4），并配合信息感知的训练策略，则可在保持脉冲稀疏优势的同时，恢复甚至超越浮点模型的精度。

| 维度 | Baseline (Spikformer) | 本文 (QSD-Transformer) |
|:---|:---|:---|
| 脉冲表达 | 1-bit 二值脉冲（0/1） | b-bit 多值脉冲（IE-LIF，b∈{1,2,4}） |
| 信息容量 | 固定极低，量化后信息集中于低值区 | 可配置提升，Q-SDSA 信息分布扩散至高值区 |
| 训练策略 | 标准交叉熵 + 量化感知训练 | FGD 细粒度蒸馏，针对注意力信息分布优化 |
| 权重精度 | 32-bit 浮点 | 2-bit / 4-bit 低比特量化 |
| 时间步长 | 固定 T=4 | T 与 b 联合配置（如 T=1,b=4 或 T=4,b=1） |

这一洞察将 SNN 的"二值脉冲"假设从物理定律降维为设计选择，打开了量化-效率-精度的联合优化空间。

## 整体框架


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/99d02b92-db86-4393-afd1-f2ed68c16b2a/figures/fig_001.png)
*Figure: Overview of the QSD-Transformer. (a) Proposed quantized spike-driven self-attention (Q-*



QSD-Transformer 的整体数据流遵循"图像 → 分块嵌入 → 堆叠的量化脉冲驱动 Transformer 块 → 分类头"的标准范式，但每个内部模块均经过重新设计：

1. **图像分块与线性嵌入（Patch Embedding）**：输入图像划分为 16×16 patches，经线性投影得到初始特征，送入后续 Transformer 块。

2. **IE-LIF 脉冲神经元层**：替代标准 LIF 的核心单元。接收膜电位输入，输出 b-bit 多值脉冲活动（非 0/1，而是 {0, 1, ..., 2^b−1} 的离散值）。该层直接决定信息容量的上限，是量化精度的第一道保障。

3. **Q-SDSA 模块（Quantized Spike-Driven Self-Attention）**：基于 IE-LIF 输出的多比特脉冲，构建量化版本的 query/key/value。注意力计算保持脉冲驱动的稀疏累加特性，但信息分布因多比特活动而显著扩散（query 均值从 0.31 提升至 0.51，key 均值从 0.22 提升至 0.69）。

4. **权重量化层**：对 Q-SDSA 及 MLP 中的权重实施低比特量化（2-bit 或 4-bit），采用对称均匀量化器，在推理时以整数运算替代浮点乘加。

5. **FGD 训练模块（Fine-Grained Distillation）**：在训练阶段并行运行。以浮点精度的 Spikformer/SD-Transformer 为教师，QSD-Transformer 为蒸馏学生，不仅对齐最终预测，更针对 Q-SDSA 内部的注意力信息分布进行逐层细粒度匹配。

6. **分类头（MLP Head）**：经全局平均池化后输出 1000 类预测。

```
Input Image
    ↓
Patch Embedding
    ↓
[IE-LIF → Q-SDSA → MLP] × L  (堆叠 L 个 Transformer 块)
    │        ↑
    └─ FGD Teacher (训练时辅助，推理时丢弃)
    ↓
Global Avg Pool → Classifier
    ↓
Output (1000-dim logits)
```

关键设计：IE-LIF 与 Q-SDSA 的深度耦合——IE-LIF 的多比特输出是 Q-SDSA 信息分布改善的物理基础，而 FGD 则从优化目标层面确保这一物理改善转化为任务精度。

## 核心模块与公式推导

### 模块 1: IE-LIF（Information-Enhanced LIF）神经元（对应框架图：每个 Transformer 块的起始位置）

**直觉**：标准 LIF 的 1-bit 脉冲在量化场景下信息容量不足，IE-LIF 通过引入可配置的活动比特 b，将脉冲输出从二值扩展为多值离散信号，等效于在脉冲域实现了低精度但非二值的激活量化。

**Baseline 公式** (Spikformer 标准 LIF):
$$s_t = \Theta(u_t - u_{th}), \quad u_t = \tau \cdot u_{t-1} + I_t$$
其中 $s_t \in \{0, 1\}$ 为 t 时刻脉冲输出，$\Theta(\cdot)$ 为 Heaviside 阶跃函数，$u_t$ 为膜电位，$\tau$ 为泄漏系数，$I_t$ 为输入电流，$u_{th}$ 为发放阈值。

**变化点**：标准 LIF 的 $s_t$ 仅 1-bit，量化后信息高度集中。IE-LIF 将输出扩展为 b-bit 表示：

**本文公式（推导）**:
$$\text{Step 1}: \tilde{s}_t = \text{clip}\left(\left\lfloor \frac{u_t}{u_{th}} \cdot (2^b - 1) \right\rfloor, 0, 2^b - 1\right) \quad \text{将膜电位映射到 b-bit 离散值}$$
$$\text{Step 2}: s_t = \tilde{s}_t \cdot \mathbb{1}_{[u_t \geq 0]} \quad \text{保留脉冲稀疏性（仅正电位激活），零电位仍输出 0}$$
$$\text{最终}: u_t = \tau \cdot u_{t-1} \cdot (1 - \mathbb{1}_{[s_{t-1} > 0]}) + I_t \quad \text{多值发放后的膜电位重置（部分重置机制）}$$

符号说明：$b$ = 活动比特数（本文取 1, 2, 4），$\tilde{s}_t$ = 中间量化值，$\lfloor \cdot \rfloor$ = 向下取整，$\text{clip}$ 限制取值范围。

**对应消融**：将 b 从 4 降至 1（T=1 固定），准确率从 77.5% 降至 67.6%，Δ = −9.9%（Table 5）。

---

### 模块 2: FGD（Fine-Grained Distillation）训练目标（对应框架图：训练阶段的旁路教师-学生结构）

**直觉**：仅恢复脉冲活动的比特数不足以完全弥补量化损失，需从优化目标层面引导 Q-SDSA 的信息分布向浮点教师对齐，而非仅匹配最终预测。

**Baseline 公式** (标准知识蒸馏):
$$L_{KD} = \alpha \cdot L_{CE}(y_{student}, y_{gt}) + (1-\alpha) \cdot \tau^2 \cdot KL\left(\frac{y_{teacher}}{\tau}, \frac{y_{student}}{\tau}\right)$$
即最终 logits 层面的软目标蒸馏。

**变化点**：标准 KD 仅对齐输出分布，忽略了 Q-SDSA 内部 query/key/value 的信息结构。FGD 引入注意力分布与中间特征的细粒度匹配：

**本文公式（推导）**:
$$\text{Step 1}: L_{attn} = \sum_{l=1}^{L} \left\| A_{teacher}^{(l)} - A_{student}^{(l)} \right\|_F^2 \quad \text{逐层注意力图对齐，} A^{(l)} = \text{softmax}(Q^{(l)}K^{(l)\text{top}}/\sqrt{d})$$
$$\text{Step 2}: L_{info} = \sum_{l=1}^{L} \sum_{X \in \{Q,K,V\}} \left\| \mu(X_{teacher}^{(l)}) - \mu(X_{student}^{(l)}) \right\|_2^2 \quad \text{信息分布均值对齐（对应 Figure 3(b) 的分布改善）}$$
$$\text{最终}: L_{FGD} = L_{CE}(y_{student}, y_{gt}) + \lambda_{attn} \cdot L_{attn} + \lambda_{info} \cdot L_{info}$$

符号说明：$A^{(l)}$ = 第 l 层的注意力矩阵，$\mu(\cdot)$ = 特征均值（反映信息分布中心），$\lambda_{attn}, \lambda_{info}$ 为平衡系数。

**对应消融**：去掉 FGD（仅 IE-LIF），准确率从 75.5% 降至 75.36%，Δ = −0.14%（Table 5）；但结合 IE-LIF 的 6.0% 提升，FGD 是实现最终 75.5% 的关键微调。

---

### 模块 3: 权重量化与 Q-SDSA 联合推理（对应框架图：Q-SDSA 模块内部）

**直觉**：权重与活动均需低比特化，但二者耦合会产生复合误差；Q-SDSA 利用脉冲驱动的稀疏累加特性，将乘法转化为掩码选择，使权重量化后的整数运算仍保持高效。

**Baseline 公式** (标准线性层量化):
$$\hat{W} = s_W \cdot \text{round}\left(\frac{W}{s_W}\right), \quad s_W = \frac{\max|W|}{2^{k-1}-1}$$
其中 $k$ = 权重比特数，$s_W$ 为缩放因子，对称均匀量化。

**变化点**：标准量化假设连续激活，而 Q-SDSA 的输入为 IE-LIF 的离散脉冲 $s \in \{0, 1, ..., 2^b-1\}$，可将乘加运算简化为查表/累加：

**本文公式（推导）**:
$$\text{Step 1}: \hat{W}_{k} = s_W \cdot q_W, \quad q_W \in \{-2^{k-1}, ..., 2^{k-1}-1\} \quad \text{k-bit 对称权重量化（k=2 或 4）}$$
$$\text{Step 2}: O = \sum_{i} \hat{W}_{k}^{(i)} \cdot s^{(i)} = s_W \cdot \sum_{i} q_W^{(i)} \cdot s^{(i)} \quad \text{脉冲驱动：} s^{(i)} \text{ 稀疏且离散，乘法转为移位累加}$$
$$\text{最终}: O_{quant} = s_W \cdot \text{accumulate}\left(q_W \cdot s\right) \quad \text{推理时仅整数运算，缩放因子 } s_W \text{ 最后统一乘回}$$

**对应消融**：2-bit 权重（k=2, b=1, T=4）下准确率 73.1%，4-bit 权重（k=4, b=1, T=4）下准确率提升至 75.5%（Table 5）；作为对比，直接量化 Spikformer 的 32-bit 活动 + 4-bit 权重仅 69.36%。

## 实验与分析



本文在 ImageNet-1K 上进行了系统评估。
![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/99d02b92-db86-4393-afd1-f2ed68c16b2a/figures/fig_002.png)
*Figure: (a) Accuracy of quantizing different modules in the SD-Transformer v2 and its same ANN*

 展示了 QSD-Transformer 在准确率-功耗-参数三维空间中的帕累托前沿位置。核心结果表明：QSD-Transformer 以 6.8M 参数和 8.7mJ 功耗达到 80.3% Top-1 准确率，相比 SD-Transformer v2（55.4M 参数 / 52.4mJ / 79.7%）在精度微升 +0.6% 的同时，参数压缩 87.58%、功耗降低 83.40%。相较于 SpikingResformer（60.4M / 9.7mJ / 78.7%），QSD-Transformer 以显著更少的参数（−88.7%）和略低的功耗（−10.3%）实现了 +1.6% 的精度优势。这一结果直接验证了"多比特脉冲 + 细粒度蒸馏"范式在极端压缩场景下的有效性。



消融实验进一步量化了各组件的贡献。IE-LIF 是决定性因素：将标准 LIF 替换为 IE-LIF（b=4, T=1），准确率从直接量化的 69.36% 跃升至 75.36%，恢复 +6.0%。在此基础上叠加 FGD，精度进一步提升至 75.5%（+0.14%）。活动比特 b 的敏感性极高：固定 T=1，b 从 4 降至 1 导致准确率从 77.5% 暴跌至 67.6%（−9.9%）。时间步 T 同样重要：固定 b=1，T 从 4 降至 1 导致从 70.0% 降至 67.6%（−2.4%）。值得注意的是，T 与 b 存在替代关系——4-1-4 配置（4-bit 权重, 1-bit 活动, T=4）达到 75.5%，而 4-4-1 配置（4-bit 权重, 4-bit 活动, T=1）达到 77.5%，说明**提升单步信息密度（b↑）比增加时间步（T↑）更高效**，这对降低推理延迟具有重要实践意义。极端压缩场景下，2-bit 权重仍保持 73.1%，证明了方法的鲁棒性。

公平性审视：作者承认自实现基线（标记 ⋆）可能存在实现差异；功耗计算基于 Yao et al. (2023b) 的模型，未经过实际神经形态芯片验证。缺失的比较包括：通用量化方法（如 BRECQ、LSQ）直接应用于 SNN 的效果，以及与非脉冲量化 Transformer（如 4-bit DeiT-Tiny）的效率对比——后者对判断"脉冲机制本身是否必要"至关重要。

## 方法谱系与知识库定位

QSD-Transformer 属于 **Spikformer → 量化脉冲 Transformer** 的演化支系，直接父方法为 **Spikformer**（Spiking Neural Network Meets Transformer, 2022）。

**谱系变更槽位**：
- **架构（architecture）**：Spikformer 的标准 LIF → IE-LIF（多比特活动输出）
- **训练策略（training_recipe）**：标准量化感知训练 → 增加 FGD 细粒度蒸馏
- **推理策略（inference_strategy）**：32-bit 权重 + 固定 T=4 → 2-bit/4-bit 权重 + 可配置 (b, T) 联合

**直接基线对比**：
- **Spikformer**：直接量化崩溃（−6.14%），QSD-Transformer 通过 IE-LIF+FGD 完全恢复并超越
- **SD-Transformer v2**：参数/功耗规模庞大（55.4M/52.4mJ），QSD-Transformer 以 1/8 资源实现更高精度
- **SpikingResformer**：残差连接增强稳定性，但未解决量化信息瓶颈，QSD-Transformer 以更精简结构胜出

**后续方向**：(1) 将 IE-LIF 推广至其他 SNN 架构（如 Spiking CNN、Spiking MLP）；(2) 联合搜索 (b, T, k) 的最优配置，替代手工设计；(3) 在真实神经形态硬件（如 Intel Loihi、IBM TrueNorth）上验证功耗模型，并与同等精度的 ANN 量化模型进行端到端能效对比。

**知识库标签**：
- **模态（modality）**：视觉（图像分类）
- **范式（paradigm）**：脉冲神经网络（SNN）+ Transformer
- **场景（scenario）**：边缘部署、极低功耗推理、模型压缩
- **机制（mechanism）**：多比特脉冲表达、细粒度知识蒸馏、权重量化
- **约束（constraint）**：低比特权重（2-bit/4-bit）、离散脉冲活动、时间步延迟

