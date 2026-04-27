---
title: 'Spiking Transformer: Introducing Accurate Addition-Only Spiking Self-Attention for Transformer'
type: paper
paper_level: C
venue: CVPR
year: 2025
paper_link: null
aliases:
- 纯加法脉冲自注意力Transformer
- Spiking Transfor
- Spiking Transformer
acceptance: poster
cited_by: 10
method: Spiking Transformer
---

# Spiking Transformer: Introducing Accurate Addition-Only Spiking Self-Attention for Transformer

**Topics**: [[T__Classification]], [[T__Compression]], [[T__Self-Supervised_Learning]] | **Method**: [[M__Spiking_Transformer]] | **Datasets**: [[D__ImageNet-1K]], [[D__CIFAR-10]], [[D__CIFAR-100]]

| 中文题名 | 纯加法脉冲自注意力Transformer |
| 英文题名 | Spiking Transformer: Introducing Accurate Addition-Only Spiking Self-Attention for Transformer |
| 会议/期刊 | CVPR 2025 (poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2503.00226) · [Code] · [Project] |
| 主要任务 | 图像分类（ImageNet-1K / CIFAR-10 / CIFAR-100） |
| 主要 baseline | Spike-driven Transformer, Spikformer, Spikingformer |

> [!abstract] 因为「标准脉冲Transformer中的自注意力仍包含昂贵的softmax、除法和矩阵乘法，阻碍能效优势」，作者在「Spike-driven Transformer」基础上改了「A²OS²A纯加法脉冲自注意力 + 混合Q-K-V编码 + 绝对值脉冲神经元」，在「ImageNet-1K」上取得「78.66% Top-1，以36.01M参数超越66.34M参数的Spike-driven Transformer-8-768达+1.59%」

- **ImageNet-1K**: 78.66% (Spiking Transformer-10-512) vs 77.07% (Spike-driven Transformer-8-768), +1.59%, 参数量仅54.3%
- **CIFAR-10**: 96.42% vs 95.51% (Spike-driven Transformer-2-512), +0.91%
- **CIFAR-100**: 79.9% vs 78.43% (Spike-driven Transformer-2-512), +1.47%

## 背景与动机

脉冲神经网络（Spiking Neural Network, SNN）以事件驱动的二进制脉冲进行通信，理论上具有极高的能效优势，但将Transformer架构迁移到SNN领域时面临根本性矛盾：标准自注意力中的softmax、除法和大规模矩阵乘法与SNN的加法-比较运算范式不兼容。

现有方法的处理方式各有局限。**Spikformer** 首次将Transformer引入SNN，但仍保留了原始自注意力的计算结构；**Spikingformer** 尝试优化脉冲化策略，但未根本改变注意力机制的计算复杂度；**Spike-driven Transformer** 作为当前主流基线，采用标准脉冲自注意力（VSSA），将softmax替换为脉冲神经元，但仍保留乘法运算和缩放因子，且Q、K、V均使用脉冲神经元编码导致负值信息丢失。

具体而言，这些方法的共同瓶颈在于：第一，注意力计算中的矩阵乘法 $QK^T$ 和 $QK^T V$ 仍是乘法密集型操作，无法发挥SNN加法运算的能效优势；第二，K投影经过脉冲神经元后变为非负值，破坏了注意力机制中对正负相似度的区分能力；第三，标准LIF神经元 $S[t] = \text{Hea}(U[t] - V_{\text{th}})$ 只能响应正输入，限制了信息表达的动态范围。本文通过重新设计自注意力的数学形式、编码策略和神经元动力学，首次实现了真正"加法-only"的脉冲Transformer注意力机制。

## 核心创新

核心洞察：自注意力的本质是对Q、K、V三重关系的聚合，而非必须依赖softmax归一化的矩阵乘法；通过元素级乘法替代矩阵乘法、以脉冲神经元的非线性替代softmax，可以在保留表达能力的同时将运算归约为纯加法，因为元素级乘法的累加等价于加法运算的重复执行。

| 维度 | Baseline (Spike-driven Transformer) | 本文 |
|:---|:---|:---|
| 注意力运算 | VSSA: $\mathcal{SN}(Q K^T V \cdot s)$，含矩阵乘法和缩放因子 | A²OS²A: $\mathcal{SN}(Q \cdot K^T \cdot V)$，元素级乘法，无softmax/除法/矩阵乘法 |
| Q-K-V编码 | Q, K, V 均为脉冲神经元 $\mathcal{SN}$ | Q, V 为 $\mathcal{SN}$，K 为 $\text{ReLU}$ 保留负值 |
| 神经元动力学 | $S[t] = \text{Hea}(U[t] - V_{\text{th}})$，仅正响应 | $S[t] = \text{Hea}(\|U[t]\| - V_{\text{th}})$，正负双极响应 |
| 位置编码 | 标准绝对位置编码或无显式RPE | Conv2d-based相对位置编码，脉冲域注入 |

四项改动协同作用：A²OS²A消除乘法运算，混合编码恢复负值信息，绝对值神经元扩展动态范围，卷积RPE在脉冲域保留空间结构。

## 整体框架


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/b2618faf-3886-4550-af5a-1c645e402c4e/figures/Figure_2.png)
*Figure 2 (pipeline): Overview of Spiking Transformer.*



Spiking Transformer的整体数据流如下：

1. **Patch Splitting Module (PSM)**：输入图像 $I \in \mathbb{R}^{T \times C \times H \times W}$（T=4时间步），分割为 $N$ 个patch并映射为 $D$ 维特征，输出 $u \in \mathbb{R}^{T \times N \times D}$

2. **初始脉冲化**：$u$ 经脉冲神经元 $\mathcal{SN}$ 得到初始脉冲特征 $s \in \mathbb{R}^{T \times N \times D}$

3. **Relative Positional Encoding (RPE)**：对 $s$ 应用 $\text{BN}(\text{Conv2d}(s))$，在脉冲域注入2D空间位置信息，输出 $\text{RPE} \in \mathbb{R}^{T \times N \times D}$

4. **初始膜电位整合**：$U_0 = u + \text{RPE}$，再经脉冲化得 $S_0 = \mathcal{SN}(U_0)$

5. **A²OS²A Block (×L)**：核心模块，每层执行：
   - A²OS²A注意力：$U'_l = \text{A}^2\text{OS}^2\text{A}(S_{l-1}) + U_{l-1}$（膜电位域残差）
   - 脉冲化：$S'_l = \mathcal{SN}(U'_l)$
   - MLP子层：$S_l = \mathcal{SN}(\text{MLP}(S'_l) + U'_l)$（第二次残差）

6. **分类头**：全局平均池化 $\text{GAP}(S_L)$ 后经分类头输出预测 $Y$

```
I ──→ PSM ──→ SN ──→ RPE ──┬──→ U_0 = u + RPE ──→ S_0
                              │
                              ↓
                    ┌─────────────────────┐
         S_{l-1} ──→│  A²OS²A attention  │──→ U'_l ──→ SN ──→ S'_l
         U_{l-1} ──→│    + residual      │              │
                    └─────────────────────┘              ↓
                                              MLP + residual ──→ S_l
                              │
                              ↓ (repeat L times)
                             GAP ──→ Classification Head ──→ Y
```

## 核心模块与公式推导

### 模块 1: 绝对值脉冲神经元（对应框架图 步骤2/4/5）

**直觉**：标准LIF神经元只能对正输入发放脉冲，导致负值信息永久丢失；取绝对值后神经元可对双极输入响应，扩展信息容量。

**Baseline 公式** (Spike-driven Transformer 标准LIF):
$$U[t] = H[t-1] + X[t], \quad S[t] = \text{Hea}(U[t] - V_{\text{th}}), \quad H[t] = V_{\text{reset}} S[t] + \beta U[t] (1 - S[t])$$
符号: $U[t]$=膜电位, $H[t]$=衰减后残留电位, $S[t] \in \{0,1\}$=输出脉冲, $V_{\text{th}}$=阈值, $V_{\text{reset}}$=重置电位, $\beta$=衰减系数, $\text{Hea}$=Heaviside阶跃函数

**变化点**：标准LIF的 $U[t] - V_{\text{th}}$ 使 $U[t] < 0$ 时永不发放，负输入信息被完全抑制；且重置项 $H[t]$ 的衰减仅针对正电位设计。

**本文公式（推导）**:
$$\text{Step 1}: S[t] = \text{Hea}(|U[t]| - V_{\text{th}}) \quad \text{绝对值阈值使正负电位同等参与发放}$$
$$\text{Step 2}: H[t] = V_{\text{reset}} S[t] + \beta U[t] (1 - |S[t]|) \quad \text{用}|S[t]|\text{替代}S[t]\text{使衰减适应绝对值激活}$$
$$\text{最终}: \text{boxed}{U[t] = H[t-1] + X[t], \; S[t] = \text{Hea}(|U[t]| - V_{\text{th}}), \; H[t] = V_{\text{reset}} S[t] + \beta U[t] (1 - |S[t]|)}$$

**对应消融**：

---

### 模块 2: 混合Q-K-V编码（对应框架图 A²OS²A Block输入）

**直觉**：注意力相似度计算 $Q \cdot K^T$ 需要K的符号信息判断正负相关性，而脉冲神经元的非负输出会破坏这一机制。

**Baseline 公式** (Spike-driven Transformer):
$$Q = \mathcal{SN}_Q(\text{BN}(XW_Q)), \quad K = \mathcal{SN}_K(\text{BN}(XW_K)), \quad V = \mathcal{SN}_V(\text{BN}(XW_V))$$
符号: $X$=输入特征, $W_Q, W_K, W_V$=投影矩阵, BN=批归一化, $\mathcal{SN}$=脉冲神经元

**变化点**：K经 $\mathcal{SN}$ 后输出 $\geq 0$，导致 $Q \cdot K^T$ 只能度量"存在性"相似度，无法区分正负相关性；且脉冲化引入量化误差。

**本文公式（推导）**:
$$\text{Step 1}: K = \text{ReLU}_K(\text{BN}(XW_K)) \quad \text{保留ReLU使K保持非负但连续，而非二值脉冲}$$
$$\text{Step 2}: Q = \mathcal{SN}^b_Q(\text{BN}(XW_Q)), \; V = \mathcal{SN}^t_V(\text{BN}(XW_V)) \quad \text{Q,V仍脉冲化保持能效，上标b/t区分编码类型}$$
$$\text{最终}: \text{boxed}{Q = \mathcal{SN}^b_Q(\text{BN}(XW_Q)), \; K = \text{ReLU}_K(\text{BN}(XW_K)), \; V = \mathcal{SN}^t_V(\text{BN}(XW_V))}$$

**对应消融**：Table 3 显示混合编码是整体精度提升的关键组件之一，CIFAR-10上完整模型96.42% vs 基线95.51%。

---

### 模块 3: A²OS²A 精确加法脉冲自注意力（对应框架图 A²OS²A Block核心）

**直觉**：标准自注意力的softmax、除法、矩阵乘法均可被脉冲神经元的非线性和元素级乘法的加法等价性所替代。

**Baseline 公式** (标准Transformer VSA):
$$\text{VSA}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
符号: $Q, K, V \in \mathbb{R}^{N \times d_k}$=查询/键/值矩阵, softmax=按行归一化指数函数, $QK^T$=矩阵乘法

**变化点**：softmax需指数运算（非SNN友好），除法需专用电路，矩阵乘法 $O(N^2 d_k)$ 复杂度且为乘法密集型；VSSA虽替换softmax为SN，但仍保留矩阵乘法和缩放因子。

**本文公式（推导）**:
$$\text{Step 1}: \text{VSSA}(Q, K, V) = \mathcal{SN}(Q ~ K^T ~ V \cdot s) \quad \text{先去除softmax，但保留乘法和缩放}s$$
$$\text{Step 2}: \text{A}^2\text{OS}^2\text{A}(Q, K, V) = \mathcal{SN}\left(Q \cdot K^T \cdot V\right) \quad \text{将矩阵乘法}QK^T\text{改为元素级乘法}\cdot\text{，去除}s$$
$$\text{关键等价}: (Q \cdot K^T)_{ij} \cdot V_{jk} \text{的元素级累加可由加法器阵列实现，因脉冲域的}0/1\text{特性使乘法退化为选择操作}$$
$$\text{最终}: \text{boxed}{\text{A}^2\text{OS}^2\text{A}(Q, K, V) = \mathcal{SN}\left(Q \cdot K^T \cdot V\right)}$$

**对应消融**：Table 3 中CIFAR-10/100多组配置验证，完整A²OS²A相比Spike-driven Transformer基线提升0.52%-0.91%（CIFAR-10）和0.96%-1.47%（CIFAR-100），且增益随模型规模扩大而增加。

## 实验与分析


![Table 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/b2618faf-3886-4550-af5a-1c645e402c4e/figures/Table_2.png)
*Table 2 (comparison): Comparison of the performance between Spiking Transformer and state-of-the-art SNNs.*



本文在三个标准基准上验证Spiking Transformer。ImageNet-1K是核心大规模实验：Spiking Transformer-10-512达到**78.66% Top-1精度**，以36.01M参数超越参数量达66.34M的Spike-driven Transformer-8-768（77.07%）**+1.59%**，同时参数减少45.7%；相比Spikformer-8-768（74.81%）提升**+3.85%**，相比Spikingformer-8-768（75.85%）提升**+2.81%**。较小配置的Spiking Transformer-8-512也以29.68M参数达到76.28%，超越Spike-driven Transformer-8-512（74.57%）**+1.71%**。这一结果表明A²OS²A的纯加法设计不仅没有损失表达能力，反而因更适配SNN的运算范式获得了更好的精度-效率权衡。


![Table 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/b2618faf-3886-4550-af5a-1c645e402c4e/figures/Table_4.png)
*Table 4 (comparison): Comparison of the performance between Spiking Transformer and existing approaches on ImageNet-1K.*



CIFAR系列实验主要用于消融验证。CIFAR-10上，Spiking Transformer-2-512达96.42%（+0.91% over Spike-driven Transformer-2-512），2-256配置达94.91%（+0.52%）；CIFAR-100上，2-512达79.9%（+1.47%），2-256达76.96%（+0.96%）。值得注意的是，增益幅度与模型容量正相关：2-512配置的提升（0.91%/1.47%）明显大于2-256（0.52%/0.96%），暗示A²OS²A的优势需要一定容量才能充分释放。


![Table 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/b2618faf-3886-4550-af5a-1c645e402c4e/figures/Table_3.png)
*Table 3 (ablation): Ablation study on the Spiking Transformer under various settings on CIFAR-10/100.*



Table 3的消融研究（CIFAR-10/100）系统验证了各组件贡献。虽然具体"去掉X后精度下降Y%"的逐项数据未在分析片段中完整呈现，但整体趋势显示：混合Q-K-V编码和绝对值神经元对维持注意力表达能力至关重要，而A²OS²A的结构替换是精度提升的主要来源。作者披露的局限性包括：未与Spikformer V2（文献[5]）直接对比；缺乏实际硬件能耗测量，"加法-only"的能效优势停留在理论层面；未与近期非脉冲高效Transformer（如线性注意力变体）比较。时间步长固定为T=4，属于低延迟SNN的典型设置。

## 方法谱系与知识库定位

**方法家族**：脉冲神经网络 × Vision Transformer（SNN-Transformer 杂交架构）

**父方法**：Spike-driven Transformer（直接继承其整体架构框架，替换核心注意力模块）

**改动槽位**：
- **architecture**: 标准脉冲自注意力 → A²OS²A（元素级乘法+SN，无softmax/除法/矩阵乘法）；Q-K-V编码改为混合SN-ReLU；新增Conv2d-based RPE
- **inference_strategy**: 标准LIF → 绝对值LIF（$|U[t]|$阈值，双极响应）
- **training_recipe**: 继承Membrane Potential BN、RMP-Loss、Optimized Potential Initialization等SNN训练技术

**直接基线对比**：
- **Spike-driven Transformer**: 本文直接父方法，本文替换其VSSA为A²OS²A，修改编码和神经元
- **Spikformer** (74.81%): 更早SNN-Transformer，保留更多原始注意力结构，本文+3.85%
- **Spikingformer** (75.85%): 同期竞争方法，本文+2.81%

**后续方向**：(1) 硬件级验证A²OS²A的实际能效收益；(2) 将加法-only思想扩展至其他Transformer变体（如线性注意力、状态空间模型）；(3) 探索T<4的超低时间步长下的稳定性；(4) 与ANN-SNN转换方法在相同延迟约束下的公平对比。

**标签**：modality=图像 / paradigm=脉冲神经网络+Transformer / scenario=边缘高效推理 / mechanism=元素级注意力+混合编码+绝对值神经元 / constraint=低时间步长(T=4)、低参数量

