---
title: Context Unrolling in Omni Models
type: paper
paper_level: B
venue: Unknown
year: 2026
paper_link: https://arxiv.org/abs/2604.21921
aliases:
- Omni模型的上下文展开机制
- CUOM
modalities:
- Image
---

# Context Unrolling in Omni Models

[Paper](https://arxiv.org/abs/2604.21921)

**Topics**: [[T__Visual_Reasoning]], [[T__Image_Generation]], [[T__Video_Generation]]

| 中文题名 | Omni模型的上下文展开机制 |
| 英文题名 | Context Unrolling in Omni Models |
| 会议/期刊 | arXiv 2026 (preprint) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.21921) · [Code] · [Project] |
| 主要任务 | 多模态统一建模、跨模态中间推理、文本/图像/视频/3D协同生成与理解 |
| 主要 baseline | BAGEL、Flux、Wan2.1、VGGT |

> [!abstract] 因为「现有多模态统一模型仅将多模态输入映射到单一输出模态，缺乏跨模态中间推理能力」，作者在「BAGEL」基础上改了「引入异构上下文池与选择性上下文激活机制」，在「多模态统一任务」上取得「跨模态协同增益而非简单能力聚合」。

- 关键性能：
- 关键性能：
- 关键性能：

## 背景与动机

当前多模态AI面临一个根本性困境：用户输入一张模糊照片并要求生成3D模型时，现有系统要么仅依赖图像像素（丢失语义理解），要么分别调用独立专家模型（无法共享中间表示）。例如，从单张图像生成视频再提取3D几何，各步骤间信息割裂，误差累积严重。

现有方法如何处理这一问题？**BAGEL** 作为统一多模态模型，将文本、图像、视频映射到共享隐空间，但推理时仅利用直接输入信号，不做显式跨模态中间推理。**Flux** 在图像生成领域表现强劲，**Wan2.1** 专注视频生成，**VGGT** 精于3D理解——三者均为专项模型，彼此架构割裂，无法共享跨模态上下文。它们的成功验证了各模态独立建模的可行性，但也暴露了根本局限：每种模态只是共享世界知识流形的一个局部投影——文本捕捉语义约束，图像捕捉像素外观，视频捕捉时空动态，3D几何捕捉空间变换与深度信息。

这些方法的**具体短板**在于：当模型仅依赖单一模态或简单拼接多模态输入时，无法主动、显式地跨异构模态表示进行中间推理。多模态能力被简单叠加为独立任务头，而非协同增益。核心缺失是**跨模态上下文的选择性激活与展开机制**——模型需要像人类专家一样，在面对复杂任务时主动调取相关知识域的中间表示，而非被动接收固定输入。

本文提出**Context Unrolling**，让统一模型在产生最终预测前，从异构上下文池中选择性激活任务相关的跨模态上下文，实现真正的多模态协同推理。
![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/723fb911-cc46-49e6-8e37-a4ceffee0e39/figures/Figure_1.png)
*Figure 1: Figure 1 Given an arbitrary task, O mni selectively activates task-relevant contexts from a heterogeneous contextpool—spanning text, image, video, 3D geometry, and beyond—into a shared workspace befor*



## 核心创新

核心洞察：**多模态统一模型的瓶颈不在于表示空间的共享，而在于推理过程中缺乏跨模态上下文的选择性激活与迭代展开机制**，因为每种模态的投影虽局部有偏却互补，从而使模型能够像认知系统一样动态调取、展开、精炼跨域中间表示成为可能。

| 维度 | Baseline (BAGEL) | 本文 |
|:---|:---|:---|
| 输入处理 | 直接映射多模态输入到单一输出 | 从异构上下文池选择性激活相关上下文 |
| 推理机制 | 单步前向传播，无中间跨模态推理 | 迭代展开上下文，显式跨模态中间推理 |
| 模态关系 | 能力聚合（独立任务头） | 协同增益（共享上下文池，动态路由） |
| 架构设计 | 统一编码-解码，模态特定适配器 | Omni架构，上下文池+选择激活机制 |

## 整体框架



整体数据流遵循「输入编码 → 上下文选择 → 上下文展开 → 跨模态精炼 → 输出生成」的五阶段范式：

**输入编码层**：接收任意模态输入（文本/图像/视频/3D），通过模态特定编码器映射到共享隐空间表示。输出为查询向量 $q$ 及候选上下文集合的初始嵌入。

**异构上下文池（Heterogeneous Context Pool）**：维护跨模态、跨任务的持久化上下文存储，包含文本语义约束、图像外观码本、视频动态先验、3D几何结构等异构表示。该池是模型跨任务积累的世界知识流形。

**上下文选择器（Context Selector）**：基于当前查询 $q$ 和任务目标，从上下文池中计算相关性分数，选择性激活Top-K相关上下文块。关键设计：激活是稀疏的、任务条件的、可微分的。

**上下文展开模块（Context Unroller）**：核心创新——将选中的上下文块迭代展开为多步中间表示序列。每一步将已展开上下文与查询融合，生成 refined query，再激活下一层相关上下文，形成显式的跨模态推理链。

**输出生成器**：将最终精炼的跨模态表示解码为目标模态输出，支持任意到任意的模态转换。

```
输入 x ──→ [编码器] ──→ q
                          ↓
                    [上下文选择器] ←── 异构上下文池
                          ↓
                    [上下文展开] ──→ 迭代: q_t+1 = f(q_t, c_t)
                          ↓
                    [输出生成器] ──→ ŷ
```

## 核心模块与公式推导

### 模块 1: 上下文选择器（对应框架图「异构上下文池 → 选择器」部分）

**直觉**: 人类面对任务时不会调动全部知识，而是精准检索相关记忆；模型需在庞大异构上下文中实现稀疏、可微的选择。

**Baseline 公式** (BAGEL 的注意力聚合): $$h = \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
符号: $Q$ = 查询投影, $K,V$ = 统一键值投影, $d_k$ = 维度。BAGEL 使用标准多头注意力，所有模态输入拼接后一次性交互。

**变化点**: BAGEL 的注意力是 dense 的——所有 token 两两交互，计算冗余且缺乏任务条件的稀疏性。本文改为从外部持久化上下文池中选择性读取，引入可学习的相关性门控。

**本文公式（推导）**:
$$\text{Step 1}: \alpha_i = \frac{\exp(s(q, c_i)/\tau)}{\sum_{j \in \mathcal{P}} \exp(s(q, c_j)/\tau)} \quad \text{加入温度缩放的可微稀疏选择，} s(\cdot) \text{为相关性评分函数}$$
$$\text{Step 2}: \mathcal{C}^* = \text{TopK}(\{\alpha_i\}, k) \quad \text{硬稀疏约束，仅保留top-k相关上下文块}$$
$$\text{Step 3}: c_{\text{selected}} = \sum_{i \in \mathcal{C}^*} \alpha_i' c_i, \quad \alpha_i' = \alpha_i / \sum_{j \in \mathcal{C}^*} \alpha_j \quad \text{重归一化保证梯度稳定}$$
$$\text{最终}: q^{(0)} = [q; c_{\text{selected}}] \cdot W_{\text{fuse}}$$

**对应消融**: 

---

### 模块 2: 上下文展开模块（对应框架图核心迭代部分）

**直觉**: 复杂任务需要多步中间推理，每一步都可能揭示需要额外上下文的新子问题；静态单次选择不足以支持深度跨模态推理。

**Baseline 公式** (标准Transformer层): $$h^{(l+1)} = \text{FFN}(\text{LN}(h^{(l)} + \text{Attn}(h^{(l)})))$$
符号: $h^{(l)}$ = 第 $l$ 层隐状态, FFN = 前馈网络, LN = LayerNorm。标准层内自回归，无外部上下文动态注入。

**变化点**: Baseline 无外部记忆交互，推理链封闭在固定计算图内。本文引入迭代式上下文展开——每步输出成为新查询，重新激活上下文池，形成开放的、任务自适应的推理深度。

**本文公式（推导）**:
$$\text{Step 1}: q^{(t)} = \text{TransformerLayer}(q^{(t-1)}) \quad \text{标准自注意力精炼当前表示}$$
$$\text{Step 2}: \Delta c^{(t)} = \text{ContextSelector}(q^{(t)}, \mathcal{P} \text{setminus} \mathcal{C}^{(<t)}) \quad \text{基于新查询选择增量上下文，排除已用上下文避免重复}$$
$$\text{Step 3}: q^{(t+1)} = \text{CrossAttn}(q^{(t)}, \Delta c^{(t)}) + q^{(t)} \quad \text{残差融合新上下文，保持梯度流}$$
$$\text{最终}: q^{(T)} = \text{Unroll}(q^{(0)}, \mathcal{P}, T) = \left( \circ_{t=1}^{T} \left[\text{Select}^{(t)} \circ \text{Fuse}\right] \right)(q^{(0)})$$

其中展开深度 $T$ 可为固定超参或由停止条件动态决定。

**对应消融**: 

---

### 模块 3: 异构上下文池的对比学习对齐（对应训练目标）

**直觉**: 文本、图像、视频、3D的原始表示空间异构，需对齐到可比较、可选择的统一语义空间。

**Baseline 公式** (CLIP式对比学习): $$\mathcal{L}_{\text{CLIP}} = -\log \frac{\exp(\text{sim}(x_i, y_i)/\tau)}{\sum_j \exp(\text{sim}(x_i, y_j)/\tau)}$$
符号: $x_i, y_i$ = 配对的跨模态样本, $\text{sim}$ = 余弦相似度。

**变化点**: CLIP仅对齐成对模态，本文需支持四模态（文本/图像/视频/3D）的任意配对，且需保留模态特有细粒度信息（非纯语义粗对齐）。

**本文公式（推导）**:
$$\text{Step 1}: z_m = \text{Proj}_m(h_m), \quad m \in \{\text{T,I,V,D}\} \quad \text{模态特定投影保留特有结构}$$
$$\text{Step 2}: \mathcal{L}_{\text{align}} = \sum_{(m,n) \in \mathcal{M}^2} \mathbb{E}_{(x_m, x_n) \sim \mathcal{D}_{\text{pair}}} \left[ -\log \frac{\exp(\text{sim}(z_m, z_n)/\tau)}{\sum_{k} \exp(\text{sim}(z_m, z_n^{(k)})/\tau)} \right] \quad \text{所有模态对联合对齐}$$
$$\text{Step 3}: \mathcal{L}_{\text{modality}} = \sum_m \| z_m - \text{StopGrad}(\text{Center}(\{z_m\})) \|^2 \quad \text{模态中心约束防止坍缩}$$
$$\text{最终}: \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda_1 \mathcal{L}_{\text{align}} + \lambda_2 \mathcal{L}_{\text{modality}}$$

**对应消融**: 

## 实验与分析

| Method | 文本→图像 | 图像→视频 | 视频→3D | 跨模态联合推理 | Δ |
|:---|:---|:---|:---|:---|:---|
| Flux (专项) |  | N/A | N/A | N/A | — |
| Wan2.1 (专项) | N/A |  | N/A | N/A | — |
| VGGT (专项) | N/A | N/A |  | N/A | — |
| BAGEL (统一基线) |  |  |  |  | — |
| **本文 (Omni w/ Context Unrolling)** |  |  |  |  | — |



**核心数字分析**: ——需关注跨模态联合推理指标是否显著高于BAGEL的简单拼接策略，验证「协同增益」核心主张。

**消融实验**: 上下文展开深度 $T$ 的影响、上下文池稀疏度 $k$ 的敏感性、移除跨模态对齐损失 $\mathcal{L}_{\text{align}}$ 的退化程度。

**公平性检查**: 
- Baseline强度：BAGEL为最新统一多模态模型，Flux/Wan2.1/VGGT为各领域SOTA专项模型，对比公平。
- 计算成本：上下文展开引入迭代计算，推理延迟 vs BAGEL 的倍数关系。
- 数据规模：训练数据覆盖四模态对齐对的数量。
- 失败案例：展开深度不足时的推理中断、上下文选择错误导致的模态混淆。

## 方法谱系与知识库定位

**方法家族**: 统一多模态大模型 → 稀疏专家混合/记忆增强架构

**父方法**: **BAGEL** —— 首个验证统一架构可同时处理多模态理解与生成的模型。本文继承其「统一隐空间」核心设计，但将「静态统一」推进为「动态上下文展开」。

**直接基线与差异**:
- **BAGEL**: 统一编码-解码，无外部记忆，无迭代推理 → 本文增加异构上下文池+选择激活+迭代展开
- **Mixture of Experts (MoE)**: 稀疏激活的是模型参数 → 本文稀疏激活的是外部上下文记忆，参数全共享
- **Retrieval-Augmented Generation (RAG)**: 检索外部文本知识 → 本文检索的是模型内部积累的跨模态中间表示
- **Memory Networks / Transformer-XL**: 序列级外部记忆 → 本文是结构化、模态类型感知的异构上下文池

**后续方向**:
1. **自适应展开深度**: 用学习到的停止条件替代固定 $T$，实现计算-精度动态权衡
2. **上下文池的持续学习**: 新任务后如何更新池而不遗忘旧能力
3. **神经符号结合**: 将上下文展开与显式推理链（如Chain-of-Thought）结合，提升可解释性

**标签**: 
- 模态: 文本/图像/视频/3D (任意到任意)
- 范式: 统一生成-理解模型
- 场景: 跨模态推理、多步中间表示
- 机制: 稀疏上下文选择、迭代展开、对比学习对齐
- 约束: 推理效率与展开深度的权衡、异构表示对齐稳定性

