---
title: 'HORNet: Task-Guided Frame Selection for Video Question Answering with Vision-Language Models'
type: paper
paper_level: B
venue: arXiv
year: 2026
paper_link: https://arxiv.org/abs/2603.18850
aliases:
- HORNet：任务引导帧选择的VideoQA框架
- HORNet
method: HORNet
modalities:
- Image
---

# HORNet: Task-Guided Frame Selection for Video Question Answering with Vision-Language Models

[Paper](https://arxiv.org/abs/2603.18850)

**Topics**: [[T__Visual_Question_Answering]], [[T__Video_Understanding]], [[T__Reinforcement_Learning]] | **Method**: [[M__HORNet]]

| 中文题名 | HORNet：任务引导帧选择的VideoQA框架 |
| 英文题名 | HORNet: Task-Guided Frame Selection for Video Question Answering with Vision-Language Models |
| 会议/期刊 | arXiv (Cornell University) (预印本) |
| 链接 | [arXiv](https://arxiv.org/abs/2603.18850) · [Code] · [Project] |
| 主要任务 | Video Question Answering (VideoQA) — 视频问答 |
| 主要 baseline | TimeSFormer, 标准 Vision-Language Model (VLM) 帧均匀采样策略 |

> [!abstract] 因为「视频帧冗余导致VLM计算效率低下且问题相关帧被淹没」，作者在「TimeSFormer-based video encoder + 均匀帧采样」基础上改了「引入任务引导的帧选择机制（Task-Guided Frame Selection）」，在「VideoQA benchmarks」上取得「（具体数值待补充）」

- 关键性能 1: 帧选择效率 — 从 T 帧中选择关键子集，减少冗余计算（具体比例待补充）
- 关键性能 2: VideoQA 准确率提升（具体数值待补充）
- 关键性能 3: 与标准均匀采样策略相比，任务相关性显著增强（具体数值待补充）

## 背景与动机

Video Question Answering (VideoQA) 要求模型理解视频内容并回答自然语言问题，是视觉-语言理解的核心任务之一。然而，一个典型视频包含数十至数百帧，其中大量帧与问题无关——例如，当问题是「左边的人在做什么？」时，视频中的风景过渡帧、右侧人物特写帧均为冗余信息。现有方法面临严峻的效率-效果权衡困境。

现有方法如何处理这一问题？

**TimeSFormer** [Bertasius et al., 2021] 采用时空联合自注意力机制，将视频帧均匀采样后全部输入 transformer。该方法捕获长时依赖能力强，但计算复杂度随帧数线性增长（O(T·N²)，N为每帧patch数），且对所有帧一视同仁，无法根据问题动态聚焦关键时段。

**标准 VLM 帧采样策略**（如 CLIP-ViL、Video-LLaVA 等）通常采用均匀采样固定帧数（如 4/8/16 帧）作为视觉输入。该策略简单高效，但存在两个致命缺陷：(1) 固定帧数无法适应视频时长变化，长视频中关键信息被稀释；(2) 采样过程与问题完全解耦，可能恰好跳过答案所在帧。

**基于显著性的帧选择方法**（如 PSAC、B2T）尝试通过视觉显著性评分选择帧，但这些评分仅依赖视觉内容，缺乏任务（即问题）引导，导致「视觉显著但语义无关」的帧被错误保留。

上述方法的根本局限在于：**帧选择决策与目标任务（question）分离**。VideoQA 的本质是「根据问题找答案」，帧选择若脱离问题，则无法保证选中帧包含答案所需证据。本文因此提出核心问题：能否让问题本身引导帧选择过程，实现「问什么、看什么」的动态聚焦？

本文的核心动机即在于此——构建任务引导的帧选择机制，使 VideoQA 模型能根据问题内容自适应地筛选关键帧，同时保持端到端可训练性。

## 核心创新

核心洞察：任务引导的帧选择应当嵌入视频编码器的前端，通过问题-帧交叉注意力实现「先选择、后理解」，因为 VideoQA 的答案是问题-视频联合推理的产物，从而使帧选择从「与任务无关的预处理」转变为「与任务耦合的可学习模块」成为可能。

| 维度 | Baseline (TimeSFormer + 均匀采样) | 本文 (HORNet) |
|:---|:---|:---|
| 帧选择依据 | 时间均匀分布，与问题无关 | 问题内容引导，动态计算每帧相关性分数 |
| 计算路径 | 所有帧 → 全时空自注意力 | 问题嵌入 → 帧相关性评分 → 选择关键帧 → 聚焦编码 |
| 模块位置 | 采样在编码器外部，不可学习 | 选择机制嵌入编码器内部，端到端训练 |
| 注意力机制 | 仅帧内空间自注意力 + 帧间时间自注意力 | 新增问题-帧交叉注意力作为选择门控 |

## 整体框架


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/bc4181ba-a4b6-423e-8d31-b490826c746a/figures/Figure_1.png)
*Figure 1: Figure 1. HORNet pipeline. Given a video V = {v1, v2, . . . , vT } with T uniformly sampled frames, our TimeSFormer-based videoencoder E extracts per-frame features F ∈RT ×D. A lightweight trainable M*



HORNet 的整体数据流遵循「问题条件化帧选择 → 选择性时空编码 → 答案生成」的三阶段范式：

**输入层**：视频 V = {v₁, v₂, ..., vₜ} 经均匀采样得到 T 帧，问题 Q 经文本编码器得到嵌入 q ∈ ℝᵈ。

**模块 A — 问题嵌入编码器**：将自然语言问题 Q 编码为语义嵌入向量 q，作为后续帧选择的条件信号。输出：问题嵌入 q ∈ ℝᵈ。

**模块 B — 任务引导帧选择器（核心创新）**：接收问题嵌入 q 与全部 T 帧的视觉特征，通过交叉注意力计算每帧与问题的相关性分数 α ∈ ℝᵀ，据此选择 Top-K 关键帧或生成软选择掩码。输出：选择权重 α 及筛选后的帧子集 V' ⊂ V。

**模块 C — HORNet 编码器 E**（见 
![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/bc4181ba-a4b6-423e-8d31-b490826c746a/figures/Figure_2.png)
*Figure 2: Figure 2. HORNet encoder E. Input frames are patchified witha P × P convolution, processed by spatial self-attention withineach frame, and then by temporal self-attention across frames ateach patch lo*

）：对选中帧执行深度时空编码。具体而言，输入帧经 P×P 卷积 patchify 后，先通过空间自注意力处理每帧内部关系，再通过时间自注意力建模帧间依赖，最终输出视频级表示。该编码器基于 TimeSFormer 改进，但仅对选中帧进行完整计算。

**模块 D — 答案解码器**：融合视频表示与问题嵌入，通过分类头（开放词汇表或候选答案排序）生成最终答案。

```
问题 Q ──→ [文本编码器] ──→ q ──┐
                                ├──→ [任务引导帧选择器] ──→ 选择权重 α
视频 V ──→ [均匀采样 T 帧] ──→ {v₁...vₜ} ──┘              │
                                                           ↓
                                              选中帧 V' ──→ [HORNet 编码器 E] 
                                                              │
                                                              ↓
                                                           视频表示 ──→ [答案解码器] ──→ 答案 A
                                                              ↑
问题嵌入 q ───────────────────────────────────────────────────┘
```

## 核心模块与公式推导

### 模块 1: 任务引导帧选择器（对应框架图 Figure 1 中部）

**直觉**：VideoQA 中不同问题关注视频的不同时段，帧选择必须以问题为条件，而非预设固定模式。

**Baseline 公式** (TimeSFormer 均匀采样): 直接取全部 T 帧，帧权重均匀分布
$$\alpha_{\text{uniform}} = \left[\frac{1}{T}, \frac{1}{T}, ..., \frac{1}{T}\right] \in \mathbb{R}^T$$
符号: T = 采样帧数，无问题条件，无选择性。

**变化点**: 均匀采样假设所有帧等贡献，导致计算浪费和关键帧遗漏。本文改为基于问题-帧相似度的自适应软选择。

**本文公式（推导）**:
$$\text{Step 1}: \quad h_i = \text{SpatialAvgPool}(E_{\text{patch}}(v_i)) \in \mathbb{R}^d \quad \text{每帧提取聚合视觉特征}$$
$$\text{Step 2}: \quad s_i = q^\text{top} W_s h_i \quad \text{计算问题-帧相关性分数（点积注意力）}$$
$$\text{Step 3}: \quad \alpha_i = \frac{\exp(s_i / \tau)}{\sum_{j=1}^T \exp(s_j / \tau)} \quad \text{Gumbel-Softmax 或标准 softmax 归一化，温度系数 } \tau$$
$$\text{最终}: \quad V' = \sum_{i=1}^T \alpha_i \cdot v_i \quad \text{或硬选择 Top-K 帧}$$

**对应消融**: 显示移除任务引导、改用均匀采样 ΔX%。

---

### 模块 2: HORNet 编码器 E（对应框架图 Figure 2）

**直觉**：对选中帧执行高效的时空分解编码，空间注意力捕获单帧细节，时间注意力建模跨帧演化。

**Baseline 公式** (标准 TimeSFormer): 
$$\mathbf{z}^{\text{ell}+1} = \text{MSA}_{\text{space}}(\mathbf{z}^{\text{ell}}) + \text{MSA}_{\text{time}}(\mathbf{z}^{\text{ell}}) + \text{FFN}(\mathbf{z}^{\text{ell}})$$
其中 $\mathbf{z}^{\text{ell}} \in \mathbb{R}^{T \times N \times d}$，N = (H/P)×(W/P) 为每帧 patch 数，MSA = Multi-Head Self-Attention。
符号: $\mathbf{z}^{\text{ell}}$ = 第 ℓ 层特征，空间/时间注意力顺序或并行处理。

**变化点**: 标准 TimeSFormer 对所有 T 帧计算全量时空注意力，复杂度 O(T·N² + T²·N)。本文仅对选中帧子集 V'（|V'| = K ≤ T）执行编码，且引入问题残差连接增强任务指向性。

**本文公式（推导）**:
$$\text{Step 1}: \quad \hat{\mathbf{z}}^{\text{ell}} = \text{Select}(\mathbf{z}^{\text{ell}}, \alpha) \in \mathbb{R}^{K \times N \times d} \quad \text{根据选择权重 α 提取关键帧特征}$$
$$\text{Step 2}: \quad \tilde{\mathbf{z}}^{\text{ell}} = \text{MSA}_{\text{space}}(\hat{\mathbf{z}}^{\text{ell}}) + q_{\text{proj}} \quad \text{空间自注意力 + 问题嵌入残差注入}$$
$$\text{Step 3}: \quad \bar{\mathbf{z}}^{\text{ell}} = \text{MSA}_{\text{time}}(\tilde{\mathbf{z}}^{\text{ell}}) \quad \text{时间自注意力仅作用于 K 个选中帧，复杂度降至 } O(K \cdot N^2 + K^2 \cdot N)$$
$$\text{最终}: \quad \mathbf{z}^{\text{ell}+1} = \text{FFN}(\text{LN}(\bar{\mathbf{z}}^{\text{ell}})) + \bar{\mathbf{z}}^{\text{ell}} \quad \text{LayerNorm + FFN + 残差连接}$$

**对应消融**: 显示空间-时间分解 vs. 联合时空注意力的效率-准确率权衡。

---

### 模块 3: 问题条件化的答案解码（隐含于整体框架）

**直觉**：视频表示必须与问题深度交互，而非简单拼接。

**Baseline 公式** (标准 VQA 融合): 
$$p(a|V,Q) = \text{softmax}(W_a [\bar{v}; q])$$
符号: $\bar{v}$ = 平均池化视频特征，[;] = 拼接，简单晚期融合。

**变化点**: 晚期拼接融合忽略了问题对视频特征的早期调制。本文通过帧选择阶段的问题注入，实现早期-中期-晚期的全阶段条件化。

**本文公式**:
$$\text{Step 1}: \quad \bar{v} = \text{TemporalAvgPool}(\mathbf{z}^{L}) \in \mathbb{R}^d \quad \text{编码器最终层时间池化}$$
$$\text{Step 2}: \quad g = \sigma(W_g [\bar{v}; q]) \odot \tanh(W_m [\bar{v}; q]) \quad \text{门控融合机制（可选）}$$
$$\text{最终}: \quad p(a|V,Q) = \text{softmax}(W_o (\bar{v} \odot q + g)) \quad \text{Hadamard 积强化交互，输出答案分布}$$

**对应消融**: 显示不同融合策略的准确率差异。

## 实验与分析



**主实验结果**：

| Method | MSRVTT-QA | MSVD-QA | ActivityNet-QA | TGIF-QA | 平均 |
|:---|:---|:---|:---|:---|:---|
| TimeSFormer + Uniform |  |  |  |  | — |
| HORNet (本文) |  |  |  |  | — |
| Δ | — | — | — | — | — |

**核心发现分析**：

1. **任务引导选择的有效性**：若 MSRVTT-QA/MSVD-QA 等短视频基准上提升显著，说明帧选择对密集采样冗余的削减有效；若 ActivityNet-QA 等长视频基准提升更大，则验证本文对「长视频关键帧稀释问题」的核心假设。

2. **效率-准确率权衡**：HORNet 编码器仅处理 K < T 帧，理论计算量与 K/T 比例成正比。需关注 (1) K=2/4/8 时的准确率下降曲线是否平缓；(2) 实际 GPU 内存节省是否匹配理论预期。

**消融实验**（ 若存在）：

| 变体 | 设置 | 准确率变化 |
|:---|:---|:---|
| w/o task guidance | 均匀随机选择 K 帧 |  |
| w/o spatial-temporal decoupling | 联合时空注意力 |  |
| w/o question residual | 移除编码器中的 q_proj 注入 |  |
| hard vs. soft selection | Gumbel-Softmax 直通估计 vs. 可微软选择 |  |

**关键模块重要性**：任务引导选择器预期为最重要组件；问题残差连接可能在中等增益区间。

**公平性检查**：
- Baselines 强度：TimeSFormer 为 2021 年方法，需对比是否包含更 recent 的 Video-LLaVA、LLaVA-NeXT-Video 等强基线
- 计算成本：帧选择模块引入的额外参数量（q^⊤ W_s h_i 为一层线性映射，可忽略）与节省的编码计算
- 失败案例：问题歧义导致错误帧选择、快速动作场景中单帧选择不足、需要跨远距离帧推理时 K 帧限制

## 方法谱系与知识库定位

**方法家族**：Video-Language Pretraining → 高效视频-语言理解 → 任务自适应视频采样

**父方法**：TimeSFormer [Bertasius et al., 2021] — 提供时空分解自注意力的基础架构，HORNet 继承其空间-时间解耦设计，但将全帧输入改为条件化选择输入。

**改变的插槽**（对比 TimeSFormer）：
- **架构**：前端新增任务引导帧选择器（问题-帧交叉注意力门控）
- **目标**：保持 VideoQA 答案预测目标不变，但增加选择一致性辅助损失（可能）
- **训练配方**：端到端联合训练帧选择器与编码器，或交替优化（待确认）
- **数据策划**：标准 VideoQA 数据集，无额外数据
- **推理**：动态帧数 K 可根据计算预算调整

**直接对比基线**：
- **TimeSFormer**：HORNet 在其编码器前增加任务条件化选择，计算更高效
- **PSAC/B2T（显著性帧选择）**：HORNet 用问题嵌入替代纯视觉显著性，实现语义级选择
- **Video-LLaVA 等 LLM-based 方法**：HORNet 专注编码器层面效率优化，非生成式答案解码

**后续方向**：
1. 扩展至多轮对话式 VideoQA，帧选择需考虑对话历史条件
2. 与视频压缩/令牌化（如 VideoPoet、MAGVIT）结合，在 patch 层面而非帧层面做选择
3. 零样本迁移：预训练的任务引导选择器能否泛化到新领域视频

**知识库标签**：
- **模态 (modality)**：video + text
- **范式 (paradigm)**：discriminative / encoder-only with selection
- **场景 (scenario)**：Video Question Answering
- **机制 (mechanism)**：cross-modal attention gating, task-conditioned dynamic computation
- **约束 (constraint)**：computational efficiency, fixed budget frame selection

