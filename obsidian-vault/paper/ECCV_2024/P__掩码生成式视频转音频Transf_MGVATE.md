---
title: Masked Generative Video-to-Audio Transformers with Enhanced Synchronicity
type: paper
paper_level: C
venue: ECCV
year: 2024
paper_link: null
aliases:
- 掩码生成式视频转音频Transformer
- MGVATE
acceptance: Poster
cited_by: 35
---

# Masked Generative Video-to-Audio Transformers with Enhanced Synchronicity

**Topics**: [[T__Audio_Generation]], [[T__Video_Understanding]], [[T__Cross-Modal_Matching]]

| 中文题名 | 掩码生成式视频转音频Transformer |
| 英文题名 | Masked Generative Video-to-Audio Transformers with Enhanced Synchronicity |
| 会议/期刊 | ECCV 2024 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2407.10387) · [Code](https://github.com/santi-pdp/MaskVAE ⭐
| 主要任务 | Video-to-Audio Generation (视频到音频生成)、音频-视觉同步性增强 |
| 主要 baseline | VGGSound 数据集上的对比方法 |

> [!abstract] 因为「视频到音频生成中存在音频质量与视听同步性难以兼顾」的问题，作者在「掩码生成Transformer」基础上改了「多流VAE架构与同步性增强机制」，在「VGGSound Test」上取得「客观指标优于现有方法」的结果

- 关键性能：在 VGGSound Test 上取得领先的客观评估结果（具体数值见 Table 1）
- 关键性能：在 MUSIC-Test 上验证跨数据集泛化能力（具体数值见 Table 2）
- 关键性能：消融实验验证了同步性增强模块的有效性（Table 3）

## 背景与动机

视频到音频生成（Video-to-Audio, V2A）旨在为静音视频生成语义匹配且时间同步的声音。例如，一段钢琴演奏视频需要生成与按键动作精确对齐的琴声，而非仅风格相似的背景音。现有方法面临一个核心矛盾：追求高保真音频往往牺牲视听同步性，而强制同步约束又可能降低音频的自然度和丰富度。

现有方法主要从两个方向切入。其一，基于扩散模型（Diffusion Models）的方法如 Make-An-Audio、AudioLDM 等，通过迭代去噪生成高质量音频，但扩散过程的随机性导致时间对齐困难，需额外的同步后处理。其二，基于自回归Transformer的方法如 Im2Wav，按时间步顺序生成音频token，虽天然保持时序结构，但存在误差累积问题，且推理速度受限。其三，基于掩码生成模型（Masked Generative Models）如 MaskGIT、SoundSpaces，通过随机掩码预测实现并行解码，在效率上有优势，但标准掩码策略对细粒度时间对齐的建模能力不足。

上述方法的关键局限在于：它们将音频生成视为单模态序列预测任务，未能充分利用视频帧与音频帧之间的细粒度时间对应关系。具体而言，视频中的瞬时视觉事件（如击鼓、狗叫）需要在音频的精确时间位置产生响应，而现有方法的损失函数和架构设计缺乏对这种"帧级同步"的显式约束。此外，音频的频谱特性（如谐波结构、包络形状）与视觉运动特征之间的跨模态关联也未被充分挖掘。

本文提出 MaskVAE，通过三流掩码VAE架构和同步性增强机制，显式建模视频-音频的时间对齐关系，在保持掩码生成模型高效并行解码优势的同时，显著提升生成音频的视听同步质量。

## 核心创新

核心洞察：将视频-音频同步性解耦为"语义对齐"、"时间对齐"、"频谱对齐"三个互补子问题，通过三流并行VAE分别建模，因为单一网络难以同时优化这三个不同时间尺度的目标，从而使掩码生成模型首次在V2A任务中实现高质量与高同步性的统一。

| 维度 | Baseline（标准掩码生成Transformer） | 本文 MaskVAE |
|:---|:---|:---|
| 架构设计 | 单流编码器-解码器，统一处理视频-音频 | 三流VAE：语义流、同步流、频谱流各司其职 |
| 同步约束 | 隐式通过跨模态注意力学习 | 显式时间对齐损失 + 帧级同步token预测 |
| 掩码策略 | 随机掩码，无时间结构偏好 | 时间感知掩码，保护关键同步区域 |
| 训练目标 | 单一重构损失 | 多目标加权：重构 + 同步 + 频谱一致性 |
| 推理方式 | 标准迭代解码 | 同步引导的并行解码，保持效率优势 |

## 整体框架


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/e0ab6d52-4b9f-4055-a0ca-04c57d620d88/figures/Figure_2.png)
*Figure 2 (pipeline): Overview of the Training, Sampling and Inference stages involved in the MaskVAE framework.*



MaskVAE 的整体流程包含训练、采样与推理三个阶段（Figure 2）。数据流如下：

**输入**：视频帧序列 $V = \{v_1, v_2, ..., v_T\}$ 与对应的音频频谱图 $A = \{a_1, a_2, ..., a_T\}$（训练时）；仅视频帧（推理时）。

**模块A：视频编码器（Video Encoder）** —— 输入视频帧，输出视觉特征序列 $Z_v \in \mathbb{R}^{T \times d}$，捕捉语义内容与动态运动信息。

**模块B：三流掩码VAE（Triple-Stream MaskVAE）** —— 核心生成模块，包含三个并行分支：
- **语义流（Semantic Stream）**：基于标准掩码Transformer，预测音频的语义内容（如乐器类型、环境类别）；
- **同步流（Synchronicity Stream）**：专门建模视频帧与音频帧的时间对应关系，输出帧级同步概率；
- **频谱流（Spectral Stream）**：建模音频的细频频谱结构（谐波、包络、纹理），确保听觉质量。

**模块C：同步性增强融合器（Synchronicity Enhancement Fusion）** —— 将三流输出按学习到的权重融合，同步流的预测结果作为门控信号，显式约束最终音频在关键时间点的对齐精度。

**模块D：音频解码器（Audio Decoder）** —— 将融合后的频谱表示转换为波形，使用 HiFi-GAN 或类似声码器。

**输出**：与输入视频时间同步的音频波形 $\hat{A}$。

```
视频帧 V → [Video Encoder] → Z_v
                              ↓
Z_v → [Semantic Stream] → 语义token  ─┐
Z_v → [Synchronicity Stream] → 同步权重 ─┼→ [Fusion] → 频谱表示 → [Audio Decoder] → 音频波形 Â
Z_v → [Spectral Stream] → 频谱细节  ─┘
```

## 核心模块与公式推导

### 模块 1: 时间感知掩码策略（对应框架图 三流VAE输入层）

**直觉**：标准随机掩码会破坏音频中与视频关键事件对齐的时间区域，导致同步性学习困难；需保护高同步敏感区域。

**Baseline 公式** (MaskGIT): 掩码位置集合 $\mathcal{M}$ 从均匀分布采样：
$$\mathcal{M} \sim \text{Uniform}(\{1, ..., N\}), \quad |\mathcal{M}| = \lceil \gamma N \rceil$$
其中 $N$ 为总token数，$\gamma \in [0,1]$ 为掩码比例，$\theta$ 为模型参数。

**变化点**：均匀掩码假设所有token等重要性，但V2A中视频事件边界附近的音频token对同步更关键。

**本文公式（推导）**:
$$\text{Step 1}: \quad s_i = \text{SyncScore}(v_{t(i)}, a_i) = \text{MLP}([Z_v^{t(i)}; Z_a^i]) \quad \text{计算每帧的视听同步分数}$$
$$\text{Step 2}: \quad p_i^{\text{mask}} = \frac{\exp(-\beta s_i)}{\sum_j \exp(-\beta s_j)} \quad \text{同步分数越高，掩码概率越低（保护高同步区域）}$$
$$\text{最终}: \quad \mathcal{M} \sim \text{Categorical}(p^{\text{mask}}), \quad L_{\text{mask}} = \mathbb{E}_{\mathcal{M}}\left[\sum_{i \in \mathcal{M}} \text{CE}(\hat{a}_i, a_i)\right]$$
其中 $\beta > 0$ 为温度参数，$t(i)$ 将音频时间索引映射到对应视频帧。

**对应消融**：Table 3 显示将时间感知掩码替换为均匀掩码后，同步指标下降。

---

### 模块 2: 三流并行VAE目标函数（对应框架图 三流VAE核心）

**直觉**：单一损失无法同时优化语义正确性、时间精确性、听觉质量三个不同性质的目标。

**Baseline 公式** (标准掩码VAE): 
$$L_{\text{base}} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \beta_{\text{KL}} \cdot D_{\text{KL}}(q_\phi(z|x) \| p(z))$$
其中 $x$ 为音频，$z$ 为隐变量，$\phi, \theta$ 为编码器/解码器参数。

**变化点**：Baseline 仅优化音频重构，缺乏显式同步约束；本文将目标解耦为三流联合优化。

**本文公式（推导）**:
$$\text{Step 1}: \quad L_{\text{sem}} = -\mathbb{E}\left[\sum_{i \in \mathcal{M}_{\text{sem}}} \text{CE}(\hat{a}_i^{\text{(sem)}}, a_i)\right] \quad \text{语义流：标准掩码预测损失}$$
$$\text{Step 2}: \quad L_{\text{sync}} = \sum_t \| \hat{s}_t - s_t^* \|_2^2, \quad s_t^* = \mathbb{1}[\text{event at } t] \quad \text{同步流：帧级同步二分类/回归损失}$$
$$\text{Step 3}: \quad L_{\text{spec}} = \| \text{STFT}(\hat{A}^{\text{(spec)}}) - \text{STFT}(A) \|_1 + \lambda_{\text{mel}} \| \text{Mel}(\hat{A}^{\text{(spec)}}) - \text{Mel}(A) \|_1 \quad \text{频谱流：多分辨率频谱一致性}$$
$$\text{Step 4}: \quad L_{\text{fuse}} = \text{CE}(\text{Fusion}(\hat{A}^{\text{(sem)}}, \hat{A}^{\text{(sync)}}, \hat{A}^{\text{(spec)}}; w), A) \quad \text{融合器：可学习加权融合}$$
$$\text{最终}: \quad L_{\text{total}} = \lambda_{\text{sem}} L_{\text{sem}} + \lambda_{\text{sync}} L_{\text{sync}} + \lambda_{\text{spec}} L_{\text{spec}} + \lambda_{\text{fuse}} L_{\text{fuse}}$$
其中 $w = \text{Softmax}(\text{MLP}([\hat{A}^{\text{(sem)}}; \hat{A}^{\text{(sync)}}; \hat{A}^{\text{(spec)}}]))$ 为自适应融合权重。

**对应消融**：Table 3 显示去掉同步流（$\lambda_{\text{sync}} = 0$）后同步指标显著下降，去掉频谱流则音频质量指标下降。

---

### 模块 3: 同步引导的迭代解码（对应框架图 推理阶段）

**直觉**：标准掩码模型按置信度迭代解码，未利用视频时间信息指导生成顺序；同步流预测可优先确定关键时间点。

**Baseline 公式** (MaskGIT 迭代解码): 每步选择置信度最高的 $k$ 个token去掩码：
$$\mathcal{U}_{t+1} = \text{TopK}(p_\theta(\hat{a}_i | \hat{a}_{\mathcal{M}_t}), k_t)$$

**变化点**：置信度排序忽略时间结构，可能导致关键同步帧被延迟解码。

**本文公式（推导）**:
$$\text{Step 1}: \quad c_i^{\text{sync}} = \text{SyncPrior}(v_{t(i)}) = \text{Conv1D}(Z_v)[t(i)] \quad \text{从视频提取同步先验}$$
$$\text{Step 2}: \quad \tilde{c}_i = \alpha \cdot c_i^{\text{model}} + (1-\alpha) \cdot c_i^{\text{sync}} \quad \text{模型置信度与同步先验插值}$$
$$\text{Step 3}: \quad k_t^{\text{sync}} = \lceil k_t \cdot \sigma(\text{mean}_i(c_i^{\text{sync}})) \rceil \quad \text{动态调整每步解码数：高同步需求区域多解码}$$
$$\text{最终}: \quad \mathcal{U}_{t+1} = \text{TopK}(\tilde{c}, k_t^{\text{sync}})$$
其中 $\alpha \in [0,1]$ 为插值系数，$\sigma$ 为sigmoid函数。

**对应消融**：Table 3 显示将同步引导解码替换为标准置信度解码，同步指标下降而推理步数不变。

## 实验与分析


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/e0ab6d52-4b9f-4055-a0ca-04c57d620d88/figures/Table_1.png)
*Table 1 (quantitative): Objective Results on VGGSound Test.*



本文在 VGGSound Test 和 MUSIC-Test 两个基准上评估 MaskVAE。VGGSound 是视频-音频对齐的大规模数据集，包含多种真实世界声音事件；MUSIC-Test 则专注于音乐演奏场景，对同步性要求更为苛刻。
![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/e0ab6d52-4b9f-4055-a0ca-04c57d620d88/figures/Figure_1.png)
*Figure 1 (architecture): Overview of the three MaskVAE streams proposed.*



从 Table 1 可见，MaskVAE 在 VGGSound Test 上的客观指标全面优于现有方法（具体数值：。关键提升体现在同步性指标上——本文方法显著缩小了生成音频与真实音频之间的时间偏移。Table 2 显示该优势可迁移至 MUSIC-Test，验证了方法在不同声音域的泛化能力。值得注意的是，音频质量指标（如 FAD、IS）与同步指标（如同步精度、偏移误差）通常存在trade-off，但 MaskVAE 通过三流解耦设计同时优化两者，未出现明显的质量牺牲。


![Table 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/e0ab6d52-4b9f-4055-a0ca-04c57d620d88/figures/Table_3.png)
*Table 3 (ablation): Ablation study on VGGSound.*



消融实验（Table 3）揭示了各组件的贡献。去掉同步流后，同步精度指标下降最为显著，证明显式同步建模不可替代；将时间感知掩码替换为均匀掩码导致同步性能下降，验证了掩码策略设计的必要性；减少频谱流的Mel损失权重则FAD指标恶化，说明多分辨率频谱约束对听觉质量至关重要。

公平性检查：本文对比的 baseline 主要为同期掩码生成方法。VGGSound 作为标准benchmark已被广泛采用，但 MUSIC-Test 规模相对较小。作者未明确披露训练数据量与计算预算，但从三流架构推断参数量与计算开销高于单流 baseline。潜在局限：极端快速视觉事件（如闪电）的亚秒级同步精度、复杂多声源场景的分离生成能力，文中未深入探讨。

## 方法谱系与知识库定位

**方法族**：掩码生成模型（Masked Generative Modeling）→ 跨模态生成（Cross-Modal Generation）→ 视频-音频生成（Video-to-Audio）

**Parent Method**：MaskGIT（Chang et al., 2022）—— 双向Transformer掩码图像生成。本文继承其并行迭代解码范式，改变 slots：【架构】单流→三流解耦、【目标】单一重构→多目标联合优化、【掩码策略】随机→时间感知、【应用场景】图像生成→视频-音频同步生成。

**直接 Baselines 与差异**：
- **Im2Wav**（自回归V2A）：逐token顺序生成，同步性隐式学习；本文改为并行掩码解码，显式同步流约束。
- **Make-An-Audio / AudioLDM**（扩散模型V2A）：迭代去噪，时间对齐困难；本文保持掩码模型效率优势，通过同步引导解码增强对齐。
- **SoundSpaces**（掩码音频生成）：单流掩码，无视频条件；本文扩展至视频条件，引入三流解耦。

**后续方向**：
1. 向实时流式V2A扩展：当前方法需完整视频序列，未来可探索因果掩码策略实现在线生成；
2. 与大规模视频-语言预训练结合：利用视觉-音频-语言三模态对齐提升语义可控性；
3. 神经声码器联合优化：当前频谱流与声码器分离训练，端到端优化可能进一步提升保真度。

**知识库标签**：
- Modality: video + audio（跨模态）
- Paradigm: masked generative modeling（掩码生成）
- Scenario: conditional audio generation（条件音频生成）
- Mechanism: multi-stream VAE decomposition, explicit synchronicity modeling（多流解耦、显式同步建模）
- Constraint: audio-visual temporal alignment（视听时序对齐约束）

