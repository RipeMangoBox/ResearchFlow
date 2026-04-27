---
title: 'Unified-IO 2: Scaling Autoregressive Multimodal Models with Vision Language Audio and Action'
type: paper
paper_level: C
venue: CVPR
year: 2024
paper_link: null
aliases:
- 统一自回归多模态大模型：视觉语言音频动作
- Unified-IO 2
acceptance: Highlight
cited_by: 312
code_url: https://unified-io-2.allenai.org/
method: Unified-IO 2
---

# Unified-IO 2: Scaling Autoregressive Multimodal Models with Vision Language Audio and Action

[Code](https://unified-io-2.allenai.org/)

**Topics**: [[T__Image_Generation]], [[T__Video_Understanding]] | **Method**: [[M__Unified-IO_2]] | **Datasets**: [[D__MS-COCO]], [[D__MMBench]], [[D__SEED-Bench]] (其他: GRIT, Kinetics-400 Action)

| 中文题名 | 统一自回归多模态大模型：视觉语言音频动作 |
| 英文题名 | Unified-IO 2: Scaling Autoregressive Multimodal Models with Vision Language Audio and Action |
| 会议/期刊 | CVPR 2024 (Highlight) |
| 链接 | [arXiv](https://arxiv.org/abs/2312.17172) · [Code](https://unified-io-2.allenai.org/) · [Project](https://unified-io-2.allenai.org/) |
| 主要任务 | 视觉-语言理解、文本到图像/音频生成、视频理解、音频理解、机器人动作预测、3D目标检测 |
| 主要 baseline | Unified-IO, BLIP-2, InstructBLIP, Flamingo, CoDi, ImageBind, LLaVa-1.5, Cube-RCNN, MBT |

> [!abstract] 因为「现有多模态模型仅能处理视觉-语言模态，无法统一支持音频、视频、动作生成」，作者在「Unified-IO」基础上改了「扩展为支持视觉、语言、音频、动作的自回归 transformer，引入 diffusion decoder 处理连续信号生成」，在「MMBench / SEED-Bench / Kinetics-400 / AudioCaps」上取得「MMB 最高分 +3.8、SEED-Bench 最强 7B 模型、Kinetics-400 73.8% Top-1、AudioCaps +10.4 超过 CoDi」

- **MMBench**: 最高分，较 LLaVa-1.5 13B 提升 +3.8 分
- **Kinetics-400**: UIO-2XXL 达到 73.8% Top-1 Accuracy，超过 ImageBind 50.0% 达 +23.8
- **AudioCaps**: UIO-2XXL 89.3，超过 CoDi 78.9 达 +10.4

## 背景与动机

当前人工智能领域的一个核心挑战是：如何让单一模型同时理解并生成多种模态的内容。例如，用户希望模型既能看懂图片、听懂声音，又能根据描述画出图像、生成音频，甚至控制机器人动作。然而，现有方案往往只能覆盖其中部分能力。

**Unified-IO** 作为前期工作，首次尝试用统一的 encoder-decoder transformer 处理视觉与语言任务，通过将所有输入输出转化为离散 token 序列实现统一。但其局限明显：仅支持图像和文本两种模态，无法处理音频、视频时序信息，也无法生成高质量的连续信号（如真实感图像或自然声音）。**BLIP-2** 采用冻结图像编码器 + 大语言模型的策略，虽在视觉-语言理解上表现强劲，但模态间仍是拼接而非真正统一表示，且不支持生成任务。**Flamingo** 通过交叉注意力机制注入视觉信息，在少样本视频理解上取得突破，但其架构专为视觉-语言设计，未扩展至音频与动作领域。**CoDi** 虽实现任意模态组合生成，但采用多阶段扩散模型拼接，缺乏自回归模型的统一序列建模能力。

这些方法的共同短板在于：**没有一种架构能在单一自回归框架内同时完成理解（图像、视频、音频、文本）与生成（图像、音频、文本、动作）**，且对连续信号的生成质量远不及专门的扩散模型。本文正是要解决这一缺口——在保持自回归模型统一序列建模优势的同时，引入 diffusion decoder 处理连续生成，并将模态覆盖扩展至视频、音频与机器人动作。
![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/2a09530a-3463-41fe-bc59-c031fbc03f3b/figures/fig_001.png)
*Figure: UNIFIED-IO 2 is an instruction-following model with a huge breadth of abilities and supported modalities. It can generate images (red box), including image editing, image generation, depth estimation, surface normal estimation, and future frame prediction etc. It can*



## 核心创新

核心洞察：自回归序列建模与扩散连续生成可以共存于统一框架，因为不同模态的信号本质可被编码为统一的 latent token 序列，从而使单一模型同时胜任离散预测（文本/动作）与连续合成（图像/音频）成为可能。

| 维度 | Baseline (Unified-IO) | 本文 (Unified-IO 2) |
|:---|:---|:---|
| **支持模态** | 图像 + 文本 | 图像 + 视频 + 音频 + 文本 + 动作 |
| **生成机制** | 离散 token 自回归解码 | 文本/动作自回归 + 图像/音频 diffusion 采样 |
| **架构组件** | Vision encoder + Text decoder | 新增 Audio encoder、Video processor、Action head、Diffusion image/audio decoder |
| **训练数据** | 图像-文本、视觉-语言数据集 | 扩展视频、音频、机器人动作数据，明确采样比例（如 3D 检测仅占 1.0%） |
| **推理策略** | 纯自回归解码 | 模态自适应：文本自回归 + 图像/音频 diffusion 多步去噪 |

## 整体框架


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/2a09530a-3463-41fe-bc59-c031fbc03f3b/figures/fig_002.png)
*Figure: UNIFIED-IO 2 architecture. Input text, images, audio, or image/audio history are encoded into sequences of embeddings which*



Unified-IO 2 的整体数据流遵循「多模态编码 → 统一序列 → 自回归处理 → 模态特定解码」的范式：

1. **Multimodal Input Encoders（多模态输入编码器）**：接收原始像素（图像/视频帧）、音频波形、文本 token。图像经 ViT 编码为 patch tokens；视频通过时序采样扩展为视频 tokens；音频经频谱编码为音频 tokens；文本直接 tokenize。所有模态最终映射到统一的 latent token 空间。

2. **Autoregressive Transformer Backbone（自回归 Transformer 主干）**：接收统一后的 token 序列，执行因果自注意力计算，输出 contextualized hidden states。这是模型的核心计算单元，参数规模随 UIO-2L / XL / XXL 递增。

3. **Text Decoder Head（文本解码头）**：从 hidden states 预测下一个文本 token，采用标准语言模型目标。

4. **Diffusion Image Generator（图像扩散生成器）**：接收 hidden states 作为条件，通过多步去噪扩散过程生成连续图像像素，替代了 Unified-IO 的离散图像 token 解码。

5. **Diffusion Audio Generator（音频扩散生成器）**：同理，以 hidden states 为条件生成音频波形/频谱。

6. **Action Prediction Head（动作预测头）**：将 hidden states 映射为机器人动作指令（如末端执行器位姿、夹爪开合等离散或连续动作参数）。

```
Raw Inputs (Image/Video/Audio/Text)
    ↓
[Image Encoder] [Video Encoder] [Audio Encoder] [Text Tokenizer]
    ↓
Unified Token Sequence
    ↓
Autoregressive Transformer
    ↓
[Text Head] [Diffusion Image] [Diffusion Audio] [Action Head]
    ↓
Outputs (Text / Image / Audio / Action)
```

## 核心模块与公式推导

本文未采用显式的数学损失函数推导作为核心创新点，其技术贡献主要体现在架构扩展与训练策略上。以下从训练目标与推理机制两个关键维度进行说明。

### 模块 1: 统一自回归训练目标（对应框架图 Transformer 主干）

**直觉**: 所有模态的理解任务均可统一为「下一个 token 预测」，这是自回归模型泛化至多任务的根基。

**Baseline 公式** (Unified-IO): 
$$L_{\text{UIO}} = -\sum_{t} \log P(x_t | x_{<t}; \theta)$$
符号: $x_t$ 为统一离散 token 序列中的第 $t$ 个 token（图像 patch ID 或文本 token），$\theta$ 为模型参数。

**变化点**: Unified-IO 仅处理图像与文本的离散 token。本文需扩展至视频时序、音频频谱、以及连续图像/音频的生成——后者无法直接用离散 token 建模。

**本文公式（推导）**:
$$\text{Step 1}: \quad L_{\text{AR}} = -\sum_{t \in \mathcal{T}_{\text{txt}}} \log P(w_t | w_{<t}, v, a; \theta) \quad \text{（文本、动作 token 的自回归预测）}$$
$$\text{Step 2}: \quad L_{\text{diff}} = \mathbb{E}_{z_0, \epsilon, t} \left[ \|\epsilon - \epsilon_\theta(z_t, t, c)\|^2 \right] \quad \text{（扩散损失：预测添加到 latent 的噪声）}$$
$$\text{最终}: \quad L_{\text{total}} = L_{\text{AR}} + \lambda_{\text{diff}} \cdot L_{\text{diff}} + \lambda_{\text{aux}} \cdot L_{\text{aux}}$$
其中 $w_t$ 为文本 token，$v, a$ 为视觉/音频条件；$z_t$ 为扩散前向过程第 $t$ 步的 noisy latent，$c$ 为 transformer 输出的条件向量，$\epsilon_\theta$ 为噪声预测网络；$L_{\text{aux}}$ 涵盖视频、音频理解等辅助任务。

**对应消融**: Table 3 显示不同模态混合比例对 GRIT 性能的影响，纯视觉-语言训练在扩展模态后显著下降。

### 模块 2: 扩散解码器用于连续生成（对应框架图 Diffusion Image/Audio Generator）

**直觉**: 图像与音频的连续高维分布不适合自回归逐 token 生成，扩散模型在像素/波形空间的迭代去噪更适合高质量合成。

**Baseline 形式** (标准 Latent Diffusion Model, e.g. Stable Diffusion):
$$L_{\text{LDM}} = \mathbb{E}_{\mathcal{E}(x), \epsilon, t} \left[ \|\epsilon - \epsilon_\theta(z_t, t, \tau_\theta(y))\|^2 \right]$$
符号: $\mathcal{E}$ 为 VAE 编码器，$z_t = \sqrt{\bar{\alpha}_t} z_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$，$\tau_\theta(y)$ 为文本条件编码（通常用 CLIP 或 T5）。

**变化点**: 标准 LDM 使用独立的文本编码器 $\tau_\theta$；本文将条件 $c$ 直接替换为自回归 transformer 的 hidden states，实现「理解-生成」一体化，无需外部条件编码器。

**本文公式（推导）**:
$$\text{Step 1}: \quad c = \text{Transformer}(\text{Encoder}(\text{prompt}, \text{image/audio context})) \quad \text{（统一编码获取条件）}$$
$$\text{Step 2}: \quad z_T \sim \mathcal{N}(0, I); \quad z_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(z_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(z_t, t, c)\right) + \sigma_t \xi \quad \text{（DDPM/DDIM 去噪）}$$
$$\text{最终}: \quad \hat{x} = \mathcal{D}(z_0) \quad \text{（VAE 解码器恢复图像/音频波形）}$$
其中 $\mathcal{D}$ 为 VAE 解码器，图像与音频分别使用独立的 VAE 与扩散参数。

**对应消融**: Table 4 显示 MS COCO 文本到图像生成结果，diffusion decoder 相比离散 token 方案在 FID 指标上显著改善（具体数值见原表）。

### 模块 3: 多模态数据混合与采样策略（对应框架图 Data Pipeline）

**直觉**: 不同模态数据量差异巨大（文本数据远多于 3D 检测），需显式控制采样率以防止小模态被淹没。

**Baseline 形式** (标准多任务学习):
$$P(\text{task}) = \frac{1}{N} \quad \text{（均匀采样各任务）}$$

**变化点**: 均匀采样导致视频、音频、动作等稀缺模态训练不足；本文采用按数据量与重要性调整的非均匀采样。

**本文公式（推导）**:
$$\text{Step 1}: \quad P(m) \propto \left(\frac{n_m}{\sum_k n_k}\right)^\alpha \cdot w_m \quad \text{（模态 } m \text{ 的采样概率，} \alpha \in [0,1] \text{ 为温度系数）}$$
$$\text{Step 2}: \quad w_m = \begin{cases} 1.0 & \text{image-text} \\ 0.5 & \text{video} \\ 0.3 & \text{audio} \\ 0.01 & \text{3D detection} \end{cases} \quad \text{（人工设定的模态权重，见 Table 10）}$$
$$\text{最终}: \quad \text{实际步数}_m = P(m) \times \text{total steps}$$

**对应消融**: Table 10 及 Figure 3 显示不同模态混合下的训练损失曲线；3D 检测仅 1.0% 采样率导致 Objectron AP3D 42.4 远低于 Cube-RCNN 的 50.8（Table 7），验证了采样策略对下游性能的直接影响。

## 实验与分析



本文在超过 20 个基准上评估了 Unified-IO 2 的三档规模（UIO-2L / XL / XXL）。核心发现如下：在视觉-语言理解领域，UIO-2XXL 在 **MMBench** 上取得最高分，较 LLaVa-1.5 13B 提升 +3.8 分；在 **SEED-Bench** 上成为当时最强的 7B 级别模型，甚至超过部分 13B 模型。这一结果表明，统一的自回归预训练配合指令微调，能在不依赖专门视觉编码器冻结策略的前提下，达到视觉-语言理解的领先水平。

在多模态扩展任务上，模型展现出明显的规模效应与模态迁移能力。**Kinetics-400** 动作分类中，UIO-2XXL 达到 73.8% Top-1 Accuracy，较 ImageBind 的 50.0% 提升 +23.8；**AudioCaps** 音频描述任务中，UIO-2XXL 89.3 较 CoDi 的 78.9 提升 +10.4，验证了音频编码器与扩散解码器的有效性。然而，视频描述任务（VATEX: 45.6）仍显著落后于 Flamingo-80B（84.2），显示长视频时序建模仍是短板。



消融实验（Table 3、Table 8-11）表明：**模型规模**是最关键的单一变量——L→XL→XXL 在绝大多数任务上呈单调提升。数据混合方面，移除音频或视频数据会导致对应模态性能崩溃，但对视觉-语言任务的负面影响相对温和，说明模态间存在一定的迁移互补性。值得注意的是，3D 检测（Objectron AP3D 42.4 vs Cube-RCNN 50.8）和深度估计（NYUv2 RMSE 0.623 预训练 / 0.423 微调）表现较弱，作者明确归因于训练数据中 3D 任务仅占 1.0%、且缺乏任务特定微调。

公平性检查：对比存在若干不对称性。首先，多数 baseline 标注 *（zero-shot）或 **（few-shot），而 UIO-2 经过指令微调，非严格同等条件。其次，Flamingo-80B 参数量远大于 UIO-2XXL，却在部分任务上被反超，但也存在视频描述大幅领先的情况。再次，GPT-4V、Gemini 等更强闭源模型未纳入对比，Qwen-VL 在图像描述 CIDEr 上实际优于 UIO-2（作者已注明）。最后，机器人操作（VIMA-Bench）与 3D 检测的 baseline 覆盖不足，难以判断是否为该领域的实用选择。

## 方法谱系与知识库定位

Unified-IO 2 属于 **统一多模态大模型（Unified Multimodal Foundation Models）** 谱系，直接继承自 **Unified-IO**（encoder-decoder transformer for vision-language）。谱系演变路径为：Pix2Seq（目标检测序列化）→ Unified-IO（视觉-语言统一）→ **Unified-IO 2（视觉-语言-音频-动作统一 + 扩散生成）**。

**改变的 slots**:
- **Architecture**: 新增 Audio encoder、Video processor、Action head；图像/音频输出由离散 token 改为 Diffusion decoder
- **Data pipeline**: 扩展至视频、音频、机器人动作数据，引入显式模态采样率控制
- **Inference strategy**: 文本/动作保持自回归解码，图像/音频采用 diffusion 多步采样

**直接 baselines 与差异**:
- **Unified-IO**: 直接父方法；UIO-2 扩展模态覆盖并引入扩散生成
- **BLIP-2 / InstructBLIP**: 采用冻结视觉编码器 + LLM；UIO-2 采用端到端统一训练
- **Flamingo**: 交叉注意力视觉注入；UIO-2 采用统一 token 序列自回归
- **CoDi**: 多扩散模型组合；UIO-2 将扩散作为单一自回归模型的输出头
- **LLaVa-1.5**: 视觉指令微调；UIO-2 在更多模态上统一，MMB/SEED 上超越其 13B 版本

**后续方向**:
1. 长视频时序建模：当前视频理解显著落后于 Flamingo-80B，需改进时序编码机制
2. 端到端机器人控制：Action head 仅为初步探索，需与真实机器人闭环交互结合
3. 模态间高效对齐：音频-视觉联合理解（Kinetics-Sounds）仍有提升空间，需更精细的跨模态注意力设计

**标签**: 模态=图像/视频/音频/文本/动作 | 范式=自回归统一建模 + 扩散生成 | 场景=通用多模态理解与生成 | 机制=统一 token 表示 / 模态自适应解码 | 约束=数据采样率敏感 / 3D/深度任务需更多专用数据

