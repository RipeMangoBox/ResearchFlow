---
title: Tell What You Hear From What You See - Video to Audio Generation Through Text
type: paper
paper_level: C
venue: NeurIPS
year: 2024
paper_link: null
aliases:
- VATT：文本桥接的视频到音频生成
- VATT (Video-to-A
- VATT (Video-to-Audio Through Text)
acceptance: Poster
cited_by: 35
code_url: https://github.com/DragonLiu1995/video-to-audio-through-text
method: VATT (Video-to-Audio Through Text)
---

# Tell What You Hear From What You See - Video to Audio Generation Through Text

[Code](https://github.com/DragonLiu1995/video-to-audio-through-text)

**Topics**: [[T__Audio_Generation]], [[T__Video_Understanding]], [[T__Retrieval]] | **Method**: [[M__VATT]] | **Datasets**: [[D__VGGSound]]

| 中文题名 | VATT：文本桥接的视频到音频生成 |
| 英文题名 | Tell What You Hear From What You See - Video to Audio Generation Through Text |
| 会议/期刊 | NeurIPS 2024 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2411.05679) · [Code](https://github.com/DragonLiu1995/video-to-audio-through-text) · [DOI](https://doi.org/10.52202/079017-3213) |
| 主要任务 | Video-to-Audio Generation（视频到音频生成） |
| 主要 baseline | AudioGen, AudioLDM-2, VATT-LLama（无文本条件消融） |

> [!abstract] 因为「视频到音频的直接映射难以捕捉复杂语义对齐」，作者在「LLaVA 多模态大模型」基础上改了「两阶段文本桥接架构（视频→音频描述→音频）」，在「VGGSound test set」上取得「Align Acc 74.89，超越 AudioGen（58.26）和 AudioLDM-2（60.32）」

- **Align Acc**: 74.89（vs. AudioGen 58.26, AudioLDM-2 60.32）
- **KLD**: 2.07（vs. VATT-LLama 2.53，降低 18.2%）
- **FAD**: 3.25（vs. VATT-LLama 3.42，降低 5.0%）

## 背景与动机

视频到音频生成（Video-to-Audio, V2A）的核心挑战在于：一段无声视频可能对应多种合理的音频——同样的"切菜"画面，刀具材质、案板材质、切割速度都会导致截然不同的声音纹理。现有方法试图建立从视觉特征到音频波形的直接映射，但这种端到端映射往往难以显式建模"视频内容→声音语义→声学细节"的层级关系。

AudioGen 采用基于语言模型的音频生成框架，但仅支持文本或无条件输入，无法利用视频视觉信息指导生成；AudioLDM-2 在 latent diffusion 框架下实现文本到音频生成，虽能生成高质量音频，但同样缺乏视频条件注入机制，导致生成音频与视频内容的时间同步性难以保证。VATT-LLama 作为同期探索，尝试将 LLM 直接用于视频到音频生成，但消融实验表明：去除文本条件后（即纯视频特征驱动），模型在 KLD 和 FAD 上显著劣化，说明**直接视觉-音频映射存在语义瓶颈**——视觉特征难以完整编码声音事件的细粒度属性（如音高、谐波结构、混响特性）。



上述方法的共同局限在于：**缺少显式的中间语义表示来桥接视觉与听觉模态**。人类在"看到画面想象声音"时，往往会先形成语言描述（"这应该是金属碰撞的清脆声"），再据此构建具体声学想象。受此启发，本文提出 VATT：通过文本作为中间表示，将视频到音频生成解耦为"视频→音频描述"和"文本→音频"两个子任务，从而利用大规模预训练语言模型的语义理解能力，显式建模视觉-音频对齐。

## 核心创新

核心洞察：**文本是视觉与音频之间的天然桥梁**，因为语言模型已经在大规模预训练中建立了丰富的"声音语义→声学属性"关联，从而使"先描述、后生成"的两阶段范式成为可能——视频内容先被翻译为音频描述文本，再以该文本为条件生成精确匹配的音频。

| 维度 | Baseline（直接视频到音频） | 本文 VATT |
|:---|:---|:---|
| 信息通路 | 视觉特征 → 直接解码音频 | 视觉特征 → 文本描述 → 条件生成音频 |
| 中间表示 | 无显式语义层 | 音频描述文本（可解释、可编辑） |
| 条件机制 | 仅视频特征 | 文本为主 + 视频特征辅助（classifier-free guidance） |
| 训练数据利用 | 依赖视频-音频配对 | 额外利用 1.77M 合成音频描述（V2A Instruction Dataset） |
| 生成可控性 | 低（黑盒映射） | 高（可修改文本描述重新生成） |

与 LLaVA 的差异：VATT 继承了 LLaVA 的视觉-语言对齐架构，但将 LLaVA 的"视觉问答"目标替换为"视频到音频描述"生成，并新增独立的 VATT Audio 模块执行文本条件音频生成，形成完整的跨模态生成闭环。

## 整体框架


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/ab7eacc7-4f89-4374-b71d-4390e1f237f8/figures/Figure_1.png)
*Figure 1 (pipeline): VATT is a flexible audio generative model capable of generating audio via two modes: i) Video-to-Audio: Given a silent video, the model generates audio aligned with the video content. ii) Text-to-Audio: The model generates audio aligned with the text prompt.*



VATT 采用两阶段流水线架构，数据流如下：

**输入**：视频帧序列（5fps 采样，共 50 帧，分辨率 336×336）

**Stage 1: 视频→音频描述（VATT Converter）**
- **eva-CLIP-L Visual Encoder**：输入 50 帧图像，输出每帧 768 维视觉特征
- **VATT Projector**：通过自适应时序池化（50→10 帧）将视觉特征投影到 LLM 嵌入维度（LLama-7B 为 4096，Gemma-2B 为 2048）
- **VATT Converter（LLM）**：基于指令微调的 LLM，输入投影后的视觉特征 + 指令模板，输出音频描述文本（如"金属刀具在木质案板上快速切割蔬菜的清脆节奏声"）

**Stage 2: 文本→音频（VATT Audio）**
- **VATT Audio**：双向 Transformer，接收音频描述文本 + 可选视频特征，通过掩码音频 token 预测生成 Encodec 离散 token（4 码本 × 500 token）
- **Encodec Decoder**：将离散 token 解码为 10 秒音频波形

```
视频帧 (50×336×336×3)
    ↓
eva-CLIP-L → 视觉特征 (50×768)
    ↓
VATT Projector → 池化+投影 (10×LLM_dim)
    ↓
VATT Converter (LLM) → 音频描述文本
    ↓
[文本描述, 视频特征] → VATT Audio
    ↓
掩码 token 预测 → Encodec tokens (4×500)
    ↓
Encodec Decoder → 音频波形
```

关键设计：两阶段均可独立运作——VATT Converter 可输出人类可读的音频描述；VATT Audio 也可接收人工编写的文本提示直接生成音频，实现灵活的"自提示"（self-prompting）或"人提示"（human-prompting）模式。

## 核心模块与公式推导

### 模块 1: Encodec 音频 token 化（数据表示层）

**直觉**: 将高维连续音频波形压缩为离散 token 序列，使生成任务转化为可预测的 token 分类问题，同时保留多尺度声学信息。

**Baseline 公式** (通用神经音频编解码): 
$$\text{Encodec}(x) \rightarrow \{z_1, z_2, z_3, z_4\}, \quad z_l \in \mathbb{R}^{500}$$

符号: $x \in \mathbb{R}^{T \times 1}$ 为原始波形（10秒，16kHz）; $z_l$ 为第 $l$ 个码本的 500 个离散 token，低层码本编码粗粒度语义，高层码本捕获细粒度细节。

**变化点**: VATT 直接采用预训练 Encodec-16kHz，无需修改；但后续生成目标基于此表示重新设计。

---

### 模块 2: 掩码音频 token 预测（VATT Audio 核心目标）

**直觉**: 借鉴 BERT 式掩码语言建模，在音频 token 序列上执行非自回归的条件生成，比自回归或扩散模型更高效；通过精心设计的掩码比例分布平衡生成质量与训练稳定性。

**Baseline 公式** (标准条件生成):
$$p(z | c_{video}) \quad \text{（直接以视频特征为条件）}$$

**变化点**: 基线方法直接以视频特征预测完整音频，缺乏显式语义引导；VATT 引入文本条件 $c_{text}$ 和掩码预测机制，且发现均匀掩码或固定比例掩码会导致训练不稳定或生成质量下降。

**本文公式（推导）**:

$$\text{Step 1}: \quad \mathcal{M} \sim \text{Truncated-Gaussian}(\mu=0.75, \sigma=0.25, [0.5, 1.0])$$
$$\text{（掩码比例从截断高斯采样，保证至少 50% token 被掩码，平均 75%）}$$

$$\text{Step 2}: \quad z_{masked}, z_{unmasked} = \text{Mask}(z_{1:4}, \mathcal{M})$$
$$\text{（按掩码比例将音频 token 分为被掩码与未掩码两部分）}$$

$$\text{最终}: \quad \mathcal{L}_{VATT} = -\mathbb{E}_{z, \mathcal{M}, c_{text}, c_{video}} \left[ \sum_{i \in \mathcal{M}} \log p_\theta(z_i | z_{\neg \mathcal{M}}, c_{text}, c_{video}) \right]$$

**对应消融**: Table 4 显示，将截断高斯掩码替换为均匀分布或其他比例，生成质量显著下降（具体 Δ 值。

---

### 模块 3: Classifier-Free Guidance（训练与推理统一）

**直觉**: 让模型同时学习条件生成和无条件生成分布，推理时通过插值增强文本条件的控制力，解决"文本描述与视频特征冲突时如何取舍"的问题。

**Baseline 公式** (标准扩散/生成模型训练):
$$\mathcal{L}_{CF} = \mathbb{E}_{t, \epsilon, c} \left[ \| \epsilon - \epsilon_\theta(z_t, t, c) \|^2 \right]$$

**变化点**: VATT 将条件 $c = [c_{text}, c_{video}]$ 以 10% 概率整体置零（而非单独 drop），使模型学会联合条件和无条件分布；推理时通过引导尺度 $w$ 显式调节文本条件强度。

**本文公式（推导）**:

$$\text{Step 1（训练）}: \quad c = \begin{cases} \emptyset & \text{with prob. } 0.1 \\ [c_{text}, c_{video}] & \text{otherwise} \end{cases}$$
$$\text{（以 10% 概率联合丢弃文本和视频条件，而非单独 dropout）}$$

$$\text{Step 2（推理）}: \quad \hat{\epsilon} = \epsilon_\theta(z_t, t, c) + w \cdot (\epsilon_\theta(z_t, t, c) - \epsilon_\theta(z_t, t, \emptyset))$$
$$\text{（无条件预测与条件预测的差值按 } w \text{ 缩放，} w=1 \text{ 为标准生成，} w>1 \text{ 增强文本对齐）}$$

**对应消融**: Table 4 中 VATT w/o Sync（去除音频-视觉同步损失）显示同步性指标下降（具体 Δ 值；classifier-free guidance 的消融。

## 实验与分析


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/ab7eacc7-4f89-4374-b71d-4390e1f237f8/figures/Table_1.png)
*Table 1 (quantitative): Quantitative results against audio generation methods on VGGSound test set. 'T' refers to Text Prompt and 'V' refers to Video.*



本文在 **VGGSound test set** 上进行主实验评估，采用 KLD（Kullback-Leibler Divergence，衡量分布相似性，越低越好）、FAD（Frechet Audio Distance，衡量感知质量，越低越好）、Align Acc（音频-视频对齐准确率，越高越好）、CLAP Score（文本-音频对齐分数，越高越好）四个指标。


![Table 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/ab7eacc7-4f89-4374-b71d-4390e1f237f8/figures/Table_2.png)
*Table 2 (quantitative): Quantitative results comparing VATT with text-to-audio generation methods on VGGSound test set. LLM (Large) refers to the LLM layers and LLM (Train) refers to the trainable LoRA parameters.*



从 Table 1 可见，VATT 的核心优势在于 **Align Acc 达到 74.89**，显著超越 AudioGen（58.26，+16.63）和 AudioLDM-2（60.32，+14.57），验证了"文本桥接"策略对视觉-音频语义对齐的显著提升。然而，VATT 在 KLD（2.07 vs. AudioLDM-2 的 1.64）和 FAD（3.25 vs. AudioGen 的 3.13、AudioLDM-2 的 1.86）上仍落后于专门的文本到音频生成模型，CLAP Score（0.376）也低于 AudioGen（0.447）和 AudioLDM-2（0.432）。这表明：**文本条件增强了对齐性，但音频本身的保真度和文本-音频匹配度仍有提升空间**——可能因为 VATT Audio 的非自回归生成方式在细粒度声学建模上不如 latent diffusion。


![Figure 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/ab7eacc7-4f89-4374-b71d-4390e1f237f8/figures/Figure_4.png)
*Figure 4 (result): Subjective evaluation results: Pairwise Comparison of generated audio (VATT) vs. other methods comparing Fidelity and synchronization aspects.*



Figure 4 的主观评估显示，在 Fidelity（保真度）和 Synchronization（同步性）两个维度上，VATT 相比基线方法获得更高的用户偏好率（具体百分比。Table 3 的 NIQ-A 评估和 BWI-Test（Best-Worst Inference Test）进一步验证了生成音频的自然度和同步质量。


![Table 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/ab7eacc7-4f89-4374-b71d-4390e1f237f8/figures/Table_4.png)
*Table 4 (ablation): Architecture Ablation Study. VATT w/o LLM means using VATT without LLM. VATT w/o Sync means training without audio-visual synchronization loss. VATT w/o LVM means without large visual model.*



Table 4 的架构消融显示关键发现：去除 LLM（VATT w/o LLM）导致生成质量显著下降，说明大规模语言模型的语义理解能力至关重要；去除音频-视觉同步损失（VATT w/o Sync）则损害时间对齐性能。值得注意的是，VATT-LLama（无文本条件版本）的 KLD 为 2.53、FAD 为 3.42，均劣于完整 VATT（2.07 和 3.25），证明文本条件的引入确实改善了生成质量。

**公平性审视**：本文比较存在明显局限——未与 FoleyCrafter、SpecVQGAN、Im2Wav 等专门视频到音频方法对比，且仅在 VGGSound 单一 benchmark 上评估；AudioCaps 在表格列表中被提及但未在可见结果中展示。VATT 的 CLAP Score 和 FAD 均落后于 AudioLDM-2，说明其并非全面 SOTA，而是在"对齐准确性"这一特定维度上取得优势。训练计算成本方面，VATT Converter 需 6 epoch（2 epoch projector-only + 4 epoch LoRA+projector），基于 AdamW 优化器、base LR 1e-4，但完整训练时间和 GPU 类型未披露。

## 方法谱系与知识库定位

**方法族**: 多模态大模型驱动的跨模态生成（Multimodal LLM → Cross-modal Generation）

**父方法**: **LLaVA**（Large Language and Vision Assistant）——VATT 继承其视觉-语言投影架构和指令微调范式，但将任务目标从"视觉问答"重构为"视频到音频描述生成"，并新增独立的音频生成模块。

**直接基线对比**:
- **AudioGen / AudioLDM-2**: 纯文本到音频生成，VATT 扩展了视频条件注入和自提示机制
- **VATT-LLama**: VATT 自身消融（无文本条件），证明文本桥接的必要性
- **LLaVA**: VATT 在视觉编码器-LLM 架构上增加时序池化、音频描述目标、以及独立的 VATT Audio

**改动槽位**（slots changed from LLaVA）:
| 槽位 | 改动内容 |
|:---|:---|
| architecture | 单模型 → VATT Converter + VATT Audio 双模块 |
| data_pipeline | 直接生成 → 视频→文本→音频 两阶段 |
| objective | 通用语言建模 → 掩码音频 token 预测 + 截断高斯掩码 |
| inference_strategy | 直接输出 → 自提示（self-prompting）+ classifier-free guidance |
| training_recipe | 单阶段 → 两阶段分步训练（projector → LoRA+projector；无条件 → 有条件）|

**后续方向**:
1. **提升音频保真度**: 探索将 VATT Audio 的掩码预测与 latent diffusion 结合，弥补 FAD 差距
2. **扩展视频-音频同步控制**: 引入显式的时间对齐机制（如音画同步标记预测），超越当前隐式学习
3. **多模态联合编辑**: 利用文本描述的可解释性，实现"修改描述即修改音频"的交互式视频配音

**知识库标签**: 
- modality: video + audio + text（三模态）
- paradigm: two-stage pipeline with intermediate text representation
- scenario: video-to-audio generation, text-to-audio generation
- mechanism: masked token prediction, classifier-free guidance, instruction tuning
- constraint: non-autoregressive generation, LoRA efficient fine-tuning

