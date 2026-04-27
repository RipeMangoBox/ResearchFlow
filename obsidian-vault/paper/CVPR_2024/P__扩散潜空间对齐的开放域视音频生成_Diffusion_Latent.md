---
title: 'Seeing and Hearing: Open-domain Visual-Audio Generation with Diffusion Latent Aligners'
type: paper
paper_level: C
venue: CVPR
year: 2024
paper_link: null
aliases:
- 扩散潜空间对齐的开放域视音频生成
- Diffusion Latent
- Diffusion Latent Aligners
acceptance: Poster
cited_by: 123
method: Diffusion Latent Aligners
---

# Seeing and Hearing: Open-domain Visual-Audio Generation with Diffusion Latent Aligners

**Topics**: [[T__Video_Generation]], [[T__Audio_Generation]], [[T__Cross-Modal_Matching]] | **Method**: [[M__Diffusion_Latent_Aligners]] | **Datasets**: VGGSound Video-to-Audio, VGGSound Image-to-Audio

| 中文题名 | 扩散潜空间对齐的开放域视音频生成 |
| 英文题名 | Seeing and Hearing: Open-domain Visual-Audio Generation with Diffusion Latent Aligners |
| 会议/期刊 | CVPR 2024 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2402.17723) · [Code] · [Project] |
| 主要任务 | Video-to-Audio (V2A)、Image-to-Audio (I2A)、Audio-to-Video (A2V)、Joint Video-Audio Generation |
| 主要 baseline | SpecVQGAN, Im2Wav, TempoTokens, MM-Diffusion, Ours-vanilla |

> [!abstract] 因为「现有方法各自独立生成视频或音频，缺乏跨模态对齐机制导致音视频语义不一致」，作者在「AudioLDM + AnimateDiff 组合基线」基础上改了「推理阶段加入基于梯度优化的扩散潜空间对齐器」，在「VGGSound Video-to-Audio」上取得「KL 2.619（相比 SpecVQGAN 降低 20.4%）」

- **Video-to-Audio**: KL ↓ 2.619 vs SpecVQGAN 3.29，ISc ↑ 5.831 vs 5.108
- **Audio-to-Video**: FVD ↓ 402.385 vs TempoToken 1866.285（降低 78.4%），KVD ↓ 34.764 vs 389.096（降低 91.1%）
- **Open-domain Joint VA**: AV-alignbind ↑ 0.096 vs Ours-vanilla 0.074（提升 29.7%），AT-alignbind ↑ 0.138 vs 0.081（提升 70.4%）

## 背景与动机

开放域视音频生成的核心挑战在于：如何让生成的视频与音频在语义和时间上真正"对齐"。例如，一段"吉他弹奏"的视频应当配有清晰的吉他弦音而非嘈杂的背景噪音；一段"海浪拍打礁石"的音频应当对应海浪飞溅的视觉画面而非静止的海滩。然而，现有方法通常将视觉生成与音频生成分解为两个独立任务，各自依赖预训练扩散模型进行单模态生成，缺乏显式的跨模态约束。

**现有方法的处理方式及其局限**：

- **SpecVQGAN** [26]：采用 VQ-GAN 架构进行视频到音频的生成，通过离散化音频表示实现重建，但生成过程完全基于视觉条件的单向映射，无法引入音频质量或跨模态一致性的显式优化，导致生成音频常包含与视频无关的背景噪声。

- **Im2Wav**：针对图像到音频任务，利用图像特征直接预测音频波形或频谱，但同样缺乏对生成音频与输入图像之间语义一致性的迭代精修机制，在复杂开放域场景下对齐质量有限。

- **TempoTokens**：尝试通过音频节奏 token 指导视频生成，但模型容量和训练数据的限制使其难以捕捉细粒度的音画对应关系，生成的视频往往视觉质量差且与输入音频语义错位。

- **MM-Diffusion** [36]：作为无条件联合音视频生成的 state-of-the-art，采用多模态联合扩散模型同时生成视频和音频，但需要从头训练大规模联合模型，且无条件设定限制了其在开放域条件生成场景的应用。

这些方法的共同瓶颈在于：**推理阶段缺乏跨模态对齐机制**。无论是独立的 V2A/A2V 模型还是联合训练模型，都未在扩散模型的去噪过程中引入显式的梯度引导来优化音视频 latent 之间的一致性。本文提出在推理阶段通过优化-based 的 latent aligner 对预训练扩散模型进行引导，无需重新训练即可实现高质量的开放域视音频对齐生成。
![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/d2d03658-e0b1-4961-8f5c-46176d9ceb9a/figures/fig_001.png)
*Figure: Overview. Our approach is versatile and can tackle four tasks: joint video-audio generation (Joint-VA), video-to-audio (V2A),*



## 核心创新

**核心洞察**：扩散模型的去噪过程本质上是在 latent 空间中的渐进优化，因此可以在该空间中直接施加跨模态对齐目标并通过梯度反向传播引导去噪方向，因为预训练扩散模型（AudioLDM、AnimateDiff）已经具备高质量单模态生成能力，从而使无需训练、仅通过推理时优化即可实现开放域音视频对齐生成成为可能。

| 维度 | Baseline (AudioLDM/AnimateDiff/Ours-vanilla) | 本文 (Diffusion Latent Aligners) |
|:---|:---|:---|
| **推理策略** | 直接文本条件去噪生成，单模态独立优化 | 每步去噪时加入基于梯度的 latent 对齐优化，学习率 0.1 (AudioLDM) / 0.01 (AnimateDiff) |
| **架构组成** | 独立预训练扩散模型，无跨模态连接模块 | 保留预训练模型权重，新增 Diffusion Latent Aligner 作为梯度引导模块 |
| **优化目标** | 标准噪声预测目标 $\|\epsilon - \epsilon_\theta(x_t, t, c)\|^2$ | 在 latent 空间增加对齐目标，联合优化单模态质量与跨模态一致性 |
| **适用场景** | 固定任务训练，难以扩展至新域 | 统一框架覆盖 V2A、I2A、A2V、Joint-VA 四项任务 |

## 整体框架


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/d2d03658-e0b1-4961-8f5c-46176d9ceb9a/figures/fig_002.png)
*Figure: The proposed diffusion latent aligner. During the denoising process of generating one specific modality (visual/audio), we*



本文提出 **Diffusion Latent Aligners**，一个统一的优化-based 框架，通过梯度引导预训练扩散模型的 latent 空间实现跨模态对齐。系统支持四种任务：Joint Video-Audio Generation（联合生成）、Video-to-Audio（视频转音频）、Image-to-Audio（图像转音频）、Audio-to-Video（音频转视频）。

**数据流与核心模块**：

1. **输入条件** → 根据任务类型接收视频帧/图像、音频片段、或文本 prompt
2. **Key Frame Extractor**（非新模块）→ 从输入视频中提取代表性关键帧，用于后续图像 caption 生成
3. **Image/Audio Caption Model**（非新模块）→ 将关键帧或音频转换为文本描述，作为 Ours-vanilla 基线的条件输入；本文方法可选择直接使用该条件或绕过它进行 latent 级优化
4. **Pretrained AudioLDM**（非新模块）→ 输入视觉特征或文本 prompt，输出生成音频的 latent 表示；在 V2A/I2A 任务中被引导优化
5. **Pretrained AnimateDiff**（非新模块）→ 输入音频特征或文本 prompt，输出生成视频的 latent 表示；在 A2V 任务中被引导优化
6. **Diffusion Latent Aligner**（**新模块**）→ 核心创新：接收双模态的扩散 latent（$z_t^{video}$ 和 $z_t^{audio}$），计算跨模态对齐损失并通过梯度反向传播更新 latent，输出对齐后的 latent 供下一步去噪使用
7. **输出生成** → 经过 25-30 步引导去噪后，解码得到最终对齐的视频与/或音频

```
输入条件 ──┬─→ [Caption Model] ──→ 文本条件 ──┐
           │                                  ↓
           └────────────────────────────→ [AudioLDM] ──┐
                                                      ├──→ [Diffusion Latent Aligner] ──→ 对齐 latent ──→ 解码 ──→ 音频/视频输出
           └────────────────────────────→ [AnimateDiff] ─┘
                                                      ↑
                                    梯度反馈: ∇_{z_t} L_align(z_t^{video}, z_t^{audio})
```

## 核心模块与公式推导

本文的核心公式围绕**扩散潜空间对齐优化**展开。由于原文未提供显式的数学公式推导，以下基于方法描述重构其优化框架。

### 模块 1: 标准扩散去噪（Baseline: AudioLDM / AnimateDiff）

**直觉**：预训练扩散模型通过逐步去噪从高斯分布生成数据，是后续对齐的基础。

**Baseline 公式 (DDPM/DDIM 去噪)**:
$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t, c)\right) + \sigma_t z$$

符号: $x_t$ = 时刻 $t$ 的 noisy latent, $\epsilon_\theta$ = 噪声预测网络, $c$ = 条件（文本/图像/音频）, $\alpha_t, \bar{\alpha}_t$ = 扩散调度参数, $z \sim \mathcal{N}(0, I)$

**变化点**：标准去噪仅优化单模态重建质量，$c$ 为固定条件，去噪过程中 latent 的演化不受其他模态影响，导致跨模态语义漂移。

### 模块 2: Diffusion Latent Aligner（核心创新，对应框架图中心位置）

**直觉**：在去噪的每一步，双模态 latent 应当满足某种"语义一致性"，因此可以构造对齐损失并对其求梯度，用该梯度修正去噪方向。

**本文公式（推导）**:

$$\text{Step 1: 对齐损失定义} \quad \mathcal{L}_{\text{align}}(z_t^{v}, z_t^{a}) = D\left(f_{\text{align}}(z_t^{v}), f_{\text{align}}(z_t^{a})\right)$$
其中加入了跨模态距离度量以解决语义不一致问题；$f_{\text{align}}(\cdot)$ 为将 latent 映射到对齐空间的函数（可能为预训练 CLIP/音频-视觉编码器或学习得到的投影），$D(\cdot, \cdot)$ 为距离函数（如余弦距离或 MSE）。

$$\text{Step 2: 梯度引导去噪} \quad \tilde{z}_{t-1}^{v} = z_{t-1}^{v} - \eta_v \nabla_{z_t^{v}} \mathcal{L}_{\text{align}}, \quad \tilde{z}_{t-1}^{a} = z_{t-1}^{a} - \eta_a \nabla_{z_t^{a}} \mathcal{L}_{\text{align}}$$
重归一化以保证去噪过程不偏离数据流形：学习率 $\eta_v = 0.01$ (AnimateDiff), $\eta_a = 0.1$ (AudioLDM) 控制引导强度。

$$\text{最终: 联合优化目标} \quad z_{t-1}^{*} = \text{arg}\min_{z_{t-1}} \left[ \mathcal{L}_{\text{denoise}}(z_{t-1}|z_t, c) + \lambda \cdot \mathcal{L}_{\text{align}}(z_{t-1}^{v}, z_{t-1}^{a}) \right]$$

**对应消融**：Table 1 及 Figure 6 显示，移除 Latent Aligner（即 Ours-vanilla）后，V2A 任务中 KL 从 2.619 退化至 3.203（+22.3%），且定性对比显示背景噪声显著增加；A2V 任务中 FVD 从 402.385 退化至 417.398（+3.6%），KVD 从 34.764 退化至 36.262（+4.1%）。

### 模块 3: Ours-vanilla 基线构造

**直觉**：验证"简单组合现有工具"不足以解决对齐问题，从而证明 latent aligner 的必要性。

**Baseline 公式 (Ours-vanilla)**:
- V2A: 关键帧提取 → BLIP/CLIP 图像 Caption → AudioLDM 文本转音频
- A2V: 音频 Caption → AnimateDiff 文本转视频  
- Joint-VA: 直接文本 prompt 同时输入 AudioLDM + AnimateDiff

**变化点**：Ours-vanilla 完全依赖文本作为中间桥梁传递语义信息，但文本的抽象性导致视觉-音频细节丢失（如节奏、纹理对应关系）。Latent Aligner 绕过文本瓶颈，直接在连续 latent 空间建立细粒度对应。

**对应消融**：Figure 4 显示 Ours-vanilla 生成的视觉内容与文本对齐较差，音频质量低且与视频错位；本文方法通过 latent 级优化将 AV-alignbind 从 0.074 提升至 0.096（+29.7%），AT-alignbind 从 0.081 提升至 0.138（+70.4%）。

## 实验与分析



本文在 VGGSound、Landscape 和自建 Open-domain 三个基准上评估了四项任务。核心结果如 Table 1 所示：在 **VGGSound Video-to-Audio** 任务上，本文方法 KL 达到 2.619，相比 SpecVQGAN（3.29）降低 20.4%，相比 Ours-vanilla（3.203）降低 18.2%；ISc 达到 5.831，相比 SpecVQGAN（5.108）提升 14.2%。在 **VGGSound Audio-to-Video** 任务上，FVD 为 402.385，相比 TempoToken（1866.285）大幅降低 78.4%；KVD 为 34.764，相比 TempoToken（389.096）降低 91.1%，AV-align 为 0.522，相比 TempoToken（0.423）提升 23.4%。这些数字表明，基于 latent 对齐的推理优化对于跨模态生成质量具有决定性作用，尤其是在 A2V 这种基线方法极弱的任务上优势显著。



然而，部分指标暴露了方法的局限性：**Image-to-Audio** 任务中，本文 KL 2.691 反而差于 Im2Wav（2.612，-3.0%），ISc 6.149 也低于 Im2Wav（7.055，-12.8%），说明对于静态图像到音频的映射，专门的端到端训练方法仍具优势；**V2A FAD** 指标上本文 7.316 也劣于 Ours-vanilla（6.850，-6.8%），暗示 latent 对齐在优化感知对齐的同时可能对音频保真度产生轻微负面影响。**Landscape Joint VA** 的 FVD（1174.856）和 KVD（135.422）与 MM-Diffusion（1141.009 / 135.368）基本持平或略差，仅在 FAD 上取得 16.6% 优势，说明在受限域联合生成场景下，专门训练的无条件联合模型仍有竞争力。



消融实验的核心对比在于 Ours（含 Latent Aligner）vs Ours-vanilla（不含）。如 Figure 6 所示，去掉对齐引导后，V2A 生成音频出现明显的背景噪音和无关声音，而定性视觉对比（Figure 3-5）也验证了跨模态语义一致性的退化。定量上，Ours-vanilla 在 Open-domain Joint VA 的 AV-align 仅为 0.226，本文提升至 0.283（+25.2%），AT-alignbind 从 0.081 跃升至 0.138（+70.4%），证明 **latent 对齐是跨模态一致性的最关键组件**。

**公平性检查**：本文的比较存在明显局限性。(1) 多个任务缺少 2023-2024 年的强基线，如 Make-An-Audio、AudioGen、MusicLM、Imagen Video、Gen-2 等未被纳入；(2) 评估样本量偏小（多数任务仅 3k，Landscape 仅 200），统计可靠性存疑；(3) Open-domain Joint VA 缺乏外部基线，仅与自建的 Ours-vanilla 对比；(4) 优化-based 推理的延迟成本未被报告，30/25 步的去噪配合每步梯度计算显著慢于直接生成；(5) 作者已披露 Image-to-Audio 和 Landscape FVD/KVD 上的劣势，但未深入分析原因。

## 方法谱系与知识库定位

**方法族系**：扩散模型推理时优化 / 跨模态对齐生成

**Parent Method**: AudioLDM + AnimateDiff 组合基线。本文直接利用这两个预训练模型的权重，未进行微调或重训练。

**改变的 slots**：
- **Inference strategy**: 从直接文本条件去噪 → 每步加入梯度-based latent 对齐优化（学习率 0.1/0.01）
- **Architecture**: 新增 Diffusion Latent Aligner 模块作为梯度引导接口，原模型权重冻结
- **Objective**: 从标准噪声预测 → 联合优化去噪损失 + 跨模态对齐损失

**直接基线及差异**：
- **SpecVQGAN**: VQ-GAN 架构，本文用扩散模型 + latent 优化替代，KL 降低 20.4%
- **Im2Wav**: 端到端图像-音频映射，本文方法在 I2A 上反而落后，说明任务特异性方法仍有价值
- **TempoTokens**: 节奏 token 指导，本文 latent 级优化在 A2V 上 FVD 降低 78.4%
- **MM-Diffusion**: 从头训练联合扩散模型，本文免训练方案在开放域更具扩展性但受限域质量持平

**后续方向**：(1) 将 latent aligner 扩展至更多模态组合（如深度、光流）；(2) 学习自适应的对齐损失权重 $\lambda$ 替代手工调参；(3) 开发蒸馏版本将推理时优化压缩为前馈网络，解决速度瓶颈。

**标签**：`modality:video+audio` | `paradigm:diffusion_optimization` | `scenario:open_domain_generation` | `mechanism:latent_alignment_gradient_guidance` | `constraint:no_training_pretrained_reuse`

