---
title: 'Foley-Flow: Coordinated Video-to-Audio Generation with Masked Audio-Visual Alignment and Dynamic Conditional Flows'
type: paper
paper_level: C
venue: CVPR
year: 2025
paper_link: null
aliases:
- 视频到音频生成的动态条件流匹配
- Foley-Flow
acceptance: poster
cited_by: 2
method: Foley-Flow
---

# Foley-Flow: Coordinated Video-to-Audio Generation with Masked Audio-Visual Alignment and Dynamic Conditional Flows

**Topics**: [[T__Video_Generation]], [[T__Audio_Generation]], [[T__Cross-Modal_Matching]] | **Method**: [[M__Foley-Flow]] | **Datasets**: [[D__VGGSound]]

| 中文题名 | 视频到音频生成的动态条件流匹配 |
| 英文题名 | Foley-Flow: Coordinated Video-to-Audio Generation with Masked Audio-Visual Alignment and Dynamic Conditional Flows |
| 会议/期刊 | CVPR 2025 (poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2603.08126) · [Code] · [Project] |
| 主要任务 | Video-to-Audio Generation (视频到音频生成) |
| 主要 baseline | SpecVQGAN, Im2Wav, Diff-Foley, FoleyGen, V2A-Mapper, Seeing & Hearing, MaskVAT, VAB, VATT |

> [!abstract] 因为「现有视频到音频生成方法依赖扩散模型或自回归模型，存在采样效率低、音视频时序对齐差的问题」，作者在「Diffusion Mamba with Bidirectional SSMs」基础上改了「引入 Video-Audio Masking Alignment (VAMA) 跨模态掩码对齐模块和 Generalized Video-Audio Flow (GVAF) 动态条件流生成机制」，在「VGGSound test set」上取得「KLD 0.97 / FAD 0.52 / Align Acc 98.97%，相比最优 baseline VATT 的 KLD 2.25 降低 56.9%」

- **KLD ↓**: 0.97，相比 VATT (2.25) 相对提升 56.9%，相比 Diff-Foley (3.15) 相对提升 69.2%
- **FAD ↓**: 0.52，相比 V2A-Mapper (0.99) 相对提升 47.5%，相比 MaskVAT (1.51) 相对提升 65.6%
- **Align Acc ↑**: 98.97%，相比 VATT (82.81%) 相对提升 19.6%

## 背景与动机

视频到音频生成（Video-to-Audio Generation）旨在为无声视频生成语义一致、时序同步的音频，例如为敲击键盘的视频配上清脆的敲击声，或为海浪视频配上波涛声。这一任务的核心挑战在于：音频不仅要与视频内容在语义上匹配（如"看到狗就听到狗叫"），还要在精细的时间尺度上与视觉动态对齐（如"爪子落地瞬间恰好出现脚步声"）。

现有方法主要从三个方向切入。**Diff-Foley** (NeurIPS 2023) 采用对比预训练对齐音视频表征，再用潜在扩散模型生成音频，但扩散模型的多步采样导致推理效率低下。**MaskVAT** (ECCV 2024) 和 **VAB** (ICML 2024) 引入掩码生成建模，通过掩码音频 token 预测实现高效生成，然而其掩码策略局限于单模态重建，缺乏显式的跨模态对齐机制。**VATT** (NeurIPS 2024) 利用大语言模型将视频特征映射到音频 token 空间，但自回归解码的并行性受限，且 LLM 的时序敏感性不足。

这些方法的共同短板在于：**音视频对齐机制与生成机制解耦**。扩散模型和自回归模型专注于音频分布建模，却将"何时发出什么声音"的对齐问题留给隐式学习；掩码模型虽能预训练对齐，但掩码重建仅利用同模态上下文，未强制视频特征参与音频语义恢复。结果是，现有方法在 Align Acc（音视频对齐准确率）上普遍低于 83%，且 FAD（Frechet Audio Distance）显示生成音频质量与真实音频仍有显著差距。

本文的核心思路是：**将跨模态对齐显式嵌入训练目标，并用动态条件流统一高效生成与时序自适应对齐**。

## 核心创新

核心洞察：**流匹配（Flow Matching）的可逆变换特性天然适合条件生成，但标准流匹配缺乏时变跨模态条件注入能力；通过在训练阶段用视频特征辅助掩码音频重建，可将音视频对齐从"隐式希望"转化为"显式约束"，从而使动态视频条件流在推理阶段生成高度同步的音频成为可能。**

| 维度 | Baseline (Diff-Foley / MaskVAT / VATT) | 本文 (Foley-Flow) |
|:---|:---|:---|
| **生成范式** | 扩散模型多步去噪 / 掩码 token 预测 / LLM 自回归解码 | 动态条件流匹配 (GVAF) + 双向 SSM (Mamba)，单步可逆采样 |
| **对齐机制** | 对比预训练隐式对齐 / 无显式跨模态掩码 / 指令微调隐式映射 | VAMA：显式掩码音频特征，强制用视频+未掩码音频联合重建 |
| **条件注入** | 静态全局视频嵌入 / 无动态时序条件 | 时变视频特征 $\mathbf{F}^v_t$ 逐帧注入流变换，实现帧级同步 |
| **训练目标** | 扩散 ELBO / 掩码重建 MSE / 语言建模交叉熵 | $\mathcal{L}_{\text{VAMA}} + \mathcal{L}_{\text{GVAF}}$ 联合优化对齐与生成 |

## 整体框架


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/342bcbfe-3f41-4d99-b683-635816dd3325/figures/Figure_2.png)
*Figure 2 (pipeline): Illustration of the proposed Foley-Flow framework.*



Foley-Flow 框架包含两条主线：**训练阶段**的 VAMA 跨模态对齐预训练，与**生成阶段**的 GVAF 动态条件流推理。数据流如下：

1. **Video Encoder (EVA-CLIP)**：输入 224×224 视频帧，输出视觉特征 $\mathbf{F}^v$。采用 LAION-400M 预训练权重初始化，提取时空一致的视觉表征。

2. **Audio Encoder (AudioMAE)**：输入 8kHz 采样、10 秒片段的对数频谱图 (128×128)，输出音频特征 $\mathbf{F}^a$。基于 MAE 预训练，编码音频的时频结构。

3. **VAMA Module**：输入被掩码的音频特征 $\mathbf{F}^a_{\text{mask}}$、未掩码音频特征 $\mathbf{F}^a_{\text{unmask}}$ 及视频特征 $\mathbf{F}^v$，输出重建的掩码音频特征 $\hat{\mathbf{F}}^a_{\text{mask}}$ 和对齐损失 $\mathcal{L}_{\text{VAMA}}$。这是训练时专属模块，强制模型学习"视频内容如何补全缺失音频"。

4. **GVAF / Dynamic Conditional Flow**：输入视频潜变量 $\mathbf{z}_v$ 和噪声 $\mathbf{z}_{\text{noise}}$，通过双向 SSM 骨干网络学习时变条件流 $\mathbf{F}^a_t = f_{\boldsymbol{\phi}}(\mathbf{z}_t, \mathbf{F}^v_t)$，最终通过逆变换 $\mathbf{z}_a = f_{\boldsymbol{\phi}}^{-1}(\mathbf{z}_v, \mathbf{z}_{\text{noise}})$ 采样音频潜变量。

5. **Audio Decoder**：将生成的音频潜变量 $\mathbf{z}_a$ 解码为最终音频波形/频谱图。

```
训练流: 视频帧 → EVA-CLIP → F^v ─┐
                                  ├──→ VAMA → L_VAMA
音频频谱 → AudioMAE → F^a ────────┘      (掩码重建)

生成流: 视频帧 → EVA-CLIP → z_v ──┐
                                  ├──→ GVAF → z_a → Audio Decoder → 音频
高斯噪声 ───────────────────────────┘      (逆变换采样)
```

## 核心模块与公式推导

### 模块 1: Video-Audio Masking Alignment (VAMA)（对应框架图：训练分支左侧）

**直觉**: 标准掩码自编码器只问"根据未掩码的音频，能猜出被掩码的部分吗"，但视频到音频生成需要回答"看到这段视频，应该知道被掩码处该有什么声音"，因此必须将视频特征拉入重建条件。

**Baseline 公式** (Masked Autoencoder):
$$\mathcal{L}_{\text{recon}} = \|\mathbf{X}_{\text{mask}} - \hat{\mathbf{X}}_{\text{mask}}(\mathbf{X}_{\text{unmask}})\|^2$$
符号: $\mathbf{X}_{\text{mask}}$ = 被掩码的输入特征, $\mathbf{X}_{\text{unmask}}$ = 未掩码部分作为条件, $\hat{\mathbf{X}}_{\text{mask}}$ = 重建输出。

**变化点**: 标准重建仅利用同模态上下文；对于视频到音频任务，模型可能学会"音频内部连贯性"却忽略"音视频对应关系"，导致生成音频自洽但与视频脱节。

**本文公式（推导）**:
$$\text{Step 1}: \quad \hat{\mathbf{F}}^a_{\text{mask}} = g_{\boldsymbol{\psi}}(\mathbf{F}^v, \mathbf{F}^a_{\text{unmask}}) \quad \text{（将视频特征 } \mathbf{F}^v \text{ 与未掩码音频特征共同输入解码器）}$$
$$\text{Step 2}: \quad \mathcal{L}_{\text{VAMA}} = \|\mathbf{F}^a_{\text{mask}} - \hat{\mathbf{F}}^a_{\text{mask}}(\mathbf{F}^v, \mathbf{F}^a_{\text{unmask}})\|^2 \quad \text{（重建目标不变，但条件扩展为跨模态）}$$
**最终**: $\mathcal{L}_{\text{VAMA}}$ 强制模型建立"视频内容 → 音频内容"的映射，掩码比例通过 Table 5 消融确定为最优值。

**对应消融**: Table 3 显示，移除 VAMA 后 Align Acc 从 98.97% 降至 93.86%，下降 5.11 个百分点；KLD 从 0.97 升至 1.92，验证跨模态掩码对齐的必要性。

---

### 模块 2: Generalized Video-Audio Flow (GVAF)（对应框架图：生成分支右侧）

**直觉**: 扩散模型通过迭代去噪生成，步数多、速度慢；标准流匹配虽可单步采样，但使用静态全局条件，无法适应视频中"这一刻 vs 下一刻"声音变化的精细时序结构。

**Baseline 公式** (Standard Flow Matching):
$$\mathcal{L}_{\text{flow}} = D_{\text{KL}}(p(\mathbf{x}) \| p_{\text{target}}(\mathbf{x}))$$
$$\mathbf{x} = f_{\boldsymbol{\phi}}(\mathbf{z}) \quad \text{（静态映射，无显式条件）}$$
符号: $p(\mathbf{x})$ = 模型分布, $p_{\text{target}}$ = 目标数据分布, $f_{\boldsymbol{\phi}}$ = 可逆流变换, $\mathbf{z}$ = 潜变量。

**变化点**: 标准流匹配假设数据分布单一且条件固定；视频到音频生成需要**时变条件**——第 $t$ 帧的视频内容决定第 $t$ 时刻的音频特征，全局视频嵌入会模糊这种帧级对应关系。

**本文公式（推导）**:
$$\text{Step 1}: \quad \mathbf{F}^a_t = f_{\boldsymbol{\phi}}(\mathbf{z}_t, \mathbf{F}^v_t) \quad \text{（引入时间索引 } t \text{ 和时变视频条件 } \mathbf{F}^v_t\text{，实现帧级动态条件）}$$
$$\text{Step 2}: \quad \mathcal{L}_{\text{GVAF}} = D_{\text{KL}}(p(\mathbf{z}_a) \| p_{\text{target}}(\mathbf{z}_a)) \quad \text{（将流损失特化到音频潜空间，与动态条件流的输出一致）}$$
$$\text{Step 3}: \quad \mathbf{z}_a = f_{\boldsymbol{\phi}}^{-1}(\mathbf{z}_v, \mathbf{z}_{\text{noise}}) \quad \text{（推理时利用可逆性，从视频条件+噪声直接逆变换采样）}$$
**最终**: GVAF 以双向 SSM (Mamba) 为骨干，在每一步流变换中注入当前时刻视频特征，既保持流模型的高效采样（单步/少步），又实现扩散模型难以达到的精细时序自适应。

**对应消融**: Table 4 显示，将 GVAF 替换为静态条件流后，FAD 从 0.52 升至 1.57，Align Acc 从 98.97% 降至 96.29%；同时移除 VAMA 和 GVAF 则性能崩塌至 KLD 3.15 / FAD 6.40 / Align Acc 82.47%，接近 Diff-Foley 水平。

---

### 模块 3: 联合训练目标（对应框架图：整体优化）

**直觉**: VAMA 和 GVAF 分别解决"对齐"和"生成"两个子问题，但需联合优化以避免各自为政。

**本文公式**:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{VAMA}} + \lambda \cdot \mathcal{L}_{\text{GVAF}}$$
其中 $\lambda$ 为平衡系数。VAMA 在训练早期主导，建立可靠的跨模态表征；GVAF 在此基础上学习条件流分布，两者共享 EVA-CLIP 和 AudioMAE 编码器，实现端到端联合训练。

**对应消融**: Table 2 显示，仅保留 GVAF 而移除 VAMA 时，Align Acc 下降 5.11%；仅保留 VAMA 而移除 GVAF（回退到标准流）时，FAD 上升 1.05，证明两者互补缺一不可。

## 实验与分析


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/342bcbfe-3f41-4d99-b683-635816dd3325/figures/Table_1.png)
*Table 1 (comparison): Comparison results on V2A-Bench and VGGSound test set.*



本文在 **VGGSound test set** 上进行主实验评估，采用三个核心指标：**KLD** (Kullback-Leibler Divergence，衡量分布匹配度，越低越好)、**FAD** (Frechet Audio Distance，衡量音频质量，越低越好)、**Align Acc** (音视频对齐准确率，越高越好)。如 Table 1 所示，Foley-Flow 在这三项指标上均取得全面 SOTA：KLD 达到 **0.97**，相比此前最优的 VATT (2.25) 相对降低 **56.9%**，甚至低于所有 baseline 的 FAD 值；FAD 达到 **0.52**，相比 V2A-Mapper (0.99) 相对降低 **47.5%**，首次在该 benchmark 上突破 1.0 大关；Align Acc 达到 **98.97%**，相比 VATT (82.81%) 相对提升 **19.6%**，接近完美对齐。这一结果表明，VAMA 的显式跨模态对齐训练与 GVAF 的动态时序条件生成形成了有效协同——不仅生成音频质量高，且与视频的时序同步性显著优于依赖隐式对齐的扩散模型和自回归模型。


![Table 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/342bcbfe-3f41-4d99-b683-635816dd3325/figures/Table_2.png)
*Table 2 (ablation): Ablation studies on Video-Audio attention for video-to-audio generation.*



消融实验进一步验证各组件的贡献。Table 2 和 Table 3 显示，**移除 VAMA** 导致 Align Acc 从 98.97% **暴跌至 93.86%** (-5.11%)，KLD 从 0.97 **飙升至 1.92** (+97.9%)，证明跨模态掩码对齐是时序同步的关键；**移除 GVAF**（替换为静态条件流）导致 FAD 从 0.52 **升至 1.57** (+201.9%)，Align Acc 降至 96.29% (-2.68%)，验证动态时序条件的必要性。Table 4 探索了流目标函数的具体设计，Table 5 确定了最优掩码比例。

编码器选择方面，Table 3 对比了不同视觉-音频编码器组合：EVA-CLIP + AudioMAE (本文选择) 取得 KLD 0.97 / FAD 0.52 / Align Acc 98.97 的全优结果；将 AudioMAE 替换为 ImageBind 后，KLD 升至 1.01，FAD 升至 0.73，Align Acc 降至 98.52；将 EVA-CLIP 替换为 ImageBind 后性能下降更为显著。这表明**专用音频编码器 AudioMAE 比通用跨模态编码器 ImageBind 更适合音频细节建模**。

公平性检查：本文比较的 9 个 baseline 覆盖了 V2A 领域的主要范式（GAN、扩散、自回归、掩码生成、LLM），且包含 CVPR/NeurIPS/ICML/ECCV 等顶会最新工作，baseline 选择较为全面。但存在以下局限：(1) **未报告推理速度对比**——尽管流模型理论上支持单步采样，实际 latency 与扩散模型的步数对比缺失；(2) **缺乏人类主观评测**——FAD 等自动指标可能无法完全反映感知质量；(3) **仅在 VGGSound 单一数据集验证**，跨数据集泛化能力未知；(4) 训练配置为 100 epochs、batch size 128、lr 1e-4，但模型参数量和 GPU 类型未披露，计算成本难以评估。

## 方法谱系与知识库定位

**方法家族**: Flow Matching + State Space Models (Mamba) for Cross-Modal Generation

**父方法**: **Diffusion Mamba with Bidirectional SSMs** (高效图像/视频生成) —— Foley-Flow 继承其双向 SSM 骨干架构，但将应用域从单模态图像生成扩展至跨模态视频到音频生成，核心差异在于引入时变视频条件注入和跨模态对齐机制。

**直接 Baseline 差异**:
- **Diff-Foley** (NeurIPS 2023): 扩散模型 + 对比预训练 → Foley-Flow 替换为流匹配 + 显式 VAMA 对齐，KLD 3.15→0.97
- **MaskVAT** (ECCV 2024): 掩码生成 + 音频 codec → Foley-Flow 保留掩码思想但扩展为跨模态 VAMA，生成端改用流而非掩码预测
- **VATT** (NeurIPS 2024): LLM 自回归并行解码 → Foley-Flow 放弃自回归，以流模型的可逆变换实现更高效采样
- **VAB** (ICML 2024): 视觉条件掩码音频 token 预测 → Foley-Flow 将掩码对齐与生成解耦为 VAMA+GVAF 两阶段，避免 VAB 无扩散模型的生成能力受限

**后续方向**:
1. **多模态扩展**: 将 VAMA 的动态条件掩码对齐推广至视频-音乐、视频-语音等更细粒度模态对
2. **实时推理优化**: 结合 Mamba 的线性复杂度与流模型的单步采样，探索实时视频配音 (real-time foley)
3. **跨数据集验证与 human evaluation**: 在 AudioSet、FSD50K 等数据集验证泛化性，补充主观 MOS 评测

**标签**: 模态(video+audio) / 范式(flow matching, masked modeling) / 场景(conditional generation, cross-modal alignment) / 机制(bidirectional SSM, dynamic conditioning, invertible transform) / 约束(efficient sampling, temporal synchronization)

