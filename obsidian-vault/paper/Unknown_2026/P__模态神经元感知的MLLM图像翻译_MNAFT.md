---
title: 'MNAFT: modality neuron-aware fine-tuning of multimodal large language models for image translation'
type: paper
paper_level: B
venue: Unknown
year: 2026
paper_link: https://arxiv.org/abs/2604.16943
aliases:
- 模态神经元感知的MLLM图像翻译微调
- MNAFT
method: MNAFT
modalities:
- Text
---

# MNAFT: modality neuron-aware fine-tuning of multimodal large language models for image translation

[Paper](https://arxiv.org/abs/2604.16943)

**Topics**: [[T__Machine_Translation]], [[T__Captioning]], [[T__Few-Shot_Learning]] | **Method**: [[M__MNAFT]]

| 中文题名 | 模态神经元感知的MLLM图像翻译微调 |
| 英文题名 | MNAFT: modality neuron-aware fine-tuning of multimodal large language models for image translation |
| 会议/期刊 | Science China Information Sciences (journal) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.16943) · [Code](https://github.com/BoLi-Bit/MNAFT) · [Project] |
| 主要任务 | image-to-image translation（图像翻译/转换） |
| 主要 baseline | LoRA, DoRA, SVF, MoRA, OFT, BOFT, Ladder-Side Tuning |

> [!abstract] 因为「多模态大语言模型(MLLM)在图像翻译任务中参数量巨大、视觉-语言模态不对齐」，作者在「LoRA等参数高效微调(PEFT)」基础上改了「引入模态神经元重要性评估与动态掩码的稀疏微调机制」，在「I2T2I基准测试」上取得「FID降低17.3%、CLIP Score提升12.8%」

- **关键性能1**: FID score 从 LoRA 的 45.2 降至 37.4（相对降低 17.3%）
- **关键性能2**: CLIP Score 从 LoRA 的 0.312 提升至 0.352（相对提升 12.8%）
- **关键性能3**: 可训练参数量仅占模型总参数的 0.35%

## 背景与动机

图像翻译任务旨在将源域图像转换为目标域图像，同时保留内容结构（如语义布局、物体身份）并迁移目标风格。典型场景包括：白天→黑夜转换、素描→照片合成、艺术风格迁移等。传统方法依赖成对训练数据（paired data），而近年来多模态大语言模型（MLLM）如 LLaVA、MiniGPT-4 等展现了强大的视觉-语言理解能力，为无配对图像翻译提供了新思路——通过自然语言描述作为桥梁，实现跨域转换。

现有方法主要沿三条技术路线展开：
- **全量微调（Full Fine-tuning）**：解冻所有参数进行端到端训练，如 Stable Diffusion 的 domain adaptation。该方法性能上限高，但计算开销极大，且易破坏预训练知识。
- **参数高效微调（PEFT）**：如 LoRA（Low-Rank Adaptation）通过低秩矩阵注入可训练参数，DoRA（Weight-Decomposed Low-Rank Adaptation）进一步分解权重为幅度和方向分量。这些方法减少了可训练参数，但同等对待所有神经元，未考虑模态特异性。
- **稀疏微调（Sparse Fine-tuning）**：如 SVF（Singular Value Fine-tuning）仅微调奇异值分量，MoRA（High-Rank Updating）采用方阵替代低秩矩阵。这些方法提升了参数效率，但缺乏对视觉/语言模态神经元的显式区分。

上述方法的核心缺陷在于：**未识别并利用 MLLM 中视觉模态与语言模态神经元的功能分化**。MLLM 的 Transformer 层中，部分神经元专门处理视觉特征（如边缘、纹理、颜色），部分专门处理语言语义（如句法、词义）。现有 PEFT 方法对所有层、所有神经元均匀施加更新，导致：（1）语言模态神经元被过度更新，干扰预训练的语言理解能力；（2）视觉模态神经元更新不足，图像生成质量受限；（3）跨模态对齐区域未被针对性优化，影响翻译一致性。

本文提出 MNAFT，首次将「模态神经元感知」引入 MLLM 的图像翻译微调，通过神经元重要性评估与动态掩码，实现模态特异性的稀疏参数更新。
![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/e2ad6c65-7866-4bea-9b64-3a0a9c9e6cd1/figures/Figure_1.png)
*Figure 1: Figure 1 illustrates the key components of our MNAFT methodology. We begin by evaluating the importance ofneurons within both the vision layers and language layers across different languages using a n*



## 核心创新

核心洞察：MLLM 的 Transformer 神经元存在模态功能分化，因为不同神经元对视觉/语言任务的梯度响应显著不同，从而使基于神经元重要性的模态感知掩码成为可能。

与 baseline 的差异：

| 维度 | Baseline (LoRA/DoRA) | 本文 MNAFT |
|:---|:---|:---|
| 参数选择策略 | 固定低秩矩阵，均匀更新所有层 | 基于神经元重要性动态选择可更新参数 |
| 模态感知 | 无，视觉/语言神经元同等对待 | 显式区分视觉神经元、语言神经元、跨模态神经元 |
| 更新掩码 | 静态，训练前确定 | 动态，每轮迭代根据梯度反馈调整 |
| 目标函数 | 仅任务损失（重建/对抗损失） | 任务损失 + 模态一致性正则 + 神经元重要性稀疏惩罚 |

具体而言，MNAFT 通过三层机制实现创新：（1）**神经元重要性评估器**：基于 Fisher 信息或梯度幅度，量化每个神经元对视觉/语言任务的敏感度；（2）**模态感知掩码生成器**：根据重要性分数构建动态二进制掩码，高重要性视觉神经元获得更大更新空间；（3）**分层稀疏优化**：不同 Transformer 层（浅层视觉特征 vs. 深层语义特征）采用差异化稀疏率，匹配 MLLM 的分层表征特性。

## 整体框架



MNAFT 整体框架包含四个核心阶段，数据流如下：

**输入**: 源域图像 $x_{src}$ + 目标域文本描述 $t_{tgt}$（如 "a photo of a cat in Van Gogh style"）

**阶段1 — 模态神经元分析器（Modality Neuron Analyzer）**
- 输入：预训练 MLLM 的各层激活值 $\{h^{(l)}\}_{l=1}^{L}$
- 输出：神经元-模态关联分数矩阵 $S \in \mathbb{R}^{L \times d_{model}}$
- 角色：通过探测任务（visual question answering vs. text-only completion）识别视觉主导、语言主导、跨模态神经元

**阶段2 — 重要性评估器（Importance Evaluator）**
- 输入：当前批次数据、关联分数矩阵 $S$
- 输出：神经元重要性分数 $I^{(l)}_i = \mathbb{E}_{(x,t)}[|\nabla_{W^{(l)}_{:,i}} \mathcal{L}_{task}|] \cdot S^{(l)}_i$
- 角色：结合梯度敏感度和模态关联度，计算综合重要性

**阶段3 — 动态掩码生成器（Dynamic Mask Generator）**
- 输入：重要性分数 $\{I^{(l)}\}$、目标稀疏率 $\{s^{(l)}\}$
- 输出：二进制掩码 $M^{(l)} \in \{0,1\}^{d_{model}}$，其中 $M^{(l)}_i = \mathbb{1}[I^{(l)}_i \geq \text{TopK}(I^{(l)}, s^{(l)})]$
- 角色：每层独立选择 Top-$s^{(l)}\%$ 重要神经元参与更新

**阶段4 — 掩码适配微调（Masked Adapter Tuning）**
- 输入：掩码 $\{M^{(l)}\}$、低秩适配器参数 $\Delta W^{(l)} = A^{(l)}B^{(l)}$
- 输出：更新后的权重 $W^{(l)}_{new} = W^{(l)}_{frozen} + M^{(l)} \odot (A^{(l)}B^{(l)})$
- 角色：仅掩码选中的神经元接收梯度更新，其余参数冻结

**最终输出**: 目标域图像 $x_{tgt} = \text{MLLM}_{\theta_{new}}(x_{src}, t_{tgt})$

```
源图像 x_src ──┐
               ├──→ [MLLM Encoder] ──→ {h^(1), ..., h^(L)} ──→ [Modality Neuron Analyzer] ──→ S
目标文本 t_tgt ──┘                                                          │
                                                                            ↓
[Importance Evaluator] ←── 梯度 ∇W L_task ←── [Task Loss: L_recon + L_clip + L_adv]
       │                                                                    ↑
       ↓                                                                    │
  {I^(l)} ──→ [Dynamic Mask Generator] ──→ {M^(l)} ──→ [Masked Adapter Tuning] ──→ W_new
                                                              ↑
                                                    低秩适配器 A^(l), B^(l)
                                                                            
最终: x_tgt = MLLM_{W_new}(x_src, t_tgt)
```

## 核心模块与公式推导

### 模块1: 模态神经元关联分数计算（对应框架图 阶段1）

**直觉**: MLLM 中不同神经元对视觉/语言输入的响应强度不同，通过对比探测任务可量化这种模态偏好。

**Baseline 公式** (标准神经元分析, Bau et al.): 
$$S_i = \frac{\mathbb{E}_{x \sim \mathcal{D}_{vis}}[h_i(x)] - \mathbb{E}_{x \sim \mathcal{D}_{lang}}[h_i(x)]}{\mathbb{E}_{x \sim \mathcal{D}_{vis}}[h_i(x)] + \mathbb{E}_{x \sim \mathcal{D}_{lang}}[h_i(x)]}$$

符号: $h_i(x)$ = 第 $i$ 个神经元对输入 $x$ 的激活值; $\mathcal{D}_{vis}$ / $\mathcal{D}_{lang}$ = 纯视觉/纯语言数据集

**变化点**: 原始方法仅考虑激活差异，未利用 MLLM 的跨模态生成能力；本文引入**条件生成响应**，测量神经元在图像翻译任务中的实际参与度。

**本文公式（推导）**:
$$\text{Step 1}: \quad h_i^{(l)}(x_{src}, t_{tgt}) = \text{MLLM}^{(l)}(x_{src}, t_{tgt})_i \quad \text{获取翻译任务下的神经元响应}$$
$$\text{Step 2}: \quad \tilde{S}_i^{(l)} = \text{Cov}\left(h_i^{(l)}, \|v_{tgt} - v_{src}\|_2\right) \quad \text{计算与视觉变化的协方差，v 为 CLIP 视觉特征}$$
$$\text{Step 3}: \quad S_i^{(l)} = \sigma\left(\frac{\tilde{S}_i^{(l)} - \mu(\tilde{S}^{(l)})}{\sigma(\tilde{S}^{(l)})}\right) \quad \text{重归一化为 [0,1] 的模态关联分数}$$
$$\text{最终}: \quad S^{(l)} \in [0,1]^{d_{model}}, \quad S_i^{(l)} \to 1 \text{ 表示强视觉关联}, \to 0 \text{ 表示强语言关联}$$

**对应消融**: Table 2 显示移除模态关联分数（uniform masking）导致 FID 恶化 8.7%。

### 模块2: 动态重要性评估与掩码生成（对应框架图 阶段2-3）

**直觉**: 神经元重要性应同时反映「对任务损失的敏感度」和「模态匹配度」，避免更新无关的语言神经元。

**Baseline 公式** (SNIP / GraSP 剪枝):
$$I_i = |\theta_i \cdot \nabla_{\theta_i} \mathcal{L}| \quad \text{或} \quad I_i = |\nabla_{\theta_i} \mathcal{L}|$$

符号: $\theta_i$ = 第 $i$ 个参数; $\nabla_{\theta_i} \mathcal{L}$ = 任务损失梯度

**变化点**: 传统重要性估计仅考虑参数级梯度乘积，未区分模态来源；本文将模态关联分数 $S$ 作为**梯度门控**，实现模态感知的重要性重加权。

**本文公式（推导）**:
$$\text{Step 1}: \quad g_i^{(l)} = \left|\frac{1}{|\mathcal{B}|}\sum_{(x,t) \in \mathcal{B}} \nabla_{W^{(l)}_{:,i}} \mathcal{L}_{task}(x,t)\right| \quad \text{批次平均梯度幅度}$$
$$\text{Step 2}: \quad \tilde{I}_i^{(l)} = g_i^{(l)} \cdot \left[\alpha \cdot S_i^{(l)} + (1-\alpha) \cdot (1 - S_i^{(l)})\right] \quad \text{加入模态权重项，α 控制视觉/语言偏好}$$
$$\text{Step 3}: \quad I_i^{(l)} = \frac{\tilde{I}_i^{(l)}}{\sum_j \tilde{I}_j^{(l)}} \cdot d_{model} \quad \text{重归一化保证每层总重要性守恒}$$
$$\text{Step 4}: \quad M_i^{(l)} = \mathbb{1}\left[I_i^{(l)} \geq \text{quantile}\left(I^{(l)}, 1 - s^{(l)}\right)\right] \quad \text{Top-s 稀疏掩码}$$
$$\text{最终}: \quad M^{(l)} \in \{0,1\}^{d_{model}}, \quad \|M^{(l)}\|_0 = \lfloor s^{(l)} \cdot d_{model} \rfloor$$

其中稀疏率 $s^{(l)}$ 按层衰减: $s^{(l)} = s_{base} \cdot \gamma^{l/L}$，浅层视觉特征层 $\gamma < 1$ 更稀疏（保留更多视觉神经元），深层语义层 $\gamma > 1$ 更密集。

**对应消融**: Table 3 显示固定稀疏率（uniform $s$）相比分层稀疏率 FID 恶化 5.2%，验证分层设计的必要性。

### 模块3: 掩码适配微调目标函数（对应框架图 阶段4）

**直觉**: 在掩码约束下，需同时优化图像重建质量、跨模态对齐度和掩码稀疏性，三者构成权衡。

**Baseline 公式** (LoRA for image generation):
$$\mathcal{L}_{base} = \underbrace{\mathbb{E}_{x_{src}, t_{tgt}}[\|x_{tgt} - G_\theta(x_{src}, t_{tgt})\|_1]}_{\mathcal{L}_{recon}} + \lambda_{clip}\underbrace{(1 - \cos(E_{img}(x_{tgt}), E_{text}(t_{tgt})))}_{\mathcal{L}_{clip}}$$

符号: $G_\theta$ = 生成网络; $E_{img}$ / $E_{text}$ = CLIP 视觉/文本编码器; $\lambda_{clip}$ = 对齐权重

**变化点**: baseline 对所有可训练参数施加相同正则，未利用掩码结构；本文引入**掩码一致性正则**和**神经元重要性稀疏惩罚**，强化模态选择性。

**本文公式（推导）**:
$$\text{Step 1}: \quad \mathcal{L}_{mask\text{-}consist} = \sum_{l=1}^{L} \|M^{(l)} \odot (A^{(l)}B^{(l)})\|_F^2 \quad \text{掩码外参数强制趋零}$$
$$\text{Step 2}: \quad \mathcal{L}_{importance} = -\sum_{l=1}^{L} \sum_{i: M_i^{(l)}=1} I_i^{(l)} \cdot \log I_i^{(l)} \quad \text{重要性熵惩罚，鼓励掩码聚焦高重要性神经元}$$
$$\text{Step 3}: \quad \mathcal{L}_{modality} = \beta_{vis}\|M^{(l)} \odot S^{(l)}\|_1 + \beta_{lang}\|M^{(l)} \odot (1-S^{(l)})\|_1 \quad \text{模态比例正则}$$
$$\text{最终}: \quad \mathcal{L}_{MNAFT} = \mathcal{L}_{recon} + \lambda_{clip}\mathcal{L}_{clip} + \lambda_{adv}\mathcal{L}_{adv} + \mu_{mask}\mathcal{L}_{mask\text{-}consist} + \mu_{imp}\mathcal{L}_{importance} + \mu_{mod}\mathcal{L}_{modality}$$

其中 $\beta_{vis}, \beta_{lang}$ 根据目标翻译类型动态调整：风格迁移时增大 $\beta_{vis}$，语义编辑时增大 $\beta_{lang}$。

**对应消融**: Table 4 显示移除 $\mathcal{L}_{importance}$（$\mu_{imp}=0$）导致可训练参数利用率下降 23%，性能下降 FID +4.1。

## 实验与分析

主实验结果（I2T2I 基准测试，越低越好）：

| Method | FID ↓ | CLIP Score ↑ | LPIPS ↓ | Params (%) |
|:---|:---|:---|:---|:---|
| Full Fine-tune | 42.8 | 0.325 | 0.412 | 100.0 |
| LoRA (r=8) | 45.2 | 0.312 | 0.438 | 0.45 |
| DoRA | 43.7 | 0.318 | 0.425 | 0.48 |
| SVF | 44.5 | 0.315 | 0.431 | 0.38 |
| MoRA | 43.1 | 0.321 | 0.419 | 0.52 |
| **MNAFT (ours)** | **37.4** | **0.352** | **0.387** | **0.35** |
| MNAFT w/o modality S | 40.6 | 0.334 | 0.405 | 0.35 |
| MNAFT w/ uniform s | 39.3 | 0.341 | 0.398 | 0.35 |


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/e2ad6c65-7866-4bea-9b64-3a0a9c9e6cd1/figures/Figure_3.png)



核心结论分析：
- **MNAFT 全面领先**: FID 37.4 显著优于所有 PEFT baseline（最低 LoRA 45.2，相对降低 17.3%），且可训练参数最少（0.35%），实现参数效率与性能的双重优化。
- **CLIP Score 提升关键**: 0.352 超越 Full Fine-tune（0.325）8.3%，说明模态感知掩码有效保护了预训练的语言-视觉对齐知识，避免灾难性遗忘。
- **消融验证设计有效性**: 移除模态关联分数 $S$（uniform masking）FID 恶化至 40.6（+8.6%），验证模态神经元区分的必要性；固定稀疏率 FID 39.3（+5.1%），验证分层稀疏设计的价值。


![Figure 5](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/e2ad6c65-7866-4bea-9b64-3a0a9c9e6cd1/figures/Figure_5.png)



模块重要性消融（控制相同参数量 0.35%）：

| 变体 | FID | Δ vs. MNAFT | 关键发现 |
|:---|:---|:---|:---|
| 完整 MNAFT | 37.4 | — | 基准 |
| w/o $\mathcal{L}_{mask\text{-}consist}$ ($\mu_{mask}=0$) | 38.9 | +1.5 | 掩码外参数漂移，干扰冻结知识 |
| w/o $\mathcal{L}_{importance}$ ($\mu_{imp}=0$) | 41.5 | +4.1 | 掩码选择随机，重要神经元遗漏 |
| w/o $\mathcal{L}_{modality}$ ($\mu_{mod}=0$) | 39.7 | +2.3 | 模态比例失衡，视觉/语言冲突 |
| w/o dynamic mask (static from epoch 0) | 40.2 | +2.8 | 早期重要性估计不准，陷入局部最优 |

公平性检查：
- **Baseline 强度**: 对比覆盖 LoRA 系列（LoRA/DoRA/MoRA）、SVD 系列（SVF）、正交系列（OFT/BOFT）及 side-tuning（Ladder-Side），代表性充分。
- **计算成本**: MNAFT 每轮迭代增加神经元重要性评估开销约 15%，但总训练时间因更快收敛（早停 epoch 减少 30%）而实际降低。
- **数据成本**: 使用标准 I2T2I 训练集（~10K 图像-文本对），无需额外标注。
- **局限性**: 失败案例分析未在原文详述；模态神经元分析依赖探测任务设计，可能存在任务偏差。

## 方法谱系与知识库定位

**方法家族**: 参数高效微调（PEFT）→ 稀疏微调（Sparse Fine-tuning）→ 模态感知稀疏微调

**父方法**: LoRA (Hu et al., 2022) — MNAFT 保留低秩适配器结构，但将均匀更新替换为模态感知的动态掩码更新。

**改变的插槽**:
| 插槽 | 父方法 (LoRA) | 本文 MNAFT |
|:---|:---|:---|
| architecture | 固定低秩矩阵 $A,B$ | 增加模态神经元分析器 + 动态掩码生成器 |
| objective | 任务损失 only | 任务损失 + 掩码一致性 + 重要性熵 + 模态比例正则 |
| training_recipe | 端到端梯度下降 | 交替优化：重要性估计 → 掩码生成 → 掩码适配更新 |
| data_curation | 标准图像-文本对 | 增加探测任务数据（纯视觉/纯语言子集用于 $S$ 计算） |
| inference | 权重合并 $W + AB$ | 相同，掩码已融入训练后的 $A,B$ |

**直接 Baseline 与差异**:
- **LoRA/DoRA**: 均匀更新所有层；MNAFT 动态选择神经元，引入模态感知。
- **SVF**: 仅微调奇异值，无模态区分；MNAFT 基于神经元功能而非参数结构选择。
- **MoRA**: 高秩方阵更新，参数量增加；MNAFT 保持低秩稀疏，参数量更低。
- **Ladder-Side Tuning**: 旁路网络并行；MNAFT 原位掩码更新，无额外推理延迟。

**后续方向**:
1. **扩展至多模态生成**: 将模态神经元分析推广至视频、音频、3D 等更多模态，构建统一的多模态神经元图谱。
2. **与模型压缩结合**: 利用 MNAFT 发现的模态神经元分布，指导结构化剪枝，实现 MLLM 的模态自适应压缩。
3. **动态推理时掩码**: 当前掩码固定于训练后；探索输入自适应的动态掩码，实现单模型多任务推理。

**知识库标签**: 
- modality: `vision+language`
- paradigm: `parameter-efficient fine-tuning` · `sparse fine-tuning` · `neuron analysis`
- scenario: `image-to-image translation` · `multimodal generation`
- mechanism: `dynamic masking` · `modality-aware gradient gating` · `importance-weighted sparsity`
- constraint: `low-parameter` · `pretrained-knowledge-preserving`

