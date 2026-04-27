---
title: Improved Baselines with Visual Instruction Tuning
type: paper
paper_level: C
venue: CVPR
year: 2024
paper_link: null
aliases:
- LLaVA-1.5：视觉指令微调的极简增强基线
- LLaVA-1.5
acceptance: Highlight
cited_by: 4923
code_url: https://llava-vl.github.io
method: LLaVA-1.5
---

# Improved Baselines with Visual Instruction Tuning

[Code](https://llava-vl.github.io)

**Topics**: [[T__Visual_Question_Answering]], [[T__Captioning]], [[T__Retrieval]] | **Method**: [[M__LLaVA-1.5]] | **Datasets**: [[D__MMBench]], [[D__MM-Vet]], [[D__SEED-Bench]] (其他: POPE, MME)

| 中文题名 | LLaVA-1.5：视觉指令微调的极简增强基线 |
| 英文题名 | Improved Baselines with Visual Instruction Tuning |
| 会议/期刊 | CVPR 2024 (Highlight) |
| 链接 | [arXiv](https://arxiv.org/abs/2310.03744) · [Code](https://llava-vl.github.io) · [Project](https://llava-vl.github.io) |
| 主要任务 | Visual Instruction Tuning, Visual Question Answering, OCR, Region-level Understanding |
| 主要 baseline | LLaVA, InstructBLIP, BLIP-2, IDEFICS, Qwen-VL, Shikra |

> [!abstract] 因为「视觉-语言指令微调模型在架构设计和数据利用上存在明显效率瓶颈」，作者在「LLaVA」基础上改了「将线性投影替换为MLP+GELU、完全微调视觉编码器、扩展指令微调数据混合」，在「MME/MMBench/MM-Vet/SEED-Bench/POPE」上取得「7B模型全面超越14B-80B竞品的新SOTA」。

- **MME**: 1510.7 分，相比 LLaVA-7B 提升 +701.1，超越 InstructBLIP-14B (+298.7)
- **MMBench**: 64.3%，相比 LLaVA-7B 提升 +25.6，超越 Qwen-VL-Chat (+3.7)
- **SEED-Bench**: 58.6%，相比 LLaVA-7B 提升 +25.1，超越 InstructBLIP-8B (+5.2)

## 背景与动机

当前大型多模态模型（LMMs）的视觉指令微调面临一个核心矛盾：复杂架构与海量预训练数据是否真的必要？以 InstructBLIP 为例，其采用 Q-Former 桥接视觉-语言模态，需要数亿级别的预训练样本；而原始 LLaVA 虽架构极简——仅用单层线性投影连接冻结的 CLIP ViT 与冻结的 Vicuna LLM——却在多项 benchmark 上明显落后于这些"重装备"对手。这引发关键问题：轻量级架构的潜力是否被低估了？

现有方法的处理方式各异。**InstructBLIP** 通过 Q-Former 提取视觉特征并进行指令感知的预训练，依赖大规模数据过滤与多阶段训练。**BLIP-2** 同样使用 Q-Former 桥接冻结编码器，但需要额外的两阶段预训练。**Qwen-VL** 引入位置感知的视觉-语言预训练，架构更为复杂。这些方法虽性能强劲，但共同特征是：预训练数据量庞大、计算成本高、开源复现困难。

然而，它们的短板同样显著：**架构复杂性掩盖了简单设计的潜力**，且对学术级计算资源极不友好。更关键的是，原始 LLaVA 的单层线性投影缺乏非线性变换能力，视觉特征与语言嵌入空间的对齐质量受限；同时其 158K 的 LLaVA-Instruct 数据覆盖任务类型单一，难以支撑广泛的视觉理解需求。此外，固定分辨率输入（224×224 或 336×336）无法处理高分辨率图像中的细粒度信息。

本文的核心动机在于：**验证极简架构配合精心设计的数据混合与训练策略，能否在学术计算预算下达到甚至超越复杂系统的性能**，并进一步扩展至任意分辨率输入。作者从 LLaVA 出发，通过三项关键改进——MLP 投影替代线性层、视觉编码器完全微调、扩展指令微调数据混合——实现了这一目标，并在此基础上提出 LLaVA-1.5-HD 支持高分辨率推理。

## 核心创新

核心洞察：**视觉-语言对齐的质量取决于投影层的表达能力与训练数据的任务覆盖度，而非架构复杂度**，因为单层线性变换无法充分捕捉视觉-语义映射的非线性特性，而受限的数据多样性会制约模型的泛化边界，从而使"极简架构 + 充分训练 + 数据扩展"成为高效且可扩展的范式成为可能。

| 维度 | Baseline (LLaVA) | 本文 (LLaVA-1.5) |
|:---|:---|:---|
| 视觉-语言投影 | 单层线性投影 $W \cdot \text{ViT}(I) + b$ | MLP + GELU 激活 $\text{MLP}(\text{GELU}(\text{ViT}(I)))$ |
| 视觉编码器训练策略 | 冻结 (frozen) | 完全微调 (full fine-tuning) |
| 指令微调数据 | 158K LLaVA-Instruct（单一来源） | 扩展混合：学术 VQA + OCR + 区域理解 + 多样化格式 |
| 输入分辨率 | 固定 224×224 或 336×336 | 336×336 标准训练；LLaVA-1.5-HD 支持任意分辨率 |
| 训练计算 | 两阶段（特征对齐 + 指令微调） | 学术级 8×A100，数据量远小于 InstructBLIP 等竞品 |

## 整体框架


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/15d10b1c-ba4f-44f3-bd6c-3f557c1089cd/figures/fig_001.png)
*Figure: LLaVA-1.5 achieves SoTA on a broad range of 11 tasks (Top), with high training sample efficiency (Left) and simple mod-*



LLaVA-1.5 的整体数据流遵循"编码-投影-生成"的三阶段范式，标准版本与 HD 高分辨率版本共享核心组件：

1. **Vision Encoder (CLIP ViT-L/336 或 EVA-CLIP)**: 输入图像 $I$（标准版 336×336，HD 版任意分辨率），输出视觉特征 token 序列。与 LLaVA 的关键差异在于该模块在指令微调阶段**不再冻结**，而是参与完全微调。

2. **MLP Projection with GELU**: 接收视觉特征 token，通过两层 MLP 中间夹 GELU 非线性激活，将视觉表示投影至 LLM 的嵌入空间。此模块**替换**了 LLaVA 中的单层线性投影，是架构层面的唯一修改。

3. **LLM (Vicuna-v1.5 或 LLaMA-2-Chat)**: 接收投影后的视觉嵌入与文本 token 拼接序列，自回归生成文本响应。该模块保持与 LLaVA 一致的语言模型骨干。

4. **Split-Encode-Merge (HD variant only)**: 针对高分辨率输入，将图像切分为不重叠的 patch 网格，每个 patch 独立编码后合并为完整特征图，突破固定分辨率的限制。

5. **Global Context Branch (HD variant only)**: 将原始图像下采样至 224×224 后编码，获取全局上下文特征，与 Split-Encode-Merge 的局部特征拼接，缓解分块边界伪影并增强全局定位能力。

```
标准 LLaVA-1.5:
Image (336×336) → [Vision Encoder] → visual tokens → [MLP+GELU] → visual embeddings 
                                                                    ↓
Text prompt ─────────────────────────────────────────────────────→ [LLM] → Response

LLaVA-1.5-HD:
High-res Image → Split into patches ──→ [Vision Encoder] × N ──→ Merge ─┐
                                                                         ├──→ [MLP+GELU] → [LLM] → Response
Downsampled (224×224) → [Vision Encoder] ──→ Global context ────────────┘
```

## 核心模块与公式推导

### 模块 1: MLP 视觉投影（对应框架图 视觉编码器 → 投影层）

**直觉**: 视觉特征与语言嵌入空间存在固有的模态鸿沟，单层线性变换的非线性表达能力不足，导致对齐质量受限。

**Baseline 公式** (LLaVA):
$$\text{Visual Features} = W \cdot \text{ViT}(I) + b$$
符号: $W \in \mathbb{R}^{d_{\text{LLM}} \times d_{\text{vis}}}$ = 线性投影矩阵, $b$ = 偏置, $\text{ViT}(I)$ = CLIP 视觉编码器输出的图像特征, $d_{\text{LLM}}$ = 语言模型嵌入维度, $d_{\text{vis}}$ = 视觉特征维度

**变化点**: LLaVA 的线性投影假设视觉-语言映射为简单仿射变换，但实际模态对齐需要更复杂的非线性变换。此外，冻结视觉编码器限制了其适应多模态指令微调的能力。

**本文公式（推导）**:
$$\text{Step 1}: h = \text{GELU}(W_1 \cdot \text{ViT}(I) + b_1) \quad \text{加入 GELU 激活引入非线性，增强表达能力}$$
$$\text{Step 2}: \text{Visual Features} = W_2 \cdot h + b_2 \quad \text{第二层线性变换完成维度映射}$$
$$\text{最终}: \text{Visual Features} = \text{MLP}(\text{GELU}(\text{ViT}(I))) = W_2 \cdot \text{GELU}(W_1 \cdot \text{ViT}(I) + b_1) + b_2$$

**对应消融**: 视觉编码器完全微调的贡献隐含于整体性能提升中；作者指出 Vicuna-v1.5 作为 LLM backbone 在 9 项 benchmark 上综合最优，LLaMA-2 系列普遍优于 LLaMA-1 系列。

### 模块 2: LLaVA-1.5-HD 高分辨率特征融合（对应框架图 HD 分支）

**直觉**: 高分辨率图像包含细粒度细节，但直接编码受限于模型固定输入尺寸；分块编码虽可扩展分辨率，但会丢失全局上下文并引入边界伪影。

**Baseline 公式** (LLaVA 固定分辨率):
$$F = \text{ViT}(\text{Resize}(I, 224 \times 224))$$
符号: $F$ = 视觉特征, $\text{Resize}$ = 双线性插值下采样, 固定分辨率导致信息损失

**变化点**: 固定分辨率无法处理高分辨率输入；简单分块编码（如 ViT 的 naive patchification）缺乏全局语义引导，模型难以定位相关区域。

**本文公式（推导）**:
$$\text{Step 1}: F_{\text{patches}} = \text{Merge}(\{\text{ViT}(I_i)\}_{i=1}^{N}) \quad \text{将图像分为 } N \text{ 个 patch 独立编码后空间合并，突破分辨率限制}$$
$$\text{Step 2}: F_{\text{global}} = \text{ViT}(\text{Downsample}(I, 224 \times 224)) \quad \text{下采样全局分支保留完整场景语义}$$
$$\text{Step 3}: F_{\text{HD}} = [F_{\text{patches}}; F_{\text{global}}] \quad \text{拼接全局上下文，缓解分块边界不连续性与定位模糊}$$
$$\text{最终}: \text{Visual Features}_{\text{HD}} = \text{MLP}(\text{GELU}(F_{\text{HD}}))$$

**对应消融**: 移除全局上下文分支后，GQA accuracy 下降 -0.9，MME score 下降 -71.7，MM-Vet accuracy 下降 -3.2，验证全局上下文对高分辨率理解的关键作用。

### 模块 3: 扩展指令微调数据混合（对应训练流程）

**直觉**: 单一来源的指令数据（LLaVA-Instruct 仅 158K）覆盖的视觉任务类型有限，模型难以泛化到多样化的真实场景查询。

**Baseline**: LLaVA-Instruct 158K，包含对话、详细描述、复杂推理三类指令，但缺乏结构化 VQA、OCR、区域级理解等任务格式。

**变化点**: 引入学术级任务导向数据集补充任务多样性，包括视觉问答（VQA-v2、GQA、OKVQA、A-OKVQA）、OCR（OCR-VQA、TextVQA）、区域理解（Visual7W、RefCOCO 系列）等，并统一为指令跟随格式。

**本文策略**:
$$\mathcal{D}_{\text{mix}} = \mathcal{D}_{\text{LLaVA-Instruct}} \cup \mathcal{D}_{\text{VQA}} \cup \mathcal{D}_{\text{OCR}} \cup \mathcal{D}_{\text{region}} \cup \mathcal{D}_{\text{format}}$$

**对应消融**: 使用 50% 训练数据即可保持 >98% 的完整性能，MMBench、ScienceQA、POPE 无下降；降至 30% 数据性能仍保持稳定，表明数据扩展的收益存在边际递减，现有数据混合已接近充分覆盖。

## 实验与分析



本文在 11 项视觉语言 benchmark 上评估 LLaVA-1.5，核心结果集中于 Table 4。在 **MME** 综合评测中，LLaVA-1.5-7B 取得 1510.7 分，较原始 LLaVA-7B（809.6）提升 +701.1，同时超越 14B 规模的 InstructBLIP（1212.8）达 +298.7，创下新 SOTA。在 **MMBench** 多项选择评测中，64.3% 的准确率超越 Qwen-VL-Chat（60.6%）+3.7，且 LLaVA-7B 基线仅 38.7%，提升幅度达 +25.6。值得注意的是，LLaVA-1.5-7B 甚至超越 80B 参数的 IDEFICS，验证了极简架构配合充分训练的参数效率优势。


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/15d10b1c-ba4f-44f3-bd6c-3f557c1089cd/figures/fig_002.png)
*Figure: LLaVA-1.5-HD. Scaling LLaVA-1.5 to higher resolutions by splitting the image into grids and encoding them independently. This*



高分辨率扩展 **LLaVA-1.5-HD** 在 GQA、MME、MM-Vet 等验证 benchmark 上进一步验证有效性。关键消融显示，全局上下文分支的移除导致系统性性能退化：GQA accuracy 下降 -0.9，MME score 骤降 -71.7，MM-Vet accuracy 下降 -3.2。这一结果直接支持了"分块编码必须配合全局语义引导"的设计假设。



数据效率方面，Figure 4 的消融实验揭示：50% 训练数据即可维持 >98% 的完整性能，MMBench、ScienceQA、POPE 零损失；继续缩减至 30% 数据，性能曲线仍保持平坦。这表明当前数据混合已覆盖主要任务类型，单纯堆叠数据量并非性能瓶颈。

**公平性检验**: 对比基线选择合理，涵盖同期主流开源方法（InstructBLIP、Qwen-VL、Shikra）及大参数闭源/半开源系统（BLIP-2、IDEFICS）。但存在以下局限：GPT-4V 结果引自第三方论文，非直接可比；Table 4 中部分 baseline 在特定 benchmark 上缺失（标记为 --）；LLaVA-1.5-HD 的具体分辨率配置未在 main table 中逐 benchmark 明确。训练成本方面，作者强调使用学术级计算（8×A100），数据量较 InstructBLIP 等"小数个数量级"，但未披露精确训练时长。

## 方法谱系与知识库定位

**方法族**: 视觉指令微调 (Visual Instruction Tuning) → 大型多模态模型 (Large Multimodal Models)

**父方法**: LLaVA ("Visual Instruction Tuning", NeurIPS 2023)
- LLaVA 开创"冻结视觉编码器 + 冻结 LLM + 可训练线性投影"的极简范式
- LLaVA-1.5 继承该范式，修改 slots: **architecture** (线性→MLP+GELU, 视觉编码器解冻)、**data_pipeline** (158K 单源→多任务混合)、**training_recipe** (两阶段→完全微调 + 高分辨率)
- LLaVA-1.5-HD 进一步扩展 **inference_strategy** (固定分辨率→任意分辨率分块编码+全局上下文)

**直接基线差异**:
- **InstructBLIP**: 使用 Q-Former + 大规模预训练，LLaVA-1.5 以 1-2 个数量级更少数据达到更优性能
- **Qwen-VL**: 引入位置编码与 grounding 预训练，LLaVA-1.5 架构更简单、训练更透明
- **IDEFICS/Flamingo**: 80B 参数 + 交错图文预训练，LLaVA-1.5-7B 以 11× 更少参数超越

**后续方向**:
1. **更高分辨率与视频扩展**: LLaVA-1.5-HD 的分块编码策略可向时空维度扩展，支持视频帧序列的高分辨率理解
2. **数据混合的自动化优化**: 当前数据配比为人工设计，可引入基于任务覆盖度的自动数据选择
3. **多模态对齐的理论分析**: MLP+GELU 的增益机制缺乏形式化解释，模态对齐的表征几何值得深入

**知识库标签**: `modality: vision+language` / `paradigm: instruction tuning` / `scenario: academic compute, open-source` / `mechanism: MLP projection, full fine-tuning, split-encode-merge` / `constraint: data efficiency, model simplicity`

