---
title: "World Model on Million-Length Video And Language With RingAttention"
venue: ICLR
year: 2025
tags:
  - Multimodal_LLM
  - task/video-understanding
  - task/visual-question-answering
  - ring-attention
  - progressive-context-extension
  - vq-vae
  - dataset/Books3
  - dataset/LOFT
  - dataset/Video-MME
  - opensource/full
core_operator: 用 Blockwise RingAttention 做精确长序列注意力，并结合渐进式扩窗、合成长上下文 QA 与 masked sequence packing，把文本与离散视觉 token 统一扩展到 1M 上下文自回归建模。
primary_logic: |
  长文档/长视频/图文交错序列输入 → 以 Blockwise RingAttention + RoPE θ 扩展做 32K→1M 渐进式训练，并用合成 QA 强化远距检索、用 masked sequence packing 稳定多模态混训 → 输出可处理 1M token 的文本/视频理解与生成统一模型
claims:
  - "在 1M multi-needle 检索中，LWM-Text-1M 在 N=4, R=1 设置下达到 0.84，而 Llama-3.1-8B/Qwen2.5-7B/Mistral-7B 仅为 0.32/0.00/0.13 [evidence: comparison]"
  - "Masked sequence packing 相比 naive packing 在 VQAv2/SQA/POPE 上从 48.3/34.8/62.5 提升到 55.8/47.7/75.2，说明它对多模态混合训练稳定性至关重要 [evidence: ablation]"
  - "在 Video-MME 长视频子集上，LWM-1M 以 7B 参数和最多 1800 帧输入达到 60.8，在 7B 级开源长视频模型中明显领先 [evidence: comparison]"
related_work_position:
  extends: "Ring Attention with Blockwise Transformers (Liu et al. 2024)"
  competes_with: "Gemini 1.5 Pro; GPT-4o"
  complementary_to: "CLIP (Radford et al. 2021); Striped Attention (Brandon et al. 2023)"
evidence_strength: strong
pdf_ref: paperPDFs/Building_World_Models_from_Language_Priors/arXiv_2024/2024_World_Model_on_Million_Length_Video_And_Language_With_RingAttention.pdf
category: Multimodal_LLM
---

# World Model on Million-Length Video And Language With RingAttention

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2402.08268), [Project](https://largeworldmodel.github.io/lwm/)
> - **Summary**: 这篇工作给出了一套开源可复现的 1M 上下文训练 recipe：用 Blockwise RingAttention 做精确长注意力，再用渐进式扩窗、书籍合成长 QA 和 masked packing，让 7B 模型真正能处理长文档与小时级视频。
> - **Key Performance**: 1M multi-needle（N=4, R=1）准确率 0.84；Video-MME Long（30–60 min）60.8（7B，≤1800 frames）

> [!info] **Agent Summary**
> - **task_path**: 长文本/长视频/图文交错序列 -> 文本答案 / 图像 / 视频自回归 token 输出
> - **bottleneck**: 1M 级精确注意力的算存成本、缺少“教会模型使用长上下文”的监督，以及多模态混合长度训练导致短答案损失被稀释
> - **mechanism_delta**: 把标准 Llama2-7B 扩展成“Blockwise RingAttention + 渐进式上下文课程 + 合成长 QA + masked packing”的统一多模态 AR 模型
> - **evidence_signal**: 1M needle/LOFT/Video-MME 的跨基线比较，加上 masked packing 消融
> - **reusable_ops**: [blockwise-ringattention, masked-sequence-packing]
> - **failure_modes**: [多针检索与跨片段综合时性能下降, OCR与细粒度视觉理解弱于CLIP式VLM]
> - **open_questions**: [连续视觉嵌入能否与1M上下文统一结合, 该训练配方扩到更大参数规模时是否仍稳定]

## Part I：问题与挑战

这篇 paper（最初 arXiv 2024，后为 ICLR 2025）真正解决的，不是“把 context window 标成 1M”，而是**让一个标准 Transformer 在训练和推理时都真正利用百万级文本/视频上下文**。

### 1. 真问题是什么
对长文档，问题是远距离检索与多文档整合；对长视频，问题更尖锐：如果模型只能抽 8～64 帧，它看到的是摘要，不是完整时序证据。很多错误并非“不会推理”，而是**根本没看到关键帧**。

### 2. 真 bottleneck 在哪里
- **算力/显存瓶颈**：标准注意力在 1M token 下几乎不可训练。
- **监督瓶颈**：即便把长书或长视频喂进去，模型也未必学会“去远处找答案”；纯 next-token 预训练对“远距定位”监督太弱。
- **混模态训练瓶颈**：文本、图像、视频长度差异极大，naive packing 会把短文本答案的 loss 权重稀释掉，最终伤害 VQA/聊天能力。

### 3. 输入/输出接口与边界
- **输入**：BPE 文本 token + VQGAN 离散视觉 token（每张 256×256 图像编码为 256 token；视频逐帧编码）。
- **输出**：统一 next-token prediction，所以同一模型可做 text-only、image/video understanding、text-to-image、text-to-video。
- **边界条件**：这里的 “world model” 更接近**长上下文多模态自回归模型**，不是带动作条件的环境动力学模型。

### 4. 为什么现在值得做
- 长时程 agent / world model 需要处理分钟到小时级观测；
- Gemini 1.5 说明 1M context 有现实价值，但开源路线缺失；
- 因而这篇工作的贡献不只是一个模块，而是一整套**从训练算子到数据构造再到混训稳定性**的可执行 recipe。

## Part II：方法与洞察

作者引入的关键 causal knob，不是某个新 decoder block，而是**让标准 AR Transformer 真正学会并稳定承载 1M 上下文**的一套训练系统。

### 方法主线

1. **Stage I：先把语言模型扩到 1M**
   - 从 LLaMA-2 7B 初始化。
   - 用 Books3 的长文档，按 `32K → 128K → 256K → 512K → 1M` 渐进训练。
   - 用 **Blockwise RingAttention** 做精确长注意力，并同步放大 RoPE 的 θ，避免直接外推造成位置失真。

2. **不是只“读长文”，而是教模型“会用长文”**
   - 把 Books3 切成 1000-token chunk。
   - 用短上下文模型先为每个 chunk 生成 QA。
   - 再把相邻 chunk 拼成长序列，并把 QA 放在序列末尾的 chat 格式中。
   - 这样相关信息可能出现在上下文任何位置，模型被迫学习远距定位，而非只做局部续写。

3. **Stage II：扩成统一视频-语言模型**
   - 从 LWM-Text-1M 初始化。
   - 用 VQGAN 把图像/视频离散化，加入 `<vision>`, `</vision>`, `<eof>`, `<eov>` 等边界 token。
   - 训练顺序是 `1K 图文 → 8K 文本视频 → 32K/128K/1M chat 多模态`，逐步提高复杂度。

4. **多模态混训的稳定器：masked sequence packing**
   - packed 后，每个样本只能 attend 自己，避免跨样本污染。
   - 同时重加权 loss，使 packed 训练更接近非 packed + padding 的真实优化。
   - 早期还混入纯文本 batch，以减轻视觉训练对语言能力的冲击。

### 核心直觉

- **把“能塞下 1M”变成“真在 1M 上训练过”**  
  Blockwise RingAttention 改变的是计算约束：从单卡显存上限，转成可并行的 blockwise ring 通信问题。结果不是近似长上下文，而是**精确 pairwise attention** 的长上下文训练，这对检索型任务很关键。

- **把“长序列预训练”变成“远距证据监督”**  
  合成 QA 改变的是监督分布。普通 LM 看到长书，不一定学会在问答时去远处找证据；QA 格式则显式要求模型从超长上下文中提取答案，因此长上下文能力从“被动记忆”变成“主动检索”。

- **把“一步到位 1M”变成“课程式扩窗”**  
  渐进式 context extension 改变的是优化难度。模型先学近程依赖，再逐步扩展到远程依赖，训练更稳定，也更省算力。

- **把“pack 了就行”变成“控制 packed 训练的统计偏差”**  
  masked packing 改变的是 loss 分布。没有它时，长视觉上下文会稀释短文本答案 token 的监督；有了它，短回答不再被长序列淹没，所以图像/视频问答能力能保住。

一句话概括：**论文真正的创新点是把“长上下文 capability”从架构问题，改造成一个可被系统训练 recipe 驯服的问题。**

### 战略取舍

| 设计选择 | 改变了什么约束 | 收益 | 代价 |
| --- | --- | --- | --- |
| Blockwise RingAttention | 从显存瓶颈转为多设备环形并行 | 能做 1M 精确 attention，保留完整远距交互 | 强依赖大规模 TPU/并行工程 |
| 渐进式扩窗 + RoPE θ 扩展 | 从一步到位难优化变为课程式迁移 | 更稳定地从 4K 扩到 1M，短上下文能力基本保留 | 多 stage 训练更复杂 |
| 书籍合成长 QA | 从被动 LM 信号变为远距检索监督 | 长上下文聊天/检索明显增强 | QA 分布与真实用户对话仍有差距 |
| VQGAN 离散视觉 token | 统一 text/image/video 的 token-in-token-out 接口 | 支持 any-to-any 生成与长视频理解 | OCR 与细粒度视觉理解损失较大 |
| Masked sequence packing | 从长度混训的统计偏差变为可控优化 | 在不牺牲吞吐的前提下保住短答案任务性能 | 实现复杂，需配合 loss reweight |

## Part III：证据与局限

### 关键证据

- **信号 1｜长上下文检索能力真的建立起来了（comparison）**  
  在 1M multi-needle 设置下，LWM-Text-1M 在 `N=4, R=1` 上达到 **0.84**，而通过位置外推到 1M 的 Llama-3.1-8B / Qwen2.5-7B / Mistral-7B 只有 **0.32 / 0.00 / 0.13**。这说明它不只是“支持 1M 输入”，而是**真的能在 1M 中找信息**。

- **信号 2｜长文档任务收益不止体现在 synthetic benchmark（comparison）**  
  在 LOFT 的 512K 设定上，LWM 在 **HotPotQA = 0.72**，明显高于 GPT-4o 的 **0.21** 和 Claude 3 Opus 的 **0.32**。这表明长上下文不是只对 needle retrieval 有用，对更自然的多文档检索/回答也有实用价值。

- **信号 3｜能力跳变体现在“能看更多帧”而不是只会聊（comparison）**  
  在 Video-MME 的 30–60 分钟子集上，LWM-1M 达到 **60.8**，并能处理最多 **1800 帧**；这让它在 7B 级开源模型里明显占优。这里的能力增益本质上来自：**更多真实时序证据进入上下文**。

- **信号 4｜作者提出的 training trick 确实在因果上起作用（ablation）**  
  masked sequence packing 把 VQAv2/SQA/POPE 从 **48.3/34.8/62.5** 提升到 **55.8/47.7/75.2**。另外，chat 与 retrieval 数据配比消融也显示：聊天比例越高，MT-Bench 越好，但 needle accuracy 会掉。这说明作者调到的是**能力分布**，不是偶然波动。

### 局限性

- **Fails when**: 需要跨多处证据做综合推理、一次检索多条 needle、或依赖 OCR/细粒度视觉识别时，性能会明显下降；长视频若必须保留更高帧率的瞬时细节，逐帧 VQ tokenization 也会吃亏。
- **Assumes**: 依赖 LLaMA-2 7B 初始化、Books3 与大规模网页图文/视频数据、VQGAN tokenizer，以及极高算力条件；训练使用 TPUv4-1024，1M 推理至少需要 v4-128，且作者用纯 float32，复现门槛很高。
- **Not designed for**: 追求短上下文视觉 benchmark SOTA、强 OCR/文档视觉理解、或直接证明这套 recipe 能无缝外推到 100B+ 模型；图像/视频生成部分也主要还是质性展示，不是本文的核心定量强项。

### 可复用组件

- **Blockwise RingAttention**：百万 token 的精确训练与解码实现。
- **Progressive context extension**：32K→1M 的课程式扩窗流程。
- **Model-generated long-context QA**：用长书籍构造远距检索监督。
- **Masked sequence packing + loss reweighting**：适合混合长度、多任务、多模态训练。

![[paperPDFs/Building_World_Models_from_Language_Priors/arXiv_2024/2024_World_Model_on_Million_Length_Video_And_Language_With_RingAttention.pdf]]