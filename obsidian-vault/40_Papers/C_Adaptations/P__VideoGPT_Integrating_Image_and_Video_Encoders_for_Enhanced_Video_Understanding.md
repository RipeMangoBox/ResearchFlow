---
title: 'VideoGPT+: Integrating Image and Video Encoders for Enhanced Video Understanding'
type: paper
paper_level: C
venue: arXiv.org
year: 2024
acceptance: null
cited_by: null
core_operator: 单一编码器在空间细节与时序动态之间存在固有权衡，无法兼得。VideoGPT+ 的核心直觉是：与其设计复杂的统一编码器，不如直接并联两个已经在各自任务上充分预训练的专用编码器，通过简单拼接保留两类互补信息。有效性来源于两点：一是图像编码器与视频编码器的预训练目标不同，天然捕捉不同层次的视觉语义；二是LLM具备足够的上下文整合能力，能够从拼接的异质特征中提取有用信息，无需显式的跨模态对齐机制。
paper_link: https://www.semanticscholar.org/paper/7391bd9f259c7624e23cfac7ddaae94d16893ed9
code_url: https://github.com/yunlong10/Awesome-LLMs-for-Video-Understanding
structurality_score: 0.35
---

# VideoGPT+: Integrating Image and Video Encoders for Enhanced Video Understanding

## Links

- Mechanism: [[C__video_large_language_model]]

> 单一编码器在空间细节与时序动态之间存在固有权衡，无法兼得。VideoGPT+ 的核心直觉是：与其设计复杂的统一编码器，不如直接并联两个已经在各自任务上充分预训练的专用编码器，通过简单拼接保留两类互补信息。有效性来源于两点：一是图像编码器与视频编码器的预训练目标不同，天然捕捉不同层次的视觉语义；二是LLM具备足够的上下文整合能力，能够从拼接的异质特征中提取有用信息，无需显式的跨模态对齐机制。

> **适配/插件型**。可快速浏览，看改了哪个 slot 和效果。

## 核心公式

$$
\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda \mathcal{L}_{reg}
$$

> 总损失函数结合交叉熵分类损失与正则化项，控制模型训练方向。
> *Slot*: training objective

$$
f_{video} = \text{Concat}(f_{img}^{1}, f_{img}^{2}, \ldots, f_{img}^{T}, f_{vid})
$$

> 将逐帧图像编码器特征与视频编码器全局特征拼接，形成增强的视频表示。
> *Slot*: dual-encoder feature fusion

## 关键图表

**Table 2**
: VideoGPT+ vs. state-of-the-art on multiple video QA and captioning benchmarks
> 证据支持: VideoGPT+ 在多个视频理解基准上超越现有方法的核心实验证据

**Figure 2**
: Architecture diagram showing dual image+video encoder integration into LLM pipeline
> 证据支持: 双编码器融合架构设计的机制说明

**Table 3**
: Ablation study on encoder combinations (image-only, video-only, dual)
> 证据支持: 双编码器相比单一编码器的增量贡献的消融实验证据

**Table 4**
: Results on video captioning benchmarks (MSVD, MSRVTT, ActivityNet)
> 证据支持: VideoGPT+ 在视频描述生成任务上的性能提升证据

## 详细分析

# VideoGPT+: Integrating Image and Video Encoders for Enhanced Video Understanding

## Part I：问题与挑战

现有视频大语言模型（Video-LLM）在视频理解任务中普遍依赖单一编码器策略：要么使用图像编码器（如CLIP ViT）逐帧提取空间特征，要么使用视频编码器（如Video Swin Transformer）提取时序特征。这两种策略各有局限——图像编码器擅长捕捉细粒度空间语义，但缺乏对帧间时序动态的建模能力；视频编码器能感知运动和时序关系，但往往牺牲了单帧的精细空间表示。单一编码器无法同时兼顾空间细节与时序动态，导致在需要综合理解视觉内容的视频问答（VideoQA）和视频描述生成任务中存在明显瓶颈。此外，现有方法（如Video-LLaMA、Video-ChatGPT）在多个标准基准（MSVD-QA、MSRVTT-QA、ActivityNet-QA）上的性能仍有较大提升空间，尤其是在需要同时理解局部帧内容与全局时序语义的问题上表现不足。VideoGPT+ 针对上述问题，提出通过整合双编码器来弥补单一编码器的信息缺口，以期在视频理解任务中获得更全面的视觉表示。

## Part II：方法与洞察

VideoGPT+ 的核心改动是在标准的「视觉编码器 → 投影层 → LLM」流水线中，将原本单一的视觉编码器替换为双编码器并行结构：同时引入图像编码器和视频编码器，分别处理视频输入，再将两路特征融合后送入大语言模型。

具体而言，图像编码器（如CLIP ViT）对视频的每一帧独立提取空间特征，得到逐帧的细粒度视觉表示 f_img^1, f_img^2, ..., f_img^T；视频编码器（如Video Swin Transformer）则对帧序列整体建模，提取跨帧的时序动态特征 f_vid。两路特征通过拼接（concatenation）方式融合：

f_video = Concat(f_img^1, f_img^2, ..., f_img^T, f_vid)

融合后的特征向量经投影层映射到LLM的输入空间，驱动语言模型生成答案或描述。训练目标为标准交叉熵损失加正则化项：L_total = L_CE + λ·L_reg。

两个编码器均基于大规模预训练权重初始化，并与语言模型联合微调。这一设计的核心洞察在于：图像编码器与视频编码器所捕捉的信息具有互补性——前者提供丰富的帧内语义细节，后者提供帧间运动与时序关系，简单拼接即可在不引入复杂融合机制的前提下保留两类信息。

从流水线角度看，该改动发生在「视觉编码」槽位：原本单一编码器被替换为双编码器并行结构，输出维度相应增大。投影层、LLM主干、训练数据、推理流程等其余组件均沿用标准Video-LLM范式，未作实质性修改。因此，该方法本质上是对视觉编码模块的替换/扩展，属于组件级改动，而非对整体架构范式的根本性重构。

### 核心直觉

单一编码器在空间细节与时序动态之间存在固有权衡，无法兼得。VideoGPT+ 的核心直觉是：与其设计复杂的统一编码器，不如直接并联两个已经在各自任务上充分预训练的专用编码器，通过简单拼接保留两类互补信息。有效性来源于两点：一是图像编码器与视频编码器的预训练目标不同，天然捕捉不同层次的视觉语义；二是LLM具备足够的上下文整合能力，能够从拼接的异质特征中提取有用信息，无需显式的跨模态对齐机制。

## Part III：证据与局限

消融实验（Table 3）直接验证了双编码器设计的有效性：在所有测试基准上，图像+视频双编码器组合均优于仅使用图像编码器或仅使用视频编码器的单一配置，证明两类特征确实存在互补性。主实验（Table 2）显示VideoGPT+在MSVD-QA、MSRVTT-QA、ActivityNet-QA等基准上超越Video-LLaMA和Video-ChatGPT，GPT辅助评估的五个维度（正确性、细节、上下文、时序、一致性）也均有提升。视频描述生成任务（Table 4）中CIDEr、METEOR等指标同样有所改善。

主要局限：第一，对比基线不够全面，未包含同期更强的方法（如InternVideo2、LLaVA-NeXT-Video），性能提升幅度可能被高估。第二，实验主要在短/中等长度视频基准上进行，对长视频（>5分钟）的泛化能力未系统评估，长视频场景下帧采样策略可能丢失关键信息。第三，GPT辅助评估存在主观性和版本依赖问题，可重复性有限。第四，特征拼接导致LLM输入维度增大，计算开销上升，但论文未充分讨论效率代价。第五，不同预训练编码器组合的效果差异未充分消融，结论的普适性有待验证。
