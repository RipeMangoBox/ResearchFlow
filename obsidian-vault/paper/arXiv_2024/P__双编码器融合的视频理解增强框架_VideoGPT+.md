---
title: 'VideoGPT+: Integrating Image and Video Encoders for Enhanced Video Understanding'
type: paper
paper_level: C
venue: arXiv
year: 2024
paper_link: https://www.semanticscholar.org/paper/7391bd9f259c7624e23cfac7ddaae94d16893ed9
aliases:
- 双编码器融合的视频理解增强框架
- VideoGPT+
- 单一编码器在空间细节与时序动态之间存在固有权衡
code_url: https://github.com/yunlong10/Awesome-LLMs-for-Video-Understanding
method: VideoGPT+
---

# VideoGPT+: Integrating Image and Video Encoders for Enhanced Video Understanding

[Paper](https://www.semanticscholar.org/paper/7391bd9f259c7624e23cfac7ddaae94d16893ed9) | [Code](https://github.com/yunlong10/Awesome-LLMs-for-Video-Understanding)

**Topics**: [[T__Video_Understanding]], [[T__Retrieval]], [[T__Captioning]] | **Method**: [[M__VideoGPT+]]

> [!tip] 核心洞察
> 单一编码器在空间细节与时序动态之间存在固有权衡，无法兼得。VideoGPT+ 的核心直觉是：与其设计复杂的统一编码器，不如直接并联两个已经在各自任务上充分预训练的专用编码器，通过简单拼接保留两类互补信息。有效性来源于两点：一是图像编码器与视频编码器的预训练目标不同，天然捕捉不同层次的视觉语义；二是LLM具备足够的上下文整合能力，能够从拼接的异质特征中提取有用信息，无需显式的跨模态对齐机制。

| 中文题名 | 双编码器融合的视频理解增强框架 |
| 英文题名 | VideoGPT+: Integrating Image and Video Encoders for Enhanced Video Understanding |
| 会议/期刊 | arXiv.org (preprint) |
| 链接 | [arXiv](https://www.semanticscholar.org/paper/7391bd9f259c7624e23cfac7ddaae94d16893ed9) · [Code](https://github.com/yunlong10/Awesome-LLMs-for-Video-Understanding) · [Project] |
| 主要任务 | VideoQA（MSVD-QA、MSRVTT-QA、ActivityNet-QA）、视频描述生成 |
| 主要 baseline | Video-LLaMA、Video-ChatGPT |

> [!abstract] 因为「单一编码器无法同时兼顾空间细节与时序动态」，作者在「Video-LLaMA/Video-ChatGPT」基础上改了「将单一视觉编码器替换为图像+视频双编码器并行结构」，在「MSVD-QA/MSRVTT-QA/ActivityNet-QA」上取得「超越现有Video-LLM的性能提升」

- **MSVD-QA / MSRVTT-QA / ActivityNet-QA**: 双编码器组合均优于单一图像或视频编码器配置（Table 3消融）
- **GPT辅助评估**: 正确性、细节、上下文、时序、一致性五个维度均有提升
- **视频描述生成**: CIDEr、METEOR等指标改善（Table 4）

## 背景与动机

视频理解的核心难点在于：同一视觉输入同时承载两种异质信息——单帧内的精细空间语义（如物体的纹理、姿态、场景布局）与跨帧的时序动态（如运动轨迹、事件演变、因果关系）。现有Video-LLM普遍采用单一编码器策略，被迫在这两者之间做出权衡。

**图像编码器路线**（如Video-LLaMA采用CLIP ViT）：逐帧独立提取空间特征，帧间通过时序池化或简单拼接处理。优势在于单帧表示精细，预训练充分；致命弱点是缺乏显式的帧间建模，对"球从A滚到B"这类运动语义理解薄弱。

**视频编码器路线**（如部分工作采用Video Swin Transformer）：直接对帧序列进行3D卷积或时空注意力建模，天然捕捉运动模式；但为换取计算效率通常降低空间分辨率，导致单帧细节模糊，且预训练数据规模通常小于图像编码器。

**具体瓶颈示例**：在ActivityNet-QA中回答"视频中的人先做了什么动作后做了什么动作"时，图像编码器路线可能准确识别每个帧的孤立姿态，却无法判断动作顺序；视频编码器路线可能感知到运动变化，但模糊的空间表示导致动作类别误判。现有方法（Video-LLaMA、Video-ChatGPT）在需要同时调用局部帧内容与全局时序语义的问题上表现不足，MSVD-QA等基准上的准确率仍有显著天花板。

VideoGPT+ 的动机直接源于此观察：与其强迫单一编码器承担不可兼得的双重职责，不如直接引入两个已在各自领域充分优化的专用编码器，让LLM自行整合互补信息。

## 核心创新

核心洞察：图像编码器与视频编码器的预训练目标天然异质，导致二者捕捉的视觉语义具有互补性，因为CLIP ViT的对比学习优化帧内判别性表示而Video Swin的时空建模优化跨帧运动一致性，从而使简单特征拼接即可在不引入复杂融合机制的前提下保留两类信息，交由LLM的上下文整合能力完成隐式对齐。

| 维度 | Baseline (Video-LLaMA/Video-ChatGPT) | 本文 (VideoGPT+) |
|:---|:---|:---|
| 视觉编码 | 单一编码器（CLIP ViT或Video Swin） | 双编码器并行：CLIP ViT + Video Swin |
| 特征表示 | 纯空间特征或纯时空特征 | 空间特征序列拼接时序特征向量 |
| 融合机制 | 无（单一来源） | Concat拼接，无显式交互模块 |
| LLM输入维度 | 标准视觉token长度 | 增大（T帧图像特征 + 1视频特征） |
| 训练目标 | 标准交叉熵 | 交叉熵 + 正则化项 |
| 架构改动范围 | — | 仅视觉编码槽位，其余组件沿用 |

## 整体框架

VideoGPT+ 的流水线遵循标准Video-LLM范式，仅在视觉编码阶段引入双路并行结构：

**输入**：视频帧序列（采样T帧）

**模块A：图像编码支路** — 输入为T帧独立图像，CLIP ViT逐帧提取空间特征，输出为T个帧级特征向量 {f_img^1, f_img^2, ..., f_img^T}，保留细粒度空间语义。

**模块B：视频编码支路** — 输入为完整帧序列，Video Swin Transformer进行时空联合建模，输出为单个视频级特征向量 f_vid，编码全局时序动态。

**模块C：特征拼接层** — 将两路特征按通道/序列维度拼接：f_video = Concat(f_img^1, f_img^2, ..., f_img^T, f_vid)，形成异质但互补的联合视觉表示。

**模块D：投影层** — 将拼接后的视觉特征映射到LLM的输入嵌入空间，维度适配由线性投影完成。

**模块E：LLM解码器** — 接收投影后的视觉token与文本指令，自回归生成答案或描述。

```
视频帧序列 ──┬──→ [CLIP ViT] ──→ 帧特征序列 ──┐
             │                              ├──→ [Concat] ──→ [投影层] ──→ [LLM] ──→ 输出文本
             └──→ [Video Swin] ──→ 视频特征 ──┘
```

关键设计决策：两编码器均基于大规模预训练权重初始化，与LLM联合微调；无需额外的跨模态对齐损失或注意力融合模块，依赖LLM的隐式整合能力。

## 核心模块与公式推导

### 模块 1: 双编码器特征提取（对应框架图 左支路与右支路）

**直觉**: 空间细节与时序动态由异构网络分别优化，避免单一网络的表示折中。

**Baseline 公式** (Video-LLaMA): 
$$f_{visual} = \text{Encoder}_{single}(\{x_t\}_{t=1}^T)$$
符号: $x_t$ = 第t帧图像, $\text{Encoder}_{single}$ = 单一图像或视频编码器

**变化点**: Baseline只能输出空间特征序列或压缩时序特征之一，无法同时保留逐帧细节与全局动态。

**本文公式（推导）**:
$$\text{Step 1}: \quad f_{img}^t = \text{CLIP-ViT}(x_t), \quad t=1,...,T \quad \text{逐帧提取空间特征}$$
$$\text{Step 2}: \quad f_{vid} = \text{Video-Swin}(\{x_t\}_{t=1}^T) \quad \text{提取时空联合特征}$$
$$\text{Step 3}: \quad f_{video} = \text{Concat}([f_{img}^1; f_{img}^2; ...; f_{img}^T; f_{vid}]) \quad \text{保留两类信息的简单拼接}$$
**最终**: 联合视觉表示 $f_{video} \in \mathbb{R}^{(T \cdot d_{img} + d_{vid})}$ 或序列形式送入投影层

**对应消融**: Table 3显示仅图像编码器、仅视频编码器、双编码器三种配置，双编码器在所有基准最优（具体Δ数值。

---

### 模块 2: 投影层与LLM联合优化（对应框架图 中部至右侧）

**直觉**: 异质拼接特征需映射到LLM语义空间，同时保持训练稳定性。

**Baseline 公式** (标准Video-LLM投影):
$$h_{LLM} = W_{proj} \cdot f_{visual} + b_{proj}$$
符号: $W_{proj}$ = 投影矩阵, $h_{LLM}$ = LLM输入嵌入

**变化点**: 双编码器使$f_{video}$维度显著增大，且包含异质特征（帧级序列+全局向量），需额外正则化稳定联合微调。

**本文公式（推导）**:
$$\text{Step 1}: \quad h_{video} = W_{proj} \cdot f_{video} + b_{proj} \quad \text{线性投影到LLM空间}$$
$$\text{Step 2}: \quad L_{CE} = -\sum_{i} \log P_{LLM}(y_i | y_{<i}, h_{video}, x_{text}) \quad \text{标准自回归语言建模损失}$$
$$\text{Step 3}: \quad L_{reg} = \lambda \cdot \Omega(\theta_{enc}, \theta_{proj}) \quad \text{加入正则化项约束编码器与投影层更新幅度}$$
**最终**: $$L_{total} = L_{CE} + \lambda \cdot L_{reg}$$

符号补充: $y_i$ = 目标token, $x_{text}$ = 文本指令, $\lambda$ = 正则化系数, $\Omega$ = 参数正则化函数（如L2或梯度裁剪相关），$\theta_{enc}$、$\theta_{proj}$分别为编码器与投影层参数。

**对应消融**: 论文未明确报告λ取值对性能的影响。

---

### 模块 3: 训练策略（隐式模块）

**直觉**: 冻结预训练编码器会损失适配性，全量微调会破坏预训练知识，需折中。

**变化点**: 两编码器均采用预训练权重初始化，与LLM联合微调（非冻结），但具体是否采用LoRA等高效微调技术未明确说明。正则化项$L_{reg}$的设计意图在于缓解大容量双编码器与LLM联合优化时的不稳定风险。

## 实验与分析

**主实验结果（VideoQA）**

| Method | MSVD-QA | MSRVTT-QA | ActivityNet-QA | 备注 |
|:---|:---|:---|:---|:---|
| Video-LLaMA |  |  |  | 图像编码器基线 |
| Video-ChatGPT |  |  |  | 另一图像编码器基线 |
| **VideoGPT+ (本文)** | **优** | **优** | **优** | 双编码器 |

Table 2显示VideoGPT+在三项VideoQA基准上均超越Video-LLaMA和Video-ChatGPT。GPT辅助评估的五个维度（正确性、细节、上下文、时序、一致性）均有提升，其中**时序维度**的提升最直接支持核心洞察——视频编码器的引入确实弥补了纯图像编码器路线在帧间关系建模上的短板。

**消融实验（Table 3）关键发现**：
- 仅图像编码器（CLIP ViT）：空间细节丰富，时序问题薄弱
- 仅视频编码器（Video Swin）：运动感知强，单帧精度受限  
- **图像+视频双编码器**：在所有测试基准上最优，证明互补性假设成立

**视频描述生成（Table 4）**：CIDEr、METEOR指标改善，说明双编码器增益不仅限于判别式QA任务，也惠及生成式任务。

**公平性检查与局限**：
- **基线强度不足**：未对比InternVideo2、LLaVA-NeXT-Video等同期更强方法，提升幅度可能被高估
- **视频长度局限**：实验主要在短/中等长度视频（<5分钟）进行，长视频场景帧采样策略可能丢失关键信息
- **效率代价未明**：特征拼接导致LLM输入token数增加（约T+1倍视觉token），计算开销上升但论文未报告FLOPs或推理延迟
- **评估主观性**：GPT辅助评估存在模型版本依赖与提示敏感性，可重复性有限
- **组合普适性**：不同预训练编码器配对（如EVA-CLIP + InternVid）的效果未充分探索

## 方法谱系与知识库定位

**方法家族**: Video-LLM（视频大语言模型）

**Parent method**: Video-LLaMA / Video-ChatGPT 所确立的「视觉编码器 → 投影层 → LLM」标准流水线

**改动槽位**: 仅**architecture（视觉编码模块）**发生组件替换，objective（交叉熵+正则化）、training_recipe（联合微调）、data_curation（标准VideoQA数据集）、inference（自回归生成）均沿用现有范式。

**直接基线对比**：
- **Video-LLaMA**: 单一CLIP ViT图像编码器 → 本文增加Video Swin时序支路
- **Video-ChatGPT**: 类似单一图像编码器路线 → 本文以最小架构改动引入双路互补
- **InternVid/UMT等视频编码器工作**: 仅用视频编码器 → 本文保留图像支路确保空间精度

**后续方向**：
1. **高效融合机制**：当前简单Concat导致输入膨胀，可探索轻量跨模态注意力（如Q-Former风格）压缩token数
2. **长视频扩展**：当前T帧均匀采样对长视频稀疏，需结合时序自适应采样或分层编码
3. **编码器组合搜索**：系统验证不同图像/视频编码器配对（EVA-CLIP、DINOv2、InternVideo2等）的增益边界

**知识库标签**: 
- modality: video + text
- paradigm: encoder-fusion → LLM
- scenario: short-to-medium video understanding
- mechanism: complementary feature concatenation
- constraint: dual-encoder computational overhead, no explicit cross-modal alignment
