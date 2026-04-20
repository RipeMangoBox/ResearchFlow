---
title: 'Video Detective: Seek Critical Clues Recurrently to Answer Question from Long Videos'
type: paper
paper_level: B
venue: arXiv (Cornell University)
year: 2025
acceptance: null
cited_by: null
facets:
  modality:
  - Image
paper_link: https://arxiv.org/abs/2512.17229
---

# Video Detective: Seek Critical Clues Recurrently to Answer Question from Long Videos

> **结构性改进**。先读 baseline，再看本文修改了哪些核心组件。

## 详细分析

# Video Detective: Seek Critical Clues Recurrently to Answer Question from Long Videos

## Part I：问题与挑战

长视频问答（LVQA）面临两个核心挑战：一是上下文长度爆炸，一段10分钟的视频以1fps采样得到600帧，在Qwen2.5-VL等模型中每帧编码为64个token，总计超过38K token，轻易突破32K的上下文限制；二是信息冗余与关键信息稀疏并存，长视频中绝大多数帧与特定问题无关，但现有方法无法有效区分关键与冗余信息。现有解决路径分为两类：其一是视觉token压缩（如合并相邻相似帧），但这种无目的性压缩可能丢失与问题相关的关键线索；其二是扩展模型上下文长度并在长视频数据上微调，但这带来极高的计算和显存开销。两类方法均未能同时兼顾效率与信息保留。本文的核心观察是：回答特定问题时，实际需要的有效信息量极少，人类在观看视频时也是带着问题边看边思考、逐步积累关键线索，而非一次性处理全部内容。因此，如何让模型以问题为导向、循环式地从长视频中主动寻找关键线索，同时将历史上下文高效传递给后续处理步骤，是本文试图解决的核心问题。此外，现有长视频基准仅评估最终预测结果，无法定量衡量模型真正定位关键线索的能力，缺乏细粒度评估工具也是领域痛点之一。
