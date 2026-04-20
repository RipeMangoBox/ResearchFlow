---
title: Long Context Transfer from Language to Vision
type: paper
paper_level: B
venue: Trans. Mach. Learn. Res.
year: 2024
acceptance: null
cited_by: 441
paper_link: https://www.semanticscholar.org/paper/d081584960c42f7793502bb496e46f682e3e43b3
code_url: https://github.com/EvolvingLMMs-Lab/LongVA
---

# Long Context Transfer from Language to Vision

> **结构性改进**。先读 baseline，再看本文修改了哪些核心组件。

## 详细分析

# Long Context Transfer from Language to Vision

## Part I：问题与挑战

现有大型多模态模型（LMMs）在理解极长视频时存在严重瓶颈。视频序列天然携带丰富的时序信息，但当视频帧数增多时，视觉token数量急剧膨胀，超出模型上下文窗口的承载能力。主流解决思路是通过视觉重采样器（visual resampler）压缩视觉token数量，例如Q-Former或Perceiver Resampler，但这类方法以信息损失为代价换取序列长度的可控性，在需要细粒度时序理解的任务上存在天花板。另一类思路是针对视频数据进行专项微调，但这需要大量高质量的长视频标注数据，成本高昂且泛化性存疑。核心矛盾在于：语言骨干模型的上下文窗口长度是固定的（通常为4K或8K token），而密集采样的长视频可能产生数十万个视觉token，两者之间存在数量级的差距。此外，学界缺乏专门评估LMMs长视觉上下文理解能力的基准，现有评测（如Video-MME）虽涵盖长视频，但无法精确诊断模型在极长视觉序列中的检索与推理能力。因此，该问题同时面临方法论和评测体系两个层面的挑战。
