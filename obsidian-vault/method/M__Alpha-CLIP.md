---
title: Alpha-CLIP
type: method
method_type: mechanism_family
maturity: seed
parent_method: null
n_papers: 1
primary_tasks:
- Few-Shot Learning
- Domain Adaptation
- OOD Detection
---

# Alpha-CLIP

Alpha-CLIP 是一种无需训练的 CLIP 模型自适应方法，发表于 ICLR 2024。该方法通过构建分类器集成（classifier ensemble）机制，在不修改 CLIP 预训练参数的前提下，提升其在下游任务上的零样本迁移性能。

作为"难以超越的基线"（hard-to-beat baseline），Alpha-CLIP 的设计目标是为训练无关的 CLIP 自适应方法提供一个简单 yet 有效的比较基准。其核心思想是利用多个分类器视角的互补性，缓解单一分类器在分布偏移或类别不平衡场景下的局限性。

**研究领域**: Few-Shot Learning, Domain Adaptation, OOD Detection

## 使用本方法的论文 (1)

| 论文 | 会议 | 年份 |
|------|------|------|
| [[P__CLIP免训练自适应的极简集成基_Alpha-CLIP_(trai]] | ICLR | 2024 |

