---
title: F-OAL
type: method
method_type: mechanism_family
maturity: seed
parent_method: null
n_papers: 1
primary_tasks:
- Continual Learning
- Few-Shot Learning
- Compression
---

# F-OAL

F-OAL（Forward-only Online Analytic Learning）是 2024 年发表于 NeurIPS 的前向-only 在线解析学习方法。该方法针对类增量学习（Class Incremental Learning, CIL）场景，实现了仅前向传播（forward-only）的模型更新，彻底消除了反向传播的计算开销和内存占用。

F-OAL 的核心优势在于"Fast Training and Low Memory Footprint"——通过解析学习的闭式解特性，在在线数据流场景下实现实时模型适配，特别适合计算资源极度受限的边缘设备部署。

**研究领域**: Continual Learning, Few-Shot Learning, Compression

## 使用本方法的论文 (1)

| 论文 | 会议 | 年份 |
|------|------|------|
| [[P__无梯度前向在线解析增量学习_F-OAL_(Forward-o]] | NeurIPS | 2024 |

