---
title: MONA
type: method
method_type: mechanism_family
maturity: seed
parent_method: null
n_papers: 1
primary_tasks:
- Object Detection
- Instance Segmentation
- Semantic Segmentation
---

# MONA

MONA 是一种突破性的视觉识别参数高效微调方法，其核心主张是"5% > 100%"——即仅用全量微调 5% 的参数即可超越完整微调的性能表现。该方法在 CVPR 2025 上发表，挑战了传统认知中"全量微调性能最优"的假设。

该方法通过精心设计的参数更新策略，在视觉识别任务上实现了对全量微调（Full Fine-Tuning）的性能超越，打破了参数高效微调方法通常存在的性能瓶颈，为大规模视觉模型的实际部署提供了更具效率的替代方案。

**研究领域**: Object Detection, Instance Segmentation, Semantic Segmentation

## 使用本方法的论文 (1)

| 论文 | 会议 | 年份 |
|------|------|------|
| [[P__MONA：5%参数超越全量微调的_MONA_(Multi-cOmp]] | CVPR | 2025 |

## 本方法的方法学基线

- Laplace-LoRA — _Core method source; LoRA is fundamental low-rank adaptation technique likely bui_

