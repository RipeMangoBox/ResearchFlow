---
title: Visual-Contrast Attention
type: method
method_type: mechanism_family
maturity: seed
parent_method: null
n_papers: 1
primary_tasks:
- Classification
- Image Generation
---

# Visual-Contrast Attention

Visual-Contrast Attention (VCA), a drop-in replacement for MHSA that uses spatially pooled contrast tokens with positive/negative differential streams, reduces complexity from O(N²C) to O(NnC) while improving both recognition and generation performance with negligible overhead.

**研究领域**: Classification, Image Generation

## 使用本方法的论文 (1)

| 论文 | 会议 | 年份 |
|------|------|------|
| [[P__线性差分视觉Transforme_Visual-Contrast_]] | NeurIPS | 2025 |

## 本方法的方法学基线

- InLine — _Reconciles softmax and linear attention; core related work on linear attention m_
- Agent Attention — _Integrates softmax and linear attention; directly relevant attention mechanism t_

