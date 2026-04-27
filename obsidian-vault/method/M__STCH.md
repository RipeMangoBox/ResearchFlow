---
title: STCH
type: method
method_type: mechanism_family
maturity: seed
parent_method: null
n_papers: 1
primary_tasks:
- Reasoning
---

# STCH

STCH（Smooth Tchebycheff Scalarization）是一种用于多目标优化的标量化方法，通过平滑化经典的Tchebycheff标量化函数，解决了原始方法在非光滑点处的优化困难。该方法在ICML 2024上提出，旨在为进化算法和基于梯度的优化提供更稳定的标量化策略。

传统的Tchebycheff方法虽然能够有效逼近Pareto前沿，但其max操作导致的不可微性限制了在深度学习等场景中的应用。STCH通过引入平滑近似，保持了Tchebycheff方法对非凸Pareto前沿的良好逼近能力，同时获得了梯度信息以便于端到端优化。

**研究领域**: Reasoning

## 使用本方法的论文 (1)

| 论文 | 会议 | 年份 |
|------|------|------|
| [[P__平滑切比雪夫标量化多目标优化_STCH_(Smooth_Tch]] | ICML | 2024 |

