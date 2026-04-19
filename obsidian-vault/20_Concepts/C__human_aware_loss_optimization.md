---
title: human_aware_loss_optimization
type: concept
concept_id: C__human_aware_loss_optimization
aliases: []
domain: RL
---
# human_aware_loss_optimization

KTO的核心洞察是认识到人类偏好表达与真实效用感知之间的差异。传统方法假设最大化偏好似然等同于最大化人类效用，但前景理论告诉我们人类的价值感知是非线性的、参考点依赖的。KTO通过直接建模这种非线性价值函数，更准确地捕捉了人类的真实效用，从而在对齐任务上取得更好的效果。这种从'拟合偏好'到'优化效用'的转变是根本性的。

## 改动的 Slot

`人类效用建模`, `损失函数设计`, `数据需求`

## 代表论文 (1)

| 论文 | Year | Venue | 结构性 | 改动 |
|------|------|-------|--------|------|
| [[P__KTO_Model_Alignment_as_Prospect_Theoretic_Optimization]] | 2024 | arXiv (Cornell University) | 0.80 | 损失函数设计, 人类效用建模 |
