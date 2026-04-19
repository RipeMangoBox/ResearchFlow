---
title: reinforcement_learning_from_human_feedback
type: concept
concept_id: C__reinforcement_learning_from_human_feedback
aliases: []
domain: RL
---
# reinforcement_learning_from_human_feedback

核心变化是将语言模型的优化目标从统计建模转向人类偏好建模。传统方法优化'什么文本在统计上可能出现'，而InstructGPT优化'什么输出人类会偏好'。这种转变通过引入人类反馈循环实现：先让人类示范期望行为，再让人类评判输出质量，最后用强化学习将这些评判转化为模型的学习信号。本质上是用人类的价值判断替代了统计规律作为模型的指导原则。

## 改动的 Slot

`evaluation_metric`, `feedback_mechanism`, `optimization_target`, `training_objective`

## 代表论文 (1)

| 论文 | Year | Venue | 结构性 | 改动 |
|------|------|-------|--------|------|
| [[P__Training_language_models_to_follow_instructions_with_human_feedback_InstructGPT]] | 2022 |  | 0.80 | training_objective, optimization_target |
