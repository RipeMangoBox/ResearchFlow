---
title: iterative_preference_optimization
type: concept
concept_id: C__iterative_preference_optimization
aliases: []
domain: RL
---
# iterative_preference_optimization

核心洞察是打破奖励模型冻结的传统范式，让模型在训练过程中同时改进指令跟随和奖励建模能力。通过将奖励建模重新框架为指令跟随任务，实现了统一模型架构下的多任务学习。这种设计允许模型为自己生成越来越高质量的训练信号，突破了人类标注数据的性能瓶颈，开启了模型自我改进的可能性。

## 改动的 Slot

`preference_generation`, `reward_model`, `training_loop`

## 代表论文 (1)

| 论文 | Year | Venue | 结构性 | 改动 |
|------|------|-------|--------|------|
| [[P__Self-Play_Fine-Tuning_Converts_Weak_Language_Models_to_Strong_SPIN]] | 2024 |  | 0.60 | reward_model, training_loop |
