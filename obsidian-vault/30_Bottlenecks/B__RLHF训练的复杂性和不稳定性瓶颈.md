---
title: RLHF训练的复杂性和不稳定性瓶颈
type: bottleneck
bottleneck_id: B__RLHF训练的复杂性和不稳定性瓶颈
domain: RL
paper_count: 1
---
# RLHF训练的复杂性和不稳定性瓶颈

## 症状

传统RLHF需要训练多个模型且RL过程本质上不稳定，容易出现训练崩溃。奖励模型的误差会传播到策略优化阶段，而RL算法对超参数敏感，需要大量调优。

## 结构性解法

- [[P__Direct_Preference_Optimization_Your_Language_Model_is_Secretly_a_Reward_Model_DPO]] — 传统RLHF需要训练多个模型且RL过程本质上不稳定，容易出现训练崩溃。奖励模型的误差会传播到策略优化阶段，而RL算法对超参数敏感，需要大量调优。
