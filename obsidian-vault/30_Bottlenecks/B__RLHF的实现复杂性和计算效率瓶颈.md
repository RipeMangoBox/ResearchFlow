---
title: RLHF的实现复杂性和计算效率瓶颈
type: bottleneck
bottleneck_id: B__RLHF的实现复杂性和计算效率瓶颈
domain: RL
paper_count: 1
---
# RLHF的实现复杂性和计算效率瓶颈

## 症状

RLHF需要协调多个大型模型和复杂的强化学习过程，导致内存占用高、训练速度慢、超参数调优困难。偏好到奖励的转换引入噪声，价值函数估计增加不稳定性。

## 插件型解法

- [[P__RAFT_Reward_rAnked_FineTuning_for_Generative_Foundation_Model_Alignment]] — RLHF需要协调多个大型模型和复杂的强化学习过程，导致内存占用高、训练速度慢、超参数调优困难。偏好到奖励的转换引入噪声，价值函数估计增加不稳定性。
