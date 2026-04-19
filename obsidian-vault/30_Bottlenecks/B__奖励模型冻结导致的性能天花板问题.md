---
title: 奖励模型冻结导致的性能天花板问题
type: bottleneck
bottleneck_id: B__奖励模型冻结导致的性能天花板问题
domain: RL
paper_count: 1
---
# 奖励模型冻结导致的性能天花板问题

## 症状

传统RLHF中奖励模型训练完成后保持冻结，无法在LLM训练过程中持续改进，限制了系统整体性能提升。这个瓶颈难以解决是因为需要在保持训练稳定性的同时实现奖励模型的持续学习。

## 结构性解法

- [[P__Self-Play_Fine-Tuning_Converts_Weak_Language_Models_to_Strong_SPIN]] — 传统RLHF中奖励模型训练完成后保持冻结，无法在LLM训练过程中持续改进，限制了系统整体性能提升。这个瓶颈难以解决是因为需要在保持训练稳定性的同时实现奖励模型的持续学习。
