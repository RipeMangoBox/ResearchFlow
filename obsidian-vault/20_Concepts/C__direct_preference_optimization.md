---
title: direct_preference_optimization
type: concept
concept_id: C__direct_preference_optimization
aliases: []
domain: RL
---
# direct_preference_optimization

DPO的核心洞察是认识到奖励模型和最优策略之间存在一一对应关系，可以通过重参数化直接从策略中提取隐式奖励。这避免了传统RLHF中奖励模型训练误差传播到策略优化的问题，同时将复杂的RL优化简化为稳定的分类学习，从根本上重新思考了偏好学习的范式。

## 改动的 Slot

`loss_function_design`, `optimization_objective`, `preference_modeling`, `preference_optimization_loss`, `preference_pair_construction`, `preference_pair_sampling`, `training_iteration_strategy`, `优化算法`, `奖励建模方式`, `学习目标`, `对比学习策略`, `损失函数设计`, `排序利用方式`, `模型架构`, `训练损失函数`, `训练流程架构`, `训练目标函数`, `辅助模型架构`

## 代表论文 (8)

| 论文 | Year | Venue | 结构性 | 改动 |
|------|------|-------|--------|------|
| [[P__Direct_Preference_Optimization_Your_Language_Model_is_Secretly_a_Reward_Model_DPO]] | 2023 |  | 0.80 | 训练目标函数, 优化算法 |
| [[P__RLHF_is_not_RL_Reward_Maximization_vs_Policy_Optimization]] | 2023 |  | 0.70 | 训练目标函数, 模型架构 |
| [[P__Zephyr_Direct_Distillation_of_LM_Alignment_dDPO]] | 2023 | arXiv (Cornell University) | 0.70 | preference_modeling, optimization_objective |
| [[P__ORPO_Monolithic_Preference_Optimization_without_Reference_Model]] | 2024 |  | 0.60 | 损失函数设计, 训练流程架构 |
| [[P__RRHF_Rank_Responses_to_Align_Language_Models_with_Human_Feedback]] | 2023 |  | 0.40 | 训练目标函数, 对比学习策略 |
| [[P__SteerLM_Attribute_Conditioned_SFT_as_an_User-Steerable_Alternative_to_RLHF]] | 2023 |  | 0.30 | preference_pair_sampling, loss_function_design |
| [[P__SimPO_Simple_Preference_Optimization_with_a_Reference-Free_Reward]] | 2024 |  | 0.30 | preference_optimization_loss, training_iteration_strategy |
| [[P__RAFT_Reward_rAnked_FineTuning_for_Generative_Foundation_Model_Alignment]] | 2023 | arXiv (Cornell University) | 0.30 | 训练损失函数, 优化算法 |
