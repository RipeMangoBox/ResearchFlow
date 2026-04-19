---
title: Bradley-Terry建模假设在确定性偏好下导致的过拟合问题
type: bottleneck
bottleneck_id: B__Bradley-Terry建模假设在确定性偏好下导致的过拟合问题
domain: RL
paper_count: 1
---
# Bradley-Terry建模假设在确定性偏好下导致的过拟合问题

## 症状

当偏好接近确定性时，现有方法会过度拟合偏好数据而忽略KL正则化，导致策略过度偏离参考模型。这个问题源于将成对偏好强制转换为点式奖励的根本假设。

## 结构性解法

- [[P__Zephyr_Direct_Distillation_of_LM_Alignment_dDPO]] — 当偏好接近确定性时，现有方法会过度拟合偏好数据而忽略KL正则化，导致策略过度偏离参考模型。这个问题源于将成对偏好强制转换为点式奖励的根本假设。
