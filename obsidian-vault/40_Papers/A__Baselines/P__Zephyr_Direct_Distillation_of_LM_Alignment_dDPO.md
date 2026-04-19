---
title: 'Zephyr: Direct Distillation of LM Alignment (dDPO)'
type: paper
paper_id: P__Zephyr_Direct_Distillation_of_LM_Alignment_dDPO
aliases:
- Zephyr_Direct_Distillation_of_LM_Alignment_dDPO
year: 2023
venue: arXiv (Cornell University)
paper_level: A
frame: rl_standard
changed_slots:
- preference_modeling
- optimization_objective
structurality_score: 0.7
keep_score: 0.26
open_code: false
concepts:
- '[[C__direct_preference_optimization]]'
bottleneck:
- '[[B__Bradley-Terry建模假设在确定性偏好下导致的过拟合问题]]'
lineage: []
same_family_papers: []
paper_link: https://arxiv.org/abs/2310.12036
---
# Zephyr: Direct Distillation of LM Alignment (dDPO)

> 基于 `rl_standard`，改了 `preference_modeling`, `optimization_objective`,
> 属于 [[C__direct_preference_optimization]],
> 目标是缓解 [[B__Bradley-Terry建模假设在确定性偏好下导致的过拟合问题]]

## 相对 baseline 改了什么

> 核心变化是从依赖Bradley-Terry建模的间接优化转向直接优化成对偏好。通过设置Ψ为恒等函数，IPO避免了将偏好信号压缩为点式奖励的信息损失，特别是在确定性偏好场景下能够更好地保持KL正则化的作用，防止模型过度偏离参考策略。


## 关键公式

- $$\mathcal{L}_{\Psi PO}(\pi; \mathcal{D}) = -\mathbb{E}_{(x,y_w,y_l) \sim \mathcal{D}}[\Psi(\beta \log \frac{\pi(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi(y_l|x)}{\pi_{ref}(y_l|x)})]$$
  - general preference optimization framework：ΨPO的核心目标函数，通过任意非递减映射Ψ直接处理成对偏好
- $$\mathcal{L}_{IPO}(\pi; \mathcal{D}) = -\mathbb{E}_{(x,y_w,y_l) \sim \mathcal{D}}[\beta \log \frac{\pi(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi(y_l|x)}{\pi_{ref}(y_l|x)}]$$
  - identity preference optimization：IPO方法通过设置Ψ为恒等函数避免Bradley-Terry建模假设
- $$\mathcal{L}_{DPO}(\pi; \mathcal{D}) = -\mathbb{E}_{(x,y_w,y_l) \sim \mathcal{D}}[\log \sigma(\beta \log \frac{\pi(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi(y_l|x)}{\pi_{ref}(y_l|x)})]$$
  - direct preference optimization：DPO作为ΨPO的特例，其中Ψ为sigmoid函数

## 关键图表

- **Figure 1**: Comparison of IPO vs DPO on synthetic bandit problems
  - 证据：IPO在简单示例上优于DPO的实证证据
- **Table 1**: Theoretical comparison of RLHF, DPO, and IPO approximations
  - 证据：不同方法在Bradley-Terry假设和奖励建模方面的理论差异

## 阅读建议

> **必读 baseline**。先理解此论文建立的标准框架，再看后续改进。

## 详细分析

# Zephyr: Direct Distillation of LM Alignment (dDPO)

## Part I：问题与挑战

现有的人类偏好学习方法（RLHF和DPO）都依赖两个关键近似：第一个假设成对偏好可以替换为点式奖励（Bradley-Terry建模），第二个假设奖励模型能够从收集数据泛化到策略采样的分布外数据。DPO虽然绕过了第二个近似，但仍然严重依赖第一个近似。当偏好是确定性或接近确定性时，这种Bradley-Terry建模假设会导致过拟合偏好数据集，忽略KL正则化项，从而引起模型漂移。缺乏统一的理论框架来理解这些实用算法的行为和潜在缺陷。

## Part II：方法与洞察

论文提出了ΨPO（Ψ-preference optimization）通用框架，通过引入任意非递减映射Ψ，将目标函数完全用成对偏好表达，从而绕过两个近似假设。核心洞察是RLHF和DPO都可以表示为ΨPO的特例：DPO对应Ψ为sigmoid函数，RLHF对应特定的Ψ选择。基于这个统一框架，论文识别出现有方法在确定性偏好下的过拟合问题。作为解决方案，提出IPO（Identity Preference Optimization），将Ψ设置为恒等函数，完全绕过Bradley-Terry建模假设。IPO直接优化成对偏好差异，避免了将偏好转换为点式奖励的中间步骤。论文还提供了IPO的高效采样损失函数实现和性能保证证明。

### 核心直觉

核心变化是从依赖Bradley-Terry建模的间接优化转向直接优化成对偏好。通过设置Ψ为恒等函数，IPO避免了将偏好信号压缩为点式奖励的信息损失，特别是在确定性偏好场景下能够更好地保持KL正则化的作用，防止模型过度偏离参考策略。

## Part III：证据与局限

理论分析表明ΨPO框架能够统一现有方法并识别其局限性。在简单bandit问题上的实验显示IPO优于DPO，验证了理论预测。然而，实证证据相对薄弱：实验仅限于说明性的简单示例，缺乏大规模语言模型任务的验证。论文提供了IPO的性能保证证明，但具体保证的细节在摘要中未详述。整体而言，这是一个理论贡献较强但实证验证有限的工作。


### Delta Statement

核心变化是从依赖Bradley-Terry建模的间接优化转向直接优化成对偏好。通过设置Ψ为恒等函数，IPO避免了将偏好信号压缩为点式奖励的信息损失，特别是在确定性偏好场景下能够更好地保持KL正则化的作用，防止模型过度偏离参考策略。
