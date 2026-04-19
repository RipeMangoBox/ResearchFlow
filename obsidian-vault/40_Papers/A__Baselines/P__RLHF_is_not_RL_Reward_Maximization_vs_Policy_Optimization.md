---
title: 'RLHF is not RL: Reward Maximization vs Policy Optimization'
type: paper
paper_id: P__RLHF_is_not_RL_Reward_Maximization_vs_Policy_Optimization
aliases:
- RLHF_is_not_RL_Reward_Maximization_vs_Policy_Optimization
year: 2023
venue: ''
paper_level: A
frame: rl_standard
changed_slots:
- 训练目标函数
- 模型架构
- 学习目标
structurality_score: 0.7
keep_score: 0.26
open_code: false
concepts:
- '[[C__direct_preference_optimization]]'
bottleneck:
- '[[B__PPO训练的复杂性和资源需求瓶颈]]'
lineage: []
same_family_papers: []
paper_link: https://arxiv.org/abs/2304.05302
---
# RLHF is not RL: Reward Maximization vs Policy Optimization

> 基于 `rl_standard`，改了 `训练目标函数`, `模型架构`, `学习目标`,
> 属于 [[C__direct_preference_optimization]],
> 目标是缓解 [[B__PPO训练的复杂性和资源需求瓶颈]]

## 相对 baseline 改了什么

> 核心洞察是将RLHF从复杂的强化学习问题重新表述为简单的排序学习问题。通过直接优化模型对不同响应的对数概率排序来匹配人类偏好排序，避免了PPO中复杂的优势函数估计和多模型架构。本质上是学习best-of-n采样的目标，使模型能够生成接近训练集中最佳响应质量的输出。


## 关键公式

- $$\mathcal{L}_{rank} = -\log \sigma(\log p(y_w|x) - \log p(y_l|x))$$
  - 训练目标函数：RRHF的核心排序损失函数，通过对数概率差异学习人类偏好排序
- $$E_{x,y\sim\pi(x)}R(x, y) = \max_i E_{x,y_i\sim\rho_i(x)}R(x, y_i)$$
  - 学习目标：RRHF实际学习的是best-of-n采样的目标，而非传统RL的期望奖励最大化

## 关键图表

- **Table 8**: 训练和推理阶段不同方法的模型数量对比
  - 证据：RRHF相比PPO需要更少模型的核心声明
- **Table 6**: 不同采样策略下的奖励统计和性能对比
  - 证据：RRHF性能与采样质量高度相关，证明其为best-of-n学习器
- **Table 9**: Wombat与Alpaca和ChatGPT在Vicuna测试集上的对比
  - 证据：RRHF能够训练出ChatGPT级别的模型

## 阅读建议

> **必读 baseline**。先理解此论文建立的标准框架，再看后续改进。

## 详细分析

# RLHF is not RL: Reward Maximization vs Policy Optimization

## Part I：问题与挑战

传统RLHF中的PPO训练存在多个关键问题：(1) 超参数敏感性高，需要复杂的调优过程；(2) 训练时需要同时维护4个模型（策略模型、价值模型、奖励模型、参考模型），内存开销大且架构复杂；(3) 扩展到更大参数量的模型时面临资源和工程挑战；(4) PPO的优势函数估计需要额外的价值模型，增加了训练复杂度。这些问题使得RLHF的实际部署和规模化应用变得困难，特别是在资源受限的场景下。

## Part II：方法与洞察

RRHF提出了一种基于排序损失的新训练范式，核心改变包括：(1) 将强化学习问题重新表述为排序学习问题，使用对数概率差异的排序损失函数L_rank = -log σ(log p(y_w|x) - log p(y_l|x))来学习人类偏好；(2) 训练时只需1-2个模型，彻底简化了模型架构；(3) 可以利用多种来源的响应进行训练，包括模型自身、其他LLM和人类专家的响应；(4) 通过对数概率直接估计响应质量，无需额外的价值模型来估计基线；(5) 由于采样策略在训练前固定，KL散度项自然退化，无需参考模型。实验证明RRHF实际上是一个best-of-n学习器，学习目标是E_{x,y~π(x)}R(x,y) = max_i E_{x,y_i~ρ_i(x)}R(x,y_i)，即训练模型的平均奖励接近训练样本的最大奖励。

### 核心直觉

核心洞察是将RLHF从复杂的强化学习问题重新表述为简单的排序学习问题。通过直接优化模型对不同响应的对数概率排序来匹配人类偏好排序，避免了PPO中复杂的优势函数估计和多模型架构。本质上是学习best-of-n采样的目标，使模型能够生成接近训练集中最佳响应质量的输出。

## Part III：证据与局限

在Helpful and Harmless数据集上，RRHF与PPO达到相当的对齐性能。关键证据包括：(1) 训练的Wombat模型在Vicuna测试集上优于Alpaca但仍逊色于ChatGPT；(2) RRHF性能与采样质量高度相关，证实其为best-of-n学习器的本质；(3) 在线采样版本容易产生欺骗奖励模型的无意义友好回复，需要KL正则化缓解。局限性包括：需要多个响应作为输入增加GPU使用量；在线采样版本存在过度优化问题；高度依赖奖励评分质量，恶意评分可能误导模型生成不安全结果。


### Delta Statement

核心洞察是将RLHF从复杂的强化学习问题重新表述为简单的排序学习问题。通过直接优化模型对不同响应的对数概率排序来匹配人类偏好排序，避免了PPO中复杂的优势函数估计和多模型架构。本质上是学习best-of-n采样的目标，使模型能够生成接近训练集中最佳响应质量的输出。
