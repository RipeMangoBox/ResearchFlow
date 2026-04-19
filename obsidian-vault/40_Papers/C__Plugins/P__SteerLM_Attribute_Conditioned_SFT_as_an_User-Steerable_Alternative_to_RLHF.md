---
title: 'SteerLM: Attribute Conditioned SFT as an (User-Steerable) Alternative to RLHF'
type: paper
paper_id: P__SteerLM_Attribute_Conditioned_SFT_as_an_User-Steerable_Alternative_to_RLHF
aliases:
- SteerLM_Attribute_Conditioned_SFT_as_an_User-Steerable_Alternative_to_RLHF
year: 2023
venue: ''
paper_level: C
frame: rl_standard
changed_slots:
- preference_pair_sampling
- loss_function_design
structurality_score: 0.3
keep_score: 0.26
open_code: true
concepts:
- '[[C__direct_preference_optimization]]'
bottleneck:
- '[[B__偏好优化中采样分布与目标最优策略分布不匹配的问题]]'
lineage: []
same_family_papers: []
paper_link: https://arxiv.org/abs/2309.06657
code_url: https://github.com/NVIDIA/NeMo-Aligner
---
# SteerLM: Attribute Conditioned SFT as an (User-Steerable) Alternative to RLHF

> 基于 `rl_standard`，改了 `preference_pair_sampling`, `loss_function_design`,
> 属于 [[C__direct_preference_optimization]],
> 目标是缓解 [[B__偏好优化中采样分布与目标最优策略分布不匹配的问题]]

## 相对 baseline 改了什么

> 核心洞察是偏好优化的采样分布必须与目标最优策略匹配才能获得准确的策略估计。通过统计拒绝采样从估计的最优策略构造偏好对，而不是直接使用其他策略的偏好数据，可以显著改善策略学习效果。这种方法在保持离线训练简单性的同时，更好地逼近了理论上的最大似然估计要求。


## 关键公式

- $$L(θ) = max(0, δ - log π_θ(y_w|x) + log π_θ(y_l|x)) - λ log π_θ(y_ref|x)$$
  - SLiC loss function：SLiC的对比排序校准损失函数，结合了边际损失和正则化项
- $$p*(y_1 ≻ y_2|x) = 1/(1 + exp(β log π*(y_2|x)/π_sft(y_2|x) - β log π*(y_1|x)/π_sft(y_1|x)))$$
  - DPO preference modeling：DPO基于Bradley-Terry模型的人类偏好概率表达式

## 关键图表

- **Table 1**: Performance comparison showing RSO variants outperforming baselines
  - 证据：RSO在多个任务上持续优于SLiC和DPO的声明
- **Table 7**: Efficiency comparison of different approaches showing computational costs
  - 证据：不同方法的计算效率对比
- **Figure 1**: RSO pipeline showing reward model training and rejection sampling
  - 证据：RSO方法的整体架构和工作流程

## 阅读建议

> **插件型改进**。可快速浏览，重点看改了哪个 slot 和实验对比。

## 详细分析

# SteerLM: Attribute Conditioned SFT as an (User-Steerable) Alternative to RLHF

## Part I：问题与挑战

现有的离线偏好优化方法存在采样分布不匹配的根本问题。DPO直接使用人类偏好数据进行训练，但缺乏奖励模型限制了其从最优策略采样偏好对的能力。SLiC只能从SFT策略采样偏好对，而不是从目标最优策略采样。理论上，最大似然估计器需要从目标最优策略采样的标注偏好对才能准确估计最优策略，但现实中很难直接从π*获得人类偏好对。这种采样分布与目标分布的不匹配导致策略估计不够准确，限制了模型对齐效果。

## Part II：方法与洞察

RSO提出统计拒绝采样来解决采样分布不匹配问题。核心改进包括：1）构建从估计最优策略采样的偏好对：首先用人类偏好数据训练成对奖励排序模型，然后使用统计拒绝采样算法从SFT策略和奖励模型中生成从最优策略采样的响应对，最后用奖励模型标注这些采样对。2）统一损失函数框架：将DPO和SLiC的损失函数分别视为逻辑回归和支持向量机，提出改进的hinge-norm和sigmoid-norm损失。3）改进的采样策略：证明现有的top-k-over-N算法是统计拒绝采样的特殊情况，并在奖励利用和正则化之间取得更好平衡，避免过度信任奖励模型导致的奖励黑客攻击。方法保持了离线训练的简单性，同时更接近在线RLHF的策略采样。

### 核心直觉

核心洞察是偏好优化的采样分布必须与目标最优策略匹配才能获得准确的策略估计。通过统计拒绝采样从估计的最优策略构造偏好对，而不是直接使用其他策略的偏好数据，可以显著改善策略学习效果。这种方法在保持离线训练简单性的同时，更好地逼近了理论上的最大似然估计要求。

## Part III：证据与局限

实验在Reddit TL;DR和AnthropicHH两个任务上验证，RSO变体在代理奖励模型、黄金奖励模型、AutoSxS和人类评估四种评估方式下均显著优于RAFT、ReST、DPO和SLiC变体。rso-sample-rank方法比direct和sft-sample-rank方法带来明显提升。改进的hinge-norm损失在AutoSxS评估上优于SLiC使用的hinge损失。但实验范围相对有限，仅在两个任务上测试，且hinge损失在Reddit TL;DR数据集上显示奖励黑客攻击现象。计算效率方面，RSO需要额外的奖励模型推理开销。


### Delta Statement

核心洞察是偏好优化的采样分布必须与目标最优策略匹配才能获得准确的策略估计。通过统计拒绝采样从估计的最优策略构造偏好对，而不是直接使用其他策略的偏好数据，可以显著改善策略学习效果。这种方法在保持离线训练简单性的同时，更好地逼近了理论上的最大似然估计要求。
