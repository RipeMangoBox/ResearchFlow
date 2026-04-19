---
title: 'SimPO: Simple Preference Optimization with a Reference-Free Reward'
type: paper
paper_id: P__SimPO_Simple_Preference_Optimization_with_a_Reference-Free_Reward
aliases:
- SimPO_Simple_Preference_Optimization_with_a_Reference-Free_Reward
year: 2024
venue: ''
paper_level: C
frame: rl_standard
changed_slots:
- preference_optimization_loss
- training_iteration_strategy
- preference_pair_construction
structurality_score: 0.3
keep_score: 0.29
open_code: true
concepts:
- '[[C__direct_preference_optimization]]'
bottleneck:
- '[[B__偏好优化方法在推理任务上的有效性瓶颈]]'
lineage: []
same_family_papers: []
paper_link: https://arxiv.org/abs/2404.19733
code_url: https://github.com/DSXiangLi/DecryptPrompt
---
# SimPO: Simple Preference Optimization with a Reference-Free Reward

> 基于 `rl_standard`，改了 `preference_optimization_loss`, `training_iteration_strategy`, `preference_pair_construction`,
> 属于 [[C__direct_preference_optimization]],
> 目标是缓解 [[B__偏好优化方法在推理任务上的有效性瓶颈]]

## 相对 baseline 改了什么

> 核心洞察是将推理任务重新框架为偏好优化问题：通过生成多个CoT候选并根据最终答案正确性构建偏好对，让模型学习区分好坏推理路径。关键创新是在DPO损失中添加NLL项，这确保模型不仅学会区分好坏答案，还要提高生成正确推理的概率。迭代过程让模型逐步改进其推理能力，每轮都基于更好的模型生成更有挑战性的偏好对。这种方法有效是因为它结合了偏好学习的对比优势和监督学习的生成质量保证。


## 关键公式

- $$L_{DPO+NLL} = L_{DPO} + \alpha \cdot L_{NLL}$$
  - preference optimization loss：核心创新：在DPO损失基础上添加负对数似然项，这被发现对性能至关重要

## 关键图表

- **Table 1**: GSM8K results comparing Iterative RPO against baselines
  - 证据：迭代RPO在GSM8K上从55.6%提升到81.6%的核心证据
- **Figure 1**: Iterative Reasoning Preference Optimization pipeline
  - 证据：方法的整体架构：生成CoT候选→构建偏好对→DPO+NLL训练

## 阅读建议

> **插件型改进**。可快速浏览，重点看改了哪个 slot 和实验对比。

## 详细分析

# SimPO: Simple Preference Optimization with a Reference-Free Reward

## Part I：问题与挑战

现有的迭代偏好优化方法在通用指令调优任务上表现良好，但在推理任务上改进有限。标准DPO等方法在数学推理等需要链式思维的任务上往往无法带来显著提升，甚至可能降低性能。这个问题的根本挑战在于：(1) 推理任务需要生成正确的中间步骤，而不仅仅是最终答案；(2) 传统偏好优化方法难以有效利用推理过程中的正确性信号；(3) 如何构建有效的偏好对来指导模型学习更好的推理路径。现有方法如STaR主要依赖监督微调，而非偏好优化，限制了其在复杂推理任务上的潜力。

## Part II：方法与洞察

本文提出迭代推理偏好优化(Iterative RPO)，核心创新是将偏好优化应用于链式思维推理任务。方法包含两个关键组件：(1) 偏好对构建策略：对每个训练问题生成N=30个CoT候选解，根据最终答案的正确性构建偏好对，正确答案作为获胜方，错误答案作为失败方；(2) 修改的DPO损失函数：L_DPO+NLL = L_DPO + α·L_NLL，在标准DPO损失基础上添加负对数似然项，作者发现这对性能至关重要。训练过程采用迭代方式：每轮生成新的偏好对，使用修改的损失函数训练，然后用更新后的模型进入下一轮。实验中进行4轮迭代，每轮生成约55-60k个偏好对。关键技术细节包括：温度参数在前两轮使用0.8，后两轮使用1.3以增加多样性；α=1，β=0.1；包含金标准解答以确保获胜集合非空。这种方法成功地将偏好优化的优势引入推理任务，通过迭代改进实现了显著的性能提升。

### 核心直觉

核心洞察是将推理任务重新框架为偏好优化问题：通过生成多个CoT候选并根据最终答案正确性构建偏好对，让模型学习区分好坏推理路径。关键创新是在DPO损失中添加NLL项，这确保模型不仅学会区分好坏答案，还要提高生成正确推理的概率。迭代过程让模型逐步改进其推理能力，每轮都基于更好的模型生成更有挑战性的偏好对。这种方法有效是因为它结合了偏好学习的对比优势和监督学习的生成质量保证。

## Part III：证据与局限

实验证据主要集中在GSM8K数据集上，显示了强有力的改进：从基线的55.6%提升到81.6%，使用32样本多数投票可达88.7%。每轮迭代都带来持续改进：第1轮73.1%，第2轮78.0%，第3轮81.1%，第4轮81.6%。对比实验显示标准DPO仅达到61.8%或60.3%，监督微调为63.5%，证明了方法的有效性。在MATH数据集上从12.5%提升到20.8%，ARC-Challenge上的改进也有提及但缺乏详细数据。然而，证据存在一些局限：(1) MATH和ARC-Challenge的详细实验结果未在实验部分充分展示；(2) 缺乏NLL项贡献的消融实验；(3) 与更强推理优化方法如Expert Iteration的对比缺失；(4) 温度参数选择缺乏理论依据。总体而言，GSM8K上的证据是充分和令人信服的，但其他数据集的证据相对薄弱。


### Delta Statement

核心洞察是将推理任务重新框架为偏好优化问题：通过生成多个CoT候选并根据最终答案正确性构建偏好对，让模型学习区分好坏推理路径。关键创新是在DPO损失中添加NLL项，这确保模型不仅学会区分好坏答案，还要提高生成正确推理的概率。迭代过程让模型逐步改进其推理能力，每轮都基于更好的模型生成更有挑战性的偏好对。这种方法有效是因为它结合了偏好学习的对比优势和监督学习的生成质量保证。
