---
title: 'RAFT: Reward rAnked FineTuning for Generative Foundation Model Alignment'
type: paper
paper_id: P__RAFT_Reward_rAnked_FineTuning_for_Generative_Foundation_Model_Alignment
aliases:
- RAFT_Reward_rAnked_FineTuning_for_Generative_Foundation_Model_Alignment
year: 2023
venue: arXiv (Cornell University)
paper_level: C
frame: rl_standard
changed_slots:
- 训练损失函数
- 优化算法
- 辅助模型架构
structurality_score: 0.3
keep_score: 0.26
open_code: true
concepts:
- '[[C__direct_preference_optimization]]'
bottleneck:
- '[[B__RLHF的实现复杂性和计算效率瓶颈]]'
lineage: []
same_family_papers: []
paper_link: https://arxiv.org/abs/2305.10425
code_url: https://github.com/diff-usion/Awesome-Diffusion-Models
---
# RAFT: Reward rAnked FineTuning for Generative Foundation Model Alignment

> 基于 `rl_standard`，改了 `训练损失函数`, `优化算法`, `辅助模型架构`,
> 属于 [[C__direct_preference_optimization]],
> 目标是缓解 [[B__RLHF的实现复杂性和计算效率瓶颈]]

## 相对 baseline 改了什么

> 核心洞察是绕过奖励模型的点式转换，直接在序列概率层面进行偏好对比。传统RLHF将成对偏好转换为点式奖励时引入噪声，而SLiC-HF保持偏好的相对性质，仅关心两个摘要的相对排序。这种直接的概率对比避免了价值函数估计的复杂性，使优化更加稳定和高效。


## 关键公式

- $$L_{cal}(\theta) = \max(0, \beta - \log P_\theta(y^+|x) + \log P_\theta(y^-|x))$$
  - 训练损失函数：SLiC-HF的核心损失函数，直接对比正负样本的序列概率
- $$\text{loss}(r_\phi) = -E_{(x,y^+,y^-)\sim D_{HF}}[\log(\sigma(r_\phi(x, y^+) - r_\phi(x, y^-)))]$$
  - 奖励模型训练：RLHF中奖励模型的训练损失，用于对比

## 关键图表

- **Table 4**: 不同模型规模和候选数量的性能对比
  - 证据：模型规模提升比增加候选数量更有效
- **Table 5**: SLiC-HF与RLHF-PPO的计算和内存效率对比
  - 证据：SLiC-HF在内存使用和并行化方面的优势

## 阅读建议

> **插件型改进**。可快速浏览，重点看改了哪个 slot 和实验对比。

## 详细分析

# RAFT: Reward rAnked FineTuning for Generative Foundation Model Alignment

## Part I：问题与挑战

传统的人类反馈强化学习（RLHF）虽然能有效对齐语言模型与人类偏好，但存在显著的实现复杂性和计算效率问题。RLHF-PPO需要维护策略网络、价值网络、奖励模型和SFT模型等多个同等规模的模型，在训练过程中占用4倍的参数内存。此外，PPO的训练循环包含采样解码步骤，导致优化步骤显著变慢，且解码并行性受限于批内并行。更重要的是，RLHF将成对的人类偏好数据转换为点式奖励分数时引入了噪声，价值函数估计也增加了优化的不稳定性。这些问题使得RLHF在实际应用中需要专业知识进行超参数调优，实现门槛较高。

## Part II：方法与洞察

SLiC-HF将序列似然校准（SLiC）技术适配到人类反馈学习场景，提供了RLHF的简化替代方案。核心改变是用排序校准损失函数L_cal(θ) = max(0, β - log P_θ(y+|x) + log P_θ(y-|x))直接对比正负样本的序列概率，而不是通过奖励模型转换。具体流程包括：(1)从SFT模型采样m个候选摘要；(2)使用人类偏好数据训练的排序模型对候选进行排序；(3)选择最佳和最差样本作为正负例；(4)应用排序校准损失进行模型更新。这种方法避免了RLHF中的价值函数估计，直接利用更清洁的偏好信号驱动参数更新。SLiC-HF支持完全并行解码，因为所有候选都使用相同的策略生成，可以缓存输入编码状态。更重要的是，该方法可以有效利用为其他模型收集的离线偏好数据，类似于离策略强化学习，无需重新收集昂贵的人类反馈数据。

### 核心直觉

核心洞察是绕过奖励模型的点式转换，直接在序列概率层面进行偏好对比。传统RLHF将成对偏好转换为点式奖励时引入噪声，而SLiC-HF保持偏好的相对性质，仅关心两个摘要的相对排序。这种直接的概率对比避免了价值函数估计的复杂性，使优化更加稳定和高效。

## Part III：证据与局限

实验证据显示SLiC-HF在TL;DR摘要任务上显著改善了SFT基线：770M参数模型的人类评估胜率从44.96%提升到86.21%，11B模型达到96.10%。计算效率方面，SLiC-HF的参数内存使用量仅为RLHF-PPO的1/4，支持完全并行解码而非批内并行。模型规模扩展比增加候选数量更有效：从770M扩展到11B显著提升性能，但候选数从8增加到64效果有限。然而，某些效率优势主要基于理论分析而非直接测量，实际加速比未量化。此外，跨模型的偏好数据泛化能力和优化稳定性的改善主要是推测性的，缺乏直接实验验证。


### Delta Statement

核心洞察是绕过奖励模型的点式转换，直接在序列概率层面进行偏好对比。传统RLHF将成对偏好转换为点式奖励时引入噪声，而SLiC-HF保持偏好的相对性质，仅关心两个摘要的相对排序。这种直接的概率对比避免了价值函数估计的复杂性，使优化更加稳定和高效。
