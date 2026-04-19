---
title: 'RRHF: Rank Responses to Align Language Models with Human Feedback'
type: paper
paper_id: P__RRHF_Rank_Responses_to_Align_Language_Models_with_Human_Feedback
aliases:
- RRHF_Rank_Responses_to_Align_Language_Models_with_Human_Feedback
year: 2023
venue: ''
paper_level: C
frame: rl_standard
changed_slots:
- 训练目标函数
- 对比学习策略
- 排序利用方式
structurality_score: 0.4
keep_score: 0.26
open_code: true
concepts:
- '[[C__direct_preference_optimization]]'
bottleneck:
- '[[B__成对对比无法充分利用完整排序信息的宏观视角]]'
lineage: []
same_family_papers: []
paper_link: https://arxiv.org/abs/2306.17492
code_url: https://github.com/opendilab/awesome-RLHF
---
# RRHF: Rank Responses to Align Language Models with Human Feedback

> 基于 `rl_standard`，改了 `训练目标函数`, `对比学习策略`, `排序利用方式`,
> 属于 [[C__direct_preference_optimization]],
> 目标是缓解 [[B__成对对比无法充分利用完整排序信息的宏观视角]]

## 相对 baseline 改了什么

> 核心洞察是将人类对齐问题重新建模为概率排序对齐：让模型生成的n个回复的概率排序与人类偏好排序保持一致。通过迭代的一对多对比而非传统的一对一对比，模型能够学习到更细粒度的偏好区分。这种方法继承了RLHF的优势但避免了其复杂性，直接通过监督学习实现了间接强化学习的效果。


## 关键公式

- $$\mathcal{L}_{PRO} = -\sum_{i=1}^{n-1} \log \sigma(\beta (\log \pi_\theta(y_i|x) - \log \pi_{ref}(y_i|x)) - \beta (\log \pi_\theta(y_{i+1}|x) - \log \pi_{ref}(y_{i+1}|x)))$$
  - 训练目标函数：PRO的核心损失函数，将成对对比扩展到任意长度排序的多位置对比
- $$\beta = 0.05 \times (l-1)^2$$
  - 超参数设置：SFT损失权重随排序长度动态调整的公式

## 关键图表

- **Table 1**: 主实验结果对比不同方法在HH-RLHF数据集上的表现
  - 证据：PRO在奖励分数上优于所有基线方法的声明
- **Table 5**: 不同自举策略的结果对比
  - 证据：使用ChatGPT生成候选回复能显著提升性能的声明
- **Figure 4**: LLM被上下文误导产生负面回复的错误案例
  - 证据：模型仍存在被上下文误导的失败模式

## 阅读建议

> **插件型改进**。可快速浏览，重点看改了哪个 slot 和实验对比。

## 详细分析

# RRHF: Rank Responses to Align Language Models with Human Feedback

## Part I：问题与挑战

现有的人类反馈对齐方法存在两个核心问题：(1) RLHF虽然有效但复杂不稳定，需要反复试错优化，对超参数敏感；(2) 现有方法将多个候选回复的对比简化为成对对比，无法充分利用完整排序信息的宏观视角。传统SFT方法要么只选择最佳回复进行训练，要么将长排序切分为多个成对对比，都没有充分挖掘人类偏好排序中蕴含的丰富信息。这导致模型无法从多维度、多位置的对比中学习到更细粒度的人类偏好区分。

## Part II：方法与洞察

PRO提出了一种直接的监督微调算法来替代复杂的RLHF。核心创新是将成对对比扩展到任意长度的排序对比：给定人类排序的n个回复y1, y2, ..., yn，PRO采用迭代对比策略，首先将最佳回复y1作为正例，其余作为负例进行对比训练；然后忽略当前最佳回复，将次佳回复y2作为正例与剩余回复对比，如此迭代直到处理完所有回复。这种方法实现了多位置、多维度的对比学习。损失函数扩展了Bradley-Terry模型，通过L_PRO = -∑log σ(β(log π_θ(y_i|x) - log π_ref(y_i|x)) - β(log π_θ(y_{i+1}|x) - log π_ref(y_{i+1}|x)))来优化模型概率排序与人类偏好排序的对齐。此外，PRO还引入了自举策略，使用ChatGPT等模型生成额外候选回复来扩展排序长度，并动态调整SFT损失权重β = 0.05 × (l-1)²。

### 核心直觉

核心洞察是将人类对齐问题重新建模为概率排序对齐：让模型生成的n个回复的概率排序与人类偏好排序保持一致。通过迭代的一对多对比而非传统的一对一对比，模型能够学习到更细粒度的偏好区分。这种方法继承了RLHF的优势但避免了其复杂性，直接通过监督学习实现了间接强化学习的效果。

## Part III：证据与局限

实验在HH-RLHF数据集上验证了PRO的有效性。在基础的2-排序设置下，PRO比SFT提升6.52奖励分数，比DPO提升2.6分。使用ChatGPT扩展到3-排序后，总体奖励分数达到67.97。PRO在无害性任务上表现明显优于有用性任务，且排序长度越长性能提升越显著。然而，方法仍存在局限：模型容易受上下文误导产生有害回复，在某些子集上BLEU分数略低于基线，且需要额外计算成本生成候选回复。实验主要基于7B模型，缺乏与ChatGPT的直接对比。


### Delta Statement

核心洞察是将人类对齐问题重新建模为概率排序对齐：让模型生成的n个回复的概率排序与人类偏好排序保持一致。通过迭代的一对多对比而非传统的一对一对比，模型能够学习到更细粒度的偏好区分。这种方法继承了RLHF的优势但避免了其复杂性，直接通过监督学习实现了间接强化学习的效果。
