---
title: 'ORPO: Monolithic Preference Optimization without Reference Model'
type: paper
paper_id: P__ORPO_Monolithic_Preference_Optimization_without_Reference_Model
aliases:
- ORPO_Monolithic_Preference_Optimization_without_Reference_Model
year: 2024
venue: ''
paper_level: B
frame: rl_standard
changed_slots:
- 损失函数设计
- 训练流程架构
structurality_score: 0.6
keep_score: 0.29
open_code: true
concepts:
- '[[C__direct_preference_optimization]]'
bottleneck:
- '[[B__多阶段偏好对齐的复杂性和资源需求]]'
lineage: []
same_family_papers: []
paper_link: https://arxiv.org/abs/2403.07691
code_url: https://github.com/shibing624/MedicalGPT
---
# ORPO: Monolithic Preference Optimization without Reference Model

> 基于 `rl_standard`，改了 `损失函数设计`, `训练流程架构`,
> 属于 [[C__direct_preference_optimization]],
> 目标是缓解 [[B__多阶段偏好对齐的复杂性和资源需求]]

## 相对 baseline 改了什么

> 核心洞察是偏好对齐不需要与监督微调分离——可以在SFT过程中通过简单的几率比项同时实现两个目标。几率比提供了一种自然的方式来对比偏好和非偏好响应，无需额外的参考模型。这种统一的方法既简化了训练流程，又提高了资源效率，同时保持了对齐效果。


## 关键公式

- $$\mathcal{L}_{ORPO} = \mathcal{L}_{SFT} + \lambda \cdot \mathcal{L}_{OR}$$
  - 整体损失函数：ORPO的核心创新是将监督微调损失与赔率比损失结合
- $$\mathcal{L}_{OR} = -\log \sigma \left( \log \frac{P(y_w|x)}{P(y_l|x)} \right)$$
  - 偏好对比机制：使用对数赔率比来对比偏好和非偏好响应的概率

## 关键图表

- **Figure 1**: AlpacaEval2.0结果对比
  - 证据：ORPO在7B模型上超越13B基线模型的性能声明
- **Table 6**: IFEval指令级松散准确率
  - 证据：66.19%的IFEval性能声明

## 阅读建议

> **结构性改进**。建议先读 baseline，再看本文如何修改核心 slot。

## 详细分析

# ORPO: Monolithic Preference Optimization without Reference Model

## Part I：问题与挑战

现有的偏好对齐方法（如RLHF和DPO）通常需要多阶段训练流程：首先进行监督微调（SFT），然后进行单独的偏好对齐阶段，这需要额外的参考模型和计算资源。传统方法将SFT和偏好对齐视为独立的阶段，但这种分离可能是不必要的。作者观察到，在偏好对齐的背景下，对不受欢迎的生成风格施加轻微惩罚就足以实现偏好对齐的SFT。现有方法的复杂性和资源需求限制了其在实际应用中的效率和可扩展性。

## Part II：方法与洞察

ORPO（Odds Ratio Preference Optimization）提出了一种单阶段、无参考模型的偏好对齐方法。核心创新是将监督微调损失与基于对数几率比的偏好损失结合：L_ORPO = L_SFT + λ·L_OR，其中L_OR = -log σ(log(P(y_w|x)/P(y_l|x)))。该方法直接在SFT阶段同时处理偏好对齐，通过对数几率比来对比偏好和非偏好响应的概率。与传统方法不同，ORPO不需要单独的参考模型或多阶段训练，而是在单一训练过程中对选择的响应给予强适应信号，对拒绝的响应施加弱惩罚。作者从理论和实验两个角度证明了几率比是在SFT期间对比偏好和非偏好风格的合理选择。该方法在125M到7B的不同模型规模上都表现出一致的有效性。

### 核心直觉

核心洞察是偏好对齐不需要与监督微调分离——可以在SFT过程中通过简单的几率比项同时实现两个目标。几率比提供了一种自然的方式来对比偏好和非偏好响应，无需额外的参考模型。这种统一的方法既简化了训练流程，又提高了资源效率，同时保持了对齐效果。

## Part III：证据与局限

实验证据显示，使用ORPO微调的Mistral-ORPO-β在AlpacaEval2.0上达到12.20%的胜率，在IFEval指令级松散评估中达到66.19%准确率，在MT-Bench上获得7.32分。这些结果超越了更大规模的基线模型（7B和13B参数）。作者在Phi-2（2.7B）、Llama-2（7B）和Mistral（7B）上验证了方法的有效性，仅使用UltraFeedback数据集训练一个epoch就取得了优异性能。控制实验表明ORPO在不同模型规模上持续优于SFT和RLHF，对DPO的胜率随模型规模增加而提高。然而，局限性包括：未测试超过7B的模型规模，未包含更广泛的偏好对齐算法进行比较，缺乏对失败案例的详细分析。


### Delta Statement

核心洞察是偏好对齐不需要与监督微调分离——可以在SFT过程中通过简单的几率比项同时实现两个目标。几率比提供了一种自然的方式来对比偏好和非偏好响应，无需额外的参考模型。这种统一的方法既简化了训练流程，又提高了资源效率，同时保持了对齐效果。
