---
title: Training language models to follow instructions with human feedback (InstructGPT)
type: paper
paper_id: P__Training_language_models_to_follow_instructions_with_human_feedback_InstructGPT
aliases:
- Training_language_models_to_follow_instructions_with_human_feedback_InstructGPT
year: 2022
venue: ''
paper_level: A
frame: rl_standard
changed_slots:
- training_objective
- optimization_target
- feedback_mechanism
- evaluation_metric
structurality_score: 0.8
keep_score: 0.26
open_code: true
concepts:
- '[[C__reinforcement_learning_from_human_feedback]]'
bottleneck:
- '[[B__语言模型的优化目标与用户意图之间的根本性错位]]'
lineage: []
same_family_papers: []
paper_link: https://arxiv.org/abs/2203.02155
code_url: https://github.com/opendilab/awesome-RLHF
---
# Training language models to follow instructions with human feedback (InstructGPT)

> 基于 `rl_standard`，改了 `training_objective`, `optimization_target`, `feedback_mechanism`,
> 属于 [[C__reinforcement_learning_from_human_feedback]],
> 目标是缓解 [[B__语言模型的优化目标与用户意图之间的根本性错位]]

## 相对 baseline 改了什么

> 核心变化是将语言模型的优化目标从统计建模转向人类偏好建模。传统方法优化'什么文本在统计上可能出现'，而InstructGPT优化'什么输出人类会偏好'。这种转变通过引入人类反馈循环实现：先让人类示范期望行为，再让人类评判输出质量，最后用强化学习将这些评判转化为模型的学习信号。本质上是用人类的价值判断替代了统计规律作为模型的指导原则。


## 关键公式

- $$\text{reward} = \text{RM}(x, y)$$
  - PPO reinforcement learning stage：奖励模型将人类偏好转化为可优化的奖励信号
- $$\text{objective} = \mathbb{E}_{(x,y) \sim \rho_{\pi_{\text{RL}}}} [r_{\theta}(x,y)] - \beta \log(\pi_{\text{RL}}(y|x) / \pi_{\text{SFT}}(y|x))$$
  - PPO optimization：PPO目标函数平衡奖励最大化与保持接近监督微调模型

## 关键图表

- **Figure 1**: Human evaluations showing InstructGPT models significantly outperform GPT-3 baselines
  - 证据：1.3B InstructGPT 优于 175B GPT-3 的核心声明
- **Figure 3**: Preference results measured by winrate against 175B SFT model across different prompt distributions
  - 证据：模型在不同提示分布上的泛化能力
- **Figure 4**: Metadata results showing InstructGPT models follow instructions better and hallucinate less
  - 证据：模型在具体行为维度上的改进证据

## 阅读建议

> **必读 baseline**。先理解此论文建立的标准框架，再看后续改进。

## 详细分析

# Training language models to follow instructions with human feedback (InstructGPT)

## Part I：问题与挑战

大型语言模型虽然参数规模庞大，但在遵循用户意图方面存在根本性缺陷。传统的语言建模目标（预测网页文本的下一个token）与实际应用需求（有用、诚实、无害地帮助用户）存在错位。GPT-3等模型经常产生不真实、有毒或无用的输出，无法可靠地遵循指令。这种错位问题在部署到实际应用中时尤为严重，因为用户期望模型能够理解并执行他们的明确指令，而不是简单地延续训练数据中的模式。现有的提示工程方法只能部分缓解这一问题，无法从根本上解决模型行为与用户期望之间的鸿沟。

## Part II：方法与洞察

InstructGPT提出了一个三阶段的训练流程来重新对齐语言模型的行为目标。首先进行监督微调（SFT），使用人类标注者编写的高质量示例来训练模型学习期望的输出行为。然后训练奖励模型（RM），通过收集人类对模型输出的偏好排序数据，学习预测人类会偏好哪种输出。最后使用PPO强化学习算法，以奖励模型的评分作为奖励信号来进一步优化模型，同时加入KL散度约束防止模型偏离监督微调的基线太远。这个方法的核心洞察是将人类反馈转化为可优化的奖励信号，从而将模型的优化目标从'预测下一个token'转变为'最大化人类满意度'。通过这种方式，1.3B参数的InstructGPT模型在人类评估中的表现超过了175B参数的GPT-3，证明了对齐比单纯扩大模型规模更为重要。该方法还展现出良好的泛化能力，能够推广到训练时未见过的任务类型和语言。

### 核心直觉

核心变化是将语言模型的优化目标从统计建模转向人类偏好建模。传统方法优化'什么文本在统计上可能出现'，而InstructGPT优化'什么输出人类会偏好'。这种转变通过引入人类反馈循环实现：先让人类示范期望行为，再让人类评判输出质量，最后用强化学习将这些评判转化为模型的学习信号。本质上是用人类的价值判断替代了统计规律作为模型的指导原则。

## Part III：证据与局限

实验证据强有力地支持了方法的有效性。在API提示分布上，175B InstructGPT的输出在85±3%的时间里被偏好于GPT-3的输出，1.3B InstructGPT甚至优于175B GPT-3。模型在具体行为维度上也有显著改进：更好地遵循明确约束、减少幻觉现象、输出更适合客户助手的语言风格。重要的是，这些改进能够泛化到未参与训练的标注者，表明模型学到的不仅仅是特定标注者的偏好。然而，该方法也存在局限性：对齐成本虽然相对预训练适中，但仍需要大量人工标注；模型仍会犯简单错误；在某些公开NLP数据集上性能略有下降；标注者群体的代表性可能影响对齐效果的普适性。


### Delta Statement

核心变化是将语言模型的优化目标从统计建模转向人类偏好建模。传统方法优化'什么文本在统计上可能出现'，而InstructGPT优化'什么输出人类会偏好'。这种转变通过引入人类反馈循环实现：先让人类示范期望行为，再让人类评判输出质量，最后用强化学习将这些评判转化为模型的学习信号。本质上是用人类的价值判断替代了统计规律作为模型的指导原则。
