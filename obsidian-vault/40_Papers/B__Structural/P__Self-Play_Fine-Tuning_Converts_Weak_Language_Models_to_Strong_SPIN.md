---
title: Self-Play Fine-Tuning Converts Weak Language Models to Strong (SPIN)
type: paper
paper_id: P__Self-Play_Fine-Tuning_Converts_Weak_Language_Models_to_Strong_SPIN
aliases:
- Self-Play_Fine-Tuning_Converts_Weak_Language_Models_to_Strong_SPIN
year: 2024
venue: ''
paper_level: B
frame: rl_standard
changed_slots:
- reward_model
- training_loop
- preference_generation
structurality_score: 0.6
keep_score: 0.29
open_code: false
concepts:
- '[[C__iterative_preference_optimization]]'
bottleneck:
- '[[B__奖励模型冻结导致的性能天花板问题]]'
lineage: []
same_family_papers: []
paper_link: https://arxiv.org/abs/2401.10020
---
# Self-Play Fine-Tuning Converts Weak Language Models to Strong (SPIN)

> 基于 `rl_standard`，改了 `reward_model`, `training_loop`, `preference_generation`,
> 属于 [[C__iterative_preference_optimization]],
> 目标是缓解 [[B__奖励模型冻结导致的性能天花板问题]]

## 相对 baseline 改了什么

> 核心洞察是打破奖励模型冻结的传统范式，让模型在训练过程中同时改进指令跟随和奖励建模能力。通过将奖励建模重新框架为指令跟随任务，实现了统一模型架构下的多任务学习。这种设计允许模型为自己生成越来越高质量的训练信号，突破了人类标注数据的性能瓶颈，开启了模型自我改进的可能性。


## 关键公式

- $$\text{DPO Loss: } \mathcal{L}_{DPO} = -\mathbb{E}_{(x,y_w,y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]$$
  - preference optimization：迭代DPO训练的核心损失函数，用于从自生成的偏好对中学习

## 关键图表

- **Figure 3**: Head-to-head win rates showing iterative improvement across M1, M2, M3
  - 证据：自奖励训练确实能逐步提升指令跟随能力
- **Table 1**: AlpacaEval 2.0 results showing M3 outperforms Claude 2, Gemini Pro, GPT-4 0613
  - 证据：第三次迭代模型在标准基准上超越了多个现有系统
- **Figure 4**: Performance breakdown by instruction category
  - 证据：自奖励方法在大多数任务类别上都有提升，但数学和逻辑推理除外

## 阅读建议

> **结构性改进**。建议先读 baseline，再看本文如何修改核心 slot。

## 详细分析

# Self-Play Fine-Tuning Converts Weak Language Models to Strong (SPIN)

## Part I：问题与挑战

现有的强化学习人类反馈(RLHF)方法面临两个核心瓶颈：首先，奖励模型从人类偏好中学习，性能上限受限于人类水平；其次，奖励模型在训练过程中保持冻结状态，无法在LLM训练期间持续改进。这种设计导致了一个根本性问题——要实现超人智能体，需要超人反馈信号，但当前方法无法突破人类性能天花板。传统方法将指令跟随和奖励建模分离为不同模型，缺乏任务间的迁移学习，限制了系统的整体改进能力。

## Part II：方法与洞察

SPIN提出自奖励语言模型(Self-Rewarding Language Models)，核心创新是让同一个模型同时承担指令跟随和奖励建模两个角色。方法采用迭代DPO框架：(1)自指令创建阶段，模型为新生成的提示创建候选回答，然后通过LLM-as-a-Judge提示为自己的生成内容打分；(2)指令跟随训练阶段，从生成数据中选择偏好对，使用DPO训练下一轮模型。关键洞察是将奖励建模视为指令跟随任务，通过多任务学习实现两种能力的协同提升。与传统方法不同，这里的奖励模型不再冻结，而是在每次迭代中持续改进，形成良性循环——更好的奖励模型生成更高质量的偏好数据，进而训练出更强的下一轮模型。实验从Llama 2 70B开始，使用Open Assistant种子数据，进行三轮迭代训练。

### 核心直觉

核心洞察是打破奖励模型冻结的传统范式，让模型在训练过程中同时改进指令跟随和奖励建模能力。通过将奖励建模重新框架为指令跟随任务，实现了统一模型架构下的多任务学习。这种设计允许模型为自己生成越来越高质量的训练信号，突破了人类标注数据的性能瓶颈，开启了模型自我改进的可能性。

## Part III：证据与局限

实验证据显示方法的有效性：M2相比M1在头对头评估中获得55.5%胜率，M3进一步提升至47.7%胜率超越M2。在AlpacaEval 2.0基准上，M3达到20.44%相对GPT-4 Turbo的胜率，超越Claude 2、Gemini Pro等现有系统。细粒度分析表明改进主要体现在大多数任务类别上，但数学和逻辑推理除外，说明方法主要帮助模型更好利用现有知识而非获得新能力。重要发现是添加评估微调(EFT)任务不影响指令跟随性能，证明两种能力可以协调发展。局限性包括：仅进行了三轮迭代的有限实验，观察到生成长度增加可能导致奖励黑客攻击，缺乏安全性评估，主要依赖单一基准测试。


### Delta Statement

核心洞察是打破奖励模型冻结的传统范式，让模型在训练过程中同时改进指令跟随和奖励建模能力。通过将奖励建模重新框架为指令跟随任务，实现了统一模型架构下的多任务学习。这种设计允许模型为自己生成越来越高质量的训练信号，突破了人类标注数据的性能瓶颈，开启了模型自我改进的可能性。
