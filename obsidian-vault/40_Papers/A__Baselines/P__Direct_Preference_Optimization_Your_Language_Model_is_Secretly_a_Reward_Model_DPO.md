---
title: 'Direct Preference Optimization: Your Language Model is Secretly a Reward Model (DPO)'
type: paper
paper_id: P__Direct_Preference_Optimization_Your_Language_Model_is_Secretly_a_Reward_Model_DPO
aliases:
- Direct_Preference_Optimization_Your_Language_Model_is_Secretly_a_Reward_Model_DPO
year: 2023
venue: ''
paper_level: A
frame: rl_standard
changed_slots:
- 训练目标函数
- 优化算法
- 奖励建模方式
structurality_score: 0.8
keep_score: 0.26
open_code: true
concepts:
- '[[C__direct_preference_optimization]]'
bottleneck:
- '[[B__RLHF训练的复杂性和不稳定性瓶颈]]'
lineage: []
same_family_papers: []
paper_link: https://arxiv.org/abs/2305.18290
code_url: https://github.com/Hannibal046/Awesome-LLM
---
# Direct Preference Optimization: Your Language Model is Secretly a Reward Model (DPO)

> 基于 `rl_standard`，改了 `训练目标函数`, `优化算法`, `奖励建模方式`,
> 属于 [[C__direct_preference_optimization]],
> 目标是缓解 [[B__RLHF训练的复杂性和不稳定性瓶颈]]

## 相对 baseline 改了什么

> DPO的核心洞察是认识到奖励模型和最优策略之间存在一一对应关系，可以通过重参数化直接从策略中提取隐式奖励。这避免了传统RLHF中奖励模型训练误差传播到策略优化的问题，同时将复杂的RL优化简化为稳定的分类学习，从根本上重新思考了偏好学习的范式。


## 关键公式

- $$L_{DPO}(\pi_\theta, \pi_{ref}) = -E_{\tau,y_1,...,y_K,x\sim D}\left[\log \prod_{k=1}^K \frac{\exp\left(\beta \log \frac{\pi_\theta(y_{\tau(k)}|x)}{\pi_{ref}(y_{\tau(k)}|x)}\right)}{\sum_{j=k}^K \exp\left(\beta \log \frac{\pi_\theta(y_{\tau(j)}|x)}{\pi_{ref}(y_{\tau(j)}|x)}\right)}\right]$$
  - 训练目标函数：DPO的核心损失函数，直接优化策略而非奖励模型
- $$\nabla_\theta L_{DPO}(\pi_\theta; \pi_{ref}) = -E_{(x,y_w,y_l)\sim D}\left[\beta\sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right) \left[\nabla_\theta \log \pi(y_w | x) - \nabla_\theta \log \pi(y_l | x)\right]\right]$$
  - 梯度计算：展示了DPO如何通过简单的分类损失实现偏好优化
- $$r(x, y) = \beta \log \frac{\pi(y|x)}{\pi_{ref}(y|x)}$$
  - 奖励模型重参数化：将奖励函数表示为策略与参考模型的对数比值

## 关键图表

- **Figure 3**: Anthropic-HH对话任务的胜率对比和训练过程中胜率演化
  - 证据：DPO在对话任务上超越基线方法的实验证据
- **Table 2**: Reddit TL;DR摘要任务的胜率结果
  - 证据：DPO在摘要任务上与PPO-based RLHF相当或更好的性能

## 阅读建议

> **必读 baseline**。先理解此论文建立的标准框架，再看后续改进。

## 详细分析

# Direct Preference Optimization: Your Language Model is Secretly a Reward Model (DPO)

## Part I：问题与挑战

现有的RLHF方法通过人类反馈训练语言模型存在复杂性和不稳定性问题。传统RLHF需要先训练一个奖励模型来反映人类偏好，然后使用强化学习优化语言模型策略以最大化估计奖励，同时避免偏离原始模型太远。这个过程涉及多个模型的训练、在训练循环中从LM策略采样，计算成本高昂且需要大量超参数调优。此外，RL训练过程往往不稳定，容易出现训练崩溃或性能退化的问题。

## Part II：方法与洞察

DPO通过重新参数化奖励模型实现了从人类偏好直接优化语言模型的方法，完全绕过了强化学习。核心洞察是将奖励函数表示为策略与参考模型的对数比值：r(x,y) = β log(π(y|x)/π_ref(y|x))。基于这个重参数化，DPO推导出了最优策略的闭式解，并将RLHF的约束优化问题转化为简单的二元分类损失。具体而言，DPO损失函数直接优化偏好对的相对概率，增加偏好响应的对数概率同时降低非偏好响应的对数概率，并通过动态的重要性权重防止模型退化。这种方法消除了显式奖励建模、RL采样和复杂超参数调优的需求，将整个偏好学习过程简化为标准的监督学习。

### 核心直觉

DPO的核心洞察是认识到奖励模型和最优策略之间存在一一对应关系，可以通过重参数化直接从策略中提取隐式奖励。这避免了传统RLHF中奖励模型训练误差传播到策略优化的问题，同时将复杂的RL优化简化为稳定的分类学习，从根本上重新思考了偏好学习的范式。

## Part III：证据与局限

实验在情感控制、摘要和对话三个任务上验证了DPO的有效性。在情感控制任务上，DPO超越了PPO-based RLHF的性能。在Reddit TL;DR摘要任务上，DPO与PPO方法性能相当或更好。在Anthropic-HH对话任务上，DPO是唯一超越chosen responses的方法。然而，实验存在一些局限：模型规模仅验证到6B参数，缺少更大规模模型的验证；评估主要依赖GPT-4作为代理，可能存在偏差；某些基线方法的实现细节不够详细，影响比较的公平性。


### Delta Statement

DPO的核心洞察是认识到奖励模型和最优策略之间存在一一对应关系，可以通过重参数化直接从策略中提取隐式奖励。这避免了传统RLHF中奖励模型训练误差传播到策略优化的问题，同时将复杂的RL优化简化为稳定的分类学习，从根本上重新思考了偏好学习的范式。
