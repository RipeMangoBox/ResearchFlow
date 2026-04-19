---
title: 'KTO: Model Alignment as Prospect Theoretic Optimization'
type: paper
paper_id: P__KTO_Model_Alignment_as_Prospect_Theoretic_Optimization
aliases:
- KTO_Model_Alignment_as_Prospect_Theoretic_Optimization
year: 2024
venue: arXiv (Cornell University)
paper_level: A
frame: rl_standard
changed_slots:
- 损失函数设计
- 人类效用建模
- 数据需求
structurality_score: 0.8
keep_score: 0.29
open_code: true
concepts:
- '[[C__human_aware_loss_optimization]]'
bottleneck:
- '[[B__偏好似然与人类真实效用之间的根本性偏差]]'
lineage: []
same_family_papers: []
paper_link: https://arxiv.org/abs/2402.01306
code_url: https://github.com/FoundationAgents/awesome-foundation-agents
---
# KTO: Model Alignment as Prospect Theoretic Optimization

> 基于 `rl_standard`，改了 `损失函数设计`, `人类效用建模`, `数据需求`,
> 属于 [[C__human_aware_loss_optimization]],
> 目标是缓解 [[B__偏好似然与人类真实效用之间的根本性偏差]]

## 相对 baseline 改了什么

> KTO的核心洞察是认识到人类偏好表达与真实效用感知之间的差异。传统方法假设最大化偏好似然等同于最大化人类效用，但前景理论告诉我们人类的价值感知是非线性的、参考点依赖的。KTO通过直接建模这种非线性价值函数，更准确地捕捉了人类的真实效用，从而在对齐任务上取得更好的效果。这种从'拟合偏好'到'优化效用'的转变是根本性的。


## 关键公式

- $$L_{KTO}(\pi_\theta, \pi_{ref}) = E_{x,y\sim D}[\omega_y v(r_\theta(x,y) - z_0)]$$
  - 损失函数设计：KTO的核心损失函数，直接优化人类效用而非偏好似然
- $$v(z; \lambda, \alpha, z_0) = \begin{cases} (z-z_0)^\alpha & \text{if } z \geq z_0 \\ -\lambda(z_0-z)^\alpha & \text{if } z < z_0 \end{cases}$$
  - 价值函数：Kahneman-Tversky价值函数，体现损失厌恶和风险厌恶
- $$L_{DPO}(\pi_\theta, \pi_{ref}) = E_{x,y_w,y_l\sim D}[-\log \sigma(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)})]$$
  - 基线方法：DPO损失函数，最大化偏好似然而非直接优化效用

## 关键图表

- **Table 2**: KTO在多个基准测试上的表现，特别是GSM8K上比DPO高13.5分
  - 证据：KTO在数学推理任务上显著优于DPO
- **Figure 3**: 1B到30B规模下KTO vs DPO的胜率对比
  - 证据：KTO在多个规模下匹配或超越DPO性能
- **Table 3 (Mistral-7B结果)**: Mistral-7B + KTO胜率0.652，超过DPO的0.600
  - 证据：KTO仅用二元信号就能超越基于偏好的方法

## 阅读建议

> **必读 baseline**。先理解此论文建立的标准框架，再看后续改进。

## 详细分析

# KTO: Model Alignment as Prospect Theoretic Optimization

## Part I：问题与挑战

现有的大语言模型对齐方法（如DPO）通过最大化偏好的对数似然来训练模型，但这种方法存在根本性问题：它并不直接优化人类的实际效用。人类的决策行为遵循Kahneman-Tversky的前景理论，表现出损失厌恶、风险厌恶等认知偏差，而当前方法忽略了这些人类认知特性。此外，基于偏好的方法需要成对的比较数据，数据收集成本高，且在处理噪声和不一致偏好时表现不佳。这导致对齐效果与人类真实效用存在偏差，特别是在数学推理等复杂任务上性能受限。

## Part II：方法与洞察

KTO提出了一个根本性的范式转变：从最大化偏好似然转向直接最大化人类效用。核心创新是引入Kahneman-Tversky价值函数v(z; λ, α, z0)，其中z0是参考点，λ控制损失厌恶程度，α控制风险厌恶程度。KTO的损失函数为L_KTO = E[ω_y v(r_θ(x,y) - z_0)]，直接建模人类对生成内容的主观价值感知。与DPO需要成对偏好数据不同，KTO只需要二元信号（好/坏），大大简化了数据需求。方法设计体现了前景理论的三个关键特性：参考点依赖（相对于z0评估得失）、损失厌恶（损失的负面影响大于等量收益的正面影响）、风险厌恶（价值函数的凹凸性）。这种设计使KTO能够更好地处理数据不平衡、噪声偏好和不一致反馈，理论分析表明KTO在处理矛盾偏好时具有更好的最坏情况保证。

### 核心直觉

KTO的核心洞察是认识到人类偏好表达与真实效用感知之间的差异。传统方法假设最大化偏好似然等同于最大化人类效用，但前景理论告诉我们人类的价值感知是非线性的、参考点依赖的。KTO通过直接建模这种非线性价值函数，更准确地捕捉了人类的真实效用，从而在对齐任务上取得更好的效果。这种从'拟合偏好'到'优化效用'的转变是根本性的。

## Part III：证据与局限

实验证据强有力地支持了KTO的有效性。在GSM8K数学推理任务上，KTO比DPO高出13.5个百分点（53.5% vs 40.0%），这是最显著的改进。在1B到30B规模的模型上，KTO匹配或超越了基于偏好的方法，特别是在7B和30B规模上差异具有统计显著性（p < 0.01）。消融实验验证了设计选择的重要性：移除参考点z0导致BBH和GSM8K性能分别下降3.6和4.0个百分点。KTO展现出强大的数据效率，即使丢弃90%的理想数据仍能优于DPO。在Mistral-7B上，仅使用每个输入对应一个输出的设置下，KTO仍超越DPO和官方指令调优模型。然而，实验主要集中在英文基准测试上，缺乏多语言和多领域的全面评估。


### Delta Statement

KTO的核心洞察是认识到人类偏好表达与真实效用感知之间的差异。传统方法假设最大化偏好似然等同于最大化人类效用，但前景理论告诉我们人类的价值感知是非线性的、参考点依赖的。KTO通过直接建模这种非线性价值函数，更准确地捕捉了人类的真实效用，从而在对齐任务上取得更好的效果。这种从'拟合偏好'到'优化效用'的转变是根本性的。
