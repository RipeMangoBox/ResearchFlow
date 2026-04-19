---
title: Specific versus General Principles for Constitutional AI (CAI v2)
type: paper
paper_id: P__Specific_versus_General_Principles_for_Constitutional_AI_CAI_v2
aliases:
- Specific_versus_General_Principles_for_Constitutional_AI_CAI_v2
year: 2023
venue: ''
paper_level: C
frame: rl_standard
changed_slots:
- feedback_source
- annotation_granularity
- data_scale
structurality_score: 0.3
keep_score: 0.26
open_code: false
concepts:
- '[[C__reinforcement_learning_from_ai_feedback]]'
bottleneck:
- '[[B__人类反馈获取的规模和成本瓶颈]]'
lineage: []
same_family_papers: []
paper_link: https://arxiv.org/abs/2310.01377
---
# Specific versus General Principles for Constitutional AI (CAI v2)

> 基于 `rl_standard`，改了 `feedback_source`, `annotation_granularity`, `data_scale`,
> 属于 [[C__reinforcement_learning_from_ai_feedback]],
> 目标是缓解 [[B__人类反馈获取的规模和成本瓶颈]]

## 相对 baseline 改了什么

> 核心洞察是用 AI 反馈突破人类反馈的规模瓶颈。通过 GPT-4 的强大能力，可以自动生成大规模、多样化的偏好数据，同时细粒度标注比整体评分提供更精确的学习信号。AI 反馈的一致性和可扩展性使得开源模型能够获得与闭源模型相当的对齐训练数据，从而缩小性能差距。关键在于规模和多样性是反馈数据发挥作用的核心因素。


## 关键图表

- **Table 2**: 奖励模型在人类偏好基准上的准确率对比
  - 证据：ULTRAFEEDBACK 训练的奖励模型优于开源基线模型
- **Figure 4**: UltraLM-13B-PPO 在不同主题上相对 ChatGPT 的表现
  - 证据：AI 反馈训练的模型在多数任务上超越 ChatGPT
- **Table 6**: PPO 前后模型在能力基准上的精确匹配分数
  - 证据：RLAIF 对基础模型能力影响较小

## 阅读建议

> **插件型改进**。可快速浏览，重点看改了哪个 slot 和实验对比。

## 详细分析

# Specific versus General Principles for Constitutional AI (CAI v2)

## Part I：问题与挑战

现有的人类反馈学习方法面临严重的可扩展性瓶颈。获取大规模、高质量的人类偏好标注受限于时间、人力成本和人类能力边界，导致当前数据集规模小、主题覆盖有限。这进一步阻碍了开源社区的对齐研究进展。具体表现为：1）现有偏好数据集要么专注特定任务（如摘要、问答），无法提升通用对话模型；2）数据集规模较小或仅提供粗粒度的社区投票偏好；3）缺乏大规模通用偏好数据集和细粒度标注。这些限制使得开源模型与闭源模型在对话能力上存在显著差距，亟需一种可扩展的替代方案来突破人类反馈的固有局限。

## Part II：方法与洞察

本文提出用大规模 AI 反馈替代人类反馈的方案，核心改变是数据构建策略的系统性重设计。方法包含三个关键组件：1）数据规模与多样性扩展 - 构建包含 100 万条 GPT-4 反馈的 ULTRAFEEDBACK 数据集，覆盖 25 万个用户-助手对话，涵盖广泛主题和交互类型；2）细粒度标注设计 - 不同于传统整体评分，采用多维度细粒度评估（有用性、诚实性、指令遵循、无害性），同时对多个回复进行批量评估以提供交叉参考，减少标注不一致性；3）偏差缓解技术 - 应用一系列技术手段减少 AI 标注中的位置偏差、长度偏差等系统性问题。在应用层面，基于 ULTRAFEEDBACK 训练奖励模型 UltraRM，然后通过 best-of-n 采样和 PPO 强化学习对齐 LLaMA 模型。这种方法本质上是将人类反馈学习范式中的'人类标注'槽位替换为'AI 标注'，同时大幅提升数据规模和标注粒度。

### 核心直觉

核心洞察是用 AI 反馈突破人类反馈的规模瓶颈。通过 GPT-4 的强大能力，可以自动生成大规模、多样化的偏好数据，同时细粒度标注比整体评分提供更精确的学习信号。AI 反馈的一致性和可扩展性使得开源模型能够获得与闭源模型相当的对齐训练数据，从而缩小性能差距。关键在于规模和多样性是反馈数据发挥作用的核心因素。

## Part III：证据与局限

实验证据支持 AI 反馈的有效性：UltraRM 在四个人类偏好基准上平均准确率达到 71.0%，超越所有开源基线模型，在 WebGPT 等未见过的数据集上也表现出良好泛化能力。UltraLM-13B-PPO 在 29 个主题中的 22 个超越 ChatGPT，平均达到 100.3%，特别在写作相关任务上表现突出。细粒度标注相比整体评分在 WebGPT 上表现更好，验证了标注设计的有效性。然而存在明显局限：模型在数学和代码相关任务上仍落后于 ChatGPT，RLAIF 对基础模型能力提升有限（约 1 个百分点），且仍落后于更大的闭源 LLaMA2 奖励模型。这表明 AI 反馈虽然可行，但在某些复杂推理任务上仍有不足。


### Delta Statement

核心洞察是用 AI 反馈突破人类反馈的规模瓶颈。通过 GPT-4 的强大能力，可以自动生成大规模、多样化的偏好数据，同时细粒度标注比整体评分提供更精确的学习信号。AI 反馈的一致性和可扩展性使得开源模型能够获得与闭源模型相当的对齐训练数据，从而缩小性能差距。关键在于规模和多样性是反馈数据发挥作用的核心因素。
