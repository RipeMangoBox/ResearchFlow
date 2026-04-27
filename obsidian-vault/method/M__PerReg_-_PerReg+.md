---
title: PerReg / PerReg+
type: method
method_type: mechanism_family
maturity: seed
parent_method: null
n_papers: 1
primary_tasks:
- Autonomous Driving
- Self-Supervised Learning
- Domain Adaptation
---

# PerReg / PerReg+

PerReg / PerReg+ 是一种面向轨迹预测任务的表示学习方法，通过双层表示学习与自适应提示机制提升模型的泛化能力。该方法旨在解决传统轨迹预测模型在跨场景、跨数据集迁移时性能下降的问题。

该方法的核心思想在于分离场景级与代理级（agent-level）的表示，并通过可学习的提示向量动态适应不同测试环境，从而实现无需重新训练的零样本或少样本泛化。

**研究领域**: Autonomous Driving, Self-Supervised Learning, Domain Adaptation

## 使用本方法的论文 (1)

| 论文 | 会议 | 年份 |
|------|------|------|
| [[P__自适应提示的双层轨迹预测表示学习_PerReg_-_PerReg+]] | CVPR | 2025 |

## 本方法的方法学基线

- Visual Fourier Prompt Tuning — _Visual Prompt Tuning is foundational parameter-efficient fine-tuning method for _

