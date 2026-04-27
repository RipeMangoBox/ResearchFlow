---
title: C-TPT
type: method
method_type: mechanism_family
maturity: seed
parent_method: null
n_papers: 1
primary_tasks:
- Self-Supervised Learning
- Domain Adaptation
---

# C-TPT

C-TPT（Calibrated Test-Time Prompt Tuning）是一种针对视觉-语言模型的测试时提示调优方法，发表于 ICLR 2024。该方法通过文本特征分散度（text feature dispersion）的校准机制，解决现有测试时自适应方法中的置信度校准问题。

在零样本迁移和分布偏移场景下，预训练的视觉-语言模型（如 CLIP）往往面临预测置信度不可靠的挑战。C-TPT 在不访问源域训练数据的前提下，通过优化提示（prompt）表示并引入分散度约束，实现更准确的不确定性估计和类别决策。

**研究领域**: Self-Supervised Learning, Domain Adaptation

## 使用本方法的论文 (1)

| 论文 | 会议 | 年份 |
|------|------|------|
| [[P__通过文本特征分散校准测试时提示微_C-TPT_(Calibrate]] | ICLR | 2024 |

