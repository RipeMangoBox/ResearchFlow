---
title: Laplace-LoRA
type: method
method_type: mechanism_family
maturity: seed
parent_method: null
n_papers: 1
primary_tasks:
- Text Generation
---

# Laplace-LoRA

Laplace-LoRA 是一种贝叶斯低秩适配方法，将拉普拉斯近似（Laplace Approximation）引入 LoRA 框架，为大型语言模型的微调提供不确定性量化能力。该方法在 ICLR 2024 的论文《Bayesian Low-rank Adaptation for Large Language Models》中提出。

通过将 LoRA 的参数视为随机变量并对其后验分布进行近似推断，Laplace-LoRA 在保持参数高效微调优势的同时，使模型能够输出预测的不确定性估计，增强了模型在关键应用中的可靠性。

**研究领域**: Text Generation

## 方法谱系

- **子方法/变体** (1):
  - [[M__MONA]] — 

## 使用本方法的论文 (1)

| 论文 | 会议 | 年份 |
|------|------|------|
| [[P__大语言模型的贝叶斯低秩适应_Laplace-LoRA]] | ICLR | 2024 |

