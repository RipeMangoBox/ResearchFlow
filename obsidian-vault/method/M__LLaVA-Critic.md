---
title: LLaVA-Critic
type: method
method_type: mechanism_family
maturity: seed
parent_method: null
n_papers: 2
primary_tasks:
- Benchmark / Evaluation
- Visual Reasoning
- Cross-Modal Matching
---

# LLaVA-Critic

LLaVA-Critic 是一种专门训练用于评估多模态模型的评判模型，通过学习生成对视觉-语言模型输出的评价和反馈，填补了自动评估领域的空白。该模型在 CVPR 2025 的论文《LLaVA-Critic: Learning to Evaluate Multimodal Models》中提出。

作为 LLaVA 家族的扩展，LLaVA-Critic 不仅具备多模态理解能力，更重要的是能够将这种理解转化为对模型输出质量的判断，为多模态模型的开发、比较和迭代优化提供了自动化的评估工具。

**研究领域**: Benchmark / Evaluation, Visual Reasoning, Cross-Modal Matching

## 使用本方法的论文 (2)

| 论文 | 会议 | 年份 |
|------|------|------|
| [[P__多模态模型自训练评估器LLaVA_LLaVA-Critic]] | CVPR | 2025 |
| [[P__多模态模型评判器LLaVA-Cr_LLaVA-Critic]] | CVPR | 2025 |

## 本方法的方法学基线

- Optimal Test-Time Compute Scaling via Verifier-Guided Search — _Core algorithmic idea about test-time compute scaling likely inspires the paper'_
- MLLM-as-a-Judge Benchmark — _Core methodology for using MLLMs as judges, directly relevant to paper's approac_
- CSR — _Self-rewarding VLM with calibration; closely related method likely compared agai_
- BPO — _Bootstrapped preference optimization for MLLMs; direct methodological competitor_

