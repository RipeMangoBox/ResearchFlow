---
title: Visual Fourier Prompt Tuning
type: method
method_type: mechanism_family
maturity: seed
parent_method: null
n_papers: 1
primary_tasks:
- Classification
- Self-Supervised Learning
- Domain Adaptation
---

# Visual Fourier Prompt Tuning

Visual Fourier Prompt Tuning（VFPT）是一种在频域空间进行提示学习的视觉模型微调方法。该方法突破了传统提示调优仅在空间域操作的局限，利用傅里叶变换将可学习的提示嵌入到频率分量中，从而实现更高效的参数更新与特征调制。

该方法在 NeurIPS 2024 上提出，面向视觉识别任务的模型适配场景。通过在傅里叶域引入可训练参数，VFPT 能够在保持预训练模型冻结的同时，以极少的参数量实现有效的任务迁移，为视觉基础模型的高效适配提供了新的技术路径。

**研究领域**: Classification, Self-Supervised Learning, Domain Adaptation

## 方法谱系

- **子方法/变体** (1):
  - [[M__PerReg_-_PerReg+]] — 

## 使用本方法的论文 (1)

| 论文 | 会议 | 年份 |
|------|------|------|
| [[P__视觉傅里叶提示微调VFPT_Visual_Fourier_P]] | NeurIPS | 2024 |

