---
title: VMamba
type: method
method_type: mechanism_family
maturity: seed
parent_method: null
n_papers: 1
primary_tasks:
- Classification
- Object Detection
- Semantic Segmentation
---

# VMamba

VMamba 是一种将状态空间模型（State Space Model, SSM）扩展至视觉识别任务的深度学习方法。该方法基于 Mamba 架构的线性复杂度序列建模优势，通过设计适合图像层次化特征提取的扫描机制，在保持全局感受野的同时实现与输入尺寸成线性关系的计算复杂度。

发表于 NeurIPS 2024 的 VMamba 代表了视觉骨干网络设计的重要转向——从基于注意力的 Transformer（二次复杂度）和基于卷积的 CNN，迈向基于选择性状态空间的第三代架构范式。

**研究领域**: Classification, Object Detection, Semantic Segmentation

## 使用本方法的论文 (1)

| 论文 | 会议 | 年份 |
|------|------|------|
| [[P__视觉状态空间模型VMamba_VMamba]] | NeurIPS | 2024 |

