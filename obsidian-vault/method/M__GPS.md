---
title: GPS
type: method
method_type: mechanism_family
maturity: seed
parent_method: null
n_papers: 1
primary_tasks:
- Classification
- Medical Imaging
- Semantic Segmentation
---

# GPS

GPS（Gradient-based Parameter Selection，基于梯度的参数选择）是一种高效的模型微调方法，通过分析参数梯度信号来智能选择需要更新的参数子集，从而减少计算开销。该方法在 CVPR 2024 的论文《Gradient-based Parameter Selection for Efficient Fine-Tuning》中提出。

与全参数微调或固定结构的 LoRA 等方法不同，GPS 利用梯度信息动态识别对目标任务最重要的参数，实现了在保持性能的同时显著提升微调效率的目标。

**研究领域**: Classification, Medical Imaging, Semantic Segmentation

## 使用本方法的论文 (1)

| 论文 | 会议 | 年份 |
|------|------|------|
| [[P__基于梯度选择的参数高效微调GPS_GPS_(Gradient-ba]] | CVPR | 2024 |

