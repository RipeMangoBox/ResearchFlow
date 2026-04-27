---
title: InfoBatch
type: method
method_type: mechanism_family
maturity: seed
parent_method: null
n_papers: 1
primary_tasks:
- Self-Supervised Learning
- Classification
- Semantic Segmentation
---

# InfoBatch

InfoBatch 是一种用于加速深度学习训练的无损动态数据剪枝方法，发表于 ICLR 2024。该方法通过在每个训练周期动态评估样本的重要性，优先选择对模型更新贡献更大的数据子集，从而在保持模型性能的同时显著减少训练迭代所需的计算量。

与传统的静态数据剪枝或随机采样策略不同，InfoBatch 的核心优势在于其"无损"特性——即通过精心设计的采样机制，确保被剪枝数据的梯度信息仍能被无偏地估计，避免因数据丢弃导致的性能下降。这一特性使其在实际应用中具有较高的可靠性。

**研究领域**: Self-Supervised Learning, Classification, Semantic Segmentation

## 使用本方法的论文 (1)

| 论文 | 会议 | 年份 |
|------|------|------|
| [[P__无偏动态数据剪枝无损加速训练_InfoBatch]] | ICLR | 2024 |

