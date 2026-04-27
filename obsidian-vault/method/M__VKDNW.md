---
title: VKDNW
type: method
method_type: mechanism_family
maturity: seed
parent_method: null
n_papers: 2
primary_tasks:
- Neural Architecture Search
- Classification
---

# VKDNW

VKDNW（Variance of Knowledge of Deep Network Weights）是一种无需训练的神经网络架构搜索（NAS）评估指标。该方法在 CVPR 2025 上提出，通过分析深度网络权重的知识方差来预测网络性能，从而避免了传统 NAS 中耗时的训练过程。

作为训练自由（training-free）NAS 方法家族的新成员，VKDNW 利用权重初始化阶段的统计特性进行架构评分，大幅降低了神经架构搜索的计算成本，使其在资源受限环境下的大规模架构探索成为可能。

**研究领域**: Neural Architecture Search, Classification

## 使用本方法的论文 (2)

| 论文 | 会议 | 年份 |
|------|------|------|
| [[P__基于Fisher信息矩阵特征值熵_VKDNW_(Variance_]] | CVPR | 2025 |
| [[P__基于Fisher谱熵的无训练神经_VKDNW_(Variance_]] | CVPR | 2025 |

## 本方法的方法学基线

- AZ-NAS — _Zero-cost NAS proxy method likely compared against as a baseline approach_

