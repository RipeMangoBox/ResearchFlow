---
title: LFA
type: method
method_type: mechanism_family
maturity: seed
parent_method: null
n_papers: 1
primary_tasks:
- Classification
---

# LFA

LFA 是一种带有异方差噪声假设的隐因子分析方法，用于特征选择任务，发表于 CVPR 2025。传统隐因子模型通常假设噪声服从同方差分布，而 LFA 通过建模噪声的异方差特性，实现了更鲁棒的特征重要性评估与选择。

在计算机视觉等高维数据分析场景中，不同特征维度往往具有不同的噪声水平。LFA 的异方差建模能够区分"信号变异"与"噪声变异"，从而避免将高噪声维度的随机波动误判为有效信号。

**研究领域**: Classification

## 使用本方法的论文 (1)

| 论文 | 会议 | 年份 |
|------|------|------|
| [[P__异方差潜在因子模型的SNR特征选_LFA_(Latent_Fact]] | CVPR | 2025 |

