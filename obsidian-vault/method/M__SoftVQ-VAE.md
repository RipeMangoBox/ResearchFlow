---
title: SoftVQ-VAE
type: method
method_type: mechanism_family
maturity: seed
parent_method: null
n_papers: 1
primary_tasks:
- Image Generation
- Compression
---

# SoftVQ-VAE

SoftVQ-VAE 是一种高效的单维连续型分词器（tokenizer），采用软向量量化（Soft Vector Quantization）机制学习紧凑的隐变量表示。该方法针对自回归生成模型和扩散模型中的离散/连续 token 化需求，优化了传统 VQ-VAE 中硬量化带来的梯度传播障碍和码本崩溃问题。

发表于 CVPR 2025 的 SoftVQ-VAE 在保持离散表示结构优势的同时，通过连续松弛实现了更稳定的训练动态和更高的重建质量。

**研究领域**: Image Generation, Compression

## 使用本方法的论文 (1)

| 论文 | 会议 | 年份 |
|------|------|------|
| [[P__SoftVQ-VAE：高效1维连_SoftVQ-VAE]] | CVPR | 2025 |

