---
title: Spiking Transformer
type: method
method_type: mechanism_family
maturity: seed
parent_method: null
n_papers: 1
primary_tasks:
- Classification
- Compression
- Self-Supervised Learning
---

# Spiking Transformer

Spiking Transformer是CVPR 2025提出的脉冲神经网络（SNN）与Transformer架构的融合创新，其核心贡献在于设计了**纯加法脉冲自注意力机制（Addition-Only Spiking Self-Attention）**，在保持Transformer强大建模能力的同时，实现了神经形态硬件友好的超低能耗计算。

传统Transformer中的点积注意力涉及大量乘法-累加运算，能耗高昂；而标准SNN虽具备事件驱动的高能效特性，但难以处理长程依赖。Spiking Transformer通过将注意力计算重构为纯加法操作，并结合脉冲神经元的二进制激活特性，为边缘AI和神经形态芯片部署开辟了新的技术路径。

**研究领域**: Classification, Compression, Self-Supervised Learning

## 使用本方法的论文 (1)

| 论文 | 会议 | 年份 |
|------|------|------|
| [[P__纯加法脉冲自注意力Transfo_Spiking_Transfor]] | CVPR | 2025 |

