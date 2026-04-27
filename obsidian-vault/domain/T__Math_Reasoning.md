---
title: Math Reasoning
type: task
domain: task
name_zh: 数学推理
---

# Math Reasoning (数学推理)

数学推理（Math Reasoning）是评估人工智能系统抽象思维和逻辑推理能力的关键任务，要求模型理解数学问题陈述，运用算术、代数、几何、微积分等知识进行多步推导，最终得出正确答案。

该任务对当前大语言模型构成显著挑战，因其不仅需要正确的计算能力，更依赖严密的逻辑链条、符号操作和定理应用。数学推理能力的提升被视为通往通用人工智能的重要里程碑，近年来涌现出专门的数学大模型和推理优化技术。

## 代表方法

- Optimal Test-Time Compute Scaling via Verifier-Guided Search (1 篇)
- GNN-SPAI (1 篇)
- PT-MoE (1 篇)
- Chain of Thought (CoT) for Theoretical Expressiveness Analysis (1 篇)
- UFO-RL (1 篇)
- MathNet (1 篇)
- Causal-R (1 篇)
- Stratagem (1 篇)

## 常用数据集

- Li et al. (4 篇)
- Häusner et al. (3 篇)
- GSM8K (2 篇)
- AIME 2024 (2 篇)
- MRQA (2 篇)
- MATH-500 (1 篇)
- MATH-500 by difficulty quartile (1 篇)
- Heat problem GPU (1 篇)

## 分布

- 年份: 2024 (1) · 2025 (7) · 2026 (7)
- 会议: NeurIPS (5) · arXiv (2) · ICLR (2) · AAAI (1) · CVPR (1)

## 相关论文 (15)

| 论文 | 会议 | 年份 | 核心贡献 |
|------|------|------|----------|
| [[P__AIMO_3竞赛揭示推理时优化瓶_MCDITO]] | arXiv | 2026 | — |
| [[P__测试时计算最优扩展超越参数扩展_Optimal_Test-Tim]] | ICLR | 2025 | — |
| [[P__GNN生成稀疏近似逆预处理器加速_GNN-SPAI]] | NeurIPS | 2025 | A GNN-based approach to construct Sparse Approximate Inverse |
| [[P__低秩分解与MoE协同的提示微调框_PT-MoE]] | NeurIPS | 2025 | PT-MoE achieves state-of-the-art PEFT performance by combini |
| [[P__CoT赋予Transformer_Chain_of_Thought]] | ICLR | 2024 | CoT empowers constant-depth transformers with constant-bit p |
| [[P__基于单次不确定性估计的高效RL数_UFO-RL]] | NeurIPS | 2025 | UFO-RL enables efficient RL training by using single-pass un |
| [[P__全球多模态数学推理与检索基准Ma_MathNet]] | AAAI | 2026 | 数学等价性是结构性属性而非语义属性——两个问题可以用完全不同的符号、语言和表述形式表达同一数学对象，而现有基于词向量的嵌 |
| [[P__因果图推理驱动的几何问题求解器_Causal-R]] | NeurIPS | 2025 | Causal-R achieves more accurate, shorter, and multiple inter |
| [[P__轨迹调制博弈自弈的可迁移推理学习_Stratagem]] | Unknown | 2026 | — |
| [[P__RL驱动的动态Handelman_APPIRL_(Automate]] | CVPR | 2025 | — |
| [[P__DiPO：解耦困惑度策略优化实现_DiPO]] | arXiv | 2026 | DiPO的核心直觉是：PPL与正确性的关系不是单调的，而是存在四个象限——真正需要干预的是「正确但高PPL」（CH，需要 |
| [[P__测试时训练的EM扩展框架TEMP_TEMPO]] | Unknown | 2026 | 现有TTT方法的失败根源在于用一个会随策略演化而漂移的自生成信号来训练策略本身——这是一个没有外部锚点的自我强化循环。T |
| [[P__混合策略蒸馏统一大语言模型知识迁_HPDL]] | Unknown | 2026 | HPD的方法贡献分为两个层次：理论统一视角和具体算法设计。

**理论层面**：论文将现有KD方法统一重新表述为toke |
| [[P__基于几何驱动的LoRA参数高效自_RLGDIP]] | Unknown | 2026 | — |
| [[P__DAPO：大规模LLM强化学习开_DAPO]] | NeurIPS | 2025 | The DAPO algorithm with four key techniques (Overlong Filter |
