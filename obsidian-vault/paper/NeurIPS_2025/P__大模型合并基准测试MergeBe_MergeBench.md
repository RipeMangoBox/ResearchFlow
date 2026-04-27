---
title: 'MergeBench: A Benchmark for Merging Domain-Specialized LLMs'
type: paper
paper_level: C
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 大模型合并基准测试MergeBench
- MergeBench
- MergeBench is the first comprehensi
acceptance: Poster
cited_by: 15
code_url: https://yifei-he.github.io/mergebench/
method: MergeBench
modalities:
- Text
paradigm: supervised
---

# MergeBench: A Benchmark for Merging Domain-Specialized LLMs

[Code](https://yifei-he.github.io/mergebench/)

**Topics**: [[T__Benchmark_-_Evaluation]] | **Method**: [[M__MergeBench]] | **Datasets**: MergeBench

> [!tip] 核心洞察
> MergeBench is the first comprehensive evaluation suite that systematically benchmarks model merging methods on large-scale (2B-9B), domain-specialized LLMs across five key domains, revealing practical guidelines and identifying fundamental limitations compared to multi-task training.

| 中文题名 | 大模型合并基准测试MergeBench |
| 英文题名 | MergeBench: A Benchmark for Merging Domain-Specialized LLMs |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2505.10833) · [Code](https://yifei-he.github.io/mergebench/) · [Project](https://yifei-he.github.io/mergebench/) |
| 主要任务 | 模型合并基准测试 / 多任务学习评估 |
| 主要 baseline | FusionBench, Compositional eval, Merging at scale, Model-GLUE; 合并方法: Model Soup, Task Arithmetic, Fisher Merging, RegMean, TIES, DARE, Consensus TA, Localize-and-Stitch |

> [!abstract] 因为「现有模型合并评估局限于小规模模型和有限任务多样性，无法反映大尺度领域专用LLM的真实表现」，作者在「FusionBench、Model-GLUE等碎片化评估」基础上改了「标准化的大规模开源评估框架——覆盖Llama-3/Gemma-2系列（2B-9B参数）、五大领域任务、三维评估指标」，在「MergeBench基准」上取得「首个系统性揭示合并方法优势与根本局限的实证结论」。

- 覆盖模型规模达 **9B 参数**，涵盖 **Llama-3** 和 **Gemma-2** 两大开源模型家族
- 评估 **8种代表性合并方法**，跨 **5个领域**：指令遵循、数学、多语言、代码、安全
- 提出**三维评估**：多任务性能、遗忘/泛化、运行效率，突破以往单一准确率指标

## 背景与动机

模型合并（Model Merging）旨在将多个领域专项微调后的模型融合为单一模型，无需重新训练即可获得多任务能力——这对于无法承受多任务训练成本的场景极具吸引力。例如，一个团队分别微调了擅长数学推理的Llama-3和擅长代码生成的Llama-3，如何不经过昂贵的联合训练就得到一个同时精通两者的模型？

现有方法大致可分为几类：**Model Soup** 直接对微调参数取平均；**Task Arithmetic** 将任务向量线性叠加回预训练模型；**TIES Merging** 和 **DARE** 引入稀疏化机制减少任务间干扰；**Consensus TA** 通过共识掩码筛选关键参数；**Fisher Merging** 和 **RegMean** 则利用参数重要性或激活统计进行加权。然而，这些方法的评估长期面临严重碎片化：**FusionBench** 虽跨模型家族但缺乏大模型和领域任务；**Model-GLUE** 缺少梯度方法和模型多样性；**Merging at scale** 不开源且缺少领域专项任务；其他工作则使用 GPT-2（124M）、RoBERTa-base（125M）等过时小模型，超参数设置各异，结果难以比较。

更根本的问题是：当模型规模扩展到数十亿参数、任务扩展到需要后训练（post-training）的领域能力时，这些在小型模型上观察到的趋势是否依然成立？现有基准无法回答。这正是 MergeBench 要解决的核心空白——建立第一个覆盖 **2B-9B** 参数、**开源可复现**、**多领域**、**标准化协议** 的系统性评估。


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/723ce8e6-7c7a-49cb-a254-625adcc19a76/figures/fig_001.png)
*Figure: Overview of MergeBench. Starting from open-source base models (Llama and Gemma), we perform*



## 核心创新

核心洞察：**模型合并的评估瓶颈不在于方法本身，而在于评估条件的不统一与规模失配**——因为现有基准在模型规模、任务领域、开源可复现性三个维度上同时缺失，导致社区无法判断何种合并方法在真实LLM场景下真正有效，从而使系统性、可复现的大规模评估成为可能。

| 维度 | Baseline（FusionBench/Model-GLUE等） | 本文 MergeBench |
|:---|:---|:---|
| 模型规模 | 最大 2.85B（mT5），多为 124M-1B 级小模型 | **2B-9B**，覆盖 Llama-3 和 Gemma-2 全系列 |
| 任务领域 | 通用NLP任务（GLUE/SuperGLUE等），无领域专项 | **五大后训练领域**：指令遵循、数学、多语言、代码、安全 |
| 训练协议 | 各工作超参数不一致，无法公平比较 | **标准化微调与评估协议**，控制所有混淆变量 |
| 评估维度 | 单一多任务准确率 | **三维评估**：多任务性能 + 遗忘/泛化 + 运行效率 |
| 开源复现 | 部分不开源或代码不完整 | **完全开源**，含训练脚本、合并实现、评测代码 |

## 整体框架


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/723ce8e6-7c7a-49cb-a254-625adcc19a76/figures/fig_002.png)
*Figure: Normalized multi-task performance across base models. We report the average normalized*



MergeBench 的整体流程遵循"预训练 → 专项微调 → 任务向量提取 → 合并应用 → 三维评估"的流水线：

1. **预训练基础模型选择**：输入为 Llama-3 或 Gemma-2 系列 checkpoint（2B-9B），输出基础参数 $\theta_{\text{pre}}$。这是所有后续实验的共同起点，确保跨方法比较的公平性。

2. **领域专项微调（Domain-specific finetuning）**：输入 $\theta_{\text{pre}}$ 和五领域训练数据，输出五个专项模型 $\theta_{\text{ft}}^{(i)}$，其中 $i \in \{\text{instruction}, \text{math}, \text{multilingual}, \text{code}, \text{safety}\}$。该模块采用**标准化超参数**，替代以往各工作随意的微调设置。

3. **任务向量计算（Task vector extraction）**：输入 $\theta_{\text{ft}}^{(i)}$ 和 $\theta_{\text{pre}}$，输出任务向量 $\tau_i = \theta_{\text{ft}}^{(i)} - \theta_{\text{pre}}$。这一差分表示封装了微调过程中获得的知识，是所有后续合并操作的统一表示。

4. **合并方法应用（Merging methods）**：输入任务向量集合 $\{\tau_i\}$ 和基础模型 $\theta_{\text{pre}}$，输出合并模型 $\theta_{\text{merged}}$。MergeBench 实现了 **8种代表性方法**：Model Soup、Task Arithmetic、Fisher Merging、RegMean、TIES、DARE、Consensus TA、Localize-and-Stitch。

5. **三维评估（Three-dimensional evaluation）**：输入合并模型和领域测试数据，输出三类指标——(a) **多任务性能**（各领域平均表现）、(b) **泛化/遗忘**（保留预训练通用能力）、(c) **运行效率**（wall-clock 时间）。

```
预训练模型 θ_pre
    ↓ [标准化微调]
五领域专项模型 {θ_ft^(i)}
    ↓ [逐元素差分]
任务向量 {τ_i = θ_ft^(i) − θ_pre}
    ↓ [八种方法之一]
合并模型 θ_merged
    ↓ [三维评估]
(多任务性能, 泛化分数, 运行时间)
```

## 核心模块与公式推导

### 模块 1: 任务向量与基础合并（对应框架图：任务向量提取 → 合并方法应用）

**直觉**：模型微调后的参数变化蕴含领域知识，差分表示使得不同模型的知识可在同一空间中线性操作。

**Baseline 公式** (Task Arithmetic):
$$\theta_{\text{merged}} = \theta_{\text{pre}} + \lambda \sum_{i=1}^{n} \tau_i$$
符号: $\theta_{\text{pre}} \in \mathbb{R}^d$ = 预训练模型参数, $\tau_i = \theta_{\text{ft}}^{(i)} - \theta_{\text{pre}}$ = 第 $i$ 个任务向量, $\lambda \in \mathbb{R}$ = 验证集上调优的缩放系数, $n$ = 任务数。

**变化点**：直接累加所有任务向量会导致任务间干扰（interference），尤其当参数更新方向冲突时。后续方法均在此基础上增加选择/加权机制。

**本文公式（推导）**:
$$\text{Step 1}: \tau_i = \theta_{\text{ft}}^{(i)} - \theta_{\text{pre}} \quad \text{[任务向量定义，统一知识表示]}$$
$$\text{Step 2}: \theta_{\text{merged}}^{\text{TA}} = \theta_{\text{pre}} + \lambda \sum_{i=1}^{n} \tau_i \quad \text{[基础任务算术，λ 控制合并强度]}$$
$$\text{最终}: \theta_{\text{merged}} = f(\{\tau_i\}, \theta_{\text{pre}}; \text{method}) \quad \text{[method-dependent 变换]}$$

**对应消融**：Table 7 显示不同算法对超参数调优的需求差异显著，Task Arithmetic 的 λ 调优被描述为"低效且 largely trial-and-error"。

---

### 模块 2: 共识任务算术 Consensus TA（对应框架图：合并方法应用中的进阶方法）

**直觉**：并非所有参数都应被所有任务更新，通过比较多任务参考向量与单任务向量，仅保留"真正属于该任务"的参数，再用多数投票聚合，可减少负迁移。

**Baseline 公式** (Task Arithmetic):
$$\theta_{\text{merged}} = \theta_{\text{pre}} + \lambda \sum_{i=1}^{n} \tau_i$$
该 baseline 对所有参数一视同仁地累加，导致冲突参数相互抵消。

**变化点**：Consensus TA 发现，若某参数的任务向量幅度小于多任务向量与该任务向量之差的幅度，则该参数可能已被其他任务"覆盖"，不应保留。进一步要求至少两个任务共同支持才保留，实现共识机制。

**本文公式（推导）**:
$$\text{Step 1}: \tau_{\text{MTL}} = \theta_{\text{merged}}^{\text{TA}} - \theta_{\text{pre}} = \lambda \sum_{i=1}^{n} \tau_i \quad \text{[从 Task Arithmetic 提取多任务参考向量]}$$
$$\text{Step 2}: m_i = \mathbb{1}\left\{|\tau_i| \geq |\tau_{\text{MTL}} - \tau_i| \cdot \lambda_i\right\} \quad \text{[任务掩码：仅保留该任务"独有"的参数，λ_i 任务特定阈值]}$$
$$\text{Step 3}: m_{\text{consensus}} = \mathbb{1}\left\{\sum_{i \in [n]} m_i \geq 2\right\} \quad \text{[共识掩码：≥2 个任务同意才保留，减少干扰]}$$
$$\text{最终}: \theta_{\text{merged}}^{\text{Consensus}} = \theta_{\text{pre}} + \sum_{i=1}^{n} m_{\text{consensus}} \odot \tau_i$$

**对应消融**：Table 8-9 显示 Consensus TA 在多任务性能与泛化之间取得较好平衡，但 sparsification 方法（TIES、DARE）在特定设置下表现更优，无单一方法全面主导。

---

### 模块 3: 稀疏化合并 DARE（对应框架图：合并方法应用中的效率优化方法）

**直觉**：随机丢弃大部分任务向量更新，仅保留少量关键参数变化，通过幅度重缩放补偿信息损失，可在极低计算开销下近似完整合并效果。

**Baseline 公式** (Task Arithmetic):
$$\theta_{\text{merged}} = \theta_{\text{pre}} + \lambda \sum_{i=1}^{n} \tau_i$$

**变化点**：直接合并保留全部参数更新，计算和存储开销大。DARE 引入随机 dropout 对任务向量进行极端稀疏化（丢弃率 $p$ 可达 0.9），并通过 $\lambda/(1-p)$ 重缩放保持期望幅度。

**本文公式（推导）**:
$$\text{Step 1}: m_i \sim \text{Bernoulli}(p) \quad \text{[对每个参数独立采样丢弃掩码，p = 丢弃概率]}$$
$$\text{Step 2}: \tilde{\tau}_i = \frac{\lambda}{1-p} \cdot (1 - m_i) \odot \tau_i \quad \text{[丢弃后重缩放：期望幅度保持不变]}$$
$$\text{最终}: \theta_{\text{merged}}^{\text{DARE}} = \theta_{\text{pre}} + \sum_{i=1}^{n} \tilde{\tau}_i$$

符号: $m_i \in \{0,1\}^d$ = Bernoulli 随机掩码, $p$ = 丢弃概率（通常 0.5-0.9）, $\odot$ = Hadamard 积, $\lambda/(1-p)$ = 补偿因子。

**对应消融**：Table 3 显示 DARE 的运行时间显著低于需要数据依赖的方法（如 RegMean、Fisher Merging）；Figure 5 展示其在性能-效率权衡中的位置。Table 8 显示在适当调参下，DARE 的稀疏化策略能有效保留知识。

## 实验与分析



MergeBench 在 **Llama-3.2-3B、Llama-3.1-8B、Gemma-2-2B、Gemma-2-9B** 四个模型规模上展开系统评估，覆盖五大领域。核心发现来自 Table 8（平均归一化多任务性能）和 Table 9（平均归一化泛化性能）：**多任务训练模型（multi-task learning, MTL）在所有设置下均优于所有合并方法**，这揭示了合并方法的根本性局限——当任务非冲突且可平衡采样时，联合训练仍是性能上限。然而，合并方法在**无法承担 MTL 成本**的场景下提供了实用替代。

具体而言，在 **Gemma-2-2B** 上（Table 10），各合并方法的相对排序随领域变化；**Consensus TA** 和 **TIES Merging** 通常位居前列，而简单的 **Model Soup** 在较大模型上表现相对更差。**Llama-3.1-8B** 结果（Table 16）显示一个关键趋势：**合并方法在更强的基础模型上表现更好**——这与"更强预训练模型提供更鲁棒的参数空间用于线性插值"的直觉一致。



效率维度（Table 3 / Figure 4）揭示重要权衡：**Model Soup** 和 **Task Arithmetic** 运行最快（几乎无额外开销），**Fisher Merging** 和 **RegMean** 因需计算 Fisher 矩阵或激活统计而显著更慢。**DARE** 和 **TIES** 通过稀疏化在性能和效率间取得较好平衡。Figure 3 的泛化-多任务性能散点图显示，部分方法（如 Consensus TA）能接近右上角的理想区域，但没有任何方法同时达到 MTL 的多任务性能和预训练模型的泛化水平。

**公平性检查**：作者坦诚三个局限：(1) 超参数调优被描述为"inefficient and largely trial-and-error"，结果可能未达各方法最优；(2) MTL 作为隐式上界未在所有主表中直接并列呈现；(3) 未包含 2024 年后新提出的合并方法。此外，合并的计算成本对 9B 模型仍"non-trivial"，限制了超参数搜索空间。

## 方法谱系与知识库定位

**方法家族**：模型合并（Model Merging）→ 任务算术族（Task Arithmetic family）

**父方法**：Task Arithmetic（Ilharco et al., 2022）— 提供任务向量线性组合的基础范式；Model Soup（Wortsman et al., 2022）— 提供简单平均的基线。

**本文改变的 slots**：
- **data_pipeline**：小模型/有限任务 → 大规模开源LLM + 五领域专项任务
- **training_recipe**：随意微调 → 标准化协议
- **evaluation_strategy**：单指标 → 三维评估（性能/遗忘/效率）

**直接 baselines 及差异**：
- **FusionBench**：MergeBench 在其跨模型家族思路上扩展，但补齐了大模型、领域任务、开源性三短板
- **Model-GLUE**：MergeBench 引入梯度方法和更大模型多样性
- **Merging at scale**：MergeBench 强调开源可复现和领域任务覆盖

**后续方向**：
1. **动态合并策略**：根据任务相似度自适应选择合并系数，替代全局 λ
2. **与 MTL 的混合范式**：探索何时合并、何时转向数据混合的决策边界
3. **更大规模验证**：将基准扩展至 70B+ 模型，检验现有趋势是否延续

**标签**：modality:text | paradigm:supervised fine-tuning + model merging | scenario:multi-task learning, benchmark | mechanism:task vector, sparsification, consensus masking | constraint:open-source reproducibility, computational efficiency

## 引用网络

### 直接 baseline（本文基于）

- How to Merge Your Multimodal Models Over Time? _(CVPR 2025, 直接 baseline, 未深度分析)_: Directly about model merging, very recent (Dec 2024), and explicitly about mergi

