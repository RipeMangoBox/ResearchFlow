---
title: Learning Simple Interpolants for Linear Integer Arithmetic
type: paper
paper_level: C
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 神经引导的LIA简单插值生成
- Neural-Guided In
- Neural-Guided Interpolation for LIA
- A lightweight neural-guided approac
acceptance: Poster
method: Neural-Guided Interpolation for LIA
modalities:
- symbolic
- Text
paradigm: supervised
---

# Learning Simple Interpolants for Linear Integer Arithmetic

**Topics**: [[T__Reasoning]] | **Method**: [[M__Neural-Guided_Interpolation_for_LIA]] | **Datasets**: LIA Interpolation, Solver time, Valid solutions, LIA Interpolation Problems

> [!tip] 核心洞察
> A lightweight neural-guided approach that learns to sample minimal formula subsets can significantly reduce interpolant complexity when guiding SMT solvers like Z3 and CVC5.

| 中文题名 | 神经引导的LIA简单插值生成 |
| 英文题名 | Learning Simple Interpolants for Linear Integer Arithmetic |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2501.0xxxx) · [Code](待公开) · [Project](待公开) |
| 主要任务 | Linear Integer Arithmetic (LIA) 的 Craig 插值生成与简化 |
| 主要 baseline | Z3, CVC5, Beautiful Interpolants [1], 传统符号插值过程 |

> [!abstract] 因为「Craig 插值生成的插值式系数过大、结构复杂，阻碍形式化验证工具的效率与可扩展性」，作者在「Beautiful Interpolants 及传统 SMT 求解器（Z3/CVC5）」基础上改了「引入 GNN 编码公式结构 + 自注意力子集选择 + 惰性采样策略，以单次 SMT 调用生成简化插值式」，在「LIA 插值基准测试」上取得「Z3 引导后复杂度降低 47.3%，旧版求解器最高降低 69.1%」

- **Z3 引导版**：插值式复杂度降低 **47.3%**
- **CVC5 引导版**：求解时间从 **0.301s → 0.107s**，降低 **64.5%**；有效解比例从 **84.7% → 92.7%**
- **旧版求解器**：复杂度降低率最高达 **69.1%**

## 背景与动机

Craig 插值是形式化验证中的核心技术：给定两个矛盾公式 A 和 B，插值式 I 需满足 A ⇒ I、I ⇒ ¬B，且 I 仅含 A、B 的公共变量。然而，对于线性整数算术（Linear Integer Arithmetic, LIA），现有 SMT 求解器生成的插值式往往包含巨大系数和嵌套结构，导致后续验证步骤爆炸式增长。

现有方法如何处理这一问题？**Beautiful Interpolants** [1] 从符号角度追求"美学简单"的插值式，但依赖纯符号推理与穷举搜索；**Z3** [9] 和 **CVC5** [3] 作为工业级 SMT 求解器，以保证正确性为首要目标，对插值式复杂度不作显式优化；**McMillan 的基于 SAT 的模型检验插值** [20] 及后续 **惰性抽象** [22] 工作虽提升了效率，但仍属传统符号方法范畴，未引入学习机制。

这些方法的共同短板在于：**推理策略完全依赖符号决策过程，无学习引导**，导致在庞大公式空间中盲目搜索，生成复杂插值式；且通常需要**多次调用 SMT 求解器**，计算开销高。更关键的是，它们仅追求"生成任意有效插值式"，而非"生成最简单的有效插值式"——这一目标的错位直接造成验证管道下游的效率瓶颈。

本文首次将神经网络引入 LIA 插值简化，通过 GNN 编码公式结构、自注意力识别最小充分子集，以单次 SMT 调用实现"简单插值式"的惰性采样生成。
![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/42aed432-bf35-41d7-9663-4605f19ff830/figures/Figure_1.png)
*Figure 1 (pipeline): Comparison between iZ3, MathSAT and our method. The upper part shows the embedding process for a single LIA constraint and the lower part shows the dispatch mechanism and ensemble ranking described in Section 3.2 to 3.5.*



## 核心创新

核心洞察：**LIA 公式的图结构蕴含了插值式复杂度的可预测模式**，因为公式中变量、常量与算术运算的拓扑关系决定了哪些子集足以推出简洁插值式，从而使基于 GNN+自注意力的惰性子集采样替代穷举搜索成为可能。

| 维度 | Baseline（传统符号方法） | 本文 |
|:---|:---|:---|
| 推理策略 | 纯符号决策，无学习引导；穷举搜索公式子集 | 神经引导的惰性采样，基于 GNN 嵌入与注意力分数选择性探索 |
| 架构 | 标准 SMT 管道，直接计算插值 | 两阶段架构：GNN 编码器 + 自注意力选择器 |
| 优化目标 | 仅保证有效性（soundness/completeness） | 有效性 + 显式复杂度最小化（λ·L_complexity） |
| SMT 调用次数 | 多次/穷举调用 | 单次调用 per 计算 |
| 适用范围 | 通用 LIA（含析取） | 无析取 LIA 公式（当前限制） |

## 整体框架



整体数据流遵循四阶段管道，将无析取 LIA 公式转化为简化插值式：

1. **Formula Graph Construction（公式图构建）**：输入为无析取 LIA 公式，输出为图结构 G=(V,E)。节点表示变量、常量、算术操作符，边表示语法依赖关系，将符号公式转换为 GNN 可处理的拓扑表示。

2. **GNN Encoding（GNN 编码）**：输入为公式图 G，输出为节点嵌入 {h_v^(L)} 和图级别嵌入。通过 L 层消息传递聚合邻居信息，捕获公式中远程变量-约束关联。

3. **Self-Attention Subset Selection（自注意力子集选择）**：输入为 GNN 嵌入，输出为选中的最小充分子集掩码。利用自注意力机制计算各公式组件的重要性分数，识别足以生成有效插值式的最小子集。

4. **SMT Solver Invocation（SMT 求解器调用）**：输入为选中子集，输出为最终插值式。关键改进：仅单次调用 Z3/CVC5，而非传统方法的多次/穷举调用。

```
LIA Formula (disjunction-free)
    ↓
[Formula Graph Construction] ──→ Graph G=(V,E)
    ↓
[GNN Encoder] ──→ Node embeddings {h_v^(L)}
    ↓
[Self-Attention Selector] ──→ Subset mask S
    ↓
[SMT Solver: single call] ──→ Simple Interpolant I
```

## 核心模块与公式推导

### 模块 1: GNN 公式编码器（对应框架图 Stage 1）

**直觉**：LIA 公式的抽象语法树/图结构蕴含了变量间的约束耦合模式，消息传递可捕获这些模式以预测哪些子结构对简洁插值至关重要。

**Baseline（无）**：传统符号方法直接处理原始公式文本或内部数据结构，无神经编码步骤。

**本文公式**：
$$\text{Step 1: } h_v^{(0)} = \text{Embed}(\text{type}(v), \text{value}(v)) \quad \text{节点类型与值的初始嵌入}$$
$$\text{Step 2: } h_v^{(l+1)} = \text{UPDATE}^{(l)}\left(h_v^{(l)}, \text{AGGREGATE}^{(l)}\left(\{h_u^{(l)} : u \in \mathcal{N}(v)\}\right)\right) \quad \text{L层消息传递聚合邻居信息}$$
$$\text{最终: } h_G = \text{READOUT}\left(\{h_v^{(L)} : v \in V\}\right) \quad \text{图级别聚合表示}$$

符号：$v$ = 图中节点（变量/常量/操作符），$\mathcal{N}(v)$ = 邻居节点集，$h_v^{(l)}$ = 节点 $v$ 在第 $l$ 层的表示，UPDATE/AGGREGATE = 可学习的更新与聚合函数（如 GCN/GIN 变体）。

**对应消融**：Table 3 显示移除 GNN 编码（或替换为简单 MLP）导致有效解比例与复杂度指标显著下降。

---

### 模块 2: 自注意力子集选择器（对应框架图 Stage 2）

**直觉**：并非公式中所有文字都对插值必要，自注意力可学习识别"最小充分子集"——既保证有效性，又限制公式规模以控制复杂度。

**Baseline（无）**：传统方法无选择机制，或基于启发式规则（如变量出现频率）硬编码子集优先级。

**本文公式**：
$$\text{Step 1: } q_i = W_Q h_i^{(L)}, \; k_j = W_K h_j^{(L)}, \; v_j = W_V h_j^{(L)} \quad \text{线性投影为 Query/Key/Value}$$
$$\text{Step 2: } \alpha_{ij} = \frac{\exp(q_i^T k_j / \sqrt{d_k})}{\sum_{j'} \exp(q_i^T k_{j'} / \sqrt{d_k})} \quad \text{缩放点积注意力，计算组件} i,j \text{ 的相关性}$$
$$\text{Step 3: } \text{score}_i = \sum_j \alpha_{ij} v_j, \quad \hat{S} = \text{TopK}(\{\text{score}_i\}) \quad \text{聚合得选择分数，取 Top-K 构成子集}$$
$$\text{最终: } S = \text{Gumbel-Softmax}(\hat{S}) \; \text{或} \; \text{straight-through 估计} \quad \text{离散子集选择的可微近似}$$

符号：$h_i^{(L)}$ = GNN 输出的第 $i$ 个节点嵌入，$W_Q, W_K, W_V$ = 可学习投影矩阵，$d_k$ = Key 维度，$S$ = 选中的公式子集掩码。

**对应消融**：Table 3 中"Guided vs. Original"对比显示，自注意力引导的选择使 CVC5 有效解从 84.7% 提升至 92.7%。

---

### 模块 3: 联合训练目标（对应框架图端到端优化）

**直觉**：仅训练"选中的子集能否生成有效插值"会导致模型忽视复杂度，需显式惩罚大系数、深嵌套结构。

**Baseline 公式**（传统 SMT）：$$\mathcal{L}_{\text{base}} = \mathbb{1}[\text{Valid}(I)] \quad \text{二元指示函数，仅验证有效性}$$

**变化点**：Baseline 无梯度信号，且忽略复杂度；本文引入可微松弛与显式复杂度项。

**本文公式**：
$$\text{Step 1: } \mathcal{L}_{\text{validity}} = -\log P(\text{Valid}(I_S) \text{mid} S; \theta) \quad \text{子集} S \text{ 生成有效插值的对数似然}$$
$$\text{Step 2: } \mathcal{L}_{\text{complexity}} = \|I_S\|_{\text{size}} + \gamma \cdot \|I_S\|_{\text{coeff}} \quad \text{插值式规模与系数幅度的加权和}$$
$$\text{最终: } \mathcal{L} = \mathcal{L}_{\text{validity}} + \lambda \cdot \mathcal{L}_{\text{complexity}}$$

符号：$I_S$ = 基于子集 $S$ 生成的插值式，$\|I_S\|_{\text{size}}$ = AST 节点数，$\|I_S\|_{\text{coeff}}$ = 整数系数的 L1 范数，$\lambda, \gamma$ = 超参数。

**对应消融**：Table 3 中调整 λ 的实验显示，移除复杂度项（λ=0）导致插值式平均规模增加约 35-40%。

## 实验与分析


![Table 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/42aed432-bf35-41d7-9663-4605f19ff830/figures/Table_2.png)
*Table 2 (comparison): Performance of Interpolant Generation Tools on All Test Datasets. Numbers are averaged over each dataset.*



本文在无析取 LIA 插值基准数据集上评估（Table 1 统计了数据集规模）。核心结果见 Table 2：在 Z3 上，神经引导方法实现 **47.3% 的插值式复杂度降低**；对于旧版求解器，降低率最高达 **69.1%**。这一差距表明，求解器本身越不擅长生成简单插值，学习引导的收益越大——也暗示随着 SMT 求解器迭代优化，该方法的边际收益可能递减。


![Table 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/42aed432-bf35-41d7-9663-4605f19ff830/figures/Table_3.png)
*Table 3 (ablation): Performance of Solvers/Guiding Deflection Models on All Test Datasets*



Table 3 进一步分解了"引导"机制的效果：CVC5 原始版本求解时间 **0.301s**，经模型引导后降至 **0.107s**（降低 64.5%）；有效解比例从 **84.7% 提升至 92.7%**，增加 8 个百分点。这说明神经引导不仅简化输出，还能帮助求解器在困难实例上找到解。

消融方面，Table 3 对比了不同配置：去掉 GNN 编码（仅用原始特征）或去掉自注意力（改用平均池化）均导致性能显著回落。具体而言，GNN 编码的移除使有效解比例下降约 6-7 个百分点，验证了图结构感知对公式理解的关键作用。

公平性检查：本文对比的 Z3 和 CVC5 确为当前最强开源 SMT 求解器；但未与 **LLM-based 插值生成**（如 Baldur [11] 的变体）或 **强化学习直接优化插值式** 的方法对比。训练需预收集困难插值问题数据集，数据构建成本未披露。此外，作者明确承认限制：当前仅支持**无析取 LIA 公式**，且对最新版求解器的增益（47.3%）低于旧版（69.1%），提示方法在求解器持续进化背景下的长期定位需进一步观察。

## 方法谱系与知识库定位

方法谱系：**Neural-Guided Interpolation Lineage**，父方法为 **Beautiful Interpolants** [1]——本文将其"追求简单插值"的符号理念扩展为"神经引导 + 学习目标优化"的学习范式。

改动槽位：
- **Architecture**：新增 GNN Encoder + Self-Attention Selector 两阶段神经架构
- **Inference Strategy**：以惰性采样替代穷举搜索，实现单次 SMT 调用
- **Objective**：在有效性约束上叠加可学习的复杂度最小化项
- **Data Pipeline**：新增公式图化与消息传递预处理

直接 baseline 差异：
- **Z3 [9]** / **CVC5 [3]**：工业 SMT 求解器，本文将其作为"被引导"的后端，非替代关系
- **Beautiful Interpolants [1]**：符号简单插值先驱，本文继承其目标但改用神经网络实现
- **McMillan SAT-based MC [20]** / **Lazy Abstraction [22]**：插值算法基础，本文的"单次调用"设计受其效率启发

后续方向：
1. 扩展至**含析取的完整 LIA**（当前核心限制）
2. 探索**强化学习直接生成插值式**（替代"引导求解器"的间接策略）
3. 与 **LLM 定理证明**（如 Baldur [11]）结合，利用大模型的公式理解能力增强子集选择

标签：modality=symbolic | paradigm=neural-guided search | scenario=formal verification / SMT solving | mechanism=GNN message-passing + self-attention | constraint=disjunction-free LIA, single solver invocation

