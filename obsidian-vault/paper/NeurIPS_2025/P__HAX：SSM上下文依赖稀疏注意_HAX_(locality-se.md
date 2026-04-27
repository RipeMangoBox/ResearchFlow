---
title: Overcoming Long Context Limitations of State Space Models via Context Dependent Sparse Attention
type: paper
paper_level: A
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- HAX：SSM上下文依赖稀疏注意力
- HAX (locality-se
- HAX (locality-sensitive Hashing Attention with sparse Key Selection)
- Integrating SSMs with Context-Depen
acceptance: Poster
cited_by: 3
code_url: https://github.com/DeepGraphLearning/HAX
method: HAX (locality-sensitive Hashing Attention with sparse Key Selection)
modalities:
- Text
paradigm: supervised
---

# Overcoming Long Context Limitations of State Space Models via Context Dependent Sparse Attention

[Code](https://github.com/DeepGraphLearning/HAX)

**Topics**: [[T__Text_Generation]], [[T__Reasoning]] | **Method**: [[M__HAX]] | **Datasets**: Multi-query joint recall, Ruler

> [!tip] 核心洞察
> Integrating SSMs with Context-Dependent Sparse Attention (CDSA), instantiated via locality-sensitive Hashing Attention with sparse Key Selection (HAX), enables sub-quadratic long-context modeling with sufficient expressiveness for context-dependent tasks that pure SSMs cannot solve.

| 中文题名 | HAX：SSM上下文依赖稀疏注意力 |
| 英文题名 | Overcoming Long Context Limitations of State Space Models via Context Dependent Sparse Attention |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2507.00449) · [Code](https://github.com/DeepGraphLearning/HAX) · [DOI](https://doi.org/10.48550/arxiv.2507.00449) |
| 主要任务 | 长上下文语言建模、文本生成、推理 |
| 主要 baseline | Mamba, Mamba2, LSH attention, CISA (sliding window / dilated / A-shaped) |

> [!abstract] 因为「纯SSM无法解决多查询联合召回等需要上下文依赖的长程依赖任务」，作者在「Mamba/Mamba2」基础上改了「引入Context-Dependent Sparse Attention (CDSA)，通过LSH attention + Key Selection (KS) attention的混合机制」，在「Ruler 2K」上取得「44.57平均得分，相比Mamba 42.43提升+2.14」

- **Ruler 2K average**: HAX 44.57 vs Mamba 42.43 (+2.14)
- **Ruler NIAHMK1**: HAX 45.4 vs Mamba 34.4 (+11.0)
- **KS单独使用**: Ruler average 43.10，仍低于HAX 44.57，验证混合必要性

## 背景与动机

当前大语言模型面临的核心矛盾是：Transformer的full attention在长序列上具有二次复杂度，而State Space Models (SSMs)如Mamba虽能实现线性复杂度，却牺牲了捕捉上下文依赖的长程依赖能力。具体而言，当模型需要同时根据多个前文token的联合信息来生成当前token时——例如"找到前文所有出现的[姓名]并统计其[年龄]之和"——纯SSM无法有效完成这种多查询联合召回。

现有方法从不同角度尝试缓解这一问题。**Mamba/Mamba2**通过选择性状态空间实现线性时间建模，但其recurrent state本质上是固定维度的压缩表示，无法动态地根据当前query选择不同的历史key进行精细交互。**LSH attention**利用局部敏感哈希将相似内容分到同一桶中，实现了内容相关的稀疏路由，但作者发现LSH难以捕捉LLM中常见的"垂直条纹"注意力模式——即少数全局重要token（如指令词、主题词）被大量后续query共同关注。**Context-Independent Sparse Attention (CISA)**包括sliding window、dilated、A-shaped等固定稀疏模式，完全不依赖输入内容，灵活性最差。

这些方法的共同瓶颈在于：它们的稀疏模式要么与当前query无关（CISA），要么虽然内容相关但无法表达"全局重要key被多query共享关注"的垂直条纹结构（LSH）。作者理论证明，纯SSM无法在亚二次时间内解决multi-query joint recall问题。为此，本文提出HAX架构，将SSM与一种新型的上下文依赖稀疏注意力（CDSA）相结合，通过LSH与Key Selection的双分支设计互补地覆盖不同类型的注意力模式。

## 核心创新

核心洞察：稀疏注意力的稀疏模式应当依赖于当前query与key的交互上下文，而非预先固定；因为真正的长程依赖往往表现为少数全局重要token被多个query共同关注（垂直条纹），而这类模式无法被基于内容相似性的LSH或固定模式捕捉，从而使SSM在保持亚二次复杂度的同时获得足够的表达能力成为可能。

| 维度 | Baseline (Mamba / LSH / CISA) | 本文 (HAX) |
|:---|:---|:---|
| 稀疏模式来源 | 无attention / 内容哈希分桶 / 固定窗口 | **query-key上下文动态决定**：LSH分桶 + MLP评分选key |
| 垂直条纹捕捉 | ❌ LSH无法捕捉；CISA完全不适应 | ✅ KS attention显式选择全局重要key |
| 训练目标 | 纯next-token cross-entropy | **联合训练**：LM loss + α × 层间ranking loss |
| 复杂度保证 | SSM线性 / sparse attention固定k | 理论保证总稀疏度≤k（各分支k/2） |
| 架构集成 | 纯SSM block | **Hybrid block**：Mamba SSM + HAX稀疏注意力模块 |

## 整体框架


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/18e6cf73-b2c3-4981-bd1f-721bb12f8889/figures/fig_001.png)
*Figure: Comparison of joint recall and associative recall. Associative recall does not account for*



HAX的整体数据流如下：输入序列首先经过**Mamba/Mamba2 SSM**处理，生成隐状态表示并派生出Q/K矩阵；随后进入并行的双分支稀疏注意力机制：

1. **LSH attention分支**：对Q、K进行局部敏感哈希分桶，将相似内容的路由到同一桶内，生成内容相关的稀疏掩码 $S^{LSH}$，擅长捕捉局部相似性模式；
2. **Key Selection (KS) attention分支**：通过**Key scoring MLP**为每个key计算重要性分数（基于该key自身表示及所有历史query的聚合），选择top-k高分key生成掩码 $S^{KS}$，专门捕捉垂直条纹模式；

两个分支的掩码通过逐元素取最大值合并为**HAX混合掩码** $S^{HAX} = \max\{S^{LSH}, S^{KS}\}$，确保每个query最多关注k个key。最终基于 $S^{HAX}$ 执行稀疏注意力计算，输出与SSM路径的结果结合。

```
Input Sequence
    ↓
[Mamba/Mamba2 SSM] ──→ Hidden States + Q, K, V
    ↓                    ↓
[LSH Attention]    [Key Scoring MLP]
    ↓                    ↓
S^LSH              [Key Selection: Top-k]
    ↓                    ↓
    └──────→ [max{S^LSH, S^KS}] = S^HAX ←──────┘
                    ↓
            [Sparse Attention]
                    ↓
            [Output Projection]
```

该架构的关键设计在于：LSH和KS各自分配k/2的稀疏预算，合并后严格保证总稀疏度≤k，维持亚二次计算复杂度。

## 核心模块与公式推导

### 模块 1: Key Scoring MLP（对应框架图 KS分支左侧）

**直觉**：模型需要一种自回归兼容的方式来判断"哪些历史key对当前及未来query最重要"，而不能依赖未来信息。

**Baseline**：纯SSM无此模块；LSH attention直接基于内容哈希，无需学习评分。

**本文公式（推导）**:
$$\text{Step 1 (理想形式)}: x_i = f_\theta(K_i, Q_{1...i}) \quad \text{为每个key计算理想重要性分数，依赖该key和所有历史query}$$
$$\text{Step 2 (MLP近似)}: f_\theta(K_i, Q_{1..i}) \triangleq \text{MLP}\left(K_i, \text{normalize}\left(\sum_{1\leq p\leq i} Q_p\right)\right) \quad \text{用历史query的归一化和代替完整序列，保证自回归兼容性}$$

符号: $K_i$ = 第i个key的表示; $Q_{1..i}$ = 位置i之前所有query; $\text{normalize}(\sum Q_p)$ = 历史query的聚合归一化向量; $\theta$ = MLP参数。

**对应消融**：Table 2显示，仅使用KS（无LSH）时Ruler average为43.10，虽优于Mamba 42.43，但低于完整HAX 44.57，说明KS单独有效但需LSH补充。

---

### 模块 2: Key Selection Mask 与 Ranking Loss（对应框架图 KS分支右侧及训练目标）

**直觉**：直接回归注意力绝对值困难且不鲁棒，转化为成对排序问题更稳定。

**Baseline 公式** (标准attention): $$A = \text{softmax}(QK^T/\sqrt{d})$$

**变化点**：标准attention计算所有query-key对的相似度，复杂度二次；本文仅保留top-k连接，且通过排序学习而非端到端梯度选择哪些连接保留。

**本文公式（推导）**:
$$\text{Step 1 (选择掩码)}: S^{KS}_{ij} = \mathbb{1}[x_j \in \text{Top-}k\{x_1, ..., x_i\}] \quad \text{每个query只关注之前k个得分最高的key，形成垂直条纹}$$
$$\text{Step 2 (参考注意力)}: A' = Q K[I]^\text{top}, \quad y = \sigma(A') \odot M[I] \quad \text{通过线性注意力计算采样键的参考权重作为监督信号}$$
$$\text{Step 3 (成对转化)}: P_{ij}(x) = x_i - x_j, \quad T_{ij}(y) = \begin{cases} 1 & y_i > y_j \\ 0.5 & y_i = y_j \\ 0 & y_i < y_j \end{cases} \quad \text{将绝对值转化为相对排序，降低监督难度}$$
$$\text{Step 4 (排序损失)}: \mathcal{L}_{\text{score}}(x, y) = \frac{1}{k^2} \sum_{i,j} \text{BCE}(P_{ij}(x), T_{ij}(y)) \quad \text{用二元交叉熵优化成对排序，对参考权重尺度变化鲁棒}$$
$$\text{Step 5 (联合目标)}: \mathcal{L} = \mathcal{L}_{\text{LM}} + \alpha \sum_{\text{layers}} \mathcal{L}_{\text{score}} \quad \text{评分损失与语言建模联合训练，α平衡两者}$$

**对应消融**：Table 2显示去掉KS、仅保留LSH时Ruler average从44.57暴跌至8.36（-36.21），验证ranking loss训练的KS模块是性能核心。

---

### 模块 3: HAX Hybrid Mask（对应框架图 掩码合并节点）

**直觉**：LSH和KS捕捉互补的注意力模式，简单相加可能超预算，取最大值可保留各自优势同时控制稀疏度。

**Baseline 公式** (单一稀疏模式): $$S^{\text{single}}_i \in \{0,1\}^l, \quad \|S^{\text{single}}_i\|_0 \leq k$$

**变化点**：单一模式（仅LSH或仅SW）无法同时覆盖多种注意力结构；本文将两种模式通过逐元素最大值合并。

**本文公式（推导）**:
$$\text{Step 1 (合并)}: S^{HAX} = \max\{S^{LSH}, S^{KS}\} \in \{0, 1\}^{l \times l} \quad \text{逐元素最大值保留任一组件选中的连接}$$
$$\text{Step 2 (稀疏度保证)}: \forall i, \|S^{HAX}_i\|_0 \leq k \quad \text{when} \quad \forall i, \|S^{LSH}_i\|_0 \leq \frac{k}{2}, \|S^{KS}_i\|_0 \leq \frac{k}{2} \quad \text{理论保证总稀疏度不超预算}$$

符号: $S^{LSH}, S^{KS}$ = 二值稀疏掩码; $\max$ = 逐元素最大值; $k$ = 总稀疏预算（实验中固定为128）。

**对应消融**：Table 2中+KS (ours) 43.10 vs HAX 44.57，+1.47的增益直接来自与LSH的协同；而所有CISA变体（D 9.78, SW 42.52, SW+D 42.26, A 41.91）均低于HAX，证明混合设计优于任何单一固定模式。

## 实验与分析



本文在合成任务和真实世界基准上全面验证HAX的有效性。合成任务方面，作者设计了**multi-query joint recall**任务——扩展传统associative recall要求模型同时检索多个key-value对并联合推理——理论证明纯SSM无法在此任务上以亚二次复杂度取得高准确率。Table 1显示HAX在该任务上 consistently 优于所有baseline及消融变体，包括Mamba、Mamba2、各CISA扩展（+D, +SW, +SW+D, +A）以及单独的+LSH和+KS。

真实世界评估聚焦于**Ruler**（2K上下文，12项任务平均）和**LongBench English**。 在Ruler 2K上，HAX取得**44.57**的平均分，相比Mamba **42.43**提升**+2.14**，相比KS单独使用（43.10）提升**+1.47**，验证了LSH与KS混合的必要性。细分任务中，HAX在需要多key联合检索的**NIAHMK1**上优势最大：45.4 vs Mamba 34.4（**+11.0**），vs +KS 33.6（**+11.8**），说明LSH分支对KS的互补性在复杂检索场景中尤为关键。LongBench English上HAX同样取得最佳平均性能（Table 3）。



消融实验揭示了各组件的边际贡献（Table 2）。**最关键的发现**：仅保留LSH、移除KS时，Ruler average从44.57崩溃至**8.36**（**-36.21**），说明LSH单独完全无法应对Ruler的多样化长程依赖挑战——这与作者关于LSH无法捕捉垂直条纹的理论分析一致。反之，仅保留KS（43.10，-1.47）虽优于Mamba，但不及完整HAX，表明LSH在捕捉内容相似性模式上仍有独特价值。CISA变体中，dilated attention表现最差（9.78），甚至远低于纯Mamba，揭示固定稀疏模式在长上下文任务中的脆弱性。

**公平性检查**：本文比较了Mamba/Mamba2作为强SSM baseline，以及LSH、CISA等稀疏注意力变体，但未与full Transformer（二次复杂度）直接对比——这在追求亚二次复杂度的研究框架内合理，但读者应注意HAX的绝对性能仍低于同等规模Transformer。实验主要限于2K和4K上下文（Table 4），更长序列的scaling行为未验证。此外，sparsity budget k在组合变体中的分配方式（各k/2 vs 单一组件k）可能影响公平性，作者未明确讨论此细节。训练采用公开Mamba checkpoint的持续预训练，降低了从零训练的计算门槛，但具体FLOPs未披露。

## 方法谱系与知识库定位

**方法家族**：SSM-based efficient language models → Hybrid SSM-attention architectures

**Parent method**：Mamba / Mamba2（[14] S4-based selective state spaces; [4] structured state space duality）。HAX直接继承Mamba的线性时间SSM核心，在其基础上以plugin形式添加稀疏注意力模块，属于**结构性扩展**而非全新架构。

**直接baseline差异**：
- **Mamba/Mamba2**：HAX增加CDSA模块（LSH+KS），将纯recurrent压缩扩展为选择性稀疏attention，解决多查询联合召回的表达能力缺陷
- **LSH attention**（HashAttention）：HAX引入KS分支补偿其垂直条纹捕捉缺陷，并通过混合掩码机制统一两种稀疏模式
- **CISA variants**（LongNet's sliding window/dilated/A-shaped）：HAX将固定稀疏模式替换为上下文依赖的动态模式，sparsity pattern随输入变化
- **Samba**（仅合成任务提及）：HAX的KS模块提供显式学习的key重要性判断，而非依赖预设规则

**后续方向**：(1) 将CDSA框架扩展到多语言/多模态长序列场景；(2) 探索k的动态分配策略替代固定k/2+k/2；(3) 结合更高效的LSH变体或学习式哈希进一步降低常数开销。

**标签**：text modality | autoregressive language modeling | long-context scenario | sparse attention mechanism | sub-quadratic complexity constraint | hybrid SSM-attention architecture | ranking-based training

