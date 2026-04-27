---
title: 'Fast attention mechanisms: a tale of parallelism'
type: paper
paper_level: A
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 基于近似最近邻搜索的快速并行注意力机制
- ANNA (Approximat
- ANNA (Approximate Nearest Neighbor Attention)
- Approximate Nearest Neighbor Attent
acceptance: Poster
cited_by: 1
method: ANNA (Approximate Nearest Neighbor Attention)
modalities:
- Text
---

# Fast attention mechanisms: a tale of parallelism

**Topics**: [[T__Reasoning]] | **Method**: [[M__ANNA]]

> [!tip] 核心洞察
> Approximate Nearest Neighbor Attention (ANNA) achieves sub-quadratic time complexity while retaining the full expressive power of standard attention for simulating MPC algorithms and solving key reasoning tasks with near-optimal depth.

| 中文题名 | 基于近似最近邻搜索的快速并行注意力机制 |
| 英文题名 | Fast attention mechanisms: a tale of parallelism |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2509.09001) · Code (未公开) · Project (未公开) |
| 主要任务 | Reasoning, MPC simulation, k-hop reasoning, Match2 |
| 主要 baseline | Standard softmax attention, Reformer (LSH attention), Performer (FAVOR+), Scatterbrain, Linformer |

> [!abstract] 因为「标准注意力机制具有 O(N²) 时间复杂度且高效注意力方法（如 Reformer 的 LSH attention）在常数深度下表达能力受限」，作者在「Standard softmax attention」基础上改了「将精确全对点积相似度替换为基于 LSH 的近似最近邻搜索（ANN）」，在「MPC 模拟与推理任务理论分析」上取得「ANNA-transformer 以 O(R) 层模拟 R 轮 MPC 协议，以 O(log k) 层解决 k-hop 推理，且常数深度即可模拟常数深度低秩 transformer」

- **核心性能**: ANNA 达到次二次时间复杂度（sub-quadratic），具体为通过 ANN 检索替代 O(N²) 全对计算
- **理论保证**: 模拟 R 轮 MPC 仅需 L = O(R) 层，头数 H = O(N^(ε′−ε)/4)，嵌入维度 m = O(N^ε′)
- **分离结果**: Reformer 在常数桶大小 B = O(1) 时需 Ω(log_B N) = Ω(log N) 层，而 ANNA 在常数深度下可解决相同任务

## 背景与动机

Transformer 的自注意力机制是现代大语言模型的核心组件，但其计算复杂度随序列长度 N 呈二次增长——当处理长文档或高分辨率序列时，O(N²) 的矩阵乘法成为不可扩展的瓶颈。例如，一个 4096 token 的序列需要计算超过 1600 万对 query-key 相似度，内存与计算开销急剧膨胀。

现有高效注意力方法主要从三个方向缓解这一问题：
- **Reformer** 采用 Locality-Sensitive Hashing (LSH) 将相似 query-key 对哈希到同一桶中，实现线性内存复杂度，但其常数桶大小的设计导致感受野受限——每个位置仅能依赖 B^L 个输入位置；
- **Performer (FAVOR+)** 通过随机特征映射将注意力矩阵近似为低秩分解，达到线性复杂度，但低秩假设可能丢失精细的局部结构；
- **Scatterbrain** 尝试统一稀疏与低秩近似，然而缺乏对表达能力损失的严格理论刻画。

这些方法的共同短板在于：**没有严格证明次二次注意力能否保留标准注意力的全部表达能力**。特别地，Sanford 等人（2023）的「Fast attention requires bounded entries」与后续工作「Fundamental limitations on subquadratic alternatives to transformers」揭示了快速注意力的根本性限制——Reformer 在常数深度下甚至无法完成全局信息传播。这引出一个核心问题：是否存在一种高效注意力机制，既能突破二次复杂度，又能在理论上保持与标准注意力同等的表达能力（如对 MPC 协议的模拟能力）？

本文提出 ANNA（Approximate Nearest Neighbor Attention），通过近似最近邻搜索替代精确全对计算，首次在保持次二次复杂度的同时，证明了其对并行计算模型的完整模拟能力。

## 核心创新

**核心洞察**：近似最近邻搜索（ANN）足以替代精确的全对点积计算，因为 LSH 能以高概率保留真正重要的 query-key 匹配关系，从而使次二次时间复杂度与完整 MPC 模拟能力的共存成为可能。

| 维度 | Baseline (Standard softmax attention) | 本文 (ANNA) |
|:---|:---|:---|
| **注意力计算** | 精确全对 QK^T，O(N²) 时间 | ANN 检索，次二次时间 |
| **关键值匹配** | 所有 N² 对参与 softmax | 仅最近邻候选参与，稀疏模式 |
| **理论保证** | 天然精确，无近似误差 | LSH 近似 + MPC 模拟正确性定理 |
| **深度效率** | Reformer 需 Ω(log N) 层全局推理 | 常数深度即可，O(log k) 解决 k-hop |
| **参数规模** | EMA-transformer: m = O(N^{5ε} log N) | ANNA: m = O(N^ε′)，头数 H = O(N^(ε′−ε)/4) |

与现有高效注意力的本质差异：ANNA 不是对注意力矩阵的「低秩近似」或「固定稀疏模式」，而是**动态地、按需地**通过 ANN 检索确定每个 query 应关注哪些 key，且这一近似在理论上是可控的——Theorem 4.1 严格证明了 ANNA-transformer 能精确模拟任意确定性 MPC 协议的输出。

## 整体框架


![Figure 6](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/d681d18f-a4cb-4137-a94f-73651bb421a6/figures/Figure_6.png)
*Figure 6 (ablation): Figure 6: Wall-clock times are averaged over 10 runs. (a) Model: x-axis denotes the number of layers, and different curves denote the number of heads. Red corresponds to the number of heads taken in the table. (b) Attention heads: x-axis denotes the number of heads taken, the color axis denotes error. Curves correspond to number of layer taken in the table. The reported values are in the best solution among the choice of (1, 2, 3, 4, 5, 6, 7, 8, 10, 16, 20, 30, 40, 50).*



ANNA-transformer 的数据流遵循标准 transformer 的层叠结构，但核心注意力模块被重新设计为基于 ANN 的稀疏检索-聚合流程：

1. **Input embedding**: 原始 token 序列 x ∈ Σ^N 经嵌入层转换为连续表示，与标准 transformer 一致；
2. **ANN query-key matching**（**新增**）: 将 Q, K 投影后，通过 LSH 构建的近似最近邻数据结构，为每个 query 检索 top 相关 key 的索引集合，替代完整的 QK^T 矩阵计算；
3. **Sparse attention aggregation**（**新增**）: 仅对 ANN 检索得到的 key-value 子集执行注意力计算与加权求和，输出上下文表示；
4. **Feed-forward network**: 标准位置前馈网络完成层内非线性变换。

关键理论组件（非直接实现模块）：
- **MPC simulation wrapper**: 将 transformer 层映射到 MPC 轮次的分析框架，用于 Theorem 4.1 的复杂性证明；
- **Binary search construction**: 针对 k-hop 推理任务（Theorem 5.6），在 ANN 结构上实现 O(log k) 层的二进制搜索式信息传递。

```
Raw tokens x ∈ Σ^N
    ↓
[Input Embedding]
    ↓
[ANN Query-Key Matching]  ← LSH-based retrieval, sub-quadratic time
    ↓ (sparse indices)
[Sparse Attention Aggregation]  ← only selected K-V pairs
    ↓
[Feed-Forward Network]
    ↓
Layer output (repeat L times)
```

整个框架的核心创新在于：ANN 检索模块不仅降低了计算复杂度，其稀疏模式还恰好与 MPC 模拟中「局部通信」的需求相契合，从而实现了效率与表达能力的双赢。

## 核心模块与公式推导

### 模块 1: ANN 注意力计算（对应框架图第 2 步）

**直觉**: 标准注意力的 O(N²) 瓶颈源于计算所有 query-key 对；而实际中每个 query 通常只与少数 key 高度相关，ANN 检索可动态定位这些关键匹配。

**Baseline 公式** (Standard softmax attention):
$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^\text{top}}{\sqrt{d_k}}\right)V$$
符号: $Q, K, V \in \mathbb{R}^{N \times d_k}$ 为 query/key/value 矩阵，$d_k$ 为 head 维度。

**变化点**: 精确计算 $QK^\text{top} \in \mathbb{R}^{N \times N}$ 需要 O(N²) 时间与空间；当 N 增大时不可扩展。LSH-based ANN 可将检索复杂度降至次二次。

**本文公式（推导）**:
$$\text{Step 1}: \mathcal{I}_i = \text{ANN-Retrieve}(q_i, \{k_1, \ldots, k_N\}) \quad \text{通过 LSH 为每个 query } q_i \text{ 找到候选 key 集合}$$
$$\text{Step 2}: \tilde{a}_{ij} = \begin{cases} \exp(q_i^\text{top} k_j / \sqrt{d_k}) & j \in \mathcal{I}_i \\ 0 & \text{otherwise} \end{cases} \quad \text{仅对候选集计算相似度，其余置零}$$
$$\text{Step 3}: \alpha_{ij} = \frac{\tilde{a}_{ij}}{\sum_{j' \in \mathcal{I}_i} \tilde{a}_{ij'}} \quad \text{在候选集上重归一化，保证概率分布性质}$$
$$\text{最终}: \text{ANNA}(Q,K,V)_i = \sum_{j \in \mathcal{I}_i} \alpha_{ij} v_j$$

**对应消融**: 本文未提供实验消融（纯理论工作），但 Proposition E.3 从反面论证：若将 ANN 替换为 Reformer 的固定桶 LSH（即 $|\mathcal{I}_i|$ 受桶大小 B 限制），则常数深度下感受野仅为 B^L，无法全局推理。

---

### 模块 2: MPC 模拟的架构参数设计（对应框架图整体，Theorem 4.1）

**直觉**: 标准 transformer 能模拟 MPC 是已知的，但需要证明 ANN 的稀疏性不破坏这一能力——关键在于 ANN 保留的边足够"随机"以模拟 MPC 的通信模式。

**Baseline 公式** (EMA-transformer，中间构造):
$$L = R+1, \quad H = O(N^\varepsilon), \quad m = O(N^{5\varepsilon} \log N)$$
符号: $L$ = 层数, $H$ = 头数, $m$ = 嵌入维度, $R$ = MPC 轮数, $\varepsilon$ 为任意小正常数。

**变化点**: EMA-transformer 的嵌入维度 $m = O(N^{5\varepsilon} \log N)$ 过大，且未利用 ANN 结构优化参数。ANNA 通过更紧的分析将维度降至 $O(N^{\varepsilon'})$，头数降至次线性。

**本文公式（推导）**:
$$\text{Step 1}: \text{利用 ANN 结构，每个 query 仅需 } O(N^{\varepsilon'}) \text{ 个候选 key 即可保证模拟正确性}$$
$$\text{Step 2}: \text{多头机制分解负载：} H = O(N^{(\varepsilon'-\varepsilon)/4}) \text{ 个头各自处理部分通信，乘积 } H \cdot m = O(N^{\varepsilon' + (\varepsilon'-\varepsilon)/4}) \text{ 控制总参数量}$$
$$\text{Step 3}: \text{层数与 MPC 轮数线性对应：每层模拟一轮通信，ANN 检索模拟该轮的消息路由}$$
$$\text{最终 (Theorem 4.1)}: L = O(R), \quad H = O(N^{(\varepsilon'-\varepsilon)/4}), \quad m = O(N^{\varepsilon'})$$

其中 $0 < \varepsilon < \varepsilon' < 1$ 为可控近似参数。与 EMA-transformer 相比，嵌入维度从 $O(N^{5\varepsilon} \log N)$ 改进为 $O(N^{\varepsilon'})$，头数从 $O(N^\varepsilon)$ 改进为 $O(N^{(\varepsilon'-\varepsilon)/4})$，均为严格的次线性改进。

---

### 模块 3: k-hop 推理的二进制搜索构造（对应框架图第 3 步扩展，Theorem 5.6）

**直觉**: 图上的 k-hop 推理需要逐层传播信息；利用 ANN 的"距离缩减"特性，可设计每层将搜索空间减半的二进制搜索策略。

**Baseline**: 标准 transformer 或 GNN 通常需要 O(k) 层完成 k-hop 推理；Reformer 因感受野限制甚至需要更多。

**本文公式（推导）**:
$$\text{Step 1}: \text{初始化查询节点表示，定义目标为找到距离恰好为 } k \text{ 的节点}$$
$$\text{Step 2}: \text{每层通过 ANN 检索"中间层"候选：} \mathcal{I}_i^{(l)} = \text{ANN-Retrieve}(h_i^{(l)}, \text{nodes at approx. } k/2^{l+1} \text{ distance})$$
$$\text{Step 3}: \text{递归减半：第 } l \text{ 层将剩余跳数从 } k/2^{l-1} \text{ 降至 } k/2^l$$
$$\text{最终 (Theorem 5.6)}: L = O(\log k), \quad H = O(N^{(\varepsilon'-\varepsilon)/4}), \quad m = O(N^{\varepsilon'})$$

这一构造的关键在于 ANN 的动态检索能力——不同于固定稀疏模式（如 BigBird 的随机/窗口/全局混合），ANN 能根据当前表示自适应地选择"中间节点"，从而实现高效的二分搜索式信息路由。

## 实验与分析



本文为一项**纯理论工作**，未在标准 NLP benchmark（如 GLUE、LRA）上进行实证评估。其核心"实验"为一系列定理证明与复杂性分析，主要结果可概括如下。



**MPC 模拟能力（Theorem 4.1）**: ANNA-transformer 以 L = O(R) 层、H = O(N^(ε′−ε)/4) 头、m = O(N^ε′) 维度的资源，精确模拟任意确定性 R 轮 MPC 协议。与 EMA-transformer 基线相比，嵌入维度从 O(N^{5ε} log N) 压缩至 O(N^ε′)，头数从 O(N^ε) 降至 O(N^(ε′−ε)/4)。这一结果表明：**ANN 近似不会损失 transformer 对并行计算模型的模拟能力**——解决了"高效注意力是否必然牺牲表达能力"的核心疑问。

**特定任务深度效率（Theorem 5.4 & 5.6）**: 对于 Match2 匹配任务，常数深度（L = O(1)）的 ANNA-transformer 即可解决；对于 k-hop 推理，仅需 L = O(log k) 层。作为对比，Proposition E.3 证明 Reformer 在常数桶大小 B = O(1) 时需要 L = Ω(log_B N) = Ω(log N) 层才能完成全局信息传播——当 k = O(log N) 时，ANNA 的 O(log k) = O(log log N) 远优于 Reform 的 Ω(log N)。



**理论消融与分离结果**: 本文的核心"消融"体现为与 Reformer 的分离定理。Reformer 的感受野公式为 $B^L$（B = 桶大小，L = 层数），当 B = O(1) 且 L = O(1) 时，每个位置仅能依赖常数个输入位置，无法完成需要全局交互的任务。ANNA 通过 ANN 的动态检索突破了这一限制——其有效感受野随层数指数增长，但检索成本保持次二次。

**公平性检验**: 本文的比较存在明显局限。首先，**缺乏实证验证**：所有结果均为渐近复杂性分析，未报告实际 wall-clock 时间或内存占用。Figure 6 虽标注为 ablation，但描述显示其为"Wall-clock times averaged over 10 runs"——然而正文未引用此图，且分析 artifacts 中未提取到具体数值。其次，**缺失关键基线**：未与 FlashAttention（IO-aware 精确加速）、Linear Attention（Katharopoulos et al.）、RWKV、Mamba 等状态空间模型比较，这些方法在实践中已证明可实现线性或次二次复杂度。第三，**参数 ε, ε′ 缺乏实用指导**：理论结果允许任意 0 < ε < ε′ < 1，但未说明何种设置能在真实硬件上平衡速度与精度。作者明确承认"practical wall-clock speedups not empirically validated"与"no empirical evaluation on standard downstream NLP tasks"。

## 方法谱系与知识库定位

**方法家族**: Efficient Transformer Attention → Sub-quadratic Attention Mechanisms

**父方法**: Standard softmax attention（Vaswani et al., 2017）— ANNA 保留其层叠架构与多头设计，但将核心注意力计算从精确全对替换为 ANN 检索。

**中间构造**: EMA-transformer — 用于 MPC 模拟证明的中间体，ANNA 改进其参数 scaling（m: O(N^{5ε} log N) → O(N^ε′), H: O(N^ε) → O(N^(ε′−ε)/4)）。

**直接基线对比**:
- **Reformer (LSH attention)**: 同样使用 LSH，但采用固定桶大小导致感受野受限；ANNA 通过 ANN 的动态检索实现常数深度全局推理，理论上严格分离
- **Performer (FAVOR+)**: 低秩核方法近似；ANNA 证明常数深度可模拟常数深度低秩 transformer，将其纳入统一分析框架
- **Scatterbrain**: 稀疏+低秩统一；ANNA 提供另一种统一视角（ANN 模拟低秩）

**后续方向**:
1. **实证验证**: 将 ANNA 实现为可运行模块，在 Long Range Arena 等长序列 benchmark 上验证 wall-clock 加速与精度 trade-off
2. **端到端训练**: 当前理论假设 ANN 结构已知；研究可学习的 LSH 参数或自适应邻居数选择
3. **与状态空间模型连接**: Mamba 等模型已证明线性复杂度的实践可行性，探索 ANNA 与选择性状态空间的理论等价性或互补性

**标签**: text / transformer / reasoning / attention_mechanism / sub_quadratic_complexity / theoretical_analysis / MPC_simulation / approximate_nearest_neighbor / LSH

## 引用网络

### 直接 baseline（本文基于）

- Simple linear attention language models balance the recall-throughput tradeoff _(ICML 2024, 实验对比, 未深度分析)_: Recent linear attention method, likely appears in experiments as comparison base

