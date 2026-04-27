---
title: Joint Relational Database Generation via Graph-Conditional Diffusion Models
type: paper
paper_level: A
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 图条件扩散模型的关系数据库联合生成
- GRDM (Graph-Cond
- GRDM (Graph-Conditional Relational Diffusion Model)
- Jointly modeling all tables in a re
acceptance: Poster
cited_by: 8
code_url: https://github.com/ketatam/rdb-diffusion
method: GRDM (Graph-Conditional Relational Diffusion Model)
modalities:
- structured data
- graph
paradigm: supervised
---

# Joint Relational Database Generation via Graph-Conditional Diffusion Models

[Code](https://github.com/ketatam/rdb-diffusion)

**Topics**: [[T__Text_Generation]] | **Method**: [[M__GRDM]] | **Datasets**: Berka RDB

> [!tip] 核心洞察
> Jointly modeling all tables in a relational database without imposing any order, using a graph-conditional diffusion model with graph neural networks, substantially outperforms autoregressive baselines in capturing multi-hop inter-table correlations while achieving state-of-the-art single-table fidelity.

| 中文题名 | 图条件扩散模型的关系数据库联合生成 |
| 英文题名 | Joint Relational Database Generation via Graph-Conditional Diffusion Models |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2505.16527) · [Code](https://github.com/ketatam/rdb-diffusion) · [Project](未提供) |
| 主要任务 | 关系数据库生成（Relational Database Generation） |
| 主要 baseline | ClavaDDPM, SDV, SingleTable, Denorm, TabDDPM |

> [!abstract] 因为「自回归方法强制固定表顺序、逐表生成导致错误累积且无法捕获多跳依赖」，作者在「ClavaDDPM」基础上改了「用图条件扩散模型联合生成所有表，以GNN进行K跳消息传递」，在「Berka RDB等六个真实数据库」上取得「Column Shapes 96.9 vs ClavaDDPM 94.6，Intra-Table Trends 98.21 vs 90.53」

- **单表保真度 SOTA**：Berka RDB 上 Column Shapes 96.9（+2.3 over ClavaDDPM），Intra-Table Trends 98.21（+7.7）
- **多表依赖捕获**：Inter-Table Trends 显著优于所有基线，1-hop 依赖建模能力超越自回归方法
- **首个非自回归范式**：首次实现关系数据库所有表的并行联合生成，无需预设表顺序

## 背景与动机

关系数据库生成（Relational Database Generation）旨在合成保留原始数据统计特性与表间关联结构的虚假数据库，广泛应用于隐私保护、数据增强与系统测试。现有主流方法采用自回归分解策略：先确定一个固定的表顺序（如父表优先），再逐表生成，每张子表以已生成的父表为条件。例如 ClavaDDPM 使用聚类标签和分类器引导扩散模型，依次生成子表；SDV 基于高斯 Copula 按层次结构顺序合成。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/e0d35e2b-2652-4601-9059-17d482cda69b/figures/Figure_1.png)
*Figure 1 (comparison): Comparison of auto-regressive and graph-conditional database generation.*



然而，这种自回归范式存在根本性缺陷。**第一**，固定表顺序限制了灵活性——对于缺失值填补等下游任务，无法利用子表信息反推父表；**第二**，条件独立性假设导致非相邻表之间的间接依赖被忽略，例如祖父表→父表→孙表的三跳关系无法有效建模；**第三**，逐表生成的错误会沿链条累积，降低整体合成质量。本文提出 GRDM（Graph-Conditional Relational Diffusion Model），首次以非自回归方式联合生成关系数据库的全部表，通过图神经网络同时去噪所有行属性，从根本上消除表顺序约束。

## 核心创新

核心洞察：将关系数据库表示为统一图结构（行→节点、外键→边），因为图神经网络的 K 跳消息传递能显式捕获多跳表间依赖，从而使所有表的并行联合去噪与生成成为可能。

| 维度 | Baseline (ClavaDDPM) | 本文 (GRDM) |
|:---|:---|:---|
| 生成范式 | 自回归：固定父→子顺序，逐表条件生成 | 非自回归：所有表同时联合生成 |
| 表间依赖建模 | 仅直接父子关系（1-hop），分类器引导条件 | K-hop 消息传递，显式多跳依赖捕获 |
| 数据结构 | 聚类标签，逐表或逐父子对处理 | 统一图表示：节点=行，边=外键 |
| 结构-属性关系 | 外键结构预设或独立生成 | 图结构与节点属性耦合联合扩散去噪 |

## 整体框架


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/e0d35e2b-2652-4601-9059-17d482cda69b/figures/Figure_2.png)
*Figure 2 (architecture): Tabular and graph representation of relational databases.*



GRDM 的整体流程包含五个核心阶段，形成从原始关系到合成数据的端到端管线：

1. **RDB-to-Graph 转换**：解析数据库模式，将每张表的每一行编码为图节点，外键关系编码为边，构建统一图表示 $G=(\mathbf{x}, \mathbf{A})$，其中 $\mathbf{x}$ 为节点属性（所有行所有列的拼接），$\mathbf{A}$ 为邻接矩阵。

2. **随机图结构初始化**：基于真实数据中观察到的组大小分布（每个父节点对应的子节点数量），使用随机图生成算法（泊松或 Molloy-Reed 配置模型）初始化扩散终点 $G_T$ 的拓扑结构。

3. **联合前向扩散**：同时对节点属性 $\mathbf{x}$ 和图结构 $\mathbf{G}$ 施加高斯噪声，生成时间步序列 $\{(\mathbf{x}_t, \mathbf{G}_t)\}_{t=1}^T$，而非仅对单表特征加噪。

4. **GNN 联合去噪**：以图神经网络为核心去噪器，通过 K 跳消息传递聚合多跳邻居信息，并行预测所有节点所有属性的干净值，同时预测外键边的存在概率。

5. **RDB 解码输出**：将最终去噪图 $(\mathbf{x}_0, \mathbf{G}_0)$ 解码回关系表结构，恢复原始模式的外键引用完整性。

```
原始RDB → [Schema Parser] → 统一图G(x,A)
                              ↓
                    [Random Graph Init] → G_T
                              ↓
              [Forward Diffusion] → {(x_t, G_t)}
                              ↓
         [GNN Denoiser with K-hop MP] → (x_0, G_0)
                              ↓
                    [RDB Decoder] → 合成数据库
```

## 核心模块与公式推导

### 模块 1: 图条件联合去噪分布（对应框架图 阶段3-4）

**直觉**：标准扩散模型逐表去噪无法建模表间依赖，需将去噪条件从"父表聚类标签"扩展为"完整图结构"。

**Baseline 公式** (ClavaDDPM 分类器引导条件扩散):
$$p_\theta(\mathbf{x}_t^{(child)} \text{mid} \mathbf{x}_{t+1}^{(child)}, \mathbf{c}_{parent}) = \mathcal{N}(\mathbf{x}_t^{(child)}; \mu_\theta(\mathbf{x}_{t+1}^{(child)}, \mathbf{c}_{parent}, t), \Sigma_t)$$
符号: $\mathbf{x}^{(child)}$ = 子表节点属性, $\mathbf{c}_{parent}$ = 父表聚类标签条件, $\theta$ = 去噪网络参数。

**变化点**：基线仅对单表去噪且条件为压缩后的聚类标签，丢失了父表完整分布信息及多表间接关系；本文将去噪对象扩展为全图所有节点，条件扩展为完整图拓扑。

**本文公式（推导）**:
$$\text{Step 1}: \quad p_\theta(\mathbf{x}_t, \mathbf{G}_t \text{mid} \mathbf{x}_{t+1}, \mathbf{G}_{t+1}) \quad \text{将DDPM从节点特征去噪扩展为联合图结构-属性去噪}$$
$$\text{Step 2}: \quad \mu_\theta, \Sigma_\theta = \text{GNN}_\theta(\mathbf{x}_{t+1}, \mathbf{G}_{t+1}, t) \quad \text{用GNN替代MLP，图结构同时作为输入和预测目标}$$
$$\text{最终}: \quad p_\theta(\mathbf{x}_t, \mathbf{G}_t \text{mid} \mathbf{x}_{t+1}, \mathbf{G}_{t+1}) = \mathcal{N}(\mathbf{x}_t; \mu_\theta, \Sigma_\theta) \cdot \text{Bernoulli}(\mathbf{G}_t; \sigma(\hat{\mathbf{G}}_\theta))$$

**对应消融**：SingleTable（K=0，无图条件）在 Inter-Table Trends 上显著劣于 GRDM，验证图条件必要性。

### 模块 2: K 跳 GNN 消息传递（对应框架图 GNN Denoiser 内部）

**直觉**：外键关系形成链式结构，2 跳以上邻居对应间接表依赖（如订单→客户→地区），需显式聚合而非隐含学习。

**Baseline 公式** (ClavaDDPM 无显式多跳):
$$\text{No explicit multi-hop; sequential conditioning on immediate parent only}$$

**变化点**：基线仅通过顺序堆叠隐式传递依赖，误差累积且无法并行；本文在单层 GNN 中通过 K 跳邻域聚合直接捕获任意跳数关系。

**本文公式（推导）**:
$$\text{Step 1}: \quad \mathcal{N}_K(v) = \{u : \text{dist}_G(u,v) \leq K\} \quad \text{定义K跳邻域，包含K跳内所有可达节点}$$
$$\text{Step 2}: \quad \mathbf{m}_v^{(l)} = \text{AGGREGATE}^{(l)}\left(\{\mathbf{h}_u^{(l-1)} : u \in \mathcal{N}_K(v)\}\right) \quad \text{聚合K跳邻居而非仅1跳邻居}$$
$$\text{最终}: \quad \mathbf{h}_v^{(l)} = \text{UPDATE}^{(l)}\left(\mathbf{h}_v^{(l-1)}, \mathbf{m}_v^{(l)}\right)$$
符号: $\mathbf{h}_v^{(l)}$ = 节点 $v$ 在第 $l$ 层的表示, $\mathcal{N}_K(v)$ = $v$ 的 K 跳邻居集合, AGGREGATE/UPDATE = 图神经网络的标准聚合与更新函数。

**对应消融**：Table 2 (K hops) 显示 K 值选择对 Inter-Table Trends 的定量影响，验证多跳机制有效性。

### 模块 3: 联合训练目标（对应框架图 训练阶段）

**直觉**：仅优化属性重建会导致生成正确数值但错误外键引用（或反之），需强制模型同时学习"什么值"和"连向谁"。

**Baseline 公式** (标准 DDPM / TabDDPM):
$$\mathcal{L}_{DDPM} = \mathbb{E}_{q(\mathbf{x}_0)}\mathbb{E}_{t\sim[1,T]}\left[\|\mathbf{x}_t - \hat{\mathbf{x}}_\theta(\mathbf{x}_t, \mathbf{c}, t)\|^2\right]$$
符号: $\mathbf{c}$ = 外部条件（如聚类标签）, $\hat{\mathbf{x}}_\theta$ = 预测干净特征的神经网络。

**变化点**：基线损失仅含节点属性 MSE，无结构项；本文增加图结构重建损失，实现属性-结构耦合优化。

**本文公式（推导）**:
$$\text{Step 1}: \quad \mathcal{L}_{attr} = \mathbb{E}_{q(\mathbf{x}_0, \mathbf{G}_0)}\mathbb{E}_{t}\left[\|\mathbf{x}_t - \hat{\mathbf{x}}_\theta(\mathbf{x}_t, \mathbf{G}_t, t)\|^2\right] \quad \text{保留标准节点属性去噪损失，但条件变为完整图}$$
$$\text{Step 2}: \quad \mathcal{L}_{struct} = \text{BCE}(\mathbf{G}_t, \hat{\mathbf{G}}_\theta(\mathbf{x}_t, \mathbf{G}_t, t)) \quad \text{新增外键边存在的二元交叉熵损失}$$
$$\text{Step 3}: \quad \lambda = \text{平衡系数，控制结构重建相对权重}$$
$$\text{最终}: \quad \mathcal{L}_{joint} = \mathcal{L}_{attr} + \lambda \cdot \mathcal{L}_{struct}$$

**对应消融**：Denorm（Join-Split 替代联合生成）Intra-Table Trends 仅 72.12 vs GRDM 98.21，证明分离结构/属性生成的灾难性后果。

## 实验与分析


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/e0d35e2b-2652-4601-9059-17d482cda69b/figures/Table_1.png)
*Table 1 (quantitative): Comparison of the Fidelity metrics described in Section 4.1.*



本文在六个真实世界关系数据库上评估 GRDM，包括 Berka RDB、MIMIC-III、PubMed 等。核心评估指标采用 SDV 定义的保真度度量：Column Shapes（列分布相似度）、Intra-Table Trends（表内趋势保留度）、Inter-Table Trends（表间趋势保留度）及 Cardinality（行数匹配度）。

以 Berka RDB 为例，GRDM 在 Column Shapes 达到 96.9，相比主要基线 ClavaDDPM 的 94.6 提升 +2.3；Intra-Table Trends 达到 98.21，相比 ClavaDDPM 的 90.53 提升 +7.7；Cardinality 达到 99.7，相比 ClavaDDPM 的 96.75 提升 +2.95。这些数字表明联合生成不仅未牺牲单表质量，反而因消除顺序约束和错误累积而全面超越自回归方法。特别值得注意的是，Inter-Table Trends（1-hop）上 GRDM 相对 ClavaDDPM 的优势最为显著——这正是多跳消息传递直接针对的痛点。



消融实验进一步验证各组件的必要性。将 K 设为 0（即 SingleTable 变体）后，模型退化为独立逐表生成，Inter-Table Trends 显著劣化，证明图条件与联合去噪的必要性。Denorm 基线采用先 Join 所有表再 Split 的策略，其 Intra-Table Trends 暴跌至 72.12（-26.09 vs GRDM），揭示分离结构生成与属性生成的致命缺陷。ClavaDDPM 的自回归顺序生成在所有指标上均落后，尤其在 Inter-Table Trends 上差距最大，验证固定顺序导致的条件独立性假设问题。

公平性方面，作者明确披露若干局限：SDV 不支持 5 张表以上的数据库，导致复杂 schema 上对比不完整；评估仅覆盖六个数据库，更复杂多跳 schema 的泛化未验证；未与同期工作 [19]（Relational data generation with GNNs and latent diffusion models）直接对比。此外，SingleTable 和 Denorm 实为作者构造的消融变体而非独立发表方法，可能低估与真正 SOTA 的差距——GOGGLE [9]、TabDiff [11] 等潜在强基线未纳入比较。

## 方法谱系与知识库定位

**方法家族**：扩散模型 → 表格数据扩散（TabDDPM [8]）→ 多关系数据扩散（ClavaDDPM [15]）→ **图条件联合扩散（GRDM）**

**父方法**：ClavaDDPM（直接基线）。GRDM 保留其 DDPM 训练骨干，但系统性替换了四大核心槽位：
- **架构**：自回归因子分解 → 统一图表示 + GNN 联合去噪
- **推理策略**：顺序表生成 → 并行全图联合去噪
- **数据管线**：聚类标签/逐表处理 → 行-节点/外键-边图编码
- **训练配方**：单表去噪 MSE → 属性-结构联合损失

**直接差异对比**：
- vs **TabDDPM**：从单表扩展到多表关系设置，引入图条件与结构生成
- vs **SDV**：从统计 Copula 方法转向深度扩散，支持无表顺序约束的并行生成
- vs **GOGGLE [9]**（未比较）：GOGGLE 学习关系结构但非扩散框架，GRDM 首次将图条件扩散用于此任务

**后续方向**：(1) 扩展到百万行级大规模数据库的采样效率优化；(2) 与 LLM-based 表格生成方法 [12] 的融合；(3) 差分隐私约束下的图条件扩散训练 [14]。

**标签**：`结构化数据` / `扩散模型` / `关系数据库生成` / `图神经网络` / `非自回归生成` / `多跳依赖建模` / `联合结构-属性生成`

## 引用网络

### 直接 baseline（本文基于）

- Mixed-Type Tabular Data Synthesis with Score-based Diffusion in Latent Space _(ICLR 2024, 实验对比, 未深度分析)_: Tabular diffusion model in latent space, likely compared against as baseline in 
- Position: Relational Deep Learning - Graph Representation Learning on Relational Databases _(ICML 2024, 方法来源, 未深度分析)_: Graph representation learning on relational databases, likely provides methodolo
- Fingerprinting Denoising Diffusion Probabilistic Models _(CVPR 2025, 方法来源, 未深度分析)_: Foundational DDPM paper, core algorithmic basis for the diffusion model approach

