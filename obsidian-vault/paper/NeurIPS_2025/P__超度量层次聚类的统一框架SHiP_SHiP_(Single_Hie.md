---
title: 'Ultrametric Cluster Hierarchies: I Want ‘em All!'
type: paper
paper_level: C
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 超度量层次聚类的统一框架SHiP
- SHiP (Single Hie
- SHiP (Single Hierarchy Partitioning) framework
acceptance: Poster
code_url: https://github.com/pasiweber/SHiP-framework/
method: SHiP (Single Hierarchy Partitioning) framework
---

# Ultrametric Cluster Hierarchies: I Want ‘em All!

[Code](https://github.com/pasiweber/SHiP-framework/)

**Method**: [[M__SHiP_(Single_Hierarchy_Partitioning)_framework]] | **Datasets**: Clustering runtime

| 中文题名 | 超度量层次聚类的统一框架SHiP |
| 英文题名 | Ultrametric Cluster Hierarchies: I Want 'em All! |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2502.14018) · [Code](https://github.com/pasiweber/SHiP-framework/) · [DOI](https://doi.org/10.48550/arxiv.2502.14018) |
| 主要任务 | 层次聚类、平面聚类、多粒度聚类解生成 |
| 主要 baseline | k-means (Euclidean)、Ward agglomerative clustering、AMD-DBSCAN、DPC、SCAR |

> [!abstract] 因为「传统聚类算法每次只能产出单一聚类解，且探索不同参数需反复从头计算」，作者在「标准聚类流程」基础上改了「单次超度量计算+多层次/划分快速生成」的两阶段架构，在「Boxes、MNIST等数据集」上取得「超度量构建后与密度方法相当，而层次生成与划分仅需毫秒级（如Boxes上33ms+3ms+134ms）」

- **运行效率**：SHiP在Boxes数据集上Cover tree超度量计算仅需0.059s，后续Stability层次生成33ms、MoE层次3ms、Elbow划分134ms
- **对比优势**：Ward在MNIST上耗时19分02秒，而SHiP Cover tree超度量计算为16分24秒，后续所有层次与划分均在毫秒级完成
- **灵活性**：单次超度量计算后可快速生成多种聚类解，无需重复运行完整算法

## 背景与动机

聚类分析是探索性数据分析的核心工具，但用户常面临一个困境：不同场景需要不同粒度或不同准则的聚类结果。例如，一位生物学家分析基因表达数据时，上午可能需要k-means风格的紧凑簇来识别细胞亚型，下午又需要密度-based的层次结构来发现稀有细胞群，晚上还想尝试不同k值的划分方案——而每次切换都意味着重新运行算法、调参、等待。

现有方法各自解决部分问题，但无法统一满足"全都要"的需求：

- **k-means**：基于欧氏距离优化，速度快但只能产出单一平面划分，且需预设k值；更换k必须重新计算。
- **Ward层次聚类**：能生成树状结构，但计算复杂度高（O(n²)或更高），且一旦建完树，提取不同层次虽快，但更换聚类准则（如从k-means目标换成k-median目标）需重新建树。
- **密度方法（DBSCAN/AMD-DBSCAN/DPC）**：能发现任意形状簇，但参数敏感（ε、MinPts等），且每次参数调整需重新执行完整的密度估计与标签传播。

这些方法的共同瓶颈在于：**聚类结构与聚类准则紧密耦合**。若想尝试不同准则（k-means vs k-median vs k-center）或不同粒度，必须放弃已计算的结构、从头开始。这种"一次一解"的模式在交互式探索中效率极低。

本文的核心动机正是打破这一耦合：能否**预先计算一次通用的数据结构**，之后以毫秒级成本任意切换聚类准则与粒度？作者发现超度量（ultrametric）结构恰好能承担这一"中间表示"的角色——它编码了所有可能的层次聚类，且支持多种优化准则的高效提取。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/2f3105f9-7cf3-4a7a-986f-6d39dd2aad6e/figures/Figure_1.png)
*Figure 1 (pipeline): Overview of our proposed SHP clustering framework in which we (1) do ultrametric closure to obtain a tree/forest (i.e. the ultrametric), and (2) cluster symbionts from the hierarchy.*



## 核心创新

核心洞察：超度量闭包（ultrametric closure）可以作为聚类的通用中间表示，因为超度量天然编码了层次嵌套结构且满足强三角不等式，从而使"一次计算、多种准则、毫秒提取"成为可能。

| 维度 | Baseline | 本文 |
|:---|:---|:---|
| 计算模式 | 每改参数/准则重新运行完整算法 | 单次超度量计算，后续层次与划分毫秒级完成 |
| 数据结构 | 直接对原始数据或距离矩阵操作 | 先构建超度量树（Cover tree或DC tree），再在其上操作 |
| 准则灵活性 | 算法与准则绑定（如k-means只优化k-means目标） | 同一超度量上可切换Stability（k-center）、MoE（k-median）、Elbow（k-means）三种准则 |
| 输出类型 | 单一平面聚类或固定层次 | 任意层次 + 任意粒度平面划分均可快速生成 |

与经典层次聚类的关键差异在于：传统方法如Ward是"自底向上"逐步合并，最终树结构由合并顺序唯一确定；而SHiP先通过超度量闭包构建**所有可能层次结构的超集**，再通过不同准则"筛选"出最优层次，实现了结构与准则的解耦。

## 整体框架



SHiP框架采用清晰的三阶段流水线设计，输入原始数据点，输出多样化的聚类解：

**阶段一：超度量计算（Ultrametric Computation）**
- 输入：原始数据点集
- 输出：超度量距离结构（以树/森林形式存储）
- 角色：构建通用中间表示，是整个框架的"一次性投资"
- 两种实现：Cover tree（基于覆盖树的高效构建）或 DC tree（密度连接树，与密度方法相当的运行时特性）

**阶段二：层次生成（Hierarchy Generation）**
- 输入：超度量结构
- 输出：具体的聚类层次（树状结构）
- 角色：根据用户选择的准则从超度量中提取最优层次
- 三种准则：Stability（基于k-center/stability）、MoE（Minimize-over-Error，基于k-median）、Elbow（基于k-means）

**阶段三：划分（Partitioning）**
- 输入：聚类层次
- 输出：指定簇数的平面聚类
- 角色：将层次结构"切割"为给定k值的扁平划分
- 实现：k-means变体，在超度量空间而非原始空间执行

数据流可概括为：

```
Raw Data → [Cover Tree / DC Tree] → Ultrametric Structure
                                          ↓
              ┌─────────────┬─────────────┼─────────────┐
              ↓             ↓             ↓             ↓
         [Stability]    [MoE]        [Elbow]      (other criteria)
              ↓             ↓             ↓             ↓
         Hierarchy 1   Hierarchy 2  Hierarchy 3   ...
              └─────────────┴─────────────┴─────────────┘
                                          ↓
                              [k-means Partitioning]
                                          ↓
                              Flat Clustering (any k)
```

关键优势：阶段一执行一次后，阶段二和阶段三可反复以毫秒级成本探索不同组合（3种准则 × 任意k值）。

## 核心模块与公式推导

本文在公式层面侧重于算法框架与复杂性分析，而非端到端的可微分损失函数。以下解析三个核心模块的设计原理与计算目标。

### 模块 1: 超度量闭包（Ultrametric Closure）（对应框架图 阶段一）

**直觉**：普通度量空间中的距离不满足层次聚类的嵌套要求；超度量通过强化三角不等式为 $d(x,z) \leq \max(d(x,y), d(y,z))$，使得任意三点中最长两边相等，天然对应树状结构的层级关系。

**Baseline 形式**（标准层次聚类如Ward）：
给定数据点集 $X = \{x_1, ..., x_n\}$，Ward方法逐步合并簇，合并代价为
$$\Delta(A,B) = \frac{|A||B|}{|A|+|B|} \|\mu_A - \mu_B\|^2$$
其中 $\mu_A, \mu_B$ 为簇中心。该过程产生**单一**层次结构，且合并顺序由欧氏距离与簇大小共同决定，无法分离"结构"与"准则"。

**变化点**：Ward的贪心合并将距离信息与聚类准则（最小方差）纠缠在一起；若改用k-median准则，需重新运行完全不同的算法。

**本文方法**：
先通过Cover tree或DC tree构建**超度量空间** $(X, u)$，其中超度量 $u$ 满足：
$$\text{Step 1}: u(x,z) \leq \max(u(x,y), u(y,z)) \quad \text{（强化三角不等式，保证树结构）}$$
$$\text{Step 2}: u(x,y) = \min\{\max_{i} d(p_i, p_{i+1}) \text{mid} p_0=x, p_n=y\} \quad \text{（最小最大路径，将任意度量转换为超度量）}$$
$$\text{最终}: u = \text{ultrametric closure}(d) \text{ 其中 } d \text{ 为原始距离}$$
该闭包操作保证：对任意原始距离 $d$，存在唯一的最大下界超度量 $u \leq d$，且 $u$ 编码了所有与 $d$ "兼容"的层次结构。

**对应消融**：Cover tree在Boxes上仅需0.059s，DC tree构建时间与密度方法相当（Table 2）。

### 模块 2: 层次准则提取（Stability / MoE / Elbow）（对应框架图 阶段二）

**直觉**：同一超度量树上，不同聚类准则对应不同的"切割"策略——k-center关注最大半径，k-median关注总距离，k-means关注平方误差。

**Baseline 公式**（标准k-means目标）：
$$L_{\text{k-means}} = \sum_{i=1}^{k} \sum_{x \in C_i} \|x - \mu_i\|^2$$
其中 $\mu_i$ 为簇 $C_i$ 的均值，优化需迭代至收敛。

**变化点**：传统方法直接在原始空间优化，每次更换k需重新初始化迭代；且k-means目标与k-median、k-center互不兼容。

**本文方法**：在超度量树 $T$ 上，三种准则转化为树上的动态规划问题：
$$\text{Step 1}: \text{Stability}(T, k) = \min_{\text{k个簇}} \max_{i} \text{radius}(C_i) \quad \text{（k-center准则，最小化最大簇半径）}$$
$$\text{Step 2}: \text{MoE}(T, k) = \min_{\text{k个簇}} \sum_{i} \sum_{x \in C_i} u(x, \text{median}_i) \quad \text{（k-median准则，总距离最小）}$$
$$\text{Step 3}: \text{Elbow}(T, k) = \min_{\text{k个簇}} \sum_{i} \sum_{x \in C_i} u(x, \mu_i)^2 \quad \text{（k-means准则，平方误差最小）}$$
$$\text{关键}: \text{所有准则均在超度量树上通过一次自底向上遍历求解，复杂度 } O(n \cdot k) \text{ 或更优}$$

由于超度量的树结构特性，簇的合并/分裂具有明确的最优子结构，避免了原始空间中的迭代优化。

**对应消融**：三种层次生成方法在Boxes上分别耗时33ms（Stability）、3ms（MoE）、未明确但同量级（Elbow）（Table 2）。

### 模块 3: 超度量划分（Partitioning）（对应框架图 阶段三）

**直觉**：层次结构确定后，提取特定k的平面聚类只需在树上做"水平切割"，但切割位置的选择仍需优化准则指导。

**Baseline 公式**（标准k-means++初始化+迭代）：
$$L_{\text{partition}} = \sum_{i=1}^{k} \sum_{x \in C_i} \|x - c_i\|^2 + \text{迭代优化开销}$$

**变化点**：原始空间k-means受限于欧氏几何，且对初始化敏感；在超度量空间中，"中心"和"距离"有组合解释。

**本文方法**：
$$\text{Step 1}: \text{在超度量树 } T \text{ 上，每个节点有明确的"高度"} h(v) = u\text{-半径}$$
$$\text{Step 2}: \text{选择k个中心节点，使得} \sum_{x} u(x, \text{nearest center}) \text{最小化}$$
$$\text{Step 3}: \text{利用超度量强三角不等式，最近中心分配可在 } O(n) \text{ 完成，无需迭代}$$
$$\text{最终}: \text{划分结果} = \text{k-means-variant}(T, k) \text{ 在超度量空间执行，耗时毫秒级}$$

**对应消融**：Elbow划分在Boxes上耗时134ms（Table 2），相比重新运行完整k-means（0.114s-1.217s）具有数量级优势，尤其当需要多个k值时。

## 实验与分析


![Table 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/2f3105f9-7cf3-4a7a-986f-6d39dd2aad6e/figures/Table_2.png)
*Table 2 (quantitative): Runtimes of our SHP framework's components. (left to right) Column groups and symbiont computation, ultrametric closure, and clustering from the hierarchy. Times are in seconds.*



本文的核心实验聚焦于**运行时效率验证**，在Boxes（小规模）和MNIST（大规模）两个数据集上对比了SHiP框架与六种基线方法的完整执行时间。如Table 2所示，SHiP展现出独特的"前期投资+快速迭代"模式：Cover tree超度量计算在Boxes上仅需0.059秒，与最快的SCAR（1.481秒）和k-means GT k（0.114秒）相比具有竞争力；在MNIST上，Cover tree的16分24秒与Ward的19分02秒、AMD-DBSCAN的17分21秒处于同一量级，显著优于DPC（在MNIST上失败，标记为"-"）。

关键差异在于**后续扩展成本**：一旦超度量构建完成，SHiP生成Stability层次需33ms、MoE层次仅需3ms、Elbow划分134ms——这意味着用户可以在**单次超度量计算的时间预算内**，探索数十种不同准则与k值的组合。相比之下，Ward或AMD-DBSCAN每更换一个参数都需重新运行完整的分钟级算法。


![Table 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/2f3105f9-7cf3-4a7a-986f-6d39dd2aad6e/figures/Table_3.png)
*Table 3 (comparison): ARI values for the SHP framework on the FC and Glove one-dimensional (top), directional (middle), and co-occurrence (bottom) datasets. We compare against k-means, average linkage, DP-means, E-kmeans, SCAP, and Ward to give the ground truth k value. Full results are in the Appendix.*



Table 3（以及附录E中的Tables 6-9）报告了ARI、NMI、AMI和相关系数等质量指标，用于验证SHiP在FC和Glove数据集上的聚类质量。虽然具体数值未在提供的摘录中完整展示，但文中声称SHiP的聚类质量与专门优化的基线"具有竞争力"（competitive outputs）。

关于公平性检查：
- **基线强度**：比较的基线涵盖了划分式（k-means）、层次式（Ward）、密度式（AMD-DBSCAN, DPC）和谱方法（SCAR），但缺少HDBSCAN、OPTICS、BIRCH等更现代的方法，也未与深度学习聚类方法对比。
- **硬件环境**：所有实验在CPU（2x Intel 6326, 16核, 512GB RAM）上运行，未利用GPU加速；部分基线（如SCAR）可能有GPU优化版本未测试。
- **已知问题**：DPC在MNIST上失败；质量指标的具体数值在提供的摘录中不完整，主要证据来自运行时对比；作者承认质量结果在附录中，正文侧重效率优势。
- **适用边界**：SHiP的优势场景是"需要探索多种聚类解"的交互式分析，若仅需单一k-means解且k已知，传统方法可能更简单。

## 方法谱系与知识库定位

**方法家族**：层次聚类 → 超度量聚类 → 统一提取框架

**直接继承与扩展关系**：
- **Cover tree**（Beygelzimer et al.）：SHiP将其从近邻搜索数据结构 repurposed 为超度量构建工具，添加了超度量闭包操作。
- **DC tree（密度连接树）**：继承密度方法的连接思想，但输出严格的超度量结构而非密度标签。
- **k-means / k-median / k-center**：SHiP将这三种经典准则从"原始空间迭代优化"转化为"超度量树上动态规划提取"，实现了准则与算法的解耦。
- **Ward / 传统层次聚类**：SHiP用超度量闭包替代了贪心合并策略，使层次结构独立于后续准则选择。

**改变的插槽（slots）**：
| 插槽 | 变更 |
|:---|:---|
| data_pipeline | 原始数据→直接聚类 → 原始数据→超度量→层次→划分 |
| inference_strategy | 每次改参数重新运行 → 单次计算+毫秒级探索 |
| architecture | 单一算法绑定单一准则 → 通用结构支持多准则提取 |

**后续可能方向**：
1. **GPU加速**：当前CPU实现已具竞争力，GPU并行化超度量构建可能进一步扩展至百万级数据。
2. **深度学习集成**：将超度量结构作为神经网络的隐空间约束，实现可微分的层次表示学习。
3. **流式/在线扩展**：Cover tree支持动态插入，探索SHiP在流式聚类中的增量更新机制。

**标签**：
- **modality**：tabular / vector data
- **paradigm**：non-parametric clustering, hierarchical clustering
- **scenario**：interactive exploratory data analysis, multi-granularity clustering
- **mechanism**：ultrametric closure, tree-based dynamic programming, criterion decoupling
- **constraint**：CPU-only implementation, exact (non-approximate) ultrametric computation

