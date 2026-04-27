---
title: 'Causal-R: A Causal-Reasoning Geometry Problem Solver for Optimized Solution Exploration'
type: paper
paper_level: A
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 因果图推理驱动的几何问题求解器
- Causal-R
- Causal-R achieves more accurate
acceptance: Poster
method: Causal-R
modalities:
- Text
- Image
paradigm: supervised
---

# Causal-R: A Causal-Reasoning Geometry Problem Solver for Optimized Solution Exploration

**Topics**: [[T__Math_Reasoning]] | **Method**: [[M__Causal-R]] | **Datasets**: Geometry Problem Solving

> [!tip] 核心洞察
> Causal-R achieves more accurate, shorter, and multiple interpretable geometry problem solutions by using causal graph reasoning theory with forward matrix deduction to compress the search space from the beginning, end, and intermediate reasoning paths.

| 中文题名 | 因果图推理驱动的几何问题求解器 |
| 英文题名 | Causal-R: A Causal-Reasoning Geometry Problem Solver for Optimized Solution Exploration |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [DOI](https://doi.org/10.1142/9789811293993_0003) |
| 主要任务 | Geometry Problem Solving (GPS) / 几何问题求解 |
| 主要 baseline | Inter-GPS, E-GPS, GeoDRL, Pi-GPS |

> [!abstract] 因为「几何问题求解中搜索空间巨大且难以获得多个可解释短解」，作者在「Inter-GPS」基础上改了「用因果图推理(CGR)替代直接定理链式推理，并引入前向矩阵演绎(FMD)实现全局可达性传播与多解回溯」，在「GeoQA」上取得「准确率提升+1.4%且超越人类专家，同时生成最短多解」

- **准确率**: Causal-R 在 GeoQA 上达到最优准确率，相比 Inter-GPS 提升 +1.4%，超越人类专家水平
- **解长度**: 平均解长度最短，优于 E-GPS 和 Inter-GPS 等可解释方法
- **多解能力**: 唯一能够生成多个最短可解释解的方法，其他 baseline 仅能生成单解

## 背景与动机

几何问题求解（GPS）是数学推理中的经典难题：给定几何图形的文字描述和图像，系统需要从初始条件出发，通过多步定理应用推导出目标结论。例如，已知三角形两边及夹角，要求证明某线段平行于另一边——中间可能涉及角相等、相似三角形、平行线判定等多个推理步骤，搜索空间随步数指数膨胀。

现有方法主要沿两条路径发展。**Inter-GPS** [19] 作为代表性符号方法，使用形式语言和符号推理，通过前向/后向链式定理应用求解，保证了可解释性，但缺乏对因果结构的显式建模，导致搜索冗余。**E-GPS** 同样基于符号推理，专注于解释性但仍受限于单解生成和局部搜索。**GeoDRL** 和 **Pi-GPS** 等神经方法则采用程序生成或强化学习策略，虽能利用额外训练数据，但推理过程不透明，且性能受限于数据规模与泛化能力。

这些方法的共同瓶颈在于：**没有系统性地压缩搜索空间**。无论是 Inter-GPS 的穷尽式定理链式搜索，还是神经方法的局部程序执行，都未能从问题起点（初始条件）、终点（目标节点）和中间路径三个维度同时约束搜索范围。此外，教育等实际应用场景需要**多个简短可解释的解法**，而现有方法仅能生成单一解，且常包含冗余定理应用。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/bd2102d5-a5b8-43a1-9113-6cf4f2fd802e/figures/Figure_1.png)
*Figure 1 (example): Illustration of solving a typical geometry problem*



本文提出 Causal-R，核心思想是将几何知识预组织为因果图结构，在求解前完成全局因果推理，从而从根本上压缩搜索空间并支持最短多解生成。

## 核心创新

核心洞察：**几何定理之间存在固有的因果依赖关系**，因为「角相等」往往是「三角形相似」的前提条件，而「三角形相似」又是「边成比例」的原因——这种先决条件-因果效应的结构可以被显式编码为因果图，从而使「从全局层面预计算可达性、一次性锁定最短推理路径」成为可能。

| 维度 | Baseline (Inter-GPS/E-GPS) | 本文 (Causal-R) |
|:---|:---|:---|
| 知识表示 | 平面定理列表，无显式结构 | 因果图：原始节点 + 因果边 + 先决条件边 |
| 推理策略 | 前向/后向链式定理应用，局部搜索 | 全局可达性传播，矩阵迭代计算 |
| 搜索空间 | 完整空间遍历，无系统压缩 | 起点-终点-路径三向同步压缩 |
| 解生成 | 单解，可能冗余 | 多解回溯，首次迭代到达即保证最短 |
| 实现机制 | 规则匹配与符号替换 | 矩阵-向量运算（FMD） |

## 整体框架


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/bd2102d5-a5b8-43a1-9113-6cf4f2fd802e/figures/Figure_2.png)
*Figure 2 (architecture): The overall framework of our Causal-R*



Causal-R 的整体流程可分为四个阶段，从静态知识构建到动态问题求解：

**阶段一：因果图构建（Causal Graph Construction）**
输入为预定义的定理规则库 KB，输出为包含原始节点（primitive nodes）、因果边（causal edges）和先决条件边（prerequisite edges）的因果图 $G = (V, E_{causal}, E_{prereq})$。原始节点对应基本几何量（如角度、线段长度），因果边表示"若源节点成立则可推出目标节点"，先决条件边编码定理应用所需的全部前提必须同时满足。

**阶段二：前向矩阵演绎（Forward Matrix Deduction, FMD）**
输入为因果图和具体问题的初始条件，输出为节点可达状态矩阵。FMD 将图结构转化为邻接矩阵 $\mathbf{M}$ 和状态向量 $\mathbf{v}$，通过迭代矩阵乘法 $\mathbf{v}^{(t+1)} = \mathbf{M} \cdot \mathbf{v}^{(t)}$ 高效传播可达性，替代了传统符号方法的逐条规则匹配。

**阶段三：目标节点验证（Target Node Validation）**
输入为迭代后的可达状态和目标节点集合，输出为确认可达的目标及其首次到达迭代数。验证条件要求目标节点 $v_t$ 的可达状态为真且存在从初始节点 $v_0$ 的完整路径——这一全局性检查确保在**第一次迭代到达目标时即停止**，为最短解提供理论保证。

**阶段四：多解回溯（Multi-solution Back-tracing）**
输入为已验证目标节点和因果图结构，输出为多个最短解路径。从目标节点沿激活的因果边反向追踪至初始节点，利用全局信息重构路径；通过调整候选解约束参数 $\lambda$ 可控制生成解的数量与多样性。

```
Rule Base KB → [Causal Graph Construction] → G(V, E_causal, E_prereq)
                                    ↓
Problem (initial conditions, target) → [FMD] → v^(t+1) = M · v^(t)
                                    ↓
                           [Target Validation] → confirmed targets
                                    ↓
                        [Multi-solution Back-tracing] → solutions
```

## 核心模块与公式推导

### 模块 1: 因果边传播规则（对应框架图阶段一→二）

**直觉**: 几何定理的应用有严格的先决条件——不能仅凭"角A=角B"就推出"三角形相似"，还需要"角C=角D"同时成立；这种"全部前提满足才能触发因果推导"的逻辑是 CGR 区别于普通图遍历的关键。

**Baseline 公式** (Inter-GPS 的直接定理应用): 无显式公式，本质为规则匹配 $\text{if } \text{premise}_1 \land \text{premise}_2 \land \cdots \Rightarrow \text{apply theorem } T_i$，但前提与结论的关系隐含在程序逻辑中，未结构化表示。

符号: $R(v_i)$ = 节点 $v_i$ 的可达状态（True/False）；$E_{causal}$ = 因果边集合；$E_{prereq}$ = 先决条件边集合

**变化点**: Inter-GPS 将定理作为独立规则逐一匹配，未显式区分"因果推导关系"与"前提依赖关系"，导致重复验证和冗余搜索。CGR 将两种关系分离为两类边，使传播逻辑形式化。

**本文公式（推导）**:
$$\text{Step 1: 节点状态初始化} \quad R(v_0) = \text{True} \text{ for all initial nodes } v_0 \in V_{init} \quad \text{（将问题给定的初始条件标记为可达）}$$
$$\text{Step 2: 因果传播} \quad R(v_i) = \text{bigvee}_{(v_j, v_i) \in E_{causal}} \left( R(v_j) \wedge \text{bigwedge}_{(v_k, v_j) \in E_{prereq}} R(v_k) \right) \quad \text{（源节点可达且其所有先决条件满足，才能激活目标节点）}$$
$$\text{最终}: R(v_i) \text{ is True iff } \exists \text{ causal path with all prerequisites satisfied}$$

**对应消融**: Table 2 显示将 CGR 替换为基础 Python 字典策略后性能下降，验证了结构化因果表示的必要性。

---

### 模块 2: 前向矩阵演绎 FMD（对应框架图阶段二）

**直觉**: 因果图的迭代传播若用传统符号遍历实现，每次迭代需遍历所有边检查条件；转化为矩阵运算后可利用线性代数的高效实现，同时保持推理的精确性。

**Baseline 公式** (Inter-GPS/E-GPS 的规则引擎): 无闭合形式，为离散状态机 $S_{t+1} = \text{ApplyRules}(S_t, \text{KB})$，时间复杂度与规则库规模成正比。

符号: $\mathbf{M}$ = 编码因果边与先决条件约束的转移矩阵；$\mathbf{v}^{(t)}$ = 第 $t$ 次迭代的节点状态向量（元素为 0/1 或实数值表示可达程度）

**变化点**: 将离散的逻辑运算转化为连续的矩阵-向量乘法，使"并行传播所有可能推导"成为单次矩阵运算，同时通过矩阵幂次自然实现多步推理的迭代展开。

**本文公式（推导）**:
$$\text{Step 1: 图到矩阵转化} \quad M_{ij} = \begin{cases} 1 & \text{if } (v_j, v_i) \in E_{causal} \text{ and prereqs of } v_j \text{ satisfiable} \\ 0 & \text{otherwise} \end{cases} \quad \text{（先决条件作为矩阵元素置零的掩码）}$$
$$\text{Step 2: 迭代传播} \quad \mathbf{v}^{(t+1)} = \mathbf{M} \cdot \mathbf{v}^{(t)} \quad \text{（每次矩阵乘法等价于扩展一层因果影响，布尔运算可推广至半环）}$$
$$\text{Step 3: 收敛检测} \quad \text{Stop when } v_t^{(t)} > 0 \text{ for any target } v_t \quad \text{（首次非零即记录迭代数，保证最短路径）}$$
$$\text{最终}: \mathbf{v}^{*} = \lim_{t \to T} \mathbf{M}^t \cdot \mathbf{v}^{(0)} \text{, where } T = \min\{t: R(v_t)=\text{True}\}$$

**对应消融**: Table 6-7 按演绎路径数量和唯一原始节点数量分组，验证了 FMD 在不同问题规模下的计算效率优势。

---

### 模块 3: 目标验证与多解回溯（对应框架图阶段三→四）

**直觉**: 传统方法到达目标后继续搜索可能发现更短路径，或根本不知是否最短；Causal-R 利用"首次迭代到达"作为全局最优证书，直接从该时刻回溯，避免任何冗余探索。

**Baseline 公式** (Inter-GPS 的单解生成): $\text{solution} = \text{TracePath}(v_0 \text{leadsto} v_t)$，其中路径通过局部启发式或穷尽搜索获得，无最短性保证，且仅返回单一路径。

符号: $\lambda$ = 候选解约束数量（控制生成解的个数）；$\mathcal{P}(v_0, v_t)$ = 从 $v_0$ 到 $v_t$ 的所有路径集合

**变化点**: Baseline 的"搜索-验证-输出"流程被重构为"全局传播-首次停止-多路回溯"，最短性由 FMD 的迭代层次保证而非后验比较。

**本文公式（推导）**:
$$\text{Step 1: 目标确认} \quad \text{TargetAchieved}(v_t) \iff R(v_t) = \text{True} \wedge \exists \text{ path } v_0 \text{leadsto} v_t \quad \text{（双重验证确保逻辑完备性）}$$
$$\text{Step 2: 迭代数记录} \quad t^* = \text{arg}\min_t \{v_t^{(t)} > 0\} \quad \text{（首次到达的迭代层即最短步数）}$$
$$\text{Step 3: 多解回溯} \quad \mathcal{S} = \{s_k = \text{BackTrace}(v_t, t^*, \lambda) \text{mid} k = 1, \ldots, K\} \quad \text{（沿激活边反向追踪，} \lambda \text{ 控制分支选择）}$$
$$\text{最终}: |\mathcal{S}| \geq 1 \text{, each } |s_k| = t^* \text{ (shortest)}$$

**对应消融**: Table 2 显示当 $\lambda=1$（仅需求单解）时准确率和解长度达到最优；增大 $\lambda$ 虽能获得多解，但可能因目标节点依赖关系改变最短组合，这与作者披露的"获取多解可能改变最短组合"的局限性一致。

## 实验与分析


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/bd2102d5-a5b8-43a1-9113-6cf4f2fd802e/figures/Table_1.png)
*Table 1 (comparison): The main performance comparison results*



本文在 **GeoQA** 数据集上评估 Causal-R，该数据集是几何问题求解的核心 benchmark，包含多模态（文本+图像）几何问题。主实验对比了 Causal-R 与 Inter-GPS、GeoDRL、Pi-GPS、E-GPS 等方法。Table 1 显示，Causal-R 在准确率上达到最优，相较此前最强的符号方法 Inter-GPS 提升 **+1.4%**；更值得注意的是，Causal-R 的准确率**超越了人类专家水平**，这在基于符号推理的 GPS 方法中尚属首次报道。解长度方面，Causal-R 同时取得最短平均解长度，显著优于同样强调可解释性的 E-GPS 和 Inter-GPS。


![Table 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/bd2102d5-a5b8-43a1-9113-6cf4f2fd802e/figures/Table_3.png)
*Table 3 (quantitative): The problem-solving performance and solution length on different methods*



Table 3 从五个维度综合评估解题质量：准确性、解长度、可解释性、多解能力和教育适用性。Causal-R 是唯一在「多解能力」维度获得满分的方法——Inter-GPS 和 E-GPS 虽保证可解释性但仅能生成单解，GeoDRL 和 Pi-GPS 等神经方法则因黑箱特性缺乏可解释性。这一独特优势直接源于 CGR 的全局因果结构和多解回溯机制。


![Table 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/bd2102d5-a5b8-43a1-9113-6cf4f2fd802e/figures/Table_2.png)
*Table 2 (comparison): Comparison of solution quality from different methods*



消融实验（Table 2）考察了候选解约束参数 $\lambda$ 的影响：当仅需求单解（$\lambda=1$）时，系统达到**最高准确率和最短解长度**；随着 $\lambda$ 增大以生成更多解，性能出现下降。作者分析这与目标节点间的依赖关系有关——多解需求可能迫使回溯路径选择非全局最优的分支。此外，将 FMD 实现替换为基础 Python 字典策略后性能下降，验证了矩阵化实现的效率与有效性。

公平性检查方面，实验存在几点需注意之处：Table 1 中部分 baseline（GeoDRL、Pi-GPS）使用了额外 GPS 训练数据，与纯符号方法的 Causal-R 并非严格同条件对比；Related Work 中提及的更强近期方法如 **UniGeo**、**GeoDANO**、**G-LLaVA**、**LANs**、**TrustGeoGen** 等未明确出现在主对比表中，可能因这些方法侧重多模态大模型或不同评测协议；人类专家对比的具体方法论未详细披露。作者也坦诚了系统局限性：性能对定理规则库 KB 的完备性和几何条件解析质量敏感；符号赋值顺序和回溯搜索顺序引入随机性；多解生成时的目标依赖问题。

## 方法谱系与知识库定位

Causal-R 属于**符号推理驱动的几何问题求解**方法家族，直接继承自 **Inter-GPS** [19]——后者建立了形式语言+符号推理的可解释 GPS 范式。Causal-R 在 Inter-GPS 基础上进行了结构性重构：将「推理策略」从直接定理链式推理替换为因果图全局传播，「推断策略」从规则引擎替换为矩阵运算，「解生成」从单解扩展为多解回溯，并新增了「搜索空间压缩」维度。

**直接 baselines 对比**:
- **Inter-GPS** [19]: 同家族父方法，Causal-R 保留其可解释性但重构推理架构为因果图优先
- **E-GPS**: 同强调可解释性，Causal-R 额外获得多解生成与三向空间压缩能力
- **GeoDRL** / **Pi-GPS**: 神经/程序生成路线，Causal-R 在无需额外训练数据条件下取得更高准确率

**后续方向**:
1. 将 CGR 理论扩展至代数、数论等其他数学推理领域，验证因果图结构的领域迁移性
2. 结合多模态大模型的视觉解析能力，缓解当前系统对几何条件解析的敏感性
3. 探索神经-符号混合架构，用神经网络学习因果边权重以处理规则库不完备场景

**标签**: 模态(multimodal: text+image) / 范式(symbolic reasoning, causal graph) / 场景(educational geometry problem solving) / 机制(matrix-vector reachability propagation, multi-solution back-tracing) / 约束(interpretability, no extra training data, shortest-path guarantee)

