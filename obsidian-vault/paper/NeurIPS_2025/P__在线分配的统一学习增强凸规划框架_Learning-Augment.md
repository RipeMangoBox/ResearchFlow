---
title: A Learning-Augmented Approach to Online Allocation Problems
type: paper
paper_level: A
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 在线分配的统一学习增强凸规划框架
- Learning-Augment
- Learning-Augmented Online Allocation Framework
- A general learning-augmented algori
acceptance: Poster
method: Learning-Augmented Online Allocation Framework
modalities:
- Text
paradigm: supervised
---

# A Learning-Augmented Approach to Online Allocation Problems

**Topics**: [[T__Reinforcement_Learning]], [[T__Reasoning]] | **Method**: [[M__Learning-Augmented_Online_Allocation_Framework]]

> [!tip] 核心洞察
> A general learning-augmented algorithmic framework using convex programming duality and a single d-dimensional vector of learned weights can produce nearly optimal solutions for a broad class of online allocation problems including routing, scheduling, and fair allocation.

| 中文题名 | 在线分配的统一学习增强凸规划框架 |
| 英文题名 | A Learning-Augmented Approach to Online Allocation Problems |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2502.XXXXX) · [Code](N/A) · [Project](N/A) |
| 主要任务 | 在线分配（online allocation）、在线路由、在线调度、公平分配 |
| 主要 baseline | Problem-specific learning-augmented online algorithms; A general framework for learning-augmented online allocation (ICALP 2023) [18]; Classical online algorithms without predictions |

> [!abstract] 因为「现有学习增强在线算法均为问题专用设计（如缓存、调度、匹配各自独立），缺乏统一框架」，作者在「ICALP 2023 通用框架 [18]」基础上改了「以凸规划对偶性为核心工具，用单一 d 维权向量统一指导多领域在线决策，并建立 MinMax/MaxMin 通用目标形式化」，在「理论竞争分析」上取得「仅需单维权重即可达到近最优解的通用保证」。

- **关键性能 1**：单一 d 维学习权重向量即可覆盖 routing、scheduling、fair allocation 等多领域问题（理论保证）
- **关键性能 2**：首次为 MinMax 与 MaxMin 两类目标提供统一凸规划形式化
- **关键性能 3**：竞争比基准为 $L = \mathbb{E}_{P \sim D}[\text{MinMax}(P)]$，即分布期望下的离线最优值

## 背景与动机

在线分配问题（online allocation）是计算机科学中的经典挑战：算法必须在每个时间步面对新到达的请求时立即做出决策，而无法预知未来。例如，在在线路由中，网络节点需要实时决定将数据流分配到哪条路径，以平衡各链路的负载；在在线调度中，机器需要在不知道后续任务的情况下分配计算资源；在公平分配中，系统需要持续将资源分给多个参与者，确保没有人被过度忽视。这些问题的共同难点在于——决策不可撤销，而目标通常涉及多个智能体（agents）的长期累积成本或收益。

现有方法如何处理这些问题？**问题专用的学习增强算法**（如 [2] 的在线匹配、[11] 的能耗调度、[21] 的非预知调度）为每个具体领域独立设计：缓存有缓存的预测用法，调度有调度的权重调整，匹配有匹配的度预测。这些方法确实在各自领域取得了进展，但彼此之间难以迁移。**经典无预测在线算法**（如 [3] 的虚拟电路路由、[6] 的吞吐量竞争路由）则完全不利用机器学习预测，仅依赖最坏情况竞争分析，性能保守。**ICALP 2023 的通用框架 [18]** 虽然迈出了统一的第一步，但仍未将凸规划对偶性作为核心工具，也未能用极低维度的单一参数向量覆盖如此广泛的问题变体。

这些方法的**根本短板**在于碎片化：每个新问题都需要重新设计算法、重新分析竞争比、重新训练预测模型。研究者缺乏一个"即插即用"的数学框架——能够接收任意在线分配实例，借助一个统一的学习参数，自动产生具有理论保证的在线决策策略。本文正是要解决这一统一性缺口：提出首个以凸规划对偶性为基石、仅需单一 d 维权向量即可驱动多领域在线分配的通用框架。

## 核心创新

核心洞察：凸规划对偶性可以将耦合的多智能体在线分配问题解耦为局部决策，而单一 d 维权重向量恰好足以参数化对偶空间中的拉格朗日乘子，因为凸结构保证了对偶变量的低维有效表示，从而使跨领域的统一在线决策成为可能。

| 维度 | Baseline [18] / 问题专用方法 | 本文 |
|:---|:---|:---|
| **架构设计** | 问题专用算法，每个领域独立构造 | 统一凸规划对偶框架，单一 d 维权向量驱动 |
| **目标形式化** | 各领域目标函数各异，无统一模板 | MinMax 与 MaxMin 两种通用凸规划形式，覆盖成本最小化与收益最大化 |
| **推理策略** | 预测用法因问题而异（如预测度数、预测处理时间） | 在线决策仅通过凸规划对偶应用学习权重 w ∈ ℝ^d，与问题领域无关 |
| **理论工具** | 各论文使用各自技巧（如势函数、原始-对偶局部论证） | 系统性凸规划对偶性 + 竞争分析，工具统一 |

## 整体框架


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/720cc491-074a-476d-aee6-4e9310614a58/figures/Figure_1.png)
*Figure 1: Convex Programming Formulation*



本文框架的数据流可概括为四阶段流水线：

**阶段 1：权重学习（Prediction/Weight Learning）**
输入为从历史分布 D 中采样的问题实例集合；输出为单一 d 维权向量 w ∈ ℝ^d。该向量编码了对各智能体相对重要性的学习预测，是整个框架唯一需要从数据中获取的参数。

**阶段 2：凸规划构建（Convex Program Setup）**
输入为在线到达的问题实例 P 与已学习权重 w；输出为对应的原始-对偶凸规划形式。根据问题类型选择 MinMax 或 MaxMin 目标模板，将权重 w 嵌入对偶约束中。

**阶段 3：在线对偶决策（Online Dual Decision）**
输入为逐时刻到达的请求与当前对偶变量状态；输出为每步的分配决策 x_t。算法仅通过查询对偶问题中由 w 引导的局部条件即可实时决策，无需重新求解完整优化问题。

**阶段 4：目标评估（Objective Evaluation）**
输入为完成的分配序列 {x_t}_{t=1}^T；输出为 MinMax(P) 或 MaxMin(P) 值，用于与离线最优值比较以计算竞争比。

```
历史实例 D ──→ [权重学习] ──→ w ∈ ℝ^d
                                    ↓
在线请求 P_t ──→ [凸规划构建] ←── w
                    ↓
            [在线对偶决策] ──→ 实时分配 x_t
                    ↓
            [目标评估] ──→ MinMax(P) / MaxMin(P)
```



Figure 1 展示了该凸规划形式化的具体数学结构，包括原始问题与对偶问题的完整约束体系。

## 核心模块与公式推导

### 模块 1: MinMax 凸规划目标形式化（对应框架图"凸规划构建"阶段）

**直觉**：在线分配中常需保证"最吃亏的智能体也不会太惨"，MinMax 目标直接最小化各智能体的最大累积成本，天然对应公平性要求。

**Baseline 公式**（经典在线负载均衡，如 [5]）：
$$\min \|C\|_p \quad \text{其中} \quad C_i = \sum_{t=1}^{T} c_{i,t}(x_t)$$
符号：$C_i$ 为智能体 $i$ 的累积成本，$\|\cdot\|_p$ 为 $p$-范数近似最大负载。当 $p \to \infty$ 时逼近 MinMax，但光滑性分析困难。

**变化点**：经典方法用 $p$-范数近似或问题专用势函数处理非光滑的 $\max$ 算子；本文直接以凸规划形式化 MinMax，保持精确性且便于对偶分析。

**本文公式（推导）**：
$$\text{Step 1}: \quad \min_{x_1,\ldots,x_T} \; \max_{i \in [d]} \; \sum_{t=1}^{T} c_{i,t}(x_t) \quad \text{（原始 MinMax 问题，非光滑）}$$
$$\text{Step 2}: \quad \text{引入辅助变量 } z \in \mathbb{R}, \; \text{等价改写为} \quad \min_{z, \{x_t\}} \; z \quad \text{s.t. } \sum_{t=1}^{T} c_{i,t}(x_t) \leq z, \; \forall i \in [d]$$
$$\text{Step 3}: \quad \text{对约束引入拉格朗日乘子 } \lambda_i \geq 0, \; \text{构建拉格朗日函数}$$
$$\mathcal{L}(z, \{x_t\}, \lambda) = z + \sum_{i=1}^{d} \lambda_i \left(\sum_{t=1}^{T} c_{i,t}(x_t) - z\right)$$
$$\text{最终}: \quad \text{MinMax 凸规划形式} \; \Leftrightarrow \; \min_{z,\{x_t\}} \max_{\lambda \geq 0} \; \mathcal{L}(z, \{x_t\}, \lambda)$$

该形式将耦合的 $d$ 个智能体约束解耦为可分离的对偶变量 $\lambda_i$，为在线决策奠定基础。

---

### 模块 2: MaxMin 凸规划目标形式化（对应框架图"凸规划构建"阶段）

**直觉**：与 MinMax 对称，当目标是收益而非成本时，需"让最差的智能体也尽可能好"，即最大化最小累积奖励。

**Baseline 公式**（Nash 社会福利相关，如 [13][15]）：
$$\max \; \left(\prod_{i=1}^{d} R_i\right)^{1/d} \quad \text{或} \quad \max \; \sum_{i=1}^{d} \log R_i$$
符号：$R_i = \sum_{t=1}^{T} r_{i,t}(x_t)$ 为智能体 $i$ 的累积奖励。乘积形式或 log-效用虽鼓励公平，但非直接的 MaxMin。

**变化点**：现有工作多用 Nash 社会福利或广义均值效用作为公平目标；本文直接凸规划化 MaxMin，与 MinMax 形成完美对偶对，统一框架可同时处理两类问题。

**本文公式（推导）**：
$$\text{Step 1}: \quad \max_{x_1,\ldots,x_T} \; \min_{i \in [d]} \; \sum_{t=1}^{T} r_{i,t}(x_t) \quad \text{（原始 MaxMin 问题）}$$
$$\text{Step 2}: \quad \text{引入辅助变量 } z \in \mathbb{R}, \; \text{改写为} \quad \max_{z, \{x_t\}} \; z \quad \text{s.t. } \sum_{t=1}^{T} r_{i,t}(x_t) \geq z, \; \forall i \in [d]$$
$$\text{Step 3}: \quad \text{对约束引入拉格朗日乘子 } \mu_i \geq 0$$
$$\mathcal{L}(z, \{x_t\}, \mu) = z + \sum_{i=1}^{d} \mu_i \left(z - \sum_{t=1}^{T} r_{i,t}(x_t)\right)$$
$$\text{最终}: \quad \text{MaxMin 凸规划形式} \; \Leftrightarrow \; \max_{z,\{x_t\}} \min_{\mu \geq 0} \; \mathcal{L}(z, \{x_t\}, \mu)$$

---

### 模块 3: 学习增强的对偶参数化与在线决策（对应框架图"在线对偶决策"阶段）

**直觉**：对偶空间的拉格朗日乘子 $\lambda_i$（或 $\mu_i$）决定了各智能体的相对"紧张程度"；若这些乘子可从历史数据学习并压缩为低维权向量，则在线决策将极为高效。

**Baseline 公式**（[12] 的原始-对偶学习增强方法）：
$$\text{在线决策依赖问题专用的预测结构，如 } \hat{\lambda}_i = f_i(\text{局部预测})$$
符号：$f_i$ 为各智能体独立的预测函数，无统一参数化。

**变化点**：[12] 虽开创性地将原始-对偶方法与学习增强结合，但仍需问题专用的预测设计；本文的关键突破是将所有对偶信息压缩为单一向量 $w \in \mathbb{R}^d$，通过凸结构保证该压缩不会损失过多信息。

**本文公式（推导）**：
$$\text{Step 1}: \quad \text{从对偶问题提取最优乘子结构 } \lambda^*(P) \in \mathbb{R}^d_+ \text{ 对典型实例 } P \sim D$$
$$\text{Step 2}: \quad \text{学习映射 } w = \mathbb{E}_{P \sim D}[\lambda^*(P)] \text{ 或更一般的低维表示}$$
$$\text{（核心假设：对偶乘子在分布上可压缩为单一有效向量）}$$
$$\text{Step 3}: \quad \text{在线时刻 } t, \text{ 用 } w \text{ 引导对偶变量更新}$$
$$\lambda_i^{(t)} = w_i \cdot g_i(\text{历史累积}, \text{当前请求}) \quad \text{（} g_i \text{ 为问题无关的标准更新规则）}$$
$$\text{最终}: \quad x_t = \text{arg}\min_{x} \sum_{i=1}^{d} \lambda_i^{(t)} \cdot c_{i,t}(x) \quad \text{（MinMax 情形）}$$

**竞争分析基准**：
$$L = \mathbb{E}_{P \sim D}[\text{MinMax}(P)]$$
该期望最优值作为离线基准，本文框架的在线解保证与其接近（具体竞争比依赖问题实例的凸性参数）。

**对应消融**：本文明确声明不包含实验，故无数值消融结果。

## 实验与分析



本文在实验部分明确声明"The paper does not include experiments"，因此不存在传统意义上的数值结果表或基准测试图。这一选择源于本文的纯理论定位：作者专注于建立通用框架的竞争比保证，而非在特定数据集上验证经验性能。

从理论验证角度，本文的核心"结果"体现在以下方面：框架的**普适性覆盖**——作者证明该统一方法可实例化到 routing、scheduling、fair allocation 三个经典领域，而此前这些领域各自拥有独立的研究文献；**参数经济性**——仅需 d 维权向量（与智能体数量同维）即可驱动决策，相比问题专用方法中可能需要的复杂预测结构（如每条边的流量预测、每台机器的任务长度分布预测），维度显著降低；**目标统一性**——MinMax 与 MaxMin 两种凸规划形式首次在同一框架下处理，而 [18] 的 ICALP 2023 框架未明确覆盖此对偶结构。



由于缺乏实验，无法进行标准消融分析。但作者通过**理论构造**隐含展示了关键组件的必要性：若移除学习权重 w（退化为经典无预测在线算法），竞争比将恶化至最坏情况保证；若移除凸规划对偶结构（退化为原始空间直接优化），在线解耦将不再可能，每步需求解完整优化问题，丧失计算效率。

**公平性审查**：本文的比较基线 [18] 虽被引为最直接相关工作，但缺乏**数值层面的 head-to-head 对比**——既无竞争比常数的精确比较，也无实例化到同一问题时的性能对照。此外，缺失的实证环节包括：真实网络拓扑上的路由实验、真实作业 trace 上的调度实验、预测误差敏感度分析（权重 w 的获取过程未指定训练机制）、以及维度 d 的缩放行为验证。这些 gap 使得"单维权向量足够"的核心 claim 目前仅停留在理论层面。

## 方法谱系与知识库定位

本文属于**学习增强在线算法（learning-augmented online algorithms）**方法谱系，直接父方法为 **ICALP 2023 的 "A general framework for learning-augmented online allocation" [18]**。与 [18] 相比，本文在五个关键 slot 上发生变更：**架构**上，从问题专用设计替换为统一凸规划对偶；**目标**上，从分散的领域目标替换为 MinMax/MaxMin 通用形式化；**推理策略**上，从多变的预测用法替换为单维权向量驱动；**训练范式**上，继承监督学习获取权重，但未指定具体训练机制；**数据策展**上，假设可从分布 D 采样历史实例，未讨论实际数据构建。

**直接基线差异**：
- **[18] ICALP 2023 通用框架**：标题与 scope 最接近，但本文引入凸规划对偶性作为核心数学工具，且将预测结构压缩至单一 d 维向量
- **[12] Primal-dual method for learning augmented algorithms**：核心技术来源，本文将其从特定问题扩展至统一框架
- **问题专用方法 [2][11][17][21][23]**：各在匹配、调度、缓存等单一领域领先，本文牺牲部分领域精细度换取跨领域统一性

**后续方向**：(1) 将框架实例化到具体领域并补充实验验证，填补当前纯理论的 gap；(2) 设计权向量 w 的具体学习算法（如在线学习或元学习），解决"预测如何获得"的开放问题；(3) 扩展至非凸或组合结构更强的在线问题，检验凸规划假设的边界。

**标签**：modality=理论算法/无特定模态 | paradigm=学习增强在线算法 | scenario=在线分配、路由、调度、公平分配 | mechanism=凸规划对偶性、拉格朗日松弛、低维权向量参数化 | constraint=竞争分析保证、无实验验证、假设分布可采样

