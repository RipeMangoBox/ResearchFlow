---
title: Differential Privacy on Fully Dynamic Streams
type: paper
paper_level: A
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 完全动态流上的差分隐私黑盒构造
- Fully Dynamic DP
- Fully Dynamic DP Mechanism (black-box construction)
- Efficient black-box constructions c
acceptance: Spotlight
method: Fully Dynamic DP Mechanism (black-box construction)
modalities:
- structured_data
- tabular
paradigm: unsupervised
---

# Differential Privacy on Fully Dynamic Streams

**Topics**: [[T__Privacy]] | **Method**: [[M__Fully_Dynamic_DP_Mechanism]]

> [!tip] 核心洞察
> Efficient black-box constructions can transform any static differentially private mechanism for linear queries into a fully dynamic one with only polylogarithmic utility degradation.

| 中文题名 | 完全动态流上的差分隐私黑盒构造 |
| 英文题名 | Differential Privacy on Fully Dynamic Streams |
| 会议/期刊 | NeurIPS 2025 (Spotlight) |
| 链接 | [arXiv](https://arxiv.org/abs/) · [Code](https://github.com/) · [Project](https://) |
| 主要任务 | 持续观察下的差分隐私；旋转门流（turnstile streaming）上的线性查询 |
| 主要 baseline | Dwork et al. continual observation [9]；Almost tight error bounds for continual counting [18]；Fully dynamic algorithms for graph databases with edge DP [27]；Private Multiplicative Weights (PMW) |

> [!abstract] 因为「静态差分隐私机制无法处理数据流中的插入与删除操作」，作者在「Dwork et al. continual observation 二叉树机制」基础上改了「用在线区间树（interval tree）替代静态数据结构，实现黑盒转换」，在「T 时间步的完全动态流」上取得「仅 O(polylog(T)) 效用退化」

- 核心性能：动态机制误差界为 O(polylog(T)) · error_static，T 为时间范围
- 隐私保证：通过高级组合定理实现 (ε, δ)-DP 的 T 时间步组合
- 查询效率：每次查询分解为 O(log T) 个子区间调用静态机制

## 背景与动机

差分隐私（Differential Privacy, DP）在静态数据集上的线性查询已有成熟理论：给定一个完整数据集 D，机制 M 可释放带噪声的查询答案 q(D)，误差可控。然而，现实数据常以流形式到达——用户记录不断新增（插入），也可能因删除请求被移除（删除），且需要在每个时间步 t 持续回答查询。例如，某平台需实时统计过去一小时的活跃用户数，但用户可随时注销账号删除其数据。

现有方法如何处理这一问题？Dwork et al. [9] 的 continual observation 框架是奠基性工作，其核心二叉树机制将时间轴分层组织，使范围查询仅需 O(log T) 个节点组合，但仅支持**插入-only**流。Chan et al. [18] 针对持续计数问题给出了几乎紧的误差界，同样假设数据单调增长。对于更一般的**完全动态**（fully dynamic / turnstile）设置——允许插入与删除——Fichtenberger et al. [19] 研究了不同元素计数问题，但未覆盖一般线性查询；Jain et al. [27] 在图数据库上实现了边级 DP 的完全动态算法，但局限于图域。

这些工作的共同短板是：**缺乏对一般线性查询的黑盒扩展**。要么假设数据只增不减（insertion-only），要么针对特定查询类型定制设计，要么需要修改静态机制的内部结构。当数据流同时存在插入与删除时，静态机制无法直接套用，而重新设计动态机制代价高昂。

本文的核心动机正是填补这一空白：提出一种**黑盒归约**，将任意静态 DP 机制无损转换为完全动态版本，仅付出多对数级效用代价。

## 核心创新

核心洞察：时间轴上的动态数据集变化具有**区间可加性**，因为任意时刻 t 的数据集 D_t 可表示为历史操作的累积效果，从而使**区间树（interval tree）的层次化分解**成为维护隐私状态的自然数据结构，进而让静态机制的黑盒复用成为可能。

| 维度 | Baseline（Dwork et al. continual observation） | 本文 |
|:---|:---|:---|
| 数据模型 | 插入-only 流（数据单调增长） | 完全动态流（turnstile，插入+删除） |
| 核心数据结构 | 二叉树（binary tree），节点存储前缀和 | 区间树（interval tree），节点存储区间聚合 |
| 机制设计模式 | 白盒设计，需针对具体查询定制噪声 | 黑盒包装，不修改静态机制内部，任意线性查询可插拔 |
| 效用退化 | O(log T) 对于计数查询 | O(polylog(T)) 对于一般线性查询 |
| 更新操作 | 仅支持增量更新 | 支持插入与删除的双向更新 |

## 整体框架


![Figure 1, Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/ab164404-5d67-4e82-a3e8-9b36337a02bf/figures/Figure_1,_Figure_2.png)
*Figure 1, Figure 2 (architecture): A standard interval tree. An arbel interval tree.*



整体数据流遵循"流输入 → 区间树维护 → 静态机制调用 → 噪声聚合 → 答案输出"的五阶段管线：

1. **Stream input processor（流输入处理器）**：接收每个时间步的操作对 (s_t, x_t)，其中 s_t ∈ {insert, delete} 表示操作类型，x_t 为数据元素。该模块将动态操作转换为对当前数据集 D_t 的更新规则：D_t = D_{t-1} ⊕ {x_t} 或 D_{t-1} ⊖ {x_t}。

2. **Interval tree maintainer（区间树维护器）**：核心新组件，替代静态直方图。将时间轴 [1, T] 层次化分解为 O(log T) 层区间节点，每个节点维护对应时间区间内数据子集的隐私化状态。插入/删除操作仅影响 O(log T) 个节点，实现高效局部更新。

3. **Static DP mechanism (wrapped)（静态 DP 机制包装器）**：对区间树中受影响的子区间，以黑盒方式调用任意静态 DP 机制 M_static。关键特性：无需知晓 M_static 的内部实现，仅需其满足标准 DP 定义。

4. **Noise addition & composition（噪声添加与组合）**：通过高级组合定理（advanced composition）或集中差分隐私（CDP）[2][21]，将 O(log T) 次静态机制调用的隐私损失组合为整体 (ε, δ)-DP 保证。

5. **Answer aggregator（答案聚合器）**：将 O(log T) 个带噪声的子区间回答聚合为最终查询答案，输出时刻 t 的结果。

```
(s_t, x_t) → [Stream processor] → D_t update
                    ↓
        [Interval tree maintainer]
        ┌─────────────────────────┐
        │  I_1: M_static(D_t∩I_1) │
        │  I_2: M_static(D_t∩I_2) │  ... O(log T) nodes
        │  I_k: M_static(D_t∩I_k) │
        └─────────────────────────┘
                    ↓
        [Noise composition] → (ε,δ)-DP over T steps
                    ↓
        [Answer aggregator] →  q̂(D_t)
```

## 核心模块与公式推导

### 模块 1: 动态数据集更新规则（对应框架图输入端）

**直觉**：将 turnstile 流的插入/删除操作形式化为数据集的对称差更新，为后续区间分解奠定基础。

**Baseline 公式**（静态设定）：数据集 D 一次性可用，查询直接执行：
$$q(D), \quad D \text{ fixed}$$
符号：$q$ 为线性查询，$D \subseteq \mathcal{X}$ 为数据域中的元素集合。

**变化点**：静态设定无法处理流式到达的操作序列；需要显式建模时间维度上的数据集演化。

**本文公式**：
$$\text{Step 1: } D_t = D_{t-1} \oplus \{x_t\} \text{ if } s_t = \text{insert}, \quad D_{t-1} \ominus \{x_t\} \text{ if } s_t = \text{delete}$$
加入了操作类型条件以区分插入与删除；⊕ 和 ⊖ 分别表示集合的增删操作。

$$\text{Step 2: } D_t = D_0 \text{triangle} \left(\text{bigcup}_{\tau \leq t, s_\tau = \text{delete}} \{x_\tau\}\right) \cup \left(\text{bigcup}_{\tau \leq t, s_\tau = \text{insert}} \{x_\tau\}\right)$$
重表达为累积形式以保证与初始状态 D_0 的一致性。

**最终**：$D_t$ 为时刻 t 的有效数据集，支持后续区间树查询。

---

### 模块 2: 区间树分解与黑盒调用（对应框架图核心模块）

**直觉**：利用区间树的范围覆盖性质，将任意时刻查询分解为 O(log T) 个不相交子区间之和，每个子区间独立调用静态机制。

**Baseline 公式**（Dwork et al. [9] 二叉树机制）：
$$\mathcal{M}_{\text{binary}}(D_{1:t}) = \sum_{j \in \text{path}(t)} \left(f(D_{I_j}) + \text{Lap}\left(\frac{\Delta f}{\varepsilon}\right)\right)$$
符号：$D_{1:t}$ 为前缀数据集，$I_j$ 为二叉树节点覆盖的时间区间，$f$ 为计数函数，$\Delta f = 1$ 为全局敏感度。

**变化点**：二叉树机制仅支持前缀和（单调累积），无法处理删除导致的区间失效；本文改用**任意区间覆盖**，允许正负贡献抵消。

**本文公式推导**：
$$\text{Step 1: } q(D_t) = \sum_{i \in \mathcal{C}(t)} q(D_t \cap I_i), \quad |\mathcal{C}(t)| = O(\log T)$$
其中 $\mathcal{C}(t)$ 为区间树中覆盖时刻 t 的规范分解（canonical decomposition），$I_i$ 为树节点区间。加入了区间可加性假设以支持线性查询的分解。

$$\text{Step 2: } \hat{q}(D_t) = \sum_{i \in \mathcal{C}(t)} \underbrace{\left[\mathcal{M}_{\text{static}}(D_t \cap I_i) + \text{Lap}\left(\frac{\Delta q}{\varepsilon_i}\right)\right]}_{\text{black-box call with calibrated noise}}$$
每个子区间独立调用静态机制并添加拉普拉斯噪声；通过并行组合（parallel composition）保证子区间间的隐私不累积。

$$\text{Step 3: } \text{Var}(\hat{q}) = \sum_{i \in \mathcal{C}(t)} O\left(\frac{\Delta q^2}{\varepsilon_i^2} \cdot \text{Var}(\mathcal{M}_{\text{static}})\right) = O(\log^2 T) \cdot \text{Var}(\mathcal{M}_{\text{static}})$$
重归一化噪声分配：取 $\varepsilon_i = \varepsilon / O(\sqrt{\log T})$，通过高级组合保证总隐私预算为 $(\varepsilon, \delta)$。

**最终**：
$$\text{error}_{\text{dynamic}} = O(\text{polylog}(T)) \cdot \text{error}_{\text{static}}$$

**对应消融**：本文未提供数值消融实验（理论工作），但定理形式地证明了：若将区间树替换为朴素逐时间步重新计算，误差将恶化为 O(T) · error_static。

---

### 模块 3: T 时间步隐私组合（对应框架图隐私保证端）

**直觉**：流式设置下每个时间步都释放答案，必须通过组合定理控制累积隐私损失。

**Baseline 公式**（单次释放）：
$$(\varepsilon, \delta)\text{-DP for single query release}$$

**变化点**：T 次连续释放的基本组合给出 O(Tε) 的线性隐私损失，不可接受；需要紧的组合界。

**本文公式**：
$$\text{Step 1: } \varepsilon_{\text{total}}^{\text{basic}} = T \cdot \varepsilon \quad \text{(基本组合，过松)}$$

$$\text{Step 2: } \varepsilon_{\text{total}}^{\text{adv}} = \sqrt{2T \ln(1/\delta')} \cdot \varepsilon + T\varepsilon(e^{\varepsilon}-1) \approx O(\sqrt{T}\varepsilon) \quad \text{[21] 高级组合}$$
加入了高级组合定理以改善 T 的依赖关系。

$$\text{Step 3 (更紧): } \rho\text{-zCDP over } T \text{ steps} \Rightarrow (\rho + 2\sqrt{\rho \ln(1/\delta)}, \delta)\text{-DP} \quad \text{[2] 集中 DP}$$
采用集中差分隐私（concentrated DP）进一步收紧，使每步可分配 $\rho = O(\varepsilon^2 / \log T)$。

**最终**：总隐私损失为 $(\varepsilon, \delta)$-DP，每步有效预算仅缩减 polylog(T) 因子，与效用界的 O(polylog(T)) 退化匹配。

## 实验与分析

本文为一项**纯理论工作**，作者明确声明未进行实验验证（"NA for experiment reproduction resources"）。因此，本节分析其理论结果的强度与局限，而非传统实验结果。


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/ab164404-5d67-4e82-a3e8-9b36337a02bf/figures/Figure_3.png)
*Figure 3 (example): Querying a shifted interval tree.*



**理论结果总览**：本文的核心"结果"是定理形式的上界证明——对于任意满足 (ε, δ)-DP 的静态线性查询机制 M_static，所提出的黑盒构造产生动态机制 M_dynamic，满足：(1) 每个时间步 t 的输出满足 (ε, δ)-DP；(2) 对于 T 时间范围，最大误差以高概率满足 error_dynamic ≤ O(log² T / log log T) · error_static（具体 polylog 因子取决于组合定理的选择）。

与最直接的理论基准 [18]（"Almost tight error bounds on differentially private continual counting"）相比：该工作针对**持续计数**（continual counting）这一特定查询给出了几乎紧的 Θ(log T) 误差界，但限于 insertion-only 流。本文将适用域扩展至**一般线性查询**的**完全动态**（turnstile）设置，代价是 polylog(T) 而非最优的 log T 因子。与 [27]（图数据库上的完全动态 DP）相比：该工作针对边级 DP 的图算法，查询类型受限；本文的黑盒构造适用于任意线性查询，但无图结构可利用。

**"消融"分析——构造组件的必要性**：
- 若移除区间树，改用朴素时间轴扫描：每查询需 O(T) 个子调用，误差退化至 O(T) · error_static，隐私组合亦失效。
- 若移除黑盒包装，要求白盒访问 M_static：则失去模块化优势，每个静态机制需单独重新设计动态版本。
- 若采用基本组合定理替代高级/CDP 组合：T 步隐私预算需缩至 ε/T，噪声标度恶化至 O(T)，完全抵消区间树的效率增益。

**公平性检视**：
- **Baseline 强度**：[18] 在 insertion-only 计数问题上是最优基准，但非直接可比；[27] 是"完全动态"设置下最接近的工作，但领域不同（图 vs 一般线性查询）。缺少与近期 turnstile 专用机制（如 [13] 的完全有界范数方法）的 head-to-head 比较。
- **计算/数据预算**：无实验故无实测开销；区间树每次更新 O(log T) 节点，每次查询 O(log T) 次静态调用，理论上是高效的。
- **作者披露的局限**：(1) 仅理论结果，无真实数据集验证；(2) polylog(T) 开销对极高频流仍可能过重；(3) 限于线性查询，非线性查询类未处理。

## 方法谱系与知识库定位

**方法家族**：Fully Dynamic DP Mechanism（完全动态差分隐私机制）—— 黑盒构造型方法家族

**父方法**：Dwork et al. continual observation [9]（二叉树机制）。本文直接扩展其 continual observation 框架，从 insertion-only 推进至 turnstile（完全动态）模型，核心继承的是"时间层次化分解 + 隐私组合"的思想，但将数据结构从二叉树替换为更灵活的区间树以支持双向更新。

**变化槽位**：
- **data_pipeline**：静态数据集 D → 完全动态流 (s_t, x_t)
- **architecture**：静态直方图/二叉树 → 在线区间树（online interval tree）
- **inference_strategy**：单次查询释放 → 每时间步持续释放，高级组合定理保障隐私

**直接 Baselines 及差异**：
- **[18] Almost tight error bounds for continual counting**：最优的 insertion-only 持续计数界；本文扩展至 turnstile 与一般线性查询，代价是 polylog 因子而非最优常数。
- **[27] Fully dynamic algorithms for graph databases with edge DP**：图域的完全动态 DP；本文脱离图结构，提出一般线性查询的黑盒归约。
- **PMW [15]**：静态设置下的乘性权重机制；本文将其（及任意静态机制）包装为动态版本。

**后续方向**：
1. **紧化 polylog 因子**：当前 O(polylog(T)) 是否为最优？插入-only 下已知 Θ(log T) 下界，turnstile 设置的下界尚开放。
2. **非线性查询扩展**：区间树分解依赖线性可加性；计数中位数、分位数等非线性查询的黑盒动态化是开放问题。
3. **实证验证与工程优化**：在真实高频流系统（如差分隐私数据库 [24] PINQ）中实现并测量区间树的常数因子开销。

**标签**：modality=structured_data | paradigm=unsupervised_theoretical | scenario=streaming_continual_observation | mechanism=interval_tree_black_box_reduction | constraint=epsilon_delta_DP_turnstile_model

