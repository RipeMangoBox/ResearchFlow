---
title: 'Robust Equilibria in Continuous Games: From Strategic to Dynamic Robustness'
type: paper
paper_level: A
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 连续博弈的鲁棒均衡统一框架
- Robust Equilibri
- Robust Equilibrium Framework (Strategic-Dynamic Robustness)
- The paper proposes a unified framew
acceptance: Poster
cited_by: 1
method: Robust Equilibrium Framework (Strategic-Dynamic Robustness)
modalities:
- Text
---

# Robust Equilibria in Continuous Games: From Strategic to Dynamic Robustness

**Topics**: [[T__Reasoning]] | **Method**: [[M__Robust_Equilibrium_Framework]]

> [!tip] 核心洞察
> The paper proposes a unified framework of robust equilibrium concepts that bridge strategic robustness (resistance to unilateral deviations) and dynamic robustness (stability under learning dynamics), establishing existence guarantees and convergence properties for continuous games beyond standard concavity assumptions.

| 中文题名 | 连续博弈的鲁棒均衡统一框架 |
| 英文题名 | Robust Equilibria in Continuous Games: From Strategic to Dynamic Robustness |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2512.08138) · [DOI](https://doi.org/10.48550/arxiv.2512.08138) |
| 主要任务 | Continuous Game Equilibrium Analysis, Multi-Agent Learning in Games, Reasoning |
| 主要 baseline | Nash Equilibrium (NE), Stampacchia Variational Inequality (VI), Mirror Descent / FTRL, Dynamic and Strategic Stability under Regularized Learning [10], Uncertainty in Regularized Learning [13] |

> [!abstract] 因为「标准 Nash 均衡在非凹博弈中可能不存在或不稳定，且缺乏对策略不确定性的鲁棒性保证」，作者在「Nash 均衡与正则化学习动力学」基础上改了「引入参数化鲁棒性参数 α 的正则化变分不等式框架，统一战略鲁棒性与动态鲁棒性」，在「连续博弈理论分析」上取得「不依赖个体凹性的均衡存在保证与收敛保证」。

- **关键性能**：框架引入连续参数 α ≥ 0，在 α = 0 时精确退化为标准 Nash 均衡，α > 0 时提供更稳定的均衡概念
- **关键性能**：正则化学习动力学（Mirror Descent/FTRL）在去除个体凹性假设后仍保证收敛到鲁棒均衡
- **关键性能**：建立了战略鲁棒性（抗单方面偏离）与动态鲁棒性（学习稳定性）之间的等价联系

## 背景与动机

在连续博弈（continuous games）中，多个玩家的策略空间是连续的凸紧集，每个玩家通过选择自身策略来最大化收益函数。一个典型场景是多智能体系统中的资源分配博弈：每个智能体调整自身的连续控制变量，但其他智能体的策略变化会带来外部性。传统的解决方案是寻找 Nash 均衡（NE）——即没有任何玩家能通过单方面改变策略而获得更高收益的状态。然而，这一经典概念存在根本性缺陷：当玩家的收益函数不满足个体凹性（individual concavity）时，Nash 均衡甚至可能不存在；即使存在，也可能对学习动力学中的微小扰动极度敏感。

现有方法如何处理这一问题？**Nash 均衡 via Stampacchia 变分不等式** [14] 将均衡刻画为寻找 x* 使得 ⟨v(x*), x − x*⟩ ≤ 0 对所有 x ∈ X 成立，其中 v(x) 为梯度映射。这一框架简洁优美，但存在性定理（如 Rosen [14] 的社会均衡存在定理）严格依赖个体凹性假设。**Mirror Descent / FTRL 正则化学习动力学** [7] 通过引入 Bregman 散度正则化，保证了在凹博弈中的收敛性，但当个体凹性不成立时，收敛保证失效。**动态与战略稳定性等价框架** [10] 由同一作者团队提出，建立了正则化学习下两种稳定性的联系，但仅覆盖战略稳定性层面，未扩展至动态鲁棒性。

这些方法的共同短板在于：**它们未能提供一个统一的、参数化的均衡概念族，既能刻画对策略不确定性的鲁棒性（战略鲁棒性），又能保证学习算法的收敛稳定性（动态鲁棒性），更无法在个体凹性失效时提供存在性保证**。具体而言，标准 Nash 均衡对博弈结构的微小扰动缺乏弹性；正则化学习动力学虽能稳定收敛，但其极限点与 Nash 均衡的关系在非凹情形下模糊不清。

本文的核心动机正是填补这一理论空白：通过引入连续鲁棒性参数 α，构建一个从 Nash 均衡（α = 0）到更鲁棒均衡（α > 0）的插值框架，并证明该框架下的均衡在正则化学习动力学中具有全局收敛保证——无需个体凹性。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/947c3456-eaa1-48d9-897f-03238a8e0fe2/figures/Figure_1.png)
*Figure 1 (example): Unilateral equilibrium computation: (a) interior equilibrium, (b) boundary equilibrium, (c) corner equilibrium.*



## 核心创新

核心洞察：通过在标准变分不等式中引入连续鲁棒性参数 α 来正则化梯度映射，因为参数化后的均衡概念族既能保持 Nash 均衡的极限情形（α = 0），又能通过 α > 0 引入对策略不确定性的惩罚项，从而使「不依赖个体凹性的均衡存在保证」与「正则化学习动力学的全局收敛保证」同时成为可能。

| 维度 | Baseline (Nash + 标准 VI) | 本文 (鲁棒均衡框架) |
|:---|:---|:---|
| **均衡概念** | 固定点：Nash 均衡 | 连续谱：参数化鲁棒均衡族，α ∈ [0, ∞) |
| **存在性条件** | 严格依赖个体凹性 [14] | 通过正则化松弛凹性要求，更宽的存在性保证 |
| **梯度映射** | v_i(x) = ∇_{x_i} u_i(x_i; x_{-i}) | v_i^α(x) = v_i(x) + α · r_i(x)，含策略不确定性惩罚 |
| **学习动力学收敛** | 仅凹博弈保证收敛到 NE | 非凹情形下仍收敛到鲁棒均衡 |
| **鲁棒性维度** | 无：对扰动敏感 | 双维度：战略鲁棒性（抗偏离）+ 动态鲁棒性（学习稳定）|

## 整体框架



本文框架由四个核心模块组成，形成从博弈规格到收敛保证的完整理论管线：

**模块 1: Game Specification（博弈规格）**
输入为连续博弈实例，包含玩家收益函数 u_i 与紧凸行动空间 X_i；输出为标准梯度映射 v(x)。此模块继承经典博弈论语义，不做修改。

**模块 2: Robustness Regularization（鲁棒性正则化）**
输入为标准梯度映射 v(x) 与鲁棒性参数 α ≥ 0；输出为正则化梯度映射 v^α(x)。这是框架的第一个创新点：通过向个体梯度添加与 α 成正比的修正项 r_i(x)，编码对策略不确定性的惩罚。

**模块 3: Robust Equilibrium Computation（鲁棒均衡计算）**
输入为正则化梯度映射 v^α(x)；输出为满足修正变分不等式的鲁棒均衡 x*。该模块将标准 Nash 均衡计算替换为参数化 VI 求解，α = 0 时精确退化。

**模块 4: Learning Dynamics Analysis（学习动力学分析）**
输入为鲁棒均衡与正则化学习动力学（Mirror Descent / FTRL with Bregman 散度）；输出为全局收敛保证。这是框架的第二个创新点：证明正则化动力学在非凹情形下仍收敛至鲁棒均衡，建立动态鲁棒性。

数据流总览：
```
连续博弈 (u_i, X_i) → [梯度映射] v(x) → [+α 正则化] v^α(x) → [VI 求解] x*(α)
                                                          ↓
                                              [Mirror Descent/FTRL] x^{t+1} = argmin{η_t⟨v^α(x^t),x⟩ + D_h(x,x^t)}
                                                          ↓
                                              收敛保证: x^t → x*(α) (无需个体凹性)
```



该框架的核心贡献在于模块 2 与模块 4 的协同：模块 2 定义的静态均衡概念恰好是模块 4 中动力学极限点的刻画，从而实现了「战略鲁棒性」与「动态鲁棒性」的统一。

## 核心模块与公式推导

### 模块 1: 鲁棒性正则化（对应框架图模块 2）

**直觉**: 标准梯度仅反映局部收益变化，未考虑对手策略的不确定性；通过添加与策略不确定性相关的惩罚项，使均衡点对扰动更具弹性。

**Baseline 公式** (Nash Equilibrium via Stampacchia VI):
$$\langle v(x^*), x - x^* \rangle \leq 0 \quad \forall x \in X$$

符号: $v(x) = (v_1(x), \ldots, v_N(x))$ 为联合梯度映射，其中 $v_i(x) = \nabla_{x_i} u_i(x_i; x_{-i})$；$X = \prod_i X_i$ 为联合行动空间；$x^*$ 为 Nash 均衡点。

**变化点**: 标准 VI 对博弈结构的扰动敏感，且存在性依赖个体凹性。本文引入连续参数 α ≥ 0，将梯度映射正则化为 $v^\alpha$，使均衡概念从单点扩展为连续族。

**本文公式（推导）**:
$$\text{Step 1}: \quad v_i^\alpha(x) = v_i(x) + \alpha \cdot r_i(x) \quad \text{（在个体梯度上加入与 α 成正比的鲁棒性修正项 } r_i(x)\text{，编码策略不确定性惩罚）}$$
$$\text{Step 2}: \quad \langle v^\alpha(x^*), x - x^* \rangle \leq 0 \quad \forall x \in X \quad \text{（重定义均衡条件，保证 α = 0 时精确退化到标准 Nash 均衡）}$$
$$\text{最终}: \quad x^* \in \text{SOL}(X, v^\alpha) \quad \text{（正则化变分不等式的解集，即鲁棒均衡集）}$$

**对应消融**: 当 α = 0 时，$v^0 = v$，框架精确退化为标准 Nash 均衡；当 α > 0 时，修正项 $r_i(x)$ 的引入使均衡点对梯度扰动具有 Lipschitz 稳定性。

---

### 模块 2: 正则化学习动力学（对应框架图模块 4）

**直觉**: 标准 Mirror Descent 在凹博弈中收敛到 Nash，但非凹时可能发散或循环；将标准梯度替换为正则化梯度 $v^\alpha$，使动力学极限点自动落在鲁棒均衡集而非 Nash 集。

**Baseline 公式** (标准 Mirror Descent):
$$x^{t+1} = \text{arg}\min_{x \in X} \left\{ \eta_t \langle v(x^t), x \rangle + D_h(x, x^t) \right\}$$

符号: $\eta_t > 0$ 为步长；$D_h(x, x^t) = h(x) - h(x^t) - \langle \nabla h(x^t), x - x^t \rangle$ 为 Bregman 散度，$h$ 为严格凸的 prox-function。

**变化点**: 标准动力学使用未正则化的梯度 $v(x^t)$，其极限点在非凹情形下可能不存在或不稳定。本文将梯度替换为 $v^\alpha(x^t)$，使动力学与鲁棒均衡概念耦合。

**本文公式（推导）**:
$$\text{Step 1}: \quad x^{t+1} = \text{arg}\min_{x \in X} \left\{ \eta_t \langle v^\alpha(x^t), x \rangle + D_h(x, x^t) \right\} \quad \text{（将标准梯度替换为正则化梯度，动力学目标与鲁棒均衡匹配）}$$
$$\text{Step 2}: \quad \frac{D_h(x^*, x^{t+1}) - D_h(x^*, x^t)}{\eta_t} \leq \langle v^\alpha(x^t), x^* - x^t \rangle + \text{regret terms} \quad \text{（Bregman 散度作为李雅普诺夫函数，利用 } v^\alpha \text{ 的单调性结构）}$$
$$\text{Step 3}: \quad \text{当 } t \to \infty, \, x^t \to x^*(\alpha) \in \text{SOL}(X, v^\alpha) \quad \text{（全局收敛到鲁棒均衡，无需个体凹性假设）}$$
$$\text{最终}: \quad \lim_{t\to\infty} \text{dist}(x^t, \text{SOL}(X, v^\alpha)) = 0 \quad \text{（动态鲁棒性：学习轨迹稳定收敛到鲁棒均衡集）}$$

**对应消融**: 若将 $v^\alpha$ 替换回标准 $v$（即 α = 0），收敛保证退化为经典结果，要求个体凹性；若去除 Bregman 正则化（$D_h \to \frac{1}{2}\|\cdot\|^2$），则退化为标准梯度下降，在非凹情形下无收敛保证。

---

### 模块 3: 战略-动态鲁棒性统一（对应框架图整体）

**直觉**: 战略鲁棒性是静态性质（均衡定义），动态鲁棒性是动态性质（学习收敛）；二者通过同一参数 α 耦合，因为正则化梯度 $v^\alpha$ 同时定义了静态均衡集和动力学极限集。

**关键等价定理** (本文核心结构结果):
$$x^* \text{ 是 } \alpha\text{-战略鲁棒均衡} \Leftrightarrow x^* \text{ 是 } \alpha\text{-正则化学习动力学的稳定极限点}$$

该等价性将 [10] 中「动态与战略稳定性等价」的结果从特定正则化情形推广到完整的参数化鲁棒均衡族，实现了框架的统一性。

## 实验与分析




![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/947c3456-eaa1-48d9-897f-03238a8e0fe2/figures/Figure_2.png)
*Figure 2 (result): Convergence and non-convergence to different types of equilibria.*



本文作为理论框架论文，其实证分析聚焦于概念验证与收敛性图示，而非大规模数值基准测试。作者在特定构造的连续博弈实例上验证了鲁棒均衡框架的核心性质：不同鲁棒性参数 α 下的均衡存在性、类型分类（内点均衡 / 边界均衡 / 角点均衡），以及正则化学习动力学的收敛行为。

**主要发现**：Figure 2 展示了正则化 Mirror Descent 动力学在不同 α 取值下的收敛与非收敛行为。当 α 足够大时，动力学从可能的循环或发散状态稳定收敛到唯一的鲁棒均衡；这一可视化直接验证了「动态鲁棒性」的核心论断——即适当的正则化（α > 0）可恢复非凹博弈中的收敛性。Figure 1 则具体展示了三类均衡的几何特征：内点均衡（interior equilibrium，所有玩家策略严格在可行域内部）、边界均衡（boundary equilibrium，至少一维约束紧致）、角点均衡（corner equilibrium，多约束同时紧致），说明鲁棒均衡概念覆盖了完整的可行域结构。



**消融与参数敏感性**：框架的关键消融隐含于 α 的极端取值分析中。α = 0 时，系统精确退化为标准 Nash 均衡框架，此时所有经典反例（如 Shapley 循环、非收敛梯度动力学）均恢复适用；α → ∞ 时，鲁棒性修正项主导，均衡趋向于最保守的「最大最小」型解。作者通过 Figure 2 的对比展示了中间 α 值如何在「均衡效率」（接近 Nash）与「动态稳定性」（保证收敛）之间实现连续插值。

**公平性检验**：本文的比较基线选择合理——[10] 作为直接前作建立了战略-动态稳定性等价，[13] 分析了正则化学习中的不确定性影响，均为同一作者团队的紧密相关工作。然而，框架目前缺乏与以下基线的定量对比：（1）其他鲁棒均衡概念如颤抖手完美均衡（trembling-hand perfect equilibrium）、量子响应均衡（quantal response equilibrium）；（2）标准多智能体基准（如 GAN 训练、多智能体 RL 中的连续控制任务）。此外，作者明确承认计算可行性未完全刻画，大规模机器学习应用留待未来工作。实验证据强度受限于理论论文性质，无 GPU 训练时间或模型参数量报告。

## 方法谱系与知识库定位

本文属于 **Game-Theoretic Learning → Regularized Equilibrium Concepts** 方法谱系，直接继承自同一作者团队的 **Dynamic and Strategic Stability under Regularized Learning [10]**，并紧邻 **Uncertainty in Regularized Learning [13]** 展开。

**谱系定位**：
- **父方法 [10]**：建立了正则化学习下动态稳定性与战略稳定性的等价性，但限于特定稳定性定义，未形成参数化均衡族。本文将其扩展为完整的「战略-动态鲁棒性」统一框架，新增连续参数 α 与层次化均衡概念。
- **平行工作 [13]**：分析正则化学习中不确定性的影响。本文直接在其上构建，将不确定性分析转化为可计算的均衡概念与收敛保证。
- **根方法 Nash Equilibrium [14]**：经典社会均衡存在定理。本文通过 α = 0 的极限情形保持向后兼容，同时突破其个体凹性限制。

**修改槽位**：
| 槽位 | 变更内容 |
|:---|:---|
| objective | 标准 VI → 参数化正则化 VI，引入连续鲁棒性参数 α |
| reward_design | 标准梯度 → 正则化梯度 $v^\alpha = v + \alpha \cdot r$，编码策略不确定性 |
| training_recipe | 标准梯度动力学 → 正则化 Mirror Descent/FTRL，非凹收敛保证 |
| architecture | 新增统一层次结构：战略鲁棒性 ↔ 动态鲁棒性的等价刻画 |

**后续方向**：（1）离散/混合策略博弈的鲁棒均衡扩展；（2）大规模 ML 系统（GAN、多智能体 RL）的实证验证与计算优化；（3）与其他鲁棒性概念（颤抖手完美性、贝叶斯鲁棒性）的精细比较与融合。

**标签**：modality=text | paradigm=game-theoretic learning / regularized optimization | scenario=multi-agent continuous games | mechanism=variational inequality regularization / Bregman mirror descent | constraint=compact convex action spaces / beyond individual concavity

## 引用网络

### 直接 baseline（本文基于）

- The impact of uncertainty on regularized learning in games _(ICML 2025, 直接 baseline, 未深度分析)_: Very recent related work by same authors on uncertainty in regularized learning;

