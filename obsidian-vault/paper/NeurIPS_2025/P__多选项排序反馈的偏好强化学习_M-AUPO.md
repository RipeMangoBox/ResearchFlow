---
title: 'Preference-based Reinforcement Learning beyond Pairwise Comparisons: Benefits of Multiple Options'
type: paper
paper_level: A
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 多选项排序反馈的偏好强化学习
- M-AUPO
- M-AUPO is the first PbRL algorithm
acceptance: Poster
method: M-AUPO
modalities:
- Text
paradigm: online preference-based reinforcement learning
---

# Preference-based Reinforcement Learning beyond Pairwise Comparisons: Benefits of Multiple Options

**Topics**: [[T__Reinforcement_Learning]] | **Method**: [[M__M-AUPO]] | **Datasets**: synthetic experiments, real-world experiments

> [!tip] 核心洞察
> M-AUPO is the first PbRL algorithm with ranking feedback that achieves provably improved sample efficiency as subset size increases, with a suboptimality gap that scales inversely with subset size and avoids exponential dependence on the unknown parameter's norm.

| 中文题名 | 多选项排序反馈的偏好强化学习 |
| 英文题名 | Preference-based Reinforcement Learning beyond Pairwise Comparisons: Benefits of Multiple Options |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2510.18713) · [Code](待补充) · [Project](待补充) |
| 主要任务 | Preference-Based Reinforcement Learning (PbRL) |
| 主要 baseline | pairwise comparison PbRL algorithms; Zhu et al. [93] offline ranking PbRL; Mukherjee et al. [49] online learning-to-rank |

> [!abstract] 因为「现有 PbRL 方法仅依赖 pairwise 比较，即使扩展到多选项也无法利用更丰富的反馈信息提升样本效率」，作者在「MNL-bandit contextual bandit」基础上改了「以 Plackett-Luce 模型接收多选项排序反馈，并采用平均不确定性最大化进行子集选择」，在「合成实验与真实世界实验」上取得「累积遗憾随子集大小 K 增加而降低，且次优性间隙为 Õ(d√(∑1/|S_t|))，避免了对未知参数范数的指数依赖」。

- **次优性间隙**: Õ(d√(∑_{t=1}^T 1/|S_t|))，随子集大小 |S_t| 增大而改善
- **下界匹配**: Ω(d/√(KT))，上下界仅差 √(log K) 因子
- **样本效率**: 避免指数依赖 ||θ*||，改为多项式依赖

## 背景与动机

在强化学习中获取精确的数值奖励往往困难且昂贵——人类标注者更容易表达"A 比 B 好"这样的偏好。Preference-based RL (PbRL) 正是利用这类反馈来训练策略。然而，现有方法几乎完全依赖 pairwise 比较：每轮只呈现两个选项，询问哪个更优。这在实践中极为浪费：当用户可以对 K=5 或 K=10 个选项进行排序时，为何只比较两个？

现有工作如何处理多选项反馈？**Zhu et al. [93]** 提出了离线排序 PbRL，但理论保证不随反馈长度增加而改善；**Mukherjee et al. [49]** 的在线 learning-to-rank 方法甚至会出现性能随 K 增大而恶化的情况；**unnamed work [76]** 同样未能利用更丰富的排序信息。更根本的是，这些方法的样本复杂度往往指数依赖于未知奖励参数 θ* 的范数 ||θ*||，导致实际部署时效率极低。

核心短板在于：**探索策略未能将"多选项比较的信息论收益"显式纳入设计**。简单地从 K 个选项中独立选择，或沿用 pairwise 的置信区间方法，无法使 Fisher 信息随子集大小 K 缩放。本文提出 M-AUPO，首次在多选项排序反馈下实现随 K 增大而改善的次优性间隙。

## 核心创新

核心洞察：通过**最大化子集内的平均不确定性**进行动作选择，因为 Plackett-Luce 模型下多选项排序的 Fisher 信息矩阵具有显式的子集大小依赖结构，从而使"样本效率随比较选项增多而提升"成为可能。

| 维度 | Baseline (pairwise / naive multi-option) | 本文 M-AUPO |
|:---|:---|:---|
| 反馈模型 | Bradley-Terry pairwise 或独立多选项 | Plackett-Luce 排序模型 |
| 探索策略 | 逐对比较或贪婪/独立选择 | 平均不确定性最大化：max (1/\|S_t\|) Σ_{a∈S_t} Uncertainty(a) |
| 样本复杂度 | 指数依赖 \|\|θ*\|\| 或不随 K 改善 | 多项式依赖，显式逆缩放 1/\|S_t\| |
| 理论保证 | 次优性间隙与 K 无关或恶化 | Õ(d√(Σ 1/\|S_t\|))，随 K↑ 而改善 |

## 整体框架

M-AUPO 的算法流程遵循标准的 online PbRL 循环，但在三个关键环节进行了重新设计：

1. **Context/State Observation**: 接收环境状态 s_t，提取每个候选动作 a 的特征表示 φ(s_t, a)
2. **Average Uncertainty Maximization Subset Selector** (核心创新): 基于当前对奖励参数 θ 的置信集，求解 max_{S_t⊆A, |S_t|≤K} (1/|S_t|) Σ_{a∈S_t} Uncertainty(a)，输出待比较的动作子集 S_t
3. **Plackett-Luce Ranking Feedback**: 执行 S_t 中动作，从人类/反馈源获得排序 σ_t
4. **Parameter Update with Inverse-Scaled Information** (核心创新): 利用子集大小感知的权重更新 θ 的估计和置信集
5. **Policy Execution**: 基于当前估计 θ̂ 贪婪选择最优动作

```
状态 s_t → 特征 φ(s_t,·) → [置信集 C_t(θ)] → 平均不确定性最大化 → 子集 S_t
                                              ↓
                                         执行获取排序 σ_t
                                              ↓
                                    PL 似然 + 子集权重更新 → 新置信集 C_{t+1}(θ) → 贪婪策略
```

## 核心模块与公式推导

### 模块 1: Plackett-Luce 排序模型（对应框架"反馈模型"位置）

**直觉**: 将 pairwise 的 Bradley-Terry 模型自然扩展到多选项排序，使"从 K 个选项中依次选择最优先项"的概率具有可分解结构。

**Baseline 公式** (Bradley-Terry): 对于两个选项 a_i, a_j，P(a_i ≻ a_j) = exp(φ(s,a_i)^⊤θ) / [exp(φ(s,a_i)^⊤θ) + exp(φ(s,a_j)^⊤θ)]

符号: θ ∈ ℝ^d 为未知奖励参数；φ(s,a) ∈ ℝ^d 为状态-动作特征；σ 表示排序。

**变化点**: Bradley-Terry 仅建模 pairwise，无法利用 K>2 时的完整排序信息；且独立 pairwise 比较会浪费 O(K²) 次查询才能覆盖 K 个选项的全部关系。

**本文公式**:
$$\text{Step 1}: P(\sigma | S, \theta) = \prod_{i=1}^{|S|} \frac{\exp(\phi(s, a_{\sigma(i)})^\text{top} \theta)}{\sum_{j=i}^{|S|} \exp(\phi(s, a_{\sigma(j)})^\text{top} \theta)} \quad \text{Plackett-Luce 全排序概率分解}$$
$$\text{Step 2}: \nabla_\theta \log P(\sigma|S,\theta) = \sum_{i=1}^{|S|} \left[\phi(s,a_{\sigma(i)}) - \frac{\sum_{j=i}^{|S|} \phi(s,a_{\sigma(j)}) \exp(\phi(s,a_{\sigma(j)})^\text{top}\theta)}{\sum_{j=i}^{|S|} \exp(\phi(s,a_{\sigma(j)})^\text{top}\theta)}\right] \quad \text{得分函数的梯度结构}$$
$$\text{最终}: \mathcal{I}(\theta; S) = \mathbb{E}_\sigma[\nabla\log P \cdot \nabla\log P^\text{top}] \text{ 显式依赖于 } |S|$$

Fisher 信息矩阵 I(θ; S) 的迹随 |S| 增大而增加，这是后续逆缩放保证的信息论基础。

### 模块 2: 平均不确定性最大化子集选择（对应框架"探索策略"位置）

**直觉**: 不独立选择高不确定性动作，而是选择"整体信息量最大"的子集，使每轮 K 个比较位的信息收益均匀分布。

**Baseline 公式** (标准线性 bandit UCB): a_t = argmax_a [φ(s_t,a)^⊤θ̂_t + β_t ||φ(s_t,a)||_{V_t^{-1}}]，其中 V_t = Σ_{τ<t} φ_τφ_τ^⊤ + λI

符号: V_t 为设计矩阵；β_t 为置信半径；||·||_{V^{-1}} 为 Mahalanobis 范数（不确定性度量）。

**变化点**: 标准 UCB 每轮只选一个动作；naive 扩展到 K 个时要么独立选 top-K（忽略动作间相关性），要么贪婪选最大不确定性（子集内信息重叠）。两者都无法使累积遗憾按 1/K 缩放。

**本文公式**:
$$\text{Step 1}: \text{Uncertainty}(a) := \|\phi(s_t, a)\|_{V_t^{-1}} \quad \text{单个动作的不确定性度量}$$
$$\text{Step 2}: S_t = \text{arg}\max_{S \subseteq \mathcal{A}, |S| \leq K} \frac{1}{|S|} \sum_{a \in S} \|\phi(s_t, a)\|_{V_t^{-1}} \quad \text{平均不确定性最大化目标}$$
$$\text{最终}: \text{等价于在大小约束下最大化子集内平均置信椭球宽度}$$

该目标确保：当 K 增大时，即使单个动作的不确定性相同，分母 |S| 的规范化使目标函数鼓励"分散的高不确定性"而非聚集。

### 模块 3: 子集大小感知的遗憾界推导（对应框架"理论保证"位置）

**直觉**: 将标准椭圆势引理扩展到多动作选择，利用平均不确定性最大化的设计使每轮信息增益按 |S_t| 缩放。

**Baseline 公式** (标准线性 bandit 遗憾): R_T = Σ_{t=1}^T r_t ≤ Õ(d√T)，其中 r_t = max_a φ(s_t,a)^⊤θ* - φ(s_t,a_t)^⊤θ*

**变化点**: 标准 bound 中每轮只观察一个动作的反馈；多选项排序提供 |S_t| 个相关观测，但 PL 模型的相关结构使简单叠加不成立。

**本文公式推导**:
$$\text{Step 1}: r_t = \max_a \phi(s_t,a)^\text{top}\theta^* - \phi(s_t,\hat{a}_t)^\text{top}\theta^* \leq 2\beta_t \max_{a \in S_t} \|\phi(s_t,a)\|_{V_t^{-1}} \quad \text{UCB 分解}$$
$$\text{Step 2}: \text{由平均不确定性最大化}: \max_{a \in S_t} \|\phi(s_t,a)\|_{V_t^{-1}} \leq \frac{1}{|S_t|} \sum_{a \in S_t} \|\phi(s_t,a)\|_{V_t^{-1}} \cdot |S_t| \cdot \frac{1}{|S_t|} \text{ (重归一化)}$$
$$\text{Step 3}: \sum_{t=1}^T \min\left(1, \frac{1}{|S_t|} \sum_{a \in S_t} \|\phi(s_t,a)\|_{V_t^{-1}}^2\right) \leq \tilde{O}\left(\sum_{t=1}^T \frac{1}{|S_t|} \cdot d\right) \quad \text{扩展椭圆势引理}$$
$$\text{最终}: R_T \leq \tilde{O}\left(d\sqrt{\sum_{t=1}^T \frac{1}{|S_t|}}\right) \quad \text{柯西-施瓦茨聚合}$$

关键：当 |S_t| = K 固定时，R_T ≤ Õ(d√(T/K))，显式随 K 增大而改善。

**对应消融**: 将平均不确定性最大化替换为 naive top-K 选择后，累积遗憾不再呈现 1/K 的逆缩放趋势，验证了该组件的必要性。

## 实验与分析

本文在合成实验环境与真实世界实验环境上评估 M-AUPO，核心验证目标为：理论预言的"性能随子集大小 K 增加而改善"是否成立。

**合成实验** (Figure 1): 在受控的线性特征环境中，M-AUPO 的累积遗憾随 K 从 2 增加到 10 而单调下降。Pairwise 比较 (K=2) 作为最弱基线，其遗憾显著高于 K>2 的配置。这一趋势与理论预测 R_T ≤ Õ(d√(T/K)) 的 1/√K 缩放高度一致。

**真实世界实验** (Figure 2): 在更接近实际部署条件的场景中，同样观察到性能随 K 增大而改善的趋势，确认了理论结果对模型假设偏差的稳健性。

**消融分析**: 作者进行了两项关键消融：(1) 将平均不确定性最大化替换为 naive 子集选择（如独立选 top-K 不确定性动作），性能退化且失去逆缩放特性；(2) 将 K 从多选项降回 K=2 的 pairwise 比较，样本效率显著下降。两者共同支持了"探索策略设计"与"多选项反馈"的核心贡献。

**公平性检验**: 实验存在明显局限。首先，**未在 LLM alignment 等最相关的实际任务上验证**——这正是多选项反馈最具应用价值的场景；其次，**缺少与 Zhu et al. [93]、Mukherjee et al. [49] 的直接实现对比**，仅作理论引用；再次，**未与现代 deep RLHF 方法（如 DPO、PPO+reward model）比较**，实验规模局限于简化域。作者明确承认这些局限，指出非线性扩展与大规模实证是未来方向。

## 方法谱系与知识库定位

**方法家族**: MNL-bandit / multinomial logit contextual bandit → PbRL with ranking feedback

**父方法**: MNL-bandit contextual bandit algorithms [3,4,5]（具体包括 Agrawal et al. 的 tractable online learning algorithm、Thompson sampling for MNL-bandit、动态 assortment selection 等）。M-AUPO 将其从"收益最大化"的运筹学场景扩展到"偏好学习"的交互式 RL 场景，并引入主动探索机制。

**改动槽位**:
- **exploration_strategy**: 从独立动作选择 → 平均不确定性最大化的子集选择
- **reward_design**: 从 Bradley-Terry pairwise → Plackett-Luce 排序模型
- **credit_assignment**: 从指数依赖 ||θ*|| → 多项式依赖 + 逆子集缩放

**直接基线差异**:
- **pairwise comparison PbRL [7,16]**: 仅 K=2，信息利用率低，样本复杂度无 K 的改善
- **Zhu et al. [93] offline ranking**: 离线设定，保证不随反馈长度改善
- **Mukherjee et al. [49] online learning-to-rank**: 在线设定但性能随 K 可能恶化

**后续方向**: (1) 非线性奖励模型扩展（超越线性 φ(s,a)^⊤θ 假设）；(2) LLM alignment 等大规模实际任务的实证验证；(3) 闭合上下界之间的 √(log K) 间隙。

**标签**: [modality: text/sequential decision] / [paradigm: online RL with human feedback] / [scenario: interactive preference elicitation] / [mechanism: uncertainty-based exploration + ranking model] / [constraint: linear reward, Plackett-Luce, finite action space]
## 引用网络

### 直接 baseline（本文基于）

- Randomized Exploration for Reinforcement Learning with Multinomial Logistic Function Approximation _(NeurIPS 2024, 方法来源, 未深度分析)_: Multinomial logistic function approximation for RL; directly relevant to multipl

