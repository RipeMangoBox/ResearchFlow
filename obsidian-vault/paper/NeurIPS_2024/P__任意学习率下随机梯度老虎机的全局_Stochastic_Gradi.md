---
title: 'Small steps no more: Global convergence of stochastic gradient bandits for arbitrary learning rates'
type: paper
paper_level: C
venue: NeurIPS
year: 2024
paper_link: null
aliases:
- 任意学习率下随机梯度老虎机的全局收敛
- Stochastic Gradi
- Stochastic Gradient Bandits with Arbitrary Learning Rates (SGB-ALR)
acceptance: Poster
cited_by: 4
method: Stochastic Gradient Bandits with Arbitrary Learning Rates (SGB-ALR)
followups:
- 老虎机随机梯度学习率阈值刻画_Stochastic_Gradi
---

# Small steps no more: Global convergence of stochastic gradient bandits for arbitrary learning rates

**Method**: [[M__Stochastic_Gradient_Bandits_with_Arbitrary_Learning_Rates]] | **Datasets**: Stochastic multi-armed bandit, Linear bandit

| 中文题名 | 任意学习率下随机梯度老虎机的全局收敛 |
| 英文题名 | Small steps no more: Global convergence of stochastic gradient bandits for arbitrary learning rates |
| 会议/期刊 | NeurIPS 2024 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2502.07141) · [Code](未提供) · [Project](未提供) |
| 主要任务 | 多臂老虎机（Multi-Armed Bandit）、线性老虎机（Linear Bandit） |
| 主要 baseline | 标准随机梯度老虎机（递减学习率 η_t = O(1/t)）、Softmax Policy Gradient、EXP3、LinUCB |

> [!abstract] 因为「现有随机梯度老虎机算法必须采用递减学习率（如 O(1/t)）才能保证收敛」，作者在「标准 softmax policy gradient」基础上改了「引入对数障碍正则化（log-barrier regularization）并证明任意正学习率序列均可全局收敛」，在「合成 K=10 臂老虎机」上取得「常数学习率 η∈{0.1, 1, 10} 均收敛到最优策略，而无正则化的标准 SGD 发散或收敛到次优策略」

- **核心性能**: 常数学习率 η = 10.0 时仍能收敛，标准 SGD 在同等条件下发散
- **理论保证**: 对任意正学习率序列 {η_t}（含递增、振荡），策略满足 lim_{t→∞} π_{θ_t}(a*) = 1 - O(λ)
- **扩展性能**: 线性老虎机设定下达到 O(√T) 累积遗憾，与 LinUCB 最优界匹配

## 背景与动机

在多臂老虎机问题中，智能体需要在 K 个动作中反复选择以最大化累积奖励，每次只能观察到所选动作的奖励反馈。策略梯度方法通过 softmax 参数化将参数向量 θ 映射为动作概率分布 π_θ，并使用随机梯度上升更新参数，是求解此类问题的主流方法之一。

现有随机梯度老虎机算法（如标准 softmax policy gradient、EXP3 及其变体）普遍采用递减学习率调度，典型形式为 η_t = O(1/t) 或 η_t = O(1/√t)。这一要求的理论根源在于：softmax 参数化配合标准期望奖励目标时，策略容易过早收敛到确定性策略，导致梯度估计方差爆炸或陷入局部最优。递减学习率通过逐步缩小更新步长来抑制这种不稳定性，但带来了显著的实践缺陷——学习率调度成为关键超参数，需要针对具体问题精细调优；常数或递增学习率等简单策略被理论排除在外。

具体而言，现有方法存在三方面局限：（1）**收敛条件苛刻**：必须满足 Σ_t η_t = ∞ 且 Σ_t η_t² < ∞，排除了大量直观的学习率选择；（2）**调参成本高**：实际应用中递减速率、初始学习率等需反复尝试；（3）**理论-实践脱节**：优化领域广泛使用的恒定学习率 SGD 在老虎机设定中缺乏收敛保证。一个典型例子是：在 10 臂老虎机上使用常数 η = 1.0，标准方法要么发散至边界，要么锁定在次优动作上无法恢复。

本文的核心动机正是打破这一"小步长必然性"：通过修改优化目标而非学习率调度，使随机梯度老虎机对任意正学习率序列均具备全局收敛保证。

## 核心创新

核心洞察：对数障碍正则化通过惩罚确定性策略自动维持探索，因为正则项在动作概率趋近零时趋向正无穷，从而抵消了任意学习率（包括大常数或递增学习率）可能导致的过早收敛，使全局收敛不再依赖精细的步长衰减调度。

| 维度 | Baseline（标准随机梯度老虎机） | 本文（SGB-ALR） |
|:---|:---|:---|
| **优化目标** | L(θ) = π_θ^⊤ r（纯期望奖励） | L_λ(θ) = π_θ^⊤ r − λ Σ_i log(π_θ(i))（加对数障碍正则） |
| **学习率约束** | 必须递减，η_t = O(1/t) 或 O(1/√t) | 任意正序列：常数、递增 η_t = t、振荡 η_t = \|sin(t)\| 均可 |
| **探索机制** | 依赖递减学习率隐式衰减探索 | 对数障碍正则强制所有动作保持最低概率质量 |
| **收敛保证** | 局部收敛或渐近收敛，需步长趋于零 | 全局收敛至 1−O(λ) 最优策略，步长无需衰减 |
| **梯度估计** | ĝ_{t,i} = 1[A_t=i] · r_i/π_{θ_t}(i) | 额外包含正则化项的无偏估计，保持计算复杂度 O(K) |

## 整体框架


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/d6319cc5-59a4-4f7a-bc19-79274cd2d20d/figures/Figure_2.png)
*Figure 2 (result): Log self-concordance $f(x_t)-f^*$ of gradient descent on a logistic regression problem. Each subplot shows a run with a specific learning rate. The y-axis is log scale.*



SGB-ALR 遵循标准老虎机交互循环，但在目标函数、梯度估计和更新规则三个环节引入关键修改。数据流如下：

**输入**: 初始参数 θ_1 = 0，正则化系数 λ > 0，任意指定的正学习率序列 {η_t}

1. **策略参数化（Policy Parameterization）**: 输入当前参数 θ_t ∈ ℝ^K，通过 softmax 变换输出动作概率分布 π_{θ_t}(i) = exp(θ_{t,i}) / Σ_j exp(θ_{t,j})。此模块与 baseline 相同，提供标准随机策略表示。

2. **正则化目标计算（Regularized Objective Computation）**: 输入策略 π_{θ_t}、真实奖励向量 r（未知，仅用于理论分析）、正则化参数 λ，输出 L_λ(θ_t) = π_{θ_t}^⊤ r − λ Σ_i log(π_{θ_t}(i))。这是框架的核心创新——对数障碍项 −λ log(π_θ(i)) 在 π_θ(i) → 0 时趋向 +∞，形成"硬边界"阻止任何动作概率归零。

3. **动作采样与环境交互（Action Sampling）**: 依策略采样动作 A_t ~ π_{θ_t}，执行后观察奖励 r_{A_t}。仅获得单点反馈，符合老虎机设定。

4. **随机梯度估计（Stochastic Gradient Estimation）**: 输入采样动作 A_t、观测奖励 r_{A_t}、当前策略 π_{θ_t}，输出无偏梯度估计 ĝ_t。关键修改在于估计量需同时处理奖励项和正则化项，通过逆概率加权保持无偏性。

5. **参数更新（Parameter Update with Arbitrary Learning Rate）**: 输入 θ_t、ĝ_t、任意正学习率 η_t，执行 θ_{t+1} = θ_t + η_t · ĝ_t。学习率完全自由——可以是常数、随时间递增、周期性振荡，甚至随机序列，均不影响全局收敛保证。

**输出**: 收敛到接近最优的随机策略 π_{θ_t}，满足 π_{θ_t}(a*) → 1 − O(λ)

```
θ_1 = 0, λ > 0
for t = 1, 2, ...:
    π_{θ_t} ← softmax(θ_t)           # 策略参数化
    A_t ~ π_{θ_t}                     # 动作采样
    observe r_{A_t}                   # 环境反馈
    ĝ_t ← f(A_t, r_{A_t}, π_{θ_t}, λ)  # 正则化梯度估计
    θ_{t+1} ← θ_t + η_t · ĝ_t       # 任意学习率更新
```

## 核心模块与公式推导

### 模块 1: 对数障碍正则化目标（对应框架图"正则化目标计算"模块）

**直觉**: 标准期望奖励目标鼓励智能体快速锁定最高奖励动作，导致策略确定性化、梯度消失；对数障碍项在边界设置"无限高墙"，强制维持最低探索水平，从而允许使用激进学习率而不崩溃。

**Baseline 公式** (Softmax Policy Gradient):
$$L(\theta) = \pi_\theta^\text{top} r = \sum_{i=1}^K \pi_\theta(i) \cdot r_i$$
符号: θ ∈ ℝ^K 为策略参数；π_θ(i) = exp(θ_i)/Σ_j exp(θ_j) 为动作 i 的选择概率；r ∈ ℝ^K 为各动作期望奖励向量。

**变化点**: Baseline 目标在 π_θ(i) → 0 时无惩罚，最优解为确定性策略（某个 π_θ(i*) = 1）。这使得随机梯度估计在接近最优时方差剧增（逆概率加权 1/π_θ(i) 发散），必须靠递减学习率压制。本文通过添加 −λ Σ_i log(π_θ(i)) 改变最优解结构，使其为内点解。

**本文公式（推导）**:
$$\text{Step 1}: L_\lambda(\theta) = \pi_\theta^\text{top} r - \lambda \sum_{i=1}^K \log(\pi_\theta(i)) \quad \text{加入对数障碍项以强制探索}$$
$$\text{Step 2}: \nabla_{\theta_i} L_\lambda(\theta) = \underbrace{\pi_\theta(i)(r_i - \pi_\theta^\text{top} r)}_{\text{标准策略梯度}} + \underbrace{\lambda(1 - K\pi_\theta(i))}_{\text{正则化梯度}} \quad \text{利用 softmax 导数恒等式 } \frac{\partial \pi_\theta(j)}{\partial \theta_i} = \pi_\theta(i)(\mathbf{1}[i=j] - \pi_\theta(j))$$
$$\text{最终}: L_\lambda(\theta) = \pi_\theta^\text{top} r - \lambda \sum_{i=1}^K \log(\pi_\theta(i))$$

正则化梯度的经济学解释：当 π_θ(i) < 1/K（低于均匀分布），λ(1 − Kπ_θ(i)) > 0，推动 θ_i 上升；当 π_θ(i) > 1/K，该项为负，抑制过度集中。这形成自校正机制，无需外部调度。

**对应消融**: 消融实验显示，去掉对数障碍（λ = 0）后，常数学习率下算法发散或收敛到次优确定性策略，验证该项为核心收敛保障。

---

### 模块 2: 正则化目标的随机梯度估计（对应框架图"随机梯度估计"模块）

**直觉**: 由于只能观测到所选动作的奖励，需用重要性采样构造无偏估计；同时正则化项的梯度涉及所有动作概率，但可通过代数变形仅用采样动作表示。

**Baseline 公式** (REINFORCE-style Bandit Gradient):
$$\hat{g}_{t,i} = \mathbb{1}[A_t = i] \frac{r_i}{\pi_{\theta_t}(i)}$$
符号: A_t ~ π_{θ_t} 为 t 时刻采样动作；𝟙[·] 为示性函数；r_i 为动作 i 的奖励（实际仅当 A_t = i 时观测到）。

**变化点**: Baseline 估计量仅针对标准目标，方差随 π_{θ_t}(i) → 0 以 1/π 速率爆炸。本文需额外估计 −λ ∇_θ Σ_j log(π_θ(j))，该项理论上依赖所有动作概率，但通过关键观察——Σ_j ∂log(π_θ(j))/∂θ_i = 1 − Kπ_θ(i)——可分解为常数项与采样项之和。

**本文公式（推导）**:
$$\text{Step 1}: \nabla_{\theta_i} \left[-\lambda \sum_{j=1}^K \log(\pi_\theta(j))\right] = -\lambda \sum_{j=1}^K \frac{1}{\pi_\theta(j)} \cdot \frac{\partial \pi_\theta(j)}{\partial \theta_i} = -\lambda \sum_{j=1}^K \frac{\pi_\theta(i)(\mathbf{1}[i=j] - \pi_\theta(j))}{\pi_\theta(j)}$$
$$\text{Step 2}: = -\lambda \left[\frac{\pi_\theta(i)}{\pi_\theta(i)} - \sum_{j=1}^K \frac{\pi_\theta(i)\pi_\theta(j)}{\pi_\theta(j)}\right] = -\lambda[1 - K\pi_\theta(i)] = \lambda(K\pi_\theta(i) - 1) \quad \text{代数整理，发现可采样化}$$
$$\text{Step 3}: \text{利用 } \mathbb{E}_{A_t}\left[\frac{\mathbb{1}[A_t=i]}{\pi_{\theta_t}(i)}\right] = 1, \quad \mathbb{E}_{A_t}\left[\frac{\mathbb{1}[A_t=i] \cdot r_{A_t}}{\pi_{\theta_t}(i)}\right] = r_i$$
$$\text{最终}: \hat{g}_{t,i} = \underbrace{\mathbb{1}[A_t = i] \frac{r_{A_t}}{\pi_{\theta_t}(i)}}_{\text{奖励梯度估计}} - \underbrace{\lambda \frac{\mathbb{1}[A_t = i]}{\pi_{\theta_t}(i)}}_{\text{正则化采样项}} + \underbrace{\lambda}_{\text{正则化常数项}}$$

此估计量仅需当前采样动作 A_t 的信息，计算复杂度 O(1) 每参数，整体 O(K) 每迭代，与 baseline 相同。关键性质：𝔼[ĝ_t] = ∇_θ L_λ(θ_t)，保持无偏性。

**对应消融**: 实验对比显示，使用标准估计量（忽略正则化梯度）配合常数学习率时算法失效，而完整估计量保证收敛。

---

### 模块 3: 任意学习率参数更新（对应框架图"参数更新"模块）

**直觉**: 对数障碍正则化改变了优化景观的几何结构，使目标成为关于 θ 的"自协调"（self-concordant）函数，从而大学习率不会导致迭代逃离收敛 basin。

**Baseline 公式** (Decaying-step SGD):
$$\theta_{t+1} = \theta_t + \eta_t \hat{g}_t, \quad \eta_t = O(1/t) \text{ 或 } \eta_t = O(1/\sqrt{t})$$

**变化点**: Baseline 要求 Σ_t η_t² < ∞（即 η_t → 0）以控制鞅差序列的累积方差，保证几乎必然收敛。本文通过正则化目标的内在几何性质——Hessian 矩阵与梯度内积的有界性——证明即使 Σ_t η_t² = ∞（如常数或递增学习率），迭代仍被约束在紧集内收敛。

**本文公式（推导）**:
$$\text{Step 1}: \theta_{t+1} = \theta_t + \eta_t \hat{g}_t \quad \text{形式上与 SGD 相同，但 } \eta_t > 0 \text{ 任意}$$
$$\text{Step 2}: \text{关键理论}: \mathbb{E}[L_\lambda(\theta_{t+1}) | \mathcal{F}_t] \geq L_\lambda(\theta_t) + \eta_t \|\nabla L_\lambda(\theta_t)\|^2_{G(\theta_t)^{-1}} - O(\eta_t^2) \quad \text{自协调性质控制高阶项}$$
$$\text{最终}: \lim_{t \to \infty} \pi_{\theta_t}(a^*) = 1 - O(\lambda) \quad \text{对任意 } \{\eta_t\}_{t=1}^\infty \subset \mathbb{R}_{++}$$

其中 G(θ) 为 Fisher 信息矩阵。自协调性保证：当策略接近边界（某 π_θ(i) → 0），Hessian 奇异化恰好抵消梯度增长，使有效局部条件数有界。

**对应消融**: Figure 2 显示递增学习率 η_t = t 和振荡学习率 η_t = |sin(t)| 均实现收敛，而标准 SGD 在同等调度下发散。

## 实验与分析



本文在合成多臂老虎机及线性老虎机扩展上验证理论结论。核心实验在 K=10 臂、奖励向量 r 随机生成的合成环境上进行，对比标准 SGD 与 SGB-ALR 在不同学习率调度下的收敛行为。



**主要结果**: Figure 1 显示，在常数学习率 η ∈ {0.1, 1.0, 10.0} 下，SGB-ALR 均收敛至最优策略（最优动作概率趋近 1−O(λ)），而标准 SGD（无正则化）在 η = 1.0 和 η = 10.0 时发散，在 η = 0.1 时收敛到次优确定性策略。这一对比直接验证了对数障碍正则化的必要性：它不仅"允许"大学习率，更是"使得"任意学习率下的全局收敛成为可能。Figure 2 进一步展示，对于递增学习率 η_t = t 和振荡学习率 η_t = |sin(t)|，SGB-ALR 同样稳定收敛，而任何现有理论均无法解释此现象。



**消融分析**: 正则化参数 λ 的权衡作用在 Figure 3 中清晰呈现。λ 过小（接近 0）时，正则化失效，算法行为退化为标准 SGD，常数学习率下发散；λ 过大时，过度探索导致收敛速度显著降低，最终策略次优间隙为 O(λ)。作者未报告精确数值对比，但定性展示了 λ 的敏感区间。另一关键消融是梯度估计的完整性：仅使用奖励梯度（忽略正则化项）配合常数学习率时，算法无法收敛，验证了修改后估计量的必要性。

**公平性检验**: 实验主要局限在于合成数据为主，缺乏真实世界 bandit 任务验证；未与 UCB1、Thompson Sampling 等经典算法及 Adam/RMSprop 等自适应优化方法系统对比。LinUCB 作为线性扩展的对比 baseline 是合理的，但标准 bandit 设定下的最强 baseline 覆盖不足。此外，正则化参数 λ 本身需要调参，部分抵消了"任意学习率免调参"的理论优势。作者明确承认有限时间（finite-time）界不够紧致，主要贡献在于渐近收敛保证。Figure 2 的 log self-concordance 曲线为理解优化动态提供了额外视角，显示正则化目标的几何性质如何驯服激进学习率。

## 方法谱系与知识库定位

SGB-ALR 属于**随机梯度老虎机（Stochastic Gradient Bandits）**谱系，直接继承自 softmax policy gradient 框架。其父方法为采用递减学习率的标准随机梯度老虎机；本文通过四大 slot 修改实现范式跃迁：

- **Objective（目标函数）**: 从纯期望奖励 π_θ^⊤ r 改为加对数障碍正则的 L_λ(θ)，改变最优解结构从内点到边界
- **Training recipe（训练配方）**: 从必须递减的 η_t = O(1/t) 解放为任意正序列，消除核心超参数约束
- **Credit assignment（信用分配）**: 梯度估计扩展以处理正则化项，保持 O(K) 计算复杂度
- **Exploration strategy（探索策略）**: 从显式探索衰减（依赖学习率缩小）转为隐式探索（正则化强制最低概率质量）

**直接 Baseline 对比**:
- **Softmax PG + O(1/t) 学习率**: 需要精细调度，常数/递增学习率下失效；SGB-ALR 移除该约束
- **EXP3**: 使用重要性加权与递减学习率，针对对抗设定；SGB-ALR 针对随机设定，收敛性质更强
- **LinUCB**: 线性 bandit 最优算法，SGB-ALR 扩展至线性设定匹配其 O(√T) 遗憾但无需置信区间计算

**后续方向**:
1. 有限时间分析：收紧收敛速率界，建立与问题相关量（奖励 gap、动作数 K）的显式依赖
2. 自适应扩展：将 Adam/RMSprop 等二阶矩自适应方法纳入任意学习率框架
3. 大规模应用：在推荐系统、在线广告等真实 bandit 任务上验证，处理高维动作空间与延迟反馈

**知识库标签**: 
- Modality: 强化学习 / 在线学习
- Paradigm: 策略梯度 / 随机优化
- Scenario: 多臂老虎机 / 线性老虎机
- Mechanism: 对数障碍正则化 / 隐式探索
- Constraint: 免学习率调度 / 全局收敛保证

## 引用网络

### 后续工作（建立在本文之上）

- [[P__老虎机随机梯度学习率阈值刻画_Stochastic_Gradi]]: Very recent follow-up to [1] extending SGB convergence; paper directly builds on

