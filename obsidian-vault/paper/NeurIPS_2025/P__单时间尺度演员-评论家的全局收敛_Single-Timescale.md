---
title: On the Convergence of Single-Timescale Actor-Critic
type: paper
paper_level: A
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 单时间尺度演员-评论家的全局收敛理论
- Single-Timescale
- Single-Timescale Actor-Critic with O(k^-2/3) Step Sizes
- Single-timescale actor-critic achie
acceptance: Poster
cited_by: 2
method: Single-Timescale Actor-Critic with O(k^-2/3) Step Sizes
modalities:
- Text
paradigm: actor-critic
---

# On the Convergence of Single-Timescale Actor-Critic

**Topics**: [[T__Reinforcement_Learning]] | **Method**: [[M__Single-Timescale_Actor-Critic_with_O(k^-2-3)_Step_Sizes]] | **Datasets**: Random MDP

> [!tip] 核心洞察
> Single-timescale actor-critic achieves global convergence to an ε-close globally optimal policy with O(ε^-3) sample complexity using O(k^-2/3) decaying step sizes for both actor and critic, improving upon existing O(ε^-4) complexity for globally optimal policies.

| 中文题名 | 单时间尺度演员-评论家的全局收敛理论 |
| 英文题名 | On the Convergence of Single-Timescale Actor-Critic |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2410.08868) · [DOI](10.48550/arxiv.2410.08868) |
| 主要任务 | Reinforcement Learning — 演员-评论家算法的收敛性分析 |
| 主要 baseline | Two-Timescale Actor-Critic (Konda and Tsitsiklis 1999); Finite-time Single-Timescale Actor-Critic [18]; Double-Loop Actor-Critic [21] |

> [!abstract] 因为「单时间尺度演员-评论家缺乏全局收敛保证且样本复杂度未知」，作者在「Finite-time Single-Timescale Actor-Critic [18]」基础上改了「将演员和评论家步长统一设为 O(k^-2/3) 并引入耦合 Lyapunov 分析」，在「有限状态空间折扣 MDP 理论分析」上取得「全局最优策略的样本复杂度从 O(ε^-4) 改进至 O(ε^-3)」

- **样本复杂度**：达到 ε-全局最优策略所需迭代次数为 O(ε^-3)，相比现有最优结果 O(ε^-4) 提升 ε^-1 倍
- **步长设计**：演员与评论家学习率均为 η_k = β_k = Θ((1+k)^-2/3)，偏离传统优化中的 O(k^-1/2)
- **理论突破**：首次证明单时间尺度演员-评论家可实现全局收敛，无需双时间尺度分离

## 背景与动机

演员-评论家（Actor-Critic）是强化学习中最核心的算法范式之一，却长期面临一个基础性理论困境：实际实现中，演员（策略更新）与评论家（价值估计）通常以相似频率同步更新，但理论分析却严重依赖「双时间尺度」假设——评论家必须远快于演员收敛，或在内循环中几乎完全收敛。这种割裂导致经典理论难以解释和指导真实代码中的 A2C/A3C/PPO 等实现。

现有方法的处理方式各异：Konda and Tsitsiklis (1999) 的 Two-Timescale Actor-Critic [25] 要求步长比 η_k/β_k → 0，评论家以更快时间尺度追踪最优价值函数，但样本效率低下且实际很少采用；Double-Loop Actor-Critic [21] 通过嵌套循环让内层评论家充分收敛后再更新演员，计算开销巨大；Finite-time Single-Timescale Actor-Critic [18] 虽开始分析单时间尺度情形，但仅能达到 O(ε^-4) 样本复杂度且全局收敛保证不足。更根本的是，这些方法都隐含假设评论家处于「近稳态」（near-stationarity），即 w_k ≈ w*(θ_k)，从而将耦合系统解耦为近似独立的优化问题。

这一假设的问题在于：当演员与评论家以同阶步长共同演化时，评论家参数 w_k 始终滞后于当前策略 θ_k 的最优价值函数 w*(θ_k)，传统误差控制技术失效。作者指出，双时间尺度框架「样本效率低下且实际很少使用」（Olshevsky and Gharesifard 2023），而单时间尺度的全局收敛理论「仍然 poorly understood」。本文的核心目标正是填补这一空白：在不假设评论家近收敛的前提下，建立单时间尺度演员-评论家的全局收敛保证。

## 核心创新

核心洞察：演员与评论家的耦合递归可以通过精心设计的同阶步长 O(k^-2/3) 来协同控制，因为该衰减率恰好平衡了「评论家追踪误差积累」与「演员策略梯度进展」之间的张力，从而使全局收敛的 O(ε^-3) 样本复杂度成为可能。

传统非凸优化使用 O(k^-1/2) 步长即可保证收敛，但演员-评论家系统中演员梯度估计依赖于非最优评论家，引入额外耦合误差项；若步长衰减过快（如 O(k^-1)），演员探索不足且评论家无法有效追踪；若过慢（如 O(k^-1/2)），耦合误差的发散速度超过收敛速度。O(k^-2/3) 是使两项望远镜求和同阶匹配的临界点。

| 维度 | Baseline [18] | 本文 |
|:---|:---|:---|
| 时间尺度结构 | 单时间尺度，但分析技术受限 | 单时间尺度，耦合递归联合分析 |
| 步长 schedule | η_k, β_k 未明确协调至最优 | η_k = β_k = Θ((1+k)^-2/3) |
| 评论家假设 | 隐式依赖评论家快速收敛 | 不假设近稳态，w_k 与 θ_k 同步演化 |
| 全局复杂度 | O(ε^-4) | O(ε^-3) |
| 分析工具 | 标准 Lyapunov 分解 | 耦合 Lyapunov 函数 + 梯度支配引理扩展 |

## 整体框架



算法流程遵循标准的演员-评论家交替更新，但关键区别在于步长调度器（Step Size Scheduler）的设计：

1. **初始化**：随机初始化策略参数 θ_0 和评论家参数 w_0
2. **数据采样**：从当前策略 π_{θ_k} 采样转移 (s_k, a_k, r_k, s_{k+1})
3. **评论家更新（Critic Update）**：w_{k+1} = w_k + β_k · TD_error · φ(s_k)，其中 TD_error = r_k + γφ(s_{k+1})^⊤w_k - φ(s_k)^⊤w_k，使用线性函数近似 φ(s)；输出更新后的价值参数 w_{k+1}
4. **演员更新（Actor Update）**：θ_{k+1} = θ_k + η_k · ∇_θ log π_{θ_k}(a_k|s_k) · Q_{w_k}(s_k, a_k)，其中 Q_{w_k}(s,a) = φ(s,a)^⊤w_k 为当前评论家的 Q 估计；输出更新后的策略参数 θ_{k+1}
5. **耦合步长衰减（Coupled Step Size Scheduler）**：η_{k+1} = β_{k+1} = Θ((k+2)^{-2/3})，演员与评论家同步衰减

核心张力在于：演员使用的梯度估计 ∇_θ J(π_{θ_k}, w_k) 依赖于非最优评论家 w_k ≠ w*(θ_k)，而评论家又在追踪不断变化的策略目标 w*(θ_{k+1})。两者形成相互依赖的闭环系统，无法单独分析任一组件的收敛。

```
θ_0, w_0 ──→ Sample (s,a,r,s') ──→ Critic: w_{k+1} = w_k + β_k·TD·φ(s_k)
                                          ↓
                                    Actor: θ_{k+1} = θ_k + η_k·∇log π·Q_{w_k}
                                          ↓
                              Scheduler: η_{k+1} = β_{k+1} = Θ((k+2)^{-2/3})
                                          ↓
                                    (repeat)
```

## 核心模块与公式推导

### 模块 1: 耦合步长调度（对应框架图 Step Size Scheduler）

**直觉**：传统优化中 O(k^-1/2) 步长对耦合系统过快，导致评论家误差失控；O(k^-1) 过慢，演员收敛停滞。-2/3 是使演员进展与评论家追踪误差在求和层面匹配的临界指数。

**Baseline 公式** (Two-Timescale [25]):
$$\eta_k = o(\beta_k), \quad \beta_k \to 0 \text{ with } \eta_k/\beta_k \to 0$$
或标准非凸优化：
$$\eta_k = \Theta(k^{-1/2})$$
符号: η_k = 演员步长, β_k = 评论家步长, k = 迭代计数

**变化点**：双时间尺度通过 η_k ≪ β_k 强制评论家快速收敛，但样本效率低下；单时间尺度前期工作 [18] 未识别最优衰减率。本文发现耦合系统的误差积累要求更慢的衰减。

**本文公式（推导）**:
$$\text{Step 1}: \eta_k = \beta_k = \Theta((1+k)^{-2/3}) \quad \text{统一演员与评论家衰减率，消除时间尺度分离}$$
$$\text{Step 2}: \sum_{k=1}^K \eta_k^2 = O(K^{1/3}), \quad \sum_{k=1}^K \eta_k = O(K^{1/3}) \quad \text{验证望远镜求和的同阶性}$$
$$\text{Step 3}: \sum_{k=1}^K \eta_k \|\nabla J(\theta_k)\|^2 \leq O(K^{1/3}) \quad \text{代入 Lyapunov 递推，保证右端有界}$$
$$\text{最终}: \min_{k \leq K} \|\nabla J(\theta_k)\|^2 = O(K^{-2/3})$$

### 模块 2: 耦合 Lyapunov 分析（对应框架图整体理论分析）

**直觉**：单独追踪演员目标 J(θ_k) 或评论家误差 ‖w_k - w*(θ_k)‖² 都会因另一组件的漂移而发散，必须构造联合势函数。

**Baseline 公式** (标准演员-评论家分析):
$$V_k^{\text{actor}} = J(\theta_k) - J^* \quad \text{或} \quad V_k^{\text{critic}} = \|w_k - w^*(\theta_k)\|^2$$
符号: J(θ) = 策略期望回报, J* = 最优回报, w*(θ) = 策略 θ 的最优价值参数

**变化点**：传统分析先固定 θ 让 w → w*(θ)，再优化 θ；或假设 w_k 已充分接近 w*(θ_k)。本文中两者同步变化，w*(θ_k) 本身随 k 漂移，标准分解失效。

**本文公式（推导）**:
$$\text{Step 1}: V(\theta_k, w_k) = \underbrace{J(\theta_k) - J^*}_{\text{演员差距}} + c \cdot \underbrace{\|w_k - w^*(\theta_k)\|^2}_{\text{评论家误差}} \quad \text{构造联合势函数，权重 } c > 0 \text{ 待调}$$
$$\text{Step 2}: \mathbb{E}[V_{k+1}] \leq (1 - O(\beta_k)) V_k + O(\eta_k^2 + \beta_k^2) + \underbrace{\text{cross terms}}_{\text{演员-评论家耦合项}} \quad \text{展开一步递推，识别交叉项}$$
$$\text{Step 3}: \text{选择 } c \propto \frac{1}{\beta_k} \text{ 使得交叉项被主导项吸收} \quad \text{关键：权重随迭代调整以平衡两项}$$
$$\text{Step 4}: \text{代入 } \eta_k = \beta_k = k^{-2/3} \Rightarrow \mathbb{E}[V_{k+1}] \leq V_k - O(k^{-2/3})\|\nabla J(\theta_k)\|^2 + O(k^{-4/3})$$
$$\text{最终}: \sum_{k=1}^K k^{-2/3}\|\nabla J(\theta_k)\|^2 \leq O(1) \Rightarrow \min_{k\leq K}\|\nabla J(\theta_k)\|^2 = O(K^{-2/3})$$

### 模块 3: 梯度支配引理扩展（对应全局最优保证）

**直觉**：策略梯度目标满足非凸的梯度支配（gradient domination）条件，可将稳定点收敛转化为全局最优收敛；但标准引理要求精确梯度，而本文使用评论家近似梯度。

**Baseline 公式** (标准梯度支配 [1][10]):
$$\|\nabla J(\theta)\|^2 \geq \frac{2\mu}{C_{PL}}(J^* - J(\theta))$$
符号: μ = 强凸性参数, C_PL = max_k ‖d^{π*}/d^{π_{θ_k}}‖_∞ = 分布不匹配系数

**变化点**：标准引理假设 ∇J(θ) 精确可算，但本文实际使用 ∇_θ J(π_θ, w_k) 且 w_k ≠ w*(θ)，需控制近似梯度与真实梯度之间的差距。

**本文公式（推导）**:
$$\text{Step 1}: \|\nabla J(\theta_k, w_k) - \nabla J(\theta_k)\| \leq L_Q \|w_k - w^*(\theta_k)\| \quad \text{Lipschitz 连续性：评论家误差线性传播至梯度}$$
$$\text{Step 2}: \|\nabla J(\theta_k, w_k)\|^2 \geq \frac{1}{2}\|\nabla J(\theta_k)\|^2 - \|w_k - w^*(\theta_k)\|^2 \quad \text{反向三角不等式，分离近似误差}$$
$$\text{Step 3}: \text{代入梯度支配和耦合 Lyapunov 界} \quad \text{将评论家误差界 } O(k^{-2/3}) \text{ 代入}$$
$$\text{最终}: J^* - J(\theta_K) \leq \epsilon \text{ with } K = O(\epsilon^{-3}) \quad \text{全局最优复杂度与稳定点复杂度同阶}$$

## 实验与分析


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/f02c82a1-a651-41a9-a9c9-e3a6b92e52eb/figures/Table_1.png)
*Table 1 (comparison): Related Works: Sample Complexity of Single Time-Scale Actor-Critic*



本文的理论贡献通过相关工作对比表（Table 1）系统呈现。该表汇总了单时间尺度、双时间尺度、双循环等演员-评论家变体的样本复杂度：本文方法在全局最优策略目标下达到 O(ε^-3) 迭代复杂度，相比最直接的先驱工作 [18]（Finite-time Single-Timescale Actor-Critic）的 O(ε^-4) 提升 ε^-1 倍；相比双时间尺度方法 [17]，避免了其样本效率低下且实际罕用的缺陷。这一改进的实质意义在于：首次证明单时间尺度架构——即实际代码中常见的同步更新模式——具备与理论最优同阶的样本效率。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/f02c82a1-a651-41a9-a9c9-e3a6b92e52eb/figures/Figure_1.png)
*Figure 1 (result): Actor-Critic convergence on (3). Random x0, γ = λ = (t+1)^{-0.5}, α_t = β_t = 2.*



Figure 1 展示了随机 MDP 上的数值验证：设置 γ = λ = (t+1)^{-0.5} 为探索参数，α_t = β_t = 2 为步长系数，从随机初始状态 x_0 出发，验证了演员-评论家迭代 (3) 的收敛行为。虽然具体数值未在可用内容中完整提取，但该图旨在直观展示 O(k^-2/3) 步长 schedule 下的实际收敛轨迹。

需要指出的是，本文的实验验证存在明显局限：全部结果限于合成随机 MDP，未在标准基准（Atari、MuJoCo）或深度演员-评论家实现（A2C、A3C、PPO、SAC）上测试。作者明确承认此点作为 limitation。此外，O(ε^-3) 与 O(ε^-4) 的比较是纯渐近的，未报告具体常数因子；步长 schedule 中的问题相关常数（如 C_PL、L-光滑常数）在实际中未知，调参可能困难。公平性方面，双时间尺度方法 [17] 未在相同随机 MDP 实例上对比，且缺乏 wall-clock 时间比较。Figure 1 中的步长设置 γ = λ = (t+1)^{-0.5} 与理论推荐的 (t+1)^{-2/3} 存在差异，其具体含义需结合原文进一步确认。

## 方法谱系与知识库定位

本文属于 **演员-评论家收敛分析** 谱系，直接父方法为 **Finite-time Single-Timescale Actor-Critic [18]**（Xu et al.），后者首次给出单时间尺度的有限时间分析但仅达 O(ε^-4)。本文通过替换训练配方（training_recipe：统一 O(k^-2/3) 步长）和修改信用分配机制（credit_assignment：耦合递归分析而非近稳态假设），将样本复杂度推进至 O(ε^-3)。

**直接 baselines 与差异**：
- **[18] Finite-time Single-Timescale Actor-Critic**：最直接先驱，同单时间尺度结构但步长未优化至 -2/3，分析技术未能解耦全局收敛
- **[25] Two-Timescale Actor-Critic (Konda and Tsitsiklis 1999)**：经典双时间尺度，评论家远快于演员，样本效率低下
- **[21] Double-Loop Actor-Critic**：嵌套循环结构，内层评论家充分收敛，计算开销大
- **[15] Single-Loop with Fast Critic (Borkar 2022)**：另一单时间尺度收敛结果，分析技术不同，无全局最优保证

**后续方向**：(1) 将分析扩展至函数近似（神经网络参数化）而非线性近似；(2) 在标准深度 RL 基准上实证验证 O(k^-2/3) 步长的实际效果；(3) 探索自适应步长选择以消除对问题相关常数的依赖；(4) 研究更一般随机逼近系统中的耦合递归分析技术。

**标签**：modality: 序列决策 / paradigm: actor-critic / scenario: 无限时域折扣 MDP / mechanism: 耦合 Lyapunov 分析 + 梯度支配 / constraint: 有限状态空间、线性函数近似、问题相关常数未知

