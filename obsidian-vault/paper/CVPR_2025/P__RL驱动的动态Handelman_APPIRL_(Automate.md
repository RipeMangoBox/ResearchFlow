---
title: Automated Proof of Polynomial Inequalities via Reinforcement Learning
type: paper
paper_level: C
venue: CVPR
year: 2025
paper_link: null
aliases:
- RL驱动的动态Handelman多项式证明
- APPIRL (Automate
- APPIRL (Automated Proof of Polynomial Inequalities via Reinforcement Learning)
acceptance: poster
method: APPIRL (Automated Proof of Polynomial Inequalities via Reinforcement Learning)
---

# Automated Proof of Polynomial Inequalities via Reinforcement Learning

**Topics**: [[T__Math_Reasoning]], [[T__Reinforcement_Learning]] | **Method**: [[M__APPIRL]] | **Datasets**: Polynomial Inequality Proofs, Maximum Stable Set Problem

| 中文题名 | RL驱动的动态Handelman多项式证明 |
| 英文题名 | Automated Proof of Polynomial Inequalities via Reinforcement Learning |
| 会议/期刊 | CVPR 2025 (poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2503.06592) · [Code](https://github.com/appirl) · [Project] |
| 主要任务 | 多项式不等式自动证明、最大稳定集问题 |
| 主要 baseline | Random Search, LDPP, S2V-DQN |

> [!abstract]
> 因为「静态Handelman松弛需要预设固定的次数上界D导致证明步骤冗余」，作者在「Random Search」基础上改了「用DQN动态选择乘子扩展证明基，并基于gamma改进设计奖励塑形」，在「PHCpack数据库10个多项式不等式基准C1-C10」上取得「平均完成步数75.4 vs Random Search平均445.1，加速5.9倍」

- **关键性能1**: 在C8实例上，APPIRL仅需75步完成证明，相比Random Search最优31步的117倍加速（相比平均445.1步为5.9倍）
- **关键性能2**: 网络架构搜索显示64-160神经元、4层MLP在不同规模问题上效果最优（Table 1）
- **关键性能3**: 在最大稳定集问题上与S2V-DQN和LDPP进行了对比验证（Table 3, Table 2）

## 背景与动机

多项式不等式证明是计算机代数和优化领域的核心问题：给定一个多项式 $f(\mathbf{x})$，如何证明它在定义域 $\mathcal{S} = [a_i, b_i]^n$ 上恒非负？这一问题直接出现在形式化验证、机器人运动规划、控制理论等场景中。例如，验证一个机械臂的轨迹是否始终满足关节角度约束，就等价于证明某个多项式不等式。

现有方法主要有三类。**Sums-of-Squares (SOS) 方法**将非负性转化为半定规划(SDP)的可行性问题，通过寻找多项式的平方和分解来完成证明，但SDP求解器在大规模问题上计算代价高昂。**静态Handelman松弛**则将问题转化为线性规划(LP)：固定次数上界 $D$，寻找非负系数使得 $f(\mathbf{x}) - \gamma$ 能表示为Handelman基 $\mathbf{x}^\alpha(1-\mathbf{x})^\beta$ 的线性组合。然而，$D$ 的选择成为瓶颈——太小则无法证明，太大则基的数量指数爆炸。**LDPP (Learning Dynamic Polynomial Proofs)** 尝试学习动态证明策略，但仅针对特定图论问题，未与核心多项式证明任务直接对比。

这些方法的共同短板在于：**证明基的构造是静态或启发式的**。无论是固定 $D$ 的Handelman方法，还是随机搜索乘子组合，都无法根据当前证明进度自适应地选择"最有价值"的扩展方向。这导致大量计算浪费在无效基的尝试上。本文的核心动机正是：能否将证明基的扩展建模为序贯决策问题，让强化学习智能体学会"下一步该乘哪个变量"？


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/32395a4d-69b2-49e5-a154-50205e8d9c69/figures/Figure_1.png)
*Figure 1: Figure 1: The framework of automated proof of polynomial inequalities based on RLAs illustrated in Fig.1, our method consists of two main components: Environment Construction and DQN Training. Inthe E*



本文提出APPIRL，首次将DQN与动态Handelman基扩展结合，通过奖励塑形引导智能体高效构造证明。

## 核心创新

核心洞察：**多项式证明基的扩展可以重新框架化为马尔可夫决策过程**，因为每一步选择乘子 $a_t$ 后，新的LP下界 $\gamma_t$ 仅依赖于当前记忆集合 $\mathcal{M}_t$ 和所选动作，从而使基于TD学习的信用分配和策略优化成为可能。

与 baseline 的差异：

| 维度 | Baseline (Random Search / 静态Handelman) | 本文 (APPIRL) |
|:---|:---|:---|
| 证明基构造 | 固定次数上界 $D$，或每步随机选乘子 | 动态记忆 $\mathcal{M}_t$，DQN选择性地扩展 |
| 信用分配 | 无（Random Search无学习） | TD学习 + gamma改进奖励塑形 |
| 动作空间 | 静态或均匀采样 | 动态增长 $\mathcal{A}_t = \mathcal{A}_{t-1} \cup \{a_{t-1}x_i, a_{t-1}(1-x_i)\}$ |
| 计算后端 | 标准多项式乘法 | FFT编码加速的整数卷积乘法 |
| 状态表示 | 无 / 图神经网络(S2V-DQN) | 动态证明状态 $(\mathcal{M}_t, \gamma_t)$ 的MLP编码 |

## 整体框架



APPIRL的完整流程如图1所示，包含8个核心模块，形成"编码-决策-扩展-验证-学习"的闭环：

1. **Problem Encoder（问题编码器）**：输入多项式 $f(\mathbf{x})$ 和定义域 $\mathcal{S} = [a_i, b_i]^n$，输出初始LP形式和初始记忆集合 $\mathcal{M}_0 = \{\mathbf{x}^\alpha(1-\mathbf{x})^\beta \text{mid} |\alpha|+|\beta| \leq k\}$，其中 $k$ 根据 $\deg(f(\mathbf{x}))$ 设置。

2. **State Extractor（状态提取器）**：将当前记忆 $\mathcal{M}_t$、当前下界 $\gamma_t$ 和问题特征编码为状态向量 $\mathbf{s}_t$，供Q网络使用。

3. **Q-Network（Q网络）**：4层MLP（64-160神经元，ReLU激活），输入 $\mathbf{s}_t$，输出动作空间 $\mathcal{A}_t$ 中各候选乘子的Q值估计。

4. **Action Selector（动作选择器）**：采用 $\epsilon$-greedy策略，以 $1-\epsilon$ 概率选择Q值最大动作，以 $\epsilon$ 概率随机探索。

5. **Proof Expander（证明扩展器）**：执行选中动作 $a_t$，通过FFT快速多项式乘法计算新乘子，更新记忆 $\mathcal{M}_{t+1} = \mathcal{M}_t \cup \{a_t\}$ 和动作空间 $\mathcal{A}_{t+1}$。

6. **LP Solver（LP求解器）**：在新的记忆基上求解 $\max \gamma$ s.t. $f(\mathbf{x}) - \gamma = \sum_{i=1}^{|\mathcal{M}_{t+1}|} \lambda_i m_i, \lambda \geq 0$，得到新下界 $\gamma_{t+1}$。

7. **Reward Computer（奖励计算器）**：根据 $\gamma$ 的变化计算即时奖励 $r_t$，驱动学习信号。

8. **Experience Buffer & Trainer（经验回放缓冲与训练器）**：存储转移 $(\mathbf{s}_i, a_i, r_i, \mathbf{s}_{i+1})$，通过DQN损失更新网络参数。

迭代流程可概括为：
```
初始化 (M0, A0, gamma0) → 状态编码 s_t → Q值估计 → ε-greedy选动作 a_t 
→ FFT乘法扩展 → 记忆更新 M_{t+1} → LP求解得 gamma_{t+1} → 奖励计算 r_t 
→ 经验存储 → 网络训练 → 循环直至证明完成(gamma≥0且可行)或达步数上限
```

## 核心模块与公式推导

### 模块 1: 动态Handelman松弛（对应框架图 核心创新/Problem Encoder → Proof Expander）

**直觉**: 固定次数上界 $D$ 的静态基会导致大量冗余单项式；而证明往往只需要一个"精瘦"的基子集即可成立，因此应该让智能体自己决定"加哪些乘子"。

**Baseline 公式** (静态Handelman松弛):
$$\max \gamma \quad \text{s.t.} \quad f(\mathbf{x}) - \gamma = \sum_{|\alpha|+|\beta|\leq D} \lambda_{\alpha,\beta} \mathbf{x}^\alpha(1-\mathbf{x})^\beta, \quad \lambda_{\alpha,\beta} \geq 0$$
符号: $D$ = 固定的总次数上界; $\lambda_{\alpha,\beta}$ = Handelman基的非负组合系数; $\gamma$ = 可证明的下界。

**变化点**: 静态方法中 $D$ 必须预先指定，且基的维度随 $D$ 指数增长 $(O(n^D))$。本文将固定基替换为**动态增长的记忆集合**，将优化重新框架化为序贯决策。

**本文公式（推导）**:
$$\text{Step 1}: \quad \mathcal{M}_0 = \{\mathbf{x}^\alpha(1-\mathbf{x})^\beta \text{mid} |\alpha|+|\beta| \leq k\} \quad \text{（用低次单项式初始化，避免冷启动）}$$
$$\text{Step 2}: \quad \mathcal{M}_{t} = \mathcal{M}_{t-1} \cup \{a_{t-1}\} \quad \text{（每步仅加入RL选中的单个乘子，控制组合爆炸）}$$
$$\text{Step 3}: \quad \max \gamma \quad \text{s.t.} \quad f(\mathbf{x}) - \gamma = \sum_{i=1}^{|\mathcal{M}_t|} \lambda_i m_i, \quad \lambda \geq 0 \quad \text{（在精简基上求解LP，保证效率）}$$
$$\text{最终}: \quad \gamma_t^* = \text{LP-Solve}(f, \mathcal{M}_t) \quad \text{（动态下界序列，单调递增收敛至目标）}$$

**对应消融**: Table 1显示不同网络架构对步数的影响，但未直接消融"动态vs静态基"。

---

### 模块 2: Gamma跟踪奖励塑形（对应框架图 Reward Computer → Experience Buffer）

**直觉**: 证明过程的"进度"天然由LP下界 $\gamma$ 的改进衡量，但不同问题的 $\gamma$ 绝对尺度差异大，需要归一化；同时需要惩罚停滞以避免无效探索。

**Baseline 公式** (Random Search): 无信用分配，每步独立随机采样，无 $r_t$ 概念。

**变化点**: Random Search没有学习信号，无法区分"好动作"和"坏动作"。本文设计**基于相对改进的稠密奖励**，使TD学习能有效传播信用。

**本文公式（推导）**:
$$\text{Step 1}: \quad \tilde{r}_t = \gamma_t - \gamma_{t-1} \quad \text{（原始改进量，但跨问题不可比）}$$
$$\text{Step 2}: \quad r_t = \frac{\gamma_t - \gamma_{t-1}}{|\gamma_0|} \quad \text{if } \gamma_t \neq \gamma_{t-1} \quad \text{（用初始下界归一化，保证跨问题稳定性）}$$
$$\text{Step 3}: \quad r_t = \epsilon \quad \text{if } \gamma_t = \gamma_{t-1} \quad \text{（停滞时给予小惩罚，激励突破平台期）}$$
$$\text{最终}: \quad r_t = \begin{cases} \frac{\gamma_t - \gamma_{t-1}}{|\gamma_0|} & \text{if } \gamma_t \neq \gamma_{t-1} \\ \epsilon & \text{if } \gamma_t = \gamma_{t-1} \end{cases}$$

符号: $\gamma_t$ = 第 $t$ 步LP求解得到的最优下界; $\gamma_0$ = 初始下界（通常为负，取绝对值归一化）; $\epsilon$ = 小的停滞惩罚常数。

**对应消融**: 未在提取文本中找到奖励设计的直接消融实验。

---

### 模块 3: FFT快速多项式乘法与动作空间扩展（对应框架图 Proof Expander）

**直觉**: 动态基扩展需要频繁计算新乘子 $a_t \cdot x_i$ 和 $a_t \cdot (1-x_i)$，多元多项式乘法的朴素实现是瓶颈。

**Baseline 公式**: 标准多元多项式乘法，单项式 $m_1 = \mathbf{x}^{\alpha}$ 与 $m_2 = \mathbf{x}^{\beta}$ 相乘为 $\mathbf{x}^{\alpha+\beta}$，需要对每个变元分别处理指数向量。

**变化点**: 将多元单项式编码为**单一整数**，利用FFT将多项式乘法转化为整数卷积，从 $O(d^n)$ 降至近线性复杂度。

**本文公式（推导）**:
$$\text{Step 1}: \quad \nu_i = \sum_{k=1}^{n} \alpha_{i_k} D^{k-1} \quad \text{（D进制编码：n维指数向量 → 单个整数，保留唯一可逆性）}$$
$$\text{Step 2}: \quad \text{FFT-Mult}(\nu_i, \nu_j) = \text{IFFT}(\text{FFT}(\nu_i) \odot \text{FFT}(\nu_j)) \quad \text{（整数卷积替代逐项乘法）}$$
$$\text{Step 3}: \quad \mu_{i_1} = d_i \text{bmod} D, \; \rho_1 = (d_i - \mu_{i_1})/D, \; \ldots, \; \mu_{i_n} = \rho_{n-1} \text{bmod} D \quad \text{（解码还原指数向量）}$$
$$\text{最终}: \quad \mathcal{A}_t = \mathcal{A}_{t-1} \cup \{a_{t-1}x_i, a_{t-1}(1-x_i) \text{mid} i=1,\ldots,n\} \quad \text{（结构化动作空间增长）}$$

符号: $\nu_i$ = 单项式 $m_i$ 的整数编码; $D$ = 编码基数（大于任何单变元次数上界）; $\odot$ = 逐元素乘法; $d_i$ = FFT乘法后的结果整数。

**对应消融**: 未在提取文本中找到FFT加速的显式消融对比。

## 实验与分析


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/32395a4d-69b2-49e5-a154-50205e8d9c69/figures/Table_1.png)
*Table 1: Table 1 records the best results obtained from exploring different network architectures. For the Random Search method,we randomly select a strategy for proving at each step, and perform 50 trials for*



本文在两类任务上验证APPIRL的有效性。主实验针对**多项式不等式证明**，使用PHCpack数据库中的10个基准实例C1-C10；次要实验在**最大稳定集问题**上与S2V-DQN和LDPP进行对比。

核心结果来自Table 1：在10个多项式不等式证明实例上，APPIRL平均仅需**75.4步**完成证明，而Random Search平均需要**445.1步**，实现了**5.9倍平均加速**。更值得注意的是，即使在Random Search的50次重复试验中的最优结果（$S_{\min} = 31$步），APPIRL在C8实例上仅需**75步**，相比Random Search最优的**1317步**实现了**17.6倍加速**，相比其平均**8744步**更是**117倍加速**。这表明学习到的策略不仅能超越随机搜索的平均水平，甚至能显著优于其运气最好的情况——证明RL确实捕捉到了结构化的证明规律，而非单纯减少方差。


![Table 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/32395a4d-69b2-49e5-a154-50205e8d9c69/figures/Table_3.png)
*Table 3: Table 3: Performance Evaluation compared to S2V-DQN on the maximum stable set problem.*



Table 1同时记录了网络架构搜索的结果：对于不同维度的问题，最优神经元数从64到160不等（4层MLP），说明问题规模与网络容量的适配是必要的，但作者未报告具体每个实例对应的最优配置细节。消融方面，**去掉DQN学习（即退化为Random Search）的代价最为巨大**：步数从75.4膨胀至445.1，且随着问题难度增加（如C8），差距从数倍扩大至两个数量级。



在最大稳定集问题上，Table 2和Table 3分别展示了与LDPP和S2V-DQN的对比。然而，提取文本中未包含具体的数值结果，仅说明"results demonstrate effectiveness"。这一部分的证据强度明显弱于主实验。

**公平性检查**：本文存在若干方法学局限。首先，**Random Search作为baseline过于薄弱**——它无学习、无记忆、无信用分配，与DQN的比较更像"有无学习"的验证而非"学习策略优劣"的对比。同任务的直接竞争者LDPP [14] 仅在最大稳定集问题上对比，未在核心多项式证明基准上正面较量。其次，**未与SDP/SOS标准方法对比**（如SOSTOOLS、MOSEK），而这些是多项式优化领域的事实标准；未与商用LP求解器（Gurobi等）在相同LP表述上比较效率。第三，**仅10个基准实例**且来自同一数据库，代表性有限；无统计显著性检验。第四，训练使用双NVIDIA L40S GPU和1TB RAM，硬件配置较高，但未报告具体训练时间。最后，作者未披露失败模式——例如当Handelman定理不适用时（非紧多面体定义域）方法行为如何。

## 方法谱系与知识库定位

APPIRL属于**强化学习 × 符号计算**的交叉方法族，直接继承自 **S2V-DQN**（Khalil et al., 用Structure2Vec图神经网络解决组合优化的DQN方法）。与S2V-DQN相比，APPIRL保留了DQN+经验回放的训练框架，但将**状态编码从图神经网络替换为动态多项式基编码**，**动作空间从固定图节点选择替换为动态增长的乘子生成**，并**新增gamma跟踪奖励塑形**以替代组合优化中的简单目标值奖励。

直接baseline差异：
- **Random Search**: APPIRL用DQN替代其均匀随机动作选择，引入时序差分信用分配
- **LDPP [14]**: 同为学习动态多项式证明，但LDPP针对图论问题设计，APPIRL扩展至一般多项式不等式并引入FFT加速
- **S2V-DQN [21]**: APPIRL继承其DQN框架，但放弃图神经网络（问题非图结构），改用MLP编码动态证明状态

后续可拓展方向：
1. **与SDP/SOS方法的混合**：在DQN选择的基上使用SDP而非LP，可能获得更强的证明能力但牺牲可扩展性
2. **策略梯度替代值函数方法**：PPO或A3C可能更适合动作空间动态增长的信用分配问题
3. **神经定理证明器的集成**：将APPIRL作为子模块嵌入Lean/Coq等形式化验证系统，提供自动化不等式证明策略

标签定位：
- **modality**: 符号数学 / 多项式代数
- **paradigm**: 深度强化学习 (DQN) + 凸松弛 (LP)
- **scenario**: 自动定理证明 / 形式化验证辅助
- **mechanism**: 动态基构造 / 奖励塑形 / FFT加速
- **constraint**: 紧多面体定义域（Handelman定理适用条件）

