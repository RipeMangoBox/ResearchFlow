---
title: Tracking The Best Expert Privately
type: paper
paper_level: A
venue: ICML
year: 2025
paper_link: null
aliases:
- 差分隐私下的动态专家追踪
- Private dynamic
- Private dynamic regret algorithms (unnamed family)
- Sub-linear dynamic regret is achiev
acceptance: Poster
cited_by: 564
method: Private dynamic regret algorithms (unnamed family)
modalities:
- Text
paradigm: online learning with differential privacy
followups:
- 统计均等性的指数权重元算法_SPEW_(Statistica
---

# Tracking The Best Expert Privately

**Topics**: [[T__Reasoning]] | **Method**: [[M__Private_dynamic_regret_algorithms]] | **Datasets**: Dynamic regret with expert advice - Stochastic shifting adversary, Dynamic regret with expert advice - Oblivious adversary, Dynamic regret with expert advice - Adaptive adversary

> [!tip] 核心洞察
> Sub-linear dynamic regret is achievable with differential privacy for stochastic shifting and oblivious adversaries, but a fundamental separation exists for adaptive adversaries in high-privacy regimes.

| 中文题名 | 差分隐私下的动态专家追踪 |
| 英文题名 | Tracking The Best Expert Privately |
| 会议/期刊 | ICML 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2503.09889) · [Code](待补充) · [Project](待补充) |
| 主要任务 | Prediction with expert advice（专家预测）、Dynamic regret minimization（动态遗憾最小化）、Dueling bandits（对决老虎机） |
| 主要 baseline | Fixed Share / Mixing Past Posteriors [7][19]、Mirror Descent Fixed Share [9]、EXP3 / Sparring-EXP3、Private online prediction from experts [5] |

> [!abstract] 因为「动态遗憾最小化要求比较器序列随时间变化，而差分隐私要求对损失添加噪声，两者结合会导致隐私误差在切换点累积」，作者在「Fixed Share / Mixing Past Posteriors」基础上改了「目标函数为动态遗憾+隐私约束、奖励设计为高斯机制加噪损失、信用分配为区间划分归约」，在「随机切换对抗」上取得「O(√(ST log(NT)) + S log(NT)/ε)」的次线性遗憾界，并证明「自适应对抗在 ε ≤ √(S/T) 时必遭线性遗憾」的分离结果。

- **随机切换对抗**：动态遗憾上界 O(√(ST log(NT)) + S log(NT)/ε)，首次实现次线性私有动态遗憾
- **遗忘对抗**：通过归约获得 ε^(-2/3) 隐私依赖，优于朴素方法的 ε^(-1)
- **自适应对抗下界**：当 ε ≤ √(S/T) 时，任何 ε-DP 算法必遭 Ω(T) 线性遗憾

## 背景与动机

在在线学习中，预测者每天需要从 N 个专家的建议中选择一个，目标是让自己的累计损失接近最佳专家。经典设定假设存在**固定的**最佳专家，用**静态遗憾**（static regret）衡量。然而现实中最佳策略会随时间漂移——例如股市中牛市和熊市的最佳分析师不同，推荐系统中用户兴趣会发生话题转移。此时需要**动态遗憾**（dynamic regret），允许比较器序列 (i₁*, ..., i_T*) 随时间变化，且切换次数 S 有限。

现有方法如何处理这一问题？**Fixed Share** [7][19] 通过在每个时刻以固定概率混合均匀分布，让算法能追踪变化的最佳专家，达到 O(√(ST log N)) 的非私有最优速率。**Mirror Descent Fixed Share** [9] 用镜像下降统一了该框架。**EXP3** 及其变体则处理只有部分反馈的 bandit 场景。

但这些方法都假设**损失可以被直接观测和使用**。当涉及用户敏感数据（如医疗记录、金融决策、个人偏好）时，直接暴露损失序列会泄露隐私。近年工作如 [5] 开始研究**私有专家预测**，但仅针对**静态遗憾**。一个自然的问题是：能否在保护隐私的同时，依然高效追踪变化的最佳专家？

核心难点在于：**动态遗憾的切换结构与高斯噪声的累积效应相互作用**。朴素地在每个时刻对损失加噪，隐私误差会在 S 个切换区间上累积，导致遗憾界爆炸；而对于**自适应对抗**（adaptive adversary），对手能根据算法过去的随机决策调整损失，进一步放大隐私噪声的破坏力。本文首次系统地回答了这一问题，揭示了不同对抗类型下隐私与动态遗憾的根本权衡。

## 核心创新

核心洞察：**动态遗憾可以分解为若干静态遗憾区间的和，而隐私噪声的累积可以通过精心设计的区间划分来控制**，因为对于遗忘对抗，几何增长的区间长度能让隐私预算的消耗从线性变为次线性，从而使 ε^(-2/3) 的隐私依赖成为可能。

| 维度 | Baseline (Fixed Share / [5]) | 本文 |
|------|------------------------------|------|
| 目标函数 | 静态遗憾：minᵢ Σ ℓₜ(i) | 动态遗憾：Σ ℓₜ(iₜ*)，允许 S 次切换 |
| 隐私机制 | 无隐私 [7][19]；或静态遗憾+高斯机制 [5] | 动态遗憾+高斯机制，针对三种对抗类型分别设计 |
| 信用分配 | 直接混合过去后验 | 遗忘对抗：区间划分归约到静态遗憾；随机切换：直接加噪 Fixed Share |
| 隐私-遗憾权衡 | 静态场景已知 | 揭示自适应 vs 遗忘的根本分离：ε ≤ √(S/T) 时自适应必线性 |

## 整体框架



本文算法家族包含三条针对不同对抗类型的管线，共享**高斯隐私机制**作为前端，但采用不同的**信用分配策略**：

1. **隐私机制层（Gaussian Mechanism）**：输入原始损失 ℓₜ(i) 或对决偏好反馈，输出加噪损失 ℓ̃ₜ(i) = ℓₜ(i) + Zₜ，其中 Zₜ ~ N(0, σ²)，σ = O(√(log(T/δ))/ε)。该层保证整个 T 轮交互满足 (ε, δ)-差分隐私。

2. **学习层（Private Exponential Weights / Mirror Descent）**：输入加噪损失 ℓ̃ₜ，维护专家分布 pₜ 并通过指数权重更新 pₜ₊₁(i) ∝ pₜ(i) · exp(-ηℓ̃ₜ(i))。学习率 η 需根据噪声方差重新标定。

3. **信用分配层（三种变体）**：
   - **随机切换（Stochastic Shifting）**：直接运行加噪版 Fixed Share，已知切换次数 S；
   - **遗忘对抗（Oblivious）**：将 [1,T] 划分为几何增长长度的区间，每个区间运行私有静态遗憾算法，通过**区间长度与隐私预算的平衡**获得改进的 ε 依赖；
   - **自适应对抗（Adaptive）**：证明下界——当 ε ≤ √(S/T) 时，任何 ε-DP 算法必遭 Ω(T) 线性遗憾，无需设计算法。

4. **动作选择层**：根据 pₜ 采样或选择动作 aₜ，接收环境反馈，循环至 T。

对于**对决老虎机**扩展，将上述管线中的损失替换为成对偏好，使用 **Sparring-EXP3** 架构，两个 bandit 实例互相对决，偏好反馈同样经高斯机制加噪。

```
原始损失 ℓₜ 或偏好反馈
    ↓
[高斯机制] 添加噪声 Zₜ ~ N(0, σ²)
    ↓
加噪损失 ℓ̃ₜ
    ↓
[私有指数权重 / 镜像下降] 更新分布 pₜ
    ↓
[信用分配] 
   ├─ 随机切换: 加噪 Fixed Share
   ├─ 遗忘对抗: 几何区间划分 → 静态遗憾子程序
   └─ 自适应: 下界证明（不可行）
    ↓
动作选择 aₜ ~ pₜ 或 argmax
    ↓
环境反馈，循环
```

## 核心模块与公式推导

### 模块 1: 动态遗憾定义与隐私化损失（对应框架图 输入→隐私机制层）

**直觉**：动态遗憾允许比较器随时间变化，但隐私要求对观测值加噪，需要明确定义"保护什么、比较什么"。

**Baseline 公式** (Fixed Share [19], 非私有静态遗憾):
$$\text{Regret}_T = \sum_{t=1}^T \text{ell}_t(a_t) - \min_{i \in [N]} \sum_{t=1}^T \text{ell}_t(i)$$
符号：$\text{ell}_t(i)$ = 专家 $i$ 在时刻 $t$ 的损失，$a_t$ = 算法选择的动作，$N$ = 专家数。

**变化点**：静态遗憾假设存在固定最佳专家，无法建模环境漂移；且未考虑隐私约束。

**本文公式（推导）**:
$$\text{Step 1}: \quad \text{D-Regret}_T = \sum_{t=1}^T \text{ell}_t(a_t) - \sum_{t=1}^T \text{ell}_t(i_t^*) \quad \text{比较器序列 } (i_t^*)_{t=1}^T \text{ 允许 } S \text{ 次切换}$$
$$\text{Step 2}: \quad \tilde{\text{ell}}_t(i) = \text{ell}_t(i) + Z_t, \quad Z_t \sim \mathcal{N}(0, \sigma^2), \quad \sigma = O\left(\frac{\sqrt{\log(T/\delta)}}{\varepsilon}\right) \quad \text{高斯机制保证 } (\varepsilon,\delta)\text{-DP}$$
$$\text{最终}: \quad \mathbb{E}[\text{D-Regret}_T^{\text{private}}] = \mathbb{E}\left[\sum_{t=1}^T \tilde{\text{ell}}_t(a_t) - \sum_{t=1}^T \tilde{\text{ell}}_t(i_t^*)\right] + \text{噪声引起的偏差项}$$

**对应消融**：Table 1 显示非私有下界为 Ω(√(ST))，隐私引入附加项。

---

### 模块 2: 随机切换对抗的私有算法（对应框架图 信用分配层-随机切换分支）

**直觉**：若切换位置随机且切换次数 S 已知，可直接在 Fixed Share 上加噪，隐私误差在 S 个区间上均匀分摊。

**Baseline 公式** (Fixed Share [7]):
$$\mathbb{E}[\text{D-Regret}_T] = O(\sqrt{ST \log(NT)})$$
符号：$S$ = 切换次数，$T$ = 时间范围，$N$ = 专家数。

**变化点**：Baseline 假设损失完全可观测且无隐私约束。加入高斯噪声后，噪声方差 σ² 会在每个区间累积，需要重新平衡学习率 η 与噪声的关系。

**本文公式（推导）**:
$$\text{Step 1}: \quad \text{将 } [1,T] \text{ 按切换点分为 } S+1 \text{ 个区间 } I_1, \ldots, I_{S+1}，\text{每段内比较器固定}$$
$$\text{Step 2}: \quad \text{每段运行私有静态遗憾：} O\left(\sqrt{|I_j| \log N} + \frac{|I_j| \cdot \sigma}{\sqrt{|I_j|}}\right) = O\left(\sqrt{|I_j| \log N} + \sqrt{|I_j|} \cdot \frac{\sqrt{\log(T/\delta)}}{\varepsilon}\right)$$
$$\text{Step 3}: \quad \text{最优平衡 } |I_j| \approx T/S，\text{ 应用 AM-GM: } \sum_{j=1}^{S+1} \sqrt{|I_j|} \leq \sqrt{(S+1)T}$$
$$\text{最终}: \quad \mathbb{E}[\text{D-Regret}_T] = O\left(\sqrt{ST \log(NT)} + \frac{S \log(NT)}{\varepsilon}\right)$$

第二项 $S \log(NT)/\varepsilon$ 是**隐私代价**：与切换次数 S 成正比，与隐私预算 ε 成反比。当 ε 为常数时，整体仍为次线性。

**对应消融**：Table 1 显示此为随机切换设置的最优上界（匹配非私有速率加隐私项）。

---

### 模块 3: 遗忘对抗的区间划分归约（对应框架图 信用分配层-遗忘对抗分支）

**直觉**：遗忘对抗的切换位置未知且可能对抗性选择，但损失序列在交互前固定。关键观察是：**无需知道切换位置，用几何增长的区间长度即可"覆盖"所有可能的切换模式**，同时让隐私预算的消耗呈次线性。

**Baseline 公式** (朴素区间划分):
$$\mathbb{E}[\text{D-Regret}_T] = O\left(\frac{ST}{\varepsilon}\right) \quad \text{或更差}$$
朴素方法：将 T 均分为 S 个区间，每段分配 ε/S 隐私预算，但误差随 S 线性爆炸。

**变化点**：朴素方法隐私预算按区间数量分配，导致 ε^(-1) 依赖且对 S 敏感。本文采用**几何增长区间 + 动态遗憾到静态遗憾的归约**，让隐私噪声的累积从"区间数量"变为"区间长度的分数幂"。

**本文公式（推导）**:
$$\text{Step 1}: \quad \text{D-Regret}_T \leq \sum_{j=1}^{K} \text{Static-Regret}(I_j) + \text{切换惩罚}，\text{其中 } K \text{ 为区间数}$$
$$\text{Step 2}: \quad \text{取几何增长区间: } |I_j| = 2^{j-1}，K = O(\log T)，\text{总长度覆盖 } [1,T]$$
$$\text{Step 3}: \quad \text{每段运行 } (\varepsilon/2)\text{-DP 静态算法，隐私由组合定理保证}$$
$$\text{Step 4}: \quad \text{关键平衡：区间长度 } L \text{ vs. 区间数量 } K \approx T/L \text{ vs. 每段隐私误差 } \propto L^{2/3}/\varepsilon^{2/3}$$
$$\text{最终}: \quad \mathbb{E}[\text{D-Regret}_T] = O\left(\sqrt{ST \log(NT)} + \frac{ST^{1/3} \log(T/\delta) \log(NT)}{\varepsilon^{2/3}}\right)$$

ε^(-2/3) 依赖的来源：几何划分使得"区间数量"与"每段误差"的乘积中，隐私预算以 2/3 次幂进入——这是**动态到静态归约**的核心收益，优于任何直接方法的 ε^(-1)。

**对应消融**：Table 1 明确对比了朴素方法的 ε^(-1) 与本文的 ε^(-2/3)；若改用固定长度区间，隐私依赖退化为 ε^(-1)。

## 实验与分析


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/0353938a-ba5a-4e0d-b384-2f614b6fc189/figures/Table_1.png)
*Table 1 (comparison): Summary of our bounds on the dynamic regret for different settings of the adversary. We omit logarithmic factors in T and 1/δ.*



本文是**理论论文**，核心结果以**上下界证明**呈现，汇总于 Table 1。该表系统比较了三种对抗设置（随机切换、遗忘、自适应）下的动态遗憾界，省略了 T 和 1/δ 的对数因子。

**随机切换对抗**（Stochastic Shifting）：本文证明上界 O(√(ST log(NT)) + S log(NT)/ε)。第一项 √(ST log(NT)) 匹配非私有最优速率 [7][19]；第二项 S log(NT)/ε 是隐私引入的**附加代价**，与切换次数 S 和隐私预算 ε 直接相关。当 ε = Ω(√(S/T)) 时，整体仍为次线性。这是**首个**该设定下的私有次线性动态遗憾算法。

**遗忘对抗**（Oblivious）：上界 O(√(ST log(NT)) + ST^(1/3) log(T/δ) log(NT) / ε^(2/3))。核心进步在于隐私依赖从朴素方法的 ε^(-1) 改进为 ε^(-2/3)。具体而言，若采用朴素区间划分（均分 T/S 段，每段 ε/S 预算），隐私项为 O(ST/ε)；本文的几何划分归约将指数从 1 降至 2/3。以 S = T^(1/3), ε = T^(-1/6) 为例，朴素方法隐私项为 O(T^(7/6))（超线性），本文为 O(T^(2/3) · T^(1/9)) = O(T^(7/9))（次线性）。

**自适应对抗**（Adaptive）：下界 Ω(T) 当 ε ≤ √(S/T)。这意味着当隐私要求足够严格（ε 小）且环境变化相对缓慢（S/T 小）时，**任何 ε-DP 算法都无法避免线性遗憾**。这是一个**负面结果**，但具有重要价值：它揭示了遗忘对抗与自适应对抗之间的**根本分离**——前者可通过巧妙设计获得次线性遗憾，后者在高压隐私下不可行。

**消融与验证**：本文未提供传统意义上的数值消融实验，但 Table 1 本身构成了方法对比的核心。关键消融隐含在"朴素区间划分 vs. 几何划分"的对比中：去掉归约框架（即使用固定长度区间），遗忘对抗的隐私依赖从 ε^(-2/3) 退化为 ε^(-1)，在 ε → 0 时差距无限放大。

**公平性检查**：
- 对比的 baseline [5] 是**直接前驱工作**（同作者，私有专家预测），但仅处理静态遗憾；[7][19] 是**非私有最优方法**，作为速率基准合理。
- **缺失**：无 empirical 实验验证，所有结果为理论界；未与 [4][6] 等最新私有在线优化工作进行数值对比。
- **已知局限**：算法需要**预先知道切换次数 S**；对决老虎机结果直接套用高斯机制，无新颖隐私会计；自适应下界在部分参数区域可能不紧。
- **计算/数据预算**：不适用，纯理论论文。

## 方法谱系与知识库定位

**方法家族**：Online Learning with Differential Privacy（差分隐私在线学习）

**父方法**：Fixed Share / Mixing Past Posteriors [7][19] —— 追踪最佳专家的经典框架，本文在其上加装隐私模块并改为动态遗憾目标。

**改变的插槽**：
- **目标函数**：静态遗憾 → 动态遗憾（允许 S 次切换）
- **奖励设计**：原始损失 → 高斯机制加噪损失
- **信用分配**：直接混合后验 → 区间划分归约（遗忘对抗）/ 直接加噪（随机切换）
- **训练/推断策略**：标准指数权重 → 噪声感知学习率标定的私有指数权重

**直接 Baselines 与差异**：
- **[5] Private online prediction from experts**：同作者前作，处理**静态遗憾**+隐私；本文扩展至**动态遗憾**，揭示 oblivious-adaptive 分离
- **[7][19] Fixed Share / Mixing Past Posteriors**：非私有动态遗憾最优算法；本文添加高斯机制，证明隐私代价可控制
- **[9] Mirror Descent Fixed Share**：镜像下降视角；本文沿用其优化框架但重新标定学习率以容纳噪声
- **Sparring-EXP3**：非私有对决老虎机；本文扩展为**私有 Sparring**，偏好反馈经高斯机制处理

**后续方向**：
1. **移除对 S 的先验知识**：设计自适应于未知切换次数的私有动态遗憾算法
2. **收紧自适应下界**：当前 Ω(T) 下界在 ε > √(S/T) 时存在空白，需匹配上界或证明更精细的相位转移
3. **实证验证与实用部署**：在真实推荐系统或医疗预测数据上验证理论界，开发更精细的隐私会计（如 Rényi DP）替代高斯机制

**标签**：
- **Modality**: text / tabular（专家损失序列）
- **Paradigm**: online learning, differential privacy
- **Scenario**: prediction with expert advice, dueling bandits
- **Mechanism**: Gaussian mechanism, exponential weights, interval partition reduction
- **Constraint**: ε-differential privacy, dynamic regret, oblivious vs adaptive adversary

## 引用网络

### 后续工作（建立在本文之上）

- [[P__统计均等性的指数权重元算法_SPEW_(Statistica]]: Foundational expert tracking with exponential weights; relevant for SPEW's exper

