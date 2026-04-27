---
title: Does Stochastic Gradient really succeed for bandits?
type: paper
paper_level: C
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 老虎机随机梯度学习率阈值刻画
- Stochastic Gradi
- Stochastic Gradient Bandit (SGB) with characterized learning rate regimes
- For two-armed bandits
acceptance: Oral
method: Stochastic Gradient Bandit (SGB) with characterized learning rate regimes
modalities:
- tabular
paradigm: supervised
baselines:
- 任意学习率下随机梯度老虎机的全局_Stochastic_Gradi
---

# Does Stochastic Gradient really succeed for bandits?

**Topics**: [[T__Reinforcement_Learning]] | **Method**: [[M__Stochastic_Gradient_Bandit_(SGB)_with_characterized_learning_rate_regimes]] | **Datasets**: K=5 bandit with equal gaps, K=10 bandit with varying gaps, K=5 bandit with extreme low rewards, K=4 bandit with extreme high rewards, Two-armed bandit asymptotic tightness

> [!tip] 核心洞察
> For two-armed bandits, there exists a sharp threshold in learning rate (scaling with the sub-optimality gap Δ) that separates logarithmic and polynomial regret regimes, and for K-armed bandits the learning rate must additionally scale inversely with K to avoid polynomial regret.

| 中文题名 | 老虎机随机梯度学习率阈值刻画 |
| 英文题名 | Does Stochastic Gradient really succeed for bandits? |
| 会议/期刊 | NeurIPS 2025 (Oral) |
| 链接 | [arXiv](https://arxiv.org/abs/2501.0xxxx) · [Code](待补充) · [Project](待补充) |
| 主要任务 | Multi-Armed Bandit / Reinforcement Learning |
| 主要 baseline | Thompson Sampling (TS), MED, UCB, SAMBA, SGB (Mei et al. [1,2]) |

> [!abstract] 因为「SGB 的学习率选择缺乏理论指导，仅知小学习率有渐近收敛但无明确遗憾界」，作者在「Mei et al. 的 SGB」基础上改了「对学习率进行 sharp threshold 刻画：K=2 时阈值与 Δ 成正比，K 臂时需额外 O(1/K) 缩放」，在「5/10 臂合成老虎机」上取得「与 TS 在规格良好时相当，但变差距场景劣于 TS/MED」

- K=2 时存在 sharp threshold：学习率低于 c·Δ 得对数遗憾 O(log T)，高于则多项式遗憾
- K 臂时学习率需 O(1/K) 缩放以避免多项式遗憾
- 经验验证：对数坐标下经验遗憾斜率与理论曲线 log(t)/(2η) 匹配

## 背景与动机

多臂老虎机（Multi-Armed Bandit）是序列决策的核心问题：智能体需在 K 个臂中反复选择以最大化累积奖励，同时平衡探索与利用。一个具体场景是广告推荐系统——每天需从 K 个候选广告中选择展示，但各广告的点击率未知，且每次选择都是一次实时权衡。

现有方法各有利弊：Thompson Sampling (TS) 通过维护奖励后验分布并采样决策，在多种场景下表现优异但需设计合适的先验；UCB 基于乐观估计构造置信上界，理论保证清晰但对分布边界敏感；MED 为各臂设计独立探索机制，在变差距场景下优化但实现复杂。近年来，Mei et al. [1] 提出 Stochastic Gradient Bandit (SGB)，将 softmax 策略参数化与 REINFORCE 梯度估计结合，用恒定学习率随机梯度上升更新，证明了渐近收敛性，但遗留关键问题：学习率如何影响有限时间遗憾？多大学习率仍能保证对数遗憾？

Mei et al. [2] 后续扩展至任意学习率的全局收敛，但仍未给出学习率与问题参数（gap Δ、臂数 K）的定量关系。这一理论空白导致实践者无法判断：给定一个具体老虎机问题，应选 η=0.01 还是 η=0.1？本文正是要回答这一核心问题——通过 sharp 刻画学习率阈值，明确 SGB 何时真正成功、何时失败。



## 核心创新

核心洞察：SGB 的恒定学习率存在由问题参数决定的 sharp threshold，因为梯度动态在次优臂附近的竞争行为具有相变特性——低于阈值时最优臂概率 p₁ₜ 快速趋近 1 使遗憾对数增长，高于阈值时次优臂被过度探索导致多项式遗憾，从而使学习率的可证明选择从"经验猜测"变为"参数化公式"成为可能。

| 维度 | Baseline (Mei et al. [1,2]) | 本文 |
|:---|:---|:---|
| 学习率选择 | 小学习率有渐近收敛保证，大学习率仅知全局收敛 | K=2 时阈值 ∝ Δ；K 臂时需 O(1/K) 缩放 |
| 遗憾界 | 无明确有限时间遗憾界，或仅多项式上界 | 阈值以下证明 O(log T) 对数遗憾 |
| 分析技术 | 标准 Lyapunov/梯度流分析 | 将遗憾转化为最优臂选择概率，精细分析梯度动态相变 |

## 整体框架



SGB 的标准 pipeline 包含四个模块，本文的创新集中在第四模块的分析而非算法结构改变：

1. **Softmax 策略参数化**：输入臂偏好 θₜ ∈ ℝᴷ，输出选择概率 pₖₜ = exp(θₖₜ)/∑ⱼexp(θⱼₜ)，将离散动作选择转化为可微概率分布。

2. **REINFORCE 梯度估计**：输入选中臂奖励 rₜ 与当前概率 pₜ，输出随机梯度估计 ∇θ log pₐₜ,ₜ · rₜ，通过采样获得无偏梯度。

3. **恒定学习率 SGD 更新**：输入当前参数 θₜ 与梯度估计，输出 θₜ₊₁ = θₜ + η · ∇θ，保持学习率 η 不变贯穿训练。

4. **学习率刻画与遗憾分析**（本文核心）：输入学习率 η 与问题参数 (K, Δ)，输出可证明的遗憾上界保证——K=2 时判定 η < cΔ 是否成立，K 臂时判定 η = O(1/K) 是否满足。

数据流：臂偏好 θ₀ → [Softmax] → 概率 p₀ → [采样] → 动作 a₀, 奖励 r₀ → [REINFORCE] → 梯度 g₀ → [SGD, η] → θ₁ → ... → 累积遗憾 Rₜ。本文在标准 flow 外增加理论判定模块，指导 η 的选择。

```
θ_t ──[softmax]──> p_t ──[sample]──> (a_t, r_t)
  ↑                                    │
  └────[SGD: θ+η·∇]←──[REINFORCE]←───┘
              ↑
         η ← [理论判定: η∝Δ or O(1/K)]
```

## 核心模块与公式推导

### 模块 1: 遗憾定义与转化（对应框架图"理论分析"模块）

**直觉**: 将多臂累积遗憾转化为对最优臂选择概率的控制，避免直接处理各臂差距的复杂耦合。

**Baseline 公式** (标准多臂老虎机): $$R^\pi_T = \mathbb{E}^\pi\left[\sum_{t=1}^T \sum_{k=1}^K p^\pi_{k,t} \Delta_k\right]$$
符号: $\pi$ = 策略, $p^\pi_{k,t}$ = 第 t 步选臂 k 的概率, $\Delta_k = \mu^* - \mu_k$ = 臂 k 与最优臂的期望奖励差距。

**变化点**: 标准定义直接求和各臂概率加权威差距，难以分析。本文关键观察——由于 $\Delta_k \leq \max_k \Delta_k$ 且 $\sum_{k \neq 1} p_{k,t} = 1 - p_{1,t}$，可将多臂耦合松弛为单臂控制问题。

**本文公式（推导）**:
$$\text{Step 1}: R^\pi_T = \mathbb{E}^\pi\left[\sum_{t=1}^T \sum_{k=1}^K p^\pi_{k,t} \Delta_k\right] \leq \mathbb{E}^\pi\left[\sum_{t=1}^T (1-p^\pi_{1,t})\right] \cdot \max_k \Delta_k \quad \text{（上界松弛与概率归一化）}$$
$$\text{Step 2}: \text{分析 SGD 动态证明 } \sum_{t=1}^T (1-p^\pi_{1,t}) = O(\log T / \eta) \quad \text{（精细梯度流分析，需学习率阈值分类讨论）}$$
$$\text{最终}: R^\pi_T = O\left(\frac{\log T}{\eta}\right) \text{ 当 } \eta < \eta_{\text{threshold}}(\Delta, K)$$

**对应消融**: 当 η 超过阈值时，理论保证失效，经验上遗憾从对数增长突变为多项式增长（Figure 1-2 显示 10-90 分位数区间显著扩大）。

### 模块 2: 渐近紧性验证曲线（对应框架图"经验验证"模块）

**直觉**: 为验证定理 1 中 log(T) 因子的紧性，构造可直接对比的理论预测曲线。

**Baseline 公式** (无——此前 SGB 工作未提供有限时间可验证的渐近形式): 无显式预测。

**变化点**: 本文首次给出学习率 η 与渐近遗憾斜率的精确关系，使理论可从对数坐标图中直接检验。

**本文公式（推导）**:
$$\text{Step 1}: \text{由定理 1 上界 } R_T \text{lesssim} \frac{\log T}{2\eta} \text{（常数因子吸收后）} \quad \text{（从 Step 2 的 O(log T/η) 提炼具体系数）}$$
$$\text{Step 2}: \text{定义理论曲线 } t \mapsto \frac{\log(t)}{2\eta} \text{，在对数-线性坐标下为斜率 } 1/(2\eta) \text{ 的直线} \quad \text{（取对数得 log-log 坐标下斜率为 1，截距由 η 决定）}$$
$$\text{最终}: \hat{R}^M_t = \frac{1}{M}\sum_{m=1}^M R^{(m)}_t \approx \frac{\log(t)}{2\eta} \text{ for large } t, \text{ with } M = 10^3$$
符号: $\hat{R}^M_t$ = M 次独立运行的蒙特卡洛平均遗憾，用于实验验证理论曲线。

**对应消融**: Figure 7 显示 η ∈ {0.05, 0.1, 0.2} 时经验斜率与理论曲线 log(t)/(2η) 匹配，验证 log(T) 因子的紧性；若移除该理论预测则无定量验证基准。

### 模块 3: K 臂 O(1/K) 缩放律（对应框架图"理论扩展"模块）

**直觉**: K 增加时，softmax 概率空间的竞争维度扩大，固定学习率导致梯度更新在各臂间"分散"，需额外缩放维持对数遗憾。

**Baseline 公式** (Mei et al. [2] 的任意学习率全局收敛): 仅保证收敛，无 K 的显式依赖。

**变化点**: 本文发现 K 增加时，学习率阈值本身需按 O(1/K) 收缩，这是此前未识别的关键缩放律——固定 η 在 K 增大时必然落入多项式遗憾区。

**本文公式（推导）**:
$$\text{Step 1}: \text{K=2 阈值 } \eta_{\text{thr}}^{(2)} = c \cdot \Delta \text{（c 为绝对常数）} \quad \text{（双臂竞争分析）}$$
$$\text{Step 2}: \text{K 臂时，softmax 梯度在 K 个方向竞争，有效步长被稀释 } \sim 1/K \quad \text{（高维 softmax 几何分析）}$$
$$\text{最终}: \eta_{\text{thr}}^{(K)} = O\left(\frac{\Delta}{K}\right) \text{ 或等价地要求 } \eta = O(1/K) \text{ 当 } \Delta = \Theta(1)$$

**对应消融**: Figure 5 (Right) 的 10 臂变差距场景显示，使用固定 η（未按 1/K 缩放）的 SGB 显著劣于 TS/MED，遗憾达 ~800-1000，而 TS/MED 更低。

## 实验与分析



本文在合成老虎机环境上系统评估 SGB，核心发现围绕"规格良好 vs 规格不良"的对比展开。Figure 5 (Left) 的 5 臂等差距场景（μ₁=0.1, μ₂=...=μ₅=0, Δ=0.1）中，SGB 在 η 按 Δ 调谐时与 TS 性能接近，两者遗憾均在 200-400 范围，显著优于 UCB 和 SAMBA。这验证了本文核心主张：当问题参数（Δ）已知且学习率匹配时，SGB 可达到 state-of-the-art 水平。

然而 Figure 5 (Right) 的 10 臂变差距场景（μ₁=0.5, μ₂=0.3, μ₃₋₈=0, μ₉₋₁₀=-0.5）揭示关键局限：固定学习率无法同时适应多个差距尺度，SGB 对明显次优臂（μ₉₋₁₀=-0.5）过度探索，遗憾升至 ~800-1000，而 TS 和 MED 通过自适应机制保持更低遗憾。此结果直接支持定理中"学习率需匹配 Δ"的结论——当多个 Δₖ 共存时，单一 η 必然错配部分臂。


![Figure 1, Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/89efde13-19b3-4d29-bb95-92a3ab1e6031/figures/Figure_1,_Figure_2.png)
*Figure 1, Figure 2 (result): Figure 1: Δ=1/8 average regret and 10-90 percentiles for 10 independent runs; Figure 2: Δ=1/8 average regret and 10-90 percentile with empirical regret distribution at time T*



Figure 6 进一步测试极端奖励分布：左图低奖励场景（μ₁=-0.8 至 μ₅=-0.95）和右图高奖励场景（μ₁=0.96, μ₂₋₃=0.92, μ₄=0.9）中，SGB 始终优于 UCB，但 MED 和 TS 利用 KL 散度或定制后验更好适应边界几何。这表明 SGB 的梯度更新机制缺乏对分布边界的显式适应，是其结构性局限。



渐近紧性验证（Figure 7，Appendix G.3）是理论贡献的关键支撑：K=2, μ₁=0, μ₂=-0.25, T=10⁵ 设置下，η ∈ {0.05, 0.1, 0.2} 的经验遗憾斜率与理论曲线 log(t)/(2η) 在对数坐标下精确匹配，验证定理 1 中 log(T) 因子的紧性。若该因子松弛（如变为 log²T），则斜率将系统性偏离。

公平性检查：对比基线包含 TS、MED、UCB、SAMBA，覆盖贝叶斯、频率派、梯度派方法，选择合理。但需注意不对称性——SGB 需 oracle 知识调谐 η∝Δ，而 TS/MED 无此要求；实验未比较自适应/衰减学习率变体；全部限于表格老虎机，无函数近似扩展。

## 方法谱系与知识库定位

方法家族：Policy Gradient / Bandit Algorithms。父方法为 Mei et al. [1] "Stochastic gradient succeeds for bandits" 及其后续 [2] "Small steps no more"，本文直接质疑其标题论断并补充 sharp 条件。

改变槽位：
- **training_recipe**: 恒定学习率 → 恒定学习率 + 阈值刻画（Δ 比例 for K=2, O(1/K) for K 臂）
- **credit_assignment**: 标准 REINFORCE → 相同估计器 + 新遗憾分析技术（最优臂概率转化上界）
- 架构/数据/推理均未改变，纯理论分析补丁

直接基线差异：
- **SGB (Mei et al. [1])**: 本文补充了 [1] 缺失的学习率-遗憾定量关系
- **SGB (Mei et al. [2])**: [2] 证明任意学习率全局收敛，本文进一步区分对数/多项式遗憾区
- **TS/MED**: SGB 在规格良好时匹配，但需 oracle Δ 知识；TS/MED 自适应但计算更复杂

后续方向：
1. 自适应学习率：消除对 oracle Δ 的依赖，如基于经验差距估计的动态 ηₜ 调整
2. 函数近似扩展：将 K 臂 O(1/K) 缩放律推广至线性/神经网络参数化老虎机
3. 方差缩减结合：融合 [22] 的方差缩减技术降低 SGB 样本复杂度

标签：modality=tabular | paradigm=supervised (bandit feedback) | scenario=online decision making | mechanism=policy gradient + softmax | constraint=requires gap knowledge for tuning

## 引用网络

### 直接 baseline（本文基于）

- [[P__任意学习率下随机梯度老虎机的全局_Stochastic_Gradi]] _(直接 baseline)_: Very recent follow-up to [1] extending SGB convergence; paper directly builds on

