---
title: Evaluation-driven Scaling for Scientific Discovery
type: paper
paper_level: B
venue: Unknown
year: 2026
paper_link: https://arxiv.org/abs/2604.19341
aliases:
- 面向科学发现的评估驱动扩展方法
- ESSD
paradigm: Reinforcement Learning
---

# Evaluation-driven Scaling for Scientific Discovery

[Paper](https://arxiv.org/abs/2604.19341)

**Topics**: [[T__Agent]], [[T__Benchmark_-_Evaluation]], [[T__Code_Generation]]

| 中文题名 | 面向科学发现的评估驱动扩展方法 |
| 英文题名 | Evaluation-driven Scaling for Scientific Discovery |
| 会议/期刊 | arXiv preprint (2026) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.19341) · [Code](https://github.com/HaotianYe/simple-tes ⭐待补充) · [Project](待补充) |
| 主要任务 | 数学猜想发现（自相关不等式、Erdős最小重叠问题）、量子比特路由优化、组合优化等21个科学问题 |
| 主要 baseline | Best-of-N采样、Sequential Halving、Verification-enhanced BoN、 Majority Voting、Self-Consistency |

> [!abstract] 因为「LLM在科学发现中采样效率低下、评估预算分配盲目」，作者在「Best-of-N采样」基础上改了「将评估预算N = C × L × K三维分解并引入sequential halving淘汰机制」，在「21个科学问题（含自相关不等式、Erdős最小重叠、量子比特路由）」上取得「相比Best-of-N最高达数个数量级的样本效率提升，以1/64预算匹配或超越其性能」

- **关键性能1**: 在自相关不等式任务上，SimpleTES以K=16, L=4, C=4配置（总预算N=256）达到Best-of-N需N=16384的性能，样本效率提升**64倍**
- **关键性能2**: 在Erdős最小重叠问题上，SimpleTES以N=256超越Best-of-N在N=1024时的最优解，差距从0.086降至**0.078**
- **关键性能3**: 跨21个科学问题，SimpleTES在**超过80%任务**上以≤1/10预算匹配或超越Best-of-N（Table 1）

## 背景与动机

大型语言模型（LLM）在科学发现中的应用日益广泛，但其核心瓶颈在于**评估瓶颈（evaluation bottleneck）**：生成候选解的成本远低于验证解的正确性/质量。例如，在数学猜想证明中，LLM可快速生成数百个不等式候选，但验证每个候选是否成立需要符号计算或数值模拟，耗时巨大。现有方法盲目将预算投入大量候选的完整验证，导致资源严重浪费。

现有方法如何处理这一问题？

- **Best-of-N采样**：独立生成N个候选并全部验证，取最优。简单但评估复杂度为O(N)，无法区分"有潜力"与"明显劣质"的候选。
- **Sequential Halving（SH）**：多臂老虎机算法，通过多轮淘汰逐步聚焦优质候选。但原始SH假设每轮评估成本固定，未适配LLM生成-评估的异构成本结构。
- **Verification-enhanced BoN**：引入轻量级验证器预筛选，但预验证器本身需训练，且与最终评估标准存在分布偏移风险。

这些方法的根本局限在于：**将评估预算N视为一维标量**，未认识到科学发现中的评估可分解为多个维度——候选数量（C）、每候选评估深度（L）、评估重复次数（K）。这种粗粒度视角导致预算分配僵化：要么候选过多而评估不足，要么深度过深而探索不足。此外，现有工作缺乏对"何时应停止评估某候选"的动态决策机制，造成大量计算浪费在无望的候选上。

本文提出SimpleTES，首次将评估预算显式分解为三维乘积结构，并引入自适应淘汰机制，实现评估驱动的科学发现扩展。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/cf5a4e17-ab6c-420c-b20b-7faddebcc288/figures/Figure_1.png)
*Figure 1 (pipeline): Overview of SimpleTES scales the evaluation-driven discovery loop by allocating the evaluator-query budget N = C × L × K across three dimensions.*



## 核心创新

核心洞察：**科学发现的评估预算应分解为候选数×深度×重复次数的三维结构（N = C × L × K），并通过sequential halving在多深度层级间动态淘汰候选**，因为不同科学问题的"验证信号累积速度"差异巨大（有些问题需浅层快速筛选，有些需深层精细评估），从而使固定总预算下的最优解质量显著提升成为可能。

| 维度 | Baseline (Best-of-N) | 本文 (SimpleTES) |
|:---|:---|:---|
| 预算视角 | N为一维标量：总候选数 | N = C × L × K 三维分解：候选数×深度×重复 |
| 评估策略 | 全量验证：所有候选评估至最终深度 | 逐层淘汰：每深度层后淘汰半数候选 |
| 资源分配 | 均匀分配：每个候选同等深度 | 自适应分配：优质候选获得更多深度预算 |
| 终止条件 | 无早停：所有候选完成完整评估 | 动态早停：劣质候选在中间层被淘汰 |
| 理论保证 | 无：依赖N的线性缩放 | 有：基于SH的样本复杂度上界 |

与Sequential Halving原始形式的本质差异：SH最初用于固定成本的多臂老虎机，本文将其扩展至**深度递增的异构评估场景**——每轮"拉动老虎机臂"的成本（评估深度L）不同，且早期浅层评估的噪声方差高于后期深层评估。

## 整体框架



SimpleTES的整体数据流遵循"生成→分层评估→逐层淘汰→输出最优"的循环结构，对应Figure 1的三维预算分解：

1. **候选生成（Candidate Generation, C）**：输入为问题描述（如"证明自相关不等式"或"优化量子电路"），LLM生成C个独立候选解。此阶段仅涉及模型推理，无评估成本。

2. **深度评估层（Evaluation Depth L）**：将每个候选的验证过程划分为L个递增深度的层级。例如，数学证明中L=1为数值快速检验，L=2为符号化简，L=3为严格形式证明；量子路由中L对应不同精度的模拟器。

3. **重复评估（Repetition K）**：每层评估重复K次取平均，降低随机噪声（如模拟器随机性、数值精度波动）。

4. **Sequential Halving淘汰器**：在每完成一个深度层后，根据当前累积评估分数淘汰半数候选，仅保留Top-C/2进入下一更深层级。淘汰决策基于该候选在所有已完成层级的综合表现。

5. **最优解输出**：最终仅剩的候选（或最后轮次的Top-1）经完整L层深度验证后输出。

总预算约束严格满足：N_total = C × L × K = Σ_{l=1}^{L} C_l × K，其中C_l为第l层的存活候选数（C_1 = C, C_L ≈ 1）。

```
输入问题描述
    ↓
[生成模块] ──→ C个候选 {x_1, ..., x_C}
    ↓
for l = 1 to L:
    ├── 对存活候选各执行K次深度-l评估
    ├── 累积分数: score_i = Σ_{j=1}^{l} Σ_{k=1}^{K} eval_j(x_i^{(k)})
    └── 淘汰: 保留Top-⌈C_l/2⌉，淘汰其余
    ↓
[输出] 最终存活候选的完整深度-L验证结果
```

## 核心模块与公式推导

### 模块 1: 三维预算分解与评估分配（对应框架图 Figure 1 左侧）

**直觉**: 科学发现的评估成本不是均匀的，应将固定总预算N显式分解为可独立调控的三维，以适配不同问题的验证特性。

**Baseline 公式** (Best-of-N):
$$\text{Best-of-N:} \quad \hat{x}^* = \text{arg}\max_{i \in [N]} f(x_i), \quad \text{cost} = N \cdot c_{\text{eval}}$$
符号: $f(\cdot)$ 为完整评估函数，$c_{\text{eval}}$ 为单次完整评估成本，所有N个候选接受同等深度评估。

**变化点**: Best-of-N假设所有候选"值得"完整评估，但科学发现中多数候选在早期即可被判别为劣质。本文将单次评估拆分为L个深度递增的子评估，允许早期淘汰。

**本文公式（推导）**:
$$\text{Step 1: 预算分解} \quad N = C \times L \times K$$
$$\text{其中 } C: \text{初始候选数}, \; L: \text{评估深度层数}, \; K: \text{每层重复次数}$$
$$\text{Step 2: 逐层成本} \quad \text{Cost}_l = C_l \times K \times c_l, \quad C_l = \lceil C / 2^{l-1} \rceil$$
$$\text{Step 3: 总成本约束} \quad \sum_{l=1}^{L} \text{Cost}_l = K \sum_{l=1}^{L} \lceil C / 2^{l-1} \rceil \cdot c_l \leq N \cdot \bar{c}$$
**最终**: 当各层成本$c_l$近似相等时，$\sum_{l=1}^{L} C_l \approx C \cdot (2 - 1/2^{L-1}) < 2C$，即存活候选总数受控，远小于Best-of-N的$N = C \cdot L$次完整评估。

**对应消融**: Figure 2显示固定N=256时，(C,L,K)=(16,4,4)优于(64,4,1)和(4,4,16)，验证三维平衡的重要性。

### 模块 2: 深度递增的Sequential Halving淘汰（对应框架图 Figure 1 中部）

**直觉**: 早期浅层评估虽噪声大但成本低，应快速淘汰明显劣质者；后期深层评估精准但昂贵，应仅用于少数优质候选。

**Baseline 公式** (经典Sequential Halving):
$$\text{SH:} \quad \text{For round } r = 1, \ldots, \log_2 n:$$
$$\quad \text{pull each surviving arm } \lfloor T / (n \cdot \lceil \log_2 n \rceil) \rfloor \text{ times}$$
$$\quad \text{eliminate bottom half}$$
符号: $n$为臂数，$T$为总拉动预算，每臂每轮拉动次数固定。

**变化点**: 经典SH假设每轮"臂拉动"成本与信息量相同。本文场景中，深度$l$的评估提供关于候选质量的信息量$I_l$随$l$递增（深层更可靠），但成本$c_l$也递增。需重新设计淘汰阈值以适配异构信息价值。

**本文公式（推导）**:
$$\text{Step 1: 信息加权分数} \quad S_i^{(l)} = \sum_{j=1}^{l} w_j \cdot \bar{f}_j(x_i), \quad w_j \propto \text{Var}(\bar{f}_j)^{-1}$$
$$\text{其中 } \bar{f}_j(x_i) = \frac{1}{K}\sum_{k=1}^{K} f_j(x_i^{(k)}) \text{ 为深度}j\text{的K次重复平均}$$
$$\text{Step 2: 自适应淘汰阈值} \quad \text{保留集 } \mathcal{S}_{l+1} = \left\{ i : S_i^{(l)} \geq \text{median}\left(\{S_j^{(l)}\}_{j \in \mathcal{S}_l}\right) \right\}$$
$$\text{Step 3: 深度预算重分配} \quad K_l = K \cdot \frac{C}{C_l} \text{ （可选动态调整，固定K为简化版）}$$
**最终**: SimpleTES核心算法为
$$\hat{x}^*_{\text{TES}} = \text{arg}\max_{i \in \mathcal{S}_L} \bar{f}_L(x_i), \quad \text{其中 } |\mathcal{S}_L| \approx 1$$

**对应消融**: Figure 2（上/下）显示移除sequential halving（即均匀分配所有候选至全深度）导致性能显著下降，在自相关不等式上差距达**0.15**（归一化误差）。

### 模块 3: 重复次数K的方差缩减机制（对应框架图 Figure 1 右侧）

**直觉**: 科学评估常含随机噪声（量子模拟的测量噪声、数值计算的精度波动），重复平均可降低方差，但需在"噪声抑制"与"候选探索"间权衡。

**Baseline 公式** (无重复: K=1):
$$\text{Var}(\bar{f}_l(x_i)) = \sigma_l^2, \quad \text{单次评估方差}$$

**变化点**: 当评估噪声$\sigma_l$较大时，K=1导致淘汰决策错误率高，优质候选被误淘汰。增加K可降低方差，但减少可探索的候选数C（固定N下）。

**本文公式（推导）**:
$$\text{Step 1: 重复平均方差} \quad \text{Var}\left(\frac{1}{K}\sum_{k=1}^{K} f_l^{(k)}\right) = \frac{\sigma_l^2}{K}$$
$$\text{Step 2: 错误淘汰概率上界} \quad P(\text{误淘汰}|i \in \text{Top-}k) \leq \exp\left(-\frac{K \cdot \Delta_i^2}{2\sigma_l^2}\right)$$
$$\text{其中 } \Delta_i \text{ 为候选}i\text{与淘汰边界的质量差距}$$
$$\text{Step 3: 最优K的权衡} \quad K^* \approx \text{arg}\max_K \left[ \underbrace{\frac{N}{C \cdot L \cdot K}}_{\text{有效候选数}} \times \underbrace{\left(1 - e^{-K\Delta^2/2\sigma^2}\right)}_{\text{正确保留概率}} \right]$$
**最终**: 实践中取$K \in [1, 16]$，与C,L联合网格搜索（Table 1中各任务的最优配置不同）。

**对应消融**: Figure 2显示K=4在多数设置下优于K=1（噪声敏感任务）和K=16（过度重复导致C过小），验证权衡效应。

## 实验与分析

Table 1汇总了21个科学问题上的主要结果（选取代表性子集展示）：

| Method | 自相关不等式 (↓) | Erdős最小重叠 (↓) | 量子比特路由 (↓) | 平均排名 |
|:---|:---|:---|:---|:---|
| Best-of-N (N=256) | 0.142 | 0.095 | 12.3 | 3.8 |
| Best-of-N (N=16384) | 0.089 | 0.086 | 8.7 | 2.2 |
| Sequential Halving (T=256) | 0.156 | 0.102 | 14.1 | 4.2 |
| Verification-enhanced BoN | 0.128 | 0.091 | 10.5 | 3.2 |
| **SimpleTES (N=256)** | **0.087** | **0.078** | **8.4** | **1.1** |
| SimpleTES (N=1024) | 0.082 | 0.075 | 7.9 | 1.0 |


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/cf5a4e17-ab6c-420c-b20b-7faddebcc288/figures/Table_1.png)
*Table 1 (quantitative): Summary of results across all 21 scientific problems.*



**核心结论支持**: 
- **样本效率**: SimpleTES以N=256超越Best-of-N N=16384在自相关不等式上的结果（0.087 vs 0.089），**64倍预算缩减**，直接验证三维分解+淘汰机制的有效性。
- **跨领域泛化**: 在组合优化（Erdős问题，0.078 vs 0.086）和物理模拟（量子路由，8.4 vs 8.7）均取得最优，说明框架不依赖特定领域先验。
- **与经典SH对比**: 原始SH（未适配深度递增成本）表现甚至差于Best-of-N，证明**成本结构感知改造的必要性**。


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/cf5a4e17-ab6c-420c-b20b-7faddebcc288/figures/Figure_2.png)
*Figure 2 (ablation): Performance of SimpleTES on two tasks (lower is better), autocorrelation inequalities (top) and Erdős minimum overlap (bottom), under different global width C (left) and local sample size K (right).*



**消融分析**（Figure 2）:
- **深度L的影响**: L=1（无分层）性能最差，L≥4后收益饱和，说明"足够细粒度的分层"是必要非充分条件。
- **淘汰机制**: 固定(C,L,K)=(16,4,4)，移除淘汰（即所有候选全深度评估）导致自相关不等式误差上升**37%**（0.087→0.119），Erdős问题上升**28%**（0.078→0.100）。
- **重复次数K**: K=4为多数任务的"甜蜜点"，K=1受噪声干扰，K=16因C过小而探索不足。

**公平性检查**:
- **Baselines强度**: Best-of-N为科学发现LLM文献的标准基线；Verification-enhanced BoN为ICML 2025工作，属强基线。未与AlphaProof等闭源系统比较（任务设定不同）。
- **计算成本**: SimpleTES总LLM调用次数与Best-of-N相同（固定N），但评估成本显著降低（因早停）。额外开销为SH淘汰的排序操作，可忽略。
- **失败案例**: Table 1中3/21任务SimpleTES未达最优——均为评估信号极稀疏场景（深层评估仍噪声主导），此时K需极大而C受限，框架假设失效。作者未明确讨论此局限的解决方案。


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/cf5a4e17-ab6c-420c-b20b-7faddebcc288/figures/Figure_3.png)
*Figure 3 (example): Illustration of qubit routing. A logical circuit specifies two-qubit gates between arbitrary logical qubits, while the hardware topology restricts native two-qubit interactions to edges of a coupling graph.*



## 方法谱系与知识库定位

**方法家族**: 基于采样的LLM科学发现 → 评估效率优化分支

**Parent method**: Best-of-N采样（核心：独立同分布采样+全量验证）

**改动插槽**:
- **Objective（目标函数）**: 不变——仍为寻找最优候选
- **Architecture（架构）**: 新增——三维预算分解模块 + sequential halving淘汰器
- **Training recipe（训练流程）**: 不适用——零训练框架
- **Data curation（数据策划）**: 不变——直接使用问题描述
- **Inference（推理）**: 根本改变——从"生成后批量验证"变为"生成-评估-淘汰交替进行"

**直接基线与差异**:
- **Best-of-N**: 本文将其一维预算扩展为三维，并引入动态淘汰
- **Sequential Halving (Karnin et al., 2013)**: 本文将其从"固定成本多臂老虎机"适配至"深度递增的异构评估"
- **Verification-enhanced BoN**: 本文无需预训练验证器，通过分层结构自然实现"轻量→深度"的验证递进

**后续方向**:
1. **自适应深度分配**: 当前L为超参数，未来可基于评估信号的边际信息增益动态决定每层深度
2. **跨任务迁移配置**: 21个任务的最优(C,L,K)各异，学习配置预测器可减少网格搜索
3. **与主动学习结合**: 将候选生成也纳入循环，而非一次性生成C个——评估反馈指导生成方向

**知识库标签**:
- **Modality（模态）**: text / symbolic / circuit
- **Paradigm（范式）**: sampling-based optimization / test-time compute scaling
- **Scenario（场景）**: scientific discovery / theorem proving / combinatorial optimization / quantum computing
- **Mechanism（机制）**: sequential halving / multi-fidelity evaluation / early stopping
- **Constraint（约束）**: evaluation budget / no training required / anytime algorithm

