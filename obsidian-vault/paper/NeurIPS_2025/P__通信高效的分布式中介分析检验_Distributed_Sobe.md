---
title: Distributed mediation analysis with communication efficiency
type: paper
paper_level: A
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 通信高效的分布式中介分析检验
- Distributed Sobe
- Distributed Sobel's test and Distributed MaxP test
- Distributed versions of Sobel's tes
acceptance: Poster
method: Distributed Sobel's test and Distributed MaxP test
modalities:
- tabular
paradigm:
- supervised
- unsupervised
- none (statistical inference method)
---

# Distributed mediation analysis with communication efficiency

**Topics**: [[T__Benchmark_-_Evaluation]], [[T__Time_Series_Forecasting]] | **Method**: [[M__Distributed_Sobels_test_and_Distributed_MaxP_test]] | **Datasets**: Educational mediation analysis: Gaokao mathematics → Calculus → Probability and Mathematical Statistics, Educational mediation analysis: combined distributed tests, Real-world educational data mediation analysis, Educational mediation analysis, Educational data mediation analysis

> [!tip] 核心洞察
> Distributed versions of Sobel's test and MaxP test can achieve nearly identical statistical power to global pooled-data tests while requiring only communication-efficient sharing of local test statistics rather than raw data.

| 中文题名 | 通信高效的分布式中介分析检验 |
| 英文题名 | Distributed mediation analysis with communication efficiency |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2025.xxxxx) · Code · Project |
| 主要任务 | 分布式环境下的因果中介假设检验（Mediation Analysis） |
| 主要 baseline | Sobel's test, MaxP test, Global pooled test |

> [!abstract] 因为「数据分散在多机且隐私限制无法汇聚原始数据」，作者在「Sobel's test 和 MaxP test」基础上改了「用方差加权聚合本地统计量替代全局原始数据计算，以 Bonferroni 校正聚合本地 p 值」，在「教育数据（高考数学→微积分→概率统计）」上取得「Dis12 MaxP p=0.0476 < 0.05，检测到单独 Class 1 (p=0.106) 和 Class 2 (p=0.6409) 均无法发现的 mediation effect」

- **关键性能 1**：Dis12 MaxP 在 α=0.05 水平检测到 mediation effect（p=0.0476），而 Class 1 单独 Sobel 检验 p=0.106、Class 2 单独 Sobel 检验 p=0.6409 均不显著
- **关键性能 2**：Dis123 MaxP 组合全部三类数据后 p=0.0007，Dis123 Sobel p=0.0032，显著性最强
- **关键性能 3**：理论保证分布式检验与全局合并检验的渐近功效等价，且与机器数量 K 无关

## 背景与动机

在因果推断中，中介分析（Mediation Analysis）旨在判断自变量 X 对因变量 Y 的影响是否通过中介变量 M 传递。例如，研究者想知道「高考数学成绩」是否通过「微积分能力」影响「概率统计成绩」——这里的微积分就是中介变量。传统方法如 Sobel's test 和 MaxP test 需要将所有数据汇聚到单台机器上计算路径系数 a（X→M）和 b（M→Y），这在医疗、教育等场景中不可行：医院隐私法规禁止患者数据出域，学校教务系统也无法跨校区汇聚原始成绩。

现有分布式统计推断方法 [6, 12, 16, 19] 虽能解决均值估计、U-统计量等问题，但均未针对中介分析的特殊结构——间接效应检验涉及 a×b 的乘积形式及其方差传播——进行设计。直接将分布式均值估计套用会导致：
- **Sobel's test**：需要全局估计 â 和 b̂ 的协方差结构，本地分块估计后无法简单拼接
- **MaxP test**：max(|Z_a|, |Z_b|) 的极值分布在各机独立时产生复杂的联合分布，需校正
- **隐私与通信**：原始数据汇聚违反隐私，而仅传回归系数会损失跨机交互信息



因此，本文提出专门针对中介分析检验统计量的分布式聚合方案，在只传输本地检验统计量（非原始数据）的前提下，实现与全局合并检验近乎一致的功效。

## 核心创新

核心洞察：中介检验的本地统计量具有可聚合的渐近正态结构，因为 Sobel 统计量和 MaxP 统计量的分布仅依赖各机可独立估计的均值与方差，从而使方差倒数加权求和（Sobel）与 Bonferroni 型极值校正（MaxP）成为通信最优的充分统计量聚合方式。

| 维度 | Baseline（Sobel's / MaxP） | 本文 |
|:---|:---|:---|
| 数据流 | 原始数据汇聚至单机后统一计算 | 各机本地计算，仅传输检验统计量 |
| Sobel 聚合 | 全局 â·b̂ / √(â²σ̂_b² + b̂²σ̂_a²) | 方差倒数加权 ∑w_k·T_Sobel^(k) |
| MaxP 聚合 | 全局 max(\|Z_a\|, \|Z_b\|) 的精确分布 | K 机最小 p 值的 Bonferroni 校正 |
| 隐私保护 | 无（需原始数据） | 原始数据不出域 |
| 功效保证 | 全局最优（理论金标准） | 渐近等价于全局，与 K 无关 |

## 整体框架


![Figure 1, Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/723efd83-185e-4461-9d1a-05fb300be9a1/figures/Figure_1,_Figure_2.png)
*Figure 1, Figure 2 (pipeline): Figure 1: The causal graph with exposure X, mediator M, and outcome Y. Figure 2: A distributed framework with K worker machines...*



系统采用 Coordinator-Worker 架构，共 K 台 worker 机器各持本地数据分块，中央 coordinator 负责全局决策：

1. **本地 Sobel 统计量计算**（Worker k）：输入本地数据 D_k，输出 T_Sobel^(k) = â^(k)·b̂^(k) / √((â^(k))²σ̂_b² + (b̂^(k))²σ̂_a²) 及其方差估计 Var(T_Sobel^(k))
2. **本地 MaxP 统计量计算**（Worker k）：输入本地数据 D_k，输出 T_MaxP^(k) = max(\|Z_a^(k)\|, \|Z_b^(k)\|) 对应的本地 p 值 p_MaxP^(k)
3. **全局 Sobel 聚合**（Coordinator）：接收 K 个 (T_Sobel^(k), Var(T_Sobel^(k)))，按方差倒数加权求和得到 T_DSobel
4. **全局 MaxP 聚合**（Coordinator）：接收 K 个 p_MaxP^(k)，取最小值后乘以 K 做 Bonferroni 校正得 p_DMaxP
5. **假设检验决策**（Coordinator）：比较聚合统计量与临界值，输出全局拒绝/接受决策

```
[D_1] ──T_Sobel^(1), Var^(1)──┐
[D_2] ──T_Sobel^(2), Var^(2)──┼──→ Coordinator ──T_DSobel──→ Decision
  ...                          │
[D_K] ──T_Sobel^(K), Var^(K)──┘

[D_1] ──p_MaxP^(1)────────────┐
[D_2] ──p_MaxP^(2)────────────┼──→ Coordinator ──p_DMaxP──→ Decision
  ...                          │
[D_K] ──p_MaxP^(K)────────────┘
```

## 核心模块与公式推导

### 模块 1: 本地 Sobel 统计量与方差加权聚合（对应框架图步骤 1→3）

**直觉**：Sobel 检验的间接效应估计 â·b̂ 在各机独立同分布假设下具有可加性，最优线性组合应按精度（方差倒数）分配权重。

**Baseline 公式** (Sobel's test): $$T_{Sobel}^{global} = \frac{\hat{a}^{global} \hat{b}^{global}}{\sqrt{(\hat{a}^{global})^2 \hat{\sigma}_{b^{global}}^2 + (\hat{b}^{global})^2 \hat{\sigma}_{a^{global}}^2}}$$
符号: $\hat{a}^{global}$, $\hat{b}^{global}$ = 全局合并数据估计的路径系数; $\hat{\sigma}^2$ = 对应方差估计

**变化点**：全局估计需要原始数据汇聚；当数据分 K 块时，直接平均本地 â^(k)·b̂^(k) 会因各机样本量不同、方差异质而损失效率。

**本文公式（推导）**:
$$\text{Step 1}: T_{Sobel}^{(k)} = \frac{\hat{a}^{(k)} \hat{b}^{(k)}}{\sqrt{(\hat{a}^{(k)})^2 \hat{\sigma}_{b^{(k)}}^2 + (\hat{b}^{(k)})^2 \hat{\sigma}_{a^{(k)}}^2}} \quad \text{各机独立计算标准 Sobel 统计量}$$
$$\text{Step 2}: w_k = \frac{[Var(T_{Sobel}^{(k)})]^{-1}}{\sum_{j=1}^{K}[Var(T_{Sobel}^{(j)})]^{-1}} \quad \text{方差倒数归一化权重，保证 } \sum_k w_k = 1$$
$$\text{最终}: T_{DSobel} = \sum_{k=1}^{K} w_k \cdot T_{Sobel}^{(k)}$$

**对应消融**：Table 1 显示 Dis12 Sobel（两类组合）p=0.0732 > 0.05 未过显著性，说明方差加权虽最优但 Sobel 检验本身对弱中介敏感；改用 MaxP 后 Dis12 MaxP p=0.0476 成功检测。

### 模块 2: 本地 MaxP 统计量与 Bonferroni 型极值聚合（对应框架图步骤 2→4）

**直觉**：MaxP 取 max(|Z_a|, |Z_b|) 的极值特性使其对单路径弱、双路径均非零的情形更稳健；分布式环境下需控制 K 个独立本地检验的族错误率。

**Baseline 公式** (MaxP test): $$T_{MaxP}^{global} = \max\left\{\left|Z_a^{global}\right|, \left|Z_b^{global}\right|\right\}, \quad p_{MaxP}^{global} = \Pr(T_{MaxP}^{global} > c \text{mid} H_0)$$
符号: $Z_a = \hat{a}/\hat{\sigma}_a$, $Z_b = \hat{b}/\hat{\sigma}_b$ = 路径 a, b 的标准化 Z 统计量

**变化点**：全局 MaxP 的分布依赖于 (Z_a, Z_b) 的联合相关性结构；各机独立计算后，coordinator 无法重构该联合分布，需用保守但可控的 Bonferroni 型边界。

**本文公式（推导）**:
$$\text{Step 1}: T_{MaxP}^{(k)} = \max\left\{\left|Z_{a^{(k)}}\right|, \left|Z_{b^{(k)}}\right|\right\}, \quad p_{MaxP}^{(k)} = \Pr(T_{MaxP}^{(k)} > T_{obs}^{(k)} \text{mid} H_0) \quad \text{各机计算本地 p 值}$$
$$\text{Step 2}: \tilde{p} = \min_{k \in [K]} p_{MaxP}^{(k)} \quad \text{取 K 机最显著（最小）p 值}$$
$$\text{最终}: p_{DMaxP} = \min\left\{1, \; K \cdot \tilde{p}\right\} = \min\left\{1, \; K \cdot \min_{k \in [K]} p_{MaxP}^{(k)}\right\} \quad \text{Bonferroni 校正控制族错误率}$$

**对应消融**：Table 1 中 Dis12 MaxP p=0.0476 < 0.05 成功，而 Class 1 单独 MaxP p=0.0848、Class 2 单独 MaxP p=0.596 均失败；Dis123 MaxP p=0.0007 为所有组合中最强。

### 模块 3: 渐近功效等价性保证（理论核心）

**直觉**：当各机样本量足够大时，本地统计量的加权平均依概率收敛于全局统计量，加权方案的方差最优性保证功效损失趋于零。

**本文公式**:
$$\lim_{n_k \to \infty} \Pr(\text{Reject } H_0 \text{mid} H_1 \text{ true})_{DSobel} = \lim_{n \to \infty} \Pr(\text{Reject } H_0 \text{mid} H_1 \text{ true})_{global}$$
该等价性对任意有限 K 成立，即分布式检验的功效与机器数量无关，仅依赖总样本量 n = Σn_k。

## 实验与分析


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/723efd83-185e-4461-9d1a-05fb300be9a1/figures/Table_1.png)
*Table 1 (quantitative): The p-values of the tests. We observe that for a given significance level of 0.05...*



本文在模拟实验和真实教育数据上验证方法。模拟部分考察了八种检验在三种零假设、七种 K 选择下的经验 size（Figure 3, Figure 5）与经验功效（Figure 4, Figure 6），以及在异质设置下的稳健性（Figure 7）；真实数据应用则来自三个班级的高考数学→微积分→概率统计中介分析。



Table 1 展示了核心实证结果。在教育数据上，分布式 MaxP 检验展现出关键优势：Dis12 MaxP（合并 Class 1 和 Class 2）p=0.0476，在 α=0.05 水平显著，成功检测到 mediation effect；而 Class 1 单独 MaxP p=0.0848、Class 2 单独 MaxP p=0.596 均不显著，Class 1 单独 Sobel p=0.106、Class 2 单独 Sobel p=0.6409 同样失败。这一结果表明，分布式聚合能够发现单个站点因样本量不足或效应微弱而遗漏的真实中介效应。值得注意的是，Dis12 Sobel p=0.0732 > 0.05 未达显著，说明 MaxP 检验在此场景下比 Sobel 检验更稳健——这与 MaxP 对 a、b 路径非对称显著的设计优势一致。当合并全部三类数据时，Dis123 MaxP p=0.0007、Dis123 Sobel p=0.0032，显著性大幅提升，验证了增加有效样本量的价值。


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/723efd83-185e-4461-9d1a-05fb300be9a1/figures/Figure_3.png)
*Figure 3 (result): the empirical sizes of the eight tests under three null hypotheses and seven choices of K, with the total sample size n = 16K or 27*



Figure 3 和 Figure 4 显示在 K 从 2 增至 27、总样本量 n=16K 或 27 的设定下，分布式检验的经验 size 控制在名义水平附近，经验功效随 K 增加保持稳定，支持「与机器数量无关」的理论论断。Figure 7 进一步在异质设置（各机分布参数不同）下验证，分布式 MaxP 功效衰减可控。

公平性检查：本文未与引用的高维中介方法 [7, 8, 17, 23] 进行系统对比实验，真实数据集仅含 3 个班级且结果部分存在截断；未报告实际通信比特节省量，也未在模拟中变化 worker 数量 K 来直接验证「独立 of K」的声明。此外，缺乏与联邦学习、安全多方计算等隐私保护基线的比较。

## 方法谱系与知识库定位

本文属于**分布式统计推断**方法族，直接父方法为 **Sobel's test** [25] 和 **MaxP test** [23] 用于中介分析，同时吸收 **Fan et al. [12]** 的通信高效估计思想与 **Jordan et al. [19]** 的分布式推断框架作为方法论基础。

**改动槽位**：
- **inference_strategy**：以方差加权聚合（Sobel）和 Bonferroni 校正极值聚合（MaxP）替代全局原始数据计算
- **data_pipeline**：以本地检验统计量传输替代原始数据汇聚
- **architecture**：新增 Coordinator-Worker 分布式架构

**直接基线差异**：
- vs. **Sobel's test** [25]：保持乘积系数检验逻辑，但将全局估计替换为本地统计量的最优线性组合
- vs. **MaxP test** [23]：保持极值检验逻辑，但将精确分布计算替换为跨机 Bonferroni 边界
- vs. **Global pooled test**：牺牲有限样本效率换取零原始数据传输，渐近无损失
- vs. **AMDP [8]** / **high-dim mediation [7, 17]**：本文专注低维单中介的分布式扩展，未涉及高维 FDR 控制

**后续方向**：(1) 扩展至高维多中介场景，结合 [7, 8, 17] 的稀疏检验工具；(2) 引入差分隐私或安全聚合，强化形式化隐私保证；(3) 针对异质数据设计自适应权重，替代当前同方差假设下的逆方差加权。

**标签**：modality=tabular | paradigm=distributed statistical inference | scenario=privacy-preserving causal analysis | mechanism=variance-weighted aggregation / Bonferroni-corrected p-value combination | constraint=communication-efficient / no raw data sharing

