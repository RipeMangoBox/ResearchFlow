---
title: Fairness under Competition
type: paper
paper_level: A
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 竞争环境下的公平性悖论框架
- Fairness under C
- Fairness under Competition framework
- Individually fair classifiers under
acceptance: Poster
method: Fairness under Competition framework
modalities:
- Text
- tabular
- structured_data
paradigm: supervised
---

# Fairness under Competition

**Topics**: [[T__Fairness]] | **Method**: [[M__Fairness_under_Competition_framework]] | **Datasets**: Lending Club loan data - selected run from Exp. 3, Lending Club loan data - selected run from Experiment 3, Lending Club loan data, Lending Club loan data - selected run example, Lending Club loan data - Experiment 2: both logistic regressions, disjoint training data

> [!tip] 核心洞察
> Individually fair classifiers under competition fail to produce fair ecosystem outcomes, and enforcing standard fairness criteria can systematically worsen ecosystem fairness.

| 中文题名 | 竞争环境下的公平性悖论框架 |
| 英文题名 | Fairness under Competition |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2505.16291) · [Code](待补充) · [Project](待补充) |
| 主要任务 | 机器学习公平性（Fairness）、多智能体竞争公平性 |
| 主要 baseline | Individual fairness adjustment (Fairlearn EO/DP)、Standard classifier training without fairness constraints |

> [!abstract] 因为「多个企业各自部署满足个体公平性约束的分类器时，生态系统层面的公平性可能反而恶化」，作者在「Algorithmic monoculture and social welfare」基础上改了「将公平性分析从单分类器扩展到多智能体竞争场景，提出 EOC 指标」，在「Lending Club 贷款数据」上取得「100k 样本下仍有 15.8%-30.1% 概率出现公平性调整后 EOC 增加」

- **关键性能 1**：Experiment 1（logistic regression vs decision tree，相同训练数据）在 100k 样本时，公平性调整后 EOC 增加的概率为 30.1%（95% CI: [26.2, 34.0]）
- **关键性能 2**：Experiment 2（双 logistic regression，不相交训练数据）在 100k 样本时，该概率降至 15.8%（95% CI: [12.6, 19.0]），但悖论未消失
- **关键性能 3**：小样本（300）时 Experiment 1 的悖论概率高达 78.6%（95% CI: [75.0, 82.2]），随样本量增加单调下降但不收敛至零

## 背景与动机

在信贷、保险、招聘等场景中，多家企业往往同时部署机器学习分类器来筛选客户或候选人。每家企业可能都遵守了监管要求的个体公平性约束——例如，使用 Fairlearn 包确保不同种族群体的贷款批准率满足 Equal Opportunity（机会均等）。然而，当这些"公平"的分类器在市场中竞争同一批用户时，整个生态系统可能出现令人不安的悖论：某些群体反而系统性获得更差的结果。

现有方法如何处理这一问题？**Individual fairness adjustment（Fairlearn EO/DP）** [1, 15] 专注于单分类器的公平性约束，通过优化确保给定分类器满足 Equal Opportunity 或 Demographic Parity，但完全不考虑竞争环境。**Standard classifier training** 则完全忽略公平性，仅最大化个体效用。**Equality of opportunity in supervised learning** [20] 奠定了公平性的理论基础，但其框架本质上仍是单智能体的。

这些方法的共同短板在于：**它们将公平性视为单分类器的属性，而非系统涌现属性**。具体而言，当两家企业的分类器竞争同一借款人池时，个体公平性调整会改变分类器的决策边界和相关结构，从而可能放大而非缩小群体间的结果差异。例如，两家银行各自调整贷款批准率以满足 EO 约束后，可能导致某一群体在其中一家银行获得贷款的概率显著高于另一家——这种"选择性差异"正是生态系统不公平的来源。本文正是要揭示并量化这一被忽视的系统性风险。

## 核心创新

核心洞察：个体公平性调整在竞争环境中会产生**耦合外部性**，因为调整后的分类器决策边界会改变竞争者的策略空间和相关结构，从而使生态系统层面的公平性恶化成为结构性可能。

| 维度 | Baseline | 本文 |
|------|---------|------|
| 公平性定义 | Individual EO/DP：单分类器的条件概率相等 | EOC：多分类器间最大-最小选择概率之差 |
| 分析框架 | 单智能体优化 | 多智能体竞争均衡 |
| 核心结论 | 公平性调整改善个体分类器公平性 | 公平性调整可能系统性损害生态系统公平性 |
| 量化工具 | 无 | 基于 copula 的 EOC 分解（相关性 × 数据重叠度）|

## 整体框架



本文框架包含四个核心模块，数据流如下：

1. **Individual classifier training（个体分类器训练）**：输入为标注训练数据，输出为 base classifier（logistic regression 或 decision tree）。此模块为标准监督学习，无新颖性。

2. **Fairness adjustment (Fairlearn)（公平性调整）**：输入为 base classifier 和受保护属性，输出为满足 EO 约束的个体公平分类器。使用 Fairlearn 包实现，属标准后处理/约束优化方法。

3. **Competitive outcome simulation（竞争结果模拟）**：输入为两个经公平性调整的分类器及共享借款人池，输出为每个借款人的联合分配决策。此模块为本文新增，刻画两家企业竞争同一用户时的市场出清机制——通常假设借款人选择批准概率更高的企业，或企业按某种规则分配。

4. **EOC computation（EOC 计算）**：输入为联合分配决策和受保护属性，输出为生态系统公平性得分。此模块替代传统的个体 EO 评估，是本文的核心创新。

整体流程可概括为：
```
Labeled data → [c1, c2 训练] → [Fairlearn EO 调整] → [竞争模拟: S1∩S2] → EOC 评估
         ↑___________________________________________________________↓
                              （反馈：EOC 可能 > 调整前）
```

关键耦合点在于：Fairlearn 对 c1 的调整会改变其在竞争中的行为，进而影响 c2 的有效市场，最终通过 EOC 指标暴露出个体公平与系统公平的冲突。

## 核心模块与公式推导

### 模块 1: EOC（Ecosystem Opportunity Cost）指标定义（对应框架图模块 4）

**直觉**：传统公平性指标只问"一个分类器是否对不同群体一视同仁"，但竞争环境下应该问"不同群体在多个竞争分类器之间获得的最佳和最差结果差距有多大"。

**Baseline 公式** (Equal Opportunity [20]):
$$EO = \Pr[\hat{Y}=1|Y=1,A=0] - \Pr[\hat{Y}=1|Y=1,A=1]$$
符号: $\hat{Y}$ = 分类器预测, $Y$ = 真实标签, $A$ = 受保护属性。目标为使 EO = 0。

**变化点**：EO 仅衡量单分类器内部的条件概率差异，完全忽略多分类器竞争时的选择性效应。当两家企业竞争时，同一群体可能在企业 1 获得高批准率、在企业 2 获得低批准率，这种"跨企业差异"正是生态系统不公平的来源。

**本文公式（推导）**:
$$\text{Step 1}: \text{定义群体 } a \text{ 在企业 } i \text{ 的选择概率 } \Pr[Y=1|A=a, \text{selected by } i]$$
$$\text{Step 2}: \text{对每个群体取竞争中的极差} \max_{i\in\{1,2\}} - \min_{i\in\{1,2\}}$$
$$\text{最终}: EOC = \mathbb{E}_{a \in A}\left[\max_{i \in \{1,2\}} \Pr[Y=1|A=a, \text{selected by } i] - \min_{i \in \{1,2\}} \Pr[Y=1|A=a, \text{selected by } i]\right]$$

**对应消融**：Table 1 显示，当移除"竞争"维度（即回到单分类器 EO）时，无法检测到 15.8%-30.1% 的悖论案例。

### 模块 2: EOC 分解与悖论阈值（对应框架图理论分析部分）

**直觉**：EOC 并非不可分析的黑箱，它可以被解耦为分类器间的统计依赖性和数据重叠程度，从而预测何时个体公平性调整会"帮倒忙"。

**Baseline 公式**：无直接 baseline；传统公平性文献缺乏竞争环境下的分解工具。

**变化点**：需要引入 copula 理论 [28] 来建模两个分类器决策的联合分布，这是单分类器设置中完全不需要的工具。

**本文公式（推导）**:
$$\text{Step 1}: \text{用 copula 建模联合决策分布 } C(F_{c_1}(x), F_{c_2}(x))$$
$$\text{Step 2}: EOC = f(\rho_{c_1,c_2}, |S_1 \cap S_2|/|S_1 \cup S_2|) \quad \text{（相关性 + 数据重叠度）}$$
$$\text{Step 3}: \text{Fairlearn 调整后 } \tilde{c}_1, \tilde{c}_2 \Rightarrow \rho_{\tilde{c}_1,\tilde{c}_2} \text{ 改变}$$
$$\text{最终}: \Pr[\tilde{EOC} > EOC] \text{xrightarrow}{n \to \infty} p > 0 \quad \text{（悖论持续性）}$$

符号: $\rho_{c_1,c_2}$ = 分类器决策相关性, $S_i$ = 企业 $i$ 的训练数据, $p$ = 正极限概率。

**关键洞见**：即使训练样本 $n \to \infty$（一致性极限），悖论概率仍收敛于 $p > 0$。这与标准统计一致性形成鲜明对比——传统 EO 估计量随样本增加收敛至真实值，但 EOC 的恶化是竞争结构的**固有特征**，非有限样本效应。

**对应消融**：Experiment 1 vs Experiment 2/3 显示，将相同训练数据改为不相交数据（降低重叠度）使 100k 样本时的悖论概率从 30.1% 降至 15.8%，验证了重叠度项的作用；但即使重叠度降低，悖论仍未消失，说明相关性项 $\rho$ 的独立贡献。

## 实验与分析


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/6924dce4-a7e0-4f71-b043-95f38f4afdd3/figures/Table_1.png)
*Table 1 (result): 95%-CI for Hold-out EOC test (increased following binarization)*



本文在 Lending Club 贷款数据（2007-2015）[25] 上开展三组实验，每组 500 runs × 6 种训练样本量（300, 1k, 3k, 10k, 30k, 100k）。核心发现如下：

Table 1 呈现 95% 置信区间下的核心统计结果。在最大样本量 100k 时，Experiment 1（logistic regression vs decision tree，相同训练数据）显示公平性调整后 EOC 增加的概率为 30.1%（CI: [26.2, 34.0]）；Experiment 2（双 logistic regression，按贷款期限分割的不相交数据）降至 15.8%（CI: [12.6, 19.0]）；Experiment 3（logistic regression vs decision tree，不相交数据）为 17.4%（CI: [14.2, 20.6]）。这一 headline result 直接验证了核心主张：**个体公平性调整在竞争环境下有显著概率损害生态系统公平性，且该悖论不随样本量增加而消失**。

小样本效应尤为剧烈：Experiment 1 在 300 样本时悖论概率高达 78.6%（CI: [75.0, 82.2]），呈现清晰的单调下降趋势，但即使在 100k 样本的"大数据" regime 仍未收敛至零。作者还报告了一个具体案例：某次运行中 EOC 从调整前的 0.020774 飙升至 0.444052，相对增幅约 20 倍，说明效应量可以极为显著。



消融分析揭示两个关键规律。其一，**数据重叠度**是重要调节变量：相同数据（Exp 1, 30.1%）vs 不相交数据（Exp 2, 15.8%）的差异达 14.3 个百分点，验证 EOC 分解中的重叠度项。其二，**分类器异质性**影响较小：Exp 2（双 logistic, 15.8%）vs Exp 3（logistic + tree, 17.4%）仅差 1.6 个百分点，说明模型架构差异不是悖论的主要驱动因素。其三，**样本量**的调节作用最强：Exp 1 从 300 到 100k 下降 48.5 个百分点，但残余效应顽固存在。

公平性检查方面，baseline 选择合理：Fairlearn EO 调整是当前工业界标准实践，与无约束训练形成清晰对照。但存在局限：仅测试 Equal Opportunity 未覆盖 Demographic Parity 或 Equalized Odds；仅两企业竞争未扩展至多企业；仅 Lending Club 单一数据集且受保护属性为 mortgage status 代理变量而非真实种族/族裔数据；效应量细节仅见于 Tables 2,4,6,8,10,12 未在正文充分讨论；9000 次运行未校正多重比较。

## 方法谱系与知识库定位

本文属于 **Fairness in Machine Learning → Multi-Agent Fairness** 方法谱系，直接继承自 **Algorithmic monoculture and social welfare** [23] 和 **Improved Bayes risk can yield reduced social welfare under competition** [22] 的竞争分析传统，首次将公平性约束引入该框架。

**改变的 slots**：
- **objective**：个体 EO/DP → EOC（生态系统机会成本）
- **architecture**：单智能体 → 多智能体竞争耦合
- **reward_design**：个体效用最大化 → 竞争均衡下的福利分布分析
- **credit_assignment**：新增 EOC 分解（相关性 × 数据重叠度）

**直接 baselines 及差异**：
- **Bias-variance games** [16]：分析学习算法间的竞争，本文添加公平性约束层
- **Best response regression** [8]：战略学习框架，本文将其扩展至公平性调整后的响应动态
- **Fairness under composition** [14]：组合系统的公平性，本文聚焦竞争而非顺序组合

**后续方向**：(1) 扩展至 >2 家企业及更复杂的竞争结构（如 Stackelberg 博弈）；(2) 引入 Demographic Parity、Equalized Odds 等其他公平性准则的 EOC 分析；(3) 设计**生态系统最优**的联合公平性调整算法，而非仅诊断悖论。

**标签**：modality=tabular | paradigm=supervised + game-theoretic equilibrium | scenario=multi-agent competition (credit/lending) | mechanism=copula decomposition + paradox threshold | constraint=Equal Opportunity, sample size, two-firm limit

