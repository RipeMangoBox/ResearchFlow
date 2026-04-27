---
title: A faster training algorithm for regression trees with linear leaves, and an analysis of its complexity
type: paper
paper_level: C
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- SMW加速的回归树线性叶子训练算法
- Modified TAO wit
- Modified TAO with Sherman-Morrison-Woodbury leaf optimization
- The modified TAO algorithm achieves
acceptance: Poster
method: Modified TAO with Sherman-Morrison-Woodbury leaf optimization
modalities:
- tabular
- structured data
paradigm: supervised
baselines:
- 最优分段线性回归树的动态规划方法_Optimal_Dynamic_
---

# A faster training algorithm for regression trees with linear leaves, and an analysis of its complexity

**Topics**: [[T__Time_Series_Forecasting]] | **Method**: [[M__Modified_TAO_with_Sherman-Morrison-Woodbury_leaf_optimization]] | **Datasets**: UCI regression datasets, Patched Fashion MNIST, Regression tree training speed, Synthetic and UCI regression datasets

> [!tip] 核心洞察
> The modified TAO algorithm achieves exactly the same results as standard TAO but with dramatically faster training by using the Sherman-Morrison-Woodbury formula to exploit the low-rank structure of leaf instance subsets, making deeper trees paradoxically faster to train and even asymptotically faster than ordinary linear regression.

| 中文题名 | SMW加速的回归树线性叶子训练算法 |
| 英文题名 | A faster training algorithm for regression trees with linear leaves, and an analysis of its complexity |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2501.0) · Code · Project |
| 主要任务 | 回归树训练（线性叶子）、时间序列预测 |
| 主要 baseline | TAO [5]、CART [3]、M5 [24]、ICML 2024 动态规划方法 [22] |

> [!abstract] 因为「TAO训练回归树时每个叶子需从头求解完整最小二乘问题，计算代价高」，作者在「TAO」基础上改了「用Sherman-Morrison-Woodbury秩k更新替代完整矩阵求逆」，在「UCI及Patched Fashion-MNIST等数据集」上取得「训练复杂度从O(D³+ND²)降至O(ND+D²)，且更深树反而更快」

- **关键性能**：单叶子更新复杂度从 O(D³ + ND²) 降至 O(ND + D²)，消除 D³ 项
- **关键性能**：超过临界深度后，总训练时间随深度增加而下降，渐近快于普通线性回归
- **关键性能**：产生与标准TAO「完全相同」的优化结果，属精确加速而非近似

## 背景与动机

训练带线性叶子的回归树时，核心瓶颈在于叶子模型的反复优化。以TAO（Tree Alternating Optimization）[5] 为例，该算法通过交替优化分裂节点和叶子模型来训练树：每次迭代中，先固定树结构优化各叶子的线性回归参数，再固定叶子参数优化分裂决策。然而，当输入维度 D 较高或树较深时，每个叶子需对路由到该叶子的 N_ℓ 个样本求解完整的最小二乘问题——即对 D×D 协方差矩阵求逆，代价为 O(D³ + N_ℓD²)。

现有方法如何处理这一问题？CART [3] 采用贪心逐层生长，叶子为常数预测，无法利用线性模型的表达能力；M5 [24] 虽支持线性叶子，但训练过程独立且未在迭代中复用计算；近期ICML 2024的动态规划方法 [22] 追求最优性，但复杂度更高、难以扩展。TAO本身是较先进的交替优化框架，但其叶子优化步骤「每次从零重新计算」，未利用关键观察：**相邻迭代间，同一叶子接收的样本集合仅有少量变化**。

这一忽视导致严重效率损失：当树深度增加时，叶子数指数增长，但每个叶子的样本数 N_ℓ 反而减少——理论上应更便宜，标准TAO却因重复的全矩阵求逆而更慢。作者由此提出核心问题：**能否将样本集合的增量变化转化为协方差矩阵的低秩更新，从而避免完整的矩阵求逆？** 本文正是通过Sherman-Morrison-Woodbury（SMW）矩阵求逆引理实现这一目标，并揭示了一个反直觉现象：足够深的树可以比浅树甚至普通线性回归训练得更快。

## 核心创新

**核心洞察**：叶子样本集合的迭代变化具有低秩结构，因为相邻两次TAO迭代中，仅有少量样本进出每个叶子；SMW公式可将这种变化转化为对逆协方差矩阵的秩k修正，从而将每次叶子优化的主要计算从「完整求逆」降为「小矩阵求逆」，使深度树的训练成本不再随深度单调增长。

| 维度 | Baseline (标准TAO) | 本文 |
|:---|:---|:---|
| 叶子优化方式 | 每次迭代从头求解完整最小二乘问题 | 基于SMW的增量秩k更新 |
| 协方差矩阵处理 | 重新计算 X_ℓ^⊤X_ℓ 并完整求逆 | 维护逆矩阵，仅修正进出样本带来的低秩变化 |
| 单叶子复杂度 | O(D³ + N_ℓD²) | O(N_ℓD + D²) 或 O(kD²)，k为变化样本数 |
| 深度扩展性 | 总时间随深度增加而增加 | 超过临界深度后，总时间随深度增加而**下降** |
| 优化结果等价性 | — | 与标准TAO**完全等价**（非近似） |

## 整体框架



改进后的TAO保持标准TAO的交替优化外层结构，仅替换叶子优化模块的计算机制。数据流如下：

1. **输入**：训练集 {(x_i, y_i)}，初始树结构（分裂节点决策规则 + 叶子线性模型参数 w_ℓ）
2. **分裂节点优化**（与标准TAO相同）：固定所有叶子模型，对每个分裂节点搜索最优决策边界，使实例被重新路由到不同子树
3. **实例路由**（与标准TAO相同）：根据当前树结构，将每个训练样本 x_i 从根节点递推至叶子，确定其所属叶子 ℓ
4. **叶子模型SMW更新**（核心创新）：对每个叶子 ℓ，接收「新增样本集 S_ℓ⁺」和「离开样本集 S_ℓ⁻」的信息，利用存储的逆协方差矩阵 Σ_ℓ^{-1} 通过SMW公式直接计算新逆矩阵，再更新线性参数 w_ℓ
5. **协方差矩阵存储与维护**：迭代间持久保存各叶子的逆协方差矩阵结构，作为下一步增量更新的基础
6. **收敛判断**：若叶子模型和分裂节点均稳定则停止，否则返回步骤2

```
迭代循环:
  ┌─→ 分裂节点优化 ──→ 实例路由 ──┐
  │                                │
  └─← 收敛判断 ←── 叶子SMW更新 ←──┘
         ↑_________↓
      (协方差矩阵存储)
```

关键不变量：步骤4的输出与标准TAO对该叶子求解完整最小二乘的结果**数值完全一致**，但计算路径完全不同。

## 核心模块与公式推导

### 模块 1: 标准叶子最小二乘与问题分解（对应框架图「叶子模型SMW更新」前置步骤）

**直觉**：将最小二乘解分解为协方差矩阵和交叉矩，为后续增量更新奠定代数结构。

**Baseline 公式** (标准TAO [5]): $$\mathbf{w}_\text{ell} = (\mathbf{X}_\text{ell}^\text{top} \mathbf{X}_\text{ell})^{-1} \mathbf{X}_\text{ell}^\text{top} \mathbf{y}_\text{ell}$$

符号: $\mathbf{X}_\text{ell} \in \mathbb{R}^{N_\text{ell} \times D}$ 为叶子 ℓ 的输入特征矩阵，$\mathbf{y}_\text{ell} \in \mathbb{R}^{N_\text{ell}}$ 为对应标签，$\mathbf{w}_\text{ell} \in \mathbb{R}^D$ 为待求线性参数。

**变化点**：标准TAO每次迭代重新计算 $(\mathbf{X}_\text{ell}^\text{top} \mathbf{X}_\text{ell})^{-1}$，未利用相邻迭代间 $\mathbf{X}_\text{ell}$ 仅少量变化的事实。

**本文公式（推导）**:
$$\text{Step 1}: \quad \mathbf{\Sigma}_\text{ell} = \mathbf{X}_\text{ell}^\text{top} \mathbf{X}_\text{ell}, \quad \mathbf{b}_\text{ell} = \mathbf{X}_\text{ell}^\text{top} \mathbf{y}_\text{ell} \quad \text{（分解协方差与交叉矩）}$$
$$\text{Step 2}: \quad \mathbf{w}_\text{ell} = \mathbf{\Sigma}_\text{ell}^{-1} \mathbf{b}_\text{ell} \quad \text{（标准形式，但不再每次重新求逆）}$$

### 模块 2: SMW秩k更新（对应框架图核心创新步骤）

**直觉**：样本进出叶子等价于对协方差矩阵施加低秩修正，SMW公式将大矩阵求逆转化为对小矩阵的求逆。

**Baseline 公式** (标准TAO重新计算):
$$\mathbf{\Sigma}_\text{ell}^{\text{new}} = \mathbf{X}_{\text{ell},\text{new}}^\text{top} \mathbf{X}_{\text{ell},\text{new}} \quad \text{(从零重新计算)}$$

**变化点**：当样本集变化较小时，$\mathbf{\Sigma}_\text{ell}^{\text{new}} - \mathbf{\Sigma}_\text{ell}^{\text{old}}$ 是秩k矩阵（k = |S_ℓ⁺| + |S_ℓ⁻| << D），直接重算浪费结构。

**本文公式（推导）**:
$$\text{Step 1}: \quad \mathbf{\Sigma}_\text{ell}^{\text{new}} = \mathbf{\Sigma}_\text{ell}^{\text{old}} + \sum_{i \in \mathcal{S}_\text{ell}^{+}} \mathbf{x}_i \mathbf{x}_i^\text{top} - \sum_{j \in \mathcal{S}_\text{ell}^{-}} \mathbf{x}_j \mathbf{x}_j^\text{top} \quad \text{（将变化表示为秩1矩阵和）}$$
$$\text{Step 2}: \quad \mathbf{\Sigma}_\text{ell}^{\text{new}} = \mathbf{\Sigma}_\text{ell}^{\text{old}} + [\mathbf{U}, \mathbf{V}] \begin{bmatrix} \mathbf{I} & \mathbf{0} \\ \mathbf{0} & -\mathbf{I} \end{bmatrix} [\mathbf{U}, \mathbf{V}]^\text{top} \quad \text{（统一为SMW适用的块低秩形式）}$$
其中 $\mathbf{U}$ 的列为新增样本 $\mathbf{x}_i$，$\mathbf{V}$ 的列为离开样本 $\mathbf{x}_j$。
$$\text{Step 3 (SMW应用)}: \quad (\mathbf{A} + \mathbf{U}\mathbf{C}\mathbf{V})^{-1} = \mathbf{A}^{-1} - \mathbf{A}^{-1}\mathbf{U}(\mathbf{C}^{-1} + \mathbf{V}\mathbf{A}^{-1}\mathbf{U})^{-1}\mathbf{V}\mathbf{A}^{-1}$$
代入 $\mathbf{A} = \mathbf{\Sigma}_\text{ell}^{\text{old}}$，将 D×D 求逆转化为对 2k×2k 矩阵求逆。
$$\text{最终}: \quad \mathbf{\Sigma}_\text{ell}^{\text{new},-1} = \mathbf{\Sigma}_\text{ell}^{\text{old},-1} - \mathbf{\Sigma}_\text{ell}^{\text{old},-1}[\mathbf{U}, \mathbf{V}] \left( \begin{bmatrix} \mathbf{I} & \mathbf{0} \\ \mathbf{0} & -\mathbf{I} \end{bmatrix}^{-1} + [\mathbf{U}, \mathbf{V}]^\text{top} \mathbf{\Sigma}_\text{ell}^{\text{old},-1} [\mathbf{U}, \mathbf{V}] \right)^{-1} [\mathbf{U}, \mathbf{V}]^\text{top} \mathbf{\Sigma}_\text{ell}^{\text{old},-1}$$

### 模块 3: 复杂度分析与临界深度现象（对应框架图整体训练流程）

**直觉**：深度增加使单叶子样本数 N_ℓ 指数下降，SMW更新的低固定开销使「更多但更便宜的叶子」总体更优。

**Baseline 复杂度** (标准TAO每叶子): $$O(D^3 + N_\text{ell} D^2)$$

**变化点**：标准TAO中 D³ 项与 N_ℓ 无关，深度增加时叶子数 2^d 增长而单叶子 N_ℓ ≈ N/2^d 下降，但 D³ 项不下降，总成本恶化。

**本文复杂度**:
$$\text{单叶子}: \quad O(N_\text{ell} D + D^2) \quad \text{或} \quad O(kD^2) \text{（当用k个秩1更新时）}$$
$$\text{全树}: \quad T_{\text{total}}(d) = \sum_{\text{ell}=1}^{2^d} O\left(\frac{N}{2^d} D + D^2\right) = O(ND + 2^d D^2)$$

**最终渐近行为**: $$T_{\text{total}}(d) \text{ decreases with } d \text{ for } d > d_{\text{critical}} \approx \log_2(N/D)$$

当 d 足够大时，2^d D² 项被 N_ℓD = (N/2^d)D 的快速下降所主导，总时间反而减少。Table 1 和 Figure 1 验证了这一理论预测的实际表现。

**对应消融**：本文未提供传统意义上的组件消融（因方法为精确等价加速，移除SMW即回退标准TAO），但Table 1通过不同深度下的时间对比间接验证了复杂度分析。

## 实验与分析


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/913243a4-a13f-44b8-9184-7b4a950806a4/figures/Table_1.png)
*Table 1 (result): Results for the Brodley Forest-MNIST*



本文在多个数据集上验证了改进TAO的训练加速效果。Table 1 展示了Brodley Forest-MNIST数据集上不同树深度进行20次迭代的平均训练时间，Table 2 和 Table 4 则聚焦于Patched Fashion-MNIST上的联合优化时间与超参数搜索时间。Figure 1 和 Figure 2 以可视化方式呈现了深度1树的自适应步长迭代加速，以及10次重复实验的平均训练时间随深度的变化趋势。

从Table 1可见，改进TAO在各深度下均实现显著加速，且加速比随深度增加而扩大——这与理论预测的「更深更快」一致。Table 2左侧报告了Patched Fashion-MNIST上的联合优化与局部模型训练时间，右侧则给出贝叶斯超参数优化时间；Table 4进一步扩展了该数据集以及CIFAR10上的对比，显示改进TAO在CAR（计算辅助回归）设置下对TAO及其加速版本均保持优势。Figure 2中训练时间随深度变化的曲线明确展示了超过某临界点后时间下降的趋势，验证了渐近分析的核心结论。


![Table 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/913243a4-a13f-44b8-9184-7b4a950806a4/figures/Table_2.png)
*Table 2 (result): Left: joint optimization and local model training times on the Packed-Fashion-MNIST dataset. Right: Bayesian hyperparameter optimization times for CART and our algorithm on CIFAR10.*



由于本文方法为精确加速（非近似），传统消融表不存在；但可通过与标准TAO的逐深度对比间接评估各组件贡献。关键观察是：去掉SMW更新（即回退标准TAO）后，训练时间从改进版的亚秒级或秒级回升至显著更高量级，且深度越大差距越悬殊。例如，在较深树设置下，标准TAO的O(D³)项成为绝对主导，而改进版通过消除该项使单叶子成本与N_ℓ成正比。

公平性检查：主要对比基线TAO [5]是同一研究组的直接前身工作，对比公平；M5 [24]和CART [3]作为经典方法属合理参照；ICML 2024动态规划方法 [22]是近期直接竞争者。未包含的潜在强基线包括GPU加速的TAO实现及其他近期树优化方法。实验在标准CPU设置下进行（Eigen [11]库），未报告GPU资源。作者明确披露的限制包括：限于回归树与线性叶子、分类树及其他叶子类型未探索；浅树时临界深度效应不明显；未讨论协方差矩阵结构的内存开销。

## 方法谱系与知识库定位

**方法族**：决策树交替优化（Alternating Optimization for Trees）

**父方法**：TAO (Tree Alternating Optimization) [5] — 本文直接扩展其叶子优化步骤，保持外层交替结构不变。

**改动槽位**：
- **training_recipe**：将叶子模型的「完整最小二乘求解」改为「SMW秩k增量更新」
- **credit_assignment**：将「独立 per-leaf 求解」改为「结构化增量计算，复用前次迭代的逆协方差矩阵」

**直接基线对比**：
- **TAO [5]**：父方法，本文与其结果完全等价但训练更快
- **CART [3]**：经典贪心算法，叶子为常数，无迭代优化结构
- **M5 [24]**：线性叶子回归树，但训练过程非交替优化且不增量复用计算
- **ICML 2024 DP方法 [22]**：追求最优性的动态规划，复杂度高、扩展性差

**后续方向**：
1. 将SMW增量机制扩展至分类树（softmax叶子）及其他叶子类型（如多项式、神经网络）
2. 结合GPU并行化进一步放大深度树的加速优势
3. 探索内存-计算权衡，优化协方差矩阵存储结构以支持更大规模数据

**标签**：tabular / supervised / regression_tree / alternating_optimization / exact_acceleration / low_rank_update / complexity_analysis / depth_scalability

## 引用网络

### 直接 baseline（本文基于）

- [[P__最优分段线性回归树的动态规划方法_Optimal_Dynamic_]] _(直接 baseline)_: Very recent ICML 2024 paper on same problem (optimal linear regression trees); d

