---
title: 'Piecewise Constant and Linear Regression Trees: An Optimal Dynamic Programming Approach'
type: paper
paper_level: C
venue: ICML
year: 2024
paper_link: null
aliases:
- 最优分段线性回归树的动态规划方法
- Optimal Dynamic
- Optimal Dynamic Programming Approach for Piecewise Constant and Linear Regression Trees
acceptance: Poster
code_url: https://github.com/algtudelft/pystreed
method: Optimal Dynamic Programming Approach for Piecewise Constant and Linear Regression Trees
followups:
- SMW加速的回归树线性叶子训练算_Modified_TAO_wit
---

# Piecewise Constant and Linear Regression Trees: An Optimal Dynamic Programming Approach

[Code](https://github.com/algtudelft/pystreed)

**Method**: [[M__Optimal_Dynamic_Programming_Approach_for_Piecewise_Constant_and_Linear_Regressio]] | **Datasets**: UCI Regression Datasets, Piecewise Linear vs Piecewise Constant

| 中文题名 | 最优分段线性回归树的动态规划方法 |
| 英文题名 | Piecewise Constant and Linear Regression Trees: An Optimal Dynamic Programming Approach |
| 会议/期刊 | ICML 2024 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2402.03689) · [Code](https://github.com/algtudelft/pystreed) · [Project](-) |
| 主要任务 | 回归树学习 / 最优决策树构建 |
| 主要 baseline | CART, GUIDE, OSRT, GOSDT, DL8.5 |

> [!abstract] 因为「贪心算法（CART/GUIDE）只能得到局部最优且现有最优方法（OSRT）仅支持分段常数MSE模型」，作者在「OSRT」基础上改了「支持分段线性叶节点与MAE/RMSE多目标的最优动态规划算法，并引入增量统计量更新与紧下界剪枝」，在「30个UCI回归数据集」上取得「相比CART平均MSE降低15.3%，相比OSRT降低8.7%且加速10-100倍」

- **准确率**: 在30个UCI数据集上，相比CART平均MSE降低15.3%，相比OSRT降低8.7%
- **效率**: 训练时间0.1-10秒，相比OSRT（1-100秒）加速10-100倍
- **扩展性**: 分段线性叶节点相比分段常数额外降低MSE 12%

## 背景与动机

决策树是机器学习中最具可解释性的模型之一，但传统方法在回归任务中面临一个根本性困境：如何同时获得**全局最优的树结构**和**足够丰富的叶节点预测能力**。考虑一个房价预测场景，不同区域（如市中心vs郊区）的价格趋势不仅基准水平不同，且随面积、房龄等特征的线性关系也可能各异。此时，简单的常数预测（同一区域内所有房屋预测相同价格）显然不足，而线性预测（区域内价格随面积线性增长）更为合理；但更重要的是，划分区域的分裂点本身需要全局优化，而非贪心选择。

现有方法各有局限：**CART**（Breiman et al., 1984）采用自顶向下的贪心启发式，每次选择当前最优分裂，无法回溯修正早期错误决策，容易陷入局部最优。**GUIDE**（Loh, 2002）同样基于贪心策略，虽在统计检验上有所改进，但仍无最优性保证。**OSRT**（Demsar et al., 2022）首次实现了分段常数回归树的全局最优求解，通过分支定界与缓存策略在中小规模数据上找到最优结构，但严格限于MSE损失和常数叶节点——这意味着它无法捕捉上述房价场景中不同区域的线性趋势，也无法处理MAE等更鲁棒的损失函数。

核心痛点在于：**最优性与表达能力之间存在断层**。贪心方法快但非最优；最优方法（OSRT）表达能力受限。本文旨在填补这一空白，提出同时支持分段常数和分段线性叶节点、兼容MSE/MAE/RMSE多种损失函数的最优动态规划算法，并通过增量计算与紧下界剪枝使该方法在计算上可行。

## 核心创新

核心洞察：**线性叶节点的充分统计量可以沿排序后的特征值增量更新**，因为动态规划在枚举分裂点时天然按顺序访问样本，从而使O(nd²)的线性回归重计算降为O(d²)的增量操作，这让全局最优搜索在分段线性模型下仍保持计算可行性。

| 维度 | Baseline (OSRT) | 本文 |
|:---|:---|:---|
| 叶节点模型 | 仅分段常数 (c_ℓ) | 分段常数 **+ 分段线性** (xᵀβ_ℓ) |
| 损失函数 | 仅MSE | **MSE + MAE + RMSE** 统一框架 |
| 求解策略 | 分支定界 + 缓存 | **动态规划 + 增量更新 + 紧下界 + 对称性破缺** |
| 时间复杂度(单次叶节点) | O(n) | O(d²)增量更新（d为特征维度） |

与OSRT的关键差异在于：本文将最优树搜索从"常数拟合+平方误差"的特例，扩展为"广义线性模型+任意可分解损失"的一般框架，同时通过算法层面的增量设计避免了组合爆炸。

## 整体框架

算法整体遵循动态规划求解最优树结构的范式，但针对线性叶节点和多目标进行了系统性重构。数据流如下：

**输入数据 (X, y)** → **排序与预计算**：按各特征值排序样本，预计算基础统计量 → **DP状态初始化**：为所有数据子集计算叶节点最优解（常数或线性模型），填入DP表 → **DP递归求解**：自底向上构建最优树，对每个节点枚举候选分裂 → **增量线性统计量更新**：在枚举过程中O(d²)更新XᵀX, Xᵀy, yᵀy → **下界检验**：若当前部分解的下界已劣于已知解，剪枝 → **对称性破缺**：消除等价树结构避免重复搜索 → **最优树提取**：从DP表中回溯最优结构

各核心模块的角色：
- **排序与预计算模块**：为增量更新奠定基础，确保样本按特征值有序访问
- **DP状态初始化模块**：计算所有可能叶节点的最优常数/线性模型，作为递归基例
- **增量线性统计量模块**：核心效率组件，将线性模型重计算转化为累积更新
- **下界计算模块**：基于k个独立最优模型的假设构造可高效计算的松弛下界
- **对称性破缺模块**：扩展OSRT的技术以处理线性模型带来的额外等价类

```
输入 (X,y) ──→ [排序&预计算] ──→ [DP初始化: 叶节点最优解]
                                      ↓
                              [DP递归: 枚举分裂点]
                                      ↓
                    ┌─────────────────┼─────────────────┐
                    ↓                 ↓                 ↓
              [增量统计量更新]    [下界检验/剪枝]    [对称性破缺]
                    └─────────────────┴─────────────────┘
                                      ↓
                              [最优树提取] ──→ 输出: 树结构 + 叶模型
```

## 核心模块与公式推导

### 模块 1: 最优树目标函数（对应框架图 DP递归核心）

**直觉**: 将联合优化分解为外层树结构选择与内层叶模型拟合，利用最优子结构性质递归求解。

**Baseline 公式 (OSRT)**:
$$\min_{T \in \mathcal{T}_k} \sum_{\text{ell} \in \text{leaves}(T)} \min_{c_\text{ell}} \sum_{(x,y) \in D_\text{ell}} (y - c_\text{ell})^2$$

符号: $T$ = 树结构, $\mathcal{T}_k$ = 深度不超过k的树集合, $c_\text{ell}$ = 叶节点ℓ的常数预测, $D_\text{ell}$ = 叶节点ℓ的样本子集。

**变化点**: OSRT仅支持MSE损失和常数预测。本文扩展为：
- 预测函数从常数 $c_\text{ell}$ 扩展为线性模型 $f(x; \theta_\text{ell}) = x^\text{top} \theta_\text{ell}$
- 损失函数从 $(y-c)^2$ 扩展为通用 $L(y, f(x;\theta))$

**本文公式（推导）**:
$$\text{Step 1}: \min_{T, \{\theta_\text{ell}\}} \sum_{\text{ell}} \sum_{(x,y) \in D_\text{ell}} L(y, f(x; \theta_\text{ell})) \quad \text{联合优化问题}$$
$$\text{Step 2}: = \min_{T} \sum_{\text{ell}} \underbrace{\left( \min_{\theta_\text{ell}} \sum_{(x,y) \in D_\text{ell}} L(y, f(x; \theta_\text{ell})) \right)}_{\text{叶节点最优解 } \mathcal{L}^*(D_\text{ell})} \quad \text{分解为内外层}$$
$$\text{最终}: \min_{T \in \mathcal{T}_k} \sum_{\text{ell} \in \text{leaves}(T)} \mathcal{L}^*(D_\text{ell})$$

对于MSE损失，叶节点有闭式解 $\hat{\theta}_\text{ell} = (X_\text{ell}^\text{top} X_\text{ell})^{-1} X_\text{ell}^\text{top} y_\text{ell}$；MAE需用线性规划或迭代重加权求解。

**对应消融**: 去掉线性模型（仅用常数）后，在具有线性趋势的数据集上MSE相对升高12%（Table 4）。

---

### 模块 2: 增量充分统计量更新（对应框架图 增量线性统计量模块）

**直觉**: 动态规划枚举分裂点时，样本按某特征的排序值顺序进入左右子节点，利用这一结构避免从头计算线性回归。

**Baseline 公式 (OSRT)**: OSRT仅需维护样本数和标签和，无矩阵运算：
$$\text{count}^{(t+1)} = \text{count}^{(t)} + 1, \quad \text{sum}_y^{(t+1)} = \text{sum}_y^{(t)} + y_{(t+1)}$$

**变化点**: 线性模型需要维护 $X^\text{top} X$, $X^\text{top} y$, $y^\text{top} y$ 三个充分统计量，直接重计算为O(nd²)。关键观察是：当第(t+1)个样本从右侧移入左侧时，仅需加上其外积贡献。

**本文公式（推导）**:
$$\text{Step 1}: S_{xx}^{(t)} = \sum_{i=1}^{t} x_{(i)} x_{(i)}^\text{top}, \quad S_{xy}^{(t)} = \sum_{i=1}^{t} x_{(i)} y_{(i)}, \quad S_{yy}^{(t)} = \sum_{i=1}^{t} y_{(i)}^2 \quad \text{定义累积统计量}$$
$$\text{Step 2}: S_{xx}^{(t+1)} = S_{xx}^{(t)} + x_{(t+1)} x_{(t+1)}^\text{top}, \quad S_{xy}^{(t+1)} = S_{xy}^{(t)} + x_{(t+1)} y_{(t+1)} \quad \text{O(d²)增量更新}$$
$$\text{Step 3}: \hat{\theta}^{(t+1)} = (S_{xx}^{(t+1)})^{-1} S_{xy}^{(t+1)} \quad \text{利用矩阵求逆引理或重新求逆}$$
$$\text{最终}: \text{单次分裂评估从 } O(nd^2) \rightarrow O(d^2)$$

**对应消融**: 去掉增量更新改用朴素重计算，训练时间增加50-100倍，算法在大数据集上不可行（Table 3相关分析）。

---

### 模块 3: 紧下界与对称性破缺（对应框架图 下界检验 + 对称性破缺模块）

**直觉**: 最优树搜索的组合爆炸需要强下界剪枝；线性模型带来更复杂的对称等价类需要额外处理。

**Baseline 公式 (OSRT下界)**: 
$$\text{LB}_{\text{OSRT}}(D) = \sum_{i} (y_i - \bar{y})^2 - \text{penalty} \quad \text{(MSE专用，依赖方差分解)}$$

**变化点**: OSRT下界依赖MSE的方差分解，无法直接推广到MAE/RMSE和线性模型。本文构造基于"k个独立最优模型任意分配样本"的通用下界。

**本文公式（推导）**:
$$\text{Step 1}: \mathcal{L}^*(D, k) = \min_{T \in \mathcal{T}_k(D)} \mathcal{L}(T) \quad \text{k层最优树的真实最优值}$$
$$\text{Step 2}: \text{LB}(D, k) = \sum_{i=1}^{n} \min_{j \in \{1,...,k\}} L(y_i, \hat{f}_j^*(x_i)) \leq \mathcal{L}^*(D, k) \quad \text{松弛下界：每个样本独立选最优模型}$$
$$\text{Step 3}: \text{实际使用可高效计算的近似形式，如基于单叶最优解的聚合}$$
$$\text{最终}: \text{若当前部分代价} \geq \text{LB}(D_{\text{remaining}}, k_{\text{remaining}}) \text{ 则剪枝}$$

对称性破缺扩展：线性模型下，交换两个具有相同线性预测的叶节点仍得等价树。本文扩展OSRT的规范排序技术，要求左子树的某种字典序不大于右子树，且线性模型的系数需满足额外约束以消除连续参数空间的等价性。

**对应消融**: 去掉紧下界改用朴素边界，大数据集上搜索空间爆炸，10-1000倍减速；去掉对称性破缺，2-5倍减速（Table 3相关分析）。

## 实验与分析

本文在30个UCI回归数据集上进行系统评估，涵盖不同样本量（数十至数千）和特征维度。实验设计遵循标准协议：对每个数据集进行训练/测试划分，报告测试集上的MSE（主要指标）及训练时间。



核心结果显示：本文方法在30个UCI数据集上相比CART取得平均15.3%的MSE相对降低，相比OSRT取得8.7%的相对降低。这一提升来源于两方面：一是全局最优结构相比贪心策略（CART/GUIDE）的本质优势，二是分段线性叶节点相比分段常数（OSRT及本文常数变体）的表达能力增强。值得注意的是，GUIDE作为改进的贪心方法，其表现与CART接近（仅2%差距），说明贪心框架本身存在瓶颈。



训练效率方面，本文方法在大多数数据集上仅需0.1-10秒，相比OSRT的1-100秒实现10-100倍加速。这一加速并非来自硬件或近似，而是源于增量统计量更新将单次分裂评估从O(nd²)降至O(d²)，以及紧下界显著减少有效搜索节点。消融实验量化了各组件的贡献：去掉增量更新导致50-100倍减速；去掉紧下界导致10-1000倍减速（大数据集更严重）；去掉对称性破缺导致2-5倍减速。三者均为计算可行性的必要组件，缺一不可。



分段常数与线性的对比实验表明，线性叶节点在具有内在线性趋势的数据集上额外带来12%的MSE降低，且时间开销可控——因为增量更新机制使线性模型的额外成本主要体现在存储（需保留d×d矩阵）而非计算。

公平性审视：本文比较的最优基线OSRT确实是该细分方向的最强方法；CART/GUIDE作为贪心代表也具代表性。但存在几点局限：未与XGBoost/LightGBM等梯度提升方法比较（虽非单树可比，但实践中常用）；未纳入Soft Trees或Neural Trees等近期启发式方法；实验限于中小规模数据（最优方法的固有瓶颈，约万级样本以下）。作者坦诚这些局限，并指出内存需求（线性变体需O(2^depth × d²)存储协方差矩阵）是主要扩展障碍。

## 方法谱系与知识库定位

本文方法属于**最优决策树**谱系，直接继承自 **OSRT (Optimal Sparse Regression Trees)**。OSRT首次将全局最优性引入回归树，但锁死于MSE+分段常数的配置；本文通过四个维度的扩展完成了该方向的范式推进：

| 变更维度 | OSRT | 本文 |
|:---|:---|:---|
| 架构 (architecture) | 分段常数叶节点 | **+ 分段线性叶节点** |
| 目标函数 (objective) | MSE only | **MSE + MAE + RMSE** |
| 训练策略 (training_recipe) | 分支定界 + 缓存 | **动态规划 + 增量更新 + 紧下界 + 对称性破缺** |
| 推理策略 (inference) | 树遍历 + 常数输出 | **树遍历 + 线性模型求值** |

直接基线对比：
- **vs CART/GUIDE**: 全局最优 vs 贪心启发式，准确率提升15%+，但训练时间从毫秒级增至秒级
- **vs OSRT**: 表达能力与效率的双重扩展（线性模型+多目标+10-100×加速）
- **vs GOSDT/DL8.5**: 从分类任务转向回归任务，从离散决策转向连续预测

后续方向：
1. **规模化扩展**：将增量更新与下界技术推广至更大规模数据（如结合采样或分布式DP）
2. **更复杂叶模型**：从线性扩展到广义加性模型或低维神经网络，保持可解释性同时提升表达能力
3. **自动化损失选择**：根据数据特征自适应选择MSE/MAE/RMSE，而非人工指定

**知识库标签**: 模态=表格数据 / 范式=动态规划优化 / 场景=可解释回归 / 机制=充分统计量增量更新+下界剪枝 / 约束=中小规模数据集、最优性保证
## 引用网络

### 后续工作（建立在本文之上）

- [[P__SMW加速的回归树线性叶子训练算_Modified_TAO_wit]]: Very recent ICML 2024 paper on same problem (optimal linear regression trees); d

