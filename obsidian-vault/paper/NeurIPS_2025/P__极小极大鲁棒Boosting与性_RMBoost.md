---
title: Robust Minimax Boosting with Performance Guarantees
type: paper
paper_level: A
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 极小极大鲁棒Boosting与性能保证
- RMBoost
- RMBoost minimizes worst-case error
acceptance: Poster
method: RMBoost
modalities:
- tabular
paradigm: supervised
---

# Robust Minimax Boosting with Performance Guarantees

**Topics**: [[T__Classification]] | **Method**: [[M__RMBoost]] | **Datasets**: Large datasets: Susy, Higgs, Forest Covertype, Large datasets, Small datasets, Running time, Large datasets: Susy

> [!tip] 核心洞察
> RMBoost minimizes worst-case error probabilities to achieve robustness to general types of label noise while providing finite-sample performance guarantees and maintaining strong classification accuracy.

| 中文题名 | 极小极大鲁棒Boosting与性能保证 |
| 英文题名 | Robust Minimax Boosting with Performance Guarantees |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2510.13445) · [Code](待补充) · [Project](待补充) |
| 主要任务 | Text Classification, 鲁棒提升 (robust boosting) |
| 主要 baseline | AdaBoost, LPBoost, LogitBoost, BrownBoost, RobustBoost, Robust-GBDT, XGB-Quad |

> [!abstract] 因为「传统Boosting方法在标签噪声下性能严重退化，且现有鲁棒Boosting缺乏对一般噪声类型的有限样本理论保证」，作者在「AdaBoost/LPBoost」基础上改了「将经验凸代理损失替换为极小极大最坏情况误差概率优化，并引入正则化参数λ控制准确率-鲁棒性权衡」，在「UCI数据集（含Susy/Higgs/Forest Covertype）」上取得「噪声环境下平均排名1.71（10%噪声）至3.88（10%对抗噪声），显著优于AdaBoost等基线」

- **关键性能**：RMBoost在10%标签噪声下平均排名1.71，AdaBoost为2.38–16.37；在20%对抗噪声下RMBoost平均排名10.82，AdaBoost性能退化至40±1.9%–50±2.2%误差
- **关键性能**：大规模数据集Susy上，RMBoost在20%对抗噪声下保持38±2.5%误差，AdaBoost恶化至40±1.9%，XGB-Quad恶化至39±2.0%
- **关键性能**：运行时间与LPBoost相当，显著优于理论最坏情况，常规桌面机秒级完成

## 背景与动机

Boosting是集成学习的核心范式，通过顺序组合弱分类器获得强预测能力。然而，一个长期存在的痛点是：当训练标签存在噪声时，传统Boosting方法会出现严重的性能退化。例如，在医疗诊断或文本分类中，标注错误难以避免——10%的标签翻转即可让AdaBoost的决策边界严重偏移，因为指数损失会指数级放大错分样本的权重，导致模型过拟合噪声。

现有方法如何应对这一问题？**AdaBoost** [2] 通过最小化指数损失的经验平均来贪婪优化基分类器权重，但对噪声极度敏感——Long & Servedio [10] 证明随机分类噪声即可击败所有凸潜力Boosting器。**LPBoost** [25] 采用线性规划进行列生成，每轮求解优化问题，虽比AdaBoost稳定但仍基于经验风险最小化，缺乏对一般噪声类型的理论保证。**BrownBoost** [12] 等鲁棒变体通过模拟布朗运动或修改损失函数来抵抗噪声，但理论分析局限于特定噪声分布（如对称噪声），且无法提供有限样本下的性能边界。

这些方法的共同瓶颈在于：**优化目标都是经验平均的凸代理损失**，而非直接控制最坏情况下的误分类概率。这导致两个后果：一是对非对称、对抗性等一般噪声类型缺乏鲁棒性；二是理论保证要么依赖渐近分析，要么仅针对特定噪声模型，无法回答"给定n个样本，噪声水平ε，我的模型误差上界是多少"。

本文提出RMBoost，首次将Boosting重新建模为**极小极大优化问题**，直接最小化噪声分布最坏情况下的0-1误差概率，并给出有限样本性能保证。

## 核心创新

核心洞察：Boosting的鲁棒性瓶颈源于凸代理损失与0-1误差不匹配，因为经验风险最小化对分布偏移敏感，而极小极大优化直接针对最坏情况噪声分布的0-1误差，从而使一般类型标签噪声下的有限样本保证成为可能。

| 维度 | Baseline (AdaBoost/LPBoost) | 本文 (RMBoost) |
|:---|:---|:---|
| **优化目标** | 经验平均的凸代理损失（指数损失/对数损失） | 极小极大最坏情况0-1误差概率 |
| **噪声处理** | 无显式建模，或假设特定噪声类型 | 对噪声分布集合取最大，覆盖一般噪声 |
| **理论保证** | 渐近一致性或特定噪声模型 | 有限样本误差上界（含噪声项） |
| **准确率-鲁棒性权衡** | 无显式控制机制 | 正则化参数λ ~ 1/√n 可调 |
| **优化方式** | 坐标下降贪婪优化 | 分布鲁棒优化（Wasserstein DRO）|

## 整体框架



RMBoost保持Boosting的经典三段式结构，但将核心的权重优化模块替换为极小极大求解：

1. **基分类器生成（Base classifier generation）**：输入带噪声标签的训练数据，输出弱分类器池（如决策树桩）。此模块与标准Boosting相同，无修改。

2. **极小极大权重优化（Minimax weight optimization）**【核心创新】：输入弱分类器池和正则化参数λ，输出极小极大最优权重。该模块替代了AdaBoost的坐标下降，通过求解分布鲁棒优化问题，在最坏情况噪声分布下最小化加权组合的误分类概率。

3. **加权多数投票（Weighted majority vote）**：输入带极小极大最优权重的弱分类器，输出最终分类。推理结构与标准Boosting一致，但权重具有鲁棒性保证。

数据流：训练数据 (X, Ỹ) → 弱学习器生成 h₁,...,h_T → 极小极大优化 w* = argmin_w max_P 𝔼_P[𝟙(sign(∑wₜhₜ(X))≠Y)] + λ·Ω(w) → 加权投票 H(x) = sign(∑wₜ*hₜ(x))

```
┌─────────────────┐     ┌─────────────────────┐     ┌─────────────────┐
│  Training Data  │────→│  Weak Learners      │────→│ Minimax Weight  │
│  (X, Ỹ noisy)   │     │  h₁, h₂, ..., h_T   │     │ Optimization    │
└─────────────────┘     └─────────────────────┘     │  (novel module) │
                                                    └────────┬────────┘
                                                             ↓
                                                    ┌─────────────────┐
                                                    │ Weighted Majority │
                                                    │ Vote: H(x)      │
                                                    └─────────────────┘
```

## 核心模块与公式推导

### 模块 1: 极小极大误差概率目标（对应框架图「Minimax weight optimization」）

**直觉**：传统Boosting优化经验平均损失，但标签噪声使经验分布偏离真实分布；直接对噪声分布最坏情况取最大，可保证任意噪声模式下的性能。

**Baseline 公式** (AdaBoost): $$L_{\text{AdaBoost}} = \min_{h \in \mathcal{H}} \frac{1}{n} \sum_{i=1}^{n} \exp\left(-y_i h(x_i)\right)$$
符号: $h = \sum_t w_t h_t$ 为加权组合分类器, $y_i \in \{-1,+1\}$, $\phi(\cdot)$ 为凸代理函数（此处为指数函数）。

**变化点**：AdaBoost的指数损失是0-1损失的上界，但优化目标与真实误差不一致；且经验平均对分布扰动敏感。RMBoost直接针对0-1损失，并对噪声分布集合取极小极大。

**本文公式（推导）**:
$$\text{Step 1}: \min_{h \in \mathcal{H}} \max_{P \in \mathcal{P}} \mathbb{E}_P\left[\mathbf{1}\left(h(X) \neq Y\right)\right] \quad \text{将凸代理替换为0-1损失的极小极大}$$
$$\text{Step 2}: + \lambda \cdot \Omega(h) \quad \text{加入Wasserstein质量运输正则化以控制分布扰动半径}$$
$$\text{最终}: \min_{h \in \mathcal{H}} \max_{P \in \mathcal{P}_\lambda} \mathbb{E}_P\left[\mathbf{1}\left(h(X) \neq Y\right)\right]$$
其中 $\mathcal{P}_\lambda = \{P : W(P, P_n) \leq \lambda\}$ 为以经验分布 $P_n$ 为中心、Wasserstein半径 $\lambda$ 的分布球。

**对应消融**：Figure 6（λ的消融）显示较小λ增强正则化提升鲁棒性但降低干净数据准确率。

---

### 模块 2: 有限样本性能保证（对应框架图「理论输出」）

**直觉**：需要量化"给定n个样本，我的模型在噪声下多差"，这是现有鲁棒Boosting缺乏的。

**Baseline**：传统Boosting理论提供渐近一致性或间隔界，但无显式噪声依赖项：如AdaBoost的泛化界 $R(h) \leq \hat{R}(h) + O\left(\sqrt{\frac{T\log n}{n}}\right)$，不含噪声变量。

**变化点**：RMBoost的极小极大框架允许显式分解噪声误差为"干净误差 + 噪声影响项 + 正则化项"。

**本文公式（推导）**:
$$\text{Step 1}: R_{\text{noise}}(h) \leq R_{\text{clean}}(h) + \text{noise bias term} \quad \text{噪声误差分解为干净误差加偏差}$$
$$\text{Step 2}: + C \cdot \frac{\lambda + 1/\sqrt{n}}{\text{margin}(h)} \quad \text{加入由样本量和正则化控制的复杂度项}$$
$$\text{最终}: R_{\text{noise}}(h) \leq R_{\text{clean}}(h) + C \cdot \frac{\lambda + 1/\sqrt{n}}{\gamma}$$
其中 $R_{\text{noise}}$ 为带噪声风险, $R_{\text{clean}}$ 为无噪声贝叶斯风险, $\gamma$ 为分类间隔, $C$ 为常数。

**关键推论**：选择 $\lambda^* = 1/\sqrt{n}$ 时，正则化项与统计波动同阶，获得最优准确率-鲁棒性权衡。

**对应消融**：理论保证的紧密度受λ选择影响，Figure 6 验证λ ~ 1/√n 附近为经验最优区域。

## 实验与分析


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/25041ced-1171-4a7c-936f-f89b6d6237b0/figures/Table_1.png)
*Table 1 (comparison): Average classification error (in %) and standard deviation for RMBoost and other state-of-the-art methods over 10-fold cross-validation on UCI datasets.*



本文在UCI机器学习库的多类数据集上评估RMBoost，涵盖小规模数据集（Table 1/Table 3）和大规模数据集Susy、Higgs、Forest Covertype各5000样本（Table 5）。评估设置包括：无噪声、10%随机噪声、20%随机噪声、10%对抗噪声、20%对抗噪声五种场景。核心指标为平均分类误差（%）及标准差，基于10折交叉验证。


![Figure 1, Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/25041ced-1171-4a7c-936f-f89b6d6237b0/figures/Figure_1,_Figure_2.png)
*Figure 1, Figure 2 (result): Figure 1: Trade-off between performance vs robustness over multiple datasets (λ = 0.1). Figure 2: Performance degradation of AdaBoost and other methods as noise level increases on UCI datasets.*



Table 1显示，RMBoost在噪声环境下展现出系统性优势：10%随机噪声下平均排名1.71，显著优于AdaBoost（排名2.38–16.37区间）和LPBoost（4.39–18.12）；20%随机噪声下排名4.10，对抗10%噪声下3.88。在大型数据集上（Table 5），RMBoost的噪声鲁棒性更为突出：Susy数据集无噪声时RMBoost为23±2.0%，与AdaBoost的24±1.8%接近；但20%对抗噪声下RMBoost仅升至38±2.5%，而AdaBoost恶化至40±1.9%，XGB-Quad恶化至39±2.0%。Higgs数据集上，20%对抗噪声时RMBoost为48±3.3%，AdaBoost达50±2.2%，LPBoost更高达49±2.8%。Forest Covertype上RMBoost在20%对抗噪声下36±2.6%，优于AdaBoost的39±2.4%。值得注意的是，XGB-Quad在干净数据上表现最佳（Susy无噪声1.65–9.42%区间），但噪声下退化严重，验证了非鲁棒方法的脆弱性。



λ的消融实验（Figure 6）表明，正则化参数是准确率-鲁棒性权衡的关键杠杆：较小λ增强鲁棒性但牺牲干净数据性能，较大λ反之。理论建议λ ~ 1/√n 在经验上位于帕累托前沿的拐点区域。

公平性检查：基线选择涵盖经典Boosting（AdaBoost、LogitBoost）、鲁棒变体（BrownBoost、RobustBoost）和现代实现（XGB-Quad、Robust-GBDT），但缺少LightGBM、CatBoost等当前工业界主流方法，亦未与[1]所强调的深度学习tabular方法对比。实验仅限tabular数据，高维图像/文本数据泛化性未验证。作者坦承理论保证基于最坏情况分析，可能偏保守；且干净数据上RMBoost偶尔略逊于最优非鲁棒方法。

## 方法谱系与知识库定位

RMBoost属于**鲁棒Boosting**方法族，直接父方法为**Minimax classification with 0-1 loss and performance guarantees** [20]（NeurIPS 2020），后者将极小极大框架引入分类问题，RMBoost将其适配至Boosting的序列集成结构。

**改变槽位**：
- **objective**：经验凸代理损失 → 极小极大0-1误差概率
- **training_recipe**：坐标下降贪婪优化 → 分布鲁棒优化（Wasserstein DRO），λ控制权衡
- **reward_design**：新增最坏情况误差概率作为显式优化目标
- **inference_strategy**：保持加权多数投票结构，权重来源改为极小极大解

**直接基线差异**：
- **vs AdaBoost** [2]：替换指数损失为极小极大0-1损失，从贪婪坐标下降改为全局鲁棒优化
- **vs LPBoost** [25]：同为每轮求解优化问题，但LPBoost最小化经验间隔，RMBoost最小化最坏情况误差概率
- **vs BrownBoost/RobustBoost** [12][11]：同为鲁棒Boosting，但后者无有限样本保证且针对特定噪声模型
- **vs XGB-Quad** [4]：干净数据上XGB-Quad更强，但噪声下RMBoost显著更稳定

**后续方向**：（1）将极小极大框架扩展至梯度提升决策树（GBDT）的大规模实现，匹配XGBoost/LightGBM的效率；（2）探索λ的自适应选择，替代固定的1/√n启发式；（3）验证在高维数据（图像、文本嵌入）上的有效性，突破当前tabular局限。

**标签**：modality:tabular | paradigm:supervised ensemble | scenario:label noise robustness | mechanism:minimax optimization / distributionally robust optimization / Wasserstein DRO | constraint:finite-sample guarantee

