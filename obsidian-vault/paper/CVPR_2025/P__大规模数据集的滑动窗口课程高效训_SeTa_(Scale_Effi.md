---
title: Scale Efficient Training for Large Datasets
type: paper
paper_level: C
venue: CVPR
year: 2025
paper_link: null
aliases:
- 大规模数据集的滑动窗口课程高效训练
- SeTa (Scale Effi
- SeTa (Scale Efficient Training)
acceptance: poster
cited_by: 7
code_url: https://github.com/mrazhou/SeTa
method: SeTa (Scale Efficient Training)
---

# Scale Efficient Training for Large Datasets

[Code](https://github.com/mrazhou/SeTa)

**Topics**: [[T__Self-Supervised_Learning]], [[T__Classification]] | **Method**: [[M__SeTa]] | **Datasets**: [[D__CIFAR-10]], [[D__CIFAR-100]] (其他: ToCa, WHU-MVS, RefCOCO)

| 中文题名 | 大规模数据集的滑动窗口课程高效训练 |
| 英文题名 | Scale Efficient Training for Large Datasets |
| 会议/期刊 | CVPR 2025 (poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2503.13385) · [Code](https://github.com/mrazhou/SeTa) · [DOI](https://doi.org/10.1109/CVPR52734.2025.01905) |
| 主要任务 | 数据集剪枝 / 高效训练 / 课程学习 |
| 主要 baseline | InfoBatch, Dataset Quantization, EfficientTrain++, UCB, Deep Learning on a Data Diet |

> [!abstract] 因为「大规模数据集训练计算开销巨大，现有动态剪枝方法在极端剪枝率下失效或引入不可忽略的选择开销」，作者在「EfficientTrain 课程学习框架」基础上改了「滑动窗口难度分组 + 轻量K-means聚类 + 部分退火策略」，在「CIFAR10/100、ImageNet-1K及10+跨模态任务」上取得「70%剪枝率下CIFAR10 95.7%（仅-0.6 vs全数据）、ImageNet-1K多架构零性能损失」

- **CIFAR10**: 70%剪枝率下达95.7%，超越UCB +1.1，InfoBatch在此剪枝率下完全失效
- **ImageNet-1K**: ResNet-18/Swin/ViT/Vim四架构均实现零性能损失，最高剪枝46.4%
- **跨任务泛化**: 覆盖视觉描述、多视图立体、指令微调、图像检索等10+任务，平均数据减少25-60%

## 背景与动机

现代深度学习模型的训练成本随数据集规模线性增长。以ImageNet-1K为例，完整训练一个ViT-L需要数百GPU小时，而实际数据中存在大量冗余样本——模型在早期epoch就能正确分类的"简单"样本反复参与梯度更新，造成计算浪费。理想情况下，我们希望动态识别并跳过这些冗余样本，仅用核心子集达到相近性能。

现有方法主要从三个角度解决这一问题：
- **静态剪枝（Dataset Pruning / Data Diet）**：训练前基于遗忘分数或影响函数一次性筛选重要样本。这类方法如"Deep Learning on a Data Diet"提前锁定核心集，但无法适应训练动态变化——早期困难的样本可能在后期变简单，静态选择导致次优。
- **动态剪枝（InfoBatch / UCB）**：每轮根据当前损失或不确定性重新采样。InfoBatch通过动态调整保留概率实现无损加速，但其随机采样缺乏结构性，在70%以上极端剪枝率时因样本多样性崩溃而失效（本文证实其无法在CIFAR10/100达到70%剪枝）。
- **课程学习（EfficientTrain / EfficientTrain++）**：按难度排序从易到难训练，但需设计手工难度度量，且每轮仍需遍历全数据集排序或计算复杂的前向传播开销。

这些方法的共同瓶颈在于：**数据选择本身引入了不可忽略的额外计算**（$O_d$不可控），或**在极端剪枝率下无法保持样本多样性**。具体而言，InfoBatch的全数据退火策略在最终阶段抵消了剪枝收益；EfficientTrain++的广义课程学习缺乏自适应的细粒度难度分组机制。本文提出SeTa，核心动机是：能否用可忽略的额外开销（$\rho_O \approx \bar{\rho}$），通过结构化难度分组与渐进式窗口调度，在70%+剪枝率下仍保持全数据性能？

## 核心创新

核心洞察：**损失值天然反映样本难度，而K-means聚类可将连续损失空间离散化为结构化难度组；通过滑动窗口在排序后的组上渐进移动，实现"自动课程"的同时保证每轮样本多样性，从而使极端剪枝率下的无损训练成为可能。**

与 baseline 的差异：

| 维度 | Baseline (InfoBatch / EfficientTrain++) | 本文 SeTa |
|:---|:---|:---|
| 难度估计 | 逐样本损失/不确定性，需全量计算或复杂度量 | K-means聚类分组，损失值自动作为难度代理，开销可忽略 |
| 采样结构 | 随机动态采样（InfoBatch）或固定课程阶段（EfficientTrain++） | 滑动窗口在排序组上连续滑动，兼顾结构性与多样性 |
| 训练进度 | 均匀难度分布或预定义阶段划分 | easy-to-hard自动渐进，窗口位置随epoch连续推进 |
| 退火策略 | 全数据退火（InfoBatch）或无显式退火 | 部分退火，仅用子集消除偏差，保持数据效率 |
| 理论保证 | 无显式效率约束或$O_d$未控制 | $\rho_O = \bar{\rho} + O_d/O_m \approx \bar{\rho}$，选择开销严格可控 |

## 整体框架


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/358c2f55-ae24-4f93-b381-2437f40aa9fc/figures/Figure_2.png)
*Figure 2 (pipeline): Overview of the proposed method for efficient training.*



SeTa 的数据流遵循六阶段流水线，核心思想是"先分组、再窗口、后渐进"：

**输入**: 完整训练数据集 $D = \{(x_i, y_i)\}_{i=1}^{|D|}$，当前模型参数 $\theta$

1. **损失计算（Loss Computation）**: 对当前模型做前向传播，计算每个样本的损失值 $\text{ell}_i^t = \text{ell}(\theta; x_i, y_i)$。此步骤与正常训练共享前向计算，无额外开销。

2. **K-means 难度分组（K-means Grouping）**: 将所有样本按损失值 $\{\text{ell}_i^t\}$ 聚为 $k$ 组 $\{\mathcal{G}_j\}_{j=1}^k$，组中心为 $\{c_j\}$。损失越低=组越"简单"，损失越高=组越"困难"。

3. **难度排序（Difficulty Sorting）**: 按组中心 $c_j$ 升序排列各组，形成难度轴 $\mathcal{G}_{(1)}, \mathcal{G}_{(2)}, ..., \mathcal{G}_{(k)}$。

4. **滑动窗口选择（Sliding Window Selection）**: 在第 $n$ 个epoch，窗口宽度 $w$，计算起始位置 $s_t = n \mod (k-w+1)$，结束位置 $e_t = s_t + w - 1$。选中组 $\{\mathcal{G}_{(s_t)}, ..., \mathcal{G}_{(e_t)}\}$ 的并集作为本轮训练子集 $S_t$。

5. **课程训练（Curriculum Training）**: 在 $S_t$ 上执行标准梯度更新，得到 $\theta_{S_t}$。随着epoch推进，窗口从简单组滑向困难组，实现自动课程。

6. **部分退火（Partial Annealing）**: 最后若干epoch，从当前窗口组中以比率 $r$ 随机采样子集进行退火，消除课程学习引入的分布偏差。

**输出**: 最终模型 $\theta_{S_T}$，满足 $|\mathcal{L}(\theta_{S_T}) - \mathcal{L}(\theta_D)| < \epsilon$

```
D ──→ Loss(θ) ──→ K-means(k组) ──→ Sort by c_j ──→ Window[st:et] ──→ Train on St ──→ Partial Anneal ──→ θ_ST
         ↑______________________________________________________________|
```

## 核心模块与公式推导

### 模块 1: 优化目标与效率保证（对应框架图 整体目标）

**直觉**: 数据集剪枝的本质是在最小化训练数据量的同时，保证最终模型与全数据训练模型的性能差异可控。

**Baseline 公式** (传统剪枝/课程学习): 无统一显式优化目标，通常采用启发式准则如
$$\mathcal{S}^* = \text{arg}\max_{\mathcal{S} \subset D, |\mathcal{S}|=m} \sum_{x_i \in \mathcal{S}} \text{Importance}(x_i)$$
符号: $\text{Importance}(x_i)$ = 手工设计的重要性分数（如遗忘分数、梯度范数、影响函数值）

**变化点**: 传统方法要么缺乏端到端优化保证（静态剪枝），要么重要性计算开销 $O_d$ 不可控（动态影响函数）。SeTa 将目标重新表述为带约束的组合优化，并显式分解效率比率。

**本文公式（推导）**:
$$\text{Step 1 (核心目标)}: \min_{\{S_t\}_{t=1}^T} \sum_{t=1}^T |S_t| \quad \text{s.t.} \quad |\mathcal{L}(\theta_{S_T}) - \mathcal{L}(\theta_D)| < \epsilon$$
$$\text{其中}: \theta_{S_T} = \text{arg}\min_{\theta} \sum_{t=1}^T \mathbb{E}_{(x,y) \sim S_t}[\text{ell}(\theta; x, y)], \quad \theta_D = \text{arg}\min_{\theta} \mathbb{E}_{(x,y) \sim D}[\text{ell}(\theta; x, y)]$$
$$\text{Step 2 (效率分解)}: \rho_O = \frac{|D| \times \bar{\rho} \times O_{m} + |D| \times O_{d}}{|D| \times O_{m}} = \bar{\rho} + \frac{O_{d}}{O_{m}}$$
$$\text{关键保证}: O_d \ll O_m \Rightarrow \rho_O \approx \bar{\rho}$$
符号: $\bar{\rho}$ = 目标剪枝率, $O_m$ = 模型训练计算开销, $O_d$ = 数据选择开销, $\rho_O$ = 实际总体效率比率

**对应消融**: 本文通过K-means聚类（$O(|D|)$）替代影响函数（$O(|D|^2)$ 或需Hessian），确保 $O_d$ 可忽略。

---

### 模块 2: K-means 损失聚类与滑动窗口调度（对应框架图 核心机制）

**直觉**: 连续损失值难以直接用于结构化采样，聚类离散化后配合滑动窗口可同时控制"难度范围"和"样本多样性"。

**Baseline 公式** (标准K-means / InfoBatch随机采样):
$$\mathcal{C}^* = \text{arg}\min_{\mathcal{C}} \sum_{j=1}^k \sum_{x_i \in \mathcal{G}_j} \|x_i - c_j\|^2 \quad \text{(原始特征空间)}$$
InfoBatch: 每轮从全数据集按保留概率 $p_i$ 随机采样，无结构化难度组织。

**变化点**: (1) 将聚类对象从原始特征 $x_i$ 改为损失值 $\text{ell}_i^t$，使"难度"成为内建属性；(2) 用滑动窗口替代随机采样，保证每轮样本覆盖连续难度区间而非离散跳跃。

**本文公式（推导）**:
$$\text{Step 1 (损失空间聚类)}: \mathcal{C}^* = \text{arg}\min_{\mathcal{C}} \sum_{j=1}^k \sum_{x_i \in \mathcal{G}_j} \|\text{ell}_i^t - c_j\|^2$$
$$\text{Step 2 (动态分组分配)}: \text{group}(x_i) = \text{arg}\min_{j} \|\text{ell}_i^t - c_j\|^2, \quad c_j = \frac{1}{|\mathcal{G}_j|} \sum_{x_i \in \mathcal{G}_j} \text{ell}_i^t$$
$$\text{Step 3 (难度排序与窗口定位)}: \mathcal{G}_{(1)}, \mathcal{G}_{(2)}, ..., \mathcal{G}_{(k)} \text{ sorted by } c_j \text{ ascending}$$
$$s_t = n \mod (k - w + 1), \quad e_t = s_t + w - 1$$
$$\text{Step 4 (子集构造)}: S_t = \text{bigcup}_{j=s_t}^{e_t} \mathcal{G}_{(j)}$$

**对应消融**: Table 13（消融实验）显示，动态采样（vs滑动窗口）CIDEr Overall 71.5 → 69.8（-1.7），静态采样 71.5 → 69.4（-2.1）；hard-to-easy替代easy-to-hard 71.5 → 69.6（-1.9）。

---

### 模块 3: 部分退火策略（对应框架图 最终阶段）

**直觉**: 课程学习的渐进采样会引入分布偏差——模型始终未见某些困难样本直到后期，退火需恢复全数据分布但不应牺牲剪枝收益。

**Baseline 公式** (InfoBatch 全数据退火):
$$\mathcal{S}_t^{\text{anneal, InfoBatch}} = D \quad \text{(full dataset)}$$

**变化点**: InfoBatch在退火阶段使用全数据，导致最终epoch无加速效果。SeTa改为仅从当前窗口组中随机采样，保持退火阶段的剪枝率。

**本文公式（推导）**:
$$\text{Step 1 (退火子集定义)}: \mathcal{S}_t^{\text{anneal}} = \{x_i \text{mid} x_i \in \mathcal{G}, u_i < r\}$$
$$\text{其中 } u_i \sim \text{Uniform}(0,1), \quad r \in (0,1) \text{ 为退火采样比率}$$
$$\text{Step 2 (与全数据退火对比)}: |\mathcal{S}_t^{\text{anneal, SeTa}}| = r \cdot |\mathcal{G}_{\text{window}}| \ll |D| = |\mathcal{S}_t^{\text{anneal, InfoBatch}}|$$
$$\text{Step 3 (偏差消除保证)}: \mathbb{E}_{x \sim \mathcal{S}_t^{\text{anneal}}}[\text{ell}(\theta; x,y)] \text{xrightarrow}{r \to 1, \text{mix}} \mathbb{E}_{x \sim D}[\text{ell}(\theta; x,y)]$$

**对应消融**: 连续调度（无退火）70.7 vs 全SeTa 71.5（-0.8）；InfoBatch式全数据退火 70.3 vs 部分退火 71.5（-1.2）。Table 13 证实部分退火是性能突破的关键组件之一。

## 实验与分析


![Table 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/358c2f55-ae24-4f93-b381-2437f40aa9fc/figures/Table_2.png)
*Table 2 (comparison): Comparison of state-of-the-art methods on CIFAR10 and CIFAR100 with the best SOTA and different pruning ratios.*



本文在15个数据集、10个任务、4类模型架构上验证SeTa的通用性。核心结果集中在CIFAR10/100的极端剪枝场景与ImageNet-1K的跨架构泛化。


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/358c2f55-ae24-4f93-b381-2437f40aa9fc/figures/Table_1.png)
*Table 1: Comprehensive overview of 15 datasets, 10 tasks, and 4 models utilized in the experiments.*



**CIFAR10/100 极端剪枝**: Table 2 显示，SeTa在CIFAR10上于30%剪枝率达95.7%（vs Baseline 95.6，几乎无损），70%剪枝率仍保持95.7%（仅-0.6）；同期InfoBatch在70%剪枝率完全失效，UCB为94.6（SeTa领先+1.1）。CIFAR100上50%剪枝率即超越InfoBatch‡（匹配epoch调整后的对比），70%剪枝率79.4（vs Baseline 79.3，-1.6但仍保持可用）。这一结果直接回应了动态剪枝方法在高剪枝率下的崩溃问题。

**ImageNet-1K 跨架构零损失**: Table 9 显示，ResNet-18在36.1%剪枝下69.5%（=Baseline），Swin Transformer在46.4%剪枝下80.0%（=Baseline），ViT在24.6%剪枝下73.3%（=Baseline），Vim/Mamba在30.0%剪枝下75.7%（=Baseline）。值得注意的是，Swin在55.5%剪枝下仍达79.3%（仅-0.7），Vim在44.8%剪枝下达74.9%（-0.8），显示方法对架构差异的鲁棒性。


![Table 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/358c2f55-ae24-4f93-b381-2437f40aa9fc/figures/Table_3.png)
*Table 3 (example): SCs including 150 image-text samples curated by LLM for semantic concepts.*



**消融实验**（Table 13 / Figure 3系列）揭示关键组件贡献：滑动窗口机制本身（vs动态随机采样）贡献+1.7 CIDEr Overall；easy-to-hard顺序（vs hard-to-easy）贡献+1.9；部分退火（vs InfoBatch式全数据退火）贡献+1.2；连续调度（vs离散跳变）贡献+0.8。窗口比例$\alpha = w/k$在0.4-0.6区间最优（Figure 3a），组数$k$在5-15范围内稳定（Figure 3c）。

**公平性检视**: 对比的baselines涵盖静态剪枝（Dataset Pruning, Data Diet）、动态剪枝（InfoBatch, UCB, Dynamic Data Pruning）和课程学习（EfficientTrain++），选择合理。但存在以下局限：(1) InfoBatch部分结果使用调整epoch（‡），可能混淆剪枝与训练轮数效应；(2) 未提供实际wall-clock时间测量，仅以剪枝率作为硬件无关指标；(3) 部分任务改进极小（RefCOCO +0.03, WHU-MVS -0.0001 MAE），统计显著性未报告；(4) 缺少与2024年最新coreset选择方法及梯度匹配剪枝的对比。

## 方法谱系与知识库定位

**方法家族**: 数据集剪枝 / 课程学习 / 高效训练

**父方法**: EfficientTrain (Exploring Generalized Curriculum Learning for Training Visual Backbones)
- SeTa 继承其"easy-to-hard"课程学习思想，但将手工设计的难度度量替换为损失值自动聚类，将固定阶段划分替换为连续滑动窗口。

**直接 baselines 与差异**:
- **InfoBatch**: 同为动态数据剪枝，SeTa差异在于结构化窗口采样替代随机概率采样，部分退火替代全数据退火
- **EfficientTrain++**: 同为课程学习，SeTa差异在于K-means自动分组替代广义课程设计，滑动窗口替代离散阶段
- **Dataset Quantization / Data Diet**: 同为数据精简，SeTa差异在于动态渐进替代静态一次性选择
- **UCB (Uncertainty-based)**: 同为不确定性采样，SeTa差异在于损失聚类的确定性分组替代逐样本不确定性估计

**修改槽位**: data_pipeline（静态/随机 → 滑动窗口课程）、training_recipe（均匀训练/全退火 → 渐进调度+部分退火）、inference_strategy（复杂重要性计算 → 轻量K-means，$\rho_O \approx \bar{\rho}$）

**后续方向**:
1. 与2024年coreset选择方法（如基于几何覆盖的方法）进行系统对比，验证损失聚类在表示质量上的差距
2. 探索损失值之外的多维难度度量（如梯度一致性、样本不确定性）的聚类效果
3. 将滑动窗口机制扩展至持续学习/联邦学习场景，验证动态数据分布下的稳定性

**标签**: #图像分类 #跨模态任务 #课程学习 #数据集剪枝 #训练加速 #架构无关 #任务无关 #K-means聚类 #滑动窗口 #部分退火

