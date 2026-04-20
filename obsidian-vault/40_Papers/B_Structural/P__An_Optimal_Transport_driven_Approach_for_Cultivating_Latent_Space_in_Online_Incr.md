---
title: An Optimal Transport-driven Approach for Cultivating Latent Space in Online Incremental Learning
type: paper
paper_level: B
venue: arXiv (Cornell University)
year: 2022
acceptance: null
cited_by: 1
facets:
  learning_paradigm:
  - Reinforcement Learning
  modality:
  - Image
paper_link: https://arxiv.org/abs/2211.16780
---

# An Optimal Transport-driven Approach for Cultivating Latent Space in Online Incremental Learning

> 本文提出OTC（Optimal Transport-driven Centroid）框架，核心由两个模块构成：MMOT（Mixture Model via Optimal Transport）和Dynamic Preservation策略。

**MMOT模块**：将每个类别的潜在空间表示从单一质心或固定多质心升级为可增量更新的高斯混合模型（GMM）。其数学基础是两个GMM之间的最优传输问题：
$$\min_{\gamma \in \Gamma(\pi^{(N)}, \pi^{(P)})} \sum_{i=1}^{K_1} \sum_{j=1}^{K_2} \gamma_{i,j} W_2(N

> **结构性改进**。先读 baseline，再看本文修改了哪些核心组件。

## 核心公式

$$
\min_{\gamma \in \Gamma(\pi^{(N)}, \pi^{(P)})} \sum_{i=1}^{K_1} \sum_{j=1}^{K_2} \gamma_{i,j} W_2(N_i, P_j)
$$

> 定义了两个高斯混合模型之间的最优传输问题，是MMOT框架的核心数学基础，用于增量更新类别质心。
> *Slot*: MMOT centroid alignment / mixture model fitting

$$
A_T = \frac{1}{T} \sum_{i=1}^{T} a_i
$$

> 平均准确率指标，是论文所有实验结果对比的主要评估标准。
> *Slot*: evaluation metric

$$
F_T = \frac{1}{T-1} \sum_{i=1}^{T-1} f_i
$$

> 平均遗忘率指标，用于衡量模型在持续学习过程中对旧任务知识的保留能力。
> *Slot*: evaluation metric / catastrophic forgetting

$$
T_{\text{MMOT}} = O(T_\phi B + BKd + SBd),\quad M_{\text{MMOT}} = O(Bd + Kd)
$$

> MMOT每类每批次的时间与空间复杂度，证明其相比EM算法避免了内循环因子和B×K责任矩阵，具有更低的内存开销。
> *Slot*: computational complexity of MMOT update

## 关键图表

**Table 5**
: Offline CIL setting: Average Accuracy on CIFAR10 and CIFAR100 with buffer sizes M=200, 500, 5120, comparing OTC vs DER++, GeoDL, Co2L.
> 证据支持: OTC在离线CIL设置下全面优于所有基线，最大差距超过6%（CIFAR100, M=200: OTC 25.22% vs Co2L 18.85%）。

**Table 3**
: Ablation on centroid-based vs random sample selection for replay buffer on CIFAR10 (M=1000), varying number of centroids 1-8.
> 证据支持: 使用MMOT质心选择回放样本在所有质心数量配置下均优于随机采样，最优配置（4质心）达75.9% vs 73.4%，支持质心驱动多样性假设。

**Table 4 (Section 10.1)**
: Performance comparison on MNIST: OTC vs GSA, MOSE, BiC+AC in average accuracy and forgetting.
> 证据支持: OTC在MNIST上平均准确率最高提升2.4%，平均遗忘率最低降低1.6%，验证方法在额外数据集上的泛化性。

**Figure 5 (referenced in Ablation)**
: Average accuracy curves on CIFAR10 as number of centroids per class varies, for different memory sizes.
> 证据支持: 质心数量存在最优阈值（M=200时为3，M=1K时为4），超过阈值后性能下降，说明多质心收益受记忆容量约束。

## 详细分析

# An Optimal Transport-driven Approach for Cultivating Latent Space in Online Incremental Learning

## Part I：问题与挑战

在线类增量学习（Online Class Incremental Learning, OCIL）面临两个核心挑战：一是灾难性遗忘（catastrophic forgetting），即模型在学习新任务时会覆盖旧任务的知识；二是数据流的多模态分布问题，即同一类别的样本在潜在空间中往往呈现多峰分布，而非单一高斯分布。现有方法通常采用两种策略来表示潜在空间中的类别：（1）单一自适应质心——能够随数据流更新，但无法捕捉类内多模态结构；（2）多个固定质心——能表达多模态性，但无法随数据流增量更新，在持续分布漂移下会逐渐失效。这两种策略的根本矛盾在于：多模态表达能力与在线增量更新能力难以兼得。此外，在线场景下旧样本的回放缓冲区容量有限，如何从有限内存中选取最具代表性的样本以维持类别可分性，也是一个未被充分解决的问题。现有基于EM算法的高斯混合模型（GMM）拟合方法虽然理论上可以建模多模态分布，但其计算复杂度为O(I_EM·B·K·d)，内循环收敛迭代次数I_EM较大，且需要存储B×K的责任矩阵，在持续特征漂移的在线场景下稳定性差、内存开销高，不适合直接应用于OCIL。

## Part II：方法与洞察

本文提出OTC（Optimal Transport-driven Centroid）框架，核心由两个模块构成：MMOT（Mixture Model via Optimal Transport）和Dynamic Preservation策略。

**MMOT模块**：将每个类别的潜在空间表示从单一质心或固定多质心升级为可增量更新的高斯混合模型（GMM）。其数学基础是两个GMM之间的最优传输问题：
$$\min_{\gamma \in \Gamma(\pi^{(N)}, \pi^{(P)})} \sum_{i=1}^{K_1} \sum_{j=1}^{K_2} \gamma_{i,j} W_2(N_i, P_j)$$
通过求解该OT问题，MMOT能够在每个mini-batch到来时增量更新混合模型参数（均值、方差、权重），而无需像EM算法那样进行多轮内循环迭代。具体实现上，MMOT使用重参数化技巧和Gumbel-Softmax进行采样，通过熵正则化OT对偶目标函数更新参数，每类每批次的时间复杂度为O(T_φB + BKd + SBd)，内存复杂度为O(Bd + Kd)，避免了EM算法的B×K责任矩阵，在持续特征漂移下具有更好的稳定性。

**MMOT驱动的回放缓冲区选样**：利用MMOT得到的多质心来指导回放缓冲区的样本选择，优先选取靠近各质心的样本，以提升缓冲区的多样性和代表性。消融实验表明，基于质心的选样策略在所有质心数量配置下均优于随机采样（最优配置下75.9% vs 73.4%）。

**MMOT驱动的推理策略**：在测试阶段，利用多质心进行类别相似度估计，替代单质心最近邻分类，从而更准确地处理多模态类内分布。

**Dynamic Preservation策略**：设计了一种正则化机制，通过约束潜在空间来维持类别可分性，缓解灾难性遗忘。该策略与MMOT协同工作，但论文中对其独立消融证据相对有限。

整体而言，OTC的改动集中在潜在空间的类别表示模块（从单/固定多质心→可增量更新的GMM）和回放样本选择策略上，骨干网络、损失函数主体、任务划分方式等均沿用标准OCIL范式。

## Part III：证据与局限

实验在CIFAR10、CIFAR100和MNIST三个数据集上进行，覆盖在线和离线CIL两种设置。主要证据如下：

1. **离线CIL（Table 5）**：OTC在CIFAR100、M=200时以25.22%的平均准确率超越最强基线Co2L（18.85%），差距超过6%；在CIFAR10和CIFAR100的所有缓冲区大小配置下均优于DER++、GeoDL、Co2L，结果较为一致。

2. **质心选样消融（Table 3）**：基于MMOT质心的选样策略在CIFAR10（M=1000）上全面优于随机采样，最优配置（4质心）达75.9% vs 73.4%，支持多质心多样性假设。

3. **MNIST补充实验（Table 4）**：OTC相比GSA、MOSE、BiC+AC平均准确率最高提升2.4%，平均遗忘率最多降低1.6%，但MNIST难度较低，泛化价值有限。

**主要局限**：（1）多质心收益受缓冲区大小约束，M=200时最优质心数仅为3，超过则性能下降，该边界条件仅在CIFAR10上验证；（2）离线设置对比基线（DER++、GeoDL、Co2L）缺少ER-ACE、FOSTER等近年强基线；（3）Dynamic Preservation策略缺乏独立消融实验，其贡献难以单独量化；（4）MMOT的计算复杂度优势仅为理论推导，未提供实际运行时间或内存占用的实验对比。
