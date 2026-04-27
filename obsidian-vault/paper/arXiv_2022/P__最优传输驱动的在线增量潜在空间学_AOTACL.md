---
title: An Optimal Transport-driven Approach for Cultivating Latent Space in Online Incremental Learning
type: paper
paper_level: B
venue: arXiv
year: 2022
paper_link: https://arxiv.org/abs/2211.16780
aliases:
- 最优传输驱动的在线增量潜在空间学习
- AOTACL
- 本文提出OTC（Optimal Transport-driven Ce
cited_by: 1
modalities:
- Image
paradigm: Reinforcement Learning
---

# An Optimal Transport-driven Approach for Cultivating Latent Space in Online Incremental Learning

[Paper](https://arxiv.org/abs/2211.16780)

**Topics**: [[T__Continual_Learning]], [[T__Self-Supervised_Learning]]

> [!tip] 核心洞察
> 本文提出OTC（Optimal Transport-driven Centroid）框架，核心由两个模块构成：MMOT（Mixture Model via Optimal Transport）和Dynamic Preservation策略。

**MMOT模块**：将每个类别的潜在空间表示从单一质心或固定多质心升级为可增量更新的高斯混合模型（GMM）。其数学基础是两个GMM之间的最优传输问题：
$$\min_{\gamma \in \Gamma(\pi^{(N)}, \pi^{(P)})} \sum_{i=1}^{K_1} \sum_{j=1}^{K_2} \gamma_{i,j} W_2(N_i, P_j)$$
通过求解该OT问题，MMOT能够在每个mini-batch到来时增量更新混合模型参数（均值、方差、权重），而无需像EM算法那样进行多轮内循环迭代。具体实现上，MMOT使用重参数化技巧和Gumbel-Softmax进行采样，通过熵正则化OT对偶目标函数更新参数，每类每批次的时间复杂度为O(T_φB + BKd + SBd)，内存复杂度为O(Bd + Kd)，避免了EM算法的B×

| 中文题名 | 最优传输驱动的在线增量潜在空间学习 |
| 英文题名 | An Optimal Transport-driven Approach for Cultivating Latent Space in Online Incremental Learning |
| 会议/期刊 | arXiv (Cornell University) (预印本) |
| 链接 | [arXiv](https://arxiv.org/abs/2211.16780) · [Code] · [Project] |
| 主要任务 | 在线类增量学习 (Online Class Incremental Learning, OCIL) |
| 主要 baseline | DER++, GeoDL, Co2L, GSA, MOSE, BiC+AC |

> [!abstract] 因为「在线增量学习中单一质心无法捕捉类内多模态分布，而固定多质心无法随数据流更新」，作者在「标准OCIL范式（DER++/Co2L等）」基础上改了「将类别表示升级为可通过最优传输增量更新的高斯混合模型（GMM），并设计基于多质心的回放缓冲区选样策略」，在「CIFAR100 (M=200)」上取得「平均准确率25.22% vs Co2L的18.85%，提升6.37%」

- **CIFAR100 (M=200)**: OTC 25.22% vs Co2L 18.85%，差距 **+6.37%**（Table 5）
- **CIFAR10 (M=1000) 质心选样消融**: 4质心配置 75.9% vs 随机采样 73.4%，提升 **+2.5%**（Table 3）
- **MNIST**: 相比GSA/MOSE/BiC+AC，平均准确率最高提升 **2.4%**，遗忘率最多降低 **1.6%**（Table 4）

## 背景与动机

在线类增量学习（OCIL）要求模型从持续到达的数据流中逐步学习新类别，同时保持对旧类别的识别能力。一个典型场景是：智能监控摄像头每天捕获新类型的人脸，模型必须在仅存储少量历史样本的条件下，不断扩展识别范围而不遗忘已学类别。然而，这一场景面临双重困境：同一类别的人脸可能因光照、角度、表情等因素呈现多峰分布，而在线约束又禁止离线重训练。

现有方法在潜在空间表示上采用两种策略。**单一自适应质心**（如BiC+AC中的adaptive centroid）能够随数据流更新位置，但将类内分布强行压缩为单点，丢失了多模态结构；**多个固定质心**（如某些基于聚类的回放方法）虽能表达多峰性，却在持续分布漂移下逐渐失效——新任务到来时旧质心不再代表当前特征空间。更根本的是，基于EM算法的高斯混合模型（GMM）虽理论上可兼顾两者，但其O(I_EM·B·K·d)的计算复杂度和B×K责任矩阵的内存开销，在在线场景下因特征漂移导致EM内循环不稳定，实际难以部署。


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/e97863e4-4983-435e-acbc-0a4920b6d4bc/figures/Figure_2.png)
*Figure 2: Figure 1. Motivation of our method (t-SNE visualization on MNIST): Left: the test latent representation of with one adaptive centroid(i.e., visualized by digits) per class. Right: the test latent repr*



上述矛盾的核心在于：**多模态表达能力与在线增量更新能力难以兼得**。本文提出OTC框架，以最优传输（Optimal Transport）替代EM算法，实现GMM参数的增量更新，从而在在线约束下同时获得多质心的表达力和自适应的更新能力。

## 核心创新

核心洞察：两个GMM之间的最优传输距离可以替代EM算法的责任矩阵更新，因为OT对偶问题的熵正则化形式允许通过重参数化和Gumbel-Softmax进行端到端梯度下降，从而使在线场景下的增量GMM参数更新在计算和内存上都成为可行。

| 维度 | Baseline (EM-GMM / 单质心 / 固定多质心) | 本文 (OTC-MMOT) |
|:---|:---|:---|
| 类别表示 | 单质心或固定多质心；或EM拟合GMM | 可增量更新的GMM，质心数自适应调整 |
| 更新机制 | EM内循环迭代（I_EM轮）或无法更新 | 每mini-batch单步OT求解，无内循环 |
| 计算复杂度 | O(I_EM·B·K·d) | O(T_φB + BKd + SBd) |
| 内存开销 | O(B×K)责任矩阵 | O(Bd + Kd)，无显式责任矩阵 |
| 回放选样 | 随机采样或基于单质心 | 基于多质心覆盖的多样性优先选样 |
| 推理方式 | 单质心最近邻 | 多质心相似度聚合 |

## 整体框架


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/e97863e4-4983-435e-acbc-0a4920b6d4bc/figures/Figure_3.png)
*Figure 3: Figure 3. Average accuracy through tasks.*



OTC框架围绕潜在空间的"培育"（cultivating）展开，数据流如下：

**输入**：当前mini-batch的样本流 (x_t, y_t) + 回放缓冲区存储的有限旧样本

**模块1：特征提取器**（标准CNN/ResNet骨干，参数θ）→ 将输入映射到潜在空间特征 z ∈ R^d

**模块2：MMOT（Mixture Model via Optimal Transport）** → 对每个类别维护一个GMM（含K个高斯分量），通过OT距离增量更新各分量的均值μ、协方差Σ、混合权重π。输入为当前batch的特征{z}，输出为更新后的GMM参数。

**模块3：Dynamic Preservation** → 正则化项，约束新旧任务的潜在空间结构保持类别可分性，缓解灾难性遗忘。作用于特征提取器的训练目标。

**模块4：MMOT驱动的回放选样** → 利用当前GMM的K个质心位置，计算缓冲区候选样本与各质心的距离，优先选取覆盖各模态的代表性样本替换入缓冲区。

**模块5：MMOT驱动的推理** → 测试时计算查询样本与各类别GMM所有分量的相似度，聚合为多质心评分，替代单质心最近邻决策。

**输出**：类别预测 + 更新后的特征提取器θ + 更新后的类别GMM + 更新后的回放缓冲区

```
Data Stream → [Feature Extractor] → Latent Space z
                                      ↓
                    ┌─────────────────┼─────────────────┐
                    ↓                 ↓                 ↓
              [MMOT Update]    [Dynamic Pres.]    [Buffer Selection]
              (GMM params)     (Regularization)   (Centroid-guided)
                    ↓                 ↓                 ↓
              [Inference] ←──────────────────────── [Replay Buffer]
              (Multi-centroid)                      (Limited memory)
```

## 核心模块与公式推导

### 模块1: MMOT——基于最优传输的GMM增量更新（对应框架图 模块2）

**直觉**：EM算法需要多轮迭代收敛责任矩阵，而在线场景下特征漂移使这种迭代不稳定；最优传输的熵正则化对偶形式允许单步梯度更新。

**Baseline公式** (标准EM-GMM):
$$L_{\text{EM}} = \sum_{n=1}^{B} \log \sum_{k=1}^{K} \pi_k \mathcal{N}(z_n | \mu_k, \Sigma_k)$$
其中E-step计算责任矩阵 $\gamma_{nk} = p(k|z_n) \propto \pi_k \mathcal{N}(z_n|\mu_k,\Sigma_k)$，M-step据此更新参数。复杂度O(I_EM·B·K·d)，需存储$\gamma \in \mathbb{R}^{B \times K}$。

**变化点**：EM的内循环迭代在在线特征漂移下收敛困难；责任矩阵内存开销大。本文将两个GMM之间的匹配转化为OT问题，避免显式责任矩阵。

**本文公式（推导）**:
$$\text{Step 1 (OT距离定义)}: \min_{\gamma \in \Gamma(\pi^{(N)}, \pi^{(P)})} \sum_{i=1}^{K_1} \sum_{j=1}^{K_2} \gamma_{i,j} W_2(N_i, P_j) \quad \text{将两个GMM的匹配转化为OT耦合矩阵优化}$$
其中$\Gamma(\pi^{(N)}, \pi^{(P)})$为边缘约束为混合权重$\pi^{(N)}, \pi^{(P)}$的耦合矩阵集合，$W_2(\cdot,\cdot)$为两个高斯分布之间的2-Wasserstein距离。

$$\text{Step 2 (熵正则化对偶)}: \min_{\gamma} \langle \gamma, C \rangle - \varepsilon H(\gamma) + \text{KL}(\gamma \| \pi^{(N)} \otimes \pi^{(P)}) \quad \text{加入熵正则化以获得平滑可微的Sinkhorn迭代}$$
其中$C_{ij} = W_2(N_i, P_j)$为代价矩阵，$H(\gamma)$为熵正则项，$\varepsilon > 0$控制正则化强度。

$$\text{Step 3 (重参数化采样)}: z_{\text{sample}} = \mu_k + \sigma_k \cdot \epsilon, \quad k \sim \text{Gumbel-Softmax}(\log \pi, \tau) \quad \text{通过Gumbel-Softmax实现离散分量选择的梯度回传}$$

$$\text{最终 (MMOT更新目标)}: \mathcal{L}_{\text{MMOT}} = \mathbb{E}_{q_\phi(z|x)}\left[ \text{OT}_\varepsilon\left( q_\phi(z|x), p_{\theta}(z) \right) \right] + \text{KL}\left( q_\phi(z|x) \| \prod_k \mathcal{N}(\mu_k, \Sigma_k)^{\pi_k} \right)$$

**对应消融**：Table 3显示基于质心的选样策略在所有质心数配置下优于随机采样，最优4质心配置75.9% vs 73.4%（+2.5%），验证了多质心表达的必要性。

---

### 模块2: Dynamic Preservation策略（对应框架图 模块3）

**直觉**：仅有多质心表示不足以防止特征提取器的灾难性遗忘，需显式约束潜在空间几何结构。

**Baseline公式** (标准知识蒸馏/回放损失):
$$L_{\text{base}} = L_{\text{CE}}(f_\theta(x), y) + \lambda \cdot L_{\text{replay}}$$
其中$L_{\text{replay}}$通常为回放样本的交叉熵或特征蒸馏。

**变化点**：标准回放仅约束输出层或特征点，未显式维护类别间的决策边界结构。本文引入基于多质心的空间约束。

**本文公式（推导）**:
$$\text{Step 1 (质心约束)}: \mathcal{L}_{\text{centroid}} = \sum_{c \in \mathcal{C}_{\text{old}}} \sum_{k=1}^{K_c} \| \mu_{c,k}^{(t)} - \mu_{c,k}^{(t-1)} \|_2^2 \quad \text{约束旧类别质心位置不漂移}$$

$$\text{Step 2 (间隔约束)}: \mathcal{L}_{\text{margin}} = \sum_{c_1 \neq c_2} \max\left(0, m - \min_{k_1,k_2} \| \mu_{c_1,k_1} - \mu_{c_2,k_2} \|_2 \right)^2 \quad \text{强制不同类别质心间保持最小间隔m}$$

$$\text{最终}: L_{\text{total}} = L_{\text{CE}} + \lambda_1 L_{\text{replay}} + \lambda_2 \mathcal{L}_{\text{centroid}} + \lambda_3 \mathcal{L}_{\text{margin}}$$

**对应消融**：—— 论文未提供Dynamic Preservation的独立消融实验，其贡献难以单独量化。

## 实验与分析

**主结果：离线CIL设置（Table 5）**

| Method | CIFAR10 (M=200) | CIFAR10 (M=500) | CIFAR100 (M=200) | CIFAR100 (M=500) | 趋势 |
|:---|:---|:---|:---|:---|:---|
| DER++ | 
| GeoDL | 
| Co2L | 
| **OTC (本文)** | **| **最优** |



核心证据分析：CIFAR100 (M=200) 上OTC以25.22%显著超越Co2L的18.85%（+6.37%），这一差距在缓冲区最小的严苛设置下尤为突出，直接支持了"多质心表示+OT更新在有限内存下更有效"的核心主张。CIFAR10上优势相对温和，可能因为10类任务分割较粗，类间混淆压力小于100类场景。

**质心选样消融（Table 3）**

| 质心数 | 随机采样 | MMOT质心选样 | Δ |
|:---|:---|:---|:---|
| 1 | 73.4% | 73.4% | 0% |
| 2 | 73.4% | 74.5% | +1.1% |
| 3 | 73.4% | 75.1% | +1.7% |
| 4 | 73.4% | **75.9%** | **+2.5%** |
| 5 | 73.4% | 75.6% | +2.2% |



关键发现：多质心选样收益呈倒U型，4质心最优。论文指出M=200时最优质心数降至3，超过则性能下降——这一边界条件揭示了**缓冲区大小对多质心表达能力的硬约束**：质心数增加需更多样本来可靠估计各分量参数，小缓冲区下过拟合风险上升。

**公平性检查**：
- **基线强度**：离线设置缺少ER-ACE、FOSTER等2022年后强基线，Co2L虽为当时SOTA但非最现代方法；在线设置基线GSA/MOSE/BiC+AC相对陈旧。
- **计算成本**：MMOT的理论复杂度优势（无I_EM内循环）未经验证——论文未报告实际训练时间或峰值内存。
- **失败案例/局限**：Dynamic Preservation缺乏独立消融；MNIST结果泛化价值有限（任务过简单）；OT的熵正则化参数$\varepsilon$敏感性未分析。

## 方法谱系与知识库定位

**方法家族**：增量学习中的潜在空间几何约束方法 → 具体属于"基于质心/原型表示的类增量学习"分支。

**父方法**：标准Experience Replay (ER) 框架 + 基于原型的分类器（如iCaRL的nearest-mean-of-exemplars）。OTC将静态原型扩展为动态GMM，并引入最优传输作为更新机制。

**直接基线与差异**：
- **DER++**：使用回放+蒸馏，单质心隐式表示；OTC改为显式多质心GMM+OT更新。
- **Co2L**：对比学习增强特征判别性，单质心；OTC保留对比思想但升级为多模态表示。
- **BiC+AC**：自适应单质心；OTC解决其无法表达多峰分布的局限。
- **EM-GMM（理论基线）**：同目标但算法不同；OTC以OT替代EM，获得在线稳定性。

**改动槽位**：
- 架构：特征提取器沿用标准CNN（未改）
- 目标函数：新增MMOT-OT损失 + Dynamic Preservation正则项（核心改动）
- 训练流程：每batch插入MMOT参数更新步骤（改动）
- 数据策展：回放选样策略改为质心引导（改动）
- 推理：多质心相似度聚合（改动）

**后续方向**：(1) 将MMOT与更现代骨干（Vision Transformer）结合，验证可扩展性；(2) 探索OT耦合矩阵在任务边界检测中的信号价值；(3) 开发自适应质心数选择机制，替代当前网格搜索。

**标签**：`modality: 图像` / `paradigm: 在线类增量学习` / `scenario: 有限内存持续学习` / `mechanism: 最优传输+高斯混合模型` / `constraint: 单轮数据流、小回放缓冲区`

