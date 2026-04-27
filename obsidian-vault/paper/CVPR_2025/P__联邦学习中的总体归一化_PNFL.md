---
title: Population Normalization for Federated Learning
type: paper
paper_level: C
venue: CVPR
year: 2025
paper_link: null
aliases:
- 联邦学习中的总体归一化
- PNFL
acceptance: poster
cited_by: 1
---

# Population Normalization for Federated Learning

**Topics**: [[T__Federated_Learning]]

| 中文题名 | 联邦学习中的总体归一化 |
| 英文题名 | Population Normalization for Federated Learning |
| 会议/期刊 | CVPR 2025 (poster) |
| 链接 | [IEEE](https://doi.org/10.1109/cvpr52734.2025.00955) · Code / Project 未公开 |
| 主要任务 | 联邦学习中的模型聚合与归一化策略优化 |
| 主要 baseline | FedAvg, 其他联邦学习基线方法 |

> [!abstract] 因为「联邦学习中各客户端数据分布异构导致本地统计量（均值/方差）与全局总体统计量存在偏差，进而使 BatchNorm 层在聚合后性能下降」，作者在「标准联邦平均（FedAvg）」基础上改了「引入 Population Normalization（PopNorm）以在服务器端用总体统计量重新校准全局模型」，在「联邦学习图像分类基准」上取得「相比 FedAvg 显著提升的收敛速度与最终精度」（具体数值。

- 关键性能：在异构数据划分下，收敛速度相比 FedAvg 提升
- 关键性能：最终测试精度相比 FedAvg 提升
- 关键性能：通信轮次减少至 FedAvg 的

## 背景与动机

联邦学习（Federated Learning, FL）允许多个客户端在不共享原始数据的前提下协同训练模型。然而，一个长期被忽视的核心问题是：当各客户端数据分布存在异构性（non-IID）时，本地训练的 Batch Normalization（BN）层会积累有偏的统计量。具体而言，假设客户端 A 只有"猫"的图像、客户端 B 只有"狗"的图像，各自计算的 batch mean 和 batch variance 与全局总体分布的统计量截然不同。当服务器按 FedAvg 方式聚合这些本地模型时，BN 层的 running statistics 并未被正确聚合，导致全局模型在推理阶段使用扭曲的归一化参数，精度严重下降。

现有方法主要从三个角度应对此问题：
- **FedAvg**（McMahan et al., 2017）：直接平均本地模型参数，但完全忽略 BN 统计量的聚合，假设各客户端统计量同质。
- **FedBN**（Li et al., 2021）：保留 BN 层在本地不聚合，仅聚合其他层参数，缓解了异构性但牺牲了全局表示一致性。
- **其他归一化变体**（如 GroupNorm、LayerNorm）：试图绕过 BN 的统计量问题，但在视觉任务上通常以牺牲精度为代价。

这些方法的根本局限在于：**没有显式估计并利用全局总体统计量（population statistics）来校准聚合后的模型**。FedAvg 的隐式假设是本地 batch statistics 的加权平均能逼近总体统计量，这在 non-IID 下不成立；FedBN 则彻底放弃了全局 BN 的协同。因此，如何在服务器端准确恢复总体均值与方差，并用其重新归一化全局模型，成为提升联邦学习在异构场景下性能的关键缺口。

本文提出 Population Normalization（PopNorm），核心思想是在聚合过程中显式维护并传播总体统计量，使全局模型的 BN 层在推理时使用正确的 population statistics。

## 核心创新

核心洞察：在联邦学习的模型聚合阶段显式引入总体统计量（population mean/variance）进行重新归一化，因为本地 batch statistics 的加权平均在 non-IID 数据下是有偏估计，从而使服务器能够维护一个统计量一致的全局 BN 层、避免推理时的分布偏移成为可能。

| 维度 | Baseline (FedAvg) | 本文 (PopNorm) |
|:---|:---|:---|
| 统计量来源 | 各客户端本地 batch statistics 隐式累积 | 服务器端显式估计全局 population statistics |
| BN 层聚合方式 | 直接平均参数，running statistics 随参数混合 | 参数聚合后，用总体统计量重新校准 BN |
| 对 non-IID 的假设 | 假设本地统计量近似总体（强假设） | 显式处理统计量异构，无此假设 |
| 通信开销 | 仅传输模型参数 | 额外传输统计量信息（可控开销） |

## 整体框架

PopNorm 的整体流程遵循"本地训练 → 统计量上传 → 服务器聚合与重归一化 → 全局分发"的循环，关键创新在于服务器端的 Population Normalization 模块。

数据流：
- **输入**: 各客户端本地数据集 $D_k$（异构分布）
- **本地训练**: 各客户端 $k$ 基于本地数据训练，得到模型参数 $\theta_k$ 及本地 batch statistics（每层的 $\mu_k, \sigma_k^2$）
- **统计量收集**: 客户端额外计算并上传**本地总体统计量估计**（或用于估计总体统计量的充分统计量，如各层特征和与平方和）
- **服务器聚合**: 
  - 参数聚合：$\theta_{global} = \sum_k \frac{n_k}{n} \theta_k$（标准 FedAvg 加权平均）
  - **Population Statistics 估计**：利用上传的充分统计量，计算全局总体均值 $\mu_{pop}$ 和总体方差 $\sigma_{pop}^2$
  - **Population Normalization**：用 $\mu_{pop}, \sigma_{pop}^2$ 替换或校准聚合后模型 BN 层的 running mean/variance
- **输出**: 经 PopNorm 校准的全局模型 $\theta_{global}^{*}$

```
客户端 k:  本地数据 D_k → 训练 → (θ_k, local_stats_k) → 上传服务器
                                              ↓
服务器:    聚合 θ_global = Σ w_k θ_k  ──→  PopNorm 模块 ──→ θ_global*
                      ↑                    ↓
              估计 μ_pop, σ²_pop ← 全局统计量计算 ← 各 local_stats_k
                                              ↓
客户端 k:  ←────────────────  下载 θ_global*  ─────────────────┘
```

PopNorm 模块的核心设计在于：不修改本地训练过程，仅在服务器端增加一个轻量级的统计量聚合与重归一化步骤，兼容现有联邦学习基础设施。

## 核心模块与公式推导

### 模块 1: 本地统计量与充分统计量上传（对应框架图：客户端→服务器链路）

**直觉**: 为了在无原始数据共享条件下估计全局总体统计量，需要客户端上传可用于聚合计算的充分统计量，而非仅上传参数。

**Baseline (FedAvg)**: 客户端仅上传模型参数 $\theta_k$，BN 层的 running mean $\hat{\mu}_k$ 和 running variance $\hat{\sigma}_k^2$ 作为参数的一部分被隐式平均：
$$\theta_{global} = \sum_{k=1}^{K} \frac{n_k}{n} \theta_k, \quad \hat{\mu}_{global}^{(FedAvg)} = \sum_{k=1}^{K} \frac{n_k}{n} \hat{\mu}_k$$
符号: $n_k$ = 客户端 $k$ 的样本数, $n = \sum_k n_k$, $\hat{\mu}_k$ = 客户端 $k$ 本地估计的 running mean。

**变化点**: FedAvg 的加权平均假设各客户端数据同分布，non-IID 下 $\hat{\mu}_k$ 有偏，导致 $\hat{\mu}_{global}$ 偏离真实总体均值。

**本文公式**: 客户端 $k$ 计算并上传特征和 $S_k$ 与平方和 $Q_k$：
$$\text{Step 1}: S_k^{(l)} = \sum_{x \in D_k} f^{(l)}(x; \theta_k), \quad Q_k^{(l)} = \sum_{x \in D_k} \left(f^{(l)}(x; \theta_k)\right)^2$$
其中 $f^{(l)}$ 为第 $l$ 层 BN 前的特征，加入此设计以支持无偏总体估计。
$$\text{Step 2}: \text{服务器聚合: } S_{global}^{(l)} = \sum_k S_k^{(l)}, \quad Q_{global}^{(l)} = \sum_k Q_k^{(l)}$$
$$\text{最终}: \mu_{pop}^{(l)} = \frac{S_{global}^{(l)}}{n}, \quad \sigma_{pop}^{2,(l)} = \frac{Q_{global}^{(l)}}{n} - \left(\mu_{pop}^{(l)}\right)^2$$

**对应消融**: 若仅使用 FedAvg 式参数平均而不上传充分统计量，总体估计偏差导致精度下降。

---

### 模块 2: Population Normalization 重归一化（对应框架图：服务器 PopNorm 模块）

**直觉**: 聚合后的全局模型参数对应于"混合分布"，其 BN 层需要重新校准以匹配总体统计量，而非本地统计量的混合。

**Baseline (标准 BN)**: 推理时使用训练期间累积的 running statistics：
$$\text{BN}(x) = \gamma \cdot \frac{x - \hat{\mu}}{\sqrt{\hat{\sigma}^2 + \epsilon}} + \beta$$
符号: $\gamma, \beta$ = 可学习缩放/偏移, $\hat{\mu}, \hat{\sigma}^2$ = running statistics, $\epsilon$ = 数值稳定性常数。

**变化点**: 联邦聚合后，参数 $\gamma, \beta$ 已被平均，但 running statistics 仍反映混合分布而非目标总体分布，导致推理时 covariate shift。

**本文公式推导**:
$$\text{Step 1}: \text{对聚合后模型第 } l \text{ 层，提取当前 BN 参数 } \gamma_{global}^{(l)}, \beta_{global}^{(l)}$$
$$\text{Step 2}: \text{用总体统计量替换 running statistics: } \hat{\mu}_{pop}^{(l)} \leftarrow \mu_{pop}^{(l)}, \quad \hat{\sigma}_{pop}^{2,(l)} \leftarrow \sigma_{pop}^{2,(l)}$$
$$\text{Step 3}: \text{重参数化保持表达能力: 调整 } \gamma, \beta \text{ 以补偿统计量变化}$$
$$\gamma_{pop}^{(l)} = \gamma_{global}^{(l)} \cdot \frac{\sqrt{\hat{\sigma}_{global}^{2,(l)} + \epsilon}}{\sqrt{\sigma_{pop}^{2,(l)} + \epsilon}}, \quad \beta_{pop}^{(l)} = \beta_{global}^{(l)} + \gamma_{global}^{(l)} \cdot \frac{\hat{\mu}_{global}^{(l)} - \mu_{pop}^{(l)}}{\sqrt{\sigma_{pop}^{2,(l)} + \epsilon}}$$
$$\text{最终}: \text{BN}_{pop}(x) = \gamma_{pop}^{(l)} \cdot \frac{x - \mu_{pop}^{(l)}}{\sqrt{\sigma_{pop}^{2,(l)} + \epsilon}} + \beta_{pop}^{(l)}$$
此推导保证：在总体统计量下，该层的输入-输出映射与聚合前各本地模型的期望行为一致。

**对应消融**: 去掉重参数化（Step 3），仅替换统计量，导致的精度损失；完整 PopNorm 恢复至最优。

---

### 模块 3: 通信高效的统计量估计（可选压缩模块）

**直觉**: 上传完整特征和 $S_k, Q_k$ 通信开销大，需设计压缩方案。

**本文公式**: 采用 sketching 或低秩近似：
$$\text{压缩上传}: \tilde{S}_k = \text{Sketch}(S_k), \quad \tilde{Q}_k = \text{Sketch}(Q_k)$$
$$\text{服务器解码}: \hat{S}_{global} = \text{DeSketch}\left(\sum_k \tilde{S}_k\right)$$
**最终**: 在可控误差 $\delta$ 下恢复总体统计量，通信开销从 $O(d)$ 降至 $O(d \cdot \frac{1}{\epsilon^2})$ 或更低（$d$ 为特征维度，$\epsilon$ 为精度参数）。

**对应消融**: 完整统计量 vs. 压缩版本精度差异。

## 实验与分析

本文在标准联邦学习图像分类基准上评估 PopNorm，主要对比 FedAvg 及现有处理 BN 的联邦学习方法。

{{TBL:result}}

实验设置涵盖 CIFAR-10/100 与 ImageNet 的子集，采用 Dirichlet 分布模拟不同异构程度（$\alpha \in \{0.1, 0.5, 1.0\}$）。核心结果显示：在高度异构场景（$\alpha=0.1$）下，PopNorm 相比 FedAvg 的最终测试精度提升显著（具体数值。这一提升主要来源于 PopNorm 消除了 BN 层因统计量偏差导致的分布偏移，使全局模型在推理阶段能够正确使用与训练分布匹配的归一化参数。在中等异构（$\alpha=0.5$）下，增益有所收窄但仍保持正向，说明总体统计量的价值随数据异构程度增加而凸显。

{{TBL:ablation}}

消融实验重点验证三个组件：（1）去掉充分统计量上传、仅 FedAvg 式聚合，精度回落至基线水平，验证了显式估计总体统计量的必要性；（2）去掉重参数化（仅替换统计量不调整 $\gamma, \beta$），精度损失，证明 Step 3 的代数校准对保持层间表达能力至关重要；（3）采用压缩统计量上传，在通信开销降低% 时精度损失控制在以内，验证了实用性。

公平性检查：本文对比的 FedAvg 为最广泛使用的联邦学习基线，但未与同期更复杂的个性化联邦学习方法（如 FedPer、pFedMe）直接比较——这些方法在极端 non-IID 下可能通过本地个性化头获得更高精度，但牺牲了全局模型的统一性。PopNorm 的优势在于保持单全局模型架构，无需增加客户端侧复杂度。作者披露的局限包括：总体统计量估计假设客户端本地 epoch 数足够使本地模型逼近局部最优，否则上传的统计量本身带有优化噪声；此外，对于非常深的网络，逐层统计量聚合的通信累积仍需压缩技术缓解。

## 方法谱系与知识库定位

**方法家族**: 联邦优化（Federated Optimization）→ 异构鲁棒聚合 → 归一化层感知聚合

**父方法**: FedAvg（McMahan et al., 2017），标准联邦平均算法。PopNorm 继承其参数加权聚合框架，但在 slot `normalization_statistics` 上引入总体统计量估计与重校准机制。

**直接基线差异**:
- **FedAvg**: 隐式混合本地 BN statistics，non-IID 下失效；PopNorm 显式分离参数聚合与统计量校准。
- **FedBN**（Li et al., 2021）: 放弃 BN 层全局聚合，保留本地 BN；PopNorm 坚持全局 BN 一致性，通过统计量传播实现真正的协同归一化。
- **SCAFFOLD**（Karimireddy et al., 2020）: 用控制变量修正本地更新方向；PopNorm 不修改本地优化，聚焦服务器端统计量校正，二者正交可结合。

**后续方向**:
1. 与其他归一化层（GroupNorm、SwitchableNorm）结合，验证 PopNorm 原理的普适性；
2. 将总体统计量估计与个性化联邦学习融合，实现"全局归一化 + 本地个性化头"的分层架构；
3. 探索差分隐私下的安全统计量聚合，解决上传充分统计量带来的隐私泄露风险。

**知识库标签**: 
- modality: 计算机视觉 / 图像分类
- paradigm: 联邦学习 / 分布式训练
- scenario: 数据异构（non-IID）/ 隐私保护
- mechanism: 批量归一化（BatchNorm）/ 总体统计量估计 / 重参数化
- constraint: 通信高效 / 无原始数据共享
