---
title: 'AFL: A Single-Round Analytic Approach for Federated Learning with Pre-trained Models'
type: paper
paper_level: C
venue: CVPR
year: 2025
paper_link: null
aliases:
- 单轮解析式联邦学习预训练模型适配
- AFL (Analytic Fe
- AFL (Analytic Federated Learning)
acceptance: poster
cited_by: 9
method: AFL (Analytic Federated Learning)
---

# AFL: A Single-Round Analytic Approach for Federated Learning with Pre-trained Models

**Topics**: [[T__Federated_Learning]] | **Method**: [[M__AFL]] | **Datasets**: [[D__CIFAR-10]], [[D__CIFAR-100]] (其他: FedFisher)

| 中文题名 | 单轮解析式联邦学习预训练模型适配 |
| 英文题名 | AFL: A Single-Round Analytic Approach for Federated Learning with Pre-trained Models |
| 会议/期刊 | CVPR 2025 (poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2405.16240) · [Code] · [Project] |
| 主要任务 | 联邦学习（Federated Learning）、图像分类、非IID数据下的分布式训练 |
| 主要 baseline | FedAvg, FedProx, MOON, FedGen, FedDyn, FedNTD, FedDisco, FedFisher |

> [!abstract]
> 因为「联邦学习多轮通信开销大且非IID数据导致收敛不稳定」，作者在「ACIL 解析学习框架」基础上改了「将迭代梯度训练替换为单轮闭式伪逆求解，并设计递归精确聚合与可移除中介（RI）机制」，在「CIFAR-10/100 和 Tiny-ImageNet 非IID设置」上取得「CIFAR-10 NIID-1 80.75%、CIFAR-100 NIID-1 58.56%，且对异质性变化零波动」

- **关键性能 1**: CIFAR-10 NIID-1 (α=0.1, α=0.01) 上 AFL 达到 80.75%，所有设置下标准差为 0，FedDyn 在 α=0.01 时显著下降
- **关键性能 2**: CIFAR-100 NIID-1 α=0.1 上 AFL 58.56%，优于 FedAvg 56.62% 和 FedDyn 57.55%；FedDyn 在 α=0.01 暴跌至 36.12% 而 AFL 保持稳定
- **关键性能 3**: 相比单轮 baseline FedFisher 的 19.31%，AFL 达到 35.87%，相对提升 85.8%

## 背景与动机

联邦学习（Federated Learning, FL）旨在让多个客户端在不共享原始数据的前提下协同训练模型。然而，实际场景中数据往往呈非独立同分布（Non-IID）：例如，不同医院的医疗影像只包含特定病种，或不同用户的手机照片集中于特定场景。这种异质性导致传统 FL 方法收敛缓慢、通信轮次多、且最终模型精度波动大。

现有方法如何应对这一问题？**FedAvg** 通过周期性加权平均本地模型来缓解漂移，但在强非IID下仍需数十至数百轮通信。**FedProx** 在本地目标中加入近端正则项约束参数偏离全局模型，增加了计算开销却未根本解决通信瓶颈。**MOON** 引入模型级对比学习，利用局部与全局表示的相似性正则化训练，但依赖多轮迭代中的负样本构造。**FedDyn** 采用动态正则化自适应调整本地目标，虽在部分设置表现最优，却对异质性程度敏感——当 Dirichlet 参数 α 从 0.1 降至 0.01 时精度可能暴跌超过 20 个百分点。

这些方法的共同瓶颈在于：**它们都依赖基于随机梯度下降（SGD）的迭代优化**。每一轮本地训练引入随机性，多轮通信累积误差，且非IID数据放大了客户端间的梯度冲突。更关键的是，预训练大模型时代，骨干网络已高度优化，FL 只需适配轻量分类头——却仍沿用端到端迭代微调，造成计算与通信的冗余浪费。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/c8c8e2a3-ec21-4973-a00d-32994f10bf5d/figures/Figure_1.png)
*Figure 1 (pipeline): Overview of the AFL. During the local step, we train one local classifier from Xi and load Wk as the basis on the server to build basis G.*



本文的核心动机由此浮现：能否将本地训练从迭代优化转化为**闭式解析求解**，并用**精确的代数聚合**替代启发式加权平均，从而实现**单轮通信、零随机性、对非IID免疫**的联邦学习？AFL 正是这一思路的首次系统实现。

## 核心创新

核心洞察：**线性分类头的最小二乘最优解可通过伪逆闭式获得，且多个客户端伪逆解的聚合存在精确的递归代数结构**，因为 Woodbury 矩阵恒等式允许将堆叠矩阵的伪逆分解为协方差加权的本地解组合，从而使**单轮联邦学习在数学上严格等价于集中式最优解**成为可能。

| 维度 | Baseline (FedAvg/ACIL) | 本文 (AFL) |
|:---|:---|:---|
| 本地训练方式 | 多轮 SGD 迭代优化 | 单步 Moore-Penrose 伪逆闭式求解 |
| 聚合机制 | 按样本数加权平均（启发式） | 递归协方差加权精确聚合（代数最优） |
| 通信轮次 | 数十至数百轮 | **单轮** |
| 正则化处理 | γ 为需调优的超参数 | RI 过程精确消除 γ 影响，**非超参数** |
| 随机性 | 梯度下降引入显著方差 | 确定性算法，**标准差为零** |
| 非IID鲁棒性 | 随异质性增强而退化 | 精度对 α 变化**零波动** |

## 整体框架



AFL 的整体流程可概括为「冻结特征提取 → 本地解析求解 → 递归精确聚合 → 可选偏差消除」四阶段：

1. **特征提取（Feature Extraction）**：输入原始图像 $\mathcal{X}_{k,j}$，经冻结的预训练骨干网络（ResNet-18/VGG11/ViT-B-16）提取特征 $\mathbf{x}_{k,j} = f_{\text{backbone}}(\mathcal{X}_{k,j}, \boldsymbol{\Theta})$。骨干参数 $\boldsymbol{\Theta}$ 全程冻结，所有客户端共享同一特征空间。

2. **本地解析求解（Local Analytic Solver）**：客户端 $k$ 将特征堆叠为矩阵 $\mathbf{X}_k \in \mathbb{R}^{N_k \times d_e}$，标签构造 one-hot 矩阵 $\mathbf{Y}_k \in \mathbb{R}^{N_k \times C}$，直接计算本地分类器 $\hat{\mathbf{W}}_k = \mathbf{X}_k^+ \mathbf{Y}_k$（或带正则化的 $\hat{\mathbf{W}}_k^\text{r}$）。**此步骤替代了传统 FL 的多轮 SGD 训练**。

3. **递归服务器聚合（Recursive Server Aggregation）**：服务器接收客户端上传的协方差矩阵 $\mathbf{C}_k = \mathbf{X}_k^\text{top} \mathbf{X}_k$ 与本地解 $\hat{\mathbf{W}}_k$，通过递归公式精确合并：$\mathbf{C}_{\text{agg},k} = \mathbf{C}_{\text{agg},k-1} + \mathbf{C}_k$，并计算全局权重矩阵 $\boldsymbol{\mathcal{W}}_{\text{agg}}, \boldsymbol{\mathcal{W}}_k$ 以组合前 $k-1$ 个客户端的聚合解与第 $k$ 个客户端的本地解。**无需迭代通信，一次完成**。

4. **可移除中介过程（RI Process, 可选）**：若本地使用了岭正则化（$\gamma > 0$）保证数值稳定性，服务器在聚合后执行 $\hat{\mathbf{W}}_{\text{agg},k}^\text{r} = (\mathbf{C}_{\text{agg},k}^\text{r})^{-1} \mathbf{C}_{\text{agg},k} \hat{\mathbf{W}}_{\text{agg},k}$，**精确消除 $\gamma$ 的累积偏差**，恢复无正则化的全局最优解。

```
Images → [Frozen Backbone] → Features X_k, Labels Y_k
                                    ↓
                         [Local Analytic Solver]
                         W_k = X_k^+ Y_k  (or W_k^r with γ)
                                    ↓
                         Send (C_k, W_k) to Server
                                    ↓
                    [Recursive Server Aggregation]
                    C_agg,k = Σ C_i,  W_agg,k = f(W_agg,k-1, W_k)
                                    ↓
                         [RI Process if γ > 0]
                         Remove γ bias exactly
                                    ↓
                              Final Global Classifier
```

## 核心模块与公式推导

### 模块 1: 本地解析求解器（对应框架图 Local Analytic Solver）

**直觉**: 预训练骨干已提供高质量特征，线性分类头的最优解无需梯度下降迭代，可直接通过最小二乘闭式获得。

**Baseline 公式 (FedAvg 本地 SGD)**:
$$\mathbf{W}_k^{(t+1)} = \mathbf{W}_k^{(t)} - \eta \nabla \mathcal{L}_{\text{CE}}(\mathbf{W}_k^{(t)}; \mathbf{X}_k, \mathbf{Y}_k)$$
符号: $\mathbf{W}_k^{(t)}$ = 第 $t$ 步本地参数, $\eta$ = 学习率, $\mathcal{L}_{\text{CE}}$ = 交叉熵损失。

**变化点**: SGD 迭代引入随机性，需多轮通信收敛；且非IID数据导致本地最优与全局最优冲突。本文假设特征到标签的映射为线性回归问题，改用 Frobenius 范数最小化。

**本文公式（推导）**:
$$\text{Step 1}: \quad \mathcal{L}(\mathbf{W}_k) = \|\mathbf{Y}_k - \mathbf{X}_k \mathbf{W}_k\|_F^2 \quad \text{将分类转化为最小二乘拟合}$$
$$\text{Step 2}: \quad \hat{\mathbf{W}}_k = \mathbf{X}_k^+ \mathbf{Y}_k = (\mathbf{X}_k^\text{top} \mathbf{X}_k)^{-1} \mathbf{X}_k^\text{top} \mathbf{Y}_k \quad \text{Moore-Penrose 伪逆直接给出闭式最优}$$
$$\text{最终}: \hat{\mathbf{W}}_k = \mathbf{C}_k^{-1} \mathbf{X}_k^\text{top} \mathbf{Y}_k, \quad \mathbf{C}_k = \mathbf{X}_k^\text{top} \mathbf{X}_k$$
符号: $\mathbf{X}_k \in \mathbb{R}^{N_k \times d_e}$ = 特征矩阵, $\mathbf{Y}_k \in \mathbb{R}^{N_k \times C}$ = one-hot 标签矩阵, $\mathbf{C}_k \in \mathbb{R}^{d_e \times d_e}$ = 协方差矩阵, $d_e$ = 特征维度, $C$ = 类别数。

**对应消融**: Table 3 显示当 $N_k < d_e$（如 K=500,1000 客户端划分）且 $\gamma=0$ 时，矩阵病态导致 CIFAR-100 精度暴跌至 1.11% (K=500) 和 0.75% (K=1000)，验证纯解析求解需配合正则化。

### 模块 2: 递归分块伪逆聚合（对应框架图 Recursive Server Aggregation）

**直觉**: 两个客户端的联合伪逆可分解为各自伪逆的协方差加权组合，此结构可递归扩展至任意数量客户端，实现增量式精确聚合。

**Baseline 公式 (FedAvg 服务器聚合)**:
$$\mathbf{W}_{\text{global}} = \sum_{k} \frac{N_k}{N} \mathbf{W}_k$$
符号: $N_k$ = 客户端 $k$ 样本数, $N = \sum_k N_k$。此为按样本数的启发式加权平均，**非统计最优**。

**变化点**: FedAvg 的加权平均忽略数据协方差结构，非IID下本地解差异大时聚合质量差。本文利用伪逆的代数结构，推导出**精确的最优线性组合权重**。

**本文公式（推导）**:
$$\text{Step 1}: \quad \mathbf{X}^\text{dagger} = \begin{bmatrix} \mathbf{X}_u \\ \mathbf{X}_v \end{bmatrix}^\text{dagger} = \begin{bmatrix} \bar{\mathbf{U}} & \bar{\mathbf{V}} \end{bmatrix} \quad \text{将两客户端堆叠矩阵的伪逆分块}$$
$$\text{Step 2}: \quad \bar{\mathbf{U}} = \left[\mathbf{I} - \mathbf{R}_u \mathbf{C}_v + \mathbf{R}_u \mathbf{C}_v (\mathbf{C}_u + \mathbf{C}_v)^{-1} \mathbf{C}_v \right] \mathbf{X}_u^\text{dagger} \quad \text{应用 Woodbury 恒等式简化，其中 } \mathbf{R}_u = \mathbf{C}_u^{-1}$$
$$\text{Step 3}: \quad \hat{\mathbf{W}} = \bar{\mathbf{U}} \mathbf{Y}_u + \bar{\mathbf{V}} \mathbf{Y}_v = \boldsymbol{\mathcal{W}}_u \hat{\mathbf{W}}_u + \boldsymbol{\mathcal{W}}_v \hat{\mathbf{W}}_v \quad \text{重组为本地解的加权组合}$$
$$\text{其中权重}: \boldsymbol{\mathcal{W}}_u = \mathbf{I} - \mathbf{R}_u \mathbf{C}_v + \mathbf{R}_u \mathbf{C}_v (\mathbf{C}_u + \mathbf{C}_v)^{-1} \mathbf{C}_v$$
$$\text{Step 4 (递归扩展)}: \quad \hat{\mathbf{W}}_{\text{agg},k} = \boldsymbol{\mathcal{W}}_{\text{agg}} \hat{\mathbf{W}}_{\text{agg},k-1} + \boldsymbol{\mathcal{W}}_k \hat{\mathbf{W}}_k$$
$$\text{最终递归权重}: \begin{cases} \boldsymbol{\mathcal{W}}_{\text{agg}} = \mathbf{I} - \mathbf{C}_{\text{agg},k-1}^{-1} \mathbf{C}_k (\mathbf{I} - \mathbf{C}_{\text{agg},k}^{-1} \mathbf{C}_k) \\ \boldsymbol{\mathcal{W}}_k = \mathbf{I} - \mathbf{C}_k^{-1} \mathbf{C}_{\text{agg},k-1} (\mathbf{I} - \mathbf{C}_{\text{agg},k}^{-1} \mathbf{C}_{\text{agg},k-1}) \end{cases}, \quad \mathbf{C}_{\text{agg},k} = \sum_{i=1}^{k} \mathbf{C}_i$$

**对应消融**: 无直接消融（此为方法核心，移除则退化为 FedAvg），但 Figure 2/3 显示 AFL 单轮收敛即达最终精度，而 FedAvg 需多轮缓慢收敛。

### 模块 3: 可移除中介（RI）过程（对应框架图 RI Process）

**直觉**: 本地正则化保证数值稳定，但聚合后正则化项累积为 $k\gamma \mathbf{I}$ 偏差；设计代数逆运算精确"挤出"此偏差，使 $\gamma$ 成为可任意选取的"中介"而非超参数。

**Baseline 公式 (标准岭回归)**:
$$\hat{\mathbf{W}}_k^\text{r} = (\mathbf{X}_k^\text{top} \mathbf{X}_k + \gamma \mathbf{I})^{-1} \mathbf{X}_k^\text{top} \mathbf{Y}_k = (\mathbf{C}_k + \gamma \mathbf{I})^{-1} \mathbf{X}_k^\text{top} \mathbf{Y}_k$$
符号: $\gamma$ = 岭正则化系数，需交叉验证调优。

**变化点**: 标准方法中 $\gamma$ 是敏感超参数，且联邦场景下 $k$ 个客户端聚合后总正则化为 $k\gamma \mathbf{I}$，严重偏差。本文发现可通过累积正则化协方差矩阵的逆运算精确消除此效应。

**本文公式（推导）**:
$$\text{Step 1}: \quad \mathcal{L}(\mathbf{W}_k^\text{r}) = \|\mathbf{Y}_k - \mathbf{X}_k \mathbf{W}_k^\text{r}\|_F^2 + \gamma \|\mathbf{W}_k^\text{r}\|_F^2 \quad \text{本地岭正则化目标}$$
$$\text{Step 2}: \quad \hat{\mathbf{W}}^\text{r} = (\mathbf{C}_u + \mathbf{C}_v + 2\gamma \mathbf{I})^{-1} (\mathbf{X}_u^\text{top} \mathbf{Y}_u + \mathbf{X}_v^\text{top} \mathbf{Y}_v) \quad \text{两客户端正则化全局解，显见 } 2\gamma \mathbf{I} \text{ 累积偏差}$$
$$\text{Step 3}: \quad \mathbf{C}_{\text{agg},k}^\text{r} = \mathbf{C}_{\text{agg},k} + k\gamma \mathbf{I} = \sum_{i=1}^{k} (\mathbf{C}_i + \gamma \mathbf{I}) = \sum_{i=1}^{k} \mathbf{C}_i^\text{r} \quad \text{累积正则化协方差结构}$$
$$\text{最终 (RI 过程)}: \quad \hat{\mathbf{W}}_{\text{agg},k}^\text{r} = (\mathbf{C}_{\text{agg},k}^\text{r})^{-1} \mathbf{C}_{\text{agg},k} \hat{\mathbf{W}}_{\text{agg},k} \quad \text{左乘 } (\mathbf{C}_{\text{agg},k}^\text{r})^{-1} \mathbf{C}_{\text{agg},k} \text{ 精确消除 } k\gamma \mathbf{I} \text{ 影响}$$

**对应消融**: Table 3 显示，**有 RI 时** $\gamma \in \{0.1, 1, 10, 100\}$ 结果恒为 58.56%；**无 RI 且 $\gamma=100$** 时降至 49.62%（-9.00%），**无 RI 且 $\gamma=0$ 且 K=1000** 时暴跌至 0.75%（-57.81%）。

## 实验与分析


![Table 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/c8c8e2a3-ec21-4973-a00d-32994f10bf5d/figures/Table_2.png)
*Table 2 (comparison): The top-1 classification accuracy (%) of AFL and FedProto.*



本文在 CIFAR-10、CIFAR-100 和 Tiny-ImageNet 上评估 AFL，采用 Dirichlet 分布构造非IID划分（NIID-1: 按标签分布偏斜；NIID-2: 更极端的异质设置），预训练 ResNet-18 骨干冻结。核心结果如 Table 2 所示：AFL 在 CIFAR-10 NIID-1 上达到 **80.75%** top-1 精度，且该数值在 α=0.1 和 α=0.01 下**完全不变**（标准差为 0）；相比之下，FedDyn 虽在 NIID-2 略优于 AFL，但在 α=0.01 时从 57.55% 暴跌至 36.12%。CIFAR-100 NIID-1 α=0.1 上 AFL 为 **58.56%**，高于 FedAvg 的 56.62% 和 FedDyn 的 57.55%，且当异质性增强时 AFL 保持稳定而 FedDyn 崩溃。与同为单轮方法的 FedFisher 相比，AFL 的 35.87% 对其 19.31% 实现 **85.8% 相对提升**，验证了解析式优于基于 Fisher 信息矩阵的近似聚合。


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/c8c8e2a3-ec21-4973-a00d-32994f10bf5d/figures/Figure_2.png)
*Figure 2 (result): Accuracy versus communication rounds.*



Figure 2 和 Figure 3 进一步展示收敛曲线：AFL 在**第 1 轮即达到最终精度**，而 FedAvg、FedProx、MOON 等方法需 50-100+ 轮缓慢收敛，且波动明显。这一"单轮收敛"特性在通信受限场景（如边缘设备、跨机构医疗协作）具有决定性优势。


![Table 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/c8c8e2a3-ec21-4973-a00d-32994f10bf5d/figures/Table_3.png)
*Table 3 (ablation): The overall top-1 accuracy (%) on CIFAR-100 with different numbers of local epochs E.*



消融实验聚焦 RI 过程与正则化敏感性（Table 3）。关键发现：当 K=1000 客户端（极端数据碎片化，$N_k \ll d_e$）时，**去掉 RI 且设 γ=0** 导致矩阵病态，CIFAR-100 精度仅 0.75%；**加 RI 后任意 γ 均恢复至 58.56%**。若去掉 RI 但保留 γ=100，精度降至 49.62%，损失 8.94 个百分点。Table 4 验证骨干泛化：ViT-T/16 和 ResNet-10 上 AFL 同样有效，证明方法不依赖特定架构。

**公平性检验**: 本文比较了 7 个 FL baseline，但缺少 SCAFFOLD、FedOpt、Mime 等方差缩减方法，以及 pFedMe、Ditto 等个性化 FL 方法。所有方法均使用冻结预训练骨干，这一设定可能削弱部分方法的相对优势——作者亦承认 FedAvg 在此设定下已具竞争力，暗示预训练质量对结果影响显著。"零标准差"声明在数学上正确（确定性算法），但实际意义有限。FedFisher 的 19.31% 可能源于不同实验配置，直接对比需谨慎。

## 方法谱系与知识库定位

AFL 属于**解析学习（Analytic Learning）** 方法族，直接继承自 **ACIL (Analytic Class-Incremental Learning)**。ACIL 在集中式持续学习场景中首次证明伪逆可实现免迭代的分类器更新；AFL 将其扩展至联邦分布式场景，核心改动包括：(1) training_recipe：单轮伪逆替代多轮 SGD；(2) objective：最小二乘替代交叉熵，引入可移除的岭正则化；(3) inference_strategy：单轮通信替代迭代同步；(4) credit_assignment：递归协方差加权精确聚合替代梯度回传；(5) architecture：冻结预训练骨干 + 可解析线性头。

**直接 Baseline 差异**：
- **FedAvg**: 多轮迭代 + 样本加权平均 → AFL 单轮闭式 + 协方差最优加权
- **FedFisher**: 单轮但基于 Fisher 信息矩阵近似 → AFL 单轮且精确代数聚合
- **FedDyn**: 动态正则化迭代优化 → AFL 无需迭代，正则化仅作数值中介

**后续方向**: (1) 将解析聚合扩展至非线性头（如 MLP、注意力层）的近似解析解；(2) 结合差分隐私机制保护上传的 $(\mathbf{C}_k, \hat{\mathbf{W}}_k)$ 对；(3) 探索更大规模预训练模型（如 CLIP、SAM）上的联邦适配效率。

**标签**: 模态=视觉/图像 | 范式=联邦学习 + 解析优化 | 场景=非IID分布式训练、通信受限边缘计算 | 机制=Moore-Penrose伪逆、递归分块矩阵聚合、可移除正则化偏差 | 约束=单轮通信、冻结预训练骨干、线性分类头

