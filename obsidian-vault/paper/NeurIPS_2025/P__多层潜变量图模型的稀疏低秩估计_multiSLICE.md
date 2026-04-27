---
title: A Latent Multilayer Graphical Model For Complex, Interdependent Systems
type: paper
paper_level: A
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 多层潜变量图模型的稀疏低秩估计
- multiSLICE
acceptance: Poster
method: multiSLICE
modalities:
- tabular
- neuroimaging
paradigm: supervised
---

# A Latent Multilayer Graphical Model For Complex, Interdependent Systems

**Topics**: [[T__Graph_Learning]] | **Method**: [[M__multiSLICE]] | **Datasets**: Simulation 1, Simulation 2, Multimodal neuroimaging, Multimodal neuroimaging - Unfamiliar faces

> [!tip] 核心洞察
> multiSLICE, a multilayer sparse + low-rank inverse covariance estimation method, enables accurate recovery of interlayer dependencies by bridging latent variable Gaussian graphical models with multilayer networks.

| 中文题名 | 多层潜变量图模型的稀疏低秩估计 |
| 英文题名 | A Latent Multilayer Graphical Model For Complex, Interdependent Systems |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/) · [Code](https://github.com/) · [Project](https://) |
| 主要任务 | Graph Learning, Multilayer Network Estimation |
| 主要 baseline | MLGGM, coglasso, CNJGL, BJEMGM, LRGQ, SLICE |

> [!abstract] 因为「真实系统是多层网络且层间存在复杂依赖，但现有方法无法处理不规则采样和异质维度」，作者在「SLICE (Chandrasekaran et al., 2012)」基础上改了「将单层稀疏+低秩精度矩阵分解扩展为多层块对角稀疏S与共享低秩L的联合结构，并推导层特异缩放样本量理论」，在「多模态神经影像 (sMRI/MEG/fMRI)」上取得「Q(Ŝ)=0.17 vs LRGQ 0.112, H(L̂)=0.626 vs LRGQ 1.29」。

- **模块度 Q(Ŝ)**：multiSLICE 在 Famous faces 任务上达到 0.17，相对 LRGQ (0.112) 提升 51.8%，相对 BJEMGM (0.084) 提升 102.4%
- **多层冯·诺依曼熵 H(L̂)**：multiSLICE 为 0.626，较 LRGQ (1.29) 降低 51.5%（越低表示层间结构越有序）
- **理论验证**：Figure 6 显示 S* 和 L* 的恢复概率曲线在缩放样本量 n'_S = n_α/(s_α log p_α) 和 n'_L = n_α/(C₁ log p_α) 下坍缩到共同相变点，验证 Lemma 9.1

## 背景与动机

许多真实世界系统天然具有多层结构：人脑同时存在结构连接（sMRI）、功能连接（fMRI）和电生理信号（MEG），社交网络中不同平台形成交互层，金融系统中多种资产类别跨市场联动。这些层并非独立——fMRI 的慢变血流动力学信号与 MEG 的快变电生理活动存在物理耦合，但现有统计方法难以准确估计这种层间依赖关系。

现有方法如何处理这一问题？**SLICE** (Chandrasekaran et al., 2012) 将单层精度矩阵分解为稀疏 S（观测变量间的直接边）和低秩 L（潜变量诱导的边际依赖），但仅限单层设置。**MLGGM** 通过惩罚最大似然联合估计多层 GGMs，却要求各层样本量 n_α 严格相等。**coglasso** 实现协作式图形 lasso，同样受限于等样本假设且对层数敏感。**CNJGL** 和 **BJEMGM** 虽可处理多图联合估计，但要求变量维度 p_α 相同，实际应用中需通过 SVD 投影到最小公共子空间，造成信息损失。

这些方法的共同短板在于：**无法同时处理 (1) 异质样本量 n_α ≠ n_β，(2) 异质维度 p_α ≠ p_β，(3) 层间共享的潜变量结构**。当神经影像实验中 MEG 有 52 个传感器、MRI 有 68 个脑区、且各模态有效试次数不同时，上述方法或失效或需降维妥协。本文提出 multiSLICE，通过块对角稀疏结构保留层内特异性，以共享低秩矩阵捕捉跨层潜变量耦合，并推导了层特异缩放样本量下的恢复理论。


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/4c77dcae-b0ad-4ab5-8e2c-f5057be8cdc2/figures/fig_001.png)
*Figure: A graphic of a weighted adjacency matrix (left), and the associated multilayer graph (right).*



## 核心创新

核心洞察：**单层精度矩阵的稀疏+低秩分解可以自然扩展为多层设置，只需让稀疏分量 S 保持块对角（各层独立）、而让低秩分量 L 跨层共享，因为层间依赖本质上是由少数共同潜变量驱动的低维耦合，从而使异质采样和异质维度的联合估计成为可能。**

| 维度 | Baseline (SLICE / MLGGM) | 本文 (multiSLICE) |
|:---|:---|:---|
| 精度矩阵结构 | Θ = S + L，单层整体 | Θ = S + L，S = block-diag(S₁,...,Sₗ) 块对角，L 跨层共享低秩 |
| 样本量要求 | 各层 n_α 必须相等 | 允许 n_α 任意不同，理论显式依赖各层 s_α, p_α |
| 维度要求 | 各层 p_α 必须相同（或需投影） | 允许 p_α 任意不同，无需降维 |
| 优化目标 | 单层图形 lasso 或等惩罚联合估计 | 分层正则化似然，对 S 和 L 施加不同惩罚，层特异缩放 |

## 整体框架


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/4c77dcae-b0ad-4ab5-8e2c-f5057be8cdc2/figures/fig_002.png)
*Figure: An illustration of the data-generating (forward) and parameter estimation (reverse) processes*



multiSLICE 的处理流程如下：

1. **多层数据输入**：观测矩阵 X₁, ..., Xₗ，其中第 α 层为 n_α × p_α，允许 n_α 和 p_α 任意不同。输入模块仅做中心化和尺度标准化。

2. **联合精度矩阵估计（核心创新）**：求解正则化最大似然问题，输出联合精度矩阵的分解 Θ̂ = Ŝ + L̂。此模块替代了传统的分层 SLICE 或等样本联合估计。

3. **稀疏分量提取 Ŝ**：从 Θ̂ 中提取块对角部分，每块 Ŝ_α 对应该层内的稀疏精度矩阵，反映观测变量间的直接条件依赖边。

4. **低秩分量提取 L̂**：从 Θ̂ 中提取共享的低秩矩阵，其非零结构揭示跨层潜变量的耦合模式，即层间依赖的"超邻接"关系。

5. **理论验证与评估**：通过缩放样本量 n'_S 和 n'_L 检验恢复相变，并用模块度 Q(Ŝ) 评估层内社区结构、用多层冯·诺依曼熵 H(L̂) 评估层间有序性。

```
[X₁(n₁×p₁), ..., Xₗ(nₗ×pₗ)] 
    → 中心化/标准化
    → multiSLICE 估计: min -log det Θ + tr(SΘ) + λ₁‖S‖₁ + λ_*‖L‖_*
    → Θ̂ = Ŝ + L̂
    → Ŝ = block-diag(Ŝ₁,...,Ŝₗ)  [层内稀疏图]
    → L̂  [共享低秩，层间潜变量耦合]
    → 评估: Q(Ŝ), H(L̂)
```

## 核心模块与公式推导

### 模块 1: 多层精度矩阵分解（对应框架图步骤 2）

**直觉**: 层内依赖是稀疏的（每脑区仅与少数其他脑区直接连接），而层间依赖由少数共同潜变量驱动（如全局认知状态），故应为低秩。

**Baseline 公式** (SLICE, Chandrasekaran et al.):
$$\Theta = S + L, \quad S \in \mathbb{R}^{p \times p} \text{ 稀疏}, \quad L \in \mathbb{R}^{p \times p} \text{ 低秩}$$
符号: Θ = 精度矩阵（逆协方差），S = 稀疏分量（条件依赖图），L = 低秩分量（潜变量诱导的边际依赖），λ₁, λ_* = 正则化参数。

**变化点**: 单层 SLICE 无法处理多层数据；MLGGM 虽多层但将 Θ 视为整体稀疏+低秩，未区分层内/层间结构，且要求 n_α ≡ n, p_α ≡ p。

**本文公式（推导）**:
$$\text{Step 1}: \quad S = \text{block-diag}(S_1, S_2, \ldots, S_l), \quad S_\alpha \in \mathbb{R}^{p_\alpha \times p_\alpha} \text{ 稀疏}$$
$$\text{Step 2}: \quad L \in \mathbb{R}^{P \times P} \text{ 共享低秩}, \quad P = \sum_\alpha p_\alpha$$
$$\text{最终}: \quad \Theta = \underbrace{\begin{bmatrix} S_1 & & \\ & \ddots & \\ & & S_l \end{bmatrix}}_{\text{层内稀疏}} + \underbrace{L}_{\text{跨层低秩}}$$
S 的块对角约束保证层间无直接边，所有跨层信息通过 L 的低秩结构传递。L 的共享性意味着各层受同一组潜变量影响。

**对应消融**: Figure 3 显示增加层数 l 对 multiSLICE 的 F1 影响远小于 coglasso，说明块对角+共享低秩结构对多层扩展的鲁棒性。

---

### 模块 2: 层特异缩放样本量与正则化目标（对应框架图步骤 2-5）

**直觉**: 不同层的稀疏度 s_α 和维度 p_α 不同，恢复难度各异，需用"有效样本量"统一理论分析。

**Baseline 公式** (Wainwright 2009 / 单层图形模型):
$$n \geq C \cdot s \log p \quad \text{(稀疏分量恢复条件)}$$
符号: n = 样本量，s = 稀疏度（非零元个数），p = 维度，C = 常数。

**变化点**: 单层理论假设固定 (n, s, p)；多层设置中各层 (n_α, s_α, p_α) 不同，且低秩分量 L 的恢复条件与稀疏分量不同，需分别推导缩放公式。

**本文公式（推导）**:
$$\text{Step 1（稀疏分量缩放）}: \quad n'_S = \frac{n_\alpha}{s_\alpha \log p_\alpha}$$
$$\text{Step 2（低秩分量缩放）}: \quad n'_L = \frac{n_\alpha}{C_1 \log p_\alpha}$$
$$\text{其中 } C_1 \text{ 为 Lemma 9.1 定义的常数，依赖于 } R(L^*) \text{（} L^* \text{ 的秩）}$$
$$\text{最终目标函数}: \quad \min_{\Theta = S + L \text{succ} 0} -\sum_{\alpha=1}^l \frac{n_\alpha}{n} \left[\log\det\Theta_\alpha - \text{tr}(\hat{\Sigma}_\alpha \Theta_\alpha)\right] + \lambda_1 \|S\|_1 + \lambda_* \|L\|_*$$
层权重 n_α/n 自动调整异质样本量的贡献；‖S‖₁ 促进块内稀疏，‖L‖_*（核范数）促进跨层低秩。

**对应消融**: Figure 6 显示 S* 和 L* 的恢复概率在原始样本量 n_α 下分散，但在 n'_S 和 n'_L 缩放后曲线坍缩到共同相变点，验证理论精确性。

---

### 模块 3: 多层网络评估指标（对应框架图步骤 5）

**直觉**: 传统单图指标无法区分层内社区结构与层间耦合复杂度，需设计多层专用指标。

**本文公式**:
$$Q(\hat{S}) = \frac{1}{2m} \text{tr}(C^T B C), \quad B = \hat{S} - \frac{kk^T}{2m}$$
（标准 Newman 模块度，衡量层内社区分离度；越高越好）

$$H(\hat{L}) = -\sum_{i=1}^{P} \lambda_i^{\hat{L}} \ln(\lambda_i^{\hat{L}})$$
（多层冯·诺依曼熵，衡量 L̂ 特征值分布的均匀性；越低表示层间结构越有序、潜变量耦合越规则）

**对应消融**: Table 2 显示 multiSLICE 的 Q(Ŝ)=0.17 显著高于 CNJGL (0.107)、BJEMGM (0.084)、LRGQ (0.112)；H(L̂)=0.626 显著低于 LRGQ (1.29)，表明同时优化了层内社区清晰度和层间有序性。

## 实验与分析



本文在两类基准上验证 multiSLICE：**仿真研究**（Simulation 1 和 2）与**多模态神经影像实验**（Wakeman & Henson 2015 数据集，含 sMRI、MEG、fMRI 三种模态）。

**仿真结果**：在 Simulation 1 中，multiSLICE 的 F1 score 和主特征向量恢复精度 sin θ(û₁, u*₁) 随样本量增加单调改善，且层数 l 的变化对其影响有限；MLGGM 表现次之，但对层数更敏感。coglasso 在 Simulation 2（变化 R(L*)）中表现波动剧烈，说明其对层数和低秩秩的联合变化不稳定，而 multiSLICE 保持稳健。Figure 3 和 Figure 4 定量展示了这一差异。

**神经影像结果**：在 Famous faces 任务中，multiSLICE 的模块度 Q(Ŝ) = 0.17，较次优方法 LRGQ (0.112) 相对提升 51.8%，较 BJEMGM (0.084) 提升 102.4%。在层间有序性指标上，multiSLICE 的 H(L̂) = 0.626，较 LRGQ (1.29) 降低 51.5%。这一优势在 Unfamiliar faces (Q=0.171) 和 Scrambled faces (Q=0.170) 中保持一致，显示跨刺激条件的泛化性。Figure 7 展示了 multiSLICE 估计的超邻接矩阵 L̂ 的可视化模式。


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/4c77dcae-b0ad-4ab5-8e2c-f5057be8cdc2/figures/fig_003.png)
*Figure: The effects of changing l, the number of layers (simulation 1) and R(L∗), the rank of the*



**消融与对比分析**：隐含消融通过竞争对手的排除机制体现——MLGGM、coglasso、BANS、JMMLE 因要求等样本量 n_α 而无法直接应用于神经影像数据；CNJGL 和 BJEMGM 因要求等维度 p_α 需 SVD 投影到 p_MEG=52 的子空间，可能损害性能；CFR 因运行时间过长被排除。Figure 5 显示在两层系统中，增加其中一层的样本量 n₁ 或 n₂ 可改善联合 L 的恢复，验证了多层联合估计的信息借用效应。


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/4c77dcae-b0ad-4ab5-8e2c-f5057be8cdc2/figures/fig_004.png)
*Figure: Recovery of L∗with l = 2, R(L∗) = 2, n = 500 for methods with non-zero estimates.*



**公平性检验**：本文 baselines 覆盖较全（MLGGM、coglasso 为直接竞品，CNJGL/BJEMGM/LRGQ 为神经影像可比方法），但存在几点局限：(1) CFR 因计算成本排除，可能遗漏强 baseline；(2) 未与"单层 SLICE 逐层独立应用"对比，无法量化联合估计的增益；(3) 高斯假设限制了对非高斯神经信号（如 MEG 的超高斯性）的适用性；(4) 计算可扩展性至大规模网络（>1000 节点）未充分验证。实验在 M1 MacBook Pro (16GB RAM, CPU) 上完成，暗示方法计算效率尚可但非极致优化。

## 方法谱系与知识库定位

**方法族**: 高斯图模型 → 稀疏精度矩阵估计（图形 Lasso）→ 潜变量 GGMs（SLICE）→ **多层潜变量 GGMs（multiSLICE）**

**父方法**: **SLICE** (Chandrasekaran et al., 2012, "Latent variable graphical model selection via convex optimization")。multiSLICE 继承其 Θ = S + L 分解思想，但将 S 从单层稀疏矩阵改为块对角结构、将 L 从单层低秩改为跨层共享，并引入层特异样本量缩放理论。

**直接 baselines 差异**：
- **MLGGM** [25]: 同为多层 GGM，但 Θ 整体稀疏+低秩（无块对角区分层内/层间），且要求等样本量
- **coglasso**: 协作式图形 Lasso，层间通过惩罚耦合，但无显式潜变量结构，且要求等样本量
- **CNJGL/BJEMGM** [20][21]: 联合估计多图，但要求等维度，需投影预处理
- **LRGQ**: 可处理异质维度，但无多层稀疏+低秩的显式分解，层间结构估计较粗糙

**后续方向**：(1) 非高斯扩展：结合 copula 或半参数方法处理 MEG/fMRI 的非高斯性；(2) 动态多层扩展：将时间维度纳入，形成时变 multiSLICE；(3) 深度学习加速：用神经网络参数化潜变量分布，提升大规模网络的可扩展性。

**标签**: 模态-tabular/神经影像 | 范式-统计学习/凸优化 | 场景-多层网络/多模态融合 | 机制-稀疏低秩分解/潜变量建模 | 约束-高斯假设/异质采样

