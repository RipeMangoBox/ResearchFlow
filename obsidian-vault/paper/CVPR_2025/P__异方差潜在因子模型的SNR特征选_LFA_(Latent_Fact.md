---
title: Feature Selection for Latent Factor Models
type: paper
paper_level: C
venue: CVPR
year: 2025
paper_link: null
aliases:
- 异方差潜在因子模型的SNR特征选择
- LFA (Latent Fact
- LFA (Latent Factor Analysis with heteroskedastic noise for feature selection)
acceptance: poster
code_url: https://github.com/barbua/FS-LFA
method: LFA (Latent Factor Analysis with heteroskedastic noise for feature selection)
---

# Feature Selection for Latent Factor Models

[Code](https://github.com/barbua/FS-LFA)

**Topics**: [[T__Classification]] | **Method**: [[M__LFA]] | **Datasets**: Simulated data, Real multi-class datasets

| 中文题名 | 异方差潜在因子模型的SNR特征选择 |
| 英文题名 | Feature Selection for Latent Factor Models |
| 会议/期刊 | CVPR 2025 (poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2412.10128) · [Code](https://github.com/barbua/FS-LFA) · [DOI](https://doi.org/10.1109/CVPR52734.2025.02863) |
| 主要任务 | 特征选择（Feature Selection）、潜在因子建模 |
| 主要 baseline | PPCA, ELF, HeteroPCA, FSA, TISP |

> [!abstract] 因为「PPCA 假设各向同性噪声导致无法区分特征级别的信噪比」，作者在「PPCA」基础上改了「将对角异方差噪声 Ψ=diag(σ²₁,...,σ²_d) 引入 EM 算法并定义逐特征 SNR 作为选择标准」，在「模拟数据 (d=110) 及 CIFAR-10/100、ImageNet」上取得「大样本下最小估计误差及可比较的分类性能」

- **估计精度**：LFA 在样本量 n 较大时，对信号方差、噪声方差、SNR 的 MAD 估计误差均为最小（Figure 2 50 次运行平均）
- **特征选择准确率**：Table 1 显示 LFA 在模拟数据上优于 PPCA，与 ELF、HeteroPCA 可比（具体数值待补充）
- **训练效率**：Table 4 对比了低秩生成方法的训练时间（具体数值待补充）

## 背景与动机

高维数据中的特征选择是机器学习的核心问题之一。以图像分类为例，从原始像素或深度特征中识别出真正与类别相关的维度，既能降低计算开销，也能提升模型泛化能力。然而，当特征维度 d 远大于样本量 n 时，如何准确识别"信号特征"与"噪声特征"变得极为困难。

现有方法从不同角度切入这一问题：

- **PPCA (Probabilistic PCA)**：假设观测数据由低维潜在因子 γ 经载荷矩阵 W 线性生成，并叠加各向同性噪声 ε ~ N(0, σ²I_d)。该模型可通过 EM 算法高效求解，但由于所有特征共享同一噪声方差 σ²，无法区分不同特征的信噪比差异。
- **HeteroPCA**：放松各向同性假设，允许每个特征具有独立的噪声方差，并通过基于残差的估计量 σ̂²_i = ‖(X̂₀)_i· - X_i·‖₂²/(n-1) 进行估计。但其在正 SNR* 值上存在明显高估。
- **ELF (Explicit Latent Factor)**：通过正交约束下的白化重构误差最小化来估计参数，在信号估计上略优于 HeteroPCA，但在噪声估计上稍逊。

这些方法的共同局限在于：即使引入了异方差噪声，也未能将特征级别的信噪比显式地作为选择标准。具体而言，PPCA 因 σ²I_d 的强假设而系统性高估信号、混淆噪声特征；HeteroPCA 和 ELF 虽改进了噪声建模，但缺乏理论保证来确保"真实特征"与"噪声特征"的 SNR 存在明确分离。本文的核心动机正是填补这一空白——在潜在因子框架内，通过 EM 算法直接估计对角噪声协方差，并构造具有理论分离保证的逐特征 SNR 准则，从而实现可解释、可证明正确的特征选择。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/922163f7-9b74-40e0-b06a-858d6cbda54d/figures/Figure_1.png)



## 核心创新

核心洞察：将 PPCA 的各向同性噪声替换为对角异方差噪声 Ψ = diag(σ²₁, ..., σ²_d)，因为 EM 算法的 M-step 中引入对角约束 D(·) 可直接估计每个特征的独立噪声方差，从而使逐特征信噪比 SNR_i = (ΣⱼW²ᵢⱼ)/σ²ᵢ 的显式计算与理论分离保证成为可能。

| 维度 | Baseline (PPCA) | 本文 (LFA) |
|:---|:---|:---|
| 噪声假设 | Ψ = σ²I_d（各向同性，全特征共享） | Ψ = diag(σ²₁, ..., σ²_d)（异方差，逐特征独立） |
| 参数估计 | EM 算法，M-step 更新单一 σ² | EM 算法，M-step 用 D(·) 约束更新对角 Ψ |
| 特征选择准则 | 无显式准则，依赖载荷矩阵范数 | SNR_i = (ΣⱼW²ᵢⱼ)/σ²ᵢ，具分离条件 min{SNR*_i, i∈S} ≥ max{SNR*_i, i∉S} + γ |

## 整体框架



LFA 的完整流程包含五个串联模块，从原始数据输入到最终特征子集输出：

1. **输入数据 X**：n×d 维观测矩阵，其中 n 为样本量，d 为特征维度，潜在因子维度 r ≪ d 预设。

2. **LFA 参数估计（E-M 算法）**：核心模块。E-step 计算潜在因子的后验期望 E(γ|x_i) 与二阶矩 E(γγᵀ|x_i)；M-step 交替更新载荷矩阵 W 和对角噪声协方差 Ψ。与 PPCA 的关键区别在于 Ψ 的更新引入对角约束 D(·)，保留各特征独立的噪声方差估计。

3. **SNR 计算**：基于收敛后的参数，为每个特征计算信噪比 SNR_i = (ΣⱼW²ᵢⱼ)/σ²ᵢ。分子量化该特征在所有潜在因子上的总载荷能量（信号强度），分母为该特征的噪声方差。

4. **特征排序与选择**：按 SNR_i 降序排列，通过理论分离条件 min{SNR*_i, i∈S} ≥ max{SNR*_i, i∉S} + γ 确定截断点，输出选中特征索引集 S。

5. **下游任务**：将选中特征输入标准分类器（如 SVM、softmax），评估特征选择的实际效用。

数据流示意：
```
X_{n×d} → [E-step: E(γ|x), E(γγᵀ|x)] → [M-step: W_new, Ψ_new=diag] 
        → 迭代收敛 → {Ŵ, Ψ̂=diag(σ̂²_i)} → SNR_i = ΣⱼŴ²ᵢⱼ/σ̂²_i 
        → 排序 + 分离阈值 γ → 特征子集 S → 分类器训练/测试
```

## 核心模块与公式推导

### 模块 1: 对角噪声协方差估计（对应框架图第 2 步 M-step）

**直觉**：PPCA 假设所有特征噪声方差相同，导致高 SNR 特征被低 SNR 特征的噪声"平均稀释"；为每个特征独立估计噪声方差，才能准确识别真正的信号维度。

**Baseline 公式 (PPCA)**:
$$\text{bPsi} = \sigma^2 \bI_d, \quad \sigma^2_{ML} = \frac{1}{d-r}\sum_{j=r+1}^d \lambda_j$$
符号: σ² 为共享噪声方差，λ_j 为样本协方差矩阵的第 j 大特征值，d-r 个最小特征值的平均作为噪声估计。

**变化点**: PPCA 的闭式解依赖于各向同性假设，无法区分特征级别的噪声差异；当真实数据存在异方差时，该估计会产生系统性偏差。

**本文公式（推导）**:
$$\text{Step 1}: \text{bSigma} = \bW\bW^T + \text{bPsi}, \quad \bx \sim \N(\text{bmu}, \text{bSigma}) \quad \text{（边缘化潜在因子 γ 得观测分布）}$$
$$\text{Step 2}: \bW_{new} = \sum_{i=1}^n \bx_i E(\text{bgamma}|\bx_i)^T \left(\sum_{i=1}^n E(\text{bgamma}\text{bgamma}^T|\bx_i)\right)^{-1} \quad \text{（标准 M-step，与 PPCA 相同）}$$
$$\text{Step 3}: \text{bPsi}_{new} = \frac{1}{n}D\left(\sum_{i=1}^n\bx_i\bx_i^T - \bW_{new}E(\text{bgamma}|\bx_i)\bx_i^T\right) \quad \text{（关键修改：D(·) 仅保留对角元素）}$$
$$\text{最终}: \text{bPsi} = \text{diag}(\hat{\sigma}^2_1, \hat{\sigma}^2_2, \cdots, \hat{\sigma}^2_d)$$

**对应消融**: 去掉对角约束（即退化为 PPCA）后，Figure 2 显示 PPCA 对信号方差的 MAD 估计误差最大，且对噪声特征的 SNR 存在系统性高估，部分真实信号的估计 SNR 接近零。

---

### 模块 2: 逐特征 SNR 计算与分离条件（对应框架图第 3-4 步）

**直觉**：噪声方差异质化后，简单的载荷范数无法公平比较不同特征的重要性——高载荷可能伴随高噪声；需要归一化的信噪比指标。

**Baseline 公式 (PPCA/传统因子分析)**:
无显式逐特征 SNR；特征重要性通常以 ‖W_i·‖₂² 或投影方差近似，缺乏噪声归一化。

**变化点**: 传统方法未利用已估计的异方差噪声信息，无法理论上保证选中特征集合的正确性。

**本文公式（推导）**:
$$\text{Step 1}: \text{bSNR}_i = \frac{\sum_{j=1}^r \bW_{ij}^2}{\sigma^2_i}, \quad i \in \{1,2,\cdots,d\} \quad \text{（载荷能量除以特征专属噪声方差）}$$
$$\text{Step 2}: \text{排序得 } \text{bSNR}_{(1)} \ge \text{bSNR}_{(2)} \ge \cdots \ge \text{bSNR}_{(d)}$$
$$\text{最终}: \min\{\text{bSNR}_{i}^{*}, i \in S\} \ge \max\{\text{bSNR}_{i}^{*}, i \text{not}\in S\} + \gamma \quad \text{（分离条件，γ > 0 为安全间隔）}$$

符号: W_ij 为第 i 个特征在第 j 个潜在因子上的载荷，σ²_i 为第 i 个特征的噪声方差，S 为真实相关特征集合，SNR* 为基于真实参数的 SNR 值。

**对应消融**: Table 1 显示（具体数值待补充），当使用 ELF 或 HeteroPCA 替代 LFA 的估计时，两者 SNR 估计性能相近，ELF 略优信号估计而 HeteroPCA 略优噪声估计，但大样本下均不及 LFA。

---

### 模块 3: ELF 对比目标（基线方法理解）

为完整理解 LFA 的估计策略差异，ELF 的优化目标为：
$$(\hat{\bW}_{ELF}, \hat{\text{bGamma}}_{ELF}) = \argmin_{(\text{bGamma}, \bW),\text{bGamma}^T \text{bGamma} = \bI_r} \|(\bX - \text{bGamma} \bW^T)\text{bPsi}^{-\frac{1}{2}} \|^2_F$$

ELF 通过白化后的重构误差最小化求解，需预设或迭代估计 Ψ；LFA 则通过 EM 算法的似然最大化联合估计 W 与 Ψ，后者在概率框架下更自然，且大样本性质更优。

## 实验与分析


![Table 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/922163f7-9b74-40e0-b06a-858d6cbda54d/figures/Table_2.png)
*Table 2: Table 2. Classification accuracy (%) for different methods on realdatasets*



本文在两类实验设置下验证 LFA：可控模拟实验与真实图像数据集分类任务。

模拟实验（Section 5.1）在 d=110 维数据上进行，其中 10 维为真实信号特征，100 维为纯噪声特征。评估指标为特征选择准确率 Acc = E(|I_true ∩ I_pred|)/|I_true|，以及参数估计的 MAD（Mean Absolute Deviation）。
![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/922163f7-9b74-40e0-b06a-858d6cbda54d/figures/Table_1.png)
*Table 1: Table 1. Feature selection accuracy (%) for simulated data.*

 对应 Table 1 的结果表明，LFA 在大样本量 n 下获得了最小的信号方差、噪声方差及 SNR 估计误差。具体而言，Figure 2 展示了 50 次独立运行中 MAD 随 n 变化的曲线：LFA 的三条误差曲线（信号、噪声、SNR）在 n 增大时均位于最下方，而 PPCA 因各向同性假设导致信号估计误差显著偏高。ELF 与 HeteroPCA 表现相近——ELF 信号估计略优，HeteroPCA 噪声估计略优——但两者在 SNR 估计上与 LFA 相当，却在参数估计精度上稍逊。

真实数据集实验（Section 5.2）在 CIFAR-10、CIFAR-100 和 ImageNet 上进行多类分类。Table 2 对比了 LFA 与 FSA、TISP 等方法的分类准确率（具体数值待补充）。虽然精确提升幅度未在提取文本中给出，但 LFA 作为生成式特征选择方法，其优势在于无需预设选择特征数即可通过 SNR 分离条件自动确定子集大小。

训练效率方面，Table 4 比较了低秩生成方法的训练时间（具体数值待补充），Table 3 则对比了 FSA 与 TISP 的训练时间。

消融分析揭示了两个关键发现：第一，去掉异方差噪声（退化为 PPCA）代价最大——PPCA 不仅信号估计误差最大，还会将噪声特征的 SNR 系统性高估，同时使部分真实信号的 SNR 估计接近零，导致特征选择完全失效。第二，LFA 的 EM 估计策略优于 HeteroPCA 的残差估计与 ELF 的正交约束优化，尤其在大样本下，联合似然最大化的一致性保证使 MAD 误差收敛更快。

公平性审视：本文的比较主要局限于经典统计方法（PPCA、ELF、HeteroPCA、FSA、TISP），未纳入现代深度学习方法（如基于神经网络的可微分特征选择）、互信息方法、LASSO 或树模型特征重要性等更强基线。此外，Table 中的具体数值未完全提取，定量评估受限。作者未明确披露失败模式，但从理论分离条件可知，当 γ 过小或 SNR 分布重叠严重时，选择准确率将下降。

## 方法谱系与知识库定位

LFA 属于**概率潜在因子模型**家族，直接继承自 **PPCA (Probabilistic PCA)**，在生成模型 x = μ + Wγ + ε 的框架上进行三项关键修改：

| 改动维度 | 具体变化 |
|:---|:---|
| **目标函数 (objective)** | 噪声协方差从 σ²I_d → diag(σ²₁,...,σ²_d)，引入逐特征异方差 |
| **训练策略 (training_recipe)** | EM 算法的 M-step 中 Ψ 更新加入对角约束 D(·)，替代 PPCA 的闭式特征值解 |
| **推断策略 (inference_strategy)** | 新增 SNR_i = (ΣⱼW²ᵢⱼ)/σ²ᵢ 计算与分离阈值决策，将参数估计与特征选择桥接 |

直接基线关系：
- **PPCA**：父方法；LFA 保留其生成模型与 EM 框架，推翻各向同性噪声假设
- **HeteroPCA**：同期异方差 PCA 方法；LFA 采用不同噪声估计策略（EM 联合估计 vs. 残差估计），大样本理论性质更优
- **ELF**：同期显式潜在因子方法；LFA 以概率似然替代白化重构误差，估计目标不同
- **FSA / TISP**：判别式特征选择基线；LFA 作为生成式方法与之互补，训练效率可比

后续可能方向：
1. **深度化扩展**：将 LFA 的对角异方差 SNR 框架嵌入 VAE 等深度生成模型，处理非线性潜在结构
2. **在线/流式估计**：开发增量 EM 更新，使 SNR 计算适应动态特征流
3. **结构化噪声**：从对角 Ψ 扩展为块对角或稀疏结构，捕捉特征组间的噪声相关性

知识库标签：
- **模态 (modality)**：表格/向量数据，可扩展至图像特征
- **范式 (paradigm)**：概率生成模型 + 无监督/自监督特征选择
- **场景 (scenario)**：高维低样本 (n ≪ d)、可解释特征筛选
- **机制 (mechanism)**：EM 算法、异方差建模、信噪比分离
- **约束 (constraint)**：线性潜在因子假设、对角噪声协方差、已知潜在维度 r

