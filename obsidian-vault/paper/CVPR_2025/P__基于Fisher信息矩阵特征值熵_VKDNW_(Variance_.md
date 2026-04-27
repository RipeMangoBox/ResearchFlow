---
title: Training-free Neural Architecture Search through Variance of Knowledge of Deep Network Weights
type: paper
paper_level: C
venue: CVPR
year: 2025
paper_link: null
aliases:
- 基于Fisher信息矩阵特征值熵的无训练NAS
- VKDNW (Variance
- VKDNW (Variance of Knowledge of Deep Network Weights)
acceptance: poster
cited_by: 2
method: VKDNW (Variance of Knowledge of Deep Network Weights)
baselines:
- 组装零成本代理的高效NAS方法_AZ-NAS
---

# Training-free Neural Architecture Search through Variance of Knowledge of Deep Network Weights

**Topics**: [[T__Neural_Architecture_Search]], [[T__Classification]] | **Method**: [[M__VKDNW]] | **Datasets**: [[D__NAS-Bench-201]], [[D__ImageNet-1K]]

| 中文题名 | 基于Fisher信息矩阵特征值熵的无训练NAS |
| 英文题名 | Training-free Neural Architecture Search through Variance of Knowledge of Deep Network Weights |
| 会议/期刊 | CVPR 2025 (poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2502.04975) · [Code](待补充) · [Project](待补充) |
| 主要任务 | 零样本神经架构搜索 (Zero-shot NAS) |
| 主要 baseline | AZ-NAS, ZiCo, NASWOT, Zen-NAS, GradSign, Synflow, SNIP, GraSP |

> [!abstract] 因为「现有零成本代理(zero-cost proxy)与网络大小高度相关导致排名偏差」，作者在「AZ-NAS」基础上改了「用Fisher信息矩阵特征值熵替代梯度统计代理，并引入对数乘积非线性聚合」，在「NAS-Bench-201 / ImageNet-1K MobileNetV2搜索空间」上取得「KT 0.743 / Top-1 78.8%」

- **NAS-Bench-201**: VKDNW_agg 的 Kendall's τ = 0.743，相比 AZ-NAS (0.717) 提升 +0.026
- **ImageNet-1K**: VKDNW_agg 在 MobileNetV2 搜索空间达到 Top-1 准确率 78.8%，超越 ZiCo (78.1%) +0.7%
- **搜索成本**: 仅 0.4 GPU 天，与 AZ-NAS/ZiCo 同级，最终模型训练 7 天 (8× A100)

## 背景与动机

神经架构搜索(NAS)旨在自动发现最优网络结构，但传统方法需要完整训练每个候选架构，计算成本极高。零样本/无训练NAS(zero-shot NAS)通过设计廉价代理指标(proxy)来预测架构性能，避免训练开销。例如，ZiCo利用梯度变异系数，AZ-NAS组合多个代理(progressivity, expressivity, trainability, Jacov, FLOPs)进行线性加权聚合，NASWOT基于激活相关性评估架构。

然而，这些现有方法存在一个根本性缺陷：**它们与网络大小（如可训练层数ℵ）高度相关**。Figure 3 显示，AZ-NAS的progressivity与ℵ的Kendall's τ达0.42，expressivity达0.56，trainability达0.25。这意味着这些代理实质上在"奖励"更大的网络，而非真正识别高效结构——当搜索空间中存在多个相似大小的架构时，它们的区分能力急剧下降。这种"大小偏见"导致代理在识别真正优质小模型时失效。

本文提出VKDNW，核心思想是利用Fisher信息矩阵的谱特性：参数估计的Cramér-Rao下界表明，FIM特征值反映了各参数方向的信息量。通过度量特征值分布的熵，可以捕捉网络权重的"知识方差"——且这一度量天然与网络大小正交。
![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/048a180d-dae4-4a50-9570-2d79306200ad/figures/Figure_3.png)
*Figure 3: Fig. 3: Components of AZ-NAS [21] and our VKDNW arecompared w.r.t. correlation with ℵ(number of trainable layers),in the NAS-Bench-201 search space [10] on ImageNet16-120[7] dataset. Our VKDNW proxy h*



## 核心创新

核心洞察：Fisher信息矩阵特征值的熵可以量化网络参数在各方向上的"知识"分布均匀性，因为Cramér-Rao下界表明FIM控制了参数估计的最小方差，从而使与网络大小无关的零成本代理成为可能——小特征值方向对应"已知"参数（扰动不影响输出），大特征值方向对应"待学习"参数，熵高意味着知识分布均衡、模型表达能力强。

| 维度 | Baseline (AZ-NAS) | 本文 (VKDNW) |
|:---|:---|:---|
| 代理来源 | 梯度统计 + 激活统计 + FLOPs (多源异构) | Fisher信息矩阵特征值熵 (单源理论驱动) |
| 与网络大小相关性 | progressivity KT=0.42, expressivity KT=0.56 | VKDNW KT=-0.11 (近乎正交) |
| 聚合方式 | 线性加权 Σ w_j · proxy_j | 对数乘积 log Π rank_j(f) (非线性、无权重调参) |
| 评估指标 | KT, SPR 仅衡量全局相关性 | 新增 nDCG@P 聚焦Top-P架构识别能力 |

## 整体框架


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/048a180d-dae4-4a50-9570-2d79306200ad/figures/Figure_2.png)



VKDNW的计算流程分为六个阶段，全部在随机初始化网络上单次完成：

1. **输入准备**: 取随机初始化的候选网络，输入一个mini-batch数据（无需标签）
2. **Jacobian计算**: 前向传播得到logits Ψ(x_n, θ)，计算关于参数θ的Jacobian矩阵 ∇_θΨ(x_n, θ)
3. **FIM组装**: 利用softmax梯度结构（diag(σ)-σσ^T），将FIM表示为Gram矩阵形式：F̂(θ) = (1/n)Σ A_n^T A_n，其中A_n为简化后的Jacobian
4. **特征分解**: 对F̂(θ)进行特征值分解，提取最大的9个特征值 {λ_k}_{k=1}^9
5. **熵计算**: 归一化特征值 λ̃_k = λ_k / Σλ_j，计算香农熵 VKDNW(f) = -Σ λ̃_k log λ̃_k
6. **排名聚合(可选)**: VKDNW_agg 将VKDNW与其他代理的排名通过对数乘积聚合：rank_agg(f) = log Π_j rank_j(f)

```
随机网络 + mini-batch ──→ 前向传播 ──→ Jacobian A_n ──→ F̂(θ)=(1/n)ΣA_n^TA_n 
                                                                  ↓
最终排名 ←── 对数乘积聚合(可选) ←── 熵 VKDNW(f) ←── 归一化 ←── 特征分解(Top-9 λ)
```

## 核心模块与公式推导

### 模块 1: Fisher信息矩阵的构造与简化（对应框架图步骤2-3）

**直觉**: 分类网络的softmax输出对参数的敏感度，可以通过得分函数（对数似然梯度）的外积期望来量化，这正是Fisher信息矩阵的经典定义。

**Baseline 公式 (传统FIM定义)**: 
$$F(\theta) \text{coloneq} \mathbb{E}\left[\nabla_{\theta}\sigma_\theta(c\,|\,x)\,\nabla_{\theta}\sigma_\theta(c\,|\,x)^T\right] \in \mathbb{R}^{p\times p}$$
符号: $\sigma_\theta(c|x)$ = softmax概率, $\nabla_\theta\sigma$ = 输出对参数的梯度, $p$ = 参数总数

**变化点**: 直接计算$p \times p$矩阵不可行（现代网络参数可达百万级）。本文利用分类问题的特殊结构：softmax梯度可分解为Jacobian与softmax Hessian的乘积。

**本文公式（推导）**:
$$\text{Step 1}: \nabla_\theta\sigma_\theta(c|x_n) = \nabla_\theta\Psi(x_n,\theta)^T \cdot \underbrace{(\text{diag}(\sigma_\theta) - \sigma_\theta\sigma_\theta^T)}_{\text{softmax梯度结构}} \quad \text{将梯度分解为Jacobian与局部结构}$$
$$\text{Step 2}: \hat{F}(\theta) = \frac{1}{n}\sum_{n=1}^{N} A_n^T A_n, \quad A_n = (\text{diag}(\sigma_\theta)-\sigma_\theta\sigma_\theta^T)^{1/2} \nabla_\theta\Psi(x_n,\theta) \quad \text{简化为Gram矩阵，利用低秩结构}$$
$$\text{最终}: \hat{F}(\theta) = \frac{1}{n}\sum_{n=1}^{N} A_n^T A_n \in \mathbb{R}^{p\times p} \text{（隐式低秩表示，避免显式存储）}$$

**对应消融**: Table III 显示，基于此FIM的VKDNW是聚合中最重要的组件；移除V(VKDNW)后KT/SPR/nDCG下降最大（具体数值待补充）。

### 模块 2: VKDNW核心——特征值熵（对应框架图步骤4-5）

**直觉**: FIM特征值分布的均匀性反映了网络"知识"在各参数方向上的分散程度。熵高=各方向信息量均衡=网络表达能力强；熵低=少数方向主导=网络存在冗余或表达瓶颈。

**Baseline 公式 (AZ-NAS各代理)**: 
- progressivity: 梯度范数相关统计
- expressivity: 激活值统计 
- trainability: 梯度协方差条件数
这些代理均与网络规模存在系统性相关（Figure 3, Table IV）。

**变化点**: 现有代理未从参数估计理论出发，缺乏对"知识"分布的几何解释。本文通过Cramér-Rao下界建立联系：$\text{Var}(\hat{\theta}_n) \geq \frac{1}{n}F^{-1}(\theta)$，说明FIM特征值直接控制各方向的估计精度。

**本文公式（推导）**:
$$\text{Step 1}: D_{KL}(\sigma_{\theta+\theta_\delta}, \sigma_\theta) \approx \frac{1}{2}\theta_\delta^T F(\theta) \theta_\delta = \frac{1}{2}\sum_k \lambda_k (v_k^T\theta_\delta)^2 \quad \text{KL散度二次近似，特征值作为Riemannian度量}$$
$$\text{Step 2}: \lambda_{\min}\|\theta_{\min}\|^2 \ll \lambda_{\max}\|\theta_{\max}\|^2 \quad \text{大小特征值方向敏感度差异巨大，motivates关注分布而非极值}$$
$$\text{Step 3}: \tilde{\lambda}_k = \frac{\lambda_k}{\sum_{j=1}^9 \lambda_j}, \quad k=1,\dots,9 \quad \text{取Top-9特征值归一化为概率分布}$$
$$\text{最终}: \text{VKDNW}(f) = -\sum_{k=1}^9 \tilde{\lambda}_k \log \tilde{\lambda}_k \quad \text{香农熵度量"知识方差"}$$

**对应消融**: Table IV 显示VKDNW与网络层数ℵ的KT=-0.11，而progressivity KT=0.42、expressivity KT=0.56、trainability KT=0.25，验证正交性设计。

### 模块 3: 非线性排名聚合 VKDNW_agg（对应框架图步骤6）

**直觉**: 线性加权聚合需要调参且对尺度敏感；排名的对数乘积天然无量纲、无需权重、对异常值鲁棒。

**Baseline 公式 (AZ-NAS线性聚合)**: 
$$\text{AZ-NAS}: \sum_j w_j \cdot \text{proxy}_j(f)$$
符号: $w_j$ = 手工或学习得到的权重, proxy_j = 各零成本代理原始值

**变化点**: 线性聚合中不同代理的量纲和范围差异大，权重选择敏感；且原始值中的异常值会直接影响最终排名。

**本文公式（推导）**:
$$\text{Step 1}: \text{rank}_j(f) = \text{架构} f \text{在第} j \text{个代理中的排名（整数，1=最优）} \quad \text{转换为排名消除量纲差异}$$
$$\text{Step 2}: \text{rank}_{\text{agg}}(f) := \log \prod_{j=1}^m \text{rank}_j(f) = \sum_{j=1}^m \log \text{rank}_j(f) \quad \text{对数乘积=对数排名之和，单调变换保持序关系}$$
$$\text{最终}: \text{VKDNW\_agg}(f) = \log\left[\text{rank}_{\text{VKDNW}}(f) \cdot \text{rank}_{\text{Jacov}}(f) \cdot \text{rank}_{\text{expressivity}}(f) \cdot \text{rank}_{\text{trainability}}(f) \cdot \text{rank}_{\text{FLOPs}}(f)\right]$$

**对应消融**: Table III 显示线性聚合版本劣于对数乘积版本（具体Δ值待补充），验证非线性设计的有效性。

## 实验与分析



本文在两大基准上验证VKDNW：NAS-Bench-201（含CIFAR-10/100、ImageNet16-120）用于代理相关性评估，以及ImageNet-1K上的MobileNetV2搜索空间用于端到端架构搜索。

在NAS-Bench-201上，VKDNW_agg取得Kendall's τ = 0.743、Spearman's ρ = 0.906、nDCG@1000 = 0.664，相比AZ-NAS的0.717/0.891/0.623分别提升+0.026/+0.015/+0.041。其中nDCG指标是本文专为NAS场景设计的——它强调识别Top-P最优架构的能力，而非全局相关性，这对实际搜索更为关键。VKDNW_single（仅VKDNW+网络大小）在单代理方法中也表现最优，证明Fisher信息熵本身即具强判别力。

在ImageNet-1K MobileNetV2搜索空间（约束约450M FLOPs），VKDNW_agg搜索到的架构达到Top-1准确率78.8%，超越AZ-NAS (78.6%) +0.2%、ZiCo (78.1%) +0.7%、DONNA (78.0%)、OFA (77.7%)等。搜索成本仅0.4 GPU天，与AZ-NAS/ZiCo同级；最终模型训练7天（8× A100）。值得注意的是，VKDNW_agg的FLOPs为480M，略高于AZ-NAS的462M和ZiCo的448M，但在相近计算预算下实现了更高精度。



消融实验（Table III）表明，在VKDNW_agg的五组件（V=VKDNW, J=Jacov, E=expressivity, T=trainability, F=FLOPs）中，移除VKDNW(V)导致所有指标最大下降，验证其为聚合的核心驱动力。非线性对数乘积聚合也优于线性替代方案。

公平性考量：对比基线涵盖AZ-NAS、ZiCo等零样本NAS代表方法，以及DONNA、OFA等高效NAS方法，选择合理。缺失对比包括TE-NAS（理论上相关的NAS方法，仅在相关工作提及）。潜在局限：最终模型训练沿用竞争对手[21,25]的超参数设置，可能对VKDNW发现的架构非最优；MobileNetV2空间的FLOPs约束下，VKDNW_agg实际FLOPs略超目标值（480M vs ≈450M）。

## 方法谱系与知识库定位

VKDNW属于**零样本神经架构搜索(Zero-shot NAS)**方法家族，直接父方法为**AZ-NAS**——本文继承了其"组装多代理"的范式，但彻底替换了代理来源、聚合机制、大小相关性处理和评估指标四个核心槽位。

**直接基线差异**：
- **AZ-NAS**: 线性聚合梯度/激活/FLOPs代理 → VKDNW改用Fisher信息熵单源代理+对数乘积聚合
- **ZiCo**: 梯度变异系数统计 → VKDNW基于参数估计理论，从FIM谱分析出发
- **NASWOT/Zen-NAS/GradSign**: 激活相关/敏感度/梯度符号方法 → VKDNW首次将Fisher信息矩阵特征值熵引入NAS代理

**后续方向**：(1) 将VKDNW扩展至Transformer架构的自注意力谱分析；(2) 结合架构参数化搜索（如DARTS）实现可微分零样本优化；(3) 探索FIM高阶矩（超越熵）的判别信息。

**标签**: 模态=视觉图像分类 | 范式=零样本/无训练NAS | 场景=资源受限架构搜索 | 机制=Fisher信息矩阵·特征值谱·信息熵 | 约束=无训练·单前向/梯度计算·低计算预算

## 引用网络

### 直接 baseline（本文基于）

- [[P__组装零成本代理的高效NAS方法_AZ-NAS]] _(直接 baseline)_: Zero-cost NAS proxy method likely compared against as a baseline approach

