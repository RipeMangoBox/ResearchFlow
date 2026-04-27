---
title: Training-free Neural Architecture Search through Variance of Knowledge of Deep Network Weights
type: paper
paper_level: C
venue: CVPR
year: 2025
paper_link: null
aliases:
- 基于Fisher谱熵的无训练神经架构搜索
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

| 中文题名 | 基于Fisher谱熵的无训练神经架构搜索 |
| 英文题名 | Training-free Neural Architecture Search through Variance of Knowledge of Deep Network Weights |
| 会议/期刊 | CVPR 2025 (poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2502.04975) · [Code](https://github.com/ondrejtybl/VKDNW) · [Project](待补充) |
| 主要任务 | Zero-shot NAS / 零成本代理评估 |
| 主要 baseline | AZ-NAS, ZiCo, TE-NAS, Zen-NAS, GradSign |

> [!abstract] 因为「现有零成本代理与网络容量强相关、缺乏理论根基」，作者在「AZ-NAS」基础上改了「以Fisher信息矩阵谱熵为核心的单代理指标，并引入非线性对数乘积排序聚合」，在「NAS-Bench-201 ImageNet16-120」上取得「KT=0.743 / SPR=0.906 / nDCG=0.664」，在「ImageNet-1K MobileNetV2搜索空间」上达到「78.8% top-1 accuracy，超越AZ-NAS 78.6% 和 ZiCo 78.1%」。

- **NAS-Bench-201**: VKDNWagg 的 Kendall's τ = 0.743，相比 AZ-NAS 提升 +0.121，相比 Jacov 提升 +0.140
- **ImageNet-1K**: VKDNWagg 达到 78.8% top-1 accuracy，搜索仅需 ~0.4 GPU days（100K次进化迭代约10小时）
- **Size invariance**: VKDNW 与网络可训练层数的 KT = -0.11，远低于 AZ-NAS 组件的 0.42/0.56/0.25

## 背景与动机

神经架构搜索（NAS）旨在自动发现最优网络结构，但传统方法需要训练成千上万个候选网络，计算成本极高。零样本NAS（Zero-shot NAS）试图在**完全不训练**的情况下预测网络性能，从而将搜索成本从数百GPU天降至数小时。例如，ZiCo通过梯度变异系数的逆作为代理指标，AZ-NAS则将多个零成本代理（expressivity、trainability、progressivity）线性聚合，在NAS-Bench-201上取得了不错的排序相关性。

然而，现有方法存在根本性缺陷：**它们与网络容量（参数量、FLOPs、深度）强耦合**。如图2所示，梯度方差较小的网络（左）与真实精度相关性差，而现有代理往往偏好更深、更宽的网络，而非真正"可训练得好"的网络。AZ-NAS的progressivity与可训练层数的Kendall τ高达0.42，expressivity更达0.56——这意味着这些代理本质上在预测"网络有多大"而非"网络能学多好"。此外，大多数零成本代理缺乏理论支撑，是启发式设计的梯度统计量。

本文的核心动机源于信息几何：Fisher信息矩阵的特征值谱反映了参数空间各方向的敏感度，其分布的"均匀性"（熵）应能刻画网络的学习潜力——且这一度量应与网络大小解耦。作者由此提出VKDNW，将Fisher谱熵作为理论根基扎实的单代理指标。


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/f447e043-226e-4d47-8d46-ab40ba27fe15/figures/Figure_2.png)
*Figure 2 (example): The example of searching on NAS-Bench-201 [18]. We plot accuracy obtained by the rankings and various metrics. Lower gradient variance (left) and Kendall's Tau (right) indicate higher performance. Our VKDOW proxy has the lowest correlation.*



## 核心创新

核心洞察：**Fisher信息矩阵归一化特征值的香农熵**，因为该熵度量了"知识"在参数维度上的分布均匀性，从而使**不依赖网络容量、仅依赖几何结构**的零成本代理成为可能。具体而言，Fisher矩阵特征值谱的病态性（λ_max >> λ_min）意味着少数方向主导了参数更新；高熵表示知识分散、各方向均可学习，暗示更好的训练潜力。

| 维度 | Baseline (AZ-NAS) | 本文 |
|:---|:---|:---|
| 核心代理 | 线性聚合多个启发式代理（E/T/P） | 单代理：Fisher谱熵 VKDNW |
| 理论基础 | 经验组合，无统一理论 | 信息几何 + Cramér-Rao界 |
| 容量相关性 | progressivity KT=0.42, expressivity KT=0.56 | VKDNW KT=-0.11（近似无关） |
| 聚合方式 | 线性加权 | 非线性对数乘积排序 + 随机森林 |
| 评估指标 | KT, SPR | 新增 nDCG@1000 聚焦头部架构识别 |

## 整体框架


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/f447e043-226e-4d47-8d46-ab40ba27fe15/figures/Figure_3.png)
*Figure 3 (ablation): Components of AZ-NAS [21] and our VKDOW are compared on CIFAR-10 (top row) and ImageNet16-120 (bottom row). Our VKDOW proxy has the lowest correlation. (Best viewed in color.)*



VKDNW 的完整流程包含五个阶段：

1. **输入**: 未训练网络 $f$（随机初始化权重）+ 小批量数据 $\{x_n, c_n\}_{n=1}^N$
2. **Fisher信息计算**: 通过一次前向-后向传播计算经验Fisher矩阵 $\hat{F}(\theta) \in \mathbb{R}^{p \times p}$，利用softmax的Jacobian结构避免显式构造大矩阵
3. **特征值分解**: 提取 $\hat{F}(\theta)$ 的前9大特征值 $\{\lambda_k\}_{k=1}^9$
4. **谱熵计算（VKDNW）**: 归一化后计算香农熵 $\text{VKDNW}(f) = -\sum_{k=1}^9 \tilde{\lambda}_k \log \tilde{\lambda}_k$
5. **（可选）聚合与搜索**: VKDNW_single 直接结合网络复杂度 $\text{aleph}(f)$ 排序；VKDNWagg 通过 $\log \prod_{j=1}^m \text{rank}_j(f)$ 非线性聚合多代理排序，再经模型驱动的随机森林精炼，最终输入进化算法搜索最优架构

```
Untrained Network f ──→ Forward/Backward ──→ ĤF(θ) = (1/n)Σ A_n^T A_n
                                                    │
                                                    ▼
                                            Top-9 Eigendecomposition
                                                    │
                    ┌───────────────────────────────┼───────────────────────────────┐
                    ▼                               ▼                               ▼
            VKDNW(f) = -Σ λ̃_k log λ̃_k      VKDNW_single = ℵ(f) + VKDNW(f)    rank_agg = log Π rank_j(f)
                    │                               │                               │
                    └───────────────────────────────┴───────────────────────────────┘
                                                    │
                                                    ▼
                                           Model-driven Random Forest
                                                    │
                                                    ▼
                                           Evolutionary Search (100K iter)
                                                    │
                                                    ▼
                                              Selected Architecture
                                                    │
                                                    ▼
                                           Full Training (7 days, 8×A100)
```

## 核心模块与公式推导

### 模块 1: 经验Fisher信息矩阵的高效计算（对应框架图阶段2）

**直觉**: 直接计算梯度外积的期望需要 $O(p^2)$ 存储和 $O(NCp^2)$ 时间，对于现代网络不可行；利用softmax输出的结构特性可转化为Jacobian乘积和。

**Baseline 公式** (标准经验Fisher): 
$$\hat{F}_{\text{base}}(\theta) = \frac{1}{n}\sum_{n=1}^N \nabla_\theta \sigma_\theta(c_n|x_n) \nabla_\theta \sigma_\theta(c_n|x_n)^T$$
符号: $\sigma_\theta(c|x)$ = softmax输出概率, $\nabla_\theta$ = 对参数梯度, $p$ = 参数量

**变化点**: 单次样本梯度外积计算慢且存储大；本文利用多分类softmax的Hessian结构 $H = \text{diag}(\sigma) - \sigma\sigma^T$，将梯度重写为 $\nabla_\theta \sigma = (\nabla_\theta \Psi) \cdot H$。

**本文公式（推导）**:
$$\text{Step 1}: \quad A_n = \left[\text{diag}(\sigma_\theta(\cdot|x_n)) - \sigma_\theta(\cdot|x_n)\sigma_\theta(\cdot|x_n)^T\right]^{1/2} \cdot \nabla_\theta \Psi(x_n, \theta) \quad \text{（构造紧凑Jacobian）}$$
$$\text{Step 2}: \quad \hat{F}(\theta) = \frac{1}{n}\sum_{n=1}^N A_n^T A_n \quad \text{（避免显式 } p \times p \text{ 矩阵，仅需存储 } A_n \in \mathbb{R}^{C \times p}\text{）}$$
$$\text{最终}: \quad \hat{F}(\theta) = \frac{1}{n}\sum_{n=1}^N \nabla_\theta \Psi(x_n, \theta)^T \left[\text{diag}(\sigma) - \sigma\sigma^T\right] \nabla_\theta \Psi(x_n, \theta)$$

**对应消融**: 

---

### 模块 2: VKDNW谱熵代理（对应框架图阶段4）

**直觉**: Fisher矩阵特征值谱的病态性（如 $\lambda_{\max} \gg \lambda_{\min}$）意味着参数空间存在"硬学习"方向；熵度量了这种不均匀性，高熵=各向同性学习=好架构。

**Baseline 公式** (ZiCo/GradSign等梯度统计代理): 
$$\text{ZiCo}(f) = \frac{\mathbb{E}[\|\nabla_\theta \mathcal{L}\|]}{\text{Std}[\|\nabla_\theta \mathcal{L}\|]} \quad \text{（梯度均值与标准差之比）}$$
符号: $\mathcal{L}$ = 损失函数, 期望/标准差在参数维度上计算

**变化点**: ZiCo等代理与网络深度/宽度强相关（深层网络梯度统计量天然更大）；本文从信息几何出发，用Fisher谱的**归一化分布形状**而非绝对量值，实现与网络容量的解耦。

**本文公式（推导）**:
$$\text{Step 1}: \quad \tilde{\lambda}_k = \frac{\lambda_k}{\sum_{j=1}^9 \lambda_j}, \quad k=1,\dots,9 \quad \text{（归一化前9大特征值，消除尺度依赖）}$$
$$\text{Step 2}: \quad \text{VKDNW}(f) = -\sum_{k=1}^9 \tilde{\lambda}_k \log \tilde{\lambda}_k \quad \text{（香农熵：均匀分布时最大，退化分布时最小）}$$
$$\text{Step 3（可选组合）}: \quad \text{VKDNW}_{\text{single}}(f) = \text{aleph}(f) + \text{VKDNW}(f) \quad \text{（加入网络复杂度 } \text{aleph}(f) \text{ 补偿纯熵的信息损失）}$$
$$\text{最终}: \quad \text{VKDNW}(f) = -\sum_{k=1}^9 \frac{\lambda_k}{\sum_j \lambda_j} \log \frac{\lambda_k}{\sum_j \lambda_j}$$

**对应消融**: Figure 3 显示 VKDNW 与可训练层数的 KT = -0.11，而 AZ-NAS 的 progressivity KT = 0.42、expressivity KT = 0.56、trainability KT = 0.25，证明 size invariance。

---

### 模块 3: 非线性排序聚合 VKDNWagg（对应框架图阶段5）

**直觉**: 线性聚合多个代理时，某个代理的极端低分可被其他代理的高分掩盖；对数乘积强调"一致高排名"，更适合识别真正优秀的架构。

**Baseline 公式** (AZ-NAS线性聚合):
$$\text{AZ-NAS}(f) = \sum_j w_j \cdot \text{proxy}_j(f) \quad \text{（固定权重线性组合）}$$

**变化点**: 线性加权对异常值敏感且权重需调优；本文采用排序空间的非线性聚合，并引入模型驱动的随机森林学习最优组合。

**本文公式（推导）**:
$$\text{Step 1}: \quad \text{rank}_j(f) = \text{architecture } f \text{ 的第 } j \text{ 个代理排名（1=最好）}$$
$$\text{Step 2}: \quad \text{rank}_{\text{agg}}(f) := \log \prod_{j=1}^m \text{rank}_j(f) = \sum_{j=1}^m \log \text{rank}_j(f) \quad \text{（对数乘积=对数和，强调一致高排名）}$$
$$\text{Step 3}: \quad \text{RandomForest}(\{\text{rank}_j(f)\}_{j=1}^m) \rightarrow \text{final score} \quad \text{（1024个网络训练100轮）}$$
$$\text{最终}: \quad \text{VKDNWagg}(f) = \text{RF}_{\text{trained}}\left(\log \text{rank}_{\text{VKDNW}}(f), \log \text{rank}_{\text{Jacov}}(f), \dots\right)$$

**对应消融**: Table III 显示，完整组合 V+J+E+T+F 在 KT/SPR/nDCG 上均最优；移除 VKDNW（V）导致最大性能下降，验证其核心贡献。

## 实验与分析





本文在两大基准上验证 VKDNW：NAS-Bench-201（15,625个架构，评估代理排序质量）和 ImageNet-1K MobileNetV2 搜索空间（验证实际搜索效果）。

**NAS-Bench-201 / ImageNet16-120 代理质量**: Table I 显示，VKDNWagg 在三个指标上均达到最优：Kendall's τ = 0.743（相比 AZ-NAS 提升 +0.121，相比 Jacov 提升 +0.140）、Spearman's ρ = 0.906（提升 +0.092/+0.125）、nDCG@1000 = 0.664（提升 +0.056/+0.325）。值得注意的是，nDCG 作为本文新提出的指标，对头部架构识别能力敏感，VKDNWagg 在此的优势（+0.325 over Jacov）远大于传统相关性指标，说明其更适合实际 NAS 场景——搜索只需识别最优的少数架构即可。

**ImageNet-1K 实际搜索**: Table II 显示，VKDNWagg 在 MobileNetV2 ~450M FLOPs 约束下达到 **78.8% top-1 accuracy**，超越 AZ-NAS 的 78.6%（+0.2）和 ZiCo 的 78.1%（+0.7），甚至超过部分训练型方法如 OFA 的 77.7% 和 DONNA 的 78.0%。搜索成本仅 ~0.4 GPU days（100K次进化迭代约10小时），与 ZiCo、AZ-NAS 同级；但需注意最终训练仍需 7 天 8×A100。


![Figure 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/f447e043-226e-4d47-8d46-ab40ba27fe15/figures/Figure_4.png)
*Figure 4 (ablation): Supernet training is required for a fair comparison. The FLOPs and number of parameters are changed drastically. The supernet training curves on CIFAR-10 (left) and ImageNet16-120 (right). (Best viewed in color.)*



**消融实验**: Figure 3 和 Table III 展示了组件贡献。VKDNW 单代理已达 KT=0.622、SPR=0.814，优于任何其他单一组件（Jacov KT=0.603）；非线性聚合进一步提升至 KT=0.743。Table IV 定量证明 size invariance：VKDNW 与可训练层数的 KT = -0.11，而 AZ-NAS 的 expressivity = 0.56、progressivity = 0.42、trainability = 0.25。Figure 4 指出 supernet 训练对公平比较的必要性——未训练 supernet 的 FLOPs 和参数量估计偏差极大。

**公平性检查**: 比较基本公平（相同搜索空间、相同最终训练设置），但存在局限：(1) 仅测试 ~450M FLOPs 单一约束，无其他计算预算；(2) 未与 TE-NAS、Zen-NAS、GradSign 直接对比（仅在 related work 提及）；(3) 最终训练成本高昂，"zero-cost"仅适用于搜索阶段；(4) NAS-Bench-201 主实验仅用 9,445 个独特结构（非全部 15,625）。

## 方法谱系与知识库定位

**方法家族**: Zero-shot NAS → Zero-cost proxy methods

**父方法**: AZ-NAS（直接继承其多代理聚合思想，但彻底改造核心代理和聚合机制）

**改动插槽**:
- **objective**: AZ-NAS 的线性加权多代理 → VKDNW 的 Fisher 谱熵单代理 + 非线性对数乘积排序聚合 + 随机森林
- **inference_strategy**: 与网络容量强相关 → 显式追求 size invariance（KT=-0.11）
- **architecture**: 梯度统计量 → Fisher 信息矩阵 + 特征值分解
- **training_recipe**: 新增 nDCG@P 评估指标，搜索流程保持进化算法不变

**直接基线差异**:
- **ZiCo**: 梯度变异系数逆，无理论根基；VKDNW 以信息几何为理论基础
- **TE-NAS**: NTK 条件数，需特定假设；VKDNW 基于更一般的 Fisher 信息
- **Zen-NAS/GradSign**: 未直接对比，但 VKDNW 的 size invariance 是其独特优势

**后续方向**:
1. 扩展至 Transformer/ViT 架构（当前仅在 CNN 空间验证）
2. 动态调整特征值截断数 K=9 的自适应机制
3. 将 Fisher 谱分析用于训练过程中的架构动态调整（超越 zero-shot）

**标签**: #modality:vision #paradigm:zero-shot-NAS #scenario:image-classification #mechanism:information-geometry #mechanism:spectral-analysis #constraint:compute-efficient #constraint:size-invariant

## 引用网络

### 直接 baseline（本文基于）

- [[P__组装零成本代理的高效NAS方法_AZ-NAS]] _(直接 baseline)_: Zero-cost proxy assembly method; directly comparable approach in same space

