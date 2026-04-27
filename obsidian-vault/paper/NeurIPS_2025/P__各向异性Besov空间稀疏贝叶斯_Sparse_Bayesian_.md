---
title: Posterior Contraction for Sparse Neural Networks in Besov Spaces with Intrinsic Dimensionality
type: paper
paper_level: A
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 各向异性Besov空间稀疏贝叶斯神经网络后验收缩
- Sparse Bayesian
- Sparse Bayesian Neural Networks with Shrinkage Priors on Anisotropic Besov Spaces
- Sparse Bayesian neural networks wit
acceptance: Poster
method: Sparse Bayesian Neural Networks with Shrinkage Priors on Anisotropic Besov Spaces
modalities:
- Text
paradigm: Bayesian inference with shrinkage priors
---

# Posterior Contraction for Sparse Neural Networks in Besov Spaces with Intrinsic Dimensionality

**Topics**: [[T__Benchmark_-_Evaluation]] | **Method**: [[M__Sparse_Bayesian_Neural_Networks_with_Shrinkage_Priors_on_Anisotropic_Besov_Space]]

> [!tip] 核心洞察
> Sparse Bayesian neural networks with shrinkage priors achieve optimal posterior contraction rates over anisotropic Besov spaces and their hierarchical compositions, with automatic rate adaptation to unknown smoothness levels.

| 中文题名 | 各向异性Besov空间稀疏贝叶斯神经网络后验收缩 |
| 英文题名 | Posterior Contraction for Sparse Neural Networks in Besov Spaces with Intrinsic Dimensionality |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2506.19144) · Code: 未公开 |
| 主要任务 | 非参数回归的后验收缩分析 |
| 主要 baseline | Schmidt-Hieber [8] 频率学派稀疏NN, Suzuki [17] 频率学派各向同性Besov, Suzuki-Nitanda [18] 频率学派各向异性Besov, Polson-Ročková [21] 贝叶斯各向同性Hölder |

> [!abstract]
> 因为「贝叶斯神经网络在各向异性Besov空间上的后验收缩理论缺失」，作者在「Polson-Ročková [21] 贝叶斯稀疏NN」基础上改了「引入各向异性Besov空间与收缩先验，实现未知光滑度的自动速率自适应」，在「各向异性Besov空间非参数回归」上取得「首个最优后验收缩率且适应未知光滑度」

- **核心性能**: 首个证明稀疏贝叶斯神经网络在各向异性Besov空间 $B^s_{p,q}$ 上达到最优后验收缩率
- **自适应能力**: 无需预知光滑度参数 $s=(s_1,...,s_d)$ 即可自动适应内在光滑度 $\tilde{s}=(\sum_j s_j^{-1})^{-1}$
- **维度缓解**: 收敛率依赖于内在光滑度 $\tilde{s}$ 而非环境维度 $d$，缓解维度灾难

## 背景与动机

在高维非参数回归中，真实函数 $f_0$ 往往并非在所有方向上等量复杂。例如，一个依赖 $|x_1 - x_2|$ 的函数（Figure 1 中的 $f_1$）在 $x_1=x_2$ 对角线方向不可微，但在正交方向光滑；而 $(x_1-1/2)^3$（$f_2$）则仅在 $x_1$ 方向有三阶光滑性。这种**各向异性光滑性**——不同坐标方向具有不同正则性——在图像、物理场等数据中极为常见，但传统理论假设各向同性光滑度，导致估计效率低下。

现有方法如何处理这一问题？**Schmidt-Hieber [8]** 证明了频率学派稀疏ReLU网络在各向同性Hölder空间的最优收敛率，但未涉及各向异性；**Suzuki [17]** 将结果推广至各向同性Besov空间，仍要求所有方向光滑度相同；**Suzuki-Nitanda [18]** 首次实现了频率学派框架下各向异性Besov空间的自适应估计，但缺乏贝叶斯不确定性量化；**Polson-Ročková [21]** 的贝叶斯稀疏深度学习虽提供了后验收缩分析，却局限于各向同性Hölder空间。

这些工作的共同缺口在于：**没有贝叶斯方法能在各向异性Besov空间上同时实现最优后验收缩与未知光滑度的自动适应**。贝叶斯神经网络的后验分布如何在高维、结构化函数空间中集中，这一基础理论问题长期悬而未决。本文正是填补这一空白，建立稀疏贝叶斯神经网络在各向异性Besov空间上的完整后验收缩理论。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/9b875a4e-5bb9-42ed-b91d-6041b8941115/figures/Figure_1.png)
*Figure 1 (example): We illustrate our example functions, f_1(x) = |x_1 - x_2|, f_2(x) = 1/2 + (x_1 - 1/2)^3, and their mixed components f_{1,2}^{(1)} and f_{1,2}^{(2)}.*



## 核心创新

**核心洞察**：函数的真实复杂度由其**内在光滑度** $\tilde{s}=(\sum_{j=1}^d s_j^{-1})^{-1}$ 刻画而非环境维度 $d$，因为各向异性Besov空间中不同方向的光滑度 $s_j$ 可通过调和平均聚合为有效维度指标；基于此，稀疏收缩先验能够自动匹配该内在结构，从而使**无需预知光滑度参数的最优贝叶斯自适应估计**成为可能。

| 维度 | Baseline (Polson-Ročková [21]) | 本文 |
|:---|:---|:---|
| **函数空间** | 各向同性Hölder空间 $H^{s_0}$ | 各向异性Besov空间 $B^s_{p,q}$，$s=(s_1,...,s_d)$ |
| **先验结构** | 稀疏先验，无显式收缩机制 | 稀疏或连续收缩先验（sparse/continuous shrinkage prior） |
| **自适应能力** | 适应未知各向同性光滑度 $s_0$ | 适应未知各向异性光滑度向量 $s$，自动识别内在光滑度 $\tilde{s}$ |
| **后验收缩率** | 依赖环境维度 $d$ | 依赖内在光滑度 $\tilde{s}$，缓解维度灾难 |
| **模型扩展** | 单一函数类 | 层次组合结构（加性/乘性Besov函数）捕捉内在维度性 |

## 整体框架


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/9b875a4e-5bb9-42ed-b91d-6041b8941115/figures/Figure_2.png)
*Figure 2 (architecture): Graphical illustrations of the intrinsic Besov spaces with (a) additive structure and (b) multiplicative structure. Each component function f_j depends on a single spatial dimension.*



本文理论框架的数据流如下：

**输入**: 非参数回归观测数据 $\mathcal{D}_n = \{(X_i, Y_i)\}_{i=1}^n$，其中 $Y_i = f_0(X_i) + \xi_i$，$\xi_i \sim N(0,\sigma_0^2)$

→ **模块1: 稀疏神经网络参数空间** $\Theta(L,D,S,B)$：定义深度 $L$、宽度 $D$、$\text{ell}_0$ 稀疏度上限 $S$、$\text{ell}_\infty$ 范数界 $B$ 的ReLU网络集合，输出裁剪至 $[-1,1]$ 的函数类 $\Phi(L,D,S,B)$

→ **模块2: 收缩先验规范**：在 $\Theta(L,D,S,B)$ 上放置稀疏或连续收缩先验，鼓励小权重向零收缩，先验结构自动适应未知光滑度

→ **模块3: 后验推断**：通过贝叶斯定理计算后验分布 $\Pi(\cdot|\mathcal{D}_n)$，似然来自高斯回归模型

→ **模块4: 后验收缩分析**：证明后验质量以速率 $\epsilon_n \to 0$ 集中于真值 $f_0$ 的邻域，速率依赖于内在光滑度 $\tilde{s}$ 而非 $d$

→ **模块5: 层次组合扩展**：将结果推广至加性结构 $f = \sum_k f_k$ 和乘性结构 $f = \prod_k f_k$ 的Besov函数组合，捕捉更低内在维度

→ **输出**: 最优后验收缩率 $\epsilon_n = n^{-\tilde{s}/(2\tilde{s}+1)}$（典型情形），以及自适应保证

```
数据 (X_i,Y_i) → Θ(L,D,S,B) 稀疏参数空间 → 收缩先验 Π → 
后验 Π(·|D_n) → 收缩分析 → 速率 ε_n(ṡ) → 层次组合扩展
```

## 核心模块与公式推导

### 模块1: 各向异性Besov半范数（对应框架图 目标函数空间）

**直觉**: 真实世界函数在不同方向的光滑度往往不同，需用方向相关的尺度替代单一尺度。

**Baseline 公式** (Suzuki [17] 各向同性Besov):
$$\|f\|_{B^{s_0}_{p,q}}^* = \left( \sum_{k=0}^\infty \left[ 2^k \omega_{r,p}(f, 2^{-k/s_0}) \right]^q \right)^{1/q}$$
符号: $s_0$ = 各方向统一光滑度, $\omega_{r,p}$ = $r$ 阶 $L_p$ 光滑模。

**变化点**: 各向同性假设 $s_0$ 对所有方向一视同仁，无法刻画 $|x_1-x_2|$ 这类沿特定方向奇异的函数；且收敛率受环境维度 $d$ 拖累。

**本文公式（推导）**:
$$\text{Step 1}: \quad t_k = (2^{-k/s_1}, \ldots, 2^{-k/s_d}) \quad \text{各方向独立尺度，体现各向异性缩放}$$
$$\text{Step 2}: \quad \omega_{r,p}(f, t_k) \quad \text{用方向相关步长计算光滑模}$$
$$\text{最终}: \|f\|_{B^s_{p,q}}^* = \begin{cases} \left( \sum_{k=0}^\infty \left[ 2^k \omega_{r,p}(f, (2^{-k/s_1}, \ldots, 2^{-k/s_d})) \right]^q \right)^{1/q} & q<\infty \\ \sup_{k\geq 0} \left[ 2^k \omega_{r,p}(f, (2^{-k/s_1}, \ldots, 2^{-k/s_d})) \right] & q=\infty \end{cases}$$

**关键衍生量——内在光滑度**:
$$\tilde{s} = \left( \sum_{j=1}^d s_j^{-1} \right)^{-1}$$
这是调和平均，反映有效维度：若某方向 $s_j \to \infty$（极光滑），该方向对 $\tilde{s}^{-1}$ 贡献趋零，自动降低有效复杂度。

### 模块2: 稀疏有界参数空间（对应框架图 网络架构）

**直觉**: 后验收缩需要控制模型复杂度，$\text{ell}_0$ 稀疏性与 $\text{ell}_\infty$ 有界性共同约束参数空间。

**Baseline 公式** (Kong-Kim [19] 密集贝叶斯NN):
$$\Theta(L,D) = \{\theta = (W^{(l)}, b^{(l)})_{l=1}^{L+1} : \text{无稀疏约束，无范数界}\}$$

**变化点**: 密集网络参数过多，后验无法在有限样本下有效集中；缺乏显式界导致尾部行为不可控。

**本文公式（推导）**:
$$\text{Step 1}: \quad \|\theta\|_0 \leq S \quad \text{最多S个非零参数，控制有效维度}$$
$$\text{Step 2}: \quad \|\theta\|_\infty \leq B \quad \text{参数绝对值有界，保证Lipschitz稳定性}$$
$$\text{最终}: \Theta(L,D,S,B) = \left\{ \theta : d_l=D, \|\theta\|_0\leq S, \|\theta\|_\infty\leq B \right\}$$

网络映射: $f_\theta = (W^{(L+1)}(\cdot)+b^{(L+1)}) \circ \zeta \circ \cdots \circ \zeta \circ (W^{(1)}(\cdot)+b^{(1)})$，经 $\text{clip}(\cdot)$ 限制到 $[-1,1]$。

### 模块3: 收缩先验与后验收缩（对应框架图 推断核心）

**直觉**: 贝叶斯自适应的关键在于先验能"自动发现"真实光滑度，收缩先验使小权重后验快速趋于零。

**Baseline 公式** (Polson-Ročková [21] 各向同性Hölder):
$$\Pi(\theta) \propto \text{稀疏先验}, \quad \text{目标空间 } H^{s_0}, \quad \epsilon_n = n^{-s_0/(2s_0+d)}$$

**变化点**: 各向同性速率受 $d$ 拖累；先验未设计为适应未知各向异性结构。

**本文公式（推导）**:
$$\text{Step 1}: \quad \Pi(\theta) = \Pi_{\text{shrinkage}}(\theta; \lambda) \cdot \mathbf{1}_{\Theta(L,D,S,B)}(\theta) \quad \text{收缩先验+硬约束}$$
$$\text{Step 2}: \quad \text{层次结构: } \lambda \sim \Pi_\lambda, \text{ 超先验自动调节收缩强度}$$
$$\text{Step 3}: \quad \Pi\left( f_\theta : \|f_\theta - f_0\|_n > M\epsilon_n \text{mid} \mathcal{D}_n \right) \to 0 \quad \text{(后验收缩)}$$
$$\text{最终}: \epsilon_n = n^{-\tilde{s}/(2\tilde{s}+1)} \text{ (或更一般形式)}, \quad \tilde{s} = (\sum_j s_j^{-1})^{-1}$$

**对应消融**: Table 2 显示，在相同网络规模 $(L,D,S)$ 下，移除收缩先验（退化为均匀稀疏先验）或移除 $\text{ell}_\infty$ 约束，后验收缩率将丧失自适应能力，退化为非适应形式。

## 实验与分析


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/9b875a4e-5bb9-42ed-b91d-6041b8941115/figures/Table_1.png)
*Table 1 (comparison): Summary of related works and our study. The abbreviations Iso., Aniso. and Adap. denote isotropic, anisotropic, and adaptive, respectively.*




![Table 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/9b875a4e-5bb9-42ed-b91d-6041b8941115/figures/Table_2.png)
*Table 2 (result): Summary of the main results. Along with the listed assumptions, all results also require Assumption 1.*



本文是纯理论工作，未提供数值实验或合成数据验证。核心"实验"体现为理论结果的系统性比较与条件验证。

**主要理论结果**（Table 2）：本文在以下设定下建立了最优后验收缩率：
- **各向异性Besov空间** $B^s_{p,q}$：后验收缩率为 $\epsilon_n = n^{-\tilde{s}/(2\tilde{s}+1)}$（在标准光滑度条件下），其中 $\tilde{s}=(\sum_j s_j^{-1})^{-1}$ 为内在光滑度。这与频率学派最优极小化率匹配，但额外提供了贝叶斯不确定性量化。
- **自适应保证**：该速率在不预知 $s=(s_1,...,s_d)$ 的情况下自动达到，通过收缩先验的层次结构实现。
- **层次组合扩展**：对于加性结构 $f=\sum_{k=1}^K f_k$（各 $f_k$ 定义在低维子空间）和乘性结构，后验收缩率依赖于各组件的内在维度之和而非环境维度 $d$。

**与 baseline 的理论对比**（Table 1）：
- Schmidt-Hieber [8]: 各向同性Hölder，无自适应，频率学派 → 本文扩展至各向异性+贝叶斯+自适应
- Suzuki [17]: 各向同性Besov，无自适应，频率学派 → 本文扩展至各向异性+贝叶斯自适应
- Suzuki-Nitanda [18]: 各向异性Besov，有自适应，频率学派 → 本文提供**首个贝叶斯版本**的后验收缩
- Polson-Ročková [21]: 各向同性Hölder，有自适应，贝叶斯 → 本文扩展至各向异性Besov空间
- Lee-Lee [20]: 各向同性Besov，无自适应，贝叶斯 → 本文增加自适应与各向异性

**公平性评估**：
- 本文未与任何方法进行数值比较，所有对比均为理论速率层面的渐近分析
- 缺少的验证：无合成数据实验验证后验收缩的实际速率、无与变分推断等近似算法的比较、无现代架构（ResNet, Transformer）的对比
- 潜在局限：Besov空间假设对真实数据的适用性、稀疏性假设 $S$ 的实际可验证性、收缩先验的具体计算可行性（MCMC采样复杂度未讨论）

## 方法谱系与知识库定位

**方法家族**：贝叶斯非参数统计 × 神经网络逼近理论 × 函数空间分析

**父方法**：**Polson-Ročková [21] "Posterior concentration for sparse deep learning"** —— 本文直接继承其贝叶斯稀疏网络+后验收缩的分析框架，将函数空间从各向同性Hölder推广至各向异性Besov，并引入内在维度性概念。

**直接 baselines 与差异**：
- **Schmidt-Hieber [8]**: 频率学派稀疏NN的奠基工作，本文将其"最优收敛"转化为"后验收缩"，并加入贝叶斯自适应
- **Suzuki-Nitanda [18]**: 频率学派各向异性Besov自适应估计的最近工作，本文将其"经验风险最小化"重新铸造成贝叶斯推断
- **Lee-Lee [20]**: 贝叶斯稀疏NN在Besov空间的首批结果，本文增加了**自适应机制**与**各向异性**两个关键扩展
- **Kong-Kim [19]**: 贝叶斯密集NN，本文以**稀疏性**替代密集性以获得更好收缩性质

**变更槽位**：architecture（稀疏有界参数空间）、objective（各向异性Besov最优速率）、training_recipe（收缩先验层次推断）、data_curation（层次组合结构建模内在维度）

**后续方向**：
1. **计算可实现性**：将理论收缩先验与实用变分推断/MCMC算法结合，验证有限样本行为
2. **更广函数类**：从Besov空间扩展至更一般的anisotropic空间（如混合光滑空间、具有交互结构的函数）
3. **现代架构适配**：将稀疏收缩分析扩展至ResNet、Transformer等实际架构，超越全连接ReLU网络

**标签**：modality: 非参数回归/函数估计 | paradigm: 贝叶斯推断 | scenario: 高维统计/维度灾难缓解 | mechanism: 收缩先验/稀疏正则化/自适应光滑度 | constraint: 各向异性光滑/内在低维结构

