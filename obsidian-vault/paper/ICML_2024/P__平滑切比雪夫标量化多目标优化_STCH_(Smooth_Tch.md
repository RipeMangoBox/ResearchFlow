---
title: Smooth Tchebycheff Scalarization for Multi-Objective Optimization
type: paper
paper_level: C
venue: ICML
year: 2024
paper_link: null
aliases:
- 平滑切比雪夫标量化多目标优化
- STCH (Smooth Tch
- STCH (Smooth Tchebycheff Scalarization)
acceptance: Poster
cited_by: 3
code_url: https://github.com/IlkhamFY/stch-botorch-public
method: STCH (Smooth Tchebycheff Scalarization)
---

# Smooth Tchebycheff Scalarization for Multi-Objective Optimization

[Code](https://github.com/IlkhamFY/stch-botorch-public)

**Topics**: [[T__Reasoning]] | **Method**: [[M__STCH]] | **Datasets**: NYUv2 Multi-Task Learning, NYUv2 -, NYUv2 - Depth Estimation AErr, Engineering Design Problems - Runtime

| 中文题名 | 平滑切比雪夫标量化多目标优化 |
| 英文题名 | Smooth Tchebycheff Scalarization for Multi-Objective Optimization |
| 会议/期刊 | ICML 2024 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2402.19078) · [Code](https://github.com/IlkhamFY/stch-botorch-public) · [Project] |
| 主要任务 | 多目标优化 (Multi-Objective Optimization)、多任务学习 (Multi-Task Learning) |
| 主要 baseline | TCH, LS, COSMOS, EPO, DB-MTL, MGDA, MoCo, GLS, RLW, EW, STL |

> [!abstract] 因为「经典切比雪夫标量化 (TCH) 中的非光滑 max 算子导致梯度优化困难且性能不佳」，作者在「TCH」基础上改了「用 LogSumExp 平滑近似替代 max 算子，引入可微平滑参数 μ」，在「NYUv2 多任务学习基准」上取得「Δp +8.54，相比经典 TCH 的 -5.67 提升显著；深度估计 AErr 0.4965 达到最优」

- **NYUv2 Δp**: STCH +8.54 vs 经典 TCH -5.67，差距 14.21 个百分点
- **NYUv2 深度估计 AErr**: STCH 0.4965，优于 DB-MTL 的 0.5251（提升 0.0286）
- **运行效率**: STCH 与 LS/COSMOS/TCH 同为 1x，EPO 慢 36x-45x

## 背景与动机

多目标优化问题在现实世界中无处不在：自动驾驶系统需要同时优化感知精度与推理延迟，神经网络架构搜索要平衡准确率与计算开销，多任务学习要协调语义分割、深度估计等多个任务的性能。这些问题的核心难点在于目标之间往往相互冲突——改善一个指标可能损害另一个，因此需要找到一组 Pareto 最优解而非单一解。

经典切比雪夫标量化 (Tchebycheff scalarization, TCH) 是多目标优化中最基础的方法之一，它将多个目标通过 max 算子聚合为单一标量：$s_{TCH} = \max_i w_i|f_i(x) - z_i^*|$，其中 $w_i$ 为偏好权重，$z_i^*$ 为理想点 (utopian point)。TCH 的优势在于对非凸 Pareto 前沿具有良好覆盖能力，但 max 算子的非光滑性使其在基于梯度的优化中面临严峻挑战——梯度在不可微点处无定义或估计不准，导致优化过程不稳定。

现有方法从不同角度应对这一困境：线性标量化 (LS) 采用加权求和，简单高效但无法处理非凸前沿；EPO 等基于梯度的 Pareto 优化方法通过复杂的梯度操纵寻找 Pareto 驻点，但运行时间高达标量化方法的 36-46 倍；MGDA、PCGrad、CAGrad 等自适应梯度方法则试图通过梯度投影或修正来缓解任务冲突，但引入了额外的计算开销和超参数调优复杂度。这些方法的共同局限在于：要么牺牲了 TCH 对非凸前沿的理论保证，要么以极高的计算成本换取多目标均衡。

本文的核心动机直指 TCH 的"阿喀琉斯之踵"——非光滑性。作者提出一个关键问题：能否在保持 TCH 优良理论性质的同时，使其变得处处可微、适配现代深度学习框架的自动微分？

## 核心创新

核心洞察：LogSumExp 算子可以平滑逼近 max 算子，因为 LogSumExp 具有天然的可微性和良好的数值稳定性，从而使经典 TCH 能够无缝接入基于梯度的深度学习优化管线，同时保留其对非凸 Pareto 前沿的逼近能力。

| 维度 | Baseline (TCH) | 本文 (STCH) |
|:---|:---|:---|
| 聚合算子 | 非光滑 max: $\max_i w_i(f_i - z_i^*)$ | 光滑 LogSumExp: $\frac{1}{\mu}\log(\sum_i \exp(\mu \cdot w_i(f_i - z_i^*)))$ |
| 可微性 | 不可微（在多个目标相等处） | 处处可微 |
| 梯度优化 | 需次梯度或特殊处理 | 标准自动微分直接适用 |
| 理论保证 | 精确 TCH，非凸覆盖保证 | μ→∞ 时收敛到精确 TCH |
| 实现复杂度 | 简单但优化困难 | 简单且优化友好 |
| 运行效率 | 1x（标量化方法） | 1x（与 TCH/LS/COSMOS 同级） |

STCH 的唯一新增超参数是平滑系数 μ，它提供了从"完全光滑"到"精确逼近 TCH"的连续谱系，用户可根据问题特性灵活调节。

## 整体框架

STCH 的整体框架遵循标准的多目标标量化流程，核心替换发生在标量化模块：

**输入**: 多目标向量 $f(x) = (f_1(x), ..., f_m(x)) \in \mathbb{R}^m$，偏好权重 $w \in \mathbb{R}_+^m$，理想点 $z^* \in \mathbb{R}^m$

**模块 1: 目标归一化** (Objective Normalization)
- 输入: 原始目标值 $f_i(x)$
- 输出: 相对于理想点的偏差 $f_i(x) - z_i^*$（或进一步归一化至 [0,1]）
- 作用: 消除不同目标量纲差异，使标量化权重具有可比性

**模块 2: STCH 标量化** (STCH Scalarization) ⭐ 核心创新
- 输入: 归一化后的目标偏差、权重 $w_i$、理想点 $z_i^*$、平滑参数 μ > 0
- 输出: 单一标量损失值 $s_{STCH}$
- 作用: 用光滑 LogSumExp 替代 TCH 的非光滑 max，产生可微聚合损失

**模块 3: 梯度优化** (Gradient-Based Optimization)
- 输入: 标量损失 $s_{STCH}$
- 输出: 更新后的模型参数 $x$
- 作用: 标准反向传播与梯度下降/Adam 等优化器

```
原始目标 f(x) ──→ [目标归一化] ──→ f_i(x) - z_i^*
                                        ↓
偏好权重 w ───────→ [STCH 标量化] ──→ s_STCH(f; w, z*, μ)
理想点 z* ────────→      ↑ μ (平滑参数)
                         ↓
                    [梯度优化: Adam/SGD] ──→ 更新参数 x
```

整个框架的精髓在于"最小侵入式"改进——仅替换标量化函数，不改变上下游任何组件，因此可以即插即用地替换现有 MTL 或 MOO 管线中的 LS 或 TCH 模块。

## 核心模块与公式推导

### 模块 1: STCH 标量化函数（对应框架图核心位置）

**直觉**: max 算子的不可微性源于其"硬选择"特性——只关注最大项而忽略其余；LogSumExp 通过"软选择"以指数权重兼顾所有项，在保持关注最大值的同时获得光滑性。

**Baseline 公式** (经典 TCH):
$$s_{TCH}(f(x); w, z^*) = \max_{i=1,...,m} w_i \left|f_i(x) - z_i^*\right|$$

符号: $f_i(x)$ = 第 $i$ 个目标函数值, $w_i$ = 偏好权重, $z_i^*$ = 理想点 (utopian point), $m$ = 目标数

**变化点**: TCH 的 max 算子在多个目标同时达到最大时不可微，导致梯度估计困难；且绝对值 $|\cdot|$ 在零点同样不可微。实际实现中常省略绝对值（假设 $f_i(x) \geq z_i^*$ 或调整符号），但 max 的非光滑性仍是核心障碍。

**本文公式（推导）**:

$$\text{Step 1}: \quad \tilde{s} = \mu \cdot w_i \left(f_i(x) - z_i^*\right) \quad \text{（对加权偏差进行 μ 倍缩放，放大差异）}$$

$$\text{Step 2}: \quad \sum_{i=1}^{m} \exp\left(\tilde{s}_i\right) = \sum_{i=1}^{m} \exp\left(\mu \cdot w_i (f_i(x) - z_i^*)\right) \quad \text{（指数化实现软选择，大值主导求和）}$$

$$\text{Step 3}: \quad \log\left(\sum_{i=1}^{m} \exp\left(\mu \cdot w_i (f_i(x) - z_i^*)\right)\right) \quad \text{（对数恢复量级，保证数值稳定性）}$$

$$\text{最终}: \quad s_{STCH}(f(x); w, z^*, \mu) = \frac{1}{\mu} \log\left(\sum_{i=1}^{m} \exp\left(\mu \cdot w_i \left(f_i(x) - z_i^*\right)\right)\right)$$

**关键性质验证**:
- 当 $\mu \to \infty$: $\frac{1}{\mu}\log(\sum_i e^{\mu a_i}) \to \max_i a_i$（逐点收敛到精确 TCH）
- 当 $\mu \to 0^+$: 近似线性平均（过度平滑，丢失 TCH 特性）

**对应消融**: Table 5 显示不同 μ 值在 5 个真实工程设计问题上的 ∆HV 表现，μ 过大或过小均会劣化性能，存在最优区间。

---

### 模块 2: 多任务学习评估指标 Δp

**直觉**: 单一任务的绝对指标难以衡量 MTL 方法的综合优劣，需要相对于单任务基线的标准化改进度量。

**本文公式**:

$$\text{Step 1（单任务改进）}: \quad \Delta_{p,t} = 100\% \times \frac{1}{N_t} \sum_{i=1}^{N_t} (-1)^{s_{t,i}} \frac{M_{t,i} - M_{t,i}^{STL}}{M_{t,i}^{STL}}$$

其中 $s_{t,i} = 0$ 表示越大越好，$s_{t,i} = 1$ 表示越小越好；$M_{t,i}$ 为 MTL 方法指标，$M_{t,i}^{STL}$ 为单任务学习 (STL) 基线。

$$\text{Step 2（跨任务平均）}: \quad \Delta p = \frac{1}{T} \sum_{t} \Delta_{p,t}$$

**符号**: $T$ = 任务数, $N_t$ = 任务 $t$ 的指标数, $M_{t,i}^{STL}$ = 单任务基线性能

**对应实验**: Table 2 以 Δp 为主指标对比 23 种方法，STCH 获得 +8.54，经典 TCH 仅 -5.67，验证了平滑化带来的巨大收益。

## 实验与分析

本文在两大场景验证 STCH：多任务学习 (NYUv2) 与多目标优化 (合成函数 + 工程设计问题)。

{{TBL:result}}

**NYUv2 多任务学习** (Table 2): 该基准包含语义分割、深度估计、表面法向量预测三项任务。STCH 取得 Δp +8.54，在所有 23 个对比方法中位列第二，仅次于 DB-MTL 的 +8.91。关键细分指标上，STCH 在深度估计 AErr 达到 0.4965，超越 DB-MTL 的 0.5251，为该指标最优值；语义分割 mIoU 为 41.35，与最优的 DB-MTL 41.42 仅差 0.07。值得注意的是，经典 TCH 在此基准上表现极差 (Δp -5.67)，甚至劣于最简单的等权平均 (EW +0.88)，这一巨大反差强烈凸显了平滑化的必要性——但作者未充分解释 TCH 为何如此糟糕，可能存在实现或调参问题。

**多目标优化 Pareto 学习** (Figure 6): 在 6 个合成函数 (F1-F6) 和 5 个工程设计问题 (Bar Truss, Hatch Cover, Disk Brake, Gear Train, Rocket Injector) 上，STCH 以 log(∆HV) 衡量 Pareto 前沿逼近质量。STCH 普遍收敛更快、终值更低，优于经典 TCH 并与 LS/COSMOS 相当或更优。

{{TBL:ablation}}

**消融实验** (Table 5): 平滑参数 μ 的敏感性分析显示，在 5 个工程问题上存在最优 μ 区间；μ 过小导致过度平滑、丧失 TCH 的"关注最差目标"特性，μ 过大则数值不稳定且逼近硬 max 的优化困难。

**运行效率** (Table 11): STCH 与 LS、COSMOS、经典 TCH 同为 1x 基准速度，而梯度-based 的 EPO 在 Bar Truss 问题上慢 45x，在 Pareto 集合学习中慢 36x-46x。这一效率优势使 STCH 适合大规模神经网络训练场景。

**公平性审视**: 对比方法覆盖了标量化 (LS, TCH, EW)、自适应梯度 (MGDA, PCGrad, CAGrad, GradNorm)、动态权重 (DWA, UW, RLW)、以及专用 MTL 架构 (DB-MTL, Nash-MTL, Aligned-MTL)，选择较为全面。但存在几点疑虑：(1) TCH 的异常差表现 (-5.67 Δp) 与 STCH 的 +8.54 差距过大，超出"平滑化"可解释的范围，可能隐含基线实现缺陷；(2) Table 2 未报告标准差或置信区间，无法判断统计显著性；(3) 运行时间比较仅基于单种子结果；(4) 缺少与进化多目标算法 (NSGA-II, MOEA/D) 在工程问题上的对比，而这些方法在小规模问题上常具竞争力。

## 方法谱系与知识库定位

**方法家族**: 多目标标量化方法 (Multi-Objective Scalarization)

**父方法**: 经典切比雪夫标量化 (TCH) —— STCH 直接继承其"最小化最大加权偏差"的核心思想，仅对聚合算子进行光滑替换。

**改动槽位**:
| 槽位 | 父方法 TCH | 本文 STCH |
|:---|:---|:---|
| objective | 非光滑 max 算子 | 光滑 LogSumExp 近似 |
| training_recipe | 需特殊梯度处理或次梯度 | 标准自动微分，无需梯度操纵 |
| inference_strategy | 同其他标量化方法 (1x) | 保持 1x，显著快于 EPO 等梯度优化法 |

**直接基线对比**:
- **TCH**: 父方法，STCH 在 μ→∞ 时退化为 TCH；实际中 STCH 因可微性大幅优于 TCH 的实现表现
- **LS (线性标量化)**: 同为简单标量化，STCH 保留 TCH 对非凸前沿的覆盖优势，LS 不能
- **EPO**: 同为 Pareto 学习方法，STCH 以 1/36-1/46 的运行时间达到可比的超体积指标
- **DB-MTL**: 当前 NYUv2 最优方法 (+8.91 Δp)，但依赖动态分支架构；STCH (+8.54) 作为纯标量化方法接近其性能，且更通用

**后续方向**:
1. **自适应 μ 调度**: 训练初期用较小 μ 保证稳定，后期增大 μ 逼近精确 TCH
2. **与梯度修正方法结合**: STCH 提供可微损失，可与 PCGrad/CAGrad 等梯度投影技术正交互补
3. **扩展到约束多目标优化**: 将 LogSumExp 平滑思想推广至处理约束违反度的 max 聚合

**知识标签**: 
- 模态 (modality): 通用优化 / 多任务学习
- 范式 (paradigm): 标量化 (scalarization) / 基于梯度的优化
- 场景 (scenario): 多目标优化 (MOO) / 多任务学习 (MTL) / 工程设计优化
- 机制 (mechanism): LogSumExp 平滑 / 软最大化 (softmax) 逼近
- 约束 (constraint): 可微性要求 / 运行效率约束
