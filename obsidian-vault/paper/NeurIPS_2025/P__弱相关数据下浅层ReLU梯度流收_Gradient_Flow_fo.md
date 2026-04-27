---
title: Convergence of the Gradient Flow for Shallow ReLU Networks on Weakly Interacting Data
type: paper
paper_level: C
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 弱相关数据下浅层ReLU梯度流收敛理论
- Gradient Flow fo
- Gradient Flow for Shallow ReLU Networks on Weakly Interacting Data
- For one-hidden-layer ReLU networks
acceptance: Poster
method: Gradient Flow for Shallow ReLU Networks on Weakly Interacting Data
modalities:
- Text
- Image
paradigm: supervised
---

# Convergence of the Gradient Flow for Shallow ReLU Networks on Weakly Interacting Data

**Topics**: [[T__Benchmark_-_Evaluation]], [[T__Self-Supervised_Learning]] | **Method**: [[M__Gradient_Flow_for_Shallow_ReLU_Networks_on_Weakly_Interacting_Data]] | **Datasets**: Synthetic convergence threshold scaling, Local-PL curvature scaling, Finite-width convergence probability, Scaling law in dimension d, Scaling law in width p

> [!tip] 核心洞察
> For one-hidden-layer ReLU networks trained by gradient flow on n weakly interacting (low-correlated/high-dimensional) data points, a width of order log(n) neurons suffices for global convergence with high probability, with exponential convergence rate of 1/n.

| 中文题名 | 弱相关数据下浅层ReLU梯度流收敛理论 |\n| 英文题名 | Convergence of the Gradient Flow for Shallow ReLU Networks on Weakly Interacting Data |\n| 会议/期刊 | NeurIPS 2025 (Poster) |\n| 链接 | [arXiv](https://arxiv.org/abs/2505.0xxxx) · Code: 未公开 · Project: 未公开 |\n| 主要任务 | 神经网络优化理论 / 梯度流收敛分析 |\n| 主要 baseline | NTK/lazy regime (Jacot et al., 2018); Mean-field/infinite-width (Mei et al., 2018; Chizat & Bach, 2018); 正交输入梯度流 (Boursier & Flammarion, 2024a,b) |\n\n> [!abstract] 因为「有限宽度浅层ReLU网络在非NTK/非平均场 regime 下的全局收敛保证缺失」，作者在「Boursier & Flammarion (2024a,b) 正交输入梯度流分析」基础上改了「将精确正交性放松为弱相互作用数据假设，引入时变PL不等式与显式宽度标度律」，在「合成弱相关数据」上取得「宽度 p = O(log n) 即可高概率全局收敛，收敛率 O(1/n)」\n\n- **宽度标度**：神经元数 p = ⌊log(n/ε)/log(4/3)⌋ + 1 足以保证全局收敛，远优于 NTK 的无限宽度要求\n- **收敛速率**：弱相互作用数据下指数收敛率 O(1/n)，正交数据下出现 O(1/n) 到 O(1/√n) 的相变\n- **实验验证**：MacBook Air CPU 上 2 小时内完成，验证理论阈值与 PL 曲率 K/√n 标度猜想

## 背景与动机

神经网络优化的理论理解存在根本性张力：实践中，有限宽度的浅层网络通过梯度下降就能有效学习；但理论保证却几乎总是要求无限宽度（NTK regime 或平均场极限）。例如，一个宽度为 100 的单隐藏层 ReLU 网络在 CIFAR-10 子集上训练时，其参数在训练过程中发生显著变化（feature learning），而非停留在初始化附近的线性近似区域——但现有理论无法为这种 commonplace 场景提供收敛保证。\n\n现有方法如何处理这一问题？**NTK/lazy regime 分析**（Jacot et al., 2018; Arora et al., 2019）假设网络宽度趋于无穷或初始化尺度精心调校，使得网络在整个训练过程中近似于固定核的线性模型，从而利用核方法的正定性证明收敛。**平均场/无限宽度分析**（Mei et al., 2018; Chizat & Bach, 2018）将参数演化视为 Wasserstein 空间中的测度流，在 p → ∞ 极限下建立全局收敛，但同样回避了有限宽度情形。**精细调优初始化尺度方法**（Boursier et al., 2022）通过特定初始化大小控制早期动态，但仍属于高度约束的设置。**正交输入分析**（Boursier & Flammarion, 2024a,b）是最近的重要进展，证明了精确正交数据下梯度流的全局收敛，但要求 xi⊥xj 的严格条件难以满足。\n\n这些方法的核心短板在于：NTK 和平均场需要 p → ∞；正交分析需要精确 xi⊥xj；而实际数据仅是「近乎正交」的高维随机向量。本文将精确正交性放松为「弱相互作用」假设，证明 O(log n) 宽度即可实现全局收敛，填补了有限宽度、标准初始化、非精确正交数据三者兼得的理论空白。

## 核心创新

**核心洞察**：高维空间中弱相关数据的低相关性结构可被显式量化并嵌入 PL 常数的下界，因为随机单位向量在 S^{d-1} 上的内积以高概率为 O(1/√d) 量级，从而使仅 O(log n) 个神经元的覆盖论证即可保证每个样本至少被一个神经元激活，进而维持正的 PL 曲率。\n\n| 维度 | Baseline (Boursier & Flammarion 2024a,b) | 本文 |\n|:---|:---|:---|\n| 数据假设 | 精确正交输入 xi⊥xj | 弱相互作用（近乎正交）：xi/‖xi‖ ~ U(S^{d-1}) |\n| 宽度要求 | 未显式给出有限宽度标度 | p = ⌊log(n/ε)/log(4/3)⌋ + 1 = O(log n) |\n| 收敛机制 | 基于正交结构的简化梯度分析 | 时变 PL 不等式 dL/dt ≤ -μ(t)L(t)，μ(t) 显式下界依赖数据相关性 |\n| 收敛率 | O(1/n)（单一 regime） | 弱相互作用：O(1/n)；正交数据：O(1/n) 到 O(1/√n) 相变 |\n| 初始化条件 | 特定尺度要求 | 标准条件：\|aj(0)\| ≥ ‖wj(0)‖ 即可 |

## 整体框架


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/d6a2c5ed-b02d-4c9e-9c4e-d55494c401c5/figures/Figure_3.png)
*Figure 3 (example): Example of group initialization with μ∞ = 3, k∞ = 1, s = 2 (support per neuron), K = 3 (number of groups), in dimension d = 2. Each group is represented by a pair of neurons shown in the same color. We show the profiles fk in a Cartesian network.*

\n\n本文的理论框架由四个顺序模块构成，形成从初始化到收敛认证的完整 pipeline：\n\n**输入**：n 个高维弱相互作用样本 {(xi, yi)}，其中 ‖xi‖ ~ U([1,2])，方向 xi/‖xi‖ ~ U(S^{d-1})。\n\n1. **初始化模块（Initialization）**：为单隐藏层 ReLU 网络 f(x) = Σ_{j=1}^p aj σ(wj^T x) 采样参数，要求 |aj(0)| ≥ ‖wj(0)‖。该条件确保训练过程中第二层权重 aj 的符号稳定，防止 collapse 到单一方向。\n\n2. **梯度流动力学（Gradient Flow Dynamics）**：连续时间动力学 dθ/dt = -∇L(θ)，其中 L(θ) = (1/2n)Σ_i (f(xi)-yi)^2 为平方损失。与 NTK 的线性化动力学不同，此处允许参数大幅偏离初始化（feature learning regime）。\n\n3. **PL 曲率监控（PL Inequality Verification）**：核心创新模块。在每个时刻 t，计算局部 PL 常数 μ(t) 的显式上下界 μ_low 和 μ_upp，这些界仅依赖于当前第二层权重 |aj|^2 和神经元-数据对齐指示函数 1_{j,i} = 1_{wj^T xi > 0}。\n\n4. **收敛认证（Convergence Certification）**：当损失降至 L < C_y^-/(2n) 阈值时，或当积分 PL 不等式保证 L(t) ≤ L(0)exp(-∫μ(s)ds) 足够小时，判定全局收敛。\n\n```\n弱相互作用数据 (xi,yi) \n    ↓\n初始化: |aj(0)| ≥ ‖wj(0)‖, p = O(log n) 神经元\n    ↓\n梯度流: dθ/dt = -∇L(θ)  [feature learning regime]\n    ↓\nPL监控: μ_low(t) ≤ μ(t) ≤ μ_upp(t)  [时变曲率界]\n    ↓\n收敛判定: L(t) ≤ L(0)exp(-∫μ(s)ds) → 0\n```

## 核心模块与公式推导

### 模块 1: 时变 PL 不等式与局部曲率界（对应框架图「PL监控」位置）\n\n**直觉**：标准优化理论假设全局 PL 常数 μ > 0，但神经网络损失景观高度非凸且曲率随轨迹演化。本文将 PL 条件局部化并显式参数化，使其可追踪。\n\n**Baseline 公式** (NTK/lazy regime): 线性化模型 f_t ≈ f_0 + Θ(θ_t - θ_0)，其中 Θ 为固定的 Neural Tangent Kernel。收敛由 Θ 的正定性保证：\n$$L(t) \leq L(0)\exp(-\lambda_{\min}(\Theta) \cdot t)$$\n符号: Θ = 无限宽度 NTK 矩阵，λ_min(Θ) > 0 为其最小特征值。\n\n**变化点**：NTK 要求 p → ∞ 且参数冻结在初始化附近；实际有限宽度网络中 Θ 随训练演化，λ_min 可能变小甚至消失。本文放弃固定核假设，直接对 ReLU 网络的梯度流建立时变 PL 关系。\n\n**本文公式（推导）**:\n$$\text{Step 1}: \frac{dL}{dt} = -\|\nabla L\|^2 \leq -\mu(t)L(t) \quad \text{链式法则 + PL 条件假设}$$\n$$\text{Step 2}: \mu_{\text{low}} = \frac{2}{n} \min_{i} \frac{1}{p}\sum_{j=1}^{p} |a_j|^2 \mathbf{1}_{j,i} \quad \text{通过 ReLU 梯度计算显式下界}$$\n$$\text{Step 3}: \mu_{\text{upp}} = \frac{16}{n} \max_{i} \frac{1}{p}\sum_{j=1}^{p} |a_j|^2 \mathbf{1}_{j,i} \quad \text{对称结构上界，常数因子 16 来自梯度范数估计}$$\n$$\text{最终}: L(t) \leq L(0)\exp\left(-\int_0^t \mu_{\text{low}}(s)\,ds\right)$$\n符号: aj = 第 j 神经元第二层权重；1_{j,i} = 1_{wj^T xi > 0} 为 ReLU 激活指示函数；n = 样本数；p = 神经元数。\n\n**对应消融**：去掉时变 μ(t) 假设（退化为静态 PL）无法捕捉训练后期的曲率衰减；Figure 6 显示 μ(t_∞) 与 ⟨μ_∞⟩ 均随 n 增大而衰减，斜率约 -1/2，证实静态 PL 的失效。\n\n### 模块 2: 宽度标度律与覆盖论证（对应框架图「初始化」位置）\n\n**直觉**：高维球面 S^{d-1} 上随机向量的覆盖需要指数多个点，但仅需对数个点即可保证每个数据方向被至少一个神经元「看见」（激活）。\n\n**Baseline 公式** (Mean-field): p → ∞，参数演化由 Wasserstein 空间中的 PDE 描述：\n$$\partial_t \rho = \nabla \cdot (\rho \nabla V) + \Delta \rho$$\n符号: ρ(t, θ) = 参数分布的密度，V = 损失泛函的变分梯度。\n\n**变化点**：平均场极限回避了「需要多少神经元」的问题；本文需要有限 p 的显式公式。\n\n**本文公式（推导）**:\n$$\text{Step 1}: \mathbb{P}(\exists i: \forall j, \mathbf{1}_{j,i} = 0) \leq n \cdot \left(\frac{3}{4}\right)^p \quad \text{单个样本未被覆盖的概率，3/4 来自高维随机向量夹角分布}$$\n$$\text{Step 2}: \text{令上式} \leq \varepsilon \text{（失败概率）} \Rightarrow n \cdot (3/4)^p \leq \varepsilon$$\n$$\text{最终}: p = \left\lfloor \frac{\log(n/\varepsilon)}{\log(4/3)} \right\rfloor + 1 = O(\log n)$$\n\n**对应消融**：Figure 4 显示固定 p 时收敛阈值 N(d,p) 随维度 d 线性增长；Figure 5 显示固定 d 时阈值随 p 次线性/对数增长，与理论预测一致。若 p 低于此阈值，全局收敛概率骤降。\n\n### 模块 3: 正交数据下的相变刻画（对应框架图「收敛认证」位置）\n\n**直觉**：精确正交数据使 PL 常数的演化可精确追踪，揭示收敛率在不同参数区域的突变——类似统计物理中的相变。\n\n**Baseline 公式** (Boursier & Flammarion 2024a,b): 正交数据下收敛率 O(1/n)，无精细 regime 区分。\n\n**变化点**：正交数据的额外结构使 μ(t) 的长时间行为可被更精细分析，发现单一 O(1/n) 率仅覆盖部分参数空间。\n\n**本文公式（推导）**:\n$$\text{Step 1}: \langle \mu_\infty \rangle = \log\left(\frac{L(0)}{L(t_\infty)}\right) = \frac{1}{t_\infty}\int_0^{t_\infty} \mu(s)\,ds \quad \text{平均 PL 曲率定义}$$\n$$\text{Step 2}: \mu(t_\infty) = \log\left(\frac{L(t_\infty-1)}{L(t_\infty)}\right) \quad \text{训练结束时的瞬时曲率}$$\n$$\text{Step 3}: \text{Conjecture 1}: \langle \mu_\infty \rangle \sim K/\sqrt{n}, \quad \mu(t_\infty) \sim K/\sqrt{n} \quad \text{大 } n \text{ 渐近}$$\n$$\text{最终}: \text{收敛率介于 } O(1/n) \text{（早期/小 } n\text{）与 } O(1/\sqrt{n}) \text{（晚期/大 } n\text{）之间，出现相变}$$\n\n**对应消融**：Figure 6（log-log 图）显示四种曲率度量的斜率接近 -1/2，支持 K/√n 标度；若正交性假设被放松为弱相互作用，相变现象平滑化，回归单一 O(1/n) 率。

## 实验与分析

\n\n本文实验完全在合成数据上进行，使用 MacBook Air CPU（无 GPU 加速），单组实验不超过 2 小时。核心验证围绕两个理论预测展开。\n\n**实验一：宽度标度律验证**。在维度 d=2 至 100、神经元数 p=3 至 400 的范围内，生成弱相互作用数据（方向均匀分布于 S^{d-1}，范数均匀于 [1,2]），运行梯度流至 t_∞ = 1.5 × √(np)/(4 log(np))。Figure 4 显示固定 p 时，收敛阈值样本数 N(d,p) 随 d 线性增长；Figure 5 显示固定 d 时，N(d,p) 随 p 呈对数/次线性下降，与 Lemma 5 的 p = ⌊log(n/ε)/log(4/3)⌋ + 1 预测高度吻合。关键数值：当 p=30 时，d=50 情形下 n≈2000 仍能以 >90% 概率收敛至零损失；而 p 低于理论阈值时收敛概率断崖式下跌。\n\n
![Figure 5](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/d6a2c5ed-b02d-4c9e-9c4e-d55494c401c5/figures/Figure_5.png)
*Figure 5 (result): Left: evolution of global convergence in dimension d. We observe a phase transition around n ∼ d^2/k^2_∞ s. Right: the heatmap of σ_min(Φ) (Rademacher complexity-like term) as n increases. Here, k_∞ = 3, s = 1. The loss landscape is L(θ), which is complex due to the loss geometry.*

\n\n**实验二：PL 曲率标度与相变**。对正交数据（精确 xi⊥xj）运行梯度流，测量四种曲率指标：μ_low、μ_upp、⟨μ_∞⟩、μ(t_∞)。Figure 6 的 log-log 坐标显示，⟨μ_∞⟩ 和 μ(t_∞) 随 n 的斜率约为 -0.52 和 -0.48，接近 Conjecture 1 预测的 -1/2。这意味着训练后期的有效收敛率从早期的 O(1/n) 过渡到 O(1/√n)，形成明确的相变现象。Figure 5 的右子图进一步以 σ_min(Φ) 热图可视化了该相变在 (n, d) 参数空间中的边界。\n\n\n\n**消融与敏感性**：初始化条件 |aj(0)| ≥ ‖wj(0)‖ 是理论关键——若违反此条件，第二层权重 aj 可 collapse 至同号同向，导致网络退化为单神经元等价类，无法达到零损失（附录 C.1 理论分析，Figure 4 的对比实验支持）。弱相互作用假设的放松（从精确正交到近乎正交）使相变现象消失，回归单一 O(1/n) 率，验证了正交结构的特殊性。\n\n**公平性检查**：本文 baselines 为理论方法（NTK、平均场、正交输入分析），未与实用 SGD、Adam 或真实数据集比较，这是理论工作的合理范围但限制了实践指导价值。实验规模小（CPU 即可），无现代大尺度验证。Conjecture 1 的 K/√n 标度获实验支持但尚未严格证明。作者明确披露：分析仅限单隐藏层，深层网络扩展不明；梯度流而非实用梯度下降；弱相关假设在真实数据中可能不成立。

## 方法谱系与知识库定位

**方法家族**：神经网络优化理论 → 梯度流分析 → 有限宽度非懒惰训练收敛保证\n\n**父方法**：Boursier & Flammarion (2024a,b)「Gradient flow dynamics of shallow ReLU networks for square loss and orthogonal inputs」。本文直接扩展其正交输入分析至弱相互作用数据，保留梯度流框架，新增宽度标度律与相变刻画。\n\n**直接 baselines 与差异**：\n- **NTK/lazy regime**（Jacot et al., 2018; Chizat et al., 2019）：需 p → ∞ 或精心调校初始化；本文以 O(log n) 有限宽度替代，机制从核线性化转为 PL 不等式。\n- **Mean-field/infinite-width**（Mei et al., 2018; Chizat & Bach, 2018）：测度极限分析；本文给出有限 p 的显式公式，覆盖论证替代最优传输。\n- **精细初始化尺度**（Boursier et al., 2022）：依赖特定初始化大小；本文仅需标准 |aj| ≥ ‖wj‖ 条件。\n\n**变化槽位**：architecture（O(log n) 宽度标度）、objective（时变 PL 不等式替代静态/核假设）、training_recipe（标准梯度流替代 NTK 线性化或平均场极限）、data_curation（新增弱相互作用数据假设）。\n\n**后续方向**：(1) 离散时间梯度下降（非连续流）的对应分析；(2) 深层网络（≥2 隐藏层）的扩展，核心障碍为 PL 结构的层级传播；(3) 真实高维数据（如预训练嵌入）的弱相关性实证检验与理论适配。\n\n**知识库标签**：modality=理论/合成数据；paradigm=监督学习；scenario=有限宽度神经网络优化；mechanism=Polyak-Łojasiewicz 不等式/高维几何覆盖；constraint=单隐藏层/连续时间/弱相关数据假设。

