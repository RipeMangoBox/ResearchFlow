---
title: Learning Sparse Approximate Inverse Preconditioners for Conjugate Gradient Solvers on GPUs
type: paper
paper_level: A
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- GNN生成稀疏近似逆预处理器加速GPU共轭梯度求解
- GNN-SPAI
- A GNN-based approach to construct S
acceptance: Poster
cited_by: 3
code_url: https://github.com/Adversarr/LearningSparsePreconditioner4GPU
method: GNN-SPAI
modalities:
- graph
paradigm: supervised
---

# Learning Sparse Approximate Inverse Preconditioners for Conjugate Gradient Solvers on GPUs

[Code](https://github.com/Adversarr/LearningSparsePreconditioner4GPU)

**Topics**: [[T__Reasoning]], [[T__Math_Reasoning]] | **Method**: [[M__GNN-SPAI]] | **Datasets**: Heat problem GPU, Li et al., Häusner et al., OOD Heat-Density GPU, GPU preconditioner

> [!tip] 核心洞察
> A GNN-based approach to construct Sparse Approximate Inverse (SPAI) preconditioners avoids triangular solves, requires only matrix-vector products compatible with GNN locality, and achieves 40%-53% reduction in GPU solution time compared to standard and prior learning-based preconditioners.

| 中文题名 | GNN生成稀疏近似逆预处理器加速GPU共轭梯度求解 |
| 英文题名 | Learning Sparse Approximate Inverse Preconditioners for Conjugate Gradient Solvers on GPUs |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2510.27517) · [Code](https://github.com/Adversarr/LearningSparsePreconditioner4GPU) · [Project](未提供) |
| 主要任务 | 偏微分方程数值求解中的线性系统迭代求解加速 |
| 主要 baseline | Diagonal preconditioner, Incomplete Cholesky (IC), Traditional SPAI (AINV), Li et al. [4] learning-based IC, Häusner et al. [5] Neural IF |

> [!abstract] 因为「不完全分解预处理器需要三角求解，阻碍GPU并行化且难以适配GNN局部传播」，作者在「Learning from linear algebra [6][9]」基础上改了「用GNN直接生成稀疏近似逆矩阵替代分解结构，并设计尺度不变损失函数」，在「GPU热方程基准测试」上取得「总求解时间相比IC减少4.0%、相比AINV减少40.3%、相比Häusner et al. [5]减少87.3%」

- **GPU总时间**：heat problem上 T_total = 197 ms，比 Diagonal (520 ms) 快 62.1%，比 IC (205 ms) 快 3.9%，比 AINV (330 ms) 快 40.3%
- **跨架构优势**：Häusner et al. [5] 数据集上 GPU 总时间 132 ms，比该方法的 1040 ms 快 87.3%；CPU 上则为 1320 ms，比 Häusner et al. 慢 15.9%
- **OOD泛化**：Heat-Large 问题相对提升 62% (out-of-distribution) / 73% (in-distribution)，Hyperelasticity 72%/68%

## 背景与动机

求解大规模稀疏线性系统 $Ax=b$ 是科学计算的核心任务，其中共轭梯度（Conjugate Gradient, CG）方法是求解对称正定稀疏系统的首选迭代算法。然而，CG 的收敛速度严重依赖系数矩阵 $A$ 的条件数——当 $A$ 来自有限元离散化的偏微分方程时，条件数往往极大，导致迭代次数爆炸。预处理器通过构造近似逆 $M^{-1} \approx A^{-1}$ 将原系统转化为 $M^{-1}Ax = M^{-1}b$，使预处理后的矩阵 $AM^{-1}$ 接近单位阵，从而加速收敛。

现有方法主要分为三类：**Diagonal preconditioner** 仅提取对角线，计算极简但效果有限；**Incomplete Cholesky (IC)** 通过不完全分解 $A \approx LL^T$ 构造预处理器，效果较好但每次 CG 迭代需执行前向/后向三角求解（forward/backward substitution）；**传统 SPAI (AINV)** 直接构造稀疏近似逆矩阵，避免了分解结构，但通常依赖预设的稀疏模式且构造代价高昂。

近期学习-based 方法试图用图神经网络（GNN）生成预处理器。Li et al. [4] 学习不完全分解因子，Häusner et al. [5] 提出 Neural Incomplete Factorization——然而这些方法仍保留三角分解结构，导致两个根本性瓶颈：其一，**三角求解的串行依赖性严重阻碍 GPU 并行化**，每次迭代中的前向/后向替换无法充分利用 GPU 的 SIMT 架构；其二，**三角分解引入长程依赖**，与 GNN 的消息传递机制（局部邻居聚合）天然不匹配，限制了 GNN 的表达能力。

本文提出 GNN-SPAI，核心思路是：让 GNN 直接生成稀疏近似逆矩阵的显式条目，彻底消除三角求解，使每次 CG 迭代仅需两次矩阵-向量乘积（SpMV），实现与 GPU 并行架构的完全兼容。


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/5990ae3a-df95-4864-8a24-890995d5e5de/figures/Figure_2.png)
*Figure 2 (example): Figure 2: Examples in our FEM-derived test cases.*



## 核心创新

核心洞察：**CG 的收敛速度仅取决于预处理矩阵的条件数而非绝对尺度**，因此预处理器训练损失应当尺度不变；同时，**稀疏近似逆的显式结构天然适配 SpMV 操作**，使 GPU 并行化与 GNN 局部消息传递达成统一，从而使学习-based 预处理器在 GPU 上首次超越传统方法成为可能。

| 维度 | Baseline [6][9] / IC-based 方法 | 本文 GNN-SPAI |
|:---|:---|:---|
| 预处理器结构 | 不完全分解 $M = LL^T$ 或三角因子 | 显式稀疏近似逆 $M^{-1}$，直接满足 $AM^{-1} \approx I$ |
| CG 迭代操作 | 三角求解（前向/后向替换） | 两次稀疏矩阵-向量乘积（SpMV） |
| 训练损失 | L2 损失或余弦相似度损失（尺度敏感） | 基于统计量的尺度不变损失 $L_{\text{SAI}}$（匹配条件数收敛特性） |
| GNN 适配性 | 长程依赖难以用局部消息传递建模 | 稀疏逆矩阵条目与图节点特征一一对应，天然局部化 |

## 整体框架


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/5990ae3a-df95-4864-8a24-890995d5e5de/figures/Figure_1.png)
*Figure 1 (pipeline): Figure 1: Overview of our approach. By equating the matrix's nonzero pattern S_{i,j} and node features F_{i,j} to a graph, the message passing in GNNs is equivalent to the sparse matrix-vector product in linear algebra. The resultant matrix G is assembled and then applied to the preconditioned CG solver.*



GNN-SPAI 的完整数据流将稀疏矩阵视为图，通过 GNN 生成预处理器，最终嵌入标准 CG 求解器：

**输入**：稀疏线性系统矩阵 $A \in \mathbb{R}^{N \times N}$（来自 PDE 离散化或合成数据），以及右端项 $b$。将 $A$ 的非零模式 $S_{i,j}$ 和节点特征 $F_{i,j}$ 编码为图结构。

**GNN Encoder**：以矩阵 $A$ 的图表示为输入，通过多层消息传递提取节点嵌入。关键设计：仅聚合局部邻居信息，与稀疏逆矩阵的局部支撑集（local support）保持一致。

**SPAI Entry Decoder**：将节点嵌入解码为稀疏近似逆矩阵 $M^{-1}$ 的非零条目。与 baseline [6][9] 的本质区别：直接输出 $M^{-1}$ 的条目，而非 $L$ 或 $L^T$ 因子。

**显式稀疏矩阵构建**：将解码得到的条目组装为 CSR/CSC 格式的稀疏矩阵 $M^{-1}$，稀疏模式可与 $A$ 相同或更稀疏。

**预处理 CG 求解器**：标准 CG 迭代，但每步的预处理操作 $z = M^{-1}r$ 通过两次 SpMV 完成（具体实现中 $M^{-1}$ 显式存储），无需任何三角求解。

**SAI 损失计算模块**：在训练阶段，基于采样向量 $w$ 计算预处理矩阵 $AM^{-1}$ 的统计特性，驱动 GNN 优化。

```
A(sparse matrix graph) → [GNN Encoder] → node embeddings
                                      ↓
                           [SPAI Entry Decoder]
                                      ↓
                           M^{-1}(explicit sparse matrix)
                                      ↓
                    ┌─────────────────┴─────────────────┐
                    ↓                                   ↓
            [Training: L_SAI loss]              [Inference: CG solver]
            (condition number stats)              (two SpMVs per step)
```

## 核心模块与公式推导

### 模块 1: 尺度不变稀疏近似逆损失 $L_{\text{SAI}}$（对应框架图"SAI loss computation"模块）

**直觉**：CG 的收敛界 $\|e_k\|_A \leq 2\left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^k \|e_0\|_A$ 明确显示收敛速度由条件数 $\kappa$ 决定，与矩阵绝对尺度无关；因此训练损失应当尺度不变，否则优化目标与真实求解效率错位。

**Baseline 公式 (L2 loss)**:
$$L_2 = \|AM^{-1}w - w\|_2^2$$
符号: $w$ = 随机采样向量, $M^{-1}$ = GNN 生成的稀疏近似逆, $A$ = 原系统矩阵。

**变化点**：$L_2$ 是尺度敏感的——若 $A$ 整体缩放 $\alpha$ 倍，$L_2$ 缩放 $\alpha^2$ 倍，但 CG 迭代次数不变（条件数不变）。这导致 GNN 可能优化"数值大小"而非"收敛效率"。

**本文公式（推导）**:
$$\text{Step 1}: \text{利用条件数性质 } \kappa(AM^{-1}) = \frac{\max_i \lambda_i}{\min_i \lambda_i}, \quad \kappa_{\text{Kaporin}}(AM^{-1}) = \frac{(\sum_{i=1}^N \lambda_i)/N}{(\prod_{i=1}^N \lambda_i)^{1/N}}$$
$$\text{Step 2}: \text{设计统计量使损失对矩阵缩放 } \alpha A \text{ 保持不变，同时保持与 } \kappa \text{ 的相关性}$$
$$\text{最终}: L_{\text{SAI}} = f\left(\text{statistics of } AM^{-1}\right) \text{ (scale-invariant, condition-number-aware)}$$
具体实现通过 $AM^{-1}w$ 的采样统计量构造，避免直接特征值分解的 $O(N^3)$ 代价。

**对应消融**：Table 6 显示，在 Synthetic 困难问题上，$L_2$ 导致迭代次数 2109.8，$L_{\text{SAI}}$ 仅 1122.0，差距 987.8 次迭代（+88.0%）；$L_{\text{CS}}$ 更差达 2185.7 次（+94.8%）。

### 模块 2: 显式 SPAI 结构与 GPU-并行 CG 迭代（对应框架图"Preconditioner M^{-1}"至"CG solver"）

**直觉**：稀疏矩阵-向量乘积（SpMV）是 GPU 上高度优化的核心算子，而三角求解因数据依赖难以并行；将预处理器从"隐式分解"转为"显式逆矩阵"，使 CG 的每步预处理变为纯 SpMV。

**Baseline 公式 (IC preconditioned CG)**:
$$\text{Solve } LL^T z = r \text{ by forward/backward substitution}$$
符号: $L, L^T$ = 不完全分解因子, $r$ = CG 残差, $z$ = 预处理后的搜索方向。

**变化点**：前向替换 $Lz = r$ 中 $z_i$ 依赖 $z_1, ..., z_{i-1}$，形成长程串行链；GPU warp 内线程因等待前置结果而大量空闲。AINV 虽显式存储逆，但构造阶段仍需三角计算。

**本文公式（推导）**:
$$\text{Step 1}: M^{-1} \leftarrow \text{GNN}(A) \text{ 直接生成，稀疏模式 } S \subseteq \{(i,j) : A_{ij} \neq 0\}$$
$$\text{Step 2}: \text{CG step: } z = M^{-1}r \text{ via SpMV}$$
$$\text{最终}: \text{Each CG iteration: } 2 \times \text{SpMV}(M^{-1}, \cdot) + 2 \times \text{SpMV}(A, \cdot) + \text{vector ops}$$
关键：$M^{-1}$ 的稀疏结构与 $A$ 的非零模式对齐，确保 SpMV 计算量可控；GNN 的 $k$-hop 消息传递范围与 $M^{-1}$ 的 $k$-step 填充模式对应。

**对应消融**：Table 3 时间分解显示，GNN-SPAI 的 $T_{\text{construct}} = 0.18$ ms，$T_{\text{apply}} = 0.29$ ms；IC 的 $T_{\text{construct}} = 1.88$ ms（高 10.4 倍），$T_{\text{apply}} = 0.80$ ms（高 2.8 倍）；AINV 的 $T_{\text{construct}} = 19.0$ ms（高 105.6 倍）。

### 模块 3: GNN 架构适配（对应框架图"GNN Encoder"至"SPAI Decoder"）

**直觉**：稀疏逆矩阵 $M^{-1}$ 的每个条目 $(M^{-1})_{ij}$ 仅依赖 $A$ 的局部子图——若 $A$ 对应网格的邻接关系，则 $(M^{-1})_{ij}$ 的精度由 $i,j$ 间的局部拓扑决定，恰好匹配 GNN 的消息传递范围。

**Baseline 架构 ([6][9])**: GNN 生成三角因子 $L$ 的条目，但 $L_{ij}$ 的数值依赖 $A$ 的全局消去顺序（fill-in 的长程传播）。

**变化点**：三角分解的 fill-in 路径可能跨越整个图，要求 GNN 捕获全局依赖；而 SPAI 的局部支撑集可显式控制。

**本文设计**：
$$\text{Node feature}: F_i = \text{diag}(A)_i, \text{ degree}_i, \text{ etc.}$$
$$\text{Message passing}: h_i^{(l+1)} = \text{UPDATE}\left(h_i^{(l)}, \text{AGGREGATE}_{j \in \mathcal{N}(i)} \text{MESSAGE}(h_i^{(l)}, h_j^{(l)}, e_{ij})\right)$$
$$\text{Output}: (M^{-1})_{ij} = \text{DECODER}(h_i^{(L)}, h_j^{(L)}) \text{ for } (i,j) \in S$$
其中 $S$ 为预设的稀疏模式（通常取 $A$ 的非零模式），$L$ 为 GNN 层数，控制局部感受野范围。

## 实验与分析


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/5990ae3a-df95-4864-8a24-890995d5e5de/figures/Table_1.png)
*Table 1 (quantitative): Table 1: GPU benchmark results across different datasets. Total time T_total (ms) and total iterations #iter are reported. Our approach reduces the total solve time and iteration counts across all datasets. The best value is in bold.*



本文在 GPU 和 CPU 双平台上评估，覆盖热方程（heat）、超弹性（hyperelasticity）及合成线性系统。核心结果来自 Table 1 和 Table 3-4：在 heat problem 的 GPU 基准上，GNN-SPAI 总时间 $T_{\text{total}} = 197$ ms（197 次迭代），相比 Diagonal 的 520 ms（468 次迭代）减少 62.1% 时间、57.9% 迭代；相比 IC 的 205 ms 减少 3.9% 时间，但构造时间 $T_{\text{prep}}$ 从 1.88 ms 降至 0.18 ms（快 10.4 倍）；相比传统 AINV 的 330 ms 减少 40.3% 时间，且构造时间从 19.0 ms 降至 0.18 ms（快 105.6 倍）。这一"构造极快、应用略快"的特性使 GNN-SPAI 在需要多次求解相似系统（如物理模拟时间步进）的场景优势显著。


![Table 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/5990ae3a-df95-4864-8a24-890995d5e5de/figures/Table_2.png)
*Table 2 (comparison): Table 2: Comparison between different preconditioners for the heat problem. Total time T_total (ms), construction time T_prep (ms), iteration counts #iter, and relative residual res are reported. The best value is in bold.*



跨方法对比（Table 4）揭示架构特异性：在 Li et al. [4] 小矩阵 CPU 测试中，GNN-SPAI 总时间 17 ms 落后于 Diagonal 的 12 ms（慢 41.7%），但优于 Li et al. [4] 的 26 ms；然而在 GPU 上，GNN-SPAI 的 26 ms 反超 Diagonal 的 29 ms（快 10.3%），并大幅领先 Li et al. [4] 的 51 ms（快 49.0%）。Häusner et al. [5] 数据集上差距更悬殊：GPU 上 GNN-SPAI 132 ms 对比 Häusner et al. 1040 ms（快 87.3%），尽管后者迭代次数更少（354 vs 456），但其 Neural IF 的构造与应用开销极高。这验证了**本文方法的核心优势是 GPU 并行效率，而非纯迭代收敛速度**。


![Table 5](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/5990ae3a-df95-4864-8a24-890995d5e5de/figures/Table_5.png)
*Table 5 (ablation): Table 5: Test on non-symmetrized data on GPUs. The total time (ms) and total iteration counts #iter are reported on the non-symmetrized matrices derived from the original SPD systems. Results indicate our approach is robust on general non-symmetric cases.*



消融实验（Table 5-6）确认两大设计选择的必要性。Table 5 的 OOD 测试显示：Heat-Density（非对称化数据）上相对提升 80%（OOD）/ 113%（ID），Heat-Large（更大分辨率）62%/73%，Hyperelasticity（不同物理域）72%/68%，Synthetic-Large 39%/75%，证明 GNN-SPAI 对问题规模和物理参数的泛化能力。Table 6 的损失函数对比是关键：在 heat 简单问题上 $L_{\text{SAI}}$ 与 $L_2$ 相近，但在 Synthetic 困难问题上 $L_{\text{SAI}}$ 的 1122.0 次迭代 vs $L_2$ 的 2109.8 次（差距 +88.0%）、$L_{\text{CS}}$ 的 2185.7 次（差距 +94.8%），确证尺度不变设计对复杂问题的决定性作用。

公平性检验：作者坦承方法局限——（1）评估限于 PDE 导出和合成数据，未覆盖更广泛科学计算领域；（2）小矩阵 CPU 场景 Diagonal 仍占优，说明优势集中在 GPU 大规模问题；（3）需预训练 GNN，单系统求解场景不经济。此外，未与 FCG-NO [8]、GNN preconditioners [21] 等最新 neural solver 直接对比，也未测试 AMG 等强 baselines。

## 方法谱系与知识库定位

**方法家族**：Learning-based preconditioner → GNN-based preconditioner generation

**直接父方法**：Li et al. [6][9] "Learning from linear algebra: A graph neural network approach to preconditioner design for conjugate gradient solvers"（同一作者团队）。GNN-SPAI 继承其"将矩阵视为图、用 GNN 生成预处理器"的核心范式，但完成四项关键替换：预处理器结构（不完全分解 → 显式 SPAI）、推理策略（三角求解 → SpMV）、训练目标（L2/余弦损失 → 尺度不变 $L_{\text{SAI}}$）、架构输出（三角因子解码器 → 稀疏逆条目解码器）。

**直接 baselines 差异**：
- **Li et al. [4] learning-based IC**：同样学习预处理器，但保留 IC 分解结构，需三角求解；GNN-SPAI 彻底消除该瓶颈
- **Häusner et al. [5] Neural IF**：最新 neural incomplete factorization，迭代收敛快但构造/应用开销极高；GNN-SPAI 以略多迭代换取 GPU 极致并行效率
- **传统 SPAI (AINV) [14]**：显式逆矩阵思想的古典来源，但依赖固定稀疏模式和昂贵构造算法；GNN-SPAI 用学习替代解析构造

**后续方向**：（1）扩展至非对称/不定系统（GMRES、BiCGSTAB）；（2）动态稀疏模式学习（当前固定为 $A$ 的非零模式）；（3）与神经网络算子（Neural Operator）联合端到端训练，替代当前"预处理器 + 传统 CG"的两阶段范式。

**知识库标签**：modality: graph | paradigm: supervised learning for numerical linear algebra | scenario: GPU-accelerated PDE solving, large-scale sparse system | mechanism: message-passing GNN + explicit sparse matrix generation | constraint: symmetric positive definite systems, GPU-preferred over CPU

## 引用网络

### 直接 baseline（本文基于）

- Neural operators meet conjugate gradients: The FCG-NO method for efficient PDE solving _(ICML 2024, 实验对比, 未深度分析)_: Neural operator + CG method, likely compared in experiments as alternative ML-ba
- A Neural-Preconditioned Poisson Solver for Mixed Dirichlet and Neumann Boundary Conditions _(ICML 2024, 实验对比, 未深度分析)_: Neural preconditioner for specific PDE problem, likely compared in experiments
- Graph Neural Preconditioners for Iterative Solutions of Sparse Linear Systems _(ICLR 2025, 实验对比, 未深度分析)_: Directly comparable GNN preconditioner approach, very likely baseline in experim

