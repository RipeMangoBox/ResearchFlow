---
title: Structured Sparse Transition Matrices to Enable State Tracking in State-Space Models
type: paper
paper_level: C
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- PD-SSM：结构化稀疏转移矩阵赋能状态追踪
- PD-SSM
acceptance: Spotlight
cited_by: 6
method: PD-SSM
modalities:
- Text
- time-series
paradigm: supervised
---

# Structured Sparse Transition Matrices to Enable State Tracking in State-Space Models

**Topics**: [[T__Reasoning]], [[T__Time_Series_Forecasting]] | **Method**: [[M__PD-SSM]] | **Datasets**: FSA State Tracking, Runtime

> [!tip] 核心洞察
> PD-SSM, which parametrizes the transition matrix as the product of a column one-hot matrix (P) and a complex-valued diagonal matrix (D), achieves optimal FSA state tracking expressivity with computational cost comparable to diagonal SSMs.

| 中文题名 | PD-SSM：结构化稀疏转移矩阵赋能状态追踪 |
| 英文题名 | Structured Sparse Transition Matrices to Enable State Tracking in State-Space Models |
| 会议/期刊 | NeurIPS 2025 (Spotlight) |
| 链接 | [arXiv](https://arxiv.org/abs/2509.22284) · Code (未公开) |
| 主要任务 | FSA 状态追踪、时间序列分类 |
| 主要 baseline | Mamba/S6、SD-SSM (dense)、Transformer、DeltaNet、RWKV-7、DPLR-SLiCE |

> [!abstract] 因为「对角 SSM 表达力不足以模拟任意有限状态自动机，而稠密矩阵计算代价过高」，作者在「Mamba/S6 选择性对角 SSM」基础上改了「转移矩阵分解为列 one-hot 矩阵 P 与复对角矩阵 D 的乘积」，在「FSA 状态追踪基准（Modular Arithmetic 任务）」上取得「100.0% 准确率，超越次优方法 Gated DeltaProduct[-1,1] 21.6 个百分点」。

- **Modular Arithmetic 准确率**: PD-SSM 100.0% vs. Mamba 33.1% vs. Transformer 23.6%
- **运行效率**: 在 D=5632 时，PD-SSM 比稠密 SD-SSM 快 71×，比纯对角 SSM 慢 7×
- **理论保证**: 单层 PD-SSM 以状态维度 N 即可精确模拟任意 N 状态 FSA

## 背景与动机

状态空间模型（State-Space Models, SSMs）通过隐状态递归来处理长序列，其核心计算为 h_t = A(u_t)h_{t-1} + B(u_t)x_t。然而，转移矩阵 A(u_t) 的结构设计面临根本性张力：对角矩阵（如 Mamba/S6）虽可实现 O(N) 并行扫描，但表达力受限——无法有效追踪有限状态自动机（FSA）的离散状态转移；而稠密无结构矩阵（如 SD-SSM）虽具备完整表达力，却需要 O(N²) 的计算代价，难以扩展。

现有方法的处理方式各有局限：
- **Mamba/S6** 采用输入依赖的对角转移矩阵 A = diag(λ₁(u_t),...,λ_N(u_t))，通过硬件友好的逐元素运算实现线性复杂度，但对需要精确状态追踪的任务（如模运算、奇偶校验）表现不佳——例如 Cycle Navigation 任务上仅得 48.4% 准确率。
- **SD-SSM** 使用稠密实值矩阵，虽在理论上可表达任意动态，但矩阵乘法导致 O(N²) 开销，在 D=5632 时比对角结构慢两个数量级。
- **DPLR-SLiCE** 等对角加低秩折中方案试图平衡效率与表达力，但在 Modular Arithmetic 等硬任务上仍显著落后（68.3% vs. 理想 100%）。


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/deb28fea-ca93-4b96-84ff-166cd01679f2/figures/fig_001.png)
*Figure: The PD parametrization can be integrated into any selective SSM by adopting the shown*



这些方法的共同短板在于：**没有任何一种结构能在保持 O(N) 计算复杂度的同时，提供对 FSA 状态追踪的理论最优表达力**。对角矩阵的独立元素缩放无法编码状态间的置换关系，而低秩扰动不足以恢复完整的置换群结构。本文提出 PD-SSM，通过将转移矩阵分解为「列 one-hot 置换选择器」与「复对角缩放器」的乘积，首次同时满足这两个看似矛盾的要求。

## 核心创新

核心洞察：转移矩阵的「列 one-hot × 复对角」分解 A(u_t) = P(u_t)D(u_t) 能够在保持 O(N) 并行扫描复杂度的同时，通过 P 矩阵的置换选择与 D 矩阵的复数旋转缩放，生成足够的代数结构来模拟任意有限状态自动机的状态转移，因为列 one-hot 性质使得矩阵乘法退化为索引重排操作，而复数值对角元提供了实数域无法实现的相位自由度，从而使单层、单头、维度 N 的 PD-SSM 即可精确表达任意 N 状态 FSA。

| 维度 | Baseline (Mamba/S6) | 本文 (PD-SSM) |
|:---|:---|:---|
| 转移矩阵结构 | 实对角矩阵 diag(λ₁,...,λ_N) | 列 one-hot P(u_t) × 复对角 D(u_t) |
| 状态更新计算 | 逐元素缩放 O(N) | 先缩放后置换，仍 O(N)（索引操作） |
| FSA 表达力 | 需 N > 状态数或多层堆叠 | 单层 N 维即最优（理论下界） |
| 参数类型 | 实值 | 复值对角元（D 矩阵） |
| 并行扫描 | 标准 associative scan | 利用 P 结构的定制 associative scan |

## 整体框架


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/deb28fea-ca93-4b96-84ff-166cd01679f2/figures/fig_002.png)
*Figure: Additionally, as proven in the following section, the diagonal matrices provide a guarantee*



PD-SSM 单层的完整数据流如下：

1. **输入投影**（Input Projection）：接收输入 token x_t，生成三组输入依赖参数——P(u_t) 的索引、D(u_t) 的对角元、以及 B(u_t)。
2. **P-矩阵生成器**（P-matrix Generator）：根据 u_t 输出列 one-hot 矩阵 P(u_t) ∈ {0,1}^{N×N}，每列恰好一个 1，决定「哪个源状态分量写入哪个目标位置」。
3. **D-矩阵生成器**（D-matrix Generator）：根据 u_t 输出复对角矩阵 D(u_t) ∈ ℂ^{N×N}，提供元素级的复数缩放（幅度+相位）。
4. **结构化状态更新**（Structured State Update）：计算 h_t = P(u_t)D(u_t)h_{t-1} + B(u_t)x_t。关键优化：D(u_t)h_{t-1} 为逐元素复数乘法；P(u_t) 作用于此结果等价于索引重排（gather 操作），无需通用矩阵乘法。
5. **输出投影**（Output Projection）：线性读出 y_t = C h_t，其中 C 可为实值或复值。

整个前向计算通过 associative scan 并行化：定义结合律算子 ⊕ 使得 (h_t, 累积项) 可在 O(log L) 并行深度内完成，其中 L 为序列长度。P-D 结构的关键性质在于 P(u_t) 的列 one-hot 特性保证了算子 ⊕ 的结合性，这是并行前缀扫描可行的代数前提。

```
x_t ──→ [Input Proj] ──→ u_t ──┬──→ [P Gen] ──→ P(u_t) ──┐
                                ├──→ [D Gen] ──→ D(u_t) ──┼──→ [Structured Update] ──→ h_t ──→ [Output Proj] ──→ y_t
                                └──→ [B Gen] ──→ B(u_t) ──┘         ↑___________________________________|
                                                                     (recurrent connection via associative scan)
```

## 核心模块与公式推导

### 模块 1: P-D 转移矩阵分解（对应框架图步骤 2-3）

**直觉**: 将一般矩阵的「全连接」约束拆解为「先选位置、再改数值」的两个简单操作的复合，既保留表达力又利用稀疏结构加速。

**Baseline 公式** (Mamba/S6 对角结构):
$$A(u_t) = \text{diag}(\lambda_1(u_t), ..., \lambda_N(u_t)) \in \mathbb{R}^{N \times N}$$
符号: λ_i(u_t) = 输入依赖的实数衰减/增长系数；N = 状态维度。

**变化点**: 对角矩阵只能独立缩放各状态分量，无法编码状态间的耦合与置换。对于需要「状态 A 转移到状态 B」的 FSA 模拟，对角结构必须借助增大维度或多层堆叠，效率低下。

**本文公式（推导）**:
$$\text{Step 1}: A(u_t) = P(u_t) D(u_t) \quad \text{将转移矩阵约束为两个简单矩阵的乘积}$$
$$\text{Step 2}: P(u_t) \in \{0,1\}^{N \times N}, \; P(u_t)_{i,j} = \mathbf{1}[j = \pi_{u_t}(i)] \quad \text{列 one-hot：每列单 1，实现输入依赖的置换选择}$$
$$\text{Step 3}: D(u_t) = \text{diag}(d_1(u_t), ..., d_N(u_t)) \in \mathbb{C}^{N \times N} \quad \text{复对角元提供幅度与相位调制}$$
$$\text{最终}: A(u_t)h_{t-1} = P(u_t)(D(u_t)h_{t-1}) = \text{gather}(D(u_t) \odot h_{t-1}, \; \pi_{u_t}^{-1})$$

**对应消融**: Table 2 显示将 P 固定为恒等矩阵（即 C Diagonal 变体）后，Cycle Navigation 从 99.8% 降至 90.4%，Modular Arithmetic 从 100.0% 降至 59.9%，Parity 从 100.0% 降至 61.8%，平均下降 40.1 个百分点。

---

### 模块 2: 结构化状态更新与并行扫描（对应框架图步骤 4）

**直觉**: 利用 P 矩阵的列 one-hot 性质，将矩阵-向量乘法降级为索引操作，从而维持 O(N) 复杂度；同时证明 P-D 复合满足结合律，解锁并行前缀扫描。

**Baseline 公式** (稠密 SD-SSM):
$$h_t = A(u_t) h_{t-1} + B(u_t) x_t, \quad A(u_t) \in \mathbb{R}^{N \times N} \text{ (dense)}$$
计算代价: O(N²) 每步，或 O(N² L) 序列总计（L = 序列长度）。

**变化点**: 稠密矩阵虽表达力强，但 O(N²) 开销 prohibitive；对角矩阵虽 O(N)，但丧失状态间耦合能力。需要一种结构，其矩阵乘法代价与对角相当，却能编码置换操作。

**本文公式（推导）**:
$$\text{Step 1}: h_t = P(u_t) D(u_t) h_{t-1} + B(u_t) x_t \quad \text{代入 P-D 分解}$$
$$\text{Step 2}: v_t = D(u_t) h_{t-1} = (d_1(u_t) h_{t-1,1}, \, ..., \, d_N(u_t) h_{t-1,N})^T \quad \text{复数逐元素乘法，O(N)}$$
$$\text{Step 3}: P(u_t) v_t = v_t[\pi_{u_t}^{-1}(\cdot)] \quad \text{索引重排（gather），O(N)，无需矩阵乘法}$$
$$\text{Step 4}: \text{定义结合算子 } (h, c) \oplus_{u} (h', c') = (P(u)D(u)h + c \cdot h', \; c \cdot c') \text{ 其中 } c \text{ 为累积标量}$$
$$\text{验证}: ((h_1, c_1) \oplus_{u_2} (h_2, c_2)) \oplus_{u_3} (h_3, c_3) = (h_1, c_1) \oplus_{u_2} ((h_2, c_2) \oplus_{u_3} (h_3, c_3)) \quad \text{P-D 结构保证结合性}$$
$$\text{最终}: \text{Blelloch 并行前缀扫描：} O(N) \text{ 工作复杂度，} O(\log L) \text{ 并行深度}$$

**对应消融**: Figure 4 显示在 D=5632、L=64 时，PD-SSM 比稠密 SD-SSM 快 71×；作为代价，比纯对角 SSM 慢 7×，这 7× 开销主要来自 P 矩阵的生成与索引调度。

---

### 模块 3: FSA 模拟理论保证（对应框架核心性质）

**直觉**: 证明 P-D 结构的表达力上界——不仅够快，而且够强，能达到理论最优。

**Baseline 公式** (一般 SSM 表达力):
$$\text{Diagonal SSM: 需要状态维度 } N > |Q| \text{（FSA 状态数）或深度 } > 1 \text{ 才能模拟 } |Q| \text{-状态 FSA}$$
$$\text{Dense SSM: 单层 } N = |Q| \text{ 足够，但 } O(N^2) \text{ 计算}$$

**变化点**: 对角结构因缺乏置换能力而需超参数补偿；稠密结构虽最优但无效率。是否存在「最优表达力 + 线性复杂度」的甜蜜点？

**本文公式（推导）**:
$$\text{Step 1}: \text{设 } M = (Q, \Sigma, \delta, q_0, F) \text{ 为任意 } N\text{-状态 FSA，} Q = \{1,...,N\}$$
$$\text{Step 2}: \forall a \in \Sigma, \text{ 定义置换 } \pi_a: Q \to Q \text{ 使得 } \pi_a(q) = \delta(q, a) \text{（确定转移扩展为置换，非单射时引入哑状态）}$$
$$\text{Step 3}: \text{构造 } P(u_t) \text{ s.t. } P(u_t)_{i,j} = \mathbf{1}[j = \pi_{u_t}(i)], \; D(u_t) = I \text{（单位复对角，纯置换情形）}$$
$$\text{Step 4}: \text{对非置换转移，利用复对角元编码分支：} d_j(u_t) = e^{i\theta_j} \cdot \mathbf{1}[\text{条件}] \text{，配合线性读出 } W \text{ 解码}$$
$$\text{最终定理}: \exists W \in \mathbb{R}^{N \times N}, \text{ s.t. one PD-SSM layer with } h_t = P(u_t)D(u_t)h_{t-1} + Bx_t, \; y_t = Wh_t$$
$$\text{exactly emulates any } N\text{-state FSA. 状态数 } N \text{ 和层数 } 1 \text{ 均为理论下界。}$$

## 实验与分析



本文在 FSA 状态追踪基准上评估 PD-SSM，包含四个核心任务：Cycle Navigation（循环导航）、Even Pairs（偶对）、Modular Arithmetic（模运算）、Parity（奇偶校验）。
![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/deb28fea-ca93-4b96-84ff-166cd01679f2/figures/fig_003.png)
*Figure: Property 2 (Computational Efficiency of Matrix Multiplication in HN×N). Let A, B ∈HN×N.*

 展示了不同模型在这些任务上的准确率对比。在 Modular Arithmetic 任务上，PD-SSM 达到 100.0% 准确率，相比次优方法 Gated DeltaProduct[-1,1] 的 78.4% 提升 21.6 个百分点；相比 Mamba 的 33.1% 提升 66.9 个百分点；相比 Transformer 的 23.6% 提升 76.4 个百分点。这一差距表明，复数域的相位操作与置换选择机制对模运算这类需要精确状态循环的任务至关重要。

在 Cycle Navigation 任务上，PD-SSM 以 99.8% 接近完美，与 BD-SLiCE（99.8%）持平，但显著优于所有其他可并行化基线：Mamba 48.4%、DeltaNet 49.8%、RWKV-7 37.8%、DPLR-SLiCE 81.1%。Parity 任务上 PD-SSM 同样达到 100.0%，与 LSTM、sLSTM、D-SLiCE 并列，但远超并行化竞争对手（Transformer 52.2%、Mamba 54.2%）。Even Pairs 任务相对简单，多个模型均达 100.0%。



消融实验（Table 2 中 C Diagonal 列）量化了 P 矩阵的核心贡献：移除置换结构后，Modular Arithmetic 从 100.0% 暴跌至 59.9%（-40.1%），Parity 从 100.0% 降至 61.8%（-38.2%），Cycle Navigation 从 99.8% 降至 90.4%（-9.4%）。这表明 P 矩阵的置换能力在「硬」状态追踪任务上不可或缺，而 D 矩阵的复数值单独无法补偿结构缺陷。


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/deb28fea-ca93-4b96-84ff-166cd01679f2/figures/fig_004.png)
*Figure: Runtimes of single-layer SSMs with*



运行效率方面，Figure 4 显示在 D=5632、序列长度 L=64 时，PD-SSM 相对稠密 SD-SSM 实现 71× 加速，但相对纯对角 SSM 仍有 7× 差距。这 7× 开销源于 P 矩阵的生成与索引调度，作者指出这是当前实现的工程瓶颈而非理论下界。

公平性检验：实验固定所有模型的状态维度为 128，可能不利于需要更大状态空间的方法；仅测试单组超参数配置（继承自 Walker et al. 2025）；缺乏大规模语言建模任务验证；未与 Gated State Spaces (GSS)、H3、Hyena、RetNet 等更强基线对比。此外，复数运算在标准硬件上的部署开销尚未充分表征。

## 方法谱系与知识库定位

PD-SSM 属于 **SSM 结构化矩阵谱系**，直接父方法为 **S6/Mamba（选择性对角 SSM）**。该谱系的演进脉络为：稠密无结构矩阵（SD-SSM, O(N²)）→ 对角结构化矩阵（S6/Mamba, O(N)）→ 结构化稀疏分解（PD-SSM, O(N) 且最优 FSA 表达力）。

**改变的插槽**:
- **architecture**: 对角/稠密转移矩阵 → P-D 乘积结构（列 one-hot × 复对角）
- **inference_strategy**: 逐元素运算或稠密 matmul → 索引重排 + associative parallel scan
- **training_recipe**: 实值参数 → 复值对角元（D 矩阵）

**直接基线与差异**:
- **Mamba/S6**: 同为选择性 SSM，但 PD-SSM 以 P-D 分解替换对角矩阵，表达力从「独立缩放」扩展至「置换+缩放」
- **SD-SSM (dense)**: 同为完整表达力，但 PD-SSM 以 O(N) 替代 O(N²)，实现 71× 加速
- **DPLR-SLiCE**: 同为结构化近似，但 PD-SSM 的 P-D 分解在理论上保证最优 FSA 模拟，而非经验性低秩近似
- **DeltaNet/RWKV-7**: 同为线性复杂度循环模型，但 PD-SSM 通过代数结构（置换群）而非门控机制实现状态追踪

**后续方向**:
1. **硬件感知优化**: 降低 P 矩阵生成的 7× 开销，探索专用 kernel 或稀疏索引加速
2. **大规模语言建模验证**: 将 PD-SSM 从合成 FSA 任务扩展至预训练尺度，检验复数运算的稳定性与梯度行为
3. **结构化矩阵扩展**: 探索 P-D 之外的群结构分解（如 Cayley 图嵌入 Figure 5 暗示的方向），或与其他结构化矩阵（Toeplitz、Hankel）的复合

**知识库标签**: modality={text, time-series} / paradigm={supervised, recurrent} / scenario={sequence modeling, reasoning, state tracking} / mechanism={structured sparse matrix, complex-valued neural network, associative scan, finite-state automaton emulation} / constraint={linear complexity, parallelizable, plug-in compatible}

