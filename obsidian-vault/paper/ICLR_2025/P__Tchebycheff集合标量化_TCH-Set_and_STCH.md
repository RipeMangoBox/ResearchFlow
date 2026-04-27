---
title: 'Few for Many: Tchebycheff Set Scalarization for Many-Objective Optimization'
type: paper
paper_level: C
venue: ICLR
year: 2025
paper_link: null
aliases:
- Tchebycheff集合标量化求解超多目标优化
- TCH-Set and STCH
- TCH-Set and STCH-Set
acceptance: Poster
cited_by: 17
code_url: https://github.com/IlkhamFY/qstch-set
method: TCH-Set and STCH-Set
---

# Few for Many: Tchebycheff Set Scalarization for Many-Objective Optimization

[Code](https://github.com/IlkhamFY/qstch-set)

**Topics**: Multi-Objective Optimization | **Method**: [[M__TCH-Set_and_STCH-Set]] | **Datasets**: Convex many-objective optimization, Mixed linear regression

| 中文题名 | Tchebycheff集合标量化求解超多目标优化 |
| 英文题名 | Few for Many: Tchebycheff Set Scalarization for Many-Objective Optimization |
| 会议/期刊 | ICLR 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2405.19650) · [Code](https://github.com/IlkhamFY/qstch-set) · [Project](待补充) |
| 主要任务 | 超多目标优化（many-objective optimization），即目标数m远大于解集大小K的优化问题 |
| 主要 baseline | Linear Scalarization (LS)、Tchebycheff Scalarization (TCH)、Smooth Tchebycheff (STCH)、MosT、SoM |

> [!abstract] 因为「经典标量化方法需要大量偏好采样才能覆盖高维目标空间，且单解优化无法同时满足众多冲突目标」，作者在「Tchebycheff标量化」基础上改了「将单解优化扩展为集合级max-min联合优化，并引入自适应平滑调度」，在「凸128目标优化基准」上取得「worst objective value从STCH的4.41降至STCH-Set的0.608，相对提升86.2%」

- **关键性能1**：凸128目标优化，STCH-Set worst值0.608，相比STCH（4.41）降低86.2%，相比SoM（1.86）降低67.3%
- **关键性能2**：混合线性回归（σ=0.1），STCH-Set worst值2.10，优于SoM的2.50
- **关键性能3**：TCH-Set与STCH-Set时间复杂度与简单线性标量化相当（Table 10验证）

## 背景与动机

超多目标优化（many-objective optimization）面临一个根本性困境：当目标数量m很大（如128个）而决策者只需要少量代表性解（如K=3或5个）时，经典方法要么需要采样海量偏好向量，要么产生的单解在多个目标上表现极差。例如，在神经网络多任务学习中，一个模型可能需要同时优化数十个损失项，但我们最终只想部署少数几个模型变体。

现有方法如何处理这一问题？**Linear Scalarization (LS)** 通过随机采样偏好向量λ，将多目标加权求和为单目标，但大量实验表明其在非凸Pareto前沿上无法找到边界解。**Tchebycheff Scalarization (TCH)** 改用max_i λ_i f_i(x)形式，虽能覆盖边界，但仍需为每个解单独采样偏好，且在高维目标空间中偏好密度呈指数增长。**Smooth Tchebycheff (STCH)** 引入固定平滑参数μ使目标可微，但依然是单解优化框架。**MosT** 和 **SoM** 尝试多解优化，前者基于最优传输理论，后者基于最小值之和，但均未直接针对"最不利目标"进行显式优化。

这些方法的核心短板在于：**优化粒度与决策粒度错配**。经典方法在"单解-单偏好"层面标量化，导致m个目标需要O(m)个偏好才能覆盖；而实际决策往往只需要K<<m个解。更关键的是，现有方法的标量化准则（加权和或加权max）无法保证"每个目标至少被一个解较好满足"这一最劣情况性能。

本文提出TCH-Set与STCH-Set，首次将Tchebycheff准则从单解扩展到解集层面，直接优化"所有目标中被满足最差者的上限"，从而用极少量解（K=3~5）覆盖大量目标（m=128）。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/2e0c935e-a118-4c13-a7d2-06956443badb/figures/Figure_1.png)
*Figure 1 (motivation): Figure 1: Large Set vs. Small Set for Multi-Objective Optimization. (a)-(c): Large Set. Classic scalarization methods find diversified solutions for multi-objective optimization problems. The required number of solutions for a good approximation would be more than exponential w.r.t. the number of objectives [Hillermeier, 2001; Lin et al., 2022], making them impractical for many objectives. (d): TchBycheff Set (TCH Set) scalarization method. We propose TCH Set scalarization which can model arbitrary many optimization objectives with very few scalarization functions.*



## 核心创新

核心洞察：**集合层面的max-min标量化**——因为Tchebycheff准则的max结构天然适合度量"最不利目标"，而min操作在集合内部实现了"自动分配"（每个目标由最适合它的解负责），从而使"少量解覆盖大量目标"成为可能，无需显式偏好采样。

| 维度 | Baseline (TCH/STCH) | 本文 (TCH-Set/STCH-Set) |
|:---|:---|:---|
| 优化变量 | 单解 x | 解集 X_K = {x_1,...,x_K} |
| 标量化准则 | max_i λ_i f_i(x)，需采样偏好λ | max_i min_{x∈X_K} f_i(x)，无偏好 |
| 目标函数性质 | 关于x非光滑（TCH）或固定μ平滑（STCH） | 集合级非光滑，通过softmin+自适应μ平滑 |
| 覆盖机制 | 单解妥协所有目标 | 集合内部分工，各解专攻不同目标 |
| 解的语义 | 一个偏好对应一个解 | K个解自动覆盖全部m个目标 |

与baseline的本质差异在于：TCH/STCH回答的是"给定偏好，最优解是什么"；TCH-Set/STCH-Set回答的是"给定解集预算K，如何联合优化使得最不利目标尽可能好"。后者消除了偏好采样的维度灾难，将问题复杂度从O(m)个偏好降至O(1)个解集优化。

## 整体框架


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/2e0c935e-a118-4c13-a7d2-06956443badb/figures/Figure_2.png)
*Figure 2 (motivation): Figure 2: From Solutions to Address More Optimization Objectives. (a)-(d): TCH Set solutions (blue). The light blue concentric bands represent optimization objectives.*



整体流程遵循"初始化→标量化评估→梯度更新"的迭代优化范式，核心创新在于集合级目标函数的设计：

**输入**：多目标优化问题（m个目标函数{f_i}_{i=1}^m），指定解集大小K

**模块1：集合初始化（Set initialization）**
- 输入：m, K
- 输出：初始解集 X_K^{(0)} = {x_1^{(0)}, ..., x_K^{(0)}}
- 作用：随机或启发式生成K个初始解，无特殊要求

**模块2：集合标量化目标评估（TCH-Set/STCH-Set objective evaluation）**
- 输入：当前解集 X_K^{(t)}, 目标函数{f_i}
- 输出：标量损失值 L(X_K^{(t)})
- 作用：计算"所有目标中最差被满足程度"——TCH-Set用硬min，STCH-Set用softmin平滑

**模块3：集合更新（Set update via gradient descent）**
- 输入：当前解集 X_K^{(t)}, 损失梯度 ∇_{X_K} L
- 输出：更新后解集 X_K^{(t+1)}
- 作用：对K个解同时执行梯度下降，STCH-Set下梯度可自动分配各解的优化方向

**关键机制**：自适应平滑调度 μ(t) = exp(−3×10^{−3} t) 控制STCH-Set的优化过程——早期大μ促进探索（解集分散覆盖不同目标区域），后期小μ精确收敛（逼近硬Tchebycheff准则）。

简化流程图：
```
Problem (m objectives, budget K)
    ↓
Initialize X_K = {x_1, ..., x_K}
    ↓
For t = 1, 2, ...:
    ├─ Evaluate all f_i(x_k) for i∈[m], k∈[K]
    ├─ Compute L(X_K) = max_i softmin^{(μ(t))}_{x∈X_K} f_i(x)
    ├─ Compute gradient ∇_{X_K} L
    └─ Update X_K ← X_K − η·∇_{X_K} L
    ↓
Return optimized set X_K*
```

## 核心模块与公式推导

### 模块1: TCH-Set 集合标量化目标（对应框架图 模块2）

**直觉**：将经典Tchebycheff从"单解对单偏好的最坏情况"扩展为"解集对所有目标的最坏情况"，让集合内部的min自动实现目标到解的分配。

**Baseline 公式** (Tchebycheff Scalarization):
$$\min_{x} \max_{1 \leq i \leq m} \lambda_i f_i(x)$$
符号: $x$ = 单解, $\lambda_i$ = 第i个目标的偏好权重, $f_i(x)$ = 第i个目标函数值。需要为每个解采样不同的$\lambda$向量。

**变化点**：baseline需要O(m)个偏好向量才能覆盖目标空间，且单解无法同时优化所有目标。本文去掉权重$\lambda_i$（实现均匀覆盖），将单解$x$替换为解集$X_K$，并将内部$f_i(x)$替换为集合最优$\min_{x\in X_K} f_i(x)$——即每个目标由解集中表现最好的解负责。

**本文公式（推导）**:
$$\text{Step 1}: \min_{x} \max_i f_i(x) \rightarrow \min_{X_K} \max_i \min_{x \in X_K} f_i(x) \quad \text{去掉λ_i，引入集合min实现自动分配}$$
$$\text{最终}: \min_{X_K} \max_{1 \leq i \leq m} \min_{x \in X_K} f_i(x)$$

**对应消融**：Table 1显示TCH-Set（硬min）在凸128目标上worst值1.02，已优于所有baseline；STCH-Set进一步降至0.608。

---

### 模块2: STCH-Set 平滑集合标量化（对应框架图 模块2+训练调度）

**直觉**：TCH-Set中的硬min操作关于解集不可微，无法直接梯度优化。用softmin（log-sum-exp）平滑替代，并设计自适应温度调度平衡探索与收敛。

**Baseline 公式** (Smooth Tchebycheff, 固定μ):
$$\min_{x} \text{softmax}_i^{(\mu)} (\lambda_i f_i(x)) = \min_x \mu \log \sum_{i=1}^m \exp\left(\frac{\lambda_i f_i(x)}{\mu}\right)$$
符号: $\mu$ = 固定平滑参数, softmax用于平滑max操作。但仍为单解优化且需偏好$\lambda$。

**变化点**：(1) 将softmax（平滑max）与softmin（平滑min）结合，实现集合级双重平滑；(2) 固定μ导致优化全程同等平滑，早期探索不足、后期收敛不精。

**本文公式（推导）**:
$$\text{Step 1}: \min_{X_K} \max_i \min_{x \in X_K} f_i(x) \rightarrow \min_{X_K} \max_i \left(-\mu \log \sum_{x \in X_K} \exp\left(-\frac{f_i(x)}{\mu}\right)\right) \quad \text{softmin替代硬min}$$
$$\text{Step 2}: \mu = \text{const} \rightarrow \mu(t) = \exp(-3 \times 10^{-3} t) \quad \text{自适应衰减，早期大μ探索、后期小μ精修}$$
$$\text{最终}: \min_{X_K} \max_{1 \leq i \leq m} \left(-\mu(t) \log \sum_{x \in X_K} \exp\left(-\frac{f_i(x)}{\mu(t)}\right)\right)$$

符号: $X_K$ = 含K个解的集合, $\mu(t)$ = 时变平滑参数, softmin$^{(\mu)}$ = $-\mu \log \sum \exp(-f/\mu)$。当$\mu \to 0$时softmin收敛到硬min；当$\mu \to \infty$时趋近于算术平均。

**对应消融**：Table 8（Appendix）对比固定μ∈{10,5,1,0.5,0.1}与自适应μ(t)，证明自适应调度在worst和average指标上均最优。固定大μ（如10）过度平滑导致收敛差，固定小μ（如0.1）早期探索不足易陷局部最优。

---

### 模块3: 自适应平滑调度（对应框架图 训练控制）

**直觉**：优化初期需要"模糊"的softmin让梯度在各解间流动，促进解集分散覆盖不同目标；后期需要"锐利"的softmin精确逼近硬min，获得精确的集合分工。

**Baseline 公式**: $\mu = \text{const}$

**本文公式**:
$$\mu(t) = \exp(-3 \times 10^{-3} t)$$

该指数衰减在t=0时μ≈1（适度平滑），t=1000时μ≈0.05（接近硬min），t=2000时μ≈0.0025（几乎精确）。无需手动调参，单参数控制全程优化特性。

## 实验与分析


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/2e0c935e-a118-4c13-a7d2-06956443badb/figures/Table_1.png)
*Table 1 (quantitative): Table 1: The multi-objective optimization problem. Full table can be found in Appendix A.*



本文在两类基准上验证方法：合成凸多目标优化（m=128，Table 1）与混合线性回归（σ=0.1，Table 2）。核心评估指标为**worst objective value**（max_i min_{x∈X_K} f_i(x)）——直接反映"最不利目标被满足的程度"，以及average objective value反映整体表现。

**凸128目标优化（Table 1）**：STCH-Set取得worst值0.608，相比直接baseline STCH（4.41）相对降低86.2%，相比SoM（1.86）降低67.3%。这一差距具有实际意义：STCH需要大量偏好采样仍无法避免某些目标被严重忽视，而STCH-Set仅用K=3个解即实现全面覆盖。值得注意的是，SoM在average指标上略优于STCH-Set（0.202 vs 0.212，Table 1中K=3时SoM标记为"-"），说明SoM的sum-of-minimum目标更侧重平均性能而非最坏情况保障。MosT（2.12）表现介于SoM与经典方法之间，但未参与混合回归实验。

**混合线性回归（Table 2）**：在非凸设置下，STCH-Set worst值2.10，优于SoM的2.50，验证方法超越凸假设的适用性。LS、TCH、STCH在此任务上worst值分别高达38.4、24.5、24.3，凸显单解标量化在目标数大时的失效。


![Table 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/2e0c935e-a118-4c13-a7d2-06956443badb/figures/Table_2.png)
*Table 2 (quantitative): Table 2: The results on neural linear regression with entry loss = 0.1. Full table in Appendix C.2.*



**消融实验**：Table 8（Appendix）系统比较固定μ与自适应μ(t)。固定μ=10时过度平滑，worst值显著劣化；固定μ=0.1时早期探索不足，解集易聚集于局部区域。自适应调度μ(t)=exp(−3×10^{−3}t)在全部设置下最优，验证了"先探索后精修"策略的必要性。此外，理论分析（Table 4）证明TCH-Set和STCH-Set在μ→0时具有（弱）Pareto最优性保证。

**公平性审视**：(1) 未包含NSGA-III、MOEA/D等进化多目标算法，这些算法在超多目标领域有长期积累；(2) 缺乏真实世界基准（如神经网络架构搜索、推荐系统多目标优化），当前结果限于合成问题；(3) SoM在average指标上偶有优势，说明若决策目标非最坏情况保障，STCH-Set未必最优；(4) 运行时间比较（Table 10）显示STCH-Set与LS同阶，但未报告具体GPU类型与绝对时间。整体证据强度中等偏上（0.75），核心结论在worst-case指标上稳健。

## 方法谱系与知识库定位

**方法家族**：多目标标量化（multi-objective scalarization）→ Tchebycheff标量化 → 平滑Tchebycheff标量化 → **集合标量化（set scalarization）**

**直接父方法**：Smooth Tchebycheff Scalarization (STCH, Lin et al., 2024)。本文继承其softmax平滑技术，但将应用层级从"单解-单偏好"跃迁至"解集-全目标"。

**改动插槽**：
- **objective（目标函数）**：单解加权max → 集合无权重max-min
- **training_recipe（训练配方）**：固定μ → 自适应指数衰减μ(t)
- **inference_strategy（推断策略）**：偏好采样生成多解 → 单轮优化输出K解覆盖全目标

**直接baseline差异**：
- **vs LS/TCH/STCH**：从"多偏好-多解"（需O(m)次独立优化）到"无偏好-单轮集合优化"（O(1)次）
- **vs MosT**：MosT用最优传输理论分配解到目标区域，本文用max-min的博弈结构隐式实现分配
- **vs SoM**：SoM优化sum of minima（∑_i min_x f_i(x)），本文优化max of minima（max_i min_x f_i(x)），前者侧重平均、后者保障最坏情况

**后续方向**：(1) 将集合标量化扩展至连续Pareto前沿近似（当前仅输出离散K解）；(2) 结合超网络（hypernetwork）实现解集的参数化共享，降低大K时的存储开销；(3) 引入目标相关性建模，对高度相关目标降维以进一步提升效率。

**标签**：modality=优化算法, paradigm=标量化/集合优化, scenario=超多目标决策, mechanism=max-min博弈+自适应平滑, constraint=解集大小预算K

