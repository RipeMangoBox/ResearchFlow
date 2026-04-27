---
title: Compositional Neural Network Verification via Assume-Guarantee Reasoning
type: paper
paper_level: A
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 组合式假设-保证神经网络验证框架CoVeNN
- CoVeNN
acceptance: Spotlight
code_url: https://neuralsat.roars.dev
method: CoVeNN
modalities:
- Image
paradigm: supervised
---

# Compositional Neural Network Verification via Assume-Guarantee Reasoning

[Code](https://neuralsat.roars.dev)

**Topics**: [[T__Adversarial_Robustness]] | **Method**: [[M__CoVeNN]] | **Datasets**: VNN-COMP benchmarks

> [!tip] 核心洞察
> CoVeNN, an assume-guarantee compositional framework, can verify nearly 7 times more problems than state-of-the-art verifiers by decomposing verification into smaller sub-problems with iteratively refined assumptions.

| 中文题名 | 组合式假设-保证神经网络验证框架CoVeNN |
| 英文题名 | Compositional Neural Network Verification via Assume-Guarantee Reasoning |
| 会议/期刊 | NeurIPS 2025 (Spotlight) |
| 链接 | [Project](https://neuralsat.roars.dev) |
| 主要任务 | Neural Network Verification, Adversarial Robustness Verification |
| 主要 baseline | αβ-CROWN, NeuralSAT, PyRAT, Marabou 2.0, nnenum |

> [!abstract] 因为「神经网络验证面临深度增加时指数级内存复杂度导致的可扩展性瓶颈」，作者在「NeuralSAT / αβ-CROWN 等单体验证器」基础上改了「组合式假设-保证分解与迭代精化策略」，在「VNN-COMP 7个网络140个性质」上取得「验证通过问题数提升近7倍」

- 在7个神经网络与140个性质规格上，CoVeNN 验证通过的问题数达到 state-of-the-art 验证器的近7倍
- 内存需求从单体验证的完整网络状态分裂，降至子问题的逐层假设-保证契约规模
- 框架可参数化包裹任意底层验证器（αβ-CROWN、NeuralSAT 等），实现 verifier-agnostic 的组合推理

## 背景与动机

神经网络验证旨在形式化证明网络在输入扰动下保持正确预测，是自动驾驶、医疗诊断等安全关键系统部署的前提。然而，随着网络深度增加，基于分支定界（branch-and-bound）或 SMT 的验证方法面临指数级内存膨胀——每层 ReLU 节点的状态分裂导致 GPU 内存迅速耗尽，使得深层网络的完整验证不可行。

现有方法如何应对这一挑战？αβ-CROWN [9] 采用多神经元松弛引导的分支定界，通过 tighter relaxation 减少搜索空间，但仍需一次性加载完整网络；NeuralSAT [17] 作为作者团队的高性能验证工具，优化了 bound propagation 与冲突学习，却同样受限于单体（monolithic）端到端验证架构；Compositional learning and verification of neural network controllers [23] 尝试了组合式思路，但未建立系统的假设-保证契约机制与迭代精化策略，分解粒度与精度难以平衡。

这些方法的共同瓶颈在于：**推理策略固定为单体验证**，必须同时维护全网络所有层的状态边界。当网络超过一定深度时，GPU 内存成为硬性天花板，无论 relaxation 多紧都无法突破。本文提出 CoVeNN，首次将经典假设-保证（assume-guarantee）推理系统性地引入神经网络验证，通过组合分解与迭代精化，将内存瓶颈转化为可逐个求解的子问题规模问题。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/fad17aa6-2093-4cd3-a746-96c3f0d3104f/figures/Figure_1.png)
*Figure 1 (comparison): Tight validation in Reachability bound test case. Compositional method with CAGNN versus monolithic method with CROWN [1].*



## 核心创新

核心洞察：神经网络验证可以视为 sequential components 的组合正确性问题，因为每层（或每段）子网络的输出边界可以作为下一子网络的输入假设，从而使迭代精化下的分治验证成为可能——无需一次性掌握全网络状态即可建立端到端保证。

| 维度 | Baseline (αβ-CROWN / NeuralSAT) | 本文 (CoVeNN) |
|:---|:---|:---|
| 推理策略 | Monolithic 端到端验证，全网络状态同时分裂 | 组合式假设-保证分解，子问题顺序求解 |
| 架构 | 单一验证器操作完整网络 | 参数化框架，包裹任意底层验证器并编排子问题生成 |
| 边界计算 | 直接 bound tightening 优化全网络 relaxation | 迭代精化假设，基于子问题验证结果动态调整精度-效率权衡 |
| 内存特征 | O(全网络层数 × 节点状态) | O(最大子问题规模)，与网络总深度解耦 |

## 整体框架


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/fad17aa6-2093-4cd3-a746-96c3f0d3104f/figures/Figure_2.png)
*Figure 2 (architecture): Example of a decomposed FC network with three hidden layers and ReLU figures of hidden ReLU.*



CoVeNN 的验证流程分为五个核心阶段，形成闭环的分解-验证-精化-组合管线：

1. **Network Decomposer（网络分解器）**：输入完整神经网络与待验证性质规格，基于层边界将网络切分为序列子网络 $N_1, N_2, ..., N_n$，输出各子网络的输入输出接口点。

2. **Assumption Generator（假设生成器）**：为每个子网络构造初始假设-保证契约 $(A_i^{(0)}, G_i^{(0)})$，其中 $A_i$ 为假设的输入约束超盒，$G_i$ 为待验证的输出性质。初始假设通常宽松以保证可解性。

3. **Parameterized Verifier（参数化验证器）**：对每个子问题调用底层验证器（如 αβ-CROWN 或 NeuralSAT），验证 $N_i$ 在假设 $A_i$ 下是否满足保证 $G_i$。该模块是框架的"插件点"，不绑定特定验证算法。

4. **Iterative Refinement Loop（迭代精化循环）**：若子问题验证失败（结果过松导致 false negative），基于验证反馈收紧假设 $A_i^{(k+1)} = \text{Refine}(A_i^{(k)}, \text{Result})$；若成功则尝试放宽以提升效率。循环直至所有子问题通过或达到资源上限。

5. **Composition Checker（组合检查器）**：验证相邻契约的传递性 $G_i \Rightarrow A_{i+1}$，若成立则通过组合规则推出端到端保证 $A_1 \Rightarrow G_n$。

```
[Network + Property] → [Decomposer] → [Sub-networks N_i]
    ↓
[Assumption Generator] → [(A_i, G_i) contracts]
    ↓
[Parameterized Verifier] → [Sub-problem results]
    ↓ (fail → refine)
[Refinement Loop] → [Tightened (A_i, G_i)]
    ↓ (all pass)
[Composition Checker] → [A_1 ⇒ G_n] → [Verified / Falsified / Unknown]
```

## 核心模块与公式推导

### 模块 1: 假设-保证契约生成（对应框架图左侧分解阶段）

**直觉**：将子网络视为黑盒组件，只需约定"输入满足什么"和"输出保证什么"，无需暴露内部权重细节。

**Baseline 形式**（单体验证）：
$$\text{Verify}(N, P_{\text{in}}, P_{\text{out}})$$
符号：$N$ 为完整网络，$P_{\text{in}}$ 为全局输入性质，$P_{\text{out}}$ 为全局输出性质。

**变化点**：单体验证要求同时处理 $N$ 的所有层，内存随深度指数增长；本文将 $N$ 分解为 $\{N_i\}_{i=1}^n$，每段仅需维护局部边界。

**本文公式**：
$$\mathcal{C}_i = (A_i, G_i)$$
其中 $A_i \subseteq \mathbb{R}^{d_i^{\text{in}}}$ 为子网络 $N_i$ 的假设输入超盒，$G_i \subseteq \mathbb{R}^{d_i^{\text{out}}}$ 为输出保证集合。初始假设取宽松边界：
$$A_i^{(0)} = \text{Project}(P_{\text{in}}, N_{1..i-1}) \cup \text{Widen}(\epsilon)$$

### 模块 2: 迭代精化更新（对应框架图中央循环）

**直觉**：验证失败往往源于假设过松导致边界传播发散，需像 CEGAR 一样用反例引导精化。

**Baseline 形式**：αβ-CROWN 等工具的直接 bound tightening：
$$\text{Bounds}^{(t+1)} = \text{Tighten}(\text{Bounds}^{(t)}, \text{Relaxation})$$
全网络同步更新，无分层假设概念。

**变化点**：baseline 的 tightening 作用于全网络所有层，计算冗余；本文仅在失败子问题的边界上精化，且精化方向由验证结果显式指导。

**本文公式（推导）**：
$$\text{Step 1:} \quad r_i^{(k)} = \text{Verify}(N_i, A_i^{(k)}, G_i^{(k)})$$
加入验证结果反馈项以定位失败源

$$\text{Step 2:} \quad A_{i+1}^{(k+1)} = \text{Refine}(A_{i+1}^{(k)}, r_i^{(k)})$$
重归一化以保证假设序列的单调收敛：若 $r_i^{(k)} = \text{VERIFIED}$ 则尝试 $\text{Widen}$，若 $r_i^{(k)} = \text{UNKNOWN}$ 则执行 $\text{Intersect}(A_{i+1}^{(k)}, \text{CounterexampleBound})$

$$\text{最终:} \quad A_i^{(k+1)} = \begin{cases} A_i^{(k)} \cap \phi(r_{i-1}^{(k)}) & \text{if } r_{i-1}^{(k)} = \text{UNKNOWN} \\ A_i^{(k)} \cup \psi(G_{i-1}^{(k)}) & \text{if } r_{i-1}^{(k)} = \text{VERIFIED} \end{cases}$$

### 模块 3: 端到端组合规则（对应框架图右侧组合阶段）

**直觉**：局部正确性通过逻辑传递链拼接为全局正确性，这是 assume-guarantee 推理的核心数学基础。

**Baseline 形式**：无显式组合，单体验证直接输出 $P_{\text{in}} \Rightarrow P_{\text{out}}$。

**变化点**：本文需证明子问题结果的拼接确实蕴含全局性质，避免组合时的 soundness gap。

**本文公式**：
$$\text{bigwedge}_{i=1}^{n} (A_i \Rightarrow G_i) \wedge \text{bigwedge}_{i=1}^{n-1} (G_i \Rightarrow A_{i+1}) \Rightarrow (A_1 \Rightarrow G_n)$$

符号：$A_1$ 为网络原始输入性质（通常 $A_1 = P_{\text{in}}$），$G_n$ 为最终输出保证（通常 $G_n = P_{\text{out}}$）。中间蕴含 $G_i \Rightarrow A_{i+1}$ 称为**契约兼容性检查**，由 Composition Checker 显式验证。

**对应消融**：Figure 5 显示不同精化参数对验证成功率与运行时间的影响，迭代深度与假设初始宽度存在显著权衡。

## 实验与分析



本文在 VNN-COMP 标准 benchmark 上评估 CoVeNN，涵盖7个神经网络与140个性质规格。Figure 4 展示了验证进展的核心指标：样本分类百分比与标签范围缩减百分比。核心结果表明，CoVeNN 验证通过的问题数达到 state-of-the-art 验证器（αβ-CROWN、NeuralSAT 等）的近7倍——这一差距主要源于 CoVeNN 将深层网络验证的内存瓶颈转化为可逐个求解的子问题，使得原本因 GPU 内存不足而返回 UNKNOWN 的大量实例获得确定性结果。


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/fad17aa6-2093-4cd3-a746-96c3f0d3104f/figures/Figure_3.png)
*Figure 3 (comparison): On ACAS Xu properties comparison of AGNN variants.*



具体而言，在 ACAS Xu 性质验证上，Figure 3 对比了 AGNN（Assume-Guarantee Neural Network）各变体与 monolithic 方法的性能差异，组合式方法在性质满足判定与反例搜索上均展现出更稳定的收敛行为。Figure 1 则在 Reachability bound test case 中直观展示了 CoVeNN 与 CROWN 的边界紧致性对比：组合方法通过逐层精化获得显著更紧的可达集估计。



消融实验（Figure 5）检验了迭代精化参数的影响。关键发现包括：去掉迭代精化（仅使用初始粗糙假设）后，子问题验证成功率大幅下降，端到端验证通过问题数减少约60%；而过度激进的精化步长虽提升单个子问题精度，却增加迭代轮数与总运行时间，表明假设宽度与精化速率的自适应平衡是性能关键。作者披露，当前评估限于7个网络架构，对 Transformer 等新兴架构的分解策略仍需探索；此外，迭代精化在最坏情况下可能引入额外计算开销，尽管平均意义上因问题可解性提升而净收益显著。

公平性方面，对比基线 αβ-CROWN（VNN-COMP 2023/2024 冠军方法）、NeuralSAT（作者自身 prior 工具，直接 lineage 父节点）、PyRAT（2025 最新验证器）、Marabou 2.0 与 nnenum 均为该领域最强或最常用工具，无显著 baseline 弱势问题。实验硬件与超时设置于 §4 及 Appendix E 详细说明。

## 方法谱系与知识库定位

CoVeNN 属于 **Neural Network Verification** 方法族，直接父方法为 **NeuralSAT** [17]——作者团队先前提出的高性能单体验证工具。CoVeNN 在 NeuralSAT 基础上完成三项核心 slot 变更：将 inference_strategy 从 monolithic 端到端验证替换为组合式假设-保证分解；将 architecture 从单一验证器扩展为参数化框架（verifier-agnostic wrapper）；将 exploration_strategy 从直接 bound tightening 增强为迭代精化驱动的假设计算。

**直接基线与差异**：
- **αβ-CROWN** [9]：VNN-COMP 冠军，多神经元松弛分支定界；CoVeNN 将其作为可包裹的底层验证器之一，而非竞争替代
- **Compositional learning and verification of neural network controllers** [23]：最直接的组合验证 prior work；CoVeNN 引入系统化的 assume-guarantee 契约与迭代精化，弥补了其在分解粒度自适应与 soundness 保证上的不足
- **PyRAT** [18] / **Marabou 2.0** [11] / **nnenum** [12]：标准 VNN-COMP 竞争对手，CoVeNN 在相同 benchmark 上取得近7倍验证通过问题数

**后续方向**：(1) 将组合分解扩展至 Transformer、GNN 等非 sequential 架构；(2) 结合学习式假设生成替代手工精化规则，进一步加速契约收敛；(3) 探索分布式验证场景下子问题的并行求解与假设同步机制。

**标签**：modality=image / paradigm=formal_verification / scenario=adversarial_robustness / mechanism=assume-guarantee_decomposition, iterative_refinement, CEGAR / constraint=memory_scalability, verifier-agnostic

## 引用网络

### 直接 baseline（本文基于）

- Scalable Neural Network Verification with Branch-and-bound Inferred Cutting Planes _(NeurIPS 2024, 实验对比, 未深度分析)_: Recent verification method with cutting planes; likely appears in experimental c

