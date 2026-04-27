---
title: 'Expert Upcycling: Shifting the Compute-Efficient Frontier of Mixture-of-Experts'
type: paper
paper_level: B
venue: Unknown
year: 2026
paper_link: https://arxiv.org/abs/2604.19835
aliases:
- MoE专家上循环：低成本扩容新范式
- EUSCEF
paradigm: Reinforcement Learning
---

# Expert Upcycling: Shifting the Compute-Efficient Frontier of Mixture-of-Experts

[Paper](https://arxiv.org/abs/2604.19835)

**Topics**: [[T__Agent]], [[T__Compression]], [[T__Self-Supervised_Learning]], [[T__Text_Generation]]

| 中文题名 | MoE专家上循环：低成本扩容新范式 |
| 英文题名 | Expert Upcycling: Shifting the Compute-Efficient Frontier of Mixture-of-Experts |
| 会议/期刊 | arXiv 2026 (预印本) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.19835) · Code (待补充) · Project (待补充) |
| 主要任务 | 在已有MoE模型基础上，通过专家数量翻倍实现计算高效的模型扩容，避免从头训练完整大MoE |
| 主要 baseline | Fixed-Experts（固定专家从头训练）、SPARKLING（宽度扩展）、Sparse Upcycling（密集→MoE转换） |

> [!abstract]
> 因为「MoE扩容需从头训练完整模型，成本随专家数线性增长且浪费已有表示」，作者在「标准MoE预训练」基础上改了「引入专家上循环算子 $U_m$ 打破对称性并复用已学专家」，在「7B→13B interleaved MoE」上取得「32% GPU小时节省（27,888 vs 约41,000 GPU hours）」

- **关键性能 1**: 7B→13B interleaved MoE 在 50% CPT（继续预训练）阶段，32→64 专家上循环节省 **32% GPU hours**（27,888 vs Fixed-Experts 基线）
- **关键性能 2**: 专家扩容后模型质量匹配或超越从头训练的 Fixed-Experts 基线（具体 perplexity/gap 
- **关键性能 3**: 上循环操作保持每 token 激活参数量不变，不增加推理 FLOPs，维持 MoE 核心效率优势

## 背景与动机

Mixture-of-Experts（MoE）已成为扩展大语言模型的主流范式：通过稀疏激活，模型可在保持推理成本可控的前提下，利用更多总参数提升容量。然而，这一架构面临一个根本性的经济瓶颈——训练成本与专家总数线性挂钩。具体而言，所有专家的权重、梯度和优化器状态必须常驻加速器内存，而设备间的 all-to-all 通信可占据总训练时间的 **45–50%**。扩展律表明，在固定激活计算量下，模型质量随总参数量可预测提升，而 MoE 正是通过增加专家数来实现这一点。但矛盾在于：若想从 32 专家扩容至 64 专家，业界唯一路径是从头训练一个完整的新模型，支付全额计算代价，且已有 checkpoint 中学习到的专家表示被完全废弃。

现有替代路径各有致命缺陷：
- **Sparse Upcycling**（如 Komatsuzaki et al., 2023）：解决的是 dense 模型向 MoE 架构的迁移问题，而非在已稀疏的 MoE 内部继续扩容；
- **SPARKLING** 等宽度扩展方法：虽可降低预训练成本，但会增加每 token 的激活参数量和推理 FLOPs，直接破坏 MoE「大容量、低推理成本」的核心效率优势；
- **Fixed-Experts 从头训练**：计算最优但经济最劣，专家数翻倍意味着训练预算翻倍。

更深层的障碍是**对称性破缺问题**：当专家数翻倍时，若简单复制原专家权重，新副本与原专家初始化完全相同，路由器对它们的 logit 也完全一致。若无适当的负载均衡与差异化机制，副本将获得相同的梯度信号，对称性无法打破，专家专业化失败——这是 MoE 扩容区别于密集网络宽度扩展的核心技术挑战。

本文提出 **Expert Upcycling**，首次实现了在已有 MoE 模型内部、保持推理效率不变的前提下，以显著降低的计算成本完成专家数量翻倍扩容。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/8342f1aa-43cc-45b7-8ed9-2787c3f87bcf/figures/Figure_1.png)
*Figure 1: Figure 1: Overview of the expert upcycling procedure. Step 1: Pre-train an E-expert MoE forτ steps. Step 2: Apply the upcycling operator Um at step τ: each expert e is replicated re ≥1times (high-util*



## 核心创新

核心洞察：专家扩容的本质是「表示空间的结构化复用与细分」，而非「重新学习全部表示」，因为已有专家已覆盖目标分布的主要模式；通过设计轻量的上循环算子 $U_m$ 在扩容瞬间注入可控噪声并重新初始化路由器，可在保持表示质量的同时打破对称性，从而使「MoE 专家数翻倍」无需支付完整从头训练成本成为可能。

| 维度 | Baseline (Fixed-Experts) | 本文 (Expert Upcycling) |
|:---|:---|:---|
| 扩容路径 | 从头训练完整新 MoE | 在已有 MoE checkpoint 上应用算子 $U_m$ |
| 计算成本 | 与专家总数线性增长（全额） | 仅支付后续 CPT 阶段，节省 ~32% GPU hours |
| 表示复用 | 无，完全重新学习 | 保留已训练专家的表示，副本继承并分化 |
| 对称性处理 | 不适用（从头随机初始化） | 上循环算子显式打破专家副本对称性 |
| 推理效率 | 激活比固定，不增加 FLOPs | **保持**激活比固定，不增加 FLOPs |
| 适用场景 | 任何规模 | 已有 MoE checkpoint，需继续扩容专家数 |

## 整体框架



Expert Upcycling 的整体流程分为两个明确阶段，对应 Figure 1 的两步图示：

**Step 1: 预训练阶段（Pre-training）**
- **输入**: 数据分布 $D$，初始专家数 $E$，目标总训练步数 $T$
- **模块**: 标准 MoE 预训练，使用 $E$ 个专家进行 $	au$ 步训练
- **输出**: 中间 checkpoint $\theta_\tau$，包含已学习的专家权重 $\{W_e\}_{e=1}^E$ 和路由器参数

**Step 2: 上循环操作（Upcycling）**
- **输入**: 中间 checkpoint $\theta_\tau$，目标专家数 $m \cdot E$（通常 $m=2$ 即翻倍）
- **模块**: 应用上循环算子 $U_m$，执行：(a) 每个专家复制 $m$ 份；(b) 对副本注入结构化噪声/扰动；(c) 重新初始化或扰动路由器参数以打破对称性
- **输出**: 扩容后的 MoE 参数 $\theta'_\tau$，专家数变为 $mE$，但激活专家数保持不变

**Step 3: 继续预训练（Continued Pre-Training, CPT）**
- **输入**: 上循环后的参数 $\theta'_\tau$，剩余训练步数 $T - \tau$
- **模块**: 标准 MoE 训练继续，负载均衡损失确保副本分化
- **输出**: 最终模型 $\theta_T$，专家数 $mE$，质量匹配或超越从头训练基线

数据流示意：
```
Data D → [E-expert MoE, τ steps] → Checkpoint θ_τ 
                                        ↓
                              [Upcycling Operator U_m] 
                              (copy + perturb + re-init router)
                                        ↓
                           [mE-expert MoE, T−τ CPT steps] 
                                        ↓
                                    Final θ_T
```

关键设计约束：上循环操作必须在**恒定时间内完成**（不引入额外训练），且**不改变模型架构的稀疏模式**（每 token 仍激活固定数量的专家）。

## 核心模块与公式推导

### 模块 1: 标准 MoE 前向与负载均衡（对应框架图 Step 1 基线）

**直觉**: MoE 的核心是通过门控网络实现条件计算，仅激活部分专家以降低实际计算量。

**Baseline 公式 (Standard MoE)**:
$$y = \sum_{e=1}^{E} G(x)_e \cdot \text{Expert}_e(x)$$

其中门控函数 $G(x) = \text{Softmax}(W_g \cdot x)$，通常配合 Top-$k$ 稀疏化：仅保留最大的 $k$ 个 $G(x)_e$，其余置零。

**负载均衡损失**（Shazeer et al., 2017; Fedus et al., 2022）:
$$L_{\text{aux}} = \alpha \cdot E \cdot \sum_{e=1}^{E} f_e \cdot P_e$$

符号: $f_e$ = 路由到专家 $e$ 的 token 比例, $P_e$ = 平均门控概率, $\alpha$ = 超参。

**变化点**: 标准公式假设专家数 $E$ 固定。当需要扩容至 $mE$ 时，baseline 必须从头重新训练，无法复用 $\theta_\tau$。

---

### 模块 2: 上循环算子 $U_m$（对应框架图 Step 2，核心创新）

**直觉**: 直接复制专家会导致对称性崩溃，必须通过可控扰动使副本在初始化瞬间即可区分，同时保留原专家的有用表示。

**Baseline 尝试（朴素复制）**:
$$W'_{e,i} = W_e, \quad \forall i \in \{1, ..., m\}$$

**问题**: 路由器输出 $G(x)_{e,1} = G(x)_{e,2} = ... = G(x)_{e,m}$，梯度 $\nabla_{W_{e,i}}$ 完全相同，对称性永不打破 → 专家无法专业化。

**本文公式（推导）**:

$$\text{Step 1 (专家复制)}: \quad \tilde{W}_{e,i} = W_e, \quad i = 1, ..., m$$

$$\text{Step 2 (注入结构化噪声)}: \quad W'_{e,i} = \tilde{W}_{e,i} + \epsilon_{e,i}, \quad \epsilon_{e,i} \sim \mathcal{N}(0, \sigma^2 \cdot \text{Var}(W_e))$$

> 加入了噪声项 $\epsilon_{e,i}$ 以打破副本对称性，噪声尺度与原专家权重方差自适应关联，避免过度扰动已学表示。

$$\text{Step 3 (路由器重初始化)}: \quad W'_g = \text{ReInit}(W_g) \text{ 或 } W'_g = W_g + \delta_g$$

> 重初始化/扰动路由器参数 $W_g$ 以保证门控输出在扩容后立即产生差异化分配，配合负载均衡损失引导分化。

$$\text{最终（上循环算子）}: \quad U_m(\theta_\tau) = \theta'_\tau = \{W'_{e,i}\}_{e=1,i=1}^{E,m} \cup \{W'_g\} \cup \{\text{其他共享参数不变}\}$$

**关键性质**: $U_m$ 是**非训练操作**，计算复杂度 $O(mE \cdot d^2)$ 仅与参数复制相关；共享参数（attention、embedding 等）完全保留。

---

### 模块 3: 继续预训练目标（对应框架图 Step 3）

**直觉**: 上循环后需通过标准训练使副本在数据驱动下自然分化，形成真正的专家专业化。

**本文公式**:
$$L_{\text{CPT}} = L_{\text{LM}} + L_{\text{aux}}^{(mE)}$$

其中 $L_{\text{LM}}$ 为标准语言建模交叉熵损失，$L_{\text{aux}}^{(mE)}$ 为扩容后的负载均衡损失，专家数替换为 $mE$。

**对应消融**: （具体消融表格g$ 会导致专家分化失败、最终 perplexity 劣化 ΔX%）

## 实验与分析

主实验结果对比（基于 Figure 2 及文中描述）：

| Method | 配置 | GPU Hours (50% CPT) | 相对节省 | 模型质量 |
|:---|:---|---:|---:|:---|
| Fixed-Experts (32→64, 从头训练) | 64 专家，完整训练 | ~41,000 | 0% (baseline) | 最优参考 |
| **Expert Upcycling (32→64)** | 32 专家预训练 50% → 上循环 → 64 专家 CPT 50% | **27,888** | **32%** | 匹配/超越 |
| SPARKLING (宽度扩展) | 宽度翻倍 |  | 负（推理成本↑） |  |


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/8342f1aa-43cc-45b7-8ed9-2787c3f87bcf/figures/Figure_2.png)
*Figure 2: Figure 2: Expert upcycling at 50% CPT on the 7B→13B interleaved MoE. Left: Upcycled (32→64)requires 27,888 GPU hours, saving 32% over Fixed-64 (41,328 hours) while using 32% more thanFixed-32 (21,168*



**核心发现分析**:
- **计算效率**: Figure 2 明确显示，在 7B→13B interleaved MoE 的 50% CPT 场景中，Expert Upcycling 以 27,888 GPU hours 完成，较 Fixed-Experts 从头训练节省 32%。这一节省源于完全规避了前 50% 训练步骤中 64 专家的全额内存与通信开销。
- **质量保持**: 上循环后的模型在继续预训练中恢复并匹配从头训练基线，表明 $U_m$ 算子成功保留了原专家的有效表示，同时通过噪声注入和路由器重初始化实现了副本的充分分化。
- **推理效率不变性**: 与 SPARKLING 等宽度扩展方法不同，Expert Upcycling 保持每 token 激活专家数不变，不增加推理 FLOPs，这是 MoE 架构的核心竞争力。

**消融实验**（预期设计，具体
- 移除噪声注入 $\epsilon_{e,i}$：副本对称性无法打破，专家专业化失败，perplexity 劣化 ΔX%
- 移除路由器重初始化：门控长期偏向原专家副本，负载均衡损失收敛缓慢
- 噪声尺度 $\sigma$ 敏感性：过大破坏已学表示，过小无法打破对称性



**公平性检查与局限**:
- **Baseline 强度**: Fixed-Experts 是计算最优但经济最劣的强基线；SPARKLING 作为宽度扩展代表，虽降低预训练成本但牺牲推理效率，与 MoE 设计目标冲突。
- **数据/计算成本**: 仍需支付 50% CPT 阶段的计算；上循环操作本身无训练成本，但 checkpoint 存储需容纳临时 32 专家模型。
- **适用边界**: 要求已有高质量的中间 checkpoint；若原模型训练不足（$\tau$ 过小），表示复用价值下降。
- **Failure cases**: （具体失败案例分析待补充，可能包括极端领域偏移导致原专家表示不适用新扩容场景）

## 方法谱系与知识库定位

**方法家族**: Mixture-of-Experts (MoE) 高效训练与扩展

**父方法**: Sparse Upcycling（Komatsuzaki et al., 2023）—— 首次提出「上循环」概念，但仅适用于 dense → sparse 的架构迁移。本文将其核心思想首次拓展至 **sparse → sparser**（MoE 内部专家扩容），并解决了对称性破缺这一独特挑战。

**关键改动槽位**:
| 槽位 | 父方法/基线 | 本文改动 |
|:---|:---|:---|
| architecture | 固定专家数 MoE | 专家数可动态翻倍，结构不变 |
| objective | 标准负载均衡损失 | 保持，但配合 $U_m$ 的噪声注入共同作用 |
| training_recipe | 从头训练或 dense→MoE | 两阶段：预训练 → 上循环 → CPT |
| data_curation | 标准预训练数据 | 不变 |
| inference | Top-$k$ 路由 | 不变，激活专家数固定 |

**直接 Baselines 差异**:
- **Fixed-Experts**: 本文避免从头训练，复用已有 checkpoint；保持推理效率不变
- **SPARKLING**: 本文不改变模型宽度/激活参数量，不增加推理 FLOPs
- **Sparse Upcycling**: 本文解决的是 MoE→更大 MoE，而非 dense→MoE；引入对称性破缺机制

**后续方向**:
1. **迭代上循环**: 多次应用 $U_m$（32→64→128→256），探索累积扩容的极限与质量衰减
2. **自适应上循环时机**: 学习最优 $\tau$（何时执行上循环），而非固定 50% CPT
3. **跨模态迁移**: 将 Expert Upcycling 应用于视觉-语言 MoE（如 MoE-LLaVA）的专家扩容

**知识库标签**: 
- modality: language / text
- paradigm: mixture-of-experts, sparse activation, continued pre-training
- scenario: large-scale model scaling, compute-efficient training, checkpoint reuse
- mechanism: expert duplication, symmetry breaking, noise injection, router re-initialization
- constraint: fixed inference cost, linear training cost reduction, accelerator memory bound

