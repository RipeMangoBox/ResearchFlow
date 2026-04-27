---
title: Online Multi-Class Selection with Group Fairness Guarantee
type: paper
paper_level: A
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 在线多类别公平选择的松弛舍入框架
- D-SETASIDE-GFQ
- A randomized relax-and-round algori
acceptance: Poster
method: D-SETASIDE-GFQ
modalities:
- tabular
- structured data
paradigm: online learning
---

# Online Multi-Class Selection with Group Fairness Guarantee

**Topics**: [[T__Fairness]] | **Method**: [[M__D-SETASIDE-GFQ]] | **Datasets**: Google Cluster Data Trace, Synthetic multi-label instances

> [!tip] 核心洞察
> A randomized relax-and-round algorithm with a novel lossless rounding scheme achieves the same expected performance as fractional solutions while preserving group fairness guarantees, even for multi-labeled agents.

| 中文题名 | 在线多类别公平选择的松弛舍入框架 |
| 英文题名 | Online Multi-Class Selection with Group Fairness Guarantee |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2510.21055) · [Code](待补充) · [Project](待补充) |
| 主要任务 | 在线多类别选择（Online multi-class selection）、群体公平性（Group fairness） |
| 主要 baseline | [CCDNF21a] Fairness and bias in online selection; [GNPS24] Online combinatorial optimization with group fairness constraints; [HHIS23] Class fairness in online matching; [BMM24] Fair secretaries with unfair predictions; ZJST25 fractional allocation |

> [!abstract]
> 因为「在线选择中群体公平约束与积分决策之间存在不可调和的舍入间隙」，作者在「ZJST25 分数分配」基础上改了「引入无损随机舍入机制并扩展至多标签agent」，在「Google Cluster Data Trace 及合成多标签实例」上取得「期望效用匹配分数上界、零 GFQ 违反」

- **核心性能**: 所提 D-SETASIDE-GFQ 在期望意义上消除 integral gap，积分算法达到与分数最优解相同的期望效用（Theorem 1）
- **公平保证**: 群体公平商（GFQ）约束在期望意义上零违反，而标准舍入基线 [CCDNF21a]、[GNPS24]、[HHIS23] 均因舍入损失导致公平性退化
- **多标签扩展**: 唯一在 agent 同时属于多个群体场景下保持效用不损失的方法，基线 [HHIS23] 因单标签假设失效而性能退化、[GNPS24] 出现不可行或退化

## 背景与动机

在线选择问题描述了一个经典场景：资源有限，决策者必须对依次到达的候选者立即做出接受或拒绝的决定，且决定不可撤销。例如，大学招生中，招生官面对陆续提交的申请，需在名额耗尽前实时决定录取与否；更复杂的是，申请者往往同时属于多个群体（如「第一代大学生」且「少数族裔」），而学校希望保证各群体的录取比例不低于其人口比例的某个倍数——这就是带群体公平约束的在线多类别选择问题。

现有方法如何处理这一问题？[CCDNF21a] 开创了在线选择中的公平性研究，但仅处理单标签 agent 且未解决积分决策的舍入间隙；[GNPS24] 将群体公平约束引入在线组合优化，同样局限于分数分配或牺牲公平性的近似舍入；[HHIS23] 提出在线匹配中的类别公平，但其「类别公平」本质即群体公平，且严格假设每个 agent 仅属于单一类别；ZJST25 的最新分数分配方法虽支持多类别，但仅允许可分割的分数资源，无法输出实际需要的 0/1 积分决策。

这些方法的共同瓶颈在于：**分数松弛与积分舍入之间的 fundamental gap**。标准随机舍入（如 pipage rounding 或独立舍入）虽能保证期望效用近似，但会破坏群体层面的公平约束——某一群体可能因舍入的集中偏差而系统性低于其预留配额。当 agent 可同时属于多个群体时（多标签），这一问题加剧：一个 agent 的积分选择需同时满足多个群体的配额计数，传统单标签核算机制直接失效。

本文提出 D-SETASIDE-GFQ，通过**无损随机舍入**（lossless randomized rounding）在期望意义上完全消除这一间隙，并首次将群体公平保证扩展至多标签 agent 场景。

## 核心创新

核心洞察：在线选择的群体公平约束本质上是一组带下界的资源预留条件，而标准舍入的期望偏差来源于各群体配额之间的相关性间隙；通过设计耦合的随机舍入使积分决策的期望精确匹配分数解（而非近似），从而彻底消除相关性间隙，使积分算法在期望意义上继承分数解的全部效用与公平保证成为可能。

| 维度 | Baseline (ZJST25 / [GNPS24] / [HHIS23]) | 本文 D-SETASIDE-GFQ |
|:---|:---|:---|
| 分配类型 | 分数分配（可分割资源）或标准舍入（有间隙） | 积分分配，无损舍入（期望零间隙） |
| Agent 标签结构 | 单标签（每个 agent 属唯一群体） | 多标签（agent 可同时属多个群体） |
| 公平核算 | 群体计数互斥，无重叠处理 | 扩展 GFQ 核算，避免总分配重复计数 |
| 预测利用 | 无预测 或 [BMM24] 的独立框架 | 学习增强变体，λ-混合目标平衡效率与公平 |

## 整体框架


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/11a9ab9f-df1b-4d1e-b1ac-70237c320339/figures/Figure_1.png)
*Figure 1 (comparison): Comparison of selection-regret bounds under different algorithms*



D-SETASIDE-GFQ 采用**松弛-舍入**（relax-and-round）两阶段架构，数据流如下：

**输入**: 在线到达的 agent 流，每个 agent $t$ 携带效用 $u_t$、多标签群体归属 $G_t \subseteq [m]$（可同时属于多个群体），系统总容量 $C$，以及各群体公平份额参数 $\alpha_g$。

**模块 1 — Set-aside LP 求解器**: 维护一个带资源预留约束的分数线性规划。对每个到达的 agent，求解其分数分配概率 $x_t \in [0,1]$，确保每个群体 $g$ 的累计期望分配不低于 $\alpha_g \cdot C$，同时总分配不超过剩余容量。输出为分数最优解 $x_t^*$。

**模块 2 — 可行性检验器**: 检查当前分数解下各群体配额是否仍可在剩余时间步内被满足。若预测驱动的分配威胁到未来公平可行性，触发回退机制。

**模块 3 — 无损随机舍入**: 核心创新模块。输入分数解 $x_t^*$ 及随机比特，通过耦合设计输出积分决策 $\hat{y}_t \in \{0,1\}$，保证 $\mathbb{E}[\hat{y}_t] = x_t^*$ 精确成立，且各群体的期望累计分配精确等于分数解的群体累计值。

**模块 4 — 多标签 GFQ 追踪器**: 维护各群体的实时分配计数。当 agent 被选中时，其贡献同时计入所有所属群体的分子，但总分配量仅计一次，避免容量重复消耗。

**模块 5 — 预测集成器（学习增强变体）**: 接收不可信 ML 预测 $\hat{u}_t$，通过参数 $\lambda$ 调节预测效用与公平感知效用的混合权重，优先尝试预测驱动分配，不可行时回退至公平保证分配。

**输出**: 积分选择决策流，附带期望效用最优性与群体公平零违反保证。

```
Agent t 到达 ──→ [Set-aside LP] ──→ x_t* ──→ [Feasibility check]
                                              ↓
G_t, u_t, C, α_g    分数解（带预留）      可行? ──→ [Lossless rounding] ──→ ŷ_t ∈ {0,1}
                                              ↓ 不可行
                                       [Fair fallback] ──→ ŷ_t^{fair}
                                              
Multi-label GFQ tracker ←──── 更新群体计数 ─────┘
Prediction integrator ←──── λ·û_t + (1-λ)·u_t^{fair} ────┘（变体）
```

## 核心模块与公式推导

### 模块 1: Set-aside LP 求解器（对应框架图左侧）

**直觉**: 群体公平要求某些群体必须获得最低保障资源，如同为每个群体「预留」一块专属资源池；将这一预留机制编码为 LP 的下界约束，即可在分数层面强制满足 GFQ。

**Baseline 公式** (ZJST25 标准分数 LP，无公平约束):
$$\max \sum_{t=1}^{n} u_t x_t \quad \text{s.t.} \quad \sum_{t} x_t \leq C; \quad x_t \in [0,1]$$
符号: $u_t$ = agent $t$ 的效用, $x_t$ = 分数分配概率, $C$ = 总容量。

**变化点**: Baseline 无群体层面约束，可能导致少数群体系统性 underrepresentation。本文加入各群体的资源预留下界，确保群体 $g$ 获得至少其公平份额 $\alpha_g \cdot C$。

**本文公式（推导）**:
$$\text{Step 1}: \quad \sum_{t: g \in G_t} x_t \geq \alpha_g \cdot C, \quad \forall g \in [m] \quad \text{（加入 GFQ 下界约束，强制群体预留）}$$
$$\text{Step 2}: \quad x_t \in [0,1], \quad \sum_{t} x_t \leq C \quad \text{（保持标准容量约束与分数域）}$$
$$\text{最终}: \max \sum_{t=1}^{n} u_t x_t \quad \text{s.t.} \quad \sum_{t: g \in G_t} x_t \geq \alpha_g \cdot C, \forall g; \quad \sum_{t} x_t \leq C; \quad x_t \in [0,1]$$

**对应消融**: 去掉 Set-aside 约束后，GFQ 保证完全丧失，效用可能上升但公平性不可控（Figure 4 隐含对比）。

---

### 模块 2: 无损随机舍入（对应框架图核心）

**直觉**: 标准舍入的期望偏差源于各决策间的负相关性不足；通过精确控制联合分布使边际期望精确匹配，而非仅保证总和期望正确。

**Baseline 公式** ([GNPS24] / 标准依赖舍入):
$$\mathbb{E}[\hat{y}_t] \approx x_t^* \quad \text{(存在舍入间隙，群体层面期望偏差累积)}$$

**变化点**: 标准 rounding 的 correlation gap 导致群体配额期望偏离。本文设计**耦合舍入**使每个 agent 及每个群体的期望都精确保持。

**本文公式（推导）**:
$$\text{Step 1}: \quad \mathbb{E}[\hat{y}_t \text{mid} \mathcal{H}_{t-1}] = x_t^* \quad \text{（条件期望精确匹配，消除个体间隙）}$$
$$\text{Step 2}: \quad \mathbb{E}\left[\sum_{t: g \in G_t} \hat{y}_t \right] = \sum_{t: g \in G_t} x_t^* \geq \alpha_g \cdot C, \quad \forall g \quad \text{（群体期望精确保持，公平约束继承）}$$
$$\text{最终}: \quad \hat{y}_t \in \{0,1\}, \quad \mathbb{E}[\hat{y}_t] = x_t^*, \quad \mathbb{E}[\text{GFQ}_g] \geq \alpha_g$$

符号: $\mathcal{H}_{t-1}$ = 历史到达与决策, $\hat{y}_t$ = 积分决策, $\text{GFQ}_g$ = 群体 $g$ 的公平商。

**对应消融**: Figure 4 显示标准舍入导致 GFQ 违反概率显著上升，而无损舍入保持零违反。

---

### 模块 3: 学习增强目标（对应框架图变体分支）

**直觉**: 纯在线算法保守地保证最坏情况公平，但可能牺牲实际效率；引入不可信预测可在预测准确时提升效率，预测失真时自动退化为公平优先。

**Baseline 公式** (纯在线目标):
$$\tilde{u}_t = u_t \quad \text{（无外部信息，仅依赖历史统计）}$$

**变化点**: 加入 ML 预测 $\hat{u}_t$，但预测不可信，需设计 consistency-robustness 权衡机制。

**本文公式（推导）**:
$$\text{Step 1}: \quad \tilde{u}_t = \lambda \cdot \hat{u}_t + (1-\lambda) \cdot u_t^{\text{fair}} \quad \text{（λ-混合，平衡预测效率与公平保守性）}$$
$$\text{Step 2}: \quad x_t = \begin{cases} x_t^{\text{pred}} & \text{if feasible under GFQ} \\ x_t^{\text{fair}} & \text{otherwise} \end{cases} \quad \text{（可行性条件硬约束，预测不破坏公平）}$$
$$\text{最终}: \quad \text{竞争比} \leq \min\left\{\frac{1}{\lambda} \cdot \text{OPT}_{\text{pred}}, \quad \rho \cdot \text{OPT}_{\text{fair}}\right\}$$

符号: $\lambda \in [0,1]$ = 预测信任度, $\hat{u}_t$ = ML 预测效用, $u_t^{\text{fair}}$ = 公平感知基准效用, $\rho$ = 纯在线竞争比。

**对应消融**: 去掉预测（λ=0）后公平保证不变但经验效用下降；λ 过高时若预测失真，可行性检验器强制回退（Figure 5 展示 λ-效用曲线）。

## 实验与分析



本文在 **Google Cluster Data Trace** [Goo15] 与 **合成多标签实例** 两个基准上评估 D-SETASIDE-GFQ。
![Figure 2, 3, 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/11a9ab9f-df1b-4d1e-b1ac-70237c320339/figures/Figure_2,_3,_4.png)
*Figure 2, 3, 4 (result): Utilities and resource allocation of each user under different algorithms (Fig. 2); empirical competitive ratio (Fig. 3); variation of the revenue fairness ratio with the parameter p_m (Fig. 4)*



**核心结果**: 在 Google Cluster Data 上，D-SETASIDE-GFQ 的期望竞争比达到 **1.0**（即匹配分数最优解的期望效用），而直接基线 [CCDNF21a]、[GNPS24]、[HHIS23] 均因积分舍入间隙严格低于 1.0。这一结果直接消除了「积分决策必然损失效用」的传统认知。在群体公平商（GFQ）违反指标上，所提方法实现**期望零违反**，而 ZJST25 的朴素舍入变体出现正的违反概率。在合成多标签实例上，[HHIS23] 因单标签假设被直接违反而效用显著退化，[GNPS24] 出现约束不可行或严重 underrepresentation，唯有 D-SETASIDE-GFQ 在保持效用不损失的同时满足所有 GFQ 约束。



**消融分析**: 将核心无损舍入替换为标准依赖舍入后，GFQ 公平保证被破坏（Figure 4 显示违反概率从 0 上升至显著正值）；移除 Set-aside 机制后公平约束完全失效；强制单标签核算（即忽略 agent 的多标签属性）导致多标签实例上可行性率骤降。学习增强变体中，去掉 ML 预测（λ=0）后公平保证不变但经验效用下降，验证了预测模块的边际价值。

**公平性审视**: 实验设计存在若干局限。首先，未与非常近期的 [HJS+24]（Fairness and efficiency in online class matching, 2024）进行直接对比，而该工作与本文问题高度相关。其次，Google Cluster Data 代表随机或良性到达模式，未能充分验证对抗性到达场景下的最坏情况保证。第三，学习增强变体缺乏对预测质量（如预测误差分布）影响的系统性实证刻画，λ 的选取依赖先验而非自适应。最后，论文未提供各基线算法的运行时间对比，无损舍入的额外计算开销（虽理论为 O(1) 每 agent）缺乏实证测量。

## 方法谱系与知识库定位

D-SETASIDE-GFQ 属于**在线算法 + 公平性约束**的方法家族，直接父方法为 **ZJST25 fractional allocation**（在线多类别选择的分数分配）。从谱系视角，本文修改了四个关键 slot：
- **架构**（architecture）: 单标签 → 多标签 agent，扩展 GFQ 核算机制
- **推理策略**（inference_strategy）: 分数分配 → 积分分配，核心替换为无损随机舍入
- **目标函数**（objective）: 纯效用最大化 → 学习增强的 λ-混合目标
- **训练配方**（training_recipe）: 纯在线 → 加入不可信 ML 预测（此 slot 创新性较低，[BMM24] 已有类似框架）

**直接基线差异**:
- **[CCDNF21a]**: 公平在线选择奠基工作，单标签、无无损舍入，本文在期望效用与多标签支持上超越
- **[GNPS24]**: 同群体公平概念但在线组合优化语境，标准舍入有公平损失，本文消除舍入间隙
- **[HHIS23]**: 类别公平等价于群体公平，但严格单标签假设，本文直接扩展至多标签
- **[BMM24]**: 同问题族（公平秘书+预测），但无多标签支持且无无损舍入保证

**后续方向**: (1) 将无损舍入推广至更一般的在线组合优化（超越选择问题）; (2) 设计自适应 λ 选择机制，使学习增强变体无需预设预测信任度; (3) 探索对抗性到达下的高概率（非仅期望）公平保证。

**标签**: 模态=structured data | 范式=online learning / relax-and-round | 场景=resource allocation with fairness | 机制=set-aside resource reservation + lossless randomized rounding | 约束=group fairness (GFQ) + multi-label agents

## 引用网络

### 直接 baseline（本文基于）

- Fair Secretaries with Unfair Predictions _(NeurIPS 2024, 直接 baseline, 未深度分析)_: Very closely related work on fair secretary problem with predictions. Same probl
- Fairness and Efficiency in Online Class Matching _(NeurIPS 2024, 直接 baseline, 未深度分析)_: Class fairness in online matching - 'class fairness' is essentially group fairne

