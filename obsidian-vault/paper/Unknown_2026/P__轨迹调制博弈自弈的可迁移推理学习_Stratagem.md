---
title: 'Stratagem: Learning Transferable Reasoning via Trajectory-Modulated Game Self-Play'
type: paper
paper_level: B
venue: Unknown
year: 2026
paper_link: https://arxiv.org/abs/2604.17696
aliases:
- 轨迹调制博弈自弈的可迁移推理学习
- Stratagem
method: Stratagem
---

# Stratagem: Learning Transferable Reasoning via Trajectory-Modulated Game Self-Play

[Paper](https://arxiv.org/abs/2604.17696)

**Topics**: [[T__Math_Reasoning]], [[T__Code_Generation]] | **Method**: [[M__Stratagem]]

| 中文题名 | 轨迹调制博弈自弈的可迁移推理学习 |
| 英文题名 | Stratagem: Learning Transferable Reasoning via Trajectory-Modulated Game Self-Play |
| 会议/期刊 | arXiv 2026 (预印本) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.17696) · Code (待补充) · Project (待补充) |
| 主要任务 | 数学推理、通用推理、代码生成 |
| 主要 baseline | PPO, DPO, RFT, Self-Play Fine-Tuning (SPIN), V-STaR, RStar-Math |

> [!abstract] 因为「传统自弈方法仅依赖终端奖励学习游戏专用启发式策略，导致推理能力难以迁移到新领域」，作者在「博弈论自弈框架」基础上改了「引入轨迹层面的调制优势函数，结合可迁移的推理演化奖励」，在「GSM8K、MATH、ARC-C、HumanEval等benchmark」上取得「相比PPO提升5.2%-12.8%，跨领域迁移显著优于专用训练模型」

- **GSM8K**: 从 PPO 的 79.1% 提升至 87.6%（+8.5%）
- **MATH**: 从 PPO 的 42.3% 提升至 50.1%（+7.8%）
- **HumanEval**: 从 PPO 的 63.4% 提升至 76.2%（+12.8%）

## 背景与动机

大型语言模型（LLM）的推理能力通常通过监督微调（SFT）或强化学习（RL）来提升，但这些方法面临一个根本性困境：模型容易过拟合到训练任务的特定模式，而难以将学到的推理策略迁移到未见过的领域。例如，一个在GSM8K数学问题上训练到87%准确率的模型，可能在ARC科学推理任务上表现骤降，因为其学到的只是"数字运算启发式"而非通用的"问题分解策略"。

现有方法如何处理这一问题？**PPO（Proximal Policy Optimization）** 通过裁剪目标函数稳定训练，但完全依赖终端奖励（如答案对错），无法区分"接近正确"和"完全错误"的推理路径。**DPO（Direct Preference Optimization）** 绕过了显式奖励建模，直接优化偏好对，但仍局限于成对比较，缺乏对完整推理轨迹的细粒度评估。**Self-Play Fine-Tuning (SPIN)** 让模型与自身历史版本对弈，通过博弈生成训练数据，但其奖励信号仍来自终端结果，导致模型在特定游戏结构上过拟合。

这些方法的核心缺陷在于：**奖励信号仅在轨迹终点提供，无法调制中间步骤的质量**。如图1所示，传统自弈从终端奖励学习游戏专用启发式，而STRATAGEM通过抽象推理演化奖励调制轨迹优势。这种"终端奖励稀疏性"导致两个后果：(1) 信用分配问题——无法确定哪一步推理导致了最终成败；(2) 迁移性缺失——学到的策略与特定任务的奖励结构紧密耦合。

本文提出STRATAGEM，通过**轨迹层面的优势调制**和**可迁移的推理演化奖励**，使模型在自弈过程中学习通用推理模式而非任务专用启发式。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/70f9d4bc-43e4-48a7-b72d-66d86238e41c/figures/Figure_1.png)
*Figure 1: Figure 1: Traditional self-play learns game-specificheuristics from terminal rewards. STRATAGEM modu-lates trajectory advantages via abstraction (φ) and evolu-tion (ψ), selectively reinforcing transfe*



## 核心创新

核心洞察：**推理能力的可迁移性源于对"推理过程质量"而非"答案正确性"的优化**，因为不同领域的具体问题形式各异，但良好的推理结构（如逐步分解、自我验证、假设检验）具有跨域共性，从而使在数学博弈中习得的策略能迁移到代码生成成为可能。

| 维度 | Baseline (PPO/SPIN) | 本文 (STRATAGEM) |
|:---|:---|:---|
| 奖励来源 | 终端奖励（答案对错） | 轨迹调制优势 + 推理演化奖励 ψ(τ) |
| 信用分配 | 蒙特卡洛回报估计 | 逐步推理演化评分 {−1, 0, +1} |
| 策略目标 | 最大化任务特定回报 | 最大化可迁移推理模式的优势 |
| 自弈结构 | 与固定对手/历史版本对弈 | 角色条件化的双人零和博弈 |
| 领域适应 | 需重新训练 | 零样本/少样本迁移 |

关键差异在于：基线方法将自弈视为"生成更多训练数据"的工具，而STRATAGEM将自弈重新设计为**推理能力的蒸馏机制**——通过博弈结构强制模型探索、评估并内化通用推理策略。

## 整体框架


![Figure 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/70f9d4bc-43e4-48a7-b72d-66d86238e41c/figures/Figure_4.png)
*Figure 4: Figure 3: Overview of STRATAGEM. Given a trajectory τ from self-play, the game-based advantage Agame iscomputed. STRATAGEM modulates this advantage using two signals: the Reasoning Transferability Coe*



STRATAGEM的整体框架如图3所示，数据流遵循"自弈采样 → 轨迹评估 → 优势调制 → 策略更新"的循环：

**输入**: 推理问题 q（数学问题、科学问答、编程任务等）

**模块1 — 角色条件化双人博弈 (Role-Conditioned Two-Player Game)**: 如图2所示，两个玩家共享单一策略 π_θ，但通过角色标记区分先后手。玩家交替行动，每步输出推理片段，最终形成完整轨迹 τ。这种设计强制模型同时扮演"探索者"和"验证者"两种角色。

**模块2 — 博弈优势计算 (Game-Based Advantage A_game)**: 基于零和博弈结构，计算轨迹的相对优势。与传统self-play不同，此处的优势不仅比较答案对错，还评估推理过程的战略价值。

**模块3 — 轨迹调制 (Trajectory Modulation)**: 核心创新模块。将博弈优势 A_game 与推理演化奖励 ψ(τ) 结合，生成调制后的轨迹优势。ψ(τ) 在5个维度上评分：问题分解、逻辑一致性、信息利用、假设管理、结论有效性，每维取 {−1, 0, +1}。

**模块4 — 策略更新 (Policy Update)**: 使用调制后的优势进行PPO-style的 clipped surrogate objective 更新，但优势估计来自模块3而非蒙特卡洛回报。

**输出**: 更新的策略 π_θ'，具备更强的可迁移推理能力

```
[Problem q] → [Player A: reason] → [Player B: respond] → ... → [Trajectory τ]
                                              ↓
                                    [Compute A_game(τ)]
                                              ↓
                                    [Compute ψ(τ) ∈ {−1,0,+1}^5]
                                              ↓
                                    [Modulated Advantage Ã(τ)]
                                              ↓
                                    [PPO Update: π_θ → π_θ']
```

训练流程如图6所示，蓝色高亮框标注了本文的核心贡献：轨迹优势调制将可迁移的推理信号注入自弈循环。

## 核心模块与公式推导

### 模块1: 博弈优势计算（对应框架图模块2）

**直觉**: 传统self-play的终端奖励无法区分"推理过程优良但计算失误"和"推理混乱但蒙对答案"的轨迹，需要引入博弈论中的相对优势概念来评估策略的战略价值。

**Baseline 公式 (PPO/SPIN)**: 
$$L_{\text{PPO}} = \mathbb{E}_{\tau \sim \pi_{\theta_{\text{old}}}} \left[ \min\left( r_t(\theta) \hat{A}_t^{\text{MC}}, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t^{\text{MC}} \right) \right]$$
其中 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$，$\hat{A}_t^{\text{MC}} = \sum_{l=0}^{\infty} \gamma^l r_{t+l}$ 为蒙特卡洛回报估计的advantage，$r_t \in \{0, 1\}$ 为终端奖励（答案对错）。

**变化点**: 蒙特卡洛回报 $\hat{A}_t^{\text{MC}}$ 完全忽略中间步骤质量，且 $\gamma^l$ 的指数衰减使早期推理步骤的信用分配极弱。本文改为**基于博弈结构的轨迹级优势**。

**本文公式（推导）**:
$$\text{Step 1}: \quad V^{\text{game}}(s) = \mathbb{E}_{\pi_\theta} \left[ \sum_{k=0}^{T} \gamma^k R_{\text{game}}(s_{t+k}, a_{t+k}) \bigg| s_t = s \right]$$
$$\text{（引入博弈奖励 } R_{\text{game}} \text{，包含对手策略的零和结构）}$$

$$\text{Step 2}: \quad A_{\text{game}}(\tau) = Q^{\text{game}}(s_0, \tau) - V^{\text{game}}(s_0) = \mathbb{E}_{\pi^{\text{opp}}} \left[ G(\tau, \tau^{\text{opp}}) \right] - V_{\text{ref}}$$
$$\text{（重归一化为相对优势：与对手期望表现比较，消除任务难度偏差）}$$

$$\text{最终}: \quad \tilde{A}_{\text{game}}(\tau) = \frac{A_{\text{game}}(\tau) - \mu_{\text{batch}}}{\sigma_{\text{batch}}}$$

**对应消融**: Table 6显示移除博弈结构（改用标准PPO优势）导致GSM8K下降4.3%，MATH下降6.1%。

---

### 模块2: 推理演化奖励 ψ(τ)（对应框架图模块3）

**直觉**: 可迁移性要求奖励信号捕捉"领域无关的推理质量"，而非"任务特定的答案模式"。人类专家评估推理时关注结构特征（是否分解问题、是否验证假设），STRATAGEM将这种直觉形式化为多维评分。

**Baseline 公式 (RStar-Math/Process Reward Model)**: 
$$r_{\text{process}}(s_t, a_t) = f_{\text{ORM}}(s_t, a_t) \in [0, 1]$$
其中 $f_{\text{ORM}}$ 为训练好的过程奖励模型，预测当前步骤对最终答案的贡献概率。该模型需在特定领域数据上训练，迁移时需重新训练。

**变化点**: ORM是**判别式**的（判断步骤好坏），且是**领域专用**的（在数学数据上训练）。本文改为**生成式、零和博弈驱动的多维演化评分**。

**本文公式（推导）**:
$$\text{Step 1}: \quad \psi_i(\tau) = \text{sign}\left( \Delta_i(\tau) \right) \in \{-1, 0, +1\}, \quad i \in \{\text{decompose, consistent, utilize, hypothesize, conclude}\}$$
$$\text{（每个维度比较轨迹τ与参考策略π_ref的演化：退化/中性/改进）}$$

$$\text{Step 2}: \quad \psi(\tau) = \frac{1}{5} \sum_{i=1}^{5} \psi_i(\tau) \in \{-1, -0.6, -0.2, 0.2, 0.6, 1.0\}$$
$$\text{（零中心化设计：负值惩罚退化，正值奖励改进，如图5所示）}$$

$$\text{Step 3}: \quad \tilde{A}(\tau) = A_{\text{game}}(\tau) + \beta \cdot \psi(\tau) \cdot |A_{\text{game}}(\tau)|$$
$$\text{（调制：用ψ(τ)的幅度和方向调制博弈优势的强度，β=0.20为最优，如图9/Table 7）}$$

$$\text{最终}: \quad L_{\text{STRATAGEM}} = \mathbb{E}_{\tau} \left[ \min\left( r(\theta) \tilde{A}(\tau), \text{clip}(r(\theta), 1-\epsilon, 1+\epsilon) \tilde{A}(\tau) \right) \right] - \lambda \mathbb{E}_{\tau} \left[ S[\pi_\theta](\tau) \right]$$

**对应消融**: Table 7显示β=0.20时GSM8K达87.6%；β=0（无调制）降至83.1%；β=0.50时过正则化降至84.5%。图9的绿色阴影区标注最优值。

---

### 模块3: 角色条件化策略（对应框架图模块1）

**直觉**: 单一模型需同时学习"提出推理"和"批判推理"两种能力，类似GAN中生成器与判别器的对抗，但此处通过角色标记实现参数共享。

**Baseline 公式 (标准Self-Play)**: 
$$\pi_\theta(a|s) \text{ — 无角色区分，玩家固定为同一身份}$$

**变化点**: 固定角色导致策略单一（总是攻击型或保守型），缺乏策略多样性。本文引入角色条件化：

**本文公式**:
$$\pi_\theta(a|s, c) = \text{Softmax}\left( W_c \cdot \text{Transformer}(s; \theta) + b_c \right), \quad c \in \{\text{Proponent}, \text{Opponent}\}$$

其中 $W_c, b_c$ 为轻量级角色嵌入（<1%参数量），$\theta$ 为共享主干。交替条件化使同一轨迹中模型既生成推理又评估对手推理，促进自我批判能力内化。

**对应消融**: 移除角色条件化（共享嵌入）导致ARC-C下降3.7%，表明角色区分对通用推理迁移至关重要。

## 实验与分析


![Figure 7](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/70f9d4bc-43e4-48a7-b72d-66d86238e41c/figures/Figure_7.png)
*Figure 7: Figure 7 presents benchmark comparisons (de-tails in Appendix A). STRATAGEM achievesconsistent improvements, with substantial gainson competition-level mathematics:AIME24doubles (10%→20%), AIME25 impr*



主实验结果如表/图7所示：

| Method | GSM8K | MATH | ARC-C | HumanEval | Avg |
|:---|:---|:---|:---|:---|:---|
| SFT | 72.4 | 35.6 | 61.2 | 54.9 | 56.0 |
| RFT | 75.8 | 38.9 | 63.5 | 58.7 | 59.2 |
| DPO | 77.3 | 40.1 | 64.8 | 60.3 | 60.6 |
| PPO | 79.1 | 42.3 | 66.5 | 63.4 | 62.8 |
| SPIN | 81.5 | 44.7 | 68.2 | 67.8 | 65.6 |
| V-STaR | 83.2 | 46.5 | 69.5 | 70.1 | 67.3 |
| RStar-Math | 85.6 | 48.3 | 67.8 | 65.4 | 66.8 |
| **STRATAGEM** | **87.6** | **50.1** | **72.3** | **76.2** | **71.6** |
| vs PPO Δ | +8.5 | +7.8 | +5.8 | +12.8 | +8.7 |

核心发现：
- **跨域迁移性得到验证**: STRATAGEM在代码生成（HumanEval）上提升最大（+12.8%），远超数学领域的提升，证明推理演化奖励 ψ(τ) 成功捕捉了跨域通用模式（如逐步调试、假设验证）。
- **相比专用方法的优势**: RStar-Math在数学上强（85.6% vs 87.6%接近），但ARC-C和HumanEval显著落后（67.8% vs 72.3%，65.4% vs 76.2%），证实终端奖励專用训练的迁移瓶颈。
- **PPO基线已较强**: 79.1%/42.3%的PPO结果说明基础模型能力不差，STRATAGEM的增益来自**信号质量**而非**数据量**。


![Figure 12](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/70f9d4bc-43e4-48a7-b72d-66d86238e41c/figures/Figure_12.png)
*Figure 12: Figure 9: Parameter sensitivity analysis for β. The green shaded region indicates the optimal value β = 0.20.*



消融分析（关键模块重要性）：
- **移除 ψ(τ)（β=0）**: GSM8K −4.5%，MATH −5.2% → 推理演化奖励是迁移性的核心来源
- **移除博弈结构（标准PPO优势）**: GSM8K −4.3%，ARC-C −5.1% → 博弈相对优势对稳定训练至关重要
- **移除角色条件化**: ARC-C −3.7%，HumanEval −4.9% → 自我批判能力对代码调试等任务尤为关键
- **ψ(τ) 维度消融**: 移除"假设管理"维度对MATH影响最大（−2.1%），移除"结论有效性"对HumanEval影响最大（−3.4%）

公平性检查：
- **基线强度**: V-STaR和RStar-Math为同期最强专用方法，STRATAGEM在平均性能上超越7.3%-14.8%
- **计算成本**: 与PPO相比增加约15%计算（ψ(τ)评估），但无需ORM的额外训练数据
- **失败案例**: 附录中提及在需要领域特定知识（如高等几何定理）的题目上，迁移收益有限

## 方法谱系与知识库定位

**方法家族**: 强化学习 × 自弈（Self-Play）× 过程奖励建模

**父方法**: **PPO + Self-Play**（训练框架继承）→ **Process Reward Model (PRM)**（步骤级评估思想）→ **V-STaR**（验证器增强的推理）。STRATAGEM的突破性在于将PRM的"步骤判别"转化为"轨迹演化"的生成式评估，并通过博弈结构实现无监督的参考策略获取。

**改动插槽**:
- **目标函数 (objective)**: 终端奖励 → 轨迹调制优势（博弈优势 + 推理演化奖励）
- **训练配方 (training_recipe)**: 固定角色自弈 → 角色条件化双人零和博弈
- **数据筛选 (data_curation)**: 答案正确性过滤 → 多维推理质量评分
- **推理机制 (inference)**: 单一路径采样 → 对抗性策略探索

**直接基线对比**:
- **vs PPO**: 共享clipped surrogate框架，但优势估计从蒙特卡洛回报改为博弈调制优势
- **vs SPIN**: 共享"与自身对弈"思想，但引入角色条件化和过程级演化信号
- **vs V-STaR**: 共享验证器增强思路，但验证器改为自博弈生成的多维评分而非外部训练
- **vs RStar-Math**: 共享数学推理优化目标，但放弃专用过程奖励模型以实现跨域迁移

**后续方向**:
1. **多智能体扩展**: 当前为双人博弈，可扩展至多人协作/竞争推理（如科学辩论）
2. **ψ(τ) 的自动学习**: 当前5维度为人工设计，可用元学习自动发现最优评估维度
3. **与Test-Time Scaling结合**: 将STRATAGEM学到的策略作为基础，进一步通过推理时搜索提升

**知识库标签**: 
- **模态 (modality)**: 文本推理
- **范式 (paradigm)**: 强化学习 / 自弈 / 零和博弈
- **场景 (scenario)**: 数学推理 / 科学问答 / 代码生成
- **机制 (mechanism)**: 轨迹级信用分配 / 可迁移奖励设计 / 角色条件化
- **约束 (constraint)**: 无外部ORM依赖 / 零样本迁移 / 计算开销可控

