---
title: Protocols for Verifying Smooth Strategies in Bandits and Games
type: paper
paper_level: A
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 老虎机与博弈中平滑策略验证协议
- Smooth Strategy
- Smooth Strategy Verification Protocol
- Verification of ε-optimal smooth st
acceptance: Poster
method: Smooth Strategy Verification Protocol
modalities:
- tabular
paradigm: theoretical analysis / proof-based
---

# Protocols for Verifying Smooth Strategies in Bandits and Games

**Topics**: [[T__Reinforcement_Learning]] | **Method**: [[M__Smooth_Strategy_Verification_Protocol]]

> [!tip] 核心洞察
> Verification of ε-optimal smooth strategies in bandits and approximate strong smooth Nash equilibria in games is possible with provably fewer queries than learning, specifically sublinear in the number of actions for sufficiently smooth strategies.

| 中文题名 | 老虎机与博弈中平滑策略验证协议 |
| 英文题名 | Protocols for Verifying Smooth Strategies in Bandits and Games |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2507.10567) · [Code](待作者发布) · [Project](待作者发布) |
| 主要任务 | 多臂老虎机策略验证、纳什均衡验证 |
| 主要 baseline | Learning-based approach (explore-then-verify)、Naive uniform sampling verification、Smooth Nash equilibria: Algorithms and complexity [14] |

> [!abstract] 因为「学习最优策略需要线性于动作数量的探索代价」，作者在「Smooth Nash equilibria [14]」基础上改了「从学习到验证的问题设定，引入平滑度引导采样实现次线性查询验证」，在「多臂老虎机与标准型博弈」上取得「查询复杂度从 Ω(K) 降至 O(S(π)·log(1/δ)/ε²)」

- 关键性能：多臂老虎机平滑策略验证查询复杂度上界为 O(S(π)·log(1/δ)/ε²)，相比学习下界 Ω(K/ε²) 实现次线性改进
- 关键性能：博弈中 ε-强平滑纳什均衡验证复杂度为 O(S(π)·poly(n)·log(1/δ)/ε²)，避免对动作组合空间 K^n 的指数依赖
- 关键性能：上下界差距为常数因子，证明协议近乎信息论最优（nearly-tight）

## 背景与动机

在强化学习与博弈论中，一个核心问题是：给定一个策略，如何判断它是否足够好？传统路径是"先学习、后验证"——智能体必须充分探索环境、估计所有动作的期望收益，才能确认最优性。例如在多臂老虎机中，要验证某策略是否 ε-最优，标准方法需要 Ω(K) 次臂拉动（K 为动作数量），当动作空间巨大时代价高昂。

现有方法如何处理这一问题？**Explore-then-verify** 策略 [15] 先通过 UCB、Thompson Sampling 等自适应探索算法学习最优策略，再进行验证，但探索阶段本身就需要线性于 K 的样本。**Naive uniform sampling** 对每个动作均匀采样估计收益，同样无法突破线性复杂度。**Smooth Nash equilibria: Algorithms and complexity [14]** 提出了平滑纳什均衡的计算框架，将"平滑性"（策略不过度集中于单一动作）引入博弈分析，但仍聚焦于"计算"均衡而非"验证"给定策略。

这些方法的共同短板在于：**它们都将验证问题归约为学习问题**，必须先获得全局最优策略的知识才能确认局部策略的质量。然而在实际场景中（如审计已部署的推荐策略、验证第三方提交的博弈策略），决策者往往已经持有一个候选策略，只想快速确认其近似最优性，而非从零开始学习。更关键的是，若该策略具有平滑性——即概率分布分散于多个动作而非单点集中——则理论上应可利用该结构避免穷举所有动作。

本文的核心动机正是：**验证是否比学习更容易？** 作者提出一个全新问题设定——直接验证平滑策略的 ε-最优性，目标是以次线性于动作数量的查询完成验证，并通过归约将结果扩展至多智能体博弈的均衡验证。

## 核心创新

核心洞察：**平滑性结构可作为"验证凭证"**，因为平滑策略的概率质量分散于多个动作，使得少量针对性采样即可高概率估计其期望收益，从而将验证复杂度从动作数量 K 降至平滑度参数 S(π)，实现无需完整学习的次线性验证。

| 维度 | Baseline (学习/计算方法) | 本文 |
|:---|:---|:---|
| **问题设定** | 学习最优策略或计算均衡 | 验证给定策略的 ε-最优性 |
| **查询策略** | 自适应探索（UCB/Thompson Sampling）或均匀采样 | 平滑度引导采样（Smoothness-guided sampling） |
| **复杂度依赖** | 线性于动作数 K 或指数于智能体数 n | 线性于平滑度 S(π) 与多项式于 n |
| **核心机制** | 不确定性驱动的全面探索 | 利用平滑结构进行针对性信用验证 |
| **扩展方式** | 独立处理博弈均衡计算 | Bandit-to-game 归约，将均衡验证归约为老虎机实例 |

与 [14] 的关键差异在于：[14] 研究如何"计算"平滑纳什均衡，本文研究如何"验证"平滑策略最优性；前者仍需遍历策略空间，后者通过采样协议直接审计给定策略，查询复杂度实现数量级下降。

## 整体框架

本文框架包含两个核心层级：多臂老虎机验证协议（基础层）与博弈均衡验证扩展（应用层），通过 Bandit-to-game 归约连接。

**多臂老虎机验证流程**：
1. **输入模块**：接收待验证的平滑策略 π 及其平滑度参数 S(π)，以及容忍度 ε、置信度 δ
2. **平滑度引导采样模块（Smoothness-guided sampler）**：根据 π 的平滑结构，非均匀地选择关键动作子集进行查询，而非遍历全部 K 个动作；利用平滑性保证未采样动作的期望收益可被边界约束
3. **效用预言机（Utility oracle）**：外部黑盒，对采样动作返回随机奖励样本
4. **验证决策模块（Verification test）**：基于集中不等式（Hoeffding/Chernoff）构建统计检验，判断 π 的期望收益与最优策略差距是否 ≤ ε
5. **输出模块**：返回 Accept（ε-最优）或 Reject（非 ε-最优）

**博弈验证扩展流程**：
1. **输入模块**：n 人标准型博弈、策略组合 σ = (σ₁,...,σₙ)、平滑度参数
2. **Bandit-to-game 归约模块**：将"无有利偏离"（no profitable deviation）的均衡条件，归约为 n 个独立的老虎机验证实例——每个实例检验某智能体单方面偏离是否获益
3. **复用基础协议**：对每个归约实例调用老虎机验证协议
4. **联合决策**：当且仅当所有实例均 Accept 时，确认 σ 为 ε-强平滑纳什均衡

```
Bandit Verification:        Game Verification Extension:
┌─────────────┐           ┌─────────────────┐
│ Smooth π, ε │           │ Game G, σ, ε    │
└──────┬──────┘           └────────┬────────┘
       ▼                          ▼
┌─────────────────┐      ┌─────────────────┐
│ Smoothness-guided│      │ Bandit-to-game  │
│ Sampling         │      │ Reduction       │
└────────┬────────┘      └────────┬────────┘
         ▼                        ▼
┌─────────────────┐      ┌─────────────────┐
│ Utility Oracle  │      │ n Bandit        │
│ (reward samples)│      │ Instances       │
└────────┬────────┘      └────────┬────────┘
         ▼                        ▼
┌─────────────────┐      ┌─────────────────┐
│ Statistical Test│      │ Bandit Protocol │
│ (Hoeffding bound)│     │ (× n times)     │
└────────┬────────┘      └────────┬────────┘
         ▼                        ▼
    Accept/Reject          Joint Accept/Reject
```

## 核心模块与公式推导

### 模块 1：平滑度引导采样的查询复杂度上界（对应框架图"Smoothness-guided sampler → Verification test"）

**直觉**：平滑策略的概率质量分散，使得关键动作的期望收益足以代表整体表现，无需穷举所有动作。

**Baseline 公式**（学习下界 [9][10]）：学习最优策略的样本复杂度为
$$\Omega\left(\frac{K}{\varepsilon^2}\right)$$
符号：$K$ = 动作数量，$\varepsilon$ = 最优性容忍度。该下界表明任何学习算法必须至少线性探索所有动作。

**变化点**：学习需要识别"哪个动作最优"，故必须区分所有动作的收益；验证只需确认"给定策略是否足够好"，且平滑策略的收益由多个动作共同决定，可利用平滑度 $S(\pi)$ 替代 $K$。

**本文公式（推导）**：
$$\text{Step 1}: \quad \hat{\mu}(\pi) = \sum_{a \in \text{Sample}(\pi)} \pi(a) \cdot \hat{\mu}_a \quad \text{（对采样动作加权估计策略期望收益）}$$
$$\text{Step 2}: \quad |\hat{\mu}_a - \mu_a| \leq \sqrt{\frac{\log(2/\delta)}{2N_a}} \quad \text{（Hoeffding集中：} N_a \text{为动作} a \text{的采样次数）}$$
$$\text{Step 3}: \quad N_a \propto \frac{\pi(a) \cdot S(\pi) \cdot \log(1/\delta)}{\varepsilon^2} \quad \text{（按平滑度分配采样预算，高概率质量动作多采样）}$$
$$\text{最终}: \quad T_{\text{query}} = O\left(\frac{S(\pi) \cdot \log(1/\delta)}{\varepsilon^2}\right)$$
其中 $S(\pi) = 1/\min_a \pi(a)$ 为平滑度参数（最大逆概率），$\delta$ 为失败概率。当 $\pi$ 接近均匀分布时 $S(\pi) = O(K)$，退化为最坏情况；当 $\pi$ 高度平滑时 $S(\pi) \ll K$，实现次线性。

### 模块 2：下界构造——硬币偏差归约（对应框架图"Lower bound"）

**直觉**：证明上述上界不可改进，需构造困难实例将硬币偏差检测问题嵌入平滑策略验证。

**Baseline 公式**：无直接 baseline；信息论下界通常通过 Le Cam 方法或 KL 散度论证。

**变化点**：本文创新性地将"区分有偏硬币"归约为"验证平滑策略最优性"，利用平滑策略的收益估计误差与硬币偏差检测的等价性。

**本文公式（推导）**：
$$\text{Step 1}: \quad \text{构造} \quad \mu_a^{(0)} = \frac{1}{2}, \quad \mu_a^{(1)} = \frac{1}{2} + \varepsilon \quad \text{（两族老虎机实例，对应公平/有偏硬币）}$$
$$\text{Step 2}: \quad \pi^* \text{ 在实例 } i \in \{0,1\} \text{ 下的最优性差距恰好为 } \varepsilon \text{ 的倍数}$$
$$\text{Step 3}: \quad D_{KL}(P^{(0)} || P^{(1)}) \leq O(\varepsilon^2 \cdot T_{\text{query}} / S(\pi)) \quad \text{（KL散度受平滑度调节的查询限制）}$$
$$\text{最终}: \quad T_{\text{query}} = \Omega\left(\frac{S(\pi) \cdot \log(1/\delta)}{\varepsilon^2}\right)$$
该下界与模块 1 的上界仅差常数因子，证明协议近乎紧（nearly-tight）。

### 模块 3：博弈归约与均衡验证（对应框架图"Bandit-to-game reduction"）

**直觉**：纳什均衡要求"无任何智能体可单方面偏离获益"，可将每个智能体的偏离检验独立为老虎机验证问题。

**Baseline 公式**：穷举搜索所有动作组合需要
$$\Omega(K^n) \quad \text{（指数于智能体数量）}$$

**变化点**：利用"强平滑纳什均衡"定义（要求对所有平滑偏离无利可图），将联合验证分解为 n 个独立的平滑策略验证实例，每个实例仅涉及单个智能体的动作空间 K 而非 K^n。

**本文公式（推导）**：
$$\text{Step 1}: \quad u_i(\sigma_i', \sigma_{-i}) - u_i(\sigma_i, \sigma_{-i}) \leq \varepsilon, \quad \forall \sigma_i' \in \mathcal{B}_{\text{smooth}} \quad \text{（强平滑均衡条件）}$$
$$\text{Step 2}: \quad \text{对每个智能体 } i, \text{ 构造老虎机实例：动作 = 自身动作，奖励 = 给定 } \sigma_{-i} \text{ 下的条件期望收益}$$
$$\text{Step 3}: \quad \text{验证 } \sigma_i \text{ 在该老虎机中是否 } \varepsilon\text{-最优（调用模块 1 协议）}$$
$$\text{最终}: \quad T_{\text{game}} = O\left(\frac{S(\pi) \cdot n \cdot \log(n/\delta)}{\varepsilon^2}\right) = O\left(\frac{S(\pi) \cdot \text{poly}(n) \cdot \log(1/\delta)}{\varepsilon^2}\right)$$
关键：复杂度仅多项式于 n，而非指数于 n；这是对 [11][13] 中纳什均衡计算复杂性结果的根本突破——验证比计算更容易。

## 实验与分析

本文为一篇纯理论论文，未提供数值实验、模拟结果或可视化图表。所有"实验"证据以定理证明与复杂度分析形式呈现。

**主要理论结果**：作者在标准的多臂老虎机模型与标准型博弈模型上建立了完整的查询复杂度理论。核心 headline number 为：对于平滑度参数 $S(\pi)$ 的策略，验证其 ε-最优性仅需 $O(S(\pi) \cdot \log(1/\delta)/\varepsilon^2)$ 次查询，而学习最优策略的下界为 $\Omega(K/\varepsilon^2)$。当 $S(\pi) = o(K)$ 时（即策略充分平滑），验证严格次线性于学习。在博弈设定中，验证 ε-强平滑纳什均衡的复杂度为 $O(S(\pi) \cdot \text{poly}(n) \cdot \log(1/\delta)/\varepsilon^2)$，相比动作组合空间的指数规模 $K^n$ 实现指数级改进。

**消融与对比分析**：由于无实证实验，作者通过上下界配对进行"理论消融"：
- 上界（模块 1 协议）与下界（模块 2 归约）匹配至常数因子，证明平滑度引导采样的必要性——任何移除平滑度利用的协议（如 naive uniform sampling）必回退到 $\Omega(K)$ 复杂度
- 与 explore-then-verify baseline 对比：学习阶段不可省略，因为验证下界已严格低于学习下界，"先学习后验证"路径在查询效率上不可最优
- 博弈归约（模块 3）的"消融"体现为：若直接检验所有动作组合需 $K^n$ 查询，而通过 bandit-to-game 归约降至 $\text{poly}(n)$，差距为指数级

**公平性检验**：
- **Baseline 强度**：主要理论对比对象为学习下界 [9][10] 与计算复杂性结果 [11][13]，这些是该领域公认强 baseline。但缺乏与 [14] 的直接复杂度比较（[14] 聚焦计算而非验证）
- **缺失验证**：无实证模拟验证理论预测；未测试平滑策略在实际应用中的出现频率；未与 instance-optimal best arm identification [10] 进行实例级对比
- **假设限制性**：结果仅适用于"充分平滑"策略，确定性策略（$S(\pi) = \infty$）被排除；需要效用预言机访问，在部分实际场景（如人类反馈、物理系统）中可能不可用
- **下界间隙**：作者明确标注"nearly-tight"，暗示常数因子间隙存在，但未精确量化

## 方法谱系与知识库定位

**方法族**：平滑策略分析 → 验证协议设计（理论计算机科学/学习理论交叉）

**父方法**：Smooth Nash equilibria: Algorithms and complexity [14]（ITCS 2024）
- [14] 首次将"平滑性"引入纳什均衡算法分析，本文将其从"计算"扩展到"验证"，并引入次线性查询协议

**直接 Baseline 与差异**：
- **Learning-based approach (explore-then-verify)**：标准路径，本文证明其查询冗余，提出绕过学习的直接验证
- **Naive uniform sampling verification**：无结构利用的基准，本文以平滑度引导采样替代均匀采样
- **Nash equilibrium computation algorithms [11][13]**：聚焦 PPAD-完全性等计算障碍，本文证明*验证*可在多项式查询内完成，与*计算*的指数障碍形成鲜明分离
- **Interactive proofs for verifying learning [5][6]**：相近的"验证学习"主题，但聚焦量子学习场景；本文针对经典老虎机与博弈，且利用平滑结构而非密码学工具

**后续方向**：
1. **实例最优性**：将当前实例无关的复杂度结果改进为 instance-optimal，匹配 [10] 的实例敏感分析
2. **部分平滑/自适应平滑**：扩展至策略仅局部平滑或平滑度未知的自适应验证协议
3. **去预言机化**：在无比特精确效用预言机的场景（如偏好反馈、成对比较）中发展验证理论

**知识库标签**：
- **模态**：tabular（表格型状态/动作）
- **范式**：theoretical analysis / proof-based（纯理论证明驱动）
- **场景**：strategy auditing, equilibrium verification（策略审计、均衡验证）
- **机制**：smoothness-structured sampling, problem reduction（平滑结构采样、问题归约）
- **约束**：oracle access required, smooth strategy assumption（需要预言机、平滑策略假设）
