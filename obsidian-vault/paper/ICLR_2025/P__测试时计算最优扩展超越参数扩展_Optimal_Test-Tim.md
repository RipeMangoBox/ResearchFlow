---
title: Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Parameters for Reasoning
type: paper
paper_level: C
venue: ICLR
year: 2025
paper_link: null
aliases:
- 测试时计算最优扩展超越参数扩展
- Optimal Test-Tim
- Optimal Test-Time Compute Scaling via Verifier-Guided Search
acceptance: Oral
method: Optimal Test-Time Compute Scaling via Verifier-Guided Search
followups:
- 多模态模型评判器LLaVA-Cr_LLaVA-Critic
- 多模态模型自训练评估器LLaVA_LLaVA-Critic
---

# Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Parameters for Reasoning

**Topics**: [[T__Math_Reasoning]] | **Method**: [[M__Optimal_Test-Time_Compute_Scaling_via_Verifier-Guided_Search]] | **Datasets**: [[D__GSM8K]], [[D__AIME_2024]] (其他: MATH-500, MATH-500 by difficulty quartile)

| 中文题名 | 测试时计算最优扩展超越参数扩展 |
| 英文题名 | Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Parameters for Reasoning |
| 会议/期刊 | ICLR 2025 (Oral) |
| 链接 | [arXiv](https://arxiv.org/abs/2408.03314) · Code  · Project  |
| 主要任务 | 数学推理（MATH-500, GSM8K, AIME 2024）上的测试时计算扩展 |
| 主要 baseline | Best-of-N sampling, Weighted majority voting, MCTS, LLM self-verification |

> [!abstract] 因为「固定推理策略无法根据问题难度自适应分配计算资源」，作者在「Best-of-N + ORM」基础上改了「引入PRM密集反馈、双策略（搜索/修正）自适应选择与最优计算分配」，在「MATH-500」上取得「78.2% Pass@1，相比Best-of-N提升+19.9，且以相同FLOPs超过4倍参数扩展效果」

- **MATH-500**: 78.2% Pass@1，相比 PaLM 2-L + Best-of-N (N=256) 的 58.3% 提升 **+19.9**
- **计算效率**: 与 GPT-4 greedy (52.9%) 相比，达到相近性能仅需 **14x 更少** 测试时计算
- **参数扩展对比**: 相同有效 FLOPs 下，测试时计算扩展 (78.2%) 超过 4x 参数扩展 (72.5%) **+5.7**

## 背景与动机

当前大语言模型（LLM）的能力提升主要依赖两条路径：一是**预训练阶段扩大模型参数**（如从 7B 增至 70B），二是**推理阶段增加测试时计算**（如采样更多候选答案）。然而，这两条路径的性价比边界尚不清晰——对于需要多步推理的数学问题，盲目增加参数或盲目增加采样数都可能造成计算浪费。

现有方法如何处理这一问题？**Best-of-N (BoN)** 从基础模型均匀采样 N 个答案，再用 Outcome Reward Model (ORM) 选择最优，这是最简单的测试时计算扩展方式，但所有问题共享相同的 N，无法区分"简单题只需少量样本"与"难题需要大量探索"。**Weighted majority voting** 通过加权聚合多个答案提升稳定性，但仍基于固定采样策略，未解决计算分配问题。**Monte Carlo Tree Search (MCTS)** 在推理树上进行探索，但树搜索的开销随问题复杂度指数增长，且缺乏对"何时该搜索、何时该修正"的系统性分析。

这些方法的共同缺陷在于：**将测试时计算视为固定超参数（uniform N），而非可针对单个问题自适应优化的资源**。具体而言，ORM 仅在最终答案处提供稀疏奖励，无法指导中间步骤的修正；所有问题使用相同推理策略，忽视了"简单问题可通过迭代修正快速收敛，困难问题需要并行探索多样路径"的经验事实。本文的核心动机正是将测试时计算扩展**形式化为最优资源分配问题**，并证明在相同计算预算下，**智能选择"搜索 vs. 修正"策略可以超越单纯扩大模型参数**。

## 核心创新

**核心洞察：问题的难度分布决定了最优推理策略的异质性**，因为简单问题存在"局部可修正性"（sequential revision 可逐步改进），而困难问题需要"全局探索性"（parallel search 覆盖多样路径），从而使**基于难度的自适应策略选择**成为测试时计算最优分配的关键杠杆。

| 维度 | Baseline (Best-of-N + ORM) | 本文 |
|:---|:---|:---|
| 奖励信号 | ORM 稀疏终局奖励 | PRM 密集逐步价值估计 |
| 推理策略 | 固定并行采样 (uniform N) | 难度条件化：搜索 (hard) / 修正 (easy) |
| 计算分配 | 每题相同样本数 | 自适应 N*(x) 与迭代次数 K*(x) |
| 策略改进 | 无（单次生成） | 顺序修正模型基于前次尝试条件化改进 |
| 优化目标 | max P(correct \| y ~ π_base) | max_π E[P(correct \| x, π, C(x))] |

## 整体框架



系统输入为数学问题 x，流经五个核心模块：

1. **Problem Difficulty Estimator（问题难度估计器）**：输入问题 x 与基础 LLM，输出难度估计 d(x) ∈ {easy, hard}。该模块通过分析基础模型首次尝试的置信度或 PRM 初始价值判断问题难度，作为后续路由依据。

2. **PRM Value Function（PRM 价值函数）**：输入部分解状态 s_t（第 t 步的中间推理），输出 V(s_t) = 最终正确的期望概率。替代传统 ORM 的稀疏终局判断，提供逐步反馈。

3. **Search Strategy (Parallel)（并行搜索策略）**：输入问题 x、计算预算 N、基础策略 π_base，输出 N 个候选解中 PRM 打分最优者。针对困难问题，通过温度缩放采样增加多样性，PRM 动态剪枝低价值分支。

4. **Revision Strategy (Sequential)（顺序修正策略）**：输入问题 x、前次尝试 y^{k-1}、PRM 价值反馈，输出修订解 y^k ~ π_revision(y|x, y^{k-1}, V(s))。针对简单问题，基于错误尝试条件化生成改进版本，而非从头重采。

5. **Optimal Strategy Selector（最优策略选择器）**：输入难度 d(x) 与总预算 B，输出策略 ∈ {search, revision} 及超参数。实现计算资源的元控制分配。

数据流示意：
```
x → [Difficulty Estimator] --hard--> [Search: N parallel samples] --PRM scoring--> best y
                |
                --easy--> [Revision: y^1 → y^2 → ...] --value-guided stopping--> final y
```

## 核心模块与公式推导

### 模块 1: 最优测试时计算目标（对应框架图 整体优化目标）

**直觉**：将传统"固定策略最大化正确率"扩展为"在预算约束下自适应选择最优策略"。

**Baseline 公式** (Best-of-N with ORM):
$$\max_{y \sim \pi_{\text{base}}(\cdot|x)} \mathbb{P}(\text{correct} | x, y)$$
符号: $y$ = 候选解, $\pi_{\text{base}}$ = 基础模型策略, 优化目标仅涉及单次采样分布。

**变化点**：Baseline 假设策略固定且计算无约束；本文引入问题相关的计算预算 C(x)，并将策略空间扩展为搜索/修正两种模式的自适应选择。

**本文公式（推导）**:
$$\text{Step 1}: \max_{\pi} \mathbb{E}_{x \sim \mathcal{D}} \left[ \mathbb{P}(\text{correct} | x, \pi, C(x)) \right] \quad \text{将目标形式化为预算约束下的期望正确率最大化}$$
$$\text{Step 2}: \max_{\pi \in \{\pi_{\text{search}}, \pi_{\text{revision}}\}} \mathbb{E}[\cdot] \quad \text{策略空间离散化为两种可解释策略，基于实证观察覆盖不同难度区域}$$
$$\text{Step 3}: \sum_{q=1}^{4} p(q) \cdot \max\left\{ \mathbb{P}_{\text{search}}(\text{correct}|q), \mathbb{P}_{\text{revision}}(\text{correct}|q) \right\} \quad \text{按难度分位数分解，揭示逐点最大化优于全局单一策略}$$
$$\text{最终}: C(x) = C_{\text{search}} \cdot \mathbb{1}(d(x) > \tau) + C_{\text{revision}} \cdot \mathbb{1}(d(x) \leq \tau)$$

**对应消融**：去掉 Optimal strategy selection（仅用 search-only）在最简单四分位数上 accuracy -8.5%；仅用 revision-only 在最困难四分位数上 -11.2%。

---

### 模块 2: PRM 价值函数与训练（对应框架图 PRM 模块）

**直觉**：用逐步正确概率替代终局对错判断，为搜索剪枝和修正方向提供密集信号。

**Baseline 公式** (Outcome Reward Model):
$$R(y) = \mathbb{1}(y = y^*)$$
符号: $y^*$ = 正确答案, $R(y)$ ∈ {0,1} 为稀疏终局奖励。

**变化点**：ORM 无法定位错误步骤，导致搜索时无法中途剪枝、修正时无方向指引。PRM 通过蒙特卡洛 rollout 估计每步价值，实现细粒度信用分配。

**本文公式（推导）**:
$$\text{Step 1}: V(s_t) = \mathbb{E}\left[ \mathbb{1}(\text{correct}) \text{mid} s_t \right] \approx \text{PRM}(s_t) \quad \text{用神经网络近似条件期望价值}$$
$$\text{Step 2}: \mathcal{L}_{\text{PRM}} = -\sum_{t=1}^{T} \left[ r_t \log \hat{V}(s_t) + (1-r_t) \log(1-\hat{V}(s_t)) \right] \quad \text{逐步二元交叉熵，r_t 通过蒙特卡洛 rollout 标注}$$
$$\text{最终}: \hat{V}(s_t) \in [0,1] \text{ 作为搜索选择、修正加权和迭代停止的统一信号}$$

**对应消融**：PRM 替换为 ORM 后 MATH-500 accuracy -6.3%。

---

### 模块 3: 顺序修正策略（对应框架图 Revision Branch）

**直觉**：人类解题时会在错误答案基础上修改，而非每次都从头重写；模型应学会"基于反馈改进"。

**Baseline 公式**：无直接 baseline（标准生成模型条件仅为问题 x）。

**变化点**：本文引入条件化生成，将前次错误尝试和 PRM 价值反馈作为额外输入，通过重要性加权引导修正方向。

**本文公式（推导）**:
$$\text{Step 1}: \pi_{\text{revision}}^{(k)}(y^{(k)} | x, y^{(k-1)}) \propto \pi_{\text{base}}(y^{(k)}|x) \quad \text{初始形式：仅基于问题，未利用反馈}$$
$$\text{Step 2}: \pi_{\text{revision}}^{(k)} \propto \pi_{\text{base}}(y^{(k)}|x) \cdot \exp\left( \lambda \cdot V(s^{(k-1)}) \right) \quad \text{加入PRM价值指数加权，高价值状态获得更高采样概率}$$
$$\text{Step 3}: \mathcal{L}_{\text{DPO-revision}} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma\left( \beta \log \frac{\pi_{\text{revision}}(y_w|x, y_l)}{\pi_{\text{ref}}(y_w|x, y_l)} - \beta \log \frac{\pi_{\text{revision}}(y_l|x, y_l)}{\pi_{\text{ref}}(y_l|x, y_l)} \right) \right]$$
$$\text{其中 } y_l = \text{前次错误解}, y_w = \text{对应正确解}, \text{训练模型学会"从错误到正确"的转换}$$

**对应消融**：去掉 revision model fine-tuning（用 base model 做修正），revision chain accuracy after 3 steps -9.7%。

## 实验与分析

本文在三个数学推理基准上评估：MATH-500（竞赛级，500题）、GSM8K（小学数学，8.5K题）、AIME 2024（美国数学邀请赛，高难度）。基础模型为 PaLM 2-L (540B)，PRM 与 revision model 在此基础上训练。



核心结果可概括为：在 MATH-500 上，本文方法达到 **78.2% Pass@1**，相比 Best-of-N (N=256) 的 58.3% 提升 **+19.9**，且仅需 GPT-4 greedy 推理成本的 **1/14** 即超越其 52.9%。更关键的是**计算效率比较**：在相同有效 FLOPs 预算下，4x 测试时计算扩展（本文方法）达到 78.2%，而 4x 参数扩展（从 7B 等效至 28B）仅达 72.5%，测试时计算扩展以 **+5.7** 优势胜出。GSM8K 上达 95.7%（+4.5 over Best-of-N），接近饱和；AIME 2024 上达 43.3%（+16.6 over Best-of-N），验证了对最难问题的有效性。





消融实验揭示了各组件的贡献。策略选择层面：纯搜索在最难四分位数上落后自适应策略 -3.1%，纯修正在最易四分位数上落后 -8.5%，证明"没有免费午餐"。PRM 替换为 ORM 导致 -6.3%，是单一组件最大降幅之一。Revision model 若未经专门微调（直接用 base model），3-step 修正链 accuracy 暴跌 -9.7%，说明"条件化改进"能力并非基础模型固有。Adaptive compute allocation（非均匀步级分配）替换为均匀分配造成 -3.1%，验证了动态调整 beam width 的价值。

公平性检查：本文 baselines 中 PaLM 2-L 并非最强可用模型（GPT-4 更强），且 GPT-4 的"14x 计算"为估算而非实测；缺失与 OpenAI o1-style 方法、AlphaZero-style MCTS、equalized-compute majority voting 的比较；revision model 训练数据构造细节未完全披露，存在潜在数据泄漏风险；difficulty estimator 可能过拟合 MATH 问题结构。整体证据强度 0.75。

## 方法谱系与知识库定位

**方法家族**：Test-Time Compute Scaling / Verifier-Guided Search

**Parent method 1**: Best-of-N sampling with verifiers — 本文将其从"固定 N 的均匀采样"扩展为"自适应策略选择与计算分配"，关键改变 slots：inference_strategy（引入搜索/修正双模式）、objective（预算约束最优分配）、exploration_strategy（PRM 引导非均匀探索）。

**Parent method 2**: Process Reward Model (PRM) methods — 本文沿用 PRM 的密集逐步反馈机制，但将其应用方式从"单纯步骤验证"升级为"搜索剪枝 + 修正方向 + 迭代停止"的统一价值信号，关键改变 slots：inference_strategy、objective。

**Direct baselines 与差异**：
- **Best-of-N**: 固定 N，ORM 终局选择 → 本文自适应 N*(x)，PRM 逐步引导
- **Weighted majority voting**: 答案空间聚合 → 本文策略空间自适应，不依赖投票
- **MCTS**: 树结构探索，UCB 选择 → 本文线性搜索/修正，PRM 价值直接剪枝，避免树扩展开销
- **LLM self-verification**: 模型自身验证 → 本文训练专用 PRM，验证与生成解耦

**Follow-up 方向**：(1) 将难度估计器扩展为连续值而非二值，实现搜索与修正的细粒度混合；(2) 结合 speculative decoding 降低 revision 的推理延迟；(3) 验证框架向代码生成、科学推理等非数学领域的迁移性。

**标签**：modality=text | paradigm=test-time scaling | scenario=mathematical reasoning | mechanism=process reward model + adaptive strategy routing | constraint=compute budget

## 引用网络

### 后续工作（建立在本文之上）

- [[P__多模态模型评判器LLaVA-Cr_LLaVA-Critic]]: Core methodology paper on test-time compute scaling; likely inspires the paper's
- [[P__多模态模型自训练评估器LLaVA_LLaVA-Critic]]: Core algorithmic idea about test-time compute scaling likely inspires the paper'

