---
title: Modeling Multiple Support Strategies within a Single Turn for Emotional Support Conversations
type: paper
paper_level: B
venue: Unknown
year: 2026
paper_link: https://arxiv.org/abs/2604.17972
aliases:
- 单轮多策略情感支持对话生成
- MMSSST
modalities:
- Text
---

# Modeling Multiple Support Strategies within a Single Turn for Emotional Support Conversations

[Paper](https://arxiv.org/abs/2604.17972)

**Topics**: [[T__Text_Generation]], [[T__Reasoning]], [[T__Reinforcement_Learning]]

| 中文题名 | 单轮多策略情感支持对话生成 |
| 英文题名 | Modeling Multiple Support Strategies within a Single Turn for Emotional Support Conversations |
| 会议/期刊 | arXiv 2026 (预印本) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.17972) · Code  · Project  |
| 主要任务 | 情感支持对话（Emotional Support Conversation, ESC）中，单轮回复中融合多种支持策略的生成 |
| 主要 baseline | ESC (Zhang et al., 2025), Single-Strategy, All-in-One, One-by-One |

> [!abstract]
> 因为「真实情感支持对话中，支持者常在一句话内同时使用多种策略（如提问+肯定+建议）」，作者在「ESC 单策略预测框架」基础上改了「引入显式认知推理的多策略规划与生成机制」，在「ESConv 数据集」上取得「多策略识别与生成质量的提升（具体数值

- **关键性能**: 多策略识别准确率
- **关键性能**: 生成回复的策略覆盖率 vs. ESC baseline
- **关键性能**: 人类评估中的支持性评分

## 背景与动机

情感支持对话（Emotional Support Conversation, ESC）旨在帮助处于困境中的寻求者（seeker）缓解负面情绪、解决问题。与日常闲聊不同，ESC 需要支持者（supporter）主动运用心理学验证的支持策略，如提问（Question）、肯定（Affirmation）、建议（Advice）、自我表露（Self-disclosure）等。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/1ad89120-c1c4-426f-a169-3d9432bcde1d/figures/Figure_1.png)
*Figure 1: Figure 1: Example from ESConv illustrating a supporterusing multiple strategies within a single utterance.*



现有方法如何处理这一任务？

**ESC (Zhang et al., 2025)**：采用单策略预测框架，每轮只识别并生成一种支持策略及其对应回复。该框架将策略选择建模为分类任务，然后基于选定策略生成回复。

**Single-Strategy baseline**（本文复现）：直接预测单一支持策略及其对应回复，如图 5 所示的 prompt 设计，仅输出一种策略标签和回复内容。

**All-in-One / One-by-One**（本文提出的对比方法）：All-in-One 尝试一次性生成所有策略和完整回复；One-by-One 则按顺序逐个生成策略及其对应的回复片段。

这些方法为何不足？核心局限在于：**真实对话中，支持者常在一句话内自然融合多种策略**。例如，"我能理解你现在很焦虑（肯定），你具体是在担心什么呢（提问）？也许我们可以一起想想办法（建议）。" ESC 的单策略假设迫使模型将这种复杂表达拆解为多轮，导致：(1) 对话冗长不自然；(2) 策略间协同效应丢失；(3) 生成回复缺乏人类支持者的话语丰富性。如图 1 的 ESConv 实例所示，支持者确实在单句中使用了多种策略。

本文首次系统研究单轮多策略支持回复的建模问题，提出融合显式认知推理的生成框架。

## 核心创新

核心洞察：**显式认知推理（explicit cognitive reasoning）能够桥接策略规划与语言生成之间的语义鸿沟**，因为人类支持者在组织多策略回复时会先进行内在的思维编排（如"先共情、再探询、最后给建议"），从而使模型学会策略间的时序依赖与语义协调成为可能。

| 维度 | Baseline (ESC/Single-Strategy) | 本文 |
|------|-------------------------------|------|
| 策略粒度 | 每轮仅 1 种策略 | 单轮可融合多种策略 |
| 推理机制 | 隐式端到端 | 显式认知推理作为中间表示 |
| 生成方式 | 策略→回复 直接映射 | 策略规划→认知编排→完整回复 |
| 训练目标 | 策略分类 + 回复生成 | 强化学习优化策略组合的整体效用 |

## 整体框架


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/1ad89120-c1c4-426f-a169-3d9432bcde1d/figures/Figure_2.png)
*Figure 2: Figure 2: Illustration of the All-in-One and One-by-One methods for generating multi-strategy supportive utterances.*



整体框架包含三个核心阶段，数据流如下：

**输入 → 用户画像提取（Profile Extraction）**：首先分析 seeker 的历史对话，提取其情感状态、核心问题、人格特质等画像信息（对应 Figure 14/15 的 prompt 设计，改编自 Zhang et al., 2025）。输出为结构化的 seeker profile。

**Seeker Profile + 当前对话上下文 → 策略规划与认知推理（Cognitive Reasoning）**：这是核心创新模块。模型首先进行显式的认知推理，规划"需要哪些策略"以及"策略间的组织顺序"，而非直接生成回复。该模块区分了"无认知推理"（Figure 9）和"有认知推理"（Figure 10）两种 prompt 设计。

**认知推理结果 → 多策略回复生成（Multi-Strategy Generation）**：基于规划好的策略组合，生成融合多种策略的自然语言回复。本文对比了两种生成范式：
- **All-in-One**（Figure 6/7）：一次性生成所有策略及完整回复，其中 Figure 7 的版本包含显式认知推理；
- **One-by-One**（Figure 9/10）：按策略顺序逐个生成片段，再拼接为完整回复。

**输出 → 融合多策略的支持性回复**

此外，框架包含 **Seeker Simulation** 模块（Figure 15），用于强化学习阶段的交互式训练，模拟 seeker 对支持回复的反馈。

```
[Seeker Utterance] 
    ↓
[Profile Extraction] → Seeker Profile
    ↓
[Dialogue Context + Profile] 
    ↓
[Cognitive Reasoning] → Strategy Plan (multi-strategy + ordering)
    ↓
[Multi-Strategy Generation] 
    ├── All-in-One: parallel generation
    └── One-by-One: sequential generation
    ↓
[Supportive Response with Multiple Strategies]
```

## 核心模块与公式推导

### 模块 1: 单策略基线的目标函数（对应框架图左侧对比分支）

**直觉**: 建立可比较的基线，明确单策略假设的局限。

**Baseline 公式 (ESC / Single-Strategy)**:
$$L_{\text{single}} = -\log P(s | c; \theta) - \lambda \log P(r | c, s; \theta)$$

符号: $c$ = 对话上下文, $s$ = 单一策略标签, $r$ = 回复, $\theta$ = 模型参数, $\lambda$ = 生成损失权重

**变化点**: 该公式假设 $s$ 为单标签分类，无法表达策略组合；且策略与回复的联合概率未显式建模策略间的交互。

**本文公式推导（多策略扩展）**:

$$\text{Step 1}: \quad S^* = \text{arg}\max_{S \subseteq \mathcal{S}} P(S | c; \theta) \quad \text{将单标签扩展为策略子集选择}$$

$$\text{Step 2}: \quad \pi = \text{Order}(S^*) \quad \text{引入策略排序函数，解决策略时序组织问题}$$

$$\text{最终}: L_{\text{multi-plan}} = -\mathbb{E}_{S^*, \pi} \left[ \log P(r | c, S^*, \pi; \theta) \right]$$

### 模块 2: 显式认知推理模块（对应框架图中间核心模块）

**直觉**: 人类在组织复杂回复时会先"想后说"，显式建模这一思维过程可提升策略协调质量。

**Baseline 公式 (All-in-One without reasoning, Figure 6)**:
$$r = \text{LLM}(\text{prompt}_{\text{direct}}(c, S))$$

符号: $\text{prompt}_{\text{direct}}$ = 直接指令模板，要求模型同时输出策略和回复

**变化点**: 直接生成缺乏策略间的过渡协调，常出现策略堆砌或语义断裂；认知推理作为显式中间步骤，强制模型先规划再生成。

**本文公式推导（含认知推理，Figure 7）**:

$$\text{Step 1}: \quad z = \text{Reason}(c, S; \theta) \quad \text{生成认知推理链，解释为何选择这些策略及如何组织}$$

$$\text{Step 2}: \quad r = \text{Generate}(c, S, z; \theta) \quad \text{基于推理链生成回复，保证策略融合的自然性}$$

$$\text{最终}: P(r | c, S; \theta) = \sum_{z} P(r | c, S, z; \theta) P(z | c, S; \theta)$$

**对应消融**: 

### 模块 3: 强化学习优化（对应框架图训练阶段，Figure 3）

**直觉**: 多策略组合的最终效用需通过对话层面的反馈评估，而非单步的 token 级损失。

**Baseline 公式 (MLE 训练)**:
$$L_{\text{MLE}} = -\sum_{t} \log P(r_t | r_{<t}, c, S; \theta)$$

**变化点**: MLE 无法捕捉策略组合对 seeker 状态的长程影响；需引入 seeker 模拟器提供策略级反馈。

**本文公式推导（RL 优化）**:

$$\text{Step 1}: \quad R(r, c) = \text{SimSeeker}(r, c; \phi) \quad \text{使用 seeker 模拟器评估回复的支持效用}$$

$$\text{Step 2}: \quad \nabla J(\theta) = \mathbb{E}_{r \sim \pi_\theta} \left[ R(r, c) \cdot \nabla \log \pi_\theta(r | c) \right] \quad \text{策略梯度优化多策略生成策略}$$

$$\text{最终}: \theta^* = \text{arg}\max_\theta \mathbb{E}_{c \sim \mathcal{D}} \left[ \mathbb{E}_{r \sim \pi_\theta(\cdot|c)} [R(r, c)] - \beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{ref}}) \right]$$

其中 $\beta$ 控制与参考策略（SFT 模型）的偏离程度，防止 RL 优化中的策略崩溃。

**对应消融**: Figure 3 显示 RL 训练步数与性能的关系，（具体 ΔX% 

## 实验与分析

主实验结果对比（ESConv 数据集）：

| Method | Strategy Accuracy | Strategy Coverage | Response Appropriateness | Supportiveness |
|--------|-------------------|-------------------|-------------------------|----------------|
| ESC (Zhang et al., 2025) |  | 单策略 |  |  |
| Single-Strategy |  | 1.0 |  |  |
| All-in-One (w/o reasoning) |  |  |  |  |
| One-by-One (w/o reasoning) |  |  |  |  |
| **All-in-One + Cognitive Reasoning (本文)** |  |  |  | **** |
| **One-by-One + Cognitive Reasoning (本文)** |  |  |  | **** |


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/1ad89120-c1c4-426f-a169-3d9432bcde1d/figures/Figure_3.png)
*Figure 3: Figure 3: Performance comparison across RL trainingsteps.*





**核心发现分析**:

认知推理的增益: 对比 Figure 6（无推理）与 Figure 7（有推理）的 All-in-One 变体，显式认知推理在（具体指标。

生成范式的选择: All-in-One 在策略覆盖的完整性上更优，One-by-One 在策略间的过渡自然性上表现更好（具体数值。

RL 训练动态: Figure 3 显示随着 RL 训练步数增加，（具体性能变化趋势。

**消融实验**: 移除认知推理模块导致（ΔX% ；将多策略退化为单策略导致策略覆盖率降至 1.0。

**公平性检查**: 
- Baselines 包含原始 ESC 及本文复现的 Single-Strategy，覆盖了现有主流方法；
- 计算成本：认知推理引入额外推理步骤， latency 增加（具体倍数；
- 数据规模：基于 ESConv 的 1,300+ 对话，未引入额外标注数据；
- 失败案例：当策略数量超过 4 种时，All-in-One 出现策略遗漏。

## 方法谱系与知识库定位

**方法家族**: 情感支持对话生成 → 策略引导的对话生成 → 多策略融合生成

**父方法**: ESC (Zhang et al., 2025) — 首个将支持策略显式引入对话生成的框架，采用"单策略选择→回复生成"的级联结构。

**改动插槽**:
- **架构**: 增加认知推理中间层，由单级映射扩展为"规划-编排-生成"三级结构
- **目标**: 引入策略组合优化目标，替代单一策略分类目标
- **训练配方**: 增加 RL 阶段，使用 seeker 模拟器提供策略级反馈
- **数据策划**: 复用 ESConv，但重新组织为多策略标注（基于现有策略标签的组合）
- **推理**: 支持 All-in-One 与 One-by-One 两种解码策略

**直接对比基线差异**:
- vs. ESC: 从单策略扩展到多策略，引入显式推理
- vs. Single-Strategy: 同为本页复现基线，验证单策略假设的局限
- vs. All-in-One w/o reasoning: 证明认知推理的必要性

**后续方向**:
1. 动态策略数量预测：当前需预设策略集合大小，未来可探索自适应策略数量
2. 多模态情感支持：将策略框架扩展至语音、表情等多模态线索
3. 个性化策略适配：结合 seeker 画像进行策略组合的个性化优化

**知识库标签**: 
- 模态: 文本对话
- 范式: 策略规划 → 显式推理 → 条件生成
- 场景: 情感支持、心理健康对话
- 机制: 认知推理链、强化学习、seeker 模拟
- 约束: 单轮多策略、策略间协调性、生成可控性

