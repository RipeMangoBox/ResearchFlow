---
title: "World Models with Hints of Large Language Models for Goal Achieving"
venue: "NAACL-HLT"
year: 2025
tags:
  - Embodied_AI
  - task/video-understanding
  - reinforcement-learning
  - dataset/HomeGrid
  - dataset/Crafter
  - dataset/MineRL
  - opensource/no
core_operator: 把LLM生成的自然语言子目标对齐到世界模型的想象转移上，并用首达且递减的内在奖励推动目标导向探索
primary_logic: |
  像素观测/任务文本/历史交互 → 观测与转移字幕化、LLM生成子目标并编码为句向量、世界模型在想象轨迹中预测转移并按语义相似度发放首达且递减的内在奖励 → 用外在奖励+内在奖励联合优化策略，提升长时程稀疏奖励任务的探索与达成效率
claims:
  - "DLLM 在论文设定下相对最强基线将 HomeGrid、Crafter、Minecraft 的表现分别提升 27.7%、21.1%、9.9% [evidence: comparison]"
  - "在 Crafter 的 1M 训练步下，DLLM with GPT-4 达到 26.4 分，超过 Achievement Distillation 的 21.8 和 Dynalang 的 16.4 [evidence: comparison]"
  - "Crafter 消融显示：不递减目标奖励、允许同一目标重复奖励、或改用随机目标都会明显降低性能，支持‘首达+递减’内在奖励设计 [evidence: ablation]"
related_work_position:
  extends: "Dynalang (Lin et al. 2023)"
  competes_with: "Dynalang (Lin et al. 2023); ELLM (Du et al. 2023)"
  complementary_to: "N/A"
evidence_strength: strong
pdf_ref: paperPDFs/World_Models_in_the_context_of_Model_Based_RL/Proceedings_of_the_2025_Conference_of_the_Nations_of_the_Americas_Chapter_of_the_Association_for_Computational_Linguisti/2025_World_Models_with_Hints_of_Large_Language_Models_for_Goal_Achieving.pdf
category: Embodied_AI
---

# World Models with Hints of Large Language Models for Goal Achieving

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2406.07381)
> - **Summary**: 这篇工作提出 DLLM，把 LLM 生成的子目标直接注入世界模型的 imagined rollouts 中，让智能体不再只是“找新奇状态”，而是沿着可语言化的中间目标做更有方向的长程探索。
> - **Key Performance**: 相对最强基线，HomeGrid / Crafter / Minecraft 分别提升 27.7% / 21.1% / 9.9%；Crafter 在 1M 步达到 26.4 score。

> [!info] **Agent Summary**
> - **task_path**: 像素观测/任务文本/环境状态字幕 -> LLM子目标 + 世界模型想象 -> 稀疏奖励任务动作策略
> - **bottleneck**: 传统内在奖励只鼓励“新奇”而不区分“是否有助于最终目标”，纯LLM模型自由方法又无法把语言提示稳定传播到长时程规划
> - **mechanism_delta**: DLLM在 imagined rollout 中对预测转移与 LLM 子目标做语义匹配，并仅在首次命中时发放随训练递减的内在奖励
> - **evidence_signal**: 三个环境都超过强基线，且随机目标/不递减/允许重复奖励的消融均显著变差
> - **reusable_ops**: [观测与转移字幕化, 想象轨迹中的目标匹配内在奖励]
> - **failure_modes**: [LLM给出不合理目标会误导探索, 奖励过大或重复奖励会诱发简单技能循环]
> - **open_questions**: [开源LLM能否替代GPT-4而保持效果, 如何在线过滤低可行性或幻觉目标]

## Part I：问题与挑战

这篇论文解决的是**长时程、稀疏奖励、部分可观测**环境中的目标达成问题，典型场景是 HomeGrid、Crafter、Minecraft。  
这类任务的共同难点不是“没有学习信号”，而是**学习信号太晚，而且中间可走的路太多**：智能体往往需要先完成一长串前置行为，最终才拿到外在奖励。

### 真正的瓶颈是什么？
真正的瓶颈是：**探索缺少方向性**。

- 传统 intrinsic reward（新奇性、惊讶度、不确定性）只能告诉 agent“这个状态没见过”；
- 但在大状态/动作空间里，大多数新奇状态对最终目标其实没用；
- 纯 LLM 引导方法虽然能给提示，但通常是**当前步局部提示**，没有进入 world model 的规划闭环，因此很难把语言知识稳定转成长程 credit assignment。

作者的判断很准：  
**问题不在于“要不要用 LLM”，而在于“如何让 LLM 产生的语义提示改变 imagined futures 的价值结构”。**

### 输入/输出接口
- **输入**：像素观测、任务文本/环境文字信息、由 captioner 生成的观测描述与转移描述、以及 LLM 给出的若干子目标。
- **输出**：一个策略网络，最大化外在奖励与语言引导的内在奖励之和。

### 为什么现在值得做？
因为两个条件同时成熟了：

1. **世界模型**已经能在像素环境里做较稳定的 latent imagination；
2. **LLM**已经有足够强的常识和任务分解能力，可以把“拿到铁/钻石”拆成若干人类可理解的中间目标。

所以现在可以把“语言先验”从 prompt 层提升到**规划层**。

### 边界条件
这个方法并不是对所有 RL 任务都天然适用，它更适合：
- 目标能被拆成简短自然语言子目标；
- 环境状态变化能被 captioner reasonably 描述；
- LLM 对环境规则有足够知识，或者 prompt 能补足规则信息。

---

## Part II：方法与洞察

DLLM 可以看成是：**Dynalang 风格多模态世界模型 + LLM 子目标奖励塑形**。

### 方法流程
1. **观测字幕化**：把当前 observation 转成自然语言描述。
2. **LLM 产出子目标**：把当前描述、任务信息和环境机制喂给 GPT，生成固定数量的目标，如“go to kitchen”“collect iron”等。
3. **句向量编码**：用 SentenceBERT 把这些目标编码成 embedding。
4. **世界模型预测转移语义**：world model 不只预测未来 latent state / reward，也预测“转移的语言表示”。
5. **想象轨迹中做目标匹配**：若某一步预测转移与某个目标 embedding 的相似度超过阈值，就把它视为“该目标在 imagined rollout 中被完成”。
6. **内在奖励只给首次命中**：同一 rollout 里同一目标只奖一次，防止 agent 刷简单动作。
7. **目标奖励随训练递减**：作者用 goal-level RND，让经常被达成的目标逐渐不再有高奖励，从而逼着 agent 去追更难、更少见的目标。
8. **策略学习**：actor-critic 在 imagined rollouts 上同时优化外在奖励和这套目标匹配内在奖励。

### 核心直觉

**what changed**：  
从“奖励任何新奇状态”改成“奖励与 LLM 子目标语义一致的转移”。

**which bottleneck changed**：  
这实际上把探索分布从一个巨大的、无结构的状态空间，压缩到一个更小的“**语义上有意义的中间进展集合**”。

**what capability changed**：  
agent 不再只是乱逛，而会更倾向于走向能串起长程任务链条的状态，因此在中高难度目标上更容易形成有效技能链。

更关键的是，作者不是把 LLM 当成在线 planner 直接指挥 agent，而是把它变成**world model 中的价值偏置源**。  
这就是 DLLM 比模型自由的 LLM-RL 更有“长程性”的原因：语言提示进入了 imagination，而不是只停留在当前时刻。

### 为什么这个设计在因果上有效？
- **LLM 子目标**提供“该往哪类中间状态走”的先验；
- **world model imagination**把这个先验传播到未来多步轨迹；
- **首次命中奖励**防止 agent 在一个简单动作上反复薅奖励；
- **RND 递减**防止 agent 永远停留在已掌握的低难度子目标。

所以真正起作用的不是“加了语言”，而是：
**语言把探索变成目标导向，首次命中约束把探索变成前进式，递减机制把前进变成持续式。**

### 战略权衡

| 设计选择 | 改变的瓶颈/约束 | 能力收益 | 代价 |
| --- | --- | --- | --- |
| 用 LLM 生成子目标替代纯新奇奖励 | 从“新奇即可”改成“语义上可能有用” | 更有目的的探索 | 依赖 prompt 与 LLM 先验质量 |
| 在 imagined rollout 中做目标匹配 | 语言信号进入规划闭环 | 长时程 credit 更清晰、样本效率更高 | 依赖 world model 预测准确度 |
| 同一目标只在首次命中时奖励 | 限制 reward farming | 避免刷简单动作，鼓励继续推进 | 可能压低某些确需重复执行动作的收益 |
| 用 RND 递减目标奖励 | 已掌握目标逐渐失去吸引力 | 促进从简单技能走向困难技能 | 增加额外网络与超参敏感性 |

---

## Part III：证据与局限

### 关键实验信号

- **跨环境比较信号**：  
  DLLM 在 HomeGrid、Crafter、Minecraft 都超过强基线，不是单环境偶然成立。最醒目的总结果是相对最强基线提升 **27.7% / 21.1% / 9.9%**。

- **难目标收益信号**：  
  在 Crafter，DLLM 的优势主要体现在 `make stone pickaxe/sword`、`collect iron` 这类需要前置技能链的成就，而不是只在简单收集任务上占优。这直接支持论文的核心主张：**它提升的是长程 goal chaining，而不只是局部 exploration bonus。**

- **指导质量决定性能的信号**：  
  HomeGrid 中，给 LLM 更多任务相关信息时，性能持续提升，且 Full info 与 Oracle 差距很小。说明 DLLM 的上限很大程度受**目标质量**约束。  
  Crafter 中 GPT-4 比 GPT-3.5 产生更高 novelty 的目标，也对应更好的得分，说明“更强 LLM → 更有效探索先验”是可观测的。

- **机制消融信号**：  
  在 Crafter，去掉“递减奖励”、允许“同一目标反复领奖”、或直接换成“随机目标”，性能都明显下降；把 intrinsic reward scale 调太大甚至会灾难性退化。  
  这说明有效的不是“任何语言奖励”，而是**语义匹配 + 首达约束 + 递减机制**这一整套组合。

### 1-2 个最关键指标
- **Crafter 1M**：DLLM (GPT-4) = **26.4**，AD = 21.8，Dynalang = 16.4。  
- **Minecraft 100M**：DLLM (GPT-4) reward = **10.0**，DreamerV3 = 9.1，Dynalang = 8.9。

### 局限性
- Fails when: LLM 目标和真实可达转移错位、captioner 无法准确描述状态变化、或任务需要大量重复步骤而首次命中机制抑制了有益重复时；此外在内在奖励过大时，策略会被 bonus 带偏。
- Assumes: 需要观测 captioner 和转移 captioner；转移 captioner 的训练依赖对 simulator 的修改和规则化语言标签；依赖闭源 GPT-3.5/GPT-4 API、SentenceBERT、cache，以及 A100 级算力。
- Not designed for: 无法自然语言化中间目标的环境、严格连续控制机器人场景、或需要对每个中间目标进行安全可验证约束的真实部署场景。

复现上也有现实门槛：论文报告单实验使用 **1×A100**，Crafter/Minecraft 训练成本约 **10.75 / 16.50 GPU days**，Minecraft 的总 GPT 查询时间还可到 **7.5 天**；文中也未给出明确代码链接，因此可复现性仍受闭源依赖影响。

### 可复用组件
- **观测/转移字幕化接口**：把视觉环境统一成可被语言模型利用的语义接口。
- **imagined rollout 目标匹配奖励**：适合迁移到其他 model-based RL 框架。
- **goal-level RND 递减机制**：适合任何“子目标奖励容易被刷”的层级/探索型 RL 系统。

## Local PDF reference

![[paperPDFs/World_Models_in_the_context_of_Model_Based_RL/Proceedings_of_the_2025_Conference_of_the_Nations_of_the_Americas_Chapter_of_the_Association_for_Computational_Linguisti/2025_World_Models_with_Hints_of_Large_Language_Models_for_Goal_Achieving.pdf]]