---
title: "AvalonBench: Evaluating LLMs Playing the Game of Avalon"
venue: arXiv
year: 2023
tags:
  - Survey_Benchmark
  - task/llm-agent-evaluation
  - task/social-deduction-game-playing
  - game-based-evaluation
  - react
  - rule-based-agents
  - dataset/AvalonBench
  - opensource/full
core_operator: "把 Avalon 隐藏身份博弈形式化为带公开讨论、私有角色信息和阶段动作的多智能体评测环境，用来测量 LLM 的社交推理—决策闭环。"
primary_logic: |
  多智能体社交推理评测目标 → 构建含规则环境、角色私有信息、公开讨论、递归摘要与规则机器人对手的 Avalon 回合制框架 → 用胜率、刺杀准确率和推理准确率比较 LLM 与基线 → 揭示 LLM 在“能推断身份”与“能稳定执行策略”之间的能力断层
claims:
  - "在善方 SERVANT 设定下，带讨论的 GPT-3.5 胜率仅 22.2%，低于规则 SERVANT 的 38.2% [evidence: comparison]"
  - "在同一 SERVANT 设定下，带讨论的 GPT-3.5 推理准确率达到 76.0%，高于规则基线的 71.8%，但胜率仍更低，说明身份判断未转化为更优决策 [evidence: comparison]"
  - "在刺客 ASSASSIN 设定下，带讨论的 GPT-3.5 总胜率为 66.7%，其中任务阶段胜率为 0.0%、刺杀 Merlin 准确率为 66.7%，收益主要体现在刺杀阶段而非任务策略 [evidence: comparison]"
related_work_position:
  extends: "AgentBench (Liu et al. 2023)"
  competes_with: "N/A"
  complementary_to: "ReAct (Yao et al. 2023)"
evidence_strength: moderate
pdf_ref: "paperPDFs/Agentic_Reasoning_Benchmarks_Core_Mechanisms_of_Agentic_Reasoning_Language_Communication_and_Social_Reasoning/arXiv_2023/2023_AvalonBench_Evaluating_LLMs_Playing_the_Game_of_Avalon.pdf"
category: Survey_Benchmark
---

# AvalonBench: Evaluating LLMs Playing the Game of Avalon

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2310.05036), [Code](https://github.com/jonathanmli/Avalon-LLM)
> - **Summary**: 这篇论文把隐藏身份桌游 Avalon 做成一个多智能体 LLM 评测场，专门测语言讨论、身份推理、协作与欺骗能否真正转化为连续决策能力。
> - **Key Performance**: GPT-3.5 在善方且允许讨论时胜率仅 **22.2%**，低于规则善方 **38.2%**；GPT-3.5 在刺客设定且允许讨论时总胜率 **66.7%**，刺杀 Merlin 准确率 **66.7%**。

> [!info] **Agent Summary**
> - **task_path**: Avalon 游戏状态（规则 + 私有角色信息 + 历史摘要 + 当轮讨论） -> 阶段动作/发言 -> 胜率、刺杀准确率、推理准确率
> - **bottleneck**: 现有评测难以同时测到多智能体博弈中的隐藏状态维护、社交推理、策略性发言与行动闭环
> - **mechanism_delta**: 把 Avalon 变成带讨论、私有信息、阶段化行动和对照基线的统一评测协议，并用有/无讨论拆出语言对策略的真实贡献
> - **evidence_signal**: GPT-3.5 善方推理准确率更高却胜率更低，说明“会猜身份”不等于“会赢游戏”
> - **reusable_ops**: [phase-structured-game-engine, separate-action-parser]
> - **failure_modes**: [identity-leakage-in-discussion, deduction-action-gap]
> - **open_questions**: [how-to-map-beliefs-to-actions, how-to-train-stealthy-role-consistent-dialogue]

## Part I：问题与挑战

这篇论文要解决的，不是“LLM 能不能看懂 Avalon 规则”，而是更难的事：**LLM 能不能在一个部分可观测、带欺骗、带协商、带长期记忆压力的多智能体环境里，持续做对行动**。

Avalon 之所以适合作为评测载体，是因为它把几类常被分开研究的能力绑在了一起：

1. **演绎推理**：根据组队、投票、任务成败、发言内容推断谁是好人/坏人。  
2. **协作与说服**：你不能只知道答案，还得让别人接受你的队伍与判断。  
3. **欺骗与伪装**：坏人要装好人，Merlin 知道真相却不能暴露。  
4. **行动一致性**：说的话、投的票、选的队、做的任务必须跨回合一致，否则会露馅。

作者认为，现有很多 LLM 评测更像静态问答或单步决策，**缺少“语言—推理—行动”闭环的多智能体测试床**。这也是“为什么现在要做”的原因：LLM agent 已经被广泛讨论，但能系统测多智能体社交推理的 benchmark 仍然稀缺。

**输入/输出接口**很明确：

- **输入**：游戏规则、角色私有信息、历史摘要、当轮讨论文本、当前阶段请求  
- **输出**：选队、投票、任务 pass/fail、刺杀对象，以及讨论发言

**边界条件**也很重要：

- 5 人 Avalon
- 只有公开讨论，没有私聊
- 讨论按固定顺序发言，leader 会说两次
- 主实验里为了和朴素基线公平比较，LLM 只喂 **任务结果**，不喂 **投票历史**
- baseline bot 的语言与决策是分离的：会“说话”，但说话不驱动决策

这里真正的瓶颈是：**模型是否能维护角色条件下的隐变量信念，并把这些信念稳定落到策略动作上**，而不只是生成一段看起来像推理的文本。

## Part II：方法与洞察

作者的核心贡献不是提出一个更强的新 agent，而是搭建一个**可控、可复现、可拆解失败模式**的评测框架 AVALONBENCH。

### 评测框架怎么搭

**1. 游戏环境层**  
实现了 Avalon 的四个决策阶段：

- Team Selection
- Voting
- Quest
- Assassination

同时保留公开讨论环节，让模型既要“做动作”，也要“说人话”。

**2. 对照基线层**  
作者设计了若干 rule-based naive bots：

- **Naive Servant**：根据任务失败数排除不可能的身份配置，选择“全好人概率最高”的队伍
- **Naive Minion / Assassin**：优先推动含坏人的队伍，并在任务中破坏
- **Naive Merlin**：只支持无坏人的队伍，但因此也更容易被识别

这类基线很重要，因为它们虽然“笨”，但**策略是显式一致的**。这让 benchmark 能测出：LLM 是真不会，还是只是不会把知道的东西贯彻到底。

**3. LLM agent 层**  
LLM agent 采用 ReAct 风格 + zero-shot CoT 来产生活动文本；但作者额外加了两个很关键的工程组件：

- **separate parser**：再用一个 LLM 把自由文本解析成结构化动作，保证动作合法、可执行
- **recursive summary**：每轮用摘要压缩历史，避免长上下文爆炸

这两个组件的意义不是“提升智力”，而是让 benchmark 能稳定运行，并把错误更多归因到策略/推理，而不是格式失败或上下文溢出。

**4. 评测协议层**  
作者做了几种对照：

- 用 LLM 替换单个 **Assassin**
- 用 LLM 替换单个 **Servant**
- **with discussion / without discussion**
- 所有玩家都由 LLM 驱动的 **multi-LLM self-play**

对应指标包括：

- total winrate
- mission winrate
- assassination winrate / assassination accuracy
- deduction accuracy

### 核心直觉

过去很多评测只问：**模型能不能“说出”一个合理判断**。  
这篇工作把问题改成：**模型能不能在一个有隐身份、有利益冲突、有多人互动的连续博弈里“持续做对”**。

这带来了一个关键测量变化：

- **变化前**：静态文本推理，输出一个答案或解释即可  
- **变化后**：部分可观测 + 多轮更新 + 公开发言 + 策略激励 + 角色约束

于是，评测瓶颈从“是否会解释”变成了：

- 能否维护跨回合信念状态
- 能否把信念变成投票/组队/任务/刺杀策略
- 能否在交流中传递信息而不暴露身份

这就是为什么 AvalonBench 能测出一种很重要的能力断层：  
**LLM 可能已经会“看起来像在推理”，但还不会“像玩家一样赢游戏”。**

### 战略权衡

| 设计选择 | 带来的诊断能力 | 代价 / 偏差 |
|---|---|---|
| 用 Avalon 作为评测环境 | 同时覆盖推理、协作、欺骗、长期一致性 | 领域较窄，外推到一般社会推理需谨慎 |
| 规则机器人作对手 | 可控、可复现、便于公平比较 | 对手较弱，且不真正理解语言 |
| separate parser | 减少动作格式错误，把失败更多归因到策略 | 引入额外 LLM 依赖，评测链条更复杂 |
| recursive summary | 让长程游戏可跑通 | 摘要可能丢失关键信息或引入幻觉 |
| 有/无讨论对照 | 能隔离语言对决策的真实作用 | 讨论协议被简化，和真实人类社交节奏不同 |
| 为公平屏蔽投票历史 | 避免 LLM 比 naive bot 多拿关键信号 | 也削弱了 Avalon 最重要的一类推理依据 |

## Part III：证据与局限

### 关键证据信号

**信号 1：善方 LLM 连朴素规则基线都打不过。**  
在 SERVANT 设定下，规则基线胜率是 **38.2%**，而 GPT-3.5 在带讨论时只有 **22.2%**，不带讨论更只有 **11.1%**。  
**结论**：当前 LLM 在善方侧并不能稳定执行哪怕是很朴素的安全策略。

**信号 2：LLM 能“猜人”，但不会把猜测转成更好的行动。**  
GPT-3.5 在带讨论的 SERVANT 设定下，**deduction accuracy = 76.0%**，甚至高于规则基线的 **71.8%**；但最终胜率仍显著更差。  
**结论**：这里暴露的是典型的 **reasoning-to-policy gap**——身份推断没有变成组队、投票上的优势。

**信号 3：讨论对坏人更有利，像是在“泄露身份”。**  
在 ASSASSIN 设定下，GPT-3.5 带讨论时总胜率 **66.7%**，而 mission winrate 是 **0.0%**，提升几乎全来自 **刺杀 Merlin**。  
**结论**：讨论当前更像给坏人提供了额外的身份线索，而不是帮助善方达成稳健协作。

**信号 4：self-play 明显偏向 Evil。**  
多 LLM 自博弈中，Evil 胜率高达 **83.3%**。  
**结论**：当前 LLM 对话机制并不天然促进集体理性，反而容易放大身份泄露、策略不一致和欺骗方优势。

### 为什么这些证据重要

这篇 paper 最有价值的地方，不是证明“GPT-3.5 很弱”，而是**把弱点定位清楚了**：

- 不是完全不会推理
- 也不是完全不会交流
- 而是 **推理、交流、策略执行三者没有闭环**

Case study 也支持这一点：模型经常直接说出自己是 Evil、Merlin，或复读他人发言，说明它并未真正掌握“带约束的信息传递”。

### 局限性

- **Fails when**: 要评估依赖投票模式精细推断、自由插话、私聊协作、复杂 meta-game 的 agent 时，这个版本会失真；因为它固定发言顺序，且主实验为了公平不把投票历史提供给 LLM。
- **Assumes**: 5 人 Avalon、公开讨论、规则机器人只基于任务结果决策；同时依赖额外的 LLM parser 来保证动作可执行。实验规模也较小（GPT-3.5 30 局、Llama2-7B 10 局），结果方差不可忽视。
- **Not designed for**: 真实人类对战生态、跨游戏泛化结论、或训练型方法（如 self-play 强化）本身的系统比较。

还要特别指出两个复现相关假设：

1. **闭源依赖**：GPT-3.5 不仅被评测，也被拿来做 parser。  
2. **语言-决策解耦**：naive bot 的发言由 LLM 生成，但动作不受发言影响，这让“语言影响决策”的生态仍然不够真实。

### 可复用组件

这篇工作里最值得复用的，不只是 Avalon 这个游戏本身，而是它的评测套路：

- **阶段化多智能体游戏引擎**
- **自由文本到结构动作的分离式 parser**
- **递归式历史摘要记忆**
- **角色条件下的推理准确率 probe**
- **有/无讨论的因果对照协议**

如果以后要做更强的 agentic reasoning benchmark，这几个组件都可以直接迁移。

## Local PDF reference

![[paperPDFs/Agentic_Reasoning_Benchmarks_Core_Mechanisms_of_Agentic_Reasoning_Language_Communication_and_Social_Reasoning/arXiv_2023/2023_AvalonBench_Evaluating_LLMs_Playing_the_Game_of_Avalon.pdf]]