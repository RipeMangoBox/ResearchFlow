---
title: "MAgIC: Investigation of Large Language Model Powered Multi-Agent in Cognition, Adaptability, Rationality and Collaboration"
venue: EMNLP
year: 2024
tags:
  - Survey_Benchmark
  - task/llm-agent-evaluation
  - task/multi-agent-reasoning
  - competition-based-evaluation
  - probabilistic-graphical-model
  - game-theory
  - dataset/MAgIC
  - opensource/full
core_operator: "用固定防守方的竞争式多智能体博弈与七项中间行为指标，量化LLM代理的社会推理与协作能力，并用PGM显式建模多方视角来增强决策。"
primary_logic: |
  多智能体能力评测目标 → 在社交推理游戏与博弈论场景中构造固定对手、角色轮换的竞争环境 → 从投票、推断、提案、合作/背叛等中间行为计算7项能力指标与胜率 → 揭示不同LLM的多智能体能力边界，并验证PGM结构化分析可提升表现
claims:
  - "MAgIC在5类多智能体场景中定义了7个显式能力指标，且指标雷达面积与平均胜率呈正相关，说明该评测能诊断而非仅排序模型 [evidence: analysis]"
  - "在MAgIC上，最强模型GPT o1与最弱模型Llama-2-70B之间存在超过3倍的综合能力差距，GPT-4系整体也显著领先开源基线 [evidence: comparison]"
  - "PGM-aware增强使被测模型的综合能力平均提升37%，平均胜率提升6.57%，且多数模型在3-4项能力上达到显著改进 [evidence: comparison]"
related_work_position:
  extends: "AgentBench (Liu et al. 2023)"
  competes_with: "SmartPlay (Wu et al. 2023); How Far Are We on the Decision-Making of LLMs? (Huang et al. 2024)"
  complementary_to: "ReAct (Yao et al. 2022); Reflexion (Shinn et al. 2023)"
evidence_strength: moderate
pdf_ref: paperPDFs/Agentic_Reasoning_Benchmarks_Core_Mechanisms_of_Agentic_Reasoning_Language_Communication_and_Social_Reasoning/EMNLP_2024/2024_MAgIC_Investigation_of_Large_Language_Model_Powered_Multi_Agent_in_Cognition_Adaptability_Rationality_and_Collaboration.pdf
category: Survey_Benchmark
---

# MAgIC: Investigation of Large Language Model Powered Multi-Agent in Cognition, Adaptability, Rationality and Collaboration

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2311.08562), [Code](https://github.com/cathyxl/MAgIC)
> - **Summary**: 论文提出 MAgIC，一个把 LLM 多智能体能力拆解到 5 类博弈场景和 7 个可量化社会能力指标上的竞争式评测框架，并进一步用 PGM 显式建模“我怎么看别人、别人怎么看我”来增强代理决策。
> - **Key Performance**: 最强与最弱模型的综合能力差距超过 3×；PGM-aware 版本平均带来 37% 的综合能力提升和 6.57% 的平均胜率提升。

> [!info] **Agent Summary**
> - **task_path**: 局部视角的多智能体博弈上下文 -> 线索/投票/提案/合作-背叛决策 -> 7维能力评分与总体胜率
> - **bottleneck**: 现有 LLM-agent 评测难以在真实多方互动里区分“判断差、推理差、协作差还是理性不足”
> - **mechanism_delta**: 把评测从静态单体任务改成固定防守方的竞争式对局，并用7个中间行为指标拆解能力；PGM版再先做双跳视角推断后决策
> - **evidence_signal**: 指标雷达面积与胜率正相关，且 PGM 让多数模型在 3-4 项能力上显著变好
> - **reusable_ops**: [fixed-defender-role-swapping, intermediate-capability-scoring, two-hop-perspective-pgm]
> - **failure_modes**: [scenario-coverage-limited, weak-model-pgm-hallucination, arithmetic-errors-in-public-good]
> - **open_questions**: [do-the-7-metrics-transfer-to-real-workflows, how-sensitive-are-rankings-to-the-defender-model]

## Part I：问题与挑战

这篇论文真正要解决的，不是“LLM 会不会做单轮推理”，而是：

**当 LLM 进入多智能体环境后，如何评测它在局部观察、动态博弈、协作/竞争并存条件下的真实能力。**

### 关键问题
现有 LLM-agent 基准大多有两个缺口：

1. **更像单体 agent 评测**：测环境理解、计划、工具使用，但不充分覆盖真实多方交互。
2. **只看最终结果，不看中间社会能力**：赢了不代表判断对、输了也未必是推理差，可能只是协作失败或不够理性。

### 这篇论文抓住的真瓶颈
多智能体系统里有三个核心难点：

- **局部视角**：每个 agent 只掌握自己的角色、历史对话和部分线索，但决策需要接近全局理解。
- **动态依赖**：别人的策略会改变你的最优行动，静态 QA 式评测不够。
- **社会博弈**：合作、欺骗、协调、理性并不是一个单一分数能表达的。

所以作者把瓶颈定义为：  
**缺少一个既能保留真实交互，又能分解社会认知能力的多智能体评测框架。**

### 输入/输出接口
- **输入**：游戏规则、角色设定、局部上下文、历史轮次对话
- **输出**：线索、投票、角色判断、谈判提案、合作/背叛/投资等动作
- **评测输出**：7 项能力分数 + 总体胜率

### 边界条件
这套 benchmark 的范围是明确受限的：

- 纯文本、多轮、3 人博弈为主
- 5 个场景，共 100+ 设定
- challenger 统一挑战固定 defender（GPT-4）
- 温度设为 0，强调可比性而非探索性

这也是它“现在值得做”的原因：LLM 正从单体助手走向 agent society，但评价工具还停留在旧范式。

## Part II：方法与洞察

### 1. 评测框架：竞争式、多场景、角色轮换
作者构建了一个 **competition-based benchmark**：

- 固定一个防守模型（defender，文中用 GPT-4）
- 让不同 challenger LLM 在**相同场景、相同位置设置**下挑战
- 在不同角色上轮换，减少“某一角色天然占优”的偏差

场景分两组：

- **社会推理/身份博弈**：Chameleon、Undercover  
  主要测 cognition / adaptability
- **博弈论场景**：Cost Sharing、Prisoner’s Dilemma、Public Good  
  主要测 collaboration / rationality

### 2. 7 个能力指标：把“会不会赢”拆成“为什么赢”
作者定义了七类能力：

- **Judgment**：最后判断对不对
- **Reasoning**：对他人身份、他人想法的多跳分析是否正确
- **Deception**：能否成功误导别人
- **Self-awareness**：能否识别自己的角色与处境
- **Cooperation**：能否推动共同目标达成
- **Coordination**：能否提出促成一致的有效方案
- **Rationality**：是否做出符合自身收益最大化的动作

这一步很重要：  
它把 benchmark 从“只出榜单”变成“能定位缺陷的诊断工具”。

### 3. PGM-aware Agent：把社会推理显式化
除 benchmark 外，作者还给出一个增强代理：

- 用 PGM 表示多方之间的依赖关系
- 在每步行动前，显式构造多个视角下的 belief state
- 例如玩家 B 会估计：
  - **B1**：B 自己怎么看局势
  - **B2**：站在 A 视角，A 怎么看
  - **B3**：站在 C 视角，C 怎么看

然后把这些结构化分析再输入 LLM 做下一步决策。

这本质上是一个 **ToM-like scaffold**：  
不是让模型直接“猜”，而是先强制它把多方视角展开。

### 核心直觉

原来的评测瓶颈是：**模型的社会推理藏在最终输出里，无法被单独观测。**

这篇论文做的关键改变是：

- **从只看最终胜负**  
  变成  
- **在真实对局中观测中间信念、投票、提案和策略选择**

进一步，PGM 把原本隐式的社会推理显式化：

**局部上下文** → **多视角 belief structuring** → **更稳定的判断/协调/理性决策**

因果链可以概括为：

> 把“互动中的隐式心智推断”显式写出来  
> → 降低了 LLM 在局部观察下的信息瓶颈  
> → 提高了对他人角色、意图和后续动作的可预测性  
> → 最终改善判断、欺骗、协调和理性等能力

### 战略权衡

| 设计选择 | 带来的收益 | 代价/风险 |
|---|---|---|
| 固定 defender + 角色轮换 | 横向可比性更强，减少对手差异干扰 | 排名会受 defender 选择影响 |
| 5 类博弈场景 | 能覆盖认知、适应、协作、理性四个方面 | 生态有效性仍受“游戏化”限制 |
| 7 个中间指标 | 能诊断失败原因，不止给总分 | 指标是人工设计的，未必完全迁移到真实任务 |
| 文本化 PGM 分析 | 显式暴露 ToM 式推断，便于增强弱点 | 弱模型会出现幻觉、自相矛盾或错误算术 |

## Part III：证据与局限

### 关键证据信号

**信号 1：基准确实能拉开模型层级。**  
在该 benchmark 上，GPT o1 / GPT-4 系列显著领先，Llama-2-70B 明显落后；作者还观察到 7 维能力雷达面积与平均胜率正相关，说明这些分解指标并非装饰性指标，而与真实对局结果有关。

**信号 2：PGM 提升不是个别案例，而是跨模型现象。**  
PGM-aware 版本平均带来：

- 综合能力面积约 **+37%**
- 平均胜率 **+6.57%**
- 其中 **Coordination +12.2%**
- **Rationality +13.0%**

而且多数模型在 **3-4 项能力** 上达到显著提升（t-test, p < 0.05）。

**信号 3：案例分析解释了“为什么有提升”。**  
作者展示了 Chameleon / Undercover / Prisoner’s Dilemma 的案例：  
PGM 让模型更容易显式追踪“谁更可疑、别人会怎么想、下一轮该怎么说/怎么背叛”。同时也暴露出一个重要事实：**PGM 的上限仍受底座模型质量限制**。弱模型虽然有结构，但仍会产生结论与解释相反、遗漏已有线索、甚至幻觉。

### 1-2 个最值得记住的指标
- **模型差距**：最强与最弱模型综合能力差距超过 **3×**
- **增强收益**：PGM-aware 平均综合能力提升 **37%**

### 局限性

- **Fails when**: 场景超出当前 5 类博弈、进入更开放的真实协作流程、长时程组织任务或多模态环境时，这 7 个指标未必还能完整刻画能力；弱模型在 PGM 分析中会出现幻觉、解释不一致、算术错误。
- **Assumes**: 评测依赖固定 GPT-4 defender、固定 prompt、温度为 0、3 人文本交互和人工/ChatGPT 构造的题设；大量被测模型是闭源 API，这会影响复现和长期可比性。
- **Not designed for**: 真实产品级 agent workflow、工具调用/网页操作/具身任务、多模态协作，以及“通过训练把 PGM 能力内化进模型参数”这类问题。

### 可复用组件
- **固定防守方 + 角色轮换** 的竞争式评测协议
- **从中间交互行为定义社会能力指标** 的设计范式
- **双跳视角 PGM 提示**，可作为多智能体 ToM scaffold
- 用 **胜率 + 诊断指标** 联合看模型，而不是只看单一榜单

## Local PDF reference
![[paperPDFs/Agentic_Reasoning_Benchmarks_Core_Mechanisms_of_Agentic_Reasoning_Language_Communication_and_Social_Reasoning/EMNLP_2024/2024_MAgIC_Investigation_of_Large_Language_Model_Powered_Multi_Agent_in_Cognition_Adaptability_Rationality_and_Collaboration.pdf]]