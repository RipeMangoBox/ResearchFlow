---
title: "AppWorld: A Controllable World of Apps and People for Benchmarking Interactive Coding Agents"
venue: ACL
year: 2024
tags:
  - Survey_Benchmark
  - task/interactive-coding-agent-evaluation
  - task/tool-use-evaluation
  - state-based-evaluation
  - api-simulator
  - contrast-set
  - dataset/AppWorld
  - opensource/full
core_operator: 通过可控多应用API世界与数据库状态差分单测，按最终世界状态而非参考轨迹评测交互式编码代理。
primary_logic: |
  交互式编码代理评测目标 → 构建含多App、多人物关系、可复现时间/数据库状态的模拟世界与任务生成器 → 用期望/允许状态变化的数据库单测评分 → 揭示复杂工具使用中的完成率、稳健性与附带破坏边界
claims:
  - "AppWorld提供9个日常应用、457个API和750个任务，任务平均需要约9.5个API与约50行代码，复杂度显著高于只需1–4次API调用的既有工具使用基准 [evidence: comparison]"
  - "在AppWorld上，GPT4O结合ReAct仅达到48.8 TGC（Test-N）和30.2 TGC（Test-C），表明当前前沿LLM距离稳定完成复杂多App交互任务仍有明显差距 [evidence: comparison]"
  - "将API检索替换为Oracle APIs后，GPT4O在Test-C上的TGC最高仅提升到35.2，说明主要瓶颈不在API检索，而在交互式代码生成、状态跟踪与适应性执行 [evidence: analysis]"
related_work_position:
  extends: "InterCode (Yang et al. 2023)"
  competes_with: "API-Bank (Li et al. 2023); ToolTalk (Farn and Shin 2023)"
  complementary_to: "WebArena (Zhou et al. 2024); AndroidWorld (Rawles et al. 2024)"
evidence_strength: moderate
pdf_ref: paperPDFs/Agentic_Reasoning_Benchmarks_Applications_of_Agentic_Reasoning_Web_Agents/ACL_2024/2024_AppWorld_A_Controllable_World_of_Apps_and_People_for_Benchmarking_Interactive_Coding_Agents.pdf
category: Survey_Benchmark
---

# AppWorld: A Controllable World of Apps and People for Benchmarking Interactive Coding Agents

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2407.18901), [GitHub](https://github.com/stonybrooknlp/appworld)
> - **Summary**: 该工作提出一个可控的多应用数字世界与基于数据库状态的评测协议，用来严格衡量代理是否真的能通过交互式写代码完成复杂日常任务，并避免“顺手搞坏别的东西”。
> - **Key Performance**: GPT-4o + ReAct 仅达 **48.8 TGC / 32.1 SGC**（Test-N）与 **30.2 TGC / 13.0 SGC**（Test-C）。

> [!info] **Agent Summary**
> - **task_path**: 自然语言日常任务 + 可控多App状态/API文档 -> 多轮交互式代码执行 -> 最终数据库状态或答案
> - **bottleneck**: 现有工具基准无法评测需要循环/分支/调试/环境交互的真实多App任务，也无法稳健检测 collateral damage
> - **mechanism_delta**: 用可控 app-person 世界和数据库状态差分单测，替代基于参考轨迹的评测
> - **evidence_signal**: GPT-4o+ReAct 在 Test-C 仅 30.2 TGC，且 Oracle API 只带来有限增益，说明难点在交互式编码而非 API 检索
> - **reusable_ops**: [状态差分评测, contrast-set任务生成]
> - **failure_modes**: [不与环境交互而直接幻觉, API参数或返回结构误解]
> - **open_questions**: [如何扩展到UI-only应用, 如何低成本生成训练规模数据]

## Part I：问题与挑战

这篇论文要解决的真实问题，不是“模型能不能调用几个 API”，而是：

**代理能否在一个开放式、长程、多应用、带副作用约束的数字环境里，边看环境反馈边写代码，把事情做成，而且只做对该做的事。**

### 为什么现有评测不够
现有 tool-use benchmark 大多有两个局限：

1. **任务过短**：通常只是 1–4 次线性 API 调用。
2. **评分过弱**：常用“对照参考解”或 LLM/人工打分，难以处理多条合法解路径，也难检查副作用。

但真实日常任务往往不是这样。  
例如：先去 note 里找信息、读出自由文本结构、根据当前日期决定逻辑、再跨 app 操作、处理中途报错或默认支付卡失效等异常。这里的关键不是单步 tool call，而是**交互式代码生成 + 状态依赖决策**。

### 输入 / 输出接口
- **输入**：自然语言任务指令、任务专属初始数据库状态、当前时间、API 文档、stateful 执行环境
- **输出**：一段或多段逐步执行的代码/API 调用，以及最终的世界状态变化或问答答案

### 为什么现在要做
LLM 的 instruction following、coding、tool use 已经足够强，大家开始认真讨论“数字助理替人办事”。  
但如果没有**可复现、无真实代价、可检查副作用**的沙盒环境，研究会停留在演示级别，很难负责任地推进。

### 边界条件
AppWorld 的边界很清楚：
- 单助手执行单个用户任务
- 以 **API** 为主，不是 UI 操作
- 时间和数据库可冻结、可重置
- 没有真实金钱/邮件/隐私后果

这让它更像一个**面向代理研究的可控实验世界**，而不是现实互联网本身。

## Part II：方法与洞察

### 核心直觉

**核心变化**：把评测从“是否复现参考调用轨迹”，改成“是否把世界改到了正确且仅正确的状态”。

这改变了原来的评测瓶颈：

- **原瓶颈**：复杂任务有多条合法路径，过程比对天然不稳；同时，错误不只是不完成任务，还包括误删、误买、误付款、误发邮件等附带破坏。
- **关键改动**：引入可控数据库/时间 + 终态状态差分 + 期望/允许变化集合。
- **能力变化**：评测终于能覆盖需要循环、条件分支、读文档、看环境输出再改代码的真实交互式任务。

更因果地说：

**可控世界状态**  
→ 让每次运行从同一起点出发  
→ 可以比较不同代理的真实完成效果  

**状态差分评测**  
→ 不依赖单一路径 gold solution  
→ 能接受多种正确解，同时抓出 collateral damage  

**contrast set 任务设计**  
→ 不只测“会不会一题”  
→ 测“在轻微条件变化下是否仍然可靠”  

### 设计拆解

#### 1. AppWorld Engine：一个可控的“日常 App 世界”
作者构建了一个高质量模拟环境，包含：
- 9 个日常 app
- 457 个 API
- 101 张数据库表
- 约 100 个虚构用户及其关系和数字生活数据

它还配套：
- **ApiDocs**：让代理可交互查文档
- **Supervisor**：提供任务委托人账户信息
- **stateful execution shell**：像 Jupyter 一样多轮执行代码
- **frozen time**：任务依赖“今天/本月”等时间语义时可复现

这一步的作用是：把“真实世界太危险、太不可控”的问题，转换成“足够真实但完全可控”的研究环境。

#### 2. Benchmark：把复杂日常任务系统化
基准包含 **250 个 task scenario × 每个 3 个变体 = 750 个任务**。

每个 scenario 通过 Setup 程序生成时，会被刻意保证四件事：
- **Is well-defined**：任务真的可解
- **Has distractors**：有足够干扰项，逼代理认真查
- **Has hurdles**：有自然障碍，如默认卡失效
- **Forms contrast set**：同一场景下多个变体，只改关键条件，测鲁棒性

这比“随手编一条 instruction”强很多，因为它把任务难度、歧义和鲁棒性测试都结构化了。

#### 3. Evaluation：按数据库状态而非过程评分
这是论文最重要的评测创新。

做法不是看代理“有没有按参考步骤做”，而是看：
- 该发生的数据库变化有没有发生
- 不该发生的变化有没有出现

作者把每个任务写成一组 **state-based unit tests**：
- **expected changes**：必须发生
- **allowed changes**：允许但非必须
- 其他变化：视为 collateral damage

因此，一个任务即使有多条完成路径，也能被稳健评分。  
这对复杂交互式任务非常关键。

#### 4. Validation solutions：先保证题目真的能做
作者还为每个 scenario 写了程序化验证解，确保：
- 任务是可解的
- 评测器不会误杀正确解
- 后续环境升级不会把旧任务悄悄弄坏

这本质上是给 benchmark 自己做了端到端回归测试。

### 战略权衡

| 设计选择 | 带来的能力 | 代价/牺牲 |
|---|---|---|
| 用模拟 app 世界代替真实服务 | 可复现、安全、无真实副作用 | 与真实互联网/API 演化仍有域差 |
| 用状态差分评测代替参考轨迹比对 | 支持多解路径，并能检查 collateral damage | 需要手工设计高质量 evaluator |
| 用作者编写的 task generator 代替 crowdsourcing | 任务质量高、干扰项和障碍可控 | 构建成本高，规模较难暴涨 |
| 设置 unseen-app 的 Test-C | 能测文档理解和泛化，而非模板记忆 | 对小模型和上下文受限模型更苛刻 |

## Part III：证据与局限

### 关键证据

- **比较信号：当前模型远未“通关”**
  - 最强结果是 GPT-4o + ReAct，但在 Test-C 也只有 **30.2 TGC**。
  - 最好的开放模型在 Test-C 只有 **7.0 TGC** 量级。
  - 结论：这不是一个已经饱和的 benchmark，而是真正能拉开能力差距的测试床。

- **稳健性信号：会做单题，不等于会做同类题**
  - GPT-4o + ReAct 在 Test-C 上 **TGC 30.2**，但 **SGC 只有 13.0**。
  - 这说明模型并不具备稳定的场景级策略，经常只在某些变体上碰巧成功。

- **诊断信号：API 检索不是主瓶颈**
  - 给 Oracle APIs 后，GPT-4o 的分数只有限提升，Test-C 最好也只是 **35.2 TGC**。
  - 说明难点主要在：交互式写代码、理解返回结果、维护执行状态、根据错误调整行为。

- **误差分析信号：失败类型很“代理化”**
  - 不与环境交互，直接臆测
  - 误读 API 参数/返回 schema
  - 指令只完成一部分
  - 常识/时间理解错误
  - 忘记先前状态，重复劳动耗尽 budget

- **难度缩放信号：代码更长、API 更多、任务更难时，所有方法都持续掉分**
  - 这说明 AppWorld 测到的不是 prompt 小技巧，而是更底层的 agentic execution 能力。

### 局限性

- **Fails when**: 任务核心依赖真实网页/手机 UI、外部网络噪声、真实服务延迟、或跨代理协作时，AppWorld 的 API 沙盒不能完整复现这些摩擦与约束。
- **Assumes**: 依赖作者手工构建的高质量模拟后端、任务生成器与 evaluator；评测默认有足够上下文窗口和多轮执行预算；论文中的强基线还依赖 GPT-4o 等闭源模型，整体实验成本约 \$10K。
- **Not designed for**: 大规模训练数据供给、真实部署安全认证、或多智能体/人类协同流程的完整评测。

### 可复用组件

- **状态差分评测器**：适合任何“多路径正确解 + 需要检查副作用”的 agent benchmark
- **contrast-set 场景生成框架**：适合把“会不会做”升级为“稳不稳定做”
- **可控时间/数据库重置机制**：适合做严格对照实验和回归测试
- **validation solution harness**：适合保证 benchmark 长期维护时不失真

## Local PDF reference

![[paperPDFs/Agentic_Reasoning_Benchmarks_Applications_of_Agentic_Reasoning_Web_Agents/ACL_2024/2024_AppWorld_A_Controllable_World_of_Apps_and_People_for_Benchmarking_Interactive_Coding_Agents.pdf]]