---
title: "MultiAgentBench: Evaluating the Collaboration and Competition of LLM agents"
venue: arXiv
year: 2025
tags:
  - Survey_Benchmark
  - task/multi-agent-evaluation
  - milestone-based-evaluation
  - graph-topology
  - cognitive-planning
  - dataset/MultiAgentBench
  - dataset/ResearchTown
  - opensource/full
core_operator: "用跨场景多智能体交互环境、里程碑KPI与通信/规划双评分，统一诊断LLM代理的协作与竞争能力"
primary_logic: |
  多智能体系统配置与环境任务 → 构造6类协作/竞争场景并控制通信拓扑、规划策略 → 以任务分、里程碑KPI、通信/规划评分联合打分 → 揭示模型能力、协调机制与社会推理的能力边界
claims:
  - "在6类场景、5个模型的对比中，gpt-4o-mini取得最高平均任务表现，其中Research TS为84.13%，Coding TS为65.10 [evidence: comparison]"
  - "在Research场景中，graph协调协议在任务表现、规划效率和token开销的综合权衡上最好，而tree最差 [evidence: ablation]"
  - "相较vanilla、CoT和group discussion，cognitive self-evolving planning取得最高协调分，并将里程碑完成率提升约3% [evidence: ablation]"
related_work_position:
  extends: "LLM-Coordination (Agashe et al. 2024)"
  competes_with: "AgentBench (Liu et al. 2023); LLM-Coordination (Agashe et al. 2024)"
  complementary_to: "AutoGen (Wu et al. 2023a); AgentVerse (Chen et al. 2023b)"
evidence_strength: strong
pdf_ref: paperPDFs/Agentic_Reasoning_Benchmarks_Core_Mechanisms_of_Agentic_Reasoning_Language_Communication_and_Social_Reasoning/arXiv_2025/2025_MultiAgentBench_Evaluating_the_Collaboration_and_Competition_of_LLM_agents.pdf
category: Survey_Benchmark
---

# MultiAgentBench: Evaluating the Collaboration and Competition of LLM agents

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2503.01935), [Code](https://github.com/MultiagentBench/MARBLE)
> - **Summary**: 这篇工作提出一个面向多智能体LLM系统的统一基准，把“是否完成任务”扩展为“如何协作、如何竞争、谁推动了进度、哪种协议更有效”的可诊断评测。
> - **Key Performance**: gpt-4o-mini 在 Research 上达到 **84.13% TS**；cognitive self-evolving planning 将 **里程碑完成率提升约 3%**

> [!info] **Agent Summary**
> - **task_path**: 多智能体配置/交互日志/环境状态 -> 协作与竞争能力评分
> - **bottleneck**: 现有benchmark大多只测单代理或单域结果，无法稳定拆分“底座模型能力”与“协调机制能力”
> - **mechanism_delta**: 用六类环境 + 里程碑KPI + 通信/规划双评分 + 协议/规划策略控制变量，替代只看最终成功率的黑箱评测
> - **evidence_signal**: 6场景×5模型比较，外加协议/规划/迭代次数/agent数量消融与人评对齐
> - **reusable_ops**: [milestone-kpi, protocol-sweep]
> - **failure_modes**: [LLM评分偏置, 过多交互导致协调退化]
> - **open_questions**: [里程碑KPI能否泛化到开放式任务, 不同judge模型下评分是否稳定]

## Part I：问题与挑战

这篇论文真正要解决的，不是“再加几个 agent 任务”，而是**多智能体评测的可诊断性**。

现有 AgentBench、GAIA 这类基准更偏向单代理执行；一些多智能体工作又常局限在单一环境，导致社区很难回答三个关键问题：

1. 系统成功，到底是因为**底座模型强**，还是因为**协调协议设计得好**？
2. 多轮通信究竟在帮忙，还是只是在增加 token、制造噪声和误导？
3. 在 Werewolf、Bargaining 这类**冲突目标**场景里，LLM agent 是否真的具备可用的社会推理、信任管理和竞争策略？

**输入/输出接口**也因此需要重新定义：

- **输入**：支持 function calling 的 LLM agents、角色画像、关系图、通信拓扑、规划策略、环境工具。
- **输出**：不仅有最终任务分（TS），还有里程碑 KPI、通信分、规划分、协调分（CS），以及竞争场景中的过程/结果表现。

**为什么现在值得做**：多智能体 LLM 已经被大量用于科研、编码、游戏和谈判，但评测仍停留在“做没做成”的粗粒度层面。没有统一 benchmark，就无法系统比较协议、规划和模型本体之间的真实贡献。

**边界条件**：
- 共 6 类场景：Research、Minecraft、Database、Coding、Bargaining、Werewolf。
- 前四类偏**共同目标协作**，后两类偏**竞争/对抗**。
- Werewolf 采用环境中介通信，并主要从 villager 侧衡量协作质量。

## Part II：方法与洞察

MultiAgentBench 的核心不是单个模型，而是一个可控评测框架 **MARBLE**。它把多智能体系统拆成几个可以被独立拨动的“旋钮”。

### 评测框架怎么搭

1. **场景层**  
   同时覆盖共同目标与冲突目标：
   - 共同目标：research co-authoring、Minecraft 建造、数据库诊断、协作编码
   - 冲突目标：bargaining、werewolf

2. **协调层**  
   统一用 agent graph 来显式表示关系，并切换不同通信结构：
   - centralized：star、tree
   - decentralized：graph、chain

3. **规划层**  
   对集中式 planner 再切换四种规划提示：
   - vanilla
   - CoT
   - group discussion
   - cognitive self-evolving planning

4. **评分层**  
   不再只看最终答案，而是三层联合诊断：
   - **Task Score**：最终任务完成质量
   - **KPI**：执行过程中达成了多少里程碑、哪些 agent 有贡献
   - **Coordination Score**：通信质量 + 规划质量

其中，一部分任务分数是规则评估（如 coding / database / werewolf），另一部分使用 rubric + LLM judge；作者还在 Werewolf 上做了人评对齐，说明这个 judge 至少在该场景下不是完全失真。

### 核心直觉

以前的多智能体评测最大问题是：**只观测结果，不观测协作过程**。  
本文的关键变化是把这个 measurement bottleneck 改掉：

- **原来**：只知道最后赢了还是输了
- **现在**：知道中间谁推进了任务、通信是否有效、计划是否合理、协议是否放大了信息流

这带来的能力变化是：

- 能把“模型本体弱”与“协调机制差”区分开
- 能分析不同拓扑到底改变了什么信息约束
- 能观察竞争场景中是否出现真实的社会行为模式，而不是表面上“多聊了几轮”

**为什么这套设计有效**：

- 里程碑把长程任务切成可观测的阶段进度，避免只看终局
- 显式拓扑把信息流模式变成实验变量，而不是隐藏实现细节
- Communication / Planning 分离，让“会说”与“会做”不再混在一起
- 同一框架下换模型、换协议、换 planner，可以做更接近因果的 controlled comparison

### 战略权衡

| 设计选择 | 改变了什么约束 | 获得的能力 | 代价 |
|---|---|---|---|
| 最终结果 → 结果 + KPI + CS | 过程可观测性提升 | 能定位失败发生在执行还是协调 | 更依赖里程碑设计和 LLM judge |
| Star/Tree vs Graph/Chain | 信息流集中度、并行度不同 | 可比较中心化与去中心化协作 | token 成本与噪声模式不同 |
| Vanilla/CoT/Group/Cognitive | 规划深度与反思回路不同 | 可分析“规划机制”是否真带来收益 | prompt 设计敏感，group discussion 可能过载 |
| 更多 agent / 更多迭代 | 群体规模和交互预算增加 | 可观察 scaling 与 emergent behavior | 容易出现协调退化和冲突指令 |

一个很重要的洞察是：**更多协作不等于更好协作**。这篇工作最有价值的地方，正是把这种非单调性测出来了。

## Part III：证据与局限

### 关键证据信号

- **跨模型比较 / 结论**  
  gpt-4o-mini 整体任务完成最稳，在 Research 上 **84.13% TS**、Coding 上 **65.10 TS**。这说明当前阶段，**底座模型能力仍然是多智能体系统表现的主导因素**。

- **反例信号 / 结论**  
  高协调分不必然带来高任务分。比如 Meta-Llama-3.1-70B 在 Minecraft 中 CS 很高，但 TS 极低。也就是说，**“看起来在协作”不等于“真的能把任务做成”**。

- **协议消融 / 结论**  
  在 Research 场景，graph 协议综合最好，tree 最差。说明知识密集型协作里，**充分互联但不过度层级化**的信息流更有效。

- **规划策略消融 / 结论**  
  cognitive self-evolving planning 的协调分最高，并带来约 **3%** 的里程碑完成率提升；group discussion 反而最差，说明群聊式规划并不会自动产生更好的组织效果，反而可能引入组织摩擦。

- **尺度效应 / 结论**  
  Minecraft 中迭代次数从 1 增到 7 时分数上升，但到 10 时协调显著下降；Research 中 agent 数从 1 到 3 改善明显，但继续增加会压低 KPI。说明多智能体系统存在真实的**协调过载点**。

- **评分可靠性 / 结论**  
  Werewolf 中机器评分与人工评分接近，支持其 prompt-based coordination judge 至少具备基础可用性。

### 能力跳跃到底在哪里

相较于只看单 agent 成败的 benchmark，这个工作最大的跃迁不是“任务更难”，而是**能更清楚地解释为什么失败**：

- 是模型本身不够强？
- 是通信拓扑限制了信息流？
- 是 planner 没把角色互补性转成收益？
- 还是在竞争环境里，信任与信息披露策略出了问题？

这就是它相对以往 benchmark 的核心增量。

### 局限性

- **Fails when**: 任务极度开放、没有稳定可定义的里程碑，或需要真实物理交互/丰富多模态感知时，当前 KPI 与 LLM-based judge 可能失稳。
- **Assumes**: agent 支持 function calling；协调评分可由 LLM judge 近似人工判断；Werewolf 对手固定为 GPT-4o；部分任务和里程碑由 LLM 生成后再人工校验；实验还默认较充足的记忆与交互预算。
- **Not designed for**: 成本/时延/安全红队评测、真人-LLM混合团队评测、以及真实生产系统中的端到端部署验证。

特别需要指出的是：虽然代码和数据开源，但**评分与部分对手设置仍依赖闭源模型**，这会直接影响严格复现和跨实验室可比性。

### 可复用组件

- 里程碑式 KPI 与贡献归因流程
- 可切换 star/tree/graph/chain 的协议评测 harness
- planner 策略对比模板（vanilla / CoT / cognitive evolve）
- 多场景日志驱动的协作/竞争评测脚本

![[paperPDFs/Agentic_Reasoning_Benchmarks_Core_Mechanisms_of_Agentic_Reasoning_Language_Communication_and_Social_Reasoning/arXiv_2025/2025_MultiAgentBench_Evaluating_the_Collaboration_and_Competition_of_LLM_agents.pdf]]