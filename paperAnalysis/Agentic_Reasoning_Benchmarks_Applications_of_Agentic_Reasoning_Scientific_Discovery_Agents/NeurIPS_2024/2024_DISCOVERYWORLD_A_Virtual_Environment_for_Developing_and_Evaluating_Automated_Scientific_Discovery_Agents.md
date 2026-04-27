---
title: "DISCOVERYWORLD: A Virtual Environment for Developing and Evaluating Automated Scientific Discovery Agents"
venue: NeurIPS
year: 2024
tags:
  - Survey_Benchmark
  - task/agent-evaluation
  - task/scientific-discovery
  - parametric-task-generation
  - process-scoring
  - llm-as-judge
  - dataset/DISCOVERYWORLD
  - opensource/full
core_operator: 用跨8个科学主题的参数化文本仿真任务和三段式自动评分，评测智能体是否真正完成“提出假设—做实验—形成解释—执行结论”的完整科学发现循环
primary_logic: |
  评测端到端科学发现能力 → 在8个主题上参数化生成120个长程发现任务与10类单元测试 → 用任务完成/过程得分/解释性知识三层评分自动评估 → 揭示当前智能体在假设生成、实验规划与证据整合上的能力边界
claims:
  - "DISCOVERYWORLD提供8个科学主题×3档难度×5个seed构成的120个发现任务，并附带10类单元测试，可系统评估端到端科学发现能力而非单点交互技巧 [evidence: analysis]"
  - "GPT-4o驱动的基线智能体在单元测试上平均完成率为44%–64%，但在完整发现任务上的最佳平均完成率仅为Easy 38%、Normal 23%、Challenge 18%，说明主要瓶颈是闭环式发现而不是基础操作 [evidence: analysis]"
  - "人类科学家在16个Normal/Challenge任务上平均完成率为66%、知识得分为55%，显著高于基线智能体，表明该环境具有可解但困难的诊断价值 [evidence: analysis]"
related_work_position:
  extends: "ScienceWorld (Wang et al. 2022)"
  competes_with: "ScienceWorld (Wang et al. 2022); MLAgentBench (Huang et al. 2024)"
  complementary_to: "CLIN (Majumder et al. 2023); Decomposed Prompting (Khot et al. 2023)"
evidence_strength: moderate
pdf_ref: paperPDFs/Agentic_Reasoning_Benchmarks_Applications_of_Agentic_Reasoning_Scientific_Discovery_Agents/NeurIPS_2024/2024_DISCOVERYWORLD_A_Virtual_Environment_for_Developing_and_Evaluating_Automated_Scientific_Discovery_Agents.pdf
category: Survey_Benchmark
---

# DISCOVERYWORLD: A Virtual Environment for Developing and Evaluating Automated Scientific Discovery Agents

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2406.06769) · [Code](https://github.com/allenai/discoveryworld)
> - **Summary**: 该工作提出首个面向“端到端科学发现智能体”的虚拟基准，用跨学科参数化任务和完成/过程/解释三层评分，把“提假设—做实验—得解释—采取行动”变成可重复、可自动评测的问题。
> - **Key Performance**: GPT-4o ReAct 在完整发现任务上的平均完成率仅 Easy 38% / Normal 18% / Challenge 18%；而人类科学家在 Normal/Challenge 任务上平均完成率 66%，知识得分 55%。

> [!info] **Agent Summary**
> - **task_path**: 文本/2D环境观察 + 任务描述 -> 假设生成/实验设计与执行/知识整理 -> 任务完成 + 解释性发现评分
> - **bottleneck**: 缺少能把基础交互技能与真正科学发现闭环分开测量的统一、低成本 benchmark
> - **mechanism_delta**: 用多主题开放式长程任务 + 单元测试 + 完成/过程/知识三层评分，显式区分“会操作”与“会发现”
> - **evidence_signal**: 基线在 unit tests 完成率有 44%–64%，但在完整发现任务上最佳完成率仅 18%–38%，且明显落后于人类 66%
> - **reusable_ops**: [parametric-task-generation, three-part-scorecard]
> - **failure_modes**: [procedural-skills-without-explanatory-discovery, long-horizon-hypothesis-experiment-loop-breakdown]
> - **open_questions**: [virtual-to-real-transfer, open-and-cheaper-knowledge-grading]

## Part I：问题与挑战

这篇论文要解决的，不是“再做一个会拿仪器的 agent”，而是补上一个更根本的空白：**如何低成本、可重复地评测 AI 是否真的具备端到端科学发现能力**。

### 真问题是什么？
现有路线大致分成两类：

1. **真实实验室自动化系统**：如化学、遗传学等方向的机器人科学家，确实能做实验，但成本高、部署复杂，而且通常强依赖具体领域与预定义假设空间。
2. **虚拟环境或游戏环境**：能测导航、操控、简单科学任务或受限假设搜索，但往往不要求 agent 自己提出假设、设计实验、排除干扰并形成解释。

因此真正缺失的是一个中间层：  
**既不像真实实验那样昂贵，又不像普通交互环境那样只测局部技能，而是能测完整科学方法闭环。**

### 真正瓶颈在哪里？
论文认为瓶颈不是单一动作能力，而是**闭环整合能力**：

- 能否从开放式任务描述中提出候选解释；
- 能否设计有区分度的实验而不是盲试；
- 能否从观测中修正假设；
- 能否给出“解释性发现”，而不只是蒙对最终答案。

这也是为什么作者额外强调：只看最终成功/失败不够，因为它无法区分：
- 卡在导航/交互；
- 卡在实验流程；
- 还是根本没发现正确的因果解释。

### 输入/输出接口与边界条件
- **输入**：以 JSON 文本观察为主，可选 2D 视图；包含邻近物体、库存、可交互对象、位置、对话选项、DiscoveryFeed 等。
- **输出**：14 个离散动作，如移动、拿取、使用仪器、对话、开关设备等；另提供两个 `teleport` handicap 动作，刻意降低导航噪声。
- **环境边界**：32×32 网格、低保真但“现实化简”的科学世界。
- **任务边界**：8 个主题（如 proteomics、radioisotope dating、reactor tuning、plant nutrients、space illness、rocket science、translation）× 3 档难度 × 5 个 seeds = **120 个官方任务**，外加 **10 类 unit tests**。

### 为什么现在做？
因为 LLM agent 已经能在不少环境里表现出局部交互与推理能力，但社区仍缺少一个明确回答下面问题的基准：

> 这些 agent 到底是在“会用工具”，还是已经接近“会做科学发现”？

DISCOVERYWORLD 试图把这个问题第一次系统化地测出来。

## Part II：方法与洞察

作者把整个 benchmark 设计成四层：**环境层、任务层、评分层、诊断层**。

### 1) 环境层：可操作、可测量、低成本的科学模拟世界
- 自研 text-first simulator，每个对象由材料和属性构成，许多属性可被仪器测量。
- 支持文本观察，也支持可选 2D 可视化覆盖。
- 交互对象包含仪器、样本、容器、NPC、文档等，适合表达“测量—记录—试验—结论”流程。

它不是高保真物理仿真，而是**为科学方法步骤服务的结构化世界**。

### 2) 任务层：从“做题”改成“做发现”
24 个高层模板 = 8 个主题 × 3 档难度；再通过 seed 参数化生成 120 个具体任务实例。

这些任务有几个关键特性：
- **长程**：不是几步就能做完；
- **开放式**：任务不会直接告诉解法；
- **带干扰项**：需要排除错误线索；
- **要求解释**：不是只要终局动作对就行。

例如：
- Proteomics：通过测蛋白数据做 clustering/outlier discovery；
- Archaeology：验证哪种 radioisotope 才能用于定年；
- Reactor Lab：通过回归关系推断晶体频率；
- Space Sick：查明殖民者生病原因并修复系统；
- Translation：通过 Rosetta-stone 式线索归纳未知语言。

### 3) 评分层：把“成功、过程、解释”分开
这是这篇论文最关键的评测设计。

1. **Task Completion**
   - 二值：最终是否完成任务。

2. **Task Process / Procedural Score**
   - 用细粒度 scorecard 记录是否执行了关键步骤。
   - 例如是否测过关键样本、是否调过反应器、是否把红旗放到正确位置等。

3. **Explanatory Knowledge Discovery**
   - 看 agent 是否发现了任务要求的核心解释知识。
   - 通过预定义 gold questions 评估，可人工评分，也可用 GPT-4o 自动评分。

这一步把 benchmark 从“游戏通关评测”推进成“发现能力评测”。

### 4) 诊断层：用 unit tests 分离低级技能与高级发现
作者额外设计了 10 类 unit tests，专门测试：
- 对话、
- 仪器使用、
- pick-and-place、
- door/key、
- 搜索、
- 与移动 NPC 交互等。

这样就能回答一个非常关键的问题：  
**如果 agent 在完整发现任务里失败，是因为不会基本交互，还是不会做科学发现？**

### 核心直觉

DISCOVERYWORLD 的关键变化，不是简单“增加任务数量”，而是**改变了被测能力的分布和测量方式**：

- **What changed**：从“给定目标下的短程交互/固定解法任务”，改成“跨主题、不给解法、带干扰、需要解释的长程发现任务”。
- **Which bottleneck changed**：测量瓶颈从“能不能执行动作序列”转成“能不能在不确定因果结构中提出并验证假设”；评分约束从“只看终局”转成“终局 + 过程 + 解释”。
- **What capability changed**：benchmark 开始能区分两类 agent：  
  1) 会局部操作但不会闭环发现；  
  2) 能把假设、实验、证据整合成解释并据此行动。

换句话说，这个 benchmark 真正测的是：

**任务描述 → 假设空间构建 → 区分性实验 → 证据整合 → 解释性结论**

而不是“看起来像在推理”的动作轨迹。

### 策略性取舍

| 设计选择 | 带来的能力增益 | 代价/取舍 |
|---|---|---|
| 文本为主、可选2D的模拟环境 | 成本低、易参数化、便于 LLM agent 接入 | 物理保真度低，现实迁移有限 |
| 多主题参数化任务 | 降低 task-specific hard-coding，鼓励通用发现策略 | 覆盖仍有限，不代表真实科学全貌 |
| 完成/过程/知识三层评分 | 能区分 brute-force 通关与真正解释性发现 | 知识评分依赖 gold questions 与判分器 |
| 加入 unit tests | 能诊断失败是交互问题还是发现闭环问题 | 无法完全隔离记忆、规划、长程搜索等因素 |
| 提供 teleport handicap | 避免导航成为主噪声，更聚焦发现能力 | 弱化了具身导航真实性 |

## Part III：证据与局限

### 关键证据信号

- **信号1｜基线比较：当前强 LLM agent 对完整发现任务仍然很弱**  
  三个 GPT-4o 驱动基线（ReAct、Plan+Execute、Hypothesizer）在 zero-shot 下表现都不理想。  
  最醒目的结果是：**ReAct 的平均完成率只有 Easy 38%、Normal 18%、Challenge 18%**。  
  这说明 benchmark 并不是靠简单 prompting trick 就能过关。

- **信号2｜诊断拆分：局部技能还可以，但闭环发现不行**  
  同一批基线在 10 个 unit tests 上平均完成率有 **44%–64%**。  
  结论很明确：问题不只是“不会拿东西/不会测量/不会对话”，而是**无法把这些局部能力组织成假设驱动的长程发现流程**。

- **信号3｜人类对照：任务是可解的，不是无意义地难**  
  11 位具有自然科学硕/博士背景的人类参与者，在 16 个 Normal/Challenge 任务上达到：
  - **平均完成率 66%**
  - **平均知识得分 55%**
  
  这说明 DISCOVERYWORLD 不是纯粹靠刁钻规则卡 agent，而是真能拉开“人类科学家 vs 当前 agent”的差距。

- **信号4｜最难的不是操作，而是解释闭环**  
  论文特别指出，像 Space Sick、Reactor Lab、Rocket Science 这类需要开放式排错、回归推断或多步公式使用的任务，对 agent 尤其困难。  
  Hypothesizer 虽然显式维护工作记忆，**Challenge 难度下知识得分仍只有 8%**，说明“显式记忆”本身还不足以解决端到端发现。

### 关键信息，不是数字堆砌
这篇论文最重要的结论可以压缩成一句话：

> 当前 LLM agent 已经具备部分科学交互部件，但远未具备稳定的端到端科学发现能力。

### 局限性
- **Fails when**: 需要高保真物理交互、连续控制、真实实验噪声、开放互联网工具链或无标准答案的原创研究时，DISCOVERYWORLD 的结论不能直接外推。
- **Assumes**: 文本/网格化模拟足以承载科学方法关键步骤；每个任务都可写成有限的 gold explanatory questions；自动知识评分可借助 GPT-4o 这类长上下文闭源模型；完整评测 120 任务需要显著 API 成本（文中估计约 \$3.3k–\$8.4k）。
- **Not designed for**: 真实 wet-lab robotics、精细具身操控、安全审查、或直接替代现实科学实验。

### 可复用组件
这篇工作最值得复用的不是某个 baseline，而是它的 benchmark 设计模板：

- **参数化任务生成**：主题 × 难度 × seed 的 benchmark 构造方式；
- **unit tests + full tasks 双层诊断**：先测部件技能，再测闭环能力；
- **三段式 scorecard**：完成、过程、解释分开看；
- **knowledge grading prompt**：把“是否发现解释”转成可自动判分的问题集合。

如果你在做 scientific agent、tool-using agent、long-horizon planning agent，这些组件都可以直接借用。

## Local PDF reference

![[paperPDFs/Agentic_Reasoning_Benchmarks_Applications_of_Agentic_Reasoning_Scientific_Discovery_Agents/NeurIPS_2024/2024_DISCOVERYWORLD_A_Virtual_Environment_for_Developing_and_Evaluating_Automated_Scientific_Discovery_Agents.pdf]]