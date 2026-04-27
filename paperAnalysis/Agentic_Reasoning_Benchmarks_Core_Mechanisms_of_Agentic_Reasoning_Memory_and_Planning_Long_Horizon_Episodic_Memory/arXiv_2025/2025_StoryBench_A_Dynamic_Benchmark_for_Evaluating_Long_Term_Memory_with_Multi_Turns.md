---
title: "StoryBench: A Dynamic Benchmark for Evaluating Long-Term Memory with Multi Turns"
venue: arXiv
year: 2025
tags:
  - Survey_Benchmark
  - task/MLLM-evaluation
  - dynamic-benchmark
  - branching-narrative
  - multi-turn-evaluation
  - dataset/StoryBench
  - opensource/no
core_operator: 用交互式小说的分支剧情与双模式回溯协议，把长期记忆评估从静态长文本检索改成动态多轮因果诊断
primary_logic: |
  长期记忆评测目标 → 将《隐形守护者》人工标注为场景节点/选择节点的叙事DAG，并构造即时反馈与自恢复两种交互模式 → 以正确率、首次命中率、重试次数、最长连续正确序列与成功通关数等指标评分 → 揭示模型在知识保持、长程因果追踪与自我纠错上的能力边界
claims:
  - "StoryBench 在作者对比的基准中是唯一同时覆盖长上下文、连续性、复杂推理、动态性、多轮、多解以及 LTM+STM 联合使用七个维度的基准 [evidence: analysis]"
  - "在 Immediate Feedback 模式下，Doubao 1.5-pro 取得最高 Overall Acc 80.98%，而 Claude 3.5 Sonnet 取得最高 Success Count 8，说明局部决策正确率与完整长链完成能力并不等价 [evidence: comparison]"
  - "在 Self Recovery 模式下，所有已报告模型的 Success Count 仅为 0–2，表明无外部提示的长程回溯纠错仍是当前 LLM 的显著短板 [evidence: analysis]"
related_work_position:
  extends: "LTM Benchmark (Castillo-Bolado et al. 2024)"
  competes_with: "LTM Benchmark (Castillo-Bolado et al. 2024); LongBench (Bai et al. 2024)"
  complementary_to: "MemGPT (Packer et al. 2023); Mem0 (Chhikara et al. 2025)"
evidence_strength: moderate
pdf_ref: paperPDFs/Agentic_Reasoning_Benchmarks_Core_Mechanisms_of_Agentic_Reasoning_Memory_and_Planning_Long_Horizon_Episodic_Memory/arXiv_2025/2025_StoryBench_A_Dynamic_Benchmark_for_Evaluating_Long_Term_Memory_with_Multi_Turns.pdf
category: Survey_Benchmark
---

# StoryBench: A Dynamic Benchmark for Evaluating Long-Term Memory with Multi Turns

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2506.13356)
> - **Summary**: 论文把长期记忆评测从“长文本里找答案”改成“在分支剧情中持续做决策并在失败后自我回溯”，更直接暴露 LLM 的长程记忆与因果修复能力。
> - **Key Performance**: Immediate Feedback 下 Doubao 1.5-pro 的 Overall Acc 达 80.98%，Claude 3.5 Sonnet 的 Success Count 最高为 8；Self Recovery 下各模型 Success Count 仅 0–2。

> [!info] **Agent Summary**
> - **task_path**: 多轮文本叙事历史与当前场景/选项 → 下一步分支选择或失败后的错误回溯点
> - **bottleneck**: 现有长上下文基准大多只测静态检索，测不到跨回合状态维护、长程因果追踪和自我纠错
> - **mechanism_delta**: 用交互式小说 DAG + 即时反馈/自恢复双模式替代静态 QA，使模型必须在交互过程中维护隐式故事状态
> - **evidence_signal**: Self Recovery 下全部模型成功通关数跌到 0–2，且作者观察到普遍的浅层回溯失败
> - **reusable_ops**: [剧情DAG标注, 双模式反馈协议]
> - **failure_modes**: [长叙事中的上下文一致性丢失, 只回退最近1到2步的浅层回溯]
> - **open_questions**: [单一文本互动小说域能否代表通用LTM, 如何在不引入阈值提示的情况下稳定评估真实自恢复]

## Part I：问题与挑战

这篇论文要解决的，不是普通的“长文本能不能读完”，而是更接近 agent 场景的长期记忆评测问题：**模型能否在持续演化的环境里，记住早先信息、维护当前隐状态、并在后续失败时追溯到更早的错误决策**。

作者指出，现有 LTM / long-context benchmark 的主要盲点有三类：

1. **Knowledge Retention 不够真实**  
   很多基准仍是静态上下文 + 单次问答，模型只要把答案从上下文中“捞出来”即可，未必真的保持了人物、事件、目标之间的长期一致性。

2. **Sequential Reasoning 不够动态**  
   真实长期记忆不只是 recall，更重要的是顺着事件序列跟踪因果、状态变化、角色关系与目标漂移。传统 benchmark 很少让“之前的错误”真实改变“之后的世界”。

3. **Flexibility 不足**  
   固定答案、单解、单轮、静态任务，难以覆盖多路径、多解、不同上下文条件下的记忆调用。

### 输入/输出接口

- **输入**：当前 scene 描述、对话、历史剧情、当前 choice 选项
- **输出**：
  - 正常模式下：选择一个分支
  - 自恢复模式下：在失败后指出最早可能出错的决策点，并尝试从该点恢复

### 边界条件

- 纯文本环境，不含多模态
- 数据主要来自单一互动小说《The Invisible Guardian》
- 当前只覆盖序章到第 5 章，长度和域仍有限
- 评测依赖 API 模型，且协议中存在内容过滤、token 限制与错误阈值干预

**一句话概括真瓶颈**：  
现有基准把长期记忆测成了“远距离检索”，而 StoryBench 试图把它测成“长期状态维护 + 多步因果推理 + 自我纠错”。

---

## Part II：方法与洞察

StoryBench 的核心不是新模型，而是**新的评测环境设计**。

### 1. 数据与环境：把互动小说改造成可评测的叙事图

作者选用互动小说游戏《The Invisible Guardian》作为来源，而不是纯 synthetic data 或真实开放环境，原因很直接：

- synthetic 数据可控，但通常过于模板化，缺少真实叙事的复杂依赖
- 真实世界数据复杂但噪声太大，难以界定成功/失败路径
- 互动小说兼具**叙事连贯性**与**因果结构可控性**

最终构建的数据集包括：

- **311 个 scene nodes**
- **86 个 choice nodes**
- 以 **DAG** 形式组织剧情

这个结构天然覆盖四类难点：

- 线性情节理解
- 长程依赖
- 复杂相互制约决策
- 多解分支通关

### 2. 双模式协议：把“记忆”拆成短程修正和长程回溯

#### Immediate Feedback
- 模型选错后，系统立刻指出并要求重试
- 主要测：
  - 局部修正能力
  - 短程交互记忆
  - 对错误信号的响应速度

#### Self Recovery
- 模型选错后不会立刻收到提示
- 错误会继续传播，直到进入失败结局
- 随后模型需要自己判断：**最早是哪里出错了**
- 主要测：
  - 长程因果追踪
  - 跨多步的错误定位
  - 无提示条件下的自我修复

### 3. 指标设计：把“记住了没”拆成多个可诊断面

作者没有只给一个总分，而是把评测分成两大维度：

- **知识保持**
  - Overall Accuracy
  - First-Try Accuracy
  - Longest Consecutive Correct Sequence

- **顺序推理**
  - Easy / Hard Accuracy
  - Retry Count
  - Max Error per Choice
  - ErrorCount≥threshold

再加上辅助指标：

- Runtime Cost
- Token Consumption
- Success Count

这使得评测不只看“局部题目做对没有”，也看“能不能真的把长链任务走通”。

### 核心直觉

StoryBench 最关键的变化是：

**把评测对象从“静态上下文中的答案匹配”改成“动态故事世界中的状态维护”。**

更具体地说，是这条因果链：

- **什么变了**：从单轮/静态长文本任务，变成多轮、分支、会被模型选择改写后续状态的叙事环境
- **哪个瓶颈变了**：评测瓶颈从“远距离检索”变成“隐式世界状态维护 + 长程因果追踪 + 失败后的反事实回溯”
- **能力上发生了什么变化**：能够区分
  - 只是会做局部题目的模型
  - 和真正能完成长程任务、并在出错后修回来的模型

为什么这个设计有效？  
因为在分支叙事里，错误不会停留在当前回合，而会**级联污染后续状态**。这样一来，模型如果没有真正记住前文、没有持续维护人物关系和目标状态，就会在 hard decision 和 self-recovery 阶段暴露出来。也因此，StoryBench 比传统“needle in a haystack”类任务更接近 agent 的真实记忆压力。

### 战略权衡

| 设计选择 | 改变了什么测量瓶颈 | 获得的诊断能力 | 代价/权衡 |
|---|---|---|---|
| 互动小说 DAG | 从静态问答变为状态演化 | 可测 continuity、dynamics、multi-turn | 领域集中在单一叙事游戏 |
| Immediate Feedback | 引入局部纠错信号 | 可分离短程修正与交互学习能力 | 反馈可能掩盖真实长期记忆短板 |
| Self Recovery | 去掉实时提示，要求回溯 | 可直接测长程错误定位和自修复 | 任务更难，易卡死，需要辅助阈值 |
| 多指标而非单总分 | 从单点分数到能力剖面 | 能区分 recall、推理、完成度、效率 | 协议更复杂，分析成本更高 |
| 重复试验 | 抵抗 API 波动 | 排名更稳健 | 成本更高，且仍受服务状态影响 |

---

## Part III：证据与局限

### 关键证据信号

**信号 1：覆盖面分析**  
作者在 Table 1 中把 StoryBench 与 Needle-in-a-Haystack、RULER、LongBench、LooGLE、InfiniteBench、LTM Benchmark 等进行维度比较。其核心结论不是“更长”，而是**更全**：它同时覆盖 continuity、dynamics、multi-turn、multi-solution、LTM+STM 联合使用，这正是现有长期记忆评测常缺的部分。

**信号 2：局部准确率和完整完成能力并不一致**  
Immediate Feedback 下：

- Doubao 1.5-pro 的 **Overall Acc 最高（80.98%）**
- 但 Claude 3.5 Sonnet 的 **Success Count 最高（8）**

这说明：  
**会在局部 decision 上做对，并不等于能把整条长期任务链走通。**  
这也是 StoryBench 相比传统 benchmark 更有价值的地方——它把“局部正确”与“全局完成”拆开了。

**信号 3：Self Recovery 明显拉开差距**  
进入无提示自恢复后，所有模型表现都明显下降，Success Count 仅在 **0–2** 之间。作者据此指出，当前 LLM 的真正短板不是普通 recall，而是：

- 长程错误定位
- 多步因果回溯
- 早期错误修复

这比单纯 long-context QA 更接近真实 agent memory 的难点。

**信号 4：hard 决策显著难于 easy 决策**  
各模型在 Hard Accuracy 上普遍低于 Easy Accuracy，说明最难的不是“记住一条事实”，而是把远处信息、隐状态变化和后果链条一起整合起来。

**信号 5：失败案例具有结构性，不只是格式问题**  
作者的 failure analysis 显示，两类错误最关键：

1. **长叙事一致性丢失**：后续选择与前文人物动机、事件事实或世界规则冲突  
2. **浅层回溯**：失败后只回退最近 1–2 步，而不是真正找到更早的根因决策

这说明 StoryBench 测到的不是表面格式错误，而是长期记忆与顺序推理的结构性缺陷。

### 1-2 个最值得记住的指标

- **Doubao 1.5-pro: Overall Acc = 80.98%（Immediate Feedback）**
- **Claude 3.5 Sonnet: Success Count = 8（Immediate Feedback）**
- **Self Recovery：全部模型 Success Count 仅 0–2**

### 局限性

- **Fails when**: 需要评估跨领域、跨模态、或比当前 6 章剧情更长得多的长期依赖时，单一文本互动小说环境的覆盖不足；对开放世界、工具调用和现实噪声环境的代表性有限。
- **Assumes**: 依赖人工构建的剧情 DAG 与细粒度标注；依赖 API 模型可稳定访问；评测中实际使用了 CoT prompting、敏感词过滤、部分模型 5,000 token/turn 限制，以及自恢复模式下“连续 9 次错误后给出正确答案”的软干预。
- **Not designed for**: 评估真实在线环境中的自主探索、外部工具使用、生产级多会话记忆系统，或多模态 agent 的长期记忆能力。

### 可复用部分

这篇工作最可复用的不是某个模型，而是一套评测设计组件：

- **scene / choice 的 JSON-DAG 标注范式**
- **Immediate Feedback / Self Recovery 双模式协议**
- **把 retention 与 sequential reasoning 分离的指标体系**
- **用 repeated trials 对抗 API 波动的评测流程**

如果你要做 memory agent、external memory、RAG-memory、或长期对话系统评测，这套协议是可以直接迁移的。

---

## Local PDF reference

![[paperPDFs/Agentic_Reasoning_Benchmarks_Core_Mechanisms_of_Agentic_Reasoning_Memory_and_Planning_Long_Horizon_Episodic_Memory/arXiv_2025/2025_StoryBench_A_Dynamic_Benchmark_for_Evaluating_Long_Term_Memory_with_Multi_Turns.pdf]]