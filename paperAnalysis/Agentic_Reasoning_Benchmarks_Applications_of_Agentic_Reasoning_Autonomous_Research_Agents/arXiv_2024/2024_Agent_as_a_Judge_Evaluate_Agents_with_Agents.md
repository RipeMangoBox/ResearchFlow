---
title: "Agent-as-a-Judge: Evaluate Agents with Agents"
venue: arXiv
year: 2024
tags:
  - Evaluation
  - task/agent-evaluation
  - task/code-generation
  - agentic-evaluator
  - graph-based-evaluation
  - requirement-dag
  - dataset/DevAI
  - opensource/full
core_operator: 用具备检索与读写感知能力的评审代理，对层级需求驱动的开发过程做逐项核验，产出过程级反馈
primary_logic: |
  长程代码代理评测目标 → DevAI 构造真实任务与层级需求/DAG 标注 → 评审代理在工作区与可选轨迹上收集证据并逐条判定 requirement → 输出依赖感知评分、对齐率与能力边界
claims:
  - "在 DevAI 的两种评测设置和 3 个被评代码代理共 6 组对比中，Agent-as-a-Judge 的 alignment rate 均高于对应的 LLM-as-a-Judge（83.88%–92.07% vs. 60.38%–84.15%）[evidence: comparison]"
  - "在 OpenHands 的组件消融中，Agent-as-a-Judge 从仅用 ask 的 65.03% 对齐率，逐步加入 graph、read、locate 后提升到 90.44%，说明证据定位与读取是主要增益来源 [evidence: ablation]"
  - "按 15 美元/小时估算，3 位人工专家完成 DevAI 全量评测约需 1297.50 美元与 86.5 小时；Agent-as-a-Judge 仅需 30.58 美元与 118.43 分钟 [evidence: comparison]"
related_work_position:
  extends: "LLM-as-a-Judge (Zheng et al. 2024)"
  competes_with: "LLM-as-a-Judge (Zheng et al. 2024)"
  complementary_to: "Process Reward Models / Let's Verify Step by Step (Lightman et al. 2023); Automated Design of Agentic Systems (Hu et al. 2024)"
evidence_strength: moderate
pdf_ref: paperPDFs/Agentic_Reasoning_Benchmarks_Applications_of_Agentic_Reasoning_Autonomous_Research_Agents/arXiv_2024/2024_Agent_as_a_Judge_Evaluate_Agents_with_Agents.pdf
category: Evaluation
---

# Agent-as-a-Judge: Evaluate Agents with Agents

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2410.10934), [Project](https://github.com/metauto-ai/agent-as-a-judge), [Dataset](https://huggingface.co/devai-benchmark)
> - **Summary**: 这篇工作把 agent 评测从“只看最终结果的单轮打分”升级为“由评审 agent 主动检查工作区与轨迹、按层级需求逐项核验”的过程级评测，并用 DevAI 证明这种方式更接近人工共识。
> - **Key Performance**: 论文摘要将结果概括为与人工共识约 90% 对齐，而 LLM-as-a-Judge 约 70%；全量评测成本/时间仅为人工的 2.29% / 2.36%。

> [!info] **Agent Summary**
> - **task_path**: DevAI 用户查询 + 层级 requirements + 工作区/可选轨迹 -> requirement 满足判断与任务级评分
> - **bottleneck**: 现有 agent 评测只看终局结果或依赖高成本人工，缺少对中间开发过程的可扩展反馈
> - **mechanism_delta**: 把 judge 从单次 LLM 打分器升级为能构图、定位、读取和提问的 agent，用证据搜集替代主观猜测
> - **evidence_signal**: 6 组对比里 Agent-as-a-Judge 的对齐率全部高于 LLM-as-a-Judge，且多数结果接近 90%
> - **reusable_ops**: [层级 requirement DAG 标注, graph+locate+read+ask 证据收集循环]
> - **failure_modes**: [memory/planning 会放大历史误判, 小型工作区下 search/retrieve 收益有限]
> - **open_questions**: [能否泛化到非代码代理任务, 能否把 judge 信号蒸馏为稳定的过程奖励模型]

## Part I：问题与挑战

**What/Why**：这篇论文抓住的真正瓶颈，不是“缺一个 benchmark 分数”，而是**agent 系统缺少能覆盖全过程、又负担得起的评测机制**。

当前 agent 评测大致有两类问题：

1. **只看最终结果**  
   像 HumanEval、MBPP、SWE-Bench 这一类评测，更接近“最后交卷看对不对”。这对长程 agent 很不友好，因为 agent 的失败往往发生在中间：没找到正确文件、读错博客、没跑起来代码、输出落错目录、或依赖关系没满足。  
   只看 solve rate，会把这些失败模式全部压扁成一个 0/1 结果。

2. **人工细查太贵且不稳定**  
   如果改成像人类开发者那样逐步检查，确实更接近真实评审，但代价很高。论文里 3 位专家做 DevAI 人工评测，一共花了 **86.5 小时**；而且单个评审之间还有 **10%–30%** 的分歧。

作者因此提出：**既然被评对象是 agent，就让 judge 也升级为 agent。**

### 输入/输出接口

- **输入**：
  - 用户开发需求
  - 层级 requirement / preference 标注
  - 被评 agent 产生的 workspace
  - 可选 trajectory（灰盒设置）
- **输出**：
  - 每条 requirement 是否满足
  - 考虑依赖关系后的 requirement 满足率
  - task solve rate
  - 不同 agent 的优劣比较与失败定位

### 边界条件

- 论文当前是一个 **proof-of-concept**，主域只覆盖**代码生成/AI 开发代理**。
- DevAI 包含 **55 个真实但相对小规模** 的 AI 开发任务，目的是保证评测可运行、可复现、成本可控。
- 被评对象是 3 个流行开源代码 agent：**MetaGPT、GPT-Pilot、OpenHands**，统一后端为 **gpt-4o-2024-05-13**，每个任务限时 **1800 秒**。

---

## Part II：方法与洞察

### 方法骨架

这篇论文实际上做了两件事，而且二者是绑定的：

1. **先造出可过程评测的数据集：DevAI**
   - 55 个真实 AI 开发任务
   - 365 条 requirement
   - 125 条 preference
   - requirement 之间用 **DAG 依赖关系**组织  
   
   这一步很关键，因为没有 requirement 级标注，就不可能做细粒度 judge。

2. **再把 judge 设计成一个会找证据的 agent**
   初始设计有 8 个模块：graph、locate、read、search、retrieve、ask、memory、planning。  
   经过消融后，作者发现最有效的核心组合是：
   - **graph**：构建项目结构图
   - **locate**：定位 requirement 对应文件/目录
   - **read**：读取代码、图像、文档等多格式内容
   - **retrieve**：从长轨迹里抽相关片段
   - **ask**：根据证据做 requirement 判定

3. **提供两种评测协议**
   - **black-box**：只看 workspace
   - **gray-box**：workspace + 人工采集 trajectory

这意味着评测不再是“把一堆结果文本塞给 LLM 问它哪个好”，而是“让 judge agent 像人类审查员一样主动查证”。

### 核心直觉

作者真正拧动的因果旋钮是：

**从“直接打分”改成“先找证据，再判定 requirement”**。

这带来三层变化：

1. **评测目标被拆细了**  
   从一个稀疏的最终成功率，变成一组局部、二元、可核验的 requirement。  
   这改变了评测的**稀疏奖励瓶颈**。

2. **评测上下文被对齐了**  
   graph + locate + read 让 judge 不再在整份工作区和长轨迹里盲猜，而是围绕当前 requirement 找最相关证据。  
   这改变了评测的**信息检索瓶颈**。

3. **评测决策更 grounded 了**  
   ask 模块是在已定位证据上做判断，而不是让 LLM 凭全局印象给分。  
   这改变了评测的**主观性/幻觉瓶颈**。

所以它之所以有效，不是因为“judge 也用了 agent 这个更大词”，而是因为它把评测变成了一个**受约束的证据搜集问题**。

更有意思的是，论文还发现：**judge 不是越复杂越好**。  
memory 和 planning 本来听起来更“agentic”，但在这里反而容易引入噪声：
- memory 会把历史误判传递到后续 requirement
- planning 会让流程更不稳定
- search 在当前小型工作区下收益不明显

这说明评测 agent 的核心不是强推理，而是**低噪声、可验证的证据管线**。

### 战略权衡

| 设计选择 | 改变了什么 | 带来的能力 | 代价/风险 |
|---|---|---|---|
| requirement + dependency DAG | 把终局评分拆成中间里程碑 | 能定位 agent 卡在哪一步 | 标注成本高，任务设计偏领域化 |
| graph + locate + read | 把“全局盲看”变成“定点查证” | 更接近人工逐项检查 | 需要稳定的文件解析与多格式读取 |
| gray-box 引入 trajectory | 让中间行为可观测 | 能更好解释失败原因 | 真实部署中轨迹未必可得 |
| 去掉 memory/planning | 降低误差传播 | judge 更稳定 | 牺牲了更复杂的自适应评审潜力 |

---

## Part III：证据与局限

**So what**：相对 prior work，这篇论文的能力跃迁不在于“又提出了一个新分数”，而在于它能以较低成本逼近**多位人工达成共识后的判断**，并给出 requirement 级失败定位。

### 关键证据信号

- **比较信号：Agent-as-a-Judge 全面优于 LLM-as-a-Judge**  
  在 3 个代码 agent、两种设置共 6 组对比里，Agent-as-a-Judge 的 alignment rate 全部高于 LLM-as-a-Judge。  
  这支持核心论点：**主动搜证**比**单轮主观打分**更适合评 agent。

- **分析信号：人工自己也不稳定**  
  三位人工评审两两分歧率达到 **10%–30%**。  
  这说明论文并不是拿 AI 去替代一个“完美人工真值”，而是在逼近**人工共识**。这个 framing 很重要，也更现实。

- **消融信号：最大增益来自证据定位，不是更复杂规划**  
  仅 ask 时对齐率 **65.03%**；加入 graph 后到 **75.95%**；再加 read 到 **82.24%**；加 locate 后到 **90.44%**。  
  结论很清楚：**judge 的关键在找到对的证据，不在想得更花。**

- **成本信号：过程评测终于变得可扩展**  
  人工全量评测约 **1297.50 美元 / 86.5 小时**；Agent-as-a-Judge 约 **30.58 美元 / 118.43 分钟**。  
  这让“过程级评测”第一次有了规模化可能。

### 1-2 个最值得记住的指标

- **对齐度**：论文摘要给出的结论是，Agent-as-a-Judge 与人工共识约 **90%** 对齐，LLM-as-a-Judge 约 **70%**。
- **成本**：Agent-as-a-Judge 的评测成本/时间仅为人工的 **2.29% / 2.36%**。

### 局限性

- **Fails when**: 任务超出当前代码开发域、需要真实线上部署验证、涉及更大规模代码库或更复杂外部交互时，当前基于 workspace/有限轨迹的 judge 可能漏证据；memory/planning 在当前实现下还会引入不稳定性。  
- **Assumes**: 任务已被人工拆成清晰、二元、可核验的 requirement，并带有依赖 DAG；被评 agent 会把关键产物保存在 workspace；灰盒模式还假设能获得细粒度 trajectory；评测依赖闭源 API（如 GPT-4o）和一定工程化读取工具。  
- **Not designed for**: 开放式创造力评价、纯偏好型/审美型判断、无显式工件的策略 agent 任务，也没有证明可以直接泛化到 web agent、GUI agent 或 embodied agent。  

这里还有一个很现实的复现约束：虽然代码与数据集是开源的，但**被评 agent 与 judge 的实际表现仍受闭源 LLM API、提示词实现、轨迹采集方式**影响。

### 可复用组件

- **层级 requirement + dependency DAG**：适合把长程任务拆成可诊断的过程监督。
- **workspace graph + locate/read/ask**：一种通用的“证据先行”评测模板。
- **black-box / gray-box 双协议**：适合按可观测性分别报告结果。
- **judge 输出作为过程奖励**：可直接接到 PRM、agent self-improvement 或自动化 workflow 优化里。

## Local PDF reference

![[paperPDFs/Agentic_Reasoning_Benchmarks_Applications_of_Agentic_Reasoning_Autonomous_Research_Agents/arXiv_2024/2024_Agent_as_a_Judge_Evaluate_Agents_with_Agents.pdf]]