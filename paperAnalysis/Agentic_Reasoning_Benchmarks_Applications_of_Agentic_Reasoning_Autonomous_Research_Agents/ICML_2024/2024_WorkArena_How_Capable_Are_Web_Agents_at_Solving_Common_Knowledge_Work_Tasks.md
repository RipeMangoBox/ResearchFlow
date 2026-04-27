---
title: "WorkArena: How Capable Are Web Agents at Solving Common Knowledge Work Tasks"
venue: ICML
year: 2024
tags:
  - Survey_Benchmark
  - task/MLLM-evaluation
  - multimodal-observation
  - oracle-validation
  - chat-based-interaction
  - dataset/WorkArena
  - opensource/full
core_operator: 以 ServiceNow 企业软件为载体构建可自动验证的 33 类网页办公任务，并用统一的 BrowserGym 接口评测 Web Agent 的真实知识工作能力
primary_logic: |
  评测目标（企业知识工作自动化） → 在 ServiceNow 上构造带自然语言目标、验证器与 oracle 的 33 类任务 → 用 BrowserGym 提供统一的多模态观测、动作与聊天接口运行代理 → 输出任务成功率与可诊断的失败边界
claims:
  - "WorkArena在ServiceNow上提供33个任务、19,912个实例，并为任务实现自动验证与手写oracle，从而支持可复现的企业网页代理评测 [evidence: analysis]"
  - "在WorkArena上，GPT-4o代理达到42.7%成功率，显著高于Llama3-70B的17.9%和GPT-3.5的6.1% [evidence: comparison]"
  - "所有被测LLM在6个list-filter任务上均为0%成功率，而GPT-4o在知识库搜索与服务目录任务上分别达到80.0%和77.8%，说明主要瓶颈集中在非标准企业UI交互而非指令理解本身 [evidence: analysis]"
related_work_position:
  extends: "WebArena (Zhou et al. 2023)"
  competes_with: "WebArena (Zhou et al. 2023); Mind2Web (Deng et al. 2023)"
  complementary_to: "MiniWoB (Liu et al. 2018); Android in the Wild (Rawles et al. 2023)"
evidence_strength: moderate
pdf_ref: "paperPDFs/Agentic_Reasoning_Benchmarks_Applications_of_Agentic_Reasoning_Autonomous_Research_Agents/ICML_2024/2024_WorkArena_How_Capable_Are_Web_Agents_at_Solving_Common_Knowledge_Work_Tasks.pdf"
category: Survey_Benchmark
---

# WorkArena: How Capable Are Web Agents at Solving Common Knowledge Work Tasks

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2403.07718) · [WorkArena](https://github.com/ServiceNow/WorkArena) · [BrowserGym](https://github.com/ServiceNow/BrowserGym) · [AgentLab](https://github.com/ServiceNow/AgentLab)
> - **Summary**: 该文把 Web Agent 评测从玩具网页/消费网站推进到真实企业软件 ServiceNow，通过可验证任务与统一浏览器环境测出：现有代理离“真正替人做办公软件操作”仍有明显距离。
> - **Key Performance**: GPT-4o 在 WorkArena 上仅 42.7% 成功率；同类 BrowserGym+GPT-4o 代理在 WebArena 上达 23.5%，高于原始 WebArena 论文报告的 14.4%。

> [!info] **Agent Summary**
> - **task_path**: 自然语言工作指令 + 浏览器多模态观测(AXTree/HTML/截图/聊天) -> 浏览器动作或聊天答案
> - **bottleneck**: 企业软件中的超长DOM、非标准组件、动态规则与复杂工作流，使现有网页代理评测低估真实自动化难度
> - **mechanism_delta**: 用 ServiceNow 真实任务 + validator/oracle + BrowserGym 统一接口，把“网页点击能力”升级为“企业知识工作微流程能力”的可诊断评测
> - **evidence_signal**: 跨模型比较与任务分解最强：GPT-4o 在 WorkArena 仅42.7%，且所有模型在 list-filter 任务上均为0%
> - **reusable_ops**: [validator+oracle task design, augmented AXTree with bid/coords]
> - **failure_modes**: [non-standard list widgets, prompt truncation under large DOM/AXTree]
> - **open_questions**: [如何扩展到更长程复合工作流, 为什么截图/视觉增强对真实网页代理帮助有限]

## Part I：问题与挑战

**What / Why：真正的瓶颈是什么，为什么现在要解决？**

现有 web-agent 评测有两个典型盲点：

1. **环境不够“像工作”**：  
   MiniWoB 这类 benchmark 偏 toy；WebArena 更真实，但主要是消费类网站或通用网站。它们能测“会不会浏览网页”，却不一定能测“能不能替知识工作者完成企业软件上的重复任务”。

2. **评测不够“可诊断”**：  
   如果环境本身缺少稳定验证、任务可行性保证、统一观测/动作接口，那么模型失败时很难判断：到底是推理差、上下文不够、UI 太怪，还是评分方式不稳。

这篇论文要解决的真实瓶颈，不是先提出一个更强的新 agent，而是先搭建一个**更接近真实办公软件、又能稳定量化能力边界**的测试床。

### 任务接口与边界

- **输入**：用户在聊天框中给出非常明确的自然语言目标。
- **观测**：浏览器当前页面的可访问性树（AXTree）、HTML、截图，以及增强属性（元素 bid、坐标、visible/clickable 标记）。
- **输出**：代理执行浏览器动作（点击、输入、坐标操作、Python/Playwright 操作）或在聊天框回复答案。
- **评测对象**：33 个 ServiceNow 任务，19,912 个实例，覆盖菜单导航、列表筛选/排序、表单填写、知识库检索、服务目录下单、仪表盘读取。
- **任务边界**：目标描述通常**显式且不含歧义**；多数任务是短程、单目标微流程，而不是开放式、长期协作型工作流。

### 为什么 WorkArena 难

ServiceNow 这类企业软件带来的难点并不在“语义理解”本身，而在**界面分布**：

- **超长上下文**：页面 HTML 清洗后仍可达 40k–500k tokens，远超普通 agent 的舒适区。
- **非标准 UI / HTML**：嵌套 iFrame、shadow DOM、专有标签、专有 JS API。
- **动态交互规则**：字段显隐随状态变化、隐藏菜单、复杂 date picker、自动补全等。
- **看似简单、实则交互脆弱**：例如列表筛选，语义非常简单，但 UI 控件极不标准，导致所有被测模型直接 0%。

一句话：**难点是“真实企业 UI 的交互分布 shift”，不是“任务说明写得多复杂”。**

## Part II：方法与洞察

这篇论文的核心贡献由两部分组成：

1. **WorkArena**：一个面向企业知识工作的 web-agent benchmark  
2. **BrowserGym**：一个统一的浏览器代理环境，用来承载和比较不同 benchmark / agent 设计

### WorkArena：把“办公微流程”拆成可验证任务

WorkArena 在 ServiceNow 上实现了 33 类任务，分成 6 大类：

- **Lists**：筛选、排序
- **Forms**：创建记录、填写复杂表单
- **Knowledge Bases**：搜索知识库并回答问题
- **Service Catalogs**：按规格下单
- **Dashboards**：读取图表数值并做简单推理
- **Menus**：菜单导航、用户 impersonation

关键不只是“任务种类多”，而是每类任务都带了两种保障：

- **validation function**：自动判断当前任务是否完成、是否填错、是否写入错误数据
- **oracle function / cheat()**：用 Playwright 写死一条正确解  
  作用是：
  - 保证任务确实可做
  - 给有学习能力的 agent 提供 ground truth
  - 便于后续平台更新时维护 benchmark

这让它比很多只靠末端文本比对的 benchmark 更像一个**工程可维护的评测系统**。

### BrowserGym：统一观测、动作、聊天与多页支持

BrowserGym 提供的不是某个特定 agent，而是一套通用环境：

- **聊天接口**：支持 user-agent 消息交互，适合问答型和分步指令型任务
- **丰富观测**：HTML、AXTree、截图、报错信息、页面列表
- **增强属性**：每个元素有 bid、bbox、visible、clickable
- **丰富动作**：bid-based、coord-based、高层 primitive、甚至任意 Python/Playwright
- **多页/复杂页面支持**：tab、popup、iframe、shadow DOM

因此，作者能在同一框架内比较：
- 纯文本 agent
- 视觉增强 agent
- 带/不带 history 的 agent
- 不同动作空间设计

### 核心直觉

作者真正改变的不是某个“更强模型参数”，而是**测量装置**本身。

**What changed**  
从 toy/消费网页评测，切换到企业软件 ServiceNow；从单一网页状态，切换到 chat + 多模态观测 + 丰富动作；从弱验证，切换到 validator + oracle 的强验证。

**Which bottleneck changed**  
这改变了评测输入分布与测量可靠性：

- 输入分布变成了：长上下文、非标准 DOM、复杂控件、真实工作流
- 评分可靠性变成了：可自动验证、可维护、可确认任务可行

**What capability changed**  
于是 benchmark 不再只测“能不能点中按钮”，而能测：

- 是否能完成真实知识工作中的微流程
- 失败是出在语言理解、UI grounding、上下文长度、还是动作设计
- 不同模型/观测/动作设计在真实企业软件上差在哪里

### 为什么这个设计有效

因果上，作者做对了三件事：

1. **把任务语义写得足够明确**  
   这样失败更大概率来自 UI 与推理，而不是指令歧义。

2. **把成功判定做成程序化验证**  
   这样结果更可信，也能细粒度发现“差一点但没完成”与“写错数据库”的区别。

3. **把环境接口统一**  
   这样对比不同 LLM、动作空间、视觉输入时，不容易把环境差异误当模型差异。

### 战略权衡

| 设计选择 | 带来的能力 | 代价 / 权衡 |
|---|---|---|
| 直接基于 ServiceNow 真实企业软件构建任务 | 评测更贴近知识工作场景，能暴露真实 UI 难点 | 平台偏置明显，结论未必直接外推到所有企业软件 |
| 使用显式模板化目标 | 降低指令歧义，隔离“交互能力” | 与真实用户的含糊、多轮、上下文依赖请求仍有距离 |
| validator + oracle | 可复现、可维护、可确保任务可行 | 需要较高工程投入，且维护依赖平台稳定性 |
| AXTree/截图/bid/coords 等丰富接口 | 便于系统化比较文本、视觉、动作设计 | 提示词变长，弱模型更容易被信息淹没 |
| 支持任意 Python/Playwright | 给 agent 最大交互自由度 | 也提高了动作空间复杂度与安全/可控性问题 |

## Part III：证据与局限

### 关键证据

**1. 比较信号：WorkArena 显著放大了模型能力差距**  
在 WorkArena 上，GPT-4o 达到 **42.7%** 成功率，Llama3-70B 为 **17.9%**，GPT-3.5 仅 **6.1%**。  
这说明企业网页任务比 MiniWoB 更能拉开 frontier model 与较弱模型的差距，适合作为能力上限测试。

**2. 任务剖面信号：失败集中在“怪 UI”，不是“看不懂任务”**  
GPT-4o 在：
- 知识库搜索上达到 **80.0%**
- 服务目录任务达到 **77.8%**
- 但在 6 个 list-filter 任务上是 **0%**

这说明当任务目标足够明确时，模型并非完全不懂“要做什么”；真正卡住它的是**非标准企业 UI 控件**。

**3. 消融信号：推理 scaffold 比朴素多模态更重要**  
- 去掉 chain-of-thought，会明显伤害表现；例如 Llama3 在 WorkArena 从 **20.0%** 掉到 **8.5%**
- GPT-4o-V 相比文本版 GPT-4o 总体提升很小（WorkArena **41.8% vs 42.7%**）
- 加太多额外描述/history，反而会让较弱模型更差

结论很明确：  
**当前真实网页代理的瓶颈更像是“推理与长上下文控制”，而不是“只要加截图就行”。**

**4. 跨 benchmark 信号：BrowserGym 本身也提升了可比性与上限**  
同类 GPT-4o agent 在 WebArena 上达到 **23.5%**，高于原始论文中的 **14.4%**。  
这说明统一环境、丰富动作/观测设计并不只是“包装层”，它会实质影响 agent 上限。

### 局限性

- **Fails when**: 遇到非标准列表控件、超长 DOM/AXTree 需要截断、或任务需要超过 15 步的长程规划时，当前代理与实验协议都会明显吃亏；WorkArena 目前对这类长流程的覆盖仍有限。
- **Assumes**: 任务指令显式且信息充分；存在手写 `validate()` 与 `cheat()`；知识库问答样本部分由 GPT 生成；最强基线依赖闭源 GPT-4o API，而开源 Llama3-70B 评测依赖 4×A100 GPU；BrowserGym 自身不提供记忆机制，agent 需自行处理历史。
- **Not designed for**: 含糊需求、多轮协商、跨多个企业系统的真实复合流程、安全/对抗性网页评测、以及完整的人机协作式 shared-control 场景。

### 可复用组件

这篇论文最值得复用的，不只是 WorkArena 数据本身，而是三类“评测操作符”：

1. **validator + oracle 任务设计范式**  
   很适合做任何可执行代理 benchmark。

2. **增强 AXTree 表示**  
   给元素绑定 bid / 坐标 / visible / clickable，兼顾文本可读性和动作可执行性。

3. **统一 browser environment**  
   把 chat、错误回传、多页、多模态观测、不同动作空间统一在同一框架内，便于做系统级 ablation。

## Local PDF reference

![[paperPDFs/Agentic_Reasoning_Benchmarks_Applications_of_Agentic_Reasoning_Autonomous_Research_Agents/ICML_2024/2024_WorkArena_How_Capable_Are_Web_Agents_at_Solving_Common_Knowledge_Work_Tasks.pdf]]