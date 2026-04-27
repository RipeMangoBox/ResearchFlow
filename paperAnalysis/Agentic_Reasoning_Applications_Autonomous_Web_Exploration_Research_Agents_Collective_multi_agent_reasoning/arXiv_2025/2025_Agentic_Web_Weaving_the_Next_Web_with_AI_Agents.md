---
title: "Agentic Web: Weaving the Next Web with AI Agents"
venue: arXiv
year: 2025
tags:
  - Survey_Benchmark
  - task/autonomous-web-interaction
  - multi-agent-systems
  - agent-protocols
  - opensource/partial
core_operator: "用“智能—交互—经济”三维框架重组面向代理的下一代 Web，并把协议、系统、经济与治理问题放到统一视角下分析"
primary_logic: |
  Web 演化脉络（PC → Mobile → Agentic） → 以智能/交互/经济三维框架梳理算法、协议与系统架构转移 → 总结应用、风险治理与开放研究问题
claims:
  - "Claim 1: 论文将 Agentic Web 定义为由自治软件代理在分布式互联网中持续规划、协调并执行任务的范式，并用智能、交互、经济三维框架统一其核心组成 [evidence: synthesis]"
  - "Claim 2: 论文综合搜索→推荐→行动三代 Web 范式转移，主张未来商业逻辑将从争夺人类点击/停留时间转向争夺代理调用与能力选择 [evidence: synthesis]"
  - "Claim 3: 论文归纳出 Agentic Web 落地所需的关键系统条件，包括机器可读接口、语义化代理协议、动态服务发现、跨代理编排与计费治理 [evidence: synthesis]"
related_work_position:
  extends: "N/A"
  competes_with: "N/A"
  complementary_to: "N/A"
evidence_strength: weak
pdf_ref: paperPDFs/Agentic_Reasoning_Applications_Autonomous_Web_Exploration_Research_Agents_Collective_multi_agent_reasoning/arXiv_2025/2025_Agentic_Web_Weaving_the_Next_Web_with_AI_Agents.pdf
category: Survey_Benchmark
---

# Agentic Web: Weaving the Next Web with AI Agents

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2507.21206), [GitHub](https://github.com/SafeRL-Lab/agentic-web)
> - **Summary**: 该文不是提出单一新模型，而是把下一代互联网定义为由 AI 代理主导的“Agentic Web”，并用“智能—交互—经济”三维框架系统化解释其算法、协议、系统、应用与治理。
> - **Key Performance**: 非 benchmark 论文；核心结构化产出是 **3 个概念维度** + **3 类算法转移**，并以 Microsoft Build 2025 中 **50+ agent tools** 作为产业落地信号。

> [!info] **Agent Summary**
> - **task_path**: 用户高层意图 / 开放式异构 Web 服务环境 -> 多代理协调执行结果（报告、交易、预订等）
> - **bottleneck**: 现有 Web 是为人类点击、页面浏览和无状态请求设计的，缺少面向代理的语义接口、服务发现、协同、计费与信任基础设施
> - **mechanism_delta**: 把 Web 的基本交互单元从“页面/链接”改成“可发现、可调用、可协作的 agent/service capability”，并用智能-交互-经济三维框架统一分析
> - **evidence_signal**: 跨历史阶段、代表性 agent 方法与工业系统信号（MCP/A2A、ChatGPT Agent、Microsoft Build 2025）的系统综合
> - **reusable_ops**: [三维概念框架, Client-Agent-Server 路线图]
> - **failure_modes**: [缺少统一定量验证, 对尚未稳定的协议与平台生态依赖强]
> - **open_questions**: [代理发现与信任如何标准化, 代理计费与责任归属如何落地]

## Part I：问题与挑战

这篇文章要回答的不是“如何把一个 agent 做得更聪明”，而是**当用户开始把整段工作流委托给 agent 时，现有 Web 为什么不够用了**。

### 1. 真正的问题是什么
传统 Web 的基本假设是：
- 人类自己搜索、点击、比较、确认；
- 服务以页面、API、App 形式分散存在；
- 交互以短时、无状态、单步请求为主。

但 Agentic Web 的目标是：
- 用户只给出**高层意图**；
- agent 负责**规划、检索、调用工具、协商、执行**；
- 任务跨多个服务、多个步骤、甚至多个代理持续展开。

所以真实瓶颈不只是模型推理能力，而是**整个 Web 缺少面向代理的执行底座**：
1. **接口瓶颈**：很多网页/API 对人类可用，但对 agent 不够语义化、机器可读。
2. **协同瓶颈**：单代理很难覆盖复杂长链任务，需要动态发现并招募其他代理或服务。
3. **系统瓶颈**：现有 HTTP/RPC 更像一次性调用，不擅长保持长期上下文、持续协商与安全执行。
4. **经济瓶颈**：未来竞争对象不再只是“谁吸引人点击”，而是“谁更容易被 agent 选中并调用”。
5. **治理瓶颈**：当 agent 能代表用户下单、支付、谈判、访问私有数据时，安全、责任、计费、监管必须前移。

### 2. 输入/输出接口
- **输入**：用户高层目标、偏好、权限、上下文约束。
- **输出**：不是单条答案，而是**完成后的任务结果**，例如报告、预订、购买、跨站流程执行结果。

### 3. 为什么是现在
作者给出的“现在性”很明确：
- LLM 已经跨过“只会聊天”阶段，开始具备规划、工具使用、长链推理能力；
- 多代理框架和 Web agent 系统已出现；
- 工业界开始提供 agent 产品与协议层信号，如 **ChatGPT Agent、MCP、A2A、NLWeb**；
- 用户也越来越愿意把“问题”升级为“任务”来委托。

一句话概括：**模型能力到了阈值，而 Web 基础设施还停留在人类浏览器时代。**

---

## Part II：方法与洞察

这篇论文的“方法”不是训练新模型，而是提出一套**组织 Agentic Web 的分析框架**，把散落的 agent 技术、协议、系统、商业和治理问题放入同一坐标系。

### 框架主线

#### 1. 三个核心前提
作者认为 Agentic Web 成立至少要满足三件事：
1. **Agent 是自治中介**：能够代表用户独立完成复杂任务；
2. **Web 资源可机器访问**：服务必须以标准化、机器可读接口开放；
3. **价值能在代理间交换**：不仅人和系统交易，agent 与 agent 也能形成经济关系。

#### 2. 三个概念维度
- **Intelligence**：感知、理解、规划、学习、反思、多模态整合；
- **Interaction**：语义协议、能力发现、工具编排、agent-to-agent 协同；
- **Economy**：agent attention economy、能力竞争、调用计费、责任与治理。

这三层是递进关系：
**智能让 agent 能思考，交互让它能接入世界，经济让它能在开放生态中长期运作。**

#### 3. 三类算法转移
作者把核心算法演化概括为三条线：

- **用户检索 → Agentic Information Retrieval**  
  从“用户发 query、系统回文档”，转向“agent 根据任务进度主动决定何时检索、检索什么、是否调用外部工具”。

- **推荐 → Agent Planning**  
  从“给用户推荐一个 item”，转向“agent 为达成目标规划多步行动”。这里代表性方法包括 ReAct、WebAgent、Plan-and-Act 等。

- **单代理 → 多代理协调**  
  从单体 agent 执行，转向按角色分工、共享上下文、共同完成复杂任务。代表性系统如 AutoGen、OWL、AutoAgent 等。

#### 4. 系统路线图
论文还提出一个面向落地的系统图景：

- **User Client**：多模态交互入口；
- **Intelligent Agent**：意图理解、规划、编排、决策中心；
- **Backend Services**：工具、API、数据库、外部代理与平台服务。

进一步，作者给出一些系统级操作件：
- **SRZ（Service Requirement Zone）**：把任务需求从单一时延/带宽扩展到成本、安全、知识、可靠性等多维约束；
- **DSVM**：把任务需求与能力向量化匹配；
- **RTR**：实时任务路由；
- **CABL**：跨代理计费账本。

这些设计说明作者关心的不只是“agent 能不能做”，而是**开放互联网中 agent 如何被稳定调度、计费和治理**。

### 核心直觉

**核心变化**：  
把 Web 的基本对象从“页面/文档”改成“能力/服务/代理”。

**改变了什么瓶颈**：  
- 过去的瓶颈是“人如何找到信息”；  
- 现在的瓶颈变成“agent 如何理解能力、发现服务、保持上下文并安全执行”。

**能力为什么会提升**：  
1. **语义接口**把“能做什么”显式化，降低 agent 调用服务的歧义；
2. **持久协议**把多步任务从碎片化 API 调用，变成连续上下文中的执行过程；
3. **多代理分工**把复杂任务拆成专长模块，提升覆盖面与可扩展性；
4. **经济/计费层**把调用、激励和资源分配闭环化，使开放生态可持续。

可用一句因果链概括：

**高层用户意图 → 语义化能力发现与协议交互 → 多代理规划与编排 → 结果导向的任务完成**

### 战略取舍

| 设计轴 | 传统 Web | Agentic Web | 得到的能力 | 付出的代价 |
|---|---|---|---|---|
| 基本对象 | 页面、文档、链接 | 能力、服务、代理 | 从“看信息”升级到“做任务” | 标准化与互操作复杂度上升 |
| 发现机制 | 搜索/推荐面向人 | 语义发现/服务注册面向 agent | 动态调用与组合能力 | 更易被排序操纵、广告干扰 |
| 执行模式 | 用户手动多步操作 | agent 持续规划与执行 | 长链任务自动化 | 失败恢复和责任追踪更难 |
| 协议层 | HTTP/RPC 为主 | MCP/A2A 等语义协议 | 上下文连续、跨代理协同 | 协议尚未成熟，生态碎片化 |
| 商业逻辑 | 点击率、停留时长 | 调用率、任务完成率 | 更结果导向的价值衡量 | 计费、清算、审计难度更高 |

---

## Part III：证据与局限

### 证据信号

这篇论文最强的证据**不是实验表格**，而是系统性综合。

1. **历史-结构信号**  
   作者把 Web 划分为 **PC / Mobile / Agentic** 三个阶段，并对比了核心范式、注意力对象、商业指标和组织方式。这个框架清楚地说明：Agentic Web 不是“把 chat 接到浏览器里”，而是一次从搜索/推荐到行动的范式切换。

2. **算法-能力信号**  
   论文把 RAG、Toolformer、ReAct、WebAgent、AutoGen、OWL 等不同路线放进统一轨道，说明今天的 agent 技术已经在检索、规划、协同三个方面分别具备雏形。

3. **协议-产业信号**  
   MCP、A2A、ChatGPT Agent、Microsoft Build 2025、NLWeb 等案例说明，这个方向已经从学术概念开始进入协议和基础设施建设阶段。  
   如果只记两个“指标”，就是：
   - **3 个概念维度**
   - **50+ agent tools 的产业整合信号**

### 局限性

- **Fails when**: 需要定量判断哪种协议、架构或编排策略最优时；或者需要用统一 benchmark 证明 Agentic Web 相比现有 Web automation 方案有明确增益时，这篇论文不给出直接证据。
- **Assumes**: LLM 已经足够可靠地进行规划、工具使用和多步执行；服务会逐步开放机器可读接口；MCP/A2A 一类协议能持续标准化；并且平台愿意提供权限、算力与计费基础设施。
- **Not designed for**: 提供可直接部署的参考实现、统一评测基准、形式化安全证明，或最终版行业标准协议。

### 复现/扩展依赖

这篇文章虽然有公开资料库，但它讨论的大量关键能力依赖于：
- **快速变化的工业协议与产品**；
- **部分闭源平台能力**，如 ChatGPT Agent、微软生态集成；
- **跨服务权限、结算与安全控制**，这些目前并未标准化。

因此，它更像一张**路线图**，而不是一套已经被完全验证的可复现实验系统。

### 可复用组件

即使没有新模型，这篇论文仍有几个很值得复用的分析部件：

- **三维框架**：智能 / 交互 / 经济  
  适合用来分析任何 agent 平台或 Web-native AI 产品。
- **三条转移主线**：检索 → 规划 → 协同  
  适合梳理技术栈演化。
- **Client-Agent-Server 架构视角**  
  适合系统设计讨论。
- **Agent Attention Economy**  
  适合分析未来 agent 搜索、排序、广告、能力市场问题。
- **SRZ 概念**  
  适合把 agent 任务的系统需求从单 KPI 扩展到多维 QoS/QoE。

## Local PDF reference

![[paperPDFs/Agentic_Reasoning_Applications_Autonomous_Web_Exploration_Research_Agents_Collective_multi_agent_reasoning/arXiv_2025/2025_Agentic_Web_Weaving_the_Next_Web_with_AI_Agents.pdf]]