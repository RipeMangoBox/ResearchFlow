---
title: "WorkArena++: Towards Agents that Act Like Employees"
venue: NeurIPS
year: 2024
tags:
  - Survey_Benchmark
  - task/web-agent-evaluation
  - compositional-task-design
  - oracle-validation
  - curriculum-sampling
  - dataset/WorkArena++
  - dataset/WorkArena
  - opensource/full
core_operator: 通过组合可认证的原子网页任务，构造带显式/隐式指令层级的企业知识工作基准，并用 oracle/validator 自动验收
primary_logic: |
  企业知识工作评测目标 → 组合 WorkArena L1 原子任务形成 L2/L3 工作流，并把步骤部分隐藏到 ticket/知识库中 → 用 oracle/validator 与种子化 curriculum 自动评分 → 揭示 web agents 在规划、检索、记忆与推理上的能力边界
claims:
  - "WorkArena++ 将 WorkArena 从 33 个基础企业网页任务扩展到 682 个组合式工作流任务，并覆盖规划、检索、数据驱动推理、记忆与不可行判断五类能力 [evidence: analysis]"
  - "在标准化评测中，GPT-4o-v 在 WorkArena++ L2 仅取得 3.8% 成功率，所有基线模型在 L3 上均为 0%，说明当前 SOTA web agents 难以完成隐式企业流程 [evidence: comparison]"
  - "15 名人类参与者在 98 个任务实例上达到 93.9% 成功率，而同一子集上的 GPT-4o 仅为 2.1%，表明该基准对人类可解但对现有模型极具挑战 [evidence: comparison]"
related_work_position:
  extends: "WorkArena (Drouin et al. 2024)"
  competes_with: "WebArena (Zhou et al. 2023); OSWorld (Xie et al. 2024)"
  complementary_to: "Mind2Web (Deng et al. 2023); WebLINX (Lù et al. 2024)"
evidence_strength: moderate
pdf_ref: "paperPDFs/Agentic_Reasoning_Benchmarks_Applications_of_Agentic_Reasoning_Autonomous_Research_Agents/arXiv_2024/2024_WorkArena++_Towards_Agents_that_Act_Like_Employees.pdf"
category: Survey_Benchmark
---

# WorkArena++: Towards Agents that Act Like Employees

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2407.05291) · [Code](https://github.com/ServiceNow/WorkArena)
> - **Summary**: 论文把企业软件中的原子 UI 操作评测，升级为可认证的组合式知识工作流评测，用来检验代理是否真的能像员工一样读工单、查知识库、做推理并完成闭环任务。
> - **Key Performance**: 人类子集成功率 93.9%，同子集 GPT-4o 仅 2.1%；GPT-4o-v 在 WorkArena++ L2 全集仅 3.8%，L3 为 0%

> [!info] **Agent Summary**
> - **task_path**: 工单/自然语言目标 + 浏览器观测(AXTree/截图) -> 网页动作序列 / infeasible 报告 / 聊天答复
> - **bottleneck**: 组合式企业任务中的跨页检索、长期记忆与隐式规程规划
> - **mechanism_delta**: 用可组合原子任务构造 L2/L3 工作流，并在 L3 中把执行步骤隐藏到工单与知识库里
> - **evidence_signal**: GPT-4o 在 WorkArena L1 为 42.7%，到 WorkArena++ L2/L3 降为 3.0%/0.0%；人类仍有 93.9%
> - **reusable_ops**: [oracle-composition, validator-certification]
> - **failure_modes**: [action-hallucination, exploration-stall]
> - **open_questions**: [trace-based-finetuning, cross-software-generalization]

## Part I：问题与挑战

这篇论文真正要补的，不是“模型会不会点按钮”，而是“模型能不能像企业员工那样完成一个完整工作流”。

现有 web-agent 基准各有缺口：
- **MiniWoB** 更像玩具环境；
- **WebArena** 更偏通用消费级网站；
- **WorkArena L1** 虽然进入企业软件场景，但多数仍是**原子型、短程、显式**任务。

而真实知识工作通常是：
**读工单 → 去知识库找规程 → 从列表/仪表盘取值 → 做逻辑/算术决策 → 修改系统状态 → 关闭工单**。

所以真瓶颈是：
1. **长时程规划**：不是一步对，而是整条流程都不能掉链子；
2. **跨页面信息整合**：需求、规则、数据分散在 ticket、KB、dashboard、form 中；
3. **记忆与状态跟踪**：中间读到的信息后面还要用；
4. **可验证完成**：不能靠模型自报完成，必须检查数据库和前端状态。

为什么现在要做这件事？因为 LLM agent 已经在 CoT / ReAct 一类范式下表现出“会推理”的迹象，但企业自动化真正高价值的场景，恰恰缺少一个**既真实、又可重复、还可认证判分**的 benchmark。

**输入/输出接口**也很清晰：
- **输入**：自然语言目标或工单、BrowserGym 观测（AXTree、截图等）、历史轨迹；
- **输出**：网页高层动作、向用户发消息、或声明任务不可行；
- **边界条件**：单一平台 ServiceNow、最多 50 步、任务终态需可认证。

## Part II：方法与洞察

WorkArena++ 的核心不是新模型，而是**新测量设计**。

### 1) 任务如何构造
作者把 WorkArena 的 L1 原子任务当成积木，组合成 **341 个工作流**，再各自提供两种难度：
- **L2**：显式给步骤，但不告诉你具体怎么点；
- **L3**：只给工单，具体规程藏在知识库里，代理必须自己找、自己记、自己规划。

最终得到 **682 个任务**。

### 2) 在测什么能力
任务覆盖 5 类能力：
- 规划与问题求解
- 信息检索
- 数据驱动决策与推理
- 复杂记忆
- 上下文理解（含不可行任务判断）

这比“填表单/点菜单”更接近员工日常。

### 3) 为什么这个 benchmark 更稳
它不只增加任务量，还强化了评测基础设施：
- **oracle + validator**：每个任务都能自动验证是否真的完成；
- **标准化 curriculum**：用种子采样评测实例，提升可复现性；
- **用户级 sandbox**：每个任务新建账号，避免任务间互相污染；
- **10 个虚构品牌主题**：增加视觉多样性；
- **可抽取 observation-action traces**：可直接产出微调数据。

### 核心直觉

作者真正拧动的因果旋钮，是**任务分布**而不是模型结构。

从：
- 短程
- 显式
- 单页面
- 只考 UI grounding

变成：
- 长程
- 部分隐式
- 跨页面/跨信息源
- 需要把“规则 + 数据 + 当前状态”合成为计划

这会把测量瓶颈从“能否定位元素”转移到：
**能否持续理解目标、检索规则、记住关键信息、判断动作后果，并把整条流程执行完**。

也因此，WorkArena++ 测到的是“像员工一样完成工作”，而不只是“像脚本一样操作网页”。

### 战略取舍

| 设计选择 | 带来的能力诊断 | 代价/折衷 |
|---|---|---|
| L2/L3 双层目标 | 区分“会执行显式步骤”与“会自主找规则并规划” | L3 更依赖记忆与产品熟悉度 |
| 组合式工作流 | 从原子交互升级到闭环任务完成 | oracle/validator 编写成本更高 |
| 单平台 ServiceNow | 环境稳定、真实、可认证 | 跨软件泛化覆盖有限 |
| 种子化 curriculum + sandbox | 可复现、可并行、公平比较 | 与真实共享状态协作仍有距离 |

## Part III：证据与局限

### 关键证据

**信号 1：从 L1 到 WorkArena++ 的断崖式掉点。**  
GPT-4o 在 WorkArena L1 上有 **42.7%**，但到了 WorkArena++ 只剩 **L2: 3.0% / L3: 0.0%**。  
这说明先前 benchmark 上的“会做网页任务”，并不等于能完成企业知识工作流。

**信号 2：基准不是不可做，而是模型不会做。**  
在 98 个任务实例的人类评测子集上，人类达到 **93.9%**，同子集 GPT-4o 只有 **2.1%**。  
所以这个 benchmark 的难点主要来自 agent 能力缺失，而不是任务定义本身不合理。

**信号 3：错误模式很集中。**  
作者的 trace 分析显示失败主要来自：
- 检索值读错；
- 不会主动探索隐藏标签/折叠菜单；
- 幻觉出不存在的动作或按钮；
- 误判自己已经完成子任务；
- 卡进重复无效动作循环。  
这表明问题不是单纯 UI 操作差，而是**目标理解、状态跟踪、后果评估**都不稳。

### 局限性

- **Fails when**: 需要跨软件/跨 OS 交互、真实多人协作、安全与对抗测试、或开放式桌面控制时，WorkArena++ 覆盖不足；L3 虽更真实，但仍是有限 protocol 驱动的隐式任务。
- **Assumes**: 任务必须存在可脚本化 oracle 与可验证终态；依赖 ServiceNow Personal Developer Instances、BrowserGym 接口与标准化实例配置；复现实验中的强基线部分依赖闭源 API，开源基线使用 4×A100 GPU。
- **Not designed for**: 安全红队、恶意 prompt 鲁棒性、跨应用办公流、组织级长期记忆、以及无明确验收终态的开放任务。

### 可复用组件

可直接复用的设计资产包括：
- **oracle/validator 组合框架**
- **种子化 curriculum 采样器**
- **轨迹抽取器**（observation-action traces）
- **任务隔离机制**
- **主题随机化的视觉多样性方案**

## Local PDF reference

![[paperPDFs/Agentic_Reasoning_Benchmarks_Applications_of_Agentic_Reasoning_Autonomous_Research_Agents/arXiv_2024/2024_WorkArena++_Towards_Agents_that_Act_Like_Employees.pdf]]