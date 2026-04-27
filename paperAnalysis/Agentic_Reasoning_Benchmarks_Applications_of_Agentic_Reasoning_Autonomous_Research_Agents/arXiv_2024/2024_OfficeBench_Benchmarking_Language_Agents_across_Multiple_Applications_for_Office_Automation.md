---
title: "OfficeBench: Benchmarking Language Agents across Multiple Applications for Office Automation"
venue: arXiv
year: 2024
tags:
  - Survey_Benchmark
  - task/MLLM-evaluation
  - application-switching
  - execution-based-evaluation
  - restricted-action-space
  - dataset/OfficeBench
  - opensource/full
core_operator: 用多应用状态机与任务定制评测函数，在模拟办公环境中诊断语言代理的跨应用规划、切换与执行能力
primary_logic: |
  办公自动化评测目标 → 在 Docker 中构建含 Word/Excel/PDF/Calendar/Email 等应用的多应用环境与受限动作空间 → 为每个任务配置 exact/fuzzy/execution 定制评分 → 揭示代理在跨应用切换、长程规划与动作落地上的能力边界
claims:
  - "Claim 1: 在 300 个 OfficeBench 任务上，GPT-4 Omni 取得最高总体通过率 47.00%，但仍显著低于人类 93.33% [evidence: comparison]"
  - "Claim 2: 多应用需求显著放大了现有语言代理的能力缺口，GPT-4 Omni 从单应用 64.52% 降至三应用 21.43% [evidence: analysis]"
  - "Claim 3: 显式 switch_app 机制优于把全部操作平铺到提示词中，且同时降低 token 开销 [evidence: ablation]"
related_work_position:
  extends: "OSWorld (Xie et al. 2024)"
  competes_with: "OSWorld (Xie et al. 2024); WebArena (Zhou et al. 2023)"
  complementary_to: "ReWOO (Xu et al. 2023); LDB (Zhong et al. 2024)"
evidence_strength: moderate
pdf_ref: paperPDFs/Agentic_Reasoning_Benchmarks_Applications_of_Agentic_Reasoning_Autonomous_Research_Agents/arXiv_2024/2024_OfficeBench_Benchmarking_Language_Agents_across_Multiple_Applications_for_Office_Automation.pdf
category: Survey_Benchmark
---

# OfficeBench: Benchmarking Language Agents across Multiple Applications for Office Automation

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2407.19056); [Code](https://github.com/zlwang-cs/OfficeBench)
> - **Summary**: 该文提出 OfficeBench，用 9 个办公应用、300 个跨应用办公任务和按任务定制的评测器，系统测量语言代理在真实办公自动化中的规划、切换与执行短板。
> - **Key Performance**: GPT-4 Omni 总通过率 47.00%，人类为 93.33%；GPT-4 Omni 在三应用任务上仅 21.43%。

> [!info] **Agent Summary**
> - **task_path**: 自然语言办公指令 + 文件/邮件/日历上下文 → 跨多个办公应用的操作序列 → 最终文件/邮件/日历状态
> - **bottleneck**: 现有评测难以同时覆盖跨应用切换、长程规划、动作 grounding，以及非唯一输出的精确验收
> - **mechanism_delta**: 将办公自动化抽象为“当前应用=状态、应用内操作=转移”的状态机，并为每个任务配置 exact/fuzzy/execution 三类定制评测器
> - **evidence_signal**: 300 任务多模型对比显示最佳模型仅 47%；switch_app 消融同时提升通过率并减少 token
> - **reusable_ops**: [system-level app switching, per-task executable evaluator]
> - **failure_modes**: [redundant-operation stagnation, hallucinated or malformed actions]
> - **open_questions**: [如何扩展到真实企业软件与开放 GUI 场景, 如何通过专门训练显著缩小三应用任务差距]

## Part I：问题与挑战

### What / Why：真正的问题是什么，为什么现在要做
这篇论文的核心不是再做一个“文档理解”数据集，而是指出：**真实办公自动化的难点并不在单个文档里抽信息，而在跨多个应用完成完整工作流**。

作者认为，现有两类评测都不够：
1. **Document AI** 多聚焦 IE/QA，能测“看懂文档”，但测不了“读完文档后去改表格、发邮件、建日历事件”。
2. **Web/OS agent benchmark** 虽然测交互，但并不专门针对办公流程，也缺少对不同办公结果格式的细粒度验收。

所以真正瓶颈是三层叠加：
- **长程规划**：先读哪个应用、后切哪个应用；
- **跨应用切换**：什么时候从 Excel 切到 Calendar/Email；
- **动作落地与验收**：即便完成了任务，结果可能体现在 `.xlsx`、`.ics`、`.eml`、`.pdf` 等不同格式里，不能只靠字符串精确匹配。

这也是“为什么现在做”的原因：LLM 已经展现出一定的规划能力，但是否能承担真实办公助手，还缺一个更贴近办公流程的评测基准。

### 输入/输出接口
- **输入**：自然语言办公任务 + Docker 内预置的文件系统、邮件、日历、文档等上下文。
- **输出**：代理逐步生成的 API/操作调用序列，以及最终造成的环境状态变化。
- **验收目标**：不是只看中间轨迹，而是看最终文件/日历/邮件状态是否满足任务要求。

### 边界条件
OfficeBench 不是开放世界桌面系统，而是一个**受控的办公模拟环境**：
- 9 个应用：System、Word、Excel、PDF、Calendar、Email、OCR、ChatGPT、Shell
- 共 23 个操作
- 300 个任务，分为：
  - Single App: 93
  - Two Apps: 95
  - Three Apps: 112
- 终止规则：
  - 正常 `submit_task`
  - 连续 5 次重复同一操作视为 stagnation
  - 最多 50 步，防止迭代溢出

这意味着它重点测的是**办公工作流能力**，不是开放桌面探索能力。

## Part II：方法与洞察

### How：它到底怎么测
OfficeBench 的设计可以拆成三层。

#### 1. 多应用环境
作者在 Docker 中预装办公相关应用，并用 Python 库把常见操作封装成可调用动作，例如：
- Excel: `read_excel_file`, `set_cell_content`
- Calendar: `create_event`, `delete_event`
- Email: `send_email`, `read_email`
- PDF: `read_pdf_file`, `convert_to_doc`

这使任务不再是抽象问答，而是要真正“做事”。

#### 2. 状态机化的工作流
作者把办公自动化形式化为一个 transition system：
- **状态**：当前选中的应用
- **动作**：该应用下的合法操作 + `switch_app` + `submit`
- **观察**：完整历史轨迹与每步操作返回结果

其中最关键的是 `System.switch_app`。它把“跨应用切换”变成一个显式决策，而不是让模型在一个巨大全动作表里瞎选。

#### 3. 按任务定制的评测器
论文的另一个关键点是：**不同办公任务不能用同一种打分方式**。

所以作者为每题配置三类评测：
- **Exact Matching**：适合结果唯一的任务
- **Fuzzy Matching**：适合允许多种合法表达的任务
- **Execution-based Evaluation**：适合存在多个正确解、需要程序判断语义正确性的任务

例如“给 Bob 明天 10:30-11:00 建会议”可以不用完全匹配事件全文，而只检查时间戳和关键字；“给 Alice 和 Bob 找共同空闲时间”则运行验证代码检查是否真的无冲突。

### 核心直觉
过去很多 agent 评测的问题，不是模型不会做，而是**评测把两类困难混在了一起**：
1. 在所有工具里找下一步；
2. 判断最终结果是否真的完成任务。

OfficeBench 的变化在于：

- **what changed**：从“全局统一动作空间 + 单一打分方式”，改成“当前应用约束的动作子空间 + 按任务定制的结果验证”。
- **which constraint changed**：
  - 动作空间从所有操作缩小到当前 app 的局部操作，减少动作 grounding 混乱；
  - 结果验证从死板的 exact match 扩展到 fuzzy / executable verification，减少对非唯一正确答案的误判。
- **what capability changed**：
  - Benchmark 能更清楚地区分代理到底败在**切换**、**规划**、还是**结果落地**；
  - 也让跨应用办公任务的评测更接近真实使用场景。

这就是它最有价值的因果点：**不是单纯把环境做复杂，而是把“多应用决策”和“结果验收”两个测量瓶颈显式化**。

### 为什么这个设计有效
一个直接证据来自消融实验：  
当模型必须通过 `switch_app` 管理当前应用时，性能高于“在 prompt 中直接列出全部操作”的做法，而且 token 更少。说明**显式应用切换 + 局部动作空间**确实减少了决策噪声，而不只是形式变化。

### 战略取舍

| 设计选择 | 解决的问题 | 获得的诊断能力 | 代价 |
|---|---|---|---|
| Docker + API 化办公应用 | 让办公任务可重复执行与复现 | 能稳定比较不同模型 | 不等同于真实 GUI/像素级桌面 |
| 当前应用限制动作空间 | 降低大动作空间下的 hallucination | 能单独观察 app switching 能力 | 仍比真实 OS 更受控 |
| exact/fuzzy/execution 混合评测 | 处理非唯一正确输出 | 更准确判断任务是否完成 | 需要为任务手写评测函数 |
| 合成数据构造文件系统 | 避免隐私问题，低成本扩展 | 易于批量生成办公场景 | 真实企业数据复杂性不足 |

## Part III：证据与局限

### So what：能力跃迁到底体现在哪
最重要的实验信号有三条。

#### 1. 当前最强模型离真实办公自动化还很远
在 300 个任务上：
- **GPT-4 Omni：47.00%**
- **人类：93.33%**

这说明今天的 LLM agent 已经“能做一些办公事”，但远未达到真实工作流所需可靠性。

#### 2. 真正拉开差距的是多应用协作
GPT-4 Omni 的表现从：
- Single App: **64.52%**
- Two Apps: **60.00%**
- Three Apps: **21.43%**

三应用场景的断崖式下降，直接支持作者的核心判断：**瓶颈不在单工具使用，而在跨应用组合规划**。

#### 3. 显式 app switching 不是装饰，而是有效机制
对 GPT-4 Omni / Llama 3 的消融表明：
- 使用 `switch_app` 比“列出所有操作”更高分
- 同时平均 token 更低

这说明 OfficeBench 不只是提出了一个任务集合，还提出了一个**更可诊断的评测交互协议**。

### 失败模式
论文给出的失败模式很具体，而且很有启发性：

1. **Stagnation at Redundant Operations**  
   模型读到有用信息后，仍反复执行同一操作，无法推进状态。

2. **Hallucinations in Action Prediction**  
   即使动作空间已被限制到当前应用，模型仍会生成不存在的操作名。

3. **Complex Planning across Applications**  
   模型缺少程序性常识，例如不知道“编辑 PDF”通常要先转 Word 再编辑再转回去。

这些失败并不只是“模型笨”，而是暴露出三个不同层面的短板：**循环控制、动作词表约束、跨工具程序知识**。

### 局限性
- **Fails when**: 任务需要 3 个及以上应用协调、隐含程序知识链条（如 PDF→Word→PDF）、或存在较多等价完成路径时，当前代理容易停滞、幻觉或走错流程。
- **Assumes**: 受控 Docker 环境、预定义 API、可文本化的操作反馈、为每题手写/配置评测函数，以及合成数据能近似真实办公内容；论文报告中的强基线还依赖闭源大模型 API。
- **Not designed for**: 开放桌面 GUI 感知、企业级 SaaS 权限系统、多人协作审批流、长期记忆与安全策略评测。

补充地说，这个 benchmark 的复现实验条件总体不错，但仍有几个现实依赖：
- 人类基线只由 2 名计算机专业研究生完成，样本较小；
- 数据由 ChatGPT 和随机程序合成，真实性有限；
- 环境是 API 化办公应用，而不是真实员工面对的复杂桌面软件栈。

### 可复用组件
这篇工作的可复用价值其实很高，主要体现在三点：
- **多应用状态机接口**：适合迁移到财务、客服、研究助理等 workflow benchmark。
- **任务定制评测器模板**：exact / fuzzy / execution 三类设计很通用。
- **办公环境合成管线**：文件、邮件、日历的合成与落盘方式可复用于其他 agent benchmark。

## Local PDF reference
![[paperPDFs/Agentic_Reasoning_Benchmarks_Applications_of_Agentic_Reasoning_Autonomous_Research_Agents/arXiv_2024/2024_OfficeBench_Benchmarking_Language_Agents_across_Multiple_Applications_for_Office_Automation.pdf]]