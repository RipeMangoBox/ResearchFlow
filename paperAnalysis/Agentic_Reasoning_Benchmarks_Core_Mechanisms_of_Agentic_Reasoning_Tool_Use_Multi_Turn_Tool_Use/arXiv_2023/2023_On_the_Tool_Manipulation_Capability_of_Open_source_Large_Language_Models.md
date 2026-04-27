---
title: "On the Tool Manipulation Capability of Open-source Large Language Models"
venue: arXiv
year: 2023
tags:
  - Others
  - task/tool-manipulation
  - instruction-tuning
  - retrieval-augmented-generation
  - system-prompt
  - dataset/ToolBench
  - opensource/partial
core_operator: 用程序化模板合成 API 使用样本做多工具对齐，再在推理时检索相似演示并用 system prompt 约束只输出可执行代码。
primary_logic: |
  自然语言目标 + API 文档/环境观察 → 以少量人工模板程序化生成对齐数据做多工具微调，并在推理时检索语义相近演示、用 system prompt 收缩输出格式 → 更准确且可执行的 API 调用序列
claims:
  - "在 ToolBench 上，零样本开源 LLM 与 GPT-4 存在显著工具操作差距，部分任务成功率最高可低 78 个百分点 [evidence: comparison]"
  - "程序化对齐 + 检索式 in-context demonstration + system prompt 可显著提升开源模型，并使其在 ToolBench 的 8 个任务中有 4 个达到或超过同样加入检索与 prompt 的 GPT-4 基线 [evidence: comparison]"
  - "消融显示模型对齐是主要增益来源：从完整系统中移除 alignment 会在最多 7 个任务上造成退化，影响大于移除 3-shot 检索或 system prompt [evidence: ablation]"
related_work_position:
  extends: "InstructGPT (Ouyang et al. 2022)"
  competes_with: "Toolformer (Schick et al. 2023); HuggingGPT (Shen et al. 2023)"
  complementary_to: "ReAct (Yao et al. 2022); Code as Policies (Liang et al. 2022)"
evidence_strength: strong
pdf_ref: paperPDFs/Agentic_Reasoning_Benchmarks_Core_Mechanisms_of_Agentic_Reasoning_Tool_Use_Multi_Turn_Tool_Use/arXiv_2023/2023_On_the_Tool_Manipulation_Capability_of_Open_source_Large_Language_Models.pdf
category: Others
---

# On the Tool Manipulation Capability of Open-source Large Language Models

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2305.16504), [Code/ToolBench](https://github.com/sambanova/toolbench)
> - **Summary**: 论文把开源 LLM 的工具调用失败拆成“API 选错、参数填错、输出不可执行”三类，再用程序化合成数据对齐、检索式示例和 system prompt 做低成本修补，使开源模型在多项工具任务上逼近 GPT-4。
> - **Key Performance**: LLaMA-30B 在 Home Search 从 0% 提升到 87.0%；CodeGen-16B-mono 在 OpenWeather 从 7.0% 提升到 97.7%。

> [!info] **Agent Summary**
> - **task_path**: 自然语言目标 / API 文档 / 可选环境观察 -> 可执行 API 调用序列
> - **bottleneck**: 开源 LLM 缺少 API 使用先验与参数绑定锚点，且输出分布未被约束到可执行代码
> - **mechanism_delta**: 用少量人工模板把 API 用法注入模型，再用语义相似演示和 code-only prompt 把推理空间收缩到“像样例、可执行”的区域
> - **evidence_signal**: 8 个真实工具任务上的对比实验 + 逐项移除模块的消融
> - **reusable_ops**: [programmatic-template-alignment, semantic-demo-retrieval]
> - **failure_modes**: [advanced-reasoning-tasks-remain-hard, per-tool-curation-still-needed]
> - **open_questions**: [how-to-handle-reasoning-heavy-tools, how-to-generalize-to-new-tools-with-less-curation]

## Part I：问题与挑战

这篇论文的真正问题不是“开源模型比 GPT-4 稍弱”，而是：**一旦任务变成真实工具/API 操作，开源 LLM 的能力会出现断崖式下滑**。这对工业落地很关键，因为很多企业不愿把内部流程和数据暴露给闭源 API。

**输入/输出接口**很清楚：
- 输入：自然语言目标，外加 API 文档；多步任务里还会有环境返回的 observation。
- 输出：一段可执行的 API 调用序列，单步或多步执行后完成目标。

作者发现，瓶颈主要不在“不会写代码”，而在工具接口层的三个连续错误：
1. **API selection**：不知道该调用哪个 API，甚至会幻觉出不存在的 API 名。
2. **Argument populating**：API 选对了，但参数槽位和值填错。
3. **Non-executable generation**：输出混入自然语言解释、格式不合法，导致代码根本无法执行。

论文里一个很强的动机信号是：在 Home Search 这类任务上，GPT-4 零样本能到约 76.6%～77%，而若干开源模型接近 0%。这说明这里的差距不是普通 NLP 任务上的“小幅退化”，而是**工具使用先验根本没学到**。

边界条件上，这篇工作主要讨论的是：
- 已知工具/API 文档；
- 以 API 调用为统一动作接口；
- 重点解决“把自然语言目标翻译成正确工具调用”，而不是从零发明复杂 planner。

## Part II：方法与洞察

作者的方法很朴素，但因果链很明确：**把三类失败分别对应到三种老方法的重新适配**，而不是另起一个复杂 agent 框架。

### 方法主线

**1. 模型对齐：用程序化模板合成工具调用训练数据**
- 人工为每个工具写少量 goal/API-call 模板与参数值池。
- 用随机实例化扩充成训练样本。
- 原则上只需做到每个 API 至少在一个模板里出现，人工量接近 `O(n)`。
- 再把多个工具混在一起联合微调成一个模型。

这一步本质上是在补“API 使用知识的内化”。

**2. 演示检索：推理时检索语义最相近的示例**
- 维护一个小型示例库，保证每个 API 至少在一个示例里出现。
- 根据目标语义检索相似示例，作为 in-context demonstration。
- 目标不是记住所有 API 组合，而是让模型借助局部相似样例去泛化到未见组合。

这一步主要缓解“参数怎么填”的问题。

**3. System prompt：强约束只输出代码**
- 用统一 prompt 模板告诉模型：只生成 API 调用，不要解释。
- 同时把 API 列表、检索到的示例、目标任务拼进上下文。
- 成本最低，但能显著减少不可执行输出。

### 核心直觉

开源 LLM 在工具操作上失败，核心不是单点能力差，而是**同时缺三种约束**：
- 缺少 **API 先验**：不知道哪些函数真的存在、怎么组合；
- 缺少 **局部类比**：自然语言里的槽位难绑定到参数；
- 缺少 **输出边界**：生成分布偏向“解释型文本”而不是“可执行代码”。

作者做的关键改变可以概括为：

**零样本自由生成**  
→ **先用 alignment 把 API 用法压进模型参数里**  
→ **再用检索示例把当前任务拉到一个局部可模仿区域**  
→ **最后用 system prompt 把输出空间收窄到代码格式**  
→ **得到更稳定的工具调用能力**

也因此，这套方法最擅长提升的是**接口对齐型能力**，而不是深层规划/推理能力。

### 策略取舍

| 组件 | 解决的核心错误 | 改变了什么瓶颈 | 优势 | 代价 / 风险 |
|---|---|---|---|---|
| 程序化对齐 | API 选错、幻觉 API | 把“外部文档查找”变成“模型内部 API 先验” | 增益最大，可多工具联合训练 | 每个工具仍需模板和值池，需微调算力 |
| 演示检索 | 参数填充错误、未见组合 | 给当前目标一个局部类比锚点 | 示例量只需 O(n)，对组合泛化有效 | 检索错例会误导模型，受上下文长度限制 |
| System prompt | 输出不可执行 | 收缩生成分布到 code-only 格式 | 实现便宜、部署简单 | 只能约束格式，不能补足 reasoning |

## Part III：证据与局限

**最强证据不是单个分数，而是“诊断—修补—再验证”的闭环。**

- **对比信号**：零样本下，开源模型在 ToolBench 上与 GPT-4 有巨大差距。像 Home Search 这类任务，LLaMA/CodeGen 接近 0，而 GPT-4 在 76% 以上，说明问题确实集中在工具调用而非一般文本生成。
- **能力跃迁信号**：加入三项技术后，开源模型在多个任务上出现明显跨档提升。最直观的例子是 OpenWeather、Cat API、Home Search、Trip Booking 这类“工具接口是主要难点”的任务。
- **因果信号**：消融里，移除 alignment 的伤害最大；移除 3-shot 检索次之；system prompt 更多是在“兜底可执行性”。这支持论文的主张：**真正的主旋钮是把 API 用法对齐进模型**。
- **边界信号**：Google Sheets、WebShop、Tabletop 这类需要额外推理、表格定位、环境决策或空间规划的任务，即使增强后仍然偏弱。说明这套 recipe 主要修复的是“工具接口层”，不是“高级 reasoning 层”。

一个很重要的现实结论是：这套方法并不要求巨量人工标注。作者报告平均**每个工具约 1 个开发者日**即可完成模板和演示整理。但它也不是“纯 prompt engineering”——因为最关键的提升来自一次监督微调。文中训练配置约为 **4×A100 80GB，8 epochs**；同时评测依赖真实 API 或模拟环境，外部服务与环境随机性会影响完全复现。

### 局限性

- **Fails when**: 任务主要难点不是 API 选择，而是高级推理/规划/环境理解时，例如 Google Sheets 的单元格定位、WebShop 的页面决策、Tabletop 的空间操作。
- **Assumes**: 每个工具都有较完整 API 文档，并且能提供少量人工模板与演示；还假设可以进行一次多工具监督微调，并拥有可执行评测环境。
- **Not designed for**: 无文档冷启动的新工具发现、长时程自主探索式 agent、多轮反思纠错、完全零人工适配的新工具生态。

### 可复用组件

- **程序化模板造数**：适合任何“自然语言 → 结构化调用”的接口任务。
- **语义示例检索器**：可直接迁移到 SQL、workflow、browser action 等场景。
- **Code-only system prompt**：低成本降低不可执行输出。
- **执行式评测框架**：比字符串精确匹配更接近真实工具使用。

![[paperPDFs/Agentic_Reasoning_Benchmarks_Core_Mechanisms_of_Agentic_Reasoning_Tool_Use_Multi_Turn_Tool_Use/arXiv_2023/2023_On_the_Tool_Manipulation_Capability_of_Open_source_Large_Language_Models.pdf]]