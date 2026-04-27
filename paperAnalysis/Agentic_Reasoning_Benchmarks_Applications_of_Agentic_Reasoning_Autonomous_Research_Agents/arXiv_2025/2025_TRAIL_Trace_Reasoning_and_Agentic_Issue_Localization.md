---
title: "TRAIL: Trace Reasoning and Agentic Issue Localization"
venue: arXiv
year: 2025
tags:
  - Survey_Benchmark
  - task/agent-trace-evaluation
  - task/issue-localization
  - error-taxonomy
  - open-telemetry
  - span-localization
  - dataset/TRAIL
  - dataset/GAIA
  - dataset/SWE-Bench-Lite
  - opensource/full
core_operator: 以 OpenTelemetry 结构化 agent 执行轨迹为输入，按“推理—规划—执行”细粒度错误分类做 span 级错误定位与诊断评测。
primary_logic: |
  评测真实 agent trace 调试能力 → 基于细粒度 taxonomy 对结构化执行轨迹做人工 span 级错误类别/位置/影响标注 → 用统一 judge 协议评测模型的分类、定位与整体打分能力 → 揭示长上下文 agent 调试的能力边界
claims:
  - "在 TRAIL 上，最佳模型 GEMINI-2.5-PRO 也仅达到 GAIA 18.3%、SWE-Bench 5.0% 的 joint accuracy，说明当前 LLM 对 agent trace 的类别+位置联合诊断能力仍然很弱 [evidence: comparison]"
  - "降低测试时推理强度会削弱诊断能力：o3 从 high 降到 low 时，Category F1 从 0.296 降到 0.264，Location Accuracy 从 0.535 降到 0.331 [evidence: ablation]"
  - "TRAIL 的输入越长模型越难诊断，输入长度与定位准确率呈负相关（Pearson r=-0.379），且多模型在长 trace 上直接触发上下文长度不足 [evidence: analysis]"
related_work_position:
  extends: "MAST (Cemri et al. 2025)"
  competes_with: "MAST (Cemri et al. 2025); ACPBench (Kokel et al. 2025)"
  complementary_to: "Prometheus 2 (Kim et al. 2024); CheckEval (Lee et al. 2025)"
evidence_strength: moderate
pdf_ref: paperPDFs/Agentic_Reasoning_Benchmarks_Applications_of_Agentic_Reasoning_Autonomous_Research_Agents/arXiv_2025/2025_TRAIL_Trace_Reasoning_and_Agentic_Issue_Localization.pdf
category: Survey_Benchmark
---

# TRAIL: Trace Reasoning and Agentic Issue Localization

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2505.08638), [Dataset](https://huggingface.co/datasets/PatronusAI/TRAIL)
> - **Summary**: 该文提出一个面向真实 agent 执行轨迹的细粒度错误 taxonomy 与人工标注 benchmark，把评测从“最终任务是否成功”推进到“错误发生在哪个 span、属于哪一类、影响多大”的可调试层面。
> - **Key Performance**: 最佳 GEMINI-2.5-PRO 仅有 GAIA 18.3% / SWE-Bench 5.0% 的 joint accuracy；输入长度与定位准确率负相关（Pearson r = -0.379）。

> [!info] **Agent Summary**
> - **task_path**: OpenTelemetry 结构化 agent trace -> 错误类别 + 错误 span 位置 + 轨迹级 rubric 分数
> - **bottleneck**: 现有 agent 评测大多停留在端到端成功率，无法在超长、工具交互密集、结构化的真实 trace 中做根因定位
> - **mechanism_delta**: 用跨推理/规划/执行的细粒度 taxonomy 与 span 级人工标注，替代只看最终结果的粗粒度 agent 评测
> - **evidence_signal**: 最强模型联合诊断仍很低，且多个模型在长 trace 上直接因上下文窗口不足而无法完整评测
> - **reusable_ops**: [error-taxonomy design, span-level trace annotation]
> - **failure_modes**: [context handling failures 几乎全体模型接近 0 F1, 超长 JSON trace 导致 CLE/性能显著下降]
> - **open_questions**: [如何扩展到多模态 agent trace, 如何覆盖高影响低频尾部错误而不过度依赖合成数据]

## Part I：问题与挑战

TRAIL要解决的，不是“agent 最后答对了吗”，而是更接近真实调试的问题：**给你一条很长、很乱、带工具调用和层级结构的执行轨迹，你能不能指出具体哪一步出了错、错在哪、为什么错**。

### 真正的瓶颈是什么
现有 agent benchmark 的常见问题是把复杂失败压成一个端到端结果：成功/失败、pass/fail、最终答案对不对。  
这对研究“agent 能不能完成任务”有用，但对“agent 为什么失败、该怎么修”帮助很小。

对真实 agent 系统来说，失败来源往往混在一起：
- 可能是 LLM 自己推理错了；
- 可能是工具输出被误解；
- 可能是 API 429/401/404 之类的执行错误；
- 也可能是多 agent 编排、上下文保持、子任务切换出了问题。

只看最终结果，**无法做 root-cause analysis**。这正是本文认为当前 agent 评测的核心缺口。

### 为什么现在必须做
因为 agent 已经从 toy setting 走向真实工作流：
- 软件工程：如 SWE-Bench 类 Github issue 修复；
- 开放域检索：如 GAIA；
- 多 agent / 长链路工作流越来越常见；
- 观测基础设施开始标准化到 OpenTelemetry / OpenInference。

这意味着“人工读完整 trace 做诊断”会迅速失去可扩展性。与此同时，长上下文 LLM 虽然看上去提供了自动化 judge 的可能，但这篇论文表明：**能读长文本，不等于能 debug 结构化 agent trace**。

### 输入/输出接口
**输入**：
- 来自 GAIA 与 SWE-Bench-Lite 的真实 agent 执行 trace；
- 格式为 OpenTelemetry/OpenInference 风格的结构化 JSON；
- 包含 span 层级、父子关系、LLM/tool 输入输出、token 统计等。

**输出**：
- span 级错误列表：`category / location(span_id) / evidence / description / impact`
- 轨迹级 rubric 分数：`reliability / security / instruction adherence / plan optimality`

### 边界条件
TRAIL有明确边界：
- 只覆盖 **text-only** traces；
- judge 主要依据 trace 内部证据，不要求去外部世界核验事实；
- 它评的是 **trace 调试与错误定位能力**，不是原始任务求解能力，也不是自动修复能力。

---

## Part II：方法与洞察

作者做的核心不是再训练一个更强的 agent，而是把“agent 调试”重新定义成一个可测、可定位、可分析的 benchmark 任务。

### 1. 细粒度错误 taxonomy
TRAIL把错误拆到三大层面：

1. **Reasoning Errors**
   - Language-only hallucination
   - Tool-related hallucination
   - Poor Information Retrieval
   - Tool Output Misinterpretation
   - Incorrect Problem Identification
   - Tool Selection Errors
   - Formatting Errors
   - Instruction Non-compliance

2. **System Execution Errors**
   - Tool Definition Issues
   - Environment Setup Errors
   - API errors：401/403/404/429/500
   - Resource Exhaustion
   - Timeout Issues

3. **Planning and Coordination Errors**
   - Context Handling Failures
   - Resource Abuse
   - Goal Deviation
   - Task Orchestration Errors

这个设计的关键点是：它不只看“推理错没错”，而是把**推理、规划、执行**统一放进一个诊断空间。  
这也是 TRAIL 相比 MAST、ACPBench 这类更偏 reasoning/planning 或 synthetic setting 的工作，最实用的扩展。

### 2. 数据构建：从真实 trace 出发，而不是从合成题出发
TRAIL的数据来自两个真实 benchmark 场景：

- **GAIA split**：开放域检索/搜索任务  
  使用层级式 multi-agent 编排（manager + search agent），backbone 为 o3-mini。
- **SWE-Bench split**：软件 issue 定位与修复场景  
  使用 CodeAct agent + sandbox + Python interpreter + gitingest，backbone 为 Claude 3.7 Sonnet。

同时，作者故意保留了真实系统中的“脏信息”：
- 原始结构化 trace；
- 长上下文；
- 工具调用细节；
- 约束提示导致的格式/上下文/编排类错误。

所有 traces 通过 **OpenTelemetry/OpenInference** 采集，以保证和真实 observability 体系兼容。

### 3. 标注协议：从“这条 trace 好不好”到“具体哪一 span 错了”
TRAIL不是只给一条总体分数，而是由专家做 **turn/span-level annotation**：
- 标错误类别；
- 标错误位置；
- 给出证据摘录；
- 写错误描述；
- 标影响级别（LOW/MEDIUM/HIGH）。

此外还给出轨迹级 rubric 分数，用于衡量模型作为 judge 的整体相关性。

论文主文报告：
- 约 **148** 条人工标注 traces；
- **1987** 个 spans；
- **575** 个带错 spans；
- **841** 个标注错误。

这说明 benchmark 的目标不是“找一个唯一 bug”，而是处理真实 trace 中**多错误共存**的情况。

### 4. 评测协议
作者用统一 prompt 要求 LLM：
- 穷尽式找出 trace 中所有错误；
- 给出 JSON 格式结果；
- 同时预测轨迹级 rubric 分数。

主要指标包括：
- **Category F1**
- **Location Accuracy**
- **Joint Accuracy**（类别和位置同时正确）
- 与人工 rubric 的 **Pearson correlation**

被测模型覆盖 GPT-4.1、o1、o3、Claude-3.7-Sonnet、Gemini-2.5 Pro/Flash、Llama-4 等。

### 核心直觉
这篇论文真正拧动的“关键旋钮”不是模型结构，而是**评测观测粒度**。

从因果上看，变化链条是：

**只看最终成败**  
→ 失败原因被压缩成单个 outcome，根因信息丢失  
→ 模型即使会“猜结果”，也不一定会“做调试”

变成

**看原始结构化 trace 的 span 级错误类型 + 位置**  
→ 错误发生时刻、局部证据、错误来源被显式保留  
→ benchmark 才能测到真正的调试能力，而不是表面相关性

为什么这有效：
1. **位置约束** 把任务从模糊评分变成可核查的局部归因；
2. **类型约束** 把错误分流到工程上可操作的修复面；
3. **真实 trace** 防止模型仅靠合成模式或表层语言线索取巧；
4. **结构化输入** 逼近真实 observability 场景，而不是对话式简化文本。

换句话说，TRAIL测的不是“模型会不会评价答案”，而是“模型会不会看日志、会不会定位错误、会不会解释 agent 为什么失控”。

### 战略取舍

| 设计选择 | 带来的收益 | 代价/风险 |
| --- | --- | --- |
| 原始 OpenTelemetry 结构化 trace，而非摘要文本 | 生态有效性高，保留 span 层级与工具细节 | 上下文极长，很多模型直接读不完 |
| 细粒度 taxonomy 覆盖推理/规划/执行 | 能做 root-cause 级诊断 | 长尾类别样本少，类不平衡明显 |
| 真实 GAIA/SWE 轨迹，而非纯合成数据 | 更接近真实 agent 失败分布 | 噪声更大，结果受选定 agent/backbone 影响 |
| 要求 span 级位置 + 证据 | 更可解释，也更可用于工程调试 | 标注成本高，模型 joint accuracy 更难提升 |

---

## Part III：证据与局限

### 关键实验信号
**1. 比较信号：现有模型还远远不会“debug agent trace”**  
最佳模型 GEMINI-2.5-PRO 在：
- GAIA 上 joint accuracy 仅 **18.3%**
- SWE-Bench 上 joint accuracy 仅 **5.0%**

这不是“离 SOTA 还差一点”，而是说明这个任务目前远未饱和。  
尤其 joint metric 很低，意味着模型经常出现“知道大概哪类错了，但找不到具体 span”，或“指到了 span，但类别判错”。

**2. 分析信号：长上下文不是背景条件，而是主瓶颈**  
论文显示输入长度与性能显著负相关：
- Location Accuracy 的 Pearson 相关为 **-0.379**
- Category F1 和 Joint Accuracy 也都随长度下降

更重要的是，不少 trace 平均长度已经逼近甚至超过一些模型的 context window；部分模型在某些 split 上直接出现 **CLE（context length exceeded）**。  
这说明 TRAIL 的难点不是普通文本推理，而是**超长结构化 trace 的可读性与可检索性**。

**3. 消融信号：test-time reasoning 确实有帮助**  
以 o3 为例，reasoning effort 从 high 降到 low：
- Category F1：**0.296 → 0.264**
- Location Accuracy：**0.535 → 0.331**

说明这类任务不是简单 pattern matching，确实需要更强的逐步探索与证据整合。

**4. 类别分析：真正难的是状态追踪，不是表层文案错误**  
最难的类别包括：
- Context Handling Failures
- Tool Selection Errors
- Task Orchestration

尤其 Context Handling Failures 基本接近全体模型的薄弱项。  
相对而言，Language-only 这类更“表面文本化”的错误更容易被检测到。  
这说明 agent judge 的核心短板在于：**跨步骤状态追踪、工具调用意图理解、以及计划执行链的一致性判断**。

### 1-2 个最值得记住的数字
- **18.3% / 5.0%**：最佳模型在 GAIA/SWE 上的 joint accuracy
- **r = -0.379**：输入越长，定位越差

### 局限性
- **Fails when**: 遇到多模态 agent trace、需要图像/音频工具证据、或需要跳出 trace 去外部核验事实时，TRAIL 当前覆盖不足；对高影响但极低频的尾部错误，统计稳定性也较弱。
- **Assumes**: 有标准化 OpenTelemetry/OpenInference 日志；有可承受长上下文推理的 judge 模型；有高成本专家标注流程（文中估计单条 trace 连同复核约 110–120 分钟）；部分轨迹生成依赖闭源 backbone（如 o3-mini、Claude 3.7）。
- **Not designed for**: 评估 agent 最终任务完成率、训练自动修复策略、或覆盖多模态 agent 的全栈安全与可靠性审计。

另外，一个实际阅读层面的限制是：**文中主文与附录在数据计数上有轻微不一致**，例如 traces 和 errors 的总数出现 148/149、841/835 等口径差异。  
这不改变论文主结论，但对严格复现和二次引用来说，应以官方发布数据版本为准。

### 可复用组件
这篇论文最值得复用的，不只是数据本身，还包括：
- 一套面向 agent trace 的 **错误 taxonomy**
- 一套 **span 级标注 schema**（location / evidence / impact）
- 基于 OpenTelemetry/OpenInference 的 **结构化 trace 表示**
- 一套可直接拿来评 judge 的 **JSON 输出协议与 rubric 设计**

如果你要做 agent observability、trace debugging、LLM-as-a-judge for agents，这些组件都可以直接拿来当起点。

![[paperPDFs/Agentic_Reasoning_Benchmarks_Applications_of_Agentic_Reasoning_Autonomous_Research_Agents/arXiv_2025/2025_TRAIL_Trace_Reasoning_and_Agentic_Issue_Localization.pdf]]