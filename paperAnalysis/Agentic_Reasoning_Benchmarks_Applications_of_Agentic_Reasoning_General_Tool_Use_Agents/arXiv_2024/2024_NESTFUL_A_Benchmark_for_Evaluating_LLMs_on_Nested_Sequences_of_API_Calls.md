---
title: "NESTFUL: A Benchmark for Evaluating LLMs on Nested Sequences of API Calls"
venue: arXiv
year: 2024
tags:
  - Survey_Benchmark
  - task/function-calling
  - task/tool-use-evaluation
  - executable-evaluation
  - nested-tool-calling
  - dag-analysis
  - dataset/NESTFUL
  - dataset/MathQA
  - dataset/StarCoder2-Instruct
  - opensource/full
core_operator: 用带变量引用的可执行嵌套API序列与执行式评分，专门测量LLM在中间结果传递和多步依赖维护上的工具调用能力边界
primary_logic: |
  评测嵌套工具调用能力 → 从 MathQA 与 StarCoder2-Instruct 构造带变量标签的可执行API序列和工具库 → 以序列匹配、执行胜率与DAG结构分析进行评分 → 揭示模型在深层依赖、变量绑定和类型遵循上的失败模式
claims:
  - "NESTFUL包含1800+条可执行的嵌套API调用序列，覆盖数学推理与通用Python工具两类场景，并为每个工具提供可执行实现 [evidence: analysis]"
  - "在19个评测模型中，最强模型的Full Sequence Accuracy仅约28-29%，Win Rate仅60%，说明当前LLM距离可靠的嵌套工具调用仍有明显差距 [evidence: comparison]"
  - "模型性能会随最大嵌套深度、总数据依赖数增加而显著下降，尤其在多父节点模式{A,B}→C上退化明显 [evidence: analysis]"
related_work_position:
  extends: "API-BLEND (Basu et al. 2024)"
  competes_with: "NesTools (Han et al. 2024); BFCL-v3 (Yan et al. 2024)"
  complementary_to: "ReAct (Yao et al. 2023); Reverse Chain (Zhang et al. 2024c)"
evidence_strength: moderate
pdf_ref: paperPDFs/Agentic_Reasoning_Benchmarks_Applications_of_Agentic_Reasoning_General_Tool_Use_Agents/arXiv_2024/2024_NESTFUL_A_Benchmark_for_Evaluating_LLMs_on_Nested_Sequences_of_API_Calls.pdf
category: Survey_Benchmark
---

# NESTFUL: A Benchmark for Evaluating LLMs on Nested Sequences of API Calls

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2409.03797), [GitHub](https://github.com/IBM/NESTFUL)
> - **Summary**: NESTFUL把LLM函数调用评测从“会不会选API”升级为“能否在可执行链路里正确维护变量、类型和依赖”的嵌套工具调用评测。
> - **Key Performance**: 三-shot下 GPT-4o 的 Full Match 为 28%、Win Rate 为 60%；DeepSeek-V3 的 Full Match 达 29%，但 Win Rate 同样只有 60%。

> [!info] **Agent Summary**
> - **task_path**: 自然语言查询 + 候选工具库 -> 嵌套API调用序列 -> 可执行最终答案
> - **bottleneck**: 现有tool-calling benchmark难以衡量中间结果传递、变量绑定和多步数据依赖维护
> - **mechanism_delta**: 用带 `label`/变量引用的可执行API序列和双层指标，把“API选择”评测改成“DAG依赖维护”评测
> - **evidence_signal**: 19个模型上最优Win Rate仅60%，且深度>1或依赖>1时性能明显崩塌
> - **reusable_ops**: [显式变量标签与输出字段引用, 可执行工具库与execution-based filtering]
> - **failure_modes**: [变量赋值/引用错误, 输出字段或数据类型与下游API不匹配]
> - **open_questions**: [如何做schema-agnostic的嵌套工具调用训练, 如何在大工具库检索下稳定维护中间状态]

## Part I：问题与挑战
NESTFUL要解决的不是“LLM会不会调一个API”，而是更真实的agent瓶颈：**能否在多步工具链里持续、正确地传递中间结果**。

现实中的工具使用常常不是独立调用，而是：
1. 先选对函数；
2. 再抽取对参数；
3. 给每一步分配变量；
4. 读取前一步的正确输出字段；
5. 把它以正确类型传给后续API；
6. 在并行分支和汇合节点上保持一致。

现有很多函数调用benchmark更偏向单步调用、独立多API调用，或只检查局部API名/参数是否匹配。它们对**nested sequencing**覆盖不足，因此很难诊断以下真实失败来源：
- API名选对了，但变量引用错了；
- 变量引用对了，但输出字段错了；
- 字段对了，但类型不匹配；
- 单链路能做，对多分支汇合 `{A, B} -> C` 就崩。

### 输入 / 输出接口
- **输入**：自然语言查询 + 候选工具库
- **输出**：带 `label` 的JSON API序列，后续参数可以引用如 `$var_1.result$`
- **评测目标**：既看生成的工具链是否忠实于金标，也看执行后是否得到正确答案

### 任务边界
- 数据来自两个域：
  - **MathQA**：数学推理，链条更长；
  - **StarCoder2-Instruct**：Python工具，函数种类更丰富。
- 这是一个**离线、可执行**benchmark，不是在线真实API环境。
- 由于上下文长度限制，作者为每个样本构造了**裁剪后的工具子集**，而不是完整工具库检索场景。
- 评测包含 direct prompting 的 one-shot / three-shot ICL，以及 zero-shot ReAct。

**真正的瓶颈**因此很明确：当前LLM不是单纯“不会调工具”，而是**不会稳定维护跨步骤的数据依赖图**。  
这也是为什么现在需要这类benchmark：agent系统已经高度依赖工具调用，但现有评测还没把最核心的链式依赖难点测准。

## Part II：方法与洞察
NESTFUL的核心不是提出新agent算法，而是把“嵌套工具调用”变成一个**可执行、可验证、可分层诊断**的问题。

### 数据与评测设计
1. **统一的嵌套调用schema**  
   每个函数调用包含 `name / arguments / label`。  
   后续API可通过变量引用读取前一步输出，例如 `$var_1.result$`。  
   这使得重复函数、并行调用、汇合依赖都能被显式表达。

2. **每个工具都可执行**  
   benchmark不仅提供工具描述，还提供Python实现。  
   因此模型输出不是“字符串看起来像对”，而是真能执行验证。

3. **两类数据源，兼顾深度与广度**
   - **MathQA分支**：1415条样本，40个工具，平均 5.1 次调用/样本。
   - **Coding分支**：446条样本，881+工具，平均 2.1 次调用/样本。  
   其中 coding 部分通过 Mixtral-8x22B 推断类型、DiGiT 合成嵌套样本，再用 validator 与执行过滤保证质量。

4. **多粒度指标**
   - **F1 Function / F1 Param**：局部函数名与参数名是否正确
   - **Partial Sequence Accuracy**：部分工具链是否对
   - **Full Sequence Accuracy**：整条工具链是否完全一致
   - **Win Rate**：执行后最终答案是否等于金标

### 核心直觉
NESTFUL真正改变的是：**把中间状态从隐式文本推理，变成显式、可执行、可引用的变量节点**。

这带来三层能力变化：

- **what changed**：引入 `label -> output field -> downstream argument` 的显式数据流表示，并要求工具可执行。
- **which bottleneck changed**：评测从“局部API分类/槽位抽取”变成“整个DAG里的依赖维护与状态传递”。
- **what capability changed**：现在可以区分  
  1) 模型是否忠实生成了正确工具链；  
  2) 模型是否虽然没走金标链路但仍得到正确答案；  
  3) 错误究竟来自工具选择、变量绑定、字段读取还是深层依赖。

这套设计有效的原因是：一旦中间变量和执行路径都被显式化，**嵌套推理失败就不再是模糊的“答错了”**，而能被定位为某种具体结构错误。

### 战略权衡
| 设计选择 | 带来的能力 | 代价 |
|---|---|---|
| 显式变量标签与 `$var_i.field$` 引用 | 精确表达嵌套依赖、并行分支与结果汇合 | 对未见过该schema训练的模型不够友好 |
| 每个工具都提供可执行实现 | 评测可验证，减少假阳性 | 只覆盖能稳定实现和执行的工具 |
| 同时报告 Full Acc 与 Win Rate | 区分“链路忠实度”和“最终任务完成度” | Win Rate 可能高估靠参数知识直接算出答案的模型 |
| 每样本裁剪工具库 | 能在有限上下文内稳定运行评测 | 不等价于真实大规模工具检索 |

## Part III：证据与局限
### 关键证据信号
1. **横向比较信号：当前模型离可靠嵌套工具调用还很远**  
   作者在19个模型上评测。最强结果依然很低：  
   - 摘要强调 GPT-4o 约 **28% Full Match / 60% Win Rate**
   - 表1中三-shot下 **DeepSeek-V3 的 Full Match 为 29%**，但 Win Rate 也只有 **60%**  
   这说明即使是顶级模型，也很难稳定生成完整正确的嵌套API序列。

2. **结构复杂度信号：深度和依赖数一上升，性能明显下滑**  
   作者把样本表示成DAG，按**最大嵌套深度**和**总数据依赖数**分析。  
   结论很清晰：深度为1时模型还能做，一旦深度≥2或依赖数增加，Win Rate快速下降。  
   这直接支持了论文主张：真正难点是**跨步依赖维护**，不是单次API选择。

3. **拓扑模式信号：多父节点汇合最难**  
   简单模式 `A -> B` 表现相对稳定；  
   更复杂的 `{A, B} -> C` 明显更难。  
   这说明模型尤其不擅长同时跟踪多个中间变量，并在下游正确对齐。

4. **agent形式信号：ReAct有帮助，但不是银弹**  
   ReAct对部分大模型有效，例如 Mixtral-8x22B 的 Win Rate 大约从 **7% 提升到 30%**，DeepSeek-V3 也有小幅提升。  
   但它并不稳定提升所有模型，Hammer2.0-7B 反而 direct prompting 更好，AgentLM-13B 也没有表现出明显优势。  
   所以：**“更agentic”不自动等于“更会处理嵌套依赖”**。

### 1-2 个最值得记住的数字
- **最好 Win Rate 只有 60%**：距离可部署工具代理还有明显差距。
- **最好 Full Sequence Accuracy 只有 28-29%**：真正按正确工具链完成任务的能力更弱。

### 局限性
- Fails when: 任务需要真实在线API副作用、长时多轮交互、跨回合记忆，或面对完整超大工具库检索时，NESTFUL的离线可执行设定不能完整覆盖这些难点。
- Assumes: 工具有明确JSON规格和可执行Python实现；每个样本只给裁剪后的工具子集；coding部分依赖Mixtral-8x22B、DiGiT与自动validator生成并过滤数据；不少被测模型也未见过该变量标注schema。
- Not designed for: 安全性/权限控制评测、真实服务时延与成本评测、开放世界工具发现、需要GUI/网页环境交互的在线agent任务。

### 可复用组件
- **显式变量引用schema**：适合任何需要评测“中间结果是否被正确消费”的工具调用任务。
- **execution-based filtering**：保证金标序列真的可运行、可复现。
- **DAG难度分层分析**：可按深度、依赖数、拓扑模式拆解失败来源。
- **Faithfulness / Outcome 双指标**：`Full/Partial Accuracy + Win Rate` 很适合区分“是否忠实调用工具”和“是否最终做对任务”。

![[paperPDFs/Agentic_Reasoning_Benchmarks_Applications_of_Agentic_Reasoning_General_Tool_Use_Agents/arXiv_2024/2024_NESTFUL_A_Benchmark_for_Evaluating_LLMs_on_Nested_Sequences_of_API_Calls.pdf]]