---
title: "MetaTool Benchmark for Large Language Models: Deciding Whether to Use Tools and Which to Use"
venue: ICLR
year: 2024
tags:
  - Survey_Benchmark
  - task/llm-evaluation
  - task/tool-selection
  - group-based-evaluation
  - similarity-based-ranking
  - reliability-testing
  - dataset/TOOLE
  - opensource/no
core_operator: 通过多样化工具触发查询、工具重叠消解和四类候选集构造，把 LLM 在“是否用工具/该选哪个工具”上的前置控制能力拆成可量化评测。
primary_logic: |
  评测目标（工具使用意识与工具选择） → 生成并人工校验 TOOLE 多样化查询、通过工具合并/分解消除标签重叠 → 构造相似工具、场景化、可靠性与多工具四类评测任务 → 输出 LLM 在边界自知、可靠 abstain 和组合调用上的能力边界
claims:
  - "METATOOL 将评测范围扩展到工具调用前的两个阶段——是否需要工具与该选哪个工具——并提供 4 个工具选择子任务，覆盖了以往基准较少涉及的前置决策过程 [evidence: analysis]"
  - "在工具使用意识测试中，zero-shot 下只有 ChatGPT 的 Accuracy 与 F1 同时超过 70%，多数开源模型仍难以稳定判断何时应该调用外部工具 [evidence: analysis]"
  - "在移除正确工具的可靠性子任务中，除 ChatGPT 外多数模型 CSR 低于 20%，而人类 CSR 达到 96%，说明当前 LLM 的核心短板是可靠地放弃错误选择而非单纯语义匹配 [evidence: analysis]"
related_work_position:
  extends: "APIBank (Li et al. 2023d)"
  competes_with: "ToolBench (Xu et al. 2023); ToolQA (Zhuang et al. 2023)"
  complementary_to: "ReAct (Yao et al. 2022); Toolformer (Schick et al. 2023)"
evidence_strength: moderate
pdf_ref: paperPDFs/Agentic_Reasoning_Benchmarks_Core_Mechanisms_of_Agentic_Reasoning_Tool_Use_Single_Turn_Tool_Use/ICLR_2024/2024_MetaTool_A_Benchmark_for_Controlling_Special_purpose_Large_Language_Models.pdf
category: Survey_Benchmark
---

# MetaTool Benchmark for Large Language Models: Deciding Whether to Use Tools and Which to Use

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2310.03128)
> - **Summary**: 这篇工作把工具调用前最关键的控制层拆成“要不要用工具”和“该选哪个工具”，并用 TOOLE + 四类子任务系统性暴露出当前 LLM 在边界自知、可靠拒答和多工具组合上的明显短板。
> - **Key Performance**: ChatGPT 在工具使用意识上 5-shot F1=81.87；可靠性子任务中人类 CSR=96%，而 8 个模型平均仅 8.59%

> [!info] **Agent Summary**
> - **task_path**: 用户查询 / 候选工具列表 -> 是否需要工具 / 选择 1 个工具、None 或 2 个工具
> - **bottleneck**: 现有工具基准主要测调用后的参数填充与结果处理，没把“调用前决策”单独测出来；同时工具功能重叠会让单标签评测失真
> - **mechanism_delta**: 用多样化查询生成 + 工具合并/分解消重叠 + 相似工具/场景/缺失真值/多工具四类候选集，把不同类型的控制错误分开测
> - **evidence_signal**: 移除正确工具后，大多数模型仍会乱选或幻觉不存在的工具，可靠性任务与人类差距最大
> - **reusable_ops**: [工具合并-分解消重叠, 基于相似度的困难候选集构造]
> - **failure_modes**: [候选列表变长时 CSR 下滑, 无真值工具时不愿输出 None]
> - **open_questions**: [动态工具库下评测是否稳定, 如何联合评测选工具与参数填充/执行]

## Part I：问题与挑战

这篇工作针对的真实问题，不是“LLM 能不能调用 API”，而是 **LLM 能不能先做对调用前的元决策**：  
1. 我自己能不能完成这个请求？  
2. 如果不能，应该选哪个工具，还是应该明确说没有合适工具？

作者指出，已有工具基准大多覆盖 ReAct 流程的后半段：**参数配置（③）和结果处理（④）**。但在 AutoGPT、MetaGPT 这类 agent 场景里，真正先出错的往往是前两步：**工具使用意识（①）** 和 **工具选择（②）**。如果这里错了，后续执行再强也没用。

这个问题难在三点：

- **数据难**：真实用户提问方式很多样，不能只靠单一 prompt 生成“教科书式” query。
- **标签难**：工具功能常有重叠，一个 query 可能对应多个可行工具，单标签准确率会失真。
- **诊断难**：模型失败可能来自语义混淆、场景偏置、幻觉、不会拒答、不会组合多工具，必须拆开评。

输入/输出边界也很清楚：

- **意识任务**：输入 query，输出 `yes/no`
- **选择任务**：输入 query + 候选工具列表，输出 `tool / None / 两个工具`
- **边界条件**：主要基于文本化工具描述和静态候选集，不涉及真实 API 执行链路

## Part II：方法与洞察

### 评测设计

作者提出的是一个 benchmark，而不是新模型。核心载体是 **TOOLE** 数据集与围绕它构建的评测协议。

- **TOOLE 数据集**：
  - 来自 OpenAI plugin 列表中的 390 个工具描述
  - 用 4 种方式生成 query：direct diverse、emotion、keyword、details
  - 再做人工清洗
  - 关键一步是 **工具合并/分解**：把功能高度重叠的工具合并，把“一工具多功能”的工具拆开，尽量让每个 query 只对应一个真值标签
  - 最终得到 **21,127** 条 query，其中 **20,630** 条单工具，**497** 条多工具

- **工具使用意识评测**：
  - 正样本：从 TOOLE 中挑选当前 LLM 本体难以完成、确实该借助工具的 query
  - 负样本：从指令数据/常识问答中挑选 LLM 可直接回答的问题
  - 输出只允许 `yes/no`，测模型是否知道自己的能力边界

- **四类工具选择子任务**：
  1. **相似工具选择**：把真值工具与 embedding 最相近的工具放在一起，测细粒度语义区分
  2. **特定场景选择**：按职业/身份或热门工具集合构造列表，测场景偏置与适应性
  3. **可靠性选择**：故意把正确工具移除，理想输出应是 `None`，测 hallucination / sycophancy / abstain 能力
  4. **多工具选择**：给出需要两个工具的 query，测组合调用和基本推理能力

### 核心直觉

作者改变的不是模型，而是 **测量瓶颈**。

过去的工具 benchmark 容易把多个问题混在一起：  
“是不是理解了用户意图？”、“是不是识别了正确工具？”、“是不是会填参数？”、“是不是会整合结果？”  
METATOOL 的做法是先把工具空间尽量清洗干净，再通过不同候选集设计，把 controller 层的不同错误模式隔离出来。

因果链可以概括为：

**工具重叠消解 + 困难候选集构造**  
→ **减少标签歧义，放大控制层冲突**  
→ **更清楚地测出自知、选择、拒答、组合调用四种能力边界**

为什么这套设计有效：

- **相似工具列表** 会迫使模型做真正的功能语义辨别，而不是靠表面词匹配
- **场景化工具表** 能暴露模型在不同职业/身份语境中的偏置
- **移除真值工具** 直接测“会不会诚实地说没有合适工具”，这是 agent 可靠性的关键
- **双工具 query** 则把“会选一个工具”与“会组合多个工具”区分开

| 设计选择 | 带来的诊断能力 | 代价/权衡 |
| --- | --- | --- |
| 工具合并/分解 | 让单标签评测更可信 | 依赖人工规则，存在主观性 |
| 相似工具候选集 | 放大细粒度语义混淆 | 难度受 embedding 相似度定义影响 |
| 场景化工具列表 | 能显式观察职业/身份偏置 | 场景覆盖有限，且列表是静态的 |
| 移除真值工具 | 直接测拒答与幻觉控制 | 不等同于真实系统中的动态工具发现 |
| 双工具任务 | 测基本组合调用能力 | 只覆盖 2-tool，不能代表长链规划 |

## Part III：证据与局限

### 关键证据信号

- **分析信号｜工具使用意识不足**  
  zero-shot 下，只有 ChatGPT 的 Accuracy 和 F1 同时超过 70%；很多开源模型即使加 few-shot 也只是部分改善，说明当前模型对“自己什么时候不够用”仍缺稳定自知。

- **分析信号｜可靠性是最大短板**  
  在“正确工具不存在”的子任务里，除 ChatGPT 外，大多数模型 CSR 仍低于 20%。这不是简单的语义检索问题，而是 **不会稳健地输出 None**，经常会硬选甚至幻觉出工具。最有力的对照是：**人类 CSR=96%，8 模型平均仅 8.59%**。

- **分析信号｜长度与场景敏感**  
  候选工具从 Top-5 扩到 Top-10/15 后，几乎所有模型的 CSR 都下降，说明长工具列表仍是明显负担。不同场景下表现也不均衡：学生场景通常更差，老人/艺术设计相关场景更好，说明模型存在显著领域偏置。

- **分析信号｜多工具能力“有潜力但不稳”**  
  多工具任务中，不同模型差异极大；而且一旦 prompt 明确要求“返回两个工具”，某些模型分数会显著上升。这说明模型未必完全不会多工具选择，而是 **控制策略和输出校准不稳**。

### 局限性

- **Fails when**: 需要动态新增工具、在线检索工具、超过两步的长链规划，或需要把工具执行结果反馈回下一步决策时，这个 benchmark 的静态文本设定不够覆盖。
- **Assumes**: 假设工具描述本身质量足够高，且基于 OpenAI plugin 生态具有代表性；数据生成与部分处理依赖 ChatGPT/GPT-4、text-embedding-ada-002，以及较重的人工作业。
- **Not designed for**: 不是为了评测 ReAct 后半段的参数填充、API 执行、结果整合、延迟/成本/失败恢复，也不是端到端真实 agent 部署评测。

这里还要特别注意复现性边界：  
论文明确使用了 **闭源 API（ChatGPT、GPT-4、OpenAI embedding）** 和 **人工校验**；而文中虽然声称数据/代码可用，但在给定文本里没有明确链接，因此从可复现角度看，扩展与复核成本并不低。

### 可复用组件

- **TOOLE 风格的多 prompt query 生成流程**：适合做更真实的 tool-triggering query 集合
- **工具合并/分解消重叠**：适合任何“候选工具语义重叠严重”的 benchmark
- **相似工具/缺失真值/场景化候选集构造**：可直接迁移到 agent controller 评测
- **工具描述分析范式**：描述长度、改写来源与下游模型适配性之间的关系，对工具文档设计很有参考价值

![[paperPDFs/Agentic_Reasoning_Benchmarks_Core_Mechanisms_of_Agentic_Reasoning_Tool_Use_Single_Turn_Tool_Use/ICLR_2024/2024_MetaTool_A_Benchmark_for_Controlling_Special_purpose_Large_Language_Models.pdf]]