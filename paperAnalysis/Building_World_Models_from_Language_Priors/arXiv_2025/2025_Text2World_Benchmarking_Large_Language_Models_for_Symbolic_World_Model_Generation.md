---
title: "Text2World: Benchmarking Large Language Models for Symbolic World Model Generation"
venue: ACL
year: 2025
tags:
  - Survey_Benchmark
  - task/symbolic-world-model-generation
  - task/llm-evaluation
  - execution-based-evaluation
  - multi-criteria-scoring
  - error-correction
  - dataset/TEXT2WORLD
  - repr/PDDL
  - opensource/full
core_operator: "以PDDL为统一符号接口，用可执行解析、结构相似度与组件级F1直接评测LLM从自然语言生成世界模型的能力"
primary_logic: |
  符号世界模型生成评测目标 → 从公开PDDL库收集并过滤原始域，再人工标注高层自然语言描述形成金标准数据 → 用可执行性、结构相似度和谓词/参数/前置条件/效果F1进行直接评分，并结合污染分析与错误纠正协议 → 揭示当前LLM在抽象描述到动作动态推断上的能力边界
claims:
  - "Claim 1: TEXT2WORLD 在论文的 n-gram 污染检测协议下平均污染率为 0.04，低于被比较先前设置的 0.47 [evidence: analysis]"
  - "Claim 2: 以 LLM 作为 PDDL 语义评审与人工标注的一致性仅为 Cohen's κ = 0.10，说明 LLM-judge 不足以作为稳健主评测 [evidence: analysis]"
  - "Claim 3: 在 16 个被测模型中 DeepSeek-R1 总体最佳，但 EC0 下 F1PRECOND/F1EFF 仍只有 57.6/58.8，显示最强模型仍未可靠掌握动作动态推断 [evidence: comparison]"
related_work_position:
  extends: "Bytesized32 (Wang et al. 2023a)"
  competes_with: "Large Language Models as Planning Domain Generators (Oswald et al. 2024); Generating Consistent PDDL Domains with Large Language Models (Smirnov et al. 2024)"
  complementary_to: "AgentGen (Hu et al. 2024b); Making Large Language Models into World Models with Precondition and Effect Knowledge (Xie et al. 2024)"
evidence_strength: moderate
pdf_ref: "paperPDFs/Building_World_Models_from_Language_Priors/arXiv_2025/2025_Text2World_Benchmarking_Large_Language_Models_for_Symbolic_World_Model_Generation.pdf"
category: Survey_Benchmark
---

# Text2World: Benchmarking Large Language Models for Symbolic World Model Generation

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2502.13092) · [Project](https://text-to-world.github.io)
> - **Summary**: 提出 TEXT2WORLD，用 PDDL 和执行式多指标把“从自然语言补全符号世界模型”的能力直接测出来，并证明当前最强推理模型仍主要卡在动作前置条件与效果的推断上。
> - **Key Performance**: DeepSeek-R1 在 EC0 下 EXEC 72.3%、F1PRECOND/F1EFF 57.6/58.8，EC3 后 EXEC 达 89.1%；TEXT2WORLD 的污染率仅 μ=0.04（对比先前设置 μ=0.47）。

> [!info] **Agent Summary**
> - **task_path**: 高层自然语言领域描述（给出谓词/动作签名与解释，但不显式给前置条件和效果） -> PDDL 符号世界模型
> - **bottleneck**: 现有评测过于依赖 LLM judge 或端到端规划成功率，既随机又间接，无法稳定定位模型是否真的学会动作动态
> - **mechanism_delta**: 用大规模 PDDL 基准 + 执行式多指标 + 组件级分解，替代主观或间接评测
> - **evidence_signal**: 16 个模型对比显示最佳模型在 abstract description 下前置条件/效果 F1 仍低于 60%，且 LLM 评审与人工一致性仅 κ=0.10
> - **reusable_ops**: [PDDL解析与可执行验证, 组件级F1分解, 基于解析错误的自纠循环]
> - **failure_modes**: [遗漏必要前置条件, 遗漏或误写动作效果]
> - **open_questions**: [如何在抽象描述下稳定推断隐藏动态, 如何扩展到连续/概率/多模态世界模型]

## Part I：问题与挑战

这篇论文要解决的，不是“把一段文字翻译成某种格式化代码”这么简单，而是更难的 **symbolic world model induction**：  
给定一个领域的自然语言描述、谓词说明和动作签名，模型需要**自行补全动作的前置条件与效果**，最终生成可执行的 PDDL 域模型。

### 真正的问题是什么
此前关于 LLM 生成符号世界模型的工作，主要有三个评测瓶颈：

1. **域覆盖太窄**：很多工作只测十几个域，结论不稳，也难泛化。
2. **评测有随机性**：把另一个 LLM 当 judge，会把生成误差和评审误差混在一起。本文先做了一个小实验，发现 LLM 评审与人工一致性只有 **Cohen's κ = 0.10**。
3. **评测过于间接**：只看下游 planning 成功率，无法回答“到底是语法坏了、谓词错了，还是动作动态没推出来”。

### 为什么现在要解决
因为 LLM 正在越来越多地被拿来做 agent、planning、world modeling，但如果没有一个**直接、可复现、可诊断**的 benchmark，社区其实并不知道模型究竟进步在了哪里。  
尤其是在 reasoning model 和 RL-based LLM 崛起之后，正需要一个基准去验证：这些“更会想”的模型，是否真的更会建世界模型。

### 任务接口与边界
TEXT2WORLD 的输入输出边界非常清晰：

- **输入**：自然语言领域描述  
  - General description
  - Predicates 的签名与解释
  - Actions 的签名与高层解释
- **输出**：完整 PDDL domain  
  - predicates
  - actions
  - parameters
  - preconditions
  - effects

关键边界条件在于：**动作描述故意保持抽象，不显式给出 preconditions/effects**。  
因此它测的是“从语言推断世界动态”的能力，而不是表层转写能力。

## Part II：方法与洞察

### 基准与评测协议

TEXT2WORLD 的构建流程是：

- 从公开仓库和规划竞赛收集 **1801** 个原始 PDDL 文件
- 经过语法校验、TF-IDF 去重、复杂度控制、token 长度过滤与人工筛选
- 得到 **264** 个高质量候选域
- 再由 6 位标注者人工撰写自然语言描述，双专家复核
- 最终形成 **103** 个金标准域  
  - 2 个 train exemplars
  - 101 个 test domains

质量控制上，人工复核的 **Fleiss κ = 0.82**，说明标注一致性较高。

评测指标设计为三层：

1. **Executability (EXEC.)**  
   输出的 PDDL 能否被 parser / validator 正确解析
2. **Structural Similarity (SIM.)**  
   预测 PDDL 与金标 PDDL 的 Levenshtein ratio
3. **Component-wise F1**  
   在可执行前提下，分别计算：
   - predicates F1
   - parameters F1
   - preconditions F1
   - effects F1

另外，作者还加入了：
- **Error correction protocol (ECk)**：让模型根据 parser 报错自修复
- **Data contamination analysis**：用 n-gram 匹配分析 benchmark 与训练语料污染风险

### 核心直觉

**What changed**  
从“让另一个 LLM 或下游规划结果来间接判断 world model 对不对”，改成“把输出直接当作 PDDL 程序去解析、验证，并按组件打分”。

**Which bottleneck changed**  
这一步实际上改变了两个测量瓶颈：

- **评测噪声下降**：主观 judge 变成确定性的 parser/validator
- **诊断粒度上升**：单一成功率被拆成谓词、参数、前置条件、效果四类信息瓶颈

**What capability changed**  
因此 TEXT2WORLD 不只是提供一个排行榜，而是能区分：

- 模型是**不会写合法 PDDL**
- 还是**能写出外形相似的壳子，但推不出真实动作动态**
- 还是**主要卡在 precondition/effect 这种因果约束层**

**Why this works**  
因为 PDDL 是一种可执行的符号接口，天然适合把 syntax correctness、structural matching、semantic completeness 分开看；而抽象描述又保留了推断负担，所以 benchmark 测到的是更“真”的 world modeling，而不是模板记忆。

### 战略权衡

| 设计选择 | 改变的瓶颈 | 获得的能力 | 代价 |
| --- | --- | --- | --- |
| 用 PDDL 作为统一表示 | 输出可被确定性检查 | 能直接测 executability 与组件正确性 | 只覆盖离散、符号化环境 |
| 使用抽象描述而非 concrete description | 保留隐含动态推断难度 | 真正测 world modeling，而非文本转写 | 分数更低，任务更难 |
| EXEC + SIM + component F1 | 避免单一指标失真 | 能定位错在谓词、参数、前置还是效果 | EXEC=0 时无法细分后续语义 |
| 引入 ECk 自纠协议 | 降低纯语法报错的干扰 | 估计“生成+修复”能力上限 | 混入了工具反馈利用能力 |
| 人工标注 + 双专家复核 + 污染分析 | 降低标注噪声与数据记忆干扰 | 排行榜更可信 | 成本高，污染也无法绝对清零 |

## Part III：证据与局限

### 关键证据信号

- **分析信号：LLM judge 不可靠**  
  人工与 LLM 在 PDDL 语义错误检测上的一致性只有 **κ = 0.10**。这直接支持了作者的核心主张：world model 评测不能继续主要依赖 LLM-based judging。

- **比较信号：最强模型仍远未“会建世界”**  
  在 16 个模型中，**DeepSeek-R1** 最强，但在零纠错 EC0 下：
  - EXEC = **72.3%**
  - F1PRECOND = **57.6**
  - F1EFF = **58.8**  
  这说明当前最强 reasoning model 也还不能稳定恢复动作动态。

- **比较信号：RL-reasoning 模型整体更强**  
  DeepSeek-R1、o3-mini、o1-mini 等 reasoning/RL 训练模型在 executability、similarity 和 component F1 上整体领先，说明“会推理”确实有助于 world model generation。

- **干预信号：纠错显著提升表现**  
  例如 gpt-4o-mini 的 EXEC 从 **48.5% 提升到 72.3%**；作者的 ANOVA 也显示多轮纠错带来显著改进。说明当前不少失败并非完全不会，而是“第一次输出不稳、但可被解析器反馈修复”。

- **诊断信号：结构相似不等于真的正确**  
  一个很有说服力的例子是：**LLaMA-3.1-70B 在 EC0 下 SIM 达 83.6，但 EXEC 为 0**。  
  这表明如果只看字符串相似度，会严重高估模型能力；也恰好证明了 TEXT2WORLD 这种执行式评测的必要性。

- **归因信号：真正瓶颈是隐含动态推断**  
  当把 abstract description 换成显式给出 preconditions/effects 的 concrete description 后，模型表现普遍提升；再加上人工错误分析里 **IncompleteModeling / 缺失必要 preconditions/effects** 是主要错误类型，基本可以确认：  
  **真瓶颈不是语法，而是从抽象语言恢复因果约束。**

### 局限性

- **Fails when**: 目标域超出 PDDL 的离散符号表示；或存在大量与金标语义等价但表面写法差异很大的实现；或域复杂度超出其筛选上限（如更多 predicates/actions、连续/概率/时序动态）。
- **Assumes**: 任务可被 PDDL 规范表达；人工描述足够高质量；parser/validator 能覆盖目标语法；实验中部分强模型依赖闭源 API，开源模型评测依赖 A100 80GB；且 PDDL 语料广泛存在，污染只能降低、不能彻底消除。
- **Not designed for**: 神经隐变量 world models、连续控制环境、纯下游 planning success 评测、多模态/具身输入场景、跨语言描述建模。

### 可复用组件

这篇工作的可复用价值很高，主要包括：

- **TEXT2WORLD 数据集本身**
- **PDDL 直接评测流水线**
- **component-wise F1 诊断框架**
- **syntax / semantic error taxonomy**
- **基于 parser 反馈的 error-correction protocol**

如果你要做“LLM 作为 world model”的后续工作，这篇论文最值得继承的，不是某个 prompt，而是它的**评测分解方式**：  
先把“能不能执行”与“是否真懂动作动态”分开，再把后者拆成 predicates / parameters / preconditions / effects 四层去看。

## Local PDF reference

![[paperPDFs/Building_World_Models_from_Language_Priors/arXiv_2025/2025_Text2World_Benchmarking_Large_Language_Models_for_Symbolic_World_Model_Generation.pdf]]