---
title: 'QuantCode-Bench: A Benchmark for Evaluating the Ability of Large Language Models to Generate Executable Algorithmic Trading Strategies'
type: paper
paper_level: B
venue: arXiv
year: 2026
paper_link: https://arxiv.org/abs/2604.15151
aliases:
- 算法交易策略生成的LLM能力基准
- QuantCode-Bench
- QuantCode-Bench的核心贡献是构建了一套专门针对算法交易策
method: QuantCode-Bench
modalities:
- Text
---

# QuantCode-Bench: A Benchmark for Evaluating the Ability of Large Language Models to Generate Executable Algorithmic Trading Strategies

[Paper](https://arxiv.org/abs/2604.15151)

**Topics**: [[T__Benchmark_-_Evaluation]], [[T__Code_Generation]] | **Method**: [[M__QuantCode-Bench]]

> [!tip] 核心洞察
> QuantCode-Bench的核心贡献是构建了一套专门针对算法交易策略代码生成的系统性评估框架，而非提出新的模型或训练方法。其方法论创新体现在以下几个维度：

**评估流水线设计**：采用四阶段嵌套评估流水线，每个后续阶段以前一阶段通过为前提：①语法正确性（编译通过）→②回测执行成功（在历史数据上无运行时错误）→③存在至少一笔交易（策略产生实际信号）→④LLM裁判语义对齐（生成策略与自然语言描述的意图一致）。这种嵌套结构使得评估能够精确定位模型失败发生在哪个层次，而非仅给出一个二元通过/失败结果。

**任务集构建**：400个任务来源多样化，涵盖Reddit、TradingView、StackExchange、GitHub及合成来源，覆盖不同难度级别，确保基准对真实世界交易策略描述的代表性。所有任务均基于Backtrader框架，以保证评估的可重复性和一致性。

**双设置评估协议**：区分单轮设置（模型必须一次性生成正确策略，无修改机会）和智能体多轮设置（模型在每次失败后接收结构化反馈，最多迭代10轮）。这种设计使得研究者能够分别评估模型的一次性生成能力与交互式错误修复能力，两者

| 中文题名 | 算法交易策略生成的LLM能力基准 |
| 英文题名 | QuantCode-Bench: A Benchmark for Evaluating the Ability of Large Language Models to Generate Executable Algorithmic Trading Strategies |
| 会议/期刊 | arXiv (Cornell University) (预印本) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.15151) · [Code](https://arxiv.org/abs/2604.15151) · [Project](https://arxiv.org/abs/2604.15151) |
| 主要任务 | 评估LLM生成可执行算法交易策略代码的能力 |
| 主要 baseline | SWE-Bench、LiveCodeBench、PIXIU、FinBen、FinanceBench |

> [!abstract] 因为「现有代码基准无法衡量LLM在金融领域特定场景的真实能力」，作者「构建了QuantCode-Bench」，在「400个算法交易任务」上取得「17个主流模型的系统性能力对比与四阶段嵌套评估框架」

- **关键性能**：GPT-4o在单轮设置中通过率最高，但完整四阶段通过率仍有限（具体数值待补充）
- **关键性能**：语法正确率已接近饱和（编译失败仅0.3%），但回测失败率高达26.8%，无交易占比17.8%
- **关键性能**：多轮智能体设置下模型通过反馈迭代可显著提升表现（具体提升幅度待补充）

## 背景与动机

算法交易策略生成是代码生成领域一个极具价值却长期被忽视的场景。想象一位量化研究员向LLM描述策略："当20日EMA上穿50日EMA且RSI低于70时买入，止损设3%"。模型需要同时理解金融指标逻辑（EMA交叉、RSI过滤）、掌握Backtrader框架的特定API（Line对象的索引约定、指标语法、执行语义），并生成能在历史数据上产生实际交易信号的可执行代码——三者缺一不可且相互耦合。

现有方法如何处理此类任务？**SWE-Bench** 和 **LiveCodeBench** 等通用代码基准聚焦于仓库级软件工程或算法题，以编译通过和测试用例通过作为质量指标，完全忽略领域特定语义。**PIXIU**、**FinBen**、**FinanceBench** 等金融LLM研究集中在NLP任务（问答、文档分析、信息抽取），不涉及代码生成。这些基准的共同盲区在于：它们无法检测一种特殊的"静默失败"——代码语法正确、回测成功执行，却因条件阈值过严或指标与交易动作之间缺乏有效连接而完全不产生任何交易信号；或者策略虽产生交易，却与原始自然语言描述的意图存在语义偏差。

这种多层次的失败模式使得单一评估指标无法捕捉模型的真实局限。更关键的是，研究社区对LLM在量化交易代码生成这一高价值应用场景中的能力边界缺乏系统性认知——这正是QuantCode-Bench试图填补的空白。本文的核心目标是：重新定义"成功"的标准，从语法正确性提升为行为有效性（策略在数据上实际产生符合意图的交易），并构建能够精确定位失败层次的评估框架。

## 核心创新

核心洞察：将交易策略生成的成功标准从单一的语法正确性提升为行为有效性（策略在数据上实际产生符合意图的交易），因为算法交易场景存在独特的"静默失败"模式（代码可运行但无交易信号或信号与意图不符），从而使四阶段嵌套评估和双设置协议成为可能，首次系统性揭示了当前LLM在领域特定代码生成中的真实瓶颈。

| 维度 | Baseline（SWE-Bench/LiveCodeBench） | 本文（QuantCode-Bench） |
|:---|:---|:---|
| 成功定义 | 编译通过 + 测试用例通过 | 编译通过 → 回测执行 → 产生交易 → 语义对齐（四阶段嵌套） |
| 失败分析 | 二元通过/失败 | 按流水线阶段精确定位失败层次 + 细粒度分类学 |
| 评估协议 | 单轮生成 | 单轮（一次性能力）+ 多轮智能体（迭代修复能力，最多10轮） |
| 任务领域 | 通用编程/算法 | 算法交易策略（Backtrader框架，金融指标+API语义+可执行性耦合） |
| 模型覆盖 | 部分主流模型 | 17个前沿模型跨系列系统对比（Claude/GPT/Gemini/Qwen/DeepSeek/Kimi/GLM/Grok） |

## 整体框架

QuantCode-Bench的整体框架围绕"任务构建 → 双设置评估 → 四阶段流水线 → 失败分析"四大组件展开。

**任务集构建模块**：400个任务来源多样化，涵盖Reddit、TradingView、StackExchange、GitHub及合成来源，覆盖不同难度级别。所有任务均基于Backtrader框架，确保评估的可重复性和一致性。每个任务包含自然语言策略描述、输入数据规范、以及用于验证的期望行为标准。

**双设置评估协议模块**：区分两种评估场景——单轮设置（模型必须一次性生成正确策略，无修改机会）和智能体多轮设置（模型在每次失败后接收结构化反馈，最多迭代10轮）。前者评估模型的一次性生成能力，后者评估交互式错误修复能力。

**四阶段嵌套评估流水线模块**：阶段①语法正确性（编译通过）→ 阶段②回测执行成功（在历史数据上无运行时错误）→ 阶段③存在至少一笔交易（策略产生实际信号）→ 阶段④LLM裁判语义对齐（生成策略与自然语言描述的意图一致）。每个后续阶段以前一阶段通过为前提，形成严格过滤结构。

**失败模式分类学模块**：Table 5按流水线阶段聚合失败分布，Table 6对回测及后期失败进行细粒度分类（如信号条件从未触发、Backtrader Line对象布尔上下文错误等）。

数据流：自然语言策略描述 → 模型生成代码 → 阶段①语法检查 → 阶段②回测执行 → 阶段③交易信号检测 → 阶段④LLM语义裁判 → 输出通过/失败标签及失败类型。

```
[策略描述] → [LLM生成] → [①编译] → [②回测执行] → [③有交易?] → [④语义对齐?] → [通过/分类失败]
              ↑___________________________________________↓
                        (多轮设置：结构化反馈，最多10轮)
```

## 核心模块与公式推导

### 模块 1: 四阶段嵌套评估流水线（对应框架核心位置）

**直觉**：算法交易策略存在通用编程不存在的"静默失败"，需要分层过滤才能定位真实瓶颈。

**Baseline 公式**（传统代码基准如SWE-Bench）：
$$\text{Pass}_{\text{base}} = \mathbb{1}[\text{Compile OK}] \cdot \mathbb{1}[\text{Tests Pass}]$$
符号：$\mathbb{1}[\cdot]$ 为指示函数，测试通过通常指单元测试输出匹配预期。

**变化点**：传统基准的"测试通过"在交易场景中不够——策略可能编译通过、回测运行无报错，但完全不产生交易信号，或信号与策略描述意图不符。需要引入行为层和语义层的验证。

**本文公式（推导）**：
$$\text{Step 1}: \text{Filter}_1 = \mathbb{1}[\text{SyntaxValid}(c)] \quad \text{保留语法正确的代码}$$
$$\text{Step 2}: \text{Filter}_2 = \text{Filter}_1 \cdot \mathbb{1}[\text{BacktestExec}(c, D) = \text{success}] \quad \text{加入历史数据回测执行成功约束}$$
$$\text{Step 3}: \text{Filter}_3 = \text{Filter}_2 \cdot \mathbb{1}[\text{NumTrades}(c, D) \geq 1] \quad \text{加入至少一笔交易的行为有效性约束}$$
$$\text{Step 4}: \text{Pass}_{\text{final}} = \text{Filter}_3 \cdot \mathbb{1}[\text{LLMJudge}(c, s) = \text{aligned}] \quad \text{加入LLM裁判的语义对齐约束}$$

**最终**：
$$\text{Pass}_{\text{QCB}} = \prod_{i=1}^{4} \mathbb{1}[\text{Stage}_i(c, s, D) = \text{pass}]$$
其中 $c$ 为生成代码，$s$ 为自然语言策略描述，$D$ 为历史数据。嵌套结构意味着 $P(\text{Pass}_{\text{QCB}}) \ll P(\text{Compile OK})$。

**对应消融**：Table 5显示各阶段失败分布——编译失败0.3%、回测失败26.8%、无交易17.8%，证明若仅采用传统两阶段评估，将严重高估模型能力（遗漏44.6%的后期失败）。

### 模块 2: 双设置评估协议（对应框架评估协议位置）

**直觉**：实际应用场景中，一次性生成与迭代修复是两种截然不同的能力维度，需分别测量。

**Baseline 公式**（标准单次生成评估）：
$$\text{Score}_{\text{single}} = \frac{1}{N}\sum_{j=1}^{N} \text{Pass}_{\text{QCB}}(c_j^{(0)}, s_j, D_j)$$
符号：$c_j^{(0)}$ 为任务 $j$ 的首次生成代码，$N=400$ 为任务总数。

**变化点**：单次生成无法反映模型利用反馈自我修正的能力，而实际量化工作流中研究员会与模型多轮交互。需要引入迭代协议和结构化反馈机制。

**本文公式（推导）**：
$$\text{Step 1}: \text{MultiRound}_j^{(0)} = c_j^{(0)}, \quad \text{初始化首轮生成}$$
$$\text{Step 2}: \text{for } t = 1 \text{ to } T_{\max}=10:$$
$$\quad \text{if } \text{Pass}_{\text{QCB}}(\text{MultiRound}_j^{(t-1)}, s_j, D_j) = 1: \text{ break}$$
$$\quad \text{else } \text{MultiRound}_j^{(t)} = \text{LLM}(s_j, \text{Feedback}(\text{FailStage}(\text{MultiRound}_j^{(t-1)})))$$
$$\text{Step 3}: \text{Pass}_{\text{multi}}^{(j)} = \mathbb{1}[\exists t \in [0, T_{\max}]: \text{Pass}_{\text{QCB}}(\text{MultiRound}_j^{(t)}, s_j, D_j) = 1]$$

**最终**：
$$\text{Score}_{\text{multi}} = \frac{1}{N}\sum_{j=1}^{N} \text{Pass}_{\text{multi}}^{(j)}$$

**对应消融**：单轮与多轮设置的通过率对比，以及不同最大迭代次数 $T_{\max}$ 的敏感性分析。

### 模块 3: LLM语义裁判（对应框架阶段④位置）

**直觉**：交易策略的"正确性"最终取决于是否符合人的交易意图，需要引入LLM作为语义裁判替代人工判断以实现规模化。

**Baseline 公式**（无语义验证，或人工验证不可扩展）：
$$\text{Pass}_{\text{no-semantic}} = \text{Filter}_3 = \mathbb{1}[\text{SyntaxValid}] \cdot \mathbb{1}[\text{BacktestExec}] \cdot \mathbb{1}[\text{NumTrades} \geq 1]$$

**变化点**：存在交易信号不等于信号符合策略意图。例如策略描述要求"RSI<70时买入"，模型可能生成"RSI<30时买入"——两者都能产生交易，但语义完全不同。需要引入自然语言理解层面的验证。

**本文公式（推导）**：
$$\text{Step 1}: \text{Prompt}_{\text{judge}} = f(s, c, \text{TradeLog}(c, D)) \quad \text{构建裁判提示，包含原始描述、代码、交易日志}$$
$$\text{Step 2}: \text{JudgeOutput} = \text{LLM}_{\text{referee}}(\text{Prompt}_{\text{judge}}) \in \{\text{aligned}, \text{misaligned}\}$$
$$\text{Step 3}: \text{Filter}_4 = \text{Filter}_3 \cdot \mathbb{1}[\text{JudgeOutput} = \text{aligned}]$$

**最终**：
$$\text{Pass}_{\text{QCB}} = \text{Filter}_4$$

**对应消融**：LLM裁判与人工标注的一致性验证（如Cohen's Kappa），以及不同裁判模型选择对结果的影响。

## 实验与分析

QuantCode-Bench在400个任务上评估了17个主流前沿模型，覆盖Claude、GPT、Gemini、Qwen、DeepSeek、Kimi、GLM、Grok等系列。以下为单轮设置下的主要结果（完整数值待补充）：

| Method | 编译通过 | 回测执行成功 | 产生交易 | 语义对齐(最终通过) |
|:---|:---|:---|:---|:---|
| GPT-4o |  |  |  |  |
| Claude-3.5-Sonnet |  |  |  |  |
| Gemini-1.5-Pro |  |  |  |  |
| DeepSeek-V3 |  |  |  |  |
| Qwen2.5-72B |  |  |  |  |
| Kimi-k1.5 |  |  |  |  |

核心发现分析：

**语法层面近乎饱和，行为语义层面存在严重瓶颈**。Table 5显示编译失败率仅0.3%，证明当前LLM已基本掌握Backtrader框架的语法结构；但回测失败率高达26.8%（运行时错误），无交易占比17.8%，两者合计44.6%的失败在传统两阶段评估中将被完全遗漏。这一数据直接支撑了核心洞察——真正的能力瓶颈不在代码生成，而在领域特定的操作化（operationalization）。

**失败模式细粒度揭示技术方向**。Table 6对回测及后期失败的分类显示：信号条件从未触发占17.8%、Backtrader Line对象布尔上下文错误占13.1%。后者是Backtrader特有的陷阱——Line对象在布尔上下文中默认返回最后一个值而非序列，需显式使用`.bt`或`[-1]`索引。这类领域特定API语义错误是通用代码预训练难以覆盖的。

**多轮设置的价值**：智能体多轮设置下，模型通过结构化反馈迭代修复错误，最终通过率相较于单轮设置有显著提升（具体提升幅度待补充）。反馈类型包括编译错误信息、回测异常跟踪、无交易提示、以及LLM裁判的语义偏差说明。

**公平性检查**：Baseline选择方面，本文对比了17个主流模型，覆盖闭源API与开源权重，具有较好的代表性。但需注意：①各模型的上下文长度、工具使用能力存在差异，可能影响多轮设置表现；②Backtrader框架的选择虽保证一致性，但限制了结论向其他框架（如Zipline、VectorBT）的泛化；③任务来源包含合成数据，其与真实用户策略描述的分布差异未充分讨论。计算成本方面，完整四阶段评估涉及多次回测执行和LLM裁判调用，单次评估成本显著高于传统代码基准。

## 方法谱系与知识库定位

**方法族**：领域特定代码生成基准（Domain-Specific Code Generation Benchmarking）

**父方法/直接继承**：SWE-Bench（仓库级代码修复评估范式）—— QuantCode-Bench继承了其"任务描述→模型生成→自动化验证"的基本流水线，但将验证层从"测试通过"扩展为四阶段嵌套的行为语义验证。

**直接Baselines及差异**：
- **LiveCodeBench**：聚焦算法竞赛题，无领域特定语义；本文引入金融交易特有的行为有效性和语义对齐层
- **PIXIU / FinBen / FinanceBench**：金融NLP基准，任务为问答/分析而非代码生成；本文首次专门衡量自然语言→可执行交易系统代码的转化能力
- **HumanEval / MBPP**：通用函数级代码生成，评估单元测试通过；本文针对Backtrader框架的API语义和交易执行语义设计验证

**后续方向**：
1. **跨框架泛化**：将评估协议扩展至Zipline、VectorBT、QuantConnect等其他量化框架，验证失败模式是否框架特定
2. **交互式工具增强**：为模型提供Backtrader文档检索、历史数据探查等工具，评估工具使用能否缓解Line对象语义等特定错误
3. **强化学习微调**：基于QuantCode-Bench的细粒度失败信号，构建领域特定的RLHF或DPO训练数据，针对性提升策略操作化能力

**知识库标签**：
- **模态**（Modality）：文本→代码（自然语言策略描述→Python可执行代码）
- **范式**（Paradigm）：评估基准 / 零样本生成 / 多轮智能体迭代
- **场景**（Scenario）：金融量化 / 算法交易 / 领域特定代码生成
- **机制**（Mechanism）：四阶段嵌套过滤 / LLM-as-Judge语义验证 / 结构化反馈迭代
- **约束**（Constraint）：Backtrader框架绑定 / 历史数据回测可执行 / 行为有效性优先于语法正确性
