---
title: "API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs"
venue: EMNLP
year: 2023
tags:
  - Survey_Benchmark
  - task/tool-use-evaluation
  - executable-benchmark
  - multi-agent-data-generation
  - api-search
  - dataset/API-Bank
  - opensource/full
core_operator: 用可执行API环境把工具使用拆成调用、检索、规划三层能力，并以真实执行结果评估工具增强LLM
primary_logic: |
  用户工具使用需求调研 → 将能力定义为 Call / Retrieve+Call / Plan+Retrieve+Call，并构建73个可执行API与314条人工标注对话 → 用API调用正确率和回复ROUGE-L评测模型，同时用多代理生成1,888条训练对话验证可提升性 → 揭示当前LLM在API检索、规划、调用格式与参数约束上的能力边界
claims:
  - "GPT-4在最难的Plan+Retrieve+Call设置上达到70.00%正确率，显著高于GPT-3.5-turbo的22.00%，说明复杂工具规划仍是主要能力分水岭 [evidence: analysis]"
  - "基于API-Bank训练的Lynx-7B将总体API使用正确率从Alpaca-7B的15.19%提升到39.58%，接近GPT-3.5-turbo的47.16% [evidence: comparison]"
  - "Multi-agent生成训练数据的可用率为94%，比单代理self-instruct高89个百分点，并将单对话构造成本从约8美元降至0.1美元 [evidence: analysis]"
related_work_position:
  extends: "N/A"
  competes_with: "APIBench (Patil et al. 2023); ToolBench (Qin et al. 2023b)"
  complementary_to: "ReAct (Yao et al. 2022); Toolformer (Schick et al. 2023)"
evidence_strength: moderate
pdf_ref: paperPDFs/Agentic_Reasoning_Benchmarks_Core_Mechanisms_of_Agentic_Reasoning_Tool_Use_Multi_Turn_Tool_Use/EMNLP_2023/2023_API_Bank_A_Comprehensive_Benchmark_for_Tool_Augmented_LLMs.pdf
category: Survey_Benchmark
---

# API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [代码/数据](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank)
> - **Summary**: 这篇论文提出了一个面向工具增强 LLM 的可执行评测基准，把真实 tool use 拆成“调用—检索—规划”三层能力，并进一步用多代理自动合成训练数据来验证这些能力是可被系统性提升的。
> - **Key Performance**: GPT-4 在 **Plan+Retrieve+Call** 上达到 **70.00%** 正确率；Lynx-7B 总体正确率 **39.58%**，显著高于 Alpaca-7B 的 **15.19%**。

> [!info] **Agent Summary**
> - **task_path**: 用户自然语言需求 + API池/检索接口 + 对话历史 -> API调用序列与最终回复
> - **bottleneck**: 现有评测无法同时测到 API 选择、严格填参与多步规划，导致“会不会用工具”被混成一个模糊分数
> - **mechanism_delta**: 作者把工具使用显式拆成 Call、Retrieve+Call、Plan+Retrieve+Call，并放入可执行 API 系统中按真实执行效果打分
> - **evidence_signal**: GPT-4 与 GPT-3.5 在 Retrieve+Call 接近，但在 Plan+Retrieve+Call 上 70% vs 22%，说明该基准确实分离出了规划瓶颈
> - **reusable_ops**: [可执行API沙箱, 分层工具能力评测, 多代理数据合成]
> - **failure_modes**: [API检索失败, API幻觉, 调用格式不可解析]
> - **open_questions**: [如何扩展到更大真实API池且仍保持可复现, 如何约束模型严格遵循参数与调用协议]

## Part I：问题与挑战

这篇 paper 真正要解决的，不是“模型能不能输出一个看起来像 API 的字符串”，而是更贴近真实 agent 的问题：

**当 API 池很大、工具文档不可能全部塞进上下文、用户需求又常常需要多步调用时，LLM 能否稳定地完成 API 检索、参数填写、顺序规划，并根据执行反馈继续行动？**

### 1. 现有评测缺的是什么

作者认为，已有工具使用数据集/基准常有几个缺口：

- **只测调用，不测检索**：现实里用户可能有几百个 API，模型首先得知道“该找哪个工具”。
- **只测单步，不测规划**：很多需求要连续调用多个 API，前一步结果还是后一步输入。
- **只看文本形式，不看系统执行后果**：API 调用不是开放式写作，名字、参数、顺序错一点，系统状态就不对。
- **缺少真实交互环境**：如果没有可执行系统，很难判断模型到底是“答得像”还是“真的能用”。

### 2. 为什么现在必须解决

工具增强 LLM 已经从 demo 走向系统形态：搜索、日程、数据库、第三方服务、其他模型都可以变成工具。但此时最大的空白反而是：

- **我们并不知道现有 LLM 到底多会用工具**；
- **也不知道提升来自哪里：更好的模型、更好的训练数据，还是更好的接口协议**；
- **更缺少能诊断失败原因的 benchmark**。

所以 API-Bank 的价值不只是“给一个榜单”，而是把 tool use 拆成可诊断的子能力。

### 3. 输入/输出接口与评测边界

作者通过 500 名用户访谈，把需求抽象成两个维度：

- **Few vs. Many APIs in Pool**
- **Single vs. Multiple API calls per turn**

最终落成三种能力层级：

1. **Call**：API 已知，只需正确调用；
2. **Retrieve+Call**：API 未知，先检索再调用；
3. **Plan+Retrieve+Call**：API 未知，且一个需求需要多步规划与调用链。

因此，benchmark 的标准输入/输出可以理解为：

- **输入**：用户需求 + 对话历史 + 已知 API 文档或统一检索接口
- **输出**：规范化 API 调用串 + 最终自然语言回复

### 4. 边界条件

这个 benchmark 的评测边界也很明确：

- 主要是**英文**；
- 评测集基于 **73 个可执行 API**；
- 采用统一的 API 调用格式；
- 对外部检索型 API 的结果做了**冻结/硬编码**，以保证复现。

也就是说，它追求的是**可执行真实性 + 可重复性**，而不是完全开放世界。

---

## Part II：方法与洞察

### 1. 评测框架：把“会不会用工具”拆开测

API-Bank 的核心设计不是新模型，而是**新的测量协议**。

#### (a) 可执行评测系统

作者实现了一个真实可运行的 API 环境：

- **73 个 API**
- **314 条人工标注评测对话**
- **753 次 API 调用**
- 支持数据库读写、状态修改、固定外部返回等

这意味着评测不再只是对比字符串，而是看：

- 是否调用了**正确 API**
- 是否填了**正确参数**
- 是否造成了**正确系统效果**
- 是否给出了与执行结果一致的**最终回复**

#### (b) 专门的 API Search 接口

为了把“未知 API 池”的场景标准化，作者设计了一个专门的 **API Search** 工具：

- 模型先把需求压缩为关键词；
- 检索器根据关键词与 API 元信息的 embedding 相似度返回最相关 API；
- 在 Retrieve+Call / Plan+Retrieve+Call 场景里，模型必须先搜索再调用。

这个设计的作用是：把“API 选择”从模糊能力变成一个**显式可测步骤**。

#### (c) 评分方式

评测分两部分：

- **API call correctness**：是否与人工标注在系统执行层面一致  
  这里不是只比字符串，而是比较是否执行了相同数据库操作、产生相同返回结果。
- **Response quality**：用 **ROUGE-L** 评估模型回复

对 benchmark 来说，真正关键的主信号是前者：**系统动作是否正确**。

### 2. 配套训练集：不仅能测，还想验证“能不能提升”

API-Bank 不只做评测，还构造了训练集：

- **1,888 条训练对话**
- **2,138 个 API**
- **1,000 个 domain**
- 总 benchmark 规模达到 **1,008 domains / 2,211 APIs / 2,202 dialogues**

这里的难点不是数量，而是**如何低成本生成既多样又像真的 tool-use 数据**。

### 3. Multi-agent 数据合成

作者发现，单条 self-instruct 式 prompt 很难同时满足：

- domain diversity
- API authenticity
- API diversity
- 三层工具能力覆盖

于是他们把复杂生成任务拆成 5 个 agent：

1. 生成 domain
2. 生成 API
3. 根据 API 和能力类型生成 query
4. 生成 API call / response
5. 用 tester agent 做质量过滤

这样一来，原本“单 prompt 过载”的问题，变成了**多步受约束生成**。

### 核心直觉

**作者真正改变的“因果旋钮”有两个：**

1. **测量层面**：  
   从“看模型输出像不像工具调用”  
   改成“看模型是否在可执行环境里完成正确动作”。

2. **数据层面**：  
   从“让一个 LLM 一次性生成完整工具对话”  
   改成“让多个 agent 分步骤生成并互相约束”。

这带来的能力变化是：

- 原来一个失败样本只知道“模型没答对”；
- 现在可以分解为：
  - 不会触发 API
  - 检索错 API
  - 参数填错
  - 调用格式错
  - 多步规划错
  - 回复与执行结果不一致

也就是说，API-Bank 的最大贡献是把 **tool use failure 从黑盒错误变成结构化错误**。

### 4. 战略取舍

| 设计选择 | 改变了什么瓶颈 | 得到什么能力诊断 | 代价 / 取舍 |
|---|---|---|---|
| 三层能力拆分 | 把“工具使用”从单一分数拆成调用/检索/规划 | 能区分 instruction-following 与 reasoning/planning 的差异 | 任务协议更复杂，人工标注更贵 |
| 可执行 API 环境 | 从文本表面匹配改成系统状态一致性 | 更接近真实服务可靠性 | 73 个 API 需要高实现成本，文中约 98 人日 |
| 统一 API Search | 把大 API 池检索标准化 | 可单独暴露 retrieval 失败 | 也限制了更自由的 agent tool-use 策略 |
| 人工测试 + 自动训练 | 同时兼顾真实性与规模 | 既能评测，也能验证训练价值 | train/test 分布存在偏移，测试域数量仍有限 |

---

## Part III：证据与局限

### 1. 关键实验信号

#### 信号 A：这个 benchmark 真的把“规划瓶颈”测出来了
最强的证据是 GPT-3.5 与 GPT-4 的分化方式。

- 在 **Retrieve+Call** 上，GPT-3.5 与 GPT-4 很接近：**38.52% vs 37.04%**
- 但在 **Plan+Retrieve+Call** 上，GPT-4 直接到 **70.00%**，而 GPT-3.5 只有 **22.00%**

这说明：
- 单步检索不是全部难点；
- **多步规划**才是高级模型真正拉开差距的地方。

这正是 benchmark 设计想揭示的东西。

#### 信号 B：训练数据确实能提升 tool use
Lynx-7B 用 API-Bank 训练后，相比 Alpaca-7B 有明显跃迁：

- 总体正确率：**15.19% → 39.58%**
- Call 正确率：**24.06% → 49.87%**
- 总体回复 ROUGE-L：**0.0318 → 0.3794**

这说明 API-Bank 不只是一个“难题集”，而是能形成**有效监督信号**。

#### 信号 C：多代理数据生成不只是便宜，而且质量更高
作者给出两个很关键的信号：

- Multi-agent 生成数据**可用率 94%**
- 单对话成本从人工约 **\$8** 降到 **\$0.1**

更重要的是，质量不是只靠量堆出来的。  
在与 ToolAlpaca 的对比中，API-Bank 训练出的 Lynx 即使样本更少，也在 Call 能力上略好：

- ToolAlpaca 训练：**53.88**
- API-Bank 训练：**54.64**

这说明作者的数据合成流程，至少在**任务贴合度**上是有效的。

#### 信号 D：不同模型暴露出不同失败模式
误差分析很有诊断价值：

- **Alpaca**：主要问题是 **No API Call**  
  说明基础 instruction-tuned 模型连“进入工具调用模式”都不稳定。
- **Lynx**：主要问题是 **API Hallucination**  
  说明训练后学会了工具调用框架，但会调用错误或虚构 API。
- **GPT-4**：主要问题是 **Failed API Retrieval** 与格式偏差  
  说明强推理能力不等于能严格服从受限的工具协议。

这也再次说明：  
**tool use 不是一个单能力，而是一组耦合能力。**

### 2. 局限性

- **Fails when**: API 池继续扩大且存在大量近义/歧义工具、任务允许并发或自由格式多 API 调用、或迁移到非英语环境时，当前统一的 API Search + 单次可解析调用协议会明显吃力。
- **Assumes**: 英文文本交互、预定义 API schema、固定 `[ApiName(...)]` 调用格式、可冻结的外部返回结果，以及较高的系统构建成本（73 个 API 约 98 人日；评测集人工标注约 \$8/对话）。
- **Not designed for**: 开放网页浏览型 agent、实时变化的商业 API、长时程自治规划、多模态工具链、安全/权限控制压力测试。

再补一个很实际的边界：

- **回复质量只用 ROUGE-L**。这对 benchmark 足够轻量，但会低估语义等价、词面不同的回答。
- **测试集只有 8 个 domain、73 个 API**。虽然是人工高质量可执行环境，但覆盖面仍有限，证据强度应保守看待。

### 3. 可复用组件

这篇 paper 最值得复用的不是 Lynx 本身，而是下面这些“评测/数据工程模块”：

- **可执行 API 沙箱**：带状态、可验证执行结果
- **三层能力拆分协议**：Call / Retrieve+Call / Plan+Retrieve+Call
- **统一 API Search 接口**：把大 API 池问题变成标准化检索步骤
- **多代理数据生成流水线**：domain → API → query → call/response → tester
- **结构化错误 taxonomy**：No API Call / Hallucination / Invalid Params / False Format 等

如果你以后要做 agent/tool-use benchmark，这些组件比单独的 leaderboard 更有迁移价值。

## Local PDF reference

![[paperPDFs/Agentic_Reasoning_Benchmarks_Core_Mechanisms_of_Agentic_Reasoning_Tool_Use_Multi_Turn_Tool_Use/EMNLP_2023/2023_API_Bank_A_Comprehensive_Benchmark_for_Tool_Augmented_LLMs.pdf]]