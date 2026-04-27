---
title: "ToolQA: A Dataset for LLM Question Answering with External Tools"
venue: NeurIPS
year: 2023
tags:
  - Survey_Benchmark
  - task/MLLM-evaluation
  - tool-use-evaluation
  - template-based-generation
  - programmatic-answer-generation
  - dataset/ToolQA
  - opensource/full
core_operator: 通过低预训练重叠参考语料、模板化问题生成与程序化答案生成，构造必须依赖外部工具才能回答的问答评测。
primary_logic: |
  评测真实工具使用能力 → 选择尽量不与预训练重叠的外部语料并配置13类工具 → 用人工筛选模板生成只能借助工具回答的问题并程序化生成标准答案 → 以开放式问答正确率与错误分析揭示模型的工具选择、组合与反馈利用边界
claims:
  - "ToolQA上的vanilla LLM基线在easy题平均成功率仅2.3%–5.6%，hard题仅1.4%–2.0%，说明该基准较好地压低了参数记忆带来的评测泄漏 [evidence: analysis]"
  - "在所评估的工具增强方法中，ReAct表现最佳，但平均成功率也只有easy 43.1%与hard 8.2%，表明多工具组合推理仍远未解决 [evidence: comparison]"
  - "对ReAct的错误分析显示，参数错误是首要失败来源，占easy/hard错误的44.56%与48.23%，其次是长上下文、错误数据源选择与幻觉 [evidence: analysis]"
related_work_position:
  extends: "N/A"
  competes_with: "API-Bank (Li et al. 2023); ToolBench (Qin et al. 2023)"
  complementary_to: "ReAct (Yao et al. 2023); Chameleon (Lu et al. 2023)"
evidence_strength: moderate
pdf_ref: paperPDFs/Agentic_Reasoning_Benchmarks_Core_Mechanisms_of_Agentic_Reasoning_Tool_Use_Single_Turn_Tool_Use/NeurIPS_2023/2023_ToolQA_A_Dataset_for_LLM_Question_Answering_with_External_Tools.pdf
category: Survey_Benchmark
---

# ToolQA: A Dataset for LLM Question Answering with External Tools

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [GitHub](https://github.com/night-chen/ToolQA)
> - **Summary**: ToolQA通过构造“几乎不能靠模型记忆答出、必须调用外部工具”的问答样本，把LLM真实的工具使用能力从参数记忆中剥离出来做评测。
> - **Key Performance**: vanilla ChatGPT平均仅easy 5.6% / hard 2.0%；最佳ReAct基线为easy 43.1%、hard 8.2%。

> [!info] **Agent Summary**
> - **task_path**: 自然语言问题 + 参考语料/工具集合 -> 开放式正确答案
> - **bottleneck**: 现有评测无法区分“模型真的会用工具”与“模型只是记住了答案”
> - **mechanism_delta**: 用低预训练重叠参考语料 + 模板约束问题生成 + 程序化标准答案，强制问题依赖外部工具而非内部知识
> - **evidence_signal**: vanilla LLM近零表现，而ReAct虽显著更好但hard题仍仅8.2%
> - **reusable_ops**: [reference-corpus-curation, template-based-question-generation, programmatic-answer-generation]
> - **failure_modes**: [argument-error, incorrect-data-source]
> - **open_questions**: [如何降低多工具调用中的参数错误, 如何在长上下文下稳定完成工具组合规划]

## Part I：问题与挑战

这篇论文真正要解决的，不是“再做一个QA数据集”，而是**修复工具使用评测里的因果混淆**：

- 现有LLM已经吸收了大量预训练知识；
- 许多所谓“tool-use QA”测试题，即使不给工具，模型也可能靠参数记忆答对；
- 因此我们很难判断：模型到底是在**调用外部工具**，还是在**回忆训练时见过的信息**。

### 真正瓶颈是什么
真正瓶颈是**评测可识别性**，而不是单纯题目难度。  
如果 benchmark 本身不能保证“答案必须来自外部工具”，那任何高分都无法证明模型真的具备工具使用能力。

### 为什么现在必须解决
因为LLM的两个现实短板正好推动了工具增强研究爆发：

1. **幻觉**：需要外部事实校验或检索；
2. **数值/结构化推理弱**：需要计算器、SQL、Python、图查询等工具。

但如果评测本身把“记忆”与“工具使用”混在一起，后续方法改进就没有可靠目标。

### 输入/输出接口
ToolQA的实例由四部分组成：

- **question**
- **answer**
- **reference corpora**
- **available tools**

模型面对的是：  
**一个自然语言问题 + 一个可调用工具集合 + 外部参考语料**，最终输出**开放式答案**。  
评测不强制固定中间工具链，只看最后答案是否正确。

### 边界条件
ToolQA明确设了几个边界：

- 问题所需信息应能完全从参考语料获得；
- 问题应尽量不能被LLM内部知识直接回答；
- easy题偏单步信息提取，hard题偏多信息聚合、比较、计算和工具组合；
- 它评测的是**端到端工具增强问答能力**，不是固定API trace复现能力。

---

## Part II：方法与洞察

ToolQA的核心不是训练一个新模型，而是设计一个**更“干净”的测量装置**。  
作者用三阶段流程，把“是否真的需要工具”这件事做成数据构造约束。

### 1. 参考语料与工具设计
数据覆盖8个领域、13类工具。

**8个领域**：
- Flight
- Coffee
- Yelp
- Airbnb
- GSM8K
- DBLP
- SciREX
- Agenda

这些语料沿6个上下文维度选取：时间、空间、社会、科学、数学、个人。  
选取原则很明确：

- 尽量不与LLM预训练数据重叠；
- 能产生上下文依赖问题；
- 所需答案可由外部语料完整支持。

**13类工具**包括：
- 文本检索工具
- 数据库加载/过滤/取值
- WolframAlpha计算器
- 图加载/邻居查询/节点查询/边查询
- Python/SQL解释器
- Finish

这使 benchmark 不只测检索，还测**结构化查询、计算、图推理、代码桥接**。

### 2. Human-Guided Question Generation
作者没有完全手工写题，也没有完全让LLM自由生成，而是采用**人工引导的模板化生成**：

1. 让ChatGPT先给每个数据源生成约50个候选模板；
2. 人工筛掉两类坏模板：
   - vanilla ChatGPT成功率超过50%，说明可能能靠内部知识回答；
   - 问题询问信息并不在参考语料里；
3. 再从真实语料中采样字段，实例化成具体问题。

最终得到：
- easy模板 55 个，共 800 题
- hard模板 62 个，共 730 题

总计 **1530题**。

### 3. Programmatic Answer Generation
这是ToolQA很关键的一步。  
作者为每类工具实现了对应 operator，并根据模板预定义 tool chain，把填入模板的真实参数送进程序，**直接从参考语料里算出标准答案**。

这带来两个结果：

- 标签准确度高；
- 可以规模化生成多步工具问题，而不是依赖人工标注答案。

### 4. 评分协议
ToolQA评估的是**开放式最终答案正确率**，不是中间轨迹是否与“标准调用链”完全一致。  
作者对时间、价格、标点、冠词、空格等做标准化，然后用 exact match 统计成功率。

这和 API-Bank / ToolBench 一类“trace correctness”基准不同：  
ToolQA更关心的是**模型能否通过任意合理工具链拿到正确答案**。

### 核心直觉

**它改变的不是模型，而是测量约束。**

- **原来**：题目可能被参数记忆直接覆盖，导致“答对”不等于“会用工具”。
- **现在**：通过低重叠参考语料 + 模板筛选，把问题构造成“不给工具基本答不出”。
- **结果**：正确率更接近模型真实的工具选择、组合、执行反馈利用能力。

更因果地说：

**题目构造方式改变**  
→ **信息来源约束从“可选外部工具”变成“必须外部工具”**  
→ **评测对象从模糊的综合QA能力，收缩为更可诊断的工具使用能力**

这也是为什么ToolQA虽然看起来只是个数据集，但实际上是在改**benchmark测到的随机变量**。

### 为什么这个设计有效
因为它把原先最大的混杂因素——**预训练记忆泄漏**——显式压低了。  
一旦vanilla LLM接近零分，而tool-augmented方法明显更高，就能更可信地说明 benchmark 确实在测“外部工具使用”。

同时，hard题又进一步把难点推到：
- 多工具组合
- 中间结果传递
- 长上下文交互
- 失败后重规划

因此它不仅能测“会不会调一个工具”，还能暴露**组合式工具推理**的缺口。

### 战略权衡

| 设计选择 | 获得的能力 | 代价/牺牲 |
| --- | --- | --- |
| 低预训练重叠参考语料 | 减少参数记忆泄漏，更忠实测工具使用 | 数据覆盖需持续维护，随模型更新可能老化 |
| 模板 + 人工筛选 | 保证问题可答且必须用工具 | 自然语言多样性弱于真实用户提问 |
| 程序化答案生成 | 标签精确、扩展性强 | 问题类型受可编程算子与模板限制 |
| 开放式最终答案评分 | 不绑定唯一工具链，更贴近端到端能力 | 无法精确评价中间计划质量与最优性 |
| easy / hard分层 | 能区分单工具与多工具组合能力 | “hard”仍是人工定义，不等于真实世界所有复杂度 |

---

## Part III：证据与局限

### 关键证据信号

#### 信号1：benchmark确实压低了“背答案”捷径
最直接证据是 vanilla LLM 几乎全线低分：

- easy题：LLaMA-2/Falcon/ChatGPT/CoT 平均约 **2.3%–5.6%**
- hard题：平均约 **1.4%–2.0%**

这说明 ToolQA 中大多数题目并不能靠内部知识直接蒙对，benchmark 的“强制工具依赖”设计是有效的。

#### 信号2：现有工具增强方法有提升，但远没解决问题
工具增强方法里，ReAct最好：

- **easy：43.1%**
- **hard：8.2%**

这说明两件事同时成立：

1. 工具确实能显著提升LLM问答；
2. 真正难的是**多工具组合、长链交互与反馈驱动规划**。

换句话说，当前能力跃迁主要发生在“单步/浅层工具使用”，而不是“稳定的组合式代理推理”。

#### 信号3：错误分布揭示了真正短板不是知识，而是执行
ReAct错误分析中，最主要问题不是“不知道答案”，而是**不会稳定地把工具用对**：

- 参数错误：easy **44.56%**，hard **48.23%**
- 长上下文：hard升到 **16.63%**
- 错误数据源选择、低质量检索、幻觉也占较高比例

这很关键：  
ToolQA测出来的主要瓶颈是**工具调用参数化、数据源判别、上下文管理、执行反馈解释**，而不只是一般语言理解。

#### 信号4：更强模型会“创新”，但也更会“幻觉”
文中案例显示：

- GPT-3风格更像“按示例照抄”，easy题有时反而更稳；
- GPT-3.5在hard题上会尝试新工具组合（如改用SQLInterpreter），因此 hard 上更强；
- 但这种“创新”也伴随**虚构观察结果**的幻觉风险。

所以 ToolQA还能测出一个更细的边界：  
**不是所有创造性工具组合都是好事；创新与幻觉在当前代理式LLM里经常绑定出现。**

### 1-2个最重要指标
我认为这篇论文最值得记住的不是绝对分数，而是下面两个对比：

- **ChatGPT easy 5.6% vs ReAct easy 43.1%**：说明benchmark确实在测工具增益；
- **ReAct easy 43.1% vs hard 8.2%**：说明真正未解决的是多工具组合推理，而非单工具调用。

### 局限性
- Fails when: 需要评估开放互联网浏览、多轮真实API交互、权限管理或安全攻击时，ToolQA会失真，因为它主要是封闭参考语料上的单轮工具增强QA。
- Assumes: 需要可执行的工具环境、参考语料与程序化算子；同时假设参考语料相对低重叠于预训练数据，这个假设会随更强LLM与更晚训练语料而逐渐变弱。
- Not designed for: 评估固定工具链trace正确性、真实商业API生态中的鲁棒性/安全性、多模态工具使用，或人类主观偏好层面的交互质量。

### 复现与依赖
有两点需要明确：

1. **优点**：数据和代码开源，benchmark本体可复现性较好；
2. **依赖**：部分基线依赖闭源模型版本（如 ChatGPT）和外部服务（如 WolframAlpha API），因此分数会受API版本漂移和外部服务可用性影响。

### 可复用组件
这篇工作最可复用的不是某个单独工具，而是整套 benchmark 构造范式：

- **低记忆泄漏参考语料筛选**
- **人工引导模板生成**
- **程序化答案生成**
- **开放式最终答案评估**
- **工具使用错误分类框架**

如果以后要做网页代理、数据库代理、代码代理的 benchmark，这套思路都可以直接迁移。

## Local PDF reference

![[paperPDFs/Agentic_Reasoning_Benchmarks_Core_Mechanisms_of_Agentic_Reasoning_Tool_Use_Single_Turn_Tool_Use/NeurIPS_2023/2023_ToolQA_A_Dataset_for_LLM_Question_Answering_with_External_Tools.pdf]]