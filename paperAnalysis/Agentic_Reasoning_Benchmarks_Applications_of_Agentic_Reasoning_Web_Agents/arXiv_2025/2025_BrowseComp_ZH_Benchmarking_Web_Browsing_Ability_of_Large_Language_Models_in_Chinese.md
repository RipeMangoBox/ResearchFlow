---
title: "BrowseComp-ZH: Benchmarking Web Browsing Ability of Large Language Models in Chinese"
venue: arXiv
year: 2025
tags:
  - Survey_Benchmark
  - task/MLLM-evaluation
  - reverse-question-design
  - human-in-the-loop
  - multi-engine-validation
  - dataset/BrowseComp-ZH
  - opensource/partial
core_operator: "以唯一事实答案为起点，反向构造中文多约束网页问题，并通过三搜索引擎首屏过滤与人机协同去歧义来诊断中文网页浏览能力"
primary_logic: |
  评测中文网页浏览能力 → 从短事实答案反向设计原生中文多约束问题并做三搜索引擎首屏难度筛查 → 通过AI+人工完成答案唯一性校验并以准确率/ECE评分 → 揭示LLM在中文网页多跳检索、推理与证据对齐上的能力边界
claims:
  - "Claim 1: BrowseComp-ZH包含289个覆盖11个领域的中文高难度网页问题，并通过三搜索引擎首屏不可直达和人机协同复核来控制题目难度与答案唯一性 [evidence: analysis]"
  - "Claim 2: 在20多个被评测系统中，最佳系统OpenAI DeepResearch准确率仅42.9%，而Qwen2.5-72B-Instruct、GPT-4o、Claude-3.5-Sonnet等多数基础LLM低于10% [evidence: comparison]"
  - "Claim 3: 推理增强与多轮检索通常带来明显收益，但检索接入并非总是正收益；DeepSeek-R1在无搜索时为23.2%，开启搜索后降至7.6% [evidence: comparison]"
related_work_position:
  extends: "BrowseComp (Wei et al. 2025)"
  competes_with: "Level-Navi Agent (Hu et al. 2024); Chinese Dynamic Question Answering Benchmark (Xu et al. 2024)"
  complementary_to: "ReAct (Yao et al. 2023); WebGLM (Lai et al. 2025)"
evidence_strength: moderate
pdf_ref: paperPDFs/Agentic_Reasoning_Benchmarks_Applications_of_Agentic_Reasoning_Web_Agents/arXiv_2025/2025_BrowseComp_ZH_Benchmarking_Web_Browsing_Ability_of_Large_Language_Models_in_Chinese.pdf
category: Survey_Benchmark
---

# BrowseComp-ZH: Benchmarking Web Browsing Ability of Large Language Models in Chinese

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2504.19314), [GitHub](https://github.com/PALIN2018/BrowseComp-ZH)
> - **Summary**: 这篇工作构建了首个原生中文网页浏览能力基准，用“从唯一事实答案反向写题 + 双阶段质控”把中文网页中的多跳检索、推理与证据对齐难题变成可稳定测量的问题。
> - **Key Performance**: 最佳系统 OpenAI DeepResearch 仅 42.9% 准确率；DeepSeek-R1 无搜索 23.2%，开启搜索后反而降至 7.6%。

> [!info] **Agent Summary**
> - **task_path**: 中文多约束网页问题 / 中文互联网信息环境 -> 简短且唯一的事实答案 + 置信度
> - **bottleneck**: 现有英文浏览基准与常规QA集无法覆盖中文网页的碎片化索引、平台异构和隐式语言现象，导致真实浏览能力测不准
> - **mechanism_delta**: 用原生中文反向造题、三搜索引擎首屏不可解筛查、以及人机协同答案唯一性验证，替代直接翻译英文基准
> - **evidence_signal**: 20+ 系统横评中最佳仅 42.9%，且多轮检索系统明显优于单轮检索或纯参数模型
> - **reusable_ops**: [reverse-question-design, first-page-search-filtering, human-in-the-loop uniqueness check]
> - **failure_modes**: [单轮检索覆盖不到多跳约束, 检索噪声会覆盖模型原有较准确的内部知识]
> - **open_questions**: [如何在网页持续变化下维护答案稳定性, 如何把检索失败与证据对齐失败分开诊断]

## Part I：问题与挑战

这篇论文真正要解决的，不是“再做一个中文问答集”，而是**如何在中文网页生态里，可靠测出大模型的网页浏览能力**。

### 1. 现有评测为什么不够
作者指出两类现有评测都有明显缺口：

1. **英文网页基准的问题**  
   像 BrowseComp 这类工作已经开始测“浏览能力”而不只是“记忆能力”，但主要建立在英文网页生态上。  
   中文网页并不是英文网页的简单翻译版：  
   - 信息分散在百度百科、知乎、政府门户、垂类站点等异构平台  
   - 命名不一致，别称、缩写、旧称很多  
   - 搜索引擎对深层页面和长尾内容的索引不稳定  
   - 中文里省略、隐喻、语境依赖更强，关键词检索路径更容易失效

2. **常规QA/检索基准的问题**  
   很多现有数据集虽然也测多跳问答，但答案常常能被简单关键词命中，或者主要依赖 Wikipedia 这类结构化来源。  
   这会让评测更像在测“关键词撞中能力”，而不是测：
   - 是否会规划搜索路径
   - 是否能跨页面整合证据
   - 是否能在冲突信息中做对齐与裁决

### 2. 真正的瓶颈是什么
**真瓶颈是 measurement bottleneck：我们缺一个原生中文、低捷径、高可验证的浏览评测。**

如果题目直接翻译自英文：
- 可能在中文里变得不自然
- 可能因为中文网站结构不同而不可检索
- 也可能意外退化成简单关键词匹配题

所以作者的判断是：**中文网页浏览能力必须在中文信息生态里原生构造，而不是从英文迁移。**

### 3. 输入/输出接口与边界条件
这个 benchmark 的接口很清晰：

- **输入**：一个中文多约束问题
- **输出**：一个简短、客观、可核验的事实答案 + 模型自报置信度

其边界也很明确：
- 问题答案必须是**短答案**，如日期、数字、专有名词
- 每题必须能追溯到至少一个权威来源
- 若答案能在 Baidu / Bing / Google 任一搜索引擎首屏轻易出现，则题目要被修改或剔除
- 它主要测**检索 + 推理 + 证据整合**
- **不直接测**长程交互式网页操作、点击效率、工具延迟或成本

一句话概括 What/Why：  
**作者要解决的是“中文网页浏览能力长期被英文中心评测低估或误测”的问题，而现在正是LLM走向工具型 agent 的节点，这个能力必须被单独、严肃地测出来。**

## Part II：方法与洞察

### 1. 评测设计：把开放网页问题变成可稳定评分的问题

作者的整体策略可以概括成三步。

#### A. 从答案出发，反向写题
不是先写问题再找答案，而是：
1. 先选一个客观、具体、可核验的事实答案
2. 再围绕这个答案设计多约束问题

这些约束通常来自多个维度的组合：
- 时间
- 空间
- 类别
- 描述性特征

这样做的好处是：**终点先确定，问题自然更容易保证可验证性与唯一性。**

#### B. 用双阶段质控提升难度与唯一性
**阶段1：难度验证**
- 标注者互相交叉检查
- 只允许用搜索引擎，不用LLM
- 每题限时 10 分钟
- 能快速找到答案的题目剔除

**阶段2：唯一性验证**
- 让多个高性能 AI agent 先尝试作答
- 再由人工核查是否存在别的答案也满足所有约束
- 若存在多个成立答案，则该题作废

数据流大致是：
- 初始样本 480
- 去掉低难度题后剩 404
- 去掉歧义题后最终得到 **289** 题

#### C. 评测协议不仅看准确率，还看校准
作者评测了 20+ 个系统，覆盖：
- 开源模型
- 闭源 API
- AI 搜索产品

评分上除了准确率，还要求模型给出置信度，并计算 **ECE 校准误差**。  
这使 benchmark 不只是问“答对了没有”，还问“你是不是在错误答案上过度自信”。

同时，评测流程也有现实考虑：
- 普通 LLM 输出相对规范，可用正则抽取答案，再由 GPT-4o 评分
- AI 搜索产品更像完整产品，指令跟随不稳定，因此由人工抽取答案并核对

### 核心直觉

**这篇工作的关键，不是把题目变难，而是把“被错误地简化掉的测量维度”重新加回来。**

具体来说：

- **What changed**：从“英文题翻译/常规QA”改成“原生中文、答案先行、反向构造、多约束问题”
- **Which bottleneck changed**：把原先容易被关键词首屏命中、翻译伪难度、答案歧义污染的测量环境，改成更接近真实中文网页检索的环境
- **What capability changed**：benchmark 开始真正区分  
  1) 纯参数记忆  
  2) 推理能力  
  3) 多轮检索编排能力  
  4) 检索证据与内部知识的对齐能力

为什么这套设计有效？

因为它分别卡住了三个常见捷径：
1. **反向造题**：保证答案客观、可核验，不靠开放式解释
2. **首屏过滤**：去掉简单关键词命中题，逼迫多步搜索
3. **唯一性校验**：把开放网页中的歧义噪声压低，避免排行榜被“多答案题”污染

最终得到的不是“更花哨的数据集”，而是**更有诊断力的测量装置**。

### 2. 战略取舍

| 设计选择 | 解决的测量盲点 | 带来的能力诊断 | 代价 |
|---|---|---|---|
| 原生中文反向造题 | 避免英文翻译后的检索路径失真 | 更真实地测中文网页搜索与推理 | 标注成本高，依赖领域专家 |
| 三搜索引擎首屏过滤 | 去除关键词直达捷径 | 更能区分搜索规划能力 | 结果受搜索排序与时间变化影响 |
| 人机协同唯一性验证 | 降低多答案歧义 | 排行榜更稳定、可评分 | 扩展到更大规模较难 |
| 准确率 + 校准误差 | 只看对错看不出“盲目自信” | 可诊断检索后过度自信 | 置信度依赖模型自报，噪声较大 |
| 覆盖开源/闭源/API/搜索产品 | 单一模型族结论不稳 | 能区分系统设计范式差异 | 评测流程更复杂，部分需人工GUI操作 |

### 3. 这篇工作相对 prior work 的“能力跃迁”
相比旧基准，这篇工作最大的进步不是样本数，而是**测量分辨率**：

- 相比英文基准：它能测到中文网页独有的信息结构问题
- 相比普通检索QA：它更少被关键词命中捷径污染
- 相比只看最终准确率的评测：它还能暴露模型的校准失真和检索对齐失败

## Part III：证据与局限

### 1. 关键证据信号

**信号1｜横向对比：这个 benchmark 确实足够难**  
在 20 多个系统里，最佳系统 OpenAI DeepResearch 也只有 **42.9%**。  
而大量常见 LLM，例如 Qwen2.5-72B-Instruct、GPT-4o、Claude-3.5-Sonnet，都在 **10% 以下**。  
这说明该 benchmark 没有被“参数记忆”或“简单搜索”轻易击穿。

**信号2｜成对分析：推理能力是明显增益项**  
作者比较了多组“同家族、是否带推理”的模型：
- DeepSeek-R1 vs DeepSeek-V3：**23.2% vs 8.7%**
- Claude-3.7-Sonnet vs Claude-3.5-Sonnet：**17.7% vs 5.5%**

结论很直接：**即使没有浏览能力，强推理也能显著提升在高难网页题上的表现。**  
说明 benchmark 不只测搜索，也测内部知识组织和约束推断能力。

**信号3｜系统设计分析：多轮检索比单轮检索更有效**  
多轮检索产品表现更好：
- DeepResearch：**42.9%**
- Doubao Deep Search：**26.0%**
- Perplexity Research：**22.6%**

而典型单轮检索系统更低：
- Kimi：**8.0%**
- Yuanbao：**12.2%**
- DeepSeek Search：**7.6%**

这支持了一个很重要的系统结论：  
**面对多约束中文问题，关键不是“能不能搜”，而是“能不能边搜边改写检索策略”。**

**信号4｜反常结果：检索接入不等于性能提升**  
最有价值的负面证据是：
- DeepSeek-R1 无搜索：**23.2%**
- DeepSeek-R1 开启搜索：**7.6%**

这说明如果系统不能把检索结果和内部知识做稳健对齐，外部搜索可能不是帮手，而是噪声源。  
也就是说，这个 benchmark 不只揭示“retrieval gap”，还揭示**retrieval-alignment gap**。

**信号5｜校准分析：检索后的自信不一定更可信**  
作者要求模型同时输出置信度，并观察到一些带搜索设置的系统校准更差。  
例如 DeepSeek-R1 的 ECE 从 **59%** 上升到 **65%**。  
这意味着模型在接入网页后，可能会**更相信自己，但并没有更正确**。

### 2. 1-2 个最值得记住的指标
- **42.9%**：当前最佳系统在 BrowseComp-ZH 上的准确率上限，说明中文网页浏览远未解决
- **23.2% → 7.6%**：同一模型接入搜索后反而退化，直接暴露“检索证据对齐”是独立难点

### 3. 局限性
- **Fails when**: 你想用它评估长程交互式网页导航、页面操作规划、工具调用效率或引用链质量时，这个 benchmark 不够覆盖；它本质上仍是“短答案网页问答”而不是完整 browser-agent 任务。
- **Assumes**: 题目存在客观、简短、唯一且可核验的答案；当前搜索引擎排序与网页内容在一段时间内相对稳定；评测流程依赖 GPT-4o 评分、人工抽取 AI 搜索产品答案、以及高成本人工质检。
- **Not designed for**: 测量网页浏览的时间/金钱成本、安全合规性、工具使用轨迹质量、开放式生成质量，或非事实型任务。

### 4. 可复用组件
这篇工作最值得复用的不是排行榜，而是它的 benchmark recipe：

- **答案先行的反向造题**：适合构建高可验证、低歧义的开放域评测
- **多搜索引擎首屏过滤**：适合控制“关键词直达”捷径
- **AI + 人工的答案唯一性审查**：适合在开放网页环境中压低歧义噪声
- **准确率 + 校准联合报告**：适合把“答错还很自信”的系统行为显式暴露出来

### 5. 一句话总结 So what
这篇论文最重要的价值是：**它证明了中文网页浏览能力不是英文 benchmark 的平移版问题，而是一个独立、尚未被现有系统解决的评测与系统设计挑战。**

## Local PDF reference

![[paperPDFs/Agentic_Reasoning_Benchmarks_Applications_of_Agentic_Reasoning_Web_Agents/arXiv_2025/2025_BrowseComp_ZH_Benchmarking_Web_Browsing_Ability_of_Large_Language_Models_in_Chinese.pdf]]