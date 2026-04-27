---
title: "InfoDeepSeek: Benchmarking Agentic Information Seeking for Retrieval-Augmented Generation"
venue: arXiv
year: 2025
tags:
  - Survey_Benchmark
  - task/rag-evaluation
  - task/agentic-information-seeking
  - dynamic-evaluation
  - llm-as-a-judge
  - agentic-rag
  - dataset/InfoDeepSeek
  - opensource/no
core_operator: "在动态网页环境中，用高难度且答案确定的问题集合与无金标文档的证据可答性指标，评测 Agentic RAG 的信息搜寻能力。"
primary_logic: |
  评测目标（真实网页中的 Agentic RAG 信息搜寻） → 按“确定性/高难度/多样性”构造并人工验证 245 个挑战问题 → 用 ACC、IA@k、EEU、IC 在无固定金标文档条件下评估检索与证据筛选 → 揭示当前系统受检索源质量、网页噪声与证据压缩能力限制
claims:
  - "在默认 DuckDuckGo 设置下，InfoDeepSeek 上的最佳模型 Gemini-2.5-Pro 仅达到 22.45% ACC 和 21.63% IA@5，说明该基准对当前 Agentic RAG 具有较高区分度和难度 [evidence: comparison]"
  - "将搜索引擎从 DuckDuckGo 切换到 Google 时，Gemini-2.5-Flash 的 ACC 从 14.29% 提升到 34.29%，DeepSeek-V3 从 8.98% 提升到 28.57%，表明检索源质量是主导瓶颈 [evidence: comparison]"
  - "针对 false-premise 问题采用单独评测 prompt 后，LLM 自动评测相对人工标注的准确率可从 95.57% 提升到 99.29% [evidence: comparison]"
related_work_position:
  extends: "CRAG (Yang et al. 2024)"
  competes_with: "BrowseComp (Wei et al. 2025); CRAG (Yang et al. 2024)"
  complementary_to: "Agentic Information Retrieval (Zhang et al. 2024); KwaiAgents (Pan et al. 2023)"
evidence_strength: moderate
pdf_ref: paperPDFs/Agentic_Reasoning_Benchmarks_Applications_of_Agentic_Reasoning_Autonomous_Research_Agents/arXiv_2025/2025_InfoDeepSeek_Benchmarking_Agentic_Information_Seeking_for_Retrieval_Augmented_Generation.pdf
category: Survey_Benchmark
---

# InfoDeepSeek: Benchmarking Agentic Information Seeking for Retrieval-Augmented Generation

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2505.15872), [Project](https://infodeepseek.github.io/)
> - **Summary**: 该工作提出 InfoDeepSeek，用真实动态网页环境中的高难度确定性问题与无金标文档评测协议，专门测量 Agentic RAG 的“信息搜寻”而不是只看最终答案。
> - **Key Performance**: 默认 DuckDuckGo 下最佳模型 Gemini-2.5-Pro 仅有 **22.45% ACC / 21.63% IA@5**；false-premise 专用评测 prompt 将自动评测准确率从 **95.57% 提升到 99.29%**。

> [!info] **Agent Summary**
> - **task_path**: 开放域复杂事实问题 + live web 搜索/浏览 -> 排序证据集 -> grounded answer
> - **bottleneck**: 静态语料与金标文档指标无法评估动态网页中的多步检索、证据筛选与噪声干扰
> - **mechanism_delta**: 把评测从“固定语料上的检索命中”改成“动态网页中证据是否足以回答、是否被有效利用、是否足够紧凑”
> - **evidence_signal**: 跨 LLM 与搜索引擎比较显示最好模型仍低分，且更换搜索引擎可带来 2x 以上 ACC 提升
> - **reusable_ops**: [difficulty-filtered query construction, false-premise-specific llm-judge prompts]
> - **failure_modes**: [retrieval interference from noisy web evidence, weak evidence compression on multi-hop and long-tail queries]
> - **open_questions**: [how to scale benchmark construction beyond manual curation, how to reduce retrieval interference without losing useful external evidence]

## Part I：问题与挑战

**What/Why：真正要测的不是“模型会不会答题”，而是“agent 能不能在真实网页里把对的证据找出来并压缩成可用上下文”。**

这篇论文抓得很准：Agentic RAG 相比传统 RAG，真正变化最大的不是最终生成器，而是**信息搜寻阶段**。现实中的 Deep Research 类系统已经在 OpenAI、Gemini、Perplexity 等产品中落地，所以现在最缺的不是另一个静态 QA 集，而是一个能反映真实部署条件的 benchmark。

### 现有评测为什么不够
现有 RAG benchmark 的核心问题有两个：

1. **环境太静态**
   - 固定语料库、固定候选文档、固定 gold document set。
   - 这和真实网页环境差异很大：网页会漂移、链接会失效、搜索结果会变、同一事实可能由多种来源支持。

2. **问题太简单**
   - 许多问题可以被模型参数知识或单轮搜索直接解决。
   - 这样测不出 agent 的关键能力：规划、多轮工具调用、跨源证据拼接、反思修正。

### 论文里的任务接口
- **输入**：一个复杂事实问题 + 动态 web 搜索/浏览工具
- **中间过程**：多步检索轨迹、观察历史、筛选后的证据集合
- **输出**：最终答案，以及可供诊断的信息搜寻质量

### 这个 benchmark 的边界条件
作者刻意把题目限制为三类性质同时成立：
- **Determinacy**：答案明确、唯一、可验证、尽量时间稳定
- **Difficulty**：单轮 web search 不应轻易答对
- **Diversity**：覆盖多跳、长尾、时效、freshness、干扰信息、false premise，以及多语言网页环境

这意味着它主要评测的是：**agent 在开放网页中的证据发现与证据组织能力**，而不是长文写作风格或最终报告美观度。

## Part II：方法与洞察

**How：他们引入的关键旋钮不是新模型结构，而是新的“测量对象”和“测量协议”。**

### 1. 数据集构造：先找难锚点，再反向出题
InfoDeepSeek 的数据构造不是从现成 QA 直接筛，而是从网页事实反向设计问题：

- 从权威网页、多语言 Wikipedia、官方站点、论文或新闻中抽取事实
- 先找 **anchor knowledge**：长尾、易混淆、低资源语言中的难事实
- 再把这些锚点与常识事实组合成多跳问题
- 每题至少覆盖两类挑战属性
- 用 GPT-4o 和 DeepSeek-R1 的**单轮 web search**做 difficulty filter  
  - 如果两者都能轻松答对，就丢弃
- 最后由两位 verifier + 一位 decider 做人工核验

最终得到：
- **245** 个高质量问题
- 覆盖 **14** 个领域
- 覆盖 **19** 种 predominant languages

### 2. 评测框架：从“命中文档”改成“证据是否足够回答”
作者给 benchmark 配了一个统一 Agentic RAG scaffold：
- 检索阶段：plan → reflect → act，多步调用搜索引擎/浏览器/时间工具
- 增强阶段：从大量 observation 中挑出 top-n 证据
- 生成阶段：基于证据回答

但重点不在 agent 本身，而在**如何评它**。他们提出四个核心指标：

- **ACC**：最终答案对不对
- **IA@k**：前 k 条证据是否已经足够答题  
  本质是在测证据质量，而不是文档 ID 是否命中 gold set
- **EEU**：模型有没有把搜到的信息真正转化成可用证据
- **IC**：证据是否紧凑，是否过度检索、冗余堆砌

此外，他们还专门为 **false-premise** 问题设计了单独的 LLM 评测 prompt，避免 judge 模型把“没指出错误前提”的答案也误判为正确。

### 核心直觉

过去的 RAG 评测默认有一个前提：**正确证据可以被枚举成固定 gold docs**。  
但在动态 web 中，这个前提失效了。

所以作者做的关键改变是：

- **what changed**：从“看是否检到预定义文档”转成“看 top-k 证据是否足够支持正确回答”
- **which bottleneck changed**：把动态网页中无法稳定定义 gold document 的标注瓶颈，转成可操作的“证据可答性”判断
- **what capability changed**：因此可以单独观察 agent 的检索质量、证据压缩能力、利用率，以及是否被网页噪声干扰

这套设计之所以有效，是因为在开放网页里：
- 同一事实常常有**多个合法来源**
- 真正重要的不是检到哪篇网页，而是**你是否拿到了足够且紧凑的支持信息**

### 策略取舍

| 设计选择 | 得到的能力 | 代价/风险 |
|---|---|---|
| live web 代替静态语料 | 更接近真实部署，能测规划与浏览 | 可复现性下降，结果随时间漂移 |
| 人工构造 + difficulty filtering | 问题更难，能逼出 agentic 行为 | 规模小，标注成本高 |
| IA@k / EEU / IC 代替 NDCG | 适配无 gold docs 的动态环境 | 依赖答案模型与 judge 的稳定性 |
| LLM-as-a-judge + false-premise 专用 prompt | 评测可扩展，特殊题型更稳 | 仍受 prompt 设计和 API 模型偏差影响 |

## Part III：证据与局限

**So what：这个 benchmark 真正证明了，当前 Agentic RAG 的主要短板仍在“找证据”和“用证据”，而不只是最终生成。**

### 关键实验信号

1. **比较信号：当前最强模型仍然很低分**
   - 默认 DuckDuckGo 下，最佳 **Gemini-2.5-Pro 仅 22.45% ACC / 21.63% IA@5**
   - 结论：InfoDeepSeek 不是“换个题库”，而是真的把现有 agentic information seeking 拉到了尚未解决的难度区间

2. **比较信号：搜索引擎质量是主导瓶颈**
   - Gemini-2.5-Flash 的 ACC 从 **14.29%（DuckDuckGo）** 升到 **34.29%（Google）**
   - DeepSeek-V3 从 **8.98%** 升到 **28.57%**
   - 结论：很多性能差异其实先由 retrieval source 决定，再由模型去放大或弥补

3. **分析信号：test-time scaling 确实有效**
   - Gemini-2.5-Flash 随最大检索步数从 1 增加到 20，ACC 从 **7.35%** 提高到 **22.86%**
   - 结论：agent 的优势之一确实是“算力换搜索深度”，但增益不是无限的，难题仍然卡在长尾与噪声上

4. **分析信号：retrieval interference 很普遍**
   - 某些题模型原本靠参数知识能答对，但加了网页检索后反而答错
   - 不同模型/搜索引擎下干扰率常在 **40%–80%** 区间
   - 结论：外部证据并不总是帮助模型；低质量网页可能覆盖掉模型原本正确的内部知识

5. **分析信号：语言感知检索有价值**
   - predominant-language prompt 效果优于单纯中文或英文
   - 结论：真实网页检索不是“统一英语世界”，多语言检索策略本身就是 agent 能力的一部分

### 局限性

- **Fails when**: 相关证据极其稀疏、同时又混有多语言长尾噪声时，当前 agent 往往能“搜到一些东西”但无法稳定提纯成高质量证据；此外，false-premise 之外的复杂语义误判仍可能影响 LLM judge。
- **Assumes**: 依赖人工高成本构造与核验；依赖 live search engine、浏览器工具和多个 LLM API；单题大约需要 **36 次 API 调用**，并消耗约 **24k 输入 token + 4k 输出 token**；还假设标准答案在评测窗口内足够稳定。
- **Not designed for**: 长篇最终报告写作质量评估、多模态网页浏览、离线静态语料检索、系统安全与对齐行为评测。

### 可复用组件
这篇工作最值得迁移的不是具体分数，而是几套评测操作件：

- **difficulty-filtered benchmark construction**：用强模型的单轮 web search 反向过滤掉“假难题”
- **answerability-based evidence metrics**：在无 gold docs 环境下评估证据质量
- **false-premise-aware LLM judging**：为特殊题型设计专门裁判 prompt
- **retrieval interference analysis**：把“检索反伤害”显式量化出来

整体上，这篇 paper 的价值在于把一个常被混在“最终 QA 正确率”里的问题拆开了：  
**Agentic RAG 到底是搜不到，筛不准，还是被噪声带偏？**  
InfoDeepSeek 给了一个比静态 benchmark 更接近真实世界的诊断面板。

![[paperPDFs/Agentic_Reasoning_Benchmarks_Applications_of_Agentic_Reasoning_Autonomous_Research_Agents/arXiv_2025/2025_InfoDeepSeek_Benchmarking_Agentic_Information_Seeking_for_Retrieval_Augmented_Generation.pdf]]