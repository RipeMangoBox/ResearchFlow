---
title: "Mind2Web 2: Evaluating Agentic Search with Agent-as-a-Judge"
venue: arXiv
year: 2025
tags:
  - Survey_Benchmark
  - task/agentic-search-evaluation
  - task/web-agent-evaluation
  - agent-as-a-judge
  - tree-structured-rubric
  - citation-verification
  - dataset/Mind2Web-2
  - opensource/full
core_operator: 基于任务定制的树状rubric构建judge agent，逐节点核验答案正确性与引用归因
primary_logic: |
  长程真实检索任务需求 → 构建130个多网站、可时间变化的任务与任务级rubric → 由Extractor/Verifier组成的judge agent执行正确性与归因核验并聚合部分得分 → 揭示agentic search在完备性、实时浏览与可信引用上的能力边界
claims:
  - "Mind2Web 2包含130个真实长程agentic search任务，rubric平均50个节点、最多603个节点，覆盖时间变化与多来源带引用回答场景 [evidence: analysis]"
  - "其Agent-as-a-Judge在15个随机任务、720个叶节点的人审复核中达到99.03%的节点级正确率，显示出高可靠自动评测能力 [evidence: analysis]"
  - "在该基准上，OpenAI Deep Research取得0.54 Partial Completion和0.28 Success Rate，约为人类0.79/0.54表现的50-70%，且平均耗时8.4分钟低于人类18.4分钟 [evidence: comparison]"
related_work_position:
  extends: "Mind2Web (Deng et al. 2023)"
  competes_with: "BrowseComp (Wei et al. 2025); AssistantBench (Yoran et al. 2024)"
  complementary_to: "WebArena (Zhou et al. 2024); Online-Mind2Web (Xue et al. 2025)"
evidence_strength: moderate
pdf_ref: paperPDFs/Agentic_Reasoning_Benchmarks_Applications_of_Agentic_Reasoning_Web_Agents/arXiv_2025/2025_Mind2Web_2_Evaluating_Agentic_Search_with_Agent_as_a_Judge.pdf
category: Survey_Benchmark
---

# Mind2Web 2: Evaluating Agentic Search with Agent-as-a-Judge

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2506.21506), [Project](https://osu-nlp-group.github.io/Mind2Web-2/)
> - **Summary**: 这篇论文的核心不是再造一个搜索代理，而是提出一个能自动评估“长程、实时、带引用开放式搜索答案”的基准与 judge agent 框架，让 agentic search 的真实能力第一次能被较可靠地量化。
> - **Key Performance**: Judge agent 在人审复核中达到 **99.03%** 节点级正确率；最佳系统 OpenAI Deep Research 达到 **0.54 Partial Completion / 0.28 Success Rate**，对比人类 **0.79 / 0.54**。

> [!info] **Agent Summary**
> - **task_path**: 真实长程检索任务 + 系统生成的带URL答案 → task-specific judge agent → 正确性/归因评分
> - **bottleneck**: 时间变化、多来源、长答案使固定答案匹配和少量LLM判分都不再可靠
> - **mechanism_delta**: 把“整体判一个复杂答案”改成“按任务定制树状rubric逐叶节点二值核验，再按critical/non-critical/sequential逻辑聚合”
> - **evidence_signal**: 15个任务、720个叶节点的人审复核显示judge agent节点级正确率99.03%
> - **reusable_ops**: [树状rubric分解, URL级引用核验]
> - **failure_modes**: [折叠或被拦截网页导致证据不可见, 系统输出伪造或失效URL并在长答案综合时出错]
> - **open_questions**: [如何评估跨多网页联合支持的陈述, 如何长期维护时间变化网站下的基准稳定性]

## Part I：问题与挑战

这篇论文真正瞄准的瓶颈，不是“agent 会不会搜”，而是**我们能不能可靠地评估它到底搜对了没有**。

### 1. 旧评测为什么不够用了
agentic search / Deep Research 系统的输出已经不是一个短答案字符串，而是：

- 要跨多个网站持续搜索
- 需要实时网页信息
- 最终输出一段长篇、综合、带引用的回答
- 回答本身还会随时间变化

这直接让很多现有评测范式失效：

- **短程 web-agent benchmark**：通常只看单站点、少量动作，覆盖不到长程信息整合。
- **固定答案匹配**：适合“唯一字符串答案”，不适合购物清单、研究综述、旅行方案这类开放式结果。
- **普通 LLM-as-a-Judge**：对几百到几千词、还带多条 URL 引用的答案，鲁棒性和可解释性都不够。

### 2. 真正的瓶颈是什么
**真瓶颈是评测对象的“高熵”**：

- 不同系统会走不同检索路径
- 同一任务在不同时间答案可能变化
- 回答不是单一事实，而是多子条件组合
- 还要验证“答案对不对”与“引用是不是支持答案”

所以，难点不是生成一个唯一 ground truth，而是把**复杂开放式答案转成可验证的局部判定**。

### 3. 为什么现在必须解决
因为 Deep Research 类系统已经进入实用阶段。如果没有稳定自动评测：

- 模型迭代缺少反馈回路
- 用户无法判断系统是否可信
- “看起来很像真的”幻觉答案会被误当成可靠搜索结果

换句话说，这项工作回答的是：**agentic search 要从 demo 走向可信产品，评测基础设施必须先到位。**

### 4. 输入/输出接口与边界
**输入**：现实中的长程信息搜集任务。  
**输出**：带 URL 引用的综合答案。  
**评测输出**：部分完成度（Partial Completion）与完整成功率（Success Rate）。

论文明确设置了边界条件，保证评测可落地：

- 聚焦**客观、可验证**任务
- 鼓励但不要求所有任务都时间变化
- 排除主观型、视频理解、非英文站点
- 排除变化过快、无法稳定评测的任务
- 默认关键陈述可追溯到**单个网页 URL**

---

## Part II：方法与洞察

### 方法骨架
这篇论文做了两件事，而且两件事是绑定的：

1. **做 benchmark**：构建 130 个真实、长程、多网站、可时间变化的 agentic search 任务  
2. **做 evaluation engine**：为每个任务配一个 task-specific judge agent，自动检查答案正确性与引用归因

如果只有前者，没有后者，这个 benchmark 很难大规模用；  
如果只有后者，没有高质量任务，评测又会偏离真实用户需求。

### 关键设计 1：任务收集不是“凑题”，而是重人工验证
Mind2Web 2 的任务通过三阶段构建：

- **Proposal**：提出真实需求型任务
- **Refinement**：专家修订，压实清晰性、客观性、可验证性
- **Validation**：至少两位验证者独立完成并确认可行

作者强调总共用了 **1000+ 小时人工**。这件事很重要，因为这说明 benchmark 的价值不只是题目数量，而是**任务质量与可评测性被认真打磨过**。

### 关键设计 2：树状 rubric，而不是一句话打分
每个任务被拆成一个**树状 rubric**：

- **叶节点**：最细粒度、可二值判断的检查项
- **内部节点**：向上聚合子节点得分
- **critical 节点**：某个关键条件错了，父节点直接失败
- **non-critical 节点**：允许部分得分
- **sequential 节点**：前一步失败，后面无需再判

这相当于把“复杂答案是否合格”拆成一串低熵判断，例如：

- 预算是否符合
- 商品是否来自指定网站
- 颜色是否正确
- 价格是否准确
- 该陈述是否被引用网页支持

最终，系统不再是和一个静态标准答案比字符串，而是和**任务要求的结构化验证逻辑**对齐。

### 关键设计 3：judge agent 评的是“正确性 + 归因”
每个 judge agent 会接收：

- 模型答案文本
- 答案里的 URL 引用
- 预缓存的网页文本和截图

然后通过两类工具工作：

- **Extractor**：从答案里抽取结构化字段
- **Verifier**：核验陈述是否成立，以及是否被 URL 支持

因此它评测的不只是“答得像不像”，而是：

1. **Correctness**：任务要求是否满足  
2. **Attribution**：每个关键陈述是否真能在引用页面里找到支撑

这让 benchmark 能直接测出很多 agentic search 的核心失败模式：  
不是“没说”，而是“说了但没证据”或“引用了但引用不支持”。

### 核心直觉

**改变了什么**：  
从“让一个通用 judge 对整篇长答案做整体印象评分”，改成“让一个任务定制的 judge agent 按树状 rubric 分解验证”。

**改变了哪种瓶颈**：  
把评测从高熵生成匹配问题，变成低熵验证问题。  
也就是利用作者说的 **generation-verification asymmetry**：

- 生成答案可以千变万化
- 但任务要求其实是已知的、可枚举的、可核验的

**能力因此怎么变了**：  
评测终于能覆盖：

- 长程任务
- 时间变化答案
- 多来源综合
- 带引用的开放式输出
- 部分完成度，而不只是全对/全错

**为什么有效**：  
因为复杂任务最难的是“整体比对”，最容易的是“局部核验”。  
作者把整体难题拆成局部可判定问题，再用明确的聚合逻辑恢复全局评分。

### 战略取舍

| 设计选择 | 解决的瓶颈 | 带来的能力 | 代价/风险 |
|---|---|---|---|
| 任务定制 judge agent，而非统一 answer match | 开放式、时间变化答案无法静态匹配 | 能评真实搜索任务 | 每题都要写/生成脚本，开发成本高 |
| 树状 rubric，而非单次 LLM judge | 长答案、多条件任务难整体判断 | 可解释、可局部定位错误、支持部分得分 | rubric 设计复杂，平均 50 节点、最多 603 节点 |
| 同时评正确性与引用归因 | 只看答案文本会放过“有理有据的幻觉” | 能测可信性与可核查性 | 依赖网页抓取、缓存与 URL 可访问性 |
| 预缓存网页文本+截图，而非完全在线评测 | 在线页面变化和抓取波动会污染评测 | 评测更稳定可复现 | 与网页实时状态存在时间差，需要维护 |

### 一个可迁移的抽象
这篇论文最有复用价值的，不只是 Mind2Web 2 本身，而是一个评测抽象：

> **真实开放任务 → 任务级结构化 rubric → 叶节点可验证工具调用 → 向上聚合全局分数**

这个抽象未来可以迁移到：

- research agent
- browser agent
- citation-heavy QA
- report generation with provenance
- 复杂 enterprise workflow 的自动验收

---

## Part III：证据与局限

### 关键证据信号

#### 1. Judge agent 不是“看起来可用”，而是被人审验证过
最强证据信号是评测器本身的可靠性验证：

- 在 **15 个随机任务**
- **720 个叶节点判断**
- 经过人工复核后
- 实际 Verifier 错误只有 **7/720**
- 节点级正确率达到 **99.03%**

这说明论文最关键的一步——“自动评复杂答案”——不是拍脑袋成立，而是有直接验证的。

#### 2. 基准确实能拉开系统差距
主实验中：

- **OpenAI Deep Research**：0.54 Partial Completion / 0.28 Success Rate
- **Human**：0.79 / 0.54
- 搜索增强型 LLM（如 ChatGPT Search / Perplexity Pro Search）显著更弱
- **Operator** 虽然会浏览网页，但整体仍落后于 Deep Research 系统

这里最重要的结论不是谁第一，而是：  
**长程 agentic search 的优势来自“搜索+综合+工具+长时程保持任务目标”的组合，而不只是会点网页。**

#### 3. 时间变化任务是当前系统的真实短板
作者单独抽出 57 个显式时间变化任务后发现：

- 大多数系统在这类任务上显著更差
- 人类和 Operator 相对更稳

这说明一个很关键的能力边界：

> **如果系统不能稳定访问实时网页、使用交互式浏览，agentic search 在很多现实任务上会直接掉档。**

#### 4. 错误分析揭示了“为什么还不行”
作者把错误分成几类：

- 信息没找到 / 部分缺失
- 明确违反任务条件
- 伪造或失效 URL
- 缺失引用
- 引用不支持答案
  - synthesis error：网页对，但综合错了
  - retrieval error：网页就找错了

这比只看总分更有价值，因为它指出当前系统的问题并不只是“检索不够”，还包括：

- 长程执行中提前放弃
- 引用习惯不稳定
- 长上下文下答案落地错误
- 从网页到最终报告的综合步骤失真

论文还提到：即便是最佳系统，幻觉相关错误仍显著存在。

#### 5. 更多测试时间确实有帮助
同一家族系统中，更长推理/执行时间通常带来更高完成度；Pass@3 也高于单次成功率。  
这说明当前 agentic search 仍处在一种**test-time scaling 有效**的阶段：算力、时间、试错次数仍能换来更高成功概率。

### 1-2 个最关键指标
如果只记两个数字，我会记：

1. **Judge agent 99.03% 节点级正确率**  
   这是这篇 benchmark 能站住的根基。

2. **最佳系统 0.54 vs 人类 0.79 的 Partial Completion**  
   这说明 agentic search 已经不弱，但离“可靠替代人做复杂搜索”还有明显距离。

### 局限性

- **Fails when**: 证据被网页折叠内容、反爬机制或动态加载隐藏时，自动归因核验会失灵；需要跨多个网页联合才能支持的陈述，也不适合当前单页归因假设。
- **Assumes**: 关键陈述可映射到单个URL；被引用URL本身可信；o4-mini 级别的抽取/核验模型足够稳定；网页缓存能够代表回答生成时的页面状态。
- **Not designed for**: 主观偏好类查询、非英文站点、视频理解任务、变化极快的实时值（如短周期汇率）、以及需要复杂外部推理才能定义标准答案的场景。

### 复现与扩展时要注意的资源/依赖
这套方案不是“零成本自动评测”，它依赖：

- 大量人工任务与 rubric 开发（1000+ 小时）
- LLM-based Extractor / Verifier
- Playwright 式网页预缓存
- 少量被拦截网页的人工干预
- 私有测试集维护，避免奖励模型化和过拟合

也就是说，它的强点是**真实、严谨、可诊断**，代价是**建设和维护成本高**。

### 可复用组件
这篇论文最值得复用的部件有：

- **树状 rubric 设计**：把复杂任务验收拆成局部条件
- **Extractor / Verifier 双工具范式**：答案抽取 + 证据核验
- **critical / non-critical / sequential 聚合逻辑**
- **网页缓存 + 截图核验流程**
- **公有 dev / 私有 test 切分**：降低 benchmark 被刷榜式过拟合的风险

**一句话结论**：  
Mind2Web 2 的真正贡献，是把“真实世界长程搜索代理的评测”从几乎不可自动化，推进到一个可规模化、可诊断、且足够可靠的阶段；而实验结果也明确告诉我们，当前最强系统已经接近人类的一部分效率优势，但在实时浏览、完备性和可信引用上仍有明显缺口。

![[paperPDFs/Agentic_Reasoning_Benchmarks_Applications_of_Agentic_Reasoning_Web_Agents/arXiv_2025/2025_Mind2Web_2_Evaluating_Agent_as_a_Judge.pdf]]