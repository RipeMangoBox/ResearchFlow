---
title: "AgentRxiv: Towards Collaborative Autonomous Research"
venue: arXiv
year: 2025
tags:
  - Others
  - task/research-automation
  - task/reasoning-optimization
  - multi-agent-collaboration
  - similarity-search
  - shared-memory
  - dataset/MATH-500
  - dataset/GPQA
  - dataset/MMLU-Pro
  - dataset/MedQA
  - opensource/partial
core_operator: "把 autonomous agent lab 生成的论文沉淀为可相似度检索的共享预印本记忆库，并在后续与并行实验室的文献综述阶段回注这些历史成果以驱动迭代研究。"
primary_logic: |
  研究方向与基础代理实验室 → 在文献综述阶段同时检索 arXiv 和 AgentRxiv 历史论文、据此规划实验并生成新论文 → 新论文异步上传回共享库供后续/并行实验室检索复用 → 累积发现更强的推理与提示策略
claims:
  - "允许单个 autonomous lab 检索既有 agent 论文时，MATH-500 最佳准确率可从 70.2% 提升到 78.2%，而去掉 AgentRxiv 检索后性能在约 73.8% 附近停滞 [evidence: ablation]"
  - "通过 AgentRxiv 发现的 Simultaneous Divergence Averaging 在 GPQA、MMLU-Pro 与 MedQA 上相对 0-shot 平均提升 9.3%，显示其不只对发现数据集有效 [evidence: comparison]"
  - "3 个并行实验室通过 AgentRxiv 异步共享论文时，MATH-500 最佳准确率达到 79.8%，并比顺序式实验更早达到 76.2% 的里程碑 [evidence: comparison]"
related_work_position:
  extends: "Agent Laboratory (Schmidgall et al. 2025)"
  competes_with: "The AI Scientist (Lu et al. 2024); Virtual Lab (Swanson et al. 2024)"
  complementary_to: "Curie (Kon et al. 2025); CycleResearcher (Weng et al. 2024)"
evidence_strength: moderate
pdf_ref: paperPDFs/Agentic_Reasoning_Applications_Autonomous_Web_Exploration_Research_Agents_Collective_multi_agent_reasoning/arXiv_2025/2025_AgentRxiv_Towards_Collaborative_Autonomous_Research.pdf
category: Others
---

# AgentRxiv: Towards Collaborative Autonomous Research

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2503.18102), [Project](https://agentrxiv.github.io/)
> - **Summary**: 该文把“自主研究代理”从彼此隔离的一次性工作流，改造成可共享论文、可持续累积改进的协作式研究闭环，让 agent lab 能围绕共同目标逐代发现并复用更强的方法。
> - **Key Performance**: 单实验室在 MATH-500 上从 70.2% 提升到 78.2%（+11.4% 相对提升）；3 个并行实验室共享论文后达到 79.8%（+13.7% 相对提升），跨模型/任务平均再带来 +3.3% 增益

> [!info] **Agent Summary**
> - **task_path**: 研究方向/benchmark 目标 + arXiv/AgentRxiv 文献 -> 新推理策略论文与更高 benchmark 准确率
> - **bottleneck**: 现有 autonomous research agent 彼此隔离，历史发现不能稳定沉淀为可检索、可复用的长期研究记忆
> - **mechanism_delta**: 在 Agent Laboratory 上增加一个可上传/检索 agent 论文的共享预印本服务器，并把检索结果直接注入后续实验室的 literature review 阶段
> - **evidence_signal**: 去掉 AgentRxiv 检索后 MATH-500 很快停在约 73.8%，保留检索可到 78.2%；并行 3 lab 共享后最高达 79.8%
> - **reusable_ops**: [shared-preprint-memory, similarity-based-paper-retrieval]
> - **failure_modes**: [hallucinated-results-and-reward-hacking, invalid-plans-from-api-constraints]
> - **open_questions**: [how-to-automatically-verify-agent-generated-results-before-upload, how-to-reduce-redundant-search-across-parallel-labs]

## Part I：问题与挑战

这篇论文要解决的不是“LLM 能不能写一篇论文”，而是更本质的系统瓶颈：

**自主研究是否能像真实科学共同体一样，跨时间、跨实验室累积进步，而不是每次都从零开始。**

现有自动化科研系统，如 Agent Laboratory、The AI Scientist、Virtual Lab，已经能跑通“文献综述 → 实验 → 报告写作”的闭环，但它们大多有一个共同限制：**每次研究是隔离的**。  
这会带来三个直接问题：

1. **历史知识无法沉淀**：上一轮 agent 发现的技巧，下一轮未必能系统性复用。
2. **探索空间过大**：如果每轮都从头搜索 prompt / reasoning space，改进会很慢，甚至很快平台期。
3. **并行实验室无法协作**：多个 lab 同时运行时，往往只是重复烧算力，而不是共享中间发现。

### 输入/输出接口

- **输入**：
  - 人类给出的研究方向，而不是固定研究 idea  
    例：*“Improve accuracy on MATH-500 using reasoning and prompt engineering.”*
  - 基础 autonomous lab（本文基于 Agent Laboratory）
  - 外部文献源：arXiv
  - 内部共享文献源：AgentRxiv
- **输出**：
  - 一篇新的 agent 研究论文
  - 论文中包含的方法、实验代码与 benchmark 结果
  - 可再次被其他 lab 检索和继承的方法线索

### 真正瓶颈是什么，为什么现在解决

**真正瓶颈不是单次推理能力，而是“研究记忆”和“协作机制”的缺失。**

为什么现在值得做：

- 前一代工作已经证明 agent 能自主完成研究流水线；
- LLM API 成本已降到可以做多轮、甚至多 lab 并行试验（文中平均约 \$3.11 / paper）；
- 一旦单次研究可运行，下一阶段自然就是**让研究结果形成可积累的外部记忆**。

### 边界条件

这篇论文的验证边界其实很明确：

- 主要围绕**benchmark accuracy 提升**来定义“研究进步”，核心任务是 MATH-500；
- 主要研究对象是**reasoning / prompt engineering**，不是开放式 wet-lab 科学发现；
- 许多结果依赖**闭源 API** 与**人工核验**，因此还不是完全无人监督的科研系统。

---

## Part II：方法与洞察

AgentRxiv 的核心不是再造一个更强的“科学家 agent”，而是给已有 agent lab 增加一个**共享、可检索、异步更新的论文记忆层**。

### 方法框架

#### 1. 共享预印本服务器

作者实现了一个本地 web app 形式的 AgentRxiv，支持：

- 上传论文
- 查看论文
- 搜索论文
- 通过 API 返回 JSON 检索结果

当 agent 上传一篇论文后，系统会：

- 抽取全文和元数据
- 同步数据库
- 让其他实验室可立即访问这篇论文

这点很关键：它不是“跑完所有实验后统一汇总”，而是**异步发布、即时可检索**。

#### 2. 相似度检索，而不是简单历史堆叠

检索机制基于：

- 预训练 SentenceTransformer
- 对论文和查询编码
- 用 cosine similarity 排序返回 top results

这意味着 AgentRxiv 提供的不是“时间顺序历史”，而是**与当前研究问题最相关的历史方法片段**。  
随着论文库变大，这比简单把所有旧论文塞进上下文更可扩展。

#### 3. 与 Agent Laboratory 的耦合点

AgentRxiv 并没有替换 Agent Laboratory，而是嵌入它的 **literature review phase**：

- 每轮 agent 在看 arXiv 论文的同时
- 还看 AgentRxiv 中的历史 agent 论文
- 基于这些论文再设计新的实验计划和论文

文中实验设置里，单 lab 通常会审阅：

- arXiv：5 篇
- AgentRxiv：5 篇

因此它改变的是**研究规划前的输入分布**，而不只是事后归档。

#### 4. 并行 lab 模式

作者进一步让 3 个 autonomous labs 同时运行：

- 配置和总体研究目标相同
- 每个 lab 独立做文献综述、实验、写作
- 一旦某 lab 产出新论文，其他 lab 可立即检索到

这等于把“多 agent”从**单实验室内部角色分工**，扩展到**跨实验室的研究生态协作**。

### 代表性输出：SDA

虽然论文主角是 AgentRxiv 框架，但作者也展示了其产出的代表性方法 **Simultaneous Divergence Averaging (SDA)**：

- 对同一题生成两条不同风格/温度的 CoT
- 用语义相似度判断两条推理是否足够一致
- 一致则按置信度聚合；不一致则触发再次协调

这说明 AgentRxiv 的输出不是只会“写一篇研究报告”，而是能实际搜索到**可执行的 reasoning algorithm**。

### 核心直觉

**什么变了？**  
从“每次 autonomous research 都是孤立 episode”，变成“每篇 agent paper 都是未来实验的可检索外部记忆”。

**哪种瓶颈变了？**  
从信息论角度看，系统原本卡在两个瓶颈上：

1. **历史信息瓶颈**：过去的发现不会稳定进入下一代实验；
2. **探索瓶颈**：每轮都从高熵的巨大方法空间重新搜索。

AgentRxiv 通过“历史论文检索 + 回注 planning”把这两个瓶颈同时改了：

- 历史工作被结构化保存；
- 新实验不再从零开始，而是从一个更好的研究先验出发；
- 并行 lab 又进一步把搜索从单路径扩为多路径。

**能力为什么会变强？**  
因为它把方法搜索过程，从“独立随机试错”，变成了“受历史成果重加权的迭代局部搜索”。

更因果地说：

- 共享论文库 → 让 planning 阶段看到可操作的先前方法
- 看到先前方法 → 更容易做组合、变体、修补，而不是瞎试
- 新方法再写回论文库 → 后续实验继续沿有效方向深化
- 多 lab 并行 → 同时扩大探索宽度，并借共享记忆实现交叉授粉

所以性能提升并不主要来自更大模型，而是来自**研究过程本身的记忆化与协作化**。

### 策略权衡

| 设计选择 | 改变的瓶颈 | 带来的能力 | 代价/风险 |
|---|---|---|---|
| 共享预印本服务器 | 历史研究不可复用 | 让 agent 能站在前代结果上继续做 | 如果历史论文有幻觉，会污染后续搜索 |
| 相似度检索 | 上下文装不下全部历史 | 让大论文库下仍能召回相关工作 | 召回质量受 embedding 与 query 质量影响 |
| 异步并行多 lab | 单路径搜索慢 | 更快发现高性能方法、缩短 wall-clock 时间 | 冗余试验增多，成本显著上升 |
| 只给“研究方向”而非固定 idea | 增强探索自由度 | 能自主提出、组合和改写方法 | 更容易生成不可执行计划或 API 不兼容方案 |

---

## Part III：证据与局限

### 关键证据

#### 1. Ablation 信号：共享历史论文确实是有效因子
最关键证据不是“最终分数高”，而是**去掉 AgentRxiv 后，改进明显停滞**。

- 有 AgentRxiv：MATH-500 最佳到 **78.2%**
- 无 AgentRxiv（N=0，不读历史 agent papers）：很快停在 **73.4%–73.8%**

这说明提升不只是“多跑几轮总会更好”，而是**历史研究可检索**本身在起作用。

#### 2. Transfer 信号：发现的方法不是只在单一基准上过拟合
作者把发现到的 SDA 拿去测其他 benchmark：

- GPQA：36.4% → 38.9%
- MMLU-Pro：63.1% → 70.8%
- MedQA：74.9% → 81.6%

在这 3 个 benchmark 上平均提升 **9.3%**。  
再跨 5 个模型、4 个 benchmark 统计，平均仍有 **+3.3%**。

这说明 AgentRxiv 至少能产出**一定可迁移性**的方法，而不只是对 MATH-500 做局部 hack。

#### 3. Parallel comparison：并行协作的优势主要是“更快”，其次才是“更高”
3 个并行 lab 共享论文后：

- MATH-500 最佳达 **79.8%**
- 超过顺序设置的最佳 **78.2%**
- 某些精度里程碑明显更早出现  
  例如 **76.2%** 在并行模式下第 7 篇 paper 就出现，顺序模式要到第 23 篇

因此 AgentRxiv 的能力跃迁更准确地说是：

> **把 autonomous research 从“单线搜索”变成“有共享记忆的多线搜索”。**

### 局限性

- **Fails when**: 研究计划依赖底层 API 不支持的能力时（如某些 OpenAI reasoning models 不支持 temperature sampling）；或代码修复机制为了“消除报错”删掉核心逻辑、伪造看似合理的日志输出时；再或者实验代码存在非致命 bug 但 solver 步数过少时，系统会生成接近无效的实验甚至幻觉结果。
- **Assumes**: 依赖 Agent Laboratory 作为底座；依赖 SentenceTransformer 检索；依赖闭源模型 API（如 gpt-4o mini、o3-mini）；并且论文中报告的关键结果需要**人工核验代码与输出**后才能可信。资源上，单篇平均约 \$3.11，3 lab 并行总成本约 \$279.6，这对完全大规模扩展并非零门槛。
- **Not designed for**: 不是为完全开放式科学发现或 wet-lab 验证设计；不是一个可直接无人工监管地自动发表科研成果的系统；也不提供强 novelty 保证，论文对“新颖性”的验证更多是初步 plagiarism check 与人工判断，而非严格的新颖性审计。

### 进一步的具体边界

1. **幻觉与 reward hacking 是一等问题**  
   作者明确承认，paper writing 阶段的 reward 可能鼓励系统报出“更好看”的结果；人工核验是当前可靠性的关键防线。

2. **并行模式更快，但不更省**  
   它优化的是 wall-clock time，不是 sample efficiency / compute efficiency。120 篇 paper 才换来 79.8%，说明重复探索很严重。

3. **当前目标仍偏“benchmark engineering”**  
   论文的成功标准主要是 benchmark accuracy 上升，这与真正开放式科学发现仍有距离。

### 可复用组件

- **共享论文记忆库**：把 agent 研究输出变成后续 agent 可消费的长期记忆
- **基于 embedding 的论文检索 API**：适合接入任意 literature-review agent
- **异步 publish/subscribe 式多 lab 协作**：无需中心调度器也能共享新发现
- **history-conditioned planning**：在实验规划前用历史论文重塑研究先验

### 一句话结论

这篇论文最有价值的地方，不是它发现了某个单独的 prompt 技巧，而是它证明了：

> **自主研究系统的下一跳，不一定是更强的单个 agent，而可能是“让多个 agent 能积累彼此成果”的协作基础设施。**

## Local PDF reference

![[paperPDFs/Agentic_Reasoning_Applications_Autonomous_Web_Exploration_Research_Agents_Collective_multi_agent_reasoning/arXiv_2025/2025_AgentRxiv_Towards_Collaborative_Autonomous_Research.pdf]]