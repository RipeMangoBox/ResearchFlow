---
title: "WebExplorer: Explore and Evolve for Training Long-Horizon Web Agents"
venue: arXiv
year: 2025
tags:
  - Others
  - task/web-navigation
  - task/information-seeking
  - reinforcement-learning
  - query-evolution
  - model-based-exploration
  - dataset/WebExplorer-QA
  - dataset/BrowseComp-en
  - opensource/partial
core_operator: "先让 LLM 围绕种子实体自主搜索与浏览以构建信息空间，再通过去显著线索的长到短查询演化合成高难 QA，并用这些数据做 SFT+GRPO 训练长程 web agent。"
primary_logic: |
  维基种子实体与少量高难示例 → LLM 通过 search/browse 自主探索信息空间并生成初始 QA，再做多轮去显著线索的长到短查询演化 → 用合成 QA 进行 SFT 与 GRPO 训练，得到支持 128K 上下文和 100 轮工具调用的长程网页智能体
claims:
  - "在 Claude-4-Sonnet 上，Evolved QA 相比 Initial QA 将平均工具调用从 7.9 提高到 9.9，同时准确率从 86.6% 降到 67.1%，说明长到短演化显著增加了求解难度 [evidence: comparison]"
  - "WebExplorer-8B (RL) 在 BrowseComp-en/zh 上分别达到 15.7% 和 32.0%，均高于 WebSailor-72B 的 12.0% 和 30.1% [evidence: comparison]"
  - "尽管训练数据主要是知识密集型 QA，WebExplorer-8B 仍在 HLE 上取得 17.3%，超过 WebThinker-32B 的 15.8% 与 ASearcher-Web-QwQ 的 12.5% [evidence: comparison]"
related_work_position:
  extends: "ReAct (Yao et al. 2023)"
  competes_with: "WebSailor (Li et al. 2025a); WebShaper (Tao et al. 2025)"
  complementary_to: "Chain-of-Agents (Li et al. 2025b); DAPO (Yu et al. 2025)"
evidence_strength: strong
pdf_ref: paperPDFs/Agentic_Reasoning_Applications_Autonomous_Web_Exploration_Research_Agents_Foundational_agentic_reasoning/arXiv_2025/2025_WebExplorer_Explore_and_Evolve_for_Training_Long_Horizon_Web_Agents.pdf
category: Others
---

# WebExplorer: Explore and Evolve for Training Long-Horizon Web Agents

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2509.06501), [GitHub](https://github.com/hkust-nlp/WebExplorer)
> - **Summary**: 论文把长程网页智能体的核心瓶颈归因于“高质量高难训练数据稀缺”，于是提出“模型驱动探索 + 长到短查询演化”的自动数据合成流程，再配合 SFT+RL，把 8B 模型训成能做长链网页搜索与推理的 agent。
> - **Key Performance**: BrowseComp-en/zh 分别达到 **15.7% / 32.0%**，8B 超过 WebSailor-72B；WebWalkerQA **62.7%**、FRAMES **75.7%**。

> [!info] **Agent Summary**
> - **task_path**: 复杂信息检索问题 + search/browse 工具环境 -> 多轮网页探索轨迹 -> 最终答案
> - **bottleneck**: 开源 web agent 缺少足够困难且自然的训练题，导致搜索入口过于明显、推理链偏短
> - **mechanism_delta**: 用模型驱动探索替代显式图构建，并用“去显著线索”的长到短多轮演化改变问题分布，再用这些 QA 做长上下文 RL
> - **evidence_signal**: 8B 模型在 BrowseComp-en/zh 上超过 WebSailor-72B，同时演化后数据的平均工具调用从 7.9 提升到 9.9
> - **reusable_ops**: [model-based exploration, long-to-short query evolution]
> - **failure_modes**: [极端模糊且接近 BrowseComp-en 上限的问题仍然困难, 对搜索结果质量与外部 browse 子系统误差敏感]
> - **open_questions**: [减少闭源教师与外部 API 依赖后能否维持数据质量, 更长 context/更多 turns 是否还能持续带来稳定收益]

## Part I：问题与挑战

这篇论文要解决的不是“网页 agent 会不会调用工具”，而是**它能否在没有明显入口的情况下，持续十几轮以上地探索网页并完成信息检索**。

### 真问题是什么？
作者认为，当前开源 web agent 落后的根因不主要在 agent 框架，而在**训练数据分布太简单**：

- 许多已有训练数据虽然也需要多网站信息，但问题里往往带有明显锚点，比如具体年份、专有名词、地点、人物名。
- 这种题目更像“有明确入口的多跳检索”，不是“需要试探、回退、分叉搜索的长程探索”。
- 真正困难的 benchmark（如 BrowseComp-en/zh）恰恰相反：它们**故意不给清晰搜索入口**，使 agent 必须经历多次尝试才逐步定位答案。

### 为什么现在要解决？
因为 benchmark 已经升级了，而训练数据没有同步升级。

- BrowseComp-en 中，超过一半问题连人工标注者花两小时都难以解出。
- 这类高难题如果靠人工构建，成本极高，规模又太小，无法支撑大规模训练。
- 开源模型在这些 benchmark 上明显落后，商业系统又缺少透明训练细节。

### 输入 / 输出接口
论文聚焦的是**文本型网页信息检索**：

- **输入**：一个复杂、线索模糊的信息检索问题
- **环境**：两类工具  
  - `search`：调用 Google 搜索，返回 top-10 结果  
  - `browse`：抓取 URL 内容后，根据指定 query 提取信息
- **输出**：多轮 Thought-Action-Observation 轨迹后的最终答案

### 边界条件
这不是一个“通用网页操作”系统，而是一个**知识密集型、文本网页上的长程检索 agent**：

- 不是 GUI 网页操作、点击按钮、表单填写
- 不是多模态页面理解为主
- 重点是：**在长上下文、长工具链下维持探索式推理**

---

## Part II：方法与洞察

作者的方法分成两层：**先造难题，再用难题训练 agent**。

### 1）模型驱动探索：先构造“信息空间”
传统 graph-based 方法会显式建图：节点是什么、怎么扩展、扩展到哪停止，都要手工设计启发式。

WebExplorer 改成更轻的思路：

- 从一个 Wikipedia seed entity 出发
- 给 LLM 三个 BrowseComp-en 风格示例
- 让模型自己执行 search / browse
- 在探索到的信息空间上，合成一个初始 QA

关键点在于：**不显式维护图，而是让 LLM 在交互过程中“隐式建图”**。  
这样少了规则工程，但保留了跨网站、多实体连接的能力。

### 2）长到短的查询演化：把“能搜到”变成“难搜到”
作者观察到：初始 QA 虽然跨多个网站，但仍太容易。原因不是链路不够长，而是**问题里显著线索太多**。

所以他们没有采用常见的“short-to-long”加信息演化，而是反过来做：

- 删除冗余且显著的描述
- 把日期、地点、名称等具体信息改写得更模糊
- 用替代表述替换显式实体名

这相当于把题目从：

- “你几乎知道该搜什么”

变成：

- “你只知道一个模糊方向，需要边搜边排除”

作者将这一过程迭代 5 次，最终得到约 **40K** 条 WebExplorer-QA。

### 3）SFT + RL：把难题分布迁移到 agent 行为上
有了数据后，训练仍分两阶段：

#### 冷启动 SFT
- 用商业模型收集正确轨迹
- 通过 rejection sampling 只保留正确示范
- 让模型先学会：
  - 如何拆问题
  - 何时 search
  - 何时 browse
  - 如何在 ReAct 格式下组织长轨迹

#### 强化学习 RL
- 用 GRPO，直接在 QA 上训练，不再需要轨迹示范
- reward 由两部分构成：
  - 格式正确
  - 最终答案正确
- 采用**渐进式上下文/turn 扩展**：
  - 64K / 50 turns
  - 96K / 75 turns
  - 128K / 100 turns

这一步的目标不是单纯提高 final answer 准确率，而是让模型学会**更长、更稳定的探索链**。

### 核心直觉

真正有效的因果旋钮，不是“再加一个更强 backbone”，而是**改变训练问题的入口结构**。

- **原来**：问题包含明确锚点，搜索空间窄，第一跳就容易命中
- **现在**：显著线索被移除或模糊化，搜索入口熵更高，模型必须尝试更多分支
- **结果**：训练分布从“短链命中型检索”转向“长链探索型检索”，agent 才会学到持续搜索、回退、再定位的策略

更具体地说：

1. **模型驱动探索**改变了数据生成约束  
   从手工图扩展规则，变成让 LLM 自主发现连接路径，提升灵活性。

2. **长到短演化**改变了信息瓶颈  
   不再给 agent 明确入口，而是逼它在模糊描述下进行探索。

3. **渐进式 RL**改变了行为预算  
   让模型在更长 context 与更多 tool turns 下，真正学会使用更深的搜索链。

### 战略权衡

| 设计选择 | 带来的能力变化 | 代价 / 风险 |
|---|---|---|
| 模型驱动探索替代显式建图 | 数据合成更灵活，少规则工程 | 更依赖教师模型质量，可控性与可解释性较弱 |
| 长到短查询演化 | 问题更自然、更接近真实困难 benchmark | 若过度模糊，可能损害唯一性与可验证性 |
| SFT 后接 GRPO | 从“会调工具”提升到“会长链探索” | 训练成本更高，推理轨迹更长 |
| 渐进式 context/turn 扩展 | 支持 128K / 100 turns 的长程行为 | 对算力、内存、基础设施要求高 |

---

## Part III：证据与局限

### 关键证据信号

#### 信号 1：数据演化确实把题变难了
最直接的证据来自 Initial QA vs Evolved QA：

- 在 Claude-4-Sonnet 上，准确率从 **86.6%** 降到 **67.1%**
- 平均工具调用从 **7.9** 升到 **9.9**

这说明演化不是“表面改写”，而是**真实提高了所需探索深度**。

#### 信号 2：8B 模型在核心 benchmark 上打过更大模型
WebExplorer-8B (RL) 的代表结果：

- **BrowseComp-en: 15.7%**，高于 WebSailor-72B 的 12.0%
- **BrowseComp-zh: 32.0%**，高于 WebSailor-72B 的 30.1%
- **WebWalkerQA: 62.7%**
- **FRAMES: 75.7%**

这表明该方法的收益不是“堆参数”，而是**训练分布更匹配长程信息检索**。

#### 信号 3：RL 真的让搜索链变长
训练过程中：

- 平均工具调用从约 **11** 增长到 **16+**
- 平均轨迹长度增长到 **40K+ tokens**
- BrowseComp-en/zh 的分数同步上升

这个现象支持作者的核心论点：**更长的行为链不是副产物，而是能力形成的一部分**。

#### 信号 4：有一定跨域泛化
虽然训练数据主要是知识密集型 QA，非 STEM 定向，但模型在 **HLE 上得到 17.3%**，超过若干 32B 级方法。  
说明学到的并不只是某个 benchmark 模板，而是某种更一般的“信息搜集 + 推理”能力。

### 结果该怎么理解
这篇论文最有价值的地方，不是再造一个 agent scaffold，而是证明了：

> **如果你把训练问题从“可直接命中”改成“需要探索式定位”，小模型也能学出明显更强的长程 web search 行为。**

换句话说，它把“web agent 能力不足”重新表述为一个**数据分布设计问题**，而不是单纯的模型规模问题。

### 局限性

- **Fails when**: 题目需要真实网页 UI 操作、登录/表单交互、强实时信息，或难度接近 BrowseComp-en 的极端上限时；论文也明确显示其 Evolved QA 与 BrowseComp-en 的 turn 分布仍有差距。
- **Assumes**: 依赖 Google 搜索结果、Jina 内容抽取、Gemini 2.5 Flash 的 browse 分析、DeepSeek-V3 作为自动 judge，以及商业模型用于高质量轨迹收集；同时默认可承受 128K context 与 100 turns 的训练/推理成本。
- **Not designed for**: 多模态网页导航、GUI 级 browser control、交易型网页任务、需要严格安全审计或低延迟部署的生产环境。

### 可复用组件

- **模型驱动探索模板**：从 seed entity 出发，隐式构建信息空间
- **长到短 query evolution 模板**：通过去显著线索制造探索需求
- **渐进式长程 RL 课程**：逐步扩展 context length 与 turn budget
- **难度诊断指标**：用平均工具调用与 turn 分布衡量数据是否真的“更难”

## Local PDF reference

![[paperPDFs/Agentic_Reasoning_Applications_Autonomous_Web_Exploration_Research_Agents_Foundational_agentic_reasoning/arXiv_2025/2025_WebExplorer_Explore_and_Evolve_for_Training_Long_Horizon_Web_Agents.pdf]]