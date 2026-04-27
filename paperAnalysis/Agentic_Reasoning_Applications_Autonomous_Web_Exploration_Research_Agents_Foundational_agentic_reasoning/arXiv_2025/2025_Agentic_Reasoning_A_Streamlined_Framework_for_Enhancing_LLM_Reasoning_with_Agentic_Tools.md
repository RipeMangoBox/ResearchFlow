---
title: "Agentic Reasoning: A Streamlined Framework for Enhancing LLM Reasoning with Agentic Tools"
venue: arXiv
year: 2025
tags:
  - Others
  - task/knowledge-intensive-reasoning
  - task/deep-research
  - tool-use
  - knowledge-graph-memory
  - reranking
  - dataset/GPQA
  - dataset/GAIA
  - dataset/FreshWiki
  - opensource/partial
core_operator: 让主推理LLM在思考过程中按需调用网页搜索、代码执行与Mind-Map知识图谱记忆代理，形成检索-计算-记忆闭环。
primary_logic: |
  专家级开放问题/研究主题 → 主推理LLM在长链思考中插入搜索/代码/Mind-Map调用并携带上下文 → 外部代理返回检索证据、计算结果与结构化记忆 → 主模型整合多轮工具结果生成答案或研究文章
claims:
  - "在 Humanity's Last Exam 上，Agentic Reasoning + DeepSeek-R1 达到 23.8% 准确率，较基座 DeepSeek-R1 的 9.4% 提升 14.4 个百分点，并接近 OpenAI Deep Research 的 26.6% [evidence: comparison]"
  - "在 GPQA 上，Agentic Reasoning + DeepSeek-R1 达到 81.2%，高于 Search-O1 + DeepSeek-R1 的 74.6% 和 o3-mini-high 的 79.7% [evidence: comparison]"
  - "在 GAIA 的记忆策略消融中，Mind-Map 平均 66.13%，高于 MemGPT 的 65.12% 与无记忆的 46.18%，说明结构化记忆对长链工具推理有直接增益 [evidence: ablation]"
related_work_position:
  extends: "Search-O1 (Li et al. 2025)"
  competes_with: "Search-O1 (Li et al. 2025); STORM (Shao et al. 2024)"
  complementary_to: "GraphRAG (Edge et al. 2024); MemGPT (Packer et al. 2023)"
evidence_strength: strong
pdf_ref: paperPDFs/Agentic_Reasoning_Applications_Autonomous_Web_Exploration_Research_Agents_Foundational_agentic_reasoning/arXiv_2025/2025_Agentic_Reasoning_A_Streamlined_Framework_for_Enhancing_LLM_Reasoning_with_Agentic_Tools.pdf
category: Others
---

# Agentic Reasoning: A Streamlined Framework for Enhancing LLM Reasoning with Agentic Tools

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2502.04644), [Code](https://github.com/theworldofagents/Agentic-Reasoning)
> - **Summary**: 这篇工作把网页搜索、代码执行和知识图谱式记忆封装成可被主推理模型按需调用的代理工具，用外部证据、可执行计算和结构化记忆来补齐开放域长链推理中的短板。
> - **Key Performance**: Humanity's Last Exam 23.8%（相对 DeepSeek-R1 的 9.4% 提升 14.4 个百分点）；GPQA 81.2%（高于 Search-O1 的 74.6%）

> [!info] **Agent Summary**
> - **task_path**: 专家级开放问题/研究主题 -> 多轮工具增强长链推理 -> 最终答案/研究文章
> - **bottleneck**: 单个LLM在开放域任务中同时承担检索、计算、记忆，容易出现证据不足、上下文遗忘和错误累积
> - **mechanism_delta**: 把检索/计算/记忆拆成外部代理，并用 Mind-Map 把推理轨迹压成可查询知识图，主模型主要负责调度与整合
> - **evidence_signal**: 多基准比较 + 工具/搜索/记忆消融，显示三工具组合与 Mind-Map 均带来稳定增益
> - **reusable_ops**: [查询分解+重排循环, 知识图谱式结构化记忆]
> - **failure_modes**: [检索源可信度未验证, 顺序工具调用导致高时延与误差累积]
> - **open_questions**: [工具调用策略能否端到端学习, 高风险领域如何做来源验证与置信控制]

## Part I：问题与挑战

这篇论文要解决的不是“LLM 会不会推理”，而是**LLM 在开放域、知识密集、需要长链外部交互的任务里，怎么持续正确地推理**。

### 1. 真问题是什么
现有推理模型（如 o1、DeepSeek-R1）在数学和代码上强，是因为：
- 问题目标清晰；
- 结果容易验证；
- 中间步骤可被形式化约束。

但当任务变成：
- 专家级开放问题回答，
- 医疗/金融/法律等需要查证的研究任务，
- 需要多轮检索、比较、计算、组织证据的 deep research，

单靠模型内部参数和长 CoT 往往不够。核心瓶颈有三个：

1. **证据瓶颈**：模型需要最新或长尾知识，闭卷推理不可靠。  
2. **记忆瓶颈**：长推理链夹杂多次工具调用时，模型容易忘掉前文关系、偏离问题或重复犯错。  
3. **执行瓶颈**：数值计算、代码验证这类步骤若让主模型自己兼做，会打断主推理流并放大错误。

### 2. 输入/输出接口
- **输入**：专家级问题，或需要调研的主题/任务。
- **输出**：最终答案，或结构化研究文章。
- **边界条件**：该方法最适合“需要外部知识 + 多步推理 + 工具协同”的任务；如果任务本身是纯闭卷、短链、低时延问答，代理开销可能不划算。

### 3. 为什么现在值得做
因为现在的基础模型已经具备较强的**长链规划能力**，差的不是“能不能想”，而是“想的过程中能不能像人一样查资料、算一算、做笔记”。这使得“agentic tools + reasoning model”从概念演示，开始变成真正能带来能力跃迁的组合。

---

## Part II：方法与洞察

作者提出的 **Agentic Reasoning** 可以理解为一个“主脑 + 三个外挂”的框架：

- **主推理模型**：负责整体思考、判断何时调用工具；
- **Web-Search agent**：负责检索外部信息；
- **Coding agent**：负责代码生成与执行；
- **Mind-Map agent**：负责把推理上下文变成可查询的结构化记忆。

### 方法主线

#### 1. 主模型的工具调用方式
主模型在推理过程中显式输出特殊 token：
- web-search token
- coding token
- mind-map token

一旦检测到这些 token，当前推理暂停，系统抽取：
- 工具查询内容；
- 当前推理上下文；

再把它们发给对应代理。代理返回结果后，结果重新并入主推理链，继续下一步思考。

这个设计的关键不是“能调工具”，而是**把工具调用嵌入 reasoning loop，而不是放在推理前后的固定流水线里**。

#### 2. Mind-Map：结构化记忆
Mind-Map 是这篇论文最有辨识度的模块。

它把原始 reasoning chain 转成知识图：
- 抽取实体；
- 建立实体间语义关系；
- 做 community clustering；
- 对每个簇生成摘要；
- 支持基于图的检索。

它有两个作用：
1. **给其他工具提供上下文**：搜索和代码代理不是只看当前一句 query，而是能看到压缩后的推理背景。
2. **给主模型提供外部记忆**：当主模型在长链推理中“丢线”时，可以回查图中的关系和摘要。

所以它不是普通 conversation memory，而是把“线性 token 记忆”改成“关系型显式记忆”。

#### 3. Web-Search agent：检索不是一次搜完
作者对搜索代理做了专门设计，流程是：

1. **Query Breakdown**：把原始模糊查询改写成多个适合搜索引擎的细粒度查询；
2. **Search**：调用 Bing 检索网页；
3. **Rerank**：用重排模型按“原问题 + 上下文”重新排序；
4. **Iterative Refinement**：若 top-10 平均相关度不够，再回去继续改写查询；
5. **RAG + Synthesis**：对高相关网页做信息提取，再综合成自然语言 snippet 返回主模型。

这里的关键改动不是简单“加搜索”，而是**把搜索从一次性召回，升级成上下文驱动的迭代式证据构建过程**。

#### 4. Coding agent：把计算从主链中剥离
作者发现，与其让主推理模型自己写代码，不如交给更擅长 coding 的专门模型：
- 接收 query + reasoning context + user query；
- 生成代码；
- 执行；
- 把结果转回自然语言。

这样做减少了主推理链被代码细节打断的问题，也能让最合适的模型承担最合适的子任务。

### 核心直觉

**这篇论文真正改变的，不是“多了几个工具”，而是把开放域推理里的三种信息瓶颈拆开治理：**

- **知识不足** → 用上下文感知的搜索代理补证据；
- **记忆易丢** → 用知识图外存保存关系与阶段结论；
- **执行干扰** → 用专门 coding agent 承担计算与验证。

因此，能力变化链条可以写成：

**单模型隐式长链推理**  
→ 改成 **主模型调度 + 检索/计算/记忆代理分工**  
→ 放松了证据、上下文窗口和执行中断这三个约束  
→ 获得了更稳定的 expert QA 与 deep research 能力。

更因果地说：

1. **Query breakdown + rerank** 改变了检索结果分布  
   - 让返回证据更贴近当前子问题；
   - 降低“搜得到但不相关”或“搜得泛泛”的情况。

2. **Mind-Map** 改变了推理时的信息保留方式  
   - 从短期、线性、易漂移的 token 记忆，
   - 变成结构化、可检索、可聚类的关系记忆，
   - 所以长链任务更不容易偏航。

3. **专门 coding agent** 改变了子任务执行负担  
   - 主模型不再同时承担高层规划和低层代码实现，
   - 因而更能保持推理主线完整。

### 战略权衡

| 设计选择 | 解决的瓶颈 | 能力收益 | 代价/风险 |
|---|---|---|---|
| Web-Search + 查询分解 + 重排 | 外部知识缺失、一次检索噪声大 | 更强事实覆盖与长尾知识获取 | 依赖搜索引擎与重排服务，来源可信度未内建验证 |
| Coding agent | 计算/执行打断主推理流 | 数值与程序验证更稳定 | 需要执行环境；错误代码仍会污染后续推理 |
| Mind-Map 知识图记忆 | 长链推理中的遗忘与关系混乱 | 保持多轮工具调用后的上下文一致性 | 图构建本身有成本，也可能引入错误实体/关系 |
| Agentic dispatch | 单模型负担过重 | 支持“研究型”多步任务 | 时延上升，系统实现更复杂 |

---

## Part III：证据与局限

### 关键证据信号

#### 1. 比较信号：开放域专家题确实出现明显跃迁
- **Humanity's Last Exam**：从 DeepSeek-R1 的 **9.4%** 提升到 **23.8%**。  
  这说明收益不是小幅提示工程，而是把模型从“几乎不会做深研究题”拉到“接近闭源 deep research 系统”的级别。
- **GPQA**：达到 **81.2%**，高于 Search-O1 + DeepSeek-R1 的 **74.6%**。  
  说明提升不只是“有搜索就行”，而是搜索 + 代码 + 结构化记忆的组合起作用。

#### 2. 比较信号：在 agent benchmark 与 deep research 上也成立
- **GAIA**：66.13，达到公开方法 SOTA，并与 OpenAI Deep Research 的 67.36 接近。
- **FreshWiki / 人评 deep research**：生成文章在 ROUGE、Entity Recall 以及人工评分上都超过 RAG、RAgent、Search-O1、STORM，且优于 Gemini Deep Research。

这说明框架不只适用于选择题式 QA，也能迁移到长文调研生成。

#### 3. 消融信号：不是工具越多越好，而是工具选得对
作者专门比较了：
- Hugging Face 默认 7 工具；
- LangChain 109 工具；
- 自己选的 3 个核心工具。

结论是：**工具质量比工具数量更重要**。工具一多，错误调用和冗余调用反而增加；而 web search + coding + Mind-Map 的三件套效果最好。

#### 4. 机制信号：Mind-Map 不是装饰件
- 在 GAIA 记忆策略对比中，Mind-Map 优于无记忆、Raw memory、Read-agent、MemoryBank、MemGPT。
- 在需要更长 reasoning chain、更多 tool calls 的问题上，Mind-Map 增益更明显。
- 作者还给出逻辑题和 Werewolf 游戏案例，显示其对**关系追踪与策略推理**有帮助。

#### 5. 机制信号：agentic tools 优于直接 API 调用
附录里，直接 API 调用版本在 GAIA 平均仅 **47.18**，而 agentic tools 版本达到 **66.13**。  
这说明论文真正有效的点不只是“接了搜索和代码 API”，而是**把它们包装成能返回不确定性、能读上下文、能被主推理模型调度的代理**。

### 1-2 个最关键指标
- **Humanity's Last Exam**: 23.8% vs base 9.4%
- **GPQA**: 81.2% vs Search-O1 74.6%

### 局限性
- **Fails when**: 检索到的网页质量差、带偏见或不可信时；需要极低时延的实时交互时；工具调用链过长时，系统仍可能偏航、重复调用或放大前序错误。
- **Assumes**: 主推理模型本身已经足够强；可以访问外部搜索、重排、代码执行与图构建服务；有稳定的网络与执行环境支持。
- **Not designed for**: 端到端学习出的最优工具策略；严格的来源可信性验证；离线无网场景；高风险场景下完全自动化、无需人工审核的决策。

### 可复用组件
1. **查询分解 + 重排 + 迭代改写**：适合所有需要“从模糊问题到高质量证据”的检索型 agent。  
2. **知识图谱式 reasoning memory**：适合长链、多实体、多关系任务，而不只是聊天记忆。  
3. **专职子模型代理化**：把 coding、检索摘要、验证等子任务交给更适合的模型，可直接迁移到其他 agent system。  
4. **显式工具调用 token**：一种低侵入的方式，把工具调度嵌入推理链。

### 复现与资源依赖
虽然代码开源，但论文主实验依赖若干外部服务：
- Bing 搜索；
- Cohere Rerank 3.5；
- Claude 3.5 Sonnet 作为 coding agent；
- DeepSeek-R1 / DeepSeek-V3 作为主推理与图构建组件。

因此它更准确地说是**代码开放、依赖部分闭源服务**的系统；复现成本、延迟和 API 可得性都会影响实际落地。

## Local PDF reference

![[paperPDFs/Agentic_Reasoning_Applications_Autonomous_Web_Exploration_Research_Agents_Foundational_agentic_reasoning/arXiv_2025/2025_Agentic_Reasoning_A_Streamlined_Framework_for_Enhancing_LLM_Reasoning_with_Agentic_Tools.pdf]]