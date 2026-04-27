---
title: "WebThinker: Empowering Large Reasoning Models with Deep Research Capability"
venue: NeurIPS
year: 2025
tags:
  - Others
  - task/complex-reasoning
  - task/report-generation
  - reinforcement-learning
  - dataset/GPQA
  - dataset/GAIA
  - dataset/WebWalkerQA
  - dataset/HLE
  - dataset/Glaive
  - opensource/partial
core_operator: "让大型推理模型在连续思考中层级式调用深度网页探索与报告写作工具，并用在线偏好优化学会更高效地搜、点、写。"
primary_logic: |
  用户研究问题/复杂问答 → 主推理器按需触发 Deep Web Explorer 进行搜索、链接点击与信息抽取，并把证据写入文档记忆、交错调用起草/检查/编辑工具 → 输出直接答案或逐步完成的研究报告
claims:
  - "WebThinker-32B-RL 在 GAIA、WebWalkerQA、HLE 上分别达到 48.5、46.5、15.8 Pass@1，超过 Search-o1-32B 的 39.8、34.1、10.8 [evidence: comparison]"
  - "在 Glaive 报告生成上，WebThinker-32B-RL 的平均评分为 8.1，高于 Gemini2.0 Deep Research 的 7.9，优势主要体现在完整性和讨论充分性 [evidence: comparison]"
  - "移除 Deep Web Explorer 会使复杂问题求解平均分从 45.4 降到 38.3，移除自动报告起草会使报告生成平均分从 8.1 降到 6.6 [evidence: ablation]"
related_work_position:
  extends: "Search-o1 (Li et al. 2025)"
  competes_with: "Search-o1 (Li et al. 2025); OpenAI Deep Research"
  complementary_to: "Self-RAG (Asai et al. 2024); RECOMP (Xu et al. 2024)"
evidence_strength: strong
pdf_ref: "paperPDFs/Agentic_Reasoning_Applications_Autonomous_Web_Exploration_Research_Agents_Foundational_agentic_reasoning/arXiv_2025/2025_WebThinker_Empowering_Large_Reasoning_Models_with_Deep_Research_Capability.pdf"
category: Others
---

# WebThinker: Empowering Large Reasoning Models with Deep Research Capability

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2504.21776), [Code](https://github.com/RUC-NLPIR/WebThinker)
> - **Summary**: 该文把大型推理模型从“只会长链思考”升级为“能在思考中主动搜网页、点链接、边写边改报告”的深研究代理，从而显著提升知识密集型问题求解与研究报告生成能力。
> - **Key Performance**: GAIA/HLE Pass@1 达到 48.5/15.8（Search-o1-32B 为 39.8/10.8）；Glaive 报告平均分 8.1（Gemini2.0 Deep Research 为 7.9）

> [!info] **Agent Summary**
> - **task_path**: 文本研究问题/复杂知识任务 -> 最终答案或长篇研究报告
> - **bottleneck**: 静态参数知识与预定义 RAG 流程无法在推理过程中动态补证据、深度导航网页并同步组织长文输出
> - **mechanism_delta**: 将搜索、链接点击、章节起草/检查/编辑都嵌入 LRM 的连续思考过程，并用在线偏好优化稳定工具使用策略
> - **evidence_signal**: 4 个复杂推理基准 + Glaive 报告生成整体领先，且去掉 Deep Web Explorer/自动起草会显著掉点
> - **reusable_ops**: [层级式搜索-点击网页探索, 思考-搜索-起草交错调度]
> - **failure_modes**: [无法处理图像和视频等多模态网页信息, 目前不支持更复杂的 GUI 网页交互]
> - **open_questions**: [如何扩展到多模态 deep research, 如何在更丰富工具与真实浏览器环境中稳定泛化]

## Part I：问题与挑战

这篇论文要解决的核心，不是“模型会不会推理”，而是“推理如何真正接上外部世界”。

大型推理模型（LRM）已经能做长链条思考，但面对 **知识密集、开放世界、需要多网页交叉核验** 的任务时，单靠参数内知识明显不够：  
- 问题可能依赖最新网页信息；  
- 证据可能藏在搜索结果后的二级/三级页面；  
- 最终输出不只是一个短答案，而是需要持续整合证据的研究报告。

作者认为，真正瓶颈在于现有方法把“检索”和“推理”割裂开了：

1. **标准 RAG 太浅**：通常只是先检索 top-k 文档，再把结果塞给模型；检索目标不能随着中间推理状态动态变化。  
2. **预定义 workflow 太僵硬**：即使有迭代 RAG 或 query planning，流程通常还是外部脚本决定，模型并没有真正掌握“何时搜、搜什么、要不要继续点开网页”的控制权。  
3. **长报告写作与信息获取脱节**：很多系统是“先搜完再写”，导致报告结构、覆盖面和后续新证据之间缺少闭环。

所以本文的输入/输出边界很清楚：  
- **输入**：文本研究问题 + 任务指令 + 工具集；  
- **输出**：要么是直接答案，要么是逐步完成的研究报告。  

它也有明确边界条件：当前系统是 **text-only**，依赖搜索引擎与网页爬取，不处理图像/视频内容，也不支持完整 GUI 浏览器交互。

**Why now**：因为 LRM 已经具备较强的长程思考能力，下一步短板不再只是“想不长”，而是“想的时候拿不到外部真信息”。WebThinker 抓住的正是这个时间点的系统瓶颈。

## Part II：方法与洞察

WebThinker 的设计哲学可以概括为一句话：

**把“搜索/导航/写作”从推理前后的外挂流程，改成推理过程中的原生动作。**

### 方法骨架

#### 1. Deep Web Explorer：把“搜一下”升级成“搜 + 点 + 再搜”

主推理器在思考时，如果发现知识缺口，不是简单调用一次搜索 API，而是触发一个 **Deep Web Explorer** 子过程。这个子过程本身可以：

- 发起新的搜索；
- 点击链接或按钮进入网页；
- 从当前网页内容中提炼与当前知识缺口最相关的信息；
- 持续递归，直到收集到足够证据，再返回给主推理器。

这使得 WebThinker 不再停留于搜索摘要层，而是能做 **网页层级遍历**。  
系统层面上，相当于把“需要什么信息”交给主推理器决定，把“如何沿网页图继续挖”交给 Explorer 决定。

#### 2. Autonomous Think-Search-and-Draft：边想边搜边写

在报告生成模式下，作者不让模型等到所有搜索结束后一次性出全文，而是允许它在推理过程中随时：

- 写某一节；
- 检查当前报告结构；
- 编辑已有报告。

这里用了一个关键拆分：  
**主 LRM 负责任务编排，辅助 LLM 负责具体写作/编辑。**

同时，所有浏览过的网页会进入一个 **document memory**。当需要起草某一节时，系统从这份记忆里取相关证据给辅助写作模型。这样做的作用是：

- 主推理链保持干净，不被整篇报告全文污染；
- 报告变成一个“可持续修改的外部工作记忆”；
- 新发现的证据可以回流到已有章节。

#### 3. Iterative Online DPO：训练模型学会“用工具”，而不只是“给答案”

作者没有把训练目标只放在最终正确率上，而是对**整条轨迹**做偏好学习。偏好优先级是：

1. 最终答案/报告质量更好；
2. 如果都对，优先工具调用更少的轨迹；
3. 如果都对且工具数相同，优先更简洁的推理轨迹。

再通过 **iterative online DPO** 反复采样新轨迹、构造偏好对、更新策略。  
这一步的本质不是单纯增强 CoT，而是让模型学会更稳定地做 **何时搜、搜几次、是否点击、何时起草** 的策略选择。

### 核心直觉

WebThinker 的关键变化有三层：

1. **从“固定上下文推理”变成“推理驱动的按需取证”**  
   - 以前：先给一坨检索结果，再让模型想。  
   - 现在：模型先想，发现缺口后再决定搜什么、点什么。  
   - 改变的是信息瓶颈：证据不再是一次性灌入，而是随不确定性动态获取。  

2. **从“浅检索”变成“层级网页探索”**  
   - 以前：停在搜索结果页或 top-k 摘要。  
   - 现在：可以点开链接深入网页。  
   - 改变的是可访问证据分布：更多关键事实从搜索摘要不可见区域进入可用上下文。  

3. **从“一次性生成长文”变成“持续维护一个外部报告对象”**  
   - 以前：长报告完全依赖隐藏状态一次性吐出。  
   - 现在：章节可以被写、查、改。  
   - 改变的是长输出约束：报告成为显式工作记忆，降低长程写作中的遗漏、漂移和结构失配。  

为什么这套设计有效？  
因为它把“思考状态”直接变成“动作触发条件”。模型不是在固定证据上强行做深推理，而是在出现不确定性时主动补证据；不是等最后再综合，而是在信息足够时即时沉淀成章节。这种闭环比 RAG 的前置式检索更贴近真实研究流程。

### 战略权衡

| 设计选择 | 带来的能力 | 代价/风险 |
|---|---|---|
| Deep Web Explorer 替代浅层 RAG | 能深入网页、提升证据召回和多跳信息获取 | 延迟更高，依赖搜索 API 与网页可访问性 |
| 主 LRM 编排 + 辅助 LLM 写作 | 推理与长文编辑解耦，报告更稳定 | 系统更复杂，多模型协同增加工程成本 |
| 在线偏好优化工具使用 | 工具调用更高效、更稳定 | 需要环境交互采样与额外训练算力 |
| 文档记忆驱动写作 | 支持边搜边写、边发现边修订 | 记忆检索本身仍可能遗漏关键证据 |

## Part III：证据与局限

### 关键证据信号

- **比较信号：体系结构本身就有效，不只是 RL 在起作用。**  
  training-free 的 WebThinker-32B-Base 已经超过 Search-o1-32B：  
  - GAIA：44.7 vs 39.8  
  - WebWalkerQA：41.9 vs 34.1  
  - HLE：13.0 vs 10.8  
  这说明能力提升首先来自“把搜索/导航嵌入推理”的系统设计，而不只是后续训练。

- **比较信号：RL 进一步把工具策略打磨出来。**  
  WebThinker-32B-RL 在 GAIA / WebWalkerQA / HLE 达到 **48.5 / 46.5 / 15.8**，相较 Base 继续提升，说明在线偏好优化确实学到了更好的工具使用策略，而非只复制现成轨迹。

- **报告生成信号：提升主要来自覆盖面与展开深度，而不是纯 factuality。**  
  Glaive 上 WebThinker-32B-RL 平均 **8.1**，略高于 Gemini2.0 Deep Research 的 **7.9**。更重要的是，它的优势集中在 **Completeness / Thoroughness**，而 factuality 基本打平。这说明它的核心增益是“搜得更全、写得更充分”，不是单纯减少事实错误。

- **消融信号：组件作用与论文叙事一致。**  
  - 去掉 **Deep Web Explorer**：复杂问题求解平均分 **45.4 → 38.3**。  
  - 去掉 **Auto Report Draft**：报告平均分 **8.1 → 6.6**，降幅最大。  
  - 去掉 **Check & Edit**：报告 coherence 明显下降（7.9 → 6.9）。  
  - 用 **offline DPO** 替代在线训练：效果弱于 iterative online DPO。  
  这组结果比较好地建立了“哪个能力来自哪个模块”的因果链。

- **迁移信号：不是只对 QwQ-32B 有效。**  
  在 DeepSeek-R1 的 7B/14B/32B 上，WebThinker 都稳定优于 direct generation 和 standard RAG，说明它更像一个可迁移的 agentic scaffold，而非某个特定 backbone 的 prompt trick。

### 局限性

- **Fails when:**  
  任务关键证据主要存在于图像、视频、复杂 GUI 页面、登录后页面或需要真实浏览器交互的网站时，当前 text-only + crawler 的设定会失效或明显退化。

- **Assumes:**  
  依赖外部搜索引擎与网页抓取（文中用 Bing Web Search API + Crawl4AI）；依赖较长上下文和较高推理预算（推理最大 81,920 tokens）；报告与评测还依赖辅助 LLM 和 LLM-as-Judge，这会影响可复现性与成本。网页结果还会受时间、地区、接口变化影响。

- **Not designed for:**  
  多模态 deep research、完整 GUI browser agent、严格带引用可核验的学术综述生成，并非其当前明确目标。论文中的报告写作甚至明确不要求引用输出，因此“研究报告”更接近 evidence-grounded synthesis，而不是正式学术论文写作。

### 可复用部件

1. **层级式网页探索器**：把“搜索”和“点击跳转”放进同一个 reasoning-conditioned 子代理。  
2. **外部文档记忆 + 分章节写作工具**：适合任何需要长文合成的 agent。  
3. **轨迹偏好规则**：正确性/质量 > 工具效率 > 推理简洁度，这种偏好设计可迁移到其他 tool-using agent。  

总体看，WebThinker 的能力跃迁主要不在于又做了一个更复杂的 RAG，而在于它把 **取证、导航、起草** 真正变成了推理过程中的内生动作。对“deep research agent”这类系统来说，这是比单纯加检索更关键的一步。

![[paperPDFs/Agentic_Reasoning_Applications_Autonomous_Web_Exploration_Research_Agents_Foundational_agentic_reasoning/arXiv_2025/2025_WebThinker_Empowering_Large_Reasoning_Models_with_Deep_Research_Capability.pdf]]