---
title: "RaDA: Retrieval-augmented Web Agent Planning with LLMs"
venue: "Findings of the Association for Computational Linguistics: ACL 2024"
year: 2024
tags:
  - Others
  - task/web-agent-planning
  - task/web-automation
  - task-decomposition
  - dense-retrieval
  - in-context-learning
  - dataset/CompWoB
  - dataset/Mind2Web
  - opensource/partial
core_operator: "将整任务规划拆成“检索增强子任务分解 + 按子任务动态检索动作示例执行”，用可复用的局部技能替代对整条轨迹的直接模仿。"
primary_logic: |
  任务指令 + 当前网页状态 + 轨迹记忆库
  → 先按整任务检索相似轨迹并生成高层子任务计划
  → 再依据当前未完成子任务动态检索局部示例并生成可执行动作
  → 通过子任务完成验证器更新计划状态并迭代直到任务完成
claims:
  - "在 CompWoB 上，RaDA + GPT-3.5 在 original/reverse 设置达到 63.6%/49.0% 成功率，高于 RCI 的 28.7%/19.2% 与 Synapse 的 25.4%/16.0% [evidence: comparison]"
  - "RaDA 的两个组件具有互补性：去掉 RaD 后 CompWoB original 成功率降至 38.0%，去掉 RaA 后为 50.0%，均显著低于完整模型的 63.6% [evidence: ablation]"
  - "在 Mind2Web 复杂子集上，RaDA + GPT-3.5 将 step success rate 从 Synapse 的 21.34% 提升到 26.71%，且 GPT-4 版本达到 41.24% [evidence: comparison]"
related_work_position:
  extends: "Synapse (Zheng et al. 2024b)"
  competes_with: "Synapse (Zheng et al. 2024b); RCI (Kim et al. 2023)"
  complementary_to: "ReAct (Yao et al. 2023); Reflexion (Shinn et al. 2023)"
evidence_strength: strong
pdf_ref: "paperPDFs/Agentic_Reasoning_Applications_Autonomous_Web_Exploration_Research_Agents_Foundational_agentic_reasoning/ACL_2024/2024_RaDA_Retrieval_augmented_Web_Agent_Planning_with_LLMs.pdf"
category: Others
---

# RaDA: Retrieval-augmented Web Agent Planning with LLMs

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [Code](https://github.com/ldilab/RaDA)
> - **Summary**: 这篇论文把网页 agent 的规划拆成“先分解子任务、再围绕当前子任务检索局部轨迹示例执行”，从而在不依赖人工 few-shot exemplars 的前提下提升未见组合任务的泛化。
> - **Key Performance**: CompWoB 上 GPT-3.5 成功率达 63.6%（reverse 为 49.0%）；Mind2Web 复杂子集上 GPT-3.5 step SR 为 26.71%，高于 Synapse 的 21.34%。

> [!info] **Agent Summary**
> - **task_path**: 文本网页指令 + 当前 HTML/动作历史 -> 可执行网页动作序列
> - **bottleneck**: 整轨迹级检索难覆盖未见组合任务，且长上下文会稀释与当前步骤最相关的示例
> - **mechanism_delta**: 把“一次性从任务直接生成整条动作计划”改成“先检索增强地分解子任务，再按当前子任务动态检索并执行”
> - **evidence_signal**: CompWoB original/reverse 均显著超过 RCI 与 Synapse，且 RaD/RaA 消融都明显掉点
> - **reusable_ops**: [子任务级检索查询, LLM 子任务完成验证器]
> - **failure_modes**: [检索库缺少相似局部轨迹时 grounding 失败, 子任务分解或完成验证出错会级联传播]
> - **open_questions**: [如何联合优化检索器与 agent, 如何在缺少现成轨迹库时主动收集 exemplar]

## Part I：问题与挑战

这篇论文要解决的不是“LLM 会不会点击网页元素”，而是更本质的问题：**当目标网页任务是已有技能的新组合时，agent 如何在没有人工编写 few-shot prompt 的情况下，既规划对步骤，又把每一步落到当前页面。**

### 1) 真正的难点是什么
现有网页 agent 大多把问题设成：

- 给定任务描述
- 再加上一些 exemplars
- 让 LLM 直接生成整条动作轨迹

但这有两个根本缺陷：

1. **组合泛化差**  
   像 Synapse 这种方法做的是“整任务级检索”：希望在记忆库中找到一条和当前任务整体相似的历史轨迹。  
   问题是，现实任务经常是“旧技能的新组合”，整条相似轨迹往往不存在。

2. **上下文利用低效**  
   就算检索到整条轨迹，prompt 也会很长。长上下文里既有无关步骤，也容易超过模型上下文预算，导致真正有用的局部动作模式被淹没。

3. **人工 prompt 工程成本高**  
   RCI、AdaPlanner 这类方法依赖人工写示例或 carefully designed few-shot prompts，迁移到新网站/新任务集成本很高。

### 2) 输入/输出接口
- **输入**：任务指令、当前网页状态（HTML；Mind2Web 里先缩减到候选元素）、少量历史动作/观察
- **输出**：程序化网页动作  
  - CompWoB：`click / move / type`
  - Mind2Web：`CLICK / TYPE / SELECT`

### 3) 这篇论文的边界条件
RaDA 的前提不是“从零学会网页操作”，而是：

- 有一批可检索的成功轨迹作为记忆库
- 任务能被拆成少量高层子任务
- 子任务层面的相似行为比整任务层面更容易命中近邻

所以它解决的是**如何把检索式 ICL 从“检索完整答案”改成“检索可组合技能”**。

## Part II：方法与洞察

RaDA 的核心做法是把 planning 明确拆成两阶段：

1. **RaD: Retrieval-augmented Task Decomposition**  
   先用整任务检索相似轨迹，帮助 LLM 生成高层子任务列表。
2. **RaA: Retrieval-augmented Action Generation**  
   再围绕“当前该做的子任务”去检索局部 exemplars，生成具体动作。

中间再加一个 **subtask completion verifier**，不断判断哪些子任务已经完成，从而更新后续检索查询。

### 方法拆解

#### 1. RaD：先把任务变成“可组合单元”
RaD 不直接产出底层动作，而是产出高层 subtasks。  
它的作用是先回答：**这件事可以拆成哪几段。**

这里的关键不是普通的 CoT 分步，而是：

- 它用检索到的相似轨迹给 decomposition 提供环境先验
- 它不需要人工写“如何分解”的 exemplars
- 它把后续 action generation 的检索粒度，从整任务改成子任务

#### 2. Verifier：显式维护计划进度
RaDA 不相信 LLM 能靠长历史一直“隐式记住”做到哪一步。  
所以它增加一个 verifier，根据最近的状态和动作历史判断：

- 哪些子任务已经完成
- 下一步该执行哪个子任务，或剩余哪些子任务

这一步的价值是把“执行进度”从隐状态，变成显式 plan state。

#### 3. RaA：围绕当前子任务做动态检索
RaA 的查询不是固定的整任务描述，而是**subtask-compositional query**：

- 可以是当前单个子任务
- 也可以是若干剩余子任务的拼接

然后从轨迹记忆库中取 top-k exemplars，用于生成当前动作。

这意味着模型每次看到的示例都更局部、更相关，也更短。

### 核心直觉

**改变了什么？**  
从“整任务级检索 + 一次性整计划生成”改成“子任务级检索 + 显式计划进度跟踪”。

**哪个约束/瓶颈被改变了？**
- **分布瓶颈**：把 OOD 的整任务，投影到更常见的子任务分布  
- **信息瓶颈**：把冗长 prompt 改成只服务当前步骤的短 prompt  
- **状态瓶颈**：用 verifier 显式更新进度，而不是让模型自己在长历史里隐式追踪

**能力发生了什么变化？**  
模型不再需要一次性“发明”一条未见过的完整轨迹，而是能把已见过的局部技能重新组合，因此在复杂组合任务、逆序说明、真实网站长任务上更稳。

### 为什么这个设计有效
因果上，RaDA 利用的是一个更容易成立的假设：

> **整任务近邻可能不存在，但子任务近邻更可能存在。**

因此：
- 检索成功率更高
- exemplar 与当前状态的相关性更高
- prompt 更短，减少无关示例干扰
- verifier 让检索 query 跟随执行状态更新，避免“一开始检对了、后面全漂了”

### 策略性 trade-off

| 设计选择 | 改变了什么瓶颈 | 收益 | 代价/风险 |
|---|---|---|---|
| 整任务 → 子任务分解（RaD） | 把未见组合任务转到更可复用的局部技能空间 | 更强组合泛化 | 分解错误会误导后续全部执行 |
| 静态示例 → 动态子任务检索（RaA） | 提高示例与当前步骤的局部相关性 | 更强 grounding、更短单次 prompt | 需要多次检索和多次 LLM 调用 |
| 隐式进度 → 显式 verifier | 让 query 随执行状态变化 | 降低计划漂移 | 验证误判会跳错子任务 |
| 人工 few-shot → 被动轨迹记忆 | 降低 prompt 工程和标注成本 | 更易迁移到新任务集 | 依赖轨迹库覆盖度与质量 |

## Part III：证据与局限

### 关键证据信号

- **比较信号｜CompWoB**  
  在 GPT-3.5 下，RaDA 在 CompWoB original/reverse 上分别达到 **63.6% / 49.0%**，明显高于 RCI 的 **28.7% / 19.2%** 和 Synapse 的 **25.4% / 16.0%**。  
  最有说服力的一点是 reverse 设置：指令顺序被扰动后，传统方法掉得很厉害，而 RaDA 仍保持明显优势，说明它学到的不是“整轨迹模板匹配”，而是更接近可重组的计划结构。

- **消融信号｜两部分缺一不可**  
  只保留 RaA（无显式 RaD）时，CompWoB original 只有 **38.0%**；只保留 RaD（无动态 RaA）时是 **50.0%**；完整 RaDA 达到 **63.6%**。  
  这说明：
  - 仅仅做子任务检索不够
  - 仅仅做高层分解也不够  
  真正有效的是**分解 + 动态检索 grounding**的组合。

- **跨基准信号｜真实网站上的收益更集中于复杂任务**  
  在 Mind2Web 全集上，RaDA 相对 Synapse 的提升不算大但一致；而在复杂子集上，step SR 从 **21.34%** 提升到 **26.71%**。  
  这很符合论文主张：RaDA 的优势主要来自**处理组合深度更高、整轨迹近邻更难命中**的场景。

- **效率信号｜更短的单次上下文**  
  相比把所有 exemplars 一次性拼进 prompt 的 RaA，RaDA 单次调用平均总 token 从 **4110.1** 降到 **1969.4**，最大 prompt 从 **9761.3** 降到 **2722.4**。  
  需要注意：RaDA 因为分子任务多次调用，**总 token 不一定更少**。论文的论点是，transformer 对长上下文的计算代价更差，因此“多次短调用”仍更划算。这个结论更偏计算分析，而不是完整的端到端时延报告。

### 能力跳跃到底体现在哪
相对 prior work，RaDA 的真正跳跃不是“换了更强的 LLM”，而是：

- 把检索单位从 **whole trajectory** 改成 **subtask**
- 把执行状态从 **隐式历史** 改成 **显式 plan state**
- 把 ICL 从“找一条像完整答案的轨迹”改成“找能拼起来的局部技能”

这使它在 prior methods 最脆弱的场景——未见组合、逆序说明、复杂长任务——上更稳。

### 局限性

- Fails when: 检索库里缺少相似的局部轨迹，或任务的子任务边界本身模糊、强依赖全局状态时；CompWoB 的 reverse n-way / transition 类任务在 GPT-3.5 下仍然困难。
- Assumes: 依赖可用的成功轨迹库（MiniWoB base tasks / Mind2Web train split）、一个足够好的 dense retriever、以及闭源 OpenAI GPT-3.5/GPT-4 与 `text-embedding-ada-002` API；Mind2Web 全量结果还受 API 成本约束，部分评测使用官方日志与 oracle decomposition trigger。
- Not designed for: 无示例记忆的从零探索、检索器与 agent 的端到端联合训练、以及超出网页动作空间定义的开放式环境控制。

### 可复用组件
这篇论文最值得复用的不是某个 prompt，而是以下几个系统操作：

- **子任务级 exemplar memory**
- **LLM 子任务分解器**
- **LLM 子任务完成验证器**
- **随执行状态变化的动态检索 query**

这些模块可以直接和 ReAct、Reflexion 一类 agent 框架结合。

## Local PDF reference

![[paperPDFs/Agentic_Reasoning_Applications_Autonomous_Web_Exploration_Research_Agents_Foundational_agentic_reasoning/ACL_2024/2024_RaDA_Retrieval_augmented_Web_Agent_Planning_with_LLMs.pdf]]