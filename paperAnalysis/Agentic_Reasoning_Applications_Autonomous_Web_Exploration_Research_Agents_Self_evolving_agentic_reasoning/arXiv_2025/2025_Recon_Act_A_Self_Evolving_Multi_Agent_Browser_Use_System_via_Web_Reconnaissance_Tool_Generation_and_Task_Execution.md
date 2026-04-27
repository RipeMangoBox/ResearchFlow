---
title: "Recon-Act: A Self-Evolving Multi-Agent Browser-Use System via Web Reconnaissance, Tool Generation, and Task Execution"
venue: arXiv
year: 2025
tags:
  - Others
  - task/browser-use
  - task/web-navigation
  - multi-agent
  - tool-generation
  - code-generation
  - dataset/VisualWebArena
  - opensource/partial
core_operator: "通过对比成功/失败网页轨迹生成并注册 hint/decision 双模工具，让执行代理在后续任务中以更少试错完成浏览器操作。"
primary_logic: |
  用户查询与当前网页上下文 → 侦察团队对成功/失败轨迹做逐步对比并生成 generalized tools（提示或决策工具） → 动作团队按路由调用工具并执行网页动作 → 轨迹反馈继续更新工具库
claims:
  - "Recon-Act 在 VisualWebArena 上达到 36.48% 总体成功率，高于文中最强自动基线 ExAct 的 33.74% [evidence: comparison]"
  - "Recon-Act 在 VisualWebArena 的 Shopping 子集上达到 39.27%，比文中先前最佳 32.30% 提升 6.97 个百分点 [evidence: comparison]"
  - "论文报告的实际系统仅达到 Level 3，需要人工 Analyst 与 Tool Manager 参与，而非端到端全自动进化 [evidence: case-study]"
related_work_position:
  extends: "Profile-aware maneuvering (Xie et al. 2025)"
  competes_with: "ExAct (Yu et al. 2025); TreeSearch (Koh et al. 2024b)"
  complementary_to: "GenTool (He et al. 2025); ToolLLM (Qin et al. 2023)"
evidence_strength: moderate
pdf_ref: paperPDFs/Agentic_Reasoning_Applications_Autonomous_Web_Exploration_Research_Agents_Self_evolving_agentic_reasoning/arXiv_2025/2025_Recon_Act_A_Self_Evolving_Multi_Agent_Browser_Use_System_via_Web_Reconnaissance_Tool_Generation_and_Task_Execution.pdf
category: Others
---

# Recon-Act: A Self-Evolving Multi-Agent Browser-Use System via Web Reconnaissance, Tool Generation, and Task Execution

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2509.21072), [Code](https://github.com/inclusionAI/AWorld/tree/main/examples/visualwebarena)
> - **Summary**: 这篇论文把网页代理的“长程试错”问题改写成“先侦察、再把失败经验编译成工具”的问题，用成功/失败轨迹对比生成可复用的 hint/decision 工具，提升浏览器任务完成率。
> - **Key Performance**: VisualWebArena 总体成功率 **36.48%**；Shopping 子集 **39.27%**，图中报告平均步数约 **4.9**

> [!info] **Agent Summary**
> - **task_path**: 用户查询 + 当前网页视觉/结构上下文 -> 多步浏览器动作序列 / 任务完成
> - **bottleneck**: 信息密集网页中的关键线索提取与可复用操作策略缺失，导致长程任务依赖大量试错
> - **mechanism_delta**: 在执行链外加入侦察团队，把失败/成功轨迹对比结果编译为可在线注册的 generalized tools（hint 或 decision），替代反复在线搜索
> - **evidence_signal**: VisualWebArena 上 36.48% 总体成功率，超过 ExAct 的 33.74%；Shopping 达到 39.27%
> - **reusable_ops**: [逐步轨迹对比分析, hint/decision 双模工具注册]
> - **failure_modes**: [新网站结构超出工具覆盖, Master 选错工具或工具合并不充分]
> - **open_questions**: [去掉人工 Analyst/Tool Manager 后是否仍能稳定进化, 提升中有多少来自 GPT-5-Chat 而非 reconnaissance 机制]

## Part I：问题与挑战

**What/Why**：这篇工作要解决的，不只是“网页代理不会点按钮”，而是**面对陌生网页时，代理无法快速抽取任务相关信息，并把反复出现的失败模式沉淀为可复用能力**。

浏览器环境的难点有三层：

1. **观察空间过密**：同一页里有大量视觉元素、文本块、URL 结构、列表排序、分页入口，但真正与任务有关的只是一小部分。  
2. **长程决策脆弱**：一旦前几步理解错页面结构，后面就会进入反复点击、回退、重试。  
3. **现有动态规划方法代价高**：像 TreeSearch、ExAct、WebDreamer 这类方法能搜索路径，但往往要靠更多模拟和回溯换成功率。

论文的关键判断是：**浏览器任务的瓶颈不只在 action space，更在 observation space**。很多错误并不是“不会执行动作”，而是“没先看懂该看什么”。所以作者提出先做 **reconnaissance（侦察）**：用少量探索动作抽取任务相关信息，再把这种经验固化成工具。

**为什么是现在**：  
一方面，多模态模型已经具备基本网页理解和工具调用能力；另一方面，代码模型和函数调用能力让“让模型自己写工具”变得可行。于是，网页代理可以从“每次都临场推理”转向“把失败经验编译成可复用工具”。

**输入/输出接口**：
- **输入**：用户 query + 浏览器上下文（截图、URL、SOM 文本、历史轨迹等）
- **输出**：通过 Playwright 执行的一步步浏览器动作，直到任务完成或失败

**边界条件**：
- 训练依赖少量人工构造的 cold-start queries，且需要成功/失败轨迹
- 当前实现仅到 **Level 3**
- Analyst 与 Tool Manager 仍保留人工介入
- 主要在 VisualWebArena 的固定网站域上验证

---

## Part II：方法与洞察

**How**：论文引入的核心因果旋钮是——**把“在线试错”转成“离线归因 + 工具沉淀”**。

### 方法骨架

Recon-Act 由两个团队组成：

#### 1) Reconnaissance Team（只在训练阶段工作）
- **Analyst**：读取失败轨迹、成功轨迹和当前浏览器上下文，做逐步对比分析，找出失败根因
- **Coder**：把 remedial strategy 写成可执行工具代码
- **Preset Recon Tools**：如 URL、图像、SOM 观察工具，用于辅助页面结构分析

其输出不是普通文字总结，而是统一称为 **generalized tools** 的资产，分两类：
- **Hint tools**：返回提示性信号，供执行代理参考
- **Decision tools**：直接返回应执行的动作，具有更强约束力

#### 2) Action Team（训练和推理都工作）
- **Master**：根据 query + context 判断当前子任务，并选择调用哪个工具
- **Tool Manager**：注册/更新工具，处理合并与条件分支
- **Execution Agent**：兜底动作生成器；若没有合适工具或工具失败，就由它直接出动作

整个闭环是：

**Rollout → Evaluate → Generate → Update**

即：
1. 用当前系统在训练集上 rollout  
2. evaluator 判断成功/失败  
3. 侦察团队对比失败/成功轨迹生成新工具  
4. 工具注册后重新跑任务，直到无明显提升

### 核心直觉

Recon-Act 最重要的变化，不是让 agent “更会搜索”，而是让 agent **把一次失败学成以后少犯同类错误**。

#### 具体因果链条
- **What changed**：从“每步都在线推理下一动作”改为“先对失败做侦察分析，再生成可复用工具”
- **Which bottleneck changed**：把高熵的网页观察/动作映射，压缩成低熵的、站点相关的操作原语
- **What capability changed**：长程任务中的回退和试错减少，网站特定结构上的动作更稳定

换句话说，作者认为浏览器环境里很多难点其实是**重复出现的结构化问题**：
- 去某个 category page
- 对商品/二手物品按价格排序
- 找到作者主页
- 在 Reddit 页面里定位帖子时间或图像描述

这些问题如果每次都靠大模型从零推理，成本高且不稳；但如果一旦被侦察团队识别并编译成工具，后续任务就可以直接复用。

这也是为什么论文强调：
- 工具最好直接返回**任务显著信息**
- 某些稳定场景下甚至直接返回**可执行动作**

本质上，Recon-Act 是在给网页 agent 增加一个“**经验编译层**”。

### 为什么这套设计可能有效

1. **网页站点高度模板化**  
   同一网站内大量页面遵循类似 DOM/URL/交互模式，适合被工具化。

2. **失败有可对比性**  
   论文显式使用成功轨迹与失败轨迹做 step-level 对比，这比单纯自反思更容易定位“哪里偏了”。

3. **Hint / Decision 分流降低风险**  
   不确定的知识只当提示，稳定的操作才直接接管动作，减少错误工具强行控制执行链的风险。

4. **统一工具接口降低生成难度**  
   所有工具接受统一风格参数并返回字符串，减少 coder 的接口复杂度，便于在线注册。

### 战略权衡

| 设计选择 | 解决的瓶颈 | 能力收益 | 代价 |
|---|---|---|---|
| 侦察后行动 | 在线盲搜导致长程试错 | 把失败沉淀成可复用技能 | 训练流程更复杂 |
| Hint 工具 | 页面信息太多但不够确定 | 给执行器补充关键信号 | 最终动作仍依赖模型 |
| Decision 工具 | 某些网页动作高度模板化 | 直接减少多步推理和点击 | 站点特异性更强 |
| 人工 Tool Manager 合并工具 | 自动生成工具易碎片化 | 保持兼容性与命名一致 | 自动化不足 |
| 硬编码工具路由 | 提高推理期工具调用稳定性 | 降低误调工具概率 | 泛化性与纯学习性受限 |

---

## Part III：证据与局限

**So what**：论文最重要的实证信号是，它表明**浏览器代理不一定非要靠更重的搜索才能提升；把 recurring failure 编译成工具，也能带来可见增益**。

### 关键证据信号

1. **主比较信号：VisualWebArena SOTA**
   - Recon-Act 总体成功率 **36.48%**
   - 高于表中最强自动基线 ExAct 的 **33.74%**
   - 也明显高于 TreeSearch 的 **26.40%** 与 WebDreamer GPT-4o 的 **23.20%**

   这说明：工具中心的闭环式演化，至少在该 benchmark 上可以和搜索式方法竞争，甚至更强。

2. **域别信号：Shopping 提升最明显**
   - Shopping 子集 **39.27%**
   - 相对表中最佳基线 **32.30%** 提升 **6.97** 个点

   这与论文机制很一致：购物网站里“分类跳转、排序、图像定位”这类结构化操作很多，最适合被蒸馏为 Decision tools。

3. **效率信号：步数适中且更稳定**
   - 图 1 报告 Recon-Act 平均步数约 **4.9**
   - 明显低于 TreeSearch 的 **11.5**

   这支持了作者关于“减少自我纠错与回退”的叙述，不过这更多是侧面信号，不是严格控制变量下的因果证据。

### 证据强度为何只是 moderate

虽然结果不错，但证据仍应保守看待：
- 只在 **1 个数据集** 上验证
- **缺少关键 ablation**：没有拆分出“仅 GPT-5-Chat”“仅工具库”“仅 recon team”“去掉人类 Tool Manager”等版本
- 与 prior work 的比较存在 **backbone 混杂**：很多基线是 GPT-4o / Dreamer 系列，而 Recon-Act 使用 **GPT-5-Chat**
- 论文强调“self-evolving”，但当前实现仍是 **Level 3 人机协作系统**

### 局限性

- **Fails when**: 网站结构、URL 规律、排序/导航控件超出当前 11 个工具的覆盖范围时；更开放、更异构的网站环境中，其效果尚未被实证验证。
- **Assumes**: 需要同时拥有成功与失败轨迹、少量人工构造训练集（每域少于 10 个示例）、人工 Analyst 与 Tool Manager、闭源 GPT-5-Chat，以及 Playwright 浏览器执行环境。
- **Not designed for**: 端到端全自动 Level 5/6 自主演化、完全无监督的工具发现、跨任意网站的零样本普适 browser-use。

还需要特别指出两点可复现性边界：
- **闭源模型依赖**：核心 agent 使用 GPT-5-Chat
- **人工合并依赖**：Tool Manager 负责命名、分支调整、工具合并，这正是最难自动化的部分之一

### 可复用组件

这篇论文里最值得迁移的不是某个具体网页工具，而是下面几种操作范式：

- **失败/成功轨迹的逐步对比分析**：适合任何长程 agent 的 error mining
- **hint / decision 双模工具设计**：把“不确定建议”和“高确定动作”分层处理
- **统一工具 schema + 在线注册**：降低自动代码生成接入成本
- **Execution Agent 兜底机制**：避免工具系统在长尾场景下完全失效

一句话评价：**Recon-Act 的真正贡献，不是多加了几个 agent，而是把网页代理中的 recurring failure 变成了可积累的工具资产；但当前证据还不足以完全分离“机制贡献”和“更强基础模型 + 人工协同”的贡献。**

## Local PDF reference

![[paperPDFs/Agentic_Reasoning_Applications_Autonomous_Web_Exploration_Research_Agents_Self_evolving_agentic_reasoning/arXiv_2025/2025_Recon_Act_A_Self_Evolving_Multi_Agent_Browser_Use_System_via_Web_Reconnaissance_Tool_Generation_and_Task_Execution.pdf]]