---
title: "AgentOccam: A Simple Yet Strong Baseline for LLM-Based Web Agents"
venue: ICLR
year: 2025
tags:
  - Others
  - task/web-navigation
  - action-space-alignment
  - observation-space-alignment
  - planning-tree
  - dataset/WebArena
  - dataset/WebVoyager
  - opensource/full
core_operator: "通过裁剪网页动作、压缩并选择性回放网页观察，再把 branch/prune 作为可生成动作加入，使 LLM 在更贴近其文本预训练分布的接口上直接执行网页任务。"
primary_logic: |
  自然语言网页任务 + 文本化页面/历史 → 规则化裁剪动作空间、压缩页面表示、按关键节点与子计划选择性保留历史 → LLM 直接生成更稳定的导航动作与最终答案
claims:
  - "AGENTOCCAM 在 WebArena 上达到 43.1% 成功率，超过 WebPilot 的 37.2% 和 SteP-replication 的 33.3% [evidence: comparison]"
  - "在同样使用 GPT-4-Turbo 的 plain web agent 设置下，观测/动作空间对齐将 WebArena 成功率从 16.5% 提升到 43.1%（+26.6 绝对点） [evidence: ablation]"
  - "在 WebVoyager 的确定答案子集上，AGENTOCCAM 达到 54.3%，高于 Agent-E 的 51.9% [evidence: comparison]"
related_work_position:
  extends: "WebArena agent (Zhou et al. 2023b)"
  competes_with: "SteP (Sodhi et al. 2024); WebPilot (Zhang et al. 2024)"
  complementary_to: "Agent Workflow Memory (Wang et al. 2024); LM-Tree Search (Koh et al. 2024b)"
evidence_strength: strong
pdf_ref: paperPDFs/Agentic_Reasoning_Applications_Autonomous_Web_Exploration_Research_Agents_Self_evolving_agentic_reasoning/arXiv_2024/2024_AgentOccam_A_Simple_Yet_Strong_Baseline_for_LLM_Based_Web_Agents.pdf
category: Others
---

# AgentOccam: A Simple Yet Strong Baseline for LLM-Based Web Agents

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2410.13825), [Code](https://github.com/amazon-science/AgentOccam)
> - **Summary**: 这篇论文认为网页智能体的核心短板不一定是“缺更复杂的 agent 策略”，而是网页观测与动作接口和 LLM 的文本预训练分布错位；只做接口对齐，就能显著释放零样本网页执行能力。
> - **Key Performance**: WebArena 成功率 43.1%（vs. WebPilot 37.2%，SteP 33.3%）；WebVoyager 确定答案子集 54.3%（vs. Agent-E 51.9%）

> [!info] **Agent Summary**
> - **task_path**: 自然语言网页任务 + 文本 DOM/可访问树 + 历史轨迹 -> 浏览器动作序列/最终答案
> - **bottleneck**: 网页 DOM 冗长噪声、低层具身动作与 LLM 预训练分布不匹配、历史回放无选择导致上下文污染
> - **mechanism_delta**: 不再堆复杂控制器，而是把网页交互改写成“紧凑文本理解 + 少量高价值动作 + 计划树管理”的同层生成问题
> - **evidence_signal**: WebArena 上从 vanilla 16.5% 到 43.1% 的逐组件消融跃升，是最直接的因果证据
> - **reusable_ops**: [action-space-pruning, pivotal-node-history-replay]
> - **failure_modes**: [relevant-node-omission, multisite-long-horizon-fragility]
> - **open_questions**: [visual-ui-transfer, adaptive-plan-and-memory-selection]

## Part I：问题与挑战

这篇论文瞄准的是 **LLM-based web agent** 的一个更基础问题：  
**LLM 本身已经很强，但网页环境给它的“输入/输出接口”并不像它训练时熟悉的语言任务。**

### 真正的难点是什么？

过去很多方法把重点放在：
- 手写 prompt 模板
- 角色分工 / 多智能体
- 反思式流程
- MCTS / 搜索 / 采样

但作者指出，网页任务里一个更底层的瓶颈是：

1. **观察空间错位**  
   HTML / accessibility tree 很冗长，充满结构性重复 token；网页上真正有用的信息只占一小部分。

2. **动作空间错位**  
   scroll、hover、tab focus、goto、press 这类动作更像“低层具身控制”，而不是 LLM 预训练中常见的语言决策。

3. **历史记忆错位**  
   长网页任务必须记忆前文，但把整段历史线性塞回上下文，会让模型被旧页面、旧分支和无关元素淹没。

### 输入/输出接口

- **输入**：任务指令 + 当前文本页面 + 部分历史
- **输出**：网页动作序列（如 click/type/go_back/note/stop/branch/prune）直到结束

### 为什么现在值得做？

因为 GPT-4 级别模型已经有很强的零样本文本理解与推理能力。  
此时如果性能仍差，未必是“不会想”，更可能是“接口喂得不对”。  
所以这篇论文的核心问题是：

> **What/Why**：网页智能体的真瓶颈是不是策略层，而其实是接口层？如果是，先修接口能不能比继续加复杂 agent workflow 更有效？

### 边界条件

这篇工作很明确地限制在：
- **text-only** 网页代理
- 不用 in-context examples
- 不用搜索/采样/多次回滚
- 不加 evaluator / reflector / 多角色控制器
- 通过对所有 markup-language 网页通用的规则完成对齐

也因此，它的结论更像是在说：  
**先把 LLM 放到它擅长的输入输出分布里，再谈复杂 agent 策略。**

## Part II：方法与洞察

方法非常“Occam”：

> 不增加上层控制器，而只改 **观察空间** 和 **动作空间**。

### 核心直觉

这篇论文最重要的因果链条是：

- **把网页交互从低层、噪声大、具身感强的接口**
- 改成 **更像语言理解与离散决策的问题**
- 从而改变了 LLM 面对的 **分布错位 / 上下文噪声 / 动作熵**
- 最终提升了 **零样本网页导航、检索、长程执行与失败恢复能力**

更具体地说：

1. **动作更少、更抽象**  
   降低无效探索和误用概率。

2. **页面更短、更可读**  
   提高信息密度，让模型更容易找对元素、读懂表格、定位关键内容。

3. **历史更“按计划作用域”组织**  
   让模型只看当前子任务相关的过去，而不是背整个流水账。

4. **规划也作为动作生成**  
   用 `branch / prune` 把计划树内化到同一套 LLM 输出里，而不是额外加一个 planner controller。

### 方法拆解

#### 1. 动作空间对齐

作者做了三类处理：

- **删除非必要动作**：如 `noop`、tab 相关操作、`go_forward`、`goto`
- **删除具身要求高的动作**：如 `hover`、`press`、`scroll`
- **保留/新增高价值动作**：`click`、`type`、`go_back`、`note`、`stop`、`go_home`

其中一个关键决定是：**禁用 scroll，直接把整页文本送给模型。**  
因为作者发现 scroll 常常变成“模型不知道怎么办时的可逆空转动作”。

#### 2. 观测空间对齐

作者把页面表征从“面向浏览器渲染”改成“面向 LLM 阅读”：

- 合并重复的静态文本与交互元素
- 把 table/list 转成 Markdown
- 删掉大量结构性冗余 token

这一步不是压缩信息量，而是 **压缩冗余表达**。

#### 3. 关键节点式历史回放

作者让 agent 在每一步额外指出当前页面的 **pivotal nodes**（关键节点）。  
之后只保留这些节点及其：
- ancestors
- siblings
- descendants

于是历史不再是“整页复制”，而是“围绕关键元素的局部结构回放”。

#### 4. 计划树：branch / prune

网页任务常常要尝试多个子路径。  
作者引入两个由 LLM 直接生成的动作：

- `branch [parent] [intent]`：创建新子计划
- `prune [resume] [reason]`：放弃当前失败计划，回到早先计划点

关键点在于：  
**branch/prune 和 click/type 是同层动作，不是额外外接的控制器。**

这使得计划能力来自模型本身，而不是手工写死的 agent workflow。

### 为什么这套设计有效？

- **动作裁剪** 减少了动作空间熵，降低“乱点、乱滚、乱跳 URL”的概率。
- **页面简化** 降低了 observation 中的格式噪声，让 token 更接近自然语言阅读。
- **关键节点回放** 把长期记忆从“全历史缓存”改成“任务相关工作记忆”。
- **计划树裁剪历史** 则把记忆进一步从“全局线性历史”改成“当前子计划作用域历史”。

这不是简单“prompt 工程”，而是在改变模型每一步决策时看到的 **信息分布**。

### 战略权衡

| 设计 | 改变的瓶颈 | 收益 | 代价/风险 |
|---|---|---|---|
| 裁剪非必要/低层动作 | 动作噪声、具身要求高 | 更少误操作与无效探索 | 多 tab、快捷键等细粒度能力被弱化 |
| 禁用 scroll、整页输入 | 可逆滚动循环 | 减少空转，直接看全局 | 当前步 token 可能变长 |
| DOM/树结构转 Markdown | 页面冗余 | 更易读表格、列表与标签 | 可能丢失少量界面结构细节 |
| 关键节点历史回放 | 长历史噪声 | 记忆更聚焦，减少重复步骤 | 关键节点选错会漏信息 |
| branch/prune 计划树 | 缺少轻量恢复机制 | 能切换失败路径，保持子任务一致性 | 依赖模型自身规划质量 |

## Part III：证据与局限

### 关键证据

1. **主结果信号：简单接口对齐 > 更复杂 agent 策略**
   - WebArena 上，AGENTOCCAM 达到 **43.1%**
   - 超过 WebPilot 的 **37.2%**
   - 超过 SteP-replication 的 **33.3%**

   这说明能力跃迁并非一定来自更多角色、更多搜索，而可能来自更合适的接口。

2. **最强因果信号：逐组件消融**
   - vanilla agent：**16.5%**
   - AGENTOCCAM：**43.1%**
   - 绝对提升 **+26.6 点**

   这是论文里最可信的部分，因为它直接回答了：
   **How：到底是哪个 knob 在起作用？**  
   答案是：动作裁剪、页面简化、选择性历史、计划树这几项接口层改动。

3. **跨环境泛化信号**
   - WebVoyager 确定答案子集：**54.3%**
   - 高于 Agent-E 的 **51.9%**

   虽然优势不算巨大，但说明收益不只存在于 WebArena 模拟器里。

4. **互补性信号**
   - AGENTOCCAM + Judge：**45.7%**
   - AGENTOCCAM + SteP：**41.1%**，反而低于 base 43.1%

   这个结果很有意思：  
   **接口对齐与“更好的动作选择”可以互补，但与硬编码任务策略结合时，后者可能伤害泛化。**

### 能力跃迁到底体现在哪里？

相对 prior work，这篇论文的“跳跃”不在于更会搜索，而在于：

- 更少在 reversible actions 上打转
- 更少被 DOM 冗余文本干扰
- 更能围绕当前子计划保持执行一致性
- 更容易在失败后换路径，而不是持续在同一错误思路里循环

这也是它相对前人最强的系统层洞察：  
**先修接口，再谈复杂 agent orchestration。**

### 局限性

- **Fails when**: 多站点长链任务、需要多 tab/视觉线索的网页、或关键节点筛选失误的高密度单页内容下，方法仍容易失效；WebArena 的 multisite 成功率仍只有 14.6%，且在 WebVoyager 的部分站点（如 Apple、Wolfram Alpha）不如 Agent-E。
- **Assumes**: 依赖 text-only DOM/accessibility tree 表示、依赖强闭源模型 API（主要是 GPT-4-Turbo）、依赖一组通用但手工设计的页面压缩/历史筛选规则；主实验还涉及对 WebArena evaluator 的纠错，这会影响与原始公开分数的直接可比性。
- **Not designed for**: 截图驱动 GUI agent、强视觉定位任务、开放式主观问答网页任务、以及需要大量搜索/回滚/试错的不可逆真实网页操作。

### 资源与复现依赖

- **闭源依赖**：主结果依赖 GPT-4-Turbo
- **环境依赖**：真实网站实验受反爬、登录失效、加载超时影响
- **评测依赖**：作者修正了部分 WebArena evaluator，但也用同样条件重跑了核心 baseline
- **开源情况**：代码与数据已公开，复现门槛相对低于搜索式 agent

### 可复用组件

这篇论文最值得复用的不是完整 prompt，而是这几个“操作原语”：

- **action-space-pruning**：删去对 LLM 不友好的低层动作
- **DOM-to-Markdown compaction**：把网页结构改写为高信噪比文本
- **pivotal-node replay**：围绕关键节点做局部历史回放
- **plan-scoped memory filtering**：用 branch/prune 把历史限制在当前子计划作用域

这些模块基本都可插到别的 web agent 上。

## Local PDF reference

![[paperPDFs/Agentic_Reasoning_Applications_Autonomous_Web_Exploration_Research_Agents_Self_evolving_agentic_reasoning/arXiv_2024/2024_AgentOccam_A_Simple_Yet_Strong_Baseline_for_LLM_Based_Web_Agents.pdf]]