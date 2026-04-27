---
title: "Mobile-Agent-v2: Mobile Device Operation Assistant with Effective Navigation via Multi-Agent Collaboration"
venue: ACL 2024
year: 2024
tags:
  - Embodied_AI
  - task/mobile-device-operation
  - multi-agent-collaboration
  - short-term-memory
  - reflection-agent
  - dataset/Mobile-Agent-v2-Dynamic-Eval
  - opensource/partial
core_operator: "把长程手机 UI 操作拆成规划、决策、反思三类代理协作，并用文本化任务进度与短期记忆缓解长上下文导航。"
primary_logic: |
  用户指令 + 当前截图/视觉感知结果 + 历史操作 →
  规划代理将长图文历史压缩成纯文本任务进度，决策代理结合进度/短期记忆/当前页面生成单步动作并更新焦点内容，反思代理根据操作前后页面判断是否回退或重试 →
  更稳定的多步移动设备操作
claims:
  - "在真实设备的中英双场景动态评测中，Mobile-Agent-v2 相比单代理 Mobile-Agent 在系统应用、外部应用与多 App 任务上均提升成功率与完成率，例如英语高级外部应用任务 SR 从 3/10 提升到 7/10 [evidence: comparison]"
  - "规划代理是最关键组件：去掉 planning agent 后 advanced SR 从 61.4% 降到 29.5%，降幅大于去掉 reflection agent 或 memory unit [evidence: ablation]"
  - "直接把 GPT-4V 当作端到端手机操作助手几乎不可行（单步 basic/advanced SR&DA 为 2.7/0.9），而嵌入 Mobile-Agent-v2 框架后 GPT-4V 可达到 92.7/83.5 [evidence: comparison]"
related_work_position:
  extends: "Mobile-Agent (Wang et al. 2024)"
  competes_with: "Mobile-Agent (Wang et al. 2024); AppAgent (Zhang et al. 2023a)"
  complementary_to: "SeeClick (Cheng et al. 2024)"
evidence_strength: moderate
pdf_ref: paperPDFs/Agentic_Reasoning_Applications_Autonomous_Web_Exploration_Research_Agents_Collective_multi_agent_reasoning/ACL_2024/2024_Mobile_Agent_v2_Mobile_Device_Operation_Assistant_with_Effective_Navigation_via_Multi_Agent_Collaboration.pdf
category: Embodied_AI
---

# Mobile-Agent-v2: Mobile Device Operation Assistant with Effective Navigation via Multi-Agent Collaboration

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2406.01014), [Code](https://github.com/X-PLUG/MobileAgent)
> - **Summary**: 论文把手机操作助手中的“长程历史导航”拆成进度规划、焦点内容记忆和操作后反思三个子问题，用多代理协作显著提升了真实设备上的多步 UI 操作成功率。
> - **Key Performance**: 合并中英场景后，advanced 指令 SR 从 27.3% 提升到 61.4%；消融中移除 planning agent 会把 advanced SR 从 61.4% 拉低到 29.5%。

> [!info] **Agent Summary**
> - **task_path**: 用户自然语言指令 + 当前手机截图/感知结果 + 历史操作 -> 单步 UI 动作序列 -> 任务完成/停止
> - **bottleneck**: 单代理难以在长图文交错历史中同时追踪任务进度与跨页面焦点内容，导致后期导航和纠错失效
> - **mechanism_delta**: 用 planning agent 生成纯文本任务进度，decision agent 只看当前界面+记忆做动作，reflection agent 用前后屏幕闭环纠错
> - **evidence_signal**: 真实设备双 OS / 双语言动态评测显著优于 Mobile-Agent，且 planning/memory/reflection 均有消融支持
> - **reusable_ops**: [history-to-text progress compression, before-after screen reflection]
> - **failure_modes**: [高级多步任务在界面语义歧义或感知错误时仍会失败, 焦点内容未被正确写入 memory 时跨 App 信息会丢失]
> - **open_questions**: [如何减少对 GPT-4/GPT-4V 和外部感知工具的闭源依赖, 如何把人工操作知识注入自动化为可扩展训练信号]

## Part I：问题与挑战

这篇论文解决的是**真实手机上的多步 UI 操作**：给定一条自然语言指令，助手需要在多个页面之间连续点击、滑动、输入，直到完成任务。

### 真正的问题是什么
作者指出，单代理手机助手的核心瓶颈不是“不会看图”本身，而是**不会在长操作链里稳定导航**。具体有两个导航难点：

1. **任务进度导航**  
   代理需要从历史操作里判断“哪些子任务已经完成、下一步该做什么”。

2. **焦点内容导航**  
   某些后续操作依赖前面页面里出现过的信息，比如天气、比赛结果、聊天内容等；这些信息不在当前屏幕上，但又必须被后续步骤引用。

### 为什么单代理会失效
现有方案把历史截图、OCR 文本、操作记录全塞进一个代理里。随着步骤增长，输入会变成**超长、图文交错的上下文**。这会带来两个直接后果：

- **上下文过长**：后期任务时，模型更难抓住真正相关的历史状态。
- **模态混杂**：既要读当前屏幕，又要回忆多轮历史图像与文本，决策负担过重。
- **错误不可恢复**：一旦点错、空点或进入无关页面，单代理缺少稳定的“事后检查”机制。

### 输入/输出接口与边界
- **输入**：用户指令、当前手机截图、OCR/图标识别结果、历史操作
- **输出**：受限动作空间中的一步操作  
  `Open app / Tap / Swipe / Type / Home / Stop`
- **环境**：真实设备动态执行，Android/HarmonyOS，含系统 App、外部 App、多 App 任务
- **不依赖**：不像 AppAgent 那样依赖 XML/UI 文件；这让方法更通用，但也更依赖视觉感知和长程控制

**一句话概括 Part I**：  
论文要解决的真瓶颈，是**长程多步手机操作中的“历史压缩 + 焦点记忆 + 错误纠正”**，而不是单步视觉 grounding 本身。

## Part II：方法与洞察

Mobile-Agent-v2 的核心设计不是再堆一个更强的 MLLM，而是把单代理中的三个耦合能力拆开：

- **Planning Agent**：负责“我已经做到了哪一步”
- **Decision Agent**：负责“当前屏幕下一步怎么操作”
- **Reflection Agent**：负责“刚才这步到底做对没有”

同时再配两个辅助模块：

- **Visual Perception Module**：OCR + icon detection + icon description
- **Memory Unit**：保存历史页面中的任务相关焦点内容

### 方法结构

#### 1. Visual Perception Module
作者没有完全相信端到端 MLLM 的屏幕感知能力，而是先用工具把截图转成更结构化的输入：

- 文本识别
- 图标识别
- 图标描述
- 坐标信息

这相当于先把“屏幕像素”变成“可操作对象列表”，降低决策代理的感知负担。

#### 2. Planning Agent：把长历史变成短文本进度
规划代理读取：

- 用户指令
- 上一步操作
- 上一轮任务进度
- memory 中的焦点内容

然后输出一个纯文本的 **Completed contents / task progress**。

关键点在于：它不再把完整历史截图继续传给决策代理，而是把历史压缩成“已完成了什么”。  
这一步本质上是在做**长程轨迹状态压缩**。

#### 3. Memory Unit：外置焦点内容
仅有 task progress 还不够，因为很多任务依赖历史页面里的具体信息，而这些信息不一定适合被压成抽象进度。

因此作者单独设计了 **memory unit**，由 decision agent 在操作过程中更新，例如：

- 未读消息内容
- 比赛结果
- 天气信息
- 需要后续引用的实体名或文本片段

这使得“历史屏幕里的关键内容”从原始图像里被外置成短期显式记忆。

#### 4. Decision Agent：只看当前屏 + 进度 + 记忆
决策代理的输入是：

- 用户指令
- 当前 task progress
- memory 中的焦点内容
- 上一步反思结果
- 当前截图及其结构化感知结果

然后只输出**一步动作**。  
这里最关键的变化是：它**不再直接吞完整历史图文轨迹**，而是只消费已经被整理过的控制状态。

#### 5. Reflection Agent：把开放环控制变成闭环控制
反思代理比较**操作前后两张屏幕**，判断结果属于：

- **Correct operation**
- **Erroneous operation**：进入错误页面，需要回退
- **Ineffective operation**：操作无效，没有页面变化

并决定是否把当前动作写入历史。  
这一步的价值在于：错误和无效操作不会污染后续规划历史，避免代理在错误路径上越走越远。

### 核心直觉

**what changed**：  
从“单代理直接读完整图文历史并决策”，改成“历史先被压成文本进度，关键内容外置成 memory，错误由独立反思器事后判定”。

**which bottleneck changed**：  
原本的瓶颈是**长图文混合上下文**带来的注意力稀释与状态混淆；改造后，决策代理面对的是一个更短、更结构化、更角色对齐的控制状态。

**what capability changed**：  
模型不再那么容易在任务后半段失去进度感，也更能跨页面、跨 App 调用前文信息，并在误操作后恢复。

**为什么这在因果上有效**：
- **Planning** 解决“历史太长导致看不清现在做到哪”
- **Memory** 解决“历史页面中的具体内容会被压缩丢失”
- **Reflection** 解决“单步预测错了之后没有恢复机制”

这三者不是简单堆模块，而是分别对准了三种不同的信息瓶颈。

### 策略权衡

| 设计选择 | 改变了什么瓶颈 | 带来的能力收益 | 代价/风险 |
|---|---|---|---|
| 任务进度文本化 | 长历史图文混杂 | 后期多步任务更稳，减少决策上下文长度 | 摘要可能丢掉细粒度事实 |
| memory unit | 历史屏幕信息不可直接复用 | 支持跨页面/跨 App 调用焦点内容 | 记忆写错会污染后续决策 |
| before-after reflection | 错误操作不可恢复 | 可回退、可重试、避免错误历史累积 | 增加一次 MLLM 调用与时延 |
| 受限动作空间 | 动作搜索空间过大 | 更易稳定执行真实设备操作 | 不覆盖复杂手势和系统级高级动作 |
| 工具化视觉感知 | 端到端屏幕识别不稳 | 更好的对象定位与文本读取 | 依赖 OCR/检测/图标描述工具质量 |

## Part III：证据与局限

### 关键证据

#### 1. 对比实验：提升主要来自“长程任务”而不是简单任务
最强信号不是 basic task，而是**advanced 和多 App**任务的提升：

- 合并中英场景后，advanced SR 从 **27.3% → 61.4%**
- 英语高级外部 App 任务从 **3/10 → 7/10**
- 非英语高级外部 App 任务从 **1/10 → 5/10**

这说明它的增益确实来自**更好的长程导航与跨页信息利用**，而不只是单步视觉操作更准。

#### 2. 消融：planning agent 是最大因果旋钮
去掉 planning agent 后：

- advanced SR 从 **61.4% 降到 29.5%**

而去掉 reflection 或 memory 也会下降，但没有这么剧烈。  
这直接支持作者主张：**长序列中的任务进度导航**是第一瓶颈。

#### 3. 序列位置分析：单代理越到后面越容易崩
作者分析失败轨迹中错误/无效操作出现的位置，发现：

- **Mobile-Agent** 的错误明显集中在任务后段
- **Mobile-Agent-v2** 没有这么明显的后期崩溃模式

这个分析很重要，因为它说明方法不是随机提升，而是**专门缓解了长序列退化**。

#### 4. 模型对比：架构收益大于“直接换强模型”
作者还测试了直接用 GPT-4V 端到端操作手机，结果几乎不可用。  
说明这篇论文的主要贡献不是“用了 GPT-4V”，而是把它放进了一个更合理的**代理控制架构**里。

#### 5. 知识注入：说明操作知识仍是短板
手工加入操作提示后，性能还能继续升。  
这表明系统仍缺少稳定的**App-specific procedural knowledge**。不过要注意：论文只对原本失败的样本注入知识，所以这更像**上界分析**，不是严格同分布对比。

### 1-2 个最值得记住的指标
- **Advanced 总体 SR**：27.3% → 61.4%
- **无 planning agent 的 advanced SR**：61.4% → 29.5%

### 局限性
- **Fails when**: 任务需要强 app-specific 隐性操作知识、界面语义含糊、OCR/图标检测出错，或关键焦点内容没有被正确写入 memory 时，系统仍会在高级多步任务上失败。
- **Assumes**: 依赖 GPT-4/GPT-4V 官方 API、OCR + GroundingDINO + Qwen-VL-Int4 等外部工具、真实设备 ADB 控制接口，以及固定的六动作操作空间；知识注入实验还依赖人工编写提示。
- **Not designed for**: 复杂手势、后台系统设置流、权限弹窗链路、需要底层 UI 树访问的精细控制，或完全端侧/完全开源可复现部署。

### 复用价值
这篇论文最可迁移的不是某个具体手机代理，而是下面几个系统操作符：

- **历史轨迹 → 文本进度摘要**
- **历史页面焦点信息 → 外置短期记忆**
- **操作前后屏幕比对 → 反思/回退机制**
- **受限动作空间 + 工具化感知 → 稳定 GUI 控制接口**

如果你要做网页代理、桌面代理、RPA 或多步 GUI agent，这几个模块都可以直接复用。

![[paperPDFs/Agentic_Reasoning_Applications_Autonomous_Web_Exploration_Research_Agents_Collective_multi_agent_reasoning/ACL_2024/2024_Mobile_Agent_v2_Mobile_Device_Operation_Assistant_with_Effective_Navigation_via_Multi_Agent_Collaboration.pdf]]