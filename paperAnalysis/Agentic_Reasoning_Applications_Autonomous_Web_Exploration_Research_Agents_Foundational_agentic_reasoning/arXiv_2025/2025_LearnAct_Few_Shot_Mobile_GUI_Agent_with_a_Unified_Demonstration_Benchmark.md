---
title: "LearnAct: Few-Shot Mobile GUI Agent with a Unified Demonstration Benchmark"
venue: arXiv
year: 2025
tags:
  - Others
  - task/mobile-gui-automation
  - demonstration-learning
  - multi-agent
  - retrieval-augmented
  - dataset/LearnGUI
  - opensource/partial
core_operator: "将人类演示轨迹转成可检索的语义步骤知识，并在执行时按任务相似度检索注入到GUI决策中"
primary_logic: |
  用户指令 + 少量同应用人类示范 + 当前截图
  → DemoParser把原始演示转成带步骤语义与记忆的知识库
  → KnowSeeker按指令相似度检索最相关示范
  → ActExecutor结合截图、历史和检索知识生成GUI动作
  → 提升未见长尾移动界面上的任务完成率
claims:
  - "在 LearnGUI-Offline 上，给 Gemini-1.5-Pro 提供 1 个示范即可将 action match accuracy 从 19.3% 提升到 51.7% [evidence: comparison]"
  - "在 LearnGUI-Online 上，LearnAct 将 UI-TARS-7B-SFT 的任务成功率从 18.1% 提升到 32.8%，接近 GPT-4o 的 34.5% [evidence: comparison]"
  - "在 Gemini-1.5-Pro 的 1-shot 消融中，去掉 DemoParser 或 KnowSeeker 会把离线准确率分别降到 40.6% 和 41.6%，说明知识解析与相关检索都不可或缺 [evidence: ablation]"
related_work_position:
  extends: "AppAgent (Zhang et al. 2023)"
  competes_with: "UI-TARS (Qin et al. 2025); Aguvis (Xu et al. 2024)"
  complementary_to: "OmniParser (Lu et al. 2024); SeeClick (Cheng et al. 2024)"
evidence_strength: moderate
pdf_ref: paperPDFs/Agentic_Reasoning_Applications_Autonomous_Web_Exploration_Research_Agents_Foundational_agentic_reasoning/arXiv_2025/2025_LearnAct_Few_Shot_Mobile_GUI_Agent_with_a_Unified_Demonstration_Benchmark.pdf
category: Others
---

# LearnAct: Few-Shot Mobile GUI Agent with a Unified Demonstration Benchmark

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2504.13805), [Project](https://lgy0404.github.io/LearnAct)
> - **Summary**: 这篇论文把“用户示范”从一次性的 few-shot 提示，升级成可解析、可检索、可执行的外部知识，从而显著提升移动 GUI agent 在长尾、个性化场景中的成功率。
> - **Key Performance**: LearnGUI-Offline 上 Gemini-1.5-Pro 的 action match accuracy 从 19.3% 提升到 51.7%（1-shot）；LearnGUI-Online 上 UI-TARS-7B-SFT 的任务成功率从 18.1% 提升到 32.8%。

> [!info] **Agent Summary**
> - **task_path**: 用户指令 + 同应用少量人类示范 + 当前手机截图 -> GUI动作序列 / 任务完成
> - **bottleneck**: 预训练/微调无法覆盖海量 app 与个体化流程，导致未见界面上的程序性操作知识缺失
> - **mechanism_delta**: 先把原始示范语义化为“步骤知识+记忆”，再按任务相似度检索并注入执行器，而不是让模型只靠零样本从截图现推动作
> - **evidence_signal**: 1-shot 即让 Gemini-1.5-Pro 离线准确率 19.3%→51.7%，且去掉 DemoParser/KnowSeeker 都会掉约 10 个点以上
> - **reusable_ops**: [语义动作描述生成, 指令级相似检索]
> - **failure_modes**: [跨应用迁移未验证, 低质量或错配示范会误导执行]
> - **open_questions**: [多条示范如何抽象成规则而非仅检索实例, agent 能否把成功执行回写知识库形成自学习]

## Part I：问题与挑战

这篇论文要解决的不是“模型会不会点按钮”这么简单，而是**移动 GUI agent 在真实世界里的长尾泛化与个性化适配问题**。

### 1. 真正的难点是什么
现有 mobile GUI agent 大致有两条路：

- 依赖通用大模型的零样本/提示能力；
- 用大量 GUI 数据做微调。

但移动端的问题是，**app 数量极多、界面变化快、用户任务高度个性化**。因此单靠更大预训练或更大微调集，很难覆盖：
- 未见 app；
- 同 app 下未见流程；
- 用户自定义参数任务；
- 结构相似但细节不同的重复任务。

论文把这个问题概括成一种更本质的瓶颈：**参数化模型缺少“面向特定 app/特定用户流程的程序先验”**。  
也就是说，模型不是不会理解截图，而是**不知道这个应用里“通常该怎么做”**。

### 2. 为什么现在值得解决
因为 mobile agent 已经从研究 demo 走向可部署阶段，真正卡住落地的不是平均能力，而是**长尾失败率**。而很多真实任务恰好具备一个很适合 few-shot 学习的结构：

- 用户会重复做；
- 同一 app 内流程有稳定模式；
- 只是参数、目标对象、页面细节在变化。

这意味着，**少量人类示范比继续扩大通用训练集更现实**。

### 3. 输入/输出接口与边界
论文的任务接口很清晰：

- **输入**：
  - 用户自然语言指令
  - 当前手机截图
  - 1/2/3 条同应用的人类示范轨迹
- **输出**：
  - 统一动作空间中的一步步 GUI 操作  
    （CLICK / TYPE / SWIPE / PRESS_BACK / PRESS_HOME / PRESS_ENTER / TASK_COMPLETE）

论文也明确了边界条件：

- 主要研究 **within-application** few-shot learning，不强调跨应用迁移；
- 执行侧是 **screenshot-only**，不依赖 UI tree；
- 离线评测看 step-level accuracy，在线评测看 end-to-end success rate。

这也是它和很多“更大模型、更强 grounding”工作不同的地方：  
**它关注的是如何把用户示范转成可用知识，而不是单纯增强视觉理解。**

## Part II：方法与洞察

LearnAct 的设计哲学很直接：

> 不再追求“把所有 app 都提前学会”，而是让 agent 在面对新场景时，能从少量人类示范里快速获得可迁移知识。

为此，论文同时做了两件事：

1. 提出 **LearnGUI**：第一个专门研究 mobile GUI demonstration learning 的统一 benchmark；
2. 提出 **LearnAct**：一个三阶段、多 agent 的演示学习框架。

### 方法结构

#### 1) DemoParser：把原始轨迹变成“可迁移知识”
输入是：
- 指令
- 截图序列
- 原始动作序列（如 CLICK[123,456]）

输出不是原始坐标，而是**语义化动作描述**，例如：
- “On Search Page, click the search box, to enter keywords”

核心价值在于，它把低层动作转成了两类更容易迁移的信息：

- **动作语义**：点了什么、目的是什么；
- **过程记忆**：中间看到但后续还要用的信息，如价格、湿度、时间、事件标题。

这一步本质上是在做：  
**从“坐标级演示”到“程序性解释”的表示变换。**

#### 2) KnowSeeker：从知识库里找最相关示范
它用 instruction embedding 做相似度检索，从知识库里取 top-k 相关演示。

这里的关键不是检索本身，而是它解决了一个实际问题：  
**few-shot 不是示范越多越好，而是示范越相关越好。**

如果把所有示范都塞进 prompt，会带来：
- 上下文污染；
- 任务错配；
- 推理负担上升。

KnowSeeker 用指令语义做过滤，相当于给执行器一个“先看哪些例子”的门控机制。

#### 3) ActExecutor：把“当前截图”与“检索到的经验”拼起来执行
ActExecutor 在每一步都联合使用：

- 当前任务指令；
- 当前截图；
- 历史动作；
- 检索到的示范知识。

然后输出下一步 GUI 动作。

因此它不是简单模仿某条示范，而是做一种**受示范约束的情境化决策**：
- 当前截图负责 grounding；
- 示范知识负责提供流程先验；
- 历史动作负责防止走回头路。

### 核心直觉

### 核心直觉

**变化点**：把 few-shot 示例从“prompt 里的原始轨迹”变成“可检索的语义程序记忆”。  
**改变的瓶颈**：从参数模型内部硬记 GUI 长尾流程，变成外部知识库按需提供任务相关先验。  
**带来的能力变化**：模型在未见但相似的 app 场景里，不必从零搜索动作，而是沿着示范暴露出的流程模板去执行。

更具体地说，这个设计之所以有效，原因有三层：

1. **语义化**  
   原始 CLICK[x,y] 不可迁移；  
   “在设备详情页点击空调图标以打开空调”是可迁移的。

2. **检索化**  
   不是盲目增加上下文，而是只拿和当前任务最像的示范，减少噪声。

3. **记忆化**  
   很多 GUI 任务失败不是不会点，而是**中途看到了关键信息却没保留**。  
   DemoParser 的 memory 机制正好补这个洞。

一句话概括它的 causal knob：

> **用外部、语义化、可检索的演示知识替代“纯靠模型参数临场推理”的执行方式。**

### 战略权衡

| 设计选择 | 带来的好处 | 代价 / 风险 |
|---|---|---|
| 用少量人类示范替代更大规模通用训练 | 更适合个性化、长尾场景 | 需要额外收集高质量示范 |
| DemoParser 将轨迹语义化 | 把低层点击变成可迁移流程知识 | 解析错误会把噪声写入知识库 |
| KnowSeeker 只检索相关示范 | 控制上下文长度，减少错配 | 仅按指令检索，可能忽略当前 UI 细节 |
| screenshot-only 执行 | 更接近真实部署，少依赖结构化树 | 视觉 grounding 难度高于有 AXTree 的设置 |
| 同应用 few-shot 学习设定 | 贴近日常重复任务 | 跨应用迁移能力暂未回答 |

## Part III：证据与局限

### 关键证据

- **comparison / 离线信号**  
  在 LearnGUI-Offline 上，Gemini-1.5-Pro 从 19.3% 提升到 51.7%（1-shot），说明示范知识对通用大模型不是“锦上添花”，而是能明显改变行为模式。  
  同时，已经很强的专用模型也继续提升：UI-TARS-7B-SFT 从 77.5% 到 82.8%，Qwen2-VL-7B 从 71.8% 到 77.3%。

- **comparison / 在线信号**  
  在线环境里，UI-TARS-7B-SFT 从 18.1% 提升到 32.8%，Qwen2-VL-7B 从 9.9% 提升到 21.1%。  
  最关键的意义不是绝对数值，而是：**7B 级模型在示范增强后，已经接近 GPT-4o 的 34.5%**。这说明能力增益并不完全依赖更大模型参数。

- **ablation / 因果信号**  
  去掉 DemoParser：51.7% → 40.6%  
  去掉 KnowSeeker：51.7% → 41.6%  
  说明这不是“随便加几个示范就行”，而是必须同时具备：
  - 把示范变成结构化知识；
  - 把相关示范挑出来。

- **analysis / 机制信号**  
  相似度分析表明，UI 相似通常更利于迁移，但 **action similarity 有时可以弥补 UI 差异**。  
  这恰好支持论文的核心假设：LearnAct 学到的不只是页面外观对齐，更是**操作模式**。

### 1-2 个最值得记住的指标
- **Offline**: Gemini-1.5-Pro，19.3% → 51.7%（1-shot）
- **Online**: UI-TARS-7B-SFT，18.1% → 32.8%

### 局限性

- **Fails when**: 支持示范与查询任务在 UI 结构和动作模式上都低相似；跨应用迁移；需要更复杂系统交互或非常长链条规划的任务，当前在线成功率仍只有 32.8%，说明真实闭环执行依然脆弱。
- **Assumes**: 有同应用的高质量人工示范；指令相似度足以找到相关示范；视觉模型能稳定解析截图；开放模型训练依赖 8×NVIDIA L40S，商业模型结果还依赖闭源 API。
- **Not designed for**: 无示范的通用零样本手机自动化；跨应用知识迁移评测；从大量历史轨迹中自动归纳抽象程序规则。

### 可复用组件

这篇论文最可复用的不只是完整框架，而是几个独立 operator：

1. **演示轨迹语义化**：把低层点击序列转成步骤语义；
2. **过程记忆标注**：显式保留执行中观察到的关键变量；
3. **示范检索门控**：用轻量 embedding 检索控制上下文；
4. **统一 few-shot benchmark 设计**：把 instruction/UI/action similarity 拆开分析。

如果后续有人做更强 GUI grounding、更强 action model 或自学习 agent，这几块都能直接拼上去。

## Local PDF reference

![[paperPDFs/Agentic_Reasoning_Applications_Autonomous_Web_Exploration_Research_Agents_Foundational_agentic_reasoning/arXiv_2025/2025_LearnAct_Few_Shot_Mobile_GUI_Agent_with_a_Unified_Demonstration_Benchmark.pdf]]