---
title: "VLM Agents Generate Their Own Memories: Distilling Experience into Embodied Programs of Thought"
venue: NeurIPS
year: 2024
tags:
  - Embodied_AI
  - task/embodied-instruction-following
  - task/web-navigation
  - retrieval-augmented-generation
  - human-in-the-loop
  - program-of-thought
  - dataset/TEACh
  - dataset/VisualWebArena
  - dataset/Ego4D
  - opensource/no
core_operator: "让VLM把带噪演示重写为含因果、状态变化与子目标注释的优化轨迹，并通过执行期人类反馈把这些抽象沉淀为可检索记忆。"
primary_logic: |
  带噪轨迹/视频 + 指令 + 历史成功样例 → VLM自反思纠错并生成程序化思维抽象 → 执行中接收自然语言反馈继续修订 → 成功后存入记忆库 → 通过检索增强或监督微调提升新任务执行
claims:
  - "Claim 1: 在 TEACh unseen validation 上，ICAL retrieval 达到 35.1% task success / 49.3% goal-condition success，超过 HELPER 手写样例的 34.5% / 36.7% [evidence: comparison]"
  - "Claim 2: 移除 programs-of-thought 生成阶段或 human-in-the-loop 阶段都会显著降低 TEACh 表现（分别降至 29.4%/44.9% 与 29.9%/41.0%），说明两部分都对性能有因果贡献 [evidence: ablation]"
  - "Claim 3: 随着记忆库增长，ICAL 在 TEACh 中后半程学习每个样例所需环境交互从 436±88 降到 267±43、人类反馈从 0.74±0.17 降到 0.21±0.08，显示其能提升后续学习效率 [evidence: analysis]"
related_work_position:
  extends: "HELPER (Sarch et al. 2023)"
  competes_with: "HELPER (Sarch et al. 2023); GPT-4V + Set-of-Marks (Koh et al. 2024)"
  complementary_to: "LoRA (Hu et al. 2021); Self-Consistency (Wang et al. 2022)"
evidence_strength: strong
pdf_ref: "paperPDFs/Agentic_Reasoning_Applications_Autonomous_Web_Exploration_Research_Agents_Self_evolving_agentic_reasoning/arXiv_2024/2024_VLM_Agents_Generate_Their_Own_Memories_Distilling_Experience_into_Embodied_Programs_of_Thought.pdf"
category: Embodied_AI
---

# VLM Agents Generate Their Own Memories: Distilling Experience into Embodied Programs of Thought

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2406.14596), [Project](https://ical-learning.github.io)
> - **Summary**: 论文提出 ICAL，让 VLM 不再直接记忆原始示范，而是把次优轨迹蒸馏成带因果、子目标、状态变化和关键视觉线索的“具身程序化思维”，再用人类反馈在线修正，最终形成更可迁移的记忆库。
> - **Key Performance**: TEACh unseen 上最高达到 **54.2% GC**（SFT+retrieval）并较 HELPER 手写样例高 **17.5 个点**；VisualWebArena 上 GPT-4V 从 **14.3% 提升到 22.7%**。

> [!info] **Agent Summary**
> - **task_path**: 带噪视觉/动作轨迹 + 指令 -> 优化后的动作程序与抽象记忆 -> 新环境中的动作执行/网页操作
> - **bottleneck**: few-shot VLM agent 缺少高质量、可迁移的示例；原始 demo 充满冗余动作、错误标签和未显式化的因果/状态知识
> - **mechanism_delta**: 把“存原始轨迹”改成“VLM 先抽象纠错、再经人类执行验证后存成程序化思维记忆”
> - **evidence_signal**: 跨 TEACh / VisualWebArena / Ego4D 的比较结果 + TEACh 上对 abstraction phase 与 HITL 的组件消融
> - **reusable_ops**: [带噪轨迹抽象, 多模态记忆检索]
> - **failure_modes**: [误导性演示或反馈会污染抽象, VLM 视觉 grounding 弱时高层思维也难纠正执行错误]
> - **open_questions**: [如何摆脱固定 action API, 如何降低对闭源 VLM 与人工反馈的依赖]

## Part I：问题与挑战

这篇论文真正要解决的，不是“VLM 会不会推理”，而是：

**VLM agent 的 few-shot 能力，往往卡在示例质量，而不是卡在模型本身。**

### 1) 难点是什么
现有做法常把成功轨迹、动作计划或人类手写提示直接塞进上下文。但对交互式任务，这些示例有三个根本问题：

1. **原始演示太脏**  
   人类非专家会绕路、漏步骤、做多余动作；从视频反推动作还会有标签噪声；agent 自己收集的轨迹也常包含探索和失败。

2. **示例只记“做了什么”，不记“为什么这么做”**  
   仅保存动作序列，缺少：
   - 因果关系
   - 物体状态变化
   - 子目标分解
   - 当前任务真正该关注的视觉元素

3. **专家手写 prompt 不可扩展**  
   TEACh 中 prior work 需要专家为每个示例写几十到上百行文本，这很难扩展到新环境、新网页、新偏好。

### 2) 输入/输出接口
ICAL 的学习接口可以概括为：

- **输入**：自然语言指令 + 带噪轨迹（观测/动作或纯视频）+ 历史成功记忆 + 执行时人类自然语言纠错
- **输出**：一个成功且可迁移的记忆样例  
  其中包含：
  - 优化后的轨迹
  - 程序化思维注释（programs of thought）

部署接口则是：

- **输入**：新指令 + 当前视觉观测/文本状态 + 检索到的记忆
- **输出**：下一步动作、动作程序，或在 Ego4D 中的未来动作预测

### 3) 真正的瓶颈
这篇论文的判断很明确：

> **瓶颈不是缺更多 demo，而是缺“可压缩进上下文且能跨环境复用”的经验表示。**

为什么现在值得做？因为当前 GPT-4V 这类 VLM 已经足够强，能：
- 看图/看视频理解轨迹
- 基于已有记忆做类比抽象
- 吃进自然语言反馈修正动作与推理

所以现在可以把“人工写经验”改成“模型生成并验证经验”。

### 4) 边界条件
ICAL 不是无条件通吃，它默认：
- 任务仍处于某个**相近任务族**里
- 环境支持**重试/回滚/复位**
- 动作空间由**固定 action API**定义
- 至少有少量初始带噪演示可用
- 某些场景还需要人类自然语言反馈

---

## Part II：方法与洞察

ICAL 的核心不是再训练一个新策略，而是**先把经验“写对”，再把写对的经验拿去检索或微调。**

### 方法主线

#### 阶段 A：VLM 驱动的轨迹抽象（abstraction phase）
给定一条带噪轨迹，VLM 做两件事：

1. **纠正轨迹**
   - 删除低效或错误动作
   - 补齐缺漏步骤
   - 生成更优的动作序列

2. **生成 4 类抽象**
   - **任务/因果抽象**：哪些动作是必要的，为什么
   - **状态变化**：某个动作会让物体状态如何变化
   - **任务分解/子目标**：长任务拆成哪些阶段
   - **任务相关状态/视觉元素**：当前该盯住哪些对象、属性、界面元素

这一步的产物，不再是“原始 trajectory”，而是一个**优化轨迹 + 高层解释**的记忆单元。

#### 阶段 B：人类在环验证（human-in-the-loop phase）
抽象后的轨迹会被拿到环境里执行。

- 如果执行失败，人类给一句自然语言反馈
- VLM 不仅修动作，还会**同步修正原有抽象**
- 成功后，这个样例才会被写入记忆库

这一步的意义是：  
**把纯语言自省，变成经过环境约束校验的经验。**

#### 阶段 C：部署
记忆库建好后，ICAL 有两种用法：

1. **RAG 式部署**
   - 按 instruction / textual state / visual state 的相似度检索 top-k 记忆
   - 把这些记忆塞进 prompt，指导当前动作生成

2. **SFT 式部署**
   - 把记忆样例转成训练对
   - 微调模型，让这些抽象直接内化到参数中

此外，一个很关键的设计是：

> 记忆不仅在推理时检索，**在学习新记忆时也检索**。

这让 ICAL 形成一种自举：  
已有好记忆会帮助生成下一条更好的记忆。

### 核心直觉

ICAL 改变的不是 decoder 结构，而是**上下文里经验的统计形态**。

从：
- 原始、局部、场景绑定的低层动作序列

变成：
- 经过压缩的、带因果与状态转移信息的高密度经验单元

这带来三个因果变化：

1. **噪声分布变了**  
   原始 demo 里的无关动作、错误标签、局部探索被过滤掉，检索到的示例更“干净”。

2. **信息瓶颈变了**  
   上下文窗口有限，不能塞太多轨迹；加入子目标、状态变化、关键视觉元素后，同样 k 个样例能覆盖更多任务变体。

3. **纠错粒度变了**  
   人类反馈不再只修一个动作，而是能写入“为什么失败”的规则，于是后续类似任务也受益。

一句话概括：
> **ICAL 把 memory 从“事件回放”升级成“可迁移的任务程序”。**

### 为什么这个设计有效
因为交互式任务需要的不是死记动作，而是一个隐式世界模型：
- 什么对象重要
- 什么状态会改变
- 哪些步骤是先决条件
- 什么失败具有可推广的因果解释

ICAL 通过程序化思维把这些中间知识显式化，所以它比“直接模仿 raw demo”更容易泛化。

### 战略取舍

| 设计选择 | 改变了什么瓶颈 | 带来的能力 | 代价/风险 |
|---|---|---|---|
| 用 VLM 抽象带噪轨迹 | 降低示例噪声与冗余 | few-shot 泛化更强 | 依赖强 VLM 的自省质量 |
| 人类自然语言反馈验证 | 把抽象落地到真实执行 | 纠错更可靠、记忆更可用 | 需要人工与环境重置成本 |
| 显式记忆库 + 多模态检索 | 把经验复用到学习和推理两侧 | 学得越多，后续学习越快 | 检索策略和上下文长度受限 |
| 固定 action API | 约束输出空间、便于修正 | 提升执行稳定性 | 难以应对新技能/新原语 |
| 同时支持 RAG 与 SFT | 兼顾外部记忆和参数内化 | 可叠加提升性能 | 训练和部署链路更复杂 |

---

## Part III：证据与局限

### 关键证据

#### 1) 比较实验：不是“多存一点 demo”，而是“把 demo 写成好记忆”
在 TEACh unseen 上，ICAL retrieval 达到 **35.1% SR / 49.3% GC**：
- 明显高于 raw visual demos 的 **17.2% / 26.6%**
- 高于 raw kinesthetic demos 的 **26.5% / 29.5%**
- GC 还超过 HELPER 手写样例的 **36.7%**

这个信号说明：  
**经验的表达质量，比经验是否原汁原味更重要。**

如果再做 SFT + retrieval，GC 进一步到 **54.2%**，说明这些抽象不只是 prompt 材料，也能转成可学习监督。

#### 2) 消融实验：抽象和 HITL 都不是装饰件
TEACh 上：
- 去掉 programs-of-thought phase：降到 **29.4% / 44.9%**
- 去掉 human-in-the-loop：降到 **29.9% / 41.0%**

结论很直接：
- 只纠动作、不显式抽象，不够
- 只靠模型自省、不做环境验证，也不够

#### 3) 跨域证据：同一 memory recipe 能迁移到网页与视频
在 VisualWebArena：
- GPT-4V 从 **14.3%** 提升到 **22.7%**
- GPT-4o 从 **18.9%** 提升到 **23.4%**
- Qwen2-VL-7B 经 ICAL SFT 从 **2.9%** 到 **8.2%**

在 Ego4D：
- 优于 few-shot GPT-4V CoT
- 在远少于监督方法的数据条件下保持接近

这说明 ICAL 学到的不是 TEACh 特定 hack，而是一种**把经验重写成高价值上下文**的通用 recipe。

#### 4) 学习效率分析：记忆库会反过来帮助生成新记忆
随着记忆库增长，TEACh 中后半程样例学习所需：
- 环境步数：**436±88 -> 267±43**
- 人类反馈：**0.74±0.17 -> 0.21±0.08**

这点很关键。  
ICAL 不只是“存更多记忆”，而是让**已有记忆改善后续记忆生成**，有明显的自举特性。

### 1-2 个最关键指标
- **TEACh unseen GC：54.2%**（SFT + retrieval，较 HELPER 手写样例 +17.5 个点）
- **VisualWebArena GPT-4V 平均成功率：22.7%**（较 14.3% 提升 1.6x）

### 局限性
- **Fails when**: 演示本身极度误导、反馈错误或相互矛盾时，ICAL 可能把坏经验抽象成坏规则；当 VLM 的视觉 grounding 很弱、界面元素或物体状态识别失败时，高层 programs of thought 也难以补救；面对需要全新动作原语的动态环境时，固定 API 会成为硬约束。
- **Assumes**: 依赖闭源 GPT-4V / GPT-3.5 / OpenAI embeddings / Azure 接口；需要环境可执行与可重试；需要一定量初始示范；核心收益在可提供自然语言反馈或至少可离线抽象的场景更明显。
- **Not designed for**: 低层连续控制、开放式技能发现、完全无人工参与的在线自主学习、以及无法复位验证的真实高风险场景。

### 可复用组件
这篇论文最值得复用的，不是某个 benchmark trick，而是以下操作原语：

- **带噪轨迹 -> 优化轨迹 + 抽象注释** 的蒸馏模板
- **失败点自然语言反馈 -> 同步修动作与思维** 的更新环
- **instruction + textual state + visual state** 的多模态记忆检索
- **学习时也检索记忆** 的自举式经验增长机制

### 总结判断
ICAL 的能力跃迁点，不在于更强的 planner，而在于它重新定义了 agent memory 的单位：

- 以前：一段轨迹
- 现在：一段**被验证过的、可迁移的程序化经验**

这使得 few-shot agent 从“靠 prompt 记住例子”走向“靠 memory 学会抽象”。

![[paperPDFs/Agentic_Reasoning_Applications_Autonomous_Web_Exploration_Research_Agents_Self_evolving_agentic_reasoning/arXiv_2024/2024_VLM_Agents_Generate_Their_Own_Memories_Distilling_Experience_into_Embodied_Programs_of_Thought.pdf]]