---
title: "Open-Ended Instructable Embodied Agents with Memory-Augmented Large Language Models"
venue: EMNLP
year: 2023
tags:
  - Embodied_AI
  - task/embodied-instruction-following
  - task/dialogue-grounded-planning
  - retrieval-augmented-generation
  - external-memory
  - in-context-learning
  - dataset/TEACh
  - opensource/partial
core_operator: "检索外部“语言-程序”记忆作为按需示例，驱动冻结LLM把自由对话、纠错与失败反馈解析成可执行家庭任务程序，并把成功程序写回记忆实现个性化。"
primary_logic: |
  人机对话/指令/纠错 + VLM失败描述 + 当前场景状态 → 检索相似语言-程序与纠错案例作为上下文示例，LLM生成或修正Python动作程序，执行器做前置条件检查、目标搜索与闭环修正 → 完成家庭长程任务并逐步适配用户私有例程
claims:
  - "HELPER在TEACh的TfD unseen上达到13.73% task success和14.17% goal-condition success，相对先前SOTA DANLI分别提升1.7x和2.1x [evidence: comparison]"
  - "将记忆增强检索替换为固定prompt后，TfD unseen task success从13.73%降至11.27%，说明输入条件化示例检索对规划与失败恢复有实质贡献 [evidence: ablation]"
  - "将估计感知替换为GT perception后，TfD unseen task success从13.73%升至30.23%、goal-condition success升至50.46%，表明系统主瓶颈仍在感知而非纯规划 [evidence: ablation]"
related_work_position:
  extends: "LLM-Planner (Song et al. 2022)"
  competes_with: "DANLI (Zhang et al. 2022); JARVIS (Zheng et al. 2022)"
  complementary_to: "Code as Policies (Liang et al. 2022); RT-1 (Brohan et al. 2022)"
evidence_strength: moderate
pdf_ref: paperPDFs/Agentic_Reasoning_Applications_Embodied_Agents_Self_evolving_Agentic_Reasoning/EMNLP_2023/2023_Open_Ended_Instructable_Embodied_Agents_with_Memory_Augmented_Large_Language_Models.pdf
category: Embodied_AI
---

# Open-Ended Instructable Embodied Agents with Memory-Augmented Large Language Models

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2310.15127), [Project](https://helper-agent-llm.github.io/)
> - **Summary**: 这篇论文把具身任务规划从“固定 few-shot prompt”升级为“可检索、可增长的语言-程序记忆”，让冻结 LLM 能从自由对话中生成、纠错并个性化家庭任务程序。
> - **Key Performance**: TEACh TfD unseen 上 SR/GC = 13.73%/14.17%，相对 DANLI 提升 1.7x/2.1x；EDH unseen 上 SR/GC = 17.40%/25.86%。

> [!info] **Agent Summary**
> - **task_path**: 自由形式人机对话/纠错/视觉失败描述 -> Python宏动作程序 -> 家庭环境中的长程任务执行
> - **bottleneck**: 固定few-shot prompt无法覆盖开放域语言、用户私有routine和执行失败恢复
> - **mechanism_delta**: 用检索式外部记忆替代静态prompt，并把成功执行写回记忆；失败时再把视觉错误转成语言继续同一规划回路
> - **evidence_signal**: TEACh TfD unseen上相对DANLI达到1.7x SR和2.1x GC，且去掉记忆增强后SR下降18%
> - **reusable_ops**: [检索式上下文示例组装, 成功轨迹写回记忆]
> - **failure_modes**: [遮挡与深度误差导致定位/交互失败, 多处个性化修改时出现逻辑性计划错误]
> - **open_questions**: [如何做视觉-语言联合记忆检索, 如何把抽象宏动作迁移到真实机器人闭环控制]

## Part I：问题与挑战

**What / Why**：这篇论文要解决的真正问题，不是“LLM 能不能写出一个计划”，而是**在开放域、多轮、带省略和纠错的人机对话里，如何让系统在推理时拿到最相关的任务先例，并把执行失败重新喂回同一个规划器**。

### 任务是什么
TEACh 里的 agent 需要根据多轮自然语言对话，在家庭环境中完成长程任务，比如做早餐、清洗物品、重排物体。这里的语言不是单句命令，而是可能包含：

- 长距离指代与省略
- 用户纠错
- 个体化简称或私有流程
- 执行中才暴露出的失败与环境不确定性

### 真正瓶颈在哪里
作者的判断很准：**固定 prompt 是主瓶颈**。

已有工作已经表明，冻结 LLM 在合适 few-shot prompt 下，可以把指令翻译成程序或高层计划；但一旦进入开放式具身任务，静态 prompt 会遇到三层问题：

1. **覆盖率不够**：prompt 工程时不可能预写所有用户表达、偏好和私有 routine。
2. **上下文代价太高**：继续堆示例会逼近上下文窗口上限，也会增加推理成本。
3. **失败恢复断裂**：执行失败来自视觉/交互层，而规划器主要吃语言，二者接口不统一。

### 输入/输出接口
- **输入**：人机对话、补充指令、用户纠错、VLM 给出的失败描述、RGB 观测。
- **输出**：基于固定 API 的 Python 程序，再由执行器落地成导航与操作动作。

### 边界条件
这项工作明确站在一个较受控的设定上：

- 家庭任务域，主要在 TEACh / AI2-THOR 中评测
- 固定的参数化宏动作 API
- 英文输入
- 依赖模拟器可用的动作失败信号与场景交互接口

所以它解决的是：**开放式对话到可执行程序的适配问题**，而不是端到端真实机器人控制的全部问题。

## Part II：方法与洞察

**How**：HELPER 的核心思路不是再训练一个更大的 embodied model，而是把适配能力外置到一个**可检索、可增长的外部记忆**里，让 LLM 每次都看到“和当前语境最像”的程序先例。

### 方法骨架

#### 1. Planner：检索增强的程序生成
HELPER 维护一个 key-value memory：

- **key**：语言上下文
- **value**：对应的动作程序

给定当前输入 \(I\)（对话片段、指令或纠错），系统先用文本 embedding 编码，再按相似度取 top-K 记忆（文中 K=3），把这些语言-程序对作为 in-context 示例塞进 prompt，最后让冻结 LLM 生成 Python 程序。

这一步的本质是：  
**不是让 LLM 从零“想计划”，而是让它在最相关的少数例子旁边“改写计划”。**

#### 2. Memory Expansion：成功后写回，支持个性化
如果一次执行被用户确认成功，系统把“用户语言 + 成功程序”写回记忆。  
这样下次用户再说一个简称，比如某个自定义清洁 routine，模型就能把它当作检索命中样例，直接复用并局部修改。

这改变的是部署时的数据分布：系统不再只依赖训练前写好的 prompt，而是能逐步吸收用户自己的语言习惯。

#### 3. VLM-Guided Correction：把失败翻译成语言
执行失败时，HELPER 用 ALIGN 将当前图像与一组预定义失败描述做匹配，得到失败反馈文本，再把：

- 原对话
- 失败描述
- 失败子目标
- 相似纠错案例

一起送入 LLM，生成 corrective program。

这一步非常关键，因为它把“视觉失败”转成了“语言可消费”的信号。  
于是初始规划和失败重规划，共享同一套检索+LLM 机制。

#### 4. Executor + Locator：把程序真正落地
作者没有只停在“计划生成”，而是补了一个很实用的执行层：

- 语义/占据地图构建
- 3D 物体检测与状态跟踪
- 动作前置条件检查
- 找不到目标物体时，用 LLM 做常识性搜索位置猜测

尤其是**前置条件检查**很重要。  
例如切东西前必须先拿刀，这种约束不一定总被 LLM 在程序里显式写出，但执行器可以补齐。

### 核心直觉

这篇论文最核心的因果链可以概括成：

- **静态 few-shot prompt → 输入条件化检索 prompt**
- **一次性规划 → 成功后写回的持续记忆**
- **像素级执行失败 → 语言化失败描述**

对应改变了三个瓶颈：

1. **示例分布错位**：当前任务终于能配上最相关的 few-shot 样例，而不是吃一个固定模板。
2. **部署分布漂移**：用户私有表达和 routine 不再只能靠“下次重写 prompt”解决，而是可以在线进入记忆。
3. **模态接口断裂**：视觉错误被翻译成语言后，能直接复用 LLM 的语言规划能力做修复。

为什么这设计有效？因为作者抓住了一个事实：  
**LLM 已经有通用语义和程序生成能力，缺的不是“更懂家庭任务”，而是“在当前时刻看到正确先例”。**

### 战略取舍

| 设计选择 | 改变的瓶颈 | 带来的能力 | 代价 / 风险 |
|---|---|---|---|
| Top-K 语言-程序检索 | 固定prompt与当前输入分布不匹配 | 更稳地解析自由对话与纠错 | 检索错会把 LLM 带偏 |
| 成功程序写回记忆 | 部署后无法学习用户私有routine | 支持个性化简称与流程复用 | 需要可靠成功判定，错误记忆会污染库 |
| VLM失败描述 + 纠错prompt | 执行失败难以接回规划器 | 初始规划与重规划接口统一 | 失败类型受预定义文本集合限制 |
| 前置条件检查 | LLM 计划常漏隐式操作约束 | 降低无效动作，提高执行稳定性 | 需要规则工程，扩展到开放动作空间更难 |
| LLM常识定位器 | 目标物体未观测时搜索盲目 | 用语言常识缩小搜索范围 | 依赖常识正确且依赖感知地图可用 |
| 固定宏动作API | 直接生成低层动作空间过大 | 程序更易生成、验证和执行 | 不等于真实机器人低级控制能力 |

## Part III：证据与局限

**So what**：HELPER 的能力跃迁，不只是“换成 GPT-4 更强”，而是把 LLM 包进了一个**检索—执行—纠错—写回**的闭环里。最有说服力的地方在于：SOTA 提升、关键模块消融、以及“感知才是主瓶颈”的诊断都比较一致。

### 关键实验信号

- **比较信号：新 SOTA**
  - 在 TEACh TfD unseen 上，HELPER 达到 **13.73% SR / 14.17% GC**，相对 DANLI 为 **1.7x / 2.1x**。
  - 在 EDH unseen 上，达到 **17.40% SR / 25.86% GC**。
  - 这说明它不只是能“写出看起来合理的程序”，而是能在长程执行设定下兑现部分收益。

- **因果信号：检索记忆确实在起作用**
  - 去掉 memory augmentation 后，TfD unseen SR 从 13.73 降到 11.27，约 **18% 相对下降**。
  - 去掉 pre-condition check、Locator、VLM correction 也都会掉点。
  - 结论：提升来自系统性闭环设计，而不是单点技巧。

- **模型尺度信号：大模型仍重要，但不是唯一因素**
  - GPT-3.5 替代 GPT-4 后，SR 相对下降约 **31%**。
  - 说明检索框架有效，但底层 LLM 的程序推理能力仍显著影响上限。

- **瓶颈诊断信号：感知是更大的天花板**
  - 用 GT perception 后，TfD unseen 直接到 **30.23% SR / 50.46% GC**。
  - 这比纯规划层的小修小补提升大得多，说明当前系统更受制于 3D 感知、检测融合和对象状态跟踪。

- **交互与个性化信号**
  - 两轮反馈可把 TfD unseen SR 提到 **17.48%**，约 **1.27x** 提升。
  - 个性化实验中，40 个用户私有请求里正确生成 37 个计划。
  - 但需要注意：个性化实验主要是**计划生成正确性**，不是完整 embodied execution；反馈实验中的“用户反馈”也由 simulator metadata 程序化生成，因此真实人类交互下的收益仍偏保守/待验证。

### 局限性

- **Fails when**: 遮挡严重、单目深度误差大、2D 检测不稳或失败类型超出预定义文本集合时，系统容易在定位、交互和失败恢复上出错；多处个性化修改时，LLM 也会出现逻辑性改写错误，甚至误改未要求改动的部分。
- **Assumes**: 依赖 GPT-4-0613 与 text-embedding-ada-002 等专有 API，依赖 ALIGN、SOLQ、ZoeDepth 等外部模型，依赖固定宏动作 API 与 TEACh/AI2-THOR 的模拟器交互；反馈实验还假设能从 simulator metadata 自动生成纠错语句。
- **Not designed for**: 真实机器人低级闭环控制、开放词表新物体/新技能学习、直接基于原始视觉状态做多模态记忆检索。

### 资源与复现注意
虽然论文提供代码/项目页，但核心结果仍绑定专有 LLM/embedding API 与特定模型版本；因此它是“方法开源、关键能力依赖闭源服务”的类型。对复现实验而言，**API 可用性、版本漂移和成本**都是真实约束。

### 可复用组件

- **检索式 in-context prompt 组装**：把静态 few-shot 改为输入条件化 few-shot。
- **成功轨迹写回记忆**：让部署时持续吸收用户私有 routine。
- **语言化失败诊断**：把视觉/执行错误转成统一文本接口，复用同一规划器。
- **前置条件守卫**：在执行层自动补齐 LLM 易漏掉的隐式约束。
- **常识性对象搜索器**：当目标未观测时，用语言常识缩小搜索空间。

一句话总结：**HELPER 的关键贡献不是让 LLM 直接当 policy，而是把 LLM 变成一个有外部记忆、可纠错、可持续适配的“具身程序解释器”；最强证据是 SOTA + 记忆消融 + 感知上限诊断三者一致。**

## Local PDF reference

![[paperPDFs/Agentic_Reasoning_Applications_Embodied_Agents_Self_evolving_Agentic_Reasoning/EMNLP_2023/2023_Open_Ended_Instructable_Embodied_Agents_with_Memory_Augmented_Large_Language_Models.pdf]]