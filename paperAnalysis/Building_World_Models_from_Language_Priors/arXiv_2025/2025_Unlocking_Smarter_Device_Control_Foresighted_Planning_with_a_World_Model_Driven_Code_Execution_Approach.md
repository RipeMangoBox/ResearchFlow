---
title: "Unlocking Smarter Device Control: Foresighted Planning with a World Model-Driven Code Execution Approach"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/mobile-device-control
  - world-model
  - code-execution
  - hierarchical-planning
  - dataset/MobileAgentBench
  - opensource/no
core_operator: 先用语言先验构建任务相关的文本图世界模型，再在图上生成可执行代码计划，并在执行中持续校验和修正。
primary_logic: |
  任务指令与初始观察 → LLM生成单应用文本有向图世界模型 → VLM基于图与截图生成带条件/循环的Python计划 → 将抽象边映射为GUI动作并通过放大验证执行 → 根据新观察在线修正图和计划，跨应用时递归派生子代理 → 完成设备控制任务
claims:
  - "FPWC在MobileAgentBench上达到39%任务成功率，相比表中最强基线AutoDroid的27%提升44.4% [evidence: comparison]"
  - "在作者构建的真实设备103任务评测中，FPWC取得33.0%成功率和62.8%完成率，均高于AppAgent与MobileAgent [evidence: comparison]"
  - "完整系统相对plain设置将成功率从0.12提升到0.39，且世界模型是最大单项增益来源 [evidence: ablation]"
related_work_position:
  extends: "Mobile-Agent (Wang et al. 2024a)"
  competes_with: "Mobile-Agent (Wang et al. 2024a); AppAgent (Zhang et al. 2023)"
  complementary_to: "Set-of-Mark Prompting (Yang et al. 2023); EXPEL (Zhao et al. 2024)"
evidence_strength: moderate
pdf_ref: paperPDFs/Building_World_Models_from_Language_Priors/arXiv_2025/2025_Unlocking_Smarter_Device_Control_Foresighted_Planning_with_a_World_Model_Driven_Code_Execution_Approach.pdf
category: Embodied_AI
---

# Unlocking Smarter Device Control: Foresighted Planning with a World Model-Driven Code Execution Approach

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2505.16422)
> - **Summary**: 论文把手机控制从“看当前屏幕做下一步”的反应式代理，改成“先构建任务相关世界模型、再写成可执行代码计划并边执行边修正”的前瞻式代理。
> - **Key Performance**: MobileAgentBench 成功率 39%（vs. 27% 最强基线）；真实设备 SR 33.0% / CR 62.8%

> [!info] **Agent Summary**
> - **task_path**: 文本任务 + 初始/当前截图（可辅以XML与局部裁剪） -> 手机GUI动作序列 / 跨App子任务执行
> - **bottleneck**: 单步截图只提供局部界面信息，导致ReAct式代理缺少全局状态转移理解，容易短视、走错分支或陷入回路
> - **mechanism_delta**: 先把单App抽象成文本有向图世界模型，再让VLM生成带条件与循环的代码计划，并在每步执行后同步修图改计划
> - **evidence_signal**: 标准基准比较 + 真实设备评测 + 组件消融，其中MobileAgentBench上39% SR和消融中世界模型带来的最大增益最关键
> - **reusable_ops**: [文本图世界模型, 代码式策略执行, 放大裁剪动作验证]
> - **failure_modes**: [小而歧义的UI元素识别错误, 高token与高延迟带来的实时性不足]
> - **open_questions**: [如何做跨任务持续积累的可复用世界模型, 如何在不爆上下文的情况下实现统一多App全局规划]

## Part I：问题与挑战

这篇论文解决的是**手机设备控制中的长程决策问题**：给定自然语言任务，代理需要跨多个页面、多个状态、甚至多个应用，执行一串 GUI 操作来完成目标。

### 真正的难点是什么？
表面上看，问题像是“看图点按钮”；但作者认为真正瓶颈不是单步点击，而是：

1. **部分可观测**：每一步只看到当前截图，看不到整个 App 的状态空间。
2. **任务是长程的**：很多任务要跨多个页面，早期动作会影响后续可达状态。
3. **现有方法过于反应式**：AppAgent、Mobile-Agent 一类方法大多在 ReAct loop 里基于当前观察做下一步，历史上下文也较粗，因此容易短视。
4. **跨 App 更难**：如 YouTube 订阅前先去 Settings 开 Wi-Fi，这种依赖外部应用状态的任务，单 App 视角很容易失效。

### 输入 / 输出接口
- **输入**：任务指令、当前截图，执行时还会用到 XML / 局部裁剪图等辅助信息。
- **输出**：Tap / Long_press / Swipe / Text / Back 等基础 GUI 动作，或调用跨 App 子代理。
- **中间表征**：不是直接预测像素未来，而是构建一个**文本化、任务导向的 App 状态图**。

### 为什么现在值得做？
因为当前 VLM/LLM 已经具备两种关键先验：
- **语言先验**：能根据任务和常识，大致“脑补”App 页面结构和页面间转换关系；
- **代码先验**：能生成带条件、循环、分支的可执行计划，而不只是自然语言“想法”。

作者的核心判断是：**如果能把任务相关的环境结构先显式写出来，再在这个结构上规划，手机代理就能从局部反应走向全局前瞻。**

### 边界条件
- 世界模型**只显式建一个当前最相关的 App**，避免图过大、prompt 过长。
- 跨 App 任务不是统一建全局大图，而是通过**递归子代理**处理。
- 这不是端到端训练的新模型，而是基于 GPT-4V 等现成 VLM/LLM 的系统式方法。

## Part II：方法与洞察

FPWC 可以概括成三步：**先构图，再写代码计划，再在线修正。**

### 1. 任务导向的文本图世界模型
作者把每个 App 表示成一个**文本有向图**：
- **节点**：页面/视图的语言描述；
- **边**：高层动作导致的状态转移，如“tap Wi-Fi button”。

关键点不在于精确建模概率转移，而在于：
- 先枚举出**任务相关的可能状态与转移**；
- 把不确定性留给后面的条件判断与执行修正。

这和传统世界模型直接预测像素不同，它更像是一个**语言抽象层**：足够粗，便于推理；又足够结构化，便于执行。

### 2. 在世界模型上生成“可执行代码计划”
有了图之后，VLM 不再只输出“下一步做什么”，而是输出一个 Python-like 计划。

其中有几个关键操作：
- `E(vertex, action)`: 执行图上的抽象边；
- `isTRUE(statement)`: 基于当前视觉观察判断条件真假；
- `other_app_agent(app, subtask)`: 跨 App 时创建子代理。

这样，计划就可以自然表达：
- 条件分支：如果 Wi-Fi 没开，就去打开；
- 循环：遍历候选项直到找到满足条件的对象；
- 异常恢复：执行失败后重规划。

这比自然语言 thought 更强，因为**代码把“意图”约束成了结构化控制流**。

### 3. 执行前验证 + 执行后自修正
仅有计划还不够，因为 VLM 对 UI grounding 仍可能不准。作者加了两层闭环：

1. **执行前验证**  
   把候选 UI 元素局部裁剪放大，再问 VLM 这是不是目标元素；若不是，换别的候选。
   
2. **执行后修正**  
   每执行一步，就结合最新截图和上一步动作，更新：
   - 世界模型中的节点/边；
   - 当前尚未执行的剩余计划。

这相当于把“世界模型先验”与“在线现实反馈”接起来，缓解先验图不完整或错误的问题。

### 4. 跨 App 递归代理
为了避免把多个 App 的世界模型塞进同一个超长上下文，作者采用分层方案：
- 父代理负责主任务；
- 如果任务需要其他 App，就调用 `other_app_agent(AppName, Subtask)`；
- 子代理在自己的 App 内重新构图、规划、执行，完成后返回。

这是一种**按需展开的局部多代理规划**，而不是一开始就做全局联合建模。

### 核心直觉

**变化是什么？**  
从“当前截图 -> 下一动作”的反应式策略，改成“任务 -> 文本状态图 -> 可执行控制流 -> 在线修正”的前瞻式策略。

**改变了什么瓶颈？**  
它把原来压在单次观察里的信息瓶颈，转移成一个显式的、可编辑的环境抽象：
- 截图只给局部信息；
- 世界模型补足了**潜在但尚未可见的状态结构**；
- 代码计划补足了**条件/循环/异常恢复**这种长期策略结构；
- 在线修正弥补了先验图和真实环境的偏差。

**能力上带来什么变化？**
- 从“看见什么做什么”变成“根据未来可能状态选择当前动作”；
- 从“thought 与 action 松耦合”变成“意图被代码显式绑定”；
- 从“单次错误导致全局失败”变成“允许中途纠偏与重规划”。

一句话说：**它把手机控制里的长期决策，外化成了一个可推理、可执行、可修补的中间程序。**

### 战略权衡

| 设计选择 | 缓解的瓶颈 | 带来的能力 | 代价 / 风险 |
|---|---|---|---|
| 文本图世界模型 | 当前截图信息不足 | 更好的任务理解与长程状态预判 | 依赖预训练先验，初始图可能漏状态/错状态 |
| 代码式规划 | 自然语言计划不够刚性 | 支持条件、循环、异常处理，执行更可验证 | token 更高，计划生成与执行更慢 |
| 自验证 + 自修正 | 视觉定位误差、先验与现实不一致 | 提高动作精度，允许在线恢复 | 需要更多模型调用，延迟升高 |
| 单 App 图 + 子代理递归 | 多 App 全图过大 | 可扩展到跨 App 任务 | 缺少统一全局优化，跨代理信息共享有限 |

## Part III：证据与局限

### 关键证据

**信号 1：标准基准比较表明它确实提升了任务完成能力。**  
在 MobileAgentBench 上，FPWC 的成功率是 **39%**，高于 AutoDroid 的 27%、MobileAgent 的 26%。这说明“先建模再规划”在单 App 设备控制里，确实优于纯反应式策略。

**信号 2：真实设备上的收益不是只停留在模拟器里。**  
在作者自建的 103 个真实设备任务上，FPWC 达到：
- **SR 33.0%**
- **CR 62.8%**
- **平均步数 11.9**（少于 AppAgent 的 14.5 和 MobileAgent 的 13.2）

这说明它不仅更能完成任务，也更少走冤枉路，尤其对真实世界与跨 App 场景更有帮助。

**信号 3：消融支持“世界模型是主因，不是附加技巧”。**  
从 plain 到完整系统，成功率从 **0.12 -> 0.39**。其中：
- 加世界模型：0.12 -> 0.24
- 加规划：0.24 -> 0.30
- 加自修正：0.30 -> 0.35
- 加自验证：0.35 -> 0.39

这说明最大跃迁来自**显式环境建模**，而不是单纯多加几个 heuristic。

**信号 4：恢复能力有一定证据，但还不算特别强。**  
附录中，自修正大约在 **16%** 的任务中被触发；在“原本会失败的路径”上，恢复率约 **38%（5/13）**。这支持了“在线修正有用”，但也说明它还不是万能补丁。

### 1-2 个最关键指标
- **MobileAgentBench SR = 39%**
- **Real-world SR / CR = 33.0% / 62.8%**

### 局限性

- **Fails when**: 目标 UI 元素很小、外观相似、语义歧义强，或任务需要非常精确的低层控制时，VLM 的定位和理解仍会失误；对长期后果预测不足时，前瞻规划也可能走错大方向。
- **Assumes**: 依赖强 VLM/LLM 先验与闭源 GPT-4V；默认可获取截图，且常常借助 XML、裁剪放大、网格标注等辅助；默认世界模型按单 App 构建，跨 App 通过递归拆解而非统一建模；真实设备评测还依赖 3 位人工专家判分。
- **Not designed for**: 硬实时交互、高频低延迟控制、安全关键操作、统一多 App 全局最优规划，也不面向完全未知 UI 体系上的强泛化学习。

### 资源与可复现性提醒
- 计算开销明显偏高：在 MobileAgentBench 上平均 **2120.45 tokens**、**26.13s latency**，显著高于多种基线。
- 没有看到公开代码/模型发布信息，因此工程复现与系统级验证门槛较高。
- 真实世界 benchmark 是作者自建，虽然有人工评估，但标准化程度仍弱于公开 benchmark。

### 可复用组件
这篇论文最值得复用的不是某个具体 prompt，而是三类系统操作符：

1. **文本图世界模型**：把 GUI 环境压缩为任务相关的状态-转移图；
2. **代码式规划接口**：用可执行控制流承载长期策略，而非只输出下一动作；
3. **执行前验证 + 执行后修正闭环**：在 grounding 不稳定的场景中提高鲁棒性。

对后续 GUI agent、数字环境 agent，甚至部分 embodied agent，这三者都具有较强迁移价值。

## Local PDF reference

![[paperPDFs/Building_World_Models_from_Language_Priors/arXiv_2025/2025_Unlocking_Smarter_Device_Control_Foresighted_Planning_with_a_World_Model_Driven_Code_Execution_Approach.pdf]]