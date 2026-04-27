---
title: "Code-as-Symbolic-Planner: Foundation Model-Based Robot Planning via Symbolic Code Generation"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/task-and-motion-planning
  - task/multi-robot-planning
  - symbolic-code-generation
  - self-verification
  - multi-round-guidance
  - dataset/BoxNet
  - dataset/Blocksworld
  - dataset/Path-Racecars
  - dataset/Path-Drones
  - opensource/full
core_operator: 让LLM生成并迭代修正可执行的符号规划代码，用代码中的搜索、优化与约束校验替代纯文本规划。
primary_logic: |
  自然语言任务与环境文本 + 预定义动作/轨迹约束
  → TaskLLM生成候选符号代码与计划
  → CheckLLM执行校验代码验证正确性，复杂度检查器判断代码是否真正进行了搜索/数值/组合计算
  → SteerLLM基于错误与复杂度反馈多轮重导
  → 输出满足逻辑、碰撞、时序与资源约束的动作序列或waypoints
claims:
  - "在 7 个 TAMP 任务、3 个 LLM 上，Code-as-Symbolic-Planner 相比最佳基线平均提升 24.1% 成功率 [evidence: comparison]"
  - "在 GPT-4o 上，去掉 CheckLLM 会使平均成功率从 52.3% 降到 42.7%，去掉 Symbolic Complexity Checker 会降到 41.3% [evidence: ablation]"
  - "在 BoxNet 的真实双臂与 3D 多臂实验中，该方法分别达到 100%/100%/95%（2 臂真实 / 3 臂仿真 / 6 臂仿真）平均成功率，并优于非符号化基线 [evidence: comparison]"
related_work_position:
  extends: "Code as Policies (Liang et al. 2022)"
  competes_with: "AutoTAMP (Chen et al. 2024); OpenAI Code Interpreter"
  complementary_to: "PDDLStream (Garrett et al. 2020); ViLD (Gu et al. 2021)"
evidence_strength: strong
pdf_ref: paperPDFs/Other_Multimodal_Large_Language_Model_MLLM_based_related_Embodied_Learning_2025/arXiv_2025/2025_Code_as_Symbolic_Planner_Foundation_Model_Based_Robot_Planning_via_Symbolic_Code_Generation.pdf
category: Embodied_AI
---

# Code-as-Symbolic-Planner: Foundation Model-Based Robot Planning via Symbolic Code Generation

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2503.01700), [Project](https://yongchao98.github.io/Code-Symbol-Planner/)
> - **Summary**: 这篇工作把 LLM 从“直接说出计划”的语言规划器，变成“直接写出搜索器与约束检查器”的符号代码规划器，并通过多轮自引导让代码真正承担 TAMP 中的搜索与验证。
> - **Key Performance**: 7 个任务 × 3 个 LLM 上平均较最佳基线提升 24.1%；GPT-4o 上平均成功率 52.3%，高于 OpenAI Code Interpreter 的 34.3%。

> [!info] **Agent Summary**
> - **task_path**: 自然语言任务与环境描述 -> 满足逻辑/几何/时序约束的动作序列或轨迹 waypoints
> - **bottleneck**: 纯文本 LLM 难以稳定完成组合搜索、数值约束和可行性校验；而直接要求写代码时又常退化成“把答案包进代码”的伪符号化
> - **mechanism_delta**: 用 SteerLLM + CheckLLM + 规则复杂度检查器组成多轮闭环，持续逼迫 TaskLLM 生成真正执行搜索与验证的符号代码
> - **evidence_signal**: 7 任务 3 模型的一致平均提升 + 去掉 CheckLLM/复杂度检查器分别下降 9.6/11.0 个百分点
> - **reusable_ops**: [multi-round self-guidance, code-based plan verification]
> - **failure_modes**: [generated search code times out on hard instances, real-world task formalization into solvable code is brittle]
> - **open_questions**: [how to hybridize generated code with efficient classical planners, how to extend from text states to multimodal sensing]

## Part I：问题与挑战

这篇论文解决的是机器人 **Task and Motion Planning (TAMP)**：既要找出高层动作顺序，又要保证这些动作在几何、时序、碰撞、容量等约束下可执行。

### 真正的问题是什么
作者认为，当前 LLM 做机器人规划主要有两条路线，但都卡在不同瓶颈上：

1. **LLM 直接规划**
   - 例如直接输出子任务、动作序列或 waypoint。
   - 优点是通用、接口自然。
   - 缺点是把组合搜索、数值约束、逻辑一致性都压进 token 级文本推理里，一旦任务复杂，成功率会急剧下降。

2. **LLM 翻译到外部规划器**
   - 例如把自然语言翻成 PDDL、逻辑约束或专用求解器输入。
   - 优点是有真正的符号求解能力。
   - 缺点是每类任务都要专门设计 formalism、planner 和集成框架，泛化性差。

所以，这篇论文抓住的**真实瓶颈**不是“LLM 不会理解任务”，而是：

- LLM 不擅长稳定地做显式搜索与约束满足；
- 但如果直接让它写代码，它又经常写出“看起来像代码、实际上只是硬编码答案”的伪规划器。

### 为什么现在值得解决
因为新一代 foundation model 的代码能力已经足够强，开始具备现场合成小型搜索器、检查器、数值计算程序的潜力。  
也就是说，现在可以尝试把“求解”从模型内部的隐式语言推理，转移到模型生成的**外部可执行程序**里。

### 输入/输出接口与边界条件
本文设定相当明确，边界也很强：

- **输入**：自然语言任务说明、文本化环境状态、机器人能力描述、要求的输出格式
- **输出**：
  - 离散任务：动作序列
  - 连续规划：带时间戳的 waypoint 序列
- **已知条件 / 假设**：
  - LLM 拥有环境和机器人能力的完整文本描述
  - 动作原语是预定义的
  - 运动轨迹由 waypoint 间线性插值得到
  - 需满足速度上限、时间上限、碰撞等约束

这意味着它不是端到端“从原始感知到控制”的系统，而是一个**在文本化世界模型上工作的规划合成器**。

---

## Part II：方法与洞察

核心思想可以概括为一句话：

> **不要让 LLM 直接说计划，而是让它写一个会搜索、会验证、会检查约束的 planner。**

### 方法骨架

系统中同一种 LLM 被提示成三个角色：

1. **TaskLLM**
   - 负责生成求解代码和最终计划。
   - 输出通常包含：文本分析 + Python 代码 + 代码执行后的答案。

2. **SteerLLM**
   - 不直接给答案，而是给“下一轮应该如何改代码”的指导。
   - 它还决定什么时候可以 finalize。

3. **CheckLLM**
   - 生成并执行检查代码，验证当前计划是否正确、是否满足约束。
   - 把检查结果和解释反馈给 SteerLLM。

此外还有一个非 LLM 模块：

4. **Symbolic Complexity Checker**
   - 规则化检测代码里是否真的出现了迭代、搜索、数值运算、排列组合等符号计算痕迹。
   - 给出复杂度分数与解释，帮助 SteerLLM 判断这段代码是不是“真 solver”，而不是把答案包成程序外壳。

整体流程是：

- 先生成一版代码
- 再检查“对不对”
- 再检查“是不是在认真算”
- 如果不行，就给出下一轮更有针对性的引导
- 最多迭代 3 轮

### 核心直觉

#### 1. 改了什么
从“输出计划文本”改成“输出可执行规划代码”。

#### 2. 改变了哪个瓶颈
把原来塞在 LLM 隐式推理里的几个难点外置了：

- **组合搜索**：交给程序里的 DFS / A* / 枚举 /约束检查
- **数值与逻辑验证**：交给执行期代码
- **答案正确性判断**：交给单独 checker
- **防止伪代码**：交给复杂度检查器

也就是把瓶颈从“模型一次采样时能不能想对”，改成“模型能不能写出一个真的会算的程序，并在反馈下逐轮修好”。

#### 3. 带来了什么能力变化
这样做后，模型更擅长处理那些**必须显式搜索或验证**的 TAMP 任务：

- Blocksworld 这种全局组合搜索
- BoxLift 这种容量匹配与协同分配
- Path-Racecars / Path-Drones 这种连续时空约束规划
- 多机器人任务中动作顺序与资源冲突并存的问题

论文最有价值的洞察是：

> **LLM 的价值不一定是“脑内直接规划”，而可能是“现场写一个任务专用小规划器”。**

### 为什么这个设计有效
因果上看，效果提升并不是因为 prompt 更长了，而是因为：

- **代码执行把隐式推理变成显式状态演化**
- **checker 提供了比文本自评更硬的外部反馈**
- **复杂度检查器改变了生成分布，减少“浅层、伪算法式代码”**
- **多轮引导把一次性采样失败，变成可纠偏过程**

这也是为什么论文里强调，很多失败不是“代码写错一行”，而是**根本没有进入符号求解范式**。

### 战略取舍

| 设计选择 | 改变的瓶颈 | 收益 | 代价 |
|---|---|---|---|
| 直接输出计划 → 生成规划代码 | 把搜索/约束满足从文本推理转到程序执行 | 更适合复杂 TAMP | 代码可能超时、出 bug |
| 单轮回答 → 多轮引导 | 把一次性失败改成迭代纠偏 | 稳定性更高 | 延迟与 API 成本更高 |
| 文本自评 → CheckLLM 代码校验 | 提供更可执行、可复现的正确性信号 | 降低幻觉式“看起来合理” | checker 本身也受模型能力限制 |
| 无结构约束 → 复杂度检查器 | 抑制“伪代码” | 更容易逼出真实搜索器 | 规则启发式可能误判代码质量 |
| 通用代码工具 → 面向 TAMP 的定向 steering | 把 generic coding 变成 task-specific symbolic planning | 相比普通 Code Interpreter 更稳定 | 依赖额外框架设计 |

---

## Part III：证据与局限

### 关键证据

#### 1. 比较信号：在“需要显式搜索”的任务上出现能力跳跃
最强证据不是平均数，而是某些任务上的断层式提升。

- **整体**：7 个任务、3 个 LLM 上，平均比最佳基线高 **24.1%**
- **GPT-4o 平均**：Code-as-Symbolic-Planner 为 **52.3%**，OpenAI Code Interpreter 为 **34.3%**
- **Blocksworld（GPT-4o）**：达到 **48.2%**，而 Only Question / Code Answer / Code Interpreter / SayCan / HMAS-2 都接近 0～1.4%

这说明它的优势不是“更会说”，而是**真的在需要搜索的任务里启用了符号计算**。

#### 2. 消融信号：收益来自闭环组件，而不是单纯多写点代码
作者做了关键消融：

- 去掉 **CheckLLM**：平均成功率从 52.3% 降到 **42.7%**
- 去掉 **Symbolic Complexity Checker**：降到 **41.3%**

此外，随着最大迭代轮数从 1 增加到 3，性能明显提升；超过 3 后趋于平台。  
这说明有效性来自“**生成—检查—重导**”闭环，而不是一次性 code prompting。

#### 3. 可扩展性信号：复杂度提升时下降更慢
图 5 表明随着对象数量增多、环境变大，直接文本规划下降最快；Code-as-Symbolic-Planner 的退化更慢。  
这支持论文的核心主张：**把求解写进代码里，比把求解压在语言 token 里更可扩展。**

#### 4. 真实系统信号：不仅停留在 2D toy tasks
在 BoxNet 上：

- 真实双臂：**100%**
- 3 臂仿真：**100%**
- 6 臂仿真：**95%**

并且 3D/真实环境还引入了感知转文本和执行误差后的重规划。  
这说明方法并非只在静态离散 benchmark 上有效。

### 证据的保守解读
虽然总体证据强，但并不是“全场碾压”：

- Claude3-5-Sonnet 在 **BoxNet** 上，Only Question 甚至高于 Code-as-Symbolic-Planner
- GPT-4o 在 **BoxLift** 上，Code Interpreter 略高于本文方法

因此更准确的结论是：

> 该方法尤其适合 **搜索、约束验证、组合优化是主瓶颈** 的任务；对较简单、格式化强或 generic code tool 已足够的任务，优势不一定稳定。

### 局限性

- **Fails when**: 任务组合复杂度过高时，生成的搜索代码可能在 50 秒执行上限内无法完成；或者 LLM 无法把现实任务抽象成适合代码求解的优化/物流问题
- **Assumes**: 环境状态与机器人能力能被完整、正确地文本化；动作原语和答案格式是预定义的；需要可执行代码环境；最强结果依赖 GPT-4o / Claude3.5 等闭源 API；3D/真实实验还依赖 ViLD 做感知到文本转换
- **Not designed for**: 从原始视觉/力觉直接端到端规划控制、未知动作原语发现、严格最优性保证、强部分可观测或高频闭环控制场景

还要补充两个很实际的边界：

1. **输出格式敏感**
   - 作者明确说，计划输出格式会显著影响编码成功率。
   - 说明这套方法对“问题如何被 formalize”仍然很敏感。

2. **代码生成本身引入新错误源**
   - 它绕开了手写 planner 的任务专用性，但也引入了程序错误、低效算法、超时等新问题。

### 可复用组件

这篇工作最值得迁移的不是某个具体 prompt，而是这几个操作子：

- **code-as-solver**：让模型生成“求解器代码”，不是只生成“接口代码”
- **verifier-as-code**：把 plan checking 也外部化成可执行代码
- **symbolic complexity prior**：用启发式判断代码是否真的做了搜索/计算
- **multi-round steering**：把 planner synthesis 做成迭代优化过程

对后续 embodied agent / tool-using LM 工作来说，这些部件都很容易复用。

![[paperPDFs/Other_Multimodal_Large_Language_Model_MLLM_based_related_Embodied_Learning_2025/arXiv_2025/2025_Code_as_Symbolic_Planner_Foundation_Model_Based_Robot_Planning_via_Symbolic_Code_Generation.pdf]]