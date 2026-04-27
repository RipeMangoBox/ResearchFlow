---
title: "Safety Aware Task Planning via Large Language Models in Robotics"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/robot-task-planning
  - multi-llm-collaboration
  - llm-as-a-judge
  - control-barrier-function
  - dataset/COHERENT
  - opensource/no
core_operator: 通过“任务规划LLM+安全规划LLM”的迭代批注回路生成更安全的多机器人计划，并将语言安全约束下推为CBF控制约束
primary_logic: |
  自然语言任务目标 + 机器人能力/环境观测 + 安全规则 → 任务LLM生成初始多机器人子任务序列，安全LLM识别冲突/遗漏并反馈修订，解析器把任务与安全描述转成控制目标和CBF约束 → 安全感知的长时程执行计划与在线安全控制
claims:
  - "Claim 1: 在 COHERENT 的 5 个场景中，SAFER+GPT-4o 的平均安全违规数均低于非安全基线 GPT-4o（如 S3: 51.5 -> 16.8，S5: 39.3 -> 19.5），且平均步骤数未明显恶化 [evidence: comparison]"
  - "Claim 2: 论文将 SAFER 的额外推理开销归因于每步新增的 2 次 API 调用，图3显示延迟和成本增幅有限 [evidence: analysis]"
  - "Claim 3: 在两台移动操作臂与人类共同在场的真实硬件实验中，系统完成了协同搬运圆柱体并放入盒子的任务，同时维持人与静态物体相关的安全约束 [evidence: case-study]"
related_work_position:
  extends: "COHERENT (Liu et al. 2024)"
  competes_with: "Safe Planner (Li et al. 2024); COHERENT (Liu et al. 2024)"
  complementary_to: "AHA (Duan et al. 2024)"
evidence_strength: moderate
pdf_ref: paperPDFs/Other_Multimodal_Large_Language_Model_MLLM_based_related_Embodied_Learning_2025/arXiv_2025/2025_Safety_Aware_Task_Planning_via_Large_Language_Models_in_Robotics.pdf
category: Embodied_AI
---

# Safety Aware Task Planning via Large Language Models in Robotics

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2503.15707)
> - **Summary**: 这篇文章把“任务规划”和“安全审查”拆成两个协作式 LLM 角色，并把语言层的安全要求进一步解析成 CBF 控制约束，从而在长时程多机器人任务中减少安全违规而不过度牺牲效率。
> - **Key Performance**: 在 COHERENT 上，SAFER+GPT-4o 在 5 个场景都降低了平均安全违规数（如 S3 从 51.5 降到 16.8）；SAFER+DeepSeek-r1 的 ASV 更低（5.8–12.8），但这部分增益同时混入了更强基座模型因素。

> [!info] **Agent Summary**
> - **task_path**: 自然语言长时程多机器人任务 + 机器人能力/环境观测 -> 安全任务序列 + 控制目标/安全约束
> - **bottleneck**: 单个 LLM 需要同时记住任务历史、机器人能力和安全规则，context budget 与目标偏好都会把安全压到次要位置
> - **mechanism_delta**: 用独立 Safety Planning LLM 对 Task Planning LLM 进行迭代式安全批注，并通过 parser+CBF 把语言安全规则落到执行控制层
> - **evidence_signal**: COHERENT 五场景中 ASV 一致下降，且有双移动操作臂真实部署验证闭环可行
> - **reusable_ops**: [任务LLM-安全LLM批注回路, 语言约束到CBF约束解析]
> - **failure_modes**: [未被15条规则覆盖的新型风险无法被judge稳定发现, 状态跟踪或文本解析错误会导致安全约束接地错误]
> - **open_questions**: [LLM-as-a-Judge 与人工安全审查的一致性有多高, SAFER 本身收益与更强基座模型收益如何解耦]

## Part I：问题与挑战

这篇文章解决的核心问题不是“LLM 能不能做任务规划”，而是：

**当 LLM 被用于长时程、多机器人、有人在场的任务时，如何让它不再只追求完成任务，而是把安全真正当成一等约束。**

### 1) 真正的瓶颈是什么

作者指出的真实瓶颈有两层：

1. **规划层瓶颈：单 LLM 的安全优先级不够高**  
   在一个 prompt 里同时塞入任务目标、历史步骤、机器人能力、环境状态和安全规则时，LLM 往往更偏向“尽快把任务做完”，而不是“系统性检查风险”。  
   典型问题包括：
   - 空间冲突：一个机器人进入另一个机器人仍在工作的区域
   - 动作依赖遗漏：出现 `[grab]` 但缺 `[place]`
   - 交接不稳定：handoff 时序或位置不安全
   - 人机距离问题：人在近旁时仍执行高风险动作

2. **执行层瓶颈：文本计划的“安全”不等于控制层真的安全**  
   即使高层计划写得比较谨慎，如果底层控制没有硬约束，真实机器人仍可能因为人靠近、障碍物位置变化、机械臂/底盘运动冲突而出问题。

### 2) 为什么现在值得解决

因为 LLM 规划已经开始进入：
- 长时程任务分解
- 多机器人协作
- 人机共域环境

这类场景里，**一次规划错误的代价不是答错题，而是碰撞、夹伤、掉落、系统停机**。  
所以作者的动机很明确：**LLM 的“会规划”已经不够，必须进一步变成“会安全地规划”。**

### 3) 输入/输出接口

**输入：**
- 高层自然语言任务目标
- 机器人能力描述与动作库
- 环境观测
- 安全规则
- 执行阶段的成功/失败反馈

**输出：**
- 按机器人分解的动作序列
- 安全反馈修正后的任务计划
- 可下推到控制器的目标与安全约束
- 执行期的在线安全控制

### 4) 边界条件

这不是一个从原始视觉端到端学控制的系统，它更像一个**语言规划-解析-控制闭环框架**。  
它默认：
- 机器人能力是已知的
- 动作集合相对结构化
- 物体/桌面/人位置能被追踪
- 安全规则能被写成可解析的文本约束

也就是说，它主要解决的是**“安全-aware task planning + execution grounding”**，不是开放世界端到端 embodied learning。

---

## Part II：方法与洞察

SAFER 的关键不是“给同一个 LLM 加更多安全提示词”，而是**把安全变成独立角色、独立回路、独立控制约束**。

### 方法骨架

整个框架有四个核心模块：

1. **Planning Module**
   - **Task Planning LLM**：先生成任务计划
   - **Safety Planning LLM**：审查该计划，指出潜在风险
   - 然后 Task LLM 根据安全反馈再次修订计划

   这形成一个最关键的回路：  
   **初始计划 -> 安全批注 -> 修订计划**

2. **Execution Module**
   - 每个机器人有自己的执行能力和动作列表
   - 机器人侧的 Execution LLM 先判断某个子任务是否可执行
   - 若失败，则触发反馈与重规划

3. **Feedback Module**
   - 成功反馈：推进到下一阶段
   - 失败反馈：将“为什么失败”发回规划模块，触发修订

4. **LLM-as-a-Judge**
   - 使用独立 LLM 按 15 类风险标准评估计划
   - 输出安全违规数，而不是只做模糊的“安全/不安全”判断

此外还有两个关键 parser：
- **Planning Parser**：把语言动作转成控制目标，如底盘位姿、末端位姿、夹爪开闭
- **Safety Parser**：把语言安全规则转成静态/动态 CBF 约束

### 核心直觉

这篇文章真正改动的“因果旋钮”是：

**把“任务完成”和“安全审查”从同一个 token budget 里解耦，并把语言级安全要求继续下推到控制级硬约束。**

具体因果链可以写成：

**单 LLM 隐式兼顾任务与安全**  
→ 安全规则和任务历史争抢上下文，安全往往被弱化  
→ 长时程场景更容易漏掉前置条件、时序依赖和空间冲突

变成

**Task LLM 负责完成任务，Safety LLM 专门做外部安全批注**  
→ 安全不再只是 prompt 中的一段提醒，而是独立审查信号  
→ 任务计划更少出现漏步骤、危险并发、错误交接

再进一步：

**把安全文本解析成 CBF 约束**  
→ 安全从“语言建议”变成“控制层可执行边界”  
→ 当人靠近、机器人接近障碍或进入危险区域时，控制器可以实时最小修正动作，而不是只能事后补救

#### 为什么这套设计有效

因为它同时改变了两个瓶颈：

- **信息瓶颈**：安全规则不再和长历史任务竞争同一个 prompt 空间
- **约束瓶颈**：高层语言安全要求不再停留在文本层，而是变成控制器可执行的显式约束

作者还明确回答了一个问题：**为什么不用 RAG/FSL？**  
他们的观点是，RAG 和 few-shot 仍然要把检索到的信息塞回一个中心 LLM 的 prompt，本质上仍没解决单模型上下文拥挤的问题；而多 LLM 分工则是直接在系统层面拆解职责。

### 策略权衡

| 设计选择 | 解决的瓶颈 | 带来的能力 | 代价/风险 |
|---|---|---|---|
| 任务LLM + 安全LLM 双回路 | 单模型同时追求效率与安全时，安全优先级过低 | 更少的时序遗漏、空间冲突和不安全 handoff | 每步要多调用模型，成本与延迟上升 |
| LLM-as-a-Judge | 缺少统一、可解释的安全指标 | 可以按规则统计违规数，便于横向比较与诊断 | judge 质量依赖提示词与规则覆盖，缺少外部校准时可能漂移 |
| 语言到CBF约束解析 | 语言安全无法直接约束连续控制 | 执行阶段可在线守住人机距离、避障、工作空间安全 | 依赖精确状态估计、人工定义约束、实时求解器 |
| 反馈驱动重规划 | 静态计划无法应对执行失败 | 失败后可修正计划，增强长时程鲁棒性 | 链路更长，系统工程复杂度上升 |

---

## Part III：证据与局限

### 关键证据信号

#### 1) 比较信号：安全违规数显著下降
作者在 **COHERENT** 上评估了 40 个任务、5 个场景，指标主要看：
- **Average Steps (AS)**：效率
- **Average Safety Violations (ASV)**：安全违规数

最关键的信号不是“步数更短”，而是：

- **SAFER+GPT-4o 在五个场景里都降低了 ASV**
- 例如：
  - S3：**51.5 -> 16.8**
  - S5：**39.3 -> 19.5**

这说明双 LLM 的安全批注回路确实在减少明确的规划级安全错误。

#### 2) 代价信号：效率没有灾难性变差
如果一个安全系统只是靠“极度保守”来减少错误，那意义有限。  
这里作者想证明的是：**安全提升不必然等于效率崩塌**。

从表格看，SAFER+GPT-4o 的平均步骤数在多数场景没有明显恶化，部分场景还更低。  
这支持它的核心卖点：**安全约束被嵌入规划后，不一定要靠显著拉长流程来换。**

#### 3) 跨模型信号：更强推理模型可能进一步放大安全收益
在 SAFER 框架下，DeepSeek-r1 的 ASV 比 SAFER+GPT-4o 更低。  
这提示了一个很实用的结论：

- **框架负责把安全做成显式回路**
- **更强的基座推理能力负责把这些回路真正用好**

但这里必须保守解读：  
论文没有给出 **non-SAFER DeepSeek-r1** 对照，因此**不能把所有更低的 ASV 都归因于 SAFER 本身**。

#### 4) 硬件信号：语言计划到控制约束的闭环真的跑通了
真实部署使用了两台移动操作臂、人类在场、Vicon 跟踪、动态/静态约束。  
系统完成了“搬运圆柱体并放入盒子”的协作任务，并在过程中处理：
- 人员靠近
- 桌面/障碍避碰
- 两机器人工作区协调
- 安全距离或减速要求

这至少证明了：**它不是只在文本层“看起来安全”，而是能落到机器人控制层执行。**

### 证据强度判断

我会把这篇论文的证据强度定为 **moderate**，原因是：

- 有标准 benchmark 对比
- 有真实硬件案例
- 但没有系统性 ablation 去拆分：
  - Safety LLM 的独立贡献
  - LLM-as-a-Judge 的可靠性
  - Feedback module 的独立价值
  - CBF 与无 CBF 的对照收益
- 评估也主要集中在 **一个 benchmark（COHERENT）**
- 受 API 成本影响，只评了 40 个任务
- “judge 公平性更强”这类观察没有用人工标注做严格验证

### 局限性

- **Fails when**: 环境状态估计不准、对象/人位姿跟踪错误、自然语言解析器把动作或约束接地错了、或者风险类型超出 15 条规则覆盖范围时，系统可能同时在规划层和控制层失效。
- **Assumes**: 已知机器人动作库与能力描述；能稳定获得物体/桌面/用户位姿；安全规则可被人工写成可解析文本；LLM API 可用且延迟可接受；CBF/QP 求解在实时控制中可运行。
- **Not designed for**: 从原始视觉输入端到端学习安全策略；开放词汇未知动作发现；大规模多人高动态环境中的全局最优协同调度。

### 资源与复现依赖

这篇工作有几个很实在的依赖项，会直接影响复现：

- **模型依赖**：使用 API 型 LLM，且开销影响了 benchmark 规模
- **系统依赖**：真实实验依赖 Vicon、marker、tracking helmet 等外部定位系统
- **工程依赖**：需要手工 parser，把语言计划映射成控制目标与 barrier 约束
- **控制依赖**：需要可实时运行的 CBF 安全控制封装
- **开源状态**：文中未给出代码/项目链接，因此复现门槛不低

### 可复用组件

这篇文章最值得复用的不是某个具体 prompt，而是下面这些系统级操作符：

1. **Planner-Critic 分工**  
   用一个 LLM 负责“把任务做成”，另一个 LLM 专门负责“找安全漏洞”。

2. **语言到约束的双解析链**  
   - 任务文本 -> 控制目标  
   - 安全文本 -> 几何/动力学约束

3. **反馈驱动重规划**  
   把成功/失败执行反馈回灌到规划模块，适合长时程 embodied pipeline。

4. **安全控制包裹层**  
   用 CBF 这类“最小侵入式”安全层包裹 nominal controller，而不是完全重写底层控制器。

---

## Local PDF reference

![[paperPDFs/Other_Multimodal_Large_Language_Model_MLLM_based_related_Embodied_Learning_2025/arXiv_2025/2025_Safety_Aware_Task_Planning_via_Large_Language_Models_in_Robotics.pdf]]