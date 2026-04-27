---
title: "DeCo: Task Decomposition and Skill Composition for Zero-Shot Generalization in Long-Horizon 3D Manipulation"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/robot-manipulation
  - task/long-horizon-manipulation
  - task-decomposition
  - skill-composition
  - vlm-planning
  - dataset/DeCoBench
  - dataset/RLBench
  - opensource/no
core_operator: "基于夹爪-物体物理交互把演示切成可复用原子技能，并用VLM检索排序后通过空间感知链式过渡把技能拼成未见长时程操作。"
primary_logic: |
  高层语言指令 + 场景RGB-D观测 → 按物理交互周期构建原子任务/技能库，并由VLM从库中检索与排序所需技能 → 通过目标位姿监控与空间感知无碰撞过渡依次执行 → 零样本完成未见但可组合的长时程3D操作任务
claims:
  - "在 DeCoBench 的12个未见长时程任务上，RVT-2/3DDA/ARP 接入 DeCo 后平均成功率分别从 0.00%/0.00%/0.14% 提升到 66.67%/21.53%/58.06% [evidence: comparison]"
  - "真实机器人上，仅用6个原子训练任务，RVT-2+DeCo 在9个未见长时程任务上的平均成功率达到 53.33%，而基线 RVT-2 为 0% [evidence: comparison]"
  - "full interaction 分解与空间感知 skill chaining 都是关键因果部件：使用 half interaction 或禁用 chaining 都会显著降低长时程泛化表现 [evidence: ablation]"
related_work_position:
  extends: "VoxPoser (Huang et al. 2023)"
  competes_with: "SCAR (Chen et al. 2024); Points2Plans (Huang et al. 2025)"
  complementary_to: "Code as Policies (Liang et al. 2023); Trust the Proc3s (Curtis et al. 2024)"
evidence_strength: moderate
pdf_ref: paperPDFs/Other_Multimodal_Large_Language_Model_MLLM_based_related_Embodied_Learning_2025/arXiv_2025/2025_DeCo_Task_Decomposition_and_Skill_Composition_for_Zero_Shot_Generalization_in_Long_Horizon_3D_Manipulation.pdf
category: Embodied_AI
---

# DeCo: Task Decomposition and Skill Composition for Zero-Shot Generalization in Long-Horizon 3D Manipulation

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2505.00527) · [Project](https://deco226.github.io/)
> - **Summary**: 这篇论文把长时程操作的难点从“让单个策略隐式学会所有组合”改成“先按物理交互切成原子技能，再由 VLM 显式规划和调度这些技能”，从而让已有多任务模仿学习模型获得零样本组合泛化能力。
> - **Key Performance**: 在 DeCoBench 的 12 个未见长时程任务上，RVT-2/3DDA/ARP 分别从 0.00%/0.00%/0.14% 提升到 66.67%/21.53%/58.06%；真实机器人上 RVT-2+DeCo 在 9 个未见任务上达到 53.33%，基线为 0%。

> [!info] **Agent Summary**
> - **task_path**: RGB-D场景观测 + 长语言指令 / 零样本长时程3D操作 -> 原子技能序列调度 -> 末端位姿与夹爪控制执行
> - **bottleneck**: 长轨迹演示中的技能边界不一致，导致 IL 策略学不到可重组技能；同时高层语义计划与可执行低层技能空间不对齐
> - **mechanism_delta**: 将训练数据重写为按夹爪-物体交互周期切分的原子任务库，并在测试时用 GPT-4o 从库中检索和排序技能，再用空间感知过渡位姿完成无碰撞衔接
> - **evidence_signal**: 基线模型在长时程零样本任务上几乎全失败，而接入 DeCo 后在 DeCoBench 和真实机器人上都出现显著成功率跃升
> - **reusable_ops**: [physical-interaction-based decomposition, goal-pose-based skill scheduling]
> - **failure_modes**: [VLM视觉-语义grounding错误导致过规划, 基座IL模型在未见组合场景下视觉鲁棒性不足]
> - **open_questions**: [如何加入闭环重规划以处理执行中目标位移, 如何扩展到灵巧手与非抓取操作]

## Part I：问题与挑战

这篇论文针对的不是“机器人不会抓取/放置”，而是一个更具体、更难的瓶颈：

**多任务模仿学习模型已经能学会很多原子技能，但不会把这些技能在未见过的长时程任务里正确组合起来。**

### 1. 真正困难在哪
长时程 3D manipulation 的难点主要有三层：

1. **训练数据层**：原始演示通常是长轨迹，里面混着多个交互阶段。  
   同一个任务里既包含“开抽屉”，又包含“放物体”，还可能包含“关抽屉”。对策略来说，技能边界并不显式。

2. **推理层**：面对新指令时，模型需要知道  
   “应该先做什么、再做什么、何时切换到下一步”。  
   现有多任务 IL 模型通常是反应式的，缺少显式任务分解与阶段管理。

3. **执行层**：就算高层 plan 是对的，两个技能之间的空间过渡也可能不连续。  
   例如上一个技能结束位姿和下一个技能起始位姿差得很远，直接切换会撞到抽屉、柜门或进入下一个策略的低置信区域。

### 2. 为什么现在值得解决
论文的判断很清楚：**底层 atomic skill 已经不是主要短板，组合泛化才是。**

从实验也能反推出这一点：RVT-2、3DDA、ARP 在原子任务上成功率都很高，但到了未见长时程任务几乎全掉到 0。说明问题不在“不会做动作”，而在“不会拆、不会接、不会按顺序执行”。

### 3. 输入/输出接口
本文延续语言条件 3D manipulation 的标准接口：

- **输入**：RGB-D 观测 + 自然语言指令
- **输出**：6-DoF 末端位姿 + 1-DoF 夹爪状态

DeCo 不改底层 IL 模型的动作空间，而是在其外部补上：

- 原子任务构建
- 高层技能规划/调度
- 技能间过渡衔接

### 4. 边界条件
这篇方法不是“任意长任务通吃”，它依赖几个前提：

- 新任务必须**可分解为已学原子技能的组合**
- 原子技能的终止可以用**目标位姿匹配**来检测
- 当前分解方式依赖**夹爪开合变化**，更适合 claw-like end-effector
- 方法目标是**零样本组合泛化**，不是在线学习新技能

---

## Part II：方法与洞察

DeCo 的核心思想可以概括为一句话：

**不要让策略自己从长轨迹里隐式发现组合结构，而是先把数据变成“可组合的技能库”，再在测试时显式拼装。**

方法由三部分构成。

### 1. 基于物理交互的任务分解：把“长任务”切成“原子技能”
作者把夹爪与物体的物理交互当作稳定分界点。

- 一个 **full interaction**：open → closed → open  
- 一个 **half interaction**：单次开合变化

主方法默认使用 **full interaction** 作为原子任务粒度。原因是它通常更接近一个完整可执行命令，例如：

- open drawer
- put item in opened drawer
- close drawer

这样处理后，每个原子任务都会配上：

- 一条自然语言 instruction
- 对应 demonstration
- 最终关键帧的 **goal pose**

然后用这些原子数据去训练任意多任务 IL 模型，如 RVT-2、3DDA、ARP。

**关键变化**：训练分布从“长且混杂的轨迹”变成“语义和物理边界一致的短片段”，策略学到的是可复用 skill，而不是某个固定长序列。

### 2. VLM-guided planning and skill scheduling：把长指令映射到技能序列
在测试时，DeCo 把：

- 新语言指令
- 当前视觉输入
- 原子 instruction library

一起喂给 GPT-4o，让它输出一个有序技能序列。

例如：

“Place the block in the closed top drawer, and then close the drawer”

会被映射成：

1. Open the top drawer  
2. Place the block in the top drawer  
3. Close the top drawer

然后 IL 模型逐个执行这些原子技能。

DeCo 本身不生成低层动作，它只负责：

- 从原子库里检索技能
- 管理 instruction pointer
- 用目标位姿监控当前技能是否完成
- 完成后切到下一技能

这点很重要：**DeCo 把长时程问题降成“离散技能选择 + 连续技能执行”两层。**

### 3. Spatially-aware skill chaining：解决“会规划但接不上”的问题
仅有技能序列仍然不够，因为两个技能之间可能出现空间断裂。

论文的做法是：

- 当前技能完成后，拿到其 **goal pose**
- 预测下一技能的 **start pose**
- 再结合场景点云，构建一个**空间感知 cost map**
- 生成 collision-free chaining poses
- 最后用 RRT 做短程过渡规划

作者强调，这不是普通全局 motion planner 的替代，而是一个**policy-aware 的 bridge**：  
它的目标不只是几何可达，还要尽量让过渡状态靠近下一个技能的演示分布，减少策略分布偏移。

### 核心直觉

DeCo 真正拧动的“因果旋钮”有三个：

1. **把训练对象从长轨迹改成交互一致的原子技能**  
   → 改变了训练分布  
   → 减少了一个策略需要同时建模的时间依赖  
   → 提高了技能可复用性

2. **把推理目标从直接跟随长指令改成检索-排序原子技能**  
   → 改变了规划空间  
   → 从连续长序列映射，变成离散技能组合  
   → 解决了未见组合的时序推理问题

3. **把技能切换从“硬切”改成空间感知过渡**  
   → 改变了技能交接时的状态约束  
   → 限制过渡状态既无碰撞又更接近策略支持域  
   → 降低链式执行中的切换失败

更直白地说，DeCo 不是让模型“更聪明”，而是让问题形式**更适合已有策略的能力边界**。

### 战略权衡

| 设计选择 | 改变了什么 | 带来的能力 | 代价/风险 |
|---|---|---|---|
| 按 full interaction 切分原子任务 | 训练数据从混合长轨迹变为物理边界一致的短轨迹 | 技能更可复用，语言对齐更稳定 | 粒度较固定，不易覆盖灵巧手/非抓取操作 |
| 用 VLM 从原子库检索技能序列 | 高层规划空间从动作级降为技能级 | 支持零样本组合新任务 | 受闭源 VLM grounding 错误影响，可能过规划或指令漂移 |
| 用 spatially-aware chaining 做技能衔接 | 技能切换状态受几何与策略兼容性双重约束 | 降低碰撞与分布偏移 | 依赖点云、cost map 和 motion planner，且仍缺少执行中重规划 |

---

## Part III：证据与局限

### 1. 关键证据信号

#### 信号 A：基线在原子任务强，但在长时程组合上几乎全灭
这是全篇最有说服力的实验逻辑。

- 原子任务上，RVT-2 / 3DDA / ARP 平均成功率分别约为 **91.83 / 98.00 / 94.67**
- 但在 12 个未见长时程任务上：
  - RVT-2：**0.00**
  - 3DDA：**0.00**
  - ARP：**0.14**

这说明：  
**底层技能能力并不差，真正缺的是组合机制。**

#### 信号 B：DeCo 显著提升零样本长时程泛化
接入 DeCo 后，在同一个 DeCoBench 上：

- RVT-2 + DeCo：**66.67%**
- 3DDA + DeCo：**21.53%**
- ARP + DeCo：**58.06%**

这不是小幅修补，而是从“几乎不会做”到“能完成相当比例任务”的能力跃迁。  
而且提升出现在三种不同范式的 IL 模型上，支撑了其 **model-agnostic** 说法。

#### 信号 C：数据重写本身是有效的，不只是“加了个 planner”
Table III 很关键。

作者拿 RVT-2 做了对比：

- 直接在 6 个长任务上训练：6 个新任务成功率 **30.00%**
- 用 DeCo 的原子任务训练并组合：6 个新任务成功率 **83.89%**

这说明收益不仅来自推理阶段的 planning，也来自**训练分布被改造成可组合的 atomic skill learning**。

#### 信号 D：真实机器人上同样成立
在 Franka Panda 上，只用 6 个原子训练任务，测试 9 个未见长时程任务：

- RVT-2：**0%**
- RVT-2 + DeCo：**53.33%**

这表明 DeCo 不是只在模拟器里“玩得转”，至少在抽屉类 object rearrangement 场景中有现实可行性。

#### 信号 E：Ablation 支撑了方法因果链
两个 ablation 最说明问题：

1. **full interaction > half interaction**  
   full interaction 切分带来更好的组合泛化，说明“正确的原子粒度”是关键因果因素。

2. **去掉 chaining module 会明显掉性能**  
   论文展示了不做衔接规划时的碰撞失败案例，说明技能切换不是细枝末节，而是长时程执行成败的重要环节。

### 2. 需要同时看到的负面信号
论文也诚实报告了一个现象：  
**DeCo 版本在部分 atomic tasks 上反而比 baseline 低。**

原因主要有两个：

- VLM 的视觉-语义 grounding 错误，导致不必要的前置技能触发
- VLM 生成的 instruction 与训练分布不完全一致，影响 IL policy 的目标位姿估计

这说明 DeCo 带来的高层泛化能力，确实引入了一个新的误差源：**VLM planning quality**。

### 3. 局限性

- **Fails when**: VLM 对场景状态判断出错并发生过规划；目标物体在执行中被大幅移位，导致当前 goal pose 无法匹配；基座 IL 模型在未见组合上下文中的视觉鲁棒性不足（如部分 sweep / retrieve-and-sweep 场景）
- **Assumes**: 新任务可由已知原子技能组合；存在可用的 goal pose 标注和场景点云；调用 GPT-4o 这类闭源 VLM；底层 IL 模型已充分学会 atomic skills；末端执行器是可用开合状态刻画交互的 claw-like gripper
- **Not designed for**: 需要在线发明新技能的任务、灵巧手精细操控、非抓取操作、强扰动下的闭环重规划场景

### 4. 复现与资源依赖
这篇方法的可扩展性受几个系统假设影响较大：

- **闭源依赖**：GPT-4o 用于规划与检索
- **感知/规划依赖**：场景点云、VoxPoser 风格 cost map、RRT 规划器
- **训练资源**：模拟实验使用 **8× RTX 4090**
- **真实数据采集**：需要 kinesthetic teaching 收集原子示范
- **执行判停机制**：依赖 pose matching，而非闭环任务重规划

### 5. 可复用组件
如果你不是要复现整套系统，而是想“拿走几个好用模块”，这篇论文最值得复用的是：

- **物理交互驱动的 demonstration 切分规则**
- **原子 instruction library + VLM 检索式规划**
- **基于 goal pose 的技能完成检测与调度**
- **面向 skill handoff 的 spatially-aware chaining**

---

## Local PDF reference

![[paperPDFs/Other_Multimodal_Large_Language_Model_MLLM_based_related_Embodied_Learning_2025/arXiv_2025/2025_DeCo_Task_Decomposition_and_Skill_Composition_for_Zero_Shot_Generalization_in_Long_Horizon_3D_Manipulation.pdf]]