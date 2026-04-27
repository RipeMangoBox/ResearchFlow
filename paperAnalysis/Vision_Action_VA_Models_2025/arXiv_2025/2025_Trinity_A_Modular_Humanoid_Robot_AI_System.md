---
title: "Trinity: A Modular Humanoid Robot AI System"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/humanoid-loco-manipulation
  - task/robot-task-planning
  - reinforcement-learning
  - adversarial-motion-priors
  - modular-hierarchy
  - opensource/no
core_operator: 以模块化分层架构把VLM感知、LLM技能规划与RL全身控制串联到全尺寸人形机器人上
primary_logic: |
  自然语言指令 + RGB-D观测 + 机器人运动学/工作空间/安全约束 → VLM定位可操作部件并恢复3D目标，LLM结合技能库生成可执行技能序列，RL在上肢扰动下维持站立/行走平衡并执行下肢控制 → 完成开门等人形机器人全身loco-manipulation任务
claims:
  - "在全尺寸'Tien Kung'人形机器人上，Trinity可完成开门这类需要感知、规划与全身平衡协同的任务，并在受门体反作用力干扰时通过后撤步恢复稳定 [evidence: case-study]"
  - "将上肢动作从下肢RL策略中解耦并在训练中视作扰动后，机器人能在快速摆臂、躯干升降和双臂持物前后移动20 cm时保持站立稳定 [evidence: case-study]"
  - "把安全约束显式注入LLM任务规划后，系统会拒绝执行'抓刀伤人'类危险指令，而单独的VLM感知模块仍会给出可抓取区域 [evidence: case-study]"
related_work_position:
  extends: "Embodied AI with Two Arms: Zero-shot Learning, Safety and Modularity (Varley et al. 2024)"
  competes_with: "Autonomous behavior planning for humanoid loco-manipulation through grounded language model (Wang et al. 2024); Bi-VLA (Gbagbe et al. 2024)"
  complementary_to: "AMP (Peng et al. 2021); Kinematic-aware Prompting (Xia et al. 2024)"
evidence_strength: weak
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Trinity_A_Modular_Humanoid_Robot_AI_System.pdf
category: Embodied_AI
---

# Trinity: A Modular Humanoid Robot AI System

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2503.08338)
> - **Summary**: 这篇论文提出一个面向全尺寸人形机器人的模块化AI系统，用 VLM 做可操作目标感知、用 LLM 做受约束的技能规划、用 RL 做抗上肢扰动的全身平衡控制，从而把“能理解任务”和“真能在实体人形机器人上做出来”接起来。
> - **Key Performance**: 双臂持物前后移动 **20 cm** 仍能稳定站立；真实机器人完成门把手抓取与开门，并在受扰时自动后撤步维持平衡。

> [!info] **Agent Summary**
> - **task_path**: 自然语言指令 + RGB-D场景 -> 技能序列规划 -> 人形机器人全身loco-manipulation执行
> - **bottleneck**: 高层语义规划与低层全身动态平衡之间缺少可部署、可解释、可控的接口
> - **mechanism_delta**: 用VLM把目标压缩为可操作3D锚点，用LLM在技能库/安全/运动学约束下生成动作链，再让RL只负责在上肢扰动下维持下肢稳定
> - **evidence_signal**: 全尺寸“Tien Kung”真实机器人上的开门、快速摆臂与持物平衡案例
> - **reusable_ops**: [上肢扰动随机化RL训练, 约束式技能库LLM规划]
> - **failure_modes**: [VLM定位或深度估计错误导致抓取点偏移, 超出技能库的长时接触任务无法可靠规划]
> - **open_questions**: [模块误差与时延如何做闭环补偿, 安全约束如何从prompt升级为可验证的控制安全机制]

## Part I：问题与挑战

这篇工作要解决的，不只是“让机器人听懂话”，也不是单独“让人形机器人能走路”，而是更难的一件事：

**如何把语言理解、视觉感知和全身动态控制接成一个真实可部署的人形机器人系统。**

### 真正瓶颈是什么？

作者认为现有机器人大模型工作多数建立在一个前提上：**底层控制器已经很好**。  
但对人形机器人，这个前提不成立。

核心困难有三层：

1. **人形机器人本体太复杂**
   - 上肢一动，重心就变；
   - 与门把手、物体接触时会产生外力；
   - 仅靠语言规划或视觉 grounding，并不能保证机器人不摔倒。

2. **高层智能与低层控制是断开的**
   - LLM/VLM擅长“理解任务、规划步骤”；
   - RL/控制器擅长“维持平衡、执行动作”；
   - 但二者之间缺一个稳定接口，否则高层一发命令，低层可能根本扛不住。

3. **安全约束不能只靠单个模块**
   - 视觉模块可能只会回答“哪里能抓”；
   - 但真正系统要回答的是“该不该抓”。

### 输入/输出接口

- **输入**：
  - 人类自然语言任务指令
  - RGB-D 图像
  - 机器人工作空间、技能库、运动学知识、安全约束
- **输出**：
  - 一串可执行技能序列
  - 由底层控制器在真实人形机器人上完成的动作执行

### 为什么现在值得做？

因为三类能力刚好成熟到可拼接：

- **RL** 已能在仿真中学到较稳的人形 locomotion；
- **LLM** 已能做长程任务分解；
- **VLM** 已能把语言和视觉目标对齐。

所以现在的关键问题不再是“单模块是否存在”，而是：

> 能否把这些能力以一种**模块化、可解释、真实可跑**的方式整合到全尺寸人形机器人上。

---

## Part II：方法与洞察

Trinity 的设计不是端到端，而是**分层模块化系统**。  
其思想很直接：**不要让一个模型同时学会“看懂世界、理解任务、生成计划、控制全身”**，而是把问题拆开，让每个模块只处理自己最擅长的瓶颈。

### 系统结构

#### 1. RL 模块：负责下肢平衡与 locomotion

底层控制不是全身端到端，而是聚焦于：

- 站立 / 行走稳定
- 对上肢运动与外力扰动的鲁棒性
- 在 manipulation 过程中保持重心可控

关键做法：

- 用 **AMP (Adversarial Motion Priors)** 让步态更像人，而不是死板跟踪关节；
- 用 **periodic reward + command reward + regularization reward** 训练可切换站立/步行的策略；
- **把手臂动作从 locomotion policy 中分离**，并在训练时把上肢关节力矩随机化，等价于让策略提前见过“上肢乱动带来的扰动”；
- 用 **FSM** 在站立与行走 gait 之间切换。

这一步的关键不是“让机器人走得更快”，而是：

> 让机器人在执行高层 manipulation 命令时，底层还能稳住。

#### 2. VLM 模块：负责找到“该操作哪里”

系统使用 **ManipVQA** 风格的视觉语言感知：

- 从 RGB-D 图像中识别物体的可操作部分；
- 输出 bounding box；
- 结合深度恢复该可操作部位的 **3D 位置**。

这一步把开放视觉问题压缩为一个对机器人有用的中间表示：

- “门在哪里”不够；
- 更重要的是“**门把手的可操作位置在哪里**”。

#### 3. LLM 模块：负责把任务变成技能链

作者用 **GPT-4** 作为任务规划器。  
它并不直接输出连续控制，而是输出一串离散技能：

- arm skill：move to pose / rotate / change arm
- hand skill：grasp / release
- body skill：upbody / downbody

同时，LLM 不是裸规划，而是带着这些约束输入：

- 任务描述
- 感知结果
- 技能库
- 工作空间限制
- 安全限制
- 关节/ articulated object 的运动学知识

所以它实际做的是：

> **受约束的技能级规划**，而不是开放式自由生成。

### ### 核心直觉

把“直接让大模型控制人形机器人”改成“**先把任务压缩成可解释技能，再让RL只负责动态稳定**”，改变了三个瓶颈：

1. **信息瓶颈改变了**
   - 原来：语言/视觉直接映射到复杂全身动作，维度太高；
   - 现在：语言和视觉先被压缩成 3D 目标 + 技能序列。

2. **约束瓶颈改变了**
   - 原来：安全、工作空间、运动学常常只在后处理里补；
   - 现在：这些约束直接写进 planner 的输入。

3. **控制瓶颈改变了**
   - 原来：上肢 manipulation 会破坏 locomotion 稳定性；
   - 现在：RL 训练时就把上肢视为扰动来源，控制器学会“边被扰动边站稳/行走”。

于是能力变化是：

- 从“会规划但不一定能做”
- 变成“能在真实全尺寸 humanoid 上把规划、感知和执行串起来做完”

### 为什么这套设计有效？

因果上看，Trinity 不是靠某个单一模型突然变强，而是靠**接口设计**：

- **VLM** 提供对 manipulation 真正有用的几何锚点；
- **LLM** 只在技能层面组合动作，避免直接碰高维控制；
- **RL** 专门吸收动态扰动，承担“物理世界最后一公里”。

这让每一层的误差模式更可控，也更容易定位。

### 战略取舍

| 设计选择 | 得到的能力 | 代价/风险 |
|---|---|---|
| 模块化分层，而非端到端 | 可解释、易替换、容易插入安全约束 | 模块间误差会累积，整体最优性受限 |
| 下肢RL与上肢技能解耦 | 操作时更稳，sim-to-real更容易 | 全身协同可能不如统一策略自然 |
| GPT-4 规划 + prompt约束 | 快速获得任务理解与技能编排能力 | 依赖闭源模型，复现实验与时延受限 |
| RGB-D + VLM找可操作部件 | 易于得到3D操作点 | 遮挡、反光、深度误差会直接影响抓取 |
| 技能库式执行 | 安全边界和接口清晰 | 超出技能库的任务扩展性有限 |

---

## Part III：证据与局限

这篇论文的证据形态更像**系统演示/技术报告**，而不是严格 benchmark 论文。  
因此最强信号来自**真实机器人案例**，不是大规模量化表格。

### 关键证据

- **Case-study 1：开门任务**
  - 机器人先抓门把手，再拉门；
  - 受门体外力影响时，会抬脚、后撤步重新稳定；
  - 说明系统不是只会“抓”，而是能在接触扰动下保持全身平衡并完成任务。

- **Case-study 2：快速上肢动作 + 身高调整**
  - 机器人站立时进行快速摆臂和上身高度变化；
  - 依然维持平衡；
  - 说明“把上肢视作扰动来训练”的 RL 设计确实提升了鲁棒性。

- **Case-study 3：双臂持物前后移动**
  - 双臂持物并前后移动 **20 cm**；
  - 系统通过调整 pitch 保持质心稳定；
  - 说明该控制器不仅能静站，还能应对载荷引起的重心变化。

- **Case-study 4：安全性对比**
  - VLM 对“抓刀伤人”的询问仍会返回刀柄框；
  - 加入 LLM 安全约束后，系统拒绝执行；
  - 说明模块化设计确实能用高层规划补足单模块的安全缺陷。

### 能力跃迁到底体现在哪？

相较于只做 humanoid control 或只做 language planning 的工作，Trinity 的提升点在于：

> 它把“语义理解—目标定位—技能规划—全身稳定执行”串成了一个在真实全尺寸人形机器人上可跑的闭环系统。

最能支持这一点的，不是单一数值，而是**真实机器人在接触扰动中的连续行为链**。

### 1-2 个关键指标

- 双臂持物前后移动：**20 cm**
- RL 训练规模：**4096 并行 Isaac Gym 环境**（体现训练设定，而非性能SOTA）

### 局限性

- **Fails when**: VLM 的可操作部件定位不准、深度恢复误差较大、或接触力显著超出训练分布时，后续抓取与平衡链条会一起失效；对超长程、超开放世界、需要新技能组合的任务也缺少直接证据。
- **Assumes**: 依赖预定义技能库、RGB-D 感知、articulated object 的运动学提示、仿真训练得到的 locomotion policy，以及 GPT-4 级别的大模型规划能力；同时假设上肢动作可通过技能接口而非连续全身优化来表达。
- **Not designed for**: 端到端从原始图像到全身力矩控制；开放式灵巧手精细接触操作；形式化可验证的安全控制；严格的多基线定量 benchmark 比较。

### 资源与复现依赖

这篇论文的可复现性受以下因素明显影响：

- **闭源依赖**：任务规划器使用 GPT-4；
- **硬件依赖**：真实部署在 Tien Kung 人形机器人、NVIDIA Jetson Orin、Gemini 335L Stereo Vision 3D camera 上；
- **训练资源**：Isaac Gym 大规模并行仿真；
- **系统工程依赖**：技能库、工作空间约束、安全提示、运动学先验都需要人工系统集成。

### 可复用组件

- **上肢扰动随机化的 humanoid RL 控制器训练范式**
- **基于技能库的 LLM 约束式规划接口**
- **VLM bounding box + depth 恢复 3D 操作点的感知接口**
- **把安全约束前置到 planner 的系统级防护思路**

## Local PDF reference

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Trinity_A_Modular_Humanoid_Robot_AI_System.pdf]]