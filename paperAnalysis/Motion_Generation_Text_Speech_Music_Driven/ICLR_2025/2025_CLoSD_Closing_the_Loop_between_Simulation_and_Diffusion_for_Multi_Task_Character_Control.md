---
title: "CLoSD: Closing the Loop between Simulation and Diffusion for Multi-Task Character Control"
venue: ICLR
year: 2025
tags:
  - Motion_Generation_Text_Speech_Music_Driven
  - task/physics-based-control
  - task/text-to-motion
  - diffusion
  - reinforcement-learning
  - dataset/HumanML3D
  - dataset/AMASS
  - repr/SMPL
  - opensource/full
core_operator: 闭环扩散-仿真控制（Diffusion Planner + RL Tracker）：扩散模型作为实时通用规划器，RL控制器作为物理仿真执行器，两者闭环交互实现多任务物理可信运动
primary_logic: |
  文本提示 + 目标位置/交互约束
  → Diffusion Planner（DiP）：自回归扩散模型实时生成短期运动规划（条件=文本+目标+环境反馈）
  → RL Tracking Controller：物理仿真中跟踪DiP输出的运动规划，产生物理可信的关节力矩
  → 环境状态反馈回传DiP → 闭环：DiP根据实际执行结果动态调整后续规划
  → 无缝切换多任务：导航、击打、坐下、站起等
claims:
  - "CLoSD首次实现扩散模型与物理仿真的闭环交互，DiP作为通用规划器根据环境反馈动态调整运动规划"
  - "在多任务场景（导航+击打+坐下+站起）中实现无缝任务切换，物理可信度和任务成功率均优于开环基线"
  - "DiP的自回归扩散设计支持实时响应（<100ms延迟），使闭环控制在仿真中可行"
pdf_ref: paperPDFs/Motion_Generation_Text_Speech_Music_Driven/ICLR_2025/2025_CLoSD_Closing_the_Loop_between_Simulation_and_Diffusion_for_Multi_Task_Character_Control.pdf
category: Motion_Generation_Text_Speech_Music_Driven
---

# CLoSD: Closing the Loop between Simulation and Diffusion for Multi-Task Character Control

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [Project Page](https://guytevet.github.io/CLoSD-page/) · [ICLR 2025](https://openreview.net/forum?id=UMfcdRIotC)
> - **Summary**: CLoSD首次实现扩散模型与物理仿真的闭环交互——自回归扩散模型（DiP）作为实时通用规划器，RL控制器作为物理仿真执行器，两者闭环反馈实现文本驱动的多任务物理可信运动控制（导航、击打、坐下、站起无缝切换）。
> - **Key Performance**:
>   - 多任务无缝切换：导航→击打→坐下→站起，物理可信且任务成功率显著优于开环基线
>   - DiP实时响应（<100ms），支持闭环控制的延迟要求

---

## Part I：问题与挑战

### 真正的卡点

物理可信的角色控制面临**运动多样性与物理可信性的矛盾**和**多任务泛化困难**两大核心挑战：

- **扩散vs仿真的互补困境**：运动扩散模型擅长生成多样、文本可控的运动，但输出不保证物理可信（穿透、浮空、力学不合理）；RL物理控制器保证物理可信，但运动多样性和文本可控性极差
- **开环规划的脆弱性**：现有方法先用扩散模型生成完整运动轨迹，再用控制器跟踪——但物理仿真中的扰动会导致实际状态偏离规划，开环规划无法修正，累积误差导致任务失败
- **多任务切换**：不同任务（导航、击打、坐下）需要不同的运动模式，现有方法通常为每个任务训练独立控制器，无法无缝切换

### 输入/输出接口

- 输入：文本提示（任务描述）+ 目标位置/交互对象
- 输出：物理仿真中的角色运动（关节力矩→SMPL姿态序列）

---

## Part II：方法与洞察

### 整体设计

CLoSD由两个闭环交互的模块组成：

1. **Diffusion Planner（DiP）**：
   - 自回归扩散模型，每步生成短期运动规划（~1-2秒）
   - 条件输入：文本提示 + 目标位置 + 当前环境状态（来自控制器反馈）
   - 快速响应：轻量化设计支持<100ms推理延迟

2. **RL Tracking Controller**：
   - 简单鲁棒的运动模仿器，在物理仿真中跟踪DiP输出的运动规划
   - 输出关节力矩，产生物理可信的运动
   - 将实际执行后的环境状态反馈给DiP

3. **闭环交互**：DiP根据控制器反馈的实际状态（而非假设的理想状态）生成下一步规划 → 自动修正累积误差 → 适应环境扰动

### 核心直觉

**什么变了**：从"扩散生成完整轨迹→控制器开环跟踪"到"扩散实时规划↔控制器执行反馈"的闭环。

**哪些分布/约束/信息瓶颈变了**：
- 闭环反馈消除了开环规划的累积误差瓶颈 → DiP始终基于真实环境状态规划，而非基于可能已偏离的假设状态
- 自回归扩散的短期规划窗口降低了单次生成的难度 → 每步只需规划1-2秒，而非整个任务的完整轨迹
- 文本条件通过DiP注入，RL控制器只需做简单跟踪 → 将"多样性+可控性"和"物理可信性"解耦到两个模块，各自优化

**为什么有效**：物理仿真的核心难点是环境交互的不确定性，闭环反馈是应对不确定性的标准范式（类比MPC）。DiP的自回归设计天然支持在线重规划，RL控制器的鲁棒跟踪能力则保证了物理可信性。

**权衡**：闭环交互增加了系统复杂度和通信开销；DiP的短期规划窗口可能限制长期全局最优性；RL控制器的跟踪精度是系统上限。

---

## Part III：证据与局限

### 关键实验信号

- **闭环vs开环**：在导航+击打任务中，闭环CLoSD的任务成功率显著高于开环基线（扩散生成完整轨迹+跟踪），尤其在存在环境扰动时差距更大
- **多任务无缝切换**：CLoSD在导航→手部击打→脚部击打→坐下→站起的连续任务序列中实现无缝切换，无需重新初始化
- **物理可信度**：生成的运动在接触力、关节力矩等物理指标上均合理，无穿透和浮空
- **实时性**：DiP推理延迟<100ms，满足闭环控制的实时要求

### 局限与可复用组件

- **局限**：当前任务类型有限（导航、击打、坐下/站起），未扩展到更复杂的操作任务；RL控制器的训练仍需大量仿真数据；DiP的短期规划可能在需要长期策略的任务中次优
- **可复用**：扩散规划器+RL控制器的闭环框架可迁移到任何需要物理可信运动的场景（机器人、游戏NPC等）；DiP的自回归扩散设计适用于任何需要在线重规划的实时控制任务；闭环反馈的设计模式为扩散模型在交互式场景中的应用提供了通用范式

---

## 本地 PDF 引用

![[paperPDFs/Motion_Generation_Text_Speech_Music_Driven/ICLR_2025/2025_CLoSD_Closing_the_Loop_between_Simulation_and_Diffusion_for_Multi_Task_Character_Control.pdf]]
