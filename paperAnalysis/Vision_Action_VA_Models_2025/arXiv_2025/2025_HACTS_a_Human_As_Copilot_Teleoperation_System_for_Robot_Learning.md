---
title: "HACTS: a Human-As-Copilot Teleoperation System for Robot Learning"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/robot-manipulation
  - task/teleoperation
  - bilateral-joint-synchronization
  - kinematically-equivalent-controller
  - human-in-the-loop
  - dataset/OpenBox
  - dataset/SteamBun
  - dataset/UprightMug
  - dataset/CloseBin
  - opensource/no
core_operator: "用低成本、运动学等价的双向关节同步遥操作器把人类从单向操作者变成可随时接管的“副驾驶”，并记录失败邻域的纠错轨迹供机器人学习。"
primary_logic: |
  机器人自主执行/人工示教需求 → 领导-跟随器双向关节位置同步与脚踏切换接管 → 采集失败附近的人工纠错动作 → 提升模仿学习的恢复与泛化能力，并支撑在线人类在环强化学习
claims:
  - "在相同总数据量下，以50条HACTS介入轨迹替代50条额外专家轨迹后，ACT在OpenBox/SteamBun/UprightMug上的成功率由full-ACT的60/80/60%提升到80/90/70% [evidence: comparison]"
  - "在ID失败纠正设置中，HACTS-ACT在OpenBox/SteamBun/UprightMug上的成功率达到50/80/70%，均高于full-ACT的40/60/40%，说明失败邻域的纠错数据比单纯增加成功示教更有效 [evidence: comparison]"
  - "在CloseBin在线强化学习中，RLPD-HACTS经过10分钟离线预训练和45分钟在线训练后达到80%成功率，平均episode长度由32步降至19步 [evidence: comparison]"
related_work_position:
  extends: "Gello (Wu et al. 2024)"
  competes_with: "RoboCopilot (Wu et al. 2024); Bi-ACT (Kobayashi et al. 2025)"
  complementary_to: "ACT (Zhao et al. 2023); RLPD (Ball et al. 2023)"
evidence_strength: moderate
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_HACTS_a_Human_As_Copilot_Teleoperation_System_for_Robot_Learning.pdf
category: Embodied_AI
---

# HACTS: a Human-As-Copilot Teleoperation System for Robot Learning

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2503.24070)
> - **Summary**: HACTS 通过低成本双向关节同步，把传统“单向示教器”升级为可在机器人自主执行中随时接管的“人类副驾驶”，从而高效采集失败附近的纠错数据并提升 IL / HITL-RL 效果。
> - **Key Performance**: 同样 100 条总轨迹下，HACTS-ACT 在 OpenBox / SteamBun / UprightMug 上达 80% / 90% / 70%，优于 full-ACT 的 60% / 80% / 60%；CloseBin 上 RLPD-HACTS 经 10 分钟离线 + 45 分钟在线训练达到 80% 成功率。

> [!info] **Agent Summary**
> - **task_path**: 人类遥操作/机器人自主执行中的在线接管 -> 机器人关节动作与纠错轨迹
> - **bottleneck**: 单向遥操作无法把机器人当前关节状态回传给操作者，导致接管起点错位、纠错不连续、高价值失败数据难采
> - **mechanism_delta**: 在低成本运动学等价控制器上加入 follower-to-leader 反向关节同步和脚踏切换，使人类能从机器人当前姿态无缝接管
> - **evidence_signal**: 同预算数据对比下优于纯专家示教；在线 RL 在 45 分钟微调后达 80% 成功率
> - **reusable_ops**: [bilateral joint mirroring, intervention-triggered correction logging]
> - **failure_modes**: [contact-rich tasks requiring force feedback, platforms without stable joint-state readout]
> - **open_questions**: [how much gain comes from bidirectional sync vs targeted data selection, whether the design scales to bimanual or humanoid setups]

## Part I：问题与挑战

这篇文章要解决的真问题，不是“如何做一个更贵的遥操作器”，而是**人类在机器人自主执行过程中，如何从正确的当前状态无缝接管**。

现有 ALOHA / Gello 一类低成本装置已经证明：运动学等价控制器很适合收集离线示教。但它们大多是**单向控制**——人发命令给机器人，机器人当前关节状态不会同步回到操作者手里的装置。于是当策略执行出错、需要人工介入时，会出现一个关键断点：

- 人看到的是视觉画面，但**手上设备的姿态不等于机器人当前姿态**；
- 一旦强行接管，容易产生动作跳变；
- 采到的数据仍以“成功演示”为主，而不是最有价值的**失败邻域纠错数据**。

为什么这个问题现在值得解决？因为机器人学习已经从“只做离线模仿”走向“边执行边纠错”：
- VLA / visuomotor policy 需要的不只是成功轨迹，还需要**恢复失败**的数据；
- HITL RL 需要人类能在在线探索中进行**短促、及时、精细**的干预；
- 但传统力反馈/触觉反馈双边系统成本高、依赖 follower 侧专门传感器，不适合大量 UR5 这类通用平台。

**输入/输出接口与边界条件：**
- 输入：人类操作者动作、机器人实时关节状态、视觉观测；
- 输出：机器人关节控制指令，以及可回放的人工纠错轨迹；
- 边界：论文聚焦的是**关节位置级双向同步**，不是力/触觉级双边遥操作；主要验证平台是 UR5，任务是桌面操作。

## Part II：方法与洞察

HACTS 的核心设计非常朴素：**不追求昂贵的力反馈，而是先解决“姿态对齐”这个最刚性的接管前提。**

### 系统工作流

1. **离线示教模式（leader → follower）**  
   像 ALOHA / Gello 一样，操作者直接带动运动学等价的 leader 装置，机器人跟随，采集高精度示教数据。

2. **在线副驾驶模式（follower → leader）**  
   当策略控制机器人自主执行时，机器人当前关节位置被反向同步到 leader 装置上。操作者此时不是“看屏幕猜状态”，而是手上就握着一个和机器人姿态对齐的控制器。需要时通过脚踏板切换，直接从当前姿态接管。

3. **学习耦合方式**  
   - 对 IL：把 HACTS 采到的纠错轨迹与原始专家轨迹混合，训练 ACT / DP；
   - 对 RL：把人类接管动作与机器人探索数据一起送入 RLPD，形成 online HITL 训练闭环。

### 核心直觉

**What changed**：从单向遥操作变成了**双向关节同步**。  
**Which bottleneck changed**：改变的不是模型结构本身，而是**接管时的信息与约束条件**——操作者从“只能看视觉估计状态”变成“手上设备已与机器人当前关节态对齐”。  
**What capability changed**：这直接带来三件事：
1. 接管更平滑，不容易产生突兀动作；
2. 纠错数据集中出现在**失败附近状态**，而不是重复采集成功轨迹；
3. 人类干预可从“整段接管”缩短成“短促修正”，因此更适合在线 RL。

为什么这在因果上有效？因为机器人学习里的薄弱点常常不在主分布中心，而在**快要失败的边界状态**。HACTS 让人类恰好能在这些边界状态下接手并纠偏，于是训练数据分布从“成功演示主导”变成“成功演示 + 失败邻域修正”。这比单纯多采一些成功轨迹更能提升恢复能力、数据效率和 OOD 鲁棒性。

另外，作者刻意只做**位置同步**而不做力反馈，也是一种战略取舍：几乎所有机器人都能读到关节位置，但并非所有机器人都有稳定的力/触觉传感能力。于是 HACTS 选择了一个“兼容性最强的最小双边信号”。

### 战略取舍

| 设计选择 | 解决的瓶颈 | 带来的能力 | 代价/折中 |
|---|---|---|---|
| 双向关节位置同步 | 接管时 leader 与 robot 姿态错位 | 平滑 takeover，易采纠错数据 | 不提供接触力/触觉信息 |
| 运动学等价 leader 装置 | 仅用末端控制时表达受 IK/FK 限制 | 更精细、直观的关节级控制 | 需要针对目标机械臂做标定/缩放 |
| 低成本舵机 + 3D 打印 | 遥操作硬件昂贵、难扩展 | 硬件成本低于 \$300，便于复制 | 扭矩、刚度、耐久性有限 |
| 脚踏板模式切换 | 人机权责切换不清晰 | 接管时机明确，操作简单 | 仍依赖人工判断，不是自动仲裁 |

## Part III：证据与局限

### 关键证据

**1. 同预算数据对比信号：纠错数据比额外成功示教更值钱。**  
在 3 个 IL 任务里，作者先用 50 条专家轨迹训练 pre-ACT，再分别加入：
- 50 条额外专家轨迹（full-ACT），或
- 50 条 HACTS 介入轨迹（HACTS-ACT）。

结果 HACTS-ACT 在 OpenBox / SteamBun / UprightMug 上达到 80% / 90% / 70%，优于 full-ACT 的 60% / 80% / 60%。这说明收益不是来自“多了 50 条数据”，而是来自**这 50 条数据更集中于失败边界**。

**2. 失败恢复信号：HACTS 学到的是 recovery，不只是 imitation。**  
在 ID 失败纠正设置下，HACTS-ACT 在 3 个任务上都超过 full-ACT；例如 UprightMug 从 pre-ACT 的 10% 提升到 70%，而 full-ACT 只有 40%。这直接支撑论文主张：**双向同步最核心的价值，是帮助采到恢复失败所需的数据。**

**3. OOD 信号：把“完全不会”变成“能纠回来”。**  
在 OOD 静态和动态场景中，pre/full ACT/DP 基本都是 0% 成功率；加入 HACTS 纠错数据后，ODSS 达到 30% / 50% / 40%，OpenBox 的 ODDS 达到 40%。这说明方法不只是拟合训练分布，而是在一定程度上扩大了策略可恢复的状态覆盖。

**4. 在线 RL 信号：人类干预可逐步缩短为细粒度修正。**  
在 CloseBin 上，RLPD-HACTS 经过 10 分钟离线预训练 + 45 分钟在线训练达到 80% 成功率，episode length 从 32 步降到 19 步。并且作者展示在线训练过程中平均 intervention length 后期降到 6 步以下，说明系统允许人类从“长时间接管”退化为“短时间校正”，这正是 HITL RL 想要的工作模式。

**整体判断：证据强度为 moderate。**  
优点是：有多任务 IL + 一个在线 RL 场景，且比较对象清晰。  
不足是：任务规模小、都是作者自建场景、每组只做 10 次 rollout，且缺少对“仅换硬件/仅换数据策略”的系统性消融。

### 局限性

- **Fails when**: 任务强依赖力/触觉反馈时，例如高接触、摩擦敏感、精细插接类操作；或者动作过快、系统频率与舵机响应不足时，单纯位置同步可能不够。
- **Assumes**: 机器人能稳定读出关节位置并做标定；存在运动学上可对应的 leader 装置；在线 RL 还依赖额外训练的奖励分类器（文中用 200 个正例 + 约 1000 个负例）；实验运行在 UR5 + 多相机 + RTX 4090 工作站上。
- **Not designed for**: 全身/双臂 humanoid 遥操作、自动化接管仲裁、标准化跨平台 benchmark 结论，也不是一个通用的力反馈遥操作系统。

补充看，论文还有两个现实约束：
- **比较缺口**：没有直接和 Gello、SpaceMouse、力反馈系统做 head-to-head 对比，因此增益究竟来自“双向同步”还是“更有针对性的数据采集协议”，尚未完全拆解。
- **复现缺口**：文中给了 BOM 和系统描述，但未提供明确代码/项目链接，因此当前更像“可实现原型”，不是现成可复现套件。

### 可复用组件

- **双向关节镜像模块**：leader ↔ follower 的偏置校准与反向同步逻辑；
- **脚踏板接管协议**：自主执行与人工接管之间的低摩擦切换；
- **失败邻域纠错采集范式**：先训练初始策略，再围绕失败点采 correction trajectories；
- **HACTS + policy learner 的通用接法**：可直接接到 ACT、DP、RLPD 这类现有策略学习框架上。

## Local PDF reference

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_HACTS_a_Human_As_Copilot_Teleoperation_System_for_Robot_Learning.pdf]]