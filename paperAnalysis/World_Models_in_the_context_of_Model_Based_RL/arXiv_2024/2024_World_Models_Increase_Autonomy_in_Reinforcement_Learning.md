---
title: "World Models Increase Autonomy in Reinforcement Learning"
venue: arXiv
year: 2024
tags:
  - Embodied_AI
  - task/reset-free-reinforcement-learning
  - task/goal-conditioned-control
  - world-model
  - go-explore
  - curriculum-learning
  - dataset/EARL
  - opensource/partial
core_operator: 通过世界模型中的“前后往返”目标探索与任务相关目标偏置想象训练，让智能体在无重置环境里同时学会探索、回到初始态和完成任务。
primary_logic: |
  无重置连续交互 + 初始态/评测目标分布 → 在真实环境中交替执行“去评测目标 / 回初始态 / 探索新状态”的 Back-and-Forth Go-Explore，并在世界模型想象中按比例训练达到评测目标、初始态和随机回放状态 → 学到更偏向任务相关状态、可自我重置且样本效率更高的 goal-conditioned policy
claims:
  - "在 8 个 reset-free 任务中，MoReFree 与 reset-free PEG 在 7/8 个任务上的最终性能和样本效率均优于 IBC、MEDAL、R3L 等基线，且不依赖环境奖励或示范 [evidence: comparison]"
  - "相对 reset-free PEG，MoReFree 在 3 个最难任务 Push(hard)、Pick&Place(hard) 和 Ant 上分别提升约 45%、13% 和 36% [evidence: comparison]"
  - "去掉 Back-and-Forth Go-Explore 或去掉任务相关 imagination goals 都会显著降低困难任务上的归一化性能，说明两者存在协同而非单独可替代 [evidence: ablation]"
related_work_position:
  extends: "PEG (Hu et al. 2023)"
  competes_with: "IBC (Kim et al. 2023); MEDAL (Sharma et al. 2022)"
  complementary_to: "N/A"
evidence_strength: strong
pdf_ref: paperPDFs/World_Models_in_the_context_of_Model_Based_RL/arXiv_2024/2024_World_Models_Increase_Autonomy_in_Reinforcement_Learning.pdf
category: Embodied_AI
---

# World Models Increase Autonomy in Reinforcement Learning

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2408.09807) | [Project](https://yangzhao-666.github.io/morefree) | [OpenReview](https://openreview.net/forum?id=ZdMIXltJzK)  
>   注：正文对应的发表版题名为 *Reset-free Reinforcement Learning with World Models*（TMLR 2025）。
> - **Summary**: 论文把世界模型探索从“只追求新奇状态”改成“围绕初始态与评测目标来回练习并适度外扩探索”，从而在几乎无人工重置、无环境奖励/无示范的条件下显著提升 reset-free RL 的自治性。
> - **Key Performance**: 在 8 个任务中 7/8 任务优于已有 reset-free 基线；相对 reset-free PEG，在 Push(hard)/Pick&Place(hard)/Ant 上约提升 **+45% / +13% / +36%**。

> [!info] **Agent Summary**
> - **task_path**: 无重置连续交互环境 + 初始态/评测目标分布 -> 可在 episodic 评测下达成目标并能把系统带回任务相关区域的 goal-conditioned policy
> - **bottleneck**: 缺少 reset 后，探索会把数据分布推向角落/墙边等任务无关或难逃逸状态，导致世界模型和策略主要在“错的区域”上学习
> - **mechanism_delta**: 将 PEG 的单向新奇探索改成前后往返的 Go-Explore，并在 imagination 中显式提高初始态与评测目标的采样权重
> - **evidence_signal**: 8 任务对比 + 困难任务消融：MoReFree 在 7/8 任务超基线，且移除任一关键机制都会明显掉点
> - **reusable_ops**: [back-and-forth-go-explore, task-relevant-imagination-goal-mixing]
> - **failure_modes**: [hard-to-model-dynamics-like-sawyer-door, overly-large-alpha-causing-under-exploration]
> - **open_questions**: [how-to-adapt-alpha-curriculum-online, whether-the-method-scales-to-pixel-and-real-robot-settings]

## Part I：问题与挑战

这篇工作讨论的不是普通 episodic RL，而是 **reset-free RL**：训练时智能体长时间连续交互，几乎不依赖外部 reset；但评测时仍按常规 episodic 方式，从初始态出发完成目标。  
对真实机器人来说，这很关键，因为“reset 环境”往往意味着人工摆放、额外机械装置、脚本工程，甚至第二台机器人。

### 真正难点是什么？

真正瓶颈不是“没有 reset”本身，而是：

1. **训练数据分布会漂走**  
   agent 一旦进入角落、墙边、门缝、物块卡死等 sink states，就会长时间停留在这些区域。
2. **强探索在 reset-free 下会变成过探索**  
   PEG 这类世界模型方法在 episodic 场景中很强，但在 reset-free 时，可能把大量预算花在“新但没用”的状态上。
3. **世界模型会被错误分布喂养**  
   回放缓冲区若充满任务无关轨迹，world model、goal-conditioned policy 和 value learning 都会一起偏掉。
4. **评测目标与训练覆盖目标错位**  
   评测只关心“从初始态去目标”的成功率；但若训练长期停留在远离初始态/目标的区域，最终策略并不适合评测接口。

### 输入 / 输出接口

- **输入**：低维状态、goal-conditioned 任务设定、初始态分布 \(ρ_0\)、评测目标分布 \(ρ_g^*\)、极少 reset 的连续训练流
- **输出**：一个既能完成目标、又能把系统带回任务相关区域的 goal-conditioned policy；外加用于收集数据的 exploration behavior
- **边界条件**：
  - 训练时基本不使用环境奖励和示范
  - 评测仍是 episodic
  - benchmark 中并非“绝对零 reset”，而是**极少量硬重置**（如每 \(10^5\) 或 \(2\times10^5\) 步一次）

### 为什么现在值得做？

因为近期 world model / MBRL 已经在两件事上成熟很多：

- **样本效率高**
- **能做长时程 goal-conditioned exploration**

作者先做了一个重要观察：即使只是把 PEG 直接改成 reset-free 版本，它都已经能在 Ant 上压过先前 SOTA 的 model-free reset-free 方法 IBC。  
这说明 **world models 确实能提高 RL 的自治性**；但也暴露出一个新问题：它们会“太会探索”，却不够聚焦任务相关状态。

---

## Part II：方法与洞察

MoReFree 的设计很克制：它没有推翻 PEG/Dreamer 风格的 MBRL 框架，而是只改两个最关键的因果旋钮：

1. **真实环境里怎么收集数据**
2. **世界模型里怎么训练 goal-conditioned policy**

### 1. Back-and-Forth Go-Explore：把探索变成“做任务—回起点—再探索”

PEG 原本的 Go-Explore 是：

- 先选一个 goal
- 用 goal-conditioned policy 去接近它（Go phase）
- 再切换到 exploration policy 扩展新区域（Explore phase）

MoReFree 认为这在 reset-free 里还不够，因为 agent 可能一直往外走，不再回到评测相关区域。  
因此它把数据采集改成 **Back-and-Forth Go-Explore**：

- 以概率 `α`：
  - 先采样一个**评测目标** `g*`
  - 再采样一个**初始态目标** `g0`
  - 先去 `g*`，再在当前状态基础上继续去 `g0`
- 以概率 `1-α`：
  - 仍执行 PEG 风格的 exploratory goal 搜索，避免只在局部来回循环

这等于在真实环境里形成一个课程：

**练习完成任务 → 练习“自重置”回起始区域 → 继续探索未知状态**

这样 agent 学到的不是单纯“走远”，而是“走远之后还能回来”。

### 2. 任务相关 imagination goals：让世界模型里的训练目标与评测目标对齐

PEG 的 goal-conditioned policy 主要对**随机 replay states**练习。  
这能提高通用 goal-reaching 能力，但 reset-free 下不够，因为真正关键的是两类状态：

- **评测目标附近的状态**
- **初始态附近的状态**（相当于 reset 目标）

所以 MoReFree 在 world model 的 imagination 训练里，把目标采样改成三路混合：

- `α/2`：评测目标分布 `ρg*`
- `α/2`：初始态分布 `ρ0`
- `1-α`：随机 replay buffer 状态

结果是：policy 不只会“达到任意见过的状态”，还会反复练习那些它在评测和回退过程中**真的需要到达**的状态。

### 核心直觉

PEG 的问题不是“不会探索”，而是 **探索分布与任务成功分布错位**。

MoReFree 的核心改动是：

- **把真实数据采样分布** 从“偏向新奇状态”改成“新奇状态 + 任务相关状态”
- **把想象训练目标分布** 从“随机状态为主”改成“随机状态 + 初始态/评测目标”

这带来的因果链条是：

**目标采样变了**  
→ **回放缓冲区里任务相关轨迹占比提高**  
→ **world model 在关键区域更准，goal-conditioned policy 在关键目标上练得更多**  
→ **agent 更会执行 forward 行为和 back/reset 行为**  
→ **下一轮又能采到更多任务相关数据**

本质上，作者修复的是一个 **状态分布与训练预算错配** 问题。  
不是让 agent 探索更广，而是让它把“广度”中的更多预算落在**与 episodic 成功率直接相关的区域**。

### 为什么这个设计有效？

因为它同时解决了 reset-free RL 的两个互相耦合的问题：

- **数据问题**：没有足够多的 task-relevant experience
- **优化问题**：policy 没有被训练去到真正重要的 goals

只改一个都不够：

- 只改探索，不改 imagination：policy 未必真会回初始态/到评测目标
- 只改 imagination，不改探索：回放中没有足够相关数据，world model 也学不准

作者的消融也正好证明：这两个模块是**协同关系**，不是任选其一。

### 战略取舍

| 设计选择 | 改变的瓶颈 | 能力收益 | 代价 / 风险 |
|---|---|---|---|
| Back-and-Forth Go-Explore | 真实环境中的状态访问分布 | 增加初始态/目标附近数据，学会 self-reset | 若 α 过大，会牺牲全局探索 |
| imagination 中混入初始态/评测目标 | policy 的训练目标分布 | 提升真实会用到的 goal-reaching 能力 | 若 world model 在这些区域不准，会放大模型偏差 |
| 保留 exploratory goals | 避免课程过窄 | 维持状态覆盖与发现新路 | 仍有 planning / compute 开销 |
| 固定 α=0.2 | 简化超参设计 | 跨任务稳定、无需细调 | 不一定是每个任务的最优 curriculum |

---

## Part III：证据与局限

### 关键证据：能力跳变到底体现在哪？

**信号 1：跨任务比较说明 world model 确实提高了自治性**  
在 8 个 reset-free 任务里，MoReFree 和 reset-free PEG 在 **7/8 个任务**上都优于 IBC、MEDAL、R3L 等方法，而且它们**不需要环境奖励或示范**。  
这不是“小幅调参胜利”，而是说明：**在低监督 reset-free 场景里，model-based 路线本身就更有优势**。

**信号 2：MoReFree 相比 reset-free PEG 的增益集中出现在最难任务**  
这点最能说明论文真正的新贡献。  
如果只是“world model 比 model-free 强”，那 reset-free PEG 已经说明了；MoReFree 的价值在于，它在最容易出现 task-irrelevant over-exploration 的任务上进一步拉开差距：

- Push(hard): 约 **+45%**
- Pick&Place(hard): 约 **+13%**
- Ant: 约 **+36%**

这说明作者引入的“任务相关状态偏置”不是锦上添花，而是**专门修复 hard reset-free exploration 的关键杠杆**。

**信号 3：状态访问分析直接支持“更多任务相关数据”这个机制解释**  
热力图显示，MoReFree 和 reset-free PEG 都比 model-free 基线覆盖更广；  
但在更难环境里，MoReFree 在任务相关区域收集到 **1.3–5×** 更多数据。  
这正对应论文的核心论点：  
> 不是只要探索广就行，而是要在广覆盖的同时，保住对初始态/目标态附近的高访问密度。

**信号 4：消融验证两大模块存在协同**  
去掉 Back-and-Forth Go-Explore 或去掉任务相关 imagination 目标，性能都会明显下降。  
这支持“正反馈闭环”的解释，而不是某一个 trick 单独起作用。

### 关键指标

- **7/8 任务**：MoReFree / reset-free PEG 优于先前 reset-free 基线
- **+45% / +13% / +36%**：MoReFree 相对 reset-free PEG 在 3 个 hardest tasks 上的提升

### 局限性

- **Fails when**: 世界模型难以准确建模动力学时，方法会明显吃亏；论文中最典型的是 **Sawyer Door**，作者甚至发现 DreamerV2/V3 在 episodic 版本里都很难学好，说明问题更像“模型学不准门的动力学”，而不是“课程不够好”。
- **Assumes**: 使用的是低维状态输入而非高维图像；知道初始态分布与评测目标分布；benchmark 里允许极少量硬重置而非绝对零 reset；训练依赖 world model imagination、goal planning 和 GPU 资源（文中报告 MBRL 实验约 1–2 天，使用 2080/3090/A100）。
- **Not designed for**: 尚未验证像素输入、真实机器人长期部署、不可逆失败环境，或缺乏清晰任务相关 goal distribution 的场景；对严格安全约束场景也没有专门机制。

### 可复用组件

1. **Back-and-Forth Go-Explore**  
   可作为任何 reset-free goal-conditioned agent 的通用数据采集课程。
2. **Task-relevant imagination goal mixing**  
   很适合迁移到其他 world-model agents：把 imagination 目标分布往 deployment / evaluation 真正关心的目标上拉。
3. **任务相关状态占比诊断**  
   用热力图和“task-relevant visitation ratio”来判断 agent 是“有效探索”还是“无关过探索”。
4. **无需环境奖励的 learned distance reward**  
   在低监督机器人训练里有现实意义，可减少 reward engineering。

### 一句话结论

这篇论文最有价值的地方，不是再次证明“MBRL 很强”，而是更具体地说明了：

> **在 reset-free RL 中，真正决定自治性的不是探索得多广，而是能否让 world model 把学习预算持续压在“会影响回起点与完成任务”的那部分状态分布上。**

![[paperPDFs/World_Models_in_the_context_of_Model_Based_RL/arXiv_2024/2024_World_Models_Increase_Autonomy_in_Reinforcement_Learning.pdf]]