---
title: "RILe: Reinforced Imitation Learning"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/imitation-learning
  - task/robotic-locomotion
  - reinforcement-learning
  - reward-learning
  - trainer-student
  - dataset/LocoMujoco
  - dataset/MuJoCo
  - opensource/no
core_operator: "用判别器反馈训练一个输出学生奖励的trainer策略，并与student策略同步协同优化。"
primary_logic: |
  专家示范 + 学生与环境交互 → 判别器评估学生行为与专家的相似度，trainer据此用RL学习输出阶段性奖励 → student在该自适应密集奖励下逐步逼近专家策略
claims:
  - "RILe学习到的奖励函数比GAIL/AIRL更具动态适应性，且奖励变化与学生性能提升呈更强正相关 [evidence: ablation]"
  - "在MuJoCo Humanoid-v2上，RILe达到5928回报，超过GAIL的5709、AIRL的5623和IQ-Learn的327 [evidence: comparison]"
  - "RILe在噪声专家示范下更稳健：Humanoid-v2在动作噪声Σ=0.5时仍达到5154，而GAIL降至902 [evidence: comparison]"
related_work_position:
  extends: "AIRL (Fu et al. 2018)"
  competes_with: "GAIL (Ho & Ermon, 2016); IQ-Learn (Garg et al. 2021)"
  complementary_to: "DRAIL (Lai et al. 2024)"
evidence_strength: strong
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_RILe_Reinforced_Imitation_Learning.pdf
category: Embodied_AI
---

# RILe: Reinforced Imitation Learning

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2406.08472)
> - **Summary**: 这篇论文把“判别器直接当奖励”的模仿学习，改成“判别器去训练一个会发奖励的教师”，让奖励能随学生能力阶段动态变化，从而在高维连续控制中更接近专家。
> - **Key Performance**: MuJoCo Humanoid-v2 上 RILe=5928（GAIL=5709，AIRL=5623）；LocoMujoco Walk-UnitreeH1 上 RILe=966.2，接入 DRAIL 判别器后达 995.8/1000。

> [!info] **Agent Summary**
> - **task_path**: 专家示范 + 在线环境交互 -> 连续控制/机器人模仿策略
> - **bottleneck**: 高维模仿学习中的相似度信号过于静态，无法为中间子目标提供密集、阶段性的探索反馈；而传统IRL学奖励又太慢
> - **mechanism_delta**: 在student和discriminator之间加入一个用RL训练的trainer，trainer的动作直接作为student奖励，并依据discriminator反馈同步更新
> - **evidence_signal**: 奖励动态性消融（RFDC/FS-RFDC/CPR）+ MuJoCo/LocoMujoco多任务比较共同支持“自适应奖励优于静态判别器奖励”
> - **reusable_ops**: [trainer-as-reward-policy, freeze-trainer-midway]
> - **failure_modes**: [discriminator过强或过拟合会削弱trainer信号, 过多显式expert数据会加快收敛但降低最终性能]
> - **open_questions**: [能否去掉discriminator仍学到有效自适应奖励, 如何在不冻结trainer的情况下维持后期稳定协同训练]

## Part I：问题与挑战

这篇文章真正要解决的，不是“有没有专家示范”，而是：

**在高维连续控制里，怎样把专家示范变成一个足够细粒度、会随学习阶段变化、还能支持探索的奖励信号。**

### 1. 现有路线的真实瓶颈

- **RL** 需要手工奖励，机器人/运动模仿里很难精确设计。
- **IRL** 能从示范中学奖励，但通常要反复执行“固定奖励训练策略到收敛 → 再更新奖励”的外循环，**算力开销大，反馈滞后**。
- **IL / AIL（如 BC, GAIL）** 虽然更高效，但本质上多在做“像不像专家”的比较：
  - BC 直接对动作监督，高维时需要很多专家数据；
  - GAIL/AIRL 用判别器给相似度，但这个信号更像一个**静态裁判**，不是一个会教步骤的老师。

高维任务里，学生往往需要先学会一些**暂时不完美、但长期有帮助的中间行为**。  
单纯“像/不像”的信号，很难告诉它：

- 现在该优先探索哪块状态空间；
- 哪些次优动作其实是通往专家行为的必要过渡；
- 学习早期和后期应该接受什么不同类型的奖励。

### 2. 为什么现在值得解决

论文选的正是最容易暴露这个问题的设置：

- **MuJoCo 高维控制**
- **LocoMujoco 动作捕捉模仿**
- 甚至包含**只有状态、没有动作**的示范场景

这类任务的共同点是：  
**搜索空间大、局部进步重要、静态奖励不够用。**

### 3. 输入/输出接口与边界条件

- **输入**：
  - 专家示范轨迹
  - 环境交互数据
- **输出**：
  - 一个能模仿专家行为的 student policy
- **中间机制**：
  - student 输出环境动作
  - trainer 观察 student 的状态-动作对，并输出一个标量奖励
  - discriminator 判断该状态-动作对是否像专家

边界上，这个方法主要面向：

- 在线交互式学习，而非纯离线模仿
- 连续控制/机器人运动任务
- 需要能训练判别器、student、trainer 三个模块的场景

---

## Part II：方法与洞察

### 方法骨架

RILe 的核心不是再造一个更强判别器，而是**改变判别器在系统中的角色**：

- **Student agent**：正常与环境交互，学习控制策略
- **Discriminator**：只负责评估 student 的行为有多像专家
- **Trainer agent**：根据 student 当前行为，输出给 student 的奖励；它自己再根据 discriminator 的反馈被训练

也就是说，RILe 把经典 GAIL 的：

> 判别器输出 = student 直接优化的奖励

改成了：

> 判别器输出 → 用来训练 trainer  
> trainer 输出 → 作为 student 的实际奖励

这让系统从“学生 vs 裁判”的零和味道，变成了更像“学生 + 教练 + 考官”的结构。

### 一个很关键但容易被忽略的点

如果 trainer 的回报只依赖 discriminator 的分数，那么会出现一个 credit assignment 问题：

- 当 student 行为很像专家时，discriminator 分数高；
- 但 trainer 此时到底应该**奖励** student，还是**惩罚** student，单靠这个分数本身并不能区分。

所以论文专门设计了一个 trainer 回报，使它在：

> **trainer 给出的奖励，与判别器对该行为的缩放判断相一致时**

获得更高回报。

这一步很关键，因为它让 trainer 学到的不是模糊的“专家相似性”，而是：

**“我在这个 student 状态-动作上，到底该给多大的奖惩，才是对的。”**

### 核心直觉

#### what changed

把“静态判别器奖励”换成了“**可学习的奖励策略**”：

- 以前：student 直接追逐判别器分数
- 现在：trainer 学会如何把判别器信号转成更适合当前 student 阶段的奖励

#### which bottleneck changed

1. **信息瓶颈变了**  
   从粗粒度的“像不像专家”，变成阶段相关的、密集的 reward shaping。

2. **优化约束变了**  
   从 IRL 那种慢速外循环，变成**reward 与 policy 同步共演化**。

3. **探索压力变了**  
   trainer 不只是在判断对错，而是在塑造 student 下一阶段更该探索哪里。

#### what capability changed

于是 RILe 获得了 prior work 最缺的一种能力：

**把中间进步也变成可学习信号。**

这意味着它不必要求 student 一开始就走在专家轨迹上；  
相反，它可以暂时鼓励一些“当前看起来不最优，但长期更有帮助”的行为，逐步形成 curriculum。

这也是论文在 maze 可视化中展示得最直观的一点：  
RILe 的奖励热点区域会随 student 进度移动，而 GAIL/AIRL 更像静态地守着“专家区域”。

### 为什么这套设计有效

从因果上看，RILe 做对了三件事：

1. **把判别器降级为校准器，而不是最终奖励源**  
   判别器只负责告诉 trainer：student 当前离专家有多近。

2. **把奖励学习升级成一个长期决策问题**  
   trainer 是用 RL 训练的，它优化的是“给出这类奖励后，未来 student 会不会更 expert-like”，而不是单步匹配。

3. **让 reward model 跟 student 的能力同步变化**  
   小 trainer buffer、更高 student 探索率、必要时冻结 trainer，都是在解决这个共适应系统的稳定性问题。

### 战略性取舍

| 设计选择 | 解决的瓶颈 | 能力收益 | 代价/风险 |
|---|---|---|---|
| 用 RL 训练 trainer，而不是直接用 discriminator 当奖励 | 静态相似度信号太粗 | 能形成阶段性 curriculum | 多了一个非平稳学习体，训练更难 |
| trainer 回报显式依赖“自己的输出是否和 discriminator 判断一致” | trainer 缺少即时 credit assignment | trainer 真正学会何时奖、何时罚 | 更依赖 discriminator 的校准质量 |
| 给 trainer 更小的 replay buffer | 旧数据让奖励更新滞后 | 更快跟住 student 的最近变化 | 方差更高，容易受噪声影响 |
| 提高 student 探索 | trainer 看到的状态空间太窄 | trainer 更容易学出有效 shaping | 早期训练更不稳定 |
| 中途冻结 trainer | 后期共适应容易过拟合/发散 | 提高收敛稳定性 | 后期失去继续适应能力 |

---

## Part III：证据与局限

### 关键证据信号

#### 1. 机制证据：RILe 的奖励确实是“动态的”，不是口头宣称

在 maze 实验中，RILe 的奖励图会随着训练阶段移动：

- 早期鼓励往某些暂时次优但有前景的区域探索
- 中期把高奖励区域推进到下一个子目标
- 后期把奖励集中到接近目标的位置

相比之下，GAIL/AIRL 的奖励景观更静态。  
这说明 RILe 真正改变的是**学习信号的时变性**，而不只是换了个训练框架。

#### 2. 消融证据：动态奖励不是“乱动”，而是和性能提升相关

论文设计了三类指标：

- **RFDC**：奖励分布整体变化幅度
- **FS-RFDC**：固定状态上奖励值的变化
- **CPR**：奖励变化与 student 性能变化的相关性

结论很清楚：

- RILe 的奖励函数变化幅度高于 GAIL 类方法
- DRAIL-RILe 的 CPR 最高，RILe 也明显优于 GAIL
- 说明 RILe 的 reward adaptation **不是噪声，而是和性能增益对齐的变化**

#### 3. 能力跳跃：优势主要出现在高维、难探索任务

**MuJoCo Humanoid-v2** 上：

- RILe: **5928**
- GAIL: 5709
- AIRL: 5623
- IQ-Learn: 327

这说明在高维 humanoid 控制里，RILe 的 adaptive reward 确实带来实质增益。

**LocoMujoco 高维 mocap 模仿**上，优势更明显。  
例如 Walk-Humanoid：

- RILe: **831.3**
- GAIL: 181.4
- AIRL: 80.1

而且 RILe 还能和更强判别器组合：  
Walk-UnitreeH1 上 **DRAIL-RILe = 995.8 / 1000**，接近专家。

这很重要，因为它说明：

> RILe 的核心价值不只是“单模型得分更高”，而是它提供了一个可插拔的 reward-learning 框架。

#### 4. 鲁棒性证据：在噪声示范下掉得更慢

Humanoid-v2 上加入专家示范噪声时，RILe 始终优于基线。  
最显著的一组是动作噪声 Σ=0.5：

- RILe: **5154**
- GAIL: 902
- AIRL: 4589

此外，在 covariate shift 测试里，冻结后学到的 RILe 奖励也比 AIRL 更稳。  
这支持论文的另一个重要点：**自适应奖励不仅更强，也更抗扰动。**

### 这篇论文最值得记住的结论

RILe 的能力提升并不是“平均每个任务都稍微涨一点”，而是：

**把 imitation signal 从一个静态裁判，变成一个会跟着 student 阶段变化的教练。**

这种提升在以下场景最明显：

- 高维控制
- 需要探索中间子目标
- 示范带噪声
- 纯相似度信号不够用的任务

### 局限性

- Fails when: 在一些相对没那么依赖动态 shaping 的任务上，RILe 并非总是最优，例如 Walker2d-v2 上 RILe 4435，低于 GAIL 4906 和 AIRL 4823；在部分复杂任务如 Carry-Talos 上，plain RILe 也明显落后于 DRAIL 变体，说明它的上限仍受判别器质量影响。
- Assumes: 需要在线环境交互、专家示范、可训练判别器，以及较多稳定化技巧共同配合（如 trainer 冻结阈值、较小 trainer buffer、更高 student exploration）；正文主公式主要基于 state-action pair，虽然实验覆盖了 state-only mocap 场景，但适配细节并不是正文主线。
- Not designed for: 纯离线模仿、无环境 rollout 的学习场景、无法稳定训练判别器的设置，以及论文未验证的离散动作/非机器人通用任务。

### 复现与依赖

- 训练依赖三方共适应：student / trainer / discriminator，系统比 GAIL/AIRL 更容易不稳定。
- 论文明确指出 **RILe 对 discriminator 容量和更新速度很敏感**；判别器太强会过拟合，反而让 trainer 学不到有信息量的信号。
- 资源上作者使用 **1×A100 GPU、AMD EPYC 7742、32GB 内存**。
- 文中未给出明确代码链接，因此虽然实验较完整，**复现门槛主要来自训练细节与超参，而不只是算力**。

### 可复用组件

- **trainer-as-reward-policy**：把奖励函数显式参数化成一个可学习策略，而不是固定打分器。
- **discriminator-to-trainer alignment**：不给 student 直接吃判别器分数，而是用判别器去校准奖励教师。
- **recent-data trainer buffer**：奖励模型优先看最近 student 行为，而不是过时样本。
- **freeze-trainer-midway**：在共适应系统中，用冻结策略换取后期稳定性。
- **plug-in discriminator**：RILe 可直接受益于更强判别器，如 DRAIL。

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_RILe_Reinforced_Imitation_Learning.pdf]]