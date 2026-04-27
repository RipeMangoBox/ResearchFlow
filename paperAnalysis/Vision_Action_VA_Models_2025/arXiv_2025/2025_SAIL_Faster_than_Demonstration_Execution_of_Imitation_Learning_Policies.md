---
title: "SAIL: Faster-than-Demonstration Execution of Imitation Learning Policies"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/robot-manipulation
  - task/offline-imitation-learning
  - diffusion
  - classifier-free-guidance
  - adaptive-speed-control
  - dataset/RoboMimic
  - dataset/MimicGen
  - opensource/no
core_operator: 以跟踪误差为门控，对动作块进行自适应一致性引导，并用 reached pose 目标、高保真跟踪、变速执行和延迟感知调度解除“示教速度绑定”。
primary_logic: |
  离线示教观测/机器人状态 + 当前跟踪误差与系统延迟
  → 误差自适应的一致性动作生成 + reached-pose 目标预测与高保真控制跟踪 + 关键阶段快慢速切换 + 过期动作调度
  → 在纯离线设置下实现快于示教的稳定 manipulation 执行
claims:
  - "在 RoboMimic/MimicGen 的 5 个仿真任务上，SAIL 在保持高成功率的同时把 speedup-over-demo 提升到最高 3.98×，例如 Lift 任务的 TPR 从 DP 的 0.46 提升到 1.68 [evidence: comparison]"
  - "在两套真实机器人 7 个任务上，SAIL 的 TPR 在 6/7 个任务上优于 DP-Fast，并在 Bimanual Serve 上将 TPR 从 1.00 提升到 5.40 [evidence: comparison]"
  - "使用 reached pose 作为训练/执行目标是关键因子：在 Square 任务上若改回 commanded pose，成功率会从 0.86 降到 0.31 [evidence: ablation]"
related_work_position:
  extends: "Diffusion Policy (Chi et al. 2023)"
  competes_with: "AWE (Shi et al. 2023); BID (Liu et al. 2024)"
  complementary_to: "ACT (Zhao et al. 2023)"
evidence_strength: strong
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_SAIL_Faster_than_Demonstration_Execution_of_Imitation_Learning_Policies.pdf
category: Embodied_AI
---

# SAIL: Faster-than-Demonstration Execution of Imitation Learning Policies

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2506.11948), [Project](https://nadunranawaka1.github.io/sail-policy)
> - **Summary**: 这篇工作把“离线模仿学习策略只能按示教速度执行”定义成一个全栈问题，并通过误差门控的一致性生成、reached-pose 跟踪、关键阶段变速和延迟感知调度，在不做在线强化学习的前提下实现快于示教的执行。
> - **Key Performance**: 仿真最高 4× speedup-over-demo；真实机器人最高 3.2×；Bimanual Serve 的 TPR 从 1.00 提升到 5.40。

> [!info] **Agent Summary**
> - **task_path**: RGB观测 + 机器人状态 / 纯离线模仿策略执行 -> 更快的末端位姿轨迹与任务完成
> - **bottleneck**: 提速后控制误差、观测-动作条件失配、动作块不连续与推理延迟共同造成 OOD 漂移和动作耗尽
> - **mechanism_delta**: 把“固定频率直接快放策略”改成“仅在跟踪可信时做条件引导 + 预测 reached pose 并高保真跟踪 + 对关键动作减速 + 对延迟做动作调度”
> - **evidence_signal**: 5个仿真任务与2套真实机器人/7任务的比较结果，加上组件消融，显示 TPR 和 SOD 持续提升
> - **reusable_ops**: [error-adaptive-guidance, controller-invariant-reached-pose]
> - **failure_modes**: [contact-dynamics-shift, high-gain-reference-noise-sensitivity]
> - **open_questions**: [how-to-model-high-speed-contact, how-to-auto-tune-thresholds-and-speed-bounds]

## Part I：问题与挑战

这篇论文要解决的不是“让轨迹播放得更快”这么简单，而是一个**时间尺度错配**问题：  
离线模仿学习策略是在示教频率下训练出来的，但机器人一旦以更短的执行间隔运行，控制器动力学、跟踪误差、观测分布、动作分布、推理延迟都会一起变化。

### 1. 真正的问题是什么？
给定一个在固定时间间隔 `δ*` 上训练的 visuomotor policy，能否在部署时用更快、且可随时间变化的间隔 `δ_t = c_t δ*` 来执行，同时仍保持较高成功率与更高任务吞吐？

这比普通 IL 难在于：  
- **策略不是学会“更快完成任务”**，而是学会了**在示教速度下该怎么动**。  
- 一旦提速，低层控制器的误差 profile 改变，机器人就会落到训练时没见过的状态。  
- receding-horizon/action-chunking 策略在高速度下更容易出现**相邻 action chunk 不一致**，表现为 jerk 或轨迹突变。  
- 真机里还有传感、同步、推理、控制带宽等物理约束，不能只靠算法层“口头加速”。

### 2. 输入/输出接口
论文采用典型的两层结构：  
- **输入**：视觉观测 `o_t`（如 wrist/front RGB）+ 机器人状态 `x_t`  
- **策略输出**：未来一段 action chunk，即末端执行器的 SE(3) 位姿序列 + gripper 命令  
- **控制器输出**：把这些高层动作目标转换成高频扭矩控制

因此，这项工作本质上在研究：**如何让“动作块策略 + 低层控制器 + 实时系统”在更快时间尺度上仍能协同。**

### 3. 真瓶颈在哪里？
作者把瓶颈拆成四个互相关联的问题：

1. **控制器动力学改变导致状态分布漂移**  
   更快执行会放大跟踪误差，观测进入 OOD，随后 policy 再输出更差动作，形成 compounding error。

2. **连续重规划时的动作不一致**  
   上一段还没执行完，下一段新预测就可能朝另一个方向走，高速下尤其明显。

3. **任务不同阶段对速度敏感度不同**  
   抓取、插入、对齐等精细阶段不能和大范围 free-space reaching 一样快。

4. **系统延迟给速度上了硬上限**  
   推理没回来之前如果动作已经执行完，就会 pause，吞吐反而下降。

### 4. 为什么现在值得解决？
因为离线 IL 在 manipulation 上已经能学到复杂技能，但**示教通常很慢**。  
如果机器人只能复现“人示教时的速度”，那它的工业吞吐上限就被人类示教过程锁死了。  
所以这篇论文的价值，不是再提高一点成功率，而是把 IL 从“能做”推进到“做得快、产能高”。

---

## Part II：方法与洞察

SAIL 不是单一模块，而是一个 full-stack 系统，核心由四部分组成：

1. **Error-Adaptive Guidance (EAG)**：基于跟踪误差决定是否对下一段动作做条件引导  
2. **Controller-invariant action target**：训练时预测 reached pose，而不是 teleop commanded pose  
3. **Adaptive speed modulation**：根据动作复杂度与 gripper 事件动态切换快/慢速度  
4. **Latency-aware action scheduling**：显式处理 sensing/inference delay，避免动作耗尽

### 核心直觉

**它真正改变的不是 policy 网络本身，而是“什么该被预测、什么时候该信前一段计划、什么阶段该快、系统最多能快到哪”。**

- **从“总是信任上一段动作尾部”到“只在跟踪误差小的时候信任它”**  
  → 改变了 action condition 的可靠性假设  
  → 减少 observation/action condition 失配带来的错误引导  
  → 相邻动作块更连续，轨迹更平滑

- **从“模仿 teleoperator 发出的 commanded pose”到“模仿机器人实际达到的 reached pose”**  
  → 改变了监督目标对 teleop controller dynamics 的依赖  
  → 降低 controller shift 导致的状态分布变化  
  → 提速后更容易保持在 policy 熟悉的状态附近

- **从“全程固定加速”到“按阶段变速，并服从延迟上限”**  
  → 把精度需求与系统物理约束显式加入执行层  
  → 精细操作阶段保成功，简单移动阶段保吞吐  
  → 避免 pause 和动作耗尽，使高平均速度可持续

一句话概括：**SAIL 不是把 policy 硬跑快，而是把 policy、controller、scheduler 重新对齐到一个新的时间尺度上。**

### 关键组件拆解

#### 1. EAG：误差自适应的一致性动作生成
作者从 CFG 得到启发：  
在重规划时，用上一轮计划的尾部作为条件，可以强制下一段动作与之前更一致。

但他们指出一个关键问题：  
**如果刚才那段动作因为高速执行已经产生较大跟踪误差，那么“上一段计划尾部”本身就可能是错条件。**  
这时继续强条件引导，只会把新预测进一步带偏。

所以 EAG 的做法很直接：  
- **跟踪误差小**：开启条件引导，强调连续性  
- **跟踪误差大**：关闭条件引导，只信当前观测下的无条件预测

这相当于把“平滑性”从硬约束改成**可信时才启用的软先验**。

#### 2. Reached pose 作为 controller-invariant target
标准 IL 常用 teleop 时的 commanded pose 作为监督目标。  
但这有个隐含假设：部署时控制器行为和采集数据时一致。

提速后，这个假设失效。  
因此 SAIL 改为让 policy 预测**机器人实际达到的 reached pose**。这个目标更“可实现”，也更少绑定到 teleop 控制器本身。  
部署时再用高保真控制器去追踪这些 reached pose。

这一步的因果作用很强：  
它把“policy 学到的动作目标”从一个**依赖示教控制器动态**的信号，变成一个**更接近机器人实际可达轨迹**的信号。

#### 3. Adaptive speed modulation：关键阶段慢，其他阶段快
作者并不追求所有时刻都最快，而是让速度成为时变量。  
关键动作通过两种方式检测：
- **motion complexity**：离线分析 demonstration waypoint 的空间复杂度
- **gripper event**：运行时检测抓手开合变化，作为交互相位的廉价代理

然后在 `c_fast` 与 `c_slow` 间切换。  
这使系统在抓取、对齐、放置等高精度阶段不过度冒进。

#### 4. Latency-aware scheduling：把延迟当成一等公民
真实系统中，推理不可能瞬时完成。  
如果等新 action chunk 回来再动，机器人就会停顿；如果跑得太快，旧动作还没衔接上新动作就会“用完”。

SAIL 的做法是：
- 推理进行时继续执行上一轮计划
- 新计划到达后，丢弃已经过期或未来不再合理的旧动作
- 同时给执行间隔设置一个**由延迟和预测 horizon 共同决定的下界**

这意味着：  
**速度不是想设多快就多快，而是必须服从可持续闭环控制的物理极限。**

### 设计取舍

| 组件 | 解开的瓶颈 | 带来的能力 | 代价 / 风险 |
|---|---|---|---|
| EAG | 相邻动作块不一致、条件失配 | 高速下更平滑、更少 jerk | 需要误差阈值；阈值过宽会把错误条件也引进来 |
| Reached pose + 高保真控制 | controller shift 导致的 OOD | 提速后仍能稳定跟踪、减少分布漂移 | 高增益控制对噪声参考更敏感 |
| 自适应变速 | 精度阶段与吞吐阶段需求冲突 | 兼顾成功率和平均速度 | 需要关键动作检测与任务相关速度预设 |
| 延迟感知调度 | 推理延迟导致 pause / 动作耗尽 | 连续运动、可持续高吞吐 | 最快速度受硬件延迟和 horizon 限制 |

---

## Part III：证据与局限

### 关键证据

- **比较信号 / 仿真主结果**  
  在 RoboMimic + MimicGen 的 5 个任务上，SAIL 整体上比 DP、DP-Fast、AWE、BID-Fast 更能同时兼顾速度和成功率。  
  最典型的是：
  - **Lift**：TPR 从 DP 的 0.46 提升到 1.68，SOD 达到 3.98  
  - **Can**：TPR 从 0.18 提升到 0.51，SOD 达到 3.20  
  - **Stack**：TPR 从 0.19 提升到 0.66，SOD 达到 3.47  
  说明它不是单纯“更快但更容易失败”，而是**吞吐真实提升**。

- **分析/消融信号 / EAG 的因果支持**  
  论文不仅做了结果比较，还专门验证了 EAG 背后的因果链：  
  1. 高速执行会让 action condition 更容易掉出 unconditional action distribution；  
  2. 跟踪误差与 KDE/kNN/MMD 这类 OOD 分数显著相关；  
  3. 因此用 tracking error 作为 CFG 开关是有效代理。  
  同时，去掉一致性组件后性能下降，说明“平滑参考轨迹”对高增益控制是必要条件。

- **消融信号 / reached pose 与变速策略的重要性**  
  在 Table K.6 中，若把目标从 reached pose 换回 commanded pose，**Square 成功率从 0.86 直接掉到 0.31**；  
  去掉 adaptive speed 时，精细任务如 Square、Mug 也明显受损。  
  这支持作者的主张：**高速度执行失败，不只是 policy 不够好，而是 target、controller、speed schedule 都得一起改。**

- **比较信号 / 真实机器人**  
  在两套真实平台的 7 个任务上，SAIL 的 **TPR 在 6/7 个任务上优于 DP-Fast，SOD 也在 6/7 个任务上更高**。  
  代表性结果：
  - **Plate Fruits**：TPR 2.22 → 5.46，ATR 13.74s → 8.53s  
  - **Pack Chicken**：TPR 0.51 → 5.22，ATR 17.33s → 9.40s  
  - **Bimanual Serve**：TPR 1.00 → 5.40，SR 0.40 → 0.70  
  这说明它不只在仿真有效，也确实能跨平台工作。

### 1-2 个最关键指标怎么读？
- **TPR (Throughput-with-Regret)**：最关键，因为它同时奖励“更快成功”并惩罚“失败”。  
  对这类论文，单看速度或单看成功率都不够，TPR 更接近真实产线吞吐目标。
- **SOD (Speedup-over-Demo)**：直接回答论文核心问题——到底有没有“快于示教”。

### 局限性

- **Fails when**: 高速下机器人-物体接触动力学显著变化的任务，尤其是持续接触或强动量效应场景；论文里 wiping board 是明显反例，模拟里的 can 甚至可能因高速动量被“甩飞”。
- **Assumes**: 需要纯离线示教数据、可用的高保真/高频控制器、可测量的系统延迟、足够长的 action horizon，以及任务相关的误差阈值与快/慢速度预设；复现上还依赖实时控制栈和多模态时间同步，且文中未声明代码开放。
- **Not designed for**: 显式建模高速接触动力学、在部署时通过在线探索学出更优速度策略、或在控制带宽/推理延迟不足的硬件上无约束提速。

### 可复用组件

1. **Tracking-error-gated guidance**：任何 action-chunking 生成式 policy 都可借鉴  
2. **Reached-pose supervision**：把监督目标改成更控制器不变的执行结果  
3. **Phase-dependent speed gating**：用动作复杂度/抓手事件切换快慢速  
4. **Latency-aware action scheduling**：把“实时延迟”纳入执行层，而不是事后补救

## Local PDF reference

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_SAIL_Faster_than_Demonstration_Execution_of_Imitation_Learning_Policies.pdf]]