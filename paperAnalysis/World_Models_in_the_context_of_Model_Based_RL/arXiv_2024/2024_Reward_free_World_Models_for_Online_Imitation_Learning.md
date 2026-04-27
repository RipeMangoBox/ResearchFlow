---
title: "Reward-free World Models for Online Imitation Learning"
venue: ICML
year: 2025
tags:
  - Embodied_AI
  - task/online-imitation-learning
  - world-model
  - model-predictive-control
  - inverse-soft-q-learning
  - dataset/DMControl
  - dataset/MyoSuite
  - dataset/ManiSkill2
  - opensource/no
core_operator: 用无奖励潜空间世界模型把逆模仿学习改写到Q-策略空间，并在MPC中直接从Q值解码奖励进行规划控制
primary_logic: |
  专家状态-动作示范 + 在线环境交互 → 在潜空间联合学习编码器/动力学/Q函数/策略先验，并用 inverse soft-Q 从Q中恢复稠密奖励 → 通过 MPPI 在潜动力学上规划动作，得到稳定接近专家的在线模仿策略
claims:
  - "Claim 1: IQ-MPC 在 DMControl、MyoSuite 和 ManiSkill2 的在线模仿学习任务上整体优于或匹配 IQ-Learn+SAC、CFIL+SAC 和 HyPE，且在灵巧手操作上优势显著 [evidence: comparison]"
  - "Claim 2: 将 inverse soft-Q 目标中的行为分布价值差改写为初始状态价值项，可带来更稳定的 Q 估计与训练过程，并在 Humanoid Walk 等高维任务中改善收敛 [evidence: ablation]"
  - "Claim 3: 从 Q 函数解码的奖励与环境真值奖励呈显著正相关，在 4 个 DMControl 任务上的 Pearson 相关系数达到 0.87-0.93，且高于 IQL+SAC [evidence: comparison]"
related_work_position:
  extends: "IQ-Learn (Garg et al. 2021)"
  competes_with: "HyPE (Ren et al. 2024); CFIL (Freund et al. 2023)"
  complementary_to: "TD-MPC2 (Hansen et al. 2023)"
evidence_strength: strong
pdf_ref: paperPDFs/World_Models_in_the_context_of_Model_Based_RL/arXiv_2024/2024_Reward_free_World_Models_for_Online_Imitation_Learning.pdf
category: Embodied_AI
---

# Reward-free World Models for Online Imitation Learning

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2410.14081)
> - **Summary**: 这篇工作把在线模仿学习里不稳定的“学奖励/学策略”过程，替换成“无奖励潜空间世界模型 + inverse soft-Q critic + MPC规划”，从而在没有环境奖励的情况下也能稳定完成高维连续控制与操作任务。
> - **Key Performance**: MyoSuite 上 Object Hold 成功率 0.96、Pen Twirl 成功率 0.73；4 个 DMControl 任务的奖励恢复 Pearson 相关达到 0.87–0.93

> [!info] **Agent Summary**
> - **task_path**: 专家状态-动作示范 + 无奖励在线环境交互 -> 连续控制动作/在线模仿策略
> - **bottleneck**: 高维观测/动作与复杂动力学下，传统在线 IL 在 reward-policy min-max 中训练不稳，且缺少可用于规划的环境动态模型来纠正分布外偏移
> - **mechanism_delta**: 用 decoder-free 潜空间世界模型承载 inverse soft-Q 学习，在 Q-策略空间训练 critic，并在 MPC 中直接由 Q 解码奖励做无梯度规划
> - **evidence_signal**: 跨 DMControl、MyoSuite、ManiSkill2 的比较实验显示其更稳定接近专家，且目标改写与梯度惩罚的消融实验支持稳定性来源
> - **reusable_ops**: [Q值解码奖励, 潜空间一致性动力学学习]
> - **failure_modes**: [低维观测或动作空间下 critic 过强导致训练不稳, 专家演示极少时收敛明显变慢且可能失稳]
> - **open_questions**: [如何在更强随机动力学下保持稳定规划, 如何迁移到真实机器人并降低 MPC 计算开销]

## Part I：问题与挑战

这篇论文解决的是一个比普通行为克隆更难的设定：

- **只有专家示范，没有环境奖励**
- **允许在线交互**
- **任务是高维连续控制/复杂动力学**
- **输入既可能是状态，也可能是视觉观测**

### 真正的问题是什么？

离线 BC 的主要问题不是“学不会专家动作”，而是**一旦离开专家分布就不会纠错**。  
在高维控制里，这个问题尤其严重：一个小偏移就会进入专家没覆盖过的状态，随后误差不断累积。

已有在线 IL 方法虽然引入了价值或奖励估计，但仍有两个根本瓶颈：

1. **优化瓶颈**：很多方法在 reward-policy 空间做 min-max，对高维任务很容易不稳定。
2. **动态理解瓶颈**：即使学到某种“像专家”的局部动作选择，没有环境动力学模型，就难以做前瞻规划，也难在偏离后拉回轨迹。

### 输入/输出接口

- **输入**：专家状态-动作轨迹；在线交互得到的行为轨迹；可选视觉输入
- **输出**：连续控制动作
- **执行形态**：训练时学 policy prior；推理时结合潜空间世界模型做 MPC 规划
- **边界条件**：主要面向连续控制、可在线交互、以模拟环境为主的任务；不是纯离线 IL，也不是极低算力部署场景

### 为什么现在值得解？

因为 world model 已经成熟到足以处理这类问题。  
尤其是 **decoder-free latent world models**（如 TD-MPC 系列）说明：在高维观测下，未必要重建像素，只要潜空间动态足够可预测，就可以支持有效控制与规划。

这给了本文一个明确机会点：

> 把 **reward-free imitation learning** 和 **latent world model planning** 合起来，既绕开显式奖励建模，又获得对未来后果的可规划表示。

---

## Part II：方法与洞察

论文的方法叫 **IQ-MPC**。直观上，它是把 **IQ-Learn 的 inverse soft-Q 思路** 嵌进 **TD-MPC 风格的潜空间世界模型** 里。

### 方法骨架

#### 1. 一个无奖励的潜空间世界模型

模型包含四个核心部件：

- **Encoder**：把观测压缩到 latent state
- **Latent dynamics**：在潜空间预测下一步状态
- **Q function**：评估状态-动作价值
- **Policy prior**：提供一个可学习的动作先验，帮助规划

关键点在于：

- **不学 reward model**
- **不做观测重建**
- **只学对控制足够的潜空间动态**

这让方法更像“为规划服务的表示学习”，而不是“为生成观测服务的建模”。

#### 2. 把逆模仿学习搬到 Q-策略空间

作者的核心观察是：在 inverse soft-Q learning 框架下，**reward 和 Q 在给定策略下可以互相映射**。  
所以，与其单独再训练一个 reward model，不如直接让 **Q 兼任价值估计器和隐式奖励恢复器**。

这一步的意义非常大：

- 以前：在 reward-policy 空间里做对抗/双层优化，容易不稳
- 现在：在 Q-policy 空间里训练，更适配 actor-critic 和 world model 体系

也因此，本文叫“reward-free world models”：  
**不是任务不需要奖励概念，而是不再显式建模奖励头。**

#### 3. 用一致性学习保证 latent rollout 可用

世界模型训练时，作者把重点放在：

- 短视界 latent rollout 的一致性
- Q 学习的稳定性

而不是像传统视觉 world model 那样强调像素重建。

因果上这很合理：  
如果目标是控制与规划，真正重要的是“潜空间里未来是否可预测”，而不是“重建图像是否逼真”。

#### 4. 规划时直接从 Q 解码奖励

执行阶段用 MPPI 做 MPC。  
流程是：

- 从当前 latent state 出发
- 采样多条候选动作序列
- 用 learned dynamics rollout
- 每一步的“奖励”直接由 **Q 与下一状态 value** 解码
- 再结合 terminal value 给出整条轨迹分数
- 输出最优动作

因此，Q 不只是训练信号，而是**真正被用于在线控制的任务进度信号**。

#### 5. 两个稳定化手段

- **初始状态改写**：把原始 inverse soft-Q 目标中的行为分布 value difference，换成初始状态 value 项。作者证明两者等价，但经验上前者更稳定。
- **Wasserstein-1 gradient penalty**：当 critic 太容易区分 expert 和 behavior 时，会把 policy 学坏。这个惩罚帮助约束 critic 的判别强度，尤其在部分 manipulation 任务中有效。

### 核心直觉

这篇论文真正调的“旋钮”不是某个新网络，而是：

> **把“任务进度信号”从显式 reward model 改成 critic 内生出来的 reward，并让这个信号在世界模型里可前瞻地被规划使用。**

#### 变化链条

- **What changed**：  
  显式 reward 建模/对抗优化  
  → 改成 Q-space imitation objective + reward decoding + latent MPC

- **Which bottleneck changed**：  
  - 减少了 reward-policy min-max 的不稳定性  
  - 让 agent 获得了“动作会导致什么后果”的潜空间动态知识  
  - 让在线交互不只是采样数据，还能通过规划主动纠偏

- **What capability changed**：  
  从“只会在专家分布附近模仿”  
  变成“偏离后也能依据动态模型和隐式奖励重新规划回去”

#### 为什么这套设计有效？

因为它把在线 IL 的两个核心需求统一了：

1. **要知道什么是更像专家的方向**  
   —— 由 inverse soft-Q critic 提供
2. **要知道往这个方向走会发生什么**  
   —— 由 latent world model 提供

这两者一结合，模仿学习就从“监督学习动作”升级成了“可规划的任务恢复”。

此外，论文的理论分析也服务于这个直觉：

- critic/policy 目标在压缩 expert 与当前策略的分布差距
- consistency loss 在压缩 learned dynamics 与真实 dynamics 的偏差

于是两部分一起在收紧“当前策略回报落后于专家回报”的上界。

### 战略权衡

| 设计选择 | 改变的瓶颈 | 带来的能力 | 代价/风险 |
|---|---|---|---|
| 不单独学习 reward model，而是从 Q 解码 reward | reward-policy 空间训练不稳；需要额外 reward 监督 | 真正 reward-free 的在线 IL/IRL | 强依赖 Q 估计质量 |
| decoder-free latent dynamics | 高维输入建模成本高 | 更高效，能扩展到视觉与高维控制 | latent 表示可解释性弱 |
| MPC + policy prior | 纯 actor 容易在分布外状态失误 | 在线纠错与局部最优动作搜索 | 训练/推理开销高于 model-free |
| 初始状态改写 + gradient penalty | critic / policy 失衡导致不稳 | 更稳定的 Q 收敛与策略学习 | 需要额外调参与任务特化 |

---

## Part III：证据与局限

### 关键实验信号

- **比较信号：跨基准整体更稳**
  - 在 DMControl locomotion 中，IQ-MPC 在 Hopper、Walker、Humanoid 等任务上更稳定地接近专家。
  - 在 Cheetah Run 和 Quadruped Run 上与 HyPE 大致相当，但整体曲线稳定性更好。
  - 这说明它的收益不是只来自某一类任务，而是在高维连续控制上普遍成立。

- **最强证据：MyoSuite 灵巧手操作**
  - 这是论文最有说服力的一组结果。
  - **Object Hold 成功率 0.96**
  - **Pen Twirl 成功率 0.73**
  - 对比基线，在这两项上几乎都接近 0。
  - 这表明：当动作维度高、接触动力学复杂时，world model + planning 的价值远大于纯 model-free 在线 IL。

- **视觉证据：只换 encoder 也能工作**
  - 作者仅把 encoder 换成浅层 CNN，其余框架基本不变。
  - 在视觉版 DMControl 上，优于 IQL+SAC(Visual) 于 Cheetah Run、Walker Run，并在 Walker Walk 上可比。
  - 这支持了一个重要判断：本文的收益主要来自“规划式 imitation 机制”，不是来自特定状态输入工程。

- **消融信号：稳定性来源可定位**
  - 把目标改写成初始状态 value 形式后，Humanoid Walk 上的 Q 估计和训练过程更稳。
  - 在 ManiSkill2 Pick Cube 上，加 gradient penalty 后成功率从 **0.51 提升到 0.79**。
  - 少量示范下依然能学：Hopper Hop 用 10 条专家轨迹、Object Hold 用 5 条专家轨迹仍可到达接近专家水平，但收敛会更慢。

- **能力外推信号：reward recovery 不是空谈**
  - 从 Q 解码出的 reward 与环境真值 reward 在 4 个 DMControl 任务上的 Pearson 相关达到 **0.87–0.93**。
  - 并且都高于 IQL+SAC。
  - 这说明“critic 兼做 reward 恢复器”在经验上站得住脚，而不只是理论包装。

### 局限性

- **Fails when**: 低维观测或低维动作空间下，critic 更容易过强地区分 expert 和 behavior，导致 policy 学习失衡；专家演示极少时会明显变慢，部分任务会失稳；更强随机转移下虽有一定鲁棒性，但本文主要不是为高随机性环境设计。
- **Assumes**: 需要专家状态-动作示范；需要在线环境交互；需要一个能学准的 latent dynamics；依赖 MPC 规划带来的额外计算；部分 manipulation 任务还需要额外 gradient penalty 才稳定。
- **Not designed for**: 纯离线 imitation learning；无交互设定；真实机器人部署级低时延控制；强随机、超长时域、严重部分可观测但未验证的场景。

### 复现与资源假设

- 文中**未给出代码链接**，因此前端实现细节与 appendix 超参较重要。
- 训练时间分析表明：
  - **比 model-free baseline 更慢**
  - **比另一个 model-based baseline HyPER 更快**
- 作者还指出：只用 policy prior 与环境交互会更快，但稳定性更差。  
  这说明其核心优势确实和 **planning** 绑定，而不是“世界模型训练后自然就行”。

### 可复用组件

这篇论文里最值得迁移的不是整套系统，而是以下三个操作：

1. **Q 值直接解码奖励**  
   适合任何想去掉显式 reward head 的 IRL / IL world model 方案。
2. **基于初始状态分布的 inverse soft-Q 改写**  
   可作为稳定 critic 学习的通用 trick。
3. **policy prior 引导的 latent MPC**  
   适合在高维控制中把“学习到的策略”与“在线局部规划”结合起来。

---

## Local PDF reference

![[paperPDFs/World_Models_in_the_context_of_Model_Based_RL/arXiv_2024/2024_Reward_free_World_Models_for_Online_Imitation_Learning.pdf]]