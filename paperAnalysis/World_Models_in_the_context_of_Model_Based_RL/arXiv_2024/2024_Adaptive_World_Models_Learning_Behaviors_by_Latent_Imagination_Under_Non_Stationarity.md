---
title: "Adaptive World Models: Learning Behaviors by Latent Imagination Under Non-Stationarity"
venue: arXiv
year: 2024
tags:
  - Embodied_AI
  - task/non-stationary-reinforcement-learning
  - task/continuous-control
  - bayesian-aggregation
  - rssm
  - latent-task-conditioning
  - dataset/HalfCheetah
  - dataset/Hopper
  - dataset/Walker
  - opensource/no
core_operator: 用近期转移集合经深度集合编码和贝叶斯聚合推断任务潜变量，并将其条件化到RSSM与actor-critic中，以在非平稳环境下进行自适应潜空间想象控制
primary_logic: |
  最近N步交互上下文 → 深度集合编码并以不确定性加权的贝叶斯聚合推断任务后验l → 用l条件化世界模型、价值函数与策略进行latent imagination与策略优化 → 输出随动力学或目标变化而自适应的连续控制行为
claims:
  - "在 HalfCheetah、Hopper 和 Walker 的 inter-/intra-episodic 非平稳基准上，HiP-Dreamer 的平均回报整体高于 Vanilla Dreamer，并在多种动力学变化场景接近 Oracle 上界 [evidence: comparison]"
  - "在目标速度变化和 DMC 多技能任务中，Vanilla Dreamer 普遍难以稳定适应，而显式任务潜变量可显著恢复学习效果 [evidence: comparison]"
  - "任务潜变量会把世界模型的 latent state 与 latent task space 组织成更按任务分离的结构；无条件 Dreamer 在奖励变化场景出现更强的任务干扰 [evidence: analysis]"
related_work_position:
  extends: "Dreamer (Hafner et al. 2020)"
  competes_with: "Dreamer (Hafner et al. 2020); TD-MPC2 (Hansen et al. 2024)"
  complementary_to: "Multi Time Scale World Models (Kumar et al. 2023); On Uncertainty in Deep State Space Models for MBRL (Becker & Neumann 2022)"
evidence_strength: moderate
pdf_ref: paperPDFs/World_Models_in_the_context_of_Model_Based_RL/arXiv_2024/2024_Adaptive_World_Models_Learning_Behaviors_by_Latent_Imagination_Under_Non_Stationarity.pdf
category: Embodied_AI
---

# Adaptive World Models: Learning Behaviors by Latent Imagination Under Non-Stationarity

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2411.01342)
> - **Summary**: 这篇工作把“环境/任务在变”从普通隐状态里拆出来，显式建成一个可由最近交互在线推断的任务潜变量，再用它统一条件化 world model 与 actor-critic，从而让 Dreamer 类 agent 在非平稳连续控制中更稳健地适应变化。
> - **Key Performance**: 评测指标为 mean return；在 HalfCheetah/Hopper/Walker 的多种非平稳设定下整体优于 Vanilla Dreamer，在多种动力学变化中接近 Oracle，并在 DMC 多技能目标变化上明显更稳。

> [!info] **Agent Summary**
> - **task_path**: 最近N步交互上下文 + 当前连续控制观测 -> 任务潜变量推断 + 潜空间想象 -> 自适应动作策略
> - **bottleneck**: 标准 POMDP/RSSM 把“当前任务模式”和“瞬时环境状态”混在同一 latent state 中，导致任务干扰与适配迟缓
> - **mechanism_delta**: 用 HiP-POMDP 将非平稳性显式建模为额外潜变量 l，并用 deep set + Bayesian aggregation 在线推断，再条件化 RSSM、actor、critic
> - **evidence_signal**: 在目标/奖励变化和多技能基准上，任务条件化版本稳定超过 Vanilla Dreamer，后者常明显失效
> - **reusable_ops**: [deep-set上下文编码, 贝叶斯上下文聚合]
> - **failure_modes**: [目标在很短频率内剧烈切换时任务推断滞后, 复杂多技能任务中部分任务潜表示混叠]
> - **open_questions**: [如何引入遗忘机制以更快适配突变任务, 如何扩展到图像和点云等高维观测]

## Part I：问题与挑战

这篇论文要解决的核心不是“让 Dreamer 更大”，而是**非平稳环境下 latent world model 的结构失配**。

传统 Dreamer 一类方法基于 POMDP：理论上，慢变化的任务也可以被吸收到隐状态里；但实践上，这会让同一个 latent 同时承担两件事：

1. 表示**瞬时物理状态**；
2. 隐式编码**当前任务/动力学/奖励模式**。

这在非平稳 RL 里会带来两个直接问题：

- **动力学变化**时：模型要一边估计当前状态，一边猜“这个世界现在是哪种物理参数”。
- **目标/奖励变化**时：同样的状态可能对应完全不同的价值语义，latent state 会出现更强的任务干扰。

论文的判断是：**真正的瓶颈不是 world model 不能做 imagination，而是 imagination 所依据的 latent 空间没有把“任务模式”单独抽出来。**

### 输入/输出接口与边界条件

- **输入**：当前 proprioceptive observation，以及最近 \(N\) 步转移上下文 \((o,a,r,o')\) 的集合。
- **输出**：连续控制动作。
- **非平稳形式**：
  - inter-episodic：任务在 episode 间切换；
  - intra-episodic：任务在 episode 内切换。
- **边界条件**：
  - 不提供真实任务标签；
  - 主要处理观测、动力学、奖励随任务变化的情形；
  - 短 imagination horizon 内假设任务潜变量近似不变。

### 为什么现在值得做

因为 world model 正被视为 embodied intelligence 的关键基础设施，但现有主流 latent MBRL 往往还是围绕**单任务、近似平稳**环境设计。真实机器人和开放环境里的变化却是常态，不解决这个问题，world model 很难成为“可适应”的基础模型。

## Part II：方法与洞察

### 方法主线

作者提出 **HiP-POMDP**（Hidden Parameter-POMDP）形式化：把非平稳性显式表示为一个额外的潜变量 \(l\)，使得**转移、观测、奖励**都依赖于这个任务变量。

然后基于 Dreamer/RSSM，做了三件事：

1. **任务推断**  
   从最近的上下文集合 \(C_l=\{(o,a,r,o')\}_{n=1}^N\) 中，用 deep set encoder 对每个转移编码，输出一个任务证据表示及其不确定性；再通过**闭式贝叶斯聚合**得到 \(p(l|C_l)\)。

2. **自适应世界模型**  
   用推断出的 \(l\) 去条件化 RSSM 的状态推断、转移、重构和奖励预测。于是模型学的就不再只是“统一的一个世界”，而是“**给定当前任务模式下的世界**”。

3. **自适应行为学习**  
   actor 和 critic 同样条件化在 \(l\) 上。这样 latent imagination 时，策略优化面对的是“当前任务下的 imagined future”，而不是混合了多任务语义的 latent rollout。

### 核心直觉

原来的 Dreamer 更像是在学：

- `history -> 一个混合 latent state -> 预测未来/做决策`

现在作者改成：

- `最近上下文 -> 任务潜变量 l`
- `history + l -> 条件化 latent state`
- `state + l -> 策略/价值`

这带来的因果变化是：

- **改变了什么**：把“慢变化的任务因素”从普通状态变量里拆出，单独建模成 \(l\)。
- **哪个瓶颈被改变**：原本被混在一起的任务不确定性，变成了显式可推断的条件变量；于是 latent dynamics 和 reward mapping 的条件分布更简单、更少多模态。
- **能力发生了什么变化**：world model 能更快识别“现在是什么任务/物理模式”，actor-critic 能在更正确的 dynamics/reward 语义下做 imagination，因此在非平稳环境里更容易适配。

再直白一点：  
**他们不是让模型“记住更多历史”，而是让模型先回答“我现在身处哪种任务模式”，再去预测和行动。**

### 为什么这个设计有效

- **贝叶斯聚合 = 不确定性感知的上下文选择**  
  每个转移都带一个方差，聚合时更“可信/更有信息量”的转移权重更大。作者把这看作一种线性复杂度的 probabilistic attention。
  
- **任务条件化降低表示干扰**  
  尤其在奖励变化场景里，同样状态在不同任务下回报不同；若没有 \(l\)，reward predictor 和 value learning 会互相拉扯。条件化后，语义被拆开了。

- **滑动窗口在线重推断**  
  交互时上下文 buffer 持续更新，agent 可以随环境变化重新估计 \(l\)，因此支持 inter-episodic 和 intra-episodic 适配。

### 战略取舍

| 设计选择 | 解决的问题 | 收益 | 代价/边界 |
|---|---|---|---|
| 显式任务潜变量 \(l\) | state/task 混叠 | 减少任务干扰，提升适配性 | 需要足够有信息的上下文 |
| deep set + Bayesian aggregation | 从短历史中稳定推断任务 | 线性复杂度、可用不确定性加权 | 快速突变时可能推断滞后 |
| 短 rollout 内固定 \(l\) | 保持 imagination 稳定 | 优化更简单 | 默认任务在短时窗内近似平稳 |
| 全栈条件化（model+actor+critic） | 只改模型不改策略会失配 | 预测与决策语义一致 | 对 backbone 耦合更强 |

## Part III：证据与局限

这篇论文最重要的“能力跳跃”其实出现在 **目标/奖励变化** 上，而不只是动力学变化上。

因为很多动力学变化下，Vanilla Dreamer 仍有机会把变化硬塞进 latent state 里勉强适应；但一旦**奖励语义发生变化**，同一状态对应的价值就不再稳定，显式任务潜变量的价值才会被明显放大。

### 关键证据信号

- **信号 1｜comparison：动力学变化基准**
  - 在 HalfCheetah 的 joint perturbation、Hopper 的 body mass/inertia 变化等场景里，HiP-Dreamer 比 Vanilla Dreamer 更稳，尤其在 intra-episodic 变化下优势更明显。
  - 说明：显式任务推断确实有助于在线适配变化的物理模式。

- **信号 2｜comparison：目标变化与多技能任务**
  - 在 target velocity 变化、DMC multi-task 技能切换中，Vanilla Dreamer 明显失效，而任务条件化版本能恢复大部分性能。
  - 这是论文最强证据，因为它直接支持作者关于“奖励变化更容易导致 latent 干扰”的核心论点。

- **信号 3｜analysis：latent space 可视化**
  - 任务条件化后，latent task space 和 latent state space 更按任务分离；Vanilla Dreamer 在奖励变化场景里更容易出现任务混叠。
  - 这不是单纯“看起来更好”，而是在解释**为什么回报提升**：表示空间更结构化，reward/value 学习更少互相污染。

- **信号 4｜analysis：失败点分析**
  - 当 target velocity 每 200 环境步就切换时，任务推断 Dreamer 也会出现断点。
  - 这表明当前聚合机制更适合“短期内近似稳定”的任务，而不擅长极快、剧烈的任务跃迁。

### 1-2 个关键指标

- **主指标**：mean return  
- **评测方式**：每 25 个 epoch 用 10 条轨迹评估，结果再对 10 个随机种子平均

### 局限性

- **Fails when**: 目标在很高频率下突变（如每 200 步切换 target velocity）时，滑动窗口上下文对新任务的后验更新不够快；在高难多技能任务中，部分任务仍会在潜空间中合并，导致性能与 Oracle 存在差距。
- **Assumes**: 主要使用 proprioceptive 输入；假设最近少量转移足以识别当前任务；假设短 imagination horizon 内任务近似固定；方法实现强依赖 Dreamer/RSSM 风格 backbone。
- **Not designed for**: 图像/点云等高维感知输入的非平稳适配；几乎零上下文的一步式任务识别；显式建模长期连续漂移的任务演化过程。

### 复现与可扩展性的现实约束

- 论文未给出开源代码，复现门槛偏高。
- 训练依赖 Dreamer 式 world model 管线与较完整的工程实现。
- 作者明确提到使用 HPC 资源，并且**没有进行 extensive hyperparameter optimization**，意味着结果有一定工程敏感性。

### 可复用组件

- **deep-set context encoder**：把最近转移编码成任务证据
- **Bayesian aggregation**：把不确定性感知的上下文聚合成任务后验
- **task-conditioned RSSM / actor / critic**：可迁移到其他 latent MBRL backbone

![[paperPDFs/World_Models_in_the_context_of_Model_Based_RL/arXiv_2024/2024_Adaptive_World_Models_Learning_Behaviors_by_Latent_Imagination_Under_Non_Stationarity.pdf]]