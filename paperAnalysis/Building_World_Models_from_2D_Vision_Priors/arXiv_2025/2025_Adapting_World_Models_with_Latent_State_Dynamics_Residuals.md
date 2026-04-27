---
title: "Adapting World Models with Latent-State Dynamics Residuals"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/robot-control
  - task/sim-to-real-transfer
  - state-token
  - reinforcement-learning
  - residual-dynamics
  - dataset/DeepMind Control Suite
  - dataset/Duckietown
  - opensource/partial
core_operator: "在冻结的离散潜状态世界模型上，对前向动力学 logits 学习小容量残差校正，以少量真实无奖励转移完成 sim-to-real 动力学适配"
primary_logic: |
  仿真域大量带奖励视觉轨迹 + 真实域少量无奖励离线转移 → 预训练并冻结离散潜状态世界模型，再在潜状态前向动力学 logits 上学习残差校正 → 用修正后的 imagined rollouts 和复用的仿真奖励头训练可部署到真实环境的策略
claims:
  - "Claim 1: 在四个视觉版 DeepMind Control 动力学错配任务上，使用 40K 目标域离线转移时，ReDRAW 持续优于零样本和世界模型微调基线，并在 3M 离线更新范围内显著更抗过拟合 [evidence: comparison]"
  - "Claim 2: 在真实 Duckiebot 视觉车道跟随任务中，ReDRAW 仅用 10K 步（约 17 分钟）无奖励真实轨迹即可完成 sim-to-real 适配；在动作反转真实环境中它是唯一能稳定完成圈跑的方法 [evidence: comparison]"
  - "Claim 3: 将修正建模为加到冻结动力学 logits 上的残差，比学习同容量的新动力学函数更能泛化到目标域；同时保持残差输入简洁优于更复杂的输入设计 [evidence: ablation]"
related_work_position:
  extends: "DreamerV3 (Hafner et al. 2023)"
  competes_with: "DreamerV3 finetuning (Hafner et al. 2023); Neural-Augmented Robot Simulation (Golemo et al. 2018)"
  complementary_to: "Plan2Explore (Sekar et al. 2020); Domain Randomization (Tobin et al. 2017)"
evidence_strength: strong
pdf_ref: "paperPDFs/Building_World_Models_from_2D_Vision_Priors/arXiv_2025/2025_Adapting_World_Models_with_Latent_State_Dynamics_Residuals.pdf"
category: Embodied_AI
---

# Adapting World Models with Latent-State Dynamics Residuals

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2504.02252), [Project](https://redraw.jblanier.net)
> - **Summary**: 该文把 sim-to-real 适配从“直接微调整个视觉世界模型”改成“在冻结潜状态动力学上学一个小残差”，因此能用少量、无奖励的真实轨迹稳定校正动力学并训练真实策略。
> - **Key Performance**: Duckiebot 真机仅用 10K 真实步（约 17 分钟）即可适配；动作反转真机场景下 ReDRAW 平均 dense reward 为 0.40，而 DRAW zeroshot / Dreamer finetune 分别为 -2.72 / -1.61。

> [!info] **Agent Summary**
> - **task_path**: 仿真视觉状态与动作 / 少量离线真实转移（无奖励） -> 校正后的潜状态世界模型 -> 真实环境控制策略
> - **bottleneck**: 高维图像状态下难以用小数据学习可泛化的动力学修正，直接微调整个世界模型会快速过拟合
> - **mechanism_delta**: 将适配参数限制为加在冻结前向动力学 logits 上的潜状态残差网络，而不是重学或微调整个动力学模型
> - **evidence_signal**: 4 个 DMC 错配动力学任务和 1 个真实 Duckiebot 任务中，ReDRAW 在长时间离线更新下持续高回报且明显更抗过拟合
> - **reusable_ops**: [冻结潜状态表征后只学习动力学残差, 复用仿真奖励头在真实域世界模型中训练策略]
> - **failure_modes**: [目标动力学变化过于复杂且真实数据过少时性能下降, 源域探索覆盖不足时残差缺少可修正的先验]
> - **open_questions**: [如何扩展到部分可观测/POMDP 场景, 是否能把同样的残差校准机制用于更大规模 foundation world models]

## Part I：问题与挑战

### 问题定义
- **输入**：大量仿真在线轨迹（有奖励）+ 少量真实环境离线转移（无奖励）。
- **输出**：一个在真实环境中表现好的控制策略。
- **基本设定**：仿真与真实共享状态空间、动作空间和奖励函数，但**转移动力学不同**。

### 真正的瓶颈是什么
这篇论文真正瞄准的，不是一般性的“迁移学习”，而是一个更具体的瓶颈：

1. **传统 residual dynamics 依赖低维显式状态**  
   一旦状态是图像，直接在观测空间学残差几乎不可行，既难以 sample-efficient，也容易学到视觉噪声。

2. **目标域数据少且没有奖励**  
   真实机器人往往只能给很少的离线轨迹，甚至没有 reward logging。此时直接 finetune 整个世界模型自由度太大，过拟合几乎是默认结果。

3. **真实部署不适合靠 early stopping 保命**  
   真机反复验证代价高、节奏慢。比起“某个时刻恰好最好”，更重要的是方法本身能否**长期稳定**。

### 为什么现在值得做
因为 DreamerV3 一类潜状态世界模型已经足够成熟：它们可以把高维视觉输入压缩到一个可 rollout、可规划的 latent space。  
这使得动力学校正可以从“像素空间修补”转成“潜状态动力学修补”，从而把原本很难做的小样本适配变成一个更可控的问题。

### 边界条件
论文的成功依赖几个明确前提：
- **全可观测 MDP**：作者为图像任务额外加入速度等向量，保证当前观测足以预测未来。
- **只改动力学，不改奖励语义**：真实域没有奖励标签，但能复用仿真 reward head。
- **感知表征要能零样本迁移**：真实图像与仿真图像的 gap 需要靠增强、轻度 domain randomization、Gaussian-splatting digital twin 等额外工程压住。
- **源域覆盖要足够广**：如果仿真数据只覆盖狭窄策略分布，真实域残差无从修正没见过的状态-动作区域。

## Part II：方法与洞察

### 方法主线
作者先训练一个世界模型 **DRAW**，再在其上加一个适配模块得到 **ReDRAW**。

1. **DRAW：在仿真中预训练**
   - 用 encoder 把观测编码成一个**离散随机潜状态**。
   - 用该潜状态预测下一潜状态、奖励、continue 信号，并重建观测。
   - 在世界模型里做 imagined rollouts，训练 actor-critic。
   - 用 Plan2Explore 收集更广覆盖的仿真经验，而不是只收集“会解源任务”的狭窄分布。

2. **ReDRAW：在真实域校正动力学**
   - 冻结世界模型参数。
   - 只训练一个小 MLP residual，输入是上一步潜状态和动作。
   - 这个 residual 不替代原动力学，而是**加到前向动力学输出的 logits 上**，从而修正下一时刻潜状态分布。
   - 训练目标是让“校正后的潜状态转移”贴近“冻结 encoder 在真实数据上给出的潜状态后验”。

3. **策略学习**
   - 使用修正后的 latent dynamics 在世界模型中 rollout。
   - 继续复用仿真中学到的 reward predictor，因此**真实域不需要 reward label**。
   - 最终将 actor 直接部署到真实环境。

### 核心直觉
**改了什么**  
从“在显式状态或整模型层面做大幅修改”，改成“只在冻结潜状态动力学上做小幅加性校正”。

**这改变了哪个约束/瓶颈**  
- 把问题从**高维观测回归**变成**低维结构化潜状态校准**；
- 把适配自由度从**整模型微调**压缩为**小容量 residual**；
- 把真实域监督需求从“需要带奖励交互”降到“只要无奖励离线转移”。

**带来了什么能力变化**  
- 小数据下更稳定；
- 离线训练更抗过拟合；
- 更适合真实机器人这类验证成本高、奖励难标的场景。

**为什么因果上有效**  
因为 ReDRAW 不是重新学习真实动力学，而是把仿真世界模型当成**强先验**保留下来，只让 residual 去修正“仿真与真实不一致”的部分。  
这背后的因果链条是：

**冻结仿真动力学先验 → 限制目标域可学习函数类 → 降低用少量真实数据记忆噪声的风险 → 提升长期离线更新的稳定性**

论文附录里的消融进一步支持这一点：
- 学一个“新动力学函数”不如学“logit residual”；
- 给 residual 增加更多连续输入反而更差；
- 说明真正关键的是**把修正放进一个足够窄的瓶颈里**。

### 策略性权衡

| 设计选择 | 解决的瓶颈 | 收益 | 代价 |
|---|---|---|---|
| 单一离散潜状态表征 | 图像状态过高维，难以低数据迁移 | 让 residual 在紧凑状态上学习 | 强依赖源域表征质量与覆盖 |
| 冻结世界模型，只学 residual | 全量 finetune 易过拟合 | 长时间离线训练更稳 | 适配能力受限于“小修正”假设 |
| residual 加在 dynamics logits 上 | 直接重学动力学自由度过大 | 保留仿真先验，降低样本需求 | 对复杂结构性偏差可能不够 |
| 复用仿真 reward head | 真实奖励难获得 | 无奖励真实轨迹也能训练策略 | 要求跨域奖励语义不变 |
| Plan2Explore 预训练 | 源域分布过窄影响迁移 | 提高对未知动力学变化的覆盖 | 增加预训练交互与计算成本 |

### 一个很有价值的细节
作者让主动力学网络额外接收“上一步预测分布”，以增强长时序动力学学习；但**没有**把 residual 也做得更复杂。  
这相当于把长期预测能力交给主模型，把目标域小样本泛化交给小 residual，从而把“表达力”和“抗过拟合”拆开处理。

## Part III：证据与局限

### 关键证据信号
1. **比较信号：4 个视觉 DMC 错配动力学任务**
   - 每个目标域只给 **40K** 离线转移。
   - zeroshot 普遍不足；直接 finetune 世界模型往往前期有提升、后期明显过拟合。
   - **ReDRAW 在 3M 次离线更新内都能维持高性能**。
   - 这说明它的优势不是短期拟合快，而是**长期稳定地利用仿真先验**。

2. **因果信号：residual vs. replacement dynamics**
   - 作者用同等小容量网络对比“学 residual”与“学全新 dynamics function”。
   - 结果是 residual 更好。
   - 结论：起作用的不是“额外增加模型容量”，而是**加性校正带来的复杂度约束**。

3. **数据策略信号：覆盖比 exploitation 更重要**
   - 源域若只用 exploit policy 收数据，迁移显著变差。
   - 目标域 expert demonstration 明显比纯随机动作更稳。
   - 这说明 ReDRAW 的成功依赖“**广覆盖仿真先验 + 有信息量的真实离线转移**”。

4. **真实机器人信号：Duckiebot sim-to-real**
   - 只用 **10K 真机步**，约 **17 分钟**，且**没有真实奖励标签**。
   - **未修改真实环境**：ReDRAW 平均 dense reward **0.38**，优于 DRAW zeroshot 的 **0.12** 和 Dreamer finetune 的 **-0.87**；平均 center offset **2.39** 也明显更低。
   - **动作反转真实环境**：ReDRAW dense reward **0.40**、center offset **2.07**，且是**唯一能稳定完成圈跑**的方法。
   - 这说明方法不仅在模拟 benchmark 上有效，也能落到真实视觉控制。

### 能力跳跃在哪里
相对以往做法，ReDRAW 的真正跃迁不是“模型更大”，而是：
- **适配位置对了**：在潜状态动力学上修，而不是在像素或整模型层面乱改；
- **适配自由度小了**：更适合真实域小数据；
- **监督需求低了**：只需无奖励真实转移即可完成策略适配。

对真实机器人来说，这种“不过拟合、少监督、少上机验证”的属性，实际价值往往比纯 benchmark 数字更重要。

### 局限性
- **Fails when**: 目标动力学变化过于复杂、非局部，或者真实数据太少以至于“小残差可修正”的假设失效时，ReDRAW 会退化；文中也观察到某些任务在更少 expert data 下会过拟合或性能下滑。
- **Assumes**: 全可观测 MDP；仿真与真实共享状态/动作/奖励语义；主要差异只在转移动力学；encoder 能在增强与随机化帮助下从仿真零样本迁移到真实图像；源域预训练覆盖足够广。
- **Not designed for**: 部分可观测场景、奖励函数变化、严重视觉域差但没有额外感知适配、以及需要在线交互式系统辨识的场景。

### 资源与复现依赖
- 需要**大量仿真预训练数据**：DMC 中使用 9M environment steps；Duckiebot 也用了长时间随机探索 + Plan2Explore。
- 需要**额外感知桥接工程**：Gaussian-splatting digital twin、camera randomization、非对称增强/重建目标。
- 适配训练也不算轻：DMC 实验离线更新到 **3M**，文中称约 **1–3 天**。
- 开源信息更明确地覆盖了项目/模拟器部分，若要完整复现实验，仍需结合 DreamerV3 改造实现与真实硬件环境。

### 可复用组件
- **latent dynamics residual adapter**：可嫁接到其他潜状态 world model 上，尤其适合“仿真强、真实弱”的场景。
- **frozen reward-head reuse**：在奖励语义不变时，适合无奖励真实域适配。
- **coverage-first pretraining**：先最大化仿真覆盖，再做目标域低数据校正，比仅优化源任务回报更有迁移价值。

## Local PDF reference
![[paperPDFs/Building_World_Models_from_2D_Vision_Priors/arXiv_2025/2025_Adapting_World_Models_with_Latent_State_Dynamics_Residuals.pdf]]