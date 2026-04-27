---
title: "COMPASS: Cross-embOdiment Mobility Policy via ResiduAl RL and Skill Synthesis"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/robot-navigation
  - reinforcement-learning
  - world-model
  - policy-distillation
  - dataset/Carter
  - opensource/no
core_operator: "冻结单机体 world-model 模仿策略作为共享移动先验，再用各机体的残差强化学习做动作补偿，并蒸馏成带机体条件的统一策略"
primary_logic: |
  单一机体的 RGB/速度/路线演示 + 多机体仿真交互 → 共享世界模型提取移动先验，残差 RL 学习各机体对基础速度命令的修正，再按机体嵌入蒸馏 specialist → 输出跨机体通用的速度控制策略
claims:
  - "在 4 种机器人与 4 类场景的评测中，残差 RL specialist 相比 IL-only 的 X-Mobility 将成功率提升约 5×–40×，并把 WTT 平均降到约原来的 1/3 [evidence: comparison]"
  - "以机体嵌入条件化的 distilled generalist 在大多数设置下与各 specialist 性能接近，G1 上部分场景还略优 [evidence: comparison]"
  - "在相同潜状态输入下，去掉 IL 基座的 RL-from-scratch 在 1000 episodes 内仍难以收敛，而 residual RL 收敛明显更快 [evidence: ablation]"
related_work_position:
  extends: "X-Mobility (Liu et al. 2024)"
  competes_with: "X-Mobility (Liu et al. 2024); Scaling Cross-Embodied Learning (Doshi et al. 2024)"
  complementary_to: "Locate3D (Arnaud et al. 2025); VR-Robo (Zhu et al. 2025)"
evidence_strength: moderate
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_COMPASS_Cross_embOdiment_Mobility_Policy_via_ResiduAl_RL_and_Skill_Synthesis.pdf
category: Embodied_AI
---

# COMPASS: Cross-embOdiment Mobility Policy via ResiduAl RL and Skill Synthesis

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2502.16372), [Project](https://nvlabs.github.io/COMPASS)
> - **Summary**: 用“单机体模仿学习先验 + 多机体残差强化学习适配 + 机体条件蒸馏”三阶段流程，把点到点移动策略从单一机器人扩展到轮式、人形、四足等多种机体，同时避免为每个机体重新采集专家演示。
> - **Key Performance**: 未见于 IL 训练的机体上，成功率相对 IL-only 基线约提升 5×；整体 travel efficiency 约提升 3×，且实机零样本 sim-to-real 约 80% 成功率（20 次）。

> [!info] **Agent Summary**
> - **task_path**: 单机体示教迁移设定下，RGB相机 + 当前速度 + route/goal + 机体ID -> 线速度/角速度命令 -> 跨机体点到点移动
> - **bottleneck**: 环境理解可共享，但动作可实现性受机体形态、动力学和传感差异强约束；纯 IL 需要每机体单独示教，纯视觉 RL 又太难探索
> - **mechanism_delta**: 把“共享移动先验”与“机体特有补偿”拆开：冻结单机体 world-model IL 基座，只让 residual RL 学习机体偏差，再用 embodiment embedding 把多个 specialist 蒸馏成一个 generalist
> - **evidence_signal**: 4 机体 × 4 场景评测中 SR 提升约 5×–40%，且去掉 IL 基座后 RL-from-scratch 难以收敛
> - **reusable_ops**: [shared-world-model-prior, residual-action-correction, embodiment-conditioned-distillation]
> - **failure_modes**: [长程路由提示不足的办公室/多货架场景, 多机体蒸馏带来的 averaging effect]
> - **open_questions**: [learned-embodiment-embedding-for-unseen-robots, hierarchical-planner-integration]

## Part I：问题与挑战

这篇论文解决的是一个很具体但很难扩展的问题：**跨机体的点到点移动（mobility/navigation）**。  
输入是机器人当前的 RGB 图像、速度、目标/路线提示，以及一个机体标识；输出不是关节力矩，而是**线速度/角速度命令**，再交给各机器人自己的低层控制器去执行。

### 真正的难点是什么？

真正的瓶颈不是“机器人能不能看懂障碍物”，而是：

1. **可共享的部分**：环境理解、障碍规避、向目标推进，这些能力在轮式、人形、四足之间有很强共性。
2. **不可直接共享的部分**：同样一条“向前+右转”的高层动作，在不同机体上对应完全不同的稳定性边界、碰撞几何、运动学和传感配置。

因此，纯 IL 会遇到两个问题：

- **数据瓶颈**：每个新机体都要重新采高质量演示，尤其人形机器人代价极高。
- **闭环分布偏移**：即便学到单机体策略，部署时一旦进入训练外状态，错误会被滚雪球式放大。

而纯 RL-from-scratch 也不现实，因为视觉导航是一个**高维感知 + 稀疏反馈 + 强机体依赖**的问题，探索太贵。

### 为什么现在值得做？

因为现在有两个条件成熟了：

- **X-Mobility** 这类 world-model 导航基座，已经能从单机体演示里学到比较稳定的移动先验；
- **Isaac Lab** 这类高并行仿真平台，使得视觉 RL 的机体适配成本降到了可接受范围。

所以这篇论文的核心不是“彻底取消机体适配”，而是把适配方式从**每机体重新采专家示教**，转成**每机体在仿真里做残差修正**。

### 边界条件

- 任务是**point-to-point mobility**，不是开放式长程任务规划。
- 策略输出是**速度命令**，不是端到端关节控制。
- 新机体并非真正零样本：仍需要该机体的仿真环境、低层控制器，以及 residual RL 训练。
- 路线提示比较弱，文中主要用**起点到终点的直线引导**，这意味着长程规划不是它的设计中心。

---

## Part II：方法与洞察

论文的方法可以概括成三步：

### 1）单机体 IL：先学“共享移动先验”

作者首先复用 **X-Mobility** 的单机体 checkpoint（基于 Carter dataset），把视觉观察和历史动作压到一个 latent policy state 里。  
这个阶段的目标不是学会所有机体的控制，而是先把下面这些“共性能力”固化下来：

- 看懂场景结构
- 根据 route/goal 形成局部决策
- 维持闭环导航所需的时序上下文

这一步的价值在于：**把跨机体共享的环境语义先学出来**。

### 2）Residual RL：每个机体只学“偏差修正”

随后，对每个目标机体单独训练一个 residual policy。  
最终动作不是从零生成，而是：

- **基座动作**：来自冻结的 IL policy
- **残差动作**：由 RL 训练的修正项给出
- **最终执行动作**：基座动作 + 残差修正

这一步的关键因果点在于：  
RL 不再需要从高维视觉输入里“发明”整套导航行为，只需要学习：

- 这个机体比 Carter 更高/更矮/更宽时该怎么过障碍
- 这个机体的动力学约束下该如何调整速度
- 哪些场景下需要更保守或更激进的修正

作者还强调 residual state 可以额外注入基座没显式用到的信息，比如目标朝向。这意味着它是一个**可插拔补偿层**，不是重训整个系统。

### 3）Policy Distillation：把多个 specialist 合成一个 generalist

当每个机体都有了 specialist 之后，再把它们蒸馏成一个统一 policy。  
统一 policy 额外接收一个 **embodiment embedding**（文中主要是 one-hot），从而在共享 backbone 上切换不同机体行为。

更细一点地说，作者不是只模仿专家动作均值，而是尽量匹配 specialist 的**动作分布**。这点很重要，因为它保留了不同 specialist 的动作不确定性与风格差异，减少“只学平均动作”带来的信息损失。

### 核心直觉

这篇论文最重要的改动，不是换了一个更强的网络，而是换了**问题分解方式**：

- **以前**：每个机体都要学一整套视觉导航策略
- **现在**：先学共享 mobility prior，再学 embodiment delta，最后统一蒸馏部署

这会改变三个瓶颈：

1. **数据瓶颈改变**  
   从“每机体都要专家示教”变成“只要一个源机体示教 + 每机体仿真交互”。

2. **探索瓶颈改变**  
   从“RL 搜索整套动作策略”变成“RL 搜索对基座动作的小修正”，搜索空间明显变小。

3. **部署瓶颈改变**  
   从“每机体一个 specialist”变成“一个 generalist + embodiment ID”，部署更统一。

为什么这套设计有效？因为对 mobility 而言，**环境语义和动作可实现性本来就不是同一个层级的问题**。  
world model 负责前者，residual RL 负责后者，distillation 负责把多个局部解重新压回一个可部署模型。

### 战略权衡

| 设计选择 | 改变了什么瓶颈 | 带来的能力 | 代价 / 风险 |
|---|---|---|---|
| 单机体 world-model IL 基座 | 先固定共享视觉-移动先验 | 降低多机体演示数据需求 | 继承源机体偏置，基座若系统性错，残差难完全修复 |
| 冻结基座 + residual RL | 把搜索从“学整策”缩到“学修正” | 收敛更快、样本效率更高 | 仍需每机体仿真环境与低层控制器 |
| embodiment-conditioned distillation | 把多 specialist 合成 1 个模型 | 统一部署、跨机体共享经验 | seen-robot setting 为主，one-hot 对未见机体外推弱 |
| 混合环境训练 | 扩大场景覆盖 | 更强泛化能力 | 算力更高，单场景最优性可能下降 |
| KL 匹配完整策略分布 | 不只学平均动作 | 保留 specialist 不确定性与行为风格 | 训练实现更复杂，仍有 averaging effect |

---

## Part III：证据与局限

### 关键证据

**1. 最强比较信号：跨机体适配不是小幅改进，而是量级变化。**  
最典型的是 G1 在 Warehouse Multi 场景里，IL-only 的 X-Mobility 只有 **1.8% SR**，而 specialist / generalist 达到 **93.7% / 94.5%**。这说明问题确实不在“是否有视觉先验”，而在“有没有机体特定补偿”。

**2. Generalist 基本没有明显蒸馏损失。**  
表 1 显示 generalist 在多数设置下与 specialist 很接近，个别场景还略优。这支持作者的核心主张：  
多机体知识并不一定要以多个独立模型存在，条件化蒸馏可以把它们压到一个统一策略中。

**3. 机制证据：residual RL 的收益来自更容易优化，而不是单纯更多训练。**  
作者做了 RL-from-scratch 对照：在相同 latent state 输入下，不用 IL 基座时，1000 episodes 仍难收敛；而 residual RL 收敛明显更快。  
这直接支持“把 RL 目标从全策略学习改成局部修正”这个因果解释。

**4. 消融说明作者的关键选择不是随便拍脑袋。**

- **Curriculum 不一定更好**：因为基座已经很强，过早简化任务反而减少了对难例的暴露。
- **Depth critic 有条件地有用**：在办公室这类高障碍场景更有利，但在低矮障碍更多的仓库场景会退化。
- **混合环境训练更泛化**：只在 warehouse 训练会略提同域表现，但跨到 office/combined 会更差。
- **KL 蒸馏略优于只做 MSE 模仿均值**：说明保留完整动作分布是有价值的。
- **剔除失败轨迹帮助有限**：失败样本可能也是 corner cases，删掉后 generalist 反而少了一部分鲁棒性来源。

**5. 部署信号：不仅是仿真里有效。**  
在 Carter 和 G1 实机上，结合 cuVSLAM 的零样本 sim-to-real 部署达到约 **80% 成功率（20 次）**；TensorRT 推理 **P50 约 29.3 ms**，显存约 **422 MB**。  
这说明它至少在“高层视觉导航 + 速度控制”这一层具有实用部署价值。

### 局限性

- **Fails when**: 需要长程全局规划、但只给短程直线 route 提示的场景，尤其是办公室和多货架仓库这类需要提前绕行或考虑高度约束的任务；另外，当前 one-hot embodiment 条件化并不能真正零样本泛化到未见新机体。
- **Assumes**: 有一个源机体的高质量 teacher demonstrations；每个目标机体都能在 Isaac Lab 中稳定仿真，且已具备把速度命令转为关节执行的低层控制器/locomotion policy；训练算力不低（specialist 用 2×L40，distillation 用 4×H100）；实机部署还依赖 cuVSLAM、TensorRT 等工程组件。正文给了 project page，但未明确说明完整代码/权重的开源状态。
- **Not designed for**: 端到端从视觉直接学 joint/torque-level locomotion；无 route/无定位条件下的全局导航；以及完全不经过 per-embodiment 仿真适配就推广到全新 morphology 的真正 zero-shot cross-embodiment learning。

### 可复用部件

这篇论文里最值得复用的，不一定是整套系统，而是下面几个操作：

- **共享 world-model state 作为跨机体公共表征**
- **动作级 residual wrapper**：把已有策略改成“base + correction”
- **residual state augmentation**：在不重训基座时给补偿层注入新信息
- **embodiment-conditioned distillation**：把多 specialist 压成一个部署模型
- **specialist rollout 作为合成数据源**：文中甚至把其用于 Gr00t N1.5 的导航后训练，说明它还能充当 VLA 数据生成器

**一句话总结**：  
COMPASS 的价值不在于证明“一个策略能神奇地零样本适配任何机器人”，而在于证明了一个更现实的工程范式：**用单机体演示学共享导航先验，再用仿真 residual RL 把差异补上，最后统一蒸馏部署。**

## Local PDF reference

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_COMPASS_Cross_embOdiment_Mobility_Policy_via_ResiduAl_RL_and_Skill_Synthesis.pdf]]