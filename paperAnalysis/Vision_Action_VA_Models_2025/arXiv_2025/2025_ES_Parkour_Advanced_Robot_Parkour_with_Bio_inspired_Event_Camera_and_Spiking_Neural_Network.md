---
title: "ES-Parkour: Advanced Robot Parkour with Bio-inspired Event Camera and Spiking Neural Network"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/robot-parkour
  - task/legged-locomotion
  - reinforcement-learning
  - spiking-neural-network
  - knowledge-distillation
  - dataset/IsaacGym
  - opensource/no
core_operator: 用事件相机的高频稀疏视觉流驱动SNN学生策略，并通过ANN教师蒸馏把四足跑酷能力迁移到低功耗控制网络。
primary_logic: |
  常规视觉/特权地形信息训练出的ANN教师 + 事件流与本体感觉输入 → IsaacGym中事件相机仿真、SNN视觉编码、动作与航向蒸馏、环境交互微调 → 在复杂地形与极端光照下输出四足机器人跑酷控制动作
claims:
  - "在ResNet视觉编码器设置下，ES-Parkour的理论能耗仅为对应ANN模型的11.7%，相当于节能88.29% [evidence: analysis]"
  - "SNN视觉编码器的总操作量显著低于ANN，对应效率比为0.46（ResNet）和0.29（MLP）[evidence: analysis]"
  - "蒸馏后的SNN策略可在hurdle/gap/parkour/step四类仿真障碍上完成跑酷，成功率分别为45%/60%/71%/29%，且平均关节电机能耗与ANN基线接近 [evidence: comparison]"
related_work_position:
  extends: "Extreme Parkour with Legged Robots (Cheng et al. 2023)"
  competes_with: "ANYmal Parkour (Hoeller et al. 2023); Robot Parkour Learning (Zhuang et al. 2023)"
  complementary_to: "Rapid Motor Adaptation (Kumar et al. 2021); Neural Volumetric Memory (Yang et al. 2023)"
evidence_strength: moderate
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_ES_Parkour_Advanced_Robot_Parkour_with_Bio_inspired_Event_Camera_and_Spiking_Neural_Network.pdf
category: Embodied_AI
---

# ES-Parkour: Advanced Robot Parkour with Bio-inspired Event Camera and Spiking Neural Network

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2503.09985)
> - **Summary**: 这篇工作把事件相机与脉冲神经网络引入四足机器人跑酷，通过“ANN教师→SNN学生”蒸馏，在极端光照与高速运动场景中保留跑酷能力，同时显著降低理论计算能耗。
> - **Key Performance**: ResNet编码器理论能耗仅为ANN的11.7%（88.29%节能）；parkour terrain 成功率 71%

> [!info] **Agent Summary**
> - **task_path**: 事件流 + 本体感觉 / 复杂地形与极端光照下的四足跑酷 -> 关节动作 + 航向控制
> - **bottleneck**: 传统深度/RGB视觉在高速运动与极端光照下存在帧率、动态范围和计算功耗三重瓶颈，难以匹配腿足控制回路
> - **mechanism_delta**: 用事件相机输入和SNN学生策略替换帧式视觉+ANN推理，并用ANN教师蒸馏稳定迁移跑酷技能
> - **evidence_signal**: 理论能耗分析显示ResNet版SNN编码器仅为ANN能耗的11.7%，同时在四类障碍上保持可用成功率
> - **reusable_ops**: [事件相机仿真, ANN到SNN动作/航向蒸馏]
> - **failure_modes**: [step地形成功率低, 仅有仿真验证且能耗为理论估算]
> - **open_questions**: [真实事件噪声下的sim2real差距有多大, 去掉ANN教师与特权信息后SNN能否稳定学到同等级跑酷能力]

## Part I：问题与挑战

这篇论文真正要解决的，不是“机器人能否跑酷”，而是：

**机器人能否在高速、极端光照、且功耗受限的条件下稳定跑酷。**

之所以这个问题现在值得做，是因为四足机器人在运动控制上已经很强，瓶颈开始从“会不会动”转向“在恶劣环境里能不能持续可靠地动”。已有 parkour / agile locomotion 方法多数依赖深度相机、LiDAR 或常规视觉网络，这在常规室内或受控场景有效，但在以下两点上明显吃亏：

1. **感知-控制频率错配**：腿足控制是高频闭环，而常规视觉传感器存在帧率和延迟限制；高速跃迁时，关键动态信息可能来不及进入策略网络。  
2. **环境鲁棒性-能耗错配**：深度/RGB 传感器容易受过曝、欠曝、强反差场景影响；同时 dense ANN 视觉与控制链路的计算代价更高，不利于轻量部署。

这篇论文的输入/输出接口很明确：
- **部署输入**：事件视觉 + 本体感觉
- **训练辅助输入**：ANN 教师侧可用 scandots、目标航向等特权信息
- **输出**：四足机器人的关节动作与航向控制

边界条件也很清楚：整套系统目前**只在 IsaacGym 仿真中验证**，事件相机数据也是在仿真器中生成的，不是真实硬件采集。因此它首先证明的是**路线可行性**，不是已经完成真实世界闭环落地。

## Part II：方法与洞察

作者的核心策略是：

**先让一个强 ANN 策略学会 parkour，再把这份能力蒸馏给面向事件输入的 SNN。**

### 1）在仿真里补出事件相机

IsaacGym 原生不支持事件相机，所以作者自己实现了事件生成过程。思路是根据图像亮度变化、光流和梯度信息，把“亮度变化超过阈值”的像素转成正/负事件。论文还引用了已有方法，使其只需单张深度图就能近似生成对应事件帧。

这一步的价值是：**先把训练环境建起来**。否则，事件相机+RL+四足 parkour 这条链路几乎没法大规模迭代。

### 2）先训练 ANN 教师策略

教师策略沿用了 Extreme Parkour 一类的训练思路：在 gap、step、hurdle、parkour terrain 等障碍上训练会跑酷的 ANN 控制器。训练中允许教师看到较强的辅助观测，如 terrain scandots 和目标 yaw。

这说明作者并没有试图“纯靠 SNN 从零学会一切”，而是把难问题拆成两步：
- 第一步：用成熟 ANN + RL 学出强能力
- 第二步：把能力迁移到更节能的 SNN 上

### 3）ANN → SNN 蒸馏

学生网络由三部分构成：
- **spiking ResNet-18**：编码事件输入
- **GRU**：融合事件特征与本体感觉
- **spiking MLP actor**：输出动作

训练流程分为两段：
- **warm-up 蒸馏**：让学生去匹配教师的动作输出与航向输出
- **环境交互微调**：学生继续在环境中交互，把复杂地形上的细节补回来

这一步解决的是 SNN 在连续控制里的现实难点：**直接做强化学习太难，尤其是稀疏脉冲表示下的策略优化更不稳定**。有了教师后，问题从“搜索一个好策略”变成“逼近一个已经有效的策略”。

### 核心直觉

- **改了什么**：把帧式视觉改成事件流，把 dense ANN 推理改成 spike-based SNN，把“直接训练 SNN”改成“先 ANN 学会，再蒸馏过去”。  
- **改变了哪个瓶颈**：  
  - 感知端：从受曝光和帧率限制的绝对亮度表征，变成对变化更敏感的事件表征  
  - 计算端：从密集 MAC 计算，变成稀疏 spike 操作  
  - 优化端：从高难度直接 RL，变成有教师约束的能力迁移  
- **能力为什么提升**：事件相机更适合高速运动和极端光照，SNN更适合处理稀疏时序信号；蒸馏则保证“换表示、降功耗”不会直接把跑酷技能一起丢掉。

### 战略取舍

| 设计选择 | 带来的好处 | 代价/风险 |
|---|---|---|
| 事件相机替代常规帧式视觉 | 高动态范围、低延迟，对高速和强光/弱光更友好 | 静态信息弱；仿真事件与真实事件可能有域差 |
| ANN 教师 → SNN 学生蒸馏 | 显著降低 SNN 控制训练难度，保留已有 parkour 能力 | 依赖强教师，且教师阶段使用特权信息 |
| SNN 编码器/控制器 | 理论操作数和能耗明显下降 | 需要时间步等超参设计，极难地形上仍会掉性能 |
| 仿真优先验证 | 训练安全、便宜、可大规模并行 | 不能直接证明真实机器人和真实神经形态芯片闭环效果 |

## Part III：证据与局限

### 关键证据

- **分析信号｜操作量确实下降**  
  SNN 视觉编码器的总操作量低于 ANN，不是只在一个 backbone 上偶然成立。论文给出的效率比是：
  - **ResNet：0.46**
  - **MLP：0.29**  
  这说明稀疏计算收益在不同网络形态下都存在。

- **分析信号｜理论能耗下降很明显**  
  以 ResNet 编码器为例，SNN 版本的理论能耗仅为 ANN 的 **11.7%**，即 **88.29% 节能**。actor 模块也有约 **69.44%** 的节能。  
  这说明方法的“bio-inspired”不只是概念包装，而是确实落到了计算代价上。

- **比较信号｜能力保留是部分成功的**  
  蒸馏后的 SNN 在四类障碍上仍能完成任务，成功率为：
  - hurdle：45%
  - gap：60%
  - parkour：71%
  - step：29%  
  同时，平均关节电机能耗与 ANN 基线接近，意味着计算节能并不是简单通过“动作更慢/更保守”换来的。

- **系统定位信号｜作者主张更广的极端场景覆盖**  
  论文用表格将自己定位为可覆盖正常光照、过曝、欠曝和高速场景的方法。这个结论更像**系统能力边界的陈述**，而不是统一公开 benchmark 下的严格 SOTA 证明，因此应保守理解。

**最关键的两项结果**：
1. **理论计算能耗：11.7% of ANN（ResNet 编码器）**
2. **parkour terrain 成功率：71%**

### 局限性

- **Fails when**: 面对 **step** 这类强垂直高度变化障碍时表现明显下降（成功率仅 29%）；另外，在低运动或近静态视觉场景里，事件信号本身就更弱。  
- **Assumes**: 依赖一个先训练好的 **ANN 教师**，且教师训练阶段使用 **scandots、目标航向等特权信息**；事件输入来自 **IsaacGym 中由深度图/光流近似生成的事件图**；训练使用的是 **10Hz 采样的事件图** 而非原生全异步事件流；能耗结论基于 **45nm 硬件模型的理论估算**，不是实机测得。  
- **Not designed for**: 这不是一个已完成的**真实机器人端到端部署方案**，也不是针对无教师纯 SNN 强化学习、静态场景语义理解或长期地图构建而设计的方法。

论文还明确指出，由于 **SNN 芯片难以获得**，目前没有真实机器人上的闭环验证。这一点对可复现性和扩展性都很关键：当前结果更像“强概念验证 + 仿真系统打通”，而不是完整落地。

### 可复用组件

- **事件相机仿真器**：在常规机器人仿真平台中，把深度/运动信息转成事件输入，便于训练 event-based policy。  
- **ANN→SNN 控制蒸馏范式**：先用 ANN 学强策略，再把动作/航向行为迁移到 SNN，适合连续控制任务。  
- **事件视觉 + 本体感觉融合骨架**：spiking ResNet 编码事件、GRU 融合状态，这个组合可以迁移到其它高速移动机器人任务。

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_ES_Parkour_Advanced_Robot_Parkour_with_Bio_inspired_Event_Camera_and_Spiking_Neural_Network.pdf]]