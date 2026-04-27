---
title: "DemoGen: Synthetic Demonstration Generation for Data-Efficient Visuomotor Policy Learning"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/robot-manipulation
  - task/imitation-learning
  - task-and-motion-planning
  - synthetic-data-generation
  - point-cloud-editing
  - dataset/MetaWorld
  - repr/point-cloud
  - opensource/promised
core_operator: 将单条真人示教拆分为接触技能段与自由运动段，在动作端用TAMP重定向、在观测端用点云3D编辑同步变换，低成本合成可直接训练策略的示教数据。
primary_logic: |
  单条人类示教 + 目标物体新配置
  → 轨迹分段为 motion/skill，技能段随物体SE(3)变换、运动段用运动规划连接
  → 对点云观测与本体感觉施加一致的3D编辑
  → 输出适配新空间布局的合成观测-动作示教集，用于训练闭环视觉运动策略
claims:
  - "Claim 1: With one human-collected demonstration replayed twice into 3 source trajectories, DemoGen improves average success from 11.0% to 74.6% across 8 real-world manipulation tasks [evidence: comparison]"
  - "Claim 2: On 8 simulated manipulation tasks, a policy trained on DemoGen data generated from 1 source demonstration reaches 88% average success, outperforming training on 10 source demonstrations (68%) and approaching 25-source performance (91%) [evidence: comparison]"
  - "Claim 3: For the real-world datasets used in this paper, DemoGen reduces estimated data generation time from 83.7 hours with MimicGen-style on-robot rollouts to 22.0 seconds of computation [evidence: analysis]"
related_work_position:
  extends: "MimicGen (Mandlekar et al. 2023)"
  competes_with: "MimicGen (Mandlekar et al. 2023); SkillMimicGen (Garrett et al. 2024)"
  complementary_to: "3D Diffusion Policy (Ze et al. 2024); Diffusion Policy (Chi et al. 2023)"
evidence_strength: strong
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_DemoGen_Synthetic_Demonstration_Generation_for_Data_Efficient_Visuomotor_Policy_Learning.pdf
category: Embodied_AI
---

# DemoGen: Synthetic Demonstration Generation for Data-Efficient Visuomotor Policy Learning

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2502.16932), [Project](https://demo-generation.github.io)
> - **Summary**: 这篇工作把“单条真人示教”扩展成“大量空间增强示教”，关键做法是把动作轨迹和点云观测一起在 3D 空间中重定向，从而用极低成本提升机器人视觉运动策略的空间泛化。
> - **Key Performance**: 8 个真实任务平均成功率 **74.6%**（source-only 为 **11.0%**）；整套真实数据生成时间 **22.0 s**（文中估算 MimicGen 式流程为 **83.7 h**）

> [!info] **Agent Summary**
> - **task_path**: 单条真人示教 / 目标物体新配置 → 合成观测-动作示教 → 闭环机器人操作策略
> - **bottleneck**: 视觉运动策略的空间泛化很弱，必须靠大量“挪物体-重采示教”覆盖工作空间；而 MimicGen 式方法虽能生成执行计划，仍需昂贵 on-robot rollout 获取对应观测
> - **mechanism_delta**: 用“skill 段刚体变换 + motion 段重规划 + 点云3D编辑”直接把新配置下的执行计划变成可训练的 observation-action 对
> - **evidence_signal**: 8 个真实任务平均成功率从 11.0% 提升到 74.6%，且在模拟中 1 demo + DemoGen 已优于 10 demos 的真人采集
> - **reusable_ops**: [轨迹按 free-space motion / contact skill 分段, 对物体与末端点云施加与动作一致的 SE(3) 变换]
> - **failure_modes**: [单视角点云导致 visual mismatch, 高精度双物体任务在远离源示教配置时性能明显下降]
> - **open_questions**: [如何缓解单视角视觉失配, 多源示教与合成覆盖范围的最优成本-性能平衡在哪里]

## Part I：问题与挑战

**What / Why：真正的瓶颈是什么，为什么现在值得解决？**

这篇论文的核心判断很准：  
视觉运动策略的数据需求高，**真正昂贵的并不总是接触技能本身，而是“让策略学会在不同空间配置下接近目标”**。

作者先做了一个经验研究来支持这个判断：

- 在按钮、精确插销等任务里，策略的有效工作范围大致等于**示教中物体位置附近几个局部区域的并集**
- 任务越精细，这个局部区域越小
- 即便用了 3D 表征或预训练视觉编码器，空间泛化有所改善，但**并没有从根本上消失**

所以问题不是“policy 不会模仿”，而是：

1. **空间支持域太窄**：只会在见过的位置附近工作  
2. **人工采集成本太高**：为了覆盖整张桌面，不得不反复 reposition 物体并重新示教  
3. **现有自动化方案落不了地**：MimicGen 一类方法能合成动作计划，但现实世界里还得让机器人真跑一遍，才能拿到训练需要的观测-动作对，这个成本几乎又回去了

### 输入 / 输出接口

- **输入**：单条人类示教轨迹、目标物体的新初始配置、单视角 RGB-D 得到的点云观测与本体感觉
- **输出**：适配新空间配置的 synthetic demonstration（观测-动作轨迹），可直接训练 DP / DP3 这类闭环策略

### 边界条件

这个方法不是“任意任务通吃”，它默认：

- 任务可分解为按对象顺序发生的 manipulation stages
- 工作区可裁剪，物体与末端执行器在点云里可近似分离
- 任务**需要空间泛化**；如果目标位姿固定、根本不需要 spatial augmentation，这套方法收益会很小

一句话总结本节：  
**作者解决的不是 policy architecture，而是 training distribution coverage。**

## Part II：方法与洞察

### 方法主线

DemoGen 的思想很工程化：  
**既然昂贵的是“真实机器人回放拿观测”，那就把“动作适配”和“观测生成”都搬到 3D 空间里做。**

#### 1. 源示教预处理：先把示教变成可编辑对象

作者先对单条源示教做三件事：

- 对点云做裁剪、聚类、下采样
- 用 Grounded SAM 在首帧拿到被操作物体的分割 mask
- 按“是否与目标物体进入接触邻域”把轨迹拆成两类段：
  - **motion 段**：自由空间中的接近/转运
  - **skill 段**：与物体发生接触的操控

这一步很关键，因为它明确区分了两类本质不同的几何问题：

- **skill 段**要保留接触关系
- **motion 段**只需要连通前后 skill 段即可

#### 2. 动作生成：skill 变换，motion 重规划

对目标新配置 \(s_0'\)，DemoGen 不是整条轨迹一起硬搬，而是分情况处理：

- **手部命令**（gripper open/close、灵巧手关节控制）默认与空间位置无关，直接复用
- **skill 段**：整段跟着对应物体做统一 SE(3) 变换  
  直觉上，接触技能在物体坐标系下近似不变
- **motion 段**：用 motion planner 在相邻两个已变换后的 skill 端点之间重新规划  
  简单场景可线性插值，复杂场景可用 RRT-Connect

另一个很实用的选择是：

- 不再用容易积累误差的 delta pose 控制
- 改用 **absolute end-effector pose + IK controller**

这让“无需真机 rollout 过滤失败轨迹”更现实，因为生成出来的动作更容易被稳定执行。

#### 3. 观测生成：把同样的几何变换同步施加到点云上

这是整篇论文最关键的机制增量。

作者没有尝试用 2D 生成模型去“改图片里的物体位置”，因为那很难保持 3D 透视一致性。  
他们直接把**点云**作为 observation modality，于是空间增强变成了显式 3D 编辑问题。

做法是一个简单但有效的 **segment-and-transform**：

- 物体在每个时间段被标成：
  - **to-do**：还没被操作，按目标初始位姿变换
  - **doing**：正在接触，被并入末端执行器一起移动
  - **done**：已完成，保持最终状态
- 末端执行器点云则按当前 proprioception / EE pose 的变换同步移动
- 本体感觉也做一致变换，而不是直接偷懒用下一步 action 替代

这样一来，**动作和观测共享同一组几何变换**，训练数据的对齐关系就成立了。

#### 4. 真实世界部署策略：针对噪声与摆放误差做小修正

作者还加了两个非常现实的补丁：

- **源示教 replay 两次**：不是再采三条人类示教，而是对同一条示教低成本回放两次，增加传感器噪声形态，避免 policy 只记住一份点云瑕疵
- **对目标配置加小扰动**：因为现实摆放很难绝对精确，所以每个目标配置附近再合成多个 ±1.5 cm 的微扰样本

这让 DemoGen 学到的不是“单点模板匹配”，而是一个更像真实放置误差分布的小邻域。

### 核心直觉

作者抓住了两个因果上真正有用的近似：

1. **接触技能在物体局部坐标系下近似可复用**  
   比如“抓起花”“把花插入花瓶”，关键接触关系主要由物体相对几何决定，不需要每次从零学

2. **自由空间接近路径可以后验重规划**  
   机器人怎么从 A 走到 B，不是示教里最稀缺的语义信息，而是一个几何连接问题

因此，DemoGen 改变的不是 policy 本身，而是**训练分布的支持域**：

- **原来**：数据只覆盖源示教附近的小区域
- **现在**：通过动作重定向 + 点云3D编辑，数据可覆盖目标工作区的一组新配置
- **结果**：policy 学到的不再是一条固定轨迹，而是“在更大空间支持域里如何观察-接近-操控”

更重要的是，这个框架还能继续扩展：

- **扰动抗性**：在合成数据里人为让物体突然偏移，再合成“重新对准并继续操作”的轨迹
- **障碍规避**：在点云里加障碍体，再用 planner 产生避障路径

也就是说，**DemoGen 学到的能力边界基本由你往示教分布里注入什么变化决定**。

### 战略取舍

| 设计选择 | 带来的收益 | 代价 / 风险 |
| --- | --- | --- |
| 用点云而不是 2D 图像做增强 | 3D 位姿编辑直接、动作与观测几何一致 | 单视角下会有缺失面与视角失配 |
| skill 段刚体变换 + motion 段重规划 | 既保留接触技能，又适配新布局 | 依赖任务可分段、对象顺序可知 |
| 全合成观测替代 on-robot rollout | 成本从小时级降到秒级 | 合成观测真实性仍弱于实拍 |
| absolute EE pose + IK | 更少控制误差积累，减少失败轨迹 | 依赖 IK 稳定性与可达性 |
| 针对目标评测配置定向生成 | 样本效率高、部署友好 | 更像“目标工作区覆盖”，不是无限开放分布泛化 |

## Part III：证据与局限

### 关键证据信号

- **Signal 1 — 瓶颈诊断（analysis）**  
  作者先证明“空间泛化差”确实是主要瓶颈：策略有效范围近似为示教点位周围局部区域并集，且高精度任务更严重。这个诊断让后续方法不是拍脑袋做 data augmentation，而是精准对准问题。

- **Signal 2 — 模拟 one-shot imitation（comparison）**  
  在 8 个模拟任务上，1 条 source demo 经过 DemoGen 后，平均成功率达到 **88%**；不仅显著高于只用 10 条 source demos 的 **68%**，还接近 25 条 source demos 的 **91%**。  
  这说明 DemoGen 不是小修小补，而是在“单位人工示教带来的有效覆盖”上产生了明显跃迁。

- **Signal 3 — 真实世界跨平台验证（comparison）**  
  在 8 个真实任务、3 类平台（平行夹爪、灵巧手、双臂人形）上，平均成功率从 **11.0%** 提升到 **74.6%**。  
  这个结果比单一平台验证更有说服力，因为它说明方法不是某个硬件或某个任务的特例。

- **Signal 4 — 能力注入式扩展（comparison）**  
  扰动抗性实验里，ADR 策略把 normalized score 从 **40.4** 提到 **92.3**；  
  避障实验里，带障碍增强的数据让策略在 **25** 次测试中成功避障 **22** 次。  
  这说明 DemoGen 不只是“扩位置覆盖”，还能往数据里显式写入闭环纠偏与避障行为。

- **Signal 5 — 实用成本（analysis）**  
  文中估算，MimicGen 式真实 rollout 生成整套数据需 **83.7 小时**，而 DemoGen 只需 **22 秒** 计算时间。  
  真正的能力跃迁在这里：**它把真实世界 demonstration generation 从“机器人在线采样问题”改成了“离线几何编辑问题”。**

### 局限性

- **Fails when**: 单视角点云的视觉失配变得严重时、目标配置离源示教过远时、任务要求极高精度且对朝向/接触误差极敏感时，性能会明显下降；在 Dex-Drill、Dex-Coffee 这类高精度双物体任务上提升幅度就更有限。
- **Assumes**: 需要可裁剪工作区、可分割的点云、已知被操作物体序列、首帧可由 Grounded SAM 获得分割、IK 控制器能稳定跟踪绝对末端位姿、motion planner 能连通相邻 skill 段；真实部署里还依赖 replay 源示教来扩展噪声多样性。代码与数据在论文时点是 promised open-source，复现仍受这些外部组件影响。
- **Not designed for**: 不需要空间泛化的任务（如固定目标位姿场景）、高度杂乱或非结构化环境、对象/末端点云难以分离的场景、需要从完全无示教生成复杂新技能的场景。

### 可复用组件

这篇工作最值得复用的不是某个特定 policy，而是几个数据层操作：

- **轨迹按 motion / skill 分段**
- **skill 段在物体坐标系下做整体变换**
- **motion 段做 planner-based reconnection**
- **点云 segment-and-transform 合成观测**
- **围绕目标配置做小扰动增强**
- **异步物体/末端变换来注入 disturbance recovery**
- **在点云中插入几何障碍物并配合规划生成避障示教**

如果你已有 DP3 / Diffusion Policy 一类闭环策略，这些操作基本都可以独立接上。

## Local PDF reference

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_DemoGen_Synthetic_Demonstration_Generation_for_Data_Efficient_Visuomotor_Policy_Learning.pdf]]