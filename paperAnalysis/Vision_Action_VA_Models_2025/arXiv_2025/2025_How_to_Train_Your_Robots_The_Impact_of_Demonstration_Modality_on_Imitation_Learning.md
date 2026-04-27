---
title: "How to Train Your Robots? The Impact of Demonstration Modality on Imitation Learning"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/imitation-learning
  - task/robot-manipulation
  - diffusion
  - hybrid-data-collection
  - teleoperation
  - dataset/OpenDrawer
  - dataset/FlipGlass
  - dataset/PushSanitizer
  - opensource/no
core_operator: 统一不同示范模态到相同状态-动作表征下比较其对数据质量的偏置，并用少量 kinesthetic 加大量 VR 的混合采集提升模仿学习效果
primary_logic: |
  人类通过 kinesthetic / VR / spacemouse 提供示范 → 将各模态统一到相同的视觉+本体状态与末端增量动作表征，并分析动作一致性与状态多样性 → 在固定扩散策略下比较下游成功率并设计混合采集方案 → 以更低采集负担获得更高策略性能
claims:
  - "在 Open Drawer 和 Flip Glass 上，使用 kinesthetic teaching 收集的数据训练出的扩散策略优于 VR 和 spacemouse；但在需要较强接触力的 Push Sanitizer 上其表现更差 [evidence: comparison]"
  - "用户在 kinesthetic teaching 上练习时间更短，并报告更低的心理负担与挫败感，但由于体力负担和回放耗时，更倾向于用 teleoperation 做大规模采集 [evidence: comparison]"
  - "将少量 kinesthetic 数据与 VR 数据混合，可使策略平均成功率比单一模态高约 20%，且在 Flip Glass 上 100 条混合示范达到 75% 成功率 [evidence: ablation]"
related_work_position:
  extends: "A Comprehensive User Study on Augmented Reality-based Data Collection Interfaces for Robot Learning (Jiang et al. 2024)"
  competes_with: "AR2-D2 (Duan et al. 2023); Universal Manipulation Interface (Chi et al. 2024)"
  complementary_to: "Diffusion Policy (Chi et al. 2023); Remix (Hejna et al. 2024)"
evidence_strength: moderate
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_How_to_Train_Your_Robots_The_Impact_of_Demonstration_Modality_on_Imitation_Learning.pdf
category: Embodied_AI
---

# How to Train Your Robots? The Impact of Demonstration Modality on Imitation Learning

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2503.07017)
> - **Summary**: 这篇论文说明，机器人模仿学习的关键不只在策略网络，而在“示范是如何被采出来的”：kinesthetic 提供更一致、更干净的监督，teleoperation 提供更低成本、更广覆盖的状态分布，因此最佳方案是两者混合而不是二选一。
> - **Key Performance**: 混合 kinesthetic+VR 数据相对单一模态平均提升约 20% 成功率；Flip Glass 上 100 条混合示范达到 75% 成功率，比 100 条纯 kinesthetic 还高 5%。

> [!info] **Agent Summary**
> - **task_path**: 人类示范模态（kinesthetic / VR / spacemouse）→ 双视角 RGB + 本体状态 → 机器人末端增量位姿与夹爪动作序列
> - **bottleneck**: 示范模态会系统性改变动作一致性、状态覆盖和采集成本；其中 kinesthetic 需要回放恢复动作，在接触任务上最容易失真
> - **mechanism_delta**: 固定同一扩散策略与统一动作表征，只改变数据采集模态，并进一步用“少量高一致性 kinesthetic + 大量低成本 VR”重构训练数据分布
> - **evidence_signal**: 三任务对比、12 人用户研究、动作方差/状态多样性分析与混合数据消融共同指向同一结论：混合模态最优
> - **reusable_ops**: [统一不同示范模态到同一末端增量动作空间, 用动作一致性与状态多样性指导多模态数据配比]
> - **failure_modes**: [强接触任务中 kinesthetic 回放产生高 jerk 和动作错位, VR 旋转控制需频繁重置参考姿态导致较高心理负担]
> - **open_questions**: [如何自动确定各模态最优采样比例, 引入力/触觉传感后 kinesthetic 在接触任务中的劣势是否还能被消除]

## Part I：问题与挑战

这篇论文真正要回答的问题不是“哪种策略网络更强”，而是更前面的一个问题：**人类通过什么接口给机器人做示范，会从源头改变模仿学习的数据质量**。

### 1. 真正的瓶颈是什么？
在人类示范到机器人动作之间，存在一个常被忽略的“接口层”：
- **kinesthetic teaching**：人直接拖动机器人，控制的是关节/位姿轨迹；
- **VR / spacemouse teleoperation**：人通过外设控制机器人末端的增量位姿。

这意味着，人类意图并不是直接落在学习器看到的动作空间里，而是先经过一个**模态相关的变换与偏置**。这个偏置会影响：
- 动作是否一致、平滑；
- 状态覆盖是否足够丰富；
- 视觉观测是否被手遮挡；
- 示范者是否容易上手、是否能持续采大规模数据。

### 2. 为什么现在值得解决？
因为当前大规模机器人数据集大多默认使用 teleoperation，但这更像是**工程上更易扩展**，而不一定是**学习上最优监督**。随着 diffusion policy 这类数据驱动方法越来越强，数据分布的小变化会被放大为明显的成功率差异，所以“示范模态”已经不是采集细节，而是性能主因之一。

### 3. 输入/输出接口与边界条件
作者把问题严格约束在一个可比设置里：
- **硬件**：7-DoF Franka Panda；
- **任务**：  
  - Open Drawer：自由空间 + 精确对齐 + 受约束直线拉动；  
  - Flip Glass：大角度旋转，且接近关节极限；  
  - Push Sanitizer：需要显式接触力；
- **观测**：双 RGB 相机 + 本体状态；
- **动作**：统一为末端增量位姿 + gripper 开合；
- **学习器**：固定为 Diffusion Policy。

因此，论文试图隔离出的因果量很清楚：**不改模型，只改示范模态，看下游 policy 能力怎么变。**

## Part II：方法与洞察

作者的方法贡献并不在于提出新 policy，而在于构造了一个**公平比较不同示范模态的数据生成实验**，并据此提出一个简单但有效的混合采集方案。

### 方法主线

#### 1. 先把不同模态压到同一动作空间
这是整篇论文最重要的设计之一。作者没有让不同模态各用各的控制定义，而是把所有示范最终都映射到：
- 相同的状态表示；
- 相同的末端增量动作表示。

这样，后续性能差异才能主要归因于**示范模态带来的数据分布差异**，而不是动作定义不同。

#### 2. kinesthetic 的关键问题：动作要“回放恢复”
kinesthetic 的优点是：
- 操作者能真实感知关节极限与接触；
- 重放时视觉干净，不会被人手遮挡；
- 常常能给出更稳定的几何轨迹。

但它有一个结构性问题：**示范时没有直接记录“命令动作”，只能先录位姿，再通过 replay 反推出动作**。  
这在自由空间里问题不大；但在接触任务里，一旦机器人因为接触力没能达到目标位姿，回放恢复出来的动作就会和真实意图错开。作者只能用一个启发式误差补偿去二次 replay，这直接导致 Push Sanitizer 上的动作变得更 jerky。

#### 3. teleoperation 的优势：直接动作、覆盖更广
VR 和 spacemouse 都是 teleoperation：
- VR 更接近“手怎么走，末端就怎么走”的空间映射；
- spacemouse 更像速度/方向控制。

它们的共同优点是：
- 不需要 replay 恢复动作；
- 采集成本低，适合大规模；
- 轨迹通常覆盖更多状态区域。

但缺点是：
- 操作负担更高；
- 旋转控制不自然；
- 动作一致性往往不如 kinesthetic。

#### 4. 用数据质量指标解释性能差异
作者借用了 imitation learning 数据质量文献里的两个透镜：
- **动作一致性**：相近状态下，动作是否稳定；
- **状态多样性**：采样是否覆盖足够多的状态。

这两者刚好揭示了三种模态的互补性：
- kinesthetic：更高动作一致性；
- teleoperation：更高状态多样性。

#### 5. 基于互补性提出混合采集
最终方案非常朴素：
- 用**少量 kinesthetic** 提供高一致性“锚点”数据；
- 用**更多 VR** 扩展状态覆盖，同时降低体力负担。

论文的价值恰恰在于：这个方案不是拍脑袋，而是由前面的性能差异、用户体验和数据质量分析共同推出来的。

### 核心直觉

**作者真正引入的因果旋钮，不是模型结构，而是训练数据的生成机制。**

- **what changed**：从“单一示范模态采集”改为“显式利用模态偏置，并进行模态混合”；
- **which bottleneck changed**：  
  - kinesthetic 降低了局部动作不确定性；  
  - VR/spacemouse 提高了状态覆盖；  
  - 混合后缓解了“一致性 vs 覆盖”的数据瓶颈；
- **what capability changed**：在相同 policy 架构和相近采集预算下，策略更容易学到既稳定又有泛化余量的动作映射。

为什么这有效？因为行为克隆类方法非常依赖“相近状态是否对应相近动作”。  
- 如果数据**一致性太差**，模型会学到模糊平均动作；  
- 如果数据**覆盖太窄**，模型容易在部署时掉出支持集。  

kinesthetic 解决前者，VR 更擅长解决后者。把二者混起来，本质上是在重塑训练分布的**局部平滑性**和**全局覆盖度**。

但这个逻辑有明显边界：一旦任务核心不是几何轨迹，而是**力/接触控制**，kinesthetic 的 replay 恢复就会破坏动作语义，原本的“高质量监督”优势就消失了。

### 策略性权衡

| 方案 | 控制方式 | 数据侧收益 | 人类侧代价 | 适用边界 |
|---|---|---|---|---|
| Kinesthetic | 直接拖动机器人，后续 replay 恢复动作 | 动作更一致、视觉更干净、精确轨迹更稳定 | 体力负担高、回放耗时、接触任务恢复动作困难 | 非接触、几何精确型任务 |
| VR Teleoperation | 手部空间位姿映射到末端增量 | 状态覆盖更广、采集更高效 | 旋转控制不直观、心理负担较高 | 大规模自由空间/中等约束任务 |
| Spacemouse | 按钮/速度式增量控制 | 低成本、单手操作 | 学习成本较高、路径偏置明显 | 预算敏感的 teleop 场景 |
| Mixed (Kinesthetic + VR) | 少量 kinesthetic + 大量 VR | 同时兼顾一致性与覆盖，整体性能最好 | 需要额外决定混合比例 | 适合规模化 imitation 数据采集 |

## Part III：证据与局限

### 关键证据信号

- **信号 1｜跨任务比较**
  在固定 Diffusion Policy 的前提下，kinesthetic 数据在 **Open Drawer** 和 **Flip Glass** 上学出的策略最好；但在 **Push Sanitizer** 上反而更差。  
  **结论**：kinesthetic 的优势依赖于“位姿 replay 仍能忠实恢复动作”这个前提；一旦任务强依赖接触力，这个前提就失效。

- **信号 2｜用户研究**
  12 名参与者中，kinesthetic 的练习时间显著更短（论文报告 p=0.038），并在心理负担、挫败感、主观表现上更优；但在“大规模数据采集会选哪种模态”这一题上，多数人仍选择 teleoperation。  
  **结论**：最直观的接口，不一定是最适合扩展到大规模数据生产的接口。

- **信号 3｜机制解释**
  数据统计显示：  
  - kinesthetic 在非接触任务上动作方差更低、jerk 更小；  
  - teleoperation 在状态空间中覆盖更广。  
  **结论**：性能差异不是偶然，而是由“动作一致性 vs 状态多样性”的分布属性驱动。

- **信号 4｜混合数据消融**
  少量 kinesthetic 与 VR 混合后，策略平均成功率比单模态高约 **20%**；在 Flip Glass 上，**100 条混合示范达到 75% 成功率**。  
  **结论**：模态混合不是折中，而是能真正带来能力增益的分布重构。

### 局限性

- **Fails when**: 任务需要显式力控制、强接触或接触过程主导动作语义时，kinesthetic 的位姿 replay 会恢复出错误或高 jerk 的动作；此外，非专家 teleoperation 数据噪声较大，机制分析对人群分布较敏感。
- **Assumes**: 论文假设所有模态都能统一到同一个末端增量动作空间，并固定使用 Franka + Polymetis + 双相机 + Diffusion Policy；主训练集主要来自 1 名熟练示范者、每模态每任务 100 条；且没有力传感器，只能用启发式误差补偿处理 kinesthetic 回放误差。
- **Not designed for**: 双臂操作、灵巧手、移动操作、长时程复合任务、跨机器人泛化，也不解决“混合比例如何自动搜索”的问题；同时未看到明确代码/项目开源，复现需要自行搭建采集与控制栈。

### 可复用组件

1. **统一动作空间的跨模态比较协议**  
   这对任何“比较不同人机示范接口”的研究都很重要，否则结论很容易混入动作定义差异。

2. **动作一致性 / 状态多样性作为数据诊断工具**  
   在机器人 imitation 数据采集阶段就能提前判断哪些数据“更值得采”。

3. **少量高质量模态 + 大量低成本模态的混合采集范式**  
   这不局限于 kinesthetic+VR，也可以推广到触觉示范、视觉 teleop、leader-follower 等接口组合。

整体上，这篇论文的“能力跳跃”不在于提出新网络，而在于把一个常被当作工程细节的变量——**示范模态**——提升为可测量、可分析、可设计的学习变量，并用相对简单的混合策略拿到了清晰收益。证据链是完整的，但任务数量、硬件单一性和接触任务处理方式限制了其外推力度，因此给 `moderate` 是合理的。

## Local PDF reference

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_How_to_Train_Your_Robots_The_Impact_of_Demonstration_Modality_on_Imitation_Learning.pdf]]