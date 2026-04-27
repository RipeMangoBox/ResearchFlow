---
title: "Vision in Action: Learning Active Perception from Human Demonstrations"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/bimanual-manipulation
  - task/active-perception
  - diffusion
  - teleoperation
  - point-cloud-rendering
  - "dataset/Bag Task"
  - "dataset/Cup Task"
  - "dataset/Lime & Pot Task"
  - opensource/no
core_operator: 通过6-DoF主动颈部、共享观测VR示教和世界坐标点云中间表示，把人类“看哪里”的策略直接蒸馏进双臂操作策略。
primary_logic: |
  遮挡严重的双臂操作任务 + 人类共享观测VR示教 → 用6-DoF头部相机和世界坐标点云解耦视图渲染与机器人相机运动，并记录头/臂协同轨迹 → 扩散策略根据头部图像与本体状态预测未来颈部和双臂动作 → 输出具备搜索、跟踪、聚焦能力的主动感知操作策略
claims:
  - "ViA在三个含严重视觉遮挡的多阶段双臂任务上，相比固定胸部+腕部相机配置平均提升45%成功率 [evidence: comparison]"
  - "在相同示教数据上，单主动头部相机的ViA平均优于“主动头部+腕部相机”配置18.33%，说明额外腕部视角在该低数据设定下未带来收益 [evidence: comparison]"
  - "在8名参与者的VR遥操作对比中，点云渲染界面将主观晕动症评分从3.375降到2.0，并获得6/8用户偏好 [evidence: comparison]"
related_work_position:
  extends: "Diffusion Policy (Chi et al. 2023)"
  competes_with: "Open-TeleVision (Cheng et al. 2024); Learning to Look Around (Sen et al. 2024)"
  complementary_to: "DP3 (Ze et al. 2024); 4D Gaussian Splatting (Wu et al. 2024)"
evidence_strength: moderate
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Vision_in_Action_Learning_Active_Perception_from_Human_Demonstrations.pdf
category: Embodied_AI
---

# Vision in Action: Learning Active Perception from Human Demonstrations

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2506.15666), [Project](https://vision-in-action.github.io)
> - **Summary**: 这篇工作把双臂机器人的“主动看哪里”从手工设计改成从人类示教中直接学习，通过6-DoF头部相机与低延迟VR共享观测，让策略在遮挡场景中学会搜索、跟踪和聚焦。
> - **Key Performance**: 相比胸部+腕部相机，三任务平均成功率提升 **45%**；VR用户研究中晕动症评分 **2.0 vs 3.375**，**75%** 用户偏好ViA。

> [!info] **Agent Summary**
> - **task_path**: 头部RGB观测 + 颈部/双臂本体状态 -> 颈部视角动作 + 双臂末端执行器动作
> - **bottleneck**: 人类示教时依赖主动视线搜索，但机器人训练常只看到固定或腕部视角，导致关键任务信息在遮挡场景中缺失
> - **mechanism_delta**: 用共享观测VR把人类看到的视角直接变成机器人训练视角，并用世界坐标点云渲染解耦“人头动”和“机器人相机真正移动”
> - **evidence_signal**: 三个遮挡型双臂任务上较胸部+腕部相机平均提升45%，且8人用户研究显示更低晕动症与更高偏好
> - **reusable_ops**: [shared-observation-teleop, world-frame-head-arm-action-parameterization]
> - **failure_modes**: [point-cloud-holes-and-depth-noise-hurt-fine-manipulation, no-explicit-memory-causes-inefficient-search]
> - **open_questions**: [how-to-fuse-multi-camera-observations-with-a-shared-representation, how-to-add-language-and-memory-for-open-ended-search]

## Part I：问题与挑战

这篇论文真正要解决的，不是“再加一个相机”这么表面的系统设计问题，而是**示教学习中的观测错配**：

- 人类完成复杂操作时，会主动转头去**搜索**目标、**跟踪**目标、在关键阶段**聚焦**细节。
- 但机器人训练时常见的输入却是固定胸部相机或腕部相机。
- 于是，**人类做决策所依据的信息**，并没有进入机器人最终学习到的观测空间。

这在**视觉遮挡严重的双臂任务**里尤其致命。腕部相机虽然“离手近”，但它的视角受机械臂动作约束，不是按感知目标来动；胸部相机虽然稳定，但遇到袋子内部、货架下层、双臂遮挡时，往往根本看不到关键物体。

### 真正瓶颈是什么？

核心瓶颈有两个，且是系统级耦合的：

1. **信息瓶颈**：机器人没有拿到与人类成功示教一致的视角，因此学不到“为什么此刻要先看再抓”。
2. **交互瓶颈**：即便想用VR采集这种主动视角，传统RGB视频直传会把机器人控制延迟带进VR回路，造成 motion-to-photon latency，用户容易晕动症，难以稳定采集数据。

### 为什么现在值得解决？

因为当前模仿学习/扩散策略在双臂操作上已经能学“怎么动手”，但在遮挡、搜索、阶段切换时仍被**“看不见”**卡住。与此同时，低成本VR、RGB-D手机和可编程机械臂硬件已经让“主动感知示教”变得可实现。

### 输入 / 输出接口与边界条件

**训练/部署输入**
- 单个主动头部相机的RGB观测
- 23维本体状态：颈部、左臂、右臂末端位姿 + 两个夹爪宽度

**策略输出**
- 未来若干步的颈部和双臂末端位姿序列
- 对应夹爪宽度  
也就是：**策略既输出“怎么看”，也输出“怎么操作”**。

**边界条件**
- 桌面级、静态基座的双臂操作
- 以严重遮挡、多阶段任务为主
- 依赖VR示教、RGB-D感知和世界坐标标定
- 控制频率约10 Hz，尚非高速动态操作

## Part II：方法与洞察

ViA的贡献不是单个网络，而是一个闭环系统：**硬件可达视角 + 可用的低延迟示教接口 + 能联合学习头部与双臂动作的策略**。

### 方法骨架

#### 1. 6-DoF 机械臂充当“机器人颈部”
作者没有设计复杂仿人颈椎，而是直接用一台6-DoF机械臂做neck，并把 iPhone 15 Pro 装在末端作为主动头部相机。  
这一步的意义很直接：把“能不能看到”从固定视角，变成可主动控制的自由度。

#### 2. 用世界坐标点云做VR中间表示
传统做法是：用户转头 → 机器人头跟着动 → 新RGB视频传回来 → VR更新。  
问题是：机器人控制延迟 + 视频传输延迟会叠加，造成晕动症。

ViA改成：
- 用头部RGB-D和相机位姿构建世界坐标下的点云
- VR端直接根据**用户最新头姿**实时渲染新视角
- 机器人头部则异步地、低频地追赶这个目标视角

于是，用户看到的是“**立即响应的渲染视图**”，而不是“等机器人真的转过去后的相机画面”。

#### 3. 用 Diffusion Policy 学“头-手协同”
策略基于 Diffusion Policy：
- 视觉编码器：DINOv2 ViT
- 视觉输入：主动头部RGB
- 动作空间：颈部 + 双臂末端位姿，统一在世界坐标系里表示

这个统一表示很关键，因为它把“先转头找东西，再伸手抓”变成一个可联合建模的问题，而不是视觉前端和操作后端各管各的。

### 核心直觉

**改变了什么？**  
从“机器人只看固定/腕部视角”变成“机器人学习时看到的，就是人类完成任务时真正依赖的主动视角”。

**哪种瓶颈被改变了？**  
- 改变的是**观测分布**：成功示教中的搜索/盯视/聚焦行为不再是隐变量，而成为训练输入的一部分。
- 改变的是**实时性约束**：VR显示不再受机器人控制延迟硬绑定，示教才真正可持续。
- 改变的是**协同控制约束**：头部和双臂动作共享世界坐标表示，策略更容易学到“先看后动”的时序结构。

**能力为什么会提升？**  
因为在遮挡任务里，失败往往不是手不够准，而是**没看到**。ViA把“看见关键物体/区域”本身也变成可学习行为，于是策略能先获取信息，再执行动作。  
换句话说，这篇文章的关键不是增强操控器，而是把**信息获取动作**纳入了策略本体。

### 设计取舍

| 设计选择 | 改变的瓶颈 | 带来的能力 | 代价/副作用 |
|---|---|---|---|
| 6-DoF机械臂作为neck | 视角可达性不足 | 能探袋内、看货架深处、切换视角对准目标 | 结构更重，仍不等同于完整人类上身运动 |
| 点云中介渲染而非直接RGB流 | VR端延迟与晕动症 | 用户头动后视角即时更新，示教更稳定 | 渲染会有空洞、深度噪声，细粒度操作不如原始视频清晰 |
| 单主动头视角学习 | 人类-机器人观测错配 | 让示教视角本身变成任务完备输入 | 对长期记忆、场景覆盖仍有限 |
| 世界坐标统一表示头和双臂动作 | 头手协同难建模 | 更容易学到“搜索→操作”的顺序 | 依赖标定、IK和执行稳定性 |

## Part III：证据与局限

### 关键证据

**信号1：相机设置对比，证明收益来自“可控视角”，不是“多装相机”**  
最强证据是 camera setup comparison。ViA 在三个遮挡型任务上，相比最常见的 **Chest & Wrist Cameras** 平均提升 **45%**。  
这说明性能增益主要来自：
- 能主动选择更有信息量的视角
- 训练观测与人类示教观测对齐

而不是简单堆更多传感器。

**信号2：额外腕部相机反而变差，说明不是“越多视角越好”**  
作者发现 **Active Head & Wrist Cameras** 比只用主动头相机的 ViA **平均下降18.33%**。  
这很有价值，因为它说明：
- 在共享观测示教下，主动头视角已经接近任务完备
- 低数据场景下，额外腕部视角更可能引入冗余和遮挡噪声，而不是帮助学习

**信号3：语义预训练对“先搜再动”很重要**  
使用 DINOv2 的 ViA 在最终成功率上优于 ResNet-DP 和 DP3。论文特别指出，ViA能先执行较长时间的主动搜索，再开始操作；而 DP3 会出现“看错/抓空”的 hallucination 式错误。  
这支持了一个机制性结论：**主动感知不仅需要可动视角，还需要足够强的语义表征去解释这个视角。**

**信号4：VR界面比较证明系统可用性**  
8人用户研究中，点云渲染方案把主观晕动症从 **3.375** 降到 **2.0**，并获得 **6/8** 用户偏好。  
虽然采集时间略长（56.74s vs 52.72s），但它换来了更舒适、更可持续的示教过程。这是系统真正能落地采集数据的关键。

### 1-2个关键指标

- **平均成功率提升**：ViA 相对 Chest & Wrist Cameras 平均 **+45%**
- **用户可用性**：晕动症 **2.0 vs 3.375**，用户偏好 **75%**

### 局限性

- **Fails when**: 深度噪声、单帧点云空洞或视角快速变化导致渲染不完整时，用户对细粒度操作的判断会变差；需要长期搜索记忆的任务中，当前策略可能重复查看同一区域。
- **Assumes**: 依赖专门硬件栈（3台ARX5机械臂、iPhone 15 Pro RGB-D、VR头显、双臂外骨骼）、稳定世界坐标标定、IK执行与数十到数百条人类示教；当前系统主要在桌面、10 Hz控制、相对受限任务分布下验证。
- **Not designed for**: 语言条件操作、移动操作/全身导航、无深度传感的设置、开放世界长时程任务推理，以及需要高保真动态场景融合的场景。

### 可复用组件

1. **shared-observation teleoperation**：让人类和机器人共享同一观测空间，用于采集“人类如何看”的示教数据。  
2. **world-frame point-cloud rendering**：把VR显示延迟从机器人控制延迟中解耦。  
3. **head-arm unified action space**：将主动感知和操作统一到同一个策略输出空间中。  

整体上，这篇工作最有价值的地方是提出了一个清晰的系统结论：  
**在遮挡型 embodied 操作里，真正的能力跃迁来自“把感知动作也作为策略的一部分学习”，而不是继续在固定视角上堆更强的操控器或更大的策略网络。**

## Local PDF reference

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Vision_in_Action_Learning_Active_Perception_from_Human_Demonstrations.pdf]]