---
title: "MANIPTRANS: Efficient Dexterous Bimanual Manipulation Transfer via Residual Learning"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/dexterous-manipulation
  - task/bimanual-manipulation
  - residual-learning
  - reinforcement-learning
  - trajectory-imitation
  - dataset/OakInk-V2
  - dataset/FAVOR
  - dataset/DEXMANIPNET
  - opensource/no
core_operator: 先预训练通用手部轨迹模仿器吸收人手到机器人手的形态差异，再用接触感知残差策略在物理约束下细化双手交互动作
primary_logic: |
  人类双手-物体参考轨迹与对象模型 → 先学习仅关注手部运动的通用模仿策略 → 再在对象状态、接触力与双手协同信息上学习动作残差并配合放松物理课程训练 → 输出满足物理接触约束、可稳定执行的机器人双手操作轨迹
claims:
  - "在 OakInk-V2 验证集上，MANIPTRANS 将双手任务成功率从 Retarget + Residual 的 13.9% 提升到 39.5%，并同时降低物体旋转/平移、关节和指尖误差 [evidence: comparison]"
  - "在 60 帧的“rotating a mouse”单手轨迹上，MANIPTRANS 约 15 分钟即可获得稳健转移结果，而 QuasiSim 官方结果约需 40 小时优化 [evidence: comparison]"
  - "接触力作为观测、奖励和终止条件都对训练有效；去掉阈值放松课程后，复杂双手任务可能无法收敛 [evidence: ablation]"
related_work_position:
  extends: "DexH2R (Zhao et al. 2024)"
  competes_with: "QuasiSim (Liu et al. 2024); DexH2R (Zhao et al. 2024)"
  complementary_to: "Diffusion Policy (Chi et al. 2023); DexCap (Wang et al. 2024)"
evidence_strength: moderate
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_ManipTrans_Efficient_Dexterous_Bimanual_Manipulation_Transfer_via_Residual_Learning.pdf
category: Embodied_AI
---

# MANIPTRANS: Efficient Dexterous Bimanual Manipulation Transfer via Residual Learning

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2503.21860), [Project](https://maniptrans.github.io)
> - **Summary**: 论文把“人类双手 MoCap 到机器人双灵巧手”的困难转移问题拆成两步：先学通用手部运动先验，再只用残差去修正接触与双手协同，因此能更高效地生成高保真、符合物理约束的双手操作轨迹。
> - **Key Performance**: OakInk-V2 上单手/双手成功率达到 **58.1% / 39.5%**（Retarget + Residual 为 **47.8% / 13.9%**）；对 60 帧轨迹，论文报告约 **15 分钟**可得稳健结果，而 QuasiSim 官方结果约 **40 小时**。

> [!info] **Agent Summary**
> - **task_path**: 人类双手-物体 MoCap 轨迹 + 对象网格/动力学信息 -> 仿真中的双灵巧手可执行控制轨迹
> - **bottleneck**: 人手与机器人手的形态差异、接触物理约束和双手高维协同耦合在一起，导致直接 retarget 或从零 RL 都难以稳定收敛
> - **mechanism_delta**: 用预训练模仿器先解决“像不像人手运动”，再让残差策略只处理“是否抓稳物体、双手是否协同”这部分物理误差
> - **evidence_signal**: OakInk-V2 对比实验中双手成功率大幅超过 Retarget + Residual，且接触/课程训练消融支持该因果链
> - **reusable_ops**: [手部轨迹模仿预训练, 基于交互状态的动作残差修正]
> - **failure_modes**: [交互姿态噪声过大, 对象网格或铰接动力学模型不准确]
> - **open_questions**: [如何从更噪声的视频手部估计中稳健扩展, 如何自动化构建大规模高保真对象物理模型]

## Part I：问题与挑战

这篇论文要解决的不是普通“手势跟踪”，而是更难的 **人类双手操纵技能向机器人双灵巧手的可执行转移**。输入是人类演示中的：

- 双手 wrist 位姿与速度
- MANO 手部关键点/关节轨迹
- 物体 6DoF 轨迹与速度
- 对象网格与基本物理属性

输出则是在 Isaac Gym 中可执行的机器人控制轨迹，包括：

- 手部关节目标
- wrist 6DoF 控制
- 最终能让物体按参考轨迹被稳定操纵

### 真正的瓶颈是什么？

作者认为难点不在“有没有参考轨迹”，而在于三件事同时发生：

1. **跨形态差异**  
   人手和机器人手的关节结构、尺寸、自由度不同，直接 retarget 往往只能“形似”，不能“可执行”。

2. **接触物理误差会累积**  
   即使 MoCap 很准，精细任务里一点点位姿误差也会在接触瞬间被放大，导致滑脱、碰撞或操作失败。

3. **双手任务的动作空间太大**  
   双手要同时协调各自手指、腕部和对象交互，若从零用 RL 学，探索成本极高；若只做几何对齐，又容易忽略真实接触约束。

### 为什么现在值得解决？

因为数据驱动的 embodied AI 正在变成主流，但高质量、可扩展、像人一样的双手操纵数据仍然稀缺：

- **纯 RL**：需要任务特定奖励，难扩展到复杂双手操作
- **真实遥操作**：昂贵、慢，而且常常只适用于单一硬件
- **直接 retarget**：快，但不满足物理接触，难以生成训练级数据

所以，这个问题本质上是在问：**能否把大量现成的人类手-物演示，低成本地变成机器人可执行的高保真操纵轨迹？**

### 边界条件

这套方法并不是无条件适用，它依赖：

- 有参考的人类手-物轨迹
- 有对象网格，且最好有较可信的物理模型
- 有仿真环境提供接触力、摩擦、重力等信号
- 主战场是仿真中的高保真转移，再进一步服务真实部署

---

## Part II：方法与洞察

MANIPTRANS 的核心设计是一个很清晰的分治思路：

- **第一阶段**：只学“手怎么像人那样动”
- **第二阶段**：只学“在真实接触约束下还要怎么修正”

也就是把一个原本耦合的难问题，拆成 **运动先验学习** 和 **交互误差修正** 两个更可解的子问题。

### 方法总览

#### Stage 1：Hand Trajectory Imitation

第一阶段训练一个通用手部模仿策略 `I`，输入是：

- 当前目标人手轨迹
- 机器人手本体感觉状态（关节、wrist、速度）

目标不是操纵物体，而是 **尽可能稳定地复现人手运动**。奖励主要包括：

- wrist 跟踪
- 手指关键点模仿
- 平滑性约束

其中作者专门强调拇指、食指、中指等高接触频率部位，使模型优先学会对操纵最关键的末端动作。

这一步的价值是：  
先把“跨形态映射”学成一个**可复用先验**，而不是每个任务都从零适配。

#### Stage 2：Residual Learning for Interaction

第二阶段固定已有模仿能力，再训练一个残差模块 `R`。它不重新输出整套动作，而是输出：

- 对 imitation action 的小幅修正 `Δa`

最终动作是：

- `a = a_I + Δa`

这一步新增的信息不再只是手本体状态，而是加入了 **交互状态**：

- 对象位姿/速度
- 质量中心、重力信息
- 对象 BPS 形状编码
- 手-物距离
- 仿真接触力

奖励也从“像人手动作”扩展为：

- 继续保持手部模仿
- 跟随对象参考轨迹
- 形成合适接触力

关键点在于：**残差策略不负责学一切，只负责纠正接触物理带来的偏差。**

#### 训练技巧：让难问题先变容易

作者还加了一个非常工程有效的策略：

- 训练初期把重力放小甚至设为 0
- 把摩擦调高
- 放宽对象/手部误差阈值
- 再逐步恢复真实物理参数

这等于先让系统学会“先抓住、先贴住、先跟上”，再逐步逼近真实物理。

### 核心直觉

**What changed?**  
从“直接在高维双手接触控制上端到端学习”变成“先学手部运动先验，再学局部残差修正”。

**Which bottleneck changed?**  
原来最难的是三种误差混在一起：

- 形态映射误差
- 接触物理误差
- 双手协同误差

分阶段之后：

- 第一阶段主要吸收 **形态差异**
- 第二阶段只处理 **物理接触与协同偏差**

于是搜索空间被显著缩小，RL 不再需要从零同时解决所有问题。

**What capability changed?**  
能力提升体现在三点：

- 更稳定的接触形成
- 更高保真的物体轨迹跟随
- 更可训练的双手复杂操作

### 为什么这个设计有效？

因果上可以这样理解：

1. **模仿器提供低熵初始化**  
   机器人手一开始就会“像人手那样动”，不必在巨大动作空间里盲目探索。

2. **残差只修局部物理误差**  
   真正难的是接触瞬间的偏差，而不是整段动作都要推翻重来；因此 residual learning 很适合。

3. **接触信息把隐变量显式化**  
   接触力、手物距离、对象形状等信息让“抓稳/滑脱”的状态更可观测，减少策略把失败归咎为随机性的可能。

4. **放松物理约束避免早期局部最优**  
   如果一开始就用真实重力和摩擦，模型很容易在“永远抓不住”的状态里卡住；课程训练先降低门槛，再逐渐逼近真实。

### 策略权衡

| 设计选择 | 解决的瓶颈 | 带来的收益 | 代价/风险 |
|---|---|---|---|
| 先训练手部模仿器 | 人手到机器人手的形态差异 | 提供可复用运动先验，减少下游探索 | 需要额外预训练阶段与手部数据 |
| 用残差而非重学整套动作 | 接触修正是局部问题而非全局重建 | 收敛更快、训练更稳 | 若基础模仿器偏差太大，残差上限受限 |
| 加入对象状态、BPS、接触力 | 接触过程不可观测 | 抓取更稳，物体跟随更准 | 强依赖仿真中的对象模型与接触信号 |
| 训练早期放松重力/提高摩擦/放宽阈值 | 复杂双手任务易陷入局部最优 | 大幅提高早期可学性 | 课程 schedule 需要调节 |
| 不用任务特定奖励 | 方法泛化性差的问题 | 更容易扩到多任务、多对象 | 对特别细粒度语义目标的控制可能不如任务定制奖励 |

### 额外产出：DEXMANIPNET

基于 MANIPTRANS，作者把 FAVOR 和 OakInk-V2 等人类演示转成机器人可执行数据，构建了 **DEXMANIPNET**：

- 61 个任务
- 3.3K episodes
- 1.34M frames
- 含约 600 段复杂双手操作

这使论文不只是提出一个 transfer 方法，还顺带提供了一个可用于后续 policy learning 的数据资源。

---

## Part III：证据与局限

### 关键证据

#### 1. 对比实验信号：双手任务提升非常明显
最强证据来自 OakInk-V2 验证集上的对比：

- 相比 **RL-Only**
- 相比 **Retarget + Residual**
- 相比几乎不可用的 **Retarget-Only**

MANIPTRANS 在物体旋转/平移误差、关节误差、指尖误差上都更好，且 **双手成功率从 13.9% 提升到 39.5%**。  
这说明它的增益不只是“动作更像”，而是**真实任务完成度**也明显提高。

#### 2. 效率信号：不是只更准，也更快
作者在与 QuasiSim 近似设置的 60 帧单手轨迹上报告：

- MANIPTRANS：约 15 分钟得到稳健结果
- QuasiSim 官方结果：约 40 小时优化

虽然这不是完全统一协议下的严格 benchmark，但足够说明论文的主张：  
**把动作先验和交互修正拆开，确实能显著缩短转移时间。**

#### 3. 消融信号：提升来源是“接触建模 + 课程训练”，不是偶然
论文做了两组关键消融：

- 去掉接触力作为观测/奖励/终止条件
- 去掉放松重力、增大摩擦、放宽阈值等课程策略

结论很清楚：

- 接触力奖励能提高成功率
- 接触力观测能加快收敛
- 没有阈值放松时，复杂双手任务甚至可能不收敛

这直接支持其核心因果链：  
**成功来自显式接触建模和 staged learning，而不是简单“多训一会儿”。**

#### 4. 泛化与落地信号：跨手型、上真机
作者还展示了：

- Shadow Hand
- articulated MANO
- Inspire Hand
- Allegro Hand

在不改网络超参与奖励权重的前提下，都能完成迁移。  
此外还在真实双臂 + Inspire Hand 系统上回放了双手操作，展示了诸如开牙膏盖等细粒度动作。

#### 5. 数据集价值信号：下游策略学习仍然很难
在 DEXMANIPNET 上，作者用 IBC、BET、Diffusion Policy 做瓶子搬运，最好成功率也只有 **18.44%**。  
这表明数据集不是“太简单的 scripted rollout”，而是真正具有挑战性的 dexterous manipulation 基准。

### 1-2 个应记住的数字

- **双手成功率：39.5%**（Retarget + Residual 为 13.9%）
- **单段 60 帧转移时间：约 15 分钟**（对比 QuasiSim 官方约 40 小时）

### 局限性

- **Fails when**: 输入的交互姿态噪声过大，或对象尤其是铰接对象的网格/质量/关节模型不准确时，策略容易出现接触不稳、物体轨迹偏离甚至整体转移失败。
- **Assumes**: 需要可用的人类手-物参考轨迹、较可靠的对象模型，以及 Isaac Gym 中可访问的接触力与并行物理仿真；训练依赖大规模并行环境（论文使用 4096 env），补充材料还提到预训练模仿器约需 1.5 天单 GPU。
- **Not designed for**: 无对象模型的开放世界操作、可变形物体、严重时序错位的在线实时模仿，或完全跳过仿真直接零样本部署到异构真实硬件。

### 可复用组件

这篇论文里最值得复用的，不是某个 reward，而是这几个“操作子”：

1. **先学 embodiment-level motion prior，再学 task-level residual**
2. **把接触力/手物距离/对象形状显式并入状态**
3. **训练初期放松物理约束，再逐步恢复**
4. **用 transfer pipeline 顺带构建机器人可执行数据集**

如果你以后做的是：

- 人类到机器人动作迁移
- 双手 dexterous manipulation
- 接触丰富操作数据生成

这些设计都很有迁移价值。

## Local PDF reference

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_ManipTrans_Efficient_Dexterous_Bimanual_Manipulation_Transfer_via_Residual_Learning.pdf]]