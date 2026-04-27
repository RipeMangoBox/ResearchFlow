---
title: "Distillation-PPO: A Novel Two-Stage Reinforcement Learning Framework for Humanoid Robot Perceptive Locomotion"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/robot-locomotion
  - task/perceptive-locomotion
  - reinforcement-learning
  - policy-distillation
  - domain-randomization
  - dataset/IsaacGym
  - opensource/no
core_operator: 用教师策略蒸馏作为动作先验正则，并在学生阶段保留PPO探索，从而在部分可观测地形中学习可落地的人形感知行走策略
primary_logic: |
  干净地形scan dots/历史状态/本体感觉/速度命令 → 教师在全可观测MDP中学习稳定步态先验，学生继承网络结构与权重并在带噪scan dots的POMDP中联合优化蒸馏正则与PPO目标 → 输出能在真实复杂地形上部署的18维关节目标位置策略
claims:
  - "D-PPO在学生阶段联合优化动作蒸馏项与PPO项，而不是只做监督模仿或只做强化学习 [evidence: theoretical]"
  - "在作者给出的定性对比表中，D-PPO相较only distillation和only RL同时表现出更低训练难度、更好控制性能和更强噪声鲁棒性 [evidence: comparison]"
  - "训练后的Tien Kung机器人可在真实高台、楼梯、坡面和小沟壑等复杂地形上完成感知行走演示 [evidence: case-study]"
related_work_position:
  extends: "PPO (Schulman et al. 2017)"
  competes_with: "Learning Vision-Based Bipedal Locomotion for Challenging Terrain (Duan et al. 2024); Neural Volumetric Memory for Visual Locomotion Control (Yang et al. 2023)"
  complementary_to: "Fast-LIO2 (Xu et al. 2022); Elevation Mapping for Locomotion and Navigation using GPU (Miki et al. 2022)"
evidence_strength: weak
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Distillation_PPO_A_Novel_Two_Stage_Reinforcement_Learning_Framework_for_Humanoid_Robot_Perceptive_Locomotion.pdf
category: Embodied_AI
---

# Distillation-PPO: A Novel Two-Stage Reinforcement Learning Framework for Humanoid Robot Perceptive Locomotion

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2503.08299)
> - **Summary**: 这篇论文把“教师蒸馏的稳定性”和“PPO持续探索的上限”拼到一起，用两阶段训练缓解人形机器人感知行走里“纯蒸馏受老师限制、纯RL又太难学”的矛盾。
> - **Key Performance**: 单台 RTX 4090 上使用 4096 个 Isaac Gym 并行环境训练；真实 Tien Kung 在高台、楼梯、坡面与小沟壑上完成通过演示（论文未报告统一成功率/跌倒率等标准化数值）

> [!info] **Agent Summary**
> - **task_path**: 带噪地形 scan dots + 本体感觉 + 历史状态 + 速度命令 -> 18维关节目标位置
> - **bottleneck**: 纯蒸馏受教师误差与性能上限束缚，纯PPO在POMDP下学习视觉-动作映射又难且不稳
> - **mechanism_delta**: 把教师动作蒸馏从“硬模仿目标”改成“RL训练中的正则项”，并用教师权重初始化对称学生网络
> - **evidence_signal**: 真实 Tien Kung 跨高台/楼梯/坡面/沟壑演示 + Table II 的定性对比
> - **reusable_ops**: [teacher-student weight inheritance, distillation-plus-ppo joint objective]
> - **failure_modes**: [感知噪声过大时策略退化为依赖接触探路, 教师在未见地形上的错误先验会经蒸馏误导学生]
> - **open_questions**: [学生能否稳定超越教师而非仅在其附近微调, 缺少标准化成功率与消融后泛化边界仍不清楚]

## Part I：问题与挑战

这篇论文真正要解决的，不是“让人形机器人会走”，而是：

**如何让机器人在复杂地形上，依赖不完美感知而不是脚底碰撞反馈，提前做出稳定动作，并且最终能落到真机。**

### 任务接口
- **输入**：地形 scan dots（由 elevation map 采样而来）、本体感觉、50 帧历史状态、步态信号、速度命令
- **输出**：18 个驱动关节的目标位置
- **场景边界**：以复杂静态地形行走为主，包括高台、楼梯、坡面、小沟壑、非规则地形

### 真正瓶颈
论文抓到的瓶颈很准：**感知行走训练里有两种都不完美的路线**。

1. **纯两阶段蒸馏路线**
   - 优点：老师在全可观测 MDP 下学得更稳，学生容易收敛
   - 缺点：
     - 学生上限容易被老师封顶
     - 老师依赖仿真里的 privileged information，真机上这些信息并不存在
     - 一旦老师在未见地形上判断错，学生几乎没有纠偏空间

2. **纯端到端 RL 路线**
   - 优点：学生不受老师限制，理论上上限更高
   - 缺点：
     - 在 POMDP 下从带噪感知直接学动作非常难
     - 视觉/地形输入使样本效率更差
     - 真机稳定性容易崩

### 为什么现在值得做
原因有两个：
- **训练侧**：Isaac Gym 这类大规模并行仿真让人形 locomotion 的 RL 终于可操作
- **部署侧**：LIO、深度相机和 elevation map 让机器人有机会在真机上获得可用的地形表征

一句话概括这部分：  
**真正难的不是步态奖励怎么写，而是如何在“教师先验”和“RL探索”之间找到一个不会互相抵消的中间点。**

## Part II：方法与洞察

### 方法结构

论文提出的 D-PPO 可以理解为：

**先学一个“看得清地形”的老师，再训练一个“看得不那么清楚”的学生；但学生不是死记老师动作，而是在老师约束下继续用 PPO 学。**

#### 1）教师阶段：先在全可观测条件下学稳定先验
- 教师直接使用仿真中干净的 scan dots
- 用 Conv1D 压缩地形向量，用历史编码器压缩 50 帧状态
- 再与本体感觉、命令、步态信息一起进入 MLP
- 在这一阶段，目标是先得到**稳定、可行、能跨复杂地形的动作先验**

一个值得注意的设计是：  
作者**刻意不往教师里塞过多 privileged information**。原因是他们认为仿真里的特权信息并不完全可靠，塞太多反而会把仿真假设硬编码到学生里，伤害真机性能。

#### 2）学生阶段：带噪观测下联合蒸馏与 PPO
- 学生网络与教师保持**对称结构**
- 学生直接**继承教师权重**作为初始化
- 学生输入的是**加噪的 scan dots**
- 训练目标不是单独 imitation，也不是单独 PPO，而是两者联合

这里最关键的不是“有蒸馏”，而是**蒸馏在 D-PPO 里只是 regularizer，不是唯一目标**。  
也就是说，学生会被老师“拉回合理动作区域”，但不会被强制锁死在老师动作上。

#### 3）感知管线：从原始点云到 scan dots
为了让真机能提供可部署输入，论文用了比较完整的一套感知链路：
- Livox MID-360 + IMU
- Fast-LIO2 做位姿估计
- 深度相机生成点云
- elevation map 维护局部地形高度
- 从地图上采样成 441 维 scan dots

同时，作者还处理了一个很实际的问题：  
**人形机器人腿部会遮挡深度图。**  
他们用关节角 + 正向运动学估计自身 body bounding box，剔除被机器人身体遮住的区域。这类工程细节对真机部署很关键。

### 核心直觉

D-PPO 的核心不是“蒸馏 + PPO”这个表面组合，而是它改变了**学生策略的因果训练方式**：

- **以前的蒸馏**：学生被要求复制老师  
  → 约束很强，训练容易  
  → 但老师错了，学生也跟着错；老师上限就是学生上限

- **以前的纯 PPO**：学生自己在部分可观测环境里摸索  
  → 自由度高  
  → 但搜索空间太大，稳定性和样本效率都差

- **D-PPO 的改变**：让老师定义“合理动作邻域”，让 PPO 决定“何时留在邻域内、何时偏离以获得更高回报”  
  → 训练分布更稳定  
  → 探索空间被收缩但没被封死  
  → 学生有机会在 noisy POMDP 下学出比纯 imitation 更适合真机的策略

这背后的因果链条可以概括成：

**把“硬模仿”改成“带先验的探索”  
→ 改变了学生面对 POMDP 时的信息瓶颈和优化难度  
→ 提升训练稳定性，并保留超过教师或纠正教师错误的可能性。**

另一个关键点是**weight inheritance**。  
学生不是从头学，而是从已经稳定的 gait basin 出发。对人形这种高维、容易摔倒的系统，这个初始化本身就是巨大收益。

### 战略权衡表

| 方案 | 学生受到的约束 | 解决的问题 | 主要代价 |
|---|---|---|---|
| 仅蒸馏 | 强动作匹配 | 训练容易、早期稳定 | 上限受教师限制，teacher error 会传递 |
| 仅PPO | 几乎无先验 | 理论上限高、能自由探索 | POMDP 下难学，视觉-动作映射样本效率低 |
| **D-PPO** | 蒸馏作软正则 + PPO探索 | 兼顾收敛稳定性与性能上限 | 需要平衡蒸馏权重与RL权重；依赖一个足够好的教师 |
| 过强蒸馏权重 | 几乎退回 imitation | 更稳 | 难以超越教师 |
| 过弱蒸馏权重 | 接近纯RL | 探索自由 | 稳定性下降，训练容易发散 |

## Part III：证据与局限

### 关键证据看什么

这篇论文最强的证据不是公开 benchmark 数字，而是**真机系统演示**。所以读者应该把它当作“有潜力的系统方案”，而不是“已被标准化证明的 SOTA”。

#### 信号 1：真实机器人案例信号
- **类型**：case-study
- **结论**：训练后的 Tien Kung 能跨越高台、楼梯、坡面、小沟壑和不规则地形
- **意义**：说明策略不只是仿真里能跑，而是与感知链路结合后能在真机上工作

论文展示的一个有价值细节是：  
机器人在不同地形上的姿态会改变——平地更直膝、跨障碍时会屈膝并摆臂保平衡。  
这比“只给一张成功截图”更能说明策略确实在用地形信息调动作。

#### 信号 2：方法对比信号
- **类型**：comparison
- **结论**：Table II 中，D-PPO 相比 only distillation 和 only RL 被作者总结为同时具备更低训练难度、更好控制性能和更强噪声鲁棒性
- **意义**：支持论文主张——D-PPO 确实是在两条路线之间做折中，而不是简单拼接

但要注意：  
这个对比是**定性表格**，不是严格数值 benchmark。

#### 信号 3：系统链路支持信号
- **类型**：analysis
- **结论**：LIO + elevation map + body-occlusion handling + noisy scan-dot training 共同支撑了 sim-to-real
- **意义**：说明论文的提升不只来自损失函数，感知表征和部署链路同样关键

### 1-2 个最关键“指标”
论文缺少标准化数值指标，因此最该记住的是这两个“硬信息”：
1. **训练配置**：1× RTX 4090，4096 个 Isaac Gym 并行环境
2. **真实覆盖场景**：高台、楼梯、坡面、小沟壑、非规则地形

### 局限性
- **Fails when**: 传感器噪声过大、LIO 漂移明显、深度图被严重遮挡、地形重建误差大，或遇到动态/可形变/极端未见地形时，学生既可能失去可靠地形输入，也可能被教师正则牵制住，难以及时纠偏。
- **Assumes**: 依赖较强的感知硬件与地图链路（Livox MID-360、IMU、Orbbec 355L、Fast-LIO2、elevation mapping）；依赖大规模并行仿真；依赖一个已经足够稳定的教师；论文未开源实现，复现门槛不低。
- **Not designed for**: 原始图像端到端控制、动态障碍规避、行走同时操作、标准化公开 benchmark 排名证明。

### 证据上的明显缺口
这篇论文目前最缺的是：
- 没有标准化成功率、跌倒率、平均速度、能耗等数值
- 没有 α/β 权重的系统消融
- 没有“是否真的超越教师”的定量验证
- 没有与公开基线在统一地形集上的严格比较

所以它的“能力跃迁”目前更像是：  
**从系统工程上把 teacher-student 和 PPO 融合得更能落地**，而不是已经在公开 benchmark 上彻底坐实优势。

### 可复用组件
下面这些部分是最值得迁移到其他腿足/人形系统里的：
- **蒸馏作为正则而不是最终目标**
- **教师学生对称结构 + 权重继承初始化**
- **elevation map → scan dots 的轻量地形表征**
- **对感知输入加噪做 sim-to-real 对齐**
- **周期奖励 + 速度跟踪 + 正则化约束的奖励组合**

## Local PDF reference

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Distillation_PPO_A_Novel_Two_Stage_Reinforcement_Learning_Framework_for_Humanoid_Robot_Perceptive_Locomotion.pdf]]