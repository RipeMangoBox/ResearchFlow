---
title: "CoinRobot: Generalized End-to-end Robotic Learning for Physical Intelligence"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/robot-manipulation
  - diffusion
  - multitask-learning
  - dataset/CoinRobot
  - opensource/no
core_operator: 通过 ROS2 翻译层统一异构机器人与传感器接口，并在统一观测空间上训练可替换的 DDIM 扩散策略，实现跨平台端到端操控。
primary_logic: |
  异构机器人多视角观测与状态输入 → Translation 模块统一 ROS2 消息并按时间戳对齐，再送入感知编码器与 DDIM 动作预测器进行单任务/多任务训练与微调 → 输出适配不同机器人平台的动作轨迹与操控策略
claims:
  - "在 PickPlace 任务上，CoinRobot 训练的 Diffusion Policy 成功率为 64%，高于 LeRobot 上的同类 Diffusion Policy（0%）和 ACT（30%）[evidence: comparison]"
  - "以 SlamDunk 单任务检查点为初始化，在 SlamDunk+Sorting 合并数据上仅追加 50 个 epoch 微调后，模型即可完成 Sorting，显示相关操作技能可跨任务复用 [evidence: case-study]"
  - "在 Sorting 任务中，加入前视/侧视各 40 条示范并额外微调 100 个 epoch 后，模型在未见相机位姿上的成功率达到 20%，而固定视角基线为 0% [evidence: comparison]"
related_work_position:
  extends: "Diffusion Policy (Chi et al. 2024)"
  competes_with: "LeRobot (Cadene et al. 2024); ACT (Zhao et al. 2023)"
  complementary_to: "Open X-Embodiment (O'Neill et al. 2024); Octo (Ghosh et al. 2024)"
evidence_strength: moderate
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_CoinRobot_Generalized_End_to_end_Robotic_Learning_for_Physical_Intelligence.pdf
category: Embodied_AI
---

# CoinRobot: Generalized End-to-end Robotic Learning for Physical Intelligence

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv:2503.05316](https://arxiv.org/abs/2503.05316)
> - **Summary**: 论文提出 CoinRobot，用统一的 ROS2 翻译层把异构机器人、传感器、数据采集、训练与部署串成同一端到端流水线，并用扩散式策略在 7 个真实操控任务上验证跨平台适配与有限泛化能力。
> - **Key Performance**: PickPlace 上 DP/CoinRobot 64%，高于 DP/LeRobot 0% 与 ACT/LeRobot 30%；Sorting 上加入双视角各 40 条示范后，未见相机位姿成功率 20%，基线为 0%。

> [!info] **Agent Summary**
> - **task_path**: 多视角相机图像 + 机器人状态 / 远程示教模仿学习 -> 机器人动作轨迹
> - **bottleneck**: 异构机器人/传感器/模型平台之间接口强耦合，且策略对物体位置与相机视角分布变化泛化差
> - **mechanism_delta**: 用 ROS2 Translation 层统一消息与时间对齐，再在统一接口上训练可替换的感知-扩散动作策略，并用合并数据微调实现跨任务迁移
> - **evidence_signal**: 7 个真实任务上的框架对比与泛化实验：PickPlace 64% vs 0%/30%，未见视角 20% vs 0%
> - **reusable_ops**: [ROS2统一翻译层, 时间戳后对齐, 多视角增量微调]
> - **failure_modes**: [固定视角训练对相机位姿变化脆弱, ACT在位置变化任务上退化到0%]
> - **open_questions**: [如何把3000+示教需求降到更低, 如何实现更强的无微调跨本体迁移]

## Part I：问题与挑战

这篇论文真正想解决的，不只是“再换一个 policy 网络”，而是**端到端机器人学习的系统级瓶颈**：

1. **工程瓶颈**：  
   每换一个机器人、相机或训练平台，就要重写数据采集、ROS 消息解析、推理下发和传感器适配代码。  
   对很多真实机器人研究来说，瓶颈不是模型本身，而是这层反复重造的 glue code。

2. **分布瓶颈**：  
   模仿学习里的动作分布天然是多模态的。同一视觉状态下，人类示教可能对应多条都能成功的动作轨迹。  
   一旦物体初始位置变化、相机角度偏移，偏确定性或单任务训练的策略容易崩。

3. **迁移瓶颈**：  
   现有工作常是“一任务一模型”。这导致相近任务之间无法有效共享抓取、移动、放置等基础动作原语。

### 输入/输出接口

- **输入**：多视角 RGB / RGB-D 图像 + 低维机器人状态（如 end-effector pose）
- **输出**：机器人可执行动作轨迹/动作块
- **学习范式**：远程示教下的 imitation learning
- **不是本文重点**：强化学习、语言条件规划、移动机器人导航

### 边界条件

这套方法主要在以下设置上验证：

- 6/7-DOF 机械臂
- 桌面操作任务
- ROS2 通信生态
- 多相机观测
- 采集后统一对齐到 10 Hz 的离线示教数据

所以它证明的是：**在“真实机械臂操控 + 异构设备接入”这个范围内，系统级抽象和扩散策略组合是有效的**；并没有证明它对更广义 embodiment 都成立。

### 为什么现在值得做

因为三个条件同时成熟了：

- 低成本机械臂/VR 示教工具更普及
- ROS2 已成为现实机器人接入的事实标准之一
- Diffusion Policy 这类生成式控制模型开始显示出比显式行为克隆更好的多模态建模能力

所以现在的关键问题，已经从“能不能训一个 policy”转向“能不能把采集、训练、部署统一起来并可迁移”。

---

## Part II：方法与洞察

CoinRobot 的核心不是某个单一网络模块，而是一个**系统抽象 + 策略建模**的组合。

### 1. 系统层：Collection / Translation / Inference 三段式

#### Collection
- 通过 ROS2 在远程操作时采集原始 topic
- 记录多源传感器的时间戳
- 采集阶段不强行做同步，先保留原始频率

#### Translation
- 订阅不同机器人/传感器的 ROS2 topic
- 把异构消息转成统一字典格式
- 再封装成 JSON 消息发布

这一步是全文最关键的工程抽象：  
**机器人差异被压缩到 translation package，而不是污染上层训练/推理代码。**

#### Inference
- 复用采集阶段的数据抓取模块
- 再通过 model bridge 模板接到具体模型平台
- 从而减少“训练能跑、部署又得重写一遍”的重复劳动

### 2. 数据层：后对齐而非采集时强同步

作者让不同传感器先按各自频率采集，再按时间戳后处理对齐，最终对齐到 10 Hz。  
这个设计改变的是系统约束：

- 采集时更简单
- 兼容不同频率设备
- 但时间分辨率最终受限于最低频传感器与对齐策略

对真实机器人系统来说，这是一个很实用的工程取舍。

### 3. 策略层：感知模块 + 动作预测模块解耦

#### 感知模块
输入包括：

- wrist camera
- 外部固定视角相机
- 低维状态（eef pose 等）

作者测试了多种视觉骨干，文本中偏好 **FPN-based ResNet34**。  
但从表格看，它并不是所有任务绝对最优，更准确地说是：**多尺度视觉表征在部分任务上更稳，但收益具任务依赖性。**

#### 动作预测模块
作者采用 **DDIM-based diffusion policy** 作为核心动作生成器，并尝试 CNN / Transformer 变体。

他们的判断是：

- ACT 擅长较平滑、较固定的轨迹
- 但在物体位置变化时，显式策略对多模态分布更脆弱
- diffusion 更适合表示“一种状态对应多条可行动作”的情况

### 4. 训练层：从单任务到多任务/多视角微调

作者没有直接追求“全能基础模型”，而是采用更务实的策略：

- 先训单任务 checkpoint
- 再在合并数据集上微调
- 让相关任务共享已有动作表示

同样地，在相机泛化上，他们不是直接追求强零样本，而是：

- 固定 wrist camera
- 给第二相机补充少量前视/侧视数据
- 再做增量微调

这体现出本文的整体路线：**不是一次性解决一切泛化，而是尽量降低“适配新分布”的成本。**

### 核心直觉

- **改变了什么**：  
  过去“每个机器人一套私有数据接口、每个模型平台一套独立推理代码”的模式，被改成“底层异构、上层统一”的 translation abstraction；同时动作建模从更偏单峰的显式输出，切到扩散式分布建模。

- **改变了哪类瓶颈**：  
  1) **系统约束瓶颈**：上层模型不再直接面对厂商各异的 ROS 消息格式。  
  2) **信息瓶颈**：多视角 + 低维状态统一编码，减少单视角观测歧义。  
  3) **分布瓶颈**：扩散策略不必把多种正确动作平均成一个“不可执行均值”。

- **能力发生了什么变化**：  
  更容易接新机器人/新传感器；对位置变化和一定程度的视角变化更稳；相近任务之间可以从已有 checkpoint 更快迁移。

- **为什么这套设计有效**：  
  因为它把“硬件差异”从学习问题中剥离出去，把“多解动作”交给更适合建模分布的 diffusion，把“任务差异”尽量压缩成共享动作原语上的微调问题。

### 策略性取舍表

| 设计选择 | 改变的约束/瓶颈 | 带来的收益 | 代价 |
|---|---|---|---|
| ROS2 Translation 层 | 硬件消息格式耦合 | 新机器人/传感器接入更快，上层代码更稳定 | 新机器人仍需写 translation package |
| 采集后时间对齐 | 采集时同步复杂度 | 兼容异频设备，工程实现简单 | 最终控制频率与时序精度受限 |
| DDIM 动作策略 | 多模态动作分布难建模 | 对位置变化更稳，避免平均动作 | 推理与训练通常更重，且数据需求仍高 |
| 单任务 checkpoint + 合并数据微调 | 每任务从零训练 | 更快迁移到相关任务 | 任务差异太大时可能相互干扰 |
| 多视角少量增量数据 | 固定视角 train-test mismatch | 提升相机位姿泛化 | 仍不是强零样本，泛化幅度有限 |

### 一个额外但实用的 insight

作者明确指出：**训练 loss 与真实任务成功率弱相关**。  
因此他们主张用：

- action prediction MSE
- 动作轨迹可视化

来辅助判断模型是否真的学到了可执行行为。  
这不是算法突破，但对真实机器人训练流程很有操作价值。

---

## Part III：证据与局限

### 关键证据信号

1. **框架对比信号（comparison）**  
   在 PickPlace 上，DP/CoinRobot 达到 **64%**，而 DP/LeRobot 为 **0%**，ACT/LeRobot 为 **30%**。  
   这说明本文的收益不只来自“换了个模型名”，还来自系统接入、训练配置和整体流水线的组合优化。

2. **分布变化诊断信号（comparison）**  
   在 CleanDish、Gathering、CollectDish 这类位置变化任务里，ACT 在固定物体位置时能到 **78%–90%**，但变位置后掉到 **0%**。  
   这很好地支撑了作者的核心论点：**多模态动作/观测分布变化是显式策略的薄弱点。**

3. **跨任务复用信号（case-study）**  
   从 SlamDunk checkpoint 出发，在 SlamDunk + Sorting 的合并数据上只追加 **50 个 epoch**，模型就能学会 Sorting。  
   这表明本文更像是在做“共享动作表示的快速适配”，而不是完全独立地重训每个任务。

4. **视角泛化信号（comparison）**  
   在 Sorting 中，只增加前视/侧视各 40 条示范并再训 100 epoch，未见相机位姿成功率达到 **20%**；固定视角基线为 **0%**。  
   这说明它对视觉分布偏移有一定恢复能力，但离强鲁棒还很远。

### 能力跃迁到底在哪里

这篇论文最有价值的“jump”不是绝对 SOTA 数字，而是两点：

- **系统层 jump**：把机器人接入、采集、训练、部署抽成统一接口，降低真实机器人研究的迁移成本。
- **分布层 jump**：用 diffusion + 少量增量数据微调，让模型对位置变化、相机变化和相近任务迁移表现出比传统单任务/固定视角设置更好的韧性。

### 1-2 个最关键指标

- **PickPlace**：64% vs 0%/30%
- **Sorting 未见视角**：20% vs 0%

这两个数分别对应本文最核心的两个主张：

- 统一框架 + diffusion 训练范式有实际收益
- 少量多视角增量数据能带来一定真实场景泛化

### 局限性

- **Fails when**:  
  相机位姿变化但训练中未覆盖该视觉分布时，固定视角模型会失败；物体初始位置变化较大时，ACT 类显式策略会显著退化；即使做了多视角微调，未见视角成功率也只有 20%，说明强泛化仍未解决。

- **Assumes**:  
  依赖远程示教和较大量数据（文中总计 3000+ episodes）；依赖 ROS2 驱动生态、多相机、leader arm/Oculus 等硬件；新机器人虽然接入更快，但仍需编写 translation package；实验主要围绕机械臂桌面操作，不是任意 embodiment。另一个现实问题是：正文虽声称会发布数据和模型，但未给出明确代码仓库链接，因此公开可复现性目前仍受限。

- **Not designed for**:  
  语言条件任务执行、完全无微调的跨本体零样本迁移、移动操作/导航/腿足机器人、跨实验室标准 benchmark 下的严格 SOTA 竞争。

### 可复用组件

- **ROS2 统一翻译层**：适合任何异构机器人接入场景
- **采集/推理共享 data-fetcher**：减少 train-deploy interface drift
- **时间戳后对齐策略**：适合多源异频传感器系统
- **checkpoint + 合并数据微调**：适合相近操作任务的快速迁移
- **action MSE + trajectory visualization**：适合真实机器人训练时的实用诊断

## Local PDF reference

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_CoinRobot_Generalized_End_to_end_Robotic_Learning_for_Physical_Intelligence.pdf]]