---
title: "NIL: No-data Imitation Learning by Leveraging Pre-trained Video Diffusion Models"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/imitation-learning
  - task/robot-locomotion
  - diffusion
  - reinforcement-learning
  - vision-transformer
  - dataset/Locomujoco
  - opensource/promised
core_operator: 将单帧初始观测和文本任务描述输入冻结视频扩散模型生成参考视频，再用视频Transformer嵌入距离与分割IoU构造无判别器模仿奖励来训练物理可行的3D控制策略
primary_logic: |
  初始状态渲染帧 + 技能文本 + 机器人形体 → 冻结视频扩散模型生成2D参考视频 → 对生成视频与模拟渲染视频计算TimeSformer嵌入距离和分割IoU并加入平滑正则 → 在物理模拟中优化得到满足动力学约束的策略
claims:
  - "On Unitree H1 walking, NIL learns without curated expert demonstrations and reaches a reported environment reward of 396.1, surpassing AMP trained on motion-capture data (393.5) and approaching the expert score of 400 [evidence: comparison]"
  - "On Talos and Unitree A1 locomotion, NIL outperforms AMP by large margins in the paper's reward metric (352.8 vs 231.1 on Talos; 360.8 vs 286.9 on A1), showing that zero-data imitation can transfer to harder or non-humanoid embodiments in the reported setup [evidence: comparison]"
  - "The composite reward is necessary: on Unitree H1, removing the IoU term increases motion-capture loss from 46.4 to 82.9, and policies trained with only IoU or only regularization fail to maintain stable forward walking [evidence: ablation]"
related_work_position:
  extends: "GAIfO (Torabi et al. 2018)"
  competes_with: "AMP (Peng et al. 2021); DRAIL (Lai et al. 2025)"
  complementary_to: "N/A"
evidence_strength: moderate
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_NIL_No_data_Imitation_Learning_by_Leveraging_Pre_trained_Video_Diffusion_Models.pdf
category: Embodied_AI
---

# NIL: No-data Imitation Learning by Leveraging Pre-trained Video Diffusion Models

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2503.10626)
> - **Summary**: 论文把“专家3D示范”替换成“单帧+文本生成的参考视频”，再用视频嵌入距离和分割IoU做模仿奖励，从而在没有任何外部示范数据的情况下训练多种机器人步行策略。
> - **Key Performance**: Unitree H1 上 NIL 得到 396.1 env reward，超过用 MoCap 数据训练的 AMP 的 393.5；Talos 上 352.8，显著高于 AMP 的 231.1。

> [!info] **Agent Summary**
> - **task_path**: 单帧初始渲染 + 文本任务描述 + 机器人形体 -> 物理模拟中的连续控制/步态策略
> - **bottleneck**: 没有3D专家轨迹时，2D生成视频缺少动作与深度信息，难以转成足够稠密、稳定、可优化的控制学习信号
> - **mechanism_delta**: 用冻结视频扩散模型生成参考视频，并以 TimeSformer 视频嵌入距离 + 身体分割 IoU 取代对抗判别器，形成无数据模仿奖励
> - **evidence_signal**: 跨 4 种机器人与 AMP/GAIfO/DRAIL 对比，且奖励消融显示“视频相似度 + IoU + 正则”组合最有效
> - **reusable_ops**: [single-frame text-conditioned reference generation, embedding-plus-mask reward shaping]
> - **failure_modes**: [复杂形体上步态自然性不足, 参考视频质量差或奖励项缺失时容易抖动/停走]
> - **open_questions**: [如何扩展到长时操控而非行走, 如何降低对固定相机与闭源视频模型的依赖]

## Part I：问题与挑战

### 1. 这篇论文到底在解决什么问题？
它要解决的是一个很现实的 Embodied AI 痛点：

- **RL 能学出物理可行的运动**，但通常需要针对“任务 × 机体”手工设计奖励；
- **Imitation Learning 不用精调奖励**，但又依赖高质量 3D 专家演示；
- 对于 **非标准形体**（如不同 humanoid、四足甚至动物），这种 3D 数据往往很难采、很贵、甚至根本没有。

NIL 的核心目标就是：

> **不用任何人工收集的专家数据，只靠生成视频，让机器人学会可执行的3D运动技能。**

### 2. 真正的瓶颈是什么？
不是“生成视频不够好看”，而是：

1. **2D 到 3D 的监督缺口**  
   生成视频只有像素，没有关节角、速度、接触状态，也没有动作标签。

2. **视觉合理 ≠ 物理可执行**  
   视频扩散模型能生成“像在走”的视频，但未必满足平衡、接触、动力学约束。

3. **控制学习需要稠密反馈**  
   高维连续控制里，只给一个粗粒度“像不像”分数不够，策略会抖、会卡住、会学出投机动作。

### 3. 为什么现在值得做？
因为三个条件在最近同时成熟了：

- **视频扩散模型** 已经能生成跨形体、跨风格的动作视频；
- **视频编码器/ViT** 已能稳定提取时空语义；
- **物理模拟 + RL** 可以把视觉目标“投影”为物理可行的轨迹。

所以现在可以尝试一种新范式：

> **让生成模型负责“给目标”，让模拟器负责“保物理”，让 RL 负责“把两者接起来”。**

### 4. 输入/输出接口与边界条件
- **输入**：机器人初始帧、文本任务描述、机器人形体。
- **输出**：在物理模拟中执行任务的连续控制策略。
- **边界条件**：
  - 固定跟随机位；
  - 生成视频与模拟渲染需要视角一致；
  - 模拟器需能提供渲染与 body mask；
  - 实验主要限于 **locomotion**，不是通用操控。

---

## Part II：方法与洞察

### 方法主线
NIL 分两阶段：

#### 阶段1：先“生成参考动作视频”
给定：
- 机器人在模拟器中的初始渲染帧 `e0`
- 文本 prompt：`"The bj agent is si, camera follows."`

冻结的预训练视频扩散模型生成一段参考视频。  
这一步的意义不是生成可执行控制，而是生成一个**视觉目标轨迹**。

#### 阶段2：再“把视觉目标变成模仿奖励”
训练时，策略在模拟器中执行动作，系统渲染出当前 agent 视频，并与参考视频比较。奖励由三部分组成：

1. **视频相似度**  
   把生成视频和模拟视频都送入预训练 **TimeSformer**，比较 8 帧 clip 的嵌入距离。  
   作用：提供**全局时空动作语义**。

2. **图像级相似度（mask IoU）**  
   用 SAM2 分割生成视频中的主体，模拟视频的 mask 由 simulator 直接给出，然后算 IoU。  
   作用：提供**更局部、更几何化的对齐信号**。

3. **控制正则**  
   包括 torque、动作变化、关节速度、足接触/滑动、躯干稳定等。  
   作用：把“看起来像”收束成“真的能跑”。

最后，论文用**无判别器的 off-policy actor-critic RL** 来最大化这个组合奖励。实验里用的是 BRO，但作者强调这不是方法绑定点。

### 核心直觉

#### 直觉1：把“监督源”从 3D 轨迹换成 2D 视觉目标
以前 IL 的关键资产是 **expert trajectory**。  
NIL 说：未必要有 3D 专家轨迹，只要能得到一个**足够像任务目标的视觉轨迹**，就可以开始学。

这改变的是**监督分布来源**：
- 过去：来自人工采集的数据集；
- 现在：来自冻结生成模型的按需生成。

#### 直觉2：只用全局视频特征不够，必须补局部几何对齐
单独依赖视频嵌入距离，策略能抓到“像在走”的时序节奏，但细节容易抖、漂、偏。
加入 mask IoU 后，系统获得了更明确的：
- 身体轮廓对齐
- 前进位置变化
- 姿态粗几何约束

这改变的是**信息瓶颈**：
- 视频编码器给“动作语义”
- IoU 给“局部形状/位置”
- 正则给“物理可执行性”

三者叠加后，奖励才足够密。

#### 直觉3：物理模拟器其实是“物理过滤器”
视频扩散模型可能生成不完全物理合理的动作，但 NIL 并不要求生成视频本身完全可执行。  
它依赖模拟器和正则把视觉目标约束到可行动作空间里。

所以它的关键不是“生成视频必须物理正确”，而是：

> **生成视频只要在视觉上给出合适目标，模拟器就能把它筛成物理可行的版本。**

这也是为什么论文发现：**视觉 plausibility 比严格 physical correctness 更重要**。

### 为什么这套设计能工作？
可以把 NIL 看成三层分工：

- **扩散模型**：提供“想学什么动作”
- **视频/分割奖励**：提供“当前像不像”
- **模拟器+正则**：提供“能不能以物理方式做到”

这使得系统摆脱了：
- 专家动作采集
- 对抗判别器不稳定性
- 手工奖励工程的主体依赖性

### 战略性取舍

| 设计选择 | 解决的瓶颈 | 收益 | 代价/风险 |
|---|---|---|---|
| 单帧+文本生成参考视频 | 无专家数据可用 | 几乎零数据启动，跨形体泛化强 | 强依赖视频模型质量、prompt 和相机设置 |
| TimeSformer 嵌入距离 | 从2D视频中提取时序语义 | 比逐帧像素更稳健 | 粒度偏粗，单独使用会抖动 |
| 分割 IoU | 提供主体轮廓与位置约束 | 补足局部几何监督 | 对分割质量、遮挡、视角敏感 |
| 物理正则 | 抑制不稳定与投机动作 | 提升平滑与可执行性 | 可能牺牲部分风格自然度 |
| 去掉 adversarial discriminator | 降低 IL 训练不稳定 | 训练流程更简单 | 奖励权重与视觉度量设计更关键 |

### 方法层面的可迁移洞察
这篇论文最值得带走的不是某个具体 reward 系数，而是两个可迁移算子：

1. **生成式参考轨迹替代人工示范**
2. **“全局视频语义 + 局部分割几何 + 物理正则”的分层奖励设计**

---

## Part III：证据与局限

### 关键实验信号

#### 1. 与数据驱动模仿学习基线相比，NIL 在若干机器人上已具竞争力
最强的信号是：**NIL 不用 curated expert demos，却能与使用 25 条专家演示的 baselines 对抗。**

- **Unitree H1**：396.1，略高于 AMP 的 393.5，接近 expert 400
- **Talos**：352.8，明显高于 AMP 的 231.1
- **Unitree A1**：360.8，高于 AMP 的 286.9

这说明能力跳跃不只是“能动起来”，而是：
> **零数据模仿在某些 embodiment 上已经达到甚至超过数据驱动 IL。**

但也要注意：
- **Unitree G1** 上 AMP 更高（393.4 vs 356.9），说明 NIL 不是对所有形体都稳胜。

#### 2. 奖励组合确实是关键因果旋钮
H1 消融显示：

- **全部组件** 最稳；
- 去掉 **IoU** 后，motion-capture loss 从 46.4 恶化到 82.9；
- 只用 **video similarity** 时还能部分走起来，但明显抖动、会中途停；
- 只用 **IoU** 或只用 **regularization**，基本学不出持续向前步行。

这支持作者的核心论点：  
**单一视觉指标不足以支撑高维连续控制，需要“语义 + 几何 + 物理”三类信号并用。**

#### 3. 生成视频质量直接影响最终策略质量
扩散模型对比中：
- **Kling** 最好，NIL performance 396.1
- **SVD** 较弱，366.5

更有意思的是，作者比较 Kling v1.0 与 v1.6 时发现：
- 定量分数差距不一定很大；
- 但新版本生成的 gait 更自然，学到的步态也更协调。

这说明 NIL 的上限明显受制于**生成参考的视觉质量**，而且这种影响有时先体现在**动作自然性**上，而不一定马上反映在单一分数里。

### 1-2 个最关键指标
- **H1 env reward**：396.1（NIL） vs 393.5（AMP） vs 400（Expert）
- **Talos env reward**：352.8（NIL） vs 231.1（AMP）

### 局限性
- **Fails when**: 参考视频本身 gait 不稳定、相机/背景设定与模拟渲染不一致、或形体动力学特别复杂时，策略容易学出不自然步态；在 G1 和 Talos 上已能看到自然性不足的问题。
- **Assumes**: 固定跟随机位；生成视频与模拟视频可对齐；可获得 simulator mask 与 SAM2 分割；依赖强预训练视频模型、视频编码器和物理模拟器。最佳结果使用 Kling/Pika/Sora/Runway 等闭源视频服务，复现性受外部 API 和版本变化影响；论文也仅承诺未来开源代码与模型。
- **Not designed for**: 长时操控、多阶段任务规划、真实机器人在线部署、或需要精确 3D 接触/动作标注的任务；当前证据主要集中在 locomotion，而非通用 embodied skill set。

### 可复用组件
- **参考视频生成接口**：`初始帧 + 文本任务 -> 视频目标`
- **视频嵌入奖励**：用预训练视频编码器把“像不像”变成稠密 reward
- **mask 级辅助奖励**：用廉价几何对齐补足时空语义的粗粒度缺陷
- **时序对齐策略**：把低帧率视频奖励映射到高频控制步长

### 一句话总结
NIL 的真正贡献不是“又用了一个扩散模型”，而是提出了一条新的监督路径：

> **把生成视频当作可按需制造的视觉专家，再用视频语义奖励和物理模拟把它转换成可执行控制策略。**

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_NIL_No_data_Imitation_Learning_by_Leveraging_Pre_trained_Video_Diffusion_Models.pdf]]