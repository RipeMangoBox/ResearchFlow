---
title: "Generalizable Humanoid Manipulation with Improved 3D Diffusion Policies"
venue: IROS
year: 2025
tags:
  - Embodied_AI
  - task/robot-manipulation
  - diffusion
  - egocentric-3d
  - teleoperation
  - repr/point-cloud
  - repr/joint-position
  - opensource/no
core_operator: "将DP3改造成以相机坐标系点云为输入的自中心3D扩散策略，并结合大点云输入、金字塔卷积编码器与长时域预测，提升人形机器人从噪声遥操作数据中学到可泛化操作的能力。"
primary_logic: |
  单场景AVP遥操作示教 + 头载LiDAR点云 + 关节本体状态
  → 在相机坐标系构建自中心3D表示，采用体素+均匀采样扩大量测点，使用金字塔卷积编码器和更长扩散预测时域学习目标关节序列
  → 在无需相机重标定与点云分割的情况下，于未见物体、未见视角和未见场景中实时输出25-DoF人形上半身操作动作
claims:
  - "在四个Pick&Place设置上，未修改的DP3在真实人形机器人评测中为0/0成功/尝试，而iDP3达到75/139，表明其改造后才具备从噪声人类示教学习稳定抓取的能力 [evidence: comparison]"
  - "预测时域是关键因果旋钮：当horizon从16缩短到4时，系统退化为0/0成功/尝试，而horizon=8也仅有33/88，表明长时域预测可显著缓解人类示教抖动与传感噪声 [evidence: ablation]"
  - "仅用单场景数据训练后，iDP3在未见物体、未见视角和未见场景的Pick&Place与Pour测试中均达到9/10成功，而强图像基线DP+finetuned R3M仅有0-3/10，显示其零样本场景泛化优势 [evidence: comparison]"
related_work_position:
  extends: "3D Diffusion Policy (DP3)"
  competes_with: "Diffusion Policy (Chi et al. 2023); 3D Diffusion Policy (DP3)"
  complementary_to: "EquiBot (Yang et al. 2024); Equivariant Diffusion Policy (Wang et al. 2024)"
evidence_strength: moderate
pdf_ref: paperPDFs/Vision_Action_VA_Models_2024/IROS_2025/2025_Generalizable_Humanoid_Manipulation_with_Improved_3D_Diffusion_Policies.pdf
category: Embodied_AI
---

# Generalizable Humanoid Manipulation with Improved 3D Diffusion Policies

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2410.10803), [Project](https://humanoid-manipulation.github.io)
> - **Summary**: 论文把原本依赖固定外参和点云分割的DP3改造成适合人形机器人头载LiDAR的自中心3D扩散策略，并结合上半身遥操作系统，使全尺寸GR1仅用单场景示教就能零样本泛化到新物体、新视角与新场景。
> - **Key Performance**: Pick&Place四设置累计 **75/139** 成功/尝试；单场景训练后，在未见 object/view/scene 的 Pick&Place 与 Pour 上 iDP3 均为 **9/10**，而强图像基线仅 **0-3/10**。

> [!info] **Agent Summary**
> - **task_path**: 单场景头载LiDAR点云 + 关节状态（模仿学习） -> 25-DoF人形上半身目标关节动作序列
> - **bottleneck**: 现有3D策略默认固定相机外参与干净分割点云，而人形机器人是可动视角 + 噪声深度 + 高成本多场景采集，导致策略很难从少量单场景人类示教中学到跨场景操作
> - **mechanism_delta**: 把DP3从世界坐标点云改成相机坐标自中心3D表示，并用更大点云输入、金字塔卷积编码器和更长预测时域一起提高鲁棒性
> - **evidence_signal**: 2000+次真实机器人rollout中，iDP3在未见物体/视角/场景上稳定达到9/10，明显优于图像基线的0-3/10
> - **reusable_ops**: [camera-frame-point-cloud, long-horizon-diffusion]
> - **failure_modes**: [fine-grained-contact-manipulation, severe-depth-noise]
> - **open_questions**: [如何与全身平衡和移动操作结合, 是否能用预训练3D表示替代从零开始学习]

## Part I：问题与挑战

这篇论文要解决的，不是“人形机器人能不能在实验桌上学会一个动作”，而是更难的版本：

**能否只在一个训练场景采少量人类示教，就让全尺寸人形机器人在新厨房、新办公室、新会议室中零样本完成操作？**

### 1) 真正瓶颈是什么？

真正瓶颈有三层，而且是连在一起的：

1. **表示瓶颈**  
   现有DP3一类3D策略默认使用**世界坐标系点云**，通常还依赖**精确相机标定**和**目标点云分割**。  
   但人形机器人头部会动、腰会动、视角会变，这个假设天然不成立。

2. **数据瓶颈**  
   人形机器人在多场景采集真实数据非常贵，难以像机械臂那样轻松扩展到很多环境。  
   所以如果方法必须依赖“20个场景的数据”才泛化，它在现实里就不够实用。

3. **学习瓶颈**  
   人类遥操作本身有抖动，深度点云也有噪声，短时域控制很容易学成“抖、停、犹豫”的策略。  
   也就是说，问题不只是“看不准”，还是“学不稳”。

### 2) 输入/输出接口

- **输入**：头载LiDAR点云、RGB图像、机器人关节状态
- **输出**：25-DoF上半身目标关节位置
- **训练范式**：模仿学习 / diffusion policy
- **部署约束**：只用机载算力，实时推理约 15Hz

### 3) 边界条件

这不是一个“全身自主人形”论文，边界画得很清楚：

- 下肢不参与控制，机器人固定在**可移动、可升降小车**上
- 主要验证上半身 manipulation，而不是 loco-manipulation
- 不依赖相机重标定与人工点云分割
- 任务是日常操作：**Pick&Place、Pour、Wipe**

### 4) 为什么现在值得解决？

因为人形硬件、遥操作链路、机载算力都已经到了可以做真实部署的阶段，但**泛化能力**仍然是把系统从“实验室演示”推到“真实可用”的最后缺口。  
这篇论文的意义在于：它把“单场景学到跨场景可用”这件事，第一次在**全尺寸人形机器人**上做到了比较可信的真实验证。

---

## Part II：方法与洞察

这篇工作不是只换一个网络，而是一个**系统级闭环**：

1. **平台层**：GR1 + 头载LiDAR + 高度可调车体  
2. **数据层**：AVP whole-upper-body teleoperation  
3. **策略层**：把DP3改成适合人形机器人的 iDP3  
4. **部署层**：机载实时推理，直接放到新场景

### 方法拆解

#### A. 平台：先把“能稳定操作”这个物理前提补上

- 使用 Fourier GR1，人形上半身共 25 DoF
- 头部挂载 RealSense L515 LiDAR，获取 egocentric 3D 观测
- 下肢关闭，放在**可移动且可升降**的小车上

这一步的作用很务实：  
作者没有强行追求“真正双腿自主平衡”，而是先用小车解决稳定性和桌面高度变化问题，从而把研究焦点放在**泛化操作**上。

#### B. 数据：whole-upper-body teleoperation

作者用 Apple Vision Pro 获取人类头/手/腕姿态：

- 手臂：用 Relaxed IK 追踪 human wrist
- 腰和头：由 human head rotation 映射
- 实时把机器人视角回传给操作者，做沉浸式遥操作

一个很重要的工程判断是：

- **动作空间直接用 joint positions**
- 而不是 end-effector pose

原因很简单：真实世界里末端位姿估计噪声更大，直接学关节目标更稳。

#### C. iDP3：四个关键改动

1. **世界坐标点云 → 相机坐标点云（egocentric 3D）**  
   不再依赖固定外参，也不需要点云分割。

2. **扩大点云输入规模**  
   在没有分割的情况下，用更多点把整个场景都覆盖进来，减少目标信息被背景吞掉的问题。

3. **MLP encoder → Pyramid Conv encoder**  
   作者发现从人类示教学习时，卷积编码器比纯MLP更容易学出平滑行为；再叠加多尺度金字塔特征，精度更高。

4. **更长 prediction horizon**  
   通过更长时域预测抵消人类示教抖动和传感噪声，减少短视的抖动控制。

此外，作者还把 DP3 的 FPS 换成了 **voxel + uniform sampling**，以更快地覆盖3D空间。

### 核心直觉

**核心变化不是“把DP3做得更大”，而是把它从“依赖固定世界坐标的桌面机械臂方法”，改成“适配可动相机的人形机器人方法”。**

可以用一条因果链看：

- **世界坐标点云 → 相机坐标点云**  
  改变了对**外参标定**的依赖  
  → 去掉了“相机必须固定”的约束  
  → 策略能直接迁移到新视角、新场景

- **小规模稀疏点云 → 大规模场景点云**  
  改变了“必须先做目标分割”的信息瓶颈  
  → 模型直接看到目标与背景、桌面、容器之间的几何关系  
  → 对 clutter、桌面变化、物体变化更鲁棒

- **短时域预测 → 长时域预测**  
  改变了“每一步都对瞬时噪声过敏”的控制方式  
  → 策略更像在跟踪一段动作趋势，而不是对每一帧做抖动反应  
  → 从 noisy human demos 学出更稳定行为

- **MLP → Conv+Pyramid**  
  改变了模型的归纳偏置  
  → 从“平坦映射”变成“局部到全局的多尺度特征整合”  
  → 动作更平滑、抓取更准

换句话说，这篇论文真正拧动的 causal knob 是：

> **把“泛化”从依赖更多场景数据，改成依赖更合适的3D自中心表示和更稳的时域学习。**

### 策略性取舍

| 设计选择 | 解决的瓶颈 | 带来的能力 | 代价 / 取舍 |
|---|---|---|---|
| 相机坐标系3D表示 | 去掉固定外参与分割依赖 | 新视角、新场景可直接部署 | 背景杂点更多，输入更脏 |
| 扩大点云输入 | 无分割时目标信息稀释 | 对复杂场景更稳 | 计算与带宽压力更大 |
| Conv + Pyramid encoder | 人类示教噪声导致控制不平滑 | 动作更顺、精度更高 | 结构更复杂 |
| 长预测时域 | 短时域对抖动敏感 | 学习更稳、少抖动 | 太长会变钝，32步反而退化 |
| 车体升降 + 固定下肢 | 桌高变化与平衡难题 | 真机部署更稳、更安全 | 不是完整全身移动操作 |

---

## Part III：证据与局限

### 关键证据看什么？

这篇论文最有说服力的，不是“seen setting 上所有指标都第一”，而是：

> **它把单场景训练得到的策略，真正带到了未见物体、未见视角、未见场景里。**

### 1) 比较信号：iDP3让DP3第一次在这类人形设定里“能学起来”

在四个 Pick&Place 设置上：

- **原始 DP3：0/0**
- **iDP3：75/139**

这说明原版 DP3 在该人形设定下几乎不可用，而 iDP3 通过表示和时域改造后，至少把系统拉到了“能稳定尝试、能成功抓取”的区间。

### 2) 反直觉信号：in-distribution 并不是 iDP3 的最大优势

作者很诚实地展示了：

- 在小数据、训练场景内的 Pick&Place 上，**DP + finetuned R3M** 是非常强的基线，达到 **99/147**
- iDP3 是 **75/139**

这意味着论文的贡献**不是**“在训练场景里纯精度碾压所有2D方法”，而是：

- 2D预训练方法在 seen setting 里可能更强
- 但它们更容易过拟合到具体物体、具体外观、具体背景

### 3) 真正能力跃迁：OOD 泛化

在 **新物体 / 新视角 / 新场景** 上，针对 Pick&Place 与 Pour：

- **iDP3：9/10**
- **DP + finetuned R3M：0-3/10**

这才是论文最关键的 “So what”：

**能力跳跃不在训练集内，而在真实部署条件下。**

### 4) 消融信号：哪些改动真的有效？

最关键的是 prediction horizon：

- horizon = 4：**0/0**
- horizon = 8：**33/88**
- horizon = 16：**75/139**
- horizon = 32：下降到 **55/130**

说明长时域预测不是装饰，而是处理**人类示教抖动 + 深度噪声**的关键旋钮。  
另外：

- 点云数量从 1024 → 4096 有明显提升
- Conv+Pyramid encoder 优于更简单的编码器变体

### 5) 证据强度怎么判断？

我会给 **moderate**，不是 strong：

- 优点：**2253 次真实机器人 rollout**，而且有比较完整的消融
- 但限制：主要还是**单平台、单实验室任务集、少数日常任务**
- 同时没有公开代码/数据，复现实证闭环还不够强

### 局限性

- **Fails when**: 需要极高接触精度或细粒度手指操作时（如拧螺丝）、深度点云严重失真时、或任务需要下肢平衡与移动协同时，当前系统容易失效或未被验证。
- **Assumes**: 依赖商用硬件链路（Fourier GR1 + Inspire Hands + Apple Vision Pro + RealSense L515）、高度可调车体、上半身25-DoF控制、约0.5秒遥操作延迟，以及足够的人类示教；论文未给出明确开源代码。
- **Not designed for**: 真正的全身 loco-manipulation、触觉/力控制主导的精细装配，以及大规模自动化数据采集。

### 可复用组件

这篇工作里最值得迁移的，不一定是整套系统，而是下面几个模块化操作：

- **自中心3D表示**：适合所有相机位姿会变化的机器人
- **长时域扩散控制**：适合 noisy demonstrations
- **Conv+Pyramid 点云编码**：适合从人类示教中学更平滑的控制
- **voxel + uniform point sampling**：对实时3D控制比较实用
- **whole-upper-body teleop 映射**：对人形上半身数据采集有参考价值

![[paperPDFs/Vision_Action_VA_Models_2024/IROS_2025/2025_Generalizable_Humanoid_Manipulation_with_Improved_3D_Diffusion_Policies.pdf]]