---
title: "PinchBot: Long-Horizon Deformable Manipulation with Guided Diffusion Policy"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/deformable-manipulation
  - diffusion
  - point-cloud-embedding
  - collision-projection
  - dataset/ShapeNet
  - opensource/partial
core_operator: "用预训练点云编码器表征当前/目标形状，以任务进度引导的目标条件扩散策略生成长时域捏塑动作，再把潜在碰撞动作投影到当前陶壁的安全边界。"
primary_logic: |
  多视角当前点云 + 目标点云 + 上一动作 → 预训练点云嵌入并注入目标/进度条件，扩散去噪生成动作块 → 基于当前碗口几何做碰撞约束投影并闭环重规划 → 逐步把初始泥柱塑形成目标陶碗并自主终止
claims:
  - "单个目标条件策略可在真实机器人上生成 8/10/12 cm 三类碗形目标；其中 PointBERT + binary 变体在 10 cm 目标上达到 CD 7.1 ± 0.2 mm、EMD 6.3 ± 0.5 mm、直径 MSE 0.05 ± 0.02 [evidence: case-study]"
  - "在 8/10/12 cm 三个目标上，PointBERT 变体的 CD/EMD/MSE 均优于对应的 DP3 PointNet 变体；其 binary 版本平均动作数为 28.4 ± 2.6，低于 DP3 的 51.3 ± 18.5 [evidence: comparison]"
  - "在 10 cm 目标的消融中，去掉预训练或碰撞投影都会升高误差，而把扩散训练替换为回归会把 PointBERT 的 CD 从 7.1 mm 恶化到 18.9 mm，说明多模态动作建模是关键 [evidence: ablation]"
related_work_position:
  extends: "SculptDiff (Bartsch et al. 2024)"
  competes_with: "SculptDiff (Bartsch et al. 2024); Ropotter (Yoo et al. 2024)"
  complementary_to: "Latent Space Reinforcement Learning for Diffusion Policy (Wagenmaker et al. 2025); RoboCraft (Shi et al. 2024)"
evidence_strength: moderate
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_PinchBot_Long_Horizon_Deformable_Manipulation_with_Guided_Diffusion_Policy.pdf
category: Embodied_AI
---

# PinchBot: Long-Horizon Deformable Manipulation with Guided Diffusion Policy

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2507.17846), [Project + Dataset](https://sites.google.com/andrew.cmu.edu/pinchbot/home)
> - **Summary**: 论文把预训练点云表征、目标条件扩散策略、任务进度预测和几何安全投影结合起来，让机器人能用单一策略完成长时域的 pinch pottery 目标塑形。
> - **Key Performance**: 最佳报告结果出现在 10 cm 目标上：CD 7.1 mm、EMD 6.3 mm、直径 MSE 0.05；对应 PointBERT binary 变体平均 28.4 步完成塑形。

> [!info] **Agent Summary**
> - **task_path**: 多视角当前点云 + 目标点云 + 上一动作 -> 8D pinch 动作块与终止/进度信号
> - **bottleneck**: 小样本下的长时域可变形塑形同时受多模态轨迹、强遮挡、阶段不可观测和单步误差可致灾难性破坏约束
> - **mechanism_delta**: 将目标条件、显式任务进度、预训练点云 latent 和碰撞动作投影耦合进扩散策略，缩窄动作分布并过滤危险位姿
> - **evidence_signal**: 真实机器人 8/10/12 cm 三目标比较与 10 cm 消融共同表明 PointBERT、预训练、碰撞投影和扩散训练均有贡献
> - **reusable_ops**: [pretrained-point-cloud-encoder, task-progress-guidance, collision-action-projection]
> - **failure_modes**: [large-diameter-subgoal-confusion, over-pinching-from-poor-phase-estimation]
> - **open_questions**: [how-to-generalize-beyond-20-demos, can-rl-or-dynamics-modeling-expand-goal-coverage]

## Part I：问题与挑战

这篇论文解决的是一个比常见 clay shaping 更难的设定：**不给工具、不用转盘、只靠夹爪的 pinch 动作，把一段泥柱逐步捏成目标陶碗**。  
输入是当前 clay 的 3D 点云和目标 3D 点云，输出是一串 8D 动作：
\[
[x, y, z, R_x, R_y, R_z, d_{ee}, \gamma]
\]
其中 \(\gamma\) 用来表示终止或任务进度。

### 真正的瓶颈是什么

不是“能不能学一个 pinch 动作”，而是以下四个瓶颈叠加：

1. **长时域**  
   以往不少 clay benchmark 只需不到 10 步；这里一条演示轨迹通常需要 **21–31 步**。  
   长时域意味着早期小误差会在后续被放大。

2. **高多模态**  
   到达同一个 bowl 目标并没有唯一正确顺序。  
   同样的最终形状，可能对应多种合法捏塑序列，这天然适合扩散式动作建模，但不适合简单回归。

3. **状态部分不可观测**  
   陶碗内部尤其在早期开口很小时会被遮挡，所以论文必须用 **4 个固定相机 + 1 个腕部相机** 融合点云，才能看到内外壁。

4. **单步错误代价极高**  
   pinch 位置或姿态稍微错一点，就可能在陶壁上打洞，进入几乎不可恢复的状态。

### 为什么现在值得做

作者的核心判断是：**现有 deformable shaping benchmark 对长时域区分度不够**。  
很多方法在短轨迹任务上都能“看起来有效”，但一旦进入 20+ 步、无遮挡不充分、动作顺序不唯一的场景，方法差异才会被真正放大。  
同时，扩散策略与 3D 点云预训练这两类工具，刚好给了这个问题一个可行的切入点。

### 输入/输出接口与边界条件

- **输入**：当前多视角融合点云、目标点云、上一动作
- **输出**：8D pinch 动作块；训练时预测 16 步，测试时执行 4 步后重规划
- **任务边界**：
  - 只做 **bowl 类简单 pottery**
  - 初始 clay 为不同尺寸的圆柱，但总体体积近似固定
  - 目标直径主要是 **8 / 10 / 12 cm**
  - 只考虑 pinch end-effector，不含工具操作或 wheel-based pottery

---

## Part II：方法与洞察

PinchBot 的做法可以概括成一句话：

> **先把“当前形状/目标形状”编码得更稳，再把“我现在做到哪一步”显式告诉策略，最后用一个几何安全层拦住最危险的动作。**

### 方法骨架

#### 1. 点云表征：先解决小数据下的 shape latent 不稳
作者比较了两种点云编码器：

- **PointBERT**
- **DP3 PointNet**

两者都先在 **ShapeNet reconstruction** 上做预训练，再接一个 3 层 MLP 投到 512 维 latent，随后联合扩散策略微调。

这一步的作用不是单纯“换个 backbone”，而是让少量真实演示下的 shape embedding 更有结构，尤其要能区分：

- 当前 pot 处于哪个阶段
- 距离目标还有多远
- 什么时候该停止

#### 2. 目标条件扩散策略：把多模态动作分布留给 diffusion
策略输入包括：

- 当前 state latent
- goal latent
- previous action

然后做动作块去噪生成。  
这里的关键不是生成单步动作，而是**生成一个局部未来动作片段**，再以 receding horizon 方式闭环执行。

#### 3. 任务引导：显式告诉模型“现在做到哪了”
论文探索了三种引导形式：

- **Binary prediction**：\(\gamma\) 仅表示是否结束
- **Continuous task progress guidance**：把 \(\gamma\) 改成 \([-1, 1]\) 连续进度，表示轨迹百分比
- **Sub-goal conditioning**：额外输入中间子目标 latent

作者主推的是第二种：  
把“任务阶段”从隐变量变成显变量，让扩散策略知道当前动作属于开口、拉壁还是修边阶段。

#### 4. 碰撞动作投影：给 policy 外挂一个几何 safety layer
系统从当前点云在 \(x,y\) 平面拟合 bowl 直径边界。  
如果策略给出的 pinch 位置落到已有陶壁内部，就把该动作的 \(x,y\) 投影到边界上，其他维度如：

- \(z\)
- 旋转
- 指尖距离

保持不变。

这相当于把最容易打穿陶壁的一类错误，直接从执行空间里裁掉。

### 核心直觉

过去这类任务难，不只是因为动作长，而是因为**同样的局部几何，在不同阶段应该采取不同动作**。

- 如果只看当前点云，模型可能不知道自己是在“早期开口”还是“后期修边”
- 一旦阶段信息缺失，同一状态会对应很多可能动作，动作分布就变得非常宽
- 在小数据条件下，这种宽分布会进一步导致“重复捏已经平滑的区域”“不知道何时停止”“切换 shaping 区域太慢”

PinchBot 实际上引入了三个因果旋钮：

1. **预训练点云 encoder**  
   改变了形状表示质量 → latent 更按目标/进度组织 → 更容易判断阶段与终止

2. **连续任务进度信号**  
   改变了 state-to-action 的条件信息 → 原本高度多模态的动作分布被 phase 信息收窄 → 区域切换和停止更稳定

3. **碰撞投影**  
   改变了可执行动作空间 → 直接去掉一批会造成不可恢复失败的动作 → 长时域 rollout 更稳

换句话说，这篇论文不是靠更复杂规划器取胜，而是靠**“更有序的表示 + 更明确的阶段 + 更硬的安全约束”**。

### 策略性取舍

| 设计选择 | 改变了什么 | 带来的能力 | 代价/风险 |
|---|---|---|---|
| ShapeNet 预训练点云编码器 | 改善小数据下 latent 结构 | 更好区分目标直径、轨迹阶段、停止时机 | 依赖外部 3D 预训练数据，存在域差 |
| Continuous progress guidance | 显式提供 phase 信息 | 减少过度 pinching，改善区域切换 | 对强 encoder（PointBERT）定量收益有限 |
| Sub-goal conditioning | 给中间状态引导 | 试图缓解长时域规划 | 容易把大碗目标与小碗中间态混淆 |
| Collision projection | 加入硬安全约束 | 减少刺穿陶壁的灾难性错误 | 只修正 \(x,y\)，无法纠正 \(z\)/旋转/夹距错误 |

---

## Part III：证据与局限

### 关键信号

- **[Comparison] 单一策略确实具备多目标适配能力**  
  同一个 goal-conditioned policy 能覆盖 **8/10/12 cm** 三种 bowl 目标，而不是每个目标单训一个 policy。  
  最好的报告结果出现在 **10 cm**：CD **7.1 mm**、EMD **6.3 mm**、直径 MSE **0.05**。

- **[Comparison] 点云 encoder 的选择非常关键**  
  PointBERT 版本在三种目标上的 CD / EMD / MSE 都优于对应 DP3 PointNet 版本。  
  更重要的是，PointBERT 的 binary 版本平均只需 **28.4** 步，而 DP3 对应版本要 **51.3** 步，说明更好的 latent 直接改善了“做到哪一步了”的判断。

- **[Comparison] task-progress guidance 的收益是“有条件成立”的**  
  它对 **DP3 PointNet** 的帮助更明显；  
  对 **PointBERT**，定量提升不大，很多结果落在方差内，但定性上更能按目标直径调整 \(R_x\)，也就是更好地控制碗壁倾角。  
  这说明：**phase signal 的价值，受表征质量约束。**

- **[Comparison] sub-goal guidance 不是这篇里的赢家**  
  它在 12 cm 大目标上明显变差。作者的解释很合理：大碗的中间态本身就像小碗终态，因此只用 sub-goal 条件时会出现目标混淆。

- **[Ablation] 因果证据最强的是 10 cm 消融**  
  去掉预训练、去掉 collision projection、或把扩散改成普通回归，都会显著变差。  
  尤其“扩散→回归”后，PointBERT 的 CD 从 **7.1 mm** 变成 **18.9 mm**，DP3 甚至到 **20.7 mm**。  
  这基本直接支持了论文的中心论点：**这个任务的动作分布本来就是多模态的，不能简单回归一个均值动作。**

- **[Analysis] latent 可视化与行为差异一致**  
  T-SNE 显示 PointBERT latent 在轨迹进度和目标直径上分离更清楚；  
  这和它更少冗余动作、更好终止判断的现象一致。

### 能力跃迁到底体现在哪

这篇论文真正的进步，不是提出一个通用 deformable manipulation 解法，而是把问题从：

- 短轨迹、低区分度 clay shaping

推进到：

- **真实机器人、20+ 步、强遮挡、高多模态、错误不可逆** 的 pinch pottery

并且证明了：  
**只要表征、阶段信号和安全约束配对得当，一个单策略就能做出可用的跨目标适配。**

### 局限性

- **Fails when**: 大直径目标在 sub-goal 条件下容易和小直径中间态混淆；当进度估计不稳时，策略会在已平滑区域重复 pinching，或过晚切换 shaping 区域。
- **Assumes**: 已知目标点云；5 相机标定、颜色阈值分割与 ICP 对齐可稳定工作；Franka + 定制软性非对称手指可用；只有 20 条真实演示，但依赖 z 轴旋转增强扩到 3600 条；ShapeNet 预训练对真实 clay latent 有帮助。
- **Not designed for**: 任意拓扑 pottery（如把手、细颈、复杂空腔）、工具辅助塑形、wheel-based pottery、跨材料零样本泛化、无稠密 3D 感知的场景。

### 复现与可扩展性备注

- 复现依赖不低：需要 **4 个固定 RealSense + 1 个腕部相机**、定制软夹爪、机器人标定与 ICP 对齐流程。
- 安全层只修正 **\(x,y\)**，对 **\(z\)、旋转、夹距** 错误无能为力。
- 项目页提供视频和演示数据，但文中未明确给出完整代码仓库，因此更接近 **partial release** 而非完整开源。

### 可复用组件

- **任务进度 token**：适合所有“同一局部观测在不同阶段应有不同行为”的长时域操作任务
- **碰撞动作投影**：可作为 policy 外挂 safety wrapper，用于降低不可恢复错误
- **预训练点云编码器 + 小样本 imitation**：适用于真实机器人 3D 观测稀缺、演示少的 deformable manipulation 任务

## Local PDF reference

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_PinchBot_Long_Horizon_Deformable_Manipulation_with_Guided_Diffusion_Policy.pdf]]