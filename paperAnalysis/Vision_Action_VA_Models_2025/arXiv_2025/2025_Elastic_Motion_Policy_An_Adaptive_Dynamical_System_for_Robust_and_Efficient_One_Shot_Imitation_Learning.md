---
title: "Elastic Motion Policy: An Adaptive Dynamical System for Robust and Efficient One-Shot Imitation Learning"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/one-shot-imitation-learning
  - task/robot-manipulation
  - dynamical-systems
  - laplacian-editing
  - lyapunov-stability
  - dataset/LASA
  - opensource/no
core_operator: "把单次示教编码成稳定的 SE(3) LPV-DS，再依据物体关键位姿在 R3×SO(3) 上对 GMM 骨架做拉普拉斯弹性编辑，并用凸 Lyapunov 优化实时重估策略。"
primary_logic: |
  单次示教轨迹 + 相关物体的关键位姿/网格与在线跟踪结果 → 学习稳定 SE(3) LPV-DS，并在位置空间与四元数切空间对 GMM joints 做拉普拉斯弹性编辑，同时用凸/GMM-informed P-QLF 快速重估稳定参数 → 输出能随场景变化在线适配、且保持收敛与顺应性的末端位姿速度控制
claims:
  - "在 LASA 数据集上，GMM-informed convex P-QLF 相比原始 P-QLF 将全轨迹学习时间从 2.62s 降到 0.09s，违反 Lyapunov 条件的比例仍保持同量级（15.4% vs 14.9%） [evidence: comparison]"
  - "在真实机器人 OOD 场景中，EMP 相比 object-centric SE(3)-LPVDS 显著提高任务成功率：Book Placing 8/10 vs 4/10，Cube Pouring 9/10 vs 4/10，Pick-and-Place 7/10 vs 1/10 [evidence: comparison]"
  - "EMP 可在不收集新演示的前提下与调制式避障和 UVD 多步分解结合，完成避障书本放置和多步 pick-and-place [evidence: case-study]"
related_work_position:
  extends: "Elastic-DS (Li and Figueroa 2023)"
  competes_with: "SE(3)-LPVDS (Sun and Figueroa 2024); Interaction Warping (Biza et al. 2023)"
  complementary_to: "FoundationPose (Wen et al. 2024); Universal Visual Decomposer (Zhang et al. 2024)"
evidence_strength: moderate
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Elastic_Motion_Policy_An_Adaptive_Dynamical_System_for_Robust_and_Efficient_One_Shot_Imitation_Learning.pdf
category: Embodied_AI
---

# Elastic Motion Policy: An Adaptive Dynamical System for Robust and Efficient One-Shot Imitation Learning

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2503.08029), [Project](https://elastic-motion-policy.github.io/EMP/)
> - **Summary**: 这篇论文把单次示教从“复现绝对轨迹”改成“在稳定动力系统内部做受任务约束的弹性形变”，让机器人在目标物位置变化时仍能实时、自适应且可收敛地执行操作。
> - **Key Performance**: LASA 上 GMM-informed convex P-QLF 较原始方法约 **50×** 加速（2.62s → 0.09s）；真实机器人 OOD 场景下 **Book Placing 8/10 vs 4/10**、**Pick-and-Place 7/10 vs 1/10** 优于基线。

> [!info] **Agent Summary**
> - **task_path**: 单次示教末端轨迹 + 在线物体关键位姿/场景变化 -> 全姿态自适应末端速度策略 -> 稳定机器人执行
> - **bottleneck**: 传统 BC/DS 能拟合示教，但当对象配置变化时，中间运动几何约束会失真，导致“能到终点但过程无效/碰撞/不再满足任务”
> - **mechanism_delta**: 将示教分解为 GMM 局部几何骨架，在位置与姿态切空间中做受关键位姿约束的拉普拉斯弹性编辑，并用凸 P-QLF 快速重估稳定动力系统
> - **evidence_signal**: OOD 真实机器人三任务相对 SE(3)-LPVDS 明显提升，同时 LASA 上 Lyapunov 学习显著加速
> - **reusable_ops**: [keypose-conditioned-policy-morphing, gmm-informed-convex-lyapunov-fitting]
> - **failure_modes**: [object-tracking-or-keypose-errors, adapted-path-infeasible-under-reachability-or-joint-limits]
> - **open_questions**: [how-to-remove-dependence-on-external-object-tracking-and-closed-api-semantic-labeling, how-to-scale-to-multi-demo-multi-goal-contact-rich-tasks]

## Part I：问题与挑战

这篇论文要解决的，不只是“少样本 imitation learning”，而是更具体的机器人问题：

**只给一次示教，机器人能否在场景几何发生变化时，仍然稳定、顺应、可恢复地完成任务？**

### 1）真正的难点是什么？
作者认为，经典 BC 的根本问题不只是数据少，而是：

- **示教学到的是轨迹表面，而不是任务结构**
- 一旦物体位置、朝向或相对几何关系变了，原轨迹就可能失效
- 即使最终仍收敛到目标点，中间路径也可能：
  - 进入未知区域
  - 碰撞环境
  - 违反接近方向/倒入姿态/插入姿态等任务约束

所以，**真实瓶颈不是“终点是否到达”，而是“在新场景中，中间运动是否仍保持任务几何有效性”**。

### 2）为什么现在值得解决？
因为作者的目标场景是**人类中心的动态环境**：家庭、仓库、医院、工厂。这里有几个现实约束：

- 很难为每种新物体配置反复采集演示
- 机器人需要允许物理接触、扰动与顺应控制
- 部署时环境可能在线变化，而不是离线固定

因此，“多收数据”并不是足够好的答案；更实际的方向是：

- 用**很少示教**
- 保留**稳定性保证**
- 同时具备**在线适配能力**

### 3）输入 / 输出接口
**输入：**
- 单次示教得到的末端位姿轨迹
- 相关物体的 3D mesh
- 运行时物体位姿跟踪结果
- 从示教中提取的任务关键位姿（keypose）

**输出：**
- 末端线速度与角速度命令
- 经过被动阻抗控制器转成关节力矩/执行命令

### 4）边界条件
EMP 不是通用端到端视觉策略，它更像一个**任务约束驱动的稳定运动策略框架**。它更适用于：

- 目标明确的 reaching/manipulation 子任务
- 能定义吸引子/终止位姿的操作
- 物体可检测、可跟踪、可建立关键位姿约束的场景

不太直接覆盖：

- 原始像素到动作的端到端控制
- 高度复杂、循环式、强接触的非线性技能
- 无法获得物体几何与跟踪的场景

---

## Part II：方法与洞察

EMP 的核心不是“重新训练一个更大的 policy”，而是：

**先学一个稳定的运动骨架，再根据场景变化对这个骨架做受约束的弹性形变。**

### 方法主线

#### 1）先从单次示教学一个稳定的全姿态 DS policy
作者基于 **SE(3) LPV-DS**：

- 位置部分：LPV-DS
- 姿态部分：Quaternion-DS
- 二者组合成全末端位姿策略

这个基座的好处是：
- 自带 Lyapunov 风格的**收敛保证**
- 天然适合顺应、反应式控制
- 比黑盒深网更容易解释与约束

#### 2）把“任务信息”显式提取成 keypose
EMP 不直接把演示当成必须逐点复现的轨迹，而是从示教视频中抽取：

- 相关语义物体
- 物体 6D pose
- 末端相对该物体的最终关键位姿

文中 pipeline 是：

- GPT-4o：从示教关键帧识别相关物体语义
- Grounded SAM：分割物体
- FoundationPose：估计和跟踪物体位姿
- 记录末端相对物体的 keypose

于是运行时如果物体变了位置，**约束也会跟着变**。

#### 3）真正的创新：对 policy 内部结构做弹性形变，而不是整体刚体变换
作者沿用并扩展了 Elastic-DS 的想法：

- 用 GMM 分解示教轨迹，得到一串局部高斯“骨架”
- 在相邻高斯之间定义 joints
- 对这些 joints 做**Laplacian editing**
- 只固定少量任务相关的几何约束（如起点、终点/关键位姿）
- 其余中间结构“弹性地”跟着调整

这一步的含义很重要：

- **基线做法**：把整条 policy 刚体旋转/平移过去
- **EMP 做法**：让 policy 内部的局部几何关系在新约束下重排

因此它更能保留：
- 接近角度
- 进入路径
- 倒料时的弧线与手腕旋转
- 多步任务中的局部子结构

#### 4）把这个思想从位置扩展到姿态
Elastic-DS 原来主要在欧式空间里做。EMP 的扩展点是：

- 把四元数 GMM 映射到吸引子定义的切空间
- 在一个 3D 欧式表示里做同样的 Laplacian 编辑
- 再映回 quaternion 空间

这使 EMP 不只改位置路径，也能改**姿态轨迹**，因此能处理 full-pose manipulation。

#### 5）为了在线更新，再把 Lyapunov 学习变快
原始 LPV-DS 中某个稳定性矩阵 \(P\) 的学习是**非凸**的，在线更新太慢。作者改成：

- 用**凸优化**直接拟合满足 Lyapunov 下降趋势的 \(P\)
- 再进一步，用 GMM 区域均值代替所有点，形成 **GMM-informed convex P-QLF**

直观上就是：

- 少看每一个点
- 多看每一段局部模式的“代表点”
- 用较少计算换来实时更新能力

这一步是 EMP 能在线运行的关键工程旋钮之一。

#### 6）框架还能接别的模块
作者把 EMP 设计成可组合框架：

- **UVD** 做长时程任务分段
- 子任务之间用 one-hot 激活切换 DS
- **Obstacle modulation** 做实时避障
- 被动阻抗控制器负责物理执行层的顺应性

### 核心直觉

**what changed：**  
从“把演示当成一条固定轨迹去模仿”变成“把演示当成带局部几何关系的任务骨架去变形”。

**which bottleneck changed：**  
原先 policy 编码的是**绝对路径**，场景一变就失效；现在编码的是**相对任务结构 + 局部几何关系**，场景变化时只需更新关键约束和骨架形状。

**what capability changed：**  
机器人不再只是“在原场景复现得像”，而是能在**新对象配置、障碍出现、甚至多步切换**时，仍保持任务有效的中间运动，并保留收敛性。

**为什么这在因果上有效？**
1. **GMM 分解**把复杂动作拆成局部阶段结构。  
2. **Laplacian editing**保留邻域关系，所以新轨迹不会完全跑偏。  
3. **keypose 约束**把“什么必须保持”显式固定住。  
4. **凸 Lyapunov 更新**让更新后的场依然稳定，而且快到能在线用。  

换句话说，作者调的关键旋钮不是“模型容量”，而是**policy 表达的几何层级**。

### 战略权衡表

| 设计选择 | 解决的瓶颈 | 带来的能力 | 代价 / 风险 |
|---|---|---|---|
| GMM joints 的拉普拉斯弹性编辑 | 场景变化后中间轨迹失真 | 保留示教局部几何，同时满足新任务约束 | 强依赖 keypose 与物体位姿准确 |
| 姿态在切空间中编辑 | 原 Elastic-DS 主要是欧式空间，难处理旋转 | 扩展到 full-pose manipulation | 大角度旋转时单切空间近似可能失真 |
| Convex / GMM-informed P-QLF | 原稳定性拟合太慢，难在线更新 | 近实时更新，作者报告约 30Hz | GMM 近似会牺牲部分细节，violation 略升 |
| 单示教 DS policy | 数据收集昂贵、在线专家不可得 | 一次示教即可部署，且解释性强 | 表达力弱于更高容量非线性策略 |
| 接入预训练感知与分段器 | 任务参数和长时程结构难手工设计 | 支持语义关键物体与多步任务 | 继承感知误差、分段误差、闭源 API 依赖 |

---

## Part III：证据与局限

### 关键证据：能力跃迁到底在哪里？

真正的能力跃迁不是 ID 场景下“是否能复现”，而是：

**当对象配置变了以后，EMP 是否还能保持任务几何有效性。**

#### 信号 1：Lyapunov 学习显著提速，支撑在线更新
- **信号类型**：comparison
- 在 LASA 上：
  - 原始 P-QLF：2.62s
  - Convex P-QLF：0.24s
  - GMM-informed Convex P-QLF：0.09s
- 结论：作者确实把“稳定性拟合”从离线慢步骤，推进到更接近实时可用的程度。

这里最关键的是：**不是只变快，还基本没把稳定性约束破坏掉**。GMM-informed 版本 violation 稍高，但仍在同一量级。

#### 信号 2：OOD 真实机器人任务明显优于基线
- **信号类型**：comparison
- 基线是 object-centric SE(3)-LPVDS，即对已有策略做对象中心的全局变换
- OOD 结果：
  - Book Placing：**8/10 vs 4/10**
  - Cube Pouring：**9/10 vs 4/10**
  - Pick-and-Place：**7/10 vs 1/10**

这组结果支持了本文最重要的论点：

> 仅做全局变换不能保证中间任务约束仍成立；而 EMP 的弹性变形能更好保留“怎么接近、怎么倒、怎么放”的局部结构。

#### 信号 3：组合能力来自框架兼容性，而非重新训练
- **信号类型**：case-study
- 作者展示了两类附加能力：
  - 书本放置时加入 obstacle modulation 实现绕障
  - UVD 分段后完成多步 pick-and-place

这说明 EMP 更像一个**稳定动作骨架层**，可以挂接感知、分段、避障模块。

### 1-2 个最关键指标
- **在线稳定性拟合速度**：LASA 全轨迹设定下约 **2.62s → 0.09s**
- **OOD 任务成功率优势**：如 **Pick-and-Place 7/10 vs 1/10**

### 局限性

- **Fails when**: 物体跟踪或 keypose 提取出错时，几何约束会被错误注入；适配后的路径虽然满足任务空间约束，但仍可能超过机器人可达域或关节限制；多步任务中若 UVD 分段过早或抓取失败，后续吸引子会被放错位置。
- **Assumes**: 需要准确的物体 3D mesh 与 6D pose 跟踪；依赖 GPT-4o + Grounded SAM + FoundationPose 这一感知栈；执行侧依赖被动阻抗控制器与机器人标定；每个子任务最好能表达为单吸引子的 goal-oriented motion。
- **Not designed for**: 高度复杂的强非线性、循环式、长时接触动力学技能；直接从原始视觉端到端学习 policy；在没有物体几何或无法定义局部关键约束时的多目标、多演示泛化。

### 复现与可扩展性备注
- 论文有项目页，但正文未明确给出代码仓库，因此我将开源状态保守记为 **opensource/no**。
- 复现实验还依赖：
  - Franka Research 3
  - UMI gripper / 外部 RGBD / AprilTags
  - FoundationPose、Grounded SAM
  - GPT-4o 语义提示
- 这些依赖意味着：**方法思想可复用，但完整系统复现不是“只下代码就能跑”**。

### 可复用组件
1. **Keypose-conditioned policy morphing**：把任务条件显式化为约束，比 end-to-end 重训更轻量。  
2. **Orientation tangent-space editing**：适合把欧式几何编辑扩展到姿态控制。  
3. **GMM-informed convex Lyapunov fitting**：适合任何需要“稳定但要快”的 DS 在线更新场景。  
4. **DS composition + modulation interface**：适合多步任务切换与避障组合。  

### 一句话结论
EMP 的价值不在于把单示教 imitation 做得“更像”，而在于把它做成了一个**可在线重构、保留稳定性、并能显式注入任务几何约束**的运动策略层；它比纯 BC 更适合动态人机环境，但前提是你能提供可靠的对象感知与关键位姿。

## Local PDF reference

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Elastic_Motion_Policy_An_Adaptive_Dynamical_System_for_Robust_and_Efficient_One_Shot_Imitation_Learning.pdf]]