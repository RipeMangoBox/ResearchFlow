---
title: "PRISM-DP: Spatial Pose-based Observations for Diffusion-Policies via Segmentation, Mesh Generation, and Pose Tracking"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/robot-manipulation
  - diffusion
  - dataset/Robosuite
  - dataset/Robomimic
  - dataset/MimicGen
  - opensource/no
core_operator: 通过“分割→自动生成物体mesh→6D位姿估计/跟踪”把RGB-D演示转换为低维对象位姿观测，再训练紧凑的扩散策略。
primary_logic: |
  RGB-D序列、机器人状态与相机内参 → 在目标物首次可见帧做分割并生成mesh → 用生成mesh初始化并持续跟踪各物体6D位姿 → 将位姿序列与机器人状态拼接为观测 → 条件化扩散策略输出未来动作序列
claims:
  - "Claim 1: 在5个Robosuite任务上，PRISM-DP在相同参数规模下始终优于图像条件扩散策略，并在Stack、Square、Mug等较难任务上继续优于8-11×更大的图像模型 [evidence: comparison]"
  - "Claim 2: 在两个真实任务上，小容量图像条件扩散策略成功率均为0，而PRISM-DP-T在Block Stacking和Drawer Interaction上分别达到0.93和0.87 [evidence: comparison]"
  - "Claim 3: 使用生成mesh的pose策略与使用ground-truth mesh的策略性能相近；仿真分析中生成mesh的FID为0.1347、平均位置误差为0.0006，说明自动mesh足以支撑下游pose tracking [evidence: analysis]"
related_work_position:
  extends: "FoundationPose (Wen et al. 2024)"
  competes_with: "SPOT (Hsu et al. 2024); Diffusion Policy (Chi et al. 2023)"
  complementary_to: "N/A"
evidence_strength: moderate
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_PRISM_DP_Spatial_Pose_based_Observations_for_Diffusion_Policies_via_Segmentation_Mesh_Generation_and_Pose_Tracking.pdf
category: Embodied_AI
---

# PRISM-DP: Spatial Pose-based Observations for Diffusion-Policies via Segmentation, Mesh Generation, and Pose Tracking

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2504.20359)
> - **Summary**: 论文把目标物分割、单图mesh生成和6D位姿跟踪串成一个前端，让扩散策略直接看“机器人状态+物体位姿”而不是原始RGB，从而在真实操作中用更小模型取得更高成功率。
> - **Key Performance**: 真实 Drawer Interaction 上，PRISM-DP-T 成功率 **0.87**，优于大图像模型 **0.57** 和同规模图像模型 **0.00**；真实任务训练耗时约 **6-10 s/epoch**，远低于图像策略的 **88-132 s/epoch**。

> [!info] **Agent Summary**
> - **task_path**: RGB-D演示 + 机器人状态 + 相机内参 -> 任务相关物体6D位姿历史 -> 未来动作序列
> - **bottleneck**: 开放集真实场景里，pose-based policy 依赖的对象6D位姿很难在无标记、无手工mesh的条件下稳定获得
> - **mechanism_delta**: 用“分割→mesh生成→FoundationPose初始化与跟踪”替代手工建模和大视觉编码器，把高维视觉输入压成低维对象中心状态
> - **evidence_signal**: 5个仿真任务和2个真实任务中，PRISM-DP同规模显著优于图像条件扩散策略，且生成mesh版本基本追平GT mesh版本
> - **reusable_ops**: [首次可见帧点提示, mask-to-mesh自举, 位姿历史条件化策略]
> - **failure_modes**: [多物体或遮挡场景下pose drift累积, 非刚体/离视野对象不适用]
> - **open_questions**: [能否去掉人工首帧提示并自动发现目标物, 能否扩展到非刚体和长时遮挡跟踪]

## Part I：问题与挑战

这篇论文真正解决的，不是“如何再发明一个新的扩散去噪器”，而是**如何在真实开放集环境里，以低人工成本获得足够好的对象中心观测**。

### 真实问题是什么
现有 diffusion policy 常直接以 RGB 图像序列为条件输入。这样做的优点是接口通用，但缺点也很明显：

- 图像维度高，含有大量任务无关信息：背景、光照、纹理、遮挡、无关物体；
- 策略网络被迫同时学习“找物体、恢复3D几何、做时序关联、再输出动作”；
- 结果是模型更大、训练更慢、现实部署更脆弱。

而很多 manipulation 任务真正决定动作的，往往只是：

- 机器人自身状态；
- 关键物体的相对位置与朝向；
- 这些几何关系随时间的变化。

所以，**pose-based observation 本身比 raw RGB 更接近控制所需状态**。

### 真瓶颈在哪里
难点不在“pose 对控制有没有用”，而在“现实里怎么拿到 pose”。

传统做法通常依赖：
- AprilTag / Motion Capture 这类外部跟踪设施；
- 或预先手工扫描 / 重建目标物 mesh，再交给 6D pose tracker。

这使得 pose-based policy 虽然理论上更紧凑，却很难真正扩展到：
- 新物体频繁变化的 open-set 场景；
- 低人工准备成本的现实机器人系统。

### 为什么现在值得做
作者的判断是：**现在三类基础模型恰好补齐了这条链路**：

- 分割模型：把目标物从场景里扣出来；
- 单图 3D mesh 生成模型：把“看见物体”变成“有一个可跟踪的几何模板”；
- 统一 pose estimation/tracking 模型：把 mesh 变成连续位姿序列。

也就是说，过去必须人工完成的 mesh 准备步骤，开始可以被自动化替代。

### 输入 / 输出接口
- **输入**：RGB-D 视频序列、机器人状态、相机内参、每个目标物首次清晰可见时的一次用户点提示
- **中间表示**：每个任务相关物体的 6D 位姿时间序列
- **输出**：receding-horizon 动作块，由 diffusion policy 预测

### 边界条件
这篇论文的“开放集”含义是**不需要预扫描 mesh**，而不是完全自动目标发现。它仍然假设：

- 任务相关物体已知；
- 用户能在物体首次出现时给一次提示；
- 物体一旦出现后基本持续在视野中；
- 场景以刚体对象为主；
- 传感器是 RGB-D，而不是纯 RGB。

---

## Part II：方法与洞察

作者的设计哲学非常明确：**不是让策略更聪明，而是让观测更对题**。

### 方法主线

PRISM-DP 的流程基本是一个“感知前端 + 标准扩散策略”的串联：

1. **选择首次可见帧**  
   对每个目标物，用户指定它第一次清晰可见的帧 \(Fi\)。

2. **分割目标物**  
   用 SAM2 接收点提示，在该帧得到目标物 mask。

3. **从 mask 图生成 3D mesh**  
   把“目标物保留、其余区域置黑”的图送入 Meshy，自动生成三角 mesh。

4. **初始化 pose**  
   用 FoundationPose 在首次可见帧上，结合 RGB、深度、mask、相机内参和生成 mesh，估计初始 6D pose。

5. **跨时间跟踪 pose**  
   用 FoundationPose 在后续帧持续更新每个物体的 6D pose。若某物体尚未出现，则其位姿填零。

最终，策略看到的不是 RGB 序列，而是：

- 机器人 proprioception；
- 多个目标物的 pose history。

扩散策略本体并没有被大改：仍然是标准 action diffusion，只是把条件输入从图像历史改成了**低维几何状态历史**。

### 核心直觉

- **改了什么**：把策略条件从“高维像素流”换成“显式的对象位姿流”。
- **改变了哪个瓶颈**：把“物体发现 + 3D恢复 + 跟踪”这部分难题，从策略网络内部隐式学习，前移到可复用的模块化感知栈。
- **能力为什么变强**：策略容量不再浪费在压缩视觉冗余上，而能更集中地拟合“几何状态 → 动作”的映射，所以在相同参数量下更容易学好，也更容易在真实场景稳定工作。
- **为什么这个设计有效**：对多数抓取、堆叠、插入、放置任务来说，动作主要由相对几何关系决定，而不是由原始纹理本身决定。只要上游给出的 pose 足够稳定，控制器就不必再从像素里重复恢复这些信息。
- **为什么生成 mesh 不必完美**：这里 mesh 的角色不是高精度 CAD 建模，而是给 pose tracker 提供一个足够可信的几何模板。只要在感知特征空间里与真实物体足够接近，通常就足以支撑后续跟踪与控制。

### 战略权衡

| 设计选择 | 收益 | 代价 / 风险 |
| --- | --- | --- |
| RGB 观测 → pose 观测 | 大幅降维，减少任务无关信息，提升训练效率 | 对上游感知误差敏感 |
| 生成 mesh 代替手工 mesh | 去掉人工建模步骤，提升 open-set 可扩展性 | mesh 拓扑可能冗余，增加内存和跟踪开销 |
| FoundationPose 持续跟踪 | 提供平滑的时序几何状态 | 遮挡、离视野、多物体干扰时会漂移 |
| 首次可见帧点提示 + 预缓存 mesh | 在线执行可达 10 Hz，部署可行 | 仍需人工提示，不是完全自动系统 |
| 模块化感知前端 + 标准 diffusion policy | 前端可复用于别的 imitation/policy 框架 | 系统性能受第三方模型上限共同决定 |

---

## Part III：证据与局限

### 关键信号

- **比较 / 仿真：同规模图像策略明显不够**
  - 在 5 个 Robosuite 任务上，PRISM-DP-T / PRISM-DP-U 都稳定超过同规模图像条件策略。
  - 代表性信号：Square 上 PRISM-DP-T 为 **0.78**，而 DP-T Img 只有 **0.08**；Mug 上 **0.61 vs 0.27**。
  - 这说明性能差距不是偶然，而是**观测表征质量**在起主导作用。

- **比较 / 仅靠放大图像模型不能根治问题**
  - 8-11× 更大的图像模型在 Lift、Can 这种简单任务上能部分追近；
  - 但在 Stack、Square、Mug 这类更依赖几何关系的任务上，仍落后于 PRISM-DP。
  - 结论：瓶颈不只是“参数太小”，而是 raw RGB 条件里混入了太多与控制无关的扰动。

- **比较 / 真实世界是最强证据**
  - 两个真实任务中，小容量图像策略全部失败，成功率都是 **0.00**。
  - PRISM-DP-T 在 **Block Stacking = 0.93**、**Drawer Interaction = 0.87**；
  - 大图像模型分别只有 **0.87** 和 **0.57**。
  - 这直接支撑本文最重要的结论：**在真实 manipulation 中，pose-conditioned diffusion policy 可以优于 raw-image-conditioned policy。**

- **分析 / 自动 mesh 确实够用**
  - 生成 mesh 的 **FID = 0.1347**，**PSNR = 26.46**；
  - 用这些 mesh 做 pose tracking，平均位置误差仅 **0.0006**。
  - 方向误差相对更高，但作者说明主要来自生成 mesh 与 GT CAD 的 canonical orientation 不一致，而不是时序跟踪崩掉。

如果只挑一条最能说明“能力跃迁”的证据，我会选真实 **Drawer Interaction**：  
**PRISM-DP-T 0.87 vs DP-T-L Img 0.57 vs DP-T Img 0.00**。  
这个任务同时包含遮挡处理、抓取、放入抽屉和关闭抽屉，比单步抓取更能体现 object-centric 几何观测的价值。

### 局限性
- Fails when: 目标物严重遮挡、首次出现后又离开视野、多个动态物体导致 pose drift 持续积累，或对象本身发生非刚体形变时，方法容易退化。
- Assumes: 需要 RGB-D 相机和已知内参；需要用户在首次可见帧提供点提示；默认任务相关物体事先已知；依赖 SAM2、Meshy、FoundationPose 等外部组件；运行时使用训练阶段缓存的 mesh。
- Not designed for: 布料/软体操作、纯 RGB 无深度场景、完全无提示的目标发现、需要高精 CAD 拓扑做精细接触建模的任务。

### 证据强度为何是 moderate
这篇论文的证据已经比很多系统论文更扎实：有仿真、有真实、也有前端感知质量分析。  
但按保守标准，它还不到 `strong`，原因有三点：

1. **缺少系统级消融**：没有明确拆分 segmentation / mesh generation / pose tracking 各模块对最终成功率的因果贡献。
2. **真实任务数量有限**：只有 2 个真实任务，且每个任务 30 次 rollout。
3. **复现依赖较重**：系统依赖第三方基础模型，尤其 mesh 生成部分可能受外部服务与闭源实现约束。

### 可复用部件
- **mask-to-mesh 自举**：从一次目标分割直接得到下游可跟踪 mesh。
- **generated-mesh pose bootstrapping**：用生成 mesh 初始化 FoundationPose，而不是依赖人工 CAD。
- **object-pose history conditioning**：把“机器人状态 + 多物体 pose history”作为 imitation learning / diffusion control 的通用条件接口。

## Local PDF reference

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_PRISM_DP_Spatial_Pose_based_Observations_for_Diffusion_Policies_via_Segmentation_Mesh_Generation_and_Pose_Tracking.pdf]]