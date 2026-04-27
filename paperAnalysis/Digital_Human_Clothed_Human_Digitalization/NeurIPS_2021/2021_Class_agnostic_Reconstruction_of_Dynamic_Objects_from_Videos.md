---
title: "Class-agnostic Reconstruction of Dynamic Objects from Videos"
venue: NeurIPS
year: 2021
tags:
  - Others
  - task/4d-reconstruction
  - task/3d-reconstruction
  - implicit-neural-representation
  - pixel-alignment
  - neural-ode
  - "dataset/SAIL-VOS 3D"
  - "dataset/DeformingThings4D++"
  - dataset/3DPW
  - opensource/no
core_operator: 在统一 canonical 空间中，用像素对齐的跨帧特征聚合驱动 4D occupancy，并通过 neural ODE 流场把几何与运动在时间上对齐。
primary_logic: |
  RGBD视频、实例mask与相机矩阵 → 将各帧可见表面提升到共享 canonical 空间，并用流场把查询点传播到各时间步后提取像素对齐特征、经时序聚合预测 occupancy → 得到 canonical 网格并传播为每帧动态网格
claims:
  - "在 SAIL-VOS 3D 上，REDO 相比 OFlow 将 mIoU 从 26.0 提升到 31.9，同时将 mCham. 从 0.732 降到 0.647、mACD 从 1.69 降到 1.47 [evidence: comparison]"
  - "在 3DPW 上，REDO 相比 OFlow 将 mIoU 从 31.5 提升到 41.6，并将 mCham. 从 0.461 降到 0.337 [evidence: comparison]"
  - "去掉像素对齐表示后，SAIL-VOS 3D 上 mIoU 从 31.9 降到 24.1、mCham. 从 0.647 恶化到 0.937，说明空间对齐是其性能关键 [evidence: ablation]"
related_work_position:
  extends: "PIFu (Saito et al. 2019)"
  competes_with: "Occupancy Flow (Niemeyer et al. 2019); SurfelWarp (Gao and Tedrake 2018)"
  complementary_to: "Mask R-CNN (He et al. 2017); Consistent Video Depth (Luo et al. 2020)"
evidence_strength: strong
pdf_ref: paperPDFs/Digital_Human_Clothed_Human_Digitalization/NeurIPS_2021/2021_Class_agnostic_Reconstruction_of_Dynamic_Objects_from_Videos.pdf
category: Others
---

# Class-agnostic Reconstruction of Dynamic Objects from Videos

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2112.02091) · [Project](https://jason718.github.io/redo)
> - **Summary**: REDO 把动态物体重建转成“共享 canonical 空间里的查询点证据汇聚”问题，用像素对齐的跨帧特征和显式流场同时解决遮挡补全与时序一致性。
> - **Key Performance**: 在 SAIL-VOS 3D 相对 OFlow 提升 +5.9 mIoU / -0.22 mACD；在 3DPW 提升 +10.1 mIoU / -0.124 mCham.

> [!info] **Agent Summary**
> - **task_path**: RGBD/标定视频 + 实例 mask + 相机矩阵 -> 每帧完整 3D 网格与跨帧对应
> - **bottleneck**: 物体长期局部可见，且刚体运动、非刚体形变、关节运动混杂，导致逐帧重建缺上下文、全局视频编码又丢空间细节
> - **mechanism_delta**: 用 canonical-space 4D implicit occupancy 替代逐帧建模，并用 neural-ODE 流场把查询点对齐到各帧后做 transformer 式像素证据聚合
> - **evidence_signal**: 3 个数据集上持续优于 OFlow/静态重建基线，且去掉 pixel alignment 或 temporal loss 明显退化
> - **reusable_ops**: [canonical-space volume construction, query-centric pixel-aligned temporal aggregation]
> - **failure_modes**: [small-scale articulation is inaccurate, long-horizon propagation drifts away from the canonical frame]
> - **open_questions**: [can temporal correspondence supervision be removed, can the method work without reliable depth/mask/camera inputs]

## Part I：问题与挑战

这篇论文解决的不是普通的“多视角拼接 3D”，而是更难的 **部分可见条件下的完整 4D 重建**：输入视频里，目标物体可能从头到尾都没被完整看见，但系统仍要恢复它在每一帧的完整几何和动态。

### 真正的难点是什么
核心瓶颈不是“用什么 3D 表示”，而是：

1. **信息不完整**：遮挡、裁切、前视角限制让很多表面永远不可见。
2. **运动类型混杂**：同一个统一框架要同时覆盖刚体平移/旋转、非刚体形变、关节 articulation。
3. **类别变化大**：不想依赖 SMPL 这类类别模板，而要做人、动物、车等类无关重建。

已有方法各有短板：
- fusion 类方法多只能恢复**看得见的表面**；
- template 类方法强依赖**类别先验**；
- OFlow 这类 4D 方法把整段视频压成单个特征，**空间位置信息丢失**，且对 rigid motion 不够友好。

### 输入/输出接口
- **输入**：17 帧 RGBD 或标定视频、实例 mask、相机矩阵。
- **输出**：canonical 空间中的完整网格，以及传播到各时间步的动态网格。

### 为什么现在值得做
作者的判断是：深度传感器、实例分割、相机/深度估计工具已经越来越可得，因此有条件从“理想化多视角/单类别模板”走向更真实的类无关视频重建设置。

### 边界条件
这不是完全无约束设置。它默认：
- 推理时有 **depth + mask + camera**，或能由外部系统估计出来；
- 训练时有 **occupancy 标注和跨帧 mesh correspondence**；
- canonical 体积的后向深度 `Zfar` 用固定启发式估计，因此对极端厚度变化并不稳健。

## Part II：方法与洞察

REDO 的设计主线可以概括成一句话：**先把运动物体放进一个共享坐标系，再对每个 3D 查询点跨帧收集证据，而不是先把整段视频压缩成一个全局向量。**

### 方法主干

#### 1. 构建 canonical 空间
作者把各帧 mask 内的深度点 lift 到 3D，再映射到中心帧坐标系，得到一个围住物体的 3D 体积。这个 canonical 空间本质上把“视频中的移动物体”变成“共享坐标中的对象”。

作用：
- 把不同时间看到的局部表面汇到同一参考系；
- 让 rigid motion 不再直接表现为“对象整体在世界里乱跑”，而变成“canonical 查询点如何随时间传播”。

#### 2. 像素对齐的 4D implicit reconstruction
对 canonical 空间中的每个查询点，REDO 不直接看全局视频特征，而是：
- 先用流场把这个点传播到各时间步；
- 再投影回对应图像；
- 如果落在目标 mask 内，就从 2D 特征图取 **pixel-aligned feature**；
- 最后用 transformer 式时间聚合器汇总这些可见证据，预测该点是否在物体内部。

这一步把“4D 重建”改成了 **query-centric 跨帧检索**。

#### 3. 显式流场建模动态
REDO 用 velocity field + neural ODE 定义连续时间流场，来表示点如何从 canonical 时刻运动到别的时刻。

这里有两个不同角色的特征：
- **static feature**：看同一个 canonical 位置在各帧的证据，帮助判断这里是否稳定/是否在动；
- **dynamic feature**：看“传播后的对应点”在各帧的证据，用于最终 occupancy 判断。

这让“建模运动”和“判断是否属于物体”不再完全混在一起。

#### 4. 推理方式
先在 canonical 空间里重建网格，再把网格顶点传播到每一帧。  
因此，输出天然带有时序对应关系，而不是每帧独立预测后再硬对齐。

### 核心直觉

**改变了什么**：  
从“整段视频 -> 单个全局表示 -> 解码 4D”改成“每个 3D 查询点 -> 跨帧对齐 -> 聚合局部像素证据”。

**哪种瓶颈被改掉了**：
- **信息瓶颈**：全局视频编码会抹平局部几何细节；query-level pixel alignment 则保留了空间细粒度。
- **约束瓶颈**：canonical 空间把多帧碎片观察统一到一个对象中心坐标系，减轻了 rigid motion 带来的分布漂移。
- **观测瓶颈**：transformer set aggregation 允许“某些帧看得见、某些帧看不见”，不要求每帧都有有效证据。

**能力上带来的变化**：
- 能从局部、遮挡、前视角视频里做更完整的形状补全；
- 能统一处理 rigid / non-rigid / articulation；
- 能在类无关设置里保持时序一致的网格输出。

### 为什么这个设计有效
因果上看，关键不是“用了 implicit function”，而是 **把监督信号和观测证据都绑到了同一个查询点上**：
- pixel alignment 让 3D 预测直接受 2D 局部证据约束，减少 OFlow 式过平滑；
- center-frame canonicalization 缩短了向前/向后传播的平均距离，降低漂移；
- temporal loss 直接监督 flow，对“看起来像对，但时序错位”的解形成约束。

### 战略权衡

| 设计选择 | 改变的瓶颈 | 得到的能力 | 代价/风险 |
|---|---|---|---|
| canonical 空间 | 把多帧碎片观察放到同一参考系 | 更容易聚合可见片段、兼容 rigid motion | 体积边界估计错误会影响整段重建 |
| pixel-aligned 特征 | 保留局部几何与外观对应 | 对遮挡补全和细节恢复更稳 | 强依赖 mask / depth / calibration 质量 |
| transformer 时间聚合 | 处理缺失观测和可见帧数量变化 | 比平均池化更能挑选有效帧 | 计算更重 |
| neural ODE 流场 | 统一连续时间运动建模 | 同时覆盖刚体与非刚体传播 | 离 canonical 帧太远会累积误差 |
| temporal correspondence loss | 显式约束动态一致性 | mACD 明显更好 | 训练需要对应 mesh，监督成本高 |

## Part III：证据与局限

### 关键实验信号

1. **跨数据集比较信号：能力不是单一数据集偶然**
   - SAIL-VOS 3D：REDO 达到 **31.9 mIoU / 0.647 mCham. / 1.47 mACD**，优于 OFlow 的 26.0 / 0.732 / 1.69。
   - DeformingThings4D++：也稳定优于 OFlow，说明不仅对单一场景有效。
   - 3DPW：达到 **41.6 mIoU / 0.337 mCham. / 0.846 mACD**，明显优于 OFlow 的 31.5 / 0.461 / 0.907，表明对真实视频仍有泛化。

2. **Ablation 信号：提升来自关键因果部件**
   - 用平均池化替代 transformer，性能整体下降；
   - 去掉 pixel alignment 后，mIoU 从 **31.9 降到 24.1**，是最大退化，说明“查询点级空间对齐”是主增益来源；
   - 去掉 temporal loss 后，canonical 帧形状还能重建，但 **mACD 从 1.47 恶化到 3.12**，说明 flow 若无显式对应监督，时序一致性会明显崩。

3. **泛化信号：零样本类别上没崩**
   - 在 dog / gorilla / puma 等未见类别上的平均结果仍显著优于大多数基线，说明它学到的不是单类别模板记忆。

### So what：相对 prior 的能力跃迁
真正的能力跃迁是：  
**从“可见面重建 / 类别模板重建 / articulation-only 建模”推进到“部分可见条件下的类无关完整 4D 重建”。**

最有说服力的证据有两个：
- 相比 OFlow，形状和动态指标都提升；
- 去掉像素对齐或 temporal supervision 后，关键能力明显回落，说明改进不是训练技巧，而是结构性变化。

### 局限性
- **Fails when**: 细粒度局部运动和高频细节较难恢复，如手部开合、衣物、车灯和轮胎等；传播到离 canonical 帧较远的位置时性能会下降；低样本类别如 airplane、bicycle 更难。
- **Assumes**: 推理依赖较可靠的 depth、实例分割和相机矩阵；训练依赖 occupancy 标注与跨帧对应 mesh；canonical 体积的后深度用启发式固定距离估计；作者报告训练约需 4 张 GPU、约 30 小时，并依赖 ODE 求解器。
- **Not designed for**: 纯单目、无深度、无标定的完全自监督 4D 重建；场景级多物体联合重建；高保真表面细节建模；不做目标分割的端到端场景理解。

### 可复用组件
- 用 lift + 聚合构造对象中心的 canonical volume；
- 基于查询点的 pixel-aligned 跨帧证据汇聚；
- 用 neural ODE 定义连续时间流场并传播 canonical mesh。

## Local PDF reference

![[paperPDFs/Digital_Human_Clothed_Human_Digitalization/NeurIPS_2021/2021_Class_agnostic_Reconstruction_of_Dynamic_Objects_from_Videos.pdf]]