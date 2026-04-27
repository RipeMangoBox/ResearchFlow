---
title: "4D Gaussian Splatting for Real-Time Dynamic Scene Rendering"
venue: arXiv
year: 2023
tags:
  - 3D_Gaussian_Splatting
  - task/novel-view-synthesis
  - task/dynamic-scene-rendering
  - gaussian-splatting
  - deformation-field
  - hexplane
  - dataset/D-NeRF
  - dataset/HyperNeRF
  - dataset/Neu3D
  - opensource/full
core_operator: 用单套canonical 3D高斯配合HexPlane式时空形变场，在任意时间预测高斯的位置、旋转与尺度变化后直接进行splatting渲染。
primary_logic: |
  多视角/单目视频帧 + 相机位姿 + 时间戳 → 初始化一组canonical 3D高斯，并用分解的4D神经体素编码局部时空结构 → 轻量多头解码器预测各时间的高斯位置/形状形变 → 对形变后的3D高斯执行高斯光栅化，得到任意视角/时刻的动态场景图像
claims:
  - "Compared with per-timestamp Gaussian tables, 4D-GS stores one canonical Gaussian set plus a deformation network, reducing memory complexity from O(tN) to O(N+F) [evidence: theoretical]"
  - "On the D-NeRF synthetic benchmark, 4D-GS reaches 34.05 PSNR and 82 FPS at 800×800, outperforming or matching prior dynamic NeRF baselines while using 18 MB storage [evidence: comparison]"
  - "The spatial-temporal HexPlane encoder and position deformation head are material: removing the encoder drops PSNR from 34.05 to 27.05, and removing the position head drops it to 26.67 on synthetic scenes [evidence: ablation]"
related_work_position:
  extends: "3D Gaussian Splatting (Kerbl et al. 2023)"
  competes_with: "TiNeuVox (Fang et al. 2022); HexPlane (Cao and Johnson 2023)"
  complementary_to: "N/A"
evidence_strength: strong
pdf_ref: paperPDFs/Dynamic_Video/arXiv_2023/2023_4D_Gaussian_Splatting_for_Real_Time_Dynamic_Scene_Rendering.pdf
category: 3D_Gaussian_Splatting
---

# 4D Gaussian Splatting for Real-Time Dynamic Scene Rendering

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2310.08528), [Project](https://guanjunwu.github.io/4dgs/)
> - **Summary**: 这篇工作把动态场景表示成“单套 canonical 3D 高斯 + 时空形变场”，把慢速的时变体渲染改成对高斯先形变再直接 splatting，从而把动态新视角渲染推进到实时区间。
> - **Key Performance**: D-NeRF 上达到 **34.05 PSNR** 和 **82 FPS@800×800**；Neu3D 上达到 **30 FPS@1352×1014**，LPIPS **0.049**。

> [!info] **Agent Summary**
> - **task_path**: 稀疏时序图像/相机位姿/时间戳 -> 任意视角与任意时刻的动态场景渲染
> - **bottleneck**: 稀疏时空观测下，复杂运动建模通常要么依赖昂贵体渲染采样，要么需要逐帧存储大量显式表示，难以同时做到实时、紧凑和高保真
> - **mechanism_delta**: 用单套 canonical 高斯和 HexPlane 式时空编码共享局部运动先验，再以轻量多头解码器预测位置/旋转/尺度形变
> - **evidence_signal**: 跨 D-NeRF、HyperNeRF、Neu3D 的比较 + 关键 ablation，显示其在保持或提升质量的同时把自由视角渲染速度提升到实时
> - **reusable_ops**: [canonical-gaussians, hexplane-deformation-field]
> - **failure_modes**: [large-motion-and-dramatic-scene-change, monocular-novel-view-overfitting]
> - **open_questions**: [how-to-disentangle-static-dynamic-motion-under-monocular-input, how-to-scale-deformation-querying-to-urban-scenes]

## Part I：问题与挑战

这篇论文真正要解决的，不是“动态场景能不能重建”，而是：

**在稀疏时间采样、稀疏视角甚至单目输入下，能否同时做到高质量、低存储、实时渲染的动态新视角合成？**

### 问题是什么
- **输入**：带时间戳的视频帧、相机位姿，以及用于初始化的点云/SfM 结果。
- **输出**：任意视角、任意时间的动态场景渲染结果。
- **目标**：保留 3D Gaussian Splatting 的实时优势，同时补上动态建模能力。

### 真正瓶颈是什么
已有动态 NeRF 路线大多是：
- 对每条光线上的大量采样点做时变查询，**渲染慢**；
- 或者给每个时间戳单独存一套显式表示，**存储随时间线性膨胀**。

而 3D-GS 虽然静态场景极快，但一旦直接用于动态场景，就缺少时间维度的运动/形变建模。  
所以核心矛盾是：

**如何只维护一个紧凑的场景表示，却能在不同时间生成准确的动态几何，并继续沿用高效的 splatting 渲染路径。**

### 为什么是现在
因为 3D-GS 已经证明：**显式高斯 + rasterization** 可以把静态 NVS 推到实时。  
现在最自然的问题就是：能否把“静态高斯”提升为“可随时间形变的高斯”，而不是退回慢速体渲染。

### 边界条件
这篇方法默认的适用前提比较明确：
- 更适合 **中等运动幅度**、相机位姿较准、可做 SfM 初始化的场景；
- 可以处理单目，但 **单目 + 大位移 + 稀疏视角** 仍容易落入局部最优；
- 不是一个跨场景泛化模型，而是 **逐场景优化** 的方法。

## Part II：方法与洞察

作者的策略很直接：  
**不要给每一帧新建一套高斯，也不要沿着每条 ray 做时变体渲染；只保留一套 canonical 3D Gaussians，然后让时间条件的形变场把它们变成当前时刻的高斯，再用 3D-GS 的 rasterizer 直接渲染。**

### 核心直觉

**变化了什么**：  
从“逐采样点查询动态辐射场 / 逐帧存储高斯”  
变成“对 canonical 高斯做时间条件形变，然后直接 splatting”。

**哪种瓶颈被改变了**：  
- 计算瓶颈：从体渲染的密集采样，转为对高斯中心做一次局部时空特征查询；
- 存储瓶颈：从按时间展开的显式表，转为一套共享的 canonical 高斯 + 小形变网络；
- 信息瓶颈：从“每个点独立看时间”转为“邻近高斯共享局部时空运动先验”。

**能力发生了什么变化**：  
- 渲染路径仍然是高效的高斯 rasterization，因此能保持实时；
- 时间变化通过形变场注入，因此动态场景不再像原始 3D-GS 那样失效；
- 因为是显式高斯轨迹，天然更容易做 **跟踪与编辑**。

### 方法拆解

#### 1. 单套 canonical 3D Gaussians
方法只维护一组基础高斯，不随时间复制。  
给定时间 \(t\) 时，网络预测每个高斯的偏移，得到该时刻的 deformed Gaussians。

这一步的意义是：
- 避免逐帧存表带来的 O(tN) 膨胀；
- 保留 3D-GS 的显式几何先验；
- 让编辑、组合、跟踪变得自然。

#### 2. 时空结构编码器：HexPlane 式分解
作者没有直接用完整 4D voxel，而是把时空体素分解成 6 个二维平面：
- 空间面：xy, xz, yz
- 时空面：xt, yt, zt

再做多分辨率查询，把某个高斯中心和时间戳映射成时空特征。

这一步的因果作用是：
- **让邻近高斯共享运动信息**，而不是每个高斯独立“瞎猜”；
- 比完整 4D voxel 更省内存；
- 比纯小 MLP 更能保留局部结构与时间连续性。

#### 3. 多头形变解码器
作者把形变拆成三个头：
- 位置头：预测平移/运动
- 旋转头：预测姿态变化
- 尺度头：预测拉伸/收缩

为什么要拆？
因为动态物体不是只会“移动”，还会发生局部拉伸、扭转、形状变化。  
如果只预测位置，宏观运动可能能拟合，但细节伸缩和跟踪一致性会差。

#### 4. 先静后动的 warm-up
训练前 3000 iter 先优化静态高斯，再联合优化 4D 形变。

这不是简单训练技巧，而是优化稳定性的关键：
- 先把高斯摆到合理位置；
- 降低形变网络一开始就处理大位移的难度；
- 避免数值不稳定和坏局部极小值。

### 战略性权衡

| 设计选择 | 改变的约束/瓶颈 | 带来的能力 | 代价 |
| --- | --- | --- | --- |
| 单套 canonical 高斯替代逐帧高斯 | 存储不再随时间线性膨胀 | 低存储、可编辑、可跟踪 | 大位移/拓扑变化更难拟合 |
| HexPlane 式时空分解替代完整 4D voxel 或纯 MLP | 降低内存，同时让局部邻域共享信息 | 收敛快、运动更连贯 | 大场景时查询成本上升 |
| 位置/旋转/尺度三头解码 | 同时显式建模运动与形变 | 更好细节和形状一致性 | 训练更依赖稳定初始化 |
| 先静后动 warm-up | 先学几何再学时变 | 训练更稳、动态区域更容易捕捉 | 增加一个预热阶段，依赖 SfM 点 |

## Part III：证据与局限

### 关键证据信号

**1. 最强主张不是单点 SOTA，而是 speed-quality-storage 三者同时占优。**  
在 D-NeRF synthetic 上，4D-GS 达到 **34.05 PSNR / 82 FPS@800×800 / 18 MB**。  
这很关键，因为它不是单纯比质量，也不是单纯比速度，而是在动态场景里把两者一起拉到了可部署区间。相比 TiNeuVox、HexPlane 这类动态 NeRF，速度提升非常明显；相比原始 3D-GS，则证明真正的增益来自“动态形变建模”而非仅仅使用 GS 渲染器。

**2. 跨真实数据集仍保住了实时自由视角能力。**  
- HyperNeRF 上达到 **25.2 PSNR / 34 FPS**，优于文中多数快速动态 NeRF 基线；
- Neu3D 上虽然 **PSNR 不是绝对最高**，但在 **30 FPS / 90 MB** 下实现 **LPIPS 0.049**，说明它的真正优势是部署友好的速度-感知质量折中，而不是只追单一重建分数。

**3. Ablation 支持了作者的因果叙事。**  
- 去掉时空 HexPlane 编码器，PSNR 从 **34.05** 掉到 **27.05**；
- 去掉位置形变头，掉到 **26.67**；
- 去掉 warm-up 初始化，掉到 **31.91**。  

这说明：
- 共享的时空结构先验是必要的；
- 仅靠外观或尺度变化无法替代真实运动建模；
- 训练稳定性不是附属问题，而是动态高斯优化能否成功的前提。

### 局限性

- **Fails when**: 单目场景中出现大位移、剧烈场景变化、背景点缺失或相机位姿不准时，方法容易陷入局部最优，产生模糊、错误关联或新视角失败。
- **Assumes**: 依赖已知/可恢复的相机位姿与 SfM 初始化；采用逐场景训练；实验主要在单张 RTX 3090 上完成；默认局部时空邻域具有一定运动相关性。
- **Not designed for**: 城市场景级别的大规模重建、强拓扑变化的稳健建模、无需额外先验的单目大运动泛化渲染，以及在线增量式训练。

### 可复用组件

- **canonical-gaussian + deformation-field**：适合任何想把静态 GS 扩展到动态场景的方案。
- **HexPlane 式 4D 分解编码**：适合把完整 4D 特征场压缩为可训练、可查询的时空先验。
- **多头形变解码器**：适合把“位置变化”和“形状变化”解耦，增强跟踪/编辑能力。
- **warm-up static GS**：对于显式动态表示的稳定优化很实用。

**一句话看“so what”**：  
这篇工作的能力跃迁，不是“在所有数据集上绝对最高清”，而是**首次把动态场景新视角渲染稳定推到高分辨率实时，并且仍保持紧凑表示与可编辑性**。

## Local PDF reference

![[paperPDFs/Dynamic_Video/arXiv_2023/2023_4D_Gaussian_Splatting_for_Real_Time_Dynamic_Scene_Rendering.pdf]]