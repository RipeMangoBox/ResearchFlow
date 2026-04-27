---
title: "GauU-Scene: A Scene Reconstruction Benchmark on Large Scale 3D Reconstruction Dataset Using Gaussian Splatting"
venue: arXiv
year: 2024
tags:
  - Survey_Benchmark
  - task/3d-reconstruction
  - task/novel-view-synthesis
  - gaussian-splatting
  - lidar-fusion
  - point-cloud-registration
  - dataset/U-Scene
  - opensource/partial
core_operator: "通过无人机 RGB-LiDAR 联合采集、COLMAP 坐标桥接和图像/点云双空间评测，构建面向大场景 3DGS 的重建基准并暴露 2D 指标的诊断盲区。"
primary_logic: |
  大尺度场景重建评测需求 → 采集并对齐无人机 RGB 与 LiDAR 得到 U-Scene → 以 Vanilla / LiDAR-Fused 3DGS 重建并同时用图像与点云指标评分 → 揭示 2D 新视角分数与 3D 结构质量之间的偏差
claims:
  - "U-Scene 提供了覆盖超过 1.5 km² 的无人机 RGB + LiDAR 大场景数据，并包含屋顶观测与点云真值 [evidence: case-study]"
  - "在 U-Scene 的三个子场景上，LiDAR-Fused Gaussian Splatting 的点云 L1 均低于 Vanilla Gaussian Splatting（23.65 vs 27.40；23.43 vs 27.69；22.22 vs 25.33） [evidence: comparison]"
  - "图像空间指标只显示边际收益：三组场景的 PSNR 提升均小于 0.4 dB，这说明仅依赖新视角图像分数不足以诊断大场景 3D 结构误差 [evidence: analysis]"
related_work_position:
  extends: "3D Gaussian Splatting (Kerbl et al. 2023)"
  competes_with: "UrbanBIS (Yang et al. 2023); Block-NeRF (Tancik et al. 2022)"
  complementary_to: "Mip-Splatting (Yu et al. 2023); COLMAP (Schönberger and Frahm 2016)"
evidence_strength: moderate
pdf_ref: paperPDFs/Large_Scale/arXiv_2024/2024_GauU_Scene_A_Scene_Reconstruction_Benchmark_on_Large_Scale_3D_Reconstruction_Dataset_Using_Gaussian_Splatting.pdf
category: Survey_Benchmark
---

# GauU-Scene: A Scene Reconstruction Benchmark on Large Scale 3D Reconstruction Dataset Using Gaussian Splatting

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2401.14032)
> - **Summary**: 论文构建了一个覆盖 1.5 km² 以上、同时包含无人机 RGB 与 LiDAR 真值的 U-Scene 基准，并用图像/点云双指标表明：仅看新视角图像分数，会明显低估大场景 3D Gaussian Splatting 的几何误差。
> - **Key Performance**: 点云 L1（越低越好）在 3/3 子场景均优于 Vanilla 3DGS：CUHK LOWER 27.40→23.65，CUHK UPPER 27.69→23.43，SMBU 25.33→22.22；但图像 PSNR 提升很小，最高仅 +0.323 dB。

> [!info] **Agent Summary**
> - **task_path**: 无人机多视角 RGB + LiDAR 采集/对齐 -> 大尺度室外场景 3DGS 重建与评测
> - **bottleneck**: 现有大场景数据集缺少可与图像严格对齐的 3D 真值，且仅靠 2D 新视角指标无法可靠反映真实几何质量
> - **mechanism_delta**: 用 COLMAP 稀疏 SfM 作为桥梁把 LiDAR 点云变换到相机坐标系，并把下采样 LiDAR 作为 3DGS 先验，同时新增点云级评测
> - **evidence_signal**: 三个子场景上点云 L1 都明显优于 vanilla，而图像 PSNR/L1 基本不变
> - **reusable_ops**: [COLMAP-LiDAR 坐标桥接, 下采样 LiDAR 初始化 3DGS]
> - **failure_modes**: [建筑边缘和近距离区域出现模糊/黑云伪影, 仅用训练视图图像指标时方法差异被弱化]
> - **open_questions**: [如何构建真正独立的 held-out 大场景 3D 评测集, 如何把 LiDAR 约束更深地注入 3DGS 而不只是初始化]

## Part I：问题与挑战

这篇论文真正要解决的，不是“再提一个 3DGS 变体”，而是**给大尺度室外 3D 重建建立一个更可信的 benchmark**。

### 1）难点到底在哪
现有大场景重建数据与评测，主要卡在三件事：

1. **数据真值不完整**
   - 卫星/街景类数据容易存在跨时间采集，造成论文里说的 *image time difference*。
   - 车载 LiDAR 往往缺少楼顶信息。
   - 一些无人机数据虽然有点云，但图像和点云坐标关系不够清楚，难以直接做 RGB-LiDAR 联合重建。

2. **坐标系统不一致**
   - 原始 LiDAR 点云在 UTM 坐标。
   - 相机位姿则更适合在 COLMAP/SfM 相对坐标系中使用。
   - 如果两者不能对齐，就没法把 LiDAR 真正变成 3DGS 的几何先验。

3. **评测本身有盲区**
   - 只看 novel-view PSNR/L1，本质上还是 2D 投影误差。
   - 大尺度场景里，**一个几何错误在投影图像里可能并不显著**，尤其当评测还依赖训练图像时，这个问题更严重。

### 2）输入 / 输出接口
- **输入**：
  - 无人机多视角 RGB 图像
  - LiDAR 点云真值
- **输出**：
  - 可用于 3DGS 的大场景 benchmark
  - Vanilla 3DGS 与 LiDAR-Fused 3DGS 的对照结果
  - 图像域与点云域两套评测信号

### 3）为什么现在值得做
因为 3D Gaussian Splatting 已经成为高效新视角合成/场景表达的重要路线，但它在**无人机采集的大尺度城市场景**上，缺少一个同时具备：
- 屋顶覆盖，
- 几何真值，
- 可对齐多模态输入，
- 面向 3D 结构而不仅是图像渲染的评测方式。

U-Scene 就是在补这个空档。

### 4）边界条件
这篇工作聚焦的是：
- 静态大尺度室外场景；
- 三个校园区域；
- 覆盖超过 1.5 km²；
- 无人机飞行高度约 120m；
- 有效点云范围约 300m；
- 点云采样约 20cm/点。

也就是说，它不是一个“任意场景通用 benchmark”，而是偏向**无人机-校园/城市场景-高质量几何真值**这个设定。

## Part II：方法与洞察

这篇论文的方法部分可以理解成两层：

1. **benchmark 构建层**：怎么把 RGB、LiDAR、相机坐标准确接起来；
2. **probe baseline 层**：怎么用一个简单的 LiDAR-fused 3DGS 去证明该 benchmark 的诊断价值。

### 方法主线

#### A. 数据采集
作者使用 DJI Matrix 300 + Zenmuse L1，以斜拍路径规划获取：
- 建筑立面
- 屋顶
- 大范围校园区域

这个设计直接针对传统车载方案的短板：**车载只能看街道，难看楼顶；无人机可以补齐顶部与高处结构。**

#### B. 坐标桥接：LiDAR → COLMAP
核心做法不是直接从 UTM 去匹配相机，而是先：
1. 用 COLMAP 从图像恢复稀疏点云和相机位姿；
2. 将原始 LiDAR 点云做尺度调整；
3. 进行粗配准；
4. 再用 ICP 精配准；
5. 把 LiDAR 变换到 COLMAP 相对坐标系。

这样，LiDAR 就从“独立的真值文件”变成了“3DGS 可消费的几何先验”。

#### C. LiDAR-Fused Gaussian Splatting
由于原始 LiDAR 太稠密，作者先做下采样，再把它当成 3DGS 初始化点云。

因此论文里的“fusion”其实很朴素：  
**不是复杂联合训练，而是把更可靠的 3D 几何先验喂给 3DGS。**

#### D. 双空间评测
作者做了两类评估：

- **图像空间**：L1、PSNR  
- **点云空间**：把 Gaussian center 视为点，与真值点云做匹配，再计算差异

这一步是论文最关键的 benchmark 洞察：  
**如果 benchmark 只给图像指标，你会误以为 LiDAR 先验几乎没帮助；但一旦看 3D 点云误差，改进就明显了。**

### 核心直觉

原来大场景 3DGS 的评测逻辑大多是：

**“渲染得像不像图像”**

这篇论文把它改成：

**“渲染像不像图像 + 几何像不像真实 3D 结构”**

这个变化看起来只是多加了一个模态，但本质上改的是**测量瓶颈**：

- **原来受限的地方**：2D 投影是 many-to-one，几何错误可能被视角、纹理和训练图像覆盖掩盖；
- **现在改变的约束**：加入 LiDAR 真值后，模型必须同时面对 3D 几何一致性；
- **能力变化**：benchmark 不再只测“像素拟合能力”，也开始测“结构真实性”。

再进一步说，LiDAR 先验为什么有效？  
因为大场景无人机重建里，最脆弱的部分往往不是颜色，而是：
- 建筑边界，
- 屋顶形状，
- 遮挡后的结构连续性，
- 稀疏 SfM 初始化不稳定的区域。

当初始化从“图像三角化得到的稀疏点”变成“对真实几何更接近的 LiDAR 子采样点”时，3DGS 的优化起点更合理，所以能减少边缘漂浮、高斯云团等伪影。

### 战略性取舍

| 设计选择 | 解决的瓶颈 | 收益 | 代价 / 假设 |
| --- | --- | --- | --- |
| 无人机斜拍 + LiDAR | 屋顶缺失、时间差、视角不足 | 覆盖屋顶与立面，适合大场景 | 依赖专用飞行平台、采集成本高 |
| COLMAP 稀疏点云做桥梁 | LiDAR 与相机坐标不统一 | 让点云可直接服务 3DGS | 需要粗配准 + ICP，流程偏工程化 |
| 下采样 LiDAR 初始化 3DGS | 纯 SfM 初始化几何弱 | 提供更稳定 3D 先验 | 对采样策略敏感，提升幅度有限 |
| 图像 + 点云双指标 | 2D 指标无法诊断几何 | 更能揭示真实结构误差 | 需要点云真值和更复杂匹配评测 |

## Part III：证据与局限

### 关键证据信号

#### 信号 1：图像指标几乎看不出明显差距
作者在三个子场景上比较了 Vanilla 3DGS 和 LiDAR-Fused 3DGS。  
结果是图像域提升很小，比如：
- CUHK LOWER：PSNR 28.660 → 28.742
- CUHK UPPER：26.911 → 26.949
- SMBU：27.010 → 27.333

**结论**：如果只看图像 PSNR/L1，你会得到“LiDAR 融合帮助不大”的判断。

#### 信号 2：点云指标能明显区分两者
点云 L1 上，LiDAR-Fused 在三个子场景全部更好：
- CUHK LOWER：27.40 → 23.65
- CUHK UPPER：27.69 → 23.43
- SMBU：25.33 → 22.22

**结论**：3D 几何真值比 2D 渲染误差更能揭示大场景重建质量，这恰恰支持了 benchmark 的核心主张。

#### 信号 3：定性结果集中暴露“边缘问题”
论文给出的可视化中，Vanilla 3DGS 更容易出现：
- 黑色模糊云团
- 建筑边缘不规则模糊
- 近距离观察时细节发糊

而 LiDAR-Fused 结果更干净。  
**结论**：LiDAR 先验的主要作用不是大幅提升像素分数，而是抑制结构性伪影。

### 1-2 个最该记住的指标
- **最能说明 benchmark 价值的指标**：点云 L1 在三组场景全部明显下降
- **最能说明旧评测盲区的指标**：PSNR 提升始终很小，说明 2D 视角分数并不等价于 3D 重建质量

### 局限性

- **Fails when**: 近距离观察建筑边界、屋顶边缘或高频结构时，3DGS 仍会出现明显边缘效应、模糊和黑云伪影；如果评测只依赖训练图像，方法差异会被显著弱化。
- **Assumes**: 需要高精度无人机 LiDAR 硬件（DJI Matrix 300 + Zenmuse L1）与相关软件链路；依赖 COLMAP 稳定恢复相机位姿；配准流程包含手工粗对齐与 ICP；默认场景基本静态。
- **Not designed for**: 动态街景、跨季节/跨时间变化建模、统一比较大量非 3DGS 重建方法的标准化 leaderboard，也没有提供严格的独立测试视角协议。

另外两个很现实的限制也值得明确指出：
1. **评测集设计还不够“严格 benchmark 化”**：文中明确说当前没有额外测试图像，因此图像指标是在训练图像上算的，这会高估方法表现、低估方法差距。
2. **可复现性有明显工程依赖**：无人机安全飞行、斜拍路径规划、CloudCompare/ICP 粗精配准、DJI Terra/Pilot 等都带来较强工具链依赖；虽然数据已发布，但看不到系统化代码/协议公开，因此更适合“可借鉴流程”，不算“开箱即复现”。

### 可复用部件
- **COLMAP ↔ LiDAR 坐标桥接流程**：适合任何需要把测绘点云转成神经重建先验的任务。
- **点云真值驱动的 3DGS 评测思路**：可用于检查“图像好看但几何不准”的问题。
- **无人机斜拍 + 屋顶覆盖采集范式**：适合校园、园区、建筑群等大尺度静态场景。

## Local PDF reference

![[paperPDFs/Large_Scale/arXiv_2024/2024_GauU_Scene_A_Scene_Reconstruction_Benchmark_on_Large_Scale_3D_Reconstruction_Dataset_Using_Gaussian_Splatting.pdf]]