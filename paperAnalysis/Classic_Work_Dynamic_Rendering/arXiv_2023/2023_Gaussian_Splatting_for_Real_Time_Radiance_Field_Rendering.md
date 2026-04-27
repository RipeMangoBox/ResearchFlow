---
title: "3D Gaussian Splatting for Real-Time Radiance Field Rendering"
venue: "ACM TOG"
year: 2023
tags:
  - 3D_Gaussian_Splatting
  - task/novel-view-synthesis
  - anisotropic-splatting
  - tile-based-rasterization
  - adaptive-density-control
  - dataset/Mip-NeRF360
  - dataset/Tanks&Temples
  - dataset/DeepBlending
  - opensource/full
core_operator: "用各向异性3D高斯与可见性感知的tile-based splatting替代逐射线体采样，在保留体渲染可优化性的同时实现实时新视角合成。"
primary_logic: |
  多视图图像 + SfM相机/稀疏点 → 初始化并优化位置/旋转/尺度/不透明度/SH颜色的各向异性3D高斯，并交替执行clone/split增密与裁剪 → 通过深度排序的tile-based splatting输出实时高质量新视角图像
claims:
  - "Claim 1: 在 Mip-NeRF360 上，Ours-7K 仅需 6m25s 训练即可达到 25.60 PSNR，与 INGP-Big 的 25.59 PSNR 相当，但渲染速度达到 160 FPS（INGP-Big 为 9.43 FPS）[evidence: comparison]"
  - "Claim 2: 在 Ours-30K 设置下，方法在 Mip-NeRF360 上取得 27.21 PSNR / 0.815 SSIM / 0.214 LPIPS，并以 134 FPS 渲染；相较 Mip-NeRF360 的 27.69 / 0.792 / 0.237，其 SSIM 与 LPIPS 更优且速度从 0.06 FPS 提升到实时 [evidence: comparison]"
  - "Claim 3: 关键设计不可替代：将高斯改为各向同性或将反向传播限制为前 10 个 splats 时，平均 PSNR 分别从 26.05 降至 25.23 和 19.19 [evidence: ablation]"
related_work_position:
  extends: "Pulsar (Lassner and Zollhofer 2021)"
  competes_with: "Mip-NeRF360 (Barron et al. 2022); InstantNGP (Müller et al. 2022)"
  complementary_to: "N/A"
evidence_strength: strong
pdf_ref: paperPDFs/Classic_Work_Dynamic_Rendering/arXiv_2023/2023_Gaussian_Splatting_for_Real_Time_Radiance_Field_Rendering.pdf
category: 3D_Gaussian_Splatting
---

# 3D Gaussian Splatting for Real-Time Radiance Field Rendering

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2308.04079) · [Project/Code](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) · [DOI](https://doi.org/10.1145/3592433)
> - **Summary**: 这篇论文用可优化的各向异性 3D 高斯替代 NeRF 的逐射线体采样，并通过可见性感知的 splatting 渲染，把高质量辐射场新视角合成首次推进到 100+ FPS 的实时区间。
> - **Key Performance**: Mip-NeRF360 上 Ours-30K 达到 27.21 PSNR / 0.214 LPIPS 且 134 FPS；Ours-7K 约 6 分钟训练即可达 25.60 PSNR 与 160 FPS

> [!info] **Agent Summary**
> - **task_path**: 多视图图像 + SfM相机/稀疏点云 -> 静态场景新视角RGB渲染
> - **bottleneck**: 连续辐射场的逐射线采样会在空空间和高深度复杂度区域浪费大量计算，导致高质量方法很难同时做到快训练与实时渲染
> - **mechanism_delta**: 把隐式连续密度场改成显式各向异性3D高斯，并用全局排序的tile splatting近似体渲染与可见性累积
> - **evidence_signal**: 跨 Mip-NeRF360 / Tanks&Temples / Deep Blending 的质量-速度对比，加上 anisotropy 与 unlimited-gradient-depth 的关键消融
> - **reusable_ops**: [SfM稀疏点初始化, clone/split自适应增密]
> - **failure_modes**: [欠观测区域出现粗大或拉长高斯伪影, 大高斯在视角切换时产生popping]
> - **open_questions**: [如何显著降低训练与推理显存, 如何在复杂遮挡与强视角相关区域做更稳健的抗锯齿和可见性排序]

## Part I：问题与挑战

这篇论文解决的是一个很具体但长期卡住 NeRF 落地的问题：

**给定多张标定好的静态场景照片，能否既保持 SOTA 级别的新视角画质，又做到 1080p 实时渲染？**

### 真正的难点是什么

表面看，NeRF 的瓶颈像是“网络太大、训练太慢”；但这篇论文指出，**真正瓶颈是表示与渲染路径不匹配**：

- **高质量 NeRF**（如 Mip-NeRF360）依赖连续场表示，优化能力强，但渲染时仍需要沿每条光线做大量采样与累积；
- 这种 **ray marching** 天然会在空空间上浪费算力，而且每个像素都要重复做；
- 更快的方法（如 InstantNGP、Plenoxels）虽然压缩了网络或转向显式结构，但依然受制于：
  - 逐射线采样范式；
  - 结构化网格/哈希格的分辨率与对齐限制；
  - 高质量与高帧率很难同时成立。

换句话说，问题不只是“怎样把 NeRF 算得更快”，而是：

> **能不能找到一种既保留体渲染可微优化特性，又像传统图形学显式原语那样适合 GPU 实时渲染的场景表示？**

### 为什么是现在解决

因为在这篇工作之前，领域里已经出现两个清晰趋势：

1. **质量上限**：Mip-NeRF360 代表了当时画质最强的一档；
2. **速度探索**：InstantNGP / Plenoxels 证明训练可以压到分钟级。

所以时机已经成熟：  
**接下来缺的不是更大的网络，而是一种重新分配计算的位置——从“沿光线采样整个空间”改为“只渲染真正有贡献的显式原语”。**

### 输入/输出接口与边界条件

- **输入**：
  - 多视图 RGB 图像
  - SfM 标定相机
  - SfM 顺带给出的稀疏点云
- **输出**：
  - 任意新视角 RGB 图像
  - 目标是 1080p 下实时（≥30 FPS，实际远超）
- **适用场景**：
  - 静态场景
  - 室内/室外都可
  - bounded / unbounded scene 都可
- **边界条件**：
  - 真实场景通常依赖 SfM 稀疏点初始化
  - 视角覆盖不能太差
  - 不是为动态场景、单图重建、极低显存部署设计的

---

## Part II：方法与洞察

作者的方法可以拆成三个互相咬合的层：

1. **表示层**：用各向异性 3D Gaussians 表示场景  
2. **优化层**：交替进行参数优化与自适应增密  
3. **渲染层**：用可见性感知的 tile-based splatting 做实时渲染与反传

### 1）表示层：把“连续场”换成“可投影的显式体原语”

每个场景元素不再是 MLP 隐式密度，也不是规则网格体素，而是一个 **3D Gaussian**，包含：

- 位置（mean）
- 不透明度
- 协方差
- 颜色的球谐系数（SH）

关键点不在“高斯”三个字，而在于它同时满足两件事：

- **对优化友好**：它本质仍是体原语，可微；
- **对渲染友好**：它能投影成 2D 椭圆 splat，直接走 rasterization / alpha blending。

更重要的是，作者没有直接优化协方差矩阵，而是把它写成：

- **scale + rotation**
- 用缩放向量和四元数参数化

这样做的作用是：  
**保证协方差始终可解释、可优化，不会在梯度更新中变成非法矩阵。**

### 2）优化层：不是固定容量，而是边学边长

初始化时，作者直接用 SfM 给的稀疏点作为初始高斯中心。  
随后优化以下参数：

- 位置
- opacity
- scale / rotation（即各向异性形状）
- SH 外观参数

但仅靠固定的一批高斯不够，因为一开始点太稀。  
所以作者加入了 **adaptive density control**，核心是两种操作：

- **clone**：如果一个小高斯所在区域梯度大，说明这里可能缺几何，复制它去补
- **split**：如果一个大高斯覆盖范围过大且梯度大，说明它太粗糙，拆成两个更小高斯

背后的判断信号很简单但有效：  
**看 view-space position gradient 是否足够大。**

这等于把表示容量动态投到“当前重建最不准确的地方”。

此外还配合：

- 裁掉几乎透明的高斯
- 周期性重置 alpha，避免 floaters 越积越多
- 对过大的高斯做剔除

所以它不是“固定表示去拟合场景”，而是  
**让表示自己长成场景需要的形状和密度。**

### 3）渲染层：把 per-ray 采样改成 per-tile 排序的 splatting

这是性能跃迁的关键。

作者把屏幕切成 `16×16` 的 tiles，然后：

1. 将 3D Gaussian 投影到屏幕空间
2. 做 frustum / tile 级裁剪
3. 按覆盖的 tile 复制实例
4. 以 `tile ID + depth` 为 key 全局排序
5. 在每个 tile 内做前到后的 alpha blending

这带来两个重要变化：

- **排序从 per-pixel 变成 per-frame / per-tile 统一处理**
- **计算只发生在真正被原语覆盖的 tile 上**

而且它不是简单的 order-independent splatting。  
作者强调其 renderer 是 **visibility-aware** 的，也就是保留了按深度顺序累积透明度的过程，这使它更接近 NeRF 的成像逻辑。

更关键的一点是：  
**反向传播不限制可接收梯度的 splat 数量。**

这和一些更快但近似更强的方法不同。作者发现，如果只让前几个 splat 接受梯度，在深度复杂场景里会严重破坏学习。

### 核心直觉

这篇论文最强的地方，不是“换了个表示”，而是抓住了一个因果链：

**表示改变**  
→ 从“连续密度场 + 逐光线采样”变成“显式各向异性原语 + 按可见性排序的 splatting”  
→ **计算分布改变**：不再对空空间反复采样，而是只处理真正可能贡献像素的高斯  
→ **容量分配改变**：不再受规则网格限制，而是通过 clone/split 把表示容量投到高梯度区域  
→ **能力改变**：既保留可优化性，又拿到传统 GPU 栅格化的实时性

更直白地说：

- NeRF 的强项是“体渲染式可微成像”
- 点/栅格化的强项是“显式、高并行、实时”
- 这篇论文的核心动作就是：  
  **证明二者不必二选一**

为什么这能工作？

1. **按深度排序的 alpha blending 与体渲染的累积形式本质相通**  
   所以换掉 ray marching，并不等于放弃 NeRF 的成像优势。
2. **各向异性高斯比体素更会“贴表面”**  
   同样数量的原语可以更紧凑地表达薄结构、边界和斜面。
3. **增密是基于梯度驱动的**  
   所以新容量不是盲目加，而是加在“当前最解释不了图像误差”的地方。
4. **排序与 tile 化把可见性问题变成 GPU 友好的批量问题**  
   这才把训练与渲染都推到实时附近。

### 战略取舍

| 设计选择 | 改变了什么瓶颈 | 收益 | 代价/风险 |
|---|---|---|---|
| 各向异性 3D Gaussian 取代连续场/规则网格 | 从“每条光线遍历空间”改成“只处理显式占据原语” | 更高渲染效率，能紧凑拟合细结构 | 模型显存更大，可能产生拉长高斯伪影 |
| clone/split 自适应增密 | 从固定容量变成按误差动态分配容量 | 更快补足欠重建区域，背景/薄结构更稳 | 需要阈值、周期性裁剪与工程调参 |
| tile-based 全局排序 splatting | 从 per-pixel 排序/采样改成批量化可见性处理 | 训练和推理都大幅加速 | 排序是近似的，极端视角下会有 popping |
| SH 表达视角相关颜色而非 MLP/CNN | 去掉重型网络与 CNN 后处理 | 保留方向外观，避免 CNN 时序闪烁 | 视角覆盖不足时容易学坏，需要分阶段启用 |

---

## Part III：证据与局限

### 关键证据信号

**信号 1：质量-速度前沿被整体抬高了（comparison）**  
在 Mip-NeRF360 上：

- **Ours-7K**：6m25s 训练，25.60 PSNR，160 FPS
- **INGP-Big**：7m30s 训练，25.59 PSNR，9.43 FPS

这说明它不是单纯“更慢但更好”，而是把**相近质量下的渲染速度**直接拉高了一个量级以上。

**信号 2：不是只快，质量上限也足够高（comparison）**  
在 **Ours-30K** 下：

- Mip-NeRF360 平均：**27.21 PSNR / 0.815 SSIM / 0.214 LPIPS，134 FPS**
- 对比 Mip-NeRF360：**27.69 / 0.792 / 0.237，0.06 FPS**

解读上要谨慎：

- PSNR 不是全面碾压；
- 但 **SSIM 和 LPIPS 更优**，而且帧率从近乎不可交互直接变成实时。

这支持论文最核心的主张：  
**3DGS 不是“低质实时”，而是“接近顶级质量的实时”。**

**信号 3：关键机制有明确因果支撑（ablation）**  
消融很到位，尤其说明两点：

- **anisotropic covariance 必要**：去掉各向异性后，表面贴合能力下降，平均 PSNR 从 26.05 掉到 25.23；
- **不能截断可接收梯度的深度复杂度**：如果只让前 10 个 splats 接受梯度，平均 PSNR 直接跌到 19.19。

这说明性能提升不是单一 trick，而是表示、增密、可见性反传三者共同作用。

### 1-2 个最值得记住的指标

- **134 FPS @ Mip-NeRF360 近 SOTA 质量**
- **7K 迭代约 6 分钟即可达到 25.60 PSNR 与 160 FPS**

### 局限性

- **Fails when**: 欠观测区域、训练视角重叠很低的外推视角、强视角相关反射区域、或非常大的场景未调学习率时；此时会出现粗大/拉长高斯、splotchy artifact 和视角切换时的 popping。
- **Assumes**: 静态场景、SfM 标定相机与稀疏点初始化（真实场景尤其重要）、足够多的多视图覆盖、CUDA/GPU 训练环境；论文原型在大场景训练时峰值显存可超过 20GB，推理也需要数百 MB 模型内存外加 30–500MB 栅格器缓存。
- **Not designed for**: 动态场景、单/少视图重建、严格移动端低内存部署、完全精确的抗锯齿与无 popping 的生产级可见性处理。

### 资源/可复现性备注

- **优点**：代码和数据开放，复现门槛相对低于闭源系统。
- **现实约束**：
  - 结果依赖 GPU/CUDA 实现；
  - 论文中约 80% 训练时间仍耗在 Python 侧，说明当前实现还不是极致工程化版本；
  - 真实场景质量很依赖 SfM 初始化与视角覆盖。

### 可复用组件

这篇论文最值得迁移的，不只是“3DGS 整体方案”，而是几个可拆出的操作符：

1. **各向异性 3D Gaussian 原语**：适合需要显式、可微、可投影表示的 3D 任务  
2. **基于 view-space gradient 的 clone/split 增密**：可迁移到其他显式场表示  
3. **visibility-aware tile rasterizer**：适合高并发、可微 splatting 系统  
4. **分阶段 SH 优化**：当角度覆盖不完整时，能稳定外观学习

### So what

这篇工作真正的能力跃迁是：

- 之前的范式大多在 **“高质量但慢”** 和 **“较快但质量受限”** 之间做选择；
- 3DGS 通过重新设计原语和渲染管线，把这条 Pareto 前沿整体向外推了一步。

最有说服力的证据不是某一个单点 SOTA，而是：

- **跨多个标准数据集的质量-速度一致优势**
- **关键设计的明确消融支撑**
- **真实达到 100+ FPS 的部署级渲染速度**

## Local PDF reference

![[paperPDFs/Classic_Work_Dynamic_Rendering/arXiv_2023/2023_Gaussian_Splatting_for_Real_Time_Radiance_Field_Rendering.pdf]]