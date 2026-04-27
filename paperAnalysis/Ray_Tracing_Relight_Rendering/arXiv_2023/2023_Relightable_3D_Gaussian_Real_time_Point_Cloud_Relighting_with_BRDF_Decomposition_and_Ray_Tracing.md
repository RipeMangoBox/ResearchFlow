---
title: "Relightable 3D Gaussians: Realistic Point Cloud Relighting with BRDF Decomposition and Ray Tracing"
venue: arXiv
year: 2023
tags:
  - 3D_Gaussian_Splatting
  - task/relighting
  - task/novel-view-synthesis
  - ray-tracing
  - brdf-decomposition
  - bounding-volume-hierarchy
  - dataset/NeRF-Synthetic
  - dataset/Synthetic4Relight
  - dataset/Mip-NeRF-360
  - opensource/no
core_operator: 为每个3D Gaussian显式附加法线、BRDF与入射光，并用BVH点式光线追踪预计算可见性，从而在点云上完成可重光照的PBR渲染。
primary_logic: |
  多视角图像（可带mask）→ 先以3DGS为骨架并用法线/深度分布约束稳定几何，再把材质分解为逐点BRDF、把光照分解为全局环境光+逐点间接光，并通过BVH点式光追预计算可见性 → 输出可编辑、可重光照、可做新视角渲染的3D Gaussian场景
claims:
  - "在Synthetic4Relight上，R3DG以0.90小时训练时间达到36.80 PSNR/0.982 SSIM/0.028 LPIPS的novel view synthesis，以及31.00 PSNR/0.964 SSIM/0.050 LPIPS的relighting，优于文中对比的逆渲染基线 [evidence: comparison]"
  - "在NeRF Synthetic上，R3DG取得31.22 PSNR、0.959 SSIM、0.039 LPIPS，显著优于NeILF++、Nvdiffrec、PhySG等可重光照方法，但低于不具备重光照能力的原始3DGS [evidence: comparison]"
  - "消融实验表明，法线梯度增密可改善薄结构法线，完整的“全局环境光+逐点间接光”建模对阴影生成是必要的，而深度分布约束可降低深度不确定性并提升AO/可见性质量 [evidence: ablation]"
related_work_position:
  extends: "3D Gaussian Splatting (Kerbl et al. 2023)"
  competes_with: "InvRender (Zhang et al. 2022); TensoIR (Jin et al. 2023)"
  complementary_to: "SuGaR (Guédon and Lepetit 2023)"
evidence_strength: strong
pdf_ref: paperPDFs/Ray_Tracing_Relight_Rendering/arXiv_2023/2023_Relightable_3D_Gaussian_Real_time_Point_Cloud_Relighting_with_BRDF_Decomposition_and_Ray_Tr acing.pdf
category: 3D_Gaussian_Splatting
---

# Relightable 3D Gaussians: Realistic Point Cloud Relighting with BRDF Decomposition and Ray Tracing

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2311.16043)
> - **Summary**: 本文把3DGS从“只能做新视角合成的显式点表示”扩展成“带法线、BRDF和可见性的可重光照点图元”，并通过BVH点式光追补上阴影与遮挡，从而实现可编辑的高保真重光照。
> - **Key Performance**: Synthetic4Relight 上 NVS 达到 **36.80 PSNR / 0.982 SSIM / 0.028 LPIPS**，relighting 达到 **31.00 PSNR / 0.964 SSIM / 0.050 LPIPS**；训练时间 **0.90h**。

> [!info] **Agent Summary**
> - **task_path**: 多视角RGB图像（静态场景） -> 可重光照3D Gaussian表示 -> 新视角PBR渲染/重光照图像
> - **bottleneck**: vanilla 3DGS缺少法线、材质与光照分解，更缺少点表示下可扩展的可见性/阴影建模，因此只能“看起来像”，不能“在新光照下成立”
> - **mechanism_delta**: 把PBR从“像素射线-表面交点上的隐式查询”改成“逐Gaussian的显式BRDF与入射光计算”，再用BVH光追把可见性离线缓存
> - **evidence_signal**: Synthetic4Relight 上同时提升NVS与relighting指标，且消融明确验证了法线增密、完整光照分解和深度分布约束的必要性
> - **reusable_ops**: [逐Gaussian BRDF分解, BVH可见性预计算]
> - **failure_modes**: [大规模高密度场景下逐点PBR与射线采样成本上升, 动态场景或频繁几何变化时预计算可见性失效]
> - **open_questions**: [如何扩展到大场景且维持效率, 如何进一步提升几何精度以减少材质-光照歧义]

## Part I：问题与挑战

这篇论文要解决的，不是“再做一个更快的3DGS”，而是把 **3DGS变成真正可重光照的图形表示**。

### 真正的问题是什么
vanilla 3DGS 的颜色本质上仍是外观拟合：它能把训练光照下的图像复现得很好，但没有显式的：
- 表面法线；
- 材质参数（如 albedo / roughness）；
- 入射光与可见性。

因此一旦换光照，原始3DGS并不知道“为什么这个像素亮”，也就无法稳定地产生真实阴影、镜面反射和遮挡关系。

### 真正的瓶颈在哪里
核心瓶颈有两个：

1. **表示瓶颈**  
   3DGS 是“软”的 Gaussian 点，而不是显式曲面。  
   这导致 inverse rendering 里常见的“在表面点上估材质和光照”范式，直接搬过来会非常别扭。

2. **可见性瓶颈**  
   阴影和真实重光照依赖 visibility，但点表示下做 ray tracing 并不天然容易，尤其 Gaussian 还是半透明、带体积影响的 primitive。

### 输入 / 输出接口
- **输入**：多视角图像，遵循3DGS重建设定；若有对象 mask，可进一步约束表面不透明性。
- **输出**：一个可重光照的 Gaussian 场景，每个点带有几何、法线、BRDF、间接光属性，外加全局环境光。

### 边界条件
方法主要面向：
- **静态场景/静态物体**
- **离线优化 + 在线渲染**
- **中小规模到中等复杂度场景**

它不是为动态场景、超大规模开放世界或严格物理精确的全局光传输而设计的。

## Part II：方法与洞察

### 核心直觉

作者最关键的改动，是**把“物理属性的承载位置”从隐式表面/像素交点，改到了 Gaussian primitive 本身**。

这会带来三个因果变化：

1. **表示层变化**  
   从“每个Gaussian只负责外观颜色”  
   变成“每个Gaussian负责局部几何 + 材质 + 局部入射光”。

2. **信息瓶颈变化**  
   从“每次渲染都要在线求表面交点、查询光照/可见性”  
   变成“许多量可以逐点缓存，尤其 visibility 可预计算”。

3. **能力变化**  
   从“只能在原光照下做新视角合成”  
   变成“能在新光照下做带阴影的PBR重光照，而且仍保持点表示的可编辑性”。

换句话说，这篇工作不是单纯给3DGS加一个 loss，而是把 3DGS 从 **appearance renderer** 推向 **editable inverse-rendering primitive**。

### 方法拆解

#### 1. 几何增强：让 Gaussian 更像“表面”
作者给每个 Gaussian 新增法线，并通过渲染出的深度/法线一致性来优化它，而不是依赖传统点云局部平面法线估计。

配套还有两类几何稳定器：
- **法线梯度驱动的增密**：在薄结构和几何细节处补点；
- **深度分布约束**：压缩沿射线的深度不确定性，把 Gaussian 往表面“收紧”。

这一步的作用不是追求最精确几何，而是让后续 PBR 所依赖的法线与可见性更可信。

#### 2. 材质与光照分解：把颜色拆成“能换灯”的因素
每个 Gaussian 额外携带：
- **BRDF参数**：albedo + roughness；
- **间接光**：逐点的低阶 SH 表示；
- **全局直射光**：共享环境图。

这里最重要的设计不是 BRDF 本身，而是**光照拆分方式**：
- **全局环境光** 负责场景一致的 direct illumination；
- **逐点间接光** 负责局部 inter-reflection 残差。

这个拆法缓解了“每个点都自己学一套光照”带来的过拟合和材质-光照歧义。

#### 3. 点式光线追踪：补上 visibility
作者提出了基于 **BVH** 的 point-based ray tracing：
- BVH 管理 Gaussian 的包围盒；
- 射线与 Gaussian 的交点不做严格几何求交，而是用“Gaussian 对射线贡献峰值的位置”近似；
- 累积透射率，得到某个方向上是否可见。

这让 Gaussian 也能参与 shadow / occlusion 的计算。

#### 4. 两阶段优化：先几何，后材质
作者没有把所有变量一锅端联合优化，而是分成两阶段：
- **Stage 1**：先优化几何与法线；
- **Precompute**：基于当前几何做 visibility 预计算；
- **Stage 2**：冻结几何，只优化材质和光照。

这样做的本质是减少优化耦合，避免“几何错一点、材质补一点、光照再补一点”的塌陷解。

### 策略取舍

| 设计选择 | 解决的瓶颈 | 带来的能力 | 代价/风险 |
| --- | --- | --- | --- |
| 逐Gaussian做PBR，而不是逐像素表面交点做PBR | 点表示下表面难稳定提取，像素级查询过贵 | 更高效、可缓存、适合显式编辑 | 物理精度受软Gaussian近似影响 |
| 全局环境光 + 逐点间接光 | 材质-光照高度耦合 | 兼顾全局一致性与局部补偿 | 高频复杂光传输表达有限 |
| BVH预计算visibility | 在线光追太慢 | 阴影/遮挡可用于实时渲染式流程 | 静态场景友好，几何改动后需重算 |
| 两阶段优化 | 联合优化不稳定 | 减少几何误差对BRDF分解的污染 | 几何冻结后，后续材质信号不能再反向修几何 |

## Part III：证据与局限

### 关键证据

- **对比信号 1：NeRF Synthetic 上的“能力增量成本”**
  - R3DG 在可重光照方法中明显更强，说明它没有因为引入BRDF分解而严重牺牲基本的新视角质量。
  - 同时它仍低于 vanilla 3DGS，说明“可重光照”不是白拿的，确实存在表示与优化成本。

- **对比信号 2：Synthetic4Relight 上的主结果**
  - R3DG 在文中拿到最好的 NVS 与 relighting 指标：**36.80 PSNR** 和 **31.00 PSNR**。
  - 训练时间 **0.90h**，也明显短于 TensoIR（3.24h）与 InvRender（14.3h）。
  - 这说明它不仅会“分解”，还把点表示带来的效率优势真正转成了可比较的结果。

- **消融信号：每个关键模块都在起作用**
  - 去掉法线梯度增密，薄结构法线变差；
  - 只用简化光照而不用“环境光 + 逐点间接光”，重光照难以形成可信阴影；
  - 去掉深度分布约束，深度不确定性上升，AO/visibility 也更差。

- **证据上的一个保留点**
  - 题目强调 **real-time**，但正文没有给出统一的 FPS benchmark。
  - 因此“实时性”更主要来自：3DGS rasterization + visibility 预计算 + 显式点表示这一系统设计，而不是严格速度榜单。

### 局限性

- **Fails when**: 场景非常大、点非常密时，逐点PBR和多方向采样会显著拉高优化与预计算成本；动态场景或频繁几何变化会让预计算的 visibility 失效。
- **Assumes**: 静态场景；几何可先被3DGS较稳定恢复；第二阶段冻结几何；直接光用低分辨率环境图、间接光用低阶SH；复现依赖3DGS/BVH的GPU实现。
- **Not designed for**: 大规模开放场景的高频实时重光照、严格mesh级精确交点的物理光追、时变场景或需要完整多次全局光照求解的任务。

### 可复用组件

- **逐Gaussian物理属性扩展**：给显式点/体图元挂法线、BRDF、局部光照属性。
- **深度分布不确定性约束**：把“软点”往表面收缩，适合其他显式体元表示。
- **BVH上的点式visibility预计算**：适合静态显式场景的阴影/遮挡缓存。
- **两阶段 inverse rendering 配方**：先稳几何，再做材质/光照分解。

## Local PDF reference

![[paperPDFs/Ray_Tracing_Relight_Rendering/arXiv_2023/2023_Relightable_3D_Gaussian_Real_time_Point_Cloud_Relighting_with_BRDF_Decomposition_and_Ray_Tracing.pdf]]