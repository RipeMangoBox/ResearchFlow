---
title: "Gaussian Frosting: Editable Complex Radiance Fields with Real-Time Rendering"
venue: arXiv
year: 2024
tags:
  - 3D_Gaussian_Splatting
  - task/novel-view-synthesis
  - task/3d-scene-editing
  - adaptive-thickness
  - mesh-based-representation
  - poisson-reconstruction
  - dataset/Shelly
  - dataset/NeRFSynthetic
  - dataset/Mip-NeRF360
  - opensource/promised
core_operator: "以SuGaR提取的基网格为骨架，在网格法向两侧构建自适应厚度的高斯壳层，并用棱柱单元参数化约束高斯随编辑稳定运动"
primary_logic: |
  多视角RGB图像与相机位姿 → 先训练无约束3DGS并经SuGaR正则化提取基网格，再结合规则化/无约束高斯沿法向估计局部壳层厚度并在层内重采样优化高斯 → 输出可实时渲染、可编辑与可动画的复杂辐射场
claims:
  - "Claim 1: 在 Shelly 数据集上，Frosting 取得 39.84 PSNR / 0.977 SSIM / 0.033 LPIPS，优于 3DGS 的 37.66 / 0.958 / 0.066 和 SuGaR 的 36.33 / 0.954 / 0.059 [evidence: comparison]"
  - "Claim 2: 在 Mip-NeRF 360 上，Frosting 在显式网格可编辑方法中平均指标最好（28.38 PSNR / 0.856 SSIM / 0.205 LPIPS），但仍低于无网格约束的 3DGS（28.69 / 0.870 / 0.182）[evidence: comparison]"
  - "Claim 3: 自动选择 Poisson 重建的 octree 深度可在 Shelly 上将平均三角形数从 939K 降到 203K，同时基本保持 PSNR 并改善感知质量（39.85/0.975/0.035 → 39.84/0.977/0.033）[evidence: ablation]"
related_work_position:
  extends: "SuGaR (Guédon & Lepetit 2023)"
  competes_with: "SuGaR (Guédon & Lepetit 2023); Adaptive Shells (Wang et al. 2023)"
  complementary_to: "N/A"
evidence_strength: strong
pdf_ref: "paperPDFs/Dynamic_Editing_Meshing/Lecture_Notes_in_Computer_Science_2025/2025_Gaussian_Frosting_Editable_Complex_Radiance_Fields_with_Real_Time_Rendering.pdf"
category: 3D_Gaussian_Splatting
---

# Gaussian Frosting: Editable Complex Radiance Fields with Real-Time Rendering

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2403.14554), [Project](https://anttwo.github.io/frosting/)
> - **Summary**: 该工作把可编辑基网格外包一层自适应厚度的 3D 高斯“糖霜”壳层，在保留 Blender 式编辑/动画能力的同时，恢复毛发、草、绒面等近表面体积细节。
> - **Key Performance**: Shelly 上达到 **39.84 PSNR / 0.977 SSIM / 0.033 LPIPS**；在 Mip-NeRF 360 上是**显式网格可编辑方法平均最佳**（28.38 / 0.856 / 0.205）。

> [!info] **Agent Summary**
> - **task_path**: 多视角RGB图像 + 相机位姿 -> 可实时渲染、可编辑/可动画的基网格+近表面高斯辐射场
> - **bottleneck**: 3DGS 一旦被压回单层表面以获得可编辑性，毛发/草等近表面体积效应就会明显丢失
> - **mechanism_delta**: 用“基网格 + 自适应厚度高斯壳层”替代“仅表面贴附高斯”，并用棱柱单元重参数化让高斯在编辑时始终被约束在壳层内
> - **evidence_signal**: [Shelly 上同时超过 3DGS 与 SuGaR, 厚度与 Poisson 深度消融直接解释性能来源]
> - **reusable_ops**: [双表示厚度估计, 棱柱单元重参数化]
> - **failure_modes**: [在 SfM 初始化极佳且场景多为规则表面时相对 3DGS 优势缩小, 壳层过厚或自交时动画会出现跨部件伪影]
> - **open_questions**: [如何用物理一致的形变模型替代分片线性更新, 如何压缩额外网格与重心参数带来的模型体积]

## Part I：问题与挑战

这篇工作解决的核心不是“再做一个更高 PSNR 的 3DGS”，而是更实际的问题：**怎样把高质量 radiance field 变成真正可编辑、可动画、可组合的 3D 资产，同时不牺牲复杂材质的渲染质量。**

- **输入 / 输出接口**  
  输入是多视角 RGB 图像与相机位姿；输出不是纯粹的体渲染场，也不是普通 mesh，而是：
  1. 一个可编辑的基网格；  
  2. 一个围绕网格、可实时 splat 的近表面高斯层。

- **真实瓶颈**  
  Vanilla 3DGS 的问题不是渲染不够好，而是**结构太弱**：高斯是无组织点集，3D 艺术家难以直接做 rigging、变形、重组。  
  SuGaR 把高斯贴到表面后，编辑问题基本解决，但新瓶颈出现了：**一旦把高斯压扁到表面，近表面体积自由度就被拿掉了**，于是毛发、草、绒毛边缘、柔软遮挡关系等都会受损。

- **为什么现在值得做**  
  3DGS 已把训练和实时渲染成本大幅降下来，所以“能不能渲染”已不再是主要矛盾；真正缺的是**连接 radiance field 与传统 CG/动画管线的表示层**。

- **边界条件**  
  Frosting 主要针对的是**贴近表面的复杂体积效应**，不是烟雾、云、火焰这类远离表面的自由体积介质，也不是面向动态场景重建的时序模型。

## Part II：方法与洞察

方法主线可以概括成一句话：**先从自由体积里抽出一个可编辑表面，再只在“确实需要体积表达”的近表面区域，把体积自由度加回来。**

### 方法主线

1. **先训练无约束 3DGS**  
   先不加表面对齐约束，让高斯自己找到合适位置。这样保留下来的，是最接近原始 3DGS 的离面分布信息，也是后续判断哪里“需要厚壳层”的依据。

2. **再做表面对齐，并抽取基网格**  
   使用 SuGaR 风格的正则，把高斯往表面拉齐，再从该表示中提取 mesh。  
   作者还补了一个很关键的工程点：**自动选择 Poisson reconstruction 的 octree 深度**，避免 SuGaR 默认固定深度导致的孔洞、椭球鼓包和过密三角形。

3. **用两套高斯联合估计 Frosting 厚度**  
   这是整篇最关键的因果旋钮。  
   - **规则化高斯**：更靠近表面，适合给出“可靠搜索区间”，减少 floaters 和过厚估计。  
   - **无约束高斯**：保留真实的 near-surface 体积分布，适合恢复“这个地方到底需要多厚”。  
   最终得到的是每个顶点沿法向的内外偏移：平坦区域壳层薄，毛发/草地/绒面区域壳层厚。

4. **在壳层内采样固定预算的高斯**  
   每个三角形的内外边界形成一个棱柱单元。作者在这些单元中重新初始化高斯：
   - 一半按单元体积采样，照顾 fuzzy 区域；
   - 一半均匀采样，避免平面大区域纹理不够。

5. **用棱柱单元重参数化，保证编辑稳定**  
   高斯中心不再是自由 3D 点，而是由棱柱单元六个顶点的重心坐标表示。  
   结果是：高斯始终被约束在壳层里，网格一旦被缩放、弯曲、骨骼驱动，高斯也能自动稳定跟随；其旋转、尺度和球谐参数再通过局部平均变换更新。

### 核心直觉

作者真正改变的，不是“高斯长什么样”，而是**高斯被允许出现在哪里**。

- **what changed**  
  从“高斯必须贴在表面上”改成“高斯只能存在于一个随网格运动、且局部厚度自适应的近表面体积带里”。

- **which distribution / constraint changed**  
  密度的空间自由度不再被完全剥夺，而是被限制在**最有信息价值的 near-surface 区域**。  
  这相当于把容量从“全空间自由漂浮”压缩成“局部受限体积”，又比“纯二维表面”保留更多表达能力。

- **what capability changed**  
  模型同时获得两种原本冲突的能力：
  1. 像 mesh 一样可编辑、可动画、可组合；  
  2. 像 volumetric 3DGS 一样保留毛发、草、绒面边界和近表面遮挡。

- **为什么这个设计有效**  
  规则化高斯回答的是“表面大概在哪里”；无约束高斯回答的是“离表面的有效体积范围有多大”。  
  两者联合，既能避免只看无约束高斯时的漂浮点和虚厚表面，也能避免只看规则化高斯时把 fuzzy 结构压没。

### 战略取舍

| 设计选择 | 改变了什么约束 | 带来的能力提升 | 代价/风险 |
|---|---|---|---|
| 基网格 + Frosting 壳层 | 从纯表面表示改成受限近表面体积表示 | 同时保留编辑性与 fuzzy 渲染 | 依赖可靠的 mesh 提取 |
| 自适应厚度 | 不再给所有区域相同的体积容量 | 毛发/草等处更厚，平面处更薄，减少无谓体积伪影 | 厚度估计错会影响动画稳定性 |
| 规则化 + 无约束联合估厚 | 用两套高斯分别负责“找表面”和“找厚度” | 更稳地过滤 floaters，同时保留真实离面结构 | 训练流程更复杂，需要中间状态 |
| 棱柱单元重参数化 | 自由高斯改成结构化单元内高斯 | 编辑、变形、缩放时高斯稳定跟随 | 形变本质仍是局部分片线性 |
| 固定预算采样替代自由 densification | 高斯数量与分布变得可控 | 预算可控，密度更聚焦困难区域 | 预算过小会损失纹理和细节 |

## Part III：证据与局限

### 关键证据

- **比较信号 1：在真正困难的 fuzzy 场景上，Frosting 不只是“可编辑且差不多”，而是显著更强。**  
  Shelly 上，Frosting 达到 **39.84 PSNR / 0.977 SSIM / 0.033 LPIPS**，不但超过 SuGaR，也超过 vanilla 3DGS。  
  这说明其收益不是单纯来自 mesh 编辑性，而是来自**更高效的 near-surface 高斯分配**。

- **比较信号 2：在标准场景上，它基本保住了 3DGS 水平，同时优于现有可编辑 mesh-based 方法。**  
  在 NeRFSynthetic 上，Frosting 超过 SuGaR，整体与 3DGS 接近。  
  在 Mip-NeRF 360 上，它是显式网格可编辑方法中的平均最佳，但略低于 vanilla 3DGS。这个结果很重要：**Frosting 的优势是有条件的**，主要体现在近表面复杂体积细节上，而不是所有场景无差别碾压。

- **消融信号：作者确实抓住了两个关键 causal knob。**  
  1. **自动 octree 深度**：减少网格伪影，并显著减少三角形数量；  
  2. **自适应厚度 + 双表示估厚**：优于固定厚度，也优于只用规则化高斯估厚。  
  论文中的动画例子还显示：固定大厚度会导致“右手的高斯参与了右膝附近重建”，一旦姿态变化就会出错。

- **效率信号：仍处在 3DGS 工程可用区间。**  
  完整 Frosting 训练约 **45–90 分钟 / 单张 V100**，相较 Adaptive Shells 报告的约 8 小时要更实用。

### 局限性

- **Fails when**: 场景主要由规则表面组成、且 SfM/COLMAP 初始化已经非常好时，vanilla 3DGS 本身就足够强，Frosting 的额外结构化约束不一定带来收益；如果壳层过厚或存在自交，不同部件附近的高斯在动画时可能出现跨部件串扰。
- **Assumes**: 依赖多视角 RGB 和相机位姿，真实场景通常还依赖 COLMAP 点云初始化；依赖先训练无约束 3DGS、再做 SuGaR 正则化和 Poisson 网格提取；当前形变更新是分片线性的局部平均变换；模型比 vanilla 3DGS 更大；论文阶段代码和 viewer 仅承诺开源。
- **Not designed for**: 远离表面的自由体积介质、强拓扑变化的动态场景、需要物理一致弹性/布料/接触的动画，以及极端轻量化部署场景。

### 可复用组件

- **双表示厚度估计**：让“规则化表示负责几何定位、无约束表示负责体积范围”这一思路，可迁移到其他 hybrid radiance field。
- **棱柱单元重参数化**：把体元素绑在 mesh 局部单元内，适合任何想把点/高斯表示接入传统动画管线的方法。
- **Poisson 深度自动选择**：根据高斯间距估计几何复杂度，再定网格重建分辨率，是很实用的 mesh extraction 工程技巧。
- **固定预算 near-surface densification**：比原始 3DGS 的自由 densification 更可控，适合资源预算明确的系统。

## Local PDF reference

![[paperPDFs/Dynamic_Editing_Meshing/Lecture_Notes_in_Computer_Science_2025/2025_Gaussian_Frosting_Editable_Complex_Radiance_Fields_with_Real_Time_Rendering.pdf]]