---
title: "Real-time Photorealistic Dynamic Scene Representation and Rendering with 4D Gaussian Splatting"
venue: ICLR
year: 2024
tags:
  - 3D_Gaussian_Splatting
  - task/novel-view-synthesis
  - gaussian-splatting
  - 4d-rotation
  - spherindrical-harmonics
  - "dataset/Plenoptic Video"
  - dataset/D-NeRF
  - opensource/partial
core_operator: 用可任意时空旋转的4D高斯直接拟合动态场景时空体，并在查询时刻切片为条件3D高斯进行实时splatting渲染
primary_logic: |
  多视角/单目视频与相机参数、时间戳 → 以带4D旋转和4D外观基的高斯集合近似场景时空体，并在给定时刻分解为时间边缘权重与条件3D高斯后投影/融合 → 输出任意时刻任意视角的高保真动态新视角图像
claims:
  - "在 Plenoptic Video 上，该方法达到 32.01 PSNR、0.014 DSSIM、0.055 LPIPS 和 114 FPS，并在文中列出的基线中同时取得最佳画质与最高渲染速度 [evidence: comparison]"
  - "在 D-NeRF 上，该方法达到 34.09 PSNR 与 0.02 LPIPS，优于文中报告的 V4D 与先前 4DGS 等方法 [evidence: comparison]"
  - "去掉 4D rotation、4D spherindrical harmonics 或时间维 densification 都会降低重建质量，说明时空耦合与时变外观建模对性能提升具有因果作用 [evidence: ablation]"
related_work_position:
  extends: "3D Gaussian Splatting (Kerbl et al. 2023)"
  competes_with: "HexPlane (Cao & Johnson, 2023); 4D Gaussian Splatting (Wu et al. 2023)"
  complementary_to: "N/A"
evidence_strength: strong
pdf_ref: paperPDFs/Dynamic_Video/arXiv_2023/2023_Real_time_Photorealistic_Dynamic_Scene_Representation_and_Rendering_with_4D_Gaussian_Splatting.pdf
category: 3D_Gaussian_Splatting
---

# Real-time Photorealistic Dynamic Scene Representation and Rendering with 4D Gaussian Splatting

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2310.10642), [Project](https://fudan-zvg.github.io/4d-gaussian-splatting)
> - **Summary**: 论文把动态场景直接看成一个连续的 4D 时空体，用可旋转的 4D Gaussian 而不是“3D 表示 + 显式形变场”来建模，从而实现任意时刻任意视角的实时高保真渲染。
> - **Key Performance**: Plenoptic Video 上 32.01 PSNR / 0.055 LPIPS / 114 FPS；D-NeRF 上 34.09 PSNR / 0.02 LPIPS。

> [!info] **Agent Summary**
> - **task_path**: 标定视频序列（多视角或单目）+ 时间索引 + 相机参数 -> 任意时刻任意视角的动态场景 RGB 渲染
> - **bottleneck**: 需要在时间上共享真实运动相关信息，但避免 6D 场函数对不相关时空位置的错误耦合；同时又不想显式追踪/形变建模带来的扩展性问题
> - **mechanism_delta**: 把动态内容编码进单个 4D 高斯的时空协方差与 4D 旋转里，再在渲染时对时间做条件切片，而不是学习独立帧或显式 deformation field
> - **evidence_signal**: 跨多视角真实场景与单目合成场景的基准领先，并有去掉 4D rotation / 4DSH / 时间 split 的消融支撑
> - **reusable_ops**: [4D时空高斯原语, 时间条件化Gaussian splatting]
> - **failure_modes**: [远处背景缺少初始化点云时重建不稳, 首帧未覆盖或后续才显露的区域恢复依赖densification且几何可能不准确]
> - **open_questions**: [如何降低对COLMAP首帧点云或场景边界初始化的依赖, 如何在更长视频和更剧烈拓扑变化下保持稳定高效]

## Part I：问题与挑战

这篇论文解决的是 **动态新视角合成（dynamic novel view synthesis）** 的核心矛盾：  
同一个表示既要跨时间共享信息，又不能把本来无关的时空位置硬绑定在一起。

- **输入**：带时间戳的视频序列、相机内外参；既支持多视角真实视频，也支持单目视频。
- **输出**：给定任意时间与任意相机位姿，渲染对应的高保真图像/视频帧。
- **边界条件**：主要验证的是**训练时间范围内**的时间查询与新视角合成，不是强时间外推任务。

作者认为现有方法有两条主路线，但各有硬伤：

1. **直接学习 6D plenoptic function**  
   如 MLP / grid / low-rank plane 类方法。问题在于：  
   - 参数共享方式由网络结构决定，不一定和真实运动对齐；
   - 要么过度共享，造成跨时刻互相干扰；
   - 要么共享不足，无法利用物体运动带来的时序相关性。

2. **显式形变/运动建模**  
   如 canonical field、deformation field、persistent Gaussian。问题在于：  
   - 需要维护较复杂的跟踪或拓扑假设；
   - 对复杂真实场景、遮挡、可见性变化的扩展性较差；
   - 很难同时保持简单、端到端和实时。

**为什么现在值得做？**  
因为 3D Gaussian Splatting 已经证明：显式 Gaussian primitive + rasterization 可以在静态场景里同时拿到高画质和实时速度。动态场景缺的不是“更大网络”，而是一个**真正匹配时空结构**的 primitive。

## Part II：方法与洞察

作者的核心做法是：**不用“静态 3D 高斯 + 形变场”解释动态，而是直接用 4D 高斯去近似整个时空体积**。

### 核心直觉

把“运动”理解成时空中的一段 **倾斜的 4D 局部体元**，而不是“某个 3D 点被显式拖着走”。

这带来三个因果变化：

1. **表示层改变**：  
   从“空间和时间分开建模”变成“空间-时间联合协方差建模”。  
   关键不是多加一个时间标量，而是允许协方差里出现 **space-time coupling**。

2. **信息瓶颈改变**：  
   过去共享信息靠网络参数共享或显式轨迹；现在共享发生在 **4D 高斯沿运动方向的局部支撑** 上。  
   于是，模型更容易只在“运动一致”的时空位置之间共享信息。

3. **能力改变**：  
   一个 Gaussian 可以跨多个时间片复用；静态背景可有大时间方差，动态前景可沿运动方向倾斜展开。  
   结果是：**既能建模复杂动态，又能保留 3DGS 的实时渲染优势**。

### 方法主线

#### 1. 4D Gaussian 原语
每个 primitive 定义在 \((x,y,z,t)\) 上，均值是 4 维，协方差也是 4 维。  
关键设计是：

- 用 **4D scaling + 4D rotation** 参数化协方差；
- 4D rotation 用一对 quaternion 组合实现；
- 因此高斯不再只是“在空间里的一团”，而是可以在时空里倾斜、拉伸，贴合潜在运动流形。

这比简单的“3D 高斯 + 1D 时间权重”更强，因为后者默认空间与时间独立，无法自然表示运动。

#### 2. 时间条件化渲染
查询某个时间 \(t\) 时，作者把每个 4D Gaussian 分成两部分理解：

- **时间边缘项**：决定它在当前时刻是否活跃；
- **条件 3D Gaussian**：表示它在该时刻对应的空间切片。

然后再像 3DGS 一样：

- 把条件 3D Gaussian 投影到图像平面；
- 做 tile-based splatting 和透明度融合。

所以渲染逻辑可以概括为：  
**先按时间筛选，再按空间投影，再做高效 rasterization**。

#### 3. 4D Spherindrical Harmonics（4DSH）
动态场景里颜色不仅随视角变，还会随时间演化。  
如果直接为不同时刻复制高斯，会导致冗余和优化困难。

作者因此把外观建模成：

- 视角方向上的 SH；
- 时间维上的 Fourier basis；

二者组合成 4DSH。  
这样，一个 primitive 就能同时表达 **view-dependent** 和 **time-evolving** 的外观。

#### 4. 时空 densification
如果还沿用 3DGS 只看空间梯度的 densification，会在动态区域出现闪烁和时间模糊。  
所以作者把时间维的梯度也纳入密化依据，并允许 **在时间维上 split**。

### 设计取舍

| 设计 | 改变了什么约束 | 带来的能力 | 代价/风险 |
|---|---|---|---|
| 全 4D 协方差 + 4D rotation | 不再假设空间与时间独立 | 能沿运动方向共享信息，隐式编码粗运动 | 参数更复杂，优化更依赖初始化 |
| 时间条件化切片渲染 | 从逐帧独立表示改为共享 4D primitive | 同一组原语可覆盖整个视频，渲染仍高效 | 时间范围外泛化未被充分验证 |
| 4DSH | 不再复制几何去适配时变外观 | 更好处理反射、光照变化、非Lambertian 动态外观 | 系数更多，长程复杂外观仍可能不足 |
| 时空 densification | 不只在空间中增密 | 减少动态区域抖动与模糊 | Gaussian 数量控制更敏感 |

## Part III：证据与局限

### 关键证据

- **对比信号（多视角真实场景）**  
  在 Plenoptic Video 上，4DGS 达到 **32.01 PSNR / 0.055 LPIPS / 114 FPS**。  
  这里真正的能力跃迁不只是“再涨一点 PSNR”，而是**同时**拿到更高画质和远超多数 NeRF 类方法的速度。  
  例如相对 HexPlane 的 0.563 FPS、HyperReel 的 2 FPS、先前 4DGS 的 36 FPS，它首次把高质量动态新视角合成推进到强实时区间。

- **对比信号（单目动态场景）**  
  在 D-NeRF 上达到 **34.09 PSNR / 0.02 LPIPS**，优于 V4D 与先前 4DGS。  
  这很重要，因为单目设置几乎没有同时刻多视角冗余，说明模型确实学到了**跨时间的信息交换**，而不只是利用多视角几何。

- **因果信号（消融）**  
  去掉 4D rotation（No-4DRot）后性能明显下降，说明“时空相关协方差”不是装饰，而是运动建模的关键旋钮。  
  去掉 4DSH 或时间维 split 也都会掉点，说明**时变外观**和**时空 densification**都是有效组件。

- **行为信号（分析）**  
  作者把 4D Gaussian 的条件均值轨迹投影成光流，可看到在没有显式运动监督的情况下，模型仍能涌现出粗粒度动态结构。  
  这支持论文的核心叙事：4D primitive 本身就在吸收运动信息。

### 局限性

- Fails when: 远处静态背景或后续才显露的区域在初始化点云中缺失时，模型较难恢复正确几何；论文附录也明确承认这类区域常被学成近似“背景贴图球”，可看但未必几何正确。
- Assumes: 需要标定相机；多视角真实场景通常依赖首帧 COLMAP 点云初始化，单目场景依赖已知体积范围内的随机点初始化；高效复现依赖 3DGS 风格的 GPU rasterizer 和良好的 density control。
- Not designed for: 无标定 in-the-wild 视频、精确点级跟踪/长期 correspondence、训练时间范围外的强时间外推，以及需要严格物理可解释形变的场景。

### 可复用组件

- **4D Gaussian primitive**：适合所有“空间 + 时间”联合建模的问题。
- **时间条件化 splatting**：可迁移到其他时空显式表示。
- **4DSH**：适合处理 view-dependent 且 time-evolving 的外观。
- **时空 densification 规则**：对其他动态 Gaussian 方法也有直接参考价值。

## Local PDF reference

![[paperPDFs/Dynamic_Video/arXiv_2023/2023_Real_time_Photorealistic_Dynamic_Scene_Representation_and_Rendering_with_4D_Gaussian_Splatting.pdf]]