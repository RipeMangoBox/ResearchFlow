---
title: "Gaussian in the Wild: 3D Gaussian Splatting for Unconstrained Image Collections"
venue: arXiv
year: 2024
tags:
  - 3D_Gaussian_Splatting
  - task/novel-view-synthesis
  - adaptive-sampling
  - appearance-decomposition
  - visibility-map
  - dataset/PhotoTourism
  - dataset/NeRF-OSR
  - opensource/full
core_operator: 以3D Gaussian为显式载体，对每个点分离可学习的内在外观与由参考图像驱动的动态外观，并通过自适应采样与可见性图实现细粒度外观建模和抗瞬时遮挡训练。
primary_logic: |
  无约束图像集合 + 已知位姿 + 参考图像 → 用3D Gaussian显式表示场景，为每个点学习内在外观，并从投影特征图与K个特征图中自适应采样动态外观 → 融合后经轻量颜色解码器与栅格化渲染，输出可调外观的新视角图像
claims:
  - "Claim 1: 在 PhotoTourism 的 Brandenburg Gate、Sacre Coeur、Trevi Fountain 三个场景上，GS-W 的 PSNR/SSIM/LPIPS 均优于 3DGS、NeRF-W、Ha-NeRF 和 CR-NeRF [evidence: comparison]"
  - "Claim 2: 在单张 RTX 3090、800×800 分辨率下，GS-W 推理达到 38–58 FPS，缓存外观特征后达 197–301 FPS，相比 NeRF-W/Ha-NeRF/CR-NeRF 的约 0.045–0.052 FPS 快超过 1000× [evidence: comparison]"
  - "Claim 3: 去掉 K feature maps、adaptive sampling 或 intrinsic/dynamic separation 会降低重建质量，而去掉 visibility map 虽可能略升指标却会引入瞬时物体伪影 [evidence: ablation]"
related_work_position:
  extends: "3D Gaussian Splatting (Kerbl et al. 2023)"
  competes_with: "CR-NeRF (Yang et al. 2023); Ha-NeRF (Chen et al. 2022)"
  complementary_to: "Spec-Gaussian (Yang et al. 2024); Mip-Splatting (Yu et al. 2023)"
evidence_strength: moderate
pdf_ref: paperPDFs/In_the_Wild_Rendering_Video/Lecture_Notes_in_Computer_Science_2025/2025_Gaussian_in_the_Wild_3D_Gaussian_Splatting_for_Unconstrained_Image_Collections.pdf
category: 3D_Gaussian_Splatting
---

# Gaussian in the Wild: 3D Gaussian Splatting for Unconstrained Image Collections

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2403.15704), [Project](https://eastbeanzhang.github.io/GS-W/)
> - **Summary**: GS-W 把无约束图像中的外观变化从“整图一个隐变量”改成“每个 3D Gaussian 点的内在外观 + 动态外观”，从而在保留 3DGS 高速渲染的同时，更好地处理局部高光、阴影、天气变化和瞬时遮挡。
> - **Key Performance**: PhotoTourism 三个场景上全面优于 CR-NeRF 等基线；800×800 推理为 38–58 FPS，缓存外观后可达 197–301 FPS。

> [!info] **Agent Summary**
> - **task_path**: 无约束多视角图像集合 + 已知相机位姿 + 参考图像外观 -> 外观可调的新视角合成
> - **bottleneck**: 图像级全局外观编码无法表达点级局部高频变化，且把材质固有外观与环境动态影响混在一起
> - **mechanism_delta**: 用每点 intrinsic/dynamic appearance split 替代全局 appearance code，并引入多特征图自适应采样与 visibility-weighted supervision
> - **evidence_signal**: PhotoTourism 三场景全指标领先，且去掉 K feature maps / adaptive sampling / separation 后质量显著下降
> - **reusable_ops**: [per-point appearance split, adaptive feature-map sampling]
> - **failure_modes**: [complex lighting and specular reflections, frequently occluded textures]
> - **open_questions**: [how to remove the known-pose assumption for reference-image conditioning, how to learn a scene-generalized dynamic appearance extractor]

## Part I：问题与挑战

### 任务是什么
论文解决的是 **来自无约束互联网图像集合的 novel view synthesis**。  
输入不是理想静态采集数据，而是：

- 不同时间、天气、曝光设置下拍摄的图像；
- 图中常有行人、车辆等瞬时遮挡；
- 相机位姿已知；
- 渲染时再给一张参考图像，要求生成带有该参考图像外观风格的新视角图像。

输出则是：

- 一个可渲染的新视角场景表示；
- 同时支持一定程度的外观调节，而不是只恢复单一“平均外观”。

### 真正难点在哪里
作者认为，旧方法的核心问题不只是“外观变化大”，而是 **外观变化的粒度和归因方式错了**：

1. **全局外观 latent 太粗**
   - NeRF-W / Ha-NeRF 这类方法通常给每张图一个全局 appearance embedding。
   - 这能对齐整体色调，但很难表示局部高频现象，比如柱子上的高光、天空云层、局部阴影。

2. **固有外观与动态环境影响纠缠**
   - 真实世界里，一个点的外观 = 材质/纹理等 intrinsic 属性 + 光照/天气等 dynamic 影响。
   - 若把两者混成一个向量，模型很难知道“什么该长期保留、什么只是当前参考图带来的变化”。

3. **瞬时遮挡会污染 3D 表示**
   - 行人/车若直接参与监督，会让高斯点错误生长，形成 floating artifacts。

4. **NeRF 系方法速度太慢**
   - 无约束外观建模已经更复杂，如果还依赖体渲染中的大量网络评估，训练与推理都会变重。

### 为什么现在值得做
这件事在 3DGS 框架里更有机会做好，因为 3DGS 同时提供了两点：

- **显式点级载体**：每个 Gaussian 天然可以绑定自己的外观状态；
- **高速 rasterization**：让更细粒度的外观建模不必再支付 NeRF 那种巨大的体渲染成本。

### 边界条件
这篇方法虽然叫 “in the wild”，但并不是无限制场景建模。它默认：

- 底层场景几何 **大体静态**；
- **相机位姿已知**，且能由 SfM 初始化点云；
- 瞬时物体被视为训练噪声而非显式动态对象；
- 外观条件主要来自 **单张参考图像**。

## Part II：方法与洞察

### 方法主线
GS-W 可以理解为：**把 3DGS 的“固定颜色点”升级为“带可分离外观状态的点”**。

#### 1. 以 3DGS 为几何骨架
作者保留 3DGS 的核心优点：

- 用 Gaussian 点表示场景；
- 用 tile-based rasterizer 高速渲染；
- 继续使用高斯点的增殖/裁剪策略。

但它不再直接用球谐系数表示颜色，而是把颜色改成一个由外观特征解码得到的量。

#### 2. 给每个点注入 dynamic appearance
参考图像先经过一个 U-Net，输出：

- 1 个 **projection feature map**；
- K 个额外的 **feature maps**。

然后每个 Gaussian 点从两类特征图中取信息：

- **投影特征图**：按真实 3D 点投影到参考图上的 2D 位置采样，提供几何对齐的外观锚点；
- **K 个特征图**：每个点自带 K 个可学习采样坐标，在这些图上自由采样，补足投影采样无法表达的局部高频变化。

这一步很关键。因为只靠投影采样会有两个问题：

- 同一条射线上的点可能采到相同位置，信息区分度不够；
- 参考图没拍到的区域会缺少有效样本。

K feature maps + learnable sampling coordinates 本质上是在给每个点一个 **“不完全受几何投影束缚的局部检索空间”**。

#### 3. 显式分离 intrinsic 和 dynamic appearance
作者给每个 Gaussian 点再加一个可学习的 **intrinsic appearance feature**，表示材料、纹理等相对稳定的信息。

于是每个点最终有两部分外观：

- **intrinsic**：点自身长期属性；
- **dynamic**：参考图像提供的环境影响。

两者与点位置一起输入一个 fusion network，得到综合 appearance feature，再结合视角方向解码颜色。

#### 4. 用 visibility map 压低瞬时遮挡的监督权重
作者另外训练一个 2D visibility map，对图像损失做加权：

- 静态区域权重大；
- 瞬时物体区域权重小。

这不会真的“建模动态物体”，而是让模型在优化时 **尽量忽略它们**，避免错误高斯点被写入场景。

#### 5. 速度来自 3DGS，而不是大网络
外观增强后的渲染仍走 3DGS 的 rasterization 路线。  
而且当参考外观固定时，还可以缓存每个点的融合外观特征，只保留很小的颜色解码器参与渲染，因此速度接近原始 3DGS。

### 核心直觉
作者真正拧动的因果旋钮是：

**把“每张图一个全局 appearance code”改成“每个 3D 点各自拥有 intrinsic 外观，并从参考图中自适应检索 dynamic 外观”。**

这带来的变化链条是：

- **What changed**：从图像级外观控制，变成点级外观控制；
- **Which bottleneck changed**：外观信息瓶颈从“单个全局向量”变成“几何对齐 + 自适应局部采样”的高维条件特征；
- **What capability changed**：模型不再只会迁移整体色调，而能迁移更局部的高光、阴影、天空细节，并保持更好的多视角一致性。

为什么这设计有效：

1. **intrinsic/dynamic split** 让模型知道什么是“场景长期属性”，什么是“当前环境影响”；
2. **projection sampling** 保证外观条件不是完全漂浮的，而是与几何位置有物理对应；
3. **adaptive sampling** 进一步放松了“只能在投影点取样”的限制，让每个点自己找最有用的局部外观信号；
4. **visibility weighting** 改变了监督分布，减少 transient object 对几何和外观的污染。

### 战略取舍

| 设计 | 解决的瓶颈 | 带来的能力 | 代价/风险 |
|---|---|---|---|
| 每点 intrinsic / dynamic 分离 | 外观纠缠 | 支持更自然的外观调节，材质保持更稳 | 点属性更多，训练更复杂 |
| projection feature map | 外观缺少几何锚点 | 参考图像外观可对齐到 3D 点 | 同射线点仍可能信息相似，且受参考图覆盖限制 |
| K feature maps + adaptive sampling | 全局特征过粗、投影采样不足 | 可抓住局部高频变化和视野外补偿信息 | 需要额外特征提取器与采样坐标学习 |
| visibility map | 瞬时物体污染监督 | 更少 floating artifacts | 指标不一定总提升，视觉质量与数值可能不一致 |
| appearance caching | 外观解码开销 | 接近 3DGS 的高 FPS | 只在参考外观固定时最有效 |

## Part III：证据与局限

### 关键证据
- **Comparison signal**：在 PhotoTourism 的三个场景上，GS-W 在 PSNR / SSIM / LPIPS 全部超过 3DGS、NeRF-W、Ha-NeRF、CR-NeRF。  
  例如 Brandenburg Gate 上达到 **27.96 PSNR / 0.9319 SSIM / 0.0862 LPIPS**，优于 CR-NeRF 的 26.53 / 0.9003 / 0.1060。  
  这说明能力提升不是只体现在某一个指标上，而是同时体现在失真、结构和感知质量上。

- **Speed signal**：在 800×800、单张 RTX 3090 下，GS-W 达到 **38–58 FPS**；缓存外观后可达 **197–301 FPS**。而 NeRF-W / Ha-NeRF / CR-NeRF 约只有 **0.045–0.052 FPS**。  
  这支持论文最重要的 “so what”：**它不是只比 NeRF-based in-the-wild 方法更好，还把它们从离线渲染拉到了接近实时。**

- **Ablation signal**：  
  - 去掉 **K feature maps** 或 **adaptive sampling**，质量明显下滑，说明点级局部动态特征检索确实是核心；
  - 去掉 **intrinsic/dynamic separation** 后，LPIPS/SSIM 下降，说明显式拆分确实改善了感知质量与可控性；
  - 去掉 **visibility map** 时数值指标有时反而略高，但会出现更明显伪影，表明 benchmark metric 与视觉稳定性并不总一致。

- **Case-study signal**：appearance tuning 实验显示，随着 dynamic feature 权重增加，GS-W 会更自然地增强光照与高光，而不像 Ha-NeRF / CR-NeRF 那样出现奇怪的建筑或天空颜色。  
  这说明“分离 intrinsic 与 dynamic”不是概念包装，而是带来了更合理的控制行为。

- **Supplementary support**：补充材料中，GS-W 在 NeRF-OSR 与加扰动的 synthetic Lego 上也保持领先。  
  不过这些结果不在主文主表中，因此更适合作为支持性证据，而不是把 evidence_strength 提到更高等级的决定性依据。

### 能力跳变总结
相对 prior work，GS-W 的真正跃迁不是“把 NeRF-W 换成 3DGS”这么简单，而是：

- 从 **全局图像级 appearance conditioning**
- 变成 **点级、可分离、可局部检索的 appearance conditioning**
- 再借助 3DGS 的高速 rasterization，把这类条件建模落到接近实时推理。

### 局限性
- Fails when: 光照变化非常复杂、镜面反射显著、或纹理长期/频繁被遮挡时，方法仍难以稳定恢复真实外观；论文明确提到在频繁遮挡场景中纹理重建较弱。
- Assumes: 需要已知图像位姿与可用的 SfM 初始化点云；默认场景主体静态；动态外观来自参考图像特征提取；训练约 70k steps、单张 RTX 3090 约 2 小时，算力门槛不算高，但 **pose 质量** 仍是关键依赖。
- Not designed for: 显式动态场景建模、无位姿参考图下的物理一致外观迁移、精确 BRDF/反射分解式 relighting。

### 可复用组件
- **per-point intrinsic/dynamic appearance split**：适合迁移到其他 appearance-conditioned 3DGS/NeRF 系统；
- **adaptive multi-map sampling**：适合作为“从参考图向 3D 表示注入局部外观信息”的通用操作；
- **visibility-weighted supervision**：适合含瞬时遮挡的数据集，尤其是互联网图像集合；
- **appearance caching**：在参考条件固定的应用里，可显著提高渲染效率。

## Local PDF reference

![[paperPDFs/In_the_Wild_Rendering_Video/Lecture_Notes_in_Computer_Science_2025/2025_Gaussian_in_the_Wild_3D_Gaussian_Splatting_for_Unconstrained_Image_Collections.pdf]]