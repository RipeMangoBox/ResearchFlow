---
title: "DreamGaussian: Generative Gaussian Splatting for Efficient 3D Content Creation"
venue: ICLR
year: 2024
tags:
  - 3D_Gaussian_Splatting
  - task/image-to-3d
  - task/text-to-3d
  - gaussian-splatting
  - diffusion
  - mesh-extraction
  - dataset/Objaverse
  - opensource/no
core_operator: 用渐进增密的3D Gaussian替代NeRF完成SDS粗生成，再通过局部密度查询提取网格并在UV空间做扩散编辑式纹理细化
primary_logic: |
  单张参考图像或文本提示 → 以SDS优化逐步增密的3D Gaussian快速获得粗几何与外观 → 通过局部密度查询与颜色回投提取纹理网格 → 在UV空间以扩散去噪生成细化目标并用MSE优化 → 输出可直接使用的高质量纹理网格
claims:
  - "在 image-to-3D 上，DreamGaussian 约 2 分钟即可生成纹理网格，相比 Zero-1-to-3 的约 20 分钟实现约 10× 加速，同时 CLIP-Sim 更高（0.738 vs 0.647）[evidence: comparison]"
  - "两阶段设计能显著提升最终质量：仅 Stage 1 的 CLIP-Sim 为 0.678，而加入网格提取与 UV 细化后提升到 0.738 [evidence: comparison]"
  - "去掉周期性 densification、timestep annealing 或参考视图损失，都会导致更模糊且更不准确的生成结果 [evidence: ablation]"
related_work_position:
  extends: "3D Gaussian Splatting (Kerbl et al. 2023)"
  competes_with: "Zero-1-to-3 (Liu et al. 2023b); DreamFusion (Poole et al. 2022)"
  complementary_to: "MVDream (Shi et al. 2023); SyncDreamer (Liu et al. 2023c)"
evidence_strength: moderate
pdf_ref: paperPDFs/Diffusion_Video/arXiv_2023/2023_DreamGaussian_Generative_Gaussian_Splatting_for_Efficient_3D_Content_Creation.pdf
category: 3D_Gaussian_Splatting
---

# DreamGaussian: Generative Gaussian Splatting for Efficient 3D Content Creation

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2309.16653) · [Project](https://dreamgaussian.github.io/)
> - **Summary**: 这篇工作把优化式 2D-to-3D 管线中的 3D 表示从 NeRF 换成渐进增密的 3D Gaussian，并把高频细节恢复转移到 UV 纹理细化，从而把单样本 3D 生成压缩到分钟级。
> - **Key Performance**: 单图到高质量纹理网格约 2 分钟；image-to-3D 上 CLIP-Sim 0.738（2 min）vs Zero-1-to-3 的 0.647（20 min）

> [!info] **Agent Summary**
> - **task_path**: 单张图像/文本提示 -> 生成式 3D 表示优化 -> 可导出纹理网格
> - **bottleneck**: SDS 监督下 NeRF 渲染昂贵且空域裁剪不稳定，导致每个样本优化极慢
> - **mechanism_delta**: 用高效 rasterization 的 3D Gaussian 做粗几何/外观搜索，再用扩散编辑产生的图像级伪目标对 UV 纹理做确定性 MSE 细化
> - **evidence_signal**: image-to-3D 上 2 分钟达到 0.738 CLIP-Sim，显著快于 Zero-1-to-3 的 20 分钟/0.647，且有用户研究与消融支撑
> - **reusable_ops**: [progressive Gaussian densification, block-wise local density query, diffusion-edit-to-MSE texture refinement]
> - **failure_modes**: [single-view back-view blur, text-to-3D Janus/multi-face]
> - **open_questions**: [能否在不牺牲速度下用多视角条件扩散进一步消除 Janus, Gaussian-to-mesh 转换如何更稳地保留细薄几何]

## Part I：问题与挑战

这篇论文要解决的硬问题，不是“能不能从 2D 先验做 3D”，而是“怎样把 **优化式** 3D 生成做得足够快，快到能真正用于内容生产”。

### 真实瓶颈是什么
优化式 2D lifting 方法（如 DreamFusion、Zero-1-to-3 系）已经证明：只要有强 2D diffusion prior，的确可以通过 SDS 把 3D 形状和外观蒸馏出来。  
但它们常见地要跑几十分钟到几小时，核心卡点有两个：

1. **NeRF 的内环太贵**  
   NeRF 需要体渲染，前向和反向都重，单个样本优化成本高。

2. **SDS 是“模糊监督”，让 NeRF 的加速手段失灵**  
   在重建任务里，occupancy pruning 有明确依据：哪里是空、哪里不是空，监督较稳定。  
   但在生成任务里，SDS 跨视角给出的梯度并不总是一致，空域裁剪和容量分配会变得不可靠。作者的判断是：**不是所有 NeRF 加速技巧都能无缝迁移到生成式设置**。

### 输入/输出接口
- **image-to-3D**：输入单张参考图像 + 前景 mask，输出带纹理的 polygon mesh。
- **text-to-3D**：输入文本提示，输出带纹理的 polygon mesh。

### 边界条件
这不是任意场景生成，而是偏 **object-centric 单物体资产生成**：
- 相机围绕物体中心采样；
- 输入图像通常需要前景分割和居中；
- 主要建模的是 **diffuse color**，不是可重光照材质；
- 目标输出是可下游使用的 mesh/UV 贴图，而不是仅供渲染的隐式场。

### 为什么现在值得做
因为现实里，开放词汇的高质量 3D 原生数据仍然稀缺，纯 3D-native 生成模型虽然快，但训练数据和泛化都受限。  
所以当前更现实的路线仍是：**借助成熟的 2D diffusion prior 做 lifting**。DreamGaussian 的价值就在于，它不是再去增强 SDS 本身，而是改造 **表示 + 优化流程**，把这条路线从“能做 demo”推进到“接近可部署”。

## Part II：方法与洞察

### 管线概览

DreamGaussian 是一个两阶段框架，但中间插入了关键的 Gaussian-to-mesh 转换。

#### 1. Stage 1：Generative Gaussian Splatting
作者把 3D 表示从 NeRF 换成 **3D Gaussian Splatting**。

- 用一组 3D Gaussians 表示空间；
- 初始只放较少高斯球，训练过程中 **周期性 densification**；
- 对随机视角渲染图像，用 SDS 从 2D diffusion prior 反传到 3D 参数；
- image-to-3D 额外加入参考视图的 RGBA 对齐损失；
- 训练时做 **timestep annealing**，让优化从粗到细推进。

对 image-to-3D，2D prior 是 Zero-1-to-3 XL；  
对 text-to-3D，2D prior 是 Stable Diffusion。

这里的目标不是一步到位做出最终高频细节，而是先在很短时间内得到 **足够靠谱的粗几何和粗外观**。

#### 2. Gaussian 到 Mesh：高效网格提取
直接用 3D Gaussians 做最终资产有两个问题：
- 下游工业流程更需要 mesh；
- SDS 下直接得到的 Gaussians 往往偏模糊。

所以作者设计了一个 **局部密度查询** 的 mesh extraction：

- 把空间切成重叠 block；
- 对每个 block 只查询局部相关的高斯；
- 在局部网格上估计密度，拼成全局 dense grid；
- 用 Marching Cubes 提取表面；
- 再做 remeshing / decimation 后处理。

之后再通过多视角渲染做 **color back-projection**，把颜色烘到 UV 贴图上，得到一个可细化的初始纹理。

#### 3. UV-space Texture Refinement
作者发现：如果继续在 UV 纹理上直接施加 SDS，会出现明显的 **块状、过饱和伪影**。  
原因是 differentiable rasterization 中的 mipmap 采样，会把模糊的 SDS 梯度扩散成不稳定的颜色块。

所以他们改成另一种监督方式：

- 先从任意视角渲染当前的粗图像；
- 加一点受控噪声；
- 让 2D diffusion prior 做多步去噪，产生一个 **更清晰但内容一致** 的“细化图像”；
- 再用这个细化图像，对当前渲染做 **像素级 MSE** 回归。

这一步本质上把“模糊的 score 监督”改成了“更确定的图像监督”。

### 核心直觉

DreamGaussian 最关键的洞察是：

> 把 3D 生成拆成两个不同难度的问题：  
> **低频 3D 结构搜索** 用 3D Gaussian 做；  
> **高频 2D 纹理修补** 用图像空间的扩散编辑 + MSE 做。

#### what changed → which bottleneck changed → what capability changed

1. **NeRF → 3D Gaussian**
   - **改变了什么**：从体渲染的连续场，换成显式、局部支持、可高效 rasterize 的 Gaussian 集合。
   - **改变了哪个瓶颈**：降低了每步渲染/反传成本，也不再依赖 SDS 下不稳定的 occupancy pruning。
   - **带来什么能力变化**：粗几何能在数秒到数十秒出现，约 500 步就能收敛到可用结果。

2. **直接 UV-SDS → 扩散编辑生成伪目标 + MSE**
   - **改变了什么**：把纹理优化从“含糊且跨尺度不稳定的 score 引导”换成“确定的像素对齐目标”。
   - **改变了哪个瓶颈**：缓解了 mipmap + SDS 组合导致的块状伪影。
   - **带来什么能力变化**：第二阶段只需几十步就能显著增强纹理清晰度。

3. **Gaussian 表示 → 显式 mesh/UV**
   - **改变了什么**：输出不再只是中间表示，而是下游可用资产。
   - **改变了哪个瓶颈**：解决了“生成能看，但难用”的问题。
   - **带来什么能力变化**：可直接导出到 Blender 等流程中做动画和资产处理。

### 为什么这个设计有效
因为 SDS 的信息特点就是：  
**低频结构可信，高频细节和局部一致性不可信。**

所以更匹配的策略不是强迫一个 3D 表示从头学完所有东西，而是：
- 第一阶段只负责快速收拢到正确的 3D 大形；
- 第二阶段在已经有几何锚点的前提下，把细节转回 2D 图像空间处理。

这是一种很典型的 **把难题“重参数化”** 的思路：  
不是让一个模块同时承受“3D 一致性 + 高频纹理 + 导出可用性”，而是让每一层表示只做自己擅长的那部分。

### 战略权衡

| 设计选择 | 主要解决的约束 | 带来的能力收益 | 代价 |
|---|---|---|---|
| 3DGS + 高频 densification | NeRF 体渲染慢，SDS 下 pruning 不稳 | 分钟级优化，约 500 步收敛 | 直接生成的结果偏模糊，仍需二阶段细化 |
| 局部密度查询提 mesh | Gaussian 结果难直接进入工业工作流 | 输出显式 textured mesh，可用于动画/下游 | 依赖阈值与后处理，细薄结构可能损失 |
| 扩散编辑伪目标 + MSE 细化 UV | UV 上直接 SDS 容易出块状伪影 | 50 步左右增强纹理细节 | 仍可能保留烘焙光照，依赖初始纹理质量 |

## Part III：证据与局限

### 关键证据：能力跃迁到底在哪里

最强证据不是单一指标更高，而是 **速度-质量曲线整体右移**。

- **比较信号 1：image-to-3D 的速度/质量平衡明显更好**  
  DreamGaussian 在约 **2 分钟** 内达到 **CLIP-Sim 0.738**。  
  对比：
  - Zero-1-to-3：0.647，约 20 分钟
  - Zero-1-to-3*（带 mesh fine-tuning）：0.778，约 30 分钟  
  这说明它不是绝对最优分数，但以远小于前者的时间拿到了有竞争力的质量。

- **比较信号 2：两阶段不是装饰，而是真有增益**  
  仅 Stage 1：CLIP-Sim 0.678；  
  完整两阶段：0.738。  
  说明 mesh 提取 + UV 细化不是附属模块，而是质量闭环的一部分。

- **比较信号 3：用户研究支持“更可用”**  
  image-to-3D 用户研究中：
  - 参考视图一致性：4.31
  - 总体模型质量：3.92  
  都高于 One-2-3-45 和 Shap-E。  
  这说明它不仅“看起来快”，而且在用户感知层面也更接近实用。

- **消融信号：作者主张的因果旋钮确实有效**  
  去掉 densification、timestep annealing、参考视图损失，都会导致更模糊、更不准的几何和外观。  
  这支持了论文的核心论点：**效率提升不是仅来自换 renderer，而来自与生成式优化匹配的训练策略**。

- **案例信号：UV 上直接 SDS 的确不合适**  
  Figure 3 明确展示：直接用 SDS 细化 UV 会产生过饱和块状 artifact，而 MSE 细化能避免。  
  这很好地支持了作者对“第二阶段要换监督形式”的论证。

### 1-2 个应记住的指标
- **单图到 textured mesh：约 2 分钟**
- **CLIP-Sim：0.738（2 分钟）vs 0.647（20 分钟，Zero-1-to-3）**

### 局限性
- **Fails when**: 单视图缺失背面或被遮挡区域时，背部纹理容易模糊；text-to-3D 仍会出现 Janus / multi-face、过饱和纹理与 baked lighting；细薄高频几何不稳定。
- **Assumes**: 依赖强 2D diffusion prior（Zero-1-to-3 XL / Stable Diffusion）、物体居中单体场景、前景分割与相机绕物体采样；颜色建模以 diffuse 为主；Gaussian-to-mesh 依赖阈值、UV 展开和后处理启发式。
- **Not designed for**: 大场景或多物体复杂背景生成、可重光照材质/BRDF 恢复、测量级精确重建。

### 资源与复现依赖
- 论文实验在 **NVIDIA V100 16GB** 上进行，报告方法本身显存需求 **< 8GB**。
- image-to-3D 两阶段各约 1 分钟；text-to-3D 两阶段各约 2 分钟。
- 复现依赖的关键组件包括：优化版 3D Gaussian renderer、NVdiffrast、U2-Net 背景去除、外部 diffusion priors。
- 给出了项目页和较完整实现细节，但在提供文本中未见明确代码仓库链接，因此开放性按保守标准记为 `opensource/no`。

### 可复用组件
1. **生成式 3DGS 的 densification schedule**：适用于“监督模糊但渲染要快”的 3D 优化问题。  
2. **block-wise local density query**：可作为 Gaussian-to-mesh 的通用导出模块。  
3. **扩散编辑生成伪目标 + MSE 纹理细化**：适用于“已有粗纹理，但直接 score distillation 太不稳”的场景。

![[paperPDFs/Diffusion_Video/arXiv_2023/2023_DreamGaussian_Generative_Gaussian_Splatting_for_Efficient_3D_Content_Creation.pdf]]