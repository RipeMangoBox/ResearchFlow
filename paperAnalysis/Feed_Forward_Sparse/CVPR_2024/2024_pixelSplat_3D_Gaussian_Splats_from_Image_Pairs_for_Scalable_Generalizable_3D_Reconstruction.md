---
title: "pixelSplat: 3D Gaussian Splats from Image Pairs for Scalable Generalizable 3D Reconstruction"
venue: arXiv
year: 2023
tags:
  - 3D_Gaussian_Splatting
  - task/novel-view-synthesis
  - task/3d-reconstruction
  - epipolar-transformer
  - probabilistic-sampling
  - rasterization
  - dataset/RealEstate10k
  - dataset/ACID
  - opensource/no
core_operator: 以带深度位置编码的双视图极线Transformer注入场景尺度信息，并沿像素射线预测离散深度分布来可微采样3D Gaussian
primary_logic: |
  双视图图像+SfM位姿（每场景尺度未知） → 极线跨视图注意力对齐对应并把三角化深度写入像素特征 → 沿每条射线预测离散深度概率、采样Gaussian中心并用采样概率设定不透明度 → 输出可实时渲染的显式3D Gaussian辐射场
claims:
  - "Claim 1: pixelSplat在RealEstate10k和ACID上均优于Du et al.、GPNR和pixelNeRF，其中RealEstate10k达到26.09 PSNR / 0.136 LPIPS，ACID达到28.27 PSNR / 0.146 LPIPS [evidence: comparison]"
  - "Claim 2: 去掉极线编码会使RealEstate10k上的PSNR从26.09降至19.89，说明尺度感知的跨视图对应对稀疏视图3D结构恢复至关重要 [evidence: ablation]"
  - "Claim 3: 将Gaussian位置从直接回归改为概率深度采样可把RealEstate10k上的PSNR从24.62提升到26.09，并明显减少局部最小值导致的散斑伪影 [evidence: ablation]"
related_work_position:
  extends: "3D Gaussian Splatting (Kerbl et al. 2023)"
  competes_with: "Du et al. (2023); GPNR (Suhail et al. 2022)"
  complementary_to: "FlowCam (Smith et al. 2023); Viewset Diffusion (Szymanowicz et al. 2023)"
evidence_strength: strong
pdf_ref: paperPDFs/Feed_Forward_Sparse/CVPR_2024/2024_pixelSplat_3D_Gaussian_Splats_from_Image_Pairs_for_Scalable_Generalizable_3D_Reconstruction.pdf
category: 3D_Gaussian_Splatting
---

# pixelSplat: 3D Gaussian Splats from Image Pairs for Scalable Generalizable 3D Reconstruction

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2312.12337), [Project](https://dcharatan.github.io/pixelsplat/)
> - **Summary**: 这篇工作把“从两张有位姿图像前向预测显式3D Gaussian场”做成了端到端模型，关键是同时解决了SfM尺度歧义和Gaussian位置直接回归的局部最优问题。
> - **Key Performance**: RealEstate10k: 26.09 PSNR / 0.136 LPIPS；ACID: 28.27 PSNR / 0.146 LPIPS；渲染仅 0.002 s/帧，较 Du et al. 快约 650×

> [!info] **Agent Summary**
> - **task_path**: 双视图图像对+相机位姿 -> 显式3D Gaussian辐射场 -> 新视角图像
> - **bottleneck**: SfM每场景尺度不确定，且直接回归Gaussian中心会因局部支持与遮挡导致局部最优
> - **mechanism_delta**: 用带深度编码的极线注意力把正确尺度写入像素特征，再把Gaussian位置预测改为沿射线的离散深度分布采样
> - **evidence_signal**: 双数据集全面优于Du et al./GPNR/pixelNeRF，且去掉极线编码与概率采样分别使PSNR降到19.89与24.62
> - **reusable_ops**: [尺度感知极线注意力, 不透明度耦合的概率深度采样]
> - **failure_modes**: [反射面容易变透明, OOD视角下Gaussian呈billboard状]
> - **open_questions**: [如何跨视图去重并融合Gaussians, 如何把极线注意力扩展到多视图而不爆内存]

## Part I：问题与挑战

这篇论文处理的是**稀疏输入下的可泛化新视角合成/3D重建**：测试时只给两张参考图和相机参数，就要恢复一个可渲染、可编辑的3D场景表示。

### 真正的难点是什么？
不是“再做一个更快的renderer”这么简单，而是要同时满足三件事：

1. **跨场景泛化**：不能像单场景3DGS那样对每个场景单独优化。
2. **显式3D表示**：不能像light-field transformer那样只输出黑盒渲染结果。
3. **端到端可训练**：又不能依赖3DGS原始方法里的非可微 pruning / splitting。

### 真瓶颈
作者强调了两个核心瓶颈：

- **尺度歧义（scale ambiguity）**  
  RealEstate10k 和 ACID 的相机位姿来自 SfM，只能恢复到“每个场景一个任意尺度因子”。  
  单图编码器即使预测出“正确几何形状”，也不一定落在与当前场景SfM坐标一致的深度尺度上。

- **primitive 回归的局部最优**  
  3D Gaussian 是局部支持的。若直接回归 Gaussian center，错误初始化的 primitive 很难靠梯度“走”到正确位置：
  - 太远时几乎收不到梯度；
  - 中间要穿过空区域时，loss 往往不单调下降；
  - 原始3DGS靠非可微的密度控制规避，但这里要训练神经网络，不能这么做。

### 输入/输出接口
- **输入**：两张参考图 + 已知内外参/相机位姿
- **输出**：显式 3D Gaussian 集合 \((\mu, \Sigma, \alpha, SH)\)，可直接用 rasterization 渲染新视角
- **边界条件**：
  - 需要相机位姿
  - 更偏向 wide-baseline 但仍需一定跨视图重叠
  - 主要面向 view synthesis / reconstruction，不是生成未观测区域

### 为什么现在值得做？
因为 3D Gaussian Splatting 已经证明了**显式primitive表示可以做到实时渲染**，但此前它基本局限于单场景优化。pixelSplat 的价值在于把这条路线推进到**可泛化、前向一次出结果**的 setting，补上了“速度、显式结构、泛化能力”三者之间的缺口。

## Part II：方法与洞察

### 方法总览
pixelSplat 可分成两步：

1. **双视图尺度感知编码**
   - 每张图先做 per-image feature extraction。
   - 对图像 \(I\) 中每个像素，在另一视图 \(\tilde I\) 的极线上采样。
   - 这些极线样本不仅有图像特征，还拼接了**由两视图三角化得到的深度位置编码**。
   - 通过 epipolar cross-attention，模型学会找到跨视图对应，同时把“与当前场景SfM尺度一致的深度”写进像素特征。
   - 后接 self-attention，把这类尺度信息传播到无直接对应的区域。

2. **像素对齐的 Gaussian 预测**
   - 每个像素特征预测一组 Gaussian 参数。
   - 协方差 \(\Sigma\) 和 SH 系数 \(S\) 直接回归。
   - **关键不再直接回归 Gaussian center**，而是沿该像素射线预测一个**离散深度分布**：
     - 预测每个 depth bucket 的概率 \(\phi\)
     - 采样 bucket 索引 \(z\)
     - 用 bucket depth + offset 反投影得到 Gaussian 均值 \(\mu\)

最终把两张参考图预测出的 Gaussian 取并集，变成整个场景的显式3D表示，再用3DGS的 rasterization renderer 渲染新视角。

### 核心直觉

**这篇论文真正改动的，不是“Gaussian renderer”，而是“Gaussian位置是怎么被学习出来的”。**

#### 1) 从“单点深度回归”改成“深度分布采样”
- **原来**：网络直接回归某个 Gaussian 的深度/中心点。
- **现在**：网络先输出“沿这条射线哪里更可能有表面”的概率分布，再从中采样。

这改变了优化地形：
- 直接回归要求一个 primitive 在3D空间中连续移动到正确位置；
- 概率采样则允许模型在整条射线上“重新分配概率质量”，本质上更像**可微的 spawn/delete**。

#### 2) 用 opacity 绑定 sampled probability，让采样可训练
作者把采样到的 bucket 概率直接设成 Gaussian opacity：**\(\alpha = \phi_z\)**。

因果上，这一步非常关键：
- 如果这次采样到的位置是对的，渲染loss会推动该 Gaussian 更“有用”，其 opacity 增大；
- opacity 的梯度又回流到对应 bucket 概率；
- 于是未来更容易再次采样到这个位置；
- 错误位置则会被抑制。

所以它不是在“移动一个错误的Gaussian”，而是在“提升正确深度被再次采样的概率”。

#### 3) 极线注意力不只是找对应，还在写入“场景尺度”
只做跨视图对应还不够。作者把**三角化深度的位置编码**也送入极线注意力，所以模型不只是知道“哪儿匹配”，还知道“这个匹配在当前场景尺度下对应多深”。  
这正是它能适配 SfM 任意尺度的原因。

### 战略取舍

| 设计选择 | 改变了什么瓶颈 | 获得的能力 | 代价/取舍 |
|---|---|---|---|
| 极线注意力 + 深度位置编码 | 从“单图无尺度深度”变成“跨视图带尺度的几何证据” | 输出与SfM坐标一致的几何 | 依赖相机位姿与视图重叠 |
| 概率深度采样 + \(\alpha=\phi_z\) | 从“连续移动primitive”变成“沿射线重分配概率质量” | 显著缓解局部最优，支持端到端训练 | 需要depth bin、near/far plane等超参 |
| 显式3D Gaussian表示 | 从黑盒渲染变成可解释3D结构 | 可实时渲染、可编辑、可导出 | 跨视图会有重复Gaussians，未做融合 |
| 双视图像素对齐预测 | 强利用局部几何与像素对应 | 两张图就可前向重建 | 对大规模多视图扩展不够经济 |

## Part III：证据与局限

### 关键实验信号

#### 1) 比较实验：质量和效率同时提升
在 **RealEstate10k** 和 **ACID** 上，pixelSplat 全面超过 Du et al., GPNR, pixelNeRF。

- **RealEstate10k**：26.09 PSNR / 0.863 SSIM / 0.136 LPIPS  
  对比 Du et al. 的 24.78 / 0.820 / 0.213
- **ACID**：28.27 PSNR / 0.843 SSIM / 0.146 LPIPS  
  对比 Du et al. 的 26.88 / 0.799 / 0.218

这说明它不是只换来更快渲染，而是真正提升了**wide-baseline 两视图场景理解与结构恢复**。

#### 2) 效率信号：显式3DGS带来巨大渲染优势
最强的系统层信号是渲染成本：

- pixelSplat：**0.002 s/帧**
- Du et al.：1.309 s/帧
- GPNR：13.340 s/帧

也就是说，pixelSplat 的单帧渲染大约比 Du et al. **快 650×**。  
训练/推理显存也更低（表中训练内存 14.4 GB，显著低于多个基线）。

需要注意的是：它的**编码时间**不是最优（0.102 s vs Du 的 0.016 s），但编码是每个场景一次性开销，随后多视角渲染会被极快的 decoder 摊薄。

#### 3) 消融实验：方法为何有效是可定位的
最重要的因果验证来自 Table 2：

- **No Epipolar Encoder**：PSNR 从 **26.09 → 19.89**  
  说明没有跨视图尺度对齐，结构会明显崩坏并出现 ghosting。
- **No Depth Encoding**：**26.09 → 24.97**  
  说明极线模块不仅在找 correspondence，更在利用 triangulated depth 解决尺度歧义。
- **No Probabilistic Sampling**：**26.09 → 24.62**  
  说明概率式位置预测确实缓解了直接回归 Gaussian center 的局部最优。

此外，attention 可视化也显示模型确实在极线上学到了正确对应，而不是只靠统计偏置。

### 能力跃迁到底在哪里？
相对 prior work，pixelSplat 的跃迁点在于：

- 相比 **light-field transformer**：不再是黑盒渲染，而是显式3D结构
- 相比 **pixelNeRF/NeRF类方法**：渲染不再依赖密集射线采样，速度和内存大幅下降
- 相比 **单场景3DGS**：第一次把3D Gaussian表示放进“从两张图前向预测”的可泛化 setting

### 局限性
- **Fails when**: 反射/镜面区域容易被表示成透明Gaussians；当视角远离训练分布时，Gaussians 会呈现 billboard-like 外观；多视图扩展时极线注意力的显存开销会迅速增大。
- **Assumes**: 需要已知且较准确的相机位姿；需要足够的跨视图重叠来形成极线对应；依赖 DINO 预训练的 ResNet-50 与 ViT-B/8；训练配置较重，补充材料中报告约需单卡 80GB VRAM；还依赖 3DGS/e3nn 工程栈。正文未明确代码开放状态。
- **Not designed for**: 未观测区域的生成式补全；跨视图 Gaussian 去重/融合；无位姿训练；大规模多视图输入下的高效推理。

### 可复用部件
- **尺度感知极线注意力**：适合任何受 SfM arbitrary scale 影响的稀疏多视图任务。
- **概率式primitive位置预测**：适合所有“局部支持显式primitive难以直接回归”的场景。
- **可选深度正则化微调**：补充材料显示其能改善点云/几何观感，且对渲染质量影响较小。

![[paperPDFs/Feed_Forward_Sparse/CVPR_2024/2024_pixelSplat_3D_Gaussian_Splats_from_Image_Pairs_for_Scalable_Generalizable_3D_Reconstruction.pdf]]