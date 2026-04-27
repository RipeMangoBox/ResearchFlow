---
title: "Depth-Regularized Optimization for 3D Gaussian Splatting in Few-Shot Images"
venue: arXiv
year: 2023
tags:
  - 3D_Gaussian_Splatting
  - task/novel-view-synthesis
  - task/3d-reconstruction
  - depth-regularization
  - scale-alignment
  - early-stopping
  - dataset/NeRF-LLFF
  - opensource/no
core_operator: "用单目深度估计与COLMAP稀疏点对齐得到稠密深度先验，并以渲染深度监督、平滑约束和深度监控早停来稳定few-shot 3DGS优化"
primary_logic: |
  少量RGB图像 + SfM相机/稀疏点 → 单目深度估计并做尺度与偏置对齐 → 以渲染深度监督高斯位置，辅以局部平滑和深度损失驱动的早停 → 更稳健的几何与新视角合成
claims:
  - "在 NeRF-LLFF 的平均 2-view 设置上，该方法将 PSNR 从原始 3DGS 的 12.25 提升到 15.94，并将 LPIPS 从 0.471 降到 0.365 [evidence: comparison]"
  - "若去掉单目深度与 COLMAP 稀疏深度的尺度/偏置对齐，Horns 场景 2-view PSNR 会从 15.91 降至 7.86，说明绝对深度对齐是方法成立的必要条件 [evidence: ablation]"
  - "早停与深度平滑都提供了额外收益：在 Horns 2-view 上，去掉 early stop 时 PSNR 从 15.91 降到 13.99，去掉 smoothness 时降到 14.75 [evidence: ablation]"
related_work_position:
  extends: "3D Gaussian Splatting (Kerbl et al. 2023)"
  competes_with: "3D Gaussian Splatting (Kerbl et al. 2023); Dense Depth Priors for NeRF (Roessle et al. 2022)"
  complementary_to: "COLMAP (Schonberger and Frahm 2016); ZoeDepth (Bhat et al. 2023)"
evidence_strength: moderate
pdf_ref: "paperPDFs/Sparse/arXiv_2023/2023_Depth_Regularized_Optimization_for_3D_Gaussian_Splatting_in_Few_Sh ot_Images.pdf"
category: 3D_Gaussian_Splatting
---

# Depth-Regularized Optimization for 3D Gaussian Splatting in Few-Shot Images

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2311.13398)
> - **Summary**: 论文把单目深度先验对齐到 SfM 稀疏几何，再用深度监督、平滑约束和基于深度偏离的早停，缓解 few-shot 3DGS 的漂浮伪影与过拟合。
> - **Key Performance**: NeRF-LLFF 平均 2-view PSNR 15.94 vs 12.25（3DGS）；平均 5-view PSNR 18.74 vs 16.17。

> [!info] **Agent Summary**
> - **task_path**: 少量带位姿 RGB 图像 / SfM 稀疏点 -> 3D Gaussian 场景表示与新视角合成
> - **bottleneck**: few-shot 下 3DGS 只有局部颜色监督、缺少全局几何锚点，容易收敛到漂浮伪影和错误深度
> - **mechanism_delta**: 把单目估计深度先用 COLMAP 稀疏点做尺度/偏置对齐，再直接监督高斯渲染深度，并在深度损失反弹时提前停止训练
> - **evidence_signal**: NeRF-LLFF 多个 2-5 view 设置下平均指标持续优于原始 3DGS，且去掉深度对齐会明显崩溃
> - **reusable_ops**: [稀疏SfM点对齐单目深度, 基于渲染深度的几何正则, 深度损失监控早停]
> - **failure_modes**: [单目深度域外偏差会把高斯拉向错误几何, COLMAP在低纹理或困难表面失效时无法提供可靠尺度锚点]
> - **open_questions**: [能否摆脱对COLMAP稀疏点的依赖, 能否在天空与自然场景等深度不可靠区域保持稳定]

## Part I：问题与挑战

这篇文章要解决的真问题，不是“3DGS 表达能力不够”，而是**few-shot 时约束不够**。

3D Gaussian Splatting 的高斯元是强局部、相互独立的显式表示。视角足够多时，多视图颜色一致性能把这些局部元慢慢拉到一致几何；但当训练图像只有 2-5 张时，颜色损失仍可能在训练视图上很低，却对应着完全错误的 3D 布局，于是出现：
- 高斯漂浮在错误深度上；
- 局部最优导致几何塌陷；
- RGB 看起来“像对了”，但深度和新视角明显错。

更难的是，few-shot 下连 **COLMAP 的稀疏点也会变得过少**，不足以直接约束所有高斯。于是瓶颈变成两层：
1. **3DGS 缺少全局几何锚点**；
2. **few-shot 下可用的真实几何监督又太稀疏**。

为什么现在值得做：  
因为 3DGS 已经把重建速度和实时渲染带到实用区间，实际应用自然会追问“能不能只拍很少几张图”。同时，预训练单目深度模型已经足够强，虽然不精确，但能提供**粗而密**的几何先验，正好适合拿来当 few-shot 训练的“防跑偏护栏”。

**输入/输出接口**：
- 输入：少量 RGB 图像、相机位姿与 SfM 稀疏点、预训练单目深度模型；
- 输出：一个可渲染的新视角 3D Gaussian 场景表示。

**边界条件**：
- 静态场景；
- 依赖外部位姿/SfM；
- 重点是 sparse-view 优化稳定性，不是从零联合求位姿与重建。

## Part II：方法与洞察

### 方法主线

作者在原始 3DGS 上加入了一条“几何护栏”：

1. **先估计稠密深度**  
   对每张训练图像用单目深度模型（文中用 ZoeDepth）估计深度，但这一步只有相对深度。

2. **再做尺度/偏置对齐**  
   用 COLMAP 投影得到的稀疏深度点，去拟合每张单目深度的 scale 和 offset。  
   这一步的作用不是追求精确深度，而是把每个视图的深度先验拉到**同一个多视图几何坐标系**里。

3. **对高斯渲染深度做监督**  
   作者沿用 3DGS 的 alpha-blending rasterization，不仅渲染颜色，也直接渲染深度。  
   然后用对齐后的稠密深度去约束渲染深度，从而把高斯的位置往合理几何上拉。

4. **加入非边缘平滑约束**  
   在 Canny 边缘之外，鼓励邻近像素的深度更平滑，缓解单目深度和 few-shot 优化带来的局部冲突。

5. **few-shot 专用训练改造**  
   - 把 SH 最大阶数降到 1，减少高频外观过拟合；
   - 移除 opacity reset，避免少视图下不可逆崩坏；
   - 用 depth loss 的移动平均作为 early stop 信号：一旦几何开始背离深度先验，就停止训练。

一个很关键的实现取舍是：**作者没有把单目深度直接反投影成大量初始化点来替代优化**，而是把它作为 soft constraint。  
消融显示，直接 unproject 稠密深度去初始化反而不如保留稀疏 COLMAP 点做初始化，再让稠密深度去“扶正”优化过程。

### 核心直觉

原始 3DGS 在 few-shot 下的问题，本质是：

**纯颜色局部拟合** → **几何解空间过大，容易落入漂浮局部最优** → **训练视图可被记住，但真实几何和新视角不稳**

作者改变的关键旋钮是：

**给每个视图加入对齐后的稠密深度锚点，并在后期几何开始漂移时提前停止**

于是约束结构变成：

**粗但密的几何先验 + 局部平滑 + 深度监控早停** → **高斯位置被限制在更小、更合理的几何可行域** → **few-shot 下也能形成更稳定的表面与更少的 floating artifacts**

为什么“粗深度”也有效？  
因为 few-shot 3DGS 的主失败模式不是缺少毫米级细节，而是**连前后层次和大致表面位置都没锁住**。只要先把高斯压回合理深度层，颜色损失就更可能去优化外观细节，而不是继续“编造几何”。

### 策略取舍

| 设计项 | 改变了什么约束 | 带来的能力变化 | 代价/风险 |
|---|---|---|---|
| 单目深度 + 稀疏点对齐 | 把相对深度变成多视图一致的粗绝对几何 | few-shot 下获得全局几何锚点 | 依赖深度模型域泛化与 COLMAP 质量 |
| 渲染深度监督 | 从只约束颜色变成直接约束空间位置 | 明显减少 floating artifacts | 深度先验错时会误导几何 |
| 非边缘平滑 | 限制局部深度抖动 | 表面更稳定、更连续 | 过强会抹平真实边界 |
| 深度损失驱动早停 | 阻止后期 color-only overfitting | few-shot 训练更稳 | 需要选择合适停止时机 |
| SH 降阶 + 去掉 opacity reset | 降低表达自由度，减少不可逆崩坏 | 更适合极少视图训练 | 高频外观表达受限 |

## Part III：证据与局限

### 关键证据信号

- **比较信号**：在 8 个 NeRF-LLFF 场景、每个 setting 10 个随机 seed 的平均结果上，作者方法在 2-view 和 5-view 都稳定优于原始 3DGS，说明收益不是单场景偶然。
- **几何信号**：论文同时展示 RGB 和深度图。基线 3DGS 经常出现“RGB 看着还行，但深度明显错”的情况；作者方法的深度更连贯，支持其改进确实来自几何稳定化，而不是单纯贴图式记忆。
- **消融信号**：去掉尺度/偏置对齐会接近训练失败；去掉 depth loss、smoothness 或 early stop 都会退化，说明核心不是某一个 trick，而是“对齐深度先验 + 稳定训练策略”的组合。
- **上界信号**：oracle 使用 pseudo-GT depth 时显著更强，说明这条路线本身是对的，当前剩余瓶颈主要在深度先验质量。
- **初始化信号**：Table 3 显示更好的初始化点仍会继续提升结果，说明深度正则虽然有效，但**没有消除 3DGS 对初始化质量的敏感性**。

### 局限性

- **Fails when**: 单目深度模型发生明显域偏移时（如自然场景、天空、细碎植被、难估深纹理），或 COLMAP 在低纹理/反光/困难表面提不出可靠稀疏点时，几何锚点会失真并误导高斯优化。
- **Assumes**: 依赖预训练单目深度模型、COLMAP 位姿与稀疏点；实验里还先用整场景图像统一 COLMAP 坐标与相机，这意味着评测不覆盖“仅靠 k 张图同时估位姿+重建”的完整 few-shot 问题；此外还需要自定义 CUDA depth rasterizer 和 GPU 环境。
- **Not designed for**: 动态场景、无位姿输入、端到端 few-shot SfM、以及要求高精度公制几何的场景。

### 可复用组件

- **稀疏 SfM 点校准单目深度**：把相对深度变成多视图可用的几何 prior。
- **3DGS 深度 rasterization**：复用颜色渲染管线，低成本给显式表示加入几何监督。
- **depth-loss-based early stop**：当主目标偏外观、辅目标偏结构时，可作为通用的防过拟合机制。

![[paperPDFs/Sparse/arXiv_2023/2023_Depth_Regularized_Optimization_for_3D_Gaussian_Splatting_in_Few_Shot_Images.pdf]]