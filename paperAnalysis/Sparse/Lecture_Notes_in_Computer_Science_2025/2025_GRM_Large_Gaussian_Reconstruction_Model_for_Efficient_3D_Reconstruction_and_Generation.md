---
title: "GRM: Large Gaussian Reconstruction Model for Efficient 3D Reconstruction and Generation"
venue: arXiv
year: 2024
tags:
  - 3D_Gaussian_Splatting
  - task/3d-reconstruction
  - transformer
  - pixel-aligned-gaussians
  - windowed-self-attention
  - dataset/Objaverse
  - dataset/GSO
  - opensource/promised
core_operator: "用跨视角 Transformer 将多视图像素翻译为沿相机射线约束的像素对齐高斯，并通过 splatting 实时得到高保真 3D 资产。"
primary_logic: |
  四个带位姿输入视图 → 全局跨视角 Transformer 编码与 shifted-window 上采样 → 预测每像素 Gaussian 属性图并沿视线反投影 → 输出可实时渲染的稠密 3D Gaussians
claims:
  - "Claim 1: 在 GSO 的 4 视图稀疏重建上，GRM 取得 30.05 PSNR、0.906 SSIM 和 0.052 LPIPS，显著优于 MV-LRM 的 25.38、0.897 和 0.068，同时 3D 表征推理仅需 0.11 秒并支持实时渲染 [evidence: comparison]"
  - "Claim 2: 结合 Zero123++ 后，GRM 在单图到 3D 上达到 20.10 PSNR、0.136 LPIPS、0.932 CLIP 和 27.4 FID，优于 LGM 的 16.90、0.235、0.855 和 42.1，且总推理时间同为 5 秒 [evidence: comparison]"
  - "Claim 3: 将高斯位置约束为沿输入相机射线的 pixel-aligned 预测优于直接预测 XYZ，ablation 中 full model 的 29.48 PSNR / 0.031 LPIPS 优于 XYZ prediction 的 28.61 / 0.037，说明射线约束降低了无结构 3D 预测难度 [evidence: ablation]"
related_work_position:
  extends: "LRM (Hong et al. 2023)"
  competes_with: "LGM (Tang et al. 2024); MV-LRM (Li et al. 2023)"
  complementary_to: "Instant3D (Li et al. 2023); Zero123++ (Shi et al. 2023)"
evidence_strength: strong
pdf_ref: paperPDFs/Sparse/Lecture_Notes_in_Computer_Science_2025/2025_GRM_Large_Gaussian_Reconstruction_Model_for_Efficient_3D_Reconstruction_and_Generation.pdf
category: 3D_Gaussian_Splatting
---

# GRM: Large Gaussian Reconstruction Model for Efficient 3D Reconstruction and Generation

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2403.14621), [Project](https://justimyhxu.github.io/projects/grm/)
> - **Summary**: GRM 用 pixel-aligned 3D Gaussians 和纯 Transformer 把 4 个带位姿视图直接 lifted 成可实时渲染的 3D 资产，绕开 triplane/NeRF 体渲染瓶颈，从而把对象级 3D 重建与下游生成同时做快、做清晰。
> - **Key Performance**: 4 视图 GSO 重建达到 30.05 PSNR / 0.052 LPIPS，3D 表征推理约 0.11s；结合 Zero123++ 后单图到 3D 为 27.4 FID、总耗时约 5s。

> [!info] **Agent Summary**
> - **task_path**: 4 张带相机位姿的物体视图（或文本/单图先生成的 4 张多视图） -> 对象级 3D Gaussian 资产
> - **bottleneck**: triplane/NeRF 体渲染过慢且分辨率受限，同时无结构 3D Gaussian 直接预测在 sparse-view 下难以稳定学习
> - **mechanism_delta**: 用跨视角全局 Transformer + shifted-window 上采样预测每像素 Gaussian 属性，并沿输入射线反投影成稠密 3D Gaussians
> - **evidence_signal**: GSO 4 视图重建显著超越 MV-LRM/LGM，且 ablation 证明 pixel-aligned 约束与 transformer upsampler 都是关键因子
> - **reusable_ops**: [ray-constrained Gaussian parameterization, shifted-window transformer upsampler]
> - **failure_modes**: [输入多视图不一致时质量明显下降, 未被任何输入观察的区域会变模糊]
> - **open_questions**: [如何把确定性重建器升级成概率式多解补全, 如何扩展到非物体中心的大场景与更复杂数据分布]

## Part I：问题与挑战

GRM 解决的不是“任意单图自由幻想 3D”的开放式问题，而是一个更聚焦、更工程化的问题：**在只有 4 张已知位姿视图时，能否用一次前向传播恢复高保真、可实时渲染的对象级 3D 资产**。

### 这个问题为什么难
已有 feed-forward 3D 生成/重建系统已经证明：  
前端可以很快拿到文本或单图对应的多视图图像，但**把这些图像稳定地抬升成 3D 表示**仍受两类瓶颈限制：

1. **表示瓶颈**  
   许多 SOTA feed-forward 方法仍依赖 triplane + volume rendering。  
   这会带来：
   - 渲染慢；
   - 内存重；
   - 3D 分辨率受限；
   - 高频纹理细节不容易保住。

2. **预测瓶颈**  
   3D Gaussian 虽然渲染高效，但若直接预测一个无结构的 Gaussian 集合，会很难学。  
   原因是：
   - 高斯参数彼此强耦合；
   - 多组不同高斯配置可能渲染出相似图像；
   - sparse-view 下更容易出现局部最优、漂浮点和几何不稳定。

3. **细节瓶颈**  
   Transformer tokenization 通常先降采样，利于全局建模，但会损失像素级高频信息。  
   对 3D 资产来说，这直接影响边缘、纹理和小结构重建。

### 输入/输出接口
- **输入**：4 张带相机位姿的图像
- **输出**：一组可实时 splatting 渲染的 3D Gaussians
- **下游扩展**：
  - 文本 → 多视图图像（Instant3D）→ GRM → 3D
  - 单图 → 多视图图像（Zero123++）→ GRM → 3D

### 边界条件
作者其实刻意回避了一个更不适定的问题：**在信息严重缺失时强行 hallucinate 未观测区域**。  
他们的选择是：
- 用 **4 个稀疏但分布合理、能覆盖物体** 的视角训练；
- 把容量优先用于 **高保真重建**，而不是自由补全。

这意味着该方法天然更偏向：
- 对象中心场景；
- 输入视图基本一致；
- 视角覆盖相对充分。

**What/Why**：真正瓶颈不是“4 张图一定太少”，而是现有 feed-forward 3D 后端仍被体渲染成本和无结构 3D 预测难度卡住；而多视图 diffusion 已经足够成熟，所以现在最值得做的是一个高效的 3D lifting 模块。

## Part II：方法与洞察

GRM 的核心不只是“把 NeRF 换成 Gaussian”，而是**把 3D 预测重新参数化为一个更容易学的、与输入像素对齐的问题**。

### 核心直觉

- **What changed**  
  从 `triplane / neural volume` 改成 `pixel-aligned Gaussian maps`；  
  从 `卷积式局部融合` 改成 `跨全部视图的全局 Transformer + shifted-window 上采样`。

- **Which bottleneck changed**  
  1. 把输出空间从“无结构 3D primitive 集合”压缩成“每个像素沿相机射线的一组属性”。  
  2. 把跨视图一致性建模从隐式局部卷积，变成显式全局 token 交互。  
  3. 把高分辨率细节恢复从普通 CNN upsampling，变成可以跨窗口传递非局部信息的 transformer upsampler。

- **What capability changed**  
  结果是模型能在约 0.11 秒内生成高密度 3D Gaussians，保持实时渲染，同时显著提升几何稳定性和纹理细节。

- **为什么这在因果上有效**  
  - **射线约束** 给每个高斯一个“可解释锚点”：它必须对应某条观测射线，而不是在 3D 空间里自由漂移。这样解空间更小、更结构化。  
  - **全局跨视图 attention** 让不同视图中同一物体区域可以直接交换信息，更像显式做 correspondence matching。  
  - **shifted-window 上采样** 在不做 full-resolution 全局注意力的情况下，仍能让细节和非局部线索跨区域传播，所以能兼顾成本与保真度。

**How**：作者真正拧动的“因果旋钮”是输出参数化方式——先用射线把 3D 预测约束住，再用跨视图 Transformer 去补全一致性和细节。

### 关键模块

#### 1. Pixel-aligned Gaussians
GRM 不直接回归一堆自由 3D 高斯，而是对每个输入像素预测一个高斯属性向量，主要包括：
- 深度
- 旋转
- 尺度
- 不透明度
- 颜色基项

然后把它沿该像素对应的相机射线反投影到 3D。  
这样 4 张输入图就会生成一大批**稠密分布**的 3D Gaussians。

这一步的本质是：  
**把“预测 3D 场”变成“预测每个像素该在 3D 哪里落点以及带什么属性”**。

#### 2. 跨视图 Transformer Encoder
作者把 4 个视图的 token 拼接后做全局 self-attention，并把相机信息以 Plücker embedding 注入到像素中。  
其作用不是简单“堆大模型”，而是让网络学会：
- 哪些 token 来自同一个 3D 表面；
- 哪些区域在不同视图中应该一致；
- 哪些局部纹理可由其他视图补证。

#### 3. Transformer-based Upsampler
Patch tokenization 会损失细节，所以作者又做了一个上采样器：
- 线性扩维
- PixelShuffle 提升分辨率
- windowed self-attention
- shifted window 让信息跨窗口流动

这部分是论文里一个很关键但容易被忽视的点：  
**高保真并不只来自 Gaussian 表示，也来自“把全局信息重新送回像素级分辨率”的机制**。

#### 4. 训练信号与稳定化
训练时除了 novel-view 图像监督，还加入：
- **alpha/mask supervision**：去除 floaters
- **bounded scale activation**：避免 Gaussian 尺度过大导致模糊和训练不稳

这些设计虽然不花哨，但和最终质量强相关。

### 战略性取舍

| 设计选择 | 解决的瓶颈 | 带来的收益 | 代价 / 假设 |
| --- | --- | --- | --- |
| Pixel-aligned Gaussians | 无结构 3D 预测难优化 | 更稳定的几何、更高密度高斯、更好纹理 | 需要相机位姿；未观测区域仍难补全 |
| 全局跨视图 Transformer | sparse-view correspondence 弱 | 更强跨视图一致性 | 训练显存/算力成本高 |
| Shifted-window upsampler | 降采样 token 丢细节 | 在可控成本下恢复高频外观 | 仍是近似的全局传播 |
| Alpha regularization + scale bound | floater 与大高斯导致模糊 | 结果更干净、更锐利 | 依赖 mask 监督和合理超参 |
| 确定性 reconstructor | 生成速度与稳定性 | 快速、单次前向即可出 3D | 不擅长表达多种合理补全结果 |

## Part III：证据与局限

**So what**：GRM 的能力跃迁在于，它不是单纯把推理时间再缩一点，而是在几乎同级别的前向速度下，把对象级 3D lifting 的保真度明显拉高，并让 Gaussian 表示真正成为 feed-forward 3D 生成的高效后端。

### 关键证据信号

- **标准比较 / 稀疏视图重建**  
  在 GSO 上，GRM 用 4 视图达到 **30.05 PSNR / 0.052 LPIPS**，显著超过 MV-LRM 的 25.38 / 0.068，也超过并发的 LGM。  
  更重要的是，它的 3D 表征推理只要 **0.11s**，而且渲染是实时的。  
  这说明它不是“用更慢换更好”，而是同时改进了质量与部署效率。

- **标准比较 / 单图到 3D**  
  接 Zero123++ 后，GRM 的 **FID 27.4**、**CLIP 0.932**，显著优于 LGM 的 42.1 和 0.855，总时间仍是 **5 秒**。  
  这证明 GRM 作为后端 lifting 模块，对 image-to-3D 是直接可迁移的。

- **人评 + 文本到 3D**  
  接 Instant3D 后，GRM 在用户偏好上达到 **29.5%**，高于 Instant3D 的 15.7%，并且只需 **8 秒**；  
  同时接近优化式 MVDream 的感知质量，但后者要 **1 小时**。  
  这说明它在“质量-速度”曲线上占据了很强的位置。

- **消融 / 因果支持**  
  论文并非只靠大模型堆性能。关键消融都支持其机制叙事：  
  - 直接预测 XYZ 不如 ray-constrained 的 pixel-aligned 预测；  
  - CNN upsampler 不如 transformer upsampler；  
  - alpha regularization 能显著减少 floater；  
  - bounded scale activation 比传统 exponential scale 更稳。  
  这使得性能提升更像是**设计选择带来的因果结果**，而不是单纯训练更大。

- **补充几何评测**  
  Supplementary 中 sparse-view 几何 F-score 和 single-image 几何 CD/F-score 也都领先，说明改进不只停留在渲染外观。

### 局限性

- Fails when: 输入多视图彼此不一致、覆盖不足，或存在大面积未被任何输入观察到的区域时；此时会出现模糊纹理、补全不稳，且确定性模型无法表达多种合理解。
- Assumes: 已知或可恢复的相机位姿、对象 mask、对象中心数据分布；生成任务依赖外部多视图扩散前端（Instant3D / Zero123++）；训练依赖约 100k 过滤后的 Objaverse 对象与 32×A100 级别计算，当前仅承诺开源。
- Not designed for: 房间/城市级大场景、动态场景、强 hallucination 式自由补全、严重输入冲突下的概率式多解建模。

### 可复用组件

- **Pixel-aligned Gaussian parameterization**  
  适合任何想把“无结构 3D 预测”转成“受图像几何约束的属性预测”的模型。

- **Shifted-window transformer upsampler**  
  适合需要在高分辨率空间里恢复细节、同时保留非局部多视图线索的 3D/多视图模型。

- **Alpha regularization for floaters**  
  对 generalizable Gaussian 模型很实用，尤其在高密度高斯预测时。

- **Bounded scale activation**  
  是一个简单但有效的稳定化技巧，能减少过大高斯导致的模糊和训练不稳。

## Local PDF reference

![[paperPDFs/Sparse/Lecture_Notes_in_Computer_Science_2025/2025_GRM_Large_Gaussian_Reconstruction_Model_for_Efficient_3D_Reconstruction_and_Generation.pdf]]