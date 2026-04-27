---
title: "Semantic Gaussians: Open-Vocabulary Scene Understanding with 3D Gaussian Splatting"
venue: arXiv
year: 2024
tags:
  - 3D_Gaussian_Splatting
  - task/3d-semantic-segmentation
  - task/object-localization
  - gaussian-splatting
  - sparse-convolution
  - feature-distillation
  - dataset/ScanNet
  - dataset/LERF
  - opensource/no
core_operator: 将2D开放词汇语义通过几何投影无训练注入3D高斯，并用稀疏3D卷积蒸馏出可快速预测的语义高斯表示
primary_logic: |
  多视角RGB图像 + 现成3DGS场景 + 2D预训练语义编码器
  → 用相机几何与深度/遮挡关系建立像素到高斯的对应并跨视角聚合语义
  → 再用3D稀疏卷积网络从原始高斯属性预测语义分量
  → 输出可供文本查询的语义高斯，用于开放词汇分割、定位与编辑
claims:
  - "在 ScanNet 的 12 个验证场景上，2D+3D 融合版达到 62.0 mIoU / 77.0 mAcc，优于 Feature 3DGS 的 59.2 / 75.1 和原始 LSeg 的 56.1 / 74.5 [evidence: comparison]"
  - "在 LERF 的 4 个场景上，基于 SAM+CLIP 的 Semantic Gaussians 达到 85.2% 平均定位准确率，略高于 LangSplat 的 84.3% [evidence: comparison]"
  - "将 3D 网络输入退化为 XYZ+RGB 后，性能从 62.0 mIoU / 77.0 mAcc 降至 58.9 / 74.7，说明高斯的尺度/旋转/不透明度等属性对语义预测有贡献 [evidence: ablation]"
related_work_position:
  extends: "3D Gaussian Splatting (Kerbl et al. 2023)"
  competes_with: "Feature 3DGS (Zhou et al. 2024); LangSplat (Qin et al. 2024)"
  complementary_to: "SAM (Kirillov et al. 2023); Dynamic 3D Gaussians (Luiten et al. 2023)"
evidence_strength: moderate
pdf_ref: paperPDFs/Language_Embedding_Segmentation/IEEE_Transactions_on_Circuits_and_Systems_for_Video_Technology_2026/2026_Semantic_Gaussians_Open_Vocabulary_Scene_Understanding_with_3D_Gaussian_Splatting.pdf
category: 3D_Gaussian_Splatting
---

# Semantic Gaussians: Open-Vocabulary Scene Understanding with 3D Gaussian Splatting

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2403.15624)
> - **Summary**: 该工作把现成 2D 开放词汇模型的语义，通过几何一致的方式直接投到 3D Gaussian Splatting 上，并再训练一个 3D 稀疏卷积头做快速预测，从而避免逐场景重训语义 3DGS。
> - **Key Performance**: ScanNet 上 62.0 mIoU / 77.0 mAcc；LERF 上 85.2% 平均定位准确率。

> [!info] **Agent Summary**
> - **task_path**: 多视角 RGB + 相机参数 / 现成 3DGS 场景 → 语义高斯表示 → 开放词汇分割、定位、编辑
> - **bottleneck**: 2D 开放词汇语义跨视角不一致，而 NeRF/语义 3DGS 路线通常需要逐场景再训练且推理慢
> - **mechanism_delta**: 用基于几何和深度的无训练 2D→3D 投影先给每个 Gaussian 赋语义，再用稀疏 3D 卷积从原始高斯属性蒸馏出快速、较一致的语义预测
> - **evidence_signal**: ScanNet 上融合版显著超过 LSeg 与 Feature 3DGS，且 LERF 上通过替换为 SAM+CLIP 教师后继续取得最优平均定位准确率
> - **reusable_ops**: [SAM 统一多类 2D 特征接口, 多视角像素到高斯的几何投影聚合, 2D 伪标签蒸馏到 3D 稀疏卷积网络]
> - **failure_modes**: [2D 教师模型对目标类别或部件完全识别失败时整体上限受限, 3DGS 新视角重建质量差时语义也会随之漂移]
> - **open_questions**: [如何减少对 2D foundation model 语义上限的依赖, 如何在动态/复杂背景中保持更稳的实例与时序一致性]

## Part I：问题与挑战

这篇论文解决的是：**如何在一个 3D 场景上做开放词汇理解**，即用户可以输入自由文本，如“chair”“glass bottle”“shoe sole”，系统要在任意视角下给出一致的 3D 语义响应。

### 1. 真正的难点不在“有没有语义”，而在“语义能否稳定落到 3D 上”
已有方案大致有三类问题：

1. **纯 2D 多视图方法**  
   依赖 CLIP/LSeg/OpenSeg 等 2D foundation model，单张图上能识别，但换视角后常常不一致。  
   本质上，2D 模型缺少跨视角几何约束。

2. **点云方法**  
   有 3D 几何，但点云稀疏，难以支持高质量、稠密、任意视角的语义渲染。

3. **NeRF / 语义 3DGS 联合优化方法**  
   能得到比较稠密的 3D 语义场，但常要**每个新场景都重新训练**，而且 NeRF 推理慢；一些语义 3DGS 方法也要额外训练语义分支，灵活性有限。

### 2. 为什么现在值得做
因为 3D Gaussian Splatting 同时具备两个关键条件：

- **显式表示**：每个 Gaussian 都是可访问、可编辑的实体；
- **渲染足够快**：接近 NeRF 的质量，但推理效率更高。

这意味着可以把问题从“为每个场景训练一个隐式语义场”改写为：

**能不能直接把 2D foundation model 的知识写入现成的 3D 高斯里？**

这正是本文的切入点。

### 3. 输入/输出接口与边界条件
- **输入**：
  - 多视角 RGB 图像
  - 相机内外参
  - 现成的 3DGS 场景
  - 一个或多个 2D 预训练语义模型（LSeg / CLIP / VLPart 等）
- **输出**：
  - 每个 Gaussian 的语义 embedding
  - 基于文本查询的分割、定位、编辑结果
- **边界条件**：
  - 需要相机标定
  - 需要已有 3DGS 重建
  - 语义上限受 2D 教师模型能力约束

## Part II：方法与洞察

方法核心可以概括为两步：

1. **先把 2D 语义投到 3D Gaussians 上**
2. **再训练一个 3D 网络，直接从原始高斯属性预测语义**

这样既保留了 2D foundation model 的开放词汇能力，又引入了 3D 一致性和更快推理。

### 方法主线

#### A. 2D versatile projection：把各种 2D 模型统一成可投影的像素语义
论文不只支持一种 2D 模型，而是试图兼容多种类型：

- **像素级模型**：如 LSeg、OpenSeg
- **实例级模型**：如 GroundingDINO、VLPart
- **图像级模型**：如 CLIP

关键辅助模块是 **SAM**。它的作用不是简单“分割一下”，而是充当一个**统一接口层**：

- 对像素级模型：用 SAM 细化边界；
- 对实例级模型：把框提示转成精细 mask；
- 对图像级模型：先用 SAM 提区域，再让 CLIP 对区域编码。

这样，不同来源的 2D 语义最终都被规整成“每个像素/区域有一个 embedding”的形式。

#### B. 几何投影：把 2D 像素语义写到 3D Gaussian 上
核心不是语义渲染训练，而是**几何对应**。

做法是：

- 用相机参数把 3D Gaussian 投到 2D；
- 通过 3DGS 的深度/遮挡关系，确定某个像素真正对应的是前景表面哪个 Gaussian；
- 一个 Gaussian 可能在多个视角下被看到，于是把这些视角的语义 embedding 做平均池化，得到该 Gaussian 的 `s2D`。

这一步的重要性质是：

- **不需要重训 3DGS**
- **不需要额外语义优化**
- **天然利用多视角一致性修正单视图噪声**

#### C. 3D semantic network：把“投影得到的语义”蒸馏成“3D 直接预测”
投影方案虽灵活，但仍依赖多视图和 2D 模型推理。为此作者又训练了一个 3D 网络：

- backbone：**MinkowskiNet**
- 输入：原始 Gaussian 属性，如颜色、不透明度、协方差等
- 监督：上一步投影得到的 `s2D`
- 输出：每个 Gaussian 的 `s3D`

这个设计的意义不是追求完全独立于 2D 教师，而是把“2D 教师 + 多视角几何聚合”得到的知识，蒸馏成一个**更快、更稳的 3D 语义预测器**。

#### D. 推理与融合
给定文本查询后：

- 用 CLIP text encoder 编码文本；
- 分别与 `s2D` 和 `s3D` 做相似度；
- 如果两者都存在，则取更强响应；
- 再把命中的高斯 splat 到图像上，得到 2D mask 或定位结果。

这使得系统既能做：
- 语义分割
- 物体定位
- 部件分割
- 实例分割
- 文本驱动编辑
- 动态场景跟踪

### 核心直觉

**什么变了？**  
从“为每个场景联合优化语义渲染场”改成“先用几何对应把 2D 语义直接写入显式高斯，再蒸馏出一个 3D 预测头”。

**哪种瓶颈被改了？**  
- 去掉了逐场景语义训练这个优化瓶颈；
- 把单视图 2D 语义的不一致，变成了多视角聚合后的 3D 一致语义；
- 把慢速的逐视角语义推理，压缩成一次 3D 稀疏卷积预测。

**能力上带来了什么变化？**  
- 更快地把开放词汇能力装进 3DGS；
- 更容易更换教师模型（LSeg/CLIP/VLPart）；
- 更适合做任意视角、可编辑的 3D 语义操作。

**为什么这套设计有效？**  
因为作者抓住了一个因果点：  
开放词汇能力本来就在 2D foundation model 里，真正缺的是**把这些语义稳定、可查询地绑定到 3D 几何实体上**。3DGS 提供了显式几何单元，几何投影提供了绑定机制，3D sparse conv 则把这种绑定进一步变成可泛化、可快速推理的 3D 预测能力。

### 战略性 trade-off

| 设计选择 | 带来的好处 | 代价 | 适用场景 |
|---|---|---|---|
| 仅 2D 投影 | 无需重训 3DGS；可直接复用任意 2D 教师；语义精度高 | 依赖多视图和教师推理，速度不如 3D 头 | 离线高质量语义注入 |
| 仅 3D semantic network | 推理快；视角更一致；可泛化到未见场景 | 监督来自伪标签，精度略低于投影版 | 在线推理、部署侧 |
| 2D+3D 融合 | 同时利用高精度投影与 3D 几何先验，效果最好 | 需要先做投影再训练 3D 头，流程更长 | 追求最优效果 |
| 使用 SAM 统一接口 | 能兼容 pixel / instance / image-level 模型 | 增加前处理复杂度，质量受 SAM 区域质量影响 | 多教师组合、部件/长尾查询 |
| 不重训语义 3DGS | 灵活、便宜、可插拔 | 语义上限更依赖外部教师和已有 3DGS 质量 | 多场景快速迁移 |

## Part III：证据与局限

### 关键证据信号

#### 1. 对比信号：在 ScanNet 上，方法真正提升了“3D 一致的开放词汇分割”
最核心结果是 ScanNet 2D 语义分割：

- **Ours 2D+3D**：62.0 mIoU / 77.0 mAcc
- **Feature 3DGS**：59.2 / 75.1
- **LSeg**：56.1 / 74.5

这说明两件事：

1. 仅把 2D 语义搬到 3D 并做多视角聚合，就能超过原始 2D 教师；
2. 在此基础上再加 3D semantic network，能进一步补充几何线索，得到最优结果。

更重要的是，它**接近闭集方法** Panoptic Lifting 的 65.2 mIoU，说明开放词汇设置下已经有较强竞争力。

#### 2. 任务迁移信号：在 LERF 上，灵活更换教师模型是有效的
在长尾物体定位任务上，LSeg 不适合，作者换成 **SAM+CLIP** 后：

- **Ours (CLIP)**：85.2%
- **LangSplat**：84.3%
- **LERF**：73.6%

这不是简单“换 backbone”，而是证明了本文一个关键卖点：  
**它不是绑定某一个 2D 教师，而是一个可插拔的 2D→3D 语义注入框架。**

#### 3. 消融信号：高斯属性本身确实提供了额外 3D 语义信息
把 3D 网络输入从完整 Gaussian 属性退化为 `XYZ+RGB` 后，性能下降到：

- 58.9 mIoU / 74.7 mAcc

而减少 Gaussian 数量到 20%，或输入视角减少到 10%，性能下降相对温和。  
这表明：

- 真正重要的不只是“3D 点的位置”，
- 而是 **高斯的尺度、形状、透明度等显式表示属性**，它们为 3D 语义判断提供了额外线索。

### 1-2 个最值得记住的指标
- **ScanNet**：62.0 mIoU / 77.0 mAcc
- **LERF**：85.2% 平均定位准确率

### 局限性

- **Fails when**: 2D 教师模型对目标类别、长尾实例或细粒度部件完全识别失败时，投影语义会整体失效；当 3DGS 本身在新视角重建不稳、遮挡深度不准或背景实例 ID 不一致时，语义结果会跟着漂移。
- **Assumes**: 需要多视角图像、准确相机参数和现成 3DGS；3D semantic network 依赖投影得到的伪标签监督；系统实际复现还依赖外部预训练模型（LSeg / CLIP / VLPart / SAM）和 3DGS 训练质量。文中实验资源为单张 RTX 4090，且仍需先训练 RGB 3DGS（10k iter）再训练 3D 语义网络（100 epochs）。
- **Not designed for**: 端到端联合优化语义与外观的 3D 表示学习；没有相机几何或没有稳定 3DGS 的场景；强实例一致性要求的复杂背景视频理解；完全摆脱 2D foundation model 的 3D 原生开放词汇理解。

### 可复用组件
这篇论文最值得复用的不是某个具体数值，而是三类操作：

1. **SAM 作为 2D 语义接口统一层**  
   把像素级、实例级、图像级模型统一到同一投影管线。

2. **几何一致的像素→Gaussian 投影**  
   适合把任何 2D dense/region embedding 注入 3DGS，而不是只做开放词汇语义。

3. **“投影伪标签 → 3D 稀疏卷积蒸馏”范式**  
   适合从昂贵、多视角、慢速的教师流程，压缩到快速 3D 推理器。

### 一句话结论
这篇论文的关键贡献不只是“给 3DGS 加语义”，而是提出了一个更实用的路线：**不重训语义场，直接把 2D foundation model 的知识通过几何关系写入显式 3D 高斯，再用 3D 网络把它蒸馏成快速、较一致的场景语义能力。**

![[paperPDFs/Language_Embedding_Segmentation/IEEE_Transactions_on_Circuits_and_Systems_for_Video_Technology_2026/2026_Semantic_Gaussians_Open_Vocabulary_Scene_Understanding_with_3D_Gaussian_Splatting.pdf]]