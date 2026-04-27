---
title: "PSGAN: Pose and Expression Robust Spatial-Aware GAN for Customizable Makeup Transfer"
venue: CVPR
year: 2020
tags:
  - Others
  - task/makeup-transfer
  - task/image-to-image-translation
  - gan
  - cross-image-attention
  - feature-modulation
  - dataset/MT
  - dataset/Makeup-Wild
  - opensource/no
core_operator: 将参考妆容蒸馏为逐像素γ/β妆容矩阵，并用结合关键点相对位置与外观的跨图注意力将其形变到源脸后再做空间特征调制
primary_logic: |
  源人脸 + 参考妆容人脸 → MDNet提取参考妆容的空间化γ/β矩阵 → AMM借助解析图与关键点计算跨图像素对应并得到适配源脸的γ′/β′ → MANet对源特征逐像素缩放/平移 → 输出保留身份且支持局部/浓淡控制的妆容迁移结果
claims:
  - "PSGAN在人工偏好评测中显著优于对比方法，在MT测试集与Makeup-Wild测试集上分别获得61.5%与83.5%的best-choice比例 [evidence: comparison]"
  - "去掉AMM后，在大姿态或表情差异下会出现刘海迁移到皮肤、唇妆迁移到牙齿等错位现象，而完整模型可纠正这些错误 [evidence: ablation]"
  - "空间感知的妆容矩阵允许通过区域mask与线性插值实现局部妆容迁移和浓淡连续控制，而无需重新训练 [evidence: case-study]"
related_work_position:
  extends: "AdaIN (Huang and Belongie 2017)"
  competes_with: "BeautyGlow; LADN"
  complementary_to: "PSPNet (Zhao et al. 2017); MTCNN (Zhang et al. 2016)"
evidence_strength: moderate
pdf_ref: paperPDFs/Digital_Human_Image_Based_Virtual_Try_On/CVPR_2020/2020_PSGAN_Pose_and_Expression_Robust_Spatial_Aware_GAN_for_Customizable_Makeup_Transfer.pdf
category: Others
---

# PSGAN: Pose and Expression Robust Spatial-Aware GAN for Customizable Makeup Transfer

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/1909.06956)
> - **Summary**: 这篇工作把妆容迁移从“整脸风格翻译”改成“先做跨脸局部对齐、再做逐像素妆容注入”，因此在姿态/表情不一致时仍能较稳定地迁移，并支持局部部位与浓淡可控。
> - **Key Performance**: AMT人工偏好 best-choice 比例：MT 测试集 61.5%，Makeup-Wild 测试集 83.5%。

> [!info] **Agent Summary**
> - **task_path**: source人脸图像 + reference妆容人脸图像 -> 保持源身份的人脸妆容迁移结果
> - **bottleneck**: 现有方法把妆容压成全局低维表示，既丢失局部空间信息，也无法处理源脸与参考脸在姿态/表情上的像素错位
> - **mechanism_delta**: 用MDNet提取空间化γ/β妆容矩阵，再用解析图和关键点约束的跨图注意力AMM把参考妆容形变到源脸坐标系后注入MANet
> - **evidence_signal**: 双测试集人工偏好显著领先，且AMM消融直接暴露“唇妆到牙齿/头发到皮肤”的区域错配
> - **reusable_ops**: [spatial gamma-beta modulation, landmark-and-parsing guided cross-image attention]
> - **failure_modes**: [face parsing或68点关键点检测错误会放大区域错配, 对遮挡和极端姿态的鲁棒性未被系统验证]
> - **open_questions**: [如何在视频中显式建模时间一致性而非逐帧处理, 如何减少对外部解析与关键点检测器的依赖]

## Part I：问题与挑战

这篇论文解决的是**参考妆容到源人脸的可控迁移**：输入是源脸 `x` 和参考妆容脸 `y`，输出需要同时满足两件事——**保留源人的身份与面部结构**，以及**呈现参考图的妆容风格**。

真正难点不在“把颜色换一下”，而在两个更底层的瓶颈：

- **空间信息瓶颈**：很多早期方法把妆容编码成全局或低维 style code。这样虽然能做“整脸变妆”，但口红、眼影、底妆这些本来就高度局部的外观信息会被压扁，结果是不容易做**局部转移**，也不容易做**浓淡连续控制**。
- **跨脸对齐瓶颈**：现有方法大多在正脸、中性表情数据上表现较好，默认“同一空间位置≈同一语义区域”。但在真实应用里，参考图和源图往往会有**转头、张嘴、微笑、尺度差异**。这时若仍按绝对位置转移，口红可能跑到牙齿，头发或阴影可能被误迁移到皮肤。

为什么现在要解决它：美妆试妆应用已经从规整照片走向 **in-the-wild 自拍与视频**。如果方法不能处理姿态/表情差异，也不能让用户只改嘴唇或调浅一点，那么落地价值就很有限。

边界条件也很明确：

- 训练是**无配对**的，不要求 source/reference/transferred 三元组；
- 主要围绕**眼、唇、皮肤**三类区域建模；
- 依赖外部**人脸解析**与 **68 点关键点**；
- 目标是裁剪后的人脸图像，不是全图场景级编辑。

## Part II：方法与洞察

### 方法总览

PSGAN 的核心不是“把两张图直接喂给 GAN”，而是把妆容拆成一个**可对齐、可编辑、可注入**的中间表示。整体分三步：

1. **MDNet（Makeup Distill Network）**  
   从参考图中提取两个与特征图同空间尺度的妆容矩阵 `γ` 和 `β`。可以把它理解为：参考妆容不再是一个抽象向量，而是一个**逐像素的妆容参数场**。

2. **AMM（Attentive Makeup Morphing）**  
   因为源脸和参考脸姿态/表情可能不同，`γ/β` 不能直接套上去。AMM 用**关键点相对位置 + 外观特征 + 人脸解析区域约束**计算跨图注意力，把参考妆容参数变形成适配源脸的 `γ′/β′`。

3. **MANet（Makeup Apply Network）**  
   在源脸的瓶颈特征上执行一次逐像素的缩放/平移调制，把 `γ′/β′` 注入进去，再由解码器生成最终结果。

### 核心直觉

**核心变化**：从“把妆容当成全局风格”改成“把妆容当成空间化参数场”；从“按绝对位置迁移”改成“按语义对应关系迁移”。

这带来了三层因果变化：

1. **表示层变化**  
   全局 latent code → 空间化 `γ/β` 参数场  
   结果：口红、眼影、粉底的二维分布被保留下来，模型不再只能做整脸粗粒度翻译。

2. **约束层变化**  
   绝对坐标对齐 → 基于相对关键点与同区域约束的跨图对齐  
   结果：当源/参考脸存在姿态、表情、尺度差异时，妆容仍能落到语义正确的位置。

3. **能力层变化**  
   不可解释 style mixing → 可直接在参数场上做 mask 与插值  
   结果：自然获得**局部迁移**、**浓淡控制**、**多参考混合**。

为什么这套设计有效：

- **关键点相对位置**提供了更稳定的“脸内坐标系”，比绝对像素位置更抗姿态和尺度变化；
- **解析图约束同区域匹配**，避免嘴唇去关注皮肤、眼影去关注背景；
- **少量外观相似度**用于细化匹配，如避开鼻孔、边界和背景，但又不会主导注意力；
- 在**瓶颈特征**上做 modulation，而不是直接 warping 像素，能更好地保留源脸的身份结构，只改变妆容外观统计。

### 策略取舍

| 设计选择 | 解决的瓶颈 | 能力提升 | 代价/风险 |
|---|---|---|---|
| 空间化 `γ/β` 妆容矩阵 | 全局style code丢失局部信息 | 支持局部迁移、浓淡插值、多参考混合 | 比向量表示更依赖稳定的空间特征 |
| AMM跨图注意力 | 姿态/表情不对齐导致区域错位 | 口红/眼影更容易落在正确位置 | 依赖关键点与解析质量，且有额外计算开销 |
| 同区域解析约束 | 语义混淆 | 降低唇→皮肤、眼→背景误迁移 | 解析类别有限，解析错误会直接传导 |
| 瓶颈处一次性feature modulation | 保身份与改妆容之间的冲突 | 结构保持更好，实现简单 | 对特别复杂的多尺度妆容细节可能表达不足 |

一个很实用的点是：**可控性不是额外训练出来的，而是表示本身带出来的**。  
因为妆容已经是空间参数场，所以：

- 给某区域加 mask，就能做**局部迁移**；
- 给参数乘一个 `α∈[0,1]`，就能做**浓淡控制**；
- 对不同参考的参数场线性混合，就能做**多参考妆容插值**。

## Part III：证据与局限

### 关键证据信号

- **对比信号：优势在“更难的非对齐场景”上被放大**  
  在 MT 测试集上，PSGAN 的 AMT best-choice 比例为 **61.5%**；在 Makeup-Wild 上达到 **83.5%**。对比 BeautyGAN 分别只有 **32.5%** 和 **13.5%**。  
  这说明 PSGAN 的收益不是一般 GAN 提升，而是正中论文声称的核心瓶颈：**姿态/表情不对齐**。

- **消融信号：AMM 是真正的关键旋钮**  
  去掉 AMM 后，论文示例里会出现**刘海被迁移到皮肤、唇妆跑到牙齿**等错位。  
  这直接说明：问题核心不是生成器更深，而是有没有一个**跨脸、跨姿态的像素级对应机制**。

- **机制信号：几何为主、外观为辅的注意力设计是必要的**  
  只用相对位置时，注意力图近似二维高斯，容易越过面部边界或覆盖鼻孔；视觉特征权重过大时，又会变得散乱。  
  这支持 AMM 的设计判断：**几何决定大方向，外观只做局部修正**。

- **功能信号：控制接口是“真可用”的，不是口头宣称**  
  论文展示了只转口红、把口红来自参考1而其他妆容来自参考2、以及单参考/双参考插值的连续结果，说明空间化妆容表示确实支持**局部与强度级控制**。

- **视频信号：有应用潜力，但证据还不强**  
  作者将方法逐帧用于视频，视觉上结果较稳定；但没有专门的时序一致性模块，也没有视频指标，因此这部分更多是**案例展示**。

### 1-2 个关键指标

- **AMT best-choice 比例**：MT `61.5%`，Makeup-Wild `83.5%`
- **相对 BeautyGAN 的提升**：MT `+29.0` 个百分点，Makeup-Wild `+70.0` 个百分点

### 局限性

- **Fails when**: 人脸解析或 68 点关键点检测失败、头发/饰品严重遮挡五官、极端姿态导致脸内相对坐标失真时，AMM 仍可能把妆容分配到错误区域；论文也未系统评估极端光照、低分辨率和严重遮挡。
- **Assumes**: 依赖外部人脸解析与关键点检测器；主要只对眼、唇、皮肤 3 个区域建模；训练/测试基于裁剪后的人脸图像；核心量化证据主要来自人工偏好而非标准客观指标；论文文本未给出代码链接，复现需要自行实现整套预处理与训练流程。
- **Not designed for**: 全图场景级风格迁移、物理真实的化妆渲染、无解析/无关键点的端到端编辑、显式视频时序一致性建模。

### 可复用组件

- **spatial gamma-beta modulation**：把“可编辑属性”表示成空间参数场，再注入生成器特征；
- **landmark-relative cross-image attention**：利用相对关键点坐标建立跨实例语义对应；
- **mask/interpolation control interface**：在参数场层面做区域开关与强度混合，适合其他精细人脸编辑任务。

## Local PDF reference

![[paperPDFs/Digital_Human_Image_Based_Virtual_Try_On/CVPR_2020/2020_PSGAN_Pose_and_Expression_Robust_Spatial_Aware_GAN_for_Customizable_Makeup_Transfer.pdf]]