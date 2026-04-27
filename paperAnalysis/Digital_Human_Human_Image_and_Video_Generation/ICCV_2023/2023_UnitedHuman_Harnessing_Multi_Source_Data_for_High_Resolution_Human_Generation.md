---
title: "UnitedHuman: Harnessing Multi-Source Data for High-Resolution Human Generation"
venue: ICCV
year: 2023
tags:
  - Others
  - task/image-generation
  - gan
  - smpl-alignment
  - continuous-scale-training
  - dataset/SHHQ
  - dataset/DeepFashion
  - dataset/CelebA
  - dataset/DART
  - repr/SMPL
  - opensource/no
core_operator: 用SMPL先验把多源局部/全身人体数据映射到统一全身坐标系，再以连续尺度GAN做patch级尺度不变训练与拼接生成。
primary_logic: |
  多源人体数据（全身/局部、不同分辨率） → 用SMPL驱动的空间变换统一到全身图像空间，并用低分辨率全身模型提供全局结构指导 → 连续GAN按位置与尺度生成并拼接patch → 输出高分辨率全身人体图像
claims:
  - "Claim 1: 在2048px评测下，UnitedHuman 的平均 kFID/pFID 为 19.56/18.94，显著优于 InsetGAN 的 32.50/27.22 和 AnyRes 的 33.12/30.49；在仅使用 10K SHHQHR 全身高分辨率图像时，与使用 100K SHHQHR 的 StyleGAN-Human 的 20.25/18.96 基本持平或略优 [evidence: comparison]"
  - "Claim 2: 基于 SMPL 的 Multi-Source Spatial Transformer 比 keypoint 对齐和 pose-mapping 对齐更有效，可将平均 kFID/pFID 从 22.45/22.88、27.81/25.32 降到 19.56/18.94 [evidence: ablation]"
  - "Claim 3: 全局结构像素约束与带 CutMix consistency 的像素级判别器是关键组件；文中报告相对仅用对抗损失的版本，平均 kFID 与 pFID 分别下降 20.7 和 17.96 [evidence: ablation]"
related_work_position:
  extends: "AnyRes GAN (Chai et al. 2022)"
  competes_with: "StyleGAN-Human (Fu et al. 2022); InsetGAN (Frühstück et al. 2022)"
  complementary_to: "N/A"
evidence_strength: moderate
pdf_ref: paperPDFs/Digital_Human_Human_Image_and_Video_Generation/ICCV_2023/2023_UnitedHuman_Harnessing_Multi_Source_Data_for_High_Resolution_Human_Generation.pdf
category: Others
---

# UnitedHuman: Harnessing Multi-Source Data for High-Resolution Human Generation

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2309.14335), [Project](https://unitedhuman.github.io/)
> - **Summary**: 论文把“高分辨率全身人体生成”中的核心瓶颈归因于训练数据，而非单纯网络容量；它用 SMPL 驱动的多源对齐把脸、手、上/下半身等高分辨率数据统一到全身坐标系，再用连续尺度 GAN 学习任意位置与尺度的局部细节。
> - **Key Performance**: 2048px 评测下平均 **kFID 19.56 / pFID 18.94**，优于 AnyRes 的 **33.12 / 30.49**；**Precision 0.74** 为表中最佳，且仅使用 **10K** 高分辨率全身图像。

> [!info] **Agent Summary**
> - **task_path**: 多源人体图像数据（全身/脸/手/局部身体，分辨率不一） -> 高分辨率全身人体图像
> - **bottleneck**: 全身数据里的脸手像素占比太低，局部高频纹理监督不足；而多源局部数据又在位置/尺度/姿态上不共轴，不能直接联合训练
> - **mechanism_delta**: 先用 SMPL 把多源图像映射到统一全身空间，再用位置-尺度条件的连续 GAN 配合全局结构教师与 CutMix 一致性做训练
> - **evidence_signal**: 2048px 下 kFID/pFID 明显优于 AnyRes/InsetGAN，且 SMPL 对齐消融优于 keypoint 与 pose-mapping
> - **reusable_ops**: [SMPL驱动的局部到全身坐标对齐, CutMix一致性的像素级判别器]
> - **failure_modes**: [继续放大时出现高频伪影与moire, 罕见姿态与服饰多样性不足]
> - **open_questions**: [如何突破StyleGAN3的高频上限, 如何在更弱或无SMPL先验下做多源人体对齐]

## Part I：问题与挑战

这篇论文的判断很明确：**现有全身人体生成方法生成不清楚脸和手，根因主要不是“不会生成”，而是“没见够高质量监督”**。

### 1) 真问题是什么
- 全身人体图像里，脸和手只占很小面积。
- 即便全图分辨率不低，落到这些局部区域上的有效纹理监督仍然有限。
- 因此，模型容易学会“整体像个人”，却学不好“局部真的清晰”。

作者进一步指出，一个真正高质量、同时覆盖**全身结构 + 脸手细节 + 多姿态多服饰**的大规模高分辨率全身数据集很难获得。但现实里，**局部数据很多**：
- 人脸数据集
- 手部数据集
- 上半身/服饰数据集
- 少量全身高分辨率数据

所以机会点不是重新造一个大而全的数据集，而是：**能不能把这些异构数据联合起来训练一个全身生成器？**

### 2) 真瓶颈在哪里
瓶颈有两个，而且都不是简单拼接数据能解决的：

1. **空间错位问题**  
   脸、手、上半身、全身图来自不同数据源，位置、尺度、视角分布完全不同。  
   对刚体物体，2D keypoint 对齐勉强可用；但人体是铰接结构，只有 2D 关键点不足以表达深度、姿态和体型。

2. **尺度不一致问题**  
   GAN 通常在固定分辨率训练，而这里的数据同时来自 256 / 1024 / 2048 等不同尺度。  
   如果生成器只能固定分辨率，就无法自然吸收这些多尺度监督。

### 3) 输入/输出接口与边界条件
- **输入**：多源人体相关数据集，包括低分辨率全身、少量高分辨率全身、以及高分辨率脸/手/局部身体图像。
- **输出**：高分辨率全身人体图像，文中重点展示 1024px 与 2048px。
- **边界条件**：
  - 需要一个相对对齐的全身数据集来定义“全身图像空间”
  - 需要人体几何先验（SMPL）去连接不同局部数据
  - 任务是**无条件全身人体生成**，不是 pose-guided 或 text-guided 生成

### 4) 为什么现在值得做
因为“局部高分辨率数据丰富、全身高分辨率数据稀缺”是现实常态，而连续尺度生成框架已经提供了“按位置和尺度生成 patch”的能力。  
**缺的就是把多源人体数据变成同一监督空间的机制。**

---

## Part II：方法与洞察

作者的方法分成两个核心部件：

1. **Multi-Source Spatial Transformer**：解决“多源数据不共轴”
2. **Continuous GAN**：解决“多尺度监督无法统一吸收”

整体设计哲学是：**先统一几何坐标，再统一尺度训练，最后用低分辨率全身教师守住整体人体结构。**

### 核心直觉

以前的方法失败，不是因为模型完全不会画高质量脸和手，而是因为这些细节在全身图中几乎没有被充分监督；而现成的高分辨率局部数据又不能直接告诉生成器“这个 patch 应该出现在全身的哪里”。

UnitedHuman 做的关键改动是：

- **把多源局部图像先投影到统一的全身坐标系**
- **再让生成器按“位置 + 尺度”去生成对应 patch**
- **同时用低分辨率全身模型提供结构锚点**

这实际改变了三件事：

- **分布变了**：局部数据不再是各自为政的独立图块，而是全身空间中“有位置语义”的监督
- **约束变了**：训练不再只能在固定分辨率进行，而能跨尺度采样
- **能力变了**：模型可以在保住整体人体结构的同时，吸收脸/手等局部高频细节

### 1) Multi-Source Spatial Transformer：把局部数据变成“全身空间里的监督”

作者定义一个**full-body image space**，本质上是一个标准化的全身人体坐标空间。关键做法是用 **SMPL** 作为几何中介。

具体因果链条是：

- 对全身图像，估计 SMPL 的姿态、体型、相机参数，得到稳定的全身坐标参考
- 对局部图像，先做初始回归，再用类似 SMPLify 的优化细化
  - 可见部分：用 2D keypoint 约束
  - 不可见部分：用在全身数据上训练的 pose prior（VAE）约束
  - 再加一个朝向正则，缓解深度歧义
- 最后把局部图像通过变换矩阵映射进统一的全身图像空间

为什么这比 keypoint 更好？  
因为它不仅对齐“点的位置”，还引入了**人体形状、姿态和深度结构**。这对于手臂弯曲、视角变化、局部遮挡的人体非常关键。

### 2) Continuous GAN：把多尺度 patch 监督变成统一生成能力

在生成端，作者基于 **StyleGAN3-T** 构建连续尺度生成器。核心思想不是直接一次输出整张超高分辨率图，而是：

- 用位置 `v` 与尺度 `s` 控制采样
- 让生成器输出对应位置/尺度的固定分辨率 patch
- 最后把多个 patch 拼起来形成高分辨率全身图像

这使得模型天然支持：
- 不同位置的局部生成
- 不同尺度的连续采样
- 多分辨率监督融合

### 3) 两阶段训练：先学结构，再补细节

#### Stage 1：低分辨率全身教师
先在 256×256 全身数据上训练基础模型。  
作用不是追求最终清晰度，而是学到**人体整体拓扑、姿态分布和全局布局**。

#### Stage 2：多源高分辨率细化
再引入多源高分辨率数据：
- SHHQHR 提供少量高分辨率全身
- DeepFashion / SHHQSR 提供上半身与下半身局部
- CelebA 提供脸部细节
- DART 合成手部细节

这里的关键不是“把它们混在一起喂给 GAN”，而是：
- 先经 MST 变到全身空间
- 再按位置/尺度采样 patch
- 用像素约束让高分辨率 patch 不偏离 Stage 1 学到的全局结构

### 4) CutMix consistency：让局部数据也能参与全尺度判别

一个实际难点是：局部数据只覆盖全身图的一部分。  
如果直接拿它们训练判别器，判别器会学到“这只是局部图”，而不是“这是不是合理的人体局部”。

作者的解法是：
- 用 **U-Net 式像素级判别器**
- 把真实局部 patch 和生成的全身 patch 用 mask 做 CutMix
- 再约束“混合图的判别结果”与“判别结果的混合”一致

它的作用很直接：  
**让局部真实图像也能在任意尺度上参与监督，而不要求每个真实样本都覆盖整个全身空间。**

### 战略取舍表

| 设计选择 | 改变了什么约束 | 带来的能力 | 代价/风险 |
| --- | --- | --- | --- |
| SMPL 对齐而非 2D keypoint 对齐 | 从“各数据源坐标不一致”变成“统一全身坐标系” | 局部高分辨率数据能监督正确身体部位 | 依赖 SMPL/相机估计质量，极端遮挡会误对齐 |
| 两阶段训练（低分辨率教师 → 高分辨率细化） | 从“只看局部纹理”变成“先保结构再补细节” | 避免高分辨率 patch 训练破坏整体人体结构 | 教师模型偏差可能限制多样性 |
| 连续尺度 patch 生成 | 从固定分辨率训练变成任意位置/尺度采样 | 能统一吸收 256/1024/2048 的监督 | 更依赖 patch 拼接与尺度一致性 |
| 像素级判别器 + CutMix consistency | 从“局部样本无法全尺度参与判别”变成“局部样本可泛化为全尺度监督” | 局部纹理质量显著提升 | 判别器更复杂，训练成本更高 |

---

## Part III：证据与局限

### 关键证据信号

**信号 1：主比较实验说明“多源数据真的被用起来了”**  
作者没有只看标准 FID，而是重点用更适合局部细节评估的 **kFID / pFID**。这很重要，因为标准 FID 会偏向分布更单一、与测试集更同源的方法。

- UnitedHuman：**平均 kFID 19.56 / pFID 18.94**
- AnyRes：**33.12 / 30.49**
- InsetGAN：**32.50 / 27.22**
- StyleGAN-Human：**20.25 / 18.96**

结论不是“UnitedHuman 绝对碾压所有方法”，而是更细一点：

- 相对 **AnyRes / InsetGAN**：能力跃迁很明显，说明“连续尺度训练 + 多源对齐”确实解决了关键问题
- 相对 **StyleGAN-Human**：在只用 **10K** 高分辨率全身图（对方用 **100K**）时，局部指标已经基本持平或略优，说明它显著降低了对大量高分辨率全身数据的依赖

**信号 2：对齐消融直接证明“SMPL 先验不是装饰件”**  
对齐方式替换为 keypoint 或 pose-mapping 后，性能明显下降：

- Keypoint：平均 kFID / pFID = **22.45 / 22.88**
- Pose-mapping：**27.81 / 25.32**
- SMPL 对齐：**19.56 / 18.94**

这说明对于人体这种铰接结构，只靠 2D 点位或局部 MLP 映射不够，**几何先验是因果关键旋钮**。

**信号 3：数据与损失消融说明“细节提升来自正确监督，而非单纯堆数据”**
- 加入 DeepFashion + SHHQSR 后，整体局部质量显著提升
- 加入 CelebA 后，脸/颈/肩部位 kFID 明显下降
- 加入 DART 后，手部 kFID 进一步改善
- 加入像素结构约束与 CutMix consistency 后，文中报告平均 kFID / pFID 相对纯对抗训练分别下降 **20.7 / 17.96**

这表明论文的收益不是单一来源：
- **数据提供细节**
- **对齐让数据可用**
- **教师与一致性训练让细节不会破坏全局结构**

**补充信号：用户研究与定性结果一致**  
附录中的 12 人用户研究里，作者方法在全图真实性、全图清晰度、局部真实性上获得 **82%+** 投票，局部清晰度也有 **78.33%**，与主实验结论一致。

### 能力跃迁到底体现在哪
相对以往方法，UnitedHuman 的真正跃迁不是“GAN 变强了”，而是：

> **把原本无法直接利用的多源局部高分辨率人体数据，转化为可定位、可跨尺度、可与全身结构兼容的监督。**

这使它在缺少大规模高清全身数据时，仍能生成更清楚的脸和手，并保持全身结构合理。

### 局限性

- **Fails when**: 继续放大到更高分辨率、需要更高频纹理时，StyleGAN3 主干会暴露高频表示上限，出现 circular grid-like moire 或纹理伪影；当训练数据缺少罕见姿态、复杂服饰时，生成多样性也会受限。
- **Assumes**: 需要至少一个全身数据集来定义统一全身空间，并依赖 2D keypoint、SMPL/PARE/SMPLify-P 拟合与 pose prior；训练还依赖多阶段 GAN 训练与较高算力（附录报告约 **43 GPU-days**）。
- **Not designed for**: 文本/姿态可控生成、视频时序一致性、极端遮挡或非标准人体形态场景；它的重点是无条件高分辨率全身人体图像生成，而不是通用人物编辑系统。

### 可复用组件

1. **SMPL 驱动的多源数据统一坐标层**  
   对任何“局部数据丰富、整体数据稀缺”的铰接对象生成任务都很有启发。

2. **低分辨率结构教师 + 高分辨率局部细化**  
   是一种很实用的“先保拓扑、再补高频”的训练范式。

3. **CutMix consistency 的像素级判别器**  
   适合那些真实数据只覆盖目标图像一部分区域的生成任务。

![[paperPDFs/Digital_Human_Human_Image_and_Video_Generation/ICCV_2023/2023_UnitedHuman_Harnessing_Multi_Source_Data_for_High_Resolution_Human_Generation.pdf]]