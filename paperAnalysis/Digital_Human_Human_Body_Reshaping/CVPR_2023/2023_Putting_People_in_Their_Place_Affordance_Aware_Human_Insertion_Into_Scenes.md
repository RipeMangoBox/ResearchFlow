---
title: "Putting People in Their Place: Affordance-Aware Human Insertion Into Scenes"
venue: CVPR
year: 2023
tags:
  - Others
  - task/image-inpainting
  - task/human-insertion
  - diffusion
  - cross-attention-conditioning
  - self-supervision
  - dataset/2.4MVideoClips
  - opensource/no
core_operator: "利用同一视频中跨帧的人物-场景配对训练条件扩散补全模型，在无显式姿态标注下学习符合场景可供性的人体重定姿与插入"
primary_logic: |
  遮挡场景图像 + 参考人物图像 → 通过同人跨帧自监督学习场景可供姿态，并以场景拼接条件约束空间布局、以人物交叉注意力迁移外观 → 输出与场景交互自然的人体插入结果，也可退化为人物/场景幻觉
claims:
  - "视频跨帧自监督优于图像级监督：在条件生成人体插入上，视频训练模型的 FID/PCKh 为 12.103/15.797，优于图像训练模型的 13.174/8.321，说明跨帧姿态变化提供了关键可供性信号 [evidence: ablation]"
  - "采用 CLIP ViT-L/14 人物编码、860M UNet 和 Stable Diffusion 初始化可把条件生成性能提升到 FID 10.078、PCKh 17.602，优于 VAE 特征拼接和从头训练 [evidence: ablation]"
  - "在人物幻觉与场景幻觉上，该方法分别优于 Stable Diffusion：人物幻觉 FID 8.390 vs 19.651、PCKh 23.213 vs 0.023；场景幻觉 FID 20.268 vs 44.687 [evidence: comparison]"
related_work_position:
  extends: "Latent Diffusion Models (Rombach et al. 2022)"
  competes_with: "Stable Diffusion (Rombach et al. 2022); DALL-E 2 (Ramesh et al. 2022)"
  complementary_to: "N/A"
evidence_strength: moderate
pdf_ref: paperPDFs/Digital_Human_Human_Body_Reshaping/CVPR_2023/2023_Putting_People_in_Their_Place_Affordance_Aware_Human_Insertion_Into_Scenes.pdf
category: Others
---

# Putting People in Their Place: Affordance-Aware Human Insertion Into Scenes

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2304.14406), [Project](https://sumith1896.github.io/affordance-insertion)
> - **Summary**: 这篇论文把“场景可供性预测”改写成“把某个人自然地放进场景里”的自监督扩散补全任务，用视频跨帧信号同时学会姿态推断、外观保持和场景融合。
> - **Key Performance**: 条件插入最佳模型达到 **FID 10.078 / PCKh 17.602**；人物幻觉显著优于 Stable Diffusion，**FID 8.390 vs 19.651，PCKh 23.213 vs 0.023**。

> [!info] **Agent Summary**
> - **task_path**: 遮挡场景图像 + 参考人物图像/空人物条件 -> 与场景可供性一致的人体插入图像或人物/场景幻觉结果
> - **bottleneck**: 在没有显式姿态或3D标注时，同时从场景中推断合理姿态、保留参考人物外观，并完成自然融合
> - **mechanism_delta**: 用同一视频中不同帧的“同人异姿”配对做自监督扩散补全，把跨帧姿态变化直接变成可供性学习信号
> - **evidence_signal**: 视频训练优于图像训练的消融 + 对 Stable Diffusion 的人物/场景幻觉显著优势
> - **reusable_ops**: [video-frame-pair-self-supervision, scene-concat-plus-person-cross-attention]
> - **failure_modes**: [bad-faces-and-limbs, lighting-mismatch-or-extreme-pose]
> - **open_questions**: [how-to-improve-high-frequency-human-fidelity, how-to-disentangle-person-from-interacted-objects]

## Part I：问题与挑战

这篇论文研究的不是普通“抠图贴人”，而是**场景可供性驱动的人体插入**：给定一张场景图、一个待插入区域，以及一个参考人物，模型要生成一张新图，使人物既保留原有外观，又以**符合场景语义和物理直觉**的姿态出现在正确位置。

真正难点有三层：

1. **姿态要由场景决定**：沙发边更可能坐，楼梯边需要匹配朝向，空地上可能站立或行走。
2. **外观要由参考人决定**：衣服、体型、局部纹理不能丢。
3. **融合要自然**：尺度、光照、阴影、边界过渡要和背景一致。

作者认为，先前方法的核心瓶颈在于：  
- 要么依赖关键点、3D、语义布局等中间表示，标注重、泛化窄；  
- 要么依赖通用生成模型，它们能“画一个人”，但未必真的学会“这个场景里人应该怎么放”。

这篇论文抓住的真实瓶颈是：**缺少一种大规模、低标注成本、又能直接暴露人-场景交互先验的监督信号**。  
作者的答案是视频：同一个人在同一场景的视频中，会自然呈现不同合理姿态。这个跨帧变化，本身就是可供性监督。

**为什么现在做得动？**  
一方面，latent diffusion 已经把高质量补全做成了可扩展范式；另一方面，互联网视频提供了足够大规模的人体活动数据，可以替代昂贵的显式姿态/3D监督。

**输入/输出边界条件**也很明确：  
- 输入是单张 **256×256** 场景裁剪、一个 mask、以及一个参考人物图；  
- 训练数据限定为**单人视频片段**；  
- 依赖人体检测、分割和关键点过滤构造数据；  
- 输出是 2D 图像级的人体插入，而不是 3D 几何或可执行动作。

## Part II：方法与洞察

整体上，这是一个**条件 latent diffusion inpainting** 模型，但关键不在“用了扩散”，而在**如何构造训练信号**。

训练时，作者从同一个视频中随机采两帧：

- 在第一帧里把人遮掉，得到 **masked scene**；
- 从第二帧里裁出同一个人，得到 **reference person**；
- 目标是恢复第一帧原图。

这个构造非常关键：模型如果想把图补对，就必须同时学会：

- 根据场景上下文推断一个**合理姿态分布**；
- 把参考人物的外观迁移到这个姿态上；
- 让生成的人和背景在视觉上协调。

架构上，作者把两类条件分开处理：

- **场景条件**（masked image + mask）：与目标图空间对齐，直接拼接到 UNet 输入；
- **人物条件**（reference person）：与目标姿态不对齐，所以先编码成 CLIP ViT-L/14 特征，再通过 **cross-attention** 注入。

这种分流是因果上合理的：  
场景负责“人该放哪、怎么摆”，人物负责“长什么样”，两者不应以同一种对齐方式进入模型。

此外，还有三个配套设计：

1. **多样 mask 策略**：框、扩张框、分割 mask、scribble 都用，使模型既能整人插入，也能做局部补全。
2. **仅对参考人物做增强**：打破“同视频帧亮度近似一致”的训练偏差，提升跨场景插入的鲁棒性。
3. **condition dropout + CFG**：随机丢弃人物条件，甚至同时丢人物和场景条件，使同一个模型还能做人物幻觉、场景幻觉和无条件采样。

### 核心直觉

**what changed**  
作者把训练分布从“单帧图像补全”改成了“同一视频里、同一人物、跨帧变化”的配对学习。

**which distribution / constraint / information bottleneck changed**  
- 原先模型缺的是“同一身份、不同姿态、受场景约束”的直接样本；
- 视频天然提供了这类样本，等价于把“姿态变化”从人工标注改成了自监督观测；
- 同时，场景条件与人物条件被解耦，减少了“把参考图原姿态硬拷过去”的风险。

**what capability changed**  
- 从普通补全升级为**可供性驱动的 re-pose + insertion**；
- 从单任务模型升级为**人物插入 / 人物幻觉 / 场景幻觉 / 局部编辑**的统一模型。

之所以有效，不是因为扩散模型“更强”这么简单，而是因为作者找到了一个正确的监督接口：  
**视频跨帧让模型直接观察“一个人如何在场景里合理变化”，于是姿态推断不再需要显式 pose label；cross-attention 又让外观迁移不受目标姿态对齐约束。**

### 战略性取舍

| 设计选择 | 解开的瓶颈 | 带来的能力 | 代价/风险 |
|---|---|---|---|
| 同视频跨帧自监督 | 无显式姿态/3D标注时难学可供性 | 学到场景驱动姿态分布 | 需要海量高质量视频筛选 |
| 场景拼接 + 人物 cross-attention | 场景与人物条件对齐方式不同 | 同时保布局与外观 | 依赖人物编码器质量 |
| 多形状 mask + 人物增强 | 训练/测试分布偏差、编辑粒度单一 | 支持插入、局部补全、换装 | 训练更复杂，数据构造更重 |
| 大 UNet + Stable Diffusion 初始化 | 从头学真实感成本高 | 更好 FID/PCKh，收敛更稳 | 继承 VAE/SD 的人体细节缺陷 |
| condition dropout + CFG | 单模型难覆盖多种生成模式 | 同时支持条件/无条件/幻觉 | guidance 过大时会损伤多样性与质量 |

## Part III：证据与局限

### 关键实验信号

- **信号 1｜数据消融：视频比图像监督更关键。**  
  论文最重要的证据不是“比谁更大”，而是视频监督对任务本身是否必要。结果显示，视频训练明显优于图像训练：图像模型只有静态同帧信息，难以学到“同一个人如何在同场景中换姿态”；视频则直接提供了这种变化，因此 PCKh 提升尤其明显。

- **信号 2｜架构消融：条件注入方式确实影响可供性学习。**  
  使用 CLIP ViT-L/14 编码参考人物、用 cross-attention 注入，优于直接把 VAE 特征拼接进去；更大的 860M UNet 与 Stable Diffusion 初始化也进一步提升到 **FID 10.078 / PCKh 17.602**。这支持了论文的核心判断：**非对齐的人物条件不该按空间特征硬拼接，而应作为语义外观条件注入。**

- **信号 3｜对比实验：不仅会补图，还学到了人-场景组合先验。**  
  在人物幻觉上，作者相对 Stable Diffusion 的提升非常大：**FID 8.390 vs 19.651，PCKh 23.213 vs 0.023**。这说明模型不是仅仅会“在空白处画纹理”，而是真的更会决定“这个场景里应出现怎样的人体”。  
  在场景幻觉上，**FID 20.268 vs 44.687**，说明从人物反推场景关系也被部分学到了。

- **信号 4｜CFG 分析：质量和多样性的权衡符合扩散模型规律。**  
  适度 guidance 会改善 realism 和 pose correctness，但 guidance 太高又会损伤图像质量。这说明模型的收益不是偶然的 prompt 工程，而是稳定的生成分布变化。

### 能力跳跃到底在哪里

相较于传统 **pose-guided human synthesis**，这篇工作不需要目标姿态显式给定；  
相较于通用 inpainting / text-to-image 模型，它能从**场景本身**推断“人应该怎么站、坐、靠”。  

所以它的真正贡献，不只是“生成质量更好”，而是把**可供性预测**从抽象识别问题，变成了一个**可直接视觉验证、也可定量评估**的生成问题。

### 局限性

- Fails when: 需要高频人体细节时容易出现坏脸、坏四肢；参考人与场景光照差异过大时融合失败；极端姿态时模型可能保留原姿态而不是重定姿；参考人物若携带明显可见物体，模型会把物体也一并“插入”；训练数据中的运动模糊也会导致生成发糊。
- Assumes: 单人视频裁剪场景；依赖 Keypoint R-CNN、OpenPose、Mask R-CNN 进行数据筛选与掩码构造；训练使用 2.4M 视频片段，其中包含 proprietary datasets；训练规模为 32×A100、100K steps，并依赖 Stable Diffusion/VAE 初始化；论文提供项目页，但文中未见代码/权重链接，复现门槛偏高。
- Not designed for: 多人交互插入、严格 3D/物理一致性建模、高分辨率身份保真编辑、精确可控的空间布局约束，或直接面向机器人执行级的 affordance grounding。

### 可复用组件

- **同人跨帧自监督配对**：适合迁移到其他“外观恒定、姿态变化”的人-场景学习任务。
- **双路条件注入**：空间对齐条件走拼接，非对齐外观条件走 cross-attention。
- **多 mask 训练**：让一个模型同时覆盖整人插入、局部补全、换装等编辑形式。
- **条件 dropout**：把条件生成和幻觉生成统一到一个扩散模型里。

## Local PDF reference

![[paperPDFs/Digital_Human_Human_Body_Reshaping/CVPR_2023/2023_Putting_People_in_Their_Place_Affordance_Aware_Human_Insertion_Into_Scenes.pdf]]