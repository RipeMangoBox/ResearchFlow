---
title: "Textured Neural Avatars"
venue: CVPR
year: 2019
tags:
  - Video_Generation
  - task/video-generation
  - neural-rendering
  - texture-mapping
  - uv-parameterization
  - dataset/CMU-Panoptic
  - opensource/no
core_operator: 以24个人体部位的可学习纹理图显式存储外观，再由卷积网络从3D骨架姿态预测像素级部位归属与UV坐标进行采样渲染
primary_logic: |
  3D关节在相机坐标系中的骨架栅格图 → 全卷积网络预测每个输出像素的人体部位概率与对应UV坐标，并从可学习部位纹理图中双线性采样、同时得到前景mask → 生成单人新姿态/新视角RGB图像
claims:
  - "在 CMU Panoptic 的 4 个持出视角设置（CMU1/2 × 6/16 cameras）中，本文方法的 SSIM 均高于 V2V 与 Direct，例如 CMU1-16 为 0.919，对比 0.908/0.899 [evidence: comparison]"
  - "在众包两两偏好测试中，本文方法相对 Direct 的胜率为 71%-92%，相对 V2V 的胜率为 50%-56%，始终获得多数偏好 [evidence: comparison]"
  - "从 CMU1-16 预训练模型迁移并用单目视频微调后，本文方法在两位新被试的持出视角测试中分别以 55% 和 65% 的用户偏好超过 V2V [evidence: comparison]"
related_work_position:
  extends: "DensePose (Güler et al. 2018)"
  competes_with: "Video-to-Video Synthesis (Wang et al. 2018); Everybody Dance Now (Chan et al. 2018)"
  complementary_to: "SMPL (Loper et al. 2015); Deep Appearance Models for Face Rendering (Lombardi et al. 2018)"
evidence_strength: moderate
pdf_ref: paperPDFs/Digital_Human_Clothed_Human_Digitalization/CVPR_2019/2019_Textured_Neural_Avatars.pdf
category: Video_Generation
---

# Textured Neural Avatars

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/1905.08776), [Project](https://saic-violet.github.io/texturedavatar/)
> - **Summary**: 这篇工作把“从姿态直接生成人体图像”改成“先预测像素该去人体纹理图哪里取颜色”，用显式纹理缓存外观，从而提升单人 avatar 在新姿态与新视角上的泛化。
> - **Key Performance**: 在 CMU1/2 × {6,16} 的 4 个持出视角设置中，SSIM 全部最佳（如 CMU1-16: 0.919 vs 0.908/0.899）；众包偏好相对 Direct 为 71%-92%，相对 V2V 为 50%-56%。

> [!info] **Agent Summary**
> - **task_path**: 3D骨架姿态（相机坐标系）+ 相机关系 -> 单人新视角/新姿态RGB渲染与前景mask
> - **bottleneck**: 直接 pose-to-image 网络把外观与几何变化纠缠在图像平面里，遇到新视角或少数据时难以保持衣物/身体纹理一致性
> - **mechanism_delta**: 不再直接回归RGB，而是先预测每个像素的人体部位归属与UV坐标，再从可学习的部位纹理图中采样合成
> - **evidence_signal**: 在 CMU Panoptic 持出视角比较中 SSIM 全面领先 Direct/V2V，并在配对用户研究中持续占优
> - **reusable_ops**: [显式人体部位纹理库, pose-to-UV像素坐标预测]
> - **failure_modes**: [尺度明显偏离训练分布时退化, 手脸姿态估计误差导致局部伪影, 光照/视角变化会被平均化]
> - **open_questions**: [如何引入视角与光照相关纹理, 如何降低对DensePose初始化与外部姿态分割预处理的依赖]

## Part I：问题与挑战

这篇论文解决的是**单人、个体定制的全身神经 avatar 渲染**：给定人体在相机坐标系中的 3D 关节姿态，以及目标相机关系，输出该人的 RGB 图像和前景 mask。

真正难点不在“把人画出来”，而在于**如何让外观在姿态变化和视角变化下仍保持一致**。  
已有 pose-to-image / video-to-video 方法在固定相机下可以工作，但一旦切到**训练中没见过的相机视角**，网络必须同时处理：

1. 衣服纹理如何随身体表面移动；
2. 遮挡/自遮挡如何变化；
3. 视角变化后哪些外观应复用、哪些应重建。

纯 2D 图像翻译网络把这些都塞进网络权重里，导致两个问题：
- **泛化差**：新视角下纹理容易漂、糊、错位；
- **数据效率低**：要靠大量视频去“记住”某个人从不同角度长什么样。

**输入/输出边界**：
- **输入**：3D 关节投影成多通道骨架栅格图，每条骨骼带深度信息；
- **输出**：前景 RGB + 前景 mask，最终可与任意背景合成；
- **边界条件**：单人、每人单独训练；训练时需要 3D pose 与前景 mask；测试虽可跨视角/跨姿态，但默认尺度与光照不要偏离训练分布太多。

为什么这件事在 2019 年值得做：因为 DensePose 这类人体 UV 参数化已经可用，终于能把**“表面对应关系”**引入神经渲染，而不必完整走传统 3D mesh + skinning 管线。

## Part II：方法与洞察

这篇工作的设计哲学很明确：**网络不负责“凭空长出外观”，而是负责“预测该去哪里取外观”**。  
它走的是图形学和深度学习之间的中间路线：

- 保留图形学里“几何/纹理解耦”的好处；
- 但不显式重建 3D clothed human mesh。

### 核心直觉

把输出空间从 **RGB 像素** 改成 **人体表面地址（部位 + UV 坐标）**，会改变问题难度。

- **what changed**：从 direct pose→RGB，变成 pose→(part assignment, UV coordinate)→texture sampling。
- **which bottleneck changed**：外观不再存放在网络权重的隐式记忆里，而被放进显式纹理图；网络只需学习姿态/相机到表面坐标的映射。
- **what capability changed**：同一件衣服在不同视角下能复用同一块纹理，因此新视角、少数据条件下更稳。

因果上，衣物和皮肤的颜色在**人体表面坐标系**里比在**图像坐标系**里稳定得多。  
所以论文把高熵任务“生成整张图”降维成低熵任务“预测每个像素对应表面哪里”，这就是它比直接 image translation 更能泛化的原因。

### 方法主线

1. **人体分块**  
   采用 DensePose 风格的人体 24 部位 UV 参数化，每个部位都有一张可学习纹理图。

2. **从姿态预测表面地址**  
   全卷积网络接收骨架栅格图，输出两类结果：
   - 每个像素属于哪个身体部位/背景；
   - 该像素在对应部位纹理图上的 UV 坐标。

3. **采样式渲染**  
   用预测的部位概率和 UV 坐标，从各部位纹理图双线性采样，再加权合成 RGB；背景概率同时给出 mask。

4. **联合学习纹理与映射**  
   训练时同时更新：
   - 姿态→UV 的网络参数；
   - 每个部位的纹理图本身。  
   监督主要来自图像感知损失和 mask 损失。

5. **初始化很关键**  
   作者明确说成功很依赖初始化：
   - 多视频场景：用 DensePose 输出预训练 pose→UV 映射；
   - 少数据场景：从另一个已训练 avatar 迁移；
   - 纹理初值：把训练图像像素按预测 UV 平均回填到纹理图。

### 战略取舍

| 设计选择 | 解决的瓶颈 | 收益 | 代价 |
|---|---|---|---|
| 显式 24 部位纹理图 | 把外观记忆从隐式网络权重转成显式缓存 | 新视角/少数据更稳，纹理一致性更好 | 光照与高光会被平均化 |
| 预测 UV 而非直接 RGB | 降低输出空间复杂度 | 更容易学跨姿态/跨视角对应 | 仍缺少强 3D 几何约束 |
| 不显式建模 3D shape | 避免 mesh 拟合与复杂 rigging | 系统更简洁，端到端学习 | 尺度外推与精细几何较弱 |
| DensePose/迁移初始化 | 解决训练早期无对应关系 | 可从多视角或单视频启动 | 依赖外部模型或源 avatar |

## Part III：证据与局限

### 关键证据

- **比较信号：持出相机 + 持出动作**
  在 CMU Panoptic 的四个设置（CMU1/2 × 6/16 训练相机）上，训练和测试在相机与动作上都不重叠。本文方法的 **SSIM 全部第一**。这直接支持了核心主张：显式纹理分解确实提升了跨视角泛化。

- **因果消融：同架构 Direct baseline**
  作者专门做了一个与本文几乎同架构、同损失、但**直接预测 RGB** 的 Direct baseline。本文优于它，说明收益主要来自“纹理-坐标分解”，而不是网络更大或训练技巧更多。

- **感知证据：用户研究**
  众包配对偏好中，本文方法始终优于 Direct，也略优于 V2V。说明它在“人看起来是否像同一个人、衣服是否连贯”这类主观 realism 上更强。

- **低数据信号：单目视频迁移**
  从 CMU1-16 预训练模型迁移后，只用单个新人的单目视频微调，也能在约 30° 的新视角上超过 V2V。能力跃迁在于：**不再必须依赖多机位大数据**才能得到可用 avatar。

- **反向信号：FID 不总是最好**
  论文也诚实报告：FID 往往不如 V2V。作者解释是本文把不同视角的光照平均到了同一张纹理里。这个负面结果与其建模假设是一致的，也恰好揭示了系统边界。

- **额外分析**
  作者提到，即便给 V2V 加入更强的 DensePose 式条件，指标也没有明显改善；因此关键不只是“条件更密”，而是必须改变生成接口本身。

### 局限性

- **Fails when**: 人物尺度明显超出训练分布；手和脸的姿态估计有误时会出现明显伪影；强视角依赖反射或光照变化下容易失真。
- **Assumes**: 单人场景、每个身份单独训练、训练视频带 3D pose 与前景 mask；通常需要 DensePose 预训练或可迁移的源 avatar；复现还依赖 OpenPose、3D pose lifting、DeepLabv3+、GrabCut 等外部预处理，且论文只给项目页未见代码公开。
- **Not designed for**: 多人交互场景、强动态光照建模、跨身份通用 avatar、需要物理一致 3D 几何的高精度重建任务。

### 可复用组件

- **显式人体 UV 纹理库**：可作为后续 human rendering / avatar 系统的外观缓存层。
- **pose-to-UV 预测头**：把生成问题变成对应预测问题，适合插入更强 backbone。
- **先迁移几何、再重建纹理** 的初始化策略：对少样本个体化 avatar 很有价值。

## Local PDF reference

![[paperPDFs/Digital_Human_Clothed_Human_Digitalization/CVPR_2019/2019_Textured_Neural_Avatars.pdf]]