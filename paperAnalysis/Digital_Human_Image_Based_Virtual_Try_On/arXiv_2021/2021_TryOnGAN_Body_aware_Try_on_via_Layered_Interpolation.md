---
title: "TryOnGAN: Body-aware Try-on via Layered Interpolation"
venue: arXiv
year: 2021
tags:
  - Others
  - task/virtual-try-on
  - stylegan
  - latent-optimization
  - pose-conditioning
  - dataset/Street2Shop
  - opensource/no
core_operator: "在姿态条件 StyleGAN2 中，通过分割引导的逐层连续潜变量插值优化，把目标人物的身份与体型和来源服装的形状纹理分层融合。"
primary_logic: |
  人物图像与服装图像（真实图像需先投影到 Z+）+ 目标人物姿态 → 姿态条件 StyleGAN2 同时生成 RGB 与人体/服装分割，并按层按通道优化插值系数以满足局部编辑、服装保持与身份保持约束 → 输出保留目标人物体型、肤色与身份且换上来源服装的高分辨率试穿图
claims:
  - "Claim 1: 在 800 个真实图像试穿结果上，TryOnGAN 的 FID 为 32.21、ES 为 0.32，优于 ADGAN 的 66.82/0.22 与 CP-VTON 的 87.0/0.27 [evidence: comparison]"
  - "Claim 2: 在 41 名参与者、6 组试穿样例的成对偏好测试中，TryOnGAN 获得 62.6% 的用户偏好，高于 ADGAN 的 31.3% 和 CP-VTON 的 6.1% [evidence: comparison]"
  - "Claim 3: 消融表明 localization、garment、identity 三个损失分别决定局部编辑约束、服装保真和身份保持；同时，连续逐层优化比 greedy 插值更能保留袖长与纹理细节 [evidence: ablation]"
related_work_position:
  extends: "StyleGAN2 (Karras et al. 2020)"
  competes_with: "ADGAN (Men et al. 2020); CP-VTON (Wang et al. 2018a)"
  complementary_to: "Image2StyleGAN++ (Abdal et al. 2020); In-domain GAN inversion (Zhu et al. 2020)"
evidence_strength: moderate
pdf_ref: paperPDFs/Digital_Human_Image_Based_Virtual_Try_On/arXiv_2021/2021_TryOnGAN_Body_aware_Try_on_via_Layered_Interpolation.pdf
category: Others
---

# TryOnGAN: Body-aware Try-on via Layered Interpolation

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2101.02285)
> - **Summary**: 这篇工作把虚拟试衣从“配对监督下的显式服装 warping”改写为“姿态条件 StyleGAN2 潜空间中的逐层插值优化”，从而在无配对数据下更好地同时保留目标人物的体型、肤色与身份，并迁移来源服装的形状与纹理。
> - **Key Performance**: 真实图像试穿上 FID 32.21、ES 0.32，优于 ADGAN 的 66.82/0.22 与 CP-VTON 的 87.0/0.27；用户偏好 62.6%。

> [!info] **Agent Summary**
> - **task_path**: 人物图像 + 服装图像（服装穿在另一人身上，非配对） -> 目标人物穿上该服装的高分辨率单张图像
> - **bottleneck**: 无配对设置下，服装区域与来源人物的姿态、体型、肤色和局部遮挡强耦合，导致“只换衣服、不改坏人”很难
> - **mechanism_delta**: 用姿态条件 StyleGAN2 + 分割分支构造可编辑生成先验，再对每层每通道的插值系数做连续优化，而不是依赖固定 warping 或离散 greedy 混合
> - **evidence_signal**: 真实图像上 FID/ES 与用户偏好均优于 ADGAN、CP-VTON，且消融表明逐层优化和三个损失项都关键
> - **reusable_ops**: [pose-conditioned StyleGAN2, segmentation-guided edit localization, per-layer latent interpolation optimization]
> - **failure_modes**: [real-image inversion 会丢失高频纹理导致细节转移不足, 罕见姿态或少见服装细节会造成形状与纹理错配]
> - **open_questions**: [更强的 GAN inversion 能否把生成图像上的细节上限迁移到真实图像, 该方法能否扩展到更多服装类别与更复杂场景]

## Part I：问题与挑战

这篇论文解决的是一个比“换纹理”更难的虚拟试衣问题：输入不是干净的商品服装图，而是**目标人物图像**和**另一人穿着该服装的图像**；输出则是目标人物穿上该服装后的写实图像，且输出姿态跟随目标人物。

### 真正难点是什么
真正的瓶颈不在于“把衣服颜色贴过去”，而在于：

1. **服装与人是耦合的**  
   来源服装图里同时包含了另一人的体型、姿态、皮肤、头发、边界阴影与遮挡。你想迁移的是衣服，但不想迁移这些无关因素。

2. **跨体型/跨姿态时必须重新合成，而不只是 warping**  
   例如长袖换短袖、宽肩换窄肩、不同胸腰比之间的转移，系统需要生成新的手臂可见区域、调整领口与衣摆，而不是简单几何拉伸。

3. **现有 VTON 常依赖 paired data**  
   许多方法需要同一人/同一件衣服的成对监督数据，现实里难以大规模采集，也会让模型更偏向“拟合训练配对模式”，而不是学会真正的语义重组。

### 为什么现在值得做
作者抓住了一个时机：  
- **StyleGAN2** 已经证明了高分辨率、强生成先验；
- **GAN inversion** 让真实图像也能进入该潜空间；
- 局部 GAN 编辑工作表明，**不同层可以承载不同语义粒度**。

所以，try-on 不一定非要走“先分割/再 warping/再修补”的显式流水线，也可以变成：**在一个会生成人体与服装的先验模型内部，寻找最合适的分层混合方式**。

### 输入/输出与边界条件
- **输入**：目标人物图像 + 来源服装图像
- **输出**：目标人物穿上来源服装的 512×512 图像
- **姿态约束**：输出姿态跟随目标人物
- **训练设置**：无配对训练
- **当前覆盖**：女性、上衣和裤装
- **真实图像额外步骤**：必须先做 latent projection/inversion

---

## Part II：方法与洞察

方法可以概括成两步：  
**先训练一个可控的人像服装生成先验，再在这个先验里做按层插值优化。**

### 方法主线

#### 1. 姿态条件 StyleGAN2
作者把 StyleGAN2 改成了一个**姿态条件生成器**：
- 输入目标人物的 2D pose heatmap；
- 不再使用原始常数输入，而是用 pose encoder 提供多分辨率姿态特征；
- 这样做的目的，是把**pose 从 appearance/style 中剥离出来**。

因果上看，这一步解决的是：  
如果 pose 不独立控制，那么换衣服时就会顺便把来源人物的姿态也带进来，导致“换衣”变成“换人+换姿势”。

#### 2. 分割分支
生成器不只输出 RGB，还额外输出**人体/服装分割**。

这不是一个附属装饰，而是整个优化能成立的关键：
- 试穿结果在每次迭代都在变化；
- 若没有当前输出的分割，就很难知道“哪里是该改的衣服区域，哪里是不该动的人脸/头发/皮肤区域”。

因此分割分支同时带来两件事：
- 改善潜空间解耦；
- 让优化阶段不必依赖外部 parser 对每一步结果反复推理。

#### 3. 逐层潜变量插值优化
给定人物图和服装图的 latent code，作者不是直接前向生成，而是学习一个**每层、每通道的连续插值系数**。

直观上，它在问：  
> 对生成器的哪几层、哪些通道，应该更像“人物图”；又有哪些部分，应该更像“服装图”？

这比统一插值或离散 greedy 选择更细，因为：
- 粗层更影响轮廓、体型适配、袖长等结构；
- 细层更影响纹理、边界、图案和高频细节。

#### 4. 三个约束损失
优化目标不是“随便混出一个像样图”，而是被三类约束夹住：

- **Localization loss**：限制编辑集中在目标服装语义区域，避免把裤子颜色泄漏到背景或皮肤上。
- **Garment loss**：让结果中的服装区域尽量保留来源服装的形状和外观。
- **Identity loss**：用 face/hair 区域约束目标人物身份不被破坏。

#### 5. 真实图像路径
对于真实照片，先把人物图和服装图各自投影到 `Z+`，再运行同样的 try-on 优化。  
这让方法能从“生成图上的潜空间编辑”扩展到真实图像，但也把性能上限部分交给了 inversion 质量。

### 核心直觉

作者真正改动的不是一个局部模块，而是**问题表述方式**：

- **从什么变成什么**：  
  从“显式 warping + 配对监督”  
  变成“在姿态解耦的生成先验中，做分割感知的逐层连续插值优化”。

- **改变了哪个瓶颈**：  
  - pose-conditioning 改变了**姿态与外观纠缠**这个瓶颈；  
  - segmentation branch 改变了**当前输出不可定位**这个瓶颈；  
  - per-layer continuous interpolation 改变了**只能粗糙整图混合**这个瓶颈。

- **带来了什么能力变化**：  
  系统不再只是复制衣服纹理，而是能在新体型/新姿态下，保留目标人物的身体结构和身份，同时重建合适的衣服轮廓、边界，甚至在短袖替换长袖时生成新的手臂可见区域。

### 为什么这种设计有效
因为 try-on 的本质不是单一像素变换，而是**多尺度语义借用**：
- 低分辨率层决定“衣服该长什么形、怎么贴合身体”；
- 高分辨率层决定“边缘、纹理、图案该长什么样”。

逐层优化允许模型在不同尺度上分别决定“借谁更多”。  
这就是它能比固定规则插值、二值层选择或单次前向 warping 更自然的原因。

### 战略权衡表

| 设计选择 | 带来的收益 | 代价/风险 |
|---|---|---|
| 姿态条件 StyleGAN2 | 将 pose 与 style 解耦，支持把来源服装重定位到目标人物姿态 | 依赖姿态估计质量，对训练分布外姿态敏感 |
| 生成器内置分割分支 | 优化时可直接知道当前输出的语义区域，避免外部 parser 回路 | 若分割质量不稳，会直接误导编辑定位 |
| 连续的逐层逐通道插值优化 | 能细粒度控制结构与纹理的借用，比 greedy 更灵活 | 每对图都要迭代优化，速度慢，难实时 |
| 真实图像先做 Z+ 投影 | 让真实照片也能用同一套优化框架 | inversion 误差会吞掉高频纹理，成为主瓶颈 |
| 无配对训练 | 避免昂贵 paired data，覆盖更自然的数据采集方式 | 更依赖生成先验质量，也更容易受训练分布偏置影响 |

---

## Part III：证据与局限

### 关键证据信号

1. **真实图像定量比较：作者的核心主张是成立的**  
   在 800 个真实试穿结果上：
   - **FID 32.21**，优于 ADGAN 的 66.82 和 CP-VTON 的 87.0  
   - **Embedding Similarity 0.32**，优于 ADGAN 的 0.22 和 CP-VTON 的 0.27  

   这两个信号共同说明：  
   TryOnGAN 不只是“更像真图”，也更能保留服装语义相似性。

2. **用户偏好与自动指标一致**  
   41 人成对偏好实验中，TryOnGAN 获得 **62.6%** 偏好，显著高于 ADGAN 的 31.3% 和 CP-VTON 的 6.1%。  
   这说明优势不是只存在于某个自动指标里，而是人眼可见的。

3. **消融支持“因果旋钮”确实有效**  
   - 去掉 localization：编辑会污染衣服外区域  
   - 去掉 garment loss：服装形状/纹理保真下降  
   - 去掉 identity loss：脸和头发更容易漂移  
   - 用 greedy search 代替连续逐层优化：袖长、边界、纹理更差

   这类消融比单纯的结果图更重要，因为它说明论文真正有效的不是“大模型本身”，而是**分层插值优化这个因果旋钮**。

4. **生成图像实验揭示了方法上限，但也暴露真实图像瓶颈**  
   在生成图像上，方法可以迁移更复杂的图案、纹理、按钮、口袋等细节；  
   在真实图像上，这些高频信息会被 projection 丢掉一部分。  
   也就是说：**优化算法本身比真实图像结果表现出来的更强，真正拖后腿的是 inversion。**

5. **仍然与真实照片存在明显差距**  
   论文给出的真实图像 FID 参考值是 **11.83**。  
   TryOnGAN 虽然优于 baseline，但离真实照片分布仍有不小差距，因此“摄影级不可分辨”还不能算完全达成。

### 局限性
- **Fails when**: 遇到训练集中少见的姿态、少见服装细节，或真实图像 inversion 质量差时，容易出现袖长错误、纹理丢失、服装轮廓失真和局部伪影。
- **Assumes**: 需要外部姿态估计与人体分割预处理；依赖 pose-conditioned StyleGAN2 的高质量训练；真实图像必须先投影到 `Z+`；当前主要面向女性上衣/裤装、单人 512×512 图像分布。
- **Not designed for**: 多件服装同时编辑、复杂遮挡和复杂背景场景、商品平铺图直接试穿、物理级布料模拟、SKU 级完全精确纹理复原。

### 资源与复现约束
这篇论文的可复现性有几个现实门槛：
- 训练用了 **8 张 Tesla V100，25M iterations，12 天**
- 每对图像的 try-on 优化约 **224.86s**
- 真实图像 projection 还需约 **227.77s**
- 依赖外部 **PoseNet** 与 **Graphonomy**
- 论文中未提供公开代码与公开命名训练集

所以它更像一个**高质量但偏离线的研究原型**，而不是可直接部署的实时系统。

### 可复用组件
尽管整体系统较重，但里面有几块很值得复用：
- **pose-conditioned StyleGAN2**：适合需要显式姿态控制的人体生成任务
- **segmentation-guided edit localization**：适合任何需要“只改某个语义区域”的 GAN 编辑任务
- **per-layer continuous latent interpolation**：适合把“语义融合”建模成优化问题，而非固定混合规则
- **generated-vs-real split evaluation**：能把“方法本身能力”与“inversion 瓶颈”分开诊断

![[paperPDFs/Digital_Human_Image_Based_Virtual_Try_On/arXiv_2021/2021_TryOnGAN_Body_aware_Try_on_via_Layered_Interpolation.pdf]]