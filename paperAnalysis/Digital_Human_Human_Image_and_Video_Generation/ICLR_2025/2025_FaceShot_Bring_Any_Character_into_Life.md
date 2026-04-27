---
title: "FaceShot: Bring Any Character into Life"
venue: ICLR
year: 2025
tags:
  - Video_Generation
  - task/video-generation
  - diffusion
  - landmark-matching
  - coordinate-retargeting
  - dataset/CharacBench
  - dataset/RAVDESS
  - opensource/promised
core_operator: "用扩散特征语义对应先为任意角色找出稳定面部关键点，再以全局/局部坐标重定向把驱动视频表情迁移到该角色上。"
primary_logic: |
  参考角色图像 + 驱动视频 → 外观引导的扩散特征关键点匹配得到角色初始关键点 → 全局/局部坐标系重定向生成时序关键点序列 → 输入预训练关键点驱动动画模型输出角色动画视频
claims:
  - "Compared with MOFA-Video on CharacBench, FaceShot improves ArcFace from 0.695 to 0.848 and reduces Point-Tracking from 14.985 to 6.935 [evidence: comparison]"
  - "FaceShot lowers landmark NME to 8.569 on CharacBench, outperforming DIFT (11.448), Uni-Pose (13.731), and STAR (24.530) in landmark localization [evidence: ablation]"
  - "FaceShot's coordinate-based retargeting achieves lower Point-Tracking error (6.935) than Deep3D (8.282), Everything's Talking (8.382), and FreeNet (8.272) [evidence: ablation]"
related_work_position:
  extends: "DIFT (Tang et al. 2023)"
  competes_with: "X-Portrait (Xie et al. 2024); LivePortrait (Guo et al. 2024)"
  complementary_to: "MOFA-Video (Niu et al. 2024); AniPortrait (Wei et al. 2024)"
evidence_strength: moderate
pdf_ref: "paperPDFs/Digital_Human_Human_Image_and_Video_Generation/ICLR_2025/2025_FaceShot_Bring_Any_Character_into_Life.pdf"
category: Video_Generation
---

# FaceShot: Bring Any Character into Life

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2503.00740), [Project](https://faceshot2024.github.io/faceshot/)
> - **Summary**: 这篇论文把“角色动画做不好”的主因重新定位为**关键点控制信号失真**，并提出一个免微调的前端插件：先用扩散特征做外观引导关键点匹配，再用坐标式重定向传递表情，从而让 emoji、玩具、动物、动漫角色也能被现有肖像动画模型稳定驱动。
> - **Key Performance**: 在 CharacBench 上，相比 MOFA-Video，ArcFace 从 **0.695→0.848**，Point-Tracking 从 **14.985→6.935**；用户偏好 Overall 达 **8.27**。

> [!info] **Agent Summary**
> - **task_path**: 参考角色图像 + 驱动视频 -> 角色动画视频
> - **bottleneck**: 非人角色上的关键点检测与重定向被人脸先验锁死，导致下游生成器收到错误控制信号
> - **mechanism_delta**: 把“先做人脸关键点检测”改成“先用扩散语义匹配找角色关键点，再用局部坐标系统传递动作”
> - **evidence_signal**: CharacBench 全面对比 + 关键点 NME/Point-Tracking 消融同时支持该因果链
> - **reusable_ops**: [diffusion-feature landmark matching, global-local coordinate retargeting]
> - **failure_modes**: [五官拓扑异常或缺失时匹配不稳, 大姿态/强遮挡超出2D局部坐标假设时重定向易失真]
> - **open_questions**: [能否摆脱预标注 appearance gallery, 非人驱动视频的关键点获取能否完全自动化]

## Part I：问题与挑战

这篇论文解决的是一个很具体但长期被忽视的问题：**现有 portrait animation 的生成器已经很强，但控制接口仍然默认“输入必须像人脸”**。  
一旦参考图像变成玩具、emoji、动物或动漫角色，系统最先崩掉的往往不是视频生成模型，而是：

1. **关键点检测失败**：监督式人脸关键点检测器学到的是人脸分布，对非人角色会把嘴、眼、边界对错位置。
2. **关键点重定向失败**：很多方法用 3DMM 或人脸参数空间传动作，这对非人脸几何并不成立，尤其抓不住眼闭合、嘴微张这类细微动作。
3. **错误控制信号传给后端生成器**：于是后端模型会“补出一个人脸先验”，出现狗脸长人嘴、五官漂移、动作不连贯等问题。

**输入/输出接口**很清晰：  
- 输入：一张参考角色图像 + 一个驱动视频  
- 输出：保持参考角色身份、同时继承驱动视频表情/动作的动画视频

**为什么现在值得做？**  
因为两个条件成熟了：
- 扩散模型中间特征已经被证明具有较强的**语义对应能力**；
- MOFA-Video、AniPortrait 这类**关键点驱动动画模型已经足够强**，缺的不是新生成器，而是一个更通用的关键点前端。

所以 FaceShot 的判断很明确：**先别急着重训生成器，先把控制信号修对。**

**边界条件**也很明确：  
它仍然在做“脸部角色动画”，不是任意拓扑的开放域视频生成；其表示仍依赖 68 点面部关键点和眼/嘴/鼻/眉/脸边界这类部件划分。

## Part II：方法与洞察

FaceShot 本质上是一个**可插拔的关键点前端**，由三段组成：

1. **外观引导关键点匹配**
2. **坐标式关键点重定向**
3. **任意关键点驱动动画后端**

### 核心直觉

过去方法失败，不是因为模型不会“画视频”，而是因为它们默认存在一个**跨角色共享的人脸几何先验**。  
FaceShot 改动的关键因果旋钮是：

- **把“人脸检测/3D人脸拟合”这个强先验拿掉**
- 换成 **“扩散语义对应找部件位置 + 局部坐标传动作”**

这带来的瓶颈变化是：

- 从：**类别绑定的人脸几何约束**
- 变成：**部件级语义对齐 + 相对运动约束**

能力变化则是：

- 从：只能可靠驱动真人脸或类人脸
- 变成：能把同样的动画后端扩展到 emoji、玩具、动物、动漫角色
- 并且能保住更细的表情变化，如眼闭合、嘴张合、整体脸部运动

为什么这设计有效？  
因为下游动画模型真正需要的不是“真实人脸语义”，而是**时序一致、位置合理、能反映运动的关键点条件**。  
只要前端把“这个角色的眼睛、嘴、眉毛到底在哪里”找对了，后端就能正常发挥。

### 1）外观引导关键点匹配

论文不是直接对参考角色做人脸关键点检测，而是先做一个**语义对应问题**。

核心做法：
- 构造一个 **appearance gallery**，按五个部件存参考域：眼、嘴、鼻、眉、脸边界。
- 对输入角色图像，先从 gallery 中为每个部件挑一个**外观最接近的目标域**，显式减小跨域外观差异。
- 然后在 Stable Diffusion 特征空间里，通过 DDIM inversion + IP-Adapter 风格的图像提示，把参考图与目标图做**外观交叉引导**，提取更稳的中间特征。
- 最后对每个标准关键点做余弦相似度匹配，把目标域中的 canonical landmarks 对齐到参考角色上。

这个设计的本质不是“检测”，而是**借助扩散特征的语义对应能力去找角色上的等价五官位置**。  
因此它能避开“必须长得像人”的监督检测器限制。

### 2）坐标式关键点重定向

找到了参考角色的初始关键点后，还要把驱动视频里的动作传过去。

作者没有走 3DMM 路线，而是做了一个很朴素但实用的二维几何重定向：

- **全局阶段**：根据脸部边界端点定义整体坐标系，传递整脸的平移与旋转。
- **局部阶段**：对眼、嘴、鼻、眉、脸边界分别建立局部坐标系，传递每个部件的相对运动。
- 同时再加一个**点级比例变化**，让眼闭合、嘴开合这类细粒度动作能被保留。

这个模块的关键优点是：  
它不再要求“驱动对象和参考对象都能被一个统一的3D人脸模板解释”，而只要求**局部部件的相对变化可以被稳定搬运**。  
所以对非人角色更稳。

### 3）后端动画模型

得到参考角色的时序关键点后，FaceShot 直接把它们喂给已有 landmark-driven animation model。  
论文默认使用 **MOFA-Video**，并额外展示了作为 **AniPortrait 插件** 的效果。

这意味着 FaceShot 更像是：
- 一个**控制前端**
- 而不是一个必须端到端重训的新视频生成模型

这也是它“training-free”的真正含义：  
**不需要为新角色再训练/微调模型。**

### 战略取舍

| 设计选择 | 解决了什么瓶颈 | 带来的能力 | 代价/风险 |
| --- | --- | --- | --- |
| 外观引导的扩散特征匹配 | 非人角色与人脸检测器分布不一致 | 能在跨域角色上找准五官对应关系 | 需要预构建 gallery，且依赖预定义关键点模板 |
| 全局 + 局部坐标重定向 | 3DMM 对非人脸泛化差、细微动作损失 | 更稳地传递嘴张合、眼闭合、整脸运动 | 假设运动可被2D局部坐标近似 |
| 插件式接入现有动画模型 | 重训成本高、系统替换成本高 | 几乎不改后端就能提升开放域角色动画 | 效果上限仍受后端动画模型限制 |

## Part III：证据与局限

### 关键证据

**1. 主对比信号：真正的提升来自“控制修正”而非换后端。**  
在新建的 **CharacBench**（46 个非人/弱人脸角色）上，FaceShot 相比其直接后端 MOFA-Video：
- ArcFace：**0.695 → 0.848**
- Point-Tracking：**14.985 → 6.935**

这说明：同一个生成后端，单纯把关键点前端换掉，就能显著提升身份保持和动作一致性。  
这是最强的系统级因果证据。

**2. 用户偏好信号：人看起来也更自然。**  
15 个案例、20 名志愿者的人类评测中，FaceShot 在 Motion / Identity / Overall 三项都排第一（8.14 / 8.32 / 8.27）。  
这说明提升不只是指标层面，而是可感知的整体动画质量提升。

**3. 关键点匹配消融：真正瓶颈是 landmark localization。**  
与 DIFT、Uni-Pose、STAR 相比，FaceShot 在 CharacBench 上把 NME 降到 **8.569**，明显更低。  
这直接支持论文核心论点：**对非人角色，关键点能不能找准，比换更大的动画生成器更关键。**

**4. 重定向消融：二维坐标传递比 3DMM/参数化 retargeting 更适合非人角色。**  
FaceShot 的 Point-Tracking 为 **6.935**，优于 Deep3D、Everything’s Talking、FreeNet 的 8.2x 水平。  
说明它的优势不是只在“静态找点”，而是在“时序动作传递”上也成立。

**5. 插件性与成本信号：工程上可复用。**  
作为 MOFA-Video 插件时，50 帧只增加约 **119ms** 开销；论文还展示了接入 AniPortrait 后非人角色畸变明显减少。  
这说明 FaceShot 不是只能单独跑的研究原型，而是一个相对实用的前端模块。

### 局限性

- **Fails when**: 角色缺少清晰可对应的眼/嘴/眉/脸边界，或其表情拓扑极不规则时，扩散语义匹配会变得不稳定；此外，大姿态、严重遮挡、强非刚性变形若超出2D局部坐标假设，重定向容易失真。
- **Assumes**: 系统依赖一个预构建且带关键点先验的 appearance gallery；驱动侧关键点需要可获取，当前实现对人类驱动主要依赖 FAN，而附录也明确提到对非人角色仍有人工标注依赖；同时还假设已有可用的关键点驱动动画后端，如 MOFA-Video 或 AniPortrait。
- **Not designed for**: 全身动画、手部/头发等非脸部拓扑控制、严格3D一致性重建、完全无人工先验的开放域视频驱动。

### 复现与可扩展性备注

- 论文声称 **training-free**，但更准确地说是**免微调**，不是完全免先验或免标注。
- 附录显示实验在 **单张 H800** 上完成；虽然插件额外开销小，但整体仍受扩散视频后端推理成本限制。
- 代码在文中表述为**将公开发布**，因此当前更适合标为 `opensource/promised`。
- 评测里使用了 ArcFace、HyperIQA、Aesthetic 等偏人类图像分布的指标；对玩具、emoji、动漫角色，这类指标可能存在偏置。因此这篇论文的证据虽然扎实，但仍应保守看作 **moderate**：主 benchmark 强、消融清楚，但任务分布仍相对集中。

### 可复用组件

- **扩散特征语义匹配前端**：适合拿去做跨域关键点发现、角色对齐、控制点初始化。
- **全局/局部坐标重定向**：适合替代人脸 3DMM 式 retargeting，尤其是二维角色或非人角色。
- **插件式控制接口**：非常适合接到现有 landmark-driven animation pipeline 上做能力外扩。

## Local PDF reference

![[paperPDFs/Digital_Human_Human_Image_and_Video_Generation/ICLR_2025/2025_FaceShot_Bring_Any_Character_into_Life.pdf]]