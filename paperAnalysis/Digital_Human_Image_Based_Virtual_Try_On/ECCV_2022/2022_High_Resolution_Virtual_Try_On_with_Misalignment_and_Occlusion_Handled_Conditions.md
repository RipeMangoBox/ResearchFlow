---
title: "High-Resolution Virtual Try-On with Misalignment and Occlusion-Handled Conditions"
venue: ECCV
year: 2022
tags:
  - Others
  - task/image-based-virtual-try-on
  - appearance-flow
  - feature-fusion
  - rejection-sampling
  - dataset/VITON-HD
  - opensource/full
core_operator: "将服装形变与人体分割合并为一个双路径特征融合条件生成器，再用条件对齐和身体遮挡裁剪保证衣物可见区域一致。"
primary_logic: |
  去服装化人物表示/姿态 + 目标服装图像
  → 在统一条件生成器中联合估计外观流与人体分割，并跨尺度交换几何/语义信息
  → 对服装区域做条件对齐并移除被身体遮挡的衣物像素
  → 由图像生成器融合人体、姿态与对齐后的服装条件，输出高分辨率试穿图像
claims:
  - "Claim 1: On the VITON-HD high-resolution test set at 1024×768, HR-VITON achieves LPIPS 0.065, SSIM 0.892, FID 10.91, and KID 0.179, outperforming the reported baselines including VITON-HD and PF-AFN [evidence: comparison]"
  - "Claim 2: Removing condition aligning or the feature fusion block worsens unpaired FID from 10.91 to 12.05 and 12.41 respectively, and removing both degrades it further to 12.73, showing that joint information exchange and alignment are both necessary [evidence: ablation]"
  - "Claim 3: A stronger warping baseline alone does not explain the gains: VITON-HD* with ClothFlow-style warping still underperforms HR-VITON at 1024×768 (FID 11.55 vs. 10.91), indicating the benefit comes from joint condition generation and occlusion handling rather than only higher warping flexibility [evidence: comparison]"
related_work_position:
  extends: "VITON-HD (Choi et al. 2021)"
  competes_with: "VITON-HD (Choi et al. 2021); PF-AFN (Ge et al. 2021)"
  complementary_to: "N/A"
evidence_strength: moderate
pdf_ref: paperPDFs/Digital_Human_Image_Based_Virtual_Try_On/ECCV_2022/2022_High_Resolution_Virtual_Try_On_with_Misalignment_and_Occlusion_Handled_Conditions.pdf
category: Others
---

# High-Resolution Virtual Try-On with Misalignment and Occlusion-Handled Conditions

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2206.14180), [Code](https://github.com/sangyun884/HR-VITON)
> - **Summary**: 论文把“衣服 warping”和“人体分割条件生成”从彼此断开的两步改成联合预测，并显式处理身体遮挡区域，从源头减少高分辨率虚拟试衣中的错位与纹理挤压伪影。
> - **Key Performance**: 在 1024×768 的 VITON-HD 测试集上，FID 10.91、KID 0.179，优于 VITON-HD 的 11.59 / 0.247。

> [!info] **Agent Summary**
> - **task_path**: 服装图像 + 去服装化人体表示/姿态 -> 高分辨率人物试穿图像
> - **bottleneck**: warping 模块与 segmentation 模块彼此断开，导致衣物几何支持区域与语义布局不一致，放大为 misalignment 与遮挡处的 pixel squeezing
> - **mechanism_delta**: 用带双路径信息交换的 try-on condition generator 同时预测外观流和分割图，并通过 condition aligning 与 body-part occlusion handling 约束最终可见衣物区域
> - **evidence_signal**: 1024×768 上对 VITON-HD 的 FID/KID 提升，以及去掉 feature fusion / condition aligning 后 FID 明显恶化的消融
> - **reusable_ops**: [joint condition generation, condition aligning]
> - **failure_modes**: [preprocessing failures in clothing mask/parsing, out-of-distribution camera views or non-human inputs]
> - **open_questions**: [can it generalize beyond frontal-view women tops, can rejection be replaced by calibrated correction instead of reject-only]

## Part I：问题与挑战

这篇论文解决的并不是“如何把衣服拉到人身上”这么表层的问题，而是更底层的 **条件不一致**。

在高分辨率 image-based virtual try-on 里，主流流程通常是三段式：

1. 先预测人穿上目标衣服后的 **segmentation map**  
2. 再把目标衣服 **warp** 到目标人体  
3. 最后把这些条件送进生成器融合成最终图像

问题在于，这两个关键条件——**warped cloth** 和 **segmentation map**——往往是分别预测的，几乎没有信息交换。于是会出现两个系统性问题：

- **Misalignment**：warped cloth 覆盖的位置，和 segmentation 里“衣服应该出现的位置”不一致。最终生成器收到互相矛盾的条件，就会在领口、袖口、边界处产生明显伪影。
- **Pixel squeezing**：当手臂或身体部位遮挡衣服时，单独的 warping 模块会试图把本应被遮住的衣物纹理硬挤进可见区域，导致 logo、条纹、图案被拉扁、挤碎。

### 输入/输出接口

- **输入**：
  - 目标服装图像 `c`
  - 去服装化的人体表示：clothing-agnostic image、clothing-agnostic segmentation
  - DensePose pose map
  - 服装 mask
- **输出**：
  - 对齐后的 warped clothing
  - 目标穿着布局 segmentation
  - 最终高分辨率试穿图像

### 为什么现在值得解决

因为高分辨率试衣把过去“还能糊过去”的不一致放大了。低分辨率下边界错一小块可能不显眼，但到了 1024×768，领口破损、纹理挤压、遮挡错误都会直接破坏可用性。对电商和真实应用来说，这种错误不是“稍差一点”，而是用户一眼能看出的失败样本。

### 边界条件

这篇工作有明确适用边界：

- 数据主要是 **frontal-view women + top clothing**
- 训练仍采用经典的 **paired reconstruction** 范式
- 强依赖预处理得到的人体解析、DensePose、服装 mask
- 目标是 **image-based** try-on，不涉及真实 3D 布料物理

---

## Part II：方法与洞察

### 方法总览

HR-VITON 的核心不是换一个更强的生成器，而是先把“生成器吃进去的条件”做好。

整体是两阶段：

1. **Try-on Condition Generator**
   - 输入服装图像/服装 mask，以及人体的 agnostic segmentation + DensePose
   - 同时输出：
     - warped clothing
     - segmentation map
   - 重点是让这两者在生成过程中持续交换信息，而不是各算各的

2. **Try-on Image Generator**
   - 以 `Ia + warped cloth + pose` 为输入
   - 以预测的 segmentation 作为 SPADE 条件
   - 负责最后的高分辨率渲染

3. **Discriminator Rejection**
   - 推理时额外利用 condition generator 的判别器分数
   - 把明显错误的 segmentation 预测拒绝掉，避免给出很差的结果

### 核心直觉

**改了什么：**  
把原来“先独立 warp、再独立分割”的串联结构，改成了一个 **联合条件生成器**。在这个生成器里，flow 路径负责“衣服往哪里变形”，seg 路径负责“最终哪里应该是衣服/身体”，两条路径在每个尺度互相传信息。

**哪个瓶颈被改变了：**  
原来系统的瓶颈是 **几何条件和语义条件属于两个不一致的分布**：  
- warp 只关心衣服怎么覆盖人体
- segmentation 只关心人体布局应该长什么样  
二者缺少约束，所以最后的“衣服支持集”不一致。

联合建模后，系统把瓶颈从“两个各自合理但彼此矛盾的条件”改成“一个共同生成、彼此可行的条件对”。再加上：

- **Condition Aligning**：强制 segmentation 里的服装区域只出现在 warped clothing mask 支持的地方
- **Body-part Occlusion Handling**：把被手臂/身体遮挡的衣物像素从 warped cloth 中去掉

这样就把“衣服该出现在哪里”与“衣服实际上能出现在哪里”绑定起来了。

**能力上发生了什么变化：**  
- 从“后处理地缓解错位”变成“机制上不制造错位”
- 从“对遮挡区域盲目拉伸纹理”变成“显式只保留可见衣物”
- 因此在高分辨率下更能保留 neckline、logo、stripe 这类细节

### 关键模块

#### 1. Feature Fusion Block：让几何和语义互相看见对方

作者设计了一个双路径模块：

- **flow pathway**：预测 appearance flow
- **seg pathway**：预测 segmentation feature

两条路径双向交换信息：

- segmentation 特征指导 flow 不要在遮挡区域乱拉纹理
- flow 结果反过来告诉 segmentation：衣服真实能落在哪些像素上

这一步的价值不在“更复杂”，而在于它把 try-on 里的两个核心中间变量变成了 **联合可解释的条件**。

#### 2. Condition Aligning：显式消除支持集不一致

作者没有只依赖网络“学会对齐”，而是在输出层加了一个很直接的约束：

- segmentation 的衣服通道，只保留与 warped clothing mask 重叠的区域

这相当于做了一个 **support-set consistency** 操作。  
好处是：即使网络局部会“幻想”出多余衣服区域，这一步也会把它裁掉，避免最终融合时产生红边、破口、错层。

#### 3. Body-part Occlusion Handling：不要把被挡住的衣服硬挤出来

对于被手臂等身体部位遮住的衣服区域，作者直接利用预测的 body-part 区域，把这些像素从 warped clothing 和 clothing mask 中移除。

因果上看，它把目标从：

- “让衣服覆盖到人体轮廓上”

改成：

- “只对真实可见的衣服区域做保真 warp”

这就是为什么它能减少 pixel squeezing，并更好保留衣服纹理。

#### 4. Discriminator Rejection：部署层面的保险丝

很多失败样本不是主干网络完全不会，而是前处理错了，比如：

- clothing mask 错
- parsing 错
- 输入本身偏离训练分布

作者把判别器分数拿到测试阶段做 rejector，用于筛掉明显错误的 segmentation 条件。  
这不是提升模型“平均能力”，而是提升系统的 **可部署稳健性**。

### 战略取舍

| 设计选择 | 解决什么问题 | 带来什么能力 | 代价 / 风险 |
| --- | --- | --- | --- |
| 联合 condition generator | warping 与 segmentation 脱节 | 几何-语义一致，减少 misalignment | 训练耦合更强，调试更复杂 |
| Feature fusion block | 两类中间变量彼此盲视 | flow 与 layout 协同优化 | 若输入条件噪声大，误差也会互相传播 |
| Condition aligning | segmentation 衣服区域“虚多” | 从机制上裁掉不一致区域 | 依赖 clothing mask 质量 |
| Occlusion handling | 遮挡附近纹理挤压 | 更好保留 logo / stripe 细节 | 若 body-part 预测错，可能误删可见衣物 |
| Discriminator rejection | 错误条件进入最终渲染 | 提高失败样本过滤能力 | 只能拒绝，不能修复；会降低覆盖率 |

---

## Part III：证据与局限

### 关键证据

- **基线比较信号**：在 1024×768 上，HR-VITON 比 VITON-HD 的 FID/KID 更低，也明显优于 parser-free 的 PF-AFN。说明高分辨率下，关键不是“最后一个图像生成器更会画”，而是前面的条件更一致。
- **消融信号**：去掉 **feature fusion block** 或 **condition aligning**，FID 都会明显变差；两者都去掉时更差。这直接支持作者的核心论点：真正有效的是“联合信息交换 + 显式对齐”。
- **因果隔离信号**：即使给 VITON-HD 换一个更自由的 warping 模块（VITON-HD*），仍然不如 HR-VITON。说明提升不只是因为 warp 更强，而是因为作者改了更本质的因果旋钮：**几何与语义条件的联合生成**。
- **质化与用户偏好信号**：可视化里领口、袖口、条纹、logo 的保真度更好，appendix 中的用户研究也更偏好其真实感与细节保留。

### 1-2 个最关键指标

- **1024×768，unpaired**：FID 10.91，优于 VITON-HD 的 11.59
- **1024×768，unpaired**：KID 0.179，优于 VITON-HD 的 0.247

### 局限性

- **Fails when:** 输入明显偏离训练分布时会失败，例如非正面视角、非人体输入、多人场景，或预处理阶段得到的 clothing mask / parsing / DensePose 明显错误；此时 discriminator rejection 只能拒绝样本，不能把错误结果修好。
- **Assumes:** 依赖外部人体解析、DensePose、服装 mask 与 clothing-agnostic 表示；训练与测试默认接近 VITON-HD 的数据分布（正面女性上衣）；同时采用两阶段 GAN 式训练，工程链路较长。
- **Not designed for:** 非上衣品类的通用试穿、强视角变化、真实 3D 布料动力学、多帧视频时序一致性、开放世界的人体/服装组合。

### 可复用组件

- **联合条件生成器**：任何需要同时预测“几何变形 + 语义布局”的图像编辑任务都可以借鉴
- **Condition aligning**：可作为“生成布局”和“显式纹理支持集”之间的一致性约束
- **遮挡感知衣物裁剪**：适合处理人体部件遮挡引起的纹理挤压问题
- **判别器拒识层**：适合部署阶段的 fail-safe 设计，尤其是在前处理不稳定的系统里

## Local PDF reference

![[paperPDFs/Digital_Human_Image_Based_Virtual_Try_On/ECCV_2022/2022_High_Resolution_Virtual_Try_On_with_Misalignment_and_Occlusion_Handled_Conditions.pdf]]