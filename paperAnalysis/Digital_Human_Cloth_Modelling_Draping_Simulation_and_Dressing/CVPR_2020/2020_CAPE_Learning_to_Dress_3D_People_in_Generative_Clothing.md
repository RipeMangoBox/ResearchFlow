---
title: "CAPE: Learning to Dress 3D People in Generative Clothing"
venue: CVPR
year: 2020
tags:
  - Others
  - task/3d-clothed-human-modeling
  - vae-gan
  - graph-convolution
  - additive-displacement
  - dataset/CAPE
  - repr/SMPL
  - opensource/full
core_operator: 将服装表示为SMPL规范空间上的、由姿态与服装类型条件控制的可采样顶点位移层，并用条件Mesh-VAE-GAN生成细节丰富的着衣人体网格。
primary_logic: |
  SMPL身体形状/姿态 + 服装类型 + 采样潜变量 → 在SMPL拓扑上生成规范空间中的姿态相关服装位移，并用mesh patch判别器抑制过平滑、保留褶皱细节 → 通过SMPL蒙皮输出可重定姿的着衣人体网格
claims:
  - "On held-out CAPE test scans, CAPE achieves the lowest mean per-vertex auto-encoding error among the compared models, reaching 5.54 mm on male data and 4.21 mm on female data, ahead of PCA and CoMA variants [evidence: comparison]"
  - "Removing the patch discriminator, residual blocks, or edge loss increases reconstruction error relative to the full model, indicating each component contributes to preserving clothing geometry detail [evidence: ablation]"
  - "When plugged into SMPLify on 120 rendered test images, CAPE reduces per-vertex MSE from 0.0223 to 0.0189 [evidence: comparison]"
related_work_position:
  extends: "SMPL (Loper et al. 2015)"
  competes_with: "DRAPE (Guan et al. 2012); Yang et al. (2018)"
  complementary_to: "SMPLify (Bogo et al. 2016)"
evidence_strength: moderate
pdf_ref: paperPDFs/Digital_Human_Cloth_Modelling_Draping_Simulation_and_Dressing/CVPR_2020/2020_CAPE_Learning_to_Dress_3D_People_in_Generative_Clothing.pdf
category: Others
---

# CAPE: Learning to Dress 3D People in Generative Clothing

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/1907.13615), [Project / Code / Data](https://cape.is.tue.mpg.de)
> - **Summary**: 这篇工作把服装从“SMPL上的固定外壳”改成“可采样、受姿态和服装类型条件控制的位移层”，从而得到既能重定姿、又能生成褶皱细节的3D着衣人体模型。
> - **Key Performance**: 持出集重建误差达到 5.54 mm（男）/ 4.21 mm（女）；接入 SMPLify 后图像拟合 MSE 从 0.0223 降到 0.0189。

> [!info] **Agent Summary**
> - **task_path**: SMPL身体形状/姿态 + 服装类型 + 隐变量采样 -> 着衣3D人体网格
> - **bottleneck**: 现有人体模型只建模最小着装身体，而现有服装偏移方法多为确定性或静态，无法同时表达一对多服装形态、姿态相关褶皱、以及SMPL兼容的可重定姿性
> - **mechanism_delta**: 在SMPL规范空间增加一个姿态/服装类型条件化的概率式服装位移层，并用mesh PatchGAN与边约束补回局部褶皱细节
> - **evidence_signal**: 持出集上优于PCA/CoMA的重建结果 + 关键模块消融有效 + 下游SMPLify拟合误差下降
> - **reusable_ops**: [canonical-space additive clothing layer, mesh patchwise discriminator]
> - **failure_modes**: [garments with non-body topology such as skirts or open jackets, fast-motion cloth dynamics outside pose-only modeling]
> - **open_questions**: [how to support topology-changing garments, how to model temporal cloth dynamics beyond static pose conditioning]

## Part I：问题与挑战

这篇论文解决的不是“如何从图像重建衣服”，而是更基础的一层：**如何给参数化人体模型补上一个可生成、可控、可重定姿的服装层**。

### 1. 真正的问题是什么
SMPL 这类人体模型很好用，但它是从最小着装扫描学出来的。于是出现两个直接问题：

1. **真实世界里的人大多穿衣服**，最小着装几何与图像/视频中的观测存在系统性域差。
2. **衣服不是刚性壳，也不是单值函数**：同一个身体和姿态，可以对应多种版型、松紧度、褶皱状态。

所以瓶颈并不是“有没有衣服”，而是：

- 要保留 **SMPL 的参数化、可微、可蒙皮、可动画化** 优点；
- 同时表达 **服装的多样性**；
- 还要表达 **姿态相关的局部褶皱**。

这三个目标放在一起，才是难点。

### 2. 以前方法卡在哪里
作者把已有方法的短板切得很清楚：

- **非参数重建类**：细节多，但结果难以重定姿、难以控制服装类型。
- **SMPL 偏移层类**：容易重定姿，但常把衣服当成固定偏移或确定性回归，难表示一对多与姿态依赖。
- **普通 mesh autoencoder**：全局结构能学到，但容易把高频褶皱“平均掉”。

换句话说，过去方法往往只能二选一：

- 要么细节好，但不可控；
- 要么可控，但衣服像“贴在身体上的平滑壳”。

### 3. 输入/输出接口与边界
**输入**：
- 身体形状 β
- 身体姿态 θ
- 服装类型 c
- 服装潜变量 z

**输出**：
- 一个与 SMPL 拓扑兼容的着衣人体网格

**边界条件**：
- 服装被表示成 **SMPL 顶点上的位移**；
- 因而它天然偏向 **与身体拓扑一致的服装**；
- 主要覆盖 4 类 full-body outfit，而不是任意拓扑服饰；
- 仅建模 **pose-dependent**，不建模 **dynamic-dependent** 布料动力学。

### 4. 为什么现在值得做
因为两个条件同时成熟了：

- **数据**：作者采集了约 80K 帧 4D 着衣人体扫描，并且额外采了最小着装 body shape，能把“身体”和“衣服位移”真正拆开。
- **模型**：Graph-CNN + VAE-GAN 已经足够强，可以在 mesh 上同时抓全局结构和局部细节。

---

## Part II：方法与洞察

### 方法总览
CAPE 的核心不是直接生成整个人体网格，而是：

1. 先用 SMPL 给出最小着装身体；
2. 再在 **规范空间** 上生成一个服装位移层；
3. 最后复用 SMPL 的蒙皮，把这个着衣模板带到目标姿态。

这让它本质上成为 **“SMPL + generative clothing term”**。

### 1. 把服装变成 SMPL 上的附加项
作者把服装定义为一个顶点位移层 `S_clo(z, θ, c)`：

- 不是独立 garment mesh；
- 不是像素/体素；
- 而是 **每个 SMPL 顶点的 3D offset**。

这样做的最大好处是：**与 SMPL 完全兼容**。  
已有依赖 SMPL 的优化、渲染、动画、合成数据流程，都能直接接入。

但这里有个关键细节：  
这个位移层虽然定义在规范空间，却**显式依赖姿态 θ**。  
这意味着模型学的是一种“为了经过 SMPL 蒙皮后得到正确衣服形状，规范空间里该怎么预补偿”的映射，而不是简单的静态外壳。

### 2. 用条件 Mesh-VAE-GAN 解决“一对多”
作者没有把服装建模成回归问题，而是建模成 **条件生成问题**：

- `z`：编码服装结构、宽松程度、局部形态；
- `θ`：控制姿态相关变形；
- `c`：控制服装类型。

这样同一个身体和姿态下，采样不同 `z`，就能生成不同的着装实例。  
这一步真正改变的是：**从单点估计变成分布建模**。

### 3. 为什么要加 PatchGAN
普通 mesh autoencoder 会把局部高频细节抹平。  
衣服最怕这个，因为褶皱、袖口、衣摆边界，恰恰都在高频部分。

所以作者做了两个补救：

- **Residual graph blocks**：缓解深层图卷积丢局部信息；
- **mesh patchwise discriminator**：把 PatchGAN 从图像扩展到 mesh，只盯局部 patch 的真假；
- 再配合 **edge loss**：鼓励边长度/局部几何更接近真实衣褶结构。

因此，重建损失负责全局形状，局部判别器负责褶皱真实性。

### 核心直觉

作者真正拧动的“因果旋钮”是：

> 把服装从“姿态后的静态几何/单值回归输出”，改成“规范空间中、由姿态与服装类型条件控制的概率式位移场”。

这带来了三层变化：

1. **分布层面**：  
   从 one-to-one 回归，变成 one-to-many 采样，服装多样性可表达。

2. **约束层面**：  
   从独立 clothing mesh 或静态 offset，变成与 SMPL 拓扑和蒙皮强耦合的位移层，重定姿与动画更自然。

3. **信息瓶颈层面**：  
   全局结构由 VAE 潜空间承担，局部高频细节由 PatchGAN + edge loss 补回，避免“平均化褶皱”。

因此能力变化很明确：

- 能 **采样不同衣服外观**；
- 能 **跨姿态保持衣服身份一致**；
- 能 **随姿态变化生成更 plausible 的褶皱**；
- 还能继续作为 **SMPL-compatible prior** 接到下游任务里。

### 4. 与下游拟合的关系
因为整个模型对 `β, θ, z` 可微，作者把它接到 SMPLify 后面：

- 先得到最小着装人体；
- 再加服装层；
- 用 silhouette 误差继续优化。

这说明 CAPE 不只是一个“看起来好看的生成器”，而是一个 **可优化的几何先验**。

### 战略取舍

| 设计选择 | 得到的能力 | 代价 / 限制 |
|---|---|---|
| 服装 = SMPL拓扑上的顶点位移 | 与SMPL无缝兼容，易于重定姿和动画 | 不能自然处理裙子、开襟外套等拓扑变化服装 |
| 条件VAE采样 z | 表达同姿态下多种服装实例 | 需要足够覆盖的扫描数据来学分布 |
| 姿态条件化的规范空间位移 | 衣服可随姿态变化，不只是静态外壳 | 仍是 pose-only，不含速度/惯性等动态因素 |
| PatchGAN + edge loss | 局部褶皱、衣摆、袖口细节更真实 | 训练更复杂，且真实性主要来自局部判别而非物理一致性 |
| 与SMPLify联动 | 可作为图像拟合的服装先验 | 拟合质量仍受上游姿态估计与轮廓质量影响 |

---

## Part III：证据与局限

### 关键证据

**Signal 1 — comparison：持出集重建优于 PCA / CoMA**  
在 CAPE 数据集的持出测试集上，作者用每顶点欧氏误差评估 auto-encoding：

- 男性：**5.54 mm**
- 女性：**4.21 mm**

都优于 CoMA 变体；对男性数据也略优于 PCA。  
更关键的是，PCA 即便在某些误差数值上接近，也**不具备姿态条件化采样能力**，所以不是同一类能力。

**Signal 2 — ablation：细节不是“白来的”**  
去掉任一关键组件都会变差：

- 去掉 discriminator：误差上升
- 去掉 residual block：误差上升
- 去掉 edge loss：误差上升更明显

而且定性结果里，衣摆上扬、背部褶皱、局部边界都会更糊。  
这说明作者的提升并非只来自“换个更大网络”，而是来自针对 mesh 细节瓶颈的结构设计。

**Signal 3 — case-study：感知真实性有所提升但仍不够接近真实**  
AMT 人类实验里，生成结果被误判为真实的比例约为：

- **35.1% ± 15.7%**
- **38.7% ± 16.5%**

离 50% 的“真假难分”还有距离，说明模型已能生成有一定真实感的衣物几何，但仍未完全逼真。

**Signal 4 — comparison：下游图像拟合有实际收益**  
把 CAPE 接入 SMPLify 后，在 120 个渲染测试样本上：

- SMPLify: **0.0223**
- Ours: **0.0189**

这说明它不仅会“生成衣服”，而且确实能作为**着衣先验**改善拟合。

### 我认为最有说服力的“能力跃迁”
真正的跃迁不是单个误差点数，而是这三件事第一次被较好统一：

1. **和 SMPL 兼容**
2. **可采样、不是单值**
3. **服装随姿态变化且保留局部细节**

相比之前“静态偏移层”或“只重建不参数化”的路线，CAPE 更像是把“衣服”真正纳入了参数化人体模型体系。

### 局限性
- **Fails when**: 输入包含裙子、开襟夹克、连指手套、鞋类等与人体拓扑差异较大的服装；或者动作很快、布料强依赖速度与惯性时；另外若上游 SMPLify 姿态估计失败，基于它的 clothed fitting 也会失败。
- **Assumes**: 需要 SMPL-compatible 注册网格、最小着装体形扫描、4D 扫描设备、人工清洗失败帧；图像拟合里还假设服装类型已知或可由上游获得；论文中男女分开训练，说明分布假设并非完全统一。
- **Not designed for**: 任意拓扑服装建模、物理精确的布料动力学模拟、端到端从单张图直接预测全部服装几何与类别、超高分辨率细皱纹生成。

### 复现与可扩展性的现实约束
这篇工作虽然 **代码/数据公开**，但复现门槛仍不低，因为核心训练数据依赖：

- 3dMD 级别 4D 扫描系统
- 同一被试的 clothed / minimally-clothed 双条件采集
- 高质量注册到 SMPL 拓扑
- 人工剔除坏帧

所以它的“模型公开”并不等于“数据生产廉价”。

### 可复用组件
- **SMPL-compatible additive clothing layer**：适合任何想在 SMPL 上加衣服先验的任务。
- **Mesh PatchGAN**：对 mesh 高频细节生成有普适价值。
- **姿态条件化服装先验**：适合图像拟合、合成数据生成、动画重定姿。
- **clothed-minus-minimal displacement 表示**：适合把“身体”和“衣服”解耦。

![[paperPDFs/Digital_Human_Cloth_Modelling_Draping_Simulation_and_Dressing/CVPR_2020/2020_CAPE_Learning_to_Dress_3D_People_in_Generative_Clothing.pdf]]