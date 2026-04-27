---
title: "HumanGAN: A Generative Model of Human Images"
venue: CVPR
year: 2021
tags:
  - Others
  - task/human-image-generation
  - task/pose-transfer
  - gan
  - variational-autoencoder
  - part-based-latent
  - dataset/DeepFashion
  - dataset/Fashion
  - repr/SMPL
  - opensource/no
core_operator: "先把人体外观映射到姿态无关的SMPL UV空间并编码为24个部位潜变量，再按目标DensePose逐部位广播回图像空间进行生成"
primary_logic: |
  源图像外观或随机采样潜变量 + 目标DensePose
  → 在姿态无关的UV纹理空间编码24个身体部位的高斯外观向量
  → 将各部位向量按目标姿态warp成空间噪声图
  → 由高保真生成器输出可控姿态、身份与局部服饰的人体图像
claims:
  - "在 DeepFashion 的姿态条件外观采样上，HumanGAN 取得 FID 24.9 和 diversity LPIPS 0.219，优于 VUNet 与 Pix2PixHD 系列基线 [evidence: comparison]"
  - "在 176 对 pose-transfer 测试集上，HumanGAN 以 0.777 的 SSIM 超过文中所有对比方法，同时保持 0.187 的竞争性 LPIPS [evidence: comparison]"
  - "部位级潜变量 warping 相比不分部位的 NoParts 基线，将非目标区域变化从 0.44 降到 0.11，同时保留目标区域可变性 [evidence: ablation]"
related_work_position:
  extends: "VAE-GAN (Larsen et al. 2016)"
  competes_with: "VUNet (Esser et al. 2018); NHRR (Sarkar et al. 2020)"
  complementary_to: "N/A"
evidence_strength: moderate
pdf_ref: "paperPDFs/Digital_Human_Image_Based_Virtual_Try_On/CVPR_2021/2021_HumanGAN_A_Generative_Model_of_Humans_Images.pdf"
category: Others
---

# HumanGAN: A Generative Model of Human Images

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2103.06902), [project](https://gvv.mpi-inf.mpg.de/projects/HumanGAN/)
> - **Summary**: 该工作把人体外观先放到姿态无关的 UV 规范空间做部位级潜编码，再按目标姿态重投影到图像空间，从而在一个模型里同时支持姿态控制、外观采样、部位采样与服饰迁移。
> - **Key Performance**: DeepFashion 上随机外观采样达到 **FID 24.9 / LPIPS diversity 0.219**；pose transfer 达到 **SSIM 0.777**。

> [!info] **Agent Summary**
> - **task_path**: 源图像 UV 外观纹理或随机部位潜变量 + 目标 DensePose -> 全身人体图像
> - **bottleneck**: 传统全局 latent 会把姿态、身份和局部服饰纠缠在一起，导致只能整体采样，难以在跨姿态下保持外观并做局部编辑
> - **mechanism_delta**: 将外观编码为 24 个姿态无关的部位潜变量，并在生成前按目标姿态把每个部位向量 warp 到对应像素区域
> - **evidence_signal**: DeepFashion 采样实验中 FID 24.9 明显优于 VUNet 50.0 与 Pix2PixHD 系基线，且用户研究对真实性偏好达 91.06%
> - **reusable_ops**: [DensePose-to-UV canonicalization, part-wise latent broadcasting]
> - **failure_modes**: [部位或服饰边界会出现交叠伪影, 数据偏置使生成结果偏向女性]
> - **open_questions**: [24 个 DensePose 部位是否足够表达细粒度服装语义, 如何在不牺牲局部可控性的前提下减少边界伪影]

## Part I：问题与挑战

这篇论文真正要解决的，不是单一的“姿态图到人物图像翻译”，而是更难的：

**如何构造一个既能随机采样、又能保持身份、还能局部编辑服饰/身体部位的人体生成模型。**

### 1. 真正的瓶颈是什么？

现有方法大致分成两类，但都各有明显缺口：

1. **GAN/StyleGAN 类生成模型**  
   能生成逼真人物，但通常用一个全局 latent 控制整张图。  
   问题是：  
   - 外观和姿态容易纠缠；
   - 不能只改“上衣”或“头部”而保持其余不变；
   - 难以在不同姿态下保持同一人的外观一致性。

2. **条件图像翻译 / pose transfer 方法**  
   能根据给定姿态和参考图生成对应人物，但通常是**确定性映射**。  
   问题是：  
   - 更像“重建/迁移器”，不是可采样的生成模型；
   - 很难自然支持“同一姿态下随机换衣服”“只采样某个部位”这类操作。

所以瓶颈不在“能不能生成人体”，而在：

> **如何把“姿态控制”与“局部外观随机性”拆开，并且让这种拆开在空间上可定位、在跨姿态时可保持。**

### 2. 为什么现在值得解决？

因为人体图像生成的应用正在从“能生成”转向“可编辑”：
- 虚拟试衣 / 时尚内容生成
- AR/VR 数字人
- 姿态数据增广
- 可控服饰与部件编辑

而 DensePose + SMPL UV 对应，恰好提供了一个可用的**人体规范坐标系**。这让“先在 canonical space 编码外观，再按目标姿态投回图像空间”成为可行设计。

### 3. 输入/输出接口与边界条件

**输入接口**
- 训练时：同一人的源图 `Is` 与目标图 `It`，两者姿态不同
- 推理时：
  - 随机采样：`z + 目标 DensePose`
  - 姿态迁移：`源图像 -> 编码外观 z`，再配合 `目标 DensePose`

**输出**
- 512×512 的全身着装人体图像

**边界条件**
- 单人、时尚摄影风格数据为主
- 强依赖 DensePose 估计与 SMPL UV 对应
- 训练需要同一人物跨姿态样本
- 更偏 2D 外观生成，不是物理准确的 3D 着装模拟

---

## Part II：方法与洞察

HumanGAN 的核心思路可以概括为：

> **把“外观”从图像坐标系搬到人体规范坐标系里编码，再把它按目标姿态放回去。**

### 方法流程

整个系统有四步：

1. **提取姿态无关外观**
   - 用 DensePose 把源图像映射到 SMPL UV 空间
   - 得到部分可见的 UV texture map `Ts`
   - 这一步把“外观”放到人体 canonical space，尽量去掉原图姿态影响

2. **编码部位级外观 latent**
   - 编码器不直接看原图，而是看 `Ts`
   - 输出 24 个身体部位各自的高斯 latent 参数
   - 每个部位有独立向量，而不是整个身体共用一个 global z

3. **按目标姿态做部位级 warping**
   - 给定目标 DensePose `Pt`
   - 将第 k 个部位 latent 只广播到目标图像中的对应部位区域
   - 最终得到一个空间噪声图 `Zt`

4. **高保真生成器解码**
   - 使用 Pix2PixHD 风格生成器
   - 输入不是“原图 + pose”，而是已经带有空间对齐语义的 `Zt`
   - 输出目标姿态下的人体图像

### 核心直觉

#### 直觉一：改变的不是生成器，而是“latent 放在哪里”

以前很多方法的问题在于：
- latent 是全局的；
- 或者 pose 直接喂给生成器，latent 只剩下一点“风格余量”。

HumanGAN 改成了：
- **在 UV 规范空间中编码外观**
- **按身体部位拆分 latent**
- **按目标姿态把 latent 空间化后再生成**

这带来的因果变化是：

**全局 latent → 部位级 canonical latent**  
→ 减少姿态与外观纠缠  
→ 让“同一外观跨姿态保持”成为可能

**无空间归属的随机向量 → 目标姿态对齐的部位噪声图**  
→ 每个 latent 只影响对应区域  
→ 支持局部采样、局部服饰迁移

#### 直觉二：不给生成器直接看 DensePose，是在“逼迫 latent 学语义”

论文附录里的一个有意思发现是：如果把 DensePose 也直接拼到生成器输入里，多样性会下降。  
这说明当 pose 条件太强时，模型会倾向于把生成任务“偷懒”成确定性翻译，latent 变得不重要。

HumanGAN 反而让生成器主要依赖 warp 后的噪声图，这相当于强迫：
- pose 信息通过空间布局进入；
- 外观信息通过部位 latent 进入。

因此“哪里生成什么”被提前编码进输入结构里。

### 为什么这个设计有效？

1. **UV 规范空间弱化了姿态泄漏**  
   外观编码来自 canonical texture，而不是原图像坐标，因此更容易学到“衣服长什么样”，而不是“衣服在图里出现在哪”。

2. **部位拆分把控制粒度从整图缩到身体区域**
   这让“只换头部 / 上身 / 下身”成为直接操作，而不是期望网络隐式学会。

3. **warp 后再生成，把语义控制变成空间先验**
   生成器不必自己学“哪个 latent 对应哪块区域”，因为这个对应关系已经被 DensePose 显式给出。

### 战略权衡

| 设计选择 | 带来的能力 | 代价/风险 |
|---|---|---|
| 在 SMPL UV 空间编码外观 | 更好地解耦姿态与外观，支持跨姿态保持 | 依赖 DensePose/UV 映射质量，自遮挡区域纹理不完整 |
| 24 个部位独立 latent | 支持部位采样与服饰迁移 | 部位粒度较粗，边界容易出现拼接/交叠伪影 |
| 先 warp latent 再生成 | 明确建立“latent→空间区域”的因果关系 | 对姿态对齐误差更敏感 |
| VAE prior 约束 latent | 推理时可从标准高斯采样 | prior 匹配可能牺牲部分细节锐度 |

---

## Part III：证据与局限

### 关键证据

#### 1. 比较信号：它不仅能采样，而且采样质量明显更高
在 DeepFashion 的 pose-guided appearance sampling 上，HumanGAN 同时兼顾了：
- ** realism **：FID 24.9，显著优于 VUNet 的 50.0，以及 Pix2PixHD+Noise / +WarpedNoise 的 109.4 / 101.9
- ** diversity **：LPIPS diversity 0.219，也高于 VUNet 0.182，远高于几乎塌缩的 Pix2PixHD 系基线

这说明它不是简单地“更清晰”，而是确实找到了一个**既可采样又可控**的外观空间。

#### 2. 比较信号：虽然不是专门为 pose transfer 设计，但迁移能力仍然很强
在 176 对 pose transfer 测试上：
- SSIM 达到 **0.777**，超过 CBI、DPT、DSC、NHRR、VUNet
- LPIPS 为 **0.187**，不是最优，但保持竞争力

这表明 HumanGAN 的部位级 canonical 外观表示确实能在换姿态时保持身份与衣着一致性，而不是只会随机采样。

#### 3. 消融信号：局部可控性来自“部位级 latent + warping”，不是自然涌现
与 NoParts 基线相比：
- 目标部位变化仍然存在
- 非目标区域变化从 **0.44 降到 0.11**

这个结果很关键：它支持论文最核心的因果主张——**局部可控性来自显式的部位级表示，而非单纯靠更强的生成器。**

#### 4. 用户研究信号：人类主观感受更偏好它的真实性
- appearance sampling 相比 VUNet：**91.06%** 偏好 HumanGAN
- pose transfer 中：
  - realism：**81.25%** 偏好 HumanGAN
  - identity preservation：**65.62%** 偏好 HumanGAN

这补强了“指标提升对应感知提升”的论点。

### 局限性

- **Fails when**: DensePose 对齐不准、自遮挡严重、服饰层次复杂或部位边界交错时，容易出现身体部位/衣物交叠的伪影；对少见服饰组合和非主流姿态的泛化也受限。
- **Assumes**: 依赖预训练 DensePose 与 SMPL UV 映射；训练需要同一人物跨姿态样本；主要在 DeepFashion 这类时尚单人图像分布上成立；复现实验还需要 512 分辨率的 GAN 训练资源。论文文本中给出项目页，但未明确说明代码发布。
- **Not designed for**: 多人场景、复杂背景交互、物理准确的试衣/布料模拟、精细 3D 几何一致性建模，也不是面向通用开放域人像生成。

### 可复用组件

1. **DensePose → UV 规范化外观提取**
   - 适合所有“想把姿态和外观分开”的人体生成任务

2. **part-wise latent broadcasting / warping**
   - 是一种很通用的“把局部 latent 显式绑定到空间区域”的做法

3. **只让生成器看空间化 latent，而不是强 pose 条件**
   - 对需要保留采样多样性的条件生成任务很有启发

### 一句话结论

HumanGAN 的真正贡献，不只是把人体图像生成做得更清晰，而是把**可采样性、跨姿态外观保持、局部部件控制**放进了同一个统一框架；其关键杠杆是把外观表示从“全局图像 latent”改成了“姿态无关的部位级 canonical latent”。

![[paperPDFs/Digital_Human_Image_Based_Virtual_Try_On/CVPR_2021/2021_HumanGAN_A_Generative_Model_of_Humans_Images.pdf]]