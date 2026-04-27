---
title: "Generating Person Images with Appearance-aware Pose Stylizer"
venue: IJCAI
year: 2020
tags:
  - Others
  - task/pose-guided-person-image-generation
  - gan
  - adaptive-normalization
  - attention
  - dataset/Market-1501
  - dataset/DeepFashion
  - opensource/full
core_operator: "将目标姿态作为内容、源人物外观作为风格，在多尺度生成器中通过自适应patch归一化逐层重耦合，生成人物图像"
primary_logic: |
  源人物图像 + 源/目标姿态 → 姿态引导注意力编码器提取人物外观并抑制背景干扰 → APS以目标姿态为内容、外观表征为风格，逐层用AdaPN调制并融合前景/背景 → 输出符合目标姿态且保留源人物外观的图像
claims:
  - "APS在Market-1501与DeepFashion上取得最高SSIM（0.312/0.775），并在PCKh上达到并列最佳（0.94/0.96），说明其同时保持了外观结构与姿态对齐 [evidence: comparison]"
  - "在相同卷积编码器下，仅将普通反卷积解码器替换为APS解码器，就把Market-1501的SSIM从0.205提升到0.301、mask-SSIM从0.750提升到0.804 [evidence: ablation]"
  - "AdaPN比AdaIN更适合空间依赖的人体生成：完整模型在DeepFashion上将SSIM从0.763提升到0.775、L1从0.102降到0.097，且训练后期重建误差继续下降 [evidence: ablation]"
related_work_position:
  extends: "StyleGAN (Karras et al. 2019)"
  competes_with: "PATN (Zhu et al. 2019); Deformable GANs (Siarohin et al. 2018)"
  complementary_to: "N/A"
evidence_strength: strong
pdf_ref: paperPDFs/Digital_Human_Human_Image_and_Video_Generation/IJCAI_2020/2020_Generating_Person_Images_with_Appearance_aware_Pose_Stylizer.pdf
category: Others
---

# Generating Person Images with Appearance-aware Pose Stylizer

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2007.09077), [Code](https://github.com/siyuhuang/PoseStylizer)
> - **Summary**: 论文把“目标姿态”当作生成骨架内容、把“源人物外观”当作风格调制信号，在解码阶段逐层重耦合二者，从而比直接拼接潜变量的做法更稳定地保留服饰与身份细节。
> - **Key Performance**: DeepFashion 上 SSIM 0.775、PCKh 0.96；Market-1501 上 SSIM 0.312、PCKh 0.94

> [!info] **Agent Summary**
> - **task_path**: 源人物图像 + 源/目标姿态 -> 目标姿态下的人物图像
> - **bottleneck**: 解码阶段把姿态与外观粗暴拼接，导致几何结构、纹理细节与背景因素难以可控地重耦合
> - **mechanism_delta**: 用APS将“姿态=content、外观=style”逐层注入生成器，并用AdaPN替代全局式AdaIN做局部区域调制
> - **evidence_signal**: 双数据集比较中SSIM最优、PCKh并列最优，且APS解码器与AdaPN消融均带来稳定增益
> - **reusable_ops**: [pose-guided attention encoder, adaptive patch normalization]
> - **failure_modes**: [姿态检测误差会直接传导到生成几何, 非标准裁剪或人体空间位置漂移会削弱AdaPN]
> - **open_questions**: [如何摆脱对齐裁剪假设推广AdaPN, 如何更稳健地补全大遮挡下未观测身体区域]

## Part I：问题与挑战

这篇论文解决的是**pose-guided person image generation**：  
输入一张源人物图像 `Is`、源姿态 `Ps` 与目标姿态 `Pt`，输出一张新人物图像 `Ig`，要求：

1. **姿态跟随目标骨架**；
2. **外观保留源人物的衣服、肤色、身份细节**；
3. **图像看起来真实**。

### 真正难点不在“编码”，而在“如何解码重耦合”
作者的判断很准：过去方法已经会把前景/背景、姿态、外观做一定程度的解耦，但**解码时常常只是把这些表示直接拼接起来**。这会带来三个核心瓶颈：

- **姿态-外观耦合难题**：同一个人换姿态后，像素布局剧烈变化，简单拼接让解码器自己去学“哪块纹理该落到哪个身体部位”。
- **前景-背景干扰**：源图里背景复杂，若编码时不压制背景，外观表征会被污染。
- **局部细节-全局结构冲突**：大姿态变化时，既要先搭出整体人体结构，又要补回衣服纹理、脸部和肢体细节。

### 边界条件
这项方法默认的应用边界比较明确：

- 单人、已裁剪的人体图像；
- 有可靠的2D pose keypoints（文中用 OpenPose）；
- 训练时有**同身份、不同姿态**的图像对；
- 背景不是显式条件，因此评测里也使用了 masked 指标。

### Why now
这类问题之所以适合在当时推进，是因为 StyleGAN/SPADE 一类“**归一化层中注入条件**”的生成思路已成熟。作者把它进一步改写为：  
**姿态负责空间结构，外观负责纹理风格**，从而把 person generation 的核心难题搬到解码阶段、以更自然的方式处理。

---

## Part II：方法与洞察

整体框架由两部分组成：

1. **Appearance Encoder**：提取源人物的外观表示；
2. **Appearance-aware Pose Stylizer (APS)**：以目标姿态为骨架，逐层把外观“写”进去。

### 1）Appearance Encoder：先把“人”从背景里捞出来
编码器是双流结构：

- **image stream**：从源图抽视觉特征；
- **pose stream**：从源姿态抽结构特征，并产生 attention mask。

这个 attention mask 在每层都去筛选 image stream，让编码器更聚焦于**人物前景**而不是背景。  
所以它不是盲目地压缩整张图，而是在“姿态指导下”提取外观。

### 2）APS：把目标姿态当 content，把源外观当 style
这是论文最关键的改动。

APS 不是把姿态和外观先混成一个向量再解码，而是：

- 用**目标姿态图**作为生成内容骨架；
- 用**源人物外观表示**作为风格调制信号；
- 在每一层生成块中，通过 **AdaNorm** 逐层注入外观信息。

这样做的含义是：

- **姿态决定“哪里该长出什么结构”**；
- **外观决定“这些结构应该长成什么颜色、纹理、服饰样式”**。

这比一次性拼接 latent 更符合任务因果结构。

### 3）双流生成：前景先长，再与整图融合
APS 内部还有两个流：

- **foreground stream**：专门生成人体前景；
- **synthesized stream**：生成完整图像（前景+背景）。

前景流先在姿态约束下把人体长出来，再通过 attention 融入整图流。  
这使得模型不会一开始就把背景和人体混在一起学。

### 4）AdaPN：局部区域归一化，而不是整图一把梭
作者提出 **Adaptive Patch Normalization (AdaPN)**，区别于 AdaIN 的地方是：

- AdaIN：一个通道一组缩放/偏置；
- AdaPN：**不同空间区域**可以有不同的缩放/偏置。

直觉很简单：对于裁剪好的人体图像，

- 头通常在上方，
- 上衣在中上部，
- 腿和鞋在下方。

所以“头部区域”和“腿部区域”用同一组归一化调制参数并不合理。  
AdaPN 把“**全局统一调制**”改成“**局部区域调制**”，更适合人体这种空间布局相对稳定的对象。

### 核心直觉

以前方法的主要问题，不是不会“解耦”，而是**解码时又把所有东西粗暴混回去**。

这篇论文的关键因果链条是：

- **改变了什么**：从“先编码解耦、再一次性拼接解码”改为“每层生成时，用姿态提供空间结构，用外观提供风格调制”。
- **改变了哪个瓶颈**：  
  - 缓解了姿态结构与外观纹理之间的重耦合难题；  
  - 缓解了单一全局归一化无法覆盖不同身体区域的约束。
- **带来了什么能力变化**：模型更容易在大姿态变化下保持衣服样式、人物身份和关节点对齐，同时兼顾局部细节与整体人体结构。

换句话说，APS 的价值是把“**在哪里生成**”和“**生成成什么样**”拆成两种不同的控制旋钮。

### 战略权衡

| 设计选择 | 解决什么瓶颈 | 带来什么能力 | 代价/假设 |
|---|---|---|---|
| 姿态作 content、外观作 style | 减少解码阶段的几何-纹理混杂 | 更稳的姿态对齐与外观保持 | 强依赖姿态图质量 |
| Pose-guided attention encoder | 抑制背景污染外观表示 | 更聚焦人物前景 | 若姿态估计偏差大，注意力也会偏 |
| AdaPN 局部归一化 | 放宽“一组参数管整图”的限制 | 不同身体区域可学不同外观调制 | 假设人体在裁剪图中位置相对稳定 |
| 前景流 + 合成流 | 分开学人体与整图 | 细节与背景融合更自然 | 结构更复杂，训练成本更高 |
| 多尺度渐进生成 | 协调全局骨架与局部纹理 | 大姿态变化更稳 | 训练更长，可能过拟合 |

---

## Part III：证据与局限

### 关键证据信号

- **Comparison signal**：  
  在 **Market-1501** 和 **DeepFashion** 上，APS 都拿到最高 SSIM（0.312 / 0.775），同时在 PCKh 上达到并列最佳（0.94 / 0.96）。  
  这说明它的主要优势是：**结构一致性和姿态对齐**。

- **Ablation signal**：  
  在相同卷积编码器下，把普通反卷积解码器换成 APS 解码器，Market-1501 的 SSIM 从 **0.205 → 0.301**，mask-SSIM 从 **0.750 → 0.804**。  
  这直接支持论文核心论点：**真正的收益主要来自解码阶段的 progressive re-coupling**。

- **AdaPN signal**：  
  用 AdaPN 替代 AdaIN 后，完整模型在 DeepFashion 上 **SSIM 0.763 → 0.775，L1 0.102 → 0.097**；训练曲线还显示 AdaPN 在后期仍能继续下降重建误差。  
  说明**局部区域调制**确实比全局 style normalization 更适配人体图像。

### 需要诚实看的地方
这篇论文并不是“所有指标都碾压”。

- 它在 **IS** 上并不总是最强；比如 DeepFashion 上低于 VUNet，Market-1501 上也低于部分基线。  
- 这意味着它的提升更集中在**姿态/结构一致性与外观保持**，而非所有“真实性”指标都全面领先。  
- 作者也提到 DeepFashion 上较弱的 IS 可能与**一定程度的过拟合**有关。

### 局限性
- **Fails when**: 姿态估计错误、极端姿态变化、大面积自遮挡或未观测身体区域需要补全时，生成几何和细节都容易出错；若输入不再是对齐良好的单人裁剪图，AdaPN 的空间先验会变弱。
- **Assumes**: 依赖同身份跨姿态训练对、OpenPose 关键点质量、单人裁剪场景，以及“头/上身/下身在图中大致处于固定区域”的数据分布假设；训练还需要标准 GAN 优化与较长训练周期（文中为 800 epochs）。
- **Not designed for**: 多人场景、复杂场景级图像编辑、视频时序一致性建模、文本驱动换装或3D一致性生成。

### 可复用组件
这篇论文最值得迁移的，不只是一个完整模型，而是三个可复用算子：

1. **pose-guided foreground attention**：先把人物从背景里分离出来再做生成；
2. **pose-as-content / appearance-as-style 的逐层重耦合**：适合任何“结构条件 + 纹理条件”的生成任务；
3. **AdaPN**：适合空间布局相对稳定的条件生成任务，不限于 person generation。

## Local PDF reference

![[paperPDFs/Digital_Human_Human_Image_and_Video_Generation/IJCAI_2020/2020_Generating_Person_Images_with_Appearance_aware_Pose_Stylizer.pdf]]