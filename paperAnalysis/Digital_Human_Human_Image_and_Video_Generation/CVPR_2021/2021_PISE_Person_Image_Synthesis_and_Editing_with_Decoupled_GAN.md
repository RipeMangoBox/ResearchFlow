---
title: "PISE: Person Image Synthesis and Editing with Decoupled GAN"
venue: CVPR
year: 2021
tags:
  - Others
  - task/human-pose-transfer
  - task/person-image-editing
  - gan
  - human-parsing
  - spatial-aware-normalization
  - dataset/DeepFashion
  - repr/human-parsing
  - repr/pose-keypoints
  - opensource/full
core_operator: "以人体解析图作为中间语义层，先解耦服装形状，再用全局+局部区域风格编码和空间感知归一化生成目标人物图像"
primary_logic: |
  源人物图像/源解析图/源姿态/目标姿态 → 解析生成器预测与目标姿态对齐的人体解析图 → 图像生成器按语义区域注入局部或全局风格并通过空间感知归一化迁移上下文 → 输出可控的人物合成与编辑结果
claims:
  - "在 DeepFashion 姿态迁移基准上，PISE 取得 FID 13.61、LPIPS 0.2059，并达到并列最佳 PSNR 31.38，优于或不劣于 PATN、BiGraph、XingGAN、GFLA、ADGAN 和 PINet [evidence: comparison]"
  - "将联合全局+局部区域编码替换为仅全局或仅局部编码后，FID 分别从 13.61 退化到 15.21 和 15.50，说明该设计提升了不可见区域的风格预测 [evidence: ablation]"
  - "移除 spatial-aware normalization 后，FID 从 13.61 变为 14.15、LPIPS 从 0.2059 变为 0.2071，表明空间上下文迁移对感知质量和真实感有贡献 [evidence: ablation]"
related_work_position:
  extends: "SEAN (Zhu et al. 2020)"
  competes_with: "PINet (Zhang et al. 2020); ADGAN (Men et al. 2020)"
  complementary_to: "N/A"
evidence_strength: moderate
pdf_ref: paperPDFs/Digital_Human_Human_Image_and_Video_Generation/CVPR_2021/2021_PISE_Person_Image_Synthesis_and_Editing_with_Decoupled_GAN.pdf
category: Others
---

# PISE: Person Image Synthesis and Editing with Decoupled GAN

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2103.04023), [Code](https://github.com/Zhangjinso/PISE)
> - **Summary**: PISE 通过“先生成人体解析图，再按语义区域注入服装风格，并用空间感知归一化迁移细节”的两阶段框架，在大姿态变化下同时提升人物合成质量与编辑可控性。
> - **Key Performance**: DeepFashion 姿态迁移上 **FID 13.61**、**LPIPS 0.2059**。

> [!info] **Agent Summary**
> - **task_path**: 单张源人物图像 + 源姿态/源解析图 + 目标姿态或编辑后的语义布局 -> 目标姿态/目标布局的人物图像
> - **bottleneck**: 大姿态变化下不可见区域难以合理补全，且服装形状与纹理耦合导致编辑不灵活
> - **mechanism_delta**: 把人物生成拆成“目标解析图预测 + 区域风格注入 + 空间上下文迁移”三步，而不是直接从源图像重排到目标图像
> - **evidence_signal**: DeepFashion 上对 6 个基线取得最优 FID/LPIPS，且联合区域编码与空间感知归一化都有独立消融收益
> - **reusable_ops**: [human-parsing-as-intermediate, global-local-region-style-code]
> - **failure_modes**: [pose/parsing 估计错误会级联到最终图像, 仅 8 类解析标签限制细粒度服饰与配件编辑]
> - **open_questions**: [如何在强遮挡下恢复更独特的纹理而非平均化先验, 如何扩展到时间一致的视频人物编辑]

## Part I：问题与挑战

**Q1 What/Why：真正瓶颈是什么，为什么现在值得解决？**

PISE 处理的是**单张人物图像的姿态迁移与可控编辑**。输入包括源人物图像、源姿态、源人体解析图，以及目标姿态；在编辑场景下，目标解析图还可以被人工修改。输出是同一人物在新姿态或新服饰布局下的自然图像。

这类问题难，不是因为“从图像生成图像”本身，而是因为以下三个瓶颈同时存在：

1. **服装形状与纹理样式耦合**  
   许多方法直接从源图像到目标图像，网络很难把“衣服是什么形状/类别”和“衣服是什么纹理/材质”分开。结果是能换姿态，但不容易做精细编辑，比如只改裤子、只换上衣纹理、只改头发形状。

2. **不可见区域没有直接证据**  
   大姿态变化会带来自遮挡。flow/warping 类方法擅长搬运源图像里已经看得见的部分，但对于目标图像中新露出的区域，本质上缺少可直接复制的内容，因此容易生成不合理的纹理或结构。

3. **空间上下文关系容易丢失**  
   即使使用区域级风格控制，如果只做区域平均池化，得到的是“这个区域总体像什么”，而不是“纹理细节在空间上如何组织”。这会影响边缘、褶皱、局部纹理和整体自然度。

**为什么现在要解决？**  
人物图像合成已被广泛用于虚拟试衣、数字人编辑、视频角色生成等场景。以往方法已经能做基本 pose transfer，但在“**可编辑性**、**遮挡补全**、**细节自然度**”三者上很难同时兼顾。PISE 的切入点就是把这三件事统一到一个框架里。

**输入/输出边界条件**
- 输入：单人图像，外加 2D pose 与人体解析图。
- 输出：同一身份人物的新姿态图像，或基于编辑后解析布局的编辑图像。
- 任务边界：不是文本驱动，也不是多人物场景，更不是时序一致的视频生成。

## Part II：方法与洞察

**Q2 How：作者拧动了哪个关键因果旋钮？**

### 方法骨架

PISE 采用一个**两阶段 Decoupled GAN**：

1. **Parsing Generator：先预测目标人体解析图**
   - 输入：源姿态 \(P_s\)、目标姿态 \(P_t\)、源解析图 \(S_s\)
   - 输出：与目标姿态对齐的解析图 \(S_g\)
   - 作用：先确定“人应该穿成什么形状”，把服装轮廓、身体区域布局显式化。
   - 实现上使用 gated convolution，适合处理源/目标不对齐的问题。

2. **Image Generator：再根据解析图生成最终图像**
   - 输入：源图像 \(I_s\)、源解析图 \(S_s\)、生成解析图 \(S_g\)、目标姿态 \(P_t\)
   - 输出：最终人物图像 \(I_g\)
   - 作用：把“形状”固定在目标解析图上，再把“风格/纹理”和“空间细节”从源图像迁移过来。

### 核心直觉

PISE 的核心变化可以概括为：

**直接像素/特征重排 → 语义形状先行 → 区域风格后注入 → 空间关系再对齐**

这背后改变了三个关键约束：

- **从稀疏关键点约束，变成稠密语义布局约束**  
  以前很多方法只知道几个关键点，很难稳定决定“袖子、裤腿、裙摆”的真实形状。PISE 先生成人体解析图，相当于先给出了更稠密的语义结构，因此服装形状更稳定、也更可编辑。

- **从“不可见区域无条件”变成“不可见区域有全局人物先验”**  
  对于目标图像里可见但源图像里不可见的区域，SEAN 式纯局部区域编码会遇到“没有该区域就只能置零”的问题。PISE 改为：
  - 区域在源图像中可见时，用**局部区域特征**
  - 区域在源图像中不可见时，用**全局人物特征**
  
  这改变了信息瓶颈：网络不再在不可见区域上“裸生成”，而是带着人物整体服饰搭配先验去生成。

- **从区域均值风格，变成带空间关系的风格迁移**  
  平均池化能保留“样式类别”，但会丢掉“局部纹理位置关系”。PISE 额外从源图像特征中提取空间相关的 scale/bias，并通过特征相关性矩阵把这些信息映射到目标布局上，从而保留更强的空间上下文。

### 关键模块

#### 1. 解析图中间表示：先解耦“形状”
解析图生成器本质上把 pose transfer 先转化为一个更容易控制的中间任务：  
**源解析图 + 源/目标姿态 → 目标解析图**

这一步把“衣服形状、人体部位布局”显式预测出来，使后续图像生成不必同时承担“结构推断 + 纹理合成”两种难题。

#### 2. Joint Global and Local Per-region Encoding：再解耦“风格”
作者把源图像编码成区域级 style code：
- 如果某语义区域在源图像中存在，取该区域的**局部平均特征**
- 如果该区域在源图像中不存在，退化为整个人物图像的**全局平均特征**

这一步的因果作用很明确：  
它让网络对可见区域“尽量忠实复制”，对不可见区域“合理推断而非瞎猜”。

#### 3. Per-region Normalization：按区域把风格注入到目标布局
有了目标解析图 \(S_g\) 和每个区域对应的 style code，网络可以针对目标解析图中的每个语义区域，预测其 scale/bias，并对生成特征逐区调制。  
这使得：
- 服装**形状**由解析图控制
- 服装**风格**由区域 style code 控制

因此编辑时可以单独改解析图，或替换某个区域的 style code。

#### 4. Spatial-aware Normalization：把“空间细节”也迁过去
区域 style code 经过平均池化后，天然丢失空间布局。PISE 的补救办法是：
- 从源图像特征中提取位置相关的 spatial scale/bias
- 用生成特征与源图像 VGG 特征之间的相似性，建立软对应关系
- 把这些空间调制量从源图像传播到目标特征

所以它不是只在回答“这块区域像什么材质”，还在回答“这些细节应该落在什么位置”。

### 设计取舍

| 设计选择 | 改变了什么瓶颈 | 获得的能力 | 代价/风险 |
| --- | --- | --- | --- |
| 解析图作为中间表示 | 从稀疏关键点到稠密语义布局 | 形状更稳定、编辑更自然 | 依赖解析图质量，误差会传递 |
| 全局+局部区域编码 | 不可见区域不再是零条件 | 遮挡区域能做更合理风格补全 | 全局特征可能平滑掉个别独特纹理 |
| Per-region normalization | 形状与风格分离 | 可做局部纹理替换、区域编辑 | 依赖语义分区足够细致 |
| Spatial-aware normalization | 风格不仅有类别，还有空间组织 | 细节更锐利、上下文更自然 | 需要额外 VGG 对应关系与计算开销 |

## Part III：证据与局限

**Q3 So what：相对 prior work，能力跃迁体现在哪里？**

### 关键证据信号

1. **标准基准比较：不仅更像真图，也更像目标人**
   - 在 DeepFashion 测试集上，PISE 达到 **FID 13.61**、**LPIPS 0.2059**，并取得并列最佳 **PSNR 31.38**。
   - 相比 PATN、XingGAN、BiGraph、GFLA、ADGAN、PINet，PISE 的提升主要体现在：
     - 更好的真实感（FID）
     - 更好的感知一致性（LPIPS）
   - 这说明它不是只优化像素误差，而是真正在“自然度 + 身份/服饰一致性”上获益。

2. **消融 1：联合全局+局部编码确实在解决不可见区域问题**
   - 只用全局编码：FID 15.21
   - 只用局部编码：FID 15.50
   - 完整模型：FID 13.61
   - 结论：可见区域需要局部细节，不可见区域需要全局先验，两者缺一不可。

3. **消融 2：空间感知归一化确实在补空间细节**
   - 去掉 spatial-aware normalization 后，FID 从 13.61 退化到 14.15，LPIPS 从 0.2059 退化到 0.2071。
   - 虽然无该模块时 PSNR 略高，但完整模型在感知质量和真实感上更优。
   - 这说明空间上下文迁移提升的是“看起来对不对”，而不只是“像素对不对”。

4. **案例信号：同一框架支持 texture transfer / interpolation / region editing**
   - 论文展示了上衣/裤子纹理迁移、纹理插值、区域编辑等案例。
   - 这些结果支持“形状-风格解耦”确实可操作。
   - 但要注意：这部分主要是**定性案例**，没有独立量化基准，所以这也是整体证据强度只能评为 `moderate` 的原因之一。

### 局限性

- **Fails when**: 目标姿态与源姿态差异极大且伴随强自遮挡时，模型仍可能把不可见区域生成成“合理但不一定真实”的平均化纹理；若 pose estimator 或 parsing extractor 出错，也会在最终图像中放大为结构伪影。
- **Assumes**: 依赖较准确的 2D 人体关键点与人体解析结果；训练与推理中还依赖外部 HPE、PGN 以及预训练 VGG-19 特征；语义标签被压缩为 8 类，这简化了学习，但也限制了细粒度服饰表达。
- **Not designed for**: 多人物交互、文本条件生成、开放域人物生成、时序一致的视频编辑，以及超高分辨率场景。

**复现/扩展依赖**
- 优点：代码已开源，没有 closed-source API 依赖。
- 约束：方法在单一数据集 DeepFashion、256×256 分辨率上验证，跨数据域泛化能力没有被系统证明。

### 可复用组件

- **human parsing as intermediate representation**：把复杂人物合成拆成“结构预测 + 图像渲染”。
- **visible/invisible split with local/global codes**：对可见区域用局部特征、对不可见区域用全局先验。
- **per-region normalization**：将语义区域级风格控制直接注入生成特征。
- **spatial correspondence-based modulation**：用相关性矩阵传播源图像的空间统计信息。
- **editing by parsing manipulation**：通过直接改解析图实现区域级编辑。

## Local PDF reference

![[paperPDFs/Digital_Human_Human_Image_and_Video_Generation/CVPR_2021/2021_PISE_Person_Image_Synthesis_and_Editing_with_Decoupled_GAN.pdf]]