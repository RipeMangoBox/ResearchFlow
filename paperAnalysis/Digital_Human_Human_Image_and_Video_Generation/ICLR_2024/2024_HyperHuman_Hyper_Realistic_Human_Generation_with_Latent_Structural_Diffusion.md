---
title: "HyperHuman: Hyper-Realistic Human Generation with Latent Structural Diffusion"
venue: ICLR
year: 2024
tags:
  - Others
  - task/human-image-generation
  - task/controllable-image-generation
  - diffusion
  - dataset/HumanVerse
  - dataset/MS-COCO
  - opensource/no
core_operator: "在共享扩散骨干中用结构专家分支同步去噪 RGB、深度与法线，并将预测结构组合成二阶段高分辨率精修条件。"
primary_logic: |
  文本描述+人体骨架 → 结构专家分支与共享骨干联合建模 RGB/深度/法线，并用同一 timestep 与零终端 SNR 稳定多模态对齐 → 组合预测结构进行高分辨率细化，输出姿态一致且更写实的人体图像
claims:
  - "Claim 1: 在 zero-shot MS-COCO Human 上，HyperHuman 在所比较的开源基线中取得最好的图像质量与姿态准确性指标，包括 FID 17.18、KID 4.11、APclean 38.84、ARclean 48.70 [evidence: comparison]"
  - "Claim 2: 将 RGB、深度与表面法线联合去噪，并配合结构专家分支，优于仅去噪 RGB 或 RGB+Depth 的变体；消融中 FID 从 21.68 / 19.89 降到 17.18 [evidence: ablation]"
  - "Claim 3: 零终端 SNR 与跨模态共享 timestep 对学习单调结构图至关重要；若为不同模态采样不同 timestep，FID 会从 17.18 恶化到 29.36，FIDCLIP 从 7.82 恶化到 18.29 [evidence: ablation]"
related_work_position:
  extends: "Stable Diffusion 2.0 (Rombach et al. 2022)"
  competes_with: "HumanSD (Ju et al. 2023b); T2I-Adapter (Mou et al. 2023)"
  complementary_to: "LoRA (Hu et al. 2021); Adapter (Houlsby et al. 2019)"
evidence_strength: moderate
pdf_ref: paperPDFs/Digital_Human_Human_Image_and_Video_Generation/ICLR_2024/2024_HyperHuman_Hyper_Realistic_Human_Generation_with_Latent_Structural_Diffusion.pdf
category: Others
---

# HyperHuman: Hyper-Realistic Human Generation with Latent Structural Diffusion

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2310.08579), [Project](https://snap-research.github.io/HyperHuman)
> - **Summary**: 这篇工作把人体的深度与法线从“外部控制信号”升级成与 RGB 同级的联合扩散生成目标，再用这些空间对齐的结构做二阶段精修，从而显著提升人体图像的姿态一致性与写实度。
> - **Key Performance**: Zero-shot MS-COCO Human 上 FID 17.18；APclean 38.84 / ARclean 48.70

> [!info] **Agent Summary**
> - **task_path**: 文本描述 + 人体骨架 -> 对齐的 RGB/Depth/Normal -> 高分辨率人体图像
> - **bottleneck**: 文本与稀疏骨架不足以约束非刚体人体的深度层次和表面几何，且外挂控制分支容易与主图像特征失配
> - **mechanism_delta**: 用共享骨干+结构专家分支把 RGB、深度、法线改成同步去噪目标，再把预测结构组合进二阶段 refiner
> - **evidence_signal**: zero-shot COCO-human 对比胜出 + 联合去噪/噪声日程消融
> - **reusable_ops**: [shared-backbone expert branches, same-timestep multi-target diffusion]
> - **failure_modes**: [手指和眼睛等细粒度部位仍会失真, 一阶段结构预测误差会传递到二阶段]
> - **open_questions**: [能否去掉必需的骨架输入并直接 text-to-pose, 能否在不依赖内部数据与超大算力下复现完整两阶段系统]

## Part I：问题与挑战

**这篇论文要解决的不是“会不会画人”，而是“能不能稳定画出结构正确、姿态自然、局部合理的人”。**

### 1) 真问题是什么
通用 text-to-image 扩散模型已经能生成高质量图像，但一到人体就经常出问题：  
- 四肢数量或连接关系错  
- 姿态僵硬、不符合骨架  
- 前后遮挡关系不自然  
- 局部几何，尤其手指、眼睛等细节容易崩

### 2) 真正瓶颈是什么
作者认为，**人体生成的核心瓶颈不是纹理能力，而是结构建模不足**：

1. **文本描述不够结构化**  
   文本能描述“一个人在滑雪”，但很难精确约束关节拓扑、身体朝向、深浅层次和表面几何。

2. **骨架只提供粗结构**  
   pose skeleton 只给出稀疏关键点，不能表达密集深度、局部表面朝向、前景-背景空间关系。

3. **现有 controllable T2I 把结构当外挂条件**  
   ControlNet / T2I-Adapter 这类方法把控制信号接到侧路，容易出现“控制信号在，图像主干没真正吸收”的分布失配。

### 3) 为什么现在值得做
因为两个条件同时成熟了：
- **预训练扩散骨干** 已经具备强纹理与开放词汇生成能力；
- **大规模人体数据与自动标注器** 让“把结构作为联合学习目标”变得可行。

作者为此构建了 **HumanVerse（340M 图像）**，给人体图像配上 pose、depth、surface normal、caption 等标注。

### 4) 输入/输出接口与边界
- **输入**：文本 caption + 人体骨架
- **第一阶段输出**：空间对齐的 RGB / depth / normal
- **第二阶段输出**：1024×1024 的高分辨率人体图像
- **边界条件**：
  - 主设定仍是 **pose-conditioned human generation**
  - 第二阶段可替换为用户自定义 depth/normal
  - 论文里的主量化结果主要评估第一阶段 RGB 输出

---

## Part II：方法与洞察

### 方法骨架

#### A. HumanVerse：先把“人类结构监督”做出来
作者从 LAION-2B-en 和 COYO-700M 中筛出高质量人体图像，保留 1-3 人、足够清晰且有人体占比的样本，再自动标注：
- 2D pose：MMPose + ViTPose-H
- depth / normal：Omnidata、MiDaS
- caption / attributes：现成视觉语言工具链

一个有意思的数据工程点是：**先 outpaint，再跑结构估计器**。  
原因是很多估计器更习惯看到“完整场景”，outpaint 能补足上下文，再只截回原图区域作为标注，提高深度/法线质量。

#### B. Latent Structural Diffusion：把结构变成联合生成目标
核心不是“给扩散模型更多条件”，而是：

- **同时去噪 RGB**
- **同时去噪 depth**
- **同时去噪 surface normal**

也就是说，模型不是只学“长什么样”，而是一起学：
- 外观纹理
- 空间前后关系
- 局部表面几何

#### C. 结构专家分支：小范围模态专用，大范围共享
作者没有训练 3 个完全独立的扩散模型，而是采用：

- **输入/输出两端为各模态保留专用分支**
- **中间大部分 U-Net 骨干共享**

这样做的因果逻辑是：
- 前端专用分支负责把 RGB / depth / normal 这种不同分布的输入变成“可融合”的中间表征
- 中间共享骨干强迫三者交换信息，形成统一的人体结构表示
- 后端专用分支再把共享表征还原成各自输出，同时保持空间对齐

这比“单独训练三个模型”更容易保持 RGB 与结构图一一对应。

#### D. 噪声日程与 timestep：为低熵结构图单独修正训练动力学
depth 和 normal 与 RGB 不同，它们往往：
- 颜色单调
- 局部值相近
- 低频统计更强

作者指出，默认 diffusion noise schedule 会让这类结构图泄漏低频信息，模型容易走“均值捷径”，不真正学结构。  
所以做了两件事：

1. **zero-terminal SNR**：尽量消除低频泄漏  
2. **三个模态共享同一个 timestep**：避免组合采样过稀，让共享特征更好融合

#### E. Structure-Guided Refiner：用预测结构做二阶段高清细化
第一阶段给出对齐的粗 RGB/depth/normal 后，第二阶段把：
- text
- pose
- predicted depth
- predicted normal

一起送入 refiner，生成更高分辨率结果。

为减少两阶段误差累积，作者在训练时对条件做 **random dropout**：
- 文本可置空
- 某一路结构图可置零

这样模型不会过度依赖单一路条件，在测试时即便第一阶段结构图有噪声，也更稳。

### 核心直觉

**关键变化**：从“结构作为外部控制”改成“结构作为内部联合生成目标”。

这带来了三个层面的因果改变：

1. **改变了信息瓶颈**  
   原先是 `text + pose -> RGB`，结构监督太弱；  
   现在是 `text + pose -> shared latent -> RGB + depth + normal`，模型必须解释人体的几何与空间关系。

2. **改变了分布约束**  
   原先控制分支和图像主干可能是“两套特征系统”；  
   现在共享中间骨干，逼迫 RGB 与结构在同一隐空间里对齐。

3. **改变了能力边界**  
   得到的不只是更“像照片”的人，而是更**姿态一致、肢体连贯、遮挡合理**的人。

一句话概括：  
**作者真正拧动的旋钮，不是更强文本编码器，而是“把人体生成从外观拟合升级为外观-空间-几何联合建模”。**

### 战略权衡

| 设计旋钮 | 改变的约束/分布 | 带来的能力 | 代价/风险 |
|---|---|---|---|
| RGB + depth + normal 同步去噪 | 从单外观分布变成多结构联合分布 | 更强的解剖一致性与几何合理性 | 需要额外标注与更复杂训练 |
| 专家分支 + 共享骨干 | 在模态适配与对齐之间折中 | 既能学不同模态，又能保持输出配对 | 共享太多或太少都会掉性能 |
| 同一 timestep + zero-terminal SNR | 减少跨模态采样稀疏，抑制低频捷径 | 结构图更好学，融合更稳 | 降低了模态独立噪声建模自由度 |
| 二阶段 refiner + 条件 dropout | 把粗结构转成高分辨率细节，并增强鲁棒性 | 高清结果更逼真 | 增加系统复杂度，仍受第一阶段上限约束 |

---

## Part III：证据与局限

### 关键证据

#### 1) 比较实验信号：在“结构正确性”上确实有跳跃
在 zero-shot **MS-COCO 2014 Human** 子集上，HyperHuman 相比通用 T2I 和 pose-control 基线都更强：

- **FID 17.18**
- **KID 4.11**
- **FIDCLIP 7.82**
- **APclean 38.84 / ARclean 48.70**

这说明它不只是“更会对齐文本”，而是把**图像质量 + 姿态一致性**一起拉起来了。  
值得注意的是，CLIP 分数只略低于更大文本栈的 SDXL，说明 HyperHuman 的增益主要来自**结构建模**而非更大语言侧容量。

#### 2) 消融信号：增益确实来自联合结构学习，而非偶然工程技巧
几个最关键的消融很说明问题：

- **只去噪 RGB** 不如 **RGB+Depth**，而 **RGB+Depth+Normal** 最好  
  → 说明几何结构 supervision 是有效的
- 专家分支复制层数过少或过多都不好  
  → 说明“模态专用 vs 共享融合”确实需要平衡
- **不同模态用不同 timestep** 会明显恶化结果  
  → 说明共享 timestep 不是小 trick，而是联合建模成立的必要条件之一

#### 3) 人类偏好信号：优势不只是指标好看
25 名标注者的成对比较里，用户对 HyperHuman 的偏好率相对各基线在 **60.45% 到 99.08%** 之间，说明它的改进在主观视觉上也成立。

### 局限性

- **Fails when**: 输入 pose/depth/normal 估计本身噪声较大，或需要极细粒度局部准确性时（尤其手指、眼睛等），结果仍会失真；第一阶段结构预测伪影也会传递到第二阶段。
- **Assumes**: 依赖大规模人体数据与自动标注管线，依赖预训练 SD 2.0 / SDXL 作为骨干，且算力开销极高（标注约 640 张 V100 跑两周；训练约 128 张 A100 和 256 张 A100 各一周）。另外，第二阶段使用了**内部数据**，仅用于视觉结果，这会影响完整复现性。
- **Not designed for**: 视频时序一致性、身份保持、多视角/3D 一致人体生成，以及把 pose 完全从主接口中拿掉的纯文本人体生成。

### 可复用组件

- **共享骨干 + 模态专家输入输出层**：适合多模态但需空间对齐的联合生成任务  
- **同一 timestep 的多目标扩散训练**：适合多个配对模态一起学  
- **zero-terminal SNR for low-entropy maps**：适合深度、法线、mask 等低纹理结构图  
- **二阶段多条件 refiner 的 random dropout**：适合缓解 pipeline 式误差累积

### 一句话结论
这篇论文最有价值的地方，不是又做了一个 pose-guided T2I，而是证明了：**对人体这类强结构对象，最有效的控制不是把结构“喂给模型看”，而是让模型“同时生成并对齐结构本身”。**

## Local PDF reference

![[paperPDFs/Digital_Human_Human_Image_and_Video_Generation/ICLR_2024/2024_HyperHuman_Hyper_Realistic_Human_Generation_with_Latent_Structural_Diffusion.pdf]]