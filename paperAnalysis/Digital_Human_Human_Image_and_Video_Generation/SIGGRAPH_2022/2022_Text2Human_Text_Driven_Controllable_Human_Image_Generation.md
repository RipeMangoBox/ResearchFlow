---
title: "Text2Human: Text-Driven Controllable Human Image Generation"
venue: SIGGRAPH
year: 2022
tags:
  - Others
  - diffusion
  - mixture-of-experts
  - vq-vae
  - dataset/DeepFashion-MultiModal
  - opensource/full
core_operator: "将人体生成拆成“姿态到解析图”和“解析图到图像”两阶段，并用层级纹理感知 VQ codebook、MoE 扩散采样器与前馈细级索引预测实现服饰形状与纹理的文本可控生成"
primary_logic: |
  人体姿态 + 服装形状/纹理文本 → 将文本映射为离散服装属性 → 先生成与姿态一致的 human parsing 固定几何与服装轮廓 → 再从层级纹理感知 codebook 中按纹理属性采样 coarse/fine 索引并解码 → 输出可控全身人像
claims:
  - "在给定 human parsing 和服饰纹理条件时，Text2Human-parsing 取得 22.95 FID，优于 Pix2PixHD、SPADE、MISC 和 HumanGAN-parsing [evidence: comparison]"
  - "纹理感知 codebook + MoE 对复杂纹理控制显著有效：floral/stripe 属性预测准确率从 20.59%/22.22% 提升到 70.59%/88.89% [evidence: ablation]"
  - "前馈细级索引预测将 fine-level 采样时间从约 25 分钟降到 0.6 秒，并在重建上优于 VQVAE2（LPIPS 0.0609，ArcFace 0.4869） [evidence: ablation]"
related_work_position:
  extends: "VQVAE2 (Razavi et al. 2019)"
  competes_with: "HumanGAN (Sarkar et al. 2021b); MISC (Weng et al. 2020)"
  complementary_to: "N/A"
evidence_strength: moderate
pdf_ref: paperPDFs/Digital_Human_Human_Image_and_Video_Generation/SIGGRAPH_2022/2022_Text2Human_Text_Driven_Controllable_Human_Image_Generation.pdf
category: Others
---

# Text2Human: Text-Driven Controllable Human Image Generation

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2205.15996), [Project](https://yumingj.github.io/projects/Text2Human.html), [Code](https://github.com/yumingj/Text2Human), [Dataset](https://github.com/yumingj/DeepFashion-MultiModal)
> - **Summary**: 该工作把文本可控人体生成拆成“先定服装轮廓、再补服装纹理”的两阶段流程，用层级纹理 codebook + MoE 采样解决复杂服饰纹理难控制、难生成的问题。
> - **Key Performance**: Parsing 条件下 FID 22.95；复杂纹理控制上 floral/stripe 属性准确率达 70.59%/88.89%。

> [!info] **Agent Summary**
> - **task_path**: 人体姿态图 / DensePose + 服装形状文本 + 服装纹理文本 -> 全身人体图像
> - **bottleneck**: 人体几何、服装轮廓与多尺度纹理在单一生成器中强耦合，导致复杂纹理既难真实又难按文本受控
> - **mechanism_delta**: 用 human parsing 中间表示先解耦几何，再将纹理生成改写为层级 VQ codebook 的条件采样，并通过 MoE 强化长尾纹理路由
> - **evidence_signal**: 同一数据集上的对比+消融同时支持：FID 最低，复杂纹理准确率大幅提升，fine-level 采样时间从 25min 降到 0.6s
> - **reusable_ops**: [几何-纹理解耦中间表示, 纹理感知层级 codebook]
> - **failure_modes**: [罕见姿态下出现结构伪影, 长尾 plaid/格纹纹理容易发糊]
> - **open_questions**: [如何摆脱离散属性词表支持开放文本, 如何在长尾纹理与非常规姿态上提升稳健性]

## Part I：问题与挑战

这篇论文解决的不是“普通 text-to-image”，而是一个更窄但更难的任务：**给定人体姿态，再用文本控制服装形状与纹理，生成完整的人体图像**。

### 1. 真正的问题是什么
难点不在“生成人”，而在**同时满足三类约束**：

1. **几何约束**：人物姿态要对。
2. **形状约束**：衣服轮廓要对，比如短袖、长裤、外套是否存在。
3. **纹理约束**：图案/材质要对，比如 floral、stripe、denim、knitted。

以往方法的问题是：
- pose-guided human generation 往往能生成“像人”的图，但**不能精细控制衣服属性**；
- parsing-guided synthesis 虽然能利用结构条件，但**复杂纹理容易塌成纯色或模糊图案**；
- 更关键的是，**文本监督并不是现成可用的**，因为公开人体数据集通常没有足够细的服装形状/纹理标注。

### 2. 真瓶颈在哪里
本文识别出的核心瓶颈其实有三层：

- **表示瓶颈**：直接从 pose + text 到 image，条件分布太复杂，几何与外观强耦合。
- **纹理瓶颈**：服装纹理本身是多尺度的，粗层是“图案结构”，细层才是“高频细节”；单尺度 latent 很难同时表示两者。
- **长尾控制瓶颈**：稀有纹理如 floral、stripe、plaid 很容易被常见的纯色/denim 模式淹没。

### 3. 输入/输出接口与边界条件
- **输入**：人体姿态图（文中实际用 DensePose 派生表示）、服装形状文本、服装纹理文本。
- **输出**：全身时尚人像图像。
- **边界条件**：
  - 文本并非真正开放词汇建模，而是先被映射到**预定义离散属性**；
  - 数据域主要是 **DeepFashion 风格、对齐后的全身人物**；
  - 更接近“可控服饰生成”，**不是身份保持的人像编辑/试穿**。

### 4. 为什么现在值得做
一方面，虚拟试衣、2D avatar、服装设计草图等场景都需要**非专业用户可直接操作的文本接口**；另一方面，VQ 表示、扩散式 token 采样、DensePose，以及作者补充构建的 **DeepFashion-MultiModal** 数据集，共同让这个任务第一次具备落地条件。

---

## Part II：方法与洞察

### 方法总览
Text2Human 的核心思想是：**不要一次性把“姿态、衣服形状、衣服纹理、人体外观”全塞进一个生成器里**，而是拆成两个因果上更干净的阶段。

#### Stage I：Pose-to-Parsing
先根据：
- 人体姿态
- 形状文本（如短袖、长裤、外套、领型）

生成 **human parsing / 人体解析图**。

这里文本被先转成离散属性，再做 embedding，并与 pose 特征融合。  
这一步的作用不是追求照片 realism，而是**先把几何布局与服装轮廓固定下来**。

#### Stage II：Parsing-to-Human
再根据：
- 解析图
- 纹理文本（如 floral、stripe、denim）

生成最终图像。

这一步是本文真正的技术重点，包括三件事：

1. **层级 VQVAE**
   - top-level codebook：负责粗粒度纹理结构；
   - bottom-level codebook：负责细粒度纹理残差与细节。
   
2. **Texture-aware codebook**
   - 不同纹理类型分开建 codebook；
   - 目的是避免下采样后不同纹理在粗尺度上“看起来太像”，造成表示混淆。

3. **MoE diffusion-based sampler + 前馈细级预测**
   - coarse-level 索引由带 MoE 的扩散式 transformer 采样；
   - fine-level 不再用慢速自回归采样，而是由前馈网络直接预测，用于 refinement。

### 核心直觉

#### 什么改变了
作者把原来的问题：

**直接建模**  
`pose + text -> image`

改成了：

**分解建模**  
`pose + shape text -> parsing`  
`parsing + texture text -> image`

并进一步把“生成纹理”改成：

**从层级离散纹理词典中做条件采样**，而不是直接在像素空间硬生成。

#### 改变了哪个瓶颈
这几个改动分别改变了不同层面的约束：

- **两阶段分解**  
  把“几何/轮廓”和“外观/纹理”拆开，降低条件分布复杂度。
  
- **层级 codebook**  
  把多尺度纹理信息显式拆开：粗层管结构，细层管高频细节。
  
- **texture-aware + MoE**  
  把“所有纹理共用一个生成头”的竞争关系，改成“按纹理属性路由到专家头”，缓解长尾纹理被主流模式压制。
  
- **前馈 fine-level prediction**  
  把细粒度层级采样从慢速 sequential 过程，改为一次前向预测，直接解决时间瓶颈。

#### 为什么这套设计有效
因果上，它有效是因为：

1. **解析图是低熵中间变量**  
   先确定衣服在哪、长什么轮廓，比直接生成带纹理的人体图容易得多。

2. **纹理适合离散词典而非单一连续 latent**  
   服饰纹理本质上可被看成一些可重组的局部模式；用 codebook 存储，比让生成器每次从头“画纹理”更稳。

3. **长尾纹理需要条件路由**  
   floral/stripe 这类样本少、形态变化大的纹理，在共享头下最容易失败；MoE 等于给它们独立容量。

4. **fine-level 细化最好利用 coarse-level 提示**  
   粗层已经定下整体图案布局，细层只需补高频残差，因此前馈预测可行而且快。

### 关键模块拆解

#### 1) Pose-to-Parsing：把“文本控制”先落到结构层
这一步本质是在学：
- 文本里的“短袖/长裤/外套/领型”
- 如何转成 spatial layout

优点是：
- 控制更稳定；
- 后续生成器无需再同时猜“衣服轮廓”和“纹理”。

代价是：
- 如果 parsing 预测错，后面会**级联放大错误**。

#### 2) Hierarchical Texture-aware VQVAE：多尺度纹理记忆库
这里的设计非常关键：
- coarse codebook 存“纹理结构”；
- fine codebook 存“细节残差”；
- 不同纹理属性用不同 codebook。

这相当于把“衣服纹理多样性”从一个难学的连续空间，改造成**按类别组织的离散记忆库**。

#### 3) Sampler with Mixture-of-Experts：在全局上下文中做纹理选择
作者没有为每种纹理训练一整套独立 transformer，而是：
- 共享全局上下文建模；
- 最后在 index prediction head 上做 expert routing。

所以它兼顾了：
- **全图一致性**：因为注意力仍然看全局；
- **属性专门化**：因为最终 token 预测走专家头。

#### 4) Feed-forward Codebook Index Prediction：把层级采样做快
传统层级 VQVAE 的问题是 fine-level 采样非常慢。  
本文直接从 coarse 特征预测 fine indices，本质是把 fine-level 看成**条件细化问题**，而不是再来一次昂贵的序列生成。

### 战略权衡

| 设计 | 解决的瓶颈 | 得到的能力 | 代价/风险 |
|---|---|---|---|
| 两阶段 pose→parsing→image | 几何与纹理强耦合 | 形状控制更稳定 | parsing 误差会传递 |
| 层级 codebook | 单尺度难表征复杂纹理 | 粗结构+细节同时保留 | 训练流程更复杂 |
| texture-aware codebook | 粗尺度纹理混淆 | 复杂纹理更可区分 | 依赖纹理标签 |
| MoE sampler | 长尾纹理被主流模式淹没 | floral/stripe 等控制更准 | 路由设计增加系统复杂度 |
| 前馈 fine-level prediction | 层级采样太慢 | 速度大幅提升，细节更清晰 | 依赖 coarse 特征质量 |
| 文本→离散属性映射 | 自然语言难直接监督 | 交互简单、可训练 | 不是开放词汇表达 |

---

## Part III：证据与局限

### 关键证据

#### 1) Comparison：在 parsing 条件生成上更真实、更可控
在“给定 human parsing + 纹理条件”的设置下，Text2Human-parsing 达到：
- **FID 22.95**，优于 Pix2PixHD、SPADE、MISC、HumanGAN-parsing；
- 在 floral / stripe / denim 上也有更高属性预测准确率。

这说明它不仅“图像像真”，而且**衣服纹理确实更符合条件**。

#### 2) Comparison：在 pose 条件生成上也更强
在“只给 pose”的设置下，Text2Human-pose 的：
- **FID 24.54**，优于 TryOnGAN 的 29.00 和 HumanGAN-pose 的 32.20；
- 复杂纹理比例也更高。

这表明它的优势不只来自 parsing 输入，而是整套结构分解+纹理建模方案确实提升了能力上限。

#### 3) User study：人看起来更真，纹理也更对
用户研究里，Text2Human 在：
- **photorealism 排名**
- **texture consistency 得分**

都最好。  
这类信号虽然主观，但和 FID/属性准确率方向一致，说明提升不是单一指标幻觉。

#### 4) Ablation：真正起作用的不是“大模型”，而是这几个因果旋钮
- **层级设计**：重建损失从 0.1415 降到 0.1192，说明多尺度纹理建模确实更强。
- **texture-aware + MoE**：floral/stripe 准确率显著提升，直接验证“专家路由”对长尾纹理有用。
- **前馈 fine-level prediction**：fine-level 采样从约 **25 分钟降到 0.6 秒**，同时 LPIPS/ArcFace 重建更好，说明它不只是更快，还更清晰。

### 局限性

- **Fails when**: 遇到训练集中少见的姿态，如交叉腿、明显侧身时，结构会出伪影；遇到长尾 plaid/格纹时，纹理容易发糊。
- **Assumes**: 依赖 DensePose、人工标注的人体解析与服装属性数据、预定义离散属性词表；虽然训练算力不算夸张（单张 V100），但**标注成本**是实质性门槛。
- **Not designed for**: 开放词汇自然语言细粒度控制、身份保持的人像生成、跨域到非时尚/非全身人体图片场景。

一个很重要的现实判断是：**它名义上是 text-driven，但本质更接近“自然语言前端 + 离散属性条件生成”**。  
所以它的可控性很强，但表达能力是被属性 taxonomy 限死的。

### 可复用部件

1. **human parsing 作为几何-外观解耦中间层**  
   很适合迁移到服装编辑、avatar 生成、可控人像合成。

2. **texture-aware hierarchical codebook**  
   对任何“结构先行、纹理后补”的生成任务都值得借鉴。

3. **MoE token sampler**  
   适合长尾属性、多模态且条件稀疏的 token 生成问题。

4. **coarse-to-fine 前馈细化**  
   是一种很实用的“替代慢速层级自回归采样”的系统套路。

### 一句话结论
Text2Human 的真正贡献，不只是“把文本加到人体生成里”，而是**找到了一条更合理的因果分解路径：先控制衣服长什么形，再控制衣服是什么纹理**。它因此把“可控性”和“复杂纹理真实感”同时往前推了一步。

![[paperPDFs/Digital_Human_Human_Image_and_Video_Generation/SIGGRAPH_2022/2022_Text2Human_Text_Driven_Controllable_Human_Image_Generation.pdf]]