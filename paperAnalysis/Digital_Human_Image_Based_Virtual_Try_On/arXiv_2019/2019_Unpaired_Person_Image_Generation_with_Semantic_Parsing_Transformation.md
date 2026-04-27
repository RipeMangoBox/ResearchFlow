---
title: "Unsupervised Person Image Generation with Semantic Parsing Transformation"
venue: arXiv
year: 2019
tags:
  - Others
  - task/person-image-generation
  - task/pose-guided-image-generation
  - semantic-parsing
  - cycle-consistency
  - semantic-aware-style-loss
  - dataset/DeepFashion
  - dataset/Market-1501
  - opensource/full
core_operator: "通过语义解析中间层把无配对跨姿态人物生成拆成结构变换与纹理合成两步，并用端到端训练把图像级监督回传到语义布局预测。"
primary_logic: |
  参考人物图像 + 源姿态 + 目标姿态
  → 先在语义解析空间预测目标姿态下的人体布局
  → 再按对应语义区域生成并迁移服饰纹理与人脸细节
  → 输出保持服装属性的目标姿态人物图像
claims:
  - "在 DeepFashion 上，E2E(Ours) 的 Inception Score 为 3.441，高于无监督 UPIS 的 2.971，并略高于监督 Def-GAN 的 3.439 [evidence: comparison]"
  - "在 Market-1501 上，E2E(Ours) 取得最佳 mask-IS 3.680 和 mask-SSIM 0.758，优于 PG2、Def-GAN 与 UPIS [evidence: comparison]"
  - "端到端联合训练相对两阶段预测语义图版本（TS-Pred）能进一步修复发型、袖长和肢体形状等语义错误，并使结果接近或超过使用目标真值语义图的 TS-GT [evidence: ablation]"
related_work_position:
  extends: "UPIS (Pumarola et al. 2018)"
  competes_with: "UPIS (Pumarola et al. 2018); Def-GAN (Siarohin et al. 2018)"
  complementary_to: "N/A"
evidence_strength: strong
pdf_ref: "paperPDFs/Digital_Human_Image_Based_Virtual_Try_On/arXiv_2019/2019_Unpaired_Person_Image_Generation_with_Semantic_Parsing_Transformation.pdf"
category: Others
---

# Unsupervised Person Image Generation with Semantic Parsing Transformation

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/1904.03379) · [Code](https://github.com/SijieSong/person_generation_spt.git)
> - **Summary**: 这篇工作把“无配对的跨姿态人物生成”改写成“先预测目标语义人体布局，再在语义区域内补纹理”的两阶段问题，从而显著缓解了非刚体形变和服饰属性保持之间的冲突。
> - **Key Performance**: DeepFashion 上 IS = 3.441；Market-1501 上 mask-IS = 3.680、mask-SSIM = 0.758（表 1 最优）

> [!info] **Agent Summary**
> - **task_path**: 参考人物图像 + 源姿态 + 目标姿态 -> 目标姿态下且保留服饰属性的人像图像
> - **bottleneck**: 无配对监督下，模型很难同时学习人体非刚体姿态变形与服装纹理/款式保持
> - **mechanism_delta**: 用语义解析图作为中间结构变量，把困难的像素级直接映射拆成“布局变换 + 语义对齐纹理生成”
> - **evidence_signal**: 两个数据集上的对比实验 + 语义模块消融都表明，E2E 训练优于无语义基线和 UPIS，且能修正语义图预测误差
> - **reusable_ops**: [语义解析中间空间, 语义感知风格损失]
> - **failure_modes**: [源图语义解析错误, 罕见或极端姿态导致目标语义图预测失败]
> - **open_questions**: [能否联合训练 human parser 与生成器, 能否在高分辨率与复杂遮挡场景中稳定泛化]

## Part I：问题与挑战

这篇论文研究的是**无监督/无配对的 pose-guided person image generation**：给定一张参考人物图像 \(I_{ps}\) 和目标姿态 \(p_t\)，生成一张新图像，使人物在目标姿态下，同时尽量保留原始服饰外观、纹理和人体外形。

### 任务接口
- **输入**：参考人物图像、源姿态、目标姿态，以及从外部模型提取的源语义解析图与 pose mask
- **输出**：目标姿态下的人物图像
- **训练条件**：只有无配对数据，看不到同一人物在目标姿态下的真实图像

### 真正的瓶颈是什么？
不是“GAN 不够强”，而是**直接从源图到目标图的映射耦合了两件同时很难的事**：
1. **结构变换**：人体是非刚体，手臂、腿、裙摆、袖长都会随着姿态发生复杂变化；
2. **外观保持**：衣服纹理、颜色、款式信息又必须被保留下来。

在有配对监督时，网络还能用像素级对应去“硬学”；但在无配对场景里，CycleGAN 类约束太弱，模型很容易：
- 把袖长、裤长改错；
- 把纹理洗掉；
- 生成局部肢体形状错误的图像。

### 为什么现在值得解决？
因为真实应用里，**同一人、同一套衣服、不同姿态的配对数据很难采集**，而电商、虚拟试衣、图像编辑恰恰更常见的是海量无配对人物图像。  
所以，谁能在无配对条件下稳定地保留服饰属性，谁就更接近真实可用系统。

### 边界条件
这篇方法默认：
- 是**单人图像**；
- 外部 pose estimator 和 human parser 基本可用；
- 语义类别较粗（10 类）但能覆盖上衣、裤子、手臂、腿等主要区域；
- 重点是**姿态迁移与服饰保持**，不是生成全新服装，也不是视频时序建模。

---

## Part II：方法与洞察

作者的核心策略很明确：**不要直接学“图像到图像”的硬映射，而是先学“语义布局到语义布局”，再学“纹理到纹理”的条件生成。**

整个系统分成两个模块：

1. **Semantic Parsing Transformation (HS)**  
   先把源人物的语义解析图变成目标姿态下的语义解析图；
2. **Appearance Generation (HA)**  
   再基于目标语义图，把参考图的服装纹理、颜色和人脸细节生成出来。

### 核心直觉

作者真正改动的“因果旋钮”是：

**从像素空间的直接变换，改成语义空间的结构变换 + 语义约束下的外观生成。**

这带来了三个层面的变化：

1. **分布被简化了**  
   在语义图空间里，网络不需要关心花纹、颜色、材质，只要关心“手臂、上衣、裤子在哪里”。  
   这让原本很难的非刚体形变学习，变成了更容易的布局预测问题。

2. **约束被变强了**  
   一旦目标语义图被预测出来，纹理生成就不再是“整图盲生成”，而是“上衣区域对上衣区域、裤子区域对裤子区域”的受限合成。  
   这直接提高了衣服属性保持能力。

3. **误差可以被下游反向修正**  
   两阶段独立训练时，语义图一旦预测错，后面生成器就只能在错误布局上作画。  
   端到端联合训练后，图像质量损失会反向推动语义图更合理，从而减轻 pseudo label 和 parser 噪声问题。

换句话说，这篇论文不是简单“多加一个语义图输入”，而是**把难点重排了**：  
先解决“人应该长成什么布局”，再解决“这块区域该长什么纹理”。

### 方法流程

#### 1. 语义解析变换模块 HS
输入包括：
- 源语义图 \(S_{ps}\)
- 源姿态与目标姿态
- 对应 pose mask

网络用一个 U-Net 风格的结构，把源人物的语义布局变到目标姿态下。

但无配对训练没有真值语义图怎么办？  
作者构造了**pseudo label**：
- 在训练集里为当前样本搜索一个“衣服类型相近、姿态不同”的语义图；
- 通过身体部位级仿射对齐评估相似性；
- 用这个搜到的语义图做伪监督。

这个模块的作用不是生成最终图像，而是学会：  
**目标姿态下，身体各语义区域应该怎么摆。**

#### 2. 外观生成模块 HA
有了目标语义图之后，再做图像合成就清晰很多：

- 一支编码参考图像的外观；
- 一支编码目标语义图；
- 用生成器把二者融合输出目标图像。

这里作者借用了 deformable skip connections 来更好地处理空间错位。  
训练上不依赖目标真图，而是使用：
- adversarial loss：保证真实感
- pose loss：保证姿态对齐
- content loss：保证 cycle consistency
- **semantic-aware style loss**：保证语义区域内纹理风格一致
- face adversarial loss：提升人脸自然度

其中最关键的是 **semantic-aware style loss**。  
相比 UPIS 的 patch-style loss，它不只盯关节点附近的小块，而是按语义区域统计风格，相当于显式告诉网络：

- 上衣纹理该从上衣区域迁移；
- 裤子纹理该从裤子区域迁移；
- 袖子长度和服饰边界也应该受语义图约束。

#### 3. 端到端联合训练
作者的训练顺序是：
1. 先预训练语义变换模块；
2. 再固定语义模块训练外观生成模块；
3. 最后全系统联合优化。

这样做的原因很实在：  
pseudo label 搜索有误差，human parser 也有误差。  
如果一直把语义图当“绝对真理”，后续生成器只能被动接受错误结构；  
联合训练则允许最终图像目标去**反向修正中间语义表示**。

### 战略取舍

| 设计选择 | 改变了什么瓶颈 | 带来的能力提升 | 代价/风险 |
|---|---|---|---|
| 语义解析作为中间空间 | 把“结构变形”和“纹理保持”解耦 | 更稳的人体轮廓、袖长/裤长更一致 | 强依赖 parser 质量 |
| 语义感知风格损失 | 从局部 patch 对齐改成语义区域对齐 | 更能保留服装纹理与款式 | 语义标签错时会错误传递风格 |
| 端到端联合训练 | 让图像级目标反向约束语义预测 | 能修复 pseudo label 与 parsing 噪声 | 训练更复杂，稳定性要求更高 |

---

## Part III：证据与局限

### 关键实验信号

#### 1. 对比实验：能力跳跃主要体现在“结构更对、纹理更稳”
- **DeepFashion**：E2E 版本取得 **IS 3.441**，高于无监督 UPIS 的 **2.971**，并略高于监督 Def-GAN 的 **3.439**。  
  这说明在“真实感/可辨别性”上，它已经不只是比无监督方法强，甚至接近或超过配对监督方法。
- **Market-1501**：E2E 版本取得最佳 **mask-IS 3.680** 与 **mask-SSIM 0.758**。  
  这里 mask 指标更重要，因为它更聚焦人物区域，能更直接反映身体形状与服饰生成质量。

作者也提醒了一个很重要的点：  
**SSIM 往往偏好模糊图像**，所以 DeepFashion 上虽然不是最高 SSIM，但视觉上反而更真实。这和很多生成论文的经验一致。

#### 2. 消融实验：真正起作用的是“语义中间层 + 联合训练”
- **Baseline（无语义）** 明显更差，说明直接从图像学 pose 变换，在无配对条件下很难同时兼顾结构和外观。
- **TS-Pred vs E2E**：E2E 明显能修复发型、袖长、手臂形状等错误，表明下游图像目标确实在帮助语义图变好。
- **TS-GT vs E2E**：在 DeepFashion 上，E2E 已接近用目标真值语义图的效果；在 Market-1501 上甚至更好，反过来说明低分辨率下 parser 本身很 noisy，而联合训练能部分“纠偏”。

#### 3. 损失设计分析：语义感知风格损失不是装饰
把 semantic-aware style loss 换成：
- mask-style loss，或
- patch-style loss，

都会出现轮廓发飘、纹理对不准的问题。  
这说明论文最值得复用的，不只是“加语义图”，而是**用语义区域定义风格对应关系**。

### 局限性

- **Fails when**: 源图语义解析本身出错；目标姿态非常罕见、复杂或扭曲时，语义变换模块会先失败，后续图像生成也随之崩掉。
- **Assumes**: 依赖外部 OpenPose 与 human parser；依赖能搜索到“衣服类型相近”的 pseudo semantic pair；默认单人、人体主体清晰；高分辨率训练还依赖 progressive strategy。
- **Not designed for**: 多人场景、严重遮挡、视频时序一致性、从无到有设计新服装、严格身份保持之外的开放域人物生成。

### 资源与复现假设
- **优点**：代码已公开，复现门槛相对友好。
- **关键依赖**：虽然不需要配对监督，但仍然“借用了”外部监督训练好的 parser 和 pose detector；因此它并非完全摆脱标注体系。
- **实际可扩展性**：系统效果上限受中间语义质量强烈限制，这也是它最主要的工程瓶颈。

### 可复用组件
1. **结构化中间表示先行**：先预测布局，再生成纹理，适合一切“几何变化大、外观要保持”的图像生成任务。
2. **语义区域级风格一致性**：比关节点 patch 对齐更稳，适合服饰、人体、场景部件等语义明确的任务。
3. **下游任务反向修正中间表示**：当中间标签 noisy 时，端到端联合训练往往比“先预测再冻结”更有效。

### 一句话结论
这篇工作的价值，不在于把 GAN 堆得更复杂，而在于它准确识别了无配对人物生成的真正瓶颈：  
**难的不是“画图”，而是先把人体结构摆对。**  
一旦把结构预测搬到语义解析空间，能力跃迁就自然发生了。

![[paperPDFs/Digital_Human_Image_Based_Virtual_Try_On/arXiv_2019/2019_Unpaired_Person_Image_Generation_with_Semantic_Parsing_Transformation.pdf]]