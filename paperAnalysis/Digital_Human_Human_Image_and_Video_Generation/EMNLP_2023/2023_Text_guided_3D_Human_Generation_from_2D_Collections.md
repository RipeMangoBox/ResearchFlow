---
title: "Text-guided 3D Human Generation from 2D Collections"
venue: EMNLP
year: 2023
tags:
  - Others
  - task/text-to-3d-human-generation
  - cross-modal-attention
  - compositional-rendering
  - adversarial-training
  - dataset/DeepFashion
  - dataset/SHHQ
  - repr/SMPL
  - opensource/no
core_operator: 以SMPL约束的分部神经渲染为骨架，把服饰文本通过跨模态注意力注入对应身体部位，并用语义判别器强化局部服饰-部位一致性
primary_logic: |
  文本服饰描述 + 人体SMPL形状/姿态条件 → 逆LBS映射到规范空间并按16个身体部位做组合体渲染，词级跨模态注意力把局部服饰语义写入对应部位，语义判别器再用分割感知的fashion map约束细粒度对齐 → 一次前向生成可多视角渲染的3D人体
claims:
  - "CCH在DeepFashion的pose-guided T3H上取得21.136 FID、25.023 CLIP-S和72.038 FA，整体优于Latent-NeRF、TEXTure和CLIP-O [evidence: comparison]"
  - "跨模态注意力是文本可控性的关键：在DeepFashion 256x128消融中，相比仅条件GAN训练，加入CA将CLIP-S从21.079提升到24.103，并把FA从69.173提升到80.028 [evidence: ablation]"
  - "CCH把T3H推理缩短到0.372秒/样本，显著快于需要外部优化的Latent-NeRF、TEXTure和CLIP-O（均超过100秒） [evidence: comparison]"
related_work_position:
  extends: "EVA3D (Hong et al. 2023)"
  competes_with: "Latent-NeRF (Metzer et al. 2023); TEXTure (Richardson et al. 2023)"
  complementary_to: "MotionDiffuse (Zhang et al. 2022)"
evidence_strength: strong
pdf_ref: paperPDFs/Digital_Human_Human_Image_and_Video_Generation/EMNLP_2023/2023_Text_guided_3D_Human_Generation_from_2D_Collections.pdf
category: Others
---

# Text-guided 3D Human Generation from 2D Collections

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2305.14312), [Project](https://text-3dh.github.io)
> - **Summary**: 论文提出 CCH，把服饰文本按身体部位注入 SMPL 约束的组合式 3D 神经渲染中，使模型仅用 2D 图文集合就能一次前向生成可控、可多视角渲染的 3D 人体。
> - **Key Performance**: DeepFashion pose-guided 上达到 FID 21.136、CLIP-S 25.023、FA 72.038；单样本推理仅 0.372 秒。

> [!info] **Agent Summary**
> - **task_path**: 服饰文本 + SMPL形状/姿态条件 -> 可多视角渲染的3D人体
> - **bottleneck**: 仅靠2D图文集合时，局部服饰语义难以稳定绑定到3D人体对应部位；同时优化式text-to-3D推理过慢
> - **mechanism_delta**: 将EVA3D式分部人体渲染改造成“部位级文本注入”架构：每个身体部位用跨模态注意力读取相关词语，再用语义判别器检查局部服饰-部位一致性
> - **evidence_signal**: 双数据集比较 + 组件消融 + 推理效率表共同显示其在质量、文本对齐和速度上同时占优
> - **reusable_ops**: [SMPL引导的inverse-LBS规范化, 部位级跨模态注意力注入]
> - **failure_modes**: [SMPL估计误差导致几何与纹理退化, 窄视角训练数据导致多视角渲染伪影]
> - **open_questions**: [如何降低对SMPL与人体分割依赖, 如何扩展到含人脸和更开放服饰空间的高保真3D人体]

## Part I：问题与挑战

这篇论文解决的是一个很具体但也很难的任务：**Text-guided 3D Human Generation (T3H)**。目标不是普通 text-to-image，而是要生成一个**可从不同视角渲染、具备人体结构、并且服饰受文本控制的 3D human**。

### 真问题是什么？

难点其实有三层：

1. **几何层**：人体是 articulated object，姿态变化大，不能只靠静态 3D 形状生成。
2. **语义层**：服饰描述往往是**局部属性**，例如“long-sleeved denim upper”只对应上身，"three-point" 只对应下装长度；全局文本-图像相似度很难把这些词精确落到正确身体部位。
3. **监督层**：多视角/3D 人体数据贵，作者希望只用 **2D 图像-文本集合** 学习，而不是依赖视频或3D扫描。

### 为什么现在值得做？

因为应用需求非常明确：游戏、动画、虚拟人、数字角色定制都需要**可控**的 3D human creation。与此同时：

- 2D fashion/human collections 相对容易获得；
- 纯 text-to-3D 方法虽然能生成 3D 内容，但多数依赖外部 CLIP/扩散模型做**迭代优化**，速度很慢；
- 现有 3D human generation（如 EVA3D）已经能从 2D collection 学 3D-aware human，但**缺少语言控制**。

所以真正的瓶颈不是“能不能生成3D人体”，而是：

> **能否在仅有2D监督下，把细粒度服饰语义稳定地绑定到人体局部3D表示里，并且做到一次前向、可实用的生成速度。**

### 输入/输出与边界条件

- **输入**：服饰文本描述；以及人体 shape/pose 条件（训练时由图像估计的 SMPL 参数，测试时也可指定姿态）。
- **输出**：可渲染的 3D human。
- **任务边界**：
  - 主要控制的是**上衣/下装的形状、面料、颜色**；
  - 人脸在数据中被模糊处理，模型**不生成可识别人脸**；
  - 依赖 SMPL 参数与人体分割等前处理。

## Part II：方法与洞察

作者的方法叫 **Compositional Cross-modal Human (CCH)**。核心路线可以概括为：

1. 用 **SMPL + inverse LBS** 把不同姿态下的采样点映射回规范空间；
2. 把人体拆成 **16 个 body parts**，每个 part 有自己的局部体渲染网络；
3. 用 **词级跨模态注意力**，让每个部位只吸收与自己相关的服饰语义；
4. 再用一个带 **fashion map** 的判别器，强化局部服饰-部位的一致性。

### 方法主线

#### 1. 用 SMPL 先把“人体几何”问题压住
CCH 不是直接在任意 3D 空间里学文本到人体，而是借助 SMPL 这一人体先验，把姿态变化大的观察空间映射回更稳定的 canonical space。这样做的因果意义是：

- 原本模型要同时解释“衣服长什么样”和“人摆什么姿势”；
- 现在先用 SMPL/LBS 吸收掉大量姿态变化；
- 剩下的学习重点就更集中在**服饰外观如何附着到稳定的人体局部结构上**。

这使得只靠 2D 集合学习变得更可行。

#### 2. 分部建模，而不是整个人体一个全局文本条件
作者沿用 EVA3D 的 compositional human 思路，把人体拆成 16 个 body parts。每个 part 都有自己的局部 rendering MLP，最后再混合成完整人体。

这样做不是简单“模块化”，而是在信息路由上改变了问题：

- “denim” 更该影响上衣或裤子，而不是整个人体；
- “sleeveless” 应主要作用于上臂/肩部附近；
- “floral lower clothing” 不应污染上半身纹理。

把人体拆开后，文本控制就从“全局条件”变成了“局部对位条件”。

#### 3. 跨模态注意力：让每个部位去读相关词
文本由编码器提取成词级特征。对于每个采样点，模型根据其所属 body part，用 cross-modal attention 去选择相关词，再把得到的文本语义注入该部位的渲染特征。

这一步是整篇论文真正的“控制旋钮”：

- 不是靠 CLIP 这种**后验全局对齐分数**去逼迫 3D 模型慢慢优化；
- 而是把词语语义**直接写入生成网络内部**；
- 因此文本控制从“外部指导”变成“内部条件化”。

#### 4. 语义判别器：让细粒度对齐被显式检查
作者还引入了 **Semantic Discrimination**。做法是：

- 先从真实图像得到 segmentation map；
- 再结合文本做出一个部位感知的 fashion map；
- 判别器不只看图像真假，还看“图像和这张 fashion map 是否匹配”。

其作用是让判别器从“这像不像人”升级为“这是不是**带着正确局部服饰语义的人**”。  
这会直接推动模型在上/下装、颜色、面料等细粒度属性上更一致。

### 核心直觉

以前的方法大致有两个问题：

- **几何约束弱**：文本到 3D 往往靠外部优化，人体结构并不稳定；
- **语义绑定粗**：全局 CLIP 对齐知道“整体像不像文本”，但不知道“哪个词该落在哪个身体部位”。

CCH 的关键变化是：

> **把“全局文本指导的3D生成”改成“SMPL约束下、部位级的文本-几何绑定”。**

这带来的连锁变化是：

- **what changed**：文本不再作为全局弱条件，而是被路由到各身体部位；
- **which bottleneck changed**：语义绑定空间从“整个人体”缩小为“局部部位”，同时姿态变化由 SMPL 吸收；
- **what capability changed**：模型能在 2D 集合监督下学到更稳定的服饰-几何对应关系，并实现一次前向的可控 3D human generation。

#### 为什么这套设计有效？

1. **SMPL 先验减少了分布难度**  
   姿态变化被 canonicalization 部分吸收，模型不用从零学人体 articulation。

2. **局部注意力缩小了语义搜索空间**  
   词语和身体部位之间的匹配变得更容易学，细粒度属性更容易对齐。

3. **判别器补上了“局部语义监督”**  
   如果只靠 GAN 或全局相似度，模型容易生成“整体看起来合理、细节不对位”的服饰；fashion map 让这种错误更容易被发现。

#### 战略权衡

| 设计选择 | 改变了什么约束 | 带来的能力 | 代价/风险 |
|---|---|---|---|
| SMPL + inverse LBS | 把姿态变化归一到规范空间 | 2D监督下也能学更稳的人体几何 | 强依赖SMPL估计质量 |
| 16-part compositional rendering | 把整个人体拆成局部可控单元 | 更容易把局部服饰语义绑定到对应部位 | 模型更复杂，部位边界可能有拼接伪影 |
| 词级跨模态注意力 | 从全局文本条件改为局部语义注入 | 更强的上/下装细粒度控制 | 仍依赖文本描述质量与词义覆盖 |
| Semantic discrimination | 判别器也检查“部位-服饰”匹配 | 提升细粒度一致性和属性正确率 | 依赖分割质量，训练更重 |

## Part III：证据与局限

### 关键实验信号

#### 1. 比较实验信号：不仅更真，也更“听话”
在 **DeepFashion** 和 **SHHQ** 的 pose-guided 设定下，CCH 相比 Latent-NeRF、TEXTure、CLIP-O 在 realism、geometry、pose correctness、text relevance 上都更均衡。

最关键的信号不是单一指标，而是**多指标同时占优**：

- DeepFashion pose-guided：**FID 21.136**, **CLIP-S 25.023**, **FA 72.038**
- SHHQ pose-guided：**CLIP-S 27.855**, **FA 76.194**

这说明它不是只提升“像真人”，也不是只提升“像文本”，而是把**3D质量和服饰可控性同时抬高**。

#### 2. 消融信号：真正起决定作用的是“局部文本注入”
消融表明：

- 没有文本时，FA 极低，说明文本对 controllability 是必要条件；
- 仅用传统 conditional GAN，文本能进模型，但控制仍然偏粗；
- 加入 **Cross-modal Attention** 后，CLIP-S 和 FA 都显著提升；
- 再加 **Semantic Discrimination**，细粒度一致性继续提升。

这直接支持论文核心因果链：  
**不是“有文本就行”，而是要把文本按身体部位注入，并在判别端继续检查局部语义对齐。**

#### 3. 效率信号：从“优化式使用”变成“生成式使用”
CCH 最大的实用跳跃之一是速度：

- CCH：**0.372 秒**
- TEXTure：103.7 秒
- CLIP-O：181.6 秒
- Latent-NeRF：755.7 秒

这说明它把 text-to-3D human 从“每次都要慢速优化”的范式，推进到更接近**一次前向推理**的范式。

#### 4. 人评信号：服饰相关性最强
人工评测中，CCH 在 **fashion relevance** 上排名最高，和自动指标趋势一致。  
这进一步说明它的提升不是 evaluator 偏差导致的假象。

### 能力跳跃到底体现在哪里？

相对 prior work，这篇论文最大的能力跳跃在于：

1. **从无控制的人体3D生成 → 文本可控的人体3D生成**
2. **从全局粗文本对齐 → 局部部位级服饰对齐**
3. **从外部优化式 text-to-3D → 内部条件化的一次前向生成**

最有说服力的实验信号，是 **FA/CLIP-S 的提升 + 0.372 秒推理速度** 这组组合证据。

### 局限性

- **Fails when**: SMPL 参数估计不准时，几何与纹理会明显退化；训练数据视角覆盖窄时，多视角渲染会出现裂缝、纹理不连续和3D一致性伪影。
- **Assumes**: 依赖 OpenPose + SMPLify-X 提供人体参数，依赖人体分割生成 fashion map；SHHQ 文本标注来自 fine-tuned GIT 自动生成，存在标注噪声风险；训练成本不低（8×A100，1M iterations）。
- **Not designed for**: 可识别人脸生成、身份保持、开放域服饰/配饰全覆盖、无人体模板条件的自由3D人类生成。

### 可复用组件

- **SMPL 引导的 canonicalization**：适合任何需要从 2D human collection 学 3D-aware representation 的任务。
- **部位级跨模态注意力**：适合局部语义必须精确落位的生成问题，不限于人体。
- **语义判别器 + fashion map**：适合需要“局部属性一致性”而非只看整体真实感的条件生成任务。

## Local PDF reference

![[paperPDFs/Digital_Human_Human_Image_and_Video_Generation/EMNLP_2023/2023_Text_guided_3D_Human_Generation_from_2D_Collections.pdf]]