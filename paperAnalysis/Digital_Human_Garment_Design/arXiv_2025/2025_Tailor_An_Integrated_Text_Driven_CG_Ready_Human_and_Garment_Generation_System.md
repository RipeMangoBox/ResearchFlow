---
title: "Tailor: An Integrated Text-Driven CG-Ready Human and Garment Generation System"
venue: arXiv
year: 2025
tags:
  - Others
  - task/text-to-3d-generation
  - task/garment-generation
  - diffusion
  - flow-matching
  - topology-preserving-deformation
  - dataset/HumGen3D
  - repr/polygonal-mesh
  - repr/hair-strands
  - opensource/promised
core_operator: LLM先把自然语言拆成身体参数与服装子任务，再结合参数化人体先验、身体约束的拓扑保持服装变形和同步多视图扩散纹理，输出可直接进入CG流程的人体与服装资产
primary_logic: |
  自然语言人物/穿搭描述 → LLM分解为身体参数、发型与服装模板/子提示 → HumGen3D生成人体并用NJF在身体几何约束下变形服装 → 同步多视图扩散与UV细化生成一致纹理 → 输出CG-ready、衣物解耦、可动画/可仿真的3D角色与服装
claims:
  - "Tailor在作者构建的10个文本到3D人物评测案例上取得最佳定量结果，CLIP 26.52、Aesthetic 4.11、IQ 0.595，优于SO-SMPL、HumanGaussian、DreamWaltz-G和Barbie [evidence: comparison]"
  - "Tailor在22个文本到3D服装案例上达到最高CLIP/FashionCLIP/Aesthetic/IQ（28.30/33.26/4.34/0.759），优于DressCode、GarmentDreamer、Garment3DGen和ChatGarment [evidence: comparison]"
  - "去除身体对齐约束、循环多视图融合或对称注意后，会出现穿模、位置漂移、多视图接缝或纹理不对称，说明这些模块直接影响可穿性与纹理一致性 [evidence: ablation]"
related_work_position:
  extends: "HumGen3D"
  competes_with: "Barbie (Sun et al. 2024); GarmentDreamer (Li et al. 2025)"
  complementary_to: "N/A"
evidence_strength: moderate
pdf_ref: paperPDFs/Digital_Human_Garment_Design/arXiv_2025/2025_Tailor_An_Integrated_Text_Driven_CG_Ready_Human_and_Garment_Generation_System.pdf
category: Others
---

# Tailor: An Integrated Text-Driven CG-Ready Human and Garment Generation System

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2503.12052) · [Project Page](https://human-tailor.github.io)
> - **Summary**: 这篇工作把“文本生成人体”和“文本生服装”重写成一个面向生产的CG资产流水线：先用LLM把语义拆成可执行参数与模板，再用带身体约束的服装几何变形和同步多视图扩散纹理，最终直接输出可动画、可换装、可仿真的3D角色。
> - **Key Performance**: 人体生成在10个评测prompt上达到 CLIP 26.52 / Aesth 4.11；服装生成在22个prompt上达到 FashionCLIP 33.26 / IQ 0.759，并在用户研究中整体排名第一。

> [!info] **Agent Summary**
> - **task_path**: 自然语言人物与穿搭描述 -> CG-ready人体网格/发丝 + 解耦服装网格 + 2K纹理
> - **bottleneck**: 现有text-to-3D常把身体与衣物纠缠在同一神经表示里，既不利于换装/仿真，也难直接进入CG管线；同时服装生成缺乏身体约束，容易穿模和失配
> - **mechanism_delta**: 将端到端隐式生成改为“LLM语义拆解 + 参数化人体先验 + 身体感知的模板变形 + 同步多视图纹理扩散”
> - **evidence_signal**: 人体与服装两组对比实验都拿到最优指标，且消融直接显示身体约束和多视图一致性模块是关键因子
> - **reusable_ops**: [LLM prompt decomposition, body-aware topology-preserving deformation]
> - **failure_modes**: [超出模板覆盖范围的大拓扑变化, 多层服装与口袋等非流形细节]
> - **open_questions**: [如何从模板变形走向拓扑生成, 如何支持多层服装与物理结构联合生成]

## Part I：问题与挑战

这篇论文真正想解决的，不是“能不能从文本生成一个3D结果”，而是：

**能不能从一段自然语言，直接产出可进入传统CG工作流的、身体与衣物解耦的、可动画/可仿真的角色资产。**

### 1. 现有方法的真瓶颈

**瓶颈一：表示层不对。**  
很多文本到3D人体方法依赖 NeRF / 3DGS / SDS 一类路线，渲染上可能有效，但对CG生产并不友好：  
- 身体和衣服常被纠缠在同一表示里；
- 换装、编辑、布料模拟都很不方便；
- 容易有 Janus、过平滑、纹理虚、脸部失真等问题；
- 很难直接接到 Blender / 动画 / cloth simulation 管线。

**瓶颈二：服装生成缺少“身体条件”。**  
单独的 text-to-garment 方法往往要么依赖 sewing pattern、模板偏置强；要么追求自由形状但缺少身体 awareness。结果就是：
- 衣服不一定贴合当前体型；
- 容易穿模、比例漂移；
- 生成后还得手工修。

**瓶颈三：纹理一致性难。**  
服装纹理比一般物体更麻烦：
- 自遮挡多；
- 非封闭网格常见；
- 左右对称图案很常见；
- 独立逐视角扩散很容易出接缝、花纹断裂。

### 2. 输入/输出接口

- **输入**：一段同时描述人物外观与服装的自由文本。
- **输出**：  
  - 参数化人体网格  
  - 发丝/发型资产  
  - 与身体解耦的服装网格  
  - 服装2K纹理  
  - 可导入CG软件继续做动画、渲染与布料模拟

### 3. 为什么现在值得做

因为 2D 侧的能力已经够强了：  
- LLM 足以把长文本解析成结构化控制信号；  
- 大型图像扩散模型足以提供高质量几何/纹理监督；  
- 但高质量3D clothed-human 数据仍然稀缺。  

所以这篇文章的策略不是再训练一个更大的端到端3D模型，而是**把问题拆成几个更受约束、可复用、训练自由（training-free）的模块**，从而绕开3D数据瓶颈。

### 4. 边界条件

Tailor 的适用边界很明确：  
- 面向**单人角色**；
- 主要支持**单件全身服装**或**上装+下装**；
- 依赖已有**人体参数空间**与**服装模板库**；
- 服装几何本质上是**模板保持拓扑的变形**，不是从零生成任意拓扑。

---

## Part II：方法与洞察

### 方法总览

Tailor 是一个三阶段系统：

1. **语义解析**：LLM 把用户文本拆成“身体描述”和“服装描述”，再分别转成身体参数、发型选择、服装模板与子提示。  
2. **服装几何生成**：从一个粗对齐的服装模板出发，用 NJF 做拓扑保持变形，并同时受文本图像模型和身体几何约束引导。  
3. **服装纹理生成**：用同步多视图扩散在多个视角上生成一致纹理，再回投到UV空间做修补、去阴影和超分。

这个设计的核心不是“更强的生成器”，而是**更强的中间表示与约束编排**。

### 核心直觉

**作者改变了什么？**  
把原来“直接生成整个人+衣服”的问题，改成了三个受限子问题：

- 人体：交给成熟的参数化CG人体先验；
- 衣服几何：限制为对模板的受控变形；
- 衣服纹理：限制为共享UV语义的多视图一致生成。

**这改变了哪个瓶颈？**  
- 把**语义理解瓶颈**从短文本编码器，转成 LLM 的显式程序化解析；
- 把**几何搜索瓶颈**从“开放式任意3D形状生成”，转成“在可穿、可仿真的拓扑内做受约束变形”；
- 把**纹理一致性瓶颈**从“事后拼接视图”，转成“在去噪过程中就持续共享同一套纹理假设”。

**带来了什么能力变化？**  
- 更长、更细粒度文本也能被解析；
- 人体与衣服天然解耦，支持换装和后续模拟；
- 服装更容易贴合身体，减少穿模和比例错误；
- 纹理跨视角更一致，且能显式控制对称图案。

### 为什么这套设计有效

#### 1. LLM 不是用来“生成3D”，而是用来“生成控制程序”
Tailor 让 GPT-4o 做三件事：
- prompt decomposition
- body parameterization
- garment template matching

这一步的因果意义是：  
**把模糊自然语言先变成结构化控制量**，再交给已有CG先验和生成模块执行。  
因此，系统不需要让一个单一3D模型同时学会“理解人物设定 + 理解服装 + 理解CG约束”。

#### 2. 人体先验把“可用性”前置了
作者直接采用 HumGen3D，而不是从零生成整个人体。  
这相当于直接获得：
- 合法拓扑
- rigging
- 发型资产
- CG兼容表示

所以 Tailor 的跳跃点不只是“图更好看”，而是**输出默认就是 production-oriented asset**。

#### 3. 服装几何用“模板变形”而不是“自由拓扑生成”
作者从模板库里选一个最接近的服装，再做 NJF 拓扑保持变形。  
同时加入三类关键约束：
- **身体碰撞约束**：避免衣服钻进身体；
- **blocking 约束**：避免袖子、裤腿、腰线漂到不合理位置；
- **可选对称约束**：适合常见服装的双边对称形态。

这一步本质上把问题从“生成一件像衣服的3D物体”变成“把这件衣服可靠地穿到这个人身上”。

#### 4. 纹理一致性在扩散过程中解决，而不是后处理补缝
Tailor 不是独立生成每个视角后再硬拼，而是做：
- 多视图同步扩散
- latent cyclic merging
- 前后视图加权
- 对称局部注意
- UV空间 refinement

这在因果上很关键：  
**每一步去噪都共享同一个纹理空间假设**，所以接缝和视角冲突不会被不断放大。  
而服装常见的“前胸图案”“后背图案”“左右对称花纹”，也能更稳定地保住。

### 三阶段机制拆解

#### 阶段一：语义解析与模板匹配
- 用 LLM 将描述拆成 body / garment 两条支路；
- body 支路映射到 HumGen3D 参数、发型模板、发色；
- garment 支路输出：
  - 服装类别（top / bottom / full_body）
  - 对应文本子提示
  - 几何/纹理是否对称
  - 匹配到的模板

**作用**：把长文本变成可执行控制图。

#### 阶段二：身体感知的服装几何生成
- 以粗对齐模板为起点；
- 用 text-to-image 模型提供语义几何监督；
- 用 Rectified Flow + ISM 风格优化替代传统 SDS 的不稳定梯度；
- 用身体碰撞与位置约束保证“穿得上、位置对、不过度膨胀”。

**作用**：生成的不是“看起来像某件衣服”的网格，而是“这个身体真的能穿”的网格。

#### 阶段三：一致服装纹理生成
- 渲染多个深度视图；
- 用 ControlNet 条件扩散生成多视图图像；
- 周期性回投 UV latent 再重投到各视图；
- 通过对称局部注意增强对称服装；
- 最后做 UV 修补、去阴影、遮挡区域补纹理和超分。

**作用**：从“多张看起来都还行的图”变成“一张能贴到3D网格上的统一纹理”。

### 战略权衡

| 设计选择 | 改变的瓶颈 | 收益 | 代价 |
|---|---|---|---|
| 参数化人体先验（HumGen3D） | 避免人体拓扑、绑定和脸部稳定性难题 | CG-ready、可动画、速度快 | 受商业工具和参数空间限制 |
| 模板保持的服装变形 | 避免开放式服装拓扑生成不稳定 | 贴身、平滑、可仿真 | 对大拓扑变化不友好 |
| 同步多视图扩散 + UV细化 | 避免逐视图生成的接缝与不一致 | 纹理统一、细节更稳、可做对称 | 流程更复杂，依赖预设UV |
| 训练自由模块化系统 | 规避3D数据稀缺 | 易替换组件、扩展性强 | 强依赖外部预训练/闭源组件 |

---

## Part III：证据与局限

### 关键证据：能力跃迁到底在哪里

**1. 比较信号：人体结果不只更高分，而且更“像资产”**  
在作者构建的 10 个文本到3D人物案例上，Tailor 的 CLIP / Aesthetic / IQ 全部最好。  
更重要的是，它输出的是：
- 可动画的人体；
- 分离的服装；
- 可做 subsurface scattering 等CG渲染；
- 整体耗时约 **12分40秒/A100**，而对比方法常需 **1小时到半天**。

**2. 比较信号：服装结果同时提升几何和纹理**  
在 22 个服装 prompt 上，Tailor 达到 **FashionCLIP 33.26**、**IQ 0.759**。  
而且它不仅优于 DressCode / GarmentDreamer / Garment3DGen / ChatGarment，也优于把其几何固定后替换成 Paint3D、SyncMVD、Hunyuan3D-2 的纹理模块版本。  
这说明提升不是单点模块偶然，而是**几何+纹理协同设计**的结果。

**3. 用户研究信号：人看也更偏好**  
25 名参与者的排序中：
- 人体生成的视觉质量和文本对齐排名第一；
- 服装生成的几何质量、纹理质量、文本对齐也都排名第一。

**4. 消融信号：关键模块确实是因果旋钮**  
- 去掉身体几何约束：衣服更容易穿模、漂位、比例失真；
- 去掉 cyclic merging：多视图纹理更不一致；
- 去掉 symmetric local attention：左右对称图案更容易破坏。

### 证据强度怎么判断

我会把这篇论文的证据强度定为 **moderate**，不是 strong。原因是：
- 有对比、消融、用户研究，信号并不弱；
- 但评测主要基于作者自建 prompt 集，而不是广泛接受的公共 benchmark；
- 系统还依赖商业/闭源组件，完整可复现性尚未被充分验证；
- “更适合生产”这件事有较强系统工程属性，难靠单一指标完全覆盖。

### 局限性

- **Fails when**: 需要相对模板进行大幅拓扑变化、结构增删的服装请求；多层嵌套穿搭；口袋等非流形或复杂结构细节。
- **Assumes**: 依赖 HumGen3D 商业人体生成器、GPT-4o、Stable Diffusion 3.5/SDXL、Blender 管线、预定义服装模板与UV；实验使用单张 A100，且代码仅为 promised release。
- **Not designed for**: 从零生成任意服装拓扑；复杂配饰/鞋帽的完整造型系统；多层服装交互式物理结构联合生成。

### 可复用组件

这篇论文最值得迁移的不是整套系统，而是以下操作符：

- **LLM-to-CG 控制翻译**：把自然语言拆成参数、模板选择和对称性标记。  
- **身体感知的模板变形**：对任何“穿在人体上的网格生成”都很有用。  
- **多视图 latent 循环聚合**：适合有UV空间、又怕接缝的 mesh texturing。  
- **对称局部注意**：适合服装、鞋、盔甲等强对称纹理对象。

## Local PDF reference

![[paperPDFs/Digital_Human_Garment_Design/arXiv_2025/2025_Tailor_An_Integrated_Text_Driven_CG_Ready_Human_and_Garment_Generation_System.pdf]]