---
title: "Dressing in the Wild by Watching Dance Videos"
venue: CVPR
year: 2022
tags:
  - Others
  - task/virtual-try-on
  - flow-fusion
  - self-supervised-learning
  - cycle-consistency
  - dataset/Dance50k
  - dataset/DeepFashion
  - repr/SMPL
  - opensource/partial
core_operator: 用SMPL顶点流优先对齐刚体身体区域、用2D像素流补足松散服装的非刚体形变，并以舞蹈视频跨帧自监督训练实现真实场景换装
primary_logic: |
  源人物图像+目标人物图像/姿态 → 条件分割预测目标服装布局，估计2D像素流与3D顶点流并按区域融合为wFlow，再结合背景修复与纹理补全生成器进行合成 → 输出保留目标身份与背景、穿上源服装的结果图
claims:
  - "在Dance50k上，wFlow的FID/LPIPS/IoU分别为8.809/0.090/0.719，优于LWG的13.080/0.107/0.484，并在人评上从0.271提升到0.729 [evidence: comparison]"
  - "在DeepFashion上，wFlow取得FID 57.652、LPIPS 0.187、IoU 0.687，优于LWG、ADGAN和DiOR，说明其收益不局限于自建数据集 [evidence: comparison]"
  - "消融实验表明，加入blended wFlow与循环在线优化后，Dance50k上的FID从仅像素流的12.077或无CO的12.106降到8.809 [evidence: ablation]"
related_work_position:
  extends: "Liquid Warping GAN (Liu et al. 2019)"
  competes_with: "ADGAN (Men et al. 2020); DiOR (Cui et al. 2021)"
  complementary_to: "N/A"
evidence_strength: strong
pdf_ref: paperPDFs/Digital_Human_Image_Based_Virtual_Try_On/CVPR_2022/2022_Dressing_in_the_Wild_by_Watching_Dance_Videos.pdf
category: Others
---

# Dressing in the Wild by Watching Dance Videos

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2203.15320) · [Project](https://awesome-wflow.github.io)
> - **Summary**: 论文把 2D 像素流与 3D SMPL 顶点流融合成 wFlow，并用舞蹈视频的跨帧自监督替代昂贵的成对试穿标注，从而把人到人的真实场景换装推进到复杂姿态、松散衣物和真实背景条件下。
> - **Key Performance**: Dance50k 上 FID 8.809、IoU 0.719；DeepFashion 上 FID 57.652、LPIPS 0.187。

> [!info] **Agent Summary**
> - **task_path**: 源人物图像（含待迁移服装）+ 目标人物图像/姿态 -> 保留目标身份与背景的换装图像
> - **bottleneck**: 单一2D流难处理大姿态变化，单一3D流难覆盖裙摆/宽松衣物等非刚体形变；同时成对试穿数据难以规模化采集
> - **mechanism_delta**: 用“顶点流负责刚体 articulation、像素流负责非刚体布料”的区域融合 wFlow 取代单一路径 warping，并加入舞蹈视频跨帧自监督与测试时循环细化
> - **evidence_signal**: 双数据集比较和组件消融均显示 full model 在 FID/LPIPS/IoU 上最优，且在线循环优化显著降低 FID
> - **reusable_ops**: [conditional-layout-prediction, rigid-nonrigid-flow-fusion]
> - **failure_modes**: [SMPL或DensePose估计失准导致错位, 低清晰度或前景不明显的查询图需要多轮在线优化]
> - **open_questions**: [能否摆脱外部人体解析与SMPL依赖, 由舞蹈视频学到的对应关系能否稳定迁移到更广泛日常场景]

## Part I：问题与挑战

这篇论文解决的是 **in-the-wild 的 person-to-person garment transfer**：输入源人物图像 \(含衣服\) 和目标人物图像，输出保持目标身份与背景、但穿上源服装的结果。

### 真正的问题是什么
真正的瓶颈不是“生成一件衣服”，而是 **在复杂姿态、真实背景和宽松服装下，可靠地找到纹理该从哪里搬到哪里**。  
已有路线各有硬伤：

- **2D pixel flow**：对裙摆、连衣裙、宽松外套这类非刚体服装更自由，但遇到大姿态变化、遮挡、交叉手臂时容易错位。
- **3D vertex flow / SMPL**：对骨架运动更稳，但 SMPL 表面并不能覆盖宽松衣物的外轮廓，所以对 loose garments 容易失真。
- **paired try-on 数据**：昂贵且难规模化，限制了模型真正进入“野外场景”。

### 为什么现在值得做
作者抓住了一个现实机会：**互联网上容易获得大量单人舞蹈视频**。  
这类视频天然提供“同一人、同一套衣服、不同姿态”的跨帧监督信号，刚好能替代难收集的成对换装标注。

### 输入/输出与边界
- **输入**：源人物图像、目标人物图像；训练时使用同一人的不同帧，测试时允许不同身份。
- **输出**：保留目标身份与背景的换装图像。
- **边界条件**：单人图像、服装需在源图中可见；依赖人体解析、关键点、DensePose/SMPL 估计质量。

## Part II：方法与洞察

方法是一个多阶段系统，核心不是“更强的生成器”，而是 **把服装纹理映射拆成更适合的几种对应关系**。

### 1) 条件分割：先预测“目标姿态下的衣服布局”
作者先用条件分割网络，根据：
- 源图的服装/人体信息
- 目标图的 DensePose 与骨架

预测目标姿态下的人体分割与服装区域。  
作用是先把“衣服大概应出现在哪里”固定下来，缩小后续流场搜索空间，也缓解训练时“同一身份”、测试时“不同身份”的分布落差。

### 2) Pixel Flow：学习非刚体纹理搬运
再用像素流网络估计源图到目标图的 2D 对应关系。  
这里作者没有直接让网络盲目回归流，而是加入 **feature correlation**，显式做跨图匹配，提升大形变下的稳定性。

### 3) wFlow：把 3D 顶点流和 2D 像素流融合
这是本文最关键的一步。

- **SMPL vertex flow**：负责人体刚体部位和大姿态变化
- **pixel flow**：负责衣物边缘、裙摆、宽松服装等非刚体区域

融合方式很直接：**在 SMPL 可覆盖的区域优先使用 vertex flow，其余区域回退到 pixel flow**。  
这等于把“姿态 articulation”和“布料非刚体形变”分开处理。

### 4) 生成与细化：背景修补 + 纹理补全 + 测试时循环优化
后端生成网络分三支：
- 背景修补
- 源图重建
- 目标换装生成

其中源图重建分支承担一部分 **纹理补全** 角色，再把这些能力蒸馏给目标换装分支。  
此外，作者还加了 **cycle online optimization**：测试时对同一对输入反复正反换装，让模型对当前样本局部过拟合，从而提升低质量 query 的边缘与纹理。

### 核心直觉

作者真正改变的是两件事：

1. **把“单一流场解释全部运动”改成“刚体/非刚体分治”**  
   - 改变前：一个流场同时承担骨架运动和布料形变，约束冲突大  
   - 改变后：3D 流先保证姿态合理，2D 流补足衣物自由变形  
   - 结果：复杂 pose 下不容易错骨架，宽松服装也不容易塌形

2. **把“昂贵配对监督”改成“跨帧自监督”**  
   - 改变前：训练数据难收集，覆盖衣服种类有限  
   - 改变后：用舞蹈视频的同人多姿态帧学习纹理搬运  
   - 结果：更容易扩展到真实场景服装与背景

### 战略权衡

| 设计选择 | 改变的约束/信息瓶颈 | 收益 | 代价/风险 |
|---|---|---|---|
| 条件分割先行 | 先固定目标衣物布局，减少后续对应搜索难度 | 跨身份测试更稳，轮廓更清晰 | 依赖解析质量 |
| wFlow 融合 2D/3D 流 | 将 articulation 与 cloth deformation 分治 | 复杂姿态与宽松衣物兼顾 | SMPL 拟合错会直接传递错误 |
| 舞蹈视频跨帧自监督 | 不再依赖 paired try-on 标注 | 数据规模可扩张，服装更丰富 | 训练域偏舞蹈视频 |
| 在线 cycle 优化 | 测试时对单个样本继续适配 | 低质 query 可明显细化 | 推理慢，作者设 k=20 |

## Part III：证据与局限

### 关键证据信号

- **比较信号：真实场景指标明显领先**  
  在 Dance50k 上，wFlow 的 FID/LPIPS/IoU 为 **8.809 / 0.090 / 0.719**，明显优于 LWG 的 **13.080 / 0.107 / 0.484**。  
  这说明它不只是“看起来更像”，而是在 **视觉真实感、感知距离、衣物形状对齐** 三个维度同时提升。

- **跨数据集信号：不是只对自建数据有效**  
  在 DeepFashion 上也保持领先：**FID 57.652、LPIPS 0.187、IoU 0.687**。  
  虽然提升幅度没有 Dance50k 那么大，但证明其收益不完全依赖舞蹈域。

- **因果信号：融合流和在线优化都不可少**  
  消融里：
  - 只用 pixel flow：Dance50k FID **12.077**
  - 只用 vertex flow：Dance50k FID **9.455**
  - 去掉在线 CO：Dance50k FID **12.106**
  - full model：Dance50k FID **8.809**

  这个结果很关键：**2D/3D 融合不是装饰，测试时循环优化也不是小修小补，而是实打实改变输出质量的因果旋钮。**

### 1-2 个最该记住的指标
- **Dance50k FID 8.809**：相对主基线 LWG 的 13.080 有明显下降。
- **Dance50k IoU 0.719**：说明对宽松服装外形的保持更好，这正是本文主打能力。

### 局限性
- **Fails when**: SMPL、DensePose、人体解析或关键点估计失败时，wFlow 会把错误对应放大；对极低清晰度、前景模糊或严重遮挡的 query，若不做在线优化容易出现纹理糊化。
- **Assumes**: 依赖多个人体先验模块（OpenPose、人体解析、SMPL fitting）；训练依赖大规模单人舞蹈视频；测试时若启用循环优化，推理成本显著增加。
- **Not designed for**: 多人场景、从商品平铺图直接试穿、严格物理级布料模拟、交互式局部编辑。

### 可复用组件
- **条件布局预测**：先预测“源衣服在目标姿态下的分割布局”，适合所有需要先定结构再搬纹理的任务。
- **刚体/非刚体流融合**：把 3D 先验用于骨架区域、把 2D 流用于布料区域，这个思路对 pose transfer、人像重渲染都可迁移。
- **测试时循环细化**：对困难样本做 query-specific refinement，是一种通用的 test-time enhancement 模式。

## Local PDF reference

![[paperPDFs/Digital_Human_Image_Based_Virtual_Try_On/CVPR_2022/2022_Dressing_in_the_Wild_by_Watching_Dance_Videos.pdf]]