---
title: "Dressing in Order: Recurrent Person Image Generation for Pose Transfer, Virtual Try-on and Outfit Editing"
venue: CVPR
year: 2021
tags:
  - Others
  - task/virtual-try-on
  - task/pose-transfer
  - recurrent-generation
  - flow-warping
  - soft-mask-compositing
  - dataset/DeepFashion
  - opensource/no
core_operator: "先生成人体底稿，再将每件服装以2D形状掩码与纹理图按顺序递归穿到隐藏表征上，显式控制遮挡、层叠与服装交互"
primary_logic: |
  目标姿态 + 源身体/服装图像 + 穿衣顺序
  → 用流场将身体与每件服装对齐到目标姿态，并分别编码为2D纹理特征和软形状掩码
  → 先生成身体，再按顺序在掩码区域内递归更新隐藏状态
  → 输出可用于pose transfer、virtual try-on与outfit editing的可控人物图像
claims:
  - "在 DeepFashion 的 pose transfer 上，DiOr-large 在 256×176 分辨率下相对 ADGAN 将 FID 从 18.63 降到 13.59，并将 sIoU 从 56.54 提升到 59.99 [evidence: comparison]"
  - "去掉递归穿衣机制会使 sIoU 从 58.99 降到 58.44，并在服装重叠区域产生明显 ghosting，说明顺序化合成有助于建模服装交互 [evidence: ablation]"
  - "在 virtual try-on 用户研究中，DiOr 相对 ADGAN 的偏好率为 80.64% vs. 19.36%，显示其在服装形状与纹理保持上更符合人类感知 [evidence: comparison]"
related_work_position:
  extends: "ADGAN (Men et al. 2020)"
  competes_with: "ADGAN (Men et al. 2020); GFLA (Ren et al. 2020)"
  complementary_to: "N/A"
evidence_strength: moderate
pdf_ref: "paperPDFs/Digital_Human_Image_Based_Virtual_Try_On/CVPR_2021/2021_Dressing_in_Order_Recurrent_Person_Image_Generation_for_Pose_Transfer_Virtual_Try_on_and_Outfit_Editing.pdf"
category: Others
---

# Dressing in Order: Recurrent Person Image Generation for Pose Transfer, Virtual Try-on and Outfit Editing

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2104.07021), [Project](https://cuiaiyu.github.io/dressing-in-order)
> - **Summary**: DiOr 把人物生成改写成“先生成身体、再按顺序穿衣”的过程，用服装顺序和2D形状/纹理显式控制遮挡关系，从而统一支持姿态迁移、虚拟试衣和服装编辑。
> - **Key Performance**: 在 DeepFashion pose transfer 上，DiOr-large 相比 ADGAN 达到 **FID 13.59 vs. 18.63**、**sIoU 59.99 vs. 56.54**；在 virtual try-on 用户研究中获得 **80.64%** 偏好率。

> [!info] **Agent Summary**
> - **task_path**: 人像/服装图像 + 目标姿态 + 穿衣顺序 -> 可控人物生成图像
> - **bottleneck**: 现有方法通常一次性生成整套穿搭，并依赖互斥服装布局或1D风格码，导致服装交互、同类叠穿和细粒度纹理保持都难以建模
> - **mechanism_delta**: 将每件服装表示成 2D 纹理图 + 软形状掩码，并按用户指定顺序递归写入隐藏人体状态
> - **evidence_signal**: DeepFashion 上对 ADGAN 的指标全面领先，且 virtual try-on 用户偏好 80.64%
> - **reusable_ops**: [2D服装形状-纹理解耦编码, 顺序条件递归遮罩合成]
> - **failure_modes**: [复杂或罕见姿态下几何失真, 异常服装形状或重叠区域出现 ghosting 与补洞失败]
> - **open_questions**: [如何对无直接监督的编辑任务做标准化评测, 如何提升高分辨率下的遮挡真实感与细节保真]

## Part I：问题与挑战

这篇论文要解决的不是单一的 try-on 或 pose transfer，而是一个更底层的问题：**如何用统一的2D框架生成“可编辑、可重排、可叠穿”的人物外观**。

### 真正瓶颈是什么？
以往方法大多有两个硬约束：

1. **整套服装一次性生成**  
   模型需要同时决定每件衣服的形状、纹理和前后遮挡关系。这样一来，像“上衣塞进裤子里/放出来”“同类上衣叠穿”这类效果，就只能被模型隐式猜，不能由用户控制。

2. **服装表示过于压缩**  
   典型做法要么先生成互斥分割图，要么把单件服装压成 1D style code。前者限制了重叠与层次，后者丢失了空间纹理结构，所以花纹、边界、袖口等细节很难保住。

### 输入/输出接口
DiOr 把一个人表示成：

- **pose**
- **body**
- **{garments}**

其中每个元素都可以来自不同图像源。输出则是目标姿态下的合成人像。  
这意味着它天然支持：

- pose transfer
- virtual try-on
- outfit editing

### 为什么现在能做？
因为已有几个关键前提成熟了：

- OpenPose 提供稳定的 2D 姿态表示
- human parser 可以抽取衣服/皮肤/背景区域
- GFLA 的全局流场模块可用于把局部外观对齐到目标姿态
- DeepFashion 提供同一人不同姿态数据，能用 pose transfer 做监督训练

### 边界条件
这仍然是一个**纯2D图像生成**方法：

- 不显式建模 3D 人体
- 强依赖解析和姿态估计质量
- 编辑任务大多没有直接监督，更多靠训练出的组合泛化能力

## Part II：方法与洞察

DiOr 的核心设计可以概括为三步：

1. **把每件服装拆成 2D 形状 + 2D 纹理**
2. **先生成身体底稿**
3. **再按顺序一件一件“穿上去”**

### 方法主线

#### 1) 服装表示：2D 纹理图 + 软形状掩码
每件服装先通过 human parsing 得到分割区域，再借助流场估计对齐到目标姿态。之后编码成：

- **纹理特征图**：保留局部空间花纹
- **软形状掩码**：控制衣服覆盖范围，甚至可表达一定透明度

这一步解决的是 ADGAN 式 1D 编码的空间信息丢失问题。

#### 2) 身体先行：先做“底稿”
模型先根据目标 pose 和 body texture 生成人体基础隐藏表征。  
这样后续衣服不是从零生成，而是在一个已有的身体语义底板上叠加。

#### 3) 递归穿衣：顺序就是控制变量
随后，模型按用户给定顺序，把每件服装逐步写入隐藏状态：

- 掩码内：由当前服装更新
- 掩码外：保留上一状态

因此，同一组衣服只要顺序变了，最终视觉效果就会变：

- top 先于 bottom → 更像 tuck-in
- top 后于 bottom → 更像放在外面
- jacket over shirt over t-shirt → 支持同类/多层叠穿

#### 4) 训练策略：pose transfer + inpainting
作者发现只做 pose transfer 时：

- try-on 结果容易不一致
- 遮挡造成的缺失区域补不出来

于是加入 **inpainting** 作为辅助任务，让模型学会：

- 保细节
- 补孔洞
- 处理头发遮挡、局部缺失等问题

### 核心直觉

过去的方法把“穿搭结果”当成一个一次性输出，导致模型必须在单步里同时解决：

- 服装摆放位置
- 服装之间谁遮谁
- 纹理如何保留
- 缺失区域如何补全

DiOr 的关键改动是把这个问题**过程化**：

- **what changed**：从“一次性生成整套服装”改成“先身体、再逐件穿衣”的递归合成
- **which bottleneck changed**：  
  - 遮挡关系不再是模型隐式猜测，而是由**顺序**显式控制  
  - 服装外观不再被压缩成全局 1D 向量，而是保留为**2D 空间特征**
- **what capability changed**：模型从“只能生成一个默认 look”跃迁到“可控制塞衣角、叠穿、删图案、换纹理、换形状”

换句话说，这篇论文真正引入的不是一个更强的 GAN 块，而是一个**新的生成因果顺序**：  
**服装交互 = 穿衣顺序 × 局部空间表示**。

### 策略权衡

| 设计选择 | 解决的瓶颈 | 带来的能力 | 代价/风险 |
|---|---|---|---|
| 2D 形状/纹理解耦 | 1D 服装编码丢空间细节 | 更好保留花纹与边界，支持形状/纹理分开编辑 | 更依赖分割和对齐质量 |
| 递归穿衣 | 单步生成难处理遮挡关系 | 显式控制 tuck-in、叠穿、同类 layering | 顺序错误会累积，推理链更长 |
| 流场对齐 | 源姿态和目标姿态不一致 | 服装定位更准，几何更自然 | 流场错误会直接破坏服装位置 |
| pose transfer + inpainting 联训 | 只做 pose transfer 时补洞差 | 更能处理头发遮挡、缺失区域 | 大面积未见区域仍难 hallucinate |

## Part III：证据与局限

### 关键证据信号

#### 1) 对 ADGAN：不仅更灵活，也更准
在 DeepFashion 的 pose transfer 上，DiOr-large（256×176）相对 ADGAN：

- **SSIM**: 0.806 vs. 0.772
- **FID**: 13.59 vs. 18.63
- **LPIPS**: 0.176 vs. 0.226
- **sIoU**: 59.99 vs. 56.54

这里最有说服力的是：它不是只在一个指标上好，而是**感知质量、结构一致性和形状一致性都更强**。尤其 sIoU 提升支持了作者的核心论点：2D 形状/纹理表示更利于保住服装结构。

#### 2) 对 GFLA：姿态迁移能力接近，但表达更通用
在 256×256 评测中，DiOr-large 与 GFLA 整体可比，且 **sIoU 58.63 vs. 57.32** 更高。  
这说明 DiOr 即便没有 GFLA 的局部注意力，只借用其全局流场，也能保住接近的 pose transfer 能力，同时额外支持 try-on 和编辑。

#### 3) 消融直接验证“顺序化递归”是有效因果旋钮
最关键的消融有三组：

- **去掉递归机制**：sIoU 从 58.99 降到 58.44，重叠区域出现 ghosting  
  → 说明顺序化合成确实在解决服装交互，不是装饰性设计。
- **2D 改 1D 编码**：FID/LPIPS/sIoU 都明显变差  
  → 说明空间纹理不能被 1D style code 轻易替代。
- **去掉流场**：FID 从 14.34 恶化到 16.47，sIoU 从 58.99 降到 56.28  
  → 说明几何对齐是高质量 try-on 的必要前提。

#### 4) 用户研究：人类明显更偏好其 try-on 结果
- pose transfer 相对 ADGAN：**57.48%**
- pose transfer 相对 GFLA：**52.27%**
- virtual try-on 相对 ADGAN：**80.64%**

尤其 try-on 偏好率差距很大，说明“递归穿衣 + 2D服装表示”的收益在人类主观观感上比在自动指标上更明显。

### 局限性

- **Fails when**: 复杂或罕见姿态、异常服装形状、多层强遮挡情况下，人体几何和服装边界仍可能出错；重叠区域会有 ghosting；大面积不可见区域的补全仍不稳定。
- **Assumes**: 依赖较可靠的 human parsing、OpenPose 和全局流场估计；训练依赖 DeepFashion 这类同一人物多姿态监督数据；编辑能力主要来自 pose transfer + inpainting 的间接泛化而非专门标注。
- **Not designed for**: 显式3D一致性建模、真实物理布料模拟、任意高分辨率真实感渲染，也不是面向纯商品图到任意人像的鲁棒通用 try-on 系统。

### 复现与资源依赖
- 训练分辨率主要为 **256×176 / 256×256**
- 使用 **1–2 张 TITAN Xp**
- 依赖外部组件：human parser、OpenPose、GFLA 的全局流场模块
- 论文提供 project page，但正文未明确给出代码链接；因此复现仍有工程门槛

### 可复用组件
- **2D 服装形状/纹理解耦表示**
- **顺序条件的递归遮罩合成器**
- **基于流场的服装局部对齐**
- **pose transfer + inpainting 联合训练范式**
- **sIoU 这种“由人像解析驱动的结构一致性评估”思路**

### So what？
这篇论文的真正价值，不只是把指标比 ADGAN 做高，而是把**服装交互关系**从“模型内部隐变量”变成了**用户可操作的生成变量**。  
这使得人物图像生成第一次较自然地支持：

- tuck-in / tuck-out
- 同类服装 layering
- 纹理替换
- 图案删除/插入
- 形状编辑

也因此，它比传统 pose transfer / try-on 方法更接近“可编辑的人物外观图形系统”。

![[paperPDFs/Digital_Human_Image_Based_Virtual_Try_On/CVPR_2021/2021_Dressing_in_Order_Recurrent_Person_Image_Generation_for_Pose_Transfer_Virtual_Try_on_and_Outfit_Editing.pdf]]