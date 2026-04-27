---
title: 'StyleID: A Perception-Aware Dataset and Metric for Stylization-Agnostic Facial Identity Recognition'
type: paper
paper_level: B
venue: Unknown
year: 2026
paper_link: https://arxiv.org/abs/2604.21689
aliases:
- 风格无关人脸身份识别的感知基准与度量
- StyleID
method: StyleID
modalities:
- Image
---

# StyleID: A Perception-Aware Dataset and Metric for Stylization-Agnostic Facial Identity Recognition

[Paper](https://arxiv.org/abs/2604.21689)

**Topics**: [[T__Benchmark_-_Evaluation]], [[T__Cross-Modal_Matching]] (其他: Face Recognition) | **Method**: [[M__StyleID]]

| 中文题名 | 风格无关人脸身份识别的感知基准与度量 |
| 英文题名 | StyleID: A Perception-Aware Dataset and Metric for Stylization-Agnostic Facial Identity Recognition |
| 会议/期刊 | 2026 (arXiv预印本) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.21689) · [Code] · [Project] |
| 主要任务 | 风格化人脸身份验证基准构建、感知对齐的身份度量学习、跨风格域人脸识别 |
| 主要 baseline | ArcFace, AdaFace, CLIP, StylizedFace |

> [!abstract] 因为「现有人脸识别编码器在风格化人脸（卡通、素描、绘画等）上严重退化，且缺乏以人类感知为基准的评估协议」，作者在「ArcFace/AdaFace 等自然照片域编码器」基础上改了「构建 StyleBench-H/S 双基准数据集 + 提出感知对齐的 StyleID 度量损失（angular margin loss + 监督对比损失 + 感知一致性损失）」，在「StyleBench-H 人类判断基准」上取得「与人类感知高度一致的身份验证性能」。

- **关键性能 1**: StyleID 在 StyleBench-H 上的身份验证准确率显著优于 ArcFace/AdaFace 等照片域编码器（具体数值
- **关键性能 2**: StyleID 在 StyleBench-S 大规模合成基准上随风格化强度增加保持稳健，而 baseline 系统性退化（Figure 6）
- **关键性能 3**: 感知一致性损失使模型输出与人类判断的 ROC 曲线更接近（Figure 14）

## 背景与动机

当用户将自拍照转换为吉卜力风格动画头像，或艺术家为游戏角色生成系列化卡通形象时，一个核心需求是：转换后的图像是否仍能被识别为同一人？现有人脸身份识别系统在此场景下表现糟糕——它们往往在风格轻微变化时就产生剧烈的身份分数波动，或在几何夸张导致真实身份改变时毫无察觉。

现有方法如何处理这一问题？**ArcFace** 和 **AdaFace** 等主流编码器在自然照片域大规模训练，采用加性角边距损失优化判别性特征，但其特征空间完全锚定于照片域分布，对风格化造成的纹理偏移极度敏感。**CLIP** 等通用视觉-语言模型通过对比学习获得开放域表征，但其相似度度量未针对人类感知层面的身份判断进行校准，常将风格相似误判为身份相同。**StylizedFace** 虽专门探索了风格化人脸识别，但既未与人类判断建立定量关联，也未公开发布数据集或代码，无法作为可靠基准。

这些方法的根本局限在于：**缺乏以人类感知为金标准的评估协议**。照片域编码器的阈值无法迁移到风格化图像；通用相似度缺乏身份特异性；而 StylizedFace 的封闭性使其无法复现。这导致两个连锁后果：开发者无法比较不同风格化流水线的身份保留能力，研究者缺乏将身份度量锚定到人类判断的监督信号。更关键的是，现有模型在风格化强度增加时呈现系统性退化（Figure 6），且全局阈值无法跨风格族泛化——这说明问题本质是评估范式的缺失，而非仅靠微调可修复。

本文的核心回应是：构建人类校准的双层基准（StyleBench-H 人类判断 + StyleBench-S 大规模合成），并据此训练感知对齐的 StyleID 度量模型。
![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/4514443e-b758-4119-9e59-a1a19c38f63e/figures/Figure_2.png)
*Figure 2: Fig. 2. Example data generated for stylization from weaker to stronger stylization strengths. As the stylization strength increases, identity preservationdecreases. The first row shows results from IP*



## 核心创新

核心洞察：**人类对风格化人脸的身份判断依赖于与风格无关的结构性身份线索，而非纹理或色彩表面特征**，因为同一风格族内的身份变化（如不同人的卡通化结果）与同一身份跨风格的变化（如某人的照片与素描）在人类感知中遵循不同的相似度结构，从而使「感知锚定的度量学习」成为可能。

| 维度 | Baseline (ArcFace/AdaFace) | 本文 StyleID |
|:---|:---|:---|
| 训练数据分布 | 自然照片域单一分布 | 照片域 + 多风格域联合分布 |
| 损失函数设计 | 判别性角边距（仅类内/类间分离） | 角边距 + 监督对比 + 感知一致性（对齐人类判断） |
| 评估基准 | 照片域基准（LFW/MegaFace等） | StyleBench-H（人类判断）+ StyleBench-S（可控强度大规模合成） |
| 阈值机制 | 全局固定阈值 | 风格无关的感知校准阈值 |
| 泛化目标 | 照片域内跨姿态/光照 | 跨风格族、跨风格强度的身份一致性 |

## 整体框架


![Figure 7](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/4514443e-b758-4119-9e59-a1a19c38f63e/figures/Figure_7.png)
*Figure 7: Fig. 7. StyleID training overview. While the angular margin loss is computedwith respect to class centers and the supervised contrastive loss is computedwith positive-negative pairs rather than indivi*



StyleID 框架包含三个协同层次：**数据层**（StyleBench 双基准构建）、**模型层**（感知对齐的度量学习）、**评估层**（与人类判断的校准验证）。数据流向如下：

**输入**：自然照片人脸图像 + 目标风格描述/参考图 → **风格化生成模块**：通过可控强度风格迁移网络生成多强度风格化图像对（Figure 2 示例，从弱到强风格化）→ **StyleBench-H 构建**：对人类被试进行配对身份判断问卷（Figure 11），收集感知层面的「同身份/不同身份」标签 → **StyleBench-S 构建**：基于 StyleBench-H 的统计特性，大规模合成扩展数据集，覆盖更广风格族与强度参数 → **StyleID 编码器训练**：以 ResNet-50/100 或 ViT 为骨干，联合优化三类损失 → **输出**：风格无关的身份嵌入向量，支持跨风格验证与风格化强度鲁棒的阈值决策。

核心模块详解：
- **可控风格化生成**：输入照片 + 风格强度参数 α∈[0,1]，输出连续谱风格化图像，α=0 为原图，α=1 为最强风格化
- **人类感知采集**：基于 Figure 11 问卷设计，收集配对比较判断，建立人类 ROC 曲线作为金标准
- **StyleID 编码器**：三损失联合训练，输出 512-dim 身份嵌入
- **感知校准评估**：模型 ROC 与人类 ROC 直接对比（Figure 14）

```
照片输入 → [风格化生成] → {弱,中,强}风格化图像
                ↓
        [人类判断采集] → StyleBench-H (金标准)
                ↓
        [大规模合成扩展] → StyleBench-S (训练数据)
                ↓
        [StyleID编码器] → 嵌入空间
           (三损失训练)
                ↓
        [与人类ROC校准] → 感知对齐度量
```

## 核心模块与公式推导

### 模块 1: Angular Margin Loss（基础判别性约束，对应框架图 Figure 7 左支）

**直觉**: 保持 ArcFace 类内紧凑、类间分离的几何特性，但将优化目标从照片域扩展到风格化域。

**Baseline 公式** (ArcFace):
$$L_{arc} = -\frac{1}{N}\sum_{i=1}^{N}\log\frac{e^{s(\cos(\theta_{y_i}+m))}}{e^{s(\cos(\theta_{y_i}+m))} + \sum_{j\neq y_i}e^{s\cos\theta_j}}$$
符号: $s$ = 特征尺度因子, $m$ = 加性角边距, $\theta_{y_i}$ = 样本与真实类中心的夹角, $y_i$ = 真实类别标签

**变化点**: ArcFace 仅在自然照片上优化，当风格化导致特征偏移时类中心假设失效。本文将其作为基础约束保留，但联合训练域扩展。

**本文公式**:
$$L_{angular} = L_{arc}\big|_{\mathcal{X}_{photo} \cup \mathcal{X}_{stylized}}$$
即相同形式，但训练样本 $\mathcal{X}$ 包含照片域与多风格域的联合分布，使角边距约束泛化到风格化特征。

**对应消融**: 移除 angular margin 导致基线判别性丧失。

---

### 模块 2: Supervised Contrastive Loss（风格不变性约束，对应框架图 Figure 7 中支）

**直觉**: 强制同一身份在不同风格下的嵌入聚集，即使风格强度变化剧烈。

**Baseline 公式** (SupCon):
$$L_{supcon} = \sum_{i\in I}\frac{-1}{|P(i)|}\sum_{p\in P(i)}\log\frac{\exp(\mathbf{z}_i\cdot\mathbf{z}_p/\tau)}{\sum_{a\in A(i)}\exp(\mathbf{z}_i\cdot\mathbf{z}_a/\tau)}$$
符号: $P(i)$ = 与样本 $i$ 同身份的正样本集合（跨风格）, $A(i)$ = 所有其他样本, $\tau$ = 温度系数, $\mathbf{z}$ = L2归一化嵌入

**变化点**: 标准 SupCon 在照片域定义正负样本；本文将正样本扩展为「同一身份任意风格化变体」，负样本包含「不同身份即使同风格」。关键假设：风格变化不应改变身份邻近关系。

**本文公式（推导）**:
$$\text{Step 1}: \tilde{P}(i) = \{j: y_j = y_i, \text{style}(j) \neq \text{style}(i)\} \quad \text{加入跨风格同身份正样本}$$
$$\text{Step 2}: \tilde{A}(i) = \{j: y_j \neq y_i\} \cup \{j: y_j = y_i, \text{但人类判断为不同身份}\} \quad \text{纳入感知冲突样本为困难负例}$$
$$\text{最终}: L_{contrastive} = \sum_{i\in I}\frac{-1}{|\tilde{P}(i)|}\sum_{p\in\tilde{P}(i)}\log\frac{\exp(\mathbf{z}_i\cdot\mathbf{z}_p/\tau)}{\sum_{a\in\tilde{A}(i)}\exp(\mathbf{z}_i\cdot\mathbf{z}_a/\tau)}$$

**对应消融**: Table 。

---

### 模块 3: Perception Consistency Loss（人类判断对齐，对应框架图 Figure 7 右支）

**直觉**: 模型输出的相似度分布应统计匹配人类在配对判断中的置信度分布，这是 StyleID 区别于所有 baseline 的核心创新。

**Baseline**: 无直接对应——ArcFace/AdaFace 无人类感知监督，CLIP 无身份特异性人类校准。

**变化点**: 引入 StyleBench-H 收集的人类判断作为软标签，最小化模型相似度与人类一致率的分布差异。

**本文公式（推导）**:
$$\text{Step 1}: p_{human}(x_i, x_j) = \frac{\text{判为同身份的人数}}{\text{总判断人数}} \in [0,1] \quad \text{人类感知概率}$$
$$\text{Step 2}: p_{model}(x_i, x_j) = \sigma\left(\frac{\mathbf{z}_i\cdot\mathbf{z}_j - \mu}{\sigma_{scale}}\right) \quad \text{模型相似度概率化}$$
$$\text{Step 3}: L_{percept} = \mathbb{E}_{(i,j)\sim\mathcal{D}_{StyleBench-H}}\left[\text{KL}\left(p_{human} \| p_{model}\right)\right] + \lambda\mathbb{E}\left[(p_{human}-p_{model})^2\right]$$
$$\text{最终}: L_{total} = L_{angular} + \beta_1 L_{contrastive} + \beta_2 L_{percept}$$
符号: $\sigma$ = sigmoid, $\mu, \sigma_{scale}$ = 可学习或可估计的校准参数, $\beta_1, \beta_2$ = 损失权重

**对应消融**: Figure 14 显示加入 $L_{percept}$ 后模型 ROC 曲线显著逼近人类 ROC（线性/对数尺度），移除则偏离。

## 实验与分析

| Method | StyleBench-H (人类判断) | StyleBench-S (强风格化) | 跨风格泛化 |
|:---|:---|:---|:---|
| ArcFace | 
| AdaFace | 
| CLIP (image-image) | 
| StylizedFace | 不可复现 | 不可复现 | 未知 |
| **StyleID (本文)** | **


![Figure 6](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/4514443e-b758-4119-9e59-a1a19c38f63e/figures/Figure_6.png)
*Figure 6: Fig. 5. Recognition accuracy as a function of stylization strength on StyleBench-S. The x-axis denotes stylization strength and the y-axis denotes recognitionaccuracy. The first row shows the Pixar st*



**主结果分析**: Figure 6 展示了最关键的趋势证据——随风格化强度增加（x-axis），各方法验证准确率的变化。Baseline（ArcFace/AdaFace）呈现陡峭下降，说明其特征空间对风格纹理过度敏感；StyleID 曲线平缓，验证了其「风格无关」核心 claim。Figure 14 的 ROC 曲线从定量角度确认：StyleID 的线性/对数 ROC 与人类 ROC 几乎重合，而 ArcFace 的阈值在风格化域产生大量 false negative（将同身份误判为不同）。

**消融实验**: 三损失组件的相对贡献：
- 仅 $L_{angular}$：照片域性能保留，风格化域退化（与 ArcFace 类似）
- $+ L_{contrastive}$：跨风格聚集改善，但相似度绝对值未对齐人类感知
- $+ L_{percept}$（完整 StyleID）：ROC 曲线与人类金标准匹配



**公平性检查**: 
- **Baseline 强度**: 对比了当前最广泛使用的照片域编码器（ArcFace、AdaFace）和开放域替代方案（CLIP），但未与最新专门针对风格化的方法（如公开发表后的 StylizedFace 改进版）比较，存在时效性局限。
- **计算成本**: StyleBench-H 的人类采集成本高、规模受限（Figure 4 显示人口统计学分布），StyleBench-S 的合成扩展缓解了此问题但引入生成器偏差。
- **失败案例**: 极端几何夸张（如毕加索式立体主义）可能超出感知一致性损失的适用范围；Figure 2 显示最强风格化时身份线索已严重扭曲，此时人类判断本身也趋于随机。
- **数据偏见**: Figure 4 的人口统计学分布显示源图像在种族/年龄/性别上的覆盖情况，需关注是否均衡（具体比例。

## 方法谱系与知识库定位

**方法家族**: 深度度量学习 → 人脸识别 → 跨域/风格无关身份识别

**Parent method**: ArcFace（Deng et al., 2019）——提供加性角边距损失的框架基础。本文保留其几何判别结构，但在三个关键 slot 上改造：

| 变更 Slot | Parent (ArcFace) | 本文修改 |
|:---|:---|:---|
| 数据 curation | MS1M/VGGFace2 自然照片 | StyleBench-H/S 照片-风格化配对 |
| 训练 recipe | 单一 angular margin | 三损失联合（+ supervised contrastive + perception consistency） |
| 评估 protocol | 照片域 ROC/阈值 | 人类感知锚定的跨风格 ROC |

**直接 Baseline 差异**: 
- **vs ArcFace/AdaFace**: 从「照片域判别性」扩展到「跨风格感知一致性」，核心差异是引入人类判断作为监督信号
- **vs CLIP**: 从「通用视觉-语言相似度」聚焦到「身份特异性度量」，并通过 StyleBench 建立人类校准
- **vs StylizedFace**: 从「封闭专有方案」转变为「公开可复现基准 + 度量」，填补评估范式缺口

**后续方向**: 
1. **多模态扩展**: 将 StyleID 嵌入整合到文本-图像生成模型（如 Stable Diffusion）的 identity-preserving 微调中，实现生成即验证
2. **动态风格强度估计**: 当前需预设风格化强度参数，未来可联合估计风格强度与身份一致性，实现自适应阈值
3. **3D/视频风格化**: 从静态图像扩展到动态 avatar 和 3D 风格化，保持时序身份一致性

**知识库标签**: 
- modality: 图像（人脸）
- paradigm: 度量学习 + 人类感知对齐
- scenario: 风格化内容生成、虚拟形象、多模态 AI 安全
- mechanism: 三损失联合优化（判别性 + 风格不变性 + 感知一致性）
- constraint: 需人类判断采集（成本高）、风格化生成器依赖、极端几何变形边界

