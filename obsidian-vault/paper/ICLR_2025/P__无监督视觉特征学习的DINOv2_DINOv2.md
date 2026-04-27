---
title: 'DINOv2: Learning Robust Visual Features without Supervision'
type: paper
paper_level: C
venue: ICLR
year: 2025
paper_link: null
aliases:
- 无监督视觉特征学习的DINOv2规模化范式
- DINOv2
acceptance: Poster
cited_by: 1021
code_url: https://github.com/facebookresearch/dinov2
method: DINOv2
followups:
- 无配对数据的盲视觉-语言匹配_IBMTVL
- 无平行数据的视觉-语言盲匹配_Factorized_Hahn-
- 任意分辨率特征上采样的注意力插件_JAFAR
- 开放世界3D对象性学习的无提示检_OP3Det
---

# DINOv2: Learning Robust Visual Features without Supervision

[Code](https://github.com/facebookresearch/dinov2)

**Topics**: [[T__Self-Supervised_Learning]], [[T__Classification]] | **Method**: [[M__DINOv2]] | **Datasets**: [[D__ImageNet-1K]] (其他: iNaturalist 2018, iNaturalist 2021)

| 中文题名 | 无监督视觉特征学习的DINOv2规模化范式 |
| 英文题名 | DINOv2: Learning Robust Visual Features without Supervision |
| 会议/期刊 | ICLR 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2304.07193) · [Code](https://github.com/facebookresearch/dinov2) · [Project](https://github.com/facebookresearch/dinov2) |
| 主要任务 | 自监督视觉表征学习（Self-Supervised Visual Representation Learning） |
| 主要 baseline | DINO, iBOT, MAE, OpenCLIP, SEERv2 |

> [!abstract] 因为「自监督视觉特征在规模化时面临训练不稳定与数据质量瓶颈」，作者在「DINO/iBOT」基础上改了「增加KoLeo正则化、构建LVD-142M精选数据集、引入Sinkhorn-Knopp中心化与SwiGLU等训练稳定技术」，在「ImageNet-1k线性评估」上取得「86.5% Top-1准确率，超越OpenCLIP ViT-G/14的86.2%」

- **ImageNet-1k线性评估**: 86.5% Top-1，较iBOT ViT-L/16提升 +4.2%，较OpenCLIP ViT-G/14提升 +0.3%
- **iNaturalist 2021**: 82.0% Top-1，较OpenCLIP ViT-G/14提升 +9.7%
- **ADE20K语义分割（frozen特征）**: 47.2 mIoU，达到frozen特征SOTA

## 背景与动机

自监督学习（SSL）旨在无需人工标注即可从海量图像中学习通用视觉特征。一个典型场景是：给定一张猫的图片，模型需要学会将其与其他物体区分，但没有任何"猫"的标签。现有方法主要通过不同代理任务来实现这一目标。

**DINO** 采用学生-教师自蒸馏框架：学生网络学习匹配教师网络（EMA更新）在全局视图上的输出分布，无需标签即可涌现出类语义分割的特性。然而，DINO主要依赖ImageNet-1k/22k数据，规模受限。**iBOT** 在此基础上引入掩码图像建模（MIM），要求学生预测教师网络在掩码patch位置的token分布，增强局部特征学习。**MAE** 则采用非对称编码器-解码器架构，通过高比例掩码重建像素，但在下游分类任务上表现弱于基于蒸馏的方法。

这些方法的根本瓶颈在于：**规模化时的训练不稳定与特征坍塌**。当模型扩大至ViT-L/16甚至ViT-g/14级别、数据扩展至数亿规模时，DINO/iBOT的训练会出现发散，且特征空间趋向坍缩——不同图像的表征聚集在一起，丧失判别性。此外，现有SSL方法多局限于ImageNet域数据，缺乏多样化、高质量的大规模预训练数据源。弱监督方法如CLIP虽利用文本-图像对获得强性能，但其依赖昂贵的标注数据且对细粒度任务迁移不足。

本文的核心问题是：能否在不使用任何标注数据的情况下，通过纯自监督学习获得与CLIP级别相当、甚至更优的通用视觉特征？DINOv2的回答是肯定的——通过系统性的数据策划、损失函数改进与训练稳定技术，实现SSL特征的规模化突破。
![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/bf721242-385a-4e95-8403-615fec5c5816/figures/fig_001.png)
*Figure: Visualization of the first PCA components. We compute a PCA between the patches of the*



## 核心创新

核心洞察：特征均匀性是规模化自监督学习的关键瓶颈，因为Kozachenko-Leonenko熵估计器可以直接最大化表示空间的微分熵，从而使十亿参数级ViT在无标注数据上稳定训练并达到CLIP级别性能成为可能。

| 维度 | Baseline (DINO/iBOT) | 本文 (DINOv2) |
|:---|:---|:---|
| 数据管道 | ImageNet-1k/22k 或原始网页数据 | LVD-142M精选数据集，多源去重过滤 |
| 损失函数 | DINO自蒸馏 + iBOT MIM | 上述两项 + **KoLeo正则化**（新增） |
| 训练稳定 | 简单移动平均中心化 | **Sinkhorn-Knopp中心化**、LayerScale、随机深度 |
| 网络架构 | 标准ViT + GELU MLP | ViT + **SwiGLU**（LLaMA风格门控激活） |
| 特征分布 | 易坍塌，均匀性差 | 显式优化均匀性，防止聚集 |

## 整体框架


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/bf721242-385a-4e95-8403-615fec5c5816/figures/fig_002.png)
*Figure: Evolution of performance when scaling in parameters. We show performance on eight*



DINOv2延续学生-教师双网络架构，但在数据输入、损失计算与训练稳定三个层面进行系统性增强。整体数据流如下：

**输入层**：图像经过增强管道生成多个视图——两个全局视图（global views）供DINO损失使用，多个局部视图（local views）供iBOT掩码预测使用。所有数据来源于LVD-142M精选数据集。

**学生编码器（Student Encoder）**：ViT架构，处理所有全局与局部视图，输出[CLS]全局token与patch token。参数通过梯度更新。

**教师编码器（Teacher Encoder, EMA）**：与学生同架构，仅处理全局视图，参数为学生参数的指数移动平均（EMA）。输出经Sinkhorn-Knopp中心化处理后作为蒸馏目标。

**DINO Head**：将学生与教师的[CLS] token投影至共享维度，计算交叉熵蒸馏损失。

**iBOT Head**：对学生处理的局部视图中的掩码patch进行预测，匹配教师对应位置的token分布。

**KoLeo Regularizer**（新增）：接收学生网络输出的特征表示，计算批次内最近邻距离，最大化微分熵以鼓励均匀分布。

```
图像 → [数据增强] → 全局视图x₁ᵍ, x₂ᵍ + 局部视图xˡᵒᶜᵃˡ
    → 学生Encoder: 处理全部视图 → [CLS]ₛ, patchₛ
    → 教师Encoder(EMA): 仅处理全局视图 → [CLS]ₜ, patchₜ
        → Sinkhorn-Knopp Centering → 稳定化目标
    → DINO Head: CE([CLS]ₛ, [CLS]ₜ) → L_DINO
    → iBOT Head: CE(patchₛ[mask], patchₜ[mask]) → L_iBOT
    → KoLeo: -Σ log(min||zᵢ-zⱼ||) → L_KoLeo
    → L_total = L_DINO + λᵢBOT·L_iBOT + λ_KoLeo·L_KoLeo
    → 反向传播更新学生 → EMA更新教师
```

## 核心模块与公式推导

### 模块 1: DINO自蒸馏损失（框架图左侧全局分支）

**直觉**: 学生网络通过匹配教师网络在全局视图上的输出分布，无需标签即可学习语义一致的表征。

**Baseline 公式** (DINO):
$$\mathcal{L}_{DINO} = -\sum_{x \in \{x_1^g, x_2^g\}} \sum_{i=1}^{K} P_t^i(x) \log P_s^i(x)$$

符号: $P_s, P_t$ = 学生/教师输出的概率分布（经温度缩放的softmax），$x_1^g, x_2^g$ = 两个全局视图，$K$ = prototype维度。

**变化点**: DINOv2完全保留此损失，但教师输出经过改进的Sinkhorn-Knopp中心化替代原始简单中心化。

**本文公式（推导）**:
$$\text{Step 1}: P_t(x) = \frac{\exp(g_t(x)/\tau_t)}{\sum_{k=1}^{K} \exp(g_t^k(x)/\tau_t)} \quad \text{温度缩放使分布更尖锐}$$
$$\text{Step 2}: g_t(x) \leftarrow g_t(x) - c, \quad c = \lambda c + (1-\lambda) \frac{1}{B} \sum_{i=1}^{B} g_t(x_i) \quad \text{运行平均中心化}$$
$$\text{最终}: \mathcal{L}_{DINO} = -\sum_{x} \sum_{i} P_t^i(x) \log P_s^i(x) \quad \text{（形式不变，但c的计算更稳定）}$$

**对应消融**: Table 1显示训练改进（含Sinkhorn-Knopp）对大规模模型稳定性至关重要，缺失则训练发散。

---

### 模块 2: iBOT掩码图像建模损失（框架图右侧局部分支）

**直觉**: 对局部视图中的掩码patch进行预测，迫使模型学习细粒度的局部特征表示，补充全局[CLS]token的语义信息。

**Baseline 公式** (iBOT):
$$\mathcal{L}_{iBOT} = -\sum_{x \in \{x^{local}\}} \sum_{i \in M(x)} \sum_{k=1}^{K} P_t^k(x_i) \log P_s^k(x_i)$$

符号: $M(x)$ = 被掩码的patch索引集合，$x^{local}$ = 局部视图，$x_i$ = 第i个patch。

**变化点**: DINOv2保留iBOT损失但调整其与DINO损失的平衡，并配合LVD-142M数据的多样性提升局部预测质量。

**本文公式（推导）**:
$$\text{Step 1}: \text{局部视图} \ x^{local} \ \text{随机掩码} \ M(x) \subseteq \{1,...,N_{patches}\}$$
$$\text{Step 2}: \text{学生编码器处理掩码视图，仅预测掩码位置} \rightarrow P_s(x_i), i \in M(x)$$
$$\text{最终}: \mathcal{L}_{iBOT} = -\sum_{x^{local}} \sum_{i \in M(x)} P_t(x_i)^\text{top} \log P_s(x_i)$$

**对应消融**: Table 3显示移除iBOT MIM损失导致ImageNet-1k准确率下降 -1.2%，在密集预测任务上下降更显著。

---

### 模块 3: KoLeo正则化与总损失（框架图底部，核心创新）

**直觉**: 自监督学习易出现特征坍塌——所有图像映射到相似表征。通过最大化表示空间的微分熵，显式鼓励特征均匀分布。

**Baseline 公式** (DINO+iBOT):
$$\mathcal{L}_{DINO+iBOT} = \mathcal{L}_{DINO} + \lambda_{iBOT} \mathcal{L}_{iBOT}$$

**变化点**: 原始组合缺乏对特征空间几何结构的显式约束，大模型时均匀性恶化。本文引入Kozachenko-Leonenko熵估计器作为正则项。

**本文公式（推导）**:
$$\text{Step 1}: \text{对批次特征} \ Z = \{z_1, ..., z_N\}, \ z_i = f_\theta(x_i) \in \mathbb{R}^d$$
$$\text{Step 2}: \mathcal{L}_{KoLeo} = -\frac{1}{N} \sum_{i=1}^{N} \log(\min_{j \neq i} \|z_i - z_j\|) \quad \text{最大化最近邻距离的对数期望}$$
$$\text{Step 3}: \text{重归一化以保证数值稳定——特征先经L2归一化再计算距离}$$
$$\text{最终}: \mathcal{L}_{total} = \mathcal{L}_{DINO} + \lambda_{iBOT} \mathcal{L}_{iBOT} + \lambda_{KoLeo} \mathcal{L}_{KoLeo}$$

符号: $z_i$ = L2归一化后的特征向量，$\min_{j \neq i} \|z_i - z_j\|$ = 到最近邻的欧氏距离，$\lambda_{KoLeo}$ ≈ 0.1（典型值）。

**对应消融**: Table 3显示移除KoLeo正则化导致ImageNet-1k准确率下降 -0.3%，特征可视化显示空间分布明显聚集。

## 实验与分析



本文在ImageNet-1k线性评估、分布偏移测试（ImageNet-V2/ReaL）、细粒度分类（iNaturalist）、密集预测（ADE20K语义分割、NYU深度估计）等多维度基准上进行系统评估。核心 headline 来自Table 4：DINOv2 ViT-g/14在ImageNet-1k线性评估达到86.5% Top-1准确率，超越此前SSL SOTA iBOT ViT-L/16的82.3%达+4.2%，同时以+0.3%微弱优势超过弱监督方法OpenCLIP ViT-G/14的86.2%。这一结果表明，纯自监督学习首次在同等规模下达到CLIP级别性能，且无需任何文本标注。

在分布鲁棒性方面，ImageNet-V2测试集上DINOv2达到78.8%，超越EVA-CLIP ViT-g/14的77.7%（+1.1%），显示其学习特征对分布偏移更具韧性。细粒度任务优势更为显著：iNaturalist 2018/2021上分别取得81.2%和82.0%，较OpenCLIP提升+8.6%和+9.7%，说明自监督特征对细粒度视觉模式更具判别力。密集预测任务中，ADE20K frozen特征语义分割达47.2 mIoU（DINO ViT-L/16为45.8），NYU深度估计δ1指标达0.936（DINO 0.891，SEERv2 0.902），验证特征的多任务迁移能力。



消融实验（Table 1/3）揭示各组件贡献：移除KoLeo正则化损失-0.3% ImageNet-1k准确率；移除iBOT MIM损失-1.2%且密集预测任务崩溃；以ImageNet替代LVD-142M数据平均下降-3.5%跨基准性能；移除Sinkhorn-Knopp等训练改进导致训练不稳定。数据策划（LVD-142M）是最大单一贡献源。

公平性审视：对比基线涵盖当时主要SSL（DINO/iBOT/MAE）与弱监督（OpenCLIP/EVA-CLIP）方法，且规模匹配。但存在三点局限：近期工作如Data2Vec 2.0、BEiT-3、EVA-02未纳入比较；弱监督基线使用不同数据模态（文本-图像对），非严格同等条件；ViT-g/14与ViT-L/16的架构差异可能放大部分比较差距。训练成本方面，DINOv2-g消耗22,016 A100-40GB GPU-hours（Table 14）。作者主动披露公平性问题：Table 12-13显示模型对高收入西方国家、特定肤色存在偏见，地理与人口统计公平性仍有显著缺陷。

## 方法谱系与知识库定位

**方法家族**: 自监督视觉表征学习 → 基于自蒸馏的SSL（DINO lineage）

**父方法**: DINO（Emerging Properties in Self-Supervised Vision Transformers, ICCV 2021）。DINOv2直接继承其核心学生-教师EMA架构与自蒸馏范式，但系统性扩展了五个关键slot：

| Slot | 父方法状态 | DINOv2修改 |
|:---|:---|:---|
| data_pipeline | ImageNet-1k/22k | LVD-142M多源精选数据集 |
| objective | DINO蒸馏 | +iBOT MIM + KoLeo正则化 |
| training_recipe | 简单中心化 | Sinkhorn-Knopp、LayerScale、随机深度 |
| architecture | 标准ViT-GELU | SwiGLU门控激活（借自LLaMA） |
| inference_strategy | 单分辨率 | 多分辨率评估支持 |

**直接基线与差异**：
- **iBOT**: DINOv2采用其MIM损失，但改用精选数据并新增KoLeo
- **MAE**: 同为SSL但架构范式不同（非对称编解码器 vs 对称蒸馏），DINOv2分类性能显著领先
- **OpenCLIP/EVA-CLIP**: 弱监督对比学习，DINOv2证明纯SSL可达同等水平
- **SEERv2**: 同为无监督但基于RegNet架构与不同数据策略，DINOv2在密集任务更优

**后续方向**：(1) 将KoLeo正则化扩展至多模态对比学习，缓解CLIP类模型的特征聚集；(2) 结合DINOv2特征与LLM构建统一视觉-语言模型；(3) 针对公平性偏见设计数据重采样或对抗去偏正则化。

**标签**: 图像模态 / 自监督预训练范式 / 通用视觉表征场景 / 自蒸馏+掩码预测机制 / 无标注约束

## 引用网络

### 后续工作（建立在本文之上）

- [[P__无配对数据的盲视觉-语言匹配_IBMTVL]]: Modern self-supervised visual encoder; likely used as feature extractor or backb
- [[P__无平行数据的视觉-语言盲匹配_Factorized_Hahn-]]: Major visual feature extractor; likely used as backbone or feature source in the
- [[P__任意分辨率特征上采样的注意力插件_JAFAR]]: DINOv2 is a core vision foundation model that JAFAR likely builds upon or direct
- [[P__开放世界3D对象性学习的无提示检_OP3Det]]: DINOv2 provides self-supervised visual features. Likely used as backbone or feat

