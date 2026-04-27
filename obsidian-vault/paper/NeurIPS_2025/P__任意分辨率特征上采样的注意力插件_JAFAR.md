---
title: 'JAFAR: Jack up Any Feature at Any Resolution'
type: paper
paper_level: C
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 任意分辨率特征上采样的注意力插件JAFAR
- JAFAR
- JAFAR is a lightweight
acceptance: Poster
cited_by: 14
code_url: https://github.com/PaulCouairon/JAFAR
method: JAFAR
modalities:
- Image
paradigm: supervised
baselines:
- 无监督视觉特征学习的DINOv2_DINOv2
---

# JAFAR: Jack up Any Feature at Any Resolution

[Code](https://github.com/PaulCouairon/JAFAR)

**Topics**: [[T__Semantic_Segmentation]], [[T__Depth_Estimation]] | **Method**: [[M__JAFAR]] | **Datasets**: Linear Probing Semantic, Grad-CAM Evaluation, Zero-Shot Open-Vocabulary, BeV Vehicle

> [!tip] 核心洞察
> JAFAR is a lightweight, task-agnostic feature upsampler that uses attention-based queries from low-level image features and SFT-modulated low-resolution keys to recover fine-grained spatial details at arbitrary resolutions, generalizing from low-resolution training to much higher output scales without high-resolution supervision.

| 中文题名 | 任意分辨率特征上采样的注意力插件JAFAR |
| 英文题名 | JAFAR: Jack up Any Feature at Any Resolution |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2506.11136) · [Code](https://github.com/PaulCouairon/JAFAR) · [DOI](https://doi.org/10.48550/arxiv.2506.11136) |
| 主要任务 | Semantic Segmentation, Depth Estimation, Feature Upsampling |
| 主要 baseline | Bilinear, CARAFE, DySample, SAPA, ReSFU, LiFT, FeatUp (JBU/Implicit) |

> [!abstract] 因为「foundation vision encoder 产生空间粗粒度特征（14×-16×下采样）制约下游像素级任务」，作者在「FeatUp」基础上改了「以低层图像特征为query、SFT调制encoder特征为key的交叉注意力上采样机制，低分辨率训练泛化到任意分辨率」，在「ADE20K/COCO-Stuff线性探测、Grad-CAM解释性、BEV分割」上取得「一致超越所有baseline的表现」。

- **任务无关**：JAFAR 同时满足 Task-Agnostic、Direct Upsampling、Lightweight 三项性质，为现有方法中唯一（Table 1 方法对比）
- **低分辨率训练泛化高分辨率**：32×32 训练可上采样至 448×448（14× 尺度扩展），无需高分辨率监督
- **即插即用**：适用于任何 foundation encoder，推理时无需针对特定 encoder 重新训练

## 背景与动机

现代视觉基础模型（如 DINOv2 ViT）通过 patch embedding 将输入图像激进地下采样 14× 至 16×，输出的特征图语义丰富但空间分辨率极低——例如 448×448 的图像仅得到 32×32 的特征 token。对于语义分割、深度估计、BEV 感知等需要像素级精度的下游任务，这种空间粗粒度特征成为明显瓶颈。

现有上采样方法各有局限：**Bilinear 插值**仅对低分辨率特征做空间插值，完全忽略原始图像中的高频结构信息，导致边界模糊；**CARAFE** 和 **DySample** 等内容感知方法虽能预测自适应核，但依赖任务特定训练，无法跨任务通用；**LiFT** 实现了任务无关，但需要为每个目标分辨率单独训练，不是直接上采样；**FeatUp** 提供模型无关的解决方案，但其 JBU 变体非直接上采样、Implicit 变体需迭代网络优化，推理开销大。这些方法的共性缺陷在于：要么无法利用高分辨率图像引导，要么牺牲即插即用的轻量性，要么需要高分辨率监督训练。

JAFAR 的核心动机正是填补这一空白：设计一个同时满足「任务无关」「直接上采样」「轻量推理」三项性质的统一模块，且仅需低分辨率训练即可泛化到任意输出分辨率。
![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/85f5c919-0cb6-4489-b223-d670b99ab659/figures/Figure_3.png)
*Figure 3 (qualitative): PCA Visualization. DINOV2 ViT-S14 tokens at 28 × 28 resolution from the ImageNet validation set are upsampled to 448 × 448. Baseline methods—bilinear training-free and token2token—fail to reconstruct the scene. CAR and JAFAR reconstruct the scene with high fidelity. CAR fails on heavily occluded images (first and last), while JAFAR consistently sharpens details across diverse image styles and objects.*



## 核心创新

核心洞察：低层图像特征（如 RGB 或浅层 encoder 特征）包含精确的空间结构信息，而深层 encoder 特征包含丰富的语义信息；通过 Spatial Feature Transform (SFT) 将二者在交叉注意力中动态耦合，可以在不依赖高分辨率监督的情况下，从低分辨率训练直接泛化到任意分辨率的上采样输出，因为 query 的空间引导使 attention kernel 具备尺度无关的位置感知能力。

| 维度 | Baseline (FeatUp 等) | 本文 JAFAR |
|:---|:---|:---|
| 上采样机制 | JBU 空间传播 / 隐式网络迭代优化 | 交叉注意力核生成，一次前向传播 |
| 高分辨率引导 | 无（JBU）或 需完整前向传播（Implicit） | 低层图像特征直接构造 query |
| 语义-空间对齐 | 无显式调制 | SFT 空间参数 γ, β 动态调制 key |
| 训练监督 | 任务相关或需高分辨率标签 | 仅低分辨率监督，泛化到任意高分辨率 |
| 推理成本 | Implicit 优化慢 / JBU 额外传播 | 轻量注意力模块，直接输出 |

## 整体框架


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/85f5c919-0cb6-4489-b223-d670b99ab659/figures/Figure_1.png)
*Figure 1 (pipeline): JAFAR upsamples features from any foundation vision encoder to any image resolution. For any shape and feature size, JAFAR directly predicts the full resolution features in one step. JAFAR can serve as a versatile drop-in module for a variety of downstream tasks, including semantic segmentation, depth estimation, open-vocabulary segmentation, GradCAM evaluation, and BEV vehicle segmentation—consistently outperforming prior work.*



JAFAR 作为即插即用模块，接收任意 foundation encoder 的低分辨率特征，输出任意目标分辨率的高分辨率特征。数据流如下：

1. **Query 构造器**（输入：原始图像或浅层 encoder 特征；输出：高分辨率 query tokens Q）：从低层图像特征提取空间信息丰富的高分辨率表示，作为交叉注意力的 query 源，提供精确的位置和结构引导。

2. **SFT 调制器**（输入：低分辨率 encoder 特征、query 导出的空间参数 γ, β；输出：调制后的 key tokens K）：将 encoder 的语义特征与 query 的空间参数进行 Spatial Feature Transform，实现语义-空间动态对齐。

3. **交叉注意力核生成器**（输入：高分辨率 Q、SFT 调制后的低分辨率 K；输出：上采样注意力权重 / 核）：计算 QK^T/√d 的注意力分布，生成内容自适应的上采样核，替代固定插值核。

4. **特征聚合器**（输入：注意力权重、encoder value 特征；输出：上采样后的高分辨率特征）：以注意力权重对 value 做加权组合，输出目标分辨率特征图。

```
Input Image ──→ [浅层特征提取] ──→ High-Res Query Q
                     │
                     ↓
Foundation Encoder ──→ Low-Res Features ──→ [SFT: γ,β from Q] ──→ Modulated Key K
                     │                                              │
                     └──────────────────────────────────────────────┘
                                                                    ↓
                                              [Cross-Attention] ← Q, K
                                                                    ↓
                                              Attention Weights ──→ [Weighted Aggregation]
                                                                    ↓
                                                              Upsampled Features
```

## 核心模块与公式推导

### 模块 1: SFT 调制器（对应框架图 SFT 模块）

**直觉**: 低分辨率 encoder key 缺乏空间细节，需用 query 的空间统计量进行逐位置调制，使语义特征具备空间可变性。

**Baseline 公式** (标准交叉注意力): 无调制，直接使用原始 key
$$K_{raw} = W_K \cdot F_{enc}$$
符号: $F_{enc}$ = 低分辨率 encoder 特征, $W_K$ = key 投影矩阵

**变化点**: 原始 key 是空间不变的，无法适应 query 中不同位置的结构差异；引入 SFT 用 query 推导的 γ, β 进行仿射变换。

**本文公式（推导）**:
$$\text{Step 1}: \gamma, \beta = \text{MLP}(\text{Pool}(Q)) \quad \text{从 query 聚合空间参数}$$
$$\text{Step 2}: K_{mod} = \gamma \odot K_{raw} + \beta \quad \text{逐通道逐空间调制}$$
$$\text{最终}: K = K_{mod} \text{ 作为交叉注意力的 key 输入}$$

### 模块 2: 交叉注意力上采样核生成器（对应框架图 Attention 模块）

**直觉**: 传统上采样核（双线性、内容感知预测）无法同时利用高分辨率空间引导和低分辨率语义信息；交叉注意力天然实现二者的动态匹配。

**Baseline 公式** (CARAFE 等内容感知上采样):
$$W_{up} = \text{KernelPredictor}(F_{enc}), \quad F_{out} = \text{Reassemble}(F_{enc}, W_{up})$$
符号: $W_{up}$ = 预测的上采样核, KernelPredictor 仅依赖低分辨率特征

**变化点**: CARAFE 的核预测缺少高分辨率图像引导，且核权重与目标位置无关；JAFAR 用 query-key 相似度显式编码空间-语义对应关系。

**本文公式（推导）**:
$$\text{Step 1}: Q = W_Q \cdot F_{image}, \quad K = W_K \cdot \text{SFT}(F_{enc}, Q) \quad \text{query 来自图像，key 经 SFT 调制}$$
$$\text{Step 2}: A = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) \quad \text{注意力权重作为上采样核}$$
$$\text{Step 3}: V = W_V \cdot F_{enc} \quad \text{value 来自 encoder 特征}$$
$$\text{最终}: F_{out} = A \cdot V \quad \text{加权聚合得到上采样特征}$$

**对应消融**: Table 5 显示 attention mechanism 的 key strategy 消融，SIFT（Spatially-Informed Feature Transform）策略提供最佳性能。

### 模块 3: 分辨率泛化机制（隐含于整体框架）

**直觉**: 训练时仅用低上采样比率（如 2×-4×），推理时通过 query 的密集化直接扩展到 14× 甚至更高，无需重新训练。

**Baseline 做法** (LiFT 等): 针对固定输入-输出分辨率对训练
$$F_{out} = \text{Upsampler}_{\theta_{s}}(F_{enc}, s) \quad \text{其中 } s \text{ 为训练时固定尺度}$$

**变化点**: 固定尺度训练无法跨尺度泛化；JAFAR 的 query 构造与分辨率解耦，任意目标分辨率通过调整 query 的密集程度实现。

**本文公式（推导）**:
$$\text{Step 1}: Q_{target} = \text{Interpolate}(F_{image}, H_{target}, W_{target}) \quad \text{query 插值到目标分辨率}$$
$$\text{Step 2}: K, V \text{ 保持低分辨率，通过 attention 的 } QK^T \text{ 实现分辨率桥接}$$
$$\text{最终}: F_{out} \in \mathbb{R}^{H_{target} \times W_{target} \times C} \text{ 与 } Q_{target} \text{ 同分辨率}$$

**对应消融**: 文中指出训练于 32×32 可泛化至 448×448（14× 扩展），Figure 3 PCA 可视化验证高分辨率输出的结构保持性。

## 实验与分析


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/85f5c919-0cb6-4489-b223-d670b99ab659/figures/Table_1.png)
*Table 1 (quantitative): Linear Probing on Perceptual Tasks. JAFAR consistently outperforms other baselines. Best results are shown in bold, second best are underlined.*



本文在多个下游任务和基准上系统评估 JAFAR。核心定量结果汇总于 Table 1（Linear Probing on Perceptual Tasks），覆盖 ADE20K、COCO-Stuff 等数据集的语义分割线性探测，以及 NYU-v2、DIODE 的深度估计。JAFAR 在所有设置下一致超越 Bilinear、CARAFE、DySample、SAPA、ReSFU、LiFT、FeatUp (JBU) 和 FeatUp (Implicit) 等全部 baseline，且同时满足任务无关、直接上采样、轻量推理三项性质——这是现有方法中唯一的（Table 1 方法对比表）。


![Table 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/85f5c919-0cb6-4489-b223-d670b99ab659/figures/Table_2.png)
*Table 2 (quantitative): GradCAM Evaluation. Integrating JAFAR into standard classification backbones improves localization. Lower is better for AD↓ and AI↓. Best results are highlighted in blue, second best in green according to AD↓.*



Grad-CAM 解释性评估（Table 2）进一步验证上采样特征的质量：在 ImageNet validation 2000 张图像上，JAFAR 相比低分辨率基线和传统上采样方法产生更清晰、更准确的类别激活图，artifact 更少。Zero-Shot Open-Vocabulary 语义分割（Table 3）中，基于 MetaCLIP 的 JAFAR 上采样特征在 mIoU 上持续提升。BEV 车辆分割（Table 4）显示 JAFAR 在 lift-splat 和 BEVFormer 两个基线上均取得最佳 mIoU。


![Table 5](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/85f5c919-0cb6-4489-b223-d670b99ab659/figures/Table_5.png)
*Table 5 (ablation): Attention mechanism ablation with respect to key strategy and number of attention blocks. SIFT provides the best performance. Best results per column are highlighted in blue, second best in green.*



消融实验（Table 5）聚焦 attention mechanism 的 key strategy 和 attention block 数量。SIFT（Spatially-Informed Feature Transform）策略优于其他 key 构造变体；增加 attention block 数量带来边际收益，但单个 block 已表现强劲，验证了轻量设计的合理性。Figure 3 的 PCA 可视化直观展示：DINOv2 ViT-S14 的 28×28 token 经 JAFAR 上采样至 448×448 后，特征空间结构清晰保留，而 Bilinear 等方法出现严重模糊。

公平性检查：主要对比的 FeatUp、LiFT 等结果在 Table 1 中使用作者复现的 checkpoint；官方 checkpoint 在附录 Tables 8-10 中显示不同数值，提示复现细节可能影响绝对对比。此外，评估主要基于 DINOv2 系列 backbone，对 CLIP、SAM 等其他 encoder 架构的泛化验证有限。作者披露全局注意力的计算开销较高，并建议局部化变体作为未来方向，但未在本文中实现。

## 方法谱系与知识库定位

JAFAR 属于 **Feature Upsampling for Foundation Models** 方法家族，直接继承自 **FeatUp** 的「模型无关、任意分辨率特征恢复」目标，但在三个核心 slot 上完成替换：

- **Architecture**: FeatUp 的 JBU 空间传播 / Implicit 网络优化 → JAFAR 的交叉注意力 + SFT 调制
- **Data pipeline**: 任务相关或 encoder 特定训练 → 纯低分辨率监督，任务无关训练
- **Inference strategy**: 迭代优化或额外传播 → 单次前向轻量推理

直接 baseline 差异：
- **FeatUp (JBU)**: 模型无关且轻量，但非直接上采样，需 joint bilateral upsampling 后处理
- **FeatUp (Implicit)**: 模型无关且直接上采样，但需迭代网络优化，非轻量
- **LiFT**: 任务无关且轻量，但需针对分辨率训练，非直接上采样
- **CARAFE/DySample/SAPA/ReSFU**: 内容感知或学习采样，但均非任务无关

后续方向：（1）局部化注意力变体以降低全局 attention 的计算和内存开销；（2）向非 DINOv2 架构（CLIP、SAM、扩散模型特征）的系统扩展；（3）与扩散/流模型上采样器的显式对比。

标签：#modality:image #paradigm:supervised_transfer #scenario:dense_prediction #mechanism:cross_attention + SFT_modulation #constraint:lightweight_plugin

## 引用网络

### 直接 baseline（本文基于）

- [[P__无监督视觉特征学习的DINOv2_DINOv2]] _(直接 baseline)_: DINOv2 is a core vision foundation model that JAFAR likely builds upon or direct

