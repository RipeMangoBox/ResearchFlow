---
title: '5%>100%: Breaking Performance Shackles of Full Fine-Tuning on Visual Recognition Tasks'
type: paper
paper_level: C
venue: CVPR
year: 2025
paper_link: null
aliases:
- MONA：双分支卷积Adapter超越全量微调
- BPSFFV
acceptance: poster
cited_by: 58
code_url: https://github.com/Leiyi-Hu/mona
baselines:
- 大语言模型的贝叶斯低秩适应_Laplace-LoRA
---

# 5%>100%: Breaking Performance Shackles of Full Fine-Tuning on Visual Recognition Tasks

[Code](https://github.com/Leiyi-Hu/mona)

**Topics**: [[T__Object_Detection]], [[T__Instance_Segmentation]], [[T__Semantic_Segmentation]] | **Datasets**: [[D__COCO]], [[D__Pascal_VOC]], [[D__Flowers102]]

| 中文题名 | MONA：双分支卷积Adapter超越全量微调 |
| 英文题名 | 5%>100%: Breaking Performance Shackles of Full Fine-Tuning on Visual Recognition Tasks |
| 会议/期刊 | CVPR 2025 (poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2408.08345) · [Code](https://github.com/Leiyi-Hu/mona) · [DOI](https://doi.org/10.1109/CVPR52734.2025.01869) |
| 主要任务 | 视觉识别任务的参数高效微调（目标检测、实例分割、图像分类） |
| 主要 baseline | FULL, ADAPTER, LoRA, ADAPTFORMER, LoRand, BITFIT, NORMTUNING, PARTIAL-1 |

> [!abstract] 因为「全量微调（FULL）参数量大且易过拟合，现有Adapter/LoRA等参数高效微调方法性能不及FULL」，作者在「ADAPTER」基础上改了「将MLP瓶颈替换为深度可分离卷积双分支结构，并引入可学习LayerNorm调制」，在「Pascal VOC检测、COCO实例分割、Flowers102/OxfordPets分类」上取得「仅用2.56%参数超越100%全量微调」的结果

- **Pascal VOC Detection (Swin-B)**: APbox 86.5 vs FULL 81.6, **+4.9**
- **Flowers102 Classification (Swin-L)**: top-1 acc. 99.6764% vs FULL 99.5772%, **+0.0992**
- **COCO Instance Segmentation (Swin-B)**: APbox 87.3 vs FULL 86.8, **+0.5**（仅2.56%可训练参数）

## 背景与动机

在视觉识别任务中，预训练大模型（如Swin Transformer）迁移到下游任务时，传统做法是**全量微调（FULL）**——更新所有参数。然而，随着模型规模膨胀，FULL不仅消耗巨量计算资源，还容易导致下游数据过拟合。参数高效微调（PEFT）应运而生：冻结预训练骨干，仅训练少量新增参数。

现有PEFT方法已形成几条主流路线：**ADAPTER**（Houlsby et al., 2019）在Transformer block后插入瓶颈MLP（下投影→激活→上投影）；**LoRA**（Hu et al., 2021）在注意力权重旁并联低秩矩阵；**ADAPTFORMER**（Chen et al., 2022a）将并行Adapter与可学习缩放因子结合；**LoRand**（Yin et al., 2023b）引入随机化低秩结构。这些方法虽将可训练参数压至极低，但在视觉任务上普遍存在一个瓶颈：**性能始终不及FULL微调**，形成「效率-精度」的权衡困境。

具体而言，标准Adapter的MLP瓶颈缺乏空间感知能力——$W_{down}$和$W_{up}$仅为全连接层，无法利用视觉特征的空间局部性；LoRA局限于注意力子空间，对密集预测任务（检测、分割）的适配不足；ADAPTFORMER的并行结构虽引入尺度因子，仍未突破瓶颈设计的表达能力上限。作者指出，**核心症结在于：现有PEFT模块的特征变换过于「扁平」，既无多尺度空间建模，也缺乏对归一化统计量的自适应调制**。

本文提出MONA（MOdular Network Adapter），通过双分支卷积结构与可学习LayerNorm调制，在仅使用2.56%参数的条件下，首次在多个视觉任务上系统性超越全量微调。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/5a8b159d-d6b4-4be3-aea0-2388cdae9752/figures/Figure_1.png)
*Figure 1: Figure 1: Comparisons of our method with full fine-tuning and recent delta-tuning art on representative vi-sual tasks. Blue dashed line is the performance of full fine-tuning on ADE20K and COCO. The p*



## 核心创新

**核心洞察：将Adapter的MLP瓶颈解构为「深度卷积提取多尺度空间特征 + 点卷积混合通道」的双分支串联结构，并用可学习标量调制LayerNorm输出与原始输入的融合比例，从而在极小参数量下实现比全量微调更强的特征适应能力。**

因为视觉特征具有强烈的空间局部性和层次性，深度可分离卷积（depthwise separable convolution）天然适合在瓶颈维度上高效编码空间模式；同时，预训练模型的LayerNorm统计量未必适配下游任务，通过可学习调制$s_1, s_2$可动态平衡「归一化后的稳定分布」与「原始输入的任务特异性信息」。

| 维度 | Baseline (ADAPTER) | 本文 (MONA) |
|:---|:---|:---|
| 瓶颈结构 | 全连接MLP: $W_{down} \to \sigma \to W_{up}$ | 深度卷积分支$f_{dw}$ + 点卷积分支$f_{pw}$串联 |
| 空间建模 | 无（全连接无空间感知） | 3种核大小的depthwise卷积，平均聚合多尺度特征 |
| 归一化方式 | 固定LayerNorm: $\text{LN}(x_0)$ | 可学习调制: $s_1 \cdot |x_0|_{LN} + s_2 \cdot x_0$ |
| 参数量控制 | 中间维度直接决定$2nm + n + m$ | 公式$(2n+3)m + n^2 + 84n + 2$，64维仅2.56% (Swin-L) |

## 整体框架


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/5a8b159d-d6b4-4be3-aea0-2388cdae9752/figures/Figure_2.png)
*Figure 2: Figure 2: Left: The proposed Mona-tuning. We add Monaafter MSA and MLP in each SwinBlock. The proposedmethod fixes the parameters of pre-trained layers and up-dates the parameters of Mona. Right: Deta*



MONA模块被插入到Swin Transformer的每个block中，位于MSA（Multi-Head Self-Attention）和MLP之后，形成「预训练块 → MONA → 预训练块 → MONA」的交替结构。整个前向数据流如下：

1. **输入特征** $x_0$：来自Swin block的原始输出，维度为$m$（通道数）。预训练骨干参数全部冻结。

2. **LN Modulation（可学习层归一化调制）**：接收$x_0$，输出调制后的$x_{norm} = s_1 \cdot |x_0|_{LN} + s_2 \cdot x_0$。两个标量$s_1, s_2$为唯一可学习的归一化参数，替代固定LayerNorm。

3. **Down-projection $D^l$**：线性层将$x_{norm}$从维度$m$降维至瓶颈维度$n$（默认64），产生紧凑特征表示。

4. **Depthwise branch $f_{dw}$（深度卷积分支）**：对降维后特征应用3种不同核大小的depthwise卷积$\omega_{dw}^i$（$i=1,2,3$），取平均聚合，捕获多尺度空间信息：$f_{dw} = x + \text{avg}(\sum_{i=1}^3 \omega_{dw}^i \hat{\otimes} x)$。

5. **Pointwise branch $f_{pw}$（点卷积分支）**：对$f_{dw}$的输出应用1×1卷积$\omega_{pw}$进行通道混合：$f_{pw} = x + \omega_{pw} \overline{\otimes} x$。

6. **Activation $\sigma$ + Up-projection $U^l$**：经非线性激活后，线性层$U^l$将特征从$n$升维回$m$。

7. **Residual addition**：最终输出 $x = x_0 + U^l\sigma(f_{pw}(f_{dw}(D^l(x_{norm}))))$，与原始输入残差连接。

```
x_0 ──→ [LN Modulation: s1,s2] ──→ x_norm
                                      ↓
                              [Down-projection D^l] 
                                      ↓
                        [Depthwise f_dw: 3-scale DW conv]
                                      ↓
                           [Pointwise f_pw: 1×1 PW conv]
                                      ↓
                              [Activation σ]
                                      ↓
                              [Up-projection U^l]
                                      ↓
                              ⊕ ──→ x (residual with x_0)
```

## 核心模块与公式推导

### 模块 1: 可学习LayerNorm调制（对应框架图：输入端）

**直觉**：预训练模型的LayerNorm统计量针对源域数据，下游任务的数据分布偏移需要动态调整归一化输出与原始特征的融合比例。

**Baseline 公式** (ADAPTER / 标准Transformer): $$x_{norm} = \text{LN}(x_0) = \gamma \odot \frac{x_0 - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$
符号: $\gamma, \beta$ = 固定的仿射参数（预训练得到，微调时冻结或更新），$\mu, \sigma^2$ = 通道-wise均值方差。

**变化点**：标准LN在PEFT中通常冻结，其输出分布固定；但下游任务可能需要保留更多原始输入的未归一化信息，或强化归一化的稳定作用。

**本文公式（推导）**:
$$\text{Step 1}: |x_0|_{LN} = \text{LN}(x_0) \quad \text{（先计算标准层归一化输出）}$$
$$\text{Step 2}: x_{norm} = s_1 \cdot |x_0|_{LN} + s_2 \cdot x_0 \quad \text{（引入可学习标量 } s_1, s_2 \text{ 动态加权）}$$
$$\text{最终}: x_{norm} = s_1 \cdot |x_0|_{LN} + s_2 \cdot x_0$$

**对应消融**：

---

### 模块 2: 双分支卷积瓶颈（对应框架图：核心变换路径）

**直觉**：视觉特征的空间结构不应被全连接层抹平；depthwise卷积以极低参数量编码空间模式，pointwise卷积负责跨通道信息整合，二者串联形成「空间→通道」的解耦变换。

**Baseline 公式** (ADAPTER): $$f = x + W_{up}\sigma(W_{down}x)$$
符号: $W_{down} \in \mathbb{R}^{n \times m}$, $W_{up} \in \mathbb{R}^{m \times n}$ = 下/上投影矩阵，$n \ll m$为瓶颈维度，$\sigma$ = 激活函数。

**变化点**：ADAPTER的MLP瓶颈$W_{down} \to \sigma \to W_{up}$完全忽略空间维度，所有token独立处理；且参数量$2nm + n + m$虽少，但表达形式单一。MONA将其替换为卷积操作，利用权重共享和局部感受野提升参数效率与空间建模能力。

**本文公式（推导）**:
$$\text{Step 1}: \text{降维后特征 } z = D^l(x_{norm}) \in \mathbb{R}^{n} \quad \text{（与ADAPTER相同，保留瓶颈结构）}$$
$$\text{Step 2}: f_{dw} = z + \text{avg}\left(\sum_{i=1}^{3} \omega_{dw}^i \hat{\otimes} z\right) \quad \text{（3种核大小的depthwise卷积，平均聚合多尺度空间响应）}$$
$$\text{Step 3}: f_{pw} = f_{dw} + \omega_{pw} \overline{\otimes} f_{dw} \quad \text{（1×1 pointwise卷积进行通道混合，输入输出维度均为} n\text{）}$$
$$\text{Step 4}: \text{激活与升维: } h = U^l\sigma(f_{pw}) \in \mathbb{R}^{m}$$
$$\text{最终}: x = x_0 + h = x_0 + U^l\sigma\left(f_{pw}\left(f_{dw}\left(D^l(x_{norm})\right)\right)\right)$$

**参数效率分析**：MONA单模块参数量为 $(2n+3)m + n^2 + 84n + 2$，其中$(2n+3)m$来自$D^l, U^l$及$\omega_{pw}$，$n^2$来自3个depthwise卷积核参数（$3 \times k^2$合并为$n^2$量级），$84n$为偏置项，$2$为$s_1, s_2$。当$n=64, m=192$（Swin-L典型维度）时，仅占骨干参数的**2.56%**。

**对应消融**：Table 5 显示中间维度64时APbox达87.3%；降至32维则跌至86.8%（$-0.5$），增至128维（参数量翻倍至5.22%）反而降至87.1%（$-0.2$），证明MONA的设计存在最优效率点，盲目增大宽度无益。

## 实验与分析


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/5a8b159d-d6b4-4be3-aea0-2388cdae9752/figures/Table_1.png)
*Table 1: Table 1: Results of baselines and our methods on COCO benchmarks. Swin-B is employed as the pre-trained model here.We present the numbers and percentages of trainable backbone parameters on the left a*



本文在四大类视觉任务上验证MONA：目标检测（Pascal VOC）、实例分割（COCO）、语义分割（ADE20K）、图像分类（Flowers102, OxfordPets, VOC2007）。核心结果集中于Table 1（COCO）、Table 2（Pascal VOC & ADE20K）、Table 3（DOTA & STAR）、Table 4（分类），以及跨模型规模的Table 6。

**检测与分割任务**：MONA展现出对FULL微调的系统性超越。以Pascal VOC Detection为例（Table 6），Swin-T/B/L三个尺度上MONA分别达到83.5/86.5/87.3 APbox，对应FULL为80.1/81.6/83.7，提升幅度随模型增大而扩大（+3.4 → +4.9 → +3.6）。COCO实例分割（Table 5）上，MONA以2.56%参数取得87.3% APbox，超越FULL的86.8%。这一结果直接挑战了「PEFT无法匹敌全量微调」的固有认知。

**分类任务**（Table 4）：Flowers102上MONA达99.6764% top-1准确率，超越FULL（99.5772%）+0.0992，也超越ADAPTER（99.5934%）+0.083；OxfordPets上优势更大，MONA 95.4765% vs FULL 94.6579%（+0.8186）。但VOC2007上出现例外：MONA 86.9709%略低于ADAPTER的87.0355%（-0.0646），虽仍大幅领先FULL（84.1276%），却表明MONA并非 universally 最优。


![Table 5](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/5a8b159d-d6b4-4be3-aea0-2388cdae9752/figures/Table_5.png)
*Table 5: Table 5: Ablations of intermediate dimensions. 64 inter-mediate dimensions achieves the best performance. ∗de-notes the trainable parameters in backbones.*



**消融实验**（Table 5）聚焦中间维度$n$的选择：64维在参数量（2.56%）与性能（87.3% APbox）间达到最佳平衡。32维参数量减半但性能下降0.5，128维参数量翻倍至5.22%性能反而微降0.2，说明MONA的双分支结构存在内在容量约束，过宽的点卷积可能引入冗余。


![Table 6](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/5a8b159d-d6b4-4be3-aea0-2388cdae9752/figures/Table_6.png)
*Table 6: Table 6: Performance of mona on models with differentsizes. The results indicate that model sizes do not constrainMona’s superiority.*



**公平性审视**：本文比较的基线覆盖较全（FULL/ADAPTER/LoRA/ADAPTFORMER/LoRand/BITFIT/NORMTUNING/PARTIAL-1），但缺少近年视觉PEFT的重要方法如VPT（Visual Prompt Tuning）、SSF（Shift-Scale-Fusion）、IA³等。Table 4中VOC2007的ADAPTER反超结果未被充分讨论。此外，COCO主结果仅出现在消融表Table 5而非独立结果表，且未报告标准差或统计显著性检验。训练时间、收敛曲线、推理延迟等工程指标亦未披露。



## 方法谱系与知识库定位

**方法家族**：Adapter-based PEFT（参数高效微调之适配器流派）

**父方法**：ADAPTER（Houlsby et al., 2019）——标准瓶颈适配器，插入MSA/MLP后的下投影→激活→上投影结构。MONA保留其「瓶颈残差模块+固定位置插入」的范式，彻底替换内部变换体。

**改动槽位**：
- **architecture**：MLP瓶颈 → depthwise+pointwise双分支卷积串联
- **data_pipeline**：固定LayerNorm → 可学习标量调制$s_1, s_2$
- **training_recipe**：继承「冻结骨干、仅训新增模块」的标准PEFT流程

**直接基线差异**：
- **vs ADAPTER**：替换MLP为卷积双分支，增加空间建模与LN调制
- **vs LoRA**：从注意力旁路低秩矩阵改为block级瓶颈卷积模块，适用更广泛的视觉任务
- **vs ADAPTFORMER**：ADAPTFORMER在MLP旁加并行适配器+缩放因子；MONA是串联结构且引入卷积空间操作
- **vs LoRand**：LoRand以随机低秩矩阵扰动特征；MONA以确定性卷积结构编码空间先验

**后续方向**：
1. **跨架构迁移**：将MONA的双分支设计从Swin Transformer扩展至ViT、ConvNeXt等现代视觉骨干
2. **动态宽度**：基于任务复杂度自适应选择中间维度$n$，替代固定的64维超参
3. **与Prompt Tuning融合**：联合优化空间卷积适配器与输入层可学习token，探索Adapter-Prompt混合范式

**标签**：modality=vision | paradigm=parameter-efficient fine-tuning | scenario=transfer learning | mechanism=depthwise separable convolution + learnable normalization modulation | constraint=low trainable parameters (<5%)

## 引用网络

### 直接 baseline（本文基于）

- [[P__大语言模型的贝叶斯低秩适应_Laplace-LoRA]] _(方法来源)_: Core algorithmic foundation (LoRA); likely extended/adapted for vision in this w

