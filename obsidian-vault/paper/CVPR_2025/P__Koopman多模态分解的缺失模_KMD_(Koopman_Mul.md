---
title: 'KMD: Koopman Multi-modality Decomposition for Generalized Brain Tumor Segmentation under Incomplete Modalities'
type: paper
paper_level: C
venue: CVPR
year: 2025
paper_link: null
aliases:
- Koopman多模态分解的缺失模态脑肿瘤分割
- KMD (Koopman Mul
- KMD (Koopman Multi-modality Decomposition)
acceptance: poster
cited_by: 5
method: KMD (Koopman Multi-modality Decomposition)
---

# KMD: Koopman Multi-modality Decomposition for Generalized Brain Tumor Segmentation under Incomplete Modalities

**Topics**: [[T__Semantic_Segmentation]], [[T__Medical_Imaging]], [[T__Domain_Adaptation]] | **Method**: [[M__KMD]] | **Datasets**: BraTS2018, BraTS2020

| 中文题名 | Koopman多模态分解的缺失模态脑肿瘤分割 |
| 英文题名 | KMD: Koopman Multi-modality Decomposition for Generalized Brain Tumor Segmentation under Incomplete Modalities |
| 会议/期刊 | CVPR 2025 (poster) |
| 链接 | [DOI](https://doi.org/10.1109/cvpr52734.2025.01460) |
| 主要任务 | 不完整多模态MRI分割、脑肿瘤分割（WT/TC/ET） |
| 主要 baseline | RFNet, mmFormer, M2FTrans, Passion, ShaSpec, D2-Net, Correlation-Fusion |

> [!abstract] 因为「临床中MRI模态（FLAIR/T1/T1ce/T2）经常部分缺失导致多模态融合失效」，作者在「Passion随机掩码框架」基础上改了「引入Koopman算子进行模态公共/特定特征分解，并显式构建模态间关系」，在「BraTS2018平衡缺失和BraTS2020不平衡缺失基准」上取得「RFNet ET Dice从30.89提升至68.50（仅T1存在时）, mmFormer ET Dice从37.23提升至70.91（仅T1ce缺失时）」

- **RFNet+KMD** 在BraTS2018平衡缺失场景下，当仅T1模态存在时ET Dice从30.89提升至68.50
- **mmFormer+KMD** 在BraTS2018平衡缺失场景下，当仅T1ce缺失时ET Dice从37.23提升至70.91
- **M2FTrans+KMD** 在BraTS2020不平衡缺失场景下相比Passion增强版本在TC和ET类别上均有显著提升

## 背景与动机

脑肿瘤分割是医学图像分析中的核心任务，临床标准通常需要四种MRI模态：FLAIR、T1、T1ce和T2，分别用于识别肿瘤的不同子区域（Whole Tumor WT, Tumor Core TC, Enhancing Tumor ET）。然而在实际临床场景中，由于扫描时间限制、患者运动伪影或设备故障，经常出现部分模态缺失的情况——例如急诊患者可能只做了T1和FLAIR，而缺少增强扫描T1ce。这种**不完整多模态问题**使得标准的多模态融合方法面临严峻挑战。

现有方法主要从三个方向应对这一问题：

**Passion** [12] 采用随机模态掩码策略，在训练时随机遮蔽部分模态以模拟缺失场景，增强模型的鲁棒性。该方法实现简单，但仅通过数据增强层面处理缺失，未对模态间的内在结构进行显式建模。

**ShaSpec** [17] 尝试通过共享-特定特征分离来处理缺失模态，构建模态间的共享表示空间。但其关系构建机制较为隐式，当模态缺失率较高时，共享特征与特定特征容易发生混淆，导致分割边界模糊。

**D2-Net** [21] 和 **Correlation-Fusion** [19] 等分解方法尝试将多模态特征解耦，但主要依赖统计相关性或简单的分解策略，缺乏对模态动态演化特性的刻画，难以保证公共特征的一致性和特定特征的判别性。

这些方法的共同局限在于：**未能显式分离模态公共信息（modality-common）与模态特定信息（modality-specific），也未建立两者之间的结构化关系**。当模态缺失时，公共特征可能漂移，特定特征可能相互干扰，导致肿瘤核心（TC）和增强区域（ET）的分割精度急剧下降——而这正是临床诊断最关键的指标。

本文提出KMD，首次将**Koopman算子理论**引入多模态医学图像分割，通过线性算子刻画模态特征的非线性动态演化，实现公共特征与特定特征的显式解耦，并构建两者间的结构化关系，从而在任意模态缺失组合下保持稳定的分割性能。


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/31246f56-5b7b-4da9-9507-bc52ba79ba91/figures/fig_001.png)
*Figure: Architecture illustration of different kinds of methods.*



## 核心创新

核心洞察：**多模态特征的公共-特定分解可以建模为动态系统的线性演化问题**，因为Koopman算子能够将非线性动态嵌入到高维线性空间中，从而使在缺失模态下保持特征空间的结构一致性成为可能。

| 维度 | Baseline (Passion) | 本文 (KMD) |
|:---|:---|:---|
| 特征表示 | 单一融合特征，隐式包含模态信息 | 显式分解为modality-common和modality-specific双分支 |
| 分解机制 | 无显式分解，依赖随机掩码的隐式学习 | Koopman算子驱动的线性演化分解 |
| 模态关系 | 无显式关系建模 | 公共特征聚类约束 + 特定特征正交约束 |
| 缺失处理 | 数据增强层面的鲁棒性 | 特征空间结构层面的不变性 |

与Passion相比，KMD不改变骨干网络结构，而是作为即插即用的分解模块插入到现有分割网络中；与ShaSpec相比，KMD的分解基于动态系统理论而非统计假设，具有更明确的数学保证；与D2-Net等分解方法相比，KMD通过Koopman算子的谱性质实现了公共特征的低维一致流形和特定特征的高维正交补空间。

## 整体框架


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/31246f56-5b7b-4da9-9507-bc52ba79ba91/figures/fig_002.png)
*Figure: Architecture of the KMD on baseline models. There are four modalities in the dataset, and sample X2*



KMD的整体框架以现有分割网络（RFNet/mmFormer/M2FTrans）为骨干，在特征提取后插入Koopman分解与关系构建模块。数据流如下：

**输入**：不完整多模态MRI图像 $\{X^m\}_{m \in \mathcal{M}_{avail}}$，其中 $\mathcal{M}_{avail} \subseteq \{FLAIR, T1, T1ce, T2\}$

**Step 1 - 多模态编码器（Multi-modal encoder）**：各可用模态独立通过编码器提取初始特征 $f_m \in \mathbb{R}^{C \times H \times W}$，保持模态间的空间对齐。

**Step 2 - KMD分解模块（KMD decomposition module）**：将各模态特征输入Koopman算子层，分解为：
- $f_m^{common}$：模态公共特征，通过Koopman算子的主导特征函数捕获跨模态共享的肿瘤语义
- $f_m^{specific}$：模态特定特征，对应算子的剩余谱分量，保留模态独有的对比度信息

**Step 3 - 关系构建（Relationship construction）**：
- 公共特征分支：通过对比学习约束，使不同模态的 $f_m^{common}$ 在嵌入空间中聚类，实现模态无关的一致性表示
- 特定特征分支：通过正交性约束，确保 $f_m^{specific} \perp f_{m'}^{specific}$，避免模态间信息泄露

**Step 4 - 分割解码器（Segmentation decoder）**：将聚合后的公共特征与加权的特定特征拼接，输入原有分割头，输出WT/TC/ET三类分割掩码。

```
[Input MRI] → [Encoder: f_m] → [KMD: f_m^common ⊕ f_m^specific]
                                    ↓
                    [Relationship: cluster + orthogonal]
                                    ↓
                    [Fusion: Σ_common + Σ_weighted(specific)]
                                    ↓
                    [Decoder] → [WT/TC/ET Masks]
```

该框架的关键特性是**即插即用性**：KMD模块可嵌入任意现有分割网络，不改变原始编码器-解码器结构，仅需在训练时加入分解损失和关系约束。

## 核心模块与公式推导

### 模块 1: Koopman算子特征分解（对应框架图 Step 2）

**直觉**：将多模态特征的非线性动态视为确定性演化系统，Koopman算子可以提取其线性谱结构，主导分量即为公共语义，残余分量即为模态特异性。

**Baseline 公式** (Passion [12]): 无显式分解，直接融合
$$f_{fuse} = \sum_{m \in \mathcal{M}_{avail}} w_m \cdot f_m$$
符号: $f_m$ = 模态m的编码特征, $w_m$ = 可学习权重, 无公共/特定分离。

**变化点**：Passion的融合是隐式的，缺失模态时权重重新归一化导致特征分布漂移；KMD通过Koopman算子 $K$ 的谱分解显式分离不变子空间。

**本文公式（推导）**：
$$\text{Step 1}: \quad \mathcal{K} \phi(f_m) = \lambda \phi(f_m) \quad \text{(Koopman特征方程，φ为观测函数)}$$
$$\text{Step 2}: \quad f_m = \underbrace{\sum_{k=1}^{K} a_k \phi_k}_{f_m^{common}} + \underbrace{\sum_{k>K} a_k \phi_k}_{f_m^{specific}} \quad \text{(按特征值幅度截断，前K个主导模态为公共特征)}$$
$$\text{最终}: \quad f_m^{common} = \Phi_K \cdot \text{softmax}(a_{1:K}), \quad f_m^{specific} = f_m - f_m^{common}$$
其中 $\Phi_K = [\phi_1, ..., \phi_K]$ 为Koopman算子前K个主导特征函数构成的基，通过EDMD（Extended Dynamic Mode Decomposition）从多模态特征样本中数据驱动学习。

**对应消融**：Figure 4的t-SNE可视化显示，加入KMD后模态公共特征聚为一簇，模态特定特征按模态类型分离且边界清晰；去掉KMD后特征混叠严重。

---

### 模块 2: 模态关系构建损失（对应框架图 Step 3）

**直觉**：公共特征应模态无关（不同模态的同类肿瘤区域表示相近），特定特征应模态互斥（避免信息冗余）。

**Baseline 公式** (ShaSpec [17]): 隐式共享约束
$$L_{ShaSpec} = \|f_{shared} - f_{target}\|_2^2 + \lambda \|f_{specific}\|_1$$
符号: $f_{shared}$ = 共享特征, $f_{specific}$ = 特定特征, 通过L2/L1稀疏约束分离，但无显式跨模态关系。

**变化点**：ShaSpec的分离是单样本内的，未利用跨模态的批次统计信息；KMD引入模态间的对比聚类和正交约束，形成显式关系图。

**本文公式（推导）**：
$$\text{Step 1 (公共特征聚类)}: \quad L_{common} = -\sum_{m,m'} \log \frac{\exp(sim(f_m^{common}, f_{m'}^{common})/\tau)}{\sum_{m''} \exp(sim(f_m^{common}, f_{m''}^{common})/\tau)}$$
$$\text{Step 2 (特定特征正交)}: \quad L_{specific} = \sum_{m \neq m'} \| (f_m^{specific})^T f_{m'}^{specific} \|_F^2$$
$$\text{Step 3 (重构保证)}: \quad L_{recon} = \| f_m - (f_m^{common} + f_m^{specific}) \|_2^2$$
$$\text{最终}: \quad L_{KMD} = L_{seg} + \alpha (L_{common} + \beta L_{specific} + \gamma L_{recon})$$
其中 $sim(\cdot,\cdot)$ 为余弦相似度，$\tau$ 为温度系数，$\alpha, \beta, \gamma$ 为平衡超参。$L_{seg}$ 为原始分割损失（Dice + Cross-Entropy）。

**对应消融**：Table 3（分解方法比较）显示，相比Correlation-Fusion和D2-Net，KMD在TC和ET上的Dice提升显著；去掉关系构建项（$\beta=0$）后，t-SNE中特定特征边界模糊，ET Dice下降约5-8%。

---

### 模块 3: 缺失模态下的特征聚合（对应框架图 Step 4）

**直觉**：当部分模态缺失时，公共特征提供稳定的肿瘤语义锚点，特定特征根据可用模态动态加权补充细节。

**Baseline 公式** (标准融合): 缺失时直接丢弃或零填充
$$f_{fuse}^{missing} = \sum_{m \in \mathcal{M}_{avail}} w_m f_m, \quad w_m \leftarrow w_m / \sum w_m \text{ (重新归一化)}$$

**变化点**：重新归一化导致特征尺度变化和语义漂移；KMD利用公共特征的模态不变性作为稳定基，特定特征按需补充。

**本文公式（推导）**：
$$\text{Step 1 (公共特征聚合)}: \quad f^{common}_{agg} = \frac{1}{|\mathcal{M}_{avail}|} \sum_{m \in \mathcal{M}_{avail}} f_m^{common} \quad \text{(公共特征平均，模态无关)}$$
$$\text{Step 2 (特定特征加权)}: \quad f^{specific}_{agg} = \sum_{m \in \mathcal{M}_{avail}} g_m \cdot f_m^{specific}, \quad g_m = \frac{\exp(w_m)}{\sum_{m'} \exp(w_{m'})}$$
$$\text{最终}: \quad f_{final} = [f^{common}_{agg}; f^{specific}_{agg}] \rightarrow \text{Decoder}$$
其中 $[;]$ 表示通道拼接。关键性质：当仅单模态可用时，$f^{common}_{agg} = f_m^{common}$ 仍保持完整语义，避免信息坍缩。

**对应消融**：Table 1显示，在极端缺失场景（仅T1存在）下，RFNet基线ET Dice仅30.89，加入KMD后提升至68.50，验证了公共特征锚定效应的有效性。

## 实验与分析



本文在BraTS2018和BraTS2020两个数据集上评估KMD的通用性和有效性。BraTS2018采用**平衡缺失协议**（各模态缺失概率均等），BraTS2020采用**不平衡缺失协议**（模拟临床中T1ce和T2更易缺失的偏态分布）。

**核心结果**：在BraTS2018平衡缺失场景下，KMD作为即插即用模块显著提升三个骨干网络的性能。以RFNet为骨干时，当仅T1模态可用这一最极端场景，ET（增强肿瘤）Dice从30.89飙升至68.50，提升幅度达+37.61；mmFormer在仅T1ce缺失时ET Dice从37.23提升至70.91。TC（肿瘤核心）类别同样获得大幅提升，验证了KMD对临床关键区域的保护能力。M2FTrans+KMD在BraTS2020不平衡缺失下相比Passion增强版本展现出更好的模态不变性，尤其在ET类别上优势显著。


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/31246f56-5b7b-4da9-9507-bc52ba79ba91/figures/fig_003.png)
*Figure: Segmentation results of (a) mmFormer and (b) M2FTrans under balanced and imbalanced missing rates, imbalanced missing*





**消融分析**：Figure 4的t-SNE可视化提供了关键的定性证据——加入KMD前，不同模态的特征混叠严重，缺失模态时公共表示漂移；加入KMD后，模态公共特征聚为紧密簇（跨模态一致），模态特定特征按FLAIR/T1/T1ce/T2四类分离且边界清晰。去掉关系构建模块后，特定特征正交性丧失，边界模糊，对应ET Dice下降约5-8%。与D2-Net和Correlation-Fusion的分解方法对比（Table 3），KMD的Koopman谱分解在TC和ET上均优于基于统计相关性的替代方案。

**公平性审视**：本文比较的基线中，Passion作为直接父方法具有代表性，但缺少与更近期的SOTA方法（如专为BraTS2020 leaderboard设计的方案）的对比。训练配置为单张24G RTX 3090 Ti，300 epochs，属于中等计算预算。作者声称"模态不变性"但未给出明确的量化不变性指标（如模态间表示距离的方差），主要依赖t-SNE和分割Dice间接验证。此外，Table 1的文本提取存在格式混乱，部分精确数字难以核对。

## 方法谱系与知识库定位

**方法家族**：不完整多模态学习 → 特征分解与掩码学习

**父方法**：Passion [12] —— KMD直接在其随机模态掩码框架上扩展，保留了训练时的随机遮蔽策略，但新增了Koopman分解和关系构建两个核心模块。

**变更槽位**：
- **architecture**：从"无显式分解的标准融合"改为"Koopman算子驱动的公共/特定双分支分解+显式关系构建"
- **data_pipeline**：从"纯随机掩码"改为"随机掩码+分解模块联合训练"
- **objective**：新增$L_{common}$对比聚类损失、$L_{specific}$正交损失、$L_{recon}$重构损失

**直接基线差异**：
- **vs Passion**：同框架下增加显式分解，从数据增强层面提升到特征结构层面
- **vs ShaSpec**：从隐式共享约束变为显式Koopman谱分解+跨模态对比学习
- **vs D2-Net/Correlation-Fusion**：从统计相关性分解变为动态系统算子分解，具有更好的谱分离性质

**后续方向**：
1. 将Koopman分解扩展到其他医学影像任务（如多序列心脏MRI、多参数前列腺MRI）
2. 结合神经算子（Neural Operator）学习连续的Koopman生成模型，实现任意模态组合的插值生成
3. 引入模态缺失的因果推断框架，显式建模缺失机制而非仅处理缺失结果

**知识库标签**：
- modality: multi-modal MRI (FLAIR/T1/T1ce/T2)
- paradigm: feature decomposition + contrastive learning
- scenario: incomplete modalities, clinical deployment
- mechanism: Koopman operator, spectral decomposition, cross-modal relationship
- constraint: plug-and-play, backbone-agnostic, 24G GPU trainable

