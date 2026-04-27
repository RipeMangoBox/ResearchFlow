---
title: 'VisDiff: SDF-Guided Polygon Generation for Visibility Reconstruction, Characterization and Recognition'
type: paper
paper_level: A
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- SDF引导扩散的可见性图重建
- VisDiff
- VisDiff introduces a novel diffusio
acceptance: Poster
code_url: https://rahulmoorthy19.github.io/VisDiff/
method: VisDiff
modalities:
- graph
- Image
paradigm: supervised
---

# VisDiff: SDF-Guided Polygon Generation for Visibility Reconstruction, Characterization and Recognition

[Code](https://rahulmoorthy19.github.io/VisDiff/)

**Topics**: [[T__3D_Reconstruction]] | **Method**: [[M__VisDiff]] | **Datasets**: Visibility Reconstruction, Vertex Prediction, Baseline, SDF Evaluation, SDF-to-polygon Error Analysis

> [!tip] 核心洞察
> VisDiff introduces a novel diffusion-based approach that first estimates the signed distance function (SDF) of a polygon before extracting vertex locations, enabling significantly more effective learning of visibility relationships than direct vertex generation.

| 中文题名 | SDF引导扩散的可见性图重建 |
| 英文题名 | VisDiff: SDF-Guided Polygon Generation for Visibility Reconstruction, Characterization and Recognition |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://doi.org/10.48550/arxiv.2410.05530) · [Code](https://rahulmoorthy19.github.io/VisDiff/) · [Project](https://rahulmoorthy19.github.io/VisDiff/) |
| 主要任务 | Visibility Graph Reconstruction（可见性图重建）、Visibility Recognition（可见性识别）、Triangulation Graph Generation（三角剖分图生成） |
| 主要 baseline | Marching Cubes Algorithm、PolygonGNN、PolyDiffuse、PolyDiff、Vertex-Diffusion、Conditional-VAE、GNN、MeshAnything |

> [!abstract] 因为「直接顶点生成对组合结构敏感，微小顶点变化导致可见性图剧烈重组」，作者在「Diffusion-SDF / PolyDiffuse」基础上改了「以SDF为中间表示的两阶段扩散架构，先用U-Net+Spatial Transformer交叉注意力做条件SDF扩散，再从SDF轮廓初始化并精炼顶点」，在「可见性图重建基准」上取得「F1 0.912，相比Marching Cubes提升+14.0%，相比联合训练提升+7.3%」

- **F1 Score 0.912**：VisDiff在顶点预测任务上达到0.912，超过传统Marching Cubes算法（0.800）+0.112
- **两阶段训练增益**：SDF扩散预训练+顶点初始化精炼的F1 0.912，优于联合训练（无初始化）的0.850，差距+0.062
- **26%相对提升**：SDF引导方法在F1-Score上比标准及最先进方法提升26%（摘要声明）

## 背景与动机

可见性图重建（Visibility Reconstruction）是一个经典但长期未被充分研究的组合几何问题：给定一个多边形的可见性图——即图中每个节点代表一个顶点，边代表顶点间相互可见的关系——如何恢复原始多边形的几何形状？这个问题的难点在于其固有的组合敏感性：多边形顶点的微小位移可能导致可见性边的剧烈增减，使得直接从图结构映射到顶点坐标成为高度不适定的问题。

现有方法主要从三个方向尝试解决。第一类是**直接顶点扩散方法**（如PolyDiffuse、PolyDiff、Vertex-Diffusion），它们在顶点坐标空间上直接施加扩散过程，但忽略了多边形内部的几何连续性，难以处理可见性图到几何的复杂映射。第二类是**GNN嵌入方法**（如PolygonGNN、标准GNN），通过图神经网络学习可见性图的表示，再解码为顶点位置，但图嵌入对几何细节的捕捉能力有限，无法恢复精确的顶点坐标。第三类是**传统几何算法**（如Marching Cubes），从标量场中提取等值面，但缺乏数据驱动的学习能力，对噪声和复杂拓扑的鲁棒性不足。

这些方法的共同缺陷在于**跳过了几何形状的连续中间表示**：它们要么直接在离散的顶点集合上操作，要么依赖图结构的全局嵌入，都未能利用多边形区域内部的几何先验。具体而言，直接顶点生成面临"小变化大问题"——顶点坐标的小扰动导致可见性图完全改变，使扩散模型的训练极不稳定；GNN方法则受限于消息传递的过度平滑，难以区分几何上相近但可见性不同的配置。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/4005c6d0-50ed-486c-bd26-491c0161e58e/figures/Figure_1.png)
*Figure 1 (motivation): A polygon P is given by an ordered list of 2D locations. Min/max locations of P are extracted and normalized to produce an SDF. Given a visibility graph of a polygon, the task is to reconstruct P. We show the intermediate SDF of each polygon. Finally, learned GNN embedding.*



本文提出VisDiff，核心思想是**先估计有符号距离函数（SDF）作为几何桥梁，再从SDF恢复顶点**，从而将组合敏感的顶点预测问题转化为几何连续的场估计问题。

## 核心创新

核心洞察：SDF作为几何连续中间表示可以解耦组合结构与精确坐标，因为SDD场对顶点扰动具有自然平滑性，从而使从可见性图到多边形的端到端学习成为可能。

| 维度 | Baseline（PolyDiffuse / PolygonGNN） | 本文（VisDiff） |
|:---|:---|:---|
| 生成目标 | 直接预测顶点坐标集合 | 先预测SDF场，再提取并精炼顶点 |
| 条件机制 | 图嵌入全局条件或无显式条件 | Spatial Transformer交叉注意力，Q来自CNN空间特征，K/V来自GNN图编码 |
| 训练方式 | 端到端联合训练 | 两阶段训练：先SDF扩散60epoch，再顶点预测（含SDF轮廓初始化） |
| 几何先验 | 无显式几何中间表示 | SDF编码器提供像素对齐特征Z_pix和全局特征Z_global |

与PolyDiffuse等直接顶点扩散方法的本质差异在于：VisDiff将"图→顶点"的硬映射分解为"图→SDF→轮廓→顶点"的软映射，利用SDF的欧氏距离特性天然稳定了扩散过程。与PolygonGNN的差异在于：不再依赖GNN的消息传递提取图特征，而是通过交叉注意力让扩散U-Net在每个空间位置直接"查询"图结构信息。

## 整体框架


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/4005c6d0-50ed-486c-bd26-491c0161e58e/figures/Figure_2.png)
*Figure 2 (architecture): VisDiff architecture. The model consists of three main components: the CNN SDF Diffusion Block, the Transformer Visibility Extraction Block, and the SDF to Polygon Conversion.*



VisDiff采用严格的两阶段流水线，输入为可见性图G，输出为重建的多边形顶点坐标。数据流如下：

**阶段一：SDF扩散生成**
- **输入**：噪声SDF X_T ~ N(0, σ_max²I)，以及可见性图G的邻接结构
- **SDF Diffusion Block**：时间条件U-Net（下采样32→64→128，瓶颈512，上采样128→64→32→1），在每次上下采样块后插入**Spatial Transformer交叉注意力层**
- **交叉注意力机制**：Query来自U-Net当前层的空间CNN特征；Key和Value来自GNN编码的可见性图G，实现图结构到几何场的条件注入
- **噪声调度**：log-linear调度，σ_min=0.005, σ_max=10，共T个时间步
- **输出**：去噪后的干净SDF X_0

**阶段二：顶点提取精炼**
- **轮廓提取**：从预测的SDF X_0中提取零等值线，得到初始顶点集合P_init
- **SDF编码模块**：U-Net编码器（64→128→256→512）产生像素对齐特征Z_pix；瓶颈5×5×512展平为25×512并加位置编码，得到全局特征Z_global
- **Vertex Prediction Block**：3层Transformer编码器（256 hidden units），输入为[Z_pix, Z_global, P_init]的拼接，MLP（256→2）输出最终顶点坐标(x,y)

训练时，两阶段分别监督：SDF扩散用ground-truth SDF，顶点预测用ground-truth polygon；测试时仅输入可见性图G，无需任何几何监督。

```
可见性图 G ──→[GNN编码]──┐
                         ↓
噪声SDF X_T ──→[U-Net + Spatial Transformer]──→ 干净SDF X_0 ──→[轮廓提取]──→ P_init
      ↑                    (交叉注意力: Q=空间特征, K/V=图编码)              ↓
   时间步 t                                                              [SDF编码器]
                                                                         ↓
                                                              [Z_pix, Z_global, P_init]
                                                                         ↓
                                                              [Transformer + MLP]──→ P_final
```

## 核心模块与公式推导

### 模块 1: 条件SDF扩散（阶段一，对应框架图左侧）

**直觉**：标准扩散在像素/体素空间操作，但多边形生成需要图结构条件；将可见性图通过交叉注意力注入，使SDF场的每个空间位置"感知"到哪些顶点应该相互可见。

**Baseline 公式** (标准无条件扩散 / Diffusion-SDF):
$$p(X_{t-1} | X_t) = \mathcal{N}(X_{t-1}; \mu_\theta(X_t, t), \Sigma_\theta(X_t, t))$$
符号: $X_t$ = 时间步t的噪声SDF, $\mu_\theta, \Sigma_\theta$ = U-Net参数化的均值方差, $t$ = 时间步。

**变化点**：Diffusion-SDF虽在SDF上做扩散，但条件机制为简单类别嵌入或图像特征；VisDiff需要处理任意图结构的可见性信息，且要求空间精确的条件注入。

**本文公式（推导）**:
$$\text{Step 1}: \quad X_T \sim \mathcal{N}(0, \sigma_{\max}^2 I), \quad \sigma_{\max}=10 \quad \text{(最大噪声初始化)}$$
$$\text{Step 2}: \quad \sigma_t = \exp\left(\log\sigma_{\min} + \frac{t}{T}(\log\sigma_{\max} - \log\sigma_{\min})\right), \quad \sigma_{\min}=0.005 \quad \text{(log-linear噪声调度)}$$
$$\text{Step 3}: \quad p(X_{t-1} | X_t, G) = \mathcal{N}(X_{t-1}; \mu_\theta(X_t, t, G), \Sigma_\theta(X_t, t, G)) \quad \text{(加入图条件G)}$$
$$\text{Step 4}: \quad \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V, \quad Q = f_{\text{CNN}}(X_t), \quad K, V = g_{\text{GNN}}(G) \quad \text{(空间-图交叉注意力)}$$
$$\text{最终}: \quad \mathcal{L}_{\text{SDF}} = \mathbb{E}_{t, X_0, \epsilon}\left[\| \epsilon - \epsilon_\theta(X_t, t, G) \|^2\right]$$

**对应消融**：Table 11显示，若将Spatial Transformer替换为简单特征拼接，SDF重建质量下降，进而导致下游顶点F1降低（见两阶段训练对比）。

---

### 模块 2: SDF编码与顶点精炼（阶段二，对应框架图右侧）

**直觉**：直接从SDF零等值线提取的顶点有抖动和拓扑错误，需要数据驱动的精炼；同时SDF内部包含丰富的几何上下文（内外区域、曲率信息），应充分利用而非仅用边界。

**Baseline 公式** (Marching Cubes / 直接顶点预测):
$$P_{\text{final}} = \text{ExtractContour}(X_0) \quad \text{或} \quad P_{\text{final}} = \text{MLP}(\text{TransformerEncoder}(G))$$
符号: $P$ = 顶点坐标集合, $G$ = 可见性图, $X_0$ = 预测SDF。

**变化点**：Marching Cubes无学习能力，对SDF噪声敏感；GNN直接预测忽略了几何场信息；VisDiff将两者结合，用SDF提供初始化，用Transformer进行上下文感知精炼。

**本文公式（推导）**:
$$\text{Step 1}: \quad P_{\text{init}} = \text{ExtractContour}(X_0) \quad \text{(从SDF零等值线提取初始顶点，解决"从哪里开始"的问题)}$$
$$\text{Step 2}: \quad Z_{\text{global}} \in \mathbb{R}^{5 \times 5 \times 512} \rightarrow \mathbb{R}^{25 \times 512} + \text{PosEmb} \quad \text{(全局特征展平并加位置编码，保留空间顺序)}$$
$$\text{Step 3}: \quad Z_{\text{pix}} = \text{SDFEncoder}_{\text{local}}(X_0) \quad \text{(像素对齐局部特征，提供细粒度几何上下文)}$$
$$\text{Step 4}: \quad H = \text{TransformerEncoder}([Z_{\text{pix}}, Z_{\text{global}}, P_{\text{init}}]), \quad 3\text{层}, 256\text{ hidden units} \quad \text{(融合三类信息)}$$
$$\text{最终}: \quad P_{\text{final}} = \text{MLP}_{256 \rightarrow 2}(H), \quad \mathcal{L}_{\text{vertex}} = \|P_{\text{gt}} - P_{\text{pred}}\|_2^2$$

**对应消融**：Table 11显示，去掉SDF初始化（即联合训练无初始化）后F1从0.912降至0.850，Δ=-0.062；Table 2（或相关消融）显示仅使用global patch特征或仅使用pixel-aligned局部特征均导致Accuracy/Precision/Recall下降，证明两类特征的互补必要性。

---

### 模块 3: 两阶段联合监督（训练策略）

**直觉**：SDF扩散和顶点预测有不同的优化动态和空间尺度，联合训练会导致梯度冲突；先稳定SDF生成再精炼顶点，类似课程学习。

**Baseline 公式** (联合训练变体):
$$\mathcal{L}_{\text{joint}} = \mathcal{L}_{\text{SDF}} + \mathcal{L}_{\text{vertex}} \quad \text{(同时优化，无显式初始化依赖)}$$

**变化点**：联合训练中顶点预测分支的梯度会干扰SDF扩散的早期学习，且顶点预测缺乏良好的几何先验初始化。

**本文公式（推导）**:
$$\text{Step 1}: \quad \min_\theta \mathcal{L}_{\text{SDF}}^{(1)} \quad \text{训练60 epoch, Adam, lr}=10^{-4}, \text{batch}=128 \quad \text{(先训练SDF扩散至收敛)}$$
$$\text{Step 2}: \quad \min_\phi \mathcal{L}_{\text{vertex}}^{(2)}, \quad P_{\text{init}} = \text{ExtractContour}(X_0^{(\theta^*)}) \quad \text{(冻结SDF扩散，用其输出初始化训练顶点预测)}$$
$$\text{最终}: \quad \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{SDF}}(X_0, \hat{X}_0) + \mathcal{L}_{\text{vertex}}(P_{\text{gt}}, P_{\text{pred}}) \quad \text{(训练阶段同时监督，测试阶段仅输入G)}$$

**对应消融**：Table 11明确量化，两阶段训练（0.912）vs 联合训练（0.850），ΔF1 = +0.062（+7.3%相对提升）。

## 实验与分析


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/4005c6d0-50ed-486c-bd26-491c0161e58e/figures/Table_1.png)
*Table 1 (comparison): Model comparison to Vector Diffusion. The Conditional VAE is cVAE. We diff. w.r.t. chamfer distance, L2, Accuracy and F1-Score.*




![Table 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/4005c6d0-50ed-486c-bd26-491c0161e58e/figures/Table_2.png)
*Table 2 (quantitative): Coverage metrics for different hyperparameters. Higher coverage indicates broader exploration.*



本文在自整理的可见性图重建数据集上评估VisDiff，涵盖三个核心任务：Visibility Reconstruction（从有效可见性图重建多边形）、Visibility Recognition（从可能无效的图识别/重建）、以及Triangulation Graph Generation（三角剖分图生成）。主要对比指标包括Chamfer Distance、L2距离、Accuracy和F1-Score。

**核心定量结果**：在顶点预测任务上，VisDiff达到**F1 Score 0.912**，相比传统Marching Cubes算法（0.800）提升**+0.112（+14.0%相对）**。这一增益直接验证了SDF中间表示的有效性：Marching Cubes虽能从理想SDF提取精确轮廓，但对扩散模型预测的有噪声SDF极为敏感；VisDiff通过数据驱动的顶点精炼补偿了SDF预测误差。在训练策略对比上，两阶段训练（SDF扩散预训练+顶点初始化精炼）相比端到端联合训练（无初始化）的F1从0.850提升至0.912，**+0.062的绝对增益**证明了几何初始化对后续精炼的关键作用。



**消融分析**：除训练策略外，作者还检验了SDF编码器特征组合的影响。仅使用global patch-based特征或仅使用pixel-aligned局部特征均导致Accuracy/Precision/Recall下降（具体数值未在提供的片段中完整展示），说明全局上下文与局部几何细节的互补必要性。Table 2展示了不同超参数下的Coverage指标，更高覆盖率表明VisDiff能够进行广泛的多样性探索，支持polygon-to-polygon和graph-to-graph插值等应用。


![Figure 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/4005c6d0-50ed-486c-bd26-491c0161e58e/figures/Figure_4.png)
*Figure 4 (result): We provide qualitative guarantees results of VisDiff on Visibility Recognition problem.*



**公平性检验**：当前实验存在若干局限。首先，Table 1和Table 6的标题虽提及Vertex-Diffusion、Conditional-VAE、GNN、MeshAnything等多个baseline，但提供的片段中仅Marching Cubes和联合训练变体有明确数值对比，PolyDiffuse、PolyDiff、PolygonGNN等最直接相关工作的定量结果缺失，证据强度受限。其次，评估主要集中于F1-Score单一指标，Chamfer Distance和L2的完整对比表未完全展示。第三，训练成本为单张NVIDIA A100 GPU约16小时，属于中等计算预算，但模型参数量和推理延迟未披露。作者也坦承，Visibility Recognition的初步结果仅针对"更难"的非保证有效图，完整benchmark覆盖仍需扩展。

## 方法谱系与知识库定位

**方法家族**：扩散模型 + 几何深度学习（Diffusion-based Geometric Learning）

**父方法**：Diffusion-SDF（[18]）—— 核心方法来源，VisDiff在其SDF扩散基础上增加了可见性图条件机制和两阶段顶点提取，将通用3D形状生成适配到2D多边形重建任务。

**直接Baseline差异**：
- **PolyDiffuse** [19]：同输出类型（多边形）同范式（扩散），但直接对顶点集合做guided set diffusion，无SDF中间表示 → VisDiff以SDF解耦组合-几何映射
- **PolyDiff** [23]：同为多边形扩散，直接生成3D polygonal mesh → VisDiff专注2D可见性图条件，引入SDF几何先验
- **PolygonGNN** [25]：同为可见性图输入，但用GNN嵌入+解码 → VisDiff替换为扩散+交叉注意力，生成质量更高
- **Marching Cubes**：传统几何算法，无学习能力 → VisDiff在其后增加数据驱动精炼模块

**改动槽位**：architecture（U-Net+Spatial Transformer替换GNN/direct diffusion）、training_recipe（两阶段训练替换联合训练）、data_curation（新增SDF标注数据集）、inference（SDF→轮廓→顶点的三阶推理链）

**后续方向**：(1) 扩展到带孔洞的多边形（polygons with holes）及更一般的可见性图类；(2) 将SDF-顶点两阶段范式迁移到其他组合几何重建问题（如平面剖分、建筑布局）；(3) 结合神经辐射场（NeRF）扩展到3D可见性重建。

**标签**：modality: graph→image | paradigm: diffusion model | scenario: geometric reconstruction | mechanism: cross-attention conditioning | constraint: two-stage supervised training

