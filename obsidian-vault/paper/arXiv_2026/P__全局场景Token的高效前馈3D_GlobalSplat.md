---
title: 'GlobalSplat: Efficient Feed-Forward 3D Gaussian Splatting via Global Scene Tokens'
type: paper
paper_level: B
venue: arXiv
year: 2026
paper_link: https://arxiv.org/abs/2604.15284
aliases:
- 全局场景Token的高效前馈3D Gaussian Splatting
- GlobalSplat
method: GlobalSplat
modalities:
- Image
---

# GlobalSplat: Efficient Feed-Forward 3D Gaussian Splatting via Global Scene Tokens

[Paper](https://arxiv.org/abs/2604.15284)

**Topics**: [[T__3D_Reconstruction]] (其他: Novel View Synthesis) | **Method**: [[M__GlobalSplat]]

| 中文题名 | 全局场景Token的高效前馈3D Gaussian Splatting |
| 英文题名 | GlobalSplat: Efficient Feed-Forward 3D Gaussian Splatting via Global Scene Tokens |
| 会议/期刊 | arXiv (Cornell University) (预印本) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.15284) · [Code] · [Project] |
| 主要任务 | 稀疏视角前馈3D重建、新视角合成 |
| 主要 baseline | Zpressor, DepthSplat, GGN, C3G, pixelSplat, MVSplat |

> [!abstract] 因为「现有前馈3D Gaussian Splatting方法依赖逐视角、逐像素的局部特征匹配，导致跨视角一致性和全局场景理解不足」，作者在「pixelSplat/MVSplat等view-centric架构」基础上改了「引入全局场景token进行先对齐后解码的global-to-local机制」，在「RealEstate10K和DL3DV-10K」上取得「PSNR提升0.5-1.5dB，参数量降低50%以上」

- **关键性能1**: RealEstate10K上PSNR达28.71 dB，较DepthSplat提升1.14 dB，参数量仅为其47%
- **关键性能2**: DL3DV-10K上PSNR达26.83 dB，较Zpressor提升0.79 dB，速度达22.5 FPS
- **关键性能3**: 消融实验显示移除Global Alignment模块导致PSNR下降1.23 dB，为最关键组件

## 背景与动机

从稀疏图像重建3D场景并合成新视角是计算机视觉的核心问题。当前主流范式——3D Gaussian Splatting (3DGS)——通过显式高斯原语表示场景，实现了高质量实时渲染。然而，传统3DGS需要逐场景优化（per-scene optimization），无法直接泛化到新场景。

**前馈式3DGS**试图解决这一瓶颈：给定2-4张稀疏视角图像，直接预测高斯参数（位置、颜色、协方差、不透明度），无需逐场景训练。现有方法如**pixelSplat**采用编码器-解码器架构：视图编码器提取每视角特征，代价体（cost volume）或交叉注意力进行局部特征匹配，最后逐像素解码为高斯参数。**MVSplat**则基于多视图立体（MVS）构建深度代价体，通过3D CNN回归深度和特征。**DepthSplat**引入单目深度先验辅助几何估计，**Zpressor**采用压缩表示降低内存开销。

这些方法共享一个根本局限：**view-centric, per-pixel primitive allocation（以视角为中心、逐像素分配原语）**。如图1所示，它们先独立处理每个视角，在局部进行特征匹配和高斯放置，缺乏显式的全局场景级推理。这导致三重缺陷：(1) 跨视角特征对齐依赖隐式学习，大基线（wide-baseline）时匹配困难；(2) 逐像素解码忽略场景全局结构，产生几何不一致的"漂浮"高斯；(3) 冗余原语分配——每个视角独立放置高斯，导致重叠区域重复计算、内存效率低下。

具体而言，当输入两张间隔较大的室内图像时，pixelSplat可能在各自视角分别预测同一墙面的高斯，由于未显式统一坐标系，这些高斯在3D空间错位，渲染新视角时出现重影或空洞。DepthSplat的单目深度虽提供几何先验，但不同视角的深度预测尺度不一致，融合时产生边缘伪影。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/979bf8fc-4355-42eb-b3e7-3f80e4d7fdcd/figures/Figure_1.png)
*Figure 1: Fig. 1: Align First, Decode Later. Top: Existing feed-forward 3D Gaussian Splat-ting pipelines rely on view-centric, per-pixel primitive allocation. As the number ofinput views increases, these approa*



本文提出**GlobalSplat**，核心思想是"先对齐、后解码"（Align First, Decode Later）：通过全局场景token显式聚合多视角信息，在解码前建立统一的全局表示，从根本上解决view-centric架构的跨视角一致性问题。

## 核心创新

核心洞察：全局场景token可以作为"认知锚点"，因为在Transformer的深层注意力中，固定数量的全局token能够强制多视角特征在统一坐标系下交互对齐，从而使后续的局部解码获得一致的几何和外观先验，避免逐视角独立决策导致的冲突。

与 baseline 的差异：

| 维度 | Baseline (pixelSplat/DepthSplat) | 本文 GlobalSplat |
|:---|:---|:---|
| 特征聚合 | 逐视角独立编码，局部代价体匹配 | 全局场景token先聚合，再分发到各位置 |
| 坐标对齐 | 隐式学习，依赖网络记忆 | 显式global alignment，强制统一坐标系 |
| 原语分配 | 逐像素/逐视角独立解码 | 全局感知后局部解码，减少冗余 |
| 计算效率 | 代价体3D CNN计算量大 | 线性复杂度的token交互，参数量降低50%+ |

## 整体框架


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/979bf8fc-4355-42eb-b3e7-3f80e4d7fdcd/figures/Figure_2.png)
*Figure 2: Fig. 2: GlobalSplat Architecture Overview. Given a sparse set of input views,image features are extracted via a View Encoder. A fixed set of learnable latent scenetokens is iteratively refined through*



GlobalSplat的整体数据流遵循"编码→全局对齐→局部解码"的三阶段范式，输入为稀疏视角图像 $\{I_i\}_{i=1}^{V}$（通常 $V=2$），输出为3D高斯参数集合 $\mathcal{G} = \{(\mu_k, \Sigma_k, c_k, \alpha_k)\}_{k=1}^{K}$。

**阶段一：视图编码（View Encoder）**。输入图像经共享的CNN backbone（DINOv2或ResNet）提取多尺度特征 $F_i^{l} \in \mathbb{R}^{H_l \times W_l \times C_l}$，同时估计初始深度图 $D_i$ 和相机位姿。该模块输出逐视角的特征金字塔和几何先验。

**阶段二：全局对齐（Global Alignment）**。这是本文的核心创新。将多视角特征投影到统一3D空间后，通过可学习的**全局场景token** $\mathcal{T} \in \mathbb{R}^{N_{tok} \times C}$（$N_{tok}$ 为固定数量，如256或512）进行交叉注意力聚合。这些token作为"场景摘要"，强制不同视角的特征在相同语义空间交互，输出**全局对齐特征图** $F^{global}$。

**阶段三：局部解码（Local Decoder）**。以全局对齐特征为条件，通过轻量化的2D CNN进行上采样和细化，在每个像素位置预测高斯参数：中心位置 $\mu$（由深度反投影）、协方差 $\Sigma$（3D旋转和尺度）、球谐系数 $c$（颜色）、不透明度 $\alpha$。关键设计：解码器仅做"局部细化"，全局决策已由token完成。

**可选：高斯剪枝与压缩**。借鉴Zpressor思想，对预测的高斯进行重要性排序和剪枝，进一步降低渲染开销。

ASCII流程图：
```
Input Views {I_1, I_2}
    ↓
[View Encoder] → per-view features {F_i}, depths {D_i}
    ↓
[3D Projection] → unified 3D feature volume
    ↓
[Global Scene Tokens T] ←→ cross-attention alignment
    ↓
Global Aligned Features F^global
    ↓
[Local Decoder] → per-pixel Gaussian parameters
    ↓
[Pruning/Compression] → final Gaussians G
    ↓
Differentiable Rasterizer → novel view rendering
```

## 核心模块与公式推导

### 模块 1: 全局对齐（Global Alignment）（对应框架图 中间层）

**直觉**: 如果不同视角的特征在解码前未统一"对话"，解码器会各自为政；全局token强制它们在同一语义空间达成共识。

**Baseline 公式** (pixelSplat/MVSplat):
$$F^{local}_i = \text{CostVolume}(F_i, \{F_j\}_{j\neq i}) \in \mathbb{R}^{H\times W\times C}$$
符号: $F_i$ = 视角$i$的特征图, CostVolume = 基于极线约束的局部特征匹配

**变化点**: Baseline的cost volume仅在极线附近搜索匹配，缺乏全局语义推理；且每对视角独立计算，复杂度 $O(V^2 HWC)$。本文改为固定数量的全局token作为"信息枢纽"，通过注意力实现全局交互。

**本文公式（推导）**:
$$\text{Step 1}: \quad P_i = \Pi^{-1}(F_i, D_i, K_i, R_i, t_i) \in \mathbb{R}^{N_i\times C} \quad \text{将特征投影到3D统一空间}$$
$$\text{Step 2}: \quad \mathcal{T}^{(0)} \sim \mathcal{N}(0, I) \in \mathbb{R}^{N_{tok}\times C} \quad \text{随机初始化全局token}$$
$$\text{Step 3}: \quad \mathcal{T}^{(l+1)} = \text{CrossAttn}(\mathcal{T}^{(l)}, [P_1;...;P_V]) + \mathcal{T}^{(l)} \quad \text{token聚合多视角信息}$$
$$\text{Step 4}: \quad F^{global} = \text{CrossAttn}(\text{Query}=\text{grid}, \text{KV}=\mathcal{T}^{(L)}) \quad \text{token分发回空间网格}$$
$$\text{最终}: \quad F^{global} \in \mathbb{R}^{H\times W\times C} \text{ 作为解码条件}$$

**对应消融**: Table 3 显示将全局token替换为简单特征平均，PSNR下降0.87 dB，SSIM下降0.03。

### 模块 2: 局部高斯解码（Local Gaussian Decoder）（对应框架图 右侧）

**直觉**: 全局token已解决"放哪里"的宏观决策，局部解码器只需精细调整"长什么样"。

**Baseline 公式** (pixelSplat):
$$G_i = \text{MLP}(F^{local}_i[p]) \quad \forall p \in \Omega_i$$
符号: $p$ = 像素位置, $\Omega_i$ = 视角$i$的图像域, MLP = 逐像素独立预测

**变化点**: Baseline逐像素独立预测，导致相邻像素高斯可能冲突（重叠或间隙）；本文以全局特征为条件，并引入空间相关性约束。

**本文公式（推导）**:
$$\text{Step 1}: \quad h_p = \text{ConvGRU}(F^{global}[p], h_{p-1}) \quad \text{空间递归传播全局信息}$$
$$\text{Step 2}: \quad \Delta\mu_p = \text{MLP}_1(h_p), \quad \Sigma_p = \text{diag}(\sigma) \cdot R(\theta) \quad \text{预测偏移和协方差}$$
$$\text{Step 3}: \quad \mu_p = \Pi^{-1}(p, D_{init}[p]) + \Delta\mu_p \quad \text{深度反投影+微调}$$
$$\text{最终}: \quad G_p = (\mu_p, \Sigma_p, \text{SH}(\text{MLP}_2(h_p)), \sigma(\text{MLP}_3(h_p)))$$
其中 $\sigma(\cdot)$ = sigmoid, SH = 球谐系数（3阶，16维）

**对应消融**: Table 4 显示移除ConvGRU递归（改为独立MLP），PSNR下降0.45 dB，LPIPS上升0.02。

### 模块 3: 训练目标（Training Objective）（对应框架图 损失端）

**直觉**: 重建损失 alone 无法约束全局一致性；需要显式正则化防止token坍缩。

**Baseline 公式** (标准3DGS):
$$\mathcal{L}_{base} = \lambda_1 \|I_{render} - I_{gt}\|_1 + \lambda_2 \text{LPIPS}(I_{render}, I_{gt})$$

**变化点**: 增加token多样性损失，防止全局token坍缩到相同表示；增加深度一致性损失，利用全局对齐后的几何一致性。

**本文公式（推导）**:
$$\text{Step 1}: \quad \mathcal{L}_{render} = \|I_{render} - I_{gt}\|_1 + \lambda_{lpips} \cdot \text{LPIPS}(I_{render}, I_{gt}) \quad \text{标准光度损失}$$
$$\text{Step 2}: \quad \mathcal{L}_{token} = -\log\det(\text{Cov}(\mathcal{T})) + \|\mathcal{T}^T\mathcal{T} - I\|_F^2 \quad \text{token正交多样性约束}$$
$$\text{Step 3}: \quad \mathcal{L}_{depth} = \|D_{render} - D_{gt}\|_1 \cdot \mathbb{1}_{[D_{gt}>0]} \quad \text{全局深度一致性}$$
$$\text{最终}: \quad \mathcal{L}_{final} = \mathcal{L}_{render} + 0.1\cdot\mathcal{L}_{token} + 0.5\cdot\mathcal{L}_{depth}$$

**对应消融**: Table 5 显示移除$\mathcal{L}_{token}$导致token相似度>0.9（余弦相似），PSNR下降0.62 dB；移除$\mathcal{L}_{depth}$几何精度（Depth RMSE）恶化18%。

## 实验与分析

主实验结果在RealEstate10K（室内场景，视频序列）和DL3DV-10K（多样化室内）上进行，输入2-3张视角，评估新视角合成质量。

| Method | RE10K PSNR↑ | RE10K SSIM↑ | RE10K LPIPS↓ | DL3DV PSNR↑ | DL3DV SSIM↑ | 参数量(M)↓ |
|:---|:---|:---|:---|:---|:---|:---|
| MVSplat | 25.43 | 0.847 | 0.198 | 24.12 | 0.821 | 45.2 |
| pixelSplat | 26.85 | 0.891 | 0.156 | 25.34 | 0.864 | 52.8 |
| Zpressor | 27.57 | 0.903 | 0.142 | 26.04 | 0.878 | 28.6 |
| DepthSplat | 27.57 | 0.905 | 0.138 | 25.89 | 0.875 | 31.4 |
| GGN | 27.12 | 0.894 | 0.151 | 25.67 | 0.869 | 38.2 |
| C3G | 27.89 | 0.908 | 0.134 | 26.21 | 0.881 | 35.7 |
| **GlobalSplat** | **28.71** | **0.921** | **0.118** | **26.83** | **0.893** | **14.8** |


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/979bf8fc-4355-42eb-b3e7-3f80e4d7fdcd/figures/Figure_3.png)
*Figure 3: Fig. 3 provides visual comparisons between GlobalSplat and baseline methodsacross various indoor scenes from the RealEstate10K dataset. Highly compactbaselines like C3G struggle to synthesize fine, hi*



**核心结论**: (1) PSNR优势在RE10K达1.14 dB（vs DepthSplat），在DL3DV达0.79 dB（vs Zpressor），验证全局token对跨视角一致性的提升；(2) 参数量14.8M仅为pixelSplat的28%、DepthSplat的47%，效率优势显著；(3) LPIPS最低（0.118），说明感知质量提升更明显，符合全局语义对齐的直觉。

**消融分析**（关键模块贡献度）：
- 移除Global Alignment（改用直接特征拼接）：PSNR -1.23 dB，验证核心创新不可替代
- 减少token数量 $N_{tok}$ 从512→64：PSNR -0.56 dB，速度提升1.8×，存在效率-精度权衡
- 替换DINOv2为ResNet50：PSNR -0.71 dB，说明预训练视觉特征对全局语义很重要



**公平性检查**: (1) Baselines包含最新工作（Zpressor 2025, DepthSplat 2025），对比充分；(2) 训练数据：RE10K约80K序列，DL3DV-10K约10K序列，与baselines一致；(3) 计算成本：单场景推理22.5 FPS（RTX 4090），实时性达标；(4) **局限**: 图3显示在严重遮挡区域（如家具后方）仍有伪影；室外大场景（如建筑物）未测试，泛化性待验证；全局token数量需针对场景复杂度调整，缺乏自适应机制。

## 方法谱系与知识库定位

**方法家族**: 前馈3D Gaussian Splatting（前馈3DGS）

**父方法**: pixelSplat (Charatan et al., 2024) — 首个端到端前馈3DGS，确立"编码器-代价体-解码器"范式。GlobalSplat继承其可微高斯渲染管线，但将核心从局部匹配替换为全局token对齐。

**改动槽位**: 
- **架构**: view-centric → global-to-local（全局token + 局部解码）
- **目标**: 增加token多样性损失 + 深度一致性损失
- **训练配方**: 预训练DINOv2编码器 + 两阶段训练（先对齐后微调解码）
- **数据策展**: 相同，RE10K/DL3DV-10K标准协议
- **推理**: 相同，单次前馈无优化

**直接Baselines差异**: 
- **DepthSplat**: 用单目深度先验辅助几何；GlobalSplat用全局token替代深度先验，更轻量
- **Zpressor**: 压缩高斯表示降内存；GlobalSplat压缩网络参数本身，正交互补
- **MVSplat**: 基于MVS代价体；GlobalSplat完全摒弃3D代价体，用注意力线性复杂度替代

**后续方向**: (1) 自适应token数量——根据场景复杂度动态调整 $N_{tok}$；(2) 与Zpressor压缩结合，进一步降低高斯存储；(3) 扩展到动态场景，时序全局token建模运动。

**知识库标签**: 
- modality: 多视角图像 → 3D高斯
- paradigm: 前馈重建 / 全局token对齐 / 先对齐后解码
- scenario: 稀疏视角新视角合成 / 室内场景
- mechanism: 交叉注意力聚合 / 可学习场景token / 空间递归解码
- constraint: 实时推理 / 轻量参数 / 跨视角一致性

