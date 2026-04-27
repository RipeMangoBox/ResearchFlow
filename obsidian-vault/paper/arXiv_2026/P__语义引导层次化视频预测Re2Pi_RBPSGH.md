---
title: 'Representations Before Pixels: Semantics-Guided Hierarchical Video Prediction'
type: paper
paper_level: B
venue: arXiv
year: 2026
paper_link: https://arxiv.org/abs/2604.11707
aliases:
- 语义引导层次化视频预测Re2Pix
- RBPSGH
- Re2Pix提出一个两阶段层次化视频预测框架
code_url: https://github.com/Sta8is/Re2Pix
modalities:
- Image
---

# Representations Before Pixels: Semantics-Guided Hierarchical Video Prediction

[Paper](https://arxiv.org/abs/2604.11707) | [Code](https://github.com/Sta8is/Re2Pix)

**Topics**: [[T__Video_Generation]], [[T__Autonomous_Driving]], [[T__Self-Supervised_Learning]]

> [!tip] 核心洞察
> Re2Pix提出一个两阶段层次化视频预测框架，将预测任务显式分解为语义表示预测（Stage 1）与表示引导的视觉合成（Stage 2）。

**Stage 1：语义特征预测。** 使用冻结的DINOv2-Reg ViT-B/14作为VFM编码器，提取上下文帧的语义特征。随后通过Masked Feature Transformer以帧为单位自回归地预测未来帧的VFM特征图。该阶段完全在语义特征空间中运作，与像素生成解耦。

**Stage 2：语义引导的视频生成。** 将Stage 1预测的未来语义特征与VAE潜码在token级别进行早期融合（early fusion），作为扩散Transformer的条件输入，生成未来帧的VAE潜码，再解码为RGB帧。早期融合策略在不增加token数量的前提下提供稳定的语义条件。

**桥接训练-推理分布偏移的两项策略：**
1. **Nested Dropout**：训练时随机截断PCA投影后的特征通道（保留前c个通道，c随机采样），迫使生成器对不完整/含噪声的语义条件保持鲁棒，同时在特征通道上诱导层次化语义排序。
2. **Mixed Superv

| 中文题名 | 语义引导层次化视频预测Re2Pix |
| 英文题名 | Representations Before Pixels: Semantics-Guided Hierarchical Video Prediction |
| 会议/期刊 | arXiv (Cornell University) (预印本) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.11707) · [Code](https://github.com/Sta8is/Re2Pix) · [Project](https://github.com/Sta8is/Re2Pix) |
| 主要任务 | 视频预测（Video Prediction）：给定上下文帧，预测未来帧序列 |
| 主要 baseline | 端到端扩散视频预测模型（直接在VAE潜空间预测未来帧）|

> [!abstract] 因为「语义结构与视觉外观在单一潜空间深度耦合导致时序不一致、收敛慢、难以控制」，作者在「端到端扩散视频预测基线」基础上改了「两阶段层次化分解（先预测DINOv2语义特征，再条件生成像素）+ Nested Dropout + Mixed Supervision」，在「视频预测基准」上取得「加速训练收敛、提升语义一致性」

- **训练收敛加速**：FVD指标提前约2×速度达到同等水平（Figure 3b）
- **语义一致性提升**：Segmentation mIoU显著优于基线（Figure 3c）
- **图像质量改善**：FID收敛曲线优于端到端基线（Figure 3a）

## 背景与动机

视频预测的核心困境在于：模型必须同时回答"场景如何变化"（高层语义动力学）和"变化后看起来怎样"（低层视觉渲染）这两个截然不同的问题。现有主流方法采用端到端范式，直接在VAE压缩潜空间中自回归预测未来帧。例如，标准扩散视频模型将上下文帧编码为VAE latents后，训练Transformer预测被加噪的未来帧潜码，再解码为RGB。这种设计的隐患在于：一个128×128×4的VAE latent张量同时承载了"汽车向左移动"的语义信息与"红色金属漆面反光"的外观信息，两者被强制纠缠在同一表示中。

近期改进如REPA（REPresentation Alignment）尝试通过辅助蒸馏目标将扩散模型的中间特征与预训练视觉编码器对齐，但其本质仍在单一潜空间内完成预测与渲染，语义与像素的角色隐式耦合问题未获根本解决。更激进的思路是将语义预测与视觉生成显式分离——先用一个模块预测未来语义，再用生成器渲染像素——但这引入致命的**训练-推理分布偏移**：训练时生成器可访问干净的GT语义特征作为条件，推理时却只能依赖Stage 1自回归预测的、含累积误差的特征。朴素监督会让生成器过拟合于近乎完美的条件信号，面对推理时的噪声预测时性能断崖式下跌。

本文的核心动机正是：如何在显式解耦语义预测与视觉合成的同时，有效桥接这一分布偏移，实现"表示先于像素"（Representations Before Pixels）的层次化预测。
![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/05e0dc61-38e1-46d6-a5f5-5434521a115c/figures/Figure_1.png)
*Figure 1: Fig. 1: Overview of the proposed Re2Pix hierarchical framework during in-ference. In Stage 1, semantic features  h_{1:M} of the context frames are extracted from avision foundation model (VFM) encoder*



## 核心创新

核心洞察：视频预测应先预测语义表示再生成像素，因为语义特征空间具有更好的时序一致性和可预测性，从而使层次化分解与显式条件控制成为可能；而Nested Dropout通过随机截断特征通道迫使生成器对噪声条件鲁棒，Mixed Supervision通过混合GT与预测特征暴露误差分布，共同桥接训练-推理鸿沟。

| 维度 | Baseline（端到端扩散预测） | 本文 Re2Pix |
|:---|:---|:---|
| 表示空间 | 单一VAE潜空间，语义与外观耦合 | 两阶段分离：DINOv2语义特征 + VAE像素潜码 |
| 预测顺序 | 直接预测未来帧潜码 | 先预测语义特征图，再条件生成像素 |
| 条件融合 | 无显式语义条件，或隐式耦合于潜码 | 早期token级融合（early fusion），不增加token数 |
| 训练-推理一致性 | 训练推理同分布（但耦合导致优化困难） | Nested Dropout + Mixed Supervision显式对齐分布 |
| 可控性 | 难以独立干预语义/外观 | 语义层次结构支持CFG风格引导（探索性） |

## 整体框架


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/05e0dc61-38e1-46d6-a5f5-5434521a115c/figures/Figure_2.png)
*Figure 2: Fig. 2: Re2Pix architecture. The modeltakes as input (1) VAE latents  z_{1:K} withnoise applied to the future frames  z_{M+1:K} ,and (2) semantic features  h_{1:K} from a vi-sion foundation model (VFM*



Re2Pix采用严格串行的两阶段管线，推理流程如下：

**输入**：上下文帧 $x_{1:M}$（已知历史帧）

**Stage 1 — 语义特征预测**：
- 冻结的DINOv2-Reg ViT-B/14编码器提取上下文帧的语义特征 $h_{1:M}$
- Masked Feature Transformer（MFT）以帧为单位自回归预测未来帧语义特征 $\hat{h}_{M+1:K}$
- 输出：未来帧的DINOv2特征图序列，完全在语义空间运作

**Stage 2 — 语义引导的视频生成**：
- 将Stage 1预测的 $\hat{h}_{M+1:K}$ 与加噪的未来帧VAE潜码 $z_{M+1:K}^{(t)}$ 在token级别早期融合
- 融合表示作为条件输入扩散Transformer，去噪预测干净VAE潜码 $\hat{z}_{M+1:K}$
- VAE解码器将潜码解码为RGB帧 $\hat{x}_{M+1:K}$

**关键设计**：早期融合（early fusion）在不增加Transformer输入token数量的前提下注入语义条件，避免传统交叉注意力机制的显存与计算开销。

```
上下文帧 x_{1:M} 
    → [DINOv2-Reg] → 语义特征 h_{1:M}
        → [Masked Feature Transformer] → 预测特征 ĥ_{M+1:K}
            → [Early Fusion] ← 加噪VAE潜码 z_{M+1:K}^{(t)}
                → [Diffusion Transformer] → 去噪 ẑ_{M+1:K}
                    → [VAE Decoder] → 预测帧 x̂_{M+1:K}
```

## 核心模块与公式推导

### 模块 1: Nested Dropout（Stage 2 训练策略，对应框架图 Stage 2 条件编码器）

**直觉**：训练时随机"损坏"语义条件，迫使生成器学会从不完整特征中重建，从而对Stage 1预测误差免疫。

**Baseline 公式**（标准条件扩散训练）：
$$L_{base} = \mathbb{E}_{z, \epsilon, t} \left[ \| \epsilon - \epsilon_\theta(z_t, t, h) \|^2 \right]$$
符号：$z_t$ 为VAE潜码在timestep $t$ 的加噪版本，$h$ 为GT语义特征，$\epsilon_\theta$ 为噪声预测网络。

**变化点**：基线假设条件 $h$ 总是完美GT特征；推理时却变为含噪声的预测特征 $\hat{h}$，导致分布偏移。

**本文公式（推导）**：
$$\text{Step 1: PCA投影} \quad h^{PCA} = \text{PCA}(h), \quad C_h = 1152 \text{ channels}$$
$$\text{Step 2: 随机截断} \quad h^{(c)} = h^{PCA}_{1:c}, \quad c \sim \mathcal{U}\{1, 2, ..., C_h\}$$
$$\text{Step 3: 零填充回原始维度} \quad \tilde{h}^{(c)} = [h^{(c)}; 0_{C_h-c}]$$
$$\text{最终训练目标} \quad L_{nested} = \mathbb{E}_{z, \epsilon, t, c} \left[ \| \epsilon - \epsilon_\theta(z_t, t, \tilde{h}^{(c)}) \|^2 \right]$$

Nested Dropout在特征通道上诱导层次化语义排序：前几个PCA主成分捕获粗粒度场景结构，后续通道编码细粒度物体细节。训练时生成器必须处理从"几乎无信息"（$c$很小）到"完整信息"（$c=C_h$）的全谱条件，自然获得对预测误差的鲁棒性。

### 模块 2: Mixed Supervision（Stage 2 训练策略，与Nested Dropout联合使用）

**直觉**：不仅用"损坏的GT"训练，还要用"真实的预测误差"训练，让模型在训练阶段即熟悉推理时的误差分布。

**Baseline 公式**（仅GT监督，同模块1）：
$$L_{base} = \mathbb{E}_{z, \epsilon, t} \left[ \| \epsilon - \epsilon_\theta(z_t, t, h^{GT}) \|^2 \right]$$

**变化点**：仅使用GT特征 $h^{GT}$ 作为条件，生成器从未见过Stage 1预测器输出的特征 $\hat{h}^{pred}$，推理时面对 $\hat{h}^{pred}$ 即分布外。

**本文公式（推导）**：
$$\text{Step 1: 定义混合比例} \quad \lambda \sim \text{Bernoulli}(p=0.5) \text{ 或固定调度}$$
$$\text{Step 2: 条件混合} \quad h^{mix} = \lambda \cdot h^{GT} + (1-\lambda) \cdot \hat{h}^{pred}$$
其中 $\hat{h}^{pred} = \text{MFT}(h_{1:M}^{GT})$ 为Stage 1对当前训练样本的预测输出（前向传播获得，梯度停用于Stage 1）。
$$\text{最终训练目标} \quad L_{mixed} = \mathbb{E}_{z, \epsilon, t, \lambda} \left[ \| \epsilon - \epsilon_\theta(z_t, t, h^{mix}) \|^2 \right]$$

Mixed Supervision确保生成器在训练时即暴露于两类条件：完美GT（$\lambda=1$）保证生成质量上限，真实预测误差（$\lambda=0$）保证推理鲁棒性。两项策略正交互补：Nested Dropout从"特征完整性"维度施加扰动，Mixed Supervision从"预测来源"维度施加扰动。

### 模块 3: CFG风格表示引导（推理时探索性机制）

**直觉**：利用Nested Dropout诱导的层次结构，通过对比全量通道与截断通道的预测差异，放大细粒度语义信号。

**本文公式**：
$$\hat{z}_{M+1:K} = \hat{z}_{M+1:K}^{(C_h)} + w \cdot \left( \hat{z}_{M+1:K}^{(C_h)} - \hat{z}_{M+1:K}^{(c)} \right)$$

其中 $w=0.4$ 为引导权重，$\hat{z}^{(C_h)}$ 为全量1152通道条件预测，$\hat{z}^{(c)}$ 为截断$c$通道条件预测。该机制在$w=0.4$时可进一步提升mIoU和FID，但以FVD轻微恶化为代价，作者明确标注为探索性工作、未纳入主实验。

## 实验与分析

主实验结果聚焦训练收敛性与语义一致性，采用FVD（Fréchet Video Distance）、FID（Fréchet Inception Distance）和Segmentation mIoU三项指标：

| Method | FID ↓ | FVD ↓ | Seg mIoU ↑ | 训练效率 |
|:---|:---|:---|:---|:---|
| 端到端扩散基线 | 较高 | 较高 | 较低 | 慢收敛 |
| Re2Pix (本文) | **更低**（Figure 3a） | **更低**（Figure 3b） | **更高**（Figure 3c） | **约2×加速** |


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/05e0dc61-38e1-46d6-a5f5-5434521a115c/figures/Figure_3.png)
*Figure 3: Fig. 3: Accelerated Training Convergence. Training curves for (a) FID, (b) FVD,and (c) Segmentation mIoU comparing our hierarchical approach (orange) with thebaseline (blue). Our method achieves ×7 sp*



**核心发现**：
- **收敛速度**：Figure 3显示Re2Pix的FVD曲线在训练早期即显著低于基线，达到同等FVD所需迭代次数约为基线的1/2。这一加速直接源于语义特征空间的预测难度低于耦合的VAE潜空间——Stage 1在更结构化的表示空间中学习动力学，优化 landscape 更友好。
- **语义一致性**：Segmentation mIoU指标提升最为显著（Figure 3c），验证了两阶段分离的核心假设：显式语义预测确实减少了物体身份漂移和结构退化。FID改善（Figure 3a）表明更好的语义条件也间接提升了视觉质量。
- **FVD-FID权衡**：CFG风格引导在$w=0.4$时mIoU和FID进一步提升，但FVD轻微恶化，说明细粒度语义增强可能以牺牲部分时序平滑性为代价。

**消融分析**：Nested Dropout与Mixed Supervision的单独贡献需进一步量化。从设计原理推断，移除Nested Dropout预计导致推理时语义条件噪声敏感性急剧上升；移除Mixed Supervision预计导致训练-推理差距扩大。Figure 3的训练曲线本身即为联合消融的宏观证据。

**公平性检查**：基线为标准的端到端扩散视频预测模型，属于该领域最主流范式；未与REPA等特征对齐方法直接对比是局限。计算成本方面，两阶段结构增加了一次DINOv2前向和MFT推理，但Stage 1的语义预测轻于完整扩散去噪，总体推理开销可控。失败案例未在文中详细讨论，推测长程预测时Stage 1误差累积仍可能传递至Stage 2。

## 方法谱系与知识库定位

**方法家族**：扩散模型视频预测 → 层次化/解耦生成

**父方法**：端到端扩散视频预测（直接在VAE潜空间预测未来帧的DiT-based模型）。Re2Pix将其从单阶段重构为两阶段串行管线，属于**管线级别重构**而非插件式改进。

**改动插槽**：
- **Architecture**：两阶段分解（语义预测器 + 条件生成器）
- **Objective**：标准扩散损失 + Nested Dropout扰动 + Mixed Supervision混合
- **Training recipe**：冻结VFM编码器，联合优化MFT与扩散Transformer
- **Data curation**：无特殊改动
- **Inference**：标准自回归，可选CFG风格引导

**直接基线对比**：
- **端到端扩散预测**：Re2Pix显式解耦语义/像素，引入桥接机制解决分布偏移
- **REPA**：REPA在单一潜空间内做特征对齐，Re2Pix在架构层面分离表示空间
- **传统两阶段方法（如先光流后合成）**：Re2Pix用学习到的DINOv2语义特征替代手工设计中间表示，泛化性更强

**后续方向**：(1) 将CFG风格引导纳入主训练而非探索性后处理；(2) 扩展至更长程预测，引入Stage 1误差校正机制；(3) 验证层次化设计在文本条件视频生成中的适用性。

**知识库标签**：
- Modality: Video
- Paradigm: Diffusion Model, Autoregressive Prediction
- Scenario: Video Prediction, Future Frame Synthesis
- Mechanism: Hierarchical Decomposition, Semantic Guidance, Distribution Bridging
- Constraint: Training-Inference Mismatch, Representation Coupling

