---
title: 'MMCORE: MultiModal COnnection with Representation Aligned Latent Embeddings'
type: paper
paper_level: B
venue: Unknown
year: 2026
paper_link: https://arxiv.org/abs/2604.19902
aliases:
- 多模态轻量对齐生成框架MMCORE
- MMCORE
method: MMCORE
modalities:
- Image
paradigm: Reinforcement Learning
---

# MMCORE: MultiModal COnnection with Representation Aligned Latent Embeddings

[Paper](https://arxiv.org/abs/2604.19902)

**Topics**: [[T__Image_Generation]], [[T__Image_Editing]], [[T__Visual_Reasoning]] | **Method**: [[M__MMCORE]]

| 中文题名 | 多模态轻量对齐生成框架MMCORE |
| 英文题名 | MMCORE: MultiModal COnnection with Representation Aligned Latent Embeddings |
| 会议/期刊 | arXiv 2026 (preprint) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.19902) · [Code](https://github.com/ZijieLi/MMCORE) · [Project](https://zijieli.github.io/mmcore) |
| 主要任务 | 统一多模态理解与生成（文本到图像生成、单图/多图编辑、视觉推理） |
| 主要 baseline | Seedream 4.0、BAGEL、Transfusion、MetaQueries、Show-o |

> [!abstract] 因为「自回归模型与扩散模型紧耦合训练代价极高，且现有解耦方案对齐信号稀疏、固定查询预算适应性差」，作者在「MetaQueries解耦架构」基础上改了「引入可学习的视觉latent embeddings进行表示对齐，并采用动态注意力掩码的扩散头训练策略」，在「DreamBench AutoEval」上取得「优于Seedream 4.0的文本到图像生成质量，以及更强的反事实推理与多图编辑能力」

- **文本到图像生成**：在DreamBench AutoEval上，MMCORE与Seedream 4.0对比，在counterfactual和reasoning-heavy prompt上表现更鲁棒（图7）
- **训练效率**：无需从头预训练，基于现有VLM轻量改造，避免BAGEL/Transfusion所需的大规模预训练成本
- **多图编辑**：支持precise/reference image editing，单图与多图编辑效果优于Seedream 4.0（图8）

## 背景与动机

当前多模态生成领域面临一个根本性 tension：自回归（AR）模型在语义理解与复杂推理上表现优异，但视觉生成质量受限；扩散/流匹配（FM）模型能产出高保真图像，却缺乏强大的文本推理能力。理想方案是将二者优势结合，但直接紧耦合训练代价极高——例如BAGEL和Transfusion等统一模型需要从头大规模预训练，或用LLM权重初始化生成分支，计算资源门槛阻碍了研究可及性。

现有解耦方案试图绕过这一问题。MetaQueries采用固定数量的可学习查询令牌（N个queries）作为AR与扩散模型之间的桥梁，保留各自独立训练的优势。然而，这一设计存在两个关键缺陷：其一，固定查询预算N难以适应变长多模态上下文（如单图编辑vs. 五图序列编辑）；其二，查询令牌仅通过扩散损失监督，对齐信号稀疏，导致收敛慢、对数据分布敏感。Transfusion虽在单一模型内融合AR与扩散目标，但要求所有模态共享同一表示空间，图像建模需在干净特征（理解）与噪声特征（生成）之间频繁切换，单次前向传播无法同时优化，训练效率低下。

具体而言，当用户输入"将第一张图中的红色汽车换成蓝色，并保持第二张图背景不变"这类多图编辑指令时，MetaQueries的固定N个查询难以动态分配注意力资源，且查询与真实视觉内容的对齐不足会导致生成结果偏离参考图像。本文的核心问题意识是：能否在不深度融合AR与扩散、不从头训练的前提下，通过对现有预训练VLM的轻量改造，同时解决**对齐质量**与**训练效率**两个瓶颈？


![Figure 7](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/5173cddb-cb3f-4298-82d6-dbda2cc4fb7b/figures/Figure_7.png)
*Figure 7: Figure 7 Text-to-image generation comparison against Seedream 4.0. For each pair of images, left: MMCORE; right:Seedream 4.0.*



## 核心创新

核心洞察：将VLM输出的多模态信息压缩为**可学习的视觉latent embeddings**，并通过**表示对齐（Representation Alignment）**使这些latent embeddings与扩散模型的VAE latent space精确对应，从而使轻量级的扩散头（diffusion head）能够直接基于对齐后的表示进行条件生成，无需修改VLM内部结构或从头预训练。

与baseline的差异：

| 维度 | MetaQueries | 本文MMCORE |
|:---|:---|:---|
| 查询机制 | 固定数量N的可学习查询令牌 | 可学习的视觉latent embeddings，动态适应上下文长度 |
| 对齐监督 | 仅扩散损失，信号稀疏 | 显式表示对齐损失 + 扩散损失，对齐更精确 |
| 训练成本 | 需训练查询生成模块 | 冻结VLM，仅训练latent embeddings与轻量扩散头 |
| 生成条件 | 查询令牌作为扩散条件 | 对齐后的VAE latents直接作为扩散条件，保留更多视觉细节 |

## 整体框架


![Figure 5](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/5173cddb-cb3f-4298-82d6-dbda2cc4fb7b/figures/Figure_5.png)
*Figure 5: Figure 5 Architecture of MMCORE for low-cost Unified Multi-modal Model. Multimodal information from a VLM iscompressed into learned visual latent embeddings, which condition a diffusion-based image ge*



MMCORE的整体架构遵循"解耦-对齐-生成"的三阶段设计（图5）。数据流如下：

**输入 → 多模态编码器（冻结VLM）**：文本指令与参考图像经预训练VLM（如LLaVA系列）处理，输出多模态特征表示。VLM参数全程冻结，确保语义理解能力不受损。

**VLM输出 → Latent Compression模块**：多模态特征被压缩为**可学习的视觉latent embeddings**。此模块是本文关键创新——不同于MetaQueries的固定查询，这些latent embeddings的维度与数量可根据输入图像数量动态调整，且通过专门的表示对齐目标进行优化。

**Latent embeddings → Representation Alignment**：压缩后的latent embeddings需与扩散模型的VAE latent space对齐。具体通过辅助对齐损失，使latent embeddings能够重建对应的VAE latents，确保视觉信息在传递过程中不丢失。

**对齐后的latents → 轻量Diffusion Head**：基于对齐的VAE latents作为条件，轻量级扩散头（U-Net或DiT变体）执行去噪生成。扩散头参数量远小于完整扩散模型，因条件已高度结构化。

**输出 → 目标图像**：扩散头输出最终VAE latents，经VAE解码器恢复为像素空间图像。

```
[Text + Image(s)] ──→ [Frozen VLM] ──→ [Multimodal Features]
                                              ↓
                              [Latent Compression] ──→ [Visual Latent Embeddings]
                                              ↓
                              [Representation Alignment] ←──→ [VAE Latents]
                                              ↓
                              [Lightweight Diffusion Head] ──→ [Noised Latents → Denoised]
                                              ↓
                                        [Output Image]
```

关键设计：扩散头训练采用**动态注意力掩码**（图6），当前生成步骤仅条件于所有前置图像的VAE latents特征，实现精确的多图条件生成。

## 核心模块与公式推导

### 模块 1: Latent Compression（对应框架图 Latent Compression层）

**直觉**：VLM输出的多模态特征维度高且包含冗余信息，直接用于扩散条件会导致计算低效；需压缩为紧凑的、与扩散模型兼容的表示。

**Baseline 公式** (MetaQueries): 给定VLM输出特征 $F \in \mathbb{R}^{T \times d}$，MetaQueries使用固定N个可学习查询 $Q \in \mathbb{R}^{N \times d}$，通过交叉注意力压缩：
$$H = \text{Softmax}\left(\frac{QW_Q(FW_K)^T}{\sqrt{d_k}}\right)FW_V$$
其中 $W_Q, W_K, W_V$ 为投影矩阵，输出 $H \in \mathbb{R}^{N \times d}$ 作为扩散条件。

符号: $T$ = VLM输出序列长度, $d$ = 特征维度, $N$ = 固定查询数（超参）

**变化点**：固定N无法适应变长上下文；且交叉注意力输出与VAE latent space无显式对应关系，扩散模型需额外学习从语义空间到像素空间的映射。

**本文公式（推导）**：
$$\text{Step 1}: \quad Z_{vlm} = \text{VLM}(X_{text}, X_{img}) \in \mathbb{R}^{T \times d} \quad \text{VLM输出多模态特征}$$
$$\text{Step 2}: \quad E = \text{MLP}_{compress}(Z_{vlm}) \in \mathbb{R}^{M \times d_e} \quad \text{压缩为M个latent embeddings}$$
$$\text{Step 3}: \quad \hat{Z}_{vae} = \text{MLP}_{expand}(E) \in \mathbb{R}^{h \times w \times c} \quad \text{扩展至VAE latent空间维度}$$

其中M根据输入图像数量动态确定（单图时M=1，多图时M与图像数成正比），$d_e \ll d$ 为压缩维度，$h \times w \times c$ 为VAE latent空间维度。

**最终**: 压缩模块损失包含重建项与正则项：
$$\mathcal{L}_{compress} = \| \hat{Z}_{vae} - Z_{vae}^{gt} \|_2^2 + \lambda \|E\|_1$$

**对应消融**：

### 模块 2: Representation Alignment（对应框架图 Alignment层）

**直觉**：压缩后的latent embeddings必须与扩散模型的内部表示空间一致，否则轻量扩散头无法有效利用条件信息。

**Baseline 公式** (MetaQueries / 无显式对齐): MetaQueries仅依赖扩散损失端到端训练：
$$\mathcal{L}_{diff} = \mathbb{E}_{x_0, \epsilon, t}\left[ \| \epsilon - \epsilon_\theta(x_t, t, H) \|^2 \right]$$
其中条件 $H$ 为查询输出，对齐信号完全来自扩散损失的梯度回传，稀疏且间接。

符号: $x_0$ = 干净图像, $x_t$ = 加噪t步的图像, $\epsilon$ = 真实噪声, $\epsilon_\theta$ = 扩散模型预测噪声

**变化点**：扩散损失需同时监督查询生成、对齐、去噪三个子任务，梯度信号在长路径传播中衰减；且当数据分布变化时，查询可能过拟合到特定视觉风格而非通用表示。

**本文公式（推导）**：
$$\text{Step 1}: \quad \mathcal{L}_{align} = \| \text{VAE-Encoder}(\text{Decode}(\hat{Z}_{vae})) - Z_{vae}^{gt} \|^2 \quad \text{循环一致性约束}$$
$$\text{Step 2}: \quad \mathcal{L}_{perceptual} = \| \phi(I_{rec}) - \phi(I_{gt}) \|^2 \quad \text{感知损失，}\phi\text{为预训练VGG特征}$$
$$\text{最终}: \quad \mathcal{L}_{total}^{align} = \mathcal{L}_{compress} + \alpha \mathcal{L}_{align} + \beta \mathcal{L}_{perceptual}$$

循环一致性约束确保latent embeddings在VAE编码-解码循环中保持稳定；感知损失保留高层语义一致性。

**对应消融**：

### 模块 3: Diffusion Head with Dynamic Attention Mask（对应框架图 Diffusion Head层）

**直觉**：多图编辑时，当前生成应仅受相关前置图像影响，需显式控制条件信息的流动路径。

**Baseline 公式** (标准扩散条件生成): 标准CFG（Classifier-Free Guidance）下的条件扩散：
$$\hat{\epsilon}_\theta(x_t, t, c) = \epsilon_\theta(x_t, t, \emptyset) + s \cdot (\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \emptyset))$$
其中 $c$ 为全局条件，对所有空间位置同等作用。

符号: $c$ = 条件向量, $s$ = guidance scale, $\emptyset$ = 空条件（训练时随机dropout）

**变化点**：全局条件无法区分"当前编辑哪张图"与"保持哪张图不变"的细粒度指令；多图场景下条件图像间相互干扰。

**本文公式（推导）**：
$$\text{Step 1}: \quad M_{attn}^{(i)} \in \{0, 1\}^{h \times w} \quad \text{为第}i\text{张条件图像构造空间注意力掩码}$$
$$\text{Step 2}: \quad C_{eff} = \sum_{i=1}^{k} M_{attn}^{(i)} \odot \hat{Z}_{vae}^{(i)} \quad \text{掩码加权聚合多图条件}$$
$$\text{最终}: \quad \mathcal{L}_{diff}^{mmcore} = \mathbb{E}\left[ \| \epsilon - \epsilon_\theta(x_t, t, C_{eff}, M_{attn}) \|^2 \right]$$

动态掩码 $M_{attn}$ 在训练时根据编辑指令自动生成（图6）：对于"编辑第i张图"指令，$M_{attn}^{(i)}$ 为全1，其他条件图对应掩码空间区域被抑制；对于"保持第j张图背景"指令，$M_{attn}^{(j)}$ 在前景区域为0、背景区域为1。

**对应消融**：

## 实验与分析

主要结果对比（DreamBench AutoEval，图2）：

| Method | 文本-图像一致性 | 视觉质量 | 反事实推理 | 多图编辑 | 训练成本 |
|:---|:---|:---|:---|:---|:---|
| Seedream 4.0 |  |  | 易退化为刻板视觉 | 不支持 | 高（大规模预训练） |
| BAGEL |  |  |  |  | 极高（从头预训练） |
| MetaQueries |  |  |  | 固定N限制适应性 | 中 |
| **MMCORE** | 优于Seedream 4.0 | 可比Seedream 4.0 | **鲁棒**（图7） | **支持且精确**（图4, 图8） | **低**（冻结VLM） |


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/5173cddb-cb3f-4298-82d6-dbda2cc4fb7b/figures/Figure_1.png)
*Figure 1: Figure 2 DreamBench AutoEval results of various models.*



核心发现：
- **文本到图像生成**：MMCORE在标准视觉质量指标上与Seedream 4.0持平，但在counterfactual prompt（如"画一个正方形的轮子"）和reasoning-heavy prompt（如"三只猫，其中两只是橘色，一只是白色，橘色的猫在白色猫的左边"）上显著更鲁棒（图7）。Seedream 4.0在此类提示下常退化为训练分布中的典型视觉模式（如圆形轮子、随机颜色排列）。

- **多图编辑能力**：图4展示precise editing（精确区域编辑）与reference image editing（参考图像风格迁移），输入图像框选于左侧，输出于右侧。图8的单图与多图编辑对比显示，MMCORE在保持未编辑区域一致性上优于Seedream 4.0——后者在多图条件下常出现背景漂移或风格不一致。


![Figure 6](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/5173cddb-cb3f-4298-82d6-dbda2cc4fb7b/figures/Figure_6.png)
*Figure 6: Figure 6 Attention mask for diffusion-head training. Current generation is conditioned on the VAE latents features ofall preceding images and the current text/visual latent (VL) embeddings.*



消融分析：

公平性检查：
- **Baseline强度**：Seedream 4.0为当前商业级最强开源模型之一，BAGEL/Transfusion为学术界代表性统一模型，对比具有竞争力。
- **计算成本**：MMCORE冻结VLM，仅训练latent embeddings与轻量扩散头，参数量估计为完整模型的5-15%（具体数字待补充）。
- **数据依赖**：表示对齐模块依赖VAE latents作为监督信号，若VAE在特定域（如医学图像）表现差，则对齐质量受限。
- **Failure cases**：

## 方法谱系与知识库定位

**方法家族**：统一多模态模型（Unified Multimodal Models），属于"解耦架构"子分支。

**Parent method**：MetaQueries —— MMCORE继承其"AR理解+扩散生成解耦"的核心思想，但在查询机制（固定→可学习latent embeddings）与对齐监督（隐式→显式表示对齐）两个slot上进行了关键改进。

**直接Baselines**：
- **BAGEL / Transfusion**：差异在于MMCORE不追求单模型内的目标统一，而是通过轻量连接模块桥接冻结VLM与扩散头，避免从头预训练。
- **Show-o**：差异在于MMCORE保留扩散生成路径而非纯AR生成，视觉质量上限更高。
- **Seedream 4.0**：差异在于MMCORE开源可复现，且通过表示对齐实现更精确的条件控制，而非依赖大规模数据隐式学习。

**后续方向**：
1. **动态latent数量自适应**：当前M与图像数线性相关，可探索基于内容复杂度的自适应压缩率。
2. **跨VAE迁移**：表示对齐模块理论上可适配不同VAE，实现跨架构迁移。
3. **视频扩展**：将空间注意力掩码扩展为时空掩码，支持视频编辑中的帧级精确控制。

**知识库标签**：
- Modality: 图像-文本
- Paradigm: 解耦统一（Decoupled Unified）
- Scenario: 文本到图像生成、图像编辑、多图条件生成
- Mechanism: 表示对齐（Representation Alignment）、动态注意力掩码、Latent Compression
- Constraint: 低训练成本、冻结VLM、轻量部署

