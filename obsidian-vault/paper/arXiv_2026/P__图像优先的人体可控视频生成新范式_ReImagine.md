---
title: 'ReImagine: Rethinking Controllable High-Quality Human Video Generation via Image-First Synthesis'
type: paper
paper_level: B
venue: arXiv
year: 2026
paper_link: https://arxiv.org/abs/2604.19720
aliases:
- 图像优先的人体可控视频生成新范式
- ReImagine
- 核心直觉是「先验分离复用」：图像生成模型（FLUX.1 Kontext
code_url: https://github.com/Winn1y/Awesome-Human-Motion-Video-Generation
method: ReImagine
modalities:
- Image
---

# ReImagine: Rethinking Controllable High-Quality Human Video Generation via Image-First Synthesis

[Paper](https://arxiv.org/abs/2604.19720) | [Code](https://github.com/Winn1y/Awesome-Human-Motion-Video-Generation)

**Topics**: [[T__Video_Generation]], [[T__Image_Generation]], [[T__Pose_Estimation]] | **Method**: [[M__ReImagine]]

> [!tip] 核心洞察
> 核心直觉是「先验分离复用」：图像生成模型（FLUX.1 Kontext）已在海量数据上学会了人体外观的强先验，视频扩散模型（Wan 2.1）已学会了时序连贯性的先验，两者都不需要从多视角视频数据中重新学习。ReImagine 只需用少量多视角数据教会图像模型响应几何条件（姿态/视角），时序问题则完全外包给预训练视频模型。有效的根本原因是：将一个困难的联合学习问题分解为两个独立的、各自有充足预训练支撑的子问题，从而规避了数据瓶颈。

| 中文题名 | 图像优先的人体可控视频生成新范式 |
| 英文题名 | ReImagine: Rethinking Controllable High-Quality Human Video Generation via Image-First Synthesis |
| 会议/期刊 | arXiv (Cornell University) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.19720) · [Code](https://github.com/Winn1y/Awesome-Human-Motion-Video-Generation) · [Project](https://arxiv.org/abs/2604.19720) |
| 主要任务 | 姿态与视角联合可控的人体视频生成（pose- and view-conditioned human video generation） |
| 主要 baseline | Qwen-Image-Edit、Wan-Animate、Wan-Fun-Control、Human4DiT、Uni-Animate DiT |

> [!abstract] 因为「多视角视频数据稀缺导致外观保真度与时序一致性相互妥协」，作者在「端到端视频扩散模型」基础上改为「图像优先合成 + 无训练时序精化」的两阶段解耦范式，在「DNA-Rendering 和 MVHumanNet」上取得「PSNR 23.99 vs 基线最优 22.74，FID 36.23 vs 基线最优 39.81」的定量优势。

- **PSNR 23.99**（Disentangled Asset Pipeline 变体 22.74，差距 +1.25）
- **FID 36.23**（Disentangled Asset Pipeline 变体 39.81，差距 -3.58）
- 用户研究中 30 名参与者在视角一致性和时序平滑性两维度均偏好 ReImagine

## 背景与动机

人体视频生成的核心难点在于：外观（identity、服装）、姿态（pose）、相机视角（view）三个因素高度耦合，而高质量的多视角人体视频数据极其稀缺。以当前最大的多视角人体数据集 MVHumanNet++ 为例，其仅包含 5000 个 subject、4 个固定视角，远不足以支撑视频模型的联合训练。

现有方法沿不同路径试图突破这一瓶颈，但均存在结构性局限：
- **ControlNet 系列**（如基于 2D 骨架的姿态控制方法）：在固定视角下可实现基本姿态跟随，但不支持任何视角变换；
- **Champ**：引入 SMPL 参数改善了运动一致性，但同样缺乏显式视角控制能力；
- **大规模视频生成模型**（Wan-Fun、Wan-Animate）：具备姿态条件输入，但视角控制能力有限，且模型容量主要消耗于同时学习外观与时序；
- **Human4DiT**：最接近目标设定，但其训练数据来自网络爬取的单目视频，视角变化不受显式控制，无法实现精确的姿态+视角联合可控生成。

这些方法的共同困境源于一个根本假设：**视频模型需要从有限的多视角视频数据中同时学习几何条件（姿态/视角）和外观先验**。然而，图像生成模型（如 FLUX 系列）已在海量互联网图像上积累了强大的人体外观先验，视频扩散模型（如 Wan 2.1）已在海量视频中学会了时序连贯性先验——这两个先验在现有「端到端视频训练」范式中被浪费或稀释，而多视角视频数据的稀缺性又使模型无法从头学好任何一方。

ReImagine 的核心动机正在于此：能否将外观建模与时序一致性解耦为两个独立阶段，分别复用各自领域已充分预训练的模型，从而完全绕开「在有限多视角视频上联合训练」的数据瓶颈？
![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/22ddd458-b67d-4b18-b1e1-167cba3be9e9/figures/Figure_1.png)
*Figure 1: Fig. 1: Our method enables controllable human synthesis at multiple levels. (a) Ourpipeline generates temporally coherent videos with explicit control over body pose andcamera viewpoint. (b) Our image*



## 核心创新

核心洞察：将「外观生成」与「时序精化」分离到两个已充分预训练的专用模型中，因为图像模型 FLUX.1 Kontext 已在海量数据上掌握人体外观强先验、视频模型 Wan 2.1 已掌握时序连贯性强先验，从而使「仅用少量多视角数据微调几何响应能力、零视频训练实现视频输出」成为可能。

| 维度 | Baseline（端到端视频扩散） | 本文 ReImagine |
|:---|:---|:---|
| **训练数据** | 直接消耗稀缺的多视角视频数据 | 仅图像模型接触多视角数据，视频模型零训练 |
| **外观先验来源** | 从有限视频重新学习 | 复用 FLUX.1 Kontext 的海量图像预训练 |
| **时序一致性来源** | 与外观联合训练，相互妥协 | 复用 Wan 2.1 预训练时序能力，无训练精化 |
| **可控性粒度** | 姿态或视角单一/受限 | SMPL-X 法线图联合控制姿态 + 视角 |
| **推理流程** | 单次视频扩散生成 | 图像逐帧合成 → 视频时序精化两阶段 |

## 整体框架


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/22ddd458-b67d-4b18-b1e1-167cba3be9e9/figures/Figure_2.png)
*Figure 2: Fig. 2: Overview of our image-first training and inference paradigm. (a) During train-ing, a powerful pretrained image backbone is fine-tuned via lightweight LoRA adap-tation using an imperfect multi-*



ReImagine 的整体框架由两个解耦阶段构成，训练阶段完全不涉及视频数据，推理阶段才引入预训练视频模型：

**阶段一：图像合成（训练阶段）**
- **输入**：canonical 正面/背面人体图像（参考外观）+ SMPL-X 法线图（几何条件）
- **模块**：FLUX.1 Kontext 主干 + LoRA（rank=128）微调 + 冻结参数的 ControlNet（Surface Normals）
- **输出**：逐帧高质量人体图像，响应指定姿态与视角
- **角色**：利用图像模型的强外观先验，仅学习「几何条件→外观映射」，无需从头学习人体外观

**阶段二：时序精化（推理阶段，无训练）**
- **输入**：阶段一输出的帧序列（含细微帧间外观抖动）
- **模块**：Wan 2.1 I2V-14B-480P 预训练视频扩散模型
- **操作**：以去噪强度 0.7、20 步推理进行时空正则化（low-noise re-denoising + spatiotemporal spectral regularization）
- **输出**：时序连贯的最终视频
- **角色**：将时序一致性问题完全外包给已具备该能力的预训练视频模型

**辅助模块：Disentangled Asset Pipeline（变体，非主线）**
- 通过 condition-aware positional encoding 将 identity、garment、footwear 等资产与空间语义角色关联
- 支持跨 identity 外观重组，但定量指标弱于主方法

数据流简图：
```
Canonical 正背面图像 + SMPL-X 法线图
        ↓
[FLUX.1 Kontext + LoRA + ControlNet]  →  逐帧图像序列
        ↓
[Wan 2.1 I2V, 去噪强度 0.7, 20步]      →  时序连贯视频
```

## 核心模块与公式推导

### 模块 1: 图像生成阶段的几何条件化微调（对应框架图阶段一）

**直觉**: 不从头训练外观，而是冻结强预训练图像模型的主体参数，仅通过轻量 LoRA 学习「如何将 SMPL-X 法线图映射为对应姿态/视角下的人体外观」。

**Baseline 公式**（标准扩散模型训练, e.g., Stable Diffusion/FLUX）：
$$L_{base} = \mathbb{E}_{x_0, t, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t, c_{text}) \|^2 \right]$$
符号: $x_0$ = 干净图像, $x_t$ = 第 $t$ 步加噪图像, $\epsilon$ = 真实噪声, $\epsilon_\theta$ = 去噪网络, $c_{text}$ = 文本条件

**变化点**: 标准 FLUX 仅接受文本条件 $c_{text}$，无法响应精细的几何控制；且全参数微调在 5000 subject 的多视角数据上会灾难性遗忘外观先验。

**本文公式（推导）**:
$$\text{Step 1}: \theta_{frozen} = \text{FLUX.1 Kontext pretrained} \quad \text{冻结全部主干参数保留外观先验}$$
$$\text{Step 2}: \Delta W = BA, \quad B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}, r=128 \quad \text{低秩分解引入可训练几何适配器}$$
$$\text{Step 3}: c_{geo} = \text{ControlNet}_{frozen}(\text{SMPL-X normal map}) \quad \text{冻结 ControlNet 提取几何特征}$$
$$\text{最终}: L_{image} = \mathbb{E}_{x_0, t, \epsilon} \left[ \| \epsilon - \epsilon_{\theta_{frozen} + \Delta W}(x_t, t, c_{text}, c_{geo}) \|^2 \right]$$

训练配置：4×A100，10 epoch，batch size=32，学习率 1e-4。

**对应消融**: Table 1 显示主方法（Image-first）vs Disentangled Asset Pipeline 变体，PSNR 23.99 vs 22.74，FID 36.23 vs 39.81，验证冻结主干+LoRA 微调优于引入额外解耦约束。

---

### 模块 2: 无训练时序精化（对应框架图阶段二）

**直觉**: 逐帧独立生成必然引入细微外观抖动（衣物褶皱、光照微变），利用预训练视频扩散模型的时空去噪能力，在极低噪声水平下做「重去噪」正则化，而非从头生成视频。

**Baseline 公式**（标准视频扩散模型 I2V 推理, e.g., Wan 2.1）：
$$x_{t-1}^{video} = \text{VDM}_\phi(x_t^{video}, t, c_{image})$$
符号: $x^{video}$ = 视频潜变量序列, $\phi$ = 视频扩散模型参数, $c_{image}$ = 首帧/参考图像条件

**变化点**: 标准 I2V 以高噪声 $x_T^{video} \sim \mathcal{N}(0, I)$ 起始，需要模型同时生成内容和时序，会覆盖图像阶段的高质量外观；本文改为以图像阶段输出的低噪声潜变量起始，仅做精化。

**本文公式（推导）**:
$$\text{Step 1}: x_{t_{low}}^{video} = \text{Encode}(\{I_1, I_2, ..., I_N\}) + \sigma_{t_{low}} \epsilon \quad \text{图像序列编码并加少量噪声}$$
$$\text{其中 } t_{low} \text{ 对应去噪强度 } 0.7 \text{（非标准 } 1.0\text{），保留图像内容结构}$$
$$\text{Step 2}: \mathcal{R}_{ST}(x) = \mathcal{F}^{-1}(M_{freq} \odot \mathcal{F}(x)) \quad \text{时空频谱正则化，抑制时序高频抖动}$$
$$\text{Step 3}: x_{t-1}^{video} = \text{VDM}_\phi(x_t^{video}, t, c_{image}=I_1) + \lambda \mathcal{R}_{ST}(x_t^{video}) \quad \text{重去噪+频谱约束}$$
$$\text{最终}: \{I_1^{final}, ..., I_N^{final}\} = \text{Decode}(x_0^{video}) \quad \text{20步推理后解码}$$

**对应消融**: Figure 7（Temporal consistency ablation via tracking visualization）显示该模块有效抑制了逐帧生成的外观漂移，跟踪点轨迹更平滑。

---

### 模块 3: Pose- and View-Guided Generation 的 SMPL-X 条件化（对应框架图 Figure 3）

**直觉**: 需要统一的隐式表征将 3D 人体几何（姿态+视角）编码为图像扩散模型可理解的 2D 条件，SMPL-X 法线图同时编码身体姿态和相机视角信息。

**本文设计**:
$$c_{pose}, c_{view} = \text{SMPL-X}(\beta, \theta_{body}, \theta_{hand}, \theta_{face}, \pi_{camera})$$
$$\text{Surface Normal Render}: n = \mathcal{R}(c_{pose}, c_{view}) \in \mathbb{R}^{H \times W \times 3}$$
$$\text{其中 } \pi_{camera} \text{ 显式编码相机外参，实现视角可控}$$

该模块将 3D 可驱动人体模型的几何先验与 2D 扩散模型的生成先验桥接，避免了从视频中学习 3D 几何的困难。

## 实验与分析

主实验结果在 DNA-Rendering 和 MVHumanNet 两个数据集上进行，对比 Qwen-Image-Edit、Wan-Animate、Wan-Fun-Control、Human4DiT 四条基线：

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | FID↓ |
|:---|:---|:---|:---|:---|
| Qwen-Image-Edit |  |  |  |  |
| Wan-Animate |  |  |  |  |
| Wan-Fun-Control |  |  |  |  |
| Human4DiT |  |  |  |  |
| **ReImagine (Ours)** | **23.99** | **0.827** | **0.165** | **36.23** |
| — Disentangled Asset Pipeline | 22.74 | 0.821 | 0.178 | 39.81 |


![Figure 6](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/22ddd458-b67d-4b18-b1e1-167cba3be9e9/figures/Figure_6.png)
*Figure 6: Fig. 5: Qualitative comparison for image-to-video human synthesis on the MVHu-manNet++ dataset [25]. We compare our method with Wan-Fun [1], Wan-Animate(Wan-Ani) [3], Qwen [47], and Human4DiT [39]. Th*



**核心结论支持**：ReImagine 在全部四项指标上优于自身变体 Disentangled Asset Pipeline，PSNR +1.25、FID -3.58 的差距明确支持「冻结图像主干+轻量 LoRA 优于引入额外解耦约束」的设计选择。与外部基线的对比（Table 1 完整版）显示该优势扩展到视频优先方法。

**消融分析**：Figure 7 的时序一致性消融通过跟踪可视化（tracking visualization）展示了无训练时序精化模块的有效性——去除该模块后，关键点轨迹出现明显抖动，加入后轨迹平滑。
![Figure 7](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/22ddd458-b67d-4b18-b1e1-167cba3be9e9/figures/Figure_7.png)
*Figure 7: Fig. 7: Temporal consistency ablation viatracking visualization.*



**定性对比**：Figure 6 在 MVHumanNet++ 上与 Wan-Fun、Wan-Animate 等对比，ReImagine 在视角一致性（view consistency）和外观保真度上更优；Figure 9 直接与视频优先基线 Uni-Animate DiT 对比，图像优先方法在保持首帧外观细节方面优势明显。
![Figure 9](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/22ddd458-b67d-4b18-b1e1-167cba3be9e9/figures/Figure_9.png)
*Figure 9: Fig. 9: Qualitative comparison between our image-first method and a video-first base-line (Uni-Animate DiT). The leftmost column shows the canonical front reference input(back reference omitted for sp*



**公平性检查与局限**：
- **基线强度**：未包含 Introduction 中提及的 Champ、MV-Performer 等方法，基线覆盖不完整；
- **任务设定公平性**：对原本不以第一帧为输入的模型（如 Qwen）强制使用相同输入格式，可能削弱其表现；
- **数据成本**：图像阶段 4×A100、10 epoch 训练，视频阶段零训练，整体计算成本低于端到端视频训练但仍需高端 GPU；
- **样本量**：零样本评测仅 15 个 subject，用户研究 30 人且含领域研究者，泛化结论与偏好结论需谨慎；
- **超参敏感性**：去噪强度 0.7 为固定值，未报告敏感性分析；训练视角仅 4 个离散方向，插值视角质量未充分验证。

## 方法谱系与知识库定位

**方法家族**：扩散模型条件生成 → 人体特定视频生成 → 解耦式（disentangled）生成范式

**父方法**：FLUX.1 Kontext（图像生成主干）+ Wan 2.1 I2V（视频时序精化）。ReImagine 并非改进单一模型，而是重新编排两个已充分预训练模型的协作方式。

**改变的 slots**：
- **架构（architecture）**：端到端视频 U-Net/DiT → 图像模型逐帧合成 + 视频模型时序精化的级联架构
- **训练配方（training_recipe）**：联合训练视频扩散 → 图像模型 LoRA 微调 + 视频模型零训练
- **数据策划（data_curation）**：直接消耗多视角视频 → 仅图像阶段使用多视角数据，视频模型完全复用预训练
- **推理（inference）**：单次扩散生成 → 两阶段级联，引入可控去噪强度参数

**直接基线与差异**：
- **Human4DiT**：同为人体视频生成，但采用视频优先训练、网络爬取单目数据，无显式视角控制；ReImagine 改为图像优先、SMPL-X 联合控制姿态+视角
- **Wan-Animate / Wan-Fun-Control**：同为 Wan 系列视频模型，但直接用于端到端生成；ReImagine 仅将其用于第二阶段时序精化，外观生成外包给 FLUX
- **Champ**：引入 SMPL 改善运动一致性，但缺乏视角控制且为视频训练；ReImagine 用 SMPL-X 同时编码姿态视角，并解耦时序学习

**后续方向**：
1. 扩展连续视角插值：当前 4 个离散视角训练限制视角变化平滑度，可探索视角条件连续编码；
2. 端到端联合微调：当前两阶段完全解耦，轻度联合微调可能进一步提升时序-外观一致性；
3. 实时推理优化：FLUX 逐帧生成 + Wan 精化的级联延迟较高，可探索模型蒸馏或并行生成策略。

**知识库标签**：
- **模态（modality）**：image-to-video, human-centric generation
- **范式（paradigm）**：cascaded generation, prior reuse, training-free temporal refinement
- **场景（scenario）**：controllable human animation, multi-view synthesis
- **机制（mechanism）**：LoRA fine-tuning, ControlNet conditioning, SMPL-X geometric guidance, spatiotemporal spectral regularization
- **约束（constraint）**：limited multi-view data, disentangled appearance-temporal modeling

