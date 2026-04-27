---
title: 'SoftVQ-VAE: Efficient 1-Dimensional Continuous Tokenizer'
type: paper
paper_level: C
venue: CVPR
year: 2025
paper_link: null
aliases:
- SoftVQ-VAE：高效1维连续Tokenizer
- SoftVQ-VAE
acceptance: poster
cited_by: 44
method: SoftVQ-VAE
---

# SoftVQ-VAE: Efficient 1-Dimensional Continuous Tokenizer

**Topics**: [[T__Image_Generation]], [[T__Compression]] | **Method**: [[M__SoftVQ-VAE]] | **Datasets**: [[D__ImageNet-1K]]

| 中文题名 | SoftVQ-VAE：高效1维连续Tokenizer |
| 英文题名 | SoftVQ-VAE: Efficient 1-Dimensional Continuous Tokenizer |
| 会议/期刊 | CVPR 2025 (poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2412.10958) · [Code] · [Project] |
| 主要任务 | 图像tokenization、类条件图像生成（ImageNet 256×256 / 512×512） |
| 主要 baseline | VQ-VAE, KL-VAE, TiTok, MAR-H, LDM-4, LlamaGen |

> [!abstract] 因为「VQ-VAE的硬量化导致不可微、需要commitment loss和codebook loss，且高压缩比下质量显著下降」，作者在「VQ-VAE」基础上改了「将argmin硬量化替换为带温度τ的softmax软分类分布，并引入DINOv2表示对齐」，在「ImageNet 256×256」上取得「gFID 2.81（无CFG）/ 1.93（有CFG），仅用64个token，相比MAR-H的256个token减少4×」

- **效率**：MAR-H (KL, 256 tokens) 吞吐量为 0.12 imgs/sec，SoftVQ-VAE (64 tokens) 达到 0.89 imgs/sec，**7.4× 加速**
- **重建质量**：SoftVQ-VAE rFID 0.65，优于 MAR-H (1.22) 和 TiTok-S-128 (1.61)，且 token 数更少
- **计算量**：SoftVQ-VAE GFLOPs 86.55，比 MAR-H (145.08) **减少 40%**

## 背景与动机

现代视觉生成模型（如Stable Diffusion、DALL-E、MAR等）普遍依赖tokenizer将高维图像压缩为紧凑的潜表示。理想情况下，这种压缩应满足三个目标：高压缩比（减少生成模型的计算负担）、低重建误差（保留视觉细节）、以及良好的下游生成性能。然而，现有方法在这三者之间存在根本性张力。

**VQ-VAE** 通过硬量化（hard quantization）将连续编码器输出映射到离散的码本条目，配合commitment loss和codebook loss训练。这种方案虽然简单有效，但argmin操作不可微，需要straight-through estimator或EMA等技巧，且高压缩比下重建质量显著退化。**KL-VAE**（如LDM所用）采用连续潜变量，避免了量化问题，但通常需要更多token（如LDM-4使用4096个token）才能达到可接受的重建质量，压缩效率不足。**TiTok** 引入了1D可学习token的概念，将图像表示为少量latent token，但其仍采用某种形式的硬量化策略，且表示容量受限于固定编码方式。

这些方法的共同瓶颈在于**量化操作的刚性**：VQ-VAE的硬量化将每个编码器输出强制绑定到单一码本向量，丢失了编码器输出与多个相近码字之间的丰富结构信息；同时，不可微性迫使训练流程引入额外的辅助损失和超参数调优。当追求极端压缩比（如32或64个token表示256×256图像）时，这种刚性导致信息瓶颈，重建和生成质量急剧下降。

本文的核心动机是：**能否设计一种完全可微的软量化机制，让每个潜变量自适应地聚合多个码本向量的信息，从而在不牺牲压缩效率的前提下提升表示质量？** 基于此，作者提出SoftVQ-VAE，以简单的softmax替代argmin，实现端到端梯度优化，并配合预训练视觉特征的表示对齐来规范潜空间结构。
![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/102050a0-c0a2-4429-899e-d42f938ee7fe/figures/Figure_3.png)
*Figure 3 (qualitative): Figure 3. Visualization of 8 steps of x0/σ, encoder output and x0 codebook. At top: VQ-VAE-2; at bottom: SoftVQ-8-64-64.*



## 核心创新

核心洞察：**用带温度参数的softmax分布替代argmin硬量化，因为softmax产生的软分类分布使每个潜变量成为码本向量的自适应加权和，从而使完全可微的端到端训练、去除commitment/codebook loss、以及极端压缩比下的高质量重建成为可能。**

与baseline的差异：

| 维度 | Baseline (VQ-VAE) | 本文 (SoftVQ-VAE) |
|:---|:---|:---|
| 量化方式 | 硬argmin：$k = \text{arg}\min_j \|\hat{\mathbf{z}} - \mathbf{c}^{[j]}\|_2$，不可微 | 软softmax：$q_\phi(\mathbf{z}\|\mathbf{x}) = \text{Softmax}(-\|\hat{\mathbf{z}} - \mathcal{C}\|_2/\tau)$，完全可微 |
| 潜变量计算 | 单码字：$\mathbf{z} = \mathbf{c}^{[k]}$ | 加权聚合：$\mathbf{z} = q_\phi(\mathbf{z}\|\mathbf{x})\mathcal{C}$，多码字信息融合 |
| 训练目标 | 重建 + commitment loss + codebook loss，需调参 | 重建 + 感知 + 对抗 + 对齐 + KL，无commitment/codebook loss |
| 优化方式 | 编码器需straight-through/EMA绕过不可微量化 | 编码器与码本直接梯度更新，端到端优化 |
| 压缩灵活性 | 固定2D latent grid | 1D可学习token，任意长度L自适应不同分辨率 |

## 整体框架


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/102050a0-c0a2-4429-899e-d42f938ee7fe/figures/Figure_2.png)
*Figure 2 (architecture): Figure 2. Illustration of SoftVQ-VAE. Left: Transformer encoder produces continuous representation h. The representation alignment regularizes the encoder output towards the learnable codebook embedding e. Right: fully differentiable SoftVQ.*



SoftVQ-VAE的整体架构遵循编码器-量化器-解码器的经典范式，但关键模块均经过重新设计以支持软量化和1D表示。数据流如下：

**输入 → ViT Encoder E**：图像$\mathbf{x}$首先被切分为patch $\mathbf{x}_p$，与一组可学习的1D latent token $\mathbf{z}_l$拼接后送入Transformer编码器。编码器输出对应的latent token表示$\hat{\mathbf{z}} \in \mathbb{R}^{L \times d}$，其中$L$为latent序列长度（如32或64），$d$为维度。

**SoftVQ Quantization**：编码器输出$\hat{\mathbf{z}}$进入核心创新模块SoftVQ层。该模块计算$\hat{\mathbf{z}}$与可学习码本$\mathcal{C} \in \mathbb{R}^{K \times d}$（$K$为码本大小）中所有码字的L2距离，通过带温度$\tau$的softmax转换为软分类分布，最终输出加权聚合的连续潜变量$\mathbf{z} = q_\phi(\mathbf{z}|\mathbf{x})\mathcal{C}$。

**ViT Decoder D**：潜变量$\mathbf{z}$与mask token $\mathbf{m}$拼接后送入Transformer解码器，重建原始图像$\hat{\mathbf{x}}$。解码器采用与编码器对称的ViT结构。

**Representation Alignment（旁路监督）**：为提升潜空间的语义质量，将$\mathbf{z}$重复扩展为$\mathbf{z}_r$（长度与图像patch数$N$相同），经MLP映射后与预训练DINOv2特征$\mathbf{y}_*$计算相似度损失$\mathcal{L}_{\text{align}}$，强制latent包含高层语义信息。

完整数据流示意：
```
Image x → Patchify → [x_p; z_l] ──→ Encoder E ──→ ẑ
                                              ↓
                                        SoftVQ(ẑ, C, τ)
                                              ↓
                                              z ──→ [m; z] ──→ Decoder D ──→ x̂
                                              ↓
                                         z_r → MLP → sim(·, DINOv2(y*))
                                              ↓
                                         L_align
```

## 核心模块与公式推导

### 模块 1: SoftVQ Soft Quantization（对应框架图 Figure 2 右侧上方）

**直觉**：VQ-VAE的硬量化类似K-Means聚类，将每个点强制分配到最近中心；SoftVQ则像软K-Means，允许点以概率权重属于多个中心，从而保留更多结构信息并恢复可微性。

**Baseline 公式 (VQ-VAE)**:
$$q_{\phi}(\mathbf{z} = k | \mathbf{x}) = \begin{cases} 1 & \text{if } k = \text{arg}\min_{j} \| \hat{\mathbf{z}} - \mathbf{c}^{[j]} \|_2 \\ 0 & \text{otherwise} \end{cases}, \quad \mathbf{z} = \mathbf{c}^{[k]}$$
符号: $\hat{\mathbf{z}}$ = 编码器输出, $\mathcal{C} = \{\mathbf{c}^{[j]}\}_{j=1}^K$ = 码本, $K$ = 码本大小。

**变化点**：argmin不可微，导致编码器梯度中断；单码字表示容量有限，极端压缩时信息损失严重。

**本文公式（推导）**:
$$\text{Step 1}: q_{\phi}(\mathbf{z} | \mathbf{x}) = \text{Softmax}\left(-\frac{\| \hat{\mathbf{z}} - \mathcal{C} \|_2}{\tau}\right) \quad \text{将硬选择松弛为带温度τ的概率分布}$$
$$\text{Step 2}: \mathbf{z} = q_{\phi}(\mathbf{z} | \mathbf{x}) \mathcal{C} = \sum_{j=1}^{K} q_{\phi}(\mathbf{z}_j | \mathbf{x}) \cdot \mathbf{c}^{[j]} \quad \text{概率加权聚合所有码字，非单一选择}$$
$$\text{最终}: \mathbf{z} \in \mathbb{R}^{d} \text{ 为连续向量，完全可微，} \tau=0.07 \text{ 控制分布尖锐度}$$

**对应消融**：温度$\tau$的鲁棒性在实验中得到验证，$\tau=0.07$为选定值；不同码本大小下性能稳定。

---

### 模块 2: Entropy-based KL Regularization（对应框架图训练目标部分）

**直觉**：VQ-VAE的KL散度退化为常数$\log K$，不提供有效正则化；SoftVQ需要利用软分布的可计算熵来同时鼓励单样本不确定性和整体码本利用率均匀性。

**Baseline 公式 (VQ-VAE)**:
$$\mathcal{L}_{\text{kl}}^{\text{VQ-VAE}} = \log K \quad \text{(常数，因后验为one-hot，先验为均匀分布)}$$

**变化点**：硬量化使后验熵恒为0，KL无信息量；SoftVQ的软分布使熵可计算，可设计有意义的正则化。

**本文公式（推导）**:
$$\text{Step 1}: \mathcal{L}_{\text{kl}} = \underbrace{H(q_{\phi}(\mathbf{z} | \mathbf{x}))}_{\text{后验熵：鼓励单个样本分布有一定展宽}} - \underbrace{H\left(\mathbb{E}_{\mathbf{x} \sim p(\mathbf{x})} q_{\phi}(\mathbf{z} | \mathbf{x})\right)}_{\text{边际熵：鼓励整体码本被均匀利用}}$$
$$\text{Step 2}: = -\sum_{j} q_{\phi}(\mathbf{z}_j|\mathbf{x}) \log q_{\phi}(\mathbf{z}_j|\mathbf{x}) + \sum_{j} \bar{q}_j \log \bar{q}_j, \quad \bar{q} = \frac{1}{B}\sum_{i=1}^{B} q_{\phi}(\mathbf{z}|\mathbf{x}_i)$$
$$\text{最终}: \mathcal{L}_{\text{kl}}^{\text{SoftVQ}} \text{ 为可学习正则项，替代commitment loss和codebook loss}$$

**对应消融**：去掉KL项或调整其权重会影响码本利用率和训练稳定性，但论文未提供精确数值对比。

---

### 模块 3: Representation Alignment with DINOv2（对应框架图 Figure 2 右侧下方）

**直觉**：纯像素级重建损失无法保证潜空间捕获高层语义；利用预训练视觉模型DINOv2的强特征作为"教师"，强制latent token的扩展版本与之对齐，可显著提升下游生成质量。

**Baseline 公式**：无（VQ-VAE无显式语义对齐机制）。

**本文公式（推导）**:
$$\text{Step 1}: \mathbf{z}_r = [\underbrace{\mathbf{z}^{[0]}, \ldots, \mathbf{z}^{[0]}}_{N/L \text{ times}}, \ldots, \underbrace{\mathbf{z}^{[L]}, \ldots, \mathbf{z}^{[L]}}_{N/L \text{ times}}] \in \mathbb{R}^{N \times d} \quad \text{将L个latent token重复至N个图像patch}$$
$$\text{Step 2}: \mathcal{L}_{\text{align}} = \frac{1}{N} \sum_{n=1}^{N} \text{sim}\left(\mathbf{y}_*^{[n]}, \text{MLP}(\mathbf{z}_r^{[n]})\right) \quad \text{与DINOv2特征计算余弦相似度}$$
$$\text{最终}: \mathcal{L} = \mathcal{L}_{\text{recon}} + \lambda_1 \mathcal{L}_{\text{percep}} + \lambda_2 \mathcal{L}_{\text{adv}} + \lambda_3 \mathcal{L}_{\text{align}} + \lambda_4 \mathcal{L}_{\text{KL}}, \quad \lambda_1=1.0, \lambda_2=0.2, \lambda_3=0.1, \lambda_4=0.01$$

**对应消融**：Table 4 显示不同编码器初始化和表示对齐维度的影响；更大的latent空间维度改善重建但恶化生成，说明对齐的适度约束对生成质量至关重要。

---

**GMMVQ变体**（非主方法）：
$$q_{\phi}(\mathbf{z} | \mathbf{x}) = \text{Softmax}\left(- \omega(\hat{\mathbf{z}}) \| \hat{\mathbf{z}} - \mathcal{C} \|_2\right)$$
用数据相关权重$\omega(\hat{\mathbf{z}})$替代固定$\tau$，实验效果与SoftVQ相当但增加复杂度，故主方法采用更简洁的SoftVQ。

## 实验与分析


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/102050a0-c0a2-4429-899e-d42f938ee7fe/figures/Table_1.png)
*Table 1 (comparison): Table 1. System-level comparison on ImageNet 256×256 conditional generation. We compare with both diffusion-based models and auto-regressive models using FID, sFID, Precision, Recall, IS and train/testing throughput.*



本文在ImageNet 256×256和512×512上进行系统级评估，对比自回归模型（MAR-H, LlamaGen）和扩散模型（DiT-XL, SiT-XL, LDM-4）的生成性能。核心效率-质量权衡在Table 1中呈现：SoftVQ-VAE使用仅64个token，在ImageNet 256×256无CFG条件下达到gFID 2.81，相比使用256个token的MAR-H (KL tokenizer) 的gFID 2.35略有差距，但token数减少4×；其rFID 0.65显著优于MAR-H的1.22和TiTok-S-128的1.61，表明tokenizer本身的重建质量更高。在推理效率方面，SoftVQ-VAE的吞吐量达到0.89 imgs/sec，是MAR-H (0.12) 的7.4倍，GFLOPs 86.55比MAR-H (145.08) 减少40%。


![Table 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/102050a0-c0a2-4429-899e-d42f938ee7fe/figures/Table_2.png)
*Table 2 (comparison): Table 2. System-level comparison on ImageNet 512×512 conditional generation. We compare with both diffusion-based models and auto-regressive models using FID, sFID, Precision, Recall, IS and train/testing throughput.*



有CFG条件下，SoftVQ-VAE gFID 1.93，仍略逊于MAR-H的1.55，但考虑token效率后这一差距被认为可接受。值得注意的是，SoftVQ-L（32 token）的gFID为3.83，反而差于SoftVQ-BL（64 token）的2.81，说明token数与生成质量之间存在非线性权衡，并非越少越好。


![Table 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/102050a0-c0a2-4429-899e-d42f938ee7fe/figures/Table_4.png)
*Table 4 (ablation): Table 4. Representation alignment with different codebook embedding dimensions.*



消融实验（Table 4）检验表示对齐的不同配置：更大的码本嵌入维度改善重建但可能损害生成，验证了语义对齐约束的必要性。Product Quantization (PQ) 和 Residual Quantization (RQ) 变体相比基础SoftVQ略有提升，GMMVQ变体效果与SoftVQ相当。温度$\tau$和码本大小在较宽范围内保持鲁棒。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/102050a0-c0a2-4429-899e-d42f938ee7fe/figures/Figure_1.png)
*Figure 1 (example): Figure 1. ImageNet 256 × 256 and 512 × 512 generation results of pretrained models trained on SoftVQ-VAE with 32 and 64 scales.*



公平性检验：本文比较存在若干问题。Table 1严重截断，大量baseline数值缺失；对比混合了不同生成模型架构（MAR-H vs DiT-XL vs SiT-XL）与相同tokenizer的组合，难以 isolate tokenizer本身的贡献；缺少与SDXL/SD3 tokenizer、OpenMagVIT等2024年后强baseline的直接对比；且未控制生成模型容量进行严格的tokenizer ablation。作者披露SoftVQ-L（32 token）生成质量下降，暗示极端压缩存在瓶颈。

## 方法谱系与知识库定位

**方法家族**：VQ-VAE → SoftVQ-VAE。直接父方法为 **VQ-VAE**（van den Oord et al.），本文通过软量化实现完全可微扩展。

**改变的slots**：
- **Architecture**：硬argmin → softmax软分类分布，潜变量变为码本向量的概率加权和
- **Objective**：去除commitment loss + codebook loss → 引入entropy-based KL + DINOv2表示对齐
- **Training recipe**：straight-through/EMA → 端到端梯度优化
- **Data pipeline**：固定2D latent grid → 1D可学习token，长度L任意可调

**直接baseline差异**：
- **VQ-VAE**：硬量化不可微，需辅助损失；SoftVQ完全可微，表示更灵活
- **KL-VAE**（LDM-4）：连续但token数多（4096）；SoftVQ压缩率更高（64 token）
- **TiTok**：同样1D token但量化策略不同；SoftVQ的软量化提供更平滑的梯度
- **MAR-H**（KL tokenizer）：系统级对比目标，SoftVQ以更少token逼近其生成质量

**后续方向**：(1) 将SoftVQ扩展至视频/3D tokenization，利用其可微性处理时序连续性；(2) 结合流匹配或一致性模型，探索软量化在连续时间生成中的优势；(3) 与LLM架构深度融合，验证1D连续token在视觉-语言联合建模中的效率。

**标签**：modality=图像 / paradigm=自编码器+量化 / scenario=高效视觉生成 / mechanism=软量化+表示对齐 / constraint=极端压缩比下保持质量

