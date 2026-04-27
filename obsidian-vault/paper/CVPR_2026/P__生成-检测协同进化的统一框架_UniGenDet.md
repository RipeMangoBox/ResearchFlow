---
title: 'UniGenDet: A Unified Generative-Discriminative Framework for Co-Evolutionary Image Generation and Generated Image Detection'
type: paper
paper_level: B
venue: CVPR
year: 2026
paper_link: https://arxiv.org/abs/2604.21904
aliases:
- 生成-检测协同进化的统一框架
- UniGenDet
- 核心直觉是：生成器对图像分布的内部表示（VAE潜变量）天然包含伪影的成
acceptance: accepted
method: UniGenDet
modalities:
- Image
---

# UniGenDet: A Unified Generative-Discriminative Framework for Co-Evolutionary Image Generation and Generated Image Detection

[Paper](https://arxiv.org/abs/2604.21904)

**Topics**: [[T__Image_Generation]], [[T__Object_Detection]], [[T__Adversarial_Robustness]] | **Method**: [[M__UniGenDet]]

> [!tip] 核心洞察
> 核心直觉是：生成器对图像分布的内部表示（VAE潜变量）天然包含伪影的成因信息，而检测器对真实性的判断标准可以反向约束生成器避免产生可检测的伪影。SMSA让检测器直接读取生成器的内部状态（而非仅看最终图像），DIGA让生成器在特征层面对齐检测器的取证知识（而非接收二元标量反馈）。两个方向的信息流动都发生在高维连续空间，避免了离散反馈导致的梯度消失和模式崩溃。有效性的根本原因在于：共享同一LLM骨干使得两个任务的特征空间天然兼容，跨模态注意力和特征对齐因此可以在语义上有意义地工作。

| 中文题名 | 生成-检测协同进化的统一框架 |
| 英文题名 | UniGenDet: A Unified Generative-Discriminative Framework for Co-Evolutionary Image Generation and Generated Image Detection |
| 会议/期刊 | CVPR 2026 (accepted) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.21904) · [Code](https://github.com/UniGenDet/UniGenDet ⭐
| 主要任务 | 图像生成（Image Generation）、生成图像检测（Generated Image Detection） |
| 主要 baseline | BAGEL、Qwen2-VL-72B、SIDA、FakeVLM、D3QE、CNNSpot |

> [!abstract] 因为「生成与检测任务长期独立演进、架构鸿沟导致无法协同」，作者在「BAGEL多模态基础模型」基础上改了「Symbiotic Multi-modal Self-Attention（SMSA）跨模态注意力模块 + Detector-Informed Generative Alignment（DIGA）特征对齐机制」，在「FakeClue检测基准」上取得「98.0% Acc / 97.7% F1，超越Qwen2-VL-72B达40.2%/41.2%」

- **检测性能**: FakeClue上Acc 98.0% / F1 97.7%，较Qwen2-VL-72B提升40.2% / 41.2%；DMImage跨数据集Acc 98.6% / F1 99.1%，超SIDA +6.8%/+6.7%
- **生成质量**: FID 17.5，优于BAGEL基线22.9（相对改善23.6%）；LPIPS 0.726，多样性未发生模式崩溃
- **效率**: GDUF阶段8×A100训练约1000步（12小时），DIGA阶段500步（6小时），总训练约18小时

## 背景与动机

当前图像生成与生成图像检测两个领域看似对立，实则面临同一根本困境：生成器不断逼近真实图像分布，检测器被迫在愈发逼真的伪造样本上寻找微弱痕迹。然而，二者长期独立演进——生成任务依赖扩散模型或自回归模型的VAE潜空间操作，检测任务依赖CNN或ViT的判别式特征提取——这种架构分裂造成了严重的协同障碍。

现有方法如何处理这一问题？**BAGEL**作为近期多模态基础模型，首次将生成与理解任务统一于同一LLM骨干（Qwen2.5），但生成与检测模块之间仍缺乏显式信息交互，各自优化各自的目标。**传统对抗训练方案**（如GAN-based检测器）向生成器反馈二元标量（真/假），信息密度极低，容易导致模式崩溃。**FakeVLM、D3QE**等专用检测器虽在特定域表现优异，但仅在推理阶段借用生成模型的输出作为辅助信号，未能实现训练阶段的双向梯度流动。

这些方法的共同短板在于三个层面：第一，**特征空间不兼容**——生成器的VAE潜变量（zgen）与检测器的ViT视觉特征（hdet）处于不同表示空间，无法直接交互；第二，**反馈信号质量低劣**——二元对抗信号丢失了检测器关于"何处伪造"的空间定位信息；第三，**任务目标内在张力**——生成器追求逼真度与多样性，检测器追求对伪影的敏感性，简单联合训练易相互干扰。

本文的核心动机正是打破这一僵局：让检测器直接读取生成器的内部状态以理解伪影成因，同时让生成器在特征层面吸收检测器的取证知识以避免产生可检测痕迹——实现真正的"互利共生"。
![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/76662a56-098a-410b-b808-fe3c9d5674f2/figures/Figure_1.png)
*Figure 1: Figure 1. Our unified framework bridges generation and authen-ticity discrimination synergistically. (a) Generation enhances de-tection by reducing distributional gaps. (b) Detection feedback re-fines*



## 核心创新

核心洞察：生成器对图像分布的内部表示（VAE潜变量）天然包含伪造伪影的成因信息，而检测器对真实性的判断标准可以反向约束生成器避免产生可检测痕迹；因为共享同一LLM骨干（Qwen2.5）使得两个任务的特征空间天然兼容，从而使跨模态注意力机制与特征层分布对齐在语义上有意义地工作成为可能。

| 维度 | Baseline (BAGEL) | 本文 (UniGenDet) |
|:---|:---|:---|
| 信息交互方式 | 生成与检测模块独立运行，无显式交互 | SMSA模块：检测ViT特征通过交叉注意力直接感知VAE生成潜变量 |
| 检测→生成的反馈形式 | 无反向传递，或仅推理阶段辅助 | DIGA机制：第8层中间特征与检测最终层特征的显式分布对齐 |
| 反馈信号类型 | 不适用/二元标量 | 高维连续特征（替代GAN式标量反馈，规避模式崩溃） |
| 训练策略 | 单任务微调 | 两阶段协同：GDUF建立统一基础 → DIGA轻量特征对齐（500步） |

## 整体框架


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/76662a56-098a-410b-b808-fe3c9d5674f2/figures/Figure_2.png)
*Figure 2: Figure 2. Overview of the Generation–Detection Unified Fine-tuning (GDUF) pipeline. (a) Generative-Assisted Fake Detection andInterpretation: the Symbiotic Multi-modal Self-Attention (SMSA) guides the*



UniGenDet的整体架构基于BAGEL多模态基础模型，采用**两阶段协同训练**范式，数据流如下：

**输入端**：
- **生成任务**：文本指令 + 初始噪声 → VAE编码得到潜变量 zgen（空间维度适配512-1024分辨率，patch stride=16）
- **检测任务**：待测图像 → SigLIP视觉编码器提取ViT特征 hdet（输入尺寸224-980，stride=14）+ 文本查询指令

**第一阶段：Generation–Detection Unified Fine-tuning（GDUF）**
- **SMSA模块（核心结构创新）**：在LLM的注意力计算层中，检测任务的ViT特征 hdet 不仅执行全局自注意力，还通过**交叉注意力分支**额外感知VAE生成潜变量 zgen。三类token（VAE潜变量、ViT特征、文本指令）应用**任务特定的注意力掩码**：生成任务中zgen使用双向掩码（全可见），检测任务中按角色区分——zgen对hdet单向可见（生成信息辅助检测），hdet对zgen屏蔽（避免检测信息反向污染生成）。该阶段冻结LLM部分层，8×A100训练约1000步/12小时，建立统一多模态基础。

**第二阶段：Detector-Informed Generative Alignment（DIGA）**
- **特征对齐模块**：固定检测器参数，将生成器第8层中间特征与检测模块最终层输出进行**显式分布对齐**（L2特征距离，权重λ=0.5）。生成器由此学习"检测器认为何为真实"的连续表征，替代传统GAN的离散二元反馈。该阶段仅500步/6小时，轻量高效。

**输出端**：
- 生成任务：50步扩散采样输出图像
- 检测任务：真实性判断 + 伪造区域解释（文本生成，temperature=0.7）

```
文本指令 ──┬──→ [VAE编码] ──→ zgen ──┐
           │                          ↓
待测图像 ──→ [SigLIP] ──→ hdet ──→ [SMSA: 交叉注意力融合] ──→ [LLM Qwen2.5]
           │                              ↑                    │
           └──→ 文本token ────────────────┘                    │
                                                               ↓
                    ┌──────────────────────────────────── [检测输出: 真/假 + 解释]
                    └──────────────────────────────────── [生成输出: 图像]
                               ↑
                    [DIGA: 生成器第8层特征 ↔ 检测器最终层特征对齐]
```

## 核心模块与公式推导

### 模块 1: Symbiotic Multi-modal Self-Attention（SMSA）（对应框架图 Figure 2(a)）

**直觉**: 检测器若仅观察最终生成图像，只能捕获表面伪影；直接访问生成器的VAE潜变量，可获知"模型打算生成什么"的分布级信息，从而理解伪影的底层成因。

**Baseline 公式** (BAGEL标准自注意力): 
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

符号: $Q, K, V \in \mathbb{R}^{n \times d}$ 分别为查询、键、值矩阵；$d_k$ 为键维度；标准实现中生成与检测任务各自独立计算注意力，无跨任务交互。

**变化点**: BAGEL的注意力计算中，生成任务的zgen与检测任务的hdet处于隔离的token序列，注意力掩码仅控制同一序列内的可见性。本文**新增跨模态注意力路径**，使检测任务的查询可以 attend 到生成任务的键值。

**本文公式（推导）**:
$$\text{Step 1}: \quad Q_{\text{det}} = h_{\text{det}} W_Q, \quad K_{\text{gen}} = z_{\text{gen}} W_K, \quad V_{\text{gen}} = z_{\text{gen}} W_V$$
$$\text{加入了跨模态键值对以让检测器读取生成器内部状态}$$

$$\text{Step 2}: \quad \text{SMSA}(h_{\text{det}}, z_{\text{gen}}) = \text{softmax}\left(\frac{Q_{\text{det}}[K_{\text{det}}; K_{\text{gen}}]^T}{\sqrt{d_k}} + M\right)[V_{\text{det}}; V_{\text{gen}}]$$
$$\text{拼接同模态与跨模态键值，任务掩码} M \text{ 控制信息流方向}$$

$$\text{最终}: \quad h'_{\text{det}} = \text{LayerNorm}(h_{\text{det}} + \text{SMSA}(h_{\text{det}}, z_{\text{gen}}))$$

其中掩码矩阵 $M$ 的关键设计：检测任务中 $M_{i,j} = -\infty$ 若token $j$ 为检测特征且token $i$ 为生成潜变量（防止检测信息反向泄露），其余位置按标准因果/双向规则填充。

**对应消融**: 移除SMSA导致检测Acc下降3.0%、F1下降3.1%、解释生成ROUGE-L下降5.4分（Table 。

---

### 模块 2: Detector-Informed Generative Alignment（DIGA）（对应框架图 Figure 3）

**直觉**: 传统GAN向生成器传递标量判别损失 $\mathcal{L}_D \in \{0, 1\}$，梯度信号稀疏且易导致模式崩溃；将检测器的最终特征作为"真实分布的锚点"，让生成器在特征空间连续逼近，信息丰富且稳定。

**Baseline 公式** (标准对抗训练):
$$\min_G \max_D \mathcal{L}_{\text{adv}} = \mathbb{E}_{x \sim p_{\text{data}}} [\log D(x)] + \mathbb{E}_{z \sim p_z} [\log(1 - D(G(z)))]$$

符号: $G$ 为生成器，$D$ 为判别器，$p_{\text{data}}$ 为真实分布，$p_z$ 为噪声先验；$D(x) \in [0,1]$ 为标量概率输出。

**变化点**: 放弃二元博弈框架，固定预训练检测器 $\mathcal{D}$，将其最终层高维特征 $f_{\text{det}} = \mathcal{D}(x_{\text{real}})$ 作为目标分布，对齐生成器中间层特征 $f_{\text{gen}}^{(8)} = G^{(1:8)}(z)$。

**本文公式（推导）**:
$$\text{Step 1}: \quad f_{\text{det}} = \text{Pool}(\mathcal{D}_{\text{final}}(x_{\text{real}})) \in \mathbb{R}^{d_{\text{align}}}$$
$$\text{提取检测器最终层特征并池化至对齐维度}$$

$$\text{Step 2}: \quad f_{\text{gen}}^{(8)} = \text{Proj}(G^{(8)}(z_{\text{gen}})) \in \mathbb{R}^{d_{\text{align}}}$$
$$\text{投影生成器第8层特征至同一空间，选择第8层因深层语义与检测器兼容}$$

$$\text{Step 3}: \quad \mathcal{L}_{\text{DIGA}} = \| f_{\text{gen}}^{(8)} - f_{\text{det}} \|_2^2$$
$$\text{显式L2特征对齐，替代离散对抗损失}$$

$$\text{最终总损失}: \quad \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{diffusion}} + \lambda \cdot \mathcal{L}_{\text{DIGA}}, \quad \lambda = 0.5$$

**对应消融**: 无DIGA时完整模型FID=19.4（BAGEL+GDUF），加入DIGA后FID降至17.5，验证特征对齐对生成质量的独立贡献；同时检测性能在DIGA阶段后保持稳定，未出现灾难性遗忘。

## 实验与分析

| Method | FakeClue Acc | FakeClue F1 | DMImage Acc | DMImage F1 | FID ↓ |
|:---|:---|:---|:---|:---|:---|
| Qwen2-VL-72B | 57.8% | 56.5% | — | — | — |
| SIDA | — | — | 91.8% | 92.4% | — |
| BAGEL (基线) | — | — | — | — | 22.9 |
| BAGEL + GDUF | — | — | — | — | 19.4 |
| **UniGenDet (完整)** | **98.0%** | **97.7%** | **98.6%** | **99.1%** | **17.5** |


![Figure 5](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/76662a56-098a-410b-b808-fe3c9d5674f2/figures/Figure_5.png)
*Figure 5: Figure 4. Comparison of detection results. For each sample (left: generated, right: real), the UniGenDet (top) outperforms the pretrainedBAGEL (bottom), providing more accurate detection and superior*



**核心结论支撑**：FakeClue上98.0% Acc较Qwen2-VL-72B提升40.2个百分点，这一巨大gap源于SMSA让检测器获得了生成器内部状态访问权限，而非仅依赖表面视觉特征。DMImage跨数据集98.6% Acc验证泛化性，超SIDA +6.8%说明统一架构的跨域迁移优势。生成维度FID 17.5 vs BAGEL 22.9（相对改善23.6%），DIGA的特征对齐有效引导生成器避开检测器敏感的伪影模式。

**边际性能与权衡**：
- ARForensics零样本平均Acc 98.1%，超越FakeVLM（97.1%）和D3QE（82.1%），但在**LlamaGen子集上仅89.4%**，低于CNNSpot（99.9%），暴露自回归生成器域的检测盲区
- 多样性指标LPIPS 0.726 vs BAGEL 0.714，轻微下降但未模式崩溃；CLIP相似度0.802 vs 0.804，几乎持平
- GenEval上Two Object（0.95 vs 0.97）和Counting（0.80 vs 0.86）略低于BAGEL基线，存在生成-检测权衡

**消融分析**（
![Figure 11](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/76662a56-098a-410b-b808-fe3c9d5674f2/figures/Figure_11.png)

若可用）：移除SMSA导致检测Acc↓3.0%、F1↓3.1%、ROUGE-L↓5.4，为单一模块最大影响项；无GDUF时检测性能受限，验证两阶段必要性。

**公平性检查**：
- **基线强度**：生成质量仅与BAGEL系列变体比较，**缺乏与FLUX.1-dev、SD3等当前最强开源模型的直接FID对比**，基线选择存在局限性
- **计算成本**：总训练约18小时（12h+6h），8×A100，属中等规模微调，但BAGEL基线本身预训练成本未公开
- **失败案例**：LlamaGen域检测显著落后CNNSpot，说明SMSA的跨模态注意力对自回归生成伪影的感知能力不足；GenEval组合计数任务下降暗示检测约束可能损害复杂组合生成能力

## 方法谱系与知识库定位

**方法家族**: 多模态统一基础模型（Unified Multimodal Foundation Models）→ 生成-理解联合优化

**Parent Method**: **BAGEL**（基于Qwen2.5 LLM骨干的多模态生成-理解统一模型）。本文继承其LLM骨干（28层，隐层3584）和SigLIP视觉编码器（27层，隐层1152），在其上新增SMSA结构改造与DIGA训练策略。

**改动插槽**:
- **Architecture**: 新增SMSA跨模态注意力路径（检测查询→生成键值）
- **Objective**: 新增DIGA特征对齐损失 $\mathcal{L}_{\text{DIGA}}$（插件式扩展）
- **Training recipe**: 两阶段训练（GDUF统一微调 → DIGA轻量对齐）
- **Data curation**: 未明确改动，沿用BAGEL训练数据
- **Inference**: 保持50步扩散采样与temperature=0.7核采样，未改动

**直接基线对比**:
- **BAGEL**: 无跨任务交互，本文新增SMSA+DIGA双向信息流
- **FakeVLM/D3QE**: 专用检测器，本文统一架构实现参数共享与双向梯度流动
- **GAN-based检测器**: 二元标量反馈，本文高维连续特征对齐规避模式崩溃

**后续方向**:
1. **扩展至视频/3D生成检测**: SMSA的跨模态注意力机制可迁移至时空潜变量与视频检测器的交互
2. **自适应对齐权重**: 当前DIGA固定λ=0.5，探索任务难度感知的动态权重调度
3. **强化自回归域检测**: 针对LlamaGen子集失效问题，设计适配自回归生成器潜空间（非VAE）的SMSA变体

**知识库标签**: 
- **Modality**: image + text
- **Paradigm**: generative-discriminative co-evolution, unified fine-tuning
- **Scenario**: generated image detection, text-to-image generation, forensic analysis
- **Mechanism**: cross-modal attention (SMSA), feature alignment (DIGA), LLM-shared representation space
- **Constraint**: two-stage training, frozen detector in DIGA, limited AR-generator domain generalization

