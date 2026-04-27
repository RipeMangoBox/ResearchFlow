---
title: 'InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks'
type: paper
paper_level: C
venue: CVPR
year: 2024
paper_link: null
aliases:
- InternVL：视觉基础模型规模化与通用视觉语言对齐
- InternVL
acceptance: Oral
cited_by: 2715
method: InternVL
followups:
- 免训练知识桥接缺失模态补全_Knowledge_Bridge
- 开放语言-视觉模型的Scalin_Scaling_Law_Comp
---

# InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks

**Topics**: [[T__Retrieval]], [[T__Classification]], [[T__Visual_Question_Answering]] | **Method**: [[M__InternVL]] | **Datasets**: [[D__ImageNet-1K]] (其他: Multi-Modal Dialogue)

| 中文题名 | InternVL：视觉基础模型规模化与通用视觉语言对齐 |
| 英文题名 | InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks |
| 会议/期刊 | CVPR 2024 (Oral) |
| 链接 | [arXiv](https://arxiv.org/abs/2312.14238) · [Code](https://github.com/OpenGVLab/InternVL) · [Project](https://internvl.opengvlab.com) |
| 主要任务 | 视觉语言预训练、零样本图像/视频分类、图像文本检索、多模态对话、语义分割 |
| 主要 baseline | CLIP、LLaVA-1.5、EVA-E、BEiT |

> [!abstract] 因为「现有视觉语言模型存在视觉编码器规模不足（通常≤1B参数）与视觉-语言特征对齐简单（线性投影）导致的表征能力瓶颈」，作者在「CLIP 对比学习框架」基础上改了「将视觉编码器扩展至 6B 参数（InternViT-6B），并引入基于可学习查询的交叉注意力语言中间件 QLLaMA 替代线性投影」，在「ImageNet-1K linear probing、Tiny LVLM 多模态对话等 benchmark」上取得「无需 JFT 数据的最佳线性评估性能，以及与 off-the-shelf LLM 更好的特征兼容性」。

- **关键性能**：InternViT-6B 在 ImageNet-1K linear probing 上达到无需 JFT 数据的 SoTA；Stage 1 使用 640×A100 GPU 训练 28.7B 样本
- **关键性能**：QLLaMA 作为 glue layer 在三种任务上均带来显著提升，相比直接连接 InternViT-6B 与 LLM 的极简设置表现更优
- **关键性能**：InternVL-C (224×224) 单 A100 编码图文对达 48.9 FPS，InternVL-G (448×448) 达 11.8 FPS

## 背景与动机

当前视觉语言（Vision-Language, VL）领域面临一个核心矛盾：视觉编码器的规模与能力严重滞后于语言模型。以 CLIP 为代表的经典方法采用双编码器架构（ViT-L/14 约 300M 参数 + 文本编码器），虽在零样本迁移上取得突破，但当面对复杂的多模态推理任务时，视觉侧的信息容量成为瓶颈。与此同时，LLaVA-1.5 等视觉指令调优方法通过简单的线性投影层将视觉特征映射到 LLM 输入空间，这种「硬对齐」方式难以充分利用大规模视觉编码器的细粒度表征。

具体而言，现有方法存在三方面局限：其一，**视觉编码器规模不足**——EVA-E 等当前较大视觉模型也仅约 1B 参数，与 LLaMA-7B/13B 等语言模型严重不对等；其二，**对齐机制过于简单**——LLaVA-1.5 的 MLP projector 是静态线性变换，无法根据语言任务动态提取视觉信息；其三，**训练策略缺乏渐进性**——单阶段对比学习难以同时优化表征学习与跨模态对齐。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/a8d63a1f-37f1-4e34-b894-0dfcc3cf1343/figures/Figure_1.png)
*Figure 1: Figure 1. Comparisons of different vision and vision-language foundation models. (a) indicates the traditional vision foundation model,e.g. ResNet [57] pre-trained on classification tasks. (b) represe*



如图 1 所示，传统视觉基础模型（如 BEiT、EVA）与视觉语言模型（如 CLIP、BLIP）分属不同范式，而 InternVL 旨在构建统一的规模化基础模型。作者提出通过三阶段渐进训练，先以对比学习训练超大视觉编码器，再引入 QLLaMA 实现深度特征对齐，最终支持灵活的下游任务适配。

## 核心创新

核心洞察：**视觉编码器的规模必须与语言模型对等，且视觉-语言对齐需要动态交叉注意力机制而非静态投影**，因为可学习查询能够根据语言上下文自适应地聚合视觉信息，从而使 6B 参数视觉编码器的细粒度表征能被 off-the-shelf LLM 有效理解成为可能。

| 维度 | Baseline (CLIP / LLaVA-1.5) | 本文 (InternVL) |
|:---|:---|:---|
| 视觉编码器规模 | ViT-L/14 (~300M) 或 EVA-E (~1B) | **InternViT-6B (6B 参数)**，depth=48, head_dim=128, MLP ratio=8 |
| 视觉-语言对齐 | 双编码器对比 (CLIP) 或线性投影 (LLaVA) | **QLLaMA**：可学习查询 + 交叉注意力层的语言中间件 |
| 训练策略 | 单阶段对比学习或两阶段指令调优 | **三阶段渐进训练**：对比预训练 → 冻结编码器+交叉注意力训练 → SFT |
| 分辨率策略 | 固定分辨率 (如 224×224) | **渐进分辨率**：196×196 (50% token masking) → 224×224 |
| 与 LLM 兼容性 | 需针对特定 LLM 微调投影层 | QLLaMA 作为通用 glue layer，兼容多种 off-the-shelf LLM |

## 整体框架


![Figure 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/a8d63a1f-37f1-4e34-b894-0dfcc3cf1343/figures/Figure_4.png)
*Figure 4: Figure 4. Different ways to use InternVL. By flexibly combining the vision encoder and the language middleware, InternVL can supportvarious vision-language tasks, including contrastive tasks, generati*



InternVL 的整体架构如图 4 所示，包含四种灵活配置，核心数据流为：**图像 → InternViT-6B → [QLLaMA 交叉注意力 / MLP 投影] → LLM → 文本输出**。

**模块分解**：

1. **InternViT-6B（图像编码器）**：输入 224×224 图像 patch，输出视觉特征 token。采用 depth=48、head dimension=128、MLP ratio=8 的架构，总参数量 6B，是截至发表时最大的单一视觉编码器。使用 BEiT 初始化方法随机初始化。

2. **QLLaMA（语言中间件）**：输入 InternViT-6B 输出的视觉特征 token，以及文本 token；通过**可学习查询（learnable queries）作为 Query**，以视觉特征作为 Key/Value，执行交叉注意力（Cross-Attention），输出与 LLM 表示空间对齐的多模态特征。替代了 LLaVA-1.5 中的简单线性投影层。

3. **LLM 解码器（LLaMA-7B / InternLM）**：输入来自 QLLaMA 的对齐特征（或直接来自 InternViT-6B 的 MLP 投影特征），输出生成文本。Stage 3 提供两种配置：w/o QLLaMA（直接连接，遵循 LLaVA-1.5 训练配方）和 w/ QLLaMA（通过语言中间件增强对齐）。

4. **MLP Projector（备用投影层）**：在不含 QLLaMA 的配置中使用，执行简单的线性变换将视觉特征映射到 LLM 输入空间。

**三阶段训练流程**：

```
Stage 1 (对比预训练):  图像+文本 → InternViT-6B + LLaMA-7B(文本编码器) → 对比损失
                        [28.7B 样本, 640×A100, 196²→224² 渐进分辨率, 50% token masking]
                        ↓
Stage 2 (对齐训练):    冻结 InternViT-6B 和 QLLaMA，仅训练新增交叉注意力参数
                        [1.6B 样本, 160×A100]
                        ↓
Stage 3 (监督微调):    配置 A: InternViT-6B → MLP → LLM (LLaVA-1.5 风格)
                        配置 B: InternViT-6B → QLLaMA → LLM (本文完整配置)
                        [高质量指令数据, 32×A100]
```

## 核心模块与公式推导

### 模块 1: 对比学习损失（Stage 1，对应框架图左支）

**直觉**：通过最大化匹配图像-文本对的相似度、最小化非匹配对的相似度，在 28.7B 大规模数据上对齐视觉与语言表示。

**Baseline 公式 (CLIP)**:
$$\mathcal{L}_{CLIP}(I, T) = -\log \frac{\exp(\text{sim}(f_I(I), f_T(T))/\tau)}{\sum_{T'} \exp(\text{sim}(f_I(I), f_T(T'))/\tau)}$$
符号: $f_I$ = 图像编码器 (ViT-L/14), $f_T$ = 文本编码器, $\tau$ = 温度系数, sim = 余弦相似度

**变化点**: CLIP 的视觉编码器仅 ~300M 参数，难以捕捉复杂视觉模式；且使用固定温度系数。InternVL 将视觉编码器扩展至 6B 参数，并采用 BEiT 初始化 + 渐进分辨率策略提升训练稳定性。

**本文公式**：
$$\text{Step 1}: \quad f_{InternViT}(I) = \text{InternViT-6B}(\text{PatchEmbed}(I)) \quad \text{[6B 参数视觉编码，提取深层特征]}$$
$$\text{Step 2}: \quad f_{LLaMA}(T) = \text{LLaMA-7B}_{enc}(T) \quad \text{[7B 参数语言编码，与视觉侧对等规模]}$$
$$\text{最终}: \quad \mathcal{L}_{contrastive} = -\frac{1}{2}\left[\log \frac{\exp(\text{sim}(f_{InternViT}(I), f_{LLaMA}(T))/\tau)}{\sum_{T'} \exp(\text{sim}(f_{InternViT}(I), f_{LLaMA}(T'))/\tau)} + \log \frac{\exp(\text{sim}(f_{InternViT}(I), f_{LLaMA}(T))/\tau)}{\sum_{I'} \exp(\text{sim}(f_{InternViT}(I'), f_{LLaMA}(T))/\tau)}\right]$$
**对应消融**：Table 11 显示 InternViT-6B 在 accuracy、inference speed、training stability 的权衡中选择 variant 3（depth=48, head_dim=128, MLP=8）。

### 模块 2: QLLaMA 交叉注意力对齐（Stage 2，对应框架图中部）

**直觉**：静态线性投影无法根据语言任务动态选择视觉信息；可学习查询能像「注意力探针」一样，主动从视觉特征中提取 LLM 需要的内容。

**Baseline 公式 (LLaVA-1.5)**:
$$h = W \cdot f_{ViT}(I) + b \quad \text{(简单线性投影)}$$
符号: $W \in \mathbb{R}^{d_{LLM} \times d_{ViT}}$ = 投影矩阵, $b$ = 偏置, $f_{ViT}(I)$ = 视觉特征

**变化点**: 线性投影是「一对多」的静态映射，所有语言任务共享相同视觉表示；QLLaMA 通过交叉注意力实现「多对多」动态映射，可学习查询 $Q_{learnable}$ 针对不同语言上下文提取不同视觉信息。

**本文公式（推导）**:
$$\text{Step 1}: \quad Q = Q_{learnable} \in \mathbb{R}^{N_q \times d}, \quad K = V = f_{InternViT}(I) \in \mathbb{R}^{N_{patch} \times d} \quad \text{[可学习查询作为 Query，视觉特征作为 KV]}$$
$$\text{Step 2}: \quad \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V \quad \text{[标准缩放点积注意力，动态加权视觉特征]}$$
$$\text{Step 3}: \quad Q_{out} = \text{LayerNorm}(\text{Attention}(Q, K, V) + Q) \quad \text{[残差连接 + 层归一化，稳定训练]}$$
$$\text{Step 4}: \quad h = \text{FFN}(Q_{out}) \quad \text{[前馈网络进一步变换，适配 LLM 输入空间]}$$
$$\text{最终}: \quad h_{QLLaMA} = \text{CrossAttnBlock}(Q_{learnable}, f_{InternViT}(I), f_{InternViT}(I)) \in \mathbb{R}^{N_q \times d_{LLM}}$$

**对应消融**：Table 12 显示使用 QLLaMA 构建多模态对话系统时，相比直接连接（w/o QLLaMA）在多个任务上均有提升。更关键的是，Table 17（Tiny LVLM 测试）的消融表明，在极简设置下（仅 MLP 可训练），EVA-E 表现弱于 InternViT-6B，而加入 QLLaMA 后三者任务性能均显著改善。

### 模块 3: 渐进分辨率训练（Stage 1 优化策略）

**直觉**：直接以高分辨率训练大模型计算昂贵；先低分辨率+高掩码率快速学习全局结构，再切换高分辨率精调细节。

**Baseline 策略**: 固定 224×224 分辨率从头训练（标准 CLIP/BEiT 做法）

**变化点**: 大模型在高分辨率下训练收敛慢、显存占用高；通过 50% token masking 在低分辨率下模拟信息缺失，强迫模型学习鲁棒表征。

**本文策略**:
$$\text{Step 1}: \quad I_{196} = \text{Resize}(I, 196 \times 196), \quad M \sim \text{Bernoulli}(0.5) \quad \text{[50% 随机掩码视觉 token]}$$
$$\text{Step 2}: \quad \mathcal{L}_{stage1a} = \mathcal{L}_{contrastive}(f_{InternViT}(I_{196} \odot M), f_{LLaMA}(T)) \quad \text{[掩码训练阶段，约 28.2B 样本]}$$
$$\text{Step 3}: \quad I_{224} = \text{Resize}(I, 224 \times 224), \quad M = \mathbf{1} \quad \text{[切换 224×224，无掩码]}$$
$$\text{最终}: \quad \mathcal{L}_{stage1b} = \mathcal{L}_{contrastive}(f_{InternViT}(I_{224}), f_{LLaMA}(T)) \quad \text{[精调阶段，约 0.5B 样本]}$$
**对应消融**：Table 20 记录训练设置，196²→224² 的渐进策略配合 50% masking 是 Stage 1 的核心效率优化。

## 实验与分析


![Table 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/a8d63a1f-37f1-4e34-b894-0dfcc3cf1343/figures/Table_4.png)
*Table 4: Table 4. Linear evaluation on image classification. We report thetop-1 accuracy on ImageNet-1K [38] and its variants [10, 60, 61,119, 141]. ∗ViT-22B [37] uses the private JFT-3B dataset [173].*



本文在多个 benchmark 上评估 InternVL，涵盖视觉感知、零样本迁移、多模态对话三大类任务。Table 4 显示，InternViT-6B 在 ImageNet-1K linear probing 上取得无需 JFT 数据的最佳性能（具体数值见原表），显著超越此前 SoTA。Table 6 的零样本图像分类中，InternVL 在 20 个数据集上的平均表现优异，"∆↓"（IN-1K 准确率与平均准确率的差距）指标显示其分布外泛化能力强劲。Table 10 汇总了 9 个 benchmark 的对比，涵盖 COCO/Flickr30K 图像描述、VQAv2/OK-VQA 视觉问答等，InternVL 在多数任务上达到或接近 SoTA。


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/a8d63a1f-37f1-4e34-b894-0dfcc3cf1343/figures/Figure_2.png)
*Figure 2: Figure 2. Comparison results on various generic visual-linguistic tasks, including image classification, video classification, image-textretrieval, image captioning, and multi-modal dialogue. The prop*



视频理解方面，Table 8 报告 Kinetics 400/600/700 的零样本视频分类结果，InternVL 展现出从图像-文本预训练到视频时序建模的有效迁移。多语言场景下，Table 13 在 XTD 数据集上的零样本多语言图像-文本检索验证了模型的跨语言能力。效率层面，Table 19 显示 InternVL-C (224×224) 编码图文对仅需 20.4ms（48.9 FPS），InternVL-G (448×448) 为 84.6ms（11.8 FPS），在单 A100 上具备实用部署速度。


![Table 12](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/a8d63a1f-37f1-4e34-b894-0dfcc3cf1343/figures/Table_12.png)
*Table 12: Table 12. Ablation studies of using InternVL to build multi-modal dialogue system. V-7B and V-13B denote Vicuna-7B/13B[184], respectively. “IViT-6B” represents our InternViT-6B.*



消融实验聚焦 QLLaMA 的有效性。Table 12 对比了 InternVL-Chat 有/无 QLLaMA 的配置：使用 Vicuna-7B/13B 作为 LLM 后端时，QLLaMA 的引入在多模态对话能力上带来一致增益。更严格的控制在 Table 17（Tiny LVLM 测试集）中：当采用「极简设置」——仅训练 MLP 层、冻结视觉编码器和 LLM——EVA-E 的表现弱于 InternViT-6B，而 QLLaMA 作为 glue layer 在视觉推理（VR）、视觉编码（VE）、视觉问答（VQ）等五类能力上均带来显著提升。Table 11 的架构搜索从 16 个候选变体中筛选出 6 个，最终选定 variant 3（depth=48, head_dim=128, MLP=8）基于准确率、推理速度和训练稳定性的帕累托最优。

公平性检查：对比的 baselines（CLIP、LLaVA-1.5、EVA-E）均为领域内代表性方法，但缺少与 PaLI-3/PaLI-X、Flamingo、GPT-4V 的直接对比——后者在论文发表时或未开源或不可获取。Stage 1 使用的 28.7B 样本来自多个数据源（Table 2），其筛选和去重细节未完全披露，精确复现存在挑战。效率测试采用 Flash Attention 和 bf16，部分 baselines 可能未使用同等优化。训练成本方面，Stage 1 消耗 640×A100 GPU，Stage 2 160×A100，Stage 3 32×A100，总计算预算显著高于学术实验室常规配置。

## 方法谱系与知识库定位

InternVL 属于 **CLIP 对比学习谱系** 的规模化延伸，核心演进路径为：CLIP（双编码器对比）→ 大规模视觉编码器 + 深度对齐机制 → 统一视觉语言基础模型。

**父方法**：**CLIP**（对比语言-图像预训练）。InternVL 继承其图像-文本对比学习的核心范式，但在四个关键 slot 上发生质变：
- **架构**：CLIP ViT-L/14 (~300M) → **InternViT-6B (6B)**，并新增 **QLLaMA** 交叉注意力语言中间件替代线性投影
- **训练配方**：单阶段对比学习 → **三阶段渐进训练**（对比预训练 → 冻结编码器+交叉注意力训练 → SFT）
- **数据治理**：LAION-400M 级别 → **28.7B 精心筛选样本**，配合渐进分辨率与 token masking
- **推理策略**：直接特征投影 → **QLLaMA 动态对齐**，作为通用 glue layer 兼容多种 off-the-shelf LLM

**直接 baselines 差异**：
- **vs CLIP**：视觉编码器扩大 20×，文本编码器采用 LLaMA-7B 对等规模，训练数据扩大 70×+
- **vs LLaVA-1.5**：Stage 3 训练配方相同，但 InternVL 提供 QLLaMA 替代 MLP projector 的增强配置
- **vs EVA-E**：InternViT-6B 参数规模更大，且通过 QLLaMA 解决与 LLM 的特征空间不一致问题

**后续方向**：(1) 进一步扩展视觉编码器至 10B+ 参数，探索与 LLaMA-70B 等更大语言模型的对齐；(2) QLLaMA 机制向视频、3D 等多模态场景迁移；(3) 降低三阶段训练的计算门槛，开发更高效的渐进训练策略。

**标签**：modality=图像+文本+视频 | paradigm=对比预训练+指令微调 | scenario=通用视觉语言任务 | mechanism=可学习查询交叉注意力对齐 | constraint=大规模计算资源需求（640×A100 Stage 1）

## 引用网络

### 后续工作（建立在本文之上）

- [[P__免训练知识桥接缺失模态补全_Knowledge_Bridge]]: Vision-language foundation model, commonly used as baseline in multimodal experi
- [[P__开放语言-视觉模型的Scalin_Scaling_Law_Comp]]: Large vision-language model; likely appears in experimental comparisons as a str

