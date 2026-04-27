---
title: The Indra Representation Hypothesis
type: paper
paper_level: C
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 基于Yoneda嵌入的关系型表示学习
- Indra Representa
- Indra Representation
- Representations from unimodal found
acceptance: Poster
code_url: https://github.com/Jianglin954/Indra
method: Indra Representation
modalities:
- Text
- Image
- Audio
paradigm:
- training-free inference
- training-free (post-hoc representation transformation)
- training-free inference-only
---

# The Indra Representation Hypothesis

[Code](https://github.com/Jianglin954/Indra)

**Topics**: [[T__Self-Supervised_Learning]], [[T__Cross-Modal_Matching]], [[T__Retrieval]] | **Method**: [[M__Indra_Representation]] | **Datasets**: [[D__CIFAR-10]], [[D__CIFAR-100]], [[D__Office-Home]] (其他: Image-text datasets, Audio-text dataset)

> [!tip] 核心洞察
> Representations from unimodal foundation models implicitly reflect a shared relational structure underlying reality, which can be revealed by representing each sample through its relational profile to other samples rather than as an independent embedding.

| 中文题名 | 基于Yoneda嵌入的关系型表示学习 |
| 英文题名 | The Indra Representation Hypothesis |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.04496) · [Code](https://github.com/Jianglin954/Indra) · [Project](未提供) |
| 主要任务 | Cross-Modal Retrieval, Cross-Modal Matching, 鲁棒性分类 |
| 主要 baseline | Standard embedding representations, CLIP-style contrastive alignment, BERT, ViT |

> [!abstract] 因为「单模态基础模型学到的独立样本嵌入缺乏关系结构表达能力，跨模态对齐需要昂贵的重训练」，作者在「标准嵌入表示」基础上改了「用V-富化Yoneda嵌入将样本表示为与其他所有样本的角距离关系轮廓」，在「CIFAR-10/100、Office-Home及图像-文本/音频-文本跨模态任务」上取得「training-free的鲁棒性提升与跨模态对齐」。

- **Training-free**: 无需任何重训练或对比损失，直接对预训练模型嵌入进行后处理
- **跨模态对齐**: 在图像-文本、音频-文本任务上实现零训练跨模态匹配
- **噪声鲁棒性**: 在CIFAR-10/100高斯噪声场景下验证表示鲁棒性

## 背景与动机

当前单模态基础模型（如BERT、ViT、wav2vec 2.0）在不同架构、目标函数和数据模态下呈现出惊人的表示收敛现象——它们似乎都在捕捉某种共享的底层现实结构。然而，这些模型输出的表示被当作独立的嵌入向量使用：每个样本 $x_i$ 被表示为固定的 $f(x_i) \in \mathbb{R}^d$，与其他样本的关系仅在下游任务的隐式学习中被间接利用。

现有方法如何处理这一问题？**标准嵌入表示**直接将预训练输出作为最终表示，强调个体信息而忽视关系模式；**CLIP-style对比学习**通过大规模对比损失训练实现跨模态对齐，但需要昂贵的重训练和外部的对齐机制与融合模块；**SimCLR/BYOL/MoCo**等自监督方法虽利用样本间关系进行预训练，但最终仍输出独立嵌入，未在推理阶段显式编码关系结构。

这些方法的共同短板在于：将模型输出视为"终点"而非"关系网络中的节点"。一旦预训练完成，样本间丰富的结构信息就被压缩进孤立的向量中，跨模态对齐必须依赖额外的训练来弥补这一信息损失。本文受此启发，提出一个根本性的重新思考：**样本的本质不在于其孤立属性，而在于它与其他所有样本的关系网络中所处的位置**——这正是"因陀罗网"（Indra's Net）的哲学隐喻，每一颗宝珠都映现全网。

本文将这一直觉形式化为严格的数学框架：用范畴论的V-富化Yoneda嵌入，将每个样本表示为它到所有其他样本的代价向量，从而在不重新训练任何模型的情况下，实现跨架构、跨模态的结构保持对齐。

## 核心创新

核心洞察：样本应由其与其他所有样本的关系轮廓来定义，而非独立嵌入向量，因为Yoneda引理保证对象可由其与所有对象的关系完全刻画，从而使免训练的跨模态结构保持对齐成为可能。

| 维度 | Baseline (标准嵌入) | 本文 (Indra Representation) |
|:---|:---|:---|
| **表示定义** | 独立向量 $f(x_i) \in \mathbb{R}^d$ | 关系轮廓 $d(\cdot, X_i) \in [0,\infty]^n$，即样本到所有其他样本的角距离向量 |
| **数学基础** | 欧氏空间中的点 | V-富化Yoneda嵌入：$Y: \mathcal{C} \to [\mathcal{C}^{op}, \mathcal{V}]$，范畴论保证唯一性、完备性、结构保持性 |
| **跨模态对齐** | 需对比损失重训练（如CLIP） | Training-free：直接匹配关系轮廓，无需任何训练 |
| **推理计算** | 单样本前向 $O(d)$ | 批次成对距离 $O(n^2 \cdot d)$，数据集级别关系矩阵 |

## 整体框架



Indra Representation的整体流程包含四个阶段，形成从原始数据到跨模态对齐表示的完整pipeline：

**阶段一：预训练编码器（Pretrained Encoder）**。输入原始样本（图像/文本/音频），输出标准嵌入 $f(x) \in \mathbb{R}^d$。此处直接使用现有预训练模型（如BERT、ViT、Qwen、WavLM），不做任何修改或重训练。

**阶段二：成对角距离计算（Pairwise Angular Distance Computation）**。输入批次嵌入 $\{f(x_i)\}_{i=1}^n$，输出距离矩阵 $\mathbf{D} \in [0, \pi]^{n \times n}$，其中 $D[i,j] = d(x_i, x_j) = \arccos\left(\frac{f(x_i) \cdot f(x_j)}{\|f(x_i)\| \|f(x_j)\|}\right)$。

**阶段三：关系轮廓构建（Relational Profile Construction via Yoneda Embedding）**。输入距离矩阵 $\mathbf{D}$，输出每个样本的关系轮廓——即 $\mathbf{D}$ 的对应列（或行）$d(\cdot, X_i) = [d(X_1, X_i), d(X_2, X_i), \ldots, d(X_n, X_i)]^\text{top}$。这一步是Yoneda嵌入的实例化：样本 $X_i$ 被表示为函子 $h_{X_i}: \mathcal{C}^{op} \to \mathcal{V}$，其在对象 $X_j$ 上的值为 $h_{X_i}(X_j) = d(X_j, X_i)$。

**阶段四：免训练对齐（Training-free Alignment）**。输入来自不同模态或不同模型的关系轮廓，输出对齐后的表示用于跨模态检索/匹配。对齐通过直接比较关系轮廓实现，无需对比损失或投影层。

```
Raw Sample x ──► [Encoder f] ──► Embedding f(x) ──► [Angular Distance] ──►
                                                                              │
                                                                              ▼
Cross-Modal Match ◄── [Alignment] ◄── Relational Profile d(·,X_i) ◄── [Yoneda Embedding] ◄── Distance Matrix D
```

## 核心模块与公式推导

### 模块 1: V-富化Yoneda嵌入（对应框架图 阶段三）

**直觉**：Yoneda引理的核心思想是"一个对象由其与所有对象的关系完全决定"——将这一思想从集合范畴推广到代价范畴，使样本表示为关系轮廓而非独立向量。

**Baseline 公式** (标准嵌入表示): $$f(x_i) \in \mathbb{R}^d$$
符号: $f$ = 预训练编码器, $d$ = 嵌入维度, 无显式关系信息。

**变化点**: 标准嵌入将样本压缩为固定维度的孤立点，丢失了样本在数据分布中的结构性位置；Yoneda引理提示我们，对象的完整信息蕴含在其与所有对象的态射集合中。

**本文公式（推导）**:
$$\text{Step 1 (Yoneda引理)}: \text{Nat}(h_A, F) \cong F(A) \quad \text{对象} A \text{由其hom-函子} h_A = \text{Hom}(A,-) \text{完全刻画}$$
$$\text{Step 2 (V-富化推广)}: \mathcal{V} = ([0,\infty], \geq, 0, +) \text{ 作为代价范畴，将Set替换为} \mathcal{V}\text{-enriched结构}$$
$$\text{Step 3 (样本范畴实例化)}: \mathcal{C}(X_i, X_j) = d(X_i, X_j) \in [0,\infty] \quad \text{hom-对象为两样本间的代价}$$
$$\text{最终 (Yoneda嵌入)}: Y(X_i) = h_{X_i}: \mathcal{C}^{op} \to \mathcal{V}, \quad h_{X_i}(X_j) = \mathcal{C}(X_j, X_i) = d(X_j, X_i)$$

**对应消融**: Table 3 和 Table 4 显示使用原始表示（Org./O）与Indra表示的对比，验证关系轮廓的必要性。

---

### 模块 2: 角距离代价函数（对应框架图 阶段二）

**直觉**：预训练嵌入空间具有非线性几何结构，欧氏距离未能充分利用角度信息；角距离在超球面上自然度量方向相似性，更适合神经网络嵌入的语义结构。

**Baseline 公式** (标准距离): $$\|f(x_i) - f(x_j)\|_2 \text{ 或内积相似度 } f(x_i) \cdot f(x_j)$$
符号: $\|\cdot\|_2$ = 欧氏范数, 内积 = 未归一化的方向+幅度混合度量。

**变化点**: 欧氏距离受嵌入幅度影响，内积同时依赖幅度与方向；角距离剥离幅度仅保留方向，使关系轮廓对嵌入尺度不变，且与余弦相似度的几何解释一致。

**本文公式（推导）**:
$$\text{Step 1 (归一化)}: \hat{f}(x_i) = \frac{f(x_i)}{\|f(x_i)\|}, \quad \hat{f}(x_j) = \frac{f(x_j)}{\|f(x_j)\|} \quad \text{投影到单位超球面}$$
$$\text{Step 2 (内积即余弦)}: \hat{f}(x_i) \cdot \hat{f}(x_j) = \cos\theta_{ij} \in [-1, 1]$$
$$\text{Step 3 (反余弦得角距离)}: d(x_i, x_j) = \arccos\left(\frac{f(x_i) \cdot f(x_j)}{\|f(x_i)\| \|f(x_j)\|}\right) = \arccos(\cos\theta_{ij}) = \theta_{ij} \in [0, \pi]$$
$$\text{最终}: d(x_i, x_j) = \arccos\left(\frac{f(x_i) \cdot f(x_j)}{\|f(x_i)\| \|f(x_j)\|}\right)$$

**对应消融**: Table 3 中随机表示（R）、纯语言表示（L）、纯文本表示（T）等变体与原始表示（Org.）及Indra表示的对比，显示不同代价函数选择的影响。

---

### 模块 3: 免训练对齐机制（对应框架图 阶段四）

**直觉**：若不同模态的预训练模型确实捕捉到共享的底层现实结构，则它们的关系轮廓应在适当匹配下呈现一致性——无需重新学习映射。

**Baseline 公式** (CLIP-style对比学习): $$\mathcal{L}_{\text{contrastive}} = -\log\frac{\exp(\text{sim}(z_i^A, z_i^B)/\tau)}{\sum_j \exp(\text{sim}(z_i^A, z_j^B)/\tau)}$$
符号: $z^A, z^B$ = 不同模态的投影嵌入, $\text{sim}$ = 相似度函数, $\tau$ = 温度参数, 需训练投影层。

**变化点**: 对比学习需要大规模配对的跨模态数据和昂贵的训练；Indra假设预训练模型已隐含收敛到共享结构，直接对齐其关系轮廓即可。

**本文公式（推导）**:
$$\text{Step 1 (提取关系轮廓)}: \text{对于模态} A: \mathbf{r}_i^A = [d^A(X_1, X_i), \ldots, d^A(X_n, X_i)]^\text{top}$$
$$\text{Step 2 (跨模态轮廓匹配)}: \text{sim}(X_i^A, X_j^B) = \text{corr}(\mathbf{r}_i^A, \mathbf{r}_j^B) \text{ 或排序一致性度量}$$
$$\text{最终 (Training-free alignment)}: \hat{Y} = \text{arg}\max_j \, \text{sim}(\mathbf{r}_i^A, \mathbf{r}_j^B)$$

**对应消融**: Table 4 中音频-文本数据集上原始表示（O）、纯文本表示（T）、纯音频表示（A）与Indra表示的对比，验证跨模态轮廓匹配的有效性。

## 实验与分析


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/eee74516-923d-4002-b74d-aea0c42a0122/figures/Table_1.png)
*Table 1 (comparison): Accuracy (%) on CIFAR-10 and CIFAR-100 under different Gaussian noise levels*



本文在三个层面验证Indra Representation的有效性：图像分类鲁棒性、跨域泛化、以及跨模态对齐。

**图像分类鲁棒性（Table 1）**：在CIFAR-10和CIFAR-100上，作者评估了不同高斯噪声水平下Indra表示与原始嵌入的分类准确率。Indra表示通过关系轮廓编码了样本在数据分布中的相对位置，对噪声扰动表现出更强的稳定性。具体而言，随着噪声标准差增大，基于独立嵌入的基线方法性能显著下降，而Indra表示利用全局关系结构进行"去噪"——噪声样本的关系轮廓仍能保持与干净样本的结构性对应。

**跨域鲁棒性（Table 2）**：在Office-Home数据集的四个域（Art, Clipart, Product, Real-World）上，Indra表示同样展现了高斯噪声下的鲁棒性。这一结果支持了核心假设：关系轮廓捕获的跨样本结构比孤立嵌入更能抵抗域偏移和扰动。


![Table 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/eee74516-923d-4002-b74d-aea0c42a0122/figures/Table_3.png)
*Table 3 (ablation): Performance on image-to-text domain P using different representations X (Org. original, R: random, L: language-only, T: text-only, I: image-only, U: unit, and B: blank)*



**跨模态对齐消融（Table 3 & Table 4）**：Table 3在图像-文本域上比较了不同表示变体：原始表示（Org.）、随机表示（R）、纯语言表示（L）、纯文本表示（T）、纯图像表示（I）与Indra表示。Table 4在音频-文本数据集上比较了原始（O）、纯文本（T）、纯音频（A）、单元表示（U）与Indra表示。这些消融表明，仅使用单模态关系信息（如仅图像或仅文本）不足以实现有效对齐，而Indra表示通过完整的V-富化Yoneda嵌入整合了多模态关系结构。

**公平性检查**：本文的baseline主要为"原始嵌入表示"，未与当前最强的端到端训练跨模态模型（如CLIP）进行直接数值对比，这是一个局限。计算开销方面，成对距离计算的$O(n^2)$复杂度在大数据集上可能成为瓶颈。此外，作者明确承认理论框架假设代价函数满足度量性质，而实际嵌入空间未必严格满足；角距离的选择虽有几何直觉，但最优代价函数的探索仍不充分。

## 方法谱系与知识库定位

**方法家族**：Indra Representation属于**关系型表示学习（relational representation learning）**家族，其直接理论渊源为**Yoneda embedding（范畴论）**，通过V-富化推广和角距离实例化，从纯数学构造转化为可计算框架。

**改动槽位**：
- **representation_definition**（替换）：独立嵌入 → 关系轮廓
- **inference_strategy**（修改）：直接使用嵌入 → 成对距离矩阵+Yoneda嵌入+免训练对齐
- **data_pipeline**（修改）：单样本前向 → 批次成对关系计算
- **architecture**（不变）：复用现有编码器，无结构改动

**直接Baseline对比**：
- **Standard embedding representations**: 本文将孤立向量替换为关系轮廓，是核心对立面
- **CLIP-style contrastive alignment**: 本文免训练实现跨模态对齐，无需对比损失
- **SimCLR/BYOL/MoCo**: 本文继承其"样本关系重要"的直觉，但在推理阶段而非训练阶段显式编码关系

**后续方向**：(1) 开发近似算法降低$O(n^2)$成对计算复杂度，使Indra表示可扩展至大规模数据；(2) 探索除角距离外的最优代价函数，如学习得到的自适应度量；(3) 将关系轮廓思想与图神经网络结合，在显式图结构上实现更高效的关系编码。

**标签**：modality=多模态(vision/language/audio) | paradigm=免训练后处理(training-free inference-only) | scenario=跨模态检索/匹配/鲁棒分类 | mechanism=范畴论Yoneda嵌入+角距离关系轮廓 | constraint=无需重训练、无需配对跨模态数据、计算开销随数据集规模二次增长

