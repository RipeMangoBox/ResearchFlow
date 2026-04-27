---
title: 'Bridging the Divide: Reconsidering Softmax and Linear Attention'
type: paper
paper_level: C
venue: NeurIPS
year: 2024
paper_link: null
aliases:
- InLine：可注入性与局部建模的线性注意力
- InLine
- Linear attention's inferior perform
acceptance: Poster
cited_by: 8
code_url: https://github.com/LeapLabTHU/InLine
method: InLine
modalities:
- Image
paradigm: supervised
baselines:
- 代理注意力：融合Softmax与_Agent_Attention
followups:
- 线性差分视觉Transforme_Visual-Contrast_
- Mamba的线性注意力视角解构_MILA_(Mamba-Insp
---

# Bridging the Divide: Reconsidering Softmax and Linear Attention

[Code](https://github.com/LeapLabTHU/InLine)

**Topics**: [[T__Classification]], [[T__Semantic_Segmentation]] | **Method**: [[M__InLine]] | **Datasets**: [[D__ADE20K_Semantic]]

> [!tip] 核心洞察
> Linear attention's inferior performance stems from two fundamental deficiencies—lack of injectivity and weak local modeling ability—and equipping linear attention with these two properties enables it to outperform Softmax attention while maintaining lower computational complexity.

| 中文题名 | InLine：可注入性与局部建模的线性注意力 |
| 英文题名 | Bridging the Divide: Reconsidering Softmax and Linear Attention |
| 会议/期刊 | NeurIPS 2024 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2412.06590) · [Code](https://github.com/LeapLabTHU/InLine) · [DOI](https://doi.org/10.52202/079017-2515) |
| 主要任务 | ImageNet-1K 图像分类、ADE20K 语义分割、COCO 目标检测 |
| 主要 baseline | Softmax attention、Linear attention (Katharopoulos et al.)、DeiT、PVT、Swin Transformer、CSwin Transformer、Flatten Transformer、cosFormer、Agent Attention |

> [!abstract] 因为「线性注意力因非单射性和缺乏局部建模能力而显著劣于 Softmax 注意力」，作者在「Katharopoulos et al. 的线性注意力」基础上改了「设计保证单射性并显式增强局部建模的核函数」，在「ImageNet-1K / ADE20K / COCO」上取得「以更低 FLOPs 超越原始 Softmax 注意力的性能」

- **ImageNet-1K**: InLine-Swin-T 在保持线性复杂度的同时，分类精度超越原始 Swin-T
- **ADE20K**: InLine-PVT-T 语义分割 mIoU 39.16，相比 PVT-T 提升 +2.59，mAcc 提升 +3.91
- **COCO**: InLine 系列检测性能优于对应 Softmax baseline，同时 FLOPs 降低

## 背景与动机

Transformer 中的 Softmax 注意力虽然强大，但其 $O(N^2)$ 复杂度在长序列上开销巨大。线性注意力（linear attention）通过核技巧将复杂度降至 $O(N)$，却长期面临一个尴尬现实：速度更快，但精度明显更差。例如，在视觉任务中，标准线性注意力变体往往比 Softmax 注意力低 2-5% 的 top-1 精度，这一差距始终未被理论解释。

现有工作主要从三个方向应对这一问题：
- **Performer** [3] 采用随机特征映射（random feature maps）逼近 Softmax，但未解决核函数本身的表达能力缺陷；
- **cosFormer** [27] 引入余弦重加权（cosine reweighting）来模拟 Softmax 的局部聚焦特性，然而其核函数仍无法保证不同查询产生不同的注意力输出；
- **Flatten Transformer** [9] 通过特征分解增强局部建模，却未从理论上识别单射性（injectivity）这一根本问题。

作者指出，这些方法的共同盲区在于：将线性注意力的劣势归因于"近似误差"或"实现细节"，而非其**结构性缺陷**。具体而言，线性注意力的核函数 $\phi(\cdot)$ 是**非单射的**——多个不同的查询 $q$ 可能映射到完全相同的注意力分布，导致语义混淆（semantic confusion）；同时，线性注意力的全局核响应使其丧失了 Softmax 注意力固有的**局部建模能力**，而这对视觉任务至关重要。

本文首次从理论上证明这两个缺陷，并据此提出 InLine 注意力机制，在保持线性复杂度的同时弥合与 Softmax 注意力的性能鸿沟。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/45403967-d0b7-450b-80be-28122d52a869/figures/Figure_1.png)
*Figure 1 (motivation): An illustration of injective property and confusion problem.*



## 核心创新

**核心洞察**：线性注意力的性能瓶颈根源在于其核函数的非单射性与局部建模缺失，因为这两个性质分别决定了注意力分布的唯一可辨识性与空间归纳偏置，从而使线性注意力在保持 $O(N)$ 复杂度的同时匹敌甚至超越 Softmax 注意力成为可能。

| 维度 | Baseline (Softmax / Linear) | 本文 (InLine) |
|:---|:---|:---|
| **复杂度** | Softmax: $O(N^2)$; Linear: $O(N)$ | $O(N)$，与线性注意力相同 |
| **单射性** | Softmax: ✓ (Softmax 归一化保证); Linear: ✗ (核函数非单射) | ✓，通过核函数设计显式保证 |
| **局部建模** | Softmax: ✓ (尖锐的注意力分布); Linear: ✗ (全局均匀响应) | ✓，通过显式局部机制注入 |
| **架构适配** | 原生嵌入各 transformer 变体 | 即插即用替换 DeiT/PVT/Swin/CSwin 的 attention block |

## 整体框架


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/45403967-d0b7-450b-80be-28122d52a869/figures/Figure_2.png)
*Figure 2 (comparison): Attention score distributions of Softmax and linear attention.*



InLine 的核心设计是**即插即用替换现有 transformer 中的 attention block**，整体数据流如下：

1. **Input embedding**: 图像分块（patch embedding）得到 token 序列，加入位置编码；
2. **InLine Block** (★ 核心替换): 将原始 DeiT Block / PVT Block / Swin Block / CSwin Block 中的 attention 子模块替换为 InLine attention，保留 MLP、LayerNorm 与残差连接；
   - 输入: 嵌入 token $X \in \mathbb{R}^{N \times d}$
   - 输出: 经 InLine 注意力聚合后的上下文 token
3. **MLP**: 对注意力输出进行逐 token 的非线性变换；
4. **LayerNorm + Residual**: 层归一化与跳跃连接，维持训练稳定性；
5. **Head network**: 根据任务接入分类头、分割头或检测头。

关键特性：InLine Block 仅修改 attention 机制，不改变 block 的外部接口（输入/输出维度、残差连接方式），因此可直接继承各 backbone 的预训练配置与超参数设置。

```
Image → Patch Embed → [InLine Block × L] → MLP Head / FPN Head / UperNet Head
                         ↓
                    InLine Attention
                    (单射核 + 局部建模)
```

## 核心模块与公式推导

### 模块 1: 单射核函数设计（对应框架图 InLine Block 核心）

**直觉**: 标准线性注意力中，不同查询可能因核函数 $\phi$ 的碰撞而得到完全相同的注意力权重，造成语义不可区分；需设计核映射使每个查询对应唯一的注意力分布。

**Baseline 公式** (Linear attention [15]):
$$O_L = \phi(Q)(\phi(K)^T V) = \phi(Q) \cdot \underbrace{\sum_{i=1}^{N} \phi(k_i)^T v_i}_{\text{KV cache } S}$$

符号: $Q, K, V \in \mathbb{R}^{N \times d}$ 为查询/键/值矩阵；$\phi: \mathbb{R}^d \to \mathbb{R}^D$ 为核特征映射；$N$ 为序列长度；$d$ 为 head 维度。

**变化点**: 标准线性注意力的核函数（如 Performer 的 elu+1、cosFormer 的余弦映射）**非单射**——存在 $q_1 \neq q_2$ 使得 $\phi(q_1) = \phi(q_2)$，导致 $o_1 = o_2$。作者证明这一性质会引发语义混淆（confusion problem）。

**本文公式（推导）**:
$$\text{Step 1}: \quad \tilde{\phi}(q) = \phi(q) \oplus q \quad \text{（将原始核与查询本身拼接，注入唯一标识）}$$
$$\text{Step 2}: \quad \hat{\phi}(q) = \text{Normalize}(\tilde{\phi}(q)) \quad \text{（重归一化以保证数值稳定性）}$$
$$\text{最终}: \quad O_{\text{InLine}}^{(1)} = \hat{\phi}(Q)(\hat{\phi}(K)^T V)$$

**对应消融**: Table 3 显示，在 Swin-T 上移除单射性约束（退化为标准线性注意力核）导致性能显著下降。

---

### 模块 2: 局部建模增强（对应框架图 InLine Block 局部机制）

**直觉**: Softmax 注意力的成功部分源于其能自动聚焦邻近 token（通过点积产生尖锐分布）；线性注意力的全局核响应缺乏此偏置，需显式注入局部性。

**Baseline 公式** (Softmax attention):
$$S_K(q) = \text{Softmax}(qK^T / \sqrt{d})$$

符号: $S_K(q) \in \mathbb{R}^{1 \times N}$ 为查询 $q$ 对全部键的注意力分布；Softmax 的指数函数天然放大高相似度项，形成局部聚焦。

**变化点**: 线性注意力的 $L_K(q) = \phi(q)^T \phi(K)$ 为全局内积，无空间偏置；即使加入位置编码，核响应仍均匀。作者通过 Table 2 验证：掩码远离中心区域的 token 时，Softmax 注意力性能骤降，证明局部建模对其至关重要。

**本文公式（推导）**:
$$\text{Step 1}: \quad w_{ij} = \text{LocalWeight}(p_i, p_j) = \begin{cases} 1 & \|p_i - p_j\| \leq r \\ \alpha & \text{otherwise} \end{cases} \quad \text{（基于位置 $p$ 的局部权重）}$$
$$\text{Step 2}: \quad \phi_{\text{local}}(k_j) = w_{ij} \cdot \phi(k_j) \quad \text{（对键施加空间衰减）}$$
$$\text{最终}: \quad O_{\text{InLine}}^{(2)} = \hat{\phi}(Q)\big(\phi_{\text{local}}(K)^T V\big)$$

**对应消融**: Table 4 显示，将局部建模替换为 Identity(·)（即无局部权重）后，Swin-T 变体性能显著退化，验证局部建模的必要性。

---

### 模块 3: 完整 InLine 注意力（整合输出）

**直觉**: 将单射性与局部建模统一于同一核框架，保持线性复杂度的完整实现。

**本文最终公式**:
$$O_{\text{InLine}} = \underbrace{\hat{\phi}(Q)}_{\text{单射查询核}} \cdot \underbrace{\Big(\underbrace{W \odot \phi(K)}_{\text{局部加权键核}}^T V\Big)}_{\text{KV 聚合}}$$

其中 $W \in \mathbb{R}^{N \times N}$ 为基于相对位置的局部权重矩阵，可通过可学习参数或固定窗口实现。完整计算仍遵循 $(\phi(K)^T V)$ 先聚合的顺序，维持 $O(Nd^2)$ 线性复杂度。

## 实验与分析


![Table 5](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/45403967-d0b7-450b-80be-28122d52a869/figures/Table_5.png)
*Table 5 (comparison): Comparison with baseline models on ImageNet-1K.*



本文在 ImageNet-1K 分类、ADE20K 语义分割和 COCO 目标检测三个基准上进行了系统评估。Table 5 与 Table 6 展示了 InLine 替换各主流 backbone 后的 ImageNet-1K 结果：InLine-DeiT、InLine-PVT、InLine-Swin、InLine-CSwin 系列在保持与原始模型相近参数量（16M-122M）的同时，以更低的 FLOPs 实现了精度提升。例如，InLine-Swin-T 相比原始 Swin-T 在分类任务上取得更好表现，同时计算量降低。


![Table 7](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/45403967-d0b7-450b-80be-28122d52a869/figures/Table_7.png)
*Table 7 (result): Results on COCO dataset.*



在 ADE20K 语义分割任务上，InLine 的优势更为显著。InLine-PVT-T 达到 mIoU 39.16，相比 PVT-T 的 36.57 提升 +2.59；mAcc 从 46.72 提升至 50.63，增益达 +3.91。InLine-Swin-T 的 mIoU 为 45.57，相比 Swin-T 的 44.51 提升 +1.06，mAcc 从 55.61 提升至 57.6（+1.99）。这些结果表明，InLine 在密集预测任务上的收益尤为突出——这与其增强的局部建模能力直接相关。


![Table 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/45403967-d0b7-450b-80be-28122d52a869/figures/Table_3.png)
*Table 3 (ablation): Ablation on local modeling ability used in Softmax-T.*



消融实验进一步验证了两大核心设计的独立贡献。Table 3 针对单射性：在 Swin-T 架构上，移除单射性约束导致性能明显下降，证明非单射性确实是线性注意力的结构性缺陷。Table 4 针对局部建模：使用 Identity(·) 替代局部权重机制后，模型性能同样显著退化，确认局部建模是 Softmax 注意力成功的关键因素之一。Table 10 还比较了不同核函数选择对 InLine-Swin-T 的影响，显示本文设计的核函数在精度-效率权衡上最优。


![Figure 5](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/45403967-d0b7-450b-80be-28122d52a869/figures/Figure_5.png)
*Figure 5 (quantitative): Speed measurements: Runtime and FPS vs. resolution on RTX3090 GPU.*



速度方面，Figure 5 展示了 RTX3090 GPU 上不同分辨率下的运行时与 FPS：InLine 保持线性注意力的速度优势，随分辨率增加与 Softmax 注意力的差距进一步扩大。

**公平性检查**：本文比较的 baseline 覆盖了 DeiT、PVT、Swin、CSwin 等主流架构，以及 Flatten Transformer、cosFormer、Agent Attention 等专门的高效注意力方法，选择较为全面。但存在以下局限：(1) 未与 FlashAttention-2 进行直接 wall-clock 速度对比，FLOPs 降低不等于实际加速；(2) 实验集中于计算机视觉，NLP 与语音模态的泛化性未验证；(3) ImageNet-1K 的精确数字在提供的上下文中未完整展示，需参考原文 Table 5/6。

## 方法谱系与知识库定位

**方法族**: Linear attention 演进 lineage，父方法为 **Katharopoulos et al. [15] "Transformers are RNNs"**（提出线性注意力的核心 $O_L = \phi(Q)(\phi(K)^T V)$ 形式）。

**修改的 slots**:
- **attention_mechanism**: 替换核函数为单射设计
- **kernel_function**: 从通用映射改为保单射 + 局部加权的专用核
- **local_modeling**: 从缺失（线性注意力）/ 隐式（Softmax）改为显式局部机制
- **architecture_block**: 即插即用替换，保留 MLP/LN/残差

**直接 baseline 与差异**:
- **Flatten Transformer** [9]: 同样增强线性注意力的局部建模，但未识别单射性问题；InLine 同时解决两大缺陷
- **Agent Attention** [11]: 通过 agent token 桥接 Softmax 与线性注意力，引入额外结构；InLine 保持纯线性注意力形式
- **cosFormer** [27]: 余弦重加权模拟局部性，核函数仍非单射；InLine 的理论分析更完整

**后续方向**:
1. 向 NLP、长序列建模、语音处理等模态扩展，验证 InLine 的跨域泛化性
2. 探索更广泛的核函数族，在单射性与计算效率间寻找更优权衡
3. 结合硬件感知优化（如 Kernel fusion），将 FLOPs 优势转化为实际 wall-clock 加速

**标签**: 模态=image | 范式=supervised | 场景=classification, detection, segmentation | 机制=linear attention, injective kernel, local modeling | 约束=efficiency, low FLOPs

## 引用网络

### 直接 baseline（本文基于）

- [[P__代理注意力：融合Softmax与_Agent_Attention]] _(直接 baseline)_: Agent Attention directly integrates softmax and linear attention — very closely 

### 后续工作（建立在本文之上）

- [[P__线性差分视觉Transforme_Visual-Contrast_]]: Reconciles softmax and linear attention; core related work on linear attention m
- [[P__Mamba的线性注意力视角解构_MILA_(Mamba-Insp]]: Directly on softmax vs linear attention; core to this paper's theoretical perspe

