---
title: On the VC dimension of deep group convolutional neural networks
type: paper
paper_level: C
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 深度群卷积网络的VC维理论分析
- VC Dimension Ana
- VC Dimension Analysis for Deep GCNNs
- We derive tight upper and lower bou
acceptance: Poster
method: VC Dimension Analysis for Deep GCNNs
modalities:
- Image
paradigm: supervised
---

# On the VC dimension of deep group convolutional neural networks

**Topics**: [[T__Classification]] | **Method**: [[M__VC_Dimension_Analysis_for_Deep_GCNNs]] | **Datasets**: Theoretical VC dimension bounds for deep GCNNs

> [!tip] 核心洞察
> We derive tight upper and lower bounds on the VC dimension of deep ReLU GCNNs that scale with the number of layers, weights, and input dimension, and show these bounds are comparable to or improve upon those for standard CNNs and fully-connected networks.

| 中文题名 | 深度群卷积网络的VC维理论分析 |
| 英文题名 | On the VC dimension of deep group convolutional neural networks |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2410.15800) · [DOI](https://doi.org/10.48550/arxiv.2410.15800) |
| 主要任务 | Image Classification（图像分类）的VC维理论刻画 |
| 主要 baseline | Standard CNN VC bounds (Bartlett et al.; Harvey et al.); Fully-connected ReLU DNN VC bounds; Two-layer continuous GCNN (Cohen & Welling 2016); VC dimensions of group convolutional neural networks (2024) |

> [!abstract] 因为「深度群卷积神经网络（GCNN）的VC维在理论上几乎空白，仅有连续两层GCNN的结果」，作者在「标准CNN与全连接DNN的VC界」基础上改了「分析目标至离散深层ReLU GCNN，引入提升嵌入构造与群轨道组合分析」，在「理论比较」上取得「深层GCNN的VC上下界与标准CNN/DNN同阶，且显式揭示输入分辨率r的依赖性」。

- **核心下界**：VC(H_{W,L,r}) ≥ VC(F_{⌊W/6⌋, L−1})，即W权重L层GCNN至少能表达⌊W/6⌋权重L−1层DNN
- **核心上界**：VC(H_{W,L,r}) = O(L · W · log W)，与标准CNN/DNN同阶增长
- **深度突破**：从先前连续两层GCNN扩展至任意深度离散ReLU架构

## 背景与动机

群卷积神经网络（GCNN）通过引入群等变卷积，将标准CNN的平移等变性扩展至旋转、反射等更一般的对称变换，在医学图像分析、分子建模等任务中展现出结构先验优势。然而，一个基础理论问题长期悬而未决：这些网络的表达能力究竟有多强？其泛化边界由什么决定？VC维作为衡量假设类复杂度的经典指标，为回答这一问题提供了严格框架。

现有工作可分为三条线索。**其一**，Bartlett et al. 与 Harvey et al. 对标准CNN和全连接ReLU DNN建立了近乎紧致的VC界，证明其随权重数W和层数L以O(L·W·log W)增长，但这些分析局限于平移等变的标准卷积结构。**其二**，Cohen & Welling (2016) 开创性地提出群等变卷积框架，但后续关于GCNN的理论分析多聚焦于表示能力或样本复杂度，而非VC维。**其三**，2024年一篇同名工作首次触及GCNN的VC维，但仅处理了连续两层GCNN的特例，且未考虑ReLU激活与任意深度。

这些工作的空白构成了明确瓶颈：当网络深度超过两层、群作用离散化、激活函数取ReLU时，GCNN的VC维如何随架构参数 scaling？输入图像分辨率r是否会通过群轨道结构影响复杂度？更关键的是，群等变性约束——这一看似限制网络自由度的结构——究竟会压缩还是保持模型的表达能力？本文首次为深层离散ReLU GCNN建立系统的VC维上下界，并证明其与标准网络同阶，从而将群等变架构纳入经典学习理论的统一图景。

## 核心创新

核心洞察：群等变卷积的"提升-投影"结构本身就蕴含了一个嵌入标准DNN的通道，因为lifting操作将输入空间扩张到群空间时保留了足够的线性可分性，从而使"GCNN的VC维至少不低于同规模DNN"这一反直觉结论成为可能——群对称性并未削弱深层网络的表达能力。

| 维度 | Baseline | 本文 |
|:---|:---|:---|
| 分析对象 | 标准CNN / 全连接DNN | 一般群等变CNN（GCNN） |
| 深度范围 | 任意深度（CNN/DNN）；仅两层（先前GCNN） | 任意深度的离散ReLU GCNN |
| 下界构造 | 直接计数激活模式（DNN）；无（先前GCNN） | 提升嵌入：将DNN显式嵌入GCNN |
| 输入分辨率 | 无显式依赖 | 通过群轨道结构显式依赖r |
| 紧致性 | O(L·W·log W)上下界匹配 | 同阶匹配，常数因子含群结构 |

## 整体框架

本文理论框架由三大模块构成，形成"下界嵌入—上界计数—跨类比较"的完整分析链。

**模块一：下界构造——DNN嵌入GCNN**。输入为具有W'权重、L'层的标准全连接ReLU DNN（函数类记为F_{W',L'}）。通过群卷积的lifting操作将输入从基空间提升到群空间，再辅以特定的核约束与通道分配，将该DNN完整嵌入到一个具有约6W'权重、L'+1层的GCNN中。输出结论：VC(H_{6W',L'+1,r}) ≥ VC(F_{W',L'})，即GCNN至少保有同规模DNN的表达能力。

**模块二：上界推导——组合分段线性分析**。输入为具有W权重、L层、群结构G、分辨率r的GCNN。利用ReLU的分段线性特性，通过计算网络输出在输入空间诱导的线性区域数目来控制增长函数。群结构通过轨道分解（orbit decomposition）影响区域计数，输入分辨率r显式出现在轨道基数项中。输出结论：VC(H_{W,L,r}) = O(L · W · log W)。

**模块三：跨类比较**。将GCNN的上下界与标准CNN、全连接DNN的已知界并置，分析群等变性对复杂度 scaling 的净效应。

整体数据流可概括为：
```
DNN F_{W',L'}  --[lifting嵌入]-->  GCNN H_{6W',L'+1,r}  --[VC下界]-->  VC ≥ VC(F)
                                         ↑
GCNN H_{W,L,r}  --[分段线性计数]-->  增长函数界  --[VC上界]-->  VC = O(L·W·log W)
                                         ↓
                              输入分辨率 r → 群轨道结构 → 轨道基数影响区域数
```

## 核心模块与公式推导

### 模块一：DNN嵌入构造（对应框架图下界部分）

**直觉**：群卷积的lifting层天然提供了一个将标量输入"广播"到群各元素上的机制，若精心设计核使得仅在单位元处保留有效连接，即可在群空间模拟全连接层的行为。

**Baseline 公式**（标准DNN VC界，Harvey et al.）：
$$\text{VC}(\mathcal{F}_{W,L}) = O(L \cdot W \cdot \log W)$$
符号：$\mathcal{F}_{W,L}$ 为具有W个权重、L层的全连接ReLU DNN函数类；VC(·) 表示Vapnik-Chervonenkis维度。

**变化点**：先前GCNN VC分析[22]缺乏下界构造，无法与DNN界对话；本文需要证明GCNN"至少不弱于"DNN。

**本文公式（推导）**：
$$\text{Step 1:} \quad \mathcal{F}_{W',L'} \text{hookrightarrow} \mathcal{H}_{6W',L'+1,r} \quad \text{（通过lifting将DNN嵌入GCNN，权重膨胀约6倍源于群轨道基数）}$$
$$\text{Step 2:} \quad \text{VC}(\mathcal{H}_{6W',L'+1,r}) \geq \text{VC}(\mathcal{F}_{W',L'}) \quad \text{（嵌入保持表达能力，故VC维不降）}$$
$$\text{最终:} \quad \text{VC}(\mathcal{H}_{W,L,r}) \geq \text{VC}(\mathcal{F}_{\lfloor W/6 \rfloor, L-1}) \quad \text{（重参数化：令 } W=6W', L=L'+1\text{）}$$

**对应消融**：本文未提供数值消融表（纯理论工作），但下界紧致性依赖于嵌入构造中"6倍权重膨胀"这一常数是否为最优——作者指出这是群结构特定的，对于具体群（如循环群C_n）可能改进。

### 模块二：上界组合分析（对应框架图上界部分）

**直觉**：ReLU网络将输入空间划分为若干线性区域，输出在每个区域上为线性函数；VC维由这些区域的组合增长所控制。群等变性约束减少了有效参数配置，但群轨道也引入了额外的对称相关区域。

**Baseline 公式**（标准CNN/DNN增长函数界）：
$$\Pi_{\mathcal{F}}(N) \leq \left(\frac{2eN}{W}\right)^{WL}$$
其中$\Pi_{\mathcal{F}}(N)$为增长函数，N为样本数。

**变化点**：标准分析将卷积核视为自由参数；GCNN中核受群等变约束（$\kappa(g \cdot x) = g \cdot \kappa(x)$），且输入分辨率r通过群轨道基数$|G \cdot x|$影响有效输入维度。

**本文公式（推导）**：
$$\text{Step 1:} \quad \text{piecewise-linear regions for GCNN} \leq \prod_{\text{ell}=1}^{L} O\left(\frac{W_\text{ell} \cdot r \cdot |G|}{m_\text{ell}}\right)^{m_\text{ell}} \quad \text{（各层区域数受群轨道结构调制，} r \cdot |G| \text{ 关联分辨率与群阶）}$$
$$\text{Step 2:} \quad \Pi_{\mathcal{H}}(N) \leq \exp\left(O(L \cdot W \cdot \log(W \cdot r \cdot N))\right) \quad \text{（增长函数取对数，利用} \log \Pi = O(\text{VC})\text{）}$$
$$\text{Step 3:} \quad \text{令 } \Pi_{\mathcal{H}}(N) < 2^N \text{ 解得 } N = O(L \cdot W \cdot \log W) \quad \text{（Sauer-Shelah引理，r依赖被吸收进对数项）}$$
$$\text{最终:} \quad \text{VC}(\mathcal{H}_{W,L,r}) = O(L \cdot W \cdot \log W)$$

**对应消融**：上界中分辨率r的显式依赖在最终简化中被对数项吸收，作者指出这是紧致性分析中的关键取舍——对于固定群结构，r仅影响常数因子；但当r随群轨道指数增长时（如高维旋转群），r的 scaling 可能成为主导项。

## 实验与分析

本文为一项纯理论工作，未包含传统意义上的实验验证或数值结果表。其核心"实验"体现为理论结果的系统性比较与边界条件分析。

**主要理论结果**可概括为三重比较：第一，GCNN下界 VC(H_{W,L,r}) ≥ VC(F_{⌊W/6⌋, L−1}) 表明，具有W权重L层的GCNN至少能表达约W/6权重、L−1层的标准DNN，这意味着群等变性约束并未造成表达能力阶的退化；第二，GCNN上界 O(L·W·log W) 与Bartlett et al.、Harvey et al. 对标准CNN和DNN建立的界同阶，说明引入一般群对称性不增加VC维的渐近增长；第三，与2024年同名工作[22]（仅两层连续GCNN）相比，本文将深度扩展至任意L、激活扩展至ReLU、群作用离散化，覆盖了实际部署中的全部主流架构设定。

**分辨率依赖性分析**是本文区别于所有baseline的独特贡献。先前CNN/DNN的VC界不显含输入分辨率，而GCNN中群轨道结构使得分辨率r通过$|G \cdot x|$（群作用下x的轨道基数）进入界中。对于紧群（如SO(2)旋转群），连续化后轨道维度与r耦合；对于离散群（如C_4四旋转），轨道基数为常数，r的效应退化为空间采样密度。这一分析为理解"为何高分辨率图像配合大群结构时GCNN泛化可能变差"提供了理论线索。

**公平性审视**：本文的比较基线确实代表了该领域最强理论结果——Harvey et al. [3]的近乎紧致DNN界、2024年GCNN工作[22]的直接前身。但存在三方面局限：其一，所有界均为最坏情形分析，未考虑数据分布或优化算法的隐式正则化；其二，下界构造中"6倍权重膨胀"的常数因子可能非最优，作者未给出紧性证明；其三，完全缺乏实证验证——未在MNIST、CIFAR等标准数据集上测量实际GCNN的泛化间隙以检验理论预测的r依赖性。此外，分析局限于单输出分类设定，多输出/回归情形的VC维（伪维，pseudodimension）推广留待未来工作。

## 方法谱系与知识库定位

**方法族**：VC维分析 → 神经网络表达能力理论 → 等变网络学习理论

**直接父方法**：VC dimensions of group convolutional neural networks (2024) [22]。本文在其基础上将深度从两层扩展至任意层、连续性扩展至离散性、激活从一般分段线性特化至ReLU，并新增输入分辨率依赖分析。

**核心改动槽位**：
- **analysis_target**：标准CNN/DNN → 一般群等变CNN
- **depth_scope**：两层连续 → 任意深度离散ReLU
- **input_resolution_dependence**：无 → 通过群轨道显式依赖r
- **proof_technique**：直接计数 → lifting嵌入 + 群轨道组合分析

**直接基线差异**：
- Harvey et al. [3]（Nearly-tight VC-dimension...）：本文将其DNN界通过嵌入转化为GCNN下界，并证明GCNN上界与之同阶
- Cohen & Welling [8,9]（Group equivariant convolutional networks）：借用其群卷积框架，但将其从架构设计转化为分析对象
- Improved generalization bounds via quotient feature spaces [26]：替代理论路径（商特征空间 vs VC维），本文结果与之互补

**后续方向**：(1) 多输出/回归设定的伪维（pseudodimension）分析；(2) 常数因子紧致化——优化嵌入构造中的权重膨胀比；(3) 实证桥接：在旋转MNIST、医学图像等真实GCNN应用上验证r依赖性的预测。

**标签**：modality: image | paradigm: supervised learning theory | scenario: equivariant representation learning | mechanism: group convolution / lifting / orbit decomposition | constraint: ReLU activation, single-output classification, discrete group actions
