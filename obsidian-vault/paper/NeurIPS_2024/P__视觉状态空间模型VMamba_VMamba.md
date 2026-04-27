---
title: 'VMamba: Visual State Space Model'
type: paper
paper_level: C
venue: NeurIPS
year: 2024
paper_link: null
aliases:
- 视觉状态空间模型VMamba
- VMamba
acceptance: Spotlight
cited_by: 2100
code_url: https://github.com/MzeroMiko/VMamba
method: VMamba
---

# VMamba: Visual State Space Model

[Code](https://github.com/MzeroMiko/VMamba)

**Topics**: [[T__Classification]], [[T__Object_Detection]], [[T__Semantic_Segmentation]] | **Method**: [[M__VMamba]] | **Datasets**: [[D__ImageNet-1K]], [[D__MS-COCO]]

| 中文题名 | 视觉状态空间模型VMamba |
| 英文题名 | VMamba: Visual State Space Model |
| 会议/期刊 | NeurIPS 2024 (Spotlight) |
| 链接 | [arXiv](https://arxiv.org/abs/2401.10166) · [Code](https://github.com/MzeroMiko/VMamba) · [DOI](https://doi.org/10.52202/079017-3273) |
| 主要任务 | ImageNet-1K 图像分类、MS COCO 目标检测与实例分割、ADE20K 语义分割 |
| 主要 baseline | Swin Transformer、ConvNeXt、Mamba（组件来源） |

> [!abstract] 因为「视觉Transformer的自注意力具有O(N²)复杂度，且Swin等线性化方法牺牲了全局感受野」，作者在「Mamba选择性状态空间模型」基础上改了「通过Cross-Scan将1D SSM扩展为2D视觉数据的SS2D模块」，在「ImageNet-1K / MS COCO / ADE20K」上取得「VMamba-T: 82.2% top-1（+0.9 over Swin-T）、46.5 AP_box（+3.7 over Swin-T）、48.0 mIoU（+3.5 over Swin-T）」

- **ImageNet-1K**: VMamba-T 82.2% vs Swin-T 81.3% (+0.9), vs ConvNeXt-T 82.1% (+0.1)
- **MS COCO检测**: VMamba-T 46.5 AP_box vs Swin-T 42.8 (+3.7), vs ConvNeXt-T 44.2 (+2.3)
- **ADE20K分割**: VMamba-T 48.0 mIoU (SS) vs Swin-T 44.5 (+3.5), vs ConvNeXt-T 46.0 (+2.0)

## 背景与动机

视觉识别任务的核心挑战在于：如何高效地建立图像patch之间的全局关联。以一张高分辨率图像为例，若将图像切分为N个patch，标准ViT的自注意力需要计算N×N的相似度矩阵，复杂度为O(N²)，这在处理高分辨率输入时成为严重瓶颈。

现有方法主要从两个方向缓解这一问题。**Swin Transformer** 采用shifted window机制，将全局注意力限制在局部窗口内，再通过窗口移位实现跨窗口信息传递，复杂度降至O(N)，但感受野受窗口大小约束，全局建模能力受限。**ConvNeXt** 则回归纯卷积架构，使用大核深度可分离卷积（如7×7 DWConv）扩大感受野，但卷积的局部性本质使其难以直接捕获长距离依赖，且大核卷积的计算开销仍随分辨率增长。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/af7a0dc0-ff6c-4c9c-9df4-db92ab9d23ae/figures/Figure_1.png)
*Figure 1 (motivation): Comparison of the establishment of correlations between image patches through (a) self-attention and (b) the proposed 2D Selective Scan (SS2D).*



这两种方案的共同困境在于：**线性复杂度与全局感受野难以兼得**。Swin牺牲了全局性，ConvNeXt牺牲了长距离建模的直接性。与此同时，NLP领域出现的**Mamba**模型通过选择性状态空间模型（Selective State Space Model, SSM）实现了线性复杂度的序列建模，且能保持全局上下文——但其设计针对1D文本序列，无法直接处理2D图像的空间结构。

本文的核心动机正是：**将Mamba的线性复杂度全局建模能力迁移到视觉领域，同时解决2D空间信息的有效编码问题**。为此，作者提出VMamba，通过创新的2D选择性扫描机制，让状态空间模型真正适用于视觉任务。

## 核心创新

核心洞察：**2D图像的空间结构可以通过多方向1D扫描来保持**，因为Cross-Scan的四个方向扫描组合能够覆盖任意两点间的空间关系，从而使Mamba的1D选择性状态空间模型直接处理视觉数据成为可能。

| 维度 | Baseline (Swin / ConvNeXt / Mamba) | 本文 (VMamba) |
|:---|:---|:---|
| **token混合机制** | Swin: 局部窗口自注意力; ConvNeXt: 大核DWConv; Mamba: 1D SSM | **SS2D模块**：Cross-Scan → SSM → Cross-Merge，替换注意力/卷积 |
| **复杂度-感受野权衡** | Swin: O(N)但受限感受野; 标准注意力: O(N²)全局; Mamba: 1D全局 | **O(N)线性复杂度 + 全局感受野**，通过四方向扫描保持2D空间性 |
| **空间信息编码** | 2D直接操作（卷积/窗口）或1D序列（Mamba） | **Cross-Scan四方向扫描**：行优先、列优先及其转置变体，将2D转为1D序列组 |
| **局部特征补充** | Swin: 窗口内局部; ConvNeXt: DWConv固有局部性 | **SS2D块内嵌DWConv**，显式保留局部特征提取能力 |

与Mamba的差异在于：Mamba的扫描是单向1D序列，丢失空间邻域信息；VMamba通过Cross-Scan的四个正交方向，使每个patch能与上下左右所有方向的patch建立依赖。与Swin/ConvNeXt的差异在于：SS2D以状态演化替代注意力计算或卷积核滑动，理论复杂度线性且感受野全局。

## 整体框架


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/af7a0dc0-ff6c-4c9c-9df4-db92ab9d23ae/figures/Figure_3.png)
*Figure 3 (architecture): Illustration of (a) the overall architecture of VMamba and (b)-(d) the structure of its main components.*



VMamba采用与Swin/ConvNeXt一致的层次化架构，包含4个stage，逐步下采样提取多尺度特征。数据流如下：

**输入图像** → **Patch Embedding** → **Stage 1** (SS2D blocks × L₁ + Downsample) → **Stage 2** (SS2D blocks × L₂ + Downsample) → **Stage 3** (SS2D blocks × L₃ + Downsample) → **Stage 4** (SS2D blocks × L₄) → **任务头** (Classification / Detection / Segmentation Head)

各模块职责：
- **Patch Embedding**：将输入图像切分为不重叠的patch并投影为token嵌入，输出维度为H/4 × W/4 × C₁。
- **SS2D Block**（核心创新模块）：替代Swin的Shifted Window Attention或ConvNeXt的DWConv，负责token间的信息混合。每个block包含：LayerNorm → **SS2D** → 残差连接 → LayerNorm → MLP → 残差连接。
- **Cross-Scan**（SS2D内部）：将2D特征图沿四个方向（左上→右下、右下→左上、右上→左下、左下→右上）扫描为四条1D序列，使后续SSM能感知空间关系。
- **Selective State Space (SSM)**（SS2D内部）：对每条1D序列应用Mamba的选择性状态空间计算，输入自适应地决定信息传播或遗忘，复杂度线性。
- **Cross-Merge**（SS2D内部）：将四条处理后的1D序列按原扫描方向逆映射回2D空间，并通过相加/拼接融合为统一特征图。
- **DWConv**（SS2D内部）：并行于SSM分支的深度可分离卷积，显式提取局部特征，与全局SSM形成互补。
- **MLP/Feed-forward**：标准的前馈网络，对每个token独立进行通道维度变换。

```
输入特征图 X ∈ R^(H×W×C)
        │
        ├──→ [Cross-Scan] ──→ 4条1D序列 ──→ [SSM] ×4 ──→ [Cross-Merge] ──→ 全局特征
        │                                                              ╲
        └──→ [DWConv] ─────────────────────────────────────────────────→ [相加融合] ──→ Y
```

## 核心模块与公式推导

### 模块 1: 选择性状态空间模型（SSM）——Mamba基础

**直觉**：通过隐状态的线性演化实现序列建模，避免二次复杂度的注意力计算。

**Baseline 公式** (Mamba):
$$h'(t) = Ah(t) + Bx(t), \quad y(t) = Ch(t) + Dx(t)$$
符号: $h(t)$ = 隐状态, $x(t)$ = 输入, $A$ = 状态转移矩阵, $B$ = 输入投影, $C$ = 输出投影, $D$ = 跳跃连接。

离散化（零阶保持法）:
$$h_k = \bar{A}h_{k-1} + \bar{B}x_k, \quad y_k = \bar{C}h_k + Dx_k$$
其中 $\bar{A} = \exp(\Delta A)$, $\bar{B} = (\Delta A)^{-1}(\exp(\Delta A) - I) \cdot \Delta B$，步长$\Delta$由输入自适应生成（**选择性机制**：不同token有不同的$\Delta, B, C$）。

**变化点**: Mamba的SSM针对1D文本序列设计，直接应用于2D图像会**破坏空间邻域结构**——相邻行末与行首在1D扫描中被强制相邻，而真正的2D邻居可能被远离。

---

### 模块 2: Cross-Scan —— 2D空间编码（核心创新）

**直觉**：四个正交方向的扫描组合，使任意空间位置的patch能在至少一个扫描序列中保持正确的相对顺序。

**Baseline 公式** (Mamba 1D扫描):
$$\text{Scan}_{1D}(X) = [x_{1,1}, x_{1,2}, ..., x_{1,W}, x_{2,1}, ..., x_{H,W}]$$

**变化点**: 单一1D扫描导致空间失真；需要**多方向扫描**保持2D拓扑。

**本文公式**:
$$\text{Cross-Scan}(X) = [\underbrace{\text{RowScan}(X)}_{\text{左上→右下}}; \underbrace{\text{RowScan}(X^T)}_{\text{左下→右上}}; \underbrace{\text{ColScan}(X)}_{\text{右上→左下}}; \underbrace{\text{ColScan}(X^T)}_{\text{右下→左上}}]$$

生成四条独立1D序列，每条序列分别输入SSM：
$$\{y^{(i)}\}_{i=1}^4 = \{\text{SSM}(\text{Cross-Scan}^{(i)}(X))\}_{i=1}^4$$

**对应消融**: Figure 13（ERF可视化）显示Cross-Scan相比Unidi-Scan（单向）、Bidi-Scan（双向）、Cascade-Scan（级联）具有更均匀的全局有效感受野。

---

### 模块 3: SS2D Block —— 完整模块整合

**直觉**：将Cross-Scan、SSM、Cross-Merge与局部DWConv组合，形成可替换Transformer block的通用视觉模块。

**Baseline 公式** (Swin Attention):
$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V, \quad O(N^2) \text{ 复杂度}$$

**变化点**: 注意力需要成对token交互，复杂度二次；SS2D通过状态演化实现线性复杂度的全局交互，但需解决2D适配和局部特征缺失。

**本文公式（推导）**:
$$\text{Step 1: } \{S_i\}_{i=1}^4 = \text{Cross-Scan}(X) \quad \text{(将2D转为4条1D序列)}$$
$$\text{Step 2: } \{H_i\}_{i=1}^4 = \{\text{SSM}(S_i)\}_{i=1}^4 \quad \text{(各序列独立经选择性状态空间处理)}$$
$$\text{Step 3: } Y_{\text{global}} = \text{Cross-Merge}(\{H_i\}_{i=1}^4) \quad \text{(融合还原为2D特征图)}$$
$$\text{Step 4: } Y_{\text{local}} = \text{DWConv}(X) \quad \text{(并行局部特征分支)}$$
$$\text{最终: } Y = Y_{\text{global}} + Y_{\text{local}} + X \quad \text{(残差连接，保证训练稳定性)}$$

完整SS2D block（含LayerNorm和MLP）:
$$\hat{X} = \text{SS2D}(\text{LN}(X)) + X$$
$$Y = \text{MLP}(\text{LN}(\hat{X})) + \hat{X}$$

**对应消融**: Table 4显示Vanilla-VMamba（无DWConv等优化）性能下降，验证DWConv和完整设计的必要性；d_state从1增至4/8/16带来轻微提升但吞吐量显著下降，故默认d_state=1为最优效率-精度权衡。

## 实验与分析


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/af7a0dc0-ff6c-4c9c-9df4-db92ab9d23ae/figures/Table_1.png)
*Table 1 (quantitative): Performance comparison on ImageNet-1K.*




![Table 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/af7a0dc0-ff6c-4c9c-9df4-db92ab9d23ae/figures/Table_2.png)
*Table 2 (quantitative): Left: Results for object detection and instance segmentation on MS COCO.*



本文在三大视觉基准上评估VMamba-T/S/B三个尺度。ImageNet-1K分类结果（Table 1）显示，VMamba-T达到82.2% top-1准确率，相比Swin-T的81.3%提升+0.9，相比ConvNeXt-T的82.1%提升+0.1；VMamba-S达到83.5%，相比Swin-S的83.0%提升+0.5，相比ConvNeXt-S的83.1%提升+0.4。这一差距在下游密集预测任务中显著放大：MS COCO目标检测（Table 2左）中，VMamba-T以Mask R-CNN 1× schedule取得46.5 AP_box，大幅领先Swin-T的42.8（+3.7）和ConvNeXt-T的44.2（+2.3）；实例分割AP_mask达到41.7，领先Swin-T的39.2（+2.5）。ADE20K语义分割（Table 8）中VMamba-T取得48.0 mIoU（单尺度），领先Swin-T 44.5（+3.5）和ConvNeXt-T 46.0（+2.0）。这些结果表明SS2D的全局感受野在需要精细空间定位的任务上优势更为明显。


![Figure 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/af7a0dc0-ff6c-4c9c-9df4-db92ab9d23ae/figures/Figure_4.png)
*Figure 4 (result): Illustration of VMamba's resource consumption with progressively increasing resolutions.*



资源效率方面，Figure 4展示了VMamba随分辨率增加的显存和计算开销增长趋势，Figure 8对比了不同扫描模式的资源消耗，验证Cross-Scan在保持性能的同时具有线性复杂度特征。Table 6的吞吐量对比显示VMamba与Swin/ConvNeXt相当或更优，同时享有全局感受野优势。



消融实验（Table 4及相关附录表格）揭示了关键设计选择：去掉SS2D块内的DWConv导致性能下降，验证局部特征补充的必要性；将Cross-Scan替换为Unidi-Scan（单向）、Bidi-Scan（双向）或Cascade-Scan（级联）均降低有效感受野质量（Figure 13 ERF可视化）；HiPPO初始化替换为零初始化在d_state=1时无显著差异，简化了训练流程；增大d_state至4/8/16带来轻微精度提升但吞吐量大幅下降，故采用d_state=1配合ssm-ratio调节作为效率-精度权衡。

公平性检查：本文主要对比Swin Transformer和ConvNeXt，两者均为该时期最强基线之一。未直接对比的基线包括DeiT、PVT、HiViT以及同期视觉Mamba变体如Vision Mamba (Vim)。训练使用8×A100 GPU，与Swin/ConvNeXt可比。潜在局限：部分消融仅在VMamba-T上进行，未扩展至S/B尺度；吞吐量数据可能受具体实现优化影响；Vanilla-VMamba与VMamba的对比未完全隔离单一变量。

## 方法谱系与知识库定位

**方法家族**：状态空间模型视觉化（Visual State Space Models）

**父方法**：Mamba（Gu & Dao, 2023）— 线性时间序列建模的选择性状态空间架构。VMamba直接继承其SSM核心计算，但将1D扫描扩展为2D Cross-Scan机制，并将架构从纯序列模型改造为层次化视觉backbone。

**改动插槽**：
- **架构**：Swin的shifted window attention / ConvNeXt的DWConv → **SS2D block**（Cross-Scan + SSM + Cross-Merge + DWConv）
- **数据流**：标准2D特征图操作 → **四方向1D扫描-处理-合并**的2D-1D-2D转换流程
- **训练配方**：Mamba的HiPPO初始化 → **零初始化**（d_state=1时等价），简化且稳定
- **推断策略**：保持O(N)线性复杂度，但实现**全局感受野**（vs Swin的局部窗口）

**直接基线对比**：
- **Swin Transformer**：同为层次化backbone，但Swin用窗口注意力牺牲全局性；VMamba以SS2D实现线性复杂度的全局建模
- **ConvNeXt**：同为现代CNN设计，但ConvNeXt依赖大核卷积的局部堆叠；VMamba以状态演化替代卷积滑动，长距离依赖更直接
- **Vision Mamba (Vim)**：同期工作，Vim采用双向序列扫描；VMamba提出四方向Cross-Scan并配套Cross-Merge，空间覆盖更完整

**后续方向**：(1) 更高效的多方向扫描策略（如自适应方向选择）；(2) SS2D与自注意力的混合架构，在关键层保留注意力精度；(3) 视频/3D点云等更高维数据的扫描扩展。

**标签**：modality=图像 / paradigm=层次化backbone / scenario=通用视觉表示 / mechanism=选择性状态空间+多方向扫描 / constraint=线性复杂度+全局感受野

