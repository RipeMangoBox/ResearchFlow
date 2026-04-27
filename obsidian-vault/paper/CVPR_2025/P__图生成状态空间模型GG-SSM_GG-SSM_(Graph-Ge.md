---
title: 'GG-SSMs: Graph-Generating State Space Models'
type: paper
paper_level: C
venue: CVPR
year: 2025
paper_link: null
aliases:
- 图生成状态空间模型GG-SSM
- GG-SSM (Graph-Ge
- GG-SSM (Graph-Generating State Space Model)
acceptance: poster
cited_by: 5
code_url: https://github.com/uzh-rpg/gg_ssms
method: GG-SSM (Graph-Generating State Space Model)
---

# GG-SSMs: Graph-Generating State Space Models

[Code](https://github.com/uzh-rpg/gg_ssms)

**Topics**: [[T__Classification]], [[T__Time_Series_Forecasting]], [[T__Object_Tracking]] | **Method**: [[M__GG-SSM]] | **Datasets**: [[D__ImageNet-1K]] (其他: Time Series Forecasting - Exchange, Time Series Forecasting - Weather)

| 中文题名 | 图生成状态空间模型GG-SSM |
| 英文题名 | GG-SSMs: Graph-Generating State Space Models |
| 会议/期刊 | CVPR 2025 (poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2412.12423) · [Code](https://github.com/uzh-rpg/gg_ssms) · [Project](待补充) |
| 主要任务 | ImageNet图像分类、时间序列预测、基于事件的眼动追踪 |
| 主要 baseline | Mamba, VMamba, S-Mamba, Swin, DeiT, iTransformer, DLinear |

> [!abstract] 因为「标准SSM（如Mamba）的线性序列扫描无法捕捉token间的非局部依赖关系」，作者在「Mamba」基础上改了「将一维序列扫描替换为基于MST的动态图结构状态传播，通过路径矩阵连乘实现图上全局信息聚合」，在「ImageNet-1K」上取得「Top-1 Accuracy 83.6%（Tiny），相比VMamba-T提升+1.0%」

- **ImageNet-1K Tiny**: Top-1 Accuracy 83.6%，超越VMamba-T（82.6%）+1.0%，超越Swin-T（81.3%）+2.3%
- **Weather预测**: MSE 0.225，超越S-Mamba（0.251）-0.026，超越iTransformer（0.258）-0.033
- **Solar-Energy预测**: MSE 0.1832，大幅领先S-Mamba（0.24）-0.0568
- **事件眼动追踪**: 仅62k参数、3.01M MACs，相比3ET（107M MACs）压缩35.5×

## 背景与动机

状态空间模型（State Space Models, SSMs）如Mamba通过线性时间复杂度的序列扫描，在语言和视觉任务中展现出替代Transformer的潜力。然而，其核心假设——状态沿一维序列顺序传播——存在根本性局限：当处理图像块或时间序列token时，空间上相近或语义相关的token可能在线性序列中相距甚远，导致信息交互效率低下。

具体而言，现有方法面临三重困境：**Mamba** 采用选择性状态空间机制，通过输入依赖的B、C矩阵调节状态转移，但A矩阵的幂次仅沿单一维度累积，本质仍是因果序列模型；**VMamba** 将二维图像展平为四向扫描序列，通过交叉扫描弥补空间性，但展平操作破坏了原始拓扑结构，且四向扫描的计算开销显著增加；**S4ND** 虽尝试多维SSM扩展，却依赖固定的多维网格结构，无法自适应输入内容动态调整连接模式。

这些方法的共同瓶颈在于：**状态传播拓扑是静态且一维的**。无论是简单的空间展平还是规则的多维网格，都未能让模型根据输入特征语义动态构建最优的信息流通路径。例如，在一张包含分散物体的图像中，属于同一物体的token应当直接通信，而非绕道遍历整个序列；在时间序列中，具有相似模式的远距时间点也应建立捷径连接。

本文的核心动机正是打破这一限制：将SSM的状态传播从预定义的线性序列解放，赋予其**根据输入内容动态生成图结构**的能力，同时保持近线性的计算效率。

## 核心创新

核心洞察：**状态空间模型的状态转移可以重新诠释为图上的路径积分**，因为一维序列的A矩阵幂次累积等价于链式图的路径矩阵连乘，从而使任意结构化数据（图像、时间序列、事件流）都能通过自适应稀疏图实现高效全局交互成为可能。

| 维度 | Baseline (Mamba/VMamba) | 本文 (GG-SSM) |
|:---|:---|:---|
| 状态传播拓扑 | 固定的一维序列或规则多维扫描 | **输入自适应的MST图结构**，边权重由token特征余弦相似度决定 |
| 状态转移算子 | 标量A的n次幂 Āⁿ（沿序列位置） | **路径矩阵连乘** S_ji = ∏ₘ Ā_{k_m}（沿图路径） |
| 信息聚合方式 | 递推式 h[n] = Ā·h[n-1] + B̄·x[n] | **全局求和** h_i = Σ_{v_j∈V} S_ji·B̄_j·x_j（所有节点通过图路径贡献） |
| 图构建效率 | 无图构建（拓扑固定） | **Chazelle近线性MST算法**，O(m α(m,n))复杂度，epoch时间仅为Kruskal的0.90× |
| 输出稳定性 | 直接投影 y = C·h + D·x | **归一化后投影** y_i = C_i·Norm(h_i) + D·x_i，抑制图传播数值发散 |

## 整体框架


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/68fb2801-642a-41ce-8992-38468a92567a/figures/fig_001.png)
*Figure: Illustration of the Graph-Generating State Space*



GG-SSM的完整数据流包含五个核心阶段，形成"特征驱动图构建→图状态传播→归一化输出"的端到端可微流程：

**阶段1：输入特征提取（Input Feature Extraction）**
接收原始输入（图像块、时间序列token或事件数据），通过标准嵌入层生成d维token表征 x_i ∈ ℝᵈ。此模块与VMamba等基线一致，无特殊修改。

**阶段2：图边权重计算（Graph Edge Weight Computation）**
对每对token (i,j)，计算余弦相似度并映射为非负权重：w_ij = exp(-x_i^⊤x_j / (||x_i||·||x_j||))。该操作将语义相似性转化为图连接强度，替代了固定位置编码或简单空间邻接矩阵。

**阶段3：MST图构建（MST Graph Construction）**
基于边权重集合 {w_ij}，调用最小生成树算法（Chazelle's / Kruskal / Prim）构建稀疏树结构。Chazelle算法以近线性复杂度确定保留哪些边，形成动态图拓扑 G = (V, E_MST)。

**阶段4：图状态传播（Graph State Propagation）**
核心创新模块。在构建的MST上，状态通过路径矩阵连乘和邻居聚合完成更新：首先计算路径转移矩阵 S_ji（沿j到i唯一路径的Ā矩阵连乘），然后聚合全局状态 h_i = Σ_j S_ji·B̄_j·x_j。此步骤替代标准SSM的线性扫描。

**阶段5：下游预测头（Downstream Prediction Head）**
对输出token y_i 应用任务特定的投影层，生成分类logits、预测序列或回归值。

```
输入数据 → [Token Embedding] → {x_i}
                              ↓
                    {x_i} ↔ {x_j} 余弦相似度
                              ↓
                    [w_ij计算] → {边权重}
                              ↓
                    [MST算法: Chazelle's] → 稀疏图G=(V,E)
                              ↓
                    [图状态传播: S_ji连乘 + h_i聚合] → {h_i}
                              ↓
                    [归一化: Norm(h_i)] → [输出: C_i·Norm(h_i)+D·x_i]
                              ↓
                    [预测头] → 任务输出
```

## 核心模块与公式推导

### 模块1: 动态图边权重计算（对应框架图阶段2）

**直觉**: 让图连接强度反映token间的语义相关性，而非预设的空间或顺序邻近性。

**Baseline公式** (Mamba/VMamba — 无动态图构建，固定邻接):
无显式图构建，依赖位置编码或扫描顺序隐含拓扑。

**本文公式**:
$$w_{ij} = \exp\left( -\frac{\mathbf{x}_i^\text{top} \mathbf{x}_j}{\|\mathbf{x}_i\| \|\mathbf{x}_j\|} \right)$$

符号: $\mathbf{x}_i, \mathbf{x}_j$ = 第i,j个token的特征向量；$w_{ij}$ = 非负边权重，值越大表示特征越相似（负指数将余弦相似度[-1,1]映射到正权重域）。

**变化点**: Baseline无此步骤；本文新增输入自适应的边权重计算，使图拓扑随数据动态变化。

---

### 模块2: 图路径状态转移与聚合（对应框架图阶段4）

**直觉**: 将一维序列的标量A幂次推广到图结构的路径矩阵连乘，使状态可以沿任意图拓扑传播。

**Baseline公式** (Mamba 离散化SSM):
$$\mathbf{h}[n] = \bar{\mathbf{A}} \mathbf{h}[n-1] + \bar{\mathbf{B}} \mathbf{x}[n], \quad \mathbf{y}[n] = \bar{\mathbf{C}} \mathbf{h}[n] + \bar{\mathbf{D}} \mathbf{x}[n]$$
符号: $\bar{\mathbf{A}}, \bar{\mathbf{B}}, \bar{\mathbf{C}}, \bar{\mathbf{D}}$ = 离散化系统矩阵；$\mathbf{h}[n]$ = 时刻n的隐藏状态；递推形式等价于卷积核 $\bar{\mathbf{A}}^n \bar{\mathbf{B}}$ 与输入的卷积。

**变化点**: 标准SSM的 $\bar{\mathbf{A}}^n$ 仅在单一维度上累积，无法处理分支、合并或跳跃连接。本文将其重新诠释为**图上的路径积分**。

**本文公式（推导）**:
$$\text{Step 1 (路径转移矩阵)}: \quad S_{ji} = \prod_{m=1}^{n} \bar{\mathbf{A}}_{k_m} \quad \text{沿路径 } v_j \to \cdots \to v_i \text{ 上所有}\bar{\mathbf{A}}\text{矩阵连乘}$$
$$\text{Step 2 (全局状态聚合)}: \quad \mathbf{h}_i = \sum_{v_j \in V} S_{ji} \bar{\mathbf{B}}_j \mathbf{x}_j \quad \text{所有节点通过各自到i的路径贡献状态}$$
$$\text{Step 3 (归一化输出)}: \quad y_i = C_i \, \text{Norm}(\mathbf{h}_i) + D \, \mathbf{x}_i$$

**推导逻辑**: Step 1将一维的标量幂 $\bar{\mathbf{A}}^n$ 替换为图上路径的矩阵连乘，处理非序列结构；Step 2将递推的局部更新展开为全局显式求和，实现非局部信息交互（类似图神经网络的message passing）；Step 3引入归一化抑制图传播导致的数值膨胀，同时C_i变为节点相关参数增强表达能力。

**对应消融**: Table 8显示MST算法选择对准确率影响极小（±0.1%），但Chazelle算法将epoch时间降至Kruskal的0.90×。

---

### 模块3: 归一化图输出（对应框架图阶段4末端）

**直觉**: 图上的全局聚合可能产生数值范围不稳定，需归一化保证训练稳定性。

**Baseline公式** (Mamba):
$$\mathbf{y}[n] = \bar{\mathbf{C}} \mathbf{h}[n] + \bar{\mathbf{D}} \mathbf{x}[n]$$

**变化点**: 标准输出直接投影，无归一化；图聚合后的h_i可能因多路径累积而方差增大。

**本文公式**:
$$y_i = C_i \, \text{Norm}(\mathbf{h}_i) + D \, \mathbf{x}_i$$

符号: Norm(·) = 层归一化或类似归一化操作；$C_i$ = 节点特定的输出矩阵（注意下标i，区别于baseline的全局C）。

**对应消融**: 

## 实验与分析



本文在三大任务域验证GG-SSM：图像分类（ImageNet-1K）、时间序列预测（Weather/Solar-Energy/Exchange）、事件眼动追踪。核心结果集中于Table 4（时间序列）与Table 5（ImageNet分类）。

**ImageNet-1K分类**（Table 5）是论文的主打结果。GG-SSM-Tiny取得**Top-1 Accuracy 83.6%**，在相同计算预算下超越直接基线VMamba-T（82.6%）**+1.0个百分点**，同时超越Swin-T（81.3%）+2.3%、DeiT-S（79.8%）+3.8%。这一差距在Small尺度保持（84.4% vs VMamba-S 83.4%，+1.0%），Base尺度为84.9% vs VMamba-B 83.9%（+1.0%）vs Swin-B 83.5%（+1.4%）。值得注意的是，论文标注Tiny/Small为"best"，但Base尺度标注为"second best"，暗示更大模型可能存在优化瓶颈。

**时间序列预测**呈现数据集依赖性（Table 4）。**Weather数据集**上GG-SSM取得MSE **0.225**，显著优于S-Mamba（0.251，-0.026）和iTransformer（0.258，-0.033），为最佳方法。**Solar-Energy**上优势更大：MSE **0.1832** vs S-Mamba（0.24，-0.0568）vs iTransformer（0.233，-0.0498）。然而**Exchange数据集**表现不佳：MSE 0.3632，弱于DLinear（0.354）和S-Mamba/iTransformer（约0.36），说明图结构对强周期性/趋势性数据的建模并非总是最优。



**效率与消融**方面，Table 8比较了三种MST构建算法：Chazelle's、Kruskal、Prim。准确率差异在±0.1%以内，但Chazelle实现**0.90×的相对epoch时间**（相比Kruskal），验证了近线性复杂度的实际收益。事件眼动追踪任务（Table 3）凸显极致效率：GG-SSM仅**62k参数、3.01M MACs**，对比3ET的107M MACs压缩35.5倍，与Retina（3.03M MACs）相当但优于VMamba+Mamba（4.52M MACs）。

**公平性检验**: 对比基线选择总体合理，VMamba/Swin/DeiT为视觉SSM/CNN/Transformer的代表性方法，iTransformer/DLinear为时间序列预测的主流基线。但存在两处局限：一是未与更新的Mamba-2或Griffin/Gemma等强基线比较；二是消融研究较单薄——仅比较MST算法，缺乏"图结构 vs 无图结构"的直接消融，无法完全隔离图生成机制本身的贡献。

## 方法谱系与知识库定位

GG-SSM属于**状态空间模型（SSM）家族**，直接父方法为 **Mamba**（Gu & Dao, 2023）。演化路径为：S4/S5/S7系列奠定SSM数学基础 → Mamba引入选择性机制实现输入依赖的状态转移 → **GG-SSM突破一维序列限制，将状态传播推广到任意图结构**。

**改变的插槽**（5个维度中的3个）：
- **architecture**: 线性序列扫描 → 图路径矩阵连乘 + 全局邻居聚合
- **inference_strategy**: 固定扫描顺序 → Chazelle近线性MST动态图构建
- **data_pipeline**: 固定token位置/展平顺序 → 余弦相似度驱动的自适应边权重

**直接基线对比**:
- **Mamba**: GG-SSM保留A,B,C,D矩阵框架，但将Āⁿ替换为路径连乘S_ji，将递推替换为全局求和
- **VMamba**: 采用相同的视觉训练协议和分层结构，但VMamba使用四向交叉扫描伪二维化，GG-SSM通过MST实现真正的自适应非局部连接
- **S-Mamba**: 时间序列领域最直接的SSM竞争者，GG-SSM在Weather/Solar上显著领先，但在Exchange上落后

**后续方向**:
1. **与Mamba-2的融合**: Mamba-2提出的SSD框架（Structured State Space Duality）可与图传播结合，探索图注意力与SSM的对偶关系
2. **可学习MST替代方案**: 当前MST为确定性算法，未来可探索可微图稀疏化或Gumbel-Softmax采样
3. **多模态扩展**: 事件相机+图像+文本的异构图构建，验证GG-SSM在跨模态场景的通用性

**知识库标签**: #modality=vision/time-series/events #paradigm=state-space-model #scenario=efficient-long-sequence-modeling #mechanism=graph-message-passing #constraint=near-linear-complexity

