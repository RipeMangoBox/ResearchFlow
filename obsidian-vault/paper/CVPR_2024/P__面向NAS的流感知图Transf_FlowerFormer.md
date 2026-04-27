---
title: 'FlowerFormer: Empowering Neural Architecture Encoding using a Flow-aware Graph Transformer'
type: paper
paper_level: C
venue: CVPR
year: 2024
paper_link: null
aliases:
- 面向NAS的流感知图Transformer编码器
- FlowerFormer
acceptance: Poster
cited_by: 11
method: FlowerFormer
---

# FlowerFormer: Empowering Neural Architecture Encoding using a Flow-aware Graph Transformer

**Topics**: [[T__Neural_Architecture_Search]] | **Method**: [[M__FlowerFormer]] | **Datasets**: [[D__NAS-Bench-101]], [[D__NAS-Bench-201]] (其他: NAS-Bench-301)

| 中文题名 | 面向NAS的流感知图Transformer编码器 |
| 英文题名 | FlowerFormer: Empowering Neural Architecture Encoding using a Flow-aware Graph Transformer |
| 会议/期刊 | CVPR 2024 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2403.12821) · [Code] · [Project] |
| 主要任务 | 神经架构性能预测 (Neural Architecture Performance Prediction) |
| 主要 baseline | GatedGCN, DAGNN, GraphGPS, DAGFormer, TA-GATES, NAR-Former |

> [!abstract] 因为「标准图神经网络/Transformer 无法捕捉神经网络 DAG 中数据前向传播与梯度反向传播的方向性信息流动」，作者在「GraphGPS/DAGFormer 等图 Transformer」基础上改了「引入异步前向-反向消息传递的流编码模块 + 流感知全局注意力」，在「NAS-Bench-101/201/301」上取得「Kendall's Tau 75.0/79.0/64.2 (1% 训练数据)，超越最优 baseline +2.6/+3.2/+2.4」

- **NAS-Bench-101 @ 1% 训练**: Kendall's Tau 75.0，超 DAGNN (72.4) +2.6
- **NAS-Bench-201 @ 1% 训练**: Kendall's Tau 79.0，超 DAGNN (75.8) +3.2  
- **NAS-Bench-101 @ 50% 训练**: Kendall's Tau 89.6，超 DAGNN/GraphGPS (85.9) +3.7

## 背景与动机

神经架构搜索 (NAS) 的核心瓶颈在于：评估一个候选架构的真实性能需要完整训练，计算成本极高。因此，**神经架构性能预测**——即用一个轻量编码器快速估计架构的最终准确率——成为加速 NAS 的关键技术。例如，在 NAS-Bench-101 中，一个 cell-based 的 CNN 架构可表示为有向无环图 (DAG)，节点是操作（如 3×3 卷积、池化），边是数据流连接。

现有方法如何处理这一问题？**GatedGCN** 和 **DAGNN** 作为 GNN baseline，采用同步消息传递机制，所有节点在同一层同时更新，忽略了 DAG 中信息流动的方向性和时序性。**GraphGPS** 和 **DAGFormer** 引入全局自注意力，让任意节点对直接交互，但注意力权重对数据流方向不敏感，无法区分"前向数据传播"与"反向梯度传播"。**TA-GATES** 和 **NAR-Former** 作为 SOTA 方法，通过复杂预处理（如特定操作的特殊处理、同构增强）来提升性能，但这些技巧增加了部署难度且泛化性受限。

这些方法的**根本缺陷**在于：它们将神经网络架构视为普通图或普通 DAG，却忽略了**神经网络的本质特征——信息以特定方向流动**：训练时数据从输入层前向传播至输出层，梯度则从输出层反向传播至输入层。标准同步 MPNN 或纯全局注意力无法显式建模这种方向性的、分阶段的信息流动，导致架构编码丢失关键的语义结构。此外，SOTA 方法的复杂预处理（如 TA-GATES 的拓扑感知门控、NAR-Former 的同构增强）虽能提升性能，却牺牲了通用性和可扩展性。

本文提出 **FlowerFormer**，核心思想是：让图 Transformer **感知并显式编码神经网络中的信息流**——通过异步前向-反向消息传递模拟数据流与梯度流，无需任何复杂预处理即可实现强大的架构编码。


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/f55a77df-073d-4c08-802d-4d0191af77e2/figures/fig_001.jpeg)
*Figure: The information flows of a neural archi-*



## 核心创新

**核心洞察：神经网络 DAG 中的性能关键信息蕴含在方向性的数据流与梯度流中，因为标准同步消息传递和全局注意力无法区分这些流动方向，从而使显式的异步拓扑消息传递成为必要。**

具体而言，FlowerFormer 将传统图 Transformer 的"一视同仁"的节点交互，替换为**按拓扑世代顺序执行的前向-反向异步消息传递**：前向传递按拓扑升序模拟数据从输入到输出的传播，反向传递按拓扑降序模拟梯度从输出到输入的回传。这种设计让模型无需任何领域特定的预处理（如 TA-GATES 的操作特殊处理或 NAR-Former 的同构增强），即可捕捉架构的本质执行语义。

| 维度 | Baseline (GraphGPS/DAGFormer) | 本文 (FlowerFormer) |
|:---|:---|:---|
| 消息传递机制 | 同步更新：所有节点同层同时聚合邻居 | 异步更新：按拓扑世代顺序分阶段传播 |
| 信息流方向 | 无方向感知，邻居关系对称处理 | 显式前向（数据流）+ 反向（梯度流）双向编码 |
| 全局注意力 | 标准自注意力，结构无关 | 流感知全局注意力，融入流动结构先验 |
| 输入预处理 | 需复杂处理（TA-GATES/NAR-Former） | 仅需邻接矩阵 A + one-hot 节点特征 X |

## 整体框架



FlowerFormer 的整体流程可概括为：**输入 DAG → 堆叠 FLOWER 层 → 预测头输出**。具体数据流如下：

1. **Input modeling（输入建模）**：将神经网络架构表示为 DAG $G = (A, X)$，其中 $A \in \{0,1\}^{N \times N}$ 为邻接矩阵，$X \in \{0,1\}^{N \times D}$ 为 one-hot 编码的节点操作类型矩阵。**无需任何额外预处理**（如特殊操作处理或同构增强）。

2. **Flow encode module（流编码模块）**：核心创新模块，接收节点嵌入矩阵 $H$ 和图 $G$，执行**异步前向消息传递**（按拓扑世代升序，模拟数据前向传播）后接**异步反向消息传递**（按拓扑世代降序，模拟梯度反向传播），输出融入流动信息的更新嵌入 $H'$。

3. **Flow-aware global attention module（流感知全局注意力模块）**：接收流编码后的节点嵌入，执行结构感知的全局自注意力，让远距离节点也能交互，同时保持对流动结构的敏感性，输出全局聚合后的嵌入。

4. **FLOWER layer（FLOWER 层）**：将上述两个模块组合为基本构建单元，可重复堆叠 $L$ 层形成深度编码器。

5. **Prediction head（预测头）**：对最终节点/图嵌入进行池化，输出架构性能预测（如测试准确率）。

```
输入架构 DAG (A, X)
    ↓
[FLOWER Layer × L]
  ├─ Flow Encode Module: 异步前向 MP → 异步反向 MP
  └─ Flow-aware Global Attention: 结构感知自注意力
    ↓
Prediction Head → 性能分数
```

## 核心模块与公式推导

### 模块 1: 异步前向消息传递（对应框架图 Flow encode module 前半部分）

**直觉**: 神经网络的数据从输入层逐层流向输出层，消息传递也应遵循这一时序，而非同时更新所有节点。

**Baseline 公式** (标准同步 MPNN, e.g. GatedGCN/DAGNN):
$$h_j^{(l+1)} \leftarrow \text{UPDATE}^{(l)}\left(h_j^{(l)}, \text{AGGREGATE}^{(l)}\{h_i^{(l)} : (i,j) \in E\}\right)$$
符号: $h_j^{(l)}$ = 节点 $j$ 在第 $l$ 层的嵌入；所有节点使用**同一层**的邻居嵌入同步更新。

**变化点**: 标准同步 MPNN 中，节点 $j$ 的邻居 $i$ 可能尚未获得稳定的第 $l$ 层表示就参与聚合，导致信息混乱；且无法区分"前驱"与"后继"的语义差异。

**本文公式（推导）**:
$$\text{Step 1}: \quad T_G = \text{TopologicalSort}(G) \quad \text{（将 DAG 节点分层为拓扑世代）}$$
$$\text{Step 2}: \quad \text{for } k = 1, \dots, |T_G|, \quad \text{for } v_j \in T_G^k: \quad h_j \leftarrow \text{Comb}\left(h_j, \text{Agg}\{m_e(h_j, h_i) : A_{ij} = 1\}\right) \quad \text{（按拓扑升序，聚合前驱节点消息，模拟数据前向流）}$$
$$\text{最终}: \quad h_j^{\text{forward}} = \text{Comb}\left(h_j, \text{Agg}_{i: A_{ij}=1}\{m_e(h_j, h_i)\}\right)$$

**对应消融**: 去掉异步机制（改为同步）后，NAS-Bench-101 @ 1% 从 75.0 降至 65.5，$-9.5$ Kendall's Tau。

---

### 模块 2: 异步反向消息传递（对应框架图 Flow encode module 后半部分）

**直觉**: 训练时的梯度从损失层反向传播回参数，这种"反向流动"蕴含架构优化难度的重要信息。

**Baseline 公式**: 标准 MPNN **无反向传递设计**，或仅通过多层堆叠隐式传播。

**变化点**: 仅有前向传递只能编码推理时的数据流，无法捕捉训练动态；需显式引入反向传递以完整建模神经网络的双向信息流。

**本文公式（推导）**:
$$\text{Step 1}: \quad \text{for } k = |T_G|, \dots, 1, \quad \text{for } v_j \in T_G^k: \quad h_j \leftarrow \text{Comb}\left(h_j, \text{Agg}\{m_e(h_j, h_i) : A_{ji} = 1\}\right) \quad \text{（按拓扑降序，聚合后继节点消息，模拟梯度反向流）}$$
$$\text{最终}: \quad h_j^{\text{backward}} = \text{Comb}\left(h_j^{\text{forward}}, \text{Agg}_{i: A_{ji}=1}\{m_e(h_j^{\text{forward}}, h_i)\}\right)$$
符号: $A_{ji}=1$ 表示存在边 $j \rightarrow i$，即 $i$ 是 $j$ 的后继；此时聚合的是**反向"梯度"信号**。

**对应消融**: 去掉反向传递（仅保留前向）后，NAS-Bench-101 @ 5% 从 86.1 降至 83.9，$-2.2$ Kendall's Tau；@ 1% 略有异常提升 (+1.7)，作者未充分解释。

---

### 模块 3: 流感知全局注意力（对应框架图 FLOWER layer 后半部分）

**直觉**: 纯局部消息传递难以捕捉长距离依赖，但标准全局注意力会破坏流动结构；需在注意力中注入流向感知。

**Baseline 公式** (标准 Graph Transformer attention):
$$\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**变化点**: 标准注意力对所有节点对一视同仁，无法利用 DAG 的拓扑结构；本文修改注意力机制使其"感知"节点间的流动关系（具体实现细节原文未完全展开，但核心是在位置编码或注意力偏置中融入拓扑世代信息）。

**本文公式**: 原文未给出显式修改后的注意力公式，但强调为"flow-aware global attention module"，与 flow encode module 互补。

**对应消融**: 去掉流感知注意力（GA）后，NAS-Bench-101 @ 5% 从 86.1 降至 83.2，$-2.9$ Kendall's Tau；@ 1% 同样略有异常 (+1.5)。

## 实验与分析



本文在 5 个 NAS benchmark 上评估 FlowerFormer，覆盖计算机视觉（NAS-Bench-101/201/301）、图神经网络（NAS-Bench-Graph）和语音识别（NAS-Bench-ASR）领域。核心评估指标为 Kendall's Tau (×100)，衡量预测排序与真实排序的一致性，对 NAS 中筛选优质架构至关重要。

**主实验结果**：在计算机视觉基准上，FlowerFormer 展现了极强的**低数据场景优势**。NAS-Bench-101 @ 1% 训练数据达到 Kendall's Tau 75.0，超越此前最优的 DAGNN (72.4) **+2.6**；@ 50% 训练数据进一步提升至 89.6，超越 DAGNN/GraphGPS (85.9) **+3.7**。NAS-Bench-201 @ 1% 达到 79.0，超越 DAGNN (75.8) **+3.2**；@ 50% 时达到 92.9，与 DAGNN (92.6) 基本持平。NAS-Bench-301 @ 1% 达到 64.2，超越 GatedGCN (61.8) **+2.4**。跨域泛化方面，NAS-Bench-Graph @ 1% 达到 49.5，超越 DAGNN (48.1) **+1.4**；但 NAS-Bench-ASR @ 1% 为 31.1，低于 TA-GATES (34.0) **-2.9**，不过在 5% 数据时反超 (44.0 vs 41.4)。



**消融实验**揭示了各组件的贡献差异。**Flow encode module 整体最为关键**：完全移除前向+反向传递后，NAS-Bench-101 @ 1% 从 75.0 暴跌至 41.5，**-33.5**，验证了流编码的核心价值。**异步机制（AS）单独移除**也造成显著下降：65.5 vs 75.0，**-9.5**，说明按拓扑顺序分阶段更新优于同步更新。然而，**反向传递（FB）和流感知注意力（GA）的消融出现反常**：在 1% 数据下，单独移除二者反而有微小提升（FB: +1.7, GA: +1.5），仅在 5% 数据下显示负面影响（FB: -2.2, GA: -2.9）。这一矛盾作者未给出充分解释，削弱了"所有组件均必要"的声称。

**公平性检查**：Baseline 选择较为全面，涵盖 GNN (GatedGCN, DAGNN)、图 Transformer (GraphGPS, DAGFormer) 和 SOTA 专用编码器 (TA-GATES, NAR-Former)。但存在几点问题：(1) TA-GATES 专为 CV 领域设计，却在非 CV 领域（Graph, ASR）与其比较，对其不利；(2) NAR-Former 因双 cell 架构不兼容未在 NAS-Bench-301 上测试；(3) 速度对比中排除了 NAR-Former 的同构增强步骤，可能偏袒 FlowerFormer。训练在 NVIDIA RTX 2080 上进行，200 epochs，batch size 128，具体耗时见 Table 6（数据未完全展示）。

## 方法谱系与知识库定位

**方法谱系**：FlowerFormer 属于**图 Transformer (Graph Transformer) 家族**，直接继承自 **GraphGPS** 和 **DAGFormer** 的架构范式——即"局部消息传递 + 全局注意力"的层次化设计。具体改变 slots 为：**architecture**（替换标准模块为流感知变体）和 **inference_strategy**（同步 → 异步拓扑消息传递）。数据管道 slot 虽被修改（简化预处理），但属于"去除复杂性"而非引入新机制。

**直接 Baseline 对比**：
- **GraphGPS**: 通用图 Transformer，FlowerFormer 继承其 MPNN+Attention 分层结构，但将同步 MPNN 替换为异步流编码模块。
- **DAGFormer**: DAG 专用 Transformer，FlowerFormer 在其 DAG 感知基础上进一步引入**方向性流动语义**（前向/反向），而非仅利用拓扑排序做位置编码。
- **TA-GATES/NAR-Former**: SOTA 专用编码器，FlowerFormer 放弃其复杂预处理路线，证明**结构化的消息传递机制本身足以替代领域技巧**。

**后续方向**：(1) 将流感知机制扩展到动态图/时序架构搜索；(2) 结合可学习的重要性权重，自适应平衡前向与反向传递的贡献（当前消融显示二者交互复杂）；(3) 探索流编码在架构生成（而非仅预测）中的直接应用。

**标签**：modality: 图结构数据 (DAG) | paradigm: 图 Transformer + 异步消息传递 | scenario: 神经架构搜索加速 | mechanism: 拓扑感知的前向-反向流动编码 | constraint: 低训练数据场景 (1%-50%)，无需复杂预处理

