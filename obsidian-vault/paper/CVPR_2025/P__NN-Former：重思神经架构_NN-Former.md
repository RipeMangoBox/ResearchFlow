---
title: 'NN-Former: Rethinking Graph Structure in Neural Architecture Representation'
type: paper
paper_level: C
venue: CVPR
year: 2025
paper_link: null
aliases:
- NN-Former：重思神经架构表示的图结构
- NN-Former
acceptance: poster
cited_by: 1
method: NN-Former
---

# NN-Former: Rethinking Graph Structure in Neural Architecture Representation

**Topics**: [[T__Neural_Architecture_Search]], [[T__Graph_Learning]] | **Method**: [[M__NN-Former]] | **Datasets**: [[D__NAS-Bench-101]] (其他: NNLQ, Cora citation prediction)

| 中文题名 | NN-Former：重思神经架构表示的图结构 |
| 英文题名 | NN-Former: Rethinking Graph Structure in Neural Architecture Representation |
| 会议/期刊 | CVPR 2025 (poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2507.00880) · [Code](待补充) · [Project](待补充) |
| 主要任务 | 神经架构表示学习（Neural Architecture Representation Learning）、基于预测器的神经架构搜索（Predictor-based NAS）、DAG 表示学习 |
| 主要 baseline | NAR-Former（直接基线）、DAG Transformer、FlowerFormer、PINAT、BRP-NAS、Generic Graph-based Neural Architecture Encoding Scheme |

> [!abstract]
> 因为「标准 Transformer 的全局自注意力无法捕捉 DAG 的拓扑结构（前向/反向边、兄弟节点关系），且需要额外的位置编码」，作者在「NAR-Former」基础上改了「用 ASMA 替换全局注意力、用 BGIFFN 替换标准 FFN、移除位置编码」，在「NAS-Bench-101（0.04% 训练数据）」上取得「Kendall's Tau 0.7654，相比 vanilla transformer 提升 +0.3056（66.4% 相对提升）」

- **NAS-Bench-101**: Kendall's Tau 0.7654 vs. global attention + vanilla FFN 0.4598，提升 +66.4%
- **NNLQ 跨平台延迟预测**: MAPE 7.93 vs. global attention 10.83，降低 26.8%；Acc(10%) 69.9 vs. 58.45，提升 +11.45 个百分点
- **Cora 引文预测**: Accuracy 88.14 vs. DAG Transformer 87.39，兄弟节点建模带来 +0.75 提升

## 背景与动机

神经架构搜索（NAS）需要快速评估大量候选网络的性能，但完全训练每个架构成本极高。预测器方法通过学习架构的向量表示来估计其准确率或延迟，核心挑战在于：如何有效编码计算图（DAG）的拓扑结构？

现有方法各有局限。**NAR-Former** 使用全局自注意力处理架构图，但将 DAG 当作普通序列，忽略了边方向性；它依赖位置编码（PE）注入拓扑信息，却需要精心设计。**DAG Transformer** 虽为 DAG 定制了注意力机制，但未显式建模兄弟节点关系——即共享同一前驱或后继的节点，而这些节点在神经网络中常对应并行操作（如 ResNet 的多个分支）。**FlowerFormer** 通过流感知编码建模信息流动，但仍未系统解构 DAG 的多重拓扑关系。

具体而言，一个神经网络的计算图中，节点 v 的前向邻居（后继）决定数据流向，反向邻居（前驱）关联梯度回传，而兄弟节点（如被同一卷积层分出的两个分支）具有功能相似性。全局注意力让所有节点两两交互，引入大量无关噪声；标准 FFN 作为位置无关的前馈变换，完全丢弃了图结构信息；位置编码（如 Laplacian PE）虽能编码距离，却增加了超参数负担，且实验表明可能损害性能。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/bce6c7d6-33a5-4f10-bcba-7cf923411ea2/figures/Figure_1.png)
*Figure 1 (comparison): Comparison of different methods of DAG representation of neural architectures.*



本文的核心动机是：**DAG 的拓扑信息应直接嵌入注意力掩码和前馈变换，而非依赖外部位置编码**。作者提出 NN-Former，用结构化的多头注意力替代全局注意力，用双向图感知前馈网络替代标准 FFN，从而在不增加位置编码的情况下，系统捕获前向边、反向边、后继兄弟、前驱兄弟四类拓扑关系。

## 核心创新

核心洞察：**DAG 的邻接矩阵及其二次型（AA^T, A^TA）天然编码了四类关键拓扑关系**，因为前向邻接 A 捕获数据流、反向邻接 A^T 捕获梯度流、AA^T 捕获后继兄弟（共享前驱）、A^TA 捕获前驱兄弟（共享后继），从而使**无需位置编码的纯结构感知架构编码**成为可能。

| 维度 | Baseline（NAR-Former / Vanilla Transformer） | 本文（NN-Former） |
|:---|:---|:---|
| 注意力机制 | 全局自注意力，所有节点两两交互 | ASMA：4 个拓扑专用头（A, A^T, AA^T, A^TA） |
| 前馈网络 | 标准 FFN，与图结构无关 | BGIFFN：注入双向图卷积的前馈网络 |
| 位置编码 | 必需（NAR PE 或 Laplacian PE） | **完全移除**，拓扑信息由结构掩码内在捕获 |
| 兄弟节点建模 | 未显式建模 | AA^T 和 A^TA 头显式捕获兄弟关系 |
| 方向性处理 | 隐式或忽略 | 前向/反向分离处理，保留方向信息 |

## 整体框架


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/bce6c7d6-33a5-4f10-bcba-7cf923411ea2/figures/Figure_2.png)
*Figure 2 (pipeline): Overview of the NN-Former framework.*



NN-Former 的整体流程遵循标准 Transformer 的层叠结构，但将核心组件替换为拓扑感知版本：

**输入嵌入层**：接收节点特征 Z（如操作类型 embedding）和邻接矩阵 A，输出初始隐状态 H^0。

**ASMA 层（Architecture-aware Structural Multi-head Attention）**：替代标准多头自注意力。输入为层归一化后的 H^{l-1} 和邻接矩阵 A，通过 4 个结构掩码头计算注意力，输出与输入残差相加得到 \hat{H}^{l-1}。这是捕获 DAG 拓扑的核心模块。

**BGIFFN 层（Bidirectional Graph-Informed Feed-Forward Network）**：替代标准 FFN。输入为层归一化后的 \hat{H}^{l-1} 和 A，在标准线性变换中注入前向图卷积 GC(H,A) 和反向图卷积 GC(H,A^T)，输出与输入残差相加得到 H^l。

**输出 MLP**：将最终层输出 H^L 映射到预测值 \hat{y}（如准确率或延迟）。

层叠 L 个 ASMA+BGIFFN 块后接输出层，形成完整的预测器。关键设计是**预归一化（Pre-LN）**结构：每个子层前均有 LayerNorm，保证训练稳定性。

数据流示意：
```
(Z, A) → Embedding → H^0
  → [LayerNorm → ASMA(H,A) → +Residual] → \hat{H}^{l-1}
  → [LayerNorm → BGIFFN(H,A) → +Residual] → H^l
  → ... (重复 L 层)
  → MLP → \hat{y}
```

## 核心模块与公式推导

### 模块 1: ASMA（架构感知结构多头注意力）（对应框架图左侧）

**直觉**：标准 Transformer 的全局注意力让查询节点关注所有其他节点，但 DAG 中节点只应与拓扑相关的邻居交互——直接后继、直接前驱、以及"兄弟"节点（共享同一前驱/后继的并行操作）。

**Baseline 公式** (Vanilla Multi-Head Attention):
$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
符号: $Q,K,V$ 为查询/键/值矩阵，$d_k$ 为键维度，softmax 实现全局归一化。

**变化点**：全局 softmax 导致非拓扑相关节点间的噪声交互；且 baseline 无显式方向性和兄弟关系建模。

**本文公式（推导）**:
$$\text{Step 1}: X_1 = \sigma\left(\frac{Q_1K_1^T \circ (I + A)}{\sqrt{h}}\right)V_1 \quad \text{加入前向邻接掩码，只关注节点及其直接后继}$$
$$\text{Step 2}: X_2 = \sigma\left(\frac{Q_2K_2^T \circ (I + A^T)}{\sqrt{h}}\right)V_2 \quad \text{加入反向邻接掩码，捕获梯度回传结构}$$
$$\text{Step 3}: X_3 = \sigma\left(\frac{Q_3K_3^T \circ (I + AA^T)}{\sqrt{h}}\right)V_3 \quad \text{AA}^T\text{编码后继兄弟：两节点有共同前驱}$$
$$\text{Step 4}: X_4 = \sigma\left(\frac{Q_4K_4^T \circ (I + A^TA)}{\sqrt{h}}\right)V_4 \quad \text{A}^TA\text{编码前驱兄弟：两节点有共同后继}$$
$$\text{最终}: \text{ASMA}(H) = \text{Concat}(X_1, X_2, X_3, X_4)W^O$$
其中 $\sigma$ 为激活函数，$\circ$ 为逐元素乘法，$I$ 保证自连接，$h$ 为每头维度，$W^O$ 为输出投影。

**对应消融**：Table 7 显示，仅保留 A+A^T（无兄弟头）得 Kendall's Tau 0.7573，加入 AA^T+A^TA 后提升至 0.7654，兄弟头贡献 +0.0081。

### 模块 2: BGIFFN（双向图感知前馈网络）（对应框架图右侧）

**直觉**：标准 FFN 是位置无关的全连接变换，对图结构"盲视"；应将邻域聚合直接注入前馈变换，且前向/反向邻居需分离处理以保留方向语义。

**Baseline 公式** (Vanilla FFN):
$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$
符号: $W_1, W_2$ 为线性权重，与图结构无关。

**变化点**：标准 FFN 对所有节点应用相同变换，无法区分 DAG 中的前向/反向信息流；且单方向图卷积会丢失方向性。

**本文公式（推导）**:
$$\text{Step 1}: H_g = \text{Concat}\left(\text{GC}(H, A), \text{GC}(H, A^T)\right) \quad \text{分离前向/反向图卷积，保留方向信息}$$
$$\text{其中}: \text{GC}(H, A) = AHW \quad \text{基础邻域聚合}$$
$$\text{Step 2}: \text{BGIFFN}(H, A) = \text{ReLU}\left(HW_1 + H_g\right)W_2 \quad \text{图分支与标准线性变换相加，注入拓扑信息}$$

**对应消融**：Table 9 显示，仅使用 A^T 方向得 0.7501，仅使用 A 方向得 0.7253，双向 Concat 得 0.7654；双向结构较最优单方向提升 +0.0153。Table 6 显示 global attention + BGIFFN 已达 0.7656，接近完整模型 0.7654，证明 BGIFFN 本身贡献显著。

## 实验与分析


![Table 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/bce6c7d6-33a5-4f10-bcba-7cf923411ea2/figures/Table_2.png)
*Table 2 (result): Accuracy prediction results on NAS-Bench-201.*



本文在三个核心基准上验证 NN-Former：NAS-Bench-101（架构属性预测）、NNLQ（跨平台延迟预测）、以及 Cora（一般 DAG 的兄弟节点建模）。

在 **NAS-Bench-101** 上，仅使用 0.04% 训练数据（约 169 个架构），NN-Former 取得 Kendall's Tau 0.7654。这一结果的关键对照来自消融实验 Table 6：vanilla transformer（全局注意力 + 标准 FFN）仅得 0.4598，ASMA 单独替换注意力得 0.6538，BGIFFN 单独替换 FFN 得 0.7656——**BGIFFN 单模块的提升幅度（+0.3058）甚至略超完整模型**，说明图感知前馈网络是性能跃迁的核心引擎。完整模型与 global+BGIFFN 基本持平（0.7654 vs 0.7656），暗示 ASMA 与 BGIFFN 存在一定功能重叠，但 ASMA 的方向性分解仍具价值。

在 **NNLQ 跨平台延迟预测**（out-of-domain，NAS-Bench-201 族）上，NN-Former 的 MAPE 为 7.93，相比 global attention 基线 10.83 降低 26.8%；Acc(10%) 为 69.9，相比 58.45 提升 +11.45 个百分点。这表明拓扑感知设计对延迟这一与硬件强相关的属性同样有效。


![Table 5](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/bce6c7d6-33a5-4f10-bcba-7cf923411ea2/figures/Table_5.png)
*Table 5 (ablation): Ablation study on the topological structure of ASMA.*



消融实验进一步验证各组件的必要性。Table 7 的系统掩码组合显示：四头全开（A, A^T, A^TA, AA^T）得 0.7654 为最优；移除兄弟头（仅 A+A^T）降至 0.7573（-0.0081），移除反向头（仅 A+AA^T）降至 0.7566（-0.0088）。**位置编码的消融（Table 8）最具反直觉性**：添加 NAR-Former 的 PE 降至 0.7449（-0.0205），添加 Laplacian PE 更降至 0.7063（-0.0591）——结构掩码已充分编码位置信息，额外 PE 反而引入噪声。Table 9 中 BGIFFN 的双向设计验证：单向前向 0.7253，单向后向 0.7501，双向 0.7654，证明方向分离的必要性。

**公平性检查**：本文主要与 NAR-Former 对比计算成本（参数量 4.9M vs 4.8M，推理延迟 11.53ms vs 10.31ms，训练时间 0.8h vs 0.7h），成本增幅可控（+11.8% 延迟）。但 Table 1 的完整 baseline 对比未在提供的摘录中显示，FlowerFormer、PINAT、NAR-Former V2 等基线的直接数值对比缺失；作者仅声称"consistently outperforms"而未给出具体数字。此外，NN-Former 在 NAS-Bench-101 上未明确声称 SOTA，0.7654 的绝对水平需结合完整文献判断。

## 方法谱系与知识库定位

**方法家族**：Graph Transformer for DAG Representation → Neural Architecture Encoding

**父方法**：NAR-Former（"Neural Architecture Representation Learning Towards Holistic Attributes Prediction"）。NN-Former 直接继承其整体预测器框架，但系统改造了四个关键槽位：
- **注意力机制**：全局自注意力 → ASMA（4 拓扑头结构掩码）
- **前馈网络**：标准 FFN → BGIFFN（双向图卷积分支）
- **位置编码**：NAR PE / Laplacian PE → **完全移除**
- **兄弟节点建模**：无 → AA^T / A^TA 显式掩码

**直接基线差异**：
- **DAG Transformer**：同为 DAG 专用 Transformer，但无兄弟节点建模（Cora 上 87.39 vs NN-Former 88.14）
- **FlowerFormer**：强调信息流方向，但未系统解构为 4 类拓扑关系
- **PINAT / BRP-NAS / Generic Graph-based**：传统 GNN/Transformer 基线，无结构掩码设计

**后续方向**：(1) 将 ASMA 的 4 头设计扩展至动态权重学习，根据架构类型自适应调整头重要性；(2) 结合 NAR-Former V2 的通用网络表示能力，验证 NN-Former 在 Transformer/ViT 等复杂拓扑上的扩展性；(3) 探索 BGIFFN 图卷积的轻量替代（如谱方法），进一步降低 +11.8% 的延迟开销。

**知识标签**：
- **模态**（Modality）：图结构数据 / 神经架构 DAG
- **范式**（Paradigm）：Predictor-based NAS / 表示学习
- **场景**（Scenario）：少样本架构属性预测（0.04% 数据）、跨域延迟预测
- **机制**（Mechanism）：结构掩码注意力、双向图卷积、兄弟节点建模
- **约束**（Constraint）：无位置编码、预归一化、残差连接保持

