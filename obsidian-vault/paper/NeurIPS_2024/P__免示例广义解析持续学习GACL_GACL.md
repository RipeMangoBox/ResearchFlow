---
title: 'GACL: Exemplar-Free Generalized Analytic Continual Learning'
type: paper
paper_level: C
venue: NeurIPS
year: 2024
paper_link: null
aliases:
- 免示例广义解析持续学习GACL
- GACL
acceptance: Poster
cited_by: 26
code_url: https://github.com/ZHUANGHP/Analytic-continual-learning
method: GACL
---

# GACL: Exemplar-Free Generalized Analytic Continual Learning

[Code](https://github.com/ZHUANGHP/Analytic-continual-learning)

**Topics**: [[T__Continual_Learning]] | **Method**: [[M__GACL]] | **Datasets**: [[D__CIFAR-100]], [[D__ImageNet-1K]], [[D__Tiny-ImageNet]]

| 中文题名 | 免示例广义解析持续学习GACL |
| 英文题名 | GACL: Exemplar-Free Generalized Analytic Continual Learning |
| 会议/期刊 | NeurIPS 2024 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2403.15706) · [Code](https://github.com/ZHUANGHP/Analytic-continual-learning) · [Project](待补充) |
| 主要任务 | 免示例持续学习（Exemplar-Free Continual Learning, EFCIL）、类增量学习（Class-Incremental Learning） |
| 主要 baseline | SLDA, MVP, LwF, EWC++, L2P, DualPrompt, RM, ER, MVP-R |

> [!abstract] 因为「持续学习中的灾难性遗忘问题与存储旧样本的隐私/内存开销」，作者在「Analytic Continual Learning (ACL)」基础上改了「引入正则化的自相关记忆矩阵与递归最小二乘更新，彻底消除对示例回放的依赖」，在「CIFAR-100 / ImageNet-R / Tiny-ImageNet」上取得「CIFAR-100 AAUC 57.99%（超越SLDA +4.99pp），训练时间仅611秒（vs RM >2天）」

- **CIFAR-100 AAUC**: 57.99% vs SLDA 53.0%（+4.99pp），训练时间 611s vs RM >2天
- **ImageNet-R AAUC**: 41.68% vs LwF 38.65%（+3.03pp）
- **Tiny-ImageNet AAUC**: 63.14% vs SLDA 62.74%（+0.40pp）

## 背景与动机

持续学习（Continual Learning）旨在让模型按顺序学习多个任务，同时不遗忘先前知识。一个典型场景是：智能家居系统需逐月识别新类别的设备图像，但无法永久存储用户的历史照片（隐私限制）。现有方法面临两难困境——基于回放的方案（如RM、ER、MVP-R）需存储旧样本，违反隐私；免示例方法（如EWC++、LwF）虽省内存，但依赖梯度优化，易陷入灾难性遗忘。

现有三类主流路径及其局限：
- **回放方法（RM / ER / MVP-R）**：存储并复演旧样本，性能强但标记为✕（非免示例），训练时间超2天，隐私不兼容。
- **梯度正则化（EWC++ / LwF）**：通过约束参数变化或知识蒸馏保护旧知识，但AAUC仅33.4%（LwF on CIFAR-100），遗忘严重。
- **提示学习（L2P / DualPrompt）**：为每个任务学习视觉提示，AAUC达65.1-65.6，但依赖大规模预训练与复杂提示机制，且非纯解析方法。

核心痛点在于：**免示例与强性能不可兼得**。梯度优化本质上是局部、迭代的，在序列任务中必然漂移；而解析方法（Analytic Learning）虽具闭式解，却未解决矩阵奇异性与持续更新问题。本文提出GACL，以正则化自相关记忆矩阵替代梯度下降，首次在免示例约束下实现高效、稳定的递归解析更新。



## 核心创新

核心洞察：持续学习的本质是「知识的递归最小二乘累积」，而非「梯度的随机漫步」，因为自相关矩阵 $X^\text{top} X$ 已编码全部二阶统计信息，从而使免示例、闭式、无遗忘的持续学习成为可能。

| 维度 | Baseline (ACL/梯度方法) | 本文 GACL |
|:---|:---|:---|
| **记忆机制** | 梯度权重 / 回放缓冲区 | 正则化自相关矩阵 $R = (X^\text{top} X + \gamma I)^{-1}$ |
| **优化方式** | SGD/Adam 迭代更新 | 递归最小二乘闭式解，无迭代 |
| **示例依赖** | 回放方法需存旧样本；EWC++/LwF虽免示例但梯度弱 | 完全免示例（✓），解析强更新 |
| **数值稳定性** | 梯度消失/爆炸；$X^\text{top} X$ 可能奇异 | 正则化项 $\gamma I$ 保证可逆 |
| **训练效率** | RM >2天；LwF需多轮蒸馏 | CIFAR-100仅611秒 |

GACL的广义性体现在：不假设任务边界已知（Generalized CIL, GCIL），同时兼容 blurry 与 clear 设定。

## 整体框架


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/1fd62c83-e375-46dd-ac64-85dd1ad20ced/figures/Figure_1.png)
*Figure 1: Figure 1: An overview of our proposed GACL. (a) Labels of the exposed class and the unexposedclass are extracted in each GCIL task (see definition in Section 3.2), respectively. (b) A frozenpre-traine*



GACL 的数据流由四个模块串联，完全绕开梯度反向传播：

1. **Feature Extraction（特征提取）**：输入图像经预训练 backbone（如 ResNet）提取特征表示 $X \in \mathbb{R}^{N \times d}$，输出固定维特征向量。该模块为现成网络，不参与持续学习阶段的参数更新。

2. **Autocorrelation Matrix Computation（自相关矩阵计算）**：接收特征 $X$，计算正则化自相关矩阵 $R_t = (X_t^\text{top} X_t + \gamma I)^{-1} \in \mathbb{R}^{d \times d}$。此矩阵替代了传统 replay buffer 的「存储-复演」机制，以二阶矩形式压缩历史知识。

3. **Recursive Update（递归更新）**：接收上一任务的 $R_{t-1}$ 与新任务数据 $X_t$，通过 Woodbury 恒等式或递归公式更新 $R_t$，无需重访旧数据。核心操作是矩阵的秩一更新，计算复杂度 $O(d^2)$。

4. **Analytic Prediction（解析预测）**：测试时，利用当前 $R_t$ 与测试特征直接计算分类权重 $W_t = R_t X_t^\text{top} Y_t$，输出预测。推理过程无梯度、无迭代，单次前向完成。

整体流程可概括为：
```
图像 → [Backbone] → 特征 X → [R_t = (X^T X + γI)^{-1}] → 递归更新 R_t
                                                      ↓
测试图像 → [Backbone] → 特征 x_test → [W_t = R_t X^T Y] → 预测输出
```

关键设计：所有知识承载于 $R$ 矩阵的 $d \times d$ 结构中，与样本数 $N$ 无关，故无需存储任何 exemplar。

## 核心模块与公式推导

### 模块 1: 正则化自相关记忆矩阵（对应框架图模块2-3）

**直觉**：最小二乘解 $W = (X^\text{top} X)^{-1} X^\text{top} Y$ 要求 $X^\text{top} X$ 可逆，但持续学习中特征维度高、样本少，矩阵必奇异。加入正则化是数值稳定的必要代价。

**Baseline 公式** (标准解析学习 / γ=0 的退化形式):
$$R_1 = (X_{\text{total},1}^\text{top} X_{\text{total},1})^{-1}$$
符号: $X_{\text{total},1} \in \mathbb{R}^{N_1 \times d}$ = 第1任务全部样本特征矩阵; $R_1 \in \mathbb{R}^{d \times d}$ = 自相关逆矩阵。

**变化点**：当仅当前批次数据 $X_1^{(B)}$ 可用且 $N_1 < d$ 时，$X_1^{(B)\text{top}} X_1^{(B)}$ 秩亏，逆矩阵不存在。Baseline 假设「全部数据可一次性获取」不成立。

**本文公式（推导）**:
$$\text{Step 1}: R_1 = (X_1^{(B)\text{top}} X_1^{(B)})^{-1} \quad \text{直接代入批次数据，但奇异}$$
$$\text{Step 2}: R_1 = (X_{\text{total},1}^\text{top} X_{\text{total},1} + \gamma I)^{-1} \quad \text{加入 Tikhonov 正则化项 } \gamma I \text{ 保证正定性}$$
$$\text{最终}: R_t = \left(\sum_{i=1}^{t} X_i^\text{top} X_i + \gamma I\right)^{-1} = (R_{t-1}^{-1} + X_t^\text{top} X_t)^{-1}$$

递归更新通过 Woodbury 恒等式实现：
$$R_t = R_{t-1} - R_{t-1} X_t^\text{top} (I + X_t R_{t-1} X_t^\text{top})^{-1} X_t R_{t-1}$$

**对应消融**：Table 3-4 显示，$\gamma$ 过大（如10000）导致欠拟合，过小则数值崩溃。去掉正则化（$\gamma=0$）时 AAUC 从 57.99% 暴跌至 8.87%（CIFAR-100），$\Delta = -49.12$pp。

### 模块 2: ECLG 模块的递归权重更新（对应框架图模块3-4）

**直觉**：持续学习需将「旧知识 $R_{t-1}$」与「新知识 $X_t$」融合，而非独立优化每个任务。

**Baseline 公式** (独立任务学习 / 无持续融合):
$$W_t^{\text{ind}} = R_t^{\text{ind}} X_t^\text{top} Y_t, \quad R_t^{\text{ind}} = (X_t^\text{top} X_t + \gamma I)^{-1}$$
符号: $Y_t \in \mathbb{R}^{N_t \times C_t}$ = 任务 $t$ 的标签 one-hot 矩阵; $W_t$ = 分类器权重。

**变化点**：独立学习每个任务导致 $W_t$ 仅编码当前任务信息，灾难性遗忘。Baseline 的 ACL 未处理任务序列的递归累积。

**本文公式（推导）**:
$$\text{Step 1}: W_t = R_t \left(\sum_{i=1}^{t} X_i^\text{top} Y_i\right) = R_t (R_{t-1}^{-1} W_{t-1} + X_t^\text{top} Y_t) \quad \text{利用历史累积量 } S_{t-1} = R_{t-1}^{-1} W_{t-1}$$
$$\text{Step 2}: S_t = S_{t-1} + X_t^\text{top} Y_t \quad \text{递归累积交叉矩，无需存储 } X_{1:t-1}$$
$$\text{最终}: W_t = R_t S_t = R_t (S_{t-1} + X_t^\text{top} Y_t)$$

关键：$S_t \in \mathbb{R}^{d \times C_{\text{total}}}$ 随任务增长列数（新类别扩展），但行数固定为 $d$；$R_t \in \mathbb{R}^{d \times d}$ 维度恒定。两者均与样本量解耦，实现真正免示例。

**对应消融**：Table 2 显示移除 ECLG 模块（退化为准独立任务学习）导致性能显著下降，验证了递归融合的必要性（具体数值待补充）。

### 模块 3: 类别比例自适应分析（辅助理解模块）

**直觉**：GACL 在 CIFAR-100 初始任务准确率偏低（Figure 2(a)），需解释该现象非方法缺陷，而是任务设定所致。

**本文公式**:
$$r_c = d_i / N$$
符号: $d_i$ = 第 $i$ 个样本点已观察到的类别数; $N$ = 总类别数（CIFAR-100 为100）。

$r_c$ 高意味着早期任务已暴露大量类别，接近小样本学习（few-shot）场景。Figure 6 显示实时准确率与 $r_c$ 负相关，解释了 CIFAR-100 初始阶段性能较低的根因——非 GACL 遗忘，而是任务难度本身高。

## 实验与分析


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/1fd62c83-e375-46dd-ac64-85dd1ad20ced/figures/Table_1.png)
*Table 1: Table 1: Comparison of AAUC, AAvg, and ALast among the GACL and other methods under theSi-Blurry setting. Data in bold represent the best EFCIL results, and data underlined are the bestamong all setti*



本文在三个标准持续学习基准上评估：CIFAR-100（100类）、ImageNet-R（200类艺术/渲染变体）、Tiny-ImageNet（200类64×64图像）。采用 Generalized CIL（GCIL）设定下的 Si-Blurry 任务划分，指标包括 AAUC（平均累积准确率曲线下面积）、AAvg（平均准确率）、ALast（最后任务准确率）。


![Table 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/1fd62c83-e375-46dd-ac64-85dd1ad20ced/figures/Table_2.png)
*Table 2: Table 2: Ablation study on the ECLG module of our GACL.*

 的核心结论：GACL 在 CIFAR-100 上 AAUC 达 57.99%，超越免示例方法中的次优者 SLDA（53.0%）+4.99pp，亦超越另一免示例方法 MVP（56.28%）+1.71pp。然而，提示学习方法 L2P（65.1%）与 DualPrompt（65.6%）仍显著高于 GACL，提示基于预训练 Transformer 的提示调优在固定骨干场景下具优势。ImageNet-R 上 GACL 的 AAUC 为 41.68%，优于 LwF（38.65%）+3.03pp，但 L2P/DualPrompt 数据未在提供片段中明确。Tiny-ImageNet 上 GACL 达 63.14%，与 SLDA（62.74%）接近，仅 +0.40pp 边际提升。


![Table 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/1fd62c83-e375-46dd-ac64-85dd1ad20ced/figures/Table_3.png)
*Table 3: Table 3: The performance at different rD withrB = 10% on CIFAR-100.*



消融实验（Table 2）聚焦 ECLG 模块：移除该模块后性能退化，确认递归解析更新的核心贡献。Table 3-4 探究超参数 $r_D$（数据保留比例）与 $r_B$（模糊比例）的影响，显示 GACL 对设定鲁棒。

正则化项 $\gamma$ 的敏感性分析（Figure 5）揭示：$\gamma=0$ 时 CIFAR-100 AAUC 暴跌至 8.87%，ImageNet-R 至 2.03%，Tiny-ImageNet 至 4.38%，证明正则化是数值稳定的必要条件；$\gamma=10000$ 时则欠拟合。最优 $\gamma$ 需权衡稳定性与表达能力。

训练效率方面，GACL 在 CIFAR-100 仅需 611秒，ImageNet-R 321秒，Tiny-ImageNet 1246秒；对比 RM 的 >2天，效率提升超两个数量级。这一优势源于解析闭式解替代了梯度迭代。

公平性审视：本文对比的 baselines 中，L2P/DualPrompt 在 CIFAR-100 上 AAUC 更高（65.1/65.6 vs 57.99），但二者依赖预训练 ViT 与复杂提示机制，非纯解析方法。缺失对比包括 DER++、FOSTER、MEMO、SimpleCIL 等近期 EFCIL 方法，可能低估竞争强度。作者未明确披露失败模式，但 Figure 2(a) 显示初始任务准确率偏低，已通过 $r_c$ 分析解释。

## 方法谱系与知识库定位

GACL 隶属于 **Analytic Continual Learning（解析持续学习）** 方法族，直接父方法为 ACL（Analytic Continual Learning）。ACL 首次将闭式最小二乘引入持续学习，但未解决矩阵奇异性与免示例递归更新的问题。

**关键改动槽位**：
- **Architecture**：以自相关记忆矩阵 $R$ 替代梯度权重或 replay buffer
- **Training recipe**：SGD/Adam → 正则化递归最小二乘
- **Inference strategy**：标准前向 → 基于 $R_t$ 的解析预测
- **Data curation**：回放缓冲区 → 纯递归更新，零示例存储

**直接 Baselines 差异**：
- vs **SLDA**：同为解析/统计方法，但 SLDA 基于线性判别分析，无正则化递归矩阵机制
- vs **MVP**：MVP 为免示例提示方法，依赖视觉提示生成；GACL 为纯解析，无需预训练提示
- vs **LwF / EWC++**：同为免示例，但梯度正则化/蒸馏弱于闭式解
- vs **L2P / DualPrompt**：提示方法性能更高，但依赖 Transformer 预训练与任务特定提示，非解析路线

**后续方向**：
1. 与预训练视觉 Transformer 结合，探索「解析更新 + 提示调优」的混合机制
2. 扩展至其他模态（视频、音频）的免示例持续学习
3. 自适应正则化：根据 $r_c$ 或任务难度动态调整 $\gamma$

**知识库标签**：
- Modality: 图像分类
- Paradigm: 解析学习 / 闭式解 / 免示例学习
- Scenario: 类增量学习 / Generalized CIL / Si-Blurry
- Mechanism: 自相关记忆矩阵 / 递归最小二乘 / Tikhonov 正则化
- Constraint: 免示例（exemplar-free）/ 低计算预算 / 隐私保护

