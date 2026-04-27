---
title: It's a (Blind) Match! Towards Vision-Language Correspondence without Parallel Data
type: paper
paper_level: C
venue: CVPR
year: 2025
paper_link: null
aliases:
- 无配对数据的盲视觉-语言匹配
- IBMTVL
acceptance: poster
cited_by: 10
baselines:
- 无监督视觉特征学习的DINOv2_DINOv2
---

# It's a (Blind) Match! Towards Vision-Language Correspondence without Parallel Data

**Topics**: [[T__Cross-Modal_Matching]], [[T__Self-Supervised_Learning]], [[T__Classification]] | **Datasets**: [[D__CIFAR-10]], [[D__CINIC-10]]

| 中文题名 | 无配对数据的盲视觉-语言匹配 |
| 英文题名 | It's a (Blind) Match! Towards Vision-Language Correspondence without Parallel Data |
| 会议/期刊 | CVPR 2025 (poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2503.24129) · [Code](https://github.com/dschnaus/blind-match) · [Project](https://dschnaus.github.io/blind-match) |
| 主要任务 | 无监督跨模态匹配（vision-language alignment without paired data） |
| 主要 baseline | Gromov-Wasserstein (GW) distance, CKA, kNN-based matching |

> [!abstract]
> 因为「视觉-语言对齐通常需要大量配对数据」，作者在「Gromov-Wasserstein distance」基础上改了「提出可分解损失函数与因子化Hahn-Grant求解器」，在「CIFAR-10 / CINIC-10」上取得「72% / 100% 匹配准确率，相比随机基线提升62/90个百分点」。

- **CIFAR-10 匹配准确率**: 72%（随机基线 10%，提升 +62 pp）
- **CINIC-10 匹配准确率**: 100%（随机基线 10%，提升 +90 pp）
- **DINOv2 最优性**: 在 CIFAR-10 上比次优预训练策略高 5.2%，CINIC-10 上高 7.7%

## 背景与动机

视觉与语言是同一世界的两种抽象表征——图像捕捉外观，文本描述语义——但现有方法强制要求二者成对出现才能学习对齐。例如，CLIP 需要数亿级别的 (image, text) 对进行对比学习；即便后续的自监督方法，也往往假设模态间存在粗略的对应关系。然而在实际场景中，我们可能只有独立的图像集合与独立的文本语料库，例如不同医院分别拥有 X 光图像和病历文本，却从未标注过对应关系。

现有方法如何处理这一困境？**Gromov-Wasserstein (GW) distance** 提供了一条理论路径：它通过比较两个度量空间内部的成对距离结构，无需点级别的对应即可衡量空间相似性。但标准 GW 的求解是 NP-hard 的二次分配问题（QAP），四维成本张量的优化在计算上不可行。**CKA (Centered Kernel Alignment)** 作为替代方案，通过核对齐衡量表征相似性，计算更高效，但初步分析发现其在视觉-语言匹配任务上表现 inferior。**kNN-based matching** 则是一种简化启发式，将匹配问题退化为近邻检索，但缺乏全局最优保证。

这些方法的共同瓶颈在于：**GW 的理论框架正确但无法扩展；CKA 可扩展但目标函数不适合跨模态匹配；kNN 过于简化而丢失全局结构**。具体而言，标准 GW 的损失函数 $l(A,B) = (A-B)^2$ 导致四维成本张量无法分解，使得精确求解仅限于 $N \leq 10$ 的玩具规模；而近似方法又牺牲了匹配精度。本文的核心动机正是：**能否保持 GW 的理论优势，同时通过损失函数的重新设计，将四维 QAP 分解为可高效求解的序列问题？**


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/4a7f90e4-a93f-4ba7-a05d-61b8c2cdc0fb/figures/Figure_1.png)
*Figure 1: Figure 1. Blind matching of vision and language: Text and im-ages are both abstractions of the same underlying world. Visionand language encoders fv and fl learn similar pairwise relationsbetween conc*



本文提出 Blind Match，通过可分解损失框架与因子化 Hahn-Grant 求解器，首次实现了无需配对数据的高效视觉-语言对应学习。

## 核心创新

核心洞察：**将 GW 损失函数约束为可分解形式 $l(A,B) = f_1(A) + f_2(B) - h_1(A)h_2(B)$，因为该形式允许四维成本张量分解为两个二维矩阵的外积，从而使 Hahn-Grant 算法的因子化扩展成为可能，将 NP-hard QAP 转化为迭代求解的二维分配问题序列。**

| 维度 | Baseline (标准 GW) | 本文 (Blind Match) |
|:---|:---|:---|
| **损失函数** | $l(A,B) = (A-B)^2$，不可分解 | $l(A,B) = f_1(A) + f_2(B) - h_1(A)h_2(B)$，可分解 |
| **优化目标** | 四维成本张量 $\sum_{ijkl} C_{ijkl} P_{ij} P_{kl}$，NP-hard QAP | 因子化为 $\sum_{ijkl} C^{(1)}_{ik} C^{(2)}_{jl} P_{ij} P_{kl}$，可迭代求解 |
| **求解算法** | 通用 QAP 求解器，指数复杂度 | Hahn-Grant 因子化求解器，leader-follower 分解，每步二维分配 |
| **数据要求** | 无显式要求（但计算限制规模） | 仅需同类样本平均表征，无需 image-text 配对 |
| **可扩展性** | $N \leq 10$ 实际不可行 | 支持更大规模，且提供 kNN/CKA 两种高效特例 |

## 整体框架


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/4a7f90e4-a93f-4ba7-a05d-61b8c2cdc0fb/figures/Figure_3.png)
*Figure 3: Figure 3. The Hahn-Grant solver (left) and the factorized Hahn-Grant solver (ours, right): The Hahn-Grant solver [13] iterativelyimproves the dual bound of the QAP by solving linear assignment problem*



Blind Match 的完整流程包含五个阶段，全部使用**冻结的预训练编码器**，无需任何微调或联合训练：

1. **Vision feature extraction**: 输入为各类别的图像集合 $\mathcal{I}_i$，经冻结的视觉编码器 $f_v$（如 DINOv2、CLIP image encoder）提取特征后，按类别取平均，输出类级别视觉表征 $\text{visionfeature}_i = \frac{1}{|\mathcal{I}_i|}\sum_{I \in \mathcal{I}_i} f_v(I)$。

2. **Language feature extraction**: 输入为每类的多个文本提示 $\mathcal{T}_i$（如 "a photo of a {class}" 的多种变体），经冻结的语言编码器 $f_l$（如 all-mpnet-base-v2、CLIP text encoder）提取特征后取平均，输出类级别语言表征 $\text{languagefeature}_i = \frac{1}{|\mathcal{T}_i|}\sum_{T \in \mathcal{T}_i} f_l(T)$。

3. **Pairwise distance computation**: 分别在视觉模态和语言模态内部计算成对距离矩阵，输出 $\text{visionpairwise}_{ij} = \|\text{visionfeature}_i - \text{visionfeature}_j\|_2$ 和 $\text{languagepairwise}_{ij} = \|\text{languagefeature}_i - \text{languagefeature}_j\|_2$。关键之处在于：**两个矩阵的类别索引无需对齐**，这正是"盲匹配"的含义。

4. **GW distance with factorized Hahn-Grant solver**: 输入为两个模态的成对距离矩阵，核心模块通过可分解损失与因子化求解器，输出最优置换矩阵 $\mathbf{P}^*$，即视觉类别与语言类别之间的对应关系。

5. **Matching accuracy evaluation**: 将预测的对应关系与真实标签比较，计算分类准确率。

```
Images ──→ Vision encoder ──→ Average per class ──→ Pairwise distances ──┐
                                                                           ├──→ Factorized GW solver ──→ Correspondence P*
Text prompts ──→ Language encoder ──→ Average per class ──→ Pairwise distances ──┘
```

## 核心模块与公式推导

### 模块 1: 可分解损失框架（对应框架图"GW distance"模块的理论基础）

**直觉**: 标准 GW 的四维成本张量无法直接优化，但如果损失函数可以拆分为单变量函数的组合，则张量可分解为外积形式，从而降维求解。

**Baseline 公式** (标准 GW): 
$$\mathcal{D}_\text{GW} = \min_{\mathbf{P} \in \mathcal{P}_N} \sum_{i,j,k,l=1}^{N} (\text{visionpairwise}_{ik} - \text{languagepairwise}_{jl})^2 \mathbf{P}_{ij} \mathbf{P}_{kl}$$
符号: $\text{visionpairwise}_{ik}$ = 视觉模态中类别 $i,k$ 的成对距离；$\text{languagepairwise}_{jl}$ = 语言模态中类别 $j,l$ 的成对距离；$\mathbf{P}_{ij} \in \{0,1\}$ = 置换矩阵指示视觉类别 $i$ 是否匹配语言类别 $j$。

**变化点**: $(A-B)^2 = A^2 + B^2 - 2AB$ 虽然数学上可分解，但标准 GW 实现并未利用此结构进行算法优化；且对于更一般的度量（如 CKA、kNN），需要统一的分解框架。

**本文公式（推导）**:
$$\text{Step 1}: \quad l(A, B) = f_1(A) + f_2(B) - h_1(A)h_2(B) \quad \text{（定义可分解损失的一般形式）}$$
$$\text{Step 2}: \quad \mathcal{D}_l(\mathbf{X}, \mathbf{Y}) = \min_{\mathbf{P}} \sum_{i,j,k,l} \left[f_1(\text{visionpairwise}_{ik}) + f_2(\text{languagepairwise}_{jl}) - h_1(\text{visionpairwise}_{ik})h_2(\text{languagepairwise}_{jl})\right] \mathbf{P}_{ij}\mathbf{P}_{kl} \quad \text{（代入 GW 目标）}$$
$$\text{Step 3}: \quad = \min_{\mathbf{P}} \left[\text{const} + \sum_{i,j}\mathbf{P}_{ij}\underbrace{\sum_{k,l}\mathbf{P}_{kl}(\cdots)}_{\text{可迭代处理}}\right] \quad \text{（分离常数项与变量项）}$$
$$\text{最终}: \quad \mathbf{P}^* \in \argmin_{\mathbf{P} \in \mathcal{P}_N} \sum_{i,j,k,l=1}^{N} \mathbf{C}^{(1)}_{ik}\mathbf{C}^{(2)}_{jl} \mathbf{P}_{ij}\mathbf{P}_{kl}$$
其中 $\mathbf{C}^{(1)}_{ik} = h_1(\text{visionpairwise}_{ik})$，$\mathbf{C}^{(2)}_{jl} = h_2(\text{languagepairwise}_{jl})$，四维张量已分解为两个二维矩阵的外积。

**对应消融**: 将 GW 替换为 CKA 后，初步分析显示匹配质量下降（Appendix B），验证了可分解损失框架对视觉-语言任务的适配性。

---

### 模块 2: 因子化 Hahn-Grant 求解器（对应框架图核心优化模块）

**直觉**: 即使成本张量可分解，直接求解仍是 QAP；Hahn-Grant 算法的核心思想是将四维问题分解为"领导者"（固定主变量）和"追随者"（求解子问题）的迭代序列。

**Baseline 公式** (标准 Hahn-Grant [13]):
$$\sum_{i,j,k,l} \mathbf{C}_{ijkl} \mathbf{P}_{ij} \mathbf{P}_{kl} = \text{const} + \sum_{i,j} \text{leader}_{ij} \mathbf{P}_{ij} + \text{bilinear remainder}$$
标准算法需要存储和操作完整的四维张量 $\mathbf{C}_{ijkl}$。

**变化点**: 本文发现当损失可分解时，双线性项可进一步利用对称性化简，无需显式构造四维张量。

**本文公式（推导）**:
$$\text{Step 1}: \quad \mathbf{C}_{ijkl} + \mathbf{C}_{klij} = 2\overline{\mathbf{C}}^{(1)}_{ik} \overline{\mathbf{C}}^{(2)}_{jl} - \overline{\mathbf{U}}_{ijk} - \overline{\mathbf{V}}_{ijl} - \overline{\mathbf{U}}_{kli} - \overline{\mathbf{V}}_{klj} \quad \text{（利用对称性）}$$
$$\text{Step 2}: \quad \Rightarrow \mathbf{C}_{ijkl} = \mathbf{C}^{(1)}_{ik} \mathbf{C}^{(2)}_{jl} - \overline{\mathbf{U}}_{ijk} - \overline{\mathbf{V}}_{ijl} \quad \text{（变量替换，消去冗余）}$$
$$\text{Step 3}: \quad \sum_{i,j,k,l} \mathbf{C}_{ijkl} \mathbf{P}_{ij}\mathbf{P}_{kl} = l + \sum_{i,j} \text{leader}_{ij} \mathbf{P}_{ij} + \sum_{i,j,k,l}(\mathbf{C}_{ijkl} - \mathbf{u}^{(ij)}_k - \mathbf{v}^{(ij)}_l) \mathbf{P}_{ij}\mathbf{P}_{kl}$$
其中 $\mathbf{u}^{(ij)}_k, \mathbf{v}^{(ij)}_l$ 为对偶变量，使得每次迭代只需固定 $\mathbf{P}_{kl}$ 求解关于 $\mathbf{P}_{ij}$ 的二维线性分配问题（LAP）。

**最终迭代形式**: 每步求解
$$\mathbf{P}^{(t+1)} = \argmin_{\mathbf{P} \in \mathcal{P}_N} \sum_{i,j} \left[\text{leader}_{ij} + \sum_{k,l}(\mathbf{C}_{ijkl} - \mathbf{u}^{(ij)}_k - \mathbf{v}^{(ij)}_l)\mathbf{P}^{(t)}_{kl}\right] \mathbf{P}_{ij}$$
即：**二维 LAP 的序列求解**，每步复杂度 $O(N^3)$ 而非 $O(N^4)$。

**对应消融**: Figure 6 显示在 CIFAR-100（$N=100$）上，因子化求解器相比标准方法显著降低 Gromov-Wasserstein 距离，且计算可行。

---

### 模块 3: kNN-based GW 距离（可扩展特例）

**直觉**: 当类别数量极大时，即使因子化求解仍可能缓慢；kNN 核函数可将 GW 距离简化为直观的近邻匹配率。

**Baseline 公式**: 标准 GW 的内积损失形式 $l_\text{inner}(A,B) = -A \cdot B$。

**变化点**: 将邻接矩阵替换为 kNN 核，即 $\text{visionpairwise}^{\text{kNN}}_{ij} = \mathbb{1}[j \in \text{top-}k_\mathcal{V}(i)]$。

**本文公式（推导）**:
$$\text{Step 1}: \quad m_\text{kNN}(\text{visionfeature}_i, \text{languagefeature}_i) = \frac{1}{k}|\text{top-}k^\mathcal{V}(i) \cap \text{top-}k^\mathcal{L}(i)| \quad \text{（定义 kNN 匹配率）}$$
$$\text{Step 2}: \quad \mathcal{D}_\text{kNN}(\text{visionpairwise}^\text{kNN}, \text{languagepairwise}^\text{kNN}) = -\frac{1}{N}\sum_{i=1}^{N} m_\text{kNN}(\text{visionfeature}_i, \text{languagefeature}_i)$$
**最终**: kNN-based GW 距离 = 负的平均 kNN 匹配率，计算复杂度降至 $O(N^2 \log k)$，且无需迭代优化。

**对应消融**: Table 2 显示结合 K-Means 聚类后，kNN-based 方法在无监督分类任务上取得有效结果（具体数值待补充）。

## 实验与分析


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/4a7f90e4-a93f-4ba7-a05d-61b8c2cdc0fb/figures/Table_1.png)
*Table 1: Figure 5. Some fine-grained problems can be matched with high accuracy: For each problem size, we select the optimal ten subsetsof classes using the optimization from Sec. 4.2 on ImageNet-100 [44] (le*



本文在 CIFAR-10 和 CINIC-10 两个标准数据集上评估匹配准确率。Figure 4 展示了核心结果：使用 DINOv2 作为视觉编码器、all-mpnet-base-v2 作为语言编码器时，**CIFAR-10 上达到 72% 匹配准确率**，相比随机基线（10%）提升 62 个百分点；**CINIC-10 上达到 100% 匹配准确率**，提升 90 个百分点。这一结果表明，在类别语义区分度较高的数据集上，纯基于内部距离结构的盲匹配可以完全恢复视觉-语言对应关系。

跨模型比较方面，DINOv2 表现最优：在 CIFAR-10 上比次优预训练策略高 5.2%，在 CINIC-10 上高 7.7%。这一发现暗示自监督视觉表征（DINOv2）可能比对比学习表征（CLIP）更适合与语言模型进行结构级对齐——因为 DINOv2 学到的视觉相似性结构更贴近人类语义分类的层次结构。


![Figure 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/4a7f90e4-a93f-4ba7-a05d-61b8c2cdc0fb/figures/Figure_4.png)
*Figure 4: Figure 4. Most vision and language models can be matchednon-trivially: We visualize the accuracy for multiple vision mod-els with the all-mpnet-base-v2 [43] language model on CIFAR-10 [23] (left) and*




![Table 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/4a7f90e4-a93f-4ba7-a05d-61b8c2cdc0fb/figures/Table_2.png)
*Table 2: Table 2. Unsupervised classification: We combine our unsuper-vised matching with K-Means clustering of the image embeddingson CIFAR-10 [23]. It shows that using the cluster means as visionrepresentati*



消融实验揭示了两个关键发现。第一，**全局最优解的必要性**：只有找到全局最优的对应关系，才能取得非平凡的匹配结果；局部最优解导致匹配质量崩溃。这验证了 Hahn-Grant 求解器收敛性的理论重要性。第二，**GW 优于 CKA**：将 GW 距离替换为 CKA 相似性后，初步分析显示视觉-语言匹配质量下降（Appendix B），说明 CKA 的归一化核对齐目标虽然适合同模态表征比较，但不适合跨模态的结构匹配任务。

公平性检查方面，本文存在若干局限：CINIC-10 的 100% 准确率可能暗示数据集过于简单或存在信息泄漏；实验规模限于 $N=10$ 的类别数，未在 realistic scale（如 ImageNet 的 1000 类）上验证；缺少与使用实际配对数据的监督基线的直接对比；亦未与近期的熵正则化 GW（entropic GW）等最优传输变体比较（作者承诺在 Sec. 5.3 讨论但原文截断）。此外，Table 2 的无监督分类实验显示结合 K-Means 后的下游性能，但具体数字需补充。


![Figure 6](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/4a7f90e4-a93f-4ba7-a05d-61b8c2cdc0fb/figures/Figure_6.png)
*Figure 6: Figure 6.Solver comparison on larger-scale problems: Us-ing CLIP and all-mpnet-base-v2 on CIFAR-100 [23], we plotthe Gromov-Wasserstein distance (solid line) and its lower bound(dashed line) where ava*



## 方法谱系与知识库定位

**方法族**: Optimal Transport → Gromov-Wasserstein distance → **Blind Match**

**父方法**: Gromov-Wasserstein distance（Memoli, 2011; Peyré et al., 2016），其通过比较度量空间内部结构实现跨空间对齐，但受限于 NP-hard QAP 求解。

**变更槽位**:
- **objective**: 标准 GW 的 $(A-B)^2$ 损失 → 可分解损失 $l(A,B) = f_1(A) + f_2(B) - h_1(A)h_2(B)$
- **inference_strategy**: 通用 QAP 求解器 → 因子化 Hahn-Grant 求解器（leader-follower 分解）
- **data_curation**: 需要配对数据 → 仅需同类样本平均，无需 image-text 配对
- **architecture**: 联合训练或微调 → 冻结预训练单模态编码器，后 hoc GW 匹配

**直接基线与差异**:
- **Gromov-Wasserstein (标准)**: 理论基础相同，但本文通过损失分解与因子化求解实现计算可行性
- **CKA**: 同为无监督表征比较方法，但 CKA 目标函数不适合跨模态结构匹配
- **kNN matching**: 本文将其推导为 GW 框架的特例，赋予理论解释

**后续方向**:
1. **规模扩展**: 将因子化求解器扩展至 $N \geq 1000$（如 ImageNet 规模），验证算法在大规模下的收敛性与精度
2. **端到端学习**: 将可分解损失嵌入可微框架，联合优化编码器与匹配目标
3. **多模态扩展**: 从视觉-语言推广至音频、点云等其他模态的盲匹配

**标签**: `modality: vision+language` | `paradigm: unsupervised alignment` | `scenario: cross-modal matching without paired data` | `mechanism: optimal transport + factorized optimization` | `constraint: frozen encoders, no fine-tuning`

## 引用网络

### 直接 baseline（本文基于）

- [[P__无监督视觉特征学习的DINOv2_DINOv2]] _(方法来源)_: Modern self-supervised visual encoder; likely used as feature extractor or backb

