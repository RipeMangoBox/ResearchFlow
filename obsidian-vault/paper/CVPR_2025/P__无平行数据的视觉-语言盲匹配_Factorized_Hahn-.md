---
title: It's a (Blind) Match! Towards Vision-Language Correspondence without Parallel Data
type: paper
paper_level: C
venue: CVPR
year: 2025
paper_link: null
aliases:
- 无平行数据的视觉-语言盲匹配
- Factorized Hahn-
- Factorized Hahn-Grant Solver for Gromov-Wasserstein Distance
acceptance: poster
cited_by: 10
method: Factorized Hahn-Grant Solver for Gromov-Wasserstein Distance
baselines:
- 无监督视觉特征学习的DINOv2_DINOv2
---

# It's a (Blind) Match! Towards Vision-Language Correspondence without Parallel Data

**Topics**: [[T__Cross-Modal_Matching]], [[T__Self-Supervised_Learning]], [[T__Classification]] | **Method**: [[M__Factorized_Hahn-Grant_Solver_for_Gromov-Wasserstein_Distance]] | **Datasets**: [[D__CIFAR-10]], [[D__CINIC-10]], [[D__ImageNet-1K]] (其他: Solver)

| 中文题名 | 无平行数据的视觉-语言盲匹配 |
| 英文题名 | It's a (Blind) Match! Towards Vision-Language Correspondence without Parallel Data |
| 会议/期刊 | CVPR 2025 (poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2503.24129) · [Code](https://github.com/dschnaus/blind-match) · [Project](https://dschnaus.github.io/blind-match) |
| 主要任务 | 无平行数据的视觉-语言表示对齐（Vision-Language Correspondence without Parallel Data） |
| 主要 baseline | Gromov-Wasserstein distance、CKA、ASIF、GeRA、Graph Optimal Transport for Cross-Domain Alignment、Latent Space Translation via Semantic Alignment |

> [!abstract] 因为「视觉-语言对齐通常需要大量成对(image, text)数据」，作者在「Gromov-Wasserstein distance」基础上改了「因式分解Hahn-Grant求解器与代价张量分解」，在「CIFAR-10/CINIC-10」上取得「DINOv2特征72%/100%匹配准确率」。

- **CINIC-10 (N=10)**：DINOv2特征达到 **100%** 匹配准确率，相比随机基线(10%)提升 **+90%** 绝对值
- **CIFAR-10 (N=10)**：DINOv2特征达到 **72%** 匹配准确率，相比随机基线(10%)提升 **+62%** 绝对值
- **求解器优势**：Factorized Hahn-Grant求解器能找到全局最优解，而替代优化算法仅能找到局部最优解

## 背景与动机

视觉-语言模型（如CLIP）的成功依赖于数亿级别的成对(image, text)数据进行对比学习。然而，在许多实际场景中，获取这样的平行数据成本高昂甚至不可能——例如，某些小众语言缺乏图文配对资源，或医疗、工业领域的数据因隐私无法共享。本文核心问题：能否仅利用**非配对的**视觉和语言数据，实现两类表示空间的有效对齐？

现有方法从不同角度尝试解决这一"盲匹配"问题：

- **ASIF**（Coupled Data Turns Unimodal Models to Multimodal Without Training）：利用耦合数据将单模态模型转为多模态，但仍需要某种形式的跨模态关联信号；
- **GeRA**（Label-Efficient Geometrically Regularized Alignment）：引入几何正则化进行跨模态对齐，依赖标签效率但非完全无配对；
- **Graph Optimal Transport for Cross-Domain Alignment**：使用图最优传输进行跨域对齐，计算复杂度高且难以扩展到大规模问题；
- **Gromov-Wasserstein (GW) distance**：经典的度量空间比较方法，将问题建模为二次分配问题(QAP)，但其标准形式需要求解NP-hard的全排列矩阵优化，无法处理较大规模。

这些方法的共同瓶颈在于：**计算复杂度与可扩展性**。GW距离虽能捕捉模态内部的成对结构相似性，但其4维代价张量的直接优化在N>10时即变得不可行；而CKA等替代方法虽计算高效，但实验表明其对视觉-语言匹配的判别力不足。此外，现有工作多依赖局部近似或熵正则化，无法保证全局最优性——而作者发现，**只有全局最优解才能产生非平凡的匹配结果**。

本文提出一种因式分解的Hahn-Grant求解器，通过将4维代价张量分解为低秩2维矩阵的乘积，首次实现了较大规模GW问题的高效全局优化。


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/9daab291-1940-4c97-8691-430ea81746a5/figures/fig_001.jpeg)
*Figure: Blind matching of vision and language: Text and im-*



## 核心创新

核心洞察：**Gromov-Wasserstein距离的4维代价张量具有可分解结构**，因为成对损失函数l(A,B)可表示为各模态独立函数的组合，从而使张量低秩分解与高效全局优化成为可能。

| 维度 | Baseline (标准GW) | 本文 |
|:---|:---|:---|
| **代价张量表示** | 完整4维张量 C_{ijkl}，直接存储与计算 | 因式分解为 C^{(1)}_{ik}C^{(2)}_{jl} - U_{ijk} - V_{ijl}，低秩近似 |
| **优化策略** | 直接QAP求解器或熵正则化局部近似 | Hahn-Grant分解：leader项 + 因式化二次项，高效全局优化 |
| **核函数设计** | 标准RBF或线性核，稠密矩阵 | kNN稀疏核：仅保留top-k邻居，O(Nk)复杂度 |
| **大规模处理** | 无法处理N>10的问题 | 子集选择模块，从L个类别中优选N个最可靠匹配子集 |

## 整体框架


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/9daab291-1940-4c97-8691-430ea81746a5/figures/fig_002.jpeg)
*Figure: Shufﬂing degrades vision-language alignment: The*



整体流程遵循"提取→聚合→核化→求解"四阶段范式：

1. **特征提取（Feature extraction）**：输入原始图像和文本提示，分别通过视觉编码器（如DINOv2）和语言编码器（如all-mpnet-base-v2）提取特征 f_v(I) 和 f_l(T)；

2. **类级别聚合（Class-level aggregation）**：对每个类别的多张图像特征取平均得到 v_i，对多个文本提示特征取平均得到 l_i，构建非配对的类级别表示；

3. **成对核计算（Pairwise kernel computation）**：计算视觉成对矩阵 V_{ij} = K_v(v_i, v_j) 和语言成对矩阵 L_{ij} = K_l(l_i, l_j)，支持三种核变体：GW核（欧氏距离+平方损失）、CKA核（中心化HSIC）、kNN稀疏核（仅保留k近邻）；

4. **因式分解Hahn-Grant求解器（Factorized Hahn-Grant solver）**：将4维代价张量 C_{ijkl} = l(V_{ik}, L_{jl}) 分解为低秩形式，通过leader项与修正项高效求解最优置换矩阵 P*；

5. **子集选择（Subset selection，可选）**：对于大规模问题（L>>N），通过组合优化从L个类别中选择N个最可靠匹配的子集 S*。

```
图像 I ──→ 视觉编码器 ──→ v_i ──┐
                                ├──→ 核矩阵 V_{ij}, L_{ij} ──→ 代价张量 C_{ijkl}
文本 T ──→ 语言编码器 ──→ l_i ──┘                                │
                                                                  ↓
                                            [子集选择 S*] ──→ 因式分解Hahn-Grant求解器
                                                                  │
                                                                  ↓
                                                            最优匹配 P*
```

## 核心模块与公式推导

### 模块 1: kNN稀疏核与距离等价（对应框架图"核计算"模块）

**直觉**：视觉-语言匹配不需要完整的稠密相似性矩阵，仅保留每个样本的k个最近邻即可保持结构信息，同时大幅降低计算量。

**Baseline 公式** (标准GW核): 
$$K_v^{\text{GW}}(v_i, v_j) = \|v_i - v_j\|_2$$
符号: $v_i$ = 第i类的平均视觉特征，$\|\cdot\|_2$ = 欧氏距离。

**变化点**：标准GW使用稠密的欧氏距离矩阵，O(N²)存储与计算；本文发现对于匹配任务，稀疏的k近邻结构已足够判别。

**本文公式（推导）**:
$$\text{Step 1}: K_v^{\text{kNN}}(v_i, v_j) = \frac{1}{\sqrt{Nk}} \text{mathbbm}{1}[j \in \text{top}_k^{v}(i)] \quad \text{仅保留k个最近邻，构造稀疏指示矩阵}$$
$$\text{Step 2}: \mathcal{D}_{\text{kNN}}(V^{\text{kNN}}, L^{\text{kNN}}) = \sum_{i,j} l(V_{ij}^{\text{kNN}}, L_{ij}^{\text{kNN}}) \quad \text{代入内积损失函数}$$
$$\text{Step 3}: = -\frac{1}{N}\sum_{i=1}^{N} m_{\text{kNN}}(v_i, l_i) \quad \text{代数化简为k近邻集合交集大小的负平均}$$
$$\text{最终}: \mathcal{D}_{\text{kNN}} = -\frac{1}{N}\sum_{i=1}^{N} |\text{kNN}(v_i) \cap \text{kNN}(l_i)|$$

**对应消融**：GW距离在初步分析中被发现优于CKA用于视觉-语言匹配。

---

### 模块 2: 可分解损失函数与代价张量因式分解（对应框架图"求解器"模块核心）

**直觉**：如果成对损失函数l(A,B)能分解为各模态独立函数的组合，则4维代价张量可表示为2维矩阵的外积，从而将QAP复杂度从O(N⁴)降至可处理范围。

**Baseline 公式** (标准GW QAP):
$$\min_{P \in \mathcal{P}_N} \sum_{i,j,k,l=1}^{N} l(V_{ik}, L_{jl}) P_{ij} P_{kl}$$
符号: $P \in \{0,1\}^{N×N}$ = 置换矩阵，$\mathcal{P}_N$ = 置换矩阵集合，$V_{ik}, L_{jl}$ = 视觉/语言成对矩阵。

**变化点**：标准形式中l(V_{ik}, L_{jl})是任意函数，导致代价张量C_{ijkl}无结构，必须直接处理4维张量；本文要求损失函数具有可分解结构。

**本文公式（推导）**:
$$\text{Step 1}: l(A, B) = f_1(A) + f_2(B) - h_1(A)h_2(B) \quad \text{假设损失可分解为单模态函数组合}$$
$$\text{Step 2}: C_{ijkl} = l(V_{ik}, L_{jl}) = C^{(1)}_{ik}C^{(2)}_{jl} - \overline{U}_{ijk} - \overline{V}_{ijl} \quad \text{代价张量精确因式分解：主项+修正项}$$
$$\text{其中}: C^{(1)}_{ik} = h_1(V_{ik}), \quad C^{(2)}_{jl} = h_2(L_{jl})$$
$$\text{Step 3}: \sum_{i,j,k,l} C_{ijkl}P_{ij}P_{kl} = l + \sum_{i,j}\text{leader}_{ij}P_{ij} + \sum_{i,j,k,l}(C_{ijkl} - u^{(ij)}_k - v^{(ij)}_l)P_{ij}P_{kl}$$
$$\text{Hahn-Grant分解：常数项 + 线性leader项 + 简化二次项}$$
$$\text{最终}: P^* \in \argmin_{P \in \mathcal{P}_N} \sum_{i,j,k,l=1}^{N} C^{(1)}_{ik}C^{(2)}_{jl} P_{ij}P_{kl} \quad \text{因式化QAP，可高效求解}$$

**对应消融**：Table 1显示，仅使用局部最优解的替代算法无法产生非平凡匹配结果，全局最优性是核心。

---

### 模块 3: 子集选择用于鲁棒匹配（对应框架图"子集选择"模块）

**直觉**：当类别总数L远大于可可靠匹配的子集大小N时，盲目匹配全部类别会引入噪声；应通过组合优化主动选择最可靠的N个类别。

**Baseline 公式**: 无（标准GW直接匹配全部N个类别）。

**变化点**：本文首次将子集选择引入GW框架，将问题从"固定集合匹配"扩展为"最优子集发现"。

**本文公式（推导）**:
$$\text{Step 1}: S^* \in \argmax_{S} A(S) \quad \text{定义子集质量函数A(S)为匹配可靠性度量}$$
$$\text{Step 2}: \mathbf{s}^* \in \argmax_{\mathbf{s}} \sum_{i,j=1}^{L} l(V_{ij}, L_{ij}) s_i s_j$$
$$\text{s.t. } \mathbf{s} \in \{0,1\}^L, \quad \mathbf{s}^T\text{mathbbm}{1} = N \quad \text{二进制变量s选择恰好N个类别}$$
$$\text{最终}: S^* = \{i \text{mid} s^*_i = 1\} \text{ 为最优匹配子集，再对该子集运行GW求解}$$

**对应消融**：Figure 5/7显示，对每个问题规模选择最优的10个子集，可实现高精度细粒度匹配。

## 实验与分析



本文在多个标准分类数据集上评估无平行数据的视觉-语言匹配性能。核心实验分为小规模精确匹配（N=10）和大规模子集匹配两个阶段。

在小规模设置中，作者使用CIFAR-10和CINIC-10的10个类别，分别提取视觉特征（DINOv2等）和语言特征（all-mpnet-base-v2等），通过因式分解Hahn-Grant求解器寻找最优类别对应。Figure 4的结果显示，**DINOv2视觉特征在CINIC-10上达到100%匹配准确率**，在CIFAR-10上达到72%——这意味着在完全没有成对训练数据的情况下，仅通过两类模态各自的内部结构相似性，就能完美或高度准确地重建类别对应关系。相比随机基线（10%），这分别代表了+90%和+62%的绝对提升。值得注意的是，不同视觉架构的表现差异显著：自监督模型DINOv2明显优于监督训练的ResNet等模型，暗示视觉表示的结构一致性对盲匹配至关重要。


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/9daab291-1940-4c97-8691-430ea81746a5/figures/fig_003.jpeg)
*Figure: The Hahn-Grant solver (left) and the factorized Hahn-Grant solver (ours, right): The Hahn-Grant solver [13] iteratively*



对于更大规模的问题（ImageNet-100、CIFAR-100，N=100），直接全局匹配变得困难。作者引入子集选择模块，从L个类别中优选N个最可靠匹配的子集。Figure 5/7表明，对于每个问题规模，选择最优的10个子集可以实现高精度的细粒度匹配。Figure 6/8的求解器比较显示，**因式分解Hahn-Grant求解器能够找到全局最优解**，而替代优化算法（如直接QAP求解器的局部近似）仅能找到局部最优——关键发现是，只有全局最优解对应于非平凡的匹配结果，局部最优解实际上等效于随机猜测。



消融实验验证了三个核心设计选择的必要性：（1）GW距离优于CKA用于视觉-语言匹配任务，CKA的归一化HSIC结构对跨模态判别不足；（2）全局最优性是必要的，局部最优解无法产生有意义的匹配；（3）kNN稀疏核在保持匹配精度的同时大幅降低计算开销。公平性方面，本文的对比基线ASIF、GeRA等虽被引用，但提供的摘录中缺乏直接的定量比较；此外，实验局限于分类数据集的离散类别设置，未涉及开放词汇场景，也未与完整监督的CLIP风格训练进行直接对比——这些是该方法向实际应用推广时需要补充的环节。

## 方法谱系与知识库定位

本文属于**最优传输（Optimal Transport）→ Gromov-Wasserstein距离 → 因式分解高效求解器**的方法谱系。直接父方法为**Gromov-Wasserstein distance**（Memoli, 2011; Peyré et al., 2016），本文在其基础上修改了两个关键slot：

- **目标函数（objective）**：引入可分解损失函数 l(A,B) = f₁(A) + f₂(B) - h₁(A)h₂(B)，使4维代价张量可因式分解为 C_{ijkl} = C⁽¹⁾_{ik}C⁽²⁾_{jl} - Ū_{ijk} - V̄_{ijl}
- **推断策略（inference_strategy）**：以Hahn-Grant分解替代直接QAP求解或熵正则化，通过leader项+低秩分解实现高效全局优化

**直接基线与差异**：
- **CKA**：同为表示相似性度量，但CKA仅衡量整体相关结构，缺乏GW的成对判别力；本文实验证实GW更优
- **ASIF / GeRA / Graph Optimal Transport / Latent Space Translation**：同为无训练跨模态对齐，但本文首次实现GW距离的高效全局求解，突破规模瓶颈
- **CLIP**：作为监督对比学习的代表，本文方法无需其依赖的数亿级平行数据

**后续方向**：（1）将因式分解求解器扩展至连续/开放词汇设置，超越离散类别限制；（2）结合近期神经最优传输方法（如神经GW变体）进一步提升可扩展性；（3）探索更一般的损失函数分解形式，放宽当前的可分解性假设。

**标签**：modality=vision-language | paradigm=unsupervised alignment / optimal transport | scenario=cross-modal matching without paired data | mechanism=tensor factorization / QAP solver | constraint=no parallel training data, class-level aggregation required

## 引用网络

### 直接 baseline（本文基于）

- [[P__无监督视觉特征学习的DINOv2_DINOv2]] _(方法来源)_: Major visual feature extractor; likely used as backbone or feature source in the

