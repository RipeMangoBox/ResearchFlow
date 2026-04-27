---
title: When Does Closeness in Distribution Imply Representational Similarity? An Identifiability Perspective
type: paper
paper_level: A
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 可识别性视角下的分布接近与表示相似
- d^λ_LLV distance
- Small KL divergence between model d
acceptance: Poster
cited_by: 4
code_url: https://github.com/bemigini/close-dist-rep-sim
method: d^λ_LLV distance
modalities:
- Image
- Text
paradigm: supervised
---

# When Does Closeness in Distribution Imply Representational Similarity? An Identifiability Perspective

[Code](https://github.com/bemigini/close-dist-rep-sim)

**Topics**: [[T__Self-Supervised_Learning]], [[T__Interpretability]] | **Method**: [[M__d^λ_LLV_distance]] | **Datasets**: [[D__CIFAR-10]] (其他: Synthetic data)

> [!tip] 核心洞察
> Small KL divergence between model distributions does not guarantee representational similarity, but there exists a specific distributional distance (d^λ_LLV) for which closeness does imply bounded representational dissimilarity.

| 中文题名 | 可识别性视角下的分布接近与表示相似 |
| 英文题名 | When Does Closeness in Distribution Imply Representational Similarity? An Identifiability Perspective |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2506.03784) · [Code](https://github.com/bemigini/close-dist-rep-sim) · [Project](https://github.com/bemigini/close-dist-rep-sim) |
| 主要任务 | Representational Similarity Analysis, Self-Supervised Learning, Interpretability |
| 主要 baseline | Kullback-Leibler divergence, CKA (Centered Kernel Alignment) |

> [!abstract] 因为「分布接近是否保证表示相似缺乏理论保证」，作者在「KL散度」基础上提出了「d^λ_LLV 对数似然变化距离与∼_L可识别性等价关系」，在「CIFAR-10与合成数据实验」上证明了「KL接近时仍可表示不相似，而d^λ_LLV接近时保证表示差异有上界2Mε」

- **核心否定结果**：KL散度接近零时，表示差异仍可达最大（Theorem 4.3）
- **核心肯定结果**：d^λ_LLV距离为ε时，表示不相似度d_{f,g} ≤ 2Mε（Theorem 4.7）
- **实证发现**：CIFAR-10上近最大似然模型可学习高度不相似的表示

## 背景与动机

深度学习社区长期面临一个根本困惑：两个神经网络在相同数据上训练到相似的预测分布，它们学到的内部表示是否必然相似？例如，两个自回归语言模型在相同语料上达到相近的perplexity，它们的隐藏层激活是否可以互换使用？传统上，研究者用CKA（Centered Kernel Alignment）等表示相似性度量直接比较激活矩阵，或用KL散度比较输出分布，但这两类方法之间缺乏理论桥梁。

现有方法的处理方式各有局限：
- **KL散度**：作为标准的分布距离，仅衡量输出概率的差异，完全不涉及内部表示结构。KL(p_f ‖ p_g) ≈ 0 只说明两个模型"说同样的话"，不说明它们"用同样的方式思考"。
- **CKA**：直接计算表示空间的核相似性，但缺乏分布层面的理论根基——为什么CKA高就意味着模型行为相似？这种联系从未被严格建立。
- **模型缝合（Model Stitching）**[3]：通过交换模型中间层评估兼容性，操作性强但无法给出普适的理论保证。

这些方法的共同短板在于：**分布接近与表示相似之间的关系完全是启发式的**。KL小不代表表示像，CKA高也不代表分布像。更危险的是，近期工作显示即使模型达到近最大似然，其表示仍可能截然不同——这意味着我们赖以评估模型等价性的工具可能系统性地失效。本文从**可识别性理论（Identifiability Theory）**出发，严格刻画了何时分布接近能蕴含表示相似，并构造了一个反例证明KL散度不足以担当此任，最终提出了一个具有保证的新型分布距离。


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/55df8db6-5910-4309-b66a-872364b3255a/figures/Figure_2.png)
*Figure 2 (example): Two models with small KL divergence but highly dissimilar representations*



## 核心创新

核心洞察：表示的等价性应由其生成的似然分布来定义，而非几何结构直接比较；因为可识别性理论告诉我们，只有产生相同似然的表示在统计意义上才是不可区分的，从而使"分布距离→表示相似"的严格蕴含关系成为可能。

| 维度 | Baseline (KL散度 / CKA) | 本文 |
|:---|:---|:---|
| **等价性定义** | 无显式定义，或基于激活空间几何 | ∼_L：产生相同似然分布的表示等价 |
| **分布距离** | KL散度 D_KL(p_f ‖ p_g)，忽略表示结构 | d^λ_LLV，基于对数似然变化，显式关联表示 |
| **理论保证** | KL小 ⇏ 表示相似（无保证） | d^λ_LLV ≤ ε ⇒ d_{f,g} ≤ 2Mε（Theorem 4.7） |
| **适用架构** | 任意 | embedding-unembedding架构（含自回归LM） |

本文的关键突破在于将表示相似性问题从"几何比较"转化为"统计可识别性"问题：不是直接问"两个表示矩阵有多像"，而是问"在什么分布距离下，相似的分布必然对应统计等价的表示"。

## 整体框架


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/55df8db6-5910-4309-b66a-872364b3255a/figures/Figure_1.png)
*Figure 1 (pipeline): When closeness in distribution does and does not imply representational similarity*



本文的理论框架围绕四个核心模块展开，形成从分布比较到表示保证的完整链条：

1. **模型分布生成（p_f, p_g）**：输入为两个带表示的模型，每个模型由编码器 f/g 和解码器（unembedding）组成，生成数据分布 p_f, p_g。该架构涵盖自回归语言模型（embedding + unembedding）和典型编码器-解码器结构。

2. **可识别性等价判定（∼_L）**：检验两个表示是否产生完全相同的似然分布，即 (f,g) ∼_L (f',g') ⟺ p_f = p_f' ∧ p_g = p_g'。这是表示"应被视为相同"的理论基准。

3. **d^λ_LLV 距离计算**：新型分布距离，通过分析对数似然随表示变化的敏感度来度量分布差异，显式编码了表示结构信息，区别于KL散度的纯输出比较。

4. **表示相似性上界输出（Theorem 4.7）**：当 d^λ_LLV(p_f, p_g) = ε 时，输出保证 d_{f,g}(r_f, r_g) ≤ 2Mε，其中 M 为Lipschitz常数，d_{f,g} 为表示不相似度度量。

数据流可概括为：
```
模型参数 (f, g) → 生成分布 p_f, p_g → 计算 d^λ_LLV 
                                    ↓
等价类验证 ∼_L ←── 若 d^λ_LLV ≤ ε ──→ 保证 d_{f,g} ≤ 2Mε
```

该框架的精髓在于：**不是先比较表示再猜测分布关系，而是先度量分布距离再推导表示界限**，从根本上逆转了传统分析的逻辑方向。

## 核心模块与公式推导

### 模块 1: 可识别性等价关系 ∼_L（框架图左侧）

**直觉**：两个表示只有在统计上不可区分——即产生完全相同的观测似然——时才应被视为等价，这是可识别性理论的基本立场。

**本文公式**：
$$(f, g) \sim_L (f', g') \iff p_f = p_{f'} \text{ and } p_g = p_{g'}$$

符号：$f, g$ 为两个模型的编码器/表示函数；$p_f, p_g$ 为对应的生成分布；∼_L 为基于似然相等的等价关系。

**变化点**：传统表示相似性分析（如CKA）直接在激活空间定义距离，缺乏统计基础；本文将等价性锚定于**观测分布的不可区分性**，使得后续的"分布→表示"蕴含关系有严格的数学根基。

**对应消融**：该等价关系是理论基石，无直接消融；但 Figure 2 显示违反此直觉的情况——KL接近的模型可具有截然不同的表示结构。

---

### 模块 2: KL散度的失效与反例构造（Theorem 4.3，框架图上方"否"分支）

**直觉**：KL散度仅约束输出概率的比值，对表示空间的内部结构完全自由，因此两个模型可以在输出几乎相同的情况下，将输入数据编码到完全不同的表示流形上。

**Baseline 公式** (KL散度): $$D_{\text{KL}}(p_f \| p_g) = \mathbb{E}_{x \sim p_f}\left[\log \frac{p_f(x)}{p_g(x)}\right]$$
符号：$p_f(x), p_g(x)$ 为模型在数据$x$上的输出概率；期望取自$p_f$分布。

**变化点**：KL散度衡量的是**输出概率的差异**，而非**表示结构的差异**。关键观察：对于embedding-unembedding架构，解码器可以"补偿"编码器的任意可逆变换——即 $f' = A \circ f$ 配合 $g' = g \circ A^{-1}$ 可保持输出分布不变，但表示空间 $r_f$ 与 $r_{f'}$ 经线性变换 $A$ 后可能面目全非。

**本文公式（反例构造）**：
$$\text{Step 1}: \exists A \in \text{GL}(d) \text{ s.t. } f' = A \circ f, \quad g' = g \circ A^{-1} \quad \text{（保持输出，改变表示）}$$
$$\text{Step 2}: D_{\text{KL}}(p_{f'} \| p_f) = 0 \text{ (精确相等) 或 } \approx 0 \text{ (数值近似)} \quad \text{（KL无法检测此变换）}$$
$$\text{最终}: d_{f,f'}(r_f, r_{f'}) = \max \text{ (表示差异可达最大)} \quad \text{（Theorem 4.3）}$$

**对应消融**：Figure 2 展示了具体反例——两个模型KL散度极小，但表示可视化显示高度不相似的CollapseMap结构。

---

### 模块 3: d^λ_LLV 距离与表示保证（Theorem 4.7，框架图下方"是"分支）

**直觉**：需要一种分布距离，能够"看穿"解码器的补偿能力，直接度量表示变化对似然的敏感度——对数似然随表示变化的速率本身就是表示结构的探针。

**Baseline 公式** (KL散度，同上): $$D_{\text{KL}}(p_f \| p_g) = \mathbb{E}_{x \sim p_f}\left[\log \frac{p_f(x)}{p_g(x)}\right]$$

**变化点**：KL散度是**全局积分量**，对表示的局部变化不敏感；d^λ_LLV 是**微分/变分结构**，通过分析对数似然关于表示的变分导数来捕获结构信息。

**本文公式（推导）**：
$$\text{Step 1}: \text{定义对数似然变化算子 } \delta_\lambda \log p_f(r) = \frac{\partial \log p_f}{\partial r} \cdot \delta r + O(\|\delta r\|^2) \quad \text{（局部线性化表示扰动）}$$
$$\text{Step 2}: d^\lambda_{\text{LLV}}(p_f, p_g) := \sup_{\|v\| \leq 1} \left| \mathbb{E}_{x}\left[ \delta_\lambda \log p_f(r_f(x)) [v] - \delta_\lambda \log p_g(r_g(x)) [v] \right] \right| \quad \text{（最大化方向上的期望差异）}$$
$$\text{Step 3}: \text{假设 } g \text{ 是 } M\text{-Lipschitz，且表示距离 } d_{f,g} \text{ 由解码器差异诱导}$$
$$\text{最终}: d_{f,g}(r_f, r_g) \leq 2M \cdot d^\lambda_{\text{LLV}}(p_f, p_g) = 2M\varepsilon \quad \text{（Theorem 4.7）}$$

符号：$\lambda$ 为控制变分尺度的超参数；$\delta_\lambda$ 表示在尺度$\lambda$下的变分；$M$ 为解码器（unembedding）的Lipschitz常数；$d_{f,g}$ 为表示不相似度度量。

**对应消融**：Table 1 显示了 d^λ_LLV 与表示相似性度量的相关性，以及KL散度在此关系上的失效；合成实验中（Figures 17-18），网络宽度变化直接调控 d^λ_LLV 大小与表示相似程度的正相关关系。

## 实验与分析


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/55df8db6-5910-4309-b66a-872364b3255a/figures/Table_1.png)
*Table 1 (quantitative): Relating distributional distance and representational similarity*



本文在两类实验设置上验证理论：CIFAR-10上的反例构造与合成数据上的正例验证。Table 1 汇总了分布距离与表示相似性之间的定量关系。

在 **CIFAR-10** 实验中，作者训练多个ResNet模型至近最大似然（near-optimal loss），发现这些模型的KL散度接近零，但表示相似性（通过CKA及CollapseMap可视化评估）却差异显著。Figure 3 展示了两个此类模型的CollapseMap表示——尽管输出分布几乎不可区分，其内部表示的结构化特征（如类别分离模式、流形拓扑）明显不同。这一结果直接验证了Theorem 4.3的核心论断：KL散度作为分布距离的失效是**系统性**的，而非数值误差。


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/55df8db6-5910-4309-b66a-872364b3255a/figures/Figure_3.png)
*Figure 3 (result): CollapseMap representations for two models trained on CIFAR-10*



在 **合成数据** 实验中，作者系统调控网络宽度，发现更宽的网络在d^λ_LLV距离下更接近，且其表示相似性更高（Figures 17-18）。这一正相关关系支持了Theorem 4.7的预测：d^λ_LLV的减小确实蕴含表示差异的上界收缩。值得注意的是，同一设置下KL散度与表示相似性无稳定关联，再次凸显d^λ_LLV的结构感知优势。



消融分析聚焦于**网络宽度**这一架构因素：窄网络在d^λ_LLV下距离更大且表示更不相似，宽网络则相反。该消融间接验证了d^λ_LLV对模型容量的敏感性——容量不足时，模型被迫学习"捷径"表示以拟合分布，导致d^λ_LLV膨胀与表示畸变。

公平性检查：本文的实证设置（CIFAR-10、合成数据）规模有限，未涉及大规模语言模型；缺乏与CKA在同一任务上的直接数值对比；d^λ_LLV的计算成本未与KL散度进行基准测试。此外，baseline KL散度的选择是合理的（作为最标准的分布距离），但缺少与其他分布距离（如Wasserstein、MMD）的比较。作者明确披露了这些局限，包括架构限制（embedding-unembedding）与计算开销问题。

## 方法谱系与知识库定位

本文属于**可识别性理论（Identifiability Theory）**方法谱系，直接继承自**非线性ICA可识别性**研究[9][17]与**因果表示学习**[10][18]的理论传统。父方法为Hyvärinen等人的可识别非线性ICA框架，本文将其从"独立成分恢复"问题迁移至"表示等价判定"问题。

**改变槽位**：
- **目标函数（objective）**：KL散度 → d^λ_LLV（对数似然变化距离）
- **推断策略（inference_strategy）**：无分布根基的CKA比较 → ∼_L等价类+Theorem 4.7保证
- **架构约束（architecture）**：通用DNN → embedding-unembedding架构（含自回归LM）

**直接baseline对比**：
- **KL散度**：本文证明其不足以保证表示相似（Theorem 4.3），d^λ_LLV填补此理论漏洞
- **CKA**：缺乏分布层面的理论根基，本文提供KL/d^λ_LLV作为其缺失的理论补充
- **模型缝合**[3]：操作性强但无普适保证，本文提供严格的上界理论

**后续方向**：
1. 将d^λ_LLV扩展至大规模语言模型，验证其在Transformer上的计算可行性与实证相关性
2. 探索其他分布距离（如Wasserstein）在可识别性框架下的类似保证，建立更完整的"分布-表示"蕴含图谱
3. 利用d^λ_LLV指导模型蒸馏或合并，以理论保证的表示相似性替代启发式匹配

**标签**：modality=[image, text] | paradigm=[theoretical_analysis, identifiability_theory] | scenario=[model_interpretability, representation_alignment] | mechanism=[log_likelihood_variation, lipschitz_bound] | constraint=[embedding_unembedding_architecture]

