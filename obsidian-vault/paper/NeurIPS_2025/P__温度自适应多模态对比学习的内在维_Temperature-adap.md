---
title: Multi-modal contrastive learning adapts to intrinsic dimensions of shared latent variables
type: paper
paper_level: C
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 温度自适应多模态对比学习的内在维度适应
- Temperature-adap
- Temperature-adaptive multi-modal contrastive learning
- Temperature optimization in multi-m
acceptance: Poster
cited_by: 4
method: Temperature-adaptive multi-modal contrastive learning
modalities:
- Image
- Text
- biological data
paradigm: self-supervised
---

# Multi-modal contrastive learning adapts to intrinsic dimensions of shared latent variables

**Topics**: [[T__Self-Supervised_Learning]], [[T__Cross-Modal_Matching]] | **Method**: [[M__Temperature-adaptive_multi-modal_contrastive_learning]] | **Datasets**: Synthetic multi-modal data with known intrinsic dimension, Single-cell CITE-seq, Vision-language

> [!tip] 核心洞察
> Temperature optimization in multi-modal contrastive learning enables the learned representations to adapt to the intrinsic dimensions of shared latent variables, producing low-dimensional yet informative representations without explicit dimensionality reduction.

| 中文题名 | 温度自适应多模态对比学习的内在维度适应 |
| 英文题名 | Multi-modal contrastive learning adapts to intrinsic dimensions of shared latent variables |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2505.12473) · [Code](未公开) |
| 主要任务 | Self-Supervised Learning, Cross-Modal Matching |
| 主要 baseline | CLIP, infoNCE, Deep CCA, Multi-view Information Bottleneck |

> [!abstract] 因为「多模态对比学习在用户指定的 d 维空间学习，但共享潜变量的内在维度 k* 可能远小于 d，导致表示冗余低效」，作者在「infoNCE / CLIP 固定温度训练」基础上改了「将温度 τ 变为可优化参数」，在「合成数据与 CITE-seq 单细胞数据」上取得「表示自动坍缩至真实内在维度 k* < d，无需显式降维」

- 关键性能 1：温度优化使表示有效维度从 d 降至 k*，合成骨设置中 d* = 2, r = 5 时观察到维度坍缩（Figure 2）
- 关键性能 2：CITE-seq 真实生物数据上，低维表示保持或提升互信息估计（Figures 10-14）
- 关键性能 3：固定温度 τ 的 infoNCE 无维度坍缩，表示维持完整 d 维

## 背景与动机

多模态对比学习（如 CLIP）旨在从成对的多模态数据（图像-文本、蛋白质-RNA）中学习共享表示。用户需预先指定表示维度 d，但数据真实的共享结构往往存在于更低维的流形上。例如，单细胞 CITE-seq 数据中，蛋白质和 RNA 的表达受少数几个底层生物过程驱动，内在维度 k* 可能仅为 2-5，而实践者常设置 d = 64 或更高。这种维度错配导致表示空间被无效填充，增加计算开销并可能引入噪声。

现有方法如何处理这一问题？**infoNCE**（Oord et al.）通过固定温度 τ 的对比损失最大化互信息，但温度需手动调参，且表示维度固定为 d，无法适应数据内在结构。**Deep CCA**（Andrew et al.）学习非线性变换最大化跨视图相关性，但同样输出固定维度，且对复杂分布假设敏感。**Multi-view Information Bottleneck** 通过信息瓶颈原则学习鲁棒表示，虽隐含维度控制，但需显式设计瓶颈结构，缺乏对内在维度的自适应机制。

这些方法的共同短板在于：**表示维度是人工预设的刚性约束，而非从数据推断的自适应属性**。固定温度 τ 的 infoNCE 尤其关键——温度控制相似度分布的锐度，但固定值无法根据数据复杂度动态调整，导致表示要么过度分散（高维冗余），要么过度坍缩（信息损失）。本文的核心动机正是：能否让温度参数 itself 成为学习的对象，从而使表示维度自动适应共享潜变量的真实复杂度？

本文证明，优化温度 τ 可使 infoNCE 的解集产生几何约束，诱导表示向低维流形坍缩，无需 PCA 等显式降维。

## 核心创新

核心洞察：温度 τ 从超参数变为可优化变量后，infoNCE 损失的解集产生范数约束与相似度约束，从而强制表示集中在与共享潜变量内在维度 k* 匹配的低维流形上，因为温度优化改变了损失景观的几何结构，从而使「无显式降维的自适应维度学习」成为可能。

| 维度 | Baseline (infoNCE/CLIP) | 本文 |
|:---|:---|:---|
| 温度 τ | 固定超参数 τ₀，需人工调参 | 可学习参数，与编码器联合优化 |
| 表示维度 | 刚性输出 d 维，无自适应 | 有效维度自动坍缩至 k* < d |
| 降维机制 | 无，或需外部 PCA/瓶颈层 | 损失函数内在诱导维度坍缩 |
| 理论保证 | 线性表示或特定分布假设 | 超越线性，一般解集刻画 A(H) |

## 整体框架


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/288e9e5a-3f8e-4b8a-8410-d71a7ae4e09c/figures/Figure_1.png)
*Figure 1 (pipeline): Multi-modal contrastive learning applied to the bone marrow single-cell CITE-seq data.*



整体数据流遵循标准多模态对比学习范式，但核心修改在于温度参数的自适应优化：

1. **输入**：成对多模态样本 (x, y) ~ p_XY，如图像-文本对或蛋白质-RNA 对
2. **编码器 f (模态 1)**：将 x 映射至表示向量 f(x) ∈ R^d
3. **编码器 g (模态 2)**：将 y 映射至表示向量 g(y) ∈ R^d
4. **温度缩放相似度计算**：计算内积 sim(f(x), g(y)) 并除以可学习温度 τ，得到自适应锐化的相似度分数
5. **自适应 infoNCE 损失**：基于温度缩放后的正负样本相似度，联合优化编码器参数与温度 τ
6. **输出**：低维有效表示——几何上集中在 k* 维流形上，虽嵌入 R^d 但信息维度与共享潜变量匹配

```
模态1 x ──→ [Encoder f] ──┐
                          ├──→ 内积 sim(·,·) ──→ /τ ──→ 自适应 infoNCE ──→ ∇f, ∇g, ∇τ
模态2 y ──→ [Encoder g] ──┘
                              ↑
                         可学习温度 τ（联合优化）
```

关键区别在于：标准流程中 τ 是训练前固定的标量，而本文 τ 作为模型参数参与反向传播，其最优值隐式编码了数据内在维度信息。

## 核心模块与公式推导

### 模块 1: 自适应 infoNCE 损失（对应框架图 温度缩放 → 损失计算）

**直觉**：温度 τ 控制对比损失中 hardest negatives 的相对权重；让 τ 可学习相当于让数据 itself 决定「应该多关注困难负样本还是均匀对待」。

**Baseline 公式** (infoNCE with fixed temperature):
$$L_N^{\text{fixed}}(f,g,\tau_0) = -\mathbb{E}_{(x,y) \sim p_{XY}} \left[ \log \frac{\exp(\text{sim}(f(x), g(y))/\tau_0)}{\sum_{y'} \exp(\text{sim}(f(x), g(y'))/\tau_0)} \right]$$
符号：$f, g$ = 两模态编码器；$\tau_0$ = 固定温度超参数；$\text{sim}(\cdot,\cdot)$ = 内积相似度；$p_{XY}$ = 联合分布。

**变化点**：固定 τ₀ 无法适应不同数据集的内在复杂度。高维冗余数据需要较小 τ 以聚焦关键结构，但手动调参无法跨数据集泛化；更根本地，固定 τ 的解集无几何约束，表示可充满整个 R^d。

**本文公式（推导）**:
$$\text{Step 1}: \quad L_N(f,g,\tau) = -\mathbb{E}_{(x,y) \sim p_{XY}} \left[ \log \frac{\exp(\text{sim}(f(x), g(y))/\tau)}{\sum_{y'} \exp(\text{sim}(f(x), g(y'))/\tau)} \right] \quad \text{将 τ 松弛为可优化变量}$$
$$\text{Step 2}: \quad \tau^* = \text{arg}\min_{\tau} L_N(f,g,\tau) \quad \text{联合优化：∂L/∂τ 驱动温度收敛至数据适配值}$$
$$\text{最终}: L_N^{\text{adaptive}}(f,g,\tau) \text{ with } \tau \in \mathbb{R}^+ \text{ learnable}$$

**对应消融**：固定 τ 时无维度坍缩，表示维持完整 d 维；可学习 τ 时有效维度降至 k*（Figure 2, Figures 6-9）。

---

### 模块 2: 解集刻画 A(H) 与维度坍缩（对应框架图 输出表示的几何结构）

**直觉**：温度优化不仅改变数值，更改变了损失的最优解集结构——强制表示具有固定范数且跨模态相似度匹配最优函数 h*，这些约束几何上「压扁」了表示空间。

**Baseline 公式**：无对应刻画——标准 infoNCE 仅保证对齐性（alignment）与均匀性（uniformity），无显式维度约束。

**变化点**：需理解温度优化如何诱导低维结构。本文通过变分分析证明，优化后的解满足特定约束，而非任意满足对齐-均匀性的表示。

**本文公式（推导）**:
$$\text{Step 1}: \quad \mathcal{A}(\mathcal{H}) = \{(f,g) : \|f(x)\|^2 = \|g(y)\|^2 = c(\tau), \; \text{sim}(f(x), g(y)) = h^*(x,y) \text{ for some } h^* \in \mathcal{H}\}$$
$$\text{加入了范数约束 } c(\tau) \text{（温度依赖的常数）与相似度匹配约束以刻画最优解集}$$
$$\text{Step 2}: \quad c(\tau) \propto \tau \cdot g(\tau) \quad \text{范数与温度耦合，τ 的收敛值决定表示的「半径」}$$
$$\text{Step 3}: \quad \text{rank}\left(\text{Cov}(f(X))\right) = \text{rank}\left(\text{Cov}(g(Y))\right) = k^* < d \quad \text{约束强制协方差低秩}$$
$$\text{最终}: \text{supp}(f(X)) \subset \mathcal{M}^{k^*} \subset \mathbb{R}^d \quad \text{表示支撑在 k* 维子流形上}$$

符号：$\mathcal{A}(\mathcal{H})$ = 最优编码器对集合；$c(\tau)$ = 温度依赖的范数常数；$h^* \in \mathcal{H}$ = 最优相似度函数类；$\mathcal{M}^{k^*}$ = k* 维黎曼子流形。

**对应消融**：Figure 2 显示温度收敛过程中内在维度从 d 降至 d* = 2；Figures 6-9 的散点图与范数直方图直观展示表示从 R^d 向低维流形的集中。

---

### 模块 3: 内在维度不等式与数据假设（对应框架图 输入数据的生成机制）

**直觉**：维度坍缩有意义的前提是数据本身有低维结构——本文明确将这一假设形式化，并证明温度优化使表示维度匹配而非超过该结构。

**Baseline 公式**：标准多模态学习隐含假设共享潜变量存在，但未显式约束其维度与表示维度的关系。

**变化点**：将「低维共享结构」从直觉假设提升为理论框架的核心变量，使维度自适应具有可验证的预测。

**本文公式**:
$$k^* = \dim(\text{supp}(Z)) < d$$

其中 Z 为共享潜变量，k* 为其支撑集维度。温度优化的关键效应：
$$\tau^* \text{xrightarrow}{\text{converges}} \tau(k^*) \quad \Rightarrow \quad \text{effective-dim}(f(X)) = k^*$$

即最优温度 τ* 是 k* 的隐函数，训练收敛后表示的有效维度自动等于（而非超过）数据内在维度。这与显式降维（如 PCA 选 k 个主成分）形成对照：本文方法无需预设 k*，而是通过 τ 的优化自动发现。

## 实验与分析



本文在三大实验场景验证理论预测：合成控制数据、单细胞 CITE-seq 生物数据、以及有限规模的视觉-语言数据。

**合成数据（骨设置 bone setting）**：这是最具理论价值的验证。数据生成时已知真实内在维度 d* = 2，冗余参数 r = 5，用户指定维度 d = 更高。Figure 2 与 Figure 3 显示，温度优化训练过程中：温度 τ 收敛至稳定值（Figure 2c），同时表示的估计内在维度从初始 d 逐步下降至 d* = 2（Figure 2b），out-of-sample 相似度分布趋于理想（Figure 2a）。关键对比：固定温度 τ 的 infoNCE 维持完整 d 维，无坍缩现象。Figure 4 的 resampled-label switching 设置进一步验证该机制对标签噪声的鲁棒性。


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/288e9e5a-3f8e-4b8a-8410-d71a7ae4e09c/figures/Figure_2.png)
*Figure 2 (result): Histograms of out-of-sample similarities, change of intrinsic dimensions, and convergence of temperature (bone setting: d* = 2, r = 5).*



**CITE-seq 单细胞数据（Figure 5）**：在真实蛋白质-RNA 配对数据上，温度优化同样诱导维度下降。Figure 5 展示内在维度变化与温度收敛的联动关系，与合成数据一致。Figures 10-14（未在可用图中完整列出，但文中报告）显示低维表示在下游生物任务（如细胞类型聚类、互信息估计）上保持或优于高维 baseline，证明维度坍缩未损失信息——反而因去除冗余提升了效率。



**消融分析**：核心消融对比固定 τ vs 可学习 τ。固定 τ 时，表示范数分散、协方差满秩；可学习 τ 时，范数集中至 c(τ)、协方差秩降至 k*。Figure 2 与 Figures 6-9 构成系统消融：去掉温度优化（即固定 τ）后，维度坍缩完全消失，Δ = d - k* 维度的差异。不同初始 τ 值的实验显示，最终收敛 τ* 与 k* 一一对应，而非依赖初始化。

**公平性审视**：实验设计存在明显偏重——合成与生物数据极其详尽，但视觉-语言基准（CLIP 对比）规模有限，缺乏 ImageNet、COCO、Flickr30K 等标准大规模评测。未与 ALIGN、FLAVA 等更强基线对比，也未与 VICReg、Barlow Twins 等含显式维度控制的现代自监督方法比较。作者承认理论分析基于理想优化器与无限样本假设，实际有限样本训练动态可能偏离预测。此外，论文未发布代码，可复现性受限。

## 方法谱系与知识库定位

**方法家族**：对比学习 → 多模态对比学习（CLIP-style）→ 温度自适应变体

**父方法**：infoNCE / CLIP-style multi-modal contrastive learning。本文直接继承其双编码器架构与对比损失范式，修改目标函数中的温度参数为可学习变量，并刻画了训练动态的新性质。

**改变的插槽**：
- **objective**：infoNCE 损失中固定 τ₀ → 可优化 τ
- **training_recipe**：标准端到端训练 → 联合优化编码器与温度，诱导维度坍缩

**直接基线对比**：
- **infoNCE (fixed τ)**：本文直接扩展，τ 可学习是核心差异
- **Deep CCA**：同样多模态，但基于相关性最大化而非对比损失，无温度机制
- **Multi-view Information Bottleneck**：信息论目标显式约束表示维度，需预设瓶颈强度；本文通过温度隐式自适应
- **Spectral Contrastive Loss / Deep InfoMax**：理论框架来源，提供对比学习的互信息视角，但未涉及温度优化与维度适应

**后续方向**：
1. 大规模视觉-语言验证：在 ImageNet-CLIP、LAION 等数据上检验维度适应是否保持，并与 SOTA 对比
2. 温度优化的网络架构：τ 是否应与样本自适应（per-sample temperature）而非全局标量
3. 与其他降维机制结合：如与 VICReg 的协方差正则化、Barlow Twins 的冗余减少项协同

**标签**：modality=多模态（图像/文本/生物） | paradigm=自监督对比学习 | scenario=表示学习/跨模态匹配 | mechanism=温度优化/维度坍缩 | constraint=无显式降维/自适应内在维度

