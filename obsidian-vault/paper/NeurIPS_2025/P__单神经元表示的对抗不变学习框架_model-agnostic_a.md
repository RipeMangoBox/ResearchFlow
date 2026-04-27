---
title: Generalized and Invariant Single-Neuron In-Vivo Activity Representation Learning
type: paper
paper_level: C
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 单神经元表示的对抗不变学习框架
- model-agnostic a
- model-agnostic adversarial training framework for single-neuron representations
- A model-agnostic adversarial traini
acceptance: Poster
method: model-agnostic adversarial training framework for single-neuron representations
modalities:
- neural activity data (electrophysiology/calcium imaging)
paradigm: adversarial training with gradient reversal layer
---

# Generalized and Invariant Single-Neuron In-Vivo Activity Representation Learning

**Topics**: [[T__Self-Supervised_Learning]], [[T__Domain_Adaptation]] | **Method**: [[M__model-agnostic_adversarial_training_framework_for_single-neuron_representations]] | **Datasets**: V1-CellType: Cell Type Prediction Across Visual Stimulus, V1-CellType: Cell Type Prediction Across Animals, IBL Brain-wide Map: Anatomical Brain Region Prediction Across Animals, LOLCAT cross-stimulus generalization, NeuPRINT cross-stimulus generalization

> [!tip] 核心洞察
> A model-agnostic adversarial training strategy with a gradient reversal layer can make single-neuron representation learning invariant to batch information, significantly improving generalization across experimental conditions while remaining compatible with all major existing paradigms.

| 中文题名 | 单神经元表示的对抗不变学习框架 |
| 英文题名 | Generalized and Invariant Single-Neuron In-Vivo Activity Representation Learning |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2025.xxxxx) · [Code](待补充) · [Project](待补充) |
| 主要任务 | 单神经元表示学习、细胞类型预测、脑区预测 |
| 主要 baseline | LOLCAT (Transformer-based)、NEMO (对比学习)、NeuPRINT (VAE) |

> [!abstract] 因为「单神经元表示学习模型受批次效应影响，在跨动物、跨刺激条件下泛化能力差」，作者在「LOLCAT/NEMO/NeuPRINT」基础上改了「加入对抗训练的批次判别器与梯度反转层」，在「V1-CellType 与 IBL Brain-wide Map 跨动物/跨刺激基准」上取得「显著提升的 top-1 accuracy」。

- **关键性能 1**：在 V1-CellType 跨视觉刺激细胞类型预测任务上，对抗训练后的 LOLCAT/NEMO/VAE 相比无对抗版本 top-1 accuracy 提升（详见 Table 1）
- **关键性能 2**：在 V1-CellType 跨动物细胞类型预测任务上，对抗训练显著提升泛化性能（详见 Table 2）
- **关键性能 3**：在 IBL 全脑图谱跨动物脑区预测任务上，对抗训练框架同样有效（详见 Table 3）

## 背景与动机

神经科学研究中，从电生理或钙成像记录中学习单神经元的低维表示是一个核心问题。然而，这些表示往往受到"批次效应"的严重干扰——同一批实验中，动物个体、记录设备、视觉刺激类型等非生物因素会在嵌入空间中留下显著痕迹，导致模型在跨实验条件时泛化崩溃。例如，一个在小鼠 A 的 drifting gratings 刺激上训练好的细胞类型分类器，遇到小鼠 B 的 locally sparse noise 刺激时可能完全失效，因为它学到的"特征"实质是批次标签而非真实的神经功能特性。

现有方法如何处理这一问题？**LOLCAT** [5] 采用 Transformer-based 隐式模型，通过交叉熵损失学习时间不变的单神经元表示，但训练时仅优化任务损失，未显式去除批次信息。**NEMO** 使用对比学习（InfoNCE）拉近同神经元样本、推开不同神经元样本，然而对比对的选择仍受批次分布影响。**NeuPRINT** 基于 VAE 重构损失学习表示，重构目标本身不区分生物变异与批次噪声。这三类方法——分别代表 Transformer-based、对比学习、VAE 三大范式——都假设训练与测试分布一致，未在优化目标中显式约束表示的批次不变性。

它们的共同短板在于：**表示空间 $z = f_\theta(x)$ 仍携带可预测的批次信息**。如果用一个简单分类器能从 $z$ 中准确判断该神经元来自哪个动物、哪种刺激，说明模型并未学到真正泛化的神经特征。这正是本文要解决的核心问题——如何让单神经元表示对批次标签"不可区分"，同时保留下游任务所需的判别信息。


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/004e8c6e-e0f4-47b5-962a-632ad5be4880/figures/Figure_2.png)
*Figure 2 (motivation): Illustration of adversarial training to improve the generalizability of single-neuron representations.*



本文提出一种模型无关的对抗训练框架，通过在现有单神经元模型上嫁接批次判别器与梯度反转层，强制编码器学习批次不变的表示，从而显著提升跨动物、跨刺激的泛化能力。

## 核心创新

核心洞察：**单神经元表示的批次效应可以通过对抗训练显式消除**，因为梯度反转层（GRL）使得编码器在最小化任务损失的同时自动最大化批次判别损失，从而使编码器被迫丢弃批次相关信息、保留生物功能相关信息成为可能。

| 维度 | Baseline（LOLCAT/NEMO/NeuPRINT） | 本文 |
|:---|:---|:---|
| 优化目标 | 仅任务损失 $L_{\text{base}}$（交叉熵/重构/对比） | 联合目标 $L_{\text{base}} - \lambda L_{\text{batch}}$，显式对抗批次判别器 |
| 网络结构 | 仅编码器 $f_\theta$ + 任务头 | 编码器 $f_\theta$ + **梯度反转层** + **两层 MLP 批次判别器 $D_\phi$** |
| 训练方式 | 标准端到端梯度下降 | 极小极大优化：$f_\theta$ 试图骗过 $D_\phi$，$D_\phi$ 试图识别批次 |
| 推理开销 | 与 baseline 相同 | **与 baseline 相同**（判别器仅训练时使用，不增加推理参数） |

关键区别在于：本文不改变任何现有单神经元模型的架构设计，仅通过"即插即用"的对抗包裹层，将领域自适应的经典思想 [12] 迁移到神经科学场景，且兼容全部三大范式。

## 整体框架


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/004e8c6e-e0f4-47b5-962a-632ad5be4880/figures/Figure_1.png)
*Figure 1 (pipeline): Schematic overview of the proposed protocol and adversarial training framework for representation learning of single-neuron activity.*



数据流从左至右依次经过以下模块：

1. **原始神经观测 $X$**：输入为电生理或钙成像记录，经预处理后得到单神经元活动样本 $x_i$。训练/验证使用 drifting gratings、static gratings、natural scenes、natural movies 等多样视觉刺激；测试时暴露于训练未见的 **locally sparse noise** 刺激，以评估真实泛化。

2. **编码器 $f_\theta$**：将 $x_i$ 映射到 $d$ 维嵌入 $z_i = f_\theta(x_i)$。这是现有模型的核心组件——可以是 LOLCAT 的 Transformer、NEMO 的对比 MLP、或 NeuPRINT 的 VAE encoder。

3. **梯度反转层（GRL）**：前向传播时恒等映射 $z_i \to z_i$；反向传播时将来自判别器的梯度取反 $-\nabla_z L_{\text{batch}}$ 传回编码器。这一层是端到端实现 min-max 优化的关键，无需交替优化。

4. **批次判别器 $D_\phi$**：两层 MLP，输入为 $z_i$（经 GRL），输出批次标签预测 $\hat{b}_i$。其目标是尽可能从表示中恢复批次信息。

5. **任务头**：接收 $z_i$ 进行下游预测——细胞类型分类、脑区分类、或重构原始活动。推理时仅使用 $f_\theta$ + 任务头，$D_\phi$ 完全丢弃。

整体优化是一个对抗博弈：$D_\phi$ 努力"揭穿"批次来源，$f_\theta$ 则通过 GRL 接收负梯度，努力"混淆"批次来源，最终被迫去除批次相关特征。

```
Raw Neural Activity x_i
    ↓
[Encoder f_θ] ──→ z_i ──→ [Task Head] ──→ ŷ_i (cell type / brain region)
    │              │
    │              ↓
    │       [Gradient Reversal Layer]
    │              │
    │              ↓
    │       [Batch Discriminator D_φ] ──→ b̂_i (batch label)
    │
    └───── Backward: ∇θ receives -λ∇z L_batch (via GRL)
```

## 核心模块与公式推导

### 模块 1: 联合对抗目标函数（对应框架图中心）

**直觉**：标准训练只关心任务做得对；对抗训练额外要求"即使有个专门抓批次信息的侦探盯着你的表示，它也猜不出来"，从而迫使表示剥离批次噪声。

**Baseline 公式** (LOLCAT/NEMO/NeuPRINT):
$$\min_{\theta} \mathcal{L}_{\text{base}}(f_\theta)$$
符号: $\theta$ = 编码器参数; $\mathcal{L}_{\text{base}}$ = 任务损失（LOLCAT 为交叉熵，NEMO 为 InfoNCE，NeuPRINT/VAE 为重构损失）。

**变化点**：Baseline 的表示 $z_i$ 可能编码批次信息，导致跨批次泛化差。本文引入批次判别器 $D_\phi$ 和对抗博弈，将批次预测作为显式优化目标。

**本文公式（推导）**:
$$\text{Step 1}: \min_{\theta} \mathcal{L}_{\text{base}}(f_\theta) + \max_{\theta} \left(-\lambda \mathcal{L}_{\text{batch}}(D_\phi(f_\theta))\right) \quad \text{编码器要最小化任务损失、同时最大化批次判别损失（让判别器困惑）}$$
$$\text{Step 2}: \min_{\theta} \max_{\phi} \mathcal{L}_{\text{base}}(f_\theta) - \lambda \mathcal{L}_{\text{batch}}(D_\phi(f_\theta)) \quad \text{合并为标量化多目标优化，λ 控制对抗强度}$$
$$\text{Step 3 (GRL 实现)}: \text{Forward } z_i = f_\theta(x_i); \text{ Backward } \nabla_\theta \leftarrow \nabla_\theta \mathcal{L}_{\text{base}} - \lambda \cdot (-\nabla_z \mathcal{L}_{\text{batch}}) \quad \text{GRL 自动实现负梯度，单次反向传播完成}$$
$$\text{最终}: \min_{\theta} \max_{\phi} \mathcal{L}_{\text{base}}(f_\theta) - \lambda \mathcal{L}_{\text{batch}}(D_\phi(f_\theta))$$

**对应消融**：Table 4/5/6 显示移除对抗训练（$\lambda = 0$ 或无判别器）后，各基线模型在跨刺激泛化上性能显著下降。

### 模块 2: 批次判别损失（对应判别器 $D_\phi$）

**直觉**：判别器越能准确预测批次，说明表示泄露的批次信息越多；编码器必须通过对抗来"堵住"这个信息通道。

**Baseline**：无此模块，基线方法不存在批次判别损失。

**本文公式**:
$$\mathcal{L}_{\text{batch}}(D_\phi(f_\theta(x_i))) = -\sum_{i=1}^{N} \left[ b_i \log D_\phi(z_i) + (1-b_i) \log(1-D_\phi(z_i)) \right]$$
符号: $b_i \in \{0,1\}$ = 样本 $i$ 的批次标签（如动物 ID 或刺激类型）; $D_\phi(z_i) \in [0,1]$ = 判别器预测的批次概率; $N$ = batch size。

对于多批次场景，扩展为标准的 **multi-class cross-entropy**:
$$\mathcal{L}_{\text{batch}} = -\sum_{i=1}^{N} \sum_{c=1}^{C} \mathbb{1}_{[b_i=c]} \log \frac{\exp(D_\phi^{(c)}(z_i))}{\sum_{c'} \exp(D_\phi^{(c')}(z_i))}$$
其中 $C$ 为总批次数，$D_\phi^{(c)}$ 为第 $c$ 类的 logit。

**变化点**：此项为完全新增。编码器通过 GRL 接收 $-\nabla_\theta \mathcal{L}_{\text{batch}}$，即梯度上升方向，与判别器的梯度下降方向相反，形成 min-max 博弈。

### 模块 3: 批次判别器架构（对应框架图新增组件）

**直觉**：判别器不需要复杂——如果简单两层 MLP 就能从表示中读出批次，说明批次信息泄露严重；如果编码器成功对抗，即使简单判别器也应失效。

**Baseline**：无此组件。

**本文公式**:
$$\hat{b}_i = D_\phi(z_i) = \text{MLP}_{\text{2-layer}}(z_i) = \sigma(W_2 \cdot \text{ReLU}(W_1 z_i + c_1) + c_2)$$
符号: $W_1 \in \mathbb{R}^{h \times d}, W_2 \in \mathbb{R}^{C \times h}$ = 两层权重; $c_1, c_2$ = 偏置; $h$ = 隐藏层维度; $\sigma$ = softmax（多类）或 sigmoid（二类）。

**对应消融**：Table 4-6 中，当移除整个判别器模块（即标准训练）时，LOLCAT/NEMO/VAE 在跨刺激测试上的 top-1 accuracy 均出现显著跌落，验证了判别器对抗训练的必要性。

## 实验与分析


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/004e8c6e-e0f4-47b5-962a-632ad5be4880/figures/Table_1.png)
*Table 1 (quantitative): Cell Type Prediction Across Visual Stimulus V1 (Top-1). Each cell shows the top-1 Accuracy (%) and standard deviation (%) across 5 random data splits.*



本文在三个核心基准上评估对抗训练框架的有效性。**V1-CellType 跨视觉刺激**（Table 1）测试模型在训练时未见过的 locally sparse noise 刺激上的细胞类型预测能力；**V1-CellType 跨动物**（Table 2）测试对全新动物个体的泛化；**IBL Brain-wide Map 跨动物脑区预测**（Table 3）则在更大规模的全脑记录上验证方法的一般性。实验覆盖 LOLCAT（Transformer-based）、NEMO（对比学习）、NeuPRINT/VAE（生成式）三种基线架构，确保"模型无关"声明的可靠性。


![Table 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/004e8c6e-e0f4-47b5-962a-632ad5be4880/figures/Table_2.png)
*Table 2 (quantitative): Cell Type Prediction Across Animals (V1 Cell Types). Each cell shows the top-1 Accuracy (%) and standard deviation (%) across 5 random data splits.*



从 Table 1-3 可见，对抗训练在所有配置下均带来一致提升。以 LOLCAT 为例，在跨刺激细胞类型预测上，标准训练版本因过拟合到训练刺激的批次特征，面对 locally sparse noise 时性能显著衰减；而加入对抗训练后，编码器被迫去除刺激类型相关的虚假相关性，top-1 accuracy 获得明显改善。类似地，在跨动物设置中，对抗训练使表示对动物个体 ID"不可区分"，从而学习到更纯粹的功能特征。NEMO 和 VAE 基线上也观察到同向提升，验证了框架的范式无关性。


![Table 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/004e8c6e-e0f4-47b5-962a-632ad5be4880/figures/Table_3.png)
*Table 3 (quantitative): Anatomical Brain Region Prediction Across Animals (BLA Multi-Session). Each cell shows the top-1 Accuracy (%) and standard deviation (%) across 5 random data splits.*



消融实验（Table 4-6）进一步定位关键组件。移除对抗训练（$\lambda = 0$）后，所有基线模型的跨泛化性能均出现最大幅度的跌落，表明批次不变性约束是提升泛化的核心驱动力。相比之下，调整判别器隐藏层维度等架构细节影响较小，说明方法对判别器容量不敏感。此外，Figure 3 的 UMAP 可视化定性展示了对抗训练前后嵌入空间的变化：对抗训练前，同一批次的神经元聚类明显；对抗训练后，聚类结构按细胞类型和脑区条件重组，批次结构被有效抹平。


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/004e8c6e-e0f4-47b5-962a-632ad5be4880/figures/Figure_3.png)
*Figure 3 (qualitative): UMAP visualization of single-neuron embeddings from the NeurDFF model colored by cell type (left) and brain/nucleus condition (right). Panels (a) and (b) correspond to models trained without and with adversarial training, respectively. Without adversarial training, embeddings cluster by condition rather than cell type.*



公平性方面，本文的比较设计较为合理：每个基线的"有对抗"与"无对抗"版本共享相同编码器架构和超参搜索空间，仅差异在于是否加入 $D_\phi$ 和 GRL。然而，实验仍存在若干局限：一是未与单细胞基因组学中成熟的批次校正方法（如 Harmony、Scanorama）或 IRM 等其他领域自适应技术直接对比；二是超参 $\lambda$ 需交叉验证，其实际取值可能因数据集而异；三是当前评估集中于视觉皮层（V1）神经元，对其它脑区或物种的泛化声明尚未经验证。

## 方法谱系与知识库定位

本文方法谱系属于**领域自适应 / 不变表示学习**分支，直接父方法为 **Domain-Adversarial Neural Networks (DANN)** [12]。DANN 最初用于计算机视觉中的跨领域图像分类，本文将其核心机制——梯度反转层驱动的对抗训练——迁移至神经科学场景，并扩展为"模型无关包装器"，可包裹任意单神经元表示学习模型。

**改变的插槽**：
- **training_recipe**：标准任务损失 → 联合对抗 min-max 优化
- **architecture**：仅编码器 → 编码器 + GRL + 批次判别器（推理时丢弃判别器）
- **objective**：单一任务目标 → 任务目标 + 批次混淆目标

**直接基线差异**：
- **LOLCAT** [5]：同组前期工作，Transformer-based 隐式模型，本文在其上添加对抗包裹层
- **NEMO**：对比学习范式，本文证明对抗训练与 InfoNCE 可兼容
- **NeuPRINT** [7]/VAE：生成式范式，本文证明对抗训练与重构损失可兼容
- **[8]**（跨物种细胞类型识别 Cell 论文）：未作为对抗训练的基线，但代表该领域最强监督方法之一

**后续方向**：
1. **无批次标签场景**：当前框架假设批次标签已知且良定义，探索自监督批次发现或连续域自适应可扩展适用场景
2. **多模态批次效应**：同时处理记录平台、动物个体、刺激类型等多维批次因子的联合不变学习
3. **因果机制解释**：将批次不变表示与已知神经生物机制（如细胞类型特异性响应特性）建立可解释联系

**标签**：modality: 神经电生理/钙成像 | paradigm: 对抗表示学习 / 领域自适应 | scenario: 单神经元分析 / 跨批次泛化 | mechanism: 梯度反转层 / 极小极大优化 | constraint: 批次标签可用 / 模型无关兼容

