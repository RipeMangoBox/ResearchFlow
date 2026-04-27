---
title: Gradient-based Parameter Selection for Efficient Fine-Tuning
type: paper
paper_level: C
venue: CVPR
year: 2024
paper_link: null
aliases:
- 基于梯度选择的参数高效微调GPS
- GPS (Gradient-ba
- GPS (Gradient-based Parameter Selection)
acceptance: Poster
cited_by: 47
code_url: https://github.com/synbol/Awesome-Parameter-Efficient-Transfer-Learning
method: GPS (Gradient-based Parameter Selection)
---

# Gradient-based Parameter Selection for Efficient Fine-Tuning

[Code](https://github.com/synbol/Awesome-Parameter-Efficient-Transfer-Learning)

**Topics**: [[T__Classification]], [[T__Medical_Imaging]], [[T__Semantic_Segmentation]] | **Method**: [[M__GPS]] | **Datasets**: [[D__FGVC]], [[D__CIFAR-100]], [[D__ImageNet-1K]] (其他: FGVC Multi-architecture)

| 中文题名 | 基于梯度选择的参数高效微调GPS |
| 英文题名 | Gradient-based Parameter Selection for Efficient Fine-Tuning |
| 会议/期刊 | CVPR 2024 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2312.10136) · [Code](https://github.com/synbol/Awesome-Parameter-Efficient-Transfer-Learning) · [Project](待补充) |
| 主要任务 | 图像分类（细粒度视觉分类 FGVC、CIFAR-100、ImageNet-1k） |
| 主要 baseline | Full Fine-tuning, Linear Probing, Bias (BitFit), Adapter, VPT, SSF, LoRA, SPT-Adapter, SPT-LoRA |

> [!abstract] 因为「传统PEFT方法要么引入额外参数（Adapter/LoRA/VPT），要么随机选择参数（Bias/Linear），导致精度-效率权衡不佳」，作者在「SSF」基础上改了「用无头监督对比损失计算梯度，按神经元粒度top-K选择重要连接并掩码更新」，在「FGVC 5数据集平均」上取得「91.78% Top-1 Accuracy，超过SSF +1.06%、超过Full Fine-tuning +2.67%」

- **FGVC 5数据集平均**: 91.78%，较SSF（90.72%）提升 +1.06%，较Full Fine-tuning（88.54%）提升 +2.67%
- **ImageNet-1k (ViT-B/16)**: 83.91%，较SSF（83.10%）提升 +0.81%，较Full Fine-tuning（83.58%）提升 +0.33%
- **参数量**: ViT-B/16 上仅 0.66M（0.77%），推理无额外计算开销

## 背景与动机

大规模预训练模型（如ViT、Swin、ConvNeXt）在下游任务微调时，全量微调（Full Fine-tuning）需要更新全部参数，存储和计算开销巨大；而线性探测（Linear Probing）只训练分类头，表达能力不足。现有参数高效微调（PEFT）方法主要分为两类："加法型"（Adapter插入瓶颈层、VPT添加可学习prompt、LoRA引入低秩矩阵、SSF做特征缩放平移）和"选择型"（BitFit仅更新bias项、Linear Probing）。以SSF为例，它虽达到此前PEFT最优，但仍需引入额外参数（0.45%）并修改特征流；Adapter、VPT等方法更因新增模块导致推理延迟增加。


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/d226e203-98fa-42b9-b377-d15264c2cd0d/figures/fig_001.png)
*Figure: Comparison between our GPS and other PEFT methods. (a) Exiting popular methods introduce extra parameters for tuning*



具体而言，现有方法存在三个关键缺陷：（1）**加法型方法引入推理开销**——Adapter的瓶颈层、VPT的prompt序列在测试时仍需计算，无法合并回主干；（2）**选择型方法缺乏任务适应性**——BitFit固定选择所有bias参数，未考虑不同任务对参数的不同需求；（3）**梯度估计不准确**——传统方法用随机初始化的分类头计算交叉熵梯度，head的不可靠性污染了参数重要性判断。例如，在细粒度鸟类分类中，预训练模型的低层边缘检测参数与高层语义参数的重要性分布截然不同，全局统一选择必然次优。

本文提出GPS：通过无头监督对比损失获得可靠梯度信号，以神经元为单位top-K选择关键连接，实现"选择即更新"的零开销推理微调。

## 核心创新

核心洞察：**参数重要性应由其在预训练知识上的梯度响应决定，而非人为预设结构**，因为预训练模型的参数已蕴含丰富语义，微调时只需激活与任务最相关的连接子网络，从而使"无额外参数、无推理开销、任务自适应"的三重目标同时成为可能。

| 维度 | Baseline (SSF/Adapter/LoRA) | 本文 GPS |
|:---|:---|:---|
| **参数来源** | 新增可学习参数（缩放/平移/低秩/适配器） | 仅选择已有参数子集，零新增 |
| **梯度计算** | 交叉熵损失 + 随机初始化分类头 | 监督对比损失，无分类头，梯度更可靠 |
| **选择粒度** | 全局统一（如所有bias，或所有层共享结构） | 逐神经元top-K，任务自适应 |
| **推理开销** | 有（Adapter需过瓶颈，VPT需加prompt） | 无（掩码可合并为稀疏权重） |
| **目标函数** | 交叉熵 $\mathcal{L}_{CE}$ | 监督对比损失 $\mathcal{L}_{SupCon}$ |

## 整体框架


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/d226e203-98fa-42b9-b377-d15264c2cd0d/figures/fig_002.png)
*Figure: Performance comparisons of 11 fine-tuning methods*



GPS的整体流程如上图所示，包含四个核心阶段：

1. **预训练主干（Pre-trained Backbone）**：接收输入图像，输出特征表示。该模块完全固定，不参与任何参数更新。

2. **无头对比梯度计算（Head-free Contrastive Gradient Estimation）**：将特征与标签送入监督对比损失，**不经过随机初始化的分类头**，直接计算每个权重参数的梯度幅值。此步骤的关键在于避免分类头的随机初始化对梯度信号的污染。

3. **逐神经元Top-K选择（Top-K Per-Neuron Selection）**：对每个神经元（输出维度）的输入连接，按梯度幅值选取top-K个最重要连接，生成二进制掩码 $\boldsymbol{M}_j$。K为超参数，控制总参数量预算。

4. **掩码微调（Masked Fine-tuning）**：仅允许掩码为1的位置更新，其余参数保持预训练值。训练完成后，掩码可与权重合并，推理时无任何额外计算。

数据流总结：
```
Input Image → [Pre-trained Backbone] → Features + Labels 
    → [SupCon Loss, No Head] → Gradient Magnitudes |∇L|
    → [Top-K per Neuron] → Binary Mask M_j
    → [Masked SGD: W ⊙ M_j] → Updated Sparse Weights
    → [Merge & Inference] → Zero Overhead
```

## 核心模块与公式推导

### 模块 1: 无头对比梯度估计器（对应框架图阶段2）

**直觉**：随机初始化的分类头会给梯度引入噪声，而监督对比损失直接在特征空间衡量类间分离度，无需head即可获得更纯净的参数重要性信号。

**Baseline 公式** (Full Fine-tuning / SSF / Adapter):
$$\mathcal{L}_{CE} = -\sum_{i} \log \frac{\exp(\boldsymbol{W}_{head}^\text{top} \boldsymbol{f}_i / \tau)}{\sum_{j} \exp(\boldsymbol{W}_{head}^\text{top} \boldsymbol{f}_j / \tau)}$$
符号: $\boldsymbol{W}_{head}$ = 随机初始化的分类头, $\boldsymbol{f}_i$ = 样本特征, $\tau$ = 温度系数

**变化点**：交叉熵依赖$\boldsymbol{W}_{head}$的随机初始化，早期训练时梯度方向不可靠；且head本身需要额外参数。本文移除head，改用特征空间的监督对比损失。

**本文公式（推导）**:
$$\text{Step 1}: \quad \mathcal{L}_{SupCon} = \sum_{i \in I} \frac{-1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(\boldsymbol{z}_i \cdot \boldsymbol{z}_p / \tau)}{\sum_{a \in A(i)} \exp(\boldsymbol{z}_i \cdot \boldsymbol{z}_a / \tau)} \quad \text{（加入同类正样本对，无需分类头）}$$
$$\text{Step 2}: \quad \nabla \mathcal{L}_{SupCon}(\boldsymbol{W}_j) = \frac{\partial \mathcal{L}_{SupCon}}{\partial \boldsymbol{W}_j} \quad \text{（直接对主干权重求梯度，保证信号纯净）}$$
$$\text{最终}: \quad \nabla \mathcal{L}(\boldsymbol{W}_j) = \nabla \mathcal{L}_{SupCon}(\boldsymbol{W}_j; \text{no head})$$

**对应消融**：使用交叉熵损失（带head）替代后，FGVC平均精度下降 0.67%（91.78% → 91.11%）。

---

### 模块 2: 逐神经元Top-K选择掩码（对应框架图阶段3-4）

**直觉**：全局top-K可能导致某些神经元过度激活而另一些"饿死"，逐神经元保证每个输出单元都有适量的输入连接可学习，实现更均衡的子网络结构。

**Baseline 公式** (BitFit / 全局剪枝):
$$\boldsymbol{M} = \mathbb{1}_{[|\nabla \mathcal{L}(\boldsymbol{W})| \geq \text{global-threshold}]} \quad \text{或} \quad \boldsymbol{M}_{bias} = \mathbf{1} \text{ (固定选所有bias)}$$
符号: $\mathbb{1}$ = 指示函数, global-threshold = 全局阈值

**变化点**：全局阈值难以适应不同层、不同神经元的梯度分布差异；固定选择bias完全忽略任务特性。本文改为逐神经元竞争，每个神经元独立选择其最重要的K个输入。

**本文公式（推导）**:
$$\text{Step 1}: \quad \text{对每个神经元 } i \text{（输出维度），计算其所有输入连接的梯度幅值} \quad |\nabla \mathcal{L}(\boldsymbol{W}_j)_{i,:}|$$
$$\text{Step 2}: \quad \text{top-K}_i(|\nabla \mathcal{L}(\boldsymbol{W}_j)_{i,:}|) \rightarrow \boldsymbol{M}_{j,i,k} = \begin{cases} 1 & \text{if } k \in \text{top-K}_i \\ 0 & \text{otherwise} \end{cases} \quad \text{（重归一化选择粒度，保证每神经元恰好K个连接）}$$
$$\text{Step 3}: \quad \boldsymbol{W}_j \leftarrow \boldsymbol{W}_j - \epsilon \nabla \mathcal{L}(\boldsymbol{W}_j) \odot \boldsymbol{M}_j \quad \text{（Hadamard积掩码，仅更新选中参数）}$$

**对应消融**：Table 6（或相关图表）显示不同K值（1-15）下各数据集存在明确性能峰值，验证"更多参数≠更好"，GPS通过K灵活控制预算。

---

### 模块 3: 掩码梯度下降的统一形式

将上述模块整合，GPS的完整更新规则为：
$$\boldsymbol{W}_j^{(t+1)} = \boldsymbol{W}_j^{(t)} - \epsilon \cdot \nabla \mathcal{L}_{SupCon}(\boldsymbol{W}_j^{(t)}) \odot \boldsymbol{M}_j$$
其中掩码 $\boldsymbol{M}_j$ 在训练前通过单次前向-反向传播确定，训练期间固定（或可选周期性更新）。推理时 $\boldsymbol{W}_j \odot \boldsymbol{M}_j$ 可作为稀疏矩阵存储，或直接合并为等效稠密权重，实现**零额外计算**。

## 实验与分析



本文在多个基准上验证GPS的有效性。主实验Table 2显示，在**FGVC 5数据集平均**上，GPS达到**91.78%** Top-1 Accuracy，超越此前PEFT最优方法SSF（90.72%）**+1.06个百分点**，更大幅领先全量微调（88.54%）**+2.67个百分点**。这一结果表明：即使仅更新0.77%的参数（ViT-B/16上0.66M），GPS仍能提取比更新全部参数更优的任务表示，说明其选择机制有效激活了预训练知识中最相关的子网络。



跨架构扩展性方面（Table 9），GPS在Swin-B上达92.56%（+1.02% over SSF），在ConvNeXt-B上达93.32%（+0.84% over SSF），验证了其**与架构无关**的通用性。在标准基准上，ImageNet-1k（ViT-B/16）83.91%（+0.81% over SSF），CIFAR-100 94.02%（+0.03% over SSF），显示在数据量充足时优势收窄但仍保持领先。



消融实验揭示两个关键发现：（1）**无头对比损失至关重要**——替换为交叉熵（带随机初始化head）后FGVC平均下降**0.67%**，证明head的随机初始化确实污染梯度估计；（2）**逐神经元选择优于全局策略**——不同K值（1-15）下各数据集存在明确性能峰值（Figure 6(c)），且GPS通过单一K即可跨任务取得竞争性能，无需逐任务搜索复杂结构。随机种子鲁棒性测试（Table 12/Table 6）显示标准差很小，选择方案稳定。


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/d226e203-98fa-42b9-b377-d15264c2cd0d/figures/fig_003.png)
*Figure: The overall pipeline of GPS. We first select a small portion of important parameters (sub-network) for each task from the original*



效率方面（Figure 4），GPS训练时间与SSF相当，训练内存显著低于Adapter/LoRA等加法方法；**测试时间和测试内存与全量微调完全一致**，因掩码可完全合并。参数量0.66-0.83M虽略高于SSF（0.45%），但精度-效率权衡更优。

公平性检查：对比基线包含SSF（此前PEFT SOTA）、Full Fine-tuning、主流加法方法（Adapter/VPT/LoRA），覆盖面较全。但缺少IA³、DoRA、OFT等2023-2024年更新的PEFT方法；VTAB-1k结果在正文中提及但未在提供的摘录中展示。此外，GPS的参数量（0.77%）高于SSF（0.45%），作者未明确讨论这一代价的必然性。

## 方法谱系与知识库定位

GPS属于**参数高效微调（PEFT）**方法族中的**选择型（selection-based）**分支，直接继承自**SSF（Scaling & Shifting Features）**——此前PEFT的精度标杆。与SSF的"新增缩放平移参数"不同，GPS转向"从已有参数中选择"，实现了从"加法"到"选择"的范式迁移。

| 改动维度 | Baseline (SSF等) | GPS |
|:---|:---|:---|
| **架构 (architecture)** | 添加并行分支或特征变换 | 零修改，仅掩码现有权重 |
| **目标函数 (objective)** | 交叉熵 + 分类头 | 监督对比损失，无头 |
| **训练策略 (training_recipe)** | 全参数或新增参数更新 | 梯度驱动top-K选择 + 掩码更新 |
| **数据策划 (data_curation)** | 标准有标签微调 | 相同 |
| **推理方式 (inference)** | 需计算新增模块 | 掩码合并，零开销 |

**直接基线对比**：
- **vs Full Fine-tuning**：GPS用0.77%参数超越其精度（FGVC +2.67%），证明全量更新存在冗余
- **vs SSF**：同为选择型思路，GPS用梯度动态选择替代SSF的固定结构学习，精度+1.06%
- **vs Adapter/LoRA/VPT**：GPS无推理开销，无需存储额外模块参数
- **vs BitFit (Bias)**：GPS任务自适应选择，非固定选bias，FGVC +3.38%

**后续方向**：（1）将GPS与LoRA结合，在选择后的子网络上做低秩更新；（2）探索动态K值或层自适应预算分配；（3）扩展至NLP/多模态大模型，验证跨模态梯度选择的有效性。

**标签**：`modality=vision` · `paradigm=parameter-efficient-finetuning` · `scenario=transfer-learning` · `mechanism=gradient-based-selection` · `constraint=inference-zero-overhead`

