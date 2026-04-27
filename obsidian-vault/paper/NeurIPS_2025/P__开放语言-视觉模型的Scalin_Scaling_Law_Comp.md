---
title: Scaling Laws for Robust Comparison of Open Foundation Language-Vision Models and Datasets
type: paper
paper_level: C
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 开放语言-视觉模型的Scaling Law比较框架
- Scaling Law Comp
- Scaling Law Comparison Framework for Open Language-Vision Models
- Scaling law derivation enables robu
acceptance: Poster
cited_by: 4
code_url: https://github.com/LAION-AI/scaling-laws-openclip
method: Scaling Law Comparison Framework for Open Language-Vision Models
modalities:
- Image
- Text
paradigm:
- contrastive learning
- supervised
baselines:
- InternVL：视觉基础模型规_InternVL
---

# Scaling Laws for Robust Comparison of Open Foundation Language-Vision Models and Datasets

[Code](https://github.com/LAION-AI/scaling-laws-openclip)

**Topics**: [[T__Retrieval]], [[T__Classification]], [[T__Semantic_Segmentation]] | **Method**: [[M__Scaling_Law_Comparison_Framework_for_Open_Language-Vision_Models]] | **Datasets**: [[D__ImageNet-1K]], [[D__MS-COCO]] (其他: DataComp eval suite, Scaling law parameter, Dataset)

> [!tip] 核心洞察
> Scaling law derivation enables robust model and dataset comparison across scale spans, revealing that MaMMUT (contrastive + captioning) exhibits stronger scaling and better sample efficiency than standard CLIP (contrastive only) across multiple datasets and downstream tasks.

| 中文题名 | 开放语言-视觉模型的Scaling Law比较框架 |
| 英文题名 | Scaling Laws for Robust Comparison of Open Foundation Language-Vision Models and Datasets |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2506.04598) · [Code](https://github.com/LAION-AI/scaling-laws-openclip) · [Project](https://arxiv.org/abs/2506.04598) |
| 主要任务 | ImageNet-1k 零样本分类、MS-COCO 跨模态检索、ADE20K 语义分割 |
| 主要 baseline | CLIP (对比学习)、MaMMUT (对比+描述生成)、Reproducible scaling laws for contrastive language-image learning |

> [!abstract] 因为「现有方法仅在单点或少点比较模型，无法区分算法差异与算力差异」，作者在「CLIP/MaMMUT」基础上改了「以幂律Scaling Law为核心的系统比较框架（密集测量、跨尺度验证、恒定学习率调度）」，在「DataComp/Re-LAION/DFN数据集」上取得「MaMMUT在10^10-10^11 GFLOPS处超越CLIP，openMaMMUT-L/14达到80.3% ImageNet-1k零样本精度」

- **openMaMMUT-L/14** 在 ImageNet-1k 零样本分类上达到 **80.3%**，超过 DataComp 原始 CLIP ViT-L-14 的 **79.2%**
- MaMMUT 与 CLIP 的 scaling crossover 稳定出现在 **10^10 至 10^11 GFLOPS** 区间
- DataComp-1.4B 在 ImageNet-1k 分类上展现优于 Re-LAION-1.4B 的 scaling 行为

## 背景与动机

当前语言-视觉基础模型的比较存在一个根本性困境：研究者通常只在单一或少数参考规模上对比模型精度，却无法判断观察到的差异究竟源于算法改进、数据质量，还是简单的算力投入不同。例如，若模型A在ImageNet-1k上比模型B高1%，这是否意味着A的架构更优？还是说A只是消耗了更多训练FLOPs？现有文献缺乏控制算力变量的系统方法论。

现有方法如何处理这一问题？**CLIP** [7] 作为对比学习的奠基工作，仅使用InfoNCE损失在固定规模上训练，后续比较多在相同参数量或相同样本数下进行，但未建立性能与算力的连续关系。**MaMMUT** [17] 在CLIP基础上增加了描述生成损失，展示了联合训练的优势，但同样停留在单点精度比较。**Reproducible scaling laws for contrastive language-image learning** [10] 首次为CLIP推导了scaling law，但仅覆盖单一架构和有限数据集，未形成跨方法、跨数据集的通用比较框架。

这些工作的共同短板在于：**比较是静态的、单点的，而非动态的、跨尺度的**。当方法在不同计算区间表现可能反转时（如本文发现MaMMUT在小规模弱于CLIP、大规模超越CLIP），单点比较会给出误导性结论。此外，标准实践使用cosine学习率调度，其性能曲线非单调，给scaling law拟合带来噪声；数据重复问题也使得超过3B样本的测量不可靠。

本文提出：通过**密集测量+幂律拟合+跨尺度验证**的三步协议，将模型比较从"点估计"升级为"曲线推断"，从而在控制算力的前提下，可靠判断何种学习程序在何种规模区间更优。

## 核心创新

核心洞察：**性能-算力的幂律关系可以作为系统比较不同学习程序的"通用坐标系"**，因为同一函数形式下的参数差异（斜率、截距、不可约误差）能够解耦算法特性与规模效应，从而使跨架构、跨数据集的公平比较和规模外推成为可能。

| 维度 | Baseline（标准实践） | 本文 |
|:---|:---|:---|
| 比较方式 | 单点或少点精度对比 | 密集测量（≤3B样本无重复）+ 幂律拟合 |
| 学习率调度 | 仅cosine调度 | cosine与constant调度均验证，constant可降低计算成本 |
| 验证协议 | 无系统验证，直接报告拟合结果 | 留点交叉验证：拟合至C_threshold，外推验证MSE |
| 比较输出 | 谁高谁低的定性结论 | crossover点、置信区间、规模外推的定量预测 |
| 数据集比较 | 单点精度或人工筛选 | 通过scaling law参数（斜率、截距）系统比较DataComp/Re-LAION/DFN |

## 整体框架

本文方法的整体流程可概括为四个串联模块，形成从数据收集到比较结论的完整pipeline：

**模块1：密集模型训练（Dense model training）**
输入：开放数据集（DataComp-1.4B、DFN-1.4B、Re-LAION-1.4B），覆盖多种模型尺寸与样本数量；输出：多个计算点(C_i, L_i)上的性能测量。关键控制：仅使用≤3B唯一样本的模型进行拟合，避免数据重复混淆。

**模块2：Scaling law拟合（Power-law fitting）**
输入：错误率-算力测量对；输出：拟合参数(A_c, B_c, α_c, E_c)及置信区间。采用双饱和函数形式（含不可约误差项），并在对数空间进行优化以保证数值稳定性。

**模块3：跨尺度验证（Cross-scale validation）**
输入：拟合至C_threshold的scaling law；输出：对>C_threshold点的预测误差（MSE）及不确定性估计。通过系统性地改变C_threshold，验证外推可靠性。

**模块4：程序比较与预测（Procedure comparison）**
输入：经验证的多条scaling law曲线；输出：crossover点、效率排序、未见过规模下的性能预测及置信区间。

```
开放数据集 → [密集训练: 多规模×多架构] → (C_i, L_i)测量点
                                                    ↓
[幂律拟合: L = aC^b + E_∞] → 参数(A_c, B_c, α_c, E_c)
                                                    ↓
[交叉验证: 留C_threshold以上外推] → MSE_on_held-out
                                                    ↓
[程序比较: CLIP vs MaMMUT vs 数据集] → crossover + 预测
```

## 核心模块与公式推导

### 模块1：CLIP与MaMMUT的目标函数（框架图左侧：训练目标差异）

**直觉**：对比学习仅拉近匹配图文对、推远非匹配对；加入描述生成损失可强制模型生成细粒度语义，可能改变scaling特性。

**Baseline公式（CLIP）**：
$$L_{\text{CLIP}} = L_{\text{contrastive}}$$
符号：$L_{\text{contrastive}}$ 为InfoNCE对比损失，作用于图像编码器与文本编码器的输出相似度矩阵。

**变化点**：MaMMUT假设仅对比损失不足以充分利用文本生成能力，引入额外的captioning损失项。

**本文公式**：
$$L_{\text{MaMMUT}} = L_{\text{contrastive}} + \lambda L_{\text{cap}}, \quad \lambda = 1$$
符号：$L_{\text{cap}}$ 为图像到文本的描述生成损失（如自回归或prefix LM损失），$\lambda=1$ 表示两项等权重。该联合目标是后续scaling law比较中"程序差异"的唯一定量来源。

**对应消融**：Table 5a显示，在DataComp-1.4B上，MaMMUT在$>10^{10}$ GFLOPS区间错误率持续低于CLIP，验证了$\lambda L_{\text{cap}}$项对scaling行为的改变。

---

### 模块2：幂律Scaling Law拟合（框架图中部：核心方法论）

**直觉**：若性能随算力平滑变化，则可用低参数函数插值测量点，进而外推未训练规模。

**Baseline公式（Kaplan et al. [8] 语言模型scaling law）**：
$$L = aC^b$$
符号：$C$为训练计算量（FLOPs），$L$为错误率，$a$为尺度系数，$b$为scaling指数（负值，绝对值越大scaling越陡）。

**变化点**：纯幂律假设错误率可无限趋近于零，但实际任务存在不可约误差（如标注噪声、任务固有模糊性）；此外，vision-language任务可能呈现不同于语言模型的饱和特性。

**本文公式（推导）**：
$$\text{Step 1}: L = aC^b + E_{\infty} \quad \text{加入不可约误差项以反映任务天花板}$$
$$\text{Step 2}: \log(L - E_{\infty}) = \log a + b \cdot \log C \quad \text{对数变换以线性化，便于数值稳定拟合}$$
$$\text{Step 3}: \hat{L}(C_{\text{new}}) = \hat{a} \cdot C_{\text{new}}^{\hat{b}} + \hat{E}_{\infty} \quad \text{参数估计后用于规模外推预测}$$
$$\text{最终}: L(C) = A_c C^{-\alpha_c} + E_c \quad \text{（等价重参数化形式，见Eq. 1）}$$
符号：$E_{\infty}$（或$E_c$）为不可约误差，$\alpha_c = -b > 0$ 为scaling指数，$A_c$ 为任务-架构相关的幅度参数。

**对比形式（Eq. 3，无不可约误差）**：
$$L = aC^b$$
作者明确比较两种形式，发现含$E_{\infty}$的双饱和形式在特定$C_{\text{threshold}}$下预测行为不同，需通过留点验证选择。

**对应消融**：Fig. 14及相关分析显示，不同函数形式在extrapolation区间的MSE差异显著，验证了$E_{\infty}$项的必要性。

---

### 模块3：跨尺度验证协议（框架图右侧：质量保证）

**直觉**：若scaling law真正"理解"了性能-算力关系，则应能从部分数据预测剩余数据。

**Baseline做法**：无系统验证，拟合所有可用点后直接报告外推。

**变化点**：引入计算预算阈值$C_{\text{threshold}}$，刻意隐瞒大于该值的数据点，检验拟合曲线的预测能力。

**本文公式**：
$$\text{Step 1}: \mathcal{D}_{\text{fit}} = \{(C_i, L_i) : C_i \leq C_{\text{threshold}}\} \quad \text{仅使用阈值以下数据拟合}$$
$$\text{Step 2}: \hat{\theta} = \text{arg}\min_{\theta} \sum_{(C_i,L_i)\in\mathcal{D}_{\text{fit}}} (L_i - f_{\theta}(C_i))^2 \quad \text{非线性最小二乘估计参数}$$
$$\text{Step 3}: \text{MSE}_{\text{hold-out}} = \frac{1}{|\mathcal{D}_{\text{test}}|}\sum_{(C_j,L_j)\in\mathcal{D}_{\text{test}}} (L_j - f_{\hat{\theta}}(C_j))^2, \quad \mathcal{D}_{\text{test}} = \{(C_j,L_j) : C_j > C_{\text{threshold}}\}$$
**最终**：通过扫描多个$C_{\text{threshold}}$值，构建MSE随extrapolation距离变化的诊断曲线，量化预测置信度。

**对应消融**：Section 3.1报告，该验证协议成功识别了数据重复导致的拟合偏差（>3B样本时MSE显著上升），从而确立了≤3B样本的拟合边界。

## 实验与分析

本文在三个开放数据集（DataComp-1.4B、DFN-1.4B、Re-LAION-1.4B）上训练了多种规模的CLIP和MaMMUT模型，覆盖从$10^9$到$2.59\times10^{12}$ GFLOPS的计算区间，并在ImageNet-1k零样本分类、MS-COCO检索、ADE20K语义分割（微调后）等任务上评估。

**核心结果**：在DataComp-1.4B上，MaMMUT展现出优于CLIP的scaling行为，两者crossover稳定出现在$10^{10}$至$10^{11}$ GFLOPS之间——即约10B到100B FLOPs的计算区间。这意味着在小规模（如边缘设备预算）CLIP更优，但在大规模训练下MaMMUT的联合训练目标带来更陡峭的scaling曲线。具体而言，openMaMMUT-L/14在12.8B样本训练后达到**80.3%** ImageNet-1k零样本精度，超过DataComp原始CLIP ViT-L-14报告的**79.2%**。值得注意的是，后者使用了约9×数据重复，而本文预测基于低重复设定，表明MaMMUT在更健康的训练条件下实现了更高精度。

**数据集比较**：通过scaling law参数比较，DataComp-1.4B在ImageNet-1k分类上展现优于Re-LAION-1.4B的scaling斜率，即同等算力投入下DataComp的错误率下降更快；但在MS-COCO检索任务上两者表现接近，说明数据集优势具有任务依赖性。

**消融实验**：恒定学习率调度的验证是重要发现——Fig. 14显示constant LR足以支持valid scaling law derivation，这意味着未来大规模scaling研究可省去cosine调度的网格搜索成本。此外，限制拟合数据至≤3B样本的决策经留点验证支持：超过此边界时数据重复导致幂律假设失效，MSE显著恶化。

**公平性审视**：本文比较聚焦于CLIP与MaMMUT两种架构，对SigLIP 2 [14]、CoCa [16]、InternVL [13]等近期方法仅作简要提及而未纳入完整scaling law框架。此外，MaMMUT在小规模弱于CLIP的发现提示"最优方法"依赖于部署预算，不存在 universally better 的算法。作者坦诚披露：scaling law拟合限于≤3B样本，更大规模（如12.8B样本的openMaMMUT-L/14）属于extrapolation预测而非直接测量验证。

## 方法谱系与知识库定位

本文属于**Scaling Law方法论家族**，直接继承自 **"Reproducible scaling laws for contrastive language-image learning"** [10]（prior CLIP scaling law工作），并向上追溯至Kaplan et al. [8]的语言模型scaling laws及Chinchilla [9]的算力最优训练框架。

**改变的slot**：
- **training_recipe**：从单点比较替换为密集测量+幂律拟合+留点验证的完整协议
- **objective**：将scaling law从描述性工具升级为比较性工具，用于判定crossover和效率排序
- **data_pipeline**：引入多数据集（DataComp/DFN/Re-LAION）控制和≤3B样本的重复约束

**直接baseline及差异**：
- **CLIP** [7]：纯对比学习，本文用其作为scaling law比较的基准程序
- **MaMMUT** [17]：对比+描述生成联合训练，本文量化证明其scaling优势的出现规模
- **Reproducible scaling laws for CLIP** [10]：仅覆盖CLIP单一架构，本文扩展至多架构、多数据集、多任务

**后续方向**：
1. 将框架扩展至SigLIP、CoCa等更近期架构，验证scaling law比较的普适性
2. 探索任务间scaling law的迁移规律（如ImageNet scaling能否预测下游检测/分割性能）
3. 开发自适应$C_{\text{threshold}}$选择策略，减少留点验证的计算开销

**标签**：modality=image+text | paradigm=supervised pre-training | scenario=foundation model comparison | mechanism=power-law scaling & cross-scale validation | constraint=open data, ≤3B sample fitting boundary, compute-controlled
## 引用网络

### 直接 baseline（本文基于）

- [[P__InternVL：视觉基础模型规_InternVL]] _(实验对比)_: Large vision-language model; likely appears in experimental comparisons as a str

