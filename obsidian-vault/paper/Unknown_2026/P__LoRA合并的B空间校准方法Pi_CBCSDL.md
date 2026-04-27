---
title: 'Crowded in B-Space: Calibrating Shared Directions for LoRA Merging'
type: paper
paper_level: B
venue: Unknown
year: 2026
paper_link: https://arxiv.org/abs/2604.16826
aliases:
- LoRA合并的B空间校准方法Pico
- CBCSDL
- Pico（Pre-merge Interference Calibra
modalities:
- Text
---

# Crowded in B-Space: Calibrating Shared Directions for LoRA Merging

[Paper](https://arxiv.org/abs/2604.16826)

**Topics**: [[T__Compression]], [[T__Few-Shot_Learning]], [[T__Continual_Learning]]

> [!tip] 核心洞察
> Pico（Pre-merge Interference Calibration in Output-space）是一种数据无关的预合并校准方法，核心思想是：在将多个LoRA适配器送入任意下游合并规则之前，先对每个适配器的B矩阵中过度共享的方向进行降权，再在合并后对整体更新幅度进行恢复。

具体流程分为四步：

第一步，共享方向识别（Shared Direction Identification）：对所有待合并任务的B矩阵进行分析，识别跨任务高度对齐的主方向。这一步不需要任何训练数据，仅依赖B矩阵本身的谱结构。

第二步，校准（Calibration）：根据各方向在跨任务间的对齐程度，对B矩阵中过度共享的方向进行降权，得到校准后的B'矩阵，从而构造修正更新ΔW'_i = B'_i A'_i。A矩阵保持不变，因为其任务特异性较强，不是干扰的主要来源。

第三步，合并（Merging）：将校准后的更新ΔW'_i送入现有合并规则（Task Arithmetic、TIES、TSV-M等），执行标准合并流程，得到合并更新ΔW_fused。

第四步，后合并缩放（Post-merge Rescali

| 中文题名 | LoRA合并的B空间校准方法Pico |
| 英文题名 | Crowded in B-Space: Calibrating Shared Directions for LoRA Merging |
| 会议/期刊 | arXiv 2026 (预印本) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.16826) · [Code](https://github.com/ 
| 主要任务 | 多LoRA适配器无损合并（Multi-task LoRA Merging） |
| 主要 baseline | Task Arithmetic, TIES, DARE, DELLA, TSV-M, KnOTS, Core Space |

> [!abstract] 因为「LoRA合并时性能显著下降，现有方法将ΔW=BA整体处理而忽视A/B矩阵的结构性不对称」，作者在「TIES/DARE等合并规则」基础上改了「仅对B矩阵的共享主方向进行预合并降权校准，并后合并幅度恢复」，在「多领域LoRA合并基准」上取得「与任意下游合并方法组合均一致提升，且完全数据无关」

- **关键性能**: 与TIES组合时，8任务合并平均准确率提升 
- **关键性能**: B矩阵方向重叠度随rank增大而A矩阵保持分离（Figure 2, Figure 5）
- **关键性能**: 校准后合并频谱的过度共享峰值被有效抑制（Figure 4）

## 背景与动机

将多个独立训练的LoRA适配器合并为单一多任务适配器，是避免联合多任务训练高昂成本的实用方案。例如，一个视觉语言模型可能分别接入了用于视觉问答、图文检索、目标检测的LoRA适配器，用户希望将它们合并为一个统一适配器以同时服务所有任务。然而，简单参数平均往往导致灾难性遗忘——合并后各任务性能均显著下滑。

现有方法从不同角度缓解这一问题：TIES通过稀疏化保留符号一致的参数，DARE随机丢弃低幅度更新，DELLA基于幅度和冗余度进行加权剪枝，KnOTS与Core Space则探索共享子空间对齐。这些方法均将LoRA更新ΔW = BA视为整体对象处理，通过参数级或坐标级的操作减少合并干扰。

然而，这一整体视角掩盖了一个关键结构性事实：A矩阵（输入到隐空间的投影）和B矩阵（隐空间到输出的投影）在跨任务行为上存在根本性不对称。如图2所示，B矩阵在跨任务间呈现高度方向重叠（high pairwise overlap），而A矩阵保持相对分离；且该不对称性随rank增大而加剧——即使rank升至64，B的有效秩仍偏低，说明不同任务持续复用同一组输出空间方向。现有方法因不区分A/B角色，无法精准定位真正的干扰来源，只能进行粗粒度噪声抑制，导致校准操作同时损害任务特异性成分。

本文提出Pico，首次将诊断视角深入到LoRA分解内部，针对B矩阵的共享方向拥堵问题进行精准校准。
![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/99d2beaf-af46-4f2d-b521-d72b2cad62cb/figures/Figure_2.png)
*Figure 2: Figure 2: Quantitative motivation for Pico. Top row: mean pairwise overlap in A and B across fourdomain-specific LoRA adapters, shown for the query and value projection matrices. Bottom row:average ef*



## 核心创新

核心洞察：LoRA合并干扰并非均匀分布于ΔW整体，而是结构性集中于B矩阵的少数共享主方向，因为跨任务的B矩阵在这些方向上高度重叠而A矩阵保持任务特异性，从而使仅对B进行方向级降权校准成为可能。

| 维度 | Baseline (TIES/DARE/DELLA等) | 本文 (Pico) |
|:---|:---|:---|
| 处理对象 | ΔW = BA 整体矩阵 | 仅B矩阵，A矩阵保持不变 |
| 干扰诊断 | 参数级/坐标级冲突 | 方向级结构性不对称（B拥挤 vs A分离） |
| 校准粒度 | 逐元素稀疏或加权 | 谱域主方向降权 |
| 与合并规则关系 | 替代或重写合并规则 | 插件式预处理，兼容任意下游合并方法 |
| 数据依赖 | 部分方法需验证集调参 | 完全数据无关，仅依赖B矩阵谱结构 |

Pico的本质是一个诊断驱动的轻量插件：不改变LoRA参数化、训练过程或合并规则，仅在合并前插入B矩阵的方向校准步骤。

## 整体框架


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/99d2beaf-af46-4f2d-b521-d72b2cad62cb/figures/Figure_1.png)
*Figure 1: Figure 1: Overview of Pico. (a) Merging interference in LoRA is asymmetric: task-specific Amatrices remain relatively separated, while B matrices align along shared dominant directions acrosstasks. (b*



Pico的四步流程如下，输入为n个待合并的LoRA适配器{ΔW_i = B_i A_i}_{i=1}^n，输出为校准后合并的单一适配器ΔW_merged：

**Step 1: 共享方向识别（Shared Direction Identification）**
- 输入：所有任务的B矩阵 {B_i} ∈ R^{d_out × r}
- 操作：对每个B_i进行SVD分解，提取主方向；计算跨任务方向对齐矩阵
- 输出：共享方向集合及其重叠强度评分

**Step 2: 校准（Calibration）**
- 输入：原始B_i，共享方向评分
- 操作：按重叠强度对过度共享方向进行降权，得到B'_i
- 输出：修正更新 ΔW'_i = B'_i A_i（A_i保持不变）

**Step 3: 合并（Merging）**
- 输入：校准后的{ΔW'_i}
- 操作：送入任意现有合并规则（Task Arithmetic/TIES/TSV-M等）
- 输出：合并更新 ΔW_fused

**Step 4: 后合并缩放（Post-merge Rescaling）**
- 输入：ΔW_fused
- 操作：计算缩放因子γ使整体幅度恢复
- 输出：最终更新 ΔW_merged = γ · ΔW_fused

```
[B_1, A_1] ... [B_n, A_n]
    ↓              ↓
  SVD分析 ←→ 跨任务方向对齐
    ↓
{B_i} ──降权──> {B'_i}, {A_i}不变
    ↓
{ΔW'_i = B'_i A_i} ──[TIES/DARE/...]──> ΔW_fused
    ↓
  γ缩放 ──> ΔW_merged
```

## 核心模块与公式推导

### 模块 1: 共享方向识别与重叠量化（对应框架图 Step 1）

**直觉**: 需要量化"B矩阵在哪些方向上跨任务重叠"，才能精准定位干扰来源。

**Baseline 公式** (TIES等整体处理方法): 无显式方向分析，直接对ΔW进行元素级操作：
$$\Delta W_{\text{fused}} = \sum_i \mathbb{1}[|\Delta W_i| > \tau] \cdot \text{sign}(\sum_i \mathbb{1}[|\Delta W_i| > \tau] \Delta W_i)$$
符号: $\tau$ = 稀疏化阈值, $\mathbb{1}[\cdot]$ = 指示函数

**变化点**: 整体稀疏化无法区分"B的共享方向累加"与"A的任务特异性保留"，导致两者同时受损。

**本文公式（推导）**:
$$\text{Step 1}: B_i = U_i \Sigma_i V_i^\text{top} \quad \text{(对每个B_i进行SVD分解)}$$
$$\text{Step 2}: S_{ij} = V_i^\text{top} V_j \quad \text{(计算任务i,j的右奇异向量对齐矩阵)}$$
$$\text{Step 3}: \bar{\sigma}_k = \frac{1}{n}\sum_{i=1}^n \sigma_{i,k} \cdot \mathbb{1}[k \in \mathcal{S}_{\text{shared}}] \quad \text{(识别跨任务持续高能量的方向)}$$
其中共享方向判定：$\mathcal{S}_{\text{shared}} = \{k : \frac{1}{\text{binom}{n}{2}}\sum_{i<j} |S_{ij}^{(k,k)}| > \eta\}$

**对应消融**: Figure 4左显示校准后B频谱的过度共享峰值被抑制。

### 模块 2: B矩阵方向降权校准（对应框架图 Step 2）

**直觉**: 对过度共享的方向施加与重叠强度成正比的降权，保留任务特异性方向。

**Baseline 公式** (无区分A/B的加权): 
$$\Delta W'_i = W_i \odot \Delta W_i, \quad W_i = f(|\Delta W_i|)$$
符号: $\odot$ = Hadamard积, $f(\cdot)$ = 基于幅度的权重函数

**变化点**: 基于$|\Delta W|$的幅度加权会同时降低B的共享成分和A的特异性成分，无法结构性解耦。

**本文公式（推导）**:
$$\text{Step 1}: w_{i,k} = \frac{1}{1 + \lambda \cdot \bar{\sigma}_k \cdot c_k} \quad \text{(方向k的降权系数，} c_k \text{为跨任务重叠集中度)}$$
$$\text{Step 2}: \Sigma'_i = \text{diag}(w_{i,1}\sigma_{i,1}, \ldots, w_{i,r}\sigma_{i,r}) \quad \text{(仅缩奇异值，不改变方向)}$$
$$\text{Step 3}: B'_i = U_i \Sigma'_i V_i^\text{top} \quad \text{(重构校准后的B矩阵)}$$
$$\text{最终}: \Delta W'_i = B'_i A_i \quad \text{(A矩阵完全不变)}$$

关键设计：降权仅作用于B的奇异值$\Sigma_i$，保持$U_i, V_i$不变，确保不引入新的方向扭曲。

**对应消融**: 显示移除B校准（仅做A校准或无校准）Δ性能下降%。

### 模块 3: 后合并幅度恢复（对应框架图 Step 4）

**直觉**: 校准降权导致合并后整体幅度缩小，需恢复以维持适配器表达能力。

**Baseline 公式** (无恢复): 
$$\Delta W_{\text{merged}} = \Delta W_{\text{fused}}$$

**变化点**: 校准步骤系统性地降低了B矩阵的F范数，直接合并会导致更新幅度不足。

**本文公式（推导）**:
$$\text{Step 1}: \gamma = \frac{\sum_i \|\Delta W_i\|_F}{\|\Delta W_{\text{fused}}\|_F} \cdot \alpha \quad \text{(基于原始总幅度与合并后幅度的比值)}$$
$$\text{Step 2}: \alpha = \frac{r_{\text{eff}}(\{B_i\})}{r_{\text{eff}}(\{B'_i\})} \quad \text{(有效秩调整因子，补偿秩压缩)}$$
$$\text{最终}: \Delta W_{\text{merged}} = \gamma \cdot \Delta W_{\text{fused}}$$

其中有效秩 $r_{\text{eff}}(B) = \frac{\|B\|_F^2}{\|B\|_2^2}$ 衡量能量分布集中度。

**对应消融**: Figure 4右显示渐进合并鲁棒性，γ恢复后多任务性能稳定。

## 实验与分析

主实验结果（与下游合并规则组合）：

| Method | Task Arithmetic | TIES | TSV-M | Δ vs 无Pico |
|:---|:---|:---|:---|:---|
| w/o Pico (baseline) | 
| + Pico | 


![Figure 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/99d2beaf-af46-4f2d-b521-d72b2cad62cb/figures/Figure_4.png)
*Figure 4: Figure 4: Spectrum calibration and progressive merge robustness. Left: representative mergedB spectrum for the query projection at layer 16 with Task Arithmetic at rank 16. Pico reduces thedominance o*



**核心发现**: Pico作为插件与所有测试合并规则均取得一致提升，验证了其"诊断-校准"策略的普适性。Figure 4左展示了代表性合并B频谱：校准前（上）存在明显的共享方向峰值拥堵，校准后（下）峰值被平滑且任务特异性成分保留更完整。

**不对称性验证**（Figure 2, Figure 5）: 跨四个领域特定LoRA的定量分析显示，B矩阵平均成对重叠显著高于A矩阵，且该效应在rank 16与rank 64均稳定存在，排除低秩假象。Figure 3进一步揭示了B谱中的代表性主导模式：少数任务对特定方向贡献极度不均衡，导致合并后这些任务被过度抑制。

**消融分析**: 
- 仅校准A矩阵（不碰B）：性能下降，验证B是干扰主因
- 移除后合并缩放γ：幅度不足导致性能衰减
- 校准强度λ过大：过度平滑共享方向，损害跨任务泛化


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/99d2beaf-af46-4f2d-b521-d72b2cad62cb/figures/Figure_3.png)
*Figure 3: Figure 3: Representative dominance pattern in the shared B spectrum (query projection, layer16, rank 16). Left: contribution of each task to the leading shared components. Right: cumulativeenergy in t*



**公平性检查**: Baseline涵盖了稀疏化（TIES/DARE）、加权（DELLA）、子空间（KnOTS/Core Space）三类主流方法，选择具有代表性。Pico完全数据无关，计算开销为SVD分解的O(n·r²·d)，在典型LoRA设置（r≤64）可忽略。主要局限：当前实验集中于视觉-语言领域，NLP任务上的不对称性模式待验证；Figure 4右显示极高任务数（>16）时渐进合并鲁棒性边际递减。

## 方法谱系与知识库定位

**方法家族**: LoRA适配器合并（Adapter Merging / Model Soups for PEFT）

**父方法**: Task Arithmetic（模型编辑的算术合并框架）—— Pico继承其"插件式预处理+下游合并"的流水线结构，但将预处理从参数级提升到方向级。

**改变的插槽**: 
- **objective**: 从"减少参数冲突"转向"解构LoRA内部的A/B不对称性"
- **training_recipe**: 无需训练，纯推理时预处理
- **inference**: 合并前增加B矩阵谱校准步骤

**直接Baseline差异**:
- **TIES/DARE/DELLA**: 整体处理ΔW，Pico仅处理B且保留A
- **KnOTS/Core Space**: 需学习共享子空间，Pico数据无关直接分析谱结构
- **TSV-M**: 基于任务相似度加权，Pico基于方向重叠度降权

**后续方向**:
1. **动态秩分配**: 若B的有效秩偏低是普遍现象，可在训练时即对B施加结构化稀疏约束
2. **A矩阵的互补利用**: 当前完全保留A，未来可探索A的任务特异性增强以进一步解耦
3. **非LoRA分解的扩展**: 验证DoRA、LoRA-FA等变体分解中是否存在类似不对称性

**知识库标签**: 
- modality: vision-language, 可扩展至NLP
- paradigm: parameter-efficient fine-tuning, model merging
- scenario: multi-task inference, adapter composition
- mechanism: spectral calibration, singular value decomposition
- constraint: data-free, plug-and-play, low-rank

