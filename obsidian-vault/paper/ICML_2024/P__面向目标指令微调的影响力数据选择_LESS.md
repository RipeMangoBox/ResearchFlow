---
title: 'LESS: Selecting Influential Data for Targeted Instruction Tuning'
type: paper
paper_level: C
venue: ICML
year: 2024
paper_link: null
aliases:
- 面向目标指令微调的影响力数据选择LESS
- LESS
- LESS (Low-rank gradiEnt Similarity
acceptance: Poster
cited_by: 454
code_url: https://github.com/princeton-nlp/LESS
method: LESS
modalities:
- Text
paradigm: supervised
followups:
- 基于单次不确定性估计的高效RL数_UFO-RL
---

# LESS: Selecting Influential Data for Targeted Instruction Tuning

[Code](https://github.com/princeton-nlp/LESS)

**Topics**: [[T__Text_Generation]], [[T__Reasoning]] | **Method**: [[M__LESS]] | **Datasets**: MMLU, TYDIQA, BBH, Transfer: LLAMA-2-13B with LLAMA-2-7B selection, Transfer: MISTRAL-7B with LLAMA-2-7B selection

> [!tip] 核心洞察
> LESS (Low-rank gradiEnt Similarity Search) enables selecting a small, influential subset of instruction tuning data that can outperform training on the full dataset across diverse downstream tasks, with strong transferability across model sizes and families.

| 中文题名 | 面向目标指令微调的影响力数据选择LESS |
| 英文题名 | LESS: Selecting Influential Data for Targeted Instruction Tuning |
| 会议/期刊 | ICML 2024 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2402.04333) · [Code](https://github.com/princeton-nlp/LESS) · [DOI](https://doi.org/10.48550/arxiv.2402.04333) |
| 主要任务 | Instruction Tuning Data Selection, Text Generation, Reasoning |
| 主要 baseline | Random Selection, BM25, DSIR, RDS (Representation-based Data Selection), InfSGD |

> [!abstract]
> 因为「大规模指令微调数据集中存在大量与目标能力无关的冗余样本，全量训练效率低下且可能损害特定能力」，作者在「Influence Functions (Grosse et al., 2023)」基础上改了「Adam优化器感知的影响估计(InfAdam)、梯度归一化与随机投影降维，并引入warmup训练构建可复用的梯度数据存储」，在「MMLU/TYDIQA/BBH」上取得「仅用5%数据即超越全量训练」的结果。

- **MMLU**: LESS 50.2 vs. Full dataset 49.8 (+0.4), vs. Random 5% 46.5 (+3.7)
- **TYDIQA**: LESS 56.2 vs. Full dataset 55.1 (+1.1), vs. Random 5% 52.7 (+3.5)
- **跨模型迁移**: LLAMA-2-7B 选择的数据用于 LLAMA-2-13B，平均 52.8 vs. Full dataset 52.1 (+0.7)

## 背景与动机

现代大语言模型的指令微调（Instruction Tuning）通常依赖数百万条多样化的指令-回复对，例如 UltraChat 和 Dolly 等数据集。然而，实际应用场景往往需要模型具备特定的目标能力——如多步推理或跨语言问答——而非泛泛的通用能力。一个具体的例子是：开发者希望提升模型在 TYDIQA（多语言信息检索问答）上的表现，但面对数百万条混合指令数据，无法判断哪些样本真正有助于这一特定能力。

现有方法从三个角度尝试解决这一问题：
- **BM25**（Robertson & Zaragoza, 2009）：基于 TF-IDF 词频统计匹配训练样本与目标任务的表面文本相似性，计算高效但完全忽略语义和深层能力关联。
- **DSIR**（Xie et al., 2023）：使用 n-gram 特征通过重要性重采样加权候选数据，考虑了分布匹配但仍局限于表层语言模式，无法捕捉推理等高级能力的相关性。
- **RDS**（Sener & Savarese, 2018; Geifman & El-Yaniv, 2017）：利用模型最后一层隐藏状态（最后一 token 的 2048 维表示）计算样本间相似性，虽引入模型语义但仅捕获前向传播的最终表示，而非训练过程中真实的参数更新影响。

这些方法的根本局限在于：**它们无法衡量单个训练样本对目标能力的真实「影响力」**。具体而言，BM25 和 DSIR 停留在表面形式匹配；RDS 的表示相似性与实际训练梯度更新无直接关联；而基于 Influence Functions 的 **InfSGD**（Grosse et al., 2023）虽能估计样本影响力，但其 Hessian 逆矩阵假设与 Adam 优化器的实际更新动态不符，且未处理指令微调中回复长度差异导致的梯度范数偏差问题（长回答的梯度范数系统性偏小）。



Figure 3 的实证分析揭示了这一关键现象：梯度范数与完成长度呈显著负相关，意味着未归一化的梯度会系统性低估长指令样本的重要性。本文的核心动机即源于此——如何将 influence estimation 适配到 Adam 优化器和变长指令数据的实际场景，从而精准识别对目标能力最具影响力的训练样本。

## 核心创新

核心洞察：**Adam 优化器的对角预条件矩阵 Γ 可替代 Hessian 逆矩阵进行影响估计**，因为 Γ 编码了 Adam 实际的一阶/二阶矩信息，从而使 influence-based data selection 首次适配到 Adam 训练的 LLM 场景；同时梯度归一化消除了变长指令的长度偏差，随机投影使高维梯度可存储、可搜索，最终让「小模型选择数据、大模型受益」的跨尺度迁移成为可能。

| 维度 | Baseline (InfSGD / RDS / BM25) | 本文 (LESS) |
|:---|:---|:---|
| **优化器适配** | Hessian 逆矩阵假设 SGD 动态；与 Adam 实际更新不符 | InfAdam: 用 Adam 对角预条件矩阵 Γ 替代 H⁻¹ |
| **长度偏差处理** | 无；梯度范数与完成长度负相关导致系统性偏差 | 梯度 L2 归一化，确保长短样本公平比较 |
| **特征维度** | RDS 用 2048 维隐藏状态；InfSGD 用完整高维梯度 | 随机投影至 d=8192 维，可复用的梯度数据存储 |
| **训练流程** | 直接在全量或随机子集上训练 | Warmup 阶段：LoRA 训练 5% 随机数据构建梯度存储 |
| **跨模型迁移** | 无显式支持 | LLAMA-2-7B 选择的数据可直接用于 13B 及 MISTRAL-7B |

## 整体框架


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/565ab71a-1d23-4ac1-ab58-b5cc14fbe302/figures/Figure_1.png)
*Figure 1 (pipeline): Overview of LESS. In Step 1, we warm up the Adam gradient features. In Step 2, we compute gradient features. In Step 3, we select data. In Step 4, we train.*



LESS 采用两阶段流水线，整体数据流如下：

**阶段一：Warmup 与梯度数据存储构建**
1. **输入**: 完整指令数据集 D，随机采样 5% 作为 Dwarmup
2. **LoRA Warmup 训练**: 在 Dwarmup 上训练 N=4 个 epoch，保存 N 个 checkpoints
3. **梯度特征计算**: 对每个 checkpoint，计算完整数据集 D 中每条样本的梯度，经归一化后通过随机高斯矩阵投影至 d=8192 维
4. **梯度数据存储 (Gradient Datastore)**: 聚合 N 个 checkpoint 的投影梯度，形成可复用的低维特征库

**阶段二：目标特定选择与训练**
5. **目标验证集 Dval**: 用户提供的少量（few-shot）体现目标能力的样本
6. **InfAdam 评分**: 计算每条训练样本与 Dval 中所有样本的 Adam 感知影响分数
7. **Top-5% 选择**: 按总分排序，选取最高 5% 构成 Dtrain
8. **最终训练**: 在 Dtrain 上对目标模型 MT 进行 LoRA 微调

```
Dwarmup (5% random) ──► LoRA Training (N epochs) ──► N Checkpoints
                                                          │
Full Dataset D ◄─────────────────────────────────────────┘
       │
       ▼
Gradient Computation (per checkpoint)
       │
       ▼
Normalization + Random Projection (d=8192)
       │
       ▼
Gradient Datastore ◄────┐
                        │
Dval (few-shot target) ─┴─► InfAdam Scoring ──► Top-5% Selection ──► Dtrain ──► Final LoRA Training ──► MT
```

关键设计：梯度数据存储一旦构建，可服务于多个不同目标任务的快速选择（每次选择 <1 分钟），实现「一次构建，多次复用」。

## 核心模块与公式推导

### 模块 1: InfAdam — Adam 感知影响估计（对应框架图 Step 6）

**直觉**: 传统 influence function 假设 SGD 更新，但现代 LLM 几乎都用 Adam 训练；Adam 的对角预条件矩阵 Γ 比 Hessian 逆矩阵更能反映真实的参数更新动态。

**Baseline 公式** (InfSGD, Grosse et al., 2023):
$$\text{Inf}_{\text{SGD}}(z, z') = \nabla_\theta L(z; \theta)^\text{top} H^{-1} \nabla_\theta L(z'; \theta)$$
符号: $z$ = 训练样本, $z'$ = 验证样本, $H^{-1}$ = Hessian 逆矩阵, $\theta$ = 模型参数, $L$ = 损失函数。

**变化点**: InfSGD 的 $H^{-1}$ 在 Adam 场景下既不准确（Adam 不沿 Hessian 方向更新）又计算昂贵（需共轭梯度近似）。此外，指令微调的变长回复导致梯度范数与长度负相关，直接比较会系统性偏好短样本。

**本文公式（推导）**:
$$\text{Step 1}: \Gamma = \text{diag}\left(\frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}\right) \quad \text{提取 Adam 一阶/二阶矩估计构成对角预条件矩阵}$$
$$\text{Step 2}: \text{Inf}_{\text{Adam}}(z, z') = \nabla_\theta L(z; \theta)^\text{top} \Gamma \nabla_\theta L(z'; \theta) \quad \text{用 } \Gamma \text{ 替代 } H^{-1}$$
$$\text{最终}: \text{Inf}_{\text{Adam}}(z, z') = \tilde{g}_z^\text{top} \Gamma \tilde{g}_{z'} \text{（结合归一化梯度，见模块 2）}$$

**对应消融**: Table 5 显示，使用预训练模型（无 Adam 状态，退化为 InfSGD）相比 warmup 训练后模型，平均性能下降约 -3.1 个百分点，验证了 Adam 感知设计的必要性。

### 模块 2: 梯度归一化 — 消除长度偏差（对应框架图 Step 3 中 Normalization）

**直觉**: Figure 3 实证显示梯度范数 $\|g_z\|_2$ 与样本的完成长度呈显著负相关——长回答的梯度反而更小，若不校正将导致选择过程系统性忽略复杂长指令样本。

**Baseline 公式** (vanilla gradient):
$$g_z = \nabla_\theta L(z; \theta) \in \mathbb{R}^p \quad \text{（直接使用原始梯度，无长度校正）}$$

**变化点**: 原始梯度范数受序列长度影响，不同长度样本的梯度幅度不可比；需要无量纲化的归一化操作。

**本文公式（推导）**:
$$\text{Step 1}: \tilde{g}_z = \frac{g_z}{\|g_z\|_2} \cdot \mathbb{1}[\|g_z\|_2 > \tau] \quad \text{L2 归一化，过滤极小范数异常值}$$
$$\text{Step 2}: \phi(z) = R \cdot \tilde{g}_z \in \mathbb{R}^d, \quad R_{ij} \sim \mathcal{N}(0, 1/\sqrt{d}) \quad \text{Johnson-Lindenstrauss 随机投影降维}$$
$$\text{最终}: \tilde{g}_z \text{ 为单位方向向量，仅保留梯度方向信息，消除长度引起的幅度偏差}$$

符号: $\tau$ = 范数阈值（过滤数值不稳定样本）, $R \in \mathbb{R}^{d \times p}$ = 随机高斯投影矩阵, $d=8192$ = 投影后维度, $p$ = 原始参数维度（数百万至数十亿）。

**对应消融**: Table 6 显示，使用更小投影维度（d=1024/2048/4096）仍优于随机选择，但 d=8192 效果最佳；同时脚注提及去掉归一化后性能下降，证实长度偏差校正的必要性。

### 模块 3: 数据选择评分与高效检索（对应框架图 Step 6-7）

**直觉**: 影响分数需在完整数据集上聚合，但高维内积计算不可行；利用随机投影的近似保内积性质，将影响计算转化为低维向量相似性搜索。

**Baseline 公式** (BM25 / DSIR / RDS):
$$s_{\text{BM25}}(z) = \text{TF-IDF}(z) \cdot \text{TF-IDF}(D_{\text{val}}) \quad \text{或} \quad s_{\text{RDS}}(z) = h_z^\text{top} h_{D_{\text{val}}}$$

**变化点**: BM25/DSIR 无模型训练信号；RDS 仅用最终隐藏状态，无梯度更新信息；均无法反映样本对目标能力的真实训练影响力。

**本文公式（推导）**:
$$\text{Step 1}: \text{Inf}_{\text{Adam}}(z, z') \approx \phi(z)^\text{top} \phi(z') \quad \text{JL 投影近似保持内积结构}$$
$$\text{Step 2}: s(z) = \sum_{z' \in D_{\text{val}}} \phi(z)^\text{top} \phi(z') = \phi(z)^\text{top} \left(\sum_{z' \in D_{\text{val}}} \phi(z')\right) \quad \text{聚合所有验证样本的影响}$$
$$\text{最终}: D_{\text{train}} = \text{TopK}_{z \in D}(s(z), k = 0.05|D|) \quad \text{选取 top 5\% 最高影响力样本}$$

**效率保证**: 投影后内积计算复杂度为 $O(|D| \cdot |D_{\text{val}}| \cdot d)$，实际运行时间 <1 分钟（Table 4）；存储开销 $O(|D| \cdot N \cdot d) = 17.7$ GB。

**对应消融**: Table 5 显示，减少 checkpoint 数量 N=4→1，平均性能下降 -1.5 个百分点，但仍优于随机选择，说明多 epoch 聚合可平滑噪声但非绝对必要。

## 实验与分析


![Table 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/565ab71a-1d23-4ac1-ab58-b5cc14fbe302/figures/Table_2.png)
*Table 2 (result): Results of LESS finetuned with superior model's gradient features and LESS-T finetuned with Llama-2-7B gradient features.*



本文在三个代表性 benchmark 上评估 LESS 的有效性：MMLU（57 学科知识问答）、TYDIQA（多语言信息检索问答）和 BBH（大基准难题，侧重多步推理）。主实验（Table 2）的核心发现是：使用 LLAMA-2-7B 进行数据选择后，仅用 5% 的选中数据训练，即可在多个任务上超越全量数据训练。具体而言，MMLU 上 LESS 达到 50.2，相比全量训练的 49.8 提升 +0.4，相比随机 5% 的 46.5 提升 +3.7；TYDIQA 上 LESS 56.2 超越全量 55.1（+1.1）和随机 52.7（+3.5）；BBH 上 LESS 41.5 超越全量 40.8（+0.7）和随机 38.9（+2.6）。这一模式表明，LESS 选择的数据不仅更高效，而且质量更优——去除冗余样本反而减少了干扰目标能力的噪声。


![Table 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/565ab71a-1d23-4ac1-ab58-b5cc14fbe302/figures/Table_3.png)
*Table 3 (comparison): Comparison of LESS finetuned with BM25, DSIR, and RVS baselines.*



与现有数据选择方法的对比（Table 3）显示，LESS 全面优于 BM25、DSIR 和 RDS。BM25 作为基于词频的强基线，在 MMLU 上 48.3、TYDIQA 上 53.4 均显著低于 LESS；DSIR 的 n-gram 方法（MMLU 47.8, TYDIQA 53.1）因同样受限于表面模式而表现不佳；RDS 利用模型表示（MMLU 48.6, TYDIQA 54.2）虽优于前两者，但仍不及基于真实梯度影响的 LESS。这说明**梯度方向信息比隐藏状态表示更能预测样本的训练价值**。


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/565ab71a-1d23-4ac1-ab58-b5cc14fbe302/figures/Figure_2.png)
*Figure 2 (result): Average performance of LESS with increasing proportions of data selected from 4 representative datasets.*



跨模型迁移是 LESS 的另一亮点。Table 2 中 LESS-T 结果显示：用 LLAMA-2-7B 选择的数据训练 LLAMA-2-13B，平均性能 52.8 超越该模型全量训练的 52.1（+0.7）；更显著的是，同一套 7B 选择数据用于 MISTRAL-7B，平均 55.3 超越其全量训练 54.6（+0.7）。这证实了梯度影响信号在不同架构间的可迁移性，为小模型辅助大模型训练提供了可行路径。


![Table 5](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/565ab71a-1d23-4ac1-ab58-b5cc14fbe302/figures/Table_5.png)
*Table 5 (ablation): Number of checkpoints (N) used for selection and average performance.*



消融实验揭示了关键设计决策的敏感度。Table 5 中，去掉 warmup 阶段（直接使用预训练 LLAMA-2-7B 或 LLAMA-2-7B-Chat）导致平均性能骤降约 -3.1，证明 warmup 训练对构建有意义的梯度空间至关重要。Table 6 显示投影维度从 d=8192 降至 4096/2048/1024 时性能逐步下降，但即使 d=1024 仍优于随机选择，说明方法对降维有一定鲁棒性。同时，减少 checkpoint 数量 N=4→1 造成 -1.5 的性能损失，表明多 epoch 聚合有助于稳定梯度估计。

公平性检查：主实验的比较对象主要为同预算（5%）的随机选择和传统方法，但未与部分相关工作中提到的强基线（如 Alpagasus、MODS、DSDM、Simfluence）进行直接对比。此外，warmup 阶段的 54 GPU 小时（A100）开销未被计入部分基线的有效比较中，实际总成本优势需结合具体场景评估。作者披露的限制包括：梯度存储随数据集规模增长、目标验证集质量影响选择效果、以及 warmup 超参数（5% 比例、N=4 epochs）缺乏系统性调优。

## 方法谱系与知识库定位

**方法家族**: Influence Functions → InfSGD (Grosse et al., 2023) → **LESS**

**父方法**: Grosse et al. (2023) 将经典 influence function 应用于 LLM，提出 InfSGD 框架。LESS 在此基础上进行四项关键改造：
- **credit_assignment**: Hessian 逆矩阵 → Adam 对角预条件矩阵 Γ（InfAdam）
- **data_pipeline**: 完整梯度 → 归一化 + 随机投影至 8192 维的梯度数据存储
- **training_recipe**: 直接训练 → 先 LoRA warmup 5% 随机数据构建存储，再目标选择
- **inference_strategy**: 新增高效相似性搜索，实现多任务复用

**直接基线差异**:
- **vs. BM25/DSIR**: 从表面文本/n-gram 匹配 → 深度梯度影响估计
- **vs. RDS**: 从最终隐藏状态表示 → 训练梯度方向（更接近参数更新因果链）
- **vs. InfSGD**: 从 SGD 假设 → Adam 实际优化动态；新增长度偏差校正与降维存储

**后续方向**:
1. **Warmup 成本优化**: 探索无 warmup 或更轻量预热（如 1% 数据、1 epoch）的可行性，降低前期 54 GPU 小时开销
2. **多目标组合选择**: 当前 Dval 为单任务 few-shot，扩展至多能力联合优化的 Pareto 选择
3. **与课程学习结合**: 将 LESS 的选择分数作为课程排序信号，实现渐进式难度递增的训练调度

**标签**: [modality: text] [paradigm: supervised fine-tuning] [scenario: instruction tuning / targeted capability development] [mechanism: influence functions / gradient similarity / Adam optimization] [constraint: compute-efficient data selection / cross-model transfer]

## 引用网络

### 后续工作（建立在本文之上）

- [[P__基于单次不确定性估计的高效RL数_UFO-RL]]: Direct precursor on data selection for instruction tuning; 'LESS' is closely rel

