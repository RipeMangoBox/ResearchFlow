---
title: 'COMPASS: COntinual Multilingual PEFT with Adaptive Semantic Sampling'
type: paper
paper_level: B
venue: TMLR
year: 2026
paper_link: https://arxiv.org/abs/2604.20720
aliases:
- 多语言PEFT自适应语义采样持续学习框架
- COMPASS
- 核心直觉是：多语言负迁移的根源不在于语言之间的语言学距离
code_url: https://github.com/Arxiv-to-code/arxiv-260420720-compass-continual-multilingual-peft-with-adaptive-semantic-s
method: COMPASS
modalities:
- Text
paradigm: Reinforcement Learning
---

# COMPASS: COntinual Multilingual PEFT with Adaptive Semantic Sampling

[Paper](https://arxiv.org/abs/2604.20720) | [Code](https://github.com/Arxiv-to-code/arxiv-260420720-compass-continual-multilingual-peft-with-adaptive-semantic-s)

**Topics**: [[T__Continual_Learning]], [[T__Domain_Adaptation]], [[T__Text_Generation]] | **Method**: [[M__COMPASS]]

> [!tip] 核心洞察
> 核心直觉是：多语言负迁移的根源不在于语言之间的语言学距离，而在于训练数据与目标使用分布之间的语义覆盖缺口。通过在共享嵌入空间中识别并优先填补这些缺口，COMPASS将数据选择问题从「哪些语言相似」转化为「哪些语义簇欠表示」。这一转变使得跨语言迁移的收益最大化，同时避免了无差别数据混合引入的干扰噪声。PEFT的参数隔离特性则天然提供了对灾难性遗忘的内在抵抗力，使持续学习扩展（ECDA）得以以较小的回放缓冲区实现接近全量重训练的效果。

| 中文题名 | 多语言PEFT自适应语义采样持续学习框架 |
| 英文题名 | COMPASS: COntinual Multilingual PEFT with Adaptive Semantic Sampling |
| 会议/期刊 | Trans. Mach. Learn. Res. (2026) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.20720) · [Code](https://github.com/Arxiv-to-code/arxiv-260420720-compass-continual-multilingual-peft-with-adaptive-semantic-s) · [Project](https://arxiv.org/abs/2604.20720) |
| 主要任务 | 多语言大模型适配、低资源语言提升、持续学习、灾难性遗忘缓解 |
| 主要 baseline | 零样本推理、All-data mixing、LangRank、LangSim、EWC、Random Rehearsal、Full Retraining |

> [!abstract] 因为「多语言微调中跨语言干扰与灾难性遗忘导致低资源语言性能差且模型随时间过时」，作者在「标准PEFT+启发式数据选择」基础上改了「将数据选择转化为分布近似问题，引入自适应语义采样与弹性持续分布适应」，在「Global MMLU / MMLU-ProX」上取得「Phi-4-Mini +8.9pp、Llama-3.1-8B +6.1pp、Qwen2.5-7B +6.7pp，COMPASS-ECDA与Full Retraining持平(0.592 vs 0.590)」

- **关键性能1**: Phi-4-Mini-3.8B 在 Global MMLU 上相对零样本提升 **+8.9 percentage points**
- **关键性能2**: COMPASS-ECDA 的 5% 回放缓冲区将灾难性遗忘从 **-10.9pp 降至 -1.6pp**
- **关键性能3**: COMPASS-ECDA 与 Full Retraining 在 MMLU Overall 上差距仅 **0.002** (0.592 vs 0.590)

## 背景与动机

大型语言模型（LLM）在多语言场景下面临严重的性能不均衡：低资源语言（LRL）如印地语、斯瓦希里语的表现远低于英语。根源在于预训练数据分布极度偏向高资源语言，导致分词器对非拉丁文字过度分片、模型核心知识储备不足。例如，同一概念在泰语中可能被拆分为数倍于英语的token，信息密度骤降。

现有方法如何处理这一问题？**All-data mixing** 将所有语言数据无差别混合微调，但因引入大量跨语言干扰，实验中三种模型架构的 Global MMLU 均低于零样本基线——负迁移严重。**LangRank / LangSim** 基于语言学相似度（如语系、地理距离）选择辅助语言，虽优于随机采样，但依赖静态特征，无法捕捉语义层面的分布差异：两种形态差异大的语言可能在特定领域（如医学）共享大量概念。**全参数微调（FFT）** 计算成本高昂，且在中低资源语言上过拟合风险显著。

更深层的挑战是动态性：用户需求演变、新兴话题涌现、人口结构迁移均导致模型随时间逐渐过时（model staleness）。然而，现有方法缺乏持续适应机制；一旦在新分布上更新，灾难性遗忘便牺牲原有知识。

本文的核心动机正是：多语言负迁移的根源不在语言学距离，而在**语义覆盖缺口**——训练数据与目标使用分布之间的结构性错配。COMPASS 将数据选择从"哪些语言相似"重新定义为"哪些语义簇欠表示"，并以轻量PEFT适配器实现动态持续扩展。

## 核心创新

**核心洞察**：多语言负迁移的根源在于训练数据与目标使用分布之间的语义覆盖缺口，而非语言间的语言学距离；通过在共享嵌入空间中识别并优先填补这些缺口，可使跨语言迁移收益最大化，同时PEFT的参数隔离特性天然抵抗灾难性遗忘，从而使轻量持续学习扩展成为可能。

| 维度 | Baseline (LangRank/LangSim) | 本文 (COMPASS) |
|:---|:---|:---|
| 选择依据 | 静态语言学特征（语系、地理距离） | 动态语义分布近似（嵌入空间聚类） |
| 优化目标 | 最大化语言相似度 | 最小化目标-辅助分布的语义覆盖缺口 |
| 与PEFT关系 | 数据选择与训练紧耦合 | 完全解耦，对LoRA/DoRA等agnostic |
| 持续学习 | 无 | ECDA机制：JS散度触发 + EWC + DAR |
| 计算开销 | 需全量数据扫描或语言学知识库 | 预处理阶段完成选择，训练轻量适配器 |

## 整体框架


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/481709c7-8a66-44c8-81e4-b3b5db2de5b9/figures/Figure_1.png)
*Figure 1 (pipeline): An overview of COMPASS for multilingual adaptation.*



COMPASS 框架分为静态适配层（COMPASS）与动态持续层（COMPASS-ECDA），数据流如下：

**输入**：目标语言使用分布（由代理评估集近似）+ 多语言辅助数据池

**模块A：语义嵌入（Semantic Embedding）**
- 输入：原始文本数据
- 输出：共享语义空间中的稠密向量
- 角色：使用多语言嵌入模型消除语言壁垒，映射到统一表示空间

**模块B：分布感知聚类（Distribution-Aware Clustering）**
- 输入：目标语言语义向量
- 输出：最优K个语义簇（K=80–120，由轮廓系数/DBCV自动选择）
- 角色：识别目标使用分布中的语义结构

**模块C：缺口填补采样（Gap-Filling Sampling）**
- 输入：目标语义簇 + 辅助数据池语义向量
- 输出：精选辅助数据子集
- 角色：优先采样覆盖不足的语义簇，使训练分布逼近目标分布

**模块D：PEFT适配器训练（LoRA/DoRA Training）**
- 输入：采样后的训练数据
- 输出：语言特定轻量适配器
- 角色：与基础模型解耦，保持原模型冻结

**模块E：持续监测与触发（ECDA Monitor）**
- 输入：生产环境输入数据流
- 输出：JS散度信号 + 触发/不触发决策
- 角色：检测分布漂移，决定是否启动适配器更新

**模块F：弹性更新（ECDA Update）**
- 输入：新分布数据 + 历史锚点缓冲区
- 输出：更新后的适配器
- 角色：联合任务损失、EWC、DAR三项防止灾难性遗忘

```
[目标分布] ──→ [嵌入模型] ──→ [聚类(K-Means/HDBSCAN)] ──→ [缺口识别]
                                              ↑
[辅助数据池] ──→ [嵌入模型] ──→ [语义匹配] ──┘
                                              ↓
                                        [采样选择] ──→ [LoRA/DoRA训练]
                                                              ↑
[数据流监测] ──→ [JS散度计算] ──→ [触发判断] ──→ [ECDA更新: L_task + λL_EWC + βL_DAR]
                                                              ↓
                                                        [5%历史缓冲区回放]
```

## 核心模块与公式推导

### 模块 1: 分布感知语义采样（Distribution-Aware Semantic Sampling）

**直觉**：数据选择的本质是让训练分布 $P_{\text{train}}$ 逼近目标使用分布 $P_{\text{target}}$，而非假设语言相似即语义相似。

**Baseline 公式** (LangRank/LangSim): 基于语言学特征 $\phi_{\text{lang}}$ 的启发式排序
$$S_{\text{base}}(d_{\text{aux}}) = \text{sim}\big(\phi_{\text{lang}}(L_{\text{target}}), \phi_{\text{lang}}(L_{\text{aux}})\big)$$
符号: $\phi_{\text{lang}}$ = 静态语言学特征向量（语系、地理、书写系统等），$L$ = 语言标识

**变化点**：静态语言学相似度无法捕捉领域特定语义重叠；两种形态迥异的语言可能在"医学诊断"领域高度相关。COMPASS 将选择依据从语言层面下沉到语义簇层面。

**本文公式（推导）**:
$$\text{Step 1}: \quad z_i = f_{\text{emb}}(d_i), \quad f_{\text{emb}}: \mathcal{X} \rightarrow \mathbb{R}^d \quad \text{（多语言嵌入模型映射到共享空间）}$$
$$\text{Step 2}: \quad \mathcal{C}^* = \text{arg}\max_{\mathcal{C}, K} \text{SC}(\{z_i\}_{i \in \mathcal{D}_{\text{target}}}) \text{ or } \text{DBCV}(\{z_i\}, K) \quad \text{（自动选择最优聚类，实验中 } K \in [80,120]\text{）}$$
$$\text{Step 3}: \quad w(c_j) = \frac{1}{|c_j \cap \mathcal{D}_{\text{aux}}| + \epsilon} \cdot \mathbb{1}_{[|c_j \cap \mathcal{D}_{\text{aux}}| < \tau]} \quad \text{（欠表示簇获得更高采样权重）}$$
$$\text{最终}: \quad \mathcal{D}_{\text{selected}} = \text{TopK}_{d \in \mathcal{D}_{\text{aux}}} \sum_{j} w(c_j) \cdot \mathbb{1}_{[z_d \in c_j]} \quad \text{（按缺口填补优先级采样）}$$

**对应消融**：未提供显式消融表，但实验表明对希腊语、日语、韩语、越南语等无相关语言族的语言收益边际化，反向验证语义重叠假设的必要性。

---

### 模块 2: 弹性持续分布适应（ECDA: Elastic Continual Distribution Adaptation）

**直觉**：生产环境分布动态漂移，需轻量机制持续适配，同时以最小缓冲成本抵抗灾难性遗忘。

**Baseline 公式** (Naive Continual Fine-tuning): 标准任务损失
$$\mathcal{L}_{\text{naive}} = \mathbb{E}_{(x,y) \sim \mathcal{D}_{\text{new}}} [\text{ell}(f_\theta(x), y)]$$
符号: $\theta$ = 模型参数，$\text{ell}$ = 交叉熵损失，$\mathcal{D}_{\text{new}}$ = 新分布数据

**变化点**：朴素更新导致灾难性遗忘（实验显示-10.9pp）。EWC单独使用需存储Fisher信息矩阵且对参数变化惩罚过于均匀；纯回放缓冲区效率低下。ECDA 设计三项联合：任务损失保证新分布适应，EWC锁定重要参数，DAR以极小规模缓冲区（5%）锚定历史分布。

**本文公式（推导）**:
$$\text{Step 1}: \quad \text{触发条件: } D_{\text{JS}}(P_{\text{stream}} \| P_{\text{current}}) > \delta_{\text{threshold}} \quad \text{（JS散度监测分布漂移）}$$
$$\text{Step 2}: \quad \mathcal{L}_{\text{EWC}} = \sum_i \frac{F_i}{2} (\theta_i - \theta_{i,\text{old}}^*)^2 \quad \text{（Fisher信息加权，保护对旧任务重要的参数）}$$
$$\text{Step 3}: \quad \mathcal{L}_{\text{DAR}} = \mathbb{E}_{(x,y) \sim \mathcal{B}_{\text{replay}}} [\text{ell}(f_\theta(x), y)], \quad |\mathcal{B}_{\text{replay}}| = 0.05 \times |\mathcal{D}_{\text{historical}}| \quad \text{（5%分布锚点回放）}$$
$$\text{最终}: \quad \mathcal{L}_{\text{ECDA}} = \mathcal{L}_{\text{task}} + \lambda \cdot \mathcal{L}_{\text{EWC}} + \beta \cdot \mathcal{L}_{\text{DAR}}$$
其中超参数：$\lambda = 2$，$\beta = 0.1$

**对应消融**：5%缓冲区将遗忘从 **-10.9pp 降至 -1.6pp**；COMPASS-ECDA MMLU Overall **0.592** vs Full Retraining **0.590**，证明联合设计以极小成本逼近全量重训练效果。但注意：早期时间步 T1 的 PROX 指标（0.405）略低于 Naive Fine-tuning（0.406），适应速度稍慢。

## 实验与分析

| Method | Phi-4-Mini Global MMLU | Llama-3.1-8B Global MMLU | Qwen2.5-7B Global MMLU | MMLU-ProX (COMPASS-ECDA) |
|:---|:---|:---|:---|:---|
| Zero-shot | baseline | baseline | baseline | — |
| All-data mixing | < Zero-shot | < Zero-shot | < Zero-shot | — |
| LangRank / LangSim | 优于随机，中等效应 | 优于随机，中等效应 | 优于随机，中等效应 | — |
| **COMPASS (本文)** | **+8.9pp** | **+6.1pp** | **+6.7pp** | — |
| Naive Fine-tuning (持续) | — | — | — | 0.406 (T1 PROX) |
| EWC only | — | — | — | 遗忘显著 |
| Random Rehearsal | — | — | — | 劣于ECDA |
| Full Retraining | — | — | — | **0.590** |
| **COMPASS-ECDA (本文)** | — | — | — | **0.592** |

统计显著性：$p < 0.05$，置换检验 10,000 次迭代；vs LangRank/LangSim 的 Cohen's $d = 0.52$–$0.64$（中等效应量）。

**核心发现分析**：
- 支持核心 claim 的数字：COMPASS 在三种架构上均显著超越零样本，且 All-data mixing 全面劣于零样本——这直接验证了"语义缺口填补优于无差别混合"的洞察。
- 边际收益：对希腊语、日语、韩语、越南语等无相关语言族的语言，提升边际化，说明跨语言迁移确实依赖语义重叠，而非万能。
- 持续学习：COMPASS-ECDA 以 5% 缓冲区实现与 Full Retraining 持平（0.592 vs 0.590），成本差距悬殊但性能几乎无别，这是最具实用价值的数字。

**消融与公平性检查**：
- 最重要模块：DAR 回放机制（5%缓冲区 → -10.9pp 降至 -1.6pp），其次是 EWC 的参数锁定。
- 基线强度：缺少 DARE、TIES-Merging 等近期参数合并方法作为持续学习基线，可能低估对比难度。
- 计算/数据成本：数据选择在预处理完成，训练仅轻量 LoRA；但嵌入模型推理与聚类带来额外预处理开销。
- **失败/边界情况**：（1）不修改分词器，非拉丁文字过度分片天花板未破；（2）JS散度触发器是"性能盲的"，可能误触发或漏触发；（3）聚类质量用轮廓系数而非下游性能评估，代理指标与实际效果存在脱钩风险；（4）T1 早期适应速度略慢于 Naive Fine-tuning。

## 方法谱系与知识库定位

**方法家族**：多语言适配（Multilingual Adaptation）→ 数据为中心微调（Data-centric Fine-tuning）→ 参数高效迁移学习（PEFT）

**Parent method**：标准 LoRA/DoRA PEFT 微调流程。COMPASS 不改变基础模型架构、训练算法或损失函数形式，仅在预处理阶段插入数据选择模块——属于**数据层面的插件式改进**。

**变化插槽**：
- **data_curation**：从语言学启发式 → 分布感知语义采样（最大创新）
- **training_recipe**：静态微调 → ECDA 持续学习扩展（EWC + DAR 联合）
- **architecture**：无变化（保持 PEFT-agnostic）
- **inference**：无变化

**Direct baselines 与差异**：
- **LangRank / LangSim**：静态语言学相似度 → COMPASS 动态语义分布近似
- **EWC**：单独参数正则化 → COMPASS-ECDA 三项联合 + 触发机制
- **Random Rehearsal**：无策略回放 → COMPASS-ECDA 分布锚点定向回放
- **All-data mixing**：无选择全量混合 → COMPASS 缺口定向采样

**Follow-up 方向**：
1. 联合优化分词器与数据选择，打破非拉丁文字天花板；
2. 将性能感知信号（如验证集准确率）融入 JS 触发器，替代纯分布散度监测；
3. 探索 DARE/TIES-Merging 等参数合并方法与 COMPASS 数据选择的协同。

**知识库标签**：
- modality: text / multilingual
- paradigm: data-centric PEFT, continual learning
- scenario: low-resource language adaptation, production deployment with distribution drift
- mechanism: semantic clustering, distribution matching, elastic weight consolidation, distribution-anchored replay
- constraint: compute-efficient (PEFT), memory-efficient (5% replay buffer), preprocessing overhead (embedding + clustering)

