---
title: Automatic Auxiliary Task Selection and Adaptive Weighting Boost Molecular Property Prediction
type: paper
paper_level: A
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- LLM自动检索与梯度对齐的分子辅助学习
- AUTAUT
- AUTAUT automatically retrieves auxi
acceptance: Poster
code_url: https://github.com/zhiqiangzhongddu/AUTAUT
method: AUTAUT
modalities:
- graph
- Text
paradigm: supervised
---

# Automatic Auxiliary Task Selection and Adaptive Weighting Boost Molecular Property Prediction

[Code](https://github.com/zhiqiangzhongddu/AUTAUT)

**Topics**: [[T__Few-Shot_Learning]], [[T__Medical_Imaging]] | **Method**: [[M__AUTAUT]] | **Datasets**: molecular property prediction

> [!tip] 核心洞察
> AUTAUT automatically retrieves auxiliary tasks using large language models and adaptively weights them via gradient alignment, outperforming existing auxiliary task-based and molecular property prediction approaches.

| 中文题名 | LLM自动检索与梯度对齐的分子辅助学习 |
| 英文题名 | Automatic Auxiliary Task Selection and Adaptive Weighting Boost Molecular Property Prediction |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [Code](https://github.com/zhiqiangzhongddu/AUTAUT) |
| 主要任务 | 分子性质预测（Molecular Property Prediction） |
| 主要 baseline | GradNorm, Forkmerge, Learning to group auxiliary datasets for molecule, Geometry-enhanced molecular representation learning |

> [!abstract] 因为「分子性质预测标注数据稀缺且手动设计辅助任务需要大量领域知识」，作者在「Learning to group auxiliary datasets for molecule」基础上改了「LLM自动检索辅助任务 + 梯度对齐动态加权」，在「分子性质预测基准」上取得「超越10种辅助任务方法和18种分子性质预测模型」

- 在多个分子性质预测数据集上，AUTAUT 超越 10 种辅助任务方法和 18 种先进分子性质预测模型
- 使用 8 张 NVIDIA A100 GPU 进行训练，LLM 检索每轮仅执行一次
- 消融实验表明，去除自适应加权后性能下降，证明梯度对齐机制有效抑制负迁移

## 背景与动机

分子性质预测是药物发现和材料科学中的核心任务，但面临严重的数据瓶颈：实验测定分子性质成本高昂，导致标注数据极度稀缺。例如，预测某种分子的血脑屏障穿透性或毒性，可能仅有数百个标注样本。为缓解这一问题，研究者引入辅助任务学习（auxiliary learning），通过共享表示从相关任务中迁移知识。

现有方法主要沿三条路径展开。**Learning to group auxiliary datasets for molecule** [17] 手动设计辅助任务分组策略，依赖化学领域专家预先定义任务关系；**Forkmerge** [18] 通过任务特定分支与合并机制缓解负迁移，但需要为每个辅助任务维护独立分支，结构复杂；**GradNorm** [3] 基于梯度幅度进行自适应损失平衡，虽能动态调整权重，但仅关注梯度大小而非方向一致性，无法识别任务间的真正冲突。

这些方法的共同瓶颈在于：**辅助任务的获取和集成高度依赖人工**。手动设计的任务可能遗漏关键辅助信息，而盲目引入不相关任务会导致负迁移（negative transfer），反而损害主任务性能。更关键的是，分子领域的知识分散在文献、数据库和专家经验中，人工遍历成本极高。

本文提出 AUTAUT，首次将大语言模型（LLM）引入辅助任务检索，并以梯度对齐实现动态加权，实现无需领域知识的全自动辅助学习。
![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/693d67d7-7d62-4e46-92ad-5bc0cb2e20e4/figures/Figure_1.png)
*Figure 1 (pipeline): Overview of the AuTAWT framework. There are three main steps: (1) auxiliary task selection; (2) adaptive weighting; and (3) primary task prediction.*



## 核心创新

核心洞察：LLM 蕴含的广泛分子知识可作为辅助任务的自动来源，而梯度方向的一致性比梯度幅度更能准确反映任务间的协同关系，从而使「无领域知识的自适应辅助学习」成为可能。

| 维度 | Baseline | 本文 |
|------|----------|------|
| 辅助任务来源 | 人工设计或数据集衍生 [17] | LLM 自动从分子知识中检索 |
| 任务筛选机制 | 预设规则或专家经验 | 基于 Task2vec 任务嵌入相似度 |
| 权重调整依据 | 梯度幅度（GradNorm [3]）或固定权重 | 梯度方向余弦相似度（对齐程度） |
| 负迁移处理 | 分支隔离（Forkmerge [18]） | 动态降权冲突任务 |
| 人工依赖 | 需要化学领域专家 | 完全自动化 |

## 整体框架



AUTAUT 采用三阶段流水线架构，如图 1 所示。

**阶段一：LLM 任务检索（LLM Task Retriever）**
输入主任务描述（如"预测分子血脑屏障穿透性"），LLM 自动从分子知识库中检索候选辅助任务描述（如"预测分子溶解度""预测分子极性表面积"等）。该步骤每轮训练仅执行一次，输出一组候选辅助任务标签描述。

**阶段二：任务选择（Task Selector）**
基于 Task2vec 启发的任务嵌入方法，计算候选任务与主任务的嵌入相似度，筛选出最相关的辅助任务集合 S，过滤掉 LLM 可能产生的幻觉任务或不相关任务。

**阶段三：自适应加权训练（Adaptive Weighted Multi-Task Training）**
分子图输入共享 GNN 编码器，通过任务特定预测头同时预测主任务和选中的辅助任务。核心创新在于梯度对齐加权模块：实时计算各辅助任务与主任务的梯度方向一致性，动态调整损失权重。

数据流：分子图 → LLM 检索候选任务 → 任务嵌入筛选得集合 S → 共享 GNN 编码器 → 多任务预测头 → 梯度对齐模块动态加权辅助损失 → 端到端训练。

```
[Primary Task Description] → [LLM Retriever] → [Candidate Tasks]
                                              ↓
[Molecular Graph] → [Task Selector] → [Selected Set S] → [Shared GNN Encoder]
                                                              ↓
                                         [Task-Specific Heads] ← [Primary + Aux outputs]
                                                              ↓
                                         [Gradient Alignment Weighting] → [Adaptive Weights]
                                                              ↓
                                                    [Combined Loss] → [Backprop]
```

## 核心模块与公式推导

### 模块 1: 梯度对齐分数（Gradient Alignment Score）（对应框架图阶段三）

**直觉**：两个任务的梯度方向越接近，它们越可能共享有益的表示；方向相反则表明任务冲突。

**Baseline 公式** (GradNorm [3]): GradNorm 使用梯度幅度进行归一化：
$$w_i \propto \|\nabla_\theta L_i\|$$
符号: $w_i$ = 第 i 个任务的权重, $\|\nabla_\theta L_i\|$ = 任务 i 的梯度 L2 范数

**变化点**: GradNorm 仅关注梯度大小，无法区分"梯度大但方向相反"（有害）与"梯度大且方向一致"（有益）的情况。本文改为测量梯度方向的余弦相似度。

**本文公式（推导）**:
$$\text{Step 1}: s_i = \cos(\nabla_\theta L_{\text{primary}}, \nabla_\theta L_{\text{aux}}^{(i)}) = \frac{\nabla_\theta L_{\text{primary}} \cdot \nabla_\theta L_{\text{aux}}^{(i)}}{\|\nabla_\theta L_{\text{primary}}\| \|\nabla_\theta L_{\text{aux}}^{(i)}\|} \quad \text{归一化消除幅度差异，仅保留方向信息}$$
$$\text{最终}: s_i \in [-1, 1]$$
其中 $s_i \approx 1$ 表示强对齐，$s_i \approx -1$ 表示严重冲突，$s_i \approx 0$ 表示无关。

**对应消融**: Figure 5 显示，去除自适应加权（改用固定权重）导致负迁移加剧，性能显著下降。

### 模块 2: 自适应权重更新规则（Adaptive Weight Update Rule）（对应框架图阶段三）

**直觉**：根据实时对齐分数，动态提升有益任务的权重，抑制冲突任务。

**Baseline 公式** (固定权重 / GradNorm): 
$$w_i = \text{constant} \quad \text{或} \quad w_i^{(t)} \propto \|\nabla_\theta L_i^{(t)}\| \text{ (GradNorm)}$$

**变化点**: 固定权重无法适应训练动态；GradNorm 的幅度归一化仍可能强化冲突任务。本文基于对齐分数 $s_i$ 设计单调递增映射。

**本文公式（推导）**:
$$\text{Step 1}: w_i^{(t+1)} = w_i^{(t)} \cdot (1 + \alpha \cdot s_i)^{\beta} \quad \text{幂律调整，}\alpha\text{控制灵敏度，}\beta\text{控制非线性程度}$$
$$\text{Step 2（替代形式）}: w_i^{(t+1)} \propto \exp(\gamma s_i) \quad \text{指数形式，}\gamma\text{为温度系数，保证正权重且区分度可控}$$
$$\text{Step 3（重归一化）}: w_i^{(t+1)} \leftarrow \frac{w_i^{(t+1)}}{\sum_j w_j^{(t+1)}} \cdot K \quad \text{保持总权重预算为常数} K$$
$$\text{最终}: w_i^{(t+1)} = f(s_i; \alpha, \beta, \gamma) \text{ 经归一化后的自适应权重}$$

**对应消融**: Figure 4 展示训练过程中辅助任务权重的动态变化，对齐任务权重持续上升，冲突任务权重被压制。

### 模块 3: 联合多任务目标函数（Combined Multi-Task Objective）（对应框架图整体）

**直觉**：将动态权重整合到标准多任务框架中，实现端到端自适应训练。

**Baseline 公式** (标准多任务学习):
$$L_{\text{total}} = L_{\text{primary}} + \lambda \sum_{i} L_{\text{aux}}^{(i)} \quad \text{固定}\lambda\text{，所有任务同等重要}$$

**变化点**: 固定 $\lambda$ 无法处理任务间动态关系；且对所有检索到的任务求和可能引入噪声。

**本文公式（推导）**:
$$\text{Step 1}: \mathcal{S} \leftarrow \text{TaskSelector}(\text{LLM}_{\text{retrieve}}(\text{primary task})) \quad \text{仅保留筛选后的任务集合}$$
$$\text{Step 2}: L_{\text{total}}^{(t)} = L_{\text{primary}}^{(t)} + \sum_{i \in \mathcal{S}} w_i^{(t)} \cdot L_{\text{aux}}^{(i),(t)} \quad \text{动态权重仅作用于选中任务}$$
$$\text{Step 3（梯度更新）}: \theta^{(t+1)} = \theta^{(t)} - \eta \nabla_\theta L_{\text{total}}^{(t)}$$
$$\text{最终}: L_{\text{total}} = L_{\text{primary}} + \sum_{i \in \mathcal{S}} w_i \cdot L_{\text{aux}}^{(i)}$$
符号: $\mathcal{S}$ = 经 LLM 检索和 Task2vec 筛选后的辅助任务集合, $w_i^{(t)}$ = 时刻 t 的自适应权重, $\eta$ = 学习率

**对应消融**: Table 2 比较不同辅助任务选择方法，证明 LLM 检索 + 任务嵌入筛选的组合优于随机选择或全量使用。

## 实验与分析


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/693d67d7-7d62-4e46-92ad-5bc0cb2e20e4/figures/Table_1.png)
*Table 1 (comparison): Model comparison of AuTAWT and related work. A model is self-contained if it does not require additional datasets.*



本文在多个分子性质预测基准上开展评估，涵盖分类任务（ROC-AUC）和回归任务（MAE/RMSE）。核心对比结果如 Table 1 所示：AUTAUT 在全部对比中超越 10 种辅助任务方法和 18 种先进分子性质预测模型，取得最优性能。Table 2 进一步验证不同辅助任务选择策略的影响，LLM 自动检索配合 Task2vec 筛选的组合显著优于纯随机选择或人工预设策略。


![Figure 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/693d67d7-7d62-4e46-92ad-5bc0cb2e20e4/figures/Figure_4.png)
*Figure 4 (result): Auxiliary task weights (left axis) and training loss (right axis) for two datasets.*



Figure 4 可视化两个数据集上的训练动态：左轴显示各辅助任务权重随 epoch 的演化，右轴显示训练损失曲线。可以观察到，与主任务对齐的辅助任务（如溶解度预测辅助血脑屏障穿透性预测）权重稳步上升至 0.3-0.5，而冲突任务权重被压制至接近 0.1，同时总损失平稳下降，验证梯度对齐机制有效避免了负迁移。


![Table 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/693d67d7-7d62-4e46-92ad-5bc0cb2e20e4/figures/Table_2.png)
*Table 2 (comparison): Performance of ML models with different auxiliary task selection methods on molecular property prediction.*



消融研究（Figure 5）揭示各组件的贡献：去除自适应加权模块（改用固定均匀权重）导致性能显著下降，表明梯度对齐是抑制负迁移的关键；去除 LLM 检索改用手动任务设计，性能下降证明自动检索的价值；去除任务选择器直接使用全部 LLM 检索结果，性能同样受损，说明 LLM 输出中存在幻觉或不相关任务需要过滤。训练使用 8 张 NVIDIA A100 GPU，LLM 检索每轮仅执行一次，额外计算开销可控。

公平性检查：对比基线涵盖该领域代表性方法，包括 Nature Machine Intelligence 发表的 Geometry-enhanced molecular representation learning 等强基线。潜在局限包括：依赖外部 LLM API 可能影响可复现性；LLM 检索的随机性未充分分析；仅验证分子领域，向其他科学领域的推广性待检验。

## 方法谱系与知识库定位

AUTAUT 属于「辅助任务学习 → 分子性质预测」方法谱系，直接继承自 **Learning to group auxiliary datasets for molecule** [17] 的问题设定，但在三个关键 slot 上完成结构性替换：

- **data_pipeline**: 人工设计 → LLM 自动检索
- **training_recipe**: 固定/手动权重 → 梯度对齐自适应加权  
- **objective**: 固定辅助损失 → 动态加权辅助损失

**直接基线对比**：
- vs [17] Learning to group...: 同问题，但 AUTAUT 用 LLM 替代人工分组，用梯度对齐替代固定权重
- vs [18] Forkmerge: 同目标（缓解负迁移），但 AUTMIT 用动态加权替代分支隔离结构
- vs [3] GradNorm: 同机制（梯度自适应），但 AUTAUT 用方向相似度替代幅度归一化，且新增 LLM 检索
- vs [5] Enhancing molecular...: 同任务（分子辅助学习），但 AUTAUT 完全自动化，无需领域专家

**后续方向**：(1) 扩展至蛋白质、材料等其他科学领域验证通用性；(2) 探索 LLM 检索的确定性增强与成本优化；(3) 结合更 recent 的梯度手术（gradient surgery）方法进一步提升对齐精度。

**标签**：modality: graph+text | paradigm: supervised multi-task learning | scenario: low-data scientific discovery | mechanism: LLM retrieval + gradient alignment | constraint: molecular domain, API-dependent

