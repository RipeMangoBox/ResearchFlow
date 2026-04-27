---
title: Bayesian Concept Bottleneck Models with LLM Priors
type: paper
paper_level: A
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 贝叶斯概念瓶颈模型：LLM先验驱动的迭代搜索
- BC-LLM
- BC-LLM uses a Bayesian framework wi
acceptance: Poster
cited_by: 11
code_url: https://github.com/kkzhang95/Awesome_Concept_Bottleneck_Models
method: BC-LLM
modalities:
- Text
- Image
- tabular
paradigm: supervised
---

# Bayesian Concept Bottleneck Models with LLM Priors

[Code](https://github.com/kkzhang95/Awesome_Concept_Bottleneck_Models)

**Topics**: [[T__Interpretability]], [[T__Classification]] | **Method**: [[M__BC-LLM]] | **Datasets**: fMoW: USA images, fMoW: OOD-China images

> [!tip] 核心洞察
> BC-LLM uses a Bayesian framework with LLMs as both concept extraction mechanisms and priors to iteratively search over a potentially infinite concept space, providing rigorous statistical inference and uncertainty quantification despite LLM miscalibration and hallucinations.

| 中文题名 | 贝叶斯概念瓶颈模型：LLM先验驱动的迭代搜索 |
| 英文题名 | Bayesian Concept Bottleneck Models with LLM Priors |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2410.15555) · [Code](https://github.com/kkzhang95/Awesome_Concept_Bottleneck_Models) |
| 主要任务 | Interpretability, Text Classification, Image Classification |
| 主要 baseline | LLM+CBM, Boosting LLM+CBM, ResNet50, Post-hoc CBM, Label-free CBM |

> [!abstract] 因为「Concept Bottleneck Models (CBMs) 面临探索足够大概念集合与控制概念提取成本之间的根本权衡，且 LLM 辅助概念提取存在幻觉和校准不良问题」，作者在「Concept Bottleneck Models (Koh et al.)」基础上改了「引入贝叶斯推断框架，将 LLM 作为概率先验而非一次性标注器，通过迭代搜索动态探索潜在无限概念空间」，在「Imagenette 和 fMoW」上取得「准确率 0.987 vs 0.902 (LLM+CBM)，Brier score 0.072 vs 0.125」

- **Imagenette**: Accuracy 0.987，相比 LLM+CBM (0.902) 提升 +0.085 (+9.4%)；Brier score 0.072 vs 0.125，降低 42.4%
- **fMoW 域内**: Accuracy 0.357 vs 0.311 (LLM+CBM)，AUC 0.904 vs 0.878，Brier score 0.78 vs 0.892
- **fMoW OOD-China**: Brier score 0.853 vs 1.023 (LLM+CBM)，在分布偏移下保持更优校准

## 背景与动机

Concept Bottleneck Models (CBMs) 承诺在深度学习中实现「先预测可解释概念，再基于概念预测标签」的透明推理。然而，这一范式面临一个根本性困境：为了获得足够表达能力，需要探索庞大的概念空间；但每增加一个候选概念，都意味着昂贵的标注成本——无论是人工标注还是 LLM 查询。现有方法通常采用折中方案：一次性固定一个较小的候选概念集，再从中筛选。例如，在鸟类分类任务中，研究者可能预先定义「有黄腹、有冠羽、喙弯曲」等几十个属性，但这些固定概念往往无法覆盖决策所需的全部判别信息。

现有三条技术路线各有局限：
- **标准 CBM** (Koh et al., [5])：依赖人工预定义概念集，固定不变，无法动态发现任务相关的新概念；
- **LLM+CBM** ([22])：利用 LLM 一次性总结概念候选，虽减少人工设计，但概念集仍固定，且 LLM 幻觉和校准不良直接传递为模型错误；
- **Boosting LLM+CBM** (本文实验基线)：尝试每轮迭代添加一个概念，但缺乏系统性搜索策略，简单累加候选反而引入噪声。

这些方法的共同瓶颈在于：**将 LLM 视为一次性概念生成器而非持续的知识来源**，且**缺乏对概念不确定性的量化机制**。当 LLM 对「某图像是否包含湿地栖息地」给出错误标注时，标准 CBM 无从察觉；当固定概念集遗漏关键判别特征时，模型无法自我修正。本文的核心动机正是：能否让 CBM 在迭代搜索中主动利用 LLM 的先验知识，同时通过贝叶斯框架对 LLM 的噪声和校准误差进行统计修正？


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/8caf3bdf-c87e-4b92-9fa0-c77b28271989/figures/fig_001.png)
*Figure: BC-LLM is initialized by having the LLM hypothesize the top concepts based on*



本文提出 BC-LLM，将 LLM 重新定位为概率先验，通过贝叶斯后验推断实现概念集的动态精炼与不确定性量化。

## 核心创新

核心洞察：LLM 的概念假设可以作为贝叶斯先验分布而非确定性候选集，因为 LLM 的语义知识天然编码了概念间的概率关联，从而使「在潜在无限概念空间中进行不确定性引导的迭代搜索」成为可能，即使 LLM 存在幻觉和校准不良，仍能通过贝叶斯更新获得统计保证。

| 维度 | Baseline (LLM+CBM) | 本文 (BC-LLM) |
|:---|:---|:---|
| 概念空间 | 固定、预定义的有限候选集 | 动态扩展，迭代搜索潜在无限空间 |
| LLM 角色 | 一次性概念总结器（标注工具） | 概率先验分布 + 持续标注机制 |
| 参数估计 | 点估计（MLE/MAP） | 分割样本后验 + Laplace 近似，带置信区间 |
| 错误处理 | LLM 错误直接传递至下游 | 贝叶斯更新自动衰减噪声影响 |
| 概念更新 | 无（训练后固定） | 每 epoch 基于后验不确定性动态增删 |

这一范式转变使 CBM 首次具备「认知不确定性」——不仅知道预测什么，还知道对哪些概念不够确定、需要进一步探索。

## 整体框架


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/8caf3bdf-c87e-4b92-9fa0-c77b28271989/figures/fig_002.png)
*Figure: Example of BC-LLM classifying Bunting birds. (Left) Left and right dendrograms are*



BC-LLM 的数据流遵循「先验初始化 → 迭代精炼 → 不确定性输出」的三阶段结构：

**阶段一：LLM 先验初始化**
- 输入：原始数据 X + LLM 查询 prompt
- 输出：初始概念集合 $\vec{c}^{(0)} = (c_1, ..., c_K)$ 及先验分布
- 作用：利用 LLM 的语义知识假设任务相关概念，而非人工设计

**阶段二：迭代贝叶斯精炼循环**（核心，重复 T=5 epochs 或最多 25 iterations）
1. **概念标注 (LLM as Annotator)**：当前概念集 $\vec{c}^{(t-1)}$ 输入 LLM，输出概念激活矩阵 $\Phi \in \mathbb{R}^{n \times K}$，其中 $\Phi_{ij} = \phi_{c_j}(x_i)$ 表示样本 $x_i$ 对概念 $c_j$ 的激活程度
2. **贝叶斯后验推断 (Split-Sample + Laplace)**：将数据分为子集 S 和 S_c，计算概念系数 $\vec{\theta}$ 的后验分布 $p(\vec{\theta} | y^{S_c}, y^S, \vec{c}, X)$，输出 MAP 估计 $\hat{\theta}$ 与 Hessian 逆矩阵 $H^{-1}$ 表征的不确定性
3. **概念集动态更新**：基于后验协方差 $\Sigma = H^{-1}$ 识别高不确定性概念，结合 LLM 新建议替换或扩展概念集，得到 $\vec{c}^{(t)}$

**阶段三：最终预测**
- 输入：测试数据 + 收敛后的概念集 + 后验分布
- 输出：预测标签及校准置信区间

```
原始数据 X ──→ [LLM 先验初始化] ──→ 概念集 c^(0)
                                      ↓
                    ┌─────────────────┘
                    ↓
            [for t = 1, ..., T]
            ├─→ [LLM 概念标注] ──→ Φ^(t)
            ├─→ [Bayesian 后验推断] ──→ (θ̂^(t), Σ^(t))
            └─→ [概念集更新] ──→ c^(t)
                    ↓
            [收敛判断] ──否──→ 继续循环
                    ↓ 是
            [带不确定性的预测输出]
```

关键设计：空列表 L=[] 累积历史概念集，确保搜索过程的可追溯性；早停机制避免无效迭代。

## 核心模块与公式推导

### 模块 1: 迭代贝叶斯概念搜索（对应框架图「迭代循环」整体）

**直觉**: 固定概念集如同一次性的「开卷考试」，而迭代搜索允许模型在「做题」过程中发现「复习遗漏」并动态「查缺补漏」。

**Baseline 公式** (LLM+CBM): 
$$\vec{c} = \text{LLM\_summarize}(X) \quad \text{(one-time)}$$
$$\hat{y} = f(g(x; \vec{c}); \hat{\theta}_{\text{MLE}})$$

符号: $\vec{c}$ = 固定概念集, $g$ = 概念编码器, $f$ = 标签预测器, $\hat{\theta}_{\text{MLE}}$ = 最大似然估计

**变化点**: LLM+CBM 的先验知识仅在初始化时使用一次，无法响应训练中发现的概念缺失；且 MLE 点估计无法区分「概念确实无关」与「数据不足无法判断」。

**本文公式（推导）**:
$$\text{Step 1 (初始化)}: \vec{c}^{(0)} \sim \text{LLM\_prior}(X), \quad L = [] \quad \text{(LLM 输出定义概率分布而非候选列表)}$$
$$\text{Step 2 (迭代)}: \text{for } t = 1, 2, \ldots, T \text{ do}$$
$$\quad \Phi^{(t)}_{ij} = \phi_{c^{(t-1)}_j}(x_i) \quad \text{(LLM 标注构建激活矩阵)}$$
$$\text{Step 3 (贝叶斯更新)}: p(\vec{\theta} | y^{S_c}, y^S, \vec{c}^{(t-1)}, X) \propto p(y^{S_c} | \Phi^{(t-1)}, \vec{\theta}) \cdot p(\vec{\theta} | y^S, \vec{c}^{(t-1)}, X) \cdot \underbrace{p(\vec{\theta})}_{\mathcal{N}(0, \gamma^2 I)}$$
$$\text{Step 4 (动态精炼)}: \vec{c}^{(t)} = \text{update}\big(\vec{c}^{(t-1)}, \underbrace{H^{-1}(\hat{\theta}^{(t)})}_{\text{后验不确定性}}, \text{LLM\_suggestions}^{(t)}\big)$$
$$L \leftarrow L \cup \{\vec{c}^{(t)}\}$$

**对应消融**: 去掉迭代搜索退化为 LLM+CBM，Imagenette accuracy 从 0.987 降至 0.902，Δ = -0.085；Boosting LLM+CBM（无贝叶斯引导的朴素迭代）进一步降至 0.639，Δ = -0.348，证明「迭代」本身不足够，必须配合不确定性引导。

---

### 模块 2: 分割样本后验的 Laplace 近似（对应框架图「Bayesian 后验推断」）

**直觉**: LLM 标注有噪声，若用全部数据既估计概念系数又评估其可靠性，会导致「自己检验自己」的过拟合；分割样本设计实现「用一半数据学习、另一半验证」的统计严谨性。

**Baseline 公式** (标准 CBM / LLM+CBM):
$$\hat{\theta} = \text{arg}\max_\theta \log p(y | \Phi, \vec{\theta}) = \text{arg}\max_\theta \sum_{i=1}^n \log p(y_i | \vec{\phi}(x_i)^\text{top} \vec{\theta})$$

符号: $\Phi$ = 概念激活矩阵, $y$ = 标签, $\vec{\theta} \in \mathbb{R}^K$ = 概念系数向量, $p(y_i | \cdot)$ = 逻辑似然

**变化点**: MLE/MAP 只输出点估计，无法量化「对概念 $c_j$ 的权重 $\theta_j$ 有多确定」；且当 LLM 对某概念系统性错误标注时，MLE 会过拟合到噪声模式。需要显式建模后验不确定性，并分离「训练」与「验证」的数据角色。

**本文公式（推导）**:
$$\text{Step 1 (数据分割)}: \text{将 } n \text{ 个样本分为 } S \text{ (训练子集) 和 } S_c \text{ (验证子集)}$$
$$\text{Step 2 (先验构建)}: p(\vec{\theta}) = \mathcal{N}(\vec{0}, \gamma^2 I_K) \quad \text{(标准高斯先验，编码系数稀疏偏好)}$$
$$\text{Step 3 (条件后验)}: p(\vec{\theta} | y^S, \vec{c}, X) \propto p(y^S | \Phi^S, \vec{\theta}) \cdot p(\vec{\theta}) \quad \text{(用 S 子集构建先验)}$$
$$\text{Step 4 (预测后验)}: p(y^{S_c} | y^S, \vec{c}, X) = \int p(y^{S_c} | \Phi^{S_c}, \vec{\theta}) \cdot p(\vec{\theta} | y^S, \vec{c}, X) \, d\vec{\theta}$$
$$\text{Step 5 (Laplace 近似)}: \hat{\theta} = \text{arg}\max_\theta \log p(\vec{\theta} | y^S, \vec{c}, X) \quad \text{(MAP 估计)}$$
$$p(\vec{\theta} | y^S, \vec{c}, X) \approx \mathcal{N}\big(\hat{\theta}, \underbrace{H^{-1}(\hat{\theta})}_{\Sigma}\big), \quad H_{ij} = -\frac{\partial^2 \log p(\vec{\theta} | \cdot)}{\partial \theta_i \partial \theta_j}$$
$$\text{最终}: \underbrace{p(y^{S_c} | y^S, \vec{c}, X)}_{\text{可计算的后验预测}} \approx \int p(y^{S_c} | \Phi^{S_c}, \vec{\theta}) \cdot \mathcal{N}(\hat{\theta}, H^{-1}) \, d\vec{\theta}$$

**对应消融**: 去掉 Laplace 近似退化为 MAP 点估计，失去不确定性量化能力；Table 3 显示 BC-LLM 的 Brier score (0.072 on Imagenette) 显著优于 LLM+CBM (0.125)，证明校准收益来自完整贝叶斯处理而非仅概念搜索。

---

### 模块 3: MIMIC 合成标签验证（对应框架图「临床验证」分支）

**直觉**: 在医疗等高风险领域，需要验证 BC-LLM 能否从 LLM 建议的众多概念中「找回」真正因果相关的临床因素，而非过拟合到相关性假象。

**本文公式**:
$$\log \frac{\Pr(Y=1|X)}{\Pr(Y=0|X)} = 4 \cdot \mathbb{1}\{\text{unemployed}\} + 4 \cdot \mathbb{1}\{\text{retired}\} + 4 \cdot \mathbb{1}\{\text{alcohol}\} - 4 \cdot \mathbb{1}\{\text{smoking}\} + 5 \cdot \mathbb{1}\{\text{drugs}\}$$

符号: $\mathbb{1}\{\cdot\}$ = 指示函数（概念激活）, 系数 4/5 = 真实因果效应大小, Y = 再入院风险

**设计目的**: 基于 5 个已知临床概念生成合成标签，检验 BC-LLM 能否在 LLM 建议的扩展概念集中识别出这 5 个真实驱动因素，并正确估计其效应方向（如 smoking 为负向保护因素）。该模块不直接贡献训练目标，但作为「概念恢复」的统计验证机制。

## 实验与分析



本文在图像分类（Imagenette、fMoW）和医疗预测（MIMIC）任务上评估 BC-LLM。核心结果来自 Table 3：在 Imagenette 上，BC-LLM 达到 accuracy 0.987、AUC 0.999、Brier score 0.072，全面优于 LLM+CBM（0.902 / 0.988 / 0.125）和 Boosting LLM+CBM（0.639 / 0.899 / 0.448）。fMoW 卫星图像的域内（USA）与分布外（China）评估显示，BC-LLM 在 accuracy（0.357 vs 0.311）和 AUC（0.904 vs 0.878）上领先 LLM+CBM，且 Brier score 优势在 OOD 场景下扩大（0.853 vs 1.023），表明贝叶斯框架的校准鲁棒性。


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/8caf3bdf-c87e-4b92-9fa0-c77b28271989/figures/fig_003.png)
*Figure: MIMIC results: (Left) Comparison of BC-LLM and existing methods in terms of perfor-*



关键数字解读：Imagenette 上 +0.085 的准确率提升并非来自更多概念——Boosting LLM+CBM 同样增加概念但跌至 0.639——而是来自不确定性引导的选择性搜索。Brier score 从 0.125 降至 0.072（降低 42.4%）直接证明后验预测的概率校准改善，这对医疗决策等需要可靠置信度的场景至关重要。



消融分析揭示两个关键结论：第一，去掉贝叶斯迭代搜索（即 LLM+CBM）导致 Imagenette accuracy 下降 0.085，fMoW 域内 AUC 下降 0.026，说明固定概念集的根本局限；第二，Boosting LLM+CBM（无贝叶斯引导的朴素迭代）表现最差，Imagenette accuracy 仅 0.639，比 BC-LLM 低 0.348，证明「迭代」本身可能引入噪声，必须有不确定性量化机制指导概念增删。

公平性检查：Table 3 仅对比 LLM+CBM 变体，缺失与 Post-hoc CBM [11,20]、Label-free CBM [8]、Sparse linear concept discovery [23] 等基线的直接数值比较；fMoW OOD-China 的 AUC 值在原文中标记为缺失；CUB-Birds 上仅报告与 ResNet50 的对比但未出现在 Table 3 中。作者披露的计算成本信息有限，迭代 LLM 查询的 API 开销可能是大规模部署的瓶颈。MIMIC 实验使用合成标签而非真实临床结局，外部有效性待验证。

## 方法谱系与知识库定位

**方法家族**: Concept Bottleneck Models → Probabilistic CBM → **BC-LLM** (贝叶斯 + LLM 先验融合)

**父方法**: Concept Bottleneck Models (Koh et al., [5]) —— 提供「概念层→预测层」的两阶段架构；Probabilistic CBM ([6]) —— 提供贝叶斯处理的技术基础。BC-LLM 的核心改造在于：将 LLM 从「外部标注工具」重新定义为「概率先验分布」，并引入迭代搜索替代固定概念集。

**直接基线与差异**:
- **LLM+CBM** [22]: 同样使用 LLM 生成概念，但一次性固定；BC-LLM 改为迭代贝叶斯搜索 + 不确定性量化
- **Boosting LLM+CBM**: 尝试迭代加概念但无贝叶斯引导；BC-LLM 证明朴素迭代有害，需后验不确定性指导
- **Post-hoc CBM** [11,20]: 事后拟合概念到预训练模型；BC-LLM 是端到端联合学习，概念与预测协同优化
- **Label-free CBM** [8]: 减少标注负担的另一路径；BC-LLM 通过 LLM 先验降低标注需求但保持统计严谨性

**后续方向**:
1. **收敛理论**: 当前缺乏对抗性 LLM 行为下的收敛速率分析，需建立迭代次数 T 与概念空间复杂度的理论联系
2. **效率优化**: 迭代 LLM 查询成本高昂，探索轻量级 LLM 或缓存机制以扩展至大规模应用
3. **多模态扩展**: 当前主要在图像/文本验证，向表格数据（如 MIMIC 真实临床标签）和时序数据的迁移

**标签**: 模态={text, image, tabular}, 范式={supervised, Bayesian inference}, 场景={interpretability, high-stakes decisions}, 机制={iterative search, uncertainty quantification, LLM prior}, 约束={LLM miscalibration robustness, statistical guarantee}

## 引用网络

### 直接 baseline（本文基于）

- Relational Concept Bottleneck Models _(NeurIPS 2024, 直接 baseline, 未深度分析)_: Foundational CBM paper. The paper title 'Bayesian Concept Bottleneck Models with
- [Re] On the Reproducibility of Post-Hoc Concept Bottleneck Models _(NeurIPS 2024, 实验对比, 未深度分析)_: Post-hoc CBM variant. Likely a key comparison baseline in experiments. The 2022 

