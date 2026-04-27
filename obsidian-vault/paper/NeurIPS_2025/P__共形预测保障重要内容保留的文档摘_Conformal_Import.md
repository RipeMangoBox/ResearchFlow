---
title: Document Summarization with Conformal Importance Guarantees
type: paper
paper_level: C
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 共形预测保障重要内容保留的文档摘要
- Conformal Import
- Conformal Importance Summarization
acceptance: Poster
code_url: https://github.com/layer6ai-labs/conformal-importance-summarization
method: Conformal Importance Summarization
modalities:
- Text
paradigm:
- conformal prediction (distribution-free statistical inference, no training required)
- supervised
---

# Document Summarization with Conformal Importance Guarantees

[Code](https://github.com/layer6ai-labs/conformal-importance-summarization)

**Topics**: [[T__Text_Generation]] | **Method**: [[M__Conformal_Importance_Summarization]] | **Datasets**: ECTSum, CSDS, CSDS + multiple datasets, Multiple datasets

> [!tip] 核心洞察
> Conformal Importance Summarization is the first framework to provide distribution-free, finite-sample coverage guarantees for importance-preserving extractive document summarization by calibrating sentence-level importance scores using conformal prediction.

| 中文题名 | 共形预测保障重要内容保留的文档摘要 |
| 英文题名 | Document Summarization with Conformal Importance Guarantees |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2509.20461) · [Code](https://github.com/layer6ai-labs/conformal-importance-summarization) · [Project](https://arxiv.org/abs/2509.20461) |
| 主要任务 | Text Generation / Extractive Document Summarization |
| 主要 baseline | GPT-4o mini extractive summarization, Gemini 2.5 Flash, classical NLP summarization methods |

> [!abstract] 因为「LLM摘要系统在高风险领域（医疗、法律、金融）缺乏对关键内容包含性的可靠保证」，作者在「Conformal risk control (Angelopoulos et al.)」基础上改了「将共形预测从分类任务扩展到提取式摘要，引入用户可指定的错误率α和召回率β」，在「ECTSum, CSDS, CNN/DailyMail等数据集」上取得「覆盖率严格落在[1-α, 1-α+1/(n+1)]理论区间内，且首次实现可调的重要句子召回保证」。

- **理论保证**：期望覆盖率严格 bounded 在 1-α 与 1-α+1/(n+1) 之间（Theorem 1），400次随机划分实验验证
- **可调控制**：通过α∈(0,1)和β∈(0,1)实现召回率-简洁性的连续权衡，GPT-4o mini基线仅能提供固定单点性能
- **模型无关**：兼容Gemini 2.5 Flash、GPT-4o mini及传统NLP方法，无需模型访问或重训练

## 背景与动机

自动文档摘要系统已被广泛部署于新闻聚合、医疗记录整理和法律文件分析等场景，但现有方法存在一个根本性的信任危机：用户无法确知生成的摘要是否遗漏了关键信息。例如，在医疗场景中，一份出院小结若遗漏"患者对青霉素过敏"这一关键句，可能导致严重医疗事故；然而当前基于LLM的提取式摘要仅输出重要性分数排序后的top-k句子，既无理论保证，也不允许用户根据风险承受能力调整召回率。

现有方法如何处理这一问题？**传统提取式摘要**（如TextRank、LexRank）基于图排序或频率启发式选择句子，优化ROUGE等表面指标，但不提供任何关于重要内容保留率的统计保证。**LLM-based提取式摘要**（如GPT-4o mini直接输出重要句子分数）虽利用强大语义理解能力，但其阈值选择是启发式的——用户设定"保留前3句"或"分数>0.5"均无法对应到任何可解释的重要句子覆盖率。**基于优化的摘要方法**则直接最大化ROUGE或BERTScore，但这些指标与"重要内容是否被包含"仅有弱相关性，且优化目标中不存在用户可控的错误率参数。

这些方法的共同短板在于：**缺乏分布无关的（distribution-free）、有限样本的（finite-sample）统计保证**。具体而言，用户无法回答"这份摘要有95%的概率包含了至少90%的重要句子"这类问题。这一局限使得LLM摘要系统难以部署于高风险领域，因为下游决策者无法量化遗漏关键信息的风险。

本文将共形预测（Conformal Prediction）这一分布无关的不确定性量化框架首次系统性地应用于提取式摘要，通过在小规模校准集上校准句子重要性分数的阈值，使摘要系统能够向用户提供可证明的、用户可定制的覆盖率保证。


![Figure 6](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/6009e1b0-251b-424d-98bf-46189688d92f/figures/Figure_6.png)
*Figure 6 (example): Figure 6: Example of Conformal Importance Summarization using Llama 2. Left: source article with sentences highlighted by importance score (darker is more important). Right: summary-level length and importance score distribution.*



## 核心创新

核心洞察：将摘要任务重新建模为**广义共形预测中的风险控制问题**，因为句子重要性分数满足交换性假设，从而使「以用户指定参数(α, β)为约束的提取式摘要」获得分布无关的有限样本保证成为可能——其中α控制错误概率，β控制最低召回比例。

| 维度 | Baseline (直接阈值/Top-k) | 本文 (Conformal Importance Summarization) |
|:---|:---|:---|
| **优化目标** | 最大化ROUGE/BERTScore或启发式覆盖 | 满足 P[重要句召回率 ≥ β] ≥ 1-α 的统计保证 |
| **阈值选择** | 人工设定固定阈值k或分数cutoff | 基于校准集分位数自动计算 q̂，适配数据分布 |
| **用户控制** | 无显式参数控制遗漏风险 | 双参数(α, β)连续调节召回率-简洁性权衡 |
| **理论性质** | 无有限样本保证 | 覆盖率期望严格 bounded 于 [1-α, 1-α+1/(n+1)] |
| **模型依赖** | 通常绑定特定模型架构 | 黑盒兼容：任何输出句子分数的模型均可接入 |

## 整体框架


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/6009e1b0-251b-424d-98bf-46189688d92f/figures/Figure_1.png)
*Figure 1 (pipeline): Main Steps to construct the conformal importance filtering procedure. Dotted lines represent steps that only happen at training time (on the calibration set).*



Conformal Importance Summarization (CIS) 的完整数据流包含四个阶段，从原始文档到带保证的摘要输出：

**阶段一：句子重要性评分** — 输入文档 x，通过任意黑盒评分函数 R(c; x) 为每个句子 c 输出重要性分数。该函数可以是Gemini 2.5 Flash、GPT-4o mini等LLM的提示输出，也可以是传统NLP方法（如TF-IDF、TextRank）。输出：句子级分数向量。

**阶段二：共形分数计算** — 输入重要性分数与真实重要句子集合 y*（仅校准阶段需要），计算共形分数 s(x,y) = Σ_{c∈y} R(c;x) − Σ_{c∈y*} R(c;x)。该分数量化摘要 y 相对于最优重要集合 y* 的覆盖不足程度。输出：标量共形分数。

**阶段三：校准阈值估计** — 在包含 n 个样本的校准集上，计算共形分数集合 {s(x_i, y_i*)}_{i=1}^n，取其 ⌈(n+1)(1−α)⌉/n 分位数作为阈值 q̂。此步骤仅在训练/部署前执行一次。输出：校准阈值 q̂。

**阶段四：测试时摘要生成** — 对新文档，保留所有满足 Σ_{c∈y} R(c;x) − q̂ ≥ β·Σ_{c∈y*} R(c;x) 的句子（实际实现为基于校准阈值的过滤），输出满足(1−α, β)保证的提取式摘要。

```
文档 x  ──►  R(c;x)  ──►  s(x,y)  ──►  {s(x_i,y_i*)}  ──►  q̂  ──►  摘要 y
              ▲            ▲              ▲
              │            │              │
         [黑盒LLM]    [真实标签y*]    [校准集，大小n]
         或NLP方法    (仅训练时需要)   (仅训练时需要)
```

关键设计：阶段一与阶段二解耦——评分函数 R(c;x) 完全可替换，阶段三和四提供与具体评分模型无关的统计保证层。

## 核心模块与公式推导

### 模块 1: 共形分数计算（对应框架图 阶段二）

**直觉**：需要一种分数来衡量"候选摘要 y 漏掉了多少重要内容"，以便后续校准能控制这种遗漏的概率。

**Baseline 公式** (标准共形预测分类)：
$$s_{\text{classify}}(x, y) = 1 - \hat{\pi}_y(x)$$
符号: $\hat{\pi}_y(x)$ = 模型对标签 y 的预测概率，分数越高表示模型越不确定。

**变化点**：分类任务的共形分数基于概率输出，但摘要任务的"标签"是句子集合而非单一类别。需要将"集合覆盖不足"量化为可加性分数。

**本文公式（推导）**：
$$\text{Step 1}: \quad R(c; x) \text{ 为句子 } c \text{ 的重要性分数} \quad \text{（来自任意黑盒评分器）}$$
$$\text{Step 2}: \quad s(x, y) = \sum_{c \in y} R(c; x) - \sum_{c \in y^*} R(c; x) \quad \text{（加入第二项以衡量相对最优集合的覆盖缺口）}$$
$$\text{最终}: \quad s(x, y) = \sum_{c \in y} R(c; x) - \sum_{c \in y^*} R(c; x)$$

符号: $y$ = 候选摘要句子集合, $y^*$ = 真实重要句子集合, $R(c;x)$ = 句子 c 在文档 x 中的重要性分数。当 $y$ 遗漏重要句子时，第二项未被充分抵消，$s(x,y)$ 升高；当 $y$ 完全覆盖 $y^*$ 时，$s(x,y)$ 可能为负（因 $y$ 可包含额外句子）。

**对应消融**：Table 2 显示不同评分函数（Gemini 2/2.5、GPT-4o mini、传统NLP）的 AUPRC 差异直接影响最终保证的紧度；Gemini 2/2.5 的分数质量最优。

---

### 模块 2: 广义覆盖率保证与分位数阈值选择（对应框架图 阶段三+四）

**直觉**：用户需要"至少保留β比例重要句子"的概率保证，而非二元包含；通过在校准集上取适当分位数，可将标准共形预测扩展到这种比例型保证。

**Baseline 公式** (标准共形预测覆盖率)：
$$\mathbb{P}[Y \in \mathcal{C}(X)] \geq 1 - \alpha$$
符号: $\mathcal{C}(X)$ = 预测集合，保证真实标签以至少 1−α 概率落入其中。

**变化点**：标准保证是二元的（标签在/不在集合中），但摘要需要比例型保证（"至少β比例的重要句子被包含"）。此外，标准共形预测直接构造预测集合，而本文需要为提取式摘要的阈值选择提供保证。

**本文公式（推导）**：
$$\text{Step 1}: \quad \text{定义覆盖率 } B(y; y^*) = \frac{|y \cap y^*|}{|y^*|} \quad \text{（将二元包含推广为比例度量）}$$
$$\text{Step 2}: \quad \hat{q} = \text{Quantile}\left(\left\{s(x_i, y_i^*)\right\}_{i=1}^{n}; \frac{\lceil(n+1)(1-\alpha)\rceil}{n}\right) \quad \text{（重归一化分位水平以适配有限样本）}$$
$$\text{Step 3}: \quad \mathbb{P}\left[\frac{|y \cap y^*|}{|y^*|} \geq \beta\right] \geq 1 - \alpha \quad \text{（Eq. 6: 核心广义保证）}$$
$$\text{最终定理}: \quad 1 - \alpha \leq \mathbb{E}[\text{Coverage}] \leq 1 - \alpha + \frac{1}{n+1} \quad \text{（Theorem 1: 期望上下界）}$$

符号: $n$ = 校准集大小, $\alpha$ = 允许的错误率, $\beta$ = 最低召回比例, $\hat{q}$ = 校准阈值。分位水平中的 $\lceil(n+1)(1-\alpha)\rceil/n$ 是有限样本修正，确保交换性假设下的覆盖保证成立；$1/(n+1)$ 项来自校准集离散化带来的固有松弛。

**对应消融**：Figure 3 显示当 α 从 0.01 增至 0.20，经验覆盖率从 ~0.99 降至 ~0.80，严格跟随理论预测；Figure 4 显示 β 从 0.5 提至 0.9，保留句子比例从 ~20% 升至 ~60%，验证双参数独立控制。

## 实验与分析


![Table 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/6009e1b0-251b-424d-98bf-46189688d92f/figures/Table_2.png)
*Table 2 (ablation): Table 2: A higher count of filters improves the performance of different loss functions as measured by the target recall τ_t (at τ_s = 0.90) on CNN/DailyMail.*




![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/6009e1b0-251b-424d-98bf-46189688d92f/figures/Figure_3.png)
*Figure 3 (result): Figure 3: Empirical marginal coverage (left) versus average empirical coverage (right) as a function of α for various base summarizers (Section 5). Dashed lines indicate nominal levels.*



本文在 ECTSum（临床对话摘要）、CSDS（对话摘要）和 CNN/DailyMail（新闻摘要）三个数据集上验证理论保证与实用性能。核心实验设计为：对每个(α, β)配置，在 n=1000 的校准集上估计阈值，然后在独立测试集上评估经验覆盖率与召回率，重复400次随机划分以消除数据划分带来的方差。

**覆盖率保证验证**：ECTSum 数据集上的 400 次随机试验显示，平均经验覆盖率始终落在 Theorem 1 预测的 [1−α, 1−α+1/(n+1)] 区间内。例如当 α=0.10、n=1000 时，理论区间为 [0.90, 0.901]，经验观测值约为 0.9005，验证有限样本保证的紧度。Figure 3（左）进一步展示不同基摘要器（Gemini 2.5 Flash、GPT-4o mini 等）的边际覆盖率随 α 的变化曲线，所有曲线均紧密贴合对角线（理想 1−α），无系统偏离。

**可调召回率-简洁性权衡**：Figure 4 展示 CSDS 数据集上目标召回率 τ_t（对应 β）与保留句子比例的关系。当 β=0.90 时，CIS 保留约 55% 句子；当 β=0.50 时，仅保留约 25%。作为对比，GPT-4o mini 无共形预测基线（图中星号标记）提供固定单点：约保留 40% 句子但无法对应到任何可控的 β 值，且其实际召回率随输入变化而波动。Figure 5 扩展至 CNN/DailyMail，显示 α=0.10 时不同 β 设置下的压缩率，验证跨数据集的一致性。


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/6009e1b0-251b-424d-98bf-46189688d92f/figures/Table_1.png)
*Table 1: Table 1: Dataset Details.*



**消融实验**：Table 2 比较不同评分函数与损失函数组合在目标召回 τ_t=0.90 时的性能。关键发现：增加 filter 数量（即更细粒度的评分函数）持续改进各损失函数的表现；Gemini 2/2.5 的评分在所有数据集上 AUPRC 最优，GPT-4o mini 次之但显著优于传统 NLP 方法。另一项消融（附录 Table 5-6）显示校准集大小 n 从 100 增至 2000 时，经验覆盖率从 ~0.895（略低于 1−α=0.90）快速收敛至理论区间，1/(n+1) 项的松弛在 n≥500 后 practically negligible。

**公平性审视**：基线选择存在局限。主要对比对象 GPT-4o mini 无共形预测版本并非专门优化的摘要系统，而是通用 LLM 的直接应用；未与当前 SOTA 摘要方法（如基于 BERTScore 优化的模型、或专为摘要微调的 Llama 3/Qwen 等开源模型）比较。实验未报告标准摘要指标 ROUGE/BERTScore，使得与传统摘要研究的直接对比困难。此外，400 次随机划分的平均值未报告置信区间，难以评估方差。作者明确披露：方法仅限提取式摘要，无法扩展至需要生成新表述的抽象式摘要；且校准集需含真实重要性标签，获取成本在高风险领域可能较高。

## 方法谱系与知识库定位

本文属于 **Conformal Prediction for Structured Prediction** 方法族，直接父方法为 **Conformal Risk Control** (Angelopoulos et al., 2021) [4]。核心继承：将风险控制框架从标准分类/回归推广到集合型输出（摘要句子集合）。关键改动 slot：

- **Objective（目标函数）**：从控制分类错误率 / 预测集大小，替换为控制「重要句子召回比例」的分布无关保证
- **Inference Strategy（推理策略）**：从直接阈值化或 top-k 选择，替换为基于校准集分位数的共形阈值 q̂
- **Data Pipeline（数据流程）**：新增显式校准集，需含真实重要性标签 y* 用于阈值估计

直接基线与差异：
- **GPT-4o mini extractive summarization**：同为提取式摘要，但无统计保证，仅提供固定启发式输出；CIS 在其上叠加共形校准层
- **Gemini 2.5 Flash**：作为 CIS 内部可替换的评分函数之一，其分数质量决定保证的紧度但不改变保证本身
- **Classical NLP methods** (TextRank/LexRank)：CIS 可兼容其评分函数，但传统方法无校准机制
- **Large language model validity via enhanced conformal prediction** [14]：相近领域（LLM+共形预测），但聚焦分类/QA 的正确性保证而非摘要内容覆盖

后续方向：
1. **扩展至抽象式摘要**：当前保证依赖句子级别的包含关系，如何为生成新表述的摘要提供语义等价性保证是开放问题
2. **降低校准集依赖**：探索半监督或无监督的重要性标签估计，减少 y* 的标注成本
3. **多文档与流式场景**：将单文档的 (α, β) 保证扩展到多文档聚合与实时文档流

**知识库标签**：
- Modality: text
- Paradigm: supervised (with calibration), model-agnostic black-box wrapper
- Scenario: high-stakes summarization (healthcare, legal, financial)
- Mechanism: conformal prediction, risk control, quantile calibration
- Constraint: distribution-free, finite-sample guarantee, extractive-only

## 引用网络

### 直接 baseline（本文基于）

- Conformal Risk Control _(ICLR 2024, 方法来源, 未深度分析)_: Core methodological foundation; conformal risk control is likely the key framewo
- BooookScore: A systematic exploration of book-length summarization in the era of LLMs _(ICLR 2024, 实验对比, 未深度分析)_: Book-length summarization benchmark; likely used for comparison or as related wo

