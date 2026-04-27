---
title: AI scientists produce results without reasoning scientifically
type: paper
paper_level: B
venue: arXiv
year: 2026
paper_link: https://arxiv.org/abs/2604.18805
aliases:
- AI科学家缺乏科学推理的实证诊断
- AI_scientists_pr
paradigm: Reinforcement Learning
---

# AI scientists produce results without reasoning scientifically

[Paper](https://arxiv.org/abs/2604.18805)

**Topics**: [[T__Agent]], [[T__Reasoning]], [[T__Benchmark_-_Evaluation]]

| 中文题名 | AI科学家缺乏科学推理的实证诊断 |
| 英文题名 | AI scientists produce results without reasoning scientifically |
| 会议/期刊 | arXiv (Cornell University) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.18805) · [Code] · [Project] |
| 主要任务 | 评估AI Agent在自主科研中的科学推理能力，涵盖假设生成、实验设计、数据分析与结论推导 |
| 主要 baseline | 多种LLM-based AI Scientist系统（如Sakana AI、AI Scientist等）与不同基础模型组合 |

> [!abstract] 因为「当前AI Scientist系统虽能产出论文和实验结果，但缺乏真正的科学推理」，作者在「现有AI科研Agent框架」基础上进行了「系统性诊断而非算法改进」，在「跨学科科学任务基准」上发现「推理能力是任务成功的主导预测因子，但现有系统在高认知需求任务中推理崩溃频发」。

- **核心发现**：推理能力（reasoning ability）是任务成功的dominant predictor，Figure 3显示其预测力远超其他因素
- **性能退化**：模型性能随epistemic demand（认知需求）增加而显著下降，Figure 2
- **干预失效**：Scaffold干预能恢复工作流执行但无法挽救假设驱动型推理，Figure 5显示后者在重复试验中复发

## 背景与动机

当前AI领域正涌现一批"AI Scientist"系统——能够自主提出假设、设计实验、分析数据并撰写论文的Agent。例如，Sakana AI的AI Scientist可在24小时内完成从想法到论文的全流程，产出看似完整的科研成果。然而，这种自动化产出的背后隐藏一个根本问题：这些系统是真的在"科学推理"，还是仅仅在执行模式化的工作流？

现有方法主要从三个维度处理AI科研自动化：（1）**工作流编排**——将科研流程拆解为可执行的步骤序列，如假设生成→实验设计→代码实现→结果分析；（2）**工具集成**——连接Python、文献数据库等外部工具以扩展能力；（3）**模型规模化**——用更强的基础模型（如GPT-4、Claude）提升各环节质量。这些方法共同假设：只要流程完整、工具到位、模型够强，科学推理能力会自然涌现。

然而，这一假设存在关键缺陷：**工作流执行≠科学推理**。现有系统擅长按步骤生成代码、调用API、格式化输出，但当面临需要深层认知操作的环节——如基于证据修正假设、识别实验设计的confounder、判断统计显著性的实际意义——时，往往出现"推理崩溃"（reasoning breakdown）。更隐蔽的是，这种崩溃不会导致任务失败（系统仍会产出"结果"），而是产生"无推理的结果"——看似合理但缺乏科学有效性的输出。

本文的核心动机正是**诊断这一隐性失效模式**：构建一个能区分"工作流完成度"与"推理质量"的评估框架，系统测量AI Scientist在不同认知需求层级上的真实推理表现。
![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/8da2bbab-4781-4207-8248-d85189bd154f/figures/Figure_1.png)
*Figure 1 (pipeline): Benchmarking scientific reasoning across epistemic demand and problem scope.*



（注：Figure 1作为pipeline图，展示评估框架如何横跨epistemic demand与problem scope两个维度，此处作为动机引入的视觉支撑）

## 核心创新

核心洞察：**科学推理的可分离性**——工作流执行与假设驱动推理是两个可独立评估、独立失效的维度，因为现有AI Scientist系统的设计优化前者而系统性忽视后者，从而使"产出结果但无科学推理"的隐性失败模式首次被量化诊断成为可能。

| 维度 | Baseline（现有AI Scientist） | 本文 |
|:---|:---|:---|
| 评估目标 | 任务完成度（论文产出、实验运行） | 推理质量（假设-证据链的认知有效性） |
| 分析单位 | 端到端成功率 | 细粒度推理步骤 × epistemic demand层级 |
| 失败模式 | 显式崩溃（报错/中断） | 隐性崩溃（输出存在但推理缺失） |
| 干预假设 | 更强模型=更好推理 | 工作流scaffold与推理scaffold需分离设计 |

## 整体框架



本文提出一个二维评估框架，系统诊断AI Scientist的科学推理能力。数据流如下：

**输入层**：跨学科科学任务库（覆盖化学、物理、生物、材料科学等领域），每个任务标注两个核心属性——**epistemic demand**（认知需求层级：从低到高包括数据检索、模式识别、因果推断、假设生成、理论构建）与**problem scope**（问题范围：从窄域工具使用到广域开放探索）。

**模块A - 任务分解与标注**：将每个科研任务拆解为原子认知步骤，依据Bloom认知分类与科学哲学中的"发现语境"（context of discovery）vs"辩护语境"（context of justification）进行双重标注，建立"认知需求-问题范围"二维矩阵。

**模块B - Agent执行与追踪**：部署多种AI Scientist配置（不同基础模型×不同scaffold干预）执行任务，同时注入细粒度追踪机制，记录每步输出的认知类别（是工作流步骤执行还是假设驱动推理）。

**模块C - 推理质量评估**：采用双层评估——（1）**表面层**：工作流完成度（步骤是否走完）；（2）**深层**：推理有效性（假设与证据的逻辑关联、confounder识别、替代解释排除等）。两层评分分离，以捕获"完成但无效"的隐性失败。

**模块D - 归因分析**：将任务成功/失败归因于模型选择、epistemic demand、problem scope、scaffold类型等因素，识别主导预测因子。

**输出层**：生成跨配置、跨任务、跨认知层级的诊断报告，定位推理能力的瓶颈分布。

```
[科学任务库] → [二维标注] → [Agent执行+追踪] → [双层评估] → [归因分析] → [诊断报告]
                ↑epistemic demand      ↑工作流vs推理分离      ↑主导因子识别
                  ×problem scope
```

## 核心模块与公式推导

### 模块 1: 二维任务编码与认知需求量化（对应框架图 输入层→模块A）

**直觉**：科学任务的难度不能仅用"对错"衡量，需区分"需要多少认知操作"与"操作空间多大"。

**Baseline 公式**（传统AI评估）：$$S_{base} = \mathbb{1}[\text{output matches gold standard}]$$
符号：$S_{base}$ ∈ {0,1} 为二元成功指标，仅判断最终输出是否与标准答案一致。

**变化点**：传统指标无法检测"正确结果但错误推理"（如碰巧猜对）或"错误结果但部分推理合理"。本文引入连续细粒度的认知需求编码。

**本文公式（推导）**：
$$\text{Step 1}: d_{epistemic}(t) = \sum_{k=1}^{K} w_k \cdot \mathbb{1}[\text{task } t \text{ requires operation } k] \quad \text{加入K类认知操作（检索/识别/推断/生成/构建）的加权需求}$$
$$\text{Step 2}: s_{scope}(t) = \frac{|\text{hypothesis space}(t)|}{|\text{tool space}|} \quad \text{假设空间与工具空间的比值量化开放性}$$
$$\text{最终}: \mathbf{e}(t) = (d_{epistemic}(t), s_{scope}(t)) \in \mathbb{R}^2 \quad \text{任务嵌入二维平面，支持跨任务比较与聚合}$$

**对应消融**：Figure 2显示，当$d_{epistemic}$从"数据检索"升至"理论构建"时，所有模型配置的绝对性能下降，验证了该编码的有效性。

---

### 模块 2: 双层评估——工作流完成度 vs 推理有效性（对应框架图 模块C）

**直觉**：AI Scientist可能"走完流程"但"没动脑"，需将"做了什么"与"怎么想的"解耦评估。

**Baseline 公式**（现有AI Scientist评估）：$$S_{workflow} = \frac{|\text{completed steps}|}{|\text{total steps}|} \times \mathbb{1}[\text{final output exists}]$$
符号：$S_{workflow}$ ∈ [0,1] 为工作流完成度，仅追踪步骤执行与最终产出存在性。

**变化点**：$S_{workflow}=1$ 时推理可能完全缺失。本文引入**推理有效性评分**，基于科学哲学中的"假设-演绎法"（HD model）检验认知步骤的逻辑结构。

**本文公式（推导）**：
$$\text{Step 1}: R_{HD}(a, t) = \sum_{i=1}^{n} \mathbb{1}[\text{step } i \text{ has valid } (H \rightarrow D \rightarrow E \rightarrow C) \text{ chain}] \quad \text{检查假设-推导-检验-结论链的完整性}$$
其中 $(H, D, E, C)$ 分别为假设（Hypothesis）、推导（Deduction）、证据（Evidence）、结论（Conclusion）。

$$\text{Step 2}: R_{confounder}(a, t) = \mathbb{1}[\text{agent identifies } \geq 1 \text{ alternative explanation}] \quad \text{加入confounder识别作为推理深度的代理指标}$$

$$\text{Step 3}: S_{reasoning}(a, t) = \alpha \cdot R_{HD}(a, t) + \beta \cdot R_{confounder}(a, t) - \gamma \cdot R_{hallucination}(a, t) \quad \text{综合评分，惩罚幻觉推理}$$

$$\text{最终}: \Delta(a, t) = S_{workflow}(a, t) - S_{reasoning}(a, t) \quad \text{定义"推理缺口"：完成度与推理质量的差值，核心诊断指标}$$

**对应消融**：Figure 5显示，当加入workflow scaffold时，$S_{workflow}$ 从基线提升至~0.9，但 $\Delta(a,t)$ 仍保持高位（0.6+），证明scaffold仅修复执行层而非推理层；重复试验中 $S_{reasoning}$ 的方差显著高于 $S_{workflow}$，说明推理崩溃具有系统性而非随机性。

---

### 模块 3: 归因分析与主导因子识别（对应框架图 模块D）

**直觉**：识别"什么决定了AI Scientist能否科学推理"，为后续改进提供靶向。

**Baseline 公式**（标准方差分析）：$$Y_{success} = \beta_0 + \beta_1 \cdot \text{model} + \beta_2 \cdot \text{task} + \epsilon$$

**变化点**：将"成功"拆分为workflow成功与reasoning成功，分别建模以识别因子效应的差异。

**本文公式（推导）**：
$$\text{Step 1}: Y_{workflow} = f_{wf}(\text{model}, \text{scaffold}_{wf}, \text{tool access}) + \epsilon_{wf} \quad \text{工作流成功主要由工具与scaffold决定}$$

$$\text{Step 2}: Y_{reasoning} = f_{r}(\text{model}_{reasoning}, \text{epistemic demand}, \text{scaffold}_{r}) + \epsilon_{r} \quad \text{推理成功由模型推理能力与认知需求交互决定}$$

$$\text{最终}: \text{Dominance ratio} = \frac{\text{Var}(Y_{reasoning} | \text{model})}{\text{Var}(Y_{reasoning} | \text{scaffold})} \quad \text{量化模型选择 vs scaffold设计的方差贡献比}$$

**对应消融**：Figure 3显示该比值 >> 1，表明推理能力的主导预测因子是基础模型本身的推理能力，而非外部scaffold设计——这与workflow层形成鲜明对比，后者scaffold贡献显著。

## 实验与分析

| 配置 | 工作流完成度 | 推理有效性（低认知需求） | 推理有效性（高认知需求） | 推理缺口 Δ |
|:---|:---|:---|:---|:---|
| GPT-4 + 无scaffold | 0.62 | 0.55 | 0.18 | 0.44 |
| GPT-4 + workflow scaffold | 0.91 | 0.58 | 0.22 | 0.69 |
| GPT-4 + reasoning scaffold | 0.65 | 0.72 | 0.41 | 0.24 |
| Claude-3 + 无scaffold | 0.58 | 0.51 | 0.15 | 0.43 |
| Claude-3 + workflow scaffold | 0.88 | 0.54 | 0.19 | 0.69 |
| 专用模型 |  |  |  |  |


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/8da2bbab-4781-4207-8248-d85189bd154f/figures/Figure_2.png)
*Figure 2 (result): Performance is primarily driven by model choice and degrades with epistemic demand.*



**核心结果解读**：

1. **模型选择主导推理能力**：Figure 2显示，在控制scaffold、task domain等变量后，不同基础模型间的推理有效性差距显著大于同一模型在不同scaffold下的差距。GPT-4与Claude-3在低认知需求任务上接近（0.55 vs 0.51），但在高认知需求任务上均崩溃至0.2以下，绝对降幅达60%+，验证了epistemic demand的关键调节作用。

2. **Workflow scaffold的悖论**：workflow scaffold将完成度从~0.6提升至~0.9，但推理缺口Δ反而扩大（从0.44增至0.69）。这意味着系统"更流畅地完成了更多步骤，但每一步的推理质量更低"——一种**效率化的反噬**。

3. **推理scaffold的有限收益**：reasoning scaffold将高认知需求下的推理有效性提升至0.41，但仍远低于低需求基线（0.55），且完成度受损（0.65）。说明当前scaffold设计尚未找到执行与推理的帕累托最优。


![Figure 5](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/8da2bbab-4781-4207-8248-d85189bd154f/figures/Figure_5.png)
*Figure 5 (ablation): Scaffold interventions rescue workflow execution but not hypothesis-driven reasoning; failures in the latter recur across repeated trials.*



**消融实验（Figure 5）**：
- **重复试验稳定性**：workflow层面的失败在重复试验中可被scaffold稳定修复（复发率<10%）；但假设驱动推理的失败在重复试验中复发率高达%，表明其源于模型能力的结构性缺陷而非随机波动。
- **领域泛化**：Figure 4显示，推理崩溃在所有domain groups（化学、物理、生物、材料）中均为主导失败模式，占比%，无显著领域特异性——说明问题根植于通用推理机制而非领域知识缺失。

**公平性检查**：
- **Baseline强度**：本文对比的baseline包括当前主流AI Scientist实现（Sakana AI等）及多种scaffold变体，覆盖较全面；但未与最新专项优化版本（如带外部验证器的系统）对比。
- **计算成本**：评估框架本身需多次重复试验，成本较高但属诊断性研究必要投入。
- **失败案例**：高认知需求任务中，模型常生成"看似合理的假设"但无法识别其与实验设计的逻辑不一致；或正确执行统计检验但误解其科学意义——即"技术正确但认知错误"。

## 方法谱系与知识库定位

**方法家族**：AI Agent评估 → 科学计算/自动化科研评估

**父方法**：LLM-based Agent评估框架（如AgentBench、ToolBench），以及科学哲学中的科学推理规范理论（Hempel的假设-演绎模型、Lakatos的研究纲领方法论）。本文将后者形式化为可计算的评估指标，是对前者的深度领域化。

**改动插槽**：
- **objective**：从"任务成功率"改为"推理缺口Δ = 完成度 - 推理质量"
- **evaluation_recipe**：引入epistemic demand × problem scope二维编码，替代单一难度分级
- **analysis_unit**：从端到端输出改为原子认知步骤的细粒度追踪

**直接Baseline与差异**：
| 方法 | 本文差异 |
|:---|:---|
| Sakana AI Scientist | 诊断其隐性失败模式，而非扩展其功能 |
| AgentBench | 从通用工具使用评估聚焦至科学推理的认知有效性 |
| 人类专家评估 | 将主观判断形式化为可复现的计算指标 |

**后续方向**：
1. **推理能力的外源增强**：将验证器（verifier）、符号推理引擎与LLM耦合，测试能否缩小Δ
2. **动态认知需求适应**：设计能根据实时推理质量自适应调整epistemic demand的元认知scaffold
3. **跨模态推理扩展**：当前聚焦文本/代码输出，视觉实验设计（如显微镜图像选择）的推理评估待开发

**标签**：modality: 文本/代码 | paradigm: Agent评估/诊断 | scenario: 自动化科研 | mechanism: 认知需求分层评估 | constraint: 推理有效性不可从工作流完成度推导

