---
title: Fostering the Ecosystem of AI for Social Impact Requires Expanding and Strengthening Evaluation Standards
type: paper
paper_level: C
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 扩展AI社会影响评估标准
- N/A (Position Pa
- N/A (Position Paper - No Proposed Technical Method)
- Researchers and reviewers in machin
acceptance: Poster
cited_by: 1
method: N/A (Position Paper - No Proposed Technical Method)
modalities:
- Text
---

# Fostering the Ecosystem of AI for Social Impact Requires Expanding and Strengthening Evaluation Standards

**Topics**: [[T__Benchmark_-_Evaluation]] | **Method**: [[M__N-A]]

> [!tip] 核心洞察
> Researchers and reviewers in machine learning for social impact must adopt both a more expansive conception of social impacts beyond deployment and more rigorous evaluations of the impact of deployed systems.

| 中文题名 | 扩展AI社会影响评估标准 |
| 英文题名 | Fostering the Ecosystem of AI for Social Impact Requires Expanding and Strengthening Evaluation Standards |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2510.18238) |
| 主要任务 | AI社会影响评估标准、基准测试与评估方法论 |
| 主要 baseline | 无（Position Paper，无技术方法） |

> [!abstract] 因为「AI for Social Impact (AISI) 领域过度推崇'新颖ML方法+实际部署'的单一理想项目范式」，作者「分析了AAAI和IJCAI等顶会的评审标准」，提出「扩展社会影响的定义范畴并强化已部署系统的评估严谨性」，在「学术生态激励机制分析」上论证了「当前评价体系的结构性偏差」。

- 指出AAAI Special Track和IJCAI Multi-Year Track的评审准则将部署和方法创新性视为卓越的最清晰路径
- 提出非随机化部署设计（non-randomized deployment designs）和事件研究法（event studies）作为更严谨的评估工具
- 引用 Mostly Harmless Econometrics 和 Causal Inference: The Mixtape 作为因果推断方法论基础

## 背景与动机

AI for Social Impact（AISI）领域近年来快速发展：AAAI设立了Special Track on AI for Social Impact，IJCAI开设了Multi-Year Track on AI and Social Good，全球多所高校（Stanford CS 21SI、Harvard CS 288等）开设了相关课程，Data Science for Social Good等fellowship项目也蓬勃兴起。然而，这一繁荣表象下隐藏着深层的结构性问题——学术评价体系正在扭曲研究者的激励方向。

具体而言，当前主流评审标准将「同时包含新颖ML方法与实际部署」视为理想项目的黄金标准。例如，AAAI Special Track的评审指南明确将部署作为影响力证明的核心；IJCAI Multi-Year Track同样强调项目需展示实际落地。这种设计初衷虽好，却产生了严重的意外后果：研究者被迫追求「全栈式」项目，即使这意味着牺牲方法深度或部署质量。

现有文献对此问题的处理方式各异。[28]（"AI as an intervention"）从医疗领域出发，呼吁采用因果方法评估AI干预效果，但聚焦于技术层面而非制度设计；[13]（"Challenges to reproducibility"）揭示了医疗ML模型的可复现性危机，但未触及评审机制的系统性偏差；[27]（"What is the bureaucratic counterfactual?"）通过社会政策中的算法优先排序研究，展示了评估算法vs人工决策的复杂性，却未提出制度性解决方案。这些工作各自揭示了评估标准的某一侧面，但缺乏对学术评价生态的整体性批判。

这些方法的共同短板在于：它们要么假设研究者已有正确的激励去做好评估，要么仅针对特定应用领域提出技术性修补。作者指出，只要评审标准继续将「部署+新方法」作为最高荣誉，研究者就会持续面临压力——将不成熟的系统匆忙部署、或为已有系统包装微薄的方法创新——而非真正服务合作伙伴的最优需求。这一洞见将讨论从「如何做更好的评估」提升至「什么样的制度能激励更好的评估」。

本文正是要系统分析这一激励机制问题，并提出双管齐下的改革路径：既扩展「社会影响」的定义边界，又强化对已部署系统的评估严谨性。

## 核心创新

核心洞察：学术评价体系的激励结构本身即是问题根源，因为评审标准将「部署+新方法」捆绑为唯一卓越范式，从而系统性地排斥了专注单一维度的深度贡献，从而使「扩展影响定义+强化评估严谨性」的双轨改革成为可能。

| 维度 | 当前主流实践（AAAI/IJCAI） | 本文主张 |
|------|------------------------|---------|
| 影响定义 | 以「实际部署」为核心甚至唯一标准 | 认可方法创新、政策分析、工具建设、知识转化等多元影响路径 |
| 评估方法 | 部署即证明，缺乏因果推断要求 | 强制非随机化部署设计、事件研究法等准实验评估 |
| 项目评价 | 偏好「全栈式」项目（方法+部署） | 允许并奖励专注单一维度的深度工作 |
| 评审标准 | 部署和方法创新性作为「最清晰卓越路径」 | 建立与合作伙伴需求对齐的多元卓越标准 |

## 整体框架

本文作为position paper，其论证框架可理解为以下逻辑结构：

**输入层**：现有AISI学术生态的实证观察——包括AAAI Special Track、IJCAI Multi-Year Track的公开评审指南，以及领域内典型项目案例（如[12]的COVID-19边境检测强化学习系统、[14]的脓毒症早期检测ML系统等）。

**诊断模块（Problem Diagnosis）**：系统分析评审标准的文本表述与激励传导机制。作者提取关键评审准则，识别出「部署+新方法」双重要求的显性偏好，论证这一设计如何通过学术职业激励（tenure、奖项、引用）产生扭曲效应。

**扩展模块（Impact Expansion）**：重构「社会影响」的概念边界。除系统部署外，纳入：(a) 面向特定社会问题的原创方法论贡献；(b) 政策分析与制度设计；(c) 开源工具与基础设施；(d) 知识转化与能力建设。每种路径配以具体案例说明其被低估的价值。

**强化模块（Evaluation Strengthening）**：针对确需部署评估的场景，引入因果推断的严谨工具。核心推荐两种设计：非随机化部署设计（利用自然变异或交错推出估计因果效应）和事件研究法（面板数据中分析干预前后结果轨迹）。

**输出层**：面向研究者、审稿人、会议组织者的可操作改革建议，以及对AISI领域长期健康发展的制度愿景。

```
现有评审指南文本 ──→ 激励结构诊断 ──→ 影响定义扩展
                              │
                              ↓
                    部署场景评估强化（因果推断工具）
                              │
                              ↓
                         制度改革建议
```

## 核心模块与公式推导

作为position paper，本文未提出新的技术模型或损失函数，但其论证依赖两个核心方法论模块的严谨阐述。以下基于作者引用的方法论文献，重构其论证中隐含的方法论框架。

### 模块 1: 非随机化部署设计（Non-randomized Deployment Design）

**直觉**：实际部署中常无法进行理想随机实验，需利用自然产生的变异（如交错推出、地理边界、政策阈值）来识别因果效应。

**Baseline 公式**（理想随机对照试验）：
$$\tau_{ATE} = E[Y_i(1) - Y_i(0)] = E[Y_i | D_i=1] - E[Y_i | D_i=0]$$
符号：$Y_i(d)$ 为个体 $i$ 在接受干预 $d$ 时的潜在结果，$D_i \in \{0,1\}$ 为处理分配。随机化保证 $(Y_i(1), Y_i(0)) \perp D_i$。

**变化点**：AISI 部署中随机化常不可行（伦理限制、操作成本、政治阻力）。Baseline 的均值差异 $E[Y_i|D_i=1] - E[Y_i|D_i=0]$ 因选择偏差而不再识别 $\tau_{ATE}$。

**本文公式（推导）**：
$$\text{Step 1}: \quad Y_{it} = \alpha_i + \lambda_t + \tau D_{it} + \epsilon_{it} \quad \text{（双向固定效应，控制不可观测的个体和时间异质性）}$$
$$\text{Step 2}: \quad \tau_{DID} = (E[Y_{it}|post, treated] - E[Y_{it}|pre, treated]) - (E[Y_{it}|post, control] - E[Y_{it}|pre, control]) \quad \text{（差分中的差分，利用平行趋势假设）}$$
$$\text{最终}: \quad \tau_{RDD} = \lim_{x \text{downarrow} c} E[Y_i|X_i=x] - \lim_{x \text{uparrow} c} E[Y_i|X_i=x] \quad \text{（断点回归，利用分配规则在阈值 $c$ 处的跳跃）}$$

**对应消融**：本文未提供定量消融，但引用 [10]（Mostly Harmless Econometrics）和 [18]（Causal Inference: The Mixtape）作为方法论权威，强调这些设计的识别假设需明确陈述并接受检验。

### 模块 2: 事件研究法（Event Study）

**直觉**：面板数据中，通过分析干预前后多期的结果动态，既可估计动态处理效应，又可检验平行趋势假设——这是差分策略可信性的关键诊断。

**Baseline 公式**（静态双向固定效应）：
$$Y_{it} = \alpha_i + \lambda_t + \beta D_{it} + \epsilon_{it}$$
符号：$\alpha_i$ 为个体固定效应，$\lambda_t$ 为时间固定效应，$D_{it}$ 为处理状态，$\beta$ 被解释为平均处理效应。

**变化点**：静态模型隐含处理效应同质且瞬时的强假设，无法捕捉效应的动态演变，也无法利用干预前趋势进行安慰剂检验。在 AISI 评估中，系统部署的效果往往随时间显现（学习曲线、行为适应、溢出效应）。

**本文公式（推导）**：
$$\text{Step 1}: \quad Y_{it} = \alpha_i + \lambda_t + \sum_{k=-K}^{-2} \beta_k D_{i,t+k} + \sum_{k=0}^{L} \gamma_k D_{i,t+k} + \epsilon_{it} \quad \text{（动态事件研究规范，以 $k=-1$ 为参照期）}$$
$$\text{Step 2}: \quad \hat{\beta}_k \approx 0, \forall k < 0 \quad \text{（平行趋势检验：干预前系数应不显著异于零）}$$
$$\text{最终}: \quad \hat{\gamma}_k = E[Y_{i,t+k} - Y_{i,t-1} | treated] - E[Y_{i,t+k} - Y_{i,t-1} | control] \quad \text{（动态处理效应，含即时与滞后成分）}$$

**对应消融**：引用 [23]（"Visualization, identification, and estimation in the linear panel event-study design"）处理交错处理时机（staggered adoption）下的识别问题，强调传统双向固定效应在异质性处理效应下可能产生偏误，需采用 Callaway-Sant'Anna 或 Sun-Abraham 等纠偏估计量。

### 模块 3: ML-驱动的异质性处理效应估计

**直觉**：AISI 系统的核心价值常在于精准 targeting——对谁有效、在何种情境下有效。传统平均效应掩盖了关键的分配效率信息。

**Baseline 公式**（经典Neyman-Rubin框架）：
$$\tau(x) = E[Y_i(1) - Y_i(0) | X_i = x]$$
符号：$X_i$ 为可观测协变量，$\tau(x)$ 为条件平均处理效应（CATE）。

**变化点**：高维 $X$ 下，传统参数方法（线性交互项）的灵活性不足；同时，ML 估计的 CATE 需配套有效的统计推断，而非仅有点预测。

**本文公式（推导）**：
$$\text{Step 1}: \quad \tau(x) = \mu_1(x) - \mu_0(x), \quad \mu_d(x) = E[Y_i | X_i=x, D_i=d] \quad \text{（通过 outcome regression 估计 CATE）}$$
$$\text{Step 2}: \quad \hat{\tau}(x) = \hat{\mu}_1(x) - \hat{\mu}_0(x), \quad \text{其中 } \hat{\mu}_d \text{ 由 generic ML（随机森林、神经网络等）估计} \quad \text{（ML 灵活性）}$$
$$\text{最终}: \quad \sqrt{n}(\hat{\theta} - \theta_0) \text{xrightarrow}{d} N(0, \Sigma), \quad \theta_0 = E[m(\tau(X_i))] \quad \text{（由 [25] 提供的样本分割/交叉拟合推断，保证 } \sqrt{n}\text{-一致性）}$$

**对应消融**：[25]（Chernozhukov et al., "Statistical inference for heterogeneous treatment effects discovered by generic machine learning in randomized experiments"）的核心贡献——证明通过 sample splitting 和 Neyman-orthogonal scores，generic ML 估计的 CATE 泛函仍可进行有效假设检验。这为 AISI 评估中「用 ML 发现异质性、用计量经济学保证推断有效性」提供了理论基础。

## 实验与分析


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/7120e8a2-ac09-401b-ac9d-a99d9992f0bf/figures/Figure_2.png)
*Figure 2 (comparison): Non-randomized deployment and semi-studies*



本文作为 position paper，不包含传统意义上的定量实验或基准测试。作者未训练任何模型、未报告 accuracy/F1/mAP 等指标，也未与现有方法进行数值对比。其「证据基础」建立在以下三个层面：

**评审指南文本分析**：作者直接引用并分析了 AAAI Special Track 和 IJCAI Multi-Year Track 的公开评审标准，指出两者均将「部署」和「方法新颖性」并列为核心评价维度，且暗示同时满足两者是「最清晰卓越路径」。这一文本证据的置信度较高（0.92），构成了论证的实证锚点。

**案例研究举证**：文中援引多个已发表工作作为激励扭曲的例证。例如，[12]（COVID-19 边境检测的强化学习系统）作为「成功部署」的标杆被频繁引用，但作者暗示此类高知名度项目可能过度代表了领域内的实际评估质量；[14]（脓毒症早期检测的内部与时间验证）则展示了即使发表后，临床 ML 系统仍面临严峻的可复现性挑战。这些案例并非受控实验，而是作为「存在性证明」支持作者的制度批判。

**方法论文献的权威援引**：作者系统性地调用因果推断的经典教材 [10][18] 和前沿方法论文 [23][25]，为其推荐的评估工具建立方法论合法性。特别值得注意的是，[28]（"AI as an intervention"）被识别为与本文核心论点「高度一致」的先前工作——这表明作者并非孤立发声，而是在一个正在形成的学术共识中推进讨论。

**公平性检验**：本文的比较对象并非具体算法，而是现有评审制度本身。因此，「baseline 是否最强」的问题转化为：作者是否充分代表了制度设计者的视角？一个潜在的遗漏是，AAAI/IJCAI 的评审指南可能已在 2023-2024 年间更新，而本文分析基于特定时间截面的文本。作者明确承认的局限包括：未实证验证所提评估标准能否实际改善社会结果（缺乏 counterfactual），且对审稿人如何操作化「扩展影响标准」缺乏具体指导。这些坦诚的局限声明增强了论证的可信度，但也揭示了 position paper 体裁的内在约束——它启动对话，而非终结对话。

## 方法谱系与知识库定位

本文属于 **AI 伦理与社会影响评估** 的方法论-制度批判传统，无单一技术 parent method。其知识谱系可定位于以下三条线索的交汇：

**因果推断传统**：直接继承 Angrist & Pischke [10] 和 Cunningham [18] 的计量经济学方法论，将潜在结果框架、工具变量、断点回归、双重差分等工具引入 AISI 评估讨论。与 [25]（Chernozhukov et al.）的衔接则代表了「ML + 计量经济学」的融合趋势。

**医疗 AI 评估批判**：与 [13][14][28] 形成紧密的对话网络。特别是 [28]（"AI as an intervention"）几乎可视为本文在技术主张上的「平行先行者」——两者均强调因果方法，但本文将讨论从医疗领域扩展至更广泛的社会影响场景，并增加了制度分析维度。

**算法公平与测量**：与 [26]（"Measurement and fairness"）和 [24]（行业从业者公平需求研究）共享对「评估什么」和「如何评估」的根本关切。本文的独特贡献在于将这一关切从技术指标提升至学术激励机制。

**直接关联工作**：
- [15]（"Evaluating the effectiveness of index-based treatment allocation"）：指数型分配方法的评估，本文扩展至更一般的 ML 系统
- [16]（"Just trial once"）：持续性因果验证，本文呼应其「评估不应止于一次试验」的主张
- [17]（"A validity perspective"）：数据驱动决策算法的有效性评估框架，本文从制度层面补充其技术框架
- [27]（"What is the bureaucratic counterfactual?"）：社会政策中的算法 vs 人工决策，本文为其经验发现提供制度解释

**后续方向**：(1) 在特定 AISI 子领域（如教育、环境、公共卫生）操作化本文的评估标准并实证检验；(2) 开发辅助审稿人应用扩展影响标准的决策支持工具；(3) 追踪 AAAI/IJCAI 等会议评审指南的后续修订，评估本文的实际政策影响。

**标签**：modality=text | paradigm=因果推断/准实验设计 | scenario=学术评价制度/社会影响评估 | mechanism=激励机制分析/评审标准改革 | constraint=无技术实现/纯论证性工作

