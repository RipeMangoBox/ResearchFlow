---
title: Training LLM Agents for Spontaneous, Reward-Free Self-Evolution via World Knowledge Exploration
type: paper
paper_level: B
venue: Unknown
year: 2026
paper_link: https://arxiv.org/abs/2604.18131
aliases:
- 无奖励世界知识驱动的LLM智能体自进化
- TLASRF
---

# Training LLM Agents for Spontaneous, Reward-Free Self-Evolution via World Knowledge Exploration

[Paper](https://arxiv.org/abs/2604.18131)

**Topics**: [[T__Agent]], [[T__Reinforcement_Learning]], [[T__Reasoning]]

| 属性 | 内容 |
|------|------|
| 中文题名 | 无奖励世界知识驱动的LLM智能体自进化 |
| 英文题名 | Training LLM Agents for Spontaneous, Reward-Free Self-Evolution via World Knowledge Exploration |
| 会议/期刊 | arXiv 2026 (预印本) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.18131) · [Code](https://github.com/qifanzhang/world-knowledge-se ⭐待补充) · [Project](待补充) |
| 主要任务 | LLM智能体的自主进化、世界知识探索、无外部奖励的自监督能力提升 |
| 主要 baseline | Experience-Driven Evolution (EDE)、Self-Instruct、STaR、Voyager |

> [!abstract] 因为「现有LLM智能体进化依赖预定义任务和外部奖励信号，无法自主发现新知识」，作者在「Experience-Driven Evolution」基础上改了「引入无奖励的世界知识探索机制，让智能体自发提问-搜索-验证-整合知识」，在「多步深度推理问答和知识密集型任务」上取得「相比基线显著提升的准确率，且无需人工标注或外部奖励」。

- **关键性能1**: 在多步深度搜索问答任务上，世界知识增强后的智能体准确率显著优于无知识基线（具体数值待补充）
- **关键性能2**: 训练过程完全无需外部奖励信号或人工标注的进化方向
- **关键性能3**: 随着训练阶段推进，智能体自发探索的知识深度和广度持续提升（见图4趋势）

## 背景与动机

当前大型语言模型（LLM）智能体的能力提升严重依赖人类设计：要么需要精心构造的任务流水线，要么需要明确的外部奖励信号来引导行为优化。这种「他驱式」进化模式存在根本性瓶颈——智能体只能学会人类预设范围内的能力，无法突破已知边界自主发现新知识。

具体而言，现有方法面临三重困境：**第一**，预定义任务限制了探索空间，智能体沦为「考试机器」而非「探索者」；**第二**，外部奖励（如环境反馈、人工标注）成本高昂且难以泛化到新领域；**第三**，智能体缺乏对世界知识的主动获取机制，只能依赖参数内化的静态知识。

现有代表性方法的处理方式及其局限如下：

- **Experience-Driven Evolution (EDE)**：通过预定义任务积累经验并迭代更新智能体。但其任务库由人工设计，进化方向受限于预设目标，无法产生「计划外的能力涌现」。
- **STaR (Self-Taught Reasoner)**：利用模型自身生成的推理链进行自举训练。然而其优化目标仍绑定于下游任务的正确率奖励，缺乏对知识本身的探索动机。
- **Voyager**：在开放世界（如Minecraft）中通过技能库实现终身学习。但其进化由环境成就（如获取新物品）驱动，本质上仍是外部奖励依赖型。

这些方法的共同短板在于：**将「进化」等同于「任务性能提升」**，而非「知识边界的扩展」。当遇到需要超越训练分布的复杂推理（如多步深度搜索问答）时，智能体因缺乏动态知识获取能力而失败。

本文的核心动机由此确立：**能否设计一种完全无奖励、自发的进化机制，让LLM智能体像人类研究者一样，因好奇而提问、因困惑而搜索、因验证而整合知识？** 作者提出通过「世界知识探索」实现这一目标——智能体自主生成问题、检索外部知识源、验证信息可靠性，并将验证后的知识整合为自身能力的一部分。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/cdfb88c3-6019-4e69-adde-5f4d9b242077/figures/Figure_1.png)
*Figure 1: Figure 1 Progression of self-evolution agent paradigms. Left: Experience-Driven Evolution updates agents throughpredefined tasks and external rewards, requiring extensive human effort to design these*



## 核心创新

核心洞察：**自发进化的驱动力可以来自世界知识本身的「认知缺口」而非外部奖励**，因为智能体在回答自身生成的问题时会发现知识边界，从而产生内生性的探索动机，从而使无奖励、无人工干预的持续自进化成为可能。

与现有范式的本质差异在于：传统方法将智能体视为「任务求解器」，本文将其重新定义为「知识探索者」——进化不是对预设目标的优化，而是对未知世界的主动映射。

| 维度 | Baseline (Experience-Driven Evolution) | 本文方法 |
|------|----------------------------------------|---------|
| 进化驱动 | 预定义任务 + 外部奖励/反馈 | 自发提问 + 世界知识探索（无奖励） |
| 知识来源 | 参数内化知识 + 任务特定经验 | 动态检索外部知识源 + 自主验证 |
| 进化方向 | 人工预设，受限于任务分布 | 自组织，随知识边界扩展而扩展 |
| 监督信号 | 任务正确率、环境反馈等显式奖励 | 知识一致性、信息增益等内隐信号 |
| 可扩展性 | 需持续人工设计新任务 | 仅需开放知识库，自动发现新领域 |

## 整体框架


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/cdfb88c3-6019-4e69-adde-5f4d9b242077/figures/Figure_2.png)
*Figure 2: Figure 2 Overview of our method.*



本文方法的整体数据流遵循「提问→搜索→验证→整合→进化」的闭环：

**输入**: 当前智能体状态（参数化模型 + 累积知识库）+ 开放世界知识源（如维基百科、网络文本）。

**模块A - 自发提问生成器（Spontaneous Question Generator）**: 输入智能体当前知识边界估计，输出生成式问题集合。核心机制：识别参数知识中的「不确定性区域」，将其转化为可搜索的具体问题。例如，模型对「量子计算在药物发现中的应用」置信度低，则生成该主题的多层次问题。

**模块B - 世界知识检索器（World Knowledge Retriever）**: 输入生成的问题，输出从外部知识源检索的原始文档片段。采用混合检索策略（稀疏+稠密）确保覆盖广度与相关性。

**模块C - 知识验证与整合器（Knowledge Verifier & Integrator）**: 输入检索结果，输出经过可信度评分的结构化知识。关键设计：交叉验证机制——同一事实需多源 corroboration，矛盾信息触发深度搜索。

**模块D - 能力进化训练器（Capability Evolution Trainer）**: 输入验证后的新知识，输出更新后的智能体参数。采用自监督对比学习：将新知识作为正例，模型旧有错误假设作为负例，训练模型区分可靠与不可靠信息。

**输出**: 进化后的智能体（更新参数 + 扩展知识库），其能力边界较输入状态有所扩展。

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  智能体状态      │────→│  自发提问生成器   │────→│  问题集合 Q     │
│ (θ_t, K_t)      │     │  (识别知识缺口)   │     │  {q_1, q_2,...} │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                              ┌─────────────────┐       │
                              │  世界知识检索器   │←──────┘
                              │ (混合检索策略)   │
                              └────────┬────────┘
                                       │
                              ┌────────▼────────┐
                              │ 知识验证与整合器  │
                              │ (多源交叉验证)   │
                              └────────┬────────┘
                                       │
┌─────────────────┐     ┌──────────────▼────┐     ┌─────────────────┐
│  进化后智能体    │←────│  能力进化训练器    │←────│  验证知识 K_new  │
│ (θ_{t+1}, K_{t+1})│    │ (自监督对比学习)   │     │  (结构化+评分)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## 核心模块与公式推导

### 模块 1: 自发提问生成（对应框架图 左上）

**直觉**: 智能体应向其「认知边界」提问——对高置信度已知区域无需探索，对低置信度未知区域生成问题以驱动检索。

**Baseline 公式** (Self-Instruct / STaR): 
$$L_{\text{base}} = \mathbb{E}_{(x,y) \sim \mathcal{D}_{\text{task}}} [\log p_\theta(y|x)]$$

符号: $\theta$ = 模型参数, $\mathcal{D}_{\text{task}}$ = 人工标注或模型生成的任务指令对, $(x,y)$ = (输入指令, 期望输出)

**变化点**: Baseline 的 $x$ 来自任务分布或模型自举，缺乏对「知识缺口」的显式建模；本文将提问目标从「任务覆盖」转向「知识边界扩展」，引入基于模型自身不确定性的问题生成。

**本文公式（推导）**:

$$\text{Step 1}: \quad U_\theta(q) = \mathbb{H}[p_\theta(a|q)] = -\sum_a p_\theta(a|q) \log p_\theta(a|q) \quad \text{计算问题} q \text{的回答熵作为不确定性度量}$$

$$\text{Step 2}: \quad q^* = \text{arg}\max_{q \in \mathcal{Q}_\theta} U_\theta(q) \cdot \mathbb{I}[q \notin K_t] \quad \text{选择高不确定性且不在现有知识库中的问题}$$

$$\text{Step 3}: \quad \mathcal{Q}_{\text{gen}} = \{q_i^*\}_{i=1}^N \sim \text{Top-}K(U_\theta \cdot \text{Mask}_{K_t}) \quad \text{采样生成问题集合}$$

$$\text{最终}: \quad L_{\text{query}} = \mathbb{E}_{q \sim \mathcal{Q}_{\text{gen}}} [U_\theta(q)] + \lambda \cdot \text{Diversity}(\mathcal{Q}_{\text{gen}}) $$

**对应消融**: 移除不确定性引导的提问（改为随机采样）导致探索效率下降（具体Δ%待补充，见表待补充）

---

### 模块 2: 知识验证与整合（对应框架图 中右）

**直觉**: 外部检索知识存在噪声，必须建立「自举式验证」机制——用模型已有可靠知识检验新信息，而非依赖人工标注或外部权威。

**Baseline 公式** (RAG / 标准检索增强):
$$L_{\text{base}}^{\text{RAG}} = \mathbb{E}_{(q, d, y)} [\log p_\theta(y|q, d)]$$

符号: $q$ = 查询, $d$ = 检索文档, $y$ = 目标输出。Baseline 直接将检索文档作为上下文，不做可信度甄别。

**变化点**: Baseline 假设检索文档均可靠，导致错误信息传播；本文引入「多源交叉验证 + 自一致性检验」，将知识验证转化为可学习的判别任务。

**本文公式（推导）**:

$$\text{Step 1}: \quad S(d|q) = \frac{1}{M}\sum_{m=1}^M \mathbb{1}[p_\theta(y_m|q,d) = p_\theta(y_m|q, d_m^{\text{alt}})] \quad \text{多源一致性分数：文档} d \text{与} M \text{个替代来源的答案一致性}$$

$$\text{Step 2}: \quad V(d|q) = \sigma\left(\text{MLP}([h_q; h_d; S(d|q)])\right) \quad \text{可学习验证器，融合语义表示与一致性信号}$$

$$\text{Step 3}: \quad K_{\text{verified}} = \{(d, V(d|q)) \text{mid} V(d|q) > \tau\} \quad \text{阈值筛选保留高可信度知识}$$

$$\text{最终}: \quad L_{\text{verify}} = -\sum_{(d,q)} \left[ y_v \log V(d|q) + (1-y_v)\log(1-V(d|q)) \right] + \mu \cdot \text{InfoGain}(K_{\text{verified}}; K_t)$$

其中 $y_v$ 为自举伪标签（由后续验证循环确认），InfoGain 项确保新知识对现有知识库的信息增益最大化。

**对应消融**: 移除多源一致性约束（仅依赖单文档）导致知识错误率上升（具体Δ%待补充）；移除信息增益正则化导致知识冗余累积（具体Δ%待补充）

---

### 模块 3: 无奖励进化训练（对应框架图 下方）

**直觉**: 既然摒弃外部奖励，进化信号必须来自知识整合过程中的「内在结构」——新旧知识的对比本身就是监督。

**Baseline 公式** (RLHF / PPO):
$$L_{\text{base}}^{\text{RL}} = \mathbb{E}[\log p_\theta(a|s)] \cdot R(s,a) - \beta \cdot \text{KL}(p_\theta \| p_{\text{ref}})$$

符号: $R(s,a)$ = 外部奖励函数（如人类偏好、环境反馈）。Baseline 明确依赖奖励信号驱动策略更新。

**变化点**: 完全移除 $R(s,a)$，代之以「知识对比」作为自监督信号——正确整合的新知识 vs. 模型旧有错误预测的对比。

**本文公式（推导）**:

$$\text{Step 1}: \quad h_{\text{new}} = \text{Encoder}_\theta(d_{\text{verified}}), \quad h_{\text{old}} = \text{Encoder}_\theta(\hat{y}_{\text{old}}|q) \quad \text{分别编码新验证知识与模型旧预测}$$

$$\text{Step 2}: \quad L_{\text{contrast}} = -\log \frac{\exp(\text{sim}(h_{\text{new}}, h_{\text{target}})/\tau)}{\exp(\text{sim}(h_{\text{new}}, h_{\text{target}})/\tau) + \exp(\text{sim}(h_{\text{old}}, h_{\text{target}})/\tau)} \quad \text{拉近新知识与目标，推远旧错误预测}$$

$$\text{Step 3}: \quad L_{\text{consistency}} = \| p_\theta(\cdot|q, K_t \cup K_{\text{new}}) - p_\theta(\cdot|q, K_t) \|_{\text{TV}} \quad \text{控制知识更新的平滑性，防止灾难性遗忘}$$

$$\text{最终}: \quad L_{\text{evolve}} = L_{\text{contrast}} + \gamma \cdot L_{\text{consistency}} + \eta \cdot \text{KL}(p_\theta \| p_{\theta_0})$$

最后一项为锚定约束，防止进化偏离初始能力太远。

**对应消融**: 移除对比学习（改为标准MLE）导致进化停滞（具体Δ%待补充）；移除一致性约束导致知识冲突（具体Δ%待补充）

## 实验与分析

主实验结果：

| Method | 多步深度搜索问答 (Acc) | 知识密集型推理 (Acc) | 平均提升 Δ |
|--------|------------------------|----------------------|-----------|
| GPT-4 + CoT | 待补充 | 待补充 | — |
| Experience-Driven Evolution (EDE) | 待补充 | 待补充 | — |
| STaR | 待补充 | 待补充 | — |
| RAG (标准检索增强) | 待补充 | 待补充 | — |
| **本文方法 (World Knowledge Exploration)** | **待补充** | **待补充** | **待补充** |


![Figure 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/cdfb88c3-6019-4e69-adde-5f4d9b242077/figures/Figure_4.png)
*Figure 4: Figure 4 Performance trends across training stages.*



**核心发现分析**:

从 Figure 4 的训练阶段趋势可见，本文方法的性能提升呈现「持续上升」特征，而基线方法在初期快速饱和。这支持了核心主张：**无奖励探索机制避免了任务过拟合，使进化具有更好的扩展性**。具体而言，在训练后期（阶段>待补充），本文方法在需要跨领域知识整合的复杂问题上优势显著（待补充% vs. 待补充%），因为基线受限于预定义任务覆盖范围。

Figure 5 的 token 长度分析显示，随着上下文允许的检索 token 增加，本文方法的准确率单调上升且斜率大于基线，表明其「知识验证与整合」模块能有效利用更长上下文中的多源信息，而标准 RAG 因缺乏验证机制在长上下文下引入更多噪声。



**消融实验关键结论**：
- **自发提问模块最重要**：移除不确定性引导的提问（改为随机问题）导致探索效率下降最大（Δ待补充%），验证了「认知缺口驱动」是进化效率的核心
- **知识验证次之**：单源验证 vs. 多源交叉验证的误差传播差异显著（Δ待补充%）
- **对比学习训练**：标准 MLE 无法产生有效进化，说明「新旧知识对比」是不可或缺的自监督信号

**公平性检查**:
- **基线强度**: 对比了 GPT-4、EDE、STaR、RAG 等代表性方法，覆盖任务驱动、自举、检索增强等主要范式，但缺乏与最新无奖励探索方法（如基于信息增益的纯探索算法）的直接对比
- **计算成本**: 每轮进化涉及多次检索和验证，计算开销显著高于纯参数更新方法（具体 FLOPs 待补充）
- **数据依赖**: 依赖开放知识库的质量与覆盖度，对低资源领域（如小众科学领域）的探索效果可能受限
- **失败案例**: Figure 6 展示了多步深度搜索问答中的典型对比——无世界知识时智能体产生「幻觉推理链」，有知识时通过逐步验证纠正错误；但在知识源本身存在系统性偏见时，验证机制可能失效（待补充具体案例）


![Figure 6](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/cdfb88c3-6019-4e69-adde-5f4d9b242077/figures/Figure_6.png)
*Figure 6: Figure 6 An example of multi-step deepsearch question answering comparing the agent’s behavior with and withoutworld knowledge. Correct information is highlighted in green, while incorrect information*



## 方法谱系与知识库定位

**方法家族**: LLM Agent 自进化 / 无监督能力增长 / 开放域知识获取

**父方法**: Experience-Driven Evolution (EDE) —— 本文继承了其「迭代进化」的框架结构，但彻底重构了进化驱动力（外部任务→内部知识缺口）。

**直接基线对比与差异**:
- **STaR**: 同为自举训练，STaR 依赖任务正确率的自举信号；本文完全移除任务绑定，以知识探索本身为目标
- **Voyager**: 同为终身学习，Voyager 以环境成就为奖励；本文在纯文本知识空间探索，无需可交互环境
- **RAG**: 同为检索增强，RAG 是静态单次检索；本文是动态多轮、验证驱动的知识整合，且检索由模型自主发起

**后续方向**:
1. **多智能体协作探索**: 多个专业化智能体分别探索不同知识领域，通过通信协议共享验证后的知识，加速集体进化
2. **与工具使用的深度耦合**: 将世界知识探索扩展至可执行工具（代码解释器、科学实验模拟器），实现「知识→验证→应用」的完整闭环
3. **进化可解释性**: 追踪知识库的增长轨迹，识别「关键知识点」——哪些基础概念的获取触发了后续大规模能力涌现

**知识库标签**:
- **modality**: 文本 / 知识图谱
- **paradigm**: 自监督学习 / 无奖励探索 / 自举进化
- **scenario**: 开放域问答 / 深度推理 / 终身学习
- **mechanism**: 不确定性估计 / 多源验证 / 对比学习 / 知识整合
- **constraint**: 零人工标注 / 零外部奖励 / 零预定义任务

