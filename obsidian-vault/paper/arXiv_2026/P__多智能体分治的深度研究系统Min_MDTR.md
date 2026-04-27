---
title: Mind DeepResearch Technical Report
type: paper
paper_level: B
venue: arXiv
year: 2026
paper_link: https://arxiv.org/abs/2604.14518
aliases:
- 多智能体分治的深度研究系统MindDR
- MDTR
- MindDR的核心直觉是：深度研究任务的复杂性源于搜索、规划、报告生成
---

# Mind DeepResearch Technical Report

[Paper](https://arxiv.org/abs/2604.14518)

**Topics**: [[T__Agent]], [[T__Benchmark_-_Evaluation]], [[T__Reasoning]]

> [!tip] 核心洞察
> MindDR的核心直觉是：深度研究任务的复杂性源于搜索、规划、报告生成三个子任务的目标异质性——将它们压缩进单一模型和单一RL目标会导致训练信号稀疏且相互干扰。通过将任务分解给三个专职智能体，并为每个智能体设计专项训练阶段（Search-RL优化检索效率，Report-RL优化报告质量），MindDR实现了'分而治之'的优化策略。步骤级信用分配解决了长链条任务中关键步骤信号被稀释的问题，使~30B参数模型能够在有限计算预算下逼近更大规模模型的性能上界。

| 中文题名 | 多智能体分治的深度研究系统MindDR |
| 英文题名 | Mind DeepResearch Technical Report |
| 会议/期刊 | arXiv (Cornell University) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.14518) · [Code]() · [Project]() |
| 主要任务 | 深度研究智能体（Deep Research Agent）：复杂长链条信息检索与长篇报告生成 |
| 主要 baseline | GPT-4级别超大规模模型、单体端到端RL系统、ARPO、TreeRL、Nanbeige4.1系列 |

> [!abstract] 因为「深度研究任务中搜索、规划、报告生成目标异质，单体端到端RL导致训练信号稀疏且相互干扰」，作者在「单体端到端深度研究智能体」基础上改了「三智能体协作架构 + 四阶段专项训练（Search-RL步骤级信用分配 + Report-RL四维RACE奖励） + 知识图谱驱动数据合成」，在「BrowseComp-ZH/WideSearch/xbench-DS/DeepResearch Bench/MindDR Bench」上取得「BrowseComp-ZH 45.7%、BrowseComp 42.8%、WideSearch 46.5%、xbench-DS 75.0%、DeepResearch Bench 52.5%、MindDR Bench RACE 51.8（SOTA）」

- **关键性能 1**: MindDR-v1.5-30B-A3B 在 BrowseComp-ZH 达 45.7%，BrowseComp 达 42.8%，以 ~30B 参数逼近更大规模模型性能
- **关键性能 2**: MindDR Bench RACE 得分 51.8，宣称 SOTA，采用 comprehensiveness/insight/instruction_following/readability 四维加权评估
- **关键性能 3**: xbench-DS 75.0%、DeepResearch Bench 52.5%、WideSearch 46.5%，覆盖中英文多基准

## 背景与动机

深度研究智能体（Deep Research Agent）旨在完成复杂的长链条信息检索与报告生成任务——例如用户提问"对比2024年中美新能源汽车政策对锂资源供应链的影响"，系统需自主规划检索路径、执行数十次搜索、聚合多源证据并输出结构化的长篇分析报告。这类任务的核心难点在于：检索链条长（数十次工具调用）、信息源异质（网页、数据库、知识库）、输出质量维度多（准确性、全面性、可读性、时效性）。

现有系统主要从三个方向切入，但各有明显局限：

**单体端到端强化学习**（如早期 GPT-4 based Deep Research）将搜索、规划、报告生成统一建模为单一RL目标。这类方法依赖超大规模模型（GPT-4级别）提供足够的上下文容量和涌现能力，但训练信号稀疏——单次轨迹的成败难以归因到具体检索步骤，且长上下文负担（数十次调用的累积上下文）导致推理成本极高。

**步骤级信用分配方法**（如 PPO）试图通过critic模型提供细粒度奖励，但critic模型本身需要与策略模型同等规模，计算开销翻倍；分支采样方法（ARPO、TreeRL）在长链条任务中面临指数级采样复杂度，实际不可行。

**单一评估指标**（如传统 RACE 或单一准确率）无法捕捉报告质量的多维需求——一篇报告可能信息准确但结构混乱、或全面但缺乏洞察，现有体系无法区分这些差异。

上述局限共同指向一个关键矛盾：**深度研究任务的子目标异质性（检索效率 vs 报告质量 vs 规划合理性）与单体优化框架的不兼容性**。MindDR的核心动机正是通过结构性拆解与配套训练流程设计，在~30B参数规模下突破这一瓶颈。
![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/75e90f3e-f6d6-495e-b96a-d999f640d475/figures/Figure_2.png)
*Figure 2: Figure 2: Overview of the MindDR multi-agent framework. A user query is first processed by thePlanning Agent, which performs intent analysis and task decomposition to produce a structuredsubtask speci*



## 核心创新

核心洞察：深度研究任务的复杂性源于搜索、规划、报告生成三个子任务的目标异质性——将它们压缩进单一模型和单一RL目标会导致训练信号稀疏且相互干扰；通过将任务分解给三个专职智能体，并为每个智能体设计专项训练阶段，使"分而治之"的优化策略在~30B参数规模下成为可能，因为每个智能体只处理自身职责范围内的上下文，天然缓解了长上下文负担，同时为各阶段提供了清晰的优化目标边界。

| 维度 | Baseline（单体端到端） | 本文（MindDR） |
|:---|:---|:---|
| 架构 | 单一模型承担搜索+规划+报告生成 | Planning/DeepSearch/Report 三智能体协作，上下文隔离 |
| RL目标 | 统一轨迹级奖励，信号稀疏 | Search-RL步骤级信用分配 + Report-RL四维RACE奖励，分阶段专项优化 |
| 信用分配 | 轨迹级广播统一奖励，或依赖昂贵critic | 每步检索对答案覆盖率的增量（Δanswer_coverage）估计优势，无需critic |
| 训练流程 | 单一RL阶段或SFT+RL两阶段 | 四阶段：SFT冷启动 → Search-RL → Report-RL → DPO+Self-SFT偏好对齐 |
| 数据合成 | 单一强约束主导，覆盖有限 | 知识图谱子图采样+条件混淆增强+多阶段过滤，与真实查询混合 |
| 评估 | 单一RACE指标 | MindDR Bench 500条真实查询 + RACE四维加权 + 动态rubrics |

## 整体框架



MindDR的整体框架遵循"用户查询输入 → 规划分解 → 并行检索 → 证据聚合 → 报告生成 → 偏好对齐精炼"的数据流，由三个专职智能体协同完成：

**Planning Agent（规划智能体）**：接收原始用户查询，执行意图分析与任务分解，输出结构化子查询计划。该智能体是系统的"调度器"，决定检索的广度与深度，并通过 Extended Chain-of-Thought（扩展思维链）机制与其他智能体协调。

**DeepSearch Agent（深度搜索智能体）**：接收Planning Agent输出的子查询，执行迭代检索与信息聚合。核心职责包括：动态选择搜索工具、评估检索结果相关性、决定继续深入或终止分支。该智能体是Search-RL的训练对象，优化检索准确率与token效率。

**Report Agent（报告智能体）**：接收DeepSearch Agent聚合的多源证据，生成长篇结构化报告。核心职责包括：信息冲突解决、逻辑组织、格式一致性保障、时效性校验。该智能体是Report-RL的训练对象，以RACE四维评分优化报告质量。

三智能体通过Extended Chain-of-Thought机制协调，实现搜索并行化和上下文隔离——每个智能体只保留自身职责所需的上下文片段，而非累积全部历史调用记录。

训练流程上，四阶段流水线与三智能体架构对齐：SFT冷启动为各智能体提供基础行为模式；Search-RL专项优化DeepSearch Agent的检索决策；Report-RL专项优化Report Agent的生成质量；最终DPO与Self-SFT在线策略自改进修正残留缺陷。

```
用户查询 → [Planning Agent] 意图分析/子查询规划
              ↓
        子查询1...n → [DeepSearch Agent] × n 并行检索
                          ↓
                    聚合证据 → [Report Agent] 长篇报告生成
                                  ↓
                            [DPO+Self-SFT] 偏好对齐精炼 → 最终输出
```

## 核心模块与公式推导

### 模块 1: Search-RL 步骤级信用分配（对应框架图 DeepSearch Agent 核心）

**直觉**: 传统轨迹级奖励对所有检索步骤广播统一信号，导致关键步骤与冗余步骤无法区分；步骤级PPO虽可细化但依赖昂贵critic模型。MindDR通过每步检索对最终答案覆盖率的增量贡献来估计步骤优势，无需额外critic。

**Baseline 公式** (REINFORCE / 轨迹级奖励):
$$L_{\text{trajectory}} = -\mathbb{E}_{\tau \sim \pi_\theta}\left[ R(\tau) \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \right]$$
符号: $\tau = (s_1, a_1, ..., s_T, a_T)$ 为完整检索轨迹, $R(\tau)$ 为轨迹级标量奖励（如最终答案正确性）, $\pi_\theta$ 为策略模型。

**变化点**: 轨迹级 $R(\tau)$ 对步骤 $t$ 的梯度贡献与 $(T-t)$ 步后的动作无关，信用分配粗糙；且 $R(\tau)$ 稀疏（仅轨迹终止时获得）。MindDR改为步骤级增量覆盖估计，利用检索结果与答案实体集合的覆盖变化作为即时信号。

**本文公式（推导）**:
$$\text{Step 1}: \quad \Delta_t^{\text{coverage}} = \text{EntityCoverage}(D_{\leq t}, \text{Answer}) - \text{EntityCoverage}(D_{<t}, \text{Answer})$$
加入了步骤级增量覆盖项以解决轨迹级信号稀疏问题，其中 $D_{\leq t}$ 为截至步骤 $t$ 的累积检索文档集合。

$$\text{Step 2}: \quad \hat{A}_t = \Delta_t^{\text{coverage}} - \lambda \cdot \text{TokenCost}(a_t) \quad \text{（重归一化以保证token效率）}$$
重归一化以保证检索效率与答案质量的联合优化，$\lambda$ 为效率权重超参数。

$$\text{最终}: \quad L_{\text{Search-RL}} = -\mathbb{E}_{\tau \sim \pi_\theta}\left[ \sum_{t=1}^{T} \hat{A}_t \cdot \nabla_\theta \log \pi_\theta(a_t|s_t) \right]$$

**对应消融**: —— 论文未提供三智能体 vs 单智能体、Search-RL独立贡献的消融数据。

### 模块 2: Report-RL 四维RACE奖励优化（对应框架图 Report Agent 核心）

**直觉**: 单一生成指标（如ROUGE或单一RACE）无法同时优化报告的全面性、洞察性、指令遵循度和可读性；MindDR将RACE扩展为四维加权评分，作为Report Agent的专项奖励信号。

**Baseline 公式** (标准RLHF / 单一奖励模型):
$$L_{\text{RLHF}} = -\mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta}\left[ r_\phi(x, y) \cdot \nabla_\theta \log \pi_\theta(y|x) \right]$$
符号: $r_\phi$ 为单一标量奖励模型, $x$ 为检索证据, $y$ 为生成报告。

**变化点**: 单一 $r_\phi$ 隐含假设报告质量可压缩为一维标量，但实际用户关注多维度且权重因查询而异；MindDR显式解耦为四维并动态加权。

**本文公式（推导）**:
$$\text{Step 1}: \quad \text{RACE}_4(y|x, q) = \sum_{k \in \mathcal{K}} w_k(q) \cdot \text{score}_k(y|x), \quad \mathcal{K} = \{\text{comprehensiveness}, \text{insight}, \text{instruction\_following}, \text{readability}\}$$
加入了查询相关的动态权重 $w_k(q)$ 以解决不同查询类型对质量维度需求不同的问题。

$$\text{Step 2}: \quad w_k(q) = \text{Softmax}\left( \frac{\text{Embed}(q) \cdot \text{Embed}(\text{rubric}_k)}{\tau} \right) \quad \text{（动态rubrics匹配）}$$
重归一化以保证权重随查询语义自适应，$\text{rubric}_k$ 为预定义的维度评分细则，$\tau$ 为温度系数。

$$\text{最终}: \quad L_{\text{Report-RL}} = -\mathbb{E}_{x,q \sim \mathcal{D}, y \sim \pi_\theta}\left[ \text{RACE}_4(y|x,q) \cdot \nabla_\theta \log \pi_\theta(y|x,q) \right]$$

**对应消融**: —— 论文未提供四维RACE vs 单一RACE、各维度独立贡献的消融数据。

### 模块 3: 知识图谱驱动的查询合成（对应框架图 数据工程层）

**直觉**: 真实用户查询分布与合成数据存在偏移，单一强约束（如固定跳数QA）导致模型过拟合特定模式；MindDR通过子图采样三约束+条件混淆增强，在可控性与生态有效性之间取得平衡。

**Baseline 方法** (标准知识图谱QA合成):
$$\mathcal{D}_{\text{synth}} = \{(q_i, a_i)\}_{i=1}^{N}, \quad q_i = \text{Template}(\text{Path}(s_i, o_i)), \; s_i, o_i \sim \mathcal{G}$$
符号: $\mathcal{G}$ 为知识图谱, $\text{Path}(s_i, o_i)$ 为采样路径, Template为固定模板。

**变化点**: 标准方法缺乏对推理结构多样性的控制，且模板化生成导致语言分布单一；MindDR引入三约束子图采样与条件混淆增强。

**本文公式（推导）**:
$$\text{Step 1}: \quad \mathcal{S}_{\text{subgraph}} = \{g \subset \mathcal{G} : \underbrace{\text{Reachable}(g, q)}_{\text{推理可达性}} \land \underbrace{\text{Necessary}(g, q)}_{\text{路径必要性}} \land \underbrace{\text{Independent}(g, g')}_{\text{结构独立性}}\}$$
加入了三约束过滤以解决子图采样冗余与结构重复问题。

$$\text{Step 2}: \quad q_i^{\text{aug}} = \text{Confuse}(q_i | \{c_j\}_{j=1}^{M}), \quad c_j \sim p(c) \neq \delta(c_{\text{fixed}}) \quad \text{（条件混淆增强）}$$
重归一化以降低单一强约束主导性，$c_j$ 为从非退化分布采样的条件集合。

$$\text{最终}: \quad \mathcal{D}_{\text{train}} = \alpha \cdot \mathcal{D}_{\text{synth}}^{\text{filtered}} \cup (1-\alpha) \cdot \mathcal{D}_{\text{real}}, \quad \alpha \in (0,1)$$

**对应消融**: —— 论文未披露知识图谱与真实查询的混合比例 $\alpha$ 及消融。

## 实验与分析


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/75e90f3e-f6d6-495e-b96a-d999f640d475/figures/Figure_1.png)
*Figure 1: Figure 1: Benchmark Performance of MindDR, comparing with mainstream deep research productsand state-of-the-art models at comparable parameter scale. MindDR 1.0 denotes the previous-generation model t*



**主实验结果**（MindDR-v1.5-30B-A3B）：

| Method | BrowseComp-ZH | BrowseComp | WideSearch | xbench-DS | DeepResearch Bench | MindDR Bench (RACE) |
|:---|:---|:---|:---|:---|:---|:---|
| MindDR-v1.5-30B-A3B | **45.7%** | **42.8%** | **46.5%** | **75.0%** | **52.5%** | **51.8 (SOTA)** |
| GPT-4级别超大规模模型 |  |  |  |  |  |  |
| Nanbeige4.1系列 |  |  |  |  |  |  |

核心数字解读：BrowseComp-ZH 45.7% 与 BrowseComp 42.8% 显示MindDR在中文检索基准上略优于英文（+2.9pp），可能受益于中文知识图谱（百度百科）的覆盖优势；xbench-DS 75.0% 为所有报告数字中最高，暗示该基准与MindDR优化目标对齐度较高；DeepResearch Bench 52.5% 处于中等水平，反映通用深度研究任务的持续挑战。MindDR Bench RACE 51.8 为团队自评SOTA，但需警惕评估同源偏差（见下文）。


![Figure 5](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/75e90f3e-f6d6-495e-b96a-d999f640d475/figures/Figure_5.png)
*Figure 5: Figure 5: Training dynamics of Search-RL over 180 steps. (a) Reward component scores: ORM(answer accuracy), PRM (average entity coverage), tool invocation success, and format compliance;dashed horizon*



**消融与训练动态分析**：
Figure 5 显示Search-RL训练180步的动态：ORM（答案准确率）、PRM（平均实体覆盖率）、token效率三 reward component 的演化趋势。。该图支持步骤级信用分配的有效性声明，但缺乏与轨迹级基线的直接对比。

Report-RL框架（Figure 6）展示了策略模型与Gemini 3.1 Pro等前沿LLM生成摘要的对比训练流程，但具体性能增益数字未披露。

**公平性与局限检查**：
- **基线强度**：论文声称"rivaling larger-scale models"，但完整对比表格缺失，无法核实是否遗漏近期强基线（如Nanbeige4.1系列）。Figure 1标注为"Benchmark Performance of MindDR, comparing with mainstream deep research products and state-of-the-art models at comparable paramet"，但具体对比数字未在摘录中呈现。
- **计算成本**：~30B参数规模 + A3B（推测为3B激活参数？）的MoE架构，训练180步Search-RL的具体GPU小时数未披露。
- **评估偏差**：MindDR Bench由理想汽车Livis用户交互日志构建，存在同源评估风险；RACE四维评分依赖LLM-as-a-judge，评判模型选择未说明。
- **失败案例**：论文未报告典型失败模式（如检索死循环、信息冲突未解决、时效性错误等）。

## 方法谱系与知识库定位

MindDR属于**多智能体协作深度研究系统**方法族，其直接技术谱系可追溯至：

**父方法/架构基础**：ReAct / Reflexion 风格的工具使用智能体 → Deep Research 类产品（OpenAI GPT-4 Deep Research、Google Gemini Deep Research）的单体架构。MindDR的结构性改动是将单体拆解为三智能体协作，而非改进单一模型的推理能力。

**直接基线与差异**：
- **GPT-4 / Gemini Deep Research（单体端到端）**：MindDR以三智能体+分阶段训练替代单一模型，以~30B参数逼近其性能，但牺牲了一体化部署的简洁性
- **ARPO / TreeRL（分支采样信用分配）**：MindDR以步骤级增量覆盖估计替代指数级分支采样，计算可行性更高但探索空间受限
- **PPO+步骤级critic**：MindDR省去critic模型，以覆盖增量直接估计优势，降低计算成本但可能损失方差缩减效果
- **标准RAG/长文本生成**：MindDR引入专项Report-RL和四维RACE评估，显式优化报告质量的多维属性

**后续方向**：
1. **动态智能体数量扩展**：当前三智能体为固定架构，未来可探索根据查询复杂度自适应增减智能体（如增加Verification Agent专项事实核查）
2. **端到端联合优化**：当前四阶段训练为串行流程，探索三智能体联合训练的可行性以消除阶段间分布偏移
3. **开放域知识图谱扩展**：当前依赖百度百科+英文维基百科，扩展至多语言、多领域知识图谱并解决本体对齐问题

**知识库标签**：
- **modality**: text-to-text（检索+报告生成）
- **paradigm**: multi-agent collaboration, reinforcement learning from AI feedback (RLAIF)
- **scenario**: deep research, long-form report generation, multi-hop information retrieval
- **mechanism**: step-level credit assignment without critic, query-dependent multi-dimensional reward, knowledge-graph-grounded data synthesis
- **constraint**: ~30B parameter budget, no GPT-4 scale model required, deployed in production (Li Auto Livis)

