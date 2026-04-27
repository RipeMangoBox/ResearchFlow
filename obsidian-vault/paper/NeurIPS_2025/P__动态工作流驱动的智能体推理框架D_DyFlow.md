---
title: 'DyFlow: Dynamic Workflow Framework for Agentic Reasoning'
type: paper
paper_level: A
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 动态工作流驱动的智能体推理框架DyFlow
- DyFlow
- DyFlow achieves superior cross-task
acceptance: Poster
cited_by: 6
code_url: https://github.com/IBM/awesome-agentic-workflow-optimization
method: DyFlow
modalities:
- Text
paradigm: Distillation
---

# DyFlow: Dynamic Workflow Framework for Agentic Reasoning

[Code](https://github.com/IBM/awesome-agentic-workflow-optimization)

**Topics**: [[T__Reasoning]], [[T__Code_Generation]] | **Method**: [[M__DyFlow]] | **Datasets**: [[D__LiveBench]] (其他: MATH, SocialMaze, PubMedQA, HumanEval)

> [!tip] 核心洞察
> DyFlow achieves superior cross-task generalization and reasoning performance by dynamically constructing and adjusting reasoning workflows based on task requirements and real-time intermediate feedback through a designer-executor architecture with dynamic operators.

| 中文题名 | 动态工作流驱动的智能体推理框架DyFlow |
| 英文题名 | DyFlow: Dynamic Workflow Framework for Agentic Reasoning |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2509.26062) · [Code](https://github.com/IBM/awesome-agentic-workflow-optimization) · [Project](待补充) |
| 主要任务 | Reasoning, Code Generation |
| 主要 baseline | CoT (Chain-of-Thought), AFlow, AutoGen, MetaGPT, CAMEL, LLM-Debate, Self-Refine, MaAS |

> [!abstract]
> 因为「现有LLM多智能体框架依赖静态预定义工作流，缺乏跨任务适应性且无法利用中间反馈进行鲁棒推理」，作者在「AFlow静态工作流生成」基础上改了「动态工作流生成与设计师-执行器反馈循环架构」，在「MATH、PubMedQA、LiveBench、HumanEval、SocialMaze」上取得「MATH 76.4%（+5.32 over CoT）、SocialMaze 17.18（+11.07 over CoT, 180%相对提升）、HumanEval 92.07% Pass@1（zero-shot泛化）」

- **MATH**: 76.4% accuracy, +5.32 over CoT (Phi-4), +2.4 over AFlow
- **SocialMaze (zero-shot)**: 17.18 score, +11.07 over CoT (180% relative improvement)
- **Cross-executor GPT-4.1-mini SocialMaze**: 42.75 score, +21.76 over CoT (103% relative improvement)

## 背景与动机

当前大语言模型（LLM）的复杂推理任务通常依赖多智能体协作完成，但现有框架存在一个根本性缺陷：工作流是静态预定义的。以数学问题求解为例，一道涉及三角函数与几何的综合题可能需要先分解子问题、再分别求解、最后验证一致性；然而静态框架（如AutoGen、MetaGPT、CAMEL）在任务开始前就固定了每个智能体的角色和执行顺序，一旦中间步骤出现错误，错误会沿固定管道传播而无法修正。

现有方法如何处理这一问题？**CoT (Chain-of-Thought)** 通过单模型生成中间推理链，但缺乏结构化的问题分解和多步骤验证机制，容易在复杂推理中遗漏关键分支。**AFlow** 尝试通过进化搜索自动生成工作流，但生成后的工作流仍是静态的，无法根据执行过程中的中间结果进行动态调整。**AutoGen、MetaGPT、CAMEL** 等多智能体框架采用固定的通信图和预定义角色，虽支持多轮对话，但不具备基于实时反馈的重新规划能力。

这些方法的共同短板在于：**推理流程在任务启动前即被锁定，没有利用执行过程中产生的中间输出进行自适应修正**。具体表现为：(1) 静态工作流无法针对不同推理领域调整策略——代码生成与生物医学问答需要截然不同的操作序列；(2) 中间步骤的错误无法被检测和修正，导致错误累积；(3) 自动化工作流生成方法（如AFlow）虽减少了人工设计成本，但其生成的模板仍绑定于特定数据集或查询类型，缺乏跨域泛化能力。

DyFlow的出发点是：将工作流从"预编译的静态脚本"转变为"运行时动态构建的程序"，通过设计师-执行器闭环架构实现任务自适应的推理流程生成与实时修正。
![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/f73fd7de-d790-44ba-b66d-0f96f2a59e7c/figures/fig_001.png)
*Figure: Each agent’s role and task sequence is fixed beforehand, proceeding rigidly without*



## 核心创新

核心洞察：动态工作流生成的关键在于将"规划"与"执行"解耦为两个持续交互的组件，因为设计师模型可以在执行过程中根据中间反馈重新分解子目标并调整算子参数，从而使跨域零样本泛化和错误自修正成为可能。

| 维度 | Baseline (AFlow/静态多智能体) | 本文 (DyFlow) |
|:---|:---|:---|
| **推理策略** | 静态工作流，任务前固定，无中间反馈 | 动态工作流生成，设计师-执行器循环，基于实时反馈重新规划 |
| **架构设计** | 单模型CoT或固定通信图多智能体 | 双组件架构：DyPlanner（训练的设计师模型）+ 执行器（动态算子） |
| **算子机制** | 固定提示模板或预定义操作 | 上下文感知动态算子（DECOMPOSE_PROBLEM, GENERATE_ANSWER, REVIEW_SOLUTION, REFINE_ANSWER），参数由设计师实时决定 |
| **训练方式** | 标准微调、提示工程或进化搜索 | 从GPT-4.1蒸馏轨迹到Phi-4初始化的DyPlanner，叠加离线偏好优化 |

## 整体框架


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/f73fd7de-d790-44ba-b66d-0f96f2a59e7c/figures/fig_002.png)
*Figure: DyFlow dynamically constructs reasoning workflows by generating stage subgraphs based*



DyFlow的整体数据流遵循"问题输入 → 动态规划 → 上下文感知执行 → 反馈聚合 → 重新规划或终止"的闭环结构：

1. **Problem Input（问题输入）**：接收原始任务查询（如数学证明题、代码描述、生物医学问题），输出标准化的问题描述供设计师处理。

2. **DyPlanner（设计师）**：核心创新组件，以Phi-4为初始化基础，经蒸馏训练得到。输入为问题描述、当前推理状态$s_t$、以及来自执行器的反馈$f_t$；输出为子目标分解$\{g_i\}_{i=1}^{k}$和操作序列$\{o_j, p_j\}_{j=1}^{m}$及其动态参数。DyPlanner根据执行过程中的中间结果实时调整后续计划，而非一次性生成完整工作流。

3. **Dynamic Operator Executor（动态算子执行器）**：接收设计师生成的操作序列与参数化指令，调用四个核心算子——DECOMPOSE_PROBLEM（问题分解）、GENERATE_ANSWER（答案生成）、REVIEW_SOLUTION（解验证）、REFINE_ANSWER（答案精炼）。每个算子的具体参数$p_j$由设计师根据当前上下文动态决定，执行器本身使用现成的LLM（如Phi-4、GPT-4o-mini、GPT-4.1-mini）。

4. **Feedback Aggregator（反馈聚合器）**：收集执行器的输出结果$r_j$和执行状态，结构化为设计师可理解的反馈信号$f_{t+1}$，驱动下一轮重新规划。

5. **Final Answer Generation（最终答案生成）**：当设计师判断所有子目标已达成且通过验证时，输出最终答案。

```
Problem q
    ↓
DyPlanner(q, s₀, ∅) → {gᵢ}, {(oⱼ, pⱼ)}
    ↓
Executor: oⱼ(q, sₜ, pⱼ) → rⱼ, sₜ₊₁
    ↓
Feedback Aggregator: (rⱼ, sₜ₊₁) → fₜ₊₁
    ↓
DyPlanner(q, sₜ₊₁, fₜ₊₁) → replan or terminate
    ↓
Final Answer
```

该闭环的关键在于：设计师不是"一次性规划师"，而是"持续规划师"——每次执行后都重新评估全局状态，决定是继续当前路径、回溯修正，还是调整子目标优先级。
![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/f73fd7de-d790-44ba-b66d-0f96f2a59e7c/figures/fig_003.png)
*Figure: Average pass@k comparisons be-*



## 核心模块与公式推导

### 模块 1: DyPlanner 动态规划（对应框架图 设计师组件）

**直觉**: 静态工作流的核心缺陷在于计划与执行脱节；设计师需要在运行时根据"当前走到了哪一步、上一步结果如何"来动态决定下一步做什么，而非遵循预设脚本。

**Baseline 公式** (AFlow 静态工作流):
$$\text{StaticWorkflow}(q) \rightarrow \text{fixed\_plan}$$
符号: $q$ = 输入问题; fixed_plan = 任务开始前生成的固定操作序列，执行过程中不可变更。

**变化点**: AFlow的进化搜索虽能自动生成工作流，但生成后即为静态模板，无法响应执行过程中的意外结果（如中间计算错误、遗漏分支）。DyFlow将规划建模为条件生成过程，引入状态-反馈依赖。

**本文公式（推导）**:
$$\text{Step 1}: \text{DyPlanner}(q, s_t, f_t) \rightarrow \{g_i\}_{i=1}^{k}, \{o_j, p_j\}_{j=1}^{m} \quad \text{加入状态}s_t\text{和反馈}f_t\text{作为条件，实现上下文感知规划}$$
$$\text{Step 2}: \text{当} f_t \neq \emptyset \text{时，设计师可选择回溯或调整子目标优先级} \quad \text{重规划机制保证错误可修正}$$
$$\text{最终}: \{g_i^{(t+1)}, (o_j^{(t+1)}, p_j^{(t+1)})\} = \text{DyPlanner}(q, s_{t+1}, f_{t+1})$$
符号: $s_t$ = 第$t$步的推理状态（已完成的子目标、中间结果）; $f_t$ = 执行器反馈（验证结果、错误信息）; $g_i$ = 子目标; $o_j$ = 算子类型; $p_j$ = 动态参数。

**对应消融**: Table 4显示移除知识蒸馏或偏好优化后DyPlanner性能下降（具体数值未在提取内容中完整呈现，但两项均为正向贡献）。

---

### 模块 2: 知识蒸馏与偏好优化训练（对应框架图 训练流程）

**直觉**: 强大的规划能力需要大量高质量轨迹数据，但GPT-4.1等顶级模型推理成本高昂；通过蒸馏将其规划模式迁移到较小的Phi-4模型，再用偏好优化精细对齐，可在保持效率的同时获得接近教师模型的动态规划能力。

**Baseline 公式** (标准SFT):
$$\mathcal{L}_{\text{SFT}} = -\mathbb{E}_{(x,y) \sim \mathcal{D}} [\log P_{\theta}(y|x)]$$
符号: $\theta$ = 学生模型参数; $(x,y)$ = 输入-输出训练对。

**变化点**: 标准SFT使用人工标注或固定数据集，而DyFlow需要模仿的是GPT-4.1的*规划轨迹*（即完整的问题分解和操作序列决策过程），而非仅最终答案。此外，单一SFT无法区分优质规划与次优规划，需引入偏好优化进行精细排序。

**本文公式（推导）**:
$$\text{Step 1}: \mathcal{D}_{\text{GPT-4.1}} = \{(q_i, \tau_i)\}_{i=1}^{N}, \quad \tau_i \sim P_{\text{GPT-4.1}}(\cdot|q_i) \quad \text{仅用训练集数据，GPT-4.1生成教师轨迹}$$
$$\text{Step 2}: \mathcal{L}_{\text{SFT}} = -\mathbb{E}_{(q, \tau) \sim \mathcal{D}_{\text{GPT-4.1}}} [\log P_{\text{DyPlanner}}(\tau | q)] \quad \text{蒸馏GPT-4.1的轨迹分布到DyPlanner}$$
$$\text{Step 3}: \mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(q, \tau_w, \tau_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{P_{\text{DyPlanner}}(\tau_w|q)}{P_{\text{ref}}(\tau_w|q)} - \beta \log \frac{P_{\text{DyPlanner}}(\tau_l|q)}{P_{\text{ref}}(\tau_l|q)} \right) \right]$$
$$\text{加入了偏好对}(\tau_w, \tau_l)\text{以区分优劣规划，} \beta \text{为温度系数，} P_{\text{ref}} \text{为参考模型（SFT后的DyPlanner）}$$
$$\text{最终}: \theta^* = \text{arg}\min_{\theta} \left( \mathcal{L}_{\text{SFT}} + \lambda \mathcal{L}_{\text{DPO}} \right)$$
符号: $\tau$ = 完整规划轨迹（子目标序列+操作序列）; $\tau_w$ = 优胜轨迹; $\tau_l$ = 失败轨迹; $\sigma$ = sigmoid函数; $\lambda$ = 损失权重。

**对应消融**: Table 4显示移除SFT知识蒸馏或离线DPO偏好优化均导致性能下降，验证了双阶段训练的必要性。

---

### 模块 3: 动态算子执行（对应框架图 执行器组件）

**直觉**: 同样的"生成答案"操作在不同推理阶段需要不同的指令细节——初步探索时需要广泛搜索，验证阶段需要严格检查，精炼阶段需要针对性修正。固定模板无法表达这种上下文依赖性。

**Baseline 公式** (固定算子):
$$\text{FixedOp}(q) \rightarrow r$$

**变化点**: 传统方法的算子是"硬编码"的提示模板，参数固定。DyFlow的设计师为每个算子实例生成上下文特定的参数$p_j$，使同一算子在不同场景下表现出不同行为。

**本文公式（推导）**:
$$\text{Step 1}: o_j \in \{\text{DECOMPOSE\_PROBLEM}, \text{GENERATE\_ANSWER}, \text{REVIEW\_SOLUTION}, \text{REFINE\_ANSWER}\} \quad \text{语义化算子集合}$$
$$\text{Step 2}: p_j = \text{ContextEncoder}(s_t, g_{\text{current}}) \quad \text{设计师根据当前状态编码动态参数}$$
$$\text{最终}: o_j(q, s_t, p_j) \rightarrow r_j, s_{t+1} \quad \text{算子输出结果并更新全局状态}$$
符号: $o_j$ = 第$j$个算子; $p_j$ = 动态参数（如分解粒度、验证严格度、修正策略）; $r_j$ = 算子输出结果; $s_{t+1}$ = 更新后的推理状态。

**对应消融**: Figure 6/8展示不同推理域的算子使用频率差异，证明动态算子确实形成了任务特定的使用模式（如MATH中DECOMPOSE_PROBLEM高频使用，HumanEval中GENERATE_ANSWER与REFINE_ANSWER交替频繁）。

## 实验与分析



本文在五个推理基准上评估DyFlow，覆盖数学推理（MATH）、科学问答（PubMedQA）、综合推理（LiveBench）、代码生成（HumanEval）和社会推理（SocialMaze）。其中HumanEval和SocialMaze为训练时未见的held-out域，用于验证zero-shot泛化能力。Table 10展示了跨三种执行器模型（Phi-4、GPT-4o-mini、GPT-4.1-mini）的详细对比。

核心结果方面，以Phi-4为执行器时，DyFlow在MATH达到76.4% accuracy，相比CoT的71.08%提升+5.32，相比AFlow的74.0%提升+2.4；在SocialMaze取得17.18分，相比CoT的6.11分提升+11.07（180%相对提升）；在held-out的HumanEval上达到92.07% Pass@1，相比CoT的87.8%提升+4.27。特别值得注意的是cross-executor泛化：当执行器换为更强的GPT-4.1-mini时，SocialMaze分数跃升至42.75，相比CoT (GPT-4.1-mini)的20.99提升+21.76（103%相对提升），表明DyFlow的动态规划能力可与更强执行器产生协同增益。



消融实验（Table 4）验证了训练策略的两个关键组件：移除知识蒸馏（SFT）或离线偏好优化（DPO）均导致性能下降，说明单纯依赖Phi-4基础能力或仅做SFT不足以获得高质量的动态规划行为。成本方面，Table 8显示DyFlow在MATH上的训练成本为617,560 input tokens / 452,044 output tokens（DyPlanner），相比直接使用GPT-4.1作为设计师的788,910 / 297,669更为经济，且准确率更高。


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/f73fd7de-d790-44ba-b66d-0f96f2a59e7c/figures/fig_004.png)
*Figure: Cross-executor performance comparison between CoT and DyFlow across five reasoning*



定性分析方面，Figure 5的案例研究对比了CoT与DyFlow在MATH几何题上的行为差异：CoT忽略了cos B = -4/5的负值分支导致错误，而DyFlow通过DECOMPOSE_PROBLEM识别两个几何分支，经REVIEW_SOLUTION验证完整性后给出正确答案。Figure 6/8的算子频率分析显示不同领域形成了 distinct 的操作模式——MATH中DECOMPOSE_PROBLEM占主导，HumanEval中GENERATE_ANSWER与REFINE_ANSWER高频交替，证明动态工作流确实实现了任务自适应。

公平性检查：主要对比聚焦于CoT而非最强工作流基线（ScoreFlow、ADAS、AgentSquare在提取内容中未见实验结果）；AFlow、MaAS仅出现在成本对比Table 8而非主结果Table 10。Zero-shot泛化声明仅基于两个held-out数据集。此外，设计师训练使用GPT-4.1（远强于Phi-4执行器），存在能力差距可能使蒸馏收益部分来自教师-学生不匹配而非真正学到可泛化策略。

## 方法谱系与知识库定位

DyFlow属于**Agentic Workflow Optimization**方法谱系，直接父方法为**AFlow**——两者均致力于自动化智能体工作流生成，但DyFlow将AFlow的静态进化搜索范式根本性转变为动态运行时生成范式。

**方法谱系变更槽位**：
- **Architecture（架构）**: AFlow的单体进化搜索 → DyFlow的设计师-执行器双组件架构
- **Inference Strategy（推理策略）**: AFlow的静态工作流模板 → DyFlow的反馈驱动动态重规划
- **Data Pipeline（数据管道）**: AFlow的固定提示模板 → DyFlow的上下文感知动态算子参数化
- **Training Recipe（训练策略）**: AFlow的进化算法 → DyFlow的GPT-4.1轨迹蒸馏 + 离线DPO偏好优化

**直接基线差异**：
- **AFlow**: 同为自动化工作流生成，但AFlow生成静态模板，DyFlow运行时动态构建
- **ScoreFlow**: 同为"Flow"系列工作流优化，ScoreFlow基于分数的偏好优化，DyFlow采用设计师-执行器架构与动态算子
- **ADAS/AgentSquare**: 自动智能体设计/搜索，DyFlow强调运行时适应性而非架构搜索
- **AutoGen/MetaGPT/CAMEL**: 静态多智能体框架，无动态重规划能力

**后续方向**：(1) 扩展动态算子至工具调用场景（搜索引擎、数据库、代码解释器），突破当前纯符号/文本推理限制；(2) 探索设计师模型与执行器的联合训练而非分离蒸馏，缩小GPT-4.1教师与Phi-4学生之间的能力鸿沟；(3) 将动态工作流机制扩展至多模态推理（视觉-语言任务）和开放域环境交互。

**标签**: 模态(text) / 范式(agentic workflow, distillation, dynamic planning) / 场景(mathematical reasoning, scientific QA, code generation) / 机制(designer-executor loop, feedback-driven replanning, context-aware operators) / 约束(无外部工具集成, 算子集限于符号推理)

