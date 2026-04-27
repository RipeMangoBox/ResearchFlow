---
title: Scaling Test-Time Compute for Agentic Coding
type: paper
paper_level: B
venue: arXiv
year: 2026
paper_link: https://arxiv.org/abs/2604.16529
aliases:
- Agentic代码生成的测试时计算扩展
- STCAC
modalities:
- Text
paradigm: Reinforcement Learning
---

# Scaling Test-Time Compute for Agentic Coding

[Paper](https://arxiv.org/abs/2604.16529)

**Topics**: [[T__Agent]], [[T__Code_Generation]], [[T__Benchmark_-_Evaluation]]

| 中文题名 | Agentic代码生成的测试时计算扩展 |
| 英文题名 | Scaling Test-Time Compute for Agentic Coding |
| 会议/期刊 | arXiv (Cornell University) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.16529) · [Code] · [Project] |
| 主要任务 | Agentic代码生成（SWE-Bench Verified, TempTest） |
| 主要 baseline | Best-of-N, Self-Refinement, REBASE, Majority Voting, Pairwise Tournament |

> [!abstract] 因为「agentic coding任务中现有测试时扩展方法（如Best-of-N、Self-Refinement）在并行聚合和迭代优化上存在局限」，作者在「Best-of-N + Self-Refinement」基础上改了「引入PDR（Prompted Differential Refinement）迭代生成差异化补丁 + RTV（Recursive Tournament Voting）并行聚合机制」，在「SWE-Bench Verified」上取得「Claude-4.5-Opus从70.9%提升至77.6%」

- **关键性能1**: Claude-4.5-Opus在SWE-Bench Verified上从70.9% → 77.6%（+6.7%），超越Best-of-N的73.0%和Self-Refinement的72.3%
- **关键性能2**: Gemini-3-Flash在TempTest上从24.0% → 43.0%（+19.0%），Best-of-N仅28.0%
- **关键性能3**: 在SWE-Bench Verified上，PDR+RTV（G=16）相比直接trace聚合提升5.6%（70.0% → 75.6%）

## 背景与动机

大型语言模型在软件工程任务中的应用已从简单代码补全演进至端到端的agentic coding——即模型作为自主agent，通过多轮工具调用、代码编辑、测试执行来修复真实GitHub issue。然而，这类任务具有极高的不确定性：同一问题可能存在多种修复策略，单次推理容易陷入局部最优。

现有测试时计算（test-time compute）扩展方法试图通过增加推理时的计算资源来提升可靠性。Best-of-N采样通过并行生成N个候选方案并选取最优者，但缺乏对候选间差异性的显式利用；Self-Refinement通过迭代让模型自我修正，但容易因确认偏误（confirmation bias）而收敛至错误方向；REBASE等基于过程奖励的方法需要昂贵的训练数据；Majority Voting在代码这种结构化输出上难以直接应用，而Pairwise Tournament虽能比较候选，但面对agentic任务的长轨迹（traces）时效率低下。

这些方法的共同短板在于：**未能有效处理agentic任务特有的长程、结构化、可执行轨迹**。具体而言，(1) 并行生成的候选方案之间缺乏差异化引导，导致冗余采样；(2) 聚合阶段直接使用原始执行轨迹，信息噪声大、上下文窗口受限；(3) 迭代优化与并行扩展被割裂对待，未形成统一recipe。本文正是针对这三重局限，提出将迭代差异化生成与结构化并行聚合耦合的扩展框架。

本文提出PDR+RTV统一框架，首次实现agentic coding任务上测试时计算的系统性扩展。
![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/ee7b0bd7-1df0-4d1b-bca0-7a0c0c18097b/figures/Figure_3.png)
*Figure 3: Figure 3 Unified PDR + RTV inference-time scaling recipe for agentic coding. The agent first executes N independent rolloutsin parallel (iteration 0), and each rollout is converted into a compact stru*



## 核心创新

核心洞察：agentic coding的测试时扩展需要「差异化迭代生成」与「结构化并行聚合」的协同，因为agent轨迹的长程依赖和可执行特性使得传统Best-of-N的独立采样和Self-Refinement的串行修正均无法有效扩展，从而使基于摘要的递归锦标赛投票成为可扩展的聚合机制。

| 维度 | Baseline | 本文 |
|------|---------|------|
| 候选生成 | N次独立采样（Best-of-N）或单轨迹迭代（Self-Refinement） | PDR：迭代生成，每轮基于前序失败反馈显式差异化 |
| 聚合输入 | 原始代码/轨迹（完整上下文） | 结构化摘要（编辑位置+摘要描述），压缩且保留关键决策信息 |
| 聚合机制 | 多数投票或成对比较 | RTV：递归锦标赛投票，分组比较→组内胜出者再比较，可扩展至大规模候选 |
| 迭代-并行关系 | 分离设计（如重复Best-of-N） | 统一recipe：每轮迭代内部并行N次，迭代间传递差异化信号 |

## 整体框架



PDR+RTV的统一推理时扩展框架包含三个核心阶段，形成"生成→摘要→聚合→迭代"的闭环：

**阶段一：并行Rollout生成（Parallel Rollouts）**。输入为自然语言描述的GitHub issue及代码库上下文。Agent执行N次独立的修复尝试（rollouts），每次包含多轮工具调用（文件浏览、代码编辑、测试运行）。与传统Best-of-N不同，这些rollouts在迭代轮次t>0时会接收前序轮次的差异化反馈信号。

**阶段二：结构化摘要生成（Structured Summaries）**。每个rollout的原始执行轨迹（可能包含数十轮交互、数千token）被压缩为结构化摘要：包含编辑位置（file paths, line ranges）和摘要描述（edit summary）。这一压缩使得后续聚合模块能在有限上下文内处理大量候选。

**阶段三：递归锦标赛投票（RTV, Recursive Tournament Voting）**。将N个候选分为G组，组内进行成对比较选出胜者；各组胜者在更高层级再次分组比较，直至选出最终候选。该机制避免了全成对比较的O(N²)复杂度，同时保留了成对比较对结构化输出的适应性。

**阶段四：迭代差异化优化（PDR, Prompted Differential Refinement）**。将RTV选出的最优候选（或前几名）的反馈——特别是其与其他候选的差异——作为prompt信号，启动下一轮迭代，生成新一代差异化rollouts。

数据流示意：
```
Issue + Repo → [Iteration t] → N Parallel Rollouts → Structured Summaries
                                                      ↓
                                            RTV (G groups, recursive)
                                                      ↓
                                            Winner + Differential Feedback
                                                      ↓
                                            [Iteration t+1] → ... → Final Patch
```


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/ee7b0bd7-1df0-4d1b-bca0-7a0c0c18097b/figures/Figure_2.png)
*Figure 2: Figure 2 Overview of RTV (Recursive Tournament Voting). RTV is our parallel aggregation technique designed foragentic tasks – it (1) executes N parallel, independent rollouts with the agent, (2) produ*



## 核心模块与公式推导

### 模块1: 结构化摘要生成（对应框架图 阶段二）

**直觉**: Agentic轨迹冗长且包含大量无关执行细节，直接用于聚合会淹没关键修复决策；结构化摘要提取"在哪里改、改什么"的核心信息，使聚合聚焦语义等价的修复策略。

**Baseline形式（直接Trace聚合）**: 
$$\text{Score}(r_i) = f(\text{trace}_i)$$
其中$\text{trace}_i$为第$i$个rollout的完整执行轨迹，$f$为评分函数（如LLM-as-judge或启发式规则）。

**符号**: $\text{trace}_i$ = 原始agent轨迹（工具调用序列+观测）；$s_i$ = 结构化摘要；$E_i$ = 编辑位置集合；$D_i$ = 编辑描述。

**变化点**: 直接trace存在(1)上下文长度超限，(2)执行噪声（如测试失败时的无关重试）干扰判断，(3)语义等价但表面不同的轨迹被误判为不同。本文将轨迹转换为与执行路径无关的编辑摘要。

**本文公式（推导）**:
$$\text{Step 1}: s_i = \text{Extract}(\text{trace}_i) = \{(e_j, d_j)\}_{j=1}^{|E_i|}$$
其中$e_j \in E_i$为编辑位置，$d_j \in D_i$为自然语言描述。\quad \text{提取关键编辑决策，丢弃执行细节}

$$\text{Step 2}: \text{Sim}(s_i, s_j) = \mathbb{1}[E_i \cap E_j \neq \emptyset] \cdot \text{LLM\_Judge}(D_i, D_j)$$
\quad \text{基于编辑位置重叠和描述语义相似度判断等价性}

**对应消融**: Figure 4显示，在SWE-Bench Verified上，结构化摘要（blue）相比直接trace（orange）在Gemini-3.1-Pro上提升5.6%（70.0% → 75.6%），在Claude-4.5-Sonnet上提升3.4%（72.0% → 75.4%）。

---

### 模块2: 递归锦标赛投票 RTV（对应框架图 阶段三）

**直觉**: 成对比较对结构化输出更可靠，但全比较O(N²)不可扩展；递归分组将复杂度降至O(N log_G N)，且保留局部比较优势。

**Baseline形式（Majority Voting / Best-of-N）**:
$$\text{Winner} = \text{arg}\max_{i \in [N]} \text{Score}(r_i), \quad \text{Score}(r_i) = \text{Pass}(r_i) \text{ or } \text{LLM\_Judge}(r_i)$$

**符号**: $G$ = 组大小；$R$ = RTV轮数；$N$ = 候选总数；$V(s_a, s_b) \in \{0,1\}$ = 成对比较投票结果。

**变化点**: Majority Voting需要精确匹配，在代码输出上过于严格；Best-of-N依赖绝对评分，对agentic任务缺乏细粒度比较能力。RTV通过结构化摘要上的成对比较，实现模糊但可靠的偏好聚合。

**本文公式（推导）**:
$$\text{Step 1 (分组)}: \mathcal{G}^{(0)} = \{G_1^{(0)}, ..., G_{\lceil N/G \rceil}^{(0)}\}, \quad |G_k^{(0)}| \leq G$$
\quad \text{将N个候选分为大小为G的组}

$$\text{Step 2 (组内锦标赛)}: w_k^{(r)} = \text{RoundRobin}(G_k^{(r)}), \quad \text{基于} \prod_{(a,b) \in \text{pairs}} V(s_a, s_b)$$
\quad \text{组内全比较选出胜者，使用结构化摘要而非原始trace}

$$\text{Step 3 (递归)}: \mathcal{G}^{(r+1)} = \{w_1^{(r)}, w_2^{(r)}, ...\}, \quad \text{重复直至}|\mathcal{G}^{(R)}| = 1$$
$$\text{最终}: \text{Winner} = \mathcal{G}^{(R)}, \quad R = \lceil \log_G N \rceil$$

**对应消融**: Figure 6（即Figure 5 Left）显示，SWE-Bench Verified上G=16时Gemini-3-Flash达最高性能；TempTest上G=8更优，表明任务复杂度影响最优组大小。

---

### 模块3: 提示差异化优化 PDR（对应框架图 阶段四）

**直觉**: 迭代优化需要避免确认偏误——模型倾向于坚持初始错误方向；通过显式提示前序候选的差异和失败模式，强制探索新的修复策略。

**Baseline形式（Self-Refinement）**:
$$\text{Prompt}_{t+1} = \text{Prompt}_t + \text{Feedback}(\text{trace}_t), \quad \text{Feedback} = \text{模型自身生成的反思}$$

**符号**: $t$ = 迭代轮次；$\Delta_t$ = 第t轮差异化信号；$\mathcal{F}_t$ = 前序失败模式集合；$r_t^{(n)}$ = 第t轮第n个rollout。

**变化点**: Self-Refinement的反馈来自单一样本且易自我强化；PDR利用并行生成的多样候选，提取"成功候选与失败候选的关键差异"作为外部监督信号。

**本文公式（推导）**:
$$\text{Step 1 (差异提取)}: \Delta_t = \{(s_{\text{win}} - s_{\text{lose}}^{(m)})\}_{m=1}^{M}$$
其中$s_{\text{win}}$为RTV胜出者摘要，$s_{\text{lose}}^{(m)}$为失败候选摘要。\quad \text{提取成功与失败的关键差异}

$$\text{Step 2 (失败模式聚类)}: \mathcal{F}_t = \text{Cluster}(\{s_{\text{lose}}^{(m)}\}), \quad \text{按编辑位置和错误类型分组}$$
\quad \text{识别系统性失败模式}

$$\text{Step 3 (差异化Prompt)}: \text{Prompt}_{t+1}^{(n)} = \text{BasePrompt} + \Delta_t + \mathcal{F}_t + \text{ExplicitDiversity}(n)$$
其中$\text{ExplicitDiversity}(n)$为第n个rollout的差异化指令（如"尝试不同的修复位置"）。\quad \text{强制生成多样化候选}

$$\text{最终}: r_{t+1}^{(n)} \sim \text{Agent}(\text{Prompt}_{t+1}^{(n)}), \quad n \in [N]$$

**对应消融**: Figure 10（即Figure 9）显示从iteration 0到1，各模型的pass count分布显著右移，Claude-4.5-Opus和Gemini-3.1-Pro的高pass count样本明显增加，验证PDR有效扩展了成功候选的多样性。

## 实验与分析

**主实验结果（SWE-Bench Verified）**：

| Method | Claude-4.5-Opus | Gemini-3.1-Pro | Claude-4.5-Sonnet | Gemini-3-Flash |
|--------|-----------------|----------------|-------------------|----------------|
| Pass@1 | 70.9% | 
| Best-of-N | 73.0% | 
| Self-Refinement | 72.3% | 
| PDR+RTV (本文) | **77.6%** | 
| Δ vs Best-of-N | +4.6% | 


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/ee7b0bd7-1df0-4d1b-bca0-7a0c0c18097b/figures/Figure_1.png)
*Figure 1: Figure 1 Main results of our agentic PDR+RTV test-time scaling method. We improve Claude-4.5-Opus from 70.9% →77.6% on SWE-Bench Verified (mini-SWE-agent) and from 47.0% →59.1% on Terminal-Bench v2.0*



**核心发现分析**：

Claude-4.5-Opus上的6.7%绝对提升（70.9% → 77.6%）是本文的核心证据。这一增益拆解来看：(1) 相比Best-of-N的+4.6%表明RTV聚合优于简单选优；(2) 相比Self-Refinement的+5.3%表明PDR的差异化迭代优于自我反思。Figure 1显示该结果在SWE-Bench Verified上为当前最优。

TempTest上的19.0%提升（24.0% → 43.0%）更为显著，说明PDR+RTV对较弱模型（Gemini-3-Flash）的增益更大——测试时计算扩展对基线能力较弱的模型具有更强的补偿效应。Figure 12（即Figure 11 Top）展示了RTV各轮次的pass@1演化，显示多数模型在2-3轮RTV后收敛，验证递归深度的合理性。

**消融实验**：

- **结构化摘要 vs 直接Trace**（Figure 4）：在Gemini-3.1-Pro上，摘要机制带来5.6%提升，验证信息压缩的必要性
- **组大小G的缩放**（Figure 6）：G=16在SWE-Bench Verified最优，G=8在TempTest最优，表明复杂任务需要更大组内比较粒度
- **迭代轮次**：Figure 10显示iteration 1相比0的pass count分布显著改善，但iteration 2收益递减，提示2轮迭代为性价比拐点


![Figure 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/ee7b0bd7-1df0-4d1b-bca0-7a0c0c18097b/figures/Figure_4.png)
*Figure 4: Figure 4 Parallel aggregation results based on generating structured summaries (blue) vs. directly using rollout traces(orange) on SWE-Bench Verified and Terminal-Bench v2.0. We find across Gemini-3-F*



**公平性检查**：
- Baselines覆盖全面：Best-of-N、Self-Refinement、REBASE、Majority Voting、Pairwise Tournament均有对比
- 计算成本：PDR+RTV的N×T次rollout（T为迭代数）显著高于单次推理，但低于训练时方法；RTV的O(N log_G N)比较复杂度优于全成对
- 未报告具体推理延迟和API调用成本，
- 失败案例：未分析PDR+RTV在何种issue类型上仍失败（如需要多文件协调的复杂重构）

## 方法谱系与知识库定位

**方法家族**: Test-Time Compute Scaling / Inference-Time Scaling for Code Generation

**父方法**: Best-of-N Sampling（并行扩展）+ Self-Refinement（迭代优化）。本文将两者统一为耦合框架，并引入结构化聚合机制。

**改动插槽**: 
- **Architecture**: 新增RTV递归锦标赛模块替代简单选优
- **Objective**: 结构化摘要上的成对比较替代绝对评分
- **Training_recipe**: 无需训练，纯推理时方法
- **Data_curation**: 利用前序迭代的失败模式作为隐式数据筛选
- **Inference**: PDR+RTV统一recipe，迭代内并行、迭代间差异化

**直接Baselines及差异**：
- **Best-of-N**: 本文增加迭代差异化（PDR）和结构化聚合（RTV）
- **Self-Refinement**: 本文将单轨迹反思改为多候选差异提取，避免确认偏误
- **REBASE**: 本文无需过程奖励模型训练，纯推理时扩展
- **Pairwise Tournament**: 本文扩展为递归分组，解决可扩展性瓶颈

**后续方向**：
1. **自适应计算分配**: 根据issue复杂度动态调整N、G、T，而非固定配置
2. **学习与推理结合**: 将PDR的差异化信号蒸馏为训练目标，实现测试时扩展的进一步放大
3. **多模态扩展**: 将结构化摘要机制扩展至UI自动化、机器人控制等更广义agentic任务

**标签**: 
- Modality: Code / Software Engineering
- Paradigm: Test-Time Compute Scaling, Agentic AI
- Scenario: Automated Program Repair, GitHub Issue Resolution
- Mechanism: Parallel Aggregation, Iterative Refinement, Structured Summarization, Tournament Voting
- Constraint: Inference-Only, No Training Required, API-Based

