---
title: A Self-Evolving Framework for Efficient Terminal Agents via Observational Context Compression
type: paper
paper_level: B
venue: arXiv
year: 2026
paper_link: https://arxiv.org/abs/2604.19572
aliases:
- 终端智能体自进化观测压缩框架TACO
- ASFETA
- 终端输出的冗余模式在单次任务内具有局部规律性（同类命令产生相似噪声）
modalities:
- Text
---

# A Self-Evolving Framework for Efficient Terminal Agents via Observational Context Compression

[Paper](https://arxiv.org/abs/2604.19572)

**Topics**: [[T__Agent]], [[T__Reasoning]], [[T__Benchmark_-_Evaluation]]

> [!tip] 核心洞察
> 终端输出的冗余模式在单次任务内具有局部规律性（同类命令产生相似噪声），跨任务也存在可迁移的结构（相似仓库类型有相似的日志格式）。TACO的核心洞察是：与其用固定规则或重新训练来捕捉这些模式，不如让规则本身成为可进化的一等公民——在任务执行中在线发现规则、在任务结束后跨任务传播规则。压缩规则的自进化将「适应性」从模型权重层面下移到「结构化文本规则」层面，实现了无需梯度更新的持续改进。有效性的根本来源是：任务内在线适应消除了固定规则无法覆盖的长尾噪声模式，而全局规则池则避免了每次任务从零开始的冷启动代价。

| 中文题名 | 终端智能体自进化观测压缩框架TACO |
| 英文题名 | A Self-Evolving Framework for Efficient Terminal Agents via Observational Context Compression |
| 会议/期刊 | arXiv (Cornell University) (预印本) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.19572) · [Code](https://arxiv.org/abs/2604.19572) · [Project](https://arxiv.org/abs/2604.19572) |
| 主要任务 | 终端环境观测上下文压缩，降低长时序交互中的token冗余，提升终端智能体（Terminal Agent）的推理效率与准确率 |
| 主要 baseline | 无压缩Baseline、6条预定义规则（Seed Rules）、200条人工规则（High-Quality Rules）、SWE-Pruner |

> [!abstract] 因为「终端输出包含大量冗余噪声导致上下文token二次增长、现有静态压缩规则无法适应异构环境」，作者在「无压缩Baseline + 固定规则基线」基础上改了「引入任务内在线规则进化与全局规则池复用的两级自进化机制」，在「TerminalBench 2.0」上取得「准确率从40.6%提升至42.7%（+2.1%），token开销增加12.2%」

- **准确率提升**: TACO Full 42.7% vs. Baseline 40.6%（+2.1% absolute）
- **关键组件贡献**: 去除任务内规则集进化 → 准确率-1.7%，token开销+30.7%
- **全局规则池效率价值**: 去除全局规则池进化 → 准确率-0.2%，token开销+18.1%

## 背景与动机

终端智能体（如SWE-Agent、OpenHands）在软件工程任务中需要与shell环境进行多轮交互，每次执行命令后需将完整终端输出保留在上下文历史中。一个具体的场景是：智能体执行`pytest`测试后，终端返回数百行包含通过/失败用例的详细日志、堆栈追踪、覆盖率报告等，其中仅少数几行错误信息对下一步决策真正有用。随着交互步数增加，这些冗余信息不断累积，导致上下文token开销呈二次方增长，最终触发上下文长度限制或稀释关键信号。

现有方法从不同角度尝试缓解这一问题：
- **静态截断/固定提示词压缩**：直接截断输出头部或尾部，或使用预定义的模板提取关键行。这类方法实现简单，但面对异构终端环境（不同代码仓库、命令类型、执行状态）时泛化能力极差——`git diff`和`npm build`的冗余模式截然不同。
- **人工设计规则**：如200条High-Quality Rules，针对常见命令类型设计提取逻辑。虽然比纯截断更精细，但在跨任务场景下表现不稳定，无法覆盖长尾命令和意外输出格式。
- **LLM通用摘要**：将终端输出送入LLM生成摘要。缺乏对终端输出结构（如ANSI颜色码、进度条、日志级别）的针对性理解，且每次摘要本身消耗额外token。
- **训练型方法（SWE-Pruner）**：通过微调模型学习自适应剪枝。虽具备更强自适应能力，但需要额外训练数据与计算资源，且高度绑定SWE-Bench风格任务，难以迁移到更广泛的终端环境。

这些方案的共同瓶颈在于：终端环境的高度异构性要求压缩策略具备任务感知能力，但任务感知能力的获取传统上依赖训练或大量人工标注，二者之间存在根本性张力。本文的核心动机正是打破这一张力——在无需训练、无需人工持续干预的前提下，让压缩框架自动适应当前任务的终端输出特征，并将积累的压缩经验跨任务复用。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/fafe0aab-8feb-41a8-aab9-07407d2b91ef/figures/Figure_1.png)
*Figure 1: Figure 1: (a) Token count comparison before and after manually extracting effective text from 50sampled trajectories on TB 2.0. (b) Performance comparison of the baseline agent, TACO, and staticcompre*



## 核心创新

核心洞察：终端输出的冗余模式在单次任务内具有局部规律性（同类命令产生相似噪声），跨任务也存在可迁移的结构（相似仓库类型有相似的日志格式），因此压缩规则本身应成为可进化的一等公民——在任务执行中在线发现规则、在任务结束后跨任务传播规则——从而使「无需梯度更新的持续自适应压缩」成为可能。

| 维度 | Baseline / 现有方法 | 本文 (TACO) |
|:---|:---|:---|
| 规则来源 | 人工预定义或模型训练得到 | 任务执行中自动发现与精炼 |
| 适应机制 | 静态固定或需重新训练 | 任务内在线进化 + 跨任务全局复用 |
| 部署成本 | 需标注数据微调或持续人工维护 | 即插即用，零训练，零梯度更新 |
| 知识层级 | 规则固定于模型权重或配置文件 | 规则以结构化文本形式显式存储、检索、复用 |

## 整体框架


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/fafe0aab-8feb-41a8-aab9-07407d2b91ef/figures/Figure_2.png)
*Figure 2: Figure 2:Overview of TACO. For each task, TACO initializes active rules from the global rulepool (Rule Initialization part), compresses terminal outputs using the rule evolved within the task(Terminal*



TACO以插件形式嵌入现有终端智能体框架，在观测层面拦截并压缩终端输出，不修改智能体本身的决策逻辑。整体数据流如下：

**输入**：原始终端观测（terminal observation），包含完整的命令输出文本，通常伴随大量冗余信息。

**模块A — 规则初始化（Rule Initialization）**：维护一个全局规则池（Global Rule Pool），存储跨任务积累的可复用压缩规则。每个新任务开始时，通过Top-k检索从全局规则池中选取最相关的规则，初始化该任务的活跃规则集（Active Rule Set）。该模块实现跨任务知识迁移的冷启动，避免每次任务从零开始。

**模块B — 批量执行与压缩（Batch Execution with TACO）**：在智能体每一步交互中，TACO拦截原始终端观测，将其与当前活跃规则集进行匹配。若某条规则命中，则按规则指定的方式压缩输出；若出现压缩错误（如过度压缩导致信息丢失），则触发规则的生成或修改（Spawn or Modify Rules）。该过程构成任务内在线自适应（Intra-Task Rule Set Evolution），是TACO效果的核心来源。

**模块C — 全局规则池更新（Global Rule Pool Update）**：任务结束后，将本次任务中产生的规则更新反馈回全局规则池。每条规则由全局置信度$c_g$和使用次数$n_g$共同决定排名分数，驱动高质量规则的跨任务复用与低质量规则的淘汰。该模块主要贡献于token效率而非准确率。

**输出**：压缩后的终端观测，保留关键决策信息，显著降低上下文冗余。

```
[原始终端观测] → [规则初始化: Top-k检索全局规则池] → [活跃规则集]
                                                      ↓
[压缩后观测] ← [规则匹配/执行/错误检测] ← [批量执行与压缩: 任务内在线进化]
                                                      ↓
[全局规则池更新: 排名分数 = f(c_g, n_g)] ← [规则生成/修改反馈]
```

## 核心模块与公式推导

### 模块 1: 规则排名与全局规则池更新（对应框架图 模块C）

**直觉**: 跨任务复用需要区分高质量规则与低质量规则，使用频率和全局表现是天然的可量化信号。

**Baseline 公式** (传统固定规则系统): 无显式排名机制，规则以静态列表形式存储，按人工预设优先级或简单FIFO淘汰。

**变化点**: 静态规则无法自适应评估规则质量，导致低效规则长期占用检索位置，高效规则难以跨任务传播。TACO引入基于置信度和使用频率的动态排名。

**本文公式（推导）**:
$$\text{Step 1}: \quad c_g(r) = \frac{\text{successful applications of } r}{\text{total applications of } r} \quad \text{（规则r的全局成功率作为置信度）}$$
$$\text{Step 2}: \quad n_g(r) = \text{accumulated usage count of } r \text{ across all tasks} \quad \text{（跨任务使用频次）}$$
$$\text{最终}: \quad \text{RankingScore}(r) = f(c_g, n_g) \quad \text{（置信度与使用次数的联合函数，具体形式未显式给出）}$$

符号: $c_g$ = 全局置信度（global confidence），$n_g$ = 全局使用次数（usage count），$r$ = 单条压缩规则。

**对应消融**: 去除全局规则池进化（即退化为准静态规则集），准确率下降0.2%，token开销增加18.1%。该组件对准确率贡献边际，但对跨任务效率至关重要。

### 模块 2: 任务内在线规则进化（对应框架图 模块B）

**直觉**: 单次任务内的终端输出具有局部规律性——同一命令多次执行、同一类型错误重复出现——实时发现这些模式比依赖预定义规则更精准。

**Baseline 公式** (静态规则匹配): 
$$\text{Compressed}_t = \text{Apply}(\mathcal{R}_{\text{fixed}}, \text{Obs}_t)$$
其中$\mathcal{R}_{\text{fixed}}$为固定规则集，若无一匹配则返回原始观测。

**变化点**: 固定规则集无法覆盖长尾噪声模式，导致大量未匹配观测直接透传，冗余未减；且错误匹配时无法自我修复。TACO将规则集变为动态：匹配失败或压缩错误时即时生成新规则或修改现有规则。

**本文公式（推导）**:
$$\text{Step 1}: \quad \text{Match}_t = \{r \in \mathcal{R}_{\text{active}}^{(t)} : \text{pattern}(r) \text{ matches } \text{Obs}_t\} \quad \text{（活跃规则匹配当前观测）}$$
$$\text{Step 2}: \quad \text{Compressed}_t = \begin{cases} \text{transform}(r^*, \text{Obs}_t) & \text{if } \text{Match}_t \neq \emptyset, r^* = \text{arg}\max_{r} \text{priority}(r) \\ \text{Obs}_t & \text{otherwise} \end{cases} \quad \text{（执行最高优先级匹配规则）}$$
$$\text{Step 3}: \quad \text{if compression error detected:} \quad \mathcal{R}_{\text{active}}^{(t+1)} = \text{SpawnOrModify}(\mathcal{R}_{\text{active}}^{(t)}, \text{Obs}_t, \text{error signal}) \quad \text{（错误驱动规则进化）}$$
$$\text{最终}: \quad \mathcal{R}_{\text{active}}^{(t)} \text{ evolves within-task; } \mathcal{R}_{\text{global}} \leftarrow \text{Update}(\mathcal{R}_{\text{active}}^{(T)}) \text{ post-task}$$

符号: $\mathcal{R}_{\text{active}}^{(t)}$ = 时刻t的活跃规则集，$\mathcal{R}_{\text{global}}$ = 全局规则池，$\text{Obs}_t$ = 时刻t的原始终端观测，"compression error" = 压缩后信息丢失导致智能体决策失败的反馈信号。

**对应消融**: 去除任务内规则集进化（即仅使用初始化的活跃规则集，无在线更新），准确率下降1.7%（绝对值），token开销激增30.7%。该组件是TACO性能提升的核心来源。

### 模块 3: Top-k规则检索与冷启动（对应框架图 模块A）

**直觉**: 跨任务复用需要解决"第一个任务用什么规则"的冷启动问题，相似任务的规则应被优先检索。

**Baseline 公式** (无跨任务复用): 
$$\mathcal{R}_{\text{active}}^{(0)} = \mathcal{R}_{\text{seed}} \quad \text{或} \quad \emptyset$$
即仅使用少量种子规则或完全空启动。

**变化点**: 空启动导致早期交互无压缩可用，种子规则覆盖有限。TACO通过全局规则池的Top-k检索实现任务感知的冷启动。

**本文公式（推导）**:
$$\text{Step 1}: \quad \text{similarity}(r, \text{task}_i) = \text{Embed}(r) \cdot \text{Embed}(\text{task}_i) \quad \text{（规则与当前任务的嵌入相似度）}$$
$$\text{Step 2}: \quad \mathcal{R}_{\text{active}}^{(0)} = \text{TopK}_{r \in \mathcal{R}_{\text{global}}}\{\text{similarity}(r, \text{task}_i)\} \quad \text{（检索最相关的k条规则）}$$
$$\text{最终}: \quad \mathcal{R}_{\text{active}}^{(0)} \subseteq \mathcal{R}_{\text{global}}, \quad |\mathcal{R}_{\text{active}}^{(0)}| = k$$

符号: $\text{task}_i$ = 当前任务描述/上下文，$k$ = 检索规则数（超参数，Figure 6显示其对准确率与token成本的影响）。

**对应消融**: 

## 实验与分析

主实验结果（TerminalBench 2.0，具体backbone模型未注明）：

| Method | Accuracy (%) | Token Overhead | Δ Accuracy | Δ Token |
|:---|:---|:---|:---|:---|
| Baseline (无压缩) | 40.6 | baseline | — | — |
| TACO Full | 42.7 | +12.2% | **+2.1** | +12.2% |
| w/o Intra-Task Evolution | 41.0 (估计) | +42.9% | -1.7 | +30.7% |
| w/o Global Pool Evolution | 42.5 (估计) | +30.3% | -0.2 | +18.1% |


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/fafe0aab-8feb-41a8-aab9-07407d2b91ef/figures/Figure_3.png)
*Figure 3: Figure 3: Agent Accuracy Under Identical Token Budgets. Across a range of fixed token budgets, wecompare the accuracy achieved by Baseline and TACO. On all six models, TACO consistently attainshigher*



**核心发现分析**:
- **准确率提升来源**: +2.1%的准确率提升几乎完全来自任务内在线规则进化（Intra-Task Evolution）。去除该组件导致-1.7%准确率下降，说明固定规则集（即使经全局池初始化）无法覆盖任务特有的长尾模式，实时适应是关键。
- **token开销矛盾**: TACO Full相比Baseline反而增加12.2% token开销，与摘要声称的"约10% token节省"存在口径不一致。后者仅适用于MiniMax-2.5特定场景，不宜泛化。消融显示去除全局规则池进化导致token开销再增18.1%，说明全局池确实贡献效率，但不足以抵消任务内进化的额外成本。
- **组件贡献不对称**: 两级进化机制分工明确——任务内进化负责准确率，全局规则池负责效率。全局池对准确率贡献边际（-0.2%），但对抑制token增长至关重要。


![Figure 5](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/fafe0aab-8feb-41a8-aab9-07407d2b91ef/figures/Figure_5.png)
*Figure 5: Figure 5: Rule-frontier convergence and performance stabil-ity. (a) Top-30 rule retention between runs (dashed line: 90%threshold). (b) Rolling standard deviation of task accuracy(dotted lines: static*



**与静态规则对比**（Figure 1b）: TACO优于6条预定义规则（Seed Rules）和200条人工规则（High-Quality Rules），支持自进化机制相对固定规则的优越性，但具体数值未在提供文本中列出。

**Pass@k对比**（Figure 4）: 在六个模型（上层：大模型；下层：）上比较Baseline与TACO的Pass@k，具体数值未在提供文本中列出。

**超参数分析**（Figure 6）: Top-k规则检索大小对准确率与自进化token成本的影响，以及（右侧）另一超参数的影响，具体最优值未在提供文本中列出。

**公平性检查与局限**:
- **Baseline强度**: 主要对比无压缩Baseline和静态规则，缺乏与其他LLM-based动态压缩方法（如在线摘要、自适应提示压缩）的系统性定量对比。
- **复现性**: 消融表未注明backbone模型；所有结果为单次运行，未报告方差。
- **泛化声明**: SWE-Bench Lite、CompileBench、DevEval、CRUST-Bench四个额外基准的具体数值完全缺失，无法独立核实。
- **失败模式**: 压缩错误检测与修复机制依赖错误信号定义，若智能体对压缩后信息丢失不敏感（即未触发错误），可能导致静默失败。

## 方法谱系与知识库定位

**方法家族**: 终端智能体上下文压缩 / 观测压缩（Observation Compression）

**Parent Method**: 基于规则的文本压缩系统（如传统log parsers、SWE-Pruner的剪枝思想）。TACO将"规则"从静态配置提升为动态进化的一等公民，将"适应性"从模型权重层面下移到结构化文本规则层面。

**改动插槽**:
- **架构**: 新增规则池（全局+活跃）双层结构，插件式嵌入现有Agent框架
- **目标函数**: 无显式损失函数，基于规则匹配成功率（$c_g$）和使用频率（$n_g$）的隐式优化
- **训练配方**: 零训练、零梯度更新，完全依赖执行时反馈
- **数据策划**: 无预训练数据需求，规则从在线交互中自动挖掘
- **推理**: 每一步增加规则匹配开销，但减少上下文token量

**Direct Baselines对比**:
- **SWE-Pruner**: 需微调，绑定SWE-Bench任务；TACO零训练，框架无关
- **High-Quality Rules (200条人工规则)**: 静态固定，跨任务不稳定；TACO在线进化，任务自适应
- **LLM通用摘要**: 无结构感知，额外LLM调用；TACO利用终端结构，规则执行轻量

**Follow-up方向**:
1. **与参数化方法融合**: 将进化规则作为课程信号，指导轻量级压缩模型的训练，兼顾自适应与执行效率
2. **多智能体规则协作**: 多个终端智能体共享全局规则池，加速规则收敛（Figure 5a显示top-30规则保留率已接近90%阈值，具备协作基础）
3. **错误信号精细化**: 当前压缩错误检测较粗粒度，引入更细粒度的决策影响评估可提升进化效率

**知识库标签**: 终端智能体(terminal agent) / 上下文压缩(context compression) / 自进化(self-evolving) / 零训练插件(zero-training plugin) / 规则进化(rule evolution) / 观测压缩(observational compression) / 软件工程智能体(SWE agent)

