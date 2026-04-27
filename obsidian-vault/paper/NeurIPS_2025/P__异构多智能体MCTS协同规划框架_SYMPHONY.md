---
title: 'SYMPHONY: Synergistic Multi-agent Planning with Heterogeneous Language Model Assembly'
type: paper
paper_level: A
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 异构多智能体MCTS协同规划框架
- SYMPHONY
- A heterogeneous multi-agent plannin
acceptance: Poster
cited_by: 5
code_url: https://github.com/ZHUWEI-hub/SYMPHONY
method: SYMPHONY
modalities:
- Text
paradigm: supervised
---

# SYMPHONY: Synergistic Multi-agent Planning with Heterogeneous Language Model Assembly

[Code](https://github.com/ZHUWEI-hub/SYMPHONY)

**Topics**: [[T__Reasoning]], [[T__Code_Generation]] | **Method**: [[M__SYMPHONY]] | **Datasets**: HotpotQA, WebShop, MBPP-Python, MBPP-Rust

> [!tip] 核心洞察
> A heterogeneous multi-agent planning framework that integrates diverse language model-based agents with dynamic scheduling via UCB-based exploration can significantly enhance rollout diversity and planning performance compared to single-agent MCTS approaches.

| 中文题名 | 异构多智能体MCTS协同规划框架 |
| 英文题名 | SYMPHONY: Synergistic Multi-agent Planning with Heterogeneous Language Model Assembly |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2601.22623) · [Code](https://github.com/ZHUWEI-hub/SYMPHONY) · [DOI](https://doi.org/10.48550/arxiv.2601.22623) |
| 主要任务 | Reasoning, Code Generation |
| 主要 baseline | RAP, LATS, Claude-3.5-Sonnet (single-agent), MASTER, AgentVerse |

> [!abstract] 因为「单LLM多次采样产生的rollout高度相似、缺乏多样性」，作者在「LATS单智能体MCTS」基础上改了「异构多智能体UCB动态调度机制」，在「HotpotQA/WebShop/MBPP」上取得「EM 0.76 / Score 0.82 / pass@1 0.947」

- **HotpotQA EM**: SYMPHONY 0.76 vs Claude-3.5-Sonnet 0.51，绝对提升 +0.25（+49%）
- **WebShop SR**: SYMPHONY 0.61 vs Claude-3.5-Sonnet 0.41，绝对提升 +0.20（+48.8%）
- **MBPP-Python pass@1**: SYMPHONY 0.947 vs Claude-3.5-Sonnet 0.894，绝对提升 +0.053（+5.9%）

## 背景与动机

现有LLM-based规划方法普遍采用单智能体MCTS框架：同一个大语言模型被重复查询多次，通过temperature采样或prompt扰动来生成搜索分支并估计价值。然而，这种范式存在一个根本性缺陷——同一模型的多次输出具有高度相似性，reflecting dominant reasoning patterns，导致rollout diversity严重不足，搜索树探索效率低下。

具体而言，RAP（Reasoning with Language Model is Planning with World Model）将LLM同时用作world model和value estimator，通过单模型多次rollout构建搜索树；LATS（Language Agents Tree Search）在此基础上优化了cost-effectiveness，但同样依赖固定单模型（K=50次rollout，n=5个分支）。这些方法假设「对同一模型施加随机性即可获得足够多样的探索」，但实证研究表明[14, 46, 12]，即使采用temperature scaling或adversarial prompting，输出相似度仍然很高。

核心局限在于：**stochasticity ≠ diversity**。同一模型的采样变异局限于其训练分布内的reasoning pattern，无法突破模型固有的认知盲区。当遇到需要complementary reasoning的critical nodes时，单智能体容易陷入local optima，反复生成结构相似的错误路径。

本文提出SYMPHONY，核心思想是将MCTS中的「动作选择」升级为「智能体选择」——用异构LLM智能体池替代单一模型，通过UCB机制动态调度不同模型，使diverse reasoning patterns在共享搜索树中协同互补。
![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/d77bdea1-a810-480c-a219-506fb7d4f0ca/figures/fig_001.png)
*Figure: SYMPHONY System Overview.*



## 核心创新

核心洞察：**异构模型的intrinsic diversity优于单模型的artificial stochasticity**，因为不同架构、训练数据和优化目标的LLM天然具备互补的reasoning patterns，从而使「低成本高覆盖的协同探索」成为可能。

| 维度 | Baseline (LATS/RAP) | 本文 (SYMPHONY) |
|:---|:---|:---|
| 探索来源 | 单模型temperature采样 / prompt扰动 | 异构模型池（Claude/GPT-4/开源模型）的intrinsic多样性 |
| 调度机制 | 固定模型，随机或贪心选择 | UCB-based动态调度（α=20），平衡exploitation与exploration |
| 价值聚合 | 单模型visit统计 | 多智能体身份感知的共享backpropagation |
| 资源约束 | K=50, n=5（高成本） | K=10, n∈{2,3,4}（consumer hardware可行） |

## 整体框架


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/d77bdea1-a810-480c-a219-506fb7d4f0ca/figures/fig_002.png)
*Figure: Branch Diversity vs. Task Performance. Bars and left y-axis shows the branch diversity,*



SYMPHONY的整体框架是一个多智能体协作的MCTS系统，包含四个核心模块：

**Agent Selection (UCB Scheduler)**：输入当前节点状态、各智能体历史表现及访问计数，输出被选中的智能体。该模块是本文核心创新，将传统MCTS中的action selection转化为agent selection，通过UCB公式动态平衡高性能智能体的利用与未充分探索智能体的发掘。

**Heterogeneous Rollout Generation**：输入被选中的智能体及当前节点状态，输出diverse rollout trajectory。与baseline的单模型生成不同，此处不同架构的LLM（如Claude-3.5-Sonnet、GPT-4、Llama系列）基于各自独特的reasoning pattern生成路径，天然具备互补性。

**Value Estimation**：输入rollout outcome，输出reward signal。该模块继承标准MCTS设计，但评估的是异构智能体生成的多样化轨迹。

**Multi-Agent Backpropagation**：输入rollout reward及智能体身份标识，输出更新后的共享节点统计（Q-values、visit counts、agent-specific statistics）。关键创新在于保留智能体身份provenance，使调度器能基于每个智能体的历史表现优化未来选择。

数据流：任务输入 → 根节点UCB调度 → 选定智能体生成rollout → 价值评估 → 多智能体反向传播更新统计 → 下一节点基于更新后统计重新调度 → 迭代至终止条件 → 输出最优路径。

```
[Task Input] → [UCB Scheduler] → {Agent i from Pool}
                                    ↓
                           [Rollout Generator_i]
                                    ↓
                              [Value Estimator]
                                    ↓
                         [Multi-Agent Backprop]
                                    ↓
                              [Shared Tree]
                                    ↓
                         (iterate until budget K=10)
                                    ↓
                              [Best Path Output]
```

## 核心模块与公式推导

### 模块 1: UCB-based Agent Scheduler（对应框架图 左/上部）

**直觉**：传统MCTS用UCB选择「动作」，但LLM planning中动作空间巨大且模糊；转而用UCB选择「智能体」，将diversity来源从采样噪声升级为模型异构性。

**Baseline 公式** (LATS/RAP [10], 标准MCTS action selection):
$$UCB(a) = \bar{Q}(s,a) + c\sqrt{\frac{\ln N(s)}{N(s,a)}}$$
符号: $s$ = 当前节点状态, $a$ = 候选动作, $\bar{Q}(s,a)$ = 动作平均价值, $N(s)$ = 节点访问次数, $N(s,a)$ = 动作访问次数, $c$ = 探索常数。

**变化点**：标准UCB选择「下一步动作」，但LLM规划中动作是自由文本，难以定义有限动作空间；且同一模型的不同动作仍受限于单一reasoning pattern。本文将UCB从action space映射到agent space。

**本文公式（推导）**:
$$\text{Step 1}: UCB_i = \bar{R}_i + \alpha \sqrt{\frac{\ln N}{N_i}} \quad \text{将动作}a替换为智能体i，状态统计聚合为全局计数$$
$$\text{Step 2}: \text{其中}\bar{R}_i = \frac{1}{N_i}\sum_{k=1}^{N_i} R_{i,k}, \quad N = \sum_{i\in\mathcal{A}} N_i \quad \text{重归一化以保证跨智能体可比性}$$
$$\text{最终}: i^* = \text{arg}\max_{i\in\mathcal{A}} \left[ \bar{R}_i + \alpha \sqrt{\frac{\ln N}{N_i}} \right]$$
符号: $\mathcal{A}$ = 异构智能体集合, $\bar{R}_i$ = 智能体$i$平均奖励, $N_i$ = 智能体$i$访问次数, $\alpha=20$ = 探索系数（经$\alpha\in\{0.1,0.5,1,2,5,10,20,30\}$调参选定）。

**对应消融**：α参数研究显示，小α导致overexploitation陷入local minima，α=20最优；Figure 4展示不同α值下HotpotQA/WebShop/MBPP性能曲线。

---

### 模块 2: Multi-Agent Value Backpropagation（对应框架图 右/下部）

**直觉**：异构智能体的贡献必须身份感知的聚合，才能支撑调度器的历史依赖决策；同时共享树结构使所有智能体受益于集体探索。

**Baseline 公式** (标准单智能体MCTS):
$$Q(s) \leftarrow \frac{1}{N(s)} \sum_{k=1}^{N(s)} R_k(s)$$
符号: $Q(s)$ = 节点$s$价值估计, $N(s)$ = 总访问次数, $R_k(s)$ = 第$k$次访问累积奖励。

**变化点**：单智能体公式假设所有rollout来自同质分布；本文多智能体场景中，不同智能体的reward分布具有不同bias-variance特性，需显式追踪provenance以支持UCB调度。

**本文公式（推导）**:
$$\text{Step 1}: Q(s) \leftarrow \frac{1}{N(s)} \sum_{i\in\mathcal{A}} \sum_{k=1}^{N_i(s)} R_{i,k}(s) \quad \text{按智能体身份分解双重求和，保留}N_i(s)\text{统计}$$
$$\text{Step 2}: N(s) = \sum_{i\in\mathcal{A}} N_i(s), \quad \bar{R}_i(s) = \frac{1}{N_i(s)}\sum_{k=1}^{N_i(s)} R_{i,k}(s) \quad \text{维护agent-specific running statistics}$$
$$\text{最终}: Q(s) = \sum_{i\in\mathcal{A}} \frac{N_i(s)}{N(s)} \cdot \bar{R}_i(s)$$
该形式等价于各智能体价值的加权平均，权重为其相对访问频率，自然融入UCB调度器的$N_i$计数。

**对应消融**：Table 结果显示，SYMPHONY-S（缩减模型池）HotpotQA EM降至0.59（-0.17），WebShop SR降至0.56（-0.05），验证完整异构池的必要性。

## 实验与分析



本文在三个代表性benchmark上评估SYMPHONY：多跳推理（HotpotQA）、web导航决策（WebShop）、代码生成（MBPP-Python/Rust）。核心发现是：在显著降低资源消耗的前提下（K=10 vs LATS的K=50，n≤4 vs n=5），SYMPHONY全面超越单智能体强baseline Claude-3.5-Sonnet。具体而言，HotpotQA上EM达到0.76，相比单智能体的0.51提升+0.25（+49%）；WebShop上Score 0.82 / SR 0.61，相比0.71/0.41分别提升+15.5%/+48.8%；MBPP-Python pass@1达0.947，MBPP-Rust达0.951，均显著优于Claude-3.5-Sonnet的0.894/0.903。这一结果表明，异构协同不仅提升复杂推理任务的绝对性能，在代码生成这类相对"确定性"的任务上也能带来稳健增益。



消融实验揭示了关键设计决策的敏感度。分支数n的削减造成最剧烈的性能崩塌：HotpotQA EM从n=4时的0.59降至n=3的0.47（-0.12）和n=2的0.34（-0.25），WebShop SR降至0.46/0.35，MBPP-Python骤降至0.869/0.684。这说明**diversity的结构性保障**（通过多分支展开）比模型异构性本身更为基础——即使有了异构智能体池，过少的分支仍无法有效覆盖搜索空间。值得注意的是，SYMPHONY-S（缩减模型池）在WebShop Score上与完整版持平（均为0.82），暗示该metric可能存在ceiling effect或insensitivity，需结合SR等其他指标综合判断。



替代diversity增强策略被证伪：adversarial prompting和temperature scaling（范围[0,2]）均「failed to achieve comparable task performance」，说明人工扰动无法替代异构模型的intrinsic diversity。α参数研究中，α=20经网格搜索确定为最优，过小导致overexploitation，过大则响应不足。

公平性检查：本文未在提取文本中展示与MASTER [12]的定量对比（仅列为secondary baseline），也未与LATS在相同K/n设置下直接比较。SYMPHONY-S与SYMPHONY使用不同模型池，其性能差异（0.59 vs 0.76 EM）可能混淆了「调度机制」与「模型能力」两个因素。此外，WebShop Score的ceiling effect提示需多指标联合评估。作者明确披露：性能依赖于足够多样的异构模型可用性，α=20可能task-dependent，n≤4的限制可能miss broader exploration benefits。

## 方法谱系与知识库定位

SYMPHONY属于**LLM-MCTS演进谱系**，直接parent为LATS [12]和RAP [10]代表的单智能体MCTS方法。核心演进路径：从「单模型重复采样」到「多模型协同规划」，关键slot变化包括：

- **Architecture**: single-agent → multi-agent assembly（异构模型共享搜索树）
- **Exploration strategy**: temperature/prompt扰动 → UCB-based agent scheduling（α=20）
- **Inference strategy**: 固定模型K=50/n=5 → 动态模型选择K=10/n∈{2,3,4}
- **Data pipeline**: artificial diversity → intrinsic diversity from heterogeneous architectures

直接baseline对比：
- **LATS**: 同为MCTS框架，SYMPHONY将模型固定改为动态调度，资源降80%（K=10/50）而性能提升
- **RAP**: 同为LLM+MCTS，SYMPHONY从world model单角色扩展为多智能体协同
- **MASTER [12]**: 同为multi-agent MCTS，SYMPHONY强调heterogeneous而非specialized agents，UCB调度机制不同
- **AgentVerse [8]**: 多agent协作框架，但非MCTS-based，缺乏systematic search和value backpropagation

后续方向：（1）自适应α调度，根据任务复杂度动态调整探索系数；（2）模型池自动构建，基于任务特征选择最优异构组合；（3）将UCB agent scheduling扩展至其他search算法如A*或beam search。

**标签**: modality:text | paradigm:planning/search | scenario:multi-agent collaboration | mechanism:UCB scheduling + MCTS | constraint:consumer hardware/low resource

## 引用网络

### 直接 baseline（本文基于）

- AgentVerse: Facilitating Multi-Agent Collaboration and Exploring Emergent Behaviors _(ICLR 2024, 实验对比, 未深度分析)_: Multi-agent collaboration framework, likely appears in experiment tables as a co

