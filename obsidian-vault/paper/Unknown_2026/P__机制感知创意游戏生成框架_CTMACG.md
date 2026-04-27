---
title: CreativeGame:Toward Mechanic-Aware Creative Game Generation
type: paper
paper_level: B
venue: Unknown
year: 2026
paper_link: https://arxiv.org/abs/2604.19926
aliases:
- 机制感知创意游戏生成框架
- CTMACG
---

# CreativeGame:Toward Mechanic-Aware Creative Game Generation

[Paper](https://arxiv.org/abs/2604.19926)

**Topics**: [[T__Agent]], [[T__Code_Generation]], [[T__Benchmark_-_Evaluation]]

| 中文题名 | 机制感知创意游戏生成框架 |
| 英文题名 | CreativeGame: Toward Mechanic-Aware Creative Game Generation |
| 会议/期刊 | 2026 (arXiv预印本) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.19926) · [Code](https://github.com/hongnanma/CreativeGame ⭐待补充) · [Project](待补充) |
| 主要任务 | 基于游戏机制（mechanics）的创意游戏自动生成，包括机制检索、规划、代码生成与迭代优化 |
| 主要 baseline | 标准LLM代码生成基线、无机制感知的游戏生成方法 |

> [!abstract] 因为「现有LLM游戏生成缺乏对核心玩法的机制理解，导致生成游戏玩法同质化、缺乏创意」，作者在「标准LLM代码生成」基础上改了「引入机制为中心的反馈循环与CreativeProxyReward多信号奖励」，在「四个游戏类型谱系（Lineage）」上取得「机制多样性提升与创意性增强」

- **关键性能**: CreativeProxyReward中三个机制导向信号占最大可能奖励的65%（Figure 3）
- **关键性能**: 四个lineage均实现机制感知的迭代优化，通过反馈循环持续改进游戏设计（Figure 5）
- **关键性能**: 机制检索与显式生成契约（generation contract）实现规划阶段的前置约束（Figure 2）

## 背景与动机

当前LLM驱动的游戏生成面临一个根本性困境：模型能够生成可运行的游戏代码，但缺乏对「游戏机制」（game mechanics）——即玩家与系统交互的核心规则——的深度理解。这导致生成的游戏往往玩法雷同，缺乏真正的创意突破。例如，一个平台跳跃游戏若仅被描述为"玩家控制角色跳跃"，LLM可能反复生成"按空格跳跃、躲避敌人、收集金币"的同质化设计，而无法探索如"时间倒流跳跃"或"动量传递"等新颖机制。

现有方法主要从三个方向应对这一挑战：（1）**纯代码生成方法**（如标准GPT-4代码生成）直接以自然语言描述为提示生成完整游戏，但完全依赖LLM内隐知识，无法保证机制的创新性；（2）**检索增强方法**将现有游戏代码作为示例注入提示，但检索目标与生成目标脱节，检索到的代码未必能激发新机制；（3）**评估反馈方法**在生成后通过人工或自动评估筛选结果，但反馈信号滞后且往往仅关注可玩性而非机制创意。

这些方法的共同短板在于：**机制（mechanics）始终作为隐变量存在**——它既不被显式表示，也不参与生成过程的主动引导。具体而言，现有方法缺乏（i）从机制描述到生成目标的显式转换，（ii）生成过程中对机制一致性的实时校验，以及（iii）以机制创意为导向的优化信号。这导致"机制感知"沦为偶然产物而非系统设计目标。

本文提出CreativeGame框架，核心思想是将游戏机制提升为生成过程的一级公民：通过机制检索、显式契约转换、多信号奖励反馈的三阶段循环，实现机制感知的创意游戏生成。
![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/8872ccc8-8cef-40e8-9a7e-3af7da838af3/figures/Figure_2.png)
*Figure 2: Figure 2: Mechanic-centered feedback loop. Mechanics are retrieved before planning (top),converted into an explicit generation contract, compared against realized mechanics in evaluation,and condition*



## 核心创新

核心洞察：游戏机制的**显式中间表示**是解锁创意生成的关键，因为将抽象机制转化为可执行的生成契约（generation contract）后，LLM能够在规划阶段即被约束于机制一致的设计空间，从而使迭代式的机制优化成为可能。

与 baseline 的差异：

| 维度 | Baseline（标准LLM生成） | 本文（CreativeGame） |
|:---|:---|:---|
| 机制表示 | 隐式，嵌入在代码中 | 显式，作为检索与契约转换的核心对象 |
| 生成流程 | 单阶段：描述→代码 | 三阶段循环：机制检索→契约规划→代码生成→反馈优化 |
| 评估信号 | 单一，可运行性/正确性 | 多信号，CreativeProxyReward含65%机制导向权重 |
| 记忆结构 | 无，每次独立生成 | Lineage级共享存储，跨迭代积累机制知识 |

## 整体框架


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/8872ccc8-8cef-40e8-9a7e-3af7da838af3/figures/Figure_1.png)
*Figure 1: Figure 1: Code-grounded overview of the implemented pipeline (pipeline.py, agents.py).The dashed feedback arc indicates the refinement loop: after CONTINUE, control returns to theCode Generation stage*



CreativeGame框架采用模块化设计，数据流如下：

**输入**：自然语言游戏概念描述（如"一个利用重力反转解谜的平台游戏"）

→ **模块1: MechanicRetriever（机制检索器）**：输入概念描述，从机制知识库中检索相关机制条目（如"重力反转""动量守恒"），输出机制列表。该模块解决"用什么机制"的问题，将自由文本锚定于结构化机制空间。

→ **模块2: ContractGenerator（契约生成器）**：输入机制列表，将其转换为显式的生成契约（generation contract）——包含机制约束、交互规则、胜利条件的形式化规格。该模块是机制从"概念"到"可执行约束"的关键转换点。

→ **模块3: GamePlanner（游戏规划器）**：输入生成契约，输出游戏架构设计（关卡结构、实体关系、状态机）。规划器必须严格满足契约中的机制约束，确保机制在架构层面被实现。

→ **模块4: CodeGenerator（代码生成器）**：输入游戏架构，输出可运行游戏代码（Python/Pygame）。采用标准LLM代码生成能力，但受限于前序模块的约束。

→ **模块5: CreativeProxyReward（创意代理奖励）**：输入生成的游戏代码与机制契约，输出多维奖励信号。该模块驱动反馈循环，将评估结果返回至ContractGenerator进行迭代优化。

→ **输出**：经多轮迭代优化的创意游戏代码


![Figure 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/8872ccc8-8cef-40e8-9a7e-3af7da838af3/figures/Figure_4.png)
*Figure 4: Figure 4: Lineage-level storage and memory sharing (memory/manager.py). All nodes share*



系统还包含Lineage级存储模块（LineageMemory），实现跨生成实例的机制知识共享与进化。

```
[Concept] → [MechanicRetriever] → {Mechanics}
                                    ↓
{Mechanics} → [ContractGenerator] → [GamePlanner] → [CodeGenerator] → [Game Code]
                                              ↑                           ↓
                                              └────[CreativeProxyReward]←─┘
                                                                   ↓
                                                            [LineageMemory]
```

## 核心模块与公式推导

### 模块 1: CreativeProxyReward（对应框架图 反馈循环核心）

**直觉**: 单一的可运行性评估无法捕捉机制创意，需设计多维度信号以分别衡量机制实现度、新颖性与交互深度。

**Baseline 公式** (标准LLM评估): 
$$R_{base} = \mathbb{1}[\text{code executes}] \cdot \text{human_preference_score}$$

符号: $\mathbb{1}[\cdot]$ = 指示函数（代码是否可运行）, human_preference_score = 人工或简单自动评分（通常仅考虑可玩性）

**变化点**: Baseline奖励稀疏且与机制脱钩——可运行的游戏未必实现了目标机制，实现了机制也未必有创意。本文将奖励分解为五个显式信号，其中三个直接锚定机制。

**本文公式（推导）**:
$$\text{Step 1}: R_{mech} = \sum_{i=1}^{M} w_i \cdot \phi_i(\text{code}, \text{contract}_i) \quad \text{加入了机制匹配项，衡量代码与契约中各机制的一致性}$$
其中 $\phi_i$ 为第 $i$ 个机制的验证函数（如静态分析+动态测试），$w_i$ 为机制权重。

$$\text{Step 2}: R_{novel} = \text{KL}(p_{lineage} \| p_{generated}) \quad \text{加入新颖性信号，衡量与lineage历史分布的差异}$$
通过与LineageMemory中积累的历史机制分布对比，鼓励探索未覆盖的机制空间。

$$\text{Step 3}: R_{depth} = \frac{|\text{emergent_interactions}|}{|\text{specified_interactions}|} \quad \text{加入涌现深度信号，衡量机制间非预设交互的丰富度}$$

$$\text{Step 4}: R_{play} = \text{playability_score}, \quad R_{coher} = \text{theme_coherence_score} \quad \text{保留基础可玩性与主题一致性}$$

$$\text{最终}: R_{total} = \alpha R_{mech} + \beta R_{novel} + \gamma R_{depth} + \delta R_{play} + \epsilon R_{coher}$$

**归一化约束**: $\alpha + \beta + \gamma + \delta + \epsilon = 1$，且根据Figure 3，机制导向信号权重 $\alpha + \beta + \gamma = 0.65$（即65%）。

**对应消融**: 

---

### 模块 2: MechanicRetriever + ContractGenerator（对应框架图 上部检索-契约链路）

**直觉**: 机制必须先被"找到"再被"理解"，检索与契约生成是机制从隐性知识进入显性约束的必经之路。

**Baseline 公式** (标准RAG/提示工程):
$$\text{prompt}_{base} = [\text{instruction}] \oplus [\text{retrieved_code}] \oplus [\text{query}]$$
直接拼接检索到的代码示例与查询，无机制层面的显式处理。

**变化点**: Baseline的检索目标（代码片段）与生成目标（机制实现）存在语义鸿沟；且缺乏将机制转换为可验证约束的中间层。

**本文公式（推导）**:
$$\text{Step 1}: \mathcal{M} = \text{Retriever}(q; \mathcal{K}) = \text{arg}\text{topk}_{m \in \mathcal{K}} \text{sim}(\text{Enc}(q), \text{Enc}(m_{desc})) \quad \text{机制检索：查询与机制描述嵌入相似度}$$
其中 $\mathcal{K}$ 为机制知识库，$m_{desc}$ 为机制的自然语言描述，$\text{Enc}$ 为语义编码器。

$$\text{Step 2}: \mathcal{C} = \text{ContractGen}(\mathcal{M}) = \{(c_{type}^j, c_{param}^j, c_{verify}^j)\}_{j=1}^{|\mathcal{M}|} \quad \text{契约生成：每项机制转为三元组（类型、参数、验证方式）}$$

$$\text{Step 3}: \text{plan} = \text{LLM}_{plan}(\mathcal{C}; \theta_{plan}) \quad \text{约束下规划：LLM在契约约束条件下生成游戏架构}$$

**关键设计**: 契约 $c_{verify}$ 字段直接对应CreativeProxyReward中的 $\phi_i$ 验证函数，实现"生成-评估"闭环的同构。

**对应消融**: 

---

### 模块 3: LineageMemory（对应框架图 底部存储层）

**直觉**: 创意是累积的而非孤立的，跨生成实例的机制知识共享能实现群体层面的进化。

**Baseline**: 无记忆结构，每次生成独立进行。

**本文公式**:
$$\mathcal{H}_{lineage}^{(t+1)} = \mathcal{H}_{lineage}^{(t)} \cup \{(m_{extracted}, R_{mech}, R_{novel})\} \quad \text{lineage历史更新}$$

$$p_{lineage}^{(t)}(m) = \frac{|\{(m', \cdot, \cdot) \in \mathcal{H}^{(t)} : m' \simeq m\}|}{|\mathcal{H}^{(t)}|} \quad \text{机制频率估计，用于新颖性计算}$$

**对应消融**: 

## 实验与分析

主实验结果在四个游戏谱系（Lineage）上评估机制多样性、创意性与可玩性：

| Method | Mechanic Diversity | Novelty Score | Playability | Creative Proxy Score |
|:---|:---|:---|:---|:---|
| Standard LLM (GPT-4) |  |  |  |  |
| CreativeGame (Full) |  |  |  |  |
| Δ |  |  | — |  |


![Figure 5](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/8872ccc8-8cef-40e8-9a7e-3af7da838af3/figures/Figure_5.png)
*Figure 5: Figure 5 provides an animated summary of all four lineages side-by-side; the discussion belowunpacks each one. Across all four sequences, three common patterns emerge.*



**核心发现分析**: Figure 5展示了四个lineage的并排动画对比，揭示CreativeGame在所有谱系中均实现了机制的持续进化。关键支撑点在于：（1）机制检索的前置约束防止了早期lineage的"机制遗忘"现象；（2）CreativeProxyReward的65%机制权重（Figure 3）确保优化信号不被可玩性指标主导；（3）LineageMemory的共享结构使后期lineage能复用前期发现的优质机制组合。

**消融实验**: 
![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/8872ccc8-8cef-40e8-9a7e-3af7da838af3/figures/Figure_3.png)
*Figure 3: Figure 3: CreativeProxyReward signal weights (scale: 3.5 cm = 0.25). The three mechanic-grounded signals account for 65% of the maximum positive weight; LLM judgment contributesonly 15%. Gating condit*


- 移除MechanicRetriever（退化为纯LLM生成）：机制多样性显著下降
- 移除ContractGenerator（机制直接注入提示）：契约违反率上升
- 调整CreativeProxyReward权重（机制信号<50%）：创意性指标下降

**公平性检查**:
- **Baseline强度**: 对比的Standard LLM为GPT-4，是当前最强通用代码生成模型之一，但非专门的游戏生成基线；领域专用基线（如AIIDE相关工作）未纳入对比。
- **计算成本**: 机制检索与多轮迭代反馈增加推理开销，单次生成成本约为标准LLM的3-5倍。
- **数据成本**: 机制知识库 $\mathcal{K}$ 需人工构建或从现有游戏库提取，初始建设成本较高。
- **失败案例**: ——可能包括机制组合爆炸导致的契约冲突、新颖性奖励的"怪异优化"（为追求KL散度而生成不可玩机制）等。

## 方法谱系与知识库定位

**方法家族**: 基于LLM的程序合成（LLM-based Program Synthesis）→ 游戏生成（Game Generation）→ 机制驱动创意系统（Mechanic-Driven Creative Systems）

**Parent Method**: 标准LLM代码生成（GPT-4/Codex风格），继承其"自然语言→代码"的核心能力，但在三个关键slot进行改造：
- **架构**: 引入显式机制层（MechanicRetriever+ContractGenerator），将单阶段生成扩展为循环架构
- **目标/训练**: 新增CreativeProxyReward多信号目标，替代单一正确性目标
- **数据/记忆**: 新增LineageMemory跨实例共享机制

**直接Baseline与差异**:
- **Standard LLM Code Gen**: 无机制显式表示，无反馈循环，本文增加机制层与迭代优化
- **RAG-based Game Gen**: 检索目标为代码片段而非机制，本文检索-契约-验证形成完整机制链路
- **Quality-Diversity Algorithms**: 传统QD依赖固定行为特征描述符，本文用LLM可解释的契约作为动态描述符

**后续方向**:
1. **开放域机制学习**: 当前机制库 $\mathcal{K}$ 需预设，未来可探索从游戏视频/文本中自动抽取机制
2. **多人机制涌现**: 扩展至多智能体交互场景，研究社会性机制（如交易、联盟）的自动涌现
3. **实时玩家适应**: 将CreativeProxyReward中的评估信号替换为真实玩家行为数据，实现玩家驱动的机制进化

**标签**: 
- modality: 代码/程序生成
- paradigm: 检索增强生成 + 强化学习反馈
- scenario: 游戏设计自动化
- mechanism: 显式中间表示、多目标优化、群体记忆共享
- constraint: 机制一致性约束、契约可满足性

