---
title: 'Chat2Workflow: A Benchmark for Generating Executable Visual Workflows with Natural Language'
type: paper
paper_level: B
venue: Unknown
year: 2026
paper_link: https://arxiv.org/abs/2604.19667
aliases:
- 可执行视觉工作流生成的评测基准
- Chat2Workflow
- 现有LLM评测体系以「任务完成率」为核心
method: Chat2Workflow
modalities:
- Text
- Image
paradigm: Reinforcement Learning
---

# Chat2Workflow: A Benchmark for Generating Executable Visual Workflows with Natural Language

[Paper](https://arxiv.org/abs/2604.19667)

**Topics**: [[T__Benchmark_-_Evaluation]], [[T__Text_Generation]] | **Method**: [[M__Chat2Workflow]]

> [!tip] 核心洞察
> 现有LLM评测体系以「任务完成率」为核心，忽视了工业部署中工作流的结构规范性与可执行性约束。Chat2Workflow的核心洞察是：格式合法（pass）≠执行成功（resolve），两者之间存在系统性鸿沟。通过引入双指标评估体系和真实平台执行验证，该基准将「可部署性」而非「格式正确性」确立为评估LLM工作流生成能力的核心标准。这一重新定义揭示了当前最优模型（71.59% resolve rate）与实用部署水平之间的真实差距，为后续研究提供了更诚实的能力边界刻画。

| 中文题名 | 可执行视觉工作流生成的评测基准 |
| 英文题名 | Chat2Workflow: A Benchmark for Generating Executable Visual Workflows with Natural Language |
| 会议/期刊 | arXiv 2026 (preprint) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.19667) · Code · Project |
| 主要任务 | 评估LLM生成可执行视觉工作流的能力（多轮对话→结构化YAML→Dify/Coze平台部署） |
| 主要 baseline | 15个代表性模型：Gemini-3-Pro-Preview、GPT-5.2、GPT-5.1、Claude-4-Opus、GLM-4.6、Qwen-3-235B等 |

> [!abstract] 因为「现有Agent评测基准忽视工作流的结构规范性与可执行性，导致格式合法≠执行成功」，作者在「传统端到端任务完成评测」基础上改了「引入pass rate与resolve rate双指标、真实平台执行验证、多轮对话迭代场景」，在「273实例覆盖6大领域的Chat2Workflow基准」上取得「当前最优模型Gemini-3-Pro-Preview resolve rate 71.59%，但所有15个模型均存在系统性虚高，GLM-4.6差距达20.96%」

- **关键性能1**：最优模型Gemini-3-Pro-Preview平均resolve rate为71.59%，pass rate与resolve rate差距显著
- **关键性能2**：GLM-4.6在Education场景pass-resolve差距达43.44%，揭示格式合法性严重高估实际可用性
- **关键性能3**：Qwen-3系列8B→32B→235B规模正向提升，但Qwen-3-Coder-480B（26.44%）低于235B（27.71%），打破简单规模定律

## 背景与动机

可执行视觉工作流（Executable Visual Workflow）已成为工业级Agent部署的主流范式。以Dify、Coze为代表的可视化编排平台中，超过70%的真实Agent部署依赖显式工作流编排而非权重微调。一个典型场景是：企业希望构建「自动处理客户退款申请」的Agent，开发者需在画布上拖拽条件判断节点、API调用节点、数据库查询节点，为每个节点编写提示词，并确保分支逻辑覆盖所有异常情况——这一过程目前几乎完全依赖人工工程，开发成本高、周期长、易出错。

现有方法如何应对？**WebArena**等端到端网页Agent评测基准关注任务最终完成率，但完全不检查中间过程的工作流结构是否规范；**SWE-bench**聚焦代码仓库编辑，其「解决方案」是自由形式代码而非结构化工作流；**AgentBench**等多任务评测虽涉及工具调用，但缺乏对视觉编排拓扑、节点间数据依赖、平台特定执行语义的约束。这些基准的共性假设是：只要任务完成即可，不关心「如何完成」的结构可复现性。

然而，工业部署场景对「如何完成」有刚性要求。真实业务需求往往复杂且隐含——用户说「帮我做个能根据学生进度调整学习计划的应用」，其中「根据进度调整」涉及循环迭代、条件分支、外部工具调用等多重控制流，仅凭自然语言难以准确推断；更棘手的是，用户需求频繁变化，要求工作流在保持正确性与一致性的前提下被修订或重新生成。现有基准无法评估LLM是否生成「结构正确、可直接部署」的工作流，导致研究社区对LLM在工业级工作流自动化上的真实能力边界缺乏清晰认知。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/1eb779af-8b90-40da-b570-e06f59ee6966/figures/Figure_1.png)
*Figure 1: Figure 1: An example task in Chat2Workflow, whichfeatures realistic, variable natural-language instructioninputs and produces outputs that can be directlytransformed and integrated into real-world wor*



本文的核心动机正是填补这一空白：构建专门评估LLM生成可执行视觉工作流能力的基准，将「可部署性」确立为核心评估标准。

## 核心创新

核心洞察：格式合法（pass）≠执行成功（resolve），两者之间存在系统性鸿沟，因为现有评测仅检查YAML语法正确性而忽视真实平台执行语义（如节点参数类型校验、环境变量解析、循环依赖检测），从而使「以可部署性为核心标准」的诚实能力评估成为可能。

| 维度 | Baseline（WebArena/SWE-bench等） | 本文（Chat2Workflow） |
|:---|:---|:---|
| 评估对象 | 端到端任务完成结果 | 中间工作流结构的可执行性 |
| 正确性标准 | 最终答案匹配/测试通过 | 真实平台（Dify 1.9.2/Coze）实际运行成功 |
| 交互假设 | 单轮静态输入 | 多轮对话迭代，模拟需求变更场景 |
| 核心指标 | 任务成功率（单一指标） | pass rate（格式合法）与resolve rate（执行成功）双指标分离 |
| 失败分析 | 仅报告是否成功 | 系统性量化pass-resolve差距，定位虚高来源 |

## 整体框架


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/1eb779af-8b90-40da-b570-e06f59ee6966/figures/Figure_3.png)
*Figure 3: Figure 3: Overview of Chat2Workflow benchmark construction and evaluation framework. Left: We collectworkflows from six task domains (Research, Document, Enterprise, Developer, Education, AIGC) and re*



Chat2Workflow的整体框架包含四个核心阶段，形成从数据采集到能力诊断的完整闭环：

**阶段一：领域工作流采集与种子构建**。从AIGC（22.2%）、Research（18.5%）、Document（18.5%）、Education（14.8%）、Enterprise（14.8%）、Developer（11.1%）六个真实业务领域收集工作流种子，经人工验证确保质量，最终形成273个实例的数据集。每个实例包含自然语言任务描述、对应的标准工作流YAML、以及多轮对话修订历史。

**阶段二：节点类型规范化与平台适配**。将真实平台繁杂的节点类型归约为20种高频节点（如LLM节点、条件分支节点、代码执行节点、知识检索节点等），以YAML结构化格式存储，保证可执行性验证的可操作性，同时支持直接部署至Dify 1.9.2或Coze平台。

**阶段三：多轮对话任务形式化**。将工作流生成定义为多轮交互过程：每轮输入为当前任务指令与历史交互上下文（含前几轮生成的工作流及用户反馈），输出为可部署的结构化YAML工作流。第1轮为初始生成，第2-3轮模拟需求变更（如「增加一个发送邮件的步骤」或「把判断条件改为成绩大于80分」），测试模型在迭代修订中保持结构正确性的能力。

**阶段四：双指标执行验证与诊断**。对模型输出的YAML依次进行格式合法性检查（pass rate：YAML语法、节点类型存在性、必填字段完整性）和真实平台执行验证（resolve rate：实际导入Dify/Coze并运行，检测参数类型错误、环境变量缺失、循环依赖等运行时故障）。两指标分离报告，量化系统性虚高。

```
自然语言指令 + 历史上下文 → [LLM生成] → 候选YAML工作流
                                    ↓
                              [格式合法性检查] → pass rate
                                    ↓
                              [Dify/Coze平台导入]
                                    ↓
                              [真实执行验证] → resolve rate
                                    ↓
                              [差距分析 + 错误模式归类]
```

## 核心模块与公式推导

### 模块 1: 双指标评估体系（对应框架图 阶段四）

**直觉**：单一成功率指标会掩盖「看起来对但实际跑不通」的系统性问题，必须将「格式正确」与「执行成功」解耦，才能诚实刻画模型的工业可用性。

**Baseline 公式**（传统端到端评测）：
$$\text{Success Rate} = \frac{\sum_{i=1}^{N} \mathbb{1}[\text{task}_i \text{ completed}]}{N}$$
符号：$N$ = 测试实例总数，$\mathbb{1}[\cdot]$ = 指示函数（任务完成则为1）。该指标不区分失败原因，无法检测「格式合法但执行失败」的中间状态。

**变化点**：传统指标将格式检查与执行验证混为一谈，导致平台运行时错误被忽略；本文引入分层验证，强制暴露执行语义层面的能力缺陷。

**本文公式（推导）**：
$$\text{Step 1}: \quad \text{Pass}_i = \mathbb{1}[\text{YAML}_i \in \mathcal{Y}_{\text{valid}}] \quad \text{加入了结构模式校验以检测语法/模式错误}$$
$$\text{Step 2}: \quad \text{Resolve}_i = \mathbb{1}[\text{PlatformExec}(\text{YAML}_i) = \text{success}] \quad \text{重归一化到真实平台执行语义}$$
$$\text{Step 3}: \quad \text{Gap}_i = \text{Pass}_i - \text{Resolve}_i \quad \text{量化系统性虚高，要求} \text{Resolve}_i \leq \text{Pass}_i$$
$$\text{最终}: \quad \text{Pass Rate} = \frac{\sum_i \text{Pass}_i}{N}, \quad \text{Resolve Rate} = \frac{\sum_i \text{Resolve}_i}{N}, \quad \overline{\text{Gap}} = \text{Pass Rate} - \text{Resolve Rate}$$

**对应消融**：Table 1显示所有15个模型的$\overline{\text{Gap}} > 0$，GLM-4.6平均Gap达20.96%，Education场景Gap达43.44%，证明移除执行验证将导致严重高估。

### 模块 2: 多轮对话迭代评估（对应框架图 阶段三）

**直觉**：真实工作流开发是迭代修订过程，模型需在需求变更时保持已有结构的正确性，而非每次都从零生成；单轮生成评测无法捕捉「修改一点、崩溃全局」的脆弱性。

**Baseline 公式**（单轮静态生成）：
$$\text{Score}_{\text{static}} = f(\text{prompt}, \theta) \rightarrow \text{YAML}_{\text{output}}$$
符号：$f$ = 模型生成函数，$\theta$ = 模型参数，prompt = 固定任务描述。无历史上下文依赖，无修订能力评估。

**变化点**：真实场景中用户需求随对话演进（如「再加一个步骤」），模型需解析增量指令、定位已有结构中的修改点、保持未变更部分的完整性；Baseline假设静态输入，无法评估增量编辑能力。

**本文公式（推导）**：
$$\text{Step 1}: \quad \text{Context}_t = \{(\text{prompt}_1, \text{YAML}_1), ..., (\text{prompt}_{t-1}, \text{YAML}_{t-1})\} \quad \text{加入了对话历史作为条件上下文}$$
$$\text{Step 2}: \quad \text{YAML}_t = f(\text{prompt}_t, \text{Context}_t, \theta) \quad \text{要求模型执行增量推理而非独立生成}$$
$$\text{Step 3}: \quad \text{Consistency}_t = \text{Sim}(\text{YAML}_t[\text{unchanged}], \text{YAML}_{t-1}[\text{unchanged}]) \quad \text{重归一化以保证未变更部分的结构稳定性}$$
$$\text{最终}: \quad \text{Resolve Rate}^{(t)} = \frac{\sum_i \mathbb{1}[\text{PlatformExec}(\text{YAML}_{i,t}) = \text{success}]}{N}, \quad t \in \{1,2,3\}$$

**对应消融**：Figure 4显示所有15个模型在第2-3轮的Pass Rate和Resolve Rate均出现性能衰减，证明多轮迭代是真实能力瓶颈。

### 模块 3: Agentic错误缓解框架（对应框架图 阶段四扩展）

**直觉**：模型在执行验证中反复出现特定错误模式（如循环执行错误），可通过规则后处理或轻量Agent干预进行针对性缓解，而非重新训练模型。

**Baseline 公式**（直接模型输出）：
$$\text{YAML}_{\text{final}} = f(\text{prompt}, \theta), \quad \text{Resolve} = \text{PlatformExec}(\text{YAML}_{\text{final}})$$

**变化点**：观察到循环执行错误（如条件判断节点配置错误导致无限循环）是高频失败模式，引入轻量Agent检测并修复此类结构缺陷。

**本文公式（推导）**：
$$\text{Step 1}: \quad \text{YAML}_{\text{draft}} = f(\text{prompt}, \theta) \quad \text{标准模型生成}$$
$$\text{Step 2}: \quad \text{ErrorPattern} = \text{DetectLoop}(\text{YAML}_{\text{draft}}) \cup \text{DetectTypeMismatch}(\text{YAML}_{\text{draft}}) \quad \text{加入了结构化错误模式检测}$$
$$\text{Step 3}: \quad \text{YAML}_{\text{fixed}} = \text{AgentRepair}(\text{YAML}_{\text{draft}}, \text{ErrorPattern}) \quad \text{重归一化以保证修复后仍满足平台约束}$$
$$\text{最终}: \quad \text{Resolve Rate}_{\text{agentic}} = \frac{\sum_i \mathbb{1}[\text{PlatformExec}(\text{YAML}_{\text{fixed},i}) = \text{success}]}{N}$$

**对应消融**：在GPT-5.1和GPT-5.2上验证，Agentic框架带来最多5.34%的resolve rate提升，但绝对增益有限且仅两个闭源模型验证，泛化性待补充。

## 实验与分析


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/1eb779af-8b90-40da-b570-e06f59ee6966/figures/Figure_2.png)
*Figure 2: Figure 2: Distribution of task types in Chat2Workflow.The benchmark covers six domains: AIGC, Research,Document, Education, Enterprise, and Developer.*



**主实验结果**（Table 1核心数据，三次独立运行平均）：

| Method | Pass Rate | Resolve Rate | Gap (Pass-Resolve) |
|:---|:---|:---|:---|
| Gemini-3-Pro-Preview |  | **71.59%** |  |
| GPT-5.2 |  |  |  |
| GPT-5.1 |  |  |  |
| Claude-4-Opus |  |  |  |
| GLM-4.6 |  |  | **20.96%** (平均) / **43.44%** (Education) |
| Qwen-3-235B |  | **27.71%** |  |
| Qwen-3-Coder-480B |  | **26.44%** |  |
| Qwen-3-32B |  |  |  |
| Qwen-3-8B |  |  |  |

**核心发现分析**：

1. **系统性虚高是普遍现象**：所有15个模型的Resolve Rate均低于Pass Rate，验证核心洞察成立。最优模型Gemini-3-Pro-Preview的71.59% resolve rate意味着近30%的工作流在真实平台无法运行，距离工业部署的可靠性要求（通常>95%）差距显著。

2. **规模定律的复杂化**：Qwen-3系列内部8B→32B→235B呈现正向规模效应，但Qwen-3-Coder-480B（26.44%）反低于Qwen-3-235B（27.71%），表明代码专用模型的优化目标（代码生成）与工作流生成所需的「结构化编排+平台语义理解」存在错位，简单扩大代码模型规模不直接迁移。

3. **领域差异性显著**：Education场景Gap最大（GLM-4.6达43.44%），可能因教育类工作流涉及更多条件分支与学生数据个性化处理，控制流复杂度高于AIGC等线性流程为主的场景。


![Figure 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/1eb779af-8b90-40da-b570-e06f59ee6966/figures/Figure_4.png)
*Figure 4: Figure 4: Performance degradation across dialogue rounds. We show the Pass Rate and Resolve Rate for all 15models across the first three dialogue rounds. Most models exhibit a steady decline in both m*



**多轮衰减分析**（Figure 4）：所有模型在第2-3轮均出现Pass Rate和Resolve Rate下降，表明增量修订能力弱于初始生成。这一发现对实际部署至关重要——用户需求变更时，现有模型难以可靠地「编辑」而非「重写」工作流。

**Agentic框架效果**：在GPT-5.1/5.2上最多提升5.34% resolve rate，但：①增益绝对值有限；②仅验证两个闭源模型，开源模型及更小规模模型是否受益未知；③未报告pass rate变化，不确定是否以牺牲格式合法性为代价。

**公平性检查与局限**：
- Baselines选择较全面（4闭源+11开源），但缺少部分最新模型（如o3、R1等推理模型）
- 数据集规模273实例，Developer场景仅11.1%样本，统计稳定性受限
- 节点类型限定20种，真实平台节点更丰富，当前评估可能低估真实复杂度
- 未报告计算成本（API调用费用、执行验证耗时），工业部署可行性待评估
- 
![Figure 5](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/1eb779af-8b90-40da-b570-e06f59ee6966/figures/Figure_5.png)
*Figure 5: Figure 5: Bad case analysis for the StudyPlanner task. We compare outputs from three representative models:Kimi-K2-Instruct produces an invalid edge connection in the iteration node; GPT-5.2 generates*

 Figure 5展示StudyPlanner任务的失败案例，GPT-5.2在Dify/Coze上的具体输出对比见Figure 8-9

## 方法谱系与知识库定位

**方法家族**：LLM-based Agent评测基准 → 结构化输出/工具使用评测 → 工作流自动化评测

**Parent Method**：WebArena（Zhou et al., 2023）作为端到端网页Agent评测的代表，本文继承其「真实环境验证」思想，但将验证对象从「最终任务完成」下移至「中间工作流结构的可执行性」；SWE-bench（Jimenez et al., 2023）提供「实例化测试验证」范式，本文将其迁移至可视化工作流领域。

**直接Baselines与差异**：
- **AgentBench / ToolBench**：评测工具调用能力，但输出为自由形式API调用序列，无结构化拓扑约束；本文要求严格的节点连接图与平台特定YAML格式
- **Visual Programming / Node-RED相关研究**：关注工作流可视化本身，未系统评估LLM生成能力；本文首次将LLM作为生成主体并量化其能力边界
- **Dify/Coze官方文档/模板**：提供人工编写的工作流范例；本文构建从自然语言到工作流的自动生成评测，并引入多轮迭代场景

**改动插槽**：
- **evaluation_metric**：从单一成功率 → pass/resolve双指标分离，引入真实平台执行验证
- **task_formulation**：从单轮静态生成 → 多轮对话迭代修订
- **data_curation**：从通用任务 → 6大领域真实业务工作流，人工验证+平台直接部署

**后续方向**：
1. **扩展节点类型与平台覆盖**：当前20种节点/2个平台限制，需验证在更复杂工业场景（如ServiceNow、Microsoft Power Automate）的泛化性
2. **引入推理模型与强化学习**：o3/R1等推理模型、以及直接优化resolve rate的RL训练，可能缩小pass-resolve差距
3. **神经符号混合生成**：结合LLM的语义理解与符号规划器的结构正确性保证，从根本上提升resolve rate至工业可用水平

**知识库标签**：
- **modality**：text-to-structured-workflow（文本到结构化工作流）
- **paradigm**：benchmark/evaluation（基准评测）
- **scenario**：industrial workflow automation（工业工作流自动化）
- **mechanism**：platform-executable validation（平台可执行验证）
- **constraint**：multi-turn consistency, deployment-ready correctness（多轮一致性、部署就绪正确性）

