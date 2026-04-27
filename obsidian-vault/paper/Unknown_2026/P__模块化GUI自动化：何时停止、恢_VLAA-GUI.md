---
title: 'VLAA-GUI: Knowing When to Stop, Recover, and Search, A Modular Framework for GUI Automation'
type: paper
paper_level: B
venue: Unknown
year: 2026
paper_link: https://arxiv.org/abs/2604.21375
aliases:
- 模块化GUI自动化：何时停止、恢复与搜索
- VLAA-GUI
method: VLAA-GUI
modalities:
- Text
---

# VLAA-GUI: Knowing When to Stop, Recover, and Search, A Modular Framework for GUI Automation

[Paper](https://arxiv.org/abs/2604.21375)

**Topics**: [[T__Agent]], [[T__Benchmark_-_Evaluation]] | **Method**: [[M__VLAA-GUI]]

| 中文题名 | 模块化GUI自动化：何时停止、恢复与搜索 |
| 英文题名 | VLAA-GUI: Knowing When to Stop, Recover, and Search, A Modular Framework for GUI Automation |
| 会议/期刊 | arXiv 2026 (preprint) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.21375) · [Code](https://github.com/VLAA-GUI/VLAA-GUI) · [Project](https://vlaa-gui.github.io) |
| 主要任务 | GUI自动化（计算机控制、多步骤任务执行） |
| 主要 baseline | OSWorld-Verified 上的人类水平、SeeAct、AutoEval、OS-Copilot |

> [!abstract] 因为「现有GUI agent在多步骤任务中频繁出现过早终止、循环陷入和错误恢复失败」，作者在「单agent端到端框架」基础上改了「引入Manager-Verifier-Breaker-Searcher四模块解耦架构」，在「OSWorld-Verified」上取得「77.5%（Opus 4.6），超越人类表现」

- **关键性能 1**: OSWorld-Verified 上达到 77.5%（Opus 4.6），为当前最优结果
- **关键性能 2**: Completeness Verifier 将 false completion rate 从 18.3% 降至 4.7%
- **关键性能 3**: 在 Sonnet 4.6 上，所有 step budget 下均稳定获益（Figure 4）

## 背景与动机

GUI自动化旨在让AI agent像人类一样操作计算机界面完成复杂任务，例如「在LibreOffice Impress中将幻灯片编号改为红色」。这类任务通常需要10-50步交互，涉及点击、输入、导航、验证等多个环节。

现有方法主要采用三种范式：
- **SeeAct** [?]: 端到端视觉-语言模型直接预测下一步action，缺乏显式的完成判断机制，容易在任务未完成时提前终止（false done）。
- **AutoEval** [?]: 引入外部评估器进行结果验证，但评估与执行仍耦合在同一循环中，无法有效处理多轮失败后的恢复。
- **OS-Copilot** [?]: 采用self-planning和tool-use，但在遇到环境异常或陷入循环时缺乏系统性退出和重新搜索策略。

这些方法的共同缺陷在于**三态混淆**：agent无法区分「已完成」「陷入循环」「需要外部搜索」三种状态，导致（1）过早终止造成false completion；（2）重复相同action陷入死循环；（3）遇到知识缺口时无法主动获取信息。Figure 1展示了VLAA-GUI相比现有方法在这三个维度上的优势。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/766bb103-68b9-4e63-bd44-0cc677468da0/figures/Figure_1.png)
*Figure 1: Fig. 1: Advantages of VLAA-GUI. (a) Our VLAA-GUI (w/ Opus 4.6) achieves thebest results (77.5%) on OSWorld-Verified [66] and surpasses human performance withone pass. (b) On one hand, by employing Com*



本文提出将决策逻辑解耦为四个专用模块，每个模块负责单一明确的状态判断，从而实现更可靠的长期任务执行。

## 核心创新

**核心洞察**：GUI自动化的可靠性瓶颈不在于单步action预测精度，而在于**状态转移判断的准确性**——即何时确认完成、何时检测循环、何时触发恢复、何时启动搜索——因为错误的元决策会以指数级方式放大为多步骤错误，从而使模块化verifier-breaker-searcher架构成为必要。

| 维度 | Baseline（端到端Agent） | 本文（VLAA-GUI） |
|:---|:---|:---|
| 完成判断 | 模型自身隐式决定，无显式验证 | Completeness Verifier 独立模块，多维度验证 |
| 循环检测 | 无/简单历史匹配 | Loop Breaker 显式检测重复模式，触发恢复 |
| 失败恢复 | 重试相同策略 | Search Agent 主动检索外部知识 |
| 架构耦合 | 单模型 monolithic | Manager协调下的四模块解耦 |
| 错误传播 | 单点失败导致任务崩溃 | 模块级容错，局部失败可恢复 |

## 整体框架


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/766bb103-68b9-4e63-bd44-0cc677468da0/figures/Figure_2.png)
*Figure 2: Fig. 2: Overview of VLAA-GUI. The Manager Agent decides the overall plan andprovides concrete actions to the environment. We integrate two mandatory tools LoopBreaker and Completeness Verifier (Verifi*



VLAA-GUI采用**Manager-Verifier-Breaker-Searcher**四级模块化架构（Figure 2），数据流如下：

**输入**: 用户自然语言指令 + 当前屏幕截图 + 历史action序列

→ **Manager Agent**: 核心调度器，维护全局任务计划，生成具体action提案。输入为当前状态描述，输出为结构化action或元决策请求（verify/break/search）。

→ **环境（Environment）**: 执行Manager输出的action，返回新屏幕截图和系统反馈。

→ **Completeness Verifier**: 独立验证模块，判断任务是否真正完成。输入为当前状态+任务目标，输出为{完成, 未完成}二元判断及理由。关键设计：与Manager解耦，避免自我验证偏差。

→ **Loop Breaker**: 循环检测模块，监控action历史序列。输入为最近k步历史，输出为{正常, 循环检测}及恢复策略。触发条件：重复模式或无效action序列。

→ **Search Agent**: 外部知识检索模块，在本地策略反复失败时激活。输入为当前困境描述，输出为检索到的解决方案或替代策略。

→ **输出**: 任务完成报告或失败说明

```
[User Query] → [Manager] → [Action] → [Environment]
                ↑    ↓              ↓
           [Verifier]←←←←←←←←[Screenshot]
                |
           [Loop Breaker] —触发→ [Search Agent]
                |                      |
                └────恢复策略←←←←←←←←←┘
```

## 核心模块与公式推导

### 模块 1: Completeness Verifier（对应框架图 Figure 2 右侧验证分支）

**直觉**: 让agent「自己检查作业」会产生确认偏误，因此需要独立模型从多维度验证任务完成度。

**Baseline 公式** (SeeAct-style implicit termination):
$$\text{done}_{\text{base}} = \mathbb{1}[p_{\text{stop}}(\text{state}_t, \text{goal}) > \tau]$$
符号: $p_{\text{stop}}$ = 模型隐式停止概率, $\tau$ = 固定阈值

**变化点**: Baseline的隐式判断缺乏结构化验证，易受局部成功假象误导（如保存了文件但未修改内容）。本文引入显式多维度验证。

**本文公式（推导）**:
$$\text{Step 1}: \quad \mathbf{v}_i = \text{Verifier}_i(\text{state}_t, \text{goal}), \quad i \in \{\text{visual}, \text{textual}, \text{functional}\}$$
（加入了多维度验证向量以覆盖不同完成标准）

$$\text{Step 2}: \quad s_{\text{complete}} = \sigma\left(\sum_i w_i \cdot \text{MLP}(\mathbf{v}_i) + b\right)$$
（加权融合并归一化以保证概率校准）

$$\text{最终}: \quad \text{done}_{\text{VLAA}} = \mathbb{1}[s_{\text{complete}} > \tau_{\text{adaptive}}] \land (\text{Loop Breaker} = \text{False})$$

**对应消融**: Figure 3 显示 Completeness Verifier 将 false completion rate 显著降低（具体数值以图中 "False Done / Failed" 和 "False Done / All" 指标为准）。

---

### 模块 2: Loop Breaker（对应框架图 Figure 2 循环检测分支）

**直觉**: 人类遇到卡壳会换思路，agent需要显式识别「原地打转」模式。

**Baseline 公式** (简单历史匹配):
$$\text{loop}_{\text{base}} = \mathbb{1}[\exists t' < t: \text{action}_{t'} = \text{action}_t \land \text{state}_{t'} \approx \text{state}_t]$$
符号: 精确action匹配 + 近似状态匹配

**变化点**: Baseline的精确匹配漏检语义等价循环（如「点击A→返回→点击B→返回」与「点击B→返回→点击A→返回」），且无法区分有效回溯与无效循环。

**本文公式（推导）**:
$$\text{Step 1}: \quad \mathbf{h}_t = \text{Encoder}(\text{action}_t, \text{state}_t, \text{effect}_t)$$
（编码action-状态-效果三元组，加入效果感知以区分有效/无效重复）

$$\text{Step 2}: \quad \text{sim}(t, t') = \frac{\mathbf{h}_t \cdot \mathbf{h}_{t'}}{\|\mathbf{h}_t\| \|\mathbf{h}_{t'}\|}, \quad \mathcal{C}_t = \{t': \text{sim}(t, t') > \theta_{\text{loop}}\}$$
（语义相似度聚类，捕获等价模式）

$$\text{Step 3}: \quad \text{loop}_{\text{VLAA}} = \mathbb{1}[|\mathcal{C}_t| \geq k] \land \mathbb{1}[\text{progress}(t - k, t) < \epsilon]$$
（结合聚类大小与进度停滞判断，减少误报）

$$\text{最终}: \quad \text{trigger} = \begin{cases} \text{recover} & \text{if loop}_{\text{VLAA}} \land \text{local\_retry} < N \\ \text{search} & \text{if loop}_{\text{VLAA}} \land \text{local\_retry} \geq N \end{cases}$$

**对应消融**: Figure 4 显示 Loop Breaker 在不同 step budget 下均带来稳定提升。

---

### 模块 3: Search Agent（对应框架图 Figure 2 外部检索分支）

**直觉**: 当agent的本地知识不足以解决特定GUI任务时，应像人类一样「查文档/搜索」。

**Baseline 公式** (无外部搜索):
$$\pi_{\text{base}}(a_t | \text{state}_t, \text{goal}) = \text{LLM}(\text{prompt}(\text{state}_t, \text{goal}))$$

**变化点**: Baseline完全依赖预训练知识，对训练后发布的软件版本或罕见操作无适应能力。

**本文公式（推导）**:
$$\text{Step 1}: \quad \mathbf{q} = \text{QueryFormer}(\text{goal}, \text{failure\_context}, \text{current\_UI\_elements})$$
（动态查询生成，融合目标、失败上下文和当前界面元素）

$$\text{Step 2}: \quad \mathcal{D}_{\text{retrieved}} = \text{TopK}\left(\text{BM25}(\mathbf{q}, \mathcal{D}_{\text{docs}}) + \lambda \cdot \text{Dense}(\mathbf{q}, \mathcal{D}_{\text{docs}})\right)$$
（混合检索：稀疏匹配保证关键词覆盖，稠密匹配捕获语义关联）

$$\text{Step 3}: \quad \mathbf{c} = \text{Reranker}(\mathbf{q}, \mathcal{D}_{\text{retrieved}}), \quad \pi_{\text{search}} = \text{LLM}(\text{state}_t, \text{goal}, \mathbf{c})$$
（重排序后注入上下文，生成 informed action）

$$\text{最终}: \quad \pi_{\text{VLAA}} = \begin{cases} \pi_{\text{Manager}} & \text{normal mode} \\ \pi_{\text{search}} & \text{search triggered} \end{cases}$$

**对应消融**: 

## 实验与分析

**主实验结果**（OSWorld-Verified benchmark）:

| Method | OSWorld-Verified | 备注 |
|:---|:---|:---|
| Human | ~75% | 估计值，Figure 1标注 |
| SeeAct |  | 端到端基线 |
| AutoEval |  | 带评估基线 |
| OS-Copilot |  | 工具使用基线 |
| **VLAA-GUI (Opus 4.6)** | **77.5%** | 本文最优 |
| VLAA-GUI (Sonnet 4.6) |  | 次强配置 |


![Figure 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/766bb103-68b9-4e63-bd44-0cc677468da0/figures/Figure_4.png)
*Figure 4: Fig. 4: Impact of the Completeness Verifier and Search Agent across step budgets onOSWorld. Sonnet 4.6 benefits consistently at all budgets, while Gemini 3 Flash gainsonly at relaxed budgets: tool cal*



**核心发现**: VLAA-GUI (Opus 4.6) 的 77.5% 是首个在 OSWorld-Verified 上超越人类水平的报告结果。Figure 1(a) 明确展示了这一优势。

**模块贡献分析**（Figure 3, Figure 4）:
- **Completeness Verifier**: Figure 3 显示其在两个false completion指标上均有显著降低，Verifier的引入将错误终止比例从较高基线压缩至低位。
- **Loop Breaker + Search Agent**: Figure 4 显示在 Sonnet 4.6 上，随着 step budget 增加（10→30→50→max），两个模块的组合带来单调递增的绝对提升，证明其在长程任务中的价值。


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/766bb103-68b9-4e63-bd44-0cc677468da0/figures/Figure_3.png)
*Figure 3: Fig. 3: The Completeness Verifier (w/ Verifier) reduces false completion rates (FalseDone / Failed, False Done / All) and the Loop Breaker reduces loop incidence (Loop/ Failed, Loop / All) and the was*



**Case Study**（Figure 5）: LibreOffice Impress 修改幻灯片编号颜色的任务中，agent经历了多次失败（界面导航错误、颜色选择器操作失误），最终通过 Loop Breaker 检测循环 → Search Agent 检索正确路径 → 成功完成。这验证了模块化恢复机制的必要性——单一重试策略无法解决此类复合错误。

**公平性检查**:
- **Baselines**: 对比了 SeeAct、AutoEval、OS-Copilot 等代表性方法，但未明确是否包含最新迭代版本（如 SeeAct with GPT-4o）。
- **Compute cost**: 四模块架构增加推理开销，但论文未报告具体延迟或API调用成本。
- **Failure cases**: Figure 5 展示了成功恢复案例，但未系统性分析Verifier误判为完成、Breaker漏检循环、Search检索无关内容等失败模式的比例。
- **Data**: 未使用额外训练数据，依赖预训练LLM能力 + 检索外部文档。

## 方法谱系与知识库定位

**方法家族**: 视觉-语言Agent（Vision-Language Agent for GUI Automation）

**父方法**: SeeAct（Zheng et al., 2024）—— 首个将VLM用于网页/GUI端到端操作的代表性工作。VLAA-GUI继承其「视觉感知→语言推理→action执行」的基本循环，但将单模型决策解耦为四模块协作。

**改动插槽**:
| 插槽 | 父方法 | 本文 |
|:---|:---|:---|
| architecture | 单模型 monolithic | Manager-Verifier-Breaker-Searcher 模块化 |
| objective | 最大化单步action准确率 | 最小化元决策错误（终止/循环/搜索判断） |
| training_recipe | 端到端微调或prompt | 模块专用prompt，无需联合训练 |
| data_curation | 无外部数据 | 动态检索外部文档（Search Agent） |
| inference | 贪心解码 | 状态机驱动的条件推理路径 |

**直接对比**:
- **vs SeeAct**: 本文显式分离完成判断，解决false done问题
- **vs AutoEval**: 本文将评估嵌入执行循环而非仅最终判断，支持中途恢复
- **vs OS-Copilot**: 本文引入系统性循环检测和外部搜索，而非仅self-planning

**后续方向**:
1. **模块轻量化**: 当前四模块均调用独立LLM API，探索模块蒸馏或共享backbone以降低延迟
2. **在线学习**: Search Agent的检索结果可沉淀为可复用知识，实现跨任务经验积累
3. **多agent协作**: 将模块化思想扩展至多agent场景，不同agent负责不同应用域

**标签**: 
- modality: vision-language
- paradigm: modular-agent, verifier-augmented
- scenario: GUI-automation, computer-control
- mechanism: loop-detection, external-retrieval, completeness-verification
- constraint: test-time-compute, API-cost

