---
title: 'SWE-bench: Can Language Models Resolve Real-world Github Issues?'
type: paper
paper_level: A
venue: ICLR
year: 2024
paper_link: null
aliases:
- 真实GitHub issue评测LLM代码修复能力
- SWE-bench
- SWE-bench is a challenging
acceptance: Oral
cited_by: 1980
code_url: https://swe-bench.github.io/
method: SWE-bench
modalities:
- Text
- code
---

# SWE-bench: Can Language Models Resolve Real-world Github Issues?

[Code](https://swe-bench.github.io/)

**Topics**: [[T__Benchmark_-_Evaluation]], [[T__Code_Generation]] | **Method**: [[M__SWE-bench]] | **Datasets**: SWE-bench

> [!tip] 核心洞察
> SWE-bench is a challenging, sustainable evaluation framework for language models on real-world software engineering tasks drawn from actual GitHub issues and pull requests.

| 中文题名 | 真实GitHub issue评测LLM代码修复能力 |
| 英文题名 | SWE-bench: Can Language Models Resolve Real-world Github Issues? |
| 会议/期刊 | ICLR 2024 (Oral) |
| 链接 | [arXiv](https://arxiv.org/abs/2310.06770) · [Code](https://swe-bench.github.io/) · [Project](https://swe-bench.github.io/) |
| 主要任务 | Benchmark / Evaluation, Code Generation, Program Repair |
| 主要 baseline | HumanEval, APPS, Multipl-E, ClassEval, Defects4J |

> [!abstract] 因为「现有编程评测如 HumanEval 已饱和且无法捕捉真实软件工程复杂度」，作者在「HumanEval 等自包含编程题」基础上改了「从真实 GitHub issue-PR 对构建任务、自动补丁修复与双重测试验证」，在「SWE-bench (2,294 instances)」上取得「Claude 2 最高仅 1.96% 解决率，暴露 LLM 在真实代码修复上的巨大差距」

- Claude 2 在 BM25 检索设置下解决率 1.96%，GPT-4 (25% 子集) 为 1.31%，所有模型表现极差
- Oracle 检索将 Claude 2 提升至 4.8%，绝对提升 2.84 个百分点，证明上下文检索是关键瓶颈
- SWE-bench 包含 2,294 个任务实例，横跨 12 个 Python 开源仓库

## 背景与动机

现有编程评测基准如 HumanEval 和 APPS 已被大语言模型迅速饱和：GPT-4 在 HumanEval 上 pass@1 超过 90%，但这些评测本质上是自包含的编程谜题——代码量少、上下文孤立、只需生成单个函数即可通过简单单元测试。然而，真实软件工程远比这复杂：当开发者面对一个 GitHub issue 时，需要理解跨文件依赖、在数千行代码库中定位问题、修改多个文件，并确保修复不破坏现有功能。

现有方法如何处理这一问题？HumanEval [8] 提供 164 个手写编程问题，模型在隔离环境中生成代码片段，通过少数单元测试即算成功；APPS [21] 扩展到竞赛级编程题，但仍为自包含问题；Multipl-E [5] 将 HumanEval 翻译至多语言，未改变任务本质；ClassEval [12] 手动构造类级代码生成任务，规模有限且非真实场景。这些基准的共同局限在于：它们回避了真实软件开发的核心挑战——在大型、演进中的代码库上进行多文件编辑。

它们为何不足？具体而言，三大缺失制约了评测的实用性：(1) **数据真实性缺失**：合成题目与真实 issue 的分布差距巨大，模型在 HumanEval 上的高分无法迁移到实际开发；(2) **评估完整性缺失**：单一通过/失败判定无法检测"修复引入新 bug"的回归问题；(3) **上下文规模缺失**：真实修复需理解跨文件依赖，而现有基准仅提供数十行上下文。这些局限导致社区严重高估了 LLM 的实际代码能力。

本文构建 SWE-bench，首次将评测锚定于真实 GitHub issue 与合并 PR 的闭环，通过自动化环境搭建与双重测试验证，建立可持续的真实软件工程能力评测框架。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/71bfaa91-aa2f-4cf2-958b-c9340e838099/figures/Figure_1.png)
*Figure 1: Figure 1: SWE-bench sources task instances from real-world Python repositories by connectingGitHub issues to merged pull request solutions that resolve related tests. Provided with the issuetext and a*



## 核心创新

核心洞察：真实软件工程任务的评测瓶颈不在于模型生成代码的语法正确性，而在于**评测基础设施能否可靠地验证多文件补丁在真实代码库中的功能正确性**，因为 GitHub issue 的解决天然涉及跨文件修改、环境依赖和回归测试，只有构建从 issue 描述到补丁应用、测试执行、自动修复的完整自动化流水线，才能使大规模真实任务评测成为可能。

| 维度 | Baseline (HumanEval/APPS) | 本文 (SWE-bench) |
|:---|:---|:---|
| 数据来源 | 手写或合成的自包含编程题 | 真实 GitHub issue 与合并 PR 的配对，2,294 实例 |
| 任务范围 | 单函数/单文件代码生成 | 多文件代码库编辑，需跨文件推理 |
| 上下文长度 | <100 行隔离代码 | 完整仓库文件，BM25 或 Oracle 检索提供相关上下文 |
| 评估方式 | 简单单元测试通过即成功 | FAIL_TO_PASS + PASS_TO_PASS 双重测试验证 |
| 补丁处理 | 直接执行生成代码 | 自动补丁修复（删除冗余上下文、重新计算 diff 头部） |
| 环境要求 | 无/最小化 | 完整可复现环境：base commit 检出、依赖安装、测试补丁应用 |

## 整体框架


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/71bfaa91-aa2f-4cf2-958b-c9340e838099/figures/Figure_2.png)
*Figure 2: Figure 2: SWE-bench task instances are created from merged pull requests that resolve an issue,contributes tests, and install successfully.*



SWE-bench 的整体框架是一条从 GitHub 数据挖掘到自动评估的完整流水线，包含五个核心模块：

**模块 1: GitHub Issue-PR 配对挖掘** — 输入为 12 个 Python 开源仓库的 issue 与合并 PR 历史，输出为原始候选任务实例（含问题描述、base commit、解决方案补丁）。该模块通过匹配 "Closes #issue" 等关键词建立 issue 到修复 PR 的关联。

**模块 2: 任务实例验证 (Task Instance Validation)** — 输入为原始候选实例，输出为通过严格筛选的验证实例。验证确保：(a) base commit 可检出，(b) 环境可激活且依赖可安装，(c) 测试补丁可应用，(d) 步骤 1-4 均不失败。未通过验证的实例被丢弃，保证评测可复现。

**模块 3: 上下文检索 (BM25/Oracle)** — 输入为任务实例的问题描述与代码库，输出为供模型使用的相关文件上下文。BM25 设置使用稀疏检索从仓库中召回相关文件；Oracle 设置则直接提供参考解决方案修改过的文件，用于分析上下文上限。

**模块 4: 补丁生成 (Patch Generation)** — 输入为问题描述 + 检索到的上下文文件，输出为模型生成的补丁 δ̂。模型需理解 issue 描述、定位相关代码、生成符合 unified diff 格式的补丁。

**模块 5: 自动化评估与补丁修复** — 输入为模型补丁 δ̂、任务实例（含 base commit、测试补丁 T），输出为测试执行结果与通过/失败判定。该模块执行 7 步流水线：检出 base commit → 激活环境 → 安装依赖 → 应用测试补丁 T → 应用预测补丁 δ̂ → **自动补丁修复**（若失败则删除冗余上下文行并重新计算头部）→ 执行测试 → 解析日志并与 ground truth 比对。

```
GitHub 仓库历史 → [Issue-PR 配对挖掘] → 原始候选实例
                                              ↓
                                    [任务实例验证] → 验证通过实例 (2,294个)
                                              ↓
问题描述 + 代码库 → [BM25/Oracle 检索] → 相关文件上下文
                                              ↓
                                    [模型补丁生成] → 预测补丁 δ̂
                                              ↓
              δ̂ + base commit + 测试补丁 T → [自动评估流水线]
                                              ↓
                                    FAIL_TO_PASS ∪ PASS_TO_PASS 测试执行
                                              ↓
                                    测试-状态映射解析 → 与 ground truth 比对 → 解决/未解决
```

## 核心模块与公式推导

### 模块 1: 双重测试验证（对应框架图 评估流水线最终判定环节）

**直觉**: 真实软件修复必须同时"治好病"和"不伤人"——仅验证原失败测试通过是不够的，还需确保原有功能未被破坏。

**Baseline 公式** (HumanEval pass@k):
$$\text{Pass@k} = \frac{1}{|P|} \sum_{i=1}^{|P|} \mathbb{1}[\text{test}(p_i) = \text{pass}]$$
符号: $P$ = 生成的程序集合, $\mathbb{1}[\cdot]$ = 指示函数, test = 单一单元测试判定。

**变化点**: HumanEval 的单一测试无法检测回归错误；SWE-bench 引入两类测试的并集验证，且要求全部通过。

**本文公式（推导）**:
$$\text{Step 1}: \mathcal{T}_{\text{fail}} = \text{FAIL\_TO\_PASS}, \quad \mathcal{T}_{\text{pass}} = \text{PASS\_TO\_PASS} \quad \text{（定义两类测试集合）}$$
$$\text{Step 2}: \mathcal{T}_{\text{total}} = \mathcal{T}_{\text{fail}} \cup \mathcal{T}_{\text{pass}} \quad \text{（合并为总测试集合）}$$
$$\text{Step 3}: \text{status}(t) \text{xleftarrow}{\text{parser}} \text{log}_{\hat{\delta}}, \quad \forall t \in \mathcal{T}_{\text{total}} \quad \text{（从执行日志解析每个测试状态）}$$
$$\text{最终}: \text{Task Solved} \iff \forall t \in \mathcal{T}_{\text{total}} : \text{status}(t) = \text{pass}$$

**对应消融**: Table 7 显示模型在 2023 年前后的任务实例上表现无显著差异，说明时间分布未造成评测偏差；Table 6 的 Oracle-collapsed 设置显示压缩无关上下文后各模型表现变化，验证上下文质量的关键性。

---

### 模块 2: 自动补丁修复（对应框架图 Step 6）

**直觉**: 大语言模型生成的补丁常包含格式瑕疵（多余上下文行、错误的 diff 头部行号），导致直接应用失败；需要启发式修复而非直接判负，以公平评估模型的实质修复能力。

**Baseline 公式**: 无——现有基准直接执行生成代码，不涉及补丁应用机制。

**本文公式（推导）**:
$$\text{Step 1}: \text{apply}(\hat{\delta}) \rightarrow \{\text{success}, \text{fail}\} \quad \text{（尝试应用原始补丁）}$$
$$\text{Step 2}: \text{if fail}: \hat{\delta}' = \text{RemoveContext}(\hat{\delta}) \quad \text{（删除不必要的上下文行）}$$
$$\text{Step 3}: \hat{\delta}'' = \text{RecalculateHeaders}(\hat{\delta}') \quad \text{（重新计算 @@ -l,s +l,s 头部）}$$
$$\text{最终}: \text{Patch Repair}(\hat{\delta}) = \begin{cases} \hat{\delta}'' & \text{if apply}(\hat{\delta}) \text{ fails and repair succeeds} \\ \hat{\delta} & \text{if apply}(\hat{\delta}) \text{ succeeds} \\ \text{fail} & \text{otherwise} \end{cases}$$

**对应消融**: 该机制未被显式消融，但文中指出其必要性——无此步骤则大量格式合规但上下文冗余的有效补丁将被错误判负。

---

### 模块 3: 评测日志处理流水线（对应框架图 日志解析与比对环节）

**直觉**: 不同仓库使用不同测试框架（pytest, unittest 等），输出格式各异；需统一解析为结构化表示才能进行标准化判定。

**Baseline 公式**: 无——现有基准使用统一测试接口，无需仓库特定解析。

**本文公式（推导）**:
$$\text{Step 1}: \text{log}_{\hat{\delta}} \in \Sigma^* \quad \text{（原始执行日志为自由文本）}$$
$$\text{Step 2}: \text{parser}_{\text{repo}}: \Sigma^* \rightarrow (\mathcal{T} \rightarrow \{\text{pass}, \text{fail}, \text{error}, \text{skip}\}) \quad \text{（仓库特定解析器）}$$
$$\text{Step 3}: M_{\text{pred}} = \{(t, \text{status}) \text{mid} t \in \mathcal{T}_{\text{total}}\} \quad \text{（预测测试状态映射）}$$
$$\text{Step 4}: M_{\text{gt}} = \{(t, \text{pass}) \text{mid} t \in \mathcal{T}_{\text{total}}\} \quad \text{（ground truth：所有测试应通过）}$$
$$\text{最终}: \text{Match} \iff M_{\text{pred}} = M_{\text{gt}} \text{ on } \mathcal{T}_{\text{total}}$$

**对应消融**: Table 8 显示成功补丁的平均编辑量，间接反映模型生成补丁与解析后实际生效补丁的差异程度。

## 实验与分析


![Table 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/71bfaa91-aa2f-4cf2-958b-c9340e838099/figures/Table_2.png)
*Table 2: Table 2: Model resolve rates with BM25 re-trieval, with different maximum context lengths.*




![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/71bfaa91-aa2f-4cf2-958b-c9340e838099/figures/Table_1.png)
*Table 1: Table 1: Average and maximum numbers char-acterizing different attributes of a SWE-benchtask instance. Statistics are micro-averages cal-culated without grouping by repository.*



本文在 SWE-bench 的 2,294 个真实任务实例上评估了多个 state-of-the-art 模型。核心发现极为严峻：即使在提供检索上下文的 BM25 设置下，所有模型的解决率都低于 2%。Claude 2 以 1.96% 的解决率位居第一，GPT-4 在 25% 随机子集上仅为 1.31%，而 SWE-Llama（本文提出的 13B 微调模型）表现更弱。这一结果彻底颠覆了社区对 LLM 代码能力的乐观估计——在 HumanEval 上接近满分的模型，在真实 issue 面前几乎全军覆没。

Oracle 检索设置的对比揭示了关键瓶颈。当直接提供参考解决方案修改过的文件（Oracle 设置）时，Claude 2 的解决率跃升至 4.8%，绝对提升 2.84 个百分点。这一差距证明：**上下文检索是当前的核心瓶颈**，而非单纯的代码生成能力。Table 3 显示 BM25 在 oracle 文件上的召回率随上下文长度变化，Table 4 进一步表明模型在更长的上下文比例下表现提升，但即使拥有完整 oracle 文件，最佳模型仍无法解决 95% 的问题。


![Table 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/71bfaa91-aa2f-4cf2-958b-c9340e838099/figures/Table_4.png)
*Table 4: Table 4: We compare the different context lengths and proportion of the “oracle” retrieval settingcovered. Models with shorter context lengths are thus inherently disadvantaged. Note that descrip-tion*



上下文长度的消融实验（Table 2）显示，限制最大上下文长度会显著损害模型表现：更短的上下文窗口导致更低的解决率，因为模型无法获取足够的跨文件信息。Table 5（即 Figure 4）的模型间对比表明，所有模型在 BM25 设置下都处于极低水平，且彼此差距不大，说明这是任务本身的难度而非特定模型的缺陷。

时间分布的稳健性检查（Table 7）显示，模型在 2023 年前后的任务实例上表现无显著差异，缓解了"训练数据污染"的担忧。Table 8 的分析则揭示成功补丁的平均编辑量较小，暗示模型倾向于做最小修改，但这并不足以保证正确性。

公平性考量：本文的比较存在若干局限。GPT-4 仅评估了 25% 随机子集（标注为预算限制），可能引入方差；BM25 作为稀疏检索器可能非最优，学习式检索器或能提升表现；评测仅限 Python 仓库，未覆盖 Java/C++ 等语言；SWE-Llama 的微调细节未在摘录中披露。此外，作者未与 CodeT5+、StarCoder、WizardCoder 等代码专用模型，以及工具使用 agent 方法进行直接对比，这些可能是更强的基线。

## 方法谱系与知识库定位

SWE-bench 属于**编程评测基准**方法族，其直接父系为 HumanEval [8] 所代表的自包含代码生成评测范式，但在三个关键 slot 上进行了结构性替换：

- **数据流程 (data_pipeline)**: 从 HumanEval 的手写合成题 → 真实 GitHub issue-PR 配对，规模从 164 题扩展至 2,294 实例，横跨 12 个活跃仓库
- **评估策略 (evaluation_strategy)**: 从简单单元测试通过判定 → 自动补丁应用 + 补丁修复启发式 + FAIL_TO_PASS ∪ PASS_TO_PASS 双重测试验证
- **任务范围 (task_scope)**: 从单函数代码生成 → 多文件代码库编辑，要求长上下文理解与跨文件推理

直接基线与差异：
- **HumanEval [8]**: 本文明确对比其饱和性，SWE-bench 解决率 <2% vs HumanEval >90% 凸显真实复杂度差距
- **APPS [21]**: 竞赛级编程评测，仍属自包含问题；SWE-bench 引入环境交互与回归测试
- **Defects4J [27]**: Java 缺陷数据集，手工收集；SWE-bench 自动化构建且面向 Python 与 LLM 评测
- **Multipl-E [5] / ClassEval [12]**: 多语言扩展与类级生成，未触及真实 issue 场景

后续方向：(1) **检索增强**: 替换 BM25 为学习式代码检索器，缩小与 Oracle 设置的 2.84pp 差距；(2) **Agent 化**: 结合工具使用（文件浏览、测试执行反馈循环）的交互式修复；(3) **跨语言扩展**: 将流水线迁移至 Java/JavaScript/Go 等语言，验证通用性。

标签: #modality:text+code #paradigm:benchmark #scenario:real-world-software-engineering #mechanism:automated-evaluation-with-patch-repair #constraint:dual-test-verification #constraint:python-only

