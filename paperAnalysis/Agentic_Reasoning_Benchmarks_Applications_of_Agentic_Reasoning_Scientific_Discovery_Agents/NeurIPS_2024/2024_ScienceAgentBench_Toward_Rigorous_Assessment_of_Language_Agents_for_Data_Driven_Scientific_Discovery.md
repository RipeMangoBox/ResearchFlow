---
title: "ScienceAgentBench: Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discovery"
venue: ICLR
year: 2025
tags:
  - Survey_Benchmark
  - task/MLLM-evaluation
  - program-execution-evaluation
  - rubric-based-evaluation
  - contamination-mitigation
  - dataset/ScienceAgentBench
  - opensource/partial
core_operator: "从真实论文中抽取科研代码任务，统一为自包含 Python 程序生成问题，并用执行结果、任务成功、细粒度 rubric 与成本做联合评测"
primary_logic: |
  真实论文中的科研任务与数据 + 可选专家知识 → 统一成“说明+数据预览(+知识)→自包含 Python 程序”的接口，并通过数据改造抑制污染与捷径 → 以执行成功、任务成功、代码相似度、图像 judge 与成本分层诊断语言代理的科学编程边界
claims:
  - "在 102 个任务、每题 3 次尝试的设定下，最佳非 o1 代理 Claude-3.5-Sonnet + self-debug 的任务成功率仅为 32.4%（无专家知识）和 34.3%（有专家知识）[evidence: analysis]"
  - "对 Claude-3.5-Sonnet 而言，self-debug 相比 direct prompting 将成功率从 16.7% 提升到 32.4%，说明执行反馈显著优于一次性出码 [evidence: analysis]"
  - "OpenAI o1-preview + self-debug 将成功率提升到 42.2%，但平均单任务 API 成本达 0.636 美元，约为 Claude-3.5-Sonnet + self-debug 的 11 倍 [evidence: analysis]"
related_work_position:
  extends: "DiscoveryBench-Real (Majumder et al. 2024)"
  competes_with: "SciCode (Tian et al. 2024); BLADE (Gu et al. 2024)"
  complementary_to: "OpenHands CodeAct (Wang et al. 2024); Self-Debug (Chen et al. 2024)"
evidence_strength: moderate
pdf_ref: paperPDFs/Agentic_Reasoning_Benchmarks_Applications_of_Agentic_Reasoning_Scientific_Discovery_Agents/NeurIPS_2024/2024_ScienceAgentBench_Toward_Rigorous_Assessment_of_Language_Agents_for_Data_Driven_Scientific_Discovery.pdf
category: Survey_Benchmark
---

# ScienceAgentBench: Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discovery

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2410.05080), [Project](https://osu-nlp-group.github.io/ScienceAgentBench/)
> - **Summary**: 这篇工作把“科学代理是否真的会做科研”拆成 102 个来自真实论文的数据驱动科研代码任务，并用可执行程序、任务成功标准、细粒度 rubric 和成本一起做严格评测。
> - **Key Performance**: Claude-3.5-Sonnet + self-debug 在有专家知识时 SR 仅 34.3%；OpenAI o1-preview + self-debug 达到 42.2%，但成本约 \$0.636/任务。

> [!info] **Agent Summary**
> - **task_path**: 自然语言科研任务说明 + 数据集目录/预览 (+ 可选专家知识) -> 自包含 Python 程序文件
> - **bottleneck**: 真实科研代码任务中的异构数据处理、学科专用工具调用与可靠评测，长期被端到端演示掩盖
> - **mechanism_delta**: 把“科学发现能力”改写为 102 个出版物来源的可执行代码任务，并引入防污染、防捷径与分阶段评分
> - **evidence_signal**: 多模型多框架评测显示最佳非 o1 代理也只到 34.3% SR，且人评把主要失效点定位到数据加载/处理
> - **reusable_ops**: [publication-grounded-task-curation, execution-based-success-checking]
> - **failure_modes**: [heterogeneous-data-processing-errors, domain-api-hallucination]
> - **open_questions**: [how-to-automate-rubric-grading, how-to-improve-domain-tool-use]

## Part I：问题与挑战

这篇论文本质上不是在提一个新 agent，而是在提一个更严格的**科学代理评测框架**。

### 这篇论文真正要解决什么问题？
当前很多“AI Scientist / 自动化科研”工作，容易直接用端到端结果来证明能力，比如生成一篇论文、写一个研究报告、或让 LLM reviewer 打分。  
但作者认为，这类评测有一个关键缺陷：

- **终局产物太开放，难以客观归因**
- 你很难知道 agent 到底是：
  - 真会处理科研数据、
  - 只是模板拼接、
  - 受益于训练污染、
  - 还是直接走了 shortcut（如偷看测试标签）

因此，**真正的瓶颈不是“agent 能不能讲出一个研究故事”，而是它能不能在真实科研工作流里，稳定写出可执行代码来处理、分析、建模和可视化数据。**

### 为什么现在要解决？
因为这类能力声称已经开始出现，但基础证据还不够扎实：

- LLM 已经具备较强代码生成、工具调用、推理能力
- 社区开始讨论端到端自动科研
- 但缺少一个**真实、细粒度、可执行、可防作弊**的 benchmark 来测量“科学 copilot”是否真的可用

作者的立场很明确：  
**在宣称自动化整个科研流程之前，先证明 agent 能完成流程里的关键原子任务。**

### 输入 / 输出接口
论文把问题定义得非常工程化、也非常可验证：

- **输入**
  - Task instruction：自然语言任务描述
  - Dataset information：数据目录结构与预览
  - Optional expert-provided knowledge：学科专家补充知识
- **输出**
  - 一个自包含的 Python 程序文件
  - 该程序执行后需保存指定结果文件，如预测 CSV、分析结果或图像

这个接口的好处是：  
它直接对齐了“科学 copilot”的使用方式——科学家不是要抽象建议，而是要**能跑的代码草稿**。

### 边界条件
这个 benchmark 的覆盖面是明确受限的：

- 只关注 **data-driven scientific discovery**
- 只收录 **Python** 工作流
- 覆盖 4 个学科：
  - Bioinformatics
  - Computational Chemistry
  - Geographical Information Science
  - Psychology & Cognitive Neuroscience
- 最终共 **102 个任务**，来自 **44 篇同行评审论文**
- 为了评测效率，任务大体限制在可在约 10 分钟内完成
- **不覆盖**
  - 文献综述
  - 研究假设生成
  - 实验规划
  - 湿实验/实验室硬件控制
  - 非 Python 主流科研生态（如 R/Matlab/Stata）

## Part II：方法与洞察

作者的核心贡献，不是训练更强模型，而是**重新设计“如何测”**。

### Benchmark 是怎么搭起来的？
作者遵循三个原则：

1. **真实性**
   - 直接从真实论文中抽取任务
   - 要求有开放数据与代码
   - 由 9 位学科专家验证任务是否符合真实科研流程

2. **严格、分层的评测**
   - 所有任务统一为“生成自包含 Python 程序”
   - 自动检查程序是否能运行、是否真正完成任务、生成结果是否正确、成本多高
   - 对图像任务还使用 GPT-4o judge 做质量评分
   - 另配 task-specific rubrics 做细粒度人工评分

3. **多阶段质量控制与防作弊**
   - 标注员构建任务
   - 专家校验
   - 交叉复现验证
   - 数据层面做防污染与防 shortcut 处理

### 评测设计的关键部件
每个 task 有四个组成部分：

- **Task Instruction**
- **Dataset Information**
- **Expert-Provided Knowledge**
- **Annotated Program**

评测指标则分为四类：

- **VER**：程序能否无报错执行并生成指定输出
- **SR**：是否满足任务成功标准
- **CBS**：与标注程序的语义级代码相似度
- **Cost**：平均单任务 API 成本

此外还有一个很重要的补充：
- **Rubric-based human evaluation**
  - 将任务拆成五阶段：
    1. Data Loading
    2. Data Processing
    3. Modeling or Visualization
    4. Output Formatting
    5. Output Saving

这使 benchmark 不只是给一个“对/错”，还能告诉你**错在什么环节**。

### 防污染 / 防捷径设计
这是这篇论文很有价值的部分。

作者观察到，agent 可能通过不正当方式“解题”，例如：
- 直接读取测试标签
- 使用训练中见过的数据 loader 或代码模板，绕过真正建模

为此他们做了两类处理：

- **随机删去测试集中的 5 个样本**
  - 让训练语料中记忆的默认 loader 与当前 benchmark 数据错位
- **对建模任务重新划分数据并隐藏测试标签**
  - 用 dummy value 替换真实标签，防止 agent 直接抄答案

这一步把 benchmark 从“容易被背答案污染的代码测试”变成了更接近真实能力测量的设置。

### 核心直觉

过去的测量瓶颈在于：  
**科研 agent 的输出太开放，导致评测更像“看起来像不像”，而不是“到底能不能做”。**

作者改变的关键旋钮是：

- **把输出空间收缩到“可执行、自包含、可验证的 Python 程序”**
- **把成功定义从“文本上说得通”改成“执行后真完成任务”**
- **把失败归因从“总分低”细化成“加载错、处理错、建模错、格式错、保存错”**
- **把污染与 shortcut 作为 benchmark 设计的一部分，而不是事后假设**

这带来的能力变化是：

- benchmark 能更可靠地区分“会写科研代码”与“会写像科研代码的文本”
- 能诊断 agent 的真实短板在**数据处理**还是**领域 API 使用**
- 能比较不同 agent framework 的**成本-性能权衡**

换句话说，论文没有改 agent，本质上是改了**测量装置**；而一旦测量装置更严，很多“agent 已可自动科研”的乐观结论就站不住了。

### 战略取舍

| 设计选择 | 改变了什么瓶颈 | 获得的能力 | 代价 |
| --- | --- | --- | --- |
| 统一输出为自包含 Python 程序 | 从抽象文本产物转为可执行对象 | 可做执行级与结果级验证 | 只覆盖代码型科研任务 |
| 从同行评审论文抽取任务 + 专家验证 | 减少 synthetic / GitHub-only 分布偏差 | 更贴近真实科学工作流 | 标注和验证成本高，规模受限 |
| 数据改造防污染、防捷径 | 降低记忆化代码与偷看标签的收益 | 评测更可信 | 某些自动 loader 会被刻意失配惩罚 |
| Outcome metrics + rubrics + 图像 judge | 从单点分数转为多层诊断 | 能定位错误阶段 | 人工与闭源 judge 增加复现成本 |
| 同时记录 Cost | 把“能做”与“值不值”放在一起 | 能比较 agent 实用性 | 价格随 API 波动，跨时间复现较难 |

## Part III：证据与局限

### 关键证据信号

1. **能力上限信号：当前 agent 离“自动科研”还很远**
   - 最佳非 o1 配置是 **Claude-3.5-Sonnet + self-debug**
   - 成功率仅 **32.4%（无专家知识）/ 34.3%（有专家知识）**
   - 这说明即便给到真实任务、三次尝试、甚至专家知识，现有 agent 也远不能覆盖大多数科研代码任务

2. **机制信号：执行反馈比“大而复杂的 agent 框架”更关键**
   - 对 Claude-3.5-Sonnet，self-debug 相比 direct prompting：
     - **SR: 16.7% → 32.4%**
   - 相比 OpenHands CodeAct：
     - **SR: 21.6% → 32.4%**
     - **Cost: \$0.958 → \$0.057**
   - 这说明在该场景里，真正有效的因果旋钮不是更大的动作空间，而是**让模型看见执行错误并迭代修复**

3. **知识使用信号：知道更多不等于用得更好**
   - 专家知识通常提升 SR/CBS
   - 但常常让 VER 下降
   - 原因是模型看到更专业的工具/API 后，容易产生：
     - 错误调用
     - 幻觉 API
     - 更复杂但更脆弱的实现
   - 这揭示出瓶颈并不只是“缺知识”，而是**不能把知识稳定落地成可执行代码**

4. **诊断信号：失败主要卡在数据加载/处理，而非简单格式遵循**
   - rubric 人评显示：
     - 成功与失败程序在 **Data Loading / Data Processing** 阶段差异最明显
     - 在 **Output Formatting / Saving** 阶段差异反而较小
   - 这说明模型并不是不会“照着要求存文件”，而是更早就在**异构科研数据处理**和**学科工具调用**上失效了

### 1-2 个最值得记住的指标
- **34.3% SR**：最佳非 o1 agent 的上限，说明“科学代理”还远非可独立工作
- **42.2% SR at \$0.636/task**：更多推理时计算能继续抬高上限，但成本陡增，实用性未必划算

### 局限性
- **Fails when**: 把该 benchmark 外推到非 Python、超长运行、大规模数据、R/Matlab/Stata 工作流，或需要文献综述/假设生成/实验规划/湿实验控制时，它不能代表“完整科学发现能力”。
- **Assumes**: 任务可从开放论文与代码中抽取；有学科专家参与验证和知识补充；评测端可用 conda + pipreqs/pip-tools 重建环境；图像任务依赖 GPT-4o judge；许多主结果依赖闭源模型 API，因此复现成本不低。
- **Not designed for**: 评估研究想法新颖性、论文写作质量、实验室安全边界，或直接判断某 agent 是否能端到端产出真正有科学价值的新发现。

### 可复用组件
这篇论文最值得复用的不是某个模型，而是它的 benchmark engineering：

- **publication-grounded task curation**：从论文而非 GitHub issue 抽任务
- **anti-shortcut dataset editing**：用删样本/标签掩蔽降低污染与作弊
- **execution-based success scripts**：把“成功”落实到可执行检查
- **five-stage rubrics**：把失败定位到加载、处理、建模、格式、保存
- **cost-aware evaluation**：把“能力”和“代价”一起报告

## Local PDF reference

![[paperPDFs/Agentic_Reasoning_Benchmarks_Applications_of_Agentic_Reasoning_Scientific_Discovery_Agents/NeurIPS_2024/2024_ScienceAgentBench_Toward_Rigorous_Assessment_of_Language_Agents_for_Data_Driven_Scientific_Discovery.pdf]]