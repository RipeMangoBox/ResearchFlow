---
title: "Agent-X: Evaluating Deep Multimodal Reasoning in Vision-Centric Agentic Tasks"
venue: arXiv
year: 2025
tags:
  - Survey_Benchmark
  - task/MLLM-evaluation
  - step-level-evaluation
  - reasoning-trace
  - tool-use-evaluation
  - dataset/Agent-X
  - opensource/full
core_operator: 通过真实视觉任务、可执行工具链与步骤级评分，把代理的最终答对率拆解为可诊断的推理、工具使用与规划能力剖面
primary_logic: |
  评测目标（真实视觉代理中的多步推理与工具使用） → 以图像/视频/文本化视觉场景和隐式工具需求查询构造任务，并经 LMM 生成+人工校验形成 reasoning trace → 用 Step-by-Step / Deep Reasoning / Outcome 三层指标评分 → 定位视觉 grounding、工具调用、参数格式与链式一致性的能力边界
claims:
  - "Agent-X构建了828个真实视觉代理任务，覆盖6个环境、14种可执行工具，并包含人工校验的步骤级推理轨迹，可用于系统评估视觉代理的多步推理与工具使用能力 [evidence: analysis]"
  - "在主评测协议下，没有任何被测模型在Agent-X上超过45%的Goal Accuracy，说明当前LMM代理在真实多步视觉任务中仍远未可靠 [evidence: comparison]"
  - "工具调用与参数/格式遵循是主要失效点，模型普遍出现无效JSON、工具幻觉和错误的单步多工具调用，显著削弱全链路成功率 [evidence: analysis]"
related_work_position:
  extends: "GTA (Wang et al. 2024)"
  competes_with: "GAIA (Mialon et al. 2023); MLGym (Nathani et al. 2025)"
  complementary_to: "ReAct (Yao et al. 2023); LLaVA-Plus (Liu et al. 2024)"
evidence_strength: moderate
pdf_ref: paperPDFs/Agentic_Reasoning_Benchmarks_Applications_of_Agentic_Reasoning_Embodied_Agents/arXiv_2025/2025_Agent_X_Evaluating_Deep_Multimodal_Reasoning_in_Vision_Centric_Agentic_Tasks.pdf
category: Survey_Benchmark
---

# Agent-X: Evaluating Deep Multimodal Reasoning in Vision-Centric Agentic Tasks

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2505.24876) · [Dataset](https://huggingface.co/datasets/Tajamul21/Agent-X) · [Code](https://github.com/mbzuai-oryx/Agent-X)
> - **Summary**: Agent-X 用 828 个真实视觉代理任务、14 个可执行工具和步骤级评测协议，把“模型会不会答题”升级为“模型能否看懂、规划、调用工具并连贯完成整条推理链”的系统诊断。
> - **Key Performance**: 最佳 **Goal Accuracy = 45%**（OpenAI-o4-mini）；最佳 **Toolset Accuracy = 68%**（GPT-4o），且所有模型完整任务成功率仍低于 50%。

> [!info] **Agent Summary**
> - **task_path**: 真实图像/视频/网页/数学视觉场景 + 隐式工具需求查询 -> 工具调用序列与多步推理链 -> 最终答案与能力诊断分数
> - **bottleneck**: 现有基准大多只看最终答案，无法定位失败究竟来自视觉 grounding、工具选择、参数格式还是跨步逻辑断裂
> - **mechanism_delta**: 把评测对象从“最终答对”改成“步骤执行 + 深层推理 + 最终结果”三层联合打分，并要求任务本身不显式提示工具
> - **evidence_signal**: 10 个主流模型在 828 个任务上无一超过 45% Goal Accuracy，且格式/工具调用错误高频出现
> - **reusable_ops**: [隐式工具查询构造, 步骤级推理链评测]
> - **failure_modes**: [工具幻觉或错误选型, JSON/参数格式错误]
> - **open_questions**: [如何评测更长时程与更开放交互的代理任务, 如何减少LLM judge带来的评测偏置]

## Part I：问题与挑战

这篇论文要解决的不是“再做一个多模态基准”，而是**如何真正测到视觉代理的深度推理能力**。

### 现有评测缺了什么
作者指出，现有 agent benchmark 往往有四个共同短板：

1. **过度看终局**：只看 final answer 是否正确，看不到中间哪一步坏掉。
2. **视觉不够真实**：很多任务偏文本、静态图像或合成环境，难反映真实视频/网页/多图比较场景。
3. **工具使用被“明示”**：问题里直接暗示该用什么工具、按什么步骤做，测到的是跟提示，而不是自主规划。
4. **缺少深推理诊断**：即使模型能写出貌似合理的 chain-of-thought，也未必真的视觉对齐、工具正确、链路一致。

### 真正的瓶颈
对视觉中心代理来说，真正瓶颈不是单次感知，而是这条链条能否稳定闭合：

**视觉输入理解 → 选择正确工具 → 传入正确参数 → 组织多步推理 → 输出与证据一致的最终答案**

任何一环出错，最终都可能“看起来会推理，但做不成任务”。

### 输入/输出接口与边界
Agent-X 的任务接口比较明确：

- **输入**：单图、多图、视频、图文混合视觉上下文
- **问题形式**：自然语言查询，且**不显式写出要调用哪些工具**
- **输出**：步骤化 reasoning trace、工具调用、最终答案、justification
- **边界条件**：
  - 固定 **14 个工具**
  - 覆盖 **6 个环境**
  - 整体是**测试集型 benchmark**
  - 当前主要是**单语（英语）**
  - 平均任务长度约 **3.4 步**

所以它瞄准的是：**视觉代理在受控工具集合下的真实世界多步任务评测**，而不是无限开放环境下的通用智能。

## Part II：方法与洞察

### Benchmark 怎么搭起来
Agent-X 的核心不是模型结构，而是**任务构造 + 评测协议**。

#### 1. 任务构造
作者最终构建了 **828** 个任务，覆盖：

- **6 个环境**：通用视觉推理、网页浏览、安全监控、自动驾驶、体育、数学推理
- **14 个工具**：OCR、SceneDescriber、RegionDescriber、ObjectCounter、WebSearch、Calculator、Solver、ImageGenerator 等
- **输入模态**：716 个图像任务、112 个视频任务
- **问题类型**：factual / interpretive / generative

一个任务被组织为：
- 视觉上下文 \(V\)
- 查询 \(Q\)
- 使用到的工具集合 \(T\)
- 推理轨迹 \(R\)
- 最终答案 \(A\)
- justification \(J\)

#### 2. 半自动但强人工校验
流程是：

1. 给 LMM 视觉输入和可用工具列表，让它先生成候选查询  
2. 人工重写与筛选，确保：
   - 问题真实
   - 不直接暴露工具名
   - 确实需要多步推理
3. 再让 LMM 生成 reasoning trace、工具调用、答案与 justification
4. 人工继续校验工具选型、参数、逻辑一致性和答案正确性

这使它比纯人工更能扩规模，又比纯合成更接近真实使用场景。

#### 3. 三层评测协议
Agent-X 的关键设计是把评测拆成三层：

- **Step-by-Step**
  - Grounding Score
  - Tool Precision
  - Tool Accuracy

- **Deep Reasoning**
  - Faithfulness
  - Context Score
  - Factual Precision
  - Semantic Accuracy

- **Outcome**
  - Goal Accuracy
  - Goal Accuracy/ImgGen
  - Toolset Accuracy

也就是说，它不只问“答对了吗”，而是问：

- 你有没有看对视觉内容？
- 你有没有选对工具？
- 你有没有把工具用对？
- 你的步骤之间是不是逻辑连贯？
- 你的最终答案是不是建立在前面那条链上？

### 核心直觉

过去多数 benchmark 的测量瓶颈在于：**只能看到终局，不能看到链路**。  
Agent-X 改变的是这个“观测点”。

**什么变了**：从单一 final-answer 评分，变成“步骤执行 + 深推理一致性 + 结果完成度”的分层评测。  
**哪个约束变了**：把工具提示从题面拿掉，并把工具调用做成可执行、可核验的结构化链。  
**能力上发生了什么变化**：评测者终于能区分——模型到底是：

- 没看懂图像/视频，
- 不会选工具，
- 会选但参数/格式错，
- 还是局部都对但整条链串不起来。

这就是本文最大的诊断增益。

### 为什么这个设计有效
因为视觉代理的失败通常不是“纯答案错误”，而是**中间状态不可见**。  
Agent-X 把中间状态显式化之后，错误被重新分桶为：

- planning failure
- formatting failure
- visual grounding failure
- spatial/temporal reasoning failure
- tool hallucination

因此它能把“模型为什么不行”说清楚，而不是只给一个低分。

### 战略权衡

| 设计选择 | 得到的能力 | 代价/风险 |
|---|---|---|
| 真实图像/视频任务替代纯合成样本 | 更接近部署场景，能暴露真实视觉和工具瓶颈 | 噪声更高，控制变量更弱 |
| 查询中去掉显式工具提示 | 真正测自主规划与工具选择 | 任务更难，分数更低 |
| LMM 生成初稿 + 人工重写校验 | 兼顾规模与质量 | 仍可能带入生成偏置 |
| 三层细粒度指标替代单一准确率 | 能定位失败环节 | 评测实现更复杂，也更依赖 judge 质量 |
| LLM judge + 开源 judge + 人类子集复核 | 可扩展且可审计 | 绝对分数仍会随 judge 波动 |

## Part III：证据与局限

### 关键证据

#### 证据 1：现有模型离“可用视觉代理”还有明显距离
主结果里，没有任何模型的 **Goal Accuracy** 超过 **45%**。  
这说明在真实多步视觉任务中，哪怕是最强的闭源模型，也无法稳定完成整条代理链路。

一个很直观的结论是：  
**视觉代理当前最大的短板，不是“不会生成答案”，而是“无法稳定走完整个过程”。**

#### 证据 2：推理强，不等于链路稳；但推理弱通常更难成功
论文观察到，深推理指标较高的模型，通常最终完成率也更高。  
例如 GPT-4o 在 Faithfulness / Factual Precision 上表现较强，o4-mini 的终局 Goal Accuracy 最高。  
这支持了一个重要判断：

**深推理质量与任务成功正相关，但真正卡住系统的是“推理能否转化为合规、正确的工具执行”。**

#### 证据 3：工具调用与格式遵循是最稳定暴露的瓶颈
代表性错误分析显示，多模型都高频出现：

- 无效 JSON 参数
- 单步多工具调用
- final answer 格式不合规
- hallucinated tools
- 视频任务中的浅层逐帧推理不足

而且这些问题不是个别模型特有，而是跨模型家族反复出现。  
这意味着当前代理系统的薄弱点，更像是**执行接口层**，而不是单纯的语言推理层。

### 1-2 个最关键指标
- **最佳 Goal Accuracy：45%（OpenAI-o4-mini）**
- **最佳 Toolset Accuracy：68%（GPT-4o）**

这两个数已经足够概括论文主结论：  
**模型可以部分推理、部分调用工具，但离稳定完成真实视觉代理任务还差一大截。**

### 局限性
- **Fails when**: 需要多语言评测、超出六个环境的开放世界任务、长时程多轮交互代理、强动态在线环境时，Agent-X 的覆盖会不够；对开放式图像生成任务，它也不直接评价生成质量本身。
- **Assumes**: 固定 14 工具的工具宇宙、英语查询、LMM 生成初稿后的人类重写与校验、结构化 JSON 输出约束，以及可用的 judge 模型；数据构建还依赖显著人工成本（5 名标注者、每人约 50 小时）与部分闭源 API。
- **Not designed for**: 具身控制、机器人动作执行、持续在线学习、无工具约束的开放式 agent 环境，或纯产品级端到端部署评测。

### 复现与可扩展性备注
这篇论文在可复现性上做得比很多 benchmark 更好：

- 数据和代码公开
- 提供开源 judge 的一致性检查
- 有人类评测子集验证主结论

但仍要看到两个现实依赖：

1. **主评测与数据生成部分依赖 GPT-4o 类闭源模型**
2. **评测质量依赖结构化输出约束与人工质检**

所以它是“相对可复现”，不是“零依赖可复现”。

### 可复用组件
对后续研究最有复用价值的部分有四个：

- **隐式工具查询设计原则**
- **LMM 生成 + 人工精修的数据构造流水线**
- **Step / Deep Reasoning / Outcome 三层评测框架**
- **工具调用、格式错误、视觉误解、时空推理错误的统一错误 taxonomy**

![[paperPDFs/Agentic_Reasoning_Benchmarks_Applications_of_Agentic_Reasoning_Embodied_Agents/arXiv_2025/2025_Agent_X_Evaluating_Deep_Multimodal_Reasoning_in_Vision_Centric_Agentic_Tasks.pdf]]