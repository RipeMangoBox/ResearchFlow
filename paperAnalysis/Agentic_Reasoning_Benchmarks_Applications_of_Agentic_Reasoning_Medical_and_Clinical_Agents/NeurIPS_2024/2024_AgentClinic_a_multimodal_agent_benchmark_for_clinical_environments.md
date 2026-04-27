---
title: "AgentClinic: a multimodal agent benchmark for clinical environments"
venue: NeurIPS
year: 2024
tags:
  - Survey_Benchmark
  - task/MLLM-evaluation
  - multi-agent-simulation
  - bias-simulation
  - tool-use-evaluation
  - dataset/AgentClinic-MedQA
  - dataset/AgentClinic-MIMIC-IV
  - dataset/AgentClinic-NEJM
  - opensource/full
core_operator: "用患者-医生-测量-裁判四代理把静态医学题重写为可问诊、可开检查、可看图像、可注入偏见的交互式临床评测。"
primary_logic: |
  临床代理评测目标 → 将 MedQA、MIMIC-IV、NEJM 病例重构为 OSCE 风格多代理环境，并加入多语种、专科、工具与偏见扰动 → 用诊断正确率、患者信任/依从性/复诊意愿与人工对话评分进行评测 → 揭示模型在信息采集、工具使用、跨语言、多模态与偏见鲁棒性上的能力边界
claims:
  - "将 MedQA 病例改写为 AgentClinic 的序贯问诊/检查环境后，模型诊断准确率显著下降，且 MedQA 分数对 AgentClinic-MedQA 仅弱预测 [evidence: analysis]"
  - "在 AgentClinic-MedQA 与 AgentClinic-MIMIC-IV 上，Claude-3.5-Sonnet 分别达到 62.1% 和 42.9% 准确率，整体高于 GPT-4、GPT-4o 与多数开源基线 [evidence: comparison]"
  - "偏见扰动暴露了静态准确率之外的风险：即使 GPT-4 的准确率下降较小，隐式偏见仍会明显拉低模拟患者的信任、依从性与再次就诊意愿 [evidence: analysis]"
related_work_position:
  extends: "SAPS (Liao et al. 2024)"
  competes_with: "AMIE (Tu et al. 2024); CRAFT-MD (Johri et al. 2023)"
  complementary_to: "MedAgents (Tang et al. 2023); Agent Hospital (Li et al. 2024)"
evidence_strength: strong
pdf_ref: paperPDFs/Agentic_Reasoning_Benchmarks_Applications_of_Agentic_Reasoning_Medical_and_Clinical_Agents/NeurIPS_2024/2024_AgentClinic_a_multimodal_agent_benchmark_for_clinical_environments.pdf
category: Survey_Benchmark
---

# AgentClinic: a multimodal agent benchmark for clinical environments

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2405.07960), [Project](https://agentclinic.github.io/)
> - **Summary**: 这篇工作把“给定全部信息后回答医学题”的传统评测，升级为“需要主动问诊、申请检查、理解图像、使用工具并处理偏见”的模拟临床环境评测，从而更真实地测出医疗代理的序贯决策能力。
> - **Key Performance**: Claude-3.5-Sonnet 在 AgentClinic-MedQA / AgentClinic-MIMIC-IV 上分别达到 62.1% / 42.9%；GPT-4 在交互步数从 20 降到 10 时准确率从 52% 降到 25%

> [!info] **Agent Summary**
> - **task_path**: 病例主诉/多轮问诊/检查请求/图像与工具交互 -> 开放式临床诊断 + 患者体验评分
> - **bottleneck**: 临床诊断的真实难点是主动获取关键信息，而不是在给定完整上下文后做一次性选择
> - **mechanism_delta**: 把静态病例重写为四代理 OSCE 式环境，并将检查、图像、偏见、多语种和工具使用都变成显式交互动作
> - **evidence_signal**: GPT-4 在 N=20 时为 52%，N=10 时降到 25%；正确案例信息覆盖率 72%，错误案例 63%
> - **reusable_ops**: [osce-case-reformatting, bias-injection, patient-centric-metrics]
> - **failure_modes**: [misses critical history or test requests under turn budget, support-agent realism can shift benchmark rankings]
> - **open_questions**: [how simulated patient ratings correlate with real patient trust, how sensitive results are to proprietary patient/measurement/moderator backbones]

## Part I：问题与挑战

这篇论文抓住的真问题不是“模型是否记住了医学知识”，而是：**模型能否在信息不完整、需要主动采集证据的临床流程里做出正确诊断**。

现有医学 LLM 评测大多是 MedQA 一类静态问答：病例背景、症状、检查结果、候选答案都提前给全，模型只需做一次性选择。但真实临床不是这样。医生需要：

1. 先决定问什么；
2. 再决定做什么检查；
3. 处理患者表达不完整、甚至带偏见或不信任的情况；
4. 在有限轮次和有限注意力内整合信息；
5. 对图像、化验、病史做跨模态判断。

所以，**当前瓶颈不是“最后一步分类”，而是“前面的信息采集与不确定性消解”**。这也是为什么今天必须解决它：LLM 在 USMLE/MedQA 这类考试上已经很高分，但这些高分并不等于能在真实门诊流程里稳定工作。

AgentClinic 的输入/输出接口也正围绕这个瓶颈设计：

- **输入**：医生代理只拿到简短主诉和目标；完整病史、症状、体征、检查结果分散在患者代理与测量代理手里；部分任务还带医学图像。
- **过程**：医生要在最多 20 次交互内主动问诊、请求检查、可选使用检索/反思/笔记等工具。
- **输出**：生成开放式诊断，而不是选项式答案；再由 moderator 根据 ground truth 判定是否正确。

边界条件也很明确：它是**模拟临床环境**，不是实际医疗部署；主要评的是诊断阶段，不是治疗执行、长期随访或医院运营。

## Part II：方法与洞察

### 核心直觉

这篇文章最关键的改变是：

**把“信息一次性给全的医学考试”改成“信息分散、需要主动获取的 OSCE 式多代理临床仿真”。**

这带来的因果变化是：

- **评测瓶颈改变了**：从“知识回忆/选项匹配”转为“信息覆盖率 + 检查选择 + 跨轮记忆 + 工具使用”。
- **暴露出的失败模式改变了**：模型不再只是答错题，而是可能根本没问到关键病史、没开到关键检查、没正确使用图像或工具。
- **可测能力改变了**：除了最终准确率，还能测患者信任、依从性、复诊意愿，以及偏见对医患互动的影响。

论文里一个很有解释力的信号是：在 AgentClinic-MedQA 中，医生代理平均只获取到约 **67%** 的相关信息；而正确诊断的案例信息覆盖率是 **72%**，错误诊断只有 **63%**。这说明错误往往不是“拿到完整信息后不会推理”，而是**上游没有拿到足够信息**。

### 基准设计

#### 1. 四代理交互结构

AgentClinic 由四个语言代理组成：

- **Doctor agent**：被评测对象，只知道简要主诉，负责问诊、申请检查并输出诊断。
- **Patient agent**：持有症状、病史、生活方式等，只能通过对话逐步透露。
- **Measurement agent**：持有体征、化验、影像读片等结果，医生必须显式请求。
- **Moderator agent**：拿 ground truth，对开放式诊断做对错判断。

这个设计的意义在于：把“信息是否存在”与“模型是否成功拿到信息”分开了。

#### 2. 数据来源与覆盖面

基准不是单一数据集，而是多个子环境：

- **AgentClinic-MedQA**：把 MedQA 题目改写成交互式病例；
- **AgentClinic-MIMIC-IV**：从真实 EHR 中筛选单一诊断患者构造病例；
- **AgentClinic-NEJM**：加入图像理解的多模态病例；
- **多语种设置**：7 种语言；
- **专科设置**：9 个医学专科。

这让它不只是“一个模拟器”，而是一个**多维能力探针**：能看一般诊断，也能看 EHR、影像、语言迁移和专科场景。

#### 3. 扰动与工具层

它还在环境上加入两类关键扩展：

- **偏见扰动**：对 doctor 或 patient 注入认知偏见、隐式偏见；
- **工具箱**：Zero-shot CoT、One-shot CoT、Reflection CoT、Adaptive RAG（web/book）、Notebook。

因此它测到的不只是“裸模型能力”，还测“模型是否会因工具变强”“是否会因偏见变差”。

#### 4. 评价指标不只看准确率

除了最终诊断是否正确，AgentClinic 还测：

- 患者对医生的**信心**
- 患者对后续治疗的**依从意愿**
- 患者是否愿意**再次就诊**
- 临床医生对对话的**真实性/同理心评分**

这一步非常关键：它把“临床上看似答对了，但患者不信你”的风险显式化了。

### 战略权衡

| 设计选择 | 获得的诊断信号 | 代价/风险 |
|---|---|---|
| 静态题改为多轮 OSCE 仿真 | 测到信息采集与序贯决策能力 | 环境复杂，评测方差更大 |
| 用 patient/measurement 分拆信息 | 能看模型是否问到、查到关键证据 | 支撑代理本身若不稳定，会引入模拟偏差 |
| 开放式诊断 + moderator 判分 | 避免多选题答案泄漏与猜选项 | LLM judge 可能有解析偏差 |
| 加入偏见与患者感知指标 | 能测“答对但伤害医患关系”的风险 | 患者感知仍来自模拟代理，不是真人 |
| 加入工具箱 | 暴露 agentic/tool-use 差异 | 工具收益高度依赖提示与底座模型 |

## Part III：证据与局限

### 关键证据

**信号 1｜静态高分不等于临床交互能力**  
论文直接比较了 MedQA 与 AgentClinic-MedQA，发现前者对后者只有弱预测性。也就是说，一个模型在考试题上高分，并不代表它在需要主动采集信息的环境里就强。这个结果是整篇论文最核心的“为什么需要新基准”的证据。

**信号 2｜真正的瓶颈是信息采集，而不是最后一步分类**  
GPT-4 在 AgentClinic-MedQA 上，当交互预算从 **20 步降到 10 步** 时，准确率从 **52% 降到 25%**；而当步数增加到 30，准确率也没有继续涨，反而下降到 **43%**。这说明：
- 步数太少：拿不到足够证据；
- 步数太多：上下文变长，模型整合能力变差。  
再结合覆盖率分析（正确 72% vs 错误 63%），可以比较有力地支持“信息获取是因果瓶颈”。

**信号 3｜工具不是普适增益，而是模型依赖能力**  
Claude-3.5 在多数工具设置下最稳；GPT-3.5 在多种工具下反而退化；Llama-3 70B 在 notebook/反思类工具下提升很大，说明**是否会用工具**本身就是能力分层，而非默认附赠能力。  
这比传统 benchmark 更接近真实 agent 使用场景。

**信号 4｜偏见影响不只体现在准确率，还体现在患者关系**  
GPT-4 在偏见注入下准确率下降不算特别大，但患者侧的信任、依从、复诊意愿会明显变差；Mixtral-8x7B 则在偏见场景下准确率退化更明显。  
这说明如果只看最终 diagnosis accuracy，会漏掉临床上非常重要的患者体验风险。

**信号 5｜跨语言与多模态仍然是明显短板**  
多语种场景里，所有模型普遍英语最好；Claude-3.5 平均多语表现最稳。多模态 NEJM 场景里，即使最好的 Claude-3.5，准确率也只有 **37.2%**（图像初始提供时）。这表明“会看图 + 会问诊 + 会整合”仍远没被解决。

### 1-2 个最值得记住的数

- **Claude-3.5-Sonnet**：62.1%（AgentClinic-MedQA），42.9%（AgentClinic-MIMIC-IV）
- **GPT-4 的交互预算敏感性**：52%（20 步）→ 25%（10 步）

### 局限性

- **Fails when**: 病例需要多重并发诊断、长期治疗规划、真实体格操作、多人协作医疗流程时，当前四代理+单一最终诊断设置会失真。
- **Assumes**: 病例可被压缩成单一 ground-truth 诊断；GPT-4 等支撑代理能稳定扮演患者/测量/裁判；翻译和模板化过程不会破坏关键诊断线索。
- **Not designed for**: 真实临床部署安全验证、治疗方案优劣评估、长期预后预测、医院资源调度或保险/护理流程建模。

资源与可复现性方面，也有几个实际限制需要明说：

- 虽然 benchmark 是开源定位，但很多实验依赖 **GPT-4/Claude 等闭源 API**；
- proprietary 模型可能存在 **MedQA 数据泄漏** 风险；
- 患者、测量和 moderator 都可能把支撑模型的偏好带入评测；
- 人类对话评分只基于 **3 位临床医生、20 段对话**，规模偏小；
- MIMIC-IV 只选了 **单一诊断** 子集，弱化了真实住院场景中的共病复杂性。

### 可复用组件

这篇工作最值得复用的，不只是 benchmark 名字，而是下面这些“操作子”：

- **OSCE 式病例重写模板**：把静态 QA 重写为可交互病例；
- **四代理评测 harness**：doctor / patient / measurement / moderator 的职责解耦；
- **偏见扰动模块**：可系统测试 doctor/patient bias；
- **patient-centric metrics**：把 trust/compliance/consult-again 纳入评测；
- **工具沙箱**：统一比较反思、RAG、笔记、CoT 等 agent 工具。

如果你以后要做医疗 agent、Embodied clinical simulation，甚至一般领域的“从静态题到交互任务”的 benchmark 迁移，这套设计都很有参考价值。

![[paperPDFs/Agentic_Reasoning_Benchmarks_Applications_of_Agentic_Reasoning_Medical_and_Clinical_Agents/NeurIPS_2024/2024_AgentClinic_a_multimodal_agent_benchmark_for_clinical_environments.pdf]]