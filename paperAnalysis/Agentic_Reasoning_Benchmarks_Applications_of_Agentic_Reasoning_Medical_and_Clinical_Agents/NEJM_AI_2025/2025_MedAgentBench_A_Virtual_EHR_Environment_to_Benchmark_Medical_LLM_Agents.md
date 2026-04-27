---
title: "MedAgentBench: A Realistic Virtual EHR Environment to Benchmark Medical LLM Agents"
venue: "NEJM AI"
year: 2025
tags:
  - Survey_Benchmark
  - task/medical-agent-evaluation
  - task/ehr-tool-use
  - fhir-api
  - interactive-environment
  - rule-based-grading
  - dataset/MedAgentBench
  - dataset/STARR
  - opensource/full
core_operator: "用医生编写的患者特异任务、FHIR 虚拟 EHR 和规则化单次评分，把医疗 LLM 的评测从静态答题升级为真实病历交互任务完成度测量。"
primary_logic: |
  高层临床指令 + 患者EHR上下文 → 受限FHIR API多轮交互与规则化评分 → 任务成功率、错误模式与能力边界
claims:
  - "MedAgentBench 将医疗代理评测从静态问答扩展到 300 个医生编写、患者特异、可验证的 EHR 交互任务，并基于 100 名患者的 FHIR 虚拟环境进行测量 [evidence: analysis]"
  - "在 12 个主流模型上，最佳模型 Claude 3.5 Sonnet v2 的总体成功率为 69.67%，说明该基准仍明显未饱和 [evidence: comparison]"
  - "多数模型在查询型任务上优于动作型任务，且常见失败来自工具调用格式错误与答案输出格式不匹配 [evidence: analysis]"
related_work_position:
  extends: "AgentBench (Liu et al. 2023)"
  competes_with: "AgentClinic (Schmidgall et al. 2024); tau-bench (Yao et al. 2024)"
  complementary_to: "Many-shot in-context learning (Jiang et al. 2024); Meta-prompting (Suzgun and Kalai 2024)"
evidence_strength: moderate
pdf_ref: paperPDFs/Agentic_Reasoning_Benchmarks_Applications_of_Agentic_Reasoning_Medical_and_Clinical_Agents/NEJM_AI_2025/2025_MedAgentBench_A_Virtual_EHR_Environment_to_Benchmark_Medical_LLM_Agents.pdf
category: Survey_Benchmark
---

# MedAgentBench: A Realistic Virtual EHR Environment to Benchmark Medical LLM Agents

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2501.14654), [GitHub](https://github.com/stanfordmlgroup/MedAgentBench), [Docker](https://hub.docker.com/r/jyxsu6/medagentbench)
> - **Summary**: 这篇工作构建了一个基于 FHIR 的虚拟 EHR 医疗 agent 基准，把“会做医疗问答”升级为“能否在病历系统里完成真实临床信息与操作任务”的单次成功率评测。
> - **Key Performance**: 最佳模型 Claude 3.5 Sonnet v2 的整体成功率为 69.67%；其 Query SR 为 85.33%，Action SR 为 54.00%。

> [!info] **Agent Summary**
> - **task_path**: 高层临床指令 + 患者EHR/FHIR上下文 -> 多轮工具调用与最终答案/动作 -> 单任务成功率评测
> - **bottleneck**: 现有医疗评测大多停留在静态 QA，无法测到真实 EHR 场景中的规划、检索、工具调用和格式遵从
> - **mechanism_delta**: 用医生编写的患者特异任务 + FHIR 兼容虚拟 EHR + pass@1 规则 grader，把“会答题”改成“会在病历系统中完成任务”的测量
> - **evidence_signal**: 12 个模型上最佳总体 SR 仅 69.67%，且查询/动作/难度分层差异明显
> - **reusable_ops**: [FHIR标准化接口, 规则化任务grader]
> - **failure_modes**: [工具调用语法不合法, 最终答案格式与要求不匹配]
> - **open_questions**: [POST 动作仅做 payload 校验而非真实执行会遗漏哪些风险, 单中心 Stanford 数据分布下的模型排名能否外推到其他医院]

## Part I：问题与挑战

这篇论文要解决的真问题，不是“模型会不会回答医学题”，而是：

**模型能不能像一个受约束的临床助理一样，在真实病历接口里把高层指令拆成正确的多步操作。**

### 为什么这是瓶颈
现有医疗 LLM 评测，如多项选择题或结构化 QA，已经越来越接近饱和。  
但临床工作里真正高频、又最耗时的任务，往往不是开放式问答，而是：

- 找到某个病人的 MRN
- 查最近 24 小时的化验值
- 聚合多次测量得到平均值
- 把新生命体征写回病历
- 满足条件时下化验、转诊或用药订单

这些任务有几个传统 QA 测不到的特征：

1. **患者特异性**：答案不在模型参数里，而在当前病人的纵向病历里。  
2. **交互性**：需要多轮查询，再决定下一步动作。  
3. **协议约束**：必须遵守 EHR/API 的调用格式。  
4. **高风险单次正确性**：临床里“差不多对”不够，尤其不能靠多次采样挑最好结果。

### 为什么现在要做
论文给出的背景很直接：

- agentic LLM 已从聊天走向“规划 + 工具调用 + 执行”
- 医疗行政负担高，医生直接临床照护时间占比有限
- 医疗系统对安全、信任、监管要求远高于普通 agent 场景

所以，**如果没有一个像样的医疗 agent benchmark，大家既无法比较系统，也无法知道离真实落地还有多远。**

### 输入 / 输出接口
MedAgentBench 的评测接口很清楚：

- **输入**：临床高层任务指令 + 医院/EHR 上下文 + 患者数据状态
- **中间过程**：agent 在 FHIR 环境中做 GET / POST / finish 三类决策
- **输出**：最终答案或动作 payload，并以 pass@1 判断是否成功

### 边界条件
它聚焦的是**病历系统内的医疗信息与行政工作流**，而不是所有医疗行为：

- 主要覆盖 inpatient / outpatient medical scenarios
- 重点是 EHR 内的信息检索、记录、聚合、下单
- 不覆盖真实的跨团队协作、线下流程、手术/护理等复杂临床执行链路

---

## Part II：方法与洞察

MedAgentBench 不是单一数据集，而是一套完整的评测构件：

1. **任务层**：300 个由执业医生编写的临床任务  
2. **患者状态层**：100 名患者、约 78.5 万条真实去标识化病历记录  
3. **交互层**：FHIR 兼容虚拟 EHR 环境  
4. **评分层**：规则化 grader + pass@1 成功率

### 它具体做了什么

#### 1. 任务设计：从“问答”变成“工作流请求”
任务由两位内科医生编写，共 300 个，覆盖 10 个细分类别，可归为 7 类大任务：

- 患者信息检索
- 检验结果检索
- 患者数据聚合
- 记录患者数据
- 化验/检查下单
- 转诊下单
- 药物下单

这些任务不是抽象问题，而是贴近临床日常的自然语言请求，并带有真实系统上下文，如：

- 当前时间
- LOINC / SNOMED / NDC 等编码
- 医院本地配置
- 结果输出格式约束

这一步很关键：**它把 benchmark 的难点从“医学常识”转移到了“面向真实系统的任务完成”。**

#### 2. 患者环境：用真实来源的纵向病历状态做 grounding
患者数据来自 Stanford 的 STARR 去标识化临床仓库，按患者级别做时间扰动，并补充伪造的 MRN、姓名、电话、地址。  
环境中包含近 5 年的：

- Observation（化验、生命体征）
- Procedure
- Condition
- MedicationRequest

核心意义在于：  
**agent 必须从外部状态拿答案，而不是靠参数记忆“猜”。**

#### 3. 接口协议：标准化到 FHIR
作者基于 HAPI FHIR JPA 搭建了 FHIR-compliant 环境，并暴露 9 个函数，如：

- patient.search
- lab.search
- vital.search / vital.create
- medicationrequest.search / medicationrequest.create
- procedure.search / procedure.create
- condition.search

这使得评测对象不是某家私有 EHR 的脚本，而是更接近可迁移的标准接口层。

#### 4. 评分协议：用 pass@1 强化临床约束
作者显式拒绝 pass@k 这类“多试几次总会中”的语言模型评测习惯，而采用：

- **pass@1**
- **最多 8 轮交互**
- 非法动作或超轮数即失败

这背后的评测哲学很清楚：  
**医疗环境容错低，单次出错就可能有代价。**

### 核心直觉

**改变了什么**  
从“给一个文本问题，判断答案对不对”  
变成“给一个高层临床任务，判断 agent 能否在标准化 EHR 接口里一步步完成它”。

**哪个信息瓶颈被改变了**  
从主要考察参数化医学知识与文本推理，  
变成主要考察：

- 患者状态 grounding
- 多步规划
- 工具调用语法
- 输出格式遵从
- 动作 payload 构造

**能力上带来了什么变化**  
因此它测到的不再只是“模型懂不懂医学”，而是：

- 会不会查
- 会不会聚合
- 会不会写
- 会不会按系统协议执行
- 会不会在高约束下稳定收尾

### 为什么这个设计有效
因为现实中的大量临床辅助任务，本质上不是“回答一个知识题”，而是：

> 在给定病人、给定时间窗、给定编码体系和给定接口约束下，完成一个可验证的系统任务。

FHIR 环境把外部状态标准化，医生写任务保证临床相关性，规则 grader 保证客观评分，pass@1 把安全约束带进评测。  
所以它能暴露传统医疗 QA benchmark 很难暴露的失败：**模型并不是不会医学，而是不会在真实系统约束下把医学意图落实为合法操作。**

### 战略取舍

| 设计选择 | 解决了什么评测盲区 | 代价 / 取舍 |
|---|---|---|
| 医生手写 300 个患者特异任务 | 从静态题库转向真实工作流需求 | 覆盖仍有限，受任务编写者偏好影响 |
| FHIR 兼容虚拟 EHR | 提供标准化、可迁移的接口层 | 不能完全覆盖各医院深度定制逻辑 |
| pass@1 + 8 轮上限 | 更贴近医疗场景的单次正确性要求 | 对探索式、反思式 agent 更苛刻 |
| Query / Action 分开统计 | 能定位“会查”与“会做”的差异 | 总分之外需要更细粒度解读 |
| POST 任务用规则检查 payload | 让动作任务可批量评测 | 不是完整的真实执行闭环 |

---

## Part III：证据与局限

### 关键证据信号

#### 1. 这个 benchmark 远未饱和
最强模型 Claude 3.5 Sonnet v2 的总体成功率也只有 **69.67%**。  
对普通 benchmark 来说这不算低，但对医疗 agent 来说，这说明：

- 当前模型已经具备一定 agent 能力
- 但离“高可靠临床执行”还明显不够

这是论文最重要的结论：**现在的模型能做一些事，但还不能放心交给它做所有事。**

#### 2. 查询能力明显先于执行能力成熟
大多数模型在 query task 上显著优于 action task。  
这说明现阶段的瓶颈更接近：

- 正确读取并整合状态后
- 生成合法、合规、精确的动作

也就是说，**医疗 agent 的短板不只是“知道答案”，更是“把答案变成正确系统动作”。**

这也给部署优先级一个很现实的启发：  
**先从 retrieval-only 或 decision-support 型场景落地，可能比直接放开自动写入/下单更安全。**

#### 3. 评测真正抓到了 agent failure，而非纯知识错误
论文分析到的常见失败很“agent”：

- 工具调用格式错误
- 输出中混入多余文本
- 本应输出纯数字，却给出完整句子或错误 JSON 结构

例如，Gemini 2.0 Flash 出现大量 invalid action。  
这说明 benchmark 在测的是**协议遵从 + 工具使用可靠性**，而不仅是医学知识问答。

#### 4. 排名会随子任务类型变化，说明它测到的是多种子能力
总体最强不代表所有子能力都最强。  
查询、动作、不同难度层级上的模型表现差异很大，说明 MedAgentBench 不是单一分数游戏，而是在同时测：

- grounding
- 规划
- API 格式遵从
- payload 生成
- 收尾格式控制

### 局限性

- **Fails when**: 任务需要真实跨团队协作、持续状态回写后的二次验证、或多系统联动时；模型在严格工具语法和标量输出格式上也容易失败，尤其是多步 action 任务。
- **Assumes**: 假设 FHIR 兼容接口能代表主要 EHR 交互；假设 Stanford 单中心 STARR 病历分布足以构造有代表性的任务；假设医生手写 reference solution 与规则 grader 能覆盖正确性；同时基线对 POST 主要做 payload 规则校验而非真实执行，许多最强结果还依赖闭源 API 模型，且环境本身没有生产级安全与企业日志。
- **Not designed for**: 直接临床部署、需要真实权限控制/审计日志的 EMR 集成、以及超出病历系统交互范围的手术、护理和线下流程场景。

### 可复用组件

这篇工作的可复用价值其实很强，至少包括：

- **FHIR 虚拟 EHR 环境**：适合做医疗 tool-use / agent 研究
- **医生编写的任务模板**：可继续扩展到更多 specialty
- **规则化 grader**：适合复现实验与持续追踪模型进步
- **Query vs Action 切分**：很适合做 agent 能力诊断，而不只看一个总分
- **标准化 API 约束**：可作为未来医疗 agent 训练或对齐的目标接口

### 一句话总结
这篇论文的贡献不是再造一个医疗题库，而是把评测单位从“答对一道题”换成“在虚拟 EHR 里完成一个可验证任务”。  
这让社区第一次能比较系统地回答：**医疗 LLM agent 到底离真实工作流还有多远。**

## Local PDF reference

![[paperPDFs/Agentic_Reasoning_Benchmarks_Applications_of_Agentic_Reasoning_Medical_and_Clinical_Agents/NEJM_AI_2025/2025_MedAgentBench_A_Virtual_EHR_Environment_to_Benchmark_Medical_LLM_Agents.pdf]]