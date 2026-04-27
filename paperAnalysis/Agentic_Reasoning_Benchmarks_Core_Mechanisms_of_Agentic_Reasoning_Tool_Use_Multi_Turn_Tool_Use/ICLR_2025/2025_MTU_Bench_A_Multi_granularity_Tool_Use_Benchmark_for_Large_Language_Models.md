---
title: "MTU-Bench: A Multi-granularity Tool-Use Benchmark for Large Language Models"
venue: ICLR
year: 2025
tags:
  - Survey_Benchmark
  - task/tool-use-evaluation
  - automatic-evaluation
  - synthetic-data
  - instruction-tuning
  - dataset/MTU-Bench
  - opensource/promised
core_operator: 将真实任务型对话自动转写为含 thought-action-observation 的工具调用轨迹，并用无需GPT裁判的多粒度指标对LLM工具使用进行细粒度自动评测
primary_logic: |
  真实任务型对话与工具文档 → GPT-4辅助生成与校验工具调用轨迹并构造正常集/困难集/OOD集 → 用TS、PS、SR、ATS、SATS、TPR、TN、TO等自动指标评分 → 揭示LLM在单/多轮、单/多工具与跨域工具使用中的能力边界
claims:
  - "MTU-Bench包含54,798段对话和136个工具，覆盖单轮单工具、单轮多工具、多轮单工具、多轮多工具与OOD五类场景，且评测阶段无需GPT或人工裁判 [evidence: analysis]"
  - "MTU-Eval中新提出的SATS、TN、TO与人工评价具有较高一致性，Pearson相关系数分别为0.8280、0.8497、0.8821 [evidence: analysis]"
  - "基于MTU-Instruct微调的MTU-LLaMA在MTU-Bench OOD、ToolTalk和API-Bank上均优于LLaMA3-8B-Instruct，并在API-Bank上取得45.66平均分，高于GPT-4的44.45 [evidence: comparison]"
related_work_position:
  extends: "API-Bank (Li et al. 2023)"
  competes_with: "API-Bank (Li et al. 2023); T-Eval (Chen et al. 2024)"
  complementary_to: "Toolformer (Schick et al. 2023); ReAct (Yao et al. 2023)"
evidence_strength: moderate
pdf_ref: "paperPDFs/Agentic_Reasoning_Benchmarks_Core_Mechanisms_of_Agentic_Reasoning_Tool_Use_Multi_Turn_Tool_Use/ICLR_2025/2025_MTU_Bench_A_Multi_granularity_Tool_Use_Benchmark_for_Large_Language_Models.pdf"
category: Survey_Benchmark
---

# MTU-Bench: A Multi-granularity Tool-Use Benchmark for Large Language Models

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [GitHub](https://github.com/MTU-Bench-Team/MTU-Bench.git)
> - **Summary**: 这篇论文把真实任务型对话自动转换成工具调用轨迹，并设计了一套无需 GPT 裁判的细粒度评测体系，用来系统测出 LLM 在多轮、多工具和跨域工具使用中的真实短板。
> - **Key Performance**: SATS/TN/TO 与人工评价的 Pearson 相关系数分别为 0.828/0.850/0.882；MTU-LLaMA 在 API-Bank 上平均分 45.66，高于 GPT-4 的 44.45。

> [!info] **Agent Summary**
> - **task_path**: 对话历史/用户请求 + 工具文档 -> 工具调用决策、参数填写、调用顺序与多轮过程评分
> - **bottleneck**: 现有工具基准覆盖窄、常依赖GPT裁判、且难以定位选错工具/填错参数/顺序错误/前序错误传播等细粒度失败来源
> - **mechanism_delta**: 将真实任务型对话映射成标准化工具调用轨迹，并把评测从“整体是否成功”拆成按回合、按工具集合、按顺序和按错误位置的自动指标
> - **evidence_signal**: 新指标与人工评价高相关，且在normal/hard/OOD多设置下能稳定拉开模型差距
> - **reusable_ops**: [dialogue-to-tool synthesis, turn-aware automatic scoring]
> - **failure_modes**: [伪工具与真实线上API语义不完全等价, 离线静态评测难覆盖权限/超时/副作用]
> - **open_questions**: [离线分数能否预测真实agent在线成功率, 如何评测工具执行失败后的恢复与重规划]

## Part I：问题与挑战

这篇论文的核心不是再提出一个新的 agent，而是回答一个更基础的问题：**我们到底有没有把“工具使用能力”测对**。

### 现有评测真正缺什么
作者指出，已有工具使用 benchmark 大多有四类缺口：

1. **场景不完整**  
   很多基准只覆盖单轮、单工具，无法代表真实助手场景中常见的“多轮澄清 + 多工具串联”。

2. **评测成本高且不稳定**  
   一些工作依赖 GPT 评委或人工打分，成本高、可重复性差，也不利于大规模迭代。

3. **只看结果，不看过程**  
   如果只看最终是否成功，就无法分辨：
   - 是不该调工具却调了；
   - 还是工具选错了；
   - 还是参数没从上下文中正确抽取；
   - 又或者前一轮错了，后面全被污染。

4. **数据与真实用户表达脱节**  
   纯从 API 文档反推指令，常常更像“API 说明书问答”，而不是用户真实会说的话。

### 真正的瓶颈
**真正的瓶颈是测量分辨率不足。**  
LLM 的工具使用不是一个单点能力，而是一串依赖链：是否调用工具 → 选哪个工具 → 参数是否 grounded → 多工具顺序是否正确 → 多轮错误是否传播。现有 benchmark 常常把这条链压成一个粗粒度分数，因此很难指导模型改进。

### 输入/输出接口
- **输入**：当前用户请求 + 历史对话 + 工具文档
- **模型输出**：`Thought`，以及 0 到多个 `Action / Action Input`
- **评测输出**：  
  - 单工具：TS、PS  
  - 多轮：SR、ATS、SATS、TPR  
  - 多工具：TN、TO  

### 为什么现在要解决
因为 LLM 正在从“会聊天”进入“会调用外部系统”的阶段。若没有低成本、细粒度、可复现的工具基准，模型能力会被误判，研究也会被粗糙指标带偏。

---

## Part II：方法与洞察

MTU-Bench 由两部分组成：

- **MTU-Instruct**：训练数据，用来提升模型工具使用能力
- **MTU-Eval**：评测集与指标体系，用来诊断能力边界

### 方法框架

#### 1. 从真实任务型对话出发，而不是从API文档倒推
作者收集了多个任务型对话数据集，如 MultiWOZ、SGD、TaskMaster、MetaLWOZ、ATIS、SNIPS。  
这些数据天然带有真实用户意图、槽位和值，更接近真实 assistant 使用场景。

#### 2. 把对话映射成“工具”
工具构造有两种方式：
- **Grammar-based**：已有 intent/slot 的数据，直接把 intent 变工具名、slot 变参数
- **LLM-based**：对无显式标注的数据，用 GPT-4 判断当前轮是否需要工具，以及需要什么伪工具

这样得到的不是自然语言答案，而是标准化的工具调用目标。

#### 3. 用工具聚类消除命名碎片
不同数据源里可能存在语义相同但名字不同的工具，如 `search_movie` 和 `find_movie`。  
作者用聚类把这类工具合并，减少评测噪声，报告中给出的压缩比例约为 **20:1**。

#### 4. 生成完整工具轨迹
作者进一步用 GPT-4 生成：
- `Thought`
- `Action`
- `Action Input`
- `Observation`
- 最终 assistant 回复

这样，每条样本不只是“答案对不对”，而是有完整的工具使用过程监督。

#### 5. 构造 normal / hard / OOD 三种测试压力
- **normal**：常规难度
- **hard**：长参数、易混工具、参数更新、无意义工具名、参数继承等复杂情况
- **OOD**：测试域与训练域不同，专门测泛化

#### 6. 用多粒度自动指标替代 GPT 裁判
作者将评测分解为三组：
- **单工具正确性**：工具是否选对、参数是否填对
- **多轮过程正确性**：整段是否成功、每轮是否成功、错误出现在多早、早错对后续伤害多大
- **多工具结构正确性**：工具集合是否对、调用顺序是否对

### 核心直觉

作者改变的关键旋钮不是模型架构，而是**评测坐标系**。

从因果链上看：

- **What changed**  
  从“用 GPT 裁判一个最终结果”  
  变成“给出显式工具轨迹 ground truth，再按工具名/参数/回合/顺序分别打分”。

- **Which bottleneck changed**  
  把原来**昂贵、主观、低分辨率**的测量方式，变成**便宜、可复现、过程敏感**的测量方式。

- **What capability changed**  
  benchmark 不再只告诉你“模型失败了”，而是能告诉你：
  - 它是**不会选工具**
  - 还是**不会抽参数**
  - 还是**多工具规划顺序错**
  - 还是**前一轮错误污染后续回合**

### 为什么这个设计有效
因为工具使用评测本质上需要两个东西：

1. **真实输入分布**：用户真的会怎么说  
   这由真实任务型对话提供。

2. **可判定的结构化目标**：什么才算正确调用  
   这由标准化工具文档和工具轨迹提供。

两者合起来，既保留 realism，又让评测变成自动可计算问题。

### 战略权衡

| 设计选择 | 解决的瓶颈 | 带来的收益 | 代价/风险 |
| --- | --- | --- | --- |
| 真实对话 → 工具轨迹 | 纯API合成不贴近真实用户表达 | 更接近真实 assistant 交互 | 仍依赖 GPT-4 合成，可能引入分布偏差 |
| 工具聚类标准化 | 同义工具名导致评测噪声 | 降低命名差异干扰 | 可能抹平部分细粒度 API 差异 |
| 无GPT自动评分 | 评测昂贵且主观 | 低成本、可复现、可大规模跑模型 | 依赖预定义 ground truth，开放式成功标准覆盖有限 |
| hard + OOD 测试 | 常规测试太“顺滑” | 能揭示鲁棒性与泛化边界 | 仍是离线环境，不等价于真实在线执行 |

---

## Part III：证据与局限

### 关键实验信号

- **信号1｜跨大量模型比较：多轮、多工具显著更难**  
  在 normal/hard 两类测试中，几乎所有模型在多轮和多工具设定下都明显掉分。  
  这说明 MTU-Bench 不是只在测“能不能输出一个 Action”，而是在测长期过程控制。

- **信号2｜新指标确实测到了人会在意的东西**  
  SATS、TN、TO 与人工评价的 Pearson 相关系数分别达到 **0.828 / 0.850 / 0.882**，而且接近人工彼此之间的一致性。  
  这说明这些指标不只是“公式漂亮”，而是真的能近似人类对工具使用质量的判断。

- **信号3｜MTU-Instruct 训练集有效**  
  基于 LLaMA3-8B-Instruct 微调得到的 MTU-LLaMA，在 normal、hard、OOD 多种设置中都明显强于原始底座；在 API-Bank 上的平均分甚至略高于 GPT-4。  
  这表明 MTU-Bench 不只是“能测”，也能“拿来训”。

- **信号4｜benchmark 能区分失败类型**  
  误差分析显示：
  - **Action Error** 最常见，尤其在 M-M 场景
  - **Format Error** 更多出现在较弱模型
  - 多轮场景中，错误经常早早出现，并向后传播  
  这正是细粒度 benchmark 的价值：能定位失败机制，而不只是报一个总分。

### 1-2 个最值得记住的指标
- **SATS**：比普通按轮平均更好，因为它会惩罚“前面出错导致后面看似对、实际已偏航”的情况。
- **TO**：多工具 agent 不只是“调了几个对的工具”，还要“顺序对”；这个指标专门测规划结构。

### 局限性
- **Fails when**: 真实线上 API 存在权限控制、延迟、超时、副作用、动态状态变化时，离线伪工具与模拟 observation 可能无法真实反映 agent 表现。
- **Assumes**: 数据构造和质检依赖 GPT-4；测试集与 hard case 依赖专家人工校验；评测假设工具文档固定、ground truth 明确、工具返回可被结构化模拟。
- **Not designed for**: 在线交互式 agent 的恢复策略、执行失败后的重规划、安全红队测试、真实金融/医疗等高风险外部操作。

### 可复用组件
- **对话 → 工具轨迹** 的自动转换流水线
- **多轮过程指标**：ATS / SATS / TPR
- **多工具结构指标**：TN / TO
- **困难样例 taxonomy**：易混工具、长参数、参数继承、无可用工具、多工具依赖链

总体看，这篇论文的能力跃迁不在于“提出了更强模型”，而在于**把工具使用评测从粗粒度结果判断，推进到过程级、结构级、可复现的诊断工具**。这对后续 agent 训练、基准构建和错误分析都很有价值。

![[paperPDFs/Agentic_Reasoning_Benchmarks_Core_Mechanisms_of_Agentic_Reasoning_Tool_Use_Multi_Turn_Tool_Use/ICLR_2025/2025_MTU_Bench_A_Multi_granularity_Tool_Use_Benchmark_for_Large_Language_Models.pdf]]