---
title: "T-Eval: Evaluating the Tool Utilization Capability of Large Language Models Step by Step"
venue: ACL
year: 2024
tags:
  - Survey_Benchmark
  - task/tool-use-evaluation
  - step-wise-evaluation
  - multi-agent-annotation
  - dataset/T-Eval
  - opensource/promised
core_operator: "将工具使用拆成六个能力维度，并用离线金标轨迹与专用评分协议逐步评估LLM的工具调用能力。"
primary_logic: |
  工具使用能力评测目标 → 以人审校验的金标解题轨迹构造计划/推理/检索/参数理解/指令生成/结果审查六类子任务，并设置string/JSON双难度协议 → 用专用评分器与端到端对照评估输出细粒度能力画像与能力边界
claims:
  - "T-Eval将工具使用分解为PLAN、REASON、RETRIEVE、UNDERSTAND、INSTRUCT、REVIEW六个能力维度，并构建了总计23,305个测试样例的细粒度评测集 [evidence: analysis]"
  - "在代表性开源模型上，T-Eval平均分与ToolBench式win-rate呈现相近排序趋势，说明其分步评分与结果导向评测在模型排序上基本一致 [evidence: comparison]"
  - "GPT-4在T-Eval上取得86.4 overall，最佳开源模型Qwen-72B为71.4；多数开源模型在JSON协议下相对string协议显著掉分，暴露出格式遵循、工具检索与结果审查的系统性短板 [evidence: analysis]"
related_work_position:
  extends: "ToolBench (Qin et al. 2023b)"
  competes_with: "ToolBench (Qin et al. 2023b); API-Bank (Li et al. 2023b)"
  complementary_to: "ReAct (Yao et al. 2022); Toolformer (Schick et al. 2023)"
evidence_strength: moderate
pdf_ref: paperPDFs/Agentic_Reasoning_Benchmarks_Core_Mechanisms_of_Agentic_Reasoning_Tool_Use_Single_Turn_Tool_Use/ACL_2024/2024_T_Eval_Evaluating_the_Tool_Utilization_Capability_of_Large_Language_Models_Step_by_Step.pdf
category: Survey_Benchmark
---

# T-Eval: Evaluating the Tool Utilization Capability of Large Language Models Step by Step

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2312.14033), [GitHub](https://github.com/open-compass/T-Eval)
> - **Summary**: 这篇工作把“LLM会不会用工具”拆成6个可单独测量的子能力，并通过离线金标轨迹评测，让我们能定位模型到底卡在计划、选工具、填参数，还是审查工具返回结果这一步。
> - **Key Performance**: GPT-4 在 T-Eval 上 overall 86.4；最佳开源模型 Qwen-72B 为 71.4，且 JSON 协议下多数开源模型明显掉分。

> [!info] **Agent Summary**
> - **task_path**: 用户查询 + 给定工具列表/文档 -> 分步工具使用能力评分 + 端到端能力诊断
> - **bottleneck**: 现有评测只看最终答案或单步调用，且受在线API波动干扰，无法稳定定位真实能力短板
> - **mechanism_delta**: 将工具使用拆为6个受控子过程，并基于离线金标轨迹分别评分，再用端到端结果做一致性校验
> - **evidence_signal**: T-Eval平均分与ToolBench win-rate趋势一致，同时显式揭示开源模型在JSON协议、retrieval和review上的短板
> - **reusable_ops**: [能力分解评测, 人审校验多agent标注]
> - **failure_modes**: [严格JSON下解析失败导致能力被低估, 单一金标路径难覆盖等价多解]
> - **open_questions**: [分步分数能否稳定预测真实在线代理成功率, 如何扩展到更长时域与更大动态工具库]

## Part I：问题与挑战

这篇论文要解决的核心不是“再做一个工具代理”，而是**怎么把工具代理的失败原因测清楚**。

### 现有评测缺什么
现有工具使用评测主要有两类问题：

1. **只看最终结果**  
   如果模型最后没答对，你不知道它是：
   - 计划错了，
   - 推理错了，
   - 选错工具了，
   - 参数填错了，
   - 格式不合法，
   - 还是工具返回后不会判断“这步已经成功/失败”。

2. **依赖实时工具交互**  
   在线 API 会有服务不稳定、时序漂移、返回内容变化等外生噪声。  
   这会让评测结果混入“工具环境的不确定性”，导致不同模型比较不公平。

### 真正的瓶颈是什么
真正瓶颈是：**工具使用是一个高耦合链式过程，但现有评测把它压成一个最终分数**。  
因此研究者很难回答最关键的问题：**模型到底不会哪一步，为什么不会。**

### 输入/输出接口
T-Eval 的评测接口本质上是：

- **输入**：工具列表 `T`、用户查询 `q`，以及某些子任务下的解题前缀
- **输出**：
  - 六个子能力分数：PLAN / REASON / RETRIEVE / UNDERSTAND / INSTRUCT / REVIEW
  - 一个端到端整体趋势对照结果

### 边界条件
T-Eval的适用边界也很明确：

- 面向**文本型工具使用**，不是多模态工具代理
- 假设**工具列表与工具文档已经给定**
- 主要评测**in-context tool use**，不是把工具内化进参数的训练范式
- 为了稳定性，尽量转向**离线金标轨迹**，而非真实线上环境

---

## Part II：方法与洞察

### T-Eval做了什么
T-Eval把完整工具使用链路拆成6个必要能力：

| 能力 | 测什么 | 典型输入 | 评分思路 |
| --- | --- | --- | --- |
| PLAN | 先规划要调用哪些动作 | 工具列表 + 查询 | 比较预测动作序列与金标序列的一致性与顺序性 |
| REASON | 当前步该怎么想 | 工具列表 + 查询 + 解题前缀 | 比较下一步 thought 与金标 thought 的语义相似度 |
| RETRIEVE | 当前该选哪个工具 | 工具列表 + 查询 + 解题前缀 | 比较工具名是否选对 |
| UNDERSTAND | 当前工具参数该怎么填 | 工具列表 + 查询 + 解题前缀 | 比较参数与金标参数的语义相似度 |
| INSTRUCT | 能否按协议生成可执行调用 | 工具名 + 参数 | 检查格式合法性与参数正确率 |
| REVIEW | 能否判断工具返回是否完成目标 | thought + tool response | 五分类判断成功/失败类型 |

其中最关键的设计点有三个。

### 1. 把“整体失败”改成“分步诊断”
T-Eval不再直接问“最后答对了吗”，而是控制住中间条件，分别测：

- 会不会规划
- 会不会想下一步
- 会不会选工具
- 会不会填参数
- 会不会按协议输出
- 会不会检查工具返回

这使得错误从“黑箱失败”变成“可定位失败”。

### 2. 用离线金标轨迹替代实时API噪声
作者通过人审校验的 solution path，把每一步的 thought / action / observation / review 固定下来。  
这样评测时不再强依赖实时工具状态，从而减少：

- API不稳定
- 时间变化导致的信息漂移
- 外部环境差异带来的随机性

换句话说，T-Eval改变的是**测量噪声来源**：  
从“模型能力 + 工具环境噪声”的混合观测，变成更接近“模型本身子能力”的观测。

### 3. 用 string / JSON 双难度协议分离“语义能力”和“协议能力”
很多开源模型不是完全不会解题，而是**不会按严格 JSON 协议输出**。  
如果只用严格格式评测，弱模型会被“解析失败”直接清零，掩盖真实能力。

所以作者设计了两层难度：

- **string**：更宽松，偏语义能力
- **JSON**：更严格，更接近真实产品协议

这让 benchmark 不只告诉你“做没做对”，还告诉你：  
**你是不会解决问题，还是不会按系统协议把答案表达出来。**

### 核心直觉

传统工具评测把“工具使用”当成一个整体结果变量，因此无法判断错误是由哪一个中间瓶颈导致。  
T-Eval 的关键改变是：**把一个高耦合链式分布拆成6个低耦合、可控输入条件下的局部判别任务。**

因果上看，变化链条是：

- **评测对象变化**：从最终答案 → 中间能力节点
- **约束变化**：从在线环境噪声混入 → 离线金标前缀控制
- **信息瓶颈变化**：从“只观测成败” → “观测每一步能力是否断裂”
- **能力变化**：从粗粒度排序 → 可诊断的能力画像与短板定位

这套设计之所以有效，不是因为它“更细”，而是因为它**把错误归因从事后猜测变成了受控测量**。

### 数据与标注流水线
T-Eval的数据构建也围绕“稳定诊断”展开：

1. **工具集合**：15个工具，覆盖 Research、Travel、Entertainment、Web、Life、Financials 等域
2. **指令生成**：随机抽 2~3 个工具，先用 GPT-3.5 生成 query，再由 GPT-4 精修
3. **金标轨迹标注**：用 planner / executor / reviewer 三角色多agent流程生成解题链
4. **人类复核**：人工审查并筛掉低质量样本
5. **子集抽取**：形成 23,305 个测试样例

这里多agent标注的意义是：  
把“一个模型同时兼任规划、执行、审查”的复杂任务，拆成角色明确的子职责，以降低标注过程自身的错误率。

### 策略权衡

| 设计选择 | 带来的好处 | 代价/风险 |
| --- | --- | --- |
| 分步评测替代只看最终结果 | 能定位真实短板 | 会弱化步骤间误差传播的整体效应 |
| 离线金标轨迹替代实时API | 更稳定、更公平、可复现 | 与真实部署环境存在偏差 |
| string + JSON 双协议 | 区分“会解题”和“会按协议输出” | 产品真实度被分成两层，不是单一结论 |
| 多agent + 人审标注 | 规模化且质量较高 | 依赖商用模型与人工审核成本 |
| 手工增强API文档 | 减少因文档差导致的假失败 | 相比真实世界文档更理想化 |

---

## Part III：证据与局限

### 关键证据信号

**信号1：T-Eval能拉开模型层级，而且商用模型明显领先。**  
GPT-4 在 overall 上达到 **86.4**，最佳开源模型 Qwen-72B 为 **71.4**。  
这说明 benchmark 既有区分度，也揭示了当前开源工具代理与顶级闭源模型仍有明显差距。

**信号2：格式遵循是开源模型的重要断点。**  
论文反复强调 JSON 协议下开源模型掉分严重。  
这不是简单的“会不会输出花括号”，而是说明很多模型无法在**受协议约束的条件下继续完成规划、推理、检索和参数生成**。  
例如 Qwen-72B 的 UNDERSTAND 在 string 下可到 **84.5**，但 JSON 下是 **66.1**，说明“语义理解”与“协议化执行”之间仍有明显缺口。

**信号3：retrieval 和 review 是最难的两步。**  
作者发现：
- **RETRIEVE**：多数模型选工具仍不稳，Qwen-72B 在 JSON 下也只有 **65.0**
- **REVIEW**：很多模型不会判断工具返回究竟是成功、输入错、工具内部错，还是根本无法完成；GPT-4 的 REVIEW 达 **94.5**，而大量模型仅在 50%~60% 区间

这很关键，因为真实 agent 不只是“会调用”，还得**会看调用结果是否可用**。

**信号4：T-Eval与结果导向评测趋势一致，但更可解释。**  
作者把 T-Eval 平均分与 ToolBench 的 win-rate 做对照，发现模型排序趋势相近。  
这说明 T-Eval 没有脱离“真实任务效果”，同时它又提供了端到端 win-rate 给不出的细分诊断。

### 能力跃迁到底体现在哪
相较于 prior tool benchmarks，T-Eval 的能力跃迁不在“更难”，而在“更能解释”：

- 之前：知道模型输了，但不知道输在哪一步
- 现在：能区分是**计划瓶颈、协议瓶颈、检索瓶颈、参数理解瓶颈还是反馈审查瓶颈**

这对后续训练非常重要，因为它直接对应不同改进方向：

- instruction tuning 改格式跟随
- agent data 改 tool retrieval / review
- planning data 改多步策略
- tool docs / schema 对齐改 parameter grounding

### 局限性

- **Fails when**: 面对真实在线API的时序变化、返回抖动、权限问题、超长工具链，或存在多条等价解路径时，离线单金标分步评测可能无法完整反映真实代理表现。
- **Assumes**: 假设工具列表和高质量文档已给定；依赖GPT-3.5/GPT-4参与数据生成、精修或对照评测；依赖人工专家复核；部分评分依赖Sentence-BERT式语义相似度与预定义错误类别，因此构建和复现实验都有额外成本。
- **Not designed for**: 多模态工具、开放世界动态工具发现、真实生产中的延迟/费用/并发/权限控制/安全策略评估，也不是为了评估“工具环境本身是否可靠”。

### 可复用组件

1. **六维能力分解框架**：适合迁移到更广义的 agent benchmark
2. **离线金标轨迹评测协议**：适合降低环境噪声
3. **string/JSON 双难度设计**：适合把“语义能力”和“协议能力”分开测
4. **planner/executor/reviewer 多agent标注流水线**：适合构建复杂过程型 benchmark
5. **顺序敏感的 plan scorer**：对多步动作规划评测很有参考价值

## Local PDF reference

![[paperPDFs/Agentic_Reasoning_Benchmarks_Core_Mechanisms_of_Agentic_Reasoning_Tool_Use_Single_Turn_Tool_Use/ACL_2024/2024_T_Eval_Evaluating_the_Tool_Utilization_Capability_of_Large_Language_Models_Step_by_Step.pdf]]