---
title: "TurnBench-MS: A Benchmark for Evaluating Multi-Turn, Multi-Step Reasoning in Large Language Models"
venue: arXiv
year: 2025
tags:
  - Survey_Benchmark
  - task/MLLM-evaluation
  - game-based-evaluation
  - rule-based-evaluation
  - llm-as-judge
  - dataset/TurnBench-MS
  - opensource/full
core_operator: 通过隐藏规则的交互式破译游戏，同时评分最终答案与每轮对验证器激活规则的推断质量
primary_logic: |
  多轮推理评测目标 → 构造带隐藏激活规则与反馈循环的 Turing Machine 式游戏（Classic/Nightmare） → 从模型多轮推理中抽取对 verifier 的 HAC 推断并与真值比对 → 输出终局正确率、过程正确性与错误恢复轨迹
claims:
  - "在 Classic 全量 270 题上，gpt-o4-mini-high 配合 CoT 达到 81.5% 准确率，但仍低于人类最佳 100% [evidence: comparison]"
  - "在 Nightmare 45 题评测中，最佳模型 Gemini-2.5-Flash 仅达到 18% 准确率，而人类平均达到 94.2% [evidence: comparison]"
  - "TurnBench 的自动过程评估在 5% 分层人工抽检中显示：推断提取器 precision 为 99.7%，Judger 分类准确率为 99.4% [evidence: analysis]"
related_work_position:
  extends: "MastermindEval (Golde et al. 2025)"
  competes_with: "MastermindEval (Golde et al. 2025); LMAct (Ruoss et al. 2025)"
  complementary_to: "Self-Refine (Madaan et al. 2023); ReFlexion (Shinn et al. 2023)"
evidence_strength: moderate
pdf_ref: paperPDFs/Agentic_Reasoning_Benchmarks_Core_Mechanisms_of_Agentic_Reasoning_Memory_and_Planning_Long_Horizon_Episodic_Memory/arXiv_2025/2025_TurnBench_MS_A_Benchmark_for_Evaluating_Multi_Turn_Multi_Step_Reasoning_in_Large_Language_Models.pdf
category: Survey_Benchmark
---

# TurnBench-MS: A Benchmark for Evaluating Multi-Turn, Multi-Step Reasoning in Large Language Models

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2506.01341), [Code/Data](https://github.com/grantzyr/TurnBench-MS)
> - **Summary**: 这篇工作把静态逻辑题改造成带隐藏规则、反馈循环和中间真值的交互式破译游戏，用来专门测 LLM 在多轮假设更新、记忆保持和错误恢复上的真实能力。
> - **Key Performance**: Classic 全量 270 题上最佳 LLM 为 81.5%；Nightmare 45 题上最佳 LLM 仅 18%，而人类最佳两种模式均为 100%。

> [!info] **Agent Summary**
> - **task_path**: 文本化游戏规则与多轮 PASS/FAIL 反馈 -> 隐藏规则/隐藏映射推断 -> 最终三位密码提交
> - **bottleneck**: 现有评测大多只看单轮终答，测不到跨轮记忆、实验设计、假设修正和出错后的自纠错
> - **mechanism_delta**: 用 Turing Machine 风格交互游戏替代静态题，并用 HAC 真值对模型的中间推断做自动评分
> - **evidence_signal**: Nightmare 模式最佳模型仅 18%，且模型首次出错后错误持续率高达 53%–99%
> - **reusable_ops**: [HAC中间真值评分, FIC错误路径追踪]
> - **failure_modes**: [初次错误后持续偏航, 长程多轮中的信息幻觉与记忆衰减]
> - **open_questions**: [如何不依赖闭源judge稳定评估中间推理, 该类封闭规则游戏对真实agent任务的外部效度有多强]

## Part I：问题与挑战

这篇论文真正要解决的，不是“模型会不会做一道逻辑题”，而是：

**模型能否在反馈闭环里持续做实验、更新假设、整合历史线索，并在犯错后把自己拉回来。**

### 现有评测缺了什么

作者指出三类核心缺口：

1. **单轮化**  
   许多推理 benchmark 本质上还是单次作答。即使题目需要多步推理，模型也常常是在一次输出里完成，而不是在多轮交互中逐步试探、修正和确认。

2. **只看终局，不看过程**  
   只用 final answer 打分，无法区分：
   - 真正通过逐步推理得到答案；
   - 靠启发式、猜测甚至偶然撞对。

3. **污染风险**  
   静态题库容易与预训练语料重叠，导致高分未必代表真实推理能力。

### 为什么现在值得做

因为 LLM 正越来越多地被放到 **agentic / interactive** 场景中：要反复提问、利用反馈、跨轮保持一致性。  
这时真正的瓶颈不再是“一次性生成一个像样答案”，而是：

- 能不能 **设计下一步测试**
- 能不能 **利用上一轮反馈修正当前假设**
- 能不能 **跨轮保持状态**
- 能不能 **从错误中恢复**

TurnBench 的价值就在于，它把这些能力显式暴露出来。

### 输入 / 输出接口

**输入：**
- 当前游戏规则
- 4–6 个 verifier 的公开描述
- 每轮自选的 3 位数字 proposal
- 对最多 3 个 verifier 的 PASS/FAIL 反馈
- 历史轮次上下文

**输出：**
- 下一轮测试 code
- 选择查询哪个 verifier / 是否跳过
- 最终提交的三位密码
- 在 CoT 模式下，还要显式写出中间推理

### 边界条件

这个 benchmark 很克制，故意把问题限定在一个**无外部知识**的封闭数字逻辑环境里：

- 3 位密码，每位取值 1–5，可重复
- 每个 verifier 有多个可能规则，但只有一个 **Hidden Active Criterion (HAC)**
- 每轮最多查 3 个 verifier
- **Classic**：查询哪个 verifier，就返回哪个 verifier 的反馈
- **Nightmare**：反馈来自一个隐藏重映射后的 verifier，模型还得先推断映射关系

也就是说，它重点测的是 **交互式逻辑推理**，不是世界知识或开放域问答。

---

## Part II：方法与洞察

### 评测设计：把“答题”改成“做实验”

TurnBench 的核心设计非常清晰：  
把模型从“直接给答案”的模式，转成“提出假设—做测试—接收反馈—修正假设”的模式。

每个 episode 的流程是：

1. **Proposal**：提出一个三位 code
2. **Question**：选择最多 3 个 verifier 查询，得到 PASS/FAIL
3. **Deduce**：决定直接提交最终答案，或进入下一轮
4. **Continue / End**

这使得模型必须主动决定：

- 当前这一轮测什么最有信息量
- 哪个 verifier 值得查
- 现有证据够不够提交
- 如果不够，下一轮该怎么设计更区分性的测试

### 两种模式分别在测什么

#### 1) Classic：测规则发现与跨轮整合
Classic 模式的难点是：  
模型需要根据不同轮次的 PASS/FAIL，逐步锁定每个 verifier 的 HAC，再综合得到唯一 code。

它主要考：
- 多轮假设更新
- 逻辑排除
- 历史信息整合
- 终局提交时机

#### 2) Nightmare：在此基础上再加一个隐变量
Nightmare 模式不是简单“更难”，而是**换了难点结构**：

模型看到的 verifier 编号和真正执行判断的 verifier 是错位的，而且这个错位映射在整局固定但未知。  
所以模型不再只是推断规则，而是要做 **规则推断 + 映射推断** 的联合推理。

这会显著放大：
- 状态跟踪负担
- 假设空间规模
- 错误传播风险

### 过程级评估：不只判终局，还判中间结论

这是论文最有方法论价值的地方。

作者没有停留在“谁答对了多少题”，而是设计了一个**自动化过程评估管线**：

1. 从模型 CoT 中抽取它对每个 verifier HAC 的显式结论  
2. 从游戏 metadata 中读取该 verifier 的真值 HAC
3. 用 Judger 比较两者，分成：
   - **Correct**
   - **Incorrect**
   - **Include**（包含正确方向但还不够精确）

这一步把原本难量化的“推理过程”，转成了可打标签、可统计的中间状态。

作者还做了人工抽检验证：
- 抽取器 precision 99.7%
- Judger 分类准确率 99.4%
- 但抽取器会漏掉约 13.7% 可用结论

这说明它**足够可用，但不是完美无偏**。

### 核心直觉

TurnBench 真正改变的不是题面，而是**测量瓶颈**。

过去的 benchmark：
- 把推理压扁成一次输出
- 最后只看答案
- 因此看不到模型是“推出来的”还是“蒙出来的”

TurnBench 改成：
- **隐藏规则 + 二值反馈**：迫使模型通过多轮对照实验逐渐缩小假设空间
- **Classic / Nightmare 分层**：把“规则发现”与“规则+映射联合发现”拆开测
- **中间真值评分**：把不可见的 reasoning path 变成可诊断对象

因果链可以概括为：

**从静态终答评测**  
→ **变成带反馈的主动试验过程**  
→ **推理不再只是生成文本，而是管理一个随时间更新的假设状态**  
→ **benchmark 因而能暴露模型的记忆、纠错、稳定性与策略质量**

这也是它相比普通逻辑题 benchmark 的能力跃迁点。

### 战略权衡

| 设计选择 | 改变了什么瓶颈 | 得到什么能力诊断 | 代价 |
|---|---|---|---|
| 隐藏 HAC 的 verifier 设计 | 从“直接答题”变成“实验式排查” | 可测假设更新与信息利用 | 任务更合成，外部效度有限 |
| 多轮 Proposal/Question/Deduce | 从单次生成变成跨轮状态管理 | 可测长期一致性与记忆 | 评测成本更高，交互更长 |
| Nightmare 隐藏映射 | 增加潜在状态推断 | 可测联合推理与鲁棒性 | 难度陡增，解释更复杂 |
| CoT 抽取 + LLM Judger | 从终局指标扩展到过程指标 | 可分析首次错误后的路径分叉 | 依赖闭源 judge，且抽取存在漏检 |
| 动态规则配置 | 降低训练集污染可能 | 更可信地测真实推理 | 可比性与标准化管理更复杂 |

---

## Part III：证据与局限

### 关键证据信号

#### 信号 1：Classic 上已有明显人机差距，但模型还没完全崩
在 Classic 全量 270 题上，最佳结果是：

- **gpt-o4-mini-high + CoT：81.5%**
- **人类最佳：100%**

这说明当前强模型已经能在封闭规则环境里做一部分多轮推理，但离稳定、接近人的水平还有明显差距。

更重要的是，人类往往**不是更省轮次**，而是**更稳**。  
模型有时会用更少轮次结束，但这是建立在更高脆弱性上的“快”，不是更可靠的“强”。

#### 信号 2：Nightmare 让模型几乎整体失效
Nightmare 模式下：

- **最佳模型 Gemini-2.5-Flash：18%**
- **gpt-o4-mini-high：11%**
- **人类平均：94.2%**
- **人类最佳：100%**

这说明一旦问题从“推规则”升级到“推规则 + 推映射”，当前 LLM 的稳定推理能力迅速坍塌。  
所以瓶颈不只是数学逻辑本身，而是：

- 隐状态管理
- 跨轮绑定
- 联合假设空间控制

#### 信号 3：一旦第一次推错，模型通常很难回来
这是论文最有诊断价值的发现。

作者追踪 **First Incorrect Conclusion (FIC)** 后的路径，发现：

- 初始 verifier 错误后的**持续率高达 53%–99%**
- 很多模型在出错后，直接不再 revisiting 那条推理链
- 条件错误持续概率随着回合数增加迅速上升，到第 5 个相对回合接近 100%

这说明模型不是“偶尔算错一步”，而是会进入一种**错误吸附态**：  
一旦偏航，就越来越难自我纠偏。

#### 信号 4：CoT 和规模都有帮助，但都不是根治
作者还发现：

- CoT 明显提升 Classic 成绩
- 更大模型普遍更好
- 7B–8B 小模型几乎接近零分
- 24B/70B 后有明显提升
- 大型 MoE（如 DeepSeek-R1）更强，但仍远低于人类

这说明当前问题不是“模型没见过这类题”，而是**交互式多轮推理本身仍然是容量、记忆和策略控制的硬瓶颈**。

### So what：这篇工作真正带来的能力增量

它最重要的贡献不是又加了一个排行榜，而是让我们第一次比较系统地看到：

- **模型在哪种交互结构下会掉队**
- **错误是怎么开始、怎么扩散、为什么回不来**
- **最终答对与真实推理质量并不等价**

相比只给 final accuracy 的 benchmark，TurnBench 更像一个**推理失效诊断台**。

### 局限性

- **Fails when**: 想把它的分数直接外推到开放域知识推理、真实工具使用、现实规划或多模态 agent 任务时；因为该 benchmark 本质上是封闭数字逻辑游戏，反馈也只有 PASS/FAIL。
- **Assumes**: 模型能输出可解析的显式 reasoning；过程评估依赖 Gemini-2.5-Flash 做 inference extraction 和 judging；Nightmare 主实验只报告了 45 个样例，压力测试结论明确但覆盖面比 Classic 全量实验更窄。
- **Not designed for**: 测试世界知识、创造性生成、真实环境交互、复杂语言歧义处理，或无需明确中间真值的自然语言任务。

另外，虽然代码和数据开源，但**过程评价链条仍有闭源 API 依赖**，这会影响完全离线复现与长期稳定复测。

### 可复用组件

这篇工作里最值得迁移到别的 benchmark/agent 评测中的部分有：

- **交互式 hidden-rule 任务模板**：把一次性答题改成反馈闭环
- **HAC 真值对齐评分**：不仅看 final answer，也看中间假设是否正确
- **FIC 错误路径分析**：首次错误 → 后续状态 → 最终输赢
- **严格阶段式协议 + 格式重试机制**：降低“格式错误”对推理测评的污染

如果以后要做更真实的 agent benchmark，可以把这些机制迁移到：
- 工具调用
- 长程任务规划
- 交互式调试
- 多轮检索与证据整合

![[paperPDFs/Agentic_Reasoning_Benchmarks_Core_Mechanisms_of_Agentic_Reasoning_Memory_and_Planning_Long_Horizon_Episodic_Memory/arXiv_2025/2025_TurnBench_MS_A_Benchmark_for_Evaluating_Multi_Turn_Multi_Step_Reasoning_in_Large_Language_Models.pdf]]