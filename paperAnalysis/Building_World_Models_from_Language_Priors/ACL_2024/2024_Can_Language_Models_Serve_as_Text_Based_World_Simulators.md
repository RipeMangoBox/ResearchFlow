---
title: "Can Language Models Serve as Text-Based World Simulators?"
venue: ACL
year: 2024
tags:
  - Survey_Benchmark
  - task/llm-evaluation
  - task/text-game-simulation
  - json-scaffolding
  - single-step-prediction
  - state-difference-prediction
  - dataset/BYTESIZED32-SP
  - opensource/full
core_operator: 以 JSON 单步状态转移基准把文本世界模拟拆成动作驱动、环境驱动和游戏进度三部分，直接诊断 LLM 的世界模型能力。
primary_logic: |
  评测“LLM 能否直接充当文本世界模拟器” → 基于 BYTESIZED32 构建包含 31 个文本游戏、76,369 条状态转移的 BYTESIZED32-SP，并提供人工规则、LLM 规则与无规则三种上下文 → 用全状态/状态差分两种单步预测分别评测整体转移、动作驱动转移、环境驱动转移和游戏进度 → 揭示当前 LLM 在环境驱动及依赖算术、常识、科学知识的转移上仍不可靠
claims:
  - "在 BYTESIZED32-SP 上，GPT-4 在发生真实状态变化的完整单步状态转移预测中的最佳平均准确率仅为 59.9%，尚不足以作为可靠的文本世界模拟器 [evidence: comparison]"
  - "动态环境驱动转移显著难于动态动作驱动转移：GPT-4 的最佳平均准确率分别为 49.7% 与 77.1% [evidence: analysis]"
  - "在游戏进度预测 FR 上，GPT-4 在带 LLM 生成规则的上下文下达到 92.1%，无规则时仅 61.5% [evidence: comparison]"
related_work_position:
  extends: "BYTESIZED32 (Wang et al. 2023)"
  competes_with: "N/A"
  complementary_to: "RAP (Hao et al. 2023); WorldCoder (Tang et al. 2024)"
evidence_strength: moderate
pdf_ref: "paperPDFs/Building_World_Models_from_Language_Priors/ACL_2024/2024_Can_Language_Models_Serve_as_Text_Based_World_Simulators.pdf"
category: Survey_Benchmark
---

# Can Language Models Serve as Text-Based World Simulators?

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2406.06485), [Code/Data](https://github.com/cognitiveailab/GPT-simulator)
> - **Summary**: 本文提出 BYTESIZED32-SP 与 LLM-Sim，把“LLM 能否直接充当文本世界模拟器”转化为可精确计分的单步状态转移评测，并发现即使是 GPT-4 也远未达到可靠模拟器水平。
> - **Key Performance**: GPT-4 在动态完整状态转移上的最佳平均准确率仅 59.9%；动态环境驱动转移最佳仅 49.7%，而带规则的游戏进度预测可达 92.1%。

> [!info] **Agent Summary**
> - **task_path**: 文本游戏任务描述/规则 + 当前 JSON 状态 + 动作 -> 下一步 JSON 状态（或状态差分） + reward/gameOver/gameWon
> - **bottleneck**: 模型难以稳定模拟与动作不直接对应的环境驱动更新，尤其是涉及算术、常识或科学知识的属性变化
> - **mechanism_delta**: 将整体世界模拟拆成 Fact/Fenv/FR 三个单步子任务，并用 full-state 与 state-diff 两种输出格式分离“格式问题”和“状态转移建模问题”
> - **evidence_signal**: GPT-4 动态 Fact 最好 77.1%，但动态 Fenv 最好仅 49.7%，整体动态 F 最好 59.9%
> - **reusable_ops**: [Fact-Fenv-FR 分解, JSON 状态差分评测]
> - **failure_modes**: [环境 tick 更新遗漏, 算术/常识/科学知识相关属性赋值错误]
> - **open_questions**: [如何降低多步 rollout 的误差累积, 是否存在比 JSON full/diff 更适合 LLM 的状态表示]

## Part I：问题与挑战

这篇论文真正要回答的问题不是“LLM 会不会玩文本游戏”，而是更基础的一层：**LLM 能不能像游戏引擎那样，忠实地执行世界状态转移**。

### 真问题是什么
过去不少工作把 LLM 用作：
- 代码/符号世界模型生成器；
- 规划器的一部分；
- 或直接生成交互世界的文本引擎。

但这些设置通常把多个因素混在一起：规划、搜索、提示、观测、记忆、状态更新。作者指出，**如果下一状态本身都不可靠，那么上层规划再强，也是在错误世界上推理**。所以真正瓶颈是：  
**LLM 是否学会了 transition function，而不只是会说“看起来合理的话”。**

### 为什么现在要解决
- 手工构建虚拟环境成本高，通常需要专家写大量规则。
- 现在 LLM 的常识与语言能力已经足够强，很多人自然会问：能不能直接拿来当 simulator？
- 但在这之前，缺少一个**直接、可量化、可诊断**的评测来回答这个问题。

### 输入 / 输出接口
论文把任务定义成一个标准的文本世界单步模拟问题：

- **输入**：任务上下文（目标、对象规则、动作规则、计分规则、示例） + 当前 JSON 状态 + 当前动作
- **输出**：下一步 JSON 状态，外加 reward / gameOver / gameWon

其中作者进一步把整体模拟拆成三部分：
- **Fact**：动作直接导致的状态变化
- **Fenv**：环境自身 tick 导致的变化
- **FR**：游戏进度变化（分数、结束与否、是否获胜）

### 边界条件
这点非常重要：该基准**不是**在测完整 agent 能力，而是在测局部状态转移能力。

- 只做**单步预测**
- 提供**ground-truth 的前一状态**
- 主要是**文本游戏**，且任务集中在常识与初等科学推理
- 数据主要来自**gold trajectory 附近一步内的有效动作**

因此它测到的是：**局部世界模型 fidelity**，而不是长程规划、部分可观测 belief tracking 或开放域物理模拟。

---

## Part II：方法与洞察

### 方法骨架

作者提出了一个评测任务 **LLM-Sim**，并配套构建数据集 **BYTESIZED32-SP**。

#### 1. 数据集怎么来
- 来源于 BYTESIZED32 文本游戏语料
- 原始 32 个游戏中，留出 1 个做 in-context 示例
- 最终得到 **31 个游戏、76,369 条状态转移**
- 每条样本包含：上下文、当前状态、动作、中间动作后状态、最终状态、分数与终止信息

为了让评测更可控，作者又构造了一个平衡测试集：
- 按 action verb 采样
- 区分 **dynamic**（真的发生变化）与 **static**（不发生变化）
- 得到 **2,954 条**实验样本

#### 2. 两种输出形式
- **Full State Prediction**：直接输出完整下一状态
- **State Difference Prediction**：只输出变化部分

这个设计很关键，因为它在测试一个现实问题：  
**LLM 的错误到底来自“不会模拟变化”，还是只是“复制完整状态时容易出格式/冗余错误”？**

#### 3. 三种上下文设置
- **Human-written rules**
- **LLM-generated rules**
- **No rules**

这使得作者能够区分：
- 模型到底是在靠预训练常识硬猜；
- 还是能利用明确规则完成模拟。

#### 4. JSON 作为评测支架
作者统一使用 JSON 表示状态：
- 更接近 LLM 熟悉的结构化文本分布
- 便于做精确对比
- 可以追踪到具体对象、属性和关系层面的错误

### 核心直觉

**关键变化**：把“LLM 作为世界模型是否有用”这个模糊问题，改写成“给定真实前态时，它能否准确生成单步后态”这个可测问题。

这一步改变了什么？

1. **去掉了混杂因素**  
   以前 end-to-end 成败会混入规划、搜索、观察不全、提示模板等因素。  
   现在直接固定前态，只测状态更新本身。

2. **把失败定位到具体因果环节**  
   通过 Fact / Fenv / FR 分解，可以区分：
   - 模型是不会执行显式动作规则；
   - 还是不会追踪环境自身动态；
   - 还是不会判断任务完成与奖励变化。

3. **把“合理文本”变成“可验证模拟”**  
   使用结构化状态与精确标签后，模型必须给出和引擎一致的世界更新，而不能只生成表面合理的自然语言。

换句话说，这篇论文的最大贡献不只是“做了个 benchmark”，而是**把 world simulation 的测量瓶颈从笼统任务成败，改成了可诊断的局部转移正确性**。

### 为什么这个设计有效
因果上看，这个 benchmark 能工作，是因为它把问题压缩到了最核心的单元：

**前态 + 动作 + 规则 → 后态**

一旦在这个单位操作上都不稳定，那么：
- 多步 rollout 一定更差；
- 规划器会建立在错误模型上；
- 自由生成式 simulator 会更难可靠。

### 战略取舍

| 设计选择 | 改变了什么 | 收益 | 代价 / 副作用 |
|---|---|---|---|
| 单步预测而非多步 rollout | 把误差累积与转移建模分开 | 能直接测局部 simulator fidelity | 低估真实部署中的长期崩溃 |
| Fact / Fenv / FR 分解 | 把动作效应、环境动态、进度判断拆开 | 能定位真正瓶颈是否在环境动态 | 需要额外中间状态标注 |
| Full state vs state diff | 区分“复制全部状态”与“只写变化”两类难度 | 能分析输出冗余是否是主因 | diff 对动态样本反而可能更复杂 |
| Human / LLM / No rules | 检查模型依赖显式规则还是先验知识 | 能测规则价值，也能测试自动规则生成是否可用 | 规则质量本身成为变量，且 LLM 规则需人工检查 |

---

## Part III：证据与局限

### 关键实验信号

#### 信号 1：整体上，GPT-4 仍不是可靠 simulator
最关键结果是：  
**GPT-4 在动态完整状态转移上的最佳平均准确率只有 59.9%。**

这说明它在“世界真的发生变化”时，仍然经常写错后态。  
这对 simulator 来说是致命问题，因为模拟误差会在多步 rollout 中快速累积。作者也明确指出，这种单步精度不足以支撑实际长期模拟。

#### 信号 2：真正短板在环境驱动转移，不在动作直接效果
把整体任务拆开后，结论非常清楚：

- **动态 Fact**：最佳 77.1%
- **动态 Fenv**：最佳仅 49.7%

也就是说，模型对“turn on the sink 之后 isOn 变成 true”这类**显式动作结果**相对还能处理；  
但对“水龙头开着后，杯子在下一 tick 被装满水”这类**隐式环境后果**明显更弱。

这回答了论文的核心问题：  
**LLM 目前最不像 simulator 的地方，不是动作语义解析，而是持续环境机制建模。**

#### 信号 3：规则确实有用，但 state diff 不是普适增益
- 对 **FR**，带规则时 GPT-4 可达 **92.1%**，无规则只有 **61.5%**
- 对状态预测，显式规则整体上也有帮助
- 但 **state difference prediction** 只在 static 样本上明显更有利；到了 dynamic 样本，反而常常不如 full-state

这说明：
- 显式程序性知识对 world simulation 很关键；
- 但“只输出变化”并不自动等于更容易，尤其当变化本身复杂时，diff 会把推理和格式要求一起变难。

#### 信号 4：这不是“任务本来就没人会”的问题
作者做了一个小规模 human study，在 GPT-4 最差的 5 个游戏上比较：

- **人类平均准确率：80%**
- **GPT-4：49%**

虽然这不是严格的大规模人类上限实验，但至少说明：  
**任务对人类并不离谱，模型确实还有明显差距。**

#### 信号 5：错误不是随机的，而是结构性的
属性级分析显示，错误主要集中在：
- **算术相关**：temperature, timeAboveMaxTemp
- **常识相关**：current_aperture, current_focus
- **科学知识相关**：如电路/光照等属性

而且在整体一步预测中，GPT-4 常常更关注动作直接效果，忽略环境更新，出现大量 **unaltered value** 错误。  
这进一步证明问题不是简单的 JSON 格式错误，而是**机制级更新没学稳**。

### 局限性
- **Fails when**: 需要连续 tick 更新、隐式环境因果、数值累积、相机/电路/温度等常识或科学知识推理，或需要多步 rollout 的场景。
- **Assumes**: 提供 ground-truth 前一状态、单步有效动作、JSON schema，以及围绕 gold trajectory 一步邻域采集的数据；主要实验依赖闭源 GPT-4/GPT-3.5 API 与 JSON mode，API 成本约 \$5,000；LLM 生成规则还经过人工检查。
- **Not designed for**: 长程规划、部分可观测 belief tracking、开放域物理/医疗模拟、像素级或具身环境模拟。

补充地说，这个 benchmark 虽然很适合诊断“单步世界更新”，但它**不能直接代表真实部署中的长期仿真质量**。另外，人类对比实验只覆盖 5 个游戏且标注者为作者，因此更适合看作辅助信号，而非严格 human upper bound。

### 可复用组件
这篇工作最值得复用的不是 GPT-4 分数，而是它的评测骨架：

- **BYTESIZED32-SP 数据集**：可直接作为文本世界模拟 benchmark
- **Fact / Fenv / FR 分解**：适合分析任何 world model 的失效位置
- **Full-state / State-diff 双表示**：适合测试表示形式是否影响 simulator fidelity
- **属性级错误分析协议**：能把错误从“答错了”细分到“哪类机制不会”
- **规则上下文对照**：可用于评估代码生成型、程序归纳型或 neurosymbolic world model

一句话总结这篇论文的意义：  
**它没有证明 LLM 已经能当世界模拟器，反而第一次比较扎实地证明了：当前 LLM 离“可靠 simulator”还差的关键一截，主要就在环境动态建模。**

## Local PDF reference

![[paperPDFs/Building_World_Models_from_Language_Priors/ACL_2024/2024_Can_Language_Models_Serve_as_Text_Based_World_Simulators.pdf]]