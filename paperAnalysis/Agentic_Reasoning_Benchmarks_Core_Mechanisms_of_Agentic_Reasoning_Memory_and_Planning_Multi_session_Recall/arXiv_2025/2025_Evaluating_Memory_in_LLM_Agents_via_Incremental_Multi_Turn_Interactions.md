---
title: "Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions"
venue: ICLR
year: 2026
tags:
  - Survey_Benchmark
  - task/llm-agent-evaluation
  - task/long-term-memory-evaluation
  - incremental-multi-turn-evaluation
  - competency-taxonomy
  - benchmark-reconstruction
  - dataset/MemoryAgentBench
  - dataset/EventQA
  - dataset/FactConsolidation
  - opensource/no
core_operator: 将长上下文任务重构为按时间逐块注入的多轮交互，并用四项记忆能力统一评测记忆代理。
primary_logic: |
  长文本/长对话与新构造样本 → 切分为顺序对话块并显式触发“记忆”写入 → 从准确检索、测试时学习、长程理解、选择性遗忘四维统一评测不同记忆代理 → 暴露各类记忆机制的能力边界与失衡
claims:
  - "MemoryAgentBench 在文中比较的基准里是唯一同时覆盖准确检索、测试时学习、长程理解、选择性遗忘四项能力，并统一覆盖长上下文、RAG、Agentic Memory 三类代理评测的基准 [evidence: analysis]"
  - "在主实验中，RAG 方法在准确检索任务上通常优于其 GPT-4o-mini 骨干基线，而长上下文模型在测试时学习与长程理解任务上整体更强 [evidence: comparison]"
  - "FactConsolidation 的短上下文版本可被强推理模型求解（o4-mini 在 6K 多跳上达 80.0），但扩展到 32K 后降至 14.0，说明瓶颈主要来自长程记忆更新与冲突整合，而非任务本身不可解 [evidence: analysis]"
related_work_position:
  extends: "LongMemEval (Wu et al. 2025)"
  competes_with: "LOCOMO (Maharana et al. 2024); StoryBench (Wan & Ma, 2025)"
  complementary_to: "MemGPT (Packer et al. 2023); MIRIX (Wang & Chen, 2025)"
evidence_strength: strong
pdf_ref: "paperPDFs/Agentic_Reasoning_Benchmarks_Core_Mechanisms_of_Agentic_Reasoning_Memory_and_Planning_Multi_session_Recall/arXiv_2025/2025_Evaluating_Memory_in_LLM_Agents_via_Incremental_Multi_Turn_Interactions.pdf"
category: Survey_Benchmark
---

# Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2507.05257)
> - **Summary**: 论文提出 MemoryAgentBench，把原本“一次性给完整长上下文”的评测改写为“按时间逐块输入、逐步记忆”的多轮交互，从而更真实地测试 LLM agent 的长期记忆写入、更新、整合与遗忘能力。
> - **Key Performance**: 基准覆盖 2071 个问答/评测项、上下文长度 103K–1.44M；主实验中所有方法在选择性遗忘多跳任务上的最高准确率仅 7.0%。

> [!info] **Agent Summary**
> - **task_path**: 增量多轮文本交互/长历史注入 -> 记忆代理四维能力诊断分数（AR/TTL/LRU/SF）
> - **bottleneck**: 现有长上下文或 RAG 评测大多一次性暴露全部信息，测不到 agent 在时间维度上的记忆写入、更新和覆盖冲突
> - **mechanism_delta**: 把长上下文数据重写成顺序对话块，并用四项认知启发能力分解统一评测不同记忆架构
> - **evidence_signal**: 跨 11 个任务、三类代理的大规模对比，加上 chunk size / top-k / backbone / 数据有效性消融
> - **reusable_ops**: [长上下文重写为增量对话, 四能力分解评测协议]
> - **failure_modes**: [选择性遗忘尤其多跳几乎全面失效, RAG 类方法缺乏测试时学习与全局长程理解]
> - **open_questions**: [如何评测参数化记忆, 如何构造更自然但仍可严格判分的长期遗忘场景]

## Part I：问题与挑战

这篇论文真正想解决的，不是“模型能不能读 100K+ token”，而是：

**在真实 agent 场景里，信息是逐轮到达的；系统必须边接收、边压缩、边更新、边在未来取回。**  
因此，**memory ≠ long context**。一次性把整本书或整段历史塞进模型，只能测“单次阅读能力”，很难测到真正的“长期记忆机制”。

### 真正的瓶颈是什么？

现有评测主要有三类缺口：

1. **上下文太短**：早期 benchmark 已经不足以挑战新模型。
2. **静态一次性输入**：很多长上下文 benchmark 假设“所有信息同时可见”，这与 memory agent 的增量写入机制不一致。
3. **覆盖维度不全**：已有 memory benchmark 往往只测检索，难以同时测：
   - **Accurate Retrieval (AR)**：能否从长历史里取回正确片段
   - **Test-Time Learning (TTL)**：能否在部署时从历史样例学到新映射/规则
   - **Long-Range Understanding (LRU)**：能否对超长历史形成整体理解
   - **Selective Forgetting (SF)**：能否用新信息覆盖旧信息，并按最终状态回答

### 为什么现在要做？

因为 memory agent 已经大量出现：MemGPT、Mem0、MIRIX、Zep、Cognee 等都在主打“长期记忆”。  
但现状是：

- 工业系统很多，**可重复、统一、横向可比的证据很少**
- 学界常把“长上下文能力”误当“记忆能力”
- 不同 memory 方案各有偏置，但之前缺少一个能把这些偏置拆开的诊断工具

### 输入 / 输出接口与边界条件

论文把所有任务统一成：

- **输入**：顺序到达的 chunk 序列 \(c_1, c_2, ..., c_n\)
- **交互方式**：每个 chunk 被包装成 user-assistant 对话，并显式要求 agent “memorize”
- **输出**：在看完全部历史后，回答一组问题 \(q_1, ..., q_m\)

边界条件也很明确：

- 主要评测**文本历史 + 外部数据库型 memory agent**
- 不重点覆盖**参数化记忆**方法
- 主要是**离线注入后问答**，不是完整在线行动闭环

### 四类能力与对应任务

| 能力 | 代表任务 | 真正在测什么 |
|---|---|---|
| AR | SH/MH-Doc QA, LongMemEval(S*), EventQA | 能否从长历史中定位必要事实或事件链 |
| TTL | BANKING77, CLINC150, TREC, NLU, Movie Recommendation | 能否从历史示例中学到新标签映射或偏好规律 |
| LRU | ∞Bench-Sum, Detective QA | 能否形成全局抽象，而不只是局部检索 |
| SF | FactConsolidation-SH/MH | 能否识别旧事实已失效，并以新事实为准推理 |

---

## Part II：方法与洞察

论文的核心产物不是一个新 memory model，而是一个**更接近真实记忆代理工作方式的评测框架**：**MemoryAgentBench**。

### 评测框架做了什么？

#### 1. 把“静态长上下文”改写成“增量多轮交互”
作者把已有长上下文数据集切块、重排、对话化，使 agent 必须像真实系统那样一段段接收信息。

这一步很关键，因为它改变了评测的可见信息分布：

- 旧设定：所有证据一次性可见
- 新设定：证据按时间到达，agent 需要自己维护长期状态

#### 2. 增补两个新数据集，补足原基准测不到的能力
- **EventQA**：测长篇叙事中基于前序事件的事件续接与检索推理
- **FactConsolidation**：测冲突事实下的选择性遗忘，要求“新事实覆盖旧事实”

#### 3. 用统一协议横向比较三类 memory agent
作者评测三大类系统：

- **Long-Context Agents**：直接把当前可容纳历史塞进上下文窗口
- **RAG Agents**：外部存储 + 检索
- **Agentic Memory Agents**：带多步检索/反思/工具或分层 memory 的 agent

这样才能看清：  
**不同 memory 机制究竟强在哪，弱在哪，而不是只看一个混合总分。**

### 核心直觉

**变化是什么？**  
把“完整历史一次性呈现”改成“历史分块按时间注入”。

**改变了哪个瓶颈？**  
它改变了评测里的**信息可见性与状态管理约束**：模型不能再只依赖一次性全局注意力；它必须真的完成记忆写入、抽象压缩、检索、覆盖更新。

**带来了什么能力变化？**  
这使 benchmark 能把原来混在一起的能力拆开，区分出：

- 谁擅长“定位片段”（AR）
- 谁能“从例子学规则”（TTL）
- 谁能“理解整体叙事”（LRU）
- 谁能“忘掉旧的、保留新的”（SF）

**为什么这设计有效？**  
因为 memory agent 的失败，往往不是“不会回答”，而是前面某一步出了问题：

- 没存进去
- 存了但找不到
- 找到了但无法全局整合
- 旧信息没被覆盖

用四维拆解后，失败原因才可诊断。

### 战略取舍

| 设计选择 | 改变的测量瓶颈 | 收益 | 代价 |
|---|---|---|---|
| 长上下文重写为增量对话 | 去掉“一次性全可见”假设 | 更接近真实 memory agent 工作流 | 构建与运行成本更高 |
| 四能力分解 | 把单一总分拆成可诊断维度 | 能定位机制短板 | benchmark 更复杂 |
| 同一长历史配多个问题 | 降低超长上下文注入成本 | 提高评测效率与统计稳定性 | 问题间可能共享上下文偏置 |
| SF 中显式规定“新事实优先” | 让遗忘目标可判定 | 能精确检查覆盖更新是否成功 | 场景相对受控、偏合成 |

### 一个很重要的设计细节

论文不是把 raw text 直接扔给模型，而是用统一 prompt 包装成：

- “这是我读过的文档/小说片段/对话历史”
- “请记住，之后我会问你问题”

对 SF 任务还专门加入规则：

- 每条事实有序号
- **更新的事实序号更大**
- 发生冲突时必须以更新事实为准

这意味着 benchmark 不只是测语言模型本身，也在测**memory interface 是否能被正确触发**。

---

## Part III：证据与局限

### 关键实验信号

#### 1. 比较信号：RAG 擅长检索，长上下文擅长学习和整体理解
主表最清楚的结论是能力分工：

- **AR 上**：不少 RAG 方法优于 GPT-4o-mini 这类纯长上下文基线，符合“检索定位片段”的强项
- **TTL / LRU 上**：长上下文模型整体更强，说明“从大量历史例子里学规律”和“形成整体理解”仍更依赖全局可见性

这说明当前很多 memory agent 本质上还是**检索型增强系统**，并不等于真正具备强长期学习与整合理解。

#### 2. 诊断信号：Selective Forgetting 是当前最硬的短板
最醒目的结果不是谁最好，而是**大家都不行**：

- 主实验里，**FactConsolidation-MH 最高仅 7.0%**
- 即使总体分数最好的系统，也没有真正解决“冲突更新后的长程推理”

这说明今天的 memory 机制大多能“加东西”，但不擅长“覆盖旧东西”。

#### 3. 有效性信号：不是题坏了，而是长程记忆更新真的难
作者还做了一个很关键的验证：

- 在 **6K** 的 FactConsolidation-MH 上，**o4-mini 可到 80.0**
- 到 **32K** 时，降到 **14.0**

这个实验很重要，因为它表明：

> 问题不是 benchmark 不可解，而是当冲突事实被放进更长历史后，当前系统无法稳定找到并整合“最新版本”的记忆。

#### 4. 消融信号：检索粒度和 top-k 能改善 AR，但救不了 LRU
消融结果显示：

- **更小 chunk size** 往往能提升 AR，因为检索更精细
- 但对 **LRU** 反而不一定好，因为任务需要整体叙事连续性
- **更大的 top-k** 通常有帮助，但代价是输入 token 急剧上升

所以这不是简单的“多检一些就行”，而是**检索和全局理解之间有结构性冲突**。

#### 5. backbone 信号：静态 RAG 的瓶颈更像 memory design，agentic memory 仍吃 backbone
作者观察到：

- 对普通 RAG，换更强 backbone 提升有限
- 对 MIRIX 这类 agentic memory，换强 backbone 提升明显

含义是：

- 静态 RAG 的问题更多在检索与记忆结构本身
- agentic memory 还明显依赖底层模型的推理与协调能力

### 1-2 个最值得记住的数字

- **基准规模**：2071 个问答/评测项，103K–1.44M 上下文
- **最关键短板**：Selective Forgetting 多跳任务主实验最高仅 **7.0%**

### 局限性

- **Fails when**: 需要评测真实在线反馈学习、跨模态长期记忆、参数化记忆更新，或更自然的“何时该忘/不该忘”决策时，这个 benchmark 覆盖不足。
- **Assumes**: 记忆主要以文本历史或外部存储存在；memory 能通过 prompt 被显式触发；部分任务依赖 GPT-4o judge；大量实验依赖闭源 API、较高 token 预算和较强算力，且作者也明确说因预算只评了代表性 agents。
- **Not designed for**: 规划、工具调用执行、安全/隐私治理、用户长期人格一致性等 full-stack agent 能力评测。

还要补充两个实际限制：

1. **复现成本不低**：一些 memory agent 的 memory construction 延迟非常高，部分 embedding/graph 系统还需要较大 GPU。
2. **开源状态不够实锤**：论文正文声称会开源代码与数据，但抽取文本里没有给出可核验链接，因此当前更适合把它看成“可复现实验承诺”，不是现成可跑的 artifact。

### 可复用组件

这篇论文最值得复用的不是某个单独分数，而是下面这些“评测算子”：

- **长上下文 → 增量多轮交互** 的重写流程
- **AR / TTL / LRU / SF** 四能力分解框架
- **EventQA**：面向长篇叙事的事件延续记忆测试
- **FactConsolidation**：面向冲突更新与选择性遗忘的受控测试
- **统一 prompt / chunk / top-k 协议**：适合后续继续扩展到更多 memory agent

### 一句话总结

这篇论文的价值不在于证明某个 memory agent 已经很强，而在于**首次把“记忆代理到底是在检索、在学习、在理解，还是在真正更新记忆”系统性拆开来测**；而结果也很明确：**现有方法高度偏科，尤其不会“忘掉旧的并以新的为准”。**

![[paperPDFs/Agentic_Reasoning_Benchmarks_Core_Mechanisms_of_Agentic_Reasoning_Memory_and_Planning_Multi_session_Recall/arXiv_2025/2025_Evaluating_Memory_in_LLM_Agents_via_Incremental_Multi_Turn_Interactions.pdf]]