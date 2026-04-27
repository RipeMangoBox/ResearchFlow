---
title: 'VoxMind: An End-to-End Agentic Spoken Dialogue System'
type: paper
paper_level: B
venue: ACL
year: 2026
paper_link: https://arxiv.org/abs/2604.15710
aliases:
- 端到端语音对话Agent系统VoxMind
- VoxMind
- 端到端语音模型的 agent 能力瓶颈本质上是两个已在文本领域解决的问
acceptance: accepted
method: VoxMind
---

# VoxMind: An End-to-End Agentic Spoken Dialogue System

[Paper](https://arxiv.org/abs/2604.15710)

**Topics**: [[T__Agent]], [[T__Speech_Processing]], [[T__Reasoning]] | **Method**: [[M__VoxMind]]

> [!tip] 核心洞察
> 端到端语音模型的 agent 能力瓶颈本质上是两个已在文本领域解决的问题在语音模态的重现：缺乏中间推理步骤（CoT）和工具集规模导致的上下文膨胀。VoxMind 的核心洞察是：语音模态的特殊性（高 token 密度）使得这两个问题的严重程度被放大，因此需要将文本 agent 的成熟解法（CoT + 工具检索）以语音感知的方式重新实现——具体而言，通过并行辅助 LLM 将工具检索延迟隐藏在主模型推理过程中，从而在不牺牲响应速度的前提下支持大规模工具调用。有效性的根本来源是训练数据（AgentChat）提供了语音模态下的 agent 行为监督信号，而非架构本身的创新。

| 中文题名 | 端到端语音对话Agent系统VoxMind |
| 英文题名 | VoxMind: An End-to-End Agentic Spoken Dialogue System |
| 会议/期刊 | ACL 2026 (accepted) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.15710) · [Code] · [Project] |
| 主要任务 | 端到端语音对话系统中的agent能力构建（工具调用、推理规划、动态上下文感知） |
| 主要 baseline | Qwen-Audio、Kimi-Audio、Gemini-2.5-Pro |

> [!abstract] 因为「端到端语音模型停留在反应式x→y映射，缺乏推理规划与工具调用能力」，作者在「标准端到端语音对话基线」基础上改了「叠加Think-before-Speak CoT机制+多智能体动态工具管理+AgentChat语音agent数据集」，在「自构建评测基准」上取得「任务完成率34.88%→74.57%，超越Gemini-2.5-Pro」。

- **任务完成率**：34.88% → 74.57%（相对提升113.9%），超越Gemini-2.5-Pro
- **延迟控制**：工具集从10增至100时，主模型等待开销始终低于15ms，实现O(1)延迟特性
- **THINK token开销**：仅占总token的约12.6%，数量约80-90且不随工具库规模增长

## 背景与动机

现有端到端语音对话系统（如Qwen-Audio、Kimi-Audio）已能流畅地进行语音问答，但面对一个具体场景——用户说"帮我查一下明天北京飞上海的航班，要上午的，然后订一辆去机场的专车"——这些系统会直接尝试生成语音回复，却无法像文本LLM agent那样先拆解目标、调用航班查询工具、再调用打车工具、最后组织回复。这种"听到就说"的反应式范式在复杂任务中必然失败。

现有方法如何处理这一问题？**Qwen-Audio**等端到端语音模型将声学特征直接映射到文本或语音输出，通过大规模预训练获得强感知能力，但完全没有显式推理模块。**Kimi-Audio**同样采用x→y的直接映射，其工具使用能力依赖文本LLM的后处理或有限的多模态对齐，无法在语音模态内部完成结构化决策。**Gemini-2.5-Pro**作为多模态大模型，虽具备agent能力，但并非专为端到端语音交互设计，其语音接口往往是文本agent的包装层，存在模态转换开销与体验割裂。

这些方法的共同短板可归结为三点：**语义-动作鸿沟**——语音模型在解析工具语义、生成格式正确的工具调用参数方面显著弱于纯文本LLM，缺乏中间推理步骤导致规划能力缺失；**数据瓶颈**——语音领域不存在同时标注结构化推理轨迹与工具交互行为的大规模语料，模型无法通过监督学习内化agent行为；**延迟膨胀**——语音输入需要远多于文本的token编码声学信息，叠加大规模工具描述后上下文长度急剧膨胀，推理延迟随工具集规模线性增长。

VoxMind的核心动机正是将文本agent领域已成熟的CoT推理与工具调用范式适配到语音模态，并通过工程创新解决语音特有的延迟瓶颈。
![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/9bf6377a-3dc7-4225-99c1-85f38fb7fd8c/figures/Figure_3.png)
*Figure 3: Figure 3: Dialogues demonstrating the agent’s six core capabilities.*



## 核心创新

核心洞察：语音模态的高token密度特性放大了文本agent已解决的两大问题（缺乏中间推理与工具集上下文膨胀），因此需要将文本agent的成熟解法以"语音感知"的方式重新实现——具体而言，通过并行辅助LLM将工具检索延迟隐藏在主模型推理过程中，从而在不牺牲响应速度的前提下支持大规模工具调用。

| 维度 | Baseline（Qwen-Audio/Kimi-Audio） | 本文VoxMind |
|:---|:---|:---|
| 推理范式 | 反应式：语音输入→直接语音输出（x→y） | 规划式：语音输入→显式推理轨迹→动作/语音输出（x→z→y） |
| 工具管理 | 全局工具集一次性载入上下文，延迟O(n) | 辅助LLM动态检索本地工具子集，主模型延迟O(1) |
| 训练数据 | 无语音模态的agent行为标注数据 | AgentChat：470小时TTS合成语音，含结构化推理轨迹与工具交互 |
| 记忆机制 | 单一语义记忆或无显式设计 | 双通道（语义+声学）记忆 + 静态/动态Agent Profile分离 |

## 整体框架


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/9bf6377a-3dc7-4225-99c1-85f38fb7fd8c/figures/Figure_2.png)
*Figure 2: Figure 2: Overall architecture of the VoxMind. Given spoken user input, the speech-centric agent first generates anexplicit reasoning trajectory in a "think-before-speak" manner. Conditioned on this r*



VoxMind的整体架构围绕"语音输入→显式推理→工具调用/语音输出"的主链路展开，包含五个核心模块：

**语音编码器（Speech Encoder）**：接收原始语音输入，输出高维声学表征。作为端到端系统的入口，负责将连续声学信号转化为离散token序列，供后续模块处理。

**Think-before-Speak推理模块（CoT Generator）**：在生成最终响应前，先采样显式推理轨迹。通过在解码阶段插入THINK token序列，将直接映射扩展为x→z→y的两阶段生成。该模块是agent能力的核心载体，决定系统是否具备规划与反思能力。

**多智能体动态工具管理器（Dynamic Tool Manager）**：由辅助LLM构成的并行子系统。异步从全局工具池T^all中检索候选工具，维护紧凑的本地工具空间T^local_t ⊂ T^all。与主模型推理过程并行执行，确保主模型等待开销低于15ms。

**动作执行器（Action Executor）**：在推理轨迹条件下选择具体动作——或生成语音回复，或输出结构化工具调用。形式化为a_t ~ π^act_θ(a | c_t, o_t, H_{t-1}, T^local_t)。

**双通道记忆与Profile模块**：维护语义记忆（对话历史H）与声学记忆（语音特征），同时管理Agent Profile的静态属性P_static与动态上下文自适应属性P_dynamic(c)。

数据流示意：
```
语音输入 → [Speech Encoder] → 声学token序列
                              ↓
                    [Dynamic Tool Manager] ←→ 全局工具池T^all
                         ↓（并行，延迟<15ms）
                    本地工具子集T^local_t
                              ↓
                    [CoT Generator: THINK tokens]
                         ↓ 显式推理轨迹c_t
                    [Action Executor]
                         ↓
              ┌─────────┴─────────┐
         语音输出              工具调用 → 外部API → 结果反馈
```

## 核心模块与公式推导

### 模块一：Think-before-Speak 显式推理机制（对应框架图 CoT Generator）

**直觉**：直接生成语音响应如同"边听边答"，缺乏对复杂任务的拆解；插入中间推理步骤使模型先"想清楚"再"说出口"。

**Baseline 公式**（标准端到端语音对话，如Qwen-Audio）：
$$p(y_t | x_{\leq t}, y_{<t}) = \prod_{t=1}^{T} \pi_\theta^{\text{base}}(y_t | x, y_{<t})$$
符号：$x$为语音输入序列，$y$为输出序列（文本或语音token），$\theta$为模型参数，$\pi_\theta^{\text{base}}$为直接映射策略。

**变化点**：Baseline缺乏显式规划变量，面对多步工具调用任务时无法保证输出格式的正确性与动作选择的合理性。本文将生成过程解耦为推理+动作两阶段。

**本文公式（推导）**：
$$\text{Step 1}: \quad c_t \sim \pi_\theta^{\text{think}}(c \text{mid} o_t, H_{t-1}, T_t^{\text{local}}) \quad \text{采样推理轨迹，加入THINK token序列}$$
$$\text{Step 2}: \quad a_t \sim \pi_\theta^{\text{act}}(a \text{mid} c_t, o_t, H_{t-1}, T_t^{\text{local}}) \quad \text{以推理为条件选择动作}$$
$$\text{最终}: \quad p(y_t, c_t | x_{\leq t}) = \pi_\theta^{\text{think}}(c_t | \cdot) \cdot \pi_\theta^{\text{act}}(y_t | c_t, \cdot)$$

THINK token作为特殊控制token插入解码序列，其数量约80-90，占总token约12.6%，且不随工具库规模增长（Table 11）。

**对应消融**：

---

### 模块二：多智能体动态工具管理（对应框架图 Dynamic Tool Manager）

**直觉**：语音输入本身token量庞大，若将全部工具描述载入主模型上下文，延迟将随工具集规模线性爆炸；通过辅助LLM预筛选，主模型只关注相关工具。

**Baseline 公式**（标准上下文内工具学习）：
$$T_t^{\text{context}} = T^{\text{all}}, \quad |T^{\text{all}}| = N$$
$$\text{Latency}_{\text{base}} \propto |T^{\text{all}}| = O(N)$$
符号：$T^{\text{all}}$为全局工具集，$N$为工具总数，Latency为推理延迟。

**变化点**：Baseline的上下文长度与工具集规模成正比，语音模态的高token密度使这一问题恶化。本文引入辅助LLM并行检索，将主模型的工具上下文压缩为动态子集。

**本文公式（推导）**：
$$\text{Step 1}: \quad T_t^{\text{local}} = \text{Retrieve}_{\phi}^{\text{LLM}}(o_t, H_{t-1}; T^{\text{all}}), \quad |T_t^{\text{local}}| \ll |T^{\text{all}}|$$
$$\text{Step 2}: \quad \text{Retrieve过程与主模型推理并行执行}$$
$$\text{Step 3}: \quad \text{主模型等待开销} = \max(0, \text{Latency}_{\text{retrieve}} - \text{Latency}_{\text{encode}}) < 15\text{ms}$$
$$\text{最终}: \quad \text{Latency}_{\text{VoxMind}} = O(1) \quad \text{（相对于工具集规模N）}$$

其中$\phi$为辅助LLM参数，Retrieve过程采用异步架构，利用语音编码与工具检索的时间重叠隐藏延迟。

**对应消融**：Table 10显示，全局工具集从10增至100时，辅助LLM检索延迟从1.3s增至2.6s，但主模型等待开销始终低于15ms，验证了O(1)延迟特性。

---

### 模块三：Agent Profile与双通道记忆（对应框架图 Memory/Profile）

**直觉**：Agent需要区分"我是谁"（静态属性）与"当前场景需要我如何"（动态适应），同时语音交互既需要语义内容记忆也需要声学风格记忆。

**Baseline**：无显式Agent Profile设计，或单一固定角色描述；记忆多为纯文本对话历史。

**本文公式**：
$$P^{\text{agent}}_t = P^{\text{static}} \oplus P^{\text{dynamic}}(c_t)$$
$$M_t = M^{\text{semantic}}_t \parallel M^{\text{acoustic}}_t$$
符号：$P^{\text{static}}$为静态角色属性（如身份、能力边界），$P^{\text{dynamic}}(c_t)$为基于当前推理轨迹$c_t$动态生成的上下文自适应属性，$\oplus$为拼接操作，$\parallel$表示双通道并行存储。

**注意**：该模块在实验中的独立贡献未被单独消融验证，更多是概念层面的统一定义。

## 实验与分析

**主实验结果**：

| Method | 任务完成率 | 相对提升 |
|:---|:---|:---|
| Baseline（端到端语音模型，无agent能力） | 34.88% | — |
| Gemini-2.5-Pro | 
| **VoxMind** | **74.57%** | **+113.9%** |


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/9bf6377a-3dc7-4225-99c1-85f38fb7fd8c/figures/Figure_1.png)
*Figure 1: Figure 1: VoxMind can dynamically perceive the inter-action context, autonomously determine when to invokeexternal tools, and drive the generation of subsequentresponses based on the tool execution re*



核心数字分析：任务完成率34.88%→74.57%的跃升直接支撑了"显式CoT推理对语音agent能力至关重要"的核心主张。然而，该对比的基线选择（具体是哪个模型）与评测协议细节在摘录中未完整呈现，与Gemini-2.5-Pro的对比也缺乏实验条件说明，且未见与同类语音agent系统（如SpeechGPT、AudioPaLM等）的系统性横向比较。

**延迟与效率分析**：Table 10是本文工程贡献的关键证据。当全局工具集从10增至100时，辅助LLM检索延迟从1.3s增至2.6s（线性增长），但主模型等待开销始终低于15ms，成功将工具集规模对主链路延迟的影响降为常数。Table 11进一步显示THINK token仅占总token约12.6%，且数量稳定在80-90之间，不随工具库规模增长，说明CoT引入的计算开销可控。

**真实语音鲁棒性测试**（Appendix H.3）：在真实语音输入条件下，FS和PF指标分别下降约7.3%和6.7%，但任务成功率维持在86%。这表明系统对真实声学条件有一定鲁棒性，但也暴露了TTS合成评测存在系统性乐观偏差——AgentChat数据集通过TTS构建，缺乏真实口语的自发性与不流畅特征。

**消融与公平性检查**：
- 各组件（CoT、动态工具管理、AgentChat数据）的独立贡献消融实验在摘录中未完整呈现
- "保留通用对话质量"的声明完全缺乏实验数据支撑（置信度仅0.45）
- 工具集超过100时的延迟行为未测试
- 数据集规模470小时与同类工作相比处于什么水平


![Figure 13](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/9bf6377a-3dc7-4225-99c1-85f38fb7fd8c/figures/Figure_13.png)
*Figure 13: Figure 13 illustrates the strict evaluation procedureof Gemini-2.5-flash for end-to-end speech agents.The process follows a tool extraction + correctnessevaluation paradigm, consisting of the followin*

（若原文有完整消融图则插入，当前摘录信息不足）

## 方法谱系与知识库定位

**方法家族**：端到端语音对话系统 + 文本LLM Agent能力迁移

**父方法/核心范式来源**：
- **Chain-of-Thought推理**（Wei et al., 2022；Kojima et al., 2022）：文本LLM中的显式推理范式，VoxMind将其适配到语音模态解码过程
- **Tool Learning / Function Calling**（Schick et al., 2023；Qin et al., 2023）：文本agent的工具调用框架，VoxMind解决其在语音模态的延迟与数据瓶颈
- **Multi-Agent并行架构**：辅助LLM异步检索的设计借鉴了检索增强生成（RAG）系统的查询-生成解耦思想

**直接基线与差异**：
| 基线 | 差异 |
|:---|:---|
| Qwen-Audio / Kimi-Audio | 本文叠加显式CoT与动态工具管理，从反应式升级为agentic |
| Gemini-2.5-Pro | 本文专为端到端语音交互设计，非文本agent的语音包装层 |
| SpeechGPT等语音LLM | 本文首次系统性地引入工具调用与大规模工具集管理能力 |

**改动插槽**：架构（+CoT生成头、+辅助LLM检索器）、训练数据（+AgentChat语音agent语料）、推理流程（+THINK token解码、+异步工具检索）、概念框架（+Agent Profile形式化定义）

**后续方向**：
1. **真实口语数据构建**：用真实对话录音替代TTS合成，解决自发性与流畅性差距
2. **超大规模工具集扩展**：验证100+工具时的延迟行为与检索精度衰减
3. **多模态工具调用**：将工具输出从文本扩展为语音、图像等多模态反馈

**知识库标签**：语音模态(spoken) / Agent范式(agentic) / 对话场景(dialogue) / CoT推理机制(chain-of-thought) / 工具学习(tool-learning) / 延迟约束(latency-constrained) / 端到端训练(end-to-end)

