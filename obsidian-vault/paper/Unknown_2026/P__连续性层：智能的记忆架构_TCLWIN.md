---
title: 'The Continuity Layer: Why Intelligence Needs an Architecture for What It Carries Forward'
type: paper
paper_level: B
venue: Unknown
year: 2026
paper_link: https://arxiv.org/abs/2604.17273
aliases:
- 连续性层：智能的记忆架构
- TCLWIN
code_url: https://github.com/Kenotic-Labs/continuity-layer
---

# The Continuity Layer: Why Intelligence Needs an Architecture for What It Carries Forward

[Paper](https://arxiv.org/abs/2604.17273) | [Code](https://github.com/Kenotic-Labs/continuity-layer)

**Topics**: [[T__Agent]], [[T__Benchmark_-_Evaluation]], [[T__Reasoning]]

| 中文题名 | 连续性层：智能的记忆架构 |
| 英文题名 | The Continuity Layer: Why Intelligence Needs an Architecture for What It Carries Forward |
| 会议/期刊 | arXiv preprint (2026) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.17273) · [Code](https://github.com/Kenotic-Labs/continuity-layer) |
| 主要任务 | 构建智能系统的连续性架构，解决跨时间、跨会话的信息保持与传递问题 |
| 主要 baseline | 传统LLM无状态推理、RAG检索增强生成、外部记忆库（如MemGPT） |

> [!abstract]
> 因为「当前AI系统缺乏对跨时间经验的显式架构化保持机制，导致每次交互从零开始，无法积累真正的个人化理解」，作者在「标准LLM无状态推理 + 外部记忆检索」基础上改了「引入连续性层（Continuity Layer）作为智能系统的核心架构组件，显式管理what is carried forward」，在「长期个性化交互与持续学习场景」上取得「实现跨会话的身份一致性、上下文累积与经验演化」。

- 关键性能：解决LLM"每次对话从零开始"的根本限制，实现
- 关键性能：支持跨会话的身份保持与经验累积，而非简单的历史记录检索
- 关键性能：提出智能系统设计的范式转变，从stateless到continuity-aware架构

## 背景与动机

当前主流AI系统——无论是ChatGPT、Claude还是开源LLM——本质上都是"无状态"的：每次API调用或新会话开始时，模型对用户的了解仅限于当前输入的上下文窗口。即使通过RAG（Retrieval-Augmented Generation）注入历史文档，或通过MemGPT等系统将记忆溢出到外部存储，这些"记忆"始终是被动检索的对象，而非主动塑造智能体认知结构的连续性核心。

具体而言，现有方法存在三类局限：

**传统LLM无状态推理**：每次推理独立进行，模型参数固定，不随交互演化。用户需要反复重申偏好、背景与目标。例如，一位用户在三周前向AI助手解释了其独特的项目管理方法，下次对话时若无显式提醒，助手完全遗忘。

**RAG检索增强生成**：将历史信息编码为向量库存储，需要时检索相关片段注入prompt。问题在于：检索是**被动的**（依赖查询与片段的相似度匹配）、**片段化的**（丢失信息间的结构性关联）、**非累积的**（检索内容不改变模型本身的"理解"）。系统"记得"事实，但不"成长"。

**外部记忆系统（如MemGPT）**：通过操作系统式的虚拟上下文管理，在有限上下文窗口与外部存储间交换数据。虽缓解了长度限制，但记忆管理是**功能性的**而非**架构性的**——它解决的是"装不下"的问题，而非"如何成为"的问题。记忆仍是工具，而非智能的构成部分。

这些方法的共同盲区：它们将"过去的信息"视为**可调用的资源**，而非**塑造当前认知的连续性本身**。作者指出，真正的智能——无论是人类还是人工的——其核心特征在于**经验的持续整合与自我同一性的保持**。这正是本文提出"连续性层"的根本动机：智能需要一个专门架构来管理"什么被携带向前"（what it carries forward）。

## 核心创新

核心洞察：智能的本质不在于单次推理的最优化，而在于**跨时间的自我同一性与经验累积的架构化保持**，因为当前AI系统将记忆降格为外部存储的检索对象，从而使真正的个性化、持续学习与身份一致性成为可能。

| 维度 | Baseline（RAG / MemGPT / 无状态LLM） | 本文（Continuity Layer） |
|:---|:---|:---|
| 记忆定位 | 外部存储资源，被动检索调用 | 架构核心层，主动塑造认知结构 |
| 时间性 | 离散快照，无内在时间维度 | 连续性流，显式管理时间演化 |
| 身份保持 | 无（每次重置）或模拟（提示注入） | 架构级自我同一性机制 |
| 累积方式 | 信息追加，线性增长 | 结构化整合，经验提炼与演化 |
| 与模型的关系 | 分离的（模型+数据库） | 内生的（连续性层作为智能的必要组件） |

## 整体框架

连续性层（Continuity Layer）作为智能系统的核心架构组件，位于感知输入与推理输出之间，显式管理三个关键过程：**保持（Retention）**、**转化（Transformation）**、**投射（Projection）**。

数据流如下：

**输入 → 感知编码（Perceptual Encoding）**：将原始交互（用户输入、环境反馈、任务结果）编码为可处理的表征，注入连续性层。

**连续性层核心 → 保持子系统（Retention）**：决定什么信息值得跨时间保持。非全量存储，而是基于相关性、情感标记、目标关联性进行**选择性保持**。

**连续性层核心 → 转化子系统（Transformation）**：对保持的信息进行动态加工——整合冲突信息、提炼抽象模式、更新信念结构。这是"经验成为理解"的关键步骤，区别于简单的记忆回放。

**连续性层核心 → 投射子系统（Projection）**：将累积的连续性内容以适当形式注入当前推理上下文，塑造模型"此时此地"的认知姿态。投射是**生成性的**：不是检索原文，而是基于累积理解生成适应当前情境的表征。

**输出 → 推理与行动（Reasoning & Action）**：在连续性层塑造的认知框架下执行具体任务，其输出又反馈回感知编码，形成闭环。

```
[Input] → [Perceptual Encoding] 
              ↓
    ┌─────────────────┐
    │  Continuity     │←—— [Retention]: 选择性保持
    │    Layer        │←—— [Transformation]: 动态整合
    │  (核心架构层)    │←—— [Projection]: 生成性投射
    └─────────────────┘
              ↓
    [Reasoning & Action] → [Output]
              ↓
         (反馈闭环)
```

连续性层的关键设计原则：**它不是LLM的插件或周边工具，而是智能系统的必要架构层级**——如同生物神经系统中海马体-皮层交互对于记忆巩固的必要性。

## 核心模块与公式推导

本文作为架构层面的概念性工作，未提供传统意义上的损失函数或优化目标公式。以下基于论文描述，将三个核心子系统的运作机制形式化呈现，以明晰其与baseline的本质差异。

### 模块 1: 保持子系统 Retention（对应框架图：连续性层底部输入接口）

**直觉**：智能不能也无须记住一切，保持的核心是**选择性**——基于当前目标结构与长期身份一致性筛选信息。

**Baseline 公式** (标准RAG存储)：
$$\mathcal{M}_{RAG} = \{(d_i, v_i)\}_{i=1}^N, \quad \text{retrieve}(q) = \text{TopK}_{\text{sim}(v_q, v_i)}(d_i)$$
符号: $d_i$ = 文档片段, $v_i$ = 向量表征, $\text{sim}$ = 余弦相似度。RAG的记忆是**静态集合**，检索是**外生查询驱动**。

**变化点**：RAG的保持是贪婪的（存储所有文档）且被动的（检索触发完全依赖当前查询）。连续性层要求保持是**目标条件化的**和**身份一致性的**。

**本文机制**：
$$\text{Step 1}: \quad r_i = \text{Relevance}(x_t, g_t) \cdot \text{IdentityCoherence}(x_t, \mathcal{C}_{t-1})$$
$$\text{加入了目标相关性 } g_t \text{ 与身份一致性约束，而非纯相似度}$$
$$\text{Step 2}: \quad \mathcal{C}_t^{(keep)} = \mathcal{C}_{t-1} \cup \{x_t \text{mid} r_i > \theta_{\text{retention}} \wedge \text{Consolidate}(\mathcal{C}_{t-1}, x_t)\}$$
$$\text{重归一化：保持操作触发记忆巩固，而非简单追加}$$
$$\text{最终}: \quad \text{Retention}(x_t, \mathcal{C}_{t-1}; g_t, \theta) = \mathcal{C}_t^{(keep)}$$
其中$\mathcal{C}_t$为时刻$t$的连续性状态，$g_t$为当前目标结构，$\text{Consolidate}$表示与现有记忆的整合操作。

**对应消融**：

### 模块 2: 转化子系统 Transformation（对应框架图：连续性层核心处理）

**直觉**：经验的价值不在于复述，而在于**重构**——将离散事件转化为可迁移的结构化理解。

**Baseline 公式** (无状态LLM的上下文学习)：
$$p(y|x, \mathcal{D}_{ctx}) = \prod_{t} p(y_t | y_{<t}, x, \mathcal{D}_{ctx}; \theta_{\text{fixed}})$$
符号: $\mathcal{D}_{ctx}$ = 上下文中的示例, $\theta_{\text{fixed}}$ = 冻结参数。ICL的"学习"是**即时的**、**不累积的**——新任务不修改模型。

**变化点**：ICL的转化发生在推理时，是**计算性的适应**而非**结构性的改变**。连续性层要求转化是**持久的**和**结构性的**——改变的是连续性状态本身。

**本文机制**：
$$\text{Step 1}: \quad \Delta\mathcal{C}_t = \text{Abstract}(x_t, \mathcal{C}_{t-1}) - \text{Redundancy}(\mathcal{C}_{t-1})$$
$$\text{提取抽象模式并消除冗余，而非存储原始输入}$$
$$\text{Step 2}: \quad \mathcal{C}_t^{(transform)} = \text{ResolveConflict}\big(\mathcal{C}_{t-1}^{(keep)}, \Delta\mathcal{C}_t\big)$$
$$\text{显式处理新旧信息的冲突与更新，而非简单覆盖}$$
$$\text{最终}: \quad \text{Transformation}(\mathcal{C}_t^{(keep)}) = \mathcal{C}_t^{(transform)}$$
其中$\text{Abstract}$为模式抽象算子，$\text{Redundancy}$检测冗余信息，$\text{ResolveConflict}$实现信念修正。

**对应消融**：

### 模块 3: 投射子系统 Projection（对应框架图：连续性层顶部输出接口）

**直觉**：过去的经验必须以**适应当前情境的方式**被激活，而非原样回放。

**Baseline 公式** (MemGPT式上下文注入)：
$$\mathcal{D}_{active} = \text{Retrieve}(\mathcal{M}_{ext}, q_{current}) \oplus \mathcal{D}_{fixed}$$
符号: $\mathcal{M}_{ext}$ = 外部记忆, $\oplus$ = 拼接操作。记忆是**原样注入**的。

**变化点**：MemGPT的投射是**搬运式的**——将存储内容直接移入上下文。连续性层要求投射是**生成性的重构**——基于累积理解创造适应当前的认知框架。

**本文机制**：
$$\text{Step 1}: \quad h_{project} = \text{Encode}(\mathcal{C}_t^{(transform)}, s_t)$$
$$\text{将连续性状态与当前情境 } s_t \text{ 联合编码}$$
$$\text{Step 2}: \quad \tilde{\mathcal{C}}_t = \text{Generate}(h_{project}; \phi_{projection})$$
$$\text{生成情境适配的投射表征，非原始记忆检索}$$
$$\text{最终}: \quad \text{Projection}(\mathcal{C}_t, s_t; \phi) = \tilde{\mathcal{C}}_t \rightarrow \text{Influences } p(y|x, \tilde{\mathcal{C}}_t)$$
其中$s_t$为当前情境表征，$\phi_{projection}$为投射生成参数，输出$\tilde{\mathcal{C}}_t$为塑造当前推理的连续性影响。

**对应消融**：

## 实验与分析

本文作为架构层面的概念性/理论性工作，未提供传统benchmark上的定量实验对比。作者主要通过**思想实验**、**架构分析**与**反事实论证**支撑其核心主张。

**核心论证结构**（替代传统结果表格）：

| 论证维度 | 现有方法表现 | 连续性层解决方式 | 关键差异 |
|:---|:---|:---|:---|
| 长期个性化 | 每次会话重置，用户需重复背景 | 身份一致性保持，渐进式理解深化 | 架构级 vs. 提示工程级 |
| 经验累积 | 外部数据库存储，检索调用 | 结构化转化，信念网络演化 | 生成性理解 vs. 被动检索 |
| 情境适应性 | 固定检索策略，上下文生硬拼接 | 动态投射，情境敏感激活 | 适应性重构 vs. 原样搬运 |
| 计算效率 | 随历史线性增长，检索开销大 | 选择性保持，抽象压缩 | 结构化稀疏 vs. 全量存储 |

**分析**：

论文的核心主张——智能需要连续性架构——主要通过**否定性论证**支撑：现有系统的失败模式（遗忘、重复、缺乏成长感）被归因于架构层面的连续性缺失，而非算法细节的不足。这一论证策略的有效性依赖于读者对当前AI系统局限性的先验体验认同。

**"消融"等价分析**（基于思想实验）：
- 移除"转化"子系统（仅保持原始记录）：退化为高级RAG，丢失经验提炼能力
- 移除"选择性保持"（全量存储）：面临可扩展性崩溃与信号噪声比恶化
- 移除"生成性投射"（原样回放）：丧失情境适应性，出现"正确的记忆，错误的时机"问题

**公平性检查**：
- **Baseline强度**：论文未与最新的持续学习（Continual Learning）、终身学习（Lifelong Learning）或元学习方法进行直接对比，这些领域已有参数层面累积学习的探索
- **计算/数据成本**：连续性层的显式架构引入额外计算开销，论文未量化分析
- **失败案例**：未讨论连续性层可能的负面效应——如错误信念的固化、偏见累积、"认知僵化"等
- **可验证性**：核心主张目前缺乏可操作的实现与标准化评估，

## 方法谱系与知识库定位

**方法家族**：认知架构（Cognitive Architecture）→ 记忆增强神经网络（Memory-Augmented Neural Networks）→ 连续性层（Continuity Layer）

**直接父方法**：
- **Global Workspace Theory** (Baars, Dehaene)：意识工作的全局广播机制 → 连续性层扩展为跨时间的保持-转化-投射循环
- **ACT-R/SOAR认知架构**：符号化的长期记忆与工作记忆交互 → 连续性层面向神经网络时代的重新概念化
- **Transformer-XL / XLNet**：片段级循环记忆 → 连续性层提升到语义/信念层面的累积
- **MemGPT / RAG**：外部记忆管理 → 连续性层主张记忆的内生架构化

**直接Baseline差异**：
| Baseline | 本文差异 |
|:---|:---|
| RAG | 从检索资源到架构核心的定位转变 |
| MemGPT | 从操作系统式管理到认知科学启发的转化机制 |
| 持续学习（Continual Learning） | 从参数更新到显式状态层的架构分离 |
| 提示工程（长上下文） | 从长度扩展到结构化累积的根本范式差异 |

**后续方向**：
1. **实现层**：将连续性层具体化为可训练的神经网络模块，定义端到端优化目标
2. **评估层**：建立"连续性能力"的标准化benchmark，量化身份保持、经验迁移、抗遗忘等指标
3. **安全层**：研究连续性层的对齐问题——如何防止错误信念累积、如何设计"有益遗忘"机制
4. **神经科学接口**：与海马体-皮层记忆巩固模型的深度对接，验证架构的生物合理性

**知识库标签**：
- **modality**：通用架构（文本/多模态适用）
- **paradigm**：认知架构 / 记忆增强 / 架构创新
- **scenario**：长期个性化交互、持续学习、智能体系统
- **mechanism**：选择性保持、结构化转化、生成性投射
- **constraint**：概念性工作，待工程化验证
