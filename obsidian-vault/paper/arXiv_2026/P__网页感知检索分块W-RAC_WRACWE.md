---
title: Web Retrieval-Aware Chunking (W-RAC) for Efficient and Cost-Effective Retrieval-Augmented Generation Systems
type: paper
paper_level: B
venue: arXiv
year: 2026
paper_link: https://arxiv.org/abs/2604.04936
aliases:
- 网页感知检索分块W-RAC
- WRACWE
- W-RAC 的核心直觉是：LLM 做语义分块决策时
---

# Web Retrieval-Aware Chunking (W-RAC) for Efficient and Cost-Effective Retrieval-Augmented Generation Systems

[Paper](https://arxiv.org/abs/2604.04936)

**Topics**: [[T__Agent]], [[T__Text_Generation]], [[T__Benchmark_-_Evaluation]]

> [!tip] 核心洞察
> W-RAC 的核心直觉是：LLM 做语义分块决策时，真正需要的是「哪些内容在语义上相邻」的结构信号，而非原始文本本身。网页的 HTML 层级结构已经预编码了这些信号。因此，将 LLM 的输入从「原始文本」替换为「结构化 ID + 层级元数据」，将 LLM 的输出从「重新生成的文本块」替换为「ID 分组列表」，就能在保留语义规划能力的同时，将输出 token 量压缩至与文档长度解耦的极小规模。有效性的根本来源是：输出空间的约束（只能输出已有 ID）同时消除了幻觉风险和冗余 token 生成，而网页结构先验则补偿了 LLM 看不到原始文本所损失的语义理解能力。

| 属性 | 内容 |
|------|------|
| 中文题名 | 网页感知检索分块W-RAC |
| 英文题名 | Web Retrieval-Aware Chunking (W-RAC) for Efficient and Cost-Effective Retrieval-Augmented Generation Systems |
| 会议/期刊 | arXiv (Cornell University) (预印本) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.04936) · [Code] · [Project] |
| 主要任务 | RAG 系统的文档分块（chunking）预处理阶段优化，针对网页 HTML 内容的语义分块 |
| 主要 baseline | Agentic Chunking（未具名实现，LLM 逐字处理并重新生成全文的分块方法） |

> [!abstract]
> 因为「Agentic Chunking 语义分块质量高但 token 成本与文档长度线性相关、存在幻觉风险且可观测性差」，作者在「传统 LLM-as-generator 分块范式」基础上改为「LLM-as-planner 仅输出 ID 分组列表」，在「RAG-Multi-Corpus benchmark（786 条查询）」上取得「Precision@3 从 0.55 提升至 0.71（+29%），输出 token 减少 84.6%，端到端延迟降低约 60%，总 LLM 成本降低 51.7%」

- **精度提升**：Precision@3 +29%（0.55→0.71），Precision@6 +40%（0.40→0.56）
- **效率增益**：输出 token 减少 84.6%，端到端分块延迟降低约 60%，总 LLM 成本降低 51.7%
- **代价**：Recall@3/6、MRR、NDCG@3/6 均低于 baseline，以牺牲召回率和排序质量换取精度与成本优化

## 背景与动机

RAG（Retrieval-Augmented Generation）系统的核心瓶颈之一在于文档分块（chunking）：如何将长文档切分为语义连贯、检索友好的片段。以企业知识库持续爬取网页内容为例，一篇技术博客可能包含多级标题、代码块、段落和列表，若分块不当——如在代码示例中间切断——会导致检索阶段返回语义残缺的上下文，严重损害生成质量。

现有方法形成明显的质量-成本光谱。**固定大小分块（fixed-size chunking）** 按字符或 token 数硬切分，实现简单但频繁切断语义边界，精度受限。**基于规则的结构化分块** 利用标题、段落等格式标记，但对内容密度变化适应性差，面对嵌套层级复杂的网页时规则难以覆盖。**Agentic Chunking** 将完整文本送入 LLM，由其逐字理解并重新生成语义连贯的 chunk，质量最佳却带来三重困境：输出 token 量与文档长度线性增长（成本爆炸）；LLM 生成新文本引入幻觉风险（可能篡改原始内容）；生成过程黑箱化，难以审计和复现。

核心矛盾在于：语义感知需要 LLM 理解文本，但理解文本就必须送入完整内容，这必然导致高成本。现有方法还普遍忽视网页文档的天然优势——HTML 标签、标题嵌套、段落边界等层级结构已编码大量语义边界信息，却被与 PDF、Markdown 等格式一视同仁处理。

W-RAC 的出发点是打破上述假设：语义规划不一定需要 LLM 看到原始文本，仅需结构化的位置标识符和层级元数据即可完成分组决策，从而将 LLM 从「内容生成器」降级为「轻量级分组规划器」。

## 核心创新

核心洞察：网页的 HTML 层级结构已预编码语义边界信号，LLM 仅需在这些预切分单元间做「合并决策」而非从零理解文本；将 LLM 的输出空间从「自然语言文本」约束为「有限 ID 集合的排列组合」，从而同时消除幻觉风险并将输出 token 量压缩至与文档长度解耦的极小规模，使低成本、高可观测性的语义分块成为可能。

| 维度 | Baseline（Agentic Chunking） | 本文 W-RAC |
|------|---------------------------|-----------|
| LLM 输入 | 原始完整文本 | 结构化 ID + 层级元数据（无原始文本） |
| LLM 输出 | 重新生成的自然语言 chunk | ID 分组列表（如 `["heading_1", "text_3"]`） |
| 输出 token 规模 | 与文档长度线性相关 | 与 chunk 数量相关，与文档长度解耦 |
| 幻觉风险 | 高（LLM 自由生成新文本） | 零（只能输出已有 ID） |
| 可审计性 | 低（黑箱生成） | 高（显式 ID 列表可缓存、重计算） |
| 适用范围 | 通用文档格式 | 专精 HTML 网页（利用结构先验） |

## 整体框架


![Table 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/b45e0f07-d114-433c-9cb2-ca530786324b/figures/Table_2.png)
*Table 2 (quantitative): Overview of the RAG-Multi-Corpus benchmark enterprises and domains.*



W-RAC 采用三阶段流水线，仅替换 RAG 预处理的分块子模块，不改变检索与生成阶段：

**阶段一：确定性网页解析（Deterministic Web Parsing）**
- 输入：原始 HTML 页面
- 处理：HTML → Markdown → AST 解析，对每个语义单元（标题、段落、列表项等）分配稳定的唯一 ID，记录层级关系（parent_heading）、行号、标题级别（h1/h2/h3）、token 计数等元数据
- 输出：结构化元数据表 + 原始文本保留库（文本不做任何修改）
- 关键特性：完全确定性、可复现，无 LLM 参与

**阶段二：LLM 语义分组规划（LLM Semantic Group Planning）**
- 输入：仅含 ID、层级、顺序、可选统计元数据的结构化表示（无原始文本内容）
- 处理：LLM 基于结构信号判断哪些语义单元应归属于同一 chunk
- 输出：ID 分组列表（chunk plans），例如 `[["heading_1", "text_3", "text_4"], ["heading_5", "text_6"]]`
- 关键特性：输出空间被严格限制为已有 ID 的排列组合

**阶段三：本地后处理与索引（Local Post-processing & Indexing）**
- 输入：chunk plan（ID 列表）+ 阶段一保留的原始文本库
- 处理：本地将 ID 映射回原始文本，按分组组装最终 chunk，进行嵌入（embedding）并写入向量索引
- 输出：可检索的 chunk 索引
- 关键特性：plan 可缓存、审计、重计算，无需重新处理源文本

```
HTML 页面 ──→ [解析器] ──→ 结构化元数据 (ID, 层级, 行号) 
                              ↓
原始文本库 ←── 保留 ───┐      LLM (轻量规划器)
                              ↓
                      ←── ID 分组列表 (chunk plan)
                              ↓
                         [本地映射组装]
                              ↓
                        最终 chunks → 嵌入 → 索引
```

## 核心模块与公式推导

### 模块 1: 结构化表示构造（对应框架图 阶段一→阶段二）

**直觉**：将 LLM 的输入从「高维文本空间」压缩到「低维结构空间」，利用网页 HTML 已编码的层级先验补偿信息损失。

**Baseline 公式**（Agentic Chunking）：
$$C_{base} = \text{LLM}_{\text{generate}}(T_{full}; \theta)$$
符号：$T_{full}$ = 原始完整文本，$C_{base}$ = 生成的 chunk 集合，$\theta$ = LLM 参数。LLM 需处理 $|T_{full}|$ token 并生成 $O(|T_{full}|)$ 输出 token。

**变化点**：Baseline 的输入输出均与文档长度线性相关，成本不可控；W-RAC 将输入替换为结构化元数据，输出约束为 ID 排列。

**本文公式**：
$$\text{Step 1}: \quad M = \text{Parse}(HTML) = \{(id_i, type_i, level_i, parent_i, lineno_i, tokencnt_i)\}_{i=1}^{n}$$
$$\text{（将 HTML 解析为 n 个语义单元的元数据集合，保留原始文本映射表 } T_{map}: id \mapsto text)$$

$$\text{Step 2}: \quad P = \text{LLM}_{\text{plan}}(M_{structured}; \theta) = \{G_j\}_{j=1}^{k}, \quad G_j \subseteq \{id_1, ..., id_n\}$$
$$\text{（LLM 仅接收结构化元数据，输出 k 个互不相交的 ID 分组，约束 } \text{bigcup}_j G_j = \{id_i\})$$

$$\text{Step 3}: \quad C_{final} = \{\text{Concat}(T_{map}[id] \text{mid} id \in G_j)\}_{j=1}^{k}$$
$$\text{（本地确定性映射，无 LLM 参与）}$$

**对应消融**：Table 5（Aggregate Efficiency Summary）显示完整三阶段流水线 vs 移除 LLM 规划阶段（退化回规则分块）的效率对比。

---

### 模块 2: 输出空间约束与幻觉消除（对应框架图 阶段二核心机制）

**直觉**：将 LLM 的输出空间从开放词汇表限制为闭集 ID，从根本上消除幻觉并压缩 token 量。

**Baseline 公式**：
$$P(c_t | c_{<t}, T_{full}; \theta) = \text{Softmax}(W \cdot h_t + b), \quad c_t \in \mathcal{V}_{vocab}$$
符号：$\mathcal{V}_{vocab}$ = 完整词汇表（通常 32K-128K token），$c_t$ = 第 t 个生成 token，$h_t$ = 隐藏状态。输出空间极大，LLM 可生成任意文本包括虚构内容。

**变化点**：Baseline 的输出分布覆盖全词汇表，幻觉风险高且输出长度不可控；W-RAC 通过解码约束将分布限制为有效 ID 子集。

**本文公式**：
$$\text{Step 1}: \quad \mathcal{V}_{valid} = \{id_1, id_2, ..., id_n\} \cup \{\text{`[`, `]`, `,`, ` `}\}$$
$$\text{（构造极小有效词汇表：仅含 n 个 ID 和 4 个格式符号）}$$

$$\text{Step 2}: \quad P(c_t | c_{<t}, M; \theta) = \begin{cases} \frac{\exp(w_{c_t}^T h_t)}{\sum_{v \in \mathcal{V}_{valid}} \exp(w_v^T h_t)} & \text{if } c_t \in \mathcal{V}_{valid} \\ 0 & \text{otherwise} \end{cases}$$
$$\text{（强制解码约束：非法 ID 的概率置零，实现零幻觉保证）}$$

$$\text{Step 3}: \quad |P|_{tokens} = O(k \cdot \bar{g}) \ll |T_{full}|$$
$$\text{（输出 token 量仅取决于 chunk 数 k 和平均组大小 } \bar{g}\text{，与文档长度解耦）}$$

**对应消融**：Table 7（Efficiency Improvements）显示输出 token 减少 84.6% 的量化结果；Table 6（Cost Analysis）显示总 LLM 成本降低 51.7%。

---

### 模块 3: 层级结构先验的语义补偿（对应框架图 阶段一解析逻辑）

**直觉**：HTML 的嵌套层级（h1→h2→p→ul→li）天然编码「内容归属」关系，LLM 无需阅读文本即可推断哪些单元应聚合。

**Baseline 假设**：Agentic Chunking 假设 LLM 必须理解文本语义才能正确分块，即：
$$\text{SemanticCoherence}(chunk) \propto \text{LLM}_{\text{understand}}(T_{full})$$

**变化点**：W-RAC 提出结构先验可替代部分语义理解：

**本文公式**：
$$\text{Step 1}: \quad \text{StructureScore}(id_i, id_j) = \mathbb{1}[parent_i = parent_j] \cdot \lambda^{|lineno_i - lineno_j|} + \mathbb{1}[level_i = level_j] \cdot \gamma$$
$$\text{（结构相似度：同父节点且邻近行加分，同级标题加分，} \lambda \in (0,1), \gamma > 0 \text{ 为超参）}$$

$$\text{Step 2}: \quad M_{enriched} = M \oplus \text{StructureScore}_{top\text{-}k}$$
$$\text{（将 top-k 结构邻居作为附加信号注入 LLM 输入）}$$

$$\text{Step 3}: \quad P = \text{LLM}_{plan}(M_{enriched}; \theta) \approx \text{LLM}_{generate}(T_{full}; \theta)$$
$$\text{（结构信号补偿文本缺失，使规划输出近似全文本理解效果）}$$

**对应消融**：论文未显式报告移除结构先验的消融实验。

## 实验与分析


![Table 5](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/b45e0f07-d114-433c-9cb2-ca530786324b/figures/Table_5.png)
*Table 5 (result): Aggregate Efficiency Summary*



主实验在 **RAG-Multi-Corpus benchmark**（786 条查询，覆盖多领域企业知识库，详见 Table 2、Table 3）上进行，对比 W-RAC 与未具名的 Agentic Chunking baseline：

| Metric | Baseline | W-RAC | Δ |
|--------|----------|-------|---|
| Precision@3 | 0.55 | **0.71** | +29.1% |
| Precision@6 | 0.40 | **0.56** | +40.0% |
| Recall@3 | **0.88** | 0.84 | -4.5% |
| Recall@6 | **0.93** | 0.91 | -2.2% |
| MRR | **0.87** | 0.83 | -4.6% |
| NDCG@3 | **0.88** | 0.83 | -5.7% |
| NDCG@6 | **0.89** | 0.85 | -4.5% |
| 输出 token 量 | 基准 | **-84.6%** | — |
| 端到端延迟 | 基准 | **-60%** | — |
| 总 LLM 成本 | 基准 | **-51.7%** | — |

**核心发现分析**：

1. **精度-召回权衡明确**：W-RAC 的 Precision@3/6 显著提升（+29%/+40%），支持其核心 claim——结构化 ID 规划能有效聚合语义相关单元。但 Recall@3/6、MRR、NDCG 全面低于 baseline，表明 W-RAC 以牺牲「找全」能力换取「找对」能力：当 LLM 未看到原始文本时，对隐含语义关联的跨段落引用、指代消解等复杂关系判断不足，导致部分相关 chunk 被遗漏或排序后移。

2. **效率数据需审慎解读**：Table 5（Aggregate Efficiency Summary）和 Table 7（Efficiency Improvements）汇总显示输出 token 减少 84.6%、延迟降低 60%、成本降低 51.7%。但摘要声称「一个数量级」（10×）成本降低与结论 51.7%（约 2×）存在明显矛盾；此外 84.6% 的 token 减少是「输出 token」而非「总 token」，输入 token 因需传递元数据仍有消耗。


![Table 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/b45e0f07-d114-433c-9cb2-ca530786324b/figures/Table_4.png)
*Table 4 (comparison): Token and Runtime Comparison by Organization.*



**消融与公平性检查**：
- **Baseline 强度**：仅与单一未具名 Agentic Chunking 对比，缺乏与 LangChain RecursiveCharacterTextSplitter、LlamaIndex SemanticSplitterNodeParser 等工业界主流方案的横向比较（Table 4 为 Token and Runtime Comparison by Organization，非方法对比）。
- **数据成本**：未报告 LLM 调用次数、模型规格（GPT-4/GPT-3.5？）、embedding 模型等关键复现细节。
- **失败案例**：未讨论动态渲染页面（JavaScript 重度依赖）的解析鲁棒性；对 PDF、扫描文档等非结构化格式适用性未验证。
- **指标选择性**：摘要「comparable or better retrieval performance」的表述具有选择性，实际多项指标低于 baseline。

## 方法谱系与知识库定位

**方法家族**：RAG 预处理分块（Document Chunking for RAG）

**父方法**：Agentic Chunking（LLM-as-generator 范式，如基于 LLM 的语义分块、Summarization-based chunking）。W-RAC 继承其「利用 LLM 语义理解能力做分块」的核心思想，但将 LLM 角色从生成器重构为规划器。

**改变的插槽**：
- **架构**：分块子模块内部流水线重构（三阶段解耦）
- **目标函数**：从「文本生成质量」转向「ID 分组准确率」
- **训练/推理 recipe**：LLM 输入输出格式根本改变（结构化元数据 in / ID 列表 out），无需微调，零样本提示即可
- **数据 curation**：专为 HTML 网页设计，利用 DOM 层级结构先验
- **推理约束**：强制解码约束（constrained decoding）限定输出空间

**直接对比基线**：
| 方法 | 与 W-RAC 的核心差异 |
|------|------------------|
| Fixed-size chunking | 无 LLM 参与，无语义感知，W-RAC 通过 LLM 规划引入语义性 |
| Agentic Chunking（本文 baseline） | LLM 处理完整文本并重新生成，W-RAC 仅处理元数据并输出 ID，成本与幻觉风险大幅降低 |
| Semantic chunking（LangChain/LlamaIndex） | 通常基于 embedding 相似度或规则，W-RAC 利用 HTML 层级结构先验 + LLM 轻量规划 |

**后续方向**：
1. **跨格式扩展**：将 HTML 结构先验泛化至 PDF 文档树、Markdown AST、Office XML 等半结构化格式
2. **动态页面鲁棒性**：结合浏览器渲染引擎处理 JavaScript 动态内容，提升实际网页爬取场景的解析稳定性
3. **精度-召回再平衡**：在保持成本优势的同时，通过轻量文本摘要（如仅输入首句/关键词）补偿召回率损失

**知识库标签**：
- **Modality**：文本（网页 HTML）
- **Paradigm**：RAG 预处理 / LLM-as-planner / 结构化约束生成
- **Scenario**：企业知识库、网页内容持续摄取、成本敏感的大规模文档索引
- **Mechanism**：输出空间约束（constrained decoding）、结构先验利用、任务解耦（提取-规划-组装分离）
- **Constraint**：成本优化、低延迟、可审计性、零幻觉

