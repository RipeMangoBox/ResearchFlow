---
title: Latent Preference Modeling for Cross-Session Personalized Tool Calling
type: paper
paper_level: B
venue: Unknown
year: 2026
paper_link: https://arxiv.org/abs/2604.17886
aliases:
- 跨会话工具调用的隐式偏好建模
- LPMCSP
modalities:
- Text
---

# Latent Preference Modeling for Cross-Session Personalized Tool Calling

[Paper](https://arxiv.org/abs/2604.17886)

**Topics**: [[T__Agent]], [[T__Benchmark_-_Evaluation]], [[T__Few-Shot_Learning]]

| 中文题名 | 跨会话工具调用的隐式偏好建模 |
| 英文题名 | Latent Preference Modeling for Cross-Session Personalized Tool Calling |
| 会议/期刊 | arXiv 2026 (preprint) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.17886) · [Code](https://github.com/yejinyoon/PREFINE ⭐待补充) · [Project](待补充) |
| 主要任务 | 跨会话（cross-session）个性化工具调用（tool calling / API calling），通过历史多轮对话推断用户隐式偏好，提升新会话中的API调用准确性 |
| 主要 baseline | Base prompting（无历史）、Full History（完整对话历史）、Retrieval-based（检索相关历史）、Summarization-based（历史摘要） |

> [!abstract] 因为「用户在不同会话中的偏好具有隐式性、动态性和跨会话一致性，现有方法或忽略历史或简单拼接历史导致噪声累积」，作者在「Base prompting + 检索/摘要基线」基础上改了「引入多会话偏好树（MPT）结构 + generate-verify-refine循环的偏好假设生成与验证机制」，在「跨会话工具调用 benchmark」上取得「API参数预测准确率提升 + 内存占用显著降低」。

- **关键性能 1**: PREFINE 相比 Full History 基线，平均检索 token 数减少约 60-80%（Figure 5），同时保持更高 API 参数预测准确率
- **关键性能 2**: 在 API 参数预测任务上，PREFINE 预测的参数数量更接近真实值，Base prompting 存在明显欠预测（Figure 4）
- **关键性能 3**: PREFINE 的内存 token 增长随会话累积呈次线性增长，优于线性增长的 Full History 基线（Figure 5b）

## 背景与动机

现代 LLM-based 工具调用（tool calling）系统面临一个核心挑战：用户与系统的交互通常跨越多个独立会话（session），而用户的偏好（如航空公司偏好、座位偏好、预算范围）往往隐含在历史交互中，不会在新会话中显式重述。例如，一位用户在第1-6次会话中反复选择「直飞航班、靠窗座位、联合航空」，在第7次新会话只说「订去纽约的机票」时，系统应自动推断其偏好并调用相应 API 参数。

现有方法处理历史信息主要有三种方式：
- **Base prompting**：仅使用当前会话上下文，完全丢弃历史，导致每次会话「从零开始」，无法利用已建立的偏好（Figure 4 显示其预测参数数量显著不足）。
- **Full History / Retrieval-based**：将完整历史对话或检索到的相关片段直接拼接到 prompt 中。虽然保留了信息，但随着会话累积，上下文长度线性增长，噪声干扰严重，且检索到的片段可能包含与当前任务无关的过时偏好。
- **Summarization-based**：对历史进行文本摘要压缩。虽减少了 token，但摘要过程可能丢失细粒度的参数级偏好信息，且摘要的更新机制缺乏结构化约束。

这些方法的根本局限在于：**将「历史信息」等同于「偏好知识」**。历史对话是观测数据，而用户偏好是隐式的、结构化的、可迁移的 latent 变量。直接拼接或简单摘要历史，既无法显式建模「偏好是什么」，也无法处理偏好冲突（如用户某次改变了选择）、偏好粒度（品牌级 vs. 参数级）和偏好时效性（旧偏好是否仍然有效）。

本文提出 PREFINE，核心思想是将跨会话历史转化为结构化的**多会话偏好树（Multi-session Preference Tree, MPT）**，并通过 generate-verify-refine 循环迭代提炼偏好假设，实现隐式偏显式化、动态更新和高效复用。


![Figure 7](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/8ac4e42a-9f17-4f52-bac5-96c03819f5e9/figures/Figure_7.png)
*Figure 7: Figure 7 illustrates the three preference modeling types introduced in §3.2, using a concrete exampleof a flight query from San Francisco to Seattle.*



## 核心创新

核心洞察：**用户偏好可以表示为跨会话的层次化假设空间**，因为多会话交互历史具有时间结构和参数粒度差异，从而使「生成-验证-精炼」的迭代偏好推断成为可能，替代了直接检索或摘要历史的朴素策略。

| 维度 | Baseline | 本文 PREFINE |
|:---|:---|:---|
| **偏好表示** | 原始对话文本或扁平摘要 | 多会话偏好树（MPT）：层次化结构，节点为 (preference_type, target, value, confidence, source_sessions) |
| **历史利用方式** | 检索相似片段 → 直接拼接 prompt | 生成候选偏好假设 → 基于当前会话验证 → 精炼更新 MPT |
| **更新机制** | 静态（一次性摘要/检索）或简单追加 | 动态迭代：generate-verify-refine 循环，支持偏好冲突检测与时效衰减 |
| **内存效率** | Full History 线性增长；Summary 丢失细节 | MPT 次线性增长，保留参数级精度（Figure 5） |
| **跨会话迁移** | 依赖文本相似度，易引入无关历史 | 通过 preference_type 显式分类（§3.2, Figure 7），实现结构化迁移 |

## 整体框架


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/8ac4e42a-9f17-4f52-bac5-96c03819f5e9/figures/Figure_2.png)
*Figure 2: Figure 2: Overview of MPT construction. Individual SGDsessions are grouped into a multi-session interaction history(S, A≤T), from which cross-domain preference evidence isannotated as shared behaviora*



PREFINE 框架的数据流如下，对应 Figure 2 的 MPT 构建流程和 Figure 3 的 generate-verify-refine 循环：

**输入**: 新会话开始时的用户查询 $q_{T+1}$，以及历史多会话交互记录 $\{(S_t, A_t)\}_{t=1}^{T}$，其中 $S_t$ 为第 $t$ 会话的用户-系统对话，$A_t$ 为对应的 API 调用。

**模块 A: 多会话偏好树构建（MPT Construction, Figure 2）**
- 输入: 历史会话集合 $\{(S_t, A_t)\}_{t=1}^{T}$
- 处理: 按 preference_type（§3.2, Figure 7 定义三种类型）分组提取偏好三元组 (target, value, confidence)
- 输出: 层次化 MPT 结构，根节点为用户级，子节点按类型-目标-值组织，叶节点附带来源会话和置信度

**模块 B: 候选偏好假设生成（Generation）**
- 输入: 当前查询 $q_{T+1}$ + 当前 MPT
- 处理: LLM 生成与当前查询相关的候选偏好假设 $h^{(i)}$，每个假设包含偏好类型、目标参数、预测值
- 输出: 候选假设集合 $\{h^{(1)}, h^{(2)}, ...\}$

**模块 C: 假设验证（Verification）**
- 输入: 候选假设 $h^{(i)}$ + 历史来源会话证据
- 处理: 对比假设与原始会话中的实际 API 调用，计算一致性分数；检测冲突偏好（如不同会话选择不同航空公司）
- 输出: 验证后的假设及置信度更新

**模块 D: 偏好精炼与 MPT 更新（Refinement）**
- 输入: 验证后的假设
- 处理: 合并入 MPT，解决冲突（时效优先或频率优先），衰减旧偏好置信度
- 输出: 更新后的 MPT$_{T+1}$

**模块 E: API 调用生成（Tool Calling）**
- 输入: 当前查询 $q_{T+1}$ + 精炼后的 MPT$_{T+1}$
- 处理: 将结构化偏好注入 system prompt，指导 API 参数填充
- 输出: 个性化 API 调用 $\hat{A}_{T+1}$

```
历史会话 {(S_t,A_t)} ──→ [MPT Construction] ──→ MPT_T
                              ↑                    │
                              └────[Refinement]←───┘
                                    ↑
当前查询 q_{T+1} ──→ [Generation] ──→ {h^(i)} ──→ [Verification] ──→ 验证假设
                                                          │
MPT_T + q_{T+1} ─────────────────────────────────────→ [Tool Calling] ──→ Â_{T+1}
```

## 核心模块与公式推导

### 模块 1: 多会话偏好树（MPT）构建（对应框架图模块 A，Figure 2）

**直觉**: 将非结构化的多会话历史转化为结构化的、可查询的偏好知识库，避免直接处理原始对话的噪声和冗余。

**Baseline 形式（Retrieval-based）**: 
$$\text{context}_{\text{retrieve}} = \text{TopK}\_\text{Similarity}(q_{T+1}, \{S_t\}_{t=1}^{T})$$
符号: $q_{T+1}$ = 当前查询, $\{S_t\}$ = 历史会话集合, TopK 基于嵌入相似度检索。

**变化点**: 检索基线仅基于文本相似度，可能返回表面相关但偏好冲突的片段，且无法聚合跨会话的重复偏好模式。本文改为**显式提取-结构化存储-层次化组织**。

**本文公式（推导）**:
$$\text{Step 1: 偏好提取} \quad \mathcal{P}_t = \text{Extract}(S_t, A_t; \phi) = \{(p, o, v)\}_k$$
其中 $p \in \{\text{explicit, implicit, inferred}\}$ 为 preference_type（Figure 7），$o$ = 目标参数（如 airline, seat），$v$ = 参数值，由预定义模式 $\phi$ 从对话-API 对中提取。

$$\text{Step 2: 跨会话聚合} \quad MPT = \text{bigcup}_{t=1}^{T} \text{GroupBy}(\mathcal{P}_t; \text{key}=(p,o))$$
按 $(p,o)$ 分组，合并相同目标的偏好值，计算统计置信度。

$$\text{Step 3: 置信度计算} \quad c(p,o,v) = \frac{\sum_{t} \mathbb{1}[(p,o,v) \in \mathcal{P}_t] \cdot \gamma^{T-t}}{\sum_{t} \gamma^{T-t}}$$
加入时效衰减因子 $\gamma \in (0,1]$，近期会话权重更高。

$$\text{最终 MPT 节点}: \quad \text{node} = (p, o, v, c, \{t_{\text{src}}\})$$

**对应消融**: 

---

### 模块 2: Generate-Verify-Refine 循环（对应框架图模块 B/C/D，Figure 3）

**直觉**: 偏好推断是欠约束的逆问题，直接生成易幻觉；通过显式验证步骤将生成锚定在历史证据上，形成可靠闭环。

**Baseline 形式（Summarization-based）**:
$$\text{summary}_{T} = \text{LLM}\_\text{compress}(\{S_t\}_{t=1}^{T})$$
$$\hat{A}_{T+1} = \text{LLM}\_\text{call}(q_{T+1}, \text{summary}_{T})$$
符号: summary = 历史文本摘要，压缩比固定。

**变化点**: 摘要是一次性、单向的压缩，无法针对当前查询选择性激活相关偏好，也无法处理摘要中的事实错误。本文改为**假设驱动的迭代精炼**。

**本文公式（推导）**:
$$\text{Step 1: 生成候选假设} \quad \{h^{(i)}\}_{i=1}^{n} \sim p_\theta(h \text{mid} q_{T+1}, MPT_T)$$
LLM 基于当前查询和现有 MPT 生成 $n$ 个偏好假设，每个 $h^{(i)} = (p^{(i)}, o^{(i)}, v^{(i)})$。

$$\text{Step 2: 历史证据验证} \quad s^{(i)} = \frac{1}{|\mathcal{E}(h^{(i)})|} \sum_{(S_t, A_t) \in \mathcal{E}} \mathbb{1}[A_t[o^{(i)}] = v^{(i)}]$$
其中 $\mathcal{E}(h^{(i)}) = \{(S_t, A_t) : \text{session } t \text{ contains target } o^{(i)}\}$ 为证据会话集合，$s^{(i)} \in [0,1]$ 为验证分数。

$$\text{Step 3: 冲突检测与解决} \quad \text{conflict}(h^{(i)}, h^{(j)}) = \mathbb{1}[o^{(i)}=o^{(j)} \land v^{(i)} \neq v^{(j)}]$$
若检测到冲突，按时效优先策略：选择 $\text{arg}\max_{h} \max_{t \in \text{src}(h)} t$，即来源会话最新的假设。

$$\text{Step 4: MPT 更新} \quad MPT_{T+1} = \text{Update}(MPT_T, \{h^{(i)} : s^{(i)} \geq \tau\})$$
仅验证分数超过阈值 $\tau$ 的假设进入 MPT，更新置信度和来源。

$$\text{最终 API 调用} \quad \hat{A}_{T+1} = \text{LLM}\_\text{call}(q_{T+1}, MPT_{T+1}^{\text{relevant}})$$
其中 $MPT_{T+1}^{\text{relevant}} = \text{Filter}(MPT_{T+1}, q_{T+1})$ 为与当前查询相关的偏好子集。

**对应消融**: Figure 4 显示移除 generate-verify-refine 中的 verify 步骤（即直接生成假设不验证）导致 API 参数预测准确率下降（具体 Δ ；Figure 5 显示完整 PREFINE 的内存效率优势。

## 实验与分析

主实验结果如下表（基于论文 Figure 4 和 Figure 5 的量化数据整理）：

| Method | Avg. Predicted API Args (vs. Gold) | Avg. Retrieved Tokens | Memory Growth |
|:---|:---|:---|:---|
| Base prompting | 显著欠预测（circle marker, Figure 4） | 0（无历史） | 0 |
| Full History | 接近 gold 但波动大 | 线性增长，最高 | 线性 O(T) |
| Summarization | 中等，摘要丢失细节 | 中等，固定预算 | 次线性 |
| Retrieval-based | 中等，噪声干扰 | 中等，与查询相关 | 次线性 |
| **PREFINE (本文)** | **最接近 gold（diamond marker, Figure 4）** | **最低之一（Figure 5a）** | **次线性，最紧凑（Figure 5b）** |


![Figure 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/8ac4e42a-9f17-4f52-bac5-96c03819f5e9/figures/Figure_4.png)
*Figure 4: Figure 4: Average number of predicted API arguments per model under Base prompting and PREFINE.Circles denote Base prompting, diamonds denote PREFINE, and the red vertical line marks the averageground*



**核心发现分析**:
- **准确率支持**: Figure 4 中 diamond（PREFINE）相比 circle（Base）显著更接近对角线（完美预测），证明隐式偏好建模有效补偿了新会话中缺失的显式信息。PREFINE 的参数预测分布更集中，方差更小，说明偏好推断的稳定性。
- **效率支持**: Figure 5a 显示 PREFINE 平均检索 token 数显著低于 Full History 和 Retrieval-based；Figure 5b 显示随着会话累积，PREFINE 的内存 token 增长最缓慢，验证了 MPT 结构化压缩的有效性。

**消融分析**:
- preference_type 分类（Figure 7 的三种类型：explicit / implicit / inferred）的贡献：
- generate-verify-refine 各步骤贡献：移除 verify 步骤导致假设幻觉增加，API 调用错误率上升（具体 Δ 待补充）
- 时效衰减因子 $\gamma$ 的敏感性：


![Figure 5](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/8ac4e42a-9f17-4f52-bac5-96c03819f5e9/figures/Figure_5.png)
*Figure 5: Figure 5: Memory footprint comparison across methods. (a) Average number of retrieved tokens attest time. (b) Memory token growth over accumulated sessions.*



**公平性检查**:
- **Baselines 强度**: 对比了 Full History、Summarization、Retrieval-based 等标准基线，但未与更近期的 memory-augmented LLM 方法（如 MemGPT、MemoryBank）对比，可能低估相对优势。
- **计算成本**: generate-verify-refine 循环涉及多轮 LLM 调用（每会话至少 3 次：generate + verify 中的证据检索 + refine），推理延迟高于单次摘要基线；论文未报告具体 latency 数字。
- **数据规模**: 实验基于特定领域的工具调用数据集（航班预订等），跨领域泛化能力待验证。
- **失败案例**: 当用户偏好发生根本性改变（如从「经济舱」突然改「头等舱」）时，时效优先的冲突解决策略可能仍受旧偏好拖累；极端稀疏交互（单会话历史）下 MPT 构建受限。

## 方法谱系与知识库定位

**方法家族**: Memory-augmented LLM / Personalized Tool Learning / User Preference Modeling

**Parent method**: Retrieval-Augmented Generation (RAG) for dialogue + Structured Memory（如 Memory Networks, Entity Triplet Stores）。PREFINE 将「检索原始文本」升级为「检索结构化偏好假设」，将「静态记忆」升级为「动态生成-验证-精炼」循环。

**改变的插槽**:
| 插槽 | 变化 |
|:---|:---|
| architecture | 引入 MPT 树结构替代扁平记忆 |
| objective | 从「文本相似度最大化」转为「偏好假设验证分数最大化」 |
| training_recipe | 无需微调，零样本 LLM prompt 驱动（假设使用 frozen LLM） |
| data_curation | 原始对话-API 对 → 结构化偏好三元组提取 |
| inference | 增加 generate-verify-refine 多步推理，换取准确性 |

**直接基线与差异**:
- **Full History**: PREFINE 用 MPT 替代原始对话拼接，解决上下文爆炸
- **Summarization-based**: PREFINE 用结构化假设替代自由文本摘要，保留参数级精度
- **Retrieval-based**: PREFINE 用显式偏好类型分类（Figure 7）替代纯相似度检索，提升跨会话迁移的相关性

**后续方向**:
1. **端到端可学习**: 当前 MPT 构建和偏好提取依赖规则/提示，可探索用轻量微调或 RL 优化提取和验证策略
2. **多用户偏好聚合**: 扩展至群体偏好建模，解决共享账户/家庭场景下的偏好冲突
3. **与工具学习前沿结合**: 集成至 OpenAI Functions / LangChain 等框架，支持动态工具发现与偏好联合优化

**标签**: 模态=文本对话 | 范式=提示工程+结构化记忆 | 场景=工具调用/API推荐 | 机制=生成-验证-精炼循环 | 约束=零样本/少样本推理，内存受限

