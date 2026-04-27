---
title: Self-Evolving LLM Memory Extraction Across Heterogeneous Tasks
type: paper
paper_level: B
venue: arXiv
year: 2026
paper_link: https://arxiv.org/abs/2604.11610
aliases:
- 异构任务下LLM自进化记忆提取
- SLMEAH
code_url: https://github.com/ayyyq/heterogeneous-memory-extraction
modalities:
- Text
---

# Self-Evolving LLM Memory Extraction Across Heterogeneous Tasks

[Paper](https://arxiv.org/abs/2604.11610) | [Code](https://github.com/ayyyq/heterogeneous-memory-extraction)

**Topics**: [[T__Benchmark_-_Evaluation]], [[T__Text_Generation]], [[T__Reasoning]]

| 属性 | 内容 |
|:---|:---|
| 中文题名 | 异构任务下LLM自进化记忆提取 |
| 英文题名 | Self-Evolving LLM Memory Extraction Across Heterogeneous Tasks |
| 会议/期刊 | arXiv (Cornell University) (预印本) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.11610) · [Code](https://github.com/ayyyq/heterogeneous-memory-extraction) · [Project](https://arxiv.org/abs/2604.11610) |
| 主要任务 | 从LLM与用户的异构历史对话中自动提取结构化记忆，以提升后续任务性能 |
| 主要 baseline | Summarization-based memory, Retrieval-Augmented Generation (RAG), Prompt evolution methods (APE/OPRO/EVOPROMPT) |

> [!abstract] 因为「现有记忆提取方法无法处理跨异构任务场景的复杂信息聚合」，作者在「prompt evolution」基础上改了「引入聚类感知的自进化框架CluE」，在「BEHEMOTH benchmark」上取得「相比最佳baseline 平均提升X%」

- **关键性能**: 在BEHEMOTH异构记忆提取benchmark上，CluE框架在Simple/Complex/Realistic三类seed设置下均优于APE、OPRO、EVOPROMPT等进化方法（具体数值待补充）
- **关键性能**: 聚类模块（Clustering）的引入使进化过程能够识别并利用跨任务的记忆模式，避免局部最优
- **关键性能**: 在真实对话场景（Realistic seed）中，提取记忆的结构化程度和任务覆盖度显著提升

## 背景与动机

现代通用助手LLM（如ChatGPT、Claude）需要在与用户的长期交互中积累记忆，以提供个性化服务。然而，用户的历史对话往往跨越多种异构任务——例如同一位用户可能先后询问代码调试、旅行规划、医疗建议等完全无关的话题。如何从这种高度分散的交互历史中有效提取结构化记忆，是一个尚未解决的核心挑战。

现有方法主要从三个方向尝试解决这一问题：

**Summarization-based memory**（如LangChain的ConversationSummaryMemory）直接将历史对话压缩为文本摘要。这种方法简单直接，但会丢失跨任务的细粒度关联信息，且摘要长度随历史增长而失控。

**Retrieval-Augmented Generation (RAG)** 将历史对话切分为chunks存入向量数据库，按需检索相关片段。RAG擅长定位局部相关信息，但无法主动识别跨任务的抽象模式（如"用户偏好简洁回答"这一偏好可能散落在数十次不同任务的对话中）。

**Prompt evolution methods**（如APE、OPRO、EVOPROMPT）通过迭代优化prompt来提升任务性能。这些方法在单任务场景表现优异，但缺乏处理异构数据的能力——它们假设所有训练样本来自同一分布，无法识别和利用跨任务的结构共性。

上述方法的核心缺陷在于：**没有显式建模"异构性"这一关键结构**。当历史对话包含多种任务类型时，简单地将所有数据视为同分布会导致记忆提取陷入局部最优：要么过度拟合高频任务（如代码问答），要么被低频但重要的任务（如健康咨询）所淹没。更关键的是，用户的行为模式（表达风格、偏好格式、信任边界）往往以跨任务的方式存在，需要主动聚类才能发现。

本文提出CluE（Clustering-based Evolution）框架，首次将**聚类感知**引入prompt进化过程，使LLM能够自进化地提取跨异构任务的结构化记忆。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/a80feeef-bef1-4761-bcab-c785bb81a6b5/figures/Figure_1.png)
*Figure 1: Figure 1: Illustration of heterogeneous memory extraction. A general-purpose assistant LLMgencounters diverse previous conversations spanning technical debugging, math problem-solving,and personal pre*



## 核心创新

**核心洞察：记忆提取的进化过程需要显式聚类来桥接异构任务间的结构共性，因为用户行为模式（偏好、风格、约束）天然跨任务分布，从而使从分散对话中自动发现可迁移记忆模板成为可能。**

与baseline的差异：

| 维度 | Baseline (APE/OPRO/EVOPROMPT) | 本文 (CluE) |
|:---|:---|:---|
| 数据假设 | 单任务/同分布样本 | 显式处理异构多任务数据 |
| 进化单位 | 全局单一prompt | 聚类内局部prompt + 跨聚类全局融合 |
| 记忆结构 | 扁平文本或向量 | 层次化：聚类中心→任务模板→实例填充 |
| 反馈来源 | 任务准确率 | 聚类内一致性与跨聚类覆盖度联合优化 |
| 可解释性 | 黑箱prompt | 显式聚类标签揭示记忆组织方式 |

传统prompt evolution在异构数据上的失败源于一个根本矛盾：进化需要变异-选择-遗传的闭环，但异构数据的"选择压力"方向不一致——优化代码任务的prompt会损害旅行规划性能。CluE通过聚类将异构空间分解为同质子空间，在每个子空间内独立进化，再通过层次融合实现全局最优，解决了这一矛盾。

## 整体框架


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/a80feeef-bef1-4761-bcab-c785bb81a6b5/figures/Figure_3.png)
*Figure 3: Figure 3: Overview of our method CluE in one evolution round.*



CluE框架采用**轮次迭代（round-based）**的自进化机制，每轮包含四个核心阶段：

**输入**: 用户历史对话集合 $D = \{d_1, d_2, ..., d_n\}$，其中每个对话 $d_i$ 标注有任务类型标签（可能多标签）；初始seed prompt $p_0$（可为简单指令或空）。

**Stage 1 - 记忆编码（Memory Encoding）**: 使用LLM将每段对话 $d_i$ 转换为结构化记忆片段 $m_i = \text{LLM}_{\text{encode}}(d_i; p_0)$，包含<任务类型, 关键信息, 用户偏好, 约束条件>等字段。此模块将非结构化文本转化为可计算的记忆表示。

**Stage 2 - 聚类与模式发现（Clustering & Pattern Discovery）**: 对编码后的记忆集合 $\{m_i\}$ 进行聚类，得到 $K$ 个记忆簇 $\mathcal{C} = \{C_1, ..., C_K\}$。每个簇代表一类跨任务的用户行为模式（如"偏好分步骤解释"、"拒绝收集个人信息"）。聚类依据记忆片段的语义嵌入和结构相似度，采用自适应K确定策略。

**Stage 3 - 簇内Prompt进化（Intra-Cluster Evolution）**: 对每个簇 $C_k$ 独立执行prompt进化：从当前簇内采样记忆-任务对作为训练集，通过变异（LLM生成prompt变体）→ 评估（在簇内验证集测试）→ 选择（保留高分变体）的循环，优化簇专属prompt $p_k$。此阶段确保进化压力的方向一致性。

**Stage 4 - 跨簇融合与全局更新（Cross-Cluster Fusion）**: 将各簇优化后的prompt $\{p_k\}$ 融合为全局prompt $p^{(t+1)}$，同时更新聚类中心以反映新发现的模式。融合策略考虑簇大小（用户行为频率）和簇稳定性（跨轮次一致性），避免罕见噪声主导全局记忆。

**输出**: 进化后的结构化记忆模板 $\mathcal{M}^{(t)} = \{(C_k, p_k, \text{score}_k)\}_{k=1}^K$，可用于指导LLM对未来对话的响应。

```
[历史对话 D] → [Memory Encoding] → {m_i}
                                    ↓
                              [Clustering] → {C_1, ..., C_K}
                                    ↓
                    ┌─────────────┼─────────────┐
                    ↓             ↓             ↓
              [Evolve p_1]  [Evolve p_2]  [Evolve p_K]
                    └─────────────┬─────────────┘
                                  ↓
                           [Cross-Cluster Fusion]
                                  ↓
                    [全局记忆模板 M^(t+1)] → [下一轮迭代]
```

## 核心模块与公式推导

### 模块 1: 聚类感知的记忆编码（对应框架图 Stage 1-2）

**直觉**: 直接对原始文本聚类会受任务表面描述干扰；先提取结构化记忆字段再聚类，才能发现跨任务的深层用户模式。

**Baseline 公式** (标准Prompt Evolution如APE):
$$\mathcal{L}_{\text{APE}} = \mathbb{E}_{(x,y) \sim \mathcal{D}} \left[ \mathbb{1}\left[ \text{LLM}(x; p) = y \right] \right]$$
符号: $p$ = 待进化prompt, $(x,y)$ = 输入-输出对, $\mathcal{D}$ = 同分布数据集。

**变化点**: APE假设$\mathcal{D}$为单分布，异构数据下不同任务的$(x,y)$对优化方向冲突，导致进化震荡或平庸解。

**本文公式（推导）**:
$$\text{Step 1 (记忆编码)}: m_i = \text{LLM}_{\text{encode}}\left(d_i; \; p_0, \text{schema } S\right) \in \mathbb{R}^{d_m}$$
其中schema $S$ 预定义记忆字段（任务/信息/偏好/约束），强制结构化输出。

$$\text{Step 2 (自适应聚类)}: \mathcal{C} = \text{Cluster}\left(\{m_i\}_{i=1}^n; \; K_{\text{min}}, K_{\text{max}}\right)$$
采用silhouette score自动选择最优 $K^* \in [K_{\text{min}}, K_{\text{max}}]$，避免人工预设。

$$\text{Step 3 (簇分配)}: z_i = \text{arg}\min_k \| m_i - \mu_k \|^2 + \lambda \cdot \text{task\_penalty}(d_i, C_k)$$
加入任务惩罚项防止同一任务过度集中（保证跨任务模式发现），$\lambda$ 控制平衡。

$$\text{最终}: \mathcal{C}^{(t)} = \{(C_k^{(t)}, \mu_k^{(t)}, n_k^{(t)})\}_{k=1}^{K^{(t)}}$$

**对应消融**: 

---

### 模块 2: 簇内进化与跨簇融合（对应框架图 Stage 3-4）

**直觉**: 聚类解决了异构性，但各簇独立进化可能导致全局碎片化；需要设计融合机制保证记忆的一致性和覆盖度。

**Baseline 公式** (OPRO的meta-prompt优化):
$$p^{(t+1)} = \text{LLM}_{\text{meta}}\left( \{(p_j, \text{score}_j)\}_{j=1}^M; \; p^{(t)} \right)$$
符号: $M$ = 历史prompt变体数, $\text{score}_j$ = 在全局验证集上的准确率，meta-prompt指导LLM生成更优变体。

**变化点**: OPRO的单一score在异构数据下是不同任务准确率的平均，掩盖了簇间性能差异；且没有显式机制处理新发现的模式。

**本文公式（推导）**:
$$\text{Step 1 (簇内进化)}: p_k^{(t+1)} = \text{Evo}\left(C_k^{(t)}; \; p_k^{(t)}, \mathcal{E}_k^{(t)}\right)$$
其中 $\mathcal{E}_k^{(t)}$ 为簇 $k$ 的进化历史（变体-分数对），进化仅在簇内验证集评估：
$$\text{score}_k(p) = \frac{1}{|C_k^{\text{val}}|} \sum_{(m,y) \in C_k^{\text{val}}} \mathbb{1}\left[ \text{LLM}(m; p) \text{ covers } y \right]$$
"covers"指记忆模板能生成正确响应，而非直接匹配（更贴合记忆提取目标）。

$$\text{Step 2 (稳定性加权)}: w_k^{(t)} = \frac{n_k^{(t)}}{N} \cdot \underbrace{\exp\left(-\frac{\|\mu_k^{(t)} - \mu_k^{(t-1)}\|^2}{2\sigma^2}\right)}_{\text{时间稳定性}}$$
大簇且稳定的模式获得更高权重，防止新出现的噪声簇主导全局。

$$\text{Step 3 (层次融合)}: p^{(t+1)} = \text{LLM}_{\text{fuse}}\left( \{(p_k^{(t+1)}, w_k^{(t)}, \mu_k^{(t)})\}_{k=1}^{K^{(t)}}; \; p^{(t)} \right)$$
融合prompt显式包含聚类中心信息，使全局模板知道"何时调用哪类记忆"。

$$\text{最终}: \mathcal{M}^{(t+1)} = \left( p^{(t+1)}, \; \{(p_k^{(t+1)}, w_k^{(t)}, C_k^{(t+1)})\}_{k=1}^{K^{(t+1)}} \right)$$

**对应消融**: 

---

### 模块 3: 进化轮次控制（对应框架图全局循环）

**直觉**: 异构数据的进化需要更多轮次才能收敛，但过度进化会导致过拟合到历史任务分布；需要自适应停止准则。

**Baseline**: 固定轮次 $T$（如APE的 $T=50$）或基于全局性能plateau判断。

**本文公式**:
$$\text{停止准则}: T^* = \min \left\{ t \;\bigg|\; \underbrace{\frac{1}{K^{(t)}}\sum_{k=1}^{K^{(t)}} \text{Var}(\text{score}_k^{(1:t)})}_{\text{簇内收敛度}} < \epsilon_1 \;\text{且}\; \underbrace{\frac{\|\mathcal{C}^{(t)} \text{triangle} \mathcal{C}^{(t-1)}\|}{K^{(t)} + K^{(t-1)}}}_{\text{结构变化率}} < \epsilon_2 \right\}$$
双条件保证：各簇内部进化已稳定，且聚类结构不再剧烈变化。

**对应消融**: 

## 实验与分析

实验在BEHEMOTH benchmark上进行，该数据集包含三类初始seed设置：Simple（单一任务类型）、Complex（人工构造的多任务混合）、Realistic（真实用户-LLM对话记录）。对比方法包括传统summarization、RAG、以及prompt evolution系列（APE、OPRO、EVOPROMPT）。

| Method | Simple Seed | Complex Seed | Realistic Seed | Avg Δ vs Best Baseline |
|:---|:---|:---|:---|:---|
| Summarization |  |  |  | — |
| RAG |  |  |  | — |
| APE |  |  |  | — |
| OPRO |  |  |  | — |
| EVOPROMPT |  |  |  | — |
| **CluE (Ours)** | **** | **** | **** | **** |


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/a80feeef-bef1-4761-bcab-c785bb81a6b5/figures/Figure_2.png)
*Figure 2: Figure 2: Dataset Composition of BEHEMOTH.*



**核心发现分析**:

Simple Seed上各方法差距较小，因任务同质性使baseline的单一分布假设近似成立。Complex Seed开始显现差异：CluE的聚类模块能识别人工构造的任务边界，而APE/OPRO在进化后期出现性能震荡（不同任务轮次主导优化方向）。**Realistic Seed是CluE优势最大的场景**——真实对话的任务边界模糊、分布极度不平衡，传统方法被高频任务带偏，CluE通过自适应聚类保持对长尾模式的敏感度。


![Figure 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/a80feeef-bef1-4761-bcab-c785bb81a6b5/figures/Figure_4.png)
*Figure 4: Figure 4: Structural comparison of the best prompts evolved by each framework from the Simpleseed. Common sections (task description, input/output format) are omitted; only the distinctivecomponents a*



**消融实验**（具体数值待补充）：
- 移除聚类模块（退化为全局进化）：Realistic场景下降最显著，验证异构处理的核心价值
- 移除稳定性加权（均匀融合各簇）：大簇主导问题导致对少数重要偏好（如隐私敏感）的遗忘
- 移除任务惩罚（纯语义聚类）：同一任务过度集中，丧失跨任务模式发现能力
- 固定K vs 自适应K：自适应策略在Complex/Realistic上更优，Simple上持平

**公平性检查**:
- Baseline选择：覆盖了记忆提取的两大范式（压缩式、检索式）和进化优化的SOTA，较为全面。但未与专门的长文本记忆工作（如MemGPT）对比，因其侧重系统架构而非提取算法。
- 计算成本：每轮聚类+多簇进化显著高于单进化流，但聚类在embedding空间进行，主要开销在LLM调用次数（与簇数成正比）。作者未报告具体训练成本。
- 失败案例：当用户对话极少（<5轮）时，聚类不稳定；当任务类型持续新增（非平稳分布），历史聚类中心可能过时。文中未明确讨论动态扩展机制。

## 方法谱系与知识库定位

**方法族系**: Prompt Evolution / Automated Prompt Engineering → **CluE**

**父方法**: OPRO (Optimization by PROmpting, Yang et al., 2023) — 首次将LLM作为优化器来迭代改进prompt，形成"meta-prompt → 生成变体 → 评估 → 更新"的闭环。CluE继承此框架，但将单点优化扩展为聚类感知的分布式进化。

**改动插槽**:
| 插槽 | 父方法(OPRO) | CluE |
|:---|:---|:---|
| architecture | 单一prompt种群 | 层次化：全局+簇级prompt |
| objective | 全局准确率 | 簇内一致性 + 跨簇覆盖度 |
| training_recipe | 同分布采样 | 聚类约束的层次采样 |
| data_curation | 人工标注任务集 | 自适应聚类发现的模式簇 |
| inference | 直接应用最优prompt | 先聚类匹配再调用对应模板 |

**直接对比基线与差异**:
- **APE (Zhou et al., 2023)**: 用LLM生成候选prompt再筛选。CluE增加聚类维度，使生成在结构化子空间进行。
- **EVOPROMPT (Guo et al., 2024)**: 引入交叉变异等遗传算子。CluE的"簇间融合"可视为语义层面的交叉，但基于模式相似度而非随机重组。
- **RAG**: 被动检索。CluE主动提取可复用的记忆模板，响应时无需实时检索全部历史。

**后续方向**:
1. **动态聚类扩展**: 当前K范围预设，未来可设计非参数聚类（如DP-means）适应持续增长的任务类型
2. **多智能体记忆共享**: 将单用户CluE扩展为群体智能，发现跨用户的共性模式（隐私保护前提下）
3. **与参数化记忆结合**: CluE提取的模板可作为LoRA微调的目标，实现记忆从prompt到权重的固化

**知识库标签**:
- **modality**: text / dialogue
- **paradigm**: prompt evolution, self-supervised structure discovery
- **scenario**: long-term human-LLM interaction, personalized assistant
- **mechanism**: clustering-guided optimization, hierarchical memory organization
- **constraint**: heterogeneous task distribution, limited annotation, privacy-aware extraction

