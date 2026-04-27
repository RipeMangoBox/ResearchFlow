---
title: "Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval for Large Language Models"
venue: arXiv
year: 2025
tags:
  - Survey_Benchmark
  - task/tool-retrieval
  - instruction-aware-retrieval
  - heterogeneous-benchmark
  - completeness-metric
  - dataset/TOOLRET
  - dataset/TOOLRET-train
  - opensource/promised
core_operator: 将分散的工具使用数据统一重构为带指令的大规模查询-工具检索基准，并用完整性与下游通过率联动评测真实工具检索能力
primary_logic: |
  评测真实工具检索需求 → 汇聚34个数据集并统一成“查询/指令→工具文档”检索任务，构建43k工具语料与7.6k测试任务 → 用NDCG/Recall/Precision/Completeness及下游Pass Rate进行评测 → 揭示现有IR模型在低词汇重叠、多工具、异构工具场景中的能力缺口
claims:
  - "在TOOLRET上，常规IR中表现很强的检索器仍显著失效；作者报告无指令设置下强双编码器 NV-Embed-v1 的平均 nDCG@10 仅为 33.83，说明通用文本检索能力不能直接迁移到工具检索 [evidence: analysis]"
  - "给查询附加 target-aware instruction 后，所有被测IR模型的检索效果均提升，表明工具检索对显式任务意图建模高度敏感 [evidence: analysis]"
  - "使用 TOOLRET-train（20万+实例）训练检索器后，检索指标与 ToolBench 端到端 Pass Rate 同时上升，作者报告下游通过率可提升约 10%-20% [evidence: comparison]"
related_work_position:
  extends: "MTEB (Muennighoff et al. 2022)"
  competes_with: "COLT (Qu et al. 2024a); MAIR (Sun et al. 2024)"
  complementary_to: "ToolGen (Wang et al. 2024c); Chain of Tools (Shi et al. 2024)"
evidence_strength: moderate
pdf_ref: paperPDFs/Agentic_Reasoning_Benchmarks_Core_Mechanisms_of_Agentic_Reasoning_Tool_Use_Single_Turn_Tool_Use/arXiv_2025/2025_Retrieval_Models_Aren't_Tool_Savvy_Benchmarking_Tool_Retrieval_for_Large_Language_Models.pdf
category: Survey_Benchmark
---

# Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval for Large Language Models

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2503.01763)
> - **Summary**: 论文把工具使用前最容易被忽略的“先找对工具”步骤单独抽出来，构建了一个 7.6k 查询、43k 工具的异构检索基准 TOOLRET，并证明现有强 IR 模型在真实工具检索上明显失灵，进而拖累 LLM agent 的端到端任务成功率。
> - **Key Performance**: 无指令时强双编码检索器 NV-Embed-v1 平均仅 33.83 NDCG@10 / 32.12 C@10；用 TOOLRET-train 训练后，ToolBench 上工具使用 Pass Rate 可提升约 10%-20%。

> [!info] **Agent Summary**
> - **task_path**: 英文文本查询（可附加任务指令）/大规模异构工具库 -> Top-K 工具ID与文档
> - **bottleneck**: 现有IR模型擅长“找语义相关文本”，但不擅长在低词汇重叠、功能相近、常需多工具组合的场景中对齐“用户意图→可执行工具”
> - **mechanism_delta**: 作者把30+工具使用数据集统一成 instruction-aware 的工具检索评测，并加入 Completeness 与下游 Pass Rate 联动，从而直接测“是否把真正需要的工具集找全”
> - **evidence_signal**: 在 7.6k 任务 / 43k 工具上，多类强IR模型分数普遍偏低，且 retrieval 分数与 ToolBench 端到端 pass rate 同步变化
> - **reusable_ops**: [heterogeneous-dataset-unification, target-aware-instruction-generation]
> - **failure_modes**: [low-lexical-overlap semantic mismatch, similar-tools one-to-many ambiguity]
> - **open_questions**: [multilingual-and-multimodal-tool-retrieval, interleaved-retrieval-and-calling-evaluation]

## Part I：问题与挑战

这篇论文抓住的真实问题，不是“LLM 会不会调用工具”，而是更前面的那一步：**面对大规模工具库，系统能不能先把真正有用的工具找出来**。

### 1) 真正的瓶颈是什么？
现有很多 tool-use benchmark，会先人为给每个任务配一个很小的候选工具集，通常只有 10-20 个工具。这样做虽然方便评测，但会把最难的部分提前做掉：  
**真实世界的困难不是从 10 个工具里选 1 个，而是从上万相似工具里先召回正确工具集。**

因此，过去不少 benchmark 更像是在测：
- 小候选集内的最终选择；
而不是在测：
- 大工具库下的真实工具检索。

### 2) 为什么现在必须解决？
因为 agent 场景正在快速走向大工具库：
- RapidAPI 上有 52k+ 工具；
- PyPI 上有 600k+ 高频更新 package；
- LLM 上下文长度有限，不可能把所有工具文档都塞进 prompt；
- 工具持续更新，靠不断微调 LLM 去“记住工具”成本过高。

所以，**tool retrieval 已经从工程细节变成了 agent 系统的前置能力瓶颈**。

### 3) 输入/输出接口是什么？
论文把问题统一成一个标准 IR 任务：

- **输入**：用户查询 `query`，可选附加任务指令 `instruction`
- **候选库**：43,215 个工具文档
- **输出**：Top-K 工具列表
- **标签**：一个查询可能对应多个目标工具

这点很关键。TOOLRET 中平均每个查询有 **2.17 个目标工具**，因此它不只是“找最相关的一个”，而是要尽量**找全**完成任务所需的工具集合。

### 4) 为什么比传统 IR 更难？
作者给出的难点很具体：

- **低词汇重叠**：query 与 target tool doc 的 ROUGE-L 平均只有 **0.06**，显著低于 NQ / MS-MARCO 等传统 IR 基准。  
  这意味着模型不能只做 surface matching，而要真正理解“用户意图”与“工具功能”的对应关系。
- **异构工具表示**：工具文档覆盖三种主流格式：Web API、Code Function、Customized App。  
  也就是要跨 JSON API 说明、代码函数、自然语言描述三种分布做匹配。
- **多工具需求**：不少任务不是单工具可解，top-K 里漏掉任何一个关键工具，都可能让下游 agent 失败。

### 5) 边界条件
这个 benchmark 的适用边界也很明确：

- 主要是 **英文、文本检索**；
- 采用 **retrieval-then-calling** 的单次先检索后调用设定；
- 即使源数据里包含多模态工具，评测本身仍将工具检索视为文本 IR 问题；
- ground truth 继承自原始数据集，因此在功能相近工具很多时会存在 one-to-many 模糊性。

---

## Part II：方法与洞察

### 1) 作者到底做了什么？
这篇论文的主贡献不是提出一个新 retriever，而是把“工具检索”从隐含步骤变成了一个**可独立评测、可诊断、可训练**的问题。

#### a. 异构数据汇聚
作者从论文 benchmark、conference resource、HuggingFace 等来源汇聚了 30+ 数据集，最终整合成 **34 个数据源**，覆盖多种工具使用场景。

最后构成：
- **7,615 个检索任务**
- **43,215 个工具**
- 三个子集：**Web API / Code Function / Customized App**

#### b. 统一成检索格式
每个样本被标准化为：
- 一个 `query`
- 一组 `target tools`
- 一个统一工具语料库

也就是说，原本“嵌在 agent 流程里的工具选择问题”，被重写成类似 MTEB / BEIR 的标准 retrieval task。

#### c. 采样与去重
作者没有直接把所有样本粗暴拼接，而是做了两层处理：

- **任务采样**：用 NV-Embed-v1 编码后做 K-means，减少大数据集中的冗余查询
- **工具集去重**：人工检查不同数据集间的重叠工具，并合并相同/高度重叠工具集

这个处理的作用是：既保留 benchmark 的广覆盖，又避免评价过度被重复样本主导。

#### d. 构造 instruction-aware retrieval
这是论文最关键的设计之一。

作者不满足于只给 query，还为每个任务构造一条 instruction，用来显式描述：
- 用户到底要完成什么任务；
- 目标工具应该具备哪些功能特征。

构造流程是：
1. 3 位专家手写 100 条 seed instructions；
2. 用 GPT-4o 基于 query + target tools 自动生成 instruction；
3. 5 位专家做人审与修订。

质量控制结果也给出来了：
- 90.1% instruction 与原 query 相关；
- 92.3% 能描述 target tool 特征；
- 89.2% 能覆盖全部 target tools 的功能；
- 5.9% 被标为存在 hallucination，随后人工修正。  
  标注一致性 Kappa 为 **0.743**。

#### e. 评测协议
作者在两个设定下评测：
- **w/o inst.**：只用 query 检索
- **w/ inst.**：用 query + instruction 检索

使用的指标除了标准 IR 的：
- NDCG@K
- Recall@K
- Precision@K

还专门加入了 **Completeness@K**：
- 只有当 **所有目标工具** 都出现在 top-K 里，才记为成功。

这非常贴合 tool-use 任务，因为很多任务的真正要求不是“命中一个像的工具”，而是**把完成任务所需的工具组尽量找全**。

#### f. 训练资源 TOOLRET-train
作者还进一步发布了一个训练集 **TOOLRET-train**：
- 超过 20 万条样本
- 每条包含 query、instruction、target tools、以及由现有强检索器挖出的 hard negatives

这样 benchmark 不只用来“证明大家不行”，还能直接用来提升工具检索模型。

### 核心直觉

这篇论文最重要的洞察，不是模型结构层面的，而是**测量瓶颈的显式化**。

#### 改了什么？
从：
- 小规模、人工预筛过的候选工具集；
- 更偏“最后一步工具选择”；

变成：
- 大规模、异构、相似工具密集的统一工具库；
- 同时评估排序质量、**多工具完整召回**、以及对下游 pass rate 的影响。

#### 改变了哪个瓶颈？
它改变的是 benchmark 隐藏掉的那层分布与信息约束：

- **候选分布变了**：从小 shortlist 变成 43k heterogeneous corpus
- **信息瓶颈变了**：低词汇重叠迫使模型做“任务意图 ↔ 工具功能”匹配，而不是词面匹配
- **成功标准变了**：从 relevance-oriented ranking，转为更接近 agent 需求的 completeness-oriented retrieval

#### 为什么这会起作用？
因为很多 tool-use failure 并不是“LLM 不会调用工具”，而是**一开始就没拿到正确工具集**。

这个 benchmark 的设计，把系统失败因果链拆清楚了：

- 如果 oracle 工具集下 agent 很强，但检索工具集下明显掉点，问题就在 retrieval；
- 如果 retrieval 提升后 pass rate 也同步上升，说明检索确实是上游瓶颈。

这让系统优化路径更明确：  
**先解决找工具，再谈调 planner / caller。**

### 战略权衡

| 设计选择 | 收益 | 代价 / 风险 |
|---|---|---|
| 合并 34 个数据源为 43k 工具统一语料 | 更接近真实工具生态，能暴露跨域、长尾、相似工具干扰 | 会引入 one-to-many 标注歧义：有些“也能用”的工具未被标为真值 |
| 给每个 query 生成 target-aware instruction | 能显式测 instruction-following retrieval，更接近 agent 给 retriever 传上下文的真实形态 | 依赖 GPT-4o 与人工复核，存在闭源 API 与人工成本 |
| 引入 Completeness@K | 更能衡量多工具任务是否“找全” | 对标签完整性和 top-K 设定更敏感 |
| 发布 TOOLRET-train 并挖 hard negatives | 让 benchmark 不只诊断问题，还能直接支持模型改进 | 负样本分布受初始检索器影响，可能继承既有偏差 |

---

## Part III：证据与局限

### 关键证据信号

- **信号 1｜现有强检索器在真实工具检索上明显失效**  
  无指令设定下，作者报告强双编码检索器 **NV-Embed-v1 仅有 33.83 NDCG@10 / 32.12 C@10**。  
  这说明通用文本检索能力并不能直接迁移到工具检索。论文还指出，一些在常规 IR 很强的模型，如 ColBERT，在这里甚至不如 BM25 稳定。

- **信号 2｜instruction 是有效因子，不是装饰项**  
  几乎所有模型在 `w/ inst.` 设定下都更好。  
  例如 **NV-Embed-v1 从 33.83 提升到 42.71 NDCG@10，C@10 从 32.12 提升到 43.41**。  
  这说明工具检索高度依赖显式任务意图建模，query-only semantic search 不够。

- **信号 3｜大统一工具库才是真正难点**  
  附录控制实验显示：如果只在各原始数据集自己的局部 toolset 上评测，最佳 reranker 可到 **74.14 NDCG@10 / 78.09 C@10**；  
  一旦放到统一 43k 工具库上，最佳 `w/ inst.` 成绩只有 **47.52 / 48.90**。  
  这直接支持作者的核心论点：**真实困难来自跨数据源、相似工具密集、候选极大的现实检索分布**。

- **信号 4｜retrieval 质量会直接传导到 agent 的端到端成功率**  
  在 ToolBench 上，把官方预标注工具集换成检索返回工具后，tool-use LLM 的 pass rate 明显下降。  
  一个具体例子是：**ToolBench-G1 上 GPT-3.5 从 oracle 的 62.0 降到使用 bge-large 检索工具时的 50.6**。  
  更重要的是，用 **TOOLRET-train** 训练 retriever 后，检索分数和下游 pass rate 会同步上涨，作者报告整体可带来 **约 10%-20%** 的端到端增益。  
  这是全文最强的“so what”信号：**tool retrieval 是 agent performance 的前置因子。**

- **信号 5｜TOOLRET 与传统 IR 相关，但显著更难**  
  与 MTEB 检索子集相比，模型排序趋势仍有相关性（论文给出 Pearson 0.790），但 TOOLRET 上绝对分数普遍更低。  
  含义是：传统 IR 能力并非没用，但**远远不够**；工具检索还需要更强的意图理解与目标工具对齐能力。

### 局限性

- **Fails when**: 需要多语言检索、图像/音频工具检索、或 retrieval 与 reasoning / calling 交替进行的场景；在功能高度相似的工具密集区域，也会受到 one-to-many 标注歧义影响。
- **Assumes**: 英文文本查询与文本化工具文档；retrieval-then-calling 的单次 top-K 工作流；instruction 由 GPT-4o 生成并依赖 3 位专家写种子、5 位专家复核；训练负样本由现有检索器挖掘，因此会带入现有模型偏差。
- **Not designed for**: 动态在线工具市场中的实时更新、延迟/成本/权限约束下的工具路由，也不直接评估工具调用后的执行质量、规划质量或安全性。

### 可复用组件

- **TOOLRET benchmark**：可以直接替换“小候选工具集”式评测的大规模工具检索测试床。
- **target-aware instruction generation pipeline**：把 query 与 target tool 功能压成 retrieval instruction 的流程。
- **Completeness@K**：适合多工具任务的补充指标。
- **TOOLRET-train**：用于 instruction-aware tool retrieval 的 20 万+ 训练资源。
- **下游联动评测范式**：先看 retrieval，再看 tool-use pass rate，用于拆解 agent 系统瓶颈。

## Local PDF reference

![[paperPDFs/Agentic_Reasoning_Benchmarks_Core_Mechanisms_of_Agentic_Reasoning_Tool_Use_Single_Turn_Tool_Use/arXiv_2025/2025_Retrieval_Models_Aren't_Tool_Savvy_Benchmarking_Tool_Retrieval_for_Large_Language_Models.pdf]]