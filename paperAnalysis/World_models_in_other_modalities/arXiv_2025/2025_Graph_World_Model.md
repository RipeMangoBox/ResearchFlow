---
title: "Graph World Model"
venue: ICML
year: 2025
tags:
  - Others
  - task/world-modeling
  - task/graph-reasoning
  - message-passing
  - action-node
  - projector-tuning
  - dataset/Goodreads
  - dataset/Cora
  - dataset/AgentClinic
  - dataset/ALFWorld
  - opensource/full
core_operator: 将多模态世界状态表示为图，并把任务写成动作节点对目标子图的查询，再通过多跳消息传递把结构化上下文送入 LLM/扩散解码器
primary_logic: |
  多模态状态与关系图 + 动作描述 → 动作节点检索目标节点并做多跳结构聚合（文本空间或 embedding 空间）→ 统一解码为下一状态的文本、图像、链接或决策
claims:
  - "同一个 GWM 在 6 类任务上与领域专用方法持平或更优；例如 LongBench v2 上 GWM-E 仅用 2k 上下文取得 33.32 准确率，高于 GPT-4o mini(128k) 的 29.01 [evidence: comparison]"
  - "在 6 个代表任务的 hop 消融中，任意图结构设置都优于无图版本，且图相关任务的相对收益至少超过 20%，说明多跳结构信息是主要性能来源之一 [evidence: ablation]"
  - "在 Agent 与 RAG 新任务上，先用其他任务训练的 GWM 具有有效的零样本/少样本迁移能力，且部分 RAG 零样本结果优于仅用单任务训练的模型 [evidence: comparison]"
related_work_position:
  extends: "L3P (Zhang et al. 2021)"
  competes_with: "LLAGA (Chen et al. 2024a); OFA (Liu et al. 2023a)"
  complementary_to: "GraphRAG (Edge et al. 2024); LoRA (Hu et al. 2021)"
evidence_strength: strong
pdf_ref: paperPDFs/World_models_in_other_modalities/arXiv_2025/2025_Graph_World_Model.pdf
category: Others
---

# Graph World Model

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2507.10539), [Code](https://github.com/ulab-uiuc/GWM)
> - **Summary**: 这篇论文把世界模型从“平铺序列建模”改成“状态图 + 动作节点”建模，使同一套模型能统一处理图结构、多模态和跨任务输出。
> - **Key Performance**: LongBench v2 上 GWM-E 以 2k context 达到 33.32 Acc，高于 GPT-4o mini(128k) 的 29.01；Multi-Modal-Paper 上多模态匹配 F1 达到 97.13。

> [!info] **Agent Summary**
> - **task_path**: 多模态图状态 / 动作描述 -> 下一状态的文本、图像、链接预测或决策输出
> - **bottleneck**: 现有世界模型难以把结构关系、跨模态信息和多任务动作接口统一到一个可扩展的状态转移框架里
> - **mechanism_delta**: 用动作节点在图上选择相关目标节点，再做多跳消息传递，把结构化证据压缩为 LLM/扩散模型可消费的条件输入
> - **evidence_signal**: 2k 上下文的 GWM-E 在 LongBench v2 上超过 128k 长上下文 LLM 基线
> - **reusable_ops**: [action-node-query, multi-hop-graph-aggregation]
> - **failure_modes**: [过多 hop 导致 over-smoothing, 图构造噪声会误导目标检索与聚合]
> - **open_questions**: [如何支持异配/动态图, 如何从一次性任务扩展到长期闭环控制]

## Part I：问题与挑战

这篇论文真正要解决的，不是“把图送进 LLM”这么简单，而是一个更底层的统一接口问题：

- **现有世界模型**擅长视频、文本等非结构化序列，但通常把上下文当作平铺 token，难以显式利用节点关系、局部结构和多跳依赖。
- **现有图基础模型**虽然会处理图，但大多绑定在节点分类、链路预测等预定义图任务上，不自然支持图像/表格/文本混合输入，也不容易接到生成、RAG、规划等输出端。
- **真正瓶颈**因此不只是“图表示能力不足”，而是：**状态表示、任务表达、解码接口三者没有统一**。

这也是它“为什么现在做”的原因：  
一方面，世界模型已经证明了大规模预训练在预测、生成、规划上的价值；另一方面，科学、推荐、文档、代理协作等真实数字世界数据，本来就是**图结构 + 多模态**。如果仍然只靠长上下文把一切拍平成序列，成本高且容易丢关系。

### 输入/输出接口

论文把世界建模成一个图状态：

- **输入状态**：节点可包含图像、表格、文本中的一种或多种；边可以是显式关系，也可以是 embedding 相似度构图。
- **动作**：被建模为一个**动作节点**。  
  - intended action：直接对应 node / edge / graph 级任务  
  - unintended action：通过相似度检索目标节点，典型就是 RAG
- **输出**：下一状态可以是文本、图像、链接是否存在、类别标签、下一步决策等。

边界上，这个工作主要覆盖 **text / image / table** 三种模态，且实验大多是**离线监督式任务**，不是完整在线交互式强化学习闭环。

## Part II：方法与洞察

GWM 的核心设计是：**把“任务”也图化**。  
状态是图，动作是节点，目标是被动作节点选中的相关子图，然后再把这部分结构化信息送给统一解码器。

### 1）统一建模：状态图 + 动作节点

这一步最关键。它把原本互不相干的任务都改写成统一接口：

- 推荐：动作节点查询用户-物品边是否应存在
- 图预测：动作节点对应 node / edge / graph 级判断
- RAG：动作节点是 query，通过相似度找到 top-k chunk 节点
- 多代理协作：动作节点聚合多个 agent 与外部知识节点
- 规划：动作节点基于历史状态图预测下一步决策

这样做的好处是：模型不再为每个任务单独设计 head，而是把问题统一成  
**“动作如何在当前世界状态图上定位证据，并预测下一状态”**。

### 2）两种实现：GWM-T 与 GWM-E

#### GWM-T
先把多模态都转成文本：

- 图像 → 用 LLaVA 转文字描述
- 表格 → 用模板转文字
- 文本 → 直接保留

随后在**token 空间**做消息传递：把中心节点文本与邻居节点文本做 prompt 式聚合，再把目标节点 + 动作提示喂给 LLM 或 Stable Diffusion。

优点是简单、可解释、直接复用现成 instruction tuning。  
缺点是 **token 成本高，context 容量受限**。

#### GWM-E
把多模态先编码到 embedding 空间：

- 文本/表格用 BERT
- 图像用 CLIP
- 缺失模态补零向量

然后做**embedding 级多跳聚合**，再用 projector 把多跳图表示转成解码器条件 token。  
LLM 侧基本是 prefix-tuning 风格：**冻结大模型，只调 projector**。

这一步本质上把瓶颈从“原始 token 长度”转移到“结构化 latent 条件注入”。  
因此它更适合长上下文、多跳图和低 token 成本场景。

### 核心直觉

**what changed**：  
从“把世界状态当长序列 + 每个任务一个专用头”，变成“状态图 + 动作节点查询 + 多跳聚合 + 统一解码器条件”。

**which bottleneck changed**：  
原来模型必须在大上下文里自己隐式找关系；现在动作节点先做**结构化定位**，多跳聚合再做**关系压缩**，解码器看到的是已经筛过的证据，而不是整段噪声上下文。

**what capability changed**：  
这让同一模型可以跨任务复用，并且在长文本/大图场景下，不靠暴力扩 context，也能利用结构关系取得更好的预测、生成和规划能力。

### 策略权衡

| 设计选择 | 带来的能力 | 主要代价 | 适用情况 |
|---|---|---|---|
| 动作节点统一接口 | 一个框架覆盖 prediction / generation / optimization | 依赖动作到目标节点的连接或检索质量 | 多任务统一建模 |
| GWM-T（文本空间） | 语义透明，易接 LLM/SD | token 贵，长上下文受限 | 文本主导、需要可解释提示时 |
| GWM-E（embedding 空间） | 低 token 成本，长上下文/多跳更强 | 更依赖编码器与 projector 设计 | RAG、推荐、图预测、规划 |
| 增加 hop 数 | 扩大结构感受野 | 可能 over-smoothing，引入冗余 | 局部关系不足的一般图任务 |

## Part III：证据与局限

### 关键证据信号

- **跨任务比较信号**：同一个 GWM 在 6 类任务上与专用方法大体持平或更优，不是只在单一 benchmark 上成立。
- **最强能力跳变信号**：在 **LongBench v2** 上，`GWM-E (2k)` 的总体准确率 **33.32**，超过 `GPT-4o mini (128k)` 的 **29.01**。这说明它真正利用了**图结构压缩与检索**，而不是单纯靠更长上下文。
- **多模态对齐信号**：在 **Multi-Modal-Paper** 的多模态匹配上，GWM-E 达到 **97.13 F1**，明显高于 Contrastive MLP 的 **50.31**，说明结构化跨模态关系确实被学到了。
- **因果支持信号（ablation）**：multi-hop 图结构在 6 个任务上都优于 no-graph 版本，但 hop 不是越多越好，过多会过平滑。
- **迁移信号**：在 Agent / RAG 上，跨任务预训练后的 zero-shot / 10% few-shot 微调能较快适配新任务，说明统一接口不是“共享壳子”，而是有实际迁移性。

### 局限性

- **Fails when**: 图边构造噪声较大、关系是异配/动态而非同配、或 hop 数过大导致 over-smoothing 时，结构聚合会失效；另外在文本推理主导的多代理诊断上，GWM-E 并未稳定优于 GWM-T。
- **Assumes**: 需要先构造状态图（显式边或相似度边）；当前只支持 text / table / image；依赖 Llama-3-8B、SD-v1-5、LLaVA-1.5-7B、CLIP、BERT 等预训练组件，训练实验使用 4×NVIDIA A6000，复现门槛不算低。
- **Not designed for**: 视频/音频/3D 等更丰富模态、在线长期闭环控制、无需构图的纯端到端环境建模，以及非同配图上的系统性验证。

### 可复用组件

- **动作节点抽象**：把不同任务统一成“查询当前世界状态”的接口。
- **多跳图聚合器**：在 decoder 前插入结构化证据压缩层。
- **projector-tuning 接口**：冻结大模型，仅调图条件投影器，适合扩展到别的 LLM / diffusion 系统。
- **双路线设计**：GWM-T 适合高可解释 prompt 场景，GWM-E 适合高效率、长上下文场景。

![[paperPDFs/World_models_in_other_modalities/arXiv_2025/2025_Graph_World_Model.pdf]]