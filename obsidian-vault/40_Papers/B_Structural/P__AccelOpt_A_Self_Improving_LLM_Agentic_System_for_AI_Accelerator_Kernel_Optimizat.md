---
title: 'AccelOpt: A Self-Improving LLM Agentic System for AI Accelerator Kernel Optimization'
type: paper
paper_level: B
venue: ArXiv.org
year: 2025
acceptance: null
cited_by: 3
facets:
  learning_paradigm:
  - Reinforcement Learning
  modality:
  - Text
core_operator: AccelOpt 的核心直觉是：内核优化的知识可以从搜索过程本身中自举积累，而无需预先注入专家经验。束搜索提供了结构化的探索宽度，使系统不会过早收敛到局部最优；演化记忆则将每次成功优化的洞察沉淀为可复用的知识，使后续迭代站在更高的起点上。两者的耦合实现了「搜索越多、记忆越丰富、下一轮搜索越有效」的正反馈循环，本质上是将推理时计算（inference-time scaling）与经验积累（memor
paper_link: https://arxiv.org/abs/2511.15915
structurality_score: 0.55
---

# AccelOpt: A Self-Improving LLM Agentic System for AI Accelerator Kernel Optimization

## Links

- Mechanism: [[C__llm_agentic_code_optimization_with_memory]]

> AccelOpt 的核心直觉是：内核优化的知识可以从搜索过程本身中自举积累，而无需预先注入专家经验。束搜索提供了结构化的探索宽度，使系统不会过早收敛到局部最优；演化记忆则将每次成功优化的洞察沉淀为可复用的知识，使后续迭代站在更高的起点上。两者的耦合实现了「搜索越多、记忆越丰富、下一轮搜索越有效」的正反馈循环，本质上是将推理时计算（inference-time scaling）与经验积累（memory accumulation）结合，用时间换取专家知识的替代。

> **结构性改进**。先读 baseline，再看本文修改了哪些核心组件。

## 核心公式

$$
\text{cost} = \sum_i (\text{input\_tokens}_i + \text{output\_tokens}_i) \times \text{price\_per\_token}_i
$$

> 定义了模型调用成本的计算方式，用于比较 AccelOpt（开源模型）与 Claude Sonnet 4 的费用差异（26×）。
> *Slot*: cost estimation for model comparison

$$
t_{pos} = 1.04,\quad t_{neg} = 1.15,\quad \text{TopK}=8,\quad \text{ExpN}=16,\quad B=6,\quad N=12,\quad T=16
$$

> 这些超参数直接控制 AccelOpt 的搜索宽度与迭代深度，是复现实验结果的关键配置。
> *Slot*: beam search 超参数配置（executor + memory agents）

## 关键图表

**Figure 6**
: AccelOpt 在 Trainium 1 和 Trainium 2 上的平均峰值吞吐量百分比随迭代次数的变化曲线
> 证据支持: 支持「AccelOpt 随迭代自我提升」的核心主张：Trainium 1 从 49% 升至 61%，Trainium 2 从 45% 升至 59%。

**Figure 7**
: Claude Sonnet 4 与 AccelOpt 在 Trainium 1/2 上逐任务的峰值吞吐量百分比对比（x 轴按基线性能排序）
> 证据支持: 支持「AccelOpt 性能与 Claude Sonnet 4 相当」的主张，同时揭示两者在不同任务上的差异分布。

**Figure 8**
: BatchMatmul+Softmax 算子的非局部循环优化三阶段演化（内存溢出消除 + 向量引擎利用率提升）
> 证据支持: 支持「AccelOpt 能发现需要多步推理的全局优化」的案例证据，延迟从 12.0 ms 降至 6.4 ms，利用率从 46% 升至 84%。

**Table 3**
: 以 Claude Sonnet 4 作为 AccelOpt 基础模型时与重复采样的成本对比
> 证据支持: 支持「AccelOpt 框架本身可将 Claude Sonnet 4 的使用成本降低 3.3×」的成本效益主张。

## 详细分析

# AccelOpt: A Self-Improving LLM Agentic System for AI Accelerator Kernel Optimization

## Part I：问题与挑战

AI 加速器（如 AWS Trainium）的内核优化极度依赖专家知识：开发者需要深入理解硬件内存层次、并行化方案、调度策略以及特定架构约束，才能将机器学习算子高效映射到硬件资源上。以 GPU 为例，H100 发布后约一年才有注意力内核达到理论峰值的 37%，再过一年才接近 85%——这说明即便是成熟平台，优化周期也极长。对于 Trainium 这类新兴加速器，情况更为严峻：NKI（Neuron Kernel Interface）编程模型相对新颖，缺乏成熟的优化食谱和性能启发式规则，开发者几乎没有可参考的先验经验。

此外，规模化部署带来了成本压力：实际生产中需要跨不同配置、硬件版本和工作负载优化数百乃至数千个内核，人工专家介入的方式在经济上不可持续。现有 LLM 辅助内核优化方法（如 Autocomp、GEPA）要么依赖人工构造的优化列表，要么仅针对特定平台，缺乏自主积累优化知识的能力。核心挑战有两个：一是如何在巨大的设计空间中进行策略性探索以控制 LLM 查询成本；二是如何让系统在探索过程中自主积累优化洞察，实现能力随时间持续提升，而无需人工干预。

## Part II：方法与洞察

AccelOpt 是一个自我提升的 LLM 多智能体系统，其核心设计围绕两个相互耦合的机制：束搜索（beam search）驱动的迭代生成，以及随候选内核演化的优化记忆（optimization memory）。

系统架构层面，AccelOpt 由多个专职 agent 组成：executor agent 负责生成优化后的内核代码，memory agent 负责从慢-快内核对中提炼经验与洞察并更新记忆库，此外还有负责评估和筛选候选内核的组件。每次迭代中，系统以当前最优候选内核集合为起点，利用最新的优化记忆指导 executor 生成新候选，再通过实际硬件执行获取性能反馈，筛选出 TopK 个候选进入下一轮。

优化记忆的设计是 AccelOpt 区别于简单重复采样的关键。记忆内容从历史慢-快内核对中提炼，记录哪些变换有效、为何有效，并随候选内核的演化而动态更新——这与 LessonL 的静态锚点（始终以基线内核为参照）形成对比，理论上能维持更高的多样性。

超参数配置方面，系统使用 tpos=1.04、tneg=1.15 控制正负样本的温度，TopK=8 控制每轮保留的候选数，ExpN=16 控制每轮扩展数，B=6、N=12、T=16 分别控制束宽、采样数和迭代轮数。模型选择上，executor 使用 Qwen3-Coder-480B-A35B-Instruct-FP8，其余 agent 使用 gpt-oss-120b，整体成本比直接调用 Claude Sonnet 4 低 26×。

系统能够发现的优化类型涵盖两个层次：局部窥孔优化（代数化简、硬件内联函数融合、指令模式识别）和非局部循环变换（消除内存溢出、重组循环结构）。后者需要多步非平凡推理，结合内核语义、底层硬件架构和 profiler 反馈，是系统能力的核心体现。

基准方面，论文同步构建了 NKIBench——一个从真实 LLM 工作负载中提取的 Trainium 内核基准套件，涵盖不同复杂度的算子，以峰值吞吐量百分比（而非相对加速比）作为绝对性能指标。

### 核心直觉

AccelOpt 的核心直觉是：内核优化的知识可以从搜索过程本身中自举积累，而无需预先注入专家经验。束搜索提供了结构化的探索宽度，使系统不会过早收敛到局部最优；演化记忆则将每次成功优化的洞察沉淀为可复用的知识，使后续迭代站在更高的起点上。两者的耦合实现了「搜索越多、记忆越丰富、下一轮搜索越有效」的正反馈循环，本质上是将推理时计算（inference-time scaling）与经验积累（memory accumulation）结合，用时间换取专家知识的替代。

## Part III：证据与局限

核心性能主张有实验支撑：AccelOpt 在 NKIBench 上将 Trainium 1 的平均峰值吞吐量从 49% 提升至 61%，Trainium 2 从 45% 提升至 59%（Figure 6），且与 Claude Sonnet 4（thinking mode）重复采样的性能相当（Figure 7）。成本优势（26×）通过 token 计价公式计算，但存在一个已知缺陷：Claude Sonnet 4 的内部推理 token 不可见，未计入其成本，导致倍数可能被高估。

消融实验验证了束搜索和优化记忆各自的贡献（Section 4.4），但具体数字未在摘录文本中呈现。优化记忆相比 LessonL「更具多样性」的主张仅为设计层面的推断，未经消融实验量化验证。

泛化性方面存在明显局限：所有系统性实验均在 NKIBench 上进行，NKIBench 以外的 Conv2D 案例（48.8% 峰值吞吐量）仅为单一教学场景数据点，且原文被截断，信息不完整。向其他加速器（AMD NPU、TPU）的迁移性未经验证，结论中的泛化性表述属于展望而非实验结论。与 AlphaEvolve 等同类系统缺乏直接数值对比（AlphaEvolve 不开源是客观原因），跨系统比较无法验证。
