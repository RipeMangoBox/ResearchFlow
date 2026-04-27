---
title: 'AccelOpt: A Self-Improving LLM Agentic System for AI Accelerator Kernel Optimization'
type: paper
paper_level: B
venue: arXiv
year: 2025
paper_link: https://arxiv.org/abs/2511.15915
aliases:
- LLM多智能体自进化加速器内核优化
- AccelOpt
- AccelOpt 的核心直觉是：内核优化的知识可以从搜索过程本身中自举
cited_by: 3
method: AccelOpt
modalities:
- Text
paradigm: Reinforcement Learning
---

# AccelOpt: A Self-Improving LLM Agentic System for AI Accelerator Kernel Optimization

[Paper](https://arxiv.org/abs/2511.15915)

**Topics**: [[T__Agent]], [[T__Code_Generation]] | **Method**: [[M__AccelOpt]]

> [!tip] 核心洞察
> AccelOpt 的核心直觉是：内核优化的知识可以从搜索过程本身中自举积累，而无需预先注入专家经验。束搜索提供了结构化的探索宽度，使系统不会过早收敛到局部最优；演化记忆则将每次成功优化的洞察沉淀为可复用的知识，使后续迭代站在更高的起点上。两者的耦合实现了「搜索越多、记忆越丰富、下一轮搜索越有效」的正反馈循环，本质上是将推理时计算（inference-time scaling）与经验积累（memory accumulation）结合，用时间换取专家知识的替代。

| 中文题名 | LLM多智能体自进化加速器内核优化 |
| 英文题名 | AccelOpt: A Self-Improving LLM Agentic System for AI Accelerator Kernel Optimization |
| 会议/期刊 | ArXiv.org (预印本) |
| 链接 | [arXiv](https://arxiv.org/abs/2511.15915) · [Code] · [Project] |
| 主要任务 | AI加速器（AWS Trainium）内核优化，提升ML算子峰值吞吐量 |
| 主要 baseline | Claude Sonnet 4 (thinking mode), LessonL, Autocomp, GEPA |

> [!abstract] 因为「AI加速器内核优化极度依赖专家知识且新兴平台缺乏成熟优化食谱」，作者在「Claude Sonnet 4 重复采样」基础上改了「束搜索+演化优化记忆的LLM多智能体迭代框架」，在「NKIBench（Trainium 1/2）」上取得「峰值吞吐量从49%/45%提升至61%/59%，成本降低26×」

- **Trainium 1**: 平均峰值吞吐量从 49% 提升至 61%（+12 pp）
- **Trainium 2**: 平均峰值吞吐量从 45% 提升至 59%（+14 pp）
- **成本**: 相比 Claude Sonnet 4 直接调用降低 26×（token 计价，Claude 内部推理 token 未计入）

## 背景与动机

AI 加速器内核优化是一个高度专业化的领域：开发者必须深入理解硬件内存层次、并行化方案、调度策略以及特定架构约束，才能将机器学习算子高效映射到硬件资源。以 NVIDIA H100 GPU 为例，发布后约一年其注意力内核仅达到理论峰值的 37%，再过一年才接近 85%——即便是成熟平台，优化周期也极长。对于 AWS Trainium 这类新兴加速器，情况更为严峻：其 NKI（Neuron Kernel Interface）编程模型相对新颖，缺乏成熟的优化食谱和性能启发式规则，开发者几乎没有可参考的先验经验。

现有方法如何应对这一挑战？**Autocomp** 等早期 LLM 辅助方法依赖人工构造的优化变换列表，限制了发现非预期优化的能力；**GEPA** 针对特定平台设计，缺乏跨硬件迁移性；**LessonL** 引入从慢-快内核对中学习经验的机制，但采用静态锚点策略——始终以原始基线内核为参照提取经验，导致记忆内容随候选演化而逐渐失效，多样性衰减。

这些方法的共同短板在于：缺乏**自主积累优化知识**的能力。要么依赖人工预设的优化空间，要么记忆机制无法随搜索过程动态演化。规模化部署进一步放大了这一问题：实际生产需要跨不同配置、硬件版本和工作负载优化数百乃至数千个内核，纯人工专家介入在经济上不可持续。核心挑战可归结为两点：一是如何在巨大的设计空间中**策略性探索**以控制 LLM 查询成本；二是如何让系统在探索过程中**自主沉淀优化洞察**，实现能力随时间持续提升而无需人工干预。

本文提出 AccelOpt，其核心思想是将束搜索的结构化探索与随候选内核演化的动态记忆相结合，构建一个自我增强的 LLM 多智能体系统。
![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/6c6f6011-8896-46e2-807b-19d19aa87c6e/figures/Figure_1.png)
*Figure 1: Figure 1. At each iteration of AccelOpt, the agentic workflow shown on the right optimizes the candidate kernels with the latestoptimization memory, and generates new candidate kernels, updating optim*



## 核心创新

核心洞察：内核优化的专家知识可以从搜索过程本身中自举积累，无需预先注入；束搜索提供结构化探索宽度防止过早收敛，演化记忆将每次成功优化的洞察沉淀为可复用知识，两者的正反馈耦合使「搜索越多、记忆越丰富、下一轮搜索越有效」成为可能。

| 维度 | Baseline (Claude Sonnet 4 / LessonL) | 本文 (AccelOpt) |
|:---|:---|:---|
| 探索策略 | 独立重复采样，无状态 | 束搜索迭代，TopK=8 保留候选，结构化状态传递 |
| 经验记忆 | 静态锚点（始终对比原始基线） | 动态演化（对比当前轮次慢-快对），随候选更新 |
| 知识来源 | 预训练知识 + 人工提示工程 | 搜索过程自举积累，零人工优化食谱 |
| 成本结构 | 单轮高成本大模型调用 | 多智能体分工（executor 用 Qwen3-Coder-480B-A35B-Instruct-FP8，其余用 gpt-oss-120b），26× 降低 |

## 整体框架



AccelOpt 采用迭代式多智能体架构，每轮循环包含四个阶段：

**输入**: 当前候选内核集合（首轮为原始未优化内核）+ 最新优化记忆（optimization memory）

**Executor Agent** → 接收当前 TopK 候选与优化记忆，生成 ExpN=16 个新优化候选内核代码。使用 Qwen3-Coder-480B-A35B-Instruct-FP8，专精代码生成任务。

**硬件执行与评估** → 所有候选内核在真实 Trainium 硬件上编译执行，获取实际峰值吞吐量百分比作为性能反馈。

**筛选（TopK Selection）** → 按性能排序，保留 TopK=8 个最优候选进入下一轮，其余淘汰。

**Memory Agent** → 从每轮产生的慢-快内核对中提炼经验：分析何种变换有效、为何有效，更新优化记忆库。使用 gpt-oss-120b，负责非代码生成的推理任务。记忆内容动态演化——始终基于当前轮次的候选对比，而非固定原始基线。

**输出**: 经过 T=16 轮迭代后的最优内核，及累积的优化记忆库。

整体控制参数：束宽 B=6，每轮采样数 N=12，温度参数 tpos=1.04（正样本）、tneg=1.15（负样本）。

```
原始内核 ──→ [Executor] ──→ 16个候选 ──→ [硬件执行] ──→ 性能数据
    ↑                              ↓
    └──── [Memory Agent] ←──── 慢-快对分析 ←────┘
           ↓
      优化记忆更新 ──→ 下一轮 Executor 输入
```

## 核心模块与公式推导

### 模块 1: 束搜索迭代生成（对应框架图 Executor → 筛选回路）

**直觉**: 单次 LLM 采样容易陷入局部最优，结构化保留多个候选维持探索多样性。

**Baseline 公式** (Claude Sonnet 4 重复采样): 
$$\hat{k} = \text{arg}\max_{i \in [1, M]} P(k_i | \text{prompt}), \quad k_i \sim \text{LLM}(\cdot | \text{prompt})$$
符号: $k_i$ = 第 $i$ 个采样内核, $M$ = 总采样数, prompt = 固定人工构造的优化指令。

**变化点**: Baseline 各采样独立同分布，无状态传递；且 prompt 固定，无法利用搜索过程中发现的优化模式。

**本文公式（推导）**:
$$\text{Step 1}: \mathcal{K}_0 = \{k_0\}, \quad \mathcal{M}_0 = \emptyset \quad \text{初始化候选集与空记忆}$$
$$\text{Step 2}: \mathcal{K}_t^{\text{expand}} = \text{bigcup}_{k \in \mathcal{K}_{t-1}^{\text{top}}} \text{Executor}(k, \mathcal{M}_{t-1}; \text{ExpN}) \quad \text{以记忆指导，每候选扩展16个}$$
$$\text{Step 3}: \mathcal{K}_t^{\text{top}} = \text{TopK}(\{\text{Throughput}(k') | k' \in \mathcal{K}_t^{\text{expand}}\}, K=8) \quad \text{硬件实测筛选}$$
$$\text{Step 4}: \mathcal{M}_t = \text{MemoryAgent}(\{(k_{\text{slow}}, k_{\text{fast}})_t\}, \mathcal{M}_{t-1}) \quad \text{从慢-快对更新记忆}$$
$$\text{最终}: \hat{k} = \text{arg}\max_{k \in \mathcal{K}_T^{\text{top}}} \text{Throughput}(k), \quad T=16$$

**对应消融**: 

---

### 模块 2: 演化优化记忆（对应框架图 Memory Agent 回路）

**直觉**: 优化知识应随候选内核的演化而更新，固定基线参照会导致记忆失效。

**Baseline 公式** (LessonL 静态锚点):
$$\mathcal{M}_t^{\text{static}} = \{(r_j, e_j) | r_j = \text{Extract}(k_0, k_j^{\text{fast}})\}$$
符号: $k_0$ = 原始基线内核（固定锚点）, $r_j$ = 变换规则, $e_j$ = 有效性解释。

**变化点**: 当候选内核 $k_t$ 已远离 $k_0$ 时，基于 $k_0$ 的经验 $r_j$ 可能不再适用当前候选；且固定锚点导致记忆内容高度相似，多样性衰减。

**本文公式（推导）**:
$$\text{Step 1}: \mathcal{P}_t = \{(k_{\text{slow}}, k_{\text{fast}})_i | i \in \text{round } t\} \quad \text{收集当前轮次慢-快对}$$
$$\text{Step 2}: \Delta_i = k_{\text{fast}} - k_{\text{slow}} \quad \text{代码差异分析（语法+语义）}$$
$$\text{Step 3}: (r_i, e_i, c_i) = \text{LLM}_{\text{memory}}(\Delta_i, \text{profile}_i) \quad \text{提炼规则、解释、适用条件}$$
$$\text{Step 4}: \mathcal{M}_t = \text{Update}(\mathcal{M}_{t-1}, \{(r_i, e_i, c_i)\}, \text{similarity\_threshold}) \quad \text{去重合并，动态更新}$$
$$\text{最终}: \mathcal{M}_t = \{(r_j, e_j, c_j, w_j)\}_{j=1}^{|\mathcal{M}_t|}, \quad w_j = \text{success\_rate}(r_j) \quad \text{带权重的经验条目}$$

关键设计：适用条件 $c_j$ 记录该规则适用的内核特征（算子类型、张量形状、内存压力等级），使记忆可**条件化检索**而非全局应用。

**对应消融**: 

---

### 模块 3: 多智能体成本优化（对应框架图 模型选择策略）

**直觉**: 不同智能体任务难度差异显著，无需统一使用最强（最贵）模型。

**Baseline 公式** (统一调用 Claude Sonnet 4):
$$\text{Cost}_{\text{base}} = M \times (C_{\text{input}} \cdot |\text{prompt}| + C_{\text{output}} \cdot |\text{response}|)$$
符号: $C_{\text{input}}, C_{\text{output}}$ = Claude Sonnet 4 输入/输出 token 单价（含不可见推理 token）。

**变化点**: 代码生成需要强推理能力，但经验提炼和记忆更新主要是模式匹配与摘要，可使用更小模型；且 Claude 的推理 token 不可见导致成本不透明。

**本文公式（推导）**:
$$\text{Step 1}: \text{Task decomposition: } \tau_{\text{code}} \oplus \tau_{\text{memory}} \oplus \tau_{\text{eval}}$$
$$\text{Step 2}: \text{Model assignment: } m(\tau) = \begin{cases} \text{Qwen3-Coder-480B-A35B-Instruct-FP8} & \text{if } \tau = \tau_{\text{code}} \\ \text{gpt-oss-120b} & \text{otherwise} \end{cases}$$
$$\text{Step 3}: \text{Cost}_{\text{ours}} = \sum_{t=1}^{T} \sum_{\tau} \left( C_{m(\tau)}^{\text{in}} \cdot |\text{prompt}_{t,\tau}| + C_{m(\tau)}^{\text{out}} \cdot |\text{response}_{t,\tau}| \right)$$
$$\text{最终}: \frac{\text{Cost}_{\text{base}}}{\text{Cost}_{\text{ours}}} = 26 \quad \text{（注：Claude 内部推理 token 未计入，倍数可能高估）}$$

## 实验与分析

主实验结果（NKIBench，峰值吞吐量 %）：

| Method | Trainium 1 | Trainium 2 | 备注 |
|:---|:---|:---|:---|
| 原始内核 | 49% | 45% | 未优化基线 |
| Claude Sonnet 4 (thinking mode, 重复采样) | ~61% | ~59% | 单轮高成本，无记忆积累 |
| **AccelOpt** | **61%** | **59%** | 迭代束搜索+演化记忆，成本 26× 降低 |


![Figure 7](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/6c6f6011-8896-46e2-807b-19d19aa87c6e/figures/Figure_7.png)
*Figure 7: Figure 7. Per-task kernel improvement achieved using Claude Sonnet 4 and AccelOpt on Trainium 1 (above) and Trainium 2 (below). Thex-axis is sorted by the baseline kernel’s percentage of peak throughp*



核心结论：AccelOpt 在达到与 Claude Sonnet 4 相当性能的同时，实现数量级成本降低。Figure 7 显示 per-task 的详细对比——Claude Sonnet 4 与 AccelOpt 在各算子上的性能分布高度重叠，验证了核心主张。

消融分析（Section 4.4）：束搜索和优化记忆各自贡献被验证，但具体数字未在可用文本中呈现。Figure 11 展示「有效探索下的早期饱和加速」现象：当非规约维度 N 较大时，内存管理搜索空间宽广，系统能在早期迭代快速发现显著优化，随后趋于饱和——这解释了为何 T=16 轮迭代足够。


![Figure 11](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/6c6f6011-8896-46e2-807b-19d19aa87c6e/figures/Figure_11.png)
*Figure 11: Figure 10. Early saturating speedup with effective exploration. Thenon-reduction dimension N is large, leading to a wide memory-management search space. However, because the operator is dom-inated by*



非局部优化发现：Figure 8 展示 AccelOpt 为 fused BatchMatmul+Softmax 发现的非局部优化——通过重组循环结构消除内存溢出，所有变量均为张量 tile，涉及跨循环边界的复杂数据流重组。这类优化需要结合内核语义、硬件架构和 profiler 反馈的多步推理，是系统能力的核心体现。

公平性检查：
- **Baseline 强度**: Claude Sonnet 4 为当前顶级商用模型，thinking mode 已启用；与 AlphaEvolve 等同类系统缺乏直接数值对比（AlphaEvolve 不开源）。
- **成本计算缺陷**: Claude Sonnet 4 内部推理 token 不可见，26× 成本降低倍数可能被高估。
- **数据局限**: 所有系统实验限于 NKIBench；外部 Conv2D 案例（48.8% 峰值吞吐量）仅为单一数据点且原文截断。
- **泛化性**: 向 AMD NPU、TPU 等迁移性未验证。

## 方法谱系与知识库定位

**方法家族**: LLM Agentic Systems for Program Optimization / Auto-tuning

**Parent method**: LessonL（从慢-快内核对中学习经验的记忆机制）

AccelOpt 改变的 slots：
- **架构**: 单智能体 → 多智能体分工（executor/memory/eval 分离）
- **训练/推理策略**: 独立采样 → 束搜索迭代，引入结构化状态传递
- **数据/记忆**: 静态锚点记忆 → 随候选演化的动态记忆，条件化检索
- **推理成本**: 统一大模型 → 任务自适应模型选择（Qwen3-Coder-480B 用于代码，gpt-oss-120b 用于推理）

**直接 Baselines 与差异**:
- **Claude Sonnet 4 重复采样**: 性能相当但成本 26× 高，无记忆积累能力
- **LessonL**: 静态锚点记忆 vs. AccelOpt 的动态演化记忆；无迭代搜索结构
- **Autocomp**: 依赖人工优化列表 vs. 零先验自举
- **GEPA**: 平台专用 vs. 设计为可迁移（未验证）
- **AlphaEvolve**: 未开源，无直接对比

**后续方向**:
1. 向 GPU/TPU/AMD NPU 的跨加速器验证，检验 NKI 特定设计与通用性张力
2. 优化记忆的显式结构化（如知识图谱）替代自然语言描述，提升检索精度
3. 与强化学习结合，将记忆权重 $w_j$ 的更新从启发式改为可学习的价值估计

**标签**: 
- modality: code/kernel
- paradigm: LLM agentic system, iterative search, self-improving
- scenario: AI accelerator optimization, compiler auto-tuning
- mechanism: beam search, evolutionary memory, slow-fast pair learning
- constraint: hardware-specific (Trainium NKI), cost-efficient inference

