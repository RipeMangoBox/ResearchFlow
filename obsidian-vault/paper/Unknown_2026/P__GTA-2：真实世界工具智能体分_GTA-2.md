---
title: 'GTA-2: Benchmarking General Tool Agents from Atomic Tool-Use to Open-Ended Workflows'
type: paper
paper_level: B
venue: Unknown
year: 2026
paper_link: https://arxiv.org/abs/2604.15715
aliases:
- GTA-2：真实世界工具智能体分层基准
- GTA-2
- 现有基准将工具使用评估停留在原子层面
method: GTA-2
modalities:
- Text
---

# GTA-2: Benchmarking General Tool Agents from Atomic Tool-Use to Open-Ended Workflows

[Paper](https://arxiv.org/abs/2604.15715)

**Topics**: [[T__Agent]], [[T__Benchmark_-_Evaluation]], [[T__Reasoning]] | **Method**: [[M__GTA-2]] | **Datasets**: GTA-Atomic, GTA-Workflow, GTA-Workflow with advanced frameworks

> [!tip] 核心洞察
> 现有基准将工具使用评估停留在原子层面，而真实生产力场景要求端到端工作流完成。GTA-2的核心洞察是：开放式工作流的不可评估性（无固定答案、无预定义路径）可以通过将目标递归分解为可验证子目标来解决，从而将不可判定问题转化为结构化检查点验证问题。同时，将执行框架纳入统一评估揭示了一个被忽视的事实：在长时域任务中，系统设计的贡献可能超越底层模型能力本身。

| 中文题名 | GTA-2：真实世界工具智能体分层基准 |
| 英文题名 | GTA-2: Benchmarking General Tool Agents from Atomic Tool-Use to Open-Ended Workflows |
| 会议/期刊 | arXiv 2026 (预印本) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.15715) · Code  · Project  |
| 主要任务 | 工具使用评估、原子工具执行、开放式工作流完成、智能体框架评估、多模态任务执行 |
| 主要 baseline | GTA (直接前身)、ToolBench、APIBench、GAIA、AgentBench |

> [!abstract]
> 因为「现有工具智能体基准使用AI生成查询和虚拟工具，与真实世界严重脱节」，作者在「GTA」基础上改了「数据管道（真实用户查询+真实部署工具+多模态输入）、任务范围（新增长程开放工作流GTA-Workflow）、评估机制（递归检查点+LLM-as-Judge）」，在「GTA-Workflow」上取得「前沿模型成功率仅14.39%，暴露严重能力悬崖」

- **关键性能 1**: GTA-Atomic 上前沿模型 AnsAcc+I < 50%，原子工具使用仍存瓶颈
- **关键性能 2**: GTA-Workflow 上最佳模型 Gemini-2.5-Pro 成功率仅 14.39%
- **关键性能 3**: 先进执行框架（Manus、OpenClaw）较原始 LLM 获得 substantial enhancement

## 背景与动机

当前大语言模型（LLM）工具智能体在实验室环境中表现亮眼，但在真实场景中却频频失效。一个典型例子是：用户要求「根据这张手写会议纪要的截图，整理出待办事项并创建日历提醒，同时给参会者发送邮件」——这类任务需要理解手写材料（多模态输入）、调用日历和邮件API（真实工具）、自主规划多步操作（长程开放工作流），而现有基准完全无法评估这种真实生产力场景。

现有方法如何处理这一问题？**ToolBench** 构建了16,000+ API的调用环境，但使用AI生成的查询和虚拟工具模拟执行，缺乏真实用户意图；**GAIA** 评估通用AI助手的多步推理，但任务封闭性强，未聚焦真实工具部署与系统级协调；**GTA（前身）** 首次提出通用工具智能体基准，但仅限原子级短程封闭任务，且同样依赖AI生成查询与文本环境。这些方法的共同短板在于：数据不真实（AI生成而非真实用户）、工具不真实（虚拟模拟而非真实部署）、任务不真实（短程封闭而非长程开放）、评估不真实（预定义轨迹或简单正确性判断）。

GTA-2 的核心动机正是填补这一鸿沟：构建一个从原子工具使用到开放工作流的完整分层评估框架，让基准测试真正反映智能体在真实世界中的生产力价值。

## 核心创新

核心洞察：开放工作流的评估瓶颈不在于「没有任务」，而在于「没有可验证的评估方式」——因为真实工作流没有唯一正确答案，传统预定义轨迹或二元正确性判断完全失效；通过递归将开放目标分解为可验证子目标（检查点），并配合结构化LLM评判与加权聚合，首次使长程开放工作流的系统级评估成为可能。

| 维度 | Baseline (GTA/ToolBench/GAIA) | 本文 (GTA-2) |
|:---|:---|:---|
| 数据来源 | AI生成查询，显式解题步骤 | 真实用户查询，人类手工设计 |
| 工具环境 | 虚拟工具，文本模拟执行 | 真实部署工具，跨感知/操作/逻辑/创意类别 |
| 输入模态 | 纯文本 | 多模态：空间场景、截图、手写材料 |
| 任务范围 | 原子短程封闭任务 | 分层框架：GTA-Atomic + GTA-Workflow（长程开放） |
| 评估机制 | 预定义轨迹 / 简单正确性 | 递归检查点分解 + LLM-as-Judge [0,10] + 加权聚合 |
| 评估对象 | 仅LLM能力 | LLM + 执行框架（harness）联合评估 |

## 整体框架


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/f45aa302-560e-462d-aca6-aac4a282ed96/figures/Figure_1.png)
*Figure 1: Table 1: Comparison of benchmarks for LLM-based agent systems. *Real-world means solving the queries is helpful for humansin real life while step-implicit and tool-implicit for LLMs.*



GTA-2 采用分层双轨架构，输入为真实用户查询与真实部署工具，根据任务复杂度分流至两条评估管线：

**GTA-Atomic（上层管线）**：接收短程封闭任务查询 → 直接调用真实工具执行 → 输出答案或操作结果 → 以正确性指标（如AnsAcc+I）直接判定。该组件直接继承自GTA前身，但升级为真实用户查询、真实工具和多模态输入。

**GTA-Workflow（下层管线）**：接收长程开放工作流查询 → **任务描述与需求结构化模块**将原始查询D转化为结构化指令I = (D, Requirements(n), Rubric(n)) → **检查点分解模块**将开放目标拆解为可验证子目标集合C → 模型/框架执行并生成输出M → **LLM评判模块**根据I对M逐检查点评分s_c ∈ [0,10] → **加权聚合模块**归一化权重并计算最终得分S。该组件是GTA-2的全新核心，独立于原子任务评估。

```
真实用户查询 + 真实部署工具 + 多模态输入
           │
    ┌──────┴──────┐
    ▼             ▼
 GTA-Atomic   GTA-Workflow
(短程封闭)    (长程开放)
    │             │
    ▼             ▼
 直接正确性    结构化指令 I
  判定        检查点分解 C
              LLM评判 s_c∈[0,10]
              加权聚合 S
```

## 核心模块与公式推导

### 模块 1: 评估指令构建（对应框架图：GTA-Workflow入口）

**直觉**: 开放工作流没有标准答案，必须先给评判者一个「评分标准」，才能避免主观随意性。

**Baseline**: 传统基准无此步骤，直接以预定义轨迹或二元标签判定。

**本文公式（推导）**:
$$\text{Step 1}: \mathcal{I} \leftarrow (D, \text{Requirements}(n), \text{Rubric}(n)) \quad \text{将原始查询D扩展为三元组：任务描述、需求列表、评分细则}$$
$$\text{最终}: \mathcal{I}\text{ 作为LLM评判者的结构化输入，确保评判维度与任务目标对齐}$$

符号: $D$ = 原始任务查询, $\text{Requirements}(n)$ = n条可验证需求, $\text{Rubric}(n)$ = n维评分标准。

**对应消融**: 

---

### 模块 2: LLM评判评分与权重分配（对应框架图：GTA-Workflow核心评估层）

**直觉**: 人类评估复杂工作流时会分解为「完成了A吗？B呢？」——检查点评分模仿这种结构化评估，加权聚合则反映不同子目标的重要性差异。

**Baseline 公式** (传统正确性判断): 
$$L_{\text{base}} = \mathbb{1}[M = M_{\text{gold}}] \in \{0, 1\}$$
符号: $\mathbb{1}[\cdot]$ = 指示函数, $M_{\text{gold}}$ = 预定义标准答案。仅适用于封闭任务，开放工作流无$M_{\text{gold}}$。

**变化点**: 二元正确性对开放工作流失效；需要连续评分空间、可分解的子目标、以及重要性加权。

**本文公式（推导）**:
$$\text{Step 1}: s_c \leftarrow \text{LLMJudge}(M, \mathcal{I}) \quad s_c \in [0, 10] \quad \text{对每个检查点c，LLM评判者输出0-10连续评分}$$
$$\text{Step 2}: \mathbf{w} \leftarrow [\text{Weight}(c) \text{mid} c \in C] \quad \text{为检查点集合C分配初始权重向量}$$
$$\text{Step 3}: \mathbf{w} \leftarrow \text{NormalizeWeights}(\mathbf{w}) \quad \triangleright \sum_{c \in C} w_c = 1 \quad \text{归一化保证权重和为1，确保评分可比性}$$
$$\text{最终}: S \leftarrow \sum_{c \in C} w_c \cdot s_c \quad \text{加权聚合得最终得分}$$

符号: $s_c$ = 检查点c的评分, $w_c$ = 检查点c的归一化权重, $S$ = 最终聚合得分, $C$ = 检查点集合。

**对应消融**: Table 13 显示不同评判模型下的鲁棒性测试；Table 8 显示LLM评判者与人类评判者一致性（Spearman ρ=1.00, Kendall τ=1.00）。

---

### 模块 3: 评判验证机制（对应框架图：评估可靠性保障）

**直觉**: LLM-as-judge常被质疑有偏，需要量化验证其评判质量与人类专家的一致性。

**Baseline**: 多数工作直接使用LLM评判而不报告与人类的一致性，或仅报告小规模样本的粗略一致率。

**本文公式（推导）**:
$$\text{Step 1}: \text{收集人类评判者评分 } s_c^{\text{human}} \text{ 与 LLM评判者评分 } s_c^{\text{LLM}} \text{ 对同一批任务}$$
$$\text{Step 2}: \rho = \text{Spearman}(\{s_c^{\text{human}}\}, \{s_c^{\text{LLM}}\}) \quad \text{计算秩相关系数}$$
$$\text{Step 3}: \tau = \text{Kendall}(\{s_c^{\text{human}}\}, \{s_c^{\text{LLM}}\}) \quad \text{计算肯德尔和谐系数}$$
$$\text{最终}: \rho = 1.00,\quad \tau = 1.00 \quad \text{完美秩相关，验证评判机制可靠性}$$

**对应消融**: Table 8（30 tasks each cross-model validation）显示跨模型、跨来源输出上的一致 agreement。

## 实验与分析

**GTA-Atomic 主结果**（Table 5）：

| 评估维度 | 前沿模型表现 | 关键发现 |
|:---|:---|:---|
| AnsAcc+I (含图像生成答案准确率) | < 50% | 原子工具使用仍存显著瓶颈 |

**GTA-Workflow 主结果**（Table 6）：

| Method | Success Rate (SR) | 关键发现 |
|:---|:---|:---|
| Gemini-2.5-Pro | **14.39%** | 最佳模型，但暴露严重能力悬崖 |
| 其他前沿模型 | ≤ 14.39% | 长程开放工作流极具挑战性 |


![Figure 7](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/f45aa302-560e-462d-aca6-aac4a282ed96/figures/Figure_7.png)
*Figure 7: Figure 6: Model performance breakdown across 6 real-world categories in GTA-Workflow. Each subplot displays the averageroot scores (0-10) calculated via the recursive checkpoint scoring mechanism, wit*



核心数字解读：14.39% 的成功率直接支撑论文核心主张——前沿模型在真实开放工作流上「 largely fail」。这一数字并非某弱模型的失误，而是当前最强模型 Gemini-2.5-Pro 的上限，说明任务难度与模型能力之间存在结构性鸿沟。

**框架对比实验**（Table 7，30-task subset）：

| 配置 | 相对表现 | 关键发现 |
|:---|:---|:---|
| 原始 LLM | 基线 | 14.39% 级别 |
| + Manus / OpenClaw 框架 | substantial enhancement | 执行框架设计与模型能力同等关键 |


![Figure 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/f45aa302-560e-462d-aca6-aac4a282ed96/figures/Figure_4.png)
*Figure 4: Figure 5: Task difficulty analysis of GTA-Workflow w.r.t. de-liverable types.Struct.Data is short for Structured Data,which contains CSV, XLSX, and JSON. Multimedia includesthe modalities of image, au*



消融分析：检查点引导反馈（checkpoint-guided feedback）带来 moderate performance improvement，说明结构化评估反馈对智能体学习有实际助益。

**公平性检验**：
- 基线强度：GTA 为直接前身，ToolBench/APIBench/GAIA/AgentBench 为领域内主流基准，对比合理；但缺乏 GPT-4o 在完整工作流上的直接对比
- 计算/数据成本：未披露完整评估的GPU/时间成本，提及 GPT-5.2 作为 judge 但未完全明确规格
- 失败案例：14.39% 暗示大量任务完全失败，但未详细分析典型失败模式
- 潜在偏差：LLM-as-judge 声称完美秩相关，但仅基于30任务子集验证，大规模稳定性待考

## 方法谱系与知识库定位

**方法家族**: 工具智能体评估基准

**父方法**: GTA（GTA: a benchmark for general tool agents）—— GTA-Atomic 直接继承，GTA-Workflow 为范式级扩展

**变更槽位详解**:

| 槽位 | 父方法 GTA | GTA-2 | 变更类型 |
|:---|:---|:---|:---|
| architecture | 单层原子任务评估 | 分层双轨（Atomic + Workflow） | 结构扩展 |
| data_curation | AI生成查询 + 虚拟工具 + 纯文本 | 真实用户查询 + 真实部署工具 + 多模态 | 完全替换 |
| evaluation_mechanism | 预定义轨迹 / 简单正确性 | 递归检查点 + LLM-as-Judge [0,10] + 加权聚合 | 新增 |
| task_scope | 原子短程封闭任务 | + 长程开放生产力工作流 | 新增 |
| inference_recipe | 仅评估LLM | 联合评估 LLM + 执行框架（harness） | 修改 |

**直接基线差异**:
- **ToolBench**: 16,000+ API 但虚拟环境 vs GTA-2 真实部署
- **GAIA**: 通用推理助手但封闭任务 vs GTA-2 开放工作流+工具编排
- **AgentBench**: 多环境LLM评估但非真实工具 vs GTA-2 真实工具+多模态

**后续方向**: (1) 弥合 14.39%→实用的框架设计研究；(2) 降低LLM-as-judge成本的轻量级评判模型；(3) 动态更新的真实工作流库构建

**标签**: 模态(multimodal) / 范式(benchmark/evaluation) / 场景(real-world productivity) / 机制(recursive checkpoint-based evaluation) / 约束(open-ended, no gold trajectory)

