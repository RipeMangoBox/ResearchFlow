---
title: "REALM-Bench: A Benchmark for Evaluating Multi-Agent Systems on Real-world, Dynamic Planning and Scheduling Tasks"
venue: arXiv
year: 2025
tags:
  - Survey_Benchmark
  - task/planning-and-scheduling
  - difficulty-tiering
  - disruption-aware-evaluation
  - leaderboard-benchmarking
  - dataset/REALM-Bench
  - opensource/full
core_operator: 用五级难度阶梯、扰动注入与统一评分指标，把单体LLM和多智能体系统放到真实规划/调度任务中做同口径评测。
primary_logic: |
  真实规划/调度需求 → 构造P系列与J系列共14个分层场景，并沿依赖复杂度/并行线程/扰动频率与实例规模扩展 → 用规划质量、约束满足、协同效率与重规划能力统一评分 → 揭示系统在动态协同与状态维护上的能力边界
claims:
  - "REALM-Bench定义了14个规划/调度场景，并按5个难度层级覆盖从单智能体静态规划到大规模动态多智能体协同 [evidence: analysis]"
  - "作者的初步评测显示，Tier 1静态任务成功率为85–95%，而动态多智能体场景下降到45–70%，表明扰动恢复与协同重规划仍是主要短板 [evidence: analysis]"
  - "在reactive JSSP案例中，GPT-4o与Claude-3.7的Pass@1经常出现工序顺序或机器故障约束违例，而追加第二轮提示后可恢复出有效计划 [evidence: case-study]"
related_work_position:
  extends: "PlanBench (Valmeekam et al. 2023)"
  competes_with: "PlanBench (Valmeekam et al. 2023); TimeBench (Chu et al. 2023)"
  complementary_to: "AutoGen (Wu et al. 2024); LangGraph"
evidence_strength: moderate
pdf_ref: paperPDFs/Agentic_Reasoning_Benchmarks_Core_Mechanisms_of_Agentic_Reasoning_Memory_and_Planning_Planning_and_Feedback/arXiv_2025/2025_REALM_Bench_A_Benchmark_for_Evaluating_Multi_Agent_Systems_on_Real_world_Dynamic_Planning_and_Scheduling_Tasks.pdf
category: Survey_Benchmark
---

# REALM-Bench: A Benchmark for Evaluating Multi-Agent Systems on Real-world, Dynamic Planning and Scheduling Tasks

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2502.18836), [Code](https://github.com/genglongling/REALM-Bench), [Datasets](https://github.com/genglongling/M-APPLE-OS)
> - **Summary**: 这篇工作提出一个面向真实规划与调度的多智能体评测基准，用分层任务、动态扰动和统一指标系统性暴露当前LLM agent在状态维护、约束跟踪和协同重规划上的缺陷。
> - **Key Performance**: Tier 1静态任务成功率约85–95%，动态多智能体场景降到45–70%；在reactive JSSP单例上，Pass@1验证率仅GPT-4o 1/10、Claude 3.7 2/10、DeepSeek-R1 7/10。

> [!info] **Agent Summary**
> - **task_path**: 结构化/自然语言规划规格 + 动态扰动事件 → 计划/调度解 → 多维评测分数
> - **bottleneck**: 现有基准大多停留在静态、单体、短时推理，难以测出多智能体在共享资源、状态一致性与中途重规划中的真实失败点
> - **mechanism_delta**: 把规划能力拆成14个分层场景，并加入扰动、依赖和规模扩展，再用统一指标跨模型与框架比较
> - **evidence_signal**: 动态多智能体任务成功率显著低于静态任务，且reactive JSSP中主流LLM的Pass@1常出现可验证的约束违例
> - **reusable_ops**: [difficulty-tiering, disruption-injection, unified-metrics]
> - **failure_modes**: [隐状态或连续控制未被显式建模时覆盖不足, 提示工程与闭源API差异可能干扰横向比较]
> - **open_questions**: [如何量化长期状态一致性与事务回滚, 如何分离模型能力与agent框架工程增益]

## Part I：问题与挑战

这篇论文要解决的不是“再提一个更强的规划器”，而是**先把多智能体规划/调度到底难在哪里测清楚**。

### 1. 现有评测的真实缺口

作者认为现有 benchmark 的核心问题有三层：

1. **时间跨度不够长**  
   很多基准测的是单轮回答、静态计划或短程推理；真实系统却要处理长时程依赖，早期决策会影响后续可行性。

2. **协作性不够真**  
   多智能体系统的难点不只是“多个 agent 同时说话”，而是共享资源、角色分工、冲突消解、状态同步。

3. **扰动不够真实**  
   现实里会有机器故障、航班延误、路况变化、资源短缺。系统必须在执行中途重规划，而不是只在 t=0 给出一个漂亮答案。

### 2. 为什么现在必须补这块

原因很直接：AutoGen、CrewAI、LangGraph、Swarm 这类 agent framework 已经大量出现，但**大家缺少一个统一、可复现实验台**来回答两个关键问题：

- 多智能体真的比单体LLM更会规划吗？
- 它们到底强在静态求解，还是强在动态恢复？

如果没有 benchmark，系统改进很容易被 prompt 工程、框架包装或 case study 误导。

### 3. 输入/输出接口与边界条件

REALM-Bench 的评测接口大致是：

- **输入**：表格化/文本化任务规格，包含资源、时间窗、依赖关系、目标函数，以及可选的动态扰动事件
- **输出**：计划、调度表、重规划结果或执行策略
- **评分**：规划质量、约束满足、协作效果、计算效率、扰动恢复、鲁棒性等多维指标

**边界条件**也很明确：

- 它主要测**规划与调度**
- 主要基于**显式约束与事件**
- 不以低层感知、视觉 grounding、连续控制为主
- 更像是“agentic planning benchmark”，不是 embodied execution benchmark

---

## Part II：方法与洞察

### 1. 评测设计骨架

REALM-Bench 用两类任务族来构建覆盖面：

- **P-series**：更接近真实世界语义场景  
  如校园导览、城市拼车、婚礼物流、感恩节家庭协调、灾害救援、全球供应链
- **J-series**：更接近经典运筹优化问题  
  即 Job-Shop Scheduling Problem，便于做严格约束校验和 makespan 比较

这两个系列的组合很关键：

- P-series 提供**现实语义复杂度**
- J-series 提供**可验证的数学约束与最优性锚点**

### 2. 难度不是单轴，而是“层级 + 扩展轴”

作者把14个任务组织成5个 tier：

1. 单智能体静态规划  
2. 多智能体静态协同  
3. 复杂静态依赖  
4. 动态扰动处理  
5. 大规模多域集成  

此外，任务还可沿多个维度放大：

- 并行规划线程数
- 依赖关系复杂度
- 扰动频率
- 具体实例规模

这比单纯“题更大”更有价值，因为它能区分：

- 是**搜索空间变大**导致失败
- 还是**状态维护/重规划机制**本身就不行

### 3. 统一评分协议

论文给出的主要评测维度包括：

- 规划质量
- 最优性/资源效率
- 协调有效性
- 约束满足率
- 计算效率
- 扰动适应能力
- 方案鲁棒性

这意味着它不只看“答没答出来”，而是看：

- 计划是否可执行
- 是否违反 deadline / precedence / capacity
- 出现扰动后多久恢复
- 多 agent 之间是否互相打架

### 核心直觉

**它真正改变的不是求解器，而是“测量瓶颈”。**

从因果链条看：

- **what changed**：把评测从“静态单轮计划输出”改成“带共享资源、依赖关系和中途扰动的持续规划任务”
- **which bottleneck changed**：原来 benchmark 测不出状态跟踪、事务一致性、协作冲突和提前干预能力；现在这些都被任务结构显式放大
- **what capability changed**：benchmark 可以区分“会写计划”与“能维护一个在扰动下仍然有效的计划”

两个例子很能说明问题：

- **P8 婚礼物流 with disruptions**：路封发生后，系统必须知道每辆车“当前在哪”，否则无法判断谁受影响  
- **P9 感恩节 with flight delay**：延误信息提前数小时就已可见，真正能力不是在原定到达时才反应，而是能否**提前重规划**

也就是说，这个 benchmark 专门把**隐含状态维护**和**时机敏感的重规划**变成显性测试项。  
这是它最有价值的地方。

### 4. 为什么这种设计有效

因为很多 LLM/agent 系统在静态场景里可以“语言上合理”，但一旦进入动态场景，就会出现可验证的错误：

- 违反工序顺序
- 忘记机器正在维修
- 没有追踪资源当前位置
- 没有利用提前可得的扰动信息
- 多 agent 输出互相冲突

REALM-Bench 通过 JSSP 这类强约束任务，把这些错误从“主观看起来不聪明”变成“客观不可执行”。

### 5. 战略取舍

| 设计选择 | 带来的能力 | 代价/风险 |
|---|---|---|
| P-series 真实语义场景 | 能测真实任务分解与协作理解 | 结果更容易受描述方式与 prompt 影响 |
| J-series 经典JSSP | 可严格检查约束与 makespan | 对开放世界语义覆盖较弱 |
| 动态扰动注入 | 能暴露状态维护与恢复能力 | 复现实验依赖事件脚本设计 |
| 多框架统一接口 | 便于横向比较 LangGraph/AutoGen/CrewAI 等 | 工程实现差异可能掩盖模型本体差异 |
| 多指标评分 | 评测更全面 | 总分聚合与权重解释更复杂 |

---

## Part III：证据与局限

### 1. 关键证据：这个 benchmark 确实能“测出差异”

**信号 A：动态任务显著更难。**  
作者的初步结果显示：

- Tier 1 单智能体静态任务：约 **85–95% success**
- 动态多智能体场景：约 **45–70% success**

这不是单纯“题更长”，而是说明**扰动恢复 + 协同重规划**确实是当前系统的能力断层。

**信号 B：reactive JSSP 能直接抓到可验证错误。**  
在附录的 MiniFactory reactive JSSP 例子里，Pass@1 常见错误包括：

- job order 违例
- machine repair 期间仍安排操作
- operation 被错误拆分

代表性结果：

- GPT-4o：Pass@1 validation rate **1/10**
- Claude 3.7：**2/10**
- DeepSeek-R1：**7/10**

这说明 benchmark 不只是“分高分低”，而是能定位**错在哪里**。

**信号 C：它能测出方法-数据分布交互，而不是只给出单一排名。**  
JSSP leaderboards 中，作者报告：

- 在 **DMU** 上，ALAS-static 表现更好
- 在 **TA** 上，ALAS-dynamic 更有优势

这说明 benchmark 对“静态优化能力”和“动态反应能力”的区分是有分辨率的，不是所有任务都被同一种策略统治。

### 2. 这篇论文最值得记住的结论

如果只看一句话，就是：

> 当前 LLM agent 的真正瓶颈，不是不会描述计划，而是**不能在共享约束与动态扰动下持续维护一份有效计划**。

REALM-Bench 的价值，在于把这个瓶颈从经验判断变成了**可系统测量的 benchmark 问题**。

### 3. 局限性

- **Fails when**: 任务难点主要来自视觉感知、部分可观测环境、连续控制或真实执行闭环时，REALM-Bench 的文本/结构化规格覆盖不足。
- **Assumes**: 问题可被显式规格化为表格/YAML/JSON约束，扰动以事件形式被外显给系统；代表性 baseline 还依赖 GPT-4o、Claude-3.7、Gemini、DeepSeek 等闭源或API模型，以及特定 agent framework，复现存在成本与接口依赖。
- **Not designed for**: 开放式社交协商、多模态 embodied 任务、以及无法以 makespan/约束满足/恢复效率这类指标客观校验的开放生成场景。

再补两条更具体的论文级局限：

1. **设计覆盖度 > 正文实证覆盖度**  
   虽然 benchmark 定义了14个任务，但正文里最细的实验主要集中在 J1/J2 与若干 JSSP leaderboard；P-series 的完整对比更多依赖附录或仓库索引。

2. **作者强调 transaction/rollback，但量化还不够细**  
   论文在动机部分强调 transactional integrity、compensation、rollback、state consistency，但公开展示的核心量化指标仍主要是 success rate、makespan、constraint satisfaction。  
   也就是说，**“事务性可靠性”是强动机，但还不是最成熟的评分子系统**。

### 4. 可复用组件

这篇工作最可复用的不是某个单独模型，而是下面这些 benchmark 设计模式：

- **difficulty-tiering**：把能力缺口按层级展开，而不是只堆更大实例
- **disruption injection**：把路封、机器故障、航班延误等事件做成标准化 stress test
- **dual-track evaluation**：用真实语义任务 + OR强约束任务共同测
- **structured logging**：保留执行轨迹，便于做错误归因
- **framework adapters**：给 LangGraph、AutoGen、CrewAI、Swarm 等统一接入面

如果你以后要做 agent benchmark，这几乎是一套可以直接复用的模板。

![[paperPDFs/Agentic_Reasoning_Benchmarks_Core_Mechanisms_of_Agentic_Reasoning_Memory_and_Planning_Planning_and_Feedback/arXiv_2025/2025_REALM_Bench_A_Benchmark_for_Evaluating_Multi_Agent_Systems_on_Real_world_Dynamic_Planning_and_Scheduling_Tasks.pdf]]