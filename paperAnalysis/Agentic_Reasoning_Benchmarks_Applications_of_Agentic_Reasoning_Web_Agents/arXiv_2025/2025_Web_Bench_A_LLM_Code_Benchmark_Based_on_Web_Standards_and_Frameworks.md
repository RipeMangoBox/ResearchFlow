---
title: "Web-Bench: A LLM Code Benchmark Based on Web Standards and Frameworks"
venue: arXiv
year: 2025
tags:
  - Survey_Benchmark
  - task/code-generation
  - task/agentic-coding
  - dependency-chained-tasks
  - e2e-testing
  - project-level-benchmark
  - dataset/Web-Bench
  - opensource/full
core_operator: "用顺序依赖的 Web 项目任务链与 Playwright 端到端验收，把 Web 标准/框架知识转成可量化的项目级编码能力评测"
primary_logic: |
  代码基准饱和与真实工程脱节 → 构造覆盖 Web 标准与框架的 50 个项目、每个项目含 20 个顺序依赖任务与 E2E 用例 → 通过 Web-Agent 按任务增量生成/修复代码并在失败后基于错误上下文重试 → 用 pass@1/pass@2 揭示 LLM 在真实 Web 项目中的持续开发能力边界
claims:
  - "Web-Bench 将代码评测从独立题目推进到 50 个项目 × 20 个顺序依赖任务的项目级工作流模拟，共 1000 个任务 [evidence: analysis]"
  - "在论文提供的 Web-Agent 与 Best-of-5 设定下，Claude 3.7 Sonnet Thinking 仅达到 25.1% Pass@1 / 35.3% Pass@2，低于同时期 SWE-Bench Verified 的 65.4% Pass@1 与 Full 的 33.8% Pass@1，说明该基准更不易饱和 [evidence: comparison]"
  - "Web-Bench 能区分推理模式与模型封闭性：同系列模型开启 thinking 往往更强，且闭源模型平均 Pass@2 高于开源模型（20.79% vs 14.84%） [evidence: comparison]"
related_work_position:
  extends: "SWE-Bench (Jimenez et al. 2023)"
  competes_with: "SWE-Bench; RepoBench"
  complementary_to: "EvalPlus; MLE-Bench"
evidence_strength: moderate
pdf_ref: "paperPDFs/Agentic_Reasoning_Benchmarks_Applications_of_Agentic_Reasoning_Web_Agents/arXiv_2025/2025_Web_Bench_A_LLM_Code_Benchmark_Based_on_Web_Standards_and_Frameworks.pdf"
category: Survey_Benchmark
---

# Web-Bench: A LLM Code Benchmark Based on Web Standards and Frameworks

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2505.07473), [GitHub](https://github.com/bytedance/web-bench), [Hugging Face Dataset](https://huggingface.co/datasets/bytedance-research/Web-Bench)
> - **Summary**: 这篇工作把代码评测从“写单个函数/修单个 issue”升级为“在真实 Web 项目里连续实现 20 个功能”，用顺序依赖任务和浏览器级 E2E 测试更真实地暴露 LLM 的工程短板。
> - **Key Performance**: 最强模型 Claude 3.7 Sonnet Thinking 也只有 **25.1% Pass@1 / 35.3% Pass@2**；闭源模型平均 **Pass@2 20.79%**，高于开源模型的 **14.84%**。

> [!info] **Agent Summary**
> - **task_path**: 自然语言任务描述 + 当前项目文件 + 构建/测试错误上下文 -> 更新后的 Web 项目代码
> - **bottleneck**: 现有代码基准已趋于饱和，且缺少跨任务依赖、浏览器运行时和框架约束，无法有效测真实工程开发能力
> - **mechanism_delta**: 把独立题目评测改成 50 个 Web 项目上的顺序依赖任务链，并用 Playwright E2E + 单次重试来测持续实现与修复能力
> - **evidence_signal**: Claude 3.7 Sonnet Thinking 在参考 Web-Agent 上仅 25.1% Pass@1，明显低于 SWE-Bench Verified 的 65.4%
> - **reusable_ops**: [顺序依赖任务链, Playwright端到端验收]
> - **failure_modes**: [早期任务失败导致后续任务无法继续评测, 长上下文输入被截断后跨文件依赖信息丢失]
> - **open_questions**: [如何把页面美学与代码质量纳入统一评分, 如何降低参考agent与商业API配置对榜单的耦合]

## Part I：问题与挑战

这篇论文要解决的不是“模型会不会写代码”，而是**现有代码 benchmark 已经很难继续区分模型真实工程能力**。

### 1）真问题是什么

早期代码基准主要测单函数或小题目正确率，例如 HumanEval、MBPP。问题是这些 benchmark 已经接近饱和：  
- HumanEval SOTA 达到 **99.4%**
- MBPP SOTA 达到 **94.2%**

这会带来两个后果：

1. **区分度下降**：高分不再代表真实工程能力提升。  
2. **优化方向偏移**：模型可能被推向“刷题式代码生成”，而非真实项目开发。

作者认为，真正该测的是：  
**模型能否在一个持续演化的项目里，理解已有代码、遵守 Web 标准/框架约束、逐步实现功能，并在失败后修复。**

### 2）现有软件工程 benchmark 还缺什么

SWE-Bench、RepoBench 已经比 HumanEval 更接近工程实践，但作者指出它们仍有一个关键缺口：

- 任务通常仍是**相对独立**的；
- 缺少**顺序依赖**；
- 很难模拟人类开发中的“前一步实现影响后一步”的工作流。

而真实 Web 开发恰恰最依赖这些因素：
- 页面结构与样式的历史累积
- 前后端接口、状态管理、路由、数据库之间的耦合
- 框架版本/API 习惯
- 浏览器运行时行为是否真的正确

### 3）输入/输出接口与边界条件

| 项目 | 内容 |
|---|---|
| **输入** | 任务描述 + 当前项目文件 + 可选错误上下文 |
| **输出** | 模型返回要修改/新增的代码文件 |
| **执行环境** | Docker + 构建检查 + Playwright E2E 测试 |
| **评测对象** | Web 全栈项目开发能力 |
| **终止条件** | 某任务第二次尝试仍失败，则该项目评测结束 |

边界上，这个 benchmark 明确聚焦于：
- **Web Standards**：HTML/CSS/JS/DOM/SVG/Canvas/WebGL/TypeScript 等
- **Web Frameworks**：React/Vue/Angular/Svelte、Redux/Zustand、Next.js/Nuxt/Express/Fastify、Prisma、Vite 等

也就是说，它不是泛编程 benchmark，而是**真实 Web 工程 benchmark**。

---

## Part II：方法与洞察

### 1）核心设计：把 benchmark 单位从“题目”升级为“项目”

Web-Bench 的基本结构是：

- **50 个项目**
- 每个项目 **20 个顺序依赖任务**
- 每个任务配有若干 **端到端测试**
- 平均每个项目：
  - **23.4 个文件**
  - **1947.9 行代码**
  - **72.4 个测试用例**
- 人类资深工程师完成单个项目平均需要 **4–8 小时**

这意味着它测的不是一次性生成，而是**持续开发轨迹**。

### 2）为什么选“标准 + 框架”这两个轴

作者的设计哲学很明确：

- **标准（Standards）** = 某开发领域的基础知识  
- **框架（Frameworks）** = 某开发领域的效率工具

因此，若 benchmark 真想贴近现实，不能只考“会不会写 JS/Python”，而必须测：
- 是否理解 Web 标准约束
- 是否熟悉框架 API 与常见模式
- 是否能在复杂依赖关系中保持一致性

这比“多语言扩展”更贴近真实开发，也比单纯“增加测试覆盖率”更能提升任务本身复杂度。

### 3）评测协议：不是只看生成对不对，而是看能不能一路做下去

Web-Agent 的工作流很简单但有代表性：

1. 读取任务描述、当前文件、错误信息
2. 调用 LLM 生成修改后的文件
3. 写回项目
4. 初始化/构建环境
5. 运行 Playwright E2E 测试
6. 如果失败，带错误上下文再试一次
7. 若再次失败，则该项目停止

对应两个核心指标：

- **Pass@1**：第一次尝试能连续完成多少任务
- **Pass@2**：允许一次带错误上下文的重试后，能连续完成多少任务

这里最关键的不是公式，而是它的含义：  
**分数表示“项目链条能推进多远”，而不只是“某一题是否答对”。**

### 核心直觉

作者真正改变的“旋钮”是：

> 从“局部、独立、函数级正确性评测”  
> 变成“有历史依赖、跨文件状态、浏览器运行时、框架约束的项目级持续实现评测”。

这会直接改变被测分布：

- **原来测的是**：局部代码补全、函数逻辑、基础 API 记忆
- **现在测的是**：
  - 历史代码理解
  - 多文件一致性
  - 前后任务依赖传播
  - 浏览器/构建/路由/状态管理的集成行为
  - 框架特定知识与版本习惯

能力变化也很明确：

- 从“会写局部代码”
- 变成“能否在真实项目里持续交付功能并修错”

这也是为什么该设计能显著降低 benchmark 饱和：  
**它把评测瓶颈从 token-level 代码生成，移动到了 system-level 工程一致性。**

### 4）策略性权衡

| 设计选择 | 带来的诊断能力 | 代价/风险 |
|---|---|---|
| 顺序依赖 20 任务 | 能测长期规划、历史一致性、错误传播 | 早期失败会截断后续评测，分数更脆弱 |
| Playwright E2E 测试 | 能测用户可见行为与全栈集成 | 定位粒度较粗，难区分“接近正确”和“完全错误” |
| 覆盖标准 + 框架 | 更贴近真实 Web 工程知识结构 | 生态面很广，当前覆盖仍不完整 |
| 参考 Web-Agent + retry | 评测流程可复现，且更接近人类调试 | 分数受 prompt、context 截断、API 参数影响 |
| 人工设计并校准项目 | 任务更像真实项目、测试更有针对性 | 构建成本高，扩展到更大规模较慢 |

---

## Part III：证据与局限

### 关键证据信号

- **信号 1｜跨 benchmark 对比（comparison）**  
  Web-Bench 上最强结果只有 **25.1% Pass@1**（Claude 3.7 Sonnet Thinking, Web-Agent, Best-of-5），明显低于 SWE-Bench Verified 的 **65.4%**、SWE-Bench Full 的 **33.8%**，更远低于 HumanEval/MBPP 的高饱和区间。  
  **结论**：它确实重新拉开了模型差距，至少当前还没有被“刷穿”。  

- **信号 2｜重试与思考模式（comparison）**  
  同系列模型启用 thinking 后，整体 **Pass@2 更高**。  
  **结论**：这个 benchmark 不只测一次性代码回忆，还对“错误分析 + 二次修复”更敏感。**

- **信号 3｜闭源 vs 开源（comparison）**  
  闭源模型平均 **Pass@2 = 20.79%**，开源模型平均 **14.84%**。  
  **结论**：在长链路工程任务里，模型总体能力差距仍明显，没有出现“小模型也能靠模板刷高分”的现象。**

- **信号 4｜框架/标准差异能被测出来（analysis）**  
  附录中的分项结果显示：
  - React 通常比 Angular 更容易
  - Zustand 往往比 Redux/Jotai 更好做
  - Express.js 相对稳定，Next.js 因 API/范式变化更容易混淆
  - Prisma 等更复杂抽象常更难  
  **结论**：Web-Bench 不是单纯“大题更难”，而是能暴露具体的框架知识盲点。**

### 1-2 个最关键指标

1. **25.1% Pass@1 / 35.3% Pass@2**：说明项目级 Web 开发远未被现有 LLM 解决。  
2. **20.79% vs 14.84% 平均 Pass@2（闭源 vs 开源）**：说明 benchmark 仍有足够分辨率来区分模型族能力。

### 局限性

- **Fails when**: 需要评估非 Web 域工程能力、页面美学质量、代码可维护性/可读性、长期多人协作流程时，这个 benchmark 不能给出完整结论；另外，若模型在早期任务失败，后续能力会被链式截断而无法观察。
- **Assumes**: 固定的 Web-Agent 交互范式、Docker + Playwright 运行环境、供应商 API 参数、Best-of-5 采样预算，以及人工设计/校准的项目与测试；当上下文超长时输入会被截断，这会影响跨文件任务的公平性。
- **Not designed for**: 非 Web 技术栈、纯算法题、UI 审美评测、代码风格与架构优雅度评测，也不专门分离“agent 设计能力”和“底层模型能力”。

### 对可复现性/扩展性有影响的依赖

- 结果高度依赖 **商业 API 模型**
- 不同提供商的 **context length / temperature / max token** 并不统一
- 使用 **Best-of-5** 会引入采样预算差异
- E2E 评测依赖 **Docker + 浏览器自动化环境**
- 当前项目仍是人工设计，扩容成本不低

### 可复用组件

- **顺序依赖任务链**：适合迁移到移动端、桌面端、数据应用等工程 benchmark
- **Playwright E2E 验收框架**：适合一切带真实 UI/交互的代码评测
- **HTTP-Agent / Local-Agent 协议**：便于接不同 agent 系统
- **项目校准流程**：可复用到未来 benchmark 的任务消歧与 testcase 审核

**一句话评价**：  
Web-Bench 的真正贡献不是又造了一个“更难题库”，而是把代码评测的测量对象改成了**真实 Web 工程中的连续交付能力**；这让它比已饱和的函数级 benchmark 更有诊断价值，也更能指导下一阶段的 agentic coding 研究。

## Local PDF reference

![[paperPDFs/Agentic_Reasoning_Benchmarks_Applications_of_Agentic_Reasoning_Web_Agents/arXiv_2025/2025_Web_Bench_A_LLM_Code_Benchmark_Based_on_Web_Standards_and_Frameworks.pdf]]