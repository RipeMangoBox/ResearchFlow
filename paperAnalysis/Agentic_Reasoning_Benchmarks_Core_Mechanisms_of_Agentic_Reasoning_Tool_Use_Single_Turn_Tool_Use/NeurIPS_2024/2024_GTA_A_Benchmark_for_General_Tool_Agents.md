---
title: "GTA: A Benchmark for General Tool Agents"
venue: NeurIPS
year: 2024
tags:
  - Survey_Benchmark
  - task/tool-agent-evaluation
  - executable-tool-chain
  - multimodal-context
  - fine-grained-metrics
  - dataset/GTA
  - opensource/full
core_operator: 以真人编写的隐式多步查询、真实可执行工具和多模态上下文，配合参考工具链与双模式评分来诊断通用工具代理能力
primary_logic: |
  真实工具代理评测目标 → 构造229个带真实图像上下文的人写隐式查询并为每题标注可执行参考工具链 → 用分步指标与端到端执行指标联合评分 → 揭示当前LLM在真实场景下的规划、参数生成与格式遵循边界
claims:
  - "在 GTA 上，没有任何被评估模型的端到端 AnsAcc 超过 50%；GPT-4-1106-Preview 为 46.59%，多数模型低于 25% [evidence: comparison]"
  - "在分步指标中，ArgAcc 与端到端 AnsAcc 的 Pearson 相关性最高，说明参数预测是 GTA 上的主要瓶颈 [evidence: analysis]"
  - "基于 ReAct/JSON 的 Agent-FLAN 微调能显著提升 Llama-2-Chat-7B 在 GTA 上的 InstAcc（30.86→71.60）和 ToolAcc（16.34→41.11），但 ArgAcc 仍仅为 6.82 [evidence: comparison]"
related_work_position:
  extends: "GAIA (Mialon et al. 2023)"
  competes_with: "ToolBench (Qin et al. 2023); m&m’s (Ma et al. 2024)"
  complementary_to: "Agent-FLAN (Chen et al. 2024); ReAct (Yao et al. 2023)"
evidence_strength: moderate
pdf_ref: paperPDFs/Agentic_Reasoning_Benchmarks_Core_Mechanisms_of_Agentic_Reasoning_Tool_Use_Single_Turn_Tool_Use/NeurIPS_2024/2024_GTA_A_Benchmark_for_General_Tool_Agents.pdf
category: Survey_Benchmark
---

# GTA: A Benchmark for General Tool Agents

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2407.08713), [Project](https://open-compass.github.io/GTA/), [Code](https://github.com/open-compass/GTA)
> - **Summary**: 这篇论文提出 GTA，一个更接近真实使用场景的工具代理基准：真人编写的隐式查询、真实部署工具、真实多模态上下文，再加上可执行参考工具链，使评测从“会不会说要调用工具”升级为“能不能真正把工具链跑通”。
> - **Key Performance**: GPT-4-1106-Preview 端到端 AnsAcc 仅 46.59%；开源最佳 Qwen1.5-72B-Chat 仅 13.32%

> [!info] **Agent Summary**
> - **task_path**: 真实用户查询 + 1-2 张图像上下文 + 可调用工具集 -> 工具链执行过程与最终文本/图像结果
> - **bottleneck**: 多步隐式工具调用中的参数生成与协议遵循，而不是单纯“选对工具名”
> - **mechanism_delta**: 把评测从 AI 生成/单步/虚拟工具/纯文本，改成真人隐式查询 + 真实工具 + 多模态输入 + 参考工具链的过程/结果联合评测
> - **evidence_signal**: 16 个 LLM 中最强的 GPT-4 也未达 50% AnsAcc，且 ArgAcc 与 AnsAcc 相关性最高
> - **reusable_ops**: [隐式真实查询构造, 可执行参考工具链标注]
> - **failure_modes**: [参数JSON格式错误, 只输出思考或向用户追问而不执行动作]
> - **open_questions**: [如何同时提升参数值正确性与格式合法性, 如何扩展到多语言与更开放的工具生态]

## Part I：问题与挑战

这篇工作要解决的不是“LLM 会不会调用 API”，而是更难也更真实的问题：

**当用户只给出一个看似简单、但步骤隐含、工具隐含、上下文可能是图片/网页截图/表格/手写材料时，模型能否像通用工具代理一样完成端到端任务？**

### 现有评测的真实缺口

作者认为已有工具使用 benchmark 与真实世界存在 4 个系统性偏差：

1. **查询不真实**  
   很多 benchmark 用 AI 生成 query，常把步骤或工具类型直接写进问题里，模型更像在做“模板匹配”。

2. **任务过于单步**  
   现实任务通常要先感知、再检索、再计算、再操作，但已有评测常只测某一跳。

3. **工具不是真执行**  
   虚拟工具或文本描述工具，测不到真实调用中的格式错误、参数错误、链路中断。

4. **交互只看文本**  
   真实任务往往依赖图像、网页截图、表格、手写公式等多模态输入，纯文本评测低估了感知-推理-执行耦合难度。

### GTA 的输入/输出接口

GTA 的单个样本由五部分组成：**文件 F、查询 Q、涉及工具 T、参考工具链 C、答案 A**。

- **输入**：
  - 1~2 张真实图片文件
  - 一个真人写的自然语言查询
  - 一个封闭工具集中的若干工具
- **输出**：
  - 工具调用序列
  - 最终文本答案或图像结果

查询分三类：

- **objective**：唯一答案
- **subjective**：开放文本答案
- **image generation**：目标是生成图像，不直接评图像本身，而评相关参数是否正确

### 边界条件

GTA 不是无限开放环境，而是一个**高真实性但闭集化**的 benchmark：

- 仅覆盖 **14 个工具**
- 工具分为 **Perception / Operation / Logic / Creativity** 四类
- 数据集总共 **229 个任务、252 张图片、557 次工具调用**
- 全部 query 为 **英文**
- 整个数据集本质上是 **test benchmark**，不是训练集

所以它测的是：**在一个受控但逼真的工具生态中，LLM 作为中央控制器的真实执行上限。**

---

## Part II：方法与洞察

### Benchmark 设计总览

GTA 的核心不是发明新 agent，而是**发明一种更能暴露 agent 真问题的测量方式**。它有三个“Real”：

1. **Real user queries**
   - 查询由人写，不是 AI 生成
   - 目标清晰，但步骤隐含、工具隐含
   - 强制模型自己推断该用什么工具、按什么顺序调用

2. **Real deployed tools**
   - 真正部署的可执行工具，而不是文字模拟
   - 每题有可执行的 ground-truth tool chain
   - 可逐步比对工具名、参数、返回、最终答案

3. **Real multimodal inputs**
   - 输入是实际图片文件：场景图、网页截图、表格、代码片段、印刷/手写材料等
   - 让评测更贴近用户真实需求而非 toy task

### 数据构造流程

作者用两步法构造 benchmark。

#### 1. Query construction

- 专家先给出若干 exemplar
- 标注员基于 exemplar 做 **diversified expansion**
- 约束包括：
  - 必须可由给定工具集解决
  - query 中不能显式泄露工具名
  - 至少需要 2 步
  - 必须有现实意义

对于 `GoogleSearch` 类题目，还额外约束：

- 问题应依赖检索，而不是 LLM 内部常识
- 要指定时间、站点或组织，降低答案漂移

#### 2. Tool-chain construction

- 3 位计算机专业标注员实际调用已部署工具
- 以 ReAct 风格 JSON 记录：
  - user query
  - assistant thought / tool call
  - tool return
  - final answer
- 如果工具本身在该题上不稳定或识别错误，**直接丢弃该题**

这一步很关键：  
它让 benchmark 测的更偏向 **agent 控制能力**，而不是工具识别器本身的偶然失误。

### 评测协议

GTA 使用两种模式：

#### 1. Step-by-step mode
给模型参考工具链前 n 步，让它预测第 n+1 步。  
作用：**细粒度定位错误发生在哪一层。**

指标包括：

- **InstAcc**：是否按协议执行
- **ToolAcc**：工具是否选对
- **ArgAcc**：参数是否正确
- **SummAcc**：能否基于历史步骤总结最终答案

#### 2. End-to-end mode
模型真正自己调用工具、逐步求解。  
作用：**测实际 agent 的任务完成能力。**

指标包括：

- **AnsAcc**：文本答案正确率
- **AnsAcc w/ ImgGen**：间接纳入图像生成类任务
- 各工具类别的 F1：Perception / Operation / Logic / Creativity

### 核心直觉

GTA 的关键洞察是：

**已有 benchmark 往往只测“模型会不会说出正确工具名”；GTA 则把测量对象改成“模型能不能在真实、多模态、隐式、多步、可执行的链路里持续做对每一步”。**

这背后的因果链是：

- **把 query 从显式提示改成隐式真实需求**  
  → 去掉工具/步骤泄漏  
  → 才能测到真正的规划与工具推断能力

- **把工具从文本模拟改成真实执行**  
  → 暴露协议遵循、参数格式、返回值利用等执行级问题  
  → 才能测到 agent 的真实落地能力

- **把评估从只看最终答案改成过程+结果联合**  
  → 能区分“没选对工具”“参数错了”“格式错了”“最后总结错了”  
  → benchmark 从排行榜变成诊断仪器

换句话说，GTA 改变的不是模型分布，而是**测量分辨率**：  
它把原本被“答案对/错”掩盖的中间失败模式显性化了。

### 为什么这个设计有效

因为现实 agent 的失败通常不是“完全不会”，而是：

- 会想，但不行动
- 会选工具，但参数不合法
- 参数格式对了，但值不对
- 中间步骤对了，最后总结错了

如果 benchmark 不记录可执行工具链，只看最终答案，这些失败会被混成一个黑箱。  
GTA 通过参考工具链 + 双模式评测，把这个黑箱拆开了。

### 战略取舍

| 设计选择 | 改变了什么测量瓶颈 | 获得的诊断能力 | 代价 |
|---|---|---|---|
| 人写隐式查询 | 去掉 query 中的工具/步骤泄漏 | 更真实地测规划与推断 | 标注成本高、规模较小 |
| 真实部署工具 | 从“文本模拟调用”变成“实际执行” | 可暴露格式/参数/执行链错误 | 依赖工具服务稳定性 |
| 多模态真实文件 | 从纯文本转向感知-推理-操作耦合 | 能测 perception + tool use 联动 | 任务更难、模型更易早期失败 |
| 参考工具链标注 | 从只看结果变成可解释过程评估 | 可定位具体失败层级 | 标注复杂、维护成本高 |
| Step-by-step + End-to-end 双模式 | 同时获得局部诊断与整体完成率 | 可分析哪个局部瓶颈拖累整体 | 实验流程更重 |

---

## Part III：证据与局限

### 关键证据信号

**1. 比较信号：真实工具任务远比已有评测更难。**  
在 16 个模型上，最好的 **GPT-4-1106-Preview 端到端 AnsAcc 也只有 46.59%**；多数模型低于 25%，开源最佳 **Qwen1.5-72B-Chat 仅 13.32%**。  
这说明 GTA 不是简单“加点图片”的 benchmark，而是真把工具代理的现实瓶颈测出来了。

**2. 分析信号：真正卡住系统的是参数预测，不只是工具选择。**  
作者比较了 InstAcc / ToolAcc / ArgAcc / SummAcc 与 AnsAcc 的相关性，发现 **ArgAcc 与最终 AnsAcc 的相关性最高**。  
这非常重要：说明 agent 失败的主因往往不是“不知道要调哪个工具”，而是**知道要调，但参数给不对、给不合法、给得不可解析**。

**3. 对照信号：更会选工具，不等于更能完成任务。**  
GPT-4o 的 InstAcc 和 ToolAcc 高于 GPT-4，但因为 ArgAcc 更弱，最终 AnsAcc 反而低于 GPT-4。  
这支持了论文的核心诊断：**参数层是木桶短板。**

**4. 错误分析信号：不同模型家族有不同失效风格。**
- **GPT 系列**：更稳，格式遵循强，但有时会“想太多”，不行动，甚至向用户追问
- **Qwen 系列**：更保守，调用次数少，但成功率较高
- **Yi / Deepseek 系列**：更激进，爱调用工具，但格式跟随能力弱

这说明 GTA 不仅能排榜，还能分析**行为风格**。

**5. 探索实验：ReAct/JSON 微调能修一部分问题，但修不完。**  
Agent-Flan-7B 相比 Llama-2-Chat-7B，InstAcc 从 **30.86 提升到 71.60**，ToolAcc 从 **16.34 提升到 41.11**。  
但 **ArgAcc 仍只有 6.82**。结论很清楚：  
**格式训练能改善“会不会按协议说话”，却不自动解决“参数值是否真的对”。**

### 1-2 个最值得记住的指标

- **GPT-4-1106-Preview：AnsAcc = 46.59%**
- **Qwen1.5-72B-Chat（开源最佳）：AnsAcc = 13.32%**

这两个数已经足够说明 GTA 的难度和区分度。

### 局限性

- **Fails when**: 超出其 14 工具闭集、需要多语言查询、依赖持续变化的网页结果、或需要直接评估图像生成质量时，GTA 的覆盖就不够；另外，若工具本身在某类图像上识别不稳，数据构造阶段会直接剔除该题，导致 benchmark 更偏向“工具可解样本”。
- **Assumes**: 假设存在可稳定部署的真实工具服务，且 GoogleSearch 等外部依赖在指定时间/站点约束下基本可复现；假设人类可提供高质量参考工具链；实验依赖 OpenCompass + Lagent + GPU 环境，且人工标注成本明显高于 AI 生成数据。
- **Not designed for**: 不是面向长期多轮交互、开放世界无限工具生态、真实 GUI 环境操作、或审美层面的图像生成评测；其 image generation 任务本质上评的是参数是否正确，而不是生成图像的感知质量。

### 可复用组件

- **可执行参考工具链 schema**：适合别的 agent benchmark 直接复用
- **Step-by-step / End-to-end 双协议**：一个测诊断，一个测真实完成率
- **错误类型分析框架**：可分离 no-action、format error、argument format error 等 failure mode
- **数据与代码开放**：GitHub 已发布，论文 datasheet 还给出维护、更新与许可信息

### So what

GTA 的价值不只是“又一个 benchmark”，而是它把工具代理研究的关注点从：

- API 覆盖率
- 单步 tool calling
- 最终答案排行榜

推进到：

- 真实隐式任务下的规划能力
- 参数生成与协议遵循
- 感知-检索-逻辑-操作的链式脆弱性

如果后续工作想真正提升通用工具代理，GTA 给出的方向很明确：  
**优先修参数预测和可解析执行，而不是只提升工具选择或 CoT 表达。**

![[paperPDFs/Agentic_Reasoning_Benchmarks_Core_Mechanisms_of_Agentic_Reasoning_Tool_Use_Single_Turn_Tool_Use/NeurIPS_2024/2024_GTA_A_Benchmark_for_General_Tool_Agents.pdf]]