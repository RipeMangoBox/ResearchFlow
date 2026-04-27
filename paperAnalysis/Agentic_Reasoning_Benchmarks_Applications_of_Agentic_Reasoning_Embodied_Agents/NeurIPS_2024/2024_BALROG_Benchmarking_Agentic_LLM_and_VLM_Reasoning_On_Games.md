---
title: "BALROG: Benchmarking Agentic LLM and VLM Reasoning On Games"
venue: ICLR
year: 2025
tags:
  - Survey_Benchmark
  - task/MLLM-evaluation
  - long-horizon-evaluation
  - progression-metric
  - procedural-generation
  - dataset/BALROG
  - opensource/full
core_operator: 聚合六个程序生成的长时程游戏，并以统一交互接口与0–100细粒度进度分数测量LLM/VLM代理能力边界
primary_logic: |
  评测长时程 agentic 推理能力 → 聚合六个程序生成游戏并统一成自然语言/视觉交互接口 → 用标准化进度分数与 NetHack 数据驱动进度曲线衡量轨迹质量 → 揭示模型在规划、探索、空间推理与视觉决策上的真实上限
claims:
  - "在 BALROG 的六个环境上，当前前沿 LLM/VLM 只在较容易游戏上取得部分进展，而在最难任务上几乎无有效推进；其中 NetHack 的最佳受测模型 o1-preview 平均进度仅 1.57% [evidence: comparison]"
  - "加入视觉观测往往会伤害而非提升代理表现；例如 GPT-4o 在 BALROG 上的平均进度从纯语言模式的 32.34% 降至视觉-语言模式的 22.56% [evidence: comparison]"
  - "论文提出的 NetHack 数据驱动进度指标基于地牢层数与经验等级，比原始游戏分数更能反映长时程实际推进程度 [evidence: analysis]"
related_work_position:
  extends: "SmartPlay (Wu et al. 2023)"
  competes_with: "AgentBench (Liu et al. 2023b); SmartPlay (Wu et al. 2023)"
  complementary_to: "WebArena (Zhou et al. 2023); OfficeBench (Wang et al. 2024)"
evidence_strength: moderate
pdf_ref: paperPDFs/Agentic_Reasoning_Benchmarks_Applications_of_Agentic_Reasoning_Embodied_Agents/NeurIPS_2024/2024_BALROG_Benchmarking_Agentic_LLM_and_VLM_Reasoning_On_Games.pdf
category: Survey_Benchmark
---

# BALROG: Benchmarking Agentic LLM and VLM Reasoning On Games

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2411.13543), [Project/Leaderboard](https://balrogai.com)
> - **Summary**: BALROG 用六类程序生成、长时程游戏和统一进度评分，把当前 LLM/VLM 作为代理时在探索、规划、空间推理与视觉决策上的短板系统性暴露出来。
> - **Key Performance**: Claude 3.5 Sonnet 在语言模式平均进度为 32.64%；最难的 NetHack 上最佳模型 o1-preview 也仅 1.57% 平均进度。

> [!info] **Agent Summary**
> - **task_path**: 长时程游戏观测（文本或图像+历史）/零样本代理评测 -> 自然语言动作 -> 0–100任务进度
> - **bottleneck**: 缺少能稳定测出长时程规划、探索、视觉决策与“知道但做不到”问题的统一 agent benchmark
> - **mechanism_delta**: 将六个异质 RL 游戏统一为同一交互接口，并用细粒度进度评分尤其是 NetHack 数据驱动进度曲线替代粗糙成败判断
> - **evidence_signal**: 跨6环境的零样本对比显示整体平均进度仅约三成，且 NetHack 最佳仅 1.57%，多模型加视觉后反而退化
> - **reusable_ops**: [统一环境包装器, 数据驱动进度评分]
> - **failure_modes**: [系统性探索失败, 视觉输入引起决策退化]
> - **open_questions**: [如何在超长上下文下做可负担的few-shot/ICL, 如何把视觉信息稳定转成行动策略]

## Part I：问题与挑战

（注：你给的元数据写的是 NeurIPS 2024，但论文正文首页明确标注为 ICLR 2025；这里按正文信息记录。）

这篇论文要解决的**真问题**不是“模型能不能在单轮问答里说出看似合理的话”，而是：

> **模型能不能在长时程、动态、部分可观察的环境里，持续做对动作并稳定推进目标。**

现有很多 agent benchmark 更像是几十步以内的短流程任务，比如网页、办公或代码问题；这类任务已经开始被更强的 prompting 和 test-time reasoning 快速抬高分数。  
但真实代理能力需要的是另一组能力组合：

- 长期规划
- 系统性探索
- 空间推理与导航
- 资源管理
- 发现环境规则/因果结构
- 在错误后继续恢复，而不是一次答错就结束

BALROG 的切入点很直接：不用抽象问答去猜 agent 能力，而是把模型放进 **6 个程序生成的游戏环境**里真正交互：

- BabyAI
- Crafter
- TextWorld
- Baba Is AI
- MiniHack
- NetHack Learning Environment

这些环境的难度跨度很大：从**非专家人类几秒可解**到**人类需要多年熟练**。这使它不像单一 benchmark 那样只测一种“会不会”，而是在难度轴上连续拉开能力差距。

### 输入/输出接口与边界条件

BALROG 的评测接口非常统一：

- **输入**：游戏规则、可执行动作列表、历史 observation-action 轨迹、当前观测  
- **输出**：模型生成的一条自然语言动作
- **模态**：
  - 纯语言：文本描述环境状态
  - 视觉-语言：文本 + 当前图像
- **鲁棒性处理**：若模型输出非法动作，系统会反馈“动作无效”，再执行 fallback 动作并记录错误
- **默认协议**：零样本、单智能体、离散动作、固定历史长度 16
- **防泄漏条件**：环境是程序生成的，同一实例几乎不会重复，难以靠记忆模板过关

所以这篇论文的“为什么现在要做”也很清楚：  
**短程 benchmark 已不足以区分“会回答”和“会行动”，而长时程 agentic reasoning 正是下一阶段能力瓶颈。**

## Part II：方法与洞察

BALROG 本质上不是一个新模型，而是一个**评测操作系统**。它的关键设计有三层。

### 1) 把异质长时程环境统一到同一接口

作者把 6 个游戏封装到同一评测框架里，统一：

- observation 接口
- action 接口
- agent 与 model 的调用方式
- evaluator 与多 seed 统计流程

并且**显式解耦**：

- 底层模型
- inference-time agentic strategy

这意味着你可以只换 `client.py` 测新模型，也可以只改 `agent.py` 测新策略。  
对于 benchmark 来说，这个解耦很重要，因为它避免把“模型能力”和“prompt 工程/agent 框架能力”混在一起。

### 2) 用细粒度进度，而不是只看成败

BALROG 没有把所有环境都压成“成功/失败”二值结果，而是统一成 **0–100 分**：

- BabyAI / MiniHack / Baba Is AI：完成则 100，否则 0
- TextWorld / Crafter / NetHack：用部分 achievement/progression 计分

这一步非常关键，因为长时程 hardest tasks 如果只看通关，结果常常会是“全 0”，那就无法区分模型到底是：

- 完全不会
- 有一些局部能力
- 只是卡在后段规划
- 还是视觉输入出了问题

### 3) 为 NetHack 设计数据驱动进度指标

NetHack 原始游戏分数并不等价于“离通关有多近”。  
作者因此用人类 NetHack 对局数据，统计：

- 到达某个 dungeon level 后的人类最终胜率
- 到达某个 experience level 后的人类最终胜率

再把这两条曲线转成 progression signal。最终一个 agent 的 NetHack 进度取两者中的较高值。

这相当于把原本失真的游戏内 reward，替换成**更接近真实长期推进程度**的代理测量。

### 核心直觉

**BALROG 真正改变的不是模型，而是“测量瓶颈”。**

- **What changed**：从“短流程、单任务、终局成败”改成“长时程、多环境、过程可计分”
- **Which bottleneck changed**：把原来被终局 0/1 掩盖的错误，变成可观察的 trajectory-level failure pattern
- **What capability changed**：现在可以区分模型到底卡在探索、空间推理、长期规划、规则发现，还是视觉输入整合

更因果地说：

> **统一长时程环境 + 程序生成实例 + 细粒度进度指标**  
> 让模型不能靠记忆模板“刷过去”，也不会因为 hardest task 全部记 0 而失去诊断能力，  
> 因而能更真实地暴露 sequential decision making 的能力边界。

这就是为什么 BALROG 能测出一个非常重要的结论：  
**很多模型并不是“完全没知识”，而是“有知识但无法在交互轨迹中把知识转成稳定行动”。**

### 战略取舍表

| 设计选择 | 解决的瓶颈 | 收益 | 代价/风险 |
| --- | --- | --- | --- |
| 多游戏聚合评测 | 单任务偏置 | 能看到跨能力 profile | 排名会受任务组合影响 |
| 程序生成环境 | 测试泄漏/记忆过拟合 | 更难靠背题拿高分 | 与旧 benchmark 结果不完全可比 |
| 0–100 进度分 | hardest task 全 0 不可诊断 | 能看部分成功与局部进步 | 指标定义本身会影响结论 |
| NetHack 人类数据驱动进度 | 原始 score 失真 | 更贴近真实通关推进 | 依赖外部人类数据与映射假设 |
| 文本 + 可选视觉双模态 | 可直接对比 LLM/VLM | 能显式测视觉决策价值 | 图像历史受成本限制，模态设置不完全对称 |
| 默认零样本协议 | 基线公平、干净 | 易于横向比较 | 低估带记忆/检索/规划 agent 的上限 |

## Part III：证据与局限

这篇论文最强的证据不是“谁排第一”，而是：

> **即便最强模型，距离真正长时程 agent 仍然非常远。**

### 关键实验信号

- **比较信号**：总体语言模式下，Claude 3.5 Sonnet 为 32.64%，GPT-4o 为 32.34%。这说明 benchmark 不是“所有模型都 0 分”的死板测试，但也说明当前最好模型平均只做到约三成推进，距离可用 agent 很远。
- **难度伸缩信号**：较简单环境上模型能部分成功，但到了 MiniHack 与 NetHack 几乎整体熄火。尤其 NetHack 上，最好模型 o1-preview 也只有 **1.57% 平均进度**；MiniHack 的 Quest/Boxoban 任务则没有模型解出。
- **模态对照信号**：视觉输入常常不是帮助，而是干扰。GPT-4o 从 32.34% 降到 22.56%，Llama 3.2 90B 从 27.29% 降到 20.99%。这说明当前很多 VLM 擅长“看图说话”，不等于擅长“看图做决策”。
- **分析信号**：作者在轨迹里反复观察到系统性探索失败、复杂空间操作失败、长期计划断裂，以及典型的 **knowing-doing gap**。例如模型能在问答里说出“腐烂食物危险”“一层上楼会直接结束游戏”，但实际轨迹里仍会去吃、去上楼。

还有一个很有价值的现象：  
不同环境的领先模型并不完全一致，比如 Baba Is AI 语言模式下 Llama 系列表现很好。这说明 BALROG 不是单纯在测“通用聊天能力”，而是在测更细粒度的决策 profile。

### 复现与依赖现实

虽然 BALROG 本身是开源、轻量模拟器、支持并行的，但其结果仍有明显现实依赖：

- 许多强基线依赖闭源 API
- 超长上下文 few-shot 很贵，论文中提到单个 NetHack 演示可达 **70 万+ token**
- 视觉模式因成本限制，只给当前单帧图像，历史主要保留文本
- o1 只在 NetHack 上评估，原因是预算限制
- Gemini 在 TextWorld 中多次被 API 错误标记为 unsafe，导致该环境结果可比性受影响

所以“代码开源”并不自动等于“结果无摩擦复现”。

**局限性**

- Fails when: 需要真实世界连续控制、视频级视觉记忆、多智能体协作或复杂工具调用的场景；BALROG 主要覆盖的是单智能体、离散动作、游戏化长时程任务。
- Assumes: 模型可以用自然语言稳定输出合法动作；评测依赖语言包装器、固定历史长度、部分闭源 API，以及 NetHack 的人类数据驱动进度映射。
- Not designed for: 训练期 RL 样本效率比较、真实机器人部署评测、纯视觉端到端策略学习、完整视频输入的多模态长期记忆评测。

**可复用组件**

- 统一的 `agent / client / env_wrapper / evaluator` 结构
- 文本/视觉双模态的统一 observation wrapper
- 可迁移到其他 long-horizon env 的进度评分思路
- 支持 few-shot、CoT、推理时 agentic strategy 的实验入口
- 可公开提交的 leaderboard 机制

## Local PDF reference

![[paperPDFs/Agentic_Reasoning_Benchmarks_Applications_of_Agentic_Reasoning_Embodied_Agents/NeurIPS_2024/2024_BALROG_Benchmarking_Agentic_LLM_and_VLM_Reasoning_On_Games.pdf]]