---
title: "EmbodiedAgent: A Scalable Hierarchical Approach to Overcome Practical Challenge in Multi-Robot Control"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/multi-robot-planning
  - task/multi-robot-control
  - next-action-prediction
  - tool-calling
  - structured-memory
  - "dataset/MultiPlan+"
  - opensource/full
core_operator: 将异构多机器人规划改写为带显式不可行错误信号的逐步工具调用，并结合结构化记忆与环境校验来抑制幻觉规划。
primary_logic: |
  自然语言任务与结构化环境描述、规划历史 → LLM 逐步预测“下一动作/终止/不可执行错误”，并通过工具调用与环境约束校验更新记忆 → 输出可执行的异构多机器人动作序列或显式失败类型
claims:
  - "EmbodiedAgent 在 32 个未见任务上的 RPAS 达到 71.85%，优于所有比较的专有与开源基线 [evidence: comparison]"
  - "采用下一动作预测范式并在 MultiPlan+ 上微调后，EmbodiedAgent 的 ASRtop-k 达到 74.01%，超过 Deepseek-R1、LLaMA-3.3-70B 与 MAP-Neo-7B-Multiplan 等基线 [evidence: comparison]"
  - "系统在真实办公室服务场景中完成了机械臂与机器狗协作的递送纸巾、抬杯和擦拭咖啡任务 [evidence: case-study]"
related_work_position:
  extends: "MultiPlan (Wan et al. 2025)"
  competes_with: "Smart-LLM (Kannan et al. 2024); COHERENT (Liu et al. 2024)"
  complementary_to: "ACT (Zhao et al. 2023); AdaPlanner (Sun et al. 2023)"
evidence_strength: moderate
pdf_ref: paperPDFs/Other_Multimodal_Large_Language_Model_MLLM_based_related_Embodied_Learning_2025/arXiv_2025/2025_EmbodiedAgent_A_Scalable_Hierarchical_Approach_to_Overcome_Practical_Challenge_in_Multi_Robot_Control.pdf
category: Embodied_AI
---

# EmbodiedAgent: A Scalable Hierarchical Approach to Overcome Practical Challenge in Multi-Robot Control

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2504.10030), [Code](https://github.com/HaronW/EmbodiedAgent)
> - **Summary**: 该工作把异构多机器人规划从“直接生成整段计划”改成“逐步预测下一动作/终止/不可执行错误”，再配合结构化记忆与环境校验，从而减少不切实际任务上的幻觉规划。
> - **Key Performance**: RPAS 71.85%；ASRtop-k 74.01%（32 个未见任务）

> [!info] **Agent Summary**
> - **task_path**: 自然语言任务 + 结构化环境描述 + 规划历史 -> 下一机器人动作/终止/不可执行错误信号
> - **bottleneck**: 多机器人 LLM 规划在不可执行任务上容易“硬编”计划，且长程协作会因早期错误级联失效
> - **mechanism_delta**: 将自由文本全序列规划改为带工具调用的下一动作预测，并显式加入 LoA/LoO/LoS/LoL 不可行错误与结构化记忆
> - **evidence_signal**: 在 32 个未见任务上取得 71.85% RPAS，超过全部对比模型
> - **reusable_ops**: [下一动作预测循环, 结构化记忆, 显式不可行错误工具]
> - **failure_modes**: [环境注册缺失或位置歧义时仍会出错, 超出已注册技能/负载/对象集合的任务只能报错终止]
> - **open_questions**: [能否扩展到更大规模真实机器人团队, 静态结构化环境描述之外如何结合在线感知反馈]

## Part I：问题与挑战

这篇工作的真正问题不是“让 LLM 会写机器人计划”，而是**让计划在异构多机器人系统中真正可执行**。

### 真实瓶颈
现有 LLM 多机器人规划器常见两个痛点：

1. **不可执行任务上的幻觉**  
   当用户提出反事实或物理上不可满足的要求时，模型往往继续编造动作序列，而不是明确指出“做不到”。论文把这类问题具体化为 impractical cases，例如缺物体、缺技能、超负载、缺能力。

2. **长程规划中的错误级联**  
   多机器人协作需要任务分解、分工、顺序协调；如果前几步动作就错了，后面整条计划基本都会失效。

### 输入/输出接口
该系统的高层输入是三部分：

- **Mission**：场景 + 任务目标
- **Environment**：工作空间、机器人、物体、用户
- **Planning Memory**：历史动作记录

高层输出不是整段文本计划，而是每一步只输出一种结果：

- 一个原子机器人技能 + 参数
- `endPlanning`
- 一个不可执行错误信号：`LoA / LoO / LoS / LoL`

### 边界条件
这套方法成立有几个前提：

- 环境需以**结构化描述**给出，而不是直接从原始视频端到端理解
- 机器人技能必须先被**注册为工具**
- 低层控制由专门策略或 SDK 执行，高层不直接学连续控制

所以它解决的是**高层可执行规划**，不是全栈端到端机器人学习。

## Part II：方法与洞察

### 方法主线
EmbodiedAgent 是一个分层系统：

1. **高层 Agent Planner**
   - 用 LLM 做推理
   - 把机器人技能封装成可调用工具
   - 用结构化记忆保存历史动作与环境状态
   - 每次只预测“下一步”

2. **环境校验与执行调度**
   - 检查动作是否在合法技能集合内
   - 检查参数、环境约束、机器人配置是否匹配
   - 成功则更新状态；失败则记录错误并终止

3. **低层执行模块**
   - 通过 SDK 或单任务策略执行具体动作
   - 真实实验里，机械臂使用 ACT，机器狗使用内置 locomotion SDK

4. **训练数据：MultiPlan+**
   - 超过 18,000 条数据，100 个场景
   - 从原始 MultiPlan 扩展到**下一动作预测**
   - 额外加入 impractical cases，让模型学会“报错”而不是“胡编”

5. **评测：RPAS**
   - 结合 top-k 成功率与 LLM 辅助专家评分
   - 再用 MRED 细分错误来源，如位置错误、技能错误、结束错误等

### 核心直觉

过去很多方法默认让模型“一次性生成整条计划”，这会把三个难题绑在一起：长程推理、约束满足、异常处理。  
EmbodiedAgent 的关键变化是：

**全序列自由生成 → 单步工具选择 + 显式错误类型 + 结构化记忆**

这带来了三个直接的因果变化：

- **输出分布被压缩**：从开放文本序列变成有限工具集合，减少胡乱生成空间
- **不可行任务被显式建模**：训练数据中不只是“正确动作”，还有“应该报错”的负样本
- **长程依赖被局部化**：每一步只做局部最优决策，并基于最新状态继续规划

因此能力上的变化是：

- 更少 hallucinated plan
- 更稳定的长程任务分解
- 更容易把高层规划迁移到不同机器人组合上

### 为什么这套设计有效
它不是单纯“加个 memory”或“换个 prompt”，而是把规划问题改造成更像**受约束的决策过程**：

- 工具调用保证动作格式标准化
- 结构化记忆减少上下文丢失
- 显式错误信号允许模型合理拒绝
- 分层执行把高层语义规划与低层控制复杂度解耦

### 策略权衡

| 设计选择 | 改变了什么 | 收益 | 代价/风险 |
|---|---|---|---|
| 下一动作预测替代全序列生成 | 降低单次决策空间 | 更稳、更易校验 | 推理轮数增加 |
| 显式 LoA/LoO/LoS/LoL | 给模型“报错出口” | 减少不可行任务幻觉 | 错误类型受预定义 taxonomy 限制 |
| 结构化记忆 | 把历史状态持续带入 | 长程规划一致性更好 | 依赖状态更新正确 |
| 分层高低控制 | 规划与执行解耦 | 易接入异构机器人 | 低层技能仍需单独训练/集成 |

## Part III：证据与局限

### 关键证据

- **比较信号**：在 32 个未见任务上，EmbodiedAgent 达到 **71.85% RPAS**、**74.01% ASRtop-k**，超过所有专有和开源基线。最重要的含义不是“分数更高”，而是**一个微调后的 8B 规划器在结构化多机器人规划上优于多种更大模型**，说明任务形式重构与领域数据比单纯参数规模更关键。
- **错误诊断信号**：该方法的 **Ending Error 为 0.00**，Planning Error 也较低，说明“终止信号 + 不可行错误信号”的显式设计确实减少了错误继续规划。
- **真实案例信号**：办公室服务实验中，系统成功协调机械臂与机器狗完成递送纸巾、抬杯、擦拭咖啡，证明它不仅停留在离线 benchmark，而是能接入真实异构执行器。

### 局限性

- **Fails when:** 环境描述缺失、位置定义含糊、对象未注册，或任务要求超出已注册机器人能力/技能/负载范围时，系统主要是报错或中止，而不是主动补感知、重规划或探索式恢复。
- **Assumes:** 依赖结构化 JSON 环境表示、预注册工具库、分层执行架构，以及 MultiPlan+ 上的监督微调；真实实验还依赖低层控制器训练，例如机械臂用 ACT 且每个任务收集约 50 条遥操作演示。训练本身使用 8×A100，RPAS 中的专家评分还引入了 LLM grader 依赖。
- **Not designed for:** 原始图像/视频到动作的端到端控制、无中心规划器的分布式协商、多机器人海量真实部署验证。并且评测集只有 32 个未见任务，其中 impractical 样本仅 2 个，因此对“稀有不可执行场景”的证据仍然有限。

### 可复用组件
这篇论文最值得迁移的不是某个单独模型，而是几个系统操作件：

- **下一动作预测式规划循环**
- **LoA/LoO/LoS/LoL 不可行错误模式**
- **结构化记忆 + 工具调用接口**
- **RPAS/MRED 这套规划评测与错误诊断框架**

![[paperPDFs/Other_Multimodal_Large_Language_Model_MLLM_based_related_Embodied_Learning_2025/arXiv_2025/2025_EmbodiedAgent_A_Scalable_Hierarchical_Approach_to_Overcome_Practical_Challenge_in_Multi_Robot_Control.pdf]]