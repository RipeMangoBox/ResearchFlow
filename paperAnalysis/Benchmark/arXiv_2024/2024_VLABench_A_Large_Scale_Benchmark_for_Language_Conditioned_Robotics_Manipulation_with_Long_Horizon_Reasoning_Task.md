---
title: "VLABench: A Large-Scale Benchmark for Language-Conditioned Robotics Manipulation with Long-Horizon Reasoning Tasks"
venue: arXiv
year: 2024
tags:
  - Survey_Benchmark
  - task/language-conditioned-manipulation
  - task/embodied-evaluation
  - progress-score
  - dsl-graph-matching
  - automated-data-collection
  - dataset/VLABench
  - opensource/partial
core_operator: 用100类长程语言操控任务、分级进度评分和DSL图匹配，把机器人操作中的技能、语义、常识、物理和推理统一到一个可诊断评测框架中
primary_logic: |
  评测目标（语言条件机器人操控的泛化与推理） → 构建100类任务、2164个对象及自动采集轨迹/自然语言指令 → 用 Progress Score 评估交互执行、用 DSL-DAG 匹配评估 VLM 规划 → 揭示现有 VLA/VLM/workflow 在跨类别泛化、隐式语义理解与长程推理上的能力边界
claims:
  - "在作者比较的机器人操作基准中，VLABench 是唯一同时覆盖语义丰富语言、逻辑推理、常识知识、强域随机化、多相机、点云、cross-embodiment 与自动轨迹采集的基准 [evidence: analysis]"
  - "在 VLABench 上，微调后的 Octo、OpenVLA、RDT-1B 在 primitive seen-object 设置下平均 PS 分别仅为 1.34、11.74、15.37，而 composite 任务最佳平均 PS 也只有 3.34，说明当前 VLA 远未具备强长程泛化 [evidence: comparison]"
  - "VLM 在该基准上同样未表现出稳健 embodied 规划能力；Qwen2-VL-7B 可在部分维度与 GPT-4-turbo 竞争，但所有模型在复杂长程任务上都明显掉分 [evidence: comparison]"
related_work_position:
  extends: "RLBench (James et al. 2020)"
  competes_with: "LIBERO (Liu et al. 2024); RoboCasa (Nasiriany et al. 2024)"
  complementary_to: "N/A"
evidence_strength: moderate
pdf_ref: paperPDFs/Benchmark/arXiv_2024/2024_VLABench_A_Large_Scale_Benchmark_for_Language_Conditioned_Robotics_Manipulation_with_Long_Horizon_Reasoning_Task.pdf
category: Survey_Benchmark
---

# VLABench: A Large-Scale Benchmark for Language-Conditioned Robotics Manipulation with Long-Horizon Reasoning Tasks

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2412.18194), [Project](https://vlabench.github.io/)
> - **Summary**: 这篇工作提出了一个面向基础模型时代机器人操作的统一评测基准，用自然语言、常识迁移、长程推理和多技能组合任务来检验 VLA/VLM 是否真的具备“会理解、会规划、会执行”的能力。
> - **Key Performance**: VLABench composite 任务平均轨迹长度达 **502.5** steps；最佳已评测 VLA 在 composite 上平均 **PS 仅 3.34**。

> [!info] **Agent Summary**
> - **task_path**: 自然语言指令 + 多视角仿真场景/点云 -> 机器人动作策略或 DSL 技能序列 -> 任务完成度/进度评测
> - **bottleneck**: 现有 manipulation benchmark 主要测模板化技能复现，缺少对隐式语义、常识迁移、跨类别泛化和长程推理的系统诊断
> - **mechanism_delta**: 用 60 个 primitive + 40 个 composite 任务配合 Progress Score 与 DSL-DAG 评分，把“是否成功”扩展成“错在感知、语义、规划还是执行”的分解式评测
> - **evidence_signal**: OpenVLA/Octo/RDT-1B、VoxPoser/CoPA、GPT-4o/Qwen2-VL 等多类强基线在 composite 与 reasoning 维度均显著受限
> - **reusable_ops**: [progress-score, dsl-dag-matching]
> - **failure_modes**: [VLA 缺少时序记忆与动作精度, workflow 易受感知误差和模块接口误差放大]
> - **open_questions**: [benchmark 分数能否稳定预测真实机器人表现, VLA 预训练规模何时会带来类似 LLM 的泛化跃迁]

## Part I：问题与挑战

这篇论文真正想解决的，不是“再造一个更多任务的仿真环境”，而是**评测瓶颈**：  
现有语言条件机器人操作基准，大多仍然在测“是否能把一个模板化指令映射成一个技能轨迹”，但基础模型时代真正被宣称的能力是：

- 从自然语言中抽取隐式目标；
- 把预训练阶段学到的常识/世界知识迁移到操作里；
- 在复杂视觉场景中做空间判断；
- 处理多步、长时程、带前置条件的任务；
- 在新对象类别、新语义表达、相似但未见任务上泛化。

### 现有评测为什么不够

论文指出，RLBench、CALVIN、LIBERO 等已有基准各有价值，但对 foundation-model-based manipulation 来说有几个共性缺口：

1. **语言太模板化**  
   很多任务直接告诉机器人“拿起 X 放到 Y”，而不是像真实用户那样说“我好渴，想来点冰的”。这会把“语义理解”退化成“关键词匹配”。

2. **泛化定义过弱**  
   许多 benchmark 的 unseen 更像“同类实例变化”，而不是**跨类别**变化。VLABench 刻意把 unseen object 定义成完全不同类别，如 seen 是 apple/banana，unseen 变成 kiwi/mango/strawberry。

3. **成功率指标过粗**  
   0/1 success 适合测单步 skill learning，但对长程任务不友好：模型已经完成了一半规划，却仍被记作 0，诊断价值很弱。

4. **长程 reasoning 缺位**  
   以前很多 benchmark 测的是技能链长度，不一定测“逻辑链深度”。而 VLABench 想测的是：你是否知道做 latte 需要咖啡+牛奶、打扑克牌先翻牌再选最大牌型、化学实验要按依赖步骤执行。

### 输入/输出接口与边界条件

- **输入**：自然语言指令 + 多视角 RGB-D/分割/点云观测
- **输出**：
  - 对 VLA：连续动作策略
  - 对 workflow：模块化规划与执行
  - 对 VLM：DSL 技能序列
- **环境边界**：
  - MuJoCo + dm_control 仿真
  - 默认评测 embodiment 为 7-DoF Franka Panda
  - 支持 cross-embodiment，但主结果主要在 Panda 上
- **任务规模**：
  - 100 类任务：60 primitive + 40 composite
  - 10 类技能族
  - 163 类资产、2164 个对象

一句话说，**VLABench 的目标是把“机器人会做动作”升级为“机器人是否真的理解任务并能稳健推理”**。

## Part II：方法与洞察

### 评测设计：不是堆任务，而是重写能力坐标系

VLABench 把 primitive task 分成 5 个能力维度，再用 composite task 把它们串起来：

- **Mesh & Texture Understanding**：识别复杂外观、纹理、角色/IP
- **Spatial Understanding**：nth left/right、inside/outside、行列关系、相对位置
- **Common Sense & World Knowledge**：把预训练知识迁移到操作选择
- **Semantic Understanding**：从自然对话中提取隐式需求
- **Physical Law**：理解杠杆、交互、动态物理约束
- **Composite / Long-horizon Reasoning**：多技能组合、前置条件满足、长程规划

其中 composite task 的平均 episode 长度超过 **500 timesteps**，显著高于大多数已有操作基准。

### 评测协议：一套 benchmark，三类对象

论文不是只评一个模型族，而是统一评三类系统：

1. **VLA 模型**
   - seen objects
   - unseen objects
   - 语义泛化
   - 相似未见任务泛化
   - composite 长程任务

2. **foundation-model workflow**
   - 如 VoxPoser、CoPA
   - 主打 zero-shot，但容易被感知模块和规划模块接口限制

3. **VLM**
   - 非交互评测：输出 DSL 技能序列，再和参考 DAG 做图匹配
   - 交互评测：把 DSL 解析为可执行动作，与仿真环境交互

### 评分机制：从“成/败”变成“进度+结构”

这篇 paper 的一个关键贡献，是把评测从单一 success rate 拆成两层：

- **Progress Score (PS)**：  
  用“目标选对了多少” + “子步骤推进了多少”来评分。  
  这能避免长程任务中“一步错全盘 0 分”的粗糙评估。

- **VLM 的 DSL 图匹配评分**：  
  从四个维度评分：
  - Skill Recall
  - Parameter Recall
  - Skill&Parameter Recall
  - Precise Matching Rate（含依赖关系）

这意味着 benchmark 不只问“最后有没有成功”，还问：

- 技能选错了？
- 参数指错了对象？
- 顺序依赖错了？
- 还是其实高层规划对了，只是执行挂了？

### 数据构建：为了让 benchmark 可训练、可比较

VLABench 不只是评测集，也提供自动化数据构建框架：

- **Domain Randomization**
  - 位置/朝向
  - 尺度
  - 布局
  - 纹理/光照
  - distractors

- **自动轨迹生成**
  - 基于 skill library + task-specific motion planner
  - RRT 生成路径
  - SLERP 做姿态插值
  - Bezier 平滑
  - reject sampling + early termination 提高效率

- **语言增强**
  - 用 GPT-4 生成更自然、更隐式、更多样的任务描述

这让 benchmark 不只是“考题集”，也变成了一个**标准化训练数据生成器**。

### 核心直觉

过去 benchmark 的测量瓶颈是：

> 模板指令 + 同类实例泛化 + 二元成功率  
> 只能测“技能是否记住了”，很难测“是否真的理解并规划了”。

VLABench 做的关键改变是：

> 自然语言/隐式意图 + 跨类别 unseen + 长程 composite + 分解式评分

这个改变带来的因果链是：

- **测量空间变了**：从单点 success 变成多维能力诊断
- **约束更接近 foundation model 声称的能力**：语言、常识、规划、物理、视觉一起受压
- **因此暴露出以前 benchmark 隐藏的问题**：  
  很多模型不是“不会抓”，而是“不会理解任务”“不会保持长程状态”“不会把知识迁移到动作里”

### 战略取舍

| 设计选择 | 改变了什么瓶颈 | 收益 | 代价/妥协 |
|---|---|---|---|
| 自然对话式、隐式目标指令 | 避免模板匹配假象 | 更接近真实用户交互 | 语言评测噪声更大，复现实验更复杂 |
| unseen object 按“新类别”定义 | 提高 OOD 难度 | 真正测跨类别视觉+知识泛化 | 分数更低，更难和旧 benchmark 直接对齐 |
| composite 长程任务 | 从技能复现转向任务规划 | 能测记忆、前置条件、逻辑链 | 执行误差会累积，诊断更难 |
| Progress Score | 避免 success-only 过粗 | 可区分“做对一半”和“完全不会” | 指标设计需要任务级子步骤标注 |
| DSL + DAG 匹配 | 把 VLM 规划错误拆解 | 区分 skill / parameter / dependency 错误 | 依赖 skill library，非完全端到端 |
| 自动数据生成 | 降低人类遥操作成本 | 可扩展、可标准化 | 依赖较强先验与模拟器标注体系 |

## Part III：证据与局限

### 关键证据信号

**信号 1｜覆盖度分析：它确实补了现有 benchmark 的空位**  
Table 1 显示，在作者比较的基准里，VLABench 是唯一同时覆盖语义丰富语言、逻辑推理、常识、强随机化、多相机、点云、cross-embodiment 与自动轨迹采集的基准。  
这说明它不是简单扩大任务数，而是**扩大评测维度**。

**信号 2｜VLA 比较：当前预训练 VLA 在这里没有出现“LLM 式泛化”**  
在 primitive seen-object 上，平均 PS：

- Octo：**1.34**
- OpenVLA：**11.74**
- RDT-1B：**15.37**

在 composite 上，最佳平均 PS 也只有：

- RDT-1B：**3.34**
- OpenVLA：**2.66**
- Octo：**0.00**

结论很清楚：**现有 VLA 还没有把“预训练世界知识/语言能力”稳定转成长程操作能力**。

**信号 3｜workflow 比较：zero-shot 泛化的上限被模块误差锁死**  
VoxPoser/CoPA 在简单维度上能拿到一定分数，但到了 physical law 和 complex task 明显掉队。  
作者的失败分析把原因拆成三类：

- 感知错：视觉模块漏检/误检
- 规划错：不知道该旋转、该先做哪一步
- 模块接口错：LLM 输出的约束不能稳健转成执行轨迹

这说明**模块化 workflow 的最大问题不是“有没有大模型”，而是链路中任何一环都会把误差放大**。

**信号 4｜VLM 比较：多模态大模型并不等于 embodied planner**  
论文发现：

- GPT-4o 在 reasoning 维度最强
- Qwen2-VL-7B 在某些维度能与 GPT-4-turbo 竞争
- 但所有模型在 complex/long-horizon 上都明显退化

这说明 VLM 的“看图+答题”能力，并不能直接等价成“面向执行的任务分解能力”。

### 1-2 个最值得记住的指标

- **Composite 平均 horizon = 502.5 steps**：这不是短链技能 benchmark。
- **Best VLA composite average PS = 3.34**：当前强基线离“通用机器人基础模型”还很远。

### 局限性

- **Fails when**: 需要真实世界中的传感器噪声、控制延迟、接触不确定性或未建模动力学时，VLABench 分数未必能可靠外推到真实机器人；同时，在抓取精度要求很高的任务里，低层控制失败会掩盖高层语义/推理能力。
- **Assumes**: 基于 MuJoCo 仿真与默认 Panda 机械臂；依赖资产级 grasp/place 标注、分割标注、DSL 技能库和任务子步骤定义；语言增强依赖 GPT-4，资产扩充/预标注用到 Tripo.ai、Runway.ai、GraspNet、SAM；VLA 微调实验使用 4×A800 80GB。
- **Not designed for**: 真实世界安全评测、人机协作安全规范、完全无技能先验的开放式动作发现；对 VLM 的非交互 DSL 评测也不能完全替代闭环执行评测。

### 可复用组件

这篇工作最有复用价值的，不只是 benchmark 名字，而是几套“评测算子”：

- **Progress Score**：适合长程 embodied task 的渐进式评分
- **DSL + DAG 匹配**：适合把 VLM 的规划错误结构化拆解
- **自动轨迹生成框架**：降低仿真数据采集成本
- **强 domain randomization 配方**：更接近 foundation model 需要的泛化压力
- **任务分类框架**：把视觉、语言、常识、物理、推理统一到同一评测空间

整体判断：  
**VLABench 的价值不在于“又一个机器人数据集”，而在于它把机器人 foundation model 的宣传点变成了可被系统检验的失败模式。** 这使它更像一个“诊断台”，而不是单纯排行榜。

## Local PDF reference

![[paperPDFs/Benchmark/arXiv_2024/2024_VLABench_A_Large_Scale_Benchmark_for_Language_Conditioned_Robotics_Manipulation_with_Long_Horizon_Reasoning_Task.pdf]]