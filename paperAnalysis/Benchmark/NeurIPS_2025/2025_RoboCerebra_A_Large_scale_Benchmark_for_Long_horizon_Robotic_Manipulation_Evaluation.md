---
title: "RoboCerebra: A Large-scale Benchmark for Long-horizon Robotic Manipulation Evaluation"
venue: NeurIPS
year: 2025
tags:
  - Survey_Benchmark
  - task/robotic-manipulation-evaluation
  - hierarchical-planning
  - closed-loop-verification
  - time-segmentation
  - dataset/RoboCerebra
  - opensource/partial
core_operator: 以LLM自顶向下生成长程操作任务、人工采集带时间边界轨迹，并在固定低层控制器下用多维指标评测System 2规划、反思与记忆能力
primary_logic: |
  长程机器人操作评测目标 → LLM生成高层任务并分解子步骤、规则化实例化场景、双环验证与人工时间标注 → 固定System 1控制器并替换System 2规划器 → 用成功率、计划匹配、效率与动作完成率揭示规划/反思/记忆边界
claims:
  - "RoboCerebra的平均轨迹长度达到2972.4个仿真步，约为既有长程机器人操作基准的6×，并补入动态场景、细粒度步骤分解与时间标注等要素 [evidence: comparison]"
  - "在固定System 1控制器下，GPT-4o作为System 2规划器取得16.04%的最高平均成功率和68.33%的计划匹配率，但仍低于GT-plan的25.16%，说明高层规划仍是主要瓶颈 [evidence: analysis]"
  - "将纯反应式OpenVLA*替换为带高层规划的层次框架后，六类任务平均成功率从4.57%提升到16.55%，且收益主要来自动态与记忆相关场景 [evidence: comparison]"
related_work_position:
  extends: "LIBERO (Liu et al. 2024)"
  competes_with: "RoboCasa (Nasiriany et al. 2024); VLABench (Zhang et al. 2024)"
  complementary_to: "OpenVLA (Kim et al. 2024); Diffusion Policy (Chi et al. 2023)"
evidence_strength: moderate
pdf_ref: paperPDFs/Benchmark/NeurIPS_2025/2025_RoboCerebra_A_Large_scale_Benchmark_for_Long_horizon_Robotic_Manipulation_Evaluation.pdf
category: Survey_Benchmark
---

# RoboCerebra: A Large-scale Benchmark for Long-horizon Robotic Manipulation Evaluation

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2506.06677), [Project](https://robocerebra.github.io)
> - **Summary**: 该工作提出一个面向长程机器人操作的 System 2 基准，把评测重点从“短程反应式控制”转到“规划、反思、记忆”等高层推理能力。
> - **Key Performance**: 平均轨迹长度 **2972.4** 步（约为既有基准 **6×**）；Hierarchical Framework 平均成功率 **16.55%**，高于 OpenVLA* 的 **4.57%**。

> [!info] **Agent Summary**
> - **task_path**: 长程家居操作指令 + 模拟视觉观测/历史上下文 -> 规划、反思、记忆能力评测结果
> - **bottleneck**: 现有机器人基准任务太短、场景太静态，无法把高层推理瓶颈与低层控制能力分离评估
> - **mechanism_delta**: 用LLM级联生成长子任务序列并配合人类时间标注，再固定System 1、替换System 2规划器做分维诊断
> - **evidence_signal**: 600次rollout的多模型比较显示，加入System 2规划后平均成功率显著提升，而GPT-4o与GT-plan仍有明显差距
> - **reusable_ops**: [LLM级联任务生成, 锚点对齐的分层评测协议]
> - **failure_modes**: [动态与记忆复合场景下总体成功率仍低, 反思/观察分数提升不必然转化为任务成功]
> - **open_questions**: [如何迁移到真实机器人评测, 如何增强System 1与System 2的双向可解释交互]

## Part I：问题与挑战

这篇论文真正要解决的，不是“再做一个机器人数据集”，而是**现有评测几乎测不到 VLM 在机器人里的 System 2 能力**。

### 1) 真问题是什么？
过去很多机器人操作基准，虽然已经从单步动作走向多步任务，但整体仍偏短：
- 通常只有 **2–5 个子任务**
- 动作长度往往 **少于 500 步**
- 缺少 **动态扰动、部分可观测、记忆依赖**
- 指标多停留在 **任务是否成功**

这导致一个核心混淆：  
如果模型失败，我们很难知道到底是：
- 低层控制不行，
- 视觉 grounding 不行，
- 还是高层规划、反思、记忆不行。

论文的判断是：**现在的基准主要还在测 System 1（快反应），没真正测到 System 2（慢推理）**。

### 2) 为什么现在要解决？
因为 VLM/VLA 已经具备一定的语义理解与规划潜力，但现有 benchmark 仍把它们当作“更强的反应式策略”使用。  
如果评测环境本身不要求：
- 长时依赖，
- 子目标分解，
- 任务进度判断，
- 动态重规划，

那么高层推理能力即使存在，也无法被显式测出来，更无法推动方法进步。

### 3) 输入/输出接口与边界条件
RoboCerebra 的评测接口本质上是：

- **输入**：长程自然语言任务、模拟环境视觉观测、历史执行上下文
- **中间过程**：高层 planner 生成/更新子目标，低层 controller 执行动作
- **输出**：任务成功率、计划匹配度、计划效率、动作完成判断等评测结果

它的边界也很明确：
- 场景是 **家庭操作仿真环境**
- 重点是 **高层 reasoning benchmark**，不是 sim-to-real
- 低层控制器会被尽量固定，用来**隔离评估 System 2**
- 任务难点来自 **长程、多子任务、记忆与扰动**，而不是极致的低层灵巧操作

**Q1 / What & Why**：  
真正瓶颈不是“机器人不会动”，而是**没有一个足够长、足够复杂、又能拆开诊断的评测环境去测高层推理**。

## Part II：方法与洞察

RoboCerebra 由三部分组成：
1. **长程任务与数据集**
2. **面向 System 2 的评测协议**
3. **一个分层 HPE baseline 用来跑 benchmark**

### 数据集与任务设计

论文定义了 6 类典型长程子任务场景：
- **Ideal**：静态、完全可见
- **Memory Exploration**：需要主动搜索与记忆
- **Memory Execution**：执行时依赖先前记忆
- **Random Disturbance**：执行中有外部扰动
- **Observation Mismatching**：观察与计划不一致
- **Mix**：记忆与动态因素叠加

数据构建流程是一个自顶向下 pipeline：
1. 从 Libero 仿真库采样物体与环境
2. 用 GPT 生成高层任务
3. 再分解成细粒度子步骤
4. 通过规则把步骤编译成可执行场景代码
5. 用**符号验证 + VLM视觉验证**做双环检查
6. 让人类在仿真中执行并提供**时间边界标注**

最终数据统计上，论文强调几个点：
- **1000** 条人类标注轨迹
- **10,000+** step-level temporal segments
- 平均每个任务 **9.1** 个原子步骤
- 平均轨迹长度 **2972.4** 仿真步，约 **6×** 既有长程基准

### 评测协议：不是只看成功率
作者认为只看 SR（success rate）太粗，因此补了四个维度：
- **Task Success Rate (SR)**：最终任务是否完成
- **Plan Match Accuracy (AccP)**：预测计划与人工计划是否一致
- **Plan Efficiency (η)**：成功率相对计划长度的效率
- **Action Completion Accuracy (AccC)**：通过 VideoQA 测模型是否能判断某一步是否完成，近似反映“反思”能力

更关键的是，它采用了一个**固定 System 1、切换 System 2** 的协议：
- 低层 VLA controller 固定
- 高层 planner 换成不同 VLM/LLM
- 用 anchor points 控制子任务切换粒度

这样评测出来的差异，更接近**高层推理能力差异**，而不是低层 action decoder 的差异。

### HPE baseline：分层计划与执行
虽然论文主贡献是 benchmark，但也给了一个基线框架 HPE：
- **VLM planner**：低频观察下做高层计划、记忆更新、进度判断
- **VLA controller**：高频观察下执行细粒度动作
- **Memory bank**：保存当前子目标和上下文

这个 baseline 的作用不是“证明新方法 SOTA”，而是给 benchmark 一个可运行、可拆解的参照系。

### 核心直觉

**改变了什么？**  
从“短程 end-to-end 成功率”改成“长程、带记忆与扰动、并能拆解计划/反思/记忆的分层评测”。

**哪个瓶颈因此被改变？**  
原先 benchmark 的主要测量瓶颈是：**低层控制误差会掩盖高层推理问题**。  
现在通过：
- 长时序任务链，
- 时间边界标注，
- 固定 System 1，
- 多维指标，

把“高层推理失败”单独暴露出来。

**能力上带来了什么变化？**  
它不只是告诉你“模型成没成功”，而是告诉你：
- 它会不会规划，
- 会不会在动态场景里修计划，
- 会不会记住之前看过的信息，
- 会不会判断一步是否已经完成。

换句话说，RoboCerebra 把机器人 benchmark 从“动作回放考试”推向“认知诊断考试”。

### 为什么这套设计有效？
因果上看，关键不在于“用了更大的模型”，而在于**重构了测量对象**：

- 长 horizon 让模型必须维护更长的因果链
- 动态扰动让静态脚本式 plan 不再够用
- memory task 迫使模型利用历史上下文，而不是只看当前帧
- 时间标注让 progress monitoring 可以训练与量化
- 固定 System 1 让对比更接近“planner 能力”的 apples-to-apples

### 战略取舍

| 设计选择 | 获得的诊断能力 | 代价/副作用 |
|---|---|---|
| LLM 生成长任务并细分步骤 | 快速扩展任务覆盖，构造长程组合任务 | 任务分布会带有 LLM 先验与 prompt 偏置 |
| 符号验证 + VLM视觉验证 | 降低不合理场景与语义冲突 | 依赖外部大模型验证，流程更重 |
| 人类执行 + 时间标注 | 提供可靠的 temporal grounding | 标注成本高，扩展速度受限 |
| 固定 System 1、替换 System 2 | 更干净地评测高层推理 | 低估真实系统里 planner-controller 耦合难度 |
| 仿真环境中的动态/记忆任务 | 可重复、可控地制造长程难点 | 与真实机器人部署仍有域差距 |

**Q2 / How**：  
作者引入的关键“因果旋钮”是：**把 benchmark 设计成一个能隔离 System 2 的测量系统**。  
短任务 → 长任务；单成功率 → 多维诊断；end-to-end 混合误差 → 固定 System 1 后比较 planner。  
于是能力变化从“看似都会做一点”变成“能具体分辨谁在规划、反思、记忆上更强”。

## Part III：证据与局限

### 关键证据信号

#### 信号 1：纯 System 1 在长程任务上几乎不可用
表 3 里最直观的结论是：
- OpenVLA-Libero100 平均成功率只有 **2.00%**
- 在 RoboCerebra 上微调后的 OpenVLA* 也只有 **4.57%**
- 加入高层 planner 后，Planner+OpenVLA* 到 **16.04%**
- HPE 到 **16.55%**

这说明 benchmark 确实把问题拉到了**高层计划能力**上：  
仅靠更强的反应式 policy，不足以跨过长程任务门槛。

#### 信号 2：当前最强 planner 也远未饱和
在 planner ablation 中：
- **GPT-4o** 平均成功率 **16.04%**
- **GT-plan** 上界 **25.16%**

这说明两件事：
1. Benchmark 没有被“秒掉”，仍然有足够难度；
2. 当前主瓶颈确实仍是高层计划与任务分解，而不是单纯低层执行。

#### 信号 3：反思、视觉观察、任务成功并不完全同向
表 5 很有意思：
- GPT-4o 的 **AccP=68.33**、**SR=16.04** 最好
- 但 Qwen2.5-VL-7B-SFT 的 **AccC=66.83** 更高，**SR 却只有 9.33**
- GPT-4o-Blind 的 **SR=15.10**，和带视觉的 GPT-4o **16.04** 很接近

这暴露出 benchmark 的一个重要诊断结果：  
**“会观察/会反思”与“最终会做成任务”并不等价。**  
同时也说明，当前设定下**语言先验与计划能力仍然比在线视觉更新更主导**。

### 1-2 个最重要指标怎么读？
- **2972.4 平均轨迹长度**：说明这个 benchmark 的难点是真正来自长时依赖，而不是把几个短动作机械拼接。
- **16.55% vs 4.57% 成功率差**：说明只做 low-level finetune 不够，System 2 规划确实带来能力跳变。

### 能力跳跃到底在哪里？
相对以往 benchmark，RoboCerebra 的价值不在于“又多了一个数据集”，而在于它把能力跃迁定义得更清楚了：
- 从 **短期反应** 到 **长程分解**
- 从 **静态执行** 到 **动态修正**
- 从 **当前帧驱动** 到 **历史记忆驱动**
- 从 **单一成功率** 到 **可诊断的认知维度**

这就是它相对 prior benchmarks 的真正增量。

### 局限性

- **Fails when:** 需要真实机器人部署验证、复杂接触动力学、极强低层灵巧操作、或 planner 与 controller 深度双向闭环协同时；当前 benchmark 更擅长测高层 reasoning，不擅长测真实世界控制鲁棒性。
- **Assumes:** 基于 Libero 仿真平台；任务生成依赖 GPT/o3-mini；场景视觉一致性验证依赖 GPT-4o；需要大规模人工示教与时间标注（约 400 小时标注 + 200 小时检查）；baseline 训练依赖 8×A100。
- **Not designed for:** 直接证明 sim-to-real transferable、比较不同机器人本体形态、或公平评测纯低层 policy 学习能力。

还有两个更具体的边界值得点出：

1. **锚点对齐切换子任务**  
   这有助于隔离 planner，但也弱化了真实 end-to-end 系统里“何时切步”的难度。

2. **视觉观察尚未成为决定性因素**  
   GPT-4o-Blind 接近 GPT-4o，说明当前任务设计虽已引入 observation/reflection，但在线感知带来的增益还不够大，后续 benchmark 也许还需要更强的闭环感知依赖。

### 可复用组件
这篇工作的可复用价值很高，主要有：
- **LLM 级联任务生成模板**
- **规则化场景实例化**
- **符号 + VLM 双环验证流程**
- **step-level 时间标注范式**
- **固定 System 1 / 替换 System 2 的评测协议**
- **面向 planning / reflection / memory 的指标分解**

**Q3 / So what**：  
这篇论文最重要的“所以呢”是：它证明了**长程机器人操作的主要瓶颈已不仅是控制，而是可测量、可分解的高层推理**。最强实验证据不是某个模型分数多高，而是：
- 纯 System 1 明显失效，
- 加入 System 2 后有大幅提升，
- 但即使 GPT-4o 也离 GT-plan 仍有明显差距。

这说明该 benchmark 既**测到了新能力**，也**留下了足够清晰的研究空间**。

![[paperPDFs/Benchmark/NeurIPS_2025/2025_RoboCerebra_A_Large_scale_Benchmark_for_Long_horizon_Robotic_Manipulation_Evaluation.pdf]]