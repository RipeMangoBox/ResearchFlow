---
title: "World-in-World: World Models in a Closed-Loop World"
venue: arXiv
year: 2025
tags:
  - Survey_Benchmark
  - task/world-model-evaluation
  - task/embodied-planning
  - closed-loop-evaluation
  - online-planning
  - action-api
  - dataset/Matterport3D
  - dataset/HM3D
  - dataset/OpenEQA
  - dataset/RLBench
  - opensource/promised
core_operator: 把异构世界模型统一包进“提案-模拟-修正”的闭环规划接口，并以具身任务成功率而非画质作为主评测信号。
primary_logic: |
  评估世界模型对具身决策的真实帮助 → 用统一动作 API 与“提案-模拟-修正”闭环协议接入文本/视角/低层动作控制的异构世界模型，并覆盖感知、导航、问答、操控四类任务 → 以任务成功率、路径效率和回答得分为主评分并分析后训练/推理扩展 → 揭示“高画质≠高闭环成功”，而可控性、动作对齐和测试时搜索更关键
claims:
  - "在 AR、ImageNav、A-EQA 和 manipulation 四类任务上，把世界模型接入闭环规划后，相比对应 base policy 整体带来更高任务表现，例如 AR 从 50.27% 提升到最高 64.79%，ImageNav 从 35.42% 提升到最高 46.53% SR [evidence: comparison]"
  - "视频生成画质不是具身成功率的可靠代理指标，而动作可控性与 AR 成功率呈更强正相关 [evidence: analysis]"
  - "使用动作-观测数据做 post-training 与增加推理时 rollout 次数，都比单纯升级更大的预训练视频生成器更稳定地提升闭环表现 [evidence: ablation]"
related_work_position:
  extends: "WorldScore (Duan et al. 2025)"
  competes_with: "WorldModelBench (Li et al. 2025); WorldScore (Duan et al. 2025)"
  complementary_to: "EmbodiedBench (Yang et al. 2025); VBench (Huang et al. 2024)"
evidence_strength: strong
pdf_ref: "paperPDFs/Evaluating_World_Models/Code_is_at_https_github_com_World_In_World_world_in_world_2025/2025_World_in_World_World_Models_in_a_Closed_Loop_World.pdf"
category: Survey_Benchmark
---

# World-in-World: World Models in a Closed-Loop World

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2510.18135), [Project](https://world-in-world.github.io), [GitHub (promised)](https://github.com/World-In-World/world_in_world)
> - **Summary**: 这篇工作提出首个把世界模型放进真实闭环具身任务中统一评测的平台，用任务成功率而不是视频画质来判断世界模型是否真的能帮助感知、规划与决策。
> - **Key Performance**: AR 中 VLM 基线为 50.27% SR，最佳开源后训练模型可达 62.61%，闭源 Runway Gen4 达 64.79%；ImageNav 中最佳开源模型达到 46.53% SR / 34.61 SPL。

> [!info] **Agent Summary**
> - **task_path**: 当前观测 + 任务目标 + 候选动作序列 -> 世界模型反事实 rollout -> 选定下一步动作/计划/答案
> - **bottleneck**: 现有评测只看开放环画质或可视合理性，不能衡量世界模型对闭环具身成功率的真实贡献；同时不同世界模型控制接口不统一，难以公平比较
> - **mechanism_delta**: 用统一动作 API 把 text/viewpoint/low-level action 条件的异构世界模型接到同一个 propose-simulate-revise 闭环协议里
> - **evidence_signal**: 四类任务上的跨模型对比 + 画质/可控性相关性分析 + 后训练与推理时扩展实验
> - **reusable_ops**: [unified-action-api, propose-simulate-revise-planning]
> - **failure_modes**: [long-horizon-drift, contact-dynamics-mismatch]
> - **open_questions**: [如何提升未见环境中的动作可控泛化, 如何让世界模型稳定支持长时程记忆与接触动力学]

## Part I：问题与挑战

**What/Why：真正的瓶颈不是“视频够不够好看”，而是“预测能不能在闭环里改变决策并提高成功率”。**

这篇论文针对的是一个被长期回避的问题：**世界模型作为“可想象未来”的生成器，是否真的能帮助 embodied agent 完成任务**。过去大多数 benchmark 评的是开放环生成质量，比如美观度、清晰度、时序一致性、可视 plausibility；但对于具身智能来说，真正关键的是：

1. 预测是否**遵守动作条件**；
2. 预测是否能帮助 agent **选更好的下一步动作**；
3. 误差在**反复执行-再观察-再规划**的闭环中会不会迅速累积。

### 真问题在哪里
作者指出了两个核心缺口：

- **测量缺口**：现有 world model benchmark 主要看视觉质量，没法回答“这个模型是否能提升任务成功率”。
- **接口缺口**：不同世界模型吃的控制信号不同，有的吃文本，有的吃相机轨迹，有的吃低层动作；没有统一接口就无法公平横评。

### 为什么现在必须解决
因为视频/世界生成模型已经足够强，研究社区开始自然地把它们视作“mental simulator”或“predictive perception”模块。如果此时仍用开放环视觉指标代替闭环效用，研究方向会被误导：**大家会优化更好看的 rollout，而不一定是更有用的 rollout**。

### 输入/输出与边界条件
在 World-in-World 里，系统输入不是单纯一张图，而是：

- 当前观测 `o_t`
- 任务目标 `g`
- 候选动作序列

输出也不只是视频，而是**用于决策的最佳选择**，它可以是：

- 下一段动作计划
- 识别答案
- 问答回答

边界上，这个 benchmark 聚焦于**模拟器中的闭环具身任务**，覆盖四类能力：

| 任务 | 测什么能力 | 主要环境/数据 |
|---|---|---|
| Active Recognition | 遮挡下主动感知 + 找更好视角 | Matterport3D / Habitat-Sim |
| ImageNav | 目标图导航与路径决策 | HM3D |
| A-EQA | 主动探索后的问答 | OpenEQA + HM3D |
| Robotic Manipulation | 动作条件下的物理交互预测 | RLBench |

---

## Part II：方法与洞察

**How：作者引入的关键因果旋钮，是把“世界模型评测对象”从开放环视频样本，改成闭环决策中的反事实模拟器。**

### 方法骨架：统一闭环协议
作者提出一个统一的闭环流程：**提案（proposal）→ 模拟（simulation）→ 修正（revision）**。

1. **Proposal**：proposal policy 根据当前观测和目标，生成多个候选动作序列。
2. **Simulation**：统一动作 API 把这些动作序列转成世界模型能接受的控制格式。
3. **Revision**：revision policy 根据每个 rollout 的预测结果，对候选计划打分，选出最优决策。
4. **Execution**：把选中的动作执行到真实环境里，再拿到新观测，继续下一轮。

这件事的关键不在“多了一层规划”，而在于：**世界模型第一次被放进真实的执行-反馈回路里接受考验**。  
如果模型只是“看起来像”，但对动作不敏感，它就会直接误导搜索与决策。

### 统一动作 API
论文的第二个关键设计，是把异构世界模型统一到一个控制抽象里。它支持三类输入：

- **文本 prompt**
- **相机轨迹 / 视角**
- **低层动作**

这一步解决的是 benchmark 里最现实的问题：  
不是所有 world model 都天生为 embodied planning 设计，但只要能被映射到统一控制 API，就能被放进同一闭环协议中比较。

### 任务覆盖设计
这个 benchmark 不只测一个任务，而是有意识地覆盖不同 failure mode：

- **AR**：看模型能否补足被遮挡/极端视角下的感知证据；
- **ImageNav**：看模型能否帮助规划出更接近目标图的路径；
- **A-EQA**：看模型能否支持探索策略，而不是只生成“看起来合理”的未来；
- **Manipulation**：看模型是否能预测接触、位姿变化和物体交互。

这让 benchmark 不只是一个 leaderboard，而是一个**能力剖面图**。

### 后训练协议
作者还加入了一个很重要的诊断维度：  
把现成视频生成器用少量**动作-观测数据**做 post-training，检查它们能否快速转化为真正有用的 embodied world model。

- Habitat 任务：用 HM3D 训练场景采集的 pano action-observation 数据
- Manipulation：用 RLBench demonstrations

这样可以单独评估一个问题：  
**“更大的预训练视频模型” vs “更贴近任务动作空间的后训练”**，哪一个更重要？

### 核心直觉

从因果上看，这篇论文改变了三件事：

1. **评测分布变了**：  
   从“给定条件，单次生成是否逼真”，变成“给定当前观测和候选动作，反事实预测是否能改善下一步决策”。

2. **约束变了**：  
   过去视频预测只受视觉 plausibility 约束；现在还必须受**动作一致性**和**任务回报**约束。

3. **因此可测能力变了**：  
   benchmark 不再只测生成质量，而开始测：
   - 是否可控
   - 是否可规划
   - 是否能在闭环中支持重规划
   - 是否能跨任务泛化

**为什么这设计有效？**  
因为闭环执行会把微小的动作误差放大。一个只会“生成看起来对”的模型，可能在视觉评分上很高，但只要它忽略动作或漂移，planner 就会在错误未来上做选择，最终任务失败。  
所以在 embodied setting 下，**controllability 比 aesthetics 更接近真正的因变量**。

### 战略取舍

| 设计选择 | 得到什么 | 代价/风险 |
|---|---|---|
| 统一动作 API | 不同控制接口的世界模型可公平接入与比较 | 映射过程可能损失各模型原生优势 |
| proposal-simulate-revise | 真正测到反事实规划价值 | 表现受 proposal/revision policy 强弱影响 |
| 四任务覆盖 | 能区分感知、导航、问答、物理交互能力 | 很难用单一分数概括全部能力 |
| 加入 post-training | 能分析“可适配性”与数据 scaling law | 不再是纯 zero-shot，对资源有要求 |
| 以任务成功为主指标 | 更接近 embodied utility | 视觉质量优秀但不服务任务的模型会被“降权” |

---

## Part III：证据与局限

**So what：这套 benchmark 给出的最大结论不是“哪个模型最好”，而是“什么样的世界模型在闭环里真的有用”。**

### 关键实验信号

- **比较信号：接入世界模型后，base policy 在四类任务上整体变强。**  
  例如在 AR 中，VLM base policy 为 **50.27% SR / 6.24 steps**，接入最强模型后可到 **64.79% SR / 4.06 steps**；  
  在 ImageNav 中，VLM base policy 为 **35.42% SR**，最佳开源后训练模型达到 **46.53% SR / 34.61 SPL**。  
  这说明世界模型确实可以作为决策辅助，而不只是“生成器”。

- **分析信号：画质高，不代表闭环成功率高。**  
  论文直接比较了 generation quality 与 task success，发现相关性并不可靠；反而是**动作可控性**与成功率更相关。  
  这基本推翻了“更好看的视频 = 更好的 world model”这一默认假设。

- **消融信号：后训练比单纯换更大预训练模型更有效。**  
  例如 Wan2.1 在 AR 上从 **58.26%** 提升到 **62.61%**，ImageNav 从 **38.19%** 到 **45.14%**；  
  SVD 也从 **57.71%** 到 **60.98%**。  
  结论很明确：**对动作空间和环境分布做对齐**，比纯粹扩大 web-video pretraining 更直接。

- **扩展信号：推理时算力也能换性能。**  
  在 AR 中，SVD† 平均 inference count 从 **3** 增加到 **11**，SR 从 **53.36%** 提升到 **60.98%**。  
  这说明世界模型的价值不只取决于训练，也取决于测试时能否进行更充分的 rollout 搜索。

- **边界信号：操控任务提升有限。**  
  manipulation 上最佳提升很小，例如 VLM baseline 为 **44.5% SR**，SVD† 为 **46.5% SR**。  
  这说明当前视觉世界模型对于**接触动力学、摩擦、物体状态变化**仍然不够可靠。

### 1-2 个最值得记住的数
- **AR**：50.27% -> 64.79% SR，证明闭环世界模型能显著提高主动感知成功率。
- **Inference-time scaling**：3 -> 11 次模型推理可把 SVD† 的 AR 成功率从 53.36% 提升到 60.98%。

### 局限性

- **Fails when**: 需要长时程历史累积、精确接触动力学或复杂物理一致性的场景；特别是 manipulation 中，模型容易在接触、摩擦、关节运动和物体状态变化上失真。
- **Assumes**: 可访问模拟器环境与任务内统一动作空间；依赖较强的 proposal/revision policy（默认 Qwen2.5-VL-72B）；部分模型依赖额外模态或外部工具，如 GT depth、semantic map、YOLO-World、SAM2；后训练仍需额外数据与算力（约 5–74 H100 GPU-hours / model / 40K clips）。
- **Not designed for**: 真实机器人部署评测、开放世界长期记忆建模、跨 embodiment 通用动作表示学习，或直接替代物理引擎的高精度仿真。

### 复现与可扩展性的现实约束
- 最佳 AR 结果包含 **Runway Gen4** 这类闭源模型，因此 SOTA 上限并非完全可复现。
- post-training 虽然比从头训练轻量得多，但仍需要专门的动作-观测数据收集管线。
- benchmark 中不同 world model 的输入规格不一，有些需要 panorama，有些要 front view，有些需要额外深度/语义，这意味着“统一协议”本身仍有适配工程成本。

### 可复用组件
这篇工作最值得复用的不是某一个分数，而是以下四个算子：

1. **统一动作 API**：把 text / viewpoint / low-level action 条件统一到一个评测接口；
2. **proposal-simulate-revise 闭环壳**：可以直接套到其他 embodied task 上；
3. **任务成功优先的评价协议**：用 success / SPL / answer score 替代单纯画质；
4. **动作-观测后训练 recipe**：用于把通用视频生成器快速变成 task-aligned world model。

总体看，这篇论文的能力跳跃在于：它把“世界模型是否有用”从一个模糊直觉，变成了一个**可闭环、可比较、可扩展分析**的问题。相比先前工作最大的不同，不是又做了一个更强模型，而是**重新定义了什么叫 world model 的有效性**。

## Local PDF reference

![[paperPDFs/Evaluating_World_Models/Code_is_at_https_github_com_World_In_World_world_in_world_2025/2025_World_in_World_World_Models_in_a_Closed_Loop_World.pdf]]