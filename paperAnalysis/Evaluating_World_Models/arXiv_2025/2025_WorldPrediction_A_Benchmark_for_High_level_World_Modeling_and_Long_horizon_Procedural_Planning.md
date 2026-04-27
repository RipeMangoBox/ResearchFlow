---
title: "WorldPrediction: A Benchmark for High-level World Modeling and Long-horizon Procedural Planning"
venue: arXiv
year: 2025
tags:
  - Survey_Benchmark
  - task/video-understanding
  - task/MLLM-evaluation
  - counterfactual-distractors
  - action-equivalent-sampling
  - posmdp-grounding
  - dataset/COIN
  - dataset/CrossTask
  - dataset/EgoExo4D
  - dataset/EPIC-KITCHENS-100
  - dataset/IKEA-ASM
  - opensource/no
core_operator: 以POSMDP为理论框架，把高层世界建模与长程程序规划统一成基于初末状态的纯视觉多选判别，并用动作等价物抑制背景捷径。
primary_logic: |
  高层世界建模/长程规划评测需求 → 从5个人类活动视频数据集中构造“初始状态-最终状态-动作/动作序列候选”样本，并进行可观测性过滤、动作等价替换与双人验证 → 用多选准确率评估模型是否识别导致状态转移的正确动作或正确顺序 → 揭示当前模型在高层因果转移理解与长程规划上的能力边界
claims:
  - "经双人独立过滤后，WorldPrediction包含825个WM样本和570个PP样本，且人类在保留样本上达到100%正确率 [evidence: analysis]"
  - "当前最强前沿模型在WorldPrediction-WM和WorldPrediction-PP上分别仅达到57.0%与38.1%准确率，显著低于人类表现 [evidence: analysis]"
  - "OEPP规划器在COIN/CrossTask域内可达49.2%，但在EgoExo4D、EPIC-KITCHENS-100与IKEA-ASM等域外数据上降至约29%，显示长程程序规划泛化性不足 [evidence: analysis]"
related_work_position:
  extends: "MMWorld (He et al. 2024)"
  competes_with: "WorldScore (Duan et al. 2025); PlanBench (Valmeekam et al. 2023)"
  complementary_to: "Open-Event Procedure Planning (Wu et al. 2024); DINO-WM (Zhou et al. 2024a)"
evidence_strength: moderate
pdf_ref: paperPDFs/Evaluating_World_Models/arXiv_2025/2025_WorldPrediction_A_Benchmark_for_High_level_World_Modeling_and_Long_horizon_Procedural_Planning.pdf
category: Survey_Benchmark
---

# WorldPrediction: A Benchmark for High-level World Modeling and Long-horizon Procedural Planning

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2506.04363)
> - **Summary**: 这篇工作提出一个纯视觉、架构无关的基准，把“高层世界建模”和“长程程序规划”统一成带反事实干扰项的多选判别题，用来测量模型是否真的理解动作导致的状态变化，而不是依赖背景连续性或文本标签。
> - **Key Performance**: 最优模型在 WM / PP 上仅 57.0% / 38.1%，而人类在筛选后样本上为 100%

> [!info] **Agent Summary**
> - **task_path**: 初始/最终状态图像 + 候选动作视频或候选动作序列 -> 选择导致状态转移的正确动作/正确排序计划
> - **bottleneck**: 现有评测过度依赖文本或开放生成输出，且常允许模型利用背景/视角连续性捷径，无法单独测出高层动作-状态因果理解
> - **mechanism_delta**: 用动作等价物替换真实动作片段，并在 POSMDP 框架下把单步与多步因果判断都改写为纯视觉反事实判别
> - **evidence_signal**: 五个数据源、双人过滤后人类 100%，但前沿模型仍只有 57.0%/38.1%，说明该基准确实测到了现有系统的能力缺口
> - **reusable_ops**: [action-equivalent substitution, observability filtering]
> - **failure_modes**: [依赖场景外观相似性而非动作因果, 长程动作排序在跨域场景中明显退化]
> - **open_questions**: [四选一判别能否外推出开放式规划能力, 去除文本标签后模型瓶颈更偏感知还是内部世界模型]

## Part I：问题与挑战

**What/Why：真正的问题是什么，为什么现在要测？**

这篇论文要解决的不是“模型会不会看视频”这么泛的问题，而是一个更尖锐的评测问题：

1. **模型能否理解高层动作如何让世界状态发生变化？**
2. **模型能否在只看到初态和终态时，推断中间正确的长程动作顺序？**

作者认为，现有世界模型/规划评测有三个核心缺口：

- **评测层级不对**：很多工作测的是低层物理、机器人控制、导航或很短的 3-4 步规划；但真实人类活动常常是语义更抽象、持续时间不固定的高层动作。
- **评测接口不公平**：有些基准要求文本回答，有些只适合视频生成模型，导致不同架构之间难以直接比较。
- **存在视觉捷径**：如果正确动作片段和前后状态共享同一背景、机位、人物或物体布局，模型可能只是在做“连续镜头匹配”，而不是理解动作-状态因果。

### 输入/输出接口

作者把任务统一成纯视觉多选题：

- **WorldPrediction-WM**：
  - 输入：初始状态图像 + 最终状态图像 + 4 个候选动作视频
  - 输出：选出哪个动作导致了该状态变化
- **WorldPrediction-PP**：
  - 输入：初始状态图像 + 最终状态图像 + 4 个候选动作序列
  - 输出：选出哪个动作序列的顺序是正确的

这里的关键边界条件是：

- 只用**视觉观察**表示状态和动作；
- 不要求模型生成文本计划或视频；
- PP 的计划长度覆盖 **3 到 10 步**；
- 每题固定 **4 选 1**；
- 场景来自烹饪、维修、装配、医疗护理等**人类技能活动**。

### 真正的评测瓶颈

这篇论文针对的“真瓶颈”不是模型参数不够大，而是**缺少一个能把高层因果理解与表面相关性区分开的评测协议**。  
如果这个测量问题不先解决，那么“世界模型是否真的存在”就很难靠现有 benchmark 说清。

---

## Part II：方法与洞察

这篇论文不是提出一个新模型，而是**重新设计测量方式**。

### 评测框架

作者用一个受 **POSMDP（部分可观测半 MDP）** 启发的形式化框架来定义问题：

- 潜在世界状态是隐藏的；
- 高层动作既有**语义抽象**，也有**时间抽象**；
- 图像/视频只是对状态和动作的部分观察；
- 模型真正该学的是隐藏的状态转移机制，而不是像素连续性。

在这个框架下：

- **WM** 测的是：给定前后状态，模型能否把正确动作排在所有候选前面；
- **PP** 测的是：给定初态和终态，模型能否把正确动作序列排在所有候选计划前面。

也就是说，作者把“世界建模/规划”从开放生成问题，转成了**反事实区分问题**。

### 核心直觉

**改变了什么？**  
从“让模型自由生成未来/文本解释”改成“让模型在反事实候选里选出真正导致状态变化的动作或计划”。

**哪个测量瓶颈被改变了？**  
这个改动把评测重心从：

- 文本表达能力、
- 视频生成质量、
- prompt 风格适配性、

转移到了：

- 是否理解**动作与状态变化的因果关系**，
- 是否能跨背景/视角识别同一动作语义，
- 是否能在看不到中间状态时推断**多步顺序约束**。

**为什么这会有效？**  
因为作者进一步加入了两个关键机制：

1. **Action Equivalents（动作等价物）**  
   正确动作不再用原视频里的那一段，而是换成“相同动作语义、不同背景/视角”的等价片段。  
   这样做等于主动打断“背景连续性捷径”，逼模型关注动作本身。

2. **Observability Filtering（可观测性过滤）**  
   如果前后状态本身看不出关键变化，题目就会变成对人也不公平。  
   所以作者用视觉特征距离、遮挡检查和双人筛选，把“视觉证据不足”的题目去掉。

于是，这个 benchmark 不是在问：  
“模型能不能说得像、生成得像？”  
而是在问：  
“模型能不能识别哪个高层动作/动作链真的解释了这次世界变化？”

### 设计细节抓手

- **多源数据覆盖**：COIN、CrossTask、EgoExo4D、EPIC-KITCHENS-100、IKEA-ASM
- **双任务拆分**：
  - WM 测单步高层因果
  - PP 测长程顺序推理
- **干扰项设计**：
  - WM：从同一任务上下文中采样合理但错误的动作
  - PP：对同一组动作做错误重排，保证动作本身都“像是对的”，错在顺序
- **人工验证**：
  - 每个任务先构造 1500 样本
  - 每题 2 位标注者独立作答
  - 仅保留两人都答对的样本

### 战略权衡

| 设计选择 | 解决的测量问题 | 带来的诊断能力 | 代价/副作用 |
|---|---|---|---|
| 纯视觉多选判别 | 开放生成难评分、跨架构不可比 | 可统一比较 VLM、LLM、diffusion、规划器 | 候选空间被压缩，难度低于开放式规划 |
| 动作等价物 | 背景/机位连续性捷径 | 更接近测“动作语义 -> 状态变化” | 构造依赖原始数据集的动作标签或分段质量 |
| 可观测性过滤 | 题目因遮挡/剪辑而对人都不可解 | 提升 benchmark 可靠性与人类上限 | 过滤后数据更干净，低估真实世界噪声 |
| PP 用同一动作集合打乱顺序 | 动作识别与顺序规划混在一起 | 更聚焦长程顺序约束和隐式中间状态推断 | 仍不是开放式计划生成 |
| 不使用文本标签作正式输入 | 避免文本标注质量成为上限 | 更强调视觉 grounding | 对纯语言规划系统不友好 |

---

## Part III：证据与局限

### 关键证据信号

**1. 上限信号：人类与模型差距非常大**  
这是最强证据。双人过滤后，人类在保留样本上是 **100%**；但最强模型只有：

- **WM：57.0%**（Qwen2.5-VL 72B）
- **PP：38.1%**（Claude-3.5 Sonnet）

这说明 benchmark 没有被做成“人也解不出来”的噪声集合，而是真正把机器能力边界测出来了。

**2. 规模信号：大模型对单步世界建模有帮助，但对长程规划帮助有限**  
例如 InternVL2.5 从 26B 到 38B，WM 有明显跃升；但 PP 没有对应幅度的提升。  
这表明**看懂单步变化**和**规划多步顺序**不是同一个瓶颈，后者更依赖隐式中间状态推断与长程约束处理。

**3. 架构信号：感知 grounding 与纯文本推理各有短板**  
Socratic LLM 用高质量 caption 后，成绩能接近甚至在 PP 上超过部分 VLM，说明文本推理仍有价值；  
但最好 WM 模型和最好 PP 模型并不相同，说明：

- VLM 更强在直接视觉 grounding；
- 文本 LLM 更强在长程符号化推理；
- 两者都还没形成足够强的“高层世界模型”。

**4. 生成式世界模型信号：视频 diffusion 并不等于更强的世界理解**  
I2VGenXL / CogVideoX 在 WM 上仅约 **26%-30%**。  
而且把最后一帧与目标状态做距离匹配，效果也没有明显提升。  
这意味着**像素空间生成能力**并不自动转化为**动作-状态因果建模能力**。

**5. 泛化信号：训练式规划器域内强，域外弱**  
OEPP 在 COIN / CrossTask 上最好可到 **49.2%**，但在 EgoExo4D、EPIC-KITCHENS-100、IKEA-ASM 上掉到约 **29%**。  
所以它学到的更多像是**数据域内程序模式**，而不是可迁移的高层规划能力。

### 局限性

- **Fails when**: 输入存在严重遮挡、剧烈镜头切换、只有非常细微的状态变化、或任务需要开放式生成整条计划时，该 benchmark 不能完整覆盖；作者事实上已把大量此类样本过滤掉。
- **Assumes**: 假设源数据集有足够好的动作分段/标签来检索“动作等价物”；假设 DINOv2 特征距离和 VLM 遮挡判定能代表“可观测性”；假设双人标注过滤足以保证样本可解；部分基线依赖闭源 API（如 GPT-4o、Claude、Gemini），影响完全复现。
- **Not designed for**: 低层控制、机器人闭环执行、交互式 replanning、基于奖励的决策、开放式文本/视频计划生成，也不直接评估模型校准度或不确定性表达。

### 复现与报告上的现实约束

- benchmark 构造依赖显著的**人工筛选成本**：34 位 WM 标注者、46 位 PP 标注者。
- 过滤流程依赖**预训练特征模型**与**VLM 可见性检查**，因此不是“完全无外部先验”的 benchmark。
- 文中提到 benchmark 已 release，但在给定文本里**没有明确项目链接**，因此元数据里只能保守记为 `opensource/no`。
- 正文 §4.3 对 WM 最佳结果的文字描述与摘要/Table 2 有一处不一致；解读实验时应以摘要和 Table 2 为主。

### 可复用组件

这篇工作最值得复用的不是某个模型，而是三种评测操作符：

1. **action-equivalent substitution**：把正确样本替换成不同背景/视角下的语义等价动作，专门打捷径。
2. **observability filtering**：先过滤“人都很难从视觉上判断”的样本，再谈模型能力。
3. **counterfactual multiple-choice evaluation**：把开放式世界建模/规划压缩成统一可比的反事实选择题，方便跨架构评估。

**So what：能力跃迁到底体现在哪？**  
这篇论文最大的价值，不是刷新某个 SOTA，而是把一个常被“生成看起来像”掩盖的问题拆开了：  
当前模型也许能感知、能描述、能生成，但在**高层动作造成的世界状态变化**与**长时程程序顺序**上，离人类还很远。这个 benchmark 给了后续工作一个更干净的诊断面。

## Local PDF reference

![[paperPDFs/Evaluating_World_Models/arXiv_2025/2025_WorldPrediction_A_Benchmark_for_High_level_World_Modeling_and_Long_horizon_Procedural_Planning.pdf]]