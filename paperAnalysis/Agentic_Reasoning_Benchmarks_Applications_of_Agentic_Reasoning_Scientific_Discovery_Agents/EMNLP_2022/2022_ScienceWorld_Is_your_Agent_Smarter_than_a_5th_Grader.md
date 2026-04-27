---
title: "ScienceWorld: Is your Agent Smarter than a 5th Grader?"
venue: EMNLP
year: 2022
tags:
  - Survey_Benchmark
  - task/scientific-reasoning
  - text-simulator
  - executable-evaluation
  - masked-object-variation
  - dataset/ScienceWorld
  - opensource/full
core_operator: 用带物理/化学/生物模拟的交互式文本世界，把小学科学问答改写为可执行实验，并用环境结果自动评分
primary_logic: |
  科学任务描述与局部文本观察 → 在含热学/电路/化学/生命过程规则的文本环境中执行动作序列，
  并通过命名/匿名对象变体与目标达成度评分 → 诊断 agent 是否具备可迁移的科学程序性推理
claims:
  - "SCIENCEWORLD 提供 10 个科学主题下的 30 个子任务、7,200 个参数化变体和 25 个高层动作，并用命名/匿名对象配对区分检索与实验性推理 [evidence: analysis]"
  - "在 30 个子任务上，最佳基线 DRRN 的平均归一化得分仅为 0.17，表明当前文本 agent 在小学科学实验推理上整体表现较弱 [evidence: comparison]"
  - "1.5M 参数的 DRRN 平均成绩超过 11B 的行为克隆/决策 Transformer 模型（0.17 vs 0.08），说明交互式 grounded 学习在该基准上优于大规模离线模仿 [evidence: comparison]"
related_work_position:
  extends: "TextLabs (Tamari et al. 2021)"
  competes_with: "TextLabs (Tamari et al. 2021); TextWorld Commonsense (Murugesan et al. 2020a)"
  complementary_to: "ALFWorld (Shridhar et al. 2020); WorldTree (Jansen et al. 2018)"
evidence_strength: moderate
pdf_ref: "paperPDFs/Agentic_Reasoning_Benchmarks_Applications_of_Agentic_Reasoning_Scientific_Discovery_Agents/EMNLP_2022/2022_ScienceWorld_Is_your_Agent_Smarter_than_a_5th_Grader.pdf"
category: Survey_Benchmark
---

# ScienceWorld: Is your Agent Smarter than a 5th Grader?

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2203.07540), [Project](https://sciworld.apps.allenai.org), [Code](https://github.com/allenai/ScienceWorld)
> - **Summary**: 这篇工作把“小学科学题答对了吗”改成“agent 能不能在文本世界里把实验做对”，从而更直接地区分答案检索与真正可执行的科学推理。
> - **Key Performance**: 最佳 DRRN 在 30 个子任务上的平均分仅 0.17；1.5M 参数 DRRN 明显超过 11B 的离线 BC/TDT 模型（0.08）。

> [!info] **Agent Summary**
> - **task_path**: 科学任务描述 + 局部文本环境观察 -> 动作序列 -> 环境验证后的任务完成分数
> - **bottleneck**: 静态 science QA 无法检验模型是否能把声明性知识转成新情境中可执行的实验步骤
> - **mechanism_delta**: 用带热学/电路/化学/生物模拟的交互式文本环境替代选择题与单一 gold explanation，以执行结果而非文本匹配评估推理
> - **evidence_signal**: 30 个子任务综合比较中，最佳 DRRN 仅得 0.17，且 1.5M 交互式 agent 超过 11B 离线模仿模型的 0.08
> - **reusable_ops**: [executable-procedure-evaluation, named-vs-masked-task-pairs]
> - **failure_modes**: [长链导航与操作组合导致探索失败, 语言模型生成动作与有效动作空间错位]
> - **open_questions**: [去掉valid-action detection aid后性能会下降多少, 如何把高层why解释与可执行how程序统一评测]

## Part I：问题与挑战

这篇 paper 抓住的核心问题不是“模型记不记得科学事实”，而是：

**模型能否把科学知识变成在新环境里可执行、可验证、可迁移的程序。**

### 现有评测为什么不够
1. **静态 QA 太容易被检索捷径污染**  
   大模型在小学/初中科学问答上已能拿到很高分，但这并不等于它能在新情境里真正“做”出科学实验。
2. **解释评测常依赖单一 gold explanation**  
   同一个实验可能有多种等价解法；文本解释只做字符串或单参考比对，很难判断“过程是否真的正确”。
3. **真正瓶颈在程序化落地**  
   例如“知道金属通常导电”不等于“能在环境中找到金属叉、搭出电路、观察灯泡是否亮、再给出正确结论”。

### 这篇工作要测什么
SCIENCEWORLD 要测的是一种更接近“会做实验”的能力：  
**声明性科学知识 + 常识操作能力 + 长时程规划 + 环境交互执行。**

### 输入 / 输出接口
- **输入**：任务描述 + 当前文本观测
- **输出**：一步步自然语言动作命令
- **评测对象**：动作序列是否真的把环境推进到目标状态

### 边界条件
- 这是一个**文本交互环境**，不是 3D 物理世界。
- 环境是**部分可观测**的，agent 需要导航、开容器、找物体。
- 每步动作空间很大：25 类高层动作，约 **20 万**个动作-对象组合候选。
- 物理/化学/生物仿真只做到**小学科学保真度**，不是高精度现实模拟。

一句话说，作者认为：  
**今天该解决这个问题，是因为“答对题”这个目标已经太弱，已经不足以区分记忆式模型和真正能推理、能操作的 agent。**

---

## Part II：方法与洞察

### 评测框架怎么搭

SCIENCEWORLD 是一个带专门模拟器的交互式文本世界，包含：
- **10 个互联地点**
- 最多约 **195-200 类对象**
- **25 个高层动作**
- 多个科学模拟引擎：
  - 热学
  - 电路
  - 设备激活
  - 化学混合
  - 生命周期
  - 繁殖与遗传
  - 斜面摩擦
  - 容器传播效应

它把小学科学课程拆成 **10 个主题、30 个子任务**，如：
- 物态变化
- 温度测量
- 电路与导电性
- 植物生长与授粉
- 生命阶段
- 孟德尔遗传
- 摩擦与斜面

并为这些任务设计了 **7,200 个参数化变体**，通过改变：
- 关键物体
- 起始房间
- 环境物品布局
- 某些设备可用性

来避免模型死记路径。

### 评测设计上的三个关键操作

#### 1. 把“回答问题”改成“执行实验”
例如，不再问：
- “某物体是否导电？”

而是要求 agent：
- 找到该物体
- 去工作间
- 连接电池、灯泡、导线和该物体
- 观察灯泡是否点亮
- 把物体放入正确答案盒

这样，**动作序列本身就成为 procedural explanation**。

#### 2. 用命名/匿名对象配对，切开“检索”与“实验”
很多任务有两种版本：
- **命名对象**：如 metal fork
- **匿名对象**：如 unknown substance B

如果模型只是记住了“metal fork 导电”，它在匿名版本就必须真正做实验。

#### 3. 用环境结果自动判分，而不是比对唯一文本答案
评分由：
- **required goals**
- **optional subgoals**

共同构成，最后归一化到 0-1。  
这让评测更接近“是否成功完成任务”，而不是“是否说出了某个参考答案”。

### 核心直觉

原来的评测链路是：

**题目 → 文本答案/解释 → 字符串级正确性**

这篇工作改成：

**题目 → 动作程序 → 环境状态变化 → 可执行正确性**

这带来的因果变化是：

- **改变了什么**：把输出空间从“文本解释”换成“可执行动作序列”
- **改变了哪个瓶颈**：不再让模型靠词面相似度或记忆模板过关，而必须跨过导航、操控、观察、实验规划这些真实约束
- **能力上发生了什么变化**：可以测到模型是否真的会把科学知识迁移到新对象、新房间、新配置中

更重要的是，这种设计天然支持**多解**。  
比如“把冰融化”可以用炉子、火堆、或其他加热方式；只要终态对，环境就能验证，而不需要唯一参考解释。

### 为什么这套设计有效
因为它把“科学推理是否真实存在”变成了**可运行的因果检验**：

- 如果模型只会背答案，匿名对象会让捷径失效。
- 如果模型不会把知识转成步骤，就会卡在导航、取物、连接、观察这些中间环节。
- 如果模型过程不对，环境终态不会变化，评分也上不去。

### 战略权衡表

| 设计选择 | 改变的测量瓶颈 | 带来的能力诊断 | 代价/风险 |
|---|---|---|---|
| 交互式文本环境替代静态 QA | 从答案匹配变成序列决策 | 能测规划、操作、实验执行 | 动作空间巨大，训练更难 |
| 多科学模拟器（热学/电路/生物等） | 从词面知识变成状态转移 | 能测因果操作是否正确 | 仿真保真度有限，只到小学水平 |
| 命名 vs 匿名对象配对 | 压制事实检索捷径 | 区分“知道答案”和“会做实验” | 任务设计更复杂 |
| required + optional goals | 缓解奖励稀疏 | 支持细粒度诊断与 RL 学习 | 可能对 canonical solution 有轻微偏置 |
| valid-action detection aid（部分基线使用） | 降低无效动作搜索难度 | 让模型比较更稳定 | 也弱化了真实开放动作空间难度 |

---

## Part III：证据与局限

### 关键实验信号

#### 信号 1：跨 30 个子任务，当前 agent 整体表现很低
作者评测了 5 类主流 agent：
- DRRN
- KG-A2C
- CALM-GPT2
- Behavior Cloning
- Text Decision Transformer

核心结论不是“谁最好”，而是：

**大家都不够好。**

最强的 DRRN 平均分也只有 **0.17**。  
这说明即便是小学科学层面的 grounded 实验推理，当前文本 agent 也远未解决。

#### 信号 2：小交互模型 > 大离线模型
最有信息量的结果是：

- **DRRN（1.5M）**：0.17
- **BC / TDT（11B）**：约 0.08

这很关键，因为 11B 模型不仅更大，还带有科学 QA 预训练与专家轨迹演示。  
但在这个 benchmark 上，它们仍输给小得多的交互式 RL agent。

**结论**：这里的主瓶颈不是“参数里有没有知识”，而是：
- 状态表示是否适合交互
- 能否选出有效动作
- 能否通过环境反馈学会长程程序

#### 信号 3：容易任务与难任务差距极大，揭示短板在“程序组合”
一些相对简单的分类/拿取类任务还能有中等得分；  
但开放式实验任务表现非常差。

例如：
- **Find a non-living thing**：DRRN 可到 **0.56**
- **Changes of State (Any)**：DRRN 仅 **0.03**

这说明模型的主要问题不是单点概念识别，而是：
**一旦需要多步导航 + 容器操作 + 设备使用 + 状态观察的长链组合，性能就迅速坍塌。**

#### 信号 4：匿名任务并未被“内置知识检索”轻易解决
作者专门设置了命名/匿名对象配对任务来分离检索与实验。  
结果显示，模型通常在到达“可以调用内部知识”的阶段之前，就已经被导航和执行难度卡住了。

这意味着 benchmark 揭示出的不是单纯的知识缺失，而是**知识到行动之间的断层**。

### 1-2 个关键指标
- **Overall**：最佳 DRRN 平均分 **0.17**
- **Scaling signal**：**1.5M DRRN > 11B BC/TDT (0.08)**

### 局限性
- **Fails when**: 任务需要长动作链、开放式实验规划、持续状态监控时容易失败，尤其是物态变化、植物生长、授粉、遗传等任务。
- **Assumes**: 文本接口、部分可观测、100 步 episode、任务评分依赖 required/optional goals；多数基线在测试时依赖 valid-action detection aid；仿真规则仅达小学科学精度；离线模型依赖手写 oracle 的 canonical demonstrations。
- **Not designed for**: 细粒度 2D/3D 空间操作、真实实验安全规程、复杂并联/含阻抗电路、连续高保真物理世界。

### 还需要特别注意的复现约束
- Benchmark 本身开源，但大模型基线训练成本不低。
- 文中明确使用了 TPU pod / 多卡 GPU 进行大模型训练与推理。
- CALM 等模型因为计算开销大，部分实验 seed 数也更少。
- 因而这篇 paper 的主要价值更像是**诊断性 benchmark**，而不是“给出一个已解决方案”。

### 可复用组件
1. **可执行程序式评测**：把 action sequence 当作 procedural explanation，并由环境自动验证。
2. **命名/匿名配对任务**：拆分“答案检索”与“实验执行”。
3. **文本科学模拟器**：在低于 3D 成本的前提下，提供热学、电路、生物等跨领域交互测试床。
4. **目标驱动评分**：required goals + optional subgoals 的组合适合 RL 和细粒度诊断。

**一句话总结 So what**：  
这篇工作真正推动的，不是又一个“更难题库”，而是把科学推理评测从“会不会说”推进到“能不能做成”，因此更适合暴露 today’s agents 在 grounded reasoning 上的真实断点。

## Local PDF reference
![[paperPDFs/Agentic_Reasoning_Benchmarks_Applications_of_Agentic_Reasoning_Scientific_Discovery_Agents/EMNLP_2022/2022_ScienceWorld_Is_your_Agent_Smarter_than_a_5th_Grader.pdf]]