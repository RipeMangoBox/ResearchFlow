---
title: "RoBridge: A Hierarchical Architecture Bridging Cognition and Execution for General Robotic Manipulation"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/robotic-manipulation
  - reinforcement-learning
  - symbolic-bridge
  - hierarchical-planning
  - dataset/MetaWorld
  - dataset/Robosuite
  - opensource/no
core_operator: "用VLM生成具物理直觉且环境不变的IOR，作为高层认知与RL低层控制之间的符号桥梁。"
primary_logic: |
  自然语言指令 + 第一/第三视角观测 → HCP将任务拆解为原子动作，并借助GPT-4o、SAM、GroundingDINO生成IOR（动作类型、对象/目标mask、第一视角masked depth、方向约束） → GEA在闭环跟踪与状态判定下输出低层控制动作，逐步完成通用机械臂操作
claims:
  - "Claim 1: RoBridge在MetaWorld的MT50及四种未见扰动设置上取得82.12%的平均成功率，较最强基线ManipGen高11.28个百分点 [evidence: comparison]"
  - "Claim 2: 在每个任务仅使用5个真实样本微调的条件下，RoBridge在四类真实世界操作任务上达到83.3%的平均成功率，显著高于ReKep的49.2%与π0的18.3% [evidence: comparison]"
  - "Claim 3: 将IOR替换为language-only、keypoints或DINOv2特征，或移除DAgger/域随机化，都会显著降低泛化表现，说明中间表示与训练策略都是关键因子 [evidence: ablation]"
related_work_position:
  extends: "ReKep (Huang et al. 2024)"
  competes_with: "ReKep (Huang et al. 2024); ManipGen (Dalal et al. 2024)"
  complementary_to: "N/A"
evidence_strength: strong
pdf_ref: paperPDFs/Other_Multimodal_Large_Language_Model_MLLM_based_related_Embodied_Learning_2025/arXiv_2025/2025_RoBridge_A_Hierarchical_Architecture_Bridging_Cognition_and_Execution_for_General_Robotic_Manipulation.pdf
category: Embodied_AI
---

# RoBridge: A Hierarchical Architecture Bridging Cognition and Execution for General Robotic Manipulation

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2505.01709), [Project](https://abliao.github.io/RoBridge/)
> - **Summary**: 该工作提出一个“三层桥接”机器人架构，用具物理直觉且具环境不变性的中间表示 IOR，把 VLM 的任务理解与 RL 的稳健执行解耦连接起来，从而在开放操作场景中同时保住“会理解”和“会动手”。
> - **Key Performance**: MetaWorld 多扰动均值 **82.12%**；真实世界 4 类任务均值 **83.3%**。

> [!info] **Agent Summary**
> - **task_path**: 自然语言指令 + 第一/第三视角视觉观测 -> 多步机械臂低层控制动作
> - **bottleneck**: VLM能做开放域理解但缺乏物理执行直觉，RL能学到鲁棒技能却缺少通用语义接口，二者之间缺少稳定且可执行的中间表示
> - **mechanism_delta**: 把高层输出从自由文本/轨迹改成IOR，使HCP只决定“做什么”，GEA只学习“怎么做”
> - **evidence_signal**: 跨MetaWorld扰动、真实世界、零样本新任务均领先，且language-only/keypoint/DINOv2/去DAgger消融均明显退化
> - **reusable_ops**: [VLM原子动作分解, mask-depth约束式中间接口]
> - **failure_modes**: [遮挡或重叠导致mask丢失, 对软体/微小/复杂形状物体易出现执行偏移]
> - **open_questions**: [IOR能否扩展到可变形物体与高精度装配, 去除GPT-4o等闭源依赖后性能能否保持]

## Part I：问题与挑战

这篇论文要解决的不是“怎么让机器人多学一点技能”这么简单，而是更根本的接口问题：**高层认知和低层执行之间，到底该用什么形式通信**。

### 1. 真正的问题是什么
在开放环境下的通用机械臂操作，输入通常是：
- 自然语言指令
- 第一视角/第三视角视觉观测
- 夹爪状态等机器人本体信息

输出则是：
- 一串可执行的低层连续控制动作

难点不在某个单点模块，而在于两种能力天然错位：

1. **Procedural skill dilemma（程序性技能困境）**  
   端到端 imitation/VLA/RL 方法能直接出动作，但一旦光照、背景、视角、相机位姿变化，性能容易崩。它们往往把太多视觉细节和执行细节绑死在一起。

2. **Declarative skill dilemma（陈述性技能困境）**  
   VLM/LLM 能理解开放指令、拆任务、说清对象关系，但它们没有真实的身体交互经验。直接让它输出关键点、轨迹或动作公式时，经常出现“语义对了、物理错了”的规划。

### 2. 真正瓶颈在哪里
**真正瓶颈是：缺少一个同时满足“语义可解释 + 物理可执行 + 对环境变化稳定”的中间接口。**

过去方法通常走两条路：
- 要么让大模型直接下沉到低层控制，结果削弱认知能力；
- 要么让低层策略直接吃原始图像和任务标签，结果泛化受限。

RoBridge 的出发点是：  
**不要让 VLM 直接负责怎么动，也不要让 RL 自己去理解开放世界。先把“认知”压缩成一个操作友好的、环境不变的符号接口，再交给执行器。**

### 3. 为什么现在值得解决
因为时机到了：
- VLM 已经足够强，能可靠做任务分解、对象选择、约束推断；
- GroundingDINO / SAM / Track-Anything 这类基础感知 API 已能把语义对象落到视觉实体；
- RL/局部策略在受控技能执行上已经很稳，但缺一个高层入口。

所以现在的关键不是继续堆更大的单体模型，而是**把认知与执行的分工重新设计清楚**。

### 4. 输入/输出边界条件
RoBridge 的适用边界也很明确：
- 任务被拆成有限原子动作集合：reach / grasp / place / press / push / pull / open / close / turn
- 当前主要面向刚体、形状相对简单的物体
- 依赖第一视角深度 + 第三视角分割信息
- 真实部署仍需少量任务内示例微调（文中为每任务 5 个真实样本）

---

## Part II：方法与洞察

RoBridge 是一个三层系统：

- **HCP（High-level Cognitive Planner）**：负责理解与规划
- **IOR（Invariant Operable Representation）**：负责桥接
- **GEA（Guided Embodied Agent）**：负责执行

可以把它理解成：
- HCP 是“脑”
- IOR 是“神经信号”
- GEA 是“身体”

### 1. HCP：只负责“想清楚”
HCP 基于 GPT-4o，并结合：
- GroundingDINO
- SAM
- Track-Anything

它先把复杂任务拆成原子动作序列。比如：
- reach yellow cylinder
- grasp yellow cylinder
- reach round slot
- place yellow cylinder

这里的关键是：**HCP 不直接输出机械臂轨迹**，只输出任务级操作意图与相关对象。

### 2. IOR：把“任务意图”变成“可执行符号”
对每个原子动作，RoBridge 构造 IOR，包含四类信息：

- **动作类型** `T`
- **第三视角 mask** `M`  
  包括夹爪、被操作物体、目标区域
- **第一视角 masked depth** `D`
- **约束** `C`  
  如末端方向、运动方向等

也就是：
- 语义对象是谁
- 当前几何关系如何
- 应该朝哪个方向完成交互

这一步是全篇最核心的设计。它不是纯语言，也不是原始图像，更不是具体轨迹，而是**“与任务强相关、与背景弱相关”的操作接口**。

### 3. GEA：只负责“做稳”
GEA 的输入不是原始 RGB，而是 IOR。  
它学习的是从 IOR 到低层动作的映射。

具体上：
- `reach` 这类更接近明确几何目标的动作，用 motion planning
- 其他涉及交互与接触的动作，由策略网络执行

训练过程分三段：
1. **RL expert per task**：先为每类训练任务学出强专家
2. **IL distillation**：用专家数据训练共享 GEA
3. **Adaptive offline DAgger**：对失败样本回灌修正，持续聚合技能

### 4. 闭环控制：慢脑 + 快手
RoBridge 不是“一次规划到底”，而是双频闭环：

- **高频更新**：Track-Anything 持续更新 mask / depth
- **低频判断**：GPT-4o + 夹爪状态判断当前原子动作是 success / wrong / normal

如果成功，就进入下一个原子动作；  
如果失败，就重新生成对应 IOR。

这让系统不仅能执行，还能在一定程度上**发现偏差并重试**。

### 核心直觉

RoBridge 真正改变的不是某个 backbone，而是**高层到低层的因果接口**。

#### 改了什么
从过去的两种接口：
- `语言/图像 -> 直接低层动作`
- `语言 -> 关键点/约束 -> 规划`

改成：
- `语言 + 观测 -> 原子动作 -> IOR -> 低层动作`

#### 哪个瓶颈被改变了
它改变了两个信息瓶颈：

1. **对 HCP 来说**  
   不再要求大模型负责连续控制，只需负责对象选择、任务分解和方向约束。  
   于是大模型的错误空间从“整条轨迹是否物理合理”缩小为“对象和关系是否选对”。

2. **对 GEA 来说**  
   不再让策略直接面对背景、光照、材质、相机扰动等大规模外观变化，而只面对任务相关的 mask-depth-constraint 输入。  
   于是输入分布更稳定，更适合学出可迁移技能。

#### 为什么有效
因为 IOR 保留了真正决定操作成功的那部分信息：
- 谁是操作对象
- 目标在哪里
- 当前局部几何关系是什么
- 需要满足什么方向/姿态约束

同时去掉了大量对执行无关的视觉细节。  
这使得：
- VLM 的优势留在“理解和拆解”
- RL 的优势留在“接触和执行”
- 两者不再互相拖累

#### 能力上带来了什么变化
直接结果就是三点：
- **跨环境更稳**：对背景、光照、颜色、相机变化更不敏感
- **长时序更稳**：原子动作级闭环降低误差累积
- **零样本组合更强**：能把已有原子技能重组成新任务

### 战略权衡

| 设计选择 | 解决的瓶颈 | 带来的收益 | 代价/权衡 |
|---|---|---|---|
| HCP 只做任务分解与对象/约束选择 | 避免让VLM直接负责物理执行 | 保留开放域理解能力 | 依赖闭源VLM与感知API |
| IOR 用 mask + masked depth + constraint | 降低外观扰动与视角偏移影响 | 泛化更稳，接口更统一 | mask 一旦跟丢会连锁影响执行 |
| GEA 基于专家蒸馏 + DAgger | 纯 imitation 容易误差累积 | 执行更鲁棒，可持续修正 | 需要为训练任务先学专家策略 |
| 高频跟踪 + 低频状态判定 | 动态环境下信息会过时 | 长时序任务更稳定 | 系统复杂度与延迟增加 |
| `reach` 用规划，其他动作用学习 | 把确定性几何与接触性交互分开 | 简化学习难度 | 整体系统不是单一端到端模型 |

---

## Part III：证据与局限

### 1. 关键证据信号

#### 信号 A：跨扰动仿真比较
在 MetaWorld 的 MT50 与未见背景/光照/颜色/相机位姿扰动上，RoBridge 平均成功率达到 **82.12%**，优于最强基线 ManipGen 的 **70.84%**。

这说明它不是只在单一视觉条件下有效，而是真的对分布变化更稳。  
尤其是相机位姿变化这一项的提升，正好支持了作者关于 **IOR 具有更强不变性** 的核心论点。

#### 信号 B：真实世界执行比较
在真实世界四类任务上，且每任务只给 **5 个真实样本** 微调，RoBridge 达到 **83.3%** 的平均成功率，明显高于：
- ReKep：49.2%
- π0：18.3%

这表明它并不只是“仿真里漂亮”，而是桥接式设计确实帮助了 sim-to-real 和真实执行稳定性。

#### 信号 C：长时序任务
在“把不同形状积木放入对应槽位”的四阶段任务中：
- RoBridge 各阶段成功率为 100 / 80 / 70 / 50
- 平均完成长度为 **3.0**
- 明显高于 ReKep 的 **1.7**

这说明它的提升不只体现在单步 skill，而是在**任务分解 + 顺序执行 + 闭环切换**这条链路上都有增益。

#### 信号 D：零样本新任务
在 5 个训练中未见的新任务上，RoBridge 平均成功率 **75%**，而：
- ReKep：28%
- ManipGen：18%
- PSL：12%

这支持论文最关键的“so what”：
**RoBridge 学到的不是某个固定任务，而是一种可重组的‘认知-执行接口’。**

#### 信号 E：消融定位因果来源
最有说服力的不是“又做了个新系统”，而是消融清楚地告诉你提升从哪里来：

- `language-only`：均值降到 **39.28**
- `keypoints`：降到 **59.88**
- `DINOv2 features`：降到 **50.36**
- `w/o DAgger`：降到 **65.12**
- `w/o domain randomization`：降到 **56.08**

这说明：
1. **IOR 不是随便一个中间表示都行**
2. **GEA 的训练配方不是可有可无**

### 2. 局限性

- **Fails when**: 遮挡或重叠导致 mask 丢失时，IOR 会失真并连带影响执行；对软体、微小物体、复杂形状物体或高精度接触任务，执行误差更容易积累。
- **Assumes**: 依赖闭源 GPT-4o 做规划与状态判断；依赖 GroundingDINO、SAM、Track-Anything、双视角相机与深度输入；依赖每个训练任务先有 RL 专家策略，并在仿真中做大量域随机化；真实落地仍需每任务 5 个演示微调。
- **Not designed for**: 无分割/无深度输入的纯端到端控制；可变形物体操作；复杂双臂协作；超高精度装配或超快动态操作。

### 3. 复现与部署依赖
这篇工作的工程依赖不轻，需要明确看到：

- **训练成本**：A100 上专家训练约 25 GPU 小时，GEA 训练约 30 GPU 小时，真实数据微调约 1 GPU 小时
- **推理依赖**：除 GPT-4o 外约需 6GB GPU 显存
- **时延结构**：
  - GEA + Track-Anything：每帧 60–80ms
  - GPT-4o：每个原子动作约 0.3s
  - SAM + GroundingDINO：每个原子动作约 1s

所以它更像一个**分层系统方案**，不是一个可直接替换进任何机器人栈里的轻量单模型。

### 4. 可复用组件
如果只提炼最值得迁移的部分，我会选这三个：

1. **IOR 作为标准化中间接口**  
   很适合把“高层多模态理解”和“低层接触控制”解耦。

2. **Adaptive-sampling offline DAgger**  
   在多任务 skill aggregation 中，比简单 offline 更新更有针对性。

3. **高频跟踪 + 低频判定的双速闭环**  
   对长时序操作尤其有价值，能平衡实时性与高层判断成本。

![[paperPDFs/Other_Multimodal_Large_Language_Model_MLLM_based_related_Embodied_Learning_2025/arXiv_2025/2025_RoBridge_A_Hierarchical_Architecture_Bridging_Cognition_and_Execution_for_General_Robotic_Manipulation.pdf]]