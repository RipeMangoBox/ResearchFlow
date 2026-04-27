---
title: "LLM+MAP: Bimanual Robot Task Planning using Large Language Models and Planning Domain Definition Language"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/robot-task-planning
  - task/bimanual-manipulation
  - pddl
  - multi-agent-planning
  - partial-order-planning
  - dataset/ServeWater
  - dataset/ServeFruit
  - dataset/StackBlock
  - opensource/full
core_operator: "先将双臂任务与场景约束翻译成PDDL，再用多智能体部分序规划搜索可并行且逻辑可验证的左右臂计划"
primary_logic: |
  自然语言任务描述 + 视觉检测得到的物体位置/区域
  → LLM依据预定义双臂PDDL域生成初始状态与目标状态
  → FMAP搜索多智能体部分序计划（复杂时回退BFWS并后处理）
  → 输出可执行的双臂协同动作序列
claims:
  - "Claim 1: LLM+MAP在ServeWater与ServeFruit上达到100%执行成功率，在StackBlock-4/5上达到96%/97%，均高于GPT-4o、V3、R1与o1直接规划基线 [evidence: comparison]"
  - "Claim 2: 在ServeWater上，LLM+MAP平均规划时间为11.34s，明显低于o1的104.29s和R1的122.42s，同时成功率更高 [evidence: comparison]"
  - "Claim 3: 相比适配后的顺序规划基线LLM+P，LLM+MAP在100次成功任务中持续降低规划步数，说明多智能体部分序规划更能挖掘双臂并行性 [evidence: ablation]"
related_work_position:
  extends: "LLM+P (Liu et al. 2023)"
  competes_with: "DAG-Plan (Gao et al. 2024); LABOR (Chu et al. 2024)"
  complementary_to: "SayPlan (Rana et al. 2024); AutoTAMP (Chen et al. 2024)"
evidence_strength: strong
pdf_ref: "paperPDFs/Other_Multimodal_Large_Language_Model_MLLM_based_related_Embodied_Learning_2025/arXiv_2025/2025_LLM_MAP_Bimanual_Robot_Task_Planning_using_Large_Language_Models_and_Planning_Domain_Definition_Language.pdf"
category: Embodied_AI
---

# LLM+MAP: Bimanual Robot Task Planning using Large Language Models and Planning Domain Definition Language

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2503.17309), [Code](https://github.com/Kchu/LLM-MAP)
> - **Summary**: 论文把双臂长时序规划从“让LLM直接写动作序列”改成“让LLM写PDDL问题、让多智能体规划器搜索部分序计划”，从而同时提升逻辑正确性与双臂并行协作效率。
> - **Key Performance**: ServeWater/ServeFruit 成功率均为 **100%**；ServeWater 平均规划时间 **11.34s**，显著低于 o1 的 **104.29s** 与 R1 的 **122.42s**。

> [!info] **Agent Summary**
> - **task_path**: 自然语言任务 + 场景区域识别 / 双臂长时序协作 -> 可执行的部分序双臂计划
> - **bottleneck**: 左右臂分配、重叠区同步与长时序逻辑一致性很难靠LLM直接生成稳定解决
> - **mechanism_delta**: 把LLM从自由文本规划器改成PDDL问题编写器，把并行协调交给多智能体部分序求解器
> - **evidence_signal**: 三个任务域对比实验 + 去掉MAP的消融都显示更高成功率和更少规划步数
> - **reusable_ops**: [左/右/重叠三区域抽象, 自然语言到PDDL转换]
> - **failure_modes**: [StackBlock中目标块顺序写错会直接导致失败, 搜索空间增大时FMAP耗时上升并可能回退BFWS]
> - **open_questions**: [如何做执行期在线重规划, 如何扩展到两臂以上的多智能体系统]

## Part I：问题与挑战

这篇论文解决的核心不是“机器人会不会抓、放、倒水”，而是：

**在已有技能原语的前提下，双臂机器人如何把一个长时序任务拆解、分配给左右手，并在共享工作空间里合法且高效地并行执行。**

### 1) 真正的难点在哪里
相较单臂，双臂任务多了两类硬约束：

- **空间耦合**：左右手有各自独占区域，也有可共同到达的重叠区域；
- **时间耦合**：有些动作可并行，有些必须异步配合，还有些必须同步完成。

所以双臂任务规划不是简单地“多一个执行器”，而是要同时解决：

1. 任务理解与分解；
2. 子任务分配给左手还是右手；
3. 哪些步骤可并行、哪些必须串行；
4. 在重叠区如何避免逻辑冲突。

### 2) 为什么直接用LLM不够
作者的判断很准确：**LLM直接出计划时，实际上把“场景建模 + 目标理解 + 长程推理 + 双臂调度”全压在一次自由文本生成里**。  
这会带来两个问题：

- **长时序推理不稳**：越长的任务越容易漏前提、错顺序；
- **幻觉与不可执行性**：即便语言上“像个计划”，也不保证满足机器人动作前提。

论文实验也证明了这一点：即便是 o1/R1 这种强推理模型，成功率和时间成本仍不理想；GPT-4o/V3 直接规划更是几乎不可用。

### 3) 为什么现在值得做
因为现在的LLM已经足够擅长两件事：

- 理解自然语言任务；
- 把结构化约束写成代码/符号表示。

这意味着可以把LLM从“负责整条计划搜索”的角色中解放出来，改成：

- **负责把任务和场景转成PDDL问题**；
- **让经典规划器负责搜索与约束满足**。

这正好击中了当前双臂规划的瓶颈：  
**LLM擅长语义翻译，规划器擅长组合搜索。**

### 4) 输入/输出接口与边界条件
- **输入**：自然语言任务描述 + 视觉模块给出的物体位置/区域信息。
- **输出**：NICOL双臂可执行的**部分序计划**（partial-order plan）。
- **边界条件**：
  - 工作空间被离散成 left / right / overlap 三类区域；
  - 依赖预定义的单臂/双臂技能原语；
  - 主要在静态仿真桌面任务中验证；
  - 重点是高层任务规划，不是低层控制学习。

---

## Part II：方法与洞察

作者的设计哲学可以概括为一句话：

**让LLM负责“写对问题”，让规划器负责“解对问题”。**

### 方法流程

#### 1. 场景先做“规划友好”的空间抽象
作者没有直接把连续几何丢给LLM，而是先把工作台划分为三类区域：

- **左侧独占区**
- **右侧独占区**
- **中间重叠协作区**

再用 OWLv2 检测目标物体，结合规则判断其处于哪个区域。

这个步骤的意义很大：  
它把复杂连续空间压缩成了**对任务分配最关键的符号状态**。  
双臂规划真正关心的不是毫米级坐标，而是：

- 这个物体谁能碰到？
- 两只手是否都能到？
- 这一步能否并行？

#### 2. 用PDDL显式写出双臂世界模型
作者定义了双臂场景的PDDL域，包括：

- 对象类型：hand / area / object / point
- 状态谓词：如 hand 是否 available、object 在哪个 area/point、hand 是否能 access 某区域等
- 动作原语：
  - 单臂：grasp, move_to, release, push, pour, move_above, place
  - 双臂：co_hold, co_move_to

关键点在于：**双臂被建模成两个agent：LEFT 和 RIGHT**。  
每只手有自己的私有控制谓词，联合动作则要求两只手同时满足可用条件。

这一步把“谁来做”“何时能一起做”从提示词里的隐性推理，变成了规划器能检查的显性约束。

#### 3. LLM不再直接出动作，而是写PDDL/UP问题
给定：

- 场景描述；
- 任务自然语言描述；
- 预定义双臂域知识；

LLM负责生成：

- 初始状态；
- 目标状态；
- 对应的 Unified Planning Python 表达。

如果代码报错，就把解释器错误信息回喂给LLM修复。  
因此，LLM承担的是**符号转写器**而不是**全局搜索器**。

#### 4. 用MAP求部分序计划，而不是总序列
作者使用 FMAP 做多智能体规划，直接得到**部分序计划**：

- 保留必要依赖；
- 不强行把所有动作串成单链；
- 天然允许并行执行。

若 FMAP 在复杂任务上超时，则回退到 BFWS 先求一个可行解，再用图工具后处理成部分序结构。

这说明作者很务实：  
核心目标不是“纯MAP教科书实现”，而是**稳定得到可执行、可并行的双臂计划**。

### 核心直觉

这篇论文真正调的“因果旋钮”是：

**把LLM的输出空间，从开放式动作序列，收缩为受域约束的符号问题描述。**

这带来三层变化：

1. **What changed**  
   从“LLM直接规划完整动作序列”  
   变成“LLM写PDDL问题，规划器搜索部分序计划”。

2. **Which bottleneck changed**  
   - 原来：LLM要同时承担世界建模、时空推理、分配与排序，信息瓶颈太重；
   - 现在：LLM只需把语义压缩到符号状态/目标，组合搜索交给规划器；
   - 同时，双臂协作被写成显式约束，而不是靠提示词“猜”。

3. **What capability changed**  
   - 计划逻辑一致性更高；
   - 双臂并行性不再靠模型“灵感”，而是靠搜索显式发现；
   - 对强推理LLM的依赖降低，用GPT-4o做转写器也能赢过o1/R1直接规划。

更直白地说：

> 这篇论文不是让LLM“更会想”，而是让系统**少靠LLM胡思乱想**，把最难的时空协调交回给符号搜索。

### 战略权衡

| 设计选择 | 改变的约束/信息瓶颈 | 收益 | 代价 |
|---|---|---|---|
| left/right/overlap 三区域抽象 | 把连续几何压成可规划的离散状态 | 显式建模可达性与协作区 | 粗粒度，难覆盖精细接触/轨迹约束 |
| LLM写PDDL而非直接写动作 | 缩小LLM输出空间，减少自由生成误差 | 更易校验，逻辑约束更强 | 若目标状态写错，仍会整条计划失败 |
| 多智能体部分序规划 | 不再强制串行，显式搜索并行性 | 更适合双臂任务分配与同步 | 搜索空间大时耗时明显上升 |
| FMAP超时回退BFWS+后处理 | 提高复杂任务可解性 | 工程上更稳 | 不是原生MAP最优流程，部分并行性依赖后处理恢复 |

---

## Part III：证据与局限

### 关键实验信号

#### 1. comparison：成功率提升是最强主证据
在四个设置上，LLM+MAP 的成功率分别为：

- **ServeWater: 100%**
- **ServeFruit: 100%**
- **StackBlock-4: 96%**
- **StackBlock-5: 97%**

对比直接规划基线：

- GPT-4o direct：2 / 13 / 2 / 0
- V3 direct：2 / 6 / 6 / 1
- R1 direct：67 / 63 / 94 / 77
- o1 direct：84 / 82 / 95 / 88

这说明能力跳跃不在“更强模型参数量”，而在**把规划问题重新拆分给合适模块**。

#### 2. comparison：时间上优于强推理模型，但不是“无代价”
在简单任务上，LLM+MAP明显快于 o1/R1：

- ServeWater：**11.34s** vs o1 **104.29s** vs R1 **122.42s**
- ServeFruit：**7.75s** vs o1 **77.68s** vs R1 **144.42s**

这很关键，因为它表明：

- 直接依赖 reasoning LLM 虽能提升一些成功率；
- 但“LLM转写 + 符号规划”在**效率/正确率比**上更优。

但要注意另一面：  
在 StackBlock 这类复杂任务上，MAP 搜索时间显著上升，说明**符号搜索成为新瓶颈**。

#### 3. comparison：Group Debits 说明它不只“能做对”，还常常“做得更短”
作者用 Group Debits 衡量相对最少步数差距。  
结果显示 LLM+MAP 在简单任务上大量分布在 **0 debit** 附近，说明它经常是成功方案里步数最短的那个。

这支持一个更细的结论：

- 它不只是纠正了LLM的幻觉；
- 它还更会利用双臂并行性。

#### 4. ablation：真正贡献来自 MAP，而不只是 PDDL
消融把 MAP 去掉，只保留顺序规划，相当于适配版 LLM+P。  
结果显示 LLM+MAP 对该顺序规划基线有持续的**规划步数减少**（PSRR 为正）。

这条证据非常重要，因为它说明：

- 提升不只是来自“有了PDDL所以更合法”；
- 还来自**多智能体部分序规划本身**，它确实学会了更好的左右手分工与并行执行。

### 局限性

- **Fails when**: 任务需要更细粒度的连续几何/接触约束而三区域PDDL抽象不足时；或者在复杂堆叠任务中，LLM把目标块顺序写错时。
- **Assumes**: 已有可靠的单臂/双臂动作原语；OWLv2检测与规则化区域识别足够准确；场景基本静态且仿真中不加入物理随机失败；依赖 GPT-4o 这类闭源API以及 FMAP/BFWS 求解器。
- **Not designed for**: 动态环境下的在线重规划、动作失败后的闭环恢复、低层技能学习、两臂以上的大规模多机器人协作。

### 可复用组件

这篇工作里最值得迁移的不是具体任务，而是下面几个“操作符”：

- **区域化场景抽象**：把连续桌面空间压成对分工最关键的符号区域；
- **双臂PDDL域建模**：用私有/公共谓词显式表达左右臂控制权和协作前提；
- **LLM→PDDL/UP转写器**：让LLM做“结构化问题编写”，而不是直接出计划；
- **部分序计划后处理**：在求解器受限时，仍尽量恢复并行执行结构。

**一句话总结 So what**：  
这篇论文的能力跃迁，不是让机器人学会了新技能，而是让它在已有技能库上，第一次更稳定地做到了**双臂长时序任务的正确分配、正确排序和有效并行**。

![[paperPDFs/Other_Multimodal_Large_Language_Model_MLLM_based_related_Embodied_Learning_2025/arXiv_2025/2025_LLM_MAP_Bimanual_Robot_Task_Planning_using_Large_Language_Models_and_Planning_Domain_Definition_Language.pdf]]