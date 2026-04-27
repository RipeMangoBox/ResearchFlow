---
title: "Retrieval Dexterity: Efficient Object Retrieval in Clutters with Dexterous Hand"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/object-retrieval
  - reinforcement-learning
  - reward-shaping
  - sim-to-real
  - dataset/HouseholdObjects
  - opensource/no
core_operator: 以目标像素可见度为核心信号，在大规模仿真堆叠场景中训练灵巧手学会拨开、搅动和戳开遮挡物，再蒸馏到真实机器人执行。
primary_logic: |
  目标物被埋在杂乱堆叠中、传统逐个抓取过慢且依赖每个遮挡物都可稳定抓取
  → 构造多样堆叠场景，并用距离/搅动/邻近清障/像素可见度奖励训练灵巧手策略，再将仿真专家蒸馏为只依赖真实可观测量的学生策略
  → 快速暴露目标物体表面，使其进入可后续抓取的状态
claims:
  - "Claim 1: On the authors’ cluttered-retrieval benchmark, the full method outperforms heuristic visual motion planning on seen targets (84.23% vs 25.31% retrieval success) and unseen large targets (62.25% vs 8.33%) while requiring fewer steps [evidence: comparison]"
  - "Claim 2: Potential-based reward shaping materially improves the learned policy; removing it drops seen-target retrieval success from 84.23% to 55.45% and increases retrieval steps from 105.26 to 149.45 [evidence: ablation]"
  - "Claim 3: A distilled student policy zero-shot transfers to a real dexterous robot, achieving 6-9 successes out of 10 across seven everyday objects and reducing average retrieval time by 51.2% vs VMP and 61.9% vs sequential grasp-pick [evidence: comparison]"
related_work_position:
  extends: "Mechanical Search (Danielczuk et al. 2019)"
  competes_with: "Visuomotor Mechanical Search (Kurenkov et al. 2020); Broadcasting Support Relations Recursively from Local Dynamics (Li et al. 2024)"
  complementary_to: "Tactile Exploration for Unknown Object Retrieval (Zhao et al. 2024); Segment Anything (Kirillov et al. 2023)"
evidence_strength: moderate
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Retrieval_Dexterity_Efficient_Object_Retrieval_in_Clutters_with_Dexterous_Hand.pdf
category: Embodied_AI
---

# Retrieval Dexterity: Efficient Object Retrieval in Clutters with Dexterous Hand

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2502.18423), [Project](https://ChangWinde.github.io/RetrDex)
> - **Summary**: 该工作把“逐个抓走遮挡物”的检索范式改成“让灵巧手直接学会拨开、搅动、探查遮挡物以暴露目标”，从而在杂乱堆叠中更快完成目标检索，并支持仿真到真实的零样本迁移。
> - **Key Performance**: 仿真中 seen / unseen-large 检索成功率为 84.23% / 62.25%（VMP 为 25.31% / 8.33%）；真实机平均检索时间较 VMP 降 51.2%，较顺序抓取降 61.9%。

> [!info] **Agent Summary**
> - **task_path**: 相机目标定位与机械臂本体状态 / 杂乱堆叠检索 -> 暴露目标物体表面以便后续抓取
> - **bottleneck**: 在高接触、强遮挡、高维灵巧手控制下，如何不依赖“每个遮挡物都能稳定抓取”而快速清障
> - **mechanism_delta**: 将优化目标从“移除遮挡物”改为“最大化目标可见度”，并用势函数整形奖励与仿真特权状态稳定RL，再蒸馏到真实可观测输入
> - **evidence_signal**: seen/unseen/clutter-generalization 对比 + reward ablation + 真实机器人零样本时间收益
> - **reusable_ops**: [pixel-visibility reward, privileged-to-partial-observation policy distillation]
> - **failure_modes**: [远离机械臂工作空间的角落目标, 大尺寸且近乎完全遮挡的密集堆叠]
> - **open_questions**: [如何去掉人工初始化mask依赖, 如何把“暴露目标”与后续抓取闭环联合优化]

## Part I：问题与挑战

这篇论文解决的核心不是“最后怎么抓”，而是抓取前更难的那一步：**在多物体堆叠里，如何快速把被埋住的目标暴露出来**。

### 真正的问题是什么
传统 clutter retrieval 往往默认一个流程：先识别遮挡物，再把它们一个个抓走，最后再抓目标。这个流程有三个根本问题：

1. **太慢**：每个遮挡物都要完成一次感知、规划、抓取、搬运。
2. **太脆弱**：它隐含假设“每个遮挡物都可被可靠抓取”，现实里这并不成立。
3. **太依赖精确感知**：遮挡严重时，周围物体的姿态估计和支撑关系分析很容易出错。

论文的判断很直接：  
**检索任务的关键约束其实不是“必须抓走所有遮挡物”，而是“尽快让目标露出足够表面”**。  
这使得非抓取动作——如 pushing、stirring、poking——变得比逐个抓取更符合任务本质。

### 为什么现在值得做
因为两个条件同时成熟了：

- **灵巧手**提供了比平行夹爪更多的接触配置，可以做“拨、扫、搅、戳”这类清障动作。
- **高并行仿真 + 强化学习**使得接触丰富、动作高维的策略可以靠大规模试错学出来，而不必完全依赖人工规则或示教。

### 输入/输出接口
论文把任务定义为“检索到可抓取状态”，而不是直接完成 pick-up。

- **仿真输入**：机械臂/手关节、本体状态、目标 bbox/面积/深度，以及若干特权状态（目标位姿、指尖位姿、近邻物体等）
- **真实输入**：机械臂关节位置 + 相机跟踪得到的目标平面位置 `(x, y)`
- **输出**：13 维关节目标动作（7-DoF arm + 6-DoF hand）

### 边界条件
这项工作有清晰边界：

- 单目标检索
- 场景是盒中堆叠，目标通常置于底部
- 成功标准是目标暴露率超过 95%
- 主要目标是**暴露目标**，不是完整抓取与搬运闭环

---

## Part II：方法与洞察

### 方法拆解

#### 1. 用“掉落生成”构造逼真的堆叠环境
作者没有手工摆 clutter，而是把 18 个 household objects 通过重力掉落到盒中，让目标位于底部，形成更自然的堆叠关系。再配合质量、位置、目标姿态、相机安装等随机化，提升策略泛化。

这个设计的价值在于：  
它让策略学到的不是某几个固定堆叠模板，而是**“在随机堆叠中搜索并清障”的行为规律**。

#### 2. 奖励直接围绕“目标暴露”设计
方法最关键的变化，是把学习目标从“抓取某个遮挡物”改成“让目标更可见”。

奖励由几类信号组成：

- **距离奖励**：鼓励手靠近目标可能所在区域
- **搅动奖励**：鼓励主动扰动 clutter，尤其适用于目标完全不可见时
- **邻近清障奖励**：鼓励把目标附近的物体清开
- **像素可见度奖励**：直接奖励目标在相机里的可见程度
- **惩罚项**：抑制过大动作、无效接触和对目标本体的不良扰动

此外，作者用了**potential-based reward shaping**来稳定学习。直观上，它不是盯着绝对奖励，而是奖励“状态有没有朝着更暴露目标的方向变好”。

#### 3. 把奖励测量从“受手遮挡的 noisy 读数”变成“周期性干净评估”
这是论文里很值得注意的工程机制。

训练时，手在盒子上方操作，会遮挡顶视相机。如果直接按每一步的当前画面算目标像素，reward 会很噪。作者的处理是：

- 设置一个 **suspending pose**
- 每 10 步把手短暂抬到这个 pose
- 在无遮挡条件下计算目标像素可见度
- 再回到原姿态继续执行

这相当于把“混杂了手部遮挡的观测噪声”从奖励中剥离出去，让 RL 学到的是**真实清障进度**，而不是相机偶然看见多少像素。

#### 4. 用仿真专家蒸馏解决真实世界观测缺失
训练时策略可以使用特权状态，但真实机器人拿不到这些量。作者的解决方式不是强行让训练和部署完全同观测，而是两阶段：

1. 在仿真里用信息更全的 RL 专家把技能学出来  
2. 收集成功轨迹并筛选，再用行为克隆训练一个学生策略

学生策略只吃真实可得输入：机械臂关节 + 目标位置。  
目标位置由 **SAM + Cutie + 相机标定**实时获得，学生策略用 transformer 建模时序，输出关节动作。

这一步本质上是在做：
**“先在全信息下把技能学会，再把技能压缩到部分观测策略里。”**

### 核心直觉

旧范式要求机器人回答：  
“哪一个遮挡物该先抓？能不能抓稳？抓完放哪里？”

这篇工作把问题改写成：  
“怎样让目标更快露出来？”

这个改写带来了三个因果变化：

1. **目标函数变了**  
   从“按物体单位顺序移除”变成“直接优化目标可见度”。

2. **信息瓶颈变了**  
   不再强依赖每个遮挡物都要被精确识别、估姿、可抓取；只需知道目标大致位置，并通过动作改变周围 clutter 分布。

3. **能力边界变了**  
   策略开始自然学出非抓取式的灵巧行为：拨开、搅动、戳开、扫动，而不是死守 grasp-first。

一句话概括：  
**作者引入的关键因果旋钮，是把“检索”监督从离散的抓取/移除逻辑，换成连续的可见度提升逻辑。**  
这让 RL 更容易在接触丰富环境中发现高效清障动作。

### 策略层面的 trade-off

| 设计选择 | 解决的瓶颈 | 带来的能力 | 代价/风险 |
|---|---|---|---|
| 目标像素可见度作为主目标 | 最终抓取前缺少直接任务监督 | 学会“先暴露再抓取”的清障策略 | 依赖目标分割/跟踪 |
| 搅动 + 邻近清障奖励 | 完全遮挡时探索过难 | 更容易出现拨、搅、戳等涌现行为 | 奖励对任务有定制性 |
| suspending pose 周期评估 | 手部自遮挡导致 reward 噪声大 | 训练更稳定、进度更可测 | 训练流程更复杂，部署时不直接使用 |
| 特权状态专家 + BC蒸馏 | 真实世界拿不到完整状态 | 支持零样本 sim-to-real | 性能受专家覆盖和蒸馏偏差限制 |
| 时序 transformer 学生策略 | 单帧观测不足以推断长期清障策略 | 利用历史信息做更平滑动作 | 依赖稳定视觉跟踪 |

---

## Part III：证据与局限

### 关键证据信号

#### 1. Comparison signal：仿真中对 heuristic baseline 的优势很大
最直接的信号来自与 VMP 的比较。

- seen objects：84.23% vs 25.31%
- unseen large objects：62.25% vs 8.33%
- 同时步数明显更少

这说明收益不只是“能做出来”，而是**在效率上真正超越了规则式搜索/刮擦**。  
尤其在大目标和未见目标上差距依然显著，说明策略学到的不是简单模板记忆。

#### 2. Ablation signal：reward shaping 是主要稳定器
去掉 reward shaping 后：

- seen success 从 84.23% 降到 55.45%
- retrieval steps 从 105.26 增到 149.45

说明性能提升不是单靠大模型或大仿真规模堆出来的，**奖励设计本身是关键因果因素**。

另外两点也很清楚：

- 去掉 `r_stir`，大物体检索掉得更厉害  
  → 说明“主动扰动 clutter”对大面积遮挡尤其关键
- 去掉 `r_clean`，小物体更受影响  
  → 说明小目标更需要“清理目标附近局部空间”

#### 3. Generalization signal：不是只会一种 clutter
作者专门测了三类泛化：

- L1：未见目标
- L2：未见 clutter 组合
- L3：更多 clutter 数量

总体趋势是：  
完整方法在三种泛化设置下都最稳，且 reward shaping 对泛化收益尤其明显。  
这支持一个更系统层面的结论：**该策略学到的是“围绕目标做清障”的操作原则，而非固定物体实例脚本。**

#### 4. Real-world signal：零样本迁移是成立的，但有明确边界
真实机器人上，7 个日常目标的成功数大致在 **6/10 到 9/10**；同时平均检索时间：

- 相比 VMP 降 **51.2%**
- 相比顺序 grasp-pick 降 **61.9%**

这说明 sim-to-real 不是只停留在 demo 级别，而是有可量化效率收益。  
但同时，位置实验也暴露了边界：**靠近盒子中心、靠近机械臂的一侧更容易成功，远角区域明显更难。**

### 为什么这些证据仍然只能算 moderate
虽然论文有：

- 仿真比较
- 奖励消融
- 多种泛化设置
- 真实机器人实验

但仍需保守看待证据强度，因为：

1. **评测主要基于自建对象集和自建堆叠场景**
2. **主要对手不是统一协议下的强学习基线**
3. **真实世界试验规模仍较小（每项 10 次）**
4. **最终任务并不是 end-to-end 抓取成功，而是暴露成功**

所以它是“很有说服力的 method paper”，但还不到 very strong。

### 局限性

- **Fails when**: 目标位于盒子远端角落、超出机械臂舒适工作空间；或大尺寸目标在近 100% 遮挡、密集堆叠下需要更长的清障链条。
- **Assumes**: 已知目标对象，并在真实部署时能够获得人工介入的初始 mask；依赖相机-机械臂标定、SAM+Cutie 跟踪、IsaacGym 大规模并行仿真以及经过校准的物理参数。
- **Not designed for**: 直接完成最终抓取/搬运闭环、无任何目标先验的开放词汇检索、动态多目标场景或非盒式开放环境。

### 复现与部署依赖
这篇工作有几项实际依赖不能忽略：

- 512 并行 IsaacGym 训练
- 真实机上的 hand-eye calibration
- 30Hz 目标跟踪链路
- 人工提供目标 mask 的初始化
- 灵巧手硬件本体和稳定低层控制

这些条件决定了它的结论更像是：  
**“在具备较强仿真与感知基础设施的前提下，灵巧手检索可以比顺序抓取更快。”**

### 可复用组件

- **drop-from-above clutter generator**：用于构造自然堆叠训练场景
- **pixel-visibility / exposure 奖励**：把“检索”从抓取逻辑改成暴露逻辑
- **periodic suspending evaluation**：降低自遮挡造成的奖励噪声
- **privileged-expert → partial-observation student distillation**：适合 sim-to-real 中“训练看得多、部署看得少”的任务

## Local PDF reference

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Retrieval_Dexterity_Efficient_Object_Retrieval_in_Clutters_with_Dexterous_Hand.pdf]]