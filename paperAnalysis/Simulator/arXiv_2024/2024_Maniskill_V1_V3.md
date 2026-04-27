---
title: "ManiSkill: Generalizable Manipulation Skill Benchmark with Large-Scale Demonstrations"
venue: NeurIPS
year: 2021
tags:
  - Survey_Benchmark
  - task/robot-manipulation
  - reinforcement-learning
  - full-physics-simulation
  - point-cloud
  - dataset/ManiSkill
  - dataset/PartNet-Mobility
  - opensource/full
core_operator: 用全物理仿真、多样化关节物体、自视角3D观测与大规模RL生成演示，评测机器人操作在未见同类物体上的泛化能力
primary_logic: |
  评测物体级泛化操作能力 → 构建多任务多对象的全物理环境并提供point cloud/RGB-D与约3.6万条演示 → 按训练/测试对象拆分、以测试成功率分轨评测 → 揭示现有3D视觉与LfD方法在跨对象泛化上的能力边界
claims:
  - "ManiSkill包含162个关节物体、4类操作任务和约36,000条成功演示，可系统评测同类未见对象上的操作泛化 [evidence: analysis]"
  - "在OpenCabinetDrawer单环境上，BC + PointNet+Transformer随演示数量从10增加到1000，成功率从0.16提升到0.90，说明演示规模与结构化3D编码器显著影响可学性 [evidence: analysis]"
  - "在跨对象测试上，最佳BC + PointNet+Transformer仅达到Door 0.11、Drawer 0.12、PushChair 0.08、MoveBucket 0.08的成功率，表明现有方法远未解决物体级泛化操作 [evidence: analysis]"
related_work_position:
  extends: "SAPIEN (Xiang et al. 2020)"
  competes_with: "RLBench (James et al. 2020); DoorGym (Urakami et al. 2019)"
  complementary_to: "Point Transformer (Zhao et al. 2020); CQL (Kumar et al. 2020)"
evidence_strength: moderate
pdf_ref: paperPDFs/Simulator/arXiv_2024/2024_Maniskill_V1_V3.pdf
category: Survey_Benchmark
---

# ManiSkill: Generalizable Manipulation Skill Benchmark with Large-Scale Demonstrations

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2107.14483) · [Code](https://github.com/haosulab/ManiSkill)
> - **Summary**: 该工作提出一个面向机器人操作“物体级泛化”的全物理仿真基准，用多样化关节物体、机器人自视角3D观测和大规模RL生成演示，统一评测3D视觉、模仿学习与强化学习方法在未见同类物体上的操作能力。
> - **Key Performance**: 162个物体 / 4个任务 / 约36k成功演示；最佳跨对象测试成功率仅为 Door 11%、Drawer 12%、PushChair 8%、MoveBucket 8%

> [!info] **Agent Summary**
> - **task_path**: 机器人自视角 point cloud/RGB-D + 训练对象上的演示/交互 -> 未见同类对象的低层控制动作
> - **bottleneck**: 现有操作基准缺少真实类内形状变化、可解的全物理任务和统一的跨对象评测协议，导致“泛化能力”无法被稳定测量
> - **mechanism_delta**: 将 PartNet-Mobility 资产重建为可验证可解的全物理任务，并用共享奖励模板 + 分治式 SAC 生成大规模演示，再按训练/测试对象拆分做评测
> - **evidence_signal**: 单环境可学但跨对象几乎学不动，最佳测试成功率普遍不超过12%
> - **reusable_ops**: [对象级train/test split, 共享奖励模板实例化, 分治式RL演示采集]
> - **failure_modes**: [几何/拓扑变化大时泛化显著下降, 双臂欠驱动与自由物体任务成功率极低]
> - **open_questions**: [如何学到跨对象共享的3D操作表征, 如何利用失败轨迹或在线交互让offline RL超过BC]

## Part I：问题与挑战

先说明一点：给定 PDF 路径名写作“ManiSkill V1-V3”，但正文实际对应的是 **ManiSkill v1 的 benchmark 论文**（NeurIPS 2021 Datasets and Benchmarks）。以下分析严格以给定正文为准，不外推 v2/v3 的新增内容。

### 这篇论文真正要解决什么问题？

它不是在提出一个更强的单一操作算法，而是在解决一个更基础、但长期缺位的问题：

**我们到底该如何可靠地评测“机器人是否能对同类未见物体形成可泛化的操作技能”？**

对机器人操作而言，真正困难的不只是“把一个固定柜门打开”，而是：

- 训练时见过一批柜子、椅子、桶；
- 测试时换成**同类别但几何、拓扑、尺寸、摩擦都不同**的新物体；
- 仍然能从**局部、遮挡、机器人自视角的3D观测**中做出正确动作。

这就是作者说的 **object-level generalizable manipulation skill**。

### 现有 benchmark 的瓶颈在哪里？

作者指出，过去基准通常卡在四个地方：

1. **物理层不够真**
   - 很多环境只支持高层动作或抽象技能，难以研究真正的低层接触与动力学。

2. **物体类内变化不够大**
   - 像 RLBench、robosuite、Meta-World 任务多，但常缺少同类物体内部的真实多样性。
   - DoorGym 有门把手/门板的程序化变化，但离真实世界的拓扑复杂性仍有差距。

3. **感知设置不贴近真实机器人**
   - 固定 2D 相机不利于研究真实移动操作中的视角、遮挡和3D理解问题。
   - 机器人实际更常依赖机载 RGB-D / point cloud。

4. **不同研究社群难公平比较**
   - 视觉研究者更关心 perception + imitation；
   - RL 研究者需要交互；
   - 机器人研究者可能想加入规划或人工规则。
   - 如果没有分轨协议，比较就容易混淆假设。

### 输入 / 输出接口是什么？

**输入：**
- `pointcloud` 或 `rgbd`，来自机器人头部安装的三相机全景自视角；
- 也支持 `state`，但作者明确说这**不适合研究泛化**。

**输出：**
- 机器人控制器目标值，可是关节空间控制，也可操作空间控制。

### 边界条件是什么？

这个 benchmark 的边界很清晰：

- 只研究 **同类别内部** 的跨对象泛化；
- 只覆盖 **4 个短时程、物理接触密集** 的 manipulation skills；
- 评测指标是 **测试对象上的平均成功率**；
- 不是 long-horizon household planning；
- 也**没有** sim-to-real 结论。

四个任务分别对应不同运动约束：

- **OpenCabinetDoor**：转动关节，单臂
- **OpenCabinetDrawer**：滑动关节，单臂
- **PushChair**：平面欠驱动移动，双臂
- **MoveBucket**：无约束搬运且重心变化，双臂

所以，论文回答的是一个非常“基础设施型”的问题：  
**如果我们真想让操作模型学会泛化，先得有一个能把泛化难点真实暴露出来的 benchmark。**

---

## Part II：方法与洞察

### 设计总览

ManiSkill 的设计不是“堆更多任务”，而是把评测链路从头到尾搭完整：

#### 1）资产层：从真实 3D 资产出发，而不是只靠程序生成

- 基于 **PartNet-Mobility** 选择并处理关节物体；
- 总计 **162 个物体**，来自 3 个类别；
- 手工清洗错误标注、修正碰撞网格、做凸分解；
- 再通过仿真与策略学习验证每个对象**确实可解**。

这一步非常关键。作者认为，如果 benchmark 里有大量“本来就坏掉/卡住/可被 exploit”的物体，那么最终测到的不是算法泛化，而是环境缺陷。

#### 2）任务层：用不同运动约束覆盖不同技能难点

四个任务并不是随意挑的，而是覆盖了几种典型物理结构：

- revolute joint
- prismatic joint
- planar underactuated motion
- unconstrained object transport

这会导致不同方法在不同物理属性上暴露不同短板：  
例如开门更像局部几何与接触定位；推椅子则更像双臂协同与欠驱动系统控制。

#### 3）感知层：强制使用贴近真实机器人的 3D 自视角观测

- 三个机载相机，120° 分布；
- 可输出 RGB-D 或融合 point cloud；
- 提供任务相关 segmentation masks；
- 明确把 benchmark 面向 3D deep learning 社区打开。

这一步改变的是“感知问题的形式”：  
不再是固定外参、无遮挡的静态识别，而是**自视角、部分可见、被机器人本体遮挡**的操作感知。

#### 4）数据层：用 RL 分治式采集大规模 demonstrations

作者没有采用人工遥操作或纯 motion planning 大规模采集，而是：

- 先给每个任务设计一个**共享 dense reward template**；
- 用 **MPC/CEM** 快速验证奖励模板是否可行；
- 再对**每个环境单独训练一个 SAC agent**；
- 为每个训练对象收集 300 条成功轨迹；
- 最终得到约 **36,000 条成功轨迹**、约 **1.5M 帧**。

核心不是“RL 比人更好”，而是：  
**在这种任务难度和规模下，RL + divide-and-conquer 是最可扩展的演示采集方案。**

#### 5）协议层：分轨评测，避免不同假设混在一起

三条赛道：

- **No Interactions**：只能用给定 demonstrations
- **No External Annotations**：可在线微调，但不能新增外部标注
- **No Restrictions**：可加规划、人工规则、新数据等

这让 benchmark 能同时服务视觉、RL 和机器人研究者。

### 核心直觉

ManiSkill 真正改变的不是某个网络结构，而是**测量方式本身**。

#### what changed → which bottleneck changed → what capability changed

- **从“多任务但低变化”的基准**
  → 变成 **“少而关键的任务 + 大类内几何/拓扑变化 + 训练/测试对象拆分”**
  → 把“记住训练对象”与“泛化到未见对象”区分开来。

- **从“固定 2D 外部视角”**
  → 变成 **“机器人自视角的 point cloud / RGB-D”**
  → 把 benchmark 变成真正考 3D 感知与操作耦合的问题。

- **从“环境能不能解都不确定”**
  → 变成 **“逐对象清洗、验证可解、检查 exploit”**
  → 低分更可信地反映算法能力，而不是环境 bug。

- **从“演示难以大规模收集”**
  → 变成 **“共享奖励模板 + 分治式 SAC 自动采集”**
  → 给 imitation / offline RL 研究提供进入门槛更低的数据入口。

### 为什么这套设计有效？

因为它把 benchmark 中最常见的三个混淆因素剥离掉了：

1. **环境坏了 vs 方法不行**
   - 通过资产修复和逐对象 solvability 验证，尽量减少环境导致的假阴性。

2. **任务太难学 vs 泛化本身难**
   - 单环境实验先证明“能学”，再看多对象泛化失败，因果更清晰。

3. **不同研究范式假设不同**
   - 通过三条赛道，把“只能离线学”与“允许交互”分开。

### 战略权衡表

| 设计选择 | 改变了什么瓶颈 | 带来的能力 | 代价 / 权衡 |
|---|---|---|---|
| 真实资产清洗 + 手工重建碰撞体 | 减少坏资产与 exploit | 评测结果更可信 | 人工成本高，扩展新资产慢 |
| 机器人自视角 3D 观测 | 从固定2D识别转向操作感知 | 更贴近真实机器人，能测遮挡/局部可见 | 渲染更慢，训练更重 |
| 共享奖励模板 + 分治式 SAC 采演示 | 降低大规模 demo 采集难度 | 可扩展生成约36k成功轨迹 | 依赖 reward engineering 和算力 |
| 对象级 train/test split | 显式测量未见对象泛化 | 能区分记忆与泛化 | baseline 成绩会非常低 |
| 多赛道协议 | 分离视觉/RL/机器人不同假设 | 更公平的社区比较 | benchmark 解释更复杂 |

---

## Part III：证据与局限

### 关键证据信号

#### 信号 1：单环境里“能学起来”，但很吃演示规模和结构偏置
在单个 OpenCabinetDrawer 环境上：

- **BC + PointNet+Transformer**
  - 10 条 demo：**0.16**
  - 300 条 demo：**0.85**
  - 1000 条 demo：**0.90**

结论不是“BC 很强”，而是：

- 这个 benchmark 即使在单环境里也不轻松；
- 演示数量很重要；
- 有 object-part relation 偏置的 3D 编码器，比纯 PointNet 更有效。

#### 信号 2：一旦从单对象变成跨对象，性能立刻塌陷
作者最强基线是 **BC + PointNet+Transformer**。  
跨对象测试成功率只有：

- OpenCabinetDoor：**0.11**
- OpenCabinetDrawer：**0.12**
- PushChair：**0.08**
- MoveBucket：**0.08**

这说明 benchmark 的难点不在“学一个对象”，而在：

- 跨对象几何/拓扑变化；
- 局部可见和遮挡；
- 双臂协同与欠驱动物理。

这也是论文最有力的 “So what”：
**ManiSkill 确实把 prior benchmarks 没有充分暴露的泛化缺口测出来了。**

#### 信号 3：offline RL 在这里并没有天然赢过 BC
在作者实验里：

- BCQ、TD3+BC 都没有稳定超过 BC；
- 尤其在成功-only demonstrations 设置下，BC 已经能直接拟合有效动作；
- 高自由度控制和任务复杂性又使 Q-learning 更难。

这揭示了一个很实际的边界：
**在成功轨迹充足但失败轨迹稀缺时，offline RL 未必比简单 imitation 更占优。**

#### 信号 4：为什么必须用分治式 demo collection？
作者还做了一个很关键的验证：  
如果直接在多个不同 cabinet 上从零训练一个共享 SAC agent，成功率会随对象数快速崩掉：

- 1 个 cabinet：100%
- 5 个 cabinet：82%
- 10 个 cabinet：2%
- 20 个 cabinet：0%

这直接支撑了 benchmark 构建中的核心设计决策：  
**大规模演示采集必须分治，不能指望一个 early-stage generalist RL agent 先学会一切。**

### 1-2 个最值得记住的指标

- **benchmark 规模**：162 个物体、4 个任务、约 36k 成功轨迹
- **最强跨对象测试结果**：最佳基线成功率仅 **8%–12%**

### 局限性

- **Fails when**: 任务涉及更大的类内拓扑变化、严重遮挡、双臂欠驱动系统或自由物体稳定搬运时，现有基线会明显失效；尤其 PushChair 和 MoveBucket 基本处于“几乎没学会泛化”的状态。
- **Assumes**: 依赖手工清洗过的 PartNet-Mobility 资产、任务级 dense reward 模板、分割 mask、SAPIEN 全物理仿真，以及相当可观的采集算力（论文报告整个 demo 训练/生成约用 4 台 8-GPU + 64-CPU 机器跑约 2 天）。
- **Not designed for**: 不面向 sim-to-real 结论、长时程多步家务规划、语言条件操作，也不是用来系统比较“人类遥操作演示 vs RL演示”多样性的通用 LfD benchmark。

### 可复用组件

- **对象级 train/test split**：适合任何想测 category-level generalization 的操作任务
- **共享奖励模板 → 实例化到每个对象**：适合大规模同类任务 reward 生成
- **分治式 RL 演示采集**：适合“单对象可解、跨对象难学”的 demo 数据构建
- **机载多相机 point cloud/RGB-D 接口**：适合把 3D 感知方法直接接入 manipulation benchmark
- **逐对象 solvability / exploit 检查流程**：对 benchmark 可靠性很关键

## Local PDF reference

![[paperPDFs/Simulator/arXiv_2024/2024_Maniskill_V1_V3.pdf]]