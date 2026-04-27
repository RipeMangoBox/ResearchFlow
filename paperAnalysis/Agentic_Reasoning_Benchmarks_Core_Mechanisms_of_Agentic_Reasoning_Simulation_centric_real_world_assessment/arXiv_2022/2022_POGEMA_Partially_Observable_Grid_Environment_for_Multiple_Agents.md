---
title: "POGEMA: Partially Observable Grid Environment for Multiple Agents"
venue: arXiv
year: 2022
tags:
  - Survey_Benchmark
  - task/multi-agent-path-finding
  - partial-observability
  - procedural-generation
  - grid-world
  - dataset/POGEMA
  - opensource/full
core_operator: 通过程序生成的局部可观测栅格环境、难度梯度与统一接口，把 PO-MAPF 标准化为可扩展的基准评测
primary_logic: |
  去中心化 PO-MAPF 评测目标 → 设计程序生成地图、局部观测与难度分级 →
  用 ISR/CSR 和 Gym/PettingZoo 接口统一训练/评测 → 揭示搜索、冲突消解与协作能力边界
claims:
  - "在文中 80 个智能体的基准设置下，POGEMA 吞吐量达到 83,000 FPS，而 Flatland 为 156 FPS [evidence: comparison]"
  - "给去中心化 A* 加入贪心补救与防振荡机制后，Pogema-16x16-extra-hard-v0 的成功率从 10% 提升到 84% [evidence: comparison]"
  - "在 8×8 extra-hard 基准上，QMIX/VDN 的学习曲线明显优于 IQL，说明高拥挤 PO-MAPF 需要显式协作而非独立导航 [evidence: analysis]"
related_work_position:
  extends: "Multi-Agent Pathfinding: Definitions, Variants, and Benchmarks (Stern et al. 2019)"
  competes_with: "Flatland-rl (Mohanty et al. 2020); MAgent (Zheng et al. 2017)"
  complementary_to: "QMIX (Rashid et al. 2018); A* (Hart et al. 1968)"
evidence_strength: moderate
pdf_ref: paperPDFs/Agentic_Reasoning_Benchmarks_Core_Mechanisms_of_Agentic_Reasoning_Simulation_centric_real_world_assessment/arXiv_2022/2022_POGEMA_Partially_Observable_Grid_Environment_for_Multiple_Agents.pdf
category: Survey_Benchmark
---

# POGEMA: Partially Observable Grid Environment for Multiple Agents

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2206.10944), [Code](https://github.com/AIRI-Institute/pogema)
> - **Summary**: POGEMA 把“无通信、局部观测”的多智能体路径规划统一成可程序生成的栅格基准，使规划方法、MARL 方法及其混合范式能在同一协议下被快速、可复现地比较。
> - **Key Performance**: 80 agents 下 83,000 FPS；A*+GA+FL 在 16×16 extra-hard 上达到 84% 成功率，而 32×32 extra-hard 仅 22%，体现出明确难度梯度。

> [!info] **Agent Summary**
> - **task_path**: 局部栅格观测 / 无通信 PO-MAPF -> 每个智能体的离散动作策略与统一基准评测
> - **bottleneck**: 现有 MAPF/MARL 环境很难同时满足部分可观测、去中心化、可扩展和可重复比较，导致规划与学习方法难以在同一设定下诊断
> - **mechanism_delta**: 用程序生成的 grid-world 固定观测约束和拥挤难度，并以 ISR/CSR + Gym/PettingZoo 接口统一训练、测试与复现
> - **evidence_signal**: 吞吐量对比 + 规划基线增强实验 + CTDE 与去中心化 RL 基线的性能分化
> - **reusable_ops**: [procedural map generation, egocentric observation patch, built-in difficulty ladder]
> - **failure_modes**: [large dense maps collapse simple replanning, CTDE state representation does not transfer across agent/map sizes]
> - **open_questions**: [how to benchmark communication-enabled PO-MAPF, how to evaluate cross-scale transfer beyond built-in map distributions]

## Part I：问题与挑战

这篇文章真正解决的不是“经典 MAPF 还能不能找到更短路径”，而是更现实的 **PO-MAPF**：没有中心控制器、通信受限、每个智能体只能看到局部环境时，如何让不同算法在同一设定下可比。

### 真正的瓶颈
- **信息瓶颈**：经典 MAPF 通常默认全局可观测、中心化规划；但灾害区域、矿井、核电巡检等场景里，这个假设不成立。
- **评测瓶颈**：现有环境要么过于任务特化（如 Flatland），要么不面向路径规划比较（如 MAgent），很难同时支持：
  - 部分可观测；
  - 多智能体协作冲突；
  - 规划法与 RL 法并行评测；
  - 大规模快速训练。

### 输入 / 输出接口
- **输入**：每个 agent 的局部 ego-centric 栅格 patch，大小为 \((2R+1)\times(2R+1)\)，编码成 3 个矩阵：
  - 障碍物；
  - 其他智能体位置；
  - 自己目标的投影。
- **输出**：一步离散动作，通常为 `{上, 下, 左, 右, 等待}`。
- **环境执行规则**：
  - 撞障碍或撞人的动作无效，agent 留在原地；
  - 到达目标后 agent 从环境中消失（disappear-at-target）；
  - 所有人到达，或达到步数上限 \(K\) 时结束。

### 边界条件
- 默认 **无通信**；
- 观测中**不包含其他 agent 的目标、计划或历史轨迹**；
- 任务是 **4-连通栅格导航**，不是连续控制；
- 对 CTDE 方法虽可提供全局 state，但该 state 维度依赖地图尺寸和 agent 数量，跨配置泛化不自然。

## Part II：方法与洞察

POGEMA 的核心不是新求解器，而是把 PO-MAPF 做成一个 **可控、可扩展、可复现的 benchmark substrate**。

### 基准是怎么搭起来的
1. **环境抽象**
   - 4-连通 grid；
   - 地图可程序生成，也可自定义；
   - 障碍密度默认约 30%，作者认为这是较难区域。

2. **难度旋钮**
   - 地图规模：8×8、16×16、32×32、64×64；
   - 难度等级：easy / normal / hard / extra-hard；
   - 主要通过 **agent 密度** 调节协作拥堵程度；
   - 内置配置统一 observation radius=5。

3. **统一指标**
   - **ISR**：单个 agent 是否到达目标；
   - **CSR**：是否所有 agent 都到达目标。
   这使得“单体能不能到”和“团队能不能共同完成”被拆开评估。

4. **统一接口**
   - 支持 Gym、PettingZoo；
   - 可直接接 SampleFactory / APPO；
   - 同时便于插入搜索式 planner 和 MARL policy。

### 核心直觉

POGEMA 改变的不是“地图长什么样”，而是 **问题的统计结构**：

- **从全局已知的离线规划**，变成 **局部观测下的在线决策**；
- **从固定实例记忆**，变成 **程序生成分布上的泛化**；
- **从纯路径搜索**，变成 **路径搜索 + 冲突消解 + 协作行为** 的组合问题。

更具体地说：

- **把观测半径固定下来**，就固定了信息瓶颈；
- **把障碍密度固定在较难区间**，就让导航复杂度可控；
- **把 agent 密度逐级提高**，就系统性放大协作/拥堵难度。

于是，benchmark 不只是给一个总分，而是能回答：
- 什么时候局部重规划就够了；
- 什么时候需要显式协作；
- 什么时候搜索和学习都开始失效。

### 为什么这套设计有效
因为它把 PO-MAPF 的几个关键因素拆成了相对独立的控制杆：
- **可见性**：观测半径；
- **地形复杂度**：障碍布局；
- **交互负载**：agent 密度；
- **团队完成度**：CSR 相对 ISR 的差距。

这让 benchmark 更像“诊断仪”，而不只是“排行榜”。

### 策略性权衡

| 设计选择 | 带来的好处 | 代价 / 偏置 |
|---|---|---|
| 仅局部观测、默认无通信 | 真实暴露 PO-MAPF 的信息受限本质 | 不覆盖通信式协同算法的完整能力 |
| 程序生成地图 | 避免只记住固定地图，强调泛化 | 难与某些手工最优实例做逐一对照 |
| 以 agent 密度定义难度 | 清楚放大冲突消解与协作瓶颈 | 难度仍会受随机地图拓扑影响 |
| ISR + CSR 双指标 | 区分个体导航能力与团队协作能力 | 不直接衡量路径最优成本或 makespan |
| 提供全局 state 给 CTDE | 方便接入 MARL 主流算法 | state 维度绑定配置，跨规模迁移受限 |

## Part III：证据与局限

### 关键证据信号

- **信号 1｜系统对比**
  - 在作者报告的 80-agent 基准下，POGEMA 达到 **83,000 FPS**，而 Flatland 为 **156 FPS**。
  - 结论：它足够快，适合大规模 RL 训练与重复评测；同时保留了部分可观测和程序生成这两个关键属性。

- **信号 2｜规划基线比较**
  - 朴素去中心化 A* 在高密度场景下很快失效，但加入贪心 fallback 和防振荡后，性能显著提升。
  - 例如 **16×16 extra-hard 从 10% 提升到 84%**；但 **32×32 extra-hard 仍只有 22%**。
  - 结论：该 benchmark 能区分“局部搜索足够”与“纯搜索难以扩展”的边界。

- **信号 3｜学习基线诊断**
  - 在 8×8 extra-hard 上，**QMIX/VDN 优于 IQL**；
  - APPO 在地图更大、密度更高时性能明显下降。
  - 结论：POGEMA 暴露的主要难点不是单体导航，而是 **协作冲突消解与规模扩展**。

### 1-2 个最值得记住的指标
- **吞吐量**：83,000 FPS（80 agents）。
- **难度分层**：A*+增强在 16×16 extra-hard 可到 84%，但在 32×32 extra-hard 仅 22%。

### 局限性
- **Fails when**: 任务需要连续控制、非栅格拓扑、动态图障碍、复杂传感噪声或显式通信协议时，POGEMA 的默认 grid abstraction 不再足够忠实。
- **Assumes**: 4-连通栅格、固定局部观测半径、无通信、稀疏奖励、disappear-at-target；若采用 CTDE，还假设训练/测试配置在 state 维度上基本一致。
- **Not designed for**: 全局最优 MAPF 代价证明、现实视觉感知评测、异构机器人动力学建模、真实世界多传感器融合。

### 资源与复现依赖
- 优点是 **开源** 且接口标准；
- RL 大规模训练明显依赖现成框架栈（如 SampleFactory/APPO）；
- 作者展示了“单 GPU、数小时”级别训练可行性，但这更多说明 **环境高效**，不代表算法本身在现实多机器人系统中就能直接部署。

### 可复用组件
- 程序生成地图器；
- ego-centric 三通道观测编码；
- easy→extra-hard 难度阶梯；
- ISR/CSR 评测协议；
- Gym / PettingZoo 适配层；
- 自定义地图 / YAML 配置接口。

## Local PDF reference

![[paperPDFs/Agentic_Reasoning_Benchmarks_Core_Mechanisms_of_Agentic_Reasoning_Simulation_centric_real_world_assessment/arXiv_2022/2022_POGEMA_Partially_Observable_Grid_Environment_for_Multiple_Agents.pdf]]