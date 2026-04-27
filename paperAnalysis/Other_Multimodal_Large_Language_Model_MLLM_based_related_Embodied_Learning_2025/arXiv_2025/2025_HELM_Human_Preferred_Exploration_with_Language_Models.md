---
title: "HELM: Human-Preferred Exploration with Language Models"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/autonomous-exploration
  - graph-serialization
  - llm-planning
  - training-free
  - dataset/DungeonEnvironments
  - dataset/IndoorOfficeSimulation
  - opensource/no
core_operator: 将稀疏信息图、自然语言偏好与历史记忆序列化为结构化提示，由LLM在图上逐步选择下一探索节点
primary_logic: |
  机载传感器构建占据图与前沿 → 采样并稀疏化得到带效用的探索图
  图描述、用户偏好与中间记忆 → 任务提示器生成可供LLM推理的混合查询
  LLM输出下一目标节点 → 局部路径规划执行并闭环更新地图直到探索完成
claims:
  - "在100个未见过的dungeon仿真中，HELM平均探索距离为557.66m，与ARiADNE(557.35m)和TARE Local(552.44m)基本持平，并优于Nearest、NBVP与DARE [evidence: comparison]"
  - "在130m×100m Gazebo办公室仿真中，带额外区域偏好的HELM PRE旅行距离为913.1m，优于ARiADNE的1025.41m、TARE的1179.23m和DSVP的1485.18m [evidence: comparison]"
  - "改变自然语言偏好会改变机器人优先探索区域（如底部左侧 vs 底部右侧），且机器人会在满足该偏好后继续完成全图探索 [evidence: case-study]"
related_work_position:
  extends: "TARE (Cao et al. 2021)"
  competes_with: "TARE Local; ARiADNE"
  complementary_to: "N/A"
evidence_strength: moderate
pdf_ref: paperPDFs/Other_Multimodal_Large_Language_Model_MLLM_based_related_Embodied_Learning_2025/arXiv_2025/2025_HELM_Human_Preferred_Exploration_with_Language_Models.pdf
category: Embodied_AI
---

# HELM: Human-Preferred Exploration with Language Models

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2503.07006)
> - **Summary**: 该文把机器人当前探索状态压缩成带效用的稀疏图，再把图结构、历史记忆和人类自然语言偏好一起写成提示词，让LLM在不重训的情况下做偏好可控的长程探索决策。
> - **Key Performance**: 100个未见dungeon上平均探索距离557.66m（距最优15.4%，与TARE Local/ARiADNE基本持平）；130×100m Gazebo办公室中带偏好的HELM PRE为913.1m，优于ARiADNE的1025.41m和TARE的1179.23m。

> [!info] **Agent Summary**
> - **task_path**: 机载LiDAR/占据图 + 自然语言偏好 + 历史轨迹 -> 下一探索节点/闭环探索轨迹
> - **bottleneck**: 现有探索器把“偏好”固化在超参数或奖励里，部署时很难按人类临时意图改策略而不重调或重训
> - **mechanism_delta**: 先把地图变成稀疏信息图，再把图、偏好和中间记忆组织成结构化prompt，让LLM替代固定目标函数做图上选点
> - **evidence_signal**: 100个未见dungeon上的标准比较显示其核心探索效率与TARE Local/ARiADNE同级；大办公室仿真中加入偏好后拿到最短路径
> - **reusable_ops**: [sparse-information-graph, preference-conditioned-graph-prompt]
> - **failure_modes**: [ambiguous-prompt-steering, graph-pruning-drops-necessary-transition-nodes]
> - **open_questions**: [how-to-guarantee-safe-decodable-actions-from-LLM, how-to-scale-to-3D-semantic-and-multi-robot-exploration]

## Part I：问题与挑战

这篇文章真正要解决的，不是“让机器人会探索”本身，而是：

**如何让机器人在未知环境中，按人类临时给出的偏好去探索，同时又不明显牺牲探索效率。**

### 1) 任务接口
- **输入**：
  - 机载传感器构建的占据图/自由空间/前沿信息
  - 人类自然语言偏好，如“先探索左半边”“优先左下角”“尽量避免回头”
  - 中间记忆，如历史路径或当前任务状态
- **输出**：
  - 下一个探索节点（waypoint）
  - 由局部规划器执行的可行路径

### 2) 真正瓶颈
现有自主探索方法的难点不只在“覆盖未知区域”，还在**偏好注入方式**：

- **传统规划器**把偏好写进代价函数或超参数里，表达能力有限，运行时改偏好不自然。
- **DRL 探索器**可以学长期策略，但偏好仍被锁在训练分布里；新偏好通常意味着重新设计奖励甚至重训。
- **人工接管**虽然能体现人类意图，但会破坏自主性，也难以在长时任务中稳定复用。

所以，这里的真瓶颈是：

> **如何把“人类的软偏好”变成机器人可执行的长期规划约束，而且这种约束应当在部署时即可修改。**

### 3) 为什么现在值得做
LLM 的价值不在于替代底层运动控制，而在于提供一种新的**偏好接口**：

- 人类可以直接用自然语言表述探索优先级；
- LLM 擅长在多约束条件下做离散决策；
- 如果把机器人状态先转换成结构化、可推理的文本上下文，LLM 就可能承担高层选点角色。

### 4) 边界条件
这篇论文的适用边界比较明确：

- 单机器人探索；
- 以室内/地下类未知环境为主；
- 核心抽象是占据图与 frontier，不是语义理解任务；
- 决策粒度是**高层 waypoint 选择**，不是端到端连续控制；
- 实验主体是 2D 探索抽象，Gazebo 中使用 3D LiDAR + Octomap 作为感知实现。

---

## Part II：方法与洞察

### 方法骨架

HELM 的系统思路可以概括为：

**地图先离散化成“LLM 能看懂的图”，再把“图 + 偏好 + 历史”写成 prompt，让 LLM 负责高层选点，传统规划器负责执行。**

#### 1. 构建 robot belief：从占据图到稀疏信息图
作者先从当前自由空间中采样候选 viewpoint，建立碰撞自由图，再计算每个节点的“效用”：

- 效用本质上是该点能看到多少 frontier；
- 无信息节点（utility = 0）会被剪枝；
- 最终得到一个**稀疏信息图**，只保留对探索真正有用的候选点。

这一步很关键，因为它把连续、冗余、低层的几何状态压缩成更适合决策的图结构。

#### 2. Graph Describer：把图结构变成文本
HELM 不让 LLM 直接读栅格图，而是用模板把图写成文本：

- 当前机器人位置
- 每个节点坐标与节点属性
- 边连接关系
- 其他图状态描述

这一步相当于把机器人 belief 做成一个**统一的文本接口**。

#### 3. Task Prompts Questioner：把“图”变成“带偏好的任务问题”
接着，系统将：
- 图描述
- 人类偏好
- 中间记忆/历史信息
- 输出格式要求

拼接成一个面向任务的 query。于是原先“写在 reward/hyperparameter 里的偏好”，被替换为**prompt 中显式表达的约束**。

#### 4. LLM 做序列决策，局部规划器做闭环执行
LLM 根据当前 prompt 选择下一探索节点。机器人到达后更新地图，再重新构图、重新生成 prompt、再次决策，形成闭环。

系统分工是：

- **LLM**：高层、离散、偏好相关的选点决策
- **经典局部规划器**：路径连通、运动执行、安全落地

这是一种典型的 hybrid 设计。

### 核心直觉

**改了什么？**  
把“偏好编码在训练期/代价函数里”改成“偏好编码在测试时的 prompt 上下文里”。

**哪个瓶颈被改变了？**  
过去的瓶颈是：策略只能在预定义目标函数或训练分布内工作。  
现在的瓶颈被改成：只要偏好能被自然语言描述，LLM 就能在当前图状态上做条件化推理。

**能力为什么会变化？**
1. **信息瓶颈被重写**：  
   原始占据图太低层、太大、太不规则；稀疏信息图只保留“候选决策点 + 连接关系 + frontier效用”，显著降低了 LLM 理解负担。
2. **约束接口被重写**：  
   偏好不再是一个固定数值权重，而是上下文条件，可以组合、替换、临时添加。
3. **控制职责被隔离**：  
   LLM 不碰底层控制，只做高层选点；因此系统既保留了语言灵活性，也避免把连续控制完全交给生成模型。

一句话概括其因果链：

> **把地图压缩成图文本 + 把偏好改成上下文条件 → 降低状态表达难度并解除训练期偏好绑定 → 获得无需重训的自然语言可控探索能力。**

### 策略权衡

| 设计选择 | 带来的能力变化 | 代价 / 风险 |
|---|---|---|
| 稀疏信息图替代原始地图 | 降低 token 压力，突出可决策节点与 frontier 价值 | 剪枝可能丢掉细粒度几何信息或必要过渡点 |
| 自然语言偏好替代 reward/超参数调节 | 新偏好可部署时直接注入，无需重训 | prompt 表述歧义会直接影响策略稳定性 |
| LLM 只做高层 waypoint 决策 | 保留经典局部规划的安全性与执行性 | 全局性能仍受 LLM 输出一致性影响 |
| training-free 推理替代 learning-based 偏好适配 | 对 OOD 偏好更灵活 | 依赖基础 LLM 质量、延迟与解析鲁棒性 |
| 模板化图描述/任务描述 | 输出更结构化、可控 | 模板设计本身成为系统性能瓶颈 |

---

## Part III：证据与局限

### 关键证据

#### 证据 1：标准比较说明“加了 LLM 以后并没有明显拖垮探索效率”
在 100 个未见过的 dungeon 仿真里，HELM 的平均探索距离为 **557.66m**：

- 与 **ARiADNE 557.35m**、**TARE Local 552.44m** 基本同级；
- 优于 Nearest、NBVP 和 DARE；
- 与最优参考路径相比 gap 为 **15.4%**。

这个信号支持的不是“HELM绝对更强”，而是更重要的一点：

> **把高层决策换成 LLM，并没有显著损害核心探索能力。**

#### 证据 2：大场景 Gazebo 仿真说明“偏好不仅能跟随，还可能改善全局顺序”
在 130m × 100m 室内办公室仿真中：

- **HELM PRE**（额外加入“先探索左半边、优先左上/左下、避免返回”）达到 **913.1m**
- 优于 **ARiADNE 1025.41m**
- 优于 **TARE 1179.23m**

这说明偏好并不只是“让轨迹看起来符合人意”，它还可能通过**更好的探索顺序**带来真实效率收益。

#### 证据 3：偏好实验说明“prompt 改了，策略顺序也会改”
论文给出左下优先 vs 右下优先两个案例。  
在其他条件不变时，机器人会优先探索相应区域，之后再继续完成全图探索。

这支持 HELM 的核心卖点：

> **偏好是运行时可修改的，而不是训练前写死的。**

### 证据的不足
尽管比较结果有说服力，但证据链仍然偏保守：

- 没有对 **Graph Describer / Questioner / memory / LLM backbone** 做系统 ablation；
- 没有量化“偏好满足度”，主要靠路径可视化与案例展示；
- Gazebo 大场景结果很亮眼，但场景数和偏好类型仍有限。

所以这篇论文的证据强度更适合评为 **moderate**，而不是 strong。

### 局限性
- **Fails when**: 图规模过大导致 prompt 过长、偏好描述含糊或互相冲突、环境强动态变化、或零效用剪枝误删必须经过的过渡节点时，LLM 的选点可能不稳定或明显次优。
- **Assumes**: 有可靠的占据图/前沿提取、较强的预训练 LLM（文中使用 DeepSeek-V3）、以及能把高层 waypoint 安全执行出来的局部规划器；同时默认人类偏好可以用自然语言相对清晰地表达。
- **Not designed for**: 多机器人协同探索、原始视觉输入下的端到端控制、语义目标搜索、或需要形式化安全保证的硬实时任务。

### 资源与复现约束
- 论文未提供代码或项目链接，复现成本不低；
- 系统依赖较强的基础 LLM；
- 文中提到由 LLM-agent 选择路径算法并生成代码，这部分工程细节不足，影响可复现性判断；
- 0.2s 级重规划时间是在稀疏图 + 特定仿真设置下报告的，未证明在更大图、更复杂 prompt 下仍稳健。

### 可复用组件
1. **稀疏信息图构建**：把 frontier-based exploration 压缩成适合高层决策的图状态。
2. **图到文本的结构化序列化模板**：可迁移到其他 embodied graph decision task。
3. **LLM 高层选点 + 经典局部规划** 的混合范式：适合那些“偏好变化快，但低层控制不能交给生成模型”的机器人任务。

![[paperPDFs/Other_Multimodal_Large_Language_Model_MLLM_based_related_Embodied_Learning_2025/arXiv_2025/2025_HELM_Human_Preferred_Exploration_with_Language_Models.pdf]]