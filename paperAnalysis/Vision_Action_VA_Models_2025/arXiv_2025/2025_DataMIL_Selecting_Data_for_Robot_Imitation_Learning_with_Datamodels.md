---
title: "DataMIL: Selecting Data for Robot Imitation Learning with Datamodels"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/robot-imitation-learning
  - datamodels
  - metagradients
  - influence-estimation
  - dataset/MetaWorld
  - dataset/LIBERO
  - dataset/Open-X-Embodiment
  - opensource/no
core_operator: "用目标任务验证损失替代真实rollout回报，训练datamodel为先验示范簇估计对最终策略成功率的影响分数，再筛选高正影响数据与目标数据共训。"
primary_logic: |
  少量目标任务演示 + 大规模异构先验示范库
  → 以 held-out 目标演示上的行为克隆验证损失作为可微代理目标，并用回归/元梯度 datamodel 估计每个轨迹或子轨迹簇的影响分数
  → 选出 top-x% 高正影响数据与目标数据共训练，得到更强的任务专用模仿策略
claims:
  - "在 MetaWorld 50 个任务的平均结果上，DataMIL 超过所有比较方法，并较最强启发式基线提升约 10 个百分点 [evidence: comparison]"
  - "用 held-out 目标演示上的行为克隆验证损失替代 rollout 成功率进行数据选择，仅带来小幅性能下降，而 metagradient 估计器训练约快 8× [evidence: ablation]"
  - "在 OXE 实世界设置上，DataMIL 的平均成功率为 61.0%，高于最佳启发式基线的 40.0% [evidence: comparison]"
related_work_position:
  extends: "Datamodels (Ilyas et al. 2022)"
  competes_with: "Behavior Retrieval (Du et al. 2023); FlowRetrieval (Lin et al. 2024)"
  complementary_to: "Octo (Ghosh et al. 2024); Sim-and-Real Co-training (Maddukuri et al. 2025)"
evidence_strength: strong
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_DataMIL_Selecting_Data_for_Robot_Imitation_Learning_with_Datamodels.pdf
category: Embodied_AI
---

# DataMIL: Selecting Data for Robot Imitation Learning with Datamodels

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2505.09603)
> - **Summary**: 这篇工作把 datamodels 引入机器人模仿学习，用目标任务上的离线验证损失近似真实任务成功率，从而直接按“对最终策略有没有帮助”来选先验示范，而不是按视觉/动作相似度猜测。
> - **Key Performance**: MetaWorld 50 任务平均成功率较最强启发式基线约 +10pct；OXE 实世界平均成功率 61.0%，高于最佳启发式基线 40.0%。

> [!info] **Agent Summary**
> - **task_path**: 大规模异构机器人示范库 + 少量目标任务演示 -> 选择可迁移先验数据 -> 目标任务模仿策略
> - **bottleneck**: 数据选择目标与最终任务成功率脱节；真实 rollout 昂贵且不可微，无法直接做端到端影响估计
> - **mechanism_delta**: 把“按相似度检索数据”改成“按对目标验证损失/成功率的估计影响排序”，并用可微代理目标与 metagradients 实现离线选择
> - **evidence_signal**: 跨 MetaWorld、LIBERO、OXE 都有增益，且代理目标相对真实 rollout 仅小幅退化但估计速度约快 8×
> - **reusable_ops**: [可微代理目标, 簇级影响打分]
> - **failure_modes**: [代理损失与真实成功率相关性不足时会误排数据, 大规模异构数据下若不聚类则影响估计噪声高]
> - **open_questions**: [如何自动选择聚类粒度与选择预算, 能否用更小代理模型近似大模型 datamodel]

## Part I：问题与挑战

这篇论文要解决的，不是“从大库里找看起来像目标任务的数据”，而是“从大库里找**真的会让目标任务策略更成功**的数据”。

### 真问题是什么
随着 OXE、Bridge、DROID、Octo 这类大规模机器人数据和通用策略出现，机器人领域已经从“缺数据”走向“数据太多、太杂、混进去反而伤模型”。  
对一个具体任务做模仿学习时，通常手里只有少量目标任务示范；如果直接把大量先验数据一起 co-train，常常会引入：

- 不相关任务干扰
- 相似外观但错误动作分布
- 跨实验室/跨机器人/跨相机的强分布偏移
- 低质量或次优示范污染

### 真瓶颈在哪里
现有检索式方法大多用视觉相似、动作相似、状态-动作相似来选数据，但这些都是**人定义的替代标准**，不等于“这条数据会不会提升最终任务成功率”。  
而如果真的按最终 success rate 来选，又要反复训练策略并做环境 rollout，这在机器人里代价太高、速度太慢，还可能有实体风险。

### 输入 / 输出 / 边界
- **输入**：大规模先验示范库 `D`，少量目标任务示范 `Dtarget`，固定的模仿学习算法/策略架构
- **输出**：一个被筛出来的先验子集 `Dsel`，以及用 `Dsel ∪ Dtarget` 训练出的更强专项策略
- **边界条件**：
  - 需要至少少量目标任务示范
  - 主要面向 imitation learning / behavior cloning，而非纯在线 RL
  - 需要训练与验证过程可微，才能高效用 metagradients
  - 当前主要验证在 manipulation 任务上

**Why now**：因为现在已有足够大的先验数据和通用机器人骨干模型，数据选择终于成为比“再去采更多同任务数据”更现实、更重要的杠杆。

## Part II：方法与洞察

DataMIL 的核心不是新 policy 架构，而是一个**面向策略表现的数据选择层**。

### 方法主线
1. **先把先验数据分簇**  
   不直接给每个单帧/单状态动作打分，而是按轨迹或子轨迹分簇。原因是单样本影响太小、估计方差太大；簇级别更稳定。

2. **用代理目标替代真实 rollout 成功率**  
   作者不用真实 success rate 来训练 datamodel，而是用 held-out 目标任务示范上的行为克隆验证损失。  
   直觉上：如果某批先验数据让模型在目标验证示范上的 BC loss 下降，它大概率也更接近会提升该任务成功率。

3. **估计每个簇对目标表现的影响分数**  
   他们使用两类 datamodel estimator：
   - **Regression estimator**：精度高，但要训练很多子模型，适合较小规模设置
   - **Metagradient estimator**：利用可微性近似/计算影响，更适合 Octo 这类大模型

4. **选 top-x% 正影响数据并与目标数据共训**  
   最终只保留高正影响分数的数据，与目标任务数据按固定比例 co-train，得到专项策略。

5. **在真实世界里额外做 target mix-in**  
   对 OXE 这类强异构场景，作者把一部分目标任务数据混入 datamodel 估计阶段，以缓解 prior-target 分布偏移。

### 核心直觉

DataMIL 改变的关键，不是“怎么看数据像不像”，而是“**把选数据信号从表面相似度，换成对最终策略优化的估计边际贡献**”。

这带来三个层面的变化：

- **什么变了**：  
  从 similarity-based retrieval 变成 policy-aware influence estimation

- **哪个瓶颈被改变了**：  
  原方法的信息瓶颈是：只能看到样本“长得像不像”，看不到它会把策略更新推向哪里。  
  DataMIL 直接问：**把这批数据加入训练后，目标任务验证表现会变好还是变坏？**

- **能力为什么提升**：  
  这样就能保留那些“看起来不太像，但功能上有迁移价值”的数据，也能删掉“看起来很像，但动作分布有害”的数据。  
  这尤其适合跨 embodiment、跨场景、跨数据源的异构机器人库。

### 为什么这个设计有效
- **代理目标**把 rollout 依赖改成离线、可微、低成本的验证信号，让 datamodel 在机器人上变得可用
- **簇级归因**降低了单样本影响估计的方差
- **metagradients**把“海量重训练”变成“少量训练 + 影响估计”，让大模型上也能跑
- **target mix-in**缓解 prior 数据与目标任务分布差太远时的估计失真

### 战略 trade-off

| 设计 | 解决的问题 | 收益 | 代价/风险 |
|---|---|---|---|
| 用目标验证 BC loss 代替 rollout success | rollout 昂贵、不可微 | 可离线、可微、能用 metagradients | 假设 loss 与真实成功率相关 |
| 轨迹/子轨迹级分簇 | 单样本影响太噪 | 打分更稳 | 粒度是超参数，过粗会丢细节 |
| Metagradient estimator | 回归式 datamodel 太贵 | 能扩展到 Octo / OXE | 仍需数倍于全数据训练的计算 |
| target mix-in | 真实世界分布偏移大 | 更贴近目标域 | 需要消耗一部分目标数据 |

## Part III：证据与局限

### 关键证据信号

**信号 1：代理目标并没有把问题“带偏”**  
在 MetaWorld 的 pick-place-wall 上，作者直接比较：
- 用真实 rollout success 训练 datamodel
- 用验证损失代理目标训练 datamodel
- 用 metagradient 版本训练 datamodel

结论是：代理目标版本只带来小幅性能下降，而 metagradient 版本再小退一点，但训练约快 **8×**。这说明它不是单纯“便宜但没用”的替代，而是一个可接受的工程-性能折中。

**信号 2：在大规模仿真任务上，DataMIL 比启发式检索更稳**  
在 **MetaWorld 50 任务**上，论文报告 DataMIL 平均成功率比最强 baseline 高约 **10 个百分点**。  
这里最重要的不是绝对数，而是它能同时完成两件事：
- 找到相关任务的数据
- 过滤掉次优/有害动作数据

这正是纯 state / action / state-action similarity 方法最容易失败的地方。

**信号 3：在更接近真实部署的异构设置中，优势更明显**  
在 **OXE 实世界设置**里，DataMIL 的平均成功率是 **61.0%**，而最强启发式基线大约 **40.0%**。  
更关键的是：
- **Tiago-Sink** 是跨 embodiment 迁移，目标机器人未出现在 prior 中，DataMIL 仍能选到有帮助的数据
- **Droid-Multitask** 说明它不只适合单任务，也能为多个目标任务共同筛数据

**信号 4：它学到的不是“外观相似”，而是“训练效用”**  
定性分析显示，DataMIL 往往会从多个数据源选数据，而不是像某些 heuristic 那样过度集中到单一数据集。  
同时，top-ranked 和 bottom-ranked 样本在视觉上可能很像，但动作含义不同；这支持了论文的核心论点：**有用/有害的差别，不一定能从表面相似度看出来。**

### 能力跃迁到底体现在哪
相对 prior work，真正的 jump 在于：

- 从**相似度驱动**变成**性能驱动**
- 从只能在干净、同构数据上凑效，变成能在**异构、多源、跨 embodiment**场景仍稳定工作
- 从“选数据是外部启发式”变成“选数据直接围绕最终策略目标”

### 局限性
- **Fails when**: 目标验证损失与真实任务成功率相关性弱时，代理目标会把影响分数估错；在极强分布偏移场景下，如果没有 target mix-in，datamodel 可能泛化失真。
- **Assumes**: 有少量目标任务示范；训练过程和评估代理可微；可承担 datamodel 估计的较高算力成本。论文也明确承认，即使用 metagradients，成本仍是“训练全数据模型”的数倍；真实世界评估多为单随机种子，复现实证强度弱于仿真。
- **Not designed for**: 无示范的纯在线 RL 数据选择；完全依赖 rollout reward 的闭环主动采样；超大规模多目标任务集下的自动化数据治理。并且在较干净、较同质的设置（如 LIBERO）里，增益更像“稳定小胜”而非压倒性优势。

### 可复用组件
这篇论文最值得迁移的，不一定是完整 DataMIL 流程，而是下面几个操作件：

- **目标任务 held-out 验证损失作为离线代理目标**
- **簇级别 influence scoring**
- **在强分布偏移下加入少量 target mix-in**
- **先选高正影响数据，再与 target 数据 co-train**

这些组件都可以独立嫁接到其他 VLA / imitation learning adaptation pipeline 里。

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_DataMIL_Selecting_Data_for_Robot_Imitation_Learning_with_Datamodels.pdf]]