---
title: "Dex1B: Learning with 1B Demonstrations for Dexterous Manipulation"
venue: RSS
year: 2025
tags:
  - Embodied_AI
  - task/dexterous-manipulation
  - task/grasp-synthesis
  - cvae
  - geometric-constraints
  - iterative-data-generation
  - dataset/Dex1B
  - dataset/DexGraspNet
  - dataset/DexYCB
  - dataset/ARCTIC
  - opensource/no
core_operator: 通过“优化种子集 → 几何约束CVAE生成 → 去偏重采样 → 后优化与仿真筛选”的迭代闭环，把灵巧手可执行演示高效扩展到10亿级。
primary_logic: |
  物体几何/任务条件 + 小规模高质量优化种子演示 → 训练带SDF几何约束与局部点条件的CVAE，并对稀有条件做去偏采样、再经轻量后优化与仿真过滤 → 大规模、多样且可执行的抓取/关节操作轨迹与更强下游策略
claims:
  - "Claim 1: 在 DexGraspNet 基准上，DexSimple 加后优化达到 86.0% success rate 和 0.125 Q1，超过 UGG 加后优化的 64.1% 和 0.036；若再加过滤器可达 92.6% success rate [evidence: comparison]"
  - "Claim 2: 用 Dex1B 训练的 BC w. PointNet 与 DexSimple 在 lifting 和 articulation 的 train/test 各划分上都优于用 DexYCB/ARCTIC 训练的对应模型；例如 DexSimple 在 Dex1B lifting test 上从 22.80% 提升到 45.40% [evidence: comparison]"
  - "Claim 3: 去掉 SDF 几何损失后，DexSimple 在 DexGraspNet 上的 success rate 从 63.7% 降到 0.7%，说明几何约束是可行抓取生成的关键因子 [evidence: ablation]"
related_work_position:
  extends: "GraspTTA (Jiang et al. 2021)"
  competes_with: "DexGraspNet 2.0 (Zhang et al. 2024); UGG (Lu et al. 2024)"
  complementary_to: "DextrAH-RGB (Singh et al. 2024); π0 (Black et al. 2024)"
evidence_strength: strong
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/RSS_2025/2025_Dex1B_Learning_with_1B_Demonstrations_for_Dexterous_Manipulation.pdf
category: Embodied_AI
---

# Dex1B: Learning with 1B Demonstrations for Dexterous Manipulation

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2506.17198), [Project](https://jianglongye.com/dex1b)
> - **Summary**: 这篇论文把灵巧手学习的核心问题重新定义为“数据不够而不是手没用”，并用“优化种子 + 几何约束生成模型 + 去偏采样 + 仿真筛选”的闭环，构建了 10 亿级 Dex1B 演示数据与一个强基线 DexSimple。
> - **Key Performance**: DexGraspNet 上 DexSimple+后优化成功率 86.0%（再加过滤器 92.6%）；真机 10 个未见物体平均成功率 96%。

> [!info] **Agent Summary**
> - **task_path**: 物体网格/点云 + 任务类型（抓取/关节操作） -> 接触关键帧手姿 -> 完整操作轨迹
> - **bottleneck**: 高自由度灵巧手缺少同时满足可行性与多样性的大规模演示，导致学习到的动作分布窄且不稳
> - **mechanism_delta**: 把纯优化数据生成改成“带 SDF 几何约束和局部点条件的 CVAE 扩样 + 去偏采样 + 轻量后优化/仿真过滤”的可迭代数据引擎
> - **evidence_signal**: DexGraspNet 上 DexSimple+后优化达到 86.0% SR，高于 UGG+后优化的 64.1%，且去掉 Lsdf 后 SR 从 63.7% 直接掉到 0.7%
> - **reusable_ops**: [sphere-SDF 几何约束, 逆频率条件采样]
> - **failure_modes**: [开放环执行对 sim-to-real 与观测误差敏感, 多物体/遮挡场景覆盖不足]
> - **open_questions**: [如何从关键帧生成走向闭环视觉触觉控制, 如何减少对大规模仿真过滤的依赖]

## Part I：问题与挑战

这篇论文抓得很准的一点是：**灵巧手难，不只是控制难，而是数据分布太稀薄**。  
高自由度手要学会抓取和关节物体操作，需要覆盖大量“接触方式 + 姿态组合 + 物体几何”的长尾情况。过去三类数据来源都不够理想：

1. **人工示范**：贵、慢，而且对机器人手型并不天然匹配。
2. **纯优化生成**：质量高，但扩到超大规模时算力成本太高，而且会偏向“容易解”的姿态。
3. **RL 生成**：能学到稳健性，但演示分布往往不够丰富。

作者进一步指出，生成模型直接拿来扩数据也有两个硬伤：

- **Feasibility**：生成出来的手姿常常穿透、接触不稳，成功率低于确定性方法。
- **Diversity**：即便是生成模型，也常常只是在已有演示之间插值，未必真正扩展分布支持集。

### 输入/输出接口与边界

- **输入**：物体 mesh / point cloud、任务类型（grasping 或 articulation）、手型、可选局部几何条件。
- **输出**：接触关键帧手姿 + 由运动规划补全的完整操作轨迹。
- **任务边界**：
  - 单物体桌面场景；
  - 抓取任务目标是抓住并抬升到 0.4m；
  - articulation 任务目标是把关节打开 0.5；
  - 真机部署是**开放环**，不是闭环反馈控制。

### 为什么是现在解决

因为现在三件事终于能接上了：

- GPU 并行仿真（ManiSkill / SAPIEN）可做大规模验证；
- Warp-Lang 让几何/SDF 优化足够快；
- 生成模型已经足够强，可以作为“数据放大器”而不只是策略本体。

所以这篇论文真正想解决的不是“再发明一个更复杂的抓取网络”，而是：**如何造一个可持续自举的数据工厂，把高质量灵巧操作演示规模化生产出来。**

## Part II：方法与洞察

论文的方法由两部分组成：

- **Dex1B**：一个迭代式数据生成引擎
- **DexSimple**：一个带几何约束的简单 CVAE 生成模型

### 核心直觉

**核心变化**：把“直接从数据里拟合平均动作分布”改成“先用优化找到可行流形，再用几何约束生成模型在这个流形附近扩展，并用条件去偏把采样推向稀有区域”。  

这带来的因果链条是：

- **加入几何约束**  
  → 把生成空间里的大量不可行解排除掉  
  → 改变了模型输出分布的“可行域”  
  → 成功率显著提升。

- **加入局部点条件 + 逆频率采样**  
  → 不再只复现常见接触区域  
  → 改变了训练/采样时的条件分布  
  → 多样性和长尾覆盖增强。

- **加入后优化 + 仿真过滤 + 迭代再训练**  
  → 把“粗生成”修成“可执行样本”  
  → 扩大了下轮训练数据的支持集  
  → 数据和模型一起滚雪球。

一句话概括：**优化负责把数据锚定在物理可行域，生成模型负责把规模拉上去，去偏采样负责把分布撑开。**

### 方法骨架

#### 1. 先用优化做一个高质量种子集

作者先不用神经网络硬凑 1B，而是先构造约 500 万个种子姿态。  
这里有两个关键工程点：

- 用**sphere 近似手部几何**，替代更重的 link mesh；
- 在抓取与 articulation 上分别用不同任务能量，前者偏 force closure，后者偏 task wrench。

这一步的作用不是追求最终规模，而是给生成模型一个**高纯度冷启动分布**。

#### 2. 用 DexSimple 学“可行的生成分布”

DexSimple 很“反潮流”：不是 diffusion，而是一个相对简单的 **PointNet + CVAE**。

但作者做了一个很关键的改动：  
**在训练里显式加入 SDF 风格的几何损失**，约束手与物体之间不要穿透，并保持合理接触。

这一步的意义不是“更复杂的网络”，而是**把物理几何先验直接写进生成目标**。  
论文的消融也证明，这不是装饰项，而是决定成败的关键项。

#### 3. 用局部几何条件做去偏扩分布

作者给每个手姿分配了一个与其朝向相关的**物体局部 3D 点**，然后把这个局部点特征作为条件输入生成模型。

接着不是按经验均匀采样，而是：

- 统计现有数据中每个条件值出现频率；
- **对低频条件反向加权采样**；
- 同时对更难的对象采更多样本。

这一步很重要，因为它不只是“多采点数据”，而是在主动纠偏：  
**从“复现已有模式”转向“补齐已有分布的盲区”。**

#### 4. 用轻量后优化与仿真做质量闭环

生成模型出样后，作者并不直接收下，而是：

- 做一次轻量后优化，修正局部穿透和接触；
- 在 ManiSkill / SAPIEN 中做成功性验证；
- 只保留成功样本回流到下一轮训练。

于是形成迭代链：

**5M seed → 50M → 500M → 950M successful trajectories**

这本质上是一个“生成—验证—再训练”的自举循环。

#### 5. 关键帧生成，轨迹靠规划补齐

论文没有直接生成整段复杂控制，而是只生成关键接触帧，然后用运动规划补 reaching / lifting / opening 过程。

这是一种很实用的任务分解：

- 把最难学的“接触几何”交给生成模型；
- 把更结构化的“到达与执行轨迹”交给优化/规划。

这也是 Dex1B 能放大到 1B 的原因之一：**没有把所有复杂度都塞进端到端策略学习里。**

### 战略权衡

| 设计选择 | 解决的瓶颈 | 带来的能力 | 代价/风险 |
|---|---|---|---|
| 优化生成 5M 种子 | 冷启动可行性 | 给模型一个高质量可行分布 | 初始阶段仍然耗 GPU/仿真 |
| CVAE + SDF 几何损失 | 穿透与不可执行样本 | 大幅提升抓取可行性 | 仍需仔细设计几何表示 |
| 局部点条件 + 去偏采样 | 生成分布塌缩到常见接触区 | 扩大长尾接触覆盖 | 需要显式统计对象上的条件分布 |
| 后优化 + 仿真过滤 | 生成样本残余误差 | 把“看起来对”变成“模拟里真能做” | 强依赖仿真效率 |
| 关键帧建模 + 规划补轨迹 | 直接学全轨迹过重 | 大幅降低学习难度 | 开放环执行更怕误差积累 |

## Part III：证据与局限

### 关键证据

- **比较信号：抓取生成质量明显跃升**  
  在 DexGraspNet 上，DexSimple + post-optimization 的 success rate 达到 **86.0%**，明显超过 UGG + post-optimization 的 **64.1%**。如果再用过滤器，能到 **92.6%**。  
  这说明作者的“简单 CVAE + 几何约束”并不是保守 baseline，而是一个更强的生成器。

- **比较信号：数据规模与多样性确实转化成下游收益**  
  无论是 BC w. PointNet 还是 DexSimple，只要训练数据从 DexYCB / ARCTIC 换成 Dex1B，lifting 和 articulation 的 train/test 表现都更好。  
  这说明 Dex1B 不只是“多”，而是**对泛化有用的多**。

- **消融信号：几何约束是因果核心，不是细节润色**  
  去掉 SDF loss 后，DexSimple 在 DexGraspNet 上的 success rate 从 **63.7%** 掉到 **0.7%**。  
  这是全篇最强的因果证据：没有几何约束，生成模型几乎失去可执行性。

- **缩放信号：更多数据持续带来提升**  
  数据缩减实验显示，两项任务都随数据量增加而变好，且 lifting 对数据规模更敏感。  
  这和论文的主张一致：高精度抓取尤其依赖更充分的几何覆盖。

- **真机信号：不是只在模拟里好看**  
  在 xArm + Ability Hand 上，面对 10 个未见物体，作者方法平均 **96%**，而对比方法 DexSampler 为 **58%**。  
  这说明该数据引擎学到的分布至少有一定 sim-to-real 可迁移性。

### 最值得记住的两个数字

1. **86.0%**：DexGraspNet 上 DexSimple+后优化的抓取成功率。  
2. **96%**：真机 10 个未见物体上的平均成功率。

### 局限性

- **Fails when**: 开放环部署遇到较大观测噪声、控制误差或 sim-to-real 偏差时；多物体、拥挤、遮挡明显的场景中，单物体条件建模容易失效。
- **Assumes**: 有可靠的对象 mesh/point cloud；能使用 ManiSkill/SAPIEN 做大规模成功过滤；依赖 Warp-Lang/GPU 加速优化；默认桌面单物体设定、相机标定和 IK/运动规划可用。
- **Not designed for**: 闭环触觉操作、长时程 in-hand manipulation、复杂多物体整理/重排、完全脱离仿真的数据生产。

### 可复用组件

这篇论文最可复用的，不一定是“1B 数据量”本身，而是下面几个操作原语：

- **sphere-SDF 几何约束训练**：适合任何高自由度接触生成任务。
- **局部几何条件 + 逆频率重采样**：适合扩展长尾接触分布，而不是只做插值增强。
- **生成器 → 轻量后优化 → 仿真 critic → 再训练**：这是一个很通用的机器人数据自举模板。
- **关键帧生成 + 轨迹规划补全**：适合把“接触决策”和“执行控制”解耦。

总体上，这篇工作的能力跃迁不在于“提出了最复杂的模型”，而在于它找到了一个更像工业化生产线的答案：  
**先把可行性抓住，再把多样性推开，最后把规模做上去。**

![[paperPDFs/Vision_Action_VA_Models_2025/RSS_2025/2025_Dex1B_Learning_with_1B_Demonstrations_for_Dexterous_Manipulation.pdf]]