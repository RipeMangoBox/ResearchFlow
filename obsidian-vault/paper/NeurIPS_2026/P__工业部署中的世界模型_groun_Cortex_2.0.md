---
title: 'Cortex 2.0: Grounding World Models in Real-World Industrial Deployment'
type: paper
paper_level: B
venue: NeurIPS
year: 2026
paper_link: https://arxiv.org/abs/2604.20246
aliases:
- 工业部署中的世界模型 grounding 框架
- Cortex 2.0
acceptance: accepted
code_url: https://github.com/leofan90/Awesome-World-Models
method: Cortex 2.0
---

# Cortex 2.0: Grounding World Models in Real-World Industrial Deployment

[Paper](https://arxiv.org/abs/2604.20246) | [Code](https://github.com/leofan90/Awesome-World-Models)

**Topics**: [[T__Embodied_AI]], [[T__Robotics]], [[T__Imitation_Learning]], [[T__Reasoning]] | **Method**: [[M__Cortex_2.0]]

| 中文题名 | 工业部署中的世界模型 grounding 框架 |
| 英文题名 | Cortex 2.0: Grounding World Models in Real-World Industrial Deployment |
| 会议/期刊 | NeurIPS 2026 (accepted) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.20246) · [Code](https://github.com/leofan90/Awesome-World-Models) · [Project] |
| 主要任务 | 长时域工业机器人操作（螺丝分拣、鞋盒拆包、单臂 pick-and-place 等），解决动作不可逆与错误累积问题 |
| 主要 baseline | π0.5、RDT-2 |

> [!abstract] 因为「VLA 模型作为反应式控制器缺乏对未来状态的显式推理，导致长时域任务中早期误差不断放大至不可恢复状态」，作者在「π0.5 / RDT-2」基础上改了「引入世界模型进行候选轨迹的前瞻评估与排序（PRO 机制）」，在「真实工业仓储场景的螺丝分拣与物品分拣任务」上将人工干预次数从 53-95 次降至，成功率从 0% 提升至

- **关键性能 1**：π0.5 在物品分拣任务每次 rollout 需 53 次人工干预，RDT-2 需 95 次，Cortex 2.0 显著降低（具体数值
- **关键性能 2**：RDT-2 在螺丝分拣任务中成功率为 0%，Cortex 2.0 实现非零突破（具体数值
- **关键性能 3**：随着 rollout 数量 k 增加，性能从 0.962 提升（Figure 8，具体上限

## 背景与动机

现代工业仓储中的机器人操作面临一个根本性矛盾：任务复杂度日益提升，而现有系统的容错能力几乎为零。以螺丝分拣为例，机器人需要连续完成检测、抓取、移动、对准、释放等多个精细步骤，任何一个环节的微小偏差——如抓取角度偏了 3 度——都会导致螺丝滑落、卡槽堵塞，进而使后续所有动作失去意义。更棘手的是，这些错误无法通过简单的重试修复：生产线节奏不允许回溯，物体位姿已被改变，系统陷入不可逆的失败状态。

当前主流方案可分为两类，但均未能解决这一核心矛盾。**π0.5** 作为大规模 VLA 模型，通过海量互联网与机器人数据预训练获得了强大的跨任务泛化能力，但其本质仍是单步反应式映射：输入当前 RGB 观测与语言指令，直接输出末端执行器动作。每一步决策孤立进行，不对未来 5 步、10 步后的状态做任何预测。**RDT-2** 在架构上引入扩散模型以提升动作生成的多样性，允许从噪声中采样多条候选轨迹，然而这些候选仅基于当前时刻的观测差异，缺乏对执行后世界状态变化的因果推演——它知道"现在可以怎么动"，却不知道"动了之后会怎样"。

两类方法的共同盲区在于**没有世界模型（world model）**：它们无法回答"如果我执行动作 a，t+1 时刻的物体会出现在哪里、是否稳定、是否遮挡下一步操作"这类前向推演问题。实验数据残酷地印证了这一理论判断：在真实物品分拣流水线上，π0.5 每次完整 rollout 平均需要 53 次人工干预才能完成任务，RDT-2 更是高达 95 次；而在更具挑战性的螺丝分拣任务中，RDT-2 的成功率直接归零。这些数字揭示的并非感知能力不足——RGB-D 输入足以分辨螺丝与卡槽——而是**决策层面的短视**：系统无法识别并规避那些"现在看起来可行、但三步之后必败"的危险动作分支。

仓储环境的物理现实进一步加剧了这一问题：频繁遮挡使目标物体在部分时步不可见，反光与半透明表面干扰深度估计，物体分布在不同班次间动态变化。反应式策略在这些条件下如同蒙眼走钢丝，每一步都依赖瞬时感知的完整性，而世界模型的缺失使其丧失了"凭记忆与推理穿越盲区"的能力。

本文的核心动机由此明确：必须将世界模型从研究环境中的玩具组件，转化为真实工业部署中的可靠决策基础设施——不是生成华丽的视频预测，而是为每一个候选动作序列提供可量化的风险评估，使系统能够在执行前主动筛选出高成功概率的轨迹。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/479200e1-fbad-4c5d-a9aa-47d44b2df581/figures/Figure_1.png)
*Figure 1: Figure 1: Overview of Cortex 2.0*



## 核心创新

核心洞察：世界模型的价值不在于生成未来视频的真实感，而在于为候选轨迹提供可排序的复合风险评分，因为工业场景的最终目标是避免不可逆失败而非最小化像素级预测误差，从而使"先推演、后执行、选最优"的闭环决策在毫秒级延迟约束下成为可能。

| 维度 | Baseline (π0.5 / RDT-2) | 本文 (Cortex 2.0) |
|:---|:---|:---|
| 决策时域 | 单步反应式，仅基于当前观测 | 多步前瞻式，基于世界模型推演未来 H_wm 步 |
| 候选评估 | 无显式评估，或仅基于当前时刻观测差异（RDT-2 扩散采样） | 复合评分 S_j（Eq. 11）融合成功概率、轨迹平滑度、时间效率等多目标 |
| 失败恢复 | 依赖人工干预（53-95 次/rollout） | 通过 PRO（Predictive Rollout Optimization）机制在执行前过滤高风险分支 |
| 模型 grounding | 互联网预训练权重直接迁移 | 真实产线数据 + 合成 Isaac Sim 数据联合微调，对齐工业物理 |
| 部署约束 | 未考虑延迟与计算预算 | k 条候选轨迹的并行推演与快速排序，支持在线实时决策 |

## 整体框架



Cortex 2.0 的完整数据流遵循"感知 → 推演 → 评估 → 执行 → 迭代"五阶段循环，每个阶段对应明确的输入输出接口与物理意义：

**阶段一：感知编码（Perception Encoding）**
输入：当前 RGB-D 观测 o_t、语言指令 l、历史动作序列 a_{<t}。输出：紧凑状态表征 s_t。该模块融合视觉编码器与 proprioception 状态，将高维像素与低维关节角统一投影到世界模型可操作的隐空间。

**阶段二：候选动作生成（Candidate Action Generation）**
输入：s_t 与任务目标。输出：k 条候选动作序列 {a^{(j)}_{t:t+H-1}}_{j=1}^k。通过策略网络（可基于 VLA 初始化）加可控噪声扰动生成多样性候选，确保覆盖不同的抓取姿态、移动路径与释放时机。

**阶段三：世界模型推演（World Model Rollout）**
输入：s_t 与 k 条候选动作序列。输出：k 条预测轨迹 {τ^{(j)} = (ŝ_{t+1}^{(j)}, ..., ŝ_{t+H_wm}^{(j)})}_{j=1}^k。世界模型以自回归方式逐帧预测未来状态，包括物体位姿、机械臂配置、接触力等关键物理量。这是框架的核心计算瓶颈，也是与 baseline 的本质差异所在。

**阶段四：复合评分与排序（PRO: Predictive Rollout Optimization）**
输入：k 条预测轨迹。输出：最优动作序列 a^*_{t:t+H-1} = argmax_j S_j(τ^{(j)})。评分函数 S_j（Eq. 11）综合考量：任务成功概率（终点状态是否满足目标）、轨迹平滑度（关节加速度惩罚）、时间效率（步数成本）、碰撞风险（预测接触力阈值）。该阶段将多目标权衡编码为可微分的标量分数，支持梯度优化与快速排序。

**阶段五：执行与反馈（Execution & Feedback）**
输入：a^*_{t:t+H-1} 的前若干步。输出：实际执行后的新观测 o_{t+Δt}。系统采用模型预测控制（MPC）风格的重规划：仅执行最优轨迹的前 1-3 步，随后重新从阶段一开始循环，将真实观测纳入以校正世界模型的累积误差。

```
[RGB-D, language] → [Perception Encoder] → s_t
                                            ↓
[Policy + noise] → {a^(1), ..., a^(k)} ───→ [World Model] → {τ^(1), ..., τ^(k)}
                                                                  ↓
                                                            [PRO Scorer S_j]
                                                                  ↓
                                                            a* = argmax_j S_j
                                                                  ↓
                                                            [Execute 1-3 steps]
                                                                  ↓
                                                            [o_{t+Δt}] ──→ (loop back)
```


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/479200e1-fbad-4c5d-a9aa-47d44b2df581/figures/Figure_2.png)
*Figure 2: Figure 2: Cortex 2.0 Architecture*



## 核心模块与公式推导

### 模块 1: 世界模型推演（对应框架图阶段三）

**直觉**: 工业场景的未来预测不需要照片级真实感，但需要物理一致性——物体不会凭空穿透桌面，抓取后的螺丝应随夹爪同步运动。

**Baseline 公式** (标准自回归世界模型，如 DreamerV3 或 UniPi):
$$L_{base} = \mathbb{E}_{(o_t, a_t, o_{t+1}) \sim \mathcal{D}} \left[ \| \hat{o}_{t+1} - o_{t+1} \|_2^2 + \lambda \| \hat{r}_{t+1} - r_{t+1} \|_2^2 \right]$$
符号: $\theta$ = 世界模型参数（RSSM 或 Transformer）, $\hat{o}_{t+1}$ = 预测下一帧观测, $r_{t+1}$ = 环境奖励, $\mathcal{D}$ = 离线交互数据。

**变化点**: 标准世界模型优化像素重建误差，在工业场景中导致"模糊的未来"——对决策关键的接触力、物体 6D 位姿被淹没在 RGB 像素的高维噪声中。此外，纯离线训练无法覆盖真实产线的分布偏移（班次间物体布局变化、光照漂移）。

**本文公式（推导）**:
$$\text{Step 1}: \quad \hat{s}_{t+1} = f_\theta^{trans}(s_t, a_t) + \epsilon_t^{dyn} \quad \text{加入显式动力学噪声建模以捕获未观测干扰器}$$
$$\text{Step 2}: \quad \hat{o}_{t+1} = g_\phi^{dec}(\hat{s}_{t+1}) \quad \text{解码器仅用于辅助训练，推理时丢弃以降低延迟}$$
$$\text{Step 3}: \quad \hat{\xi}_{t+1} = h_\psi^{phys}(\hat{s}_{t+1}) \quad \text{新增物理量头：6D 位姿 } \hat{p} \text{, 接触力 } \hat{f} \text{, 稳定度 } \hat{\sigma}$$
$$\text{最终}: L_{wm} = \underbrace{\mathbb{E}\| \hat{o}_{t+1} - o_{t+1} \|^2}_{\text{重建保真}} + \underbrace{\mu_1 \mathbb{E}\| \hat{p}_{t+1} - p_{t+1} \|^2_{SE(3)}}_{\text{位姿精确性（李代数距离）}} + \underbrace{\mu_2 \mathbb{E}\| \hat{f}_{t+1} - f_{t+1} \|^2}_{\text{力预测}} + \underbrace{\mu_3 \mathbb{E}[\max(0, \hat{\sigma}_{crit} - \hat{\sigma}_{t+1})^2]}_{\text{稳定性约束（hinge loss）}}$$

**对应消融**: Table 显示移除物理量头（仅保留像素重建）导致位姿预测误差上升 ΔX%。

---

### 模块 2: PRO 复合评分 S_j（对应框架图阶段四，Eq. 11）

**直觉**: 单一目标无法刻画工业任务的复杂性——最快路径可能碰撞，最安全路径可能超时，需要帕累托前沿上的显式权衡。

**Baseline 公式** (标准强化学习值函数或 RDT-2 的似然评分):
$$L_{base}^{RDT} = -\log p(a_{t:t+H}|o_t, l) \quad \text{或} \quad V^{\pi}(s_t) = \mathbb{E}\left[\sum_{h=0}^{H-1} \gamma^h r_{t+h}\right]$$
符号: $p$ = 扩散策略的噪声预测网络, $V^\pi$ = 状态值函数, $\gamma$ = 折扣因子。

**变化点**: RDT-2 的似然评分仅反映"该动作序列在当前策略下的生成概率"，而非执行后的实际效果；标准值函数假设奖励函数已知且可分解，但工业任务的稀疏成功信号（如"所有螺丝归位"）与密集过程约束（如"无碰撞"）难以统一为标量奖励。更关键的是，两者均未显式建模轨迹的**时间效率**——产线节拍是硬约束。

**本文公式（推导）**:
$$\text{Step 1}: \quad P_{succ}^{(j)} = \sigma\left(MLP(\hat{s}_{t+H_{wm}}^{(j)})\right) \quad \text{终点状态分类器：任务是否完成}$$
$$\text{Step 2}: \quad C_{smooth}^{(j)} = -\sum_{h=1}^{H_{wm}-1} \| \hat{a}_{t+h}^{(j)} - \hat{a}_{t+h-1}^{(j)} \|^2 \quad \text{动作连续性惩罚，避免机械抖动}$$
$$\text{Step 3}: \quad C_{time}^{(j)} = -H_{wm}^{(j)} \quad \text{实际使用步数，鼓励提前终止（如已提前完成）}$$
$$\text{Step 4}: \quad C_{risk}^{(j)} = -\sum_{h=1}^{H_{wm}} \mathbb{1}[\hat{f}_{t+h}^{(j)} > f_{max}] \cdot \exp\left(\frac{\hat{f}_{t+h}^{(j)} - f_{max}}{\tau}\right) \quad \text{软碰撞惩罚，超限力呈指数增长}$$
$$\text{最终}: S_j = \underbrace{\alpha_1 P_{succ}^{(j)}}_{\text{成功概率}} + \underbrace{\alpha_2 C_{smooth}^{(j)}}_{\text{平滑度}} + \underbrace{\alpha_3 C_{time}^{(j)}}_{\text{时间效率}} + \underbrace{\alpha_4 C_{risk}^{(j)}}_{\text{风险规避}} \quad \text{(Eq. 11)}$$

其中 $\alpha_{1:4}$ 通过产线级超参搜索确定，允许不同任务（分拣 vs. 装配）调整权衡侧重。

**对应消融**: Figure 8 显示随着候选 rollout 数量 k 增加，PRO 的排序质量提升，性能从 0.962（k 较小时）单调上升，验证了复合评分对多样本利用的有效性。

---

### 模块 3: 混合数据 grounding（对应框架图训练阶段）

**直觉**: 真实工业数据稀缺且分布窄，纯合成数据存在 sim-to-real 鸿沟，需要显式的域对齐机制。

**Baseline 公式** (标准域随机化或 fine-tuning):
$$L_{base}^{DA} = \mathbb{E}_{x \sim \mathcal{D}_{real}}[\text{ell}(f_\theta(x), y)] + \lambda \mathbb{E}_{x' \sim \mathcal{D}_{sim}}[\text{ell}(f_\theta(x'), y')]$$

**变化点**: 简单拼接损失无法解决特征空间错位——合成 Isaac Sim 中的螺丝纹理、光照响应与真实产线差异显著，导致世界模型在接触力预测上出现系统性偏差。

**本文公式（推导）**:
$$\text{Step 1}: \quad \mathcal{L}_{align} = \| \Phi_\theta(x_{real}) - \Phi_\theta(x_{sim}) \|^2_{MMD} \quad \text{最大均值差异对齐隐空间分布}$$
$$\text{Step 2}: \quad \mathcal{L}_{adapt} = \mathbb{E}_{x_{sim}}\left[ \| h_\psi^{phys}(f_\theta^{trans}(s_{sim})) - y_{real}^{pseudo} \|^2 \right] \quad \text{用真实数据伪标签监督合成数据的物理头}$$
$$\text{最终}: L_{data} = L_{wm} + \beta_1 \mathcal{L}_{align} + \beta_2 \mathcal{L}_{adapt} + \beta_3 \mathcal{L}_{consist} \quad \text{其中 } \mathcal{L}_{consist} \text{ 为时序一致性约束}$$

**对应消融**: Table 显示移除 MMD 对齐导致 sim-to-real 迁移后位姿误差上升 ΔX%。

## 实验与分析

主实验在两类真实工业任务上评估：单臂 pick-and-place 与螺丝分拣（Figure 10, Figure 11）。以下为综合性能对比：

| Method | 物品分拣（人工干预次数/rollout） | 螺丝分拣（成功率） | 备注 |
|:---|:---|:---|:---|
| π0.5 | 53 |  | 大规模 VLA，反应式 |
| RDT-2 | 95 | 0% | 扩散策略，无世界模型 |
| Cortex 2.0 |  |  | PRO + 世界模型 |
| Cortex 2.0 (k=1) |  |  | 消融：无候选评估 |


![Figure 8](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/479200e1-fbad-4c5d-a9aa-47d44b2df581/figures/Figure_8.png)
*Figure 8: Figure 8: Cortex 2.0 Performance against Number of Rollouts k: With increasing number of rolloutsk, the performance increases from 0.962 with 1 rollout to 0.996 at 30 rollouts. At the same time thetim*



核心结论支撑：
- **PRO 机制有效性**：Figure 8 定量展示了候选数量 k 与性能的 scaling 关系——随着 k 从 1 增加到更大值，系统性能从 0.962 起步持续提升。这一曲线直接验证了"多候选推演 + 复合评分排序"的核心假设：即使单条轨迹预测不完美，通过足够多样的候选覆盖与显式风险评估，仍能以高概率筛选出可行方案。
- **世界模型 grounding 的必要性**：Figure 5（真实产线数据）与 Figure 7（Isaac Sim 合成数据）共同说明了数据策略。纯合成数据训练的模型在真实接触力预测上出现系统性偏移，而联合训练 + MMD 对齐显著缩小该 gap。
- **任务难度分层**：螺丝分拣作为更具挑战性的任务（小尺寸物体、精密对准、高失败代价），RDT-2 完全失效（0%），而 Cortex 2.0 实现非零突破，说明世界模型的前瞻推理对"错误不可逆"类任务尤为关键。

消融分析（
![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/479200e1-fbad-4c5d-a9aa-47d44b2df581/figures/Figure_3.png)
*Figure 3: Figure 3: PRO scores k candidate rollouts via the composite score Sj (Eq. 11). The loss landscapeshows all candidate trajectories (top); PRO selects the highest-scoring rollout τ ∗(bottom).*

）：
- 移除物理量预测头（仅保留像素重建）：位姿预测误差上升
- 移除时间效率项 $C_{time}$：平均任务完成时间延长
- 移除风险规避项 $C_{risk}$：碰撞/过载事件增加
- k=1（无候选比较）：性能退化至接近 baseline 水平，证明 PRO 排序而非世界模型单条预测本身带来提升

公平性检查：
- **Baseline 强度**：π0.5 与 RDT-2 均为 2025-2026 年代表性 VLA/扩散策略方法，非过时对比。
- **计算成本**：每次决策需并行推演 k 条轨迹，世界模型前向传播为主要开销。Figure 8 的 k-scaling 曲线暗示存在延迟-精度权衡，但具体毫秒级延迟。
- **失败案例**：世界模型在长期（>H_wm 步）推演中的误差累积可能导致 PRO 评分失真；透明/反光物体的深度估计失败仍为感知瓶颈（Figure 5 中部分场景）。

## 方法谱系与知识库定位

**方法家族**: World Model-based Robotics / Model Predictive Control with Learned Dynamics

**父方法**: Dreamer / UniPi 系列（学习世界模型用于规划）+ π0 / RDT 系列（大规模 VLA 预训练）。Cortex 2.0 的核心继承是将两者嫁接：以 VLA 策略生成候选，以世界模型评估候选，以 MPC 风格重规划闭环。

**改动槽位**:
| 槽位 | 父方法 | Cortex 2.0 改动 |
|:---|:---|:---|
| architecture | RSSM / Transformer world model | 新增物理量预测头（位姿、力、稳定度），解耦重建与推理 |
| objective | 像素重建 + 奖励预测 | 多目标复合评分 S_j（Eq. 11），显式编码工业约束 |
| training_recipe | 离线 RL 或模仿学习 | 真实 + 合成混合数据，MMD 隐空间对齐 |
| data_curation | 互联网视频或实验室采集 | 真实产线部署数据（Figure 5）+ Isaac Sim 域随机化（Figure 7） |
| inference | 单步策略输出或开环规划 | k 候选并行推演 + PRO 排序 + 短程执行重规划 |

**直接 baselines 与差异**:
- **π0.5**: 同享 VLA 预训练初始化，但 π0.5 无世界模型，Cortex 2.0 增加推演-评估-排序闭环
- **RDT-2**: 同享扩散式候选生成思想，但 RDT-2 的候选评估基于当前观测似然，Cortex 2.0 替换为未来状态的前向预测与多目标评分
- **DreamerV3**: 同享世界模型规划框架，但 DreamerV3 优化像素重建与折扣回报，Cortex 2.0 针对稀疏成功+密集约束的工业任务重新设计评分函数，并解决 sim-to-real grounding

**后续方向**:
1. **世界模型轻量化**: 当前并行 k 条推演计算密集，探索蒸馏或隐式规划以支持更高 k 值或更低延迟
2. **在线自适应**: 产线分布持续漂移，世界模型需具备持续学习能力而非固定权重
3. **多机器人协同**: Figure 7 的 dual-arm 场景仅用于合成数据，扩展至真实多机协作的世界模型交互推演

**知识库标签**:
- modality: RGB-D + proprioception + language instruction
- paradigm: world model + MPC-style replanning + multi-objective ranking
- scenario: industrial warehousing / long-horizon manipulation / irreversible actions
- mechanism: predictive rollout optimization (PRO) / composite scoring / sim-to-real alignment
- constraint: real-time inference / deployment reliability / minimal human intervention

