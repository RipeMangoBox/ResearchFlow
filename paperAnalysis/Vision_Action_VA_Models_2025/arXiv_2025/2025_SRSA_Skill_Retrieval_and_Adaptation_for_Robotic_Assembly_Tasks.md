---
title: "SRSA: Skill Retrieval and Adaptation for Robotic Assembly Tasks"
venue: ICLR
year: 2025
tags:
  - Embodied_AI
  - task/robotic-assembly
  - skill-retrieval
  - self-imitation-learning
  - reinforcement-learning
  - dataset/AutoMate
  - opensource/no
core_operator: 联合编码装配任务的几何、动力学与拆卸动作特征，预测源技能对目标任务的零样本迁移成功率来检索初始化策略，再用自模仿强化学习完成快速适配。
primary_logic: |
  新装配任务的CAD/点云与可程序生成的拆卸轨迹 + 既有装配技能库 → 编码源/目标任务的几何、动力学、动作特征并预测零样本迁移成功率，检索top-k技能后以零样本试跑校验，再用PPO+self-imitation微调 → 高样本效率的新任务装配策略与可持续扩展的技能库
claims:
  - "在 AutoMate 的 10 个未见装配任务上，SRSA 在 dense-reward 设定下把最终平均成功率从 69.4% 提升到 82.6%，并将达到目标成功率所需训练轮次至少减少 2.4× [evidence: comparison]"
  - "在 sparse-reward、无演示且无课程学习的设定下，SRSA 的平均成功率从 30.1% 提升到 70.9%，相对提升 135% [evidence: comparison]"
  - "在从 10 个初始技能逐步扩展到覆盖 100 个任务的 continual learning 设定中，SRSA 平均将样本效率提升 84%，并把跨 100 个任务的平均成功率提升到 79%，高于基线的 70% [evidence: comparison]"
related_work_position:
  extends: "AutoMate (Tang et al. 2024)"
  competes_with: "AutoMate (Tang et al. 2024); Behavior Retrieval (Du et al. 2023)"
  complementary_to: "R3M (Nair et al. 2022); DINOv2 (Oquab et al. 2023)"
evidence_strength: moderate
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_SRSA_Skill_Retrieval_and_Adaptation_for_Robotic_Assembly_Tasks.pdf
category: Embodied_AI
---

# SRSA: Skill Retrieval and Adaptation for Robotic Assembly Tasks

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2503.04538) · [Project](https://srsa2024.github.io/)
> - **Summary**: 这篇工作把“新装配任务该从技能库里挑哪个旧策略来微调”显式建模为迁移成功率预测问题，并结合自模仿强化学习，让接触丰富装配中的迁移更快、更稳。
> - **Key Performance**: AutoMate 未见任务上 dense-reward 平均成功率 82.6% vs 69.4%；达到给定成功阈值所需训练轮次至少减少 2.4×

> [!info] **Agent Summary**
> - **task_path**: 目标装配任务的CAD/点云+拆卸轨迹+已有技能库策略 -> 新任务装配控制策略
> - **bottleneck**: 接触丰富装配里“看起来像”的旧任务不一定是“最容易微调”的旧策略，错误初始化会放大稀疏奖励下的探索难题和训练不稳定
> - **mechanism_delta**: 用几何/动力学/专家动作三路任务表征直接预测零样本迁移成功率，并以检索到的 specialist policy 初始化 PPO，再用自模仿放大早期高回报行为
> - **evidence_signal**: 未见任务上 dense 82.6% vs 69.4%，sparse 70.9% vs 30.1%，真实机器人 5 任务平均成功率 90%
> - **reusable_ops**: [task-pair transfer predictor, top-k retrieve-then-verify, replay-buffer self-imitation]
> - **failure_modes**: [skill library 覆盖不足导致无可迁移初始化, 几何相似但接触动力学或摩擦失配时检索偏差]
> - **open_questions**: [如何扩展到视觉观测与更强通用策略, 如何处理旋转或螺旋类复杂装配]

## Part I：问题与挑战

这篇论文真正解决的，不是“如何从零训练一个装配策略”，而是：

**当你已经有很多旧装配技能时，怎样选出那个最值得迁移、最容易微调的技能？**

### 1) 硬问题是什么
装配任务比常见 pick-and-place 更难，难在三点：

- **接触丰富**：成败取决于插入时的细微接触过程，不只是大致到位。
- **精度要求高**：几何小差异、摩擦差异、初始姿态差异都会放大为失败。
- **探索极难**：尤其在 sparse reward 下，从随机策略几乎碰不到成功轨迹。

所以，单纯“从零学”很低效；但“随便找个相似旧策略来微调”也不够，因为**几何相似不等于可迁移性相似**。

### 2) 真正瓶颈在哪里
论文把瓶颈定位得很准：

- 过去 assembly 里很多方法是**单任务 specialist 从零训练**。
- 即使现在有技能库，**怎么从库里挑对源策略**仍然是开放问题。
- 对装配这种短时程但高精度任务来说，**初始化选错**，后续 PPO 微调会很慢、很不稳，甚至在 sparse reward 下完全学不起来。

换句话说，**瓶颈不是缺技能，而是缺“迁移可用性评估器”**。

### 3) 输入/输出接口
- **输入**：
  - 目标装配任务的 CAD / mesh / point cloud
  - 目标任务可程序生成的拆卸轨迹
  - 一个已有 skill library：每个旧任务对应一个 specialist policy
- **输出**：
  - 适用于该新装配任务的控制策略
- **策略接口**：
  - 论文采用的是**状态/物体位姿输入**，不是原始视觉
  - 动作为 6D 增量位姿目标，交给阻抗控制器执行

### 4) 边界条件
该方法的成立依赖几个明确边界：

- 任务是 **two-part assembly / insertion**
- 各任务共享相同 state/action space
- 任务之间主要变化在：
  - 部件几何
  - 接触动力学
  - 初始状态分布
- 新任务**没有 assembly expert demos**，但可以生成**disassembly trajectories**

### 5) 为什么现在值得做
因为 AutoMate 提供了 100 个多样装配任务、资产 mesh 和拆卸轨迹，这让“构建技能库 + 学迁移预测器”第一次变得系统可做。也就是说，**数据基础设施出现了，skill retrieval 才从想法变成方法。**

---

## Part II：方法与洞察

这篇论文的方法观很明确：

**不要训练一个包打天下的大而全策略；先在技能库里找一个最可能迁移成功的专家，再做局部适配。**

### 核心直觉

#### what changed
从“按几何相似度/轨迹相似度检索”改成“**直接预测某个源策略对目标任务的零样本迁移成功率**”。

#### which bottleneck changed
这一步改变的是**信息瓶颈**：

- 只看几何，会漏掉接触区动力学差异
- 只看行为轨迹，会漏掉几何约束
- 新任务没有专家装配演示，传统行为表征拿不到

SRSA 用三类互补信号来补这个缺口：

- **几何**：零件形状是否相近
- **动力学**：接触/状态转移模式是否接近
- **专家动作**：任务求解策略是否相近

而且后两者不是靠难拿的装配演示，而是靠**容易生成的拆卸轨迹**来学。

#### what capability changed
这带来的能力变化是：

- 检索到的初始化策略更“接近可微调解”
- PPO 不再从很差起点开始
- 自模仿能把早期偶然成功迅速固化下来
- 在 sparse reward 下尤其明显

#### why this design works
作者的因果逻辑是：

1. 若一个源策略在目标任务上**零样本成功率高**，说明它在目标任务上的有效行为已经不差。
2. 这通常意味着源任务和目标任务在**动力学/初始分布/求解方式**上足够接近。
3. 那么 fine-tuning 只需要做**局部修正**，而不是重新发现整套策略。
4. 检索到的好初始化会在训练初期产生少量高回报轨迹；
5. **self-imitation** 再把这些高回报行为反复强化，降低 on-policy RL 的波动。

### 方法拆解

#### 1. Skill Retrieval：先学“任务是什么”，再学“能不能迁移”
SRSA 不直接把任务塞进一个黑箱相似度函数，而是先分解任务表征：

- **Geometry feature**
  - 用 PointNet autoencoder 编码 plug、socket 及装配后几何
- **Dynamics feature**
  - 用拆卸轨迹的 transition segment 预测 next state
  - 让 latent 学到任务的动态结构
- **Action feature**
  - 用拆卸轨迹重建动作序列
  - 让 latent 学到“专家怎么解这个任务”

然后把源任务和目标任务的这三类特征拼起来，交给一个 MLP，预测：

- **这个源策略在目标任务上的 zero-shot transfer success**

测试时不是只信 predictor 一次输出，而是：

1. 对每个候选源技能做多次采样平均预测
2. 取 **top-k**
3. 再在目标任务上做零样本试跑验证
4. 从 top-k 里选真正 transfer success 最好的技能

这一步很关键：**它把 predictor 的误差风险降了一层。**

#### 2. Skill Adaptation：不是只初始化，还要稳住微调
选好技能后，用它初始化 PPO 的策略网络，再加一个简单但有效的增强：

- 维护 replay buffer
- 按回报优先采样
- 对高回报 state-action 做 **self-imitation learning**

它的作用不是替代 RL，而是：

- 把“检索带来的早期正确行为”尽量留住
- 避免 PPO 在后续更新中把刚找到的好行为冲掉
- 尤其缓解 sparse reward 下的高方差问题

#### 3. Continual Learning：技能库会越用越强
SRSA 不把 skill library 当静态资源，而是：

- 新任务学会后，把新 policy 加回 skill library
- 后续任务再从更大的库里检索
- 形成“**学习新任务 → 扩库 → 更容易学未来任务**”的闭环

### 战略取舍

| 设计选择 | 带来的收益 | 代价 / 假设 |
| --- | --- | --- |
| 几何 + 动力学 + 动作三路任务表征 | 比纯几何检索更接近真实可迁移性 | 需要 CAD 和拆卸轨迹，且要额外训练表征模型 |
| 预测 zero-shot transfer success | 用低成本代理目标找“最易微调”的源技能 | 代理指标不是最终 fine-tune 上限，存在偏差 |
| top-k 检索后再零样本验证 | 降低 predictor 误判风险 | 多一次候选评估开销 |
| retrieved specialist + PPO + SIL | 稀疏奖励下更快、更稳 | 依赖初始技能至少能产生部分正向轨迹 |
| 持续扩展 skill library | 长期样本效率越来越高 | 早期库太小时，覆盖不足仍会限制效果 |
| 使用状态/位姿而非视觉输入 | 简化 sim-to-real，突出迁移机制 | 不能直接证明对视觉端到端策略同样有效 |

---

## Part III：证据与局限

### 关键证据

#### 1. 检索质量确实更好
最直接的信号不是最终训练曲线，而是**检索出来的源技能本身更可迁移**：

- 在 10 个未见测试任务上，SRSA 检索到的技能零样本 transfer success **整体约高 20%**
- 除一个极难任务外，SRSA 基本都能做到最好或次好

这说明它学到的不是表面相似度，而是更接近“可微调性”的信号。

#### 2. 适配不仅更高，而且更稳
在 dense-reward 设定下：

- SRSA：**82.6%**
- AutoMate：**69.4%**

而且：

- 达到目标成功率所需训练轮次至少减少 **2.4×**
- 跨随机种子标准差降低 **2.6×**

这两个信号合起来很重要：  
**不是只把上限抬高，而是把训练过程变得更可靠。**

#### 3. 稀疏奖励下优势更大
在更贴近真实微调条件的 sparse-reward 设定里：

- SRSA：**70.9%**
- AutoMate：**30.1%**

相对提升 **135%**。  
这基本说明：**检索初始化 + self-imitation** 对解决探索问题是核心有效的，而不仅仅是 dense reward 下的“小优化”。

#### 4. 消融支持了机制因果链
论文的 ablation 不是装饰性的，能支撑机制解释：

- **只用几何检索**：起点更差，适配更慢、更不稳  
  → 说明装配迁移不能只看形状
- **去掉 self-imitation**：曲线波动更大、方差更高  
  → 说明检索后的“好起点”需要被显式保留
- **用 generalist policy 初始化**：比检索到的 specialist 更弱  
  → 说明对这类高精度装配，局部专精的起点比小容量通才更有用

#### 5. 不只在仿真里成立
真实机器人上，直接把仿真里 fine-tuned 的策略零样本部署：

- SRSA：**90% 平均成功率**
- AutoMate：**54%**

这说明它不是只在 benchmark 上“学会了分数”，而是把更好的初始化和更稳定的策略带到了真实系统。

#### 6. 对长期扩库也有效
在 continual learning 设置中：

- 学习 90 个新任务时平均样本效率提升 **84%**
- 覆盖 100 个任务后的平均成功率 **79% vs 70%**

所以 SRSA 的价值不只是一次性迁移，而是**把 skill library 变成一个会增值的资产**。

### 局限性

- **Fails when**: 技能库对目标任务覆盖不足、没有任何可零样本迁移的近邻技能时，检索初始化帮助有限；几何相似但摩擦或接触区动力学明显不同时，检索可能偏差；论文也明确未覆盖旋转/螺旋类装配（如 nut-and-bolt）。
- **Assumes**: 需要可获得的 CAD / mesh、可程序生成的拆卸轨迹、以及每个源任务已有 specialist policy；训练迁移预测器需要在仿真中离线评估大量源-目标对；策略输入依赖物体位姿与本体状态而不是原始视觉；论文文本未明确给出公开代码链接，复现仍依赖实现细节和 AutoMate 资产。
- **Not designed for**: 长时序多阶段装配、端到端视觉装配、以及直接满足工业级 99%+ 成功率要求的生产部署。

### 可复用部件

这篇论文最值得迁移出去的，不是某个具体网络，而是这几个操作模式：

- **task-pair transfer predictor**：把“选哪个源技能”建模为源任务-目标任务对的可迁移性预测
- **disassembly-as-supervision**：当 assembly demo 难拿时，用可生成的拆卸轨迹学习动态与行为特征
- **top-k retrieve-then-verify**：先用便宜预测缩小候选，再用真实试跑兜底
- **retrieval-initialized self-imitation RL**：先拿到好初始化，再用 replay 中高回报片段稳住微调
- **continual skill library expansion**：把新学到的专家策略反哺技能库，形成长期复利

一句话总结“So what”：

**SRSA 的能力跃迁不在于训练了更大的装配模型，而在于把“技能库怎么被正确利用”这件事做成了一个可学习、可扩展、可验证的系统。**

## Local PDF reference

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_SRSA_Skill_Retrieval_and_Adaptation_for_Robotic_Assembly_Tasks.pdf]]