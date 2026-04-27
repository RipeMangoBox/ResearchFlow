---
title: Reinforcement Learning via Value Gradient Flow
type: paper
paper_level: B
venue: arXiv
year: 2026
paper_link: https://arxiv.org/abs/2604.14265
aliases:
- 值梯度流：无策略网络的行为正则化RL
- RLVGF
- VGF的核心洞察是：行为正则化不必通过显式惩罚项实现
paradigm: Reinforcement Learning
---

# Reinforcement Learning via Value Gradient Flow

[Paper](https://arxiv.org/abs/2604.14265)

**Topics**: [[T__Agent]], [[T__Reinforcement_Learning]]

> [!tip] 核心洞察
> VGF的核心洞察是：行为正则化不必通过显式惩罚项实现，传输距离本身就是正则化。通过控制粒子从参考分布出发能走多远（传输预算），自然地限制了策略偏离行为分布的程度。这使得训练时正则化和推理时正则化可以解耦——训练时用小步数保证稳定性，推理时可以用更大步数换取更高性能，无需重新训练。同时，用TD学习而非in-sample学习训练Q函数，使值函数具备跨状态拼接能力，这是VGF能真正超越行为支撑集的关键。

| 属性 | 内容 |
|------|------|
| 中文题名 | 值梯度流：无策略网络的行为正则化RL |
| 英文题名 | Reinforcement Learning via Value Gradient Flow |
| 会议/期刊 | arXiv (Cornell University) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.14265) · [Code] · [Project] |
| 主要任务 | 离线RL、离线到在线微调、RLHF中的行为正则化策略优化 |
| 主要 baseline | FQL (Flow Q-Learning)、Diffusion-QL、IDQL、SfBC、ReBRAC、IQL |

> [!abstract] 因为「行为正则化RL中显式策略梯度反向传播穿越采样链不稳定、正则化系数难以调参且训练/推理耦合」，作者在「FQL等基于重参数化策略的方法」基础上改了「消除显式策略网络，将行为正则化重新表述为从参考分布到Boltzmann最优分布的最优传输问题，用离散梯度流迭代更新粒子」，在「D4RL AntMaze + OGBench操作任务」上取得「antmaze-m-d 86.7 vs FQL 71.0；cube-double 70±8 vs FQL 29±2」

**关键性能**
- OGBench cube-double: 70±8（FQL 29±2），提升141%
- OGBench puzzle-3x3: 75±4（FQL 30±1），提升150%
- D4RL antmaze-m-d: 86.7（FQL 71.0），提升22%

## 背景与动机

行为正则化强化学习的核心矛盾在于：策略既要最大化价值，又要忠实于参考分布（离线数据集中的行为策略或SFT模型），否则分布外（OOD）动作会导致值函数过估计和奖励黑客。以机器人操作任务为例，若智能体完全脱离人类示教数据的分布去探索，可能学会用物理上不可行的方式完成任务，而值函数却错误地赋予其高分。

现有方法分为两大类。**第一类：重参数化策略梯度方法**，如Diffusion-QL和FQL。它们用扩散模型或流匹配模型显式参数化策略，并通过KL散度或L2距离惩罚约束策略偏离行为分布。这类方法能表达复杂多模态策略，但面临两个根本性困难：梯度需反向传播穿越多步采样链（扩散链或流匹配链），数值不稳定且计算代价高昂；单一正则化系数同时约束值学习和策略改进，调参困难，要么过于保守要么OOD漂移失控。**第二类：拒绝采样方法**，如IDQL和SfBC。它们从行为策略采样后用值函数筛选动作。这类方法稳定，但本质上只能在行为支撑集内选择，无法真正超越行为分布，在困难任务上表现保守。

更深层的局限是，上述方法普遍将训练时正则化强度与推理时正则化强度绑定。例如，FQL训练时设定的KL系数直接决定了推理时的策略分布，无法在不重新训练的情况下灵活调整测试时计算预算。这一问题在大规模生成模型（如LLM RLHF）场景下尤为突出——显式策略梯度方法难以扩展到数十亿参数的语言模型。

本文提出VGF（Value Gradient Flow），核心思想是：将行为正则化重新表述为最优传输问题，完全消除显式策略网络，通过控制"传输预算"实现训练与推理正则化的解耦。

## 核心创新

**核心洞察：行为正则化不必通过显式惩罚项实现，传输距离本身就是正则化。** 因为从参考分布出发的粒子只能沿值梯度方向移动有限步数，偏离行为分布的程度自然受限于传输预算（流步数L和步长ε），从而使训练时正则化与推理时正则化解耦成为可能。

| 维度 | Baseline (FQL/Diffusion-QL) | 本文 (VGF) |
|------|---------------------------|-----------|
| 策略表示 | 显式策略网络（扩散/流匹配模型） | 隐式粒子集合（N个动作粒子） |
| 正则化机制 | 显式KL/L2惩罚系数α | 隐式传输预算（步数L × 步长ε） |
| 梯度传播 | 反向传播穿越采样链 | 无需反向传播，直接梯度更新粒子 |
| 训练/推理耦合 | 训练α固定推理分布 | 训练用小L，推理可增大L，无需重训练 |
| 值函数学习 | In-sample学习（仅数据集内状态-动作对） | TD学习，跨粒子取平均目标Q值 |
| 扩展到LLM | 困难（需策略梯度） | 一阶梯度引导SFT策略，无需显式RL循环 |

## 整体框架


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/29f36cbb-6f40-46af-a5e3-06495689df2b/figures/Figure_1.png)
*Figure 1: Figure 1: VGF: Value Gradient Flow. VGF reframes behavior-regularized RL as an optimal transportfrom the behavior distribution towards the Boltzmann value distribution, with the transport budget as im*



VGF框架的数据流如下：

**输入**：状态s，参考分布π_ref（离线数据集的行为策略或SFT模型）

**Step 1: 粒子初始化** → 从参考分布采样N个动作粒子：{a^0_i ~ π_ref(·|s)}_{i=1}^N。实践中N=5，远少于Best-of-N常用的N=20。

**Step 2: 梯度流迭代（L步）** → 对每个粒子，用Q函数的动作梯度迭代更新：a^{l+1}_i = update(a^l_i, ∇_{a^l_i} Q(s, a^l_i))。更新规则可为梯度上升或带噪声的Langevin动力学。

**Step 3: 隐式策略形成** → 经过L步后，N个粒子的均匀混合构成隐式策略：π_VGF(·|s) = Unif{a^L_i}_{i=1}^N。

**Step 4: 评估时动作选择** → 从所有L步后的粒子中，用值函数做Best-of-N选择最终动作：a* = argmax_{a ∈ {a^L_i}} Q(s,a)。

**辅助模块**：梯度网络f(s,a) ≈ ∇_a Q(s,a)，用于加速训练和推理，避免每次计算Q网络梯度。

**Q函数训练**：使用标准TD学习，而非in-sample学习。关键设计：TD目标对所有粒子的目标Q值取平均，增强泛化能力。

```
参考分布 π_ref ──→ 采样N粒子 {a^0_i} 
                        ↓
              [梯度网络 f(s,a) ≈ ∇_a Q]
                        ↓
              L步梯度流: a^{l+1} = update(a^l, ∇_a Q)
                        ↓
              隐式策略: Unif{a^L_i} ──→ Best-of-N选择 ──→ 输出动作a*
                        ↑
              Q函数 (TD学习, 跨粒子平均目标Q值)
```

## 核心模块与公式推导

### 模块 1: 离散梯度流更新（对应框架图 Step 2）

**直觉**：将策略改进视为粒子在动作空间中的物理运动，沿值函数上升方向移动，运动距离自然限制策略偏离。

**Baseline 公式** (FQL的重参数化策略梯度):
$$\pi_\theta(a|s) = \text{FlowMatch}_\theta(s, \epsilon), \quad \epsilon \sim \mathcal{N}(0,I)$$
$$L_{\text{FQL}} = \mathbb{E}_{s \sim \mathcal{D}, a \sim \pi_\theta}[Q(s,a)] - \alpha \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$$
符号: $\theta$ = 策略网络参数, $\alpha$ = KL惩罚系数, $D_{\text{KL}}$ = KL散度

**变化点**：FQL需要梯度$\nabla_\theta$反向传播穿越流匹配采样链，数值不稳定；且$\alpha$同时约束训练和推理。VGF消除$\theta$，直接用粒子坐标作为可优化变量。

**本文公式（推导）**:
$$\text{Step 1}: a^0_i \sim \pi_{\text{ref}}(\cdot|s), \quad i=1,...,N \quad \text{从参考分布初始化粒子}$$
$$\text{Step 2}: a^{l+1}_i = a^l_i + \varepsilon \cdot \nabla_a Q(s, a^l_i) \quad \text{确定性梯度上升步}$$
$$\text{或带噪声版本}: a^{l+1}_i = a^l_i + \varepsilon \cdot \nabla_a Q(s, a^l_i) + \sqrt{2\varepsilon} \cdot \xi, \quad \xi \sim \mathcal{N}(0,I) \quad \text{Langevin动力学，保证收敛到Boltzmann分布}$$
$$\text{最终}: \pi_{\text{VGF}}(a|s) = \frac{1}{N}\sum_{i=1}^N \delta(a - a^L_i) \quad \text{隐式策略为粒子经验分布}$$

**对应消融**：

---

### 模块 2: 基于粒子平均的TD学习（对应框架图 Q函数训练）

**直觉**：传统in-sample学习仅用数据集中的(s,a)对训练Q函数，无法泛化到OOD动作；VGF通过粒子集合生成大量候选动作，对其目标Q值取平均，迫使Q函数在更广阔的动空间上保持准确。

**Baseline 公式** (IQL/FQL的in-sample学习):
$$y = r + \gamma \cdot Q_{\text{target}}(s', a'), \quad a' \sim \pi_{\text{ref}}(\cdot|s')$$
$$L_{\text{in-sample}} = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}}[(Q(s,a) - y)^2]$$
符号: $y$ = TD目标, $\gamma$ = 折扣因子, $\mathcal{D}$ = 离线数据集

**变化点**：in-sample学习仅用参考分布采样的下一个动作$a'$，Q函数从未见过梯度流后的粒子位置，导致在推理时对这些OOD动作的值估计不准。VGF用当前策略（粒子集合）生成下一个状态的动作候选。

**本文公式（推导）**:
$$\text{Step 1}: \{a'^l_j\}_{j=1}^N = \text{GradientFlow}(s', \pi_{\text{ref}}, L) \quad \text{对下一状态}s'\text{执行L步梯度流}$$
$$\text{Step 2}: y_i = r + \gamma \cdot \frac{1}{N}\sum_{j=1}^N Q_{\text{target}}(s', a'^L_j) \quad \text{对所有粒子的目标Q值取平均}$$
$$\text{最终}: L_{\text{TD}} = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}, \{a^l_i\} \sim \text{Flow}}\left[\frac{1}{N}\sum_{i=1}^N (Q(s, a^L_i) - y_i)^2\right]$$

关键设计：即使训练时用小L，Q函数也学习评估流后粒子的值，使得推理时增大L仍能给出可靠值估计。

**对应消融**：原文未提供TD目标计算方式的直接消融，但指出TD学习（vs in-sample）是VGF能超越行为支撑集的关键（Part II）。

---

### 模块 3: 辅助梯度网络（对应框架图 梯度网络f(s,a)）

**直觉**：每次梯度流迭代都需计算$\nabla_a Q(s,a)$，对大型Q网络计算昂贵；预训练一个小型梯度网络加速推理。

**Baseline**：无对应，标准做法直接计算Q网络梯度。

**本文公式**:
$$f_\phi(s,a) \approx \nabla_a Q(s,a)$$
$$L_{\text{grad}} = \mathbb{E}_{(s,a) \sim \mathcal{D}}\left[\|f_\phi(s,a) - \nabla_a Q(s,a)\|^2\right]$$

梯度流更新改为：$a^{l+1}_i = a^l_i + \varepsilon \cdot f_\phi(s, a^l_i)$

**局限**：原文指出"辅助梯度网络的近似误差对最终性能的影响未被系统分析"（Part III）。

## 实验与分析


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/29f36cbb-6f40-46af-a5e3-06495689df2b/figures/Figure_2.png)
*Figure 2: Figure 2: Toycase results. VGF generates actions with higher ground-truth reward than other methods.*



**主结果表：OGBench 困难操作任务**

| Method | cube-double | puzzle-3x3 | puzzle-4x4 | antmaze-giant |
|--------|-------------|-----------|-----------|---------------|
| FQL | 29 ± 2 | 30 ± 1 | 17 ± 2 |  |
| ReBRAC |  |  |  | 26 ± 8 |
| **VGF** | **70 ± 8** | **75 ± 4** | **45 ± 4** | **3 ± 1** |

**D4RL AntMaze导航**

| Method | antmaze-m-p | antmaze-m-d |
|--------|-------------|-------------|
| FQL | 78.0 | 71.0 |
| **VGF** | **89.4** | **86.7** |

**核心结论分析**：
- **支持核心claim的数据**：OGBench操作任务上VGF大幅领先（cube-double 141%提升，puzzle-3x3 150%提升），证明梯度流能有效探索行为分布之外的高值区域，TD学习的Q函数具备强泛化能力。
- **边际/负面结果**：antmaze-giant任务VGF仅3±1，远低于ReBRAC的26±8。这说明在**大规模稀疏奖励导航任务**上，Q函数外推误差导致梯度流方向误导——粒子被推向错误的高值幻觉区域。这验证了VGF的测试时退化机制：当值函数不可靠时，应设L=0退化为Best-of-N。


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/29f36cbb-6f40-46af-a5e3-06495689df2b/figures/Figure_3.png)
*Figure 3: Figure 3: OGBench offline-to-online RL results. Learning curves for online fine-tuning of VGF and FQLacross all default tasks. VGF not only provides a stronger initialization from offline training but*



**消融与机制分析**：
- 粒子数N=5远少于Best-of-N的N=20，但结合L步梯度流后性能更优，说明"少粒子+梯度优化"优于"多粒子+纯采样"。
- 离线到在线微调实验（Figure 3）显示VGF提供更强的离线初始化并收敛更快，但**仅与FQL对比**，缺乏与IQL、ReBRAC等的全面对比。

**公平性检查**：
- **基线强度**：FQL是当前SOTA流匹配RL方法，但ReBRAC等更强基线未在所有任务上对比。
- **计算成本**：VGF推理时需L×N次Q（或梯度网络）前向传播，L步梯度流增加推理计算；但消除了策略网络训练，整体训练更稳定。
- **失败案例**：antmaze-giant明确展示了Q函数外推误差的危害；RLHF实验缺乏正文数值，证据不对称。

## 方法谱系与知识库定位

**方法家族**：行为正则化离线RL → 最优传输/粒子方法视角下的策略优化

**父方法**：FQL (Flow Q-Learning, 2024)。VGF继承其流匹配/连续动作建模的动机，但彻底放弃显式流匹配策略网络，将"流"从策略参数空间转移到动作粒子空间。

**改变的插槽**：
- **架构**：消除策略网络，改为粒子集合隐式表示
- **目标函数**：消除显式KL/L2惩罚，改为传输预算约束
- **训练流程**：Q函数从in-sample学习改为TD学习+跨粒子平均目标
- **推理机制**：引入测试时缩放（test-time scaling），L可独立于训练调整
- **数据策划**：无变化，仍使用标准离线数据集

**直接基线对比**：
- **FQL/Diffusion-QL**：VGF消除反向传播穿越采样链，解耦训练/推理正则化
- **IDQL/SfBC**：VGF通过梯度流超越行为支撑集，而非仅拒绝采样
- **ReBRAC**：ReBRAC在antmaze-giant上更鲁棒，VGF在操作任务上更优，体现不同正则化机制的适用场景差异
- **Best-of-N**：VGF训练时L>0使Q函数学习评估流后粒子，推理时L=0即退化为Best-of-N，是严格泛化

**后续方向**：
1. **自适应传输预算**：根据状态依赖的不确定性动态调整L，而非全局固定
2. **与LLM RLHF的深度整合**：将梯度流思想扩展到离散token空间，验证十亿参数规模的有效性
3. **Q函数不确定性量化**：利用梯度网络f(s,a)的近似误差或集成Q网络，检测外推区域并自动退化

**知识库标签**：
- Modality: 连续控制 / 潜在可扩展至离散语言
- Paradigm: 离线RL / 离线到在线微调 / RLHF
- Scenario: 机器人操作、导航、大规模生成模型对齐
- Mechanism: 最优传输、梯度流、粒子方法、隐式策略
- Constraint: 行为正则化、分布外泛化、测试时计算预算解耦

