---
title: 'TEMPO: Scaling Test-time Training for Large Reasoning Models'
type: paper
paper_level: B
venue: Unknown
year: 2026
paper_link: https://arxiv.org/abs/2604.19295
aliases:
- 测试时训练的EM扩展框架TEMPO
- TEMPO
- 现有TTT方法的失败根源在于用一个会随策略演化而漂移的自生成信号来训练
method: TEMPO
paradigm: Reinforcement Learning
---

# TEMPO: Scaling Test-time Training for Large Reasoning Models

[Paper](https://arxiv.org/abs/2604.19295)

**Topics**: [[T__Math_Reasoning]], [[T__Self-Supervised_Learning]] | **Method**: [[M__TEMPO]]

> [!tip] 核心洞察
> 现有TTT方法的失败根源在于用一个会随策略演化而漂移的自生成信号来训练策略本身——这是一个没有外部锚点的自我强化循环。TEMPO的洞察是：critic和actor必须分离，且critic需要周期性地用有标注数据重新校准，才能在策略能力提升后继续提供准确的奖励信号。从EM视角看，这就是补全被省略的E步。有效性来自于：外部标注数据提供了一个稳定的真值锚点，防止奖励估计随策略漂移，从而将测试时计算转化为持续的性能增益而非多样性崩溃。

| 中文题名 | 测试时训练的EM扩展框架TEMPO |
| 英文题名 | TEMPO: Scaling Test-time Training for Large Reasoning Models |
| 会议/期刊 | arXiv 2026 (预印本) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.19295) · [Code](https://github.com/BytedTsinghua-SIA/DAPO) ⭐待补充) · [Project](待补充) |
| 主要任务 | 数学推理（AIME 2024/2025/Beyond AIME）、通用推理（BBH/AGI Eval/ZebraLogic/GPQA-Diamond） |
| 主要 baseline | TTRL、EMPO、SFT、PPO-continue |

> [!abstract]
> 因为「现有测试时训练（TTT）方法依赖自生成奖励信号，导致奖励漂移和多样性崩溃，性能在100-200步内迅速plateau」，作者在「TTRL/EMPO等仅执行M步的退化EM变体」基础上改了「补全E步（critic重校准），交替执行E-step与M-step」，在「AIME 2024/2025」上取得「OLMO3-7B +18.1pp（33.0%→51.1%），Qwen3-14B +23.5pp（42.3%→65.8%）」

- **OLMO3-7B AIME 2024**: 33.0% → 51.1%（+18.1pp）
- **Qwen3-14B AIME 2024**: 42.3% → 65.8%（+23.5pp）
- **通用领域BBH**: +21.4pp，**AGI Eval**: +24.5pp（OLMO3-7B）

## 背景与动机

大型推理模型（LRM）的测试时训练（Test-Time Training, TTT）旨在通过部署阶段的额外计算提升推理能力，但当前方法面临根本性瓶颈。以AIME数学竞赛为例：模型在测试时针对新题目生成多条推理轨迹，通过自一致性投票或熵奖励筛选答案，期望迭代优化策略——然而实践中性能往往在100-200步后停滞，甚至下降。

现有方法如何处理这一问题？**TTRL**（Test-Time Reinforcement Learning）使用模型自生成的多数投票结果作为伪标签，通过PPO持续优化策略；**EMPO**（Expectation-Maximization Policy Optimization）虽以EM命名，但仅执行M步（策略精炼），用自一致性作为固定奖励信号。两者核心依赖相同机制：模型自身输出构造奖励，无需外部标注。

这些方法的致命缺陷在于**奖励漂移（reward drift）**：随着策略演化，模型对少数推理模式越来越自信，自生成奖励系统性地高估这些模式的质量，形成自我强化循环。更深层的问题是**多样性崩溃**——pass@k指标随avg@k提升而下降，模型收敛到窄推理模式集合，从根本上限制了测试时计算的可扩展性。根因在于：测试时无ground-truth标签，奖励信号必须从模型自身输出推断，缺乏外部校准锚点。从EM视角看，现有方法仅执行M步而完全省略E步（后验分布估计/critic校准），导致估计后验随训练持续偏离真实正确性分布。即便在有标注数据上继续监督PPO训练，收敛模型也几乎无法获得额外增益，说明瓶颈不在于数据量而在于分布外泛化能力——只有接触新颖测试时问题才能突破已建立的能力边界。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/ef65b40f-1c1e-4664-bbe6-86bca01cebac/figures/Figure_1.png)
*Figure 1 (motivation): Scalability of TEMPO on the AIME benchmark.*



本文提出TEMPO，核心洞察是将TTT重新形式化为完整EM算法，通过交替E-step（critic重校准）和M-step（策略精炼）打破自我强化循环，使测试时计算转化为持续的性能增益。

## 核心创新

**核心洞察：critic与actor必须分离且critic需周期性用有标注数据重校准，因为自生成奖励会随策略演化而漂移，从而使测试时计算突破100-200步的plateau、实现持续可扩展的性能提升成为可能。**

现有TTT方法的本质是一个没有外部锚点的自我强化循环：策略生成答案→用自一致性/熵构造奖励→优化同一策略→生成更自信的答案→奖励进一步漂移。TEMPO的关键突破是引入外部真值锚点：周期性地在有标注数据上重训critic，使其与当前策略的能力水平对齐，防止奖励估计随策略漂移。

| 维度 | Baseline (TTRL/EMPO) | 本文 (TEMPO) |
|:---|:---|:---|
| 模型架构 | 单一策略模型，自举奖励 | actor-critic分离，独立维护 |
| EM执行 | 仅M步（策略精炼） | 完整E-step + M-step交替 |
| 奖励来源 | 自生成信号（熵/多数投票/自一致性） | critic打分，周期性用标注数据重校准 |
| 可扩展性 | ~100-200步后plateau，多样性崩溃 | 持续改进，保留多样性 |
| 理论框架 | 启发式优化 | 原则性EM算法，ELBO收紧 |

## 整体框架


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/ef65b40f-1c1e-4664-bbe6-86bca01cebac/figures/Figure_2.png)
*Figure 2 (pipeline): TEMPO alternates between (i) Critic Recalibration (E-step) and (ii) Policy Refinement (M-step).*



TEMPO的整体流程遵循经典EM算法的交替结构，但针对测试时无标注场景进行了关键适配。数据流如下：

**输入**：无标注测试问题 $x \sim D_U$（测试时核心输入）+ 少量有标注数据 $D_L$（用于critic重校准）

**E-step（Critic Recalibration）**：
- 输入：当前actor生成的推理轨迹、有标注数据集 $D_L$
- 处理：在有标注数据上重新训练critic模型 $q_\phi(z|x)$，使其与当前策略能力对齐
- 输出：更新后的critic，提供准确的后验分布估计

**M-step（Policy Refinement）**：
- 输入：无标注测试问题、当前critic打分
- 处理：actor生成多条推理轨迹，critic评估质量，actor通过PPO优化策略参数 $p_\theta(y|x,z)$
- 输出：更新后的actor策略

**交替迭代**：E-step与M-step周期性交替执行，每次E-step将critic重新锚定到外部监督信号，防止错位积累。

关键设计细节：actor和critic均从在DAPO-Math-17K上PPO预训练的收敛模型初始化；TTT阶段batch size 256，最大响应长度16K；使用GSPO的sequence clip机制稳定离策略训练；通用领域实验使用gpt-oss-120b作为外部判断模型验证正确性。

```
初始化: actor θ₀, critic φ₀ (均来自DAPO-Math-17K预训练)
for iteration t = 1, 2, ...:
    if t mod T_E == 0:        # 周期性执行E-step
        φ_{t+1} ← argmax_φ E_{(x,y)~D_L}[log q_φ(z|x) · 𝟙[y correct]]
                              # 在有标注数据上重校准critic
    else:                     # M-step（默认每步执行）
        采样 x ~ D_U
        生成轨迹 z ~ p_{θ_t}(·|x)
        评分 r = q_{φ_t}(correct|x,z)
        θ_{t+1} ← PPO_update(θ_t, r)  # 基于critic分数优化策略
```

## 核心模块与公式推导

### 模块 1: M-step 策略精炼（对应框架图右侧）

**直觉**：在给定critic估计的后验分布下，最大化策略生成正确推理的期望对数似然。

**Baseline 公式** (TTRL/EMPO): 
$$L_{\text{base}} = \mathbb{E}_{x \sim D_U, z \sim p_{\theta}(\cdot|x)}\left[ R_{\text{self}}(x, z) \cdot \log p_{\theta}(y|x, z) \right]$$

符号: $\theta$ = 策略参数, $R_{\text{self}}$ = 自生成奖励（多数投票/熵/自一致性）, $z$ = 推理轨迹, $y$ = 最终答案

**变化点**：Baseline的 $R_{\text{self}}$ 随策略演化而漂移——模型对少数模式越自信，$R_{\text{self}}$ 系统性地高估这些模式。TEMPO将其替换为独立critic的打分，且critic经E-step重校准。

**本文公式（推导）**:
$$\text{Step 1}: q_{\phi}(z|x) \text{ 由E-step提供，替代 } R_{\text{self}} \quad \text{（分离奖励估计与策略生成）}$$
$$\text{Step 2}: \theta^* = \text{arg}\max_{\theta} \mathbb{E}_{q_{\phi}(z|x)}\left[ \log p_{\theta}(y|x, z) \right] \quad \text{（标准EM的M-step）}$$
$$\text{最终}: L_{\text{M-step}} = \mathbb{E}_{x \sim D_U, z \sim p_{\theta}(\cdot|x)}\left[ q_{\phi}(\text{correct}|x, z) \cdot \log p_{\theta}(y|x, z) \right] - \beta \cdot \text{KL}\left[p_{\theta} \| p_{\theta_{\text{ref}}}\right]$$
其中KL项为PPO的标准约束，防止策略偏离过远。

**对应消融**：Figure 6显示冻结critic在~100步后plateau，与完整TEMPO差距持续扩大。

---

### 模块 2: E-step Critic重校准（对应框架图左侧）

**直觉**：随着actor生成越来越复杂的推理路径，冻结的critic（基于早期、能力较弱的策略输出训练）无法准确评估新模式，引入噪声梯度。周期性重校准将critic重新锚定到外部监督信号。

**Baseline 公式** (TTRL/EMPO — 无E-step，critic不存在或冻结):
$$L_{\text{base}} = \text{None} \quad \text{（现有方法完全省略E-step）}$$

**变化点**：现有方法省略E步导致ELBO松弛，后验估计 $q(z|x)$ 随训练偏离真实正确性分布。TEMPO恢复完整EM循环，周期性收紧ELBO。

**本文公式（推导）**:
$$\text{Step 1}: \text{ELBO分解 } \log p(y|x) \geq \underbrace{\mathbb{E}_{q_{\phi}(z|x)}[\log p(y|x,z)]}_{\text{期望似然}} - \underbrace{\text{KL}(q_{\phi}(z|x) \| p(z|x))}_{\text{后验近似误差}}$$
$$\text{Step 2}: \phi^* = \text{arg}\max_{\phi} \mathbb{E}_{(x,y) \sim D_L}\left[ \mathbb{E}_{q_{\phi}(z|x)}[\log p_{\theta}(y|x,z)] - \text{KL}(q_{\phi}(z|x) \| p(z|x)) \right]$$
$$\text{最终}: L_{\text{E-step}} = \mathbb{E}_{(x,y) \sim D_L}\left[ \text{ell}_{\text{BCE}}\left(q_{\phi}(\text{correct}|x,z), \mathbb{1}[y = \hat{y}]\right) \right]$$
其中 $D_L$ 为有标注数据，提供外部真值锚点；实际实现中critic直接学习预测答案正确性。

**对应消融**：Figure 6中"frozen critic"曲线在~100步后flat，而交替E-step的TEMPO持续上升，验证E-step的必要性。

---

### 模块 3: 完整EM交替与收敛保证（框架整体）

**直觉**：单独E-step或M-step均不完整——仅M步导致critic-actor错位，仅E步浪费测试时计算。交替执行使两者相互追赶。

**Baseline 公式** (EMPO):
$$\theta^{(t+1)} = \text{arg}\max_{\theta} \mathbb{E}_{q^{(t)}(z|x)}[\log p_{\theta}(y|x,z)] \quad \text{（固定 } q^{(0)} \text{，永不更新）}$$

**变化点**：EMPO虽以EM命名，但 $q^{(t)}$ 永不更新，退化为纯M-step。TEMPO恢复 $q^{(t)}$ 的周期性更新。

**本文公式（推导）**:
$$\text{Step 1}: \phi^{(t+1)} = \text{arg}\max_{\phi} \mathcal{L}(\theta^{(t)}, \phi; D_L) \quad \text{（E-step，收紧ELBO）}$$
$$\text{Step 2}: \theta^{(t+1)} = \text{arg}\max_{\theta} \mathcal{L}(\theta, \phi^{(t+1)}; D_U) \quad \text{（M-step，提升策略）}$$
$$\text{最终}: \mathcal{L}_{\text{ELBO}}^{(t)} = \mathbb{E}_{q_{\phi^{(t)}}(z|x)}[\log p_{\theta^{(t)}}(y|x,z)] - \text{KL}(q_{\phi^{(t)}}(z|x) \| p(z|x))$$

**对应消融**：Figure 3显示TEMPO保留模型多样性（vs. baseline的多样性崩溃）；Figure 4显示TEMPO在更长训练步数上继续提升，验证交替机制的持续有效性。论文明确将EM交替过程的正式收敛性证明列为未来工作。

## 实验与分析

| Method | AIME 2024 (OLMO3-7B) | AIME 2025 (OLMO3-7B) | Beyond AIME (OLMO3-7B) | AIME 2024 (Qwen3-14B) |
|:---|:---|:---|:---|:---|
| SFT | 33.0 | — | — | 42.3 |
| PPO-continue | 33.0 | — | — | — |
| TTRL | 46.7 | — | — | — |
| EMPO | 45.2 | — | — | — |
| **TEMPO** | **51.1** | **50.0** | **35.0** | **65.8** |
| Δ vs. best baseline | +4.4pp | — | — | — |
| Δ vs. SFT | +18.1pp | — | — | +23.5pp |


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/ef65b40f-1c1e-4664-bbe6-86bca01cebac/figures/Table_1.png)
*Table 1 (result): Main results on mathematical reasoning benchmarks.*



**核心结果分析**：TEMPO在OLMO3-7B上相对SFT提升18.1pp（33.0%→51.1%），相对最强baseline TTRL仍提升4.4pp，说明E-step的补全确实带来额外增益。Qwen3-14B的+23.5pp（42.3%→65.8%）验证了跨模型有效性。Figure 4显示TEMPO在更长训练步数上继续提升，突破baseline的plateau。

**消融实验**：Figure 6（Necessity of alternating critic recalibration）是最关键的机制验证——冻结critic在~100步后完全停滞，与完整TEMPO差距持续扩大，直接证明周期性E-step的必要性。Figure 3（TEMPO preserves model diversity）显示pass@k指标未随avg@k提升而下降，反驳了baseline的多样性崩溃问题。


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/ef65b40f-1c1e-4664-bbe6-86bca01cebac/figures/Figure_3.png)
*Figure 3 (ablation): TEMPO preserves model diversity.*



**通用领域泛化**：Table 2显示OLMO3-7B在BBH（+21.4pp）、AGI Eval（+24.5pp）、ZebraLogic（+12.9pp）、GPQA-Diamond（+10.5pp）均获提升，支持跨领域适用性。但需注意：通用领域实验依赖外部LLM（gpt-oss-120b）作为判断模型，引入额外依赖和潜在评判质量风险；且仅在OLMO3-7B上验证，未在Qwen3系列重复。

**公平性检查**：
- **Baseline强度**：与TTRL、EMPO对比充分，但与Theta-Evolve、LaSeR等近期方法对比缺失。
- **异常结果**：Qwen3-8B上TEMPO（74.2）略逊于TTRL（74.9），论文未分析此异常，存在选择性报告倾向。
- **计算成本**：维护双模型+周期性critic重训练，内存和计算开销高于单模型baseline。
- **失败案例/局限**：消融实验均在单一模型/基准组合上进行，跨模型鲁棒性未充分验证；EM收敛性无正式证明。

## 方法谱系与知识库定位

**方法家族**：Test-Time Training (TTT) / Test-Time Adaptation → 强化学习微调（RLFT）→ EM算法框架下的策略优化

**Parent method**：EMPO（Expectation-Maximization Policy Optimization）— TEMPO将其重新解释为"仅M-step的退化EM变体"，通过恢复E-step补全完整EM循环。同时继承TTRL的测试时PPO训练范式，但关键改进在于分离actor-critic并引入周期性重校准。

**Changed slots**：
- **架构**：单模型自举 → actor-critic双模型分离
- **目标函数**：固定自生成奖励 → 周期性重校准的critic打分
- **训练流程**：纯M-step → E-step + M-step交替（完整EM）
- **数据利用**：仅无标注测试数据 → 无标注数据（M-step）+ 少量有标注数据（E-step重校准）
- **推理**：不变（测试时训练后单模型推理）

**Direct baselines与差异**：
- **TTRL**：使用多数投票自举，单模型；TEMPO分离critic并周期性用标注数据重校准
- **EMPO**：虽以EM命名但仅M-step，固定critic；TEMPO恢复完整E-step+M-step交替
- **PPO-continue**：在有标注数据上继续训练收敛模型；TEMPO证明此路径无效，关键在新颖测试时问题而非更多同分布数据

**Follow-up directions**：
1. **收敛性理论**：为EM交替过程提供正式收敛保证（论文明确列为未来工作）
2. **无标注E-step**：探索无需有标注数据的critic自校准机制（如对抗验证、合成数据），降低对外部标注的依赖
3. **动态交替频率**：当前固定周期 $T_E$；自适应调整E-step频率（如基于critic-actor分歧度）可能提升效率

**知识库标签**：
- **modality**: text / mathematical reasoning
- **paradigm**: test-time training, expectation-maximization, actor-critic RL
- **scenario**: deployment-time adaptation, out-of-distribution generalization
- **mechanism**: critic recalibration, reward de-drifting, diversity preservation
- **constraint**: requires small labeled dataset for critic recalibration, dual-model memory overhead

