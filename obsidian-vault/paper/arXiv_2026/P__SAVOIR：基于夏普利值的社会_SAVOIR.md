---
title: 'SAVOIR: Learning Social Savoir-Faire via Shapley-based Reward Attribution'
type: paper
paper_level: B
venue: arXiv
year: 2026
paper_link: https://arxiv.org/abs/2604.18982
aliases:
- SAVOIR：基于夏普利值的社会对话奖励归因方法报告
- SAVOIR
method: SAVOIR
---

# SAVOIR: Learning Social Savoir-Faire via Shapley-based Reward Attribution

[Paper](https://arxiv.org/abs/2604.18982)

**Topics**: [[T__Reinforcement_Learning]], [[T__Reasoning]] | **Method**: [[M__SAVOIR]] | **Datasets**: SOTOPIA-Hard, SOTOPIA Self-Play, SOTOPIA-All Self-Play

## 基本信息

**论文标题**: SAVOIR: Learning Social Savoir-Faire via Shapley-based Reward Attribution

**基座模型**: Qwen2.5-7B-Instruct (7B参数)

**评估基准**: SOTOPIA, SOTOPIA-Hard, SOTOPIA-All

**核心方法**: 合作博弈论 + 夏普利值信用分配 + 蒙特卡洛rollout期望效用

**代码/数据**: 未在提取信息中明确提供链接

**主要对比基线**: Sotopia-RL, SOTOPIA-π, DSI, SDPO, EPO

## 核心主张

SAVOIR提出了一种基于合作博弈论的社会对话奖励归因框架，通过**蒙特卡洛rollout计算期望效用**并结合**夏普利值进行公理化信用分配**，首次将前瞻式评估与公平归因机制引入多轮对话的强化学习训练。核心证据：在SOTOPIA-Hard基准上，7B参数的SAVOIR以**GOAL=7.18超越Sotopia-RL的6.68（+7.5%）**，并在Self-Play中达到**7.93**，超越DSI（7.31）和Sotopia-RL（7.81）；在SOTOPIA-All上达到**8.43**，超过GPT-4o（8.19）和Claude-3.5-Sonnet（8.29）。消融实验显示，移除期望效用（-0.29）或夏普利值（-0.22）均显著降低性能，验证了双组件的必要性。置信度：**0.95**。

## 研究动机

现有社会对话智能体训练面临**信用分配困境**：多轮对话中单个话语对最终社交结果的贡献难以量化。Sotopia-RL等现有方法采用**启发式LLM事后归因**，将回合级奖励简单回溯到各话语，缺乏理论保证；SOTOPIA-π依赖行为克隆与过滤数据的自增强，未显式建模信用分配；DSI等动态策略注入方法虽有一定效果，但仍属启发式范畴。核心缺口在于：**(1)** 现有方法均为**回顾式评估**，仅依赖最终回合结果，忽略话语的战略前瞻性价值；**(2)** 信用分配缺乏**公理化公平性保证**，易出现归因偏差。SAVOIR填补此缺口，将信用分配形式化为合作博弈，引入夏普利值的效率性、对称性、虚拟玩家和可加性四大公理，实现理论可证的最优归因。

## 方法流程

SAVOIR采用**三阶段七模块**管道：

```
输入对话 τ (含n个话语)
    ↓
[阶段1] 联盟采样 → KernelSHAP加权采样得到子集 S⊆N
    ↓
[模块2] 历史重建 H(S) — 重组S中话语及伙伴回复
    ↓
[阶段2] 蒙特卡洛Rollout — 从H(S)执行J次完整对话模拟 τ_j
    ↓
[模块4] 期望效用计算 v(S) = (1/J)Σ_j U(τ_j)
    ↓
[阶段3] 夏普利值计算 — 基于所有联盟值{φ_i}
    ↓
[模块6] 奖励归一化 φ̂_i ∈ [0,10]
    ↓
[模块7] 奖励模型训练 R_θ(c,a) = MLP(LLM_θ([c;a]))
```

**关键创新**：以"前瞻式期望效用"替代"最终结果回溯"，以"公理化公平分配"替代"启发式LLM归因"。

## 关键公式

**1. 期望效用价值函数（新颖）**
```latex
v(S) = \mathbb{E}_{\tau' \sim \mathcal{R}(H(S))} \left[ U(\tau') \right] \approx \frac{1}{J} \sum_{j=1}^{J} U(\tau_j)
```
前瞻式评估：从重建历史H(S)出发，通过J次rollout近似未来轨迹期望效用。

**2. 多维度效用聚合（继承SOTOPIA）**
```latex
U(\tau) = \sum_{d=1}^{D} w_d \cdot G_d(\tau)
```
将目标完成、关系维护等维度加权为标量。

**3. 夏普利值（新颖应用）**
```latex
\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(n - |S| - 1)!}{n!} \left[ v(S \cup \{i\}) - v(S) \right]
```
公理化公平归因：效率性、对称性、虚拟玩家、可加性。

**4. KernelSHAP高效近似（源自Lundberg & Lee，适配应用）**
```latex
\phi^* = \arg\min_{\phi} \sum_{k=1}^{K} w_k \left( v(S_k) - \sum_{i=1}^{n} \phi_i \cdot z_{ki} \right)^2, \quad w_k = \frac{n - 1}{\binom{n}{|S_k|} \cdot |S_k| \cdot (n - |S_k|)}
```
将指数复杂度降为多项式可解。

**5. 奖励模型（新颖）**
```latex
R_\theta(c, a) = \text{MLP}(\text{LLM}_\theta([c; a])), \quad \mathcal{L}_{\text{RM}} = \mathbb{E}\left[ \left( R_\theta(c, a) - \hat{\phi} \right)^2 \right]
```
将昂贵夏普利计算蒸馏为高效推理。

## 实验结果

| 基准 | 指标 | SAVOIR | 对比基线 | 提升 |
|:---|:---|:---|:---|:---|
| SOTOPIA-Hard (GPT-4o伙伴) | GOAL | **7.18** | Sotopia-RL 6.68 | **+7.5%** |
| SOTOPIA Self-Play | GOAL | **7.93** | DSI 7.31 / Sotopia-RL 7.81 | SOTA |
| SOTOPIA-All Self-Play | GOAL | **8.43** | GPT-4o 8.19 / Claude-3.5-Sonnet 8.29 | **+13.8%** (vs专有模型) |
| SOTOPIA-Hard (推理模型) | GOAL | **7.93** | o3-mini 5.14 | **+54.3%** 差距 |
| SOTOPIA-Hard (Claude 4.5伙伴) | GOAL | **6.64** | Sotopia-RL 6.54 | +1.5% |

**消融实验**（§5.2, Table 2）：
- 移除期望效用（EU）：7.18 → **6.89** (-0.29)
- 移除夏普利值（启发式归因）：7.18 → **6.96** (-0.22)
- 同时移除EU+Shapley（即Sotopia-RL）：**6.68** (-0.50)

**证据强度评估**：0.8。主要局限：SDPO、EPO等引用的基线未在主要结果中展示；与未针对SOTOPIA微调的专有LLM对比公平性存疑；自对弈结果可能不泛化到异构伙伴场景。

## 相关工作

**核心基线与方法来源分类**：

| 角色 | 方法 | 关系说明 |
|:---|:---|:---|
| **主要基线** | Sotopia-RL | 直接对比对象，SAVOIR在其上系统性替换5个核心slot |
| 次要基线 | SOTOPIA-π | 交互式学习先驱，SAVOIR在Self-Play中超越 |
| 次要基线 | DSI (Sotopia-ω) | 动态策略注入，Self-Play中7.31 vs SAVOIR 7.93 |
| **引用但未充分对比** | SDPO | 段级直接偏好优化，同领域但主实验缺失 |
| **引用但未充分对比** | EPO | 显式策略优化，战略推理RL基线 |
| **核心组件来源** | KernelSHAP (Lundberg & Lee 2017) | 夏普利值高效计算的核心技术来源，通过加权最小二乘将指数复杂度降为多项式 |

**最重要5篇引用**：
1. **Sotopia-RL** — 直接父方法，启发式信用分配被完全替换
2. **KernelSHAP (Lundberg & Lee)** — 使夏普利计算可扩展的关键技术
3. **SOTOPIA-π** — 同领域交互学习先驱
4. **DSI/Sotopia-ω** — 同基准竞争方法
5. **SDPO/EPO** — 同任务空间方法，但实验对比不足

## 方法谱系

**谱系位置**：Sotopia-RL → **SAVOIR**（直接继承与系统性改造）

| Slot | 父方法 Sotopia-RL | 子方法 SAVOIR | 变更类型 |
|:---|:---|:---|:---|
| **reward_design** | 启发式LLM事后归因，回合级奖励回溯 | **前瞻式期望效用** + 夏普利公平分配 | **替换** |
| **credit_assignment** | 直接LLM提示，无理论保证 | **夏普利值公理化归因**（效率/对称/边际/可加） | **替换** |
| **value_function** | 最终回合结果 U(τ_full) 或启发式判断 | **v(S)=E[U(τ')] 经J次rollout** | **替换** |
| **exploration_strategy** | 未显式建模（标准BC+自增强） | **KernelSHAP联盟采样** + 反事实轨迹探索 | **新增** |
| **training_recipe** | 直接RL或过滤行为克隆 | **两阶段**：(1)夏普利奖励计算 → (2)MLP奖励模型蒸馏 → 策略优化 | **修改** |

**继承**：SOTOPIA评估框架、多轮对话任务设定、7B LLM基础架构
**改造**：从"启发式-回顾式"范式跃迁至"公理化-前瞻式"范式，引入博弈论形式化保证

## 局限与展望

**论文明确局限**：
- 伙伴智能提升时性能退化（Figure 7）：面对更强社交智能伙伴，SAVOIR优势收窄
- 计算成本：每轮训练需J次蒙特卡洛rollout，联盟采样仍带来额外开销

**分析推断局限**：
- **对比公平性**：主结果强调与GPT-4o/Claude等专有LLM对比，但这些模型未针对SOTOPIA微调，对比不够公平
- **基线覆盖不足**：SDPO、EPO、SOTOPIA-ω等引用方法未在主要实验中出现
- **泛化性质疑**：自对弈（Self-Play）结果可能无法推广到异构伙伴、真实人类交互场景
- **推理模型零样本评估**：o3-mini等未适配即测试，54.3%差距可能被夸大
- **伙伴依赖性**：Claude 4.5-sonnet伙伴下仅+1.5%提升，显示方法对伙伴能力的敏感性

**未来方向**：(1) 异构伙伴鲁棒性提升；(2) 人类在环验证；(3) 更高效的夏普利近似替代KernelSHAP；(4) 多智能体扩展 beyond双人对话。

## 知识图谱定位

**任务节点**：
- `social dialogue`（社交对话）— 核心任务域，涵盖谈判、协作、说服
- `credit assignment in RL`（RL信用分配）— 核心子问题，多轮设置下的动作归因

**方法节点**：
- `SAVOIR` — 新增方法节点：合作博弈论+夏普利值+期望效用的融合框架
- `Sotopia-RL` — 父方法节点，启发式信用分配代表
- `KernelSHAP` — 机制节点，使指数级夏普利计算可扩展
- `Shapley value` / `Expected Utility` / `Monte Carlo rollout` — 支撑机制节点

**数据集/基准节点**：
- `SOTOPIA` / `SOTOPIA scenarios` — 评估基准，含Hard变体

**结构贡献**：SAVOIR在知识图谱中架设了**"博弈论机制 → 对话信用分配"**的新边，将传统用于特征归因的夏普利值（Lundberg & Lee领域）迁移至**序列决策的奖励设计**领域，填补了social dialogue与cooperative game theory之间的结构空白。其两阶段蒸馏架构（昂贵夏普利计算 → 高效奖励模型）也为可扩展信用分配提供了可复用的模式模板。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/3b8823be-2942-4d18-bbf5-5969d45731bb/figures/Figure_1.png)
*Figure 1: Figure 1: Overview of the social agent training pipeline.Stage 1: Collect social interaction episodes throughLLM self-play. Stage 2: Design utterance-level, multi-dimensional rewards through attributi*


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/3b8823be-2942-4d18-bbf5-5969d45731bb/figures/Figure_2.png)
*Figure 2: Figure 2: Overview of the SAVOIR framework. Step 1: Input social dialogue τ with agent utterances N ={a1, . . . , an}. Step 2: Sample coalitions C using KernelSHAP weighting. Step 3: For each coalitio*


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/3b8823be-2942-4d18-bbf5-5969d45731bb/figures/Figure_3.png)
*Figure 3: Figure 3: Shapley value computation for a2. For eachof the n! = 6 permutations, we compute a2’s marginalcontribution when it joins. The Shapley value is theaverage across all permutations. See Appendi*


![Figure 5](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/3b8823be-2942-4d18-bbf5-5969d45731bb/figures/Figure_5.png)
*Figure 5: Figure 4: SHAP kernel weight distribution. Extremecoalition sizes (small: individual effects; large: synergyeffects) receive higher weights, enabling efficient Shap-ley approximation.*


![Figure 7](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/3b8823be-2942-4d18-bbf5-5969d45731bb/figures/Figure_7.png)
*Figure 7: Table 2: Component ablation on SOTOPIA-Hard (GPT-4o partner). EU and Shapley each improve over thebaseline independently, and their combination is strictlybetter than either alone.*


![Figure 9](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/3b8823be-2942-4d18-bbf5-5969d45731bb/figures/Figure_9.png)
*Figure 9: Figure 8: Effect of training data scale. Both Goal andAvg improve consistently from 2K to 7.5K episodes.*


