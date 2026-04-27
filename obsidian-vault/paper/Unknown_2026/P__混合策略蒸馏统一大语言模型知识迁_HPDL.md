---
title: Hybrid Policy Distillation for LLMs
type: paper
paper_level: B
venue: Unknown
year: 2026
paper_link: https://arxiv.org/abs/2604.20244
aliases:
- 混合策略蒸馏统一大语言模型知识迁移
- HPDL
- 'HPD的方法贡献分为两个层次：理论统一视角和具体算法设计。


  **理论'
code_url: https://github.com/AnhaoZhao-LLMer/On_Policy_Distillation_Paper_List
modalities:
- Text
---

# Hybrid Policy Distillation for LLMs

[Paper](https://arxiv.org/abs/2604.20244) | [Code](https://github.com/AnhaoZhao-LLMer/On_Policy_Distillation_Paper_List)

**Topics**: [[T__Agent]], [[T__Knowledge_Distillation]], [[T__Math_Reasoning]], [[T__Code_Generation]]

> [!tip] 核心洞察
> HPD的方法贡献分为两个层次：理论统一视角和具体算法设计。

**理论层面**：论文将现有KD方法统一重新表述为token级别的重加权对数似然目标（reweighted log-likelihood at token level）。在此框架下，FKLD、RKLD、JSD等方法的差异可归结为对不同token赋予不同权重 $w_t$，从而在同一公式体系内建立各方法之间的联系。这一统一视角本身是理论贡献，但论文未为其提供独立的实验验证。

**算法层面**：HPD的核心损失函数为：
$$\mathcal{L}_{\text{HPD}} = \mathcal{L}_{\text{FKLD}} + \mathcal{L}_{\text{RKLD}} + \mathcal{L}_{\text{Reinforce}}$$
该公式结合了三个信号：(1) 前向KL项保证模式覆盖；(2) 反向KL项实现模式聚焦；(3) Reinforce项在学生模型产生不合理动作时显式增强专家token的概率，提供更稳定的优化信号。

HPD的两个关键设计组件为：
- **学生采样（Student Sampling）**

| 中文题名 | 混合策略蒸馏统一大语言模型知识迁移 |
| 英文题名 | Hybrid Policy Distillation for LLMs |
| 会议/期刊 | 2026 (arXiv预印本) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.20244) · [Code](https://github.com/AnhaoZhao-LLMer/On_Policy_Distillation_Paper_List) · [Project](待补充) |
| 主要任务 | 大语言模型知识蒸馏（LLM KD），覆盖数学推理、对话、代码生成 |
| 主要 baseline | FKLD（前向KL蒸馏）、RKLD（反向KL蒸馏）、JSD（JS散度）、SFT、DPO |

> [!abstract]
> 因为「现有KD方法在散度方向、优化策略和数据制度三个维度上各执一端，导致模式覆盖与模式聚焦无法兼得、在线采样成本过高」，作者在「FKLD/RKLD/JSD等传统蒸馏方法」基础上改了「提出token级重加权统一框架，叠加FKLD+RKLD+Reinforce三项混合损失，并引入轻量级学生采样实现近似在线探索」，在「Qwen2.5和LLaMA3的7B/8B→1B/1.5B/3B蒸馏设置」上取得「不引入额外超参数即实现稳定高效蒸馏，且可作为DPO前置初始化」。

- **关键性能1**：HPD在Qwen2.5 7B→1.5B蒸馏上，相比FKLD基线在数学推理任务上提升显著（具体数值待补充）
- **关键性能2**：消融实验（Figure 3）显示，去除学生采样后性能快速收敛但随即停滞；去除Reinforce操作后KL损失下降速度变慢
- **关键性能3**：HPD作为DPO前置初始化，可提升后续DPO训练效果（具体数值待补充）

## 背景与动机

大语言模型知识蒸馏（KD）旨在将大模型（教师）的能力迁移到小模型（学生），但效果深受三个耦合维度的影响：散度方向（如何衡量师生分布差异）、优化策略（离线数据 vs. 在线采样）以及数据制度（训练数据的来源与组织）。现有方法往往在这些维度上各执一端，导致实际应用中的显著缺陷。

具体而言，**前向KL散度（FKLD）**驱动学生覆盖教师分布的所有模式（mode coverage），但容易在低概率区域产生过估计，且完全依赖离线教师数据，学生缺乏自我探索空间，常陷入过早收敛。例如，在数学推理中，FKLD训练的学生可能机械复制教师的多种解题路径，却无法判断哪些路径真正可靠。**反向KL散度（RKLD）**让学生专注于教师的高概率区域（mode-seeking），但存在严重的模式坍塌风险——学生可能只学会最"安全"的回答而忽略多样性；同时RKLD需要在线采样，计算成本较高。**JSD等混合散度**虽尝试折中，但仍未能系统性解决优化稳定性问题，且往往引入额外超参数。

更深层的问题在于，这些方法缺乏统一的理论视角。FKLD、RKLD、JSD看似是独立的损失函数选择，研究者难以清晰理解它们之间的联系与差异，导致方法改进缺乏原则性指导。此外，在线策略采样虽理论上能提升蒸馏质量，但其高昂的计算开销限制了实际部署。

综上，核心挑战在于：如何在不引入额外超参数和过高计算成本的前提下，同时获得FKLD的模式覆盖优势和RKLD的模式聚焦优势，并通过轻量级在线采样改善优化动态。本文提出Hybrid Policy Distillation（HPD），将现有方法统一为token级重加权框架，并设计无超参数的混合损失实现这一目标。

## 核心创新

核心洞察：现有KD方法（FKLD/RKLD/JSD）的差异可完全归结为对同一token级重加权对数似然目标中不同token赋予不同权重，因为这一统一视角揭示了各方法的本质联系与互补性，从而使设计无超参数的混合损失——同时叠加FKLD的模式覆盖、RKLD的模式聚焦与Reinforce的稳定优化信号——成为可能。

| 维度 | Baseline | 本文 |
|:---|:---|:---|
| **理论框架** | FKLD/RKLD/JSD作为独立损失函数，缺乏统一解释 | 统一表述为token级重加权对数似然 $\sum_t w_t \cdot \log P_\theta(y_t)$，差异仅在于权重 $w_t$ 的计算方式 |
| **散度设计** | 单一散度（前向/反向/JS），需手动选择或调参 | 无超参数混合：FKLD + RKLD + Reinforce三项自动融合，无需权衡系数 |
| **采样策略** | 纯离线教师数据（FKLD）或高成本在线采样（RKLD） | 轻量级近似在线采样：学生采样自身偏好动作，计算开销可控 |
| **优化动态** | FKLD易过早收敛，RKLD易模式坍塌，JSD稳定性不足 | Reinforce项在学生产生不合理动作时显式增强专家token，提供额外稳定信号 |

## 整体框架


![Figure 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/524cfa46-69bf-4f2d-a627-9e87df1531ad/figures/Figure_4.png)
*Figure 4: Figure 4. Self-distillation Evolving. Stage 1: SFT + DPO/PPOinitialization. Stage 2: Iterative self-distillation with teacher modelupdates, while keeping the SFT data fixed.*



HPD的整体数据流遵循"统一重加权框架 → 三信号混合损失 → 轻量级在线采样"的架构，具体模块如下：

**输入**：教师模型 $P_{\text{teacher}}$（如Qwen2.5-7B或LLaMA3-8B）生成的离线数据，以及学生模型 $P_\theta$ 当前策略参数。

**模块A：Token级重加权基础框架**。将任意KD目标重新表述为对序列中每个token的对数似然进行加权求和。此模块统一了FKLD、RKLD、JSD的数学形式，为后续混合设计提供理论基础。

**模块B：混合损失计算**。基于模块A的框架，同时计算三项损失：(1) FKLD项，保证学生对教师分布的完整模式覆盖；(2) RKLD项，引导学生聚焦教师的高概率区域；(3) Reinforce项，当学生采样产生低质量输出时，显式提升专家token的概率。三项以固定比例相加，不引入额外超参数。

**模块C：学生采样（Student Sampling）**。允许学生模型基于当前策略 $P_\theta$ 采样自身偏好的动作，而非完全依赖教师生成的离线数据。这一轻量级近似在线采样使学生能够探索多样化轨迹，避免直接对齐教师分布导致的过早收敛。

**模块D：策略更新**。综合模块B的三项损失梯度，更新学生模型参数。训练过程中，学生策略持续演化，教师模型可保持固定或参与迭代更新（如Figure 4所示的自蒸馏扩展）。

**输出**：蒸馏后的学生模型 $P_\theta$，可直接部署或作为DPO等后续对齐方法的前置初始化。

```
[教师模型 P_teacher] ──→ [离线数据 D_off]
                                ↓
[学生模型 P_θ] ←────── [模块C: 学生采样] ←── 当前策略
       ↓                              ↑
[模块A: Token重加权框架] ←─────────────┘
       ↓
[模块B: 混合损失 = FKLD + RKLD + Reinforce]
       ↓
[模块D: 策略更新] ──→ [更新后的 P_θ]
       ↓
[输出: 蒸馏学生模型 / DPO初始化]
```

## 核心模块与公式推导

### 模块1: Token级重加权统一框架（理论基石）

**直觉**：现有KD方法看似差异巨大，实则共享同一数学结构——对token对数似然的加权平均，区别仅在于权重分配策略。

**Baseline公式（FKLD）**：$$\mathcal{L}_{\text{FKLD}} = \mathbb{E}_{x \sim \mathcal{D}} \mathbb{E}_{y \sim P_{\text{teacher}}(\cdot|x)} \left[ \sum_{t=1}^{|y|} \log P_\theta(y_t | y_{<t}, x) \right]$$
符号: $P_\theta$ = 学生模型，$P_{\text{teacher}}$ = 教师模型，$y_t$ = 第t个token，$y_{<t}$ = 历史上下文。

**变化点**：FKLD仅从教师采样，权重隐含为1；RKLD需从学生采样并引入重要性权重 $\frac{P_{\text{teacher}}}{P_\theta}$，导致高方差；JSD试图平均两者但权重设计复杂。这些方法的"独立性"阻碍了系统性组合。

**本文公式（推导）**：
$$\text{Step 1}: \quad \mathcal{L}_{\text{KD}} = \mathbb{E}_{x,y} \left[ \sum_{t=1}^{|y|} w_t \cdot \log P_\theta(y_t | y_{<t}, x) \right] \quad \text{提出统一形式，所有KD方法差异归于 } w_t$$
$$\text{Step 2}: \quad w_t^{\text{FKLD}} = 1, \quad w_t^{\text{RKLD}} = \frac{P_{\text{teacher}}(y_t|y_{<t})}{P_\theta(y_t|y_{<t})}, \quad w_t^{\text{JSD}} = f\left(\frac{P_{\text{teacher}}}{P_\theta}\right) \quad \text{显式导出各方法权重}$$
$$\text{最终}: \quad \mathcal{L}_{\text{unified}} = \mathbb{E}_{x,y} \left[ \sum_{t} w_t^{\text{method}} \cdot \log P_\theta(y_t | y_{<t}, x) \right]$$

**对应消融**：该统一框架本身为理论贡献，无直接消融；但其揭示了FKLD与RKLD的互补性，为模块2的混合设计奠基。

---

### 模块2: 三信号混合损失（HPD核心）

**直觉**：FKLD保证"见多识广"（覆盖所有模式），RKLD保证"去伪存真"（聚焦高概率模式），Reinforce保证"纠错强化"（对错误探索给予明确反馈），三者缺一不可。

**Baseline公式（FKLD + RKLD 简单相加）**：$$\mathcal{L}_{\text{naive}} = \mathcal{L}_{\text{FKLD}} + \lambda \mathcal{L}_{\text{RKLD}}$$
符号: $\lambda$ = 需手动调谐的权衡超参数。

**变化点**：简单相加引入敏感超参数$\lambda$，且缺乏对"学生探索出错时如何纠正"的显式机制。RKLD的mode-seeking特性在学生采样不合理时可能加剧错误累积。

**本文公式（推导）**：
$$\text{Step 1}: \quad \mathcal{L}_{\text{FKLD}} = \mathbb{E}_{y \sim P_{\text{teacher}}} \left[ \sum_t \log P_\theta(y_t) \right] \quad \text{前向KL：教师采样，权重为1，保证模式覆盖}$$
$$\text{Step 2}: \quad \mathcal{L}_{\text{RKLD}} = \mathbb{E}_{y \sim P_\theta} \left[ \sum_t \frac{P_{\text{teacher}}(y_t)}{P_\theta(y_t)} \log P_\theta(y_t) \right] = \mathbb{E}_{y \sim P_\theta} \left[ \sum_t \log P_{\text{teacher}}(y_t) \right] \quad \text{反向KL：学生采样，聚焦教师高概率区域}$$
$$\text{Step 3}: \quad \mathcal{L}_{\text{Reinforce}} = \mathbb{E}_{y \sim P_\theta} \left[ R(y) \cdot \sum_t \log P_\theta(y_t) \right] \quad \text{加入奖励信号：当学生采样质量低时，提升专家token概率}$$
$$\text{最终}: \quad \mathcal{L}_{\text{HPD}} = \mathcal{L}_{\text{FKLD}} + \mathcal{L}_{\text{RKLD}} + \mathcal{L}_{\text{Reinforce}}$$

关键设计：三项以**固定1:1:1比例**相加，不引入任何额外超参数。Reinforce项的奖励$R(y)$基于学生样本与教师分布的匹配程度自动计算，无需人工设计。

**对应消融**：Figure 3b显示，去除Reinforce操作（仅保留FKLD+RKLD）导致KL损失下降速度变慢，加入后加速与教师分布对齐。

---

### 模块3: 学生采样与轻量级在线探索（优化动态）

**直觉**：纯粹离线蒸馏使学生成为教师的"复读机"，而完整在线策略梯度成本过高；让学生"半自主地尝试"再及时纠正，是效率与效果的平衡点。

**Baseline公式（纯离线FKLD）**：$$y \sim P_{\text{teacher}}(\cdot|x), \quad \mathcal{L} = \sum_t \log P_\theta(y_t)$$
符号: 数据完全由教师生成，学生无自主采样权。

**变化点**：纯离线数据导致学生缺乏探索空间，尤其在训练初期学生与教师分布差异大时，直接模仿易陷入局部最优；而标准PPO/REINFORCE等在线方法需大量采样和优势估计，计算开销 prohibitive。

**本文公式（推导）**：
$$\text{Step 1}: \quad y^{\text{student}} \sim P_\theta(\cdot|x) \quad \text{学生基于当前策略采样，获得探索轨迹}$$
$$\text{Step 2}: \quad y^{\text{mixed}} = \begin{cases} y^{\text{student}} & \text{with prob } \alpha \\ y^{\text{teacher}} & \text{with prob } 1-\alpha \end{cases} \quad \text{轻量级混合：默认 } \alpha=0.5 \text{ 或自适应}$$
$$\text{Step 3}: \quad \mathcal{L}_{\text{HPD}} \text{ 在 } y^{\text{mixed}} \text{ 上计算，Reinforce项专门处理 } y^{\text{student}} \text{ 中的低质量样本}$$
$$\text{最终}: \quad \text{实现"近似在线"效果，计算成本远低于完整策略优化}$$

关键：学生采样与Reinforce项形成闭环——学生探索 → 出错被Reinforce纠正 → 策略改善 → 更有意义的探索。

**对应消融**：Figure 3a显示，去除学生采样（纯离线FKLD+RKLD）后性能快速收敛但随即停滞；保留学生采样则持续优化。

## 实验与分析

主实验在Qwen2.5（7B→1.5B/3B）和LLaMA3（8B→1B/3B）两个模型族上验证，覆盖数学推理、对话、代码生成任务。

| Method | 数学推理 (待补充具体bench) | 对话 (待补充) | 代码生成 (待补充) | 超参数数量 |
|:---|:---|:---|:---|:---|
| FKLD | 基线 | 基线 | 基线 | 0 |
| RKLD |  |  |  | 0 |
| JSD |  |  |  | ≥1 |
| SFT |  |  |  | 0 |
| **HPD (本文)** | **** | **** | **** | **0** |


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/524cfa46-69bf-4f2d-a627-9e87df1531ad/figures/Figure_3.png)
*Figure 3: Figure 3. Ablation study of HPD.*



**核心结论支持**：HPD的关键优势在于"无超参数"前提下的稳定提升。相比JSD等混合方法需调谐混合系数，HPD的三项固定比例设计简化了实际部署。Figure 3a的学生采样消融直接支持"探索-利用权衡"的核心主张：纯离线方法快速收敛后停滞，而学生采样的引入使优化曲线持续上升。Figure 3b的Reinforce消融验证了第三项的必要性——仅FKLD+RKLD的混合在KL散度下降效率上弱于完整HPD。

**扩展应用**：Figure 4展示了HPD作为自蒸馏框架的两阶段扩展——Stage 1以SFT+DPO/PPO初始化，Stage 2进行迭代自蒸馏并更新教师模型。这表明HPD不仅适用于固定师生设置，还可嵌入持续学习流程。



**公平性检查**：
- **Baselines强度**：FKLD/RKLD/JSD/SFT为标准蒸馏基线，但未与更近期的专用蒸馏方法（如MiniLLM、GKD等）直接对比，存在局限。
- **计算成本**：学生采样引入额外前向-反向传播，但相比完整PPO的rollout+advantage estimation仍属轻量级；精确FLOP对比。
- **失败案例**：未报告学生采样在训练后期的噪声引入问题——当学生策略已接近教师时，继续采样可能引入无益方差；论文未讨论此阶段的退火策略。
- **数据制度细节**：离线/在线数据比例、混合概率$\alpha$的具体设置。

## 方法谱系与知识库定位

**方法家族**：知识蒸馏（Knowledge Distillation）→ 大语言模型专用蒸馏（LLM KD）→ 策略蒸馏/序列级蒸馏（Policy/Sequence-level Distillation）。

**Parent method**：FKLD（前向KL蒸馏，如Hinton经典KD在LLM中的直接应用）。HPD保留了FKLD的离线数据效率和模式覆盖特性，但通过统一重加权框架将其扩展为可组合的基础组件。

**改动插槽**：
- **目标函数（objective）**：从单一散度 → 无超参数三项混合（FKLD+RKLD+Reinforce）
- **训练配方（training_recipe）**：从纯离线 → 轻量级近似在线采样（学生采样）
- **数据制度（data_curation）**：固定教师数据 → 动态混合师生数据源
- 架构（architecture）与推理（inference）未改动，保持插件式兼容

**直接Baselines对比**：
- **FKLD**：HPD将其作为子模块保留，但补充了RKLD的聚焦能力和Reinforce的稳定信号
- **RKLD**：HPD避免其高方差在线采样成本，以"近似在线"替代
- **JSD**：HPD提供类似"混合"效果但无需调谐超参数
- **MiniLLM/GKD等专用方法**：HPD未直接对比，但其统一框架理论上可包容这些方法的变体

**后续方向**：
1. **自适应混合权重**：当前固定1:1:1比例虽无超参数，但是否可根据训练动态自动调整三项贡献？
2. **多教师扩展**：统一框架天然支持多教师token级权重融合，尚未探索
3. **与RLHF/DPO的深度耦合**：Figure 4展示了两阶段流程，但端到端联合优化待研究

**标签体系**：
- **modality**：文本（大语言模型）
- **paradigm**：知识蒸馏 / 策略优化
- **scenario**：模型压缩（7B/8B→1B/1.5B/3B）、能力迁移（数学/对话/代码）
- **mechanism**：混合散度优化、轻量级在线采样、token级重加权
- **constraint**：无额外超参数、计算效率优先、插件式兼容现有流程

