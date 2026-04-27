---
title: 'DR-Venus: Towards Frontier Edge-Scale Deep Research Agents with Only 10K Open Data'
type: paper
paper_level: B
venue: Unknown
year: 2026
paper_link: https://arxiv.org/abs/2604.19859
aliases:
- 10K开放数据训练边缘端深度研究智能体
- DR-Venus
- 核心直觉是：小模型的学习瓶颈不在于模型容量本身
method: DR-Venus
modalities:
- Text
paradigm: Reinforcement Learning
---

# DR-Venus: Towards Frontier Edge-Scale Deep Research Agents with Only 10K Open Data

[Paper](https://arxiv.org/abs/2604.19859)

**Topics**: [[T__Agent]], [[T__Reinforcement_Learning]] | **Method**: [[M__DR-Venus]]

> [!tip] 核心洞察
> 核心直觉是：小模型的学习瓶颈不在于模型容量本身，而在于监督信号的质量与密度。SFT阶段通过严格清洗消除噪声轨迹（冗余工具调用、格式不一致），让模型从干净信号中学习；RL阶段通过将稀疏的轨迹级奖励分解为密集的轮次级信息增益奖励，解决长时序任务中信用分配稀疏导致的优势崩溃问题。两者本质上都是在解决同一个问题：如何在有限数据和有限模型容量的双重约束下，最大化每个训练步骤的有效监督信号。

| 中文题名 | 10K开放数据训练边缘端深度研究智能体 |
| 英文题名 | DR-Venus: Towards Frontier Edge-Scale Deep Research Agents with Only 10K Open Data |
| 会议/期刊 | arXiv 2026 (preprint) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.19859) · [Code](待补充) · [Project](待补充) |
| 主要任务 | 边缘端（≤9B参数）深度研究智能体训练，覆盖BrowseComp、BrowseComp-ZH、GAIA Text-Only、xBenchDS-2505、xBenchDS-2510、DeepSearchQA六个基准 |
| 主要 baseline | ≤9B开源小智能体（如REDSearcher衍生模型）、≥30B开源智能体（如REDSearcher-32B）、前沿闭源模型（如GPT-4o、Claude等） |

> [!abstract] 因为「边缘端小模型在深度研究任务中面临数据噪声敏感、RL优势函数崩溃、数据极度稀缺三重挑战」，作者在「Qwen3-4B-Thinking-2507 + IGPO」基础上改了「严格四步数据清洗 + 轮次级信息增益奖励与格式感知正则化」，在「BrowseComp/BrowseComp-ZH等六个深度研究基准」上取得「DR-Venus-4B-RL超越所有≤9B先前智能体、缩小与30B级系统差距」

- **数据效率**：仅用10,001条原始REDSearcher轨迹经清洗后完成SFT，1K精选QA对完成RL
- **规模突破**：4B参数模型在BrowseComp系列上超越所有≤9B开源智能体（具体数值待补充）
- **能力上限**：Pass@K分析显示4B模型能力上限"出人意料地高"（具体K值与数值待补充）

## 背景与动机

深度研究（Deep Research）任务要求智能体在多轮交互中自主调用搜索、浏览等工具，逐步收集信息并生成综合答案。一个典型场景是：用户提出复杂查询（如"比较2024年三种新型电池技术的能量密度与商业化进展"），智能体需执行数十轮search/browse操作，筛选网页内容，最终整合出结构化报告。这类任务对模型的工具使用规划、长时序信用分配和信息整合能力提出极高要求。

现有系统主要沿三条技术路线展开：

**路线一：大模型+闭源数据**。如OpenAI的Deep Research、Google的Gemini Deep Research，依赖专有30B+参数模型和闭源高质量轨迹数据，性能前沿但无法复现，更无法部署至边缘设备。

**路线二：开源大模型+开源数据**。如REDSearcher-32B基于Qwen-32B在开源REDSearcher数据集上训练，证明了开源数据训练的可行性，但其32B规模远超边缘设备承载能力（典型边缘约束为≤9B参数）。

**路线三：小模型+直接迁移**。将大模型训练流程直接缩放到≤9B模型，面临三重系统性失败：（1）**数据噪声敏感**——REDSearcher等开源轨迹包含大量冗余工具调用（如重复browse事件）、格式不一致和错误监督信号，小模型容量有限，对噪声的"吸收能力"远低于大模型，容易学到错误的工具调用模式；（2）**RL训练崩溃**——小模型能力边界狭窄，在长时序深度研究任务的RL训练中，rollout组（典型大小为8）常常不包含任何成功轨迹，导致优势函数计算时分母趋零、训练信号极度稀疏，传统稀疏轨迹级奖励（仅最终答案正确与否）无法提供有效梯度；（3）**数据利用率低下**——在仅有约10K开放数据的硬约束下，每条轨迹需被"榨取"最大价值，但长时序轨迹中的有效学习信号分散在数十轮交互中，现有方法缺乏针对性的信号挖掘机制。

这三重挑战共同指向一个核心问题：**在有限开放数据监督下，如何同时提升训练数据质量与数据利用率，以训练出强大的小型深度研究智能体？** DR-Venus的答案是：SFT阶段通过严格数据清洗消除噪声监督，RL阶段通过轮次级奖励分解提升信号密度，两者协同突破小模型的学习效率瓶颈。

## 核心创新

**核心洞察：小模型的学习瓶颈不在于模型容量本身，而在于监督信号的质量与密度**，因为小模型对噪声更敏感且RL训练中成功轨迹更稀缺，从而将"数据清洗+密集奖励分解"作为突破口成为可能。

具体而言，SFT阶段的严格四步清洗将原始10,001条轨迹中的冗余工具调用（15,728次重复交互）、不支持工具（3,378次调用）和错误答案轨迹系统性剔除；RL阶段将稀疏的轨迹级奖励分解为每轮可计算的信息增益奖励，使信用分配粒度从"整段轨迹"细化到"单次工具调用"。两者本质上是同一思路在不同训练阶段的应用：最大化每个训练步骤的有效监督信号。

| 维度 | Baseline（REDSearcher直接微调 + 标准IGPO） | 本文（DR-Venus） |
|:---|:---|:---|
| **数据预处理** | 直接使用原始轨迹，含冗余调用与格式噪声 | 四步清洗：环境对齐→禁用工具剪枝→去重→正确性过滤→长时序重采样 |
| **SFT监督信号** | 全序列损失，包含系统提示和工具响应的梯度干扰 | 仅assistant token计算损失，屏蔽非助手输出 |
| **RL奖励粒度** | 轨迹级稀疏奖励（最终答案正确与否） | 轮次级密集奖励：信息增益奖励（browse-aware IG分配 + IG-Scale）+ 格式感知正则化（λ_fmt=1.0） |
| **信用分配** | 整段轨迹单一回报，长时序信用模糊 | 折扣因子γ=0.95的轮次级信用分配，明确每轮工具调用的贡献度 |

## 整体框架


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/ab84dcf0-d94a-4def-bf49-2839ab6c3c82/figures/Figure_1.png)
*Figure 1: Figure 1 Performance comparison of DR-Venus-4B against other open-source models on BrowseComp and BrowseComp-ZH.*



DR-Venus采用两阶段训练方案，数据流如下：

**输入**：10,001条原始REDSearcher开放轨迹 → 经严格清洗后得到高质量SFT数据 → 再精选1K QA对用于RL训练。

**阶段一：Agentic SFT**
- **模块A：四步数据清洗引擎**。输入原始轨迹，依次执行：（1）环境对齐——统一交互格式为在线推理管道格式；（2）禁用工具剪枝——移除运行时环境不支持的Python-Interpreter调用（涉及1,064条轨迹、3,378次调用）；（3）去重——删除重复的search/browse调用及配对响应（涉及6,821条轨迹、15,728次重复交互，主要为重复browse事件）；（4）正确性过滤——仅保留答案正确的轨迹；（5）长时序轨迹重采样——针对长时序轨迹进行重采样以提升数据利用率（具体策略待补充）。输出清洗后的SFT训练集。
- **模块B：掩码SFT训练器**。以Qwen3-4B-Thinking-2507为骨干，对所有非assistant token进行掩码，仅在助手输出上计算交叉熵损失。输出DR-Venus-4B-SFT。

**阶段二：Agentic RL**
- **模块C：轮次级奖励计算器**。基于IGPO算法框架，为每个交互轮次计算两类奖励：（1）信息增益奖励——评估工具调用带来的信息增益，启用browse-aware IG分配和IG-Scale缩放；（2）格式感知正则化奖励——以固定权重λ_fmt=1.0惩罚格式错误。输出轮次级奖励序列。
- **模块D：IGPO-DR训练器**。使用折扣因子γ=0.95进行轮次级信用分配，rollout组大小为8，最大上下文长度256K tokens，在1K精选QA对上执行策略优化。输出DR-Venus-4B-RL。

**输出**：具备深度研究能力的边缘端4B参数智能体，支持BrowseComp、GAIA等多基准评估。

```
原始REDSearcher轨迹 (10,001条)
    ↓
[模块A] 四步数据清洗引擎
    ↓
清洗后SFT数据 ──→ [模块B] 掩码SFT训练器 ──→ DR-Venus-4B-SFT
    │                                              │
    └──── 精选1K QA对 ──→ [模块C] 轮次级奖励计算器 ──┘
                              ↓
                        [模块D] IGPO-DR训练器
                              ↓
                        DR-Venus-4B-RL
```

## 核心模块与公式推导

### 模块1：掩码SFT损失（对应框架图 阶段一/模块B）

**直觉**：避免系统提示和工具响应的梯度干扰，让模型专注于学习"如何作为助手生成正确的工具调用与答案"。

**Baseline公式**（标准因果语言建模）：
$$L_{\text{base}} = -\sum_{t=1}^{T} \log p_\theta(x_t | x_{<t})$$
符号：$\theta$为模型参数，$T$为序列总长度，$x_t$为第$t$个token，$x_{<t}$为历史上下文。

**变化点**：标准LM损失对所有token平等计算梯度，但深度研究轨迹中系统提示（固定模板）和工具响应（环境返回，非模型生成）不应贡献梯度。直接训练会浪费计算并引入噪声。

**本文公式**：
$$\text{Step 1}: \quad m_t = \mathbb{1}[x_t \in \text{assistant tokens}] \quad \text{构造掩码，仅标记助手生成位置}$$
$$\text{Step 2}: \quad L_{\text{mask}} = -\sum_{t=1}^{T} m_t \cdot \log p_\theta(x_t | x_{<t}) \quad \text{掩码后仅在助手输出上计算损失}$$
$$\text{最终}: \quad L_{\text{SFT}} = \frac{L_{\text{mask}}}{\sum_{t=1}^{T} m_t} \quad \text{归一化保证批次间可比性}$$

**对应消融**：

---

### 模块2：轮次级信息增益奖励（对应框架图 阶段二/模块C）

**直觉**：将稀疏的"最终答案是否正确"分解为每轮工具调用带来的信息增量，使小模型在RL训练中即使未完整解决问题，也能获得可优化的中间信号。

**Baseline公式**（标准IGPO/轨迹级PPO）：
$$L_{\text{IGPO}} = \mathbb{E}_{(o,a) \sim \pi_{\theta_{\text{old}}}} \left[ \min\left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$
$$R_{\text{trajectory}} = \mathbb{1}[\text{final answer correct}] \cdot r_{\text{sparse}}$$
符号：$r_t(\theta) = \frac{\pi_\theta(a_t|o_t)}{\pi_{\theta_{\text{old}}}(a_t|o_t)}$为重要性采样比，$\hat{A}_t$为广义优势估计，$R_{\text{trajectory}}$为整段轨迹的稀疏回报。

**变化点**：小模型rollout组（大小8）在长时序任务中常无成功轨迹，导致$\hat{A}_t$估计方差极大甚至无法计算（分母为零）。需将$R_{\text{trajectory}}$分解为轮次级密集奖励$R_{\text{turn}} = \sum_{k=1}^{K} \gamma^{k-1} r_k$，其中$K$为交互轮数。

**本文公式（推导）**：
$$\text{Step 1}: \quad r_k^{\text{IG}} = \text{InfoGain}(o_k, a_k, o_{k+1}) \cdot \text{IG-Scale}(a_k) \cdot \mathbb{1}[a_k \text{ is browse-aware}] \quad \text{计算第}k\text{轮信息增益，启用browse-aware分配和缩放}$$
$$\text{Step 2}: \quad r_k^{\text{fmt}} = -\lambda_{\text{fmt}} \cdot \text{FormatError}(a_k) \quad \text{加入格式感知正则化，}\lambda_{\text{fmt}}=1.0$$
$$\text{Step 3}: \quad r_k = r_k^{\text{IG}} + r_k^{\text{fmt}} \quad \text{轮次级总奖励为信息增益与格式惩罚之和}$$
$$\text{Step 4}: \quad R_{\text{turn}} = \sum_{k=1}^{K} \gamma^{k-1} r_k, \quad \gamma=0.95 \quad \text{折扣累积保证远期信用衰减}$$
$$\text{最终}: \quad L_{\text{RL}} = \mathbb{E}\left[ \min\left( r_t(\theta) \hat{A}_t^{\text{turn}}, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t^{\text{turn}} \right) \right], \quad \hat{A}_t^{\text{turn}} \text{基于} R_{\text{turn}} \text{计算}$$

**关键设计**：browse-aware IG分配确保browse类工具（核心信息获取手段）获得更精细的增益计算；IG-Scale对增益值进行自适应缩放防止数值不稳定；γ=0.95的折扣使模型优先优化近期轮次，同时保留远期规划信号。

**对应消融**：

---

### 模块3：长时序轨迹重采样（对应框架图 阶段一/模块A之步骤5）

**直觉**：长时序轨迹包含更多有效学习信号，但原始数据分布可能使其被欠采样，需主动提升其训练权重。

**Baseline公式**（均匀采样）：
$$p(\tau) = \frac{1}{N}, \quad \forall \tau \in \mathcal{D}$$
符号：$\mathcal{D}$为清洗后数据集，$N = |\mathcal{D}|$，每条轨迹等概率采样。

**变化点**：深度研究任务中轨迹长度差异显著（短至3-5轮，长至50+轮），均匀采样使长轨迹中的密集信号被稀释。小模型尤其需要从完整长轨迹中学习多步规划。

**本文公式**：（具体策略待补充，原文摘录未完整呈现长时序重采样的数学形式。推测为长度感知或难度感知的加权采样，如$p(\tau) \propto \text{Length}(\tau)^\alpha$或$p(\tau) \propto \mathbb{1}[\text{Length}(\tau) > \tau_{\text{th}}]$）

**对应消融**：

## 实验与分析

主实验在六个深度研究基准上进行，核心结果如下：

| Method | BrowseComp | BrowseComp-ZH | GAIA Text-Only | xBenchDS-2505 | xBenchDS-2510 | DeepSearchQA |
|:---|:---|:---|:---|:---|:---|:---|
| 前沿闭源模型（GPT-4o/Claude等） |  |  |  |  |  |  |
| ≥30B开源智能体（REDSearcher-32B等） |  |  |  |  |  |  |
| ≤9B先前智能体 |  |  |  |  |  |  |
| **DR-Venus-4B-SFT** | 超越≤9B基线 | 超越≤9B基线 | 超越≤9B基线 | 超越≤9B基线 | 超越≤9B基线 | 超越≤9B基线 |
| **DR-Venus-4B-RL** | 进一步缩小与30B差距 | 进一步缩小与30B差距 | 进一步缩小与30B差距 | 进一步缩小与30B差距 | 进一步缩小与30B差距 | 进一步缩小与30B差距 |



**核心发现分析**：

1. **SFT即超越**：DR-Venus-4B-SFT在多数基准上超越所有≤9B先前智能体，说明严格数据清洗的独立价值显著——即使不引入RL，干净的SFT信号已足以让小模型超越使用原始数据训练的同类模型。但具体超越幅度（绝对值或相对百分比）因原文摘录缺少数值而无法量化。

2. **RL持续增益**：DR-Venus-4B-RL在SFT基础上进一步提升，缩小与30B级系统的差距。这一增益归因于轮次级奖励解决了SFT阶段无法覆盖的探索问题：SFT学习"如何模仿成功轨迹"，RL学习"如何通过试错优化工具调用策略"。


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/ab84dcf0-d94a-4def-bf49-2839ab6c3c82/figures/Figure_3.png)
*Figure 3: Figure 2 Pass@K performance of DR-Venus on BrowseComp (left) and BrowseComp-ZH (right).*



**Pass@K分析**（Figure 3）：


4B模型在BrowseComp和BrowseComp-ZH上的Pass@K性能"出人意料地高"（原文描述），暗示小模型的能力上限被严重低估。但具体K值范围（K=1,5,10,20?）和通过率数值均待补充。工具使用分析表明成功轨迹比失败轨迹更依赖browse操作，验证了browse-aware IG分配的设计合理性。

**公平性检查与局限**：
- **基线强度**：论文排除了使用额外上下文管理或测试时扩展技术的同规模竞争方法（RE-TRAC-4B、Marco-DR-8B），可能低估同等规模的竞争上限
- **计算/数据成本**：SFT使用清洗后10K级数据，RL使用1K精选QA对，总数据成本显著低于30B系统，但具体训练GPU小时数待补充
- **消融缺失**：数据清洗各步骤的独立贡献、IGPO轮次级奖励的增量效果、长时序重采样的具体策略均未通过消融实验量化
- **失败案例**：未报告典型失败模式（如是否仍存在于超长轨迹>50轮、多跳推理>5跳等场景）

## 方法谱系与知识库定位

**方法家族**：基于开源数据的小型语言模型Agent训练 → **父方法**：REDSearcher（开源深度研究数据集与32B基线系统）+ IGPO（Iterative Generative Policy Optimization，迭代生成策略优化算法）

**变更插槽**：
| 插槽 | 父方法 | 本文改动 |
|:---|:---|:---|
| **data_curation** | 直接使用REDSearcher原始轨迹 | 四步严格清洗 + 长时序重采样 |
| **training_recipe** | 标准SFT + 轨迹级RL | 掩码SFT + 轮次级密集奖励RL |
| **objective** | 轨迹级稀疏奖励 | 信息增益奖励 + 格式感知正则化 |

**直接基线与差异**：
- **REDSearcher-32B**：同为REDSearcher数据训练，但模型规模32B vs 本文4B；本文核心差异在于证明小模型可通过数据清洗+密集奖励达到可比性能
- **标准IGPO应用于Agent任务**：IGPO原用于通用RLHF，本文将其适配为Agent场景，核心差异为轮次级奖励分解与browse-aware设计
- **RE-TRAC-4B / Marco-DR-8B**：同规模竞争方法，但使用额外上下文管理或测试时扩展技术，本文刻意排除以聚焦"纯训练"方案

**后续方向**：
1. **测试时扩展集成**：当前DR-Venus未使用推理时扩展（如多次采样投票、树搜索），与RE-TRAC-4B等方法的公平对比需补充
2. **数据清洗自动化**：当前四步清洗为规则驱动，可探索基于模型反馈的自动质量评估替代人工规则
3. **跨模态扩展**：当前为Text-Only设置，视觉信息增益奖励的设计可扩展至多模态深度研究

**知识库标签**：
- **modality**：text-only / web-browsing
- **paradigm**：agentic LLM / tool-use / deep-research
- **scenario**：edge-scale deployment / low-resource training / open-data only
- **mechanism**：data curation / per-turn reward shaping / information gain
- **constraint**：≤9B parameters / ≤10K open data / no closed-source data

