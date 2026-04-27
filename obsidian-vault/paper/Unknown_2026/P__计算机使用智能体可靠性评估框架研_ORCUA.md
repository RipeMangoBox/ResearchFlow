---
title: On the Reliability of Computer Use Agents
type: paper
paper_level: B
venue: Unknown
year: 2026
paper_link: https://arxiv.org/abs/2604.17849
aliases:
- 计算机使用智能体可靠性评估框架研究报告
- ORCUA
- 可靠性不等于性能——一个能偶尔成功的智能体与一个能始终成功的智能体之间
---

# On the Reliability of Computer Use Agents

[Paper](https://arxiv.org/abs/2604.17849)

**Topics**: [[T__Agent]], [[T__Benchmark_-_Evaluation]], [[T__Reasoning]] | **Datasets**: OSWorld, OSWorld - Deterministic Decoding Intervention, OSWorld - Strategy Determinism Intervention, OSWorld - Environment Perturbation

> [!tip] 核心洞察
> 可靠性不等于性能——一个能偶尔成功的智能体与一个能始终成功的智能体之间存在本质差距。消除随机性（确定性解码）并不能提升可靠性，反而可能削弱智能体对环境微小变化的适应能力。真正的可靠性需要同时解决三个相互交织的来源：执行随机性、任务歧义性和规划变异性。本文的核心贡献在于提供了一套能够区分这三类来源、并在任务级别检测可靠性变化的评估工具，从而将「可靠性」从模糊概念转化为可测量、可分解的研究对象。

## 基本信息

**论文标题**: On the Reliability of Computer Use Agents

**作者信息**: （未在提取数据中提供完整作者列表）

**发表场所**: （未明确提供）

**年份**: （未明确提供）

**代码/数据链接**: （未在提取数据中提供）

**基准环境**: OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments

## 核心主张

本文核心主张：当前计算机使用智能体的评估指标（单次运行、平均多轮运行、Best-of-N）无法捕捉可靠性问题，即智能体在同一任务上多次执行时成功率波动剧烈。论文提出**Pass^k可靠性指标**、**可靠性转换统计(b-c)** 和**受控干预实验框架**，首次系统量化并分解智能体不可靠性的来源（token采样随机性、决策重规划随机性、环境观测扰动）。关键证据：六种主流模型（Claude/GPT-5/Kimi/Qwen/OpenCUA/UI-TARS-1.5）在OSWorld上均表现出显著的跨运行方差，确定性解码对Qwen产生负面影响（b-c=-20*）而对OpenCUA/UI-TARS-1.5产生正面影响（b-c=+20*/+19*），证明可靠性问题具有模型特异性。置信度：0.92。

## 研究动机

计算机使用智能体（Computer-Use Agents）在OSWorld等基准上取得了显著进展，但存在根本性评估盲区：现有工作仅报告**Pass^1（单次成功率）**或**Best-of-N（多轮最优）**，忽略了智能体在重复执行同一任务时的**成功率波动**。这一盲区导致：

1. **无法区分"偶尔成功"与"可靠成功"**：一个智能体可能50%任务一次成功，但从未稳定复现；另一个可能80%任务稳定成功——两者Pass^1可能相同但可靠性迥异。

2. **无法定位不可靠根源**：是token采样导致的执行路径分叉？还是每次重新规划引入的决策变异？或是环境微小变化引发的连锁失败？

论文填补了这一评估方法论空白，借鉴软件工程中的可靠性测试思想，首次为智能体领域建立了形式化的可靠性评估体系。

## 方法流程

基于提取的pipeline_modules，方法流程如下：

```
OSWorld任务采样 → 条件干预分配 → n次重复执行 → 成功计数聚合 → 可靠性指标计算 → 统计显著性检验
```

**模块详解**：

| 模块 | 输入 | 输出 | 是否创新 |
|:---|:---|:---|:---|
| 1. 任务采样 | OSWorld任务实例 x=(s₀, I) | 初始状态与指令 | 否 |
| 2. 重复执行循环 | 任务x, 策略π, 运行次数n | 二元成功指标 r_{x,1},...,r_{x,n} | **是**（替代单次/Best-of-N评估） |
| 3. 确定性解码干预 | π, x, temperature=0 | 确定性轨迹τ | 否（技术借用） |
| 4. 策略确定性干预 | π, x | 固定高层计划 + 确定性执行轨迹 | **是**（隔离决策变异） |
| 5. 环境扰动干预 | x, 非功能性观测变异 | 扰动下轨迹 | **是**（隔离执行随机性） |
| 6. 可靠性指标计算 | 所有任务的c_x | Pass^k, b-c, Δc_x | **是**（替代标准准确率） |

**干预设计逻辑**：通过控制变量法，逐层剥离三种随机性来源——(3)消除token采样变异，(4)消除重规划决策变异，(5)引入受控环境变异观测执行敏感性。

## 关键公式

**1. 基础框架（借用）**

$$\mathcal{M} = \langle \mathcal{S}, \mathcal{O}, \mathcal{A}, \mathcal{T}, \mathcal{I}, R \rangle$$

计算机使用智能体的MDP形式化，包含状态空间、观测空间、动作空间、转移函数、指令空间和奖励函数。

**2. 核心创新：Pass^k可靠性指标**

$$\text{Pass}\hat{\ }k = \mathbb{E}_{x \sim \mathcal{X}} \left[ \frac{\text{binom}{c_x}{k}}{\text{binom}{n}{k}} \right]$$

> 任务在n次独立运行中至少成功k次的期望概率。通过超几何组合数比率计算，**替代传统Pass^1**。当k=n时为最严格的完全可靠性标准。

**3. 基线指标（被扩展）**

$$\text{Pass}\hat{\ }1 = \mathbb{E}_{x \sim \mathcal{X}} \left[ \frac{c_x}{n} \right]$$

传统单次/平均成功率，等价于成功次数的期望比例。

**4. 可靠性转换统计（创新）**

$$b = \sum_{x} \mathbf{1}[z_x^{(\text{base})} = 0,\ z_x^{(\text{new})} = 1], \quad c = \sum_{x} \mathbf{1}[z_x^{(\text{base})} = 1,\ z_x^{(\text{new})} = 0]$$

> b: 不可靠→可靠的任务数；c: 可靠→不可靠的任务数。z_x = 1[c_x=n]为完全可靠性指示变量。

**5. McNemar检验（创新应用）**

$$\chi^2 = \frac{(b - c)^2}{b + c}$$

检验干预效果的方向性显著性，表中*标记p<0.05。

**6. 平均可靠性差异（创新）**

$$\Delta c_x = \frac{1}{|\mathcal{X}|} \sum_{x \in \mathcal{X}} (c_x^{(\text{new})} - c_x^{(\text{base})})$$

效应量度量，与b-c配合使用。

**公式谱系**：MDP框架借用自强化学习标准形式；Pass^k由Pass^1通过组合概率扩展；McNemar检验借用自配对分类检验；其余均为本文首创。

## 实验结果

**基准环境**: OSWorld（真实计算机环境开放任务基准）

**表1：确定性解码与策略确定性干预（开源模型）**

| 模型 | 基线Pass^1 | 基线Pass^3 | 确定性解码b-c | 策略确定性b-c |
|:---|:---|:---|:---|:---|
| Qwen (S3) | 0.329 | 0.222 | **-20\*** | -1 |
| OpenCUA | 0.226 | 0.125 | **+20\*** | +23\* |
| UI-TARS-1.5 | 0.253 | 0.152 | **+19\*** | +14\* |

> \*表示p<0.05显著。关键发现：**消除token采样随机性的效果具有模型特异性**——Qwen可靠性显著下降（可能依赖采样多样性探索），而OpenCUA/UI-TARS-1.5显著改善。

**表2：环境扰动干预（前沿模型）**

| 模型 | b-c | Δc_x |
|:---|:---|:---|
| GPT-5 | -10 | -0.033 |
| Claude | **-20\*** | **-0.080\*** |
| Kimi | -1 | +0.025 |

> Claude对环境扰动最敏感（Δc_x=-0.080），Kimi几乎不受影响。

**消融结论**：
- 确定性解码：混合效应，不支持"消除随机性必提升可靠性"的朴素假设
- 策略确定性：缓解Qwen退化但不超越全随机基线
- 环境扰动：所有模型均有一定敏感性，但程度差异大

**证据强度评估**: 0.78。局限：仅OSWorld单环境；前沿模型未测试确定性干预；统计检验样本量未明确报告。

## 相关工作

按角色分类的关键参考文献：

**1. 主要基线方法**
- **OSWorld** (Xie et al.): 核心评估基准，提供开放任务环境。本文批判其评估指标（单次/平均/Best-of-N）无法捕捉可靠性。
- **Agent S3**: 实验设置来源（动作空间、grounding方案），作为次级基线框架被采用而非被超越。

**2. 被评估的对比模型**
| 模型 | 类型 | 关键发现 |
|:---|:---|:---|
| Claude Sonnet 4.6 | 前沿闭源 | 环境扰动最敏感（b-c=-20*） |
| GPT-5 | 前沿闭源 | 中等敏感，仅测试扰动条件 |
| Kimi K2.5 | 视觉智能体 | 扰动下最稳健（b-c=-1） |
| Qwen3-VL | 开源VLM | 确定性解码导致可靠性崩溃 |
| OpenCUA | 开源CUA | 确定性解码显著获益 |
| UI-TARS-1.5 | 开源GUI模型 | 双重干预均获益 |

**3. 技术来源**
- **温度采样/随机解码**: 通用LLM推理技术，本文将其作为干预变量系统研究
- **McNemar检验/Wilcoxon检验**: 经典非参数统计方法，本文创新应用于智能体可靠性评估

**关系定位**：本文并非提出新的智能体架构，而是**评估方法论论文**——为现有智能体提供可靠性诊断工具。

## 方法谱系

本文方法在演化链中的位置：**评估方法论节点**，非架构节点。

**继承自父方法**：
- **OSWorld评估协议**: 继承任务定义、环境接口、成功判定标准
- **Agent S3执行框架**: 继承动作空间设计、grounding机制、多模态输入处理
- **经典统计检验**: 继承McNemar检验、Wilcoxon符号秩检验的数学形式
- **LLM确定性解码**: 继承temperature=0、batch-invariant inference的技术实现

**核心修改（slot-level）**：

| Slot | 父方法值 | 本文修改值 | 修改类型 |
|:---|:---|:---|:---|
| **objective** | 最大化Pass^1平均成功率 | Pass^k + b-c转换统计 + Δc_x差异度量 | **替换（创新）** |
| **inference_strategy** | 随机解码（temp>0），每轮独立重规划 | 确定性解码(temp=0) + 策略确定性（固定高层计划） | 修改（部分创新） |
| **data_pipeline** | 单次或独立重复执行 | 受控重复执行 + 环境扰动干预 | 修改（创新） |

**未建立lineage_edge的原因**：本文是**横向评估方法论创新**，非纵向架构改进——不继承自某个特定"父方法"的完整架构，而是跨多个现有方法施加统一评估框架。应创建method_node但无需lineage_edge。

## 局限与展望

**论文明确陈述的局限**：
1. **单环境限制**：所有实验仅在OSWorld进行，未验证是否推广到Web（VisualWebArena）、移动端或其他操作系统
2. **策略确定性可扩展性**：需要类似oracle的计划生成，实际部署中可能难以自动获取高质量高层计划
3. **前沿模型测试不完整**：GPT-5、Claude、Kimi仅测试环境扰动，未测试确定性解码和策略确定性干预

**分析推断的额外局限**：
4. **统计效力未报告**：b-c检验的样本量（任务数）和功效分析缺失，部分*显著性可能边缘
5. **n值选择敏感性**：Pass^k依赖运行次数n，n=3或n=5的选择对结论影响未讨论
6. **"非功能性"扰动定义模糊**：环境扰动的具体实现（像素级？OCR噪声？时间戳变化？）未充分技术披露
7. **成本与可复现性**：多次运行（n repetitions）+ 单独计划生成，评估成本显著高于标准协议

**未来方向**：
- 扩展至WebArena、Mind2Web等跨域基准验证框架普适性
- 自动化计划生成（如利用 stronger model 蒸馏）降低策略确定性成本
- 建立可靠性-效率帕累托前沿，指导实际部署的n值选择
- 探索可靠性作为训练目标（非仅评估指标），开发可靠性感知的强化学习算法

## 知识图谱定位

本文在知识图谱中的**枢纽位置**：连接"评估方法论"与"计算机使用智能体"的桥梁节点。

**触碰的任务节点**：
- `computer-use agent evaluation`（核心）：首次引入可靠性维度
- `operating system task automation`（应用层）：OSWorld具体实例

**触碰的方法节点**：
- **本文核心方法** `Reliability Evaluation Framework for Computer-Use Agents`：新建评估范式
- **机制节点**：`Pass^k metric`（核心创新）、`strategy determinism`（干预工具）、`deterministic decoding`（干预工具）、`environment perturbation`（干预工具）、`reliability transition statistics`（统计工具）
- **被评估基线**：Agent S3、Qwen3-VL、OpenCUA、UI-TARS-1.5、GPT-5、Claude Sonnet 4.6、Kimi K2.5

**触碰的数据集节点**：
- `OSWorld`（唯一评估场域，形成强绑定）

**对领域结构的贡献**：
1. **分解评估维度**：将智能体评估从单一"能力维度"（能否完成任务）扩展为"能力×可靠性"二维空间
2. **建立干预语言**：提供deterministic decoding / strategy determinism / environment perturbation三种标准干预，使后续研究可对比引用
3. **揭示模型异质性**：发现不同架构（VLM vs. 专用CUA vs. 通用LLM）对随机性来源的敏感性差异，暗示不存在通用"最优解码策略"
4. **方法-任务解耦**：评估框架可迁移至任何具有可重复执行特性的智能体任务，不仅限于计算机使用


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/5b50b3ba-0f28-4f95-b30b-aea77f14a73b/figures/Figure_1.png)
*Figure 1: Figure 1 | (left) Performance of a strong computer-use agent (Agent S3 with GPT-5) across repeatedattempts. While Pass@10 reaches approximately 78%, the corresponding Pass^10 indicates that theagent s*


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/5b50b3ba-0f28-4f95-b30b-aea77f14a73b/figures/Figure_2.png)
*Figure 2: Figure 2 | We illustrate three metrics for analyzing consistency in agent performance over multipleruns of the same task. (a) Pass^k (repeated-run success) estimates the probability that 𝑘executionsof*


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/5b50b3ba-0f28-4f95-b30b-aea77f14a73b/figures/Figure_3.png)
*Figure 3: Figure 3 | Task-level transitions in reliability under instruction clarification measured using McNemaranalysis (left), and repeated-run success under clarified and unclarified instructions measured u*


