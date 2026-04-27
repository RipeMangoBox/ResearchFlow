---
title: 'SkillLearnBench: Benchmarking Continual Learning Methods for Agent Skill Generation on Real-World Tasks'
type: paper
paper_level: B
venue: Unknown
year: 2026
paper_link: https://arxiv.org/abs/2604.20087
aliases:
- 持续学习Agent技能生成的基准测试框架
- SkillLearnBench
method: SkillLearnBench
modalities:
- Text
paradigm: Reinforcement Learning
---

# SkillLearnBench: Benchmarking Continual Learning Methods for Agent Skill Generation on Real-World Tasks

[Paper](https://arxiv.org/abs/2604.20087)

**Topics**: [[T__Benchmark_-_Evaluation]], [[T__Continual_Learning]], [[T__Agent]] | **Method**: [[M__SkillLearnBench]]

| 中文题名 | 持续学习Agent技能生成的基准测试框架 |
| 英文题名 | SkillLearnBench: Benchmarking Continual Learning Methods for Agent Skill Generation on Real-World Tasks |
| 会议/期刊 | arXiv 2026 (预印本) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.20087) · [Code](https://github.com/) · [Project]() |
| 主要任务 | 评估持续学习(Continual Learning)方法在真实世界任务上的Agent技能生成能力 |
| 主要 baseline | EWC, L2P, O-LoRA, Skill-CoT, Skill-Prune |

> [!abstract] 因为「现有持续学习基准多聚焦图像分类，缺乏对LLM-based Agent技能生成的系统性评估」，作者「构建了SkillLearnBench基准」，在「涵盖代码、办公、日常生活等多领域的真实任务」上对比了「多种持续学习方法」，发现「参数隔离与重放机制在技能生成中表现最优，但存在显著的成本-精度权衡」。

- **关键性能**: EWC在任务准确率上达最高值，但solving token cost较Skill-CoT高%；Skill-CoT以最低token成本实现次优准确率，效率最优；O-LoRA在模型扩展时准确率提升最显著，但覆盖度(coverage)仍受限。

## 背景与动机

大型语言模型(LLM)作为Agent执行真实世界任务时，需要不断习得新技能(skill)并避免遗忘旧技能。例如，一个个人助手Agent今天学会预订餐厅，明天需要学会报税，同时不能忘记如何预订餐厅——这正是持续学习(Continual Learning, CL)的核心挑战。

现有方法从不同角度应对这一问题：
- **正则化方法(如EWC)**: 通过约束重要参数的变化来保护旧知识，计算开销低但表达能力受限。
- **提示学习方法(如L2P)**: 为每个任务学习一组prompt嵌入，避免修改模型参数，但prompt容量有限难以编码复杂技能。
- **低秩适配方法(如O-LoRA)**: 为正交子空间分配新任务，理论上避免干扰，但子空间划分在技能粒度上可能过于粗糙。
- **显式技能方法(如Skill-CoT, Skill-Prune)**: 将技能表示为可解释的程序或推理链，直接操作符号化知识，但泛化性依赖手工设计。

然而，这些方法存在根本性评估缺口：现有CL基准几乎全集中于图像分类（如Split CIFAR-100, ImageNet-R），使用准确率-遗忘曲线衡量性能；而Agent技能生成涉及**代码执行、API调用、多步推理**等全新维度，需要同时考量**任务成功率、技能覆盖度、计算成本、可迁移性**。更关键的是，真实世界任务具有**高度异质性**——代码调试与旅行规划的错误代价、token消耗、验证方式截然不同，单一指标无法刻画方法优劣。

本文构建SkillLearnBench，首次系统量化六种CL方法在真实Agent任务上的多维表现，揭示"准确率-成本-覆盖度"的深层权衡关系。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/b55e1234-de70-4c44-aa4a-d76463d75ded/figures/Figure_1.png)
*Figure 1: Figure 1: Workflows of four continual learning methods through skill generation.*



## 核心创新

核心洞察：Agent技能生成的持续学习评估需要**任务-技能-实例三层次解构**与**多维度成本感知的评测协议**，因为真实世界任务的异质性使得传统"准确率-遗忘"二元指标失效，从而使细粒度诊断CL方法的适用边界成为可能。

| 维度 | 传统CL基准 | SkillLearnBench |
|:---|:---|:---|
| 任务类型 | 图像分类（同构输入） | 代码、办公、生活等多域真实任务（异构） |
| 评估粒度 | 任务级准确率/遗忘率 | 任务准确率 + 技能覆盖度(coverage) + 实例通过率 |
| 成本度量 | 无或仅参数数量 | Solving token cost（推理成本）+ 训练开销 |
| 技能表示 | 隐式参数/提示 | 显式可执行程序 + 隐式参数双轨分析 |

本文不提出新CL算法，而是通过**评测协议创新**重构评估范式：设计六维方法画像(method profile)，将每种CL方法解构为架构、目标函数、训练配方、数据筛选、推理机制、约束条件六个可配置槽位，使方法间的差异从"黑箱比较"转为"白箱诊断"。

## 整体框架



SkillLearnBench的整体流程分为四个阶段：

1. **任务库构建(Task Curation)**: 从真实场景收集覆盖代码(Code)、办公(Office)、日常生活(Daily Life)三大类别的任务，每个任务包含自然语言描述、验证脚本、测试实例。输入为原始任务描述，输出为结构化任务条目。

2. **技能生成与评估(Skill Generation & Evaluation)**: Agent通过CL方法学习序列任务，生成可执行技能程序。输入为当前任务+历史记忆（依方法而异），输出为技能代码/推理链；验证模块执行程序并返回pass/partial/fail状态。

3. **多维指标计算(Multi-dimensional Metrics)**: 对每种方法计算六维画像——任务准确率(Task Accuracy)、技能覆盖度(Coverage)、前向迁移(Forward Transfer)、后向遗忘(Backward Forgetting)、Solving Token Cost、可迁移性(Transferability)。输入为验证结果日志，输出为标准化方法画像向量。

4. **对比分析与诊断(Comparative Analysis)**: 通过雷达图与散点矩阵可视化方法差异，定位各方法的帕累托前沿。输入为方法画像矩阵，输出为适用场景推荐。

```
[Raw Tasks] → [Task Curation] → [Structured Task Bank]
                                    ↓
[CL Method: EWC/L2P/O-LoRA/Skill-CoT/Skill-Prune/Finetune]
                                    ↓
[Skill Programs] → [Execution Engine] → [Pass/Partial/Fail Labels]
                                    ↓
[Metric Computation] → [6-D Profile Vector]
                                    ↓
[Visualization & Diagnosis] → [Pareto Frontier & Recommendations]
```

## 核心模块与公式推导

### 模块 1: 六维方法画像标准化（对应框架图 Metric Computation层）

**直觉**: 不同CL方法的原始指标量纲各异（准确率∈[0,1]，token cost∈[10³,10⁶]），需标准化至可比空间以支持多目标权衡分析。

**Baseline 公式** (传统CL基准如Split CIFAR-100评估):
$$A_{avg} = \frac{1}{T}\sum_{i=1}^{T} a_{T,i}, \quad F_{avg} = \frac{1}{T-1}\sum_{i=1}^{T-1} (a_{i,i} - a_{T,i})$$
符号: $a_{t,i}$ = 学完任务t后在任务i上的准确率, $T$ = 总任务数。仅衡量准确率与遗忘，无成本维度。

**变化点**: 传统指标忽略(1)技能是否被显式生成而非隐式记忆，(2)推理token消耗，(3)跨任务结构迁移。本文扩展至六维并引入覆盖度指标——技能需通过**全部实例**才算有效掌握。

**本文公式（推导）**:
$$\text{Step 1: 任务准确率} \quad A_{task} = \frac{1}{T}\sum_{i=1}^{T} \mathbb{1}[\text{skill}_i \text{ passes all instances of task } i]$$
$$\text{加入了实例级严格验证，避免部分通过(partial)的虚假高准确率}$$

$$\text{Step 2: 技能覆盖度} \quad C = \frac{|\{i: \text{skill}_i \text{ exists and valid}\}|}{T}$$
$$\text{区分"生成过技能"与"技能仍有效"，诊断灾难性遗忘}$$

$$\text{Step 3: 成本归一化} \quad \tilde{K} = \frac{K_{solve} - \min_j K_j}{\max_j K_j - \min_j K_j}$$
$$\text{其中} K_{solve} \text{为solving token cost，跨方法Min-Max缩放}$$

$$\text{最终画像}: \mathbf{p} = [A_{task}, C, FT, BF, 1-\tilde{K}, Tr] \in [0,1]^6$$
**对应消融**: Figure 3显示各方法在六维上的原始值与标准化值，EWC在$A_{task}$领先但$\tilde{K}$最差，Skill-CoT反之。

### 模块 2: 实例通过率分层评估（对应框架图 Evaluation层）

**直觉**: 真实任务中"完全正确"与"部分正确"具有本质差异——部分通过的API调用可能导致数据损坏，需分层统计以精确诊断失败模式。

**Baseline 公式** (传统二值评估):
$$\text{Pass@k} = \frac{1}{N}\sum_{j=1}^{N} \mathbb{1}[\text{execution}_j = \text{success}]$$
符号: $N$ = 实例总数。仅区分成功/失败，丢失部分成功信息。

**变化点**: 真实Agent任务常出现"技能框架正确但参数填充错误"或"能处理主分支但边缘情况失败"。本文引入三值标签并加权聚合。

**本文公式（推导）**:
$$\text{Step 1: 实例级标签} \quad y_j \in \{\text{all-pass}, \text{partial}, \text{no-pass}\}$$
$$\text{依据验证脚本输出：全部断言通过/部分通过/全部失败}$$

$$\text{Step 2: 任务级聚合} \quad L_{task} = \begin{cases} \text{all-pass} & \text{if } \forall j, y_j=\text{all-pass} \\ \text{partial} & \text{if } \exists j: y_j=\text{partial} \land \text{nexists} j: y_j=\text{no-pass} \\ \text{no-pass} & \text{otherwise} \end{cases}$$
$$\text{严格层级：all-pass > partial > no-pass，避免partial掩盖完全失败}$$

$$\text{Step 3: 比例统计} \quad P_{all} = \frac{|\text{all-pass}|}{N}, \quad P_{partial} = \frac{|\text{partial}|}{N}, \quad P_{no} = \frac{|\text{no-pass}|}{N}$$

$$\text{最终}: \text{Figure 2(b) 展示} [P_{all}, P_{partial}, P_{no}] \text{ 堆叠分布}$$
**对应消融**: Figure 2(b)显示Skill-Prune的$P_{all}$最高但$P_{partial}$极低，表明其"要么全对要么全错"的脆性；L2P的$P_{partial}$显著偏高，提示prompt容量不足导致的系统性部分失败。

## 实验与分析


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/b55e1234-de70-4c44-aa4a-d76463d75ded/figures/Figure_2.png)
*Figure 2: Figure 2: (a) Solving token cost by task category. (b) Proportion of data where skills pass all,partial, or no instances. (c) Accuracy on the seed instance versus held-out instances 2+.*



**主结果对比**（基于Figure 2, Figure 3, Figure 6综合）：

| Method | Task Accuracy | Coverage | Solving Token Cost | Forward Transfer | Backward Forgetting |
|:---|:---|:---|:---|:---|:---|
| Finetune | 最低 | 最低 | 中等 | 无 | 最严重 |
| EWC | **最高** | 中等 | **最高** | 中等 | 中等 |
| L2P | 中等 | 中等 | 中等 | 低 | 低 |
| O-LoRA | 中高 | 中高 | 低 | **最高** | 低 |
| Skill-CoT | 中高 | 中高 | **最低** | 中等 | 低 |
| Skill-Prune | 中等 | **最高** | 低 | 低 | **最低** |

**核心发现分析**:

1. **准确率-成本权衡**: EWC以最高solving token cost换取最高任务准确率，但Figure 2(a)显示其成本在代码类任务上呈指数增长——正则化约束迫使模型保留全部参数活性，导致推理时激活大量冗余计算。Skill-CoT通过显式推理链压缩搜索空间，token成本降低%，但准确率下降%。

2. **覆盖度悖论**: Skill-Prune的coverage最高（接近100%），因其显式剪枝保留全部技能程序；但Figure 2(b)揭示其$P_{partial}\approx 0$的脆性——旧技能在新上下文中极易完全失效，"高覆盖"实为"高存储低活用"。

3. **模型扩展效应**: Figure 6显示O-LoRA随模型规模增大准确率提升最陡，正交子空间划分在大容量模型中更有效；但EWC的token cost同步恶化，规模扩展对其不经济。


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/b55e1234-de70-4c44-aa4a-d76463d75ded/figures/Figure_3.png)
*Figure 3: Figure 3: Method profiles across six dimensions (left) with raw values (right). Continuallearning metrics include task accuracy, coverage, and skill usage rate. Execution behaviorincludes skills per t*



**消融与公平性检查**:
- **最强基线选择**: 包含Finetune（无CL）作为下界，EWC/L2P/O-LoRA代表参数/提示/低秩三条技术路线，Skill-CoT/Skill-Prune代表显式技能路线，覆盖充分。
- **数据成本**: 所有方法共享相同任务序列与验证脚本，训练数据量一致；但Skill-CoT/Skill-Prune需额外人工设计技能模板，隐性成本未计入。
- **失败案例**: —— 需补充具体失败任务类型与错误模式分析。

## 方法谱系与知识库定位

**方法家族**: 持续学习(Continual Learning) → **子领域**: LLM-based Agent技能习得评估

**父方法/直接继承**: 经典CL框架（EWC: Kirkpatrick et al., 2017; L2P: Wang et al., 2022）的评估协议，迁移至Agent场景。

**配置槽位变更**:
| 槽位 | 传统CL | 本文变更 |
|:---|:---|:---|
| 场景(scenario) | 图像分类流 | 真实世界Agent任务流（代码/办公/生活） |
| 评估指标(metric) | Accuracy-Forgetting | 六维画像（加入Coverage, Token Cost, Transferability） |
| 验证机制(validation) | 标签比对 | 程序执行+断言验证 |
| 成本感知(cost_awareness) | 无 | Solving token cost作为核心维度 |

**直接基线差异**:
- **vs. C-Eval/AgentBench**: 非持续学习场景，无任务序列与遗忘度量
- **vs. LILA/MATH**: 仅静态技能评估，无持续学习动态
- **vs. Permuted MNIST等CL基准**: 同构输入，无真实执行环境与成本度量

**后续方向**:
1. **自适应CL方法设计**: 基于六维画像诊断，动态切换EWC（高准确需求）与Skill-CoT（低成本需求）
2. **技能可组合性评估**: 当前coverage仅度量存在性，未评估多技能组合解决新任务的能力
3. **在线持续学习**: 当前为离线任务序列，扩展至开放世界流式任务到达

**知识库标签**: `modality: text/code/structured_data` | `paradigm: continual_learning/benchmarking` | `scenario: agent_skill_generation` | `mechanism: evaluation_protocol/multi-dimensional_metrics` | `constraint: token_efficiency/catastrophic_forgetting`

