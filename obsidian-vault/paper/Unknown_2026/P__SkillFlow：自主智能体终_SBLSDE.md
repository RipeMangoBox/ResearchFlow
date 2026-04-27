---
title: SkillFlow:Benchmarking Lifelong Skill Discovery and Evolution for Autonomous Agents
type: paper
paper_level: B
venue: Unknown
year: 2026
paper_link: https://arxiv.org/abs/2604.17308
aliases:
- SkillFlow：自主智能体终身技能发现与演化的系统性基准评测
- SBLSDE
- SkillFlow 的核心直觉是：技能演化能力是模型能力的一个独立维度
---

# SkillFlow:Benchmarking Lifelong Skill Discovery and Evolution for Autonomous Agents

[Paper](https://arxiv.org/abs/2604.17308)

**Topics**: [[T__Agent]], [[T__Benchmark_-_Evaluation]], [[T__Continual_Learning]] | **Datasets**: SkillFlow

> [!tip] 核心洞察
> SkillFlow 的核心直觉是：技能演化能力是模型能力的一个独立维度，无法从静态任务完成率推断。强模型能将经验外化为紧凑可复用的程序，并在后续任务中持续精炼；弱模型则陷入「创建-复用协调失败」——生成大量碎片化技能但无法有效整合，导致后续任务认知负担加重而非减轻。高技能使用率既不等于高技能质量，也不等于高任务完成率。这一发现的有效性来自 DAEF 提供的受控迁移条件：在工作流拓扑一致的任务家族内，技能迁移的成败可被归因于模型的自我演化能力，而非任务表面差异。

## 基本信息

**论文标题**: SkillFlow: Benchmarking Lifelong Skill Discovery and Evolution for Autonomous Agents

**作者**: （未在提取数据中明确列出）

**发表场所**: （未在提取数据中明确列出）

**年份**: （未在提取数据中明确列出）

**代码/数据链接**: （未在提取数据中明确列出）

**核心数据集**: 166个任务，跨越20个任务家族，基于DAEF框架组织

## 核心主张

本论文提出**SkillFlow**是首个系统性评测自主智能体**发现、演化和维护技能**能力的基准测试。核心主张为：现有基准仅测试模型能否使用给定技能，无法评估从经验中发现技能、失败后修复技能以及长期维护连贯技能库的能力。关键证据包括：（1）Agentic Lifelong Learning协议，要求智能体从空技能库开始，通过执行轨迹和规则反馈迭代生成技能补丁；（2）DAEF框架实现跨领域任务结构统一；（3）覆盖11个前沿模型的实验显示Claude Opus 4.6在完成率上获得+8.43的绝对提升。**置信度：0.95**

## 研究动机

当前智能体评估存在关键空白：**SkillsBench**、**SWE-Skills-Bench**和**SkillCraft**等基准仅测试模型能否使用**预先提供的技能**，忽略了智能体在真实场景中必须面对的三大挑战：（1）从零发现技能；（2）失败后修复技能；（3）长期维护技能库的连贯性。具体而言，现有工作采用**静态技能供给**模式，无法捕捉技能随任务序列演化的动态过程。此外，任务集合缺乏结构化组织，导致跨领域评估困难。SkillFlow填补了这一空白，将评估范式从"技能使用能力"转向"终身技能学习能力"。

## 方法流程

SkillFlow的Agentic Lifelong Learning协议包含五个核心模块：

```
[DAEF任务家族] → [空技能库 S₀]
       ↓
[首个任务τ₁ 无技能执行] → [执行轨迹 + 规则反馈r₁]
       ↓
[Model_g(∅, τ₁, r₁)] → [技能补丁Δ₁]
       ↓
[Apply(Δ₁, ∅)] → [初始技能库 S₁]
       ↓
循环：对于后续任务τ_t
  [使用S_{t-1}执行任务] → [轨迹与反馈r_t]
       ↓
  [Model_g(S_{t-1}, τ_t, r_t)] → [Δ_t]
       ↓
  [Apply(Δ_t, S_{t-1})] → [S_t]
```

**创新模块**：DAEF框架（任务结构形式化）、Model_g技能补丁生成器、Apply库更新函数、单轮标准化输出格式。

## 关键公式

**1. 任务结构图（DAEF基础）— 新颖**
```
\mathcal{T} = (V, E, \lambda, \gamma)
```
V为操作步骤节点，E为依赖边，λ/γ为节点/边标签函数。

**2. 领域无关工作流映射 — 新颖**
```
\mathcal{F} = \phi(\mathcal{T}) = (V_F, E_F, \lambda_F)
```
通过φ将具体任务抽象为跨域统一工作流。

**3. 迭代技能库演化 — 核心创新**
```
\Delta_t = \text{Model}_g(\mathcal{S}_{t-1}, \tau_t, r_t), \quad \mathcal{S}_t = \text{Apply}(\Delta_t, \mathcal{S}_{t-1})
```
Model_g根据前一库状态、当前任务和规则反馈生成补丁，Apply函数更新库状态。

**推导路径**：任务结构图 → DAEF抽象映射 → 初始补丁生成 → 递归推广为终身学习动态方程。

**借用公式**：边集定义 $E \subseteq V \times V$（标准图论）。

## 实验结果

在SkillFlow基准上测试11个模型变体，主要结果（任务完成率%comp.）：

| 模型 | Vanilla | Skills Evolve | 变化 |
|:---|:---|:---|:---|
| Claude Opus 4.6 | 62.65 | **71.08** | **+8.43** ✓ |
| Claude Sonnet 4.5 | 49.40 | **55.42** | **+6.02** ✓ |
| MiniMax M2.5 | 28.31 | **34.94** | **+6.63** ✓ |
| Claude Opus 4.5 | 58.43 | **60.84** | +2.41 ✓ |
| Kimi K2.5 | 55.42 | **56.02** | +0.60 ✓ |
| Claude Sonnet 4.6 | 56.63 | 56.63 | 0.00 — |
| MiniMax M2.7 | **37.35** | 36.75 | -0.60 ✗ |
| Qwen-Coder-Next | **45.18** | 44.58 | -0.60 ✗ |
| Qwen3-Coder-480B | **24.70** | 24.10 | -0.60 ✗ |
| GPT 5.3 Codex | **52.41** | 46.39 | -6.02 ✗ |

**消融发现**：技能演化效果呈**模型依赖性**，仅部分模型受益，部分出现退化。**证据强度：0.75**（存在包装器特异性方差、单轮格式限制等潜在问题）。

## 相关工作

按角色分类的关键参考文献：

**直接基线（数值对比缺失）**：
- **SkillsBench**：技能多样性评估的主要对标基准，SkillFlow直接继承其评估目标但扩展为动态演化场景
- **SWE-Skills-Bench**：软件工程领域技能基准，验证技能在实际开发中的效用
- **SkillCraft**：LLM智能体工具学习技能基准，关注技能习得过程

**内部基线（有数值对比）**：
- **vanilla agent execution**：各模型的无技能执行版本，Table 1中直接对比

**方法组件来源**：
- **"Internalizing meta-experience into memory for guided reinforcement learning in large language models"**：元经验记忆机制的方法来源

**重要关系**：SkillFlow与三大基线的核心差异在于——它们测试"能否使用给定技能"，而SkillFlow测试"能否从经验中发现并演化技能"。

## 方法谱系

**谱系位置**：SkillFlow ← **extends** ← **SkillsBench**

**继承内容**：
- 从SkillsBench继承：跨领域技能评估的核心目标、多样化任务测试理念

**关键修改（slot-level）**：
| Slot | SkillsBench值 | SkillFlow值 | 修改类型 |
|:---|:---|:---|:---|
| **evaluation_protocol** | 静态技能供给或单任务无演化评估 | Agentic Lifelong Learning协议：序列任务求解、轨迹驱动补丁生成、迭代库更新 | **替换** |
| **data_pipeline** | 临时任务集合，无结构化工作流对应 | DAEF框架：166任务/20家族，共享一致工作流结构 | **替换** |
| **inference_strategy** | 直接执行（有技能或无技能） | Model_g(S_{t-1}, τ_t, r_t)单轮标准化输出，迭代演化 | **替换** |
| **architecture** | 单体智能体，无外部技能库 | 可外部化技能库S_t，跨任务序列累积补丁Δ_t | **修改** |

**本质变化**：从"技能使用测试"进化为"终身技能学习测试"，引入时间维度和状态持续性。

## 局限与展望

**论文明确/分析推断的局限**：

1. **模型依赖性风险**：技能演化效果高度不均——Claude Opus 4.6提升+8.43，但GPT 5.3 Codex退化-6.02，MiniMax M2.7/Qwen系列亦退化，表明协议并非普适增益
2. **评估环境限制**：仅限命令行环境，可能无法泛化到GUI或多模态场景
3. **单轮格式约束**：Model_g的标准化单轮输出可能不利于需多步推理才能生成有效技能的模型
4. **包装器特异性**：各模型使用专有包装器与匹配的智能体 harness，引入难以量化的方差
5. **缺失跨家族迁移**：提取结果未显示跨任务家族的技能迁移评估
6. **成本波动**：USD成本指标受API定价变化影响，时序可比性受限

**未来方向**：探索多轮技能生成格式、扩展至GUI/Web环境、建立技能库可解释性指标、研究跨家族技能迁移机制、开发针对技能演化的专门模型训练方法。

## 知识图谱定位

**任务节点**：终身技能发现（lifelong skill discovery）、自主智能体评估（autonomous agent evaluation）、技能演化（skill evolution）

**方法节点**：
- **核心**：SkillFlow（新基准+协议）
- **机制组件**：DAEF（领域无关执行流）、Skill Patch Generator（Model_g）、Skill Library Apply函数
- **基线方法**：SkillsBench、SWE-Skills-Bench、SkillCraft、vanilla agent execution

**数据集节点**：SkillFlow benchmark tasks（166任务/20家族/DAEF结构）

**知识贡献结构**：
- **横向连接**：桥接"技能评估"与"终身学习"两个传统分离的研究社区
- **纵向深化**：在自主智能体评估链条中，从静态能力测试延伸至动态适应性测试
- **方法论创新**：提出可形式化的技能演化循环（Model_g + Apply），为后续研究提供可复现的评估框架
- **领域影响**：为LLM智能体的"自我改进"能力提供标准化测量工具，支撑元认知与持续学习研究


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/de0e5660-9182-48c0-81d3-49e95db5e17f/figures/Figure_1.png)
*Figure 1: Figure 1: Conceptual Overview of SKILLFLOW. The figure contrasts conventional static-skillevaluation with our lifelong setting, in which agents externalize experience into reusable skillartifacts, rev*


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/de0e5660-9182-48c0-81d3-49e95db5e17f/figures/Figure_2.png)
*Figure 2: Figure 2: Task-Construction Pipeline of SKILLFLOW. Step 1 collects candidate seed tasks and acurated external skill pool. Step 2 uses embedding-based pair matching to attach relevant skills toeach see*


![Figure 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/de0e5660-9182-48c0-81d3-49e95db5e17f/figures/Figure_4.png)
*Figure 4: Figure 3: DAEF correspondence across domains.Distinct tasks can instantiate the same abstractworkflow, enabling cross-domain skill transfer.*


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/de0e5660-9182-48c0-81d3-49e95db5e17f/figures/Figure_2.png)
*Figure 2: Figure 2 summarizes the process used to build SKILLFLOW. We organize it into four steps.*


![Figure 6](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/de0e5660-9182-48c0-81d3-49e95db5e17f/figures/Figure_6.png)
*Figure 6: Figure 5:Completion–Cost Pareto Frontier.Each point is one evaluated agent-model settingunder vanilla execution or lifelong skill evolu-tion. Some settings shift toward higher comple-tion with compara*


