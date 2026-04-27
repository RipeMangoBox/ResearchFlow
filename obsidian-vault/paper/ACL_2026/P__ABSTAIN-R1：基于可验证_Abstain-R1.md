---
title: 'Abstain-R1: Calibrated Abstention and Post-Refusal Clarification via Verifiable RL'
type: paper
paper_level: B
venue: ACL
year: 2026
paper_link: https://arxiv.org/abs/2604.17073
aliases:
- ABSTAIN-R1：基于可验证强化学习的校准弃权与拒绝后澄清方法报告
- Abstain-R1
acceptance: accepted
method: Abstain-R1
modalities:
- Text
---

# Abstain-R1: Calibrated Abstention and Post-Refusal Clarification via Verifiable RL

[Paper](https://arxiv.org/abs/2604.17073)

**Topics**: [[T__Reasoning]], [[T__Text_Generation]], [[T__Reinforcement_Learning]] | **Method**: [[M__Abstain-R1]] | **Datasets**: ABSTAIN-TEST

## 基本信息

**论文标题**: Abstain-R1: Calibrated Abstention and Post-Refusal Clarification via Verifiable RL

**作者**: （未在提供的分析中提取完整作者列表）

**发表 venue/年份**: （未明确提取）

**代码/数据链接**: （未在提供的分析中提取）

**基础模型**: Qwen2.5-3B-Instruct

**核心方法**: 三阶段管道（Abstain-CoT数据构建 → AbstAIN-SFT冷启动 → GRPO强化学习）

## 核心主张

本文核心主张是：**通过澄清感知可验证强化学习（clarification-aware RLVR），可以在仅3B参数的小模型上有效学习校准弃权（calibrated abstention）与语义对齐的拒绝后澄清（post-refusal clarification）**。关键证据：在ABSTAIN-TEST上，ABSTAIN-R1取得U-Ref 68.1%和U-Clar 55.1%，超越DeepSeek-R1（52.2%/46.5%）和DeepSeek-V3（58.4%/50.1%）等更大规模模型；在SELFAWARE上U-Ref达91.4%，领先所有基线。置信度：0.85（实验结果可靠，但存在基线规模不对等和评估指标不完整等公平性问题）。

## 研究动机

当前LLM强化微调提升推理能力的同时，**在不可回答查询上产生幻觉式猜测**；现有弃权方法存在双重缺陷：要么输出**通用拒绝（generic refusal）**缺乏信息量，要么生成**未经核实的澄清（unverified clarifications）**。Know-Guard等近期工作未直接对比。研究空白在于：缺乏将"何时拒绝"与"拒绝后如何提供有用澄清"统一学习的可扩展框架，且现有RLVR奖励设计未针对弃权场景进行条件化扩展。

## 方法流程

三阶段管道流程：

```
Stage 1: Abstain-CoT 数据构建
    ↓ 输出：带is_answerable标签、参考答案、参考澄清c*的结构化数据
Stage 2: Abstain-SFT 冷启动
    ↓ 输入：Qwen2.5-3B-Instruct + Abstain-CoT
    ↓ 输出：具备格式遵循与澄清推理能力的初始化策略
Stage 3: RLVR via GRPO
    ↓ 输入：旧策略π_old采样G个输出
    ↓ 复合奖励计算：格式检查 + 答案验证/弃权检查 + 澄清验证
    ↓ 组相对优势归一化 + KL(π_θ || π_ref)正则
    ↓ 输出：更新后的策略π_θ
```

关键控制流：Answerability Router根据查询可回答性分支至不同奖励计算路径。

## 关键公式

**1. 复合奖励函数（核心创新）**

```latex
r(o, y) = \begin{cases} r_{\text{fmt}} + r_{\text{ans}}, & \text{if } q \in \mathcal{D}_{\text{ans}} \\ r_{\text{fmt}} + r_{\text{ref}}, & \text{if } q \in \mathcal{D}_{\text{unans}} \end{cases}
```

根据查询可回答性条件分支，替代标准RLVR的单一正确性奖励。

**2. 带虚假弃权惩罚的可回答奖励（创新）**

```latex
r_{\text{ans}} = \begin{cases} 1, & \text{匹配真值} \\ -1, & \text{错误输出}\boxed{\text{``I don't know''}} \\ 0, & \text{其他错误} \end{cases}
```

**-1惩罚**显式抑制对可解问题的过度保守，区别于标准{0,1}奖励。

**3. GRPO目标（继承GRPO，非创新）**

```latex
J_{\text{GRPO}}(\theta) = \mathbb{E}_{q,\{o_i\}} \left[ \frac{1}{G} \sum_{i=1}^{G} \min(r_i A_i, \text{clip}(r_i) A_i) - \beta\, \text{KL}(\pi_\theta \| \pi_{\text{ref}}) \right]
```

组内相对优势替代critic网络，降低显存消耗。

## 实验结果

**主实验结果（Table 1）**：

| 基准 | 指标 | ABSTAIN-R1 | 最强基线 | 提升 |
|:---|:---|:---|:---|:---|
| ABSTAIN-TEST | U-Ref | **68.1** | DeepSeek-V3 58.4 | +9.7 |
| ABSTAIN-TEST | U-Clar | **55.1** | DeepSeek-V3 50.1 | +5.0 |
| ABSTAIN-TEST | U-Clarc | 80.9 | DeepSeek-R1 89.1 | -8.2 (未SOTA) |
| ABSTAIN-TEST | A-Acc | 57.2 | DeepSeek-R1 78.6 | -21.4 (落后) |
| ABSTAIN-TEST | A-FU | 20.4 | DeepSeek-R1 8.5 | +11.9 (更差) |
| ABSTAIN-QA | U-Ref | **40.1** | Qwen2.5 7B 35.5 | +4.6 |
| SELFAWARE | U-Ref | **91.4** | Qwen2.5 3B 82.3 | +9.1 |

**消融发现**：移除澄清奖励(r_ref)导致U-Clar下降；移除-1虚假弃权惩罚导致A-FU上升（过度保守）；移除SFT导致RL训练不稳定。

**证据强度评估**：0.7——弃权指标提升显著，但A-Acc和A-FU弱于大模型，且基线规模不对等影响公平性。

## 相关工作

**按角色分类**：

| 角色 | 方法 | 关系 |
|:---|:---|:---|
| **直接基线** | DeepSeek-R1 | 主要对比对象，ABSTAIN-R1在其上扩展弃权能力 |
| **直接基线** | Qwen2.5-3B/7B/32B-Instruct | 初始化基座与规模对比基线 |
| **组件来源** | GRPO (DeepSeekMath) | 策略优化算法，消除价值模型 |
| **间接基线** | DeepSeek-V3, Llama3.1 8B | 额外规模对比 |
| **未充分对比** | Know-Guard [7] | 同期弃权工作，作者未直接比较 |

**关键关系**：ABSTAIN-R1与DeepSeek-R1形成"继承-改造"关系——继承GRPO训练范式，但彻底重构奖励设计与数据管道。GRPO作为方法组件提供无critic的优化框架，但复合奖励的条件分支设计为本文独有。

## 方法谱系

**父方法**: DeepSeek-R1

**继承 slots**:
- GRPO无critic策略优化框架
- RLVR训练范式（强化学习+可验证奖励）
- 推理时CoT生成模式

**改造 slots**:
| Slot | 父方法值 | ABSTAIN-R1值 | 变更类型 |
|:---|:---|:---|:---|
| `reward_design` | 标准结果奖励 {0,1} | **四组件复合奖励**（格式+答案+弃权+澄清） | 替换 |
| `data_pipeline` | 标准QA数据 | **Abstain-CoT**（显式可回答标签+参考澄清） | 替换 |
| `training_recipe` | SFT→RL通用流程 | **三阶段管道**（数据构建→SFT冷启动→RLVR） | 修改 |
| `inference_strategy` | 直接CoT生成 | **条件结构化输出**（<thinking>/<answer> + \boxed{}） | 修改 |

**谱系定位**: ABSTAIN-R1属于"R1推理能力→特定能力增强"分支，是首个将R1范式系统扩展至弃权+澄清场景的专用模型。

## 局限与展望

**论文明确局限**：
- SELFAWARE仅报告U-Ref，缺乏澄清指标（数据集本身限制）
- 未与Know-Guard [7]直接对比

**分析推断局限**：
- **规模不对等**：3B基座 vs 7B-32B及DeepSeek-V3/R1对比，A-Acc差距大（57.2 vs 78.6）可能源于容量限制而非方法缺陷
- **校准问题**：A-FU 20.4%高于所有更大基线，存在过度拒绝倾向
- **评估偏差**：LLM-as-judge（xVerify-3B-Ia/o4-mini）可能奖励与训练信号相似的澄清
- **未验证声明**：GRPO内存优势未与PPO直接消融

**未来方向**：扩展至7B/32B验证规模效应；探索DPO替代方案；开发更鲁棒的澄清质量自动评估；将框架迁移至代码生成、医疗问答等高风险领域。

## 知识图谱定位

**任务节点**：
- `abstention detection`（弃权检测）— 核心任务，U-Ref指标直接衡量
- `question answering`（问答）— 承载场景，解决不可回答查询的幻觉问题
- `post-refusal clarification generation`（拒绝后澄清生成）— 差异化任务，U-Clar/U-Clarc衡量

**方法节点**：
- `ABSTAIN-R1` — 新节点，3B专用弃权模型
- `GRPO` — 继承节点，提供优化基础设施
- `DeepSeek-R1` — 父节点，提供能力基线与训练范式

**数据集节点**：
- `Abstain-CoT`（训练数据）、`ABSTAIN-TEST`/`ABSTAIN-QA`/`SELFAWARE`（评估基准）

**机制节点**：
- `composite clarification-aware reward` — 核心贡献，连接"可验证奖励"与"弃权学习"
- `LLM-as-judge` — 澄清质量验证的关键使能技术

**结构贡献**：本文填补了"推理能力增强→可靠弃权"的知识缺口，建立了从R1通用推理到特定可信AI能力的扩展路径，为后续"能力专用化RLVR"研究提供模板。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/83c8c018-1516-463c-9a90-37d712c8d7b7/figures/Figure_1.png)
*Figure 1: Figure 1: U-Clar (left) and U-Ref (right) on ABSTAIN-TEST across model sizes, showing that explicit absten-tion training is more effective than scaling alone.*


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/83c8c018-1516-463c-9a90-37d712c8d7b7/figures/Figure_2.png)
*Figure 2: Figure 2: Comparison of model behaviors on an unanswerable query caused by a missing definition of the variabley. From left to right, we illustrate: answering without abstention, which results in hall*


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/83c8c018-1516-463c-9a90-37d712c8d7b7/figures/Figure_3.png)
*Figure 3: Figure 3: Overview of the proposed RLVR training pipeline via GRPO. The framework consists of three stages:(1) constructing training data with explicit answerability labels and reference clarification*


![Figure 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/83c8c018-1516-463c-9a90-37d712c8d7b7/figures/Figure_4.png)
*Figure 4: Figure 4: Mean response length (in tokens) across train-ing steps.*


