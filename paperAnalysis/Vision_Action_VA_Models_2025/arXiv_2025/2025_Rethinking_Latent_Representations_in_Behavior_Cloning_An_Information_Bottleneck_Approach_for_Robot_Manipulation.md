---
title: "Rethinking Latent Redundancy in Behavior Cloning: An Information Bottleneck Approach for Robot Manipulation"
venue: ICML
year: 2025
tags:
  - Embodied_AI
  - task/robot-manipulation
  - task/imitation-learning
  - information-bottleneck
  - mutual-information-estimation
  - dataset/CortexBench
  - dataset/LIBERO
  - opensource/no
core_operator: 在行为克隆的特征融合层对输入表征 X 与潜变量 Z 施加信息瓶颈，用互信息压缩冗余同时保留动作预测相关信息。
primary_logic: |
  多模态观测先编码并拼接为输入表征 X → 通过融合模块得到潜变量 Z，并用 MINE 约束 I(X,Z) 以压缩冗余信息 → 在维持动作监督的同时学习更紧凑、更可泛化的表示并输出机器人动作
claims:
  - "在 CortexBench 与 LIBERO 上，为各类 BC 基线加入 IB 后均获得一致性能增益，例如 ResNet 在 CortexBench 平均成功率由 73.40 提升到 78.11，BC-VILT 在 LIBERO 平均成功率由 48.21 提升到 53.79 [evidence: comparison]"
  - "在 LIBERO-Goal 上，BC+IB 将估计的 I(X,Z) 降至 vanilla BC 的约 1/4，并带来约 7.7 个百分点的成功率提升；β 过大时收益回落，说明压缩强度需要折中 [evidence: ablation]"
  - "论文给出一般化误差与 I(X;Z) 正相关的理论界，并证明在中间特征层 X 上施加瓶颈与直接在原始多模态输入 O 上施加瓶颈之间的优化差距可被有界 [evidence: theoretical]"
related_work_position:
  extends: "Information Bottleneck (Tishby et al. 1999)"
  competes_with: "VC-1 (Majumdar et al. 2023); BC-VILT (Liu et al. 2024)"
  complementary_to: "Diffusion Policy (Chi et al. 2023); OpenVLA (Kim et al. 2024)"
evidence_strength: strong
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Rethinking_Latent_Representations_in_Behavior_Cloning_An_Information_Bottleneck_Approach_for_Robot_Manipulation.pdf
category: Embodied_AI
---

# Rethinking Latent Redundancy in Behavior Cloning: An Information Bottleneck Approach for Robot Manipulation

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2502.02853)
> - **Summary**: 这篇工作把信息瓶颈直接接到行为克隆的多模态融合表征上，核心结论是：机器人 BC 的一个被忽视瓶颈不是“信息不够”，而是“潜变量太冗余”，适度压缩后泛化会更好。
> - **Key Performance**: CortexBench 上 ResNet 平均成功率 73.40 → 78.11；LIBERO 上 BC-VILT 平均成功率 48.21 → 53.79。

> [!info] **Agent Summary**
> - **task_path**: 多模态观测（图像/本体状态/语言）→ 行为克隆策略 → 连续机器人动作
> - **bottleneck**: 融合后的潜表示 Z 含有大量与动作无关的背景、跨模态重复和历史冗余，导致 BC 记住数据细节而非任务充分统计量
> - **mechanism_delta**: 在 `X→Z` 的融合通道加入基于 MINE 的互信息压缩项，让模型只能保留对动作预测真正有用的信息
> - **evidence_signal**: 跨 CortexBench、LIBERO 和真实机器人实验的稳定增益，同时伴随估计互信息 `I(X,Z)` 明显下降
> - **reusable_ops**: [feature-level information bottleneck, MINE-based redundancy estimation]
> - **failure_modes**: [过大β导致过压缩并破坏空间结构信息, 低冗余场景或轻量模型容量受限时收益有限]
> - **open_questions**: [能否稳定扩展到大规模VLA/动作token模型, β与互信息估计器能否按任务自适应选择]

## Part I：问题与挑战

这篇论文真正抓住的问题是：**当前机器人行为克隆并不一定缺信息，反而可能因为多模态输入、长历史和大预训练编码器带来过多“无关但可记忆”的信息**。

### 1）真瓶颈是什么
传统 BC 的优化目标只要求“把动作拟合对”，但**没有约束 latent representation 到底该保留什么**。于是模型很容易把以下内容一起编码进潜变量：

- 与动作无关的背景纹理、视角细节
- 跨模态重复信息
- 对当前动作帮助不大的历史片段
- 数据集特有的偶然相关性

这会带来一个典型问题：**训练误差低，但泛化差**。尤其在机器人操作中，测试时经常会遇到新物体组合、位置扰动、未见实例，冗余表征会让策略更脆弱。

### 2）为什么现在值得解决
因为现在很多 BC 工作的主线是：

- 用更大数据
- 加更多模态
- 上更强 backbone
- 拉长历史窗口

这些做法确实能提升上限，但也会同步扩大冗余。论文的判断是：**“加信息”不是免费的，如果不控制表示容量，额外信息会转化成过拟合通道。**

### 3）输入/输出接口与边界
论文研究的接口很标准：

- **输入**：视觉观测 `o`、本体状态 `s`、可选语言指令 `l`
- **输出**：连续动作 `a`

它聚焦于**行为克隆**而不是 RL；主要实验也以 **vanilla BC + MLP policy head** 为主，目的是把“冗余压缩”的作用单独看清楚。  
因此它回答的是：**在不改任务定义的前提下，怎么让 BC 的中间表示更像任务充分统计量，而不是数据记忆缓存。**

## Part II：方法与洞察

### 方法骨架

论文把机器人 BC 统一写成：

`多模态编码器 → 拼接成输入表征 X → 融合模块 F 得到潜变量 Z → policy head 输出动作`

关键变化不在 encoder，也不在 action head，而在 **`X→Z` 这段信息通道**。

#### 具体做法
1. **先把各模态编码后再拼接成 X**
   - 图像、本体状态、语言分别编码
   - 不在原始模态上逐个加 IB，而是在融合前的中间特征层统一处理

2. **在 X 到 Z 上加入信息瓶颈**
   - 训练目标从“只拟合动作”改为“动作拟合 + 压缩 `I(X,Z)`”
   - 互信息由 MINE 估计

3. **同时覆盖两类 BC 架构**
   - **空间融合**：MLP/CNN 等，偏单时刻或短历史聚合
   - **时间融合**：RNN/Temporal Transformer，显式建模时序依赖

4. **理论上把泛化误差与 `I(X,Z)` 连起来**
   - 论文适配已有 IB 泛化界，说明 `I(X,Z)` 越大，泛化误差上界越松
   - 还证明了在中间特征 `X` 上做瓶颈，只要 `X` 保留了原始输入结构，本质上仍在控制原输入到潜变量的信息流

### 为什么瓶颈放在特征级 X，而不是原始模态级 O
这是论文最实用的设计点之一。

如果对图像、语言、本体状态分别做 IB：
- 工程复杂
- 难扩展到更多模态
- 很难处理跨模态冗余
- 冻结 encoder 时不够自然

而在拼接后的 `X` 上做：
- 统一压缩跨模态冗余
- 更容易兼容现有 BC pipeline
- 对 frozen encoder 场景更友好
- 更像一个即插即用正则器

### 核心直觉

原来的 BC 只问：**“Z 能不能预测动作？”**  
这篇论文新增的问题是：**“Z 为了预测动作，到底需要从 X 里带走多少信息？”**

因果链条可以概括为：

**只做动作监督**  
→ `Z` 可以顺手记住大量与动作无关但训练集可利用的细节  
→ 表征容量过宽，数据集偶然相关性被保留  
→ 测试时遇到分布变化就脆弱

**加入 `I(X,Z)` 压缩**  
→ `X→Z` 的传输容量被约束  
→ 模型被迫优先保留动作相关、任务充分的信息  
→ 背景噪声、跨模态重复、无用时序细节更难进入 `Z`  
→ 泛化更稳，尤其在多任务、长历史、语言条件场景下更明显

这也是它和“普通正则化”的区别：它不是泛泛地让模型更小，而是**定向限制输入表征进入潜变量的信息量**。

### 战略取舍

| 设计选择 | 改变了什么约束 | 带来的能力变化 | 代价 / 风险 |
|---|---|---|---|
| 在 `X→Z` 上加 IB，而非只用 BC loss | 约束潜变量容量，而非只约束输出误差 | 更少记忆训练集偶然细节，泛化更强 | 需要额外互信息估计器 |
| 在特征级 `X` 上统一压缩，而非逐模态压缩 | 把跨模态冗余也纳入控制 | 更通用、可插拔、兼容 frozen encoder | 依赖 `X` 仍保留原始输入关键信息 |
| 空间融合 | 弱化显式时序建模 | 在简单单任务里收敛更快、效果更好 | 复杂长程依赖任务能力不足 |
| Temporal Transformer 融合 | 强化时序依赖建模 | 在多任务、长历史环境中更强 | 简单任务可能训练更慢、收益不一定更高 |
| 较小到适中的 β | 温和压缩冗余 | 通常能稳定提升性能 | β 过大时会伤害必要信息，尤其空间结构任务 |

## Part III：证据与局限

### 关键证据

**信号 1：跨基线一致提升，而不是只对某个 backbone 有效。**  
在 CortexBench 上，论文测试了 full fine-tuning 与 partial fine-tuning、多种视觉 backbone。结果是所有加入 IB 的版本都优于原版 BC。最有代表性的两组：

- **ResNet**：73.40 → 78.11
- **VC-1**：57.04 → 59.28

这说明它不像是“给某个模型调参调好了”，而更像是一个普适表征约束。

**信号 2：在更复杂的多任务场景，IB 更有价值。**  
LIBERO 上四类策略全部提升，且在 Goal/Object 这类对象与目标多样性更强的任务上更明显：

- **BC-Transformer**：48.37 → 52.59
- **BC-VILT**：48.21 → 53.79
- **BC-MLP**：16.79 → 25.71

这支持论文的核心判断：**输入越复杂、冗余越多，去冗余越有用。**

**信号 3：能力提升与“互信息下降”是联动的。**  
作者不只报告成功率，还直接估计 `I(X,Z)`。结果显示 BC+IB 在 LIBERO 各套件上都呈现：

- 更低的 `I(X,Z)`
- 更高的成功率

尤其在 LIBERO-Goal 中，`I(X,Z)` 大约降到 vanilla BC 的四分之一，同时成功率提升约 7.7 个点。  
这比“只看最终分数”更能支持它的因果解释：**提升来自去冗余，而不只是额外正则噪声。**

**信号 4：few-shot 和真实机器人也有增益。**  
论文还报告了：
- 10-shot 的 LIBERO few-shot 训练中仍有提升
- 真实机器人单任务与语言条件多任务中，大多数任务成功率更高
- 对未见 object-bowl 组合也更稳

这意味着该方法不只是 simulator trick，而是对实际部署也有一定价值。

### 1-2 个最值得记住的指标
- **CortexBench**：ResNet + IB 平均成功率 **73.40 → 78.11**
- **LIBERO**：BC-VILT + IB 平均成功率 **48.21 → 53.79**

### 局限性

- **Fails when:** 输入本身冗余很低时收益有限，例如 TriFinger 这类视觉场景简单、背景干扰少的任务；当 β 过大时，依赖精细空间结构的任务（如 LIBERO-Spatial）可能被过压缩；在 LIBERO-Long 上，轻量 baseline 的模型容量本身就是更大的瓶颈。
- **Assumes:** 需要高质量专家 demonstrations；主要建立在连续动作的 vanilla BC 设定上；需要额外训练一个 MINE 互信息估计器并调节 β；理论上默认中间特征 `X` 保留原始多模态输入的关键结构；真实多任务实验里，CogAct 训练依赖 8×A100，资源门槛不低。
- **Not designed for:** 大规模 VLA / action-token policy 的系统性验证；替代 policy head（如 diffusion 或纯 transformer action head）的全面比较；强 domain shift、跨环境鲁棒性的完整研究；无监督或强化学习场景。

### 可复用组件

1. **Feature-level IB 插件**  
   对现有 BC pipeline 最容易复用的点，不必重写各模态 encoder。

2. **MINE-based 冗余诊断**  
   不仅能训练，还能把 `I(X,Z)` 当成分析工具，判断模型是否在“携带过多无关信息”。

3. **融合策略经验**  
   - 简单单任务：空间融合往往更高效  
   - 复杂多任务/长历史：Temporal Transformer 更合适  
   - IB 可作为这两类融合模块上的统一正则器

4. **实用结论**  
   如果你的机器人策略已经用了更大数据、更多模态、更强 encoder，但泛化仍不稳，那么下一步未必是继续加模型，而可能是先检查 **latent 是否太冗余**。

## Local PDF reference

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Rethinking_Latent_Representations_in_Behavior_Cloning_An_Information_Bottleneck_Approach_for_Robot_Manipulation.pdf]]