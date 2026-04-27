---
title: "Implementing Spiking World Model with Multi-Compartment Neurons for Model-based Reinforcement Learning"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/model-based-reinforcement-learning
  - task/continuous-control
  - multi-compartment-neuron
  - dendritic-gating
  - world-model
  - "dataset/DeepMind Control Suite"
  - dataset/SHD
  - dataset/TIMIT
  - "dataset/LibriSpeech 100h"
  - opensource/no
core_operator: 用带顶树突门控与基树突积分的多隔室脉冲神经元替换世界模型中的循环动态单元，增强长时序记忆与潜状态预测
primary_logic: |
  视觉观测、上一时刻动作与隐状态脉冲 → 顶/基树突分路积累时序信息，并由顶树突非线性门控体细胞积分 → 生成脉冲隐状态与潜变量，用于观测/奖励/继续信号预测和策略控制
claims:
  - "在 DeepMind Control Suite 的 19 个视觉连续控制任务上，Spiking-WM 在 1M frames 时取得 662.4 平均分，达到 Dreamer(GRU) 的 90.4%，并超过同等参数预算下的 TC-LIF 版本 608.7 [evidence: comparison]"
  - "在 SHD、TIMIT 和 LibriSpeech 100h 上，所提多隔室神经元分别达到 89.57%、83.01% 和 88.77%，均优于文中对比的其他 SNN 神经元模型，并逼近 GRU [evidence: comparison]"
  - "将膜时间常数设为可学习并不能稳定提升控制表现，而基底电导与顶树突门控参数的联动会显著影响性能，说明增益主要来自树突协同积分而非单纯时间常数调参 [evidence: analysis]"
related_work_position:
  extends: "Dreamer (Hafner et al. 2023)"
  competes_with: "Dreamer (Hafner et al. 2023); TC-LIF (Zhang et al. 2024)"
  complementary_to: "PLIF (Fang et al. 2021); STSC-SNN (Yu et al. 2022)"
evidence_strength: strong
pdf_ref: paperPDFs/World_Models_in_the_context_of_Model_Based_RL/arXiv_2025/2025_Implementing_Spiking_World_Model_with_Multi_Compartment_Neurons_for_Model_based_Reinforcement_Learning.pdf
category: Embodied_AI
---

# Implementing Spiking World Model with Multi-Compartment Neurons for Model-based Reinforcement Learning

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2503.00713)
> - **Summary**: 论文把 Dreamer 式世界模型中的时序状态更新单元改成带顶/基树突协同计算的多隔室脉冲神经元，使 SNN 在 model-based RL 中首次表现出接近 GRU world model 的控制能力。
> - **Key Performance**: DMC 19 任务平均分 662.4，达到 Dreamer(GRU) 的 90.4%；TIMIT 83.01%，优于全部 SNN 对比神经元。

> [!info] **Agent Summary**
> - **task_path**: 视觉观测 + 历史动作/隐状态 -> 潜在动力学预测 -> 连续控制动作
> - **bottleneck**: SNN 缺少能在世界模型中稳定保留长时上下文的状态更新机制，导致 imagined rollout 不准
> - **mechanism_delta**: 用“顶树突门控 + 基树突内容积分 + 体细胞放电”的 MCN 取代点神经元/普通循环单元
> - **evidence_signal**: 同参预算下 19 个 DMC 任务均值显著高于其他 SNN world model，且 3 个长序列语音基准持续领先 SNN 基线
> - **reusable_ops**: [顶树突sigmoid门控, 多隔室脉冲状态更新]
> - **failure_modes**: [高速动力学 locomotion 任务仍落后 Dreamer-GRU, 对树突参数 gB/β 有一定敏感性]
> - **open_questions**: [是否能在神经形态硬件上兑现能效优势, 在真实机器人和部分可观测环境中是否仍稳定]

## Part I：问题与挑战

这篇论文真正要解决的，不是“把 SNN 塞进 RL 框架里”这么宽泛的问题，而是更具体的一点：

**SNN 能不能承担 world model 里最关键的“时序记忆核心”？**

在 model-based RL 里，样本效率往往取决于世界模型是否能把长时间的“观测-动作-结果”历史压成一个**可预测、可想象、可用于规划**的潜状态。Dreamer 之类方法之所以强，关键不只是 actor-critic，而是 latent dynamics 学得足够稳。

而现有 SNN 在这里有两个硬瓶颈：

1. **记忆瓶颈**：常见 LIF/点神经元主要靠单一体细胞膜电位做时间积累，长时依赖容易衰减，也容易把不同来源的信息混在一起。
2. **训练瓶颈**：RL 数据分布是非平稳的，世界模型又要同时预测状态、奖励、continue 信号；如果时序积分不够稳，imagined rollout 很快就漂。

所以作者把焦点放在一个非常具体的改动上：  
**不是先改 policy，而是先改“世界模型里的神经元”。**

### 输入/输出接口与边界

- **输入**：视觉观测、上一时刻动作、上一时刻隐状态脉冲
- **中间表示**：脉冲编码后的观测 + MCN 维护的时序隐状态 + Dreamer 风格潜变量
- **输出**：下一状态表征、观测重建、奖励预测、continue 预测，以及连续控制动作

边界条件也很明确：

- 主要验证场景是 **DeepMind Control Suite 的视觉连续控制**
- 语音数据集（SHD/TIMIT/LibriSpeech 100h）主要用于验证**长序列记忆能力**
- **没有**涉及真实机器人部署、神经形态芯片运行或能耗测试

一句话说，作者在回答的是：

> 为什么现在值得做？  
> 因为 world model 已经证明了“长时潜状态”是 RL 的核心，而多隔室神经元正好提供了一个新的、可能更适合 SNN 的记忆单元。

## Part II：方法与洞察

作者整体上**保留了 Dreamer 的系统骨架**，但把最关键的 latent dynamics 从 GRU/普通循环单元改成了 **MCN（multi-compartment neuron）**。

### 方法主线

整个 Spiking-WM 可以概括成 4 步：

1. **Spiking Encoder**  
   用脉冲卷积网络把图像观测编码成 spike 序列。

2. **MCN 时序状态更新**  
   - 基树突和顶树突分别接收外部输入与递归隐状态
   - 两条支路各自积累本地膜电位
   - 顶树突膜电位经过 sigmoid 变成门控信号
   - 该门控决定基树突内容对体细胞积分的影响强度

3. **Dreamer 式 latent state learning**  
   利用 MCN 的脉冲隐状态去参数化 posterior / prior，再做观测、奖励、continue 的预测。

4. **全脉冲 actor-critic**  
   actor 和 critic 也都建立在 SNN 上，使用 world model 的状态表征做控制。

换句话说，它不是“用 SNN 做个 feature extractor”，而是把 **encoder / latent dynamics / decoder / actor / critic** 都尽量脉冲化了，其中真正决定成败的是 MCN 这一步。

### 核心直觉

作者最关键的因果旋钮是：

**把“单通道膜电位记忆”改成“基树突存内容、顶树突给门控、体细胞做选择性写入”的三部分协同记忆。**

这带来的是一个很明确的变化链条：

- **What changed**：  
  点神经元只有一个体细胞膜电位；MCN 则把信息拆成顶树突、基树突、体细胞三部分处理。

- **Which bottleneck changed**：  
  历史信息不再是“谁来都往一个衰减槽里堆”，而是变成：
  - 基树突：保留候选时序内容
  - 顶树突：提供上下文依赖的调制信号
  - 体细胞：只在合适上下文下才放大并输出 spike

- **What capability changed**：  
  世界模型更容易保住长时上下文，减少无关历史的干扰，从而提升 latent rollout 的稳定性与控制性能。

这套设计为什么有效，不是因为“更生物”，而是因为它在功能上更像一个**内容-上下文分离的门控记忆单元**：

- 基树突像“内容缓存”
- 顶树突像“是否允许写入/放大”的上下文门
- 体细胞只在门打开时才形成强响应甚至 burst firing

这比单纯拉长时间常数更有意义，因为它改变的不是“记忆多久”，而是**哪些记忆能在什么时候被写进去并影响状态转移**。

### 战略取舍

| 设计选择 | 解决的瓶颈 | 带来的收益 | 代价/风险 |
| --- | --- | --- | --- |
| 用 MCN 替代 LIF/普通循环状态单元 | 单通道时序记忆不足 | 更强长依赖建模，world model 预测更稳 | 神经元结构更复杂，参数更多 |
| 顶树突 sigmoid 门控 | 无关历史直接污染体细胞状态 | 实现上下文选择性更新，减少时序干扰 | 对门控参数 β 与电导比更敏感 |
| 全脉冲 encoder-decoder-actor-critic | 表示空间不统一 | 获得完整 SNN world model 路径 | 训练仍依赖 STBP/替代梯度，工程难度高 |
| 保留 Dreamer 式 latent imagination | 环境交互成本高 | 继承 model-based RL 的样本效率优势 | 仍会受到 model bias 限制 |

## Part III：证据与局限

### 关键证据信号

**1. 比较信号：DMC 19 个视觉连续控制任务**  
在约相同参数预算下，Spiking-WM 的平均分为 **662.4**，明显高于其他 SNN world model 版本；其中最强 SNN 对照 TC-LIF 为 **608.7**，而 Dreamer(GRU) 为 **709.2**。  
这说明它的能力跃迁不是“终于能训练起来”，而是已经把 SNN world model 推到了**接近 ANN-RNN world model** 的区间。

更强的一点是：它在 **4/19 个任务**上还超过了 Dreamer(GRU)。这说明 MCN 不是纯粹补短板，有些任务上确实能带来更优的时序归纳偏置。

**2. 跨域信号：长序列语音基准**  
在 SHD、TIMIT、LibriSpeech 100h 上，MCN 都是文中 SNN 神经元里最强或并列最强，并持续逼近 GRU。  
这很重要，因为它说明收益不只是“控制头调得好”，而是**序列建模单元本身**更强。

**3. 机制信号：参数与神经活动分析**  
论文做了两个有价值的机制检查：

- **可学习时间常数**并不稳定带来提升  
  说明性能增益不是简单来自“更多自由度”。

- **基树突电导与顶树突门控参数存在明显协同区间**  
  再结合神经元活动统计，能看到：当顶树突门控处于打开区间时，基树突波动才真正驱动体细胞放电。  
  这支持作者的核心论点：**关键在树突协同积分，而不是单一衰减参数。**

### 局限性

- **Fails when**: 高速、强接触、对动力学想象精度要求很高的 locomotion 任务；例如 Hopper Hop、Walker Run、Quadruped Run 上仍明显落后于 Dreamer(GRU)。
- **Assumes**: Dreamer 式潜状态世界模型、连续动作控制、STBP/替代梯度训练，以及对树突参数进行任务相关调节；论文虽提到模型属于 BrainCog Embot 体系，但未给出本文方法的直接代码仓库链接，复现仍有工程门槛。
- **Not designed for**: 神经形态硬件上的真实能耗/延迟验证、真实机器人 sim2real、离散动作 Atari、大规模在线规划，或严格意义上的生物局部学习训练规则。

### 可复用组件

这篇论文最值得迁移的，不一定是整套 Spiking-WM，而是下面几个“操作件”：

1. **MCN 状态更新单元**  
   可作为 SNN 版 RSSM / RNN / world model 的替代 cell。

2. **顶树突门控 + 基树突内容通道**  
   本质上是一个适合长序列的“内容-上下文分离”记忆机制。

3. **全脉冲 world model 骨架**  
   对想做 spiking model-based RL 的工作来说，这是一个可直接复用的系统模板。

总体判断：  
这篇工作的价值，不只是把 SNN 接到了 Dreamer 上，而是给出了一个更有说服力的答案——**如果把记忆能力真正放进神经元内部的树突协同计算里，SNN 在 model-based RL 里是可以逼近传统 RNN world model 的。**

## Local PDF reference

![[paperPDFs/World_Models_in_the_context_of_Model_Based_RL/arXiv_2025/2025_Implementing_Spiking_World_Model_with_Multi_Compartment_Neurons_for_Model_based_Reinforcement_Learning.pdf]]