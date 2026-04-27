---
title: "Uncovering Untapped Potential in Sample-Efficient World Model Agents"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/model-based-reinforcement-learning
  - task/sample-efficient-reinforcement-learning
  - intrinsic-motivation
  - prioritized-replay
  - regression-as-classification
  - dataset/Atari-100K
  - dataset/DeepMind-Control-Suite
  - dataset/Craftax-1M
  - opensource/full
core_operator: 以统一多模态 token 接口承载世界模型学习，并用不确定性驱动探索、损失优先回放和回归分类化共同提升少样本想象控制。
primary_logic: |
  多模态观测/动作轨迹 → 模块化离散 token 化与并行下一观测预测 → 用 JSD 分歧产生内在奖励并按世界模型损失优先回放难样本 → 在想象中训练 actor-critic 并提升无规划少样本 RL 性能
claims:
  - "Claim 1: Simulus 在 Atari-100K 上取得 0.990 的 IQM human-normalized score，并成为首个在人类水平 IQM 与 median 上同时达标的 planning-free world model [evidence: comparison]"
  - "Claim 2: Simulus 在 DeepMind Control Proprioception 500K 上达到 796 平均回报，高于 DreamerV3 的 754；在 Craftax-1M 上达到 6.59% 的最大分数占比，高于 TWM 的 5.44% [evidence: comparison]"
  - "Claim 3: 在 Atari 与 DMC 的消融中，移除内在奖励、优先回放或回归分类化都会降低表现，其中内在奖励影响最大，且三者联用最强 [evidence: ablation]"
related_work_position:
  extends: "REM (Cohen et al. 2024)"
  competes_with: "DreamerV3 (Hafner et al. 2023); TWM (Robine et al. 2023)"
  complementary_to: "TD-MPC2 (Hansen et al. 2024); EfficientZeroV2 (Wang et al. 2024)"
evidence_strength: strong
pdf_ref: paperPDFs/World_Models_in_the_context_of_Model_Based_RL/arXiv_2025/2025_Uncovering_Untapped_Potential_in_Sample_Efficient_World_Model_Agents.pdf
category: Embodied_AI
---

# Uncovering Untapped Potential in Sample-Efficient World Model Agents

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2502.11537), [Code](https://github.com/leor-c/Simulus)
> - **Summary**: 这篇工作把多模态 token 化、基于世界模型不确定性的内在探索、优先回放和回归分类化整合进一个模块化 token-based world model，使 planning-free agent 在极少交互下也能跨视觉、连续控制和符号环境取得更强样本效率。
> - **Key Performance**: Atari-100K 上 IQM HNS = 0.990；DMC Proprioception 500K 上平均回报 796（DreamerV3 为 754）

> [!info] **Agent Summary**
> - **task_path**: 多模态观测（图像/连续向量/2D 符号网格）+ 动作历史 -> token 世界模型想象 -> 策略/价值输出
> - **bottleneck**: 少量真实交互下，TBWM 的核心瓶颈不是单纯模型容量，而是世界模型训练数据覆盖不足、连续/多模态接口不统一，以及奖励/价值目标不稳定
> - **mechanism_delta**: 在 REM 式 TBWM 上加入统一多模态 tokenization、JSD 分歧驱动的内在奖励、按世界模型损失的优先回放，以及 RaC 奖励/价值头
> - **evidence_signal**: 三个异质基准上均达到 planning-free SOTA，且 Atari/DMC 消融显示各组件单独有效并存在协同
> - **reusable_ops**: [modality-wise tokenization to fixed token sequences, mixed uniform+loss-prioritized world-model replay]
> - **failure_modes**: [continuous-vector quantization inflates sequence length and slows training, intrinsic exploration can drift toward task-irrelevant uncertain regions]
> - **open_questions**: [can continuous inputs be tokenized more compactly without hurting online RL, do the gains persist on richer multimodal online RL benchmarks]

## Part I：问题与挑战

这篇 paper 真正要解的，不是“再做一个 world model”，而是把 **token-based world model (TBWM)** 从 Atari 式的“图像 + 离散动作”舒适区，推进到 **在线、少样本、跨模态 RL** 的更一般场景。

**真正瓶颈**有三层：

1. **输入接口瓶颈**  
   现有 TBWM 大多只支持视觉观测和离散动作；一旦进入连续控制或混合模态环境，表示层就先卡住了。

2. **数据分布瓶颈**  
   在 100K / 500K / 1M 这种极小交互预算下，controller 的上限其实由 world model 决定。  
   如果收集到的数据没有覆盖“模型最不确定、最欠拟合”的区域，后续想象训练再强也只是建立在坏模拟器上。

3. **监督稳定性瓶颈**  
   奖励与 return 在不同 benchmark 上尺度差异很大，直接回归容易让 imagined actor-critic 训练变脆。

**输入/输出接口**可以概括为：

- **输入**：图像、连续向量、categorical、2D categorical grid，以及动作历史
- **中间表示**：统一为固定长度 token 序列
- **输出**：下一步观测 token、奖励、终止信号，以及基于 imagined rollouts 学出的策略/价值

**为什么现在值得做**：  
一方面，大规模 world model/视频模型已经说明“多 token 表示”有扩展潜力；另一方面，在线 sample-efficient RL 仍然缺一个真正能在小数据条件下落地的、模块化的 TBWM 配方。

**边界条件**：

- 本文关注的是 **planning-free** world model agents
- 学习范式是 **online RL with fixed interaction budget**
- 需要为每种 modality 提供可用的 encoder/decoder 或 token 接口
- 不与 planning-based 方法正面对比，因为作者将 planning 视为正交增强项

## Part II：方法与洞察

Simulus 可以看成：**REM 的模块化 TBWM 骨架 + 四个关键增强件**。

### 机制拆解

**1. 多模态表示模块 V：先把异构输入变成统一 token API**  
- 图像：VQ-VAE  
- 连续向量：逐特征量化，且把 vocabulary 从大词表压到更适合在线 RL 的小词表（如 125）  
- categorical / 2D categorical：直接离散化并拼接成统一序列  

这一步的意义不只是“能处理更多输入”，而是让后续 world model 和 controller 都只面对 **同一种离散序列接口**。

**2. 世界模型 M：用 RetNet + POP 在 token 空间预测未来**  
- 用 RetNet 建模时序
- 用 POP 并行预测下一观测 token，降低逐 token 生成的顺序瓶颈
- 用 modality-specific heads 分别预测不同模态的 token、奖励和终止

**3. 用 epistemic uncertainty 驱动探索**  
- 给下一观测预测加一个小型 ensemble
- 用 ensemble 分歧的 JSD 作为不确定性估计
- 在 imagination 中把它作为 intrinsic reward 加到外部奖励上

核心点在于：controller 被鼓励去“世界模型最没把握”的地方，但这个搜索过程是在 imagination 中完成的，因此不需要额外增加真实环境试错成本。

**4. 用 prioritized replay + RaC 提升学习效率与稳定性**  
- replay buffer 记录样本的世界模型损失
- 训练世界模型时，混合均匀采样与按损失优先采样
- 奖励/价值预测不用直接数值回归，而改为 regression-as-classification

这两步分别对应：
- **把训练算力集中到 hard examples**
- **把高方差回报目标变得更稳**

### 核心直觉

一句话说，Simulus 改的不是某个大 backbone，而是 **有限数据应该被分配到哪里、以及监督信号应该如何稳定传递**。

#### 因果链条

1. **统一多模态 token 接口**
   - **改变了什么**：把图像、连续状态、符号网格都映射到同一种离散序列接口
   - **改变了哪个瓶颈**：解除“TBWM 只能服务单模态 Atari”的输入约束
   - **带来什么能力**：同一套 world model/controller 设计能扩展到 DMC 和 Craftax

2. **内在奖励 + 优先回放**
   - **改变了什么**：训练时看到的数据分布，从“平均采样”变成“更关注高不确定/高损失区域”
   - **改变了哪个瓶颈**：缓解世界模型的数据覆盖不足与难样本稀释
   - **带来什么能力**：在同样交互预算下，更快修复 world model 盲区，从而让 imagination 更可信

3. **RaC 奖励/价值预测**
   - **改变了什么**：把不稳定的数值回归改成更平滑的分类式监督
   - **改变了哪个瓶颈**：降低回报尺度变化、稀疏奖励对 critic/reward head 的冲击
   - **带来什么能力**：imagined actor-critic 更稳，尤其是在跨 benchmark 的不同 reward regime 下

**为什么这个设计有效**：  
因为 world model agent 的真正上限在于“模拟器质量”，而不是 policy optimizer 本身。Simulus 的几个改动都在围绕这个上限做文章：

- 多模态 tokenization：解决“能不能建模”
- uncertainty bonus：解决“该去哪里补数据”
- prioritized replay：解决“哪些样本该多学几次”
- RaC：解决“学到的奖励/价值信号能不能稳定传下去”

这也是它相对“单独加一个探索 bonus”更强的原因：作者的消融显示，这几项不是平行堆料，而是存在协同。

### 战略权衡

| 设计选择 | 放松的瓶颈 | 收益 | 代价/风险 |
|---|---|---|---|
| 模态分离 tokenization | 输入接口不统一 | TBWM 可跨视觉/连续/符号环境 | 连续向量会导致序列变长 |
| JSD 内在奖励 | 探索不到模型盲区 | 更快补齐 world model 覆盖 | 可能去到任务无关的高不确定区域 |
| 损失优先回放 | hard samples 被均匀采样稀释 | 更快修复模型弱点 | 仍需设置混合比例与初始优先级 |
| RaC 奖励/价值头 | 回报尺度大时回归不稳 | imagined training 更稳定 | 引入 binning 与平滑超参 |
| 模块化分离优化 | 表征/动态/控制目标互扰 | 更易扩展、调试和复用 | 整体训练链更长、更慢 |

## Part III：证据与局限

### 关键实验信号

**信号 1：跨基准比较，说明它不是只在 Atari 上成立**  
- 在 **Atari-100K** 上，Simulus 拿到 **IQM 0.990**，并首次让 planning-free world model 同时达到 human-level 的 IQM 和 median。  
- 这说明它的提升不是某几个游戏的偶然峰值，而是整体分布层面的提升。

**信号 2：连续控制结果，证明 TBWM 不再局限于离散视觉域**  
- 在 **DMC Proprioception 500K** 上，Simulus 平均回报 **796**，高于 DreamerV3 的 **754**。  
- 这直接支持作者的核心主张：token-based world model 也可以在连续观测/动作环境里工作得很好。

**信号 3：Craftax 结果，证明多模态长序列接口是可用的**  
- 在 **Craftax-1M** 上，Simulus 达到 **6.59%** 的最大分数占比，优于 TWM 的 **5.44%** 和多个探索型 model-free baseline。  
- 这里更关键的不是绝对分数，而是它说明：当单步观测已经是很长的多模态 token 序列时，系统仍能保持样本效率。

**信号 4：消融给出因果支持，而不只是“集成更大所以更强”**  
- 移除 intrinsic rewards、prioritized replay、RaC 任一组件都会掉点
- 其中 intrinsic rewards 最关键
- 在 Atari 上，三者联合明显强于任一删减版本，说明存在正反馈协同，而不是简单相加

### 局限性

- **Fails when**: 连续输入维度较高、逐特征量化导致 token 序列过长时，训练会显著变慢；如果高不确定区域与任务回报弱相关，内在探索可能浪费有限交互预算。
- **Assumes**: 需要可用的模态专属 encoder/decoder；需要按 benchmark 调整内外奖励权重与回放超参；需要不低的算力预算（文中报告 Atari 约 12h/RTX4090 或 29h/V100，DMC 约 40h/V100，Craftax 约 94h）。
- **Not designed for**: planning-based search、超大规模真实多传感器系统、以及对“离散 token 化一定优于连续 latent”给出理论证明。

补充看，**证据边界**也要谨慎：
- DMC 上公开可比的 planning-free baseline 本来就少，主要对手是 DreamerV3
- Craftax 与更丰富多模态 RL benchmark 的覆盖仍有限
- 作者在附录里关于“模块化可减少目标干扰”的实验更多是提示性证据，不应当比主实验更强地外推

### 可复用组件

这篇 paper 最值得迁移的，不一定是整套 Simulus，而是下面几个操作件：

1. **统一的 modality-wise token interface**：适合把异构 RL 输入整理成统一 world model API  
2. **JSD ensemble uncertainty bonus**：适合任何想把“模型不知道什么”显式注入 exploration 的 world model agent  
3. **loss-based prioritized world-model replay**：适合在线 RL 下把训练预算集中给 hardest transitions  
4. **RaC reward/value heads**：适合 reward scale 跨环境差异较大的 imagined RL 系统

## Local PDF reference

![[paperPDFs/World_Models_in_the_context_of_Model_Based_RL/arXiv_2025/2025_Uncovering_Untapped_Potential_in_Sample_Efficient_World_Model_Agents.pdf]]