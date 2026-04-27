---
title: "Learning Long-Context Diffusion Policies via Past-Token Prediction"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/robot-imitation-learning
  - task/robot-manipulation
  - diffusion
  - past-token-prediction
  - embedding-caching
  - dataset/RoboMimic
  - dataset/Push-T
  - opensource/no
core_operator: 通过联合预测过去与未来动作 token，并将过去动作重建用于缓存式长上下文训练与测试时一致性重排序，强化 diffusion policy 的长程时序依赖。
primary_logic: |
  历史视觉/本体感觉观测序列 → 先以短上下文预训练并冻结视觉编码器、缓存长序列嵌入，再训练解码器联合预测过去与未来动作 token → 输出更稳定的长上下文动作块，并可在推理时按过去动作一致性选择候选
claims:
  - "PTP 使 diffusion policy 的动作可预测性比率接近专家示范，而无 PTP 的长上下文 diffusion policy 在该指标上比专家弱约 10×–100× [evidence: analysis]"
  - "在 6 个仿真任务上，PTP 长上下文 diffusion policy 的平均成功率约为 80.7%，高于无历史 diffusion baseline 的 49.5% 和无 PTP 长上下文 baseline 的 16.5% [evidence: comparison]"
  - "短上下文编码器预训练 + 长上下文缓存嵌入微调可在约 20% 的训练时间内追平未缓存训练性能，论文总体报告训练开销降低超过 10× [evidence: ablation]"
related_work_position:
  extends: "Diffusion Policy (Chi et al. 2023)"
  competes_with: "Diffusion Policy (Chi et al. 2023); Bidirectional Decoding (Liu et al. 2024)"
  complementary_to: "TraceVLA (Zheng et al. 2024); Keyframe-Focused Visual Imitation Learning (Wen et al. 2021)"
evidence_strength: strong
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Learning_Long_Context_Diffusion_Policies_via_Past_Token_Prediction.pdf
category: Embodied_AI
---

# Learning Long-Context Diffusion Policies via Past-Token Prediction

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2505.09561), [Project](https://long-context-dp.github.io/)
> - **Summary**: 这篇工作不是继续“裁掉历史”，而是强迫 diffusion policy 同时解释“过去发生了什么”和“接下来要做什么”，从而把长上下文中真正有用的时序信息保留下来。
> - **Key Performance**: 6 个仿真任务平均成功率约 80.7%（无历史 49.5%，无 PTP 16.5%）；训练开销报告降低超过 10×

> [!info] **Agent Summary**
> - **task_path**: 历史视觉/本体感觉观测序列 + 离线专家示范 -> 当前与未来机器人动作块
> - **bottleneck**: 长上下文 diffusion policy 虽然看到了历史，但没有学到专家级的过去-未来动作依赖；同时端到端长历史视觉训练的显存与算力成本过高
> - **mechanism_delta**: 把 future-only 动作预测改成 joint past-and-future token prediction，并把过去 token 重建同时用于训练正则和测试时候选重排序
> - **evidence_signal**: 6 个仿真 + 4 个真实任务比较、动作依赖诊断、decoder/encoder 归因消融与缓存训练消融共同支持性能和效率提升
> - **reusable_ops**: [past-token auxiliary prediction, frozen-encoder embedding caching]
> - **failure_modes**: [长历史闭环推理仍有延迟压力, 依赖下采样历史与 action chunk 时会牺牲部分反应性]
> - **open_questions**: [是否可迁移到 action-tokenizer 或 flow-matching policy, 如何在不增加推理延迟的前提下做更强的自验证]

## Part I：问题与挑战

这篇论文研究的是**部分可观测机器人模仿学习**：输入是过去若干步的视觉/本体感觉观测，输出是当前与未来一段动作。典型场景包括：

- 单帧看不全状态（遮挡、视角受限）
- 多阶段任务需要知道“现在进行到哪一步”
- 早期动作决定后续策略或执行风格（如放哪边、已经 scoops 几次、物体起始位置在哪）

### 真正的难点是什么？

作者认为难点不是“历史太长不好算”这么简单，而是两个耦合瓶颈：

1. **长历史会引入更多伪相关特征**  
   模型可能抓住和动作共现、但不真正因果相关的历史视觉线索，训练时拟合得不错，部署时却偏离 expert manifold。

2. **长历史的端到端视觉训练太贵**  
   图像序列越长，显存和计算越线性上升，长上下文训练很快变得不可承受。

### 这篇论文最关键的诊断

作者重新审视 imitation learning 里的 copycat 问题后发现：  
**经典 regression policy 往往过度依赖过去动作；但现代 diffusion policy 恰恰相反——它们经常没有学到足够强的过去-未来动作依赖。**

这点非常重要，因为它改变了问题定义：

- 过去很多工作默认“历史可能有害”，所以选择**截断历史、挑 keyframe、做摘要**
- 这篇论文认为：对 diffusion policy 来说，更大的问题是**模型没有被训练去保留真正有用的历史时序结构**

因此，真正瓶颈不是“如何把历史扔掉”，而是：

> **如何让策略在看到长历史时，真的把“过去发生过什么”编码进当前决策。**

---

## Part II：方法与洞察

方法可以概括成一句话：

> **不是只监督“未来动作”，而是让策略同时重建过去动作与预测未来动作。**

### 方法主线

#### 1. PTP：Past-Token Prediction

标准 diffusion policy 通常只预测当前/未来动作 chunk。  
PTP 把训练目标扩展为：给定历史观测，**联合预测过去到未来的一整段动作 token**。

这相当于强迫策略回答两个问题：

- 你接下来要做什么？
- 你能不能根据当前看到的历史，解释已经发生过的动作？

如果第二个问题答不好，说明模型内部并没有真正保留历史中的时序信息。

#### 2. 多阶段训练：把“感知”和“长程时序建模”拆开

作者进一步发现，PTP 的收益主要来自**policy head / decoder**，而不是视觉 encoder。  
于是训练被拆成三步：

1. 用**短上下文**预训练视觉 encoder
2. 冻结 encoder，并为训练集所有帧**缓存 embedding**
3. 用缓存好的长序列 embedding 训练**长上下文 decoder + PTP**

这个设计的关键在于：

- 最贵的图像前向/反向传播不再出现在长历史训练中
- 长程依赖主要交给 decoder 学
- 因为 PTP 的收益主要在 decoder，这样拆开几乎不掉性能

#### 3. 测试时自验证：让过去动作变成 verifier

PTP 还有一个很自然的副产品：  
每个采样候选不仅给出未来动作，也会重建过去动作。

而过去动作在执行时是**已知真值**，所以可以：

- 从同一观测采样多个候选 action chunk
- 比较每个候选对“已执行过去动作”的重建误差
- 选那个**最能解释过去**的候选

这比只检查“和上一次模型预测是否一致”更稳，因为它依赖的是**真实过去动作**，而不是历史上的模型猜测。

### 核心直觉

#### 改变了什么？
从“只预测未来动作”  
→ 变成“同时重建过去动作 + 预测未来动作”。

#### 哪个信息瓶颈被改变了？
future-only 监督下，模型只要生成一个局部看起来合理的未来 chunk 即可，**不必把轨迹阶段、执行风格、初始条件等长程变量保留在隐藏状态里**。  
PTP 则要求隐藏状态必须对“过去为何如此”也有解释力，因此更像一个**关于轨迹历史的充分统计量**。

#### 能力为什么会变强？
当 decoder 必须保留“过去发生过什么”的信息时：

- 过去与未来动作之间的依赖被显式监督
- rollout 的时序统计更接近 expert
- 长程、记忆敏感任务不再只靠当前帧做“短视猜测”

一个非常有力的侧证是：  
**即使把 past actions 直接喂给模型，没有 PTP 也依然不行。**  
说明关键旋钮不是“有没有访问过去”，而是“训练目标有没有要求模型使用过去”。

### 策略性取舍

| 设计选择 | 改变的约束 | 收益 | 代价 |
|---|---|---|---|
| PTP 辅助目标 | 输出从 future-only 变成 past+future | 恢复 expert-like temporal dependency，长程任务成功率显著上升 | 解码目标更长 |
| 冻结 encoder + embedding caching | 把长上下文学习集中到 decoder | 显存/训练时间显著下降，便于扩长历史 | encoder 不能再被长上下文端到端适配 |
| test-time verification | 从单候选变成多候选 + 过去一致性筛选 | 困难闭环任务继续提升 | 推理延迟随 sample budget 增加 |

---

## Part III：证据与局限

### 关键证据

- **诊断信号（analysis）**  
  作者用 action predictability ratio 衡量 rollout 中“当前动作能否由过去动作预测”。结果显示：  
  regression policy 倾向 copycat，而 diffusion policy 反而**对过去动作依赖过弱**。PTP 会把这一依赖拉回到接近 expert 的水平。

- **主比较信号（comparison）**  
  在 6 个仿真任务上，PTP 的长上下文 diffusion policy 平均成功率约 **80.7%**，显著高于：
  - 无历史 diffusion：**49.5%**
  - 长上下文但无 PTP：**16.5%**

  最能说明问题的是 long-horizon square / aloha 这类“必须记得早先信息”的任务：baseline 常常低于 30%，PTP 接近满分。

- **真实世界信号（comparison）**  
  在 4 个真实 history-critical tasks 上，PTP 平均成功率约 **70%**；无历史基线平均只有 **15%**，无 PTP 基线在 4 个任务里有 3 个几乎为零。  
  尤其是 Tape Replacement，PTP 达到 **80%**，两个基线都失败。

- **因果消融（ablation）**  
  decoder-only PTP 基本追平 full PTP，encoder-only PTP 明显差很多。  
  这直接支持了论文的核心判断：**增益主要来自时序建模头，而非视觉特征本身。**

- **效率信号（ablation）**  
  缓存 embedding 后，作者在两天预算实验中约用 **20% 时间**就追平未缓存训练，约 **40% 预算**即可超越；论文总体报告训练开销下降 **10×+**。  
  另外，test-time verification 在困难任务上还能再带来约 **5%** 增益。

### 局限性

- **Fails when**: 需要极低延迟的高频闭环控制、不能接受历史下采样、或无法承担多候选重排的场景；论文也明确承认 inference overhead 仍是实际瓶颈。
- **Assumes**: 有离线专家示范；任务的关键信息可以从观测历史和过去动作中恢复；短上下文预训练 encoder 的表征足以支撑长上下文 decoder；有条件做 embedding caching 和 GPU 并行采样。
- **Not designed for**: 在线 RL / 主动探索、语言条件通用 VLA、超长 streaming memory 管理，或必须依赖长上下文端到端共同更新视觉编码器的设定。

### 复现与可扩展性提醒

- 真实实验依赖 **Franka / ALOHA** 硬件与任务特定 subsampling
- 每个真实任务需要 **50–200** 条示范
- 论文给了 project/videos，但正文未声明代码发布，故可复现性仍受限

### 可复用组件

1. **PTP 辅助目标**：可迁移到其他序列决策模型，核心思想是“让模型解释过去，而不只预测未来”。
2. **冻结 encoder + 缓存 embedding**：适合任何“长历史贵在视觉前端、收益主要在时序头部”的策略。
3. **past-consistency verifier**：适合 diffusion / sampling-based policy 的测试时重排序。

**一句话总结**：  
这篇论文的价值不只是加了一个 auxiliary loss，而是指出了一个更本质的现象：**长上下文 diffusion policy 的问题不是记得太多，而是没有被要求真正记住。**

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Learning_Long_Context_Diffusion_Policies_via_Past_Token_Prediction.pdf]]