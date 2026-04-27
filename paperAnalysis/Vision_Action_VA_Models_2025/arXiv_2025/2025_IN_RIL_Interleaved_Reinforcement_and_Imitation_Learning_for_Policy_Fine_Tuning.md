---
title: "IN-RIL: Interleaved Reinforcement and Imitation Learning for Policy Fine-Tuning"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/robot-manipulation
  - task/locomotion-control
  - reinforcement-learning
  - imitation-learning
  - gradient-surgery
  - dataset/FurnitureBench
  - dataset/Robomimic
  - dataset/D4RL
  - opensource/full
core_operator: 在RL微调过程中周期性插入IL更新，并用梯度手术或残差网络将IL/RL梯度分离，持续利用示范先验来稳定探索
primary_logic: |
  预训练策略 + 专家演示 + 在线环境交互/奖励 → 按1:m节奏交替执行RL更新与IL回灌，并通过梯度手术或网络分离抑制目标冲突 → 输出更稳定、更高样本效率的微调机器人策略
claims:
  - "在14个机器人操作与运动控制任务、3个基准上，IN-RIL整体上比RL-only微调更稳定且样本效率更高，优势在长时程稀疏奖励任务上最明显 [evidence: comparison]"
  - "在Robomimic Transport上，将IN-RIL与IDQL结合可把成功率从12%提升到88% [evidence: comparison]"
  - "去掉梯度分离会明显削弱Hopper与One-Leg (Low) 的性能，说明IL与RL梯度冲突若不处理会破坏训练 [evidence: ablation]"
related_work_position:
  extends: "Diffusion Policy Policy Optimization (DPPO, Ren et al. 2024)"
  competes_with: "DPPO (Ren et al. 2024); IDQL (Hansen-Estruch et al. 2023)"
  complementary_to: "Diffusion Policy (Chi et al. 2023); Action Chunking (Fu et al. 2024)"
evidence_strength: strong
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_IN_RIL_Interleaved_Reinforcement_and_Imitation_Learning_for_Policy_Fine_Tuning.pdf
category: Embodied_AI
---

# IN-RIL: Interleaved Reinforcement and Imitation Learning for Policy Fine-Tuning

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2505.10442), [Code](https://github.com/ucd-dare/IN-RIL)
> - **Summary**: 该文把“先IL预训练、再纯RL微调”的两阶段流程改成“全程交错IL与RL”，并用梯度分离避免两种目标互相破坏，从而让机器人策略在稀疏奖励、长时程任务上更稳、更省环境交互。
> - **Key Performance**: Robomimic Transport 上 IDQL 从 **12%→88%**；DPPO 在 Square 上收敛所需环境步数减少约 **62%**。

> [!info] **Agent Summary**
> - **task_path**: 专家演示 + 预训练策略 + 在线环境交互 -> 微调后的机器人控制策略
> - **bottleneck**: RL-only 微调会快速偏离演示分布，在稀疏奖励下探索失锚；而IL/RL若直接混合更新，同一网络中会发生梯度冲突
> - **mechanism_delta**: 把“一次性IL初始化”改成“训练全程周期性IL回灌 + 梯度分离”，让示范先验持续约束探索
> - **evidence_signal**: 跨 14 个任务、3 个基准的一致比较实验，尤其是 Transport 上 IDQL 12%→88%
> - **reusable_ops**: [interleaved-il-rl-schedule, gradient-separation]
> - **failure_modes**: [demo-coverage不足时IL锚点弱化, m设置不当会过度约束探索或退化为RL-only]
> - **open_questions**: [如何在线自适应选择m, 梯度对齐信号在更大规模视觉策略上是否稳定]

## Part I：问题与挑战

这篇论文解决的不是“如何从零开始学机器人策略”，而是更现实也更难的阶段：**已经有一个IL预训练策略后，如何安全且高效地做RL微调**。

### 真正的问题是什么
现有主流范式通常是：

1. 用专家演示做 IL / BC 预训练；
2. 再切换到纯 RL 做在线微调。

问题在于，**一旦进入 RL-only 阶段，演示先验几乎被“丢掉”了**。这会带来三层瓶颈：

- **探索失锚**：策略逐渐偏离演示分布，尤其在稀疏奖励任务里，reward 信号不足以把策略拉回有效区域。
- **性能塌陷**：预训练策略本来已有一定成功率，但 RL 的噪声更新会破坏已有行为。
- **目标冲突**：IL 希望贴近专家动作，RL 希望追求更高回报；两者对同一参数直接更新时可能互相抵消。

### 为什么现在值得解决
原因很直接：机器人策略越来越常采用**IL 预训练 + RL 精修**，尤其是 diffusion policy 等大策略模型。此时瓶颈不再只是“能不能学到一个起步策略”，而是：

- 如何在**有限演示**下继续榨干示范数据价值；
- 如何让在线微调不再靠脆弱的随机探索；
- 如何让复杂策略在长时程、稀疏奖励任务上稳定提升，而不是先升后崩。

### 输入 / 输出接口
- **输入**：
  - 预训练策略 \(\pi_0\)
  - 专家演示数据 \(D_{exp}\)
  - 在线环境交互与奖励
- **输出**：
  - 一个经微调后、回报更高且不易崩溃的策略

### 边界条件
这篇方法默认：

- 有可用的专家演示；
- 可以进行在线环境交互；
- 演示**不需要奖励标注**；
- 适用于 manipulation 与 locomotion；
- 可挂接到 on-policy 与 off-policy RL 算法上。

---

## Part II：方法与洞察

### 方法框架

IN-RIL 的做法很简单，但关键在于时机与隔离机制：

1. **先做 IL 预训练**  
   用 BC 学一个 warm-start policy。

2. **微调时不再切到纯 RL**  
   而是采用交错更新：
   - 连续做 \(m\) 次 RL 更新；
   - 然后插入 1 次 IL 更新；
   - 如此循环。

3. **用 IL 充当“持续锚点”**  
   IL 不再只是初始化，而是在整个微调过程中周期性把策略拉回专家支持区域。

4. **处理 IL/RL 梯度冲突**  
   论文提出两种分离机制：
   - **Gradient Surgery**：对同一网络中的 IL/RL 梯度做投影，减少相互干扰。
   - **Network Separation**：把 RL 放到 residual policy 上，只让 base policy 吃 IL 梯度，天然隔离两类更新。

5. **理论上把 m 与梯度对齐关系关联起来**  
   作者给出一个分析框架：IL 与 RL 的梯度关系会影响最优 interleaving ratio；实验中则采用固定 \(m\)，且发现 **5-15** 通常效果较好。

### 核心直觉

**什么改变了？**  
从“IL 只在训练开始时起作用”变成“IL 在整个 RL 微调阶段持续参与”。

**哪个瓶颈被改变了？**  
改变的是**策略搜索所处的分布与约束**：
- RL-only 会让策略快速跑出演示支持集；
- 周期性 IL 更新相当于定期把策略重新投影回“可行且有经验支持”的区域；
- 梯度分离则避免这种回拉过程被 RL 直接抵消。

**能力如何改变？**  
这样一来，RL 不再是在整个动作空间里盲目探索，而是在**演示先验附近做有边界的探索**。结果是：
- 更不容易 collapse；
- 更快找到高回报轨迹；
- 尤其在长时程、稀疏奖励任务上，样本效率提升更明显。

换句话说，IN-RIL 的关键不是“把两个 loss 加起来”，而是：
**让 IL 成为在线微调阶段持续生效的行为锚点，同时确保这个锚点不会与 RL 更新互相打架。**

### 为什么这个设计有效
论文给出的解释是一个“非凸景观”视角：

- IL 和 RL 各自都有自己的局部最优；
- 纯 IL 会停在“像专家但不一定高回报”的区域；
- 纯 RL 会掉进“探索不足或高噪声”的局部最优；
- 交错更新让两种目标可以互相“拉出”对方的坏盆地。

这个解释是否完全严密可以讨论，但**经验现象是清楚的**：  
RL-only 时 IL loss 往往持续恶化，而 IN-RIL 能把 IL loss 控制在较合理区间，并带来更高回报。

### 策略性权衡

| 设计选择 | 好处 | 代价 | 适用场景 |
|---|---|---|---|
| 小 \(m\)（更频繁 IL） | 更稳、抗塌陷、更贴近演示 | 可能过度约束探索，额外监督更新更多 | 稀疏奖励、长时程、warm-start 较弱 |
| 大 \(m\)（更少 IL） | 更接近纯 RL，探索更自由 | 更容易漂移，稳定性下降 | dense reward、RL 已较强 |
| Gradient Surgery | 不改网络结构，适合 full-network fine-tuning | 需同时计算并投影两类梯度，计算更重 | 同一网络同时吃 IL/RL 梯度 |
| Network Separation | 冲突隔离更直接，训练更稳定 | 需要 residual policy 结构，设计更受限 | residual fine-tuning 场景 |

---

## Part III：证据与局限

### 关键证据

- **比较实验信号：困难稀疏奖励任务收益最大**  
  在 Robomimic Transport 上，**IDQL 从 12% 提升到 88%**。这说明 IN-RIL 的核心价值不是小修小补，而是把“几乎不会做”的 RL 微调，变成“可以做成”的有锚探索。

- **比较实验信号：长时程装配任务更稳定**  
  在 FurnitureBench 上，IN-RIL 普遍优于 residual PPO。比如 Round-Table Low 从 **0.73→0.93**，Lamp Low 从 **0.63→0.98**。论文还特别强调 RL-only 在这类任务上更容易 collapse。

- **样本效率信号：早期收敛更快**  
  在 Square 上，IN-RIL 让 DPPO 达到高成功率所需环境步数减少约 **62%**。这支持了“IL 作为在线锚点能缩小有效探索空间”的说法。

- **消融信号：中等 interleaving ratio 更合理**  
  对不同 \(m\) 的实验表明，RL-only（\(m=\infty\)）时 IL loss 会显著上升；而适中的 interleaving 能在保持回报增长的同时，控制对演示行为的遗忘。

- **机制消融信号：梯度分离不是装饰件**  
  去掉 gradient separation 后，Hopper 和 One-Leg 的性能都会显著受损。说明“交错训练有效”不只因为多加了一份监督，而是因为**冲突梯度被显式管理了**。

### 1-2 个最值得记住的指标
- **Robomimic Transport + IDQL：12% → 88%**
- **FurnitureBench Round-Table Low：0.73 → 0.93**

### 局限性

- **Fails when**: 演示覆盖严重不足、质量噪声过高或与目标任务分布偏差很大时，IL 无法提供可靠锚点，交错更新的收益会下降；若 \(m\) 设得过小，也可能把探索压得过死。
- **Assumes**: 需要可用的专家演示、可在线交互的环境、一个已预训练的 warm-start policy，以及显式的梯度分离/残差结构；实验中还依赖 50-300 条演示和大量环境步数（\(10^7\) 到 \(10^8\) 量级），这对真实机器人成本并不低。
- **Not designed for**: 无演示的纯 RL、纯离线微调、无在线试错预算的高安全场景，以及需要语言理解/开放词汇泛化的 VLA 级系统。

### 额外需要注意的边界
- 理论部分主张最优 \(m\) 应随梯度关系动态变化，但**实验里实际用的是固定 \(m\)**；因此理论更像设计指导，而不是被完整实现的算法。
- 方法虽说算法无关，但当前验证仍集中在 **DPPO / IDQL / residual PPO** 及标准机器人基准上，跨传感模态、跨机器人形态的泛化证据还不够。

### 可复用组件
- **interleaved IL/RL schedule**：任何“先模仿后强化”的系统都可参考这一训练编排。
- **gradient surgery for multi-objective policy updates**：适合共享参数网络中的目标冲突处理。
- **network separation via residual policy**：适合不想让 RL 直接破坏 base policy 的场景。
- **demo-without-reward reuse**：对没有 reward 标注的演示数据尤其有价值。

## Local PDF reference

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_IN_RIL_Interleaved_Reinforcement_and_Imitation_Learning_for_Policy_Fine_Tuning.pdf]]