---
title: "IMLE Policy: Fast and Sample Efficient Visuomotor Policy Learning via Implicit Maximum Likelihood Estimation"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/visuomotor-policy-learning
  - task/imitation-learning
  - implicit-maximum-likelihood-estimation
  - rejection-sampling
  - temporal-consistency
  - dataset/Push-T
  - dataset/Robomimic
  - opensource/partial
core_operator: 用条件RS-IMLE把行为克隆训练成“每条示范轨迹都必须被某个单步生成候选覆盖”，并在推理时用轨迹重叠段最近邻选择维持时序一致性。
primary_logic: |
  图像观测/机器人状态 + 随机潜变量 → 对每个条件生成多条候选动作轨迹，经过拒绝采样后选取最接近示范的候选进行训练，并在推理时按与上一段未执行轨迹的重叠距离选模式 → 单步输出保持多模态的动作序列
claims:
  - "在 Push-T 的数据量扫描中，IMLE Policy 达到 0.5 reward 所需数据少于 30%，而 Diffusion Policy 约需 43%，1-step Flow Matching 超过 80% [evidence: comparison]"
  - "在真实鞋架任务的推理速度测试中，IMLE Policy 为 111 Hz，而 100-step Diffusion Policy 为 1.8 Hz [evidence: comparison]"
  - "在 20-demonstration 的仿真基准设置中（Kitchen 未报告结果），IMLE Policy 在其余 7 个任务上都达到最高或并列最高成功率 [evidence: comparison]"
related_work_position:
  extends: "Rejection Sampling IMLE (Vashist et al. 2025)"
  competes_with: "Diffusion Policy (Chi et al. 2023); Flow Matching (Lipman et al. 2023)"
  complementary_to: "3D Diffusion Policy (Ze et al. 2024); Affordance-Centric Policy Learning (Rana et al. 2024)"
evidence_strength: strong
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_IMLE_Policy_Fast_and_Sample_Efficient_Visuomotor_Policy_Learning_via_Implicit_Maximum_Likelihood_Estimation.pdf
category: Embodied_AI
---

# IMLE Policy: Fast and Sample Efficient Visuomotor Policy Learning via Implicit Maximum Likelihood Estimation

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2502.12371), [Project](https://imle-policy.github.io/)
> - **Summary**: 论文把视觉行为克隆从“多步扩散去噪”改成“单步生成多条候选并强制覆盖每条示范”的条件 RS-IMLE，因此在少量示范下更能保住多模态，同时显著提升实时推理速度。
> - **Key Performance**: Push-T 上达到 0.5 reward 所需数据量 <30%（Diffusion≈43%，1-step Flow Matching>80%）；真实鞋架任务推理速度 111 Hz（Diffusion 1.8 Hz）。

> [!info] **Agent Summary**
> - **task_path**: 少量示范下的图像/机器人状态序列 -> 动作轨迹 chunk -> 闭环执行
> - **bottleneck**: 低数据条件下既要覆盖多峰动作分布，又不能依赖扩散式多步采样
> - **mechanism_delta**: 用条件 RS-IMLE 强制每条示范都被某个候选轨迹命中，并在推理时用与上一步未执行轨迹的重叠距离来锁定模式
> - **evidence_signal**: 8 个仿真任务 + 2 个真实任务对比，辅以 Push-T 数据量扫描、时间一致性消融和超参稳定性分析
> - **reusable_ops**: [conditional RS-IMLE objective, overlap-based trajectory selection]
> - **failure_modes**: [highly-multimodal long-horizon mode switching without consistency, sensitivity to noisy or suboptimal demonstrations]
> - **open_questions**: [scaling to large heterogeneous robot datasets, better consistency mechanisms without locking into bad futures]

## Part I：问题与挑战

这篇论文要解决的不是“机器人能否从示范学动作”，而是更尖锐的三难题同时成立：

1. **动作本身是多模态的**  
   同一个视觉状态下，可能有多条都正确的动作轨迹。比如先拿左鞋还是右鞋、先推左侧还是右侧，都可能成功。

2. **机器人示范数据很贵**  
   真机采集常常只有十几到几十条轨迹，扩散类方法虽然强，但通常更依赖较大数据量。

3. **部署时需要实时控制**  
   扩散策略往往要多步迭代去噪，推理慢；而很多单步生成方法又容易把多模态“平均掉”，出现 mode collapse。

**真正瓶颈**是：  
现有方法里，“保多模态”与“多步采样”常被绑定在一起。扩散策略能表达复杂分布，但代价是慢；单步方法快，但在少数据、模式不平衡时容易塌缩到主模态或平均轨迹。

**输入/输出接口**：
- **输入**：最近 `To=2` 步观测，包含图像与机器人状态
- **输出**：未来 `Tp=16` 步动作序列
- **控制方式**：每次执行前 `Ta=8` 步，再闭环重规划
- **适用边界**：以 manipulation 为主的行为克隆场景，尤其是少样本、存在多条正确解的任务

**为什么现在值得做**：  
Diffusion Policy 已是视觉模仿学习的强基线，但真实机器人越来越需要“少样本 + 边缘实时部署”。这使得一个**天然单步、仍能保多模态**的替代训练原则变得很有价值。

## Part II：方法与洞察

### 方法主线

论文的核心做法，是把 RS-IMLE 从无条件图像生成改造成**条件行为克隆**。

#### 1. 条件 RS-IMLE 训练
对每条示范 `(观测 o, 动作轨迹 a)`：

- 采样多个潜变量 `z`
- 在同一个条件 `o` 下，单步生成多条候选动作轨迹
- 对“离示范太近”的候选做拒绝采样过滤
- 在剩下的候选里，选择**最接近示范轨迹**的一条来更新模型

直观上，它不是在问“这次采样像不像数据分布整体”，而是在问：

> **每条训练示范，是否至少能被某个生成样本解释到？**

这会把训练压力从“拟合高频平均趋势”转向“覆盖每个观测下出现过的所有有效动作模式”。

#### 2. 单步生成器
实现上，作者沿用了 Diffusion Policy 的 1D U-Net 骨干，但去掉扩散时间步嵌入。  
也就是说，**架构几乎不变，训练目标变了**。这点很重要：收益主要来自目标函数与推理机制，而不是换了一个更重的模型。

#### 3. 推理时的时间一致性
IMLE 的单步多模态生成有一个自然副作用：  
如果每次都重新采样，它可能在多个合理模式之间来回跳。

为此作者加了一个很轻量的推理策略：

- 每次先批量生成多条候选轨迹
- 比较这些候选轨迹的前半段，与上一轮“还没执行完”的轨迹后半段是否衔接
- 选最连续的一条执行

并且每隔若干步随机重置一次，避免长期锁死在一条差轨迹上。

### 核心直觉

**改变了什么**：  
从“逐步去噪恢复一条轨迹”改成“在同一条件下一次性生成多条候选，并强制每条示范都被某个候选覆盖”。

**哪个瓶颈被改变了**：  
- 原先的瓶颈是：少样本时，少数模式梯度弱、易被主模态吞没；同时多步采样带来实时性瓶颈。
- 现在变成：每个示范都必须被命中，所以**少数模式也能稳定收到训练信号**；而单步生成直接消除了扩散式迭代推理约束。

**能力为什么会变强**：  
- **对数据点而非平均分布负责**：每条示范都需要“有个候选能解释它”，所以不容易把不同模式平均化。
- **拒绝采样避免重复照顾已拟合样本**：当某些候选已经足够接近某条示范时，不再反复让它们占用选择机会，训练会更愿意去覆盖其它区域/模式。
- **一致性选择把“会生成多解”变成“能稳定执行一种解”**：训练阶段保留多模态，执行阶段再用轨迹重叠约束决定当前选哪条模式。

可以把它概括成一条因果链：

**把训练目标从“逼近平均趋势”改成“覆盖每条示范”**  
→ **少数模式不再缺梯度**  
→ **低数据下仍能保持多模态**  
→ **再配合单步生成，获得实时控制能力**

### 策略权衡

| 设计选择 | 解决的瓶颈 | 收益 | 代价/风险 |
|---|---|---|---|
| 条件 RS-IMLE | 少样本下模式遗漏 | 少数模式也被覆盖，样本效率更高 | 训练时每个样本都要生成多候选并做最近邻选择 |
| 单步生成器 | 扩散推理太慢 | 推理极快，适合实时控制 | 单次采样彼此独立，天然缺少时序黏性 |
| 重叠段一致性选择 | 长时执行中的模式跳变 | 轨迹更平滑、决策更稳定 | 若已选未来片段有误，可能把错误延续几步 |
| 复用 Diffusion Policy 的 U-Net 骨干 | 新方法难迁移 | 便于直接替换现有 pipeline | 论文尚未证明在更大模型/更大数据上仍最优 |

## Part III：证据与局限

### 关键证据

1. **数据量扫描（comparison）**  
   在 Push-T 上，IMLE Policy 达到 0.5 reward 所需数据量不到 30%，而 Diffusion Policy 约需 43%，1-step Flow Matching 超过 80%。  
   **结论**：它不是“靠更多训练步数补回来”，而是在低数据区间就表现出更好的学习曲线。

2. **多任务基准（comparison）**  
   作者在 8 个仿真任务上做全数据和 20-demo 评测。  
   - 全数据下：IMLE 总体与 Diffusion 竞争，部分任务更优，整体不是靠单一环境取胜。  
   - 20-demo 下：在有结果的 7 个任务上都达到最高或并列最高成功率。  
   **结论**：优势主要集中在论文真正主打的区域——**少样本 + 多模态**。

3. **真实世界验证（comparison）**  
   在真实 Push-T 和 Shoe Racking 上，只用 17/35 条示范训练，IMLE Policy 都在低数据下领先基线。  
   同时真实鞋架任务的推理速度为 **111 Hz**，而 100-step Diffusion Policy 仅 **1.8 Hz**。  
   **结论**：它不是仅在仿真里“快”，而是在真实视觉输入上也具备部署价值。

4. **模式捕获与消融（analysis + ablation）**  
   - 定性分析显示，IMLE 能保留 Push-T 中少数但合理的边缘模式；Diffusion 更容易偏向主模态，1-step Flow Matching 更容易塌缩成平均轨迹。  
   - 时间一致性机制在高多模态任务里改善执行稳定性。  
   - 对 `ϵ` 和每条件采样数的 sweep 相对平稳。  
   **结论**：论文的收益和它的机制是对得上的，不只是“换个模型碰巧更好”。

### 局限性

复现与扩展时，最该注意的不是模型结构，而是它的**依赖条件**。

- **Fails when**: 高多模态、长时序任务如果不加一致性约束，策略会在合理模式之间反复切换；而如果一致性机制选中了带误差的未来片段，也可能短时间内持续执行次优行为。  
- **Assumes**: 需要质量较高、相对一致的专家示范；默认轨迹间欧氏距离足以衡量“行为接近”；训练时依赖每条件多候选生成与最近邻搜索；当前验证主要在小到中等规模 manipulation 数据集、固定 horizon 和 U-Net 设定下完成。  
- **Not designed for**: 大规模异构多任务数据上的统一策略训练、显式安全约束控制、以及论文中尚未实测的 RL fine-tuning 场景。

**额外资源/复现约束**：
- 推理很快，但训练比普通单步策略多出**候选采样 + 最近邻搜索**开销。
- 真机速度结果基于 RTX 3090 工作站测得；边缘设备上的绝对吞吐仍需单独验证。
- 论文写法显示项目页提供代码/视频，但训练细节与数据释放表述部分仍带有承诺色彩，因此开源成熟度我更倾向记为 `partial`。

### 可复用组件

- **条件 RS-IMLE 目标**：可替换现有 action-chunk policy 的训练目标，尤其适合少样本多模态行为克隆。
- **重叠段最近邻一致性选择**：适用于任何 chunked policy，不限于 IMLE。
- **“同骨干、换目标”评测范式**：对比时尽量固定 backbone，更能看清训练原则本身的贡献。

## Local PDF reference

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_IMLE_Policy_Fast_and_Sample_Efficient_Visuomotor_Policy_Learning_via_Implicit_Maximum_Likelihood_Estimation.pdf]]