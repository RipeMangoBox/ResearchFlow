---
title: "Responsive Noise-Relaying Diffusion Policy: Responsive and Efficient Visuomotor Control"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/visuomotor-control
  - task/robot-manipulation
  - diffusion
  - sequential-denoising
  - noise-relaying-buffer
  - dataset/ManiSkill2
  - dataset/ManiSkill3
  - dataset/Adroit
  - opensource/no
core_operator: 通过带噪动作接力缓存与顺序去噪，在每个控制周期输出基于最新观测的干净动作，同时保持未来动作的模式一致性
primary_logic: |
  最新观测 + 递增噪声动作缓存 → 对缓存内各动作同步去噪一步、执行头部无噪动作、尾部补入全噪动作 → 在单步rollout下同时获得响应性、模式稳定性与低推理开销
claims:
  - "Claim 1: 在 5 个响应敏感任务上，RNR-DP 在文中所有已报告的 state/visual 设置下都优于 Diffusion Policy，总体成功率提升 18.0%，其中 Adroit Relocate 提升 38.6% [evidence: comparison]"
  - "Claim 2: 在 4 个常规任务上、相同 NFEs/a=1 的条件下，RNR-DP 的总体成功率比 8-step DDIM 高 6.9%，并以 12.5× 更少的每动作去噪评估保持与 100-step DDPM 相当的平均性能 [evidence: comparison]"
  - "Claim 3: 混合噪声调度与 laddering 初始化是关键组件；改为纯线性/纯随机调度或纯噪声初始化，会使 Adroit Relocate/Door 的成功率下降 6.3% 到 26.9% [evidence: ablation]"
related_work_position:
  extends: "Diffusion Policy (Chi et al. 2023)"
  competes_with: "Streaming Diffusion Policy (Høeg et al. 2024); Consistency Policy (Prasad et al. 2024)"
  complementary_to: "Policy Decorator (Yuan et al. 2024)"
evidence_strength: strong
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Responsive_Noise_Relaying_Diffusion_Policy_Responsive_and_Efficient_Visuomotor_Control.pdf
category: Embodied_AI
---

# Responsive Noise-Relaying Diffusion Policy: Responsive and Efficient Visuomotor Control

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2502.12724), [Project](https://rnr-dp.github.io)
> - **Summary**: 论文把 Diffusion Policy 每步“从零采样整段动作”的方式改成“跨时刻接力去噪”的缓冲机制，从而在单步执行时同时保留最新观测响应性和多模态动作的一致性。
> - **Key Performance**: 响应敏感任务总体成功率较 Diffusion Policy 提升 **18.0%**；在常规任务的 **NFEs/a=1** 条件下较 8-step DDIM 提升 **6.9%**。

> [!info] **Agent Summary**
> - **task_path**: 视觉/状态观测下的模仿学习控制 -> 单步机器人动作
> - **bottleneck**: 多模态 demonstrations 下，Diffusion Policy 依赖较大 action horizon 抑制 mode bouncing，但这会让多数执行动作脱离最新观测
> - **mechanism_delta**: 用递增噪声的动作缓冲区跨时间复用去噪进度，使每个控制周期只需再去噪一步就能输出当前干净动作
> - **evidence_signal**: 3 个基准 9 个任务上的对比与消融共同表明：RNR-DP 在响应敏感任务整体优于 DP，且关键收益来自缓冲接力而非单纯加速
> - **reusable_ops**: [noise-relaying-buffer, mixture-noise-scheduling]
> - **failure_modes**: [真实机器人闭环效果未验证, buffer容量与任务时长失配时性能下降]
> - **open_questions**: [能否稳定迁移到真实机器人控制频率约束下, 能否与Transformer策略或在线RL微调结合]

## Part I：问题与挑战

这篇论文抓住的核心矛盾不是“Diffusion Policy 推理慢”，而是：

- **为了稳定，多模态策略需要长 action horizon**
- **为了响应环境，又希望每一步都看最新观测**

而 Diffusion Policy 恰好把这两件事绑死在一起了。

### 1）真正的问题是什么？

在 Diffusion Policy 里，每次控制都会从噪声中重新生成一段未来动作序列。  
对于**多模态 demonstrations**，相邻两次推理可能会落到不同动作模式上，于是出现作者说的 **mode bouncing**：动作在不同合理策略之间来回跳，导致控制不稳定。

因此原始 DP 往往要设置较大的执行 horizon `Ta`，一次推理后连续执行多步动作。这样至少这一小段动作来自同一次采样，模式更一致。

### 2）为什么这会成为瓶颈？

大 `Ta` 带来的代价是：  
**后面被执行的动作并不是基于最新观测生成的。**

这在静态、容错高的任务里问题不大，但在以下场景会很伤：

- 动态物体
- 接触丰富 manipulation
- 灵巧手微调
- 需要连续纠偏的任务

例如球滑了、门把接触偏了、椅子要倒了，此时你最需要的是“下一拍马上改动作”，而不是继续执行几步旧计划。

### 3）作者为什么说这是“多模态问题”而不是“短 horizon 天生不行”？

这是论文里一个很重要的因果诊断。

作者用 **单模态 RL 数据** 在 StackCube 上测试 DP，发现 `Ta=1` 并不会像多模态 demo 那样明显掉点，甚至略好。  
这说明：

- **不是单步控制本身有问题**
- **而是多模态数据下，每次独立采样会反复换 mode**

所以真正瓶颈是：

> **“模式一致性”目前依赖长 horizon 来硬维持，而长 horizon 又损失响应性。**

### 4）输入/输出接口与边界条件

- **输入**：最新状态或视觉观测（state / RGBD / image）+ 模仿学习数据
- **输出**：连续控制动作；论文目标是把它退化为**每次只执行 1 个动作**但仍然稳定
- **主要适用边界**：响应敏感、动态、接触丰富任务
- **非核心场景**：静态、易任务；这时 RNR-DP 的主要收益更多体现在**加速**而非能力跃迁

---

## Part II：方法与洞察

RNR-DP 的关键不是“把 diffusion 步数硬砍少”，而是把：

- **动作一致性**  
从  
- **一次推理里生成整段动作**  

转移到了

- **跨时间维护一条逐渐成形的未来动作轨迹**

上。

### 方法主线

#### A. Noise-relaying buffer：带噪动作接力缓存

作者维护一个长度为 `f` 的动作缓存，里面不是干净动作，而是：

- 头部噪声最小
- 尾部噪声最大
- 整体是“逐渐变脏”的未来动作序列

每个控制周期做三件事：

1. **对缓存中所有动作统一去噪一步**
2. **取出头部已经干净的动作，立即执行**
3. **其余动作左移，并在尾部补一个全噪动作**

于是，一个动作不会在同一时刻被完整去噪，而是**跨多个控制周期逐步成熟**。

#### B. Sequential denoising：顺序去噪而不是每步重开

这意味着：

- 当前动作总是看了**最新观测**
- 未来动作又不是从零开始重采样，而是继承上一步的去噪进度

因此它同时保住了两件事：

- **响应性**：当前动作每次都重看最新观测
- **一致性**：未来动作沿着同一条“已部分成形”的轨迹继续演化，不容易频繁换 mode

#### C. Mixture noise scheduling：训练时对齐这种“多噪声位点”结构

RNR-DP 训练时不再像 DP 一样整段动作用同一噪声级别，而是按动作位点分别注入噪声。

它混合两种调度：

- **random schedule**：让模型学会独立处理不同噪声级别的单个动作
- **linear schedule**：让模型适应推理时缓存里“噪声递增”的结构

二者混合的目的很明确：  
既学“局部去噪能力”，又学“缓存接力的结构一致性”。

#### D. Laddering initialization：启动阶段别让缓冲区分布错位

推理开始时，缓存全是纯噪声。  
如果直接拿来跑，会和训练时见到的“递增噪声缓冲区”分布不一致。

所以作者先做一段初始化去噪，让缓存从“全噪”变成“阶梯噪声”状态，再进入正式控制。这个步骤是为了减轻 cold-start 失配。

#### E. Noise-aware conditioning：每个动作槽位都知道自己处于哪个噪声级别

因为缓存里不同位置对应不同 diffusion level，单一时间嵌入不够了。  
作者为每个动作槽位编码自己的 noise level，再和 observation feature 一起喂给策略网络。

这一步解决的是：  
**模型必须知道“我正在给哪个噪声阶段的动作去噪”**。

### 核心直觉

把一句话说透：

> **RNR-DP 把“每步从纯噪声独立采样一个动作序列”改成了“跨时间维护一条持续成形的未来动作链”。**

这带来的因果变化是：

- **What changed**：从“独立重采样整段动作”变成“对接力缓存持续去噪”
- **Which bottleneck changed**：  
  - 消除了“模式一致性必须依赖长 `Ta`”这一约束  
  - 减少了“当前动作无法看最新观测”的信息滞后
- **What capability changed**：  
  - 可以在 `Ta=1` 的单步 rollout 下仍保持稳定  
  - 对动态环境更快响应  
  - 每动作只需很少的去噪评估

更因果地说，为什么它有效？

1. **缓存继承了上一步的部分去噪结果**  
   所以 mode 不会每步重新抽签。
2. **头部动作每次都基于最新观测再去噪一次**  
   所以当前动作是响应式的。
3. **尾部持续补入纯噪声**  
   所以未来仍保留多模态生成能力，而不是退化成固定轨迹。
4. **训练时用 mixture schedule 对齐这种结构**  
   避免“训练看到一种噪声布局，推理用另一种布局”的失配。

### 战略权衡

| 设计 | 改变了什么约束/分布 | 带来什么能力 | 代价/权衡 |
|---|---|---|---|
| Noise-relaying buffer | 从“每步独立重采样”改为“跨时间持续演化” | 单步响应 + 模式稳定 | 需要维护 buffer，并调容量 `f` |
| Sequential denoising | 把多次去噪摊到多个控制周期 | 每次 1 个动作只需 1 次新增去噪 | 依赖连续闭环执行，不是一次性整段规划 |
| Mixture noise scheduling | 让训练覆盖随机噪声与递增噪声两类分布 | 训练/推理更对齐，鲁棒性更强 | 训练协议更复杂 |
| Laddering initialization | 解决启动时全噪缓冲区与推理目标分布不匹配 | 冷启动更稳定 | 首帧前需要额外初始化去噪 |
| Noise-aware conditioning | 从单一时间嵌入变为槽位级噪声感知 | 不同位置动作可按各自噪声阶段解码 | 条件编码更复杂 |

---

## Part III：证据与局限

### 关键证据信号

- **比较信号：响应敏感任务全面领先**  
  在 5 个 response-sensitive 任务上，RNR-DP 在文中所有已报告的 state/visual 设置中都优于 Diffusion Policy。最强结果是总体 **+18.0%**，其中 Adroit Relocate **+38.6%**。  
  这支持“单步响应 + 模式稳定”确实解决了动态 manipulation 的核心问题。

- **比较信号：不只更响应，也更高效**  
  在 4 个常规任务上，按作者提出的 **NFEs/a**（每动作神经网络评估次数）做公平比较时，RNR-DP 在 **NFEs/a=1** 下比 8-step DDIM 高 **6.9%**，并接近 100-step DDPM 的平均性能。  
  这说明它不是单纯“拿性能换速度”，而是通过复用跨时刻去噪进度来省计算。

- **分析信号：作者的瓶颈诊断是有因果支撑的**  
  单模态 RL 数据上，DP 的 `Ta=1` 没有明显退化；退化主要发生在多模态 demonstrations。  
  这直接支持论文主张：**大 horizon 的根因是抑制多模态 mode bouncing，而不是单步控制天然无效。**

- **消融信号：关键收益不是偶然调参**  
  去掉 mixture noise scheduling、去掉 laddering initialization、改成直接预测 action 而非噪声，性能都会明显下降。  
  说明真正起作用的是“buffer + 去噪接力 + 训练/推理分布对齐”这一整套设计。

### 1-2 个最关键指标

- **响应性收益**：相对 Diffusion Policy，响应敏感任务总体成功率 **+18.0%**
- **效率收益**：相对 8-step DDIM（同 NFEs/a=1），常规任务总体成功率 **+6.9%**

### 局限性

- **Fails when**: 需要真实机器人时延、控制噪声、感知误差共同作用的场景；论文没有真实机器人实验，而且部分视觉响应任务的绝对成功率仍然偏低（例如 Door / RollBall 视觉设定下仍只在较低水平）。
- **Assumes**: 有可用 demonstrations；任务时长允许设置合适的 noise-relaying buffer 容量；依赖 diffusion-policy 风格骨干与多噪声级条件编码；效率主要用 **NFEs/a** 衡量，而不是端到端 wall-clock latency；正文未明确给出代码开放信息。
- **Not designed for**: 无演示的纯在线 RL、长程高层规划、非响应敏感任务上的显著能力提升证明；在这些场景里它更像一种保持性能的加速封装，而不是新的任务求解范式。

### 可复用组件

这篇论文最值得迁移的不是某个具体超参，而是几个“操作符”：

- **noise-relaying buffer**：适合任何“每步重新生成未来动作序列”的 diffusion policy
- **mixture noise scheduling**：适合训练/推理噪声结构不完全一致的序列生成器
- **laddering initialization**：适合流式生成或缓存式生成的冷启动
- **noise-aware slot conditioning**：可迁移到 Transformer 或其他时序策略骨干

### 一句话评价

这篇工作的价值在于，它没有把问题看成“如何更快采样 diffusion”，而是更准确地指出：  
**真正卡住机器人闭环控制的是“长 horizon 保一致 vs 短 horizon 保响应”的结构性矛盾。**  
RNR-DP 用一个跨时间接力的去噪缓冲区，把这两个目标第一次较干净地解耦了。

## Local PDF reference

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Responsive_Noise_Relaying_Diffusion_Policy_Responsive_and_Efficient_Visuomotor_Control.pdf]]