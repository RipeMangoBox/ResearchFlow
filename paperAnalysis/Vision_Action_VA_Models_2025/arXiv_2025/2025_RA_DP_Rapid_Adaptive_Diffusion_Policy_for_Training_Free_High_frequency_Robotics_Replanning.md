---
title: "RA-DP: Rapid Adaptive Diffusion Policy for Training-Free High-frequency Robotics Replanning"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/robot-manipulation
  - task/trajectory-replanning
  - diffusion
  - training-free-guidance
  - action-queue
  - dataset/MetaWorld
  - dataset/ManiSkill2
  - opensource/no
core_operator: 将动作序列改写为单调噪声动作队列，并在每次单步去噪时叠加训练自由损失引导，从而持续输出可执行动作并在线重规划。
primary_logic: |
  历史观测/慢变目标 + 在线新反馈（如障碍） → 以不同噪声级别维护动作队列并执行单步去噪 →
  对预测的干净动作施加可微约束的训练自由引导 → 每个控制周期输出一个可执行动作，同时刷新未来动作队列
claims:
  - "Claim 1: 在无障碍 MetaWorld 上，RA-DP 在状态输入下将平均成功率从 63.2% 提升到 66.1%，并把重规划频率从 36.7 Hz 提升到 130.9 Hz；在点云输入下将平均成功率从 58.3% 提升到 68.6%，频率从 16.1 Hz 提升到 82.3 Hz [evidence: comparison]"
  - "Claim 2: 在未见静态/动态障碍的 reach-v2 变体上，RA-DP 无需再训练即可将成功率从 Guided-DP 的 15.0%/10.0% 提升到 63.3%/45.0% [evidence: comparison]"
  - "Claim 3: 混合噪声训练优于纯单调或纯独立噪声；在 MetaWorld easy 子集上，mixed ratio=0.6 取得 66.31% 成功率，为所测试配置中的最佳设置 [evidence: ablation]"
related_work_position:
  extends: "Diffusion Policy (Chi et al. 2023)"
  competes_with: "Consistency Policy (Prasad et al. 2024); Streaming Diffusion Policy (Høeg et al. 2024)"
  complementary_to: "DPM-Solver (Lu et al. 2022); Genie (Dockhorn et al. 2022)"
evidence_strength: strong
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_RA_DP_Rapid_Adaptive_Diffusion_Policy_for_Training_Free_High_frequency_Robotics_Replanning.pdf
category: Embodied_AI
---

# RA-DP: Rapid Adaptive Diffusion Policy for Training-Free High-frequency Robotics Replanning

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2503.04051), [PDF](https://arxiv.org/pdf/2503.04051)
> - **Summary**: 这篇工作把扩散控制的“整段动作先采样完再执行”改成“滚动维护一个不同噪声级别的动作队列”，并在每次去噪时插入训练自由的约束梯度，因此无需重训就能高频响应未见动态障碍。
> - **Key Performance**: MetaWorld 状态输入下重规划频率 130.9 Hz vs 36.7 Hz；未见动态障碍规避成功率 45.0% vs 10.0%。

> [!info] **Agent Summary**
> - **task_path**: 历史观测 + 慢变目标 + 在线新反馈（如障碍位置） -> 连续机器人动作 / 闭环高频重规划
> - **bottleneck**: 扩散策略采样慢，导致新反馈进入控制回路太晚；同时传统条件化方式无法无训练适配未见反馈
> - **mechanism_delta**: 用按噪声深度排序的动作队列替代整段同噪声动作采样，并在每次单步去噪后施加训练自由 loss guidance
> - **evidence_signal**: 标准 MetaWorld 上在提速 3.5×~5× 的同时维持或提升成功率，且在未见障碍规避上显著优于 Guided-DP
> - **reusable_ops**: [monotonic-noise-action-queue, training-free-loss-guidance]
> - **failure_modes**: [very-fast-obstacles-exceed-replanning-bandwidth, over-strong-guidance-causes-goal-deviation-or-action-divergence]
> - **open_questions**: [how-to-handle-non-differentiable-constraints, how-to-scale-to-long-horizon-multi-stage-manipulation]

## Part I：问题与挑战

这篇论文抓住的**真瓶颈**不是“扩散策略不会生成动作”，而是：

1. **扩散控制的闭环带宽太低**  
   传统 Diffusion Policy 往往要先做多步去噪，再一次执行若干动作，之后才看新观测。这样一来，控制器虽然动作质量高，但面对快速变化环境时，**新反馈进入决策链路的时机太晚**。

2. **已有 guidance 方式不适合未见动态反馈**  
   - classifier-free guidance：要把条件在训练时就喂进去，测试时对新条件泛化有限。  
   - classifier-guided：每种新反馈都要额外训练分类器/判别器，部署成本高，也不适合临时出现的新障碍或新约束。

3. **机器人真实部署需要“边采样边反应”**  
   在人机共域、动态避障、抓取干扰等场景里，问题不是离线规划出一条漂亮轨迹，而是**能不能每个控制周期都重新纠偏**。

### 输入/输出接口

- **输入**：
  - 历史观测 \(O_t\)（状态或点云）
  - 慢变条件（如目标位置）
  - 在线新反馈 \(G\)（如障碍物位置、速度约束、途经点约束）
- **输出**：
  - 当前时刻可直接执行的动作
  - 以及一串仍在去噪中的未来动作队列

### 边界条件

这套方法成立有几个前提：

- 在线反馈最好能写成**可微的标量损失**；
- 仍然依赖一个先验上训练好的 diffusion policy；
- 论文主打的是**局部高频重规划**，不是全局任务级规划。

**Why now?**  
因为扩散策略已经证明了在高维连续动作上的表达力，但它在动态环境中的部署瓶颈越来越明显：如果不能把“好动作”变成“来得及更新的好动作”，就很难进真实机器人场景。

---

## Part II：方法与洞察

RA-DP 的核心目标很明确：**不重训地接入新约束，同时把重规划频率提到接近低层控制需要的水平。**

### 核心直觉

#### 1）把“整段动作一起去噪”改成“队列里每个动作处于不同去噪深度”
- **改变了什么**：训练时不再让整段 action horizon 共享同一个噪声级别，而是让每个动作独立/按单调顺序带不同噪声。
- **改变了哪个瓶颈**：消除了“标准 DDPM 训练分布”和“高频滚动推理分布”之间的不匹配。
- **带来什么能力**：推理时只做一次去噪，也能稳定得到一个可执行的当前动作，而不是整段动作都要重新采样完。

#### 2）把“完整采样一次后执行多步”改成“每去噪一步就执行队首动作”
- **改变了什么**：维护一个动作队列，队首最干净、越往后越噪；每次单步去噪后，执行队首、弹出它，再补一个新噪声动作到队尾。
- **改变了哪个瓶颈**：把每个动作本来要走完的多步采样，分摊到连续控制时刻里。
- **带来什么能力**：实现高频闭环重规划，而不是“采样一大段、执行一大段”的低频开环式行为。

#### 3）把“为新条件重训模型”改成“对预测动作直接施加训练自由约束”
- **改变了什么**：不再为障碍、路径约束等新反馈训练额外 classifier，而是定义一个可微损失 \(f(\hat A_0, G)\)，对预测的干净动作反向传播梯度。
- **改变了哪个瓶颈**：将新反馈的接入成本从“新增监督/重训”降为“定义可微约束”。
- **带来什么能力**：测试时可即插即用地适配未见环境反馈。

### 方法拆解

#### 1. 训练阶段：为“滚动式推理”重写噪声分布
RA-DP 不沿用 DP 的“整段序列同噪声”训练范式，而是让动作序列中每个位置拥有自己的 diffusion step。论文还发现：

- 直接预测 **clean action** 比预测噪声更稳；
- 单调噪声和独立噪声混合训练，能兼顾收敛性与推理匹配性。

这一步的本质是：**先把模型训练成会处理“同一队列里噪声深度不一致”的情况。**

#### 2. 推理阶段：动作队列 + 出队/入队
推理时维护一个固定长度动作队列：

- 队首：接近干净、马上可执行
- 队尾：更噪、更远期

每个控制周期：
1. 对整个队列做**一步去噪**；
2. 得到当前可执行动作；
3. 执行它；
4. 队首出队；
5. 队尾补进一个新的高噪动作。

所以 RA-DP 不是靠“把每个动作只采样 1 步”提速，而是靠**把一个动作完整的 H 步去噪过程，沿时间轴流式展开**。  
这就是它比简单 one-step consistency policy 更关键的地方：**速度提升不必直接牺牲采样质量。**

#### 3. 训练自由 guidance：把新环境反馈变成可微能量
论文用的是 loss-based guidance：

- 给定当前预测的干净动作序列；
- 根据障碍物、速度约束等构造可微损失；
- 通过 U-Net 反传梯度，修正当前采样结果。

一个很巧的副作用是：  
由于队列后面的动作噪声更大，同样的 guidance 对它们的有效影响更弱。于是系统会**天然地优先修正近期动作**，而不是对很远期的未来过度反应。这对稳定重规划是有利的。

### 战略性权衡

| 设计选择 | 解决的瓶颈 | 收益 | 代价 / 风险 |
| --- | --- | --- | --- |
| 单调噪声动作队列 | 低重规划频率、训练推理不匹配 | 每步都能输出新动作，保持闭环 | 需要重写训练噪声机制 |
| 直接预测 clean action | varying-noise 训练不稳 | 收敛更好、便于 guidance 回传 | 仍保留扩散模型推理复杂度 |
| 训练自由 loss guidance | 未见反馈需要重训 | 新约束可即插即用 | 依赖约束可微、步长敏感 |
| 混合噪声训练 | 纯单调/纯独立各有偏差 | 在收敛与部署一致性间折中 | 需要额外调 mixed ratio |

---

## Part III：证据与局限

### 关键实验信号

#### 1. 标准基准上：不是“提速换性能”，而是“提速同时不掉甚至更强”
在无障碍 MetaWorld 上，RA-DP 依然优于或持平原始扩散策略：

- **状态输入**：66.1% vs 63.2%，同时 **130.9 Hz vs 36.7 Hz**
- **点云输入**：68.6% vs 58.3%，同时 **82.3 Hz vs 16.1 Hz**

这个结果很关键，因为它说明动作队列不是单纯缩短 horizon 或粗暴降采样，而是**在保持扩散质量的同时提升闭环频率**。

#### 2. 未见动态反馈上：高频 + training-free guidance 两者缺一不可
在加入未见障碍的 reach-v2 变体上：

- 静态障碍：63.3% vs 15.0%
- 动态障碍：45.0% vs 10.0%

这里最强的证据不是“能避障”，而是**不用重训也能避障**，并且相对 Guided-DP 的大幅优势表明：  
仅仅有 guidance 还不够，**还必须足够快地重规划**。

#### 3. 消融支持了机制解释
- **mixed noise ratio**：0.6 最优，说明训练时既要学会 independent noise 的鲁棒性，又不能偏离推理时的 monotonic queue 太远。
- **guidance step size**：步长越大，轨迹离障碍越远，但过强 guidance 也更容易带来偏航。
- **动态障碍速度测试**：当障碍速度过快（归一化速度 > 0.14），成功率跌到 0，直接暴露了系统的闭环带宽边界。

### 局限性

- **Fails when**: 障碍移动过快、超出当前重规划带宽；或者避障导致末端执行器偏离目标过远，轨迹过长被截断，最终任务失败。
- **Assumes**: 存在一个已训练好的扩散策略；在线反馈可转为对动作的可微损失；真实部署依赖稳定感知与定位模块（文中用双 RGBD 相机 + Grounding DINO / SAM / SAM-6D 类模块），并默认有足够算力支撑实时推理。
- **Not designed for**: 非可微符号约束、需要长时全局任务分解的多阶段操作、反馈严重延迟或不可观测的场景。

### 复现与可扩展性判断

- 方法本身模块化很强，**动作队列机制**和**training-free guidance 接口**都很可复用。
- 但真实机器人部分更偏 demo 性质，而非大规模统计评测；再加上**未见代码发布**，所以尽管理念清晰，工程复现成本仍不低。
- 论文还显式指出其方法可与更快的 diffusion solver 结合，这意味着 RA-DP 更像一个**控制时序层 / 采样编排层**，而不仅是单一模型结构。

### 可复用组件

1. **Monotonic-noise action queue**：适合任何需要“每个控制 tick 都产出动作”的扩散控制器。  
2. **Training-free differentiable guidance**：把新约束接成 loss，而不是重新训练条件模型。  
3. **Mixed noise schedule training**：解决特殊推理轨迹下的 train-test mismatch。  
4. **Queue-based rolling denoising**：可作为高频闭环扩散控制的一般模板。

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_RA_DP_Rapid_Adaptive_Diffusion_Policy_for_Training_Free_High_frequency_Robotics_Replanning.pdf]]