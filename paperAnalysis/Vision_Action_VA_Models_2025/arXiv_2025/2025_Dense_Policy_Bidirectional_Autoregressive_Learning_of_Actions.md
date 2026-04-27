---
title: "Dense Policy: Bidirectional Autoregressive Learning of Actions"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/robotic-manipulation
  - autoregressive
  - bidirectional-expansion
  - coarse-to-fine
  - dataset/Adroit
  - dataset/DexArt
  - dataset/MetaWorld
  - opensource/promised
core_operator: "从零初始化动作种子出发，通过“线性上采样 + 观测条件跨注意力”做双向递归扩展，逐层把稀疏关键帧细化为连续动作序列"
primary_logic: |
  RGB/点云/本体感觉观测 → 编码观测特征并从单个动作种子生成稀疏关键帧 →
  对上一层动作做双向上采样并结合观测进行跨注意力细化 →
  经过对数层级递归得到固定时域的稠密连续动作序列
claims:
  - "在 11 个仿真任务上，3D Dense Policy 平均成功率 72%，高于 DP3 的 53%；2D 版本平均 52%，高于 Diffusion Policy 的 25% [evidence: comparison]"
  - "将 Dense Policy 改为 next-token 或 next-chunk 后，在 Door、Bin Picking、Shelf Place、Box Close 上的学习速度和最终性能均下降，说明双向扩展是关键因果因素 [evidence: ablation]"
  - "在真实机器人 Flower Arrangement 任务上，3D Dense Policy 将至少插入一朵花的成功率从 50% 提升到 70%，平均插花数从 0.6 提升到 1.0 [evidence: comparison]"
related_work_position:
  extends: "CARP (Gong et al. 2024)"
  competes_with: "Diffusion Policy (Chi et al. 2023); 3D Diffusion Policy (Ze et al. 2024)"
  complementary_to: "OpenVLA (Kim et al. 2024); π0 (Black et al. 2024)"
evidence_strength: strong
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Dense_Policy_Bidirectional_Autoregressive_Learning_of_Actions.pdf
category: Embodied_AI
---

# Dense Policy: Bidirectional Autoregressive Learning of Actions

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2503.13217), [Project](https://selen-suyue.github.io/DspNet/)
> - **Summary**: 这篇论文把机器人连续动作生成从“单向逐步预测/整体一次性生成”改成“先生成稀疏关键帧、再双向递归补全”的层级自回归过程，从而在不做动作离散化的前提下提升长程时序一致性与控制精度。
> - **Key Performance**: 仿真中 3D 平均成功率 72% vs DP3 53%，2D 平均成功率 52% vs DP 25%；真实 3D Flower Arrangement 成功率 70% vs RISE 50%。

> [!info] **Agent Summary**
> - **task_path**: RGB/点云/本体感觉观测 -> 固定时域末端执行器 TCP pose 动作序列
> - **bottleneck**: 连续动作空间里，左到右 next-token/chunk 难以同时建模长程依赖与细粒度控制精度，而整体生成式策略头又往往训练和推理更重
> - **mechanism_delta**: 将动作头改成从零动作种子出发的双向层级稠密化：每层先上采样已有关键帧，再用观测条件细化新增动作
> - **evidence_signal**: 同骨干、同示范、同训练预算的仿真对比中显著超过 DP/DP3，且双向扩展消融优于 next-token 与 next-chunk
> - **reusable_ops**: [hierarchical action densification, observation-conditioned action refinement]
> - **failure_modes**: [2D long-horizon spatial reasoning, gripper over-tightness in 6-DoF pouring]
> - **open_questions**: [scaling to VLA foundation models, handling variable-length or non-dyadic horizons]

## Part I：问题与挑战

这篇论文要解决的不是“如何再造一个更大的机器人模型”，而是一个更具体也更关键的瓶颈：

**现有动作策略头的生成顺序不对。**

主流 imitation learning 操作策略大致分两类：

1. **整体生成式**：如 diffusion / VAE，一次性对整段动作序列建模。  
   - 优点：全局一致性较好。
   - 问题：训练/推理重，尤其 diffusion 需要多步采样。

2. **传统自回归式**：如 next-token / next-chunk，从左到右逐步生成。  
   - 优点：符合序列建模范式。
   - 问题：在机器人连续动作里，单向预测很难利用“未来关键动作”约束当前动作，长程依赖弱，误差会沿时间累积。

论文的判断是：**机器人动作序列的难点不只是序列长，而是“连续、精细、双向相关”**。  
比如插 peg、转笔、开抽屉这类任务，当前时刻的动作是否合理，往往取决于整段轨迹后续要到达什么姿态。只靠过去预测未来，容易得到局部合理但全局不顺的轨迹。

### 输入/输出接口

- **输入**：时刻 \(t\) 的观测 \(O_t\)，可包含
  - RGB 图像
  - 点云
  - 可选本体感觉（末端状态等）
- **输出**：固定 horizon 的未来动作序列
  - 论文中主要是 **末端执行器 TCP pose** 序列
  - 属于**原始连续动作空间**，不是离散 token

### 真正瓶颈

**What / Why：真正瓶颈是什么，为什么现在值得做？**

真正瓶颈是：  
**机器人动作的时序依赖是“跨全局 + 需高精度”的，但现有 AR 方式只给局部上下文，现有整体生成方式又成本过高。**

为什么现在做：
- 自回归范式在语言/视觉里已经显示出强大扩展性；
- 机器人领域也开始尝试 AR，但 next-token/chunk 结果明显不如 diffusion；
- 因此关键问题不是“AR 不行”，而是**动作的 AR 学习顺序还没设计对**。

### 边界条件

Dense Policy 的设定边界也很明确：

- 目标是 **behavior cloning / imitation learning**
- 输出是 **固定长度** 动作序列
- 任务是 **机器人操控**
- 重点创新在 **action head**，不是视觉 backbone

所以这不是一个通用 VLA，也不是开放长度规划器，而是一个更强的**机器人动作生成头**。

---

## Part II：方法与洞察

Dense Policy 的核心不是“更复杂的网络”，而是**换了动作展开顺序**：

> 不再按时间一步步往后生成，  
> 而是先得到覆盖整段任务的稀疏关键帧，再逐层向中间补点、细化，直到生成完整稠密动作序列。

### 方法主线

#### 1. 观测编码

论文刻意保持视觉部分通用：

- 2D 默认用 ResNet18 + GroupNorm
- 3D 默认用 sparse convolution
- 本体感觉用 MLP 编码
- 训练时随机 mask 一部分末端位姿，减少模型死记固定位置偏置

这里的意图很明确：**证明增益主要来自 action head，而不是 backbone 换得更强。**

#### 2. 从单个动作种子开始

Dense Policy 不是从第一步动作开始往后 rollout，  
而是从一个**常量初始动作向量** \(A_0=0\) 开始。

这个零种子不提供任务先验，只提供一个统一起点，让模型完全根据观测去生成轨迹结构。

#### 3. 先出稀疏关键帧，再递归稠密化

动作序列被分成多层表示：

- 上层：很稀疏的关键帧
- 下层：更密的动作点
- 最底层：完整 horizon 的动作序列

每一层做两件事：

1. **线性上采样**  
   把已有关键帧之间插入中点，形成更长但仍较粗糙的序列。

2. **观测条件细化**  
   将上采样后的动作序列与观测特征做 cross-attention，通过 encoder-only 结构修正这些中间点，使其变成下一层更准确的动作表示。

这样每层长度翻倍，经过约 `log2(T)` 层，就能得到目标长度的动作序列。

#### 4. 最终输出

最后用线性层把最细层表示投影到动作空间，用 L2 监督真实动作。

整个过程可以概括为：

**粗计划 → 中间补点 → 观测纠偏 → 更细计划**

### 核心直觉

Dense Policy 真正改动的是：

**把动作建模的因子分解方式，从“按时间顺序”改成“按分辨率顺序”。**

这带来三个因果变化：

1. **当前动作拥有双向锚点**
   - next-token 只能看左边历史
   - Dense Policy 在细化某个中间动作时，左右两侧的粗关键帧都已经存在
   - 于是当前点不再是“盲猜未来”，而是“在两端约束下补全中间”

2. **搜索空间被逐层缩小**
   - 直接预测完整稠密连续轨迹，空间太大
   - 先预测关键骨架，再补细节，问题被拆成一系列更容易的条件生成子问题

3. **避免离散 token 化误差**
   - 不走 VQ-VAE / codebook 路线
   - 对机器人高精度控制更友好，少了量化误差与额外优化负担

于是能力变化是：

**从“局部合理但全局易漂”的动作生成，变成“先有全局骨架、再补局部细节”的动作生成。**

这特别有利于：
- 长时域任务
- 接触敏感任务
- 高自由度操作
- 依赖姿态连续性的任务

### 为什么这套设计有效

**How：作者拧动了哪个关键旋钮？**

关键旋钮就是：

**双向层级扩展（bidirectional dense process）**

它改变的不是 loss，而是**信息流结构**：

- 过去的 AR：信息从左往右单向流动
- Dense Policy：信息先跨尺度建立全局骨架，再在每个尺度上用观测进行双向细化

因此变化链条可以写成：

**单向时间 rollout → 改为跨尺度双向补全 → 降低连续动作的条件不确定性 → 提升轨迹平滑性、长程一致性和任务成功率**

### 策略性 trade-off

| 设计选择 | 改变了什么瓶颈 | 收益 | 代价/边界 |
|---|---|---|---|
| 双向层级稠密化 替代 next-token | 从单向局部依赖改成跨尺度双向约束 | 长程一致性更强，学习更快 | 需要预定义 horizon 与层级结构 |
| 连续动作直接建模 替代离散 token | 去掉量化误差与 codebook 学习负担 | 更适合高精度控制 | 没有离散压缩带来的统一 token 接口 |
| encoder-only 动作头 替代 diffusion/VAE 头 | 降低训练和采样复杂度 | 推理更快、训练更稳 | 表达力在超大规模 VLA 中是否仍占优尚未验证 |
| plug-in 式 action head | 将创新集中在动作生成而非视觉 backbone | 易于复用到 2D/3D 编码器 | 论文尚未展示和大规模通用模型的系统级结合 |

---

## Part III：证据与局限

### 关键证据

#### 1. 同骨干公平对比下，动作头本身带来显著增益
这是最重要的实验信号。论文在仿真里刻意保持视觉骨干一致，只替换 action head：

- **3D**：Dense Policy 平均 **72%**，DP3 为 **53%**
- **2D**：Dense Policy 平均 **52%**，Diffusion Policy 为 **25%**

这说明提升不主要来自视觉编码，而是来自**动作生成顺序的改变**。

#### 2. 双向扩展不是“装饰”，而是关键因果因素
在 Door、Bin Picking、Shelf Place、Box Close 四个困难任务上，作者把 Dense Policy 改成：

- next-token
- next-chunk

结果显示双向版本：
- 收敛更快
- 上限更高

这类消融直接支持论文主张：  
**性能提升来自 bidirectional dense process，而不只是 Transformer 换皮。**

#### 3. 真实机器人上，优势集中体现在“长程一致性”和“更稳的动作流”
真实任务的信号也比较一致：

- **Put Bread into Pot**：3D Dense Policy **85%** vs RISE **75%**
- **Pour Balls**：平均倒入球数 **7.30/10** vs **6.85/10**；完全倒完成功率 **60%** vs **25%**
- **Flower Arrangement**：成功率 **70%** vs **50%**；平均插花数 **1.0** vs **0.6**

这些任务共同特征是：  
**不是只看单步精度，而是要看整段动作是否顺、是否能中途纠偏、是否能维持连贯姿态。**

#### 4. 效率信号支持其“轻量 AR 替代扩散头”的定位
论文还给出一个很实用的系统信号：

- 相比 DP，Dense Policy **推理接近快 10 倍**
- 相比 ACT，Dense Policy **推理速度相近，但动作头参数少于其一半**
- 训练稳定性上也优于 ACT / DP

这意味着它不是只换来更高成功率，也换来了**更实用的部署效率**。

### 1-2 个最值得记住的指标

如果只记住两个结果，我会选：

1. **仿真平均性能**：3D 72% vs 53%，2D 52% vs 25%  
2. **真实长程任务**：Flower Arrangement 70% vs 50%，平均插花数 1.0 vs 0.6

这两组结果分别对应：
- **方法本身有效**
- **长程连续操控真的更强**

### 局限性

- **Fails when**: 仅依赖 2D 表征去处理需要复杂 3D 空间推理的长程多物体任务时表现不足，典型如 Flower Arrangement；在 3D Pour Balls 中也会出现夹爪持续过紧，导致“至少倒进一个球”的指标偶尔落后于 RISE。
- **Assumes**: 依赖行为克隆与专家示范；输出是固定 horizon 的 TCP pose 连续动作；层级扩展天然更适合预定义长度、规则分辨率扩张的序列；真实实验每任务需要 50 条遥操作示范，且代码当下仍是“承诺开源”而非已发布。
- **Not designed for**: 通用语言条件 VLA、开放长度规划、超大规模 foundation policy 的稳定性验证；论文也明确承认尚未探索将 Dense Policy 扩展为更一般的 VLA 头。

### 可复用组件

这篇论文最有复用价值的，不是具体 benchmark 数字，而是下面几个操作子：

- **plug-in action head**：保留现有视觉编码器，只替换动作头
- **hierarchical densification**：先关键帧、后稠密化的动作生成顺序
- **bidirectional anchors for control**：用左右粗锚点约束中间动作
- **continuous-action refinement**：不做动作离散化，直接在连续空间逐层细化
- **proprio masking**：降低对固定姿态记忆偏置

### So what：相对前人真正跳到哪了？

相对 prior work，Dense Policy 的能力跃迁不在于“更大模型”，而在于：

- **比 diffusion/VAE 更轻、更快**
- **比 next-token/chunk 更能建模整段动作一致性**
- **在连续动作精度上不必依赖离散 token 化**

最有说服力的支撑实验是两类：

1. **同骨干替换 action head 的公平对比**
2. **bidirectional vs next-token/next-chunk 的消融**

这两类证据一起说明：  
Dense Policy 的收益主要来自**动作序列因子分解方式的变化**，而不是别的训练技巧。

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Dense_Policy_Bidirectional_Autoregressive_Learning_of_Actions.pdf]]