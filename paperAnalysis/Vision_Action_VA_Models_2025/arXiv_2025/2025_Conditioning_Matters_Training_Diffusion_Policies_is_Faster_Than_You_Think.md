---
title: "Conditioning Matters: Training Diffusion Policies is Faster Than You Think"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/robot-manipulation
  - task/vision-language-action
  - diffusion
  - flow-matching
  - conditional-source-distribution
  - dataset/LIBERO
  - dataset/MetaWorld
  - opensource/promised
core_operator: 将无条件高斯源分布替换为由条件语义决定均值的条件高斯先验，阻止扩散策略训练退化为拟合边缘动作分布。
primary_logic: |
  多视角视觉/语言条件与机器人状态 → 将条件编码压缩到动作空间并构造条件相关的源分布 q(x0|c) → 在条件流匹配下学习对条件敏感的动作向量场 → 通过 ODE 生成动作序列
claims:
  - "当训练采用独立采样 q(x1,c)q(x0) 且模型难以区分条件时，条件梯度会收缩并推动策略退化为条件无关的平均动作估计；改用 q(x0|c) 可打破这一塌缩机制 [evidence: theoretical]"
  - "在 LIBERO 上，40M 参数扩散策略加入 Cocos 后平均成功率从 86.5% 提升到 94.8%，并在约 30K gradient steps 达到与 π0 相当的表现，收敛约快 2.14× [evidence: comparison]"
  - "Cocos 训练出的策略在条件注入前后表现出更低的 hidden-state cosine similarity 和更大的 norm scale change，且与更高任务成功率一致，说明增益来自更强的条件利用而非仅仅先验注入 [evidence: analysis]"
related_work_position:
  extends: "Flow Matching for Generative Modeling (Lipman et al. 2023)"
  competes_with: "π0 (Black et al. 2024); OpenVLA-OFT (Kim et al. 2025)"
  complementary_to: "FAST (Pertsch et al. 2025); SpatialVLA (Qu et al. 2025)"
evidence_strength: strong
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Conditioning_Matters_Training_Diffusion_Policies_is_Faster_Than_You_Think.pdf
category: Embodied_AI
---

# Conditioning Matters: Training Diffusion Policies is Faster Than You Think

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2505.11123)
> - **Summary**: 这篇工作指出扩散式 VLA 训练慢的根因不是动作生成本身，而是条件流匹配在“条件难分”时会塌缩为无条件动作建模；Cocos 通过把源分布改成条件相关先验，强迫策略真正使用视觉和语言条件。
> - **Key Performance**: LIBERO 平均成功率 94.8%（vs. 86.5%）；约 30K steps 达到 π0 级别表现，收敛约提升 2.14×

> [!info] **Agent Summary**
> - **task_path**: 多视角图像 + 语言指令 + 机器人状态 -> 动作 chunk / 未来动作序列
> - **bottleneck**: 条件流匹配在条件难区分时退化为拟合边缘动作分布，策略会主动忽略条件
> - **mechanism_delta**: 把标准高斯源分布替换为条件相关高斯 q(x0|c)，让每个条件对应不同的扩散起点
> - **evidence_signal**: 跨 LIBERO、MetaWorld 与两种真实机器人平台的稳定提升，并有 hidden-state 条件敏感性分析支持
> - **reusable_ops**: [condition-dependent source prior, condition autoencoder to action space]
> - **failure_modes**: [条件编码本身语义不足时仍会混淆, 条件先验过窄（β=0.1）会明显拖累训练]
> - **open_questions**: [更强的可学习 q(x0|c) 是否能继续提升, 能否扩展到大规模 VLA 预训练而非多任务模仿学习]

## Part I：问题与挑战

这篇 paper 关注的是一个很“反直觉”的问题：**机器人动作序列本身维度并不高，但扩散策略/VLA 训练仍然很慢、很吃资源**。作者的判断是，真正的瓶颈不在“生成动作”本身，而在**模型是否真的把条件信息吃进去**。

### 任务接口
- **输入**：当前/历史观测中的多视角图像、语言指令、机器人关节状态
- **输出**：未来一段动作序列（action chunk）
- **训练范式**：多任务 imitation learning + conditional flow matching / diffusion policy

### 真正的难点
标准做法里，扩散/流匹配从一个**与条件无关**的高斯源分布开始采样，再把视觉和语言条件通过 cross-attention 等方式注入策略网络。问题是：

1. 如果条件本身难解释、难区分；
2. 或者视觉编码虽然强，但还不足以让策略早期就稳定地区分任务；
3. 那训练目标会逐渐退化成“拟合所有动作的平均分布”。

作者把这件事叫做 **loss collapse**。

它的直接后果不是 loss 爆炸，而是更隐蔽的：
- 模型看起来在收敛；
- 但实际上学的是“平均动作”；
- 部署时会**忽略语言、忽略第三视角、忽略当前场景差异**。

论文里举的直观例子是：面对“向左移动”和“向右移动”两种语言条件，塌缩后的策略会更像按数据频率输出一个平均方向。

### 为什么现在值得解决
因为当前很多提升 VLA 的路线都在往大模型、强 encoder、强 tokenizer 上堆：
- 更强视觉编码器
- 更紧凑动作 token
- 更大规模预训练

这些都有效，但也更贵。本文的切入点是：**也许不是 scale 不够，而是条件训练目标本身就让模型容易忽略条件**。如果把这个机制改对，40M 级别的小模型也可能逼近大规模预训练 VLA。

### 边界条件
这篇工作解决的是一个比较明确的问题边界：
- 面向**去噪式/流匹配式生成策略**
- 主要是**多任务模仿学习**
- 条件来自冻结的视觉/语言 encoder
- 目标是**训练效率 + 条件利用能力**

它**不直接解决**：
- 数据量不足
- 机器人硬件噪声
- 纯自回归 VLA 的训练问题
- 开放世界大规模预训练泛化

---

## Part II：方法与洞察

### 方法概览

Cocos 的改动非常小，但抓得很准：

**把标准高斯源分布 `q(x0)`，改成条件相关的高斯 `q(x0|c)`。**

具体做法是：
- 先用条件编码器得到条件表征 `E(c)`；
- 再通过一个小型 autoencoder 的 encoder `Fϕ`，把条件表征压到动作空间维度；
- 用这个向量作为高斯均值，构造：
  - 均值：`α Fϕ(E(c))`
  - 方差：固定 `β² I`

于是扩散/流匹配不再从“完全无语义的噪声”开始，而是从**围绕该条件语义的起点**开始。

训练与推理几乎都不改骨架：
- **训练时**：把 `x0 ~ N(0,I)` 改成 `x0 ~ q(x0|c)`
- **推理时**：同样从 `q(x0|c)` 采样，再解 ODE 生成动作

作者还给了两种训练 source encoder 的方案：
1. **两阶段**：先训练条件 autoencoder，再冻结后训练 policy
2. **联合训练 + EMA**：用 EMA target encoder 稳定在线训练

### 核心直觉

#### 1）到底改了什么
改的不是 backbone，不是 loss 形式的大框架，而是**扩散起点的分布结构**：

- 旧：`x0` 与条件 `c` 独立
- 新：`x0` 由条件 `c` 决定

#### 2）它改变了哪个瓶颈
它改变的是**优化过程中条件是否会被“积分掉”**这个瓶颈。

标准做法中，如果网络暂时分不清不同条件，那么不同条件带来的梯度会越来越像，优化就自然滑向一个**与条件无关的共享方向**，最后只学会边缘动作分布。

Cocos 的关键因果点是：

- 即使网络还没完全学会解释条件；
- 由于每个条件的 source distribution 已经不同；
- 所以不同条件对应的训练 measure 也不同；
- 梯度不再容易收缩成同一个“平均更新方向”。

换句话说，**条件不再只是“后面注入一点信息”**，而是从生成起点开始就进入了优化几何。

#### 3）能力上发生了什么变化
这会带来两个直接变化：

- **更快收敛**：因为模型不用先学会“从纯噪声中再费力区分条件”
- **更强条件利用**：网络 hidden states 在注入条件前后变化更大，说明视觉/语言真的影响了动作生成

所以这不是简单的“给模型更好的 initialization”，而是**把条件从可选信息变成不可回避的训练约束**。

### 为什么这个设计有效
从系统层面看，Cocos 的有效性来自三点：

1. **把条件信息前置到 source distribution**
   - 条件不再只通过 cross-attention 进网络
   - 而是直接改变样本从哪里开始流动

2. **防止早期训练进入坏局部最优**
   - baseline 一旦早期开始忽略条件，目标本身会进一步鼓励忽略条件
   - Cocos 相当于切断这条正反馈链

3. **代价极低**
   - 不改 RDT 主体
   - 不要求更大模型
   - 不要求更复杂 diffusion solver
   - 只多一个很小的条件投影/自编码模块

### 策略性权衡

| 设计选择 | 带来的好处 | 代价/风险 |
|---|---|---|
| 条件相关源分布 `q(x0|c)` | 阻止 loss collapse，提升收敛速度与条件敏感性 | 需要额外的条件投影模块 |
| 固定方差高斯先验 | 实现简单、稳定、易兼容各种 flow/diffusion policy | 方差太小会过度集中，反而损伤训练 |
| 两阶段训练 source encoder | 最稳，容易复现 | pipeline 更长，训练流程更碎 |
| EMA 联合训练 | 更实用，接近端到端 | 仍依赖 target encoder 稳定更新 |
| 小模型 + 目标级修正 | 以较低参数量逼近大模型表现 | 不能替代大规模数据与预训练本身 |

---

## Part III：证据与局限

### 关键证据

#### 1）比较信号：训练更快，不只是最终分数更高
最核心结果来自 LIBERO：

- baseline DP-DINOv2：**86.5%**
- Cocos：**94.8%**

而且不是靠更长训练换来的：
- 约 **30K gradient steps** 就达到与 **π0** 相当的表现
- 相比 vanilla DP，收敛约 **2.14× 更快**

这直接支持论文主张：**conditioning 方式本身就是训练效率瓶颈**。

#### 2）跨数据集信号：收益不只出现在一个 benchmark
在 MetaWorld 30 任务上：
- w/o Cocos：**59.5%**
- w/ Cocos：**74.8%**

而且很多任务的增益很大，比如 door-lock、stick-push、sweep 等。  
这说明它不是只对 LIBERO 这种特定设置有效。

但也要注意，并非所有任务都单调提升：
- 某些困难任务仍然很低
- 个别任务甚至会下降（如 disassemble）

所以 Cocos 更像是**把“条件没吃进去”这个主要瓶颈拆掉**，但不是万能解法。

#### 3）分析信号：它真的让网络更依赖条件
论文没有只报成功率，还看了策略网络内部表示：

- 条件注入前后 hidden state 的 **cosine similarity 更低**
- **norm shift 更大**
- 并且这两者与成功率正相关

这说明 Cocos 的效果不是“把先验塞进 source 里就结束了”，而是**迫使 policy hidden states 真正对条件发生响应**。

#### 4）案例信号：减少了典型的条件误解
在真实机器人和 LIBERO case study 里，baseline 常见失败是：

- 过度依赖 wrist camera，忽略第三视角
- 只跟着语言走，不检查场景是否已经满足任务
- 空间方位词理解错误
- 丢失目标后无法利用外部视角恢复

Cocos 在这些地方更稳，符合其理论主张：**它提升的是 condition grounding，而不是单纯运动平滑性**。

### 1-2 个最该记住的指标
- **LIBERO 平均成功率：94.8% vs 86.5%**
- **收敛速度：约快 2.14×**

### 局限性
- **Fails when**: 任务难点主要来自精细接触、复杂动力学或长程规划而非条件辨识时，Cocos 提升有限；在 MetaWorld 的 handle-pull-side、soccer、shelf-place 等任务上，成功率仍然偏低；另外当条件先验过于集中（β=0.1）时性能会明显恶化。
- **Assumes**: 依赖冻结的 DINOv2/T5 条件编码器已经提供足够可压缩的语义；依赖多任务 imitation learning 数据与语言标注；需要额外的条件 autoencoder/EMA 目标网络；实验训练资源为 4× RTX 4090；代码在文中仅说明“将开源”，尚未实质公开。
- **Not designed for**: 纯自回归 VLA、非生成式控制器、RL fine-tuning 场景，以及大规模开放域 VLA 预训练泛化问题；它也不解决机器人硬件抖动、感知传感器滞后这类平台级问题。

### 可复用组件
- **condition-dependent source prior**：任何 conditional diffusion / flow policy 都可考虑替换 `q(x0)` 为 `q(x0|c)`
- **条件到动作空间的轻量投影器**：把高维视觉/语言条件压成 source anchor
- **EMA target encoder**：联合训练条件先验时的稳定器
- **condition sensitivity diagnostics**：用 hidden-state similarity / norm shift 检查模型是否真的在用条件

**一句话总结**：这篇 paper 的价值不在于又造了一个更大的 VLA，而在于指出并修正了一个更基础的训练病灶——**如果 source distribution 与条件解耦，扩散策略会在优化上自然滑向“忽略条件”；把条件写进 source，才能让模型更早、更稳地学会看图听话做动作。**

## Local PDF reference
![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Conditioning_Matters_Training_Diffusion_Policies_is_Faster_Than_You_Think.pdf]]