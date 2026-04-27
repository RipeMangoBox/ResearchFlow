---
title: "CCDP: Composition of Conditional Diffusion Policies with Guided Sampling"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/robot-manipulation
  - task/failure-recovery
  - diffusion
  - product-of-experts
  - guided-sampling
  - dataset/DoorOpening
  - dataset/ButtonPressing
  - dataset/ObjectManipulation
  - dataset/ObjectPacking
  - dataset/Bartender
  - opensource/no
core_operator: 将扩散策略拆成动作先验、状态、历史与失败特征多个专家，并在采样时用失败专家对已失败区域施加负引导。
primary_logic: |
  成功示范 + 当前状态/短历史 + 失败检测得到的失败特征集合
  → 训练动作/状态/历史/失败四类扩散专家，并用成功示范合成“近状态-异动作”的恢复样本
  → 通过 PoE 式引导采样输出既贴合当前状态又避开已失败区域的下一步动作
claims:
  - "当 ws=0、所有 failure weight 为 0、wh=1 且将当前状态并入历史时，CCDP 退化为标准 Diffusion Policy，说明它是对 DP 的严格扩展而非另起炉灶的控制器 [evidence: theoretical]"
  - "在 100 个随机测试场景中，CCDP 在五个任务上都优于标准 Diffusion Policy，例如 Door Opening 99% vs 76%，Object Packing 94% vs 10% [evidence: comparison]"
  - "在 Button Pressing 任务中，CCDP 在不做手工区域离散化的情况下达到 96% 成功率，高于分区+重采样基线 DP* 的 86% [evidence: comparison]"
related_work_position:
  extends: "Diffusion Policy (Chi et al. 2023)"
  competes_with: "Diffusion Policy (Chi et al. 2023); Implicit Behavioral Cloning (Florence et al. 2022)"
  complementary_to: "Recover (Cornelio and Diab 2024); Bottom-up Skill Discovery (Zhu et al. 2022)"
evidence_strength: moderate
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_CCDP_Composition_of_Conditional_Diffusion_Policies_with_Guided_Sampling.pdf
category: Embodied_AI
---

# CCDP: Composition of Conditional Diffusion Policies with Guided Sampling

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2503.15386), [Project](https://hri-eu.github.io/ccdp/)
> - **Summary**: 这篇工作把扩散策略从“失败后继续盲目重采样”改成“失败后在线排斥已证伪动作区域”，只用成功示范就能形成一个可恢复的低层机器人策略。
> - **Key Performance**: 5 个仿真任务全部优于标准 Diffusion Policy；例如 Door Opening 99% vs 76%，Button Pressing 96% vs 73%。

> [!info] **Agent Summary**
> - **task_path**: 成功示范 + 当前状态/短历史 + 已检测失败集合 -> 下一步机器人动作/短轨迹
> - **bottleneck**: 标准扩散策略在失败后仍从几乎同一条件分布采样，无法显式避开已证伪动作区域；若把失败全塞进长历史，又会导致可变长度记忆与组合爆炸
> - **mechanism_delta**: 将单一扩散策略拆成状态/历史/失败特征专家，并在推理时用 failure expert 对失败邻域施加负引导
> - **evidence_signal**: 5 个任务对比中 CCDP 全面超过标准 DP，且在 Button Pressing 上 96% > DP* 86%
> - **reusable_ops**: [PoE式扩散专家组合, 仅用成功示范合成恢复数据]
> - **failure_modes**: [失败根因非静态时负引导可能失效, 手工定义的 failure feature z 不充分时会错误排斥]
> - **open_questions**: [如何自动学习 z(·), 如何稳定自适应设置各专家权重]

## Part I：问题与挑战

这篇论文要解决的，不是“扩散策略能不能学到多模态动作”，而是更实际的一步：

**机器人第一次采样失败后，第二次该怎么更聪明地采？**

标准 Diffusion Policy 已经能从成功示范里学出多峰动作分布，但它的默认做法是：**失败了就再采一次**。问题在于，新的样本往往还会落回刚刚失败过的区域。对门把手方向未知、按钮位置未知、物体质量未知这类任务，这会非常低效。

### 真正瓶颈

真正的瓶颈是两层：

1. **失败信息没有进入下一轮采样分布**
   - 标准 DP 只看当前状态/历史，不会显式把“这个动作区域已经试过且失败”编码成排斥约束。

2. **若想靠长历史记住失败，又会带来学习与推理爆炸**
   - 要恢复，就得记住多次失败。
   - 但失败次数是可变的，直接把所有失败塞进 history，会让输入维度、数据覆盖和训练难度一起爆炸。

### 为什么现在值得做

因为扩散策略已经成为一类很强的低层 imitation learning 控制器，但现实机器人任务里常见失败恢复方案往往依赖：

- 仿真器或环境模型
- 高层 planner / foundation model
- 失败示范或额外探索
- 分层决策结构

这些条件在真实部署里常常不具备。作者的目标很明确：**只靠成功示范，在低层控制器内部完成 failure-aware sampling。**

### 输入 / 输出接口

- **输入**：
  - 成功示范集 \(D\)
  - 当前状态 \(x_t\)
  - 短历史 \(h_t^H\)
  - 失败检测系统给出的失败特征集合 \(z^f_{1:N}\)

- **输出**：
  - 下一步动作或短时间动作序列 \(a_t\)

### 边界条件

这篇方法成立有几个前提：

- 失败后，**示范分布里确实存在可替代的恢复动作**
- 失败根因在短时间内**近似静态**
  - 例如门开方向、按钮位置不会因为试一次就改变
- 有一个外部**failure detector**
- 训练时不依赖显式物理模型，但实验中的示范数据仍是在仿真里由 planner 采集的

---

## Part II：方法与洞察

### 1）把“一个大条件分布”拆成多个可组合专家

作者的核心做法是：不要直接学一个巨大的  
“动作 | 状态 | 历史 | 所有失败”的分布，  
而是把它拆成几个可组合的 diffusion experts：

- **动作先验 expert**
  - 保证采样还在示范动作分布附近，不会完全乱飞

- **状态 expert**
  - 让动作匹配当前机器人/环境状态
  - 相当于“只看当前状态”的低历史策略

- **历史 expert**
  - 保持时序连续和平滑

- **失败 expert**
  - 根据每一次失败的特征，推动采样远离对应失败区域

这本质上是一个 **product-of-experts / compositional diffusion** 思路：  
不是训练一个大而全的恢复策略，而是训练几个简单模块，推理时按需组合。

### 2）为什么要把 state 和 history 分开

这是文中一个很关键但容易被忽略的点。

如果 history 和 state 完全绑在一起，那么一旦过去轨迹里包含失败动作，模型就可能被“坏历史”拖回原来的失败模态。作者把它拆开后：

- **state expert** 提供当前可行性
- **history expert** 提供平滑性
- 出现失败时，可以适当降低 history 的作用，减少“沿着错误惯性继续走”的风险

所以，这不是简单模块化，而是在改变控制器里的**因果权衡**：
从“强连续性”转向“失败后可转向”。

### 3）没有失败示范，怎么学 recovery expert

这是方法最巧妙的部分。

作者没有真实失败数据，于是从成功示范里**合成恢复训练集**：

- 在示范中采样状态附近的局部状态
- 再从动作分布中采样多个候选动作
- 如果某个候选动作与失败动作在 failure feature 空间里“足够不同”，但状态又“足够接近”，就把它当作一个 recovery 样本

直觉上，这等于说：

> “如果系统状态差不多，而上次那类动作刚失败了，那么这次就应该试一个在关键特征上明显不同的动作。”

这样，failure expert 学到的不是“失败长什么样”，而是：

**给定一个失败特征，哪些动作更像恢复动作。**

而且作者只需要训练**一个** \(p(a|z_f)\) 模块；推理时遇到多个失败，就把同一个 failure expert 重复组合多次。  
这把“可变失败次数”从训练难题，变成了推理时的模块堆叠问题。

### 核心直觉

- **What changed**：从“固定条件分布里反复重采样”变成“依据失败集合在线重塑采样分布”。
- **Which bottleneck changed**：把原本依赖长历史的失败记忆，压缩成可组合的 failure features；把恢复问题从高层离散规划，转成低层连续采样约束。
- **What capability changed**：机器人能在门方向未知、按钮位置未知、操作模态未知时，逐步排除错误模态，切换到示范中其他少见但有效的动作。

为什么这在因果上有效？

1. **负引导只排除已证伪区域，不会像硬分区那样过度限制搜索空间**
   - 所以还能保留示范里的软偏好

2. **单失败模块可重复组合**
   - 避免为“失败1+失败2+失败3”的每种组合单独学模型

3. **恢复数据来自成功示范附近，而不是额外探索**
   - 所以方法能在“只有成功示范”的现实设定下落地

### 策略权衡

| 设计选择 | 解决的瓶颈 | 带来的能力 | 代价 / 风险 |
|---|---|---|---|
| 用多个 diffusion experts 代替单一大模型 | 长历史与可变失败数导致的建模爆炸 | 支持模块化、可变数量失败组合 | 需要额外权重设计，组合不当会不稳定 |
| 分离 state expert 与 history expert | 坏历史会把策略拉回失败轨迹 | 失败后更容易“转向” | history 权重过低会损失平滑性 |
| 用 failure expert 做负引导 | 失败信息无法进入下次采样 | 显式避开已试错区域 | 若 failure feature 不准，会排斥错区域 |
| 从成功示范合成 recovery 数据 | 没有失败示范 | 不需要额外探索或模拟训练 | 强依赖“恢复动作已在示范支持内” |

---

## Part III：证据与局限

### 关键证据信号

**1. 比较信号：对标准 Diffusion Policy 的提升是稳定的。**  
在 5 个任务上，CCDP 都优于标准 DP。最有代表性的两个点：

- **Door Opening**：99% vs 76%
- **Object Packing**：94% vs 10%

这说明主增益确实来自“失败感知采样”，而不是某个单任务技巧。

**2. 与分区+重采样基线 DP\* 相比，CCDP 的优势在于不需要手工划区，且更能保留软偏好。**

- **Button Pressing**：CCDP 96%，DP* 86%
  - 说明在连续搜索空间里，负引导比手工离散+区域剔除更灵活
- **Object Packing 的隐式目标**：CCDP 73%，DP* 48%
  - 说明 CCDP 更能保留“示范频率里隐含的偏好”，而不是只做硬切换

**3. 但 CCDP 不是全能优于所有基线。**  
在 **Object Manipulation** 上：

- 成功率：CCDP 70%，DP* 72%
- 隐式顺序偏好：CCDP 66%，标准 DP 88%

这说明当任务天然更适合离散 primitive 分区时，显式分区可能仍有优势；同时，failure-aware recovery 会在某些场景里牺牲部分示范里的软偏好。

### 证据强度怎么判断

我会把这篇论文的证据强度定为 **moderate**：

- 优点：
  - 有 5 个任务的系统比较
  - 不只看成功率，还看隐式目标保持
  - 有与标准 DP 和增强基线的对比

- 不足：
  - 主要是自建仿真任务
  - 缺少系统化 ablation（如权重、feature 选择、failure 数量增长）
  - 没有真实机器人泛化证据
  - failure detector 不是本文核心内容，但实际性能依赖它

### 局限性

- **Fails when**: 失败根因会随时间变化、同一动作不再必然失败，或者系统状态在一次失败后发生了显著改变；此时“远离上次失败动作”不再等价于有效恢复。
- **Assumes**: 有可靠的 failure detector；恢复动作存在于成功示范支持中；failure feature \(z(\cdot)\)、阈值和组合权重需要人工设计；实验示范由仿真中知道真实参数的 planner 生成。
- **Not designed for**: 高层语义重规划、超出示范分布的新技能发现、强非平稳环境、需要主动试错探索的新恢复策略学习。

还要特别指出两个会影响复现与扩展的现实依赖：

1. **外部 failure detector 是必要组件**
2. **论文给出项目页，但未明确提供完整代码/数据，且任务为定制仿真环境**

### 可复用组件

这篇论文最值得迁移的不是某个具体任务，而是三个操作符：

1. **PoE 式扩散专家组合**
   - 可把状态、历史、约束、失败记忆拆成独立 expert，再在采样时组合

2. **成功示范驱动的恢复数据合成**
   - 在没有失败数据时，仍可构造 recovery supervision

3. **可变长度失败记忆的模块化注入**
   - 用重复的 failure expert 处理任意数量失败，而不是扩展 history 长度

---

## Local PDF reference

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_CCDP_Composition_of_Conditional_Diffusion_Policies_with_Guided_Sampling.pdf]]