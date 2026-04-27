---
title: "Toward Stable World Models: Measuring and Addressing World Instability in Generative Environments"
venue: arXiv
year: 2025
tags:
  - Evaluation
  - task/video-generation
  - task/world-model-evaluation
  - diffusion
  - inverse-action-consistency
  - reference-free-metric
  - "dataset/CS:GO"
  - dataset/DMLab
  - opensource/no
core_operator: "用“动作→逆动作”闭环回访测试世界模型，并以起点/终点差异相对中途动态幅度的比值定义参考无关的世界稳定性分数。"
primary_logic: |
  世界模型是否能在回到原位置时保持同一世界 → 在可定义逆动作的环境中执行 N 次动作再执行 N 次逆动作形成闭环轨迹 → 用起点/终点差异相对中途运动幅度计算 WS score，并结合 LPIPS/MEt3R/DINO 作为距离度量 → 揭示现有扩散世界模型的场景保持缺陷并评估可行改进策略
claims:
  - "Claim 1: 在 CS:GO 与 DMLab 上，现有扩散世界模型在动作-逆动作回访后都会出现明显场景漂移，说明其 world stability 不足 [evidence: analysis]"
  - "Claim 2: 在 CS:GO 上，逆向预测微调（IRP）比单纯增加上下文长度更有效地降低 WS-LPIPS（0.7774 vs 0.8159），且不需要更长上下文推理 [evidence: comparison]"
  - "Claim 3: Refinement sampling 在两种环境和多种训练设置下都能进一步降低 WS 分数，但代价是约 2× 推理时间 [evidence: comparison]"
related_work_position:
  extends: "N/A"
  competes_with: "FVD (Unterthiner et al. 2018); MEt3R (Asim et al. 2025)"
  complementary_to: "DIAMOND (Alonso et al. 2024); Diffusion Forcing (Chen et al. 2024)"
evidence_strength: moderate
pdf_ref: paperPDFs/Evaluating_World_Models/arXiv_2025/2025_Toward_Stable_World_Models_Measuring_and_Addressing_World_Instability_in_Generative_Environments.pdf
category: Evaluation
---

# Toward Stable World Models: Measuring and Addressing World Instability in Generative Environments

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2503.08122)
> - **Summary**: 这篇论文把“世界模型绕出去再回来时能否还是同一个世界”正式定义为可测的稳定性问题，并提出动作-逆动作闭环评测与若干改进手段，证明现有扩散世界模型普遍存在场景保持性缺陷。
> - **Key Performance**: CS:GO 上 WS-LPIPS 从 0.8791（Base）降到 0.7451（Ref-IRP）；DMLab 上 WS-MEt3R 从 1.1798（Base）降到 1.0813（Ref-IRP），且 FVD 可降到 422.7。

> [!info] **Agent Summary**
> - **task_path**: 初始观察+动作序列+逆动作序列 / 生成式环境回访测试 -> 世界稳定性分数与更稳定的下一帧生成
> - **bottleneck**: 现有世界模型优化了画质与多样性，但没有显式约束“离开后再回来仍是同一场景”的闭环一致性
> - **mechanism_delta**: 把稳定性诊断从普通生成质量评测改成动作-逆动作闭环一致性测试，并用逆向预测微调与细化采样直接针对回访漂移
> - **evidence_signal**: 两个环境中 WS 指标与定性案例高度一致，且 IRP/Refinement 能稳定降低 WS 分数
> - **reusable_ops**: [动作-逆动作闭环评测, 逆向预测短微调]
> - **failure_modes**: [小物体回访时消失, 非可逆或混合动作场景难覆盖]
> - **open_questions**: [WS 提升是否稳定转化为下游 RL 收益, 无显式逆动作时如何大规模训练稳定世界模型]

## Part I：问题与挑战

这篇论文抓住的不是“世界模型画得够不够真”，而是一个更接近可交互环境本质的问题：**当智能体离开某个位置、执行一串动作、再回到原位置时，世界是否仍然保持一致**。

### 真正的问题是什么
以往扩散世界模型在视觉质量、可控性、可玩性上已经很强，但它们常把世界建模成“局部合理的下一帧生成”，而不是“长期自洽的可回访环境”。结果就是：

- 门、相框、箱子这类对象在回到原视角时会消失或位置漂移；
- 地板颜色、墙体纹理会发生无根据变化；
- 这种漂移会给 RL 训练带来观测噪声，也会破坏游戏/仿真的一致体验。

### 为什么现在要解决
因为世界模型已经开始被当作：
- RL 数据生成器；
- 可交互游戏引擎；
- 机器人/自动驾驶等低容错场景的模拟器。

在这些场景里，**“回到同一地点却看到不同世界”** 不再只是视觉瑕疵，而是训练信号污染和安全风险。

### 输入/输出接口
这篇工作的评测接口很清楚：

- **输入**：初始观察 \(x_1\)、动作序列 \(A\)、对应逆动作序列 \(A^{-1}\)
- **过程**：先执行 N 步动作，再执行 N 步逆动作回到原位置
- **输出**：初始帧与最终返回帧之间的一致性分数，即 WS score

### 边界条件
这个评测框架并不是对所有世界模型都无条件适用，它依赖几个前提：

- 环境里需要能定义**逆动作**；
- 论文实验主要限制在**单一动作类型**的闭环序列上，避免混合动作的额外随机性；
- 它评测的是**回访稳定性**，不是完整物理正确性，也不是策略学习效果本身。

---

## Part II：方法与洞察

这篇论文的价值在于：它先把“world stability”从模糊直觉变成可操作评测，再尝试几个有明确因果指向的改进旋钮。

### 评测框架：动作-逆动作闭环
核心协议非常简单但很有力：

1. 从初始状态出发执行动作序列 \(A\)
2. 再按相反顺序执行逆动作序列 \(A^{-1}\)
3. 比较起点和终点是否一致

这相当于对世界模型做一次“闭环回访测试”。  
如果模型真的保留了世界结构，那么回到起点时就应看到近似相同的场景。

### WS score：不是只看“回没回来”，还要看“中途有没有真动”
作者没有只比较首尾差异，而是引入了一个关键归一化思想：

- **Discrepancy**：起点和终点差多大
- **Dynamics**：中途状态与起点/终点差多大

最终 WS 本质上是：

**返回误差 / 中途运动幅度**

这样设计的意义很重要：  
否则一个“无论输入什么动作都几乎输出同一帧”的退化模型，也可能看起来很“稳定”。  
加入 dynamics 后，模型必须既能响应动作，又能回到原世界，才会得到好分数。并且 WS 是**reference-free** 的，不要求真实 simulator 轨迹作 GT。

### 核心直觉

过去的问题不是大家不会测“画质”，而是**没有测到闭环一致性这个真正的世界建模瓶颈**。

- **what changed**：从单向生成质量评测，改成动作-逆动作闭环一致性评测
- **which bottleneck changed**：把原来被 FVD/FID/LPIPS 掩盖的“场景保持信息瓶颈”显式暴露出来
- **what capability changed**：能区分“会生成局部合理视频”和“能维护可回访世界状态”的模型

更具体地说，这个设计之所以有效，是因为它把世界模型必须掌握的隐藏能力直接转成了可观测测试：

- 模型是否保存了不可见时刻的场景记忆；
- 模型是否理解动作与逆动作的双向对应；
- 模型是否能在长链 rollout 中抑制语义漂移。

### 论文探索的四个改进旋钮
在“测出来不稳定”之后，作者还测试了四种改善思路：

1. **Longer Context Length (LCL)**  
   给模型更多历史帧，试图缓解记忆不足。

2. **Data Augmentation (DA)**  
   在训练序列里显式加入“回放/回退”数据，让模型见过 revisit 模式。

3. **Inject Reverse Prediction (IRP)**  
   引入逆动作 embedding，并做短程微调，让模型学会“按逆动作预测回去”。

4. **Refinement Sampling**  
   先生成一帧，再加噪重采样一次，相当于 inference-time 的二次修正。

### 策略取舍表

| 策略 | 改动的因果旋钮 | 预期收益 | 代价/局限 |
|---|---|---|---|
| LCL | 增加历史可见范围 | 缓解短程记忆丢失，改善回访一致性 | 训练/推理成本上升，长程不可扩展 |
| DA | 让训练分布显式包含“出去再回来” | 降低 revisit 场景的分布外偏差 | 依赖可定义逆动作，很多真实动作不可逆 |
| IRP | 显式学习 inverse-conditioned prediction | 不靠更长 context 也能建模回退一致性 | 需要构造逆向样本并做额外微调 |
| Refinement Sampling | 在采样时重审并修正已生成帧 | 改善物体位置、纹理与局部漂移 | 推理约 2×，更像补救而非根治 |

### 这篇论文最重要的方法洞察
**更长上下文并不是唯一解，直接让模型学“反向回去”往往更对症。**

这点在 CS:GO 上尤其明显：IRP 用更小的结构改动，就比 LCL 更好地降低 WS-LPIPS，说明 world stability 的关键不只是“记得更久”，而是“学会闭环”。

---

## Part III：证据与局限

### 关键证据信号

#### 1. 诊断信号：现有 SoTA 世界模型确实不稳定
在 CS:GO 和 DMLab 上，基线模型都表现出明显 world instability：

- CS:GO 中门会消失、视角回不到原位置；
- DMLab 中相框消失、地板/墙面颜色变化。

这说明论文不是人为定义了一个“新指标”，而是在测一个**真实存在且可观察的失效模式**。

#### 2. 比较信号：IRP 比单纯加长上下文更直接
在 CS:GO 上：

- Base-LCL 的 WS-LPIPS 为 **0.8159**
- Base-IRP 的 WS-LPIPS 为 **0.7774**

而且 IRP 不要求更长上下文推理。  
这支持了一个关键结论：**针对逆向一致性的训练信号，比机械地堆上下文更有效。**

#### 3. 采样信号：Refinement sampling 是稳定有效的 inference-time patch
Refinement sampling 基本在两套环境、多种训练设置下都继续降低 WS 分数。  
它最强的信号不是“单次最优”，而是**跨设置的一致增益**。论文也通过序列长度分析表明：随着序列变长，它的帮助更明显。

### 1-2 个最值得记住的指标
- **CS:GO**：WS-LPIPS 从 **0.8791 → 0.7451**（Base → Ref-IRP）
- **DMLab**：WS-MEt3R 从 **1.1798 → 1.0813**（Base → Ref-IRP），同时 FVD 到 **422.7**

### 这篇工作的“所以呢”
能力跃迁不在于“生成更漂亮”，而在于第一次把世界模型的一个核心系统能力拆开测清楚：

- 以前：只知道模型能否生成逼真视频
- 现在：能知道模型是否能维护**可回访、可交互、可依赖**的世界

这对后续工作非常关键，因为它把改进方向从泛化的“更强 backbone / 更多数据”转成了更明确的：
- 闭环一致性建模
- 逆动作建模
- 长程记忆与回访恢复

### 局限性
- **Fails when**: 小物体、远处结构或长时间离开视野后的细节仍容易消失；长序列、撞墙场景、混合动作下仍会出现漂移与颜色变化。
- **Assumes**: 环境中存在可定义的逆动作；实验主要基于单动作类型闭环；IRP/DA 需要可构造逆向序列，LCL 需要更高算力，refinement 需要约 2× 推理时间。
- **Not designed for**: 无动作条件的一般视频生成、无法定义逆动作的复杂交互行为、直接替代下游 RL policy evaluation 或安全认证。

### 复现与可扩展性提醒
- 论文基于已有预训练世界模型（DIAMOND、Diffusion Forcing）做评测与短微调；
- IRP 的额外训练开销相对可控，但 LCL 会显著抬高训练/推理成本；
- 文中未给出明确代码链接，因此开源可复用性目前偏弱；
- 还没有把 WS 的改善直接和下游 agent learning 增益做强因果闭环。

### 可复用组件
- **动作-逆动作闭环评测协议**
- **WS 参考无关稳定性指标**
- **逆动作 embedding + 短程 IRP 微调**
- **Refinement sampling 作为即插即用推理补丁**

![[paperPDFs/Evaluating_World_Models/arXiv_2025/2025_Toward_Stable_World_Models_Measuring_and_Addressing_World_Instability_in_Generative_Environments.pdf]]