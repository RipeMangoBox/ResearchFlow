---
title: "Aux-Think: Exploring Reasoning Strategies for Data-Efficient Vision-Language Navigation"
venue: NeurIPS
year: 2025
tags:
  - Embodied_AI
  - task/vision-language-navigation
  - chain-of-thought
  - auxiliary-supervision
  - receding-horizon-planning
  - dataset/R2R-CoT-320k
  - dataset/R2R-CE
  - dataset/RxR-CE
  - opensource/full
core_operator: 将CoT从测试时显式推理链中移除，只在训练时作为辅助监督，并联合指令重构与短视野动作规划提升VLN数据效率
primary_logic: |
  历史观测+当前视角+导航指令 → 训练时通过提示切换联合学习动作预测、CoT推理和指令重构，并用receding-horizon预测强化短期规划 → 测试时跳过显式推理直接输出下一动作
claims:
  - "在仅使用 R2R-CoT-320k 训练的公平设置下，Aux-Think 在 R2R-CE val-unseen 上取得 41.3 SR / 35.8 SPL，优于 No-Think 的 35.1 / 30.2、Pre-Think 的 11.4 / 8.6 和 Post-Think 的 29.0 / 23.8 [evidence: comparison]"
  - "显式测试时推理在 VLN 中会出现 Test-time Reasoning Collapse：随着任务步数增加，Pre-Think 和 Post-Think 的成功率快速衰减，在 70+ 步任务上接近失效，而 Aux-Think 保持明显更高的长程鲁棒性 [evidence: analysis]"
  - "CoT辅助推理、指令重构和 3-step receding-horizon action planning 具有互补性；作者的消融结果显示完整配置将 R2R-CE val-unseen 的 SR 提升到 46.0，优于去除任一模块的变体 [evidence: ablation]"
related_work_position:
  extends: "NVILA (Liu et al. 2024)"
  competes_with: "Uni-NaVid (Zhang et al. 2024); NaVILA (Cheng et al. 2024)"
  complementary_to: "DAgger (Ross et al. 2011); MonoDream (Wang et al. 2025)"
evidence_strength: strong
pdf_ref: paperPDFs/Vision_Language_Navigation_VLN_Models_2025/arXiv_2025/2025_Aux_Think_Exploring_Reasoning_Strategies_for_Data_Efficient_Vision_Language_Navigation.pdf
category: Embodied_AI
---

# Aux-Think: Exploring Reasoning Strategies for Data-Efficient Vision-Language Navigation

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2505.11886), [Project](https://horizonrobotics.github.io/robot_lab/aux-think)
> - **Summary**: 本文先证明了 VLN 中“测试时显式思考”会因分布外状态而崩塌，再把 CoT 改造成训练期辅助监督，从而在不增加测试推理负担的前提下提升导航成功率与数据效率。
> - **Key Performance**: 仅用 R2R-CoT-320k 时，R2R-CE val-unseen 上 SR 41.3（No-Think 35.1，Pre-Think 11.4，Post-Think 29.0）；加入额外 1600K 数据时，R2R-CE SR 达 54.8。

> [!info] **Agent Summary**
> - **task_path**: 单目历史观测 + 当前RGB视角 + 自然语言导航指令 -> 连续环境中的下一导航动作
> - **bottleneck**: 测试时显式CoT在偏离oracle轨迹的分布外状态上容易漂移，错误推理会被动作闭环逐步放大
> - **mechanism_delta**: 把CoT从“测试时必须生成的决策中间件”改成“训练时辅助监督信号”，再叠加指令重构和3-step短视野规划
> - **evidence_signal**: 公平320k训练设置下 Aux-Think 的 SR 41.3 明显高于 No-Think 35.1，且长轨迹任务中 Pre/Post-Think 近乎崩溃而 Aux-Think 保持稳定
> - **reusable_ops**: [auxiliary-cot-supervision, receding-horizon-action-planning]
> - **failure_modes**: [单目局部视野下超过3步的前瞻预测会退化, 测试时若重新启用显式CoT会暴露TRC]
> - **open_questions**: [TRC在真实机器人和更大backbone上是否同样显著, 能否只在异常恢复阶段条件触发推理而非每步推理]

## Part I：问题与挑战

这篇工作的核心问题不是“VLN 能不能用 CoT”，而是**CoT 应该放在训练阶段还是执行阶段**。

### 1) 任务接口是什么
作者研究的是 **VLN-CE**：  
输入是
- 自然语言导航指令；
- 当前单目 RGB 观测；
- 历史观测（从历史帧中均匀采样 8 帧，包含首帧）。

输出是下一步导航动作，动作空间包括：
- 前进 25/50/75cm，
- 左/右转 15/30/45 度，
- 停止。

这是一个**连续环境、长时程、部分可观测**的决策任务，不是一次性静态问答。

### 2) 真正的瓶颈是什么
作者指出，VLN 中的真实瓶颈不是“模型不会生成理由”，而是：

**显式 reasoning 一旦进入测试时决策闭环，就会成为新的错误源。**

原因在于：
- 训练时 CoT 只见过 **oracle trajectory** 上的理想状态；
- 测试时 agent 很容易偏离理想轨迹，进入 **off-distribution states**；
- 在这些状态上，显式 CoT 更容易漂移、幻觉、误判环境；
- 一步推错后，后续观察分布继续恶化，形成 **cascading errors**。

作者将这一现象命名为 **Test-time Reasoning Collapse (TRC)**。

### 3) 为什么现在值得解决
现在的大模型 VLN 系统已经具备不错的视觉-语言对齐能力，性能提升越来越依赖：
- 更大的训练数据，
- 更强的 backbone，
- 更复杂的系统工程。

但 CoT 在静态任务里很成功，社区自然会问：**能不能把“思维链”也带进 embodied navigation？**

这篇 paper 的价值就在于：它不是默认“加推理就更好”，而是先做系统性验证，发现**对动作闭环任务，测试时显式推理可能反而有害**。这对 embodied AI 很关键，因为它直接关系到：
- 是否值得在控制回路里生成长文本；
- 如何用 reasoning 提升小数据效率，而不是只增加时延和脆弱性。

### 4) 边界条件
这篇工作的结论主要成立于以下边界内：
- 单目 RGB；
- Habitat 模拟环境；
- SFT 模式，监督来自专家轨迹；
- 重点考察 R2R-CE / RxR-CE 的 unseen split 泛化。

所以它回答的是：**在当前主流 VLN 大模型范式下，显式 CoT 对连续导航是否真的有用。**

---

## Part II：方法与洞察

Aux-Think 的设计哲学很清楚：

> **让 reasoning 改变表征，但不要让 reasoning 文本直接介入测试时动作选择。**

### 方法骨架

#### 1. 先做一件重要的基础工作：构建 R2R-CoT-320k
作者发布了首个面向 VLN 的 CoT 数据集 **R2R-CoT-320k**。  
做法是：
- 在 Habitat 中重建 step-wise 导航轨迹；
- 对每个 step 提供历史观测、当前观测、指令和 ground-truth action；
- 用 Qwen-2.5-VL-72B 为每一步生成能导向正确动作的 reasoning trace。

这一步的作用不是让模型“学会在测试时讲道理”，而是给训练提供**结构化语义监督**。

#### 2. 训练时做三件事
Aux-Think 不是单一损失，而是三任务共训：

- **主任务：Receding-Horizon Action Planning**  
  给定历史观测、当前观测和指令，预测接下来 \(n\) 步动作，但测试时只执行第一步。  
  本质上是给动作模型一点短期前瞻。

- **辅助任务 A：CoT-based Reasoning**  
  让模型从同样输入生成该步的 reasoning trace。  
  作用是把“怎么理解环境和指令”的模式压进表征里。

- **辅助任务 B：Instruction-based Reasoning**  
  给定视觉轨迹，反向重建 instruction。  
  这相当于让模型学会“从看到的路径反推语言意图”，增强轨迹-语言对齐。

作者通过 **prompt switching** 在同一 backbone 上切换任务，而不是把 action 和 CoT 强行串成同一条输出链。

#### 3. 测试时只做一件事
测试阶段只激活动作预测：
- 不生成 CoT；
- 不让 action 依赖显式 reasoning token；
- 只输出动作，并执行第一步。

因此它保留了 reasoning 的训练收益，但删掉了 reasoning 的测试风险和延迟。

### 核心直觉

**改变了什么？**  
把 CoT 从 inference path 挪到了 training path。

**哪个瓶颈被改变了？**  
从前是：
- `观测/指令 -> 生成CoT -> 依据CoT选动作`
- 动作依赖一个容易在 OOD 状态下出错的中间文本变量

现在变成：
- `观测/指令 -> 训练时用CoT塑形表示 -> 测试时直接选动作`
- CoT 不再是运行时脆弱接口，而是训练期先验

**能力上发生了什么变化？**
- 降低了测试时 hallucinated CoT 对动作的直接污染；
- 提升了数据效率，因为每条样本不只提供 action label，还提供 reasoning 与 instruction 重建信号；
- 在长轨迹上更稳，因为不会每一步都暴露给“先想一段文本再行动”的误差累积链条。

这也是作者对 Pre-Think / Post-Think 失效的因果解释：
- **Pre-Think**：动作直接依赖错误 CoT，最脆弱；
- **Post-Think**：即便先出动作，模型仍需为动作保留解释能力，隐藏状态会被扰动；
- **Aux-Think**：训练吸收 reasoning，测试不再显式调用 reasoning。

### 战略取舍

| 设计选择 | 改变的瓶颈 | 带来的能力 | 代价/权衡 |
|---|---|---|---|
| CoT 只在训练中使用 | 去掉 action 对生成文本中间态的依赖 | 长轨迹更稳、时延更低 | 测试时可解释性不如显式 CoT 直接 |
| 指令重构辅助任务 | 轨迹-语言对齐不足 | 更强语义 grounding，小数据更有效 | 增加多任务训练复杂度 |
| 3-step receding-horizon 预测 | 单步模仿缺乏短期规划 | SR / SPL 提升，动作更有前瞻性 | horizon 过长时在单目局部视野下会退化 |
| Prompt-switched 多任务共训 | action/CoT 联合输出互相干扰 | 共享 backbone 又分离目标 | prompt 设计与任务平衡需要调参 |

一句话总结：**Aux-Think 学的是 reasoning-shaped policy，而不是 reasoning-producing policy。**

---

## Part III：证据与局限

### 关键证据

- **Comparison signal：公平控制实验直接证明“测试时显式推理会伤害 VLN”。**  
  在只用 R2R-CoT-320k 训练的设置下，Aux-Think 在 R2R-CE val-unseen 上达到 **41.3 SR / 35.8 SPL**；No-Think 为 **35.1 / 30.2**；Pre-Think 只有 **11.4 / 8.6**；Post-Think 为 **29.0 / 23.8**。  
  这说明提升不是来自更多数据，而是来自**reasoning 的使用位置被改对了**。

- **Analysis signal：TRC 是系统性长程问题，不是个别样例。**  
  Fig. 6 显示，随着任务步数增加，Pre-Think 和 Post-Think 的 SR 快速下降，超过 70 步时几乎失效；Aux-Think 在长轨迹上明显更稳。  
  最强支持点在于：作者不仅说“CoT 会错”，而是证明了**错误会在 embodied 闭环里被步步放大**。

- **Efficiency signal：它不只更准，也更快。**  
  在同一公平设置里，Aux-Think 平均测试时间约 **1.25s**，而 Pre-Think / Post-Think 分别约 **30.62s / 28.97s**。  
  对真实导航系统而言，这个差距非常实际：显式长文本推理既拖慢控制回路，也增加不稳定源。

- **Benchmark signal：能力提升主要体现在“到达率”，并且更省数据。**  
  R2R-CE val-unseen 上，Aux-Think 用 **1600K** 额外数据达到 **54.8 SR**，高于 Uni-NaVid 的 **47.0 SR（5570K）**，也略高于 NaVILA 的 **54.0 SR（2770K）**。  
  RxR-CE val-unseen 上，Aux-Think 达到 **52.2 SR**，也超过 Uni-NaVid 的 **48.7** 和 NaVILA 的 **49.3**。  
  这说明它的优势更像是**样本效率和长程到达能力**，而不是纯堆数据。

- **Ablation signal：三个组件确实互补。**  
  仅加 CoT reasoning 就能把 SR 从 35.1 提到 41.3；再加 instruction reconstruction 到 44.2；完整加上 receding-horizon planning 后到 **46.0**。  
  另外，预测 **3 步** 最优，说明短期前瞻有效，但过长 horizon 会被局部观测限制反噬。

### 局限性

- **Fails when**: 单目 RGB 的局部感受野不足以支撑更长 horizon 预测时，4-5 步 receding-horizon 会退化；对特别长、需要更强全局恢复或最短路径效率的任务，SR 提升不一定同步转化为更好的 NE/SPL。
- **Assumes**: 依赖 oracle 轨迹上的 CoT 监督和由 Qwen-2.5-VL-72B 生成的 R2R-CoT-320k；依赖 NVILA-lite 8B 作为 backbone 并采用 SFT；实验主要在 Habitat 模拟器的 R2R-CE / RxR-CE、单目 RGB 设定下完成；训练资源约为 8×NVIDIA H20、约 60 小时。
- **Not designed for**: 真实机器人动态避障、在线建图、多传感器融合（depth / panorama / localization）、或者必须在部署时输出可审计显式推理链的场景。

### 可复用组件

- **Auxiliary CoT supervision**：适合任何“动作闭环中不想暴露显式推理风险”的 embodied 任务。
- **Prompt-switched multi-task co-training**：在同一 backbone 上分离动作、CoT、指令重构三类目标，减少 token 级互扰。
- **3-step receding-horizon action planning**：适合局部观测控制任务，用小前瞻换更稳的执行。

### 最后一句判断

这篇 paper 的真正贡献，不只是提出了一个 VLN trick，而是给出了一个更一般的经验法则：

**在长时程、部分可观测、动作闭环的 embodied 任务里，reasoning 更适合作为训练期表征塑形信号，而不是测试期显式中间变量。**

![[paperPDFs/Vision_Language_Navigation_VLN_Models_2025/arXiv_2025/2025_Aux_Think_Exploring_Reasoning_Strategies_for_Data_Efficient_Vision_Language_Navigation.pdf]]