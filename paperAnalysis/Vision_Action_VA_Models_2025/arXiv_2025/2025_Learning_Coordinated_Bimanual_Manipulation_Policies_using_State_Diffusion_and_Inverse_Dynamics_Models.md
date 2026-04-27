---
title: "Learning Coordinated Bimanual Manipulation Policies using State Diffusion and Inverse Dynamics Models"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/bimanual-manipulation
  - task/robot-manipulation
  - diffusion
  - inverse-dynamics
  - state-prediction
  - dataset/Push-L
  - "dataset/Franka Kitchen"
  - "dataset/Block Push"
  - repr/point-cloud
  - repr/keypoints
  - opensource/no
core_operator: 先用扩散模型预测未来对象/场景状态轨迹，再用逆动力学把状态过渡翻译成双臂动作，将任务级场景演化与机器人执行解耦。
primary_logic: |
  历史状态/观测序列 → 扩散模型生成未来对象/场景状态轨迹 →
  逆动力学结合历史与预测未来状态输出动作 →
  在多模态目标、接触丰富和双臂协调场景中提升成功率与稳定性
claims:
  - "在 Push-L 仿真任务上，完整模型达到 79.3% 成功率，高于 IDP 的 50.2% 和去掉逆动力学版本的 61.3%，说明显式逆动力学对协调接触控制至关重要 [evidence: ablation]"
  - "在 Franka Kitchen 位置控制中，该方法将 5 个子任务完成率提升到 29.3%，而 DP 与 IDP 均仅为 3.0%，且训练演示每条轨迹只覆盖 4 个子任务，显示出超出演示组合的行为合成能力 [evidence: comparison]"
  - "在真实双臂任务中，该方法显著优于改进版 diffusion baseline，例如 cluttered shelf picking 达到 14/15 成功而基线为 0/15，零样本 sim-to-real Push-L 达到 9/12 而基线为 2/12 [evidence: comparison]"
related_work_position:
  extends: "Diffusion Policy (Chi et al. 2023)"
  competes_with: "Diffusion Policy (Chi et al. 2023); Implicit Behavioral Cloning (Florence et al. 2021)"
  complementary_to: "Gello (Wu et al. 2023); Universal Manipulation Interface (Chi et al. 2024)"
evidence_strength: strong
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Learning_Coordinated_Bimanual_Manipulation_Policies_using_State_Diffusion_and_Inverse_Dynamics_Models.pdf
category: Embodied_AI
---

# Learning Coordinated Bimanual Manipulation Policies using State Diffusion and Inverse Dynamics Models

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2503.23271)
> - **Summary**: 这篇工作把双臂模仿学习从“直接学动作”改成“先预测物体未来怎么动，再求能实现该状态变化的动作”，从而更好地处理双臂协同、可变形物体和多目标配置。
> - **Key Performance**: Push-L 仿真成功率 79.3% vs IDP 50.2%；真实 cluttered shelf picking 14/15 vs 0/15，零样本 sim-to-real Push-L 9/12 vs 2/12。

> [!info] **Agent Summary**
> - **task_path**: 历史状态/关键点/点云 -> 未来对象状态轨迹 -> 单臂或双臂控制动作
> - **bottleneck**: 直接 state-to-action 模仿只约束“机器人怎么动”，却没有充分约束“物体是否被正确协同地移动”，导致双臂协调失败常被动作损失掩盖
> - **mechanism_delta**: 把策略拆成 state diffusion 的任务级未来状态建模 + inverse dynamics 的 embodiment 级动作求解
> - **evidence_signal**: Push-L 消融中移除 inverse dynamics 使成功率从 79.3% 降到 61.3%，且真实三项双臂任务全面优于 diffusion baseline
> - **reusable_ops**: [future-state diffusion, inverse-dynamics decoupling]
> - **failure_modes**: [state-estimation noise-or-latency hurts control, large state spaces increase training cost]
> - **open_questions**: [can it scale to raw-image end-to-end bimanual control, can real-world demonstration demand be reduced]

## Part I：问题与挑战

这篇论文解决的不是一般的模仿学习，而是**接触丰富、目标多模态、需要双臂协作的 manipulation**。其核心场景包括：

- 两只机械臂共同搬运或稳定物体；
- 可变形物体，如枕头、衣物；
- 多物体交互，如水果成组搬运、拥挤货架中取目标物；
- 同一个任务存在多种成功路径和多种成功终态。

### 真正的瓶颈是什么？

作者强调的真实瓶颈是：**end-to-end 的 state-to-action 映射把“动作相似”当成学习目标，但双臂任务真正关心的是“物体状态是否被正确推进”**。

这会带来两个问题：

1. **协调失败难以被惩罚**  
   例如双臂搬运时轨迹看起来差不多，但物体掉了。对动作克隆来说，这种失败未必产生很大的 action loss；但对状态预测来说，物体状态会明显偏离，错误更容易暴露。

2. **多模态目标与多模态动作被混在一起学**  
   同一个成功状态可由多种动作实现，尤其在双臂协作和接触任务中更明显。直接学动作会把“任务层未来应该是什么样”和“机器人具体该怎么发力/接触”混成一个问题，学习难度大、泛化差。

### 输入/输出接口与边界条件

- **输入**：一段历史状态/观测序列。  
  仿真中主要是低维 object state + robot state；真实世界里是关键点或点云，再压缩为低维状态。
- **输出**：机器人动作，支持位置控制和速度控制，也支持双臂联合动作。
- **任务边界**：论文并不是直接从原始 RGB 端到端输出双臂动作；它依赖可用的状态表征或额外训练的状态编码器。

### 为什么现在值得做？

因为扩散模型已经很擅长建模**多峰分布的序列预测**，很适合表达“未来可能怎么演化”；而机器人模仿学习又迫切需要一种**不依赖 reward 标注、但比纯行为克隆更懂任务结果**的方式。作者正是把这两点结合起来，用在双臂操控上。

---

## Part II：方法与洞察

### 方法骨架

整套系统分成两层：

1. **State Prediction Diffusion Model**  
   给定过去若干步状态，预测未来一段时间内的世界状态轨迹。  
   这里的“世界状态”重点是物体/场景怎么变化，而不是只看机器人关节怎么动。

2. **Inverse Dynamics Model**  
   输入历史状态 + 预测未来状态，输出当前动作。  
   它不再只看相邻两帧，而是看一个时间窗口，从而更好地理解接触、惯性和协同动作。

3. **联合训练，部署时串联执行**  
   训练时同时优化“未来状态预测”和“动作求解”；  
   推理时先 rollout 出未来状态，再根据这些未来状态反推出动作。

### 核心直觉

**改了什么？**  
从“直接模仿动作”改成“先想象未来状态，再解动作”。

**改变了哪个瓶颈？**  
把原来纠缠在一起的两个不确定性拆开了：

- **任务层不确定性**：未来物体/场景会往哪里演化？  
  由 diffusion 处理，适合多模态目标。
- **执行层不确定性**：为了实现该状态变化，机器人该怎么发力、接触、协同？  
  由 inverse dynamics 处理，适合 embodiment-specific 动作生成。

**能力为什么会变强？**  
因为监督信号从“机器人做得像不像示范”转成了“物体最终有没有按任务要求演化”。  
这在双臂任务中特别关键：双手轨迹看似合理，但只要物体没被托住、没被稳定推进，状态误差就会很大，模型会被明确惩罚。

也就是说，这个设计不是简单多加一个模块，而是**把学习目标从动作相似性，改成了结果一致性 + 动作可实现性**。

### 设计上的关键细节

- **预测对象是状态，不是视频**：避免视频生成缺少接触、形变、受力等细粒度操控信息。
- **inverse dynamics 看历史和未来窗口**：不是只看 \(s_t, s_{t+1}\)，而是看一段过去和一段预测未来，更利于接触-rich 动作求解。
- **真实世界先学状态压缩**：点云太高维，作者先训练 encoder，把点云压到可用于 diffusion 和 inverse dynamics 的低维状态。

### 战略权衡

| 设计选择 | 解决的瓶颈 | 带来的收益 | 代价/风险 |
|---|---|---|---|
| 用状态扩散代替直接动作扩散 | 动作监督无法体现物体结果 | 更能建模目标多模态、长时场景演化 | 状态空间通常比动作空间更大，训练更慢 |
| 显式加入 inverse dynamics | 未来状态到动作的映射存在 embodiment 差异 | 吸收接触细节、双臂协同和控制模式差异 | 仍需高质量 state-action 配对数据 |
| 用历史+未来窗口求动作 | 单步逆动力学上下文不足 | 动作更时序一致，能处理长时任务 | 对时序对齐和 latency 更敏感 |
| 依赖结构化状态/编码器 | 原始视觉过于高维、噪声大 | 训练稳定、落地更容易 | 感知误差会直接影响策略上限 |

---

## Part III：证据与局限

### 关键证据

- **[比较] 仿真多任务稳定胜出，不局限于单一控制模式**  
  在 Block Push、Franka Kitchen、Push-L 上，方法普遍优于 DP/IDP。  
  特别是 Franka Kitchen 中，位置控制下 5-task 完成率达到 **29.3%**，而 DP/IDP 都只有 **3.0%**，说明它不仅更稳，还能合成超出演示组合的新行为。

- **[消融] inverse dynamics 不是可有可无的附件，而是关键因果部件**  
  Push-L 上完整模型 **79.3%**，去掉 inverse dynamics 后降到 **61.3%**。  
  这说明“只会预测未来状态”还不够，真正把未来状态落实成接触动作，需要独立的 inverse dynamics 模块。

- **[比较] 数据效率更高**  
  在较小 demonstration 规模下也持续领先：例如 100 条演示时成功率约 **0.32 vs 0.17**；200 条时 **0.793 vs 0.502**。  
  这支持了作者的主张：状态预测提供了比纯动作模仿更丰富的监督。

- **[真实世界] 不只是仿真好看，双臂实机也有明显收益**  
  - 零样本 sim-to-real Push-L：**9/12 vs 2/12**  
  - Laundry Cleanup 第二个枕头：**8/15 vs 0/15**  
  - Cluttered Shelf Picking：**14/15 vs 0/15**  
  这些结果直接说明：当任务需要双臂稳定协同、处理多物体或柔性对象时，显式建模场景未来演化确实有用。

### 局限性

- **Fails when:**  
  - 感知得到的状态不准，尤其是关键点/点云存在较大噪声、遮挡或时延时；  
  - 任务需要极快反应或超出预测 horizon 的闭环纠偏时，预先预测的未来状态可能失效；  
  - 接触动力学非常复杂但状态编码不足时，inverse dynamics 仍可能学不到足够精细的动作映射。

- **Assumes:**  
  - 有成对 demonstration 数据，且真实双臂任务中作者每个任务收集了约 **200 条 teleoperation 演示**；  
  - 有结构化状态或可训练的低维状态编码器，而不是纯原始视觉端到端；  
  - 有较明显的硬件与系统依赖，包括 **双 UR5e、Gello teleoperation、点云感知/多相机系统、A40 GPU** 等；  
  - 可接受更高训练成本，论文也明确指出其训练时间高于 Diffusion Policy（如 Franka Kitchen 为 **706 min vs 307 min**）。

- **Not designed for:**  
  - 无状态估计、纯 RGB 端到端的开放世界操控；  
  - 无示教数据前提下的自主探索学习；  
  - 需要通用语言指令理解或任务发现的 broader agent setting。

### 可复用组件

- **future-state diffusion**：可复用到其他 manipulation policy 中，作为任务级未来状态先验。
- **inverse-dynamics decoupling**：适合把“世界演化”和“机器人执行”分开学，尤其适用于多臂/多执行器系统。
- **point-cloud-to-state encoder**：适用于把高维真实观测压缩到可控、可预测的低维状态空间。

**一句话总结 so what：**  
这篇论文的能力跃迁不在于“又用了 diffusion”，而在于把操控策略的学习对象从“模仿动作”改成了“预测结果 + 实现结果”，因此在双臂协调、柔性物体和多模态任务上，比端到端 action diffusion 更对题。

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Learning_Coordinated_Bimanual_Manipulation_Policies_using_State_Diffusion_and_Inverse_Dynamics_Models.pdf]]