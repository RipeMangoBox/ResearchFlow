---
title: "Diffusion Policy Policy Optimization"
venue: ICLR
year: 2024
tags:
  - Embodied_AI
  - task/robot-learning
  - task/continuous-control
  - diffusion
  - reinforcement-learning
  - dataset/D4RL
  - dataset/ROBOMIMIC
  - dataset/Furniture-Bench
  - opensource/full
core_operator: 将扩散去噪链显式建成内层MDP，并在环境MDP外层上用PPO对每个高斯去噪步做策略梯度微调
primary_logic: |
  预训练扩散策略 + 状态/像素观测 + 环境奖励
  → 把动作生成的多步去噪过程嵌入环境交互，形成双层MDP
  → 在逐步可解析的高斯似然上做PPO更新，并配合状态值函数、末段去噪微调、DDIM与噪声下限裁剪
  → 得到更稳定、探索更结构化、部署更鲁棒的连续控制策略
claims:
  - "Claim 1: DPPO在OpenAI Gym、ROBOMIMIC、Franka Kitchen与Furniture-Bench的扩散策略微调中，相比IDQL、DQL、QSM、RLPD、Cal-QL及Gaussian/GMM PPO基线展现出更稳定且整体更优的最终性能 [evidence: comparison]"
  - "Claim 2: 在ROBOMIMIC的Transport任务上，DPPO从state输入达到>90%成功率、从pixel输入达到>50%成功率，而Gaussian基线在pixel设定下几乎无法从0%预训练成功率提升 [evidence: comparison]"
  - "Claim 3: 仅微调最后若干去噪步、采用带随机性的DDIM微调、以及对扩散噪声设置探索/似然下限，可在保持最终性能的同时改善DPPO的训练稳定性与效率 [evidence: ablation]"
related_work_position:
  extends: "Training Diffusion Models with Reinforcement Learning (Black et al. 2023)"
  competes_with: "IDQL (Hansen-Estruch et al. 2023); QSM (Psenka et al. 2023)"
  complementary_to: "AdaptSim (Ren et al. 2023); Planning with Diffusion (Janner et al. 2022)"
evidence_strength: strong
pdf_ref: paperPDFs/Vision_Action_VA_Models_2024/ICLR_2024/2024_Diffusion_Policy_Policy_Optimization.pdf
category: Embodied_AI
---

# Diffusion Policy Policy Optimization

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2409.00588), [Project](https://diffusion-ppo.github.io/)
> - **Summary**: 这篇论文把扩散策略的去噪链重写成可做策略梯度的双层MDP，用PPO稳定微调预训练 Diffusion Policy，从而在机器人连续控制中获得更强的探索结构、更高鲁棒性和更好的 sim-to-real 表现。
> - **Key Performance**: ROBOMIMIC `Transport` 上 state 输入成功率 >90%、pixel 输入 >50%；`One-leg` 零样本 sim-to-real 成功率 80%（16/20）。

> [!info] **Agent Summary**
> - **task_path**: 状态/像素观测 + 预训练扩散策略 + 在线奖励 → 连续动作 chunk / 机器人控制策略
> - **bottleneck**: 扩散策略没有显式最终动作 likelihood，长去噪链又让PG被认为高方差；而Q-learning式微调在稀疏奖励、高维连续动作上容易不稳定
> - **mechanism_delta**: 把去噪过程视作内层MDP并对逐步高斯似然做PPO，同时只微调后几步去噪并裁剪扩散噪声以稳住训练
> - **evidence_signal**: 跨ROBOMIMIC/Furniture-Bench的基线对比 + Avoid机制分析 + 真实机器人zero-shot sim-to-real
> - **reusable_ops**: [diffusion-mdp-unrolling, last-k-denoising-finetuning]
> - **failure_modes**: [poor-pretraining-coverage, tasks-needing-aggressive-off-manifold-exploration]
> - **open_questions**: [how-to-improve-sample-efficiency-on-real-hardware, how-to-scale-with-large-multitask-vision-pretraining]

## Part I：问题与挑战

**问题是什么？**  
机器人策略越来越常先用示范做 behavior cloning 预训练，再用 RL 微调。但对 **Diffusion Policy** 来说，这一步并不直接：它擅长拟合复杂多峰动作分布，却不天然提供一个像高斯策略那样易于做 policy gradient 的显式动作概率。

**真正瓶颈是什么？****
1. **信用分配难**：最终动作是多步去噪生成的，不是一次性采样出来的显式分布。  
2. **训练稳定性难**：如果直接依赖 Q-learning 类方法，在稀疏奖励、连续高维动作、长 action chunk 下，Q 估计误差容易把 actor 带崩。  
3. **探索质量难**：机器人微调不是“随便加噪声就行”，需要尽量留在预训练示范流形附近，否则很容易学到抖动、危险或不自然动作。

**输入/输出接口**  
- **输入**：状态或像素观测，外加一个已预训练的扩散策略。  
- **输出**：连续动作 chunk（一次预测多个未来动作，执行其中前几个）。  
- **训练信号**：在线环境奖励，既有 dense reward（Gym）也有 sparse reward（操控任务）。

**边界条件**  
这篇论文主要讨论的是 **“预训练后微调”**，不是从零开始 RL；任务主要是连续控制与机器人操作；效果很依赖并行仿真与一个“还不错”的预训练扩散策略。

## Part II：方法与洞察

**方法主线**  
DPPO 的关键不是改 Diffusion Policy 本体，而是改“怎样对它做 RL”：

1. **把去噪过程展开成内层 MDP**  
   每个去噪步都视作一个子动作，具有可解析的高斯 likelihood。  
2. **把环境交互当外层 MDP**  
   于是整个策略变成“环境步 × 去噪步”的双层 MDP。  
3. **在双层 MDP 上做 PPO**  
   这样就能像普通随机策略那样做 on-policy policy gradient，而不必把优化全部押在 Q 函数上。  
4. **配套稳定化设计**  
   - value 只估环境状态，不估去噪中间动作  
   - 给更早的去噪步更小权重（denoising discount）  
   - 只微调最后几步去噪，或改用少步 DDIM 微调  
   - 对扩散噪声设置探索下限与似然计算下限，兼顾探索和数值稳定

### 核心直觉

**改了什么**  
从“对最终动作的隐式分布做RL”改成“对每个去噪步的局部高斯分布做RL”。

**哪个瓶颈被改变了**  
- **概率建模瓶颈**：最终动作 likelihood 难算 → 逐步 likelihood 可解析。  
- **探索分布瓶颈**：末端一次性加噪 → 多步“加噪再拉回示范流形”的结构化探索。  
- **稳定性瓶颈**：依赖偏差较大的 Q 反传 → 用 PPO 做更稳的 on-policy 更新。

**能力因此怎么变**  
- RL 可以真正“穿过”扩散采样链去微调策略；  
- 探索不再是无结构乱抖，而更像沿着示范 manifold 的局部扩展；  
- 更新是渐进式的，不容易一下把预训练策略推离可行动作分布，因此更鲁棒、更适合长时程稀疏奖励任务。

**为什么这设计有效（因果上）**  
扩散去噪的每一步都同时做两件事：  
1. 注入随机性，提供探索；  
2. 把样本往训练数据流形方向“纠正”。  

这使得 DPPO 的探索天然带有“结构”。对机器人来说，这比给最终动作直接加高斯噪声更重要，因为它减少了离谱动作和策略坍塌的概率。

| 设计选择 | 解决的瓶颈 | 带来的能力 | 代价/取舍 |
|---|---|---|---|
| 双层 MDP + PPO | 扩散策略最终动作 likelihood 难直接做PG | 可对扩散策略稳定做 on-policy 微调 | 有效 horizon 变长 |
| 仅依赖环境状态的 value | 去噪中间动作高随机性导致 value/Q 难估 | 降低方差，提升稀疏奖励任务稳定性 | 价值近似更粗 |
| 只微调最后 K' 步 / DDIM 微调 | 全链微调太慢、太耗显存 | 更高效率，常常不损性能 | K' 太小会欠调 |
| 噪声下限裁剪 | 噪声太小探索不足；太大 likelihood 不稳 | 同时保证探索与数值稳定 | 需要调参，存在 sweet spot |

## Part III：证据与局限

### 关键证据信号

**信号1｜与 diffusion-based RL 的直接对比**  
在 Gym、ROBOMIMIC 上，DPPO 相比 IDQL、DQL、QSM、DIPO、AWR/RWR 类扩散基线整体更稳。特别是在 **ROBOMIMIC 这类稀疏奖励操控任务** 上，DPPO 的优势更明显，说明它确实缓解了“Q 估计误差 + 高维连续动作 + 长 action chunk”带来的训练不稳定。

**信号2｜与其他策略参数化的公平对比**  
在同样用 PPO 微调时，DPPO 明显优于 Gaussian / GMM 策略，尤其在更难的 `Square` 和 `Transport` 上。  
最强信号是：**`Transport` 上 state 输入 >90%，pixel 输入 >50%**，而 Gaussian 在 pixel 设定下几乎无法从 0% 预训练成功率起飞。

**信号3｜机制分析支持“为什么它有效”**  
在 D3IL Avoid 可视化中，DPPO 的探索轨迹覆盖更广，但仍围绕示范 manifold 展开；加入动作噪声或增大 action chunk 后，Gaussian/GMM 更容易崩，而 DPPO 仍更稳。这直接支撑了论文的核心解释：**结构化探索 + 多步渐进更新**。

**信号4｜部署层面的实际价值**  
在 Furniture-Bench 的 `One-leg` 任务上，**零样本 sim-to-real 达到 80%（16/20）**；而 Gaussian 虽然仿真中能到 88%，真实机器人上却是 **0%**。这说明 DPPO 的收益不只是 benchmark 分数，而是更“自然、可纠错、抗扰动”的控制行为。

### 局限性

- **Fails when**: 预训练示范覆盖不足，或任务需要强烈的 off-manifold 探索时，DPPO 优势会减弱；论文也提到在 `Lamp` 的低随机性设定下它略逊于 Gaussian，`Kitchen-Partial/Mixed` 因示范不完整也难达到近完美性能。  
- **Assumes**: 需要一个质量尚可的预训练 Diffusion Policy、连续动作 chunk、可大规模并行的仿真环境，以及较大的 batch / GPU 显存；其 sample efficiency 仍低于典型 off-policy 方法，且 wall-clock 相比 Gaussian 参数化可慢到约 2×。  
- **Not designed for**: 极端样本受限的纯真实机器人在线学习、没有预训练覆盖时追求最高样本效率的场景、以及非扩散/离散动作类策略优化问题。  

**复现实用依赖**  
- 训练强依赖并行仿真与 GPU；  
- 显存开销随 fine-tuned denoising steps 线性增长；  
- 真实机器人结果来自“仿真训练 + 零样本部署”，不是在线真实世界 RL。

**可复用组件**  
- 双层 diffusion-MDP 展开：适合任何“扩散序列生成 + 在线反馈”的问题；  
- state-only value baseline：适合高随机性生成式策略；  
- 只微调末端去噪步 / DDIM 微调：适合低成本适配；  
- 探索噪声与 likelihood 噪声分开裁剪：适合稳定训练。

![[paperPDFs/Vision_Action_VA_Models_2024/ICLR_2024/2024_Diffusion_Policy_Policy_Optimization.pdf]]