---
title: "Diffusion Policy Attacker: Crafting Adversarial Attacks for Diffusion-based Policies"
venue: NeurIPS
year: 2024
tags:
  - Embodied_AI
  - task/robot-manipulation
  - diffusion
  - projected-gradient-descent
  - adversarial-patch
  - dataset/PushT
  - dataset/Can
  - dataset/Lift
  - dataset/Square
  - dataset/Transport
  - dataset/Toolhang
  - opensource/no
core_operator: "把扩散策略的攻击目标从端到端动作误差改写为单步噪声预测误差，并用 PGD 在像素级全局扰动与物理补丁上优化，从而系统性破坏条件去噪过程。"
primary_logic: |
  视觉观测或离线演示数据 → 以目标/非目标噪声预测误差为替代目标优化全局扰动或场景补丁 → 扰乱视觉编码与逐步去噪链路 → 生成错误动作序列并降低任务成功率
claims:
  - "With an ℓ∞ perturbation budget of 0.03, DP-Attacker lowers success rates of pretrained diffusion policies across six manipulation tasks in both online and offline digital attack settings, often driving transformer checkpoints from near-perfect performance to near-zero success [evidence: comparison]"
  - "A task-specific adversarial patch covering about 5% of the image degrades multiple tabletop manipulation policies, e.g., Toolhang Transformer drops from 0.86 clean success to 0.02 under offline patch attack [evidence: comparison]"
  - "Optimizing a one-step noise-prediction surrogate is faster and stronger than end-to-end action-loss attacks on Lift-PH Transformer (targeted online: ~1.3s and success rate 0.02 vs DDIM-8 end-to-end ~5.8s and 0.24) [evidence: ablation]"
related_work_position:
  extends: "Diffusion Policy (Chi et al. 2023)"
  competes_with: "PGD (Madry et al. 2018)"
  complementary_to: "PGD adversarial training (Madry et al. 2018)"
evidence_strength: moderate
pdf_ref: paperPDFs/Vision_Action_VA_Models_2024/NeurIPS_2024/2024_Diffusion_Policy_Attacker_Crafting_Adversarial_Attacks_for_Diffusion_based_Policies.pdf
category: Embodied_AI
---

# Diffusion Policy Attacker: Crafting Adversarial Attacks for Diffusion-based Policies

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2405.19424), [Project/Video](https://sites.google.com/view/diffusion-policy-attacker)
> - **Summary**: 本文首次系统化研究视觉扩散策略的对抗攻击，用“单步噪声预测损失”替代“端到端动作损失”，把原本难以实时攻击的随机多步扩散控制器变成可在线、离线和物理补丁攻击的对象。
> - **Key Performance**: 在 $\ell_\infty=0.03$ 下，targeted-online 将 Transformer 扩散策略在 10 个任务/数据设置中的 8 个直接打到 0 成功率；Lift-PH Transformer 上该方法约 1.3s/次、SR=0.02，优于 DDIM-8 端到端攻击的约 5.8s/次、SR=0.24。

> [!info] **Agent Summary**
> - **task_path**: 多视角视觉观测 / 扩散行为克隆 -> 错误动作序列 / 任务失败
> - **bottleneck**: 沿整条随机扩散采样链对最终动作回传梯度成本高且不稳定，难以形成有效在线攻击
> - **mechanism_delta**: 将攻击目标改为单步条件噪声预测误差，并在全局像素扰动或局部物理补丁上做 PGD 优化
> - **evidence_signal**: 6 个 manipulation 任务上，在线/离线数字攻击都能显著降成功率，且比端到端 action-loss 攻击更快更强
> - **reusable_ops**: [单步噪声预测替代目标, 通用离线扰动训练, 变换增强 patch 优化]
> - **failure_modes**: [部分 CNN 主干上攻击效果明显下降, patch 攻击对任务几何和视角位置敏感]
> - **open_questions**: [能否在黑盒条件下保持攻击迁移性, 如何为 diffusion policy 构建有效防御与鲁棒训练]

## Part I：问题与挑战

这篇论文的核心不是“让扩散策略更强”，而是回答一个更基础的安全问题：**扩散策略是否真的比普通视觉策略更难被攻击？**

### 真实问题是什么
扩散策略（DP）把动作序列看成一个从高斯噪声逐步去噪生成的对象。和普通一次前向的 policy network 不同，它的输出来自：

- 一个**多步去噪链**
- 每一步都要读取**条件视觉输入**
- 采样过程里还有**随机性**

因此，攻击目标不再是“一个前向网络的输出”，而是“一个被视觉条件反复调制的随机生成过程”。

### 真正瓶颈在哪里
作者指出，直接攻击最终动作的端到端方法虽然概念直观，但在 DP 上会遇到两个硬障碍：

1. **梯度链太长**：要穿过完整 diffusion sampling chain，代价高。
2. **随机性太强**：同一观测下动作是采样出来的，端到端动作损失不稳定。

这也是为什么以前针对普通 DNN policy 的攻击方法，或者针对 text-to-image latent diffusion 的攻击方法，不能直接搬过来。  
这里的攻击面不是 latent，也不是图像编辑过程，而是**条件观测图像本身**。

### 为什么现在要解决
因为 diffusion policy 已经从“论文方法”走向“机器人控制基线”，并开始进入真实系统原型。  
如果它在视觉层面存在廉价、可迁移、甚至物理可部署的攻击面，那么部署风险会非常直接：

- 相机输入被篡改
- 环境里被放置小 patch
- 机器人执行稳定但错误的动作

### 输入/输出接口与边界
本文研究的是**视觉驱动的 diffusion policy**：

- **输入**：当前多视角图像观测 \(I_t\) 或离线演示数据
- **输出**：动作序列 \(\tau^t\)，最终体现为任务成功率下降

威胁模型边界也很明确：

- **白盒**：可访问 policy / denoiser
- **在线攻击**：每个时刻根据当前图像重新算扰动
- **离线攻击**：预先学一个固定扰动，整段 rollout 复用
- **物理攻击**：预先训练 patch，放到场景里

---

## Part II：方法与洞察

### 方法框架

作者把 DP-Attacker 做成了一个统一攻击框架，覆盖三条轴：

- **Global vs Patch**
- **Online vs Offline**
- **Targeted vs Untargeted**

#### 1. 全局数字攻击
直接对输入图像加小扰动：

- **在线版**：每帧重新优化 \(\delta_t\)
- **离线版**：从训练数据学一个固定 \(\delta\)，所有帧复用

这类攻击适合“黑入相机流”的设定。

#### 2. 物理补丁攻击
学习一个小 patch，训练时对它做随机仿射变换，使其对：

- 位置
- 旋转
- 视角

有一定鲁棒性。测试时把它作为环境中的一个物体贴图放进场景。

#### 3. Targeted / Untargeted
- **Untargeted**：只要让原本正确动作变坏即可
- **Targeted**：把动作推向预定义坏动作

这让方法不仅能“让任务失败”，还可以在更强预算下“朝某类错误动作引导”。

### 核心直觉

**这篇论文最关键的变化，不是换了优化器，而是换了攻击面。**

#### what changed
从“攻击最终动作输出”  
变成  
“攻击共享 denoiser 的单步噪声预测误差”。

#### which bottleneck changed
这一步改变了两个约束：

1. **计算约束变了**  
   不再需要穿过整条 diffusion chain 去优化最终动作，梯度更短、更便宜。

2. **信息瓶颈变了**  
   视觉条件不是只影响一次，而是会在每个 denoising step 被反复使用。  
   所以，只要把条件特征推偏一点，这个误差就会沿整个去噪链累计。

#### what capability changed
于是攻击从“难以实时、难以稳定优化”  
变成“可以在线求解、可以做跨帧固定扰动、还可以做物理 patch”。

### 为什么这个设计有效
因果链条可以概括成：

**像素小扰动 → 编码器条件特征偏移 → 每一步噪声预测更差/更偏目标 → 整条动作去噪链被系统性带偏 → 任务失败**

这也是本文后面分析部分的重点：  
作者发现**真正被打穿的是 encoder**，而不是 diffusion sampler 的某个单独 step。

### 策略性 trade-off

| 方案 | 攻击载体 | 需要的信息 | 优点 | 主要代价/边界 |
|---|---|---|---|---|
| 在线全局攻击 | 每帧像素扰动 \(\delta_t\) | 当前观测 + 白盒梯度 | 最强、可时变、定向能力好 | 需要在线优化；1-2s/次对高频控制仍偏慢；必须能控制相机流 |
| 离线全局攻击 | 固定通用扰动 \(\delta\) | 训练数据 + 白盒模型 | 一次训练、跨帧复用、部署便宜 | 对场景分布变化更敏感；通常弱于在线版 |
| 物理补丁攻击 | 场景小 patch | 训练数据 + 可放置 patch 的物理权限 | 更接近现实攻击 | 效果更依赖视角、位置、任务几何；稳定性不如数字攻击 |

---

## Part III：证据与局限

### 关键证据信号

- **[comparison] 在线数字攻击确实能打穿 DP，而随机噪声不行**  
  在 Transformer 主干上，clean 成功率常常接近 1.0；targeted-online 后，10 个设置里有 8 个直接降到 0，其余也接近 0。  
  相比之下，random noise 往往只带来小幅下降，说明问题不是“DP 怕噪声”，而是“DP 怕结构化对抗噪声”。

- **[comparison] 离线固定扰动具有跨帧迁移性**  
  例如 Can-PH Transformer 从 0.92 降到 0.08，Toolhang-PH 从 0.86 降到 0。  
  这很重要，因为它说明攻击者不一定要每帧实时优化，**一个固定扰动就能在整个 rollout 中持续起作用**。

- **[comparison] 物理 patch 有现实意义，但稳定性更依赖任务**  
  约 5% 视野大小的 patch 能显著伤害多个任务：  
  - Toolhang Transformer：0.86 → 0.02  
  - Can CNN：0.98 → 0.16  
  但在 Lift 等任务上下降有限，说明 patch 攻击更受环境几何与视角约束。

- **[ablation] 单步噪声预测替代目标比端到端动作损失更合适**  
  在 Lift-PH Transformer 上：  
  - DP-Attacker targeted-online：约 1.3s，SR 0.02  
  - DDIM-8 端到端攻击：约 5.8s，SR 0.24  
  - DDPM 端到端攻击：约 67s，SR 0.52  
  结论很明确：**对 DP 来说，攻击共享 noise predictor 比攻击最终 sampled action 更有效。**

- **[analysis] 编码器是主要脆弱点**  
  作者比较了 clean feature 与 attacked feature 的编码距离，发现成功攻击带来的特征偏移显著大于随机噪声。  
  这支持了前面的机制解释：**攻击首先改变的是条件视觉表征，再把误差传播到后续去噪。**

### 1-2 个最重要指标
如果只记两个结果，建议记这两个：

1. **\(\ell_\infty=0.03\)** 的小扰动就足以把多个任务从接近满分打到 **0**  
2. **1.3s vs 5.8s, SR 0.02 vs 0.24**：说明 surrogate loss 不只是更快，也更强

### 局限性

- **Fails when**: 部分 CNN backbone 上攻击明显更难，尤其在某些较简单或几何约束更强的任务上，在线攻击可能几乎无效；物理 patch 在 Lift 一类任务上也没有数字攻击稳定。
- **Assumes**: 白盒访问 diffusion policy 与 noise predictor；可以篡改多视角观测或在场景中放置 patch；离线攻击依赖训练数据；在线攻击仍需较强算力，论文报告大约 1.3-1.8s/次。
- **Not designed for**: 黑盒迁移攻击、真实机器人实机验证、防御机制设计、非视觉条件或更大规模多模态 diffusion policy。

### 额外的复现/扩展提醒
这篇论文的证据虽然覆盖了 6 个 manipulation tasks、20 个 checkpoint，但仍然有几个保守点：

- 主要基于 **Chi et al. 的 diffusion policy family**
- 评测环境是 **simulation**
- 未见公开代码，只有论文与视频站点

所以它更像是在明确指出：**DP 的安全问题是真实且系统性的**，而不是已经把真实世界攻击闭环完全做完。

### 可复用组件

- **单步噪声预测替代损失**：适合攻击带随机采样链的生成式 policy
- **固定通用扰动训练**：适合评估 rollout 级稳定脆弱性
- **变换增强的 patch 优化**：适合从数字攻击走向物理部署
- **encoder feature drift 诊断**：适合判断攻击到底打在“表示层”还是“采样层”

![[paperPDFs/Vision_Action_VA_Models_2024/NeurIPS_2024/2024_Diffusion_Policy_Attacker_Crafting_Adversarial_Attacks_for_Diffusion_based_Policies.pdf]]