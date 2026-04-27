---
title: "GauDP: Reinventing Multi-Agent Collaboration through Gaussian-Image Synergy in Diffusion Policies"
venue: NeurIPS
year: 2025
tags:
  - Embodied_AI
  - task/multi-agent-manipulation
  - task/visuomotor-policy-learning
  - diffusion
  - gaussian-splatting
  - sparse-view-reconstruction
  - dataset/RoboFactory
  - opensource/full
core_operator: 从多视角RGB重建共享3D Gaussian场，并将与各agent视角对齐的全局几何特征回流到局部图像特征中，驱动联合扩散策略生成协同行为
primary_logic: |
  多视角RGB观测 → 稀疏无位姿3D Gaussian重建并用深度辅助微调 → 按视角选择高斯特征并与局部图像像素级融合 → 全局扩散策略输出多agent动作序列
claims:
  - "Claim 1: 在论文报告的6任务 RoboFactory 对比中，GauDP 仅用 RGB 输入就取得最高平均成功率 19.67%，超过所有对比的2D/3D扩散策略基线（最佳基线 14.33%） [evidence: comparison]"
  - "Claim 2: 像素级预融合是关键因果部件；去掉 pre-fusion 后平均成功率从 19.67% 降到 1.17% [evidence: ablation]"
  - "Claim 3: 对 Gaussian 重建器做场景微调并加入深度监督后，稀疏视角重建质量显著提升（PSNR 23.424 vs 17.918，LPIPS 0.148 vs 0.492） [evidence: comparison]"
related_work_position:
  extends: "Diffusion Policy (Chi et al. 2023)"
  competes_with: "3D Diffusion Policy (Ze et al. 2024); Dense Policy (Su et al. 2025)"
  complementary_to: "Causal Diffusion Policy (Ma et al. 2025); VIMA (Jiang et al. 2023)"
evidence_strength: moderate
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/NeurIPS_2025/2025_GauDP_Reinventing_Multi_Agent_Collaboration_through_Gaussian_Image_Synergy_in_Diffusion_Policies.pdf
category: Embodied_AI
---

# GauDP: Reinventing Multi-Agent Collaboration through Gaussian-Image Synergy in Diffusion Policies

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2511.00998), [Project](https://ziyeeee.github.io/gaudp.io/)
> - **Summary**: 论文把多视角 RGB 先重建为共享 3D Gaussian 全局场，再把与各 agent 视角相关的全局几何按像素对齐回灌到局部视觉特征中，使扩散策略同时获得精细操控与多臂协同能力。
> - **Key Performance**: RoboFactory 6任务平均成功率 19.67%，高于最佳基线 14.33%；重建 PSNR 23.42 vs 17.92。

> [!info] **Agent Summary**
> - **task_path**: 多视角RGB观测 / 多臂协作操控 -> 多agent联合动作序列
> - **bottleneck**: 仅局部视角缺乏全局协同状态，纯全局表征又缺少执行级局部细节，导致多臂任务中“协调”和“精定位”难以兼得
> - **mechanism_delta**: 先用稀疏无位姿 3D Gaussian 重建共享场，再将每个视角对应的高斯特征按像素对齐回流到该 agent 的局部图像特征中，并用全局扩散策略联合生成动作
> - **evidence_signal**: RoboFactory 平均成功率 19.67% 优于最佳基线 14.33%，且去掉 pixel-level prefuse 后降到 1.17%
> - **reusable_ops**: [sparse-view Gaussian reconstruction, pixel-aligned global-to-local feature dispatch]
> - **failure_modes**: [4-arm高难任务成功率接近0, 对动态场景与更广任务分布的泛化尚未被充分验证]
> - **open_questions**: [如何在动态/遮挡场景中稳定维护共享Gaussian场, 如何把Gaussian表示扩展成可被VLA或世界模型直接消费的token]

## Part I：问题与挑战

这篇工作的真正问题，不是“再做一个更大的 policy”，而是**如何给每个 agent 同时提供可执行的局部细节与一致的全局协作状态**。

### 1. 问题是什么
在多臂协作操控里，每个机械臂都需要：
- 看到自己手边的细节，完成抓取、放置、对位；
- 同时理解其他机械臂、目标物和场景的整体关系，避免碰撞和时序冲突。

现有做法通常落在两端：
1. **只看局部图像**：每个 agent 只依赖自己的视角，细节够，但缺少 joint state，容易出现“别人还没准备好，我先执行了”的失配。
2. **只看全局观测/点云**：场景一致性更强，但 agent-specific 的精细局部线索变弱，低层控制不稳定。

作者认为，**真正瓶颈是表示层**：不是 action head 不够强，而是输入给 policy 的状态表征既没有被 3D 约束，也没有针对 agent 做选择性分发。

### 2. 输入/输出接口
- **输入**：来自 2–4 个机械臂对应多视角的同步 RGB 图像
- **输出**：未来若干步的多 agent 联合动作序列
- **训练/部署边界**：
  - 推理时只用 RGB
  - 深度图和相机位姿只在 Gaussian 重建器微调阶段使用
  - 主实验使用的是**共享的全局 Diffusion Policy**，不是每个臂单独一个 local policy

### 3. 为什么现在值得做
这件事现在可做，靠的是两类条件成熟：
- **扩散策略**已经成为强视觉模仿学习基线，但多 agent 场景下仍受限于表征；
- **稀疏无位姿 3D 重建**（如 Noposplat）让“只靠多视角 RGB 构造共享 3D 场”变得可行；
- **RoboFactory** 这类多 agent benchmark 出现，使得该问题能被系统评估。

---

## Part II：方法与洞察

GauDP 的核心不是把 3DGS 当成一个额外输入模态，而是用它**重写信息流**：  
先把多视角图像在 3D 空间里对齐成一个共享场，再把“与某个 agent 相关的全局信息”送回它的局部特征中。

### 方法拆解

#### 1. 全局上下文重建：从多视角 RGB 到共享 3D Gaussian 场
作者基于 **Noposplat** 做前馈式稀疏无位姿重建：
- 每个视角图像先经共享 ViT encoder 编码；
- 再通过 cross-view decoder 融合多视角信息；
- 为每个像素预测对应的 3D Gaussian 参数；
- 用重建损失监督，并在机器人场景上额外加 **depth supervision** 微调。

这一步的作用是：把原来彼此分裂、互相不对齐的多视角 2D 观测，压缩成同一个 3D 坐标系下的共享场景表征。

#### 2. 选择性分发：不是把全局场整包喂给每个 agent
作者没有让每个 agent 都消费整套 global context。  
相反，他们只把**与该视角来源对应的 Gaussian 子集**分发回去。

关键点在于：
- 这些 Gaussian 虽然来自该视角像素，但在 cross-view decoder 里已经吸收了其他视角的信息；
- 所以它不是“原始局部信息”，而是“对该 agent 有用的、带全局感知的摘要”。

这一步解决的是**信息污染**问题：  
若把全局场全部丢给每个 agent，反而会引入大量无关内容，干扰局部决策。

#### 3. 像素级协同融合：全局几何必须落回可执行的2D对齐空间
作者把选出的 Gaussian 特征重新映射成和原图同分辨率的 2D grid，然后：
- 与局部图像特征拼接
- 通过轻量卷积 fusion module 融合

这比“高层 feature concat”更细，因为它保留了**空间对齐关系**。  
对于抓取点、放置点、端执行器-物体关系这类低层控制，像素对齐非常重要。

#### 4. 全局扩散策略：联合生成多 agent 动作
融合后的 per-agent feature 再进入共享的 Diffusion Policy，通过 cross-attention 生成联合动作序列。  
这意味着：
- 表征层共享 3D 场
- 决策层共享全局 policy
- 从输入到输出都在建模 inter-agent dependency

附录里也验证了这件事：**Global GauDP 19.67% vs Local GauDP 5.33%**，说明多臂依赖不能只靠独立 agent 各自学。

### 核心直觉

GauDP 改变的是一个很具体的因果链：

**从“在2D里隐式猜几何和协作关系”  
变成“先用3D约束统一场景，再把与执行相关的全局几何按像素送回局部控制”**

#### what changed → which bottleneck changed → what capability changed
- **What changed**：从直接拼接多视角图像/特征，变为“共享 Gaussian 场 + 按视角选择性回流 + 像素级融合”
- **Bottleneck changed**：
  - 多视角之间原本缺少 3D 一致性约束
  - 每个 agent 原本接收的信息要么太局部，要么太泛全局
  - 全局/局部融合原本太粗，难落到执行层
- **Capability changed**：
  - 更强的全局协同意识
  - 更稳定的局部对位能力
  - RGB-only 输入下逼近甚至超过部分 3D baseline 的效果

#### 为什么这个设计有效
因为它把最难学的那部分“跨视角几何对齐”和“谁该看什么全局信息”，从下游 policy 的隐式学习负担中拿了出来，交给一个显式的 3D 中间表示去完成。  
换句话说，policy 不需要再从示范数据中同时学：
1. 多视角几何；
2. 多 agent 交互关系；
3. 低层动作分布。  

它只需要在一个已经被结构化过的输入上学动作生成。

#### 一个细节上的洞察
坐标系统的消融表明：**保留 agent-centric 的局部坐标，比统一到世界坐标更好**。  
这说明对于执行型控制，绝对全局一致性不是唯一目标，**相对“我与目标/其他臂”的局部几何关系**更重要。

### 战略权衡

| 设计选择 | 带来的收益 | 代价 / 风险 |
|---|---|---|
| 稀疏无位姿 Gaussian 重建 | 部署时只用 RGB 也能得到共享 3D 场 | 需要额外预训练/微调重建器 |
| 深度辅助微调 | 提高 3D 结构 fidelity | 微调阶段依赖深度与相机位姿 |
| 按视角选择性分发高斯 | 降低无关全局信息干扰 | 可能损失一部分跨视角远程信息 |
| 像素级预融合 | 更适合抓取、对位等执行层控制 | 训练更复杂，耗时略增 |
| 全局共享 policy | 显式建模 agent 间依赖 | 更依赖联合训练数据质量 |

---

## Part III：证据与局限

### 关键证据

#### 1. 比较信号：RGB-only 方案在总体上超过对比基线
最核心结果是 RoboFactory 6任务平均成功率：
- **GauDP: 19.67%**
- 最佳对比基线 **DP3(XYZ+RGB): 14.33%**

尤其在需要明显协同的任务上，优势更突出：
- **Lift Barrier**：72% vs 31%
- **Align Camera**：26% vs 18%

这说明它的提升主要来自**协同和几何感知被同时改善**，而不是单纯参数变大。

但也要注意，它**不是每个任务都赢**：
- Place Food 上仍低于最强 3D baseline（15% vs 25%）
- 3-arm / 4-arm 的高难任务整体仍然很难

所以这篇论文的 capability jump 更准确地说是：  
**在 RGB-only 多 agent 表征上，显著缩小了与 3D 输入策略的差距，并在平均表现上反超。**

#### 2. 消融信号：真正起作用的是“几何 + 选择性分发 + 像素对齐”
几个最关键的消融结果很有因果指向性：
- **w/o prefuse**：19.67% → 1.17%  
  说明 coarse fusion 不够，像素级对齐几乎是必要条件。
- **w/o Gaussian**：19.67% → 5.00%  
  退化回普通 2D DP 水平，证明共享 3D 场确实在贡献全局几何。
- **w/o Image**：19.67% → 11.17%  
  只靠 Gaussian 也不够，局部外观细节仍不可替代。
- **Local policy vs Global policy**：5.33% → 19.67%  
  说明共享表征之外，联合动作建模本身也重要。

结论很清楚：  
**不是“3D 比 2D 好”，而是“几何全局上下文必须以 agent-aware、pixel-aligned 的方式接入局部控制”。**

#### 3. 重建信号：3D 场质量确实被提升
重建质量从预训练模型到微调模型明显提升：
- PSNR：17.918 → 23.424
- SSIM：0.580 → 0.779
- LPIPS：0.492 → 0.148

这支持论文的基本前提：  
下游协作能力的提升，与上游共享 3D 表示质量改进是相关的。

#### 4. 真实机器人与鲁棒性信号：有一定外部有效性，但证据仍有限
- 真实机器人三项任务上，GauDP 都优于 DP
- 在随机光照和干扰物设置下，Grab Roller 成功率 **50%**，高于 DP 的 20%，也略高于 DP3 的 46%

这说明方法不是只在仿真中有效。  
但由于真实任务数量和规模都有限，这部分更像**支持性证据**，还不足以单独证明广泛实用性。

#### 5. 资源与效率信号
- 训练时间：**6.5 GPU h**，高于 DP 的 4.8 和 DP3 的 2.5
- 推理速度：**1.28 FPS**，略慢于 DP/DP3，但仍在同一量级
- 参数上，GauDP-full 约 **0.75B–0.82B**，不过作者强调 policy 额外开销很小，很多额外参数在策略训练时是冻结的

所以它的代价不是不可接受，但也不是“白捡”的性能提升。

### 局限性

- **Fails when**: agent 数继续增多、任务需要更强时序同步与高精度堆叠时，成功率仍会明显塌缩；3-arm Stack Cube 和 4-arm Take Photo 基本仍未解决；动态场景、剧烈遮挡、非刚体交互并未被充分验证。
- **Assumes**: 有同步多视角 RGB；存在可迁移的 Gaussian 重建预训练模型；微调阶段可获得深度图与相机位姿；任务主要接近 RoboFactory 里的多臂刚体操控分布；主设置依赖共享全局 policy。
- **Not designed for**: 语言条件规划、开放世界语义理解、多模态决策、快速动态世界建模、跨机器人形态的大规模零样本泛化。

### 可复用组件
这篇论文最值得迁移的，不是完整系统，而是几类操作：
- **feed-forward sparse-view Gaussian reconstruction**：把多视角 RGB 压成共享 3D 场
- **view-selective Gaussian dispatch**：不是广播全部全局信息，而是按视角定向下发
- **pixel-aligned global-local fusion**：让全局几何真正服务于执行层
- **shared global diffusion policy**：在动作层显式建模 inter-agent dependency

一句话总结它的“so what”：
**GauDP 证明了，多 agent RGB 操控的关键不是把 2D policy 做得更大，而是先把多视角观测组织成一个结构化、可分发、可执行的共享 3D 中间表示。**

## Local PDF reference

![[paperPDFs/Vision_Action_VA_Models_2025/NeurIPS_2025/2025_GauDP_Reinventing_Multi_Agent_Collaboration_through_Gaussian_Image_Synergy_in_Diffusion_Policies.pdf]]