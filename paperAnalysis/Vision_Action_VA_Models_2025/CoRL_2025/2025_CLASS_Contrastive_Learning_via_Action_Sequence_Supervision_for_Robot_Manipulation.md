---
title: "CLASS: Contrastive Learning via Action Sequence Supervision for Robot Manipulation"
venue: CoRL
year: 2025
tags:
  - Embodied_AI
  - task/robot-manipulation
  - task/imitation-learning
  - contrastive-learning
  - dynamic-time-warping
  - action-sequence-supervision
  - dataset/robomimic
  - dataset/MimicGen
  - dataset/Aloha-Transfer
  - dataset/LIBERO-Object
  - dataset/Push-T
  - opensource/no
core_operator: "用DTW从未来动作序列中挖掘软正样本，并以软InfoNCE把视觉观测按“未来行为相似性”对齐成可检索、可微调的操控表征"
primary_logic: |
  异质示教中的观测序列与未来动作块 → 用DTW计算跨轨迹动作相似度并按分位数筛选/加权正样本 → 以软监督对比学习训练视觉编码器 → 输出按行为而非视角聚类的表征，可直接做kNN动作检索或作为BC/DP初始化
claims:
  - "在异质仿真设置（动态相机/随机颜色）下，CLASS 预训练的 Diffusion Policy 平均成功率达到 85%，高于最强基线的 57% [evidence: comparison]"
  - "在 3 个真实机器人任务上，CLASS-DP 相比 ImageNet-DP 在 parametric 评测中平均提升 41% 子任务成功率和 55% 最终任务完成率 [evidence: comparison]"
  - "软正样本加权与 DTW 序列相似度是关键因子：改为硬对比或用 L2 替代 DTW 都会显著降低性能，且更长的动作窗口在 T=16 前持续带来收益 [evidence: ablation]"
related_work_position:
  extends: "Supervised Contrastive Learning (Khosla et al. 2020)"
  competes_with: "VINN (Pari et al. 2021); DynaMo (Cui et al. 2024)"
  complementary_to: "Diffusion Policy (Chi et al. 2024); Equivariant Diffusion Policy (Wang et al. 2024)"
evidence_strength: strong
pdf_ref: "paperPDFs/Vision_Action_VA_Models_2025/CoRL_2025/2025_CLASS_Contrastive_Learning_via_Action_Sequence_Supervision_for_Robot_Manipulation.pdf"
category: Embodied_AI
---

# CLASS: Contrastive Learning via Action Sequence Supervision for Robot Manipulation

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2508.01600)
> - **Summary**: 这篇工作把“未来动作序列是否相似”变成监督信号，让机器人先学到按行为聚类、而不是按视角或外观聚类的视觉表征，因此在异质示教下比直接 BC 更稳健。
> - **Key Performance**: 异质仿真设置下 CLASS+DP 平均成功率 85%（最佳基线 57%）；真实 3 任务上相对 ImageNet-DP 的最终任务完成率提升 55%（parametric）。

> [!info] **Agent Summary**
> - **task_path**: 异质视觉示教序列（观测历史 + 未来动作块） -> 行为表征 -> kNN 动作序列检索或 BC/DP 微调策略
> - **bottleneck**: BC 容易把监督绑定到单条示教的外观细节上，导致相机位姿/物体外观变化时学不到跨示教共享的行为结构
> - **mechanism_delta**: 用 DTW 度量跨轨迹未来动作块相似度，并以软正样本 InfoNCE 替代直接动作回归来训练视觉编码器
> - **evidence_signal**: 多任务异质视觉比较 + 关键设计消融；随机相机位的真实机器人实验也保持稳定增益
> - **reusable_ops**: [DTW正样本挖掘, 软正样本InfoNCE]
> - **failure_modes**: [高精度插入时检索控制不够细, 训练分布外相机位姿明显退化]
> - **open_questions**: [如何把DTW挖掘扩展到更大规模数据, 对次优或噪声示教是否稳健]

## Part I：问题与挑战

这篇论文要解决的不是“再换一个更强的 policy head”，而是更前面的瓶颈：**在异质示教数据里，视觉表征没有学到“接下来要做什么”，而是记住了“这一帧长什么样”**。

### 真正的问题是什么
在传统 BC 里，观测直接监督动作。这个监督在**同质数据**上通常够用：固定相机、固定外观、固定采集条件时，视觉差异和行为差异大致一致。但一旦进入更真实的离线示教场景——相机位置变化、视角动态变化、物体颜色变化——这种一一对应监督会把模型推向对单条示教的过拟合。

作者认为，真正缺的是一种**跨示教对齐“行为等价性”**的方法：  
两个观测哪怕视觉上差很多，只要它们后续会导向相似的动作序列，就应该在表征空间里更接近。

### 输入 / 输出接口
- **输入**：示教轨迹中的观测历史 \(O_t\) 与对应未来动作块 \(A_t\)
- **学习目标**：编码器 \(f_\theta(O_t)\) 学到行为驱动的 latent
- **输出用法**：
  1. 直接做 **kNN 检索 + 动作序列集成**（无需 policy head）
  2. 作为视觉 encoder，进一步微调到 **MLP / Diffusion Policy**

### 为什么现在值得解决
因为机器人示教数据正在变“大”，也正在变“杂”。固定相机、单一外观的干净实验室设定，不再代表真实数据分布。若不解决这个瓶颈，更多数据只会把 BC 的外观过拟合问题放大，而不是自然带来泛化。

### 边界条件
- 这不是纯视频自监督；它**仍然依赖动作标注的示教轨迹**
- 当前范围主要是**视觉模态**
- 论文重点测试的异质性主要来自：
  - 相机位姿变化 / 动态相机
  - 物体与背景颜色变化  
而不是更大尺度的 domain gap 或 embodiment gap

---

## Part II：方法与洞察

CLASS 的核心思路很直接：**不用“当前帧对应哪个动作”监督 encoder，而用“这段观测接下来会走向哪类动作序列”监督 encoder。**

### 方法主线

#### 1. 先从动作序列里挖出“行为相似”的正样本
作者不直接看图像是否相似，而是看**未来动作块是否相似**。  
具体做法是：

- 从每个时间点取未来一段动作序列
- 用 **DTW** 计算不同轨迹之间动作序列的距离
- 按距离分位数阈值选出正样本对
- DTW 越小，正样本权重越大

这里 DTW 的作用很关键：它允许两个动作序列有**时间错位**，但整体行为模式仍被识别为相似，这比简单 L2 更适合示教中的时序变形。

#### 2. 用“软正样本”对比学习训练视觉表征
有了正样本后，CLASS 用的是**软版 supervised contrastive / InfoNCE**：

- anchor 和正样本拉近
- 其他样本推远
- 但不同正样本并不是一视同仁  
  而是按 DTW 相似度分配权重

也就是说，CLASS 不是在问“是不是相似行为”，而是在问“**有多相似**”。

#### 3. 表征学完后，两种用法都能跑
- **Rep-Only / non-parametric**：直接在 latent 空间做最近邻检索，把邻居动作块做相似度加权平均后 rollout
- **Parametric**：把 encoder 拿去微调 MLP 或 DP policy head

这点很重要：它说明作者不是把 CLASS 当成一个只能配某个特定控制器的 trick，而是把它定位为一个**可复用的行为表征学习层**。

#### 4. 理论解释
论文给了一个轻量理论结论：优化该损失，相当于让**latent 相似度分布**去匹配**动作相似度分布**。  
这不证明它一定最优，但至少把“为什么这个损失在行为表征上是合理的”说清楚了。

### 核心直觉

**改了什么**：  
从“逐点动作回归监督”改成“跨轨迹的未来动作相似性监督”。

**改变了哪个瓶颈**：  
监督信号不再绑死在单个视角/外观实例上，而是把多个视觉不同、但未来行为相似的状态压到同一邻域中。

**能力为什么会变强**：  
因为在异质数据里，真正稳定的是“行为后果”，不是“像素外观”。  
当 encoder 被迫对齐“未来会怎么动”，它就更容易学出对视角、颜色、背景变化不敏感的行为表征。

**为什么软权重比硬正样本更好**：  
DTW 选出来的正样本并不都同样可靠。  
如果全部硬拉近，会把“部分相似”的样本也强行压到一起，增加假正样本伤害。软权重等于给相似度一个强弱刻度，减少错误聚类。

### 策略权衡

| 设计选择 | 改变的约束/分布 | 收益 | 代价/风险 |
|---|---|---|---|
| 用未来动作块而不是单步动作做监督 | 从瞬时配对转向行为片段对齐 | 更能表达技能结构与长时行为 | 需要设定窗口长度，过短会丢信息 |
| 用 DTW 而不是 L2 做动作相似度 | 允许时序错位 | 更稳地识别“同一行为，不同节奏” | 预计算成本高 |
| 用软正样本权重而不是硬正样本 | 把二值相似改为连续相似 | 减少粗糙正样本带来的误导 | 权重质量依赖 DTW 质量 |
| 先学表征，再接检索/微调 | 把“看懂行为”和“输出动作”解耦 | 编码器可复用、迁移性更好 | 非参数检索在精细控制上可能不够细 |

---

## Part III：证据与局限

### 关键证据

1. **异质视觉条件下的能力跃迁很明显**  
   最有说服力的信号不是固定相机上的小涨点，而是**动态相机 / 随机颜色**这些真正破坏 BC 的设置。  
   在这些异质设置下，CLASS 预训练的 DP 平均成功率 **85%**，而最强基线只有 **57%**。  
   这说明收益来自“行为表征更稳健”，不是单纯多训了几轮。

2. **表征本身就足够强，不只是给更强 policy head 打工**  
   在 Rep-Only 设定下，CLASS 平均成功率达到 **83%**，只比 parametric DP 低 **9%**。  
   这说明核心增益确实在 encoder 学到的表示，而不只是后端控制器变强。

3. **真实机器人上依然成立**  
   在 Two-Stack、Mug-Hang、Toaster-Load 三个真实任务上，随机挪动相机后，CLASS-DP 相比 ImageNet-DP 在 parametric 设置下平均提升 **41% 子任务成功率** 和 **55% 最终任务完成率**。  
   这证明它不是只在模拟器里对视觉 shift 有效。

4. **消融支持“因果旋钮”确实是对的**  
   论文明确显示：
   - 把软对比改成硬对比，性能明显掉
   - 把 DTW 换成 L2，性能掉
   - 动作窗口加长到 16 之前，性能持续更好
   - 正样本阈值太小信息不够，太大又会引入假正样本  
   所以提升并不是“任何对比学习都行”，而是这套**动作序列监督 + 软权重**设计在起作用。

5. **表征分析与训练动态也吻合主张**  
   t-SNE 和近邻可视化显示，CLASS 会把不同视角但相同行为的示教聚到一起，而 BC 学到的表示更像按外观邻近。  
   同时，CLASS 预训练后的微调收敛更快，部分抵消了额外预计算成本。

### 局限性
- Fails when: 需要高精度、细粒度接触控制时，尤其是非参数检索模式下更容易失手；真实任务中还会出现“当前子任务没完成就提前进入下一阶段”的错误；相机位置超出训练采样区域时性能显著下降。
- Assumes: 需要带动作标注的示教轨迹；需要预计算跨样本 DTW 相似度，成本随样本对数近二次增长；当前只覆盖视觉模态；效果明显受益于 ImageNet 初始化，论文附录显示从头训练在异质设置下会大幅掉点；文中未给出代码链接。
- Not designed for: 仅视频无动作监督场景、含大量次优/噪声示教的数据、跨 embodiment 的统一动作空间、超大规模在线增量训练或超大库实时检索。

### 可复用组件
- **DTW-based positive mining**：用未来动作块自动构造跨轨迹正样本
- **Soft positive-weighted InfoNCE**：把“相似程度”而不是“是否相似”注入对比学习
- **Rep-Only kNN rollout**：无需 policy head 的动作块检索控制
- **Pretrain-then-finetune recipe**：先学行为表征，再挂到 BC/DP 上微调

### 一句话结论
这篇论文最重要的贡献，不是提出了一个更复杂的控制器，而是把监督从“像素到动作”改成了“观测到未来行为模式”。这个因果旋钮一旦拧对，机器人在异质示教上的泛化就明显上了一个台阶。

## Local PDF reference

![[paperPDFs/Vision_Action_VA_Models_2025/CoRL_2025/2025_CLASS_Contrastive_Learning_via_Action_Sequence_Supervision_for_Robot_Manipulation.pdf]]