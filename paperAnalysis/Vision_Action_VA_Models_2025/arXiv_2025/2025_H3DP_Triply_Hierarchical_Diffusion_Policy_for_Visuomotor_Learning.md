---
title: "H3DP: Triply-Hierarchical Diffusion Policy for Visuomotor Learning"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/visuomotor-learning
  - task/robotic-manipulation
  - diffusion
  - hierarchical-conditioning
  - depth-layering
  - dataset/MetaWorld
  - dataset/ManiSkill
  - dataset/Adroit
  - dataset/DexArt
  - dataset/RoboTwin
  - opensource/no
core_operator: "通过“深度分层输入 + 多尺度视觉表征 + 粗到细分阶段扩散去噪”把视觉层级与动作层级显式对齐。"
primary_logic: |
  RGB-D观测与机器人位姿 → 按深度把图像分层并提取多尺度视觉特征 → 在扩散去噪的不同阶段分别注入粗/细粒度视觉条件 → 生成更稳定且更具视觉语义对齐的动作序列
claims:
  - "在 5 个仿真基准的 44 个任务上，H3DP 的平均成功率达到 75.6±18.6，相比 DP3 的 59.3±24.9 带来 +27.5% 相对提升 [evidence: comparison]"
  - "在 4 个杂乱真实世界双臂操作任务上，H3DP 相比 Diffusion Policy 的平均成功率提升 +32.3% [evidence: comparison]"
  - "去掉深度分层、多尺度表征或层级动作生成中的任一模块，消融基准平均成功率都会从 59.6 降到 46.5–49.0，说明三层级设计均有独立贡献 [evidence: ablation]"
related_work_position:
  extends: "Diffusion Policy (Chi et al. 2023)"
  competes_with: "3D Diffusion Policy / DP3 (Ze et al. 2024); CARP (Gong et al. 2024)"
  complementary_to: "Consistency Policy (Prasad et al. 2024); One-Step Diffusion Policy (Wang et al. 2024)"
evidence_strength: strong
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_H3DP_Triply_Hierarchical_Diffusion_Policy_for_Visuomotor_Learning.pdf
category: Embodied_AI
---

# H3DP: Triply-Hierarchical Diffusion Policy for Visuomotor Learning

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2505.07819), [Project](https://lyy-iiis.github.io/h3dp/)
> - **Summary**: 这篇工作把 RGB-D 输入、视觉表征和扩散动作生成同时做成层级结构，用“粗到细”的条件对齐来增强杂乱场景中的视觉-动作耦合。
> - **Key Performance**: 44 个仿真任务平均成功率 75.6±18.6，较 DP3 相对提升 +27.5%；4 个真实任务平均较 DP 提升 +32.3%

> [!info] **Agent Summary**
> - **task_path**: RGB-D观测 + 机器人位姿 -> 多步连续控制动作
> - **bottleneck**: 视觉编码与扩散动作生成是分离设计，导致深度结构、多尺度语义与动作细节无法在同一层级上对齐
> - **mechanism_delta**: 把输入、表征、去噪过程都分层，并让早期去噪看粗特征、后期去噪看细特征
> - **evidence_signal**: 跨 5 个仿真基准和 4 个真实任务的比较结果都显著优于 DP/DP3，且三项层级组件消融都会明显掉点
> - **reusable_ops**: [depth-aware layering, stage-wise multi-scale conditioning]
> - **failure_modes**: [diffusion推理延迟, 深度质量差时收益受限]
> - **open_questions**: [层级条件能否蒸馏成单步策略, 透明或反光物体下深度分层是否仍稳健]

## Part I：问题与挑战

这篇论文针对的是**视觉模仿学习中的 visuomotor policy**：输入是单相机 RGB-D 观测与机器人本体状态，输出是一个连续动作序列 chunk。

### 真正的问题是什么
作者认为，现有方法的主要短板**不是动作生成器不够强**，而是：

1. **深度信息没有被结构化使用**  
   很多方法只是把 RGB 和 depth 直接拼接，结果深度只像“多了一个通道”，没有真正形成前景/背景、远近关系、遮挡结构等几何先验。

2. **视觉表征过于扁平**  
   把图像压成单一尺度特征，容易把“全局任务语义”和“局部接触细节”混在一起，尤其在 clutter、遮挡、多物体场景里更容易失真。

3. **视觉条件与扩散去噪过程没有对齐**  
   Diffusion Policy 类方法通常把同一种视觉条件喂给所有去噪步，但扩散去噪本身存在“先恢复低频、再补高频”的天然偏置。  
   换句话说，**动作生成是分阶段变精细的，但视觉条件不是分阶段提供的**。

### 为什么现在要解决
因为机器人操作的难点已经从“能不能生成动作”转向“能不能在复杂视觉条件下生成对的动作”：

- 真实任务越来越多是**杂乱、遮挡、长时程、多目标**；
- RGB-D 传感器已很常见，但现有 RGB-D policy 还没充分吃到深度红利；
- 点云方案如 DP3 虽强，但往往依赖**高质量深度**和**理想分割**，实操成本高。

### 输入/输出与边界条件
- **输入**：RGB-D 图像 + 机器人位姿/本体状态
- **输出**：动作 chunk
- **学习范式**：模仿学习，依赖专家示教
- **场景边界**：重点解决 cluttered manipulation，而不是语言规划、在线探索 RL 或超高速控制

**一句话概括 What/Why**：  
这篇论文要解决的核心瓶颈，是**视觉层级结构与动作生成层级结构之间没有被显式绑定**；而在真实机器人越来越依赖 RGB-D 和复杂场景操作的今天，这个瓶颈已经直接限制泛化与部署。

## Part II：方法与洞察

H3DP 的关键思想不是单独改视觉、也不是单独改动作，而是把整个 pipeline 都做成三层级：

1. **输入层级**：Depth-aware layering  
2. **表征层级**：Multi-scale visual representation  
3. **动作层级**：Hierarchically conditioned diffusion

### 1）输入层级：深度感知分层
作者先按深度把 RGB-D 图像切成多个 layer，而不是把 depth 当成普通通道直接拼进去。

这样做的作用是：

- 把前景/背景、近处/远处显式拆开；
- 在 clutter 场景中抑制无关背景和遮挡干扰；
- 给后续编码器一个更强的几何归纳偏置。

核心收益：**让“看见深度”变成“按深度组织视觉信息”。**

### 2）表征层级：多尺度视觉表示
对每个 depth layer，H3DP 不只提一个视觉特征，而是提取多个尺度的特征：

- 粗尺度：更适合全局上下文、物体布局、任务语义；
- 细尺度：更适合边缘、接触点、局部几何细节。

这里作者还用了类似 VQ/VQ-VAE 的量化 codebook 来组织多尺度表征，并加了跨尺度一致性约束。  
直观上，它不是让模型记一个“大特征向量”，而是让模型保留“从大轮廓到小细节”的视觉金字塔。

### 3）动作层级：分阶段条件扩散
这是最关键的一步。

H3DP 把整个 diffusion denoising 过程分成多个 stage：

- **早期去噪**：用粗尺度视觉特征，先确定动作的大体结构；
- **后期去噪**：再用细尺度视觉特征，补充精细接触和局部修正。

这相当于把“视觉粗细粒度”与“动作生成粗细粒度”一一对齐。

与 prior work 的差异在于：

- 以往层级方法多半只层级化**动作生成**；
- H3DP 是把**输入、视觉表示、动作生成**三者一起层级化；
- 所以它优化的不是某一端，而是**视觉-动作耦合方式本身**。

### 核心直觉

**改了什么：**  
把“单一视觉条件作用于所有去噪步”改成“不同尺度视觉条件作用于不同去噪阶段”。

**改变了哪个瓶颈：**  
把原来混杂、扁平、弱对齐的信息流，改成了**有几何结构的输入 + 有语义分辨率的表征 + 与频谱演化匹配的去噪条件**。

**带来了什么能力变化：**  
模型更容易先决定“做什么大动作”，再决定“如何精确接触”，因此在杂乱、深度变化大、长时程任务里更稳。

更因果地说，这个设计有效是因为：

- 粗特征更稳定，适合在高噪声早期阶段决定全局动作趋势；
- 细特征更脆弱但更精确，适合在后期低噪声阶段修正局部细节；
- 深度分层先把几何结构整理好，避免 fine detail 一开始就被背景和遮挡污染。

这其实回答了 **How**：  
作者引入的关键 causal knob 是**“按层级匹配视觉条件与扩散阶段”**。  
变化链条是：  
**统一条件 -> 分阶段条件**，  
**弱结构视觉输入 -> 深度与尺度显式结构化**，  
最终得到**更强的空间理解与更稳定的动作细化能力**。

### 战略权衡

| 设计 | 改变的瓶颈 | 能力收益 | 代价/风险 |
|---|---|---|---|
| 深度分层输入 | depth 不再是附加通道，而是前景/背景与远近结构先验 | 杂乱场景、更大深度变化下更稳 | 依赖可用 depth；层数 N 过大可能过分切碎 |
| 多尺度视觉表征 | 单尺度表征难同时兼顾全局语义和局部细节 | 提升语义保真与实例泛化 | 增加编码与一致性训练复杂度 |
| 分阶段扩散条件 | 所有去噪步共享同一视觉条件，与 diffusion 频谱偏置不匹配 | 粗到细动作生成更自然，动作与视觉更对齐 | 推理延迟仍主要受 diffusion 本身限制 |

## Part III：证据与局限

### 关键证据

- **比较信号：跨 5 个仿真基准的广泛领先**  
  在 44 个仿真任务上，H3DP 的平均成功率为 **75.6±18.6**，高于 DP 的 48.1 和 DP3 的 59.3。  
  这说明它的收益不是某个单一任务上的偶然提升，而是跨 articulated、deformable、dexterous、dual-arm 等多类任务都有效。

- **比较信号：真实世界 cluttered 双臂任务仍成立**  
  在 4 个真实任务中，平均比 DP 提升 **+32.3%**。尤其是 Clean Fridge 和 Pour Juice 这类“目标识别 + 长程操作”的任务提升更明显，说明改进不是只在低层控制上，而是在**视觉理解到动作决策的耦合**上。

- **消融信号：三层级不是可替换装饰，而是共同起作用**  
  去掉任一模块——深度分层、多尺度表征、层级动作生成——平均性能都会明显下降。  
  这支持一个重要结论：H3DP 的提升来自**整条感知-动作链路的层级对齐**，而非单点 trick。

- **分析信号：动作频谱演化支持粗到细设计**  
  DFT 分析显示去噪早期先出现低频动作成分，后期再补高频细节。  
  这给“早期看粗特征、后期看细特征”提供了机理层面的支持，虽然它更像 supporting evidence，而不是严格因果证明。

- **对照信号：不是单纯靠更大视觉 backbone**  
  文中附加实验显示，给 DP 换预训练 DINOv2 只有小幅增益，而 H3DP 用更轻量的视觉编码器仍更强。  
  这说明性能提升主要来自**结构化耦合方式**，不是简单堆模型规模。

### 局限性

- **Fails when**:  
  深度质量差、透明/反光物体导致深度失真时，深度分层的收益会受影响；对需要极低延迟闭环控制的场景，diffusion 推理仍偏慢；当分层数 N 过大时，会出现过度切分、表征能力下降。

- **Assumes**:  
  依赖 RGB-D 传感器与稳定标定；依赖模仿学习示教数据（仿真每任务约 50–1000 条，真实任务约 100–500 条）；真实长时程任务中还使用了预训练 ResNet18，并依赖异步推理、temporal ensembling、p-masking 等工程技巧来稳定部署。

- **Not designed for**:  
  无 depth 的纯 RGB 设定、无示教的在线 RL、语言驱动的通用机器人规划，以及要求远高于 10–15 Hz 实际控制频率的超高速操作场景。

### 可复用组件

- **depth-aware layering**：可直接作为 RGB-D policy 的通用前端，把深度从“附加通道”变成“结构先验”。
- **stage-wise multi-scale conditioning**：可迁移到 diffusion / flow-matching 类动作生成器中。
- **p-masking + 异步执行 + temporal ensembling**：对真实机器人部署有实用价值，尤其适合缓解视觉 grounding 不足和 diffusion 延迟。

**一句话回答 So what**：  
H3DP 的能力跃迁，不在于把 diffusion policy 做得更大，而在于首次较完整地把**输入结构、视觉语义层级、动作去噪阶段**绑在了一起，因此能在不依赖点云精分割的前提下，用原始 RGB-D 获得更强的空间泛化与真实部署表现。

## Local PDF reference

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_H3DP_Triply_Hierarchical_Diffusion_Policy_for_Visuomotor_Learning.pdf]]