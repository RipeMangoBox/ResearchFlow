---
title: "DART: A Diffusion-Based Autoregressive Motion Model for Real-Time Text-Driven Motion Control"
venue: ICLR
year: 2025
tags:
  - Motion_Generation_Text_Speech_Music_Driven
  - autoregressive
  - diffusion
  - motion-primitives
  - real-time
  - spatial-control
  - latent-optimization
  - reinforcement-learning
  - status/analyzed
core_operator: 运动原语（overlapping短片段）+ latent diffusion VAE（压缩原语为紧凑潜空间）+ 自回归rollout，支持实时在线生成任意长度动作及latent空间空间控制
primary_logic: |
  动作序列分解为overlapping运动原语（H=2历史帧+F=8未来帧）→ VAE压缩原语为紧凑潜变量z →
  latent denoiser（条件：文本CLIP嵌入+历史帧+噪声latent）→ 自回归rollout生成无限长动作序列
  空间控制：(1) 潜噪声优化 (2) RL-MDP学习控制策略
pdf_ref: paperPDFs/Motion_Generation_Text_Speech_Music_Driven/ICLR_2025/2025_DART_A_Diffusion_Based_Autoregressive_Motion_Model_for_Real_Time_Text_Driven_Motion_Control.pdf
category: Motion_Generation_Text_Speech_Music_Driven
created: 2026-03-14T03:39
updated: 2026-04-06T23:55
---

# DART: A Diffusion-Based Autoregressive Motion Model for Real-Time Text-Driven Motion Control

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [Project](https://zkf1997.github.io/DART/) | [arXiv](https://arxiv.org/pdf/2410.05260.pdf) | 本地 PDF 见文末 Local Reading
> - **Summary**: 将扩散模型与自回归动作原语结合，实现实时在线文本驱动的任意长动作序列生成（比 FlowMDM 快10倍），同时通过潜空间优化或 RL 策略支持精确空间控制（轨迹跟随、场景感知、目标到达）。
> - **Key Performance**:
>   - 实时生成速度：比 SOTA 离线方法 FlowMDM **快 10 倍**，支持真正在线流式生成
>   - 在动作过渡（in-between）、场景感知、目标到达等多任务上均优于或匹配各专家基线

---

## Part I — 问题与挑战 / The "Skill" Signature

**真正的卡点：长程动作的在线实时生成与精确空间控制的同步满足**

- **核心能力**：给定文本流（多个连续动作描述）+ 空间约束，实时（在线，无需预先知晓完整时间线）生成连续、无缝衔接的任意长度 3D 动作序列。
- **挑战来源**：
  1. *离线 vs 在线*：FlowMDM 等 SOTA 方法是离线的（需要提前知道完整动作时间线），无法响应实时输入流；
  2. *语义控制与空间控制冲突*：文本语义定义动作类型，空间约束（到达某点/跟随轨迹/与场景适配）是额外几何信息——两者在潜空间中难以同时满足；现有方法（DNO 等）在整个序列上操作，难以实现精细局部控制；
  3. *长程连贯性*：直接对整个长序列建模导致数据分布复杂、训练困难、推理耗时。
- **输入/输出接口**：H帧种子动作 + N个文本提示流 + 空间目标（可选：关键帧姿态/关节轨迹/场景点云） → 任意长度 SMPL-X 动作序列（连续、实时流式输出）。
- **边界条件**：运动原语长度固定（H=2, F=8），对极长步幅动作的精确空间控制有累积误差；CLIP 语义嵌入对细粒度动作区分有上限；RL 控制方案训练成本较高。

---

## Part II — 方法与洞察 / High-Dimensional Insight

**设计哲学：将连续长动作"原子化"为短原语，在潜空间实现高效自回归，再用潜空间控制框架统一空间约束**

### The "Aha!" Moment

**作者改变了什么**：将人体运动建模单元从"完整序列"降维为**运动原语（motion primitive）**——每个原语由 H 帧历史（与上一原语末端重叠）和 F 帧未来组成，然后用 **latent diffusion VAE** 将每个原语压缩为紧凑潜变量 z。

**为什么有效**：
- *数据分布简化*：短原语（10帧）的数据分布远比完整序列（100+帧）更简单、更紧凑，diffusion 模型学习效率更高，所需扩散步数更少（推理快）；
- *自回归天然实时*：每个原语的生成仅依赖上一原语末端（history frames），无需全局时间线信息，支持流式在线推理；
- *历史帧重叠保证连贯*：history frames 作为条件注入 denoiser，确保生成的下一段运动在动力学上与上一段平滑衔接；
- *潜空间控制的优越性*：在压缩的原语潜空间内进行优化，比在全动作序列的显式空间（DNO 方式）效果更好——潜空间过滤了数据中的噪声和伪影（VAE 天然去噪），使优化更稳健；RL-MDP 方案则将潜空间视为行动空间，策略网络学习在此空间中探索，比 DNO 更具可复用性。

**战略权衡**：
- 优势：实时、任意长度、统一框架支持语义+空间控制；架构简单，无任务专属模块；
- 局限：原语长度（H+F=10帧）为超参，需根据动作特性调整；CLIP 文本编码器细粒度不足；RL 控制训练需额外成本；H=2的历史过短，对于需要长历史依赖的动作（如太极拳缓慢转换）可能不足。

---

## Part III — 实验与证据 / Technical Deep Dive

**核心 Pipeline**：

```
历史帧 H + 未来帧 X（原语 Pi）→
  VAE Encoder（Transformer）→ latent z（通过 distribution tokens Tμ/Tσ 参数化）
  VAE Decoder（Transformer）→ 重建 X̂（条件：H + z）
      ↓（VAE训练完后固定权重）
Latent Denoiser（Transformer）
  条件：时间步t + CLIP文本嵌入 + 历史帧H + 噪声latent zt
  → 预测干净 latent ẑ0
      ↓
自回归 rollout：
  X1 ~ denoiser(text_1, H_seed) → X2 ~ denoiser(text_2, X1的末尾H帧) → ...

空间控制（可选）：
  (1) 潜噪声优化：对生成的 zt 求梯度，使解码动作满足空间约束
  (2) RL-MDP：策略网络以 z 为行动空间学习满足空间约束的潜变量探索策略
```

**运动表示**：SMPL-X，每帧 D=276 维（root translation + orientation + 局部关节旋转 + 关节位置 + 时序差分特征）；每个原语在第一帧骨盆为中心的局部坐标系中规范化。

**关键实验信号**：
- 速度优势最直接：FlowMDM 生成相同时长动作需分钟级，DART 实现实时（100ms级），这不是微小改进而是范式级差异；
- 动作过渡（in-between）与目标到达：DART 在空间精度指标上与专门设计的 CLoSD 相当，同时兼顾文本语义；
- 无 VAE 的消融（直接在原始动作空间去噪）显示输出中出现明显抖动和伪影，证明 VAE 压缩在动作生成中的关键去噪作用。

**实现约束**：AMASS 数据集预训练；HumanML3D 文本标注；H=2, F=8；SMPL-X；CLIP-L 文本编码；ICLR 2025。

---

![[paperPDFs/Motion_Generation_Text_Speech_Music_Driven/ICLR_2025/2025_DART_A_Diffusion_Based_Autoregressive_Motion_Model_for_Real_Time_Text_Driven_Motion_Control.pdf]]
