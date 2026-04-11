---
title: "DartControl: A Diffusion-Based Autoregressive Motion Model for Real-Time Text-Driven Motion Control"
venue: ICLR
year: 2025
tags:
  - Motion_Generation_Text_Speech_Music_Driven
  - task/text-to-motion
  - diffusion
  - autoregressive
  - reinforcement-learning
  - dataset/BABEL
  - dataset/HumanML3D
  - dataset/AMASS
  - repr/SMPL-X
  - opensource/full
core_operator: 潜在运动基元扩散模型（Latent Motion Primitive Diffusion）：将长序列分解为重叠短基元，在VAE压缩的潜空间中用条件扩散自回归生成，实现实时文本驱动+空间控制
primary_logic: |
  文本提示 + 历史运动种子
  → VAE编码器压缩运动基元为紧凑潜变量z
  → 潜扩散去噪器（Transformer，10步DDPM）：条件=历史H + CLIP文本嵌入 + 噪声步t → 预测干净z₀
  → VAE解码器重建未来帧 → 自回归滚动拼接 → 任意长度实时生成（>300fps）
  → 空间控制：潜噪声优化（梯度下降）或RL策略（PPO，潜噪声作为动作空间）
claims:
  - "DART在BABEL时序运动组合任务上以FID 3.79超越离线SOTA FlowMDM(5.81)，同时实现334fps实时在线生成"
  - "潜噪声空间RL策略在目标到达任务上实现100%成功率和0.59cm Goal Error，优于DNO(4.24cm)和OmniControl(7.79cm)"
  - "Scheduled training策略将长序列自回归生成的FID从8.08降至3.79，有效抑制分布漂移"
pdf_ref: paperPDFs/Motion_Generation_Text_Speech_Music_Driven/ICLR_2025/2025_DART_A_Diffusion_Based_Autoregressive_Motion_Model_for_Real_Time_Text_Driven_Motion_Control.pdf
category: Motion_Generation_Text_Speech_Music_Driven
---

# DartControl: A Diffusion-Based Autoregressive Motion Model for Real-Time Text-Driven Motion Control

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [Project Page](https://zkf1997.github.io/DART/) · [ICLR 2025](https://openreview.net/forum?id=DART)
> - **Summary**: DART通过在VAE压缩的运动基元潜空间上训练条件扩散模型，首次实现实时（>300fps）、在线文本驱动的长序列运动生成，并通过潜噪声优化或RL策略统一支持精确空间控制。
> - **Key Performance**:
>   - 时序运动组合FID **3.79** vs. FlowMDM(5.81)，生成速度334fps vs. FlowMDM(31fps)，延迟0.02s vs. 161s
>   - 运动中间帧生成Goal error **0.59cm** vs. DNO(4.24cm) vs. OmniControl(7.79cm)

---

## Part I：问题与挑战

### 真正的卡点

文本驱动运动生成面临**在线实时性**与**空间精确控制**的双重挑战：

- **在线实时性**：现有方法（FlowMDM等）为离线全序列生成，需要预知完整动作时间线，无法响应实时文本流；生成速度慢（31fps），延迟高（>160s），不适用于交互场景
- **空间精确控制**：纯文本条件无法指定精确空间目标（到达特定位置、跟随轨迹、与场景交互）；现有空间控制方法（OmniControl、DNO）在平衡空间约束与文本语义对齐时表现不佳，且仅支持离线短序列
- **长序列稳定性**：自回归生成面临分布漂移和误差累积，导致长序列质量退化

### 输入/输出接口

- **输入**：历史运动种子H（2帧）+ 实时文本提示序列C + 可选空间目标g
- **输出**：连续运动序列M（任意长度），SMPL-X 276维表示（含关节旋转、位置、速度）
- **基元规格**：H=2帧历史 + F=8帧未来，重叠自回归滚动

### 边界条件

- 训练数据：BABEL（帧级文本标注）或HML3D（序列级标注）；AMASS运动捕捉
- 帧级标注时文本-动作对齐精确；序列级标注时多动作描述会导致语义歧义
- 空间控制的优化方法计算开销较大（60帧序列+100步优化≈74s）；RL策略高效但需额外训练

---

## Part II：方法与洞察

### 设计哲学

**"将复杂长序列问题分解为简单短基元问题"**：不直接建模全序列分布，而是将运动分解为重叠的短基元（10帧），在VAE压缩的低维潜空间中用少步扩散（10步DDPM）生成每个基元，通过自回归滚动组合为任意长序列。空间控制则统一为潜空间中的优化问题。

### 核心直觉

**运动基元潜空间的核心直觉**：原始运动数据包含各种噪声和抖动伪影，直接在原始空间训练扩散模型会继承这些伪影。VAE压缩将运动基元映射到紧凑、平滑的潜流形上——**压缩本身就是去噪**，使得扩散模型只需极少步数（10步）即可生成高质量样本。

**自回归基元 vs. 全序列生成**：短基元（8帧未来）的数据分布远比完整长序列简单，扩散模型更容易学习；同时基元天然对应原子动作语义，文本-运动对齐更精确。Scheduled training（渐进引入rollout历史）解决了训练-推理分布不一致导致的长序列漂移。

**潜噪声作为统一控制接口**：DDIM采样将标准高斯噪声确定性映射到合理运动，因此潜噪声空间天然是"合理运动的参数化"——对噪声做梯度优化或RL策略学习，等价于在合理运动流形上搜索满足空间约束的解。

**战略权衡**：

| 优势 | 局限 |
|------|------|
| 实时生成（>300fps），延迟0.02s | 帧级文本标注数据稀缺，限制开放词汇泛化 |
| 统一框架支持文本+空间控制 | 优化方法仍需数十秒；RL需额外训练 |
| 10步扩散即可高质量生成 | 序列级标注时多动作文本会导致语义歧义 |
| Scheduled training保证长序列稳定 | 运动学方法，物理合理性需后处理（如PHC） |

---

## Part III：证据与局限

### 关键实验信号

- **时序运动组合**（BABEL）：FID 3.79（最优）vs. FlowMDM 5.81，转场FID 1.86 vs. FlowMDM 2.39——在线生成质量超越离线SOTA；人类偏好研究中DART在真实性和语义对齐上均优于所有基线（vs. FlowMDM: 53.3% vs. 46.7%）
- **运动中间帧生成**（HML3D）：Goal error 0.59cm vs. DNO 4.24cm vs. OmniControl 7.79cm；Skate 2.98 vs. DNO 5.38——潜基元空间在协调空间控制与文本语义方面显著优于直接在运动空间优化的DNO
- **RL目标到达**：100%成功率 vs. GAMMA 95%；Skate 2.67cm/s vs. GAMMA 5.14cm/s——RL策略在潜空间中高效学习，生成速度240fps
- **消融**：移除VAE导致PJ从0.06→0.20（抖动显著增加）；移除scheduled training导致FID从3.79→8.08（长序列崩溃）；10步扩散与100步性能相当

### 局限与可复用组件

- **局限**：依赖帧级文本标注（BABEL）实现精确控制；运动学生成可能有滑步/穿透等物理伪影
- **可复用**：运动基元VAE + 少步潜扩散的架构范式；潜噪声优化/RL控制框架可迁移到其他基于扩散的生成任务；Scheduled training策略适用于任何自回归生成模型

---

## 本地 PDF 引用

![[paperPDFs/Motion_Generation_Text_Speech_Music_Driven/ICLR_2025/2025_DART_A_Diffusion_Based_Autoregressive_Motion_Model_for_Real_Time_Text_Driven_Motion_Control.pdf]]
