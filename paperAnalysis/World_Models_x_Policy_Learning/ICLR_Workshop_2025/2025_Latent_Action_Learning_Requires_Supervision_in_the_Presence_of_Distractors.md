---
title: "Latent Action Learning Requires Supervision in the Presence of Distractors"
venue: ICML
year: 2025
tags:
  - Embodied_AI
  - task/imitation-learning
  - task/latent-action-learning
  - multi-step-idm
  - latent-temporal-consistency
  - semi-supervision
  - "dataset/Distracting Control Suite"
  - opensource/full
core_operator: 用多步逆动力学与潜空间时序一致性替代量化重建式潜动作学习，并用极少量真实动作标签把潜动作锚定到控制相关变化上
primary_logic: |
  含动作相关干扰的观测序列对 + 极少量动作标签 → 共享视觉编码器提取表示，多步IDM生成高维连续潜动作，FDM在潜空间做时序一致性预测，并对有标签样本线性监督 → 得到更可解码、对新干扰更稳的潜动作与下游行为克隆策略
claims:
  - "在 Distracting Control Suite 的四个离线控制环境中，原始 LAPO 在有干扰时的下游表现明显落后于 IDM 和同标签预算的 BC [evidence: comparison]"
  - "LAOM 通过去量化、引入多步 IDM、改用潜空间时序一致性并配合增强，把潜动作对真实动作的线性 probe 质量提升约 8× [evidence: ablation]"
  - "仅复用约 2.5% 数据的真实动作标签在 LAOM 训练期做监督，可将平均归一化回报提升到 0.44，较无监督 LAOM 平均提升约 4.2×，且对未见干扰比 IDM 泛化更好 [evidence: comparison]"
related_work_position:
  extends: "LAPO (Schmidt & Jiang, 2023)"
  competes_with: "LAPO (Schmidt & Jiang, 2023); IDM relabeling (Baker et al., 2022"
  complementary_to: "N/A"
evidence_strength: moderate
pdf_ref: paperPDFs/World_Models_x_Policy_Learning/ICLR_Workshop_2025/2025_Latent_Action_Learning_Requires_Supervision_in_the_Presence_of_Distractors.pdf
category: Embodied_AI
---

# Latent Action Learning Requires Supervision in the Presence of Distractors

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2502.00379), [Code](https://github.com/dunnolab/laom)
> - **Summary**: 论文证明：一旦视频中存在与动作相关的背景变化、相机抖动、颜色变化等干扰，纯无监督 latent action learning 会把“更容易解释像素变化”的外生因素误当成动作，因此必须在 LAM 训练阶段用极少量真实动作标签做锚定。
> - **Key Performance**: LAOM 将潜动作线性 probe 质量提升约 8×；仅用约 2.5% 动作标签时，LAOM+supervision 的平均 normalized return 达 0.44，较无监督 LAOM 平均提升约 4.2×

> [!info] **Agent Summary**
> - **task_path**: 含干扰的视频观测序列 + 少量动作标签 -> 潜动作学习 -> 解码为真实动作 -> 行为克隆控制策略
> - **bottleneck**: 无监督 LAM 在干扰存在时会优先编码更容易解释视频变化的背景/相机等外生因素，而非真实控制信号
> - **mechanism_delta**: 用多步 IDM + 潜空间时序一致性替代量化+像素重建，并把同一小撮真实动作标签前置到 LAM 训练阶段做线性监督
> - **evidence_signal**: DCS 四任务上，LAOM+supervision 在所有标签预算下整体优于 LAPO、LAOM、IDM 和同预算 BC，且对未见干扰泛化更强
> - **reusable_ops**: [multi-step IDM, latent-space temporal consistency, sparse action-grounding]
> - **failure_modes**: [cross-embodied 预训练收益有限, latent representation 仍保留大量背景等控制无关信息]
> - **open_questions**: [能否用手部轨迹等代理监督替代真实动作标签, 在真实互联网视频而非 DCS 合成干扰上是否仍成立]

## Part I：问题与挑战

这篇论文真正击中的问题，不是“latent action 能不能学”，而是：

**当观测变化里混有大量与动作相关但并非由 agent 控制的干扰时，latent action 是否还可被识别为真实动作？**

过去的 LAPO/LAM 路线大多默认数据近似“无干扰”：相邻帧之间最主要的变化来自 agent 的真实动作。这在静态背景机器人数据里还能成立，但在 web-scale 视频或复杂现实场景里并不成立。背景人群、镜头抖动、光照与颜色变化，都可能比真实动作更容易解释像素变化。

### 真瓶颈是什么？

真瓶颈是 **control signal 的可辨识性**，不是简单的表示容量不足。

- **输入**：仅有观测序列 \(o_t, o_{t+1}\)（或 \(o_{t+k}\)），外加极少量真实动作标签。
- **输出**：可被解码为真实动作的 latent action，以及能用于行为克隆的预训练策略。
- **困难点**：如果训练目标鼓励“谁最能解释下一帧就编码谁”，那么模型会优先编码背景视频、相机扰动等外生变化，而不一定编码真实动作。

### 为什么现在要解决？

因为 latent action learning 的愿景就是吃下海量 observation-only 视频数据做 embodied pretraining；而一旦离开“干净数据”，这个愿景首先遇到的就是**干扰导致的错对齐**。如果这一点不解决，LAM 在现实视频上的可用性会被高估。

### 边界条件

本文实验是一个受控离线设置，而不是开放世界真实互联网视频：

- 基准：Distracting Control Suite，4 个控制任务
- 干扰：动态背景视频、相机抖动、agent 颜色变化
- 数据规模：每环境 5k trajectories、约 5M transitions
- 评估：先学 LAM，再做 latent BC，再用少量标签训练 action decoder；标签预算从 2 到 128 条轨迹

所以它回答的是：**在可控但足够麻烦的视觉干扰下，LAM 还是否可靠。**

## Part II：方法与洞察

作者没有推翻经典三阶段 pipeline，而是重做了 **LAM 本身的归纳偏置**。核心思想很直接：

> 在有干扰时，不要过早强迫 latent action “极简”；先确保它至少保住真实动作，再用少量标签把它拉回控制相关方向。

### 从 LAPO 到 LAOM

LAOM 相比 LAPO 的关键改动有 5 个：

1. **去掉量化**
   - 作者发现 VQ/FSQ 量化会明显伤害 latent action 质量。
   - 原因不是单纯训练不稳，而是：在有干扰时，硬信息瓶颈会迫使模型压缩“最容易解释动态的因素”，这往往是噪声而不是真动作。

2. **引入 multi-step IDM**
   - 不只看 \(o_t \to o_{t+1}\)，而是从 \(o_t, o_{t+k}\) 预测 latent action。
   - 这样更偏向控制相关的跨时变化，而不是短时像素扰动。

3. **显著增大 latent action 维度**
   - 从 128 提到 8192。
   - 这看似反直觉，但作者的判断是：在干扰存在时，先保证“真动作还在 latent 里”比“latent 足够小”更重要。

4. **去掉像素级重建，改为潜空间时序一致性**
   - 不再让 FDM 重建整张下一帧图像。
   - 改成在 encoder latent space 里预测下一时刻表示，避免被迫解释每个像素上的背景变化。

5. **对同一小部分标签加线性动作监督**
   - 这才是论文最关键的因果旋钮。
   - 不是额外引入更多标签，而是把原本只在最后 decoder 阶段用到的少量标签，提前到 LAM 训练阶段就使用。

### 核心直觉

无监督 LAPO 的默认逻辑是：

**量化/重建约束 → 让 latent action 成为“最能解释下一帧变化的压缩码” → 在无干扰数据上，这个压缩码可能接近真动作。**

但在有干扰时，分布变了：

- **变化来源变复杂了**：下一帧不只由 agent 动作决定，也由背景视频、相机、颜色变化决定。
- **错误的瓶颈被放大了**：量化和重建不会自动偏好“控制相关信息”，只会偏好“最解释预测误差的信息”。
- **结果**：latent action 学成了“视频变化摘要”，而不是真动作。

LAOM + supervision 改变的是这个因果链：

**放松错误的信息瓶颈 + 用少量真动作标签提供 grounding → latent action 不再只服务于重建，而被迫对控制变量保真 → 下游 BC 更容易从中恢复真实动作。**

也就是说，这篇论文最重要的洞察不是“多步 IDM 更强”，而是：

> **在干扰存在时，latent action learning 需要 supervision 来解决语义指向问题：到底哪些变化才叫 action。**

### 战略取舍

| 设计改动 | 改变了什么瓶颈 | 能力提升 | 代价/风险 |
|---|---|---|---|
| 去量化 | 去掉过强硬信息瓶颈 | 更容易保住真实动作信息 | latent 不再天然 minimal |
| multi-step IDM | 把短时像素扰动问题改成跨时控制变化问题 | 更偏向 control-relevant 信息 | 依赖更长时序对应关系 |
| 潜空间时序一致性替代像素重建 | 不再强迫模型解释背景每个像素 | 更抗视觉干扰，训练更快 | 训练更容易不稳，需要增强/EMA |
| 大 latent 维度 | 从“强压缩”改成“先保真” | 提高 probe 可解码性 | 会混入更多无关动态，BC 阶段负担更大 |
| 少量动作监督 | 为 latent action 指定语义锚点 | 下游性能和泛化显著提升 | 需要少量标签，且每个 label budget 要单独训练 |

## Part III：证据与局限

### 关键证据链

**信号 1：原始 LAPO 在干扰下确实会失效。**  
在 DCS 四个环境上，含干扰时的 LAPO 被更简单的 IDM 和同标签预算 BC 超过。这说明问题不是“复杂方法一定更强”，而是原有 LAM 目标本身在错误地对齐信息。

**信号 2：LAOM 的结构改动确实改善了 latent action 质量。**  
作者逐项 ablation 显示，去量化、多步 IDM、增大 latent 维度、改成 latent consistency、加入增强，累计把线性 probe 的动作质量改善约 8×。  
但这一步只说明“真动作被保住更多”，还不等于“latent 足够干净、足够好用”。

**信号 3：真正带来能力跃迁的是训练期 supervision。**  
把同一批少量动作标签前置到 LAM 训练阶段后，LAOM+supervision 在所有标签预算上都优于无监督 LAOM。  
最关键指标是：**约 2.5% 标签时平均 normalized return = 0.44**，相对无监督 LAOM 平均提升约 **4.2×**。

**信号 4：它比 IDM 更能泛化到新干扰。**  
在未见过的新背景干扰上，LAOM+supervision 的动作预测优于 IDM。这支持作者的核心论点：  
IDM 只在少量有标签数据上学动作，容易把监督分布记死；而带监督的 LAM 在全量无标签数据上先见过更广的干扰分布。

**信号 5：但它仍没有学到 minimal state。**  
作者训练额外 decoder 重建观测，发现 LAM 表征里仍保留大量背景/颜色等控制无关信息；反而 multi-step IDM 更接近 control-endogenous minimal state。  
这意味着：**LAM 的 latent 在“可解码动作”与“最小控制状态”之间并不等价。**

### 两个最该记住的结果

- **8×**：LAOM 相比原始 LAPO，把潜动作对真实动作的线性 probe 质量提升约 8×。
- **0.44 normalized return @ ~2.5% labels**：说明少量标签前置监督，比“先无监督学完 latent，再末端解码”更值钱。

### 局限性

- **Fails when**: 跨 embodiment 预训练时，收益并没有稳定超过同标签预算的从零开始 BC；当标签极少且没有训练期 grounding、或干扰强到比动作更能解释视频变化时，latent 仍会混入大量噪声。
- **Assumes**: 有一小部分真实动作标签可复用到 LAM 训练期；离线大规模无标签轨迹可得；DCS 风格的合成干扰能代表现实问题的一部分；实验依赖较大算力与工程设置（单卡 H100、每环境约 5M transitions、受控超参调优）。
- **Not designed for**: 完全零标签的互联网视频动作挖掘、在线探索型 RL、证明已学到 control-endogenous minimal state 的理论完备方案。

### 可复用组件

- **multi-step IDM**：适合在有外生干扰时增强控制相关时序信息。
- **latent-space temporal consistency**：避免像素重建把模型拖去解释背景变化。
- **sparse action-grounding**：把原本只用于末端 decoder 的少量标签，提前用于表示学习阶段。
- **“先保真、后压缩”思路**：在干扰强时，先确保真动作留在 latent 里，再谈 minimality。

![[paperPDFs/World_Models_x_Policy_Learning/ICLR_Workshop_2025/2025_Latent_Action_Learning_Requires_Supervision_in_the_Presence_of_Distractors.pdf]]