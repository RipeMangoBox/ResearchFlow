---
title: "X-IL: Exploring the Design Space of Imitation Learning Policies"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/imitation-learning
  - task/robot-manipulation
  - diffusion
  - flow-matching
  - sequence-modeling
  - dataset/LIBERO
  - dataset/RoboCasa
  - opensource/full
core_operator: 通过统一可插拔的 X-Block 与 AdaLN 条件化接口，解耦观测表示、骨干、架构和策略头并系统搜索更优模仿学习策略
primary_logic: |
  多模态示范与任务条件 → 由可替换编码器提取 RGB/点云/语言表示，并通过 X-Block 选择 Transformer/Mamba/xLSTM 骨干与 Decoder-only 或 Encoder-Decoder 架构 → 以 BC、扩散或 Flow Matching 生成动作序列并在统一基准上比较 → 找到更强的数据高效机器人模仿学习配置
claims:
  - "Claim 1: On LIBERO, X-BESO with a Mamba backbone achieves 92.7 average success with full demonstrations, exceeding EnerVerse (88.5) and OpenVLA (76.5) [evidence: comparison]"
  - "Claim 2: On RoboCasa, RGB+Point Cloud X-BESO with xLSTM reaches 60.9 average success, outperforming RGB-only xLSTM (53.6) and point-cloud-only xLSTM (32.8) [evidence: comparison]"
  - "Claim 3: At matched model size, Mamba/xLSTM outperform Transformer in all three RoboCasa input settings and dominate the best LIBERO averages; AdaLN-based encoder-decoder also beats decoder-only on most tested architecture comparisons [evidence: ablation]"
related_work_position:
  extends: "CleanDiffuser (Dong et al. 2024)"
  competes_with: "MDT (Reuss et al. 2024c); MaIL (Jia et al. 2024)"
  complementary_to: "N/A"
evidence_strength: strong
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_X_IL_Exploring_the_Design_Space_of_Imitation_Learning_Policies.pdf
category: Embodied_AI
---

# X-IL: Exploring the Design Space of Imitation Learning Policies

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2502.12330)
> - **Summary**: 这篇论文把模仿学习策略拆成“观测表示—序列骨干—整体架构—策略表示”四个可插拔模块，并用统一框架系统搜索组合，从而在 LIBERO 与 RoboCasa 上找到比既有方案更强的机器人模仿学习配置。
> - **Key Performance**: LIBERO 平均成功率最高 **92.7%**（X-BESO + Mamba, 100% demos）；RoboCasa 最高 **60.9%**（X-BESO + xLSTM, RGB+Point Cloud）

> [!info] **Agent Summary**
> - **task_path**: 多模态观测（RGB/点云/语言）+ 离线示范 -> 机器人连续动作序列
> - **bottleneck**: IL 性能由表示、骨干、架构和策略头共同决定，但这些因素过去高度耦合，难以公平比较与组合搜索
> - **mechanism_delta**: 用 X-Block + AdaLN 统一条件注入接口，使 Transformer/Mamba/xLSTM、Decoder-only/Encoder-Decoder、BC/Diffusion/Flow 可在同一 pipeline 下互换
> - **evidence_signal**: 跨 LIBERO 与 RoboCasa 的系统比较和多组消融共同表明，非 Transformer 骨干、AdaLN 条件化架构与多模态输入组合能稳定带来提升
> - **reusable_ops**: [X-Block统一骨干接口, AdaLN条件化编码器-解码器]
> - **failure_modes**: [复杂场景下FPS点云会丢失小物体关键细节, 冻结CLIP在机器人视觉域存在明显域差]
> - **open_questions**: [如何做比简单拼接更强的RGB-点云融合, 这些结论能否从仿真迁移到真实机器人]

## Part I：问题与挑战

这篇论文真正要解决的，不是“再造一个更大的 imitation policy”，而是 **现代 IL policy 的设计空间已经过于庞大且彼此耦合**。  
输入可以是 RGB、点云、语言；骨干可以是 Transformer、Mamba、xLSTM；整体结构可以是 Decoder-only 或 Encoder-Decoder；策略头又可以是 BC、diffusion、flow。过去很多工作只固定一条 pipeline，然后报告最终成功率，因此社区很难回答：

1. 到底是 **骨干** 更重要，还是 **policy representation** 更重要？
2. 多模态收益来自 **模态本身**，还是来自 **更好的融合/条件注入方式**？
3. 新序列模型如 Mamba、xLSTM 在 IL 中是否真的优于 Transformer，还是只是换了别的配套设计？

### 这篇论文的输入/输出接口
- **输入**：离线示范中的多视角 RGB、点云、语言目标，以及历史时序上下文。
- **输出**：机器人连续动作序列（动作 chunk / action tokens）。
- **问题设定**：离线模仿学习，主要评测在仿真机器人操作任务。
- **边界条件**：论文框架支持语言，但主实验重点其实放在 RGB、点云及其组合上；不涉及在线 RL 或真实机器人部署验证。

### 为什么现在做这件事
- 新序列模型（Mamba、xLSTM）已经成熟，但在 IL 里缺少统一、等参、可复现实验。
- 生成式策略头（diffusion / flow）已经替代“单高斯 BC”成为主流候选，需要统一平台比较。
- LIBERO 与 RoboCasa 提供了更能区分设计选择的 benchmark：一个强调数据效率与长时序，一个强调场景变化与泛化压力。

**一句话概括瓶颈**：  
过去 IL 的问题不是“没有足够多的新模块”，而是“没有统一框架把这些模块放到同一坐标系里比较”。

## Part II：方法与洞察

### 方法拆解

X-IL 把 IL pipeline 拆成四个模块，并且每一块都可替换：

1. **Observation Representations**
   - **RGB**：支持 ResNet、FiLM-ResNet、CLIP 等编码器。
   - **Point Cloud**：先做 FPS 下采样，再用 MLP+MaxPool 或 attention encoder 编码。
   - **Language**：用冻结的 CLIP text encoder 得到语言嵌入。

2. **Backbones: X-Block**
   - 核心是一个统一的 **X-Block**。
   - 其中真正处理时序的是 **X-Layer**，可替换成：
     - Transformer
     - Mamba
     - xLSTM
   - 同时用 **AdaLN conditioning** 注入时间/观测上下文。

3. **Architectures**
   - **Decoder-only**：观测和动作一起喂给解码器，结构更简单。
   - **Encoder-Decoder**：先编码多模态观测，再条件化生成动作。
   - 关键点是：作者没有依赖 Transformer 式 cross-attention，而是用 **AdaLN** 作为统一条件接口，因此 Mamba/xLSTM 也能自然进入 Enc-Dec 结构。

4. **Policy Representations**
   - **BC**：简单高斯回归基线。
   - **DDPM/BESO**：扩散式动作生成。
   - **RF / Flow Matching**：流匹配式动作生成。

### 核心直觉

X-IL 的关键改变，不是发明一种全新的 policy loss，而是把 **“表征 → 时序建模 → 动作生成”** 之间的接口标准化了。

#### changed → bottleneck changed → capability changed
- **What changed**：把原本绑死的 IL pipeline 拆成四个正交模块，并用 X-Block + AdaLN 统一它们的连接方式。
- **Which bottleneck changed**：把“只有 Transformer 风格 cross-attention 才能好好做条件建模”的约束，改成“任何骨干都能通过 AdaLN 接收上下文”的更弱约束；同时把实验上的耦合变量拆开，减少“换了骨干但其实也换了别的东西”的混杂。
- **What capability changed**：Mamba、xLSTM 这类非 Transformer 骨干能被公平地纳入 IL；不同模态与 policy head 的组合也可以被系统搜索，因此更容易发现真正有效的配置。

#### 为什么这个设计有效
1. **统一接口降低了比较噪声**  
   过去很难知道性能提升来自哪个模块；现在因为模块可替换且近似等参，观察更接近因果归因。

2. **AdaLN 是关键兼容层**  
   它让条件信息不再依赖 cross-attention 这一特定机制，因此 Mamba/xLSTM 也能进入 Encoder-Decoder 设计。

3. **生成式策略头比 BC 更能表达多峰动作分布**  
   机器人操作中的动作常常不是单峰高斯，扩散/流式表示更适合复杂动作生成。

4. **多模态不是“越多越好”，而是“融合方式决定收益”**  
   点云提供几何，RGB 提供纹理与语义；如果只是简单拼接，收益有限，但在复杂任务上依然能看到互补性。

### 战略取舍表

| 设计维度 | 选项 | 优势 | 代价/风险 | 论文观察 |
|---|---|---|---|---|
| 序列骨干 | Transformer / Mamba / xLSTM | 都能建模长时依赖 | Transformer 计算更重；新骨干需统一条件接口 | 等参下 Mamba/xLSTM 整体更强 |
| 架构 | Decoder-only / Encoder-Decoder | Dec 简洁；Enc-Dec 更利于表征学习与扩展 | Enc-Dec 更复杂 | AdaLN-EncDec 在多数测试任务更优 |
| 输入模态 | RGB / 点云 / RGB+点云 | RGB 强语义；点云强几何；组合可互补 | 点云采样会丢细节；简单拼接融合不充分 | RGB+点云最佳，但点云单独不稳定 |
| 策略表示 | BC / Diffusion / Flow Matching | BC 简单；后两者更能表达复杂动作分布 | 生成式头需要采样推理 | BESO/RF 通常优于 BC，少步推理也可用 |

## Part III：证据与局限

### 关键证据信号

- **比较信号：统一框架真的找到了更强组合**  
  在 LIBERO 上，X-BESO + Mamba 达到 **92.7%** 平均成功率，超过 EnerVerse 的 88.5；在 20% 数据设置下，xLSTM 版本也能到 **74.9%**，说明收益不只是来自更多数据，而是来自更好的设计组合。

- **跨基准信号：结论不只在单一 benchmark 成立**  
  RoboCasa 更强调场景变化和泛化压力。这里 RGB+Point Cloud 的 X-BESO + xLSTM 达到 **60.9%**，明显高于 RGB-only 的 53.6，也高于点云-only 的 32.8，说明多模态组合有价值，但也暴露出点云表示本身并不稳定。

- **机制信号：提升有可解释的因果来源**  
  - 等参比较下，Mamba/xLSTM 普遍优于 Transformer。  
  - AdaLN 的 Encoder-Decoder 在多数对比任务上优于 Decoder-only。  
  - 微调的 FiLM-ResNet 明显好于冻结 CLIP，说明机器人域适配很关键。  
  - attention point-cloud encoder 好于 max-pooling encoder，说明几何结构建模方式重要。

- **实用信号：少步推理可行，但 flow 并未明显更快**  
  在 RoboCasa 的 TurnOnStove 上，BESO 和 RF 在少量 inference steps 下已有较好表现；但由于动作维度不高，flow-matching 的速度优势并不明显。

### 局限性

- **Fails when**: 任务依赖小物体、按钮、插入位姿等细粒度几何细节，而点云又经过 FPS 均匀采样时，关键区域很容易被稀释；RoboCasa 的复杂场景里，点云-only 在这类任务上表现尤其差。
- **Assumes**: 依赖离线人类示范、仿真环境中的 RGB/点云观测，以及可任务化微调的视觉编码器；此外，系统搜索设计空间本身需要不小的训练成本（多骨干 × 多架构 × 多策略头 × 3 seeds × 100/200 epochs）。
- **Not designed for**: 真实机器人部署验证、在线纠错或 RL fine-tuning、foundation-scale VLA 预训练比较；尽管支持语言输入，但本文并没有系统验证语言条件泛化。

### 可复用组件

- **X-Block**：把不同序列骨干放进统一接口，适合做 backbone ablation。
- **AdaLN-conditioned Encoder-Decoder**：适合把非 attention 骨干也纳入条件生成式 policy。
- **多模态 IL 对比模板**：同一代码库下比较 RGB / 点云 / 融合输入，减少实验不公平。
- **点云 attention encoder**：相对 max-pooling 更适合保留几何结构。

**总结一句**：  
这篇论文最有价值的地方，不只是报了几个更高分，而是指出了一个更本质的事实——**IL 的性能跃迁往往来自“接口设计与模块组合”而不是单一新模型名词**。如果你在做 embodied policy，这篇 paper 更像一张“如何系统做设计搜索”的路线图。

## Local PDF reference

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_X_IL_Exploring_the_Design_Space_of_Imitation_Learning_Policies.pdf]]