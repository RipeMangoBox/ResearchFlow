---
title: "Diffusion-Based Imaginative Coordination for Bimanual Manipulation"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/bimanual-manipulation
  - task/robot-imitation-learning
  - diffusion
  - state-token
  - action-conditioned-attention
  - dataset/ALOHA
  - dataset/RoboTwin
  - opensource/no
core_operator: 将未来多帧视频潜变量作为双臂共享的“想象共识”，在扩散策略中与动作联合训练，并用单向注意力让训练受益于视频预测而推理只生成动作
primary_logic: |
  当前单目图像与机器人本体状态 → 编码观测并对未来动作块和多帧视频latent进行联合扩散去噪，且视频token仅单向读取历史动作 → 输出可跳过视频分支的双臂动作序列
claims:
  - "在 ALOHA 上，该方法取得 71.9% 平均成功率，相比 ACT 的 47.0% 提升 24.9 个百分点 [evidence: comparison]"
  - "在 RoboTwin 的 16 个任务上，该方法取得 56.3% 平均成功率，超过 RDT-1B 的 51.1% [evidence: comparison]"
  - "加入未来视频预测后，ALOHA 提升 7.3 个百分点，RoboTwin 的 Seq-coordinate 任务提升 6.2 个百分点，而 Sync-bimanual 任务略降 0.7 个百分点，说明收益主要来自顺序协同 [evidence: ablation]"
related_work_position:
  extends: "GR-1 (Wu et al. 2024)"
  competes_with: "ACT (Zhao et al. 2023); RDT-1B (Liu et al. 2024)"
  complementary_to: "3D Diffusion Policy (Ze et al. 2024); FiLM (Perez et al. 2018)"
evidence_strength: strong
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Diffusion_Based_Imaginative_Coordination_for_Bimanual_Manipulation.pdf
category: Embodied_AI
---

# Diffusion-Based Imaginative Coordination for Bimanual Manipulation

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2507.11296)
> - **Summary**: 论文把“未来视频想象”作为双臂共享的协调中介，在扩散式动作策略训练中联合预测未来动作与未来视觉潜变量，从而提升双臂时序协同，同时保持推理阶段的高控制频率。
> - **Key Performance**: ALOHA 平均成功率 71.9%（较 ACT +24.9pt）；真实世界 4 任务平均成功率 60.0%（较 ACT +32.5pt）

> [!info] **Agent Summary**
> - **task_path**: 单目RGB + 本体状态 / 双臂模仿学习 -> 未来动作chunk
> - **bottleneck**: 双臂操作不仅动作空间高维且多模态，更关键是仅靠当前观测难以提前对齐两只手的时序依赖
> - **mechanism_delta**: 在动作扩散训练中加入多帧未来视频latent作为共享未来表征，并用“动作独立、视频读动作”的单向注意力把协调信息留在训练里、把推理开销留在训练外
> - **evidence_signal**: 跨 ALOHA、RoboTwin 与真实机器人均优于 ACT/RDT-1B，且去掉视频预测后顺序协同任务下降最明显
> - **reusable_ops**: [多帧latent未来预测, 动作到视频的单向条件注意力]
> - **failure_modes**: [松耦合同步任务收益有限甚至略降, 小物体或遮挡场景受限于单视角与轻量视觉编码器]
> - **open_questions**: [如何选择更有信息量的关键未来帧而非均匀采样, 如何加入触觉/多视角/闭环重规划而不牺牲实时性]

## Part I：问题与挑战

这篇论文解决的是**双臂模仿学习中的隐式协调**问题。

### 1) 任务接口
输入是当前时刻的：
- 单视角 RGB 图像
- 机器人本体状态（关节、夹爪等）

输出是：
- 未来一段时间的双臂动作序列（action chunk）

训练时额外引入：
- 未来多帧视觉状态的潜变量表示

### 2) 真正瓶颈是什么
作者强调的核心不是“动作生成不够强”，而是：

1. **双臂行为高度多模态**  
   同一任务可由左手主导、右手主导、或双手共同完成。

2. **协调是时空耦合的，不是静态共享观测就能解决**  
   很多任务需要一只手先做、另一只手再接上；只看当前帧，策略只能反应式决策，缺少“接下来会发生什么”的共识。

3. **视频预测虽有潜力，但常见做法太重**  
   如果直接把未来图像/视频也放进推理链路，控制频率会受影响，不适合实时机器人执行。

### 3) 为什么现在值得做
因为两类技术条件成熟了：
- **扩散策略**已经证明能更好建模机器人动作的多模态性；
- **视频 tokenizer**让“预测未来”不必在像素空间硬做，可以在压缩 latent 空间里做，成本更可控。

所以，这篇论文的判断是：  
**双臂协调的关键缺口，不在更复杂的规则，而在给两只手一个共享的“未来想象”。**

## Part II：方法与洞察

### 方法骨架

整体是一个**统一的 Transformer-based diffusion 框架**，联合做两件事：
- 未来动作预测
- 未来视频 latent 预测

具体分三步：

1. **观测编码**  
   当前 RGB 用 ResNet-18 提特征，再与 proprio token 拼接，送入 Transformer encoder。

2. **未来状态压缩表示**  
   未来多帧不是直接预测像素，而是先用预训练 **Cosmos tokenizer** 压成视频 latent token，再做 patch 化与时空位置编码。

3. **联合扩散去噪**  
   用共享 decoder 同时去噪动作 token 与未来视频 token；最后分别接动作头和视频头。

### 核心直觉

传统双臂策略的协调依据是：

- “两只手都看到了同一个当前观测”

这不够，因为它只提供**当前状态共识**，不提供**未来结果共识**。

这篇论文真正改的因果旋钮是：

- 从“只根据当前观测生成动作”
- 变成“训练时让策略同时对未来动作和未来视觉后果负责”

这带来三个层面的变化：

1. **信息瓶颈变化**  
   从只编码当前状态，变成编码“未来会往哪里走”的轨迹摘要。

2. **约束变化**  
   多帧 future latent 迫使模型学习跨时间的一致性，而不是只做一步短视预测。

3. **部署图结构变化**  
   通过单向注意力，动作分支不依赖视频分支，所以推理时可以跳过视频预测，只保留训练时学到的因果先验。

也就是说，它不是把视频预测当最终产物，而是把它当作一种**协调正则化/未来共识载体**。

### 为什么这个设计有效

#### a) 多帧 latent 预测优于单帧像素预测
因为双臂协调主要依赖：
- 物体关系怎么变化
- 两只手先后顺序怎么接
- 环境状态将如何演化

这些是**轨迹级语义**，不是像素细节本身。  
所以用压缩 latent 做多帧预测，更接近控制所需的信息。

#### b) 单向注意力优于全连接或完全解耦
作者的注意力设计是：
- 动作 token 只看动作 token
- 视频 token 看自己 + 历史动作 token

这样做的好处是：
- 训练时，视频分支能学到“动作导致什么视觉后果”的因果关系；
- 推理时，动作生成不需要等待视频 token，因此速度不掉。

#### c) 未来预测的收益主要体现在顺序协同
如果任务要求一手先铺垫、另一手再接续，那么未来想象能显著帮助。  
如果任务本身是两只手并行、弱耦合地各做各的，未来预测反而可能引入多余依赖。

### 策略权衡

| 设计选择 | 改变了什么 | 直接收益 | 代价/边界 |
|---|---|---|---|
| 多帧 future latent 而非像素 | 把未来建模从像素重建转成任务相关状态摘要 | 更强的长时程协调、计算更省 | 依赖 tokenizer 质量；不追求视频逼真度 |
| 视频与动作联合训练 | 给动作学习加入环境动态先验 | 提升 Seq-coordinate 类任务 | 若任务松耦合，额外预测可能带来干扰 |
| 单向动作条件注意力 | 保留动作→未来视觉因果，而切断未来视觉→动作推理依赖 | 训练有效、推理高频 | 信息流更受控，设计上比 full attention 更偏任务化 |
| action chunk 开环执行 | 降低控制调用频率压力 | 执行更平滑、适合高频控制 | 响应性下降，不适合强扰动闭环修正 |

## Part III：证据与局限

### 关键证据

- **比较信号：ALOHA**
  - 71.9% 平均成功率，显著高于 ACT 的 47.0%。
  - 说明该方法在细粒度双臂操作中，确实优于只做动作生成的强基线。

- **比较信号：RoboTwin**
  - 16 个任务平均 56.3%，高于 RDT-1B 的 51.1%。
  - 且在作者划分的 **Seq-coordinate** 类任务中优势最大，说明提升主要来自时序协调而非单纯动作拟合。

- **机制信号：消融**
  - 去掉视频预测后，ALOHA 下降 7.3pt；Seq-coordinate 任务下降 6.2pt。
  - 但 Sync-bimanual 略降 0.7pt，说明“未来想象”不是通用增益，而是**主要解决顺序依赖**。

- **部署信号：真实机器人**
  - 4 个真实任务平均成功率 60.0%，高于 ACT 的 27.5%。
  - 同时控制频率 35.8 Hz，高于 Diffusion Policy 的 15.4 Hz，也略高于 ACT 的 33.9 Hz，支持“训练用视频、推理去视频”的效率主张。

### 局限性

- **Fails when**: 松耦合同步任务中，未来预测可能引入不必要的跨臂依赖；对小物体、遮挡物体、精细旋转任务（如文中的 Coffee Stir）表现仍弱；单视角输入下对局部细节和遮挡恢复有限。  
- **Assumes**: 依赖人类遥操作示教数据；依赖预训练 Cosmos tokenizer 与 ResNet-18 视觉骨干；采用 action chunk 的开环执行；未来帧使用均匀采样而非任务关键帧选择。  
- **Not designed for**: 强动态、强扰动、需要在线重规划/闭环修正/触觉反馈融合的场景；也不是为了生成高保真视频，而是为了学任务相关动态。

### 复现与资源依赖

- 关键外部依赖是 **Cosmos video tokenizer**，这会影响可复现性与迁移成本。
- 推理虽只用动作分支，但训练仍需联合视频预测，工程复杂度高于纯动作策略。
- 论文文本声称模型和代码公开，但给定材料中未提供可验证链接，因此按保守标准记为 `opensource/no`。

### 可复用组件

1. **多帧 latent future prediction**：适合把“未来状态约束”注入控制策略，而不必做昂贵像素预测。  
2. **单向动作条件注意力**：适合训练时利用未来/辅助分支，推理时裁剪掉辅助分支。  
3. **未来想象作为协调接口**：适合多执行器、多阶段、强时序依赖的 embodied policy 设计。

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Diffusion_Based_Imaginative_Coordination_for_Bimanual_Manipulation.pdf]]