---
title: "CLAM: Continuous Latent Action Models for Robot Learning from Unlabeled Demonstrations"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/robot-imitation-learning
  - task/learning-from-observation
  - continuous-latent-action
  - inverse-dynamics-model
  - joint-training
  - dataset/DMControl
  - dataset/MetaWorld
  - opensource/full
core_operator: "用连续潜在动作替代离散动作码，并在自监督动力学预训练中联合训练动作解码器，把无动作标签演示稳定地映射为可执行控制信号。"
primary_logic: |
  大规模无动作标签观测序列 + 少量非专家动作标签 → 用逆/前向动力学自监督学习连续潜在动作，并通过联合动作解码器把潜在空间约束到真实动作流形 → 用潜在动作重标注无标签专家轨迹并训练潜在动作策略，输出可执行机器人控制
claims:
  - "在 DMControl 的状态输入实验中，TF-CLAM 在 HalfCheetah/Hopper 上达到 0.72/0.81 的归一化回报，显著高于 VPT 的 0.32/0.41，并接近或超过使用专家动作标签的 BC-Expert [evidence: comparison]"
  - "在 MetaWorld 的消融实验中，仅当连续潜在动作与联合动作解码器训练同时使用时，平均任务成功率才从离散或非联合训练的 16%-23% 提升到约 74% [evidence: ablation]"
  - "在 4 个真实 WidowX 任务上，ST-CLAM 分别取得 7/10、8.5/10、8/10、4/10 的成功率，均优于 VPT 与 LAPA，且训练中未使用专家动作标签 [evidence: comparison]"
related_work_position:
  extends: "LAPO (Schmidt and Jiang, 2023)"
  competes_with: "VPT (Baker et al., 2022); LAPA (Ye et al., 2024)"
  complementary_to: "ACT (Zhao et al., 2023); Diffusion Policy (Chi et al., 2024)"
evidence_strength: strong
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_CLAM_Continuous_Latent_Action_Models_for_Robot_Learning_from_Unlabeled_Demonstrations.pdf
category: Embodied_AI
---

# CLAM: Continuous Latent Action Models for Robot Learning from Unlabeled Demonstrations

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2505.04999), [Project/Code](https://clamrobot.github.io)
> - **Summary**: 本文提出 CLAM，用“连续潜在动作 + 联合动作解码器”把大量无动作标签演示转成可执行监督，从而只靠少量非专家带标签数据也能学到高性能机器人策略。
> - **Key Performance**: DMControl 上 HalfCheetah/Hopper 达到 0.72/0.81；真实 WidowX 四个任务达到 7/10、8.5/10、8/10、4/10。

> [!info] **Agent Summary**
> - **task_path**: 无动作标签观测序列 + 少量非专家动作标签 -> 机器人连续控制动作
> - **bottleneck**: 连续控制需要细粒度动作表征，但仅靠未来重建学出的潜在动作既容易被离散化损伤表达力，也不一定能被少量带标签数据可靠解码为真实动作
> - **mechanism_delta**: 把 latent action 从离散 VQ 改成连续表示，并在 LAM 预训练时联合更新动作解码器，使潜在空间从一开始就被约束为“既能解释转移、又能落地执行”
> - **evidence_signal**: MetaWorld 消融中，continuous + joint training 将平均成功率从 16%-23% 提升到约 74%
> - **reusable_ops**: [continuous-latent-relabeling, joint-latent-to-action-grounding]
> - **failure_modes**: [low-labeled-coverage hurts precise manipulation, grounding degrades under unseen action/state regions]
> - **open_questions**: [can grounding work with much less labeled play data, how to bridge human-video-to-robot embodiment mismatch]

## Part I：问题与挑战

### 1) 这篇论文真正要解决什么问题？
论文瞄准的是机器人模仿学习中的一个核心扩展性瓶颈：**高质量动作标签太贵**。  
现实里最容易大规模拿到的是视频或观测序列，但现代控制策略训练又通常需要动作监督。于是问题变成：

- 我们有很多 **无动作标签** 的演示；
- 只有很少 **带动作标签** 的数据，而且这些标签甚至可以不是专家数据；
- 能不能仍然学出一个可执行、性能高的控制策略？

### 2) 真正瓶颈不是“没有标签”，而是“伪标签既要可表达，又要可执行”
作者的关键判断很准确：  
过去方法失败，不只是因为缺少动作标签，而是因为它们学到的 latent action 往往不满足连续控制的双重要求：

1. **要足够细**：能表达 manipulation / locomotion 里的微小动作差异。  
2. **要可落地**：能被少量真实动作数据稳定地映射回环境动作。

已有路线各有缺陷：

- **监督式 IDM / VPT 路线**：latent 不存在，直接学真实动作，容易 ground，但性能上限被少量带标签数据卡死。
- **离散 latent action 路线（如 VQ）**：能从无标签数据学伪动作，但把连续控制压成离散 codebook，会损失细粒度动作几何。
- **只做重建、不做 grounding**：latent space 可能对预测下一帧有用，但未必适合解码成真实机器人动作。

### 3) 为什么现在值得解决？
因为机器人领域越来越不缺“观测”，而越来越缺“人工动作标签”：

- 大量离线 replay / play data 容易收集；
- 真正昂贵的是专家遥操作标签；
- 如果能把“观测-only 数据”转成有效监督，机器人学习的数据规模就能更接近视觉/NLP 的预训练范式。

### 4) 输入 / 输出接口与边界条件
**输入：**

- `D_unlabeled`：大量无动作标签观测序列，质量混杂
- `D_labeled`：少量带动作标签数据，可来自 random / play policy
- `D_unlabeled-expert ⊆ D_unlabeled`：其中一部分是专家级观测序列，但无动作标签

**输出：**

- 一个 latent action policy：`o -> z`
- 一个 action decoder：`z -> a`
- 合起来形成可执行控制策略

**边界条件：**

- 单任务、单 embodiment、离线 setting
- 需要无标签数据中确实包含可模仿的 expert subset
- 还**不是**互联网人类视频到机器人动作的跨 embodiment 学习

---

## Part II：方法与洞察

### 方法主线
CLAM 分两阶段：

#### Stage 1：先学潜在动作空间
用一个 **latent IDM** 从相邻观测中推断潜在动作 `z_t`，再用 **latent FDM** 结合当前观测和 `z_t` 预测下一观测。  
训练信号来自未来观测重建。

但作者做了两个关键改动：

1. **潜在动作改为连续，而不是离散 VQ code**
2. **动作解码器与 LAM 联合训练**
   - 用少量带标签的 `D_labeled`
   - 学 `p(a_t | z_t)`，同时反向约束 IDM 学到更容易 ground 的 latent

这样 latent space 不再只是“对重建有用”，而是“对重建有用且对执行友好”。

#### Stage 2：再把无标签 expert 轨迹变成监督
用训练好的 latent IDM 给 `D_unlabeled-expert` 打上潜在动作标签，然后训练一个 latent policy：

- 输入观测 `o_t`
- 输出潜在动作 `z_t`

测试时再通过动作解码器把 `z_t` 解成真实动作 `a_t` 执行。

---

### 核心直觉

#### 什么变了？
从 prior work 到 CLAM，真正变化的是两个“因果旋钮”：

- **动作表示**：离散 codebook → 连续 latent manifold
- **预训练约束**：只有未来重建 → 未来重建 + 动作可解码性

#### 哪个信息瓶颈被改变了？
以往离散 latent action 方法，本质上把连续控制的动作分布塞进一个量化后的有限词表里。  
这对游戏等离散动作场景还合理，但对机器人连续控制会带来：

- 量化误差
- 动作细节丢失
- 精细操作不可分

同时，只靠重建目标学出来的 latent，可能是“预测友好”的，但不是“执行友好”的。  
CLAM 的联合动作解码器训练，等于给 latent space 增加了一个额外结构约束：  
**它必须贴近真实动作流形，至少在少量带标签样本覆盖的区域里可被解码。**

#### 为什么这个设计有效？
因为它同时解决了两个相互冲突的需求：

- **连续 latent** 保住表达力
- **联合 grounding** 保住可执行性

于是 Stage 2 中的 latent relabeling 才真正能拿来训练策略，而不是生成一堆“预测上有意义、控制上没法落地”的伪标签。

#### 能力上发生了什么变化？
结果上，CLAM 把能力边界从：

> “只能依赖少量 action-labeled expert demos 学策略”

推进到：

> “可以先用大量 observation-only 数据学动作结构，再用少量非专家带标签数据做 grounding，最后从无标签 expert 轨迹中学出接近专家的控制策略”

这就是它相对 VPT / LAPO / LAPA 的真正跃迁。

### 战略权衡

| 设计选择 | 改变的约束/分布 | 带来的能力 | 代价 |
| --- | --- | --- | --- |
| 连续潜在动作 | 从离散量化 codebook 变为连续动作流形 | 保留细粒度控制差异，更适合 manipulation / locomotion | latent 更难直接对齐到真实动作 |
| 联合动作解码器训练 | latent 不再只为重建服务，还必须可被解码执行 | 少量 play/random 标签也能完成 grounding | 仍需要一定带标签状态-动作覆盖 |
| 用 IDM 重标注无标签 expert 轨迹 | 把 observation-only 数据转成 imitation 监督 | 策略训练规模可随无标签数据增长 | 依赖数据中确实存在 expert-like 轨迹 |
| 复用 IDM 视觉编码器 | 预训练不仅学动作，也学视觉表征 | 图像输入下对下游策略有额外迁移收益 | 依赖 backbone 质量和预训练稳定性 |

### 一个值得注意的细节
作者还指出：训练 LAM 本身也相当于一种 **self-supervised representation learning**。  
因此在图像输入实验里，CLAM 的优势不只是“标了 latent action”，还可能来自：

- IDM 视觉编码器预训练带来的视觉迁移
- 更适合下游 BC 的视觉特征

这也是它在图像任务上有时能接近甚至超过 BC-Expert 的一个合理解释。

---

## Part III：证据与局限

### 关键实验信号

#### 1. 比较信号：CLAM 在低标签 regime 下显著优于现有方法
**DMControl（状态输入）** 中，TF-CLAM 在 HalfCheetah/Hopper 上达到 **0.72 / 0.81**，而 VPT 只有 **0.32 / 0.41**。  
这说明当 `|D_labeled| << |D_unlabeled|` 时，**能利用无标签数据学动作结构** 比只靠少量带标签数据监督 IDM 更重要。

#### 2. 消融信号：连续 latent + joint training 缺一不可
这是论文最强的因果证据。

MetaWorld 消融表明：

- 离散 latent、无 joint training：约 **16%**
- 连续 latent、无 joint training：约 **23%**
- 连续 latent、联合动作解码器：约 **74%**

结论非常直接：  
**连续 latent 解决表达力问题，联合训练解决 grounding 问题，二者必须同时成立。**

#### 3. 比较信号：图像控制任务里对离散 latent 方法形成代际优势
在 MetaWorld 图像输入实验中，ST-ViViT-CLAM 的平均成功率约 **76%**，相对最强基线达到 **2–3×** 提升。  
尤其相较 LAPO/LAPA 这类离散 latent 路线，差距很大，说明论文的核心判断——**“连续控制不应被离散化”**——是被实验支持的。

#### 4. 扩展信号：真实机器人上也成立
在真实 WidowX 上，ST-CLAM 在四个任务上分别达到：

- Reach Block：**7/10**
- Push Button：**8.5/10**
- Close Microwave：**8/10**
- Put in Pot + Slide：**4/10**

并且都优于 VPT / LAPA / BC-AL。  
这说明 CLAM 不是只在模拟器里“重建好看”，而是真的把 latent grounding 成了可执行控制。

#### 5. 标度信号：CLAM 随无标签数据和少量 play labels 一起扩展
论文还展示了两点很关键的 scaling 迹象：

- 随 `|D_unlabeled-expert|` 增长，latent policy 继续提升
- 随 `|D_labeled|` 增长，CLAM 持续变强，而 BC 很快平台化

这正是该方法的实用意义：  
**它把昂贵的数据需求从“专家动作标签”转移成了“少量非专家标签 + 大量无标签观测”。**

### So what：它相对 prior work 的能力跃迁到底在哪？
一句话概括：

**CLAM 把“从无动作标签演示学控制”从概念验证推进到了连续控制可用阶段。**

相比 prior work，它的跳跃不是简单涨点数，而是把数据范式改了：

- 不再要求 action-labeled expert demos
- 不再依赖离散 latent 去近似连续动作
- 允许用 random / play labeled data 完成 grounding
- 在 manipulation 和真实机器人任务上都能跑通

### 局限性

- **Fails when**: 带标签数据对状态-动作空间覆盖不足时，动作解码器难以准确 grounding，精细任务会明显掉点；论文也明确提到像 Shelf Place 这类精确操作在低标签量下更脆弱。
- **Assumes**: 需要单任务、单 embodiment、离线数据设置；需要无标签数据中存在可模仿的 expert subset；还需要一小部分但尽量多样的带动作标签数据，即使这些标签来自 random/play policy。
- **Not designed for**: 互联网人类视频到机器人动作的直接迁移、跨视角/跨 embodiment 泛化、完全零动作标签设置、多任务开放世界部署。

### 资源与复现依赖
虽然它避免了专家动作标签，但并不是“零成本”：

- 真实机器人实验仍需要约 **50k play transitions**
- 每个任务还需要无标签 expert demonstrations
- 图像模型训练依赖 GPU（文中使用 A6000/V100/A100）
- 方法效果对 `D_labeled` 的覆盖质量较敏感

所以它更准确的定位是：  
**降低最贵的标签成本，而不是消灭所有监督成本。**

### 可复用组件
这篇论文最值得迁移的，不只是一个具体模型，而是三类操作符：

1. **continuous latent relabeling**：先学连续潜在动作，再给无标签轨迹打伪标签  
2. **joint latent-to-action grounding**：在预训练期就把 latent 约束为可执行  
3. **IDM/FDM pretraining as representation learning**：把动作建模和视觉特征学习统一起来

这些组件都可以与更强策略头结合，例如作者自己也指出可与 ACT、Diffusion Policy 之类架构互补。

## Local PDF reference

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_CLAM_Continuous_Latent_Action_Models_for_Robot_Learning_from_Unlabeled_Demonstrations.pdf]]