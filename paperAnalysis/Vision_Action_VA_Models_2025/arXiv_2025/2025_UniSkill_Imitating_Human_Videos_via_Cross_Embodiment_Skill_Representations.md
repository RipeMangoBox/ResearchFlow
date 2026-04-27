---
title: "UniSkill: Imitating Human Videos via Cross-Embodiment Skill Representations"
venue: CoRL
year: 2025
tags:
  - Embodied_AI
  - task/cross-embodiment-imitation
  - task/robot-manipulation
  - diffusion
  - inverse-dynamics
  - forward-dynamics
  - dataset/DROID
  - dataset/LIBERO
  - dataset/H2O
  - "dataset/Something-Something V2"
  - opensource/no
core_operator: 通过逆向技能动力学提取跨帧动态技能，并用扩散式前向图像编辑预测未来帧，把人类/机器人视频压缩为与 embodiment 无关的可执行技能表示
primary_logic: |
  无标注的人类/机器人视频 + 少量带动作机器人轨迹 → 用 ISD 从相隔 k 帧图像中提取技能、用 FSD 以图像编辑方式预测未来帧并逼迫技能只编码动态 → 用机器人轨迹训练技能条件策略，并在测试时将人类提示视频转成技能序列驱动机器人执行
claims:
  - "在真实 tabletop 基准的人类提示上，UniSkill 平均成功率为 0.36，明显高于 GCBC 的 0.11 和 XSkill 的 0.00 [evidence: comparison]"
  - "在 LIBERO 仿真人类提示上，UniSkill 平均成功率为 0.48，高于 GCBC 的 0.09；将 UniSkill 的 FSD 生成帧作为子目标后，GCBC-U 提升到 0.24，说明其技能表示缓解了 embodiment gap [evidence: comparison]"
  - "加入大规模人类视频预训练可将 LIBERO 上的人类提示成功率从 0.19 提升到 0.49，表明无需场景对齐的人类视频也能显著增强跨 embodiment 技能学习 [evidence: ablation]"
related_work_position:
  extends: "LAPO (Schmidt & Jiang 2024)"
  competes_with: "XSkill (Xu et al. 2023); Goal-Conditioned Behavioral Cloning (GCBC)"
  complementary_to: "N/A"
evidence_strength: strong
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_UniSkill_Imitating_Human_Videos_via_Cross_Embodiment_Skill_Representations.pdf
category: Embodied_AI
---

# UniSkill: Imitating Human Videos via Cross-Embodiment Skill Representations

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2505.08787), [Project](https://kimhanjung.github.io/UniSkill)
> - **Summary**: UniSkill 把“看人类视频学机器人动作”改写为“从跨帧变化中提取可执行技能”的问题，用图像编辑式未来帧预测学到跨 embodiment 技能表示，再用该表示直接条件化机器人策略。
> - **Key Performance**: Tabletop 人类提示平均成功率 **36%**，高于 GCBC 的 **11%** 和 XSkill 的 **0%**；LIBERO 人类提示平均成功率 **48%**，高于 GCBC 的 **9%**。

> [!info] **Agent Summary**
> - **task_path**: 人类/机器人提示视频 + 当前机器人观测 -> 机器人动作块
> - **bottleneck**: 缺少可扩展的人机对齐监督，导致表示容易学到外观/场景差异，而不是跨 embodiment 可执行的动态模式
> - **mechanism_delta**: 用 ISD+FSD 的“未来帧图像编辑”监督替代显式配对/聚类对齐，让技能向量只需解释两帧间发生的动态变化
> - **evidence_signal**: 真实世界与 LIBERO 上的人类提示均显著优于 GCBC/XSkill，且加入人类视频预训练后跨 embodiment 表现明显提升
> - **reusable_ops**: [ISD skill encoder, FSD embodiment-preserving future-frame generator]
> - **failure_modes**: [固定技能间隔导致速度不匹配, 视角突变或接触失败后缺乏在线纠错]
> - **open_questions**: [如何自适应学习 skill duration, 如何与语言/VLA 语义推理联合]

## Part I：问题与挑战

这篇论文解决的是 **cross-embodiment imitation**：给定一个只含 RGB 的提示视频，视频里的执行者可以是人类、同类机器人，甚至是另一种机器人，目标机器人要输出动作序列去复现其中的行为。

### 真正的瓶颈是什么？

真正的瓶颈不是“没有人类动作标签”本身，而是：

1. **视频里混着太多 embodiment-specific 信息**  
   人手、机械臂、背景、视角、物体外观会和真正需要迁移的“动作模式”缠在一起。

2. **现有跨 embodiment 方法通常依赖额外对齐约束**  
   比如：
   - 人机同场景/同任务数据
   - 配对演示
   - 手轨迹、2D/3D flow、pose 等显式监督
   - 多视角或特定采集装置

3. **机器人真正需要的是“可执行技能接口”，不是像素级相似**  
   目标不是复原人类画面，而是把“打开、拉出、推动、放置”这类动态模式转成机器人可执行条件。

### 为什么现在值得做？

因为现在同时具备了两类条件：

- **大规模无标注视频数据可得**：Something-Something V2、H2O、DROID、BridgeV2、LIBERO。
- **成熟的视觉基础模块可借用**：单目深度估计、扩散式图像编辑，使“用未来帧预测约束技能表示”变得可行。

### 输入/输出接口与边界条件

- **输入**：提示视频帧序列 + 当前机器人观测。
- **输出**：动作 chunk，由策略逐段执行。
- **训练边界**：
  - 技能表示可用无标注人类/机器人视频学；
  - 真正的控制策略仍需少量 **带动作的机器人轨迹**；
  - 测试时提示视频是未见过的；
  - 不依赖语言指令；
  - 主要面向操作类任务，且默认提示视频与机器人任务在行为层面存在可映射性。

---

## Part II：方法与洞察

UniSkill 的整体思路可以概括为两阶段：

1. **先学一个跨 embodiment 的 skill representation**
2. **再用机器人数据学一个 skill-conditioned policy**

### 方法拆解

#### 1. Universal Skill Representation Learning

核心是两个模块：

- **ISD (Inverse Skill Dynamics)**  
  输入相隔 \(k\) 帧的两张图像，输出技能向量 \(z_t\)。  
  它不是去识别“是谁在动”，而是压缩“这两帧之间发生了什么动态变化”。

- **FSD (Forward Skill Dynamics)**  
  输入当前帧和技能向量，预测未来帧。  
  这里作者没有做普通像素回归，而是把它建成 **image editing**：从当前图像出发，只编辑那些需要变化的部分。

关键实现是：把 **InstructPix2Pix** 的“语言指令”替换成技能向量 \(z_t\)。  
这样一来，如果模型想成功生成未来帧，最有用的信息就必须是“运动变化”，而不是静态背景。

另外，ISD 里引入了 **预测深度** 作为中间表征，目的是削弱纯 RGB 中的外观/材质偏置，让技能更偏向几何与动态。

#### 2. Universal Skill-Conditioned Policy

学好技能后，冻结 ISD，用机器人演示数据抽取机器人自己的 skill token，再训练一个 **diffusion policy**：

- 输入：当前机器人观测 + skill 表示
- 输出：未来一段动作 chunk

这一步很关键：  
**策略训练时只见机器人技能，测试时却要吃人类视频提取出的技能。**

为缓解这个 train-test gap，作者在策略训练时对抽取 skill 的图像对做增强，模拟不同 prompt 源带来的偏移。

#### 3. 推理时如何模仿？

对提示视频做滑窗：

- 从连续帧对中抽 skill 序列
- 依次把这些 skill 喂给策略
- 机器人按 skill 序列逐段执行

所以，UniSkill 的接口不是“给我一句话”也不是“给我目标图像”，而是：

**给我一个视频，我把它拆成一串可执行 skill。**

### 核心直觉

UniSkill 最重要的改变是：

**把跨 embodiment 对齐问题，改写成“用一个紧凑向量解释两帧之间的动态变化”。**

这带来的因果链条是：

- **What changed**：从依赖场景对齐、原型聚类、显式轨迹监督，改为基于未来帧编辑的预测式技能学习。
- **Which bottleneck changed**：监督来源从稀缺的人机对齐数据，变成几乎任何视频都具备的时序动态；表示瓶颈从“外观+动作混合”变成“主要编码动态变化”。
- **What capability changed**：机器人可以从未见过的人类视频中提取可执行 skill，并在未见场景、未见 embodiment 下仍有较强泛化。

更具体地说，它为什么有效：

1. **图像编辑天然强调变化区域**  
   当前帧和未来帧的大部分静态内容相同，真正需要被编码的是变化部分，因此 z 更容易成为“动作摘要”。

2. **深度削弱了纹理和外观捷径**  
   人手、机械臂、背景颜色都不同，但“向右旋、向上提、向前推”这类几何动态更稳定。

3. **策略看的是 skill，而不是人类像素**  
   这比直接拿人类子目标图像去条件化机器人更稳，因为表示层已经先做了一次 embodiment abstraction。

### 战略权衡

| 设计选择 | 直接收益 | 代价/风险 |
|---|---|---|
| 用 FSD 做扩散式未来帧编辑 | 不需要人机配对或任务对齐；动态监督更直接 | 训练成本高，效果受生成质量影响 |
| 在 ISD 中加入深度中间表征 | 降低外观泄漏，提升跨 embodiment 聚类性 | 依赖单目深度质量 |
| 冻结 ISD、再单独训策略 | 接口清晰，可复用 skill encoder | 端到端纠错弱，接触失败后难自我修正 |
| 固定 skill interval \(k\) | 训练稳定、部署简单 | 对不同执行速度不鲁棒 |

---

## Part III：证据与局限

### 关键证据信号

- **比较信号：真实世界跨 embodiment 模仿成立**  
  在 tabletop 的人类提示上，UniSkill 达到 **36%**，而 GCBC 为 **11%**、XSkill 为 **0%**。  
  在 kitchen 的人类提示上，UniSkill 达到 **87%**，GCBC 为 **33%**。  
  说明它不是只在“机器人提示 -> 机器人执行”时有效，而是真能把人类视频转成机器人可执行技能。

- **比较信号：未见 embodiment / 未见环境仍有优势**  
  在 kitchen 的 Anubis 提示（未见机器人 embodiment、未见环境）上，UniSkill **54%**，GCBC **33%**。  
  在 LIBERO 的人类提示上，UniSkill **48%**，GCBC **9%**。  
  这说明提升点主要来自表示层，而不是仅靠特定场景微调。

- **消融信号：大规模人类视频确实有用**  
  在 LIBERO 人类提示上，加入人类视频预训练后成功率 **0.19 -> 0.49**。  
  这直接支持论文最核心主张：**即便没有场景对齐，人类视频仍能提升跨 embodiment 技能学习。**

- **机制信号：表示更像 skill，不只是 latent code**  
  FSD 在相同当前图像下，只要换不同 z，就能生成不同未来动态；  
  t-SNE 中 embedding 更按“技能类型”而不是按“人/机器人”聚类。  
  这支持作者关于“embodiment-agnostic skill representation”的解释。

- **泛化信号：不是只记任务模板**  
  UniSkill 在组合任务上仍能工作，四段组合任务 A→B→C→D 还有 **42%**；在未见 scene A/B 中也优于 GCBC。  
  这说明它学到的是可拼接的低层技能，而不只是任务 ID。

### 局限性

- **Fails when**: 提示视频存在剧烈视角切换，尤其是 egocentric 人类视频；物体初始空间位置与提示偏差过大；任务对接触精度要求高且首次接触失败后需要在线纠偏时。
- **Assumes**: 有大量无标注人类/机器人视频和少量带动作机器人轨迹；依赖外部 DepthAnythingV2 与 InstructPix2Pix 类模块；skill interval 固定；各 benchmark 需要单独训练/微调策略；预训练规模较大。正文给出 project 页面，但未明确代码发布状态，复现仍有工程门槛。
- **Not designed for**: 需要高层语义推理或语言消歧的任务；行为节奏差异很大的跨主体模仿；与机器人可供性差异过大、仅靠“动作模仿”无法解释的人类演示。

### 可复用组件

- **ISD**：可直接当作跨 embodiment skill encoder。
- **FSD**：可作为保持机器人 embodiment 的“目标帧翻译器”，论文里的 GCBC-U 已证明这一点。
- **skill-conditioned diffusion policy**：可作为 VLA 或高层规划器下游的运动执行器。

一句话评价：  
**UniSkill 的真正价值，不是又做了一个 imitation policy，而是把“跨 embodiment 视频理解”收缩成了一个可扩展的动态表示学习问题。**

## Local PDF reference

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_UniSkill_Imitating_Human_Videos_via_Cross_Embodiment_Skill_Representations.pdf]]