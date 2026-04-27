---
title: "Learning Diffusion Policies from Demonstrations For Compliant Contact-rich Manipulation"
venue: "IEEE/SICE International Symposium on System Integration (SII)"
year: 2026
tags:
  - Embodied_AI
  - task/contact-rich-manipulation
  - diffusion
  - compliance-control
  - temporal-ensemble
  - dataset/PowderGrinding
  - dataset/PencilEraser
  - dataset/BimanualRoundInsertion
  - dataset/BimanualCuboidInsertion
  - repr/6D-rotation
  - opensource/no
core_operator: 以力觉条件扩散策略联合生成末端笛卡尔位姿与刚度序列，再交给顺应控制器执行，从而在长时程接触任务中维持稳定受力与连续动作。
primary_logic: |
  图像/力扭矩/机器人状态输入 → 条件扩散模型逐步去噪生成长时域动作块（末端位姿+夹爪+刚度），并用时间集成平滑相邻预测 → 顺应控制器输出关节命令，完成接触密集操作中的稳定力控执行
claims:
  - "在粉末研磨任务中，DIPCOM 的细粉产出比例为 55.88%，显著高于 Comp-ACT 的 9.96%，且两者法向接触力幅值相近 [evidence: comparison]"
  - "在铅笔擦除任务中，DIPCOM 的平均擦除比例为 77.32%、成功率为 52.3%，而 Comp-ACT 分别为 26.0% 和 0% [evidence: comparison]"
  - "在双臂圆柱/方柱插接任务中，DIPCOM 与 Comp-ACT 成功率相当（100%/95%），但展示出更丰富的双臂协同轨迹与支撑臂调整行为 [evidence: case-study]"
related_work_position:
  extends: "Comp-ACT (Kamijo et al. 2024)"
  competes_with: "Comp-ACT (Kamijo et al. 2024)"
  complementary_to: "Universal Manipulation Interface (Chi et al. 2024)"
evidence_strength: moderate
pdf_ref: paperPDFs/Vision_Action_VA_Models_2024/IEEE_SICE_International_Symposium_on_System_Integration_SII_2026/2026_Learning_Diffusion_Policies_from_Demonstrations_For_Compliant_Contact_rich_Manipulation.pdf
category: Embodied_AI
---

# Learning Diffusion Policies from Demonstrations For Compliant Contact-rich Manipulation

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2410.19235)
> - **Summary**: 该文将扩散策略接入顺应控制回路，在少量示教下同时预测末端轨迹与机械刚度，显著提升了长时程、重复型接触操作中的稳定受力与动作连续性。
> - **Key Performance**: 粉末研磨细粉比例 55.88% vs 9.96%（Comp-ACT）；铅笔擦除成功率 52.3% vs 0%

> [!info] **Agent Summary**
> - **task_path**: 少量示教的 RGB/F-T/本体状态 -> 末端位姿+夹爪+刚度序列 -> 顺应执行接触密集操作
> - **bottleneck**: 长时程接触任务存在多峰动作分布与跨时间块不连续问题，传统 VAE/ACT 风格策略容易卡住、均值化或陷入僵硬重复
> - **mechanism_delta**: 用力觉条件扩散替代 CVAE 式动作生成，并预测更长动作块配合 temporal ensemble 平滑过渡
> - **evidence_signal**: 在真实机器人研磨与擦除两类长时程任务上明显超过 Comp-ACT，但在插接任务上仅持平
> - **reusable_ops**: [force-conditioned action diffusion, stiffness prediction with compliance control]
> - **failure_modes**: [hyperparameter sensitivity and control-frequency disturbance, limited gain on short-horizon insertion tasks]
> - **open_questions**: [can one policy generalize across task variations, would relative action spaces improve robustness]

## Part I：问题与挑战

这篇工作的核心问题，不是“机器人能不能接触物体”，而是**能否在持续接触中同时保持几何轨迹、接触力和动作节律**。  
像粉末研磨、橡皮擦除、双臂插接这类任务，真正难点在于：

1. **接触是持续的**：不是碰一下就结束，而是要在表面上持续滑动、施压、微调。
2. **动作是重复且多模态的**：同样的研磨或擦除目标，可以有多种微轨迹和节律；如果策略把这些可行解“平均化”，就容易停滞、抖动或动作变形。
3. **控制变量不只是一条轨迹**：还要决定“压多硬”，即刚度/顺应性，否则要么力不够、要么位置不稳。
4. **长时程更容易累积误差**：每次只预测短动作块、再丢弃后半段的做法，在接触任务里会带来 chunk 边界不连续，表现成 jerk 或冻结。

这也是为什么作者把目标锁定在**compliant contact-rich manipulation**：  
传统位置/速度控制在这类任务里很难长期维持稳定接触；而仅靠机械顺应又会牺牲末端精度。作者此前的 Comp-ACT 已经把示教学习和顺应控制结合起来，但在**长时程、反复型任务**上仍容易失效，因此这里进一步引入 diffusion policy。

### 输入 / 输出接口

- **输入**：RGB 图像、腕部 F/T 力扭矩、机器人本体状态
- **输出**：一段动作序列，每步动作包括  
  - 末端绝对笛卡尔位姿  
  - 夹爪开合量  
  - 机械刚度参数
- **执行接口**：策略不直接出电机力矩，而是把位姿与刚度交给顺应控制器，由控制器产生最终关节命令

### 边界条件

- 每个任务单独训练一个策略，不是统一多任务模型
- 数据规模较小：约 40–60 条示教/任务
- 依赖真实机器人硬件：UR5e、腕部 F/T 传感器、多相机、可调顺应控制
- 主要验证的是**真实世界接触操作可行性**，不是大规模泛化

## Part II：方法与洞察

### 方法骨架

DIPCOM 的整体思路可以概括为：**“看见 + 感到 + 推出一整段顺应动作”**。

1. **多模态观测编码**  
   图像由 ResNet18 提取视觉特征，再与力扭矩、本体状态一起送入 transformer encoder。

2. **条件扩散动作生成**  
   使用 classifier-free conditional diffusion / DDIM 风格去噪，从噪声逐步恢复动作序列，而不是一次性回归一个动作。

3. **动作空间显式包含刚度**  
   输出不仅有 EE pose 和 gripper，还有 stiffness，对应顺应控制器中的刚度对角项。  
   这意味着网络直接学习“怎么动”和“动得多硬”。

4. **长时域动作块 + Temporal Ensemble**  
   作者强调接触任务不能像普通抓取那样只预测短 horizon。  
   他们把预测长度提升到平均约 48 个动作，并把 ACT 中的 temporal ensemble 扩展到 diffusion setting，用来平滑相邻推理步骤之间的动作切换。

5. **姿态表示改为 6D rotation**  
   相比 axis-angle，6D 表示更连续、唯一性更好，减少了姿态学习中的不稳定性。

### 核心直觉

真正的变化不是“把生成模型从 VAE 换成 diffusion”这么表面，而是：

**从“单次压缩/回归一个可能被均值化的动作块”**  
→ 变成 **“在观测条件下逐步去噪恢复一整段多峰动作序列”**

这带来的因果变化是：

- **动作分布层面**：  
  原本 VAE/CVAE 风格更容易把多种可行的接触微动作压成平均解；  
  diffusion 更适合保留“同一目标下存在多种细粒度轨迹”的分布结构。

- **控制约束层面**：  
  原本策略主要在预测“去哪儿”；  
  现在策略同时预测“以什么刚度接触”，把力控从固定控制器参数变成了策略决策的一部分。

- **时序连续性层面**：  
  原本 chunk-based 预测在长时程接触中容易出现边界不连续；  
  更长 horizon + temporal ensemble 让策略更像在“延续动作流”，而不是每次重新起步。

- **能力结果层面**：  
  机器人更能持续完成圆周研磨、往返擦除这类重复动作，而不是在动作中途冻结或退化成局部小循环。

一个很关键的观察是：**DIPCOM 的优势并不只是“压得更大力”**。  
在粉末研磨中，论文显示它与基线的法向力幅值相近，但效果差距很大。这说明能力提升主要来自**动作模式与时序连续性**，而不是单纯力更大。

### 战略性取舍

| 设计选择 | 改变的瓶颈 | 带来的能力 | 代价/风险 |
|---|---|---|---|
| 条件扩散替代 CVAE 式动作生成 | 缓解多模态动作被平均化 | 更能生成持续、细粒度、重复型接触轨迹 | 训练/推理更耗算力，调参更敏感 |
| 将 stiffness 纳入动作输出 | 把“轨迹控制”和“接触软硬”联合建模 | 同时保证接触力与位姿稳定 | 依赖可调顺应控制器与 F/T 感知 |
| 更长 horizon + temporal ensemble | 降低 chunk 边界不连续 | 长时程动作更平滑，减少 jerk/freeze | 推理延迟与误差累积仍需控制 |
| 6D 姿态表示 | 解决 axis-angle 的一对多和不连续性 | 姿态学习更稳定，一致性更好 | 只是表示改良，不直接解决跨任务泛化 |

## Part III：证据与局限

### 关键证据

**1. 粉末研磨：优势来自“持续正确地磨”，不是“更用力地压”**  
最有说服力的信号是：DIPCOM 细粉产出比例达到 **55.88%**，Comp-ACT 只有 **9.96%**。  
而论文中的力曲线显示，两者法向力规模接近。说明性能差异主要不是力幅值，而是 DIPCOM 更能复现示教中的**圆周研磨 + 周期性抬起观察**这一长时程行为结构。

**2. 铅笔擦除：几何瞄准与接触施力被同时改善**  
DIPCOM 在擦除任务上平均擦除比例 **77.32%**、成功率 **52.3%**；Comp-ACT 分别为 **26.0%** 和 **0%**。  
这说明方法不仅能施加更合适的力，还能更稳定地把擦除轨迹对准字迹区域。

**3. 双臂插接：不是所有接触任务都显著受益**  
在圆柱/方柱插接上，DIPCOM 与 Comp-ACT 成功率相同：**100% / 95%**。  
这里的结论很重要：**diffusion 的收益主要集中在长时程、重复型、非线性节律任务**；对于较短程、结构化更强的插接任务，它至少在成功率上没有明显超越。

### 局限性

- **Fails when**: 控制频率受系统扰动、扩散推理预算不足、或任务更偏短时程且轨迹高度单一时，DIPCOM 的优势会减弱；论文也明确指出其对超参数更敏感，在插接任务上并未超过基线。
- **Assumes**: 需要真实机器人具备主动顺应控制能力、腕部 F/T 传感器、VR 示教采集链路，以及每任务 40–60 条任务专用示教；同时扩散推理带来更高计算负担，对系统实时性有要求。论文也未提供代码/开源实现，复现成本不低。
- **Not designed for**: 跨任务零样本泛化、无力传感器的纯位置控制、统一策略覆盖多种工具/材料变化的开放场景，也没有验证大规模数据下的泛化规律。

### 可复用组件

1. **force-conditioned diffusion policy**：适合把视觉、力觉、本体状态统一进动作生成。
2. **stiffness-as-action 设计**：对任何需要把“怎么动”和“接触多硬”一起决策的顺应控制任务都可迁移。
3. **diffusion + temporal ensemble**：对长时程 chunk-based 控制的平滑化很有启发。
4. **6D orientation representation**：在学习绝对笛卡尔动作时是低成本但有效的稳定性增强。

### 一句话结论

这篇论文的真正贡献，不是单纯把 diffusion policy 搬到机器人上，而是证明了：  
**在接触密集、长时程、重复型操作中，把“多模态动作生成 + 刚度决策 + 时序平滑”一起建模，能比 VAE/ACT 式策略更稳定地维持真实接触行为。**  
它的能力跳跃主要出现在研磨、擦除这类“持续动作流”任务上，而不是所有接触任务都无条件受益。

## Local PDF reference

![[paperPDFs/Vision_Action_VA_Models_2024/IEEE_SICE_International_Symposium_on_System_Integration_SII_2026/2026_Learning_Diffusion_Policies_from_Demonstrations_For_Compliant_Contact_rich_Manipulation.pdf]]