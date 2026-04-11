---
title: "InterMimic: Towards Universal Whole-Body Control for Physics-Based Human-Object Interactions"
venue: CVPR
year: 2025
tags:
  - Human_Object_Interaction
  - task/physics-based-control
  - task/human-object-interaction
  - reinforcement-learning
  - dataset/OMOMO
  - dataset/BEHAVE
  - repr/SMPL
  - opensource/full
core_operator: 课程式师生蒸馏框架（Teacher-Student Distillation）：多个教师策略在小数据子集上学习并修正不完美MoCap，蒸馏为统一学生策略，实现大规模全身人-物交互的物理可信模仿
primary_logic: |
  不完美MoCap数据（接触误差、手部缺失、多体型）
  → 教师策略（MLP）：每个教师在小子集上RL训练，同时完成运动模仿+重定向+接触修正
  → 物理状态初始化（PSI）+ 交互早停（IET）缓解MoCap误差影响
  → 学生策略（Transformer）：从教师蒸馏（参考蒸馏+策略蒸馏）+ RL微调
  → 统一策略支持多物体、多动作的全身交互，零样本泛化到运动学生成器
claims:
  - "InterMimic是首个训练物理仿真人类在多样动态物体上学习大规模全身交互技能的框架"
  - "教师策略通过物理仿真自动修正MoCap中的接触误差和手部缺失，无需人工标注"
  - "学生策略零样本泛化到运动学生成器（InterDiff、HOI-Diff），实现从模仿到生成的跨越"
pdf_ref: paperPDFs/Human_Object_Interaction/CVPR_2025/2025_InterMimic_Towards_Universal_Whole_Body_Control_for_Physics_Based_Human_Object_Interactions.pdf
category: Human_Object_Interaction
---

# InterMimic: Towards Universal Whole-Body Control for Physics-Based Human-Object Interactions

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [Project Page](https://sirui-xu.github.io/InterMimic/) · [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Xu_InterMimic_Towards_Universal_Whole-Body_Control_for_Physics-Based_Human-Object_Interactions_CVPR_2025_paper.pdf)
> - **Summary**: InterMimic提出课程式师生蒸馏框架，多个教师策略在小数据子集上学习并修正不完美MoCap数据（接触误差、手部缺失），蒸馏为统一Transformer学生策略，首次实现大规模全身人-物交互的物理可信模仿，并零样本泛化到运动学生成器。
> - **Key Performance**:
>   - OMOMO数据集上学生策略成功率88.8%（训练集）/98.1%（测试集），显著优于从头训练基线
>   - 零样本集成HOI-Diff和InterDiff，实现文本→交互和交互预测的物理可信生成

---

## Part I：问题与挑战

### 真正的卡点

大规模物理可信人-物交互模仿面临**MoCap数据不完美**和**技能规模化**两大核心挑战：

- **MoCap不完美**：真实交互数据集中普遍存在接触浮空、手部姿态缺失/平均化、不同体型间的重定向误差——直接在物理仿真中模仿这些数据会导致物体掉落、穿透等不可恢复的失败
- **技能规模化困难**：RL的样本效率低，直接在大规模数据上训练单一策略需要极长时间；不同交互模式（搬运、踢、推等）之间存在冲突，单一策略难以同时掌握
- **全身交互的复杂性**：不同于简单抓取，全身交互涉及多身体部位与动态物体的协调接触，物体几何形状多样，控制维度高

### 输入/输出接口

- 输入：参考MoCap运动序列 + 物体几何信息
- 输出：物理仿真中的全身交互运动（关节力矩→SMPL姿态 + 物体6DoF）

---

## Part II：方法与洞察

### 整体设计

InterMimic采用两阶段课程式框架：

1. **教师阶段（Imitation as Perfecting）**：
   - 每个教师策略（MLP）在小数据子集上训练，专注于特定主体/物体
   - 同时完成三个任务：运动模仿 + 体型重定向（统一到标准体型）+ 接触修正（物理仿真自动修正MoCap误差）
   - PSI（物理状态初始化）：用仿真修正后的状态替代不完美的MoCap状态作为rollout初始化
   - IET（交互早停）：物体偏离、接触丢失时提前终止，避免无效训练

2. **学生阶段（Imitation with Distillation）**：
   - Transformer学生策略从所有教师蒸馏
   - 参考蒸馏：用教师rollout（已修正）替代原始MoCap作为参考
   - 策略蒸馏：DAgger + PPO渐进式训练——先模仿教师动作，逐步过渡到RL自主优化
   - RL微调使学生超越简单模仿，解决教师间冲突

### 核心直觉

**什么变了**：从"在不完美数据上直接训练大规模策略"到"先用小策略修正数据，再蒸馏为大策略"。

**哪些分布/约束/信息瓶颈变了**：
- 教师策略将不完美MoCap转化为物理可信的参考运动 → 学生策略的训练信号从"有噪声的MoCap"变为"物理修正后的运动"，学习难度大幅降低
- 空间换时间：多个教师并行训练在小子集上 → 绕过了RL在大规模数据上的样本效率瓶颈
- Transformer的序列建模能力 + 更长的观察窗口 → 学生策略能区分相似但不同的交互模式，解决教师间冲突

**为什么有效**：物理仿真天然提供了接触修正能力——只要策略能大致跟踪参考运动，仿真器就会自动产生物理可信的接触力。教师策略利用了这一点，将"修正MoCap"转化为"在仿真中模仿MoCap"的副产品。

**权衡**：两阶段训练增加了总计算量；教师策略的质量是学生的上限；严重的MoCap错误仍无法修正（约少量数据被丢弃）。

---

## Part III：证据与局限

### 关键实验信号

- **规模化成功**：在OMOMO（15物体、~10小时数据）上，学生策略成功率88.8%（训练集）/98.1%（测试集），远超从头训练的PPO基线（23.9%/9.6%）
- **MoCap修正**：教师策略自动修正了接触浮空、手部姿态缺失、对称物体旋转错误等MoCap伪影
- **零样本泛化**：学生策略零样本集成HOI-Diff（文本→交互）和InterDiff（交互预测），在物理仿真中执行运动学生成器的输出
- **消融验证**：去掉PSI后成功率下降（36.1→42.6）；去掉参考蒸馏后测试集泛化显著退化；去掉RL微调后学生陷入教师间的"平均"行为

### 局限与可复用组件

- **局限**：当前不支持软体物体；严重MoCap错误仍需丢弃；教师训练的并行化需要大量GPU；手部交互精度受限于MoCap数据质量
- **可复用**：师生蒸馏框架可迁移到任何需要从不完美数据学习物理可信技能的场景；PSI和IET可用于任何物理模仿任务；参考蒸馏的思路（用仿真修正数据再训练）适用于任何数据质量不佳的模仿学习

---

## 本地 PDF 引用

![[paperPDFs/Human_Object_Interaction/CVPR_2025/2025_InterMimic_Towards_Universal_Whole_Body_Control_for_Physics_Based_Human_Object_Interactions.pdf]]
