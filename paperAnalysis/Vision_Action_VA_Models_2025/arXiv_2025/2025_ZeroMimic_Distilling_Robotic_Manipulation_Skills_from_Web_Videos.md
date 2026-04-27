---
title: "ZeroMimic: Distilling Robotic Manipulation Skills from Web Videos"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/robotic-manipulation
  - affordance-learning
  - structure-from-motion
  - action-chunking-transformer
  - dataset/EpicKitchens
  - dataset/RoboCasa
  - opensource/full
core_operator: "将网页第一视角人类操作视频拆成“人类接触先验引导抓取 + SfM对齐的6D腕部轨迹模仿”，蒸馏为可零样本部署的机器人技能策略。"
primary_logic: |
  第一视角网页视频 + 技能文本/目标图像 + 当前机器人观测
  → VRB预测任务相关接触区域并由AnyGrasp选择可执行抓取，HaMeR+COLMAP/EPIC-Fields恢复世界坐标腕部6D轨迹，再用ACT学习相对动作块
  → 输出可跨对象、场景与机器人本体直接执行的图像目标条件操控技能
claims:
  - "在 EpicKitchens 上训练后，ZeroMimic 在真实世界 9 项技能、30 个场景上的总体成功率为 71.0%，在 RoboCasa 仿真 4 项技能上为 73.8% [evidence: comparison]"
  - "抓取阶段中，VRB 接触点 + AnyGrasp 抓取选择明显优于仅用 AnyGrasp 分数或直接在接触点闭合夹爪；在抽屉把手抓取上分别为 8/10、0/10、0/10 [evidence: ablation]"
  - "在成功抓取后，SfM 对齐的 6D 后抓取策略显著优于去掉相机几何的变体和 VRB 2D 轨迹；开抽屉/开柜门均达到 10/10，而去 SfM 仅为 4/10 和 6/10 [evidence: ablation]"
related_work_position:
  extends: "H2R (Bharadhwaj et al. 2023)"
  competes_with: "H2R (Bharadhwaj et al. 2023); ReKep (Huang et al. 2024)"
  complementary_to: "N/A"
evidence_strength: moderate
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_ZeroMimic_Distilling_Robotic_Manipulation_Skills_from_Web_Videos.pdf
category: Embodied_AI
---

# ZeroMimic: Distilling Robotic Manipulation Skills from Web Videos

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2503.23877) · [Project](https://zeromimic.github.io)
> - **Summary**: 这篇工作把网页第一视角人类视频转成“可直接上机器人”的零样本操作技能，关键在于把问题拆成任务相关抓取与 SfM 对齐的后抓取 6D 运动模仿两段。
> - **Key Performance**: 真实世界 9 项技能/30 场景总体成功率 71.0%；RoboCasa 仿真 4 项技能成功率 73.8%。

> [!info] **Agent Summary**
> - **task_path**: 第一视角网页人类操作视频 + 目标图像 + 当前RGB-D观测 -> 机器人零样本图像目标条件操控策略
> - **bottleneck**: 人类视频到机器人动作之间存在形态失配、移动相机导致的几何歧义，以及“接触前抓哪/接触后怎么动”混在一起的学习难题
> - **mechanism_delta**: 将操作分成“affordance 引导抓取”和“SfM 对齐的相对 6D 腕部动作块预测”，把人类视频中的语义与几何信息变成机器人可执行接口
> - **evidence_signal**: 真实机器人 9 技能零样本评测 + 去掉 affordance / 去掉 SfM 的消融都显著退化
> - **reusable_ops**: [human-contact-point-guided grasp reranking, SfM-grounded wrist-trajectory retargeting]
> - **failure_modes**: [small-or-reflective handles break depth/grasp prediction, non-egocentric camera placement hurts post-grasp policy]
> - **open_questions**: [how to handle release/in-hand/non-prehensile skills, how to make retargeting morphology-aware rather than direct wrist-to-gripper transfer]

## Part I：问题与挑战

这篇论文要解决的核心问题不是“让机器人看懂视频”，而是：

**能否只靠网页中的人类操作视频，不用同机器人、同场景、同物体的示教数据，也不做额外探索，就直接蒸馏出可部署的机器人操控技能？**

### 真正瓶颈是什么？

真正瓶颈是**动作可执行性的跨域转移**，具体有三层：

1. **形态失配**
   - 人手不是双指夹爪。
   - 人类腕部运动不能直接当机器人末端执行器轨迹。
   - 尤其在接触前，抓取姿态、夹爪朝向、可达性都和人手不同。

2. **移动相机带来的动作歧义**
   - 网页第一视角视频常有抖动、遮挡、出框。
   - 图像中的“手在动”，不等于“手相对物体在动”。
   - 如果不恢复相机几何，学到的很可能是“手+相机联合运动”的假动作标签。

3. **web 视频分布太杂且动作多模态**
   - 同一任务有很多做法。
   - 物体种类、尺寸、朝向、场景背景变化极大。
   - 直接从视频到机器人控制，容易学成模糊平均动作。

### 为什么现在值得做？

因为现在刚好有几类成熟组件能拼起来形成闭环：

- 大规模第一视角人类视频数据：如 **EpicKitchens**
- 手部重建：**HaMeR**
- 相机/场景几何恢复：**COLMAP / EPIC-Fields**
- 人类交互 affordance：**VRB**
- 机器人抓取模型：**AnyGrasp**
- 多模态动作生成策略：**ACT**

换言之，ZeroMimic 的价值不只是“用了 web 数据”，而是证明：**当语义、几何、抓取、动作建模这些模块都成熟到一定程度后，纯人类网页视频第一次可以被系统性蒸馏成零样本机器人技能。**

### 输入/输出接口与边界条件

**输入：**
- EpicKitchens 中按技能筛出的第一视角人类视频
- 技能文本描述（如 open drawer）
- 测试时的当前 RGB-D 观测
- 每个任务一张人类“目标完成图像”

**输出：**
- 可直接部署在机器人上的、图像目标条件的技能策略
- 生成抓取动作与后抓取 6D 轨迹

**边界条件：**
- 机器人是**静态机械臂 + 双指夹爪**
- 相机视角需**大致接近人类第一视角**
- 任务需能分解成：
  1. 抓取前接近并抓住  
  2. 抓住后做相对稳定的刚体操作
- 不覆盖双臂、夹内精细操作、非抓取交互、释放时机学习等更复杂情形

---

## Part II：方法与洞察

ZeroMimic 的设计哲学很明确：

**不要端到端地把“人类视频→机器人动作”硬映射；而是先找到人类视频中真正可迁移的接口，再让机器人专属模块补上不可迁移的部分。**

### 方法分解

#### 1. 抓取阶段：人类视频负责“抓哪里”，机器人模型负责“怎么抓”

这一阶段分两步：

- **Affordance prediction（VRB）**  
  给定 RGB 图像和任务文本，VRB 预测一个任务相关的接触点/区域，例如“开抽屉时该碰把手哪里”。

- **Grasp selection（AnyGrasp）**  
  在这个区域附近，由 AnyGrasp 为双指夹爪挑选真正可执行的抓取姿态。

这实际上做了一个很关键的职责划分：

- 人类视频提供的是**任务语义上的接触先验**
- 机器人抓取模型提供的是**硬件约束下的可执行抓取**

这比“直接把人手姿态映射到夹爪”稳得多。

#### 2. 后抓取阶段：从 web 视频恢复 6D 腕部轨迹，再学习动作块

抓住物体后，ZeroMimic 的目标是学会“拿着它怎么动”。

具体做法：

- 用 **HaMeR** 从视频中恢复 3D 手部姿态
- 用 **COLMAP / EPIC-Fields** 提供的相机外参与场景几何，把手腕轨迹从图像坐标恢复到**世界 3D 坐标**
- 只保留**腕部 6D 轨迹**
- 按技能分别训练 ACT 策略：
  - 输入：当前图像、目标图像、当前腕部姿态
  - 输出：未来一段 wrist pose chunk

测试时，机器人把当前 gripper pose 送入模型，直接把预测的 6D 轨迹转到机器人坐标执行。

#### 3. 一个很重要但容易忽视的实现点：把动作转到当前相机坐标系

训练数据来自移动第一视角视频，测试时机器人相机是静态的。  
作者因此把当前与未来 wrist pose 都转换到**当前帧相机坐标系**下。

这一步的作用是：

- 不让模型额外学习“相机怎么动”
- 让模型更专注于“手相对当前观察该怎么动”
- 降低移动相机视频与静态机器人视角之间的分布差

### 核心直觉

ZeroMimic 真正改变的不是网络结构本身，而是**迁移接口**：

- **以前**：试图从人类视频直接学机器人动作，导致语义、几何、抓取、硬件约束混在一起
- **现在**：把可迁移信息拆成两类  
  1. **接触语义**：任务相关地该碰哪里  
  2. **抓住后几何运动**：物体抓稳后，腕部/夹爪该如何在 6D 空间运动

这带来的因果链条是：

**问题拆分**  
→ 人手与夹爪最严重的失配被局部化到“抓取执行”  
→ 人类视频只承担自己最擅长提供的信息（接触意图 + 操作轨迹）  
→ 机器人模块只承担自己最擅长的部分（可执行抓取）  
→ 最终得到能零样本部署的技能策略

再进一步：

**加入 SfM 几何对齐**  
→ 去掉“手轨迹里混入的相机运动噪声”  
→ 训练标签从像素运动变成 3D/6D 可执行运动  
→ 后抓取策略明显更稳、更可部署

最后：

**用相对 6D 动作块而非绝对动作**  
→ 减轻人手到夹爪的分布偏移  
→ 更适合多模态动作预测  
→ 执行时更稳定

### 战略性取舍

| 设计选择 | 解决的瓶颈 | 带来的能力 | 代价/边界 |
| --- | --- | --- | --- |
| 抓取/后抓取两阶段分解 | 人手与夹爪形态差异太大，端到端难学 | 可把人类视频直接转成机器人可执行接口 | 只适合可稳定抓取后再操作的任务 |
| VRB 提供接触区域，AnyGrasp 提供抓取姿态 | “知道碰哪里”不等于“知道怎么抓” | 兼顾任务相关性与夹爪可执行性 | 依赖深度质量与 grasp model |
| SfM 对齐的 6D 腕部轨迹 | 第一视角移动相机污染动作标签 | 后抓取策略更像真实几何动作 | 依赖手部重建与相机重建质量 |
| 相对动作 + ACT action chunking | web 视频动作多模态，单步预测不稳 | 更稳地预测一段未来动作 | 仍然是按技能单独训练，不是统一通用策略 |
| 每个 skill 单独训练一个模型 | 统一模型难覆盖技能差异 | 单技能表现更稳 | 技能扩展成本更高，组合能力有限 |

---

## Part III：证据与局限

### 关键证据信号

#### 1. 消融信号：抓取不是“有接触点就够”，也不是“有 grasp score 就够”

在开抽屉/开柜门上：

- **ZeroMimic**：VRB 接触点 + AnyGrasp 抓取选择
- **去 affordance**：只按 AnyGrasp 分数抓
- **去 grasp model**：直接去接触点闭合夹爪

结果表明作者的方法最好。尤其在抽屉把手抓取上，作者方法 **8/10**，两个简化版都 **0/10**。  
这说明：

- 人类视频擅长告诉机器人**哪个区域与任务相关**
- 但真正的夹爪姿态仍必须由**机器人抓取模型**决定

两者缺一不可。

#### 2. 消融信号：SfM 几何是从 web 视频学动作能否落地的关键

在“成功抓到之后”的后抓取执行上：

- **Ours**：SfM 对齐的 6D wrist policy
- **Ours w/o SfM**：去掉相机几何，只保留更像 H2R 的表示
- **VRB 2D trajectory**：只用 2D 轨迹并随机补深度/固定朝向

开抽屉与开柜门任务中，作者方法都达到 **10/10**；去 SfM 下降到 **4/10** 和 **6/10**；VRB 2D 更差。  
最强结论是：

**如果不先把人类视频中的动作“几何去混淆”，模型学到的不是可执行操控，而更像是带着相机抖动噪声的观测轨迹。**

#### 3. 系统级信号：确实学到了跨对象、跨场景、跨机器人本体的技能

整体零样本部署结果：

- **真实世界**：9 项技能，2 个机器人，18 类对象，30 个场景，**71.0%**
- **仿真 RoboCasa**：4 项技能，**73.8%**

而且论文明确强调：

- 测试对象实例与场景**不在训练数据中**
- 还能迁移到**未在人类训练视频中出现的对象类别**，如 spoon 倒盐进锅、切蛋糕
- Franka 与 WidowX 大多数技能表现接近，说明有一定**跨 embodiment** 能力

#### 4. 直接对比信号：比 VLM 约束规划式零样本系统更稳

与 **ReKep** 的 4 个真实任务对比中：

- Open Drawer：**8/10 vs 0/10**
- Place Pasta Bag into Drawer：**8/10 vs 4/10**
- Pour Food from Bowl into Pan：**8/10 vs 0/10**
- Close Drawer：**6/10 vs 6/10**

论文给出的失败分析也有说服力：ReKep 常死在
- 关键点提错
- 物体关联错
- 空间/关节轴推理错
- 约束量值不对

这反过来支持了 ZeroMimic 的路线：  
**对几何与接触进行数据驱动蒸馏，比完全依赖高层 VLM 约束推理更接近“可执行操控”。**

#### 5. 失败归因信号：最大瓶颈已经不是“看不懂图”，而是执行链路中的几何与控制噪声

真实世界失败试验中：

- AnyGrasp 阶段：31.1%
- VRB 阶段：24.1%
- 后抓取策略阶段：44.8%

这说明当前最大误差源已经转移到**后抓取动作执行**，尤其是：
- 视角偏离第一视角太多
- 人类动作重建噪声较大
- 精细任务对 6D 轨迹精度要求更高

### 1-2 个最值得记住的指标

- **71.0%**：真实世界 9 技能、30 场景总体零样本成功率
- **10/10 vs 4/10**：成功抓取后，SfM 对齐的后抓取策略相对去 SfM 版本在开抽屉任务上有明显跃升

### 局限性

- **Fails when:** 相机视角明显偏离人类第一视角；深度对小型、反光把手不稳定；任务要求非常细粒度的出液、切割或姿态控制时，后抓取轨迹误差会被放大。
- **Assumes:** 任务可拆成“抓取前 + 稳定抓取后”两阶段；物体能被双指夹爪稳定持握并主要做刚体操作；依赖高质量 RGB-D、HaMeR 手部重建、EPIC-Fields/COLMAP 相机几何、VRB 和 AnyGrasp 等现成模块。
- **Not designed for:** 夹内操作、非抓取交互、释放策略学习、双臂任务、显式的人手到夹爪形态补偿。

### 资源/复现依赖

这项工作的可复现性并不只取决于策略网络本身，还依赖：

- **预训练外部模块**：VRB、AnyGrasp、HaMeR、COLMAP/EPIC-Fields
- **硬件条件**：RGB-D 相机质量会直接影响抓取与执行
- **视角约束**：测试相机最好接近 egocentric
- **训练形态**：每个技能单独训练一个模型，论文报告约 **18 小时 / skill / RTX 3090**

所以它是“零机器人示教”，但不是“零系统工程依赖”。

### 可复用部件

这篇论文最可迁移的，不只是整套系统，而是几个可复用操作符：

1. **任务相关接触点 → 抓取候选重排**
2. **第一视角视频的 SfM 对齐 6D 腕部轨迹提取**
3. **相机坐标系下的相对 6D action chunking**
4. **单张目标图像驱动的后抓取技能执行**

如果后续要扩到更通用的机器人系统，我认为最值得保留的是第 1 和第 2 点：  
它们定义了一个很清晰的“人类视频如何变成机器人可执行中间表示”的接口。

## Local PDF reference

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_ZeroMimic_Distilling_Robotic_Manipulation_Skills_from_Web_Videos.pdf]]