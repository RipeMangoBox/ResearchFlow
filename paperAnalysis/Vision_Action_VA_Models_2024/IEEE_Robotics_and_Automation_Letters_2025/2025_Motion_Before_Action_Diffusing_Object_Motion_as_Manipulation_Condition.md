---
title: "Motion Before Action: Diffusing Object Motion as Manipulation Condition"
venue: IEEE Robotics and Automation Letters
year: 2025
tags:
  - Embodied_AI
  - task/robotic-manipulation
  - diffusion
  - dataset/Adroit
  - dataset/DexArt
  - dataset/MetaWorld
  - opensource/no
core_operator: 先用扩散模型预测未来对象位姿序列，再将该序列作为条件引导第二个扩散头生成机器人动作序列。
primary_logic: |
  当前视觉观测/点云与机器人状态 → 编码观测并扩散预测未来对象位姿序列 → 将位姿特征与观测联合输入动作扩散头 → 输出多步末端执行器位姿与夹爪动作
claims:
  - "Adding MBA to Diffusion Policy raises the average success rate over 57 simulation tasks from 53.6% to 67.8%, and adding it to DP3 raises it from 71.3% to 77.5% [evidence: comparison]"
  - "On real-world tasks, RISE with MBA improves Put Bread into Pot success from 80% to 95% and Open Drawer success from 37.5% to 52.5% under matched evaluation protocols [evidence: comparison]"
  - "Under the same DP base policy, MBA outperforms ATM on Open Drawer success (30% vs 5%) and Pour Balls average balls poured (3.60 vs 2.05), but increases inference time to 197.50 ms versus 105.85 ms for ATM [evidence: comparison]"
related_work_position:
  extends: "Diffusion Policy (Chi et al. 2023)"
  competes_with: "Any-point Trajectory Modeling (ATM; Wen et al. 2023)"
  complementary_to: "RISE (Wang et al. 2024); FoundationPose (Wen et al. 2024)"
evidence_strength: moderate
pdf_ref: paperPDFs/Vision_Action_VA_Models_2024/IEEE_Robotics_and_Automation_Letters_2025/2025_Motion_Before_Action_Diffusing_Object_Motion_as_Manipulation_Condition.pdf
category: Embodied_AI
---

# Motion Before Action: Diffusing Object Motion as Manipulation Condition

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2411.09658), [Project](https://selen-suyue.github.io/MBApage/)
> - **Summary**: 论文提出一个可插拔的双扩散模块 MBA，让机器人先预测“对象接下来会怎么运动”，再据此生成动作，从而把 manipulation policy 从“记外观直接出动作”改成“先做对象运动推理，再做动作决策”。
> - **Key Performance**: 57 个仿真任务上，DP 平均成功率从 53.6% 提升到 67.8%，DP3 从 71.3% 提升到 77.5%；真实世界中 RISE 在 Put Bread into Pot 上从 80% 提升到 95%。

> [!info] **Agent Summary**
> - **task_path**: RGB/RGB-D/点云观测 + 机器人状态 -> 未来对象位姿序列 -> 多步末端执行器位姿与夹爪动作
> - **bottleneck**: 直接从观测到动作会把“对象未来状态推理”和“动作生成”混在一起，模型更容易记忆视觉外观而不是学习对象运动与动作之间的因果关系
> - **mechanism_delta**: 在原 diffusion action head 前串联一个 object-motion diffusion head，先采样对象未来位姿，再将其与观测联合条件化动作去噪
> - **evidence_signal**: 跨 DP、DP3、RISE 的一致增益，且在 Open Drawer、Pour Balls、Cut Clay 这类高精度/6DoF 任务上提升更明显
> - **reusable_ops**: [未来对象轨迹扩散预测, 中间运动条件化动作生成]
> - **failure_modes**: [细长或遮挡目标会放大位姿估计误差并导致重复抓取, 固定对象数与不可追踪变形体限制适用范围]
> - **open_questions**: [如何摆脱MoCap监督获得对象运动标签, 如何降低双扩散带来的推理时延]

## Part I：问题与挑战

这篇论文要解决的不是“如何再造一个新的 manipulation policy”，而是一个更底层的瓶颈：**现有 diffusion-based imitation policy 往往直接把观测映射到动作，却没有显式建模对象未来会如何运动**。

### 真正的难点是什么？
在机器人操作里，动作并不是只由“当前看到了什么”决定，还取决于：
- 对象下一步会在什么位置、朝什么方向转动；
- 机器人与对象之间的接触是否能持续；
- 任务是否需要 6DoF 姿态控制、精细接触或多阶段执行。

直接学习 `observation -> action` 时，网络要同时隐式完成两件事：
1. 从视觉里推断对象的状态与未来运动；
2. 再据此生成一串动作。

这会让学习问题过于纠缠。尤其在以下场景里更明显：
- articulated object（如抽屉）
- soft object（如面包）
- tool-use/contact-rich task（如切黏土）
- 需要精确旋转与姿态控制的 6DoF 任务（如倒球）

### 论文的输入/输出接口
- **输入**：RGB 图像或点云观测，以及机器人自身状态
- **中间变量**：未来对象位姿序列
- **输出**：未来多步机器人动作序列（末端位姿 + gripper width）

### 为什么现在值得做？
因为 diffusion action head 已经成为 imitation manipulation 中很强的生成器，但它缺一个结构化中间变量。作者抓住了一个关键事实：

- 对象位姿与机器人末端位姿都可表示在相近的 6DoF/SE(3)-like 空间中；
- 二者都具有连续轨迹和多模态未来；
- 因而都适合由 diffusion 模型生成。

换句话说，**现在的条件已经足够成熟，可以把“对象未来运动”做成一个显式、可学习、可插拔的动作条件**。

### 边界条件
这篇方法有几个明确边界：
- 主要针对**带 diffusion action head 的策略**
- 训练阶段需要**对象未来位姿监督**
- 部署时**不需要 MoCap 或 marker**
- 对象数量在设计上是**固定的**

---

## Part II：方法与洞察

MBA 的核心做法非常清晰：**把原本一步完成的动作生成，拆成两步条件生成**。

1. 先根据观测预测未来对象位姿序列
2. 再在对象位姿引导下生成机器人动作序列

作者把联合条件分布拆成：
- `对象运动 | 观测`
- `动作 | 对象运动, 观测`

这使得 MBA 不是一个独立 policy，而是一个**插在原 policy 感知模块与 diffusion action head 之间的 plug-and-play 模块**。

### 方法结构

#### 1）对象运动生成模块
- 输入：原 policy 的观测特征
- 输出：未来 `T_m` 步对象位姿序列
- 表示：对象位姿用 3D 平移 + 6D 旋转表示

作者的关键选择是：**直接预测对象 pose，而不是在视觉空间预测 flow/track**。  
这样做的好处是，中间表示和机器人动作空间更对齐，减少了从“视觉运动”再映射到“控制运动”的间接误差。

#### 2）动作生成模块
- 输入：观测特征 + 预测到的对象位姿特征
- 输出：未来 `T_a` 步机器人动作序列
- 动作表示：3D 平移 + 6D 旋转 + gripper width

这里对象位姿序列不是额外监督信号，而是**直接作为动作去噪条件**。  
因此，action head 不再只是“看图猜动作”，而是“结合对象未来运动，生成更匹配的动作”。

#### 3）执行方式
每个时间步：
- 先预测对象未来运动
- 再生成一段动作 chunk
- 执行前若干步后重新观测、循环

论文中还约束 `T_m >= T_a`，确保动作生成所依赖的对象运动预测覆盖整个动作时域。

### 核心直觉

**变化了什么？**  
把“对象未来运动”从动作网络内部的隐变量，提升成了显式中间条件。

**哪个瓶颈被改变了？**  
原本高熵、强耦合的 `观测 -> 动作` 映射，被拆成了更结构化的两段：
- `观测 -> 对象未来运动`
- `观测 + 对象未来运动 -> 动作`

这改变了两个约束：

1. **信息瓶颈变小**：动作网络不必自己从原始视觉里再猜一遍对象会怎么动  
2. **几何约束更强**：对象 pose 与末端 pose 共享相近表示，给动作生成提供了几何锚点和时间锚点

**为什么这会有效？**  
因为很多 manipulation 失败不是“不会生成动作”，而是**动作生成时缺少一个关于对象未来状态的明确条件**。  
一旦这个条件被显式化，很多与对象未来状态不一致的动作模式会被提前排除，动作去噪空间更小、更稳定，学习也更快。

**能力上发生了什么变化？**
- 更准的抓取定位
- 更稳的接触维持
- 更好的 6DoF 姿态控制
- 更快的策略收敛

### 策略层面的取舍

| 设计选择 | 改变了什么 | 收益 | 代价 |
| --- | --- | --- | --- |
| 用未来对象位姿作中间变量 | 把隐式对象状态推理从动作生成中拆出来 | 降低学习难度，增强可解释性 | 训练需要位姿监督 |
| 级联双扩散 | 同时建模对象轨迹与动作轨迹的多模态性 | 可插拔接入现有 diffusion policy | 推理变慢 |
| 直接用 pose 而不是 flow | 缩小 vision-motion gap | 更适合精细抓取和 6DoF 操作 | 对严重变形体不友好 |
| 与原 policy backbone 解耦 | 不重做感知与主干 | 易于集成到 DP / DP3 / RISE | 效果受基座视觉编码器误差影响 |

---

## Part III：证据与局限

### 关键证据信号

- **跨基座、跨 57 个仿真任务的对比信号**  
  在 DP 和 DP3 两类基座上都带来稳定提升，说明 MBA 的收益不是某个 backbone 的偶然适配，而更像是中间对象运动条件本身带来的系统性收益。  
  **关键指标**：DP 平均成功率 `53.6% -> 67.8%`，DP3 为 `71.3% -> 77.5%`。

- **高难任务增益更大的信号**  
  在 MetaWorld hard / very hard、Open Drawer、Pour Balls、Cut Clay 这类需要精细接触或 6DoF 控制的任务里，增益更明显。这直接支持了论文的主论点：**对象运动条件不是锦上添花，而是在高精度任务里真正改变动作生成质量的关键条件**。

- **真实世界迁移信号**  
  作者不仅在仿真中验证，还在 4 个真实任务上测试 DP、DP3、RISE。  
  其中最有说服力的是：RISE + MBA 在 Put Bread into Pot 上从 `80% -> 95%`，在 Open Drawer 上从 `37.5% -> 52.5%`，说明该中间条件对真实噪声和接触误差也有帮助。

- **相对 flow-based 条件的比较信号**  
  与 ATM 相比，MBA 在 Open Drawer 成功率上 `30% vs 5%`，在 Pour Balls 的平均入碗球数上 `3.60 vs 2.05`。这说明**直接建模 pose 条件**比在视觉空间建模 point flow 更适合精细操作和旋转相关控制。

- **学习效率信号**  
  学习曲线显示带 MBA 的策略更早达到高成功率、波动更小，说明显式对象运动确实降低了策略学习难度。

### 证据还不够强的地方
虽然对比覆盖面很广，但论文的证据主要还是**benchmark comparison**。  
它缺少更细的机制级 ablation，例如：
- 用 GT object pose 条件 vs 用 predicted pose 条件的差距
- `T_m` / `T_a` 变化对性能的影响
- pose 表示、编码器设计、级联顺序各自贡献多少

所以这篇论文更强地证明了“**这个范式实用**”，但对“**每个设计选择为何必要**”的因果拆解还不算充分。

### 局限性
- **Fails when**: 目标细长、严重遮挡、低视角点云缺失时，对象位姿估计误差会被放大；基座视觉编码器误差较大时，MBA 也会跟着积累误差并出现重复抓取。
- **Assumes**: 训练期可通过 MoCap 获得对象未来 6D 位姿监督；方法依赖 diffusion action head；真实系统中作者使用了 5 个 OptiTrack 相机做标注，且推理延迟约为 197.50 ms（不含视觉骨干），明显高于 DP 的 95.98 ms。
- **Not designed for**: 可变对象数量建模、强变形且无稳定 6D pose 的对象、无位姿标注的大规模低成本训练、极低时延闭环控制场景。

### 可复用组件
- **对象未来位姿扩散头**：适合作为任何 diffusion manipulation policy 的前置预测器
- **pose-to-action 条件接口**：把对象运动作为动作生成条件，而不是只做辅助监督
- **统一 pose 表示**：对象 pose 与末端 pose 对齐，有利于 6DoF 操作建模
- **plug-and-play 集成方式**：论文已展示其可接入 DP、DP3、RISE，这一点工程价值很高

### 一句话结论
这篇论文最重要的贡献，不是“又做了一个更强的 diffusion policy”，而是明确指出并验证了：**在 manipulation 里，先显式推理对象会怎么动，再生成动作，比直接从观测猜动作更稳、更快学、也更适合精细和 6DoF 操作。**

![[paperPDFs/Vision_Action_VA_Models_2024/IEEE_Robotics_and_Automation_Letters_2025/2025_Motion_Before_Action_Diffusing_Object_Motion_as_Manipulation_Condition.pdf]]