---
title: "Object-Centric Representations Improve Policy Generalization in Robot Manipulation"
venue: arXiv
year: 2025
tags:
  - Survey_Benchmark
  - task/robot-manipulation
  - task/imitation-learning
  - slot-attention
  - object-centric-learning
  - temporal-modeling
  - dataset/MetaWorld
  - dataset/LIBERO-90
  - opensource/no
core_operator: "在统一冻结编码器与策略骨干下，对全局、稠密与对象中心表征进行跨仿真、真实世界与 OOD 场景对比，以隔离视觉表征结构对机器人操作泛化的影响"
primary_logic: |
  机器人操作泛化评测目标 → 在 MetaWorld、LIBERO 与真实 SO-100 任务上固定视觉编码器并统一 BAKU/ACT 策略接口 → 比较 ID/OOD 成功率以及机器人视频再预训练、视频时序建模的增益 → 揭示对象中心 slot 表征在多物体交互和外观扰动下更稳健
claims:
  - "在 LIBERO 中，VIDEOSAUR* 相比最佳稠密基线 Theia 的总体成功率提升 9 个百分点，表明对象中心视频表征在复杂多物体场景中更有效 [evidence: comparison]"
  - "在真实世界五任务上，VIDEOSAUR* 达到 36/50（70%）总体成功率，高于最佳稠密基线的 50%，且在纹理与光照变化下 OOD 成功率最高 [evidence: comparison]"
  - "对 OCR 模型加入机器人混合数据再预训练可稳定提升控制表现：VIDEOSAUR* 相比原始 VIDEOSAUR 在 MetaWorld、LIBERO、LeRobot 分别提升 13、11、10 个百分点 [evidence: ablation]"
related_work_position:
  extends: "What Makes Pre-trained Visual Representations Successful for Robust Manipulation? (Burns et al. 2023)"
  competes_with: "Theia (Shang et al. 2024); DINOv2 (Oquab et al. 2024)"
  complementary_to: "BAKU (Haldar et al. 2024); ACT (Zhao et al. 2023)"
evidence_strength: moderate
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Object_Centric_Representations_Improve_Policy_Generalization_in_Robot_Manipulation.pdf
category: Survey_Benchmark
---

# Object-Centric Representations Improve Policy Generalization in Robot Manipulation

> [!abstract] **Quick Links & TL;DR**
> - **Summary**: 该文在统一的冻结视觉编码器+模仿学习框架下系统比较全局、稠密与对象中心表征，发现 slot 式对象中心表示，尤其是结合机器人视频再预训练与时序建模的 VIDEOSAUR*，能显著提升机器人操作策略的跨场景泛化。
> - **Key Performance**: VIDEOSAUR* 在 LIBERO 上比最佳稠密基线 Theia 高 **+9%**；在真实世界五任务上达到 **70%** 成功率，高于最佳稠密基线的 **50%**。

> [!info] **Agent Summary**
> - **task_path**: 多视角 RGB/视频 + 本体状态 + 任务文本 → 机器人连续动作/动作块
> - **bottleneck**: 全局或稠密视觉特征会把目标物体、背景和干扰物纠缠在一起，导致策略在纹理、光照和杂物变化下失稳
> - **mechanism_delta**: 将策略输入从整体/patch 特征替换为 slot 化对象中心表示，并测试机器人混合数据再预训练与时序对象建模是否能提升控制泛化
> - **evidence_signal**: VIDEOSAUR* 在 LIBERO 和真实世界总体成功率分别比最佳稠密基线高 9 和 20 个百分点，且在 OOD 纹理/光照扰动下掉点更小
> - **reusable_ops**: [统一冻结编码器评测接口, 机器人视频上重训 slot-attention/时序模块]
> - **failure_modes**: [小目标难以被稳定分槽, 强干扰物会与目标进入同一 slot]
> - **open_questions**: [如何给 slot 加语义或 affordance 绑定, 如何把对象表征与机器人动力学更紧密对齐]

## Part I：问题与挑战

这篇论文真正要回答的，不是“再设计一个更大的策略网络是否有效”，而是：

**当策略学习框架基本固定时，视觉表征本身是否决定了机器人操作的泛化上限？**

### 1. 现有瓶颈在哪里
当前很多 visuomotor policy 都采用类似范式：  
**视觉编码器 → 多模态融合 trunk → 动作头**。  
问题不在于这个范式不能学，而在于大多数视觉编码器输出的是：

- **全局特征**：把整张图压成一个向量；
- **稠密 patch 特征**：保留局部信息，但仍缺少明确“对象边界”。

这两类表示的共同问题是：  
**任务相关信息与无关信息混在一起**。  
对机器人操作而言，策略真正需要的是“哪个物体、在哪里、与机械臂/其他物体如何交互”；但全局/稠密表征往往同时编码了背景纹理、台面材质、灯光变化、无关杂物等 nuisance factors。结果就是：

- 训练域内能学会；
- 一旦出现 **新纹理 / 新光照 / 干扰物**，策略就掉得很快。

### 2. 为什么现在值得做
这件事之所以现在重要，有两个背景：

1. **预训练视觉模型已成为机器人学习默认组件**  
   机器人领域越来越依赖冻结的 foundation encoder，因此“表征选型”本身已经成为一等设计变量。

2. **对象中心学习终于能处理真实图像/视频**  
   早期 OCR 更多停留在 toy scene、重建或分解任务；DINOSAUR/VIDEOSAUR 这类方法才让 slot-based 表征开始具备进入真实视觉场景的可能。

因此，这篇论文的核心动机很明确：  
**过去 OCR 在机器人控制里证据不足，尤其缺少跨仿真、复杂多物体、真实世界、以及 OOD shift 的统一评测。**

### 3. 输入/输出接口与边界条件
论文评测的接口很清楚：

- **输入**：相机视觉观测（图像/视频特征）+ proprioception + 任务文本
- **输出**：连续控制动作或 action chunks

但边界也很明确：

- 只研究 **模仿学习**
- 视觉编码器 **冻结**
- 主要是 **tabletop manipulation**
- 不讨论端到端 VLA 联合训练
- 不把语义 grounding 或 affordance 显式建模进 slot

所以它评测的是：  
**“在固定策略学习范式下，不同视觉表示结构对泛化的影响”**，而不是完整机器人系统的最终上限。

---

## Part II：方法与洞察

这篇论文的方法贡献不在“发明了全新的 OCR 架构”，而在于建立了一个相对干净的比较框架，去隔离“表征结构”这一因果旋钮。

### 1. 统一评测框架
作者比较了 3 类视觉表示：

- **Global**：如 DINOv2 global、VC-1
- **Dense**：如 ResNet-50、R3M、Theia、DINOv2 dense
- **Object-Centric**：DINOSAUR、VIDEOSAUR 及其机器人再预训练版本

为了尽量公平，他们做了两件关键事：

- **所有视觉编码器在策略训练中都冻结**
- **下游策略骨干尽量统一**
  - 仿真：用 BAKU
  - 真实世界：用 ACT（LeRobot 实现）

这样比较出来的差异，主要就能归因到**视觉表示**，而不是谁的 policy head 更强。

### 2. 评测覆盖面
作者把场景难度分成了三层：

- **MetaWorld**：相对简单、单物体/结构化任务
- **LIBERO-90**：复杂多物体、多域场景
- **真实世界 SO-100**：低成本机械臂上的 5 个桌面任务

并且专门做了 OOD 测试：

- distractors
- new textures
- lighting changes

这使得论文不只回答“谁在 ID 上更高”，而是回答：  
**哪种表征在视觉分布变化时更不容易崩。**

### 3. 机器人视频再预训练
作者没有停在“直接拿现成 OCR 模型来比”，而是进一步测试了一个很重要的变量：

- 原始 DINOSAUR / VIDEOSAUR 主要训练在自然图像/视频上；
- 作者额外用 **BridgeData V2 + Fractal + DROID** 的混合机器人视频，对 OCR 的 attention 模块做自监督再训练。

这一步的意义是把 OCR 的对象分解偏置，从“互联网视频里的运动/物体统计”拉向“机器人操作场景里的物体/视角/运动统计”。

### 核心直觉

**这篇论文的关键变化是：把策略看到的视觉世界，从“像素/patch 的连续纹理场”改成“有限个可竞争的对象 slot”。**

对应的因果链是：

- **从全局/稠密表示 → slot 化对象表示**
- 改变了信息瓶颈：  
  视觉输入不再平均地编码整幅图，而是被压到一组离散实体上
- 改变了约束：  
  Slot Attention 的竞争机制迫使不同 slot 去分担不同实体/区域
- 带来的能力变化：  
  策略更容易聚焦于可操作对象及其关系，而不是背景外观细节

再进一步：

- **从自然视频 OCR → 机器人混合数据再预训练 OCR**
  - 改变的是域偏置
  - slot 更容易对准机器人场景里的稳定对象、机械臂和常见运动模式

- **从单帧 OCR → 视频 OCR**
  - 改变的是时序一致性
  - 物体在连续帧中的持续性更好，利于操作控制而不是只做静态分解

所以作者想证明的不是“slot 本身神奇”，而是：

> 机器人操作需要的是对象级、时序稳定、对外观变化不敏感的表示；  
> OCR 恰好把这个偏置直接写进了视觉表征里。

### 4. 策略层面的取舍

| 设计选择 | 改变的约束/信息流 | 带来的能力 | 代价/风险 |
|---|---|---|---|
| 全局特征 | 强压缩整图 | 简洁、易接入策略 | 背景与目标严重纠缠 |
| 稠密特征 | 保留局部细节 | 对局部操作较友好 | 仍缺显式对象边界，易受纹理影响 |
| OCR slots | 有限实体集 + 竞争分配 | 多物体场景、更强纹理/光照泛化 | 小目标和强干扰物可能混槽 |
| 视频 OCR | 显式利用跨帧一致性 | 真实世界与复杂任务提升更明显 | 训练更复杂，对数据分布更敏感 |
| 机器人再预训练 | 把对象分解偏置对齐机器人域 | 下游控制更稳、更贴近操作场景 | 需要大规模机器人视频和额外算力 |

一个很重要的洞察是：  
**OCR 的优势不是在所有场景都碾压，而是在“多物体 + OOD + 真实噪声”条件下开始明显拉开。**  
这说明它更像是在修复泛化瓶颈，而不是单纯提高 in-domain 拟合能力。

---

## Part III：证据与局限

### 1. 关键证据信号

#### 信号 A：复杂多物体场景里，OCR 优势最明显
最强的实验信号来自 **LIBERO** 和 **真实世界**：

- 在 **LIBERO** 上，VIDEOSAUR* 比最佳稠密基线 Theia 高 **9 个百分点**
- 在 **真实世界五任务** 上，VIDEOSAUR* 达到 **70%**，最佳稠密基线约 **50%**

这说明能力跳跃主要发生在：

- 多物体交互更复杂
- 感知噪声更强
- 视觉干扰更多

而不是只在简单仿真里刷分。

#### 信号 B：OOD 结果支持“对象结构比外观细节更稳”
在 MetaWorld 与真实世界的 OOD 测试中，OCR 尤其在：

- **纹理变化**
- **光照变化**

下更稳。  
这支持作者的解释：slot 表征更偏向对象结构，而不是死记背景或低层纹理。

不过结果也给出一个重要反例：

- **Theia 在 distractor 场景里常常更强**

这暗示：  
OCR 虽然擅长过滤外观扰动，但对“哪些对象语义上重要、哪些只是干扰物”的判断还不够强；而 Theia 这类融合了 CLIP 等语义信号的表征，在“忽略无关物体”这件事上可能更占优。

#### 信号 C：再预训练和时序建模不是小修小补，而是主效应
作者比较 OCR 变体后发现：

- **VIDEOSAUR* 相比原始 VIDEOSAUR**
  - MetaWorld: +13
  - LIBERO: +11
  - LeRobot: +10

并且：

- **VIDEOSAUR* 相比 DINOSAUR***
  - LIBERO: +9
  - LeRobot: +26

这两个信号分别对应两个关键结论：

1. **机器人域再预训练很重要**
2. **时序对象建模很重要**

也就是说，论文真正有效的不是“只要做 slot 就行”，而是：
**slot + 机器人域对齐 + 视频时序一致性** 这个组合。

### 2. 这篇论文真正说明了什么
它最有价值的结论可以浓缩成一句话：

> 在机器人操作中，视觉表征的“结构化程度”会直接决定策略在分布偏移下的稳定性，而对象中心表示是一个比全局/稠密特征更合适的归纳偏置。

更具体地说，能力提升主要体现在：

- 从单物体简单场景走向多物体复杂场景
- 从仿真 ID 走向真实世界与 OOD
- 从“能学会”走向“学会后不容易因外观变化而失效”

### 3. 局限性

- **Fails when**: 目标物体很小、细长或难分割时，slot 难以稳定对齐；在强干扰场景中，目标与干扰物可能落入同一 slot。论文也明确提到 MetaWorld 的 Shelf Place 等小物体任务，以及 hammer distractor failure case 中出现了 slot 混叠。
- **Assumes**: 依赖冻结的预训练视觉骨干、模仿学习示范数据、以及机器人视频混合数据的再预训练；OCR 再训练使用约 **188k 轨迹**，并需要 **1×H100 / 约 80 GPU-hours**。真实世界实验还假设相对固定的桌面视角、两路相机和任务文本条件。
- **Not designed for**: 开放词汇语义 grounding、显式 affordance 绑定、与机器人动力学联合建模、移动操作或开放环境中的通用具身智能控制。

### 4. 可复用组件
这篇论文即使不把其结论当作最终答案，也留下了几个可复用操作：

- **统一视觉编码器适配层**：把 global / dense / slot 特征都接入同一类 policy；
- **机器人混合数据 OCR 再预训练配方**：可直接迁移到别的 manipulation benchmark；
- **跨 ID/OOD/真实世界的表示评测协议**：适合做 representation selection，而不只是 policy selection。

另外，论文声称框架开源，但文中未给出明确代码链接，  
因此从严格可复现角度看，**代码可获得性仍待核实**。

## Local PDF reference

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Object_Centric_Representations_Improve_Policy_Generalization_in_Robot_Manipulation.pdf]]