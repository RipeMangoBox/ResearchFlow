---
title: "HGDiffuser: Efficient Task-Oriented Grasp Generation via Human-Guided Grasp Diffusion Models"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/task-oriented-grasping
  - diffusion
  - guided-sampling
  - diffusion-transformer
  - dataset/OakInk-Shape
  - dataset/ACRONYM
  - repr/MANO
  - opensource/full
core_operator: "将人类示范中的接触区域与手腕朝向转成可微引导项，直接在扩散去噪过程中采样出满足任务约束的6-DoF平行夹爪抓取。"
primary_logic: |
  人类抓握示范 + 物体点云 → 先用任务无关抓取扩散模型建模稳定抓取先验，再用人类接触区域/手腕朝向对采样过程施加梯度引导，并通过DiT融合抓爪-物体-时间特征 → 单阶段输出稳定且任务一致的6-DoF平行夹爪抓取
claims:
  - "Claim 1: 在 340 个 OakInk-Shape 对象实例（33 类）上，HGDiffuser 的平均任务导向抓取成功率达到 81.21%，高于 RTAGrasp-MV 的 71.47% 和 Ours-TS 的 78.12% [evidence: comparison]"
  - "Claim 2: 相较 RTAGrasp-MV，HGDiffuser 将平均推理时间从 1.019s 降至 0.191s，减少 81.26% [evidence: comparison]"
  - "Claim 3: 在相同 OakInk 评测上，将 MLP 特征骨干替换为 DiT 后，任务无关抓取成功率从 71.35% 提升到 80.65%，并超过 GraspLDM 的 74.32%，且推理时间基本不变 [evidence: ablation]"
related_work_position:
  extends: "SE(3)-DiffusionFields (Urain et al. 2023)"
  competes_with: "RTAGrasp (Dong et al. 2025); DITTO (Heppert et al. 2024)"
  complementary_to: "FoundationPose (Wen et al. 2024); CPF (Yang et al. 2021)"
evidence_strength: moderate
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_HGDiffuser_Efficient_Task_Oriented_Grasp_Generation_via_Human_Guided_Grasp_Diffusion_Models.pdf
category: Embodied_AI
---

# HGDiffuser: Efficient Task-Oriented Grasp Generation via Human-Guided Grasp Diffusion Models

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2503.00508), [Project](https://sites.google.com/view/hgdiffuser)
> - **Summary**: 这篇工作把“人类示范中的任务约束”从后处理筛选改成扩散采样内的连续引导，在不需要大规模任务导向抓取标注的前提下，直接生成稳定且符合任务的 6-DoF 平行夹爪抓取。
> - **Key Performance**: OakInk-Shape 子集平均成功率 81.21%、推理 0.191s；相对 RTAGrasp-MV 推理时间降低 81.26%。

> [!info] **Agent Summary**
> - **task_path**: 单视角 RGB-D 人类抓握示范 + 物体点云/mesh -> 6-DoF 平行夹爪任务导向抓取位姿
> - **bottleneck**: 两阶段方法先在巨大 6-DoF 空间暴力采样稳定抓取，再用人类示范过滤，导致候选覆盖不足、推理慢且容易漏掉真正满足任务约束的抓取
> - **mechanism_delta**: 将人类接触区域与手腕朝向写成可微引导损失，直接加入扩散去噪更新，使采样过程本身朝“稳定 + 任务一致”的解收缩
> - **evidence_signal**: OakInk-Shape 上 HGDiffuser 以 81.21% 成功率、0.191s 推理同时优于 RTAGrasp-MV 的 71.47%、1.019s
> - **reusable_ops**: [显式约束引导的扩散采样, 基于 DiT 的抓爪-物体-时间特征融合]
> - **failure_modes**: [细长或把手类区域抓取不稳定, mesh 重建或位姿估计误差导致抓取配准偏差]
> - **open_questions**: [能否学习比 region/orientation 更丰富的任务约束, 能否在无同物体示范或不同末端执行器下保持泛化]

## Part I：问题与挑战

这篇论文研究的是一个**窄但非常实用**的任务导向抓取设定：机器人看到同一目标物体的人类抓握示范，并通过上游感知模块得到物体点云/mesh 与人手姿态，目标是输出一个适用于**6-DoF 平行夹爪**的抓取位姿，使其既**稳定**，又**符合任务意图**。

### 真正的难点是什么？

真正的瓶颈不是“能不能从人类示范里读出任务意图”，而是：

1. **任务信息进入得太晚**  
   现有主流方法先做任务无关抓取采样，再做示范约束过滤。  
   问题在于：如果第一阶段没有采到覆盖目标区域/方向的候选，第二阶段再强也筛不出来。

2. **6-DoF 抓取空间太大**  
   为了避免漏掉可行解，两阶段法只能不断增加候选数量，导致：
   - 推理变慢；
   - 仍有漏检风险；
   - 在单视角或局部观测条件下更容易失败。

3. **任务导向标注数据不够大**  
   端到端学习“人类示范 -> 机器人任务抓取”需要大量 task-oriented 数据，但这类数据昂贵且稀缺。  
   因此，问题不只是建模能力，而是**监督来源不足**。

### 输入/输出接口与边界条件

- **输入**：单视角 RGB-D 人类示范图像，以及通过多视角重建得到的目标物体点云/mesh。
- **输出**：一个 6-DoF 平行夹爪抓取位姿。
- **边界条件**：
  - 假设示范对象与执行对象是**同一目标物体**；
  - 依赖上游的 3D 重建、手部姿态估计、物体 6D pose 估计；
  - 关注的是**抓取变换模块**，不是完整感知系统本身。

### 为什么现在值得解决？

因为扩散模型提供了一个很合适的接口：

- 先用大规模**任务无关**抓取数据学“稳定抓取先验”；
- 再在**推理阶段**把人类示范提取出的任务约束加进采样过程。

这使得论文可以绕开“大规模任务导向标注不足”的根本障碍。

---

## Part II：方法与洞察

HGDiffuser 的整体思路可以概括为：

> **先学稳定抓取分布，再用人类示范把采样过程拉向任务一致的区域。**

### 方法拆解

#### 1）先学习任务无关的稳定抓取先验

作者先训练一个条件扩散抓取模型，输入物体点云，学习“这个物体上哪些 6-DoF 抓取是稳定的”。

这里的关键不是直接把抓取位姿当成一个裸的 SE(3) 向量去回归，而是：

- 用 **VN-PointNet** 编码物体点云；
- 把当前抓取位姿映射成一组**预定义 gripper points**，再做几何编码；
- 再加上扩散时间步特征。

这样做的好处是：抓爪和物体都在几何空间里表示，更容易做结构化融合。

#### 2）把人类示范变成“采样时的引导”

论文没有尝试直接训练一个大规模的任务导向条件生成器，而是把人类示范中的任务信息转成两个**显式、可微的约束**：

- **抓取区域约束**：由人手与物体的接触区域推一个目标抓取中心；
- **抓取方向约束**：由人手腕姿态推一个目标方向。

在每一步扩散去噪时，模型同时考虑两股信号：

- 来自任务无关扩散模型的“**稳定抓取分数**”；
- 来自人类示范约束的“**偏离任务要求的惩罚梯度**”。

因此，HGDiffuser 不再是“先采很多，再删很多”，而是“**一开始就朝着更可能满足任务的方向采样**”。

#### 3）用 DiT 提升抓爪-物体特征融合

作者还把原先 MLP 风格的特征骨干改成了 **DiT blocks**：

- 把 gripper points 当成 token；
- 把物体特征和时间步特征作为条件；
- 用 transformer 的注意力机制完成更细粒度的融合。

这部分带来的不是任务定义的改变，而是**score 估计更准**，从而让采样本身更可靠。

### 核心直觉

原来的两阶段方法，本质上是在做：

> **先覆盖整个稳定抓取流形，再从里面找任务一致的点。**

HGDiffuser 改成：

> **在采样过程中就把分布压缩到“稳定且任务一致”的子空间。**

这背后的因果链是：

- **What changed**：任务约束从“后验筛选条件”变成“前向采样偏置”。
- **Which bottleneck changed**：有效搜索分布从“所有稳定抓取”缩窄为“与人类示范一致的稳定抓取”，显著降低了采样熵和候选覆盖压力。
- **What capability changed**：模型不需要成百上千个候选，也能直接命中高质量 task-oriented grasp。

为什么这招有效？

- **扩散先验**负责“不偏离稳定抓取流形”；
- **人类引导**负责“在流形上选对区域和方向”；
- **DiT**进一步提高了“当前抓爪状态”和“物体几何上下文”之间的匹配精度。

所以它不是简单把两个模块拼起来，而是把“稳定性”和“任务性”分工得更合理。

### 战略取舍

| 设计选择 | 改变了什么 | 收益 | 代价/风险 |
|---|---|---|---|
| 单阶段 guided diffusion 代替两阶段 sample+filter | 任务约束进入采样过程，而不是事后过滤 | 少量采样即可得到可用解，显著降时延 | 引导能力受显式约束设计上限限制 |
| DiT 代替 MLP 骨干 | 从粗粒度特征拼接变成 token 级几何交互 | 抓爪-物体关系建模更强，score 更准 | 模型更复杂，训练更依赖数据/调参 |
| 使用完整物体点云 | 覆盖更多物体表面几何 | 降低局部观测漏抓风险 | 依赖多视角重建与精确配准 |
| 显式 region + orientation 约束 | 用简单、可微的任务先验指导采样 | 不需要大规模 task-oriented 标注 | 只能覆盖被这两类约束表达的任务结构 |

---

## Part III：证据与局限

### 关键实验信号

#### 1）比较信号：单阶段引导同时提升成功率和效率

在 OakInk-Shape 子集（340 个对象实例，33 类）上：

- DemoGrasp：**27.85%**
- RTAGrasp：**59.06%**
- RTAGrasp-MV：**71.47%**
- Ours-TS（作者的两阶段版本）：**78.12%**
- **HGDiffuser：81.21%**

更关键的是时间：

- RTAGrasp-MV：**1.019s**
- Ours-TS：**3.428s**
- **HGDiffuser：0.191s**

最强信号不是“只比别人高几点”，而是它在**成功率更高的同时还更快**。这直接支持了论文的核心主张：**把任务约束前置到采样过程，比后置筛选更有效。**

#### 2）采样量分析：方法的收益来自“分布改写”，不是单纯更强算力

作者专门比较了不同采样数下的表现：

- Ours-TS 从 100 个样本增加到 1000 个样本，成功率从 **71.18% -> 77.68%**，但时间从 **0.671s -> 6.793s**；
- HGDiffuser 只采 **1 个样本**就有 **81.21%**，采 100 个样本也只是 **81.38%**。

这说明 HGDiffuser 的核心收益不是“多试几次”，而是**每一次采样都更接近目标分布**。

#### 3）消融信号：DiT 的提升是真实存在的

在任务无关抓取评测里：

- GraspLDM：**74.32%**
- Ours w/o DiT：**71.35%**
- **Ours：80.65%**

且推理时间几乎不变（约 0.16~0.18s）。

这说明 DiT 不是装饰性改动，而是确实改善了抓爪-物体条件融合质量。

#### 4）真实世界信号：方法能落地，但端到端瓶颈转移到了感知

在 30 个 object-task pair 上：

- 感知成功：**26/30**
- 规划成功：**22/30**
- 动作成功：**20/30**

这表明方法具备实际可用性，但真实系统里的失败已不只来自生成器本身，还明显受上游感知链路影响。

### 局限性

- **Fails when**: 目标抓取位于细长、难稳定接触的区域时（如耳机头梁、剪刀把手、茶壶把手），任务无关抓取先验本身就可能给不出稳定解；另外当 mesh 重建不准、遮挡严重或 pose estimation 有误时，示范转移后的抓取会明显偏位。
- **Assumes**: 假设有同一物体的人类示范；假设可以获得多视角重建、手部姿态估计和物体 6D pose；假设“接触区域 + 手腕朝向”足以表达任务约束；还依赖一个在大规模任务无关数据（文中指向 ACRONYM）上训练好的抓取扩散先验。
- **Not designed for**: 不面向多指灵巧手抓取、仅用语言描述任务的抓取生成、强动态/重遮挡场景、以及无示范条件下学习全新任务语义。

此外还有两个证据层面的保守点：

1. **主量化基准主要集中在一个数据源（OakInk 子集）**，因此证据强度更适合记为 moderate，而不是 strong。  
2. **任务相关性评估带有人为判断成分**，虽然稳定性由 Isaac Gym 自动测，但 task compliance 仍非完全自动化指标。

### 可复用组件

- **显式约束引导的扩散采样**：先学通用先验，后用测试时约束改写采样分布。
- **gripper points tokenization**：把 SE(3) 抓取状态转换成几何 token，便于 transformer 融合。
- **低任务标注依赖的建模范式**：用廉价的任务无关数据训练生成器，把稀缺任务信息放到 inference-time guidance 中。

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_HGDiffuser_Efficient_Task_Oriented_Grasp_Generation_via_Human_Guided_Grasp_Diffusion_Models.pdf]]