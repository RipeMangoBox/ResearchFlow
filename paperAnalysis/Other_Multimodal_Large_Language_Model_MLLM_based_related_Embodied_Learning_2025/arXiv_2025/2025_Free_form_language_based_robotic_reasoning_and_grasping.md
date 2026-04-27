---
title: "Free-form language-based robotic reasoning and grasping"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/robotic-grasping
  - task/language-conditioned-grasping
  - mark-based-visual-prompting
  - keypoint-localization
  - contextualized-prompting
  - dataset/FreeGraspData
  - dataset/MetaGraspNetV2
  - opensource/no
core_operator: "先把拥挤场景中的物体离散成带编号的关键点候选，再让 GPT-4o 在受限动作空间内判断应直接抓目标还是先移除最上层无遮挡遮挡物。"
primary_logic: |
  顶视 RGB 图像 + 用户自由语言指令
  → Molmo 定位全部物体并生成带 ID 的标注图
  → GPT-4o 基于机器人任务上下文推理“目标物体 / 可先移除的无遮挡遮挡物”的 ID 与类别
  → LangSAM 分割对应实例、GraspNet 基于深度估计抓取位姿并执行
  → 重复直到目标可抓并被取出
claims:
  - "在 FreeGraspData 上，FreeGrasp 在含歧义和中高难度场景中显著优于 ThinkGrasp 的分割成功率，例如 Easy with Ambiguity 为 0.64±0.04 对 0.46±0.02，Medium without Ambiguity 为 0.40±0.04 对 0.13±0.03 [evidence: comparison]"
  - "在真实机器人最严格设置 (S,P,M) 下，ThinkGrasp 在全部 Medium/Hard 场景成功率均为 0，而 FreeGrasp 在 Medium without Ambiguity、Medium with Ambiguity 和 Hard with Ambiguity 上分别达到 0.20、0.20 和 0.10 [evidence: comparison]"
  - "Molmo 作为前端对象定位器时的 F1=0.91，高于 GPT-4o+LangSAM 的 0.85 和其后处理变体的 0.86，支撑了‘关键点化 + 编号提示’这一前端设计 [evidence: ablation]"
related_work_position:
  extends: "MOKA (Liu et al. 2024)"
  competes_with: "ThinkGrasp (Qian et al. 2024)"
  complementary_to: "SpatialPIN (Ma et al. 2024); SpatialCoT (Liu et al. 2025)"
evidence_strength: moderate
pdf_ref: "paperPDFs/Other_Multimodal_Large_Language_Model_MLLM_based_related_Embodied_Learning_2025/arXiv_2025/2025_Free_form_language_based_robotic_reasoning_and_grasping.pdf"
category: Embodied_AI
---

# Free-form language-based robotic reasoning and grasping

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2503.13082), [Project](https://tev-fbk.github.io/FreeGrasp/)
> - **Summary**: 这篇工作把“自由语言指代 + 拥挤场景遮挡推理”重写成一个带编号候选物体的离散决策问题，从而让零样本 VLM 更稳定地决定“现在先抓谁”。
> - **Key Performance**: FreeGraspData 上 Easy+Ambiguity 的 SSR 为 0.64 vs 0.46；真实机器人 Setting(P) 下 Medium+Ambiguity 的 SR 为 0.40 vs 0.10。

> [!info] **Agent Summary**
> - **task_path**: 顶视 RGB-D clutter bin + 自由语言目标描述 -> 当前应抓取的物体实例 -> 机械臂逐步移除遮挡并抓取目标
> - **bottleneck**: VLM 能理解语言，但在单张拥挤图像里对遮挡链和空间关系的判断不稳定，尤其在同类多实例时更容易失配
> - **mechanism_delta**: 先用 Molmo 把场景转成带 ID 的关键点集合，再用带任务上下文的 GPT-4o 在受限输出格式下选择“目标或顶层遮挡物”
> - **evidence_signal**: 在 FreeGraspData 与 UR5e 实机上，相比 ThinkGrasp 的优势主要出现在歧义和遮挡场景
> - **reusable_ops**: [ID编号视觉提示, 受限动作空间提示词]
> - **failure_modes**: [深遮挡下遮挡链判断错误, 物体移除后原始语言指代不再自适应]
> - **open_questions**: [如何显式建模3D遮挡图而非依赖2D提示, 如何在多步操作中维护可更新的场景记忆]

## Part I：问题与挑战

这篇论文解决的核心不是“看到目标后怎么夹”，而是**人在自然语言里说的那个物体，现在到底能不能夹；如果不能，第一步应该先移开什么**。

具体输入/输出接口是：

- **输入**：顶视 RGB-D 场景、用户自由语言指令
- **输出**：当前 step 应抓取的物体实例，以及最终把目标从红色 bin 转移到蓝色 bin 的动作序列

真正的难点有三层：

1. **自由语言指代不规范**：同一物体可能被说成不同名字，甚至名字不完全对。
2. **同类多实例歧义**：场景里可能有多个 Rubik’s cube、多个 screwdriver，仅靠类别词不够。
3. **遮挡推理**：目标不可抓时，系统要判断“哪个遮挡物是当前可抓且值得先移除的”。

作者的判断很清楚：**现在最该补的不是语言理解，而是视觉空间推理**。  
论文中的指令分析也暗示了这一点：不同用户对同一目标的句式差异很大，但 GPT-4o 对目标语义的识别总体仍较稳；真正掉点的是**遮挡关系与可抓取顺序**。

边界条件也很明确：

- 固定顶视相机
- 静态 bin-picking 场景
- 两指并联夹爪
- 每一步抓走一个物体后重新感知
- 不对 GPT-4o / Molmo / LangSAM / GraspNet 做任务特定训练

因此，这是一篇典型的**零样本、模块化 embodied grasp reasoning** 论文。

## Part II：方法与洞察

### 方法主线

FreeGrasp 的流水线可以概括为 5 步：

1. **Molmo 做对象定位**  
   直接让 VLM “point to all objects”，输出每个物体的 2D 关键点。

2. **编号式视觉提示**  
   在图上给每个物体打上唯一 ID，把原始拥挤图像变成“可枚举候选集”。

3. **GPT-4o 做抓取推理**  
   用专门写好的机器人任务提示词，要求模型只回答：
   - 目标如果无遮挡，就返回目标 ID
   - 否则返回一个“遮挡目标且自身无遮挡”的物体 ID

4. **LangSAM 做实例分割**  
   先用 GPT-4o 返回的类别名做语义分割，再用 ID 过滤到正确实例，避免纯点提示下的分割漂移。

5. **GraspNet 做位姿估计并执行**  
   基于深度恢复点云，裁剪出该实例，再估计抓取位姿，机械臂执行后重新进入下一轮。

### 核心直觉

这篇论文最关键的改变是：

**把“让 VLM 直接理解拥挤图像并自由回答”改成“先把场景离散成编号候选，再做受限动作选择”。**

这带来了三个因果层面的变化：

- **变化 1：从稠密视觉 grounding 变成离散候选选择**  
  以前的问题是“图里到底哪个像素区域对应用户说的东西”；现在变成“1,2,3,... 哪个编号对应用户要的物体或应先移除的遮挡物”。

- **变化 2：从开放式生成变成受限动作输出**  
  提示词明确告诉 GPT-4o：你只需要在“返回目标 / 返回顶层遮挡物”之间做决策，而不是生成一段解释或完整计划。

- **变化 3：把语义推理和几何执行拆开**  
  VLM 负责“选谁”，GraspNet 负责“怎么抓”。这样避免让单一模型同时承担语言、空间和精细几何控制。

为什么这有效？  
因为当前通用 VLM 的强项更像是**世界知识 + 指代解析 + 离散选择**，而不是**稳定的 3D 遮挡恢复**。编号标注给了它明确锚点，任务上下文又把“obstruction”这件事说得足够具体，于是模型不再需要从开放空间里胡乱生成答案。

一个很有价值的负结果是：  
作者尝试过 **CoT + scene graph**，但没有提升，反而容易 hallucinate 空间关系。这说明在这个任务里，**更长的推理链并不等于更好的空间因果结构**；先把输入形式改对，比“逼模型多想一步”更重要。

### 策略权衡

| 设计选择 | 解决的瓶颈 | 收益 | 代价/风险 |
| --- | --- | --- | --- |
| Molmo 关键点定位 + ID 标号 | 拥挤场景中难以稳定指代对象 | 把自由语言 grounding 变成离散选择题 | 重遮挡物体可能被漏检 |
| 上下文化 GPT-4o 提示词 | 开放式推理不稳定 | 稳定输出首个应抓取对象 | 仍依赖 GPT-4o 的隐式空间能力 |
| LangSAM 类名分割 + ID 过滤 | 同类多实例分割混淆 | 比纯 SAM 点提示更有语义约束 | 邻近实例仍可能粘连 |
| 模块化零样本设计 | 缺少任务专用训练数据 | 易替换组件、可快速验证 VLM 能力边界 | 误差会在模块间传播，整体时延较高 |

## Part III：证据与局限

### 关键证据信号

- **分析信号**：FreeGraspData 中，同一目标的三条人类指令在句式上差异很大，但 GPT score 整体较高。  
  **结论**：纯语言表述差异不是主要障碍，主要瓶颈还是空间/遮挡判断。

- **组件比较**：Molmo 的对象定位 F1 达到 **0.91**，高于 GPT-4o+LangSAM 的 **0.85/0.86**。  
  **结论**：先把物体变成编号候选，再交给 VLM 推理，是有效前端。

- **基准比较**：在 FreeGraspData 上，FreeGrasp 在含歧义和遮挡的设置里明显优于 ThinkGrasp。  
  代表性指标：Easy+Ambiguity 的 SSR **0.64 vs 0.46**；Medium w/o Ambiguity 的 SSR **0.40 vs 0.13**。  
  **结论**：优势主要来自“编号提示 + 受限推理”对歧义和 clutter 的缓解。

- **实机比较**：在最严格的真实机器人设置下，ThinkGrasp 在所有 Medium/Hard 场景 SR 都是 0；FreeGrasp 仍在部分 Medium/Hard 场景保留非零成功率。  
  **结论**：方法收益不只停留在静态数据集，也能传到真实执行链条。

- **系统代价**：平均总执行时间约 **15.39s**，其中 Molmo **9.12s**、GPT-4o **5.46s** 是主要瓶颈。  
  **结论**：这是一个可运行但仍偏慢的决策式系统，更像能力验证而非高频工业部署方案。

### 局限性

- **Fails when**: 遮挡链很深、多个相似实例紧贴、或顶视 2D 线索无法可靠恢复真实 3D 遮挡关系时；论文在 Hard 场景下的成功率依然偏低，说明问题远未解决。
- **Assumes**: 固定顶视 RGB-D 相机、并联夹爪、静态 bin-picking 场景、可调用闭源 GPT-4o，以及可用 Molmo / LangSAM / GraspNet 和约 24GB GPU；平均时延约 15.39s，也会影响复现与扩展。
- **Not designed for**: 动态场景、非抓取型操作（推、拨、双手协作）、以及在多步移除过程中自动改写人类指令或维护长期场景记忆的任务。

### 可复用组件

- **ID 编号式视觉提示**：适合任何“自由语言 -> 场景内单实例选择”的 embodied 任务
- **受限动作空间提示词**：把 VLM 输出约束为“目标 / 可先移除障碍物”
- **语义分割 + ID 过滤**：适合同类多实例场景
- **由 occlusion graph 反推抓取序列的标注方法**：可复用于类似 clutter manipulation 数据集构建

## Local PDF reference

![[paperPDFs/Other_Multimodal_Large_Language_Model_MLLM_based_related_Embodied_Learning_2025/arXiv_2025/2025_Free_form_language_based_robotic_reasoning_and_grasping.pdf]]