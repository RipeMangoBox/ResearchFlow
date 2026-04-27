---
title: "Dreamitate: Real-World Visuomotor Policy Learning via Video Generation"
venue: arXiv
year: 2024
tags:
  - Embodied_AI
  - task/robot-manipulation
  - task/visuomotor-policy-learning
  - diffusion
  - 3d-tracking
  - tool-trajectory-transfer
  - dataset/Rotation
  - dataset/Scooping
  - dataset/Sweeping
  - dataset/Push-Shape
  - opensource/promised
core_operator: 将互联网预训练视频扩散模型微调为“给定新场景生成持工具的人类操作视频”，再从生成视频中跟踪工具3D轨迹并直接转成机器人动作。
primary_logic: |
  双目场景图像 + 任务人类演示视频 → 微调视频扩散模型生成新场景下的人类持工具操作双目视频 → 用CAD先验做工具6D跟踪并经逆运动学执行 → 机器人完成操作
claims:
  - "Dreamitate在4个真实机器人操作任务上都优于Diffusion Policy，其中旋转成功率92.5% vs 55%，舀取85% vs 55%，清扫92.5% vs 12.5%，Push-Shape 的 mIoU 为0.731 vs 0.550且平均旋转误差为8.0° vs 48.2° [evidence: comparison]"
  - "在旋转任务中将训练数据缩减到2/3和1/3后，Dreamitate的成功率仍基本稳定，而Diffusion Policy显著退化，说明其更能利用预训练视频先验提升小样本泛化 [evidence: comparison]"
  - "该方法无需机器人动作标注、仅凭人类工具演示即可产生可执行的3D工具轨迹，但在工具小、遮挡重或追踪不稳时更容易失败 [evidence: case-study]"
related_work_position:
  extends: "Stable Video Diffusion (Blattmann et al. 2023)"
  competes_with: "Diffusion Policy (Chi et al. 2023)"
  complementary_to: "N/A"
evidence_strength: moderate
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Video_Generators_Are_Robot_Policies.pdf
category: Embodied_AI
---

# Dreamitate: Real-World Visuomotor Policy Learning via Video Generation

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2406.16862), [Project](https://dreamitate.cs.columbia.edu/)
> - **Summary**: 这篇论文把机器人策略从“视觉直接回归动作”改成“先生成一段人在该场景中用工具完成任务的视频，再把工具轨迹提取出来执行”，从而把互联网视频中的人类操作先验迁移到真实机器人控制。
> - **Key Performance**: 清扫任务成功率 **92.5% vs 12.5%**（相对 Diffusion Policy）；Push-Shape 的 **mIoU 0.731 vs 0.550**，平均旋转误差 **8.0° vs 48.2°**。

> [!info] **Agent Summary**
> - **task_path**: 双目场景图像（可含目标mask） -> 生成的人类持工具操作视频 -> 工具SE(3)轨迹 -> 机器人执行
> - **bottleneck**: 互联网视频里有人类操作先验，但人手到机械臂存在 embodiment gap；传统BC又依赖昂贵的机器人动作标注且泛化弱
> - **mechanism_delta**: 用“视频生成 + 工具跟踪”替代“图像到动作直接回归”，把跨 embodiment 的映射压缩到共享的刚性工具轨迹上
> - **evidence_signal**: 4个真实世界任务上稳定超过 Diffusion Policy，且在少数据设置下退化更小
> - **reusable_ops**: [stereo-conditioned video diffusion finetuning, CAD-based 6D tool tracking]
> - **failure_modes**: [heavy tool/end-effector occlusion, non-rigid or fine-grained manipulation beyond rigid-tool control]
> - **open_questions**: [how to make it closed-loop and real-time, how to remove the known-CAD rigid-tool assumption]

## Part I：问题与挑战

这篇工作的真问题，不是“能不能让视频模型参与机器人决策”，而是：

**怎样把互联网规模的人类操作视频先验，变成机器人里可执行、可泛化的控制策略。**

### 1. 真正瓶颈在哪里
传统 visuomotor policy（如 BC / Diffusion Policy）把问题写成：

- 输入：场景图像
- 输出：机器人动作

这条路径的问题是两层叠在一起的：

1. **数据瓶颈**：要有机器人动作标注，通常来自遥操作或机器人演示，采集成本高。
2. **泛化瓶颈**：模型必须同时学会视觉理解、接触几何、动作选择和机器人 embodiment，容易对训练场景过拟合。

另一方面，视频生成模型确实有巨大先验，但直接拿来做机器人控制又卡在另一个核心障碍：

- **人类视频多，但人手动作难直接转成机器人动作**
- **机器人视频可执行，但预训练数据规模远小于人类互联网视频**

所以真正的 bottleneck 不是“缺一个更强的 policy backbone”，而是：

> **如何在不丢失人类视频先验的前提下，把人类操作转成机器人可执行的控制接口。**

### 2. 这篇论文的输入/输出接口
Dreamitate 的接口很明确：

- **输入**：新场景的双目首帧图像（Push-Shape 任务还额外给目标 mask）
- **输出**：工具的 6D 轨迹，再转成机器人 SE(3) 动作序列

它不是直接输出关节动作，也不是先做语言计划，而是先生成一个**可视化的执行视频计划**。

### 3. 边界条件
这篇方法能成立，有几个明确前提：

- 每个任务都要**单独微调**一个视频模型
- 需要**双目相机标定**
- 需要**已知 CAD 的刚性工具**
- 需要工具在生成视频里**能被稳定跟踪**
- 执行是**开环**的，不是实时闭环控制

### 4. 为什么是现在
因为现在的视频扩散模型已经具备两个关键能力：

- 能从大规模互联网视频学到**人-工具-物体交互先验**
- 能做条件视频生成，把“给定场景 -> 未来操作过程”表达出来

Dreamitate 利用的不是视频模型“会生成好看视频”这件事，而是它已经学到大量**常识性操作分布**。

---

## Part II：方法与洞察

Dreamitate 可以概括成三步：

1. **Dream**：给定新场景图像，生成一段“人在这个场景里用工具完成任务”的双目视频  
2. **Track**：在生成视频里跟踪工具的 6D 轨迹  
3. **Execute**：把这段轨迹映射为机器人动作执行

### 方法骨架

#### Step 1：用人类工具演示微调视频扩散模型
作者基于 **Stable Video Diffusion**，用少量任务级人类演示视频做微调。

关键点不在“人类演示”本身，而在于：

- 人类和机器人都使用**功能对应的工具**
- 这个工具有**已知 CAD**
- 演示视频是**双目**的，便于后续恢复 3D 轨迹

为了降低训练成本，作者只微调 attention 层，编码器/解码器冻结。

#### Step 2：生成双目操作视频
测试时输入新场景的双目首帧，模型输出一段未来视频：

- 前半段帧对应 view 1
- 后半段帧对应 view 2

这一步本质上是在做：

> “给定当前场景，想象一个可行的人类工具操作过程”

#### Step 3：从视频里抽出几何动作
作者不用视频直接驱动机器人，而是进一步做一个离散化的几何抽象：

- 用 **MegaPose** + 工具 CAD 在生成视频中逐帧估计工具 6D pose
- 用双目一致性恢复更稳定的 3D 轨迹
- 再把工具轨迹交给机器人执行

于是，视频只是**中间计划表示**，真正执行的是显式轨迹。

### 核心直觉

以前的方法做的是：

**场景图像 → 机器人动作**

Dreamitate 改成：

**场景图像 → 人类工具操作视频 → 工具轨迹 → 机器人动作**

这个变化真正改变了三个东西：

1. **分布变了**  
   以前要学的是机器人专属动作分布；现在先学的是更贴近互联网预训练分布的**人类工具操作视频分布**。

2. **约束变了**  
   以前模型要同时解决“操作语义 + embodiment 映射”；现在把跨 embodiment 的部分收缩为一个共享接口：**刚性工具轨迹**。

3. **能力边界变了**  
   以前泛化靠有限机器人示范；现在可以借助视频模型已有的“人如何在杂乱视觉环境中使用工具”的先验，因此对新物体、新背景、新干扰更稳。

更因果地说，这个设计之所以有效，是因为作者发现：

> 操作任务里最难迁移、也最关键的，不是整个人手，而是**末端与物体接触时的几何行为**。  
> 只要让人和机器人共享“工具”这一末端接口，剩下的 embodiment 可以交给逆运动学和机械执行去解决。

### 战略取舍

| 设计选择 | 解决了什么 | 代价 |
|---|---|---|
| 用人类持工具视频，而不是机器人动作监督 | 复用互联网人类视频先验，降低数据采集成本 | 需要设计共享工具，且工具必须可跟踪 |
| 生成视频后再跟踪，而不是直接回归动作 | 得到可解释的中间计划，并能利用生成模型的 multimodal 先验 | 推理更重，链路更长 |
| 双目视频生成 | 让工具轨迹可恢复为 3D，可直接执行 | 需要双相机标定和更复杂的数据处理 |
| 用工具轨迹作为共享接口 | 显著缩小 hand-to-robot embodiment gap | 只适用于刚性工具主导的任务 |
| 开环整段执行 | 执行逻辑简单，计划可视化 | 对中途扰动、接触误差和跟踪偏差不够鲁棒 |

---

## Part III：证据与局限

### 关键证据信号

#### 1. 跨四个真实任务都赢过 Diffusion Policy
作者在四类任务上做了真实机器人对比，且训练/测试物体不重合，重点看的是**泛化**而不是记忆。

最强信号有两个：

- **Sweeping**：92.5% vs 12.5%  
  这是最能说明问题的一组结果，因为该任务有明显多模态性、障碍物和干扰项。Dreamitate 显示出更强的“在复杂视觉场景中选一条可行工具路径”的能力。

- **Push-Shape**：mIoU 0.731 vs 0.550；旋转误差 8.0° vs 48.2°  
  这说明它不仅能把物体“推过去”，还能更好地处理**姿态调整**，即更复杂的接触结果预测。

#### 2. 少数据下更稳
在 rotation 任务上把训练集降到 2/3 和 1/3 后：

- Dreamitate 性能下降很小
- Diffusion Policy 明显退化

这条证据支持论文的核心叙事：  
**视频预训练先验确实在替代一部分任务数据需求。**

#### 3. 不是只会“生成好看视频”，而是真的能抽出可执行轨迹
Scooping 和 Push-Shape 都说明一点：  
生成视频里的工具轨迹，经过双目跟踪后，已经足够支持**真实 3D 操作**，而不是只能做可视化 planning demo。

### 局限性

- **Fails when**: 工具或末端执行器在生成视频中严重遮挡；工具尺寸太小导致跟踪不稳；长时开环执行中出现未预料接触变化时，轨迹会偏离真实最优行为。
- **Assumes**: 已有互联网预训练视频模型；任务级人类工具演示；已知刚性工具 CAD；双目相机标定；MegaPose 级别的6D跟踪能力；机器人能稳定复现工具轨迹。
- **Not designed for**: 灵巧手指级操作、柔性工具/可变形物体操作、强实时反馈控制、无已知工具模型的通用操控。

### 资源与复现依赖
这篇方法的可复现性还受几个现实条件约束：

- 依赖 **Stable Video Diffusion** 作为强预训练底座
- 视频生成推理成本高，作者也明确说**实时闭环控制不可行**
- 部分 tracking 初始化在失败时需要**人工修正框**
- 每个任务要重新微调一个模型，暂时不是统一多任务策略
- 代码和数据在文中表述为**将发布**，因此当前更接近 `opensource/promised`

### 可复用组件
这篇论文最值得迁移的，不一定是整套系统，而是三个可复用算子：

1. **Tool-as-interface**：用共享工具几何接口缩小 human/robot embodiment gap  
2. **Generate-then-track**：把生成模型输出转成显式几何轨迹  
3. **Stereo-conditioned video planning**：让“未来视频”变成可执行 3D 中间表示

## Local PDF reference

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Video_Generators_Are_Robot_Policies.pdf]]