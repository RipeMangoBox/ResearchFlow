---
title: "SHADOW: Leveraging Segmentation Masks for Cross-Embodiment Policy Transfer"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/robot-manipulation
  - segmentation-mask
  - data-editing
  - diffusion
  - dataset/MimicGen
  - dataset/RoboMimic
  - opensource/no
core_operator: 用与末端位姿对齐的源/目标机器人复合分割掩码替换真实机器人像素，从而把跨机器人训练与测试观测对齐到近似同一分布。
primary_logic: |
  单源机器人第三视角 RGB 示教轨迹 + 源/目标机器人的 CAD、运动学和相机标定 → 训练时将源机器人涂黑并叠加与其末端位姿对齐的目标机器人 mask，测试时对目标机器人执行对称替换 → 输出无需目标机器人示教数据即可部署的跨 embodiment 笛卡尔末端动作策略
claims:
  - "在仿真中的全部已报告跨 embodiment 设定上，Shadow 都优于 Mirage，并在 5/6 个任务上达到与源机器人上界近似无退化的表现 [evidence: comparison]"
  - "在实机 4 个任务、两种迁移场景下，Shadow 相比最强基线 Mirage 的平均成功率提升超过 2× [evidence: comparison]"
  - "在 Stack 任务消融中，仅遮掉源机器人无法迁移到 UR5e/IIWA（0.02/0），而叠加目标机器人 mask 后提升到 0.97/0.97，说明目标 mask 叠加是关键因子 [evidence: ablation]"
related_work_position:
  extends: "Mirage (Chen et al. 2024)"
  competes_with: "Mirage (Chen et al. 2024)"
  complementary_to: "CACTI (Mandi et al. 2022); GenAug (Chen et al. 2023)"
evidence_strength: strong
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_SHADOW_Leveraging_Segmentation_Masks_for_Cross_Embodiment_Policy_Transfer.pdf
category: Embodied_AI
---

# SHADOW: Leveraging Segmentation Masks for Cross-Embodiment Policy Transfer

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2503.00774), [Project](https://shadow-cross-embodiment.github.io)
> - **Summary**: 这篇工作把“跨机器人迁移”从学一个大而泛化的多形态策略，改成了一个更直接的数据编辑问题：把训练和测试中的机器人都替换成同一种复合分割掩码，从而显著减少 embodiment 带来的视觉分布漂移。
> - **Key Performance**: 实机相对 Mirage 平均成功率提升超过 **2×**；仿真中在 **5/6** 个任务上达到与源机器人上界近乎无退化的迁移表现。

> [!info] **Agent Summary**
> - **task_path**: 单源机器人第三视角 RGB 示教 + 源/目标 CAD/运动学/标定 -> 目标机器人笛卡尔末端动作策略
> - **bottleneck**: 机器人外观和夹爪形态变化会造成严重视觉分布漂移，使仅用单一机器人数据训练的视觉策略在新 embodiment 上几乎失效
> - **mechanism_delta**: 用与当前末端位姿对齐的“源 mask + 目标 mask”复合阴影替代真实机器人像素，让训练和测试都落到近似同一观测分布
> - **evidence_signal**: 仿真 6 任务与实机 4 任务中全面优于 Mirage，且关键消融证明“仅遮源机器人”不足以迁移
> - **reusable_ops**: [末端位姿对齐渲染, 训练测试对称式 mask 替换]
> - **failure_modes**: [相机标定误差导致 mask 错位, 宽物体或高精度抓取时末端定位误差放大]
> - **open_questions**: [能否单策略覆盖多个目标 embodiment, 能否与 scene randomization 结合泛化到新场景]

## Part I：问题与挑战

这篇论文解决的是一个非常具体但现实的问题：**只有源机器人有示教数据，目标机器人完全没有数据，能否把视觉策略直接迁移过去？**

### 真正难点是什么？
真正的瓶颈不是动作空间本身，而是**观测空间的 embodiment shift**：

- 不同机械臂和夹爪在图像里的颜色、轮廓、连杆比例、遮挡关系都不同；
- 视觉策略很容易把“机器人长什么样”当成任务线索；
- 一旦测试时机器人外观变了，哪怕任务、相机、物体摆放都没变，策略也会崩。

论文里的仿真结果几乎把这个问题钉死了：**直接用源机器人原始图像训练，再在目标机器人原始图像上测，成功率几乎接近 0。**

### 为什么现在值得解决？
因为机器人数据正在快速增长，但这些数据天然分散在不同硬件上。继续要求“每来一个新机器人就重新采很多数据”不现实。相比再去等更大规模多 embodiment 数据集，这篇工作想回答的是一个更节省数据的问题：

- **只用一个源机器人数据集**
- **目标机器人零示教**
- **仍然做跨 embodiment 迁移**

### 输入 / 输出接口
- **输入**：固定第三视角相机下的 RGB 历史帧；另外在数据编辑阶段可访问源/目标机器人的 CAD、运动学、关节角和相机标定。
- **输出**：目标机器人执行的**绝对笛卡尔末端位姿动作**。
- **适用边界**：源和目标机器人必须能用近似相同的末端操作策略完成任务；场景、光照、相机视角基本固定；作者主要关注单臂 manipulation。

一句话说，本文不是在解决“所有机器人都能互相迁移”，而是在解决：**当任务策略在末端空间基本共享时，如何消掉机器人外观差异带来的视觉 OOD。**

## Part II：方法与洞察

Shadow 的核心不是换一个更强的 policy backbone，而是**在观测层做人为的分布对齐**。作者用的是 Diffusion Policy，但真正的新意在输入编辑。

### 方法流程

#### 训练时
对每张源机器人图像做两步：

1. **把源机器人像素分割出来并统一涂成固定颜色**（论文里用纯黑）；
2. 根据当前源机器人的末端位姿，**渲染一个目标机器人**到相同末端位姿，再把它的 segmentation mask 也叠上去。

于是训练图像里，真实机器人不再以 RGB 外观出现，而是变成了一个**源+目标的复合黑色轮廓**。

#### 测试时
对目标机器人执行镜像式处理：

1. 先把目标机器人真实像素涂黑；
2. 再渲染一个处在相同末端位姿的源机器人 mask 叠上去。

这样测试输入也变成了同样风格的**源+目标复合 mask 图像**。

#### 学习器
在这个编辑后的观测上训练 imitation policy。本文用的是 **Diffusion Policy**，但方法本身并不依赖 diffusion 才成立。

### 核心直觉

**变化了什么？**  
把“真实机器人 RGB 外观”替换成“与末端位姿绑定的统一复合 mask 表示”。

**改变了哪个瓶颈？**  
它直接削弱了策略对 embodiment-specific 视觉细节的依赖，把机器人颜色、材质、关节外形等“干扰变量”从输入里拿掉，同时保留：

- 机器人大致占据哪些像素；
- 夹爪/末端相对物体的位置关系；
- 场景与背景的真实 RGB 细节。

**带来了什么能力变化？**  
策略不再需要从单源数据里“猜”如何对新机器人外观泛化，而是始终在近似统一的观测分布上做决策，因此能零目标数据地迁移到新机器人。

### 为什么这个设计有效？
因果上，这个方法成立依赖两个判断：

1. **对这些 manipulation 任务，机器人本体 RGB 并不是核心信息。**  
   附录实验显示，把源机器人直接替换成 segmentation mask，在源机器人自身上也几乎不掉性能。这说明策略主要依赖的是物体、场景和末端交互几何，而不是机械臂纹理。

2. **跨 embodiment 真正该对齐的是“末端语义”，不是整张图的像素细节。**  
   Shadow 通过“相同末端位姿下的双机器人 mask”把源和目标都投影到了一个共享中间表示上。  
   这对“同机械臂不同夹爪”尤其重要，因为夹爪长度/控制点不同会导致同一动作对应不同法兰位置；单纯 inpainting 很容易在这里失真，而 Shadow 用几何对齐直接处理掉。

3. **它避免了 inpainting 的伪影问题。**  
   Mirage 的问题不是思路不对，而是生成式填补会在精细接触处和复杂背景中引入失真。Shadow 则主动丢弃机器人 RGB，不去“生成逼真图像”，只保留任务需要的几何占位。

### 战略取舍

| 设计选择 | 解决的问题 | 收益 | 代价 / 边界 |
| --- | --- | --- | --- |
| 用复合 segmentation mask 替代机器人 RGB | embodiment 外观差异导致的视觉分布漂移 | 输入分布强对齐，简单稳定 | 依赖准确 CAD、运动学、标定 |
| 训练/测试对称编辑 | 训练看见 source，测试看见 target 的不一致 | train-test 更接近同分布 | 每个 source-target 对通常要单独训练 |
| 不做 inpainting，只保留 mask | 避免背景和接触区伪影 | 更稳、更轻量、接触边界更清晰 | 放弃机器人细节纹理信息 |
| 使用绝对笛卡尔末端动作 | 给不同机器人共享动作语义 | 更容易跨机器人执行同策略 | 前提是两种 embodiment 能用相似物理策略完成任务 |

一个很重要的理解是：**Shadow 不是在“学会跨 embodiment 泛化”，而是在“先把 embodiment 差异从观测里手工规约掉”。**  
这也是它为什么能在单源数据下表现得如此数据高效。

## Part III：证据与局限

### 关键证据

#### 1) 比较信号：直接迁移几乎失败，说明问题真实存在
在仿真中，naive baseline（源机器人原图训练，目标机器人原图测试）在各任务和目标 embodiment 上几乎都接近 0 成功率。  
这说明本文抓到的不是小幅退化，而是**跨机器人视觉 shift 会让策略几乎完全失效**。

#### 2) 比较信号：Shadow 稳定优于 Mirage
- 在仿真 6 个任务、多个目标机器人/夹爪设定上，**Shadow 全面优于 Mirage**；
- 且在 **5/6** 个任务上，Shadow 的目标机器人表现接近“源训练源测试”的上界；
- 实机 4 个任务、两类迁移场景中，**平均成功率超过 Mirage 的 2×**。

最值得注意的是：作者不只测了“不同机器人 + 不同夹爪”，还测了“同机器人 + 不同夹爪”。后者仍然很难，因为控制点偏移会改变视觉几何；Shadow 在这里也明显更稳。

#### 3) 消融信号：关键不是“把机器人涂黑”，而是“叠加目标机器人 mask”
Stack 消融最有说服力：

- 只做 black-out 源机器人：在 UR5e / IIWA 上成功率仅 **0.02 / 0**
- 完整 Shadow：提升到 **0.97 / 0.97**

这说明 Shadow 的效果不只是“去掉机器人 RGB 干扰”，而是**通过目标 embodiment 的几何占位，让策略在训练时就看到未来测试时会出现的机器人形态支持域**。

#### 4) 机理验证：机器人 RGB 细节确实不是必须
附录中，作者证明在源机器人上把机器人 RGB 替换成 mask，本身并不会显著降低表现。  
这条证据非常关键，因为它支撑了 Shadow 的核心前提：**任务所需信息主要在交互几何，不在机器人表面纹理。**

#### 5) 鲁棒性边界：对标定误差敏感
附录还测了相机外参噪声。随着平移/旋转噪声增大，成功率明显下降。  
这说明 Shadow 的强项来自几何对齐，但它也因此**对几何对齐误差敏感**。

### 1-2 个最该记住的指标
- **实机**：相对 Mirage，平均成功率提升 **>2×**
- **关键消融**：Stack 上 UR5e/IIWA 从 **0.02/0 → 0.97/0.97**

### 局限性
- **Fails when**: 相机标定误差较大、关节角/运动学不准、机器人与物体存在严重遮挡、任务需要极高精度抓取、或不同 embodiment 必须采用不同物理避碰策略时，mask 对齐误差会直接破坏策略输入。
- **Assumes**: 已知源/目标机器人的 CAD/URDF、准确 proprioception、可用运动学与渲染、固定相机标定；源与目标场景中的物体分布相对相机基本一致；任务可用相似的笛卡尔末端策略完成；每个新目标 embodiment 通常需要重新训练一份策略。
- **Not designed for**: 新场景泛化、背景/光照大变化、从平行夹爪迁移到灵巧手、以及需要显式利用机器人外观纹理信息的任务。

### 复用价值
这篇工作最可复用的不是某个网络细节，而是三个操作：

1. **末端位姿对齐渲染**：把目标 embodiment 放到与源相同的任务语义位置；
2. **训练/测试对称式观测编辑**：强行把 train/test 拉回同分布；
3. **把外观不变性前移到数据层**：不必依赖更大模型或更多 embodiment 数据。

因此，它很适合作为一个**前端观测规范化模块**，接到其他 imitation / RL policy 上；也能和场景泛化类方法正交组合。

## Local PDF reference

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_SHADOW_Leveraging_Segmentation_Masks_for_Cross_Embodiment_Policy_Transfer.pdf]]