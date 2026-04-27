---
title: "Slot-Level Robotic Placement via Visual Imitation from Single Human Video"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/robot-imitation-learning
  - task/object-placement
  - modular-pipeline
  - image-diff-prompting
  - 3d-keypoint-matching
  - dataset/SLeRP-Benchmark
  - opensource/no
core_operator: "通过手物跟踪、差分提示的 Slot-Net 槽位分割、跨视角重识别与 RGB-D 几何配准，把单段人类演示视频转成机器人对所有相似空槽的 6-DoF 放置目标。"
primary_logic: |
  单段人类 RGB-D 放置视频 + 机器人腕部 RGB-D 图像
  → 跟踪被操纵物体并用起止帧差分提示的 Slot-Net 找出演示槽位
  → 在机器人视角重识别该物体与所有相似空槽并建立 2D/3D 对应
  → 链式求解每个槽位的 6-DoF 放置变换并交给下游抓取与规划执行
claims:
  - "SLeRP 在作者构建的 288 段真实 RGB-D 视频基准上，在不同视角、背景和槽位占用三种设置下均显著优于 ORION、ORION++、CLIPort++ 与 VideoCap+FMs；例如不同视角设置中对象/槽位 IoU 达到 73.85/54.37 [evidence: comparison]"
  - "Slot-Net 相比图像差分、变化检测、GPT4o+SAM 和使用终帧提示的变体，在真实 seen/unseen 任务上的槽位 IoU 分别达到 61.59/54.26，为最优结果 [evidence: comparison]"
  - "系统已在 Franka 机器人上完成真实执行示例，表明其输出的 6-DoF 放置变换可接入现有抓取与规划栈 [evidence: case-study]"
related_work_position:
  extends: "ORION (Zhu et al. 2024)"
  competes_with: "ORION (Zhu et al. 2024); CLIPort (Shridhar et al. 2021)"
  complementary_to: "Contact-GraspNet (Sundermeyer et al. 2021); CuRobo (Sundaralingam et al. 2023)"
evidence_strength: moderate
pdf_ref: "paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Slot_Level_Robotic_Placement_via_Visual_Imitation_from_Single_Human_Video.pdf"
category: Embodied_AI
---

# Slot-Level Robotic Placement via Visual Imitation from Single Human Video

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2504.01959), [Project](https://ddshan.github.io/slerp)
> - **Summary**: 这篇工作把“看一段人类演示就让机器人学会精细放入某个槽位”分解为槽位理解、跨视角重识别和 3D 位姿传递三步，使机器人无需额外任务级示教视频也能完成 slot-level placement。
> - **Key Performance**: 在 288 段真实 RGB-D 视频基准上，不同视角设置达到 **73.85 Obj IoU / 54.37 Slot IoU**；Slot-Net 在真实未见任务上达到 **54.26 Slot IoU**。

> [!info] **Agent Summary**
> - **task_path**: 单段人类 RGB-D 放置视频 + 机器人腕部 RGB-D 图像 -> 目标物体掩码 + 相似空槽掩码 + 每个槽位的 6-DoF 放置变换
> - **bottleneck**: 单次人类演示里“放到哪个细粒度槽位”难以被对象级方法或文本名称可靠抽取，而且还要跨视角迁移到机器人场景
> - **mechanism_delta**: 只对“槽位检测”新增一个差分提示的 Slot-Net，其余步骤复用基础模型做跟踪、重识别、匹配和 3D 几何求解
> - **evidence_signal**: 三种 domain shift 下全面领先基线，且去掉 Slot-Net / SAM2 / MASt3R 都会明显退化
> - **reusable_ops**: [差分提示槽位分割, 人到机器人位姿链式组合]
> - **failure_modes**: [动态相机或放置器具明显移动, 双手操作或强遮挡导致手物跟踪与槽位解析不稳定]
> - **open_questions**: [能否在仅 RGB 条件下恢复可执行 6-DoF 放置, 能否把多模块流水线做成闭环纠错执行]

## Part I：问题与挑战

这篇论文解决的不是普通 pick-and-place，而是**slot-level robotic placement**：给机器人一段人类 RGB-D 演示视频，再给它一张新的机器人视角 RGB-D 图像，机器人要回答三个问题：

1. 人拿的是哪个物体？
2. 人最终放进的是哪个“精细槽位”？
3. 在机器人当前视角里，对应的空槽在哪里，物体该以什么 6-DoF 位姿放进去？

### 真正的难点是什么？

真正瓶颈不是“学会抓取动作”，而是**从一次人类演示中提取可执行的视觉目标**。  
已有方法大多卡在两处：

- **粒度不够**：很多方法只做到“放到某个容器/物体上”，但这里要求区分紧致、相邻、外观相似的多个槽位。
- **迁移链条太长**：必须把人类视角中的目标槽位，迁移到机器人新视角和新布局中，还要输出几何上可执行的 6-DoF 变换，而不是一句语义描述。

### 输入 / 输出接口

- **输入**：
  - 一段人类 RGB-D 放置视频
  - 一张机器人腕部 RGB-D 图像
- **输出**：
  - 机器人视角下待抓物体 mask
  - 所有与演示槽位同类的空槽 mask
  - 每个候选槽位对应的 6-DoF 放置变换

### 为什么现在值得做？

因为以前要么依赖大量机器人示教，要么要训练跨 embodiment 的策略；而现在，SAM2、DINOv2、MASt3R 这类视觉基础模型已经足够强，可以把问题改写成：

**“从人类视频中恢复一个视觉-几何目标，再交给下游机器人规划执行。”**

这比直接从人视频学机器人控制策略更现实，也更省数据。

### 边界条件

论文默认的任务边界比较明确：

- 重复性的放置任务，如装盒、摆放、整理
- RGB-D 可用，且相机内参已知
- 人类视频通常是相对稳定的观察视角
- 单手交互为主
- 放置对象本身不大幅移动

---

## Part II：方法与洞察

### 设计哲学

SLeRP 的核心思想很干脆：

> **不去端到端学“机器人怎么动”，而是先恢复“人到底把东西放到了哪里”，再把这个目标几何化。**

因此作者只为一个基础模型薄弱的环节专门训练了新模块——**Slot-Net**；其余部分尽量复用现成强模型。

### 方法主链条

#### 1. 解析人类视频：先找被拿物体，再找目标槽位

- 用手-物检测器找出与手接触的物体
- 用 MASA + SAM2 把这个物体在整段视频中稳定跟踪出来
- 得到起始帧中的 pick object mask

然后是论文最关键的新点：

- 用 **起始帧 + 起止帧灰度差分图** 输入 **Slot-Net**
- 输出起始帧中的 placement slot mask

这个设计很重要：  
槽位本身未必像普通“物体”那样容易直接分割，但**“某个位置在起止帧之间发生了被占用变化”**是非常强的视觉信号。作者把“槽位检测”从通用语义分割，改成了**变化引导的定向分割**。

#### 2. 为什么需要 Slot-Net，而不是直接用 VLM / 差分 / 变化检测？

因为这些替代方案都不稳定：

- 纯差分：只能看到变化，难以恢复精确槽边界
- 通用变化检测：不能保证变化区域就是目标槽位
- VLM 命名 + 检测：对“烤面包机第几个槽”“蛋托空位”这类细粒度空间语义非常脆弱

作者因此对 SAM 做了 slot-specific finetuning。  
为了训练它，又设计了一个半自动数据管线：

- 从 object-centric 图像中移除一个物体
- 人工标注暴露出来的槽位 mask
- 再通过 outpainting 扩展到多背景
- 最终从 2,138 张图扩展到约 156K 训练图像

这相当于用生成式编辑，把“难采集的槽位数据”变成“可以合成的数据”。

#### 3. 跨视角重识别：把人类槽位对到机器人视角

拿到人类起始帧里的 object mask 和 slot mask 后：

- 用 SAM2 在两帧“短视频” `{H1, R}` 上做跨视角重识别，得到机器人视角中的物体和一个最佳槽位
- 如果存在多个相似空槽，再用 SAM 提 proposals，DINOv2 做相似性检索，补齐所有同类空槽

这样，系统就从“人放的是哪个槽”推广到了“机器人当前能放的所有同类空槽”。

#### 4. 从视觉目标到可执行目标：3D 对应 + 刚体链式组合

最后一步不是预测动作 token，而是恢复几何关系：

- 用 MASt3R 在局部 patch 上做 2D keypoint matching
- 用 RGB-D 和相机参数把对应点 lifting 到 3D
- 用 RANSAC + Procrustes 求三段刚体变换
- 再把这三段变换链起来，得到机器人把当前物体放进目标槽位所需的最终 6-DoF 变换

这一步的价值在于：  
**它把“看懂人类视频”变成了“输出机器人可执行目标位姿”。**

### 核心直觉

**作者真正调的因果旋钮**是：

- **从学动作，改为学视觉目标**
- **从语义命名，改为差分提示的槽位锚定**
- **从 2D 对齐，改为 RGB-D 支撑的 3D 几何一致性**

对应的能力变化是：

- 以前：知道“把杯子放到杯垫上”
- 现在：能知道“把这个物体放到这个精确槽里，并给出机器人执行位姿”

更具体地说：

1. **差分提示**改变了信息瓶颈  
   原本模型要从整幅图里猜“哪个洞/哪个槽是目标”；现在它只需围绕“放置前后哪里发生了结构变化”去恢复槽位。

2. **模块化几何链**改变了迁移约束  
   不再试图学习跨人-机 embodiment 的动作映射，而是转成跨视角刚体对应，几何上更稳。

3. **只训练缺失模块**改变了数据需求  
   不需要额外的大规模人-机配对视频，训练压力集中到 Slot-Net 上。

### 策略权衡

| 设计选择 | 解决的瓶颈 | 带来的能力 | 代价 |
|---|---|---|---|
| 模块化“视觉目标恢复”而非端到端策略学习 | 降低任务专属机器人示教需求 | 单视频迁移到新任务/新视角 | 多模块串联，误差会累积 |
| 差分提示 Slot-Net 而非纯 VLM / 变化检测 | 细粒度槽位难以被语义名词稳定描述 | 更精确地锁定 tight-fitting 槽位 | 需要额外合成数据与微调 |
| RGB-D + 3D 配准而非仅 2D policy | 2D 方法难应对姿态和视角变化 | 输出可执行 6-DoF 放置目标 | 依赖深度、标定和稳定匹配 |

---

## Part III：证据与局限

### 关键证据信号

#### 1. 比较实验信号：不是“小幅更好”，而是明显拉开

作者构建了 288 段真实 RGB-D 视频、9 类任务、3 种测试设置（视角变化 / 背景变化 / 槽位占用变化）。  
在这个基准上，SLeRP 对 ORION、ORION++、CLIPort++、VideoCap+FMs 都有明显优势。

最直接的信号是不同视角设置下：

- **Object IoU: 73.85**
- **Slot IoU: 54.37**

相比之下，ORION++ 的 slot IoU 只有 **8.89**，说明这不是“把已有对象级方法直接套过来”就能解决的问题。

#### 2. 因果消融信号：提升来自关键模块，而不是堆模型

最有说服力的因果证据来自 Table 3：

- 去掉 **Slot-Net** 后，slot IoU 从 **54.37** 掉到 **29.63**
- 去掉 **SAM2** 后，object IoU 从 **73.85** 掉到 **31.05**
- 去掉 **MASt3R** 后，2D 检测不变，但放置精度从 **36.40** 降到 **32.74**

这说明三件事分别成立：

- 槽位检测必须专门建模
- 跨视角 mask 重识别是主干能力
- 几何匹配质量直接影响可执行位姿

#### 3. Slot-Net 专项信号：差分提示确实比“看终帧”更有效

在真实未见任务上：

- **Slot-Net(ours)**: 54.26 IoU
- **Slot-Net(end image)**: 28.37 IoU

这非常关键。它说明作者的核心创新不是“又 finetune 了一个 SAM”，而是**把起止帧差分作为提示**，让模型专注于“被放置行为激活的槽位”。

#### 4. 部署信号：能接机器人栈，但仍是系统级验证

论文在 Franka 上展示了真实执行，如 block into container、strawberry into organizer。  
这证明 SLeRP 输出的不是抽象中间表示，而是能接到 Contact-GraspNet 和规划模块里的目标位姿。

### 局限性

- **Fails when**: 相机显著移动、放置器具本身发生较大位移、双手操作或强遮挡时，手物跟踪、槽位检测和跨视角对应都会更不稳定。
- **Assumes**: 任务有 RGB-D 输入和相机标定；人类演示多为静态视角；单手操作；放置对象变化相对有限；还依赖多种外部视觉基础模型与一个合成数据训练流程。
- **Not designed for**: 长时程操作、闭环纠错控制、无深度的纯 RGB 高精度装配、完全动态场景中的开放式操作规划。

### 复现/扩展依赖

这篇工作虽然避免了大规模机器人示教，但并不“轻”：

- 需要组合多个第三方模块：SAM/SAM2、DINOv2、MASt3R、手物检测器等
- Slot-Net 训练依赖半自动数据管线和人工标注
- 真实执行还依赖抓取生成与运动规划栈
- 代码是否完整开放，从正文无法确认

### Reusable components

可复用的部分很明确：

- **差分提示的槽位分割范式**：适合任何“放置前后结构变化显著”的 fine-grained receptacle/slot 任务
- **跨视角 mask 重识别 + 多槽检索**：适合从演示视角迁移到执行视角
- **视觉对应到 6-DoF 目标的链式组合**：适合把“模仿目标”而非“模仿动作”交给下游机器人执行

## Local PDF reference

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Slot_Level_Robotic_Placement_via_Visual_Imitation_from_Single_Human_Video.pdf]]