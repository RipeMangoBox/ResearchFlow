---
title: "PhysVLM: Enabling Visual Language Models to Understand Robotic Physical Reachability"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/visual-question-answering
  - task/robot-task-planning
  - reachability-map
  - dual-branch-architecture
  - voxelization
  - dataset/Phys100K
  - dataset/EQA-phys
  - dataset/RoboVQA
  - dataset/OpenEQA
  - opensource/no
core_operator: 将机器人运动学与深度观测压缩成机型无关的 S-P Map，并通过独立约束编码器与视觉分支融合，让 VLM 显式推理“机器人是否真的够得到”。
primary_logic: |
  RGB图像 + 文本指令 + 机器人DH参数/关节范围/深度 → 离线构建可达工作空间并在线投影成机型无关的 S-P Map，经独立约束分支与视觉 token 融合 → 输出符合物理可达性的问答与任务步骤
claims:
  - "PhysVLM-3B 在 EQA-phys 上获得 71.0 的平均分，相比 GPT-4o 的 57.0 提升 14.0 个点，并在零样本真实机器人 UR3/XArm6 上分别达到 64.1/63.0 [evidence: comparison]"
  - "将 S-P Map 替换为 Depth Map 或完全移除后，EQA-phys 真实/模拟成绩分别从 63.5/74.8 降至 58.1/62.4 或 54.2/58.8，说明增益主要来自显式可达性表示而非深度本身 [evidence: ablation]"
  - "PhysVLM-3B 在 RoboVQA-val 上达到 BLEU-4 43.5，超过 RoboMamba 的 36.3；在 OpenEQA 总分 57.4，优于 SpatialVLM、SpatialBot 和 GPT-4V [evidence: comparison]"
related_work_position:
  extends: "SpatialVLM (Chen et al. 2024)"
  competes_with: "SpatialVLM; SpatialBot"
  complementary_to: "SayCan (Ahn et al. 2022); VoxPoser (Huang et al. 2023)"
evidence_strength: strong
pdf_ref: paperPDFs/Other_Multimodal_Large_Language_Model_MLLM_based_related_Embodied_Learning_2025/arXiv_2025/2025_PhysVLM_Enabling_Visual_Language_Models_to_Understand_Robotic_Physical_Reachability.pdf
category: Embodied_AI
---

# PhysVLM: Enabling Visual Language Models to Understand Robotic Physical Reachability

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2503.08481)
> - **Summary**: 论文把不同机器人的运动学约束先变成一张统一的可达性图 S-P Map，再送入 VLM，使模型在回答 embodied QA 和任务规划时不再只“看见场景”，还能判断“当前机器人够不够得到”。
> - **Key Performance**: EQA-phys 71.0（GPT-4o 为 57.0，+14.0）；RoboVQA-val BLEU-4 43.5（RoboMamba 为 36.3）

> [!info] **Agent Summary**
> - **task_path**: RGB图像 + S-P Map + 指令 / embodied QA与高层任务规划 -> 可达性感知的文本答案与步骤
> - **bottleneck**: 通用 VLM 缺少跨机器人共享的物理可达性表示，导致它能看懂场景语义，却无法判断动作是否真实可执行
> - **mechanism_delta**: 将机器人参数与深度先转换为机型无关的二维可达域图，再用独立约束编码器与视觉分支融合
> - **evidence_signal**: EQA-phys 相比 GPT-4o 提升 14.0pt，且移除 S-P Map 后真实/模拟性能分别下降 9.3/16.0pt
> - **reusable_ops**: [S-P Map可达域投影, 独立约束编码分支]
> - **failure_modes**: [真实机器人域移位导致性能下降, 深度或标定误差会扭曲可达域]
> - **open_questions**: [如何扩展到动态障碍与碰撞约束, 如何从文本规划闭环过渡到动作执行]

## Part I：问题与挑战

这篇论文要解决的，不是“VLM 看不懂图像”，而是 **VLM 没有机器人身体边界感**。

现有 VLM 在场景理解上很强，但到了机器人任务里，经常默认“看见物体 = 可以抓到物体”。这会造成一种很典型的 embodied failure：  
- 语义上回答没错，  
- 物理上根本做不到。  

例如它知道碗在桌上、知道目标是把罐子放进碗里，但不知道当前机械臂并不能直接够到碗，于是给出不可执行的回答或计划。

### 真正瓶颈是什么？

作者把瓶颈拆成两层：

1. **表示瓶颈**  
   不同机器人有不同连杆长度、关节范围、DH 参数、相机外参。  
   如果直接让 VLM 从这些异构参数里学“工作空间”，学习目标太碎、跨机器人泛化太难。

2. **架构瓶颈**  
   就算你提供了 reachability 信息，若只是粗暴塞进原有视觉分支，也可能破坏模型原本的通用视觉语言能力。  
   换句话说，问题不只是“加信息”，而是“怎么加而不把原能力搞坏”。

### 为什么现在值得解决？

因为 VLM 已经不再只是做离线问答，而是在：
- embodied QA
- 高层任务规划
- 工业/辅助机器人决策

这些场景里被真正拿来“指导动作”。  
这时错误不再只是答错题，而可能意味着：
- 抓取失败
- 无效路径
- 设备碰撞风险
- 任务执行不稳定

### 输入/输出接口与边界条件

**输入**
- 机器人第一视角 RGB 图像
- 文本指令/问题
- 机器人参数与深度信息（实际在送入模型前先转成 S-P Map）

**输出**
- 面向可达性的 QA 回答
- 或高层任务步骤文本

**边界条件**
- 主要针对机械臂场景
- 主要是单视角 RGB-D 感知
- 假设场景相对静态
- 目标是“能否够到/是否需要先移动靠近”的推理
- 不直接做低层控制、轨迹优化、力控制或碰撞自由运动规划

---

## Part II：方法与洞察

### 方法主线

PhysVLM 的核心做法可以概括成一句话：

> **先把“机器人参数差异”压缩成统一的空间约束图，再让 VLM 读这张图。**

#### 1. S-P Map：把机器人身体约束变成“可读图像”

作者提出 **Space-Physical Reachability Map (S-P Map)**。

它的生成逻辑是：

- **离线阶段**  
  根据每个机器人机械臂的 DH 参数和关节范围，采样并计算其可达工作空间；
  再把这个空间离散成体素网格，得到“哪些 3D 位置理论上可达”。

- **在线阶段**  
  用机器人第一视角深度图恢复场景点云；
  把点云变换到机器人坐标系；
  检查每个点是否落在预计算的可达体素空间内；
  再把这些“可达点”投影回图像平面，生成一张 S-P Map。

最终效果是：  
模型看到的不是一堆难学的机械臂参数，而是一张直接表达 **哪里可达、哪里不可达** 的约束图。

这一步非常关键，因为它把问题从：

- “从多种机器人参数中学运动学”

转成了：

- “从统一的空间图上读约束”

#### 2. 双分支架构：视觉和约束分开编码

PhysVLM 不是把 S-P Map 当普通图片混进去，而是专门做了双分支：

- **视觉分支**：编码 RGB 图像
- **约束分支**：编码 S-P Map
- 然后把两边 token 融合，送入 Qwen-2.5-Instruct-3B 解码

这里的设计点不在“多一个 encoder”本身，而在于：

> **S-P Map 和 RGB 的统计属性不同，最好不要共享同一个视觉表征空间。**

作者后面的消融也支持这一点：共享特征编码器会同时伤害 reachability 推理和一般视觉推理。

#### 3. 两阶段训练：补物理约束，但不丢通用能力

训练数据并不只来自 reachability 数据，还混合了：
- Phys100K（作者构建）
- LLaVA-Pretrain
- ShareGPT4V
- RoboVQA
- OpenX-Embodiment 等

训练策略是：
- **第一阶段**：先做多模态对齐，只训练投影层
- **第二阶段**：全参数训练，让模型同时学会一般视觉问答和可达性约束

这个训练设计的意图很明确：  
不是把模型做成一个只会判断“够不够到”的专用模块，而是让它在保留通用 VLM 能力的同时，新增一层 embodied physical reasoning。

### 核心直觉

#### 改变了什么？
作者把输入表示从：
- “图像 + 隐式机器人参数差异”

改成：
- “图像 + 显式、机型无关的可达性图”

并且把约束模态从共享表征改成独立分支处理。

#### 改变了哪个瓶颈？
这实际上改变了两个瓶颈：

1. **分布瓶颈**  
   原本不同机器人之间的差异体现在高维运动学参数里，分布高度异构；  
   现在这些差异被折叠成统一的“可达/不可达区域”，跨机器人泛化更容易。

2. **信息瓶颈**  
   原来模型必须从视觉语义里“猜”物理能力；  
   现在 physical reachability 变成显式可见的信息，推理链条更短、更直接。

#### 能力发生了什么变化？
能力提升不是“所有视觉任务都更强”，而是：

- 在 **reachability 是主瓶颈** 的任务上明显跃升；
- 在一般 embodied QA 上尽量不退化；
- 对 unseen robots 保持一定零样本泛化能力。

#### 为什么这个设计有效？
因为 **Depth Map 只告诉你“离相机多远”，不告诉你“对机器人是否可达”**。  
可达性取决于：
- 机械臂长度
- 关节极限
- 机器人基座坐标
- 相机与机器人标定关系

S-P Map 正是把这些“机器人身体条件”前置编码进输入里。  
模型无需再学复杂的运动学映射，只需要判断“目标是否落在可达域中”。

### 战略权衡

| 设计选择 | 解决的瓶颈 | 代价/风险 | 实验信号 |
| --- | --- | --- | --- |
| S-P Map 统一表示 | 把跨机器人异构运动学压缩为统一可达域 | 依赖准确 DH/关节范围/标定；丢失抓取姿态与碰撞细节 | 替换成 Depth Map 或移除后性能显著下降 |
| 独立约束编码器 | 避免 S-P Map 与 RGB 共享表征造成模态污染 | 额外参数与融合复杂度 | 独立分支优于共享分支（EQA-phys 71.0 vs 68.2） |
| 两阶段多源训练 | 在新增物理约束的同时保留通用 VLM 能力 | 训练流程更复杂，依赖伪标注质量 | RoboVQA-val 最优，OpenEQA 保持竞争力 |

---

## Part III：证据与局限

### 关键实验信号

#### 1. 能力跳跃发生在“物理可达性”真正限制任务的地方
最强信号来自 EQA-phys：
- **PhysVLM-3B：71.0**
- **GPT-4o：57.0**

而且在真实机器人零样本测试上：
- UR3：64.1
- XArm6：63.0

这说明模型不是只在训练过的机器人上记忆模板，而是学到了一种可迁移的 reachability 表达。

更有意思的是任务规划实验：
- **当所有物体都在范围内时**，PhysVLM 并不是最强（69.2，低于 GPT-4o 的 75.9）
- **当只有部分物体在范围内时**，PhysVLM 变成最强（48.4，高于 GPT-4o 的 35.8）

这很有说服力，因为它把收益定位得很清楚：  
**它补的不是通用语言能力，而是“可执行性判断”这块短板。**

#### 2. 因果旋钮确实是 S-P Map，而不是“又加了一个图”
消融最重要：

- 用 **S-P Map**：真实/模拟 63.5 / 74.8
- 换成 **Depth Map**：58.1 / 62.4
- **不输入约束图**：54.2 / 58.8

这说明：
- 深度本身不够
- 真正有效的是“深度 + 机器人运动学”被编码成的 reachability 表示

此外，**GPT-4o-mini + S-P Map** 在 EQA-phys 总体也从 52.8 提升到 59.8。  
这说明 S-P Map 不是只对 PhysVLM 有用，而是一个可插拔的表示层。

#### 3. 新增物理约束并未明显牺牲通用 embodied QA
- RoboVQA-val：BLEU-4 达到 **43.5**，优于 RoboMamba 的 **36.3**
- OpenEQA 总分：**57.4**，优于 SpatialVLM、SpatialBot 和 GPT-4V，仅低于 GPT-4o

这支持作者第二个核心目标：  
**加入 reachability 能力后，模型并没有明显丢掉一般视觉语言推理能力。**

### 局限性

- **Fails when**: 真实机器人与仿真存在明显域移位、深度缺失严重、相机外参不准、目标被遮挡，或任务需要精细抓取姿态/碰撞规避/动态障碍建模时，S-P Map 可能无法准确对应真实可执行空间。
- **Assumes**: 需要可用的 RGB-D 输入、机器人 DH 参数、关节范围与相机标定；训练数据构建依赖 DepthAnything-v2、GroundingDINO、SAM2 和 GPT-4 生成伪标注；训练资源为 8×A800、48 小时；文中声明发布 benchmark，但提供文本里未给出代码或数据链接。
- **Not designed for**: 低层闭环控制、碰撞自由轨迹生成、6D 抓取姿态优化、力/接触推理、多机械臂协作或 humanoid 全身可达性建模。

### 可复用组件

1. **S-P Map 生成范式**  
   这是本文最可迁移的部分。任何有：
   - 机器人运动学
   - RGB-D 感知
   - 基本标定  
   的系统，都可以尝试先做 reachability projection，再交给任意 VLM。

2. **独立约束编码分支**  
   对于所有“结构化物理约束 + 视觉语言推理”的场景，这种双分支设计都值得复用。  
   它的抽象层级很合适：既不要求把约束硬编码到 decoder，也不会污染原图像表征。

3. **物理约束感知的数据模板**  
   Phys100K 和 EQA-phys 的意义不只是数据量，更在于提供了：
   - reachability-aware QA 模板
   - 多机器人统一评测方式
   - 从问答延伸到规划的测试接口

### 一句话结论

PhysVLM 的关键贡献不是“又做了一个机器人 VLM”，而是把 **机器人身体可达性** 从隐式常识变成了显式输入，从而让模型在真正需要“知道自己够不够得着”的场景里，首次表现出稳定、可迁移的优势。

![[paperPDFs/Other_Multimodal_Large_Language_Model_MLLM_based_related_Embodied_Learning_2025/arXiv_2025/2025_PhysVLM_Enabling_Visual_Language_Models_to_Understand_Robotic_Physical_Reachability.pdf]]