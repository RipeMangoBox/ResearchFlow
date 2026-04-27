---
title: "Visual Embodied Brain: Let Multimodal Large Language Models See, Think, and Control in Spaces"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/robot-control
  - task/spatial-reasoning
  - keypoint-based-control
  - multimodal-chain-of-thought
  - closed-loop-control
  - dataset/VeBrain-600k
  - opensource/no
core_operator: "把机器人控制重写为2D视觉空间中的“关键点定位+具身技能识别”文本任务，再用闭环机器人适配器把文本决策转成真实动作。"
primary_logic: |
  图像/视频观测 + 语言指令
  → MLLM在统一对话中完成感知、空间推理，并输出下一步关键点与技能文本
  → 机器人适配器执行点跟踪、2D到3D变换、技能调用与失败接管
  → 真实机器人完成导航、交互、搬运与操作任务
claims:
  - "VeBrain在基本保持Qwen2.5-VL通用多模态能力的同时，将MMVet从67.1提升到72.7，13个多模态基准的归一化平均分从76.9提升到77.1 [evidence: comparison]"
  - "在分阶段消融中，从Qwen2.5-VL基线到加入robotic adapter与混合数据后，选定任务平均分从43.2提升到78.0，Complex Find成功率从0%提升到80% [evidence: ablation]"
  - "在真实机器人上，VeBrain把四足机器人7项任务总体成功率做到86.4%，把机械臂7项任务总体成功率做到74.3%，显著高于Qwen2.5-VL+adapter与π0等基线 [evidence: comparison]"
related_work_position:
  extends: "Qwen2.5-VL (Bai et al. 2025)"
  competes_with: "ChatVLA (Zhou et al. 2025); RoboBrain (Ji et al. 2025)"
  complementary_to: "N/A"
evidence_strength: strong
pdf_ref: "paperPDFs/Other_Multimodal_Large_Language_Model_MLLM_based_related_Embodied_Learning_2025/arXiv_2025/2025_Visual_Embodied_Brain_Let_Multimodal_Large_Language_Models_See_Think_and_Control_in_Spaces.pdf"
category: Embodied_AI
---

# Visual Embodied Brain: Let Multimodal Large Language Models See, Think, and Control in Spaces

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2506.00123)
> - **Summary**: 论文把机器人控制改写成MLLM更擅长的2D文本化决策问题，再用执行端适配器补齐3D动作落地，从而把“看懂-推理-控制”统一进同一套模型接口。
> - **Key Performance**: MMVet 72.7（较Qwen2.5-VL +5.6）；四足机器人7任务总体成功率 86.4%，机械臂7任务总体成功率 74.3%

> [!info] **Agent Summary**
> - **task_path**: RGB/RGB-D图像或视频 + 语言指令 -> 文本化关键点与技能决策 -> 真实机器人动作执行与任务完成
> - **bottleneck**: MLLM的2D跨模态文本建模目标，与VLA的低层动作策略学习目标不一致，联合训练容易出现能力冲突与遗忘
> - **mechanism_delta**: 把控制压缩成“去哪里（关键点）+做什么（技能）”两类2D文本输出，再由闭环adapter负责三维执行、跟踪和接管
> - **evidence_signal**: 跨13个多模态、5个空间推理和14个真实机器人任务的统一评测，同时有框架/数据消融支撑
> - **reusable_ops**: [control-as-2d-text-task, keypoint-skill-decoupling]
> - **failure_modes**: [目标长时间离开视野导致跟踪失效或频繁接管, 超出预置policy pool的精细连续操作无法直接表达]
> - **open_questions**: [离散关键点+技能表示能否扩展到高自由度灵巧操作, 统一收益有多大来自任务重写而非大量人工和闭源CoT数据]

## Part I：问题与挑战

这篇论文抓得很准的点，不是“机器人数据还不够多”，而是**监督空间不统一**。

### 1. 真问题是什么
现有两条路各有硬伤：

- **MLLM路线**：擅长图像/视频理解、问答、OCR、常识推理，但直接拿文本去控机器人，控制精度和鲁棒性不够。
- **VLA路线**：擅长把观测映射成动作策略，但一旦大量偏向机器人动作学习，往往会损伤原本的通用多模态能力。

作者认为根因在于：

- MLLM主要在学：**2D视觉内容 ↔ 文本 token**
- 机器人控制主要在学：**观测 ↔ 物理动作/运动策略**

这两种目标的输出空间、误差形态、时序要求都不同，硬拼在一个模型里，容易出现任务冲突和知识遗忘。

### 2. 输入/输出接口
VeBrain试图把所有任务都改写到同一种接口里：

- **输入**：图像/视频观测 + 文本指令
- **输出**：
  - 对多模态理解/空间推理：普通文本答案
  - 对机器人控制：下一步**关键点** + **具身技能名**

也就是说，它不让MLLM直接学低层连续动作，而是先让MLLM给出“下一步应该朝哪里去、接下来该做什么”。

### 3. 为什么现在值得做
因为MLLM的感知和语言推理已经足够强，下一步自然是走向具身控制；但如果继续沿“直接把动作 token 塞进MLLM”的路线，模型很容易变成一个只会控机器人、但通用能力明显退化的系统。VeBrain要解决的是这个当下非常关键的折中点：**能不能不丢掉MLLM的通用脑子，同时获得可部署的控制能力。**

### 4. 边界条件
这套方案适合的是：

- 有视觉输入的导航、跟随、交互、抓取、开抽屉、搬运
- 可以被拆成“目标位置 + 技能调用”的任务
- 有RGB-D、相机标定、技能库支持的机器人平台

它并不直接覆盖：

- 纯连续力控
- 开放式新技能发现
- 无深度/无标定条件下的直接3D执行

---

## Part II：方法与洞察

VeBrain由两部分组成：

1. **MLLM本体**：负责看、想、做决策  
2. **Robotic Adapter**：负责把文本化控制信号变成真实机器人动作

核心不是换了一个更大的模型，而是**换了控制问题的表述方式**。

### 1. 统一任务建模：把控制也写成MLLM会做的题
作者把机器人控制拆成两个子任务：

- **关键点检测**：机器人下一步应该移动到/交互到图像中的哪里
- **具身技能识别**：移动后该执行什么技能，如 Turn Right、Dump、Grasp、Pull

并且控制过程不是一拍脑袋直接出动作，而是放进一个CoT式过程里：

- 环境感知
- 目标可达性分析
- 全局规划
- 当前最优移动
- 下一步关键点与技能

这样一来，机器人控制被重新表述成了一个MLLM熟悉的“看图说理后输出文本”的问题。

### 2. Robotic Adapter：把2D文本决策补成可执行控制
MLLM输出只是“关键点+技能”，离真实执行还差几步。作者用adapter补上：

- **Point Tracker**：机器人移动后视角变化，原关键点会漂移；用LocoTrack实时追踪点位
- **Movement Controller**：结合RGB-D深度与标定，把2D点转成3D位置，再变成底层移动控制
- **Skill Executor**：把“Shake / Dump / Grasp / Pull”等技能映射到预训练policy pool
- **Dynamic Takeover**：目标丢失、子任务完成或执行异常时，把控制权切回MLLM重新决策

这一步非常关键：它把MLLM从“必须直接会机器人学”的负担里解放出来，只要求它做高层可解释决策。

### 3. 数据侧：VeBrain-600k不是单一机器人数据，而是三种能力混训
VeBrain-600k由三部分组成：

- 200k 多模态理解数据
- 312k 视觉-空间推理数据
- 88k 机器人控制数据

关键不是简单拼盘，而是作者用**多模态CoT**把感知、推理、控制揉进同一段对话格式里。  
其中机器人数据由人工采集和标注，空间推理由ScanNet/GPT4Scene等来源扩充，CoT由GPT-4o和Gemini-2.0辅助生成并交叉验证。

### 核心直觉

**真正的因果旋钮**是：  
把“让模型直接输出低层动作”改成“让模型输出视觉指称的关键点与语义化技能”。

这带来了三个变化：

1. **输出分布变了**  
   从机器人专属的动作空间，变成MLLM本就擅长的离散文本/指称空间。

2. **信息瓶颈变了**  
   模型不必在参数里硬记整套动力学与执行细节，而只需要决定：
   - 去哪里
   - 做什么

3. **能力边界变了**  
   通用多模态理解不再必须为动作建模让路；3D落地和时序纠错交给adapter，于是整体系统同时得到：
   - 更好的通用能力保留
   - 更强的空间/控制组合能力
   - 更高的可解释性

换句话说，VeBrain不是让MLLM“亲自变成低层控制器”，而是让MLLM成为**高层可解释具身决策器**。

### 战略权衡表

| 设计选择 | 解决的瓶颈 | 带来的能力 | 代价/风险 |
|---|---|---|---|
| 把控制写成“关键点+技能”文本任务 | MLLM目标与动作策略目标不一致 | 统一接口，减少任务冲突，保住通用多模态能力 | 控制表达粒度有限，难覆盖精细连续操作 |
| 用robotic adapter补2D到3D执行 | 文本输出不能直接落地真实世界 | 可部署、可闭环、可纠错 | 依赖深度相机、标定、点跟踪和低层控制器 |
| 用多能力混合CoT数据训练 | 单任务数据难学组合能力 | 更擅长长程、多步、复合任务 | 高人工成本，且依赖闭源模型生成/审核CoT |
| 依赖预训练policy pool执行技能 | MLLM不适合承担所有低层运动学 | 易迁移到多平台，控制更稳定 | 技能上限受policy pool覆盖范围约束 |

---

## Part III：证据与局限

### 关键证据

#### 证据1：框架层面的trade-off被明显推高
最能说明问题的不是单个benchmark，而是表2这种“统一权衡”对比：

- 文本控制MLLM框架：通用理解还行，但机器人控制弱
- 直接动作策略VLA框架：控制稍强，但通用多模态能力掉很多
- **VeBrain**：在MMVet、ScanQA、VSI-Bench和机器人任务上同时更均衡，平均分 **78.0**，明显高于文本MLLM的 **46.5** 和VLA的 **48.2**

这直接支持作者的核心判断：  
**瓶颈确实不是模型容量，而是任务表述与监督空间冲突。**

#### 证据2：多模态能力基本保住了，但提升主要不是靠“刷通用榜”
VeBrain相对Qwen2.5-VL：

- MMVet：**67.1 → 72.7**
- 13个多模态基准归一化平均：**76.9 → 77.1**

这个信号很重要：  
VeBrain的价值不是把通用MLLM benchmark大幅刷爆，而是**在不明显牺牲通用能力的前提下，把空间推理和控制拉起来**。

也要注意，提升并非无代价、也非全面领先：

- MMMU：56.6，低于Qwen2.5-VL的58.6
- InfoVQA：80.5，也低于Qwen2.5-VL的82.6

所以更准确的结论是：**它实现了“保住大盘 + 打开具身能力”**，而不是“所有通用任务全面碾压”。

#### 证据3：最大的能力跳跃发生在空间推理和真实机器人控制
这里才是VeBrain最有说服力的地方。

- **3D空间推理**
  - ScanQA CIDEr：**101.5**，相比Qwen2.5-VL的 **62.7** 提升很大
  - 在ScanRefer / Multi3DRef等3D任务上，也明显强于直接拿2D MLLM硬迁移

- **四足机器人**
  - 7项任务总体成功率：**86.4%**
  - Qwen2.5-VL + adapter：**42.1%**
  - VLA基线：**32.1%**

- **机械臂**
  - 7项任务总体成功率：**74.3%**
  - π0：**31.4%**
  - OpenVLA：**11.4%**

尤其在复杂、多步、组合任务上，差距最大。这说明VeBrain真正拉开的不是“单步反应”，而是**感知-推理-控制组合链条**。

#### 证据4：消融证明“适配器 + 混合数据”缺一不可
表1的分阶段结果很有解释力：

- 只有Qwen2.5-VL时，Complex Find成功率是 **0%**
- 加adapter后，机器人任务开始能做，但理解/空间能力并没提升
- 加control data后，控制变强，但空间推理仍不够稳
- 再加spatial reasoning data和multimodal understanding data后，整体平均分一路到 **78.0**

这说明VeBrain不是靠某个单点技巧取巧，而是**任务重写、执行适配、数据配方**三者共同起作用。

### 局限性

- **Fails when**:
  - 目标长时间离开视野、被严重遮挡，导致关键点跟踪失效
  - 任务需要比现有policy pool更细粒度的连续控制、接触力控制或灵巧手操作
  - 场景需要更强的长时空间记忆/路线规划时，VeBrain并非所有子项都最优（如VSI-Bench中并未全面超过更强闭源模型）

- **Assumes**:
  - 依赖RGB-D相机、相机标定和2D到3D转换
  - 依赖预训练低层技能库（walk / turn / dump / grasp / pull 等）
  - 依赖较高数据成本：80+小时机器人数据采集、多人关键点与动作标注
  - 依赖闭源模型 GPT-4o 与 Gemini-2.0 生成/验证CoT
  - 训练需要 **32×A100**，部署时MLLM在云端 **0.5Hz** 推理，点跟踪在Jetson上 **15Hz**
  - 数据中存在重复扩增（附录中空间/机器人数据有重复采样），有效多样性可能低于表面规模

- **Not designed for**:
  - 端到端连续动作/力矩控制
  - 无技能库前提下的开放式新技能发现
  - 纯机载、低延迟、完全离线的实时自主控制
  - 摆脱深度与几何适配器、单靠MLLM内部表征直接完成真实3D执行

### 可复用组件

这篇论文最值得复用的，不一定是整套系统，而是这几个“操作子”：

1. **control-as-text**：把控制重写为MLLM原生可处理的离散文本任务  
2. **keypoint + skill decomposition**：把“位置决策”和“动作语义”分开  
3. **closed-loop adapter**：把2D语义决策与3D执行解耦  
4. **multimodal CoT mixing**：把感知、推理、控制放进同一对话样本中训练  
5. **dynamic takeover trigger**：当目标丢失/子任务完成时，显式切回高层决策

## Local PDF reference

![[paperPDFs/Other_Multimodal_Large_Language_Model_MLLM_based_related_Embodied_Learning_2025/arXiv_2025/2025_Visual_Embodied_Brain_Let_Multimodal_Large_Language_Models_See_Think_and_Control_in_Spaces.pdf]]