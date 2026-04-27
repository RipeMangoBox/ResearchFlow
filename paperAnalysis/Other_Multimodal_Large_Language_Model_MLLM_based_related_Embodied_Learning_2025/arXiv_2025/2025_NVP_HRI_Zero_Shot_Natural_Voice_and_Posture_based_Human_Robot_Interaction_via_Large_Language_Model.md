---
title: "NVP-HRI: Zero Shot Natural Voice and Posture-based Human-Robot Interaction via Large Language Model"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/human-robot-interaction
  - task/robot-manipulation
  - segment-anything
  - deictic-posture-grounding
  - prompt-constrained-decoding
  - opensource/promised
core_operator: "以SAM+RGBD构建未知物体的3D结构槽位，用指向姿态绑定目标，再用受约束的GPT-4生成并经几何校验的机器人动作序列"
primary_logic: |
  语音命令 + 指向姿态 + 场景RGBD → SAM分割并重投影得到未知物体的3D结构表示，OpenPose/前臂射线选中目标，GPT-4在动作约束与轨迹约束下生成计划并由swept-volume cross-check回馈修正 → 机械臂对未知物体执行零样本、无碰撞的抓取/放置/倾倒操作
claims:
  - "NVP-HRI在三类桌面操控场景中将用户输入交互时长相对gesture/NLP/VLM基线分别降低59.2%/65.2%/64.8% [evidence: comparison]"
  - "加入轨迹cross-check后，系统的无碰撞轨迹成功率从73%提升到94% [evidence: ablation]"
  - "在作者设置的相似物体桌面任务中，NVP-HRI在S2/S3上的交互准确率约为94-96%和91%，高于gesture/NLP/VLM基线 [evidence: comparison]"
related_work_position:
  extends: "Segment Anything (Kirillov et al. 2023)"
  competes_with: "Communicating Human Intent to a Robotic Companion by Multi-type Gesture Sentences (Vanc et al. 2023); Interactive Multimodal Robot Dialog using Pointing Gesture Recognition (Constantin et al. 2022)"
  complementary_to: "ProgPrompt (Singh et al. 2023); VoxPoser (Huang et al. 2023)"
evidence_strength: moderate
pdf_ref: "paperPDFs/Other_Multimodal_Large_Language_Model_MLLM_based_related_Embodied_Learning_2025/arXiv_2025/2025_NVP_HRI_Zero_Shot_Natural_Voice_and_Posture_based_Human_Robot_Interaction_via_Large_Language_Model.pdf"
category: Embodied_AI
---

# NVP-HRI: Zero Shot Natural Voice and Posture-based Human-Robot Interaction via Large Language Model

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2503.09335) · [Code](https://github.com/laiyuzhi/NVP-HRI.git) · [Video](https://youtu.be/EbC7al2wiAc)
> - **Summary**: 这篇工作把“说做什么”和“指的是哪个物体”分配给语音与指向姿态，再用 SAM+RGBD 做未知物体几何建模、用受约束 GPT-4 做动作规划，从而让机器人在不知道物体名字时也能零样本完成自然交互。
> - **Key Performance**: 用户输入交互时长相对 gesture / NLP / VLM 基线降低 59.2% / 65.2% / 64.8%；cross-check 将无碰撞轨迹成功率从 73% 提升到 94%。

> [!info] **Agent Summary**
> - **task_path**: 语音动作/参数 + 指向姿态 + RGBD场景 -> 机械臂动作序列与轨迹执行
> - **bottleneck**: 未知物体无法靠名字或闭集检测器稳定指代，同时LLM自由规划易产生语义幻觉与碰撞风险
> - **mechanism_delta**: 把目标选择从“说出物体名”改为“姿态指向3D对象槽位”，再用受动作/轨迹约束的GPT-4加几何校验闭环生成计划
> - **evidence_signal**: 24人三场景对比显示交互时长显著下降，且cross-check消融把无碰撞成功率从73%提到94%
> - **reusable_ops**: [SAM+depth三维对象槽位化, 受约束LLM规划+扫掠体校验]
> - **failure_modes**: [低光照导致姿态估计退化, 上半身/指向不可见或目标严重遮挡时目标绑定失败]
> - **open_questions**: [能否摆脱闭源GPT-4并本地部署, 在医院/真实养老照护场景中是否仍保持同等效率与鲁棒性]

## Part I：问题与挑战

这篇论文要解决的核心，不是普通的“机器人抓取”，而是更贴近真实服务机器人的问题：

**当用户不知道物体名字、也不想记复杂手势时，如何自然地告诉机器人“对哪个东西做什么”，并且让机器人安全执行？**

### 真正的瓶颈是什么
现有 HRI 系统常卡在三个地方：

1. **目标指代依赖闭集语义**
   - 许多系统默认用户能说出物体名，或视觉端已经见过该类别。
   - 一旦遇到稀有物体、未知物体、同类但不同位置的物体，这种假设就失效。

2. **交互负担高**
   - gesture-only 方案要求用户记住大量手势；
   - language-only 方案需要用户说得非常精确，容易歧义；
   - 对老年人或病患，这两类负担都不现实。

3. **LLM 直接控机器人不可靠**
   - LLM 擅长把高层意图变成步骤，但如果输入是模糊语言、输出又没有结构约束，就容易出现 hallucination；
   - 对机器人而言，这会直接转化成**错误动作**或**碰撞风险**。

### 为什么是现在
这件事现在值得做，是因为两个基础能力成熟了：

- **SAM** 让系统能在不依赖封闭类别词表的前提下，先把“物体轮廓/区域”分出来；
- **GPT-4 级别 LLM** 让系统可以把多模态意图编译成可执行动作序列。

作者的判断是：  
**不要逼用户“命名未知物体”，而是让系统从几何与指向里理解目标。**

### 输入/输出接口
- **输入**
  - 语音：动作命令、确认命令、可选参数命令
  - 姿态：指向哪个物体
  - RGBD 场景：物体分割、深度、空间关系
- **输出**
  - 机器人动作序列
  - 满足基本避障要求的执行轨迹

### 边界条件
这套方法并非完全开放世界，论文默认了这些条件：

- 主要是**桌面/室内服务机械臂**场景；
- 用户上半身需要被 RGBD 相机看见；
- 依赖英文语音输入；
- 物体尺寸需在夹爪可操作范围内；
- 目标选择基于指向姿态和确认命令，不是纯自然对话式开放引用。

---

## Part II：方法与洞察

作者的设计哲学可以概括成一句话：

**语音负责“做什么”，姿态负责“对谁做”，几何校验负责“能否安全做”。**

### 核心直觉

传统方法往往把三件事混在一起处理：

- 目标是谁；
- 要做什么；
- 路径是否安全。

这会导致一个连锁问题：  
**语言越自然，语义越可能模糊；感知越依赖类别，未知物体越难处理；LLM 输出越自由，动作越难约束。**

NVP-HRI 的关键变化是把它们解耦：

1. **把目标 grounding 从“语义命名”改成“几何指向”**
   - 不要求用户说出物体名字；
   - 用 SAM 分割物体区域，再结合深度得到 3D 点云簇；
   - 用前臂延长线指向最近的 3D 物体簇来锁定目标。

2. **把动作表达从“完整句子描述一切”改成“动作/参数分离”**
   - 语音只负责动作意图和参数，如 pick / pour / 90 degrees；
   - 目标物体不靠语言消歧，而靠姿态选择。

3. **把 LLM 从“自由生成器”改成“受约束的规划器”**
   - 限制可用动作；
   - 限制轨迹表达格式与坐标规则；
   - 用示例任务固定输出样式；
   - 再用几何 cross-check 把不安全轨迹打回重生成。

### 这为什么有效
因果上看，它改变了两个瓶颈：

- **信息瓶颈变化**：  
  从“开放词汇语义理解未知物体”  
  变成“在有限3D对象槽位中选目标”。

- **规划空间变化**：  
  从“LLM 对机器人动作自由写作”  
  变成“在受限动作语法中搜索可行计划”。

因此能力变化也很直接：

- 对未知物体：更稳，因为不依赖名称；
- 对用户：更快，因为不必记复杂手势或长描述；
- 对机器人：更安全，因为有显式几何校验闭环。

### 方法链路

#### 1. 未知物体的结构化表示
作者没有把 SAM 当成“语义识别器”，而是当成**零样本区域提议器**：

- 用 SAM 分割场景中潜在物体区域；
- 将 mask 与深度对齐，重投影成 3D 点云；
- 为每个物体构造一个简化结构表示：
  - centroid
  - width / height / thickness
  - cluster index

一个很关键的细节是：

**即使 SAM 给出的语义名字是错的，系统也不依赖这个名字。**  
它只需要物体轮廓与几何位置。

#### 2. 并行多模态命令
作者把人类意图拆成三部分：

- **动作意图**：语音提取
- **目标意图**：指向姿态提取
- **参数意图**：语音中的角度、速度等可选指标

具体上：

- 语音经 VOSK 转文本，再解析动作与参数；
- OpenPose 提取人体关键点；
- 用右前臂方向线作为 deictic posture；
- 选择与该方向线最近的 3D 物体簇作为目标；
- 再通过 verbal approval 锁定最终目标。

这一步本质上是在做**模态分工**，而不是简单 fusion。

#### 3. 受约束的 LLM 计划生成
GPT-4-turbo 在这里不是直接看图像，而是接收已经结构化过的输入：

- 目标物体的几何信息
- 障碍物信息
- 动作命令和参数
- 动作约束、轨迹约束、示例任务

其输出是统一格式的机器人动作序列。  
这相当于把 LLM 的任务从“感知+规划+控制一把抓”收缩为“在结构化世界模型上做高层编排”。

#### 4. Cross-check 闭环
作者没有完全相信 GPT-4。

他们增加了一个低成本安全层：

- 把被操纵物体和障碍物简化为长方体；
- 沿生成轨迹做 swept-volume 检查；
- 若发生相交，则给 GPT-4 返回失败原因与碰撞位置；
- GPT-4 基于反馈重规划。

所以它不是单次生成，而是一个简化的**生成-验证-修正**闭环。

### 战略取舍

| 设计选择 | 带来的收益 | 代价/风险 |
|---|---|---|
| 用 SAM mask + depth，而非闭集检测类别 | 支持未知物体、无需知道名称 | 只靠几何槽位，缺少更细语义区分 |
| 语音负责动作、姿态负责目标 | 降低语言歧义和记忆负担 | 需要相机看见上半身，且通常要确认一步 |
| GPT-4 受约束生成动作序列 | 多步任务更灵活，可处理 pick/place/pour 组合 | 依赖闭源 API，行为受 prompt 质量影响 |
| 几何 swept-volume 校验 | 用较低成本提升安全性 | 长方体近似较粗糙，细粒度接触仍可能漏检 |

---

## Part III：证据与局限

### 关键证据

#### 1. 比较信号：交互效率明显提升
作者在 3 个桌面操作场景中，与 gesture-based / NLP-based / VLM-based 方法比较用户输入时间。最强结论不是“机器人更快”，而是：

**用户表达意图更快。**

- 相对 gesture-based：交互时长下降 **59.2%**
- 相对 NLP-based：下降 **65.2%**
- 相对 VLM-based：下降 **64.8%**

这说明能力跳跃主要来自**交互接口设计**，不是单纯底层运动学更强。

#### 2. 比较信号：在相似物体场景里准确率更高
在多个外观接近、位置相邻的物体场景中，NVP-HRI 在更复杂任务上仍保持较高准确率：

- S2 约 **94-96%**
- S3 约 **91%**

并高于 gesture / NLP / VLM 基线。  
这表明“姿态锁目标 + 语音表动作”的分工，确实比纯语言消歧稳定。

#### 3. 消融信号：cross-check 不是装饰
加入几何 cross-check 后：

- 无碰撞轨迹成功率从 **73% → 94%**

这是论文里最关键的因果证据之一：  
**真正提升安全性的不是单靠 GPT-4，而是“LLM + verifier feedback”组合。**

#### 4. 分析信号：年龄影响较小
作者的 ANOVA 分析显示：

- 基线方法的交互时长会受到年龄组显著影响；
- NVP-HRI 的交互时长影响不显著。

这支持了论文面向老年辅助场景的动机：  
**系统把用户负担从“记忆和描述”转成了“自然说+自然指”。**

#### 5. 偏好调查：外部可接受性较强，但证据偏软
- 390 份问卷中，超过 97% 参与者偏好 NVP-HRI。
- 但这是基于视频演示和主观排序，不是公开 benchmark。

所以这部分更像**可用性侧证**，不是强实证。

### 能力跃迁到底在哪里
相对 prior work，这篇论文最重要的提升不是“更强 detector”，也不是“更强 planner”，而是：

**把未知物体交互从“靠说名字”切换到“靠指向几何对象槽位”。**

这使得系统在面对新物体时不再被词汇表卡住，同时又通过结构化 prompt 和 cross-check，把 LLM 从高风险生成器压缩成较可控的决策模块。

### 局限性

- **Fails when**: 低光照导致姿态识别不稳定、用户上半身或前臂未完整入镜、目标严重遮挡、语音识别出错时，目标选择与意图解析都会退化；对于超出夹爪可操作尺寸的大物体，系统也不适用。
- **Assumes**: 依赖 RGBD 相机标定、OpenPose 可稳定跟踪上半身、英文语音输入、目标物体可通过 SAM 得到可用 mask、且可调用闭源 GPT-4-turbo API；同时默认用户能够给出 approval/finish 等控制词。
- **Not designed for**: 医院级真实部署验证、多语种或听障/失语用户、强隐私/离线环境、复杂非桌面开放场景、以及需要精细接触建模的高精度操作任务。

### 复现与扩展的现实约束
这篇论文的可扩展性还受几个实际依赖影响：

- **规划核心依赖闭源 GPT-4-turbo API**
- **代码是 promised open-source，不是论文内已完整复现包**
- **数据仅按请求提供**
- **实验主要在作者自建环境中完成，没有公共基准**
- **真实医院/养老机构场景尚未系统验证**

因此我会把证据强度定为 **moderate**，而不是更高。

### 可复用组件
这篇工作里最值得迁移到别的 embodied 系统的，不是整套 pipeline，而是这几个操作子：

1. **几何优先的未知物体槽位化**  
   用 SAM + depth 先建 3D 对象槽位，而不是硬做类别识别。

2. **姿态绑定目标、语言表达动作的模态解耦**  
   对 HRI 来说，这比简单多模态拼接更实用。

3. **受约束 LLM + verifier 的闭环规划**  
   先缩小输出空间，再用外部几何检查器兜底。

## Local PDF reference

![[paperPDFs/Other_Multimodal_Large_Language_Model_MLLM_based_related_Embodied_Learning_2025/arXiv_2025/2025_NVP_HRI_Zero_Shot_Natural_Voice_and_Posture_based_Human_Robot_Interaction_via_Large_Language_Model.pdf]]