---
title: "Chain-of-Modality: Learning Manipulation Programs from Multimodal Human Videos with Vision-Language-Models"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/video-understanding
  - sequential-prompting
  - multimodal-reasoning
  - code-generation
  - dataset/OpeningBottle
  - dataset/InsertingPlug
  - dataset/PlayingDrum
  - opensource/partial
core_operator: 按模态顺序链式提示VLM，先用力觉/音频做时序切分，再用手位姿补动作细节、用图像绑定对象语义，最后生成机器人API程序
primary_logic: |
  单段多模态人类示范视频（RGB + EMG/音频 + 手位姿） → 依次解析发力时刻、手部运动与目标对象并逐步细化任务计划 → 输出带方向/力度/时序参数的操作序列与机器人可执行代码
claims:
  - "CoM将从单段多模态人类视频中提取精确任务计划与控制参数的准确率提升到约60%，而视觉-only方法为0%、朴素多模态合并提示约为17% [evidence: comparison]"
  - "去掉力觉模态后，各任务上的精确成功率降为0，而加入力觉能显著改善任务阶段切分与计划相似度，平均相似度提升约42% [evidence: ablation]"
  - "基于CoM分析生成的机器人程序在六个真实机器人设置上平均成功率约73%，并在未见物体、随机摆放和双机器人平台上表现出可迁移性 [evidence: comparison]"
related_work_position:
  extends: "Code as Policies (Liang et al. 2023)"
  competes_with: "Vid2Robot (Jain et al. 2024); MimicPlay (Wang et al. 2023)"
  complementary_to: "HaMeR (Pavlakos et al. 2024); ReKep (Huang et al. 2024)"
evidence_strength: moderate
pdf_ref: paperPDFs/Other_Multimodal_Large_Language_Model_MLLM_based_related_Embodied_Learning_2025/arXiv_2025/2025_Chain_of_Modality_Learning_Manipulation_Programs_from_Multimodal_Human_Videos_with_Vision_Language_Models.pdf
category: Embodied_AI
---

# Chain-of-Modality: Learning Manipulation Programs from Multimodal Human Videos with Vision-Language-Models

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2504.13351), [Project](https://chain-of-modality.github.io)
> - **Summary**: 论文提出一种按模态逐步推理的提示策略 CoM，把单段人类多模态示范视频转成带力度与时序细节的机器人操作程序，解决视觉单模态难以恢复接触与发力信息的问题。
> - **Key Performance**: 单视频任务计划/控制参数提取准确率约 60%（vision-only 0%，naive merged 17%）；真实机器人六种设置平均成功率约 73%。

> [!info] **Agent Summary**
> - **task_path**: 单段人类多模态示范视频（RGB/EMG或音频/手位姿） -> 任务计划与控制参数 -> 机器人Python API程序
> - **bottleneck**: 视觉视频缺少力度与接触阶段信息，而VLM把多模态一次性混合输入时又难以稳定完成跨模态对齐与时序分段
> - **mechanism_delta**: 把“直接从全部模态到最终答案”改成“力觉/音频 -> 手位姿 -> 图像”的阶段式提示，每一步只补当前模态最可靠的信息
> - **evidence_signal**: 模态消融与推理流程消融都表明CoM优于merged/separated baselines，且真实机器人平均成功率约73%
> - **reusable_ops**: [force-first temporal segmentation, staged multimodal prompt refinement]
> - **failure_modes**: [缺少力觉时难以恢复阶段边界与控制参数, 开环执行遇到扰动或定位误差时缺乏在线纠错]
> - **open_questions**: [如何把CoM接入闭环反馈控制, 如何利用超越音量的更丰富音频特征]

## Part I：问题与挑战

这篇论文解决的核心问题，不是普通的“看视频识别动作”，而是：

**机器人能否只看一段人类示范视频，就恢复出可执行的 manipulation program？**

这里的真正难点在于，很多操作任务的关键不在“动作名字”，而在**控制参数**：
- 什么时候开始用力、什么时候松手；
- 力度大小如何变化；
- 抓、放、扭、插、敲这些动作如何分段；
- 目标对象到底是谁。

仅靠视觉，很多这类信息并不显式可见。  
例如插头插入时，先要轻握调整朝向，再大力推进；打鼓要区分轻敲和重击；开瓶盖要识别反复抓放与旋转方向。纯 RGB 视频往往只能看到外观变化，看不到接触强度和隐含意图。

作者认为真实瓶颈有两层：

1. **信息瓶颈**：视觉缺少力觉线索，无法稳定恢复 contact-rich manipulation 的细节。
2. **推理瓶颈**：即便 VLM 能吃下长上下文，把图像、EMG/音频、手位姿一次性塞进去，模型也常常会忽略某个模态，或者从错误模态里“猜”答案。

### 输入/输出接口

- **输入**：单段人类多模态示范视频  
  - RGB 视频帧
  - 肌电 EMG 或交互音频
  - 手部位姿（文中用 fingertip pixel locations）
- **输出**：
  1. 结构化任务计划：动作序列 + 对象 + 时序 + 方向 + 力度
  2. 机器人可执行 Python API 程序

### 为什么是现在？

这件事现在值得做，主要因为三个条件成熟了：
- 长上下文 VLM 已经能同时读视频和长数值序列；
- 可穿戴 EMG、麦克风让人类示范中的“隐含控制线索”可采集；
- 机器人侧已有 `Code as Policies` 一类 API 化执行范式，能把计划直接落成程序。

### 边界条件

这篇工作的适用边界也很明确：
- **one-shot**：只给一段示范，不给任务文字说明；
- **技能库已知**：机器人动作来自预定义 API；
- **力觉是近似量**：EMG 用最大通道值、音频只用 loudness；
- **执行是开环**：生成代码后直接执行，不做在线闭环修正。

---

## Part II：方法与洞察

这篇论文的方法本质上是一个**prompting strategy**，不是新训练一个模型。  
作者提出 **Chain-of-Modality (CoM)**：让 VLM 不要一次性同时处理所有模态，而是**按模态顺序逐步分析、逐步修正答案**。

### 方法流程

#### 1. 多模态人类示范采集
每个时间步包含：
- RGB 图像
- EMG 或音频信号
- 手位姿

其中：
- EMG/音频主要提供**何时发力、发力强弱**；
- 手位姿主要提供**抓放、旋转方向、手部细粒度运动**；
- 图像主要提供**对象身份与场景语义**。

#### 2. CoM：按模态链式分析
CoM 的 prompt 包含三部分：
- 每种模态的输入格式说明；
- 可用动作库及其参数说明；
- 一个“视频到分析结果”的示例。

然后按顺序推理：

1. **先看力觉/音频**  
   找到发力峰值、接触阶段、动作次数，先形成粗粒度时序骨架。

2. **再看手位姿**  
   在已有时序骨架上，补出抓取/释放、扭转、方向、角度等动作细节。

3. **最后看图像**  
   把动作绑定到具体对象，例如 bottle cap、plug、drum。

这样最终得到结构化任务计划，而不是直接让模型一步到位输出最终程序。

#### 3. 从任务计划到机器人代码
作者随后复用同一个 VLM，根据：
- CoM 生成的结构化分析，
- 机器人 API 定义，

自动生成 Python 程序，例如：
- `Find`
- `Move_to`
- `Grasp`
- `Release`
- `Twist`
- `Insert`

这样做的好处是：输出层不是具体关节轨迹，而是**跨 embodiment 的技能程序**，更容易迁移到不同机器人。

### 核心直觉

CoM 的关键不是“加更多模态”，而是：

**把高耦合的联合多模态推理，改成受约束的逐步潜变量补全。**

更具体地说：

- **先加力觉/音频**  
  改变的是“时序切分”瓶颈。  
  先确定何时接触、何时发力、动作发生了几次，视频被切成更合理的子阶段。

- **再加手位姿**  
  改变的是“动作几何细节”瓶颈。  
  模型不必再从原始像素里硬猜手部运动，而是直接读出抓放切换、扭转方向等信息。

- **最后加图像**  
  改变的是“对象语义落地”瓶颈。  
  当动作骨架已稳定后，再做对象识别，能减少早期视觉猜测带来的错误传播。

能力上的变化是：
- 从“看懂大概发生了什么”
- 变成“能恢复出机器人可执行的动作-参数程序”。

这也是它相对 merged prompting 的真正增量：  
**不是更大的输入，而是更好的因果顺序。**

### 策略性取舍

| 设计选择 | 改变的瓶颈 | 带来的能力 | 代价/风险 |
|---|---|---|---|
| 先看力觉/音频 | 难以稳定做阶段切分 | 更可靠地定位发力时刻、动作次数、接触阶段 | 只能知道“何时用力”，不知道对象是谁 |
| 再看手位姿 | 像素级手部细节难读 | 更好恢复抓/放、旋转方向、细粒度动作 | 依赖外部手部估计器 HaMeR |
| 最后看图像 | 动作语义与对象语义纠缠 | 把动作绑定到 bottle cap / plug / drum 等对象 | 遮挡、视角变化仍会影响 grounding |
| 输出 API 程序而非轨迹 | 直接学轨迹难以跨机器人迁移 | 更容易部署到 ViperX / KUKA 等不同 embodiment | 强依赖预定义技能库与感知 API |
| 开环执行 | 降低系统复杂度 | 便于快速验证 video-to-program | 缺少在线纠错与自适应 |

---

## Part III：证据与局限

### 关键证据信号

#### 1. 推理流程的改动本身有效
作者比较了多种推理方式：
- Merged
- Merg-Sep
- Sep-Merg
- Sep-Sep
- CoM

核心信号是：**不是简单把模态分开就够了，而是“按顺序逐步精化”更强。**  
文中报告 CoM 相比最强分步基线 Sep-Sep：
- 在 **Gemini 1.5 Pro** 上还能提升 **19%+**
- 在 **GPT-4o** 上提升 **17%+**

这说明有效因素不是“多一次问答”，而是**顺序化的跨模态约束**。

#### 2. 力觉模态是关键因子，不是可有可无的附加信息
模态消融给出的结论非常直接：
- 去掉 **force** 后，精确成功率基本掉到 **0**
- 加入 force 后，任务计划与 GT 的平均相似度提升约 **42%**

这支持了论文的主论点：  
**操作视频里的关键缺失信息，不是更多视觉细节，而是发力与接触线索。**

#### 3. 图像与手位姿各自承担不同角色
- 去掉 **image** 时，对象识别失败，成功率为 0，说明图像主要负责对象 grounding。
- **Opening Bottle** 这类任务只有在包含手位姿时才有非零成功，说明手位姿对旋转方向、抓放切换等细节至关重要。

因此，三种模态并不是冗余关系，而是：
- force/audio：阶段边界与力度
- hand pose：细粒度动作几何
- image：对象与场景语义

#### 4. 真正“落地到机器人”了，但距离上限还有差距
真实机器人实验共 6 个设置，平均成功率约 **73%**，而人工写的 Oracle 程序约 **92%**。  
这说明 CoM 已经能把“视频理解”变成“可运行程序”，但系统误差还不小。

有意义的泛化信号包括：
- **未见瓶子** 的开瓶任务；
- **随机摆放** 的插头/插座/盒子；
- **不同图案** 的擦板；
- **不同节奏** 的打鼓；
- 同一开瓶程序迁移到 **ViperX** 和 **KUKA** 两种双臂平台。

### 局限性

- **Fails when:** 没有力觉/音频时，系统难以稳定恢复阶段边界与控制参数；执行时若 `Find` 定位错误、物体初始位姿偏差较大、或中途出现扰动，开环程序缺乏自我修正能力。
- **Assumes:** 依赖强长上下文闭源 VLM（Gemini 1.5 Pro / GPT-4o）、Gemini 支持的开放词汇目标定位、RGB-D 感知、预定义技能库、EMG armband/麦克风、以及外部手位姿估计器 HaMeR；这些依赖直接影响可复现性、成本与部署门槛。
- **Not designed for:** 端到端连续控制、精确闭环力控、无技能库条件下的低层动作发现，也不面向利用频率/音色等 richer audio cues 的完整声学推理。

### 可复用组件

这篇论文最值得复用的不是具体任务，而是几个系统算子：

- **per-modality staged prompting**：按模态顺序逐步求精，而不是一次性拼接输入；
- **force-first segmentation**：先用力觉/音频建立时序骨架；
- **structured analysis -> code synthesis**：先产出结构化计划，再生成程序；
- **API-level embodiment abstraction**：把人类示范转成技能级代码，而不是机器人专属轨迹；
- **open-vocabulary object grounding**：让程序中的对象名通过视觉 API 在线落地。

**一句话总结 So what：**  
这篇工作把“从人类视频学机器人”从单纯的视频模仿，推进到了**多模态示范 -> 结构化计划 -> 可执行程序**的链路上；真正的能力跃迁来自对“力觉缺失 + 联合推理失稳”这两个瓶颈的同时处理。

## Local PDF reference

![[paperPDFs/Other_Multimodal_Large_Language_Model_MLLM_based_related_Embodied_Learning_2025/arXiv_2025/2025_Chain_of_Modality_Learning_Manipulation_Programs_from_Multimodal_Human_Videos_with_Vision_Language_Models.pdf]]