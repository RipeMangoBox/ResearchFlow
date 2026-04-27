---
title: "MindEye-OmniAssist: A Gaze-Driven LLM-Enhanced Assistive Robot System for Implicit Intention Recognition and Task Execution"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/intention-recognition
  - task/robot-manipulation
  - large-language-model
  - open-vocabulary-detection
  - transformer
  - "dataset/eye movement dataset"
  - opensource/no
core_operator: 将“注视到哪些对象”的低带宽眼动信号分解为逐对象交互判别，再用LLM把对象序列补全为完整意图并映射成机器人动作序列。
primary_logic: |
  场景图像与2D注视流 → 用开放词汇检测得到候选对象，并以大小归一化的 gaze-object 几何特征做逐对象“是否想交互”判别 → 将被选对象序列交给 LLM 推断完整意图并通过注视确认 → LLM 基于动作 API 生成执行序列，机器人完成辅助任务
claims:
  - "在自建 eye movement dataset 的实验划分评测中，逐对象意图识别准确率达到 96.52±1.69%，高于 MIDAS-Net 的 91.13±2.44% 和 fixation 基线的 61.17±2.93% [evidence: comparison]"
  - "在跨受试者评测中，该意图识别网络平均准确率为 93.63±1.39%，高于 MIDAS-Net 的 88.18±2.07% 和 fixation 基线的 59.83±0.69% [evidence: comparison]"
  - "在 55 次真实桌面辅助任务中，系统端到端完成 41 次，且意图识别与动作规划阶段分别达到 52/55 和 50/52 的成功数，显示主要剩余瓶颈在物理执行而非高层语义推断 [evidence: analysis]"
related_work_position:
  extends: "Gaze-based, Context-Aware Robotic System for Assisted Reaching and Grasping (Shafti et al. 2019)"
  competes_with: "MIDAS-Net (Festor et al. 2022); Hybrid implicit intention inference (Gao et al. 2023)"
  complementary_to: "N/A"
evidence_strength: moderate
pdf_ref: paperPDFs/Other_Multimodal_Large_Language_Model_MLLM_based_related_Embodied_Learning_2025/arXiv_2025/2025_MindEye_OmniAssist_A_Gaze_Driven_LLM_Enhanced_Assistive_Robot_System_for_Implicit_Intention_Recognition_and_Task_Execution.pdf
category: Embodied_AI
---

# MindEye-OmniAssist: A Gaze-Driven LLM-Enhanced Assistive Robot System for Implicit Intention Recognition and Task Execution

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2503.13250)
> - **Summary**: 这篇工作把“眼睛看向哪些物体”先转成逐对象交互意图，再借助 LLM 做常识级任务补全与 API 级动作规划，从而让注视驱动的辅助机器人不再局限于预定义抓取任务。
> - **Key Performance**: 单物体意图识别准确率 96.52±1.69%；55 次真实桌面任务端到端成功 41/55，意图识别 52/55、规划 50/52。

> [!info] **Agent Summary**
> - **task_path**: 场景图像+2D眼动注视流 -> 隐式用户意图 -> 机器人动作序列与任务执行
> - **bottleneck**: gaze 只显式给出“看哪里”，却不直接给出“想做什么”；传统系统又依赖预定义任务图或 FSM，导致开放任务泛化很差
> - **mechanism_delta**: 把闭集任务分类改写为“开放词汇对象检测 + 逐对象 gaze 交互二分类 + LLM 常识补全 + API 组合执行”的分解式管线
> - **evidence_signal**: 自建眼动数据上稳定优于 MIDAS-Net，且真实系统中高层识别/规划成功率明显高于端到端执行成功率，表明语义链路基本可用
> - **reusable_ops**: [size-normalized gaze-object geometry, object-wise intention filtering]
> - **failure_modes**: [gaze drift或个体差异导致对象判错, 抓取/倾倒/按压等低层执行失败拖累端到端表现]
> - **open_questions**: [如何泛化到真实肢体障碍用户, 如何超越固定API库并加入执行期纠错]

## Part I：问题与挑战

这篇论文真正要解决的，不是“让用户用眼睛选一个杯子”，而是：

1. **从低带宽、噪声大的 gaze 信号中恢复隐式意图**；
2. **不把意图空间限制在预定义任务集**；
3. **把推断出的意图转成机器人可执行的动作链**。

### 问题是什么
现有 gaze-based assistive system 大多停留在两类能力：
- 只支持抓取/放置等少数基础动作；
- 或者虽然能触发任务，但任务与对象库都预先写死，依赖 FSM 或人工先验矩阵。

这意味着系统虽然“能看懂你看了什么”，却**看不懂你为什么看它**。而辅助机器人真正需要的是后者。

### 真正瓶颈在哪里
核心瓶颈是一个**语义鸿沟**：

- gaze 给的是空间注意线索，不是自然语言命令；
- 若直接学习 `gaze → task label`，标签空间会随着对象组合和任务种类迅速膨胀；
- 一旦场景里出现未定义对象或未枚举任务，传统系统就会失效。

### 输入/输出接口
- **输入**：场景图像 + 2D gaze 点序列
- **中间表示**：候选对象、每个对象是否是交互目标、LLM 推断的候选意图
- **输出**：经用户确认的意图 + 机器人 API 动作序列

### 为什么现在能做
作者抓住了一个时机点：  
**视觉基础模型**可以把场景转成开放词汇对象集合，**LLM**可以用常识把对象序列补全成活动意图。  
因此可以把原来难以泛化的端到端任务分类，拆成：

`对象级交互判断` + `常识级任务推理`

这是本文最关键的切分。

---

## Part II：方法与洞察

MindEye-OmniAssist 不是直接把图像和 gaze 丢给一个大模型，而是做了一个更稳的分层流水线。

### 方法主线

1. **开放词汇对象检测**
   - 用 YOLO-World 检测场景中的对象。
   - 这一步把系统从“固定物体数据库”解耦出来。

2. **逐对象 gaze 意图识别**
   - 对每个检测到的对象，都构造一个与该对象相关的 gaze 时序特征。
   - 特征不是纯 gaze 点，而是加入了**对象尺寸归一化后的 gaze-object 相对几何关系**。
   - 网络结构采用 multi-scale convolution + positional encoding + Transformer encoder。
   - 输出是一个二分类：用户是否打算与这个对象交互。

3. **基于 LLM 的完整意图推断**
   - 把被筛出来的对象序列变成文本提示，例如“用户依次看了 toothbrush, toothpaste, cup”。
   - 用 DeepSeek-R1 输出可能意图。
   - 这一步承担的是“对象组合 -> 活动语义”的 commonsense reasoning。

4. **注视式确认**
   - 场景图像被切成上/中/下三个区域。
   - 上区 = Reject，下区 = Agree，中区保留给自然观察。
   - 这相当于给低带宽 gaze 交互加了一个轻量级安全闭环。

5. **LLM 动作规划 + API 执行**
   - 确认后，LLM 基于预定义操作 API 生成动作序列。
   - 简单动作手工编程，复杂动作用 behavior cloning 预先学习。
   - 所以它的“开放性”主要体现在高层任务组合，而不是低层技能无限开放。

### 核心直觉

这篇工作的关键不是“用了 LLM”，而是：

> **把难学的开放任务分类问题，改写成更容易学的逐对象交互判别问题；再把剩余的高层语义交给 foundation model。**

#### 具体因果链
- **What changed**  
  从 `gaze -> 预定义任务标签/FSM状态`  
  改成 `gaze + scene -> 想交互的对象集合 -> LLM补全任务意图 -> API组合执行`

- **Which bottleneck changed**  
  学习器不再需要记住大量“对象组合-任务名”的闭集映射，只需判断“这个对象是不是当前交互目标”。  
  这显著降低了监督空间的复杂度，也减少了对场景先验矩阵的依赖。

- **What capability changed**  
  系统从“只能做写死的抓取/几个任务”变成“能够对未在 FSM 中逐一枚举的对象组合做语义补全并执行”。

#### 为什么这个设计有效
- **gaze 对对象选择比对完整任务语义更直接**：用户看哪个物体，往往比“他最终要完成什么任务”更容易从眼动里读出来。
- **LLM 擅长补全常识缺口**：比如看 toothbrush + toothpaste + cup，更容易由 LLM 推断到“刷牙”这类活动。
- **确认机制抑制高风险误执行**：先让用户 agree/reject，再执行动作，降低开放推理带来的错误代价。

### 战略取舍

| 设计选择 | 改变了什么约束 | 带来的能力 | 代价/风险 |
|---|---|---|---|
| 开放词汇检测替代预置物体库 | 对象空间从 closed-set 变为 open-vocabulary | 场景中未预录入物体也可被纳入推理 | 检测错误会向后级联 |
| 逐对象二分类替代直接任务分类 | 学习目标从多任务标签降到二元交互判断 | 小数据下更容易学，跨场景更稳 | 仍强依赖 gaze 校准质量 |
| 把对象序列文本化后交给 LLM | 不需要大规模 gaze-language 端到端数据 | 能借常识补全未定义任务 | 丢失部分细粒度视觉/时序信息，且有语义歧义风险 |
| API 组合替代 FSM | 从任务枚举转为动作组合 | 任务覆盖更灵活 | 上限被 API 库和低层技能库锁死 |
| 注视确认闭环 | 从一次性推断变为人机协同修正 | 增强安全性和可用性 | 增加一步交互时延 |

---

## Part III：证据与局限

### 关键证据

1. **比较信号：对象级意图识别前端确实有效**
   - 在实验划分评测上，作者方法达到 **96.52±1.69%**，高于 MIDAS-Net 的 **91.13±2.44%**。
   - 在跨受试者评测上，作者方法仍有 **93.63±1.39%**，高于 MIDAS-Net 的 **88.18±2.07%**。
   - 这说明“逐对象二分类 + gaze-object 几何特征”这个前端，不只是记住单次实验，而有一定跨人泛化能力。

2. **系统分解信号：高层语义链路已基本打通**
   - 真实系统 55 次任务中：
     - 意图识别：**52/55**
     - 动作规划：**50/52**
     - 端到端完成：**41/55**
   - 这组分解很有信息量：**识别和规划已经明显强于最终执行**，说明瓶颈不再主要在“看不懂用户想做什么”，而转向“机器人能否稳定把动作做出来”。

3. **能力边界信号：比传统 gaze grasp 更广，但还不是完全开放世界**
   - 系统覆盖取物、放入、浇水、开关、倒水等多类桌面任务，确实超出“只会抓一下”的传统设置。
   - 但论文没有提供端到端系统级 baseline 对比，也没有做去掉 LLM / 去掉开放词汇检测 / 去掉确认模块的消融，因此“到底哪一部分最关键”还缺少强因果证据。

### 1-2 个最值得记住的指标
- **96.52±1.69%**：单物体交互意图识别准确率
- **41/55**：真实桌面辅助任务端到端成功数

### 局限性
- **Fails when**: gaze 漂移、眨眼干扰、个体眼动模式差异较大时，系统可能把注视归错到对象上；对象组合存在多种合理解释时，LLM 推断会有歧义；抓取、倾倒、开关按压等接触操作失败会直接拖垮端到端成功率。
- **Assumes**: 需要头戴式眼动仪并频繁校准；默认是单用户、桌面、有限物体数场景；需要人工标注的 gaze-intent 数据训练前端网络；需要已有动作 API 库与 behavior cloning 技能；实验对象主要是年轻志愿者，未验证目标残障用户群体的 gaze 行为分布。
- **Not designed for**: 大范围移动操作、拥挤遮挡严重的家庭开放环境、需要长对话澄清的复杂任务、超出既有 API/技能库的新型操作。

### 复现与扩展上的现实约束
- 硬件依赖明显：Pupil Invisible、四协作臂、RGB-D 相机、ROS 集成。
- 数据依赖明显：自建数据集仅 8 名受试者、约 120 分钟、人工标注。
- 开放性有限：论文未公开系统代码/数据；高层语义是开放的，但低层动作仍被 API 库封顶。

### 可复用组件
- **对象级 gaze 意图过滤器**：适合任何“用户目光 -> 目标对象”场景。
- **对象序列到任务意图的 LLM 桥接**：适合低带宽输入到高层语义补全。
- **Agree/Reject 注视确认界面**：适合无需手部输入的人机协作闭环。

## Local PDF reference

![[paperPDFs/Other_Multimodal_Large_Language_Model_MLLM_based_related_Embodied_Learning_2025/arXiv_2025/2025_MindEye_OmniAssist_A_Gaze_Driven_LLM_Enhanced_Assistive_Robot_System_for_Implicit_Intention_Recognition_and_Task_Execution.pdf]]