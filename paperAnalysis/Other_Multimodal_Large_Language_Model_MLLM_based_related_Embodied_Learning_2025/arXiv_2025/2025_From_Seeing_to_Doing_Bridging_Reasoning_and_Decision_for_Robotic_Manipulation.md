---
title: "From Seeing to Doing: Bridging Reasoning and Decision for Robotic Manipulation"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/robotic-manipulation
  - task/spatial-reasoning
  - chain-of-thought
  - self-consistency
  - intermediate-representation
  - dataset/VABench
  - dataset/SimplerEnv
  - dataset/BridgeDataV2
  - opensource/full
core_operator: "把动作预测改写为空间关系图驱动的视觉CoT推理，生成对象中心的2D视觉中间表示以指导机器人操控"
primary_logic: |
  图像+自然语言指令 → 通过区域描述与对象关系构建空间关系图，并在SrCoT中逐步推理起点/终点/中间点 → 输出affordance框/点或8点visual trace，并用正逆任务自一致对齐坐标语义后交给规划器执行
claims:
  - "Claim 1: FSD在5个空间推理基准的15个子任务上取得平均排名1.3，显著优于其他13B开源模型，并接近GPT-4o [evidence: comparison]"
  - "Claim 2: 在VABench上，FSD的affordance点准确率达到61.82%，visual trace的RMSE/MAE为78.26/63.44；去掉SrCoT后点准确率降至26.21%，说明显式空间CoT是主要增益来源 [evidence: ablation]"
  - "Claim 3: 在零样本操控中，FSD在SimplerEnv达到40.6%成功率、在8个真实机器人任务上达到72%成功率，分别显著高于RoboPoint和真实场景最强基线 [evidence: comparison]"
related_work_position:
  extends: "ASMv2 (Wang et al. 2025)"
  competes_with: "RoboPoint (Yuan et al. 2024b); MOKA (Liu et al. 2024a)"
  complementary_to: "CuRobo (Sundaralingam et al. 2023); GraspNet (Fang et al. 2020)"
evidence_strength: strong
pdf_ref: paperPDFs/Other_Multimodal_Large_Language_Model_MLLM_based_related_Embodied_Learning_2025/arXiv_2025/2025_From_Seeing_to_Doing_Bridging_Reasoning_and_Decision_for_Robotic_Manipulation.pdf
category: Embodied_AI
---

# From Seeing to Doing: Bridging Reasoning and Decision for Robotic Manipulation

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2505.08548), [Project](https://embodied-fsd.github.io/), [Code](https://github.com/pickxiguapi/Embodied-FSD), [Datasets](https://huggingface.co/collections/IffYuan/fsd)
> - **Summary**: 这篇工作把机器人操控从“直接学动作”改成“先做带坐标约束的空间推理，再生成对象中心的2D视觉中间表示”，从而把VLM的“看懂”能力更稳定地转成零样本可执行的操控决策。
> - **Key Performance**: VABench-Point 61.82%；零样本真实机器人8任务成功率72%（SimplerEnv 40.6%）

> [!info] **Agent Summary**
> - **task_path**: 单张RGB图像 + 自然语言操控指令 -> affordance框/点或8点object-centric visual trace -> 深度回投/运动规划 -> 机器人执行
> - **bottleneck**: embodied数据稀缺且跨机器人异构，导致从视觉/语言直接监督到动作的映射难以零样本泛化；同时坐标token缺乏预训练语义
> - **mechanism_delta**: 用空间关系图锚定的SrCoT把“坐标预测”改写为“多步空间推理”，并用正逆任务自一致让坐标与图像语义绑定
> - **evidence_signal**: 多空间基准 + VABench消融 + SimplerEnv/真实机器人零样本部署一致提升
> - **reusable_ops**: [spatial-relationship-graph-anchored-cot, generation-understanding-self-consistency]
> - **failure_modes**: [长时程或模糊指令需额外任务分解, 接触丰富或关节物体任务中2D轨迹到控制执行仍可能失真]
> - **open_questions**: [如何把2D visual aids升级为3D或闭环控制信号, 如何把视觉中间表示扩展到长时序多子任务规划]

## Part I：问题与挑战

这篇论文要解决的，不是“机器人能不能看见”，而是**看见之后怎样变成可泛化的决策**。

### 1. 真正的问题是什么
现有 VLA 路线常把：
- 输入：图像 + 语言指令  
- 输出：机器人动作/动作token

直接连成端到端映射。问题在于，这条映射对**机器人本体、相机视角、控制频率、抓取器形态**都非常敏感。同一个语义任务，在不同 embodiment 下对应的动作分布差异极大。

作者的判断是：  
**VLA 零样本泛化差，不只是模型不够大，而是监督接口选错了。**

### 2. 真正瓶颈在哪里
论文把瓶颈拆成两层：

1. **数据稀缺**  
   embodied 数据远小于互联网级图文数据，难以靠 scaling law 直接学出鲁棒的“图像/语言→动作”映射。

2. **数据异构**  
   不同机器人、不同相机、不同控制空间下，动作标签不可直接共享；这使得端到端动作监督很难抽取跨平台共性。

除此之外，还有一个更细的技术瓶颈：  
**坐标本身并不是 VLM 预训练时天然理解的语义单位。**  
模型即使能“看懂”锅、盘子、抽屉，也不代表它能稳定输出与图像严格对齐的坐标点或轨迹。

### 3. FSD 重新定义了 I/O 接口
FSD 不直接输出动作，而输出一种**机器人无关的视觉中间表示**：
- spatial affordance box：可放置区域框
- spatial affordance points：可放置目标点
- visual trace：对象中心的2D操作轨迹

于是接口变成：

**单张RGB图像 + 指令 → 可解释的2D空间中间表示 → 规划器/抓取器执行**

这一步的意义是：  
把原本高度依赖机器人本体的动作空间，替换成更通用的**对象中心空间结构**。

### 4. 这件事为什么现在值得做
因为现在的 VLM 已经具备相当强的：
- 目标识别
- 区域描述
- 空间关系理解
- 多步语言推理

但这些能力此前并没有被一个合适的中间接口转化为 manipulation decision。  
FSD 的核心就是把这层“看懂世界”的能力，接到“做动作”之前。

### 5. 任务边界
这篇工作主要针对：
- 单视角桌面操作
- 短程或中等复杂度指令
- 以几何位置为核心的操作目标
- 零样本场景/对象/指令泛化

它**不是**一个完整的闭环低层控制系统，更像是一个“高层可执行空间意图生成器”。

---

## Part II：方法与洞察

FSD 的设计哲学可以概括为一句话：

**不要让模型直接猜动作；先让模型生成一个对机器人无关、但对执行足够有用的空间计划。**

### 方法总览

FSD 由三条主线组成：

1. **SrCoT：Spatial Relationship-Focused Visual Chain-of-Thought**  
   用空间关系图作为锚点，逐步推理 visual aids。

2. **Weak-to-Strong 数据构造**  
   从 grounding → 空间关系 → 空间推理 → affordance → visual trace，分层训练能力。

3. **Self-Consistent Alignment**  
   同时训练“生成 visual aids”和“根据 visual aids 反推指令”，让坐标真正带上语义。

### 核心直觉

#### 改了什么
FSD 把输出空间从：
- **robot-specific action**
改成：
- **object-centric 2D visual aids**

又把生成方式从：
- **直接坐标回归**
改成：
- **有锚点的空间推理**

#### 改变了哪一类瓶颈
这相当于把学习问题从：

**异构、难共享的动作分布**

转成：

**相对稳定、跨平台共享的空间关系分布**

同时，SrCoT 让模型不必“凭空”输出坐标，而是先：
1. 找对象
2. 建关系
3. 定起点/终点
4. 推中间点或目标区域

所以被改变的不是表面输出格式，而是**信息瓶颈**：  
模型从“直接把视觉压缩成动作”变成“先显式展开任务相关空间结构，再做决策”。

#### 为什么这会有效
因果上，FSD 的收益来自三件事叠加：

- **对象中心表示**削弱了 embodiment 差异  
  同一任务在不同机器人上，视觉目标位置往往比动作序列更容易共享。

- **空间关系图提供稳定推理锚点**  
  模型不再直接 hallucinate 坐标，而是围绕已定位对象做多跳推理。

- **正反向自一致训练给坐标赋语义**  
  如果模型能从 trace 反推出任务意图，说明它不仅“会输出数字”，而是理解这些数字代表什么。

#### 能力上带来了什么变化
从 prior work 的“能指一个点/一个框”，推进到：
- 更强的 free-space 推理
- 更强的 object-centric visual trace 生成
- 更好的零样本操控迁移

换句话说，FSD 的能力跃迁不是更强的末端控制，而是**更强的可泛化中间决策**。

### 关键组件拆解

#### 1. Visual aids：把“做”前移成可解释空间表示
论文定义三种 2D normalized visual aids：
- **Box**：适合“放到哪里”
- **Points**：适合更精细的放置目标
- **Trace**：适合“怎么移动过去”

其中 visual trace 固定为 8 个点，便于统一训练与执行。

一个重要选择是：  
**它是 object-centric，而不是 agent-centric。**  
这意味着它描述的是“被操作物体如何移动”，而不是某个特定机械臂末端如何移动，因此更利于跨 embodiment 泛化。

#### 2. SrCoT：把坐标生成变成结构化推理
SrCoT 分两步：

- **Description**：生成对象级区域描述与空间关系图  
  节点是对象及其框，边是 left/right/above/below 等关系。

- **Reasoning**：基于关系图逐步推理  
  先确定起点对象，再定位目标区域，再补出中间点或避障路径。

实现上，作者用：
- `<ref>` 绑定对象
- `<box>` 绑定框
- `<point>` 绑定点

这相当于做了**符号级视觉-坐标对齐**，减少“说对了但点错了”的情况。

#### 3. Weak-to-Strong 数据构造：先补空间前置能力
FSD 不是直接拿 trace 数据硬训，而是设计了五级能力课程：

1. Region Grounding  
2. Spatial Relationship  
3. Spatial Reasoning  
4. Spatial Affordance Generation  
5. Visual Trace Generation  

数据主要从 BridgeDataV2、RT-X、Droid 等大规模 embodied 数据自动构造，并结合通用 VQA/对话数据混训。  
核心思想是：**先让模型会“找”和“比”，再让它会“放”和“画轨迹”。**

#### 4. Self-Consistent Alignment：让坐标真正可理解
前向任务：
- 图像 + 指令 → trace / affordance

逆向任务：
- 图像 + trace → 指令

这个设计很关键，因为坐标本身不是预训练语义单元。  
通过正反配对，模型被迫学到：  
**某个点、某条轨迹、某个目标框，在图像里到底意味着什么操作意图。**

#### 5. 执行层：FSD 负责“决策表示”，不负责“低层控制”
落地执行时：
- affordance box/point → 用 CuRobo 规划
- visual trace → 深度回投到 3D，再结合 GraspNet 和轨迹插值执行

所以 FSD 的创新重点是**决策接口**，不是重新发明一个低层 policy。

### 战略取舍

| 设计选择 | 解决了什么 | 带来的能力 | 代价/风险 |
|---|---|---|---|
| 对象中心2D visual aids | 降低跨机器人动作异构性 | 跨平台、零样本更稳 | 丢失精细3D/力学信息 |
| 空间关系图锚定的 SrCoT | 避免直接坐标回归过拟合 | free-space/trace 生成更可解释 | 依赖前置 grounding 准确 |
| 五级弱到强课程数据 | 补齐空间推理前置能力 | seeing 与 doing 共增益 | 自动标注噪声会传递 |
| 正反向自一致对齐 | 让坐标 token 带语义 | 降低 hallucination、提升一致性 | 训练和数据管线更复杂 |
| 单模型生成 + 外部规划执行 | 简化系统接口 | 零样本部署更轻量 | 最终成功率受执行器/规划器制约 |

---

## Part III：证据与局限

### 关键证据

#### 信号1：一般空间推理显著增强
在 5 个空间基准、15 个子任务上，FSD 的**平均排名为 1.3**，优于其他 13B 开源模型，并接近 GPT-4o。  
这说明它不是只在机器人数据上记住模板，而是确实提升了更一般的空间理解能力。

可读结论：
- 它的“doing”提升，建立在“seeing + spatial reasoning”先变强的基础上。
- 这支持作者的核心论点：**先补空间推理，再做操控，会比直接学动作更稳。**

#### 信号2：对 free-space 和目标区域的定位更强
在 RoboRefIt 上，FSD 达到 **56.7%**，明显高于 RoboPoint 的 49.8% 和 GPT-4o 的 15.3%。  
在 Where2Place 上，FSD **45.8%**，与 RoboPoint **46.0%** 接近。

这说明：
- FSD 在**对象引用**上更强；
- 在更难的**自由空间放置**任务上，也至少达到了强基线级别。

#### 信号3：VABench 上的提升最能说明机制有效
VABench 是论文专门构建的 visual aids benchmark。这里的提升最能直观看到“从 seeing 到 doing”的桥接是否成功。

关键结果：
- **VABench-Point**：61.82%，远高于 RoboPoint 19.09%
- **VABench-VisualTrace**：RMSE 78.26，MAE 63.44，优于 GPT-4o、RoboBrain 和 DINOv2 Predictor

更重要的是消融：
- 去掉 **SrCoT** 后，点准确率从 61.82% 掉到 **26.21%**
- 去掉 **Alignment** 后，点准确率降到 **55.92%**

最强证据指向很明确：  
**SrCoT 是主增益来源，自一致对齐提供额外稳定增益。**

#### 信号4：零样本操控真正落地
这篇论文最有说服力的地方，不是只在 benchmark 上分数高，而是它真的做了零样本执行。

- **SimplerEnv**：40.6% 成功率  
  明显高于 RoboPoint 的 17.7%；虽然没有超过最强的专门调优端到端 VLA，但在**无需任务专门微调**的前提下已经很强。

- **真实机器人 8 任务**：72% 成功率  
  超过最强基线 30%+，且能完成基线做不了的 visual-trace 类任务，如毛巾折叠。

#### 信号5：推理-执行链条也更轻
单模型一次性生成 visual aids，再交给规划器执行，使 FSD 在真实系统中延迟低于 OpenVLA(FT) 和 MOKA。  
这说明它不只在“能力”上有收益，在部署复杂度上也有优势。

### So what：真正的能力跃迁在哪里
相较于 prior work，FSD 的跳跃点不是“更会预测动作”，而是：

**把 VLM 的空间理解变成了一个可执行、可泛化、跨 embodiment 共享的中间决策接口。**

最能支撑这个结论的，不是单一 benchmark，而是两类证据同时成立：
1. **VABench + 消融**：说明机制真的学会了更好的 visual aids generation  
2. **真实机器人零样本执行**：说明这些 visual aids 确实足够“有用”，而不只是评测友好

### 局限性

- **Fails when**: 指令很长、很模糊、需要显式子任务分解时；或任务涉及连续接触、关节物体、大幅遮挡、强动态环境时，2D visual aids 到实际控制的误差会被放大。
- **Assumes**: 依赖高质量自动标注管线与外部模块，包括 GPT-4o 生成部分训练数据/思维链、GroundedSAM/Metric3Dv2/CoTracker 等感知工具；执行时依赖深度相机、GraspNet、CuRobo；训练使用 8×A100 40G，且方法建立在 ASMv2 + CLIP-ViT-L + Vicuna-13B 这类骨干之上。
- **Not designed for**: 端到端闭环低层控制、精确3D力控制、长时序复杂计划的完整求解；当前重点是生成高层空间决策信号，而不是替代完整机器人控制栈。

### 可复用组件

- **空间关系图锚定的视觉 CoT**：适合任何“从图像生成可执行空间结构”的任务，不限于 manipulation。
- **生成/理解双向自一致**：适合所有坐标、框、关键点类输出任务，用来给离散数字 token 注入语义。
- **对象中心 visual trace 接口**：可作为闭环 policy、规划器或 VLA 的上游高层指导信号。
- **VABench**：可作为 visual aids generation 的专门评测集，补齐现有 manipulation benchmark 对中间表示评估的空白。

![[paperPDFs/Other_Multimodal_Large_Language_Model_MLLM_based_related_Embodied_Learning_2025/arXiv_2025/2025_From_Seeing_to_Doing_Bridging_Reasoning_and_Decision_for_Robotic_Manipulation.pdf]]