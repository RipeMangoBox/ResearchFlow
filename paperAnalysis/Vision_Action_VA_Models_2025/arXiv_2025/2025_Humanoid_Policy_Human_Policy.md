---
title: "Humanoid Policy ~ Human Policy"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/robot-manipulation
  - action-chunking
  - retargeting
  - state-unification
  - dataset/PH2D
  - opensource/no
core_operator: 以统一的人类中心状态-动作空间联合建模人类与类人机器人，再把预测的人手腕与指尖轨迹通过逆运动学和手部重定向转换为机器人动作。
primary_logic: |
  自我中心图像与头/腕/手指状态（来自人类或类人机器人） → 在统一状态-动作空间中联合训练 Human Action Transformer，并用动作降速、姿态约束和基础视觉增强缩小 embodiment gap → 预测未来手腕/指尖轨迹并经逆运动学与手部重定向输出机器人控制
claims:
  - "在 4 个真实机器人操作任务上，共训人类数据的 HAT 将 Humanoid A 的 O.O.D. 总成功数从 59/170 提升到 101/170，而 I.D. 成绩仅小幅变化（42/60→49/60） [evidence: comparison]"
  - "统一状态-动作空间与人类动作降速都是必要的：在垂直抓取 O.O.D. 设置中，去掉统一状态或不对人类动作做插值降速时，成功数分别降至 0/10 和 1/10，而完整设计为 4/10 [evidence: ablation]"
  - "在异构平台 Humanoid B 的 few-shot 适配中，仅用 20 条 B 平台演示时，加入 Humanoid A 与 human 数据的共训策略在所有测试对象上都优于只用 Humanoid B 数据训练 [evidence: comparison]"
related_work_position:
  extends: "ACT (Zhao et al. 2023)"
  competes_with: "EgoMimic (Kareer et al. 2024); HumanPlus (Fu et al. 2024)"
  complementary_to: "π0 (Black et al. 2024); RDT-1B (Liu et al. 2024)"
evidence_strength: moderate
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Humanoid_Policy_Human_Policy.pdf
category: Embodied_AI
---

# Humanoid Policy ~ Human Policy

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2503.13441), [Project](https://human-as-robot.github.io/)
> - **Summary**: 这篇工作把“任务导向的第一视角人类演示”当作另一种 humanoid embodiment 来直接共训策略，用统一的人类中心动作空间而非中间感知代理，从而显著提升类人机器人操作在 OOD 场景下的泛化与鲁棒性。
> - **Key Performance**: Humanoid A 四任务 O.O.D. 总成功数从 **59/170** 提升到 **101/170**；杯子传递背景泛化从 **55/80** 提升到 **72/80**。

> [!info] **Agent Summary**
> - **task_path**: 第一视角 RGB + 头部/手腕/指尖状态（human 或 humanoid） -> 双臂类人机器人未来动作块
> - **bottleneck**: 机器人遥操作演示昂贵且难扩展，而 human/humanoid 在动作速度、身体运动、视觉传感器和末端形态上的差异会让直接共训不稳定
> - **mechanism_delta**: 用统一的人类中心状态-动作空间把 human 与 humanoid 视作两个 embodiment 共同建模，并通过动作降速与部署时 retargeting 吸收时序和执行差异
> - **evidence_signal**: 4 个真实机器人任务上，加入 human 数据后 O.O.D. 总成功数 59/170 -> 101/170
> - **reusable_ops**: [统一人类中心状态动作空间, 人类动作插值降速]
> - **failure_modes**: [VR 手关键点在重遮挡下会失真, 人类自然 whole-body movement 会把不可执行动作模式带进训练分布]
> - **open_questions**: [能否扩展到非五指灵巧手或更异构机器人形态, PH2D 的语言标注能否支撑语言条件跨 embodiment 策略]

## Part I：问题与挑战

这篇论文要解决的核心问题不是“再收一点机器人数据”，而是：

**类人机器人操作策略的数据瓶颈，是否可以用大规模、高质量、任务导向的人类第一视角演示来打破？**

### 1. 真正的瓶颈是什么

现有 humanoid manipulation 很依赖真实机器人遥操作数据，但这类数据有三个结构性问题：

1. **贵**：需要机器人本体、遥操作系统、场地和人工监督。
2. **慢**：机器人执行和示教速度显著慢于人类。
3. **难扩展**：换场景、换物体、换平台都要重新布置和采集。

论文的判断很直接：  
如果 humanoid teleoperation 本质上就是“把人的动作映射到机器人上”，那么与其只收机器人数据，不如直接把**人类动作视为另一种机器人 embodiment**，把人类示教当作可扩展数据源。

### 2. 为什么现在值得做

作者认为现在具备了两个关键条件：

- **消费级 VR 设备**已经能稳定提供头部 pose 和手部关键点追踪；
- **视觉基础模型**（如 DINOv2）让策略对摄像头差异、颜色变化等有更强鲁棒性，降低了 human/robot 视觉域差异的难度。

因此，过去“human video 只能做中间表征监督”的局面，开始可以被更直接的 end-to-end 策略学习替代。

### 3. 输入/输出接口

**输入：**

- 第一视角图像
- 头部 pose
- 左右手腕 6DoF pose
- 手指关键点/指尖位置

**输出：**

- 未来一段动作块（action chunk）
- 在训练中输出的是**人类中心动作表示**
- 在部署时再通过**逆运动学 + 手部重定向**变成机器人关节/手部控制

### 4. 这项工作的边界条件

它不是“任意 human video -> 任意 robot policy”的设定，而是更受控的条件：

- 人类数据必须是**任务导向**的，而不是日常无目标行为；
- 人类任务要与机器人执行任务有明显重叠；
- 主要面向**双臂、带头部、带灵巧手**的 humanoid manipulation；
- 当前评测集中在带五指灵巧手的平台上，手指映射较自然。

换句话说，这篇论文解决的是：

> **如何在“任务对齐、传感对齐、动作可重定向”的前提下，把人类第一视角演示变成 humanoid policy 的有效训练数据。**

---

## Part II：方法与洞察

方法由两个互相配合的部分组成：

1. **PH2D 数据集**：收大规模、任务导向的人类第一视角操作数据  
2. **HAT 策略**：用统一状态-动作空间联合建模 human 和 humanoid

### 1. 数据侧：PH2D 不是普通 egocentric dataset

作者收集了 **PH2D (Physical Humanoid-Human Data)**：

- human：约 **26.8k demos / 3.02M frames**
- robot：约 **1.55k demos / 668k frames**

它和一般第一视角人类数据集的区别在于：

- **任务导向**：直接对齐机器人操作任务，如抓取、传递、倒水
- **带 3D 手部/手腕/头部信息**：可直接作为 imitation supervision
- **设备成本低**：Apple Vision Pro、Meta Quest 3、ZED 等消费级设备
- **相机多样**：故意引入不同视觉传感器，增强泛化

这一步的关键不是“数据更多”，而是**数据分布被设计得更接近机器人可执行任务**。

### 2. 模型侧：HAT 如何把 human 和 humanoid 放进同一个学习问题里

HAT 本质上沿用了 ACT 风格的 **Transformer action-chunk policy**，但做了关键改造：

- 图像用冻结的 **DINOv2 ViT-S**
- 状态不是机器人关节空间，而是**统一的人类中心状态-动作空间**
- 模型同时吃 human data 和 robot data
- 预测结果始终落在这个统一空间里

统一空间包含：

- 头部姿态
- 左右手腕旋转与平移
- 双手指尖位置

这意味着模型学到的不是“Humanoid A 的关节控制习惯”，而是更接近于：

> **从第一视角视觉和手部状态出发，应该形成什么样的双手操作几何轨迹。**

部署到机器人时，再把这组“人类式手腕/手指目标”转回机器人控制。

### 3. 关键对齐策略

#### 3.1 统一状态-动作空间

这是全文最核心的设计。

如果 human 和 robot 用不同状态表示，模型很容易学到一个“快捷方式”：

- 一看输入编码形式，就知道当前样本来自 human 还是 robot；
- 然后分别记忆两个域，而不是学习共享任务结构。

统一表示后，这个 shortcut 被削弱，模型更被迫去学习：

- 手-物交互几何
- 双手配合规律
- 第一视角视觉与手部动作之间的关系

#### 3.2 人类动作降速

human demonstration 天然比 robot teleop 快得多。  
如果直接混训，策略输出的时序统计会在“快的人”和“慢的机器人”之间摇摆，导致执行不稳定。

所以作者把人类动作按固定因子 **α=4** 做插值降速。  
这不是小工程细节，而是**时间尺度对齐**。

#### 3.3 限制 whole-body movement

人类做操作时会自然晃肩、探身、挪腰。  
但当前 humanoid 很难精确复现这种全身动作。

因此作者在采集时要求操作者保持坐姿、尽量减少上身运动。  
本质上是在**过滤掉机器人目前不可承载的自由度**。

#### 3.4 不依赖复杂视觉伪装

与一些需要严格相机对齐、手部 mask 或生成式视觉转换的方法不同，本文发现：

- 当 human data 足够多样
- 视觉 backbone 足够强

只用**基础增强**（color jitter / blur）也能学到有效策略。

这使数据采集流程更可扩展，也更接近“先规模化，再统一建模”的路线。

### 核心直觉

**what changed**：  
作者把“人类示教”从传统的外部监督源，改成了和 humanoid 演示同地位的**另一种 embodiment 训练样本**。

**which bottleneck changed**：  
这改变的不是某个局部模型模块，而是整个训练分布：

- 从“小规模机器人专属分布”
- 变成“共享任务结构的大规模 human+robot 混合分布”

同时，统一状态空间和动作降速又把最危险的分布偏差压下去：

- **表示偏差**：不同编码让模型偷懒分域学习
- **时序偏差**：human 太快，robot 太慢
- **运动学偏差**：人类 whole-body movement 机器人学不了

**what capability changed**：  
于是策略提升的不是纯 ID 记忆能力，而是：

- 背景变化下的鲁棒性
- 物体外观变化下的抓取稳定性
- 未见摆放位置下的可泛化性
- 少量新平台数据下的快速适配能力

### 为什么这个设计有效

因果上，这个方法有效不是因为“人类数据更多”这么简单，而是因为：

1. **任务导向采集**保证 human data 和 robot task 真有重叠；
2. **统一状态-动作空间**迫使模型共享任务结构，而不是分开背两套策略；
3. **动作降速**让 human/robot 的控制时间尺度可兼容；
4. **部署时 retargeting**把高层共享策略与具体机器人机构学分离。

所以它不是“把 human video 硬塞给 robot model”，而是：

> **先把可共享的策略层抽象到人类中心动作空间，再把 embodiment-specific 的东西放到执行端处理。**

### 战略取舍表

| 设计选择 | 解决的瓶颈 | 带来的能力变化 | 代价 / 风险 |
|---|---|---|---|
| 统一人类中心状态-动作空间 | human/robot 表示不一致导致的分域 shortcut | 更强跨 embodiment 共享与 OOD 泛化 | 依赖可定义的末端/手指映射 |
| 人类动作插值降速 | human 与 robot 的时序统计差异 | 执行更稳定，减少输出速度抖动 | 可能抹掉部分真实人类动态优势 |
| 采集时约束 whole-body movement | 人类动作包含机器人难复现的自由度 | 提高训练信号可部署性 | 牺牲部分自然性与任务覆盖 |
| 多设备采集 + 简单视觉增强 | 传感器和外观域差异 | 不必严格相机对齐，也能提升鲁棒性 | 需要较强视觉 backbone 与足够数据量 |
| 部署时 IK + hand retargeting | 不同机构的执行差异 | 模型训练与机器人控制解耦 | 仍受机器人工作空间和运动学限制 |

---

## Part III：证据与局限

### 1. 关键实验信号

#### 信号 A：主结论不是“ID 更强”，而是“OOD 明显更强”
- **类型**：comparison
- **结论**：加入 human data 后，Humanoid A 在四个任务上的 **OOD 总成功数从 59/170 提升到 101/170**，而 ID 只从 **42/60 到 49/60**。
- **解释**：human data 的价值主要不是替代机器人数据做记忆，而是在机器人没见过、但人类见过的变化上提供视觉和动作先验。

#### 信号 B：泛化提升覆盖三类变化
- **类型**：comparison
- **结论**：
  - 背景泛化：cup passing 从 **55/80 -> 72/80**
  - 物体外观泛化：horizontal grasp 在多种新物体上整体更稳
  - 物体摆放泛化：vertical grasp 在未覆盖网格区域上成功率更高
- **解释**：模型不是只学会某个对象或某个桌面，而是更好地抽取了“第一视角下如何接近、对齐、抓住目标”的共享规律。

#### 信号 C：few-shot 跨平台适配更强
- **类型**：comparison
- **结论**：在 Humanoid B 上，即使只有 **20 条**本机示教，加入 Humanoid A 和 human 数据共训也明显优于只用 B 数据训练。
- **解释**：human data 不只是补充视觉多样性，也在提供一种更平台无关的任务结构先验。

#### 信号 D：作者做了关键因果消融
- **类型**：ablation
- **结论**：去掉统一状态空间或去掉人类动作降速，垂直抓取 OOD 成功数分别掉到 **0/10** 和 **1/10**，完整设计为 **4/10**。
- **解释**：说明改进不是“多加数据自然变好”，而是确实依赖于对 embodiment gap 的建模。

#### 信号 E：human data 的采样效率更高
- **类型**：comparison
- **结论**：在相同 20 分钟预算下，混合数据训练比纯机器人数据训练在垂直抓取网格评测上更好（**35/90 vs 28/90**）。
- **解释**：这支持论文的现实意义：human demos 不只是理论上更多，而是**单位时间更值钱**。

### 2. 1-2 个最值得记住的指标

- **O.O.D. 总成功数**：59/170 -> 101/170  
- **背景泛化（杯子传递）**：55/80 -> 72/80

### 3. 局限性

- **Fails when**: VR 手部关键点在重遮挡或复杂手部接触下跟踪失败时，监督会变噪；涉及明显 whole-body coordination、复杂在手操作或极端机器人工作空间限制时，human 动作未必能稳定重定向；对低矮且与背景相近的小物体（如文中 box2）仍会出现抓取不稳和滑落。
- **Assumes**: 需要任务导向且与机器人任务对齐的人类演示；需要消费级 VR 设备和对应 SDK 提供头手 tracking；需要一定量机器人示教做对齐与落地（文中每任务仍有约 250-400 条 robot demos）；依赖冻结视觉 backbone（DINOv2）和可实现的 IK / hand retargeting。
- **Not designed for**: 长时程语言条件操作、开放词汇指令泛化、非灵巧手平台的大规模验证、强 locomotion 参与的全身操作任务；也还没有证明能直接泛化到任意机器人形态。

### 4. 资源与复现依赖

这篇工作虽然强调“比机器人遥操作更便宜”，但仍有明显依赖：

- Apple Vision Pro / Meta Quest / ZED 等设备
- 现成 VR hand-tracking SDK 的精度与稳定性
- 真实 humanoid 平台和 teleop 数据作为锚点
- 论文给出 project，但文中**未明确完整开源代码/数据状态**，因此可复现性仍受限

### 5. 可复用组件

这篇论文最可复用的，不一定是完整 HAT，而是下面几个操作子：

- **统一的人类中心状态-动作空间**
- **human/robot 时序对齐的人类动作降速**
- **采集时主动抑制机器人不可复现的 whole-body movement**
- **把跨 embodiment 共享策略与执行端 retargeting 解耦**

如果以后要做更大的 VLA / diffusion / flow policy，这些设计都可以直接继承。

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Humanoid_Policy_Human_Policy.pdf]]