---
title: "Human2Robot: Learning Robot Actions from Paired Human-Robot Videos"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/robot-manipulation
  - task/imitation-learning
  - diffusion
  - dataset/H&R
  - opensource/no
core_operator: 将人类到机器人的跨 embodiment 对齐改写为“人视频条件下的机器人视频预测”，再从单步去噪特征解码动作。
primary_logic: |
  精确同步的人类视频 + 机器人初始观测 → 条件视频预测模型学习帧级人-机器人动态对应并形成机器人运动表征 → 冻结该表征后由动作解码器输出机器人动作序列
claims:
  - "在作者的基础任务评测中，HUMAN2ROBOT 的平均成功率达到 95%，高于 VPP 的 80%、XSkill 的 53% 和 DP 的 28% [evidence: comparison]"
  - "移除视频生成预训练后平均成功率降至 10%，直接用人视频驱动动作解码仅 23%，说明 VPM 预训练与预测表征是性能关键 [evidence: ablation]"
  - "在外观、位置、实例、背景、任务组合和 brand-new 六类泛化设置中，HUMAN2ROBOT 仍取得 50%-100% 成功率，而 XSkill/VPP 在多数设置下为 0 [evidence: comparison]"
related_work_position:
  extends: "Video Prediction Policy (Hu et al. 2024)"
  competes_with: "XSkill (Xu et al. 2023b); Video Prediction Policy (Hu et al. 2024)"
  complementary_to: "N/A"
evidence_strength: moderate
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Human2Robot_Learning_Robot_Actions_from_Paired_Human_Robot_Videos.pdf
category: Embodied_AI
---

# Human2Robot: Learning Robot Actions from Paired Human-Robot Videos

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2502.16587)
> - **Summary**: 论文通过 VR 遥操作采集精确同步的人-机器人视频对，并把“从人类演示学机器人动作”改写成“根据人类视频生成机器人视频”，再用生成模型的预测表征驱动策略学习，从而提升了跨任务泛化。
> - **Key Performance**: 基础任务平均成功率 95%；六类泛化测试中最高 100%，在 brand-new task 上仍有 70%

> [!info] **Agent Summary**
> - **task_path**: 第三人称人类演示视频 + 机器人初始观测 / 当前场景 -> 机器人动作序列 -> 真实机械臂执行
> - **bottleneck**: 现有方法多用粗对齐视频和全局表征匹配，只学到任务级语义，学不到抓取/放置所需的帧级时序对应
> - **mechanism_delta**: 用条件机器人视频预测替代全局表征对齐，并把单步去噪的生成特征作为动作先验输入策略头
> - **evidence_signal**: 主结果对比 + 关键消融最强：完整模型 95%，无预训练 10%，直接人视频解码 23%
> - **reusable_ops**: [paired-video-teleoperation, one-step-denoise-as-action-prior]
> - **failure_modes**: [复杂 embodiment gap 任务如 screwing 未覆盖, 依赖视角一致且精确同步的配对数据]
> - **open_questions**: [能否扩展到双手灵巧操作与更复杂接触任务, 去掉精确配对监督后是否还能保持同等泛化]

## Part I：问题与挑战

这篇工作的核心问题不是“机器人能不能看懂人类视频”，而是：

**机器人如何从人类演示里学到可执行、可泛化、且细粒度的动作对应关系？**

### 1. 真正的瓶颈是什么？

作者认为，过去人类到机器人学习的主要问题不在模型容量，而在**监督信号过粗**：

- 现有数据多是**粗对齐的人-机器人视频对**
- 相应方法也多做**全局特征匹配**、任务级对齐或自监督表征学习
- 结果是模型更容易学到“这是个 pick-and-place 任务”这种**任务标签式语义**
- 却学不到“什么时候靠近、何时夹取、怎样抬升、如何对齐放下”这种**帧级动态对应**

对机器人操作而言，后者才是决定成功与泛化的关键信息。

### 2. 为什么这个问题现在值得解决？

因为两件事同时成熟了：

1. **视频生成/视频扩散模型**已经足够强，能逼迫模型保留时序与动作细节，而不只是压缩成全局语义向量。
2. **VR 遥操作采集**让精确同步的人-机器人 paired video 变得可行，至少在单臂基础操作上可规模化收集。

作者把这看成一个“打破坏循环”的时机：

- 没有细粒度数据 → 方法只能做粗粒度对齐
- 方法只做粗粒度对齐 → 社区也缺少动力去采细粒度数据

Human2Robot 同时改数据和方法，试图打断这个循环。

### 3. 输入/输出接口与边界条件

**输入**：
- 人类第三人称演示视频
- 机器人初始观测/首帧
- 或在 seen task 下，用 KNN 从训练集检索一个最匹配的人类演示作为条件

**输出**：
- 机器人动作序列
- 最终驱动真实机械臂执行

**边界条件**：
- 主要聚焦**单臂 pick-and-place / push-pull / 简单旋转类操作**
- 数据来自**固定视角、精确同步、同场景风格**的人-机器人视频对
- 作者明确承认：**复杂 embodiment gap**（如拧螺丝）当前遥操作方案还不适合

---

## Part II：方法与洞察

整体方法可以概括成一句话：

> 不再问“人视频和机器人视频是否相似”，而是直接要求模型“根据人视频生成对应的机器人视频”，再把这个预测过程里的动态特征拿来指导动作生成。

### 方法主线

#### A. 先造出细粒度配对数据：H&R

作者先构建了 **H&R** 数据集：

- 2,600 条 episode
- 每条包含精确同步的人手视频和机械臂视频
- 来自 VR 遥操作采集
- 通过三锚点坐标对齐，把人的动作幅度与机器人的运动范围尽量映射一致

这一步的作用不是单纯“多一个数据集”，而是给后续模型提供**帧级可学习的监督**。

#### B. Stage 1：Video Prediction Model, VPM

第一阶段训练一个 **人视频条件下的机器人视频预测模型**。

其功能不是最终直接出动作，而是先学会：
- 人类手部动作的**位置线索**
- 人类动作随时间变化的**运动线索**
- 这些线索如何映射到**机器人未来视觉状态**

VPM 大致由三部分组成：

- **Behavior Extractor**：从人类图像/视频提取位置与运动 clues
- **Spatial UNet**：从机器人首帧抽取空间参考特征
- **Spatial-Temporal UNet**：结合人视频条件与机器人参考，显式建模时间动态，预测未来机器人视频

其底座初始化来自 Stable Diffusion，这让它从一开始就有较强的视觉生成先验。

#### C. Stage 2：冻结 VPM，当作动作先验编码器

作者没有在测试时真的完整生成整段机器人视频再控制机械臂，因为那样太慢。

他们采用的关键 trick 是：

- 给 VPM 输入加噪后的条件
- **只取第一次去噪后的中间特征**
- 尤其使用**第一层上采样层输出**
- 把它当作“预测性的机器人动态表征”

然后再接一个动作解码器：

- **Video Former**：把视频特征汇聚成固定长度 token
- **Diffusion Policy**：基于这些 token 生成机器人动作序列

这样做的含义是：  
模型先在生成任务里被迫学会“机器人接下来会怎么动”，再把这种隐式动态知识转给策略学习。

#### D. KNN + HUMAN2ROBOT

对于**已见过的任务**，作者又加了一个很工程化但实用的组件：

- 用 DINOv2 + CLIP 对当前场景做特征检索
- 从训练集中找最像的 episode
- 把对应的人类演示拿来当条件

这样即便测试时没有显式人类演示，也能执行 seen task。

### 核心直觉

**What changed**  
从“粗粒度的人-机器人表征对齐”改成“细粒度的机器人未来视频预测”。

**Which bottleneck changed**  
这改变了模型的信息瓶颈：  
以前只要保留“任务是什么”就够了；  
现在若想预测未来机器人视频，模型必须保留：
- 抓取物体的位置变化
- 末端执行器的接近/离开轨迹
- 动作顺序与时间因果关系

**What capability changed**  
因此，得到的不是一个任务标签式 embedding，而是一个更接近**动作先验**的视觉动态表征，进而提升：
- seen task 执行稳定性
- 对新位置/新外观/新实例的鲁棒性
- 甚至对新任务组合与 brand-new writing 的一次性泛化

### 为什么这个设计在因果上有效？

关键因果链是：

1. **同步 paired data 降低跨 embodiment 对齐噪声**  
   没有精确同步，人手和机械臂动作之间的对应关系本身就含糊。

2. **视频预测目标迫使模型保留动作细节**  
   仅做对比学习时，模型可以靠全局语义“投机取巧”；  
   但预测未来机器人帧时，若不理解细粒度动作，根本无法生成正确结果。

3. **单步去噪特征比直接从人视频解动作更接近机器人控制域**  
   这是本文和“直接人视频 -> 动作”方案的本质差别。  
   它先把输入投影到“机器人会如何运动”的表示空间，再做动作解码。

### 战略权衡

| 设计选择 | 解决的瓶颈 | 带来的能力 | 代价/约束 |
|---|---|---|---|
| 精确同步的人-机器人 paired video | 人-机器人监督信号过粗 | 学到帧级对应关系 | 采集成本高，任务范围受遥操作能力限制 |
| 条件机器人视频预测替代表征匹配 | 全局 embedding 丢失时序细节 | 更强的动作相关表征 | 训练更重，依赖生成模型底座 |
| 冻结 VPM，仅取单步去噪特征 | 全视频生成推理过慢 | 保留动态先验同时提升推理效率 | 表征无法端到端随策略继续适配 |
| KNN 检索人类演示 | seen task 测试时没有人类视频 | 无需显式演示也能执行已知任务 | 对 unseen task 帮助有限，性能低于完整条件输入 |

---

## Part III：证据与局限

### 关键证据信号

#### 1. 对比信号：完整方法在基础任务上明显优于基线
最强主结果来自基础任务平均成功率：

- **HUMAN2ROBOT：95%**
- VPP：80%
- XSkill：53%
- Diffusion Policy：28%

这说明它不仅优于纯策略学习，也优于：
- 只做人视频表征对齐的方法
- 已经用了视频预测预训练、但主要是语言条件的方法

**结论**：  
能力提升不只是“加了 diffusion 就更强”，而是**paired human-video conditioning + predictive representation** 这条链路确实更适合跨 embodiment 操作学习。

#### 2. 消融信号：性能增益确实来自 VPM 预训练与中间表征
两个消融尤其关键：

- **Action Decoder w. Human：23%**  
  直接把人视频送进动作解码器，执行会抖、抓取不稳。
- **HUMAN2ROBOT w/o. Pretrain：10%**  
  没有视频生成预训练，几乎学不会任务。

**结论**：  
真正起作用的不是“多了一个大模型编码器”，而是**先通过机器人视频预测把人视频变成机器人动态先验**。

#### 3. 泛化信号：能力跃迁主要体现在新分布
在 6 类泛化测试中：

- Appearance：100%
- Position：80%
- Instance：70%
- Background：80%
- Combination：50%
- Brand-New：70%

而 XSkill / VPP 在大多数更强泛化设置中接近 0。

**最值得关注的点**不是某一个数字，而是它在：
- 新实例
- 新背景
- 任务组合
- 全新 writing character

这些更强 distribution shift 下仍然保持非零且较高成功率。

#### 4. 质性信号：单步去噪已经包含足够动作信息
作者展示了：
- 1-step denoise 的结果就已经有明显动作与位姿趋势
- 完整 30-step 去噪生成的视频接近 GT robot video

这支持了第二阶段设计：  
**不必真的生成完整视频，也能抽出足够强的控制相关特征。**

### 能力跳跃到底体现在哪？

相对 prior work，这篇论文的能力跳跃主要不在“seen task 再提几个点”，而在：

1. **从任务标签式理解，转向动作过程式理解**
2. **从只会训练内任务，转向能跟着新的人类演示做新变体**
3. **从“需要明确演示”扩展到 seen task 可 KNN 检索执行**

也就是说，它把“看懂人类在做什么”进一步推进成了“推断机器人应该怎样逐步做”。

### 局限性

- **Fails when**: 需要复杂手指接触、强工具操作或显著 embodiment gap 的任务（如 screwing、灵巧手操作）时，这套数据采集与策略学习链路并未被验证；在大幅偏离训练视角/场景结构的条件下，成功率也可能明显下降。
- **Assumes**: 依赖精确同步的人-机器人 paired video、固定第三人称视觉设置、VR 遥操作采集流程，以及 Stable Diffusion 初始化；训练资源要求不低（Stage 1 约 3 天、4×A100；Stage 2 约 6 小时、8×A100）；硬件上依赖 Meta Quest 3、xArm 7-DoF 与 Realsense D435。
- **Not designed for**: 无配对互联网人类视频直接迁移、双手灵巧操作、开放世界长时规划、多机器人协同、以及不提供或无法检索任务示范的完全开放式测试。

### 可复用组件

这篇工作最值得复用的不是某个具体网络块，而是三类操作模板：

1. **paired-video teleoperation pipeline**  
   用锚点对齐把人和机器人的运动映射到更一致的视觉空间。

2. **video-prediction-as-alignment**  
   把跨 embodiment 对齐问题改写为条件视频预测，而不是全局特征匹配。

3. **one-step denoise feature for policy learning**  
   不完整生成视频，只取生成模型早期去噪特征作为动作先验，兼顾信息量与效率。

### 总结判断

这是一个很典型、也很清晰的**“数据范式 + 学习目标”一起改**的 embodied AI 方法论文。

它最重要的启发是：

> 如果你希望机器人真正从人类演示中学到“怎么做”，监督目标就不能只是“人和机器人视频在语义上相似”，而必须迫使模型显式保留未来动作过程。

证据上，它在作者自建数据和真实机器人实验中表现亮眼，也有关键消融支撑；但由于评测基本集中在单一自建平台和有限任务族上，整体证据强度应保守地看作 **moderate**，而不是更高。

## Local PDF reference

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Human2Robot_Learning_Robot_Actions_from_Paired_Human_Robot_Videos.pdf]]