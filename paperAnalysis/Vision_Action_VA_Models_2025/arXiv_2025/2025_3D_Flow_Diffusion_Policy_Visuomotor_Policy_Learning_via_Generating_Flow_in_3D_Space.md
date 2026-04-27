---
title: "3D Flow Diffusion Policy: Visuomotor Policy Learning via Generating Flow in 3D Space"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/robot-manipulation
  - diffusion
  - scene-flow
  - dataset/MetaWorld
  - opensource/partial
core_operator: "先预测场景级稀疏查询点的未来3D流，再将其压缩为动作条件，驱动扩散策略生成低层控制序列。"
primary_logic: |
  单视角RGB-D点云与机器人状态历史 → 采样场景查询点并预测其未来3D位移轨迹 → 将流编码为计划级条件向量 → 条件化扩散生成未来动作序列
claims:
  - "在 MetaWorld 50 个任务上，3D FDP 在 easy、medium、hard、very hard 四个难度分组的平均成功率均高于 DP3 和 MBA，其中 medium/hard 分别达到 55.3%/55.6% [evidence: comparison]"
  - "在 8 个真实机器人任务上，3D FDP 将平均成功率从 DP3 的 27.5% 提升到 56.9%，并在 Hang、Shelve、Non-prehensile 等 DP3 为 0% 的任务上取得非零成功率 [evidence: comparison]"
  - "消融实验显示，更多查询点和场景级而非仅物体级的采样会提升性能：MetaWorld hard 集平均成功率从 50 个点时的 52.2% 升至 200 个点时的 60.8% [evidence: ablation]"
related_work_position:
  extends: "3D Diffusion Policy (Ze et al. 2024)"
  competes_with: "3D Diffusion Policy (Ze et al. 2024); Motion Before Action (Su et al. 2025)"
  complementary_to: "Hierarchical Diffusion Policy (Ma et al. 2024); DINOv2 (Oquab et al. 2023)"
evidence_strength: strong
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_3D_Flow_Diffusion_Policy_Visuomotor_Policy_Learning_via_Generating_Flow_in_3D_Space.pdf
category: Embodied_AI
---

# 3D Flow Diffusion Policy: Visuomotor Policy Learning via Generating Flow in 3D Space

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2509.18676), [Project](https://sites.google.com/view/3d-fdp)
> - **Summary**: 论文把机器人策略学习从“直接由观测出动作”改成“先预测场景中稀疏查询点的未来 3D 流，再依据这份交互动力学摘要生成动作”，从而提升接触丰富、非抓取和场景耦合任务的泛化能力。
> - **Key Performance**: MetaWorld 50 任务四个难度分组均领先，medium/hard 达到 **55.3% / 55.6%**；真实机器人 8 任务平均成功率 **56.9% vs 27.5% (DP3)**。

> [!info] **Agent Summary**
> - **task_path**: 单视角 RGB-D 点云 + 机器人状态历史 -> 未来查询点 3D 流 -> 低层连续机械臂动作序列
> - **bottleneck**: 直接 observation-to-action 会压缩掉局部接触与场景连锁运动信息，导致在接触丰富和复杂动力学任务上泛化不足
> - **mechanism_delta**: 用场景级稀疏 3D 流替代全局特征或物体位姿作为中间表示，并将“流预测 + 动作生成”放进统一扩散框架联合学习
> - **evidence_signal**: MetaWorld 50 任务跨难度稳定领先，并在真实机器人 8 任务上将平均成功率从 27.5% 提升到 56.9%
> - **reusable_ops**: [scene-level-query-sampling, flow-conditioned-action-diffusion]
> - **failure_modes**: [severe-occlusion-or-missing-depth, large-initial-state-shift]
> - **open_questions**: [online-flow-supervision-without-offline-tracking, language-or-semantic-fusion-with-scene-flow]

## Part I：问题与挑战

这篇论文要解决的不是“动作生成器不够强”，而是**策略输入里缺少对局部交互动力学的显式表示**。

### 1. 真问题是什么？
在机器人操作中，很多关键决策依赖于非常局部、但因果性很强的运动线索，例如：

- 夹爪是否真正接触到物体；
- 推一下后，物体会不会转动而不是平移；
- 插入、压合、挂接时，周围环境是否也在发生受力响应；
- 杂乱环境中，一个动作会不会扰动邻近物体。

现有很多 visuomotor policy 虽然已经从图像/点云走向 transformer 或 diffusion，但主流范式仍是：

1. 把视觉与本体状态压成一个全局 latent；
2. 直接生成动作。

问题在于，这种压缩常常会把**局部接触、运动传播、场景响应**这些对 manipulation 很关键的信息抹平。  
而另一类中间表示方法，如物体位姿预测，又常常太“物体中心”，难以表达**场景级后果**；至于未来帧生成，信息丰富但代价偏高，不适合高效策略学习。

### 2. 这篇论文的输入/输出接口
论文的接口很明确：

- **输入**：过去一小段历史观测  
  - 单个固定 RGB-D 相机得到的点云  
  - 机器人本体状态
- **中间变量**：从初始场景中采样的一组查询点的未来 3D 位移轨迹
- **输出**：未来一段低层动作序列，执行时采用 receding horizon，只落地第一个动作

所以它不是直接学 `观测 -> 动作`，而是学：

`观测历史 -> 场景查询点未来 3D 流 -> 动作`

### 3. 真正的瓶颈在哪里？
真正瓶颈是：**策略缺少一个既保留局部交互，又足够紧凑、可条件化到控制器的动态表示。**

作者认为 3D flow 正好卡在这个甜点上：

- 比全局 latent 更局部、更物理；
- 比物体 pose 更场景化；
- 比生成整帧未来更轻量。

### 4. 为什么是现在解决？
因为几项技术条件刚好成熟：

- **扩散策略**已经证明适合多峰动作分布；
- **点云策略**（如 DP3）让 3D 几何输入更实用；
- **现代点跟踪器**（如 CoTracker）使真实数据里的查询点轨迹监督变得可行，不再必须依赖 mocap 或复杂 6D pose 重建。

### 5. 边界条件
这篇方法并不是开放世界通用操控，而是在如下前提下成立：

- 单视角、固定、已标定 RGB-D 相机；
- 主要依赖几何，不显式用大规模语义视觉特征；
- imitation learning / demonstration learning 设定；
- 短时域动作与流预测；
- 真实世界的 3D flow 监督依赖 2D 跟踪 + 深度回投。

---

## Part II：方法与洞察

方法核心可以概括成一句话：**先让模型“想象”场景中哪些点会怎么动，再让动作生成器据此出手。**

### 1. 方法骨架

#### (1) 场景表示与查询点
- 用单视角 RGB-D 构建点云；
- 去除静态背景后，采样固定数量点；
- 用 PointNet 风格编码器提取：
  - 全局场景特征；
  - 点级局部特征。
- 再从第一帧点云中用 FPS 采样 `M` 个查询点，作为后续流预测的锚点。

这里的关键不是做“稠密 scene flow”，而是做**稀疏但覆盖场景的 query-point flow**。

#### (2) 3D 流监督如何获得
- **仿真中**：利用已知 mesh pose，把初始查询点绑定到物体 mesh 顶点，再随时间变换得到 3D 轨迹。
- **真实中**：
  1. 从第一帧深度图得到 3D 查询点；
  2. 投影到 RGB 上得到 2D 起点；
  3. 用 CoTracker 追踪 2D 轨迹；
  4. 用每帧深度把 2D 轨迹重新 lift 回 3D；
  5. 用 overlapping temporal chunks 降低长序列遮挡累计误差。

这一步很重要，因为它把真实世界 flow 标注从“昂贵硬件/重建管线”降成了“跟踪 + 深度回投”。

#### (3) 第一阶段：Flow Generator
第一个扩散模块负责预测未来查询点轨迹。

它的条件输入不是只有场景全局特征，而是：
- 查询点对应的局部特征；
- 全局场景特征；

两者组合后，得到更适合“点会往哪动”的条件表示。  
直观上，这一步在建模：

- 夹爪附近的局部受力关系；
- 物体与周围环境的相互作用；
- 场景中哪些区域可能被连带影响。

#### (4) 第二阶段：Action Generator
第二个扩散模块不直接吃原始观测，而是吃：

- 观测历史的全局特征；
- 预测出的 flow 序列经时间卷积压缩后的“计划级嵌入”。

也就是说，动作生成器拿到的不只是“现在看到了什么”，而是“接下来场景会怎么动”的结构化摘要。

---

### 核心直觉

**改变了什么？**  
从“直接把观测压缩成 latent 后出动作”，改成“先显式预测局部未来运动，再依据这份未来运动摘要出动作”。

**哪个瓶颈被改变了？**  
原先的瓶颈是：**信息在 observation-to-action 压缩中丢失了局部接触与因果后果。**  
现在变成：**把未来交互动态作为可解释、可条件化的中间变量显式保留下来。**

**能力为什么会变？**
因为 manipulation 的关键难点往往不是“识别物体是什么”，而是“动作施加后，场景会如何响应”。  
3D FDP 把这个响应先建模出来，等于把控制问题拆成两步：

1. **未来交互建模**：哪些点会如何移动；
2. **控制落地**：为了得到这种移动，机器人该怎么动。

这样做带来三个因果层面的好处：

1. **局部接触被显式化**  
   查询点轨迹比全局特征更容易承载“接触是否建立、受力后往哪走”的信息。

2. **场景连锁效应被纳入策略条件**  
   不是只看被操作物体，而是把周围环境也纳入采样与预测，因此更适合 clutter、插入、挂接、非抓取旋转等任务。

3. **动作生成负担下降**  
   动作扩散器不必同时从原始观测里推断“将发生什么”和“该怎么控制”，而是只需在一个已经包含动态摘要的条件上生成动作。

换句话说：

**观测历史 → 未来流场摘要 → 动作**  
这一步把策略从“纯反应式映射”推向了“带结构先验的动态条件控制”。

### 2. 为什么 scene-level 而不是 object-level？
这是论文最关键的判断之一。  
如果只跟踪物体表面点，模型会看到“锤子在动”，但未必看到“钉子被推进去了”；  
如果查询点覆盖整个场景，模型就能把“动作后果”也纳入计划。

这解释了为什么在 Hammer、Shelve、Non-prehensile 等任务上，scene-level flow 比 object-only 表示更有效。

### 3. 策略性取舍

| 设计选择 | 得到的能力 | 代价 / 风险 |
|---|---|---|
| 场景级稀疏查询点流，而非直接动作回归 | 显式建模局部交互与场景响应 | 需要额外流监督与查询点设计 |
| 两阶段扩散（flow -> action） | 把“未来动态推断”和“控制生成”解耦 | 第一阶段误差会传递到动作阶段 |
| 场景级采样，而非仅物体采样 | 能看见环境反馈与连锁效应 | 可能引入更多无关点，增加学习难度 |
| 仅用几何点云，不用大模型语义特征 | 训练部署更轻，真实系统更直接 | 语义泛化能力可能弱于语义增强方法 |
| CoTracker + 深度回投构造真实 3D 流 | 避开 mocap/复杂 pose 管线，真实数据更可扩展 | 受遮挡、深度噪声、相机标定误差影响 |

---

## Part III：证据与局限

### 1. 关键证据信号

#### 信号 A：跨 50 个 MetaWorld 任务的一致领先
最强的模拟证据不是某个单点 SOTA，而是**跨难度分组都稳定领先**：

- easy: 86.5
- medium: 55.3
- hard: 55.6
- very hard: 47.9

而对比基线：
- **DP3** 代表无中间表示的直接 3D diffusion policy；
- **MBA** 代表用物体级 motion/pose 作为中间表示。

结论是：**把中间表示从“没有”或“物体位姿”换成“场景级 3D 流”，收益主要体现在更复杂、更依赖接触与动态推理的任务上。**

#### 信号 B：真实机器人上出现“能力跳变”
真实机器人 8 个任务中，平均成功率从 **27.5% 提升到 56.9%**，这是最重要的“so what”证据。

尤其关键的是几个差异最大的任务：

- **Hang**：DP3 为 0%，3D FDP 为 30%
- **Shelve**：DP3 为 0%，3D FDP 为 20%
- **Non-prehensile**：DP3 为 0%，3D FDP 为 35%
- **Press**：25% -> 100%

这些任务共同特点不是“视觉更难”，而是**动作后果更依赖接触几何与间接运动**。这恰好对应论文声称解决的瓶颈。

#### 信号 C：消融真正支持了机制，而不只是堆参数
论文做的消融是有机制指向的：

- **更多 query points 更好**：hard 集平均成功率从 52.2 提升到 60.8  
  说明更丰富的局部动态线索确实在帮助控制，而不只是随机收益。
- **scene-level 采样优于 object-only 采样**：例如 Hammer 明显提升  
  说明模型真正利用了“环境反馈”，而不是仅仅跟踪被操纵物体。

#### 信号 D：学习效率也更高
学习曲线显示 3D FDP 比 DP3：
- 收敛更快；
- 峰值更高。

这意味着流中间表示不仅提升最终上限，也降低了策略从示范中抽取交互规律的难度。

### 2. 1-2 个最值得记住的指标
- **真实机器人平均成功率**：56.9% vs 27.5%  
- **MetaWorld hard**：55.6% vs DP3 的 52.4%、MBA 的 53.0%

### 3. 局限性

- **Fails when**: 单视角深度缺失、重遮挡、透明/反光物体或点跟踪漂移会直接破坏 3D 流监督与推理；当初始姿态、物体动力学或接触模式显著超出训练分布时，预测流可能失真并连带误导动作生成。
- **Assumes**: 固定且已标定的 RGB-D 相机；仿真中可获得 mesh pose，真实中可用 CoTracker 与深度回投构造 3D flow；示范学习数据可得；短时域预测足以支持控制；点云几何信息能够覆盖任务关键状态。
- **Not designed for**: 纯语言条件操作、开放世界语义泛化、多视角/移动相机设置、长时程分层任务规划，或完全无监督的在线流学习。

### 4. 复现与扩展时的真实依赖
这篇工作虽然网络训练本身不算重（单张 RTX 3090、约 6 小时），但真实系统复现并不只是“跑模型”：

- 需要稳定的 RGB-D 采集与相机标定；
- 需要离线 2D 跟踪与深度回投管线；
- 真实评测成功率依赖人工判定；
- 真实训练数据仍需遥操作采集。

所以它的计算成本不高，但**数据准备与标注流水线成本**并不低。

### 5. 可复用组件
以下组件有较强迁移价值：

- **scene-level query point sampling**：把操作对象与环境一起纳入动态表示；
- **2D tracking + depth lifting 的 3D flow 标注管线**：适合无 mocap 的真实机器人数据；
- **flow-conditioned action diffusion**：适合作为“中间计划变量 -> 控制器”的通用结构模板。

**一句话总结**：  
这篇论文真正有价值的不是“又一个 diffusion policy”，而是证明了**场景级 3D 流可以作为机器人操控里有效、可学习、可部署的结构中间表示**；它把策略的改进点放在了“未来交互动态的显式建模”上，因此在接触丰富和复杂场景耦合任务中出现了比直接行为克隆更明显的能力跃迁。

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_3D_Flow_Diffusion_Policy_Visuomotor_Policy_Learning_via_Generating_Flow_in_3D_Space.pdf]]