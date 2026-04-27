---
title: "FetchBot: Learning Generalizable Object Fetching in Cluttered Scenes via Zero-Shot Sim2Real"
venue: CoRL
year: 2025
tags:
  - Embodied_AI
  - task/object-fetching
  - diffusion
  - occupancy-prediction
  - voxel-representation
  - dataset/UniVoxGen
  - opensource/promised
core_operator: 用基础模型预测深度统一仿真与真实输入分布，再以局部语义占据补全遮挡区域，并将RL低扰动示范蒸馏到扩散策略中。
primary_logic: |
  多视角RGB与机器人状态 → 通过DepthAnything得到跨域一致的深度线索，并在ROI内做2D-3D融合与语义占据补全 → 冻结视觉编码器后模仿RL oracle示范，用扩散头输出低扰动取物动作
claims:
  - "在 3000 个高密度仿真场景中，FetchBot 的 Occupancy 策略在吸盘/平行夹爪上分别达到 81.46% / 91.02% 成功率，优于所有非 oracle 基线，并同时降低环境扰动 [evidence: comparison]"
  - "在真实世界零样本 sim2real 评测中，FetchBot 在吸盘/平行夹爪上分别达到 86.6% / 93.3% 成功率，平均为 89.95% [evidence: comparison]"
  - "在文中的真实世界消融设置下，去掉 DepthAnything 或 occupancy 分支后成功率从 86.60% 分别降至 60.00% 和 73.33%，仅用 RGB 时为 33.33% [evidence: ablation]"
related_work_position:
  extends: "Diffusion Policy (Chi et al. 2023)"
  competes_with: "3D Diffusion Policy (Ze et al. 2024); SafePicking (Wada et al. 2022)"
  complementary_to: "N/A"
evidence_strength: moderate
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_FetchBot_Object_Fetching_in_Cluttered_Shelves_via_Zero_Shot_Sim2Real.pdf
category: Embodied_AI
---

# FetchBot: Learning Generalizable Object Fetching in Cluttered Scenes via Zero-Shot Sim2Real

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2502.17894) · [Project](https://pku-epic.github.io/FetchBot/)
> - **Summary**: 论文提出一个零样本 sim2real 取物框架：先用大规模体素仿真和 RL oracle 学出低扰动示范，再用“基础模型预测深度 + 局部语义占据补全”学习可迁移的 3D 表征，最终在真实拥挤货架中稳定取出目标物体。
> - **Key Performance**: 真实世界平均成功率 89.95%；仿真中吸盘/平行夹爪成功率 81.46% / 91.02%。

> [!info] **Agent Summary**
> - **task_path**: 多视角 RGB + 已指定目标 / 拥挤货架取物 -> 末端执行器连续 SE(3) 动作与取出轨迹
> - **bottleneck**: 遮挡导致局部 3D 几何不完整，同时真实深度噪声使 sim2real 失稳，导致策略既看不全也迁不过去
> - **mechanism_delta**: 把“直接基于 RGB/传感器深度做策略”改为“预测深度桥接域差 + ROI 语义占据补全 + oracle 蒸馏扩散策略”
> - **evidence_signal**: 真实世界零样本对比结果 + 去除 predicted depth / occupancy 的显著消融降幅
> - **reusable_ops**: [foundation-depth-bridge, roi-semantic-occupancy-pretraining]
> - **failure_modes**: [joint-limit-violation-under-heavy-occlusion, large-object-single-arm-failure, fully-occluded-target-needs-rearrangement]
> - **open_questions**: [how-to-add-multi-step-rearrangement-reasoning, how-to-extend-the-same-representation-to-bimanual-fetching]

## Part I：问题与挑战

这篇论文解决的不是普通“抓取”，而是**把指定物体从高密度拥挤场景里安全取出来**。输入是双视角 RGB、机器人本体状态，以及任务中已指定的目标；输出是连续末端位姿/动作，再由低层 IK 控制器执行。成功标准也比常规 grasping 更苛刻：不仅要拿到目标，还要尽量不扰动周围物体。

真正瓶颈有三层：

1. **看不全**：货架/柜体中的密集摆放会造成严重遮挡。直接投影得到的 RGB-D 体素、点云或原始深度，只能看到可见表面，看不到被遮住的目标边缘和障碍体积。
2. **迁不过**：真实深度传感器在物体边界、透明物体、反光物体上最不可靠，而这些正是零售和仓储场景中的高频难例。
3. **走不稳**：在拥挤场景里，严格 collision-free path 往往根本不存在。传统 motion planning 只能在静态几何约束下找路，但真实取物需要对“动作会如何扰动周边物体”有隐式动态判断。

所以，这篇论文的“真问题”可以概括为一句话：**如何在大量遮挡、深度不可靠、且常常不存在无碰路径的情况下，学到一个可零样本迁移到真实世界的低扰动取物策略。**

为什么现在值得做？因为论文恰好踩中了三个成熟条件：
- 基础深度模型已经能从 RGB 提供跨域稳定的几何线索；
- GPU 仿真和体素化生成允许用很低成本合成高密度拥挤场景；
- 模仿学习/扩散策略足以承接复杂连续动作分布，而不必在线求解困难规划。

边界条件也很明确：论文主打**单臂、局部取物、目标已指定**，主评测在 shelf 场景，外加 tabletop / drawer 的扩展示例。

## Part II：方法与洞察

FetchBot 的设计可以理解为三步：**先造难数据和老师，再学可迁移的局部 3D 表征，最后蒸馏成能执行的动作策略。**

### 1) 先把训练分布做对：UniVoxGen + dynamics-aware oracle

论文先用 **UniVoxGen** 在统一体素空间里生成场景。核心不是“体素更炫”，而是它把场景生成中的碰撞检测变成了轻量级集合操作（并、交、差、变换），所以能快速生成**1M 个高密度拥挤场景**。这一步同时带来两个收益：

- 训练分布从“稀疏、容易”的仿真，变成“密集、频繁遮挡”的现实风格分布；
- 还能天然得到**完整 occupancy ground truth**，这对后面的遮挡补全监督非常关键。

接着，论文不用传统 planner 来造示范，而是训练一个 **RL oracle**。原因很直接：在拥挤取物中，planner 只知道静态几何可行性，但不知道某个动作会不会把旁边物体带倒或挤开。RL oracle 用“成功取物 + 行为约束 + 环境扰动惩罚”的奖励学出**低扰动动作先验**，再用它收集 **500k demonstrations**。

这一步改变的是**动作分布**：从“找无碰轨迹”变成“学会低扰动地拿出来”。

### 2) 再把输入分布做对：预测深度作为 sim2real 中介

FetchBot 没有直接依赖真实深度传感器，而是把仿真 RGB 和真实 RGB 都先送进 **DepthAnything**，映射到同一个 predicted-depth 空间。

这里的关键洞察不是“深度比 RGB 好”这么简单，而是：
- RGB 的域差主要来自纹理、光照、材质；
- predicted depth 更偏几何，跨域变化小；
- 对透明/反光物体，RGB 上的基础模型预测反而常比廉价深度传感器稳定。

因此，论文把 **foundation depth** 当成“跨域统一前端”，而不是把真实传感器深度硬塞给策略。

### 3) 最后把“看见的表面”变成“可行动的完整局部 3D”

单有 predicted depth 还不够。论文自己也做了 baseline，直接拿 predicted depth 做策略并不强，因为它有**尺度歧义**，而且仍然只能描述可见区域。

真正有效的是下一步：  
把多视角 predicted-depth 特征用 2D-3D 查询融合到局部 3D 网格里，并在**末端执行器附近 ROI**上做**语义 occupancy prediction**，区分 target / obstacle / robot，而不是只做二值占据。

这一步的作用非常因果：
- 监督信号来自 UniVoxGen 的**完整 scene occupancy**；
- 因此网络被迫学会“从局部可见线索推断被遮挡体积”；
- 下游策略虽然**不直接吃最终 occupancy map**，但会吃被 occupancy 任务塑形后的 latent feature，因此隐式获得更完整的 3D 几何先验。

随后论文冻结这个 3D 视觉编码器，再训练一个 **transformer + diffusion head** 去模仿 oracle 动作。高层策略 10 Hz 输出目标末端位姿，低层 100 Hz IK 控制器负责平滑执行。

### 核心直觉

最关键的变化不是某一个模块，而是同时拧动了三个“因果旋钮”：

1. **输入分布变了**：  
   从 RGB/原始深度 → foundation-model predicted depth  
   结果是策略面对的输入不再强依赖纹理和传感器噪声，sim 与 real 更接近。

2. **信息瓶颈变了**：  
   从“只对可见表面建模” → “在 ROI 内学习语义 occupancy completion”  
   结果是策略不再只基于表面几何做动作，而是能对被挡住的目标和障碍做隐式推断。

3. **动作先验变了**：  
   从“静态无碰规划” → “低扰动 dynamics-aware expert prior”  
   结果是策略学到的是“如何拿出来”，而不是“能否找到一条理想路径”。

一个很重要的分析点是：**predicted-depth baseline 本身并不强**。这恰恰说明论文真正有效的不是“把 RGB 变深度”，而是“把 predicted depth 当作跨域稳定线索，再用多视角 occupancy 学习把它变成可用的 metric 3D latent”。

| 设计选择 | 主要解决的瓶颈 | 带来的能力变化 | 代价 / 风险 |
|---|---|---|---|
| predicted depth 替代传感器深度 | 视觉域差、深度噪声 | 零样本 sim2real 更稳，对透明/反光物体更鲁棒 | 单独使用有尺度歧义，必须配合 3D 学习 |
| ROI 语义 occupancy 预训练 | 严重遮挡导致信息缺失 | 能推断不可见几何，降低取物碰撞 | 只建模局部区域，远距离全局上下文较弱 |
| RL oracle 示范蒸馏 | planner 不理解动态后果 | 学到低扰动、可执行的动作先验 | 依赖仿真物理、奖励设计与较高训练成本 |
| UniVoxGen 大规模场景生成 | 缺少高密度真实数据 | 覆盖难场景分布，并提供完整 occupancy 标签 | 仍受合成规则和资产多样性限制 |

## Part III：证据与局限

相对 prior work，这篇论文的能力跃迁主要体现在三点：**更能迁移、更能补全遮挡、更能低扰动取出目标。**

- **比较信号（仿真）**：在 3000 个高密度场景中，Occupancy 版本达到 **81.46% / 91.02%**（吸盘 / 平行夹爪）成功率，优于启发式、motion planning 以及 RGB、点云、原始深度、predicted depth、RGB-D voxel 等学习基线；同时平移/旋转扰动也更低。说明它不仅“能拿到”，而且“拿得更稳”。
- **比较信号（真实世界零样本）**：不做真实数据微调，真实世界达到 **86.6% / 93.3%** 成功率，平均 **89.95%**。这支持其 sim2real 主张，而且论文还展示了对透明、反光、不规则物体的可用性。
- **消融信号**：去掉 DepthAnything 后成功率从 **86.60%** 降到 **60.00%**；去掉 occupancy 后降到 **73.33%**；两者都去掉、只用 RGB 时仅 **33.33%**。这表明跨域桥接和遮挡补全不是可有可无，而是互补关系。
- **分析信号**：数据规模扩大能稳定提升 occupancy 预训练和策略学习效果；ROI 消融显示小而合适的局部体积优于大范围输入，说明该方法的优势确实来自“局部可行动几何”，不是盲目堆上下文。

但证据边界也要看清：
- 真实世界主要报告**成功率**，没有像仿真那样精确量化环境扰动；
- 评测以自建零售/货架场景为主，不是公共大规模 benchmark；
- 因而“安全性”的强证据主要仍来自仿真的扰动指标与真实示例结合，而非全量实测物理量。

**复现依赖**
- UniVoxGen 生成 1M 场景需约 **8×RTX 4090 / 12h**；
- occupancy 预训练需约 **40×RTX 4090 / 16h**；
- 策略训练还需 500k demonstration frames；
- 真实部署依赖 **DepthAnything、精确多视角标定、Flexiv Rizon 4S、双相机 RGB 输入**。  
这意味着方法并非“轻量即插即用”，但系统工程路径是清晰的。

**局限性**
- Fails when: 遮挡过强时，策略可能为了绕障输出复杂动作，进而触发机械臂关节极限；目标完全被封死、初始不可达时也会失败。
- Assumes: 目标对象在任务中已指定；相机内外参与视角设置已知；可获得大规模仿真资产、完整 occupancy 标签，以及 foundation depth 模型与较高算力支持。
- Not designed for: 双臂搬运大体积/重物、先清障再恢复场景的长时序推理、开放词汇目标发现或目标定位。

**可复用组件**
- `predicted depth as sim2real bridge`：适合真实深度很脏的机器人任务。
- `ROI semantic occupancy pretraining`：适合严重遮挡下的局部操作。
- `oracle-to-policy distillation`：适合把难以手工规划的动态安全动作蒸馏成可部署策略。
- `voxel-based clutter generation`：适合需要大规模困难场景和完整 3D 标签的仿真学习。

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_FetchBot_Object_Fetching_in_Cluttered_Shelves_via_Zero_Shot_Sim2Real.pdf]]