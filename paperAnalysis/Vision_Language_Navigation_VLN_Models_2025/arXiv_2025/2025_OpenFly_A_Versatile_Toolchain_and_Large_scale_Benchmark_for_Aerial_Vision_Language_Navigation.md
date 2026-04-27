---
title: "OpenFly: A Versatile Toolchain and Large-scale Benchmark for Aerial Vision-Language Navigation"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/aerial-vision-language-navigation
  - automatic-data-generation
  - keyframe-selection
  - token-merging
  - dataset/OpenFly
  - opensource/promised
core_operator: "用多渲染引擎与自动化轨迹/指令生成扩展空中VLN数据分布，再用地标感知关键帧选择和token压缩提升UAV导航。"
primary_logic: |
  多源仿真/真实场景与航拍图像 → 统一接口下的点云重建、语义地标提取、A*轨迹搜索与VLM指令生成 → 构建100K空中VLN数据集，并训练关键帧感知模型输出UAV离散动作
claims:
  - "OpenFly constructs a 100K-trajectory aerial VLN dataset spanning 18 scenes and 4 rendering engines, substantially exceeding prior aerial VLN datasets in scale [evidence: comparison]"
  - "OpenFly-Agent achieves 34.3% SR on test-seen and 22.6% SR on test-unseen, outperforming NaVila by 14.0 and 7.9 percentage points respectively [evidence: comparison]"
  - "Combining keyframe selection with visual token merging raises test-seen SR from 2.3% with the OpenVLA baseline to 34.3% [evidence: ablation]"
related_work_position:
  extends: "OpenVLA (Kim et al. 2024)"
  competes_with: "NaVila (Cheng et al. 2024); Navid (Zhang et al. 2024)"
  complementary_to: "DAgger (Ross et al. 2011); Ego-Planner (Zhou et al. 2021)"
evidence_strength: moderate
pdf_ref: paperPDFs/Vision_Language_Navigation_VLN_Models_2025/arXiv_2025/2025_OpenFly_A_Versatile_Toolchain_and_Large_scale_Benchmark_for_Aerial_Vision_Language_Navigation.pdf
category: Embodied_AI
---

# OpenFly: A Versatile Toolchain and Large-scale Benchmark for Aerial Vision-Language Navigation

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2502.18041)
> - **Summary**: OpenFly把空中VLN的数据生产链条从“单引擎+人工飞行+人工标注”改成“多引擎+自动轨迹生成+VLM自动写指令”，并在此基础上提出关键帧感知导航模型，显著提升了空中视觉语言导航的规模与可学性。
> - **Key Performance**: test-seen/test-unseen SR = **34.3% / 22.6%**；相对 NaVila 提升 **+14.0 / +7.9** 个百分点；真实场景 SR/OSR = **26.09% / 34.78%**

> [!info] **Agent Summary**
> - **task_path**: 语言指令 + 当前/历史无人机第一视角图像 -> 6类离散飞行动作
> - **bottleneck**: 空中VLN缺少大规模多样数据，且均匀历史采样会淹没与指令相关的关键地标
> - **mechanism_delta**: 用多引擎自动化工具链生成训练分布，并把历史建模从均匀采样改成“动作转折 + 地标框面积”筛关键帧，再做跨帧token合并
> - **evidence_signal**: 相对NaVila在seen/unseen SR上提升14.0/7.9 pts，且消融中KS+VTM把SR从2.3%提到34.3%
> - **reusable_ops**: [多引擎统一接口, 地标感知关键帧选择]
> - **failure_modes**: [地标与周边外观相近时误定位, 动作幅度估计不准导致偏航或过冲]
> - **open_questions**: [自动生成指令的噪声上限会不会限制模型上界, 关键帧启发式能否迁移到连续控制和更长时域任务]

## Part I：问题与挑战

这篇工作的真问题，不只是“空中VLN模型还不够强”，而是**整个任务缺少可扩展的数据供给**。

### 1. 真正的硬点在哪里
现有空中VLN数据集大多依赖 AirSim/UE 生态，通常需要：
1. 飞手手动控制 UAV 在模拟器里飞；
2. 人工再给轨迹写语言指令；
3. 最终得到的还是 10K 级数据规模。

这会导致三个直接瓶颈：
- **分布太窄**：主要受限于单一渲染引擎和可用资产，场景风格不够丰富。
- **成本太高**：轨迹和指令都需要大量人工参与，难以持续扩容。
- **规模太小**：难以支撑更强的 VLM/VLA 式模型，尤其是需要大量多模态行为数据的范式。

### 2. 为什么现在值得解决
空中VLN正好处在一个“可以被放大”的时间点：
- **3D Gaussian Splatting** 让真实场景重建进入训练分布，sim-to-real 不再只能靠纯虚拟资产；
- **GPT-4o 这类 VLM/LLM** 可以自动生成轨迹描述，降低语言标注成本；
- **OpenVLA 这类开放骨干** 提供了一个可迁移到导航动作预测的初始化起点。

所以现在要解决的核心，不是再手工做一个小数据集，而是把**场景构建、轨迹生成、指令生成、模型训练**连成一个规模化流水线。

### 3. 任务接口与边界
- **模型输入**：语言指令 + 当前帧/历史帧的无人机第一视角图像
- **模型输出**：6类离散动作 `{Forward, Turn Left, Turn Right, Move Up, Move Down, Stop}`
- **成功判据**：终点距离目标 20m 以内算成功，碰撞算失败
- **数据边界**：
  - 18 个场景，来自 4 类渲染/重建来源
  - 平均轨迹长度 99.1m，平均指令长度 59
  - 以**短中程空中导航**为主，而不是超长时域精细控制

一句话概括：**真正的瓶颈是“数据生产系统”与“历史观测的信息筛选”，不是单个动作头本身。**

## Part II：方法与洞察

OpenFly分成两个相互支撑的层面：
1. **平台/数据层**：把空中VLN数据自动化做出来；
2. **模型层**：让导航模型只关注真正重要的历史观测。

### 核心直觉

#### 直觉1：先改变训练分布，而不是先堆模型
过去的数据生成方式把研究者锁死在“少量、单风格、昂贵”的分布里。OpenFly把它改成：
- 多渲染引擎接入：UE、GTA V、Google Earth、3D GS
- 统一 movement / lidar / image 接口
- 自动点云、语义、轨迹、指令流水线

**变化链条**：
单引擎+人工标注  
→ 多引擎+自动生成  
→ 数据分布更广、样本量更大、真实感更强  
→ 模型的泛化与 sim-to-real 能力变强

#### 直觉2：空中导航的关键证据不是“更多帧”，而是“对的帧”
均匀抽帧在视频理解里常见，但在空中VLN里不合适，因为：
- 大量前进帧视觉上高度冗余；
- 真正决定“接下来转向/上升/停止”的，常常是**看到了某个关键地标**的瞬间。

OpenFly-Agent 的改动是：
- 先用**运动变化点**找候选关键帧；
- 再用**地标 grounding 框面积**筛掉无关帧；
- 保留下来的历史帧再做 **visual token merging**。

**变化链条**：
均匀历史采样  
→ 地标相关关键帧采样  
→ 历史上下文信噪比提高、视觉token冗余下降  
→ 更好的语言-视觉对齐与动作预测

#### 直觉3：token 压缩不是为了省算力而已，而是为了避免跨模态失衡
作者在消融中给出的一个重要解释是：如果历史视觉token太多，而文本token很少，文本对视觉的约束会被“淹没”。  
因此 token merging 的价值不仅是压缩计算，更是**减少背景噪声对跨模态注意力的稀释**。

### 方法拆解

#### 1. 多引擎统一平台
OpenFly统一了四类场景来源：
- **UE4/UE5**：大规模城市数字资产
- **GTA V**：高真实感城市街景
- **Google Earth**：真实地理区域的高空视角
- **3D GS**：由真实无人机采集图像重建的校园/现实场景

关键不是“多接了几个引擎”，而是作者做了一个**统一接口层**：
- agent movement interface
- lidar acquisition interface
- image acquisition interface

这使得后续的轨迹搜索、点云拼接、图像采样都可以在同一套坐标与控制规范下运行。

#### 2. 自动数据生成工具链
自动生成链条可以概括为：

**场景几何**  
→ 点云获取（栅格采样重建 / COLMAP稀疏重建）

**场景语义**  
→ landmark 级语义分割（3D scene understanding、点云投影轮廓、必要时人工标注）

**轨迹生成**  
→ 基于全局体素图和地标目标，用 A* 搜索无碰撞轨迹  
→ 通过重复设定局部终点，形成更复杂路径

**指令生成**  
→ 将完整轨迹按动作转折切成子轨迹  
→ 每段只给 VLM 看关键动作和末端几张图  
→ 先生成子指令，再用 LLM 融合成完整自然语言

这个设计的关键在于：**不是把整段视频粗暴扔给 VLM，而是先做结构化切分，再做语言生成**。这样既降低成本，也更符合“人类会围绕转折和地标来描述路线”的语言习惯。

#### 3. OpenFly-Agent
模型基于 **OpenVLA**，但把单帧动作预测扩成了“当前帧 + 历史关键帧”的导航决策。

核心模块：
- **Keyframe Selection**
  - 用 UAV 轨迹中的急剧变化点生成候选帧
  - 用地标 grounding 模块预测 instruction-relevant landmark 的 bbox
  - 只保留 bbox 面积足够大的帧作为最终关键帧
- **Visual Token Merging**
  - 选 landmark 框最大的帧作为参考帧
  - 对相邻关键帧做 patch token 相似度计算并合并
  - 压缩历史信息，但保留当前帧的高分辨率视觉token

这相当于把历史视频变成一种**稀疏但高价值的证据缓存**，而不是无差别地堆一长串帧。

### 战略权衡

| 设计选择 | 解决的瓶颈 | 获得的能力 | 代价/约束 |
| --- | --- | --- | --- |
| 多引擎 + 3D GS | 单引擎数据分布窄 | 更强场景多样性与一定 real-to-sim 能力 | 工程接入复杂，点云质量不均 |
| A* 轨迹 + VLM指令生成 | 人工飞行与标注成本高 | 100K 级数据可扩展生成 | 依赖 GPT-4o，自动指令存在噪声 |
| 动作转折驱动关键帧选择 | 均匀采样漏掉关键地标 | 历史观测更聚焦 instruction-relevant 证据 | “转折≈看见关键地标”是启发式假设 |
| Visual token merging | 历史帧冗余、跨模态失衡 | 更强 grounding 与更低计算负担 | 可能损失细粒度背景信息 |

## Part III：证据与局限

### 关键证据信号

#### 1. 比较信号：数据规模和覆盖面显著扩大
OpenFly 数据集达到：
- **100K trajectories**
- **18 scenes**
- **4 rendering engines**
- **15.6K vocabulary**

相较于 AerialVLN 和 OpenUAV 的 10K 级规模，这是一个明显的量级提升。  
这说明作者解决的不是某个 benchmark 上的小修小补，而是**把空中VLN从“小样本研究任务”推进到了“可系统训练”的阶段**。

#### 2. 比较信号：OpenFly-Agent 的主结果提升明显
在 test-seen / test-unseen 上：
- OpenFly-Agent SR = **34.3% / 22.6%**
- NaVila SR = **20.3% / 14.7%**

这两个数字最重要，因为它们直接说明：  
**同样是基于强VLM/VLA思路，关键帧感知历史建模比常规做法更适合空中VLN。**

#### 3. 消融信号：历史信息只有“筛过+压过”才真正有用
最有价值的消融不是简单证明“加模块有效”，而是揭示了因果关系：
- OpenVLA baseline：**2.3% SR**
- 只加 History：**6.9% SR**
- History + VTM：**16.6% SR**
- KS + VTM：**34.3% SR**

这说明：
- 历史帧本身有帮助，但如果不压缩，收益有限；
- 关键帧选择本身也不够，必须和 token merging 配合；
- 真正起作用的是**“关键证据保留 + 冗余剔除”**这两个旋钮同时转动。

#### 4. 真实场景信号：并非只在模拟器里有效
作者在 23 个真实户外任务上测试，OpenFly-Agent 达到：
- **SR 26.09%**
- **OSR 34.78%**

虽然绝对数值仍不高，但相对比较方法更强，且作者还展示了“在 OpenFly 上训练优于在 AerialVLN 上训练”的 real-world 结果趋势。  
这支持了一个更系统的结论：**多引擎 + 真实重建数据，确实在缩小 sim-to-real gap。**

#### 5. 数据质量信号：自动标注可用，但不是完美
作者对 3K 随机样本做人工检查，合格率 **91%**。  
这说明自动指令生成已经足够支撑大规模训练，但依然存在：
- 模糊地标描述
- 重复/歧义表达
- VLM 可能带来的幻觉问题

### 局限性

- **Fails when**: 未见场景分布偏移较大、地标与周边建筑视觉相似、或任务需要精确控制转向/前进幅度时；论文附录也给出失败例子，表现为地标误识别或动作幅度错误。
- **Assumes**: 依赖点云获取与语义地标提取流程；自动指令生成依赖 GPT-4o 这类闭源VLM；导航模型依赖 OpenVLA 骨干与离散动作空间；真实部署还依赖 Jetson Xavier NX、外部PC、局部规划器（Super）和 MPC 控制器。
- **Not designed for**: 端到端连续控制、动态障碍密集规避、多轮对话式导航、超长时域规划，以及极高精度着陆/控制类任务。

### 可复用部件

1. **多引擎统一 API 层**：适合后续做空中 embodied data collection。
2. **基于点云/语义/A* 的自动轨迹生成**：可迁移到其他 UAV 导航任务。
3. **VLM 分段式指令生成**：适合把长轨迹转成更自然的导航语言。
4. **地标感知关键帧选择 + token merging**：对任何“历史帧很多但关键帧很少”的导航/机器人视频决策问题都有参考价值。

## Local PDF reference

![[paperPDFs/Vision_Language_Navigation_VLN_Models_2025/arXiv_2025/2025_OpenFly_A_Versatile_Toolchain_and_Large_scale_Benchmark_for_Aerial_Vision_Language_Navigation.pdf]]