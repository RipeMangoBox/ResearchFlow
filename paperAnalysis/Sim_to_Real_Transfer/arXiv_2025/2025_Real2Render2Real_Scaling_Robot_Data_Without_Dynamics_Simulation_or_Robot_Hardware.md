---
title: "Real2Render2Real: Scaling Robot Data Without Dynamics Simulation or Robot Hardware"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/robot-manipulation
  - task/imitation-learning
  - 3d-gaussian-splatting
  - differential-inverse-kinematics
  - trajectory-interpolation
  - dataset/Real2Render2Real
  - opensource/no
core_operator: "基于手机扫描与单段人类示范视频重建对象级6DoF运动，再经抓取采样、差分逆运动学和并行渲染批量合成机器人视觉-动作数据。"
primary_logic: |
  手机多视角物体扫描 + 单目人类示范视频 + 机器人URDF
  → 3DGS重建与部件分割、对象/部件6DoF轨迹跟踪
  → 轨迹插值重定向、抓取采样、差分逆运动学生成机器人执行
  → IsaacLab在关闭动力学/碰撞建模依赖下并行渲染RGB-动作-本体感觉训练数据
claims:
  - "在5个物理机器人任务中，1000条R2R2R合成数据训练的策略在多项任务上达到或超过150条真人teleoperation训练结果，例如 π0-FAST 在“杯子放上咖啡机 / 开抽屉 / 关水龙头”上分别达到80.0% / 86.6% / 80.0%，对比 teleoperation 的73.3% / 60.0% / 80.0% [evidence: comparison]"
  - "在“把杯子放到咖啡机上”任务中，去掉轨迹插值后，1000条R2R2R数据训练的π0-FAST成功率从80.0%降到0.0%，Diffusion Policy从53.3%降到6.7% [evidence: ablation]"
  - "单张RTX 4090上，R2R2R平均以51条演示/分钟生成数据，而人工teleoperation约为1.7条/分钟；作者报告平均吞吐快约27×，且可随GPU数扩展 [evidence: comparison]"
related_work_position:
  extends: "Robot See Robot Do (Kerr et al. 2024)"
  competes_with: "Video2Policy (Ye et al. 2025); Phantom (Lepert et al. 2025)"
  complementary_to: "Sim-and-Real Co-training (Maddukuri et al. 2025); RoVi-Aug (Chen et al. 2024)"
evidence_strength: moderate
pdf_ref: paperPDFs/Sim_to_Real_Transfer/arXiv_2025/2025_Real2Render2Real_Scaling_Robot_Data_Without_Dynamics_Simulation_or_Robot_Hardware.pdf
category: Embodied_AI
---

# Real2Render2Real: Scaling Robot Data Without Dynamics Simulation or Robot Hardware

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2505.09601) · [Project](https://real2render2real.com)
> - **Summary**: 这篇工作把“手机扫描+一段人类示范视频”转成可批量扩增的机器人视觉-动作训练数据，用对象级真实运动和运动学重定向替代动力学仿真与机器人采集。
> - **Key Performance**: 单张 RTX 4090 平均 51 demos/min，约为人工 teleoperation 的 27×；在 5 个真实机器人任务上，1000 条 R2R2R 数据训练的策略在多项任务上达到或超过 150 条 teleoperation 训练结果。

> [!info] **Agent Summary**
> - **task_path**: 手机多视角扫描 + 单目人类示范视频 + 机器人URDF -> 合成RGB/本体感觉/动作对 -> 真实机器人操作策略
> - **bottleneck**: 机器人视觉-动作数据严重依赖慢速、昂贵、具 embodiment 绑定的 teleoperation；而高保真动力学仿真又难以同时满足接触真实性与视觉一致性
> - **mechanism_delta**: 用对象中心的真实6DoF运动重建与运动学重定向，替代基于物理接触求解的数据生成路径
> - **evidence_signal**: 5任务、1050次真实机器人评测显示高规模 R2R2R 数据可接近或追平 150 条 teleoperation，并且轨迹插值消融会出现断崖式掉点
> - **reusable_ops**: [对象级6DoF轨迹提取, 轨迹插值+差分逆运动学重定向]
> - **failure_modes**: [遮挡/反光/快速运动导致跟踪失败, 无碰撞约束的插值轨迹在杂乱场景中可能穿模]
> - **open_questions**: [如何加入碰撞与动力学一致性而不显著牺牲吞吐, 如何扩展到可变形物体与非抓取操作]

## Part I：问题与挑战

这篇论文真正要解决的，不是“再做一个更强的策略网络”，而是**机器人数据生产方式本身不可扩展**。

### 1. 真问题是什么
当前机器人学习主要有两条数据来源：

1. **人工 teleoperation**：数据真实，但慢、贵、需要机器人硬件，而且强绑定具体 embodiment。
2. **物理仿真**：可以并行扩展，但要付出高昂的接触建模、摩擦参数、资产碰撞几何、视觉 domain gap 成本。

对于视觉驱动的 manipulation policy，作者认为真正卡住规模化的瓶颈是：

- 没有一种方式能同时做到  
  **视觉像真实世界** + **动作对目标机器人可执行** + **生成速度远超人工**。

### 2. 为什么现在值得做
因为新一代 VLA / diffusion policy 明显是**吃数据规模**的，但机器人数据量和 LLM/VLM 相比差了几个数量级。与此同时，近两年 3DGS、part-level tracking、手机视频三维重建已经成熟到足以支撑一个新思路：

> 不再先把世界“物理模拟正确”，而是先把**对象运动和视觉观测**做对，再把它运动学地映射到机器人上。

### 3. 输入 / 输出接口
- **输入**：
  - 手机拍摄的多视角物体扫描
  - 一段单目人类示范视频
  - 目标机器人 URDF
  - 近似相机位姿标定
- **输出**：
  - 可训练策略的 RGB 观测
  - 机器人本体感觉状态
  - 对应动作序列/轨迹

### 4. 边界条件
这套方法不是通用世界模型，而是有明确适用边界：

- 桌面、准静态操作
- 物体为**刚体或关节体**
- 表面低反光，利于重建
- 演示过程中不能长期完全互相遮挡
- 策略输入默认为 RGB + proprioception
- 任务主要是**抓取式操作**，不是力控制或复杂接触操作

---

## Part II：方法与洞察

### 方法主线
R2R2R 可以概括为四段式流水线：

1. **真实资产提取**
   - 用手机多视角视频做 3D Gaussian Splatting
   - 用 GARField 做部件级分组/分割
   - 再把 3DGS 转成 mesh，兼容 IsaacLab 渲染

2. **真实对象运动提取**
   - 从单段人类示范视频中，用 4D-DPM 跟踪对象/部件的 6DoF 轨迹
   - 关键点不在“手怎么动”，而在“物体/部件怎么动”

3. **从单轨迹扩成多轨迹**
   - 把原始对象轨迹规范化到 canonical 空间
   - 对新的起终状态做仿射变换与姿态 Slerp
   - 用手部关键点推断抓取对象，再做 antipodal grasp 采样
   - 用 differential IK 把对象轨迹转成目标机器人的关节/末端轨迹

4. **高吞吐并行渲染**
   - IsaacLab 在这里不是动力学求解器，而是**并行 photorealistic renderer**
   - 对象设为 kinematic bodies，不解接触、不算摩擦
   - 做光照、相机、物体初始位姿随机化，生成大批 RGB-action 数据

### 核心直觉

**作者真正拨动的因果旋钮**是：

- **过去**：先依赖物理引擎把接触过程“算对”，再生成机器人数据  
- **现在**：先从真实视频把对象运动“拿到”，再让机器人在运动学上“跟上”

这带来的变化是：

1. **什么被改变了**  
   从“以动力学仿真为中心的数据生成”切换到“以对象真实轨迹为中心的数据生成”。

2. **哪个瓶颈被转移了**  
   训练数据质量不再依赖摩擦、碰撞、接触参数是否准确，而依赖：
   - 物体重建是否真实
   - 部件轨迹是否稳定
   - 机器人是否能运动学可达

3. **能力为什么会变强**  
   对 RGB + proprio 的 imitation policy 来说，训练时最关键的是**观测-动作配对的一致性**。  
   如果视觉外观接近真实、对象运动语义正确、机器人轨迹可执行，那么缺少精确动力学并不会像在力控任务里那样致命。  
   于是单段人类示范就能被扩展成**one-to-many** 的机器人训练集。

换句话说，R2R2R 不是在解决“如何更真实地模拟物理世界”，而是在解决：

> **如何以最低必要正确性，生产对视觉策略最有用的大规模样本。**

### 战略取舍

| 设计选择 | 解决的瓶颈 | 获得的能力 | 代价 / 风险 |
| --- | --- | --- | --- |
| 用对象轨迹 + 运动学替代动力学仿真 | 去掉接触建模、摩擦调参、碰撞网格制作 | 不需要物理仿真也能批量生成数据 | 无法建模力、滑移、柔顺、变形 |
| 用真实扫描的 3DGS 资产渲染 | 缩小 sim-real 视觉差距 | 训练时可直接喂 raw RGB，减少额外感知模块依赖 | 依赖可重建物体，反光/纹理差会崩 |
| 轨迹插值到新起终态 | 从单示范扩到多示范 | 获得 one-to-many 数据扩增能力 | 可能忽略环境碰撞，轨迹不一定物理可行 |
| 抓取采样 + differential IK | 将对象运动转成 embodiment-specific 动作 | 机器人无关，可迁移到不同 URDF | 目前主要适合 parallel-jaw、抓取式任务 |
| 大规模域随机化渲染 | 覆盖视觉变化 | 提升真实部署泛化 | 随机化过强时反而伤害学习 |

---

## Part III：证据与局限

### 关键证据信号

#### 1. 扩规模后，合成数据能逼近真人 teleop
最重要的信号不是“单条数据质量更高”，而是“**规模一上去之后能追平**”。

- 在 5 个任务、1050 次真实机器人评测里，R2R2R 数据的表现会随样本数上升而稳定增长。
- 代表性例子：
  - **杯子放上咖啡机**：π0-FAST 用 1000 条 R2R2R 达到 **80.0%**，高于 150 条 teleop 的 **73.3%**
  - **开抽屉**：π0-FAST 为 **86.6% vs 60.0%**
  - **关水龙头**：π0-FAST 为 **80.0% vs 80.0%**

但也要看到一个很关键的事实：

> **R2R2R 解决的是吞吐瓶颈，不是单条样本的信息密度瓶颈。**

在低数据区间，真人数据通常更高效。比如杯子任务上，150 条 teleop 的 π0-FAST 是 73.3%，而 150 条 R2R2R 只有 33.3%。

#### 2. 机制证据很强：轨迹插值是核心，不是装饰
在“杯子放上咖啡机”任务里，去掉轨迹插值后：

- π0-FAST：**80.0% → 0.0%**
- Diffusion Policy：**53.3% → 6.7%**

这说明 R2R2R 的关键不是“把原始视频重新渲染一遍”，而是**把对象轨迹重定向到新初始条件**。  
没有这个步骤，单演示基本无法扩成真正可泛化的数据分布。

#### 3. 效率增益非常明确
- 单张 RTX 4090：约 **51 demos/min**
- 人工 teleoperation：约 **1.7 demos/min**
- 平均约 **27×** 吞吐提升
- 代价是每个任务有约 **10 分钟级** 的一次性前处理

所以它的价值主张非常清楚：  
**前处理一次，之后用 GPU 持续“印数据”。**

#### 4. 统计结论要保守解读
论文写到两者没有显著差异，但从 appendix 的 TOST 来看：

- “相近”是有证据支持的
- 但全局上**并没有严格证明 ±5% 等效**

因此最稳妥的表述不是“已经严格等同于 teleop”，而是：

> **在当前任务范围内，高规模 R2R2R 数据表现接近 teleop，且没有明显劣势证据。**

### 局限性

- **Fails when**: 遮挡严重、运动过快、反光强、纹理弱时，重建和跟踪容易失效；场景有障碍物或 clutter 时，无碰撞约束的插值轨迹可能穿过环境几何。
- **Assumes**: 任务是桌面、准静态、刚体/关节体、可抓取的操作；需要近似相机位姿、手机可完成较稳定扫描、3DGS/GARField/4D-DPM 这类视觉链路能稳定工作；生成虽不需要机器人硬件，但仍依赖 GPU 渲染与多阶段三维处理工具链。
- **Not designed for**: 可变形物体、非抓取操作（推、拨、滑）、显式力反馈任务、多指灵巧手接触建模、需要精确摩擦/柔顺/滑移物理的场景。

### 资源与复现依赖
- 数据生成不需要机器人硬件，但需要：
  - 手机扫描与示范采集
  - 3DGS / segmentation / 4D tracking / meshification 工具链
  - 至少单卡 GPU 做并行渲染
- 训练侧作者使用：
  - Diffusion Policy：单 GH200 约 3 小时
  - π0-FAST LoRA：约 11 小时
- 若没有较稳的视觉重建与跟踪能力，整个系统收益会显著下降。

### 可复用组件
1. **对象中心的 6DoF/部件轨迹提取**：适合把真实视频转成可学习的操作结构。
2. **轨迹 canonicalization + 插值重定向**：从一条语义动作扩成多初始条件版本。
3. **抓取推断 + differential IK retargeting**：把对象级运动转换为机器人动作。
4. **把 IsaacLab 当作并行渲染器而非物理引擎**：这是很值得借鉴的系统设计思想。
5. **R2R2R + real co-training**：附录显示对 Diffusion Policy 可能更强，说明它也可作为真实数据的放大器，而不只是替代品。

## Local PDF reference

![[paperPDFs/Sim_to_Real_Transfer/arXiv_2025/2025_Real2Render2Real_Scaling_Robot_Data_Without_Dynamics_Simulation_or_Robot_Hardware.pdf]]