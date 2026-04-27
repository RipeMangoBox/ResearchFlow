---
title: "ManiCM: Real-time 3D Diffusion Policy via Consistency Model for Robotic Manipulation"
venue: arXiv
year: 2024
tags:
  - Embodied_AI
  - task/robotic-manipulation
  - task/imitation-learning
  - diffusion
  - consistency-distillation
  - dataset/Adroit
  - dataset/MetaWorld
  - dataset/RLBench
  - repr/point-cloud
  - opensource/no
core_operator: "将点云条件下的多步动作扩散策略蒸馏为满足自一致性的单步动作恢复函数，实现实时3D操控"
primary_logic: |
  点云观测+机器人位姿+带噪动作 → 紧凑3D编码并施加动作自一致约束，通过教师扩散策略做一致性蒸馏、直接预测干净动作样本 → 单步/少步输出机械臂与夹爪动作序列
claims:
  - "Claim 1: 在Adroit与MetaWorld共31个任务上，1-step ManiCM将平均单步推理时间从DP3的177.6ms降到17.3ms，同时平均成功率从77.5%提升到78.5% [evidence: comparison]"
  - "Claim 2: 在一致性蒸馏中直接预测动作样本比预测噪声更稳定，平均成功率约78.5%对24.4%，且学习曲线收敛更快 [evidence: ablation]"
  - "Claim 3: 在RLBench多任务和3个真实世界UR3e任务上，ManiCM保持与现有3D扩散策略相当的任务成功率，同时实现约10×到13.3×的推理加速 [evidence: comparison]"
related_work_position:
  extends: "Consistency Models (Song et al. 2023)"
  competes_with: "DP3 (Ze et al. 2024); 3D Diffuser Actor (Ke et al. 2024)"
  complementary_to: "N/A"
evidence_strength: strong
pdf_ref: paperPDFs/Vision_Action_VA_Models_2024/arXiv_2024/2024_ManiCM_Real_time_3D_Diffusion_Policy_via_Consistency_Model_for_Robotic_Manipulation.pdf
category: Embodied_AI
---

# ManiCM: Real-time 3D Diffusion Policy via Consistency Model for Robotic Manipulation

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2406.01586), [Project](https://ManiCM-fast.github.io)
> - **Summary**: 这篇工作把3D点云条件下的扩散操控策略蒸馏成一个满足一致性约束的单步动作生成器，在基本不牺牲成功率的前提下把机器人决策延迟降到实时级。
> - **Key Performance**: 在31个Adroit/MetaWorld任务上，1-step ManiCM达到 **17.3ms/step** vs DP3 的 **177.6ms/step**；平均成功率 **78.5%** vs **77.5%**。

> [!info] **Agent Summary**
> - **task_path**: 单目RGB-D生成的稀疏点云 + 机器人位姿 / 离线模仿学习 -> 机械臂末端与夹爪动作序列
> - **bottleneck**: 3D点云条件下的扩散策略需要多步串行去噪，单步决策延迟约160-180ms，难以满足闭环实时控制
> - **mechanism_delta**: 将“逐步预测噪声”的扩散推理改成“从ODE轨迹任一点直接恢复干净动作样本”的一致性蒸馏推理
> - **evidence_signal**: 31任务主结果显示 1-step ManiCM 以17.3ms/step 获得78.5%平均成功率，约10×快于DP3且成功率略高
> - **reusable_ops**: [point-cloud-to-compact-3d-embedding, diffusion-teacher-to-consistency-student]
> - **failure_modes**: [epsilon-prediction-under-consistency-training-is-unstable, one-step-inference-can-be-slightly-worse-than-4-step-on-fine-grained-tasks]
> - **open_questions**: [can-this-distill-large-pretrained-robot-foundation-policies, how-robust-is-single-step-action-generation-under-depth-view-shift]

## Part I：问题与挑战

这篇论文要解决的，不是“机器人动作分布学不出来”，而是**学得出来但采样太慢**。

### 1. 真正的问题是什么
近两年扩散策略在机器人操控上效果很好，尤其是像 DP3 这样引入点云条件后，能更好理解3D空间关系、接触位置和目标几何。但这类方法有一个很现实的瓶颈：

- 每次输出动作都要做多步去噪；
- 3D输入是高维点云，条件编码本身就不便宜；
- 最终导致单步决策延迟常在 **160-180ms** 量级。

对离线评测来说这还可以接受，但对闭环控制来说就很尴尬：环境在变、接触在变、抓取窗口很短，策略还在“慢慢采样”。

### 2. 为什么现在值得解决
因为前一代工作已经证明了两件事：

1. **3D扩散策略的效果确实强**，不是一个“没必要保留”的路线；
2. **速度已经成为主要短板**，而不是表达能力短板。

也就是说，现在最值得调的旋钮，不再是“换更大的 backbone”，而是**把扩散的多步推理链压缩掉**。论文的核心判断是：在机器人动作空间里，尤其是低维动作流形上，这种压缩是可能的。

### 3. 输入/输出接口与边界条件
本文的设定很明确：

- **输入**：单视角 RGB-D 相机得到的稀疏点云 + 机器人 proprioception / pose
- **输出**：机械臂末端动作与夹爪控制构成的动作序列
- **学习范式**：离线模仿学习
- **训练数据**：每个任务仅使用少量专家示范（如 10 demos）
- **主要场景**：MetaWorld、Adroit、RLBench，以及3个真实机器人任务

所以它不是通用机器人基础模型，也不是语言规划器；它聚焦的是一个很具体但很关键的问题：**如何把3D diffusion policy 从“好但慢”变成“好且够快”**。

---

## Part II：方法与洞察

ManiCM 的思路可以概括成一句话：

**不要再让策略一步步去噪，而是直接学会“从任意噪声时刻一跳回到干净动作”。**

### 方法主线

#### 1. 先保留 DP3 的 3D 条件建模优点
作者没有抛弃 3D 表示，而是延续点云路线：

- 从单视角 RGB-D 恢复点云；
- 用 FPS 下采样到固定点数；
- 用 MLP 编码成紧凑的 3D 表示；
- 再和机器人位姿一起作为动作生成条件。

这一步的目的不是创新表示，而是保留 DP3 的空间感知能力，同时把条件输入压紧，减少后续生成负担。

#### 2. 把动作扩散改写成“动作自一致恢复”
传统 diffusion policy 的推理逻辑是：

- 从噪声开始；
- 一步步预测噪声残差；
- 经过多次更新后得到动作。

ManiCM 改成：

- 无论当前动作在扩散轨迹的哪个时刻；
- 模型都应该直接输出同一个干净动作。

也就是说，模型学的不是“下一步怎么去一点噪”，而是“这条轨迹最终对应哪个真实动作”。

#### 3. 用 teacher diffusion policy 做一致性蒸馏
训练时有三类网络角色：

- **teacher network**：预训练好的扩散策略（如 DP3）
- **online network**：当前要学习的一致性模型
- **target network**：online 的 EMA 副本，用来稳定训练

大致过程是：

1. 把真实专家动作加噪到较晚时刻；
2. teacher 负责提供高质量去噪参考；
3. 用 ODE solver 从 teacher 结果构造相邻时刻目标；
4. 要求 online 与 EMA target 在不同时间点上预测出一致的干净动作。

这样训练好后，推理时只需要 online network，本质上就把“多步 diffusion sampling”压缩成了“单步 consistency decoding”。

### 核心直觉

**改变了什么**  
从“局部噪声残差预测”改成“全轨迹终点恢复”。

**哪个瓶颈被改变了**  
原方法的瓶颈是**串行采样依赖**：第 \(n\) 步要等第 \(n-1\) 步，延迟线性累积。  
ManiCM 把它变成**单次映射问题**：任意噪声级别都映射到同一个干净动作。

**带来了什么能力变化**  
从“每步要花 10 次以上函数评估才能出动作”，变成“1-step 就能给出可执行动作”，于是 3D diffusion policy 首次真正接近实时闭环控制。

### 为什么这个设计在机器人动作空间里成立
关键不是 consistency model 本身，而是**动作空间与图像空间不同**。

#### 原因 1：动作流形低维，直接预测样本更容易
图像生成里常默认预测 epsilon，因为图像分布太高维，直接回归样本很难。  
但机器人动作维度很低，论文中大约 28 维左右，这时直接预测干净动作反而更自然：

- 目标更直接；
- 方差更低；
- 收敛更快。

#### 原因 2：一致性训练会放大噪声预测的不稳定性
在普通 diffusion 里，模型只要学好“相邻一步”的局部更新即可。  
但在 consistency setting 里，不同时间点都要对齐到同一个终点。如果你预测的是 epsilon，误差会沿轨迹被放大，最终更难稳定收敛。

这也是作者最重要的机制判断之一：  
**在低维动作空间里，sample prediction 不是小技巧，而是 ManiCM 能否工作的重要因果开关。**

#### 原因 3：点云条件被压缩后，计算开销主要转向采样链本身
也就是说，真正要砍掉的不是 backbone，而是 NFE。论文也用实验证明：**减采样步数比单纯缩小模型更有效**。

### 战略取舍

| 设计选择 | 改变的瓶颈 | 直接收益 | 代价/风险 |
|---|---|---|---|
| 稀疏点云 + 紧凑编码 | 高维3D观测开销 | 保留3D几何信息，兼容现有DP3路线 | 依赖深度质量、相机标定和视角覆盖 |
| 一致性蒸馏替代多步采样 | 串行去噪延迟 | 1-step/4-step 推理，显著降时延 | 需要先有一个可用的 diffusion teacher |
| 直接预测动作样本 | epsilon 预测高方差 | 收敛更快，训练更稳 | 依赖“动作流形较低维”这一前提 |
| 1-step 推理 | 实时性 | 达到约 17ms/step | 某些精细任务略逊于 4-step 版本 |
| 4-step 推理 | 细粒度修正不足 | 成功率偶尔更高 | 速度回退到约 66ms/step |

---

## Part III：证据与局限

### 关键证据信号

#### Signal 1：主对比结果表明能力跃迁来自“采样机制改变”而非单纯压模型
在 Adroit + MetaWorld 共 31 个任务上：

- **DP3**：177.6ms/step，77.5% 平均成功率
- **ManiCM (1-step)**：17.3ms/step，78.5% 平均成功率

这说明能力跳跃点非常明确：  
**不是牺牲效果换速度，而是在保持甚至略增成功率的同时，把时延打到原来的约 1/10。**

#### Signal 2：最关键的因果证据来自 sample-vs-epsilon 消融
作者对比了两种参数化方式：

- **预测动作样本**：收敛快，平均成功率约 78.5%
- **预测噪声 epsilon**：明显不稳定，平均成功率约 24.4%

这组消融非常关键，因为它不是“调参有效”，而是直接支持论文的机制假设：  
**一致性蒸馏在低维动作空间中更适合做 sample prediction。**

#### Signal 3：跨设置验证说明不是只在单一仿真基准里成立
- **RLBench 多任务**：与 3D Diffuser Actor 竞争性成功率，同时 **13.3×** 更快
- **真实世界 3 个 UR3e 任务**：推理从约 **177ms** 降到 **17ms**，成功率保持可比，某些任务更高

这表明 ManiCM 的收益不只是 benchmark trick，而是对真实机器人控制频率有直接意义。

### 1-2 个最该记住的指标
- **17.3ms/step vs 177.6ms/step**：这是论文最核心的系统指标。
- **78.5% vs 77.5%**：说明加速不是靠明显牺牲策略质量换来的。

### 局限性
- **Fails when**: 使用 epsilon prediction 做一致性蒸馏时，训练会出现复合不稳定，论文报告平均成功率仅约 24.4%；另外某些更精细的任务仍会从 4-step 推理中受益，说明 1-step 并非统一最优。
- **Assumes**: 需要一个高质量预训练 diffusion teacher（文中主要是 DP3）；需要离线专家示范数据；需要单视角 RGB-D、点云重建与下采样流程；文中的速度结论是在 RTX 4090 上测得。
- **Not designed for**: 大规模预训练通用机器人基础策略的蒸馏扩展；无深度/无点云的输入场景；语言驱动规划或长程层级任务分解不是本文重点。

### 复用价值高的组件
- **一致性蒸馏配方**：可把已有 diffusion policy 蒸馏成少步甚至单步 student。
- **低维动作上的 sample prediction**：对其他 action diffusion / policy distillation 工作都有启发。
- **点云紧凑条件接口**：证明 3D表示可以和快速生成策略兼容，而不一定要退回2D图像策略。

### 一句话结论
ManiCM 的真正贡献，不是再做一个更强的扩散策略，而是证明了：  
**对机器人这种低维动作、高时延敏感的场景，可以把“扩散策略的生成方式”从多步采样改写成单步一致性恢复，并且几乎不丢任务成功率。**

![[paperPDFs/Vision_Action_VA_Models_2024/arXiv_2024/2024_ManiCM_Real_time_3D_Diffusion_Policy_via_Consistency_Model_for_Robotic_Manipulation.pdf]]