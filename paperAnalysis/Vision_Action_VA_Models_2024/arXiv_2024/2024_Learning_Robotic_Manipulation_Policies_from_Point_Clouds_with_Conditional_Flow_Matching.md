---
title: "Learning Robotic Manipulation Policies from Point Clouds with Conditional Flow Matching"
venue: arXiv
year: 2024
tags:
  - Embodied_AI
  - task/robotic-manipulation
  - task/imitation-learning
  - flow-matching
  - pointnet
  - so3-manifold
  - dataset/RLBench
  - opensource/full
core_operator: 将多视角点云与机器人状态编码为条件特征，用条件流匹配学习从高斯噪声到专家末端执行器轨迹的速度场，并通过数值积分生成动作序列。
primary_logic: |
  多视角RGB-D与机器人本体状态 → 融合/裁剪成点云并用改造PointNet编码，条件1D U-Net学习从噪声到专家轨迹的连续速度场（含6D旋转或SO(3)变体） → 通过ODE积分生成未来末端执行器位姿与夹爪动作
claims:
  - "PointFlowMatch在RLBench八个操作任务上达到67.8%平均成功率，较次优基线OL-ChainedDiffuser的34.6%高出33.2个百分点 [evidence: comparison]"
  - "在相同框架下将观测从点云换成RGB图像会使平均成功率从67.8%降至40.1%，说明观测表示是最大性能瓶颈 [evidence: ablation]"
  - "CFM在50步推理时与DDIM总体相当，但在少步推理时明显更强，例如k=2时61.9%对13.5% [evidence: ablation]"
related_work_position:
  extends: "Diffusion Policy (Chi et al. 2023)"
  competes_with: "3D Diffusion Policy (Ze et al. 2024); ChainedDiffuser (Xian et al. 2023)"
  complementary_to: "ACT3D (Gervet et al. 2023); Hierarchical Diffusion Policy (Ma et al. 2024)"
evidence_strength: moderate
pdf_ref: paperPDFs/Vision_Action_VA_Models_2024/arXiv_2024/2024_Learning_Robotic_Manipulation_Policies_from_Point_Clouds_with_Conditional_Flow_Matching.pdf
category: Embodied_AI
---

# Learning Robotic Manipulation Policies from Point Clouds with Conditional Flow Matching

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2409.07343), [Project](http://pointflowmatch.cs.uni-freiburg.de)
> - **Summary**: 论文把机器人模仿学习中的轨迹生成从“扩散去噪 + RGB观测”切换到“条件流匹配 + 多视角点云观测”，更直接利用3D几何信息来学习多模态、长时域的末端执行器轨迹。
> - **Key Performance**: RLBench 8任务平均成功率 **67.8%**，比次优基线高 **33.2** 个百分点；真实机器人 **open box 72%**。

> [!info] **Agent Summary**
> - **task_path**: 多视角RGB-D/机器人本体状态的固定示教模仿学习 -> 未来末端执行器位姿与夹爪动作轨迹
> - **bottleneck**: 低数据机器人IL里，RGB难以稳定显式表达3D几何，而动作轨迹又是多模态且含6-DoF旋转，导致长时域轨迹建模和精细接触控制都不稳
> - **mechanism_delta**: 用点云编码器提供显式几何条件，并用条件流匹配替代扩散去噪来学习“从噪声到专家轨迹”的连续速度场
> - **evidence_signal**: RLBench八任务平均成功率67.8%，较OL-ChainedDiffuser的34.6%提升33.2点；点云替代RGB单独带来27.7点提升
> - **reusable_ops**: [multi-view point cloud fusion, conditional trajectory flow matching]
> - **failure_modes**: [out-of-distribution states without corrective demos, precise grasping can miss the object by a small margin]
> - **open_questions**: [whether CFM still dominates in multi-task/generalist manipulation, when manifold-native SO(3) flow matching beats Euclidean 6D projection]

## Part I：问题与挑战

这篇论文关注的不是“CFM 能不能用于机器人”这么宽泛的问题，而是一个更具体的设计问题：

**在固定示教数据、低样本、需要高精度6-DoF操作的设定下，机器人策略到底应该如何组合观测表示、轨迹生成目标和旋转表示，才能真正提升成功率？**

### 1) 任务接口

- **输入**：最近 \(T_{obs}\) 步观测，包括  
  - 机器人本体状态  
  - 多视角RGB-D经过投影、融合、裁剪后的点云
- **输出**：未来 \(T_{pred}\) 步动作轨迹，动作是 **10维**：
  - 末端位置 3维
  - 末端朝向 6维
  - 夹爪开合 1维
- **执行方式**：闭环 receding horizon——每次预测一段轨迹，但只执行第一步，然后重新感知再规划。

### 2) 真正瓶颈是什么

作者识别了三个纠缠在一起的瓶颈：

1. **观测瓶颈**  
   RGB把语义、纹理、几何混在一起；但 manipulation 的关键往往是物体边界、相对位姿、抓取几何。低数据下，让模型从图像里“自己恢复3D结构”很贵。

2. **动作分布瓶颈**  
   机器人示教不是单峰回归问题。很多状态对应多种合理动作，尤其是长时域轨迹预测。单纯 BC 容易平均化，造成“不够果断但看起来合理”的失败动作。

3. **旋转建模瓶颈**  
   位置在欧氏空间里，但朝向在旋转流形上。若处理不好，训练目标和推理输出会在“可学性”和“几何合法性”之间拉扯。

### 3) 为什么现在值得做

因为两个趋势刚好汇合：

- **Diffusion Policy** 已经证明“生成式轨迹策略”在机器人中有效；
- **Conditional Flow Matching (CFM)** 近期在生成建模里表现出比传统 diffusion 更灵活的路径建模能力；
- 同时，**点云观测**在 manipulation 中相对 RGB 的优势也越来越明确。

所以现在适合系统性地问：  
**如果把机器人策略从 diffusion 换到 CFM，再把观测从图像换到点云，会不会得到真正的能力跃迁？**

### 4) 边界条件

这篇工作有明确边界：

- 固定离线示教，不在线探索
- 单任务策略，不做多任务泛化
- 依赖深度相机和多视角标定
- 关注的是**非层次化**的低层轨迹生成，而不是高层 waypoint/hierarchy

---

## Part II：方法与洞察

### 方法总览

PointFlowMatch 的结构很直接，但关键在于几个选择是配套成立的。

#### a. 观测侧：多视角点云而不是RGB
作者将每个相机的深度投影到统一3D坐标系，再做：

- 多视角融合
- voxel downsampling
- 工作空间裁剪

然后用一个**去掉 T-Net 的 PointNet**编码点云。  
这里的直觉很重要：T-Net追求旋转/平移不变性，但机器人操作恰恰需要对绝对位姿敏感，所以作者把这部分拿掉。

#### b. 策略侧：用 CFM 学习“噪声到轨迹”的速度场
不是直接回归动作，也不是标准 diffusion 的去噪，而是：

- 从随机噪声轨迹出发
- 学一个条件速度场
- 让它在积分后流向专家轨迹分布

条件来自：

- 点云编码特征
- 机器人本体状态

速度场由一个**条件 1D U-Net**预测。推理时从噪声起步，做若干步 ODE 积分，得到一段未来轨迹。

#### c. 旋转侧：比较两种路径
作者专门比较了两种处理末端朝向的方法：

- **Euclidean formulation**：在普通6D欧氏空间里学，最后再投影回合法旋转
- **SO(3) formulation**：直接在旋转流形上定义起点、速度和积分路径

这部分是论文的一个关键洞察来源。

### 核心直觉

**改变了什么？**  
把“从观测回归单点动作/沿固定扩散路径去噪”改成“在显式3D几何条件下，学习从噪声到专家轨迹的连续运输过程”。

**哪个瓶颈被改变了？**

1. **信息瓶颈改变**：  
   RGB → 点云  
   模型不再需要从少量示教中隐式恢复3D结构，而是直接看到几何。

2. **分布建模瓶颈改变**：  
   diffusion 的固定前向加噪路径 → CFM 的条件速度场  
   模型学习的是“怎么把噪声推到专家轨迹分布”，路径更灵活，少步推理更有机会保住性能。

3. **旋转学习瓶颈改变**：  
   “几何上更正确的流形建模”并不一定“更好学”。  
   SO(3) 虽然理论上更自然，但在学习上会遇到不连续性；简单的 6D 欧氏回归 + 最后投影，反而更平滑、更稳定。

**能力上带来了什么变化？**

- 对接触式、连续交互型任务更强
- 对多模态长轨迹预测更稳
- 在少步推理时比 DDIM 更能保住性能
- 单任务非层次策略也能超过现有强基线

### 为什么这个设计有效

因果链条可以概括为：

- manipulation 成功依赖**相对几何**而非纯视觉纹理
- 点云直接暴露几何结构，减少样本浪费
- CFM学的是连续速度场，不必绑定到预设扩散路径
- 当推理步数受限时，直线路径/流匹配往往比 diffusion 式迭代更稳
- 对旋转而言，**学习连续性**有时比**几何原生性**更重要

这也是论文最有价值的一点：  
**不是“更几何正确”就一定更好，而是“更容易被网络稳定学习”的表示，往往在政策学习中更重要。**

### 战略权衡表

| 设计旋钮 | 改变的瓶颈 | 收益 | 代价 | 论文结论 |
|---|---|---|---|---|
| 点云 vs RGB | 几何信息可见性 | 更直接表达物体和抓取几何，低数据更省样本 | 需要深度相机和标定 | **最大收益来源** |
| CFM vs DDIM | 生成路径刚性、少步推理能力 | 目标更灵活，低步数推理更强 | 仍是迭代式推理，非BC级时延 | 长步数近似持平，**少步数更优** |
| Euclidean 6D vs SO(3) | 旋转合法性 vs 学习连续性 | 6D更容易回归；SO(3)几何更自然 | 6D需后投影；SO(3)可能有不连续点 | **6D欧氏 + 投影略胜** |
| 单任务非层次 vs 层次策略 | 问题范围控制 | 更能隔离低层轨迹生成能力 | 不享受高层规划收益 | 论文只证明低层生成器很强 |

---

## Part III：证据与局限

### 关键实验信号

#### 1) 比较信号：主结果是明显跃迁，不是小幅领先
在 RLBench 的 8 个连续交互任务上，PointFlowMatch 的平均成功率是 **67.8%**。  
对比基线：

- Diffusion Policy：18.7%
- AdaFlow：19.0%
- 3D-DP：28.5%
- OL-ChainedDiffuser：34.6%

所以它不是“略好一点”，而是对最强基线也有 **+33.2 个点** 的差距。  
这说明提升不是来自某个任务上的偶然收益，而是跨任务成立。

#### 2) 消融信号：最大提升来自“看什么”，其次才是“怎么生成”
同样的总体框架下：

- **点云 + CFM**：67.8%
- **RGB + CFM**：40.1%

这 **-27.7 个点** 的落差非常说明问题：  
真正主导 manipulation 成功率的，首先是观测表示是否把几何显式化。

#### 3) 训练目标信号：CFM 的价值主要体现在低步数推理
当推理步数固定为 50 时：

- 点云 + DDIM：68.0%
- 点云 + CFM：67.8%

几乎打平。  
但一旦把推理步数降下来，CFM的优势就明显出现：

- \(k=1\)：36.8% vs 19.4%
- \(k=2\)：61.9% vs 13.5%

所以这篇论文关于 CFM 的真正结论不是“绝对上限一定更高”，而是：

> **在机器人轨迹生成里，CFM 更适合少步推理。**

而且图中推理时间也说明：  
**CFM 并不是单步更快，而是因为更少步时还能工作。**

#### 4) 旋转建模信号：SO(3) 原生建模并没有赢
作者比较了：

- 点云 + R6 + CFM：67.8%
- 点云 + SO(3) + CFM：67.4%

SO(3) 并未带来提升。  
作者还做了一个 SO(2) toy experiment，解释原因是：  
流形上的 geodesic 目标在“对极点”附近会出现不连续，导致神经网络更难平滑拟合。  
这给出一个很实用的经验：

> **机器人策略学习里，几何上更“正统”的表示，不一定比可学习性更好的近似表示更有效。**

#### 5) 真实机器人信号：有迁移潜力，但证据还有限
真实 Panda 机械臂上：

- open box：72%
- sponge on plate：48%

这说明方法不是只在仿真里成立。  
但任务数少，且失败模式主要是**已经伸到目标附近，却差一点没抓到**，说明它对精细定位误差仍然敏感。

> 注：正文结论处写过一次 68.6%，但主表结果是 67.8%；这里按主表与详细分任务结果解读。

### 局限性

- **Fails when**: 遇到示教分布外状态、需要恢复/纠错行为但数据中未覆盖时；真实抓取中对小幅位姿误差敏感，可能“到位但抓空”。
- **Assumes**: 固定离线示教；多视角深度相机且内外参已知；点云可稳定融合；单任务训练；迭代式生成推理时延在系统上可接受；真实机器人部署需要人工采集专家示教。
- **Not designed for**: 多任务/通用机器人策略、无深度或无标定场景、超低时延直接控制、依赖在线探索或大规模跨场景泛化的设置。

### 可复用组件

- **多视角 depth-to-point-cloud 融合流程**：投影、融合、体素下采样、工作空间裁剪
- **去掉 T-Net 的 PointNet 编码器**：适合对绝对位姿敏感的 manipulation
- **条件轨迹级 CFM 框架**：从噪声生成未来动作序列，可替换 diffusion 作为低层轨迹生成器
- **6D 旋转 + 推理后投影**：作为比 SO(3) 更稳的工程基线，值得优先尝试

## Local PDF reference

![[paperPDFs/Vision_Action_VA_Models_2024/arXiv_2024/2024_Learning_Robotic_Manipulation_Policies_from_Point_Clouds_with_Conditional_Flow_Matching.pdf]]