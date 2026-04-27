---
title: "Reinforcement Learning with Generalizable Gaussian Splatting"
venue: arXiv
year: 2024
tags:
  - Embodied_AI
  - task/robot-manipulation
  - task/offline-reinforcement-learning
  - gaussian-splatting
  - reinforcement-learning
  - graph-neural-network
  - dataset/RoboMimic
  - opensource/no
core_operator: 把传统需逐场景优化的3DGS改成图像条件的可泛化Gaussian预测器，并冻结为离线RL的显式3D状态编码器。
primary_logic: |
  多视角RGB观测与相机参数 → 深度估计得到3D位置、逐像素回归Gaussian旋转/尺度/不透明度并经GNN邻域平滑 → 形成3D一致且几何感知的Gaussian点集 → 输入离线RL策略输出机器人动作
claims:
  - "在Transport任务的IRIS设置下，GSRL成功率为36.0，高于图像表征的33.0、点云的25.0和体素的31.5 [evidence: comparison]"
  - "在Square任务的BCQ设置下，GSRL成功率为48.5，高于图像42.0、点云28.0和体素42.0 [evidence: comparison]"
  - "将Gaussian点数从4096减少到2048时，BCQ在Can和Square上的成功率分别从68.0/47.5降至61.0/35.0，而Lift基本不受影响 [evidence: ablation]"
related_work_position:
  extends: "GPS-Gaussian (Zheng et al. 2024)"
  competes_with: "GNFactor (Ze et al. 2023); SNeRL (Shim et al. 2023)"
  complementary_to: "BCQ (Fujimoto et al. 2019); IQL (Kostrikov et al. 2021)"
evidence_strength: moderate
pdf_ref: paperPDFs/Misc_Robotics/IROS_2024/2024_Reinforcement_Learning_with_Generalizable_Gaussian_Splatting.pdf
category: Embodied_AI
---

# Reinforcement Learning with Generalizable Gaussian Splatting

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2404.07950)
> - **Summary**: 论文把传统需要逐场景优化的3D Gaussian Splatting改造成可预训练、可冻结的视觉状态编码器，使离线机器人操作策略能够直接利用显式3D局部几何表征学习动作。
> - **Key Performance**: Transport + IRIS 成功率 36.0，优于图像/点云/体素的 33.0/25.0/31.5；Square + BCQ 达到 48.5，优于 42.0/28.0/42.0。

> [!info] **Agent Summary**
> - **task_path**: 多视角RGB观测（带相机参数）/离线机器人操作 -> 3D Gaussian场景表示 -> 机器人动作
> - **bottleneck**: 视觉RL缺少一种既保留3D局部几何、又能对新场景实时泛化且无需逐场景优化的状态表征
> - **mechanism_delta**: 用“深度估计 + 像素级Gaussian属性回归 + GNN平滑”的前向预测器，替代传统3DGS的逐场景拟合流程
> - **evidence_signal**: 在RoboMimic四任务×三种offline RL上多数优于图像/点云/体素表征，且Transport+IRIS相对三类表征提升约10%/44%/15%
> - **reusable_ops**: [image-conditioned Gaussian prediction, KNN-GNN Gaussian refinement]
> - **failure_modes**: [Transport任务下BCQ与IQL整体崩溃且表征改进不足以挽救算法失效, Gaussian点数过少或重建质量偏低时较难任务明显掉点]
> - **open_questions**: [能否在真实机器人和传感器噪声下保持泛化, 与NeRF类3D表征在统一协议下相比的收益边界有多大]

## Part I：问题与挑战

这篇论文要解决的核心，不是“再换一个视觉编码器”，而是**视觉RL里的状态表征瓶颈**：机器人看到的是多视角图像，但真正做决策需要的是对3D结构、局部几何和遮挡关系都敏感的状态。

### 真正难点是什么？
现有表示大致都卡在三类问题上：

1. **2D图像表征**  
   容易用，但天然缺少3D结构信息；对遮挡、视角变化和空间关系不够稳。

2. **点云 / 体素等显式3D表征**  
   虽然进入3D了，但局部几何细节有限。点云通常只有位置，体素受分辨率约束，都难以表达复杂局部形状。

3. **NeRF类隐式表征**  
   能做3D一致性，但通常更像“黑箱”特征场；一些方法依赖前景mask或额外语义先验，而且对未见场景泛化不够自然。

### 为什么现在要解这个问题？
因为 **3DGS** 刚好提供了一个很有吸引力的中间点：

- 它是**显式**的：每个高斯点都有位置、尺度、旋转、不透明度等属性；
- 它是**3D一致**的：能从多视角约束中获得稳定结构；
- 它还能描述比普通点云更细的**局部几何**。

但传统3DGS有个致命问题：**需要逐场景优化**。  
这对RL几乎不可用，因为策略交互时不可能每一步都先跑一轮重建优化。

### 输入 / 输出接口与边界条件
- **输入**：多视角RGB图像 + 相机内外参（也支持单目变体）
- **中间表示**：3D Gaussian点集
- **输出**：离线RL策略动作
- **实验边界**：RoboMimic仿真环境、离线示教数据、任务内分布相对稳定的机器人操作场景

一句话概括：  
**真实瓶颈是“如何把视觉观测 amortize 成可实时调用的3D结构表征”，而不是单纯换一个更深的policy backbone。**

## Part II：方法与洞察

### 设计哲学
作者的关键改写是：

> 把“测试时对每个场景单独优化3DGS”的流程，改成“训练时在任务分布上学习一个从图像到Gaussian参数的共享映射”。

也就是把 3DGS 从一个**scene fitting**问题，变成一个**conditional prediction**问题。

### 方法主线
整个 GSRL 分成两阶段：

1. **预训练通用Gaussian预测器**
   - 输入多视角图像与相机参数；
   - 学习直接输出该场景的3D Gaussian表示；
   - 用目标视角的重建误差做监督。

2. **冻结表征，训练RL策略**
   - 将预训练好的Gaussian预测器接到环境观测后面；
   - 每一步把图像实时变成Gaussian点集；
   - RL策略直接在这套3D表征上学动作。

### 关键模块
1. **Depth Estimation**
   - 用双目/邻近视角预测深度；
   - 把2D像素网格抬升到3D坐标，得到Gaussian中心位置。

2. **Gaussian Regressor**
   - 对每个像素回归高斯属性：旋转、尺度、不透明度；
   - 颜色直接复用RGB，而不是再学SH系数；
   - 于是得到每个点的完整高斯参数。

3. **Gaussian Refinement**
   - 用KNN图上的GNN自编码器做平滑；
   - 目的不是“增加表达力”，而是去掉多视角带来的颜色/几何不一致噪声。

4. **RL Integration**
   - 预训练编码器冻结；
   - 下游BCQ / IQL / IRIS只看到Gaussian集合，不直接看原始图像。

### 核心直觉

**改了什么？**  
把 3DGS 的“逐场景优化”改成“任务分布上的前向预测”。

**改变了哪个瓶颈？**  
原来每个新场景都要从头拟合，时间和泛化都不适合RL；现在模型在训练时就学会了“2D局部patch → 3D局部几何”的共享先验，把场景重建成本前移并摊销掉了。

**能力发生了什么变化？**  
- 能在RL循环中实时得到3D表示；
- 比普通点云多了局部几何属性；
- 比纯2D图像更稳地处理遮挡与多视角一致性。

**为什么这会有效？**  
因为 RoboMimic 这类任务里，场景分布并不是无限开放的：物体类别、相机布局、操作模式都有重复性。  
这意味着“图像纹理/轮廓到局部3D结构”的映射可以被预先学成一个任务先验，而不是每次在线重新求解。

### 战略取舍

| 设计选择 | 带来的收益 | 代价 / 风险 |
|---|---|---|
| 可泛化GS预测器替代逐场景优化 | 让3DGS进入RL闭环成为可能 | 依赖任务分布相似性，超出分布可能失效 |
| Gaussian而非点云/体素 | 显式保留局部几何与3D一致性 | 表征更复杂，对预测误差更敏感 |
| GNN refinement | 抑制跨视角噪声，提高一致性 | 可能过平滑，增加额外计算 |
| 冻结表征后再训策略 | 训练更稳定、便于插拔到不同RL算法 | 表征不能随策略联合适配 |

### 训练信号
作者没有把重点放在复杂loss设计上，而是保持直接：
- **新视角渲染重建损失**：让预测Gaussian能渲染出目标视角；
- **refinement自重建损失**：让GNN平滑后仍保留有效高斯属性。

## Part III：证据与局限

### 关键实验信号

**信号1：只换表征，不换RL算法时，Gaussian表示多数更强。**  
这是最关键的证据，因为它更接近“控制变量实验”。

- 在 **Square + BCQ** 上，GSRL 为 **48.5**，高于图像 **42.0**、点云 **28.0**、体素 **42.0**。  
  **结论**：显式Gaussian属性确实比普通点云更能支撑需要局部几何判断的操控任务。

- 在最难的 **Transport + IRIS** 上，GSRL 为 **36.0**，图像/点云/体素分别是 **33.0 / 25.0 / 31.5**。  
  **结论**：在复杂双臂协作任务里，Gaussian表示的收益最明显，尤其相对点云提升最大。

**信号2：表征粒度对难任务更重要。**  
将Gaussian点数从 4096 降到 2048：
- Lift 几乎不受影响；
- Can 从 **68.0** 降到 **61.0**；
- Square 从 **47.5** 降到 **35.0**。  

**结论**：简单任务对几何分辨率不敏感，但难任务确实需要更丰富的3D细节。

**信号3：重建质量会传导到控制性能。**  
更高的3DGS重建PSNR通常带来更好的RL表现，尤其在 Square 上更明显（总体从 **42.0** 提升到 **48.5**）。  
**结论**：这个方法不是“重建和控制两张皮”；重建 fidelity 与 policy performance 存在因果关联。

**信号4：核心模块不是摆设。**
- 去掉 feature reuse 或 refinement，重建指标下降；
- 用预测深度替代真实深度，性能差距很小。  

**结论**：  
1. 级联式特征回归与GNN平滑都在起作用；  
2. 方法并不强依赖真深度传感器，这对落地是加分项。

### 证据边界
这篇论文的证据是**可信但保守地说还不算特别强**，原因有三点：

1. 主要只在 **RoboMimic** 上验证，数据域较单一；
2. 没有在统一协议下直接比较 **NeRF-based RL** 方法；
3. 作者明确说**没有为每个任务专门调最优超参**，所以结果更像“表征有效性验证”，不是最终性能上限竞争。

### 局限性
- **Fails when**: RL算法本身在超难任务上失稳时，表征改进不足以挽救整体失败；例如 Transport 上 BCQ/IQL 对所有表征都崩溃。Gaussian点数过少或重建质量偏低时，Can/Square 这类任务性能明显下滑。
- **Assumes**: 需要任务相关的预训练数据、相机内外参，以及由仿真器提供的图像-深度监督；默认场景分布在任务内相对稳定；实验依赖较强GPU资源（文中使用 A6000）。
- **Not designed for**: 无标定真实环境、强动态场景、大幅外观域偏移、需要在线探索或快速跨任务迁移的RL设置。

### 可复用组件
- **图像条件3DGS编码器**：可作为其他机器人策略或世界模型的前端状态构造器。
- **KNN-GNN高斯平滑模块**：适合多视角显式点式表示的去噪与一致性增强。
- **“重建质量 ↔ 控制性能”联动分析范式**：可复用于后续3D表征型RL论文的诊断。

## 总结一句
这篇工作的价值不在于提出了一个全新的RL算法，而在于证明了：  
**如果把3DGS从“慢重建器”改造成“快表征器”，它就能成为机器人离线RL里比图像、点云、体素更有效的3D状态接口。**

![[paperPDFs/Misc_Robotics/IROS_2024/2024_Reinforcement_Learning_with_Generalizable_Gaussian_Splatting.pdf]]