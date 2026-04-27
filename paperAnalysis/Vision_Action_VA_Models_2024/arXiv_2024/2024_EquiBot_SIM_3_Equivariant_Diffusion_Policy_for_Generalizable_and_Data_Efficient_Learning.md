---
title: "EquiBot: SIM(3)-Equivariant Diffusion Policy for Generalizable and Data Efficient Learning"
venue: CoRL
year: 2024
tags:
  - Embodied_AI
  - task/robot-manipulation
  - task/imitation-learning
  - diffusion
  - sim3-equivariance
  - vector-neuron
  - dataset/Robomimic
  - dataset/Push-T
  - repr/point-cloud
  - repr/end-effector-velocity
  - opensource/no
core_operator: "将SIM(3)等变点云编码、中心/尺度规范化与SO(3)等变扩散U-Net结合，使每一步去噪都满足几何等变性，从而输出可随场景平移、旋转、缩放一致变化的动作分布。"
primary_logic: |
  单视角对象点云与机器人本体状态 → 估计场景中心/尺度并将状态与动作规范化到规范坐标系 → 用SO(3)等变条件扩散U-Net逐步去噪动作序列 → 重缩放得到对平移、旋转、尺度具等变性的末端执行器控制
claims:
  - "EquiBot在Cloth Folding、Object Covering、Box Closing和Push T的OOD旋转/尺度/位置扰动下，相比Diffusion Policy、DP+Aug和EquivAct表现出最小的性能下降，并保持更稳定训练 [evidence: comparison]"
  - "在Robomimic的Can和Square任务中，当演示数从100降到25时，EquiBot比Diffusion Policy保留更高的在分布内性能，显示更强数据效率 [evidence: comparison]"
  - "去掉平移、旋转或尺度等变性的任一组件，都会在对应OOD设置上明显退化，说明三类对称性都对泛化能力有因果贡献 [evidence: ablation]"
related_work_position:
  extends: "Diffusion Policy (Chi et al. 2023)"
  competes_with: "Diffusion Policy (Chi et al. 2023); EquivAct (Yang et al. 2023)"
  complementary_to: "Octo (Open X-Embodiment Team et al. 2024); OpenVLA (Kim et al. 2024)"
evidence_strength: strong
pdf_ref: paperPDFs/Vision_Action_VA_Models_2024/arXiv_2024/2024_EquiBot_SIM_3_Equivariant_Diffusion_Policy_for_Generalizable_and_Data_Efficient_Learning.pdf
category: Embodied_AI
---

# EquiBot: SIM(3)-Equivariant Diffusion Policy for Generalizable and Data Efficient Learning

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2407.01479), [Project](https://equi-bot.github.io/)
> - **Summary**: 论文把 SIM(3) 等变结构直接嵌入扩散策略的每一步去噪过程，使机器人在极少示教下也能对物体位置、朝向与尺度变化做更稳健的零样本泛化。
> - **Key Performance**: 真实机器人 6 个任务中每任务仅用约 5 分钟、15 条人类示教；例如 Push Chair 在长桌/圆桌场景达 8/10、10/10，而 DP 为 0/10、0/10；Laundry Door Closing 为 8/10 vs DP 的 3/10。

> [!info] **Agent Summary**
> - **task_path**: 单视角对象点云 + 机器人本体状态 -> 多步末端执行器 6D 运动/速度与夹爪动作
> - **bottleneck**: 小样本模仿学习把“操作技能”与“场景坐标系”耦合，导致对平移/旋转/尺度变化泛化差；同时确定性等变策略难处理多模态动作与时序一致性
> - **mechanism_delta**: 将 Diffusion Policy 的编码器、FiLM 和 U-Net 逐层改造成 SIM(3)/SO(3) 等变版本，并用中心/尺度规范化让整条扩散链路按几何对称性工作
> - **evidence_signal**: 6 个仿真任务 + 6 个真实任务中，OOD 泛化、低数据学习和等变性消融都系统支持该结构偏置的有效性
> - **reusable_ops**: [centroid-scale canonicalization, equivariant conditional diffusion, vector-scalar channel routing]
> - **failure_modes**: [occlusion or large viewpoint shift corrupts point clouds, multi-object relative layout changes violate global SIM(3) assumptions]
> - **open_questions**: [how to handle nonlinear shape/dynamics shifts, how to combine equivariance with large-scale multitask robot pretraining]

## Part I：问题与挑战

这篇论文要解决的不是“如何再做一个扩散策略”，而是更具体的瓶颈：

**真实瓶颈**：在低数据模仿学习里，策略往往把“该怎么操作”与“物体当前放在哪里、朝哪边、尺寸多大”绑定在一起。  
因此即便任务本质不变，只要场景发生**平移、旋转、缩放**，普通策略就会把它当成新分布，必须靠更多示教或增强去补。

### 为什么现在值得解
- **Diffusion Policy** 已证明动作扩散对机器人控制很强：稳、能处理多模态、能预测多步动作。
- 但它的弱点也明显：**几何泛化主要靠数据覆盖或增强**，而真实家居机器人通常每个任务只有几分钟示教，根本不可能覆盖各种桌子、箱子、门、布料摆放。
- 另一方面，已有等变方法如 **EquivAct** 有几何归纳偏置，但**确定性结构**不擅长多峰动作，也不适合多步动作序列的一致性预测。

### 输入/输出接口
- **输入**：单视角、对象中心的点云观测 \(X_t\) + 机器人本体状态 \(S_t\)
- **输出**：未来一段末端执行器动作序列，包括 3D 平移/速度、旋转相关控制和夹爪开合
- **任务边界**：闭环 3D manipulation，强调对**全局几何变化**的泛化，而不是语言理解或大规模多任务策略学习

### 这篇论文针对的难点
1. **普通扩散策略**：会建模动作分布，但不自带几何一致性。
2. **数据增强路线**：能补一部分旋转/缩放，但训练更重，且不保证对未见变换稳定。
3. **已有等变策略**：有几何偏置，但对多模态和长时序动作不够友好。

---

## Part II：方法与洞察

### 方法骨架

EquiBot 的核心做法是：**不是只让输入表示等变，而是让扩散过程的每一步去噪都等变。**

1. **SIM(3)-等变点云编码器**
   - 输入场景点云
   - 输出：
     - 旋转等变特征 `ΘR`
     - 旋转不变特征 `Θinv`
     - 场景中心 `Θc`
     - 场景尺度 `Θs`

2. **中心/尺度规范化**
   - 用 `Θc` 和 `Θs` 对本体状态、动作中的位置/速度量做平移与尺度归一化
   - 这样后续网络主要处理“规范坐标系”里的动作，而不是记住具体绝对位置

3. **SO(3)-等变条件扩散 U-Net**
   - 将 noisy action、点云特征、本体状态、时间步嵌入送入条件 U-Net
   - 把原本非等变的卷积层、FiLM 层替换成等变版本
   - 通过 vector neuron 风格的向量/标量通道处理，保持旋转一致性

4. **动作反规范化**
   - 网络先预测规范坐标系中的动作
   - 最后再用 `Θs` 等量恢复到原场景尺度，得到真实控制输出

5. **理论支撑**
   - 扩散初始噪声是各向同性高斯，本身对旋转自对称
   - 若每一步马尔可夫转移都等变，则最终输出动作分布也等变
   - 这让“随机采样的动作分布”也继承几何一致性，而不只是单个确定性输出

### 核心直觉

**改了什么**：  
把“靠数据增强学几何变化”的扩散策略，改成“每一步去噪天然遵守几何对称性”的扩散策略。

**哪个瓶颈被改变了**：  
原来模型必须从数据里分别学会“同一个操作在不同位置/朝向/尺度下怎么做”；  
现在这些变化被**规范化 + 等变层**吸收，模型不再需要为每个几何变体单独记忆。

**能力上发生了什么变化**：  
- **几何泛化**：对新位置、朝向、尺度更稳
- **数据效率**：少量示教也能覆盖更多几何变化
- **多模态与时序一致性**：保留 diffusion 的优势，不像确定性 BC/等变回归那样容易平均化或抖动

### 为什么这套设计有效
- **等变性把外部扰动变成内部约束**：场景整体旋转/平移/缩放后，网络输出会同步变化，不必重新学习。
- **扩散保留多峰动作分布**：像 Push T 这类存在多条可行轨迹的任务，不会被迫回归成单一平均动作。
- **多步动作预测更连贯**：比单步回归更容易维持执行时的时序一致性和稳定性。

### 战略取舍

| 设计选择 | 带来的收益 | 代价/限制 |
| --- | --- | --- |
| 点云 + SIM(3) 等变编码 | 直接利用 3D 几何，对位姿/尺度变化更稳 | 依赖点云质量、分割和深度估计 |
| 扩散式动作序列预测 | 支持多模态、idle actions、时序一致性 | 推理需要多步去噪，时延更高 |
| 中心/尺度规范化 | 把平移和尺度变化从数据问题变成结构问题 | 假设场景能被一个全局中心/尺度良好描述 |
| 单视角人类视频示教 | 采集成本低，适合家庭任务 | 依赖手部检测、点云对齐、对象分割等外部模块 |

---

## Part III：证据与局限

### 关键证据

**1. OOD 泛化比较信号（最强主证据）**  
在 4 个仿真任务上，作者设计了 Original、OOD(R)、OOD(S)、OOD(P)、OOD(R+S+P) 多种测试：
- **DP**：原分布表现好，但一到几何变化就明显掉点
- **DP+Aug**：对训练时覆盖到的旋转增强有帮助，但对更复杂位置/联合变化仍退化明显
- **EquivAct**：部分任务可以，但在多模态 Push T 和若干复杂任务上不稳
- **EquiBot**：在各类 OOD 设置下性能下降最少，说明收益不是“扩散”或“等变”单独带来的，而是两者结合

**2. 低数据证据**  
在 Robomimic 的 Can / Square 上，把演示数从 100 降到 25：
- DP 性能下降明显
- EquiBot 仍保留更高性能  
这直接支持论文的主张：**几何等变性降低了示教覆盖需求**。

**3. 真实机器人证据**  
6 个移动操作任务、每任务 15 条人类视频示教（约 5 分钟）：
- Push Chair：长桌 8/10、圆桌 10/10；DP 为 0/10、0/10
- Laundry Door Closing：8/10；DP 为 3/10
- 多个新物体/新场景测试中，整体显著优于 DP 与 DP+Aug  
这说明方法不是只在仿真里“几何上好看”，而是真能转化成真实泛化收益。

**4. 因果消融信号**
- 去掉**平移/旋转/尺度**等变中的任一项，对应 OOD 设置性能就掉
- 把 EquivAct 粗暴接一个 diffusion head 不行，说明不是“任何等变 + diffusion”都可以
- 训练稳定性图显示其 checkpoint 波动小于 EquivAct

### 1-2 个关键指标
- **样本效率**：真实任务每个任务仅约 **5 分钟示教**
- **真实泛化差距**：Push Chair 上 **8/10, 10/10 vs DP 的 0/10, 0/10**

### 局限性

- **Fails when**: 场景有严重遮挡、视角大幅变化、点云分割错误、非线性形变/动力学变化明显、或多物体相对布局变化复杂时，策略容易失效；真实实验里也出现过夹爪开早/开晚、动作没做完整、误分割 comforter 等失败。
- **Assumes**: 可获得对象中心点云与本体状态；场景变化大体可由全局 SIM(3) 变换描述；依赖手部检测、对象分割、坐标对齐，以及文中提到的 **proprietary learned stereo-to-depth model**；真实复现还需要移动底盘、Kinova 机械臂和额外控制基础设施。
- **Not designed for**: 非线性形状变化、显式场景动力学建模、多物体关系推理、大幅跨视角鲁棒性、以及大规模多任务泛化。

### 可复用组件
- **centroid/scale canonicalization**：把几何变化转成规范坐标系问题
- **equivariant conditional diffusion**：让每一步去噪都遵守对称性
- **vector-scalar routing**：将位置/方向/标量分开处理，便于构造等变控制网络

## Local PDF reference

![[paperPDFs/Vision_Action_VA_Models_2024/arXiv_2024/2024_EquiBot_SIM_3_Equivariant_Diffusion_Policy_for_Generalizable_and_Data_Efficient_Learning.pdf]]