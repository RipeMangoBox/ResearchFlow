---
title: "DiffCloth: Differentiable Cloth Simulation with Dry Frictional Contact"
venue: arXiv
year: 2021
tags:
  - Others
  - task/cloth-simulation
  - projective-dynamics
  - dry-frictional-contact
  - iterative-adjoint-solver
  - dataset/White2007Flag
  - opensource/no
core_operator: 在带 Signorini-Coulomb 干摩擦接触的 Projective Dynamics 中，把接触约束隐式求导为可并行组装的接触修正项，并复用预分解全局矩阵做快速反向传播。
primary_logic: |
  当前布料状态/材料参数/控制参数 + 碰撞检测得到的接触集与局部接触坐标 →
  用含干摩擦接触的 Projective Dynamics 做前向时间积分，并把接触力对下一时刻速度的依赖显式写成局部可并行的梯度修正项 →
  复用 PD 的预分解矩阵迭代求解伴随量，得到对状态、材料、控制与设计参数的梯度，用于识别、控制和逆向设计
claims:
  - "在 48×48 的 Slope 基准上，1e-4 阈值的迭代伴随求解器相对 sparse LU 将反向传播时间最高加速 12.0× [evidence: comparison]"
  - "在该 dry-friction cloth simulator 中，接触分支切换本身保持连续但不光滑，而接触集增删是最严重的非连续来源 [evidence: analysis]"
  - "在可泛化的戴帽闭环控制训练中，基于可微仿真的 Adam 用 23,200 步仿真达到与 PPO 相近的最终损失，而 PPO 需要 1,978,000 步（约 85× 更多） [evidence: comparison]"
related_work_position:
  extends: "Projective Dynamics with Dry Frictional Contact (Ly et al. 2020)"
  competes_with: "Differentiable Cloth Simulation for Inverse Problems (Liang et al. 2019); gradSim (Murthy et al. 2021)"
  complementary_to: "N/A"
evidence_strength: moderate
pdf_ref: paperPDFs/Digital_Human_Cloth_Modelling_Draping_Simulation_and_Dressing/arXiv_2021/2021_DiffCloth_Differentiable_Cloth_Simulation_with_Dry_Frictional_Contact.pdf
category: Others
---

# DiffCloth: Differentiable Cloth Simulation with Dry Frictional Contact

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2106.05306), [DOI](https://doi.org/10.1145/3527660)
> - **Summary**: 这篇工作把布料仿真里最难微分的“频繁自接触 + 干摩擦”显式纳入可微框架，并通过复用 PD 预分解矩阵的迭代伴随求解器，让系统辨识、逆向设计和机器人辅助穿衣都能高效使用梯度优化。
> - **Key Performance**: 反向传播相对 sparse LU 最高 **12.0×** 加速；训练可泛化戴帽控制器时，相比 PPO 以约 **85×** 更少的仿真步数达到相近最终损失。

> [!info] **Agent Summary**
> - **task_path**: 当前布料状态/材料/控制参数 + 接触几何 -> 下一时刻布料状态与对参数的梯度
> - **bottleneck**: 频繁自接触与干摩擦让接触力对状态的依赖既非光滑又强耦合，导致梯度难定义、难高效求解
> - **mechanism_delta**: 把 Signorini-Coulomb 接触在固定接触分支下隐式求导为接触修正项 ΔR，并把反传改写成复用 PD 全局矩阵的迭代伴随求解
> - **evidence_signal**: 求解器基准显示反传最高 12× 加速，且多项逆问题/控制任务中梯度法显著快于 ES 与 PPO
> - **reusable_ops**: [块对角接触 Jacobian 组装, 复用预分解 PD 矩阵的迭代伴随求解]
> - **failure_modes**: [接触集频繁切换导致局部损失景观崎岖, 三角化接触面法向跳变导致状态与梯度不连续]
> - **open_questions**: [如何扩展到 vertex-face/edge-edge 自碰撞, 平滑或近似梯度是否比精确梯度更适合高频接触]

## Part I：问题与挑战

这篇论文真正要解决的，不是“如何再做一个布料前向模拟器”，而是：**如何在大量接触、摩擦和自碰撞存在时，仍然得到对材料参数、控制轨迹和设计变量有用的梯度**。

为什么这件事难：

1. **布料接触太频繁**：和很多刚体任务不同，cloth 的接触不是偶发事件，而是几乎每步都在发生，尤其是自碰撞。
2. **干摩擦是分支式约束**：Take-off / Stick / Slip 三种状态遵循 Signorini-Coulomb 定律，不像 penalty force 那样天然平滑。
3. **就算梯度存在，也很难算快**：隐式积分 + 接触约束会把反向传播变成一个大规模稀疏线性系统；直接求解代价高，抵消掉梯度优化的优势。

为什么现在值得做：
- 可微物理已经在刚体、软体、流体上展示出明显收益；
- 布料相关应用（系统辨识、辅助穿衣、逆向设计、real-to-sim）都越来越依赖**sample-efficient** 的优化；
- 但 cloth 是这些方向里“接触最密、最不光滑”的一类系统，正是过去方法的短板。

**输入/输出接口**可以概括为：
- **输入**：当前节点位置/速度、材料参数、控制/外力参数、碰撞检测得到的接触集与局部接触坐标；
- **输出**：下一时刻布料状态，以及损失对状态/材料/控制/设计变量的梯度。

**边界条件**也很明确：
- 基于 **Projective Dynamics (PD)**，所以材料模型要满足 PD 可处理的形式；
- 接触采用**节点级** dry friction 处理；
- 接触检测与局部法向/切向坐标依赖外部几何与碰撞模块；
- 若接触表面是离散三角网格而非法解析曲面，梯度质量会明显下降。

## Part II：方法与洞察

### 方法主线

**1. 前向模拟基座：沿用 Ly et al. 2020 的 PD + dry friction**
- 前向部分不是从零设计，而是建立在已有的高质量布料模拟器上；
- 其核心优点是：PD 的全局矩阵是常量，可预分解，前向求解快；
- 同时接触遵守 Signorini-Coulomb 干摩擦，比 penalty-based 接触更物理。

**2. 先复用 DiffPD 的思路处理“无接触”反传**
- 对纯 PD 部分，作者沿用 DiffPD 的思路：
  - 把反向传播写成一个伴随线性系统；
  - 利用 PD 结构，把大问题拆成“局部可并行项 + 复用全局矩阵的回代”。

**3. 再把“接触梯度”单独抽出来**
- 核心新增点在这里：
  - 对每个接触节点，先固定它当前属于 take-off / stick / slip 哪一类；
  - 把对应接触约束写成等式系统；
  - 对这些约束做隐式微分，求出“接触力如何随下一时刻速度变化”。

这一步的关键不是公式本身，而是它带来的结构变化：
- 每个接触节点只贡献一个很小的局部 Jacobian；
- 这些 Jacobian 在整体上是**块对角**的，可并行计算；
- 接触对反传系统的影响被压缩成一个附加修正项，可直接加到原有 PD 伴随求解框架里。

**4. 最后用“复用 P 的迭代伴随求解器”做高效反传**
- 反向传播不再每步都把整个大稀疏系统从头直接解一遍；
- 而是继续复用 PD 的预分解矩阵 \(P\)，只在局部加入弹性修正和接触修正；
- 如果迭代法偶尔不收敛，再回退到直接 sparse LU。

### 核心直觉

**这篇论文最关键的因果链是：**

- **改变了什么**：  
  不再把接触只当成前向里的“碰了就修正”的黑盒，而是把干摩擦接触约束本身也纳入隐式微分，并显式写成反传修正项。

- **改变了哪个瓶颈**：  
  原来“接触力对状态的影响”被埋在一个巨大、耦合、非平滑的系统里；现在它被拆成了**按接触节点局部计算**的块对角结构，再通过 PD 的常量全局矩阵统一求解。

- **带来了什么能力变化**：  
  梯度不再只在“少接触、弱接触”的软体问题里可用，而能在**contact-rich cloth** 这种高频接触场景中仍然有实用价值，从而支撑系统辨识、轨迹优化、逆向设计和闭环控制训练。

为什么这套设计有效：
1. **分支固定后，局部可微**：虽然整体接触法则不光滑，但在某个时间步、某个已激活分支内，局部约束是可微的。
2. **接触 Jacobian 很小**：每个接触节点只涉及 3×3 局部关系，计算和并行都便宜。
3. **PD 提供了一个可复用的“全局骨架”**：最贵的全局线性部分已经有预分解，反传能共享这项资产。
4. **真正最伤梯度的不是分支本身，而是接触集变化**：这也是作者后面实验分析的重点。

### 战略取舍

| 设计选择 | 解决的瓶颈 | 收益 | 代价 |
| --- | --- | --- | --- |
| 以 PD 为前向骨架 | 每步全局求解太贵 | 前向/反向都可复用预分解矩阵 | 材料模型受 PD 形式限制 |
| 节点级 Signorini-Coulomb 干摩擦 | penalty 接触不够物理、摩擦不真实 | 接触行为更接近物理约束 | 只能覆盖节点级接触，碰撞类型不完整 |
| 把接触梯度压成块对角修正项 | 频繁接触导致反传耦合严重 | 可并行、适合 contact-rich cloth | 分支切换与接触集变化处仍不光滑 |
| 迭代伴随求解 + LU 回退 | 直接 sparse LU 反传太慢 | 低精度下可获明显加速 | 无完整收敛保证，少数步需回退直接解 |

## Part III：证据与局限

### 关键实验信号

**1. 非光滑性分析：真正的问题主要来自接触集变化，而不是分支切换**
- 作者把潜在问题源拆成三类：
  1. 接触分支切换；
  2. 接触面法向/局部坐标变化；
  3. 接触集增删。
- 结论很清楚：
  - **分支切换**（stick/slip/take-off）通常是**连续但不光滑**；
  - **离散三角接触面**会因为法向跳变引入明显不连续；
  - **接触集变化**最伤，因为它会突然引入/移除冲量，造成局部损失景观变“坑洼”。

这点很重要：它解释了为什么 cloth 的梯度不是“完全没用”，但也解释了为什么在某些区域会突然变差。

**2. 反向求解器确实更快**
- 在密集接触的 Slope 基准上，48×48 网格、1e-4 阈值时，迭代伴随求解器相对 sparse LU **最高 12.0× 加速**。
- 在较宽松精度下，这个加速最明显；高精度时优势会下降，甚至在个别设置下会出现不收敛并触发回退。

**3. 梯度的“能力跃迁”体现在下游任务上**
- 在 T-shirt、Hat、Sock、Dress、Flag 等逆问题中，L-BFGS-B 借助梯度通常比 CMA-ES 和 (1+1)-ES 更快达到低损失；
- 作者还专门做了高维控制变量实验，发现**变量维数越高，梯度法优势越大**，这和可微仿真的预期一致。

**4. 对闭环控制训练也有实质价值**
- 在 Hat Controller 任务中，基于可微仿真的 Adam 达到与 PPO 相近的最终效果，但只用了 **23,200** 个仿真步；
- PPO 需要 **1,978,000** 步，约 **85×** 更多。  
这说明它不只是“做参数拟合更快”，而是真的能把复杂 cloth control 的训练效率显著拉起来。

**5. 接触模型本身比 prior differentiable cloth 更物理**
- 与 Liang et al. 2019 的 bowl 场景对比中，本文的 dry friction 接触更稳定、更物理；
- 对方方法会出现 napkin 面积突变和 popping artifact，而本文没有同样明显的伪影。

### 局限性

- **Fails when**: 接触集在相邻时间步频繁增删、接触面由离散三角片表示导致法向跳变、或高精度反传下迭代伴随求解不收敛时，梯度会变得崎岖、不稳定，甚至需要切回慢速直接求解。
- **Assumes**: 材料模型需兼容 PD；接触采用节点级 Signorini-Coulomb 模型；依赖外部碰撞检测与接触法向估计；需要预分解并存储大稀疏矩阵；正文未给出代码链接，因此复现仍依赖实现细节。
- **Not designed for**: vertex-face / edge-edge 自碰撞、超出 PD 范式的更丰富布料本构、以及无需更强风场/真实材料模型就能直接高保真 sim-to-real 的场景。

### 可复用组件

1. **接触约束隐式微分模板**：把局部接触法则写成等式，再转成块对角 Jacobian。
2. **复用预分解矩阵的伴随求解模式**：适合所有“前向已存在常量全局矩阵”的隐式物理系统。
3. **接触可微性诊断框架**：把问题拆成“分支切换 / 法向离散 / 接触集变化”三个来源来排查梯度失效点。

## Local PDF reference

![[paperPDFs/Digital_Human_Cloth_Modelling_Draping_Simulation_and_Dressing/arXiv_2021/2021_DiffCloth_Differentiable_Cloth_Simulation_with_Dry_Frictional_Contact.pdf]]