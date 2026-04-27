---
title: "Two by Two: Learning Multi-Task Pairwise Objects Assembly for Generalizable Robot Manipulation"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/object-assembly
  - task/robot-manipulation
  - se-3-equivariance
  - two-step-prediction
  - cross-object-fusion
  - dataset/2BY2
  - opensource/partial
core_operator: 将日常成对装配拆成“先接收件B、后配合件A”的两步规范空间SE(3)位姿估计，并用B的不变几何特征条件化A的预测
primary_logic: |
  两个随机姿态的成对物体点云 + 任务定义的 canonical space
  → 双尺度 SE(3) 等变编码先估计接收件 B 的规范位姿，再将 B 的 SO(3) 不变特征与 A 的等变特征融合
  → 输出两物体装配到 canonical pose 所需的 6D 位姿
claims:
  - "2BY2 构建了包含 1,034 个对象实例、517 对日常物体和 18 个细粒度装配任务的数据集，并提供位姿与对称性标注，显式覆盖日常 pairwise assembly 场景 [evidence: analysis]"
  - "该两步 SE(3) 位姿估计方法在 2BY2 的 18 个细粒度任务及 ALL 设置上均优于 Jigsaw、Puzzlefusion++、Neural Shape Mating 和 SE(3)-Assembly；在 ALL 上达到 0.110 translation RMSE 与 41.44 rotation RMSE [evidence: comparison]"
  - "双尺度等变编码与 two-step 分解都是关键因子：ALL 上去掉 two-step 后性能退化到 0.139/45.20，换成单尺度 VN DGCNN 退化到 0.123/44.67 [evidence: ablation]"
related_work_position:
  extends: "SE(3)-Assembly (Wu et al. 2023)"
  competes_with: "SE(3)-Assembly (Wu et al. 2023); Neural Shape Mating (Chen et al. 2022)"
  complementary_to: "AnyGrasp (Fang et al. 2023); Edge Grasp Network (Huang et al. 2023)"
evidence_strength: moderate
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Two_by_Two_Learning_Multi_Task_Pairwise_Objects_Assembly_for_Generalizable_Robot_Manipulation.pdf
category: Embodied_AI
---

# Two by Two: Learning Multi-Task Pairwise Objects Assembly for Generalizable Robot Manipulation

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2504.06961) · [Project](https://tea-lab.github.io/TwoByTwo/)
> - **Summary**: 论文提出首个面向日常成对物体装配的大规模数据集 2BY2，并把装配建模为“先定位接收件、再条件化定位配合件”的两步 SE(3) 等变位姿估计，从而提升跨任务、跨形状泛化。
> - **Key Performance**: 2BY2 的 ALL 任务上达到 **0.110 translation RMSE / 41.44 rotation RMSE**；真实机器人四类任务平均成功率 **77.5%**，而 SE(3)-Assembly 为 **22.5%**。

> [!info] **Agent Summary**
> - **task_path**: 成对物体点云 / 随机初始姿态 → canonical space 下两物体的装配 SE(3) 位姿
> - **bottleneck**: 现有方法把日常装配当成几何碎片匹配，且常联合预测两物体位姿，忽略了“B 主要由世界基准决定、A 才依赖 B 几何/功能约束”的非对称结构
> - **mechanism_delta**: 把联合装配回归改成 B→A 的顺序位姿分解，并用双尺度 SE(3) 等变编码与跨物体特征融合来降低姿态扰动带来的学习难度
> - **evidence_signal**: 在 18 个细粒度任务、3 个跨类别任务和 ALL 设置上全线领先，并有 two-step/encoder 消融与真实机器人验证
> - **reusable_ops**: [B先行规范化, 双尺度SE(3)等变点云编码]
> - **failure_modes**: [高精度插入或弱几何线索场景对小角度误差很敏感, 测试时B的预测误差仍会传递到A]
> - **open_questions**: [如何扩展到多零件长时序装配, 如何把抓取与装配策略联合而非手工拼接]

## Part I：问题与挑战

这篇论文真正想解决的，不是“把碎片拼回原物体”，而是**家居环境中的日常成对装配**：比如插头插入插座、花插进花瓶、面包放进烤面包机、钥匙插进锁孔。  
这类任务对机器人更有现实意义，但也比传统 fracture assembly 更难，因为它不仅需要几何对齐，还需要**功能性和空间关系**对齐。

### 1) 真问题是什么？

现有装配数据集和方法，大多围绕：

- 断裂碎片重组
- 工业/规则零件装配
- 单一任务、弱日常语义场景

但日常 pairwise assembly 有三个额外难点：

1. **功能约束强于纯几何相似性**  
   例如花和花瓶、信件和邮箱，不是表面局部最像就能装进去，而是要满足“可插入/可容纳/可覆盖”的功能关系。

2. **两物体的因果角色不对称**  
   在很多任务里，接收件 B（如插座、花瓶、烤面包机）的位置更多由世界基准或 canonical space 决定；  
   配合件 A（如插头、花、面包）才需要根据 B 的几何和姿态决定最终位姿。  
   所以“同时预测 A 和 B”其实不符合任务结构。

3. **泛化目标是真实日常物体，而不是同分布碎片**  
   测试集包含未见过的新几何形状，要求模型学到的是**装配约束**，不是记住对象模板。

### 2) 真正的瓶颈在哪里？

我认为这篇论文瞄准了三个核心瓶颈：

- **数据瓶颈**：此前缺少大规模、日常化、带 pose/symmetry 标注的 pairwise assembly benchmark。
- **建模瓶颈**：联合预测两物体位姿会引入不必要的耦合误差。
- **表示瓶颈**：输入点云有任意旋转/平移扰动，普通点云网络需要用更多数据去“记住姿态变化”。

### 3) 输入/输出接口与边界条件

- **输入**：两个点云 \(P_A, P_B\)，各为 `(1024, 3)`，来自两个待装配物体；输入时带随机 SO(3) 旋转并平移到质心。
- **输出**：两个物体装配到预定义 canonical pose 所需的 SE(3) 位姿。
- **边界条件**：
  - 只处理**刚体、二元成对装配**
  - 依赖任务定义好的 **canonical space**
  - 依赖对称性标注来避免把等价旋转误判为错误
  - 默认输入已经是分离好的点云，而不是从原始 RGB 场景端到端感知

### 4) 为什么现在值得做？

因为家居机器人正在从“抓取/搬运”走向**接触密集型操作**，而装配正是代表性能力；但如果没有日常任务 benchmark，模型就只能在几何碎片数据集上“看起来会拼”，却很难迁移到真实机器人。

---

## Part II：方法与洞察

这篇论文的方法主线很清楚：

> **先把装配问题改写成对 shared canonical space 的顺序位姿估计，再用等变表示降低姿态扰动难度。**

### 方法主线

#### 1. 用 canonical space 统一任务目标

作者没有直接学“A 相对 B 的位姿”，而是给每个任务定义一个**标准世界坐标系**：

- 大多数任务：物体在重力作用下稳定放在 XY 平面，最低点对齐 `Z=0`
- Plug 任务：插座放在 XZ 平面，模拟墙面

这样做的好处是：不同任务都能转成“预测物体应该如何对齐到人类世界中的标准状态”。

#### 2. 先预测接收件 B，再预测配合件 A

网络被拆成两个分支：

- **Branch B**：只看 \(P_B\)，预测接收件 B 的 pose
- **Branch A**：再结合已经对齐后的 B 与 \(P_A\)，预测 A 的 pose

这对应论文最关键的结构性判断：

- B 的 pose 更像是“世界对齐问题”
- A 的 pose 才是“相对 B 的装配问题”

这比同时预测 A/B 更符合任务因果结构。

#### 3. 双尺度 SE(3) 等变编码器

作者用的是改造后的 **two-scale Vector Neuron DGCNN**：

- VN 层负责处理旋转等变
- 通过点云中心化处理平移
- 双尺度 KNN 分支同时抓取：
  - 局部装配接触线索
  - 全局形状和朝向线索

这相当于把“姿态变化”从学习负担里部分拿掉，让网络更关注**哪个几何结构决定装配**。

#### 4. 跨物体融合：用 B 的不变特征条件化 A

在 A 分支中：

- 从 B 提取 **SO(3) 不变特征** \(IB\)
- 从 A 提取 **SE(3) 等变特征** \(EA\)
- 通过逐点乘法做融合

核心目的不是简单拼接信息，而是：

- 把 B 的几何约束注入 A 的预测
- 同时尽量保持 A 特征的旋转等变性

这很适合“插入孔位/容器口径/槽位方向”等条件化装配判断。

#### 5. 训练策略：分开训练，测试时串联

作者没有端到端联训两个分支，而是：

- 训练 A 分支时，给它 **canonical pose 下的 B**
- 测试时，才使用 B 分支的预测结果

这是一种很实用的工程决策：先避免 joint training 时两边误差互相污染。

### 核心直觉

**这篇论文真正改变的，不只是网络骨架，而是任务的概率分解方式。**

#### what changed → bottleneck changed → capability changed

- **What changed**  
  从“同时回归两个物体的位姿”，改成“先回归 B，再在 B 条件下回归 A”。

- **Which bottleneck changed**  
  原先模型要直接学习一个高度耦合的联合位姿分布；现在变成：
  1. 先解决更稳定的 B 对齐问题  
  2. 再解决依赖 B 几何/姿态的 A 装配问题  
  同时用 SE(3) 等变表示吸收输入姿态扰动。

- **What capability changed**  
  模型不再主要依赖记忆特定对象组合，而更像是在学习“接收件决定装配约束”的结构规律，因此对**未见形状、跨任务混合训练、真实机器人执行**更稳。

#### 为什么这个设计有效？

因为在日常 pairwise assembly 里，**装配约束本身就是非对称的**。  
如果仍然像碎片重组那样联合预测，模型会把很多本不该耦合的误差耦合在一起；而作者的 B→A 分解，相当于把任务重写成更符合真实世界因果关系的形式。

### 战略性 trade-off

| 设计选择 | 带来的收益 | 代价 / 风险 |
|---|---|---|
| 定义 canonical space | 把多任务装配统一成标准 pose 预测，便于跨任务学习 | 依赖人工定义世界先验；不同行业/场景可能要重设 canonical rule |
| B→A 两步预测 | 降低联合回归耦合，符合接收件先确定的任务结构 | 测试时 B 的误差仍会传播到 A |
| 双尺度 VN-DGCNN | 同时建模局部接触几何与全局形状，且对姿态更鲁棒 | 比 PointNet/DGCNN 更复杂，计算成本更高 |
| 用 B 的不变特征条件化 A | 把装配约束显式注入 A，同时保留等变性 | 交互建模较轻量，可能不如更强 cross-attention 丰富 |
| 分开训练 BA / BB | 减少联合训练误差污染 | 训练/测试存在分布差：训练时 BA 看到的是 GT B，测试看到的是预测 B |

---

## Part III：证据与局限

整体看，这篇论文的证据链是完整的：**新 benchmark + 多基线比较 + 关键消融 + 真实机器人验证**。  
但按保守标准，它的核心量化证据仍主要来自**单一新数据集 2BY2**，所以我会把证据强度记为 **moderate**。

### 关键证据信号

- **比较信号：18 个细粒度任务全胜**  
  方法在 2BY2 的 18 个细粒度任务上都优于 Jigsaw、Puzzlefusion++、NSM、SE(3)-Assembly。  
  这说明收益不是某个单类任务的偶然，而是覆盖 lid covering / inserting / high precision placing 三大类。

- **跨任务泛化信号：ALL 任务仍明显领先**  
  在最难的 ALL 设置中，论文报告 **0.110 translation RMSE / 41.44 rotation RMSE**；  
  对比 SE(3)-Assembly 的 **0.233 / 52.34**，说明方法不仅会做单任务，还能在混合任务和未见几何上维持稳定。

- **几何对齐信号：Chamfer Distance 也同步改善**  
  Appendix 中 ALL 的 CD 从 SE(3)-Assembly 的 **0.679** 降到 **0.268**，支持“姿态更准”确实转化成“几何装配更贴合”。

- **消融信号：能力提升来自结构改动，而非单纯换 backbone**  
  去掉 two-step 后，ALL 从 **0.110 / 41.44** 退化到 **0.139 / 45.20**；  
  换成单尺度 VN DGCNN 也退化到 **0.123 / 44.67**。  
  这比较直接地支持了论文的核心因果旋钮：**顺序分解 + 双尺度等变编码**。

- **真实机器人信号：从“离线 pose 预测”走到“可执行装配”**  
  在 Cup / Flower / Bread / Plug 四类真实任务上，成功率从基线 **22.5%** 提升到 **77.5%**。  
  这说明它不是只在 synthetic benchmark 上好看，至少能支撑一个真实执行 pipeline。

### 能力跃迁到底体现在哪里？

相对 prior work，最大的能力跳跃不是“数值再好一点”，而是：

1. **从碎片重组走向日常功能装配**
2. **从联合 pose 拟合走向因果分解的装配建模**
3. **从单任务几何匹配走向跨任务、跨形状泛化**

也就是说，这篇论文把“assembly”从视觉几何问题，往**机器人可用的结构化 manipulation subproblem** 推进了一步。

### 局限性

- **Fails when**: 接收件的几何线索弱、重复或容错极低时，方法对小姿态误差依然敏感；从结果看，像 Children’s Toy、Tissue 这类任务仍明显更难，说明在弱约束或高精度接触场景下鲁棒性有限。

- **Assumes**: 假设输入是已分离好的双物体点云、对象是刚体、任务有可定义的 canonical pose，并且有对称性标注；数据集构建依赖大量人工 mesh 清洗、配对和标注。训练时 A 分支看到的是 canonical B，而测试时使用预测 B，存在 train-test mismatch。真实机器人实验还依赖 **UR5 + Robotiq 2F-85**、点云扫描和**手工设计的抓取位姿**，这会影响完全复现与扩展。

- **Not designed for**: 多零件长时序装配、原始 RGB 场景下的端到端感知-规划-控制、可变形/铰接对象装配，以及联合抓取与装配策略学习。

### 复用价值高的组件

- **canonical-space 装配建模**：把相对装配问题改写成统一世界基准下的 pose 预测
- **B→A 顺序分解**：适用于“接收件先定、配合件后对齐”的大量 manipulation 子任务
- **双尺度 SE(3) 等变点云编码**：可迁移到其他 6D pose / manipulation 场景
- **不变条件化融合**：把 context object 的几何约束注入 target object，同时尽量保持 target 的等变性
- **2BY2 benchmark**：可作为日常装配泛化能力的标准测试床

### 一句话结论

这篇论文最值得记住的不是“又一个装配网络”，而是它抓住了**日常 pairwise assembly 的非对称因果结构**：  
**先把接收件对齐到人类世界，再在其几何约束下对齐配合件。**  
这个分解让装配从“难学的联合配准”变成“更可泛化的结构化 pose estimation”。

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Two_by_Two_Learning_Multi_Task_Pairwise_Objects_Assembly_for_Generalizable_Robot_Manipulation.pdf]]