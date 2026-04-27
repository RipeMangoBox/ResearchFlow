---
title: "HOOD: Hierarchical Graphs for Generalized Modelling of Clothing Dynamics"
venue: CVPR
year: 2023
tags:
  - Others
  - task/cloth-simulation
  - graph-neural-network
  - hierarchical-graph
  - physics-based-loss
  - dataset/AMASS
  - dataset/VTO
  - opensource/full
core_operator: 在服装-人体网格上构建嵌套层级图，用多层消息传递高效传播布料长程弹性信号，并以物理增量势自监督预测顶点加速度。
primary_logic: |
  人体姿态序列 + 服装网格/材质/拓扑 → 构建带 body edges 的层级嵌套图并执行多层消息传递 → 预测每个服装顶点加速度并自回归生成动态服装形变
claims:
  - "在同样 15 次消息传递预算下，HOOD 将单步有效传播半径从 15 条边提升到 48 条边，并在 all-garments 设置中把 Ltotal 从 1.68 降到 1.07，同时保持 13.6 fps 推理速度 [evidence: ablation]"
  - "在 30 人感知实验中，HOOD 相比 SNUG 和 SSCH 在大多数视频对比中被认为更真实，并与物理模拟器 ARCSIM 的偏好基本持平 [evidence: comparison]"
  - "单个训练好的 HOOD 网络可泛化到训练未见服装，并支持在推理时改变尺码、材料参数，以及通过启停边实现拓扑变化 [evidence: case-study]"
related_work_position:
  extends: "MeshGraphNets (Pfaff et al. 2020)"
  competes_with: "SNUG (Santesteban et al. 2022); SSCH (Santesteban et al. 2021)"
  complementary_to: "N/A"
evidence_strength: moderate
pdf_ref: paperPDFs/Digital_Human_Cloth_Modelling_Draping_Simulation_and_Dressing/CVPR_2023/2023_HOOD_Hierarchical_Graphs_for_Generalized_Modelling_of_Clothing_Dynamics.pdf
category: Others
---

# HOOD: Hierarchical Graphs for Generalized Modelling of Clothing Dynamics

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2212.07242) · [Project](https://dolorousrtur.github.io/HOOD/)
> - **Summary**: 这篇论文把服装动力学从“对特定衣服做蒙皮/顶点偏移回归”改成“在服装-人体层级图上学习局部物理更新”，从而让单个模型就能实时泛化到未见服装、不同材质和拓扑变化。
> - **Key Performance**: 单步传播半径 48 edges（单层基线为 15）；all-garments 消融中 HOOD 为 13.6 fps、Ltotal=1.07（Fine15 为 13.1 fps、1.68）

> [!info] **Agent Summary**
> - **task_path**: 人体姿态序列 + 服装网格/材质/拓扑 -> 逐帧服装顶点加速度/动态网格
> - **bottleneck**: 单层局部消息传递无法在单步内传播布料的刚性拉伸波，导致长程耦合缺失与过度拉伸；同时基于蒙皮的服装专用回归难以泛化到新衣服
> - **mechanism_delta**: 将服装网格递归粗化为嵌套层级图，在相邻层同时做消息传递，并用物理增量势进行无监督训练
> - **evidence_signal**: 同为 15 步消息传递时，层级图把有效传播半径从 15 提升到 48，并显著降低物理损失；感知实验优于 SNUG/SSCH
> - **reusable_ops**: [nested-hierarchical-graph, incremental-potential-self-supervision]
> - **failure_modes**: [missing-self-collision-and-garment-garment-interaction, high-velocity-body-motion-out-of-distribution]
> - **open_questions**: [how-to-learn-self-collision-without-gt-simulation, how-to-extend-to-multi-layer-clothing]

## Part I：问题与挑战

这篇论文要解决的核心问题是：**给定人体形状、动作序列和任意服装网格，如何在实时预算下生成看起来物理合理的服装动态**。

### 1. 真实难点不只是“预测褶皱”，而是“传播长程耦合”
服装和一般局部变形物体不同。布料的拉伸往往很硬，某处受力后，影响会非常快地传到远处区域。  
因此，真正的瓶颈不是局部几何细节本身，而是：

- **长程耦合传播太慢**：普通 GNN 每一步只看局部邻域，消息传播半径有限；
- **布料会被预测得像橡皮筋**：如果传播不够快，远端区域来不及响应，容易出现过拉伸和“rubbery”动态；
- **不能简单靠更多步数解决**：步数拉高会明显拖慢推理，而且衣服网格大小并不固定，难以提前设定一个总是足够大的步数。

### 2. 现有学习方法还有另一个瓶颈：表示方式错了
很多已有方法本质上仍依赖 **linear blend skinning (LBS)**，把衣服形变主要看成身体姿态的函数。  
这类方法对紧身衣有效，但对长裙、宽松连衣裙、下摆摆动这类**不严格贴身**的服装就不够自然。

更关键的是，很多方法是**garment-specific** 的：网络输出固定数量的顶点偏移，换一件衣服就要重训。  
这使它们很难处理：

- 新服装形状
- 新拓扑
- 推理时修改材质
- 宽松/自由摆动服装

### 3. 输入/输出接口与边界
**输入**：
- 服装网格
- 人体网格/姿态序列
- 每个局部区域的材料参数
- 可动态启停的图边（用于纽扣、拉链等拓扑变化）

**输出**：
- 每个服装顶点的加速度
- 通过自回归积分得到下一时刻服装位置与速度

**边界条件**：
- 主要建模的是**衣服-人体交互**
- 不处理完整的**衣服自碰撞/衣服-衣服交互**
- 训练不需要离线布料 GT 轨迹，但依赖物理能量项设计与身体模型

---

## Part II：方法与洞察

HOOD 的方法可以概括成三件事：

1. **把服装动力学改写成图上的局部更新问题**  
2. **用层级图解决局部 GNN 的长程传播不足**  
3. **用物理增量势做自监督训练，摆脱 GT 模拟轨迹**

### 方法骨架

#### A. 图表示：从“衣服模板回归”变成“局部动力学推理”
作者把服装建成图：

- 服装顶点 = 图节点
- 服装网格边 = 图边
- 额外加入 body nodes / body edges，用于表达服装与人体的接触关系

节点特征包含：
- 节点类型（服装/身体）
- 当前速度
- 法向
- 质量
- 材料参数

边特征包含：
- 当前相对位置
- canonical/rest 状态下的相对几何

网络最终输出的是**每个服装顶点的加速度**，而不是某件固定衣服的顶点偏移。  
这一步很关键，因为它把学习目标从“记住一件衣服怎么变形”改成“学会局部结构和材料怎样决定下一步动力学”。

#### B. 层级图：用粗层传远距信号，用细层保留褶皱
作者递归粗化服装图，得到一个**嵌套层级图**。核心性质是：

- 粗层节点是细层节点的子集
- 不同层共享节点特征
- 各层有各自的边特征与边更新 MLP

消息传递时，不是只在细图上跑，而是**同时处理多个层级**。  
这样带来的效果是：

- **粗层**：快速传播长程拉伸/约束信息
- **细层**：保留局部折痕、皱褶和接触细节

作者使用近似 UNet 的 3 层结构，在相邻两层上同时消息传递。  
论文给出的直观结果是：**在类似计算预算下，有效传播半径从 15 提升到 48 条边**。

#### C. 自监督训练：不依赖 GT 服装轨迹
训练时，HOOD 不使用真实或离线模拟得到的服装 GT 序列，而是最小化一个**物理增量势**。  
它把以下因素揉在一起：

- stretching
- bending
- gravity
- friction
- collision
- inertia

直观理解：  
网络只要预测出一个“下一步服装状态”，如果这个状态在这些物理约束下更合理，损失就更低。

这使模型能直接学到：

- 材料参数如何影响动态
- 衣服与人体碰撞和摩擦如何影响运动
- 不同网格/拓扑下的局部动力学规律

而且训练时会随机采样：
- SMPL body shape
- garment template
- 尺码扰动
- 材料参数

所以模型学到的是**局部规律**，不是某个固定服装实例。

### 核心直觉

过去的方法卡住，根源在于两个信息瓶颈：

1. **表示瓶颈**：如果输出是固定服装的顶点偏移，模型天然难泛化到新服装；
2. **传播瓶颈**：如果消息只能在平面细图上慢慢扩散，布料那种“近乎全局、立即发生”的拉伸耦合学不出来。

HOOD 的关键改变是：

- **把预测对象改成局部加速度**
- **把单层图改成嵌套层级图**
- **把监督信号改成物理能量一致性**

对应的因果链是：

**局部动力学表示**  
→ 模型不再绑定某件衣服的顶点编号  
→ 对新服装形状、大小、局部材质更稳健

**层级消息传递**  
→ 单步有效感受野显著增大  
→ 刚性拉伸波能更快传播  
→ 少步数下也不容易出现 rubbery overstretching

**物理自监督**  
→ 不需要昂贵 GT 服装序列  
→ 可以直接在材料、碰撞、摩擦约束下学习  
→ 训练覆盖更多衣服/材质/形状组合

### 战略权衡

| 设计选择 | 解决的瓶颈 | 获得的能力 | 代价/妥协 |
|---|---|---|---|
| 层级嵌套图 + 多层消息传递 | 单层 GNN 长程传播太慢 | 同样步数下更大传播半径，更自然的宽松服装动态 | 需要预计算图粗化结构 |
| 预测局部加速度而非固定顶点偏移 | garment-specific 表示难泛化 | 可迁移到未见服装、尺寸变化、拓扑变化 | 需要自回归推理，误差会累积 |
| 物理增量势自监督 | 依赖离线 GT 模拟数据 | 无需 GT 服装轨迹，材料和接触可一起学 | 物理项设计决定了能学到的现象边界 |
| body-edge 图增强 | 仅靠姿态条件不足以表达接触 | 更直接地建模衣服-身体接触与摩擦 | 依赖最近体表对应，极端姿态下会出错 |

---

## Part III：证据与局限

### 关键证据

#### 1. 感知对比信号：比现有学习法更真实
作者做了一个 30 人感知实验，把 HOOD 与 SNUG、SSCH、以及物理模拟器 ARCSIM 做视频两两对比。  
主要结论不是“略好一点”，而是：

- 对 **SNUG / SSCH**：用户明显更偏好 HOOD
- 对 **ARCSIM**：偏好基本接近持平

这说明 HOOD 至少在视觉 realism 上，已经接近物理模拟器，同时明显超过先前学习方法。

#### 2. 架构消融信号：层级结构不是装饰，而是直接解决传播半径问题
最关键的消融是和单层 MeshGraphNet 风格结构比较：

- **Fine15**：15 步单层消息传递
- **Fine48**：48 步单层消息传递
- **Ours**：15 步，但带层级传播

结论非常清楚：

- 相比 **Fine15**：HOOD 在几乎相同速度下显著降低物理损失
- 相比 **Fine48**：HOOD 质量接近，但速度明显更快

最能体现“因果有效性”的指标是 all-garments 设置：
- HOOD：**13.6 fps**, **Ltotal = 1.07**
- Fine15：13.1 fps, 1.68
- Fine48：4.99 fps, 1.04

这说明收益来自**更好的传播机制**，而不是单纯“算得更多”。

#### 3. 泛化信号：一个网络处理新衣服、改材质、改拓扑
论文展示了多个非标准场景：

- 训练未见服装
- 从扫描中提取的真实服装网格
- 同一件衣服在推理时改尺码
- 局部多材料服装
- 通过启停边实现“解扣子/系扣子”拓扑变化

这组证据说明 HOOD 的泛化来源确实是**局部图动力学建模**，而不是模板记忆。

### 局限性

- **Fails when**: 需要处理服装自碰撞、服装-服装交互、多层穿搭时；人体运动速度明显超过训练分布时；身体本身出现严重自相交、导致最近 body correspondence 错误时。
- **Assumes**: 输入有明确的服装网格、身体网格和局部材料参数；训练动作分布主要来自 AMASS/SMPL；通过最近身体顶点连边近似建模接触；训练虽不需要离线 GT 布料模拟，但依赖人工设计的物理项与图构建流程。
- **Not designed for**: 自动解决多层服装穿插、精确连续碰撞检测、复杂远程自碰撞发现与解析、任意失真/严重异常人体网格上的鲁棒仿真。

### 复现与可扩展性判断

好的地方：
- 代码和模型公开，复现门槛相对低；
- 不依赖闭源仿真 GT 数据，监督成本更低；
- 组件化清晰，很多模块可迁移到其他 mesh-based 物理建模任务。

真正限制扩展的点：
- 自碰撞仍然是核心短板；
- 目前更像“高质量单层服装动力学器”，不是完整多层服饰系统；
- 性能和稳定性仍与训练运动分布、身体对应关系质量强相关。

### 可复用组件

- **层级嵌套图构建**：适合任何需要长程信号快速传播的 mesh/GNN 物理问题  
- **跨层共享节点的消息传递**：避免手工设计插值/聚合算子  
- **基于增量势的自监督训练**：适合缺少 GT 动态轨迹、但有物理结构先验的任务  
- **body-edge 增强图表示**：适合接触驱动的人体-物体交互建模

## Local PDF reference

![[paperPDFs/Digital_Human_Cloth_Modelling_Draping_Simulation_and_Dressing/CVPR_2023/2023_HOOD_Hierarchical_Graphs_for_Generalized_Modelling_of_Clothing_Dynamics.pdf]]