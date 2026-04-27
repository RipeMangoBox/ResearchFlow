---
title: "General Force Sensation for Tactile Robot"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/robot-force-estimation
  - task/robot-manipulation
  - diffusion
  - domain-translation
  - spatiotemporal-regression
  - dataset/GelSight
  - dataset/uSkin
  - dataset/TacTip
  - repr/marker-image
  - opensource/promised
core_operator: "把多种触觉传感器的原始信号统一成 marker 图像，用条件扩散完成 marker-to-marker 形变迁移，再结合材料先验补偿和时序回归，把源传感器的力感知能力迁移到目标传感器。"
primary_logic: |
  已标定源传感器的触觉序列与力标签 + 目标传感器少量位置配对/参考图像
  → 统一为 marker 表示并进行条件扩散 M2M 翻译，再用材料硬度先验修正迁移标签
  → 训练目标传感器的时空力回归器，输出三轴力并用于抓取/防滑控制
claims:
  - "在 132 组仿真传感器组合上，M2M 将平均 FID 从大于 400 降到 4、平均 KID 从大于 0.75 降到 0.01，说明跨 marker 图案的形变翻译可稳定对齐源/目标图像分布 [evidence: comparison]"
  - "在 5 个同构 GelSight 传感器组合上，GenForce 将法向力最大误差压到 1 N 以下、三轴 R² 平均提升到 0.8 以上；例如 C-II→D-I 的法向误差从 4.8 N 降到 0.96 N [evidence: comparison]"
  - "在异构 GelSight/uSkin/TacTip 迁移中，加入材料补偿后 6 组组合的平均法向 MAE 低于 0.92 N，且 uSkin→TacTip 的法向 MAE 从 7.76 N 降到 0.52 N（93% 改善） [evidence: comparison]"
related_work_position:
  extends: "TransForce (Chen et al. 2024)"
  competes_with: "Touch-to-Touch Translation (Grella et al. 2024); ACROSS (El Amri et al. 2024)"
  complementary_to: "EasyCalib (Li et al. 2024); In-situ Mechanical Calibration (Zhao et al. 2023)"
evidence_strength: moderate
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_General_Force_Sensation_for_Tactile_Robot.pdf
category: Embodied_AI
---

# General Force Sensation for Tactile Robot

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [Project Page](https://zhuochenn.github.io/genforce-project/), [Under Review](https://www.researchsquare.com/article/rs-6513579/v1)
> - **Summary**: GenForce 将不同触觉传感器的观测统一到 marker 形变空间，并通过条件扩散翻译 + 材料补偿，把一个已标定传感器的力觉能力迁移到其他未标定传感器，从而避免为每个新传感器重复采集大规模力标签。
> - **Key Performance**: 132 组仿真翻译中平均 FID 从 >400 降到 4、KID 从 >0.75 降到 0.01；异构传感器迁移后 6 组组合平均法向 MAE <0.92 N，uSkin→TacTip 从 7.76 N 降到 0.52 N。

> [!info] **Agent Summary**
> - **task_path**: 已标定源触觉序列 + 目标传感器少量位置配对图像 -> 目标传感器三轴力估计 -> 抓取/防滑控制
> - **bottleneck**: 跨触觉传感器同时存在表征域差异（模态/图案/密度/照明/曲率）与材料力学差异（硬度/摩擦/迟滞），导致源域力标签不能直接复用
> - **mechanism_delta**: 先把多模态触觉信号统一成 marker 图像，再用条件扩散做 M2M 形变迁移，并用材料先验修正迁移标签，最后用时空回归器输出三轴力
> - **evidence_signal**: 132 组仿真 + 74 组真实组合中翻译指标显著下降，且异构 uSkin→TacTip 的法向 MAE 从 7.76 N 降至 0.52 N
> - **reusable_ops**: [统一marker表示, 条件扩散M2M翻译, 材料先验补偿, 时空力回归]
> - **failure_modes**: [GelSight在大接触面积下会因弹性体上浮产生闪烁, 参考marker过小或过敏会使异构数组翻译不稳, 静载时存在零点漂移与迟滞]
> - **open_questions**: [能否摆脱任何已标定源传感器与材料先验, 能否扩展到大面积电子皮肤和无显式taxel的EIT传感器]

## Part I：问题与挑战

这篇稿件正文题为 *Training Tactile Sensors to Learn Force Sensing from Each Other*，核心方法名为 **GenForce**。它解决的不是“单个触觉传感器如何做力回归”这个局部问题，而是更关键的系统问题：

**如何让一个传感器学到的力觉，被另一个结构不同、材料不同、甚至模态不同的传感器复用。**

### 真正的瓶颈是什么？

真正瓶颈不在于回归器容量不够，而在于**校准数据无法跨传感器复用**。  
机器人手上的触觉传感器往往来自不同技术路线：

- 视觉式：GelSight、TacTip
- 阵列式/磁式：uSkin
- 它们还会有不同的 marker 图案、密度、照明、曲率、最大压入深度、软皮肤硬度

因此，同一个接触事件在不同传感器上会呈现出完全不同的原始信号分布。更糟的是，力标签获取依赖昂贵的 F/T 传感器与机械化采集流程，而软弹性体又会老化、磨损、替换后失配，导致每换一个传感器都要**重新采力标签、重新训练**。

### 为什么现在值得解决？

因为机器人操作已经从“单指感知”走向“多指/多部位大规模部署”：

- 多指灵巧手需要多个传感器协同控力
- 真实场景中传感器会频繁替换和老化
- 触觉控制越来越需要跨平台、跨模态复用，而不是一次性定制校准

所以现在最需要的，不是再做一个更准的单传感器 force regressor，而是建立一种**可迁移、可扩展、低标注成本**的触觉力感知基础设施。

### 输入 / 输出接口与边界条件

**输入：**

- 一个或多个已标定源传感器的触觉序列 + 力标签
- 目标传感器的少量**位置配对**图像
- 目标传感器的非接触参考图像
- 材料先验（硬度/力-深度曲线）

**输出：**

- 目标传感器的三轴力估计
- 基于该估计的抓取控力、滑移检测与避滑控制

**边界条件：**

- 目标传感器必须能被转换成某种 **marker 表示**
- 仍然需要至少一个**已校准源传感器**
- 仍然需要少量位置配对数据，而不是完全零样本
- 需要材料先验来补偿硬度差异

---

## Part II：方法与洞察

### 设计哲学

GenForce 的关键不是做一个端到端“大一统黑盒”，而是把跨传感器迁移拆成三个因果上更清晰的问题：

1. **先统一“看见的是什么”**：不同模态的原始信号先转成统一的 marker 形变表示  
2. **再统一“形变长什么样”**：把源传感器的形变翻译成目标传感器风格  
3. **最后修正“相同形变对应多大力”**：用材料先验补掉不同软皮肤的力学差异

这比直接做 feature alignment 更扎实，因为作者认为跨传感器误差里混着两类不同问题：

- **表征差异**：图像风格、marker 分布、阵列分辨率、曲面/平面
- **力学差异**：硬度、摩擦、迟滞、加载/卸载曲线

如果只对齐 feature 而不处理材料差异，目标域上的“图像像了”，**标签仍然是错的**。

### 方法主线

#### 1. 统一 marker 表示

作者把不同传感器的原始信号都变成 marker-based image：

- **GelSight / TacTip**：通过 marker segmentation 提取 marker
- **uSkin**：把 4×4 taxel 的三轴电信号转换为 marker 位移/尺寸变化图像

这一步的意义是：  
把“视觉图像、磁阵列、曲面触觉”都投到一个可比较的**形变载体**上。

#### 2. Marker-to-Marker Translation（M2M）

有了统一表示后，源传感器的接触形变就能通过**条件扩散模型**翻译成目标传感器风格：

- 输入：源传感器的变形 marker 图像
- 条件：目标传感器的参考图像（未接触）
- 输出：保留源域接触形变、但呈现目标域 marker 风格的图像

作者强调这是一个**many-to-many** 框架：  
不是每对传感器训练一个独立 translator，而是用单个条件模型根据 reference image 选择目标风格。

#### 3. 时空力回归

翻译后的目标风格序列，再交给一个时空力回归器：

- RAFT 特征编码
- ConvGRU 捕获时间依赖
- ResNet 分层建模
- MLP 输出三轴力

这里作者显式采用**序列建模**，原因是软皮肤存在迟滞：  
同样的单帧图像，在 loading / unloading 阶段可能对应不同力值，单帧预测先天有歧义。

#### 4. 材料补偿

即使形变图像翻译得很好，**软硬不同的皮肤仍然会把“同样深度的形变”映射成不同大小的力**。  
因此作者又加了一层材料补偿：

- 先测每个皮肤的力-深度关系
- 再根据 loading / unloading 阶段修正转移后的力标签
- 剪掉由硬度和摩擦差异带来的标签偏置

这一步是整篇论文最关键的“物理补丁”：  
它把图像翻译问题和力学标签问题拆开处理，明显提升了异构迁移精度。

### 核心直觉

**what changed：**  
作者没有再把“跨传感器力估计”视为直接的 domain adaptation regression，而是改成  
**统一形变表示 → 翻译目标形变外观 → 修正材料力学映射 → 再做时序力回归**。

**which bottleneck changed：**

- 原始多模态差异 → 被压缩到统一 marker 空间
- 传感器外观/布局差异 → 由条件扩散翻译处理
- 材料硬度/摩擦/迟滞差异 → 由材料先验和时序建模处理

**what capability changed：**  
从“每个新传感器都要单独采大量 force labels”  
变成“只要少量位置配对数据 + 一个已标定源传感器，就能给目标传感器装上可用的力觉”。

### 为什么这个设计有效？

因为“力”本质上不是视觉外观，而是**形变历史在特定材料模型下的结果**。

- marker 表示保住了最关键的**几何形变信息**
- M2M 把“这个形变如果发生在目标传感器上会长什么样”补出来
- 材料补偿把“这个形变在目标材料上应该对应多大力”补回来
- 时空模型则处理“同样形变但不同历史”的迟滞问题

也就是说，作者不是单纯在做风格迁移，而是在做一个分层的**感知-力学解耦**。

### 战略权衡表

| 设计选择 | 解决的瓶颈 | 带来的能力 | 代价 / 边界 |
| --- | --- | --- | --- |
| 统一 marker 表示 | 多模态原始信号无法直接对齐 | 视觉式与阵列式都能进入同一翻译管线 | 依赖稳定 marker 提取或 signal-to-marker 转换 |
| 条件扩散 M2M | marker 图案、密度、曲率、照明差异大 | 少量位置配对数据即可生成目标域接触图像 | 极端图案/尺寸差时可能出现闪烁或不稳 |
| 材料补偿 | 软硬度和摩擦差导致“标签错域” | 异构/跨材料迁移显著变准 | 需要材料先验与加载/卸载建模 |
| 时空力回归 | 单帧无法解决迟滞歧义 | 动态接触、滑移检测更稳 | 训练与部署更依赖时间同步和序列质量 |

---

## Part III：证据与局限

### 关键证据信号

**Signal 1｜仿真跨图案翻译有效**  
在 12 类 marker 图案构成的 **132 组仿真组合**上，M2M 把平均 FID 从 >400 降到 4，把平均 KID 从 >0.75 降到 0.01。  
这说明它至少在“形变保真 + 目标风格对齐”这一层是成立的，而不是只在少数组合上碰巧有效。

**Signal 2｜同构 GelSight 迁移从不可用变成可用**  
在 5 个 GelSight 传感器组合上，source-only 方法法向误差可超过 4.8 N，很多组合 R² 为负；加入 GenForce 后，法向最大误差降到 1 N 以下，三轴 R² 平均到 0.8 以上。  
这直接支持作者的主张：**同类传感器也不能简单共享 force model，必须做显式迁移。**

**Signal 3｜材料补偿不是锦上添花，而是必要项**  
作者用 7 种硬度不同的硅胶皮肤测试，显示仅做图像迁移后，硬度差越大，力误差越大；加入材料补偿后，法向误差在 hard-to-soft 方向平均下降约 30%，在 soft-to-hard 方向平均下降约 16%。  
这说明跨传感器迁移里真正难的不只是“图像像不像”，还包括“标签是不是还物理正确”。

**Signal 4｜异构迁移是最关键的能力跃迁**  
在 GelSight / uSkin / TacTip 六组异构迁移中，加入材料补偿后全部 R² 变成正值，平均法向 MAE 低于 0.92 N。  
最强信号是 **uSkin→TacTip**：法向 MAE 从 7.76 N 下降到 0.52 N。  
这基本说明方法确实跨过了“不同模态、不同曲率、不同硬度”的最难门槛。

**Signal 5｜机器人案例证明它不是离线玩具**  
作者把迁移后的模型部署到真实抓取与滑移控制中：

- 9 个未见日常物体抓取
- 目标力 0.6–1.2 N
- 4 个物体的滑移检测与避滑

这些实验更像**系统级 case study**，不是标准 benchmark，但足以说明模型输出已达到闭环控制可用水平。

### 1-2 个最关键指标

- **翻译层面**：平均 FID >400 → 4，KID >0.75 → 0.01  
- **异构力估计层面**：uSkin→TacTip 法向 MAE 7.76 N → 0.52 N

### 局限性

- **Fails when:** 大接触面积导致 GelSight 弹性体上浮时，目标域图像会出现 flicker；异构数组转换中如果 reference marker 过小或灵敏度设置不当，翻译会不稳；静态加载下存在零点漂移和迟滞；uSkin 在金属物体附近可能受磁干扰。
- **Assumes:** 至少存在一个 fully calibrated 源传感器；目标域仍需少量位置配对图像和非接触 reference；需要材料先验做硬度补偿；传感器必须能被稳定转换为 marker 表示。复现还依赖较重硬件链条：ATI Nano17、机械臂/移动平台、以及文中用于 M2M 训练的 1×A100 80GB。代码与数据目前仅承诺在录用后公开。
- **Not designed for:** 完全零样本、无任何校准源的迁移；追求超过专门监督校准方法的极致精度；超大面积薄膜电子皮肤或无显式 taxel 的 EIT 传感器。

### 可复用组件

1. **统一 marker 表示**：适合作为跨触觉模态的公共中间层  
2. **条件扩散 M2M 翻译器**：可复用于跨传感器 deformation transfer  
3. **材料先验补偿**：适合任何“图像能对齐但力学不对齐”的跨域触觉任务  
4. **时空触觉回归头**：不仅能做 force，还可能迁移到姿态估计、3D 重建等任务

**一句话评价：**  
这篇论文最有价值的地方，不是又做了一个更准的 tactile force regressor，而是把“跨触觉传感器迁移”拆成了**形变表示、样式翻译、材料修正、时序回归**四个因果清晰的模块，因此真正把“力觉可以从一个传感器教给另一个传感器”变成了可操作的工程范式。

## Local PDF reference

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_General_Force_Sensation_for_Tactile_Robot.pdf]]