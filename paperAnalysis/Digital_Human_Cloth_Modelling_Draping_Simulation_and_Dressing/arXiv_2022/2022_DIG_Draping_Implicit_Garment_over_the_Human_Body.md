---
title: "DIG: Draping Implicit Garment over the Human Body"
venue: arXiv
year: 2022
tags:
  - Others
  - task/3d-garment-draping
  - task/clothed-human-reconstruction
  - implicit-surface
  - differentiable-skinning
  - interpenetration-regularization
  - dataset/CLOTH3D
  - repr/SMPL
  - repr/SDF
  - opensource/full
core_operator: 以隐式SDF表示服装，并学习受SMPL形状与姿态条件控制的服装专属可微蒙皮场，再用防穿插预处理与顺序约束实现稳定披挂
primary_logic: |
  规范空间服装隐变量 z + 身体形状/姿态参数 (β, θ) → 用隐式SDF重建服装薄壳，并通过共享体积权重预测服装蒙皮、形状位移和姿态位移，同时施加人体穿插惩罚与壳层顺序约束 → 输出贴合目标人体、穿插更少且可对 β/θ/z 联合反传优化的服装网格
claims:
  - "通过防穿插预处理，DIG 在 CLOTH3D 的衬衫/裤子重建中将 IR 从约 18% 降到 0%，同时保持或提升几何精度 [evidence: ablation]"
  - "在 CLOTH3D TEST HARD 上，DIG 对衬衫和裤子的变形均优于 SMPLicit 与 DeePSD，例如衬衫 ED/IR 为 26.5mm/3.0%，对比 SMPLicit 的 35.4mm/12.9% 与 DeePSD 的 95.6mm/46.4% [evidence: comparison]"
  - "在合成图像拟合中，DIG 的衬衫 CD/IR 为 4.69×10^-4/3.9%，显著优于带平滑后处理的 SMPLicit 的 18.73×10^-4/37.3% [evidence: comparison]"
related_work_position:
  extends: "SMPLicit (Corona et al. 2021)"
  competes_with: "SMPLicit (Corona et al. 2021); DeePSD (Bertiche et al. 2021)"
  complementary_to: "VPoser (Pavlakos et al. 2019); FrankMocap (Rong et al. 2020)"
evidence_strength: moderate
pdf_ref: paperPDFs/Digital_Human_Cloth_Modelling_Draping_Simulation_and_Dressing/arXiv_2022/2022_DIG_Draping_Implicit_Garment_over_the_Human_Body.pdf
category: Others
---

# DIG: Draping Implicit Garment over the Human Body

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2209.10845) · [Code](https://github.com/liren2515/DIG)
> - **Summary**: 这篇工作把“任意拓扑服装的隐式表示”与“对任意3D点学习的可微服装蒙皮场”结合起来，并显式处理人体穿插与壳层自交，因此既能更稳定地披挂衣服，也能把梯度回传到人体参数，实现人体与服装的联合优化。
> - **Key Performance**: CLOTH3D TEST HARD 上衬衫变形 ED/IR = 26.5mm/3.0%，优于 SMPLicit 的 35.4mm/12.9%；合成图像拟合中衬衫 CD/IR = 4.69×10^-4/3.9%，优于 SMPLicit 的 18.73×10^-4/37.3%。

> [!info] **Agent Summary**
> - **task_path**: 规范空间隐式服装 + SMPL 身体参数(β, θ) / 单张图像分割 -> 披挂后的服装网格与可联合优化的人体-服装参数
> - **bottleneck**: 最近身体顶点的离散蒙皮不可微且过平滑；隐式薄壳膨胀后易与人体或自身发生穿插
> - **mechanism_delta**: 把离散最近点蒙皮改为连续的服装专属体积蒙皮场，并加入防穿插预处理、人体SDF惩罚和壳层顺序损失
> - **evidence_signal**: CLOTH3D 上对 SMPLicit/DeePSD 的多指标比较 + 预处理/Lgrad/Lorder 消融，其中重建 IR 从约18%降到0
> - **reusable_ops**: [implicit-SDF-thin-shell, volumetric-skinning-field]
> - **failure_modes**: [未见过的服装类别或多层服装, 图像分割与人体初始化误差会传递到拟合结果]
> - **open_questions**: [如何引入物理约束而不破坏端到端可微性, 能否用统一模型覆盖更多服装类别与更强自接触]

## Part I：问题与挑战

这篇论文要解决的**真实瓶颈**，不是“怎样表示一件衣服”这么简单，而是：

1. **服装拓扑是可变的**  
   模板网格方法通常绑定固定拓扑，换一类衣服就要换模板和变形器，不适合真实应用里的“任意衣服”。

2. **衣服必须跟着身体动，但又不能只是复制身体的运动**  
   先前隐式方法 SMPLicit 虽然能表示任意拓扑，但它给每个服装点分配蒙皮权重时，依赖“找最近的人体顶点”。这一步是离散搜索：
   - 不可微，难以做端到端反传；
   - 动力学表达弱，容易过平滑。

3. **隐式服装为了可网格化，常被膨胀成薄壳，这会引入新的几何问题**  
   开口服装本来不是 watertight surface，于是常用“加厚”方法转成闭合薄壳再做 SDF。问题是：衣服本来就贴着身体，一膨胀就容易直接穿进身体。

4. **逆问题需要联合优化人体与服装**  
   从图像恢复 clothed human 时，理想情况应同时更新：
   - 人体形状 `β`
   - 人体姿态 `θ`
   - 服装隐变量 `z`  
   但如果披挂过程本身不可微，人体参数就很难被服装观测有效纠正。

### 输入/输出接口

- **输入 1**：规范空间中的服装隐变量 `z`，以及 SMPL 身体形状/姿态 `(β, θ)`
- **输出 1**：披挂到目标人体上的服装网格
- **输入 2**：单张图像的服装分割（外加人体初始化）
- **输出 2**：联合优化后的人体与服装参数

### 论文的边界条件

这套方法并不是“通用服装物理模拟器”，而是在以下设定内工作：

- 服装先被放到**规范空间 / T-pose / 中性身体**下建模；
- 训练数据来自 **CLOTH3D**；
- 实验中分别为**衬衫**和**裤子**训练独立模型；
- 图像拟合依赖外部分割与人体初始化工具。

一句话概括：**真正难点是让任意拓扑服装的披挂过程既可微、又少穿插、还能用于逆优化。**

## Part II：方法与洞察

DIG 的整体思路是：  
**先在规范空间用隐式场表示衣服，再学习一个对任意3D点都可查询的服装专属蒙皮场，把衣服从规范空间连续地带到目标身体上。**

### 方法骨架

#### 1. 用隐式 SDF 表示任意拓扑服装

作者沿用了 DeepSDF/SMPLicit 这一路线：  
用一个神经网络表示服装表面的 SDF，隐变量 `z` 控制具体衣服形状。

但服装是开口的，不是闭合曲面，所以他们采用“**薄壳膨胀**”技巧，把衣服变成一个有厚度的 watertight shell，再做 SDF 重建。这样做的好处是：

- 拓扑可以变；
- 表示可微；
- 可以通过 Marching Cubes 提取网格。

#### 2. 不再用“最近身体顶点”，而是学习连续的体积蒙皮场

这是本文最关键的一步。

先前 SMPLicit 的问题在于：  
服装点怎么跟人体骨架关联，是通过最近身体顶点的 blending weights 决定的。这个最近点查找是离散的，因此：

- 梯度不连续；
- 服装动态只能粗糙地“贴身跟随”。

DIG 改成学习一个**对任意空间点 `x` 都能输出权重分布的函数 `w(x)`**。  
这个 `w(x)` 不是直接从零学，而是建立在 SMPL 先验上，把身体网格顶点权重扩展到整个空间。随后它同时用于：

- 生成服装的 skinning weights；
- 生成由身体形状引起的服装 shape displacement。

此外，再单独学习一个与姿态相关的 pose displacement 网络，补充细节变形。

结果就是：  
**服装不再“抄最近人体点的答案”，而是通过一个连续场来决定如何被身体驱动。**

#### 3. 把“穿插问题”前移到训练中处理

作者发现，如果只把服装做成膨胀薄壳再训练，人体穿插会直接进入训练数据，模型就会把这种错误几何学会。

因此 DIG 做了两层处理：

- **数据预处理层**：先把膨胀后落到人体内部的壳层点推回到人体表面外侧，得到干净训练目标；
- **训练约束层**：在变形时用人体 SDF 惩罚掉入人体内部的点。

这样做的因果关系很直接：  
如果训练目标本身是“合法几何”，模型更容易学到低穿插的披挂结果。

#### 4. 用顺序约束防止薄壳翻面自交

由于服装是“有厚度的壳”，会存在内外两层。  
若只优化变形误差，原来靠近身体的一层，在变形后可能跑到外层之外，导致自交或翻面。

DIG 引入一个**ordering loss**，约束壳层中原本更靠近身体的点，变形后仍应保持相对靠近。  
这不是完整物理模拟，但足够抑制薄壳表示特有的几何伪影。

### 核心直觉

**变化点**：  
从“离散最近点蒙皮”改成“连续可学习的体积蒙皮场”，再把“穿插/壳层翻转”从后处理问题改成训练时的几何约束。

**改变了什么瓶颈**：  
- 把梯度通路从离散、断裂，变成连续、可回传；
- 把服装几何合法性从“事后修补”变成“训练目标的一部分”；
- 把服装对人体的依赖从“最近邻拷贝”变成“带 SMPL 先验的连续插值”。

**能力上带来的变化**：  
- 能处理任意拓扑服装；
- 服装变形更细致、不过度平滑；
- 可以从图像里联合优化人体和服装，而不只优化衣服。

更直白地说：  
**这篇论文真正拧动的“因果旋钮”是：把服装披挂从一个离散几何流程，改造成一个连续、受约束的场函数。**

### 战略性权衡

| 设计选择 | 解决的瓶颈 | 能力收益 | 代价/权衡 |
|---|---|---|---|
| 隐式 SDF 薄壳表示 | 固定模板无法覆盖变拓扑 | 任意拓扑、可微重建 | 依赖膨胀厚度 `ϵ` 与 meshing 分辨率，开口服装被近似成闭合薄壳 |
| 基于 SMPL 先验的体积蒙皮场 | 最近点蒙皮不可微且过平滑 | 可微披挂、对未见姿态更稳 | 仍强依赖 SMPL 身体先验质量 |
| 防穿插预处理 + 人体SDF惩罚 | 训练目标与结果都易穿进身体 | 显著降低 IR，结果更适合下游网格处理 | 需要准确的人体表面与 canonical 对齐 |
| 壳层顺序损失 | 薄壳内外层翻转、自交 | 减少自交伪影 | 只约束几何顺序，不等价于真实布料物理 |

## Part III：证据与局限

### 关键证据信号

- **Ablation signal — 防穿插预处理确实在解决真实问题**  
  在 CLOTH3D 的服装重建中，不做预处理时衬衫/裤子 IR 大约在 18% 左右；加入预处理后 IR 直接到 0。  
  这说明作者不是只“换了个表示”，而是抓住了隐式薄壳最致命的几何缺陷。

- **Comparison signal — 连续蒙皮场比 SMPLicit / DeePSD 更稳、更准**  
  在 TEST EASY 和 TEST HARD 上，DIG 都在 ED、NC、IR 上更优。  
  尤其 TEST HARD 更能说明问题：DeePSD 在未见分布上退化明显，而 DIG 因为复用了 SMPL 的结构先验，泛化更稳。  
  代表性结果：TEST HARD 衬衫 ED/IR 为 **26.5mm / 3.0%**，优于 SMPLicit 的 **35.4mm / 12.9%**。

- **Inverse-fitting signal — 端到端可微带来更好的图像拟合**  
  合成图像拟合中，DIG 的衬衫 CD/IR 为 **4.69×10^-4 / 3.9%**，显著优于 SMPLicit 的 **18.73×10^-4 / 37.3%**。  
  这支持了论文的核心主张：**如果披挂过程本身可微，就能避免在 canonical/posed 空间之间做近似投影，从而减少拟合误差。**

- **Case-study signal — 真实图像上可以联合修正人体与服装**  
  真实图像实验主要是定性展示，但能看出：当初始人体估计不准时，DIG 还能继续更新 `β, θ, z`，而 SMPLicit 只能固定身体、调衣服。

### 局限性

- **Fails when**: 分布外服装类别、复杂多层穿搭、强烈布料自接触/衣物间接触时，当前的薄壳顺序约束可能不够；图像端若分割噪声或人体初始化偏差大，拟合结果也会受影响。  
- **Assumes**: 需要带 SMPL 参数和服装几何的 3D 数据集（本文主要是 CLOTH3D）；依赖 SMPL 身体先验、canonical 化流程、Marching Cubes 的近似梯度；并且为不同服装类别分别训练模型。  
- **Not designed for**: 高保真物理模拟、统一单模型覆盖所有服装类别、从原始 RGB 直接端到端恢复而不借助外部分割/人体估计器。

### 复现与可扩展性上的现实约束

- **数据依赖**：需要有规范空间服装与 SMPL 身体对应关系的数据；
- **外部依赖**：真实图像拟合用到 FrankMocap、人体解析模型、VPoser 和可微渲染器；
- **表示依赖**：Marching Cubes 本身并不严格可微，论文使用的是近似梯度；
- **类别拆分**：衬衫和裤子分别建模，尚未证明一个统一模型能稳定覆盖更多服装类型。

### 可复用组件

这篇论文最值得迁移的，不只是“一个新模型”，而是几种很通用的操作：

1. **implicit-SDF-thin-shell**  
   适合把开口表面转成可微隐式表示。

2. **volumetric-skinning-field with body prior**  
   适合任何“非刚体表面跟随人体骨架，但不能直接用最近点蒙皮”的任务。

3. **interpenetration-aware preprocessing**  
   当训练目标本身有几何错误时，先清洗目标再训练，往往比单纯加 loss 更有效。

4. **ordering loss for shell consistency**  
   对“带厚度壳层”的隐式/显式几何变形都很有参考价值。

![[paperPDFs/Digital_Human_Cloth_Modelling_Draping_Simulation_and_Dressing/arXiv_2022/2022_DIG_Draping_Implicit_Garment_over_the_Human_Body.pdf]]