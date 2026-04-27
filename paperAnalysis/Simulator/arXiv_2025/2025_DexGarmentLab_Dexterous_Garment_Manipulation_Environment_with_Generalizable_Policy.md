---
title: "DexGarmentLab: Dexterous Garment Manipulation Environment with Generalizable Policy"
venue: NeurIPS
year: 2025
tags:
  - Embodied_AI
  - task/garment-manipulation
  - task/deformable-object-manipulation
  - diffusion
  - hierarchical-policy
  - dense-visual-correspondence
  - dataset/ClothesNet
  - dataset/DexGarmentLab
  - opensource/full
core_operator: "在高真实度灵巧手衣物仿真中，先用类别级稠密对应把单次示范抓取点迁移成可泛化affordance，再用结构感知扩散策略生成双臂操作轨迹"
primary_logic: |
  单次专家示范 + 当前衣物/场景点云 → GAM在新衣物上定位左右手可迁移操作点 → SADP融合衣物/交互物/环境/机器人状态生成60DoF动作序列 → 完成跨形状与形变的衣物操作
claims:
  - "HALO在14个模拟衣物操作任务上均优于DP与DP3，典型如Hang Tops达到0.92±0.04、Hang Coat达到0.90±0.01 [evidence: comparison]"
  - "去掉GAM或SADP都会显著降低成功率，例如Hang Tops从0.92降至0.64(w/o GAM)和0.70(w/o SADP)，说明目标点定位与结构感知轨迹生成两阶段都关键 [evidence: ablation]"
  - "HALO在4个真实世界任务上达到Fold Tops 13/15、Hang Tops 13/15、Wear Scarf 11/15、Wear Hat 14/15，均高于DP和DP3 [evidence: comparison]"
related_work_position:
  extends: "Diffusion Policy (Chi et al. 2023)"
  competes_with: "Diffusion Policy 3D (Ze et al. 2024); Diffusion Policy (Chi et al. 2023)"
  complementary_to: "UniGarmentManip (Wu et al. 2024)"
evidence_strength: strong
pdf_ref: paperPDFs/Simulator/arXiv_2025/2025_DexGarmentLab_Dexterous_Garment_Manipulation_Environment_with_Generalizable_Policy.pdf
category: Embodied_AI
---

# DexGarmentLab: Dexterous Garment Manipulation Environment with Generalizable Policy

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2505.11032), [Project](https://wayrise.github.io/DexGarmentLab/)
> - **Summary**: 这篇工作把“高真实度双手灵巧衣物仿真环境 + 单次示范自动采数 + 分层泛化策略 HALO”打通，核心是先找准新衣物上该抓的区域，再按衣物结构和场景生成可迁移轨迹。
> - **Key Performance**: 模拟任务中 Hang Tops 达到 **0.92±0.04**；真实任务中 Wear Hat 达到 **14/15**。

> [!info] **Agent Summary**
> - **task_path**: 单次任务示范 + 当前衣物/环境点云 + 机器人状态 -> 双臂60DoF动作序列 -> 折叠/抛展/悬挂/穿戴
> - **bottleneck**: 直接模仿学习把“抓哪里”和“怎么动”耦合在一起，面对新衣物形状/形变时既抓不准也放不准；同时缺少真实的双手灵巧衣物仿真与低成本数据来源
> - **mechanism_delta**: 把策略拆成GAM先输出可迁移操作点、SADP再按衣物结构和场景生成轨迹，并用单次示范自动扩充训练数据
> - **evidence_signal**: 14个模拟任务全量比较 + GAM/SADP消融 + 4个真实世界任务验证
> - **reusable_ops**: [类别级抓取点迁移, affordance条件扩散控制]
> - **failure_modes**: [遮挡或重度褶皱导致操作点预测偏移, 非标准或不对称衣物导致轨迹失配]
> - **open_questions**: [如何减少对手工任务序列与示范点的依赖, 如何扩展到多衣物与移动平台]

## Part I：问题与挑战

**这篇论文真正打的不是“衣物控制”本身，而是“跨衣物实例的定位+轨迹泛化”问题。**

衣物操作难，不只是因为它是可变形体，而是因为同一类别衣物在**长度、袖型、摆放、褶皱、遮挡、与环境物体相对位置**上都可能变化很大。对双臂灵巧手来说，动作空间又很高维：两只手加双臂共 **60 DoF**。这使得传统“看观测直接回归动作”的模仿学习很容易在两个地方同时失效：

1. **先天抓点错位**：新衣物上真正可操作的区域没有被显式建模。
2. **后续轨迹不自适应**：即便抓到了附近位置，也不知道该按衣长、挂点位置、人体位置去调抬升高度和放置位姿。

作者把问题拆成了三层瓶颈：

- **环境瓶颈**：现有衣物仿真要么偏简单布料，要么对灵巧手交互不真实，尤其是抓取、托举、悬挂这些接触过程。
- **数据瓶颈**：灵巧手示教很贵，靠遥操作或专家RL采集 demonstration 低效。
- **算法瓶颈**：端到端IL在新形状/新形变上很难同时学会“精确抓取区域定位”和“结构自适应轨迹”。

**为什么现在值得做？**  
因为作者把几个此前分散的条件拼起来了：Isaac Sim 可支撑更复杂的接触仿真，ClothesNet 提供了 **2500+** 高质量衣物资产，而家庭服务机器人又确实需要折叠、悬挂、穿戴这类能力。换句话说，**问题需求已经存在，数据与仿真基础设施也刚好成熟到可以系统做一轮。**

**输入/输出接口与边界条件：**

- **输入**：衣物点云、环境点云、交互物点云（可选，如 hanger/human）、机器人状态，以及单次专家示范中给出的抓点/任务序列/手型信息。
- **输出**：双臂+双灵巧手的 60DoF 动作序列。
- **任务范围**：15 个任务场景，覆盖 fling / fold / hang / wear / store。
- **泛化边界**：核心是**类别级泛化**，前提是同类别衣物共享相近结构；这不是任意拓扑跨类别零样本泛化。

---

## Part II：方法与洞察

这篇工作不是只提了一个 policy，而是搭了一个闭环系统：

1. **DexGarmentLab 环境**
2. **单次示范驱动的自动数据采集**
3. **分层泛化策略 HALO**

### 方法拆解

#### 1) 环境：让“灵巧手抓衣物”在仿真里先变得可信

DexGarmentLab 基于 Isaac Sim 4.5.0，包含：

- **2500+ garments**
- **8 个衣物类别**
- **15 个任务场景**
- 双臂 UR10e + Shadow Hand 的灵巧操作设定

环境上的关键不只是资产规模，而是**接触物理建模**。作者认为 GarmentLab 那种 attach block 方式不适合灵巧手：它会导致“碰一下就黏住”的不自然抓取。于是改为：

- particle-rigid **adhesion**
- particle-rigid **friction**
- particle-particle **adhesion/friction scale**

并按衣物类型区分：

- **PBD**：大件衣物，如 tops / dress / trousers
- **FEM**：小而相对弹性的物体，如 glove / hat

这一步的意义是：让后续自动采数和 policy 学到的，不是“假吸附轨迹”，而是更接近真实接触力学的动作分布。

#### 2) 自动采数：把单次专家示范扩成大规模 demonstrations

作者不是每个衣物都遥操作一次，而是只给**单次专家示范**，从中提取：

- demo grasp points
- demo task sequence
- demo hand grasp pose

然后用 **Garment Affordance Model (GAM)** 把示范抓点迁移到新衣物上。核心基础是：**同类别衣物在结构上具有稳定对应关系**。GAM 继承 UniGarmentManip 的 dense correspondence 思路，把“示范衣物上的关键点”映射到“新衣物当前形变下的对应点”。

得到目标点后，系统再用：

- 双臂 IK 执行任务序列
- 灵巧手 PD 控制复用示范手型
- 按衣物长度/结构与环境物体位置做简单 trajectory retargeting

这样就能在不同形状、不同摆放、不同褶皱下自动执行并采集数据。作者报告每个任务采 **100 条 demonstrations**，单条大约 **30–80 秒**。

#### 3) HALO：先定位操作区域，再生成结构感知轨迹

HALO 分两阶段：

- **Stage I: GAM**
  - 输入示范衣物点云、示范抓点、新衣物点云
  - 输出左右手目标点 affordance heatmap / target point

- **Stage II: SADP (Structure-Aware Diffusion Policy)**
  - 将衣物点云 + 左右手 affordance + 交互物点云 + 环境点云 + 机器人状态融合
  - 用 diffusion policy 生成后续动作序列

其中最关键的设计是：SADP 不只是看原始点云，而是看**带有“左右手目标区域绑定信息”的点云表示**。这让模型知道“场景里哪一块布是我要操作的那一块”。

### 核心直觉

这篇论文最重要的因果链可以概括为三条：

1. **从端到端模仿动作，改成先显式找操作点再生成轨迹**  
   → 把“目标区域定位”从高维长时序控制里剥离出来  
   → 降低了信息瓶颈  
   → 新衣物、新褶皱下更容易先抓对位置

2. **从仅看衣物外观，改成看“衣物结构 + 目标点 + 环境关系”**  
   → 轨迹生成不再是固定模板复现  
   → 可以根据衣长、袖长、hanger/pothook/human 的位置做自适应  
   → 抬升高度、前送距离、放置位置更稳

3. **从 attach-block 式伪抓取，改成基于摩擦/附着的真实接触**  
   → 训练分布更接近真实衣物交互  
   → 自动采数更可信  
   → sim-to-real gap 更容易收敛

可以用一句话总结它的能力跳变：

**它不是把扩散模型做得更大，而是把“该抓哪里”这个决定性变量显式化了。**

### 策略权衡

| 设计选择 | 改变了什么瓶颈 | 带来的能力 | 代价/风险 |
|---|---|---|---|
| GAM 显式迁移抓取点 | 抓取区域信息被埋在动作标签里 | 新形状/新褶皱下先抓对位置 | 依赖类别内结构一致，遮挡时会退化 |
| SADP 融合衣物/交互物/环境/状态 | 轨迹只会复现，不会按几何自适应 | 折叠、悬挂、穿戴时能调高度与放置位姿 | 训练更重，依赖高质量点云 |
| adhesion/friction 真实接触仿真 | attach-block 交互失真 | 采到更自然的 demonstration | PBD/FEM 仍需大量调参，无法完全真实 |
| 单次示范自动扩数 | 遥操作采数成本高 | 快速生成大规模数据 | 仍需人工定义任务序列、示范点和手型 |

---

## Part III：证据与局限

### 关键证据信号

**信号 1：跨 14 个模拟任务的一致提升，比单个 SOTA 数字更有说服力。**  
HALO 在表 2 的 14 个任务上都优于 DP / DP3。最明显的提升出现在既要求“抓准”又要求“放准”的任务上：

- **Hang Tops**: 0.92 vs 0.45 (DP) / 0.53 (DP3)
- **Hang Coat**: 0.90 vs 0.52 (DP) / 0.58 (DP3)

这说明它的优势不是某个单任务技巧，而是对**衣物结构变化 + 环境变化**的系统性适应。

**信号 2：消融结果和机制解释高度一致。**  
去掉 **GAM** 后，机器人更容易抓错区域；去掉 **SADP** 后，机器人会抓到但后续轨迹不对，比如挂衣时前送过头、折叠时位置不整齐。定量上：

- Hang Tops：**0.92 → 0.64 (w/o GAM) / 0.70 (w/o SADP)**
- Hang Coat：**0.90 → 0.62 / 0.71**

这类“方向正确的退化”比单纯涨点更能支持作者的因果叙述。

**信号 3：真实世界结果证明这不是纯仿真故事。**  
在 4 个真实任务上：

- Fold Tops: **13/15**
- Hang Tops: **13/15**
- Wear Scarf: **11/15**
- Wear Hat: **14/15**

均超过 DP / DP3。说明“先定位目标区域，再按结构生成轨迹”的设计在真实深度噪声和真实接触误差下仍然有效。

**信号 4：自动采数不是低质替代品。**  
作者还比较了自动采数 vs 遥操作数据训练 HALO 的效果，代表任务上差距很小：

- Hang Tops（sim）: **0.92 vs 0.88**
- Wear Bowlhat（sim）: **0.72 vs 0.70**
- Fold Tops（real）: **13/15 vs 13/15**

这说明自动采数确实在减少人力，而不是牺牲数据质量。

**补充信号：sim-to-real 仍然存在明显鸿沟，但可以用少量真实数据补齐。**  
纯仿真训练时：

- Hang Trousers: **53.3%**
- Wear Hat: **60.0%**

加入 **15 条真实数据**后，两者都提升到 **86.7%**。这很重要：它说明环境和策略已经具备迁移潜力，但也说明**完全零样本 sim-to-real 还不够稳**。

### 局限性

- **Fails when**: 衣物关键区域被严重遮挡、褶皱过重，或衣物结构明显偏离常规类别形状（如不对称设计、重装饰服饰）时，GAM 容易找错操作点；点云噪声较大时 sim-to-real 性能明显受影响；像 glove 这类需要精细插入的任务，当前仿真与控制仍然吃力。
- **Assumes**: 类别内衣物具有稳定结构对应；每个任务至少有一次专家示范，并且抓取点、任务序列带有人工定义成分；真实部署依赖分割/点云质量（文中用到 SAM2、RealSense/Kinect）；训练也依赖较重算力（GAM 用 RTX 4090，SADP 用到 A800 75GB 级显存）。
- **Not designed for**: 多衣物任务、带移动底盘的全身操作、跨类别拓扑零样本泛化、完全不加真实数据的高保真 sim-to-real、以及更复杂的辅助穿衣/打结类多阶段精细操作。

### 可复用组件

1. **类别级关键点迁移**：把单次示范抓点映射到新实例，可迁移到其他 deformable object manipulation。
2. **affordance-conditioned diffusion control**：先输出目标点，再做轨迹生成，适合“目标区域明确但动力学复杂”的任务。
3. **灵巧手衣物交互物理参数化**：adhesion / friction / particle-scale 这套思路对 deformable-hand simulation 很有参考价值。
4. **单次示范自动扩数 pipeline**：对高维动作空间任务很实用，尤其适合遥操作昂贵的场景。

## Local PDF reference

![[paperPDFs/Simulator/arXiv_2025/2025_DexGarmentLab_Dexterous_Garment_Manipulation_Environment_with_Generalizable_Policy.pdf]]