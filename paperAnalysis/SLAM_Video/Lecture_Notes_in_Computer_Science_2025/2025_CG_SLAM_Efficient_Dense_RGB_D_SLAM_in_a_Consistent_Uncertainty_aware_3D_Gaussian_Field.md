---
title: "CG-SLAM: Efficient Dense RGB-D SLAM in a Consistent Uncertainty-aware 3D Gaussian Field"
venue: arXiv
year: 2024
tags:
  - 3D_Gaussian_Splatting
  - task/rgb-d-slam
  - gaussian-splatting
  - uncertainty-modeling
  - gpu-rasterization
  - dataset/Replica
  - dataset/TUM-RGBD
  - dataset/ScanNet
  - opensource/promised
core_operator: "通过各向同性正则、深度对齐/方差约束与高斯不确定性筛选，把原本偏重渲染的3D Gaussian Splatting改造成适合RGB-D SLAM的稳定一致场。"
primary_logic: |
  RGB-D序列 → 以深度初始化并增量维护3D高斯集合，GPU栅格化渲染颜色/α深度/中值深度/不透明度/不确定性
  → 用各向同性正则、α深度-中值深度对齐、几何方差约束和不确定性筛选，把高斯压到稳定表面附近
  → 基于低不确定性高斯做全图位姿优化与滑窗BA，输出相机轨迹和致密地图
claims:
  - "在Replica数据集上，CG-SLAM将平均ATE RMSE降至0.27 cm，优于SplaTAM的0.36 cm、GS-SLAM的0.50 cm和Point-SLAM的0.54 cm [evidence: comparison]"
  - "在Replica重建评测中，CG-SLAM取得平均1.01 cm的最佳Accuracy，优于Point-SLAM的1.26 cm和Vox-Fusion的1.88 cm，但Completion（2.84 cm）弱于Co-SLAM（2.08 cm） [evidence: comparison]"
  - "去掉关键几何约束会显著恶化结果：移除Liso会在Replica场景中出现跟踪失败，而同时移除Lalign与Lvar会把平均Chamfer Distance从3.85 cm升高到4.79 cm，并把RMSE从0.26 cm升到0.33 cm [evidence: ablation]"
related_work_position:
  extends: "3D Gaussian Splatting (Kerbl et al. 2023)"
  competes_with: "SplaTAM (Keetha et al. 2023); GS-SLAM (Yan et al. 2024)"
  complementary_to: "N/A"
evidence_strength: strong
pdf_ref: paperPDFs/SLAM_Video/Lecture_Notes_in_Computer_Science_2025/2025_CG_SLAM_Efficient_Dense_RGB_D_SLAM_in_a_Consistent_Uncertainty_aware_3D_Gaussian_Field.pdf
category: 3D_Gaussian_Splatting
---

# CG-SLAM: Efficient Dense RGB-D SLAM in a Consistent Uncertainty-aware 3D Gaussian Field

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2403.16095), [Project](https://zju3dv.github.io/cg-slam/)
> - **Summary**: 这篇论文把原本更适合新视角合成的3D Gaussian Splatting，改造成一个几何一致、带深度不确定性感知的RGB-D SLAM场，从而同时提升跟踪稳定性、建图精度和在线效率。
> - **Key Performance**: Replica平均ATE RMSE 0.27 cm；轻量版系统在Replica Office0上达到15.4 FPS。

> [!info] **Agent Summary**
> - **task_path**: RGB-D序列 / 静态室内场景 -> 相机轨迹 + 致密3D高斯地图/网格
> - **bottleneck**: 原始3DGS虽快，但几何约束弱且易各向异性过拟合，导致高斯不贴表面、位姿梯度不稳；NeRF-SLAM则受体渲染速度限制
> - **mechanism_delta**: 在3DGS渲染链路中加入各向同性正则、α深度-中值深度对齐、深度方差约束与高斯不确定性筛选，并用GPU位姿求导支撑全图直接跟踪
> - **evidence_signal**: 三个RGB-D基准上的跨数据集跟踪领先，以及几何损失/不确定性模块的消融验证
> - **reusable_ops**: [各向同性高斯正则, 深度不确定性驱动的高斯筛选]
> - **failure_modes**: [动态物体场景, 大面积未观测区域补全较弱]
> - **open_questions**: [如何压缩高斯场内存, 如何扩展到动态SLAM与更强回环机制]

## Part I：问题与挑战

### 1. 这篇论文真正要解决的是什么问题？
目标是**在线 dense RGB-D SLAM**：输入RGB-D视频流，输出相机轨迹和可用于重建/渲染的致密3D地图。

表面上看，瓶颈是“**NeRF-SLAM太慢**”；但论文指出更深层的问题其实是：

1. **NeRF式体渲染太贵**  
   只能采样有限射线做跟踪与建图，难以利用整张图像的结构信息，也往往需要很多优化步数才能避免局部最优。

2. **直接把3DGS搬进SLAM又不够稳**  
   原始3D Gaussian Splatting是为逼真渲染设计的，不是为位姿优化设计的。它容易学出：
   - 强各向异性的细长高斯；
   - 不严格贴合真实表面的高斯分布；
   - 对输入视角过拟合、对外推视角/新位姿不稳定的几何结构。

3. **SLAM真正依赖的是“可优化的几何一致性”**  
   对SLAM来说，光是渲染快还不够；更关键的是：  
   **渲染梯度必须和真实表面几何一致**，这样相机位姿优化才稳定。

### 2. 输入/输出接口与边界条件
- **输入**：静态室内场景的RGB-D序列
- **输出**：
  - 相机轨迹
  - 3D Gaussian地图
  - 由TSDF-Fusion提取的网格重建
- **边界条件**：
  - 依赖深度输入，不是单目SLAM
  - 主要面向静态场景
  - 采用滑窗优化，但不是完整长期回环系统

### 3. 为什么现在值得解决？
因为3DGS提供了一个非常有吸引力的新拐点：  
它能用**栅格化而不是体渲染**做可微渲染，从根上降低渲染开销。问题变成：

> 能不能把“为渲染而生”的3DGS，改造成“为跟踪和建图而稳”的SLAM表示？

CG-SLAM的回答是：可以，但必须给3DGS补上**几何一致性约束**和**不确定性筛选机制**。

---

## Part II：方法与洞察

### 方法总览
CG-SLAM把场景表示为一组3D高斯，每个高斯带有：
- 位置
- 尺度
- 旋转
- 不透明度
- SH颜色系数
- 额外引入的**不确定性属性**

系统通过自定义GPU栅格器，从当前高斯场中渲染出：
- 颜色图
- α-blending深度
- median depth
- opacity
- uncertainty map

然后将这些渲染量分别用于：
- **mapping**：更新高斯属性，让地图更稳、更贴表面
- **tracking**：直接优化当前相机位姿
- **sliding BA**：联合优化局部关键帧位姿与地图

### 核心直觉
**论文最关键的改动不是“把NeRF换成3DGS”，而是把3DGS的最优解空间收紧到“适合SLAM”的那一类高斯场。**

具体因果链是：

- **原来**：3DGS只需把训练视角渲染对  
  → 会接受很多“看起来能解释图像、但几何并不稳定”的细长/漂浮高斯  
  → 位姿优化时梯度噪声大、容易漂

- **现在**：加入各向同性、表面对齐、深度方差与不确定性筛选  
  → 高斯分布被压到更接近真实表面的区域，异常原语被降权  
  → 相机位姿优化面对的是一个**更平滑、更一致、更可信**的能量面

- **能力变化**：  
  从“快但不稳的渲染表示”  
  变成“能支撑全图直接跟踪的几何表示”

### 关键模块

#### 1. 面向SLAM的GPU栅格化与位姿求导
论文先对3DGS中相机位姿的导数做了完整分析，并实现了面向SLAM的CUDA框架。  
这一步的意义不是“数学更漂亮”，而是让**tracking/mapping可以基于整图重渲染误差做高效优化**，而不再受NeRF体渲染速度掣肘。

作者还发现：在高斯场中做位姿优化时，**Lie algebra表示**对旋转更有利。

#### 2. 各向同性正则：先把“箭头状伪几何”压下去
原始3DGS容易长出很细长的各向异性椭球。  
对新视角合成，这有时是“好事”；但对SLAM，这意味着：
- 高斯可能不贴表面
- 图像边缘易出现“箭头状”伪影
- 跟踪优化更容易被错误几何牵引

所以CG-SLAM引入**scale isotropy regularization**，鼓励高斯从“过扁/过长的椭球”回到更接近球形的分布。

这改变的是**形状先验**：  
不再允许高斯仅靠极端形状去拟合单视角外观，而必须以更稳定的几何方式解释场景。

#### 3. α深度 vs 中值深度对齐：让高斯更贴真实表面
作者观察到：只用α-blending depth做监督，并不能强力约束高斯“到底落在哪个真实表面上”。

所以他们额外渲染了**median depth**，并最小化：
- α深度
- 中值深度

两者的差异。

直觉上，这相当于要求：
- 一个像素最主要贡献的高斯，
- 不只是“平均意义上解释深度”，
- 而是要更靠近真正的前景表面位置。

这一步改变的是**几何锚定方式**：  
把原本偏“软平均”的深度解释，拉向更“表面中心化”的分布。

#### 4. 方差损失与不确定性建模：只信稳定高斯
CG-SLAM把深度误差看成一种不确定性信号。

- **像素级不确定性**：  
  看α-blending深度相对观测深度的方差
- **高斯级不确定性**：  
  聚合该高斯在多个关键帧“主导像素”上的深度偏差

然后把高不确定性的高斯降到低opacity，而不是立刻硬删除。  
这样做的效果是：
- 先减少它们对跟踪/建图的干扰
- 同时保留二次被优化修正的机会

这一步改变的是**优化时的信息选择**：  
从“所有高斯都平等参与”  
变成“优先相信稳定、信息量高的高斯”。

#### 5. Tracking + Sliding BA
跟踪阶段：
- 用常速度模型给当前位姿初始化
- 采用颜色+几何重渲染误差直接优化位姿

局部BA阶段：
- 在共视关键帧窗口内联合优化位姿与高斯属性
- 共视关键帧通过**NetVLAD描述子相似度**来选，而不是传统视锥重叠启发式

这相当于在系统层面继续减少：
- 误匹配关键帧
- 局部累计误差

### 战略性权衡

| 设计 | 改变的约束/信息流 | 带来的能力 | 代价 |
|---|---|---|---|
| 各向同性正则 | 限制极端各向异性高斯 | 跟踪更稳，伪影更少 | 可能牺牲少量纯渲染拟合自由度 |
| α深度-中值深度对齐 | 强化“高斯要贴表面”的约束 | 重建更准，位姿梯度更可靠 | 需要额外深度渲染量与超参 |
| 深度方差 + 高斯不确定性筛选 | 过滤低价值/高噪声原语 | 提升跟踪效率与抗异常性 | 阈值过严可能伤害薄结构/边缘 |
| 轻量版半分辨率 | 降低计算量 | FPS显著提升 | 精度略有下降 |
| 3DGS栅格化替代NeRF体渲染 | 从射线采样切到整图渲染 | 实时性更强 | 需要更高显存存储高斯属性 |

---

## Part III：证据与局限

### 关键证据信号

#### 1. 比较信号：跨数据集跟踪结果说明“稳态高斯场”确实能改善位姿优化
- **Replica**：平均ATE RMSE **0.27 cm**，优于  
  - SplaTAM 0.36 cm  
  - GS-SLAM 0.50 cm  
  - Point-SLAM 0.54 cm
- **TUM-RGBD**：平均 **4.0 cm**，优于  
  - SplaTAM 5.48 cm  
  - Co-SLAM 8.38 cm
- **ScanNet**：平均 **8.08 cm**，优于  
  - Co-SLAM 9.37 cm  
  - SplaTAM 11.88 cm

**结论**：改造后的高斯场不仅在合成数据上有效，也能迁移到真实噪声深度、模糊和反光更明显的场景。

#### 2. 比较信号：重建更准，但补洞不如NeRF式表示
- 在Replica上，CG-SLAM取得最佳平均**Accuracy 1.01 cm**
- 但**Completion 2.84 cm**弱于Co-SLAM的2.08 cm

**结论**：  
CG-SLAM更擅长“已观测区域的表面精度”，但由于3DGS本身缺少NeRF/MLP那样的连续场外推能力，对未观测区域的补全较弱。

#### 3. 效率信号：速度提升主要来自栅格化与原语筛选
- 完整版系统：**8.5 Hz**
- 轻量版：**15.4 FPS**
- 对比并发高斯SLAM：
  - GS-SLAM 8.34 FPS
  - SplaTAM 0.21 FPS

**结论**：  
CG-SLAM把“3DGS的渲染快”真正转化成了“SLAM端到端更快”，尤其轻量版已经逼近论文标题强调的实时级。

#### 4. 消融信号：作者提出的约束不是装饰，而是因果有效
- 去掉**Liso**后，Replica中出现跟踪失败，并伴随明显箭头状伪影
- 去掉**Lalign + Lvar**后：
  - RMSE从0.26 cm升到0.33 cm
  - Chamfer Distance从3.85 cm升到4.79 cm
- 不确定性模型的误差曲线显示：它能减少极端错误并降低方差

**结论**：  
性能提升不是单纯来自“换成3DGS”，而是来自“让高斯更贴表面、并降低不稳定原语权重”这两个核心旋钮。

### 局限性
- **Fails when**: 场景中存在动态物体；需要大量未观测区域补全时；对高不确定性阈值较敏感的薄结构/边缘区域可能受影响。
- **Assumes**: 有RGB-D输入和相对可靠的深度观测；场景基本静态；依赖自定义CUDA栅格器与高斯场优化超参；局部滑窗足以约束误差。
- **Not designed for**: 单目SLAM、动态SLAM、强语义交互场景、完整大规模长期回环闭合系统。

### 资源与复现依赖
- 论文实验使用 **RTX 4090 + i9-14900K**
- 高斯场是**非MLP但高存储开销**的表示，内存占用是明确短板
- 论文写明“**将公开代码**”，就本文信息看仍应视为 `opensource/promised`

### 可复用组件
1. **3DGS位姿求导 + GPU栅格化跟踪框架**  
   适合任何想做“直接法高斯场位姿优化”的系统。
2. **α深度 / 中值深度对齐损失**  
   是一个很实用的“把高斯往表面压”的几何一致性工具。
3. **基于主导像素的高斯不确定性估计**  
   可用于剪枝、降权、关键原语选择。
4. **NetVLAD驱动的共视关键帧选择**  
   是比纯视锥重叠更灵活的局部BA关键帧检索思路。

## Local PDF reference

![[paperPDFs/SLAM_Video/Lecture_Notes_in_Computer_Science_2025/2025_CG_SLAM_Efficient_Dense_RGB_D_SLAM_in_a_Consistent_Uncertainty_aware_3D_Gaussian_Field.pdf]]