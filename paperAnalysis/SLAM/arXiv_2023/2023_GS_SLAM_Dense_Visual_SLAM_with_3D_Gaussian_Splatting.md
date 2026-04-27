---
title: "GS-SLAM: Dense Visual SLAM with 3D Gaussian Splatting"
venue: arXiv
year: 2023
tags:
  - 3D_Gaussian_Splatting
  - task/dense-visual-slam
  - gaussian-splatting
  - adaptive-expansion
  - coarse-to-fine-tracking
  - dataset/Replica
  - dataset/TUM-RGBD
  - opensource/no
core_operator: "用3D Gaussian Splatting替代NeRF体渲染，并通过在线高斯增删与粗到细可靠高斯筛选实现RGB-D稠密SLAM。"
primary_logic: |
  RGB-D序列与相机内参 → 以3D高斯显式建图并在关键帧上按不可靠像素增添/按深度一致性删除高斯，同时用粗到细可靠高斯渲染优化位姿 → 输出相机轨迹、稠密场景表示与高速高质量渲染
claims:
  - "在 Replica 上，GS-SLAM 的平均 ATE 为 0.50 cm，优于 Point-SLAM 的 0.54 cm，同时系统速度达到 8.34 FPS，而 Point-SLAM 为 0.42 FPS [evidence: comparison]"
  - "在 Replica 渲染评测中，GS-SLAM 达到 34.27 dB PSNR、0.975 SSIM、0.082 LPIPS 和 386.91 FPS，并在全部报告的渲染指标上超过 NICE-SLAM、Vox-Fusion、CoSLAM 与 ESLAM [evidence: comparison]"
  - "在 Room0 消融中，去掉新增高斯会导致系统失败；去掉删除策略会使 ATE 从 0.48 升至 0.58、Recall 从 61.29% 降至 49.32%，说明自适应增删策略对在线建图是必要的 [evidence: ablation]"
related_work_position:
  extends: "3D Gaussian Splatting (Kerbl et al. 2023)"
  competes_with: "Point-SLAM (Sandström et al. 2023); ESLAM (Johari et al. 2023)"
  complementary_to: "N/A"
evidence_strength: strong
pdf_ref: paperPDFs/SLAM/arXiv_2023/2023_GS_SLAM_Dense_Visual_SLAM_with_3D_Gaussian_Splatting.pdf
category: 3D_Gaussian_Splatting
---

# GS-SLAM: Dense Visual SLAM with 3D Gaussian Splatting

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2311.11700), [Project](https://gs-slam.github.io/)
> - **Summary**: 这篇工作把 3D Gaussian Splatting 真正改造成可在线优化的 RGB-D SLAM 表示，用“高斯增删 + 粗到细可靠高斯跟踪”同时缓解 NeRF-SLAM 的渲染慢问题和显式高斯在在线建图中的漂浮噪声问题。
> - **Key Performance**: Replica 上平均 ATE 0.50 cm；渲染达到 34.27 dB PSNR 和 386.91 FPS，系统速度 8.34 FPS。

> [!info] **Agent Summary**
> - **task_path**: RGB-D序列 + 已知相机内参 -> 相机位姿轨迹 + 稠密3D高斯地图 + 可渲染RGB-D图像
> - **bottleneck**: NeRF-SLAM 的体渲染过慢，只能稀疏采样像素；而原始 3DGS 又不具备在线扩图、去浮点和稳定位姿优化能力
> - **mechanism_delta**: 把 3DGS 从“静态场景新视角合成表示”改成“可增量增长/裁剪的 SLAM 地图”，并用粗到细两阶段只对可靠高斯做位姿优化
> - **evidence_signal**: Replica 上渲染质量和速度显著领先，且增删策略、粗到细跟踪、深度监督都有独立消融支撑
> - **reusable_ops**: [基于低透明度/深度残差的不可靠像素回投扩图, 基于深度一致性的可靠高斯筛选]
> - **failure_modes**: [深度质量差时跟踪和建图明显退化, 大场景下高斯与球谐系数带来较高内存占用]
> - **open_questions**: [如何加入回环闭合与长期全局一致性, 如何压缩高斯表示以扩展到更大尺度场景]

## Part I：问题与挑战

这篇论文解决的是 **RGB-D 稠密视觉 SLAM**：输入连续 RGB-D 帧和已知内参，输出逐帧相机位姿与可渲染的稠密场景表示。

### 真问题是什么

作者瞄准的不是“能不能建图”，而是更难的平衡问题：

1. **NeRF-based SLAM 太慢**  
   现有神经隐式 SLAM 大多依赖 ray-based volume rendering。为了把优化时间压下来，实际只能采样少量像素做跟踪和建图，因此：
   - 全图细节监督不足；
   - 稠密重建容易缺少边缘与纹理细节；
   - 高分辨率渲染很难实时。

2. **原始 3DGS 不能直接拿来做 SLAM**  
   3D Gaussian Splatting 在静态新视角合成里很快，但默认前提通常是：
   - 已有较好的初始化点云/相机位姿；
   - 主要面向静态对象或离线优化；
   - 没有处理在线 SLAM 里的“新区域不断出现”“错误高斯漂浮”“位姿梯度被伪影污染”等问题。

### 真瓶颈在哪里

真正瓶颈不是单一的“表示能力”，而是三件事耦合在一起：

- **计算瓶颈**：体渲染让全图监督不可负担。
- **表示增长瓶颈**：在线 SLAM 必须持续覆盖新观察区域，不能只靠初始点集。
- **优化污染瓶颈**：错误高斯/浮点会把位姿优化的梯度带偏，造成漂移。

### 为什么现在值得做

因为 3DGS 已经证明：**高分辨率可微渲染可以从 ray marching 切换到高效 splatting**。这意味着过去 NeRF-SLAM 为了速度不得不牺牲的“全图监督”和“渲染质量”，现在有机会被重新拿回来。

### 边界条件

这篇方法的适用前提比较明确：

- **输入**：RGB-D 序列，且相机内参已知；
- **场景**：以室内、静态场景为主；
- **系统形态**：关键帧式 tracking-mapping-BA；
- **硬件依赖**：GPU + 自定义 CUDA splatting 实现；
- **不包含**：回环闭合、动态图建模、单目设定。

---

## Part II：方法与洞察

GS-SLAM 的核心思路是：**把 3D 高斯当作可直接优化的显式地图单元，用 splatting 做快速 RGB-D 渲染，再围绕“在线扩图”和“鲁棒跟踪”补上 SLAM 所需机制。**

### 1. 3D 高斯显式场景表示

每个地图单元是一个 3D Gaussian，包含：

- 位置
- 协方差/尺度与方向
- 不透明度
- 颜色球谐系数

和 NeRF-SLAM 最大的结构差异是：**这里没有解码 MLP**，地图就是一组可直接优化的高斯参数。  
给定相机位姿后，系统通过深度排序和 alpha blending 直接 splat 出 **颜色图和深度图**。

这一步改变了什么？  
它把“每条光线都要积分”的体渲染，换成了“对高斯做光栅化/混合”的显式渲染，因此高分辨率渲染与反传都更快。

### 2. 自适应高斯扩张建图

这是论文最关键的在线化改造。

#### 初始化

第一帧只用一半像素回投初始化高斯，而不是把所有像素都塞满。  
这样做的目的，是给后续 densification / split / clone 留空间。

#### Adding：只在“不可靠区域”长新高斯

作者先用历史高斯渲染当前关键帧，再看哪些像素不可靠：

- 累积 opacity 太低；
- 或渲染深度和观测深度差太大。

这些像素大概率对应：
- 新出现的区域；
- 还没被地图覆盖的几何；
- 历史表示明显错掉的地方。

然后把这些像素回投成新 3D 点，初始化成新高斯加入地图。  
这一步本质上把“在线场景增长”从被动 densify，改成了**由渲染失败信号驱动的主动补图**。

#### Deleting：按深度一致性压掉浮点高斯

只增不删会有很多 floaters。  
GS-SLAM 对当前视锥内可见高斯做一次深度一致性检查：如果某个高斯明显漂在真实表面前面，就把它的不透明度大幅衰减。

注意这不是硬删除几何，而是**先削弱其对渲染和梯度的影响**。  
这会直接改善：
- 地图噪声；
- 渲染伪影；
- 跟踪阶段的错误监督。

### 3. 粗到细位姿跟踪

作者没有直接拿全图和全部高斯去做位姿优化，因为那样很容易被局部伪影拖偏。

#### Coarse stage

- 先用半分辨率、稀疏像素渲染；
- 得到一个较稳定的粗位姿。

这个阶段的作用是先落到一个大致正确的 basin，减少高频噪声影响。

#### Fine stage

接着用粗位姿 + 深度观测，筛出**靠近真实表面**的可靠高斯，只用这些高斯渲染全分辨率图像来细化位姿。

这一步的关键不只是“分两阶段”，而是：
- **第二阶段不是更密地看所有内容**，
- 而是**更密地只看可信内容**。

因此位姿梯度主要来自结构清晰、几何一致的区域，而不是来自漂浮高斯或尚未优化好的区域。

### 4. 局部 Bundle Adjustment

在 BA 中，作者对关键帧窗口内的位姿和高斯一起优化。  
为了稳定性，前半程只优化地图，后半程再联合优化位姿与地图，避免一开始被不稳定地图拖着跑。

### 核心直觉

**这篇论文真正拧动的“因果旋钮”有三个：**

1. **把渲染约束从 ray-based 体渲染换成 splatting 光栅化**  
   - 改变的瓶颈：计算约束  
   - 结果：全图 RGB-D 渲染与反传变得足够快，能承受更密监督

2. **把地图维护从“固定初始化 + 被动优化”换成“按失败信号增量补图 + 按深度一致性抑噪”**  
   - 改变的瓶颈：表示覆盖与在线可扩展性  
   - 结果：3DGS 不再只是离线静态表示，而能在线覆盖新区域并控制 floaters

3. **把位姿优化从“对全部渲染结果求梯度”换成“先粗定位，再只用可靠高斯细化”**  
   - 改变的瓶颈：优化信息污染  
   - 结果：跟踪更稳，伪影更不容易把相机拖偏

换句话说，这不是单纯“把 3DGS 塞进 SLAM”，而是把 3DGS 改造成一个 **可增长、可裁剪、可选择性参与优化** 的在线地图系统。

### 战略取舍

| 设计选择 | 改变了什么瓶颈 | 带来的能力 | 代价/风险 |
|---|---|---|---|
| 3DGS + splatting 替代 NeRF 体渲染 | 全图渲染/反传太慢 | 高速高分辨率 RGB-D 渲染，支持更密监督 | 依赖 GPU/CUDA；高斯参数内存更大 |
| 不可靠像素驱动新增高斯 | 新区域无法被在线覆盖 | 可增量重建新观察几何 | 若不配套删噪，容易累积错误高斯 |
| 深度一致性驱动 opacity 衰减 | floaters 污染地图和梯度 | 更干净的建图与渲染 | 高度依赖深度质量 |
| 粗到细 + 可靠高斯筛选 | 全量优化容易被伪影带偏 | 更鲁棒的位姿估计 | 两阶段优化更复杂，且会忽略未建好区域 |
| 无 MLP 解码器、直接优化高斯 | 隐式表示优化慢 | 0 个网络参数，优化链路更直接 | 表示压缩性较差，场景大时内存上升 |

---

## Part III：证据与局限

### 关键实验信号

1. **比较信号：Replica 上的“速度-精度平衡”确实提升**
   - 平均 ATE 为 **0.50 cm**
   - 优于 Point-SLAM 的 **0.54 cm**
   - 系统速度 **8.34 FPS**，而 Point-SLAM 只有 **0.42 FPS**

   这说明作者的核心卖点不是单点最优，而是：**在接近或更好跟踪精度下，把端到端 SLAM 速度显著拉起来。**

2. **比较信号：渲染是最明显的能力跳变**
   - Replica 上平均 **34.27 dB PSNR / 0.975 SSIM / 0.082 LPIPS**
   - 平均渲染速度 **386.91 FPS**

   这是全篇最强证据。它直接支持论文的主因果链：  
   **splatting 渲染更快 → 能承受更密、更高分辨率的监督 → 渲染质量与细节更好。**

3. **比较信号：建图精度偏“高精度表面”，但不一定在完整性上统治**
   - Replica 上平均 **Depth L1 = 1.16 cm**
   - 平均 **Precision = 74.0%**
   - Recall / F1 与 CoSLAM、ESLAM 是可比而非全面压制

   这说明 GS-SLAM 更像是把**表面精细度和渲染质量**做强，而不是在每个几何指标上都绝对第一。

4. **泛化信号：真实 TUM-RGBD 上并非最强跟踪器**
   - 平均 ATE **3.7 cm**
   - 优于 iMAP / NICE-SLAM / Vox-Fusion
   - 但落后于 ESLAM / CoSLAM / Point-SLAM，也明显落后于 ORB-SLAM2、BAD-SLAM 等传统/几何强基线

   所以这篇论文的“能力跳变”主要体现在 **3DGS 表示带来的渲染-建图效率收益**，而不是已经彻底解决真实复杂场景下的最强跟踪问题。

5. **消融信号：核心模块是必要的，不是装饰件**
   - **w/o add**：直接失败，说明原始 3DGS 的 densification 不足以支撑在线 SLAM
   - **w/o delete**：Room0 上 ATE 从 **0.48 → 0.58**，Recall 从 **61.29 → 49.32**
   - **w/o depth**：ATE 从 **0.48 → 0.80**，Depth L1 从 **1.31 → 3.21**
   - **仅 coarse / 仅 fine** 都不如 coarse-to-fine

   这些消融很好地对上了作者的机制主张：  
   **在线扩图、抑制 floaters、以及可靠高斯筛选，都是把 3DGS 变成 SLAM 系统所必需的。**

### 局限性

- **Fails when**: 深度噪声大、深度缺失严重、场景动态明显、或长序列全局漂移较强时，可靠高斯筛选和增删判断都会变脆弱；真实场景上的 TUM 结果也说明它不是当前最稳的 tracker。
- **Assumes**: 依赖高质量 RGB-D 输入、已知内参、静态室内场景、关键帧优化流程，以及自定义 CUDA splatting 实现和较强 GPU 资源；文中主实验运行于 RTX 4090，且 Replica #Room0 的场景表示内存约 **198.04 MB**。
- **Not designed for**: 单目 SLAM、动态场景 SLAM、带回环闭合的长期全局一致性优化、以及直接从 3DGS 中输出高质量网格；其网格评测实际仍依赖 **TSDF Fusion**，不是原生高斯网格化结果。

### 可复用组件

这篇论文里最值得迁移的，不是完整系统，而是下面几个操作符：

1. **不可靠像素驱动的增量扩图**  
   用低 opacity / 大深度残差找“地图未覆盖区域”，再回投新增显式表示。

2. **深度一致性驱动的 floater 抑制**  
   不急着删点，而是先衰减错误单元的 opacity，减少其对渲染和位姿梯度的污染。

3. **可靠显式单元筛选后的 coarse-to-fine pose optimization**  
   先低分辨率收敛到粗位姿，再只用可信几何做高分辨率细化。

4. **显式 RGB-D 可微渲染前端**  
   对任何想摆脱 NeRF 体渲染开销的在线重建/SLAM 系统都很有借鉴价值。

## Local PDF reference

![[paperPDFs/SLAM/arXiv_2023/2023_GS_SLAM_Dense_Visual_SLAM_with_3D_Gaussian_Splatting.pdf]]