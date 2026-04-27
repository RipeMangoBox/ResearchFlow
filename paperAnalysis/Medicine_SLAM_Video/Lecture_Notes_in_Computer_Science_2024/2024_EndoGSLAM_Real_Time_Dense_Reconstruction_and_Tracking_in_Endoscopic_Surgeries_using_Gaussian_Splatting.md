---
title: "EndoGSLAM: Real-Time Dense Reconstruction and Tracking in Endoscopic Surgeries using Gaussian Splatting"
venue: arXiv
year: 2024
tags:
  - 3D_Gaussian_Splatting
  - task/3d-reconstruction
  - task/camera-tracking
  - gaussian-splatting
  - differentiable-rasterization
  - dataset/C3VD
  - opensource/no
core_operator: 针对内窥镜随相机移动的局部光照，将标准3DGS简化为各向同性彩色高斯，并结合可微栅格化、增量扩图和局部精炼，实现术中可用的实时跟踪与稠密重建
primary_logic: |
  RGB-D内窥镜序列与相机内参 → 用简化各向同性3D高斯初始化场景，并通过亮度/可见性筛选做位姿跟踪、对未观测区域做增量扩图、对新近区域做关键帧驱动的部分精炼 → 输出在线相机轨迹、高保真稠密组织地图与实时新视角渲染
claims:
  - "在 C3VD 上，EndoGSLAM-H 的新视角渲染指标达到 SSIM 0.77、LPIPS 0.22，优于 NICE-SLAM 的 0.73/0.33 和 Endo-Depth 的 0.64/0.33 [evidence: comparison]"
  - "在对比方法中，只有 EndoGSLAM 同时支持在线稠密重建与 100+ fps 在线渲染；NICE-SLAM 的在线渲染仅 0.27 fps，而 ORB-SLAM3 与 Endo-Depth 不支持在线稠密重建 [evidence: comparison]"
  - "在 EndoGSLAM-R 上移除亮度 pre-filter 或 Gaussian 简化会明显损害稳定性与效率：ATE 从 1.23 mm 升至 2.14/2.26 mm，且去掉简化后跟踪/重建时延从 62.4/65.1 ms 增至 90.0/98.0 ms [evidence: ablation]"
related_work_position:
  extends: "3D Gaussian Splatting (Kerbl et al. 2023)"
  competes_with: "NICE-SLAM (Zhu et al. 2022); Endo-Depth-and-Motion (Recasens et al. 2021)"
  complementary_to: "N/A"
evidence_strength: moderate
pdf_ref: paperPDFs/Medicine_SLAM_Video/Lecture_Notes_in_Computer_Science_2024/2024_EndoGSLAM_Real_Time_Dense_Reconstruction_and_Tracking_in_Endoscopic_Surgeries_using_Gaussian_Splatting.pdf
category: 3D_Gaussian_Splatting
---

# EndoGSLAM: Real-Time Dense Reconstruction and Tracking in Endoscopic Surgeries using Gaussian Splatting

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2403.15124), [Project](https://EndoGSLAM.loping151.com)
> - **Summary**: 这篇工作把 3D Gaussian Splatting 改造成适配内窥镜场景的 RGB-D SLAM 系统，用更轻的高斯表示与局部优化流程，同时实现了在线跟踪、在线稠密重建和术中实时新视角可视化。
> - **Key Performance**: C3VD 上 EndoGSLAM-H 达到 **SSIM 0.77 / LPIPS 0.22**；EndoGSLAM-R 支持 **100+ fps 在线渲染**。

> [!info] **Agent Summary**
> - **task_path**: RGB-D 内窥镜视频流 / 术中 SLAM -> 相机位姿 + 稠密组织重建 + 实时新视角渲染
> - **bottleneck**: 内窥镜场景中弱纹理、随相机移动的局部光源和严格时延约束，使现有方法很难同时兼顾高保真稠密重建与术中实时性
> - **mechanism_delta**: 用简化的各向同性彩色 Gaussian 替代标准 3DGS 的高参数表示，并以可见性筛选的跟踪、增量扩图和部分关键帧精炼替换重型全局优化
> - **evidence_signal**: C3VD 上只有该方法同时实现在线稠密重建和 100+ fps 在线渲染，且 EndoGSLAM-H 的 SSIM/LPIPS 优于 NICE-SLAM 与 Endo-Depth
> - **reusable_ops**: [简化高斯参数化, 可见性驱动的增量扩图]
> - **failure_modes**: [强非刚性组织形变时地图与位姿可能失配, 纯 RGB 或深度噪声较大时跟踪与重建会明显退化]
> - **open_questions**: [如何去掉对深度输入的依赖, 如何在高光与组织形变并存时保持长期稳定]

## Part I：问题与挑战

这篇论文解决的不是“单独把定位做准”或“单独把重建做漂亮”，而是一个更苛刻的术中系统问题：

**输入**：连续的 RGB-D 内窥镜帧、相机内参。  
**输出**：每帧相机位姿、在线更新的稠密组织地图，以及任意已观测区域的实时新视角渲染。

### 真正的难点是什么？

内窥镜手术场景里，SLAM 的瓶颈比普通室内场景更尖锐：

1. **几何特征稀缺且不稳定**  
   组织表面往往弱纹理、重复纹理多，传统特征点法很难稳定建图。

2. **光照不满足常规光度一致性**  
   内窥镜光源通常跟着相机走，导致同一块组织在不同视角下亮度变化很大。  
   这会直接破坏很多基于 photometric consistency 的跟踪/重建假设。

3. **术中要求不是“离线精美”，而是“在线可用”**  
   外科场景需要的是：
   - 位姿要在线更新；
   - 地图要持续补全；
   - 渲染要足够快，医生能即时查看历史区域。  
   很多 NeRF/隐式表示方法虽然重建好，但速度远不够术中使用。

### 为什么现在值得做？

因为 **3D Gaussian Splatting 提供了一个新的系统拐点**：  
它比 NeRF 更显式、更易渲染，也更适合在线优化。如果能把它进一步改造成适配内窥镜光照和实时约束的表示，就有机会把“高质量重建”和“术中实时性”放到同一个框架里。

### 边界条件

这篇方法有明确边界：

- 依赖 **RGB-D** 输入，不是纯 RGB 单目 SLAM。
- 假设场景基本是 **近刚性/弱形变**，并未解决强非刚性组织运动。
- 需要已知内参，图像预先去畸变。
- 目标是 **术中在线稠密可视化**，不是大规模全局回环或长时间场景级建图。

---

## Part II：方法与洞察

EndoGSLAM 的思路非常系统化：**不是简单把 3DGS 搬到医学场景，而是围绕“哪些参数没必要、哪些区域值得算、哪些像素可信”做了整套删繁就简。**

### 核心直觉

这篇论文引入的关键因果旋钮是：

> **把标准 3DGS 中高自由度、强视角相关的表示，替换为更适合内窥镜光照分布的轻量显式表示；再把优化预算集中在“当前可靠像素”和“新近扩展区域”。**

这带来了三层变化：

- **表示层变化**：  
  从带 SH 的标准 Gaussian 改成 **各向同性半径 + 直接颜色属性**。  
  由于内窥镜光照基本随相机移动，显式建模复杂 view-dependent appearance 的收益下降，反而会增大参数量和优化不稳定性。

- **约束层变化**：  
  跟踪时只使用 **亮度可靠** 且 **当前可见性高** 的像素，避免被过暗/过曝区域和错误重建区域拖偏。

- **计算层变化**：  
  不做昂贵的全局反复重建，而是用 **增量扩图 + 部分精炼**，把算力主要花在“新看到的地方”和“近期关键帧”。

最终能力变化是：

- 从“能重建但太慢/太模糊”  
  变成  
- “能在线跟踪、在线补图、还能实时渲染给医生看”。

### 方法主线

#### 1. 简化的 Gaussian 表示

作者把标准 3DGS 做了两处关键简化：

- **SH 系数 → 直接颜色属性**
- **各向异性尺度 → 单一半径（各向同性）**

这样每个 Gaussian 的参数量从 **59 降到 8**，论文称计算开销约减少 **86%**。

这不是纯工程压缩，而是利用了内窥镜场景的特殊性：

- 光源跟相机移动，复杂视角相关颜色并不稳定；
- 组织重建更需要快速、稳定、可优化的显式表示。

#### 2. 基于可微栅格化的相机跟踪

在每个新帧到来时，系统会：

- 用上一帧位姿和常速度先验初始化当前位姿；
- 用当前 Gaussian 地图渲染颜色、深度、可见性；
- 用颜色+深度重投影误差优化位姿。

但作者不是对所有像素一视同仁，而是加了两个过滤器：

- **亮度 pre-filter**：过滤掉过暗/过曝像素；
- **可见性过滤**：只用当前地图中已被可靠解释的区域做优化。

这一步针对的就是内窥镜里最麻烦的噪声源：**不稳定光照** 和 **地图未成熟区域**。

#### 3. Gaussian Expanding：只在该补的地方补图

跟踪完当前帧后，系统把新观测到的组织补进地图。扩展条件主要包括：

- 当前地图对该区域可见性低，说明还没覆盖好；
- 当前帧在现有表面前方观测到新几何；
- 像素颜色可靠，不是极端亮度区域。

这使得地图是 **增量生长** 的，而不是每帧都全图重建。

#### 4. Partial Refining：只精炼“近期重要区域”

新扩展出来的 Gaussian 往往还比较粗糙，如果立即做全局优化，实时性会崩。

所以作者采用 **部分精炼**：

- 每隔若干帧选 keyframe；
- 优先采样与当前帧在时间或空间上更近的 keyframe；
- 重点优化新扩展 Gaussian 和最近加入但质量未稳定的部分。

这本质上是在做一个实时 SLAM 常见但很有效的取舍：

> 不追求全局最优，只追求术中当前足够稳定、足够快。

### 战略取舍

| 设计选择 | 改变的瓶颈 | 带来的能力 | 代价 |
|---|---|---|---|
| SH → 直接颜色；各向异性 → 各向同性 | 降低表示自由度和优化维度 | 更快优化，更稳定渲染，适合在线系统 | 不擅长显式建模强视角相关反射 |
| 亮度 pre-filter + 可见性筛选跟踪 | 去掉不可靠监督 | 位姿优化更稳，减少光照干扰 | 有效像素过少时可能削弱约束 |
| 增量扩图 | 只在未观测/新几何区域加点 | 在线补全场景，避免全局重建开销 | 全局一致性修正能力有限 |
| 部分关键帧精炼 | 把算力集中到近期区域 | 在实时约束下保持局部质量 | 远期区域可能缺少充分再优化 |

---

## Part III：证据与局限

### 关键证据

#### 1. 比较信号：能力跃迁主要体现在“系统级帕累托前沿”
论文最强的结论不是“所有单项指标都第一”，而是：

- **ORB-SLAM3** 定位很强，但是稀疏地图，不能给术中稠密在线可视化；
- **NICE-SLAM** 能做在线 dense mapping，但渲染速度只有 **0.27 fps**，术中基本不可用；
- **Endo-Depth** 有较强定位/重建思路，但依赖后处理融合，难以在线闭环成一个实时可视化系统；
- **EndoGSLAM** 是唯一把 **在线跟踪 + 在线稠密重建 + 100+ fps 渲染** 放在一起的方案。

**这就是它相对 prior work 的真正能力跳跃。**

#### 2. 质量信号：显式 Gaussian 地图比隐式基线更适合清晰新视角渲染
在 C3VD 上，EndoGSLAM-H 的结果是：

- **SSIM 0.77**
- **LPIPS 0.22**

优于 NICE-SLAM 的 **0.73 / 0.33**，也明显优于 Endo-Depth。  
这说明该方法不只是“更快”，而是**在术中可用速度下仍保住了较好的视觉质量**。

#### 3. 跟踪信号：定位是“竞争力强”，但不是绝对第一
一个需要诚实指出的点是：

- ATE 最优的是 **ORB-SLAM3: 0.32 mm**
- EndoGSLAM-H 是 **0.34 mm**

也就是说，**如果只看纯定位误差，EndoGSLAM 并没有碾压传统 SLAM**。  
但它的优势在于：在几乎相当的定位精度下，额外提供了在线稠密地图和实时渲染，这才是临床工作流更需要的系统能力。

#### 4. 消融信号：作者的三个小设计都不是可有可无
在 EndoGSLAM-R 上：

- 去掉 **pre-filter**，ATE 从 **1.23 mm** 上升到 **2.14 mm**；
- 去掉 **partial refining**，渲染与重建质量下降；
- 去掉 **Gaussian simplification**，跟踪/重建时延从 **62.4/65.1 ms** 增至 **90.0/98.0 ms**，且颜色稳定性更差。

这说明论文的贡献不是单一 trick，而是一组彼此配合的系统设计。

### 局限性

- **Fails when**: 场景存在明显非刚性组织形变、强烈镜面高光、深度噪声很大或大量区域过暗/过曝时，地图和跟踪都可能失稳；纯 RGB 场景下该方法无法直接工作。
- **Assumes**: 依赖 RGB-D 输入、已知内参、预去畸变图像，以及“内窥镜光照可用视角无关颜色近似”的建模假设；论文报告的实时性建立在 **RTX 4090** 上。
- **Not designed for**: 单目无深度 SLAM、显著变形组织重建、长期全局回环优化、跨器械大范围动态交互场景。

### 复现与可扩展性备注

- 证据主要来自 **单一数据集 C3VD**，所以结论更像是“在该类结肠镜 RGB-D 条件下非常有前景”，而不是对所有内窥镜场景已充分验证。
- 论文给了 **project page**，但从正文看不出明确代码开放信息，因此开源可复现性目前应保守看待。

### 可复用组件

1. **场景特化的轻量 Gaussian 参数化**  
   当场景不需要复杂视角相关外观时，可直接减少 3DGS 自由度。

2. **可见性/亮度筛选的 dense pose tracking**  
   对移动光源、亮度漂移大的视觉系统尤其有用。

3. **增量扩图 + 局部精炼**  
   适合任何需要“在线更新但不能全局重算”的实时重建系统。

![[paperPDFs/Medicine_SLAM_Video/Lecture_Notes_in_Computer_Science_2024/2024_EndoGSLAM_Real_Time_Dense_Reconstruction_and_Tracking_in_Endoscopic_Surgeries_using_Gaussian_Splatting.pdf]]