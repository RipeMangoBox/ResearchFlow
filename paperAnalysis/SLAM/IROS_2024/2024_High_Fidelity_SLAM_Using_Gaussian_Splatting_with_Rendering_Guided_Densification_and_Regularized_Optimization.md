---
title: "High-Fidelity SLAM Using Gaussian Splatting with Rendering-Guided Densification and Regularized Optimization"
venue: arXiv
year: 2024
tags:
  - 3D_Gaussian_Splatting
  - task/rgbd-slam
  - task/3d-reconstruction
  - rendering-guided-densification
  - continual-regularization
  - dataset/Replica
  - dataset/TUM-RGBD
  - opensource/full
core_operator: 以渲染误差驱动高斯增密，并用基于参数重要性的正则化抑制连续建图遗忘，从而同时提升在线RGB-D SLAM的重建保真度与跟踪稳定性
primary_logic: |
  顺序RGB-D帧与初始位姿预测 → 在当前高斯地图上渲染颜色/深度/不透明度并按孔洞与重渲染误差增密 → 用重要性加权正则约束高斯参数更新并交替优化位姿与地图 → 输出高保真3D高斯地图与相机轨迹
claims:
  - "在Replica上，该方法取得最佳平均渲染质量，达到36.19 dB PSNR、0.98 SSIM和0.05 LPIPS [evidence: comparison]"
  - "在Replica上，该方法取得最佳平均跟踪精度，ATE RMSE为0.25 cm，优于表中的SplaTAM与GS-SLAM [evidence: comparison]"
  - "在Room0消融中，去掉正则化或去掉基于渲染的增密都会使PSNR从35.74 dB降至约25–26 dB，并使ATE恶化到0.89–10.02 cm，说明两项设计都不可或缺 [evidence: ablation]"
related_work_position:
  extends: "3D Gaussian Splatting (Kerbl et al. 2023)"
  competes_with: "SplaTAM (Keetha et al. 2023); GS-SLAM (Yan et al. 2023)"
  complementary_to: "Loopy-SLAM (Liso et al. 2024); LangSplat (Qin et al. 2023)"
evidence_strength: moderate
pdf_ref: paperPDFs/SLAM/IROS_2024/2024_High_Fidelity_SLAM_Using_Gaussian_Splatting_with_Rendering_Guided_Densification_and_Regularized_Optimization.pdf
category: 3D_Gaussian_Splatting
---

# High-Fidelity SLAM Using Gaussian Splatting with Rendering-Guided Densification and Regularized Optimization

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2403.12535) · [Code](https://github.com/ljjTYJR/HF-SLAM)
> - **Summary**: 这篇工作把离线 3D Gaussian Splatting 扩展到在线 RGB-D SLAM，通过“渲染失败位置增密 + 重要性正则防遗忘”同时提升了相机跟踪与高保真重建。
> - **Key Performance**: Replica 平均 **36.19 dB PSNR / 0.05 LPIPS / 0.25 cm ATE**；TUM 上渲染质量也显著领先基线，例如 fr1/desk 达 **22.60 dB PSNR**。

> [!info] **Agent Summary**
> - **task_path**: 顺序 RGB-D 帧 / 在线 SLAM -> 相机位姿 + 高保真 3D 高斯地图
> - **bottleneck**: 在线 3DGS 既缺少可靠的增密判据来修补未观察与重观察区域，又容易在连续优化中被最新帧“改写”旧区域
> - **mechanism_delta**: 用“不透明度孔洞 + 颜色/深度重渲染误差”决定何处增密，再用梯度重要性正则限制关键高斯被最新帧过度更新
> - **evidence_signal**: Replica 上渲染与 ATE 同时领先，且 Room0 消融显示两项模块缺一都会明显退化
> - **reusable_ops**: [opacity-hole densification, gradient-importance regularization]
> - **failure_modes**: [真实场景中的 motion blur 与曝光变化会削弱跟踪, 颜色与几何共享高斯参数会带来深度-外观耦合]
> - **open_questions**: [如何提升真实噪声场景下的跟踪鲁棒性, 如何加入 loop closure 或全局优化而不牺牲在线效率]

## Part I：问题与挑战

这篇论文解决的不是“能不能用 Gaussian Splatting 做重建”，而是更具体的一个系统瓶颈：

**如何把原本依赖已知相机位姿、偏离线优化的 3DGS，变成一个可在线运行的 RGB-D SLAM 系统，同时不牺牲重建保真度。**

### 1. 真正的问题是什么？
传统 RGB-D SLAM 往往更偏重：
- **跟踪精度**：把相机位姿估准；
- **几何重建**：把表面建出来；

但对**高保真外观重建**支持不足。  
NeRF 系方法虽然有逼真渲染能力，但在线 SLAM 场景里常因 **ray marching 成本高**，不得不只采样少量像素做优化，导致：
- 映射更新稀疏；
- 全分辨率渲染质量不高；
- 在线系统效率受限。

3DGS 给了一个更现实的机会：它通过 **rasterization** 而不是沿射线密集采样来渲染，因此更适合在线系统处理全分辨率图像。

### 2. 但把 3DGS 直接搬到 SLAM 里还差什么？
作者指出有两个核心瓶颈：

1. **增密判据不够好**  
   在线 SLAM 要一帧一帧扩展地图。若只在“明显未观察”区域补高斯，或随机采样扩图，就会漏掉一种很重要的情况：  
   **某些区域虽然以前见过，但当前渲染仍然差，说明地图还没真正拟合好。**

2. **连续优化的遗忘问题**  
   在线映射时，当前帧会反复推动高斯参数向自己拟合。  
   如果没有约束，同一批高斯会逐渐“过拟合最新视角”，从而让旧视角重渲染质量下降。  
   这会直接伤害：
   - 地图的一致性；
   - 后续基于渲染对齐的位姿跟踪。

### 3. 输入/输出与边界条件
- **输入**：顺序 RGB-D 帧
- **输出**：相机轨迹 + 3D Gaussian 地图
- **场景假设**：
  - 主要面向室内静态场景；
  - 有可用深度输入；
  - 不依赖先验相机位姿；
  - 当前版本**没有 loop closure / 全局图优化**；
  - 仍未达到严格实时。

### 4. 为什么现在值得做？
因为 3DGS 已经把“高保真渲染”从离线重建带入了一个足够高效的表示层。  
现在的关键不再是“有没有更强的表示”，而是：

**如何让这个表示在在线 SLAM 的时序更新中稳定工作。**

这正是本文卡住的系统级瓶颈。

---

## Part II：方法与洞察

方法结构很清晰：**同一个高斯地图同时承担 tracking 和 mapping**，都通过“渲染—对比—优化”的可微流程完成。

### 1. 渲染引导的高斯增密

作者不是简单把新深度点都加进地图，而是用当前帧与地图重渲染的差异来决定哪里该增密。

具体分两类：

#### a) 填补未观察区域
先在当前估计位姿下渲染**不透明度图**。  
如果某个像素的不透明度很低，说明当前地图在这里基本还是“空的”，应补新高斯。

这解决的是：
- 洞没填上；
- 视野中新出现区域没进入地图。

#### b) 修复已观察但渲染不好的区域
作者进一步比较：
- 渲染颜色 vs 当前 RGB；
- 渲染深度 vs 当前深度。

如果颜色误差大，或归一化后的深度误差大，也触发增密。

这一步很关键，因为它把增密条件从“有没有几何”扩展为：

**“当前地图是否真的能解释观测。”**

因此不仅能补“新区域”，还能细化“旧区域但拟合差”的部分。

---

### 2. 连续建图中的正则化优化

作者认为在线 3DGS 建图的主要风险是 **forgetting**：  
共享高斯在连续帧上被反复更新，容易越来越偏向最新帧。

为此，论文引入了一个**重要性加权正则**：

- 每个高斯额外维护“被看见次数”和“历史梯度累计”；
- 用这些量估计该高斯各参数对过去建图损失的**重要性**；
- 当该高斯再次被当前帧更新时，重要参数不允许被大幅修改。

直觉上，这相当于：

- 不显式回放大量旧帧，
- 但把“旧帧对哪些参数敏感”压缩成一个轻量记忆。

因此它是一个**替代大 keyframe buffer 的稳定化机制**。

---

### 3. 基于重渲染误差的位姿跟踪

跟踪阶段同样在当前高斯地图上做 differentiable rendering，然后最小化：
- 颜色差异；
- 深度差异。

一个小设计是：  
作者把 RGB 转到 **LAB** 空间，并在跟踪时丢掉亮度通道，只比较更稳定的颜色信息，以减少光照变化带来的影响。

位姿初始化则采用**恒速模型**。

---

### 核心直觉

这篇论文真正改动的不是 3DGS 本体，而是 **在线 SLAM 中两个因果开关**：

1. **地图增长的触发条件变了**  
   从“随机/只补未观察区”  
   变成“哪里渲染失败，就在哪里长地图”。

2. **参数更新的自由度变了**  
   从“当前帧可以任意重写已有高斯”  
   变成“对历史重要的参数更新要受限”。

于是能力变化链条可以写成：

**增密准则更贴近渲染失败分布**  
→ 地图会优先补齐真正影响观测解释的区域  
→ 重建更清晰、更少浮点伪影  

**连续优化加入重要性约束**  
→ 单帧过拟合被压制，多视角一致性更稳  
→ 旧区域不会越优化越差  
→ 跟踪时面对的是更稳定的可微地图。

换句话说，这篇工作提升能力的根因不是“用了高斯”，而是：

**让在线高斯地图的更新规则开始对“渲染误差”和“历史稳定性”负责。**

### 为什么这套设计有效？
- 跟踪本质上依赖当前地图提供足够好的渲染梯度；
- 如果地图局部有孔洞、浮点伪影或旧区域被遗忘，位姿优化就会更不稳定；
- 因此“重建更好”在这里并不是附加收益，而是**反过来改善跟踪优化条件**。

这也解释了为什么作者在 Replica 上能同时拉高重建和 ATE。

### 战略权衡

| 设计选择 | 改变了什么 | 收益 | 代价/风险 |
|---|---|---|---|
| 基于不透明度与重渲染误差的增密 | 把扩图依据从“是否见过”改为“是否解释观测失败” | 更容易填洞，也能细化重观察区域 | 高斯数量可能更快增长，计算与显存压力上升 |
| 重要性加权正则 | 把在线映射从单帧主导改为带历史记忆的更新 | 抑制 forgetting，保住旧区域细节 | 需要维护额外统计量，仍是近似历史约束 |
| 全分辨率 3DGS 渲染 | 改变了 NeRF 类方法的采样稀疏约束 | 渲染质量更高，收敛更快 | 依赖较强 GPU，系统仍非实时 |
| LAB 去亮度跟踪 | 降低颜色项对曝光波动的敏感性 | 跟踪更鲁棒于部分光照变化 | 对严重模糊/强噪声仍不足 |

---

## Part III：证据与局限

### 关键证据

#### 1. Benchmark comparison：Replica 上同时赢重建和跟踪
最强信号来自 Replica：

- 平均 **PSNR 36.19 dB**
- 平均 **SSIM 0.98**
- 平均 **LPIPS 0.05**
- 平均 **ATE 0.25 cm**

这说明作者不是只做到了“好看渲染”，而是把**高保真建图**和**位姿估计**一起推上去了。  
相对并行 Gaussian-SLAM 方法，如 SplaTAM、GS-SLAM，本文在平均指标上更强。

#### 2. Real-world comparison：TUM 上渲染明显更强，但跟踪只算竞争性
在 TUM-RGBD 上，本文渲染质量提升很明显，例如：
- fr1/desk：**22.60 dB PSNR**，显著高于 Point-SLAM 的 13.87 dB

但跟踪并不总是最好。  
这说明一个重要事实：

**高保真地图 ≠ 真实世界噪声下的最强跟踪器。**

真实数据中的运动模糊、曝光变化，会让基于重渲染对齐的跟踪更脆弱。

#### 3. Ablation：两个模块都不是可有可无
在 Replica Room0 的消融里：
- 完整模型达到 **35.74 dB PSNR / 0.98 SSIM / 0.05 LPIPS / 0.34 cm ATE**
- 去掉任一关键模块后，PSNR 下降到约 **25–26 dB**
- ATE 恶化到 **0.89–10.02 cm**

这类结果比单纯的可视化更有说服力，因为它直接说明：
- **渲染引导增密**不是 cosmetic 改动；
- **正则化防遗忘**也不是小修补，而是稳定在线建图的核心组件。

#### 4. Runtime signal：比某些基线更高效，但仍非实时
作者报告在 Replica Room0、V100 24GB 上：
- Tracking：**1.80 s / frame**
- Mapping：**4.59 s / frame**

比 SplaTAM 更快一些，但离实时系统还有距离。  
因此这篇论文的主价值更偏向：

**高保真在线建图机制验证**，而不是工业级实时落地。

---

### 局限性

- **Fails when**: 真实 RGB-D 数据存在明显 motion blur、曝光变化、传感器噪声时，跟踪会变差；在 TUM 上该方法并不总是最好。
- **Assumes**: 依赖 RGB-D 输入、室内相对静态场景、较强 GPU 资源；连续建图仍需要维护高斯统计量与全分辨率渲染，复现实验使用的是单张 V100 24GB。
- **Not designed for**: 目前不是实时 SLAM；没有 loop closure、pose graph/global BA 等全局一致性机制；也不是单目/纯 RGB 场景方案。

### 更具体的技术边界
1. **颜色与几何参数耦合**  
   作者自己也指出，与 Point-SLAM 这类把 appearance / geometry 分开建模的方法相比，本文共享部分高斯参数，可能导致深度最优性受影响，所以在部分 depth 指标上不一定始终最优。

2. **真实场景评价存在“GT 不干净”问题**  
   由于真实图像可能本身模糊、曝光不稳定，方法渲染出来的图反而更清晰，这会让 PSNR 一类与 GT 逐像素对比的指标低估实际视觉效果。

3. **全局优化还没解决**  
   作者尝试过简单引入 BARF 风格 bundle adjustment，但效果更差，说明 3DGS-SLAM 的全局优化机制并不能直接照搬 NeRF 做法。

### 可复用组件
- **渲染失败驱动增密**：可迁移到其他在线 3DGS / 神经场建图系统。
- **梯度重要性正则**：可作为连续场景优化中的轻量防遗忘模块。
- **LAB 去亮度跟踪损失**：适合需要一定光照鲁棒性的重渲染式 pose tracking。

## Local PDF reference

![[paperPDFs/SLAM/IROS_2024/2024_High_Fidelity_SLAM_Using_Gaussian_Splatting_with_Rendering_Guided_Densification_and_Regularized_Optimization.pdf]]