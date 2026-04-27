---
title: "RGBD GS-ICP SLAM"
venue: arXiv
year: 2024
tags:
  - 3D_Gaussian_Splatting
  - task/rgb-d-slam
  - task/3d-reconstruction
  - generalized-icp
  - covariance-sharing
  - scale-alignment
  - dataset/Replica
  - dataset/TUM-RGBD
  - opensource/full
core_operator: 以单一3D高斯地图同时承载G-ICP跟踪与3DGS建图，并通过协方差共享和尺度对齐把几何配准与地图优化耦合起来
primary_logic: |
  RGB-D帧 → 重投影/下采样得到当前点云并构造源高斯 → 与地图中的目标高斯做G-ICP直接3D配准估计位姿 → 将源高斯经尺度对齐后写入同一3DGS地图并并行优化颜色/透明度/协方差 → 输出相机轨迹与高保真可渲染地图
claims:
  - "在Replica上，30 FPS受限设置下该方法的平均ATE RMSE为0.16 cm，优于SplaTAM的0.36 cm和GS-SLAM的0.50 cm [evidence: comparison]"
  - "在Replica上，无跟踪限速时系统平均98.11 FPS，而GS-SLAM和SplaTAM分别为8.34 FPS和0.23 FPS；同时其平均PSNR为35.93 dB，高于GS-SLAM的34.27 dB和SplaTAM的33.89 dB [evidence: comparison]"
  - "将跟踪中的平面尺度正则替换为文中的椭圆尺度正则后，TUM上的平均ATE从29.12 cm降至2.37 cm，说明保留地图高斯形状特征对稳健配准至关重要 [evidence: ablation]"
related_work_position:
  extends: "3D Gaussian Splatting (Kerbl et al. 2023)"
  competes_with: "SplaTAM (Keetha et al. 2023); GS-SLAM (Yan et al. 2023)"
  complementary_to: "ORB-SLAM3 (Campos et al. 2021)"
evidence_strength: strong
pdf_ref: paperPDFs/SLAM_Video/arXiv_2024/2024_RGBD_GS_ICP_SLAM.pdf
category: 3D_Gaussian_Splatting
---

# RGBD GS-ICP SLAM

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2403.12550), [Code](https://github.com/Lab-of-AI-and-Robotics/GS-ICP-SLAM), [Video](https://youtu.be/ebHh_uMMxE)
> - **Summary**: 这篇工作把 G-ICP 的三维几何配准和 3D Gaussian Splatting 的显式地图表示放进同一个“高斯世界”，用协方差共享与尺度对齐同时提升 RGB-D SLAM 的跟踪速度、轨迹精度和地图质量。
> - **Key Performance**: Replica 上 30 FPS 模式平均 ATE 0.16 cm、PSNR 38.83 dB；不限速时平均 98.11 FPS，最高 107.06 FPS。

> [!info] **Agent Summary**
> - **task_path**: RGB-D序列 / 实时密集SLAM -> 相机位姿 + 可渲染3D高斯地图
> - **bottleneck**: 现有3DGS/NeRF式SLAM往往仍用2D重渲染误差做跟踪，没直接利用显式3D几何，导致跟踪慢、跟踪与建图重复计算且信息不共享
> - **mechanism_delta**: 用G-ICP直接对齐当前帧高斯与地图高斯，并把跟踪得到的协方差直接作为3DGS新高斯初始化，再用尺度对齐和分离式关键帧策略稳定两端接口
> - **evidence_signal**: Replica/TUM双数据集比较加上尺度正则、尺度对齐、关键帧策略、局部最小规避的多组消融共同支持结论
> - **reusable_ops**: [shared-gaussian-covariance, ellipse-scale-regularization, mapping-only-keyframes]
> - **failure_modes**: [depth-noise-or-missing-regions, reduced-mapping-iterations-in-unlimited-speed-mode]
> - **open_questions**: [how-to-fuse-rgb-cues-to-compensate-depth-noise, how-to-add-loop-closure-and-relocalization]

## Part I：问题与挑战

这篇论文解决的不是“能不能用 3DGS 做 SLAM”，而是**3DGS-SLAM 的真实瓶颈到底在哪**。作者的判断很明确：**瓶颈主要不在建图，而在跟踪**。

- **输入 / 输出接口**  
  输入是 RGB-D 视频流；输出是逐帧相机位姿和一张可实时渲染的 3D Gaussian 地图。

- **现有方法的问题**  
  1. **耦合式方法**（如 GS-SLAM、SplaTAM）虽然地图是显式 3D 高斯，但跟踪仍主要依赖 2D photometric/rendering error。  
     这意味着系统要经过“渲染图像 → 与观测图像比误差 → 反复优化位姿”的间接路径，速度和稳健性都受限。
  2. **解耦式方法**（如 Photo-SLAM）把 tracking 交给 ORB-SLAM 一类成熟前端，精度常更稳，但代价是要维护另一套特征地图/前端管线。  
     结果是：**跟踪和建图用的是两套表示，前端算出来的几何统计量不能直接服务后端地图优化**。

- **真正瓶颈**  
  3DGS 地图本来就是显式 3D 高斯，但主流方法没有把这张地图直接当作 3D 几何对象去做位姿配准。  
  换句话说，系统已经拥有“高斯”，却还在通过 2D 渲染去间接访问它，造成：
  - 跟踪慢；
  - 前后端重复计算；
  - 地图中的结构统计量没有被充分复用。

- **为什么现在值得解决**  
  3DGS 已经把高保真建图和渲染速度推到很高，但如果 tracking 仍停留在 2D 重渲染优化，整个系统上限依旧被卡住。  
  所以现在最值得做的，不是再把地图渲染得更快，而是**让显式高斯地图真正参与定位**。

- **边界条件**  
  该方法默认有 RGB-D 深度输入，主要面向静态室内场景和实时密集 SLAM；并不针对纯 RGB、强动态场景或大型闭环全局一致性问题。

## Part II：方法与洞察

作者的核心设计哲学可以概括成一句话：

**既然 tracking 和 mapping 最终都在处理“高斯”，那就不要维护两个世界。**

### 核心直觉

- **改了什么**  
  把相机位姿估计从“基于 2D 渲染误差的图像对齐”，改成“基于 3D 高斯分布的 G-ICP 直接配准”。

- **改变了哪个瓶颈**  
  以前的瓶颈是：显式 3D 地图要先被渲染成 2D，tracking 才能用。  
  现在改成：当前帧深度重投影出的点云先变成源高斯，地图中的 3DGS 原语直接充当目标高斯，二者在 3D 空间里对齐。  
  这一步同时消除了：
  - 2D 间接观测带来的效率损失；
  - 跟踪/建图两套几何统计的重复构造。

- **能力上发生了什么变化**  
  - 跟踪变快：不再依赖逐步渲染-比对；
  - 跟踪变准：直接利用深度几何而不是只看 2D 图像误差；
  - 建图变稳：tracking 产出的协方差直接作为新高斯初始化，减少后续地图优化负担；
  - 系统更简洁：不需要像解耦式方法那样再维护一张独立特征地图。

- **为什么这在因果上成立**  
  G-ICP 的核心对象是“带协方差的 3D 点”；3DGS 的核心对象是“带协方差的 3D 高斯”。  
  这两者共享的不是表面名字，而是**同一类局部几何统计对象**。  
  因此，一旦把接口对齐，tracking 的输出天然就能成为 mapping 的输入，反过来 map 也天然能成为 tracking 的目标。

### 关键模块拆解

#### 1. 单一高斯地图：跟踪和建图共用同一套原语

系统流程是：

1. 从当前 RGB-D 帧的 depth 图重投影得到点云，并计算当前帧源高斯；
2. 从现有 3DGS 地图中取出目标高斯；
3. 用 G-ICP 做源/目标高斯的 3D 配准，估计当前相机位姿；
4. 若当前帧被选为关键帧，则把这些源高斯直接作为新 primitive 写入地图；
5. mapping 线程并行优化高斯的位置、协方差、颜色和透明度。

这样做有两个直接效果：

- **map → tracking**：地图里的高斯已经有协方差，G-ICP 不必再为 map 重新估计；
- **tracking → map**：当前帧在 G-ICP 中算出的协方差，直接成为新高斯的初始化，不必重复计算。

#### 2. 椭圆尺度正则：让 tracking 适应“被优化过的地图高斯”

传统 point-to-plane 风格正则化会把高斯尺度压成近似平面形状，但 3DGS 地图中的高斯并不全是平面；它们可能对应边、角点、细长结构等。

作者因此提出**椭圆尺度正则**：  
不是把所有高斯硬压成平面，而是按奇异值的相对比例保留其各向异性，再做归一化。

直观上，这等于告诉 G-ICP：

- 地图高斯不是“全都一样的平面片”；
- 它们已经经过 mapping 优化，带有更真实的局部形状信息；
- tracking 时应尊重这种形状，而不是抹掉它。

#### 3. 尺度对齐：让 tracking 产出的高斯能平滑接入 3DGS 地图

从单帧深度图生成的点云有一个天然问题：**越远越稀疏**。  
如果直接用 kNN 协方差估计，高斯尺度会随着深度增大而被夸大。若这种高斯直接加入地图，就会破坏地图中已有高斯的尺度分布。

所以作者在写图前做了**按深度 z 的幂次归一化**，本质是把“传感器采样稀疏性”补偿掉，让新高斯尺度更接近地图里最终收敛后的合理范围。  
消融里最好的设置是除以 \( z^{1.5} \)。

#### 4. 分离式关键帧：tracking keyframe 和 mapping-only keyframe 不再绑死

这是本文很实用的一个系统设计点。

- **tracking** 需要少而精的关键帧，否则 scan matching 误差会积累；
- **mapping** 则需要更多、视角更丰富的帧，否则渲染质量上不去。

作者的处理是：

- 用 G-ICP 中已有的几何对应比例做**动态 tracking keyframe 选择**；
- 另外再加入 **mapping-only keyframe**，只给建图用，不参与 tracking 链。

这样就把两个目标拆开了：

- 跟踪端追求稳定、少漂移；
- 建图端追求视角覆盖、细节恢复。

#### 5. 避免局部最小：随机历史关键帧训练 + pruning

实时 GS mapping 容易陷入一个典型局部最小：  
如果总是拿最近视角训练，高斯会沿当前视线方向被拉长，表面上对当前画面拟合很好，但几何实际上变坏。

作者用了两个简单但有效的招：

- 每次训练随机从已有关键帧中抽一个，而不是只盯着最近帧；
- pruning 掉那些因为过拟合而拉长、或已不必要的高斯。

同时，由于 G-ICP 已经提供了较合理的高斯初始化，**densifying 和 opacity reset 不再是必需项**。

### 战略取舍表

| 设计选择 | 解决的瓶颈 | 收益 | 代价 / 风险 |
| --- | --- | --- | --- |
| 单一高斯地图共享 tracking 与 mapping | 前后端两套表示重复计算 | 更省算力，信息可双向复用 | 若错误高斯被写入地图，误差会反向污染后续配准 |
| 用 G-ICP 替代 2D photometric tracking | 2D 间接跟踪慢且不充分用 3D 几何 | 跟踪更快、更准 | 对深度噪声与缺失更敏感 |
| 椭圆尺度正则 | 地图高斯并非全是平面 | 更匹配真实各向异性结构 | 需要稳定的尺度估计 |
| scale aligning 写图 | 单帧深度导致远处高斯过大 | 新高斯更容易接入地图分布 | 需要经验参数 \(p\) |
| mapping-only keyframes | 跟踪与建图对关键帧需求冲突 | 同时保住 ATE 和 PSNR | 映射调度更复杂 |
| 不做 densifying | 传统 GS 管线开销较高 | 系统更简洁、更快 | 前提是初始高斯注入质量足够高 |
| 不限速运行 | 追求极高实时性 | 可达 ~100 FPS | mapping 可用优化迭代变少，PSNR 会下降 |

## Part III：证据与局限

### 关键实验信号

- **比较信号：轨迹精度确实跳了一档**  
  在 Replica 上，30 FPS 模式下平均 ATE 只有 **0.16 cm**，相比 SplaTAM 的 **0.36 cm** 和 GS-SLAM 的 **0.50 cm** 有明显优势。  
  这直接支持论文的主张：**把 tracking 从 2D 重渲染优化改成 3D G-ICP 配准，是真正的性能杠杆。**

- **比较信号：速度-质量前沿很强**  
  在 Replica 上不限速时，系统平均 **98.11 FPS**，最高 **107.06 FPS**；同时平均 PSNR 还有 **35.93 dB**。  
  这不是单纯“更快但更糙”，而是在速度显著领先时仍保持更高地图质量。  
  30 FPS 受限模式下，平均 PSNR 更进一步到 **38.83 dB**，说明该系统能通过“放慢 tracking、给 mapping 更多迭代”继续换取质量。

- **真实场景信号：优势主要体现在耦合式方法中，但深度噪声会限制上限**  
  在 TUM-RGBD 上，作者优于大多数耦合式 dense SLAM（平均 ATE **2.4 cm** vs SplaTAM **3.2 cm**、GS-SLAM **3.7 cm**），但仍不如 Photo-SLAM / ORB-SLAM3 的 **1.3 cm**。  
  这说明论文的核心机制是有效的，但在真实噪声深度下，**传感器质量会成为新的主导瓶颈**。

- **消融信号：论文的关键设计基本都被单独验证了**
  - 椭圆尺度正则把 TUM 平均 ATE 从 plane regularization 的 **29.12 cm** 降到 **2.37 cm**；
  - 共享 G-ICP 协方差并做尺度对齐，让 Replica 平均 PSNR 从 **24.81** 抬到 **38.83**；
  - mapping-only keyframe 让系统同时拿到更好的 tracking 和 mapping；
  - 随机历史关键帧训练显著优于只训练最近关键帧；
  - pruning 有帮助，而 densifying 基本不再必要。

### 局限性

- Fails when: 深度图存在大面积缺失、强噪声、远距离稀疏采样或传感器较老时，G-ICP 的协方差估计和3D配准会退化，地图质量会明显下降；TUM 上落后于 Photo-SLAM/ORB-SLAM3 就体现了这一点。
- Assumes: 需要 RGB-D 输入、相对静态场景、kNN 协方差能代表局部几何，并默认较强算力支撑其实时性结论（文中实验硬件为 Ryzen 7 7800X3D + RTX 4090）；此外，高速模式默认接受“更少 mapping 迭代换更高 FPS”的折中。
- Not designed for: 纯单目 SLAM、动态场景建图、显式闭环检测/重定位/大规模全局一致性优化，以及以语义对象级建模为目标的系统。

### 可复用部件

这篇论文最值得迁移的，不只是“G-ICP + 3DGS”这个组合名词，而是下面几个可复用操作：

- **shared-gaussian-covariance**：把前端几何配准产生的协方差直接作为后端显式地图原语初始化；
- **ellipse-scale-regularization**：当目标地图高斯已经被学习过时，不要再用统一平面先验粗暴压尺度；
- **depth-aware scale aligning**：修正单帧深度导致的远距离高斯尺度膨胀；
- **mapping-only keyframes**：把 odometry 稳定性需求和渲染训练覆盖需求解耦；
- **random-history training + pruning**：实时高斯地图避免视角方向局部最小的一种低成本策略。

![[paperPDFs/SLAM_Video/arXiv_2024/2024_RGBD_GS_ICP_SLAM.pdf]]