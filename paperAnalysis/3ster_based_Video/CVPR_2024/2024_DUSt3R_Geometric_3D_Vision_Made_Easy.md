---
title: "DUSt3R: Geometric 3D Vision Made Easy"
venue: arXiv
year: 2023
tags:
  - Others
  - task/3d-reconstruction
  - task/camera-pose-estimation
  - pointmap-regression
  - cross-view-attention
  - global-alignment
  - dataset/CO3Dv2
  - dataset/DTU
  - dataset/ETH3D
  - repr/pointmap
  - opensource/no
core_operator: "将图像对直接回归到同一参考系下的稠密pointmap，并通过3D空间全局对齐统一多视图几何与相机"
primary_logic: |
  无标定单图/图像对/图像集合 → 共享编码器与跨视图解码器回归共享坐标系pointmap和置信度 → 通过3D全局对齐与几何读取恢复深度、对应、相机位姿和一致3D重建
claims:
  - "DUSt3R 512 在 CO3Dv2 的 10 帧多视图位姿评测中以全局对齐达到 96.2 RRA@15、86.8 RTA@15、76.7 mAA@30，显著高于 PoseDiffusion 的 80.5、79.8、66.5 [evidence: comparison]"
  - "在零样本单目深度上，DUSt3R 512 在 NYUv2 达到 6.50 AbsRel 和 94.09 δ1.25，优于 SlowTv 的 11.59 和 87.23，并接近监督方法 DPT-BEiT 的 5.40 和 96.54 [evidence: comparison]"
  - "CroCo 预训练与更高输入分辨率带来稳定收益，例如 NYUv2 的 AbsRel 从 14.51（224-NoCroCo）降至 10.28（224）再降至 6.50（512） [evidence: ablation]"
related_work_position:
  extends: "CroCo (Weinzaepfel et al. 2022)"
  competes_with: "COLMAP (Schönberger et al. 2016); DeepV2D (Teed and Deng 2020)"
  complementary_to: "NeRF (Mildenhall et al. 2020); PixSfM (Lindenberger et al. 2021)"
evidence_strength: strong
pdf_ref: "paperPDFs/3ster_based_Video/CVPR_2024/2024_DUSt3R_Geometric_3D_Vision_Made_Easy.pdf"
category: Others
---

# DUSt3R: Geometric 3D Vision Made Easy

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2312.14132), [Project](https://dust3r.europe.naverlabs.com)
> - **Summary**: 这篇工作把“先估相机、再做SfM/MVS”的传统链式几何流程，改写成“直接回归共享参考系下的稠密pointmap”，从而在未知内外参的单图、双图和多图设置下统一恢复深度、匹配、位姿与3D重建。
> - **Key Performance**: CO3Dv2 多视图位姿达到 96.2 RRA@15 / 76.7 mAA@30（GA）；NYUv2 零样本单目深度达到 6.50 AbsRel / 94.09 δ1.25。

> [!info] **Agent Summary**
> - **task_path**: 无标定单图/图像对/图像集合 -> 稠密pointmap、深度、匹配、相机位姿与一致3D重建
> - **bottleneck**: 传统 SfM→BA→MVS 依赖先验相机与链式中间结果，前面一步失效会级联破坏后续稠密重建
> - **mechanism_delta**: 用跨视图 Transformer 直接预测同一参考系下的双pointmap与置信度，再在3D空间做全局对齐而非最小化2D重投影误差
> - **evidence_signal**: 在 CO3Dv2/RealEstate10K 多视图位姿上显著超过 PoseDiffusion，且零样本深度在多个基准上接近或达到 SOTA
> - **reusable_ops**: [共享参考系pointmap回归, 3D全局对齐]
> - **failure_modes**: [天空与透明体等3D定义不清区域误差大, 需要亚像素三角化精度的受控MVS场景上不如显式相机方法]
> - **open_questions**: [如何保证pointmap满足物理相机一致性, 如何降低尺度歧义并提升未知内参下的绝对定位稳定性]

## Part I：问题与挑战

DUSt3R解决的不是普通的深度估计，而是更难的版本：**输入是一组任意拍摄的RGB图像，内参未知、位姿未知、视角和焦距都可能大幅变化，输出却希望是一致的3D几何与相机信息**。

### 真正的难点在哪里
传统几何重建通常走这条链：

1. 找局部特征和匹配；
2. 估本质矩阵/相对位姿；
3. 三角化得到稀疏点；
4. 做 SfM 和 BA；
5. 再做稠密 MVS。

真正的瓶颈，不是某个单点模块不够强，而是这条链本身有**强前置依赖**。  
一旦相机估计在前面出错，后面的稠密重建几乎必然被污染。论文明确指出，少视角、非Lambertian表面、相机运动不足、未知焦距等现实条件，都会让 SfM 很脆弱。

### 这篇论文想改掉什么
作者的核心判断是：**相机不该再是必须先求准的中间变量**。  
与其先解一串耦合的几何最小问题，不如让模型直接输出一个足够丰富的3D表示，使得：

- 场景几何在表示里；
- 像素和3D点的对应关系在表示里；
- 两个视角之间的关系也在表示里。

这样，深度、匹配、相机位姿、重建就不再是串行依赖，而是从同一个表示中“读出来”。

### 输入/输出接口
- **输入**：单张图、图像对、或多张无序图像集合。
- **输出**：pairwise 时输出两个 pointmap 和对应置信度；二者都落在第一张图的参考系。
- **可恢复量**：深度图、像素对应、相对位姿、绝对位姿、全局一致3D点云。

### 为什么现在能做
这件事之所以现在可行，靠的是两点：

- 有足够大的多域3D监督数据混合（文中共 8.5M 图像对）；
- 可以利用 **CroCo** 这类跨视图预训练，把“跨视角几何先验”先学到模型里。

### 边界条件
- 输出天然存在**尺度歧义**；
- 默认**一条射线只命中一个表面点**，因此天空、透明体、半透明区域更难；
- 内参恢复采用简化假设：**主点近中心、像素近似方形**；
- 多图场景仍需要后处理的全局对齐，不是一次前向就直接得到全局最优场景。

## Part II：方法与洞察

这篇论文最关键的不是网络更深，而是**把几何问题的表示层改了**。

### 方法主线

#### 1. 用 pointmap 取代 depth+pose 的分离输出
每个像素不再只预测深度，而是直接预测一个 3D 点坐标。  
对输入图像对 \((I_1, I_2)\)，网络输出：

- 第一张图的 pointmap \(X_{1,1}\)
- 第二张图的 pointmap \(X_{2,1}\)

注意：**两个 pointmap 都表达在第一张图的坐标系里**。  
这是整个方法最关键的设计。

它的结果是：模型不需要先显式求相机再三角化，而是直接学习“这个像素在共享3D里应该落在哪里”。

#### 2. 用跨视图 Transformer 让两张图共同决定几何
架构上是共享权重的 ViT 编码器 + 双分支 Transformer 解码器。  
重点不在 Siamese 本身，而在**decoder 中持续的 cross-attention 信息交换**：

- 每个视图先看自己；
- 再看另一视图；
- 两支路在解码过程中不断共享信息。

这意味着对应关系、遮挡判断、形状先验，不再是后处理阶段才做，而是在特征推理阶段就共同决定。

#### 3. 用简单的3D回归 + 置信度学习训练
训练目标很直接：让预测的 pointmap 贴近 GT pointmap。  
作者没有在推理时强行施加几何约束，而是依靠监督数据让模型学会“物理上合理的几何”。

同时，模型还学习一个**逐像素置信度**，用于：
- 下调天空、透明体、遮挡等难区域的影响；
- 在后续匹配、融合、全局对齐时做加权。

#### 4. 多图时做 3D global alignment，而不是传统 BA
pairwise 网络只能看两张图。  
对于多图集合，作者先构图，再对每条边的 pairwise pointmap 做全局对齐，目标不是最小化 2D 重投影误差，而是直接在 **3D 空间**里对齐各 pair 输出。

这一步的好处是：
- 优化更直接；
- 收敛更快；
- 实现上比经典 BA 更简单。

### 核心直觉

**改变了什么？**  
从“显式相机 + 深度/匹配分头求解”改成“共享参考系下的稠密 pointmap 回归”。

**改变了哪个瓶颈？**  
把原来多个相互依赖、前后串行的几何子问题，变成一个统一的密集回归问题。  
尤其是把“必须先有准相机”这个硬约束，改成“通过跨视图信息交互直接学习几何一致性”。

**带来了什么能力变化？**
- 单目：直接用 \(F(I,I)\) 也能工作；
- 双目：即使大视角变化，也能直接出共享3D；
- 多图：通过 3D 对齐得到全局一致点云；
- 下游任务：深度、匹配、相机位姿都从同一表示读取。

换句话说，这篇论文不是把 SfM/MVS 的每个零件都替换成神经网络，而是试图**绕开这条零件链**。

### 为什么这个设计有效
核心因果链可以概括为：

**输出表示变了**  
→ 不再强依赖显式针孔模型与相机预估  
→ correspondence / geometry / relative viewpoint 可以联合推理  
→ 从无标定图像中直接恢复可用的3D几何

更具体地说：

- **pointmap** 保留了像素到3D的对齐关系，因此不像纯 latent 3D 表示那样难以回到图像空间；
- **共享参考系** 让两个视图天然可比较，不必先显式恢复相机；
- **cross-attention** 让一个视图补另一个视图的信息缺口；
- **confidence** 让模型学会在 ill-posed 区域保守；
- **3D global alignment** 让多图优化目标与网络输出形式一致。

### 从 pointmap 读出下游结果
这也是方法“统一性”最强的地方：

- **像素匹配**：在 3D pointmap 空间做最近邻/互近邻；
- **内参恢复**：从相机坐标系 pointmap 拟合焦距；
- **相对位姿**：用 Procrustes 或 PnP-RANSAC；
- **绝对位姿**：将 query 与数据库图像匹配，再转到世界坐标；
- **深度图**：直接取 pointmap 的 z 分量；
- **多图重建**：对齐所有 pairwise pointmap。

### 战略权衡

| 设计选择 | 得到的能力 | 代价/风险 |
|---|---|---|
| 共享参考系 pointmap 表示 | 不依赖已知相机即可统一输出几何、对应与位姿 | 输出不一定严格满足真实针孔相机，且有尺度歧义 |
| 跨视图 Transformer 解码 | 在特征层联合解决匹配与形状 | 模型较重，依赖大规模预训练和数据 |
| 置信度联合预测 | 对天空、透明体、遮挡更稳健，也便于融合 | 置信度无显式监督，可能通过“降权”绕开难点 |
| 3D global alignment 代替 BA | 优化快、实现简单、与 pointmap 形式匹配 | 超高精度场景下不如显式重投影与亚像素三角化 |

## Part III：证据与局限

### 关键实验信号

- **信号1：多视图位姿比较**
  - 在 CO3Dv2 上，DUSt3R 512 + Global Alignment 达到 **96.2 RRA@15 / 86.8 RTA@15 / 76.7 mAA@30**。
  - 对比 PoseDiffusion 的 **80.5 / 79.8 / 66.5**，提升很明显。
  - 这说明：**共享参考系 pointmap + 3D 对齐** 确实能稳定承载跨视图几何关系，而不是只会输出“看起来像深度”的结果。

- **信号2：零样本单目深度比较**
  - 直接把同一张图喂两次 \(F(I,I)\)，在 NYUv2 上达到 **6.50 AbsRel / 94.09 δ1.25**。
  - 这优于零样本基线 SlowTv 的 **11.59 / 87.23**，并接近监督方法。
  - 这说明：模型学到的不是窄任务的双目技巧，而是更一般的几何先验。

- **信号3：无GT相机的多视图深度**
  - 在不使用 GT pose/intrinsics 的设置下，DUSt3R 512 在 ETH3D 上达到 **2.91 rel / 76.91 τ**，总体平均 **4.73 / 64.52**。
  - 相比 Robust MVD Baseline 的 **6.3 / 56.0** 更强。
  - 这证明：即便不走传统“先相机、后MVS”的流程，也能做出有竞争力的 calibration-free MVS。

- **信号4：消融实验**
  - CroCo 预训练与高分辨率输入都显著增益性能。
  - 例如 NYUv2 的 AbsRel 从 **14.51（224-NoCroCo）→ 10.28（224）→ 6.50（512）**。
  - 这说明方法有效，但也说明其成功并不只来自表示本身，还强依赖**预训练几何先验 + 高分辨率视觉细节**。

### 1-2 个最能说明问题的指标
- **CO3Dv2**：96.2 RRA@15 / 76.7 mAA@30（多视图位姿）
- **NYUv2**：6.50 AbsRel / 94.09 δ1.25（零样本单目深度）

### 局限性
- **Fails when**: 天空、透明/半透明表面、几何定义本身不清楚的区域；以及需要毫米级、亚像素三角化精度的受控MVS场景，此时回归式 pointmap 明显不如显式相机与域内训练方法。
- **Assumes**: 大规模有监督数据混合与 CroCo 预训练；每条射线只对应单一表面；焦距恢复默认主点近中心且像素方形；多图推理与全局对齐默认有较强 GPU 资源支撑（文中报告 H100 上 pair inference 约 40ms）。
- **Not designed for**: 严格实时 SLAM、动态/非刚体场景建模、天然带绝对尺度保证的 metric reconstruction，以及必须显式满足物理相机模型的高精度摄影测量流程。

补充一个很具体的边界：附录中未知焦距的 visual localization 在 Cambridge 上误差明显变大，作者归因于参考 pointmap 稀疏导致尺度无法稳定锚定。也就是说，**当“从 pointmap 读出相机”这一步缺少几何支撑时，绝对位姿恢复仍然脆弱**。

### 可复用组件
- **pointmap 表示**：适合任何想把深度、对应、位姿统一到一个输出空间的几何任务。
- **跨视图共享解码器**：适合需要在特征层联合推理 correspondence 与 geometry 的多视图模型。
- **置信度图**：可直接用于过滤、加权融合、pair 筛选。
- **3D global alignment**：适合作为 pairwise 几何预测到 scene-level 一致重建之间的轻量桥梁。

## Local PDF reference

![[paperPDFs/3ster_based_Video/CVPR_2024/2024_DUSt3R_Geometric_3D_Vision_Made_Easy.pdf]]