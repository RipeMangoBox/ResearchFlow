---
title: 'TrackingWorld: World-centric Monocular 3D Tracking of Almost All Pixels'
type: paper
paper_level: A
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 世界坐标系单目稠密三维跟踪
- TrackingWorld
- TrackingWorld achieves accurate and
acceptance: Poster
cited_by: 4
code_url: https://github.com/IGL-HKUST/TrackingWorld
method: TrackingWorld
modalities:
- Video
paradigm: optimization-based
---

# TrackingWorld: World-centric Monocular 3D Tracking of Almost All Pixels

[Code](https://github.com/IGL-HKUST/TrackingWorld)

**Topics**: [[T__Object_Tracking]], [[T__3D_Reconstruction]] | **Method**: [[M__TrackingWorld]] | **Datasets**: Sintel Camera Pose Estimation, ADT World-Coordinate 3D Tracking, Sintel Video Depth

> [!tip] 核心洞察
> TrackingWorld achieves accurate and dense world-centric 3D tracking of almost all pixels by lifting sparse 2D tracks to dense tracks, filtering redundant tracks, and optimizing camera poses with 3D back-projection.

| 中文题名 | 世界坐标系单目稠密三维跟踪 |
| 英文题名 | TrackingWorld: World-centric Monocular 3D Tracking of Almost All Pixels |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2512.08358) · [Code](https://github.com/IGL-HKUST/TrackingWorld) · [Project](待补充) |
| 主要任务 | World-Centric 3D Tracking, Camera Pose Estimation, Video Depth Estimation |
| 主要 baseline | Uni4D, DELTA, CoTrackerV3, OmniMotion, SpatialTracker, MonST3R, DUSt3R |

> [!abstract] 因为「现有单目三维跟踪方法无法将相机运动与前景动态运动分离，且无法稠密跟踪视频中新出现的动态物体」，作者在「Uni4D」基础上改了「增加 tracking upsampler 进行稀疏到稠密的二维轨迹提升、连通域过滤消除冗余重叠轨迹、clip-to-global 并行优化替代顺序优化，并显式引入世界坐标系静动态分解」，在「Sintel / ADT / Bonn / TUM-D」上取得「ATE 0.087（相对 Uni4D+DELTA 降低 26.3%）、APD3D 75.18（提升 9.0%）、Sintel video depth Abs Rel 0.222（SOTA）」

- **Sintel camera pose**: ATE 0.087 vs Uni4D+DELTA 0.118（-26.3%），RTE 0.036 vs 0.048（-25.0%），RPE 0.406 vs 0.610（-33.4%）
- **ADT world-coordinate 3D tracking**: APD3D 75.18 vs Uni4D+DELTA 68.95（+9.0%）
- **Sintel video depth**: Abs Rel 0.222，优于 MonST3R 0.335（-33.7%）和 DUSt3R 0.422（-47.4%），δ<1.25 达到 72.6%

## 背景与动机

从单目视频中恢复每个像素在三维世界中的运动轨迹，是计算机视觉的核心挑战之一。想象一段手持相机拍摄街道的视频：背景建筑因相机移动而 apparent motion，而行人则有自己的真实运动——现有方法往往将这两类运动混为一谈，要么输出相机坐标系下的轨迹（无法区分真实物体运动与相机运动），要么只能稀疏跟踪预定义的点，无法处理新进入画面的动态物体。

现有方法的处理方式各有局限：
- **OmniMotion**（Tracking Everything Everywhere All at Once）通过神经场表示实现稠密跟踪，但隐式编码无法显式分离相机与物体运动，且对长序列存在漂移；
- **SpatialTracker** 将二维像素跟踪到三维空间，但仍缺乏显式的世界坐标系建模，静态背景与动态前景未被解耦；
- **Uni4D** 采用顺序优化联合估计相机位姿和三维轨迹，然而计算开销大（Sintel 上 19 分钟、ADT 上 28 分钟），且直接将稀疏二维轨迹输入三维重建，未处理稠密化过程中的冗余问题。

这些方法的共同瓶颈在于：**没有显式的世界坐标系框架来分离静态背景（仅受相机运动影响）与动态前景（有独立运动）**，同时**缺乏从稀疏到稠密的高效跟踪提升机制**，导致无法"跟踪几乎所有像素"且计算效率低下。TrackingWorld 正是针对这一缺口，提出了一套以世界坐标系为中心、从稀疏二维轨迹出发的稠密三维跟踪框架。
![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/b3ebe44f-a815-4f6d-bcc7-110ae69e0c6c/figures/fig_001.png)
*Figure: TrackingWorld estimates world-centric dense tracking results from monocular videos. Our*



## 核心创新

核心洞察：显式建立世界坐标系并将场景分解为静态背景与动态前景，因为相机运动与物体运动在世界坐标下具有可分离的几何约束，从而使联合优化相机位姿和稠密三维轨迹时避免运动耦合歧义成为可能。

| 维度 | Baseline (Uni4D 等) | 本文 (TrackingWorld) |
|:---|:---|:---|
| 坐标系 | 相机坐标系或隐式编码，无显式世界坐标 | 显式世界坐标系，Ostatic 掩码分离静动态 |
| 数据流 | 稀疏二维轨迹直接输入三维重建 | 稀疏轨迹 → upsampler → 稠密轨迹 → 连通域过滤 |
| 优化策略 | 顺序优化（sequential） | Clip-to-global 并行优化，动态掩码过滤 |
| 冗余处理 | 无，重叠区域轨迹冗余 | 连通域分析，τ=50 阈值剔除孤立像素和冗余轨迹 |
| 运行效率 | Sintel 19 min / ADT 28 min | Sintel 15 min / ADT 20 min |

## 整体框架


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/b3ebe44f-a815-4f6d-bcc7-110ae69e0c6c/figures/fig_002.png)
*Figure: Overview. Given a video sequence, TrackingWorld first generates dense 2D tracking results*



TrackingWorld 采用多阶段流水线，从单目视频输入到世界坐标系下的稠密三维跟踪输出，核心数据流如下：

1. **Sparse 2D Tracker**（输入：原始视频帧；输出：稀疏二维点轨迹）：调用现成跟踪器（DELTA 或 CoTrackerV3）产生初始稀疏对应点；
2. **Tracking Upsampler**（输入：稀疏二维轨迹；输出：稠密二维轨迹）：逐帧将稀疏轨迹提升到像素级密度，实现"几乎所有像素"的覆盖；
3. **Redundancy Filter**（输入：含重叠的稠密二维轨迹；输出：过滤后的干净稠密轨迹）：通过连通域分析，保留 |C_i| > τ=50 的连通区域，剔除孤立像素和冗余重叠轨迹；
4. **Depth & Mask Priors**（输入：视频帧；输出：单目深度估计 + 动态物体分割）：UniDepth 提供深度先验，动态分割器提供 Ostatic 静态掩码；
5. **Clip-to-Global Parallel Optimizer**（输入：过滤后的稠密轨迹 + 深度 + 动态掩码；输出：相机位姿 P + 世界坐标三维轨迹 X）：并行优化替代 Uni4D 的顺序优化，联合最小化重投影误差与光度一致性损失。

```
Video ──► [Sparse 2D Tracker] ──► sparse tracks ──► [Tracking Upsampler] ──► dense tracks
                                                                              │
                                        Depth + Masks ◄── [UniDepth + Segmentor]
                                              │
                                              ▼
                    dense tracks ──► [Redundancy Filter: |C_i| > 50] ──► clean dense tracks
                                                                              │
                                                                              ▼
                                    [Clip-to-Global Parallel Optimizer]
                                              │
                                    ┌─────────┴─────────┐
                                    ▼                   ▼
                              Camera Poses P      World 3D Tracks X
```

## 核心模块与公式推导

### 模块 1: 连通域冗余过滤（对应框架图 Redundancy Filter 位置）

**直觉**: 稠密上采样后，重叠帧区域会产生冗余轨迹和孤立噪声点，需通过几何连贯性约束保证保留的轨迹对应实际物体部件。

**Baseline 做法** (Uni4D 等): 无显式过滤，稀疏轨迹直接输入或简单稠密化后直接进入优化，导致重叠区域冗余和孤立像素干扰重建。

**变化点**: 引入帧间可见性补集计算，仅处理新出现的轨迹区域，并通过连通域大小阈值剔除几何无意义的孤立像素。

**本文公式（推导）**:
$$\text{Step 1: 计算补集区域} \quad \mathcal{T}_{\text{new}} = \mathcal{T}_{\text{dense}} \text{setminus} \mathcal{T}_{\text{prev}}$$
$$\text{Step 2: 连通域标记} \quad \{C_i\} = \text{ConnectedComponents}(\mathcal{T}_{\text{new}})$$
$$\text{Step 3: 尺寸阈值过滤} \quad \mathcal{C}_{\text{retain}} = \{C_i : |C_i| > \tau, \tau = 50\}$$
$$\text{最终保留轨迹} \quad \mathcal{T}_{\text{clean}} = \text{bigcup}_{C_i \in \mathcal{C}_{\text{retain}}} C_i$$

符号: $\mathcal{T}_{\text{dense}}$ = 上采样后的稠密轨迹集, $\mathcal{T}_{\text{prev}}$ = 先前帧已可见的轨迹, $C_i$ = 第 $i$ 个连通域, $\tau$ = 大小阈值（固定为 50 像素）

**对应消融**: Table 12 显示去掉过滤机制后，Sintel 上 ATE 从 0.088 升至 0.105（+19.3%），RTE 0.035→0.038（+8.6%），RPE 0.410→0.442（+7.8%）；Bonn 上 RTE 恶化更显著 0.005→0.007（+40%）。

---

### 模块 2: 世界坐标系静动态分解（对应框架图 Ostatic / Energy Decomposition 位置）

**直觉**: 静态背景仅受相机运动影响，动态前景有独立运动——在世界坐标系下显式分离两者，可避免优化中的运动耦合歧义。

**Baseline 做法** (现有方法): 统一在相机坐标系或隐式场中处理，无显式静动态分离，导致相机位姿估计受前景动态物体干扰。

**变化点**: 引入 Ostatic 掩码显式标记静态区域，将总能量分解为静态项与动态项分别约束。

**本文公式（推导）**:
$$\text{Step 1: 掩码定义} \quad \mathcal{M}_{\text{static}} = O_{\text{static}}(I_t), \quad \mathcal{M}_{\text{dynamic}} = 1 - \mathcal{M}_{\text{static}}$$
$$\text{Step 2: 能量分解} \quad E_{\text{total}} = E_{\text{static}} + E_{\text{dynamic}}$$
$$\text{其中} \quad E_{\text{static}} = \sum_{i \in \mathcal{M}_{\text{static}}, t} \rho\left(\|\pi(P_t, X_i^{\text{world}}) - x_{i,t}\|^2\right) + \lambda_{\text{photo}} E_{\text{photo}}^{\text{static}}$$
$$E_{\text{dynamic}} = \sum_{i \in \mathcal{M}_{\text{dynamic}}, t} \rho\left(\|\pi(P_t, X_i^{\text{world}}) - x_{i,t}\|^2\right) + \lambda_{\text{flow}} E_{\text{flow}}^{\text{dynamic}}$$
$$\text{Step 3: 联合优化} \quad \min_{P, X^{\text{world}}} E_{\text{total}}$$

符号: $O_{\text{static}}$ = 静态区域掩码估计器, $P_t$ = 第 $t$ 帧相机位姿, $X_i^{\text{world}}$ = 第 $i$ 个点的世界坐标, $\pi$ = 投影函数, $\rho$ = 鲁棒核函数

**对应消融**: Table 12 中静动态分解的消融显示，该机制对相机位姿估计的稳定性至关重要，与过滤机制协同作用。

---

### 模块 3: Clip-to-Global 并行优化（对应框架图 Optimizer 位置）

**直觉**: 顺序优化在长视频上累积误差且无法并行，将视频分 clip 并行处理再全局融合，可同时提升效率和精度。

**Baseline 公式** (Uni4D 顺序优化):
$$\min_{P_{1:T}, X_{1:N}} \sum_{t=1}^{T} \sum_{i=1}^{N} \rho\left(\|\pi(P_t, X_i) - x_{i,t}\|^2\right) + \lambda E_{\text{photo}} \quad \text{（逐帧顺序求解）}$$

符号: $T$ = 总帧数, $N$ = 轨迹数, 顺序求解导致 $O(T)$ 的依赖链

**变化点**: 将序列划分为可重叠的 clip，clip 内并行优化，clip 间通过共享关键帧全局对齐，配合动态掩码过滤减少优化变量。

**本文公式（推导）**:
$$\text{Step 1: Clip 划分} \quad \{C_k\}_{k=1}^{K}, \quad C_k = \{t : t_k^{\text{start}} \leq t \leq t_k^{\text{end}}\}$$
$$\text{Step 2: Clip 内并行优化} \quad \forall k: \min_{P^{(k)}, X^{(k)}} E_{\text{total}}^{(k)} \quad \text{（独立并行）}$$
$$\text{Step 3: 动态掩码过滤减少变量} \quad \mathcal{V}_{\text{active}}^{(k)} = \{i : \mathcal{M}_{\text{dynamic}}(i, t) = 1, t \in C_k\}$$
$$\text{Step 4: 全局对齐} \quad \min_{\{P_t\}} \sum_{k} \sum_{t \in C_k \cap C_{k+1}} \|P_t^{(k)} - P_t^{(k+1)}\|^2$$
$$\text{最终联合目标} \quad L_{\text{final}} = \sum_{k} E_{\text{total}}^{(k)} + \mu E_{\text{global-align}}$$

符号: $K$ = clip 数量, $\mathcal{V}_{\text{active}}^{(k)}$ = clip $k$ 中的活跃轨迹集, $\mu$ = 全局对齐权重

**对应消融**: Table 11 显示相比 Uni4D 顺序优化，运行时间从 Sintel 19 min 降至 15 min（-21.1%），ADT 从 28 min 降至 20 min（-28.6%），同时 ATE/RTE/RPE 均有改善。

## 实验与分析



本文在四个基准数据集上评估：Sintel（相机位姿估计与视频深度）、ADT（世界坐标三维跟踪）、Bonn 和 TUM-D（视频深度）。核心结果如 Table 10 与 Table 11 所示：在 Sintel 相机位姿估计上，TrackingWorld 以 DELTA 为跟踪输入时达到 ATE 0.087、RTE 0.036、RPE 0.406，相对 Uni4D+DELTA 基线分别降低 26.3%、25.0% 和 33.4%；在 ADT 世界坐标三维跟踪上，APD3D 达到 75.18，超越 Uni4D+DELTA 的 68.95，提升 9.0%。视频深度方面，Sintel 上 Abs Rel 0.222 为所有方法最优，显著优于 MonST3R 的 0.335（-33.7%）和 DUSt3R 的 0.422（-47.4%），δ<1.25 达到 72.6%；TUM-D 上 Abs Rel 0.086 同样领先，较 Unidepth 的 0.113 降低 22.5%，较 MonST3R 的 0.301 降低 71.4%。


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/b3ebe44f-a815-4f6d-bcc7-110ae69e0c6c/figures/fig_003.png)
*Figure: Qualitative results on DAVIS dataset. Our method can output both reliable camera*





消融实验（Table 12）聚焦冗余过滤机制：去掉连通域过滤后，Sintel 上 ATE 从 0.088 恶化至 0.105（+0.017，+19.3%），RTE 0.035→0.038（+8.6%），RPE 0.410→0.442（+7.8%）；Bonn 上 RTE 从 0.005 升至 0.007（+40%），表明过滤机制对维持轨迹几何一致性至关重要。此外，clip-to-global 并行优化在 Table 11 中验证了效率收益：Sintel 30-50 帧从 Uni4D 的 19 min 降至 15 min，ADT 前 64 帧从 28 min 降至 20 min。

公平性检验：对比基线中，Uni4D 作为主要方法基线直接可比，MonST3R 和 DUSt3R 在视频深度上为强基线且被超越。但 TAPIP3D 和 SpatialTracker 虽被引用为直接基线，未出现在主实验对比表中；St4RTrack 仅在 limitations 中讨论其漂移问题而无定量比较。运行时间比较仅覆盖 Sintel 帧 30-50 和 ADT 前 64 帧，范围有限。方法依赖上游跟踪器（DELTA/CoTrackerV3）、深度估计（UniDepth）和动态分割的质量，这些辅助模型的失败模式未被充分分析。固定阈值 τ=50 虽声称跨场景鲁棒，但仅在标准基准上验证。

## 方法谱系与知识库定位

TrackingWorld 属于 **optimization-based 3D reconstruction and tracking** 方法家族，直接父方法为 **Uni4D**（顺序优化联合相机位姿与三维轨迹估计）。方法演进路径：Uni4D 的顺序优化 → TrackingWorld 的 clip-to-global 并行优化 + 显式世界坐标系 + 数据流水线增强（upsampler + filtering）。

**直接基线与差异**：
- **Uni4D**：核心基线，TrackingWorld 替换其顺序优化为并行优化，增加数据预处理和静动态分解；
- **OmniMotion**：稠密跟踪先驱，TrackingWorld 与之区别在于显式世界坐标系而非隐式神经场；
- **SpatialTracker / TAPIP3D**：同为世界坐标跟踪概念，但 TrackingWorld 通过 upsampler 实现"几乎所有像素"的稠密覆盖；
- **MonST3R / DUSt3R**：几何运动联合估计方法，TrackingWorld 在视频深度指标上超越之，但依赖其深度先验作为输入。

**变化槽位**：data_pipeline（新增 upsampler + filtering）、inference_strategy（顺序→并行）、architecture（新增世界坐标系静动态分解）、training_recipe（保持 optimization-based，但引入辅助模型依赖）。

**后续方向**：(1) 端到端训练替代辅助模型依赖；(2) 场景自适应阈值 τ 替代固定值 50；(3) 实时化：当前 30 帧视频约 20 分钟，需进一步压缩以满足在线应用。

**标签**：modality=video | paradigm=optimization-based | scenario=dynamic scene understanding | mechanism=dense tracking + world-centric decomposition | constraint=monocular input, auxiliary model dependent

## 引用网络

### 直接 baseline（本文基于）

- TAPIP3D: Tracking Any Point in Persistent 3D Geometry _(NeurIPS 2025, 直接 baseline, 未深度分析)_: TAPIP3D explicitly tracks points in persistent 3D geometry, which is essentially
- CoTracker3: Simpler and Better Point Tracking by Pseudo-Labelling Real Videos _(ICCV 2025, 实验对比, 未深度分析)_: Recent state-of-the-art point tracking method; likely compared against in experi
- SpatialTracker: Tracking Any 2D Pixels in 3D Space _(CVPR 2024, 直接 baseline, 未深度分析)_: Closely related 3D-aware tracking method; directly comparable approach tracking 
- DELTA: DENSE EFFICIENT LONG-RANGE 3D TRACKING FOR ANY VIDEO _(ICLR 2025, 实验对比, 未深度分析)_: Dense 3D tracking method; recent competitor in same space, likely compared in ex
- St4RTrack: Simultaneous 4D Reconstruction and Tracking in the World _(ICCV 2025, 实验对比, 未深度分析)_: Simultaneous 4D reconstruction and world tracking; very closely related to world

