---
title: "Pixel-GS: Density Control with Pixel-aware Gradient for 3D Gaussian Splatting"
venue: arXiv
year: 2024
tags:
  - 3D_Gaussian_Splatting
  - task/novel-view-synthesis
  - density-control
  - gradient-reweighting
  - gradient-scaling
  - "dataset/Mip-NeRF 360"
  - "dataset/Tanks & Temples"
  - opensource/full
core_operator: "按每视角覆盖像素数重加权3DGS增密梯度，并按深度缩放近相机梯度以促进稀疏区生长、抑制floater"
primary_logic: |
  多视图图像+相机位姿/SfM稀疏点云
  → 将Gaussian的跨视角NDC梯度从等权平均改为按覆盖像素数加权，并对近相机梯度做深度平方缩放
  → 更合理的split/clone决策、更均匀的点增长与更高保真的新视角渲染
claims:
  - "在 Mip-NeRF 360 全场景上，Pixel-GS 相比 retrained 3DGS* 将 LPIPS 从 0.202 降到 0.176，同时 PSNR/SSIM 也提升 [evidence: comparison]"
  - "在 Tanks & Temples 上，仅用 pixel-aware gradient 会显著恶化结果，而加入 scaled gradient field 后完整模型达到最佳平均结果 24.38 PSNR / 0.850 SSIM / 0.178 LPIPS，说明深度缩放是抑制 floater 的关键 [evidence: ablation]"
  - "随机丢弃初始化 SfM 点后，Pixel-GS 始终优于 3DGS；即使丢弃 99% 初始化点，其 LPIPS 仍优于使用完整 SfM 点初始化的 3DGS [evidence: comparison]"
related_work_position:
  extends: "3D Gaussian Splatting (Kerbl et al. 2023)"
  competes_with: "3D Gaussian Splatting (Kerbl et al. 2023); GaussianPro (Cheng et al. 2024)"
  complementary_to: "Mip-Splatting (Yu et al. 2023); LightGaussian (Fan et al. 2023)"
evidence_strength: strong
pdf_ref: paperPDFs/Densification_Rendering/Lecture_Notes_in_Computer_Science_2025/2025_Pixel_GS_Density_Control_with_Pixel_aware_Gradient_for_3D_Gaussian_Splatting.pdf
category: 3D_Gaussian_Splatting
---

# Pixel-GS: Density Control with Pixel-aware Gradient for 3D Gaussian Splatting

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2403.15530), [Project](https://pixelgs.github.io)
> - **Summary**: 这篇工作不改3DGS主体渲染框架，而是重写“何时该长点”的统计方式：用每个视角的覆盖像素数给梯度加权，再用深度缩放抑制近相机伪点，从而更有效地修复稀疏初始化区域的模糊与针状伪影。
> - **Key Performance**: Mip-NeRF 360 上 LPIPS 0.176 vs 3DGS* 0.202；Tanks & Temples 上 LPIPS 0.178 vs 0.194。

> [!info] **Agent Summary**
> - **task_path**: 多视图图像 + 相机位姿/SfM稀疏点云 -> 3D Gaussian 场景表示 -> 新视角渲染
> - **bottleneck**: 3DGS 用跨视角等权平均的 NDC 梯度做增密判据，稀疏初始化区域的大 Gaussian 会被大量“只扫到边缘”的视角稀释掉有效生长信号
> - **mechanism_delta**: 把增密梯度改成按覆盖像素数加权的跨视角平均，并对近相机区域施加深度平方缩放
> - **evidence_signal**: 双数据集比较 + 组件消融 + 初始化点云随机丢弃鲁棒性实验
> - **reusable_ops**: [pixel-count-weighted densification, depth-scaled gradient field]
> - **failure_modes**: [仅用 pixel-aware gradient 时大场景近相机区域易出现 floater, 固定 gamma_depth 可能过度抑制近景真实细节]
> - **open_questions**: [能否自适应学习深度缩放强度, 能否与压缩/抗锯齿/动态3DGS无缝叠加]

## Part I：问题与挑战

这篇 paper 针对的不是“3DGS 渲染不够快”或“表达能力不够强”，而是一个更底层、也更实际的瓶颈：**3DGS 的 densification 决策对初始 SfM 点云质量过于敏感**。

### 真问题是什么
输入仍然是标准 3DGS 设置：**多视图图像 + 已知相机位姿 + SfM 稀疏点云**。输出是可实时渲染的新视角 3D Gaussian 场景表示。

问题在于，SfM 在以下区域经常给不出足够好的初始化点：
- 重复纹理区域
- 低纹理区域
- 观测视角少的区域

这些区域会被初始化成**更大尺度的 Gaussian**。而原始 3DGS 的增密条件，是看一个 Gaussian 在可见视角上的 **NDC 坐标梯度均值** 是否超过阈值。这里的关键缺陷是：

- 它按“视角”做等权平均；
- 但真正产生有效生长信号的，不是“看到了几个视角”，而是“这些视角里有多少像素真正覆盖到这个 Gaussian 的有效中心区域”。

于是，大 Gaussian 虽然被很多视角看到，但很多视角只扫到投影边缘，这些视角贡献的梯度很弱，却仍然和“真正看到了中心区域”的视角有同等投票权。结果就是：

**该长点的地方长不出来，不该长点的地方却可能被阈值调低后盲目增密。**

### 为什么现在值得解决
3DGS 已经成为实时新视角合成的主流基座之一，很多后续工作在做：
- 抗锯齿
- 压缩
- 动态场景
- 反射建模

但如果 densification 本身在稀疏初始化区域就失真，那么上层优化都建立在一个“不够对的点分布”之上。这个问题越接近真实采集场景，就越会暴露出来，因此是一个很值得优先修补的基础环节。

### 边界条件
这篇方法的适用前提很明确：
- 依赖 **静态场景**
- 依赖 **已知相机位姿与 SfM 初始化点云**
- 作用点是 **3DGS 的 split/clone 决策**
- 不试图替代 SfM / pose estimation / appearance model 本身

---

## Part II：方法与洞察

作者的策略非常克制：**不引入额外深度、法线或外部先验，只改“增密信号如何统计”**。这让 Pixel-GS 更像一个可插拔的 densification 修补件，而不是另起一套系统。

### 核心直觉

#### 1) 从“按视角平均”改成“按像素支持加权”
变化前：
- 每个视角对 Gaussian 是否该 split/clone，权重一样。

变化后：
- 每个视角的梯度按该 Gaussian 在该视角下**参与计算的像素数**加权。

本质变化是：

**统计单位从“看到了多少视角”变成“这些视角里有多少真实有效的投影支持”。**

这为什么有效？
- Gaussian 投影的梯度主要来自**靠近投影中心**的少量像素；
- 若某个视角只覆盖边缘，虽然 technically 可见，但对 densification 几乎没帮助；
- 覆盖像素数更大的视角，更可能真的覆盖到投影中心区域；
- 因此，像素数加权相当于让“真正有信息的视角”拥有更高投票权。

能力变化：
- **大 Gaussian 更容易在稀疏初始化区域触发 split/clone**
- **小 Gaussian 基本不受影响**，因为它们跨视角的覆盖像素数变化本来就不大

#### 2) 再用深度缩放修正空间梯度场
Pixel-aware gradient 会让该长点的地方更容易长点，但也会带来副作用：**近相机 floater 更容易增殖**。

作者观察到，近相机区域由于投影面积更大，会天然拿到更强的优化信号。于是他们引入第二个旋钮：

- 对 NDC 梯度再乘一个**随深度增加而增大的缩放因子**
- 近处衰减更多，远处保留更多
- 这样可以抑制近相机区域过快、过量的点增长

这一步不是在提升表达力，而是在**重平衡空间中的优化速度**。

### 机制拆解

#### A. Pixel-aware Gradient
可以把它理解成一句话：

> 对一个 Gaussian，不再问“它在多少个视角上平均梯度大不大”，而是问“按各视角投影覆盖像素数加权后，它的有效梯度大不大”。

这样做的结果是：
- 稀疏区域的大 Gaussian 不再被大量边缘视角冲淡
- dense 区域的小 Gaussian 与原始 3DGS 行为差异较小
- 因此额外点数更偏向长在“真缺点”的地方，而不是全局无差别膨胀

#### B. Scaled Gradient Field
可以把它理解成一句话：

> 不同深度区域的梯度不该天然享受同等增长权，近相机区域需要被抑制，否则更容易形成挡住后方几何的 floater。

它借鉴了 Floaters No More 的分析：近处区域的投影像素数与深度近似呈平方反比，因此需要对应的深度相关缩放来平衡优化速度。

### 战略取舍

| 设计 | 改变的瓶颈 | 直接收益 | 代价 / 风险 |
| --- | --- | --- | --- |
| Pixel-aware Gradient | 修正“视角等权平均”导致的大 Gaussian 梯度稀释 | 稀疏初始化区更容易增密，减少模糊与针状伪影 | 点数和内存上升；单独使用时会放大近相机 floater |
| Scaled Gradient Field | 修正“近处天然拿更多梯度”的空间不平衡 | 抑制近相机伪点，稳定大场景训练 | 依赖手工超参 `γdepth`；过强可能抑制近景细节 |
| 两者组合 | 同时解决“该长不长”和“长错地方” | 最稳定的质量提升 | 相比原版 3DGS 仍有一定训练/内存开销 |

### 为什么不是简单降低阈值
这是论文里一个很重要的洞察。

如果只把原始 3DGS 的增长阈值 `τpos` 调低：
- 的确会长出更多点；
- 但这些点往往继续长在原本就较密的区域；
- 稀疏初始化区的大 Gaussian 仍然因为统计方式不合理而拿不到足够票数。

所以 Pixel-GS 的核心不是“让更多点生长”，而是**让点长在正确的位置**。

---

## Part III：证据与局限

### 关键证据

- **比较信号：跨数据集平均结果都更好。**  
  在 Mip-NeRF 360 上，Pixel-GS 相比 retrained 3DGS* 从 `27.71 / 0.826 / 0.202` 提升到 `27.88 / 0.834 / 0.176`（PSNR / SSIM / LPIPS）。  
  在 Tanks & Temples 上，从 `24.19 / 0.844 / 0.194` 提升到 `24.38 / 0.850 / 0.178`。  
  最显著的改进集中在 **LPIPS**，说明它更像是在修复人眼感知上更明显的模糊和结构伪影。

- **消融信号：单独做像素加权不够，必须配合深度缩放。**  
  在 Mip-NeRF 360 上，pixel-aware gradient 单独就能提升结果；  
  但在 Tanks & Temples 上，它单独使用会显著掉点，平均 PSNR 甚至从 24.23 降到 21.80。  
  当加入 scaled gradient field 后，完整模型恢复并达到最好结果。  
  这说明作者抓到的不是一个“总是单调增益”的 trick，而是一个**有明确副作用、也有对应修复项的机制改动**。

- **对照信号：不是“多长点”就行。**  
  把原始 3DGS 的 `τpos` 调低到接近 Pixel-GS 的点数规模后，质量仍落后于 Pixel-GS，同时更费内存。  
  例如在 Mip-NeRF 360 上，低阈值 3DGS* 的 LPIPS 为 0.181、内存 1.4GB；Pixel-GS 为 0.176、内存 1.2GB。  
  这证明提升来自**更合理的点分布**，不是简单粗暴的密度膨胀。

- **鲁棒性信号：对差初始化点云更稳。**  
  作者随机丢弃 SfM 初始化点。随着 drop rate 上升，Pixel-GS 在 PSNR / SSIM / LPIPS 上始终优于 3DGS。  
  更强的一点是：论文明确报告 **即使丢弃 99% 初始化点，Pixel-GS 的 LPIPS 仍优于使用完整 SfM 点初始化的 3DGS**。  
  这几乎直接对应了它要解决的真瓶颈：**初始化稀疏时 densification 信号失真**。

### 1-2 个最值得记住的指标
1. **Mip-NeRF 360：LPIPS 0.202 → 0.176**，说明感知质量改善明显。  
2. **Tanks & Temples：完整模型 LPIPS 0.178 vs 3DGS* 0.194**，且消融证明深度缩放对 suppress floater 是必要的。

### 局限性
- **Fails when**: 近相机投影特别大的场景里，如果深度缩放不启用或超参不合适，pixel-aware growth 会放大 floater；若相机位姿或 SfM 几何本身严重错误，单靠 densification 规则无法纠正。
- **Assumes**: 静态场景、可用的相机参数与 SfM 初始化点云、3DGS 风格的 split/clone 训练流程；`γdepth` 是人工固定超参；为了换取更合理的点分布，仍需承担额外 Gaussian、内存和训练时间开销。
- **Not designed for**: 动态场景建模、替代 COLMAP/SfM、反射/高光专门建模、抗锯齿或压缩优化本身。

### 可复用组件
- **像素计数加权的增密统计**：适合接到其他 3DGS densification 策略中，且不需要额外深度/法线监督。
- **深度感知的梯度场缩放**：可作为独立 anti-floater 插件，和许多 3DGS 变体正交。
- **工程兼容性较好**：作者明确说只需对原始 3DGS 做小改动，说明复用门槛不高。

**一句话评价**：  
Pixel-GS 的价值不在于发明了新的表示，而在于它精准修补了 3DGS 最脆弱的一个环节——**densification 的统计对象错了**；一旦把“视角均值”改成“像素支持加权”，再把近处梯度做平衡，稀疏初始化区的生长逻辑就终于对了。

## Local PDF reference
![[paperPDFs/Densification_Rendering/Lecture_Notes_in_Computer_Science_2025/2025_Pixel_GS_Density_Control_with_Pixel_aware_Gradient_for_3D_Gaussian_Splatting.pdf]]