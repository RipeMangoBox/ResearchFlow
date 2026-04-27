---
title: "GART: Gaussian Articulated Template Models"
venue: arXiv
year: 2023
tags:
  - 3D_Gaussian_Splatting
  - task/monocular-3d-reconstruction
  - task/novel-view-synthesis
  - gaussian-splatting
  - learnable-skinning
  - latent-bones
  - dataset/ZJU-MoCap
  - dataset/People-Snapshot
  - dataset/UBC-Fashion
  - repr/SMPL
  - opensource/full
core_operator: 在模板骨架先验上，用canonical空间的3D高斯混合显式逼近辐射场，并通过可学习前向蒙皮与潜骨建模复杂非刚性变形。
primary_logic: |
  单目视频 + 模板姿态估计 → 在canonical空间初始化并优化3D Gaussian混合，结合可学习前向蒙皮与潜骨驱动姿态/衣物变形，再用Gaussian Splatting与平滑正则进行图像监督优化 → 输出可重定姿、可实时渲染的 articulated avatar/animal model
claims:
  - "在 ZJU-MoCap 上，GART 以约 2.5 分钟训练达到 32.22 PSNR / 0.977 SSIM / 29.21 LPIPS，优于 Instant-NVR 等基线；即使训练约 30 秒也仍有 31.76 PSNR [evidence: comparison]"
  - "在 People-Snapshot 上，GART 以约 30 秒训练获得与 InstantAvatar 可比的重建质量，同时在 540×540 分辨率下推理速度超过 150 FPS，而 InstantAvatar 约为 15 FPS [evidence: comparison]"
  - "在 UBC-Fashion 上，移除 learnable skinning 或 latent bones 会使 PSNR 从 25.65 降至 23.76 或 25.00，说明两者都对宽松服饰的动态建模有效 [evidence: ablation]"
related_work_position:
  extends: "3D Gaussian Splatting (Kerbl et al. 2023)"
  competes_with: "InstantAvatar (Jiang et al. 2023); Instant-NVR (Geng et al. 2023)"
  complementary_to: "ReFit (Wang and Daniilidis 2023); BITE (Rüegg et al. 2023)"
evidence_strength: strong
pdf_ref: paperPDFs/Avatar/arXiv_2023/2023_GART_Gaussian_Articulated_Template_Models.pdf
category: 3D_Gaussian_Splatting
---

# GART: Gaussian Articulated Template Models

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2311.16099) · [Project](https://www.cis.upenn.edu/~leijh/projects/gart/)
> - **Summary**: 这篇工作把单目人体/动物重建中的隐式 NeRF 表示，替换为可显式蒙皮的 3D Gaussian 模板，从而同时拿到更快训练、更快渲染，以及对宽松衣物和姿态噪声更稳的重建效果。
> - **Key Performance**: ZJU-MoCap 上 PSNR 32.22；People-Snapshot 上 540×540 渲染速度 >150 FPS，约为 InstantAvatar 的 10×。

> [!info] **Agent Summary**
> - **task_path**: 单目RGB视频 + 模板姿态估计 -> 可重定姿的显式3D articulated avatar / novel-view renderings
> - **bottleneck**: 隐式人体辐射场渲染慢、变形常依赖复杂反向蒙皮且对姿态误差敏感；模板网格又难表达衣物/毛发等细节
> - **mechanism_delta**: 用 canonical 3D Gaussian mixture 显式替代隐式辐射场，并以可学习前向蒙皮 + 潜骨而非反向蒙皮驱动复杂变形
> - **evidence_signal**: 多数据集对比显示速度/质量同时提升，且 UBC-Fashion 消融直接验证 latent bones 与 learnable skinning 的贡献
> - **reusable_ops**: [canonical Gaussian mixture, latent-bone forward skinning]
> - **failure_modes**: [严重模板姿态误差会传导到形变结果, 无模板或遮挡重的物种/场景难稳定重建]
> - **open_questions**: [如何降低对外部pose estimator的依赖, 如何从视频集合学习类别级articulated Gaussian prior]

## Part I：问题与挑战

这篇论文解决的是：**如何从单目视频快速重建一个可重定姿、可高质量渲染的非刚性 articulated 主体**，包括人和动物。

### 真正的难点是什么
不是“能不能拟合一段视频”，而是同时满足三件事：

1. **表示要足够细**：能表达衣物、毛发、纹理和局部几何；
2. **形变要可控**：能跟随骨架运动，还要容纳裙摆、宽松衣物这类超出模板骨架的运动；
3. **优化和渲染要快**：单目视频场景里，本来观测就稀疏，如果方法又慢又脆弱，就很难落地。

### 先前方法的瓶颈
- **模板网格类方法**：有骨架先验、好驱动，但拓扑固定，细节不够。
- **NeRF/隐式场方法**：细节强，但渲染慢；动态人体常要做 backward skinning / root finding，工程复杂，而且对姿态估计误差敏感。
- **显式点/mesh 方法**：快，但要么表达力不足，要么对拓扑/多视角条件依赖重。

### 输入 / 输出接口
- **输入**：单目 RGB 视频 + 每帧模板姿态估计（人用 SMPL，狗用 D-SMAL/BITE）。
- **输出**：一个 canonical 空间下的显式 articulated 表示，可在新视角、新姿态下渲染。

### 边界条件
这不是从零学习 articulation，也不是开放类别 4D 重建。它明确依赖：
- 类别级模板先验；
- 可用的 pose estimator；
- 单主体、相对可跟踪的视频序列。

## Part II：方法与洞察

GART 的核心设计是：**把“隐式连续辐射场 + 复杂变形求解”改成“显式高斯集合 + 简单前向蒙皮”**。

### 方法主线
#### 1. 用 canonical 3D Gaussian Mixture 表示形状与外观
作者不再在 canonical 空间中用 MLP 隐式表示 radiance field，而是直接用一组 3D 高斯来近似它。  
每个高斯显式存：
- 位置
- 朝向
- 尺度
- 不透明度
- 颜色/方向相关外观

这样做的结果是：
- 表达仍然足够细；
- 不受固定 mesh topology 限制；
- 可以直接走 Gaussian Splatting 的高效渲染路径。

#### 2. 用可学习前向蒙皮驱动高斯
每个高斯绑定模板骨架的 skinning weight，但不是直接照搬模板，而是在模板权重上加一个**learnable correction**。  
这一步解决的是：**实例的真实形变通常偏离模板默认蒙皮**。

相比很多隐式人体方法常用的 backward skinning，前向蒙皮更直接：
- 不用做 root finding；
- 推理和训练都更高效；
- 对姿态估计噪声更稳，因为优化更像“显式形变调整”。

#### 3. 用 latent bones 补足模板骨架表达力
SMPL 这种预定义骨架对人体主体运动足够，但对**长裙、宽松衣服、摆动附件**不够。  
GART 在真实骨架之外，再引入一组**latent bones**：
- 其变换可由 pose-conditioned MLP 或每帧表驱动；
- 每个高斯再学习对这些 latent bones 的附加蒙皮权重。

这一步的意义是：把“模板骨架解释不了的残差形变”变成一个**低维、共享的额外运动子空间**，而不是让外观场自己硬扛。

#### 4. 用平滑先验弥补显式高斯缺少隐式场天然平滑性的问题
显式高斯的优势是快，但缺点是：单目监督稀疏时，未观测区域容易长出噪声或漂点。  
作者用了两类正则：
- **voxel-distilled skinning**：让 skinning field 空间上更平滑；
- **KNN attribute regularization**：约束近邻高斯的旋转/尺度/颜色/蒙皮权重变化不要过于跳跃。

这一步本质上是在补上 NeRF 里由 MLP 自带的“局部平滑偏置”。

### 核心直觉

**变化是什么**：  
把 articulation-aware avatar 的表示，从“隐式辐射场 + 复杂变形求逆”改成“显式高斯辐射近似 + 前向蒙皮 + 潜骨残差形变”。

**改变了什么瓶颈**：  
- 计算约束：渲染从大量场查询，变成高斯 splatting；
- 形变约束：从 backward skinning 的求根难题，变成每个高斯的显式前向变换；
- 信息瓶颈：把宽松衣物等额外自由度，从外观场中剥离给 latent bones；
- 先验缺口：用 voxel/KNN 正则补足显式表示的局部平滑性不足。

**带来什么能力变化**：  
- 训练从小时级压到秒/分钟级；
- 推理到 150+ FPS；
- 对裙摆、狗耳朵/尾巴这类非模板细节更能跟住；
- 对姿态误差和单目未观测区域比纯隐式方法更稳。

### 战略取舍

| 设计选择 | 缓解的瓶颈 | 收益 | 代价 |
| --- | --- | --- | --- |
| 3D Gaussian 替代 NeRF | 渲染慢、查询重 | 显式、高速、可解释 | 未观测区天然平滑性更弱 |
| 可学习前向蒙皮 | 模板蒙皮与实例不匹配 | 动画简单、避免 root finding | 仍依赖模板骨架质量 |
| latent bones | 宽松衣物/附属物无法由固定骨架解释 | 能表达额外自由度 | 若约束不足会漂移或过拟合 |
| voxel + KNN 平滑正则 | 单目监督稀疏导致空域伪影 | 稳定优化、减少噪声 | 可能带来过平滑 |

## Part III：证据与局限

### 关键证据
1. **速度-质量同时成立，而不是二选一**  
   在 ZJU-MoCap 上，GART 约 2.5 分钟训练即可达到 32.22 PSNR，优于 Instant-NVR；即使只训约 30 秒，仍接近强基线。  
   这说明它不是“牺牲效果换速度”，而是表示与变形建模一起改了瓶颈。

2. **推理效率有明显代际差**  
   People-Snapshot 上，GART 在 540×540 分辨率下可达 **150+ FPS**，而文中对比的 InstantAvatar 约 **15 FPS**。  
   对 avatar 系统来说，这个信号很关键：它不只是论文指标更好，而是更接近交互式渲染。

3. **真正的能力跳跃出现在宽松服饰场景**  
   UBC-Fashion 上，GART 明显优于 InstantAvatar。这里的增益更能证明方法核心，因为这类场景正是固定骨架 + 隐式外观最容易失效的地方。  
   论文把优势归因到两点：前向蒙皮更稳，以及 latent bones 给了裙摆额外运动自由度。

4. **消融支持因果链条**  
   去掉 learnable skinning 或 latent bones，指标都会下降；去掉 KNN/voxel 平滑，会出现背面或侧面噪声伪影。  
   说明论文的提升不是单靠 3DGS 渲染器，而是**表示、形变、正则**三者共同作用。

5. **跨类别泛化到狗**  
   在 in-the-wild dog videos 上，GART 相比改造版 InstantAvatar 也更稳，说明该框架并不局限于人类主体，只要存在可用模板先验，就有迁移性。

### 1-2 个最关键指标
- **ZJU-MoCap**: PSNR **32.22**
- **People-Snapshot**: inference **>150 FPS @ 540×540**

### 局限性
- **Fails when**: 模板姿态估计误差很大、遮挡严重、可见视角过少，或目标类别缺乏合适模板时，GART 仍可能出现错误蒙皮、ghost artifact 和未观测区域伪影。
- **Assumes**: 需要每帧模板姿态估计、类别级模板先验（如 SMPL / D-SMAL）、单主体视频，以及 GPU 上的 Gaussian Splatting 渲染与优化；狗实验还依赖筛选出遮挡较少、姿态较可靠的片段。
- **Not designed for**: 开放类别无模板动物、多人强交互场景、强拓扑变化对象，或直接从大规模视频集合学习类别级 articulated prior。

### 可复用组件
- **canonical 高斯模板**：适合把静态 3DGS 扩展到可动画主体；
- **learnable forward skinning**：对任何模板驱动的显式表示都可借鉴；
- **latent bones**：是处理衣物/附件残差形变的通用模块；
- **voxel-distilled skinning + KNN 正则**：适合单目、稀疏监督下稳定显式点/高斯表示；
- **Text-to-GART 扩展**：说明该表示还能接上 SDS / diffusion 做生成，而不只是重建。

![[paperPDFs/Avatar/arXiv_2023/2023_GART_Gaussian_Articulated_Template_Models.pdf]]