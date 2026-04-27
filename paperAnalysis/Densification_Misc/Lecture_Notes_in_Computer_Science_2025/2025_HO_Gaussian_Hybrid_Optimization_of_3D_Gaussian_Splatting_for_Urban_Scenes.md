---
title: "HO-Gaussian: Hybrid Optimization of 3D Gaussian Splatting for Urban Scenes"
venue: arXiv
year: 2024
tags:
  - 3D_Gaussian_Splatting
  - task/novel-view-synthesis
  - grid-based-volume
  - directional-encoding
  - neural-warping
  - dataset/Waymo
  - dataset/Argoverse
  - opensource/no
core_operator: 用网格体积为3DGS周期性补充高密度候选点，并通过方向编码与神经warping联合优化多相机城市街景渲染
primary_logic: |
  多相机RGB图像与已知位姿 → 先用网格体积学习密度/颜色并在训练中为3DGS补点，再用高斯显式渲染、方向编码和虚拟视角warping做联合优化 → 无需SfM/LiDAR点初始化的实时城市新视角合成
claims:
  - "Claim 1: 在不使用 SfM 或 LiDAR 点初始化的条件下，HO-Gaussian 在 Waymo 上达到 28.03/0.8364/0.3282、在 Argoverse 上达到 30.98/0.9043/0.2287（PSNR/SSIM/LPIPS），并超过 RGB-only 的 NGP、MERF、Block-NeRF 与 3DGS 基线 [evidence: comparison]"
  - "Claim 2: 在 Argoverse 消融中，加入 Point Densification 后，性能从 28.42/0.8758/0.3026 提升到 30.58/0.8954/0.2372，说明补点是主要质量增益来源 [evidence: ablation]"
  - "Claim 3: 半分辨率设置下，HO-Gaussian 以 123MB 模型大小实现 71 FPS 实时渲染，而半分辨率 SfM 初始化 3DGS 为 557MB，显示出显著的存储压缩同时保持实时性 [evidence: comparison]"
related_work_position:
  extends: "3D Gaussian Splatting (Kerbl et al. 2023)"
  competes_with: "3D Gaussian Splatting (Kerbl et al. 2023); LocalRF (Meuleman et al. 2023)"
  complementary_to: "SUDS (Turki et al. 2023)"
evidence_strength: moderate
pdf_ref: paperPDFs/Densification_Misc/Lecture_Notes_in_Computer_Science_2025/2025_HO_Gaussian_Hybrid_Optimization_of_3D_Gaussian_Splatting_for_Urban_Scenes.pdf
category: 3D_Gaussian_Splatting
---

# HO-Gaussian: Hybrid Optimization of 3D Gaussian Splatting for Urban Scenes

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2403.20032)
> - **Summary**: 该文把“体积场补点 + 高斯显式渲染 + 虚拟视角一致性”整合进同一训练流程，使 3DGS 在没有 SfM/LiDAR 点云初始化时也能覆盖城市街景中的天空、远处和低纹理区域。
> - **Key Performance**: Argoverse 上达到 30.98 PSNR / 0.9043 SSIM / 0.2287 LPIPS；半分辨率模型大小 123MB、渲染速度 71 FPS。

> [!info] **Agent Summary**
> - **task_path**: 多相机RGB图像+相机位姿 -> 城市街景新视角图像
> - **bottleneck**: 3DGS 的增密只能围绕初始点云局部展开，城市场景里 SfM 漏掉的天空/远处/低纹理区域长期无几何支撑；多相机重叠不足又会破坏视角相关颜色一致性
> - **mechanism_delta**: 用网格体积先在全局预测密度并周期性向高斯管线注入候选点，再用共享方向编码和虚拟视角 warping 共同约束外观与几何
> - **evidence_signal**: 双数据集对比显示其在 RGB-only 设定下优于 NGP/MERF/Block-NeRF/3DGS，且 Argoverse 消融中 Point Densification 带来最大增益
> - **reusable_ops**: [体积引导补点, 方向编码替代高阶SH]
> - **failure_modes**: [动态物体或遮挡变化导致补点不稳, 相机重叠极低时跨视角颜色仍可能不一致]
> - **open_questions**: [如何扩展到显式动态街景建模, 如何进一步降低体积分支带来的训练成本]

## Part I：问题与挑战

这篇论文做的是**城市街景的新视角合成**：输入是多相机 RGB 图像和相机位姿，输出是可实时渲染的新视角图像。它的目标很明确：**在不依赖 SfM/LiDAR 点云初始化的前提下，把 3DGS 用到自动驾驶常见的长距离、多相机、低纹理城市场景中。**

真正的难点不是“3DGS 不够快”，而是**3DGS 的几何优化搜索空间被初始点云卡住了**：

- **初始化瓶颈**：原始 3DGS 的 clone/split 主要在已有点附近工作。若 SfM 在天空、远处、白墙、道路等区域根本没恢复出点，后续高斯优化几乎无从开始。
- **无界城市场景的表示瓶颈**：城市场景尺度跨度极大，有限数量的高斯容易集中在近处，导致远处和低纹理区域表示不足。
- **多相机一致性瓶颈**：多相机系统重叠区域有限，导致同一物体的视角相关颜色难以统一，容易过拟合到观测过的特定 camera 视角。
- **存储瓶颈**：若给大量高斯都存高阶 SH，磁盘和显存开销会迅速膨胀，这在大范围街景里尤其明显。

**为什么现在要解决这个问题？**  
因为 3DGS 已经证明了显式高斯渲染在速度上比 NeRF 更适合实时应用。如果能补上“城市级初始化缺洞”和“多相机一致性”这两个短板，3DGS 就更接近自动驾驶仿真、交互式重建和大规模街景回放的实际需求。

**边界条件**：

- 需要已知且较准确的相机内外参；
- 更偏向静态或准静态场景，而非显式动态建模；
- 论文实验只选取了较长序列场景进行评估，说明方法默认有较充足的多帧监督。

## Part II：方法与洞察

作者的思路可以概括成一句话：**让体积场负责“找到哪里应该有高斯”，让高斯管线负责“把它渲染得快且清晰”。**

### 方法主线

1. **Hybrid Volume + Point Densification**
   - 先训练一个 grid-based volume，从图像中学习密度与颜色。
   - 体积场 warm-up 后，在训练过程中周期性地从高密度区域提出候选 3D 位置，补进 Gaussian pipeline。
   - 这些新点不是最终几何，只是“把优化带进原本没有初始点的区域”；之后仍由 3DGS 的梯度优化、clone/split 和低透明度裁剪继续细化。

2. **Gaussian Positional Encoding**
   - 借鉴 unbounded scene contraction，把无界城市场景压到有限球域。
   - 它解决的是“高斯预算怎么分配”的问题，而不是直接替代几何学习本身。

3. **Gaussian Directional Encoding**
   - 不再依赖每个高斯都携带高阶 SH。
   - 改为让一个小型 grid+MLP 共享地生成 view-dependent color，高斯自身主要保留几何与基础外观。
   - 本质上是把方向外观从“显式逐点存储”改成“共享神经函数生成”。

4. **Neural Warping**
   - 对已有相机位姿做小扰动，利用体积场生成虚拟视角。
   - 这些 warped views 作为额外监督，让高斯管线学习更稳定的跨相机外观和几何一致性，缓解真实相机重叠不足的问题。

### 核心直觉

这篇论文真正拧动的因果旋钮是：

**把 3DGS 的几何搜索空间从“已有点附近的局部修补”改成“先由体积场给出全局候选，再由显式高斯做局部收敛”。**

- **what changed**：从纯点初始化的 3DGS，变成体积场与高斯共同训练的 hybrid pipeline。
- **which bottleneck changed**：
  - 几何分布从“被 SfM 稀疏点云截断”变成“由图像监督驱动的稠密候选密度场”；
  - 方向外观从“每高斯存高阶 SH”变成“共享网络生成视角相关颜色”；
  - 视角监督从“只看真实相机”变成“真实相机 + 虚拟扰动视角”。
- **what capability changed**：因此模型能在天空、远处、低纹理区域长出高斯，减少跨相机颜色不一致，并在更小模型体积下保持实时渲染。

为什么这套设计有效？因为**体积场更适合先从纯图像监督中恢复粗几何支持**，而**显式高斯更适合把粗支持收敛成可快速渲染的最终表示**。前者解决“哪里该有东西”，后者解决“如何高效画出来”。

### 战略取舍

| 设计 | 缓解的瓶颈 | 收益 | 代价/风险 |
|---|---|---|---|
| 体积引导补点 | 初始点云缺洞，3DGS 无法凭空扩展 | 无需 SfM/LiDAR 也能补天空、远处、低纹理区域 | 训练更复杂；体积生成的点初始精度有限 |
| 位置编码/场景压缩 | 无界场景下高斯预算分配失衡 | 近景与远景都能被有限高斯覆盖 | 远处细节被压缩，精度依赖映射 |
| 方向编码替代高阶 SH | 大规模场景模型体积爆炸 | 显著降低存储开销 | 引入额外神经颜色分支，略增训练/推理负担 |
| 神经 warping | 多相机重叠不足导致外观不一致 | 提高跨相机几何与颜色一致性 | 虚拟视角质量受体积场本身误差影响 |

## Part III：证据与局限

### 关键证据

- **比较信号（双数据集）**：在 RGB-only 设定下，HO-Gaussian 在 Waymo 达到 28.03 PSNR / 0.8364 SSIM / 0.3282 LPIPS，在 Argoverse 达到 30.98 / 0.9043 / 0.2287，整体优于 NGP、MERF、Block-NeRF 和 RGB-only 3DGS。最重要的结论不是“又涨了几点”，而是**它把 3DGS 从必须依赖点云初始化，推进到了可以直接从图像起步**。
- **更强监督对比信号**：半分辨率下，RGB-only 的 HO-Gaussian* 仍优于 RGB+LiDAR 的 EmerNeRF*（Waymo 28.97 vs 28.62；Argoverse 31.52 vs 30.14），说明它并不只是“省掉外部几何”，质量本身也有竞争力。
- **消融信号**：Point Densification 是最大增益来源。Argoverse 上从加入 position encoding 的 28.42/0.8758/0.3026，提升到加入 densification 的 30.58/0.8954/0.2372；再加 neural warping 后达到 30.98/0.9043/0.2287。也就是说，**能力跃迁主要来自“能往缺失区域补高斯”**，而不是仅靠编码技巧。
- **效率信号**：半分辨率下 HO-Gaussian 为 123MB、71 FPS；相比 3DGS* 的 557MB、87 FPS，HO-Gaussian 明显更省存储，但牺牲了一些速度和训练时间（76 分钟 vs 31 分钟）。

一个值得注意的细节是：Argoverse 上 LocalRF 的 PSNR 略高于 HO-Gaussian，但 HO-Gaussian 的 LPIPS 更低。这意味着它的优势更偏向**感知质量与整体纹理观感**，而不一定在所有像素级误差上都绝对最优。

### 局限性

- Fails when: 动态物体占比高、遮挡变化剧烈、或真正无观测的区域过大时，体积分支补出的点可能不准，出现模糊、拖影或跨相机不一致。
- Assumes: 已知且准确的多相机位姿；足够长的训练序列；图像监督足以先把体积场 warm-up；有不低的算力预算（文中使用 V100 32GB，训练 30K iterations）。
- Not designed for: 显式动态场景分解、在线 SLAM/增量建图、超大城市级分块流式渲染，以及需要语义可控编辑的任务。

**资源与复现依赖**：

- 论文正文未给出代码/项目链接，复现仍依赖实现 hybrid schedule、补点阈值和 warping 细节。
- 与部分 baseline 的对比存在输入模态或分辨率差异，因此更适合把结果解读为“该机制是否有效解除瓶颈”，而不是机械地视为完全同条件的绝对 SOTA。

**可复用组件**：

- `体积场 -> 显式高斯` 的周期性补点机制；
- 用共享神经方向编码替代 per-Gaussian 高阶 SH；
- 用虚拟视角 warping 做多相机一致性正则。

![[paperPDFs/Densification_Misc/Lecture_Notes_in_Computer_Science_2025/2025_HO_Gaussian_Hybrid_Optimization_of_3D_Gaussian_Splatting_for_Urban_Scenes.pdf]]