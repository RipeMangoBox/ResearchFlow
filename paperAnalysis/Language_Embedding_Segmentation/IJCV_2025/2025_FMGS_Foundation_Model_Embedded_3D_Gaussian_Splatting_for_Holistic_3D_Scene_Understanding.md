---
title: "FMGS: Foundation Model Embedded 3D Gaussian Splatting for Holistic 3D Scene Understanding"
venue: arXiv
year: 2024
tags:
  - 3D_Gaussian_Splatting
  - task/open-vocabulary-object-detection
  - task/open-vocabulary-segmentation
  - multi-resolution-hash-encoding
  - feature-distillation
  - pixel-alignment
  - dataset/LERF
  - dataset/3D-OVS
  - opensource/promised
core_operator: "用3D Gaussian承载几何与外观，用多分辨率哈希场在高斯中心生成共享语义，再以多尺度CLIP蒸馏和DINO像素对齐学习跨视角一致的开放词汇3D特征。"
primary_logic: |
  标定多视图RGB与相机位姿 → 先训练并冻结3D Gaussian几何/外观，再在高斯中心查询MHE并经CLIP/DINO解码头渲染语义特征 → 用多尺度混合CLIP监督与DINO边界约束优化3D语义场，输出可用于文本检索的relevancy map与开放词汇定位/分割结果
claims:
  - "FMGS在LERF目标定位基准上取得93.2%平均准确率，超过LERF的83.0%约10.2个百分点 [evidence: comparison]"
  - "FMGS在480×270分辨率下联合渲染RGB、CLIP和DINO特征达到103.4 FPS，而LERF为0.1214 FPS，推理约快851× [evidence: comparison]"
  - "移除hybrid CLIP监督会使目标定位准确率从93.2%降至30.8%，移除MHE降至84.4%，表明多尺度监督与共享哈希语义场是关键因素 [evidence: ablation]"
related_work_position:
  extends: "LERF (Kerr et al. 2023)"
  competes_with: "LERF (Kerr et al. 2023); 3D-OVS (Liu et al. 2023)"
  complementary_to: "SAM (Kirillov et al. 2023)"
evidence_strength: moderate
pdf_ref: paperPDFs/Language_Embedding_Segmentation/IJCV_2025/2025_FMGS_Foundation_Model_Embedded_3D_Gaussian_Splatting_for_Holistic_3D_Scene_Understanding.pdf
category: 3D_Gaussian_Splatting
---

# FMGS: Foundation Model Embedded 3D Gaussian Splatting for Holistic 3D Scene Understanding

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2401.01970)
> - **Summary**: 这篇工作把 CLIP/DINO 语义蒸馏进 3D Gaussian Splatting，并用多分辨率哈希编码与像素对齐约束，得到一个既快又跨视角一致的开放词汇 3D 场景表示。
> - **Key Performance**: 在 LERF 开放词汇目标定位上达到 93.2% 平均准确率，较 LERF 提升 10.2 个百分点；推理速度 103.4 FPS，约快 851×。

> [!info] **Agent Summary**
> - **task_path**: 标定多视图 RGB + 相机位姿 -> 开放词汇 3D 语义场 / 2D relevancy map -> 目标定位与语义分割
> - **bottleneck**: room-scale 场景里若给每个 Gaussian 直接挂高维语言特征会造成巨大内存与训练负担，而 2D CLIP 又缺少像素级边界和多视图一致性
> - **mechanism_delta**: 把 per-Gaussian 语义向量改成由共享 MHE 场在高斯中心查询，再用 hybrid multi-scale CLIP + DINO dotsim 把全局语义压成局部对齐、跨视角稳定的 3D 特征
> - **evidence_signal**: LERF 数据集目标定位准确率比 LERF 高 10.2 个百分点，且推理快约 851×
> - **reusable_ops**: [gaussian-center-to-hash-query, hybrid-multiscale-clip-supervision]
> - **failure_modes**: [相机标定或位姿质量差时语义场会漂移, 近景且上下文不足时背景语义容易混淆]
> - **open_questions**: [能否直接蒸馏SAM类局部模型到3D以增强分割, 能否扩展到动态场景或更大规模室外环境]

## Part I：问题与挑战

这篇论文要解决的不是“把 2D foundation model 特征塞进 3D”这么简单，而是一个更硬的系统问题：

**如何在 room-scale 3D 重建里，同时保留**
1. **高质量几何/外观渲染**，
2. **开放词汇语义查询能力**，
3. **跨视角语义一致性**，
4. **可实时推理的效率**。

### 真正瓶颈是什么？

**瓶颈 1：语义表示太贵。**  
Gaussian Splatting 往往要用到数百万个高斯。如果给每个 Gaussian 都附一个高维 CLIP 向量，显存和训练开销都会爆炸。

**瓶颈 2：CLIP 天生不是像素对齐特征。**  
CLIP 更擅长图像级语义，边界模糊、邻域区分弱。直接监督 3D 特征场，会让 relevancy map 不够尖锐，目标边界不清。

**瓶颈 3：2D foundation model 缺少天然多视图一致性。**  
同一个 3D 物体从不同视角看到的 2D CLIP 特征并不会自动一致；如果不在 3D 表示层面对齐，查询结果会飘。

### 为什么现在值得做？

因为两个条件同时成熟了：

- **GS** 让高质量、快速的 3D 渲染成为现实；
- **CLIP / DINO** 让开放词汇语义和局部视觉边界都可被利用。

FMGS 的切入点，就是把这两条线真正拼起来：  
**GS 负责“几何与可渲染性”，foundation model 负责“开放语义”，中间用一个高效 3D 特征场打通。**

### 输入 / 输出 / 边界条件

- **输入**：多视图 RGB 图像 + 相机位姿（由 COLMAP 等 SfM 获取）
- **输出**：可渲染的 3D 语义场；给定文本 query 后输出 2D relevancy map，可用于目标定位或分割
- **训练监督**：无 3D 语义标注；主要依赖 2D CLIP/DINO 特征蒸馏
- **适用边界**：静态、可标定、可重建场景

---

## Part II：方法与洞察

FMGS 的核心设计是：

> **用 3D Gaussian 表示几何与外观，用 MHE 表示语义。**

这不是简单拼接，而是一个有明确分工的结构：

1. **先训练普通 GS**
   - 从多视图图像恢复场景几何和外观；
   - 训练完后，几何属性和外观基本冻结。

2. **再训练语义特征场**
   - 不给每个 Gaussian 存一个独立高维语义向量；
   - 而是在 Gaussian 的 3D 中心位置查询一个 **多分辨率哈希编码场（MHE）**；
   - 再经过小型 MLP 输出 CLIP / DINO 特征。

3. **把 2D foundation model 特征蒸馏回 3D**
   - 用 GS 光栅化渲染出预测的 CLIP / DINO 特征图；
   - 用图像上的 foundation model 特征作为监督。

4. **让语义更稳、更准**
   - CLIP 用多尺度 crop 得到 feature pyramid，再平均成一个 **hybrid CLIP feature map**；
   - DINO 提供更强的局部边界信号；
   - 用一个基于邻域点积相似度的 **pixel-alignment / dotsim loss**，逼迫 CLIP 渲染特征沿着 DINO 的边界结构对齐。

5. **查询时单次渲染即可**
   - 渲染出 CLIP feature map；
   - 把文本 query 编成 CLIP embedding；
   - 计算每个像素的相关性分数，输出 relevancy map。

### 核心直觉

FMGS 真正改变的，不是“用了 GS”本身，而是：

#### 1) 把语义从“每个高斯一个大向量”改成“由连续 3D 哈希场共享生成”
- **改变前**：语义存储与 Gaussian 数量线性增长；
- **改变后**：语义由共享 MHE 场生成，参数量不再直接跟高斯数爆炸式绑定。

**结果**：  
内存压力下降，room-scale 训练变可行，而且局部空间中的语义更连续。

#### 2) 把“推理时多尺度搜索”改成“训练时多尺度融合”
LERF 的一个核心代价是：查询时需要搜索不同尺度，找最相关的特征。  
FMGS 则把多尺度信息在训练阶段融成一个 hybrid CLIP 目标。

**改变的约束**：
- 从 **query-time 搜索** 变成 **train-time 蒸馏**
- 从 **每条 ray 额外试很多尺度** 变成 **单次渲染即可查询**

**结果**：  
速度大幅提升，且语义不再那么依赖后处理式尺度搜索。

#### 3) 用 DINO 的“边界感”修补 CLIP 的“全局感”
CLIP 有语义，但边界糊；DINO 更局部、更像素对齐。  
FMGS 没有直接抛弃 CLIP，而是让 CLIP 保持开放词汇语义，让 DINO 提供邻域结构约束。

**改变的信息瓶颈**：
- 从“只有全局语义监督”
- 变成“全局语义 + 局部边界结构”

**结果**：  
relevancy map 更聚焦，物体与背景更容易分开。

### 战略性 trade-off

| 设计选择 | 解决的瓶颈 | 带来的能力 | 代价 / 权衡 |
|---|---|---|---|
| GS 负责几何外观，MHE 负责语义 | per-Gaussian 语义太重 | 高效、可扩展的 3D 语义场 | 语义表达受 MHE 分辨率与 MLP 容量限制 |
| hybrid multi-scale CLIP | 单尺度 CLIP 上下文不足 | 兼顾近景细节与远景语义 | 牺牲显式尺度可控性 |
| DINO regularization + dotsim | CLIP 边界模糊 | 更强目标边界与局部可分性 | 训练更复杂，依赖额外 foundation model |
| 冻结几何后再学语义 | 减少耦合优化难度 | 训练稳定，语义学习更聚焦 | 几何误差无法被语义阶段纠正 |

### 方法为什么会比 prior work 跳得更远？

因为它同时解决了 prior work 的两个常见矛盾：

- **语义强 vs 渲染快**
- **开放词汇 vs 像素对齐**

LERF 更像是在 NeRF 框架里“接入语言”；  
FMGS 则是在 GS 的实时渲染能力上，重新设计了语义嵌入方式和监督方式。

这就是为什么它不只是“更快的 LERF”，而是把：
- 语义表示方式，
- 多尺度处理方式，
- 像素对齐方式，

一起换了一遍。

---

## Part III：证据与局限

### 关键证据

- **比较信号：开放词汇目标定位明显更强**  
  在 LERF 数据集上，FMGS 的平均准确率为 **93.2%**，相比 LERF 的 **83.0%** 提升 **10.2 个百分点**。这说明它不仅更快，而且相关性图更“准”。

- **系统信号：实时性是实打实的能力跃迁**  
  FMGS 在 480×270 分辨率下联合渲染 RGB、CLIP、DINO 特征可达 **103.4 FPS**，而 LERF 只有 **0.1214 FPS**。  
  这不是小幅加速，而是把开放词汇 3D 查询从“离线分析”推到“接近实时交互”。

- **泛化信号：分割虽非主任务，但 3D 特征质量确实更高**  
  在 3D-OVS 的 6 个场景里，FMGS 的 mIoU / mAP 都显著优于 LERF。说明它学到的不是只对检测有利的启发式，而是整体更干净的语义特征场。

- **消融信号：最关键的是 hybrid CLIP，其次是 MHE 与像素对齐**
  - 去掉 **hybrid CLIP**：93.2% → 30.8%
  - 去掉 **MHE**：93.2% → 84.4%
  - 去掉 **dotsim**：93.2% → 90.4%

  结论很明确：  
  **多尺度语义监督** 是最主要的增益来源；  
  **共享哈希语义场** 和 **边界对齐损失** 进一步把图做“尖”。

### 局限性

- **Fails when**: 输入图像标定不准、位姿噪声大、场景不是静态、或测试视角只看到局部背景且缺乏上下文时，语义场容易漂移，背景类别尤其容易混淆。
- **Assumes**: 依赖高质量多视图图像与 COLMAP 式相机标定；依赖预训练 CLIP/DINO 的语义质量；训练/推理使用较强 GPU 资源（文中为 RTX A5000 24GB）；代码仅“计划发布”，当前复现便利性受限。
- **Not designed for**: 动态场景、在线 SLAM、显式 3D instance mask 预测、或专门优化的语义分割任务。它的分割结果本质上仍是由 relevancy map 派生，而不是原生实例/掩码头输出。

### 可复用组件

- **GS + MHE 解耦范式**：几何/外观和语义分开建模，适合大规模 3D 语义场。
- **hybrid multi-scale CLIP 监督**：把多尺度信息前移到训练期，避免推理期暴力搜索。
- **DINO-guided dotsim 对齐**：适合把“语义强但边界弱”的 foundation model 特征变得更适合定位。
- **单次渲染 relevancy 查询**：对交互式 3D 检索、AR 选取、机器人语言指向都有直接价值。

## Local PDF reference

![[paperPDFs/Language_Embedding_Segmentation/IJCV_2025/2025_FMGS_Foundation_Model_Embedded_3D_Gaussian_Splatting_for_Holistic_3D_Scene_Understanding.pdf]]