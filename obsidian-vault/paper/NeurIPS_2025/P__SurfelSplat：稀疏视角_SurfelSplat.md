---
title: 'SurfelSplat: Learning Efficient and Generalizable Gaussian Surfel Representations for Sparse-View Surface Reconstruction'
type: paper
paper_level: A
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- SurfelSplat：稀疏视角表面重建的高效可泛化高斯曲面元
- SurfelSplat
- SurfelSplat achieves efficient
acceptance: Poster
method: SurfelSplat
modalities:
- Image
paradigm: supervised
---

# SurfelSplat: Learning Efficient and Generalizable Gaussian Surfel Representations for Sparse-View Surface Reconstruction

**Topics**: [[T__3D_Reconstruction]] | **Method**: [[M__SurfelSplat]] | **Datasets**: DTU sparse-view, DTU sparse-view inference efficiency

> [!tip] 核心洞察
> SurfelSplat achieves efficient, generalizable, and geometrically accurate sparse-view surface reconstruction by applying Nyquist sampling theorem-guided surfel adaptation and cross-view feature aggregation in a feed-forward framework.

| 中文题名 | SurfelSplat：稀疏视角表面重建的高效可泛化高斯曲面元 |
| 英文题名 | SurfelSplat: Learning Efficient and Generalizable Gaussian Surfel Representations for Sparse-View Surface Reconstruction |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.08370) · [Code](待补充) · [Project](待补充) |
| 主要任务 | 稀疏视角表面重建（Sparse-view Surface Reconstruction） |
| 主要 baseline | pixelSplat, 2DGS, UFORecon, FatesGS, NeuSurf, GausSurf, NeuS |

> [!abstract] 因为「现有 feed-forward 网络无法恢复高斯曲面元的准确几何属性，其像素对齐基元的空间频率违反奈奎斯特采样定理」，作者在「pixelSplat」基础上改了「将 3D 高斯替换为 2D 高斯曲面元，并引入奈奎斯特定理引导的曲面元自适应与跨视图特征聚合」，在「DTU 稀疏视角（2-view）」上取得「Chamfer Distance 1.12，相比 FatesGS 的 1.18 提升 5.1%」

- **关键性能 1**：DTU 2-view 重建 Chamfer Distance 1.12，优于 FatesGS (1.18)、UFORecon (1.91)、NeuSurf (1.44)
- **关键性能 2**：推理速度 < 1 秒，相比逐场景优化方法（~10 分钟）实现 100× 加速
- **关键性能 3**：消融实验显示，去掉跨视图特征聚合模块后 CD 从 1.12 恶化至 1.96，性能下降 75%

## 背景与动机

从少量图像重建精确的三维表面是计算机视觉的核心挑战之一。现有方法分为两类：神经隐式表示（如 NeuS、NeuSurf）需要逐场景优化，耗时数十分钟；而 feed-forward 网络（如 pixelSplat）虽能快速推理，却难以恢复准确的几何属性。具体而言，当仅有 2-3 个稀疏视角时，pixelSplat 预测的像素对齐 3D 高斯基元会出现方向估计错误，导致重建表面出现伪影与孔洞。


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/305eea01-de47-428d-915c-5bf79ea70572/figures/Figure_2.png)
*Figure 2 (motivation): Experimental Observations.*



现有方法的局限可从三个角度理解：
- **pixelSplat** [23] 直接从跨视图特征回归深度与外观，但缺乏频率感知处理，3D 高斯的方向在稀疏采样下无法正确恢复；
- **2DGS** [18] 使用扁平 2D 高斯曲面元提升几何精度，但仍需逐场景优化，且稀疏视角下产生粗糙不完整的表面；
- **FatesGS** [66] 等稀疏视角高斯方法虽专门优化少视角设置，但未解决像素对齐基元与采样率之间的根本矛盾。

核心问题在于：像素对齐的几何基元的空间频率超过了输入图像的奈奎斯特采样率，导致高频几何信息混叠（aliasing）。如图 2 所示，当采样不足时，高斯曲面元的方向与尺度估计产生系统性偏差。本文据此提出：在跨视图处理前，先以奈奎斯特采样定理为指导对曲面元进行频率自适应滤波，从而保证几何信息可被准确重建。

## 核心创新

核心洞察：像素对齐的 2D 高斯曲面元在稀疏视角下的几何不可辨识性，本质上是空间频率混叠问题，因为输入图像的采样率低于曲面元表征所需的最小频率，从而通过奈奎斯特采样率引导的低通滤波可使曲面元几何属性恢复成为可能。

| 维度 | Baseline (pixelSplat) | 本文 (SurfelSplat) |
|:---|:---|:---|
| 几何基元 | 3D 高斯椭球 | 2D 高斯曲面元（扁平有向圆盘） |
| 频率处理 | 无，直接回归深度/外观 | 奈奎斯特定理引导的空间采样率自适应低通滤波 |
| 特征聚合 | 标准跨视图特征匹配 | 滤波后曲面元投影至所有视图，获取跨视图特征相关性 |
| 监督目标 | 渲染损失 L_render | L_render + λ_geo·(λ_align·L_align + λ_d·L_d + λ_n·L_n) |
| 推理范式 | 逐场景优化或 feed-forward | 纯 feed-forward，< 1 秒 |

与 pixelSplat 的关键差异在于：本文将几何基元从体积式 3D 高斯替换为表面式 2D 高斯曲面元，并在特征聚合前插入频率自适应模块，从根本上解决了稀疏采样下的几何混叠问题。

## 整体框架


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/305eea01-de47-428d-915c-5bf79ea70572/figures/Figure_3.png)
*Figure 3 (architecture): Pipeline: Given an input pair, our method first extracts initial image features using a vision transformer.*



SurfelSplat 的完整数据流如下：

1. **输入**：稀疏视角图像对（如 2-view），通过 Vision Transformer 提取初始图像特征；
2. **像素对齐曲面元预测**：基于 pixelSplat 的编码器-解码器结构，但将输出从 3D 高斯参数改为 2D 高斯曲面元参数（中心位置、法向方向、各向异性尺度、不透明度、SH 系数）；
3. **奈奎斯特曲面元自适应（Nyquist Surfel Adaptation）**：根据当前视图的空间采样率，对预测的几何形式（方向、尺度）应用低通滤波，抑制超过奈奎斯特频率的高频成分；
4. **跨视图特征聚合（Cross-view Feature Aggregation）**：将滤波后的 2D 曲面元投影至所有输入视图，在每个视图上获取对应的特征响应，形成跨视图特征相关性体；
5. **特征融合网络**：将多视图相关性特征通过专门设计的融合网络，回归最终的几何精确的 2D 高斯曲面元；
6. **可微曲面元渲染**：使用 2D 高斯曲面元的可微光栅化，输出渲染图像、深度图与法线图；
7. **联合监督**：渲染损失与几何约束（对齐、深度、法向）共同优化。

```
图像对 → ViT 特征提取 → 像素对齐 2D 曲面元预测 
                                    ↓
              奈奎斯特自适应滤波（低通滤波几何参数）
                                    ↓
              跨视图投影 → 特征相关性计算 → 融合网络
                                    ↓
              几何精确 2D 曲面元 → 可微渲染 → 图像/深度/法线
                                    ↓
              L_render + λ_geo·L_geo（联合监督）
```

整个框架保持 feed-forward 特性，训练分为两阶段：先在 RealEstate10K 上预训练 300,000 轮，再在 DTU 上微调 2,000 轮。

## 核心模块与公式推导

### 模块 1: 奈奎斯特采样定理引导的曲面元自适应（对应框架图步骤 3）

**直觉**：稀疏视角下，图像像素的空间采样率有限，若曲面元的几何变化频率超过该采样率，则方向与尺度估计会出现混叠；需在跨视图处理前以低通滤波限制其频率内容。

**Baseline 公式 (pixelSplat)**：pixelSplat 直接预测 3D 高斯参数，无显式频率约束：
$$\mathbf{G}_{3D} = \{(\mu_i, \Sigma_i, c_i, \alpha_i)\}_{i=1}^{N}$$
其中 $\mu_i \in \mathbb{R}^3$ 为中心，$\Sigma_i \in \mathbb{R}^{3\times 3}$ 为协方差，$c_i$ 为颜色，$\alpha_i$ 为不透明度。

**变化点**：pixelSplat 的 3D 高斯在稀疏视角下方向不可辨识；本文改用 2D 曲面元 $\mathbf{S}_i = (\mathbf{p}_i, \mathbf{n}_i, s_i^{(1)}, s_i^{(2)}, \alpha_i, c_i)$，其中 $\mathbf{p}_i$ 为 3D 位置，$\mathbf{n}_i$ 为法向，$s_i^{(1)}, s_i^{(2)}$ 为切平面内两个轴向的尺度。关键假设：曲面元的几何属性（方向、尺度）在图像平面上的变化率应受限于该视图的空间采样率。

**本文公式（推导）**：
$$\text{Step 1}: \quad f_{Nyq}^{(v)} = \frac{1}{2\Delta x^{(v)}}$$
其中 $f_{Nyq}^{(v)}$ 为视图 $v$ 的奈奎斯特频率，$\Delta x^{(v)}$ 为像素间距对应的三维空间采样间隔。

$$\text{Step 2}: \quad H^{(v)}(f) = \begin{cases} 1 & |f| \leq f_{Nyq}^{(v)} \\ 0 & \text{otherwise} \end{cases}$$
设计理想低通滤波器，截止频率为奈奎斯特频率。

$$\text{Step 3}: \quad \tilde{\mathbf{S}}_i^{(v)} = \mathcal{F}^{-1}\left(H^{(v)} \cdot \mathcal{F}(\mathbf{S}_i)\right)$$
对曲面元几何形式进行频域滤波，得到自适应后的曲面元 $\tilde{\mathbf{S}}_i^{(v)}$。

**对应消融**：Figure 6 展示了空间频率元自适应的可视化验证，证明滤波后的曲面元频率内容符合奈奎斯特约束。

---

### 模块 2: 跨视图特征聚合（对应框架图步骤 4-5）

**直觉**：单视图特征不足以确定曲面元的精确几何，需将滤波后的曲面元投影到所有可用视图，利用多视图一致性约束来融合特征。

**Baseline 公式 (pixelSplat)**：标准跨视图特征匹配，直接基于特征相似性聚合：
$$\mathbf{F}_{agg}^{(pixelSplat)} = \text{Softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$
其中 $Q, K, V$ 来自不同视图的特征，无显式几何投影。

**变化点**：pixelSplat 的特征匹配缺乏几何一致性约束；本文先将曲面元投影到各视图获取像素级特征响应，再融合，使特征聚合与几何投影一致。

**本文公式（推导）**：
$$\text{Step 1}: \quad \pi_v(\tilde{\mathbf{S}}_i^{(v)}) \rightarrow (u_i^{(v)}, v_i^{(v)}, \text{cov}_i^{(v)})$$
将自适应后的曲面元 $\tilde{\mathbf{S}}_i^{(v)}$ 投影至视图 $v$ 的图像平面，得到投影位置与投影协方差。

$$\text{Step 2}: \quad \mathbf{f}_i^{(v)} = \text{Sample}\left(\mathbf{F}^{(v)}, (u_i^{(v)}, v_i^{(v)}), \text{cov}_i^{(v)}\right)$$
以投影协方差为核，在视图 $v$ 的特征图 $\mathbf{F}^{(v)}$ 上进行可变形采样，获取该曲面元在视图 $v$ 上的特征响应。

$$\text{Step 3}: \quad \mathbf{C}_i = \left[\mathbf{f}_i^{(1)}; \mathbf{f}_i^{(2)}; ...; \mathbf{f}_i^{(V)}\right]$$
拼接所有视图的响应，形成跨视图特征相关性矩阵 $\mathbf{C}_i \in \mathbb{R}^{V \times d}$。

$$\text{Step 4}: \quad \hat{\mathbf{S}}_i = \text{FusionNet}(\mathbf{C}_i)$$
通过专门设计的特征融合网络（含跨视图注意力与 MLP），回归几何精确的曲面元参数 $\hat{\mathbf{S}}_i$。

**对应消融**：Table 3 显示，去掉聚合模块（w/o Aggre.）后 CD 从 1.12 升至 1.96，性能下降 75%，证明该模块对几何精度至关重要。

---

### 模块 3: 联合几何监督目标（对应框架图步骤 7）

**直觉**：仅依靠渲染损失无法保证表面重建的几何准确性，需显式引入深度、法向与跨视图对齐约束。

**Baseline 公式 (pixelSplat)**：仅渲染损失
$$L_{pixelSplat} = L_{render} = \sum_{v}\|I^{(v)} - \hat{I}^{(v)}\|_1$$

**变化点**：本文针对表面重建任务，增加显式几何约束项。

**本文公式**：
$$L_{final} = L_{render} + \lambda_{geo}L_{geo}$$

其中几何损失：
$$L_{geo} = \lambda_{align}L_{align} + \lambda_d L_d + \lambda_n L_n$$

各项含义：
- $L_{align}$：跨视图曲面元投影位置的对齐损失，确保同一点在不同视图投影一致；
- $L_d = \sum_v\|D^{(v)} - \hat{D}^{(v)}\|_1$：深度图监督，$D^{(v)}$ 为 GT 深度；
- $L_n = \sum_v(1 - \mathbf{n}^{(v)} \cdot \hat{\mathbf{n}}^{(v)})$：法向一致性损失，促进表面平滑。

权重设置：$\lambda_{geo}, \lambda_{align}, \lambda_d, \lambda_n$ 为超参数，在实现中通过网格搜索确定。

## 实验与分析


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/305eea01-de47-428d-915c-5bf79ea70572/figures/Table_1.png)
*Table 1 (quantitative): The quantitative comparison results of Chamfer Distance (CD) on DTU dataset.*



本文在 DTU 数据集的稀疏视角（2-view）设置下进行评测，使用 Chamfer Distance (CD) 作为主要指标。如 Table 1 所示，SurfelSplat 取得 mean CD 1.12，在全部对比方法中排名第一。具体而言，该结果相比直接竞争对手 FatesGS（CD 1.18）提升 5.1%，相比基于神经隐式的 NeuSurf（CD 1.44）提升 22.2%，相比需要逐场景优化的 GausSurf（CD 4.36）提升 74.3%。这一差距表明，通过奈奎斯特频率自适应与跨视图聚合，feed-forward 网络首次在稀疏视角表面重建上达到了与专门优化方法相当甚至更好的几何精度。


![Table 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/305eea01-de47-428d-915c-5bf79ea70572/figures/Table_2.png)
*Table 2 (quantitative): Comparison with previous methods on the DTU dataset.*



Table 2 进一步展示了效率对比：SurfelSplat 的推理时间低于 1 秒，而 NeuS 等隐式方法需要约 10 分钟逐场景优化，实现约 100× 加速。这一速度优势来自纯 feed-forward 设计，无需测试时优化。


![Figure 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/305eea01-de47-428d-915c-5bf79ea70572/figures/Figure_4.png)
*Figure 4 (qualitative): Qualitative Comparison of Surface Reconstruction with Sparse Views on DTU Benchmarks.*



图 4 的定性对比显示，在 DTU 典型场景上，SurfelSplat 重建的表面细节更丰富、边缘更清晰，而 2DGS 出现大面积缺失，UFORecon 产生过度平滑的全局形状。


![Table 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/305eea01-de47-428d-915c-5bf79ea70572/figures/Table_3.png)
*Table 3 (ablation): Ablation study of surface aggregation module on DTU benchmark.*



消融实验（Table 3）量化了各组件贡献：去掉跨视图特征聚合模块（w/o Aggre.）导致 CD 从 1.12 恶化至 1.96（+75.0%）；去掉奈奎斯特自适应（w/o Adapt.）导致 CD 升至 1.58（+41.1%）；两者均去掉（w/o Both）进一步恶化至 2.34。这验证了频率自适应与特征聚合的互补性——前者解决单视图混叠，后者利用多视图一致性精化几何。


![Figure 6](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/305eea01-de47-428d-915c-5bf79ea70572/figures/Figure_6.png)
*Figure 6 (ablation): Spatial Frequency Meta-Adaption.*



公平性检查：本文对比的 baselines 涵盖了当前稀疏视角重建的主流方法，包括 feed-forward 的 FatesGS、UFORecon，逐场景优化的 2DGS、GausSurf，以及神经隐式的 NeuS、NeuSurf。但需注意：（1）SurfelSplat 在 1024×1024 高分辨率下会产生超过 100 万曲面元，显著降低渲染与推理速度；（2）对于输入视图未观测到的区域，feed-forward 架构无法恢复合理几何；（3）训练依赖 RealEstate10K 的大规模预训练，小数据集上泛化能力未验证。

## 方法谱系与知识库定位

SurfelSplat 属于 **3D Gaussian Splatting → 稀疏视角重建** 方法族，直接父方法为 **pixelSplat** [23]。

**关键改动槽位**：
- **几何表示**：3D 高斯椭球 → 2D 高斯曲面元（表面基元替代体积基元）
- **数据处理**：新增奈奎斯特采样定理引导的低通滤波模块（频率自适应）
- **特征聚合**：标准跨视图匹配 → 滤波后曲面元投影 + 跨视图相关性融合
- **训练目标**：纯渲染损失 → 渲染 + 对齐/深度/法向联合几何约束

**直接 baselines 与差异**：
- **pixelSplat**：基础框架，SurfelSplat 在其上替换基元并新增频率模块
- **2DGS** [18]：同为 2D 高斯曲面元，但需逐场景优化，无 feed-forward 泛化
- **GausSurf** [19]：同名"Gaussian surfels"，但采用逐场景优化，CD 4.36 vs 1.12
- **FatesGS** [66]：同为 feed-forward 稀疏视角高斯方法，但无频率自适应机制
- **UFORecon** [8]：通用稀疏视角重建，仅能产生粗糙全局几何

**后续方向**：
1. 扩展至多视图（>4）设置，探索采样率自适应与视图数量的联合优化；
2. 结合单目深度先验，缓解未观测区域的几何缺失；
3. 开发自适应曲面元密度控制，解决高分辨率下百万级 surfel 的效率瓶颈。

**知识库标签**：
- 模态：image → 3D
- 范式：feed-forward supervised learning
- 场景：sparse-view surface reconstruction
- 机制：Nyquist sampling, cross-view feature aggregation, 2D Gaussian surfel
- 约束：real-time inference, no per-scene optimization

## 引用网络

### 直接 baseline（本文基于）

- SuGaR: Surface-Aligned Gaussian Splatting for Efficient 3D Mesh Reconstruction and High-Quality Mesh Rendering _(CVPR 2024, 直接 baseline, 未深度分析)_: Directly combines Gaussian splatting with surface reconstruction. Very close to 

