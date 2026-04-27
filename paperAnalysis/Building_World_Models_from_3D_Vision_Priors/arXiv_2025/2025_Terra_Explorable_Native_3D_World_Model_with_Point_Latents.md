---
title: "Terra: Explorable Native 3D World Model with Point Latents"
venue: arXiv
year: 2025
tags:
  - 3D_Gaussian_Splatting
  - task/3d-world-modeling
  - task/3d-scene-generation
  - flow-matching
  - point-latent
  - point-to-gaussian
  - dataset/ScanNet-v2
  - opensource/no
core_operator: "以稀疏点潜变量作为原生3D状态，通过P2G-VAE压缩并解码为3D Gaussian，再用联合位置-特征的流匹配在潜空间中渐进外扩场景。"
primary_logic: |
  RGBD反投影得到彩色点云或局部点潜变量条件 → P2G-VAE将场景压缩为稀疏语义点并解码成可渲染3D Gaussian，SPFlow对点的位置与特征联合去噪并做条件外扩 → 输出可任意视角渲染、具多视角一致性且可逐步探索的完整3D环境
claims:
  - "Terra在ScanNet v2重建上取得最高PSNR 19.742、SSIM 0.753以及最佳深度指标Abs. Rel. 0.026和δ1 0.978，但LPIPS并非最佳 [evidence: comparison]"
  - "在ScanNet v2无条件生成上，Terra的几何质量显著更好，P-FID 8.79优于Trellis的19.62和Prometheus的32.35 [evidence: comparison]"
  - "Distance-aware trajectory smoothing对未结构化点流匹配至关重要；移除后P-FID从8.79恶化到24.84，FID从307.2恶化到401.8 [evidence: ablation]"
related_work_position:
  extends: "Can3Tok (Gao et al. 2025)"
  competes_with: "Prometheus (Yang et al. 2025); Trellis (Xiang et al. 2025)"
  complementary_to: "VGGT (Wang et al. 2025a); DUST3R (Wang et al. 2024a)"
evidence_strength: moderate
pdf_ref: paperPDFs/Building_World_Models_from_3D_Vision_Priors/arXiv_2025/2025_Terra_Explorable_Native_3D_World_Model_with_Point_Latents.pdf
category: 3D_Gaussian_Splatting
---

# Terra: Explorable Native 3D World Model with Point Latents

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2510.14977)
> - **Summary**: 这篇工作把世界模型的内部状态从“像素/深度/位姿”切换为“稀疏点潜变量”，再一次性解码为3D Gaussian，因此能在原生3D空间里做一致的场景生成与渐进式探索。
> - **Key Performance**: ScanNet v2上重建达到 **PSNR 19.742 / δ1 0.978**；无条件生成几何质量达到 **P-FID 8.79**。

> [!info] **Agent Summary**
> - **task_path**: 部分RGBD观测/彩色点云或局部点潜变量条件 -> 可渲染3D Gaussian场景 -> 渐进式可探索完整环境
> - **bottleneck**: 像素对齐世界模型需要隐式学习跨视角重投影与像素对应，导致3D一致性差且新视角常要重复生成
> - **mechanism_delta**: 用原生3D点潜变量替代2D/2.5D状态，并在点的位置与特征上做联合流匹配，把难题从“学相机几何耦合”改成“学3D场景分布”
> - **evidence_signal**: 无条件生成的几何指标优势最强，P-FID 8.79显著优于Prometheus 32.35和Trellis 19.62，且轨迹平滑消融影响巨大
> - **reusable_ops**: [P2G-VAE点到Gaussian解码, distance-aware trajectory smoothing]
> - **failure_modes**: [纹理图像指标落后于带2D diffusion预训练的Prometheus, 依赖可靠深度与位姿反投影而非纯单目开放世界输入]
> - **open_questions**: [如何扩展到动态时序世界而非静态室内场景, 如何在更大尺度室内外数据上同时保持几何一致性与纹理逼真度]

## Part I：问题与挑战

Terra真正要解决的，不是“再做一个3D生成器”，而是**世界模型的状态表示选错了**。

现有很多world model本质上仍把世界状态写成多视角的 RGB / depth / pose。这样做的问题是：模型在生成时必须同时学会：

1. 图像外观分布；
2. 深度与相机位姿；
3. 跨视角像素对应；
4. 透视投影下的重投影一致性。

这几个约束在像素域里是强耦合的，所以模型即便能生成“看起来不错”的单帧图像，也容易在多视角时出现几何漂移、纹理错位，或者必须针对新视角重新跑一遍生成流程。  
**真正的瓶颈**因此不是采样器不够强，而是**世界状态仍然是视角相关的表示**。

### 输入/输出接口与边界

- **输入**：由RGBD和相机位姿反投影得到的彩色点云；探索阶段还可输入局部点潜变量条件。
- **输出**：稀疏点潜变量，以及其解码后的3D Gaussian场景，可从任意视角渲染。
- **任务形式**：把“可探索世界模型”写成**原生3D潜空间中的outpainting**，逐步扩展未知区域。
- **边界条件**：主要验证于 **ScanNet v2 静态室内场景**；这里的“world model”更接近**可探索静态3D环境生成**，不是动作驱动的动态视频世界模拟。

### 为什么现在值得做

因为几个关键条件成熟了：

- 3D Gaussian splatting让“生成后可渲染”变得高效；
- 稀疏点/稀疏3D卷积让大尺度3D表征更可训练；
- flow matching让点集生成比传统像素扩散更直接。

所以 Terra 的切入点很清楚：**与其在2D里补3D约束，不如直接把状态搬到3D里。**

## Part II：方法与洞察

Terra由两部分组成：

1. **P2G-VAE**：把输入彩色点云压缩为稀疏点潜变量，再解码成3D Gaussian；
2. **SPFlow**：在点潜空间上做生成，联合去噪点的位置和特征。

其训练分三阶段：

- 重建训练；
- 无条件生成预训练；
- 带mask条件的条件生成训练。

最终，探索过程就是：给定已知局部区域，在点潜空间里继续外扩。

### 核心直觉

**核心变化**：  
把世界状态从“多视角像素+深度+位姿”换成“位于3D表面上的稀疏语义点”。

**改变了什么瓶颈**：  
这一步直接去掉了生成过程中最难的隐式约束——跨视角像素对应与重投影一致性。模型不再需要一边生成图像、一边偷偷学透视几何，而是直接学生成3D场景本身。

**带来什么能力变化**：  
一旦点潜变量被解码为3D Gaussian，渲染任何视角都只是3D到2D的栅格化问题，而不是再次生成问题。因此 Terra 获得了：

- 天然的多视角一致性；
- 一次生成、多视角渲染；
- 在3D空间中逐步探索未知区域。

**为什么有效**，从因果上看有三层：

1. **点潜变量比像素更低冗余**：  
   它只保留表面上的稀疏语义点，减少了多视角重复信息。
2. **Gaussian解码把“世界表示”和“视角渲染”解耦**：  
   先生成世界，再渲染视角。
3. **联合去噪位置与特征**：  
   几何和外观不是分开的两个头，而是同一个点上的耦合属性，能互相补强。

### 方法拆解

#### 1. 原生点潜变量表示

每个探索状态都不是一组图像，而是一个点集：
- 每个点包含 **3D位置 + 语义特征**；
- 点数可随区域复杂度变化；
- 历史上下文可直接拼接到当前点集上。

这比像素对齐状态更适合“探索”——因为扩一个新区域，本质就是往当前3D点集里接着长。

#### 2. P2G-VAE：点到Gaussian的可渲染压缩

P2G-VAE的目标不是简单重建点云，而是学一个**适合生成的3D潜空间**。

- **robust position perturbation**  
  不直接把点坐标硬拉向高斯先验，而是对坐标加入小幅高斯扰动。  
  因果含义是：让解码器学会容忍“生成时不可避免的轻微位置噪声”，从而提升生成鲁棒性。

- **adaptive upsampling + refine**  
  从稀疏潜点逐步长出更密的点结构，并预测位移做细化。  
  这让潜空间可以稀疏，而输出结构依然足够密、足够可渲染。

- **explicit color supervision**  
  直接把Gaussian颜色对齐到输入点云最近邻颜色，而不是只依赖渲染损失。  
  作用是给外观学习一条更短的监督路径。

#### 3. SPFlow：未结构化点集上的流匹配生成

SPFlow在点潜变量上做flow matching，并且**同时**对以下两者去噪：

- 点的位置；
- 点的语义/外观特征。

这很关键，因为 Terra 不是先生成几何再贴纹理，而是让两者在潜空间里一起演化。

其中最重要的设计是：

- **distance-aware trajectory smoothing**  
  点集没有固定网格索引，如果把噪声点和目标点随意按序号配对，流匹配轨迹会很弯、很难学。  
  Terra用距离感知的匹配，把噪声点尽量分配给附近的目标点，相当于把运输轨迹“拉直”，显著提升收敛和生成质量。

#### 4. 条件机制与探索方式

条件设计很简单：把条件点潜变量与噪声点拼起来，整个去噪过程都固定这个条件。

支持三种条件：
- **crop**：给一个连通局部区域，要求模型向外想象未知部分；
- **uniform sampling**：给稀疏全局锚点，要求模型补全与细化；
- **combination**：模拟RGBD局部观测。

这使 Terra 的“探索”不是视频rollout，而是**3D latent outpainting**。

### 策略性权衡

| 设计选择 | 改变的约束/瓶颈 | 带来的能力 | 代价/副作用 |
|---|---|---|---|
| 原生点潜变量替代像素对齐状态 | 去掉跨视角重投影学习负担 | 多视角一致、一次生成后任意视角渲染 | 需要3D输入预处理，适用域更偏有几何先验的数据 |
| P2G-VAE解码为3D Gaussian | 把“生成世界”和“渲染视角”解耦 | 输出直接可渲染、适合大场景表示 | 训练链条更长，重建与生成要共同兼顾 |
| robust position perturbation | 缓解解码器对精确坐标的脆弱性 | 生成时更稳，对噪声更鲁棒 | 纯重建指标可能不一定提升，存在重建-生成权衡 |
| 联合位置-特征流匹配 | 几何/纹理解耦学习效率低 | 结构与外观互相补强 | 视觉逼真度仍不如强2D先验模型 |
| distance-aware trajectory smoothing | 点集无固定索引导致流轨迹弯曲 | 收敛更稳、几何质量显著提升 | 需要额外匹配求解开销 |
| 渐进式条件外扩 | 一次性生成大场景难 | 支持exploration与局部补全 | 长链探索的误差累积未被系统评估 |

## Part III：证据与局限

### 关键证据

#### 1. 对比实验：Terra的优势主要在“几何一致性”，不是“2D纹理先验”

- **重建（Table 1）**：  
  Terra拿到最高 **PSNR 19.742、SSIM 0.753、Abs. Rel. 0.026、δ1 0.978**。  
  这说明P2G-VAE不仅能压缩3D场景，还能把它解码回高质量可渲染Gaussian。

- **无条件生成（Table 2）**：  
  Terra 的 **P-FID 8.79**，显著优于 Prometheus 的 32.35 和 Trellis 的 19.62。  
  这是全文最强的证据：它证明 Terra 学到的是更好的**3D几何分布**。

- **图像条件生成（Table 2, Figure 6）**：  
  Terra在 **Chamfer Distance / EMD** 上优于Prometheus，说明给定局部观测时，补全出来的3D结构更一致。  
  但FID/KID仍不如Prometheus，说明 Terra 当前的优势在**结构正确**，不是**图像风格更像2D diffusion**。

#### 2. 消融实验：最关键的因果旋钮是“轨迹变直”和“解码器抗噪”

- **distance-aware trajectory smoothing** 去掉后，  
  P-FID 从 **8.79 → 24.84**，FID 从 **307.2 → 401.8**。  
  这是最强的因果证据：未结构化点集做flow matching时，点-噪声匹配策略不是细节，而是成败关键。

- **robust position perturbation**  
  会牺牲一些重建表现，但明显提升生成质量。  
  这说明作者不是单纯追求自编码器重建，而是在为生成阶段的坐标噪声做准备。

#### 3. 定性结果：能力跃迁在“可探索的一致3D世界”

Figure 7 展示了逐步探索未知区域的结果。  
和很多“多视角图像看起来像”但底层几何不稳的方法相比，Terra的能力跳跃在于：

- 世界只生成一次；
- 后续视角由渲染得到，不靠重复采样；
- 探索是在3D状态空间中扩展，而不是在像素轨迹上补帧。

### 局限性

- **Fails when**:  
  - 任务把**图像逼真度**放在几何一致性之前时，Terra会输给带强2D diffusion预训练的模型；  
  - 输入没有可靠深度/位姿、无法稳定反投影到3D时，本文方案未验证；  
  - 分布超出静态室内ScanNet风格场景时，如室外大尺度、动态物体场景，泛化未知。

- **Assumes**:  
  - 需要 **RGBD + 相机位姿/内参** 来构造彩色点云；  
  - 假设场景主要是**静态可表面化**的3D环境；  
  - 依赖稀疏3D网络、点集处理与匹配求解器；  
  - 论文正文未给出开源链接，硬件细节也未在给定文本中明确，复现门槛不低。

- **Not designed for**:  
  - 动作条件的时序世界动力学；
  - 纯文本到3D世界生成；
  - 长时闭环规划或在线机器人控制；
  - 完全单目、无深度先验的开放世界场景建模。

### 可复用组件

- **点潜变量作为世界状态**：适合任何想从“视角状态”切到“3D状态”的world model。
- **P2G-VAE**：把稀疏3D潜变量和可渲染Gaussian输出连接起来。
- **robust position perturbation**：适合所有“坐标本身也是latent”的生成模型。
- **distance-aware trajectory smoothing**：对无序点集/集合数据上的flow matching都很有参考价值。
- **latent-space outpainting训练范式**：可迁移到3D补全、可探索场景生成、局部条件扩展等任务。

## Local PDF reference

![[paperPDFs/Building_World_Models_from_3D_Vision_Priors/arXiv_2025/2025_Terra_Explorable_Native_3D_World_Model_with_Point_Latents.pdf]]