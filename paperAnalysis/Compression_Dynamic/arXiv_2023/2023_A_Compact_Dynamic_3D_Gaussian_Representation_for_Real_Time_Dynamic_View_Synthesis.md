---
title: "A Compact Dynamic 3D Gaussian Representation for Real-Time Dynamic View Synthesis"
venue: arXiv
year: 2023
tags:
  - 3D_Gaussian_Splatting
  - task/video-generation
  - gaussian-splatting
  - fourier-approximation
  - optical-flow-supervision
  - dataset/D-NeRF
  - dataset/DyNeRF
  - dataset/HyperNeRF
  - opensource/no
core_operator: 将每个3D Gaussian的中心与旋转参数化为随时间变化的低维函数，并用图像平面光流监督增强跨帧一致性。
primary_logic: |
  带时间戳的视频帧与相机参数 → 用傅里叶基表示Gaussian位置、用线性函数表示旋转并共享跨时刻参数 → 结合图像重建与光流监督优化高斯集合 → 输出内存随基函数阶数而非序列长度增长的实时动态视角合成表示
claims:
  - "在 D-NeRF 上，该方法达到 32.19 PSNR 和 150 FPS，重建质量接近 TiNeuVox-B/K-Planes，同时显著优于静态 3DGS 与逐帧 D-3DGS 的质量表现 [evidence: comparison]"
  - "在 DyNeRF 上，该方法达到 30.46 PSNR、118 FPS、约 338MB 参数内存，相比 Dynamic3DGaussians 的 27.79 PSNR、51 FPS、约 6.6GB 更快且更紧凑 [evidence: comparison]"
  - "在 D-NeRF 的设计消融中，位置的傅里叶近似（L=2）取得最佳平均 PSNR/SSIM；更复杂的 spline 或更大 L 只带来场景依赖的小幅收益并降低渲染效率 [evidence: ablation]"
related_work_position:
  extends: "3D Gaussian Splatting (Kerbl et al. 2023)"
  competes_with: "Dynamic3DGaussians (Luiten et al. 2024); K-Planes (Fridovich-Keil et al. 2023)"
  complementary_to: "N/A"
evidence_strength: strong
pdf_ref: paperPDFs/Compression_Dynamic/arXiv_2023/2023_A_Compact_Dynamic_3D_Gaussian_Representation_for_Real_Time_Dynamic_View_Synthesis.pdf
category: 3D_Gaussian_Splatting
---

# A Compact Dynamic 3D Gaussian Representation for Real-Time Dynamic View Synthesis

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2311.12897)
> - **Summary**: 该文把动态 3DGS 从“每一帧存一套 Gaussian 状态”改成“每个 Gaussian 携带一条低维时间函数轨迹”，从而在单目/少视角动态场景里同时实现更低内存、更强时序共享和实时渲染。
> - **Key Performance**: DyNeRF 上达到 **30.46 PSNR / 118 FPS / ~338MB**；在 **1352×1014** 高分辨率下单卡实时渲染。

> [!info] **Agent Summary**
> - **task_path**: 带时间戳的单目/多视角视频 + 相机参数 -> 任意时刻任意视角图像
> - **bottleneck**: 动态 3DGS 若逐帧存储 Gaussian 的位置与旋转，会导致 O(TN) 内存增长，并使每个时刻的监督过稀，单目/少视角下难以稳定优化
> - **mechanism_delta**: 用连续时间低维函数替代逐帧 Gaussian 状态，并引入图像平面光流监督把相邻时刻耦合起来
> - **evidence_signal**: DyNeRF 上在保持接近 NeRF 质量的同时达到 118 FPS，且参数内存从 Dynamic3DGaussians 的约 6.6GB 降到约 338MB
> - **reusable_ops**: [连续时间Gaussian轨迹参数化, 图像平面光流监督]
> - **failure_modes**: [拓扑变化与高斯生灭, 超长复杂序列下固定低阶基函数欠拟合]
> - **open_questions**: [如何为Gaussian建模生命周期与拓扑变化, 如何在保持3DGS速度时恢复NeRF式连续性与泛化]

## Part I：问题与挑战

先说明一下：你给出的标题与正文不一致；按提供的全文内容，这里实际分析的是 **《A Compact Dynamic 3D Gaussian Representation for Real-Time Dynamic View Synthesis》**，而不是文中引用的并行工作 *SpacetimeGaussian*。

### 这篇论文真正要解决什么问题？

它要解决的不是“动态场景能不能重建”，而是：

**能不能在单目/少视角条件下，用 3DGS 风格的显式表示，做出既快、又省内存、还能保持质量的动态新视角合成。**

已有方法的矛盾很明确：

- **NeRF 系动态方法**：质量往往不错，但训练/渲染慢，难实时。
- **静态 3DGS**：速度极快，但直接用于动态场景会失败。
- **动态 3DGS 的直接扩展**：最自然的做法是“每个时间步存一套位置/旋转”，但这会带来两个核心问题：
  1. **内存随序列长度线性增长**；
  2. **每个时间步只能靠该时刻少量视角监督**，单目或 few-view 时尤其不稳定。

### 真正的瓶颈是什么？

真正瓶颈不是 rasterization 速度，而是**时间维表示方式**：

- 如果时间维是“逐帧离散状态表”，那每个 Gaussian 在每个时刻都要单独学；
- 这会让监督碎片化，无法让不同时间的观测共享信息；
- 最终导致：**内存大、数据效率低、对多视角同步采集依赖强**。

换句话说，这篇论文的关键不是发明一种更快的渲染器，而是把动态 3DGS 的时间表示从**离散记忆**改成**连续函数**。

### 输入/输出接口

- **输入**：
  - 带时间戳的视频帧
  - 相机内外参
  - 相邻帧的光流伪标签（真实数据上由 RAFT 估计）
- **输出**：
  - 一个连续时间定义的动态 3D Gaussian 集合
  - 可在任意时间、任意视角下渲染图像

### 边界条件

这套方法默认了几个很重要的前提：

- 大多数物体的**尺度、颜色、不透明度随时间基本不变**
- 运动是**相对平滑、可被低维基函数拟合**
- 相机位姿已知，且真实场景可借助 **SfM** 初始化
- 更适合**持续存在的对象**，而不是频繁出现/消失或拓扑剧变的内容

---

## Part II：方法与洞察

### 方法骨架

这篇论文的表示设计非常直接：  
**只让最需要动的参数动起来，而且用低维函数去动。**

#### 1. 时间变化的参数：中心 + 旋转

每个 Gaussian 的动态部分只有两个：

- **3D 中心位置**：用少量 **Fourier 基** 表示随时间变化的轨迹
- **旋转**：用**线性函数**表示四元数分量的时间变化

这意味着，一个 Gaussian 不再为每个时间步存独立状态，而是只存一小组轨迹参数。

#### 2. 时间不变的参数：尺度 + 颜色 + 不透明度

作者把以下属性固定为时间不变：

- scale
- SH color
- opacity

这样做的目的很明确：**压缩参数量，避免把本不需要动态化的量也带进时间维。**

#### 3. 渲染仍然保持纯 3DGS

在任意时刻 \(t\)：

- 先根据时间函数求出该 Gaussian 的中心和旋转
- 得到当前时刻的协方差
- 然后直接走标准 3DGS splatting 流程

这点很关键：作者没有引入额外 MLP 或 deformation network，因此保住了 3DGS 最核心的优点——**超快渲染**。

#### 4. 训练策略：先静态、再动态

作者采用两阶段优化：

- **静态阶段**：先把所有帧“当成静态”，拟合共享的外观与基础几何
- **动态阶段**：再放开时间相关参数，学习运动

这个设计的作用是：  
避免一开始就同时学习外观、几何和运动，导致优化发散或高斯轨迹乱漂。

#### 5. 用光流补足时间一致性

仅靠图像重建，少视角动态场景仍然会有歧义。作者额外加入：

- 相邻帧光流监督
- 监督对象不是像素级隐式场，而是 **Gaussian 导出的投影运动**

这一步的作用是把“相邻时刻应该怎么动”显式约束住，减少 ghosting 和颜色漂移。

### 核心直觉

这篇论文最重要的因果链条可以概括成一句话：

**把动态 Gaussian 从“逐帧独立记忆”改成“连续时间低维轨迹”，会直接改变时间维的自由度结构；自由度下降后，跨时刻监督被共享，稀疏视角下的优化更稳，内存也从依赖序列长度转为依赖基函数阶数。**

更具体地说：

1. **改了什么？**  
   从“每时刻一套位置/旋转参数”改成“每个 Gaussian 一条时间函数”。

2. **改变了什么瓶颈？**  
   - 把 per-timestep 的高自由度表示压缩成低维轨迹；
   - 让所有时间帧共同约束同一组轨迹参数；
   - 从而缓解单目/少视角下“每时刻监督不足”的问题。

3. **带来了什么能力变化？**  
   - 内存从类似 **O(TN)** 转向 **O(LN)**，其中 \(L \ll T\)
   - 对未见视角更稳，因为轨迹更平滑、不会逐帧抖动
   - 仍保留 3DGS 的实时渲染能力和显式可编辑性

### 为什么作者选 Fourier，而不是更常见的多项式？

作者的判断很实用：

- 低阶多项式：表达力不够，容易欠拟合
- 高阶多项式：容易过拟合且不稳定
- **Fourier 基**：对平滑、准周期或中等复杂度运动更自然，参数也更经济

从结果看，这不是拍脑袋：他们确实做了不同基函数的对比，证明 Fourier 在质量/速度之间更均衡。

### 为什么旋转只做线性近似？

因为四元数本身有单位范数约束，做复杂时间函数更麻烦。  
作者采取的是一个工程上很稳的折中：

- 位置用更强的表达力
- 旋转用更轻的线性近似

这说明他们判断：**动态场景的主要变化瓶颈在位置轨迹，而不是高阶旋转轨迹。**

### 战略取舍表

| 设计选择 | 改变的约束/信息流 | 直接收益 | 代价/风险 |
| --- | --- | --- | --- |
| 位置用 Fourier 低维轨迹 | 把逐帧独立位置改成连续时间共享参数 | 内存更小、运动更平滑、少视角更稳 | 对超复杂/非平滑/超长序列可能欠拟合 |
| 旋转用线性函数 | 限制时变旋转自由度 | 训练更稳、参数更少 | 快速复杂旋转表达有限 |
| scale/color/opacity 设为静态 | 把时变外观从表示中剔除 | 压缩显著、优化更容易 | 难处理明显时变外观/尺度变化 |
| 加光流监督 | 用相邻帧运动对应补齐时间约束 | 减少 ghosting，提高时序一致性 | 依赖外部 RAFT；某些相机设置下不可用 |
| 保持纯 3DGS splatting | 不引入 MLP/deformation net | 维持 100+ FPS 与显式编辑性 | 继承 3DGS 对位姿误差和泛化的敏感性 |
| 两阶段训练 | 先学静态先验，再学动态细节 | 减少早期优化不稳定 | 多一个训练调度超参数 |

---

## Part III：证据与局限

### 关键证据

#### 1) 比较信号：D-NeRF（单目式设置）
- 结果：作者方法在 D-NeRF 上达到 **32.19 PSNR / 0.97 MS-SSIM / 0.04 LPIPS / 150 FPS**
- 结论：它在质量上接近 TiNeuVox-B、K-Planes 这类 NeRF 系方法，但渲染速度快得多；同时显著优于直接把静态 3DGS 或逐帧 D-3DGS 硬套到动态场景上的结果  
- 含义：这最能支持论文的核心主张——**时间参数共享确实缓解了单目/少视角下的监督稀疏问题**

#### 2) 比较信号：DyNeRF（多视角真实场景）
- 结果：**30.46 PSNR / 118 FPS / ~338MB**
- 对比：Dynamic3DGaussians 为 **27.79 PSNR / 51 FPS / ~6.6GB**
- 结论：即使在多相机真实场景里，这种紧凑时序参数化也没有只换来“更省内存”，而是同时带来了**更高速度 + 更小内存 + 不差的质量**

#### 3) 比较信号：HyperNeRF（两部手机拍摄）
- 结果：平均达到 **25.6 PSNR / 0.890 SSIM / 188 FPS**
- 结论：在较稀疏真实采集条件下，方法仍保持竞争性质量，并远快于 HyperNeRF、TiNeuVox-B、V4D
- 含义：说明这套表示不是只对合成数据或密集多视角有效

#### 4) 消融信号：时间基函数的选择
- 结果：在 D-NeRF 上，**L=2 的 Fourier 近似**取得最佳平均表现
- 结论：不是“参数越多越好”，而是要找到能表达运动、又不过拟合的甜点区
- 含义：支撑了作者“低维但足够表达”的设计哲学

#### 5) 案例/消融信号：光流损失
- 结果：加入 flow loss 后，定性结果显示 ghosting 减少、颜色更准
- 结论：图像重建本身不足以稳定动态时序，光流确实提供了额外的时序约束
- 含义：这不是主要创新点，但对真实场景稳定性很关键

### 能力跃迁到底在哪里？

这篇论文的提升，不是“绝对 PSNR 碾压所有方法”，而是把三件事第一次较好地放在一起：

1. **接近 NeRF 系动态方法的重建质量**
2. **接近静态 3DGS 的实时渲染速度**
3. **把动态表示的内存依赖从视频长度中解耦**

所以它的价值更像是一个**表示层折中点**：  
不是追求最强表达，而是追求“足够表达 + 极高效率 + 稀疏视角可用”。

### 局限性

- **Fails when**: 场景存在明显拓扑变化、高斯的出现/消失、流体这类非持续存在结构时；或视频非常长、运动非常复杂时，固定低阶时间基函数会欠拟合，提升 \(L\) 又会抬高内存和复杂度。
- **Assumes**: 相机参数可用；多数 Gaussian 在整个时间段持续存在；scale/color/opacity 大体不随时间变化；真实数据上依赖 SfM 初始化与外部 RAFT 光流；文中速度结果基于单张 RTX A6000。
- **Not designed for**: 显著时变外观建模、拓扑剧变场景、相机位姿纠错、在严重位姿误差下维持高泛化；也没有把 NeRF 式连续体平滑性完整带回 3DGS。

### 可复用组件

- **连续时间 Gaussian 轨迹参数化**：把逐帧状态压缩成共享轨迹，可迁移到别的动态显式表示
- **图像平面光流监督**：适合作为单目/少视角动态重建的时序正则
- **静态预热 + 动态微调** 两阶段训练：适合任何“外观先验 + 运动细化”的动态建模
- **动态场景中的 densify/prune 机制**：把 3DGS 的结构自适应继续保留到时间建模里

### 复现与可扩展性备注

- 文中未给出明确代码/项目链接，按保守标准记为 **opensource/no**
- 真值光流并非原生标注，而是依赖外部 RAFT 估计
- D-NeRF 因相邻帧存在 camera teleport，不使用光流损失；这说明其监督设计对采集协议是敏感的

## Local PDF reference

![[paperPDFs/Compression_Dynamic/arXiv_2023/2023_A_Compact_Dynamic_3D_Gaussian_Representation_for_Real_Time_Dynamic_View_Synthesis.pdf]]