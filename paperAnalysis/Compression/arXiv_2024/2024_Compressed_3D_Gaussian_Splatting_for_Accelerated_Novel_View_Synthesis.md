---
title: "Compressed 3D Gaussian Splatting for Accelerated Novel View Synthesis"
venue: arXiv
year: 2024
tags:
  - 3D_Gaussian_Splatting
  - task/novel-view-synthesis
  - vector-quantization
  - quantization-aware-training
  - hardware-rasterization
  - dataset/Mip-NeRF360
  - "dataset/Tanks&Temples"
  - dataset/DeepBlending
  - dataset/NeRF-Synthetic
  - opensource/full
core_operator: "用敏感度感知码本压缩、量化感知微调与空间顺序熵编码，把3DGS的高冗余颜色/形状参数变成低比特共享表示，并配合硬件光栅化实现高速新视角渲染。"
primary_logic: |
  已优化的3D Gaussian场景 + 训练图像/相机 → 按参数敏感度对SH颜色与归一化形状做向量聚类，并在低比特约束下量化感知微调，再按Morton顺序进行熵编码与硬件光栅化渲染 → 更小、更快、可在低显存设备上运行的3DGS新视角合成表示
claims:
  - "Across Mip-NeRF360, Tanks&Temples, and Deep Blending, the method reduces dataset-average scene size from hundreds of MB to 17–29 MB while keeping benchmark-average PSNR within 0.23 of the 3DGS baseline on each benchmark [evidence: comparison]"
  - "On the Bicycle scene at 1080p, the compressed renderer reaches 321 FPS on RTX A5000 and 211 FPS on RTX 3070M versus 93 FPS and 54 FPS for the original 3DGS compute renderer [evidence: comparison]"
  - "On Garden, quantization-aware fine-tuning improves PSNR from 25.781 after clustering to 26.746, and final Morton-order encoding further reduces size from 86.69 MB to 46.57 MB without additional quality loss [evidence: ablation]"
related_work_position:
  extends: "3D Gaussian Splatting (Kerbl et al. 2023)"
  competes_with: "Compressing Volumetric Radiance Fields to 1 MB (Li et al. 2023); MERF (Reiser et al. 2023)"
  complementary_to: "N/A"
evidence_strength: strong
pdf_ref: paperPDFs/Compression/arXiv_2024/2024_Compressed_3D_Gaussian_Splatting_for_Accelerated_Novel_View_Synthesis.pdf
category: 3D_Gaussian_Splatting
---

# Compressed 3D Gaussian Splatting for Accelerated Novel View Synthesis

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2401.02436), [Code](https://github.com/KeKsBoTer/c3dgs)
> - **Summary**: 这篇工作把原始3DGS中“每个高斯独立全精度存储”的场景表示改成“按敏感度分配比特的共享码本表示”，从而在几乎不损伤画质的前提下显著降低存储并提升渲染速度。
> - **Key Performance**: 真实场景平均约26×压缩、单场景最高约31×；1080p 下最高 321 FPS，相比原始 3DGS 计算管线最高约 4× 加速。

> [!info] **Agent Summary**
> - **task_path**: 已训练3DGS场景/训练图像与相机 -> 压缩3D高斯码本表示 -> 新视角RGB实时渲染
> - **bottleneck**: 3DGS的SH颜色与高斯形状参数占用大且冗余高，显存/带宽与专用compute渲染流程共同限制了流式传输和低功耗部署
> - **mechanism_delta**: 用敏感度加权的共享码本替代大部分逐高斯全精度参数，并让量化误差在微调阶段被模型吸收
> - **evidence_signal**: 多数据集压缩对比加消融共同显示，大幅降存储后PSNR几乎不变，且硬件光栅化把FPS显著拉高
> - **reusable_ops**: [sensitivity-aware vector clustering, quantization-aware finetuning]
> - **failure_modes**: [position quantization introduces visible errors, aggressive pruning removes thin details such as leaves]
> - **open_questions**: [how to compress Gaussian positions beyond 16-bit without quality collapse, can 3DGS be trained end-to-end for compression-friendly structure]

## Part I：问题与挑战

### 1. 论文要解决的真问题是什么？
3D Gaussian Splatting 已经把新视角合成从“重网络、慢渲染”的 NeRF 路线推进到“显式表示、实时渲染”的阶段，但它在部署侧还有两个核心障碍：

1. **场景太大**：一个场景往往包含数百万个高斯，SH 颜色、形状、位置等参数总量很大，常到数百 MB 甚至 GB。
2. **渲染链路不友好**：原始 3DGS 依赖专门的 GPU compute 渲染管线，难与游戏/VR/AR 常见的硬件光栅化流程无缝融合。

所以这篇论文不是单纯提高重建精度，而是回答一个更工程化、也更关键的问题：

**如何把高质量 3DGS 变成真正可传输、可压缩、可在低显存设备上跑的场景格式。**

### 2. 真正瓶颈在哪里？
作者的判断很准确，瓶颈不是“所有参数都太多”，而是三类结构性问题：

- **重要性不均匀**：并非每个高斯、每个参数都同样影响图像质量。
- **颜色和形状高度冗余**：SH 系数与高斯形状在场景内存在明显重复模式。
- **渲染受带宽约束**：即使算子本身不复杂，大量参数读写也会让渲染变成 memory-bound。

这意味着最有效的思路不是统一降精度，而是做**重要性感知的非均匀压缩**。

### 3. 输入/输出接口与边界条件
这项方法的输入不是原始多视图图像直接到结果，而是：

- 一个已经训练好的 **3DGS 静态场景**；
- 训练图像与相机参数，用于计算参数敏感度和做压缩后微调。

输出是：

- 压缩后的 3D Gaussian 表示；
- 一个支持 GPU 排序和硬件光栅化的新视角渲染器。

**边界条件**也很清楚：

- 面向静态场景；
- 假设相机已知；
- 依赖先得到一个高质量 3DGS 重建；
- 主要解决部署/推理侧压缩，不是从零开始的端到端重建方案。

## Part II：方法与洞察

### 1. 方法总览
整条方法链可以概括为四步：

1. **敏感度感知向量聚类**：压缩 SH 颜色和高斯形状。
2. **量化感知微调**：让低比特表示重新适配训练图像。
3. **Morton 顺序 + 熵编码**：继续利用空间局部一致性压缩文件大小。
4. **硬件光栅化渲染**：把压缩后的表示转成更易部署的渲染路径。

### 2. 敏感度感知聚类：把“误差预算”花在关键处
作者先用训练图像上的梯度大小估计每个参数的**敏感度**。直观上：

- 梯度大：这个参数稍微改动就会影响输出图像；
- 梯度小：这个参数更适合被共享码本近似。

于是，作者不是做普通 k-Means，而是做**敏感度加权的聚类**：

- 高敏感参数的聚类误差代价更高；
- 低敏感参数更容易共享同一个中心。

这一步真正改变的是**失真分配机制**：  
从“所有高斯平均受损”，变成“让不重要的高斯承担更多压缩误差”。

#### 颜色压缩
每个高斯的视角相关颜色由 SH 系数表示，这部分既大又敏感。  
作者为 SH 向量单独建立码本，并设置敏感度阈值：

- 高于阈值的少量高敏感 SH 向量不参与普通聚类；
- 这些“关键颜色”被直接加入码本，避免被粗暴合并。

#### 形状压缩
高斯形状由旋转和缩放决定。作者观察到很多高斯形状本质相似，只是**尺度不同**。  
所以他们把形状拆成：

- **归一化形状**：用于聚类，学习形状原型；
- **标量尺度因子**：每个高斯单独保存。

这一步非常关键，因为它把“同形不同尺度”的分布对齐了，让码本复用更有效。

### 3. 量化感知微调：让低比特成为训练约束，而不是测试时灾难
聚类后，作者继续做 **quantization-aware fine-tuning**：

- 前向模拟低比特量化；
- 反向仍用稳定的全精度梯度更新。

被微调的包括：

- 每个高斯的位置、透明度、尺度因子；
- 颜色码本；
- 形状码本。

最终大多数参数可压到 **8-bit**，但**位置仍需 16-bit**，因为位置压得太狠会明显伤害画质。

这一步的因果作用在于：  
它把量化误差从“部署时突然出现的近似误差”，变成“训练时模型已经适配过的结构约束”。

### 4. Morton 顺序 + 熵编码：先改布局，再让压缩器更聪明
仅做聚类与量化还不够，作者进一步把高斯按 **Morton order / Z-order curve** 排序，然后用 DEFLATE 做熵编码。

为什么这有效？

- 空间相邻的高斯更可能共享相近颜色、尺度和码本索引；
- 一旦排序后相似值在一维序列上聚集，运行长度编码和哈夫曼编码就更容易获益。

这本质上是一个系统优化：  
**不是改变内容，而是改变数据布局，让通用编码器能看到更多统计结构。**

### 5. 渲染器：把 3DGS 拉回标准图形管线
论文不只压缩，也重构了渲染路径：

- 预处理阶段剔除不可见高斯、计算 SH 颜色、投影到屏幕空间；
- 用 GPU 上的 Onesweep 排序做深度排序；
- 每个高斯渲染成一个屏幕空间 quad/splat，交给硬件光栅化。

这里有两个直接收益：

1. **压缩后内存带宽需求下降**，预处理更快；
2. **硬件光栅化更符合游戏/浏览器/VR 图形栈**，部署摩擦更小。

### 核心直觉
原始 3DGS 的默认假设是：

> 每个高斯都值得用独立的全精度参数来表达。

这篇论文把它改成了三个更贴近数据分布的先验：

1. **重要性不均匀**：少数参数比大多数参数更关键；
2. **形状/颜色有原型复用**：很多高斯可以共享表示；
3. **空间局部一致性可编码**：邻近高斯的统计结构可以进一步压缩。

所以能力变化链条是：

**逐高斯全精度存储**  
→ **敏感参数保真、冗余参数共享、低比特量化可适配**  
→ **显存与带宽压力下降**  
→ **低端设备和硬件光栅化部署变得可行**。

这也是本文真正的“causal knob”：

- **敏感度权重**改变了误差该分配给谁；
- **归一化形状分解**改变了形状参数的聚类几何；
- **QAT**改变了量化误差出现的位置；
- **Morton 排列**改变了编码器看到的数据统计；
- **硬件光栅化**改变了渲染瓶颈所在的硬件路径。

### 战略权衡
| 设计选择 | 解决的瓶颈 | 直接收益 | 代价/权衡 |
|---|---|---|---|
| 敏感度感知颜色/形状聚类 | 统一量化先伤关键参数 | 在高压缩率下保持画质稳定 | 阈值越保守，压缩率越低 |
| 归一化形状 + 单独尺度因子 | 同形不同尺度难共享 | 提高形状码本复用率 | 仍需保存每个高斯的尺度因子 |
| 量化感知微调 | 聚类后质量难恢复 | 回收 PSNR/SSIM，并允许低比特存储 | 需要训练图像和额外优化时间 |
| Morton 排序 + 熵编码 | 量化后文件仍有冗余 | 基本不掉质地继续缩小 | 更偏离线压缩，不适合频繁在线更新 |
| 硬件光栅化 | 原始 compute 渲染难集成、带宽重 | 更高 FPS，更易接入图形系统 | 仍依赖 GPU 排序与 alpha 混合支持 |

## Part III：证据与局限

### 1. 关键实验信号
**信号 A｜多数据集压缩对比：从“百MB级”压到“十MB级”，质量基本守住**  
在 Mip-NeRF360、Tanks&Temples、Deep Blending 上，数据集平均场景大小分别从 **795.26 / 421.90 / 703.77 MB** 降到 **28.80 / 17.28 / 25.30 MB**；对应 PSNR 为 **26.981 / 23.324 / 29.381**，与 3DGS 基线非常接近。  
这说明被拿掉的主要是冗余，而不是关键表示能力。  
**信号类型**：comparison  
**结论**：压缩收益是跨数据集成立的，不是单场景偶然。

**信号 B｜渲染速度对比：不仅省空间，还明显解了带宽瓶颈**  
Bicycle 场景在 1080p 下，RTX A5000 从 **93 FPS** 提升到 **321 FPS**，RTX 3070M 从 **54 FPS** 提升到 **211 FPS**。  
同时在 Intel UHD Graphics 11 和 AMD Radeon R9 380 上，作者的渲染器仍能运行。  
**信号类型**：comparison  
**结论**：压缩表示 + 光栅化渲染确实带来更强的部署可用性。

**信号 C｜消融：真正起作用的是“三段式叠加”，不是单一技巧**  
Garden 场景中：

- 仅做聚类后，大小从 **1379.99 MB** 降到 **164.15 MB**，但 PSNR 降到 **25.781**；
- 加入 QAT 后，PSNR 回升到 **26.746**，大小继续降到 **86.69 MB**；
- 再做熵编码与 Morton 排序，大小进一步降到 **46.57 MB**，PSNR 不再继续下降。

**信号类型**：ablation  
**结论**：码本压缩负责主压缩率，QAT 负责质量恢复，布局编码负责最后的无损型收缩。

### 2. 1-2 个最值得记住的指标
- **压缩率**：真实场景平均约 **26×**，单场景最高约 **31×**。
- **速度**：1080p 下最高 **321 FPS**，相对原始 3DGS 计算管线最高约 **4×** 加速。

### 3. 这些证据说明了什么能力跃迁？
相对 prior work，本文的跃迁不在“更高 PSNR”，而在于：

- 把 3DGS 从一种**高质量但偏重的研究表示**，
- 变成一种**更像可部署场景格式**的表示。

也就是说，能力跳跃体现在：
**可流式传输、可低显存运行、可接入硬件光栅化图形栈**。

最支持这一点的实验不是单一画质表，而是：

1. 多数据集的压缩后画质保持；
2. 跨 GPU 的 FPS 提升；
3. 消融中 QAT 与空间编码的明确贡献。

### 4. 局限性
- Fails when: 试图进一步激进压缩 **高斯位置** 时；作者明确表示位置量化会明显损伤渲染质量，另外过强 pruning 会删掉叶片等细薄结构。
- Assumes: 已有高质量 **静态** 3DGS 重建；可以访问训练图像与相机参数来做敏感度估计和压缩后微调；依赖 GPU 排序与硬件光栅化能力。
- Not designed for: 动态场景、未知位姿输入、纯模型文件级“无数据后处理压缩”，以及从原始视频直接端到端学得可压缩表示。

### 5. 复现与扩展时要注意的资源/依赖
- **训练数据依赖**：最佳结果并不是只拿到 `.ply` 或参数文件即可复现，还需要训练图像参与敏感度计算和 QAT。
- **额外时间成本**：压缩过程约 **5-6 分钟**，其中约 **70% 时间**花在 QAT 微调。
- **剩余主要开销**：压缩后位置与码本索引仍是重要内存占用来源，这也解释了为什么进一步压缩很难。
- **工程依赖**：实现基于 WebGPU/Rust，代码已开源，但运行仍需要支持图形 API 与 GPU 排序。

### 6. 可复用组件
- **sensitivity-aware codebook learning**：适合任何“参数多、重要性分布极不均匀”的显式表示。
- **normalized-shape factorization**：适合存在“原型相似但尺度不同”的几何参数。
- **QAT for explicit scene representations**：不仅适用于网络权重，也适用于显式神经场/显式辐射场参数。
- **space-filling reordering + entropy coding**：是低成本但高回报的系统级压缩套路。
- **hardware-rasterized splat renderer**：对要接入传统图形栈的 3DGS 系统尤其有价值。

![[paperPDFs/Compression/arXiv_2024/2024_Compressed_3D_Gaussian_Splatting_for_Accelerated_Novel_View_Synthesis.pdf]]