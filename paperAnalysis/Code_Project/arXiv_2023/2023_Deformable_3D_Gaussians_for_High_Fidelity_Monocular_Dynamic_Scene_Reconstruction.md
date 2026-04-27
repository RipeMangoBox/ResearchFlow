---
title: "Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction"
venue: arXiv
year: 2023
tags:
  - 3D_Gaussian_Splatting
  - task/dynamic-scene-reconstruction
  - task/novel-view-synthesis
  - gaussian-splatting
  - deformation-field
  - annealing-smoothing
  - dataset/D-NeRF
  - dataset/NeRF-DS
  - dataset/HyperNeRF
  - opensource/full
core_operator: 在 canonical space 中学习可变形 3D Gaussians，并通过退火式时间平滑训练抑制位姿误差带来的时序抖动。
primary_logic: |
  单目多视角图像 + 时间标签 + SfM位姿/稀疏点云 → 初始化 canonical 3D Gaussians，用时间条件变形 MLP 预测位置/旋转/尺度偏移，并通过可微 Gaussian rasterizer 与自适应密度控制联合优化，同时在训练早期对时间编码注入退火噪声 → 输出支持新视角合成与时间插值、且可实时渲染的动态场景表示
claims:
  - "在 D-NeRF 合成数据集的 8 个场景上，该方法在 PSNR、SSIM、LPIPS 三项指标上均优于文中所有对比基线 [evidence: comparison]"
  - "在 NeRF-DS 七场景均值上，加入 AST 后达到 24.11 PSNR、0.8524 SSIM、0.1769 LPIPS，优于无 AST 版本及所有对比方法 [evidence: ablation]"
  - "当 3D Gaussians 数量低于约 250k 时，该方法在 RTX 3090 上可实现超过 30 FPS 的实时渲染 [evidence: analysis]"
related_work_position:
  extends: "3D Gaussian Splatting (Kerbl et al. 2023)"
  competes_with: "HyperNeRF (Park et al. 2021); K-Planes (Fridovich-Keil et al. 2023)"
  complementary_to: "BARF (Lin et al. 2021)"
evidence_strength: strong
pdf_ref: paperPDFs/Code_Project/arXiv_2023/2023_Deformable_3D_Gaussians_for_High_Fidelity_Monocular_Dynamic_Scene_Reconstruction.pdf
category: 3D_Gaussian_Splatting
---

# Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2309.13101), [Code](https://github.com/ingra14m/Deformable-3D-Gaussians)
> - **Summary**: 这篇工作把静态 3D Gaussian Splatting 扩展到单目动态场景：用 canonical-space 高斯 + 时间条件变形场建模运动，并用 AST 缓解真实数据中的位姿噪声，从而同时获得更清晰的细节、更平滑的时间插值和接近实时的渲染。
> - **Key Performance**: D-NeRF 上 8/8 场景量化全胜；NeRF-DS 均值达到 24.11 PSNR / 0.8524 SSIM / 0.1769 LPIPS，且高斯数 <250k 时可 >30 FPS。

> [!info] **Agent Summary**
> - **task_path**: 单目多视角图像 + 时间戳 + 相机位姿 -> 动态场景 canonical 3D Gaussian 表示 -> 新视角/时间插值渲染
> - **bottleneck**: 隐式动态 NeRF 很难兼顾高频细节与实时性，而显式 3DGS 在动态单目场景里又会被位姿误差放大成时间抖动
> - **mechanism_delta**: 将场景改写为“共享 canonical 高斯 + 随时间变化的偏移场”，并在训练早期对时间编码加退火噪声，避免模型记忆化 pose noise
> - **evidence_signal**: D-NeRF 8/8 场景定量领先，NeRF-DS 均值优于全部基线且 AST 消融带来稳定增益
> - **reusable_ops**: [canonical-space deformation field, annealed time-noise smoothing]
> - **failure_modes**: [位姿估计不准时出现抖动甚至不收敛, 训练视角过少时过拟合并导致高斯数量膨胀]
> - **open_questions**: [能否联合相机位姿优化, 能否稳定处理细粒度面部与复杂人体运动]

## Part I：问题与挑战

这篇论文要解决的是：**给定单目动态场景的一组图像、时间标签和 SfM 相机位姿，重建一个连续时间的动态 3D 表示**，使它既能做新视角合成，也能做时间插值，而且最好还能实时渲染。

### 真正瓶颈是什么？
真正的瓶颈不是“怎么给 NeRF 加一个时间变量”，而是如何同时满足三件事：

1. **细节保真**：动态场景里人物肢体、细杆、球体边界等高频结构容易被隐式场抹平。
2. **时间连续**：若按帧独立建模，再做后处理插值，单目场景很容易失去时空连续性。
3. **渲染效率**：动态 NeRF 大多仍依赖 ray marching，训练和推理都慢，很难实时。

作者的判断是：**静态 3DGS 已经证明“显式高斯 + rasterization”能同时要质量和速度，但它天生只适合静态场景。**  
因此现在值得解决的问题变成：**如何把 3DGS 变成连续时间的动态表示，而不是退回到慢且容易过拟合的隐式动态场。**

### 输入/输出接口
- **输入**：单目动态图像序列、多视角/相机轨迹、时间标签、COLMAP/SfM 位姿与稀疏点云
- **输出**：一个在 canonical space 中定义、可随时间变形的 3D Gaussian 场景表示
- **支持任务**：新视角合成、时间插值、实时渲染

### 边界条件
这不是“任意视频都能稳”的方法。它默认：
- 位姿大体可用；
- 视角覆盖不能太差；
- 场景运动复杂度中等；
- 目标是**有相机位姿的单目动态重建**，不是无位姿、极少视角或超复杂面部运动建模。

## Part II：方法与洞察

方法主线很清楚：**把动态场景拆成“时间无关的 canonical 3D Gaussians”与“时间相关的 deformation field”。**

### 方法骨架
1. **初始化 canonical 3D Gaussians**  
   从 SfM 点云初始化一组 3D Gaussians，带位置、旋转、尺度、不透明度和 SH 外观。

2. **变形场建模动态**  
   给定高斯中心位置和时间，变形 MLP 输出：
   - 位置偏移 `δx`
   - 旋转偏移 `δr`
   - 尺度偏移 `δs`

3. **可微 Gaussian rasterization 联合优化**  
   把变形后的高斯送入 3DGS 的可微光栅化管线，直接从渲染误差反传到：
   - canonical 高斯参数
   - deformation MLP

4. **自适应密度控制**  
   在梯度大的区域 clone/split 高斯，在透明区域裁剪高斯，让动态细节区域逐渐被“加密”。

5. **训练策略**
   - 前 3k iter 只训高斯，不训变形场：先把 canonical 几何站稳
   - 后续联合训练
   - 在真实数据上引入 **AST（Annealing Smooth Training）**

### 核心直觉

作者做的关键变化，不是简单把 3DGS “加个 time”，而是同时改了**表示、优化、训练分布**三个层面。

#### 1) 从“每条射线积分隐式场”改成“显式高斯光栅化”
- **改变了什么**：从 ray marching 切到 point-based Gaussian rasterization
- **改变了哪个瓶颈**：计算瓶颈从高代价体渲染转成高效 splatting
- **带来什么能力**：实时渲染成为可能，且局部几何/纹理细节更容易保住

#### 2) 从“时间缠绕在辐射场里”改成“共享 canonical 模板 + 时间偏移”
- **改变了什么**：所有时刻共享同一套 canonical 高斯，运动只负责解释偏移
- **改变了哪个瓶颈**：把单目稀疏监督下的“每时刻都像重新建场景”变成“跨时间共享结构”
- **带来什么能力**：时间插值更自然，细节不会因逐帧拟合而飘掉

#### 3) 从“精确时间编码直接拟合”改成“早期带噪时间编码 + 后期退火”
- **改变了什么**：AST 在训练早期给时间编码加入逐渐衰减的高斯噪声
- **改变了哪个瓶颈**：缓解模型把 pose noise 当成真实动态去记忆
- **带来什么能力**：真实数据时间插值更平滑，同时后期仍能恢复高频细节

### 为什么这套设计会有效？
一个很关键的因果链是：

**显式高斯表示**让每个 primitive 都带有局部几何意义；  
**canonical-space 变形**让不同时间的监督共同塑造同一组高斯；  
**动态区域的渲染梯度**又会推动 density control 在这些区域分裂/复制更多高斯；  
于是模型不仅“会动”，还会在该细的地方变细。

另外，作者没有把变形场做成 grid/plane，而是保留了一个纯 MLP。理由也很直接：**动态场景的时空结构 rank 更高，低秩 grid/plane 假设未必合适；显式点渲染还会进一步放大这种高秩性。**

### 战略取舍

| 设计选择 | 解决的瓶颈 | 收益 | 代价/风险 |
|---|---|---|---|
| canonical 高斯 + deformation MLP | 几何与运动耦合 | 跨时间共享结构，时间插值更稳 | 假设场景能被较稳定的 canonical 模板描述 |
| Gaussian rasterization | ray marching 太慢 | 高质量 + 高 FPS | 对位姿误差更敏感 |
| AST | 真实数据 pose noise 导致时序抖动 | 无额外 loss/计算的平滑化 | 退火日程不合适时可能压制高频细节 |
| adaptive density control | 动态细节区域表示不足 | 动态局部会被自动加密 | 高斯数变多会抬高显存与渲染成本 |
| 直接偏移而非 SE(3) 约束 | 刚性约束过强/代价更高 | 实现简单，真实数据更稳 | 对严格物理变换没有显式结构先验 |

## Part III：证据与局限

### 关键证据

- **比较信号：D-NeRF 合成集**
  - 在 8 个场景上，方法在 PSNR / SSIM / LPIPS 三项指标全部领先。
  - 这说明能力提升不是单一指标上的“取巧”，而是画质、结构一致性和感知质量一起变好。
  - 代表性例子：如 Hell Warrior、Mutant、Stand Up 等场景，相比 D-NeRF、TiNeuVox、K-Planes 都有明显跃升。

- **比较 + 消融信号：NeRF-DS 真实集**
  - 七场景均值达到 **24.11 PSNR / 0.8524 SSIM / 0.1769 LPIPS**。
  - 比无 AST 版本更好，也超过 HyperNeRF 与 NeRF-DS。
  - 这直接支撑 AST 的核心论点：**它确实在真实 pose noise 下改善了时间泛化，而不是只在合成数据上有效。**

- **效率信号**
  - 当高斯数低于约 **250k** 时，可在 RTX 3090 上达到 **>30 FPS**。
  - 这表明该方法的能力跃迁不仅是“更清晰”，更是把**动态重建**和**实时渲染**放到同一套表示里。

- **附加设计证据**
  - 作者还测试了 SE(3) deformation field：合成集收益很小，真实集略降，同时训练时间约增加 50%，FPS 下降约 20%。
  - 这说明论文最终选择的“直接预测偏移”不是简化偷懒，而是一个有效的效率/效果折中。

### 局限性
- **Fails when**: 相机位姿明显不准时，显式高斯会把误差放大为跨帧 jitter，严重时甚至不收敛；训练视角过少或覆盖范围窄时，也容易过拟合并出现高斯数量暴涨。
- **Assumes**: 依赖 COLMAP/SfM 提供相对可靠的位姿与稀疏点云；依赖自定义 CUDA Gaussian rasterizer 与 RTX 级 GPU；默认场景运动复杂度中等，且可由共享 canonical 高斯加连续偏移描述。
- **Not designed for**: 无位姿重建、极端 few-view/few-shot 设置、超复杂细粒度人脸运动、严重拓扑变化或大规模动态场景的鲁棒建模。

### 可复用组件
- **canonical-space + deformation field**：把动态问题拆成共享几何与时间残差
- **AST**：对真实数据中的时间编码做退火式噪声平滑
- **3k warm-up**：先稳定高斯，再放开动态变形
- **gradient-driven densification**：利用动态区域的梯度自动加密高斯

**一句话评价**：这篇工作的真正价值，不只是“把 3DGS 用到动态场景”，而是证明了**显式高斯表示也能在单目动态重建里兼顾细节、时间连续性和实时性**；但前提是位姿质量和视角覆盖不能太差。

![[paperPDFs/Code_Project/arXiv_2023/2023_Deformable_3D_Gaussians_for_High_Fidelity_Monocular_Dynamic_Scene_Reconstruction.pdf]]