---
title: "4D Gaussian Splatting: Towards Efficient Novel View Synthesis for Dynamic Scenes"
venue: SIGGRAPH
year: 2024
tags:
  - "3D_Gaussian_Splatting"
  - "task/video-generation"
  - "gaussian-splatting"
  - "temporal-slicing"
  - "rotor-parameterization"
  - "dataset/Plenoptic Video Dataset"
  - "dataset/D-NeRF Dataset"
  - "opensource/full"
core_operator: "用可时间切片的4D XYZT各向异性高斯显式表示动态场景，在查询时刻切成3D高斯并直接splat到图像平面"
primary_logic: |
  带位姿的动态视频与时间查询 → 用4D高斯与rotor参数化时空分布，并在目标时刻做时间切片得到当前3D高斯及其局部运动 → 通过高斯splatting光栅化输出新视角图像
claims:
  - "Claim 1: 在 Plenoptic Video Dataset 上，4DRotorGS 取得 31.62 PSNR、0.94 SSIM 和 277.47 FPS，超过 HyperReel、MixVoxels 与 RealTime4DGS 等基线 [evidence: comparison]"
  - "Claim 2: 在 D-NeRF 上，4DRotorGS 达到 1257.63 FPS，并以 34.26 PSNR 超过 TiNeuVox、Deformable4DGS 和 RealTime4DGS，但低于 Deformable3DGS 的 39.31 PSNR [evidence: comparison]"
  - "Claim 3: 熵正则、4D一致性正则与 batch training 能稳定优化：熵项将点数从 1.7e5 降到 5.4e4，完整配置把 D-NeRF 消融 PSNR 提升到 33.06 [evidence: ablation]"
related_work_position:
  extends: "3D Gaussian Splatting (Kerbl et al. 2023)"
  competes_with: "RealTime4DGS (Yang et al. 2024); Deformable4DGS (Wu et al. 2023)"
  complementary_to: "Total-Recon (Song et al. 2023b); SuGaR (Guédon and Lepetit 2023)"
evidence_strength: strong
pdf_ref: "paperPDFs/Dynamic_Video/Special_Interest_Group_on_Computer_Graphics_and_Interactive_Techniques_Conference_Conference_Papers_24_2024/2024_4D_Gaussian_Splatting_Towards_Efficient_Novel_View_Synthesis_for_Dynamic_Scenes.pdf"
category: 3D_Gaussian_Splatting
---

# 4D Gaussian Splatting: Towards Efficient Novel View Synthesis for Dynamic Scenes

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2402.03307), [Code](https://github.com/weify627/4D-Rotor-Gaussians), [DOI](https://doi.org/10.1145/3641519.3657463)
> - **Summary**: 论文把动态新视角合成从“canonical space + deformation”的隐式时变建模，改成“可按时间切片的显式 4D 高斯”，从而同时提升了突发运动建模能力和实时渲染速度。
> - **Key Performance**: Plenoptic 上 31.62 PSNR @ 277 FPS（RTX 3090）；D-NeRF 上 34.26 PSNR @ 1257.63 FPS

> [!info] **Agent Summary**
> - **task_path**: 带位姿的单目/多视角动态视频 + 查询时间/相机位姿 -> 目标时刻的新视角RGB图像
> - **bottleneck**: 动态场景的时空耦合过强，canonical deformation 难处理突发出现/消失，而体渲染又难以达到实时
> - **mechanism_delta**: 用 4D XYZT 高斯在时间维做切片，直接得到当前 3D 高斯及其局部线性运动，再沿用 3DGS 的 splatting 渲染
> - **evidence_signal**: 双数据集比较 + 消融；Plenoptic 上同时拿到最高 PSNR 与显著更高 FPS
> - **reusable_ops**: [temporal-slicing-of-4d-gaussians, 4d-motion-consistency-regularization]
> - **failure_modes**: [极稀疏视角且快速大运动时会出现floaters与模糊, 透明物体较多时熵正则可能伤害重建]
> - **open_questions**: [如何更强地约束4D高斯而不牺牲速度, 如何结合depth/flow/mask监督提升超稀疏动态场景表现]

## Part I：问题与挑战

这篇论文的方法名在文中写作 **4DRotorGS**。它解决的是 **动态场景新视角合成**：输入是带相机位姿的动态视频，既可以是多视角真实视频，也可以是单目随时间变化的视频；输出是在任意给定时间和新相机位姿下的渲染图像。

### 真正的难点是什么？

真正的瓶颈不只是“多了一个时间维”，而是两层问题叠加：

1. **表示瓶颈**  
   以往动态 NeRF 常用 `canonical space + deformation field`。这类方法默认动态内容能被一个较平滑、可追踪的变形场解释。  
   但一旦出现：
   - 突然出现/消失的物体或火焰
   - 遮挡变化
   - 高频局部细节 + 快速运动  
   这个假设就会变脆弱。

2. **渲染瓶颈**  
   即使建模成功，很多方法仍依赖体渲染，对大量射线做密集采样，速度难以上实时。

### 为什么现在值得做？

因为 **3D Gaussian Splatting** 已经证明：  
对静态场景，只要表示足够显式、渲染足够 rasterization-friendly，就能同时做到高质量和实时速度。

所以这篇论文的核心问题变成：

> 能不能把 3DGS 的“显式高斯 + 快速 splatting”范式，扩展到动态 4D 场景，同时不退回到慢速体渲染？

### 输入/输出与边界条件

- **输入**：带相机位姿的动态 RGB 视频
- **查询**：任意时间戳 + 新相机位姿
- **输出**：该时刻的新视角 RGB 图像
- **边界**：
  - 依赖已知位姿
  - 主要是逐场景优化，不是跨场景泛化模型
  - 局部运动默认可由短时间窗口内的线性近似覆盖；更复杂轨迹需要多个高斯共同拟合

---

## Part II：方法与洞察

方法主线很清楚：**不要再学“点怎么从 canonical 空间被扭过去”，而是直接把动态场景表示成 4D 时空高斯本体。**

### 核心直觉

把动态信息塞进一个 **4D XYZT 高斯** 里，然后在查询时刻用一个时间超平面去“切”它。

这带来三个关键变化：

1. **表示变化**：  
   从“3D 几何 + 时间形变函数”  
   变成“时空一体的 4D 显式原语”。

2. **约束变化**：  
   每个高斯天然带有时间范围和时空协方差。  
   这意味着“出现/消失”不再需要 deformation field 去硬解释，而是高斯在时间切片下自然变强或变弱。

3. **能力变化**：  
   - 更适合表示突发运动与拓扑变化
   - 保留 3DGS 的显式 rasterization 路线，因此推理速度极快

更具体地说，时间切片后，每个 4D 高斯会产生两个很重要的效果：

- **时间衰减项**：决定它在当前时刻是否可见  
  → 让“突然出现/消失”变成表示的一部分
- **中心偏移项**：给出随时间变化的 3D 中心  
  → 等价于局部线性运动，因此还能“免费”得到速度场/光流线索

### 关键设计

#### 1. 4D 高斯表示：把动态编码进原语本身
每个高斯不再只在 XYZ 空间里有中心和协方差，而是扩展到 **XYZT**。  
这样一个高斯不是“某一帧的点”，而是“一个时空局部事件”。

#### 2. Rotor 参数化 4D 旋转
作者没有直接用更常见的 dual-quaternion，而是用 **4D rotor** 表示 4D 旋转。其意义主要有两点：

- **空间旋转和时空旋转可解释地分开**
- **当时间相关分量置零时，可退化回 quaternion**
  → 于是同一套框架既能建静态 3D，也能建动态 4D

这让它成为 3DGS 的一个“严格扩展版”。  
但要注意：作者在补充实验里也承认，**rotor 相比 dual-quaternion 并没有明显 PSNR 优势**。  
所以这里真正造成能力跃迁的，不是 rotor 单独本身，而是：

- 4D 显式表示
- 时间切片
- 正则化
- CUDA 高性能实现

#### 3. 两个正则：给 4D 自由度“上缰绳”
4D 表示更强，但自由度也更大，训练会更容易飘。

作者加了两类稳定器：

- **Entropy loss**：把 opacity 往 0/1 推  
  作用：减少 floaters，压缩无用高斯  
  但对透明物体较多的数据不一定友好

- **4D consistency loss**：约束 4D 邻域内高斯的运动一致  
  作用：减少运动噪声，让局部 motion field 更平滑  
  关键点在于它找的是 **4D 近邻** 而不是 3D 近邻，因为 3D 上相近的点未必属于同一运动

#### 4. CUDA 实现：把 4D slicing 真正做成实时系统
作者把 rotor 变换、时间切片、复制/裁剪等都写进 CUDA。  
因此它不是“理论上可快”，而是工程上也真的快：
- RTX 3090：Plenoptic 上 277 FPS
- RTX 4090：高分辨率可到 583 FPS
- 相比 PyTorch 版，训练加速约 16.6x

### 战略权衡

| 设计选择 | 改变了什么瓶颈 | 能力收益 | 代价/风险 |
|---|---|---|---|
| 4D 高斯 + 时间切片 | 去掉对全局 deformation field 的依赖 | 支持突发出现/消失，且仍可快速 splat | 单个高斯只建模局部近似线性运动，复杂非线性轨迹要靠更多高斯 |
| rotor 4D 旋转 | 让时空旋转可分解，并兼容 3DGS | 静态/动态统一，解释性更强 | 补充实验未证明其量化性能显著优于 dual-quaternion |
| entropy loss | 收缩不确定 opacity | 减少 floaters、压缩点数 | 透明物体多时可能伤效果 |
| 4D consistency loss | 限制 4D 邻域运动自由度 | motion field 更稳，细节更清晰 | 需要额外近邻约束与训练开销 |
| CUDA sliced splatting | 去掉框架开销 | 实时渲染与更快训练 | 复现依赖定制 CUDA kernel 与较强 GPU |

---

## Part III：证据与局限

### 关键证据信号

#### 证据 1：Plenoptic 上，质量/速度 Pareto 前沿明显前移
- **信号类型**：comparison
- **结论**：在真实多视角动态视频、1352×1014 分辨率下，方法同时拿到更高 PSNR 和远高于基线的 FPS。
- **关键信号**：
  - Ours: **31.62 PSNR / 277.47 FPS**
  - MixVoxels: 30.85 / 16.70 FPS
  - RealTime4DGS: 29.95 / 72.80 FPS

这说明 4D 显式高斯 + 时间切片不是只换了表示名字，而是真的把“动态质量”和“渲染速度”一起推高了。

#### 证据 2：D-NeRF 上，速度优势极大，但不是所有场景都绝对最优 PSNR
- **信号类型**：comparison
- **结论**：在单目动态场景上，它的实时性优势非常夸张，但在某些更适合直接形变跟踪的数据分布上，PSNR 不是全场最佳。
- **关键信号**：
  - Ours: **34.26 PSNR / 1257.63 FPS**
  - RealTime4DGS: 32.71 / 289.07 FPS
  - Deformable4DGS: 32.99 / 104.00 FPS
  - Deformable3DGS: **39.31 PSNR**（但速度 85.45 FPS）

这个对比很重要：  
**4DRotorGS 的优势是“更通用的动态显式表示 + 极快速度”**，而不是在所有平滑单目运动上都无条件刷最高 PSNR。

#### 证据 3：消融验证了“稳不稳”主要来自正则和 batch 训练
- **信号类型**：ablation
- **结论**：
  - 熵正则明显减少 floaters，并把点数从 **1.7e5 降到 5.4e4**
  - 4D consistency 让运动更平滑、细节更好
  - batch training 进一步稳定稀疏视角下的几何一致性

尤其值得注意的是：论文自己也显示了光流可视化，加入 4D consistency 后运动场噪声显著下降。  
这说明它不只是“渲染看起来更好”，而是 **时空结构本身变得更一致**。

#### 证据 4：极难数据上仍有边界
- **信号类型**：comparison / case-study
- **结论**：在 Total-Recon 这种超稀疏、快速大运动场景中，虽然优于 RGB-only 基线，但仍会明显退化。
- 这给了很清晰的能力边界：方法虽然强，但还不是“任意动态场景都稳”。

### 局限性

- **Fails when**: 视角极稀疏、运动幅度很大、相机也快速跟拍时，4D 高斯会难以被充分约束，容易出现 floaters、模糊和错误结构；Total-Recon 的例子最明显。
- **Assumes**: 已知相机位姿、逐场景优化、局部运动可在短时间窗内近似为线性；高性能结果依赖定制 CUDA 实现和 RTX 级 GPU；Plenoptic 训练还依赖 COLMAP 初始化；熵正则在透明物体多的场景中需要关闭或谨慎使用。
- **Not designed for**: 跨场景泛化、开放域视频生成、长时程动力学预测、以及无需 per-scene optimization 的在线通用模型。

### 可复用组件

- **4D temporal slicing**：适合任何想把时空显式原语投影回当前 3D 状态的表示
- **4D consistency regularization**：可迁移到其他动态点云/高斯/显式时空表示
- **speed field derivation**：时间切片自然给出局部速度，可用于光流、跟踪或动态分析
- **CUDA splatting pipeline**：对其他显式动态表示也有工程复用价值

### 一句话总结

这篇工作的真正价值，不是“把 3DGS 简单加一个时间维”，而是把 **动态场景表示** 从“通过隐式变形去解释时间”改成“时间就是高斯本体的一部分”，于是同时解决了两件事：

1. 动态拓扑变化更自然  
2. 渲染仍保留 3DGS 的实时速度路径

如果你关心的是 **动态 NVS 的速度-质量平衡**，这篇是很关键的一步；  
如果你关心的是 **极端稀疏视角下的最强重建**，它还需要更强监督或更强约束。

![[paperPDFs/Dynamic_Video/Special_Interest_Group_on_Computer_Graphics_and_Interactive_Techniques_Conference_Conference_Papers_24_2024/2024_4D_Gaussian_Splatting_Towards_Efficient_Novel_View_Synthesis_for_Dynamic_Scenes.pdf]]