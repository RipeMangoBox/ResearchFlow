---
title: "Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion"
venue: arXiv
year: 2024
tags:
  - 3D_Gaussian_Splatting
  - task/novel-view-synthesis
  - task/3d-reconstruction
  - screen-space-approximation
  - pose-optimization
  - visual-inertial-odometry
  - dataset/Deblur-NeRF
  - opensource/partial
core_operator: "用 VIO 初始化的相机线速度与角速度把单帧曝光建模为短时连续轨迹，并在屏幕空间以像素速度近似高斯中心运动，从而在 3DGS 中联合补偿运动模糊与滚动快门。"
primary_logic: |
  手持视频帧 + IMU/VIO 速度先验 + 初始相机位姿 → 将每帧从“单一静态位姿”改写为“曝光期内带逐行时间偏移的连续相机轨迹”，并在像素空间对高斯中心做多样本模糊/滚快门渲染与位姿-速度联合优化 → 输出更清晰、更几何一致的 3DGS 场景与新视角图像
claims:
  - "在 BAD-NeRF 重渲染的 Deblur-NeRF 基准上，该方法取得最优平均指标（PSNR≈29.87、SSIM≈0.925、LPIPS≈0.062），优于 BAD-NeRF 与 Deblur-NeRF [evidence: comparison]"
  - "在作者重渲染的受控合成实验中，该方法对滚动快门和位姿噪声尤其有效，例如 Cozyroom 的 PSNR 在滚动快门设置下由 19.21 提升到 35.84，在位姿噪声设置下由 16.76 提升到 36.30 [evidence: comparison]"
  - "在 11 个智能手机序列上，完整系统平均 PSNR 为 28.59，高于 Splatfacto 的 26.91；去掉运动模糊补偿、滚动快门补偿、位姿优化、速度优化或 VIO 初始化任一组件都会降低平均性能 [evidence: ablation]"
related_work_position:
  extends: "3D Gaussian Splatting (Kerbl et al. 2023)"
  competes_with: "BAD-NeRF (Wang et al. 2023); Deblur-NeRF (Ma et al. 2022)"
  complementary_to: "Mip-Splatting (Yu et al. 2024); AbsGS (Ye et al. 2024)"
evidence_strength: strong
pdf_ref: paperPDFs/Deblurring_Rendering_Video/Lecture_Notes_in_Computer_Science_2025/2025_Gaussian_Splatting_on_the_Move_Blur_and_Rolling_Shutter_Compensation_for_Natural_Camera_Motion.pdf
category: 3D_Gaussian_Splatting
---

# Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2403.13327), [Code](https://github.com/SpectacularAI/3dgs-deblur), [Smartphone Dataset](https://doi.org/10.5281/zenodo.10848124), [Deblur-NeRF Variant](https://doi.org/10.5281/zenodo.10847884)
> - **Summary**: 该文把手持相机在曝光期内的连续运动直接并入 3DGS 渲染过程，利用 VIO 速度先验与屏幕空间像素速度近似，同时补偿运动模糊和滚动快门，从而让手持手机视频也能训练出更清晰的静态场景 3DGS。
> - **Key Performance**: 在 BAD-NeRF 版 Deblur-NeRF 上平均 PSNR/SSIM/LPIPS 约为 **29.87 / 0.925 / 0.062**；在 11 个手机序列上平均 PSNR **28.59 vs 26.91**（Splatfacto）。

> [!info] **Agent Summary**
> - **task_path**: 手持视频帧 + IMU/VIO 先验 -> 静态场景 3DGS 重建与新视角合成
> - **bottleneck**: 3DGS 默认假设“每帧=单一静态位姿”，无法解释曝光期内连续相机运动带来的运动模糊与逐行读出带来的滚动快门失真
> - **mechanism_delta**: 把每帧位姿扩展为带线速度/角速度的短时轨迹，并在屏幕空间只对 Gaussian 均值施加像素速度近似，以低开销实现曝光积分
> - **evidence_signal**: 合成比较 + 真实手机消融都表明完整模型优于基线，且在 BAD-NeRF 基准平均指标领先
> - **reusable_ops**: [screen-space pixel-velocity rendering, VIO-initialized pose/velocity refinement]
> - **failure_modes**: [pose optimization may fall into local minima, screen-space linearization may weaken under extreme fast motion or poor initialization]
> - **open_questions**: [can spline-based within-frame trajectories beat linear motion, how to extend from static scenes to dynamic scenes or non-camera-motion blur]

## Part I：问题与挑战

这篇论文真正要解决的，不只是“模糊图像去模糊”本身，而是 **3DGS 的前向成像假设与真实手机采集过程不匹配**。

### 1) 真问题是什么
传统 3D Gaussian Splatting 假设：
- 一张训练图像对应一个**瞬时、静态**相机位姿；
- 图像失真主要来自几何/外观建模误差，而不是相机在曝光过程中的连续运动。

但手持手机视频恰恰违反了这两个前提：
- **运动模糊**：曝光时间内相机在动，单帧其实是沿一小段轨迹的时间积分；
- **滚动快门**：CMOS 逐行读出，不同行对应不同采样时刻，所以一帧内部本身就是“行时变位姿”。

如果仍把这类帧当作静态照片喂给 3DGS，优化会把这些时序失配错误地“吞进”：
- Gaussian 的形状/位置，
- 颜色纹理，
- 相机位姿本身。

结果就是重建变糊、边缘变形、姿态注册不稳，甚至 COLMAP 初始化失败。

### 2) 输入/输出接口
- **输入**：手持视频关键帧、相机内参、初始位姿、曝光/滚动快门参数、可选 IMU/VIO 速度先验
- **输出**：更清晰、几何更一致的静态场景 3DGS，以及更好的 novel view synthesis

### 3) 真正的瓶颈
真瓶颈不是“2D 图像不够清晰”，而是：

> **训练时的 forward model 没有表达相机在单帧内部的连续运动。**

这会让逆渲染目标变得错配：系统用错误的物理模型去解释输入图像，导致几何和位姿都被污染。

### 4) 为什么现在值得解决
因为 3DGS 已经足够快、足够实用，但现实世界最常见的采集方式恰恰是：
- 手机手持拍摄，
- 自带 IMU，
- 常见滚动快门传感器。

也就是说，**传感器先验已经存在，真实应用需求也已经到位**，现在把物理成像过程直接接进 3DGS，是一个很自然且必要的升级。

### 5) 边界条件
这篇方法默认的适用边界很明确：
- 场景基本**静态**
- 模糊主要来自**相机运动**而非物体运动
- 单帧内相机运动可由**局部线性/恒速**近似
- 有可用的初始位姿/内参，最好还有 VIO 速度先验

---

## Part II：方法与洞察

这篇论文的核心思路不是额外学一个 blur MLP，而是把 **“模糊/滚快门是怎么形成的”** 直接写进 3DGS 的渲染器。

### 核心直觉

把方法压缩成一句话：

> 把“一帧一个 pose”的静态成像模型，改成“曝光期内一条短轨迹 + 每一行各自采样时刻”的动态成像模型，并用屏幕空间近似把这个模型做得足够快。

这背后的因果链是：

- **改变了什么**：从静态帧渲染，改为曝光期多时刻积分 + 行相关时刻偏移；
- **改变了哪个瓶颈**：把原本无法解释的 blur/RS 噪声，变成与几何、位姿一致的可解释成像约束；
- **带来了什么能力**：模型不再需要用“错误几何/错误纹理”去拟合模糊，而是能恢复更清晰、更稳的 3D 结构与新视角。

更具体地说，这个设计有效，是因为短时间内相机运动对图像的主要影响，本质上就是 **投影后 2D 位置的连续偏移**。论文抓住这个主效应，只在像素空间更新 Gaussian 均值，而不是每个时间样本都重跑完整 3D 投影，从而把物理正确性和计算效率折中起来。

### 方法骨架

#### 1. 把单帧改写为“曝光期轨迹”
作者为每一帧引入：
- 线速度 \(v_i\)
- 角速度 \(\omega_i\)

并把该帧相机在曝光时间内看成一段短时连续运动。  
同时，滚动快门通过“**图像第 y 行对应额外时间偏移**”进入渲染过程，于是同一帧的不同行会对应不同位姿时刻。

直观上：
- motion blur = 曝光时间内的时间积分
- rolling shutter = 这个积分的中心时刻还要随图像行变化

#### 2. 屏幕空间近似：只移动 Gaussian 的像素均值
这是全文最关键的工程/算法点。

作者把 3DGS 渲染拆成两步：
1. 世界坐标下 Gaussian 投影到像素空间
2. 像素空间 rasterization

然后只近似相机运动对 **Gaussian 投影中心** 的影响：
- 不重复更新完整协方差
- 不重复更新 view-dependent color
- 主要更新像素均值的时间变化

为此，作者定义了每个 Gaussian 的**像素速度**。  
这样一来，单个 Gaussian 在曝光期间的轨迹就能被看作图像平面上的线性平移。

这一步的意义非常大：

- 不需要为每个 blur sample 重新做完整 3D 投影
- 保留了 3DGS 原本高效的 rasterization 结构
- 让 motion blur / rolling shutter 的建模成本变得可接受

#### 3. 用多样本曝光积分近似模糊渲染
曝光期积分通过固定数量的时间样本近似。  
每个样本时刻的 Gaussian 中心由像素速度线性外推得到，再进行 alpha blending。

因此，渲染时：
- 行偏移负责 rolling shutter
- 时间采样负责 motion blur
- 两者在同一个 forward model 里统一处理

#### 4. 位姿与速度联合优化
作者没有把速度只当作外部输入，而是把它们也作为可优化变量：
- 初值来自 VIO
- 没有 VIO 时可设为 0

此外还做了 **pose optimization**，并通过近似梯度把相机位姿更新转化为对 Gaussian 均值梯度的利用，从而避免引入太重的新求导路径。

这个设计的作用是：
- 用 VIO 先验给 ill-posed deblurring 一个物理锚点
- 用 3DGS 光度误差继续细化 pose / velocity
- 缓解 COLMAP 在滚动快门和模糊条件下的注册偏差

#### 5. 评测时固定 Gaussian，只优化测试帧姿态/速度
论文还注意到一个评测细节：  
由于逆渲染存在 gauge ambiguity，训练后测试视角可能与重建场景有轻微整体错位。

所以他们在评测阶段固定 Gaussian，只优化测试帧的 pose/velocity，以减少这种坐标系漂移对指标的干扰。  
这是一个很实用的“评测对齐”技巧。

### 战略取舍

| 设计选择 | 改变的约束 | 直接收益 | 代价/风险 |
|---|---|---|---|
| 每帧从静态 pose 改为曝光期连续轨迹 | 把单时刻成像改为时变成像 | 能同时解释 motion blur 与 rolling shutter | 需要曝光时间、读出时间和较好的初值 |
| 屏幕空间像素速度近似 | 不再为每个时间样本重复完整投影 | 大幅降低计算量，保住 3DGS 的效率 | 忽略协方差和颜色的时间变化，极端运动下近似误差会变大 |
| VIO 初始化速度 | 把纯视觉盲去模糊变成带物理先验的优化 | 降低问题病态性，提升稳定性 | 依赖外部 VIO 质量；文中最佳流程依赖专有 SDK |
| 位姿/速度联合优化 | 不再假设外部 pose 完全正确 | 能修正模糊和滚快门造成的配准误差 | 更容易出现局部最优，训练时间明显上升 |
| 评测时固定 Gaussian 优化测试姿态 | 把 gauge 误差从模型误差中分离 | 指标更公平 | 更偏向“评测校正”，不是训练主机制 |

### 这一设计为什么比“先做 2D 去模糊再重建”更合理
作者的立场很明确：  
与其先在 2D 图像上盲目“修图”，不如直接在 3D 成像模型里解释模糊。

因为对静态场景来说，模糊不是任意纹理退化，而是：
- 相机轨迹、
- 场景几何、
- 投影模型

共同决定的。  
把这些信息统一到 3DGS forward model 里，优化的约束更强，也更不容易把假细节引进输入图像。

---

## Part III：证据与局限

### 关键证据

#### 1. 跨方法比较：在 BAD-NeRF 基准上平均最优
在 BAD-NeRF 重渲染的 Deblur-NeRF 数据上，作者与：
- Splatfacto
- MPR + Splatfacto
- Deblur-NeRF
- BAD-NeRF

进行了比较。

最强信号不是单个场景，而是**平均指标**：
- **Ours**: PSNR ≈ **29.87**, SSIM ≈ **0.925**, LPIPS ≈ **0.062**
- **BAD-NeRF**: PSNR ≈ **29.28**, SSIM ≈ **0.873**, LPIPS ≈ **0.104**

这说明：
- 3DGS 路线不是只能追平 NeRF 去模糊方法；
- 物理建模 + 屏幕空间近似，已经足以在这个任务上取得更强整体结果。

#### 2. 受控合成实验：对 rolling shutter 和 pose noise 特别有效
作者重新渲染了三类受控退化：
- 纯 motion blur
- 纯 rolling shutter
- 纯 pose noise

这里最有说服力的是“拆因素”实验，因为它直接对应论文机制。

例如 Cozyroom：
- rolling shutter：**19.21 -> 35.84**
- pose noise：**16.76 -> 36.30**

这说明该方法不仅在“模糊”上有效，更关键的是它确实在修复：
- 行时变位姿造成的投影错配
- 位姿噪声对 3DGS 的连锁破坏

也即，方法命中了真正瓶颈，而不是只做了视觉上更锐利的表面增强。

#### 3. 真实手机数据：完整系统平均最好，且每个组件都有贡献
在 11 个手机序列上：
- Splatfacto 平均 PSNR：**26.91**
- Ours 平均 PSNR：**28.59**

更重要的是 ablation：
- 去掉 motion blur compensation：降到 **27.99**
- 去掉 rolling shutter compensation：降到 **27.62**
- 去掉 pose optimization：降到 **27.44**
- 去掉 velocity optimization：降到 **27.65**
- 去掉 VIO 初始化：降到 **28.19**

这表明论文不是“某一个 trick 在起作用”，而是：
- 物理建模
- VIO 先验
- 姿态/速度联合优化

三者形成了完整闭环。

#### 4. 与 2D 预处理路线相比更稳健，但并非处处绝对胜
作者也测试了外部去滚快门方法 CVR 作为预处理。
结果是：
- 在 Pixel 5 上有时 CVR 版本更强
- 但在 iPhone / S20 上更差，且容易引入伪影

这说明论文的结论更准确的表述应是：

> **把补偿写进 3D 成像过程，整体上比先做通用 2D 预处理更稳健。**

而不是“任何场景都绝对最好”。

#### 5. 代价：训练明显更慢
效率上，完整方法在 A100 上平均约：
- baseline：**19 min**
- full method：**91 min**

即大约 **4.8x** 开销。  
其中主要成本来自 motion blur 多样本渲染；rolling shutter 本身额外开销相对较小。

所以这是一个典型的 trade-off：
- **能力换时间**
- 但仍保持在可训练范围内，且作者也说明 16GB 消费级 GPU 可运行

### 1-2 个关键指标该怎么看
如果只记两个数字，我会记：
1. **BAD-NeRF 基准平均 LPIPS 0.062 vs 0.104**：表明视觉质量提升不只是 PSNR 小涨；
2. **手机数据平均 PSNR 28.59 vs 26.91**：说明方法不只在合成数据上成立，真实手持视频也有效。

### 局限性

- **Fails when**: 极强模糊、初始化很差或位姿优化易陷局部最优时效果会退化；论文中 factory 合成场景就出现过 pose optimization 收敛到局部极小；若单帧内运动过快或明显非线性，屏幕空间线性近似可能不够。
- **Assumes**: 场景基本静态；主要退化来自相机运动而非动态物体；需要可用的内参、曝光/滚动快门参数与初始位姿；最佳效果依赖 VIO/IMU 速度先验，而文中采用的 Spectacular AI SDK 是专有系统，这会影响完全复现。
- **Not designed for**: 动态场景重建、严重散焦模糊/极近景、通用 2D 去模糊增强、任意复杂非刚体运动模糊。

### 可复用组件

1. **屏幕空间像素速度近似**  
   凡是要把“短时相机运动”接入 3DGS/可微渲染器的场景，都可复用这类“只更新投影均值”的近似思路。

2. **逐行时间偏移的曝光建模**  
   对 rolling shutter 相机的 3D 重建、SLAM、NeRF/3DGS 训练都很有参考价值。

3. **VIO 初始化 + 光度细化的 pose/velocity 联合优化**  
   这是把传感器先验接入逆渲染的典型范式，适合移动端采集场景。

### 一句话结论
这篇论文的能力跃迁点在于：**它不是在 3DGS 外面补一个去模糊模块，而是把“手机相机真实如何成像”这件事直接写进了 3DGS 的 forward model。**  
因此，提升不只是“更锐”，而是**更物理一致、更姿态稳定、更适合真实手持采集**。

![[paperPDFs/Deblurring_Rendering_Video/Lecture_Notes_in_Computer_Science_2025/2025_Gaussian_Splatting_on_the_Move_Blur_and_Rolling_Shutter_Compensation_for_Natural_Camera_Motion.pdf]]