---
title: "BAD-Gaussians: Bundle Adjusted Deblur Gaussian Splatting"
venue: arXiv
year: 2024
tags:
  - 3D_Gaussian_Splatting
  - task/novel-view-synthesis
  - task/image-deblurring
  - bundle-adjustment
  - differentiable-rasterization
  - trajectory-interpolation
  - dataset/Deblur-NeRF
  - dataset/MBA-VO
  - opensource/no
core_operator: "把曝光期内的相机运动显式建模为 SE(3) 轨迹，并对沿轨迹渲染的多帧虚拟清晰图做平均，以光度误差联合优化 3D Gaussians 与相机位姿。"
primary_logic: |
  模糊多视图图像 + 不准确相机位姿 + COLMAP 稀疏点初始化
  → 用曝光期起止位姿/样条插值得到虚拟相机轨迹，并用 3D Gaussian Splatting 渲染多帧虚拟清晰图后按物理成像过程求平均
  → 反向传播联合校正 Gaussians 与相机轨迹，输出清晰 3D 场景表示与可实时渲染的新视角结果
claims:
  - "Claim 1: 在 Deblur-NeRF 合成数据集上，BAD-Gaussians 的去模糊结果平均 PSNR 比次优方法高 3.6 dB，新视角合成平均 PSNR 比次优方法高 1.7 dB [evidence: comparison]"
  - "Claim 2: BAD-Gaussians 渲染速度超过 200 FPS、训练约 30 分钟，而 Deblur-NeRF、DP-NeRF 和 BAD-NeRF 的渲染均低于 1 FPS、训练超过 10 小时 [evidence: comparison]"
  - "Claim 3: 在 7 个合成模糊序列的位姿评测中，BAD-Gaussians 相比 COLMAP-blur 全部降低 ATE，并在其中 5 个序列上优于 BAD-NeRF [evidence: comparison]"
related_work_position:
  extends: "BAD-NeRF (Wang et al. 2023)"
  competes_with: "BAD-NeRF (Wang et al. 2023); DP-NeRF (Lee et al. 2023)"
  complementary_to: "Colmap-Free 3D Gaussian Splatting (Fu et al. 2023)"
evidence_strength: strong
pdf_ref: paperPDFs/Deblurring_Rendering/Lecture_Notes_in_Computer_Science_2025/2025_BAD_Gaussians_Bundle_Adjusted_Deblur_Gaussian_Splatting.pdf
category: 3D_Gaussian_Splatting
---

# BAD-Gaussians: Bundle Adjusted Deblur Gaussian Splatting

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2403.11831), [Project](https://lingzhezhao.github.io/BAD-Gaussians/)
> - **Summary**: 该工作把“曝光期内相机轨迹 + 物理模糊成像”引入 3D Gaussian Splatting，在模糊图像与不准位姿条件下联合恢复清晰 3D 场景和相机运动，并保留 3DGS 的实时渲染优势。
> - **Key Performance**: 在 Deblur-NeRF 合成集上，去模糊/新视角平均 PSNR 分别比次优方法高 3.6 dB / 1.7 dB；渲染速度 >200 FPS，而先前隐式方法 <1 FPS。

> [!info] **Agent Summary**
> - **task_path**: 模糊多视图图像 + 粗位姿/稀疏点云 -> 清晰 3D Gaussian 场景 -> 去模糊视图与新视角图像
> - **bottleneck**: 模糊同时破坏监督图像、COLMAP 位姿估计和 3DGS 初始化，导致场景几何与相机参数都不再可靠
> - **mechanism_delta**: 用曝光期 SE(3) 轨迹积分替代固定相机监督，把模糊图建模为多帧虚拟清晰渲染的平均，并在 3DGS 上做联合光度束调
> - **evidence_signal**: 在合成、真实、以及含加速度运动的 MBA-VO 数据上都优于 BAD-NeRF/DP-NeRF，并保持实时渲染
> - **reusable_ops**: [SE(3)曝光轨迹插值, 基于虚拟视图平均的物理模糊成像]
> - **failure_modes**: [天空与无界背景建模弱于 NeRF, 初始 COLMAP 位姿/稀疏点过差时可能难以稳定收敛]
> - **open_questions**: [能否完全摆脱 COLMAP 初始化, 能否扩展到动态物体运动模糊与 rolling shutter]

## Part I：问题与挑战

这篇工作的核心问题不是“把一张模糊图去清晰”这么简单，而是：

**如何从一组带有严重运动模糊、且相机位姿本身也不准的多视图图像中，恢复可用于新视角合成的清晰 3D 场景，并且还能实时渲染。**

### 真正瓶颈是什么？
作者指出 3DGS 在模糊场景下会同时遭遇三重失效：

1. **监督失真**：3DGS 默认假设输入图像是清晰的，但运动模糊使一个像素对应的是曝光时间内多个视角的混合，破坏了多视图几何对应。
2. **位姿失真**：COLMAP 在模糊图上恢复的相机位姿不准，而 3DGS/NeRF 都高度依赖准确位姿。
3. **初始化失真**：3DGS 还依赖 COLMAP 输出的稀疏点云做 Gaussian 初始化；模糊会让点云更稀、更差，进一步恶化优化。

所以真正的瓶颈不是单独的“模糊核估计”，而是**模糊、位姿误差、Gaussian 初始化失败三者耦合**，使 3DGS 的监督和优化基础都不成立。

### 为什么现在值得做？
现有去模糊 NeRF（如 Deblur-NeRF、DP-NeRF、BAD-NeRF）已经能一定程度处理模糊，但它们仍然有两个痛点：

- **隐式表示难保细节**，尤其在严重模糊下恢复复杂纹理更难；
- **无法实时渲染**，训练和推理成本都高。

而 3DGS 已经证明自己在**细节恢复和实时渲染**上优于很多 NeRF 变体。于是问题自然变成：**能不能把 BAD-NeRF 式的“模糊感知束调”搬到 3DGS 上？**

### 输入/输出与边界
- **输入**：模糊多视图图像、COLMAP 给出的不准位姿、以及稀疏点云初始化。
- **输出**：清晰 3D Gaussian 场景表示、曝光中点对应的去模糊视图、以及新视角合成结果。
- **边界条件**：默认场景静态，模糊主要来自**相机在曝光期内的运动**，而不是独立的物体运动。

---

## Part II：方法与洞察

方法可以概括为一句话：

**不要把模糊当作后处理卷积，而要把它当作“曝光期间连续相机运动下，多帧清晰渲染的时间平均”。**

### 方法主线

#### 1. 用物理成像过程重写模糊监督
作者把每张模糊图视为曝光时间内多个虚拟清晰图的平均。  
也就是说，先沿着相机曝光轨迹渲染出一串虚拟清晰帧，再把它们平均，得到与真实模糊图对应的合成结果。

这一步的意义是：  
**监督信号从“错位的模糊像素”变成了“可解释的时间积分结果”。**

#### 2. 把每张模糊图对应的相机轨迹也当作待优化变量
每张图不再只对应一个固定位姿，而是对应一条曝光期内的连续轨迹。

- 合成数据中：用 **起始位姿 + 终止位姿** 做线性插值；
- 真实数据中：用 **cubic B-spline** 表达更复杂的长曝光运动。

这相当于把传统 bundle adjustment 从“单帧位姿”扩展到了“曝光期轨迹”。

#### 3. 在 3DGS 上联合优化场景与轨迹
由于 Gaussian rasterization 对场景参数和相机位姿都是可微的，作者能把模糊图的光度误差直接反传到：

- 3D Gaussians 的位置、尺度、颜色、透明度
- 曝光期轨迹的控制参数

于是，**场景恢复与位姿校正不再分开做，而是互相约束、共同收敛。**

### 核心直觉

**改变了什么？**  
从“固定错误位姿 + 隐式 NeRF + 图像空间模糊建模”，改成了“曝光期轨迹建模 + 显式 3DGS + 渲染空间时间平均”。

**哪个瓶颈被改变了？**  
原先的约束是失真的：模糊像素无法直接监督一个清晰静态场景。  
现在的约束变成物理一致的：模糊图被解释为一段相机轨迹上的多次清晰观测平均。

**能力为什么会提升？**  
因为这同时解决了两个错配：

- **观测错配**：模糊图不再被硬当成清晰图监督；
- **位姿错配**：相机在曝光期内的运动被显式建模，而不是被压成一个错误的单一位姿。

再加上 3DGS 的显式表示比隐式 MLP 更容易保留细节、渲染更快，于是得到的不是仅仅“能训起来”的方法，而是**既能去模糊又能实时渲染**的方法。

### 为什么这种设计有效
因果上看，这个设计有效不是因为“用了 3DGS”，而是因为它把监督和优化对象对齐了：

- 模糊来自**时间积分**，方法就按时间积分建模；
- 位姿误差来自**曝光期运动被压缩成单帧位姿**，方法就恢复整条轨迹；
- 细节损失来自**隐式表示与慢渲染**，方法就换成显式 Gaussian 表示。

### 战略取舍

| 设计选择 | 收益 | 代价/风险 | 适用场景 |
|---|---|---|---|
| 显式 3DGS 替代隐式 NeRF | 细节更强，渲染实时 | 对天空/无界背景不一定占优；仍依赖初始化 | 静态场景重建与新视角 |
| 线性轨迹（起止位姿） | 参数少、训练稳、效率高 | 难刻画长曝光下复杂非线性运动 | 合成短曝光/近似匀速场景 |
| Cubic B-spline 轨迹 | 能拟合更复杂相机运动 | 参数更多、训练更重 | 真实长曝光场景 |
| 更多虚拟相机采样 n | 更准确逼近模糊积分，尤其对严重模糊有效 | 训练更慢，计算更高 | 严重运动模糊 |
| 联合优化场景+位姿 | 可同时修正几何和轨迹 | 对初始化仍敏感，优化更耦合 | 位姿不准的多视图模糊重建 |

---

## Part III：证据与局限

### 关键证据信号

#### 1. 多数据集比较：能力提升不是单一数据集偶然
- 在 **Deblur-NeRF 合成集** 上，作者报告：
  - 去模糊平均 PSNR 比次优方法高 **3.6 dB**
  - 新视角合成平均 PSNR 比次优方法高 **1.7 dB**
- 在 **真实数据** 上，也整体优于 Deblur-NeRF、DP-NeRF、BAD-NeRF。
- 在 **MBA-VO** 这种带加速度、非匀速相机运动的数据上依然最好，说明方法并不只适用于理想匀速模糊。

#### 2. 效率信号：不是只提质，还提速
这是本文最明显的“能力跳变”之一：

- **渲染速度 >200 FPS**
- 训练约 **30 分钟**
- 对比的隐式方法通常 **<1 FPS**，训练 **>10 小时**

也就是说，它不是单纯把 BAD-NeRF 搬到 3DGS，而是把“可用性”从离线方法推进到了接近实时系统。

#### 3. 位姿恢复信号：联合优化确实在纠正轨迹
在 7 个合成模糊序列上：

- 相比 **COLMAP-blur**，ATE 全部下降；
- 相比 **BAD-NeRF**，在 5/7 个序列上更优。

这说明它不仅生成图像更好看，而是**真的恢复了更一致的相机运动与几何结构**。

#### 4. 消融信号：关键旋钮是有效的
- **虚拟相机数量 n**：对严重模糊场景提升明显，但会逐渐饱和，所以作者选 n=10 折中质量与效率。
- **轨迹表示**：真实长曝光场景里 cubic B-spline 明显优于线性插值；合成短曝光里线性已足够。

### 局限性
- **Fails when**: 天空、无界背景、低纹理远景等 3DGS 本身不擅长的区域；论文明确提到 Factory 场景中对天空的表示弱于 BAD-NeRF。若曝光时间更长、相机运动更复杂，而仍使用简化轨迹表示，也会欠拟合。
- **Assumes**: 场景基本静态；模糊主要由相机运动引起；COLMAP 至少能提供一个可用的粗位姿和稀疏点初始化；用有限个虚拟相机采样近似曝光积分；速度结论建立在 RTX 4090 和 3DGS 实现之上。
- **Not designed for**: 动态物体运动模糊、rolling shutter、完全无 SfM 初始化的重建、通用单张 2D 去模糊任务。

### 复现与扩展时需要注意的依赖
- 依赖 **COLMAP** 提供初始位姿和稀疏点云；
- 依赖 **differentiable Gaussian rasterization**；
- 训练效率结论基于 **单张 RTX 4090**；
- 论文正文给出项目页，但从文中不能明确确认完整开源代码状态。

### 可复用组件
1. **曝光期 SE(3) 轨迹参数化**：适合任何“时间积分成像”的可微渲染问题。
2. **虚拟清晰帧平均的物理模糊模型**：比图像空间 blur kernel 更几何一致。
3. **模糊感知光度束调**：把位姿优化从“单帧”扩展到“轨迹”。
4. **n 与 densification 阈值联动**：是把时间采样并入 3DGS 训练时很实用的工程细节。

## Local PDF reference

![[paperPDFs/Deblurring_Rendering/Lecture_Notes_in_Computer_Science_2025/2025_BAD_Gaussians_Bundle_Adjusted_Deblur_Gaussian_Splatting.pdf]]