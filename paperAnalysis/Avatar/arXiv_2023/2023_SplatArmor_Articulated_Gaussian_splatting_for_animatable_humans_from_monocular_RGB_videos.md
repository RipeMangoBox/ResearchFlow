---
title: "SplatArmor: Articulated Gaussian splatting for animatable humans from monocular RGB videos"
venue: arXiv
year: 2023
tags:
  - 3D_Gaussian_Splatting
  - task/animatable-human-reconstruction
  - task/novel-view-synthesis
  - forward-skinning
  - se3-field
  - neural-color-field
  - dataset/ZJU-MoCap
  - dataset/PeopleSnapshot
  - repr/SMPL
  - opensource/no
core_operator: 在SMPL规范空间中布置3D高斯，并用扩展前向蒙皮与姿态条件SE(3)场驱动，再以神经颜色场正则外观并反向约束高斯位置
primary_logic: |
  单目RGB视频+前景mask+初始SMPL → 在canonical空间以SMPL+D表面初始化并优化3D高斯 → 用扩展LBS前向蒙皮和姿态条件SE(3)非刚性场把高斯映射到观测空间 → 用神经颜色场提供颜色正则与位置监督 → 输出可新视角/新姿态驱动的人体avatar
claims:
  - "在 ZJU-MoCap 的 6 个主体上，SplatArmor 的 PSNR 与 LPIPS 均优于文中比较的 SMPLPix、NeuralBody、AniNeRF、SA-NeRF 与 HumanNeRF [evidence: comparison]"
  - "用神经颜色场替代每高斯独立颜色后，ZJU-MoCap 平均 LPIPS 从 26.90 降至 26.08，并减少 novel pose 下的 stray Gaussian 伪影 [evidence: ablation]"
  - "在 ZJU-MoCap 上，SplatArmor 训练约需 7 小时和 9GB 显存，而 HumanNeRF 约需 3 天和 48GB；推理可实时，而 HumanNeRF/AnimNeRF 低于 0.2 FPS [evidence: comparison]"
related_work_position:
  extends: "3D Gaussian Splatting (Kerbl et al. 2023)"
  competes_with: "HumanNeRF (Weng et al. 2022); SA-NeRF (Xu et al. 2022)"
  complementary_to: "Mesh Strikes Back (Jena et al. 2023)"
evidence_strength: strong
pdf_ref: paperPDFs/Avatar/arXiv_2023/2023_SplatArmor_Articulated_Gaussian_splatting_for_animatable_humans_from_monocular_RGB_videos.pdf
category: 3D_Gaussian_Splatting
---

# SplatArmor: Articulated Gaussian splatting for animatable humans from monocular RGB videos

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2311.10812)
> - **Summary**: 这篇工作把可动画人体建模从“观测空间采样并逆蒙皮的 NeRF”改成“canonical 空间高斯并前向蒙皮的 splatting”，再用神经颜色场稳定纹理与高斯位置，从单目视频获得更稳、更快的人体 avatar。
> - **Key Performance**: 在 ZJU-MoCap 6/6 主体上 PSNR 与 LPIPS 双第一；ZJU 训练约 7h / 9GB，而 HumanNeRF 约 3 天 / 48GB。

> [!info] **Agent Summary**
> - **task_path**: 单目RGB视频+前景mask+初始SMPL -> 可新视角/新姿态渲染的个体化人体高斯avatar
> - **bottleneck**: 逆蒙皮 canonicalization 的多解歧义，以及每高斯独立颜色在联合优化时的外观过拟合
> - **mechanism_delta**: 把高斯固定在 canonical 空间并用扩展前向LBS+姿态条件刚体场驱动，再用位置条件颜色场替代自由颜色参数
> - **evidence_signal**: 双数据集比较 + 颜色场/姿态变换/预训练消融，且 ZJU-MoCap 六个主体上 PSNR 和 LPIPS 全面领先
> - **reusable_ops**: [canonical-space Gaussian armoring, arbitrary-point forward skinning, training-only neural color field regularization]
> - **failure_modes**: [未见区域或自阴影区域发黑, 初始化差或极端外推姿态时出现错位/纹理伪影]
> - **open_questions**: [如何建模视角相关外观与高光, LBS+局部SE(3)对大幅未见姿态的泛化上限在哪里]

## Part I：问题与挑战

这篇论文解决的是：**从单目 RGB 视频恢复可动画、可控、细节丰富的人体模型**。  
输入并不只是视频本身，还依赖：

- 每帧 RGB 图像
- 前景 mask
- 初始 SMPL 形状/姿态
- 可优化的相机外参

输出则是：

- 一个 canonical 空间中的 3D Gaussian 集合
- 优化后的 SMPL 形体与姿态参数
- 能在新视角、甚至未见姿态下驱动的人体表示

### 真正的瓶颈是什么？

核心难点不是“能否拟合训练帧”，而是**如何在姿态变化时保持几何与外观的一致性**。

现有 animatable human NeRF 方法常见流程是：  
在**观测空间**沿射线采样点，再把这些点**逆蒙皮**回 canonical 空间。问题在于：

1. **逆蒙皮有多解歧义**  
   一个观测点可能对应多个 canonical 身体部位，尤其在肢体交叠、衣物褶皱、单目遮挡时更严重。训练帧里能“记住”，但新姿态常扭曲。

2. **Mesh 太刚，隐式场太慢**  
   纯 mesh 方法受拓扑限制，难覆盖 loose clothing 和 pose-dependent 细节；NeRF 又要体渲染，训练和推理都贵。

3. **显式高斯颜色容易过拟合**  
   如果每个 Gaussian 自己持有颜色参数，在 pose、camera、canonical geometry 一起优化时，很容易把训练帧纹理“写死”，导致 novel pose 时 stray Gaussians 暴露成伪影。

### 为什么现在值得做？

因为 **3D Gaussian Splatting** 给了一个新折中：

- 比 NeRF 的 ray marching / ray tracing 更快
- 仍保留可学习的非 mesh 几何自由度
- 天然适合“把 primitive 放在 canonical 空间，再前向驱动出去”

换句话说，这篇论文抓住的时机是：**高斯 splatting 已经证明了静态/动态场景的效率，而人体场景又恰好最需要一个“前向骨架驱动”的显式表示。**

### 边界条件

这不是无条件的人体重建系统，它默认：

- 单人、人体主体清晰
- 有可用的前景分割
- 有初始 SMPL 对齐
- 身体运动大体可由 SMPL/LBS 解释，再由局部非刚性修正补足

---

## Part II：方法与洞察

方法的总体哲学可以概括为一句话：

**不要从观测空间“猜”每个点属于哪个 canonical 身体部位；而是先把高斯安放在 canonical 身体上，再用身体先验把它们前向送到观测空间。**

### 方法结构

#### 1. 用 SMPL 给高斯“穿铠甲”
SplatArmor 把人体表示成：

- 一个 canonical SMPL/SMPL+D 粗模型
- 一组位于 canonical 空间的 3D Gaussians

这些高斯不是漂浮在无约束 3D 空间里，而是被 SMPL 几何“锚定”住，所以作者称之为 **armor**。

#### 2. 把 LBS 从网格顶点扩展到任意 3D 点
标准 SMPL 的线性混合蒙皮只定义在网格顶点上。  
论文做的关键一步是：对任意 canonical 点 \(x\)，找到附近的 SMPL 顶点，用这些顶点的局部刚体变换与混合权重，定义一个**任意点的前向蒙皮**。

这带来两个直接后果：

- Gaussian center 可以稳定地随姿态前向运动
- Gaussian covariance 也能随刚体变换一起旋转/拉伸

所以它不是只搬运“点”，而是在搬运**有方向和形状的各向异性高斯**。

#### 3. 用姿态条件 SE(3) 场补 pose-dependent 非刚性
仅靠 LBS 还不够，因为衣物褶皱、软组织、局部滑动都有 pose-dependent effects。  
作者没有像很多 NeRF 一样只学一个点偏移，而是让 MLP 输出一个**刚体修正**（rigid transform）。

这一步很关键，因为 Gaussian 是各向异性的：

- 只平移，不够
- 全 affine，太自由，容易过补偿
- rigid / SE(3) 修正，刚好能同时处理位置和方向

#### 4. 用神经颜色场替代每高斯独立颜色
这是论文第二个很漂亮的设计。

不是给每个 Gaussian 一个独立颜色向量，而是学习一个 **color field**：颜色由 canonical 位置决定。  
这样做的作用有两层：

- **隐式颜色正则化**：邻近位置颜色更连续，减少纹理过拟合
- **反向监督高斯位置**：如果颜色由位置决定，渲染误差会通过颜色场的空间梯度，反过来约束 Gaussian center 的 3D 位置

所以颜色场不只是“更平滑的纹理参数化”，而是把 appearance loss 也变成了 geometry supervision。

#### 5. 先做粗初始化，再做高斯优化
作者先用已有方法恢复一个粗的 SMPL+D mesh 和每面片颜色，再从表面采样约 20000 个点初始化高斯与颜色场。  
这一步的作用是减少欠约束性：否则颜色场、非刚性场、姿态、相机、高斯位置会互相甩锅。

### 核心直觉

这篇论文真正改变的是三个“因果开关”：

1. **表示空间变了**  
   从“观测空间采样点 + 逆蒙皮”  
   变成“canonical 空间显式高斯 + 前向蒙皮”  
   → 改变了 canonicalization 的信息流方向  
   → 减少多解歧义  
   → novel pose 更稳

2. **非刚性修正的对象变了**  
   从“给点加 offset”  
   变成“给各向异性高斯加刚体修正”  
   → 变换同时作用于位置与方向  
   → 衣物褶皱、手臂附近细节更不容易糊或扭

3. **颜色参数化变了**  
   从“每高斯自由颜色”  
   变成“颜色是位置函数”  
   → 改变了 appearance supervision 的约束形态  
   → 纹理更连续，隐藏高斯更难携带任意颜色  
   → novel pose 伪影减少

归根结底，SplatArmor 的能力提升不是来自更大的网络，而是来自**把人体 articulation 的物理先验嵌进高斯表示本身**。

### 战略性折中

| 设计选择 | 解决的瓶颈 | 带来的能力变化 | 代价 / 折中 |
| --- | --- | --- | --- |
| canonical 高斯 + 前向蒙皮 | 逆蒙皮多解、NeRF 体渲染慢 | novel pose 更稳，渲染更快 | 依赖较好的 SMPL 初始化 |
| 姿态条件 rigid/SE(3) 修正 | translation-only 不能处理各向异性，affine 易过拟合 | 细节在姿态变化下更自然 | 表达力弱于完全自由变形 |
| 神经颜色场 | 每高斯颜色过拟合、stray Gaussians | LPIPS 更好，颜色还能监督几何 | 不处理强视角相关外观 |
| 粗 SMPL+D 预训练 | 联合优化欠约束 | 更快收敛、更稳泛化 | 需要额外初始化流程 |

---

## Part III：证据与局限

### 关键证据：能力跃迁到底体现在哪？

#### 1. 比较信号：ZJU-MoCap 上“全面领先”
最强证据来自 ZJU-MoCap。  
论文表 1 显示，SplatArmor 在 6 个主体上 **PSNR 和 LPIPS 都排第一**。这说明它不只是“更快”，而是在人像保真度上也赢了。

更有意思的是，作者指出：

- AniNeRF 和 SA-NeRF 在单相机训练下会输出空白图
- 因而这些 baseline 还改成了 **4 个相机训练**
- 即便如此，SplatArmor 仍然领先

这说明它对**单目设置**本身更友好，而不只是调参占优。

#### 2. 泛化信号：未见姿态下比 HumanNeRF 更稳
在 AMASS 驱动的未见姿态测试中，作者给出了和 HumanNeRF 的定性比较：

- HumanNeRF 在手臂等部位出现明显形变扭曲
- SplatArmor 能更稳定维持身体轮廓与衣物细节

这个证据主要是 **case study**，不是严格量化 benchmark，但它直接支撑了论文最核心的论点：  
**前向蒙皮比逆蒙皮更适合做 animatable human 的 canonical correspondence。**

#### 3. 消融信号：颜色场和 rigid pose MLP 都是“有效因果旋钮”
论文的消融不是摆设，结论比较清楚：

- **去掉颜色场**：平均 LPIPS 从 **26.08 恶化到 26.90**
- **不做预训练**：novel pose 伪影明显增加
- **pose MLP 只做平移**：高斯方向错，纹理更噪
- **pose MLP 做全 affine**：会过补偿，头/躯干形状异常
- **rigid 变换**：在表达力和稳定性之间最好

这说明作者提出的三个模块不是松散拼装，而是分别对应三个真实瓶颈。

#### 4. 效率信号：不仅更稳，还显著更省
这篇论文的“so what”很大一部分就在这里：

- ZJU-MoCap：SplatArmor 训练约 **7 小时 / 9GB**
- HumanNeRF：约 **3 天 / 48GB**
- 推理：SplatArmor **实时**，HumanNeRF / AnimNeRF **< 0.2 FPS**

所以它的能力提升不是“更贵换更好”，而是**表示更对路，因此又快又稳**。

### 两个最值得记住的结果

- **ZJU-MoCap subject 377**：33.06 PSNR / 19.77 LPIPS，并且 6 个主体上都保持第一
- **颜色场消融**：平均 LPIPS 从 26.90 降到 26.08，直接验证“颜色函数化”不是装饰设计

### 局限性

- **Fails when**: 测试视角暴露出训练中几乎没见过的区域、或出现强自阴影/遮挡时，纹理会发黑或不一致；极端未见姿态下，若 LBS 不能解释真实变形，局部仍可能失真。
- **Assumes**: 需要前景 mask、较可靠的初始 SMPL/相机、以及一个额外的粗 SMPL+D 预训练步骤；方法默认人体运动大体服从骨架驱动，再由局部刚体场补充。
- **Not designed for**: 强视角相关外观、高光/反射显著材质、完全脱离 SMPL 拓扑先验的大尺度拓扑变化、以及一般动态场景建模。

此外，复现上还有两个现实约束：

- 文中未见公开代码/项目链接，`opensource/no`
- 虽然训练资源比 NeRF 小很多，但初始化链条（mask、SMPL、粗 mesh）仍是成败关键

### 可复用组件

- **任意 3D 点的前向蒙皮**：适合把骨架先验扩展到点/高斯/其他显式 primitive
- **姿态条件 rigid 修正场**：适合任何各向异性显式表示，而不只是 Gaussian
- **训练时颜色场、推理时缓存颜色**：兼顾正则化与速度
- **粗几何先行的初始化策略**：对所有“显式几何 + 联合外观优化”的单目问题都很有价值

## Local PDF reference

![[paperPDFs/Avatar/arXiv_2023/2023_SplatArmor_Articulated_Gaussian_splatting_for_animatable_humans_from_monocular_RGB_videos.pdf]]