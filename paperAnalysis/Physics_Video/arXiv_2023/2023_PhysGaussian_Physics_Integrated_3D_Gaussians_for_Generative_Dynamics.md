---
title: "PhysGaussian: Physics-Integrated 3D Gaussians for Generative Dynamics"
venue: arXiv
year: 2023
tags:
  - 3D_Gaussian_Splatting
  - task/video-generation
  - material-point-method
  - continuum-mechanics
  - dataset/BlenderNeRF
  - opensource/no
core_operator: 以3D Gaussian兼作MPM物质点与渲染核，用形变梯度同步更新几何协方差和外观方向，实现“所见即所模拟”的物理动画。
primary_logic: |
  静态多视图图像与相机参数 → 重建静态3D Gaussian并可选做内部填充/各向异性约束 → 用连续介质力学与MPM推进位置、速度、形变梯度和材料状态 → 将形变梯度映射为Gaussian协方差并同步旋转球谐方向 → 输出可新视角渲染的物理一致动态序列
claims:
  - "PhysGaussian在6个合成lattice deformation测试上均取得最高PSNR，超过NeRF-Editing、Deforming-NeRF和PAC-NeRF [evidence: comparison]"
  - "让Gaussian协方差随形变梯度演化、并旋转球谐方向，相比fixed covariance、rigid covariance和fixed harmonics变体能获得更好的变形后渲染质量 [evidence: ablation]"
  - "基于opacity field的internal filling能避免空心壳体在重力下塌陷，并带来更符合体积材料直觉的仿真结果 [evidence: case-study]"
related_work_position:
  extends: "3D Gaussian Splatting (Kerbl et al. 2023)"
  competes_with: "PAC-NeRF (Li et al. 2023); Deforming-NeRF (Xu and Harada 2022)"
  complementary_to: "SuGaR (Guédon and Lepetit 2023); Segment Any 3D Gaussians (Cen et al. 2023)"
evidence_strength: moderate
pdf_ref: paperPDFs/Physics_Video/arXiv_2023/2023_PhysGaussian_Physics_Integrated_3D_Gaussians_for_Generative_Dynamics.pdf
category: 3D_Gaussian_Splatting
---

# PhysGaussian: Physics-Integrated 3D Gaussians for Generative Dynamics

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2311.12198), [Project](https://xpandora.github.io/PhysGaussian/)
> - **Summary**: 这篇工作把3D Gaussian从“只能渲染的显式表示”升级成“可被连续介质力学驱动的物质点”，从而在不抽取网格的前提下，直接把静态场景重建变成可物理控制的动态新视角生成。
> - **Key Performance**: 在6个合成lattice deformation测试上PSNR全部第一；简单场景可达 plane 30 FPS、toast 25 FPS、jam 36 FPS。

> [!info] **Agent Summary**
> - **task_path**: 静态多视图图像/相机 + 手工指定材料与控制条件 -> 动态3D Gaussian场景 -> 新视角运动序列
> - **bottleneck**: 仿真几何与渲染几何长期分离，导致形变分辨率不匹配；同时普通3DGS缺少可积累的体积力学状态
> - **mechanism_delta**: 把每个Gaussian同时当作MPM粒子和渲染核，并用形变梯度更新协方差、用局部旋转修正球谐方向
> - **evidence_signal**: 合成lattice deformation基准上对三类基线全胜，并有协方差/球谐旋转消融支持核心设计
> - **reusable_ops**: [Gaussian-as-particles, deformation-gradient-to-covariance, inverse-rotate-view-for-SH]
> - **failure_modes**: [无internal filling时体积物体容易表现为空心壳体并塌陷, 未建模阴影演化时大光照变化下真实感受限]
> - **open_questions**: [如何从视频自动估计材料参数, 如何扩展到液体和更复杂多相材料]

## Part I：问题与挑战

这篇论文要解决的不是“如何重建一个静态3DGS”，而是更难的版本：

**能否从静态多视图重建出的场景，直接生成符合物理规律的新运动，同时保持3DGS级别的高质量新视角渲染？**

### 真正瓶颈是什么？

核心瓶颈不是“不会做形变”，而是**表示不统一**：

1. **传统物理动画管线**里，渲染几何和仿真几何是两套东西。  
   你先重建/建模，再做tet/cage/mesh，再仿真，最后再渲染。中间每一步都可能引入误差和分辨率错配。

2. **已有NeRF/动态GS编辑方法**通常只解决“几何怎么动”，但不真正携带**速度、应力、塑性、体积守恒、碰撞**这些物理状态。  
   于是它们更像视觉变形器，而不是物理生成器。

3. **3DGS天然更偏表面外观**。  
   直接拿表面高斯做体积材料仿真，内部是空的，重力或压缩下就会像空壳一样塌。

### 输入 / 输出接口

- **输入**：静态场景的多视图图像和相机参数  
- **额外条件**：用户指定仿真区域、材料参数、部分粒子的初速度/控制条件  
- **输出**：物理一致的动态3D Gaussian，以及对应的新视角运动渲染

### 为什么是现在？

因为两个条件终于同时成熟：

- **3D Gaussian Splatting**给了一个可编辑、可实时渲染的显式3D表示；
- **MPM**给了一个对弹性体、塑性体、颗粒、黏塑流体、碰撞、断裂都较通用的连续体仿真框架。

论文的判断很明确：  
**如果同一组Gaussian既负责“看起来是什么”，又负责“物理上怎么动”，就能把渲染与仿真之间的鸿沟直接抹平。**

---

## Part II：方法与洞察

### 方法主线

PhysGaussian的主线可以概括成四步：

1. **先重建静态3DGS**  
   从多视图图像得到一组Gaussian核：位置、协方差、opacity、球谐系数。

2. **给Gaussian补上“物理身份”**  
   把每个Gaussian当成连续体离散粒子，赋予：
   - 位置、速度
   - 质量、体积、密度
   - 形变梯度
   - 应力/塑性相关状态

3. **用MPM推进动力学**  
   通过连续介质力学 + MPM，在时间上推进粒子运动，支持：
   - 弹性
   - 金属塑性
   - 断裂
   - 砂土类颗粒
   - Herschel-Bulkley黏塑材料
   - 碰撞

4. **把物理状态直接转回可渲染Gaussian**  
   最关键的一步：不是先导出mesh再渲染，而是直接把物理形变映射到Gaussian本身：
   - 粒子位置更新 Gaussian 中心
   - 形变梯度更新 Gaussian 协方差
   - 局部旋转更新球谐方向
   - 然后直接用原始3DGS splatting渲染

### 核心直觉

以前的方法里，Gaussian/NeRF更像“外观容器”；  
PhysGaussian把它变成“**局部仿射连续体单元**”。

这带来了一个关键因果链：

**把Gaussian从纯渲染基元改成同时承载一阶形变信息的物理粒子**  
→ **局部形变不再只体现在中心点移动，而是体现在支撑域本身的拉伸/剪切/旋转**  
→ **渲染覆盖范围、视角相关外观、物理状态三者保持一致**  
→ **在大形变下仍能保住细节，同时出现真实的惯性、体积行为和材料差异**

更具体地说，有三个决定性开关：

#### 1) 协方差跟着形变梯度走，而不是只移动中心

如果只平移Gaussian中心，表面覆盖会断裂；  
如果只做刚体变换，也表达不了局部拉伸和剪切。

论文的关键改动是：**用局部形变梯度去更新Gaussian协方差**。  
这让Gaussian的“形状”本身随材料形变而变化，因此渲染时仍能贴住物体。

#### 2) 球谐方向也要跟着物体转

仅仅几何变了还不够。  
3DGS里颜色与视角有关，如果物体转了但球谐基还停留在原坐标系，外观会飘。

作者用局部旋转对视角做逆旋转补偿，等价地实现**球谐基随物体旋转**。  
这一步虽然数值增益不总是巨大，但对外观稳定性很关键。

#### 3) 内部填充把“表面模型”补成“体积材料”

3DGS重建往往把高斯堆在表面附近。  
对仿真来说，这意味着对象内部是空心的。

作者基于opacity field做internal filling，把内部体素也补成粒子。  
这一步改变的是**材料分布**，所以会直接改变重力下是否塌陷、压缩下是否保体积。

### 一个重要但容易忽略的点

作者还给出了一种**增量式Gaussian演化**思路：  
不必总依赖完整形变梯度，也可以直接用局部速度梯度更新协方差。  
这为以后接入不以总形变梯度为核心的材料模型留了接口。

### 策略取舍

| 设计选择 | 解决的瓶颈 | 带来的能力 | 代价 / 风险 |
| --- | --- | --- | --- |
| 同一组Gaussian同时做仿真与渲染 | 仿真几何/渲染几何错配 | 无需mesh/tet/cage，真正WS2 | 物理质量高度依赖初始3DGS重建质量 |
| 用MPM而不是显式网格FEM | 复杂材料、接触、拓扑变化难处理 | 断裂、颗粒、碰撞、黏塑材料更自然 | 需要网格步进、材料模型和稳定性调参 |
| 用形变梯度更新协方差 | 大形变下表面覆盖破裂 | 渲染质量在变形后更稳定 | 局部仅是一阶仿射近似，极端非仿射运动可能失真 |
| 旋转球谐方向 | 视角相关外观与物体姿态脱节 | 旋转场景下外观更一致 | 实现上依赖局部旋转估计 |
| internal filling | 表面高斯导致空壳仿真 | 更像真实实体，材料参数才有意义 | 额外预处理与阈值设计 |
| anisotropy regularizer | 过细高斯在大形变下冒刺 | 降低毛刺/绒毛伪影 | 会限制表示自由度，可能略影响静态拟合效率 |

---

## Part III：证据与局限

### 关键证据

#### 1) 对比实验信号：不是只会“看起来在动”，而是变形后仍能保住渲染质量

作者构建了一个可控的**lattice deformation benchmark**，在 wolf / stool / plant 上做 bend 和 twist，并和三类方法比较：

- NeRF-Editing
- Deforming-NeRF
- PAC-NeRF

**结论**：PhysGaussian在6个测试上PSNR全部第一。  
最能说明问题的不是平均值，而是难例上的差距：

- **Stool-bend**：31.15 dB，显著高于 NeRF-Editing 25.00 / Deforming-NeRF 22.32 / PAC-NeRF 21.83
- **Plant-bend**：25.81 dB，高于 19.85 / 17.90 / 18.50

这说明它的优势不只是“能接入物理”，而是**在变形后仍保住3DGS级渲染 fidelity**。

#### 2) 消融信号：核心收益确实来自“高斯形变”和“球谐旋转”

作者做了三组关键消融：

- Fixed Covariance：只移动Gaussian中心
- Rigid Covariance：只做刚体协方差变换
- Fixed Harmonics：不旋转球谐方向

结论很清楚：

- **不让Gaussian本体形变，会出现明显覆盖错误和视觉伪影**
- **不旋转球谐，外观一致性会下降**

这直接支持论文的核心因果主张：  
**物理状态必须被传到渲染基元自身，而不是只传中心轨迹。**

#### 3) 质性证据信号：能力跨度比纯编辑方法更大

论文展示了多种材料与动力学：

- elasticity
- metal plasticity
- fracture
- sand / soil
- viscoplastic paste / gel
- collision

这不是单一“编辑效果”扩展，而是**统一表示下的多材料动力学生成**。

#### 4) 效率信号：简单场景已接近实时

在简单动态场景中，作者报告：

- plane：30 FPS
- toast：25 FPS
- jam：36 FPS

这说明方法不是纯离线概念验证，至少在轻量场景下具备交互潜力。

### 局限性

- Fails when: 光照/阴影随形变显著变化时，方法不会同步建模shadow evolution；若初始3DGS主要覆盖表面、且internal filling不足，体积行为仍会失真；极端大变形下过细高斯若未被正则化也容易产生毛刺伪影。
- Assumes: 需要静态多视图重建作为前提；材料参数、仿真区域和部分初速度由人工设定；物理行为依赖选定的MPM constitutive model；真实数据采集还依赖COLMAP等外部重建流程；实验硬件基于RTX 3090级别GPU；正文给出project page，但未明确给出代码仓库，因此复现便利性一般。
- Not designed for: 自动从视频估计材料参数；从单目动态视频直接恢复并生成复杂动力学；液体与更复杂多相材料；文本驱动或高层语义控制的开放式动画生成。

### 可复用组件

这篇论文最值得迁移的，不只是整套系统，而是几个可复用操作：

1. **Gaussian-as-particles**  
   把显式3DGS基元直接升格为物理离散粒子。

2. **形变梯度到协方差的运输规则**  
   适合任何“局部仿射形变 + Gaussian渲染”的系统。

3. **球谐方向的逆旋转补偿**  
   适合所有带view-dependent appearance的可变形显式表示。

4. **基于opacity field的内部填充**  
   对表面偏置的显式重建表示都很有用。

5. **各向异性正则**  
   对动态场景下的3DGS稳定性是很实用的工程补丁。

**一句话总结 So what：**  
PhysGaussian真正的能力跃迁，不是“把物理加到3DGS里”这么简单，而是**把物理和渲染统一到同一离散实体上**，因此它第一次比较完整地实现了：**从静态真实场景重建出发，无需中间网格代理，就能做多材料、可新视角、较高保真的物理动态生成。**

## Local PDF reference

![[paperPDFs/Physics_Video/arXiv_2023/2023_PhysGaussian_Physics_Integrated_3D_Gaussians_for_Generative_Dynamics.pdf]]