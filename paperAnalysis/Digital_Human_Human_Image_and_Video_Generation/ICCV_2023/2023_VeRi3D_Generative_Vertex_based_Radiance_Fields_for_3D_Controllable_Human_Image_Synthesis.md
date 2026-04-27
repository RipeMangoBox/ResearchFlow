---
title: "VeRi3D: Generative Vertex-based Radiance Fields for 3D Controllable Human Image Synthesis"
venue: ICCV
year: 2023
tags:
  - Others
  - task/human-image-synthesis
  - task/novel-view-synthesis
  - radiance-field
  - vertex-based
  - gan
  - dataset/DeepFashion
  - dataset/AIST++
  - dataset/Surreal
  - dataset/ZJU-MoCap
  - repr/SMPL
  - opensource/no
core_operator: 基于SMPL顶点定义随姿态运动的局部坐标系，并用邻近顶点特征查询每个采样点的颜色与密度，从而生成可控人体辐射场
primary_logic: |
  随机噪声 + 相机位姿 + SMPL姿态/形状 → 2D CNN在UV空间生成顶点特征，并将射线采样点映射到K近邻SMPL顶点的局部坐标系 → 聚合顶点特征并经MLP预测颜色/密度，再经体渲染输出可控人体图像
claims:
  - "在 DeepFashion、Surreal、AIST++ 上，VeRi3D 的 FID10k 分别为 21.4、6.7、32.3，均优于 ENARF 的 60.3、21.3、95.7，也优于作者实现的 Tri.+Surf. 基线的 23.7、8.0、32.4 [evidence: comparison]"
  - "在 AIST++ 上，VeRi3D 的 PCKh@0.5 为 0.973，高于 ENARF 的 0.916 和 Tri.+Surf. 的 0.949，说明其更能保持输入姿态 [evidence: comparison]"
  - "在 ZJU-MoCap 的消融中，去掉局部坐标方向信息会使 novel-pose LPIPS 从 28.53 恶化到 30.92，说明仅依赖顶点距离统计不足以表达局部几何 [evidence: ablation]"
related_work_position:
  extends: "ENARF (Noguchi et al. 2022)"
  competes_with: "GNARF (Bergman et al. 2022); AvatarGen (Zhang et al. 2022)"
  complementary_to: "N/A"
evidence_strength: moderate
pdf_ref: paperPDFs/Digital_Human_Human_Image_and_Video_Generation/ICCV_2023/2023_VeRi3D_Generative_Vertex_based_Radiance_Fields_for_3D_Controllable_Human_Image_Synthesis.pdf
category: Others
---

# VeRi3D: Generative Vertex-based Radiance Fields for 3D Controllable Human Image Synthesis

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2309.04800), [Project](https://XDimlab.github.io/VeRi3d)
> - **Summary**: 论文把人体生成辐射场从“共享 canonical 空间”改成“绑定在 SMPL 顶点上的局部辐射场”，从而同时提升新姿态泛化、形体控制和部位级编辑能力。
> - **Key Performance**: DeepFashion 上 FID10k = 21.4；AIST++ 上 PCKh@0.5 = 0.973。

> [!info] **Agent Summary**
> - **task_path**: 2D人体图像集合 + 相机/SMPL姿态形状分布 -> 可控3D人体辐射场 -> 新视角/新姿态人体图像
> - **bottleneck**: articulated human 的 pose-agnostic 表示很难学；learned blend field 泛化差，surface warp 又容易把空背景错映射到前景导致鬼影
> - **mechanism_delta**: 用“SMPL顶点局部坐标系 + 邻近顶点特征聚合”替代全局 canonical tri-plane / learned weight field
> - **evidence_signal**: 三数据集 FID/PCKh 对比 + DeepFashion 模型跨到 AIST++ 姿态驱动时的 OOD pose 泛化
> - **reusable_ops**: [SMPL顶点局部坐标系, UV空间顶点特征生成]
> - **failure_modes**: [SMPL姿态估计与图像不匹配时会出现姿态-外观耦合, 背部和手部等少观测或分割噪声区域会出现伪影]
> - **open_questions**: [如何联合优化或校正姿态分布, 如何扩展到非SMPL拓扑或更自由服饰几何]

## Part I：问题与挑战

这篇论文要解决的是：**只用 2D 图像集合训练，却希望得到一个真正 3D-aware、且可控的人体生成模型**。目标输出不是单张 2D 人像，而是一个可渲染的 3D 人体辐射场，推理时能自由控制：

- 相机视角
- 人体姿态
- 人体形状
- 外观
- 甚至头部 / 上身 / 下身这样的部位级编辑

### 真正的瓶颈是什么？

真正的难点不是“NeRF 会不会渲染”，而是**articulated human 的表示方式**。

过去两类主流做法各有硬伤：

1. **学习 observation space → canonical space 的 blend weight field**
   - 好处：理论上灵活。
   - 问题：只靠单视角 2D 对抗训练去学这个映射太难。
   - 后果：对训练分布外姿态泛化差，尤其关节位置容易出错。

2. **用固定 surface-based warping 映射到 canonical space**
   - 好处：不用学 blend field，训练更稳。
   - 问题：映射是“表面驱动”的，全局 canonical 对齐并不总是准确。
   - 后果：背景空点可能被映到 canonical 前景区域，形成 **ghosting artifacts**。

所以，这篇论文瞄准的核心瓶颈可以概括为：

> **如何构造一个随人体运动而稳定变化、又不会把全局几何错配到 canonical 空间的人体局部表示。**

### 为什么现在值得解决？

因为 3D-aware GAN 在脸、车等刚体/弱形变对象上已经很成熟，但人体还卡在两件事上：

- **大幅 articulation**
- **服饰与局部外观的细粒度控制**

同时，SMPL 这类参数化人体模板已经提供了一个非常强的结构先验：它不直接解决服装外观，但提供了稳定的**顶点、姿态、形状、身体部位语义**。VeRi3D 的思路就是：**不要再把人体强行塞回一个共享 canonical 场，而是直接把神经场“绑”在 SMPL 顶点上。**

### 输入 / 输出接口与边界

**训练输入**：
- 2D 人体图像集合
- 相机参数分布
- SMPL 姿态/形状分布

**推理输入**：
- 随机噪声 `z`
- 相机位姿 `ξ`
- SMPL 姿态 `θ`
- SMPL 形状 `β`
- 可选的部位编辑系数

**输出**：
- 可体渲染的人体 radiance field
- 对应的新视角 / 新姿态 / 新形状的人体图像

**边界条件**：
- 依赖 SMPL 与相机估计质量
- 默认人体服从 SMPL 拓扑
- 主要针对人体主体生成，不是完整场景生成
- 视角相关反射不作为建模重点，作者默认人体近似 Lambertian

---

## Part II：方法与洞察

### 方法主线

VeRi3D 的结构可以拆成 4 步：

1. **先在 UV 空间生成顶点特征**
   - 用一个 style-based 2D CNN（类似 StyleGAN 风格）从随机噪声生成 UV feature map。
   - 再通过 UV 映射，把这个 2D feature map 采样到 SMPL 顶点上。
   - 这样每个 SMPL 顶点都有一个 feature vector。

2. **给每个 SMPL 顶点定义局部坐标系**
   - 在 T-pose 下先为每个顶点定义局部坐标轴。
   - 局部朝向由顶点法向和一个固定 up direction 决定。
   - 再通过线性蒙皮，把这个局部坐标系随姿态一起带到当前 observation space。

3. **对射线上的每个采样点，只看它的 K 个近邻顶点**
   - 对每个 3D 采样点，找其最近的 K 个 SMPL 顶点。
   - 把该点分别变换到这些顶点的局部坐标系里。
   - 这一步的含义是：不再问“它在全局 canonical 哪里”，而是问“它相对身体哪几个局部表面是什么关系”。

4. **聚合邻近顶点特征并预测颜色/密度**
   - 顶点特征按反距离加权聚合。
   - 局部坐标则做顺序无关的统计汇总：平均方向 + 距离的最小/最大/均值/方差。
   - 再把“聚合特征 + 局部坐标统计”送入 MLP，预测该点的 color / density。
   - 最后通过 volume rendering 得到图像。

### 核心直觉

VeRi3D 最关键的变化是：

> **把“全局 canonical 空间中的统一查询”改成“围绕 SMPL 顶点的局部查询”。**

这带来了三个层面的瓶颈变化：

1. **表示分布变了**
   - 旧方法：一个全局场要同时解释所有姿态的人体。
   - 新方法：网络只需学习“顶点附近局部区域”的 pose-agnostic 外观与几何。
   - 结果：学习目标从“全局大形变”变成“局部稳定模式”，更容易泛化到新姿态。

2. **错误映射约束变了**
   - 旧的 surface-based canonical warp 会把某些空背景点错误投到 canonical 前景。
   - 新方法不再依赖单一共享 canonical 位置，而是直接相对多个局部顶点建模。
   - 结果：ghosting 明显缓解。

3. **控制接口变了**
   - 顶点位置天然由 SMPL 姿态/形状控制。
   - 顶点特征天然带有身体部位归属。
   - 结果：姿态控制、形状控制、部位编辑都变得直接。

可以用一句因果链概括：

**全局 canonical 场 → 换成顶点局部场 → 降低跨姿态对齐难度与错误映射风险 → 提升 OOD pose 泛化、减少鬼影，并获得部位级控制。**

### 为什么这个设计有效？

- **姿态泛化**：局部坐标系是跟着 SMPL 顶点运动的，姿态变化主要体现在顶点位置变化，而不是让神经场重新学一套形变。
- **形状控制**：改 `β` 本质是在改 SMPL 顶点位置，辐射场支撑几何随之变化，因此可以对体型做外推。
- **部位编辑**：作者把顶点按 SMPL skinning weights 分成 head / upper body / lower body 等，再对该部位顶点特征做 PCA，因此能以低维方式操控局部服饰和外观。

### 战略取舍

| 设计选择 | 解决了什么 | 带来的能力 | 代价/风险 |
|---|---|---|---|
| 学习 blend weight field → 改为固定 SMPL 顶点局部坐标 | 避免 adversarial 条件下学习复杂 backward warp | 对新姿态更稳 | 强依赖 SMPL 拟合质量 |
| 全局 canonical 查询 → K近邻顶点局部查询 | 避免背景点错映到前景的 ghosting | 渲染更干净、姿态更准确 | 需要近邻搜索，远离身体表面的区域表达弱 |
| 直接 latent 控整人 → 基于部位顶点特征的 PCA 编辑 | 获得部位级控制接口 | 可编辑头发、上衣、下装 | 语义不完全解耦，遇到姿态噪声会串扰 |
| 直接学 per-vertex latent → UV 空间 2D CNN 生成顶点特征 | 借用成熟 2D 卷积/GAN 先验 | 训练更稳定、可生成高分辨率特征 | 受 UV 展开和模板拓扑限制 |

---

## Part III：证据与局限

### 关键证据

- **比较信号｜三数据集生成质量**
  - 在 DeepFashion / Surreal / AIST++ 上，VeRi3D 的 FID10k 分别为 **21.4 / 6.7 / 32.3**。
  - 这三项都优于 ENARF，也优于作者实现的 Tri.+Surf.。
  - 说明顶点局部表示不只是“更可控”，而是**在图像质量上也没有为了控制性而明显妥协**。

- **比较信号｜姿态遵循性**
  - 在 AIST++ 上，PCKh@0.5 达到 **0.973**，高于 ENARF 的 0.916 和 Tri.+Surf. 的 0.949。
  - 这个信号很关键，因为 AIST++ 的姿态变化更剧烈，正好检验 articulation 表示是否稳健。
  - DeepFashion 上 VeRi3D 与 ENARF 接近，也符合论文论点：**优势主要体现在高姿态变化分布下**。

- **案例信号｜OOD pose 泛化**
  - 作者用 DeepFashion 训练的模型去驱动 AIST++ 的姿态。
  - ENARF 因 learned blend field 在关节处更容易出现伪影，而 VeRi3D 更稳定。
  - 这不是单纯“好看一点”，而是直接支持其核心论点：**局部顶点表示比 learned canonical warp 更能外推到新姿态**。

- **比较 + 消融信号｜表示本身有效**
  - 在 ZJU-MoCap 重建实验里，VeRi3D 在 novel pose 上 LPIPS 为 **28.53**，优于 HumanNeRF 的 32.87 和 Tri.+Surf. 的 31.44。
  - 同时，去掉局部坐标中的方向信息后，novel-pose LPIPS 变为 **30.92**。
  - 这说明收益并不只是来自 GAN prior，而是**局部坐标设计本身确实承载了有效几何信息**。

### 局限性

- Fails when: SMPL 姿态估计与真实图像姿态有明显失配时（如弯腿被估成直腿），模型会把姿态误差“吸收”为外观/衣物变化；另外前视偏置数据中的背部、以及 AIST++ 中分割不准的手部附近，容易出现伪影。
- Assumes: 训练时能获得较准确的相机参数、SMPL 姿态/形状分布和较干净的人体前景；人体拓扑基本可由 SMPL 覆盖；还需要承担体渲染 GAN 训练成本（文中不同数据集训练约 400k–640k iterations）。这些前处理依赖会实质影响复现。
- Not designed for: 非人体或非SMPL拓扑对象、文本驱动生成、完整场景/背景生成、以及与身体模板差异很大的自由服饰拓扑。

补充一点：论文提供了 project page，但正文里**没有明确给出代码或预训练权重链接**，因此开放性与可复现性仍受限于数据预处理细节和实现成本。

### 可复用组件

- **SMPL 顶点局部坐标系**
  - 适合任何需要把 articulated body 表示成 pose-agnostic local fields 的任务。

- **UV → 顶点特征生成**
  - 把 2D CNN / StyleGAN 风格生成器迁移到 mesh vertex domain 的简单桥梁。

- **K近邻顶点特征聚合 + 顺序无关局部统计**
  - 一个轻量但有效的 radiance query 模板，可复用于点/网格驱动神经场。

- **按部位的顶点特征 PCA 编辑**
  - 不需要额外监督就能获得部位级控制接口，适合 controllable generation 场景。

## Local PDF reference

![[paperPDFs/Digital_Human_Human_Image_and_Video_Generation/ICCV_2023/2023_VeRi3D_Generative_Vertex_based_Radiance_Fields_for_3D_Controllable_Human_Image_Synthesis.pdf]]