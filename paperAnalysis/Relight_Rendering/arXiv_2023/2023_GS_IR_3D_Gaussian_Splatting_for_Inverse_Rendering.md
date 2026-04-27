---
title: "GS-IR: 3D Gaussian Splatting for Inverse Rendering"
venue: arXiv
year: 2023
tags:
  - 3D_Gaussian_Splatting
  - task/inverse-rendering
  - depth-regularization
  - occlusion-baking
  - spherical-harmonics
  - "dataset/TensoIR Synthetic"
  - "dataset/Mip-NeRF 360"
  - opensource/full
core_operator: 以3D高斯为显式场景表示，并用深度导数法线正则与SH遮挡烘焙补齐3DGS在逆渲染中缺失的法线和可见性建模能力
primary_logic: |
  多视角RGB图像（未知静态光照） → 用3DGS重建几何并以线性插值深度导出伪法线监督高斯法线 → 烘焙SH遮挡/间接光体积并结合PBR优化材质与环境光 → 输出可重光照的几何、材质与照明分解
claims:
  - "Claim 1: 在 TensoIR Synthetic 上，GS-IR 的 novel-view synthesis 达到 35.333 PSNR，略高于 TensoIR 的 35.088，同时训练时间从约 5 小时降到 1 小时以内 [evidence: comparison]"
  - "Claim 2: 在 TensoIR Synthetic 上，GS-IR 的 albedo 重建达到 30.286 PSNR，高于 TensoIR 的 29.275 和 NVDiffrec 的 29.174 [evidence: comparison]"
  - "Claim 3: 采用线性插值深度并用高斯存储法线作为代理后，法线 MAE 可从 16.347 降到 4.948，并把 novel-view PSNR 从 25.756 提升到 35.333 [evidence: ablation]"
related_work_position:
  extends: "3D Gaussian Splatting (Kerbl et al. 2023)"
  competes_with: "TensoIR (Jin et al. 2023); NVDiffrec (Munkberg et al. 2022)"
  complementary_to: "N/A"
evidence_strength: strong
pdf_ref: paperPDFs/Relight_Rendering/arXiv_2023/2023_GS_IR_3D_Gaussian_Splatting_for_Inverse_Rendering.pdf
category: 3D_Gaussian_Splatting
---

# GS-IR: 3D Gaussian Splatting for Inverse Rendering

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2311.16473), [Code](https://github.com/lzhnb/GS-IR)
> - **Summary**: 这篇工作把 3D Gaussian Splatting 从“快速新视角合成表示”扩展成“可做几何、材质、光照分解的逆渲染框架”，关键补上了稳定法线估计和可见性/间接光建模这两个 3DGS 天生缺口。
> - **Key Performance**: TensoIR Synthetic 上 novel-view PSNR 35.333、albedo PSNR 30.286，训练时间 <1h（TensoIR 约 5h）。

> [!info] **Agent Summary**
> - **task_path**: 多视角 RGB 图像（相机已标定、未知静态光照） -> 几何/法线/材质/环境光分解 -> 新视角渲染与重光照
> - **bottleneck**: 3DGS 的前向 splatting 不擅长稳定求法线，也无法像 NeRF 射线积分那样直接追踪遮挡和间接光
> - **mechanism_delta**: 用线性插值深度导出的伪法线监督每个高斯的共享法线，再用 SH 烘焙的遮挡/照明体积替代昂贵的逐射线可见性求解
> - **evidence_signal**: 双数据集比较加消融同时表明，法线正则和遮挡烘焙都能稳定改善渲染指标，且相对 TensoIR 训练提速约 5x
> - **reusable_ops**: [线性插值深度估计, SH 遮挡/照明体积烘焙]
> - **failure_modes**: [高频镜面间接光场景效果弱, 极薄结构或几何松散时法线不稳]
> - **open_questions**: [如何在 3DGS 中建模镜面间接光, 如何减少遮挡阈值与体积分辨率等启发式设计]

## Part I：问题与挑战

GS-IR 解决的是一个比普通新视角合成更难的问题：  
**输入**是多视角、已标定、但光照未知的 RGB 图像；  
**输出**不是只要“渲染像”，而是要恢复：

- 几何/法线
- 材质参数（albedo、metallic、roughness）
- 环境光照

从而支持：

- 新视角渲染
- 重光照
- 材质编辑

### 真正瓶颈是什么？

论文抓得很准：**3DGS 很快，但它缺少逆渲染所需的两个物理核心量。**

1. **法线不稳定**
   - 3DGS 在训练时会自适应调整高斯密度。
   - 这对前向渲染很好，但会让几何变“松”，深度容易漂在真实表面前面。
   - 一旦深度不稳，基于深度梯度的法线就会噪声大、跨视角不一致。

2. **前向 splatting 天然不擅长算遮挡**
   - NeRF 类方法是 backward mapping / ray marching，天然能沿射线问“这条光路被谁挡住了”。
   - 3DGS 是把高斯直接投到图像平面上，速度快，但不方便做 visibility tracing。
   - 可是逆渲染又必须面对阴影、AO、间接光、互反射这些可见性问题。

### 为什么现在值得做？

因为 3DGS 已经把**高质量 + 实时级渲染**这件事做出来了。  
如果能把它从“快的前向渲染器”升级成“快的逆渲染器”，就意味着 inverse rendering 不再局限于 NeRF 式的慢优化流程，而能进入更实用的交互场景。

### 边界条件

这篇方法有明确适用边界：

- **静态场景**
- **静态但未知的光照**
- **多视角已标定图像**
- **Cook-Torrance / PBR 参数化材质假设**
- 面向**物体级 + 无界真实场景**，但不是动态场景建模方案

---

## Part II：方法与洞察

GS-IR 的思路不是把 NeRF inverse rendering 生硬搬到 3DGS 上，而是：

> **保留 3DGS 的显式表示与前向 splatting 速度优势，再用两个兼容 3DGS 的补丁，补齐法线与可见性。**

整体是一个三阶段管线：

1. **初始阶段**：先把 3DGS 几何学好，同时稳定法线
2. **烘焙阶段**：把遮挡与间接光缓存到体积里
3. **分解阶段**：结合 PBR 优化材质和环境光

### 核心直觉

**what changed**  
- 从“在线逐射线追踪可见性”改为“离线烘焙低频遮挡/间接光”
- 从“每个视角单独从深度图导法线”改为“每个高斯维护共享法线，并由深度导数提供监督”

**which bottleneck changed**  
- 计算瓶颈：把高代价 visibility tracing 变成局部体积查表
- 信息瓶颈：把噪声大、视角依赖的法线估计，变成跨视角共享的显式法线变量

**what capability changed**  
- 3DGS 开始具备可用的 inverse rendering 能力
- 在保持较快训练/渲染的同时，能恢复材质与光照并支持 relighting

### 三阶段机制

#### 1) 用“线性插值深度”稳住法线来源

作者先指出两个常见但不适合 3DGS 的深度做法：

- **体渲染累积深度**：会有 floating problem，深度飘在表面前
- **peak selection**：虽然避免漂浮，但会带来 disc aliasing

GS-IR 的替代做法是：

- 把像素深度视为参与该像素的各高斯深度的**加权线性插值**
- 同时保证该深度落在最小/最大高斯深度之间

这样做的直觉是：  
它不追求精确求射线交点，而是求一个**更稳定、与前向 splatting 相容的表面位置估计**。

然后再做两步：

- 从渲染深度图取梯度，得到伪法线
- 用这个伪法线去监督每个高斯里存储的法线

关键点在于：**最终参与分解的是“高斯共享法线”，不是每帧单独算出来的 noisy depth normal。**  
这相当于把“视角局部信号”蒸馏成“场景级共享几何属性”。

#### 2) 用 SH 体积烘焙解决遮挡与间接光

这是整篇最有系统味道的设计。

因为 3DGS 不方便像 NeRF 那样沿射线算 occlusion，于是作者转而采用类似游戏图形学的 precomputation 思路：

- 在 3D 空间规则放置**occlusion volumes**
- 对每个体积位置渲染 6 个方向的 depth cubemap
- 用阈值把 depth cubemap 变成二值 occlusion cubemap
- 再把这个 cubemap 投影为 **SH 系数**

之后在线阶段只需：

- 对任意表面点，从附近 volume 做插值
- 恢复该点的低频遮挡/AO
- 类似地查询间接照明 volume

这一步本质上把“难以微分且昂贵的全局可见性”压缩成了“低频、局部、可插值的缓存表示”。

#### 3) 用 PBR 分解材质与环境光

在最终阶段，作者把：

- 高斯中存储的材质参数
- 环境光 map
- illumination volumes

一起放进 PBR 渲染管线中优化。

这里采用 image-based lighting 和 split-sum 近似，重点不是公式，而是系统结构：

- **直接光**：由环境图提供
- **间接漫反射**：由烘焙体积提供
- **镜面项**：用预积分近似

所以 GS-IR 得到的是一个能做重光照的显式场景，而不是单纯的外观拟合器。

### 为什么这套设计有效？

因果链可以概括为：

**稳定深度**  
→ 深度梯度更可信  
→ 伪法线监督不再发散  
→ 高斯法线更一致  
→ 材质/光照分解更稳

以及：

**把 visibility 从在线求解改成离线缓存**  
→ 避免前向 splatting 的结构短板  
→ 保留 3DGS 的高速渲染  
→ 仍能建模 AO 与低频间接漫反射  
→ relighting 与 decomposition 变得可行

### 战略取舍表

| 设计选择 | 解决的瓶颈 | 带来的能力变化 | 代价/边界 |
| --- | --- | --- | --- |
| 线性插值深度 | 避免 floating depth 与 peak aliasing | 为法线提供更稳定的几何来源 | 仍是近似深度，不是严格表面交点 |
| 高斯存储共享法线 + 深度导数监督 | 单视图法线噪声大、跨视角不一致 | 法线可跨视角共享，利于后续 BRDF 分解 | 法线质量仍略逊于 TensoIR |
| SH 遮挡/间接光体积 | 3DGS 无法自然做 ray-based visibility | 以查表代替追踪，保留高效率 | 只擅长低频，主要覆盖 diffuse indirect light |
| 分阶段训练 | 几何/材质/光照耦合过强 | 收敛更稳，工程上更易实现 | 可能弱于端到端全局联合最优 |

---

## Part III：证据与局限

### 关键实验信号

#### 1) 它真正赢在哪里：速度 + NVS/albedo 质量
在 **TensoIR Synthetic** 上：

- **Novel view PSNR = 35.333**
  - 高于 TensoIR 的 35.088
- **Albedo PSNR = 30.286**
  - 高于 TensoIR 的 29.275
- **训练时间 < 1h**
  - 对比 TensoIR 的约 5h

这说明：  
GS-IR 的能力跳跃不只是“能做 inverse rendering”，而是**把 inverse rendering 从慢的 NeRF 路线搬到了更快的 3DGS 路线，同时没有明显牺牲主渲染质量**。

#### 2) 但它不是所有维度都最强
同一张表里也要看到负信号：

- **法线 MAE 4.948**
  - 仍略差于 TensoIR 的 4.100
- **Relight PSNR 24.374**
  - 明显低于 TensoIR 的 28.580

所以更准确的结论是：

> GS-IR 在 **效率、novel-view synthesis、albedo 分解** 上很强，  
> 但在**法线精度和重光照保真**上还没有完全超过最强 NeRF-based inverse rendering。

#### 3) 在真实无界场景上证明了泛化价值
在 **Mip-NeRF 360** 上：

- GS-IR 达到 **25.381 PSNR**
- 高于 NeRF++（25.112）、Plenoxels（23.079）、INGP-Base（25.303）
- 但低于专做 NVS 的 3DGS（27.21）

这个信号很重要：  
GS-IR 不是只在合成物体上做 decomposition，它确实能扩展到真实复杂场景。  
但也说明其目标不是纯 NVS 最优，而是**在可分解、可重光照前提下保持可接受的 NVS 质量**。

#### 4) 消融真正支持了“因果按钮”
最关键的消融有两个：

- **法线正则消融**
  - 从直接用体渲染深度法线，到“线性插值深度 + 高斯法线代理”
  - 法线 MAE 从 **16.347 降到 4.948**
  - Novel-view PSNR 从 **25.756 升到 35.333**
  - 这是非常强的因果证据：法线稳定性不是小修小补，而是决定整个 inverse rendering 是否能成立的关键

- **遮挡/间接光消融**
  - 去掉 occlusion 或 indirect illumination，指标都会下降
  - 且 AO 可视化显示阴影与遮挡细节更完整
  - 说明 baking 不是装饰性模块，而是支撑物理一致 shading 的必要组件

### 局限性

- **Fails when**: 场景包含高频镜面间接光、强互反射、镜面多次弹射，或存在很薄且难稳定定位的几何结构时，GS-IR 的法线与重光照质量会明显受限。
- **Assumes**: 依赖多视角已标定图像、静态场景与静态光照、Cook-Torrance 材质模型、低频 SH 表示的遮挡/间接漫反射；实现上还依赖手工阈值、体积缓存布局、自定义 CUDA/光栅化组件。
- **Not designed for**: 动态场景、时变光照、精确镜面全局光照、焦散、完全物理精确的多次反射模拟。

### 复现与工程依赖

- 优点：
  - 开源代码已提供
  - 单张 V100 即可训练
  - 总训练流程约 30K + 10K iteration，工程成本低于 NeRF-based inverse rendering

- 代价：
  - 需要自定义 splatting / baking 实现
  - 需要多次 cubemap 渲染和体积缓存
  - 遮挡阈值与 SH 表示能力会影响结果上限

### 可复用组件

这篇论文最值得迁移的不是整套系统，而是两个中间算子：

1. **面向 forward splatting 的线性插值深度估计**
   - 适合所有需要从 3DGS 稳定提取几何信号的任务

2. **SH 遮挡/照明体积缓存**
   - 适合所有“在线 visibility 太贵，但允许低频近似”的实时场景渲染/编辑任务

---

## Local PDF reference

![[paperPDFs/Relight_Rendering/arXiv_2023/2023_GS_IR_3D_Gaussian_Splatting_for_Inverse_Rendering.pdf]]