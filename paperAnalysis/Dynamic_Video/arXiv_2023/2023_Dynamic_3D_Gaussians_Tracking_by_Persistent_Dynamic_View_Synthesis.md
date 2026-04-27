---
title: "Dynamic 3D Gaussians: Tracking by Persistent Dynamic View Synthesis"
venue: arXiv
year: 2023
tags:
  - 3D_Gaussian_Splatting
  - task/novel-view-synthesis
  - task/3d-tracking
  - gaussian-splatting
  - analysis-by-synthesis
  - local-rigidity
  - dataset/PanopticSports
  - dataset/Particle-NeRF
  - opensource/no
core_operator: 以跨时间持久化的3D高斯粒子表示动态场景，仅允许位置与旋转随时间变化，并用局部物理先验把新视角重建转化为可产生稠密6-DOF跟踪的优化问题
primary_logic: |
  多相机同步视频与标定相机 → 首帧建立静态3D高斯并固定颜色、尺度、不透明度等外观属性 → 后续帧只优化每个高斯的位置与旋转并施加局部刚性、旋转相似和长期等距约束 → 输出可实时渲染的动态场景、跨时刻一致对应关系与稠密6-DOF轨迹
claims:
  - "在 PanopticSports 上，该方法将 150 帧长期 3D 跟踪的 median trajectory error 降到 2.21cm，显著优于在线静态高斯基线 3GS-O 的 55.9cm，并达到 100% survival rate [evidence: comparison]"
  - "在作者的 2D 长期跟踪评测协议下，该方法达到 1.57 的 MTE，相比 PIPs 的 15.7 约低一个数量级，同时 survival rate 从 79.0% 提升到 100% [evidence: comparison]"
  - "消融显示局部刚性、参数固定和背景分割是关键因果部件：去掉局部刚性后 3D MTE 从 1.90cm 恶化到 4.32cm，去掉参数固定后恶化到 30.7cm [evidence: ablation]"
related_work_position:
  extends: "3D Gaussian Splatting (Kerbl et al. 2023)"
  competes_with: "Particle-NeRF (Abou-Chakra et al. 2022); OmniMotion (Wang et al. 2023)"
  complementary_to: "N/A"
evidence_strength: moderate
pdf_ref: paperPDFs/Dynamic_Video/arXiv_2023/2023_Dynamic_3D_Gaussians_Tracking_by_Persistent_Dynamic_View_Synthesis.pdf
category: 3D_Gaussian_Splatting
---

# Dynamic 3D Gaussians: Tracking by Persistent Dynamic View Synthesis

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2308.09713)
> - **Summary**: 论文把静态 3D Gaussian Splatting 改造成“属性持久、位姿时变”的动态高斯粒子系统，在不输入光流或对应监督的前提下，同时实现动态新视角合成与稠密 6-DOF 跟踪。
> - **Key Performance**: PanopticSports 上达到 28.7 PSNR；150 帧长期 3D 跟踪仅 2.21cm MTE，2D 跟踪为 1.57 MTE。

> [!info] **Agent Summary**
> - **task_path**: 同步多视角 RGB 视频 + 相机标定 -> 动态新视角渲染 + 稠密 3D/2D 对应 + 6-DOF 场景轨迹
> - **bottleneck**: 仅靠逐帧重建损失难以让同一物理区域跨时间保持身份一致，弱纹理/遮挡区域会出现“能渲染但不能稳定跟踪”
> - **mechanism_delta**: 将静态 3DGS 改为跨时持久的定向高斯粒子，并用局部刚性、旋转相似和长期等距约束把对应关系内生到重建优化里
> - **evidence_signal**: PanopticSports 上相对 3GS-O 的 3D MTE 从 55.9cm 降到 2.21cm，且关键消融显示局部刚性与参数固定是决定性组件
> - **reusable_ops**: [persistent-attribute gaussians, local-rigidity priors]
> - **failure_modes**: [首帧后新进入场景的物体无法被建模, 单目或极稀疏视角下方法不能直接工作]
> - **open_questions**: [如何支持高斯的在线 birth/death 以处理新物体进入, 如何在单目视频中维持同等级别的持久跟踪稳定性]

## Part I：问题与挑战

这篇论文真正要解决的，不只是“动态场景能否渲染得好”，而是**能否在渲染的同时，让场景中每一小块物理空间跨时间保持可追踪的身份**。

### 这个问题为什么难
现有动态场景方法常见有三类缺口：

- **逐帧/欧拉式表示**：每个时间点都能重建，但天然缺少跨时对应。
- **canonical + deformation field**：能回到参考帧，但任意两帧对应、前向对应和大形变表达都不够自然。
- **点表示**：更适合拉格朗日跟踪，但渲染效率、可微优化稳定性、旋转建模能力都弱于 3DGS。

所以真正瓶颈是：  
**图像重建损失本身不足以约束“谁在动”。**  
尤其在大面积相似纹理、遮挡、前景背景颜色接近时，优化很容易让表示“漂移”到另一个也能解释像素的位置，结果是画面看起来对，但轨迹是错的。

### 为什么现在值得做
因为静态 **3D Gaussian Splatting** 已经把高分辨率可微渲染速度拉到实时级，作者才能把 20-30 万个粒子的跨时间优化真正做起来。换句话说，之前动态 NeRF 太慢，这类“持久粒子 + test-time optimization”的路线很难落地；3DGS 把这个系统瓶颈打通了。

### 输入/输出接口与边界
- **输入**：同步多视角 RGB 视频、相机内外参。
- **实验依赖**：首帧稀疏点云初始化（文中用 10 个深度相机做初始化）、空背景参考图生成伪分割掩码。
- **输出**：  
  1. 动态场景的新视角渲染；  
  2. 每个高斯跨时间的位置与旋转；  
  3. 任意 3D 点/2D 像素的跨帧对应。  

**边界条件**很明确：它默认多机位、精确标定、首帧可见、背景基本静态；不是开集场景发现，也不是单目 in-the-wild 视频方案。

## Part II：方法与洞察

### 方法骨架

作者把动态场景表示成一组 **Dynamic 3D Gaussians**。每个高斯有两类属性：

- **跨时间固定**：颜色、尺寸、不透明度、背景属性
- **随时间变化**：3D 位置、3D 旋转（四元数）

这一步很关键：它把高斯从“每帧都可重新解释像素的渲染原语”，变成“跨时间有身份的粒子”。

整个优化流程是：

1. **首帧**：像静态 3DGS 一样优化全部参数，并做 densification，建立高质量高斯集合。
2. **后续帧**：冻结颜色/尺寸/不透明度，只更新位置和旋转。
3. **监督来源**：仍然只是多视角图像重建，经由 3D Gaussian splatting 可微渲染。
4. **时序稳定器**：加入局部刚性、局部旋转相似、长期局部等距三个物理先验。
5. **跟踪读出**：对任意点，绑定到影响它最大的高斯，在该高斯的局部坐标系中传播到别的时刻/视角。

这意味着：**tracking 不是额外模块，而是表示的直接读出。**

### 核心直觉

作者引入的关键因果旋钮是：

**把“逐帧可变外观的渲染优化”改成“持久粒子的位姿估计”。**

这会带来三个层面的变化：

1. **约束变强**  
   固定颜色、尺寸和不透明度后，高斯不能每帧“换身份”去解释别的像素区域，跨时身份更稳定。

2. **信息瓶颈被重写**  
   原来只有光度重建，弱纹理区域约束不足；现在多了局部物理一致性，邻域高斯必须近似共同运动，漂移空间大幅缩小。

3. **能力跃迁**  
   因为高斯是有方向的椭球，不只是点，所以它自带局部坐标系，能把平移与旋转耦合起来，最终输出的是 **dense 6-DOF** 而不只是点轨迹。

更具体地说：

- **局部刚性**：约束邻近高斯在相邻帧间近似遵循同一个局部刚体变换；这是最核心的 tracking 约束。
- **旋转相似**：让邻域内高斯的旋转变化更一致，提升收敛与 6-DOF 稳定性。
- **长期等距**：补足只看相邻帧时的慢性漂移问题。
- **前向传播初始化**：把上一帧速度外推到下一帧，显著降低非凸优化的搜索难度。
- **背景分割与背景静止约束**：防止衣服/背景同色时被错误吸附到静态背景上。

### 战略权衡

| 设计选择 | 改变了什么约束 | 带来的能力 | 代价/权衡 |
| --- | --- | --- | --- |
| 固定颜色/尺寸/不透明度 | 把“每帧自由解释像素”改为“同一高斯持续代表同一物理区域” | 跨时身份稳定、对应关系自然出现 | 难以表达真实外观变化、光照变化 |
| 仅优化位置与旋转 | 大幅减少时序自由度 | 长期跟踪更稳，优化更快 | 首帧建模误差会被带到后续帧 |
| 定向高斯 + 局部刚性 | 从纯光度约束变为光度 + 局部物理约束 | 无需光流即可获得 6-DOF 跟踪 | 对新物体进入、剧烈拓扑变化不友好 |
| 长期等距约束 | 抑制累计漂移 | 长序列一致性更好 | 对真实大非等距形变有潜在限制 |
| 前向传播初始化 | 缩小当前帧搜索半径 | 收敛更稳、更快 | 对超大位移或突变运动敏感 |
| 背景分割/静止约束 | 显式切断前景-背景混淆 | 同色区域跟踪稳定性显著提升 | 需要空背景参考图，增加数据依赖 |

## Part III：证据与局限

### 关键实验信号

- **比较信号｜PanopticSports 多任务统一评测**  
  相比在线静态高斯基线 3GS-O，作者方法在新视角合成上只小幅提升（PSNR 28.7 vs 28.21），但在跟踪上是质变：  
  **3D MTE 55.9cm → 2.21cm，survival 43.8% → 100%**。  
  这说明能力跃迁不在“更会渲染”，而在“把持久对应真的学出来了”。

- **比较信号｜2D 跟踪协议下对 PIPs**  
  在作者的 2D 评测协议里，方法达到 **1.57 MTE vs PIPs 的 15.7**，survival **100% vs 79%**。  
  信号含义是：在多视角、几何一致的设置下，analysis-by-synthesis 的显式 3D 表示可以压过专门的 2D learned tracker。  
  但要注意，这个对比并非完全 apples-to-apples：作者自己也指出，两边输入和训练条件不同。

- **消融信号｜真正起作用的不是所有模块，而是几项关键约束**  
  去掉局部刚性，3D MTE **1.90 → 4.32cm**；  
  去掉背景分割，PSNR **29.48 → 24.14**，3D MTE **1.90 → 8.46cm**；  
  去掉参数固定，3D MTE 直接恶化到 **30.7cm**。  
  这说明论文的核心不是“把 3DGS 搬到视频”，而是**持久属性 + 运动自由度收缩 + 局部物理先验**三件事的组合。

- **次级比较信号｜Particle-NeRF 数据集**  
  在更简单的合成场景上，作者达到 **39.49 PSNR / 0.99 SSIM / 0.02 LPIPS**，说明该表示在低复杂度动态场景下几乎可拟合到接近完美。

### 证据边界
- 3D tracking 的 GT 只有 **21 条 3D 轨迹**，主要来自手和脸关键点，不是对全场景“dense tracking”做全面标注验证。
- 最强对手 OmniMotion、MAP Visibility Estimation 没有公开代码，因此缺少更直接的强基线复现对比。
- 因此，这篇论文的实证信号很强，但在“全面密集跟踪评测覆盖度”上仍应保守解读。

### 局限性
- **Fails when**: 首帧后才进入场景的新物体出现时；需要显式增删拓扑时；前景与背景高度相似但没有分割先验时。
- **Assumes**: 同步多视角与精确标定；首帧可得到稀疏点云初始化（文中实验用 10 个深度相机）；可取得空背景参考图生成伪分割；依赖 3DGS 高效 CUDA 渲染；150 帧训练约需单张 RTX 3090 上 2 小时；正文未确认可复现代码链接。
- **Not designed for**: 单目视频直接建模；开放世界场景发现；需要时间变化外观/材质/光照的场景。

### 可复用组件
- **持久属性、运动分离的参数化**：首帧学外观，后续只学运动，适合很多 4D 表示学习任务。
- **局部刚性/旋转/等距先验**：可迁移到其他显式 4D 表示，不局限于高斯。
- **前向传播初始化**：对 test-time 时序优化非常通用。
- **dominant-primitive 局部坐标跟踪**：把 correspondence 变成表示读出，而非额外 tracker。

## Local PDF reference
![[paperPDFs/Dynamic_Video/arXiv_2023/2023_Dynamic_3D_Gaussians_Tracking_by_Persistent_Dynamic_View_Synthesis.pdf]]