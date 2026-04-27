---
title: "A-NeRF: Articulated Neural Radiance Fields for Learning Human Shape, Appearance, and Pose"
venue: NeurIPS
year: 2021
tags:
  - Others
  - task/novel-view-synthesis
  - task/human-pose-estimation
  - nerf
  - volumetric-rendering
  - skeleton-relative-encoding
  - dataset/Human3.6M
  - dataset/MPI-INF-3DHP
  - dataset/MonoPerfCap
  - opensource/no
core_operator: 将NeRF的查询点与视线对每根骨骼做相对编码并施加局部cutoff，在重建损失下联合优化体辐射场与骨架姿态。
primary_logic: |
  单目/多视角未标注人体视频 + 粗3D骨架初始化
  → 将射线采样点与视线重参数化为相对各骨骼的距离/方向表示
  → 通过体渲染重建图像，并用光度误差反向更新NeRF与每帧姿态
  → 输出个体化人体形状/外观模型、细化后的姿态以及新视角渲染结果
claims:
  - "在 Human3.6M Protocol I 上，相比用于初始化的 SPIN 基线，A-NeRF 将 PA-MPJPE 从 42.7 降至 39.3 [evidence: comparison]"
  - "在 Human3.6M 的 held-out 新视角评测中，完整 A-NeRF 达到 27.45 PSNR / 0.9277 SSIM，优于以 A-NeRF 细化姿态驱动的 NeuralBody 的 22.55 / 0.8782 [evidence: comparison]"
  - "消融表明，相对距离骨骼编码比相对三维位置编码带来更强的姿态细化，而 cutoff 进一步提升 wrist 细化效果 [evidence: ablation]"
related_work_position:
  extends: "NeRF (Mildenhall et al. 2020)"
  competes_with: "NeuralBody (Peng et al. 2021); SMPLify (Bogo et al. 2016)"
  complementary_to: "SPIN (Kolotouros et al. 2019); NeRF in the Wild (Martin-Brualla et al. 2021)"
evidence_strength: strong
pdf_ref: paperPDFs/Digital_Human_Clothed_Human_Digitalization/NeurIPS_2021/2021_A_NeRF_Articulated_Neural_Radiance_Fields_for_Learning_Human_Shape_Appearance_and_Pose.pdf
category: Others
---

# A-NeRF: Articulated Neural Radiance Fields for Learning Human Shape, Appearance, and Pose

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2102.06199), [Project](https://lemonatsu.github.io/anerf/)
> - **Summary**: 这篇工作把 NeRF 的空间查询从“全局世界坐标”改成“相对每根骨骼的局部过参数化编码”，从而能在未标注单目人体视频上同时学到个体化人体体渲染模型并细化 3D 姿态。
> - **Key Performance**: Human3.6M Protocol I 上 PA-MPJPE 42.7→39.3；Human3.6M held-out 渲染质量 27.45 PSNR / 0.9277 SSIM，显著高于 NeuralBody 的 22.55 / 0.8782

> [!info] **Agent Summary**
> - **task_path**: 单目未标注人体视频 + 粗3D骨架/相机初始化 -> 个体化人体辐射场 + 细化3D姿态 + 新视角渲染
> - **bottleneck**: 隐式场需要把世界坐标反映射到骨骼局部坐标，但单个3D点无法唯一归属到某根骨骼，导致跨姿态的几何/外观对齐极不稳定
> - **mechanism_delta**: 用“每根骨骼一个局部编码”的过参数化骨骼相对表示替代全局坐标输入，并把姿态作为可优化变量并入重建闭环
> - **evidence_signal**: 多数据集比较显示姿态与渲染均提升，且编码消融证明相对距离+cutoff是关键因子
> - **reusable_ops**: [骨骼相对过参数化编码, 基于光度重建的测试时姿态细化]
> - **failure_modes**: [训练分布外极端姿态会产生伪影, 视角覆盖不足时背面与细结构恢复不稳]
> - **open_questions**: [如何从单人转向跨身份通用人体模型, 如何支持物理正确光照与重光照]

## Part I：问题与挑战

这篇论文真正要解决的，不只是“把 NeRF 用到人体上”，而是更具体的一件事：

**给定同一人的一段未标注单目视频，只依赖粗糙的初始 3D 骨架，能否同时学出该人的 3D 体外观模型，并把每帧姿态优化到与图像严格对齐？**

### 真正的难点在哪里

传统参数化人体模型（如 SMPL 系）处理 articulation 的思路是：  
**先有模板表面，再用前向运动学把表面带到目标姿态。**

但 NeRF 这类隐式场反过来是：  
**给你一个世界坐标里的查询点，网络要判断这个点在人体哪里、是什么颜色、是否有密度。**

这会带来一个关键瓶颈：

1. **隐式场需要“逆向”理解骨骼运动**  
   不是把网格点从 canonical pose 变到当前 pose，而是要把当前世界坐标里的采样点解释成“它相对哪根骨骼、在什么局部位置”。

2. **单点到骨骼的归属是病态的**  
   靠近关节、衣物、遮挡处时，一个点并不能唯一对应某根骨骼。  
   如果硬分配给最近骨骼，表示会不稳定；如果仍用全局坐标，网络看到的是“同一条胳膊在不同帧跑到完全不同位置”，很难聚合跨帧证据。

3. **单目、未标注、动态人体同时出现**  
   原始 NeRF 假设静态场景 + 多视角标定；这里却是动态非刚体 + 单目 + 相机估计 + 无 3D GT。

### 输入 / 输出接口

- **输入**：同一人的单目或多视角视频；每帧粗 3D 骨架初始化；估计的相机内参
- **输出**：  
  1. 个体化的 volumetric human model  
  2. 每帧细化后的 3D 骨架姿态  
  3. 可用于新视角合成、动作迁移的可渲染人体表示

### 为什么这件事值得现在做

因为两个条件终于同时成立了：

- **NeRF 提供了高保真的可微渲染监督**
- **现成 3D pose estimator（如 SPIN）已经能提供“够用但不准”的初始化**

所以现在可以把“判别式粗初始化”和“生成式重建细对齐”接起来，试图摆脱对模板网格、扫描数据、严格多视角标定的依赖。

### 边界条件

这不是一个“通用人体模型”工作，而是**transductive / person-specific** 设定：

- 训练时就知道目标视频
- 一次训练只针对一个人
- 需要看到足够多姿态，最好能覆盖人体各个朝向
- 光照只被近似建模，不追求物理正确 relighting

---

## Part II：方法与洞察

### 方法主线

A-NeRF 的整体流程可以概括成四步：

1. **用现成 3D pose estimator 初始化每帧骨架与相机参数**  
   论文用的是 SPIN 作为初始化来源。

2. **把 NeRF 查询从全局坐标改成骨骼相对编码**  
   对每个射线采样点，不再只喂给网络世界坐标，而是把它表示成：
   - 相对每根骨骼的距离
   - 相对每根骨骼的方向
   - 相对每根骨骼的视线方向

3. **通过体渲染重建图像**  
   仍然保持 NeRF 的 volumetric rendering 框架。

4. **联合优化人体场和姿态**  
   用图像重建误差反向更新：
   - NeRF 参数（形状 + 外观）
   - 每帧姿态参数  
   同时加姿态正则与时间平滑，防止姿态漂移。

### 核心直觉

**它最关键的改动，不是“给 NeRF 加个 skeleton 条件”，而是改变了网络看到人体运动的坐标系。**

#### what changed

从：
- **全局世界坐标下建模动态人体**

变成：
- **在每根骨骼的局部坐标系里，过参数化地描述同一个查询点**

#### which bottleneck changed

原来在全局坐标里，人体运动是一个高度复杂的非刚体分布：  
同一块表面会随着动作出现在完全不同的位置和方向。

改成骨骼相对坐标后，这个问题被重写成：

- 大尺度人体运动被拆成**分段近似刚体**
- 采样点不需要先做“硬骨骼归属”
- 网络可以在多个局部坐标里自己学习哪些骨骼相关、哪些无关

也就是说，论文改变的是**信息对齐方式**：  
让跨帧、跨姿态的证据更容易落到同一个局部参考系里。

#### what capability changed

这直接带来三种能力变化：

1. **能从单目视频聚合跨帧几何与纹理证据**
2. **能把光度重建误差反过来当成 pose refinement 信号**
3. **不用模板表面，也能学到可新视角渲染的个体化人体体表示**

### 为什么“过参数化到每根骨骼”有效

如果只把点分配给最近骨骼，关节附近会非常脆弱。  
A-NeRF 反而采用**每根骨骼都编码一次**的 overcomplete 表示：

- 不先做离散决定
- 让网络自己决定哪些骨骼局部信息该被利用
- 再用 **cutoff** 抑制远处骨骼，保留局部性

这是一个很系统的设计：  
**用连续过参数化替代 brittle 的离散 part assignment。**

### 为什么最终选“相对距离 + 相对方向 + 相对视线”

作者并没有停在“相对位置”上，而是进一步做了压缩和归纳偏置设计：

- **相对距离**：更低维，也更适配人体肢体的“围绕骨骼分布”结构
- **相对方向**：补回距离编码丢失的方向信息
- **相对视线**：帮助建模 view-dependent appearance
- **per-image appearance code**：吸收动态光照/曝光等帧级变化

### 策略性取舍

| 设计选择 | 解决的瓶颈 | 能力收益 | 代价 / 风险 |
|---|---|---|---|
| 全局坐标直接喂 NeRF | 实现最简单 | 无需额外结构 | 遇到大姿态变化时几何与纹理严重混乱 |
| 相对骨骼“位置”编码 | 避免硬分配到单骨骼 | 能表达局部几何关系 | 维度高，优化更难 |
| 相对骨骼“距离+方向”编码 | 降维并保留局部结构 | 更稳、更利于姿态细化 | 仍需额外机制恢复视角相关外观 |
| cutoff 位置编码 | 抑制远骨骼噪声 | 局部性更强，优化更稳 | 需要设置阈值与温度 |
| 联合姿态优化 | 修正初始姿态误差 | 末端关节提升明显，渲染更锐利 | 训练更慢，依赖合理初始化 |

### 方法的因果链总结

可以把这篇论文浓缩成下面一句：

**把“动态人体难以对齐”的问题，改写成“在骨骼局部坐标里做分段近刚体建模”，于是 NeRF 可以更稳定地积累跨帧监督，重建损失也因此变成可用的姿态校正信号。**

---

## Part III：证据与局限

### 关键证据

#### 1) 比较信号：姿态细化确实有效
- 在 **Human3.6M Protocol I** 上，A-NeRF 把初始化基线的 **PA-MPJPE 从 42.7 降到 39.3**
- 在 **Protocol II 的 wrist** 上，改进更明显：**66.5 → 57.3**
- 这说明它的收益不只是“平均误差小降一点”，而是**对最容易错的末端关节更有帮助**

**解读**：  
如果一个方法真的用到了体渲染几何一致性，那么它最该改善的就是手腕、脚踝、手肘这类初始化噪声大的部位。实验确实符合这个模式。

#### 2) 比较信号：渲染质量的提升不只是来自更好的驱动姿态
- 在 **Human3.6M held-out** 上，完整 A-NeRF 达到 **27.45 PSNR / 0.9277 SSIM**
- 即使给 NeuralBody 喂 A-NeRF 细化后的姿态，它也只有 **22.55 / 0.8782**

**解读**：  
这说明提升不只是“pose 变准了”，而是**表示本身**更适合单目、噪声姿态、无模板的设定。

#### 3) 消融信号：核心收益来自骨骼相对编码，而不是普通 pose conditioning
- 补充实验表明：
  - **relative distance** 比 **relative position** 更适合 pose refinement
  - 加上 **cutoff** 后，腕部细化进一步提升
  - 去掉 pose refinement 时，渲染会出现 limbs / face 的 ghosting 和模糊

**解读**：  
这组实验直接对应论文的因果主张：  
不是“任何带 skeleton 的条件输入都行”，而是**局部、过参数化、带 cutoff 的编码**才让 NeRF 真正稳定。

### 能力跳跃相对 prior work 在哪里

相对前作，这篇论文的“跳跃”主要体现在：

1. **不依赖参数化人体表面模板**
2. **不依赖 3D 扫描或显式 surface supervision**
3. **单目视频即可训练，且允许姿态在测试时继续优化**
4. **同时服务于两类目标：新视角渲染 + 3D pose refinement**

特别是和 NeuralBody 相比，A-NeRF 更像是在回答：  
**如果没有可靠模板表面、只有噪声骨架初始化，NeRF 还能不能工作？**  
这正是它的价值所在。

### 局限性

- **Fails when**: 训练时没见过的极端姿态会产生明显伪影；人物若没有被从足够多方向观察到，背面几何和细薄结构（如箭杆）恢复会不稳定。
- **Assumes**: 依赖外部 3D pose estimator（文中为 SPIN）提供可用初始化，也依赖估计的相机内参；是单人、逐视频训练的 transductive 设定；训练成本较高，文中报告约需 2×V100 32GB、约 60 小时，512×512 单张渲染约 1–4 秒。
- **Not designed for**: 跨身份通用人体建模、实时重建、多演员长视频、物理正确光照建模与 relighting。

### 可复用组件

1. **骨骼相对过参数化编码**  
   对任何“关节驱动的隐式场”都很有借鉴价值，不限于人体。

2. **cutoff 的局部位置编码**  
   适合把大范围条件输入变成局部有效的 inductive bias。

3. **基于重建误差的 test-time pose refinement**  
   可以作为“判别式初始化 + 生成式细化”的通用范式。

4. **relative ray + per-image appearance code**  
   这是在动态、非物理一致光照场景中处理 view-dependent appearance 的实用折中。

![[paperPDFs/Digital_Human_Clothed_Human_Digitalization/NeurIPS_2021/2021_A_NeRF_Articulated_Neural_Radiance_Fields_for_Learning_Human_Shape_Appearance_and_Pose.pdf]]