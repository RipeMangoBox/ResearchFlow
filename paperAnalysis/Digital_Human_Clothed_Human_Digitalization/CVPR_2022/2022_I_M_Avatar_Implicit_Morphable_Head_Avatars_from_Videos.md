---
title: "I M Avatar: Implicit Morphable Head Avatars from Videos"
venue: CVPR
year: 2022
tags:
  - Others
  - task/head-avatar-reconstruction
  - task/monocular-3d-reconstruction
  - implicit-surface
  - forward-skinning
  - blendshape
  - dataset/COMA
  - dataset/VOCA
  - dataset/MakeHuman
  - opensource/full
core_operator: 将FLAME式表情基与蒙皮权重提升为canonical空间连续隐式场，并用前向morphing与解析可微求交从单目视频学习可控头部avatar
primary_logic: |
  单目视频与每帧表情/姿态参数 → 在变形空间沿像素光线采样并通过前向morphing与root finding回到canonical表面 → 用canonical几何/纹理场重建像素并以解析梯度端到端训练 → 得到可外推到未见表情和姿态的隐式头部avatar
claims:
  - "在10个合成主体的测试上，IMavatar同时取得最低表情误差2.558、最低法线误差5.901和最佳LPIPS 0.01581，优于C-Net、D-Net、B-Morph与Fwd-Skin [evidence: comparison]"
  - "在真实单目视频测试上，IMavatar将表情关键点误差降至2.548，优于NerFACE的2.994，并取得更高SSIM 0.9655与更低LPIPS 0.02085 [evidence: comparison]"
  - "消融表明，使用canonical空间前向blendshape+skinning场的设计，在强表情和大jaw pose外推时误差增长明显慢于姿态/表情条件化、位移warping和backward morphing方案 [evidence: ablation]"
related_work_position:
  extends: "FLAME (Li et al. 2017)"
  competes_with: "NerFACE (Gafni et al. 2021); Neural Head Avatars from Monocular RGB Videos (Grassal et al. 2021)"
  complementary_to: "NeRF (Mildenhall et al. 2020)"
evidence_strength: strong
pdf_ref: paperPDFs/Digital_Human_Clothed_Human_Digitalization/CVPR_2022/2022_I_M_Avatar_Implicit_Morphable_Head_Avatars_from_Videos.pdf
category: Others
---

# I M Avatar: Implicit Morphable Head Avatars from Videos

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2112.07471), [Project](https://ait.ethz.ch/projects/2022/IMavatar/)
> - **Summary**: 这篇工作把 3DMM/FLAME 的“可控表情变形”机制移植到隐式头部表示里，用 canonical 空间的连续 blendshape 与蒙皮场，兼顾了高保真几何、动画可控性，以及对未见强表情/姿态的外推能力。
> - **Key Performance**: 合成数据上法线误差 5.901、PSNR 28.75、LPIPS 0.01581；真实视频上表情误差 2.548，优于 NerFACE 的 2.994。

> [!info] **Agent Summary**
> - **task_path**: 单目RGB视频 + 预估表情/姿态参数 -> 个性化、可动画控制的隐式头部avatar
> - **bottleneck**: 动态单目视频下，不同表情/姿态帧难以共享同一个canonical几何，导致变形对应欠约束、外推差
> - **mechanism_delta**: 用canonical空间的连续blendshape/pose corrective/LBS场替代按帧条件化或backward warping，并用解析梯度穿过ray marching与root finding过程
> - **evidence_signal**: 合成+真实两类数据上的系统对比，以及随表情/下颌幅度增加的外推误差曲线
> - **reusable_ops**: [canonical-space forward morphing, implicit differentiation through root-found correspondences]
> - **failure_modes**: [noisy 3DMM tracking, fine hair occlusion and mouth-interior artifacts]
> - **open_questions**: [can volumetric-surface hybrids better model hair/teeth?, can tracking be jointly refined during training to remove FLAME dependency?]

## Part I：问题与挑战

这篇论文要解决的核心，不是“从视频重建一张脸”这么简单，而是：

**如何从单目视频学到一个既高保真、又可控、还能驱动到未见表情与姿态的头部 avatar。**

### 1. 输入/输出接口
- **输入**：单目 RGB 视频；每帧预估的 FLAME 表情与姿态参数；前景 mask。
- **输出**：一个**个体专属**的隐式头部表示，可在新表情、新头部姿态下生成对应几何与外观。

### 2. 真正的瓶颈是什么？
瓶颈是**动态变形下的 canonical correspondence（规范空间对应）**。

单目视频本来就只有 2D 监督；如果模型把每一帧的表情变化都“记忆化”到条件输入里，或者用一个依赖当前姿态的 backward warping 去解释图像，那么：
- 不同帧之间很难共享同一个稳定的 canonical 几何；
- 表情变化与几何细节容易互相混淆；
- 一旦测试表情比训练更强，模型就会崩到“中性脸”、产生噪声几何，或出现不合理变形。

### 3. 以前方法为什么不够？
论文把已有路线分成两类：

1. **3DMM / mesh-based 方法**
   - 优点：表情控制非常强，blendshape / pose 参数可解释。
   - 缺点：细节上限低，难表示头发、眼镜、复杂拓扑，分辨率受限。

2. **NeRF / 隐式体渲染方法**
   - 优点：图像逼真、细结构强。
   - 缺点：通常通过“条件化表达”或“frame-dependent deformation”建模动态，**可控性和外推性弱**，几何也往往不够干净。

所以这篇论文的目标很明确：  
**把 3DMM 的控制能力，与隐式表示的细节能力合起来。**

### 4. 为什么现在值得做？
因为 VR/AR、telepresence、数字人都需要：
- 低门槛采集：最好单摄像头即可；
- 可驱动：能编辑表情和姿态；
- 高保真：不仅脸，连头发/头部轮廓都要像。

此前隐式表示已经证明“能做得很真”，但“真”不等于“可控、可外推”。这正是 IMavatar 想补的空白。

---

## Part II：方法与洞察

IMavatar 的核心设计，可以概括成一句话：

**不要让网络按帧去记住“这张图长什么样”，而是让它在 canonical 空间里学会“这个点在不同表情/姿态下该怎么动”。**

### 方法骨架

论文用三个隐式场来表示一个人：

1. **几何场（geometry field）**
   - 在 canonical 空间中预测 occupancy。
   - 代表这个人的基础头部几何。

2. **变形场（deformation field）**
   - 对 canonical 空间中的每个点，预测：
     - expression blendshapes
     - pose correctives
     - linear blend skinning weights
   - 也就是把 FLAME 里原本定义在网格顶点上的变形控制量，升级成**连续场**。

3. **纹理场（texture field）**
   - 在 canonical 点上预测颜色。
   - 额外条件化于变形后法线、jaw pose、expression，用来补偿口腔区域和非均匀光照。

### 它是怎么渲染和训练的？
对于每个像素：
- 先在**变形空间**沿光线采样；
- 对每个采样点，用 root finding 去找它在 canonical 空间的对应点；
- 再查询 canonical 几何场判断是否击中表面；
- 最后用 canonical 纹理场给该点着色。

难点在于：  
**这个 canonical 对应点不是直接回归出来的，而是通过迭代求解找到的。**

如果直接对整个迭代过程反传，训练会很麻烦。  
因此论文给了一个关键技术件：

- 把“该点位于隐式表面上”
- 和“该点变形后落在当前光线上”

视为两个隐式约束，
然后通过**隐式微分**写出 canonical 交点对网络参数的解析梯度。

这使得：
- RGB 重建误差不仅能更新纹理，
- 也能直接更新几何场和变形场。

### 核心直觉

#### changed what
从：
- **按帧条件化的隐式表示**
- 或 **依赖 deformed-space 的 backward deformation**

改成：
- **canonical-space、pose-independent 的连续 blendshape + skinning + pose-corrective 场**

#### which bottleneck changed
这一步改变的不是“网络更大了”，而是**约束结构变了**：

- 以前：每一帧都可以有自己的一套几何解释，欠约束；
- 现在：所有帧都必须对齐到同一个 canonical 空间，再通过共享的形变规则解释差异。

结果是：
- 监督被聚合到同一套 canonical 几何上；
- 表情变化不再轻易“泄漏”进几何噪声；
- 变形学习从“记忆每帧”变成“学习通用运动规律”。

#### what capability changed
直接带来的能力提升是：
- **更稳定的几何**
- **更强的表情/姿态外推**
- **更可解释的控制接口**

这也是为什么它在强表情、张嘴、颈部转动等超出训练分布时，明显比条件化基线更稳。

### 为什么这个设计有效？
因果链可以写成：

**把变形基定义在 canonical 空间**  
→ **降低 frame-specific 自由度，强迫跨帧共享几何**  
→ **像素监督能跨表情/姿态累计到同一个 3D 点附近**  
→ **学到更干净的对应关系与表面**  
→ **未见强表情时仍能按规则变形，而不是失真或塌缩。**

再加上：
- **forward morphing** 比 backward warping 更不依赖当前姿态分布；
- **解析梯度** 让 correspondence search 真正可训练；
- 于是整个系统才能从视频端到端收敛。

### 战略权衡

| 设计选择 | 带来的好处 | 代价 / 风险 |
|---|---|---|
| canonical 空间的前向 morphing | 共享几何强、控制可解释、外推更好 | 依赖较准的 3DMM 跟踪；对应搜索更复杂 |
| 用 occupancy 隐式表面而非纯体渲染 | 几何更干净，适合做法线/表面分析 | 对细发丝、薄遮挡、复杂口腔不如体表示自然 |
| 解析梯度穿过 root finding | 端到端训练可行，不需对迭代过程硬反传 | 实现复杂，训练仍偏慢 |
| 纹理网络条件化法线与口腔相关参数 | 更好解释光照和嘴部外观 | 仍不能完全解决口腔内部真实感 |

---

## Part III：证据与局限

### 关键证据信号

#### 1. 对比实验信号：不只是图像更好，几何和表情也更好
在 10 个合成主体上，IMavatar 相比 C-Net、D-Net、B-Morph、Fwd-Skin：
- **表情误差最低**：2.558
- **法线误差最低**：5.901
- **图像质量最好**：PSNR 28.75、LPIPS 0.01581

这说明它的提升不是“渲染讨巧”，而是**几何、表情、外观三者同时改善**。

#### 2. 外推分析信号：随着表情变强，基线退化更快
论文专门画了“误差 vs 表情强度 / jaw pose 强度”的曲线：
- 多个基线在 mild expression 上还能工作；
- 一到 strong expression，就迅速恶化，甚至塌回中性脸或产生无效几何；
- IMavatar 的误差增长更缓。

这个证据最能支撑论文主 claim：  
**真正提升的是分布外表情/姿态的泛化，而不只是训练集内重建。**

#### 3. 真实视频信号：在未见强表情上优于 NerFACE
在真实单目视频上：
- IMavatar 表情误差 **2.548**
- NerFACE 为 **2.994**

同时图像质量也保持竞争力：
- SSIM **0.9655**
- LPIPS **0.02085**

视觉上，论文展示了当表情/姿态难度逐步升高时，很多方法开始出现：
- 几何不完整
- 表情对不上
- 局部失真

而 IMavatar 仍能保持完整和较可信的头部输出。

#### 4. 补充实验信号：机制本身而非单个 trick 在起作用
补充材料还表明：
- 去掉 FLAME pseudo GT 监督后，性能略降，但加更多训练数据后可接近恢复；
- 对 mask 噪声较鲁棒，但对 3DMM tracking 噪声明显敏感；
- 在 MakeHuman 上，不做 test-time pose optimization 也能得到竞争性的几何结果。

这说明：
- **canonical morphing 是主要增益来源**
- FLAME 监督更像是“在数据不足时加速收敛与稳定训练”的辅助项。

### 1-2 个最关键指标
- **Synthetic**: normal error 5.901，PSNR 28.75
- **Real**: expression error 2.548，优于 NerFACE 2.994

### 局限性

- **Fails when**:  
  - 头发产生细粒度遮挡或拓扑变化时，表面表示不够自然；  
  - 口腔内部、牙齿等强遮挡/显露区域容易不真实；  
  - 3DMM tracking 有明显噪声时，纹理和几何会变糊、变形会失真。

- **Assumes**:  
  - 需要较准确的预处理：DECA/FLAME 参数、前景分割、2D facial keypoints；  
  - 真实数据设置基本是**单人、单目、固定相机**；  
  - 属于**subject-specific** 训练，不是一个通用跨身份模型；  
  - 在数据较少时，依赖 FLAME pseudo GT 监督更稳；  
  - 训练较慢，论文报告约 **2 GPU days**。

- **Not designed for**:  
  - 全身 avatar 或多人场景；  
  - 无跟踪先验的自由动态建模；  
  - 细发丝、半透明结构、复杂口腔拓扑的高保真建模；  
  - 无需个体训练的即时泛化。

### 可复用组件
这篇论文最值得复用的，不是某个网络层数，而是下面几个“操作子”：

1. **canonical-space forward morphing**
   - 把 blendshape / skinning / pose corrective 变成连续场；
   - 适合任何“既要隐式细节、又要参数化控制”的动态 avatar 任务。

2. **root-found correspondence 的解析梯度**
   - 适合“交点/对应点通过迭代求出来”的动态隐式模型；
   - 能避免对整个迭代过程直接反传。

3. **non-rigid ray marching + canonical texture querying**
   - 提供了一条从动态观察空间回到共享 canonical 表示的通用渲染路径。

**一句话总结 so what**：  
IMavatar 的能力跃迁，不在于把头像做得更“神经”，而在于把“高保真隐式表示”重新绑回一个**可解释、可共享、可外推的变形坐标系**里。

## Local PDF reference

![[paperPDFs/Digital_Human_Clothed_Human_Digitalization/CVPR_2022/2022_I_M_Avatar_Implicit_Morphable_Head_Avatars_from_Videos.pdf]]