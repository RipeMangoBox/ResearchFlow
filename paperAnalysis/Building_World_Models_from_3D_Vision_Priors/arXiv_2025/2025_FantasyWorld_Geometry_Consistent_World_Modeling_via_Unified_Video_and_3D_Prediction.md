---
title: "FantasyWorld: Geometry-Consistent World Modeling via Unified Video and 3D Prediction"
venue: arXiv
year: 2025
tags:
  - Video_Generation
  - task/video-generation
  - diffusion
  - bidirectional-cross-attention
  - implicit-3d-field
  - dataset/WorldScore
  - dataset/RealEstate10K
  - opensource/no
core_operator: "在冻结的视频扩散骨干中插入可训练几何分支，并通过双向跨分支注意力在一次前向中联合生成视频潜变量与隐式3D场"
primary_logic: |
  参考图像/可选文本/相机轨迹 → PCB先用冻结Wan2.1前层对视频潜变量做预去噪 → IRG中视频分支与几何分支经双向跨注意力互相约束并联合预测深度/点图/相机 → 输出几何一致视频与可复用3D表示
claims:
  - "在WorldScore photorealistic评测中，FANTASYWORLD在Small与Large两种相机运动设置下都取得最高3D Consist.，且Photo/Style Consist.也领先所比较方法，说明其多视角一致性更强 [evidence: comparison]"
  - "移除几何分支与双向跨分支注意力后，WorldScore中的3D Consist.从83.31降到79.77（Small），从74.83降到72.06（Large），表明显式几何建模是性能提升的主要来源 [evidence: ablation]"
  - "在RealEstate10K上的3DGS重建实验中，使用相同VGGT初始化时，完整模型将PSNR/SSIM/LPIPS从26.89/0.84/0.17改进为28.24/0.86/0.14，说明联合视频-几何表示提升了几何保真度 [evidence: comparison]"
related_work_position:
  extends: "VGGT (Wang et al. 2025a)"
  competes_with: "AETHER (Aether Team et al. 2025); Voyager (Huang et al. 2025a)"
  complementary_to: "Context-as-Memory (Yu et al. 2025b); AnySplat (Jiang et al. 2025)"
evidence_strength: moderate
pdf_ref: paperPDFs/Building_World_Models_from_3D_Vision_Priors/arXiv_2025/2025_FantasyWorld_Geometry_Consistent_World_Modeling_via_Unified_Video_and_3D_Prediction.pdf
category: Video_Generation
---

# FantasyWorld: Geometry-Consistent World Modeling via Unified Video and 3D Prediction

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2509.21657)
> - **Summary**: 该文把冻结的Wan视频扩散骨干与一个可训练几何分支对齐到同一潜空间，使模型在单次前向中同时生成视频和隐式3D表示，从而提升多视角几何一致性而不需要per-scene优化。
> - **Key Performance**: WorldScore Large相机运动下，3D Consist. 达到 **74.83**；在RealEstate10K的3DGS重建中（相同VGGT初始化），达到 **PSNR 28.24 / LPIPS 0.14**。

> [!info] **Agent Summary**
> - **task_path**: 参考图像/可选文本/相机轨迹 -> 几何一致视频 + 深度/点图/相机姿态
> - **bottleneck**: 视频基础模型有强生成先验，但内部特征缺少显式3D grounding，导致跨视角一致性弱、且难直接复用到3D任务
> - **mechanism_delta**: 在冻结Wan2.1内部先做预去噪，再插入几何分支与双向跨分支注意力，让视频潜变量和隐式3D场共同演化
> - **evidence_signal**: WorldScore一致性指标在Small/Large两种相机运动下均领先，且去掉几何分支后性能显著回退
> - **reusable_ops**: [latent-preconditioning, bidirectional-cross-attention]
> - **failure_modes**: [固定长度clip无法维持长程世界状态, 相机/内容控制不如专门控制型方法]
> - **open_questions**: [如何把隐式3D表示做成可缓存的长期记忆, 如何直接验证导航/novel-view等下游泛化]

## Part I：问题与挑战

这篇论文抓住的**真实难点**不是“把视频生成得更好看”，而是：

**如何让视频生成模型内部真的携带可用的3D世界结构，而不是仅仅在2D投影上看起来像一致。**

### 1. 为什么这是瓶颈
现有相机可控视频模型已经能生成多视角视频，但大多仍是**视频域内部自洽**：

- 有强想象先验，但**没有显式3D监督**
- 特征主要服务于2D去噪，不天然支持**深度/点云/相机**等3D推理
- 真要做3D重建时，常要再接NeRF/3DGS等**场景级优化**
- 视频生成与3D感知往往只是“并排做”，不是**同一潜表示里互相约束**

所以问题不是“有没有3D信号”，而是**3D信号是否进入了生成主干的内部表示流**。

### 2. 为什么现在值得做
作者认为现在有两个条件成熟了：

1. **视频基础模型**已经有很强的外观、时序、遮挡补全先验  
2. **VGGT/DUSt3R/Fast3R**这类前馈3D模型说明：3D几何可以在单次前向里预测出来

因此，最自然的下一步不是重新训练一个完整世界模型，而是：

**保留视频大模型的生成能力，同时把几何能力接入其隐藏特征。**

### 3. 输入/输出接口
FANTASYWORLD的输入输出非常明确：

- **输入**：参考图像、可选文本、目标相机轨迹
- **输出**：
  - 沿指定视角生成的视频
  - 可解码的隐式3D表示，进一步输出深度图、点图、相机位姿

### 4. 边界条件
这篇论文的结论主要成立于以下边界内：

- 重点是**photorealistic世界生成**
- 评测集中在**WorldScore static**的photorealistic子集
- 更像“相机绕静态场景运动”的世界建模，而不是完整的动态交互物理世界
- 当前生成是**固定长度clip**，不是流式长期世界状态建模

---

## Part II：方法与洞察

### 设计哲学
作者没有选择“把整个视频基础模型拿来重训/微调”，而是采取更保守但更工程可行的路线：

> **冻结Wan2.1主干，只训练一个几何分支和轻量交互模块，让几何能力附着到现有视频潜空间上。**

整个方法可分成三步理解。

### 1. PCB：先把纯噪声变成“有结构的潜变量”
作者观察到，扩散模型越往后层，特征越有结构；如果几何分支一上来就吃高噪声latent，监督会不稳定。

所以他们把Wan2.1前16层当作**Preconditioning Blocks (PCB)**：

- 不训练
- 只负责把视频latent做一段“预去噪”
- 让几何分支接触到已经带有空间结构线索的特征

这一步改变的不是输出，而是**几何学习的输入分布**。

### 2. IRG：视频分支与几何分支在同一主干里共同演化
核心模块是 **Integrated Reconstruction and Generation (IRG) Blocks**。

其中有两个不对称分支：

- **Imagination Prior Branch**：沿用预训练Wan主干，保留外观与时序生成能力
- **Geometry-Consistent Branch**：把Wan隐藏特征投到几何对齐的latent space，用于预测深度/点图/相机

两者通过**双向跨分支注意力**连接：

- **geometry -> video**：用几何约束视频，减少多视角漂移
- **video -> geometry**：用视频生成先验补足遮挡和未观测区域，避免几何只会“看见什么预测什么”

这比“视频输出后再做3D重建”更深一层，因为**约束进入了生成过程本身**。

### 3. 两阶段桥接训练
训练也很有针对性：

- **Stage 1: Latent Bridging**
  - 冻结Wan
  - 只训练几何分支
  - 让几何分支先学会“读懂”Wan block 16的隐藏特征

- **Stage 2: Unified Co-Optimization**
  - 在后续24个block后插入双向cross-attention adapter
  - 继续冻结核心backbone
  - 只优化轻量交互模块和几何路径

这意味着作者的关键操作不是重写基础模型，而是**建立潜空间桥梁**。

### 4. 3D解码头的一个关键小设计
作者还改了DPT式解码头的使用方式：

- 传统DPT偏向从浅层提细节
- 但扩散模型浅层更噪，深层才更稳定、更有结构

因此他们反过来更多依赖**更深、更干净的扩散特征**做几何解码，并加时间上采样对齐视频帧。

这说明作者不是机械地“加个3D head”，而是考虑了**扩散特征的层内语义分布**。

### 核心直觉

**改变了什么：**  
从“视频latent只服务2D去噪，3D靠后处理或弱耦合分支补上”，变成“视频latent与几何latent在同一前向里共同更新”。

**改变了哪个瓶颈：**  
把原本的**信息瓶颈**从“只保留外观与时序线索”改成“同时保留外观先验 + 显式几何监督”。  
也就是说，3D信息不再只出现在输出端，而是进入了**隐藏状态演化过程**。

**能力发生了什么变化：**
- 视频跨视角更稳
- 遮挡区域的几何预测更容易借助生成先验补全
- 内部特征更接近“可复用世界表示”，而不是一次性视频噪声轨迹

**为什么这设计有效：**
1. PCB先降低噪声，避免几何分支在无意义latent上学习  
2. geometry->video把3D一致性“钉”回生成过程  
3. video->geometry利用大规模视频先验填补不可见区域  
4. 冻结主干减少了破坏已有生成能力的风险

### 战略取舍

| 设计选择 | 解决的瓶颈 | 收益 | 代价/妥协 |
|---|---|---|---|
| 冻结Wan2.1，只训练几何分支和adapter | 全量微调成本高且可能伤害生成先验 | 保留已有视频生成能力，训练更稳 | 上限受原始主干潜空间约束 |
| PCB预去噪后再接几何 | 几何监督直接面对高噪声latent不稳定 | 降低梯度方差，几何更容易收敛 | 需要人为选择切分层位 |
| 双向跨分支注意力 | 2D生成与3D感知弱耦合 | 视频与几何互相增益 | 需要额外几何标签和模块复杂度 |
| 隐式3D场而非首帧显式点云先验 | 首帧先验容易随视角变化失效 | 大视角运动下更稳，不易掉出视野 | 可解释/可编辑性不如显式场景图 |
| 反向reassemble的3D DPT head | 扩散浅层特征噪声大 | 深度/位姿预测更稳 | 对扩散层级特性有依赖 |

---

## Part III：证据与局限

### 关键证据信号

**信号1：基准比较表明它真正提升的是“几何一致性”，不是所有指标都强。**  
在WorldScore上，FANTASYWORLD在Small和Large两种相机运动下都拿到了最佳的：

- **3D Consistency**
- **Photo Consistency**
- **Style Consistency**

而在camera/object/content alignment上并不占优。  
这很重要：它说明该方法的能力跃迁主要发生在**几何一致性轴**，符合论文的目标，而不是泛泛地“所有控制都更强”。

**信号2：大相机运动更能说明问题。**  
Large setting里，很多依赖首帧点云或浅层几何先验的方法更容易出现：

- 几何撕裂
- 风格漂移
- 多视角错位

FANTASYWORLD的优势在这里更有说服力，因为它的几何表示不是“首帧静态外挂”，而是在视频生成过程中**持续演化**。

**信号3：消融实验说明收益不是来自更大模型，而是来自几何分支。**  
拿掉几何分支和双向交互后：

- Small: 3D Consist. **83.31 -> 79.77**
- Large: 3D Consist. **74.83 -> 72.06**

这基本回答了核心因果问题：  
**性能提升来自显式几何耦合，而不是普通camera conditioning或纯视频先验。**

**信号4：3DGS重建代理指标说明其视频latent更“可重建”。**  
在RealEstate10K上，固定相同VGGT初始化时：

- PSNR: **26.89 -> 28.24**
- SSIM: **0.84 -> 0.86**
- LPIPS: **0.17 -> 0.14**

这说明生成出来的视频不仅“看起来像”，还更容易被下游3D重建管线解释成一致结构。

### 一个值得注意的细节
作者还报告了直接用自己预测的点云做初始化的结果，虽有竞争力，但**仍低于使用VGGT初始化的版本**。  
这说明：

- 几何分支确实学到了有用3D结构
- 但它还**没有完全替代**强专用3D基础模型作为重建初始化器

也就是说，FANTASYWORLD已经把“视频latent可几何化”这件事做出来了，但“几何头本身就是最强3D重建器”这一点，证据还不够强。

### 局限性
- Fails when: 需要跨长时间持续维护同一世界状态、流式更新隐式3D记忆、或处理强动态/交互世界时；当前固定长度clip设定不适合这类场景。
- Assumes: 依赖冻结Wan2.1骨干、重建管线/Cut3R生成的深度/点图/相机监督，以及较重训练资源（Stage 1用64张H20约36小时，Stage 2用112张H20约144小时）；文中未提供公开代码链接。
- Not designed for: stylized视频的3D一致性评测、以camera/object/content alignment为主要目标的控制型视频生成、以及直接证明导航/novel-view synthesis下游泛化的完整任务评测。

### 可复用组件
这篇论文里最值得迁移的，不是完整系统，而是几个操作模式：

- **PCB式latent preconditioning**：先从冻结扩散骨干中取“半去噪”特征，再做结构预测
- **双向跨分支注意力**：让生成先验和结构监督在隐藏态里互相约束
- **面向扩散特征的反向DPT解码策略**：优先利用更深层、更干净的特征做几何预测

如果你以后要做“冻结大生成模型 + 结构/物理/3D分支”的方案，这三个组件都很有复用价值。

## Local PDF reference
![[paperPDFs/Building_World_Models_from_3D_Vision_Priors/arXiv_2025/2025_FantasyWorld_Geometry_Consistent_World_Modeling_via_Unified_Video_and_3D_Prediction.pdf]]