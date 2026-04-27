---
title: "OmniWorld: A Multi-Domain and Multi-Modal Dataset for 4D World Modeling"
venue: arXiv
year: 2025
tags:
  - Survey_Benchmark
  - task/video-understanding
  - task/video-generation
  - synthetic-data
  - multi-modal-annotation
  - benchmark-construction
  - dataset/OmniWorld
  - dataset/OmniWorld-Game
  - opensource/full
core_operator: 以自采游戏视频和多域公开视频为底座，自动补齐深度、位姿、文本、光流与前景掩码等模态，并建立统一的4D世界建模双任务基准
primary_logic: |
  4D世界建模评测目标 → 汇集模拟器/机器人/人类/互联网四域视频并生成深度、相机位姿、文本、光流、前景掩码 → 在3D几何预测与相机可控视频生成上统一评测并用微调验证训练价值 → 揭示现有方法在长时序、高动态、复杂相机运动场景下的能力边界
claims:
  - "Claim 1: OmniWorld 汇集 12 个异构数据集、60万+ 视频序列和 3亿+ 帧，其中 OmniWorld-Game 提供 96K clips 与 18.5M frames，并在表1中比现有公共合成数据集覆盖更完整的深度/位姿/文本/光流/前景掩码模态 [evidence: comparison]"
  - "Claim 2: 在 OmniWorld-Game benchmark 上，没有任何单一 3D 几何基础模型同时在单目深度与视频深度全部指标中占优，且相机可控视频生成模型普遍难以兼顾轨迹遵循与视频质量，说明该基准能暴露高动态长时序场景下的时空一致性瓶颈 [evidence: analysis]"
  - "Claim 3: 使用 OmniWorld 微调后，DUSt3R、CUT3R、Reloc3r 和 AC3D 在多个外部基准与 OmniWorld-Game 上都获得可测提升，例如 AC3D 在 OmniWorld-Game 上 CamMC 从 6.6965 降至 4.4854 [evidence: comparison]"
related_work_position:
  extends: "N/A"
  competes_with: "TartanAir (Wang et al. 2020); RealEstate10K (Zhou et al. 2018)"
  complementary_to: "DUSt3R (Wang et al. 2024c); AC3D (Bahmani et al. 2024)"
evidence_strength: moderate
pdf_ref: paperPDFs/Evaluating_World_Models/arXiv_2025/2025_OmniWorld_A_Multi_Domain_and_Multi_Modal_Dataset_for_4D_World_Modeling.pdf
category: Survey_Benchmark
---

# OmniWorld: A Multi-Domain and Multi-Modal Dataset for 4D World Modeling

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2509.12201)；论文首页标注了 GitHub / Data / Homepage，但提取文本未保留具体 URL
> - **Summary**: 这篇工作不是提出新的 world model，而是构建一个覆盖模拟器、机器人、人类、互联网四域、带几何与语义多模态标注的统一数据底座与 benchmark，用来更真实地诊断并提升 4D 世界建模能力。
> - **Key Performance**: OmniWorld 总规模为 600K+ 序列、300M+ 帧；AC3D 经 OmniWorld 微调后在 OmniWorld-Game 上 CamMC 由 6.70 降到 4.49，FVD 由 1745.8 降到 1437.2。

> [!info] **Agent Summary**
> - **task_path**: 多域原始视频序列 -> 多模态补标注数据集 -> 3D几何预测 / 相机可控视频生成 benchmark
> - **bottleneck**: 缺少同时覆盖长时序、高动态、复杂相机运动与显式几何监督的数据分布，导致现有 world modeling 评测过于“短、静、单域”
> - **mechanism_delta**: 把数据构建从单域静态数据集升级为“多域采集 + 质量切片 + 自动多模态标注 + 双任务统一评测”
> - **evidence_signal**: 现有 SOTA 在 OmniWorld-Game 上明显暴露短板，且用 OmniWorld 微调后可在多个外部基准上持续提升
> - **reusable_ops**: [video-slicing-filter, foreground-aware-two-stage-camera-pose-annotation]
> - **failure_modes**: [synthetic-to-real-gap, auto-annotation-noise-in-fast-dynamic-scenes]
> - **open_questions**: [which-modalities-contribute-most-to-transfer, how-to-cover-non-visual-physical-signals]

## Part I：问题与挑战

这篇论文要解决的真问题，不是“再做一个更大的视频集”，而是 **4D world modeling 缺少真正能同时测几何、动态和相机控制的数据接口**。

### 1. 真实瓶颈是什么
现有 world model 相关任务已经开始分化出两条核心能力线：

1. **3D geometric foundation models**：从图像/视频恢复稳定几何。
2. **camera-controlled video generation**：在生成视频时遵守相机轨迹与时空约束。

但这两类模型常用的数据各自都很窄：

- 几何 benchmark 往往 **序列短、运动小、场景单一**。  
  例如文中指出 Sintel 平均只有约 50 帧，KITTI/Bonn/NYU-v2 也都强烈偏某类场景。
- 可控视频生成数据往往 **主体动态弱、相机轨迹平滑、几何标注缺失**。  
  如 RealEstate10K 主要是静态场景摄影式轨迹。

于是模型很容易学到“捷径”：
- 靠短时局部纹理匹配过几何评测；
- 靠静态场景先验过相机控制评测；
- 却不需要真正建模 **物体运动 + 相机运动 + 场景几何** 的联合一致性。

### 2. 为什么现在值得解决
因为现在模型侧已经到了一个拐点：

- 3D GFM 已经出现 DUSt3R、CUT3R、VGGT、MoGe 等代表方法；
- 相机可控视频生成也已有 AC3D、CamCtrl、MotionCtrl、CAMI2V 等方法。

**模型原型已经有了，但数据分布还停留在旧时代。**  
所以现在的上限，越来越像是被数据与评测设计卡住，而不是被网络结构本身卡住。

### 3. 输入/输出接口与边界
**输入**：来自四个域的原始视频  
- Simulator
- Robot
- Human
- Internet

**输出**：统一的 4D world modeling 数据接口  
- depth
- camera poses
- text captions
- optical flow
- foreground masks

并在其上建立两个 benchmark：
- **3D geometric prediction**
- **camera-controlled video generation**

**边界条件**：
- 重点是视觉 4D 建模，不覆盖音频、触觉、力反馈等非视觉物理信号。
- benchmark 的核心压力测试来自 **OmniWorld-Game**，因此其挑战性很强，但也天然带有一部分 synthetic bias。
- 作者主要证明的是“数据资源价值”，不是提出新的 SOTA 模型。

---

## Part II：方法与洞察

这篇工作的关键不是新网络，而是 **改变训练分布与测量分布**。

### 方案结构

#### 1. 数据层：一个“合成核” + 多域真实补充
OmniWorld 由 12 个异构数据集组成，覆盖四个域，总量超过 60 万序列、3 亿帧。

其中最关键的是 **OmniWorld-Game**：
- 96K clips
- 18.5M frames
- 720P
- 混合场景、强动态、第一/第三人称视角
- 同时提供深度、位姿、文本、光流、前景掩码

它相当于一个 **高精度、高动态、多模态的 synthetic core**。  
然后再用 robot / human / internet 数据，把分布往真实世界拉宽。

#### 2. 清洗层：先做 video slicing，再做标注
作者不是把原视频直接喂进去，而是先做质量过滤，去掉：
- motion blur
- 特征不足
- 动态区域过大
- 不适合几何或运动分析的片段

这一步很重要，因为它决定后续自动标注是否稳定。

#### 3. 标注层：按模态分治
他们没有用一个“大一统标注器”，而是按模态做专门流水线：

- **Depth**
  - 游戏环境：直接从渲染过程拿 depth
  - 公共数据：用 Prior Depth Anything / FoundationStereo 补齐或优化
- **Foreground masks**
  - robot 数据用 RoboEngine + SAM 2
  - game 数据用 Grounding DINO + SAM
- **Camera poses**
  - 先利用前景 mask 剔除动态主体干扰
  - 再做 coarse pose（VGGT 或 DroidCalib）
  - 再用 CoTracker / feature tracking + BA refine
- **Text captions**
  - 用 Qwen2-VL-72B 分段生成长描述
- **Optical flow**
  - 用 DPFlow 直接在原分辨率上预测

这里最有代表性的可复用算子，是 **foreground-aware 的两阶段位姿标注**：  
先把动态主体影响降掉，再做位姿估计与优化，这对动态视频尤其关键。

#### 4. 基准层：统一压测两类能力
作者把 benchmark 分成两部分：

- **3D 几何预测**
  - mono-depth
  - video-depth
  - 指标：Abs Rel, δ<1.25，且区分 scale / scale&shift 对齐
- **相机可控视频生成**
  - 指标：RotErr, TransErr, CamMC, FVD

重点不在发明新指标，而在于 **把评测数据本身换成更难、更真实的长时序高动态分布**。

### 核心直觉

**改变了什么**：  
`短序列 / 静态场景 / 单域数据 / 弱几何标注`  
→ `长序列 / 高动态 / 多域分布 / 强几何与语义多模态标注`

**改变了哪个瓶颈**：  
过去模型可以依赖：
- 平滑相机轨迹先验
- 静态场景假设
- 局部匹配与短程一致性

而 OmniWorld 把问题改成必须同时处理：
- scene geometry
- object motion
- camera motion
- temporal consistency

**能力上发生了什么变化**：  
这使 benchmark 不再只测“能不能在熟悉分布上看起来有效”，而是能测：
- 几何在长时序中是否持续稳定；
- 生成模型是否真能跟随复杂相机轨迹；
- 模型是否能在动态主体与复杂环境交互下维持时空一致性。

**为什么这套设计有效**：
1. **合成数据**提供高精度 dense geometry supervision；  
2. **真实多域数据**提供泛化所需的分布跨度；  
3. **自动补标注流水线**把原本碎片化的数据统一成同一训练接口；  
4. **双任务 benchmark**让“几何建模能力”和“可控生成能力”共享同一世界建模压力测试。

### 战略权衡

| 设计选择 | 带来的收益 | 代价 / 风险 |
|---|---|---|
| 以游戏数据为核心 | 可获得高精度深度、位姿和复杂动态 | 存在 sim-to-real gap，且受游戏版权/使用条款约束 |
| 多域数据统一整合 | 提升场景与主体分布多样性 | 各域模态完整性不均，标注一致性难统一 |
| 自动化补标注 | 才能扩到 300M+ 帧规模 | 标注误差会级联传播，尤其在极端动态/弱纹理场景 |
| 长时序高分辨率 benchmark | 更能暴露真实 4D 建模短板 | 评测成本高，部分模型甚至吃不下完整序列 |

---

## Part III：证据与局限

### 关键证据

#### 信号 1：benchmark 确实能“揭短”
在 **OmniWorld-Game 几何 benchmark** 上：
- 单目深度里 **MoGe-2** 最好；
- 视频深度里 **VGGT** 最好；
- 但**没有任何一个模型能同时统治所有设置**。

这说明 OmniWorld-Game 不是在复读旧 benchmark，而是在逼模型同时面对：
- 单帧几何精度
- 多帧时序一致性
- 动态场景鲁棒性

也就是说，旧 benchmark 上的“强”，到这里不再自动成立。

#### 信号 2：可控视频生成在这个分布上明显更难
在 **camera-controlled video generation** 上：
- I2V 里 **CamCtrl** 最强；
- 但整体上各模型都难以同时兼顾相机控制精度和视频质量；
- T2V 的 **AC3D** 在 OmniWorld-Game 上 FVD 很高，说明“文本 + 复杂动态 + 相机轨迹”联合建模仍很不成熟。

所以这个 benchmark 的价值不只是排序，而是指出：  
**当前 controllable video 主要还是建立在静态摄影数据假设上。**

#### 信号 3：OmniWorld 不只是测得难，也真的能拿来训
最有说服力的证据，是作者不是只做 benchmark，而是把它拿来微调现有方法：

- **AC3D** 在 OmniWorld-Game 上  
  - CamMC: **6.6965 → 4.4854**
  - FVD: **1745.778 → 1437.247**
- **Reloc3r** 在 DynPose-100K 上  
  - AUC@10: **15.4 → 25.5**
- **CUT3R** 在 KITTI 视频深度上也有稳定提升

这意味着 OmniWorld 的价值不是“更难的考试题”，而是 **更有效的训练分布**。  
能力跃迁点在于：它把模型从“适应窄域静态数据”推向“适应动态、长时序、跨域的几何-运动联合建模”。

### 局限性

- **Fails when**: 需要音频、触觉、力学状态、动作闭环等非视觉物理信号时，OmniWorld 不能完整覆盖；在极端遮挡、超高速运动、弱纹理背景下，自动位姿/深度补标注也可能不稳定。
- **Assumes**: 依赖 ReShade/OBS、Qwen2-VL-72B、SAM/SAM2、Grounding DINO、VGGT/DroidCalib、CoTracker、DPFlow 等外部工具链；微调实验依赖 8×A800；部分游戏内容还受非商业与版权合规约束。
- **Not designed for**: 交互式策略学习、强化学习环境闭环评测、音频驱动 world modeling、显式物理参数辨识。

### 可复用组件

1. **video slicing quality filter**：适合任何大规模视频几何/运动数据清洗。  
2. **foreground-aware two-stage pose annotation**：对动态视频位姿标注很有通用价值。  
3. **domain-specific caption prompting**：适合把长视频切段生成密集文字描述。  
4. **统一双任务 benchmark 设计**：可复用于“重建 + 生成”联合评测范式。

**一句话结论**：  
OmniWorld 的真正贡献，不是把数据做“大”，而是把 world modeling 的评测分布从“容易被捷径骗过”改成“必须真正建模 4D 时空结构”——这也是它对后续通用 world model 最有价值的地方。

![[paperPDFs/Evaluating_World_Models/arXiv_2025/2025_OmniWorld_A_Multi_Domain_and_Multi_Modal_Dataset_for_4D_World_Modeling.pdf]]