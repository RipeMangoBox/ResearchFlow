---
title: "VMem: Consistent Interactive Video Scene Generation with Surfel-Indexed View Memory"
venue: arXiv
year: 2025
tags:
  - Video_Generation
  - task/video-generation
  - diffusion
  - surfel-indexing
  - memory-retrieval
  - dataset/RealEstate10K
  - dataset/Tanks-and-Temples
  - opensource/no
core_operator: 以粗粒度 surfel 作为3D可见性索引，把历史视图绑定到被其观测到的表面上，并从目标视角检索最相关的少量参考帧来生成新视图
primary_logic: |
  单张场景图像与用户指定的下一段相机轨迹 → 用点图估计器把已生成帧写入 surfel 记忆、从目标视角渲染 surfel 并对历史帧索引投票选出 top-K 相关视图 → 将检索视图与目标相机条件送入图像集生成器，输出长时一致的交互式场景视频
claims:
  - "在 RealEstate10K 循环轨迹评测上，VMem (K=17) 相比 SEVA (K=17) 将 LPIPS 从 0.401 降到 0.304、PSNR 从 11.82 提升到 18.15，并将 Tdist 从 0.492 降到 0.165，说明回访场景时的一致性更强 [evidence: comparison]"
  - "在 Tanks-and-Temples 循环轨迹上，VMem (K=4) 在 LPIPS、PSNR 与 Tdist 上优于 LookOut、GenWarp、MotionCtrl 和 ViewCrafter，表明其在域外与更大相机运动下仍具优势 [evidence: comparison]"
  - "采用仅 4 个上下文视图的 VMem 可在保持接近 17 视图版本性能的同时，把推理速度从约 50 秒/帧降到 4.2 秒/帧，约 12× 加速 [evidence: ablation]"
related_work_position:
  extends: "SEVA (Zhou et al. 2025)"
  competes_with: "SEVA (Zhou et al. 2025); ViewCrafter (Yu et al. 2024)"
  complementary_to: "CUT3R (Wang et al. 2025)"
evidence_strength: strong
pdf_ref: paperPDFs/Memory_in_World_Model/arXiv_2025/2025_VMem_Consistent_Interactive_Video_Scene_Generation_with_Surfel_Indexed_View_Memory.pdf
category: Video_Generation
---

# VMem: Consistent Interactive Video Scene Generation with Surfel-Indexed View Memory

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2506.18903), [Project](https://v-mem.github.io/)
> - **Summary**: 论文提出一种基于 surfel 的几何索引记忆模块，把“只看最近几帧”改成“只看对当前视角最相关的历史视图”，从而显著提升单图驱动交互式场景视频在长程回访时的一致性。
> - **Key Performance**: RealEstate10K 循环轨迹上，VMem (K=17) 达到 LPIPS 0.304、PSNR 18.15（SEVA: 0.401 / 11.82）；K=4 版本推理约 4.2 s/帧，相比 SEVA 的 50 s/帧快约 12×。

> [!info] **Agent Summary**
> - **task_path**: 单张场景图像 + 用户交互指定短程相机轨迹 -> 长时一致的新视角场景视频
> - **bottleneck**: 固定短上下文窗口无法在回访区域取回真正相关的历史视图，导致长期场景一致性崩溃
> - **mechanism_delta**: 用 surfel 记录“哪些表面被哪些视图看过”，并从目标视角做遮挡感知检索，替代按时间或相机距离选参考帧
> - **evidence_signal**: 循环轨迹评测与检索策略消融都显示 surfel 检索显著优于 temporal / camera-distance / field-of-view 检索
> - **reusable_ops**: [surfel-indexed-memory, occlusion-aware-top-k-view-retrieval]
> - **failure_modes**: [dynamic-objects-or-nonstatic-scenes, point-map-estimation-errors-and-ood-natural-landscapes]
> - **open_questions**: [can-memory-and-geometry-be-learned-end-to-end, how-to-benchmark-true-long-horizon-revisitation-consistency]

## Part I：问题与挑战

这篇论文真正要解决的问题，不是“从单张图能不能生成新视角”，而是：

**当用户沿任意相机路径持续探索一个想象空间，甚至回到之前看过的位置时，模型能否还记得那个地方原本长什么样。**

### 1. 任务接口

- **输入**：一张起始场景图像 + 用户当前指定的下一小段相机轨迹
- **输出**：与历史内容一致、受相机控制的若干新帧，并可继续自回归扩展成长视频
- **交互设定**：每次只生成少量帧（文中常用 `M=4`），让用户看完再决定接下来往哪里走

### 2. 真正瓶颈是什么

现有方法主要有两类，但都卡在长期一致性：

1. **显式3D + 补全/outpainting 路线**  
   一边估计几何，一边补新视角。问题是深度、点图、拼接误差会不断累积，走远后画面会越来越漂。

2. **纯图像/视频条件生成路线**  
   不显式构建3D场景，直接把历史帧作为条件喂给生成器。问题是计算量太高，通常只能保留一个很短的上下文窗口，于是模型只“记得最近发生的事”，回到老地方就忘了。

所以**真瓶颈不是有没有记忆，而是：在算力固定时，如何从无限增长的历史中选出“对当前视角最有用”的那几帧。**

### 3. 为什么现在值得做

这类能力正好对应世界模型、游戏、VR、具身探索中的关键需求：  
**用户不是被动看一段视频，而是主动控制视角探索一个世界。**  
一旦“回头看”就穿帮，交互体验会立刻崩掉。

### 4. 边界条件

- 主要面向**静态场景**的新视角探索
- 假设用户给出**相机位姿**
- 目标是**一致的视频生成**，不是高精度3D重建
- 从论文实验看，训练/评测分布以室内真实场景为主，室外和动态物体不是主战场

---

## Part II：方法与洞察

VMem 的核心不是换一个更大的生成器，而是在生成器前面加一个**几何索引记忆层**。  
它把历史视图按“看见了哪些表面”组织起来，再按目标视角检索最相关的参考帧。

### 方法总览

#### 1. 写入记忆：把新帧挂到 surfel 上

每次生成出新视图后，方法会：

- 用现成点图估计器（文中用 **CUT3R**）估计新帧几何
- 将点图下采样成粗粒度表面元 **surfels**
- 每个 surfel 记录：
  - 位置
  - 法向
  - 半径
  - **哪些历史视图看过它**

如果新 surfel 与已有 surfel 在位置/法向上足够接近，就合并；否则新建。  
于是，场景被组织成一个“**表面 -> 观察过它的视图索引**”的记忆库。

#### 2. 读取记忆：从目标视角找最相关历史视图

当要生成下一批目标相机位姿时，VMem 不再取“最近 K 帧”，而是：

- 先把 surfel 记忆从目标视角（更准确地说是目标位姿的平均姿态）渲染出来
- 渲染时考虑深度关系和遮挡
- 每个像素会对应若干 surfel，也就对应一批“看过这些 surfel 的历史帧索引”
- 对这些索引做频次统计，取 **top-K** 历史视图

直观上，它在问：

> “为了生成当前这个画面，历史中哪些视图看过最多当前可见表面？”

这比“离得近”“时间最近”都更接近真正所需的信息。

#### 3. 生成新视图：把检索结果喂给图像集生成器

检索出的参考 RGB 和相机条件，一起送入图像集生成器。  
论文使用 **SEVA** 作为 backbone，并额外微调了一个更轻量的 `K=4, M=4` 版本。

因此，VMem 本质上是一个**plug-and-play 的 memory retrieval front-end**，而不是重新发明底层生成器。

### 核心直觉

**变化前**：参考视图按时间近邻或相机近邻选  
**变化后**：参考视图按“与当前可见表面重合程度”选

这改变了整个系统的信息瓶颈：

- 从前的瓶颈：**上下文窗口太短**
- 现在的瓶颈：**粗几何是否足够把真正相关的历史帧找回来**

而这正是 VMem 有效的原因：

1. **它只需要粗几何，不需要高精度重建**  
   几何在这里不是最终表示，只是检索索引。  
   所以点图有误差也不一定致命，只要还能把相关历史视图找出来即可。

2. **它显式利用了遮挡信息**  
   相机距离或视野重叠可能把“被墙挡住的区域”也当成相关；surfel 渲染则更接近“当前真的能看见什么”。

3. **它把长期记忆压缩成少量高价值上下文**  
   不必把所有历史帧都送入生成器，也不必只依赖最近几帧。

### 为什么这个设计在因果上成立

关键不是“记住更多帧”，而是**让生成器在该回忆的时候回忆到对的帧**。  
一旦参考帧真的来自同一片已见表面，生成器就更容易复用正确的纹理、布局和外观，从而在回访区域保持一致。

### 战略性 trade-off

| 设计选择 | 解决的瓶颈 | 收益 | 代价/风险 |
|---|---|---|---|
| 用 surfel 做粗粒度场景索引 | 历史视图无限增长，无法高效查找 | 检索快，且能建模遮挡 | 依赖点图估计质量 |
| 几何只用于检索，不用于最终渲染 | 显式3D误差累积 | 比 outpainting 式3D管线更稳健 | 仍可能因检索错帧而漂移 |
| top-K 相关视图替代最近 K 帧 | 时间邻近不等于内容相关 | 回访场景时更一致 | 需要额外 memory read/write 过程 |
| K=4 轻量 generator + memory | 交互式推理成本高 | 约 12× 加速 | 轻量模型的上限仍受 backbone 限制 |

---

## Part III：证据与局限

### 关键证据

#### 1. 最强证据：循环轨迹评测真正测到了“回访一致性”
作者指出，常规 benchmark 的轨迹很少回到看过的地方，所以并不能充分暴露长期记忆问题。  
因此他们构造了**cycle trajectory**：先沿轨迹走出去，再沿同一路径反向走回来。

这项评测最能支持论文主张：

- 在 **RealEstate10K cycle** 上，`VMem (K=17)` 相比 `SEVA (K=17)`：
  - LPIPS：**0.304 vs 0.401**
  - PSNR：**18.15 vs 11.82**
  - Tdist：**0.165 vs 0.492**

这说明能力跃迁主要发生在：**重新看到旧区域时，模型还能保持同一场景身份。**

#### 2. 泛化信号：在 Tanks-and-Temples 上仍优于多种基线
在更大相机运动、含室内外场景的 **Tanks-and-Temples cycle** 上，`VMem (K=4)` 仍在 LPIPS / PSNR / Tdist 上优于 LookOut、GenWarp、MotionCtrl、ViewCrafter。  
这说明它不是只对单一室内轨迹有效。

#### 3. 因果信号：消融证明“检索机制”才是关键旋钮
作者把 surfel 检索与三种替代策略比较：

- temporal retrieval
- camera-distance retrieval
- field-of-view retrieval

结果显示，无论 `K=17` 还是 `K=4`，**VMem 的 surfel 检索都最好或最稳**。  
这很重要，因为它证明提升不是简单来自“换了 backbone”或“加了更多算力”，而是来自**检索准则本身**。

#### 4. 效率信号：少参考帧也能保住性能
`K=4` 轻量版配合 VMem，单帧推理约 **4.2 秒**，而原始 SEVA 约 **50 秒**，接近 **12× 加速**。  
这说明记忆模块不仅提升一致性，还把“长期历史”压缩成了可交互的计算量。

### 局限性

- **Fails when**: 动态物体较多、场景非静态、自然景观等域外输入、点图估计严重失准、或存在论文未充分覆盖的复杂遮挡时，surfel 检索可能失效并带来一致性崩溃。
- **Assumes**: 假设用户提供相机轨迹；依赖外部点图估计器 CUT3R 与图像集生成器 SEVA；轻量版仅在 RealEstate10K 上 LoRA 微调；训练成本不低（文中为 8×A40、60 万 iterations）；正文未明确代码/权重开放状态。
- **Not designed for**: 实时 VR 级别部署、精确3D重建、动态世界状态建模、物理一致的交互仿真。

补充两点很实际的限制：

1. **评测本身还不够成熟**  
   作者也承认，cycle trajectory 只是回访问题的 proxy，遮挡复杂度仍有限；现有指标也更偏低层纹理相似，而不是真正的多视角世界一致性。

2. **速度仍未达到实时**  
   即使轻量版也约 4.16 s/帧，离沉浸式交互仍有距离。

### 可复用组件

这篇论文最值得复用的，不是某个特定 backbone，而是下面几个操作模式：

- **surfel-indexed memory**：把历史信息按“表面被谁看过”组织
- **occlusion-aware retrieval**：从目标视角渲染记忆并做 top-K 投票
- **geometry-as-index, not renderer**：用粗几何做检索，而不是把几何误差直接传到最终渲染
- **pose-level NMS / 去冗余记忆**：避免反复采样同一区域、提升覆盖率

如果以后有更强的 image-set generator 或更稳的 point-map predictor，这套 memory 机制大概率还能直接受益。

## Local PDF reference

![[paperPDFs/Memory_in_World_Model/arXiv_2025/2025_VMem_Consistent_Interactive_Video_Scene_Generation_with_Surfel_Indexed_View_Memory.pdf]]