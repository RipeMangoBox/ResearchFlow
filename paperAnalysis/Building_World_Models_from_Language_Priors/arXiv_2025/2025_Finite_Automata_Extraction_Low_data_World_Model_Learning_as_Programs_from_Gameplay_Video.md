---
title: "Finite Automata Extraction: Low-data World Model Learning as Programs from Gameplay Video"
venue: arXiv
year: 2025
tags:
  - Others
  - task/world-model-learning
  - task/video-generation
  - neuro-symbolic
  - program-synthesis
  - domain-specific-language
  - "dataset/Pac-Man"
  - "dataset/River Raid"
  - opensource/no
core_operator: "先用自监督视觉把 gameplay 视频离散成 sprite 网格，再在 Retro Coder DSL 上为每类 sprite 搜索 if-then 程序，得到可编辑的低数据世界模型。"
primary_logic: |
  gameplay 视频帧（仅视频、无模拟器） → 学习 sprite 字典并提取离散网格状态 →
  对手工选定的可学习 sprite 独立搜索 Retro Coder 条件-动作程序 →
  输出可解释的程序化世界模型，并用于下一帧与滚动预测
claims:
  - "在 Pac-Man 与 River Raid 的低数据设定下，FAE 在两种评测协议中都显著优于 GameGAN 的 FID 与 Prediction Error，例如 Pac-Man test FID 为 0.28/0.29，而 GameGAN 为 10.46；River Raid test FID 为 0.25/0.13，而 GameGAN 为 58.81 [evidence: comparison]"
  - "FAE 用显著更短的程序达到与 Game Engine Learning 可比的前端预测质量：Pac-Man 平均条件数 1.75±0.83 vs 73±22.55，River Raid 为 2.00±1.87 vs 125.42±19.58 [evidence: comparison]"
  - "FAE 能恢复部分确定性局部规则（如 Pac-Man 中敌人追逐玩家、道具被吃后消失），但无法正确建模随机、时间依赖、稀有或尺寸变化的 sprite 行为 [evidence: case-study]"
related_work_position:
  extends: "Game Engine Learning from Video (Guzdial et al. 2017)"
  competes_with: "GameGAN (Kim et al. 2020); Game Engine Learning from Video (Guzdial et al. 2017)"
  complementary_to: "N/A"
evidence_strength: moderate
pdf_ref: paperPDFs/Building_World_Models_from_Language_Priors/arXiv_2025/2025_Finite_Automata_Extraction_Low_data_World_Model_Learning_as_Programs_from_Gameplay_Video.pdf
category: Others
---

# Finite Automata Extraction: Low-data World Model Learning as Programs from Gameplay Video

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2508.11836)
> - **Summary**: 该文把少量 gameplay 视频先压成 sprite 网格，再为每类 sprite 搜索可解释 DSL 规则，从而学习可编辑的程序化世界模型，而不是只学一个黑盒视频预测器。
> - **Key Performance**: Pac-Man（Approach 1）test FID 0.28 vs GameGAN 10.46；平均条件数仅 1.75±0.83，对比 GEL 的 73±22.55

> [!info] **Agent Summary**
> - **task_path**: gameplay RGB 视频 -> sprite-grid 符号状态 + per-sprite DSL 程序 -> 可编辑世界模型/下一帧预测
> - **bottleneck**: 低数据下既要学到环境动态，又要保持代码级可解释与可编辑；纯神经模型太吃数据，纯规则搜索又容易得到冗长规则
> - **mechanism_delta**: 先做对象级离散化，再把全局环境学习拆成单 sprite 的局部程序搜索
> - **evidence_signal**: 在 Pac-Man 与 River Raid 上，FAE 显著优于 GameGAN，并以 1~2 条平均条件达到与 GEL 可比或更优的 FID
> - **reusable_ops**: [sprite网格离散化, 单实体DSL邻域搜索]
> - **failure_modes**: [稀有sprite或随机行为学不到稳定程序, 多尺寸/旋转/跨格sprite会破坏网格表示]
> - **open_questions**: [如何去掉人工筛选sprite, 如何把时间性随机性与GameManager逻辑纳入DSL]

## Part I：问题与挑战

这篇论文真正要解决的，不是“再做一个更大的游戏视频生成模型”，而是：**能不能只看少量 gameplay 视频，在没有模拟器、没有状态标注的条件下，学出一个开发者可读、可改、可复用的世界模型程序**。

### 问题定义
- **输入**：游戏 RGB 视频帧。
- **输出**：  
  1. 一个离散的 sprite 字典与网格状态表示；  
  2. 每个 sprite 的行为程序；  
  3. 基于这些程序的下一帧/滚动预测能力。

### 真正瓶颈
现有路线各有硬伤：
- **纯神经 world model**：能预测画面，但通常是黑盒、数据重、算力重，也不适合开发者直接修改规则。
- **纯 DSL / 规则枚举**：可解释，但如果直接从像素或全局规则空间搜索，容易爆炸，最后得到很长、很碎的条件组合，泛化弱。

因此这篇文章的核心 bottleneck 是：  
**如何把像素视频先压缩成足够稳定的对象级状态，再在足够小、足够贴近游戏机制的程序空间里搜索规则。**

### 为什么现在值得做
游戏 world model 正在从 RL 辅助工具转向“可交互游戏生成/创建工具”。一旦目标从“让 agent 玩得好”变成“让开发者能编辑规则”，表示形式就很关键了。代码式 world model 在这个语境下比神经 latent state 更有产品和分析价值。

### 边界条件
这篇方法明显依赖以下场景：
- 2D、复古、sprite 主导的游戏；
- 行为大多可被局部碰撞/移动/变身规则描述；
- 背景相对固定或可裁剪；
- 某些 sprite 需要人工指定是否学习其程序；
- 玩家控制对象通常不在可学习程序内。

---

## Part II：方法与洞察

方法由两段组成：**先离散化表示，再做程序搜索**。

### 1. 表示抽取：从像素到 sprite 网格
作者先用改造版 **MarioNette/Marionette** 学一个有限大小的 sprite dictionary，再把每帧转成一个 `w × h` 的离散网格，每个格子对应一个 sprite token。

他们做了两个重要改动：
- **支持指定静态背景**：从输入/输出中减掉背景，让模型更专注 sprite；
- **提高重建损失权重**：希望先把“看清 sprite”这件事做好，因为后续程序学习完全依赖这个离散表示。

这一步的作用不是追求最强视觉重建，而是把原始像素序列变成**适合符号搜索的状态空间**。

### 2. Retro Coder：把可学习行为限制成一个小而实用的 DSL
作者设计了一个面向 2D 复古游戏的 DSL：**Retro Coder**。  
每个 sprite 的程序都由 if-then 规则组成，条件和动作主要覆盖：
- 是否存在某类实体；
- 是否相邻/碰撞；
- 朝某个实体或方向移动；
- 变成另一类实体；
- 朝某个目标位置移动。

这相当于给世界模型注入了一个强先验：  
**游戏里的很多规则，本质上就是“在某个局部条件下，实体执行一个动作或状态转移”。**

### 3. Finite Automata Extraction：单 sprite、局部邻域的程序搜索
FAE 不直接学全局程序，而是：
1. 对每个目标 sprite 单独处理；
2. 找出它在视频中出现的帧；
3. 在短窗口内衡量“当前程序预测的下一帧网格”和真实下一帧网格之间的误差；
4. 通过“加一条 if-then / 删一条 if-then”的邻域搜索，逐步改进程序。

这一步的关键不是搜索技巧本身多复杂，而是**搜索空间被前两步压得足够小**：
- 状态已经是离散 sprite 网格；
- 程序只针对一个 sprite；
- DSL 只保留与游戏机制强相关的原语。

### 核心直觉

**变化是什么**：  
作者把学习目标从“像素级黑盒动态建模”改成了“对象级、单实体、受限 DSL 的程序提取”。

**哪个瓶颈被改变了**：  
- 像素序列的高熵状态空间 → 低熵的 sprite 网格状态空间  
- 全局规则组合爆炸 → 单 sprite 的局部搜索  
- 通用函数逼近 → 带机制先验的 if-then 程序空间

**能力因此如何变化**：  
- 小数据下更容易学到稳定规则；
- 结果可以直接读成代码；
- 程序长度更短，更接近“可编辑规则”而不是“条件枚举器”。

**为什么这设计有效**：  
因为复古 2D 游戏里的大量行为确实符合这种归纳偏置：  
“某个对象在局部邻域内，根据附近对象和存在性条件做位移/碰撞/变身。”  
当真实机制和 DSL 原语足够对齐时，就不需要大模型、大数据去近似一个黑盒动力学。

### 战略取舍

| 设计选择 | 带来的收益 | 代价/风险 |
|---|---|---|
| sprite 网格离散化 | 显著降低状态复杂度，碰撞可离散建模 | 跨格、旋转、多尺寸 sprite 容易表示错 |
| 每个 sprite 独立学习程序 | 搜索空间小，结果更可编辑 | 难建模全局生成器、跨实体协调逻辑 |
| 受限 DSL | 规则短、可解释、低数据友好 | 表达不了随机性、时间依赖、复杂生成机制 |
| 固定背景与裁剪预处理 | 提升 sprite 抽取质量 | 依赖场景先验，泛化到复杂背景更难 |

---

## Part III：证据与局限

### 关键实验信号

**信号 1｜低数据下，结构先验明显优于纯神经像素世界模型**  
对比 GameGAN，FAE 在 Pac-Man 和 River Raid 两个域上都大幅领先。  
最有代表性的不是单个数值，而是趋势：**GameGAN 在这种小数据设定下基本学不稳，而 FAE 能稳定给出低 FID 的预测。**  
这说明对象级离散化 + DSL 搜索，比直接学像素动态更适合 low-data world model。

**信号 2｜与 DSL 基线相比，FAE 的“代码更短”，但前端表现不差**  
Game Engine Learning（GEL）在某些 FID 上与 FAE 接近，甚至局部更好，但作者指出 GEL 往往没有真正学到可迁移动态，而是退化为“复制前一帧”式行为，并且未收敛。  
更关键的是，FAE 的平均条件数只有：
- Pac-Man：**1.75±0.83**
- River Raid：**2.00±1.87**

而 GEL 分别需要：
- Pac-Man：**73±22.55**
- River Raid：**125.42±19.58**

这说明 FAE 的优势不只是“能预测”，而是**用更短的规则表达得到相近的前端效果**。

**信号 3｜可解释性是真实存在的，但只在 DSL 覆盖到的机制上成立**  
定性案例里，FAE 能学到一些人类可读规则，例如：
- 敌人追逐玩家；
- 道具被吃后消失；
- 子弹向上移动。

但一旦遇到：
- 稀有 sprite；
- 随机行为；
- 依赖时间的状态切换；
- 屏幕外生成/刷怪；
- 不同尺寸或旋转的 sprite；

程序就开始失败。这说明它确实在提取“程序”，不是在做万能近似器。

### 1-2 个最关键指标
- **Pac-Man, Approach 1, test FID**：FAE 0.28，GameGAN 10.46  
- **程序复杂度**：FAE 平均仅 1.75~2.00 条条件，远低于 GEL 的 73~125 条

### 局限性
- **Fails when**: sprite 稀有、行为随机、依赖时间、需要全局刷怪/生成器逻辑，或 sprite 跨格/多尺寸/旋转导致网格表示失真时，程序学习会失败或学出伪条件。
- **Assumes**: 需要先学到足够稳定的 sprite 字典；依赖静态背景/裁剪等预处理；依赖人工挑选哪些 sprite 应该学习程序；默认局部、确定性的 2D 规则足以解释主要动态。
- **Not designed for**: 玩家控制策略学习、3D 或连续物理、复杂全局事件调度、带显式随机过程的游戏机制。

### 资源与可复现性备注
- FAE 的训练时间约 **800s**，明显少于 GameGAN 的 **7200s**，但仍高于 GEL。
- 结果高度依赖前端表示质量；如果 MarioNette 抽错 sprite，后端程序几乎一定被污染。
- 论文未给出代码或项目链接，当前应视为 **opensource/no**。
- 评测只覆盖两个游戏域，且没有系统 ablation，因此证据强度应保守看作 **moderate**。

### 可复用组件
- **sprite-grid 离散化**：把视频世界模型问题变成对象级状态建模；
- **per-entity DSL 搜索**：把世界模型分解成可编辑的局部程序；
- **短窗口邻域优化**：用局部预测误差引导程序增删；
- **表示先于规则**：先把视觉状态压到稳定符号层，再谈程序抽取。

## Local PDF reference

![[paperPDFs/Building_World_Models_from_Language_Priors/arXiv_2025/2025_Finite_Automata_Extraction_Low_data_World_Model_Learning_as_Programs_from_Gameplay_Video.pdf]]