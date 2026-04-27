---
title: "COSMO: Combination of Selective Memorization for Low-cost Vision-and-Language Navigation"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/vision-language-navigation
  - state-space-model
  - hybrid-architecture
  - dual-stream
  - dataset/REVERIE
  - dataset/R2R
  - dataset/R2R-CE
  - opensource/no
core_operator: 先用面向 VLN 的状态空间模块按指令筛选并压缩视觉历史，再用 Transformer 在过滤后的候选上做精确动作决策。
primary_logic: |
  自然语言指令 + 当前全景观测 + 历史拓扑图
  → 用 RSS 在单次扫描中建模全景视角间全局关系、用 CS3 进行双流跨模态选择性记忆
  → 用 Transformer / GASA / Cross-Attention 在已过滤的局部与全局候选上做动作选择
  → 输出下一步导航动作或停止决策
claims:
  - "在 REVERIE val unseen 上，COSMO 相比 DUET 将 SR 从 46.98 提升到 50.81、SPL 从 33.73 提升到 35.93，同时参数量仅 28M（DUET 为 181M）、FLOPs 仅 0.46G（DUET 为 4.95G）[evidence: comparison]"
  - "在 R2R-CE test unseen 上，COSMO 相比 DUET 将 SR 从 42 提升到 47、SPL 从 36 提升到 40，说明该低成本结构可迁移到连续环境设置 [evidence: comparison]"
  - "直接把单流 Mamba 用于 VLN 会使 REVERIE val unseen 的 SR 降到 32.25，而去掉 RSS 或 CS3 会带来约 3-4 个点的 SR/SPL 下降，支持 VLN 定制化 SSM 与混合架构的必要性 [evidence: ablation]"
related_work_position:
  extends: "DUET (Chen et al. 2022)"
  competes_with: "DUET (Chen et al. 2022); BEVBert (An et al. 2023)"
  complementary_to: "KERM (Li et al. 2023); Bird’s-eye-view Scene Graph (Liu et al. 2023)"
evidence_strength: strong
pdf_ref: paperPDFs/Vision_Language_Navigation_VLN_Models_2025/arXiv_2025/2025_COSMO_Combination_of_Selective_Memorization_for_Low_cost_Vision_and_Language_Navigation.pdf
category: Embodied_AI
---

# COSMO: Combination of Selective Memorization for Low-cost Vision-and-Language Navigation

> [!abstract] **Quick Links & TL;DR**
> - **Summary**: 这篇工作把 VLN 中最耗算力的“历史视觉记忆 + 跨模态融合”前移为两个定制状态空间模块（RSS、CS3）做选择性过滤，再保留 Transformer 负责最终动作选择，从而在显著降本的同时维持甚至提升导航表现。
> - **Key Performance**: REVERIE val unseen 上 SR/SPL = **50.81/35.93**（DUET: 46.98/33.73）；FLOPs 仅 **0.46G**，约为 DUET 的 **9.3%**。

> [!info] **Agent Summary**
> - **task_path**: 自然语言指令 + 当前全景观测 + 历史拓扑图 -> 下一步局部/全局导航动作或停止
> - **bottleneck**: 纯 Transformer 处理长历史与跨模态交互成本高，而纯 SSM 又难以处理全景空间关系与显式动作候选选择
> - **mechanism_delta**: 把 SSM 放到决策前端做“指令条件下的选择性记忆”，并用 RSS 解决单节点空间建模、用 CS3 解决双流跨模态选择，再交给 Transformer 做精确选路
> - **evidence_signal**: 三个 VLN 基准上的比较结果 + RSS/CS3/结构级消融共同表明其在更低参数和 FLOPs 下保持或提升 SR/SPL
> - **reusable_ops**: [round selective scan, dual-stream cross-modal selective state space]
> - **failure_modes**: [需要最后几步精细靠近目标物体时不够稳, 相似走廊或房间导致指令歧义时容易选错分支]
> - **open_questions**: [后端 Transformer 是否还能继续压缩, 与外部知识或 BEV 表征结合后是否仍能保持同等效率优势]

## Part I：问题与挑战

这篇论文研究的是 **Vision-and-Language Navigation (VLN)**：给定自然语言指令，智能体在室内环境中结合当前全景观测和历史轨迹，选择下一个可导航节点或停止。

### 真正的问题是什么？

作者认为，VLN 的核心难点不只是“看懂指令”，而是：

1. **历史观测会快速膨胀**  
   路越长、指令越长，智能体积累的视角、节点、候选动作越多。若全部交给 Transformer 做全注意力，计算成本会上升，而且噪声也会增加。

2. **VLN 不是单纯序列建模，而是“有条件地记住什么”**  
   智能体需要只保留与当前指令相关的视觉记忆，而不是把所有历史都均匀编码。

3. **纯 SSM 不够，纯 Transformer 太贵**  
   SSM 擅长长序列线性复杂度建模，但原生更适合单流 1D 序列；VLN 却要求：
   - 在全景视图中建模空间关系；
   - 在长短明显不均衡的文本/视觉之间做跨模态对齐；
   - 在每一步从候选动作中显式选择一个动作。  
   作者实验表明，直接把 Mamba 套到 VLN 上会明显掉点。

### 为什么现在值得做？

因为近年的 VLN 提升路径大多是：
- 加更强地图；
- 加外部知识；
- 加更多视觉信号；
- 甚至上 LLM / 大规模训练语料。

这些方法通常能涨性能，但也显著增大模型和推理成本。对于真实机器人或家居助手，**更低成本的推理结构**有现实意义。因此，作者试图把 SSM 的线性复杂度优势引入 VLN，但不是硬套，而是做任务定制。

### 输入/输出接口与边界条件

- **输入**：自然语言指令 + 当前节点的全景观测 + 历史拓扑图/访问轨迹
- **输出**：下一步导航动作（局部 view / 全局 node）或 stop
- **环境边界**：
  - 主要在 Matterport3D 室内环境上评测
  - 覆盖离散环境（R2R, REVERIE）和基于 waypoint predictor 的连续环境变体（R2R-CE）
- **非目标**：
  - 不是原始低层连续控制
  - 不是开放世界户外导航
  - 也不是依赖大模型知识增强的通用 embodied agent

---

## Part II：方法与洞察

COSMO 的设计哲学可以概括为一句话：

> **先决定“该记住什么”，再决定“该走哪一步”。**

也就是把 VLN 分成两个阶段：
- 前端：用定制 SSM 做 **selective memorization**
- 后端：用 Transformer 做 **action decision**

### 核心直觉

VLN 的瓶颈并不是缺少更大的模型，而是**历史记忆里有太多无关信息**，而最终动作选择又必须保持精确。

- **改变了什么**  
  从“所有历史都走重型注意力编码”改成“先用 RSS/CS3 做指令条件下的选择性记忆，再用 Transformer 做动作决策”。

- **改变了哪类瓶颈**  
  - 计算瓶颈：减少对长历史做全面注意力的需求  
  - 信息瓶颈：过滤与指令无关的视觉观测  
  - 对齐瓶颈：避免把长视觉序列和短文本序列粗暴拼成一条单流序列

- **带来了什么能力变化**  
  - 长指令下更稳
  - 参数和 FLOPs 大幅下降
  - 仍保留基于注意力的显式候选动作选择能力

- **为什么这样有效（因果上）**  
  1. **RSS** 让全景视角在一次扫描内获得更接近非因果的全局上下文，不再局限于单向状态传播。  
  2. **CS3** 把“谁更新状态”“谁决定读出”分给不同模态，相当于做了更适合 VLN 的双流条件筛选。  
  3. **Transformer 保留在后端**，因为动作选择本质上仍是离散候选比较问题，SSM 负责记忆过滤，Transformer 负责最后一步精确判别。

### 1. RSS：把全景空间关系放进一次扫描

作者认为，现有视觉 Mamba 类方法常用双向或多方向扫描来补足空间感知，但这会增加扫描轮次，而且仍带有较强的因果扫描痕迹。

RSS 的关键改动是：
- 先把输入序列与其反转序列拼接成一条“环形”序列；
- 只做一次 selective scan；
- 再把输出按两半还原并融合。

这样做的效果是：
- 类 token 在序列两端都能接触全局信息；
- 后半段扫描时，状态空间中已经包含前半段的全局压缩信息；
- 不必像双向扫描那样做两轮独立传播。

对 VLN 来说，这很适合 **一个节点内多个视角的空间理解**，因为全景理解本来就不是严格因果任务。

### 2. CS3：双流跨模态选择性记忆

VLN 的文本和视觉序列长度很不均衡，简单拼接成单流输入会有两个问题：
- 对齐粗糙；
- 不同模态被同一套选择机制处理，不够合理。

CS3 的思路是：
- 若目标是“用文本更新视觉”，则让 **文本进入状态空间**；
- 由 **视觉端的类 token** 决定状态空间读出的方式；
- 最终从文本状态读出的结果充当门控，去过滤视觉特征。

直觉上，它不像普通 cross-attention 那样直接做 token-to-token 匹配，而是在状态空间里先完成一种 **条件记忆转移**，再回到特征维度上做门控筛选。

这比“文本+视觉硬拼一条序列喂给 Mamba”更适合 VLN，因为 VLN 需要：
- 长度不对称模态对齐；
- 文本条件下的视觉记忆过滤；
- 反复在局部/全局两种尺度上做跨模态交互。

### 3. 混合架构：SSM 做记忆，Transformer 做决策

COSMO 沿用了 DUET 的双尺度思想，但重排了职责。

#### 节点编码器
- 先在 node encoder 中建立拓扑图；
- 对每个节点的多视角观测先做自注意力编码，再压缩成节点表示；
- 当前节点保留更细粒度视图给 local encoder，历史/ghost nodes 则进入 global encoder。

这一步的意义是：**把历史压缩提前做掉**，减轻后续跨模态模块的负担。

#### 全局跨模态编码器
- 面向拓扑图节点和 ghost nodes
- 用 GASA 建模图结构内关系
- 用 CS3 按指令筛选与导航目标相关的节点记忆
- 再用 cross-attention 做更精确的 grounding

#### 局部跨模态编码器
- 面向当前节点的多视角观测
- 先用 CS3 做文本条件下的选择性过滤
- 再用 RSS 在视图间传播全景上下文
- 后接 cross-attention 和 self-attention 做最终局部判断

#### 最终决策
- 局部和全局两路预测再动态融合
- 输出最终导航动作

### 战略取舍表

| 设计 | 主要解决的瓶颈 | 带来的能力 | 代价 / 保留问题 |
|---|---|---|---|
| RSS | 全景视角间关系建模；多方向扫描低效 | 单次扫描获得更全局的上下文，适合非因果全景理解 | 序列长度翻倍，但作者称在并行 scan 下时间影响较小 |
| CS3 | 文本/视觉长度不对称、单流 SSM 对齐差 | 更适合做“用一模态筛另一模态”的条件记忆 | 结构更任务定制，不如单流 Mamba 通用 |
| SSM → Transformer | 长历史编码成本高，但动作选择又要精细 | 同时获得低成本记忆过滤和高精度候选比较 | 仍保留部分 Transformer，不能做到纯 SSM |
| 提前做 node encoding / topo compression | 后端跨模态编码负担重 | 历史与 ghost node 更早压缩，后续更轻 | 仍依赖 DUET 式拓扑图表示与双尺度框架 |

---

## Part III：证据与局限

### 关键证据

#### 1. 比较信号：效率-性能前沿明显前移
最重要的结果不是“所有榜单绝对第一”，而是 **在非常低成本下达到竞争性性能**。

- **REVERIE val unseen**：  
  COSMO 的 **SR 50.81 / SPL 35.93**，高于 DUET 的 **46.98 / 33.73**
- **计算成本**：  
  COSMO 只有 **28M 参数、0.46G FLOPs**；DUET 为 **181M、4.95G FLOPs**  
  即约 **15.5% 参数**、**9.3% FLOPs**

这说明作者不是单纯“做小模型换掉点”，而是把效率-性能曲线整体往前推了一步。

#### 2. 跨设置信号：连续环境下仍有收益
在 **R2R-CE test unseen** 上：
- COSMO: **SR 47 / SPL 40**
- DUET: **SR 42 / SPL 36**

说明这种“先记忆过滤、再动作选择”的分工，不只在离散图导航中有效，也能迁移到连续环境变体。

#### 3. 长指令分析：优势主要来自“更会筛记忆”
作者最有说服力的分析之一是按指令长度分组比较：
- 随着指令变长，所有方法都会下降；
- 但 COSMO 对 DUET 的优势在长指令上更明显；
- 在 **REVERIE 长指令（>30 words）** 上，COSMO 比 DUET 高 **7.42% SR**

这直接支持了论文的主叙事：  
**COSMO 的收益不是泛化的偶然涨点，而是来自对长历史/长指令场景下记忆过滤能力的提升。**

#### 4. 因果消融：不是“换个骨干就行”，而是三件事都重要
作者做了相对完整的消融：

- **直接用单流 Mamba 做 VLN**：  
  REVERIE val unseen 的 SR 只有 **32.25**，远低于 COSMO
- **去掉 RSS**：SR/SPL 明显下降
- **去掉 CS3**：SR/SPL 进一步明显下降
- **纯 SSM 架构** 不行，**SSM → Transformer** 优于 **Transformer → SSM**

这说明：
1. 纯 SSM 不适合直接承担 VLN 的动作选择；
2. RSS 和 CS3 都不是可有可无的小修补；
3. 真正有效的是 **“选择性记忆前置 + 注意力决策后置”** 的职责分解。

### 局限性

- **Fails when**: 指令对相似走廊/相邻房间的描述存在歧义时，模型容易选错分支；当任务要求最后几步精细靠近小目标或侧边物体时，模型可能能到对房间但停得不够近。
- **Assumes**: 依赖 DUET 式双尺度拓扑图框架、预训练 TinyBERT/视觉编码器，以及 R2R/REVERIE 的增强数据预训练；在 R2R-CE 中还沿用了与前作一致的 waypoint predictor；效率比较主要基于模型 FLOPs 和单步推理设置，而非完整机器人系统延迟。
- **Not designed for**: 无拓扑抽象的原始连续控制、开放世界室外导航、需要额外知识库/地图/深度先验才能完成的更强语义推理场景。

### 可复用组件

这篇工作的可迁移价值主要有三点：

1. **RSS**：适合需要“非因果全局上下文、但又想保持线性扫描效率”的视觉/全景序列建模。
2. **CS3**：适合长度不均衡的双流多模态融合，不只是 VLN，也可用于“文本条件筛视觉”类任务。
3. **Memory-before-decision 结构**：先用轻量结构压缩和过滤历史，再把注意力预算留给真正的候选选择，这个范式对很多 embodied task 都有启发。

整体评价：  
这篇论文最强的地方不是把 VLN 做成纯 SSM，而是很清楚地承认 **SSM 擅长记忆，Transformer 擅长选择**，然后据此做了合理分工。因此它更像是一篇 **效率导向、因果设计清晰** 的结构改造论文。

## Local PDF reference

![[paperPDFs/Vision_Language_Navigation_VLN_Models_2025/arXiv_2025/2025_COSMO_Combination_of_Selective_Memorization_for_Low_cost_Vision_and_Language_Navigation.pdf]]