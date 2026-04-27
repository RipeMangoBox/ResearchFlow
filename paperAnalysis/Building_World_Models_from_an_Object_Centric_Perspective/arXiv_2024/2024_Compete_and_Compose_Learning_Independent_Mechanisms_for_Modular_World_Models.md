---
title: "Compete and Compose: Learning Independent Mechanisms for Modular World Models"
venue: arXiv
year: 2024
tags:
  - Others
  - task/video-understanding
  - mixture-of-experts
  - winner-takes-all
  - "dataset/Particle Interactions"
  - "dataset/Traffic"
  - "dataset/Team Sports"
  - opensource/promised
core_operator: 用赢家通吃的竞争式专家路由先从多环境序列中分化出独立交互机制，再在新环境里学习按对象对选择并组合这些机制
primary_logic: |
  多环境对象级观测序列 → 对每个对象并行评估所有“机制-上下文对象”组合并仅更新误差最小者以分化出专门化机制 → 在新环境中冻结机制并训练分类器选择机制与上下文对象 → 输出可解释、可复用且样本高效适应的模块化世界模型
claims:
  - "COMET在Particle Interactions、Traffic和Team Sports三域中都学出由单一主机制解释的真实交互模式分配，而NPS的机制分配更纠缠 [evidence: analysis]"
  - "在未见环境中使用最优机制-上下文选择时，COMET的平均rollout error在三域均低于NPS，说明其机制可直接跨环境复用而无需微调 [evidence: comparison]"
  - "在Particle Interactions和Traffic的低数据适应场景中，仅训练组合模块的COMET比对整模型进行finetune的C-SWM与NPS更具样本效率 [evidence: comparison]"
related_work_position:
  extends: "Learning Independent Causal Mechanisms (Parascandolo et al. 2018)"
  competes_with: "C-SWM (Kipf et al. 2020); Neural Production Systems (Goyal et al. 2021)"
  complementary_to: "Slot Attention (Locatello et al. 2020); SAVi++ (Elsayed et al. 2022)"
evidence_strength: moderate
pdf_ref: paperPDFs/Building_World_Models_from_an_Object_Centric_Perspective/arXiv_2024/2024_Compete_and_Compose_Learning_Independent_Mechanisms_for_Modular_World_Models.pdf
category: Others
---

# Compete and Compose: Learning Independent Mechanisms for Modular World Models

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2404.15109)
> - **Summary**: 论文提出 COMET，把世界模型训练拆成“先竞争学机制、再在新环境里学组合”两步，在没有交互标签监督的情况下学出可复用的对象交互原语，并提升跨环境低样本适应效率。
> - **Key Performance**: 关键指标是未见环境上的 **average rollout error** 与适应阶段的 **20-step average error**；在最优机制选择下，COMET 在三域都优于 NPS，而在 Particle Interactions 与 Traffic 的低数据适应中优于 C-SWM 与 NPS。

> [!info] **Agent Summary**
> - **task_path**: 多环境对象级观测序列（无动作） -> 未来对象状态/rollout预测，并在新环境中选择机制组合
> - **bottleneck**: 单体动力学模型把不同环境的交互规律纠缠在一起，导致旧知识无法被选择性复用
> - **mechanism_delta**: 用 winner-takes-all 竞争先分化出独立机制，再把新环境适应改写成“选哪条机制、配哪个上下文对象”的分类问题
> - **evidence_signal**: 三个域上最优机制选择时的 rollout error 全面优于 NPS，且 Particle/Traffic 的低样本适应优于 finetuning 基线
> - **reusable_ops**: [winner-takes-all expert routing, mechanism-context composition classifier]
> - **failure_modes**: [single-mechanism collapse without warm-start, degraded performance under higher-order/global interactions]
> - **open_questions**: [how to scale from binary to n-ary interactions, how to instantiate new mechanisms in truly novel environments]

## Part I：问题与挑战

这篇工作的真正目标，不是把一步预测做得更“黑箱地更准”，而是让世界模型能够从**多个动力学不同的环境**里抽出一组**可迁移的交互规律**，并在新环境中只用少量数据就学会“什么时候用哪条规律”。

### 这篇论文在解决什么问题？
现有 object-centric world model 已经能把场景拆成对象槽位，但**状态可分解 ≠ 动力学可分解**。  
很多方法仍然用一个单体转移模型去解释所有环境、所有交互模式，于是会出现两个问题：

1. **不同环境的规则被参数纠缠在一起**  
   同一个模型同时学“吸引”“排斥”“停车”“减速”等不同规律，梯度互相干扰。
2. **迁移时只能整体微调**  
   新环境来了之后，模型通常需要重写整套动力学参数，而不是显式复用旧规则。

### 真正的瓶颈是什么？
真正瓶颈是：**缺少“选择性更新”机制**。  
如果训练时每条样本都去更新整模型，那么模型更容易学到一个平均化、纠缠的解释；但如果能只更新“当前最能解释该样本的那部分模块”，模块才有机会稳定地专门化成独立机制。

### 输入/输出接口与边界
- **输入**：来自多个环境的观测序列；论文主设定是**无动作**、对象级观测。
- **中间表示**：每一帧被分解为若干对象 slot；图像实验里依赖对象分割后的 masked RGB，再编码成对象嵌入。
- **输出**：每个对象的下一步状态更新，以及多步 rollout。
- **适应目标**：给定少量新环境数据，不重学全部动力学，而是学会如何**组合已学机制**。

### 为什么现在值得做？
因为 object-centric 表示已经让“对象是谁”变得相对可学，但“对象如何相互作用”仍常被单体模型吞掉。  
换句话说，表征层的结构化已经到位，**动力学层的模块化**成了下一个更关键的泛化瓶颈。

---

## Part II：方法与洞察

COMET 的核心不在于更复杂的网络，而在于**改变梯度被分配到哪里**。

### 方法骨架

#### 1. Competition：先学“有哪些机制”
COMET 维护一组彼此独立参数化的 mechanisms。  
每个 mechanism 接收：
- 一个目标对象
- 一个上下文对象

并预测目标对象的状态更新。于是，一个 mechanism 可以被理解为一种**二元交互原语**，例如“靠近时排斥”“看到红灯时停车”。

训练时，对每个对象，模型会并行尝试所有：
- 机制 \(m\)
- 上下文对象 \(j\)

然后只把梯度分配给**预测误差最小**的那一对 mechanism-context pair。  
这就是 winner-takes-all 的竞争训练。

结果是：
- 同类交互样本会持续流向同一个 mechanism；
- 该 mechanism 因为被更多同类数据更新，会越来越擅长这类交互；
- 形成正反馈，最终逼出**专门化机制**。

#### 2. Composition：再学“什么时候用哪个机制”
在新环境里，COMET **冻结 mechanisms**，只训练一个组合模块。  
这个模块本质上是个分类器：对每个对象，预测应该选
- 哪个 mechanism
- 哪个上下文对象

于是适应问题从“重学动力学”变成了“识别当前状态对应哪条已知规则”。

论文还用了两个稳定化技巧：
- **warm-start**：训练初期先不给 winner-takes-all，避免某个机制过早垄断全部样本。
- **更长时间窗的 winner 选择**：要求某个机制-上下文组合在连续多个时间步上都表现最好，减少机制切换抖动。

### 核心直觉

- **What changed**：从“所有样本都更新同一套动力学参数”，改成“每条样本只更新当前最能解释它的机制模块”；同时把迁移阶段从 joint finetuning 改成固定机制后的组合选择。
- **Which bottleneck changed**：这改变了多环境训练中的**梯度干扰分布**。原来不同交互模式会互相覆盖，现在它们被稀疏路由到不同模块；新环境适应也从高维参数重写，变成低维规则选择。
- **What capability changed**：模型更容易形成**可辨识、可复用、可解释**的交互原语，并在少样本新环境中更快适应。

更因果地说，这个设计有效是因为：

1. **竞争阶段解决“学什么”**  
   机制不是被 attention 软选择地同时训练，而是被硬竞争地分工训练，因此更容易形成互斥专长。
2. **组合阶段解决“何时用”**  
   新环境往往不是出现全新物理，而是旧规律的新组合；这时学一个选择器比重写整个动力学网络更省样本。
3. **对象对接口提供最小可解释单元**  
   机制输入被限制为“目标对象 + 一个上下文对象”，自然把机制含义压缩到交互原语层面。

### 策略权衡

| 设计选择 | 改变了什么瓶颈 | 带来的收益 | 代价/风险 |
|---|---|---|---|
| winner-takes-all 竞争更新 | 减少不同环境/规则的梯度互扰 | 促进机制专门化，学出可复用原语 | 早期容易塌缩到单一机制 |
| 冻结 mechanisms，只训练 composition | 把迁移从“改动力学”变成“选规则” | 低样本适应更高效，也更可解释 | 无法发明全新机制 |
| mechanism 只看对象对 | 强迫交互在局部、二元层面表示 | 结构清晰，选择空间可控 | 难处理高阶/全局协同 |
| 多步时间窗决定 winner | 给机制选择加入时间一致性约束 | 减少 flickering，提升可分化性 | 超参更敏感，训练更复杂 |
| 依赖 object slots | 先把“谁是谁”固定好 | 让论文专注研究动力学机制本身 | 真实复杂视觉场景需额外对象发现模块 |

---

## Part III：证据与局限

### 关键证据信号

#### 信号 1：机制是否真的被“分开学出来”了？
- **证据类型**：analysis
- **看什么**：机制选择与真实交互模式的相关矩阵。
- **结论**：在 Particle Interactions、Traffic、Team Sports 三个域里，COMET 学到的机制分配都呈现“一个真实行为主要对应一个机制”的结构；NPS 没有形成类似结构。
- **意义**：这直接支持论文最核心的主张——竞争训练确实促成了**可辨识的机制分化**，而不只是多放几个模块而已。

#### 信号 2：这些机制能不能直接拿到新环境里复用？
- **证据类型**：comparison
- **看什么**：未见环境中，在给定最优 mechanism-context 选择时的 rollout error。
- **结论**：COMET 在三域都比 NPS 低。
- **意义**：这说明 COMET 学到的不只是“训练集内可分”，而是**跨环境也可复用**；也证明 competition 学出的机制比 joint attention 学出的更稳。

#### 信号 3：组合旧机制是否比整模型微调更省样本？
- **证据类型**：comparison
- **看什么**：新环境中，随着 adaptation episodes 增加，20-step average error 如何变化。
- **结论**：在 Particle Interactions 和 Traffic 中，COMET 在低数据区间优于 C-SWM 与 NPS。
- **意义**：这支持“把适应变成选择问题”确实降低了样本需求。

### 1-2 个最关键指标
- **average rollout error in unseen environments**：用于验证机制本身能否被直接复用。
- **20-step average error vs. adaptation episodes**：用于验证适应效率是否提升。

### 局限性

- **Fails when**: 需要高阶或全局交互时效果会明显受限，例如 Team Sports 里某些决策依赖整场信息而不是单个对象对；如果新环境包含训练中从未出现的全新交互机制，冻结的 mechanism 库也无法仅靠重组解决。
- **Assumes**: 假设已有对象级表示，实验里甚至使用 ground-truth segmentation masks 与预训练表示模型；假设交互主要可用二元关系近似；还假设训练阶段准备的 mechanism 数量足以覆盖多环境中的动力学模式。代码在文中仅说明“camera-ready 时发布”，当前可复现性仍受限于实现未公开。
- **Not designed for**: 端到端从原始复杂视频里同时做对象发现与机制发现；动作条件 world model；在线增量地创建新机制的 continual/lifelong learning 场景。

### 资源与复现备注
- 训练资源本身不算极端：作者报告单次训练只用 **1 张 GPU**，且 **24 小时内完成**。
- 真正的隐性依赖在于：
  - 对象分割/对象槽位来源；
  - 表示模型预训练；
  - 自建多环境数据生成流程；
  - 代码尚未正式公开。

### 可复用组件
1. **winner-takes-all expert routing**：适合把“混合规律”拆成独立模块。
2. **frozen-mechanism + lightweight selector**：适合低样本迁移，把适应从参数更新改成路由学习。
3. **warm-start + multi-step winner selection**：对任何竞争式模块化训练都很实用，可减少塌缩和抖动。
4. **object-centric dynamics interface**：机制只操作对象表示，能与别的 object-centric encoder 组合。

## Local PDF reference

![[paperPDFs/Building_World_Models_from_an_Object_Centric_Perspective/arXiv_2024/2024_Compete_and_Compose_Learning_Independent_Mechanisms_for_Modular_World_Models.pdf]]