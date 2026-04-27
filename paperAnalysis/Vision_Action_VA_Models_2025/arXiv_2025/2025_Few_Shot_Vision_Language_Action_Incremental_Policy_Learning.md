---
title: "Few-Shot Vision-Language Action-Incremental Policy Learning"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/robotic-manipulation
  - task/imitation-learning
  - prompt-learning
  - task-relation-graph
  - policy-weight-evolution
  - dataset/RLBench
  - opensource/promised
core_operator: "用任务特定提示从少样本多模态演示中提炼任务判别信息，再依据提示相似度构建任务关系图以演化新任务策略权重。"
primary_logic: |
  少量多视角视觉观测 + 语言指令 + 历史任务提示/策略 → 在多视角 Transformer 中通过 Task-Specific Prompts 聚合跨模态任务信息，并按提示相似度在任务关系图上复用旧任务权重与基座通用技能 → 输出对新旧任务更稳健的动作预测
claims:
  - "在 RLBench 构造的 FSAIL 1-shot 设置下，TOPIC 将 SAM-E 的平均准确率从 30.5% 提升到 58.7%，将 RVT-2 从 34.2% 提升到 60.6% [evidence: comparison]"
  - "1-shot 消融中，仅加入 TSP 就能相对各基线提升 13.4/15.1/15.4 个点，而再加入 CES 后增益扩大到 25.3/28.2/26.4 个点（RVT/SAM-E/RVT-2）[evidence: ablation]"
  - "在真实机器人 FSAIL 实验中，SAM-E + TOPIC 的平均准确率从 15.8% 提升到 24.6% [evidence: comparison]"
related_work_position:
  extends: "S-Prompts (Wang et al. 2022)"
  competes_with: "Adaptive Memory Replay (Smith et al. 2024); EWC (Kirkpatrick et al. 2017)"
  complementary_to: "R3M (Nair et al. 2022); OpenVLA (Kim et al. 2024)"
evidence_strength: moderate
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Few_Shot_Vision_Language_Action_Incremental_Policy_Learning.pdf
category: Embodied_AI
---

# Few-Shot Vision-Language Action-Incremental Policy Learning

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2504.15517), [Code (promised)](https://github.com/codeshop715/FSAIL)
> - **Summary**: 这篇工作把机器人操作中的“少样本学习”和“持续增量学习”合并成一个更实际的 FSAIL 场景，并通过任务特定提示 + 任务关系图，让模型在只有 1/5 个新任务示范时也能更好复用旧技能、减少灾难性遗忘。
> - **Key Performance**: RLBench FSAIL 1-shot 上，SAM-E 从 30.5% 提升到 58.7%，RVT-2 从 34.2% 提升到 60.6%；真实机器人实验中平均准确率从 15.8% 提升到 24.6%。

> [!info] **Agent Summary**
> - **task_path**: 多视角 RGB-D 观测 + 语言指令 + 少量增量示范 -> 机器人动作/操控策略
> - **bottleneck**: few-shot 下任务判别信息太稀薄，同时增量更新会覆盖旧任务技能
> - **mechanism_delta**: 将任务知识显式压缩进 Task-Specific Prompts，并用提示相似度图混合历史任务权重与 base 通用权重
> - **evidence_signal**: RLBench FSAIL 1-shot/5-shot 上跨 RVT、SAM-E、RVT-2 都有大幅提升，且 TSP/CES 消融显示两部分都贡献显著
> - **reusable_ops**: [task-specific prompt tokens, prompt-similarity graph weight fusion]
> - **failure_modes**: [real-world 绝对性能仍低, 增量会话变长后旧任务性能仍持续下滑]
> - **open_questions**: [任务关系图在更大任务库下是否会负迁移, 若没有大规模 base session 是否还能保持同等增益]

## Part I：问题与挑战

这篇论文真正瞄准的不是普通的机器人模仿学习，而是一个更苛刻的设定：**Few-Shot Action-Incremental Learning (FSAIL)**。

### 任务是什么
- **输入**：多视角视觉观测（RLBench 中为 4 个 RGB-D 相机）+ 语言指令
- **输出**：机器人动作/末端执行器控制
- **训练流程**：
  - base session：若干任务有充足示范
  - incremental sessions：每个新任务只有 **1-shot 或 5-shot** 示范
  - **过去会话数据不可访问**
- **评估方式**：到第 \(t\) 个会话时，要在**所有已见任务**上一起评估

### 真正的瓶颈是什么
核心不是“少样本”或“持续学习”单独难，而是两者叠加后出现的双重瓶颈：

1. **信息瓶颈**：  
   只有极少示范时，视觉 token 和语言 token 中与“当前任务真正相关”的判别信息很难被稳定提取出来。
2. **更新瓶颈**：  
   新任务微调时，梯度几乎被当前 few-shot 样本主导，容易把旧任务能力覆盖掉，形成灾难性遗忘。
3. **迁移瓶颈**：  
   机器人任务之间明明共享技能（如 pick/place、reach），但现有 Transformer policy 没有显式机制把“旧任务技能”转成“新任务初始化偏置”。

### 为什么现在值得解决
因为当前 Transformer-based manipulation policy（如 RVT、SAM-E、RVT-2）已经有了不错的基础能力，但它们仍依赖大量演示数据。现实里新增机器人任务往往是**持续到来**的，而不是一次性收集完整数据集再统一训练。因此，“**用极少演示增量添加新技能**”是部署阶段的真实瓶颈。

### 边界条件
这篇论文默认的边界很明确：
- 有一个较强的 **base session** 作为技能库起点
- 任务有明确的 session/task 边界
- 主要处理的是**语言条件下的静态操控任务**
- 不是在线 RL，不依赖探索回报，也不是无边界流式学习

---

## Part II：方法与洞察

TOPIC = **Task-prOmpt graPh evolutIon poliCy**。  
它由两个关键部件组成：

1. **TSP: Task-Specific Prompts**
2. **CES: Continuous Evolution Strategy**

### 1）TSP：先把“当前任务是什么”抽出来
作者给每个任务预定义一组可学习 prompt token，把它们和：
- 语言 token
- 视觉 token

一起送进多视角 Transformer 编码器里做深度交互。

这样做的作用不是简单“加几个 prompt”，而是给模型提供一个**显式的任务信息槽位**：
- prompt 会同时看见视觉和语言
- few-shot 中零散的任务线索被汇聚到 prompt 上
- 最后再把 prompt 投影回动作特征，直接影响 action prediction

直观理解：  
原来模型只能从少量示范里“自己猜当前任务最重要的线索”；  
现在作者给了一个专门的容器，让模型把“这项任务的判别信息”先集中编码出来。

### 2）CES：再把“旧任务里哪些技能能复用”显式算出来
作者把每个任务学到的 task-specific prompt 当成一个任务节点表示，并把任务间相似度组成一个**任务关系图**。

然后，新任务的策略权重不是独立训练出来，而是：
- 参考与其相似的历史任务权重
- 再加上当前任务自己的权重
- 再融合 base session 学到的通用技能权重

也就是说，新任务不再从“几条新示范”硬学全部技能，而是从：
- **相似旧任务**继承 task-specific skill
- **base session**继承 generic skill

这就把“过去数据不可访问”转换成了“过去技能仍可通过图结构复用”。

### 3）训练流程
论文的训练流程分三步：
1. **base multi-task training**：先训练 backbone 和 base 通用权重
2. **per-task base fitting**：为各 base task 学各自 prompt 和策略权重
3. **incremental few-shot learning**：面对新任务时，只基于 few-shot 数据增量适配  
   同时文本编码器、视觉编码器保持冻结

这意味着它更像一种**轻量增量适配层**，而不是每次全模型重训。

### 核心直觉

**作者真正调的“因果旋钮”有两个：**

1. **把任务身份从隐式变成显式**
   - 变化前：few-shot 样本太少，任务判别信息分散在视觉/语言 token 中
   - 变化后：TSP 把跨模态任务信息收敛到少数 prompt token
   - 改变的瓶颈：缓解了 few-shot 下的**任务识别与表征稀疏问题**
   - 带来的能力：新任务即使只有少量示范，也更容易形成稳定任务表示

2. **把增量学习从“孤立微调”变成“相似任务技能复用”**
   - 变化前：新任务更新主要受当前 few-shot 数据支配，容易忘旧任务
   - 变化后：CES 按 prompt 相似度把历史策略权重混进来
   - 改变的瓶颈：缓解了**无回放条件下的参数漂移与技能断裂**
   - 带来的能力：新任务更快起步，旧任务保留更多

**为什么这套设计有效：**
- TSP 让模型不必在极少样本里同时完成“识别任务 + 学动作”两件事，而是先显式压缩任务信息，再用于动作预测。
- CES 让“历史经验”不必以旧数据形式存在，而是以“任务提示 + 对应策略权重”的形式被重用。
- 冻结视觉/文本编码器缩小了更新空间，减少了增量过程中的表征漂移。

### 战略取舍

| 设计选择 | 改变了什么约束 | 收益 | 代价/风险 |
|---|---|---|---|
| TSP 插入多模态 Transformer | 给 few-shot 任务增加显式任务槽位 | 更容易提取任务判别特征 | prompt 数量与投影方式需要调参 |
| 任务关系图 + CES | 从孤立微调改为相似任务权重复用 | 减轻遗忘、提升新任务适配 | 相似度估计错误会导致负迁移 |
| 融合 base 通用权重 | 保留跨任务共性技能 | 新任务起点更稳 | 依赖 base session 质量和覆盖度 |
| 冻结视觉/文本编码器 | 缩小可训练空间 | 参数更省、训练更稳 | sim2real/新域适配能力受限 |

---

## Part III：证据与局限

### 关键证据

**信号 1：主对比实验显示，TOPIC 的提升不是某个 backbone 的偶然现象。**  
在 RLBench 构造的 FSAIL 上，作者把 TOPIC 分别接到 RVT、SAM-E、RVT-2 上，1-shot 和 5-shot 都明显优于原始 policy。
- 1-shot：  
  - RVT: 23.8 → 49.1  
  - SAM-E: 30.5 → 58.7  
  - RVT-2: 34.2 → 60.6
- 5-shot：  
  - RVT: 24.3 → 47.7  
  - SAM-E: 29.0 → 58.4  
  - RVT-2: 35.2 → 60.2

**结论**：增益具有跨 backbone 一致性，说明它更像一个“通用增量适配机制”，而非绑定某个模型。

**信号 2：对 replay / regularization / prompt-based continual learning 都有明显优势。**  
作者把 Replay、EWC 类 regularization、S-Prompts 都接到相同 Transformer policy 上比较。TOPIC 相对这些经典持续学习策略仍有约 15–22 个点级别的平均提升。  
**结论**：单纯“保参数”或“加通用 prompt”不够，任务关系驱动的技能复用更关键。

**信号 3：消融表明 TSP 和 CES 各自都有效。**  
- 只有 TSP：已经明显优于 baseline  
- TSP + CES：进一步大幅提升  
**结论**：这不是单一 prompt trick；few-shot 任务提取与图式技能复用是叠加生效的。

**信号 4：分析实验支持其机制解释。**
- prompt 数量在 5 个时最好
- 平均池化投影优于更复杂 MLP/linear
- base session 任务越多，后续增量表现越好
- prompt 相似度可视化和直觉任务关系一致，如 drawer 类任务更接近、reach 类任务更接近

**结论**：作者关于“prompt 是任务表示”“图关系支持技能复用”的解释，至少得到了实验侧面的支持。

**信号 5：真实机器人实验有提升，但绝对值仍低。**  
在 Cobot Mobile ALOHA 上，SAM-E + TOPIC 的平均准确率从 15.8% 提升到 24.6%。  
**结论**：方向成立，但距离可部署仍有差距。

### 1-2 个最该记住的指标
- **RLBench FSAIL 1-shot**：SAM-E 30.5% → **58.7%**
- **真实机器人**：SAM-E 15.8% → **24.6%**

### 局限性

- **Fails when**:  
  真实世界分布偏移较大、增量会话较长、或任务差异较强时，旧任务性能仍会持续衰减；最终 session 的绝对准确率并不高。

- **Assumes**:  
  需要一个较强且覆盖面不错的 base session（仿真里是 10 个 base tasks、每个 100 条示范）；依赖冻结的视觉/文本编码器；依赖任务级 session 划分；依赖多视角 RGB-D + 语言输入；还需要为每个任务保存 prompt/策略权重，存储会随任务数增长。

- **Not designed for**:  
  在线探索式强化学习、无清晰任务边界的流式终身学习、动态移动场景、双臂复杂协作场景；真实实验里实际上只用了单臂、静态任务。

### 复现与证据边界
- 主要证据仍集中在**作者构造的 RLBench FSAIL 划分**上，外部任务分布覆盖有限，因此证据强度应保守看待。
- 真实世界实验规模较小，且正文描述“每个新任务 5 个示范”与 Table IX 标注“1-shot”之间有轻微不一致，复现时需要核对协议。
- 代码是**承诺开源**，不是已验证的完整开源复现。

### 可复用组件
- **Task-Specific Prompts**：可作为任意多模态 Transformer policy 的 few-shot task adapter
- **Prompt-similarity task graph**：可作为无回放 continual adaptation 的技能复用模块
- **FSAIL 协议**：适合作为“少样本 + 增量任务 + 禁止访问旧数据”的机器人评测设定

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_Few_Shot_Vision_Language_Action_Incremental_Policy_Learning.pdf]]