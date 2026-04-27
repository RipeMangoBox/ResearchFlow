---
title: "Sparse Diffusion Policy: A Sparse, Reusable, and Flexible Policy for Robot Learning"
venue: arXiv
year: 2024
tags:
  - Embodied_AI
  - task/robot-imitation-learning
  - task/continual-robot-learning
  - mixture-of-experts
  - diffusion
  - dataset/MimicGen
  - dataset/DexArt
  - dataset/Adroit
  - dataset/robomimic
  - opensource/full
core_operator: 用任务特定路由将 diffusion policy 的 Transformer FFN 稀疏化为 MoE 专家池，并通过冻结旧专家、增量添加新专家或仅微调路由来实现可复用技能学习
primary_logic: |
  历史状态/观测 + 任务标识 + 示教数据 → 在每层由任务特定 router 选择 Top-K 专家并执行动作扩散去噪，配合任务-专家互信息约束促使专家专门化且可复用 → 输出未来动作序列；面对新任务时通过新增少量专家或仅微调 router 完成持续学习与迁移
claims:
  - "在 MimicGen 8 任务多任务学习中，SDP 取得 0.76 的平均成功率，超过 TH/TCD/Octo 等基线，且活跃参数仅为 53.3M、相比 52.6M 的单任务级别只小幅增加 [evidence: comparison]"
  - "在连续学习 Can→Lift→Square 中，SDP 在 Stage 3 仍保持 Can=0.94、Lift=1.00，同时 Square=0.75；全量微调在学习新任务后旧任务成功率降为 0.00 [evidence: comparison]"
  - "在 Coffee Preparation 迁移实验中，SDP 的可训练策略参数仅 0.1M（主要是 router），成功率达 0.80，高于从头训练 25.9M 策略参数模型的 0.70 [evidence: comparison]"
related_work_position:
  extends: "Diffusion Policy (Chi et al. 2023)"
  competes_with: "TCD (Ajay et al. 2022); LoRA (Hu et al. 2021)"
  complementary_to: "R3M (Nair et al. 2022); SkillDiffuser (Liang et al. 2024)"
evidence_strength: strong
pdf_ref: paperPDFs/Vision_Action_VA_Models_2024/arXiv_2024/2024_Sparse_diffusion_policy_A_sparse_reusable_and_flexible_policy_for_robot_learning.pdf
category: Embodied_AI
---

# Sparse Diffusion Policy: A Sparse, Reusable, and Flexible Policy for Robot Learning

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2407.01531), [Project/Code](https://forrest-110.github.io/sparse_diffusion_policy/)
> - **Summary**: 这篇工作把扩散策略里的 FFN 改造成按任务和状态稀疏激活的 MoE 专家池，使机器人策略能在多任务里少激活参数、在连续学习里尽量不覆盖旧知识、并在新任务上通过轻量路由快速复用已有技能。
> - **Key Performance**: MimicGen 8任务平均成功率 **0.76**（AP **53.3M**，接近单任务级别）；Coffee Preparation 迁移时可训练策略参数仅 **0.1M** 仍达 **0.80** 成功率，高于从头训练的 **0.70**。

> [!info] **Agent Summary**
> - **task_path**: 多任务/连续示教学习中的历史状态与任务ID -> 未来动作序列
> - **bottleneck**: 稠密统一策略在多任务中会把所有任务压进同一参数通道，造成梯度干扰、推理全参数激活，以及新任务学习时覆盖旧知识
> - **mechanism_delta**: 将 diffusion policy 的 FFN 替换为 task-specific router 控制的 Top-K MoE，并用任务-专家互信息促进专家分工与复用
> - **evidence_signal**: 多基准比较 + 连续学习对比显示在近恒定活跃参数下能保留旧任务性能并提升新任务/迁移任务表现
> - **reusable_ops**: [FFN→MoE替换, 冻结旧专家并增量加专家, 仅微调router做任务迁移]
> - **failure_modes**: [共享知识不足却被不同任务路由到同一专家时性能退化, task-specific router限制开放式通用任务执行]
> - **open_questions**: [能否把task-specific router升级为language-conditioned universal router, 专家池如何自动剪枝/合并/扩容以控制总参数增长]

## Part I：问题与挑战

这篇论文要解决的，不是“单个机器人任务怎么做得更好”，而是更难的三合一问题：

1. **多任务学习**：一个策略如何同时学很多操控任务，而不是每个任务一套模型。
2. **连续学习**：学新任务时，如何不把旧任务忘掉。
3. **任务迁移**：已有任务里学到的局部技能，能否被重新组合到新任务上。

### 真正的瓶颈是什么？

作者认为，核心瓶颈不在于 diffusion policy 本身够不够强，而在于**策略参数的组织方式**：

- 传统通用策略大多是**稠密单体网络**；
- 即使任务很简单，推理时也要激活全部参数；
- 多任务训练时，所有任务都更新同一套主干，容易出现**跨任务梯度冲突**；
- 连续学习时，新任务继续改同一套参数，很容易产生**灾难性遗忘**；
- LoRA 这类适配器方法虽然减少训练量，但**推理时活跃参数会继续增加**；
- 若为每个任务单独训练策略，则几乎没有知识复用，迁移效率也低。

换句话说，过去很多方法把问题当成“如何在一个更大的模型里塞进更多任务”，而这篇论文把问题改写成：

> **能否让策略网络本身成为一个稀疏、可复用、可增量扩展的技能库？**

### 为什么现在值得解决？

因为 diffusion policy 已经证明了自己在机器人 imitation learning 上很强，但它默认仍是**密集执行**。一旦任务规模变大、任务分布变杂、学习流程从单次训练走向终身学习，稠密结构会直接变成瓶颈。

### 输入/输出接口与边界条件

- **输入**：历史状态/观测（2D视觉或3D点云等）+ 任务标识
- **输出**：未来动作序列
- **训练范式**：基于人类示教的 imitation learning / behavior cloning
- **边界条件**：
  - 任务是**离散可区分**的；
  - 训练和推理时可使用**任务特定 router**；
  - 不是开放词汇的通用机器人智能；
  - 不是强化学习在线探索设定，而是离线示教学习为主。

---

## Part II：方法与洞察

### 方法框架

SDP 的基座仍然是 **Transformer-based Diffusion Policy**。关键改动非常集中：

#### 1. 把 Transformer 里的 FFN 换成 MoE
每个 MoE 层包含：

- 多个 expert（MLP）
- 一个 router
- router 从多个 expert 中选 **Top-K** 激活

于是，原来“每次都经过同一条稠密 FFN 路径”，变成了“每次只经过少量被选中的 expert 路径”。

#### 2. 多任务学习：每个任务一个 task-specific router
专家池可以共享，但不同任务通过不同 router 来选专家。

这带来两个效果：

- **任务隔离**：不同任务不必总挤在同一参数子空间里；
- **技能共享**：若多个任务都需要类似子技能（如 pick-and-place），它们可以复用同一 expert。

#### 3. 连续学习：冻结旧专家，增量添加新专家
当新任务到来时：

- 旧 experts / old routers 冻结；
- 每层加入少量新 expert；
- 为新任务训练新的 router。

这样做的直觉很直接：  
**不要再改旧知识，而是在旧技能库旁边扩一个新抽屉。**

#### 4. 任务迁移：只学会“怎么选旧技能”
对新任务，作者还研究了更激进的做法：

- 冻结已有 expert pool；
- 主要微调 router（论文的迁移实验里还配合微调 vision encoder，但策略侧可训练参数极少）。

这相当于不重学控制器，而是**重排已有技能组合**。

#### 5. 训练目标：不用常见 load balancing，而用任务-专家互信息
一般 MoE 会加负载均衡，避免某些 expert 被用太多。  
但作者指出：机器人里某些技能本来就应当高频复用，比如 pick-and-place。

所以他们不强调“每个 expert 用得一样多”，而强调：

- **每个任务应该更明确地拥有/偏好自己的 expert 组合**

因此引入 **task-expert mutual information** 正则，鼓励 expert 对任务形成更清晰分工。

### 核心直觉

真正的机制变化可以概括成一句话：

> **把“共享的基本单位”从整张策略网络，缩小为“可选择的 expert”，再把“学新任务”从覆盖旧权重，改成扩展和重组技能库。**

更具体地说：

#### 变化 1：从稠密共享到条件稀疏共享
- **原来**：所有任务都走同一条网络路径，任务差异只能挤在同一套参数里表达。
- **现在**：任务通过 router 只激活少数 expert，梯度主要流向对应子网络。

这改变的是**参数冲突约束**。  
结果是：多任务学习时，不同任务更容易学出不同动作模式，而不是被迫平均化。

#### 变化 2：从“更新旧知识”到“增量扩技能”
- **原来**：新任务必须继续改旧参数。
- **现在**：旧专家冻结，新知识写入新增专家。

这改变的是**遗忘机制**。  
因为旧任务依赖的参数不再被覆盖，所以遗忘显著减轻。

#### 变化 3：从“重学整策略”到“重排技能组合”
- **原来**：迁移到新任务常常要从头学很多控制逻辑。
- **现在**：router 可以把已有 experts 重新编排成新的 skill chain。

这改变的是**信息复用方式**。  
因此，即使预训练任务不多，也可能通过 expert 组合覆盖更复杂的新任务。

#### 变化 4：用 MI loss 解决 router 偏向旧专家的问题
如果只靠行为克隆损失，router 容易一直选旧专家，因为旧专家已经训练好、输出更稳定；新增 expert 因为一开始随机，反而得不到梯度，形成恶性循环。

MI loss 的作用是把这个选择偏置打破：

- 让任务与 expert 之间形成更强对应关系；
- 让新增 expert 真正获得训练机会；
- 尤其对连续学习中的新任务更重要。

### 战略权衡

| 设计选择 | 改变了什么 | 带来的能力 | 代价/风险 |
| --- | --- | --- | --- |
| FFN → MoE + Top-K | 从全参数激活变为条件稀疏激活 | 更低推理活跃参数，减少多任务互扰 | 需要稳定的路由训练 |
| task-specific router | 显式把任务差异注入专家选择 | 同时支持任务隔离与技能共享 | 依赖任务ID，泛化到开放任务受限 |
| freeze old + add new experts | 连续学习从覆盖旧知识变为扩展技能库 | 显著缓解遗忘 | 总参数会随任务增长 |
| task-expert MI loss | 从“均匀负载”改为“任务分工” | 新专家更容易学到新任务所需技能 | 过强时可能削弱跨任务共享 |
| router-only style transfer | 从重学策略改为重组已有技能 | 极低训练成本的快速迁移 | 前提是旧 expert pool 已覆盖足够子技能 |

---

## Part III：证据与局限

### 关键证据信号

#### 1. 比较信号：多任务学习中，稀疏结构确实比稠密结构更合适
在 **MimicGen 8任务** 上：

- SDP 平均成功率 **0.76**
- 高于 TH / TCD / Octo 等基线
- 活跃参数 **53.3M**，与单任务级别 **52.6M** 接近

这里最关键的不是“绝对分数稍高”，而是：

> **它是在几乎不增加活跃参数的情况下赢的。**

这正支持了论文的中心主张：  
**稀疏激活不是单纯省算力，而是能减少多任务学习中的策略互扰。**

作者在 3D 设置（DexArt / Adroit）和真实机器人实验上也给出一致趋势，说明现象不只存在于单一 2D 仿真环境。

#### 2. 比较信号：连续学习里，结构性隔离比参数微调更抗遗忘
在 **Can → Lift → Square** 连续学习里：

- SDP 到 Stage 3 时仍保持 **Can=0.94, Lift=1.00**
- 新任务 **Square=0.75**
- 且活跃参数维持在 **9.2M**
- 相比之下，全量微调旧任务性能掉到 **0.00**

这说明 SDP 的优势不只是“学新任务更快”，而是它把连续学习问题从参数覆盖改成了参数扩展。  
**能力跳跃点**在于：旧任务性能保住了，而不是靠回放、正则化或复杂记忆机制勉强维持。

#### 3. 比较信号：迁移时真正复用的是“技能组合”，不是参数规模
在 **Coffee Preparation** 上：

- 从头训练：25.9M 可训练策略参数，成功率 **0.70**
- SDP 迁移：仅 **0.1M** 可训练策略参数，成功率 **0.80**

这说明预训练 expert 确实学到了可复用技能，而 router 能把这些技能重新组装成更长程的新任务行为。

#### 4. 消融/分析信号：MI loss 和完整 MoE 结构是关键因子
论文做了两类关键补充证据：

- **MI loss ablation**：复杂连续学习序列里，新任务表现明显依赖该损失；
- **expert 频率/分数可视化**：不同任务会共享部分专家，也会激活各自特有专家；在 Coffee Preparation 中，接近咖啡机相关动作时，Coffee 任务对应 experts 会被更频繁选中。

这让“expert 是技能、router 是技能规划器”不只是口号，而有一定可解释性证据支撑。

### 局限性

- **Fails when**: 不同任务实际上共享知识很少，但 router 仍把它们路由到相同 experts 时，专家复用会变成负迁移；另外如果新任务所需技能超出已有 expert pool 覆盖，仅靠 router 重组会失效。
- **Assumes**: 需要任务级示教数据与明确任务标识；依赖 transformer-based diffusion policy 骨架；连续学习通过“新增专家”实现，因此总参数会增长；2D 多任务实验训练成本不低（单卡 A6000 上 130-150 小时量级），真实机器人实验也依赖特定硬件与控制栈（FANUC 机械臂 + admittance control）。
- **Not designed for**: 开放词汇、无任务ID的通用机器人策略；固定总参数预算下的长期终身学习；完全不依赖任务划分的 universal policy。

### 可复用组件

这篇论文最值得复用的，不一定是完整系统，而是以下几个操作子：

1. **MoE 化 FFN**：把 diffusion/transformer policy 的 FFN 替换成 Top-K MoE。
2. **task-specific routing**：显式把“任务差异”放进路由层，而不是全塞进共享主干。
3. **freeze-old add-new**：做连续学习时冻结旧专家、只加少量新专家。
4. **router-first transfer**：先尝试只调 router（或极少量策略参数）做新任务迁移。
5. **task-expert MI regularization**：在机器人多任务场景下，用“任务分工”替代通用 load balancing。
6. **expert usage diagnostics**：用专家激活频率/时间序列分数图检查技能共享和路由塌缩。

## Local PDF reference

![[paperPDFs/Vision_Action_VA_Models_2024/arXiv_2024/2024_Sparse_diffusion_policy_A_sparse_reusable_and_flexible_policy_for_robot_learning.pdf]]