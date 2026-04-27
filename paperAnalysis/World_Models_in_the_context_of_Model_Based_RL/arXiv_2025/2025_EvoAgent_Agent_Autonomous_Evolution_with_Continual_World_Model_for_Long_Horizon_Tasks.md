---
title: "EvoAgent: Agent Autonomous Evolution with Continual World Model for Long-Horizon Tasks"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/long-horizon-task-completion
  - reinforcement-learning
  - world-model
  - curriculum-learning
  - dataset/MineRL
  - dataset/Atari100k
  - opensource/no
core_operator: 用持续更新的世界模型把规划、控制、反思做成闭环，并通过课程式经验筛选来稳定增量更新世界知识
primary_logic: |
  当前观测/自状态/资产 + 长程目标 + 历史多模态经验
  → LLM分解子任务
  → 世界模型引导低层动作并通过自验证写回新经验
  → 两阶段课程学习筛选关键子任务与关键轨迹来更新世界模型
  → 提升长程任务成功率并减少无效探索
claims:
  - "On the Minecraft long-horizon benchmark, EvoAgent reaches 30.29% overall success rate; the paper reports this as a 105.85% average relative improvement over Jarvis-1, DreamerV3, and Optimus-1 [evidence: comparison]"
  - "On Diamond tasks, EvoAgent achieves 26.83% exploration efficiency, versus 7.31% for Optimus-1 and 3.69% for DreamerV3, indicating much less ineffective exploration under sparse rewards [evidence: comparison]"
  - "In ablations, adding the continual world model on top of planning, control, and reflection raises Gold success from 17.53% to 21.69% and Diamond success from 10.09% to 17.36%; the paper attributes 72% of the total gain to the continual WM [evidence: ablation]"
related_work_position:
  extends: "DreamerV3 (Hafner et al. 2025)"
  competes_with: "Optimus-1 (Li et al. 2024); Jarvis-1 (Wang et al. 2023c)"
  complementary_to: "GenRL (Mazzaglia et al. 2024b); RoboDreamer (Zhou et al. 2024)"
evidence_strength: strong
pdf_ref: paperPDFs/World_Models_in_the_context_of_Model_Based_RL/arXiv_2025/2025_EvoAgent_Agent_Autonomous_Evolution_with_Continual_World_Model_for_Long_Horizon_Tasks.pdf
category: Embodied_AI
---

# EvoAgent: Agent Autonomous Evolution with Continual World Model for Long-Horizon Tasks

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2502.05907)
> - **Summary**: 这篇工作把长程 embodied agent 的规划、控制、反思接成一个由持续世界模型驱动的闭环，让代理能边执行边积累经验、筛选经验并更新世界知识，从而在开放世界里更稳地完成长链任务。
> - **Key Performance**: Minecraft Overall SR 30.29%，相对 Jarvis-1 / DreamerV3 / Optimus-1 的平均相对提升为 105.85%；Diamond 任务 EE 26.83%，高于 Optimus-1 的 7.31%

> [!info] **Agent Summary**
> - **task_path**: 开放世界多模态状态（视觉观测+自状态+资产）+ 长程目标 → 子任务序列 → 低层动作与交互轨迹 → 任务完成与更新后的世界模型
> - **bottleneck**: 部署后新任务带来的经验分布变化无法被自主筛选并稳定写入世界知识，导致稀疏奖励下探索浪费、长链策略断裂和灾难性遗忘
> - **mechanism_delta**: 用“经验驱动规划 + WM引导控制 + 两阶段课程反思更新WM”的闭环替代静态记忆或一次性训练
> - **evidence_signal**: Minecraft 长程任务对比和消融都显示优势主要出现在 Gold/Diamond 这类深工具链、强稀疏奖励场景
> - **reusable_ops**: [自验证式子任务终止, 两阶段课程经验选择]
> - **failure_modes**: [Diamond级任务成功率仍低于20%, 对GPT-4o规划器与相似度阈值设定较敏感]
> - **open_questions**: [若替换为开源规划器是否仍能保持优势, 在真实机器人连续控制和安全约束下能否稳定迁移]

## Part I：问题与挑战

这篇论文研究的是一个很具体但很难的问题：**在部分可观测、开放式环境中，代理如何从空经验池出发，仅靠低层动作完成长程任务，并且在执行过程中自己更新“该怎么做”的世界知识。**

### 1. 问题接口
论文把输入/输出接口定义得比较清楚：

- **输入**：当前多模态状态  
  - 第一人称视觉观测
  - agent 自身状态（如 health / hunger）
  - 资产状态（如工具、材料）
  - 长程目标任务
  - 历史多模态经验池
- **输出**：
  - 上层：子任务序列
  - 下层：低层动作序列
  - 训练侧：更新后的经验池与世界模型

### 2. 真正的瓶颈是什么
作者认为，长程任务难，不只是因为“步数长”或者“奖励稀疏”，而是因为有两个更底层的瓶颈：

1. **经验不能自主积累与筛选**  
   许多方法依赖人工构造的 demonstration、人工 curriculum 或预先固定好的 memory。这样一来，部署后遇到新任务/新环境时，agent 不知道哪些新经验值得保留、哪些只是噪声探索。

2. **世界知识不能持续更新且不遗忘**  
   现有 LLM-memory 或图结构 memory 往往更像“检索历史片段”，不是一个真正可持续更新的动态知识载体。新任务到来时，容易出现：
   - 旧知识被新分布覆盖
   - 新经验无法稳定融入已有知识
   - 长链策略中间步骤失真，导致整体失败

### 3. 为什么现在值得解决
这件事之所以“现在”重要，是因为两类能力已经分别成熟，但还没被真正闭环起来：

- **LLM/VLM** 已经能做语义级任务分解；
- **DreamerV3 类世界模型** 已经能在像素空间里学到可用于控制的 latent dynamics。

但此前大多是：
- 规划强、持续学习弱；或
- 控制强、任务分解弱；或
- 有记忆、但记忆不会“进化”。

EvoAgent 的核心主张就是：**把世界模型从“控制器的预测器”升级成“贯穿规划-控制-反思的持续知识底座”。**

### 4. 边界条件
这篇工作默认的使用边界也比较明确：

- 环境是**开放世界 / 部分可观测**
- 控制接口是**低层动作**
- 训练过程允许**持续在线交互**
- 初始经验池为空
- 主要验证在 **Minecraft(MineRL)**，并在附录里补充 **Atari100k** 泛化实验

---

## Part II：方法与洞察

EvoAgent 的方法不是简单把 LLM 和 Dreamer 拼起来，而是把 agent 设计成一个**会自我进化的闭环系统**：

**规划器生成子任务 → 控制器执行并验证 → 反思器筛经验更新世界模型 → 更新后的世界模型反过来影响下一轮规划与控制。**

### 1. 经验驱动任务规划器：先把“长目标”拆成当前能做的事

规划器接收当前状态、长程目标和经验池，输出可执行的子任务序列。

其做法是：

- 用视觉 tokenizer 编码观测、自状态、资产
- 用文本 tokenizer 编码目标与历史经验
- 经过投影后送入 LLM 生成子任务
- 若某个子任务执行失败，则从经验池中抽取该子任务相关轨迹，对规划器做 **LoRA 轻量微调**

这里的关键点不在“用了 LLM”，而在于：

- 子任务不是只看目标生成，而是**结合当前状态与历史经验**生成；
- 失败不是被丢掉，而是被转成**局部规划修正信号**。

这让 planner 不再只是静态常识调用器，而开始具备“遇错后调整拆解方式”的能力。

### 2. WM-guided Action Controller：用世界模型降低盲探索

控制器的职责是：给定当前状态与子任务，用世界模型预测未来并选动作。

它采用 Dreamer/RSSM 风格世界模型来做 latent dynamics 预测，再据此搜索未来动作序列。直观上，它相当于让 agent 在真正动手前，先在内隐空间里“试走几步”。

这一步解决的是：

- 单靠 LLM 拆出子任务，并不能保证低层动作真的能到达；
- 长程任务失败常常不是“不会规划”，而是**中间动作探索太盲**。

#### 自验证机制
控制器执行动作后，还会做一个自验证：

- 若当前 latent state 与子任务 embedding 足够接近，判定子任务完成；
- 或达到最大步数则终止该子任务。

然后把整段轨迹连同子任务完成比例写回经验池。

这一步很重要，因为它把原始交互轨迹变成了**按子任务切分、带完成度语义的经验单元**，为后续反思更新提供更干净的数据。

### 3. CL-based Reflector：不是所有经验都值得更新世界模型

这是 EvoAgent 最关键的“持续学习”部分。

作者没有直接把所有新经验都拿去更新世界模型，而是用了两阶段 curriculum selection。

#### 阶段一：先挑“该学哪些子任务”
子任务优先级由四类信号决定：

- 与当前目标的相关性
- 该子任务已有探索效率
- 该子任务对世界模型变化的重要性
- 当前完成比例

直觉上，这一步是在决定：**下一轮更新应该更关注哪类子问题。**

#### 阶段二：再挑“这些子任务里哪些经验最值钱”
在选出的子任务里，再根据三类信号挑具体经验：

- TD-error：当前模型预测错得多不多
- Gradient norm：这条样本会不会真正推动参数更新
- Information gain：它是否显著改变模型对后续状态的信念

这相当于从 replay buffer 的“均匀/启发式抽样”，升级成**任务相关 + 模型敏感 + 信息增量导向**的经验筛选。

### 4. 持续世界模型更新：把“经验池”变成“世界知识”

最终，反思器用筛出的高价值经验更新世界模型，同时加上重要参数保护项（Fisher 风格正则）来降低历史遗忘。

所以这里的世界模型不再只是 controller 的内部模块，而是系统中的**知识中枢**：

- planner 会受它间接影响
- controller 直接依赖它做动作搜索
- reflector 持续修正它
- experience pool 为它提供增量学习数据

### 核心直觉

**这篇论文真正调的“因果旋钮”不是模型规模，而是世界模型的更新分布。**

#### 改变了什么
从：
- 静态 memory / 历史检索
- 所有经验一股脑回放
- 规划、控制、反思彼此弱耦合

变成：
- 子任务级经验写回
- 课程式选择高价值经验
- 带遗忘约束的持续世界模型更新
- 规划-控制-反思共享同一个演化中的知识底座

#### 哪条瓶颈被改写了
它改写的是两个分布问题：

1. **探索分布**：从大量无效探索，变成更偏向目标相关且高信息增量的轨迹；
2. **更新分布**：从新任务样本直接冲刷旧知识，变成“有筛选、有保护”的增量更新。

#### 能力为什么会提升
因果链可以概括为：

- **更好的子任务分解**  
  ↓  
- **更聚焦的低层探索**  
  ↓  
- **更干净的经验写回**  
  ↓  
- **更稳定的世界模型更新**  
  ↓  
- **下一轮规划/控制更可靠**

因此能力提升主要发生在**需要长工具链、长依赖、多次中间状态切换**的任务上，而不是简单 Wood/Stone 级任务上。

### 战略权衡表

| 设计选择 | 带来的能力 | 代价 / 风险 |
|---|---|---|
| LLM planner + failure-triggered LoRA | 能按当前状态重拆长任务，并从失败中修正子任务分解 | 依赖 GPT-4o；LoRA 质量受经验池质量约束 |
| RSSM world model 做动作搜索 | 低层控制不再纯盲搜，更适合稀疏奖励 | 模型误差会累积，长视距 imagined rollout 仍可能漂移 |
| 自验证式子任务终止 | 把轨迹切成可复用的子任务经验单元 | 相似度阈值与最大步数设置不当会误终止或拖延 |
| 两阶段 curriculum 经验选择 | 过滤噪声探索，聚焦高价值更新样本，缓解遗忘 | 需要更多打分计算和超参调节，可能偏向已有任务结构 |
| Fisher 风格正则保护旧知识 | 连续更新时更不容易灾难性遗忘 | 可能限制对剧烈新分布的快速适应 |

---

## Part III：证据与局限

### 关键证据信号

#### 1. 比较实验信号：优势主要出现在真正困难的长程层级
在 Minecraft 的 Wood / Stone / Iron / Gold / Diamond 分组任务上：

- **Overall SR**：EvoAgent 达到 **30.29%**
- 论文报告其相对 Jarvis-1、DreamerV3、Optimus-1 的平均相对提升为 **105.85%**

更重要的是，它的提升不是平均撒开的，而是集中在高难层级：

- **Gold SR**：21.69% vs 10.62%(Optimus-1)
- **Diamond EE**：26.83% vs 7.31%(Optimus-1)

这说明它真正改善的是**长链依赖和稀疏奖励下的探索/更新机制**，而不只是简单任务上的微调。

#### 2. 消融信号：性能增益来自“闭环完整性”，不是单一模块
消融表很能说明问题：

- 只有 Planning：Wood/Stone 有些提升，但 Iron+ 基本打不开
- 加上 Control：开始能做 Iron，说明世界模型控制确实减少了盲探索
- 再加 Reflection：Gold/Diamond 明显提升，说明经验筛选和反思更新开始发挥作用
- 最后加 Continual WM：从  
  - Gold 17.53% → 21.69%  
  - Diamond 10.09% → 17.36%

这条链路很关键，因为它回答了“为什么不是普通 planner + RL 就够了”：**难点在持续更新世界知识，而 ոչ 仅仅在任务分解。**

#### 3. 泛化信号：附录 Atari100k 并非只在 Minecraft 有效
附录中，EvoAgent 在 Atari100k 上也超过 DreamerV3 的多个任务，例如：

- Alien：1392 vs 1118
- Assault：981 vs 683
- Battle Zone：24830 vs 20300

虽然这部分不是主实验，但至少支持了一个点：**其经验筛选 + 持续 WM 更新机制，不完全是 Minecraft 特化技巧。**

### 1-2 个最值得记住的指标

- **Minecraft Overall SR**：30.29%
- **Diamond EE**：26.83%

一个代表最终任务完成能力，一个代表它是否真的减少了无效探索。

### 局限性

- **Fails when:** 任务层级很深且奖励极稀疏时仍然容易失败；例如 Diamond 任务成功率仍只有 17.36%，说明方法离稳定完成超长链任务还有明显距离。
- **Assumes:** 需要持续在线交互来填充经验池；依赖 GPT-4o 这类闭源规划器；依赖 DreamerV3/RSSM 风格世界模型、自验证相似度阈值和课程选择超参；训练资源报告为单张 A100，且仍需多天运行。
- **Not designed for:** 真实机器人安全关键场景、严格离线 setting、没有明确自状态/资产表征的环境，以及需要严格可解释验证的部署场景。

### 可复用组件

这篇论文里最容易迁移到别的 embodied/agent 系统中的，不一定是整套 EvoAgent，而是下面几个操作符：

1. **自验证式子任务终止**  
   用状态 embedding 与子任务 embedding 的相似度来决定何时结束当前子任务，适合把长轨迹切成语义一致的经验片段。

2. **两阶段课程经验选择**  
   先选“该学的子任务”，再选“最值钱的样本”，比直接 prioritized replay 更贴近长程任务的层级结构。

3. **重要性约束下的持续 WM 更新**  
   用重要经验加权 + 旧知识保护项更新世界模型，适合做跨任务增量 world model learning。

### 一句话总结
EvoAgent 的贡献不在于又加了一个 planner，而在于把**“世界模型如何被经验持续改写”**变成了核心设计对象；它让长程 agent 的能力提升更多来自**更新机制的改造**，而不是单纯更强的推理器。

## Local PDF reference

![[paperPDFs/World_Models_in_the_context_of_Model_Based_RL/arXiv_2025/2025_EvoAgent_Agent_Autonomous_Evolution_with_Continual_World_Model_for_Long_Horizon_Tasks.pdf]]