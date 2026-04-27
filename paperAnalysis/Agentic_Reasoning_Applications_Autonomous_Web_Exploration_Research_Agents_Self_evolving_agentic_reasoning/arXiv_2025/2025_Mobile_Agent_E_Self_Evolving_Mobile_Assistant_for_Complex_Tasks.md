---
title: "Mobile-Agent-E: Self-Evolving Mobile Assistant for Complex Tasks"
venue: arXiv
year: 2025
tags:
  - Others
  - task/MLLM-evaluation
  - hierarchical-multi-agent
  - long-term-memory
  - self-evolution
  - dataset/Mobile-Eval-E
  - opensource/full
core_operator: "将高层任务规划与低层 GUI 动作决策解耦，并把跨任务经验沉淀为带前置条件的 Shortcuts 与通用 Tips 持续优化执行。"
primary_logic: |
  用户复杂手机任务 + 当前/历史截图 → Manager 分解子目标、Perceptor 感知界面、Operator 结合 Tips/Shortcuts 执行动作、Action Reflector/Notetaker 更新进度与笔记并在连续错误时回退到高层重规划 → 在后续任务中以更高成功率和更高效率完成跨 App 长程任务
claims:
  - "在 Mobile-Eval-E 上、GPT-4o 骨干下，Mobile-Agent-E 的 Satisfaction Score 为 75.1%，高于 Mobile-Agent-v2 的 53.0%，同时 Termination Error 从 52.0% 降至 32.0% [evidence: comparison]"
  - "启用自演化后，Mobile-Agent-E + Evo 在 GPT-4o 上把 Satisfaction Score 从 75.1% 进一步提升到 86.9%，并把 Termination Error 从 32.0% 降到 12.0% [evidence: comparison]"
  - "在控制为不使用新生成 Shortcuts 的子集上，evolved Tips 仍将 GPT-4o 的 Satisfaction Score 从 79.7% 提升到 87.5%，说明收益不只来自动作压缩 [evidence: analysis]"
related_work_position:
  extends: "Mobile-Agent-v2 (Wang et al. 2024a)"
  competes_with: "Mobile-Agent-v2 (Wang et al. 2024a); AppAgent (Zhang et al. 2023)"
  complementary_to: "Cradle (Tan et al. 2024)"
evidence_strength: moderate
pdf_ref: "paperPDFs/Agentic_Reasoning_Applications_Autonomous_Web_Exploration_Research_Agents_Self_evolving_agentic_reasoning/arXiv_2025/2025_Mobile_Agent_E_Self_Evolving_Mobile_Assistant_for_Complex_Tasks.pdf"
category: Others
---

# Mobile-Agent-E: Self-Evolving Mobile Assistant for Complex Tasks

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2501.11733), [Project/Code/Data](https://x-plug.github.io/MobileAgent)
> - **Summary**: 这篇工作把手机代理拆成“高层规划-低层执行-跨任务经验演化”三层闭环，使其在跨 App、长程、探索式真实任务上显著更稳，也能越用越熟练。
> - **Key Performance**: 在 Mobile-Eval-E 上，GPT-4o 骨干的 Satisfaction Score 从 Mobile-Agent-v2 的 53.0% 提升到 86.9%（+Evo）；Termination Error 从 52.0% 降到 12.0%。

> [!info] **Agent Summary**
> - **task_path**: 用户复杂手机指令 + 实时/历史截图 -> 跨 App GUI 动作序列、信息汇总与最终停止页面
> - **bottleneck**: 现有手机代理把“下一步做什么”和“屏幕上点哪里”混在同一次推理里，且不会把跨任务经验沉淀为可复用知识
> - **mechanism_delta**: 用 Manager/Operator 分离高低层决策，并在任务后把经验反思成通用 Tips 与带前置条件的 Shortcuts 注入后续任务
> - **evidence_signal**: 相同 GPT-4o 骨干下，SS 从 53.0% -> 75.1% -> 86.9%，TE 从 52.0% -> 32.0% -> 12.0%
> - **reusable_ops**: [hierarchical-plan-act-split, tip-shortcut-long-term-memory]
> - **failure_modes**: [invalid-shortcut-precondition, malformed-generated-shortcuts]
> - **open_questions**: [scalable-memory-retrieval, safety-for-sensitive-mobile-actions]

## Part I：问题与挑战

这篇工作的真实问题，不是“手机上能不能点对一个按钮”，而是：

**能不能让代理像一个有经验的人类用户一样，在真实手机上完成跨 App、长步骤、带约束、需要探索的信息任务。**

作者指出现有 mobile agent 的主要短板有三层：

1. **任务本身更复杂了**  
   现实任务常常不是单 App、单目标、短轨迹，而是同时包含：
   - 多约束推理：如价格、距离、评分、偏好要一起满足；
   - 长程规划：可能要连续十几步甚至跨多个 App；
   - 探索式交互：不是按固定轨迹走，而是要查、比较、筛、记。

2. **现有 agent 的决策粒度混在一起**  
   同一个模型既要决定“下一阶段应该去哪个 App、查什么信息”，又要决定“当前屏幕上应该点哪个坐标”。  
   一旦页面有弹窗、布局变化、搜索框残留文本、跳错页，这种**高层规划与低层 grounding 混合**的设计很容易让 agent 陷入局部错误循环。

3. **不会从过去任务中变熟练**  
   人类第一次用某个 App 会试错，但之后会更快。现有方法基本把每个任务都当第一次做，无法把“如何清空搜索框”“怎么快速搜索并回车”“什么时候该切回 Notes 记信息”这类经验沉淀下来。

还有一个被作者明确指出的**评测瓶颈**：  
真实手机任务常常没有唯一 ground-truth trajectory，也没有清晰二元 success flag。比如“规划一天行程”并不存在唯一正确路径，所以只看成功率或轨迹匹配不够。

### 输入 / 输出接口与边界条件

- **输入**：用户自然语言指令 + 当前手机截图 + 交互历史
- **输出**：原子 GUI 操作或 Shortcut、过程中记录的笔记、最终停止页面
- **环境**：真实 Android 真机，通过 ADB 控制；视觉感知为主，不依赖 XML
- **评测**：人工 rubric 计算 Satisfaction Score，而不是只看是否命中唯一轨迹
- **运行边界**：最大 40 次迭代；连续错误或重复动作会强制终止；不自动处理支付等高风险操作

为什么现在值得做？因为作者认为之前的 mobile benchmarks 已经偏短、偏简单，难以区分真正的复杂任务能力。为此他们构造了 **Mobile-Eval-E**：25 个任务、5 个真实场景、19/25 是多 App 任务，平均 14.56 步/任务，明显比过去 benchmark 更长、更接近真实需求。

## Part II：方法与洞察

方法上，这篇 paper 的关键不是“再加几个 agent”，而是做了两个解耦：

- **单任务内解耦**：把高层规划和低层动作分开
- **跨任务间解耦**：把经验分成可迁移的 Tips 与可执行的 Shortcuts

### 方法骨架

#### 1. Manager：只管高层计划
Manager 负责：
- 根据用户目标、当前截图、已有计划、进度、笔记、可用 Shortcuts
- 生成或更新整体计划
- 给出当前子目标

一个很重要的设计点是：**Manager 不看细粒度 OCR/icon 感知结果**。  
作者的判断是，高层规划不需要那么多局部感知细节，强行输入反而会把噪声带进战略层。

#### 2. Perceptor + Operator：看清界面后执行具体动作
- **Perceptor**：OCR + icon grounding + icon captioning，产出文本/图标及坐标
- **Operator**：结合任务、计划、子目标、进度、笔记、最近动作/错误历史、Tips、Shortcuts，输出下一个动作

动作空间不只包含原子操作（Tap / Swipe / Type / Enter / Switch App 等），还包含 **Shortcuts**。  
Shortcut 本质上是“带前置条件的多步动作宏”，例如 `Tap_Type_and_Enter`。

#### 3. Action Reflector：把错误显式化
Action Reflector 比较动作前后截图，判断上一步是：
- 成功或部分成功
- 失败且跳到错误页面
- 失败且页面没变化

这一步很关键，因为它把“轨迹已经偏了”从隐式问题变成显式监督信号。  
如果连续出错，系统会触发 **Error Escalation**，把问题提升给 Manager，从高层重写当前子目标或修正计划，而不是让 Operator 在局部死磕。

#### 4. Notetaker：专门维护任务相关信息
复杂任务往往需要边查边记。  
Notetaker 的作用是把价格、评论摘要、餐厅电话、候选项比较结果这类信息持续聚合成任务记忆，避免 agent 来回重复翻页面。

#### 5. Self-Evolution：任务结束后再“复盘”
每个任务结束后，两个 Experience Reflectors 会根据整条轨迹更新长期记忆：

- **Tips**：自然语言形式的通用经验  
  例如“比较多平台价格时要确认型号完全一致”
- **Shortcuts**：带前置条件的可执行操作序列  
  例如“先点搜索框，再输入，再回车”

作者把这两类知识类比为：
- Tips ≈ episodic-style lessons
- Shortcuts ≈ procedural knowledge

这不是在线训练底模，而是**用冻结 LMM 做经验压缩与再注入**。

### 核心直觉

**1. 把“想什么”与“点哪里”分开**  
过去：同一个 agent 同时承担高层任务分解和低层像素级动作选择。  
改变：Manager 维护任务方向，Operator 专注当前界面动作。  
结果：**高层不再被局部视觉噪声污染**，长程任务中子目标更稳定。

**2. 把“局部修错”与“高层改计划”分开**  
过去：agent 一旦点错，往往会在错误页面继续局部操作，形成 error loop。  
改变：Action Reflector 显式判错；连续错误时升级到 Manager 重规划。  
结果：**错误恢复从局部试错变成结构化回退**，终止错误显著下降。

**3. 把“经验”拆成两种不同的可迁移知识**  
过去：每次搜索、记笔记、切 App 都从零做。  
改变：
- Tips 改变决策偏置，减少重复犯错；
- Shortcuts 改变动作粒度，压缩重复子程序。  
结果：**能力提升和效率提升同时发生**，而不是二选一。

更因果地说，这个设计之所以有效，是因为复杂手机任务失败的主要来源并不是单次点击精度，而是：
- 高层目标漂移；
- 低层错误无法被及时升维处理；
- 重复子程序反复耗费推理预算。  
Mobile-Agent-E 分别用层级控制、错误升级、长期记忆对这三个瓶颈逐一对症下药。

### 战略取舍

| 设计 | 改变的瓶颈 | 主要收益 | 代价 / 风险 |
|---|---|---|---|
| Manager / Operator 分层 | 高层规划与低层 grounding 混杂 | 更稳的长程计划与跨 App 子目标切换 | 更多 agent 编排与调用成本 |
| Action Reflector + Error Escalation | 局部错误难以打断 | 更强错误恢复，减少死循环 | 依赖前后截图判错质量 |
| Notetaker | 探索式任务中信息易丢失 | 支持比较、汇总、跨页面记忆 | 需要持续维护状态一致性 |
| Tips 长期记忆 | 跨任务不学习 | 降低重复犯错概率 | Tips 可能过时或泛化过度 |
| Shortcuts + 前置条件 | 重复子程序成本高 | 减少多步重复推理，提速 | 若前置条件判断错，会更快出错 |

## Part III：证据与局限

### 关键实验信号

- **信号 1：同骨干公平比较，层级架构本身就有效**  
  在相同 GPT-4o 骨干下，Mobile-Agent-E 把 Satisfaction Score 从 Mobile-Agent-v2 的 **53.0% 提升到 75.1%**，同时 Termination Error 从 **52.0% 降到 32.0%**。  
  这说明收益不是因为换了更强底模，而是来自**高低层解耦 + 反思闭环**。

- **信号 2：自演化不是装饰，而是持续增益来源**  
  加上 self-evolution 后，GPT-4o 上进一步到 **86.9% SS / 12.0% TE**。  
  作者还按任务顺序分析，发现越靠后的任务通常收益越大，符合“经验越积越有用”的预期。

- **信号 3：Tips 有独立价值，不只是 Shortcut 压缩动作**  
  在不使用新 Shortcut 的对照子集上，GPT-4o 的 Satisfaction Score 仍从 **79.7% 提升到 87.5%**。  
  这说明自演化不仅是在“把三步变一步”，还在**改变 agent 的决策偏置与操作习惯**。

- **信号 4：跨骨干有效，且效率没有因多 agent 明显崩掉**  
  在 Gemini / Claude / GPT-4o 上，相对 Mobile-Agent-v2，Mobile-Agent-E 的 SS 平均绝对提升为 **+15.6**，加 Evo 后达到 **+22.1**。  
  同时 SSS 曲线更高更陡，Shortcut 使用比例约 **12%–14%**，说明其确实在压缩重复子程序。

一个很值得注意的细节是：**Reflection Accuracy 本来就已经很高**（v2 的 GPT-4o 上为 96.7%，E 也只是到 97.4/97.8）。  
这意味着性能跃迁的主要来源并不是“更会判错”，而是**更会规划、执行和在高层纠错**。

### 局限性

- **Fails when**: 当前 UI 状态被误感知，导致 Shortcut 前置条件判断错误；或自生成 Shortcut 本身有缺步/冗余步，后续调用会放大错误。
- **Assumes**: 依赖冻结但较强的多模态基础模型（GPT-4o / Claude / Gemini）与多组件感知栈；评测依赖人工 rubric 与人工打分；自演化实验在 scenario 内顺序执行，且更新记忆时可访问剩余任务查询，这会影响“纯在线终身学习”设定下的可比性。
- **Not designed for**: 支付、登录、隐私敏感操作的生产级全自动执行；对长期记忆检索扩展性的定量验证（论文只做了 retrieval case study）；对 Shortcut 正确性的形式化保证。

补充地说，这篇工作的复现虽然代码和数据开源，但强结果仍明显依赖：
- 闭源 API 级 LMM；
- 真机 + ADB 的动态执行环境；
- 人工评价流程。  

另外，证据虽然覆盖三种 backbone，但**核心实验仍集中在单一 benchmark（Mobile-Eval-E，25 个任务）**，因此更合理的证据评级是 **moderate**，而不是 strong。

### 可复用组件

这篇 paper 里最值得迁移到别的 GUI / agent 系统的，不一定是完整框架，而是这几个操作子：

1. **高层计划 / 低层动作分治**
2. **动作前后截图反思 + 连续错误升维处理**
3. **把长期记忆拆成 Tips 与带前置条件的 Shortcuts**
4. **探索任务中的 Notetaker 状态聚合**
5. **面向开放任务的人类对齐评测：Satisfaction Score 与 SSS curve**

## Local PDF reference

![[paperPDFs/Agentic_Reasoning_Applications_Autonomous_Web_Exploration_Research_Agents_Self_evolving_agentic_reasoning/arXiv_2025/2025_Mobile_Agent_E_Self_Evolving_Mobile_Assistant_for_Complex_Tasks.pdf]]