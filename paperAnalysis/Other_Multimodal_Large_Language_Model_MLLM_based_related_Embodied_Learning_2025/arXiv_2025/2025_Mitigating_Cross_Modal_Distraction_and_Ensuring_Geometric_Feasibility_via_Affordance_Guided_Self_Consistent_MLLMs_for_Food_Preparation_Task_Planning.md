---
title: "Mitigating Cross-Modal Distraction and Ensuring Geometric Feasibility via Affordance-Guided and Self-Consistent MLLMs for Task Planning in Instruction-Following Manipulation"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/instruction-following-manipulation
  - task/closed-loop-task-planning
  - chain-of-thought
  - self-consistency
  - affordance-guidance
  - dataset/QuARC
  - opensource/partial
core_operator: 用跨迭代自一致的 CoT 稳定 MLLM 技能选择，再用谓词化 affordance 前置条件过滤并重规划几何上不可执行的动作
primary_logic: |
  自然语言指令 + 当前图像/对象列表 + 历史动作与失败反馈 → MLLM 先生成整段技能序列并做跨迭代自一致校验，再由 affordance 谓词检查可达性与碰撞前置条件并在失败时重规划 → 当前可执行原子技能与闭环任务计划
claims:
  - "在 QuARC 基准上，所提方法在无需额外微调的情况下达到 76.7% 总成功率，显著高于 ViLa 基线的 36.7% [evidence: comparison]"
  - "在仅需语义推理的任务中，加入图像输入会使成功率从 Naive LLM 的 86.7% 降到 Naive MLLM 的 26.7%，说明跨模态干扰会直接破坏规划 [evidence: comparison]"
  - "CoT+SC 阶段共出现 18 次不一致技能选择，其中只有 50% 被正确修正，表明自一致验证能提升稳定性但仍不足以完全解决闭环错误传播 [evidence: analysis]"
related_work_position:
  extends: "Self-Consistency (Wang et al. 2023)"
  competes_with: "ViLa (Hu et al. 2024)"
  complementary_to: "SCONE (Tai et al. 2023); Integrated Task and Motion Planning (Garrett et al. 2021)"
evidence_strength: moderate
pdf_ref: paperPDFs/Other_Multimodal_Large_Language_Model_MLLM_based_related_Embodied_Learning_2025/arXiv_2025/2025_Mitigating_Cross_Modal_Distraction_and_Ensuring_Geometric_Feasibility_via_Affordance_Guided_Self_Consistent_MLLMs_for_Food_Preparation_Task_Planning.pdf
category: Embodied_AI
---

# Mitigating Cross-Modal Distraction and Ensuring Geometric Feasibility via Affordance-Guided and Self-Consistent MLLMs for Task Planning in Instruction-Following Manipulation

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2503.13055), [Project/Code/Data](https://hcis-lab.github.io/Affordance-Guided-Self-Consistent-MLLM)
> - **Summary**: 该文提出 QuARC 食物准备闭环规划基准，并用“跨迭代自一致 CoT + affordance 前置条件重规划”缓解 MLLM 在指令跟随操控中的跨模态干扰与几何不可行问题。
> - **Key Performance**: QuARC 总成功率 76.7%；显著高于 ViLa 基线 36.7%。

> [!info] **Agent Summary**
> - **task_path**: 自然语言指令 + 当前俯视图/对象列表 + 历史动作与失败反馈 -> 原子技能序列 -> 当前可执行技能
> - **bottleneck**: 图像输入会干扰本应由语义完成的规划推理，且纯 MLLM 不会显式检查可达性与碰撞等几何前提
> - **mechanism_delta**: 让 MLLM 一次生成整段计划并用历史计划做跨迭代自一致校验，再用谓词化 affordance 过滤不可执行动作并触发重规划
> - **evidence_signal**: QuARC 上 76.7% 成功率优于 ViLa 的 36.7%，且语义任务中图像输入会把成功率从 86.7% 拉低到 26.7%
> - **reusable_ops**: [跨迭代计划缓冲区自一致校验, 谓词化前置条件反馈驱动重规划]
> - **failure_modes**: [停止条件判断仍不稳定, 依赖可靠目标检测与手工技能/谓词]
> - **open_questions**: [真实机器人与感知噪声下是否仍有同等收益, 能否把手工 affordance 扩展为学习式连续几何评估]

## Part I：问题与挑战

这篇工作的核心问题不是“MLLM 会不会看图”，而是：

1. **MLLM 在闭环操控中会不会被图像干扰到不会规划**  
2. **即使它规划了，动作序列是否真的几何可执行**

### 任务定义
作者研究的是一个闭环的 instruction-following manipulation 场景：食物准备。  
每一轮输入包括：

- 自然语言指令 `I`
- 当前视觉观测 `O_t`
- 预定义技能集 `Π`
- 历史已执行动作
- 若前一动作不可执行，还会追加 affordance 失败反馈

输出是当前轮应执行的一个原子技能，直到输出 `DONE`。

### 真正难点在哪
作者把成功规划拆成四个必要条件：

- **Quantity Estimation**：判断“更满的碗”“一些/两勺”等数量相关描述
- **Reachability Analysis**：目标能不能够到，不够到时要不要先挪
- **Relative Positioning**：理解“在工具架旁边”“黄色碗旁边”等相对位置
- **Collision Avoidance**：动作是否会撞到挡路的碗或 dumbwaiter 门

这里的真实瓶颈有两个：

#### 1. Cross-Modal Distraction
一些任务本来只靠文字就能做对，但一旦把图像也塞进 prompt，模型反而开始：

- 重复动作
- 漏动作
- 错误停机
- 误把图像中的无关细节当成决策依据

也就是说，**问题不是信息不够，而是多模态信息在错误时机进入后扰乱了决策分布**。

#### 2. Geometric Feasibility
即便 MLLM 理解了指令，它也不天然知道：

- 这个碗现在是否够得到
- 勺子是否在手上
- dumbwaiter 门会不会撞到旁边的碗
- 当前动作是否满足执行前提

因此，**MLLM 的语义推理能力和机器人动作的几何可行性之间存在断层**。

### 为什么现在值得做
因为当前最有吸引力的路线是：**不微调，直接用通用 MLLM + in-context learning 做规划**。  
这条路线成本低、适配快，但如果没有稳定性与可行性护栏，就很难落到真实 embodied setting。

### 边界条件
这篇论文的设定比较明确，也意味着结果边界比较清楚：

- 单机械臂台面食物准备场景
- IsaacGym 仿真
- 固定原子技能库
- 单视角高角度图像
- 假设有可靠 object detector，直接给出 bowl/content list
- 重点是**高层技能规划**，不是低层控制学习

---

## Part II：方法与洞察

作者没有去训练一个新的端到端机器人模型，而是给 API 级 MLLM 加了两个“护栏”：

1. **自一致护栏**：减少跨模态干扰导致的技能漂移  
2. **可行性护栏**：把几何约束从“让 MLLM 猜”改成“先验检查 + 明确反馈”

### 方法骨架

#### 1. Zero-Shot CoT 规划
MLLM 不直接只输出“下一步动作”，而是先输出：

- 当前子目标描述
- 从当前轮到 `DONE` 的完整技能序列

这样做的作用有两个：

- 把一步决策变成带长期目标的序列推理
- 给后续 self-consistency 提供“未来动作样本”

#### 2. 跨迭代 Self-Consistency Verification
标准 self-consistency 往往要对同一道题多次采样再投票，代价很高。  
这篇文章的改法是：

- 每轮让模型生成**整段计划**
- 当前轮只执行其中第一个动作
- 剩余动作存入 sequence buffer
- 下一轮时，把当前候选动作与历史计划里“该时间步多数会做什么”进行比较
- 若冲突，就让 MLLM 在“当前新计划”和“历史多数计划”之间重新判断

关键点在于：  
**它不是每轮额外多采样，而是复用前几轮已经生成过的完整计划。**

这使得 self-consistency 从“单题多样本投票”变成了“闭环多轮历史一致性检查”。

#### 3. Skill Affordance + Replan
作者设计了一组二值谓词来表达几何/执行前提，例如：

- `spoon_on_hand`
- `food_on_hand`
- `dumbwaiter_opened`
- `close_to_target`
- `obstacle_blocked_holder`
- `obstacle_blocked_dumbwaiter`
- `reachable`

这些谓词再组成每个技能的前置条件。  
如果 MLLM 选出的动作不可执行，就返回结构化失败原因，例如：

- “目标碗太远”
- “勺子还在手上，不能 pull bowl”
- “有障碍挡住 dumbwaiter 门”

然后触发重规划。  
并且，**一旦重规划，历史 buffer 会被清空**，避免旧的错误计划继续污染后续一致性判断。

### 核心直觉

**改了什么**：  
从“图像+文本直接解码下一步动作”改成“先生成整段计划，再让一致性模块和 affordance 模块共同筛掉不稳定/不可执行的动作”。

**改变了哪个瓶颈**：  

- Self-consistency 改变的是**输出稳定性约束**：把跨模态干扰造成的随机漂移显式化为“与历史多数冲突”
- Affordance 改变的是**信息瓶颈**：把 MLLM 最不可靠的几何判断外包给可验证谓词，而不是让模型从图像里隐式猜 reachability/collision

**带来了什么能力变化**：  

- 更不容易重复、漏掉或误停
- 在 reachability / collision 相关任务上不再盲目执行
- 出错时能拿到有方向性的反馈，而不是纯粹重新问一遍模型

**为什么它因果上成立**：  
如果当前环境真的发生变化，那么“当前计划和历史多数不一致”是合理的，MLLM 可以保留新动作；  
如果环境没本质变化，那这种不一致更可能是视觉噪声或解码不稳定，自一致模块就相当于一个低成本正则器。  
而 affordance 失败反馈把“为什么不行”明确告诉模型，缩小了重规划搜索空间。

### 战略取舍

| 设计选择 | 改变的瓶颈 | 带来的能力 | 代价/风险 |
|---|---|---|---|
| 一次生成整段技能序列 | 单步贪心、弱终止意识 | 更好维持长期子目标与 stop condition | 整段计划可能整体偏航，对 prompt 格式较敏感 |
| 跨迭代 self-consistency | 图像引入的技能漂移 | 不额外多采样也能做稳定化 | 历史多数本身也可能错，修正能力有限 |
| 谓词化 affordance + 失败反馈 | MLLM 缺少显式几何前提 | 能处理可达性与碰撞 | 依赖手工 predicates、检测器和领域知识 |

---

## Part III：证据与局限

### 关键证据

#### 1. 比较信号：整体成功率显著提升
作者方法在 QuARC 上达到 **76.7%**，而文中 ViLa 基线为 **36.7%**。  
这说明：**单靠 CoT 风格的 MLLM 规划不够，必须加稳定性与可行性护栏。**

#### 2. 诊断信号：图像会伤害本来能做对的任务
在只需要语义推理的任务上：

- Naive LLM：**86.7%**
- Naive MLLM：**26.7%**

这个差距非常关键，因为它直接支持论文最核心的诊断：  
**并不是“更多模态一定更好”，无关视觉输入会实质性破坏规划。**

#### 3. 机制信号：CoT 与 SC 主要修的是“规划稳定性”，affordance 修的是“可执行性”
论文分析指出：

- CoT 能让模型在解释时说出正确理由，说明问题不完全是“不会想”，而是“直接出动作时不稳定”
- Self-consistency 主要帮助 stop condition 等长程目标一致性
- 但 CoT+SC 中一共 18 次不一致技能选择，仅有 **50%** 被正确修正，说明这一步是有效但不充分
- Reachability / Collision 相关能力提升主要来自 affordance 前提检查与失败反馈

### 1-2 个关键指标怎么读
- **76.7%**：证明“低成本 API 级护栏”对闭环 embodied planning 有现实增益
- **86.7% → 26.7%**：证明论文提出的 cross-modal distraction 不是描述性概念，而是可测的性能崩塌

### 局限性
- **Fails when**: 任务需要更细粒度的数量/空间判断时（如“半勺”“稍微靠左一点”）、停止条件本身很模糊时、或检测结果与图像不一致时，系统仍可能重复动作、误停或选错目标。
- **Assumes**: 使用闭源 GPT-4o API；假设可靠 object detector 能输出 bowl 及内容物；依赖手工定义的原子技能、预定义轨迹、谓词和前置条件；主要在 IsaacGym 仿真中验证。
- **Not designed for**: 端到端低层控制学习、开放世界未知物体操作、真实软食物复杂接触动力学、无需显式 detector 的纯视觉 grounding。

### 可复用组件
- **跨迭代 sequence buffer 自一致校验**：适合任何闭环 LLM/MLLM agent，不必每步多次采样
- **谓词化 affordance → 文本失败反馈 → 重规划**：可迁移到其他具备显式技能前提的机器人场景
- **QuARC 基准**：适合系统性测试 quantity / reachability / relative positioning / collision 四类能力

### So what
这篇工作的价值不只是“把一个 benchmark 做高了”，而是给出一个很清楚的系统结论：

- **MLLM 在 embodied planning 中的短板，往往不是知识不足，而是多模态干扰下的决策不稳定**
- **几何可行性不应完全寄希望于 MLLM 内隐推断，最好通过外部可验证约束显式注入**

这让“API-only MLLM 规划器”从 demo 风格，向更可控的闭环机器人规划器迈了一步。

## Local PDF reference
![[paperPDFs/Other_Multimodal_Large_Language_Model_MLLM_based_related_Embodied_Learning_2025/arXiv_2025/2025_Mitigating_Cross_Modal_Distraction_and_Ensuring_Geometric_Feasibility_via_Affordance_Guided_Self_Consistent_MLLMs_for_Food_Preparation_Task_Planning.pdf]]