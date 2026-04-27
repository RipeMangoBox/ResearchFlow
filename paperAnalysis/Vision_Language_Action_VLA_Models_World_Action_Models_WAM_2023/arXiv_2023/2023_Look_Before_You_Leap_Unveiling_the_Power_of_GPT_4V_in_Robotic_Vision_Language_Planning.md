---
title: "Look Before You Leap: Unveiling the Power of GPT-4V in Robotic Vision-Language Planning"
venue: arXiv
year: 2023
tags:
  - Embodied_AI
  - task/robotic-planning
  - task/robotic-manipulation
  - closed-loop-planning
  - multimodal-prompting
  - visual-feedback
  - dataset/RAVENS
  - opensource/no
core_operator: "把当前场景图像、任务目标和执行历史一起输入 GPT-4V，由其联合视觉-语言推理生成原语步骤，并在每步执行后基于新观测重规划。"
primary_logic: |
  当前环境图像 + 高层语言/目标图像 + 已完成步骤历史
  → GPT-4V识别任务相关对象并生成多步文本计划
  → 只执行首个原语动作、获取新视觉反馈并继续重规划
  → 输出直到 done 的可执行长时程计划
claims:
  - "On 8 real-world manipulation tasks requiring visually grounded commonsense, VILA achieves 80% average success rate, exceeding SayCan's 13% and Grounded Decoding's 20% [evidence: comparison]"
  - "Closed-loop VILA substantially outperforms its open-loop variant on 4 dynamic tasks, including Pack Chip Bags (100% vs 0%) and Find Stapler (90% vs 30%) [evidence: comparison]"
  - "In RAVENS-based simulation, VILA outperforms CLIPort, LLM-only planners, and Grounded Decoding across both seen and unseen task groups, reaching 78.3%-88.3% on seen groups and 81.0%-82.0% on unseen groups [evidence: comparison]"
related_work_position:
  extends: "GPT-4V(ision) (OpenAI 2023)"
  competes_with: "SayCan (Ahn et al. 2022); Grounded Decoding (Huang et al. 2023)"
  complementary_to: "RT-2 (Brohan et al. 2023); CLIPort (Shridhar et al. 2021)"
evidence_strength: moderate
pdf_ref: paperPDFs/Vision_Language_Action_VLA_Models_World_Action_Models_WAM_2023/arXiv_2023/2023_Look_Before_You_Leap_Unveiling_the_Power_of_GPT_4V_in_Robotic_Vision_Language_Planning.pdf
category: Embodied_AI
---

# Look Before You Leap: Unveiling the Power of GPT-4V in Robotic Vision-Language Planning

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2311.17842), [Project](https://robot-vila.github.io/)
> - **Summary**: 该工作把 GPT-4V 直接作为机器人高层规划器，让视觉观测进入推理主回路，并通过“执行一步、再看一步”的闭环机制显著提升开放场景长时程操作规划能力。
> - **Key Performance**: 真实机器人中 8 个依赖视觉常识的任务平均成功率 80%（SayCan 13%，GD 20%）；RAVENS 仿真四个任务组上 VILA 均为最佳，成功率约 78.3%–88.3%（seen）与 81.0%–82.0%（unseen）。

> [!info] **Agent Summary**
> - **task_path**: 当前场景图像 + 高层语言/目标图像 -> 文本原语动作序列 -> 逐步执行与重规划
> - **bottleneck**: LLM 规划与外部感知/affordance 模块割裂，导致任务相关视觉信息无法在推理时被联合选择和利用
> - **mechanism_delta**: 用 GPT-4V 同时承担“看场景”和“拆任务”，并采用只执行首步的闭环重规划
> - **evidence_signal**: 真实机器人视觉常识任务上 80% 成功率且显著优于 SayCan/GD，动态任务上闭环远胜开环
> - **reusable_ops**: [first-step-only closed-loop replanning, image+language goal prompting]
> - **failure_modes**: [输出不严格落入预定义原语语法, 视觉域差或合成场景识别不足导致规划偏差]
> - **open_questions**: [能否用开源VLM复现GPT-4V级规划性能, 如何把高层VLM规划与可泛化低层技能联合训练]

## Part I：问题与挑战

这篇论文解决的是**开放世界中的长时程机器人操作规划**：  
输入不是结构化符号状态，而是**当前场景图像**与**高层任务描述**；输出不是连续控制，而是可映射到技能库的一串**文本原语动作**。

### 真正的问题是什么？
作者认为，先前 LLM 规划器的核心短板不只是“没有视觉”，而是：

**视觉信息进入规划器的方式错了。**

典型做法如 SayCan、Grounded Decoding，通常是：
1. 用外部 affordance / detection 模块看场景；
2. 再把结果摘要给 LLM；
3. LLM 基于这些摘要做规划。

这会带来两个根本问题：

- **信息瓶颈**：视觉被压缩成少量文本、分数或检测结果，复杂空间关系很容易丢失。
- **任务错配**：独立感知模块并不知道当前任务真正关心什么，因此无法主动提供“任务相关”的视觉细节。

所以它们在下面几类任务上特别脆弱：
- **空间布局理解**：目标被遮挡、被其它物体挡住、或藏在抽屉/冰箱里。
- **属性理解**：同一个物体在不同上下文里属性不同，比如剪刀在“儿童艺术课”里是工具，不一定是危险品。
- **动态环境**：执行失败、人类介入、目标不在当前可见区域时，需要依赖反馈继续找和改。

### 为什么现在值得做？
因为 GPT-4V 这一类 VLM 已经具备了足够强的：
- 图像理解能力，
- 常识推理能力，
- 语言规划能力。

这意味着“让同一个模型既看图又规划”第一次变得可行。  
换句话说，作者抓住的时间点是：**VLM 已经强到可以把视觉 grounding 从外围插件，升级成规划过程本身。**

### 输入/输出接口与边界条件
- **输入**：
  - 当前视觉观测 \(x_t\)
  - 高层语言指令 \(L\)
  - 可选目标图像 \(x_g\)
  - 已执行步骤历史
- **输出**：
  - 一串文本动作，如 `pick up blue container`，每步映射到已有 primitive skill
- **前提假设**：
  - 当前图像能较准确代表世界状态
  - 机器人已经拥有所需的低层技能
  - 本文不解决低层控制学习，只解决高层规划

论文中实际使用的 primitive skills 只有 5 类：
`pick up / place in-on / pour / open / close`。  
所以它研究的边界非常清楚：**高层规划如何更好地“看后再动”**。

---

## Part II：方法与洞察

### 方法主线
VILA 的流程很简单，但关键在于接口设计：

1. 把**当前场景图像 + 高层任务 + 已完成步骤**一起输入 GPT-4V。
2. 在 prompt 中显式约束它只能输出预定义 primitive skills。
3. GPT-4V 先识别任务相关物体，再生成一个多步计划。
4. 机器人**只执行第一步**。
5. 获取新图像后，再把新观测和已执行历史喂回去重规划。
6. 直到模型输出 `done`。

这相当于把规划从“一次性生成整条长链”改成了：

**每一步都重新 grounding 的局部决策过程。**

### 它具体改变了什么？
作者强调了 VILA 的三个能力来源：

#### 1. 视觉世界中的常识理解
VILA 不是先把视觉变成简短标签再规划，而是直接在图像上推理，因此更容易处理：

- **空间布局**：  
  如目标被纸杯、可乐罐挡住，必须先移开障碍物；
  或目标不在视野中，但从场景可推断它“可能在柜子/冰箱里”，因此应先开门搜索。

- **物体属性**：  
  不是只识别“这是什么”，而是识别“在当前任务下它意味着什么”。  
  例如艺术课场景里，剪刀保留、刀具收走。

#### 2. 灵活的目标指定
目标不必只靠文字。VILA 支持：
- 语言目标，
- 图像目标，
- 图像 + 语言混合目标。

因此它可以做：
- 按参考图摆寿司，
- 按儿童画摆拼图，
- 根据手指指向决定要放到哪个盘子，
- 参考桌面图片但再附加“交换杯子与笔筒位置”的语言条件。

#### 3. 原生视觉反馈
在动态环境里，VILA 的闭环机制让它能：
- 发现执行失败并补救，
- 继续搜索未找到的目标，
- 检查任务是否真的完成，
- 等待人类动作后再继续。

这里 VLM 同时承担了：
- **scene descriptor**
- **success detector**
- **replanner**

### 核心直觉

真正的机制变化可以概括为：

**从“LLM + 单向感知摘要”变成“VLM 直接做任务条件化联合推理”。**

更具体地说：

- **原来变的是什么？**  
  视觉只是外接模块，给 LLM 提供一个低带宽、任务无关的描述。

- **现在变的是什么？**  
  图像、语言目标、执行历史在同一个模型里共同决定下一步动作。

- **因此哪个瓶颈被改变了？**  
  被改变的是**任务相关信息选择**这个瓶颈：  
  模型不再被动接受别人总结好的场景，而是自己决定“当前任务下哪些视觉线索最重要”。

- **能力上带来什么变化？**  
  能处理之前最难的四类情况：
  1. 遮挡与阻挡
  2. 隐藏目标搜索
  3. 上下文依赖属性判断
  4. 动态反馈下的重规划

再进一步，**“只执行第一步”**是另一个关键旋钮。  
它把长时程规划的风险从“整条轨迹一次出错就崩”变成“局部一步出错还能纠正”。  
这就是为什么它在找订书机、等待人手、堆叠失败回补这些任务里提升特别明显。

### 战略性取舍

| 设计选择 | 带来的能力提升 | 代价/风险 |
|---|---|---|
| 用 GPT-4V 直接替代 `LLM + affordance` 组合 | 视觉与语言联合推理，减少任务相关信息丢失 | 强依赖大规模 VLM，且是黑盒 API |
| 只执行首步、每步重规划 | 对动态环境和技能失败更鲁棒 | 推理调用次数更多，延迟和成本更高 |
| 支持图像/图像+文本目标 | 目标表达更自然，减少语言压缩损失 | 对跨域图像理解更敏感 |
| 使用固定 primitive skill 集合 | 把高层规划问题隔离得更干净 | 性能上限受技能库覆盖度约束 |
| 真实世界严格 zero-shot prompt | 减少 prompt engineering 和示例依赖 | 输出格式更容易漂移，结构错误更难控 |

---

## Part III：证据与局限

### 关键证据

#### 1. 最强证据：真实机器人上的“视觉常识”任务
作者设计了 8 个必须依赖视觉 grounding 常识的任务。  
VILA 平均成功率 **80%**，而：
- SayCan: **13%**
- Grounded Decoding: **20%**

这组结果最能说明问题，因为这些任务正好卡在 prior work 的痛点上：
- `Bring Empty Plate`：盘子上有苹果和香蕉，必须先清空再拿盘子；
- `Take Out Marvel Model`：目标被障碍物挡住，不能直接抓；
- `Prepare Art Class`：要结合场景语境判断哪些物品不合适。

这不是简单“看见物体”就能解决，而是要**看懂关系和语境**。

#### 2. 多模态目标不是噱头，确实能用
在 4 个图像/混合目标任务上，VILA 分别达到：
- Arrange Sushi：80%
- Arrange Jigsaw Pieces：100%
- Pick Vegetables：100%
- Tidy Up Study Desk：60%

这里的信号是：VILA 不只是“根据图回答问题”，而是能把图像目标转成**可执行的长程动作计划**。

#### 3. 闭环反馈是实打实的增益
和 open-loop 版本相比，闭环版 VILA 在 4 个动态任务上显著提升：
- Stack Blocks：20% → 90%
- Pack Chip Bags：0% → 100%
- Find Stapler：30% → 90%
- Human-Robot Interaction：20% → 80%

这说明性能跃迁不只来自“更会看”，还来自**每步都重新 grounding**。

#### 4. 仿真 RAVENS 上也不是只靠少数 demo
在 RAVENS 的 seen / unseen 任务组上，VILA 都显著优于：
- CLIPort
- LLM-only planner
- Grounded Decoding

尤其 important 的是 unseen 组仍维持 **81%–82%** 左右成功率，说明它的提升不只是背 prompt 模板。

#### 5. 误差分析支持其因果解释
作者的 error breakdown 显示，baseline 的主要错误来源是：
- understanding error

而 VILA 明显降低了这类错误。  
这和论文主张是对齐的：**真正的收益来自视觉-语言联合推理，而不仅仅是更好的识别器。**

### 局限性
- **Fails when**: 视觉域差较大或图像风格不自然时，GPT-4V 的识别/推理会掉性能；输出若不严格满足 primitive skill 语法也会导致执行失败；当任务需要更细粒度物理推理或技能库中不存在的动作时，方法无法补足。
- **Assumes**: 当前图像足以准确代表世界状态；已有稳健的 5 类 primitive skills；可访问闭源 GPT-4V API；真实实验依赖固定视角相机与 Franka Panda 平台；仿真中的 Blocks & Bowls 甚至需要 3 个 in-context examples 来弥补合成域差。
- **Not designed for**: 低层控制学习、端到端动作生成、形式化安全保证、强部分可观测场景下的长期记忆与状态估计。

### 复现与扩展时要特别注意
- 这是一个**强依赖闭源模型**的方法，复现成本不只是代码，还包括 API 可用性、价格、延迟和模型版本漂移。
- 作者自己也承认黑盒 VLM 限制了 steerability 和错误解释。
- 论文把低层技能问题“假设掉”了，因此如果你换到更复杂的真实机器人平台，高层规划优势未必能无缝兑现。

### 可复用组件
这篇论文最值得迁移的不是某个 prompt，而是以下三个操作模式：

1. **first-step-only 闭环重规划**  
   适合所有长时程 embodied planning。
2. **VLM 兼任场景理解 + 成功检测**  
   减少额外感知模块接口错配。
3. **图像目标 + 语言约束的混合目标接口**  
   很适合整理、摆放、示范跟随类任务。

**一句话总结 So what**：  
VILA 的能力跃迁，不在于它比以前“更会说计划”，而在于它把**任务相关视觉信息**真正纳入了规划因果链，因此在遮挡、语境属性、多模态目标和动态反馈这几类 prior work 最脆弱的场景里出现了明显台阶式提升。

![[paperPDFs/Vision_Language_Action_VLA_Models_World_Action_Models_WAM_2023/arXiv_2023/2023_Look_Before_You_Leap_Unveiling_the_Power_of_GPT_4V_in_Robotic_Vision_Language_Planning.pdf]]