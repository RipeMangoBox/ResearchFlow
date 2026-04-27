---
title: "FlexVLN: Flexible Adaptation for Diverse Vision-and-Language Navigation Tasks"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/vision-and-language-navigation
  - hierarchical-planning
  - multi-model-integration
  - multimodal-verification
  - dataset/R2R
  - dataset/REVERIE
  - dataset/SOON
  - dataset/CVDN-target
  - opensource/no
core_operator: 用LLM把跨数据集的高层VLN指令编译成受限动作空间内的细粒度guidance，再由集成式Instruction Follower执行，并以MLLM验证可行性来抑制幻觉。
primary_logic: |
  OOD高层导航指令 + 多视角环境观测 + 导航历史
  → LLM Planner生成受限动作短语组成的细粒度guidance，并由MLLM检查其可执行性
  → 多模型Instruction Follower执行低层动作、分歧时再由LLM仲裁
  → 到达终点区域后用Object Locator完成目标物定位
claims:
  - "在不对目标数据集进行额外训练或微调的设定下，FlexVLN (GPT-4o) 在 REVERIE val-unseen 上达到 39.80 SR / 29.74 SPL，显著高于 MapGPT 与直接从 R2R 迁移的监督式基线 [evidence: comparison]"
  - "在更复杂的 SOON val-unseen house 上，FlexVLN (GPT-4o) 取得 23.08 SR / 6.94 RGSPL，超过 in-domain GBE，并远高于所有 R2R-only 直接迁移基线 [evidence: comparison]"
  - "在 REVERIE 消融集上，去掉可行性验证会使 SR 从 37 降到 34、SPL 从 26.38 降到 22.83；去掉动作短语约束会使 SR 进一步降到 30、SPL 降到 20.88 [evidence: ablation]"
related_work_position:
  extends: "NavGPT (Zhou et al. 2024)"
  competes_with: "MapGPT (Chen et al. 2024); NavGPT (Zhou et al. 2024)"
  complementary_to: "AutoVLN (Chen et al. 2022); PanoGen (Li and Bansal 2024)"
evidence_strength: strong
pdf_ref: paperPDFs/Vision_Language_Navigation_VLN_Models_2025/arXiv_2025/2025_FlexVLN_Flexible_Adaptation_for_Diverse_Vision_and_Language_Navigation_Tasks.pdf
category: Embodied_AI
---

# FlexVLN: Flexible Adaptation for Diverse Vision-and-Language Navigation Tasks

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2503.13966)
> - **Summary**: 这篇工作把“跨数据集高层导航指令理解”交给 LLM Planner，把“局部精确移动执行”交给在 R2R 上训练好的 Instruction Follower，并通过可行性验证与分歧仲裁把两者接成一个可零微调跨任务泛化的层次式 VLN 系统。
> - **Key Performance**: REVERIE val-unseen 上达到 **39.80 SR / 29.74 SPL**；SOON val-unseen house 上达到 **23.08 SR / 6.94 RGSPL**（均为 GPT-4o 版本）。

> [!info] **Agent Summary**
> - **task_path**: 高层/OOD VLN指令 + 当前多视角观测 + 导航历史 -> 细粒度guidance -> 导航轨迹与目标物定位
> - **bottleneck**: 监督式VLN模型会学到数据集特定的指令风格，遇到高层OOD指令就失效；而纯LLM逐步选动作又缺乏精确局部执行能力且成本高
> - **mechanism_delta**: 把“LLM每步直接选节点”改成“LLM间歇式生成可执行guidance，监督式follower负责低层执行，验证模块过滤不可行计划，模型分歧时再让LLM仲裁”
> - **evidence_signal**: 3个OOD benchmark 上均显著优于直接迁移的 R2R followers 与 LLM-only 基线，且关键模块有消融支撑
> - **reusable_ops**: [OOD指令重写到受限动作空间, guidance可行性验证后再执行]
> - **failure_modes**: [环境感知漏掉关键出口/物体导致规划偏航, follower未按guidance及时停步或绕障时偏离]
> - **open_questions**: [如何摆脱闭源API与多外部模型依赖, 如何把同样的泛化能力迁移到连续控制和真实机器人]

## Part I：问题与挑战

这篇论文真正想解决的，不是“再把某个 VLN benchmark 做高一点”，而是一个更接近部署的问题：

**能不能只训练一次导航能力，然后在不同 VLN 数据集、不同指令风格之间直接迁移？**

### 1. 问题是什么
VLN 任务的输入/输出接口是：

- **输入**：自然语言导航指令、当前多视角视觉观测、历史轨迹
- **输出**：到目标区域的导航轨迹，以及终点处的目标物定位

但不同数据集的指令分布差异很大：

- **R2R**：细粒度、一步一步的指令
- **REVERIE / SOON**：更高层、更面向目标和区域探索
- **CVDN-target**：从对话里截出的目标导向短句，信息更稀疏

多数监督式方法即使很强，也还是要对每个数据集单独训练/微调。  
所以**真正的瓶颈不是“不会走路”，而是“不会跨指令分布”**。

### 2. 为什么以前方法不够
作者把现有方法的短板拆成两类：

#### A. 监督式 VLN 很强，但强在“域内”
像 BEVBert、GridMM、ScaleVLN 这类方法在各自训练域内很强，但本质上学到的是：

- 某种指令风格
- 某种导航-语言对齐模式
- 某个 benchmark 的常见目标表达方式

一旦从 R2R 的细粒度描述，跳到 REVERIE/SOON 这种高层探索型指令，性能会明显掉。

#### B. 纯 LLM 导航有泛化，但执行不稳
LLM-based 方法能处理开放语言，但直接让 LLM 每步选动作有三个问题：

1. **局部动作 grounding 不精确**  
   LLM 只知道邻居节点的大致朝向，不知道精确位置，容易混淆同方向不同节点。

2. **绕障时容易偏航**  
   当环境局部不可直行时，LLM 很难在多步局部动作中维持全局计划不漂移。

3. **成本高**  
   每一步都调用 LLM，时延与 token 成本都很重。

### 3. 这篇论文抓到的真瓶颈
**跨任务泛化的核心，不是让一个模型同时“理解所有指令 + 执行所有动作”，而是把这两件事解耦。**

- **LLM 擅长**：理解高层目标、做常识推理、把抽象指令转成可执行意图
- **监督式 follower 擅长**：在离散环境里执行精细局部动作

所以这篇论文的关键判断是：

> 问题不该被建模成“让 LLM 直接导航”，而应该被建模成“让 LLM 先把 OOD 指令编译成 follower 看得懂的细粒度 guidance”。

### 4. 边界条件
这篇工作有明确边界：

- 关注的是**跨数据集/跨指令风格泛化**
- 不是重点研究**未见环境泛化**
- 场景是 **Matterport 离散图导航**
- 终点成功判定依赖目标物在 **3 米可见范围内**
- follower 主要建立在 **R2R 风格细粒度指令** 上

---

## Part II：方法与洞察

FlexVLN 是一个典型的**层次式 embodied policy**：

- 上层：**LLM Planner**
- 下层：**Instruction Follower**
- 末端：**Object Locator**

外加两个关键保护机制：

- **Feasibility Verification**：防 hallucination / 不可执行 guidance
- **Multi-model Integration**：防 follower 单模型执行失误

### 方法主线

整个系统是一个 5 步闭环。

#### Step 1. Environmental Perception
把当前节点的多视角观测先变成文本化环境描述。

- 4 个 90° 视角图像
- 用 **InternVL** 做两轮理解：
  - 先联合理解四张图，形成整体场景认知
  - 再推断当前位置
- 用 **Faster R-CNN + depth** 检测 3m 内物体
- 用高度信息估计所在楼层

得到的不是原始视觉 token，而是更适合 LLM 的结构化文本观察。

#### Step 2. LLM Planning
LLM Planner 输入：

- 原始指令
- 历史导航文本
- 当前环境描述

输出：

- 对历史的总结
- 下一步探索方向
- 一条**细粒度 guidance**

这里有一个关键设计：  
作者**手工定义了受限动作空间**，要求 guidance 只能用这类短语：

- go forward
- turn left/right
- go into / go out of
- go upstairs/downstairs
- go past / go through
- go to
- stop

也就是说，LLM 不是自由发挥，而是在一个 follower 更熟悉的“指令子语言”里说话。

#### Step 3. Feasibility Verification
为了避免 planner 因视觉描述误差或 LLM 幻觉给出不可执行 guidance，作者用 **Qwen2-VL** 检查：

- 当前视角下，这条 guidance 是否可行
- 如果不可行，原因是什么

若不可行，就把反馈发回 planner，重新生成 guidance。

#### Step 4. Guidance Execution
执行层不是单模型，而是 3 个强 follower 的集成：

- **BEVBert**
- **GridMM**
- **ScaleVLN**

执行逻辑是：

- 如果 3 个模型动作一致：直接执行
- 如果不一致：把候选动作转成文本选项，调用 **GPT-4o-mini** 结合当前 guidance 仲裁

此外，planner **不是每一步都调用**。  
只有当前 guidance 被执行完，才再次请求新的 guidance。  
这把 LLM 从“step-level controller”变成了“segment-level planner”。

#### Step 5. Object Localization
到达终点后：

- 用 GPT-4o-mini 从 instruction 抽取目标物
- 用 **BLIP-2** 比较终点节点物体候选与目标文本的相似度
- 选相似度最高者作为定位结果

---

### 核心直觉

#### 1) 改了什么
从：

- **直接把 OOD 指令喂给导航策略**
- 或 **让 LLM 每步直接选动作**

改成：

- **LLM 负责把 OOD 指令“编译”为 follower 熟悉的细粒度 guidance**
- **follower 负责局部动作执行**
- **验证模块负责过滤不可执行计划**
- **分歧时再用 LLM 仲裁，而不是全程接管**

#### 2) 哪个瓶颈被改变了
这其实改的是三个瓶颈：

| 原瓶颈 | FlexVLN 的改法 | 被改变的约束 |
|---|---|---|
| 指令分布偏移 | 把高层/OOD 指令转写成受限动作空间 guidance | follower 不再直接面对 OOD 语言 |
| LLM 局部动作不精确 | LLM 只给段级计划，低层导航交给监督式 follower | LLM 不再承担精细控制 |
| 幻觉/不一致 | 先验证 guidance，再在 follower 分歧时仲裁 | 错误被截断在执行前或冲突点 |

#### 3) 为什么这样会有效
因果链条很清楚：

- **OOD 指令难**，不是因为目标难，而是因为语言接口变了  
  → 用 LLM 做“接口翻译”

- **局部移动难**，不是因为 reasoning 不够，而是因为动作 grounding 要稳定  
  → 用域内训练强的 follower 做执行

- **LLM 容易说得通但走不通**  
  → 用 MLLM 做可行性审查

- **单个 follower 在复杂局部情形下可能犯错**  
  → 用多模型一致性判断 + LLM 仲裁提高鲁棒性

所以它的能力提升不是简单“模型更大”，而是**任务分工正确**。

### 战略权衡表

| 设计选择 | 带来的收益 | 代价/风险 |
|---|---|---|
| LLM 负责高层规划，不直接逐步控制 | 更强跨指令泛化，减少 step-wise LLM 调用 | 需要高质量环境文本化，规划粒度更粗 |
| 受限动作空间 guidance | 提高 follower 可理解性，降低语言失配 | 表达能力受限，复杂动作被压缩 |
| MLLM 可行性验证 | 降低不可执行计划与感知幻觉传导 | 额外推理开销，链路更长 |
| 3 模型 follower + LLM 仲裁 | 局部执行更稳，减少单模型失误 | 系统复杂、依赖多个预训练模型 |
| 文本化轨迹历史反馈 | planner 能利用已探索区域信息避免重复 | 文本抽象会损失几何细节 |

---

## Part III：证据与局限

### 关键证据

#### 1. OOD 泛化确实成立：REVERIE
最能说明问题的是 REVERIE，因为它和 R2R 的指令风格差得很大。

- **FlexVLN (GPT-4o)**：39.80 SR / 29.74 SPL / 26.71 RGS
- 直接从 R2R 迁移的 follower：
  - BEVBert：26.67 SR
  - GridMM：30.02 SR
  - ScaleVLN：28.60 SR
- LLM-based baseline：
  - NavGPT：9.83 SR
  - MapGPT：28.40 或 31.60 SR

这说明它不是单纯比“纯 LLM”强，而是同时比：

- **R2R 直接迁移**
- **LLM-only 规划执行**

都更好。

#### 2. 在更难的高层探索任务上，层次分工更重要：SOON
SOON 指令更长、更复杂、探索性更强。  
这里的 gap 更能体现“LLM 规划 + follower 执行”的必要性。

- **FlexVLN (GPT-4o)**：23.08 SR / 14.96 SPL / 6.94 RGSPL
- R2R 直接迁移 follower：SR 只有 3–5 左右
- NavGPT：2.09 SR
- in-domain GBE：19.52 SR / 1.16 RGSPL

作者特别指出：在 SOON 上，FlexVLN 相对纯 follower baseline 的提升比 REVERIE 更明显。  
这支持一个关键结论：

> 指令越高层、越偏离 R2R 风格，越需要先做“指令编译”，而不是直接执行。

#### 3. 在极少语言信息场景下仍然有效：CVDN-target
CVDN-target 只保留对话里包含目标物的短句，信息很稀疏。

- **FlexVLN (GPT-4o-mini)**：GP 3.63
- HAMT：2.74
- DUET：3.70

它没有在这里全面碾压 in-domain 强基线，但已经说明：
即便 instruction 极短、缺少明确路径描述，LLM 规划仍能补足部分推理能力。

#### 4. 增益来自关键机制，而不只是“用了 GPT”
消融更重要，因为它回答了“到底是哪根旋钮起作用”。

- **去掉 feasibility verification**  
  SR: 37 → 34  
  SPL: 26.38 → 22.83  
  说明 hallucination / 不可执行 guidance 确实会伤害最终表现。

- **去掉动作短语约束**  
  SR: 37 → 30  
  SPL: 26.38 → 20.88  
  说明 follower 需要的是它熟悉的“操作语言”，不是任意自然语言。

- **换掉环境感知方式**  
  4-view InternVL 比 8-view BLIP-2 文本摘要更好，说明**联合场景理解 + 当前位置推断**对 planner 有帮助。

- **历史表示方式**  
  用带 landmark/scene 的自然语言轨迹，比纯角度/距离符号化描述更有用，说明 planner 更依赖语义地标，而非几何符号。

#### 5. 成本也下降，而不是只换来更复杂
对比 NavGPT（同用 GPT-4o-mini）：

- LLM calls：1430 → 692
- cost：\$0.58 → \$0.08
- time：70 min → 34 min

所以它的收益不是“多花钱换性能”，而是靠**稀疏规划**减少了 LLM 调用频率。

### 1-2 个最关键指标
如果只记两个数字：

- **REVERIE**：39.80 SR / 29.74 SPL
- **SOON**：23.08 SR / 6.94 RGSPL

这两个结果共同说明：  
它不仅能跨数据集迁移，而且在高层、探索型、目标导向的指令上更有优势。

### 局限性

- **Fails when**: 场景拓扑本身高度歧义、关键门口/物体在环境感知中被漏掉、或 follower 在完成 guidance 后未及时停止而继续偏航。作者对 REVERIE 的 100 个错误样本分析中，37 个是高难样本，23 个来自 planner 错误，40 个来自 follower 执行错误。
- **Assumes**: 有一个已经在 R2R 类细粒度指令上训练好的 follower；环境是离散图；能访问 InternVL、Qwen2-VL、BLIP-2、Faster R-CNN 以及 GPT-4o/GPT-4o-mini；同时存在 planner 调用预算（最多 10 次）和每条 guidance 的执行预算（最多 5 步）。
- **Not designed for**: 连续控制、真实机器人低延迟闭环控制、端到端统一训练、无外部视觉基础模型/闭源 API 的部署场景，以及完全不需要语言重写的纯局部导航情形。

### 可复用组件
这篇工作最值得迁移走的，不是某个具体 follower，而是几个系统操作符：

- **OOD instruction canonicalization**：把开放式高层指令转成受限动作子语言
- **feasibility check before execution**：先验验证计划再执行
- **disagreement-triggered arbitration**：只有模型分歧时才让更贵的模型介入
- **trajectory verbalization for planner memory**：把低层轨迹转成带地标的自然语言历史

### 一句话结论
FlexVLN 的贡献不是“又做了一个 VLN agent”，而是证明了：

> 对跨任务 VLN 来说，最有效的路线是把 LLM 当作**规划与接口编译器**，而不是当作**逐步动作控制器**。

![[paperPDFs/Vision_Language_Navigation_VLN_Models_2025/arXiv_2025/2025_FlexVLN_Flexible_Adaptation_for_Diverse_Vision_and_Language_Navigation_Tasks.pdf]]