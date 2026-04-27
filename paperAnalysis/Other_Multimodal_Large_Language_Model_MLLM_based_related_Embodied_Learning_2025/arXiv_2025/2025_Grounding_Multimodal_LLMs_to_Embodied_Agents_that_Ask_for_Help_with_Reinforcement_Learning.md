---
title: "Grounding Multimodal LLMs to Embodied Agents that Ask for Help with Reinforcement Learning"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/interactive-instruction-following
  - task/embodied-rearrangement
  - reinforcement-learning
  - llm-generated-reward
  - dataset/ASK-TO-ACT
  - dataset/ReplicaCAD
  - opensource/no
core_operator: 用LLM按步判断“当前提问是否真正帮助消歧”，再把这个语言过程奖励接入在线RL，把MLLM训练成会边探索边提问的VLA策略。
primary_logic: |
  欠指定任务指令 + 多步视觉观测/动作历史 + 用户回复 → Perceiver压缩视觉历史、MLLM自回归输出技能或澄清问题，并由LLM基于特权状态对“有用提问”给出逐步奖励 → 在部分可观测家庭环境中更高成功率地完成重排任务
claims:
  - "AUTOASK 在 ASK-TO-ACT 的部分可观测评测中达到 45.9%/38.7% Success Rate（Unseen Scenes/Unseen Tasks），高于 LLaVA-OneVision SFT 的 33.6%/29.5%，也显著高于 GPT-4o + SoM + ReAct 的 15.7%/14.3% [evidence: comparison]"
  - "若去掉 LLM 生成的“有用问题”逐步奖励，仅用稀疏成功奖励或程序化子目标奖励训练，性能分别降到 0.0% 与 22.4%/16.5% Success Rate，说明密集问题级奖励是任务可学的关键 [evidence: ablation]"
  - "放宽可提问预算会提升成功率但同时提高 Question Ratio，表明该方法通过多问来换取更强的消歧与泛化能力 [evidence: analysis]"
related_work_position:
  extends: "Grounding Multimodal Large Language Models in Actions (Szot et al. 2024a)"
  competes_with: "GPT-4o + SoM + ReAct; LLaVA-OneVision SFT"
  complementary_to: "Set-of-Mark Prompting (Yang et al. 2023)"
evidence_strength: moderate
pdf_ref: paperPDFs/Other_Multimodal_Large_Language_Model_MLLM_based_related_Embodied_Learning_2025/arXiv_2025/2025_Grounding_Multimodal_LLMs_to_Embodied_Agents_that_Ask_for_Help_with_Reinforcement_Learning.pdf
category: Embodied_AI
---

# Grounding Multimodal LLMs to Embodied Agents that Ask for Help with Reinforcement Learning

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2504.00907)
> - **Summary**: 这篇论文把“问得对不对”从难以手写的奖励，改成由 LLM 按步打分的过程奖励，并用在线强化学习把 MLLM 训练成能在部分可观测家庭环境中主动探索、提澄清问题并完成重排任务的 embodied agent。
> - **Key Performance**: ASK-TO-ACT 上 UNSEEN SCENES / UNSEEN TASKS 的 SR = 45.9% / 38.7%，ARS = 40.6 / 35.2；相对 LLaVA-OneVision SFT 的 SR 提升 12.3 / 9.2 个点。

> [!info] **Agent Summary**
> - **task_path**: 欠指定语言指令 + egocentric RGB/历史动作/用户回复 -> 高层技能或澄清问题 -> 单/多物体重排完成
> - **bottleneck**: 部分可观测下“何时问、问什么、问几次”缺少可扩展训练信号，人工奖励与人类演示都很难覆盖
> - **mechanism_delta**: 把有用提问的判断从人工规则改成 LLM 过程奖励，并与成功/子目标奖励一起做在线 RL 微调 MLLM
> - **evidence_signal**: 奖励消融显示去掉 useful-question 奖励后，性能从 45.9/38.7 SR 降到 22.4/16.5 甚至 0
> - **reusable_ops**: [LLM过程奖励标注, 多帧视觉token压缩]
> - **failure_modes**: [需要开放式超模板提问时性能下降, 需要更多轮消歧的未见任务上成功率明显下滑]
> - **open_questions**: [LLM奖励需要多高准确率才足够训练, 更大模型能否支持开放式问题空间]

## Part I：问题与挑战

**What/Why：** 这篇论文真正要解决的，不是“机器人能不能说话”，而是**在部分可观测环境里，机器人如何把探索、提问、执行三件事联合起来**。用户给出的家庭指令常常是欠指定的，比如“把杯子放到茶几上”，但环境里可能有多个杯子、不同颜色/大小/位置，甚至还有隐含的放置偏好。此时如果 agent 不会问问题，就只能瞎猜；如果总在问，又会拖慢任务并恶化交互体验。

### 真正瓶颈
核心瓶颈是一个典型的 **credit assignment** 问题：

1. **提问是否有用** 很难手写成奖励。  
   “你要的是红杯子吗？”和“杯子在不在桌子上？”表面上都是问题，但对当前消歧是否推进，取决于上下文与环境状态。

2. **人类示范难扩展**。  
   要收集“探索动作 + 自然语言问答 + 操作执行”交织的轨迹，成本比普通 embodied imitation learning 更高。

3. **现有零样本 LLM 方案常依赖强假设**。  
   许多方法默认环境可被无误地文本化，或者默认全局可观测；这与真实机器人需要主动探索的设定不符。

### 任务接口与边界
论文提出 **ASK-TO-ACT** 任务来专门研究这个问题：

- **输入**：任务指令、当前与历史视觉观测、过去动作、用户对问题的回复。
- **输出**：125 个高层技能之一，或一个自然语言澄清问题。
- **环境**：Habitat 3.0 中的家庭场景，基于 ReplicaCAD，含 42 类家居物体。
- **任务类型**：属性消歧、空间消歧、尺寸消歧、组合消歧、清理杂物、放置偏好等 7 类。
- **评测轴**：
  - **Unseen Scenes**：场景布局没见过；
  - **Unseen Tasks**：歧义组合更难、通常需要更多问题轮次。

### 为什么是现在
现在值得做这件事，是因为 MLLM 已经具备了三块基础能力：**看图、生成语言、做常识推断**。缺的不是能力原件，而是一个能把这些能力**压进 sequential embodied policy** 的训练接口。作者的判断是：与其去造海量人类演示，不如让 LLM 在训练时扮演“过程裁判”，为 agent 的提问行为提供密集反馈。

---

## Part II：方法与洞察

**How：** 作者引入的关键因果旋钮是：**把“有用提问”从隐含能力目标，变成一个可被 RL 直接优化的逐步奖励信号。**

### 方法骨架

#### 1. 把 MLLM 改成能输出“动作或问题”的 VLA
作者从 **LLaVA-OneVision 0.5B** 出发，改成一个 vision-language-action policy：

- 输入任务指令；
- 接收当前与过去的 egocentric 图像；
- 拼接过去动作文本与用户回复；
- 输出一个高层技能，或者输出一个自然语言问题。

因为 embodied 场景需要长历史，而标准 MLLM 每帧视觉 token 太多，作者用 **Perceiver** 把每帧视觉 token 从 729 压到 4，使多步历史进入上下文成为可能。

#### 2. 用 LLM 生成“有用提问”奖励
训练时，如果 agent 采取 `ask_question(...)`，系统会把以下信息转成文本交给 LLM：

- 当前任务指令；
- 环境特权状态；
- 目标物体元数据；
- 期望放置位置；
- 到目前为止的问答历史；
- 当前问题内容。

然后让 LLM 判断：**这个问题是否真的帮助缩小歧义或识别用户偏好**。  
这个判断被当作逐步奖励的一部分，与成功奖励、子目标奖励、步长惩罚、超预算提问惩罚共同组成 RL 信号。

#### 3. 用 LLM 模拟用户回答
为了在训练与评测中闭环，作者还构建了一个 answering module：给它任务、环境文本状态、目标信息，它返回用户对 agent 问题的回答。  
这让训练过程变成：**探索 → 提问 → 得到回答 → 继续规划/执行**。

#### 4. 在线 RL 而不是纯 SFT
作者采用 DD-PPO 在 8 张 A40 上训练 50M steps。  
直觉上，SFT 更像在拟合“示范轨迹分布”；而 RL 可以直接优化“最终成功 + 问题效率 + 问题有效性”这一复合目标，尤其适合这种需要动态决定是否发问的任务。

### 核心直觉

作者真正改变的不是 backbone，而是**学习目标的形状**：

- **原来**：agent 只知道最后成功没成功，或只知道是否达成若干子目标。  
  这样它很难学会“提一个好问题”这种中间策略。
- **现在**：agent 在每次发问后都能拿到“这句话是否帮助消歧”的反馈。  
  于是“语言交互”不再只是副产物，而变成可优化的决策动作。

更因果地说：

> **从稀疏终局监督 / 粗粒度子目标监督**  
> → **变成对问题质量的过程监督**  
> → **降低了部分可观测下语言动作的 credit assignment 难度**  
> → **提升了 agent 何时问、问什么、问几次的能力**

这也是为什么作者的方法不仅比零样本 GPT-4o 好，也比用合成轨迹做 SFT 的 MLLM 更强：  
**RL 训练到的是“交互策略”，不只是“轨迹模仿”。**

### 战略性取舍

| 设计选择 | 带来的能力 | 代价/风险 |
|---|---|---|
| LLM 作为 useful-question reward model | 不用手写复杂语言奖励，也不用大规模人类示范 | 训练依赖特权状态与 LLM 打分质量 |
| 只允许 9 类问题模板 | 明显缩小语言动作空间，便于 0.5B 模型学习 | 交互不开放，真实场景表达力受限 |
| 使用 oracle 高层技能 | 聚焦“高层决策与提问”而不是低层控制 | 不是端到端机器人控制 |
| Perceiver 压缩视觉 token | 能处理更长观测历史 | 可能损失细粒度视觉信息 |
| 问题预算 B | 可显式控制“少问”与“高成功率”的权衡 | 预算过紧会压制必要提问 |

---

## Part III：证据与局限

**So what：** 能力跃迁最直接的结论是：**这个方法学到的不是更会“说”，而是更会“通过提问来推进任务”**。

### 关键实验信号

#### 1. 比较信号：在部分可观测 setting 下，AUTOASK 明显强于已有可行替代
在 ASK-TO-ACT 上：

- **AUTOASK**：SR 45.9 / 38.7，ARS 40.6 / 35.2
- **LLaVA-OneVision SFT**：SR 33.6 / 29.5
- **GPT-4o + SoM + ReAct**：SR 15.7 / 14.3

这说明两件事：

- 单靠强 MLLM 零样本推理，在视觉部分可观测条件下远远不够；
- 单靠合成 SFT 数据也不够，因为它倾向学成“保守少问”的策略，而不是按需提问。

#### 2. 机制信号：没有 question-aware dense reward，几乎学不会
奖励消融最关键：

- **仅成功奖励**：SR 直接到 0
- **成功 + 子目标奖励**：SR 只有 22.4 / 16.5
- **加入 useful-question reward**：SR 提升到 45.9 / 38.7

这几乎直接回答了论文的核心问题：  
**Ask-for-help 这类 embodied 能力，必须让训练信号覆盖“提问本身是否推进了消歧”。**

#### 3. 行为分析信号：多问能提升成功，但代价是更高问题比率
作者进一步控制可提问预算，发现：

- 允许更多问题时，成功率上升；
- 但 Question Ratio 也会上升。

这表明该方法不是“天然最少问”，而是在学一个**可调的成功率—交互成本折中**。这其实很符合真实产品需求：不同用户对机器人“多问还是少问”的容忍度不同。

### 1-2 个关键指标怎么读
- **SR**：最终能否完成任务。
- **ARS**：不仅看成没成功，还看是否用接近“最少必要问题数”完成、有没有问废话。  
  所以它比 SR 更接近“会不会高质量 ask-for-help”。

### 局限性
- **Fails when**: 需要开放式、超出 9 类模板的问题表达时；或在未见任务中需要比训练时更多轮的复杂消歧时，成功率会明显下降。
- **Assumes**: 训练时可访问模拟器特权状态来生成用户回答与 useful-question 奖励；依赖 LLaVA-OneVision 0.5B、oracle 高层技能、约 50M RL steps、8×A40 GPU 以及并行 vLLM reward server。
- **Not designed for**: 原始低层控制、真实机器人噪声下的开放域对话、完全自由形式的人机问答。

### 可复用组件
这篇论文最值得迁移的不是具体 task，而是下面几个“操作符”：

1. **LLM 过程奖励器**：把难以程序化定义的中间决策质量转成 step-level reward。  
2. **问题预算控制**：把交互成本显式纳入策略学习。  
3. **历史多模态交错 + 视觉 token 压缩**：让 MLLM 能在 embodied setting 下看更长时间范围的上下文。  

如果以后要扩展到真实机器人，最自然的方向是：  
把这里的 reward-modeling 思路迁到更开放的对话空间，同时替换掉 oracle 高层技能和模拟器特权状态。

## Local PDF reference

![[paperPDFs/Other_Multimodal_Large_Language_Model_MLLM_based_related_Embodied_Learning_2025/arXiv_2025/2025_Grounding_Multimodal_LLMs_to_Embodied_Agents_that_Ask_for_Help_with_Reinforcement_Learning.pdf]]