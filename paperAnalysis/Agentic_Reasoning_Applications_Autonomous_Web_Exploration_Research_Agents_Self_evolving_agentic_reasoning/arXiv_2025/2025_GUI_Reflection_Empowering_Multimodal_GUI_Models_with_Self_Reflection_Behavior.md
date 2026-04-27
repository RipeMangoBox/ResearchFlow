---
title: "GUI-Reflection: Empowering Multimodal GUI Models with Self-Reflection Behavior"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/gui-automation
  - self-reflection
  - synthetic-data
  - curriculum-learning
  - dataset/AndroidControl
  - dataset/GUI-Odyssey
  - dataset/ScreenSpot
  - dataset/AndroidWorld
  - opensource/promised
core_operator: 将GUI反思拆成“动作验证—动作撤销—知错重试”三类原子能力，并通过自动合成错误轨迹与在线失败挖掘，在预训练、离线SFT和在线调优三阶段持续注入。
primary_logic: |
  任务指令 + 当前/历史GUI截图 + 动作历史/记忆
  → 预训练学习动作验证、撤销与知错重试
  → 离线从成功轨迹自动构造错误-纠正样本，并在线从失败交互中标注纠错与反思监督
  → 输出能识别错误、回退并重试的GUI原子动作
claims:
  - "在GUI预训练中加入GUI-Reflection Task Suite后，8B模型的Action Verification准确率从57.95%提升到87.56%，Action Reversal从40.71%提升到93.81% [evidence: ablation]"
  - "在作者在线环境的level-2任务上，反思式离线SFT将成功率从14.58%提升到23.61%，再加在线reflection tuning进一步提升到34.72% [evidence: ablation]"
  - "最终8B端到端模型在AndroidWorld上达到34.5%成功率，超过UI-TARS-7B的33.0%和OS-Genesis-8B的16.9% [evidence: comparison]"
related_work_position:
  extends: "UI-TARS (Qin et al. 2025)"
  competes_with: "UI-TARS (Qin et al. 2025); OS-Genesis (Sun et al. 2024)"
  complementary_to: "Digi-Q (Bai et al. 2025)"
evidence_strength: strong
pdf_ref: paperPDFs/Agentic_Reasoning_Applications_Autonomous_Web_Exploration_Research_Agents_Self_evolving_agentic_reasoning/arXiv_2025/2025_GUI_Reflection_Empowering_Multimodal_GUI_Models_with_Self_Reflection_Behavior.pdf
category: Embodied_AI
---

# GUI-Reflection: Empowering Multimodal GUI Models with Self-Reflection Behavior

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2506.08012), [Project](https://penghao-wu.github.io/GUI_Reflection/)
> - **Summary**: 这篇工作把GUI智能体最缺的“发现自己做错了—撤销—再试一次”能力变成显式可训练对象，并在预训练、离线SFT、在线交互三个阶段用自动生成数据持续注入，从而提升端到端GUI模型的鲁棒性。
> - **Key Performance**: level-2在线任务成功率从 14.58% → 23.61% → 34.72%；AndroidWorld 上 34.5% SR，高于 UI-TARS-7B 的 33.0%。

> [!info] **Agent Summary**
> - **task_path**: 任务指令 + 当前/历史GUI截图 + 动作历史/记忆 -> 带反思thought的grounded GUI原子动作
> - **bottleneck**: GUI模型主要从近乎零错误的离线成功轨迹学习，部署时却必须在错误状态中继续决策，导致不会判断前一步是否走偏
> - **mechanism_delta**: 把“反思”从隐式能力改成显式监督对象，并把失败轨迹从被丢弃样本改造成可自动标注的纠错训练信号
> - **evidence_signal**: 分阶段消融显示，反思SFT与在线reflection tuning累计将复杂level-2任务成功率提升到34.72%
> - **reusable_ops**: [reflection-task-suite, failure-mined-correction-data]
> - **failure_modes**: [high-level-planning-errors, non-mobile-gui-transfer]
> - **open_questions**: [can-it-handle-long-horizon-planning-failures, can-the-pipeline-remove-closed-api-dependence]

## Part I：问题与挑战

这篇论文抓住的**真正瓶颈**不是“模型能不能看懂按钮”，而是：

**当模型点错、滚错、进错页面后，它能不能意识到自己错了，并把任务拉回正轨。**

现有端到端GUI模型的主流训练范式是：
- 先做 GUI pre-training，学视觉 grounding、OCR、UI understanding；
- 再用离线成功轨迹做 SFT，学习“当前界面 → 下一步动作”。

问题在于，这种训练分布几乎全是**成功演示**。  
而真实部署时，GUI agent 经常会遇到：
- 不熟悉的 App 界面；
- 错误点击后的偏航状态；
- 无效操作或执行失败；
- 长链任务中间步骤的局部失误。

于是训练和部署之间出现一个核心错配：

**训练时学的是“如何模仿正确下一步”，部署时需要的是“如何在做错之后继续活下来”。**

论文还提出一个很重要的观察：  
基础 MLLM 往往本来带有一些验证、反思、纠错的潜在能力，但常规 GUI-specific pre-training + 成功轨迹 SFT 反而会把这种能力“磨掉”。这也是为什么现在很多 GUI 模型一旦走错，就会一路错下去。

### 输入/输出接口

该模型是一个端到端多模态 GUI agent：
- **输入**：任务指令、当前截图、过去若干步截图、完整动作历史、memory bank
- **输出**：action thought、action description、grounded atomic action

动作空间不只包含 Click / Scroll / Type / Back，还加入了：
- **Answer**：直接回答任务所需信息
- **Memorize**：把中间信息存入记忆

这说明它不是单步点击模型，而是一个要处理**长程任务执行**的交互控制器。

### 边界条件

本文主要聚焦：
- **Android mobile GUI**
- **纯视觉输入**（不依赖 accessibility tree）
- 长程、多步、可能需要回退与重试的任务

所以它关注的不是单次 grounding 精度，而是**错误状态下的持续决策能力**。

## Part II：方法与洞察

整体方法可以概括成一句话：

**先把反思拆成更小、可监督的原子能力；再把原本几乎看不到的“错误状态”系统性地造出来；最后在在线环境里把失败轨迹重新变成训练信号。**

### 1）预训练阶段：先保住“会反思的种子”

作者提出 **GUI-Reflection Task Suite**，包含三类任务：

1. **Action Verification**  
   给前后两张连续截图和一个动作目的，判断这个隐式动作是否真的达成了该目的。  
   本质上学的是：**看执行结果，判断前一步是否成功**。

2. **Action Reversal**  
   给动作前后截图和已执行动作，让模型选出最合理的撤销动作。  
   本质上学的是：**识错之后怎么回退**。

3. **Mistake-informed Reattempt**  
   告诉模型前一次 grounding 错在哪里，再要求重新预测。  
   本质上学的是：**把负反馈变成下一次尝试的约束**。

这一步的作用不是直接训练完整 GUI 反思流程，而是先把反思行为分解成几个原子技能，避免常规 GUI 预训练把基础 MLLM 原有的反思能力冲掉。

### 2）离线 SFT：从“成功轨迹”自动合成“犯错后怎么改”

这是本文最关键的机制之一。

难点在于：公开 GUI 数据大多是成功轨迹，没有“错误动作 + 错误后果”标注。  
作者用两种自动化方法，把成功轨迹改造成反思训练样本：

- **方法A：改任务目标，让原本正确动作变成自然错误**  
  这样模型在下一步必须：
  - 识别上一动作不对；
  - 采取纠正动作；
  - 必要时回退；
  - 再做一次更合理的尝试。

- **方法B：在正确动作前插入一个“无效错误动作”**  
  比如已经到底了还 scroll，或点击不响应的元素。  
  这样屏幕不变，但模型要学会：
  - 观察到“动作没有产生预期变化”；
  - 承认刚才操作无效；
  - 立即转向正确动作。

这一步改变了离线 SFT 的训练分布：  
从只看“专家永远正确”的轨迹，变成能看到**偏航、停滞、回退、再尝试**。

### 3）在线阶段：不再丢弃失败，而是从失败里挖监督

作者构建了一个 Android 在线环境：
- 11 个 App
- 215 个任务模板
- 支持程序化 verifier 和 MLLM-based verifier
- 分布式 host-worker 设计，便于采样和训练

其 **iterative online reflection tuning** 的关键点是：

- 对**成功轨迹**：不是整条全收，而是做 step-wise correctness filtering，只保留真正正确的步骤；
- 对**失败轨迹**：找到**第一个错误步**，保留此前正确前缀，并自动标注：
  - **pre-error correction**：这一步本来该做什么；
  - **post-error reflection**：做错后下一步该如何承认错误、回退、纠正。

这很重要，因为它把在线训练从“成功才有用”改成了：

**失败轨迹也有用，而且是学反思最有价值的样本。**

严格说，这里的在线算法更像：
- 失败轨迹挖掘
- 自动纠错标注
- 迭代式 filtered behavior cloning

而不是传统意义上直接用 reward 做 policy gradient 的 RL。

### 4）关键工程技巧：无人工标注的 grounded action 生成

端到端 GUI 模型不仅要输出 thought，还要输出**坐标级 grounded atomic action**。  
作者的自动标注做法是：

- 通用 MLLM 生成 action thought + action description
- GUI 模型根据这些文本再生成 grounded atomic action
- 再让 MLLM 过滤 thought / description / action 的一致性

这是一个很实用的分解技巧：  
**把“高层语义生成”和“低层坐标 grounding”拆开做，再做一致性过滤。**

### 核心直觉

这篇论文真正改动的不是模型骨干，而是**训练信号的分布**。

- **以前**：只监督“当前状态下正确下一步是什么”  
- **现在**：额外监督“上一动作是否成功、如何撤销、如何基于错误再试”

于是发生了三个因果变化：

1. **把执行结果纳入监督**  
   从只看当前截图，变成要看“动作前后状态变化是否符合目的”。  
   → 改变了信息瓶颈  
   → 模型开始具备“验证前一步”的能力

2. **把失败状态纳入训练分布**  
   从几乎只见成功状态，变成系统性看见偏航状态与恢复路径。  
   → 改变了分布错配  
   → 模型开始具备“从错误状态恢复”的能力

3. **把失败轨迹从垃圾变成信号**  
   在线训练中，不再只收成功轨迹，而是从失败里抽第一错误步做纠错监督。  
   → 改变了探索利用效率  
   → 模型开始持续增强反思与纠错行为

一句话总结因果链：

**只模仿正确动作 → 不会识错；  
显式监督“验证/撤销/重试” + 从失败中提标签 → 才会恢复。**

### 战略取舍表

| 设计选择 | 解决的瓶颈 | 带来的能力 | 代价 / 风险 |
|---|---|---|---|
| GUI-Reflection Task Suite | 常规GUI预训练会削弱基础MLLM的反思能力 | 保住并强化动作验证、撤销、知错重试原子技能 | 需要额外构造任务与评测集 |
| 离线自动合成错误样本 | 成功轨迹缺少错误状态 supervision | 让模型在SFT阶段就见过偏航、回退、再尝试 | 合成错误主要覆盖局部视觉/动作错误，复杂规划错误覆盖不足 |
| 在线 reflection tuning | 失败轨迹被直接丢弃，学习信号稀疏 | 从失败中持续学习纠错与恢复 | 需要在线环境、verifier 和较高采样成本 |
| 通用MLLM + GUI模型分工标注 | 无人工标注时难以同时得到 thought 与 grounded action | 自动构建较一致的训练数据 | 依赖 Gemini / GPT-4o 等闭源API，复现成本高 |

## Part III：证据与局限

### 关键证据

**信号1｜常规GUI预训练确实会“伤害反思”，而任务套件能把它救回来。**  
在 Action Verification / Action Reversal 上，普通 GUI-Pretrain 相比基础通用 MLLM 明显退化；加入 GUI-Reflection Task Suite 后，8B 模型从 57.95% / 40.71% 提升到 87.56% / 93.81%。  
这直接支持作者的核心论点：**问题不只是没学到反思，而是现有训练范式会主动削弱它。**

**信号2｜反思能力不是单点收益，而是三阶段叠加收益。**  
在作者在线环境的 level-2 任务上：
- 无 reflection SFT + filtered BC：14.58%
- 加 reflection SFT：23.61%
- 再加 online reflection tuning：34.72%

这说明增益不是来自某个偶然组件，而是来自**预训练 → 离线SFT → 在线调优**的一条完整因果链。

**信号3｜模型是真的在利用“已知错误”，不是单纯多采样碰运气。**  
在 Mistake-informed Reattempt 上，GUI-Pretrain-Ref 的第 3 次尝试表现高于 pass@3，说明它不是随机重试，而是在利用“前几次错在哪里”这一负信息做约束更新。  
这是本文相对扎实的一点：它测的不是“多试几次会不会中”，而是“能不能从错里学”。

**信号4｜外部基准上具备可迁移性，但应保守解读。**  
最终 8B 模型在 AndroidWorld 上达到 34.5% SR，超过 UI-TARS-7B 的 33.0% 和 OS-Genesis-8B 的 16.9%，说明该框架不只在自建环境里有效。  
但它并未超过大模型版本如 UI-TARS-72B，因此更准确的结论是：**在 8B 级端到端 GUI 模型里，反思训练带来了竞争力提升。**

### 局限性

- **Fails when**: 错误来自更高层的规划、任务分解或长期子目标管理，而不是局部视觉理解、点击偏差、元素功能误判时；此外跨到桌面/Web GUI 时，当前训练分布未必足够。
- **Assumes**: 有较多成功轨迹可用于反向构造错误样本；依赖 Gemini-2.0/2.5 与 GPT-4o 做数据生成、过滤和部分 verifier；在线训练需要分布式 Android 环境；训练使用 32 张 H100；某些数据构造还默认 `Press Back` 能把状态恢复到之前页面。
- **Not designed for**: 零工程迁移到 desktop / web；纯高层规划型失败诊断；以及完全不依赖闭源API的低成本复现。

### 复现与比较时需要额外注意

- AndroidWorld 评测里，作者**修改了 scroll / type 的实现**，并给部分任务增加了 step budget；因此和原始默认配置下的结果相比，最好保守解释，不要把它当成完全等价的横向数字。
- 虽然论文强调“无需人工标注”，但这不等于“低成本可复现”：  
  它把人类标注成本换成了**闭源 MLLM API 成本 + 在线环境工程成本**。

### 可复用组件

这篇论文最值得复用的，不只是最终模型，而是这几个操作件：

1. **Reflection Task Suite**：把反思拆成可监督原子任务  
2. **成功轨迹 → 错误轨迹** 的自动合成管线  
3. **失败轨迹第一错误步挖掘** 与 pre/post correction 标注  
4. **thought 生成与 grounded action 生成解耦** 的自动标注策略

如果后续有人做 Web agent、Desktop agent、甚至机器人 UI/设备控制，这几块都可以迁移。

## Local PDF reference

![[paperPDFs/Agentic_Reasoning_Applications_Autonomous_Web_Exploration_Research_Agents_Self_evolving_agentic_reasoning/arXiv_2025/2025_GUI_Reflection_Empowering_Multimodal_GUI_Models_with_Self_Reflection_Behavior.pdf]]