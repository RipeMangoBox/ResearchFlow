---
title: "MobileUse: A GUI Agent with Hierarchical Reflection for Autonomous Mobile Operation"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/mobile-operation
  - task/gui-automation
  - hierarchical-reflection
  - proactive-exploration
  - confidence-gating
  - dataset/AndroidWorld
  - dataset/AndroidLab
  - opensource/full
core_operator: 通过按需触发的动作级、轨迹级、全局级三级反思，并在执行前主动探索沉淀 App 通识，提升移动 GUI 代理的纠错与冷启动适应能力。
primary_logic: |
  陌生移动 App 环境 + 用户指令 → 先自主探索并总结可复用环境知识，再结合当前截图与历史轨迹由 Operator 生成动作 → 在低置信、重复停滞或终止时按需触发动作/轨迹/全局反思并由 Progressor 更新进度 → 输出更稳健的 GUI 动作序列与完成状态
claims:
  - "MobileUse 在 AndroidWorld 上达到 62.9% success rate，超过 V-Droid 的 59.5% 与 Agent-S2 的 54.3% [evidence: comparison]"
  - "MobileUse 在 AndroidLab 上达到 44.2% task success rate 与 50.01% sub-goal success rate，均为文中最佳结果 [evidence: comparison]"
  - "相对仅含 Operator+Progressor 的 49.5% 基线，加入分层反思与 Reflection-on-Demand 后 AndroidWorld 成功率提升到 61.6%，再加主动探索提升到 62.9% [evidence: ablation]"
related_work_position:
  extends: "Mobile-Agent-V2 (Wang et al. 2024)"
  competes_with: "V-Droid; Agent-S2"
  complementary_to: "UGround; UI-TARS"
evidence_strength: strong
pdf_ref: paperPDFs/Agentic_Reasoning_Applications_Autonomous_Web_Exploration_Research_Agents_Self_evolving_agentic_reasoning/arXiv_2025/2025_MobileUse_A_GUI_Agent_with_Hierarchical_Reflection_for_Autonomous_Mobile_Operation.pdf
category: Embodied_AI
---

# MobileUse: A GUI Agent with Hierarchical Reflection for Autonomous Mobile Operation

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2507.16853), [Code](https://github.com/MadeAgents/mobile-use)
> - **Summary**: 论文把移动 GUI 代理从“一次性前馈执行”改成“主动探索 + 按需分层反思”的闭环系统，重点解决长程任务中的误差累积与陌生 App 冷启动问题。
> - **Key Performance**: AndroidWorld 成功率 62.9%；AndroidLab 成功率 44.2%，Sub-Goal 成功率 50.01%

> [!info] **Agent Summary**
> - **task_path**: 用户自然语言指令 + 手机截图/历史轨迹/环境知识 -> GUI 动作序列 + 任务完成状态
> - **bottleneck**: 长程动态任务中的误差累积，以及陌生 App 下缺乏可迁移环境先验
> - **mechanism_delta**: 把单次前馈执行改为“任务前主动探索 + 执行时按需三级反思”的多代理闭环控制
> - **evidence_signal**: 双基准 SOTA，且 AndroidWorld 上基线 49.5% 经分层反思提升到 61.6%
> - **reusable_ops**: [屏幕前后差分高亮, 触发式三级反思]
> - **failure_modes**: [弱底座模型下反思与探索收益显著下降, 反思器误判会带来额外步骤与错误纠偏]
> - **open_questions**: [如何将主动探索变成更高效的奖励驱动探索, 如何把该框架稳定压缩到端侧小模型]

## Part I：问题与挑战

这篇论文要解决的核心问题，不是“模型能不能看懂一个手机截图”，而是：

1. **长程移动任务里，小错误会滚雪球。**  
   打开错 App、点错坐标、忘了先激活输入框、提前 terminate，这些单步错误如果不能及时发现，会在后续步骤中累积成整条轨迹偏航。

2. **单步正确，不代表整任务在正确轨道上。**  
   代理可能每一步都“局部合理”，但整体上已经偏离用户目标；或者陷入重复点击、重复页面、局部修错失败后的循环。

3. **冷启动是真实移动环境里的硬约束。**  
   遇到新 App、新图标语义、新颜色规则时，代理缺少“操作常识”。例如某个红色 checkbox 表示高优先级，这类知识不在通用视觉理解里，却决定任务成败。

### 输入 / 输出接口

- **输入**：用户指令、当前截图、历史动作、历史反馈、进度摘要，以及任务前探索得到的环境知识。
- **输出**：原子 GUI 动作（点击、滑动、输入、打开 App、返回、等待、回答、终止等）及最终完成状态。

### 为什么是现在？

因为强 MLLM 已经足够支持“看图 + 跟指令 + 产出动作”，真正卡住移动代理上限的瓶颈，开始从**感知能力**转移到**执行闭环能力**：  
- 能不能在错误发生后知道自己错了？
- 能不能在新环境里先建立一点操作先验？
- 能不能在不把推理成本炸掉的前提下做这些事？

MobileUse 的定位，就是在**不重新训练新 backbone**的前提下，直接通过测试时系统设计补上这几个缺口。

## Part II：方法与洞察

### 方法骨架

MobileUse 是一个两阶段、多代理框架：

- **阶段一：Proactive Exploration**
  - 在没有具体任务约束时，先让代理主动探索 App。
  - 将探索轨迹总结为可复用的通用知识，再在正式任务时检索相关部分。
  - 目标不是记住某条历史任务，而是积累**App 级常识**。

- **阶段二：Autonomous Mobile Operation**
  - **Operator**：基于指令、截图、历史、反馈和知识生成当前动作。
  - **Progressor**：压缩并更新任务进度，避免长轨迹上下文失控。
  - **Hierarchical Reflectors**：在不同时间尺度检查“这一步对不对”“最近几步有没有偏航”“任务真的完成了吗”。

### 核心直觉

这篇论文最关键的因果拨杆是：

- **把错误检测从单尺度、事后暴露，改成多尺度、执行中暴露**
  - 动作级反思看“这一步是否达到预期”
  - 轨迹级反思看“最近几步是否在推进任务”
  - 全局级反思看“你以为完成了，但其实没完成”

- **把反思从 always-on 改成 on-demand**
  - 大多数步骤其实是对的；
  - 如果每步都反思，不仅慢，还会引入“反思器自身误判”的新噪声；
  - 所以只在低置信、重复动作、重复页面、累计错误或 terminate 时触发。

- **把冷启动从“执行中临场猜”改成“执行前先建环境先验”**
  - 主动探索先把新 App 的图标语义、优先级颜色、常见操作路径等总结出来；
  - 正式任务时再检索相关知识，降低陌生界面带来的信息赤字。

更具体地说，它改变了三种约束：

- **观测约束**：从只看当前截图，变为看前后截图差分与变化区域高亮，使动作后果更可见。
- **时间约束**：从单步决策，变为同时监控短期轨迹和全局完成性，使偏航与早停更可见。
- **知识约束**：从零先验冷启动，变为先探索再执行，使陌生 App 的隐含规则更可见。

能力变化也就很直接：

- 更强的**局部纠错**
- 更强的**长程稳定性**
- 更强的**冷启动适应性**

### 关键模块怎么起作用

#### 1. Action Reflector：抓“单步没生效”
它比较动作执行前后的截图，并借助视觉差分高亮变化区域。  
这让反馈更 anchored 在**可观测状态变化**上，而不是只依赖语言模型自说自话地复盘。

适合处理：
- 点错位置
- 没点到输入框就开始输入
- UI 理解错导致操作无效

#### 2. Trajectory Reflector：抓“看似都对，但实际在绕圈”
它查看最近几步动作、动作级反馈和进度摘要。  
触发条件不是模型置信度，而是**重复动作、重复截图、累计动作错误**这类停滞信号。

适合处理：
- 键盘挡住输入框导致来回点击
- 局部修错失败后进入循环
- 每步都“合理”，但整体目标没推进

#### 3. Global Reflector：抓“过早宣布成功”
只在代理输出 terminate 时调用。  
它验证：你是真的完成了，还是只是“感觉差不多”。

适合处理：
- 漏掉子目标
- 复杂任务中过早结束
- 幻觉式完成判断

#### 4. Reflection-on-Demand：防止“反思过度”
论文一个很重要的洞察是：**反思不是越多越好**。  
动作级反思只有在动作置信度低时才触发；轨迹级和全局级也有明确触发条件。  
这实质上把“反思”从默认操作变成稀缺资源。

#### 5. Proactive Exploration：把陌生环境转成知识库
与基于既往任务轨迹的 self-evolution 不同，MobileUse 更强调**任务前、任务无关的主动探索**。  
这让它得到的是更泛化的“环境通识”，而不是一条条难以迁移的历史经验。

### 战略性 trade-off

| 设计选择 | 改变的瓶颈 | 能力收益 | 代价 / 风险 |
|---|---|---|---|
| 动作级反思 + 屏幕差分 | 单步错误难被观察 | 更快发现 grounding / interaction 失败 | 若频繁调用会增时延，且反思器会误判 |
| 轨迹级反思 | 单步正确但整体偏航 | 抓循环、停滞、累计漂移 | 依赖触发规则，可能漏掉隐性偏航 |
| 全局级反思 | terminate 不等于完成 | 减少早停与漏做子任务 | 会增加收尾验证步骤 |
| 按需触发 | 反思本身会制造噪声 | 兼顾性能与效率 | 阈值与触发条件需要调节 |
| 主动探索 | 冷启动时环境语义缺失 | 对陌生 App 更快建立操作先验 | 额外探索成本，且探索质量依赖底座模型 |

## Part III：证据与局限

### 关键实验信号

- **信号 1：双 benchmark 对比，说明能力跳跃不是单点偶然**
  - AndroidWorld：**62.9%**
  - AndroidLab：**44.2%**
  - 在 AndroidWorld 上超过 V-Droid 的 59.5%，也明显高于 Agent-S2 的 54.3%。  
  这说明收益不仅来自更强底座，而是**系统级执行闭环**本身有效。

- **信号 2：消融清楚表明，真正的大头来自分层反思**
  - Base（Operator + Progressor）：49.5%
  - 加完整 hierarchical reflection + Reflection-on-Demand：61.6%
  - 再加 proactive exploration：62.9%  
  也就是说，**主增益来自纠错闭环**，主动探索是额外加成，而不是主因。

- **信号 3：按需反思确实不是“省算力但掉点”的妥协**
  - 阈值分析表明，在较合理阈值下，反思次数下降但成功率反而上升；
  - 当阈值设为 -0.01 时，省掉了 **85%+** 的反思，成功率下降仍 **<1.5%**。  
  说明关键不在“多反思”，而在“只保留关键反思”。

- **信号 4：错误类型分析与纠错矩阵支持机制解释**
  - 分层反思纠正了 18 个原本失败的任务，校正率 30.51%，误判率 7.02%；
  - 在 perception、navigation 等失败类型上下降明显。  
  这和它的设计目标是对齐的：不是只修某一种错，而是修多尺度执行偏差。

### 局限性

- **Fails when**: 底座模型本身指令跟随、视觉 grounding、复杂推理能力不足时，分层反思和主动探索都很难真正起效；文中 7B 版本在 AndroidWorld 只有 21.6%，说明这套框架并不能替代基础能力。  
- **Assumes**: 依赖强 MLLM 作为 Operator/Reflector（72B 效果最佳），需要 ADB 接入真实设备或基准环境；若自部署，作者使用 vLLM + 4×A100 80GB 运行模型，或替换为第三方 API。  
- **Not designed for**: 端侧低时延部署、最少步数优化、以及有显式奖励函数的高效探索场景；从 AndroidLab 的 RRR/ROR 看，它也不是以“最少冗余操作”作为首要目标。

### 额外的边界判断

- **主动探索的收益并不均匀。**  
  从 AndroidWorld 消融看，它总体只带来 +1.3% 提升，且主要改善容易任务/冷启动情形，而不是对所有中高难任务都稳定增益。

- **反思会带来额外步骤。**  
  作者也承认 MobileUse 在 RRR/ROR 上不是最优，这意味着它更像是“稳健优先”的系统，而不是“最短路径优先”的系统。

### 可复用组件

- **屏幕前后差分高亮**：把“动作是否生效”显式化，适合各种 GUI agent。
- **触发式多粒度反思**：可迁移到网页代理、桌面代理、RPA 等长程交互任务。
- **Progressor 进度摘要器**：用紧凑状态摘要替代无限增长的原始轨迹上下文。
- **任务前探索 → 知识总结 → 任务时检索**：适合处理新环境冷启动，而不必依赖历史任务 replay。

## Local PDF reference

![[paperPDFs/Agentic_Reasoning_Applications_Autonomous_Web_Exploration_Research_Agents_Self_evolving_agentic_reasoning/arXiv_2025/2025_MobileUse_A_GUI_Agent_with_Hierarchical_Reflection_for_Autonomous_Mobile_Operation.pdf]]