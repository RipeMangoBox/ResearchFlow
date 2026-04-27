---
title: "Mobile-Agent-V: A Video-Guided Approach for Effortless and Efficient Operational Knowledge Injection in Mobile Automation"
venue: arXiv
year: 2025
tags:
  - Others
  - task/mobile-automation
  - task/video-understanding
  - multi-agent
  - sliding-window
  - reflection
  - dataset/Mobile-Knowledge
  - dataset/AndroidWorld-Knowledge
  - opensource/no
core_operator: "把一次手机操作演示视频直接转成可执行操作知识，并通过滑动窗口、视频定位与反思校正确保逐步对齐执行。"
primary_logic: |
  操作演示视频 + 当前设备截图 + 用户任务 →
  提取关键帧并用滑动窗口定位当前相关片段 →
  决策代理生成动作、反思代理校正动作、视频代理推进时间对齐 →
  在移动端逐步完成目标操作
claims:
  - "在相同 GPT-4o 基座下，Mobile-Agent-V 在 Mobile-Knowledge 上达到 86.7% SR，比最佳手写知识基线 Agent-S2 高 23.4 个百分点，且步骤数从 13.6 降到 7.3 [evidence: comparison]"
  - "在每个场景只提供一个最简单任务视频的 AndroidWorld-Knowledge 上，Mobile-Agent-V 获得 31.3% SR，超过最佳基线 18.9% [evidence: comparison]"
  - "加入 deep-reflection agent 后，复杂任务上的决策准确率提升更明显，说明其能纠正多帧时序理解带来的动作偏差 [evidence: ablation]"
related_work_position:
  extends: "Mobile-Agent-v2 (Wang et al. 2024)"
  competes_with: "Agent-S2 (Agashe et al. 2025); Mobile-Agent-v2 (Wang et al. 2024)"
  complementary_to: "Ferret-UI (You et al. 2024); OmniParser (Wan et al. 2024)"
evidence_strength: moderate
pdf_ref: "paperPDFs/Agentic_Reasoning_Applications_Autonomous_Web_Exploration_Research_Agents_Collective_multi_agent_reasoning/arXiv_2025/2025_Mobile_Agent_V_A_Video_Guided_Approach_for_Effortless_and_Efficient_Operational_Knowledge_Injection_in_Mobile_Automation.pdf"
category: Others
---

# Mobile-Agent-V: A Video-Guided Approach for Effortless and Efficient Operational Knowledge Injection in Mobile Automation

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2502.17110)
> - **Summary**: 这篇工作把“专家手写操作知识”替换为“一次手机录屏演示”，再用滑动窗口、视频定位和反思纠错，把演示视频变成可复用的移动端操作先验。
> - **Key Performance**: Mobile-Knowledge 上 SR 86.7%（最佳基线 63.3%）；AndroidWorld-Knowledge 上 SR 31.3%（最佳基线 18.9%）

> [!info] **Agent Summary**
> - **task_path**: 操作演示视频 + 当前手机截图 + 用户指令 -> 下一步 GUI 动作
> - **bottleneck**: 长尾 App 操作知识难以低成本注入，同时长视频里的当前步骤定位和多帧时序对齐很不稳定
> - **mechanism_delta**: 用演示视频替代手写知识，并把全视频理解拆成“局部窗口决策 + 反思纠错 + 时序重定位”的循环
> - **evidence_signal**: 在共享 GPT-4o 基座下，Mobile-Knowledge 上 SR 从最佳基线 63.3% 提升到 86.7%，且步骤更少
> - **reusable_ops**: [keyframe-redundancy-filtering, local-video-window-grounding]
> - **failure_modes**: [poor-video-quality-causes-keyframe-miss, large-gap-between-demo-and-target-task-reduces-generalization]
> - **open_questions**: [how-to-generalize-without-related-demo-video, how-to-reduce-dependence-on-closed-gpt4o-and-token-cost]

## Part I：问题与挑战

**真正的瓶颈不是“会不会点按钮”，而是“知不知道这个 App 的正确操作路径”。**  
现有 mobile agent 往往具备一定视觉感知、定位和基础操作能力，但一旦遇到训练数据没覆盖的界面流程、品牌特有设置路径、或跨 App 的细粒度操作规则，就会失败。论文把这个问题定义为：

- **外部操作知识注入成本高**：现有做法常靠人手写文本知识，但质量强依赖作者经验。
- **知识载体不自然**：用户本来就更容易“做一遍并录屏”，而不是“抽象成高质量文本说明”。
- **视频难直接用**：手机录屏里大量静止帧 + 少量快速状态跳变，MLLM 直接吃长视频会有冗余和时序混乱。
- **评测混杂**：传统 benchmark 常把规划、定位、操作、知识利用混在一起，导致很难单独判断“知识注入”是否有效。

### 输入 / 输出接口

这篇方法的接口很清晰：

- **输入**
  - 一段完成过某任务的手机操作视频
  - 当前设备截图
  - 视频中完成的任务描述 `Iv`
  - 当前用户要完成的任务描述 `Iu`
- **输出**
  - 下一步 GUI 动作（点击、滚动、输入、返回、主页、完成；实现中还补了按文本点击）

### 为什么现在值得做

- 手机录屏获取非常便宜，用户行为示范是天然存在的数据源。
- MLLM 已经能看懂截图、界面元素和短序列图像，但还缺把**演示轨迹转成可执行先验**的机制。
- 移动自动化正在从“会操作”走向“会迁移知识”，而这正是长尾任务落地的关键。

### 边界条件

这篇方法不是无条件通用，它默认：

- 演示视频和目标任务**至少有相近的操作路径**；
- 基础感知/点击能力已经存在；
- 评测任务被刻意设计为**弱化规划与基础操作难度，突出知识利用**。

---

## Part II：方法与洞察

### 方法骨架

Mobile-Agent-V 的核心不是单个模型，而是一个**围绕视频对齐展开的多代理闭环**：

1. **视频处理**
   - 先均匀采样；
   - 再按相邻帧相似度去冗余；
   - 再按时间间隔过滤过密关键帧。  
   目的：把“长录屏”压缩成“有变化的关键步骤”。

2. **滑动窗口**
   - 不把整段视频一次性喂给模型；
   - 只给当前最相关的一小段关键帧窗口。  
   目的：降低 token 冗余，避免长视频把当前决策淹没。

3. **Decision Agent**
   - 根据当前窗口、当前截图、源任务 `Iv`、目标任务 `Iu` 和历史动作，预测下一步操作。  
   这一步不是简单模仿，而是“**参考演示路径，执行目标任务**”。

4. **Deep-Reflection Agent**
   - 对决策代理的动作做二次核对；
   - 检查这个动作是否真的和视频里的当前步骤一致；
   - 若不一致则改写动作。  
   目的：缓解多帧推理时常见的“看到了对的页面，却做错了下一步”。

5. **Video Agent**
   - 根据操作前后截图和当前窗口，判断设备现在对齐到视频中的哪一帧；
   - 再决定窗口起点如何往前推进。  
   目的：把“我现在处于演示轨迹的哪里”显式化，而不是让模型隐式记忆。

### 核心直觉

**这篇论文真正改变的，不是模型参数，而是知识进入 agent 的方式。**

#### 1）从“静态文本知识”改成“动态轨迹知识”
手写知识是摘要后的、离散的、质量高度依赖作者的；  
演示视频则天然保留了：

- 操作顺序
- 页面切换关系
- 中间状态
- 何时停止

这相当于把知识从“规则描述”换成“轨迹示范”。

#### 2）从“全局长视频理解”改成“局部对齐问题”
原始难题是：模型要在很长的视频里，找出当前设备状态对应的步骤，再决定下一步。  
作者把它拆成三个更容易的问题：

- **局部看**：滑动窗口只保留当前附近关键帧；
- **先做再查**：决策代理先提案；
- **显式纠错与重定位**：反思代理验证，视频代理定位。

这样就把一个高噪声的长时序理解问题，变成了可循环求解的**局部时序同步问题**。

#### 3）从“纯模仿”改成“模仿路径 + 迁移目标槽位”
论文同时给模型看 `Iv` 和 `Iu`。  
这意味着模型不必原样复刻视频中的具体对象，而是可以学：

- **哪些步骤是通用路径**
- **哪些内容需要按目标任务替换**

例如：同样是“进入设置某项功能”，路径可以沿用，但电话号码、联系人、目标开关可以替换。  
这也是它能在 video-misaligned 和 AndroidWorld-Knowledge 中保持一定泛化的原因。

### 为什么这套设计有效

因果上看，能力提升来自三层约束同时变化：

- **信息瓶颈变化**：滑动窗口去掉长视频冗余，保留当前最相关上下文。
- **对齐瓶颈变化**：视频代理把“当前进度”显式估计，不再靠模型隐式记住全轨迹。
- **错误传播瓶颈变化**：反思代理在执行前拦截偏航动作，减少长链路错误级联。

最终带来的能力变化是：

- 知识注入更便宜；
- 决策更贴近真实操作路径；
- 对相近任务有更好的迁移性；
- 同时减少盲目探索步骤。

### 战略权衡

| 设计选择 | 解决的核心问题 | 带来的收益 | 代价 / 风险 |
|---|---|---|---|
| 演示视频替代手写知识 | 降低专家编写成本 | 知识注入更自然，时间更短 | 依赖视频质量与示范完整性 |
| 滑动窗口 | 降低长视频噪声 | 提升当前步骤决策聚焦度 | 窗口太小会漏帧，太大会引噪声 |
| Deep-Reflection Agent | 修正多帧推理错位 | 尤其改善复杂任务决策准确率 | 增加一次推理成本 |
| Video Agent 重定位 | 显式追踪执行进度 | 避免窗口推进失控 | 一旦重定位错，可能连锁偏航 |
| 同时输入 `Iv` 与 `Iu` | 支持相近任务迁移 | 能复用路径而不只是复刻演示 | 若源/目标任务差异过大，模型易混淆 |

---

## Part III：证据与局限

### 关键证据

- **[comparison] 共享基座下的显著提升**  
  所有方法都用 GPT-4o，说明提升主要来自**知识注入机制**而不是更强 backbone。  
  在 Mobile-Knowledge 上，Mobile-Agent-V 的 **SR 86.7%**，明显高于最佳基线 Agent-S2 的 63.3%，且 **Step 从 13.6 降到 7.3**。  
  这说明它不仅更容易成功，而且不是靠更多试错换来的。

- **[comparison] 单视频也能提供迁移价值**  
  在 AndroidWorld-Knowledge 上，每个场景只给最简单任务的一段视频，其他任务没有直接对应演示。  
  这种设定下方法仍达到 **31.3% SR**，比最佳基线 18.9% 高。  
  这表明它学到的不是纯粹的视频-动作硬匹配，而是一定程度的**路径迁移**。

- **[comparison] 成本-性能比优于人工知识编写**  
  知识注入时间上，操作视频平均 **0.7 分钟**，显著短于专家手写知识的 **5 分钟**；  
  性能上又接近专家知识（86.7% vs 90.0%）。  
  这直接对应论文的核心主张：**更 effortless、更 efficient**。

- **[ablation] 反思代理主要在难任务上发挥作用**  
  消融结果显示，deep-reflection agent 对简单任务提升有限，但对复杂任务更明显，尤其提升决策准确率。  
  这支持作者的解释：多帧时序理解一复杂，就更容易“看对页面、做错动作”，而反思阶段能补这个洞。

- **[analysis] 方法有泛化，但不是无限泛化**  
  Video-Misaligned 设置下，basic 任务较稳，但 normal/advanced 的 SR 和 DA 会下降。  
  结论很明确：它确实能迁移操作知识，但迁移半径受源任务与目标任务相似度限制。

### 1-2 个最关键指标

- **Mobile-Knowledge SR：86.7%**
- **视频知识注入时间：0.7 分钟 / 任务**

### 局限性

- **Fails when:** 演示视频质量差、关键帧提取漏掉关键状态、目标任务与演示任务路径差异过大、或界面变化超出视频可迁移范围时，方法容易错位或泛化失效。
- **Assumes:** 需要相关演示视频；依赖 GPT-4o 官方闭源 API；依赖 ADB 执行与 SoM/检测框等界面标注能力；评测中大量任务被设计为低规划复杂度，以突出知识利用本身。
- **Not designed for:** 无演示可参考的开放式长程规划任务；需要强探索、强记忆或复杂 UI grounding 的通用手机代理；完全脱离视频轨迹的零样本新流程学习。

### 复现与可扩展性上的现实约束

- **闭源模型依赖**：实验统一使用 GPT-4o API，这让框架效果更可控，但也限制了公开复现。
- **Benchmark 规模有限**：Mobile-Knowledge 仅 30 个设备特定任务，AndroidWorld-Knowledge 48 个任务，证据有说服力，但还不足以完全覆盖真实世界长尾。
- **视频示范依赖**：虽然“录一遍就行”比手写轻量得多，但依然要求用户先能正确完成一次。

### 可复用组件

- **关键帧冗余过滤**：适合手机录屏这种“大量静帧 + 少量跃迁”的视频。
- **局部视频窗口决策**：把长视频代理问题改成局部可解问题。
- **执行后反思校正**：适合任何“先生成动作、再用轨迹核对”的 GUI agent。
- **状态-轨迹显式重定位**：可迁移到桌面 GUI、网页 agent、甚至视频示范驱动的其它交互系统。

![[paperPDFs/Agentic_Reasoning_Applications_Autonomous_Web_Exploration_Research_Agents_Collective_multi_agent_reasoning/arXiv_2025/2025_Mobile_Agent_V_A_Video_Guided_Approach_for_Effortless_and_Efficient_Operational_Knowledge_Injection_in_Mobile_Automation.pdf]]