---
title: "History-Aware Reasoning for GUI Agents"
venue: arXiv
year: 2025
tags:
  - Multimodal_LLM
  - task/gui-automation
  - reinforcement-learning
  - dataset/AITW
  - dataset/Mind2Web
  - dataset/GUI-Odyssey
  - opensource/partial
core_operator: "通过“错误反思场景 + 教师纠错指南 + 记忆增强奖励”显式奖励模型在 GUI 长程任务中利用历史交互线索做推理"
primary_logic: |
  用户目标 + 当前截图 + 文本化历史 → 暖启动注入 GUI 感知、动作语义与 System-2 推理知识 → 在历史错误样本上用反思模板、纠错指南与混合奖励进行 GRPO 训练 → 再做推理格式对齐并与 grounding 任务混训 → 输出具备历史感知的 GUI 动作
claims:
  - "HAR-GUI-3B 在 AITW 上取得 70.2 SSR，超过 MP-GUI 的 69.2 与 InfiGUI-R1-3B 的 67.7 [evidence: comparison]"
  - "HAR-GUI-3B 在 Mind2Web 的 Cross-Task / Cross-Website / Cross-Domain 上分别达到 42.2 / 41.2 / 44.0 SSR，高于 UI-R1-3B 的 36.8 / 36.7 / 36.3 和 GUI-R1-3B 的 38.8 / 38.5 / 38.9 [evidence: comparison]"
  - "在消融中，去掉 reflection scenario、MAR 与 TMTS 的 vanilla GRPO 在 AITW / Mind2Web / ScreenSpot 上均低于 HAR，而在提示中显式强制关注历史的版本也劣于 HAR [evidence: ablation]"
related_work_position:
  extends: "UI-TARS (Qin et al. 2025)"
  competes_with: "UI-R1-3B (Lu et al. 2025); GUI-R1-3B (Luo et al. 2025)"
  complementary_to: "OS-Atlas (Wu et al. 2024); OmniParser (Lu et al. 2024b)"
evidence_strength: strong
pdf_ref: "paperPDFs/Agentic_Reasoning_Applications_Autonomous_Web_Exploration_Research_Agents_Self_evolving_agentic_reasoning/arXiv_2025/2025_History_Aware_Reasoning_for_GUI_Agents.pdf"
category: Multimodal_LLM
---

# History-Aware Reasoning for GUI Agents

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2511.09127), [Code](https://github.com/BigTaige/HAR-GUI)
> - **Summary**: 该文通过把 GUI agent 放进“带错误反思与历史线索奖励”的训练场景中，让模型不再把每一屏当成独立识别任务，而是按整个 episode 连续推理。
> - **Key Performance**: AITW **70.2 SSR**；Mind2Web SSR **42.2 / 41.2 / 44.0**（Cross-Task / Cross-Website / Cross-Domain）。

> [!info] **Agent Summary**
> - **task_path**: 用户目标 + 当前截图 + 文本化历史交互 -> GUI 动作（点击/输入/滚动/完成）
> - **bottleneck**: 现有 GUI CoT 在长程任务中对历史交互“失忆”，把 episode 退化成逐屏理解
> - **mechanism_delta**: 用教师生成的纠错指南构造 reflection scenario，并用含记忆奖励的混合 GRPO 奖励“正确且引用历史”的推理
> - **evidence_signal**: 多基准比较与 ablation 同时表明 HAR 优于 vanilla GRPO，且显式提示强制看历史并不能替代该训练机制
> - **reusable_ops**: [reflection-scenario construction, memory-augmented reward]
> - **failure_modes**: [纯 episodic RL 会削弱 grounding, 文本历史摘要若遗漏关键线索则历史感知上限受限]
> - **open_questions**: [MAR 是否衡量真实历史利用而非表面提及, 能否去除强教师与外部奖励模型实现更轻量闭环训练]

## Part I：问题与挑战

这篇论文要解决的，不是“GUI agent 会不会看屏幕”，而是**它能不能把多步 GUI 交互当成一个连续 episode 来推理**。

### 1. 问题定义
作者把 GUI 执行建模为：给定
- 用户总体目标 `G`
- 当前屏幕截图 `I_t`
- 文本指令模板 `P`
- 前序动作历史 `T_{<t}`

模型需要输出当前一步动作 `A_t`，动作空间包括 CLICK、SCROLL、TYPE、BACK、COMPLETE 等。

也就是说，接口本质上是：

**目标 + 当前观察 + 历史交互 → 当前动作**

### 2. 真正瓶颈是什么？
作者的核心判断是：现有 native GUI agents 即便已经具备 System-2 式显式推理，**其 CoT 仍然偏 history-agnostic**。具体表现为：

- 把长程任务拆成互相独立的“逐屏理解”；
- 当前动作主要由本屏局部视觉内容驱动，而不是由整条任务链驱动；
- 已完成的步骤、跨 app 切换、阶段性子目标这些关键信号，难以稳定进入当前决策。

因此真正的瓶颈不是单纯的视觉识别，而是**episodic reasoning 的短期记忆弱**。

### 3. 为什么现在要解决？
因为 GUI agent 已经过了“纯感知起步期”：
- grounding 已经有较强模型；
- GUI-specific MLLM 也开始出现；
- RL + System-2 CoT 已能提升 reasoning。

但一旦进入真实长程任务，瓶颈就从“看不懂屏幕”转向“**记不住前面做过什么**”。这也是为什么作者把重点放在**历史感知推理模式**，而不是继续堆更复杂 prompt 或更大模型。

### 4. 边界条件
这篇方法有几个明确边界：

- **历史以文本形式传递**，而不是拼接整段图像历史，原因是图像 token 成本高；
- 重点任务是**长程 GUI 自动化/导航**，不是开放式桌面代理的全部问题；
- 训练依赖**历史错误样本**与**教师模型生成的纠错指南**；
- 目标不只是提高 episodic reasoning，也要维持 grounding 精度。

---

## Part II：方法与洞察

HAR 的设计目标不是简单“把更多历史喂给模型”，而是**改变模型使用历史的方式**：从“看当前页面猜下一步”，变成“结合 episode 上下文做当前决策”。

### 核心直觉

**什么发生了变化？**  
作者没有停留在 inference prompt 里提醒模型“请关注历史”，而是构造了一个**错误驱动的反思训练场景**：让模型看到自己过去的错误、收到针对该错误的简短纠错指南、再通过 RL 奖励它在 CoT 中真正调用历史线索。

**改变了哪个约束/信息瓶颈？**  
改变的是**历史信息的使用机制**，而不是历史信息的可见性：
- 过去：历史在输入里，但经常没进入 CoT；
- 现在：奖励函数显式偏好“动作正确 + 格式正确 + 推理中用到了历史”的输出。

**带来了什么能力变化？**  
模型从逐屏 reactive actor 更接近 episode-level reasoner：
- 更能判断某个子任务是否已完成；
- 更少被当前屏幕的局部显著元素误导；
- 更能处理跨页面、跨 app 的长序列交互；
- 在 OOD 中文小程序场景里更稳。

### 方法主线

#### 1）GUI Scenario Warm-up：先补齐 GUI 基础知识
HAR 先做一个暖启动阶段，通过 SFT 给底座模型注入 GUI 专业能力，主要包含三类数据：

1. **GUI understanding 数据**  
   包括 screen QA、screen summary、widget captioning、clickability、grounding 等，让模型先“看懂界面”。

2. **Act2Sum（Action-to-Summary）**  
   给定目标、当前截图和动作，利用教师模型生成一句动作摘要。  
   它的意义不是生成自然语言本身，而是让模型学会：**一个动作在整个任务链里代表什么语义**。这样历史记录不只是“点了哪里”，而是“为什么这样点”。

3. **System-2 CoT 蒸馏**  
   使用更强教师模型生成 GUI episodic reasoning 的 slow-thinking CoT，并筛掉低质量样本，让学生模型继承 GUI 场景下的推理结构。

这一步解决的是：如果底座连 GUI 元素、动作语义和基本 CoT 都不稳，后面的“反思式强化”就无从发力。

#### 2）Learning From Failure：在错误样本上学会“带着历史纠错”
这是 HAR 的核心阶段。

作者先用暖启动后的模型在 episode 数据上跑推理，把做错的样本提取为 **historically incorrect samples**。然后围绕这些 hard samples 做三件事：

##### a. 合成针对错误的纠错指南
对每个错误样本，教师模型生成最多三条简短 guideline。它们不直接泄露标准答案，而是指出：

- 这类错误可能忽略了什么；
- 当前应检查哪些任务线索；
- 是否需要结合历史操作再判断当前动作。

这相当于给 RL 探索引入了**外部 episodic reasoning 知识**。

##### b. 构造 reflection scenario
作者不再沿用普通 inference-format 指令，而是构造一个反思模板，显式告诉模型：

- 你之前输出了什么错误动作；
- 你当时的错误 CoT 是什么；
- 这里有几条纠错指南；
- 现在请先说明自己哪里错了，再重新推理并给出答案。

这会把训练过程从“直接猜动作”改成：

**错误归因 → 历史回看 → 重做当前决策**

##### c. 设计 hybrid reward
HAR 的 RL 奖励不是单一 success reward，而是三部分组合：

- **格式奖励 `R_format`**：保证模型按反思模板输出；
- **动作奖励 `R_action`**：鼓励动作正确；对 click 这类坐标动作还加入距离 shaping，鼓励更细粒度的定位；
- **记忆奖励 `R_memory` / MAR**：判断 CoT 是否真正纳入了历史交互信息。

其中最关键的是 **MAR**。  
它要解决的问题是：模型可能“碰巧答对”，但不是因为真的理解了 episode 历史。MAR 通过奖励“CoT 中显式涉及相关历史”的输出，把优化目标从“这一步答对”推进到“按整条 episode 做推理”。

#### 3）Round-2 RL：把“带扶手训练”的能力迁回真实推理接口
Round-1 RL 中有外部 guideline，但真实测试时没有。于是作者再做一轮 RL：

- 把输入恢复到正常 inference-format；
- 采用 **TMTS（task mixing training strategy）**，把 grounding 任务和 episodic reasoning 任务混合训练。

其动机很直接：作者观察到**纯 episodic reasoning 的 RL 会伤害 grounding**。  
因此第二轮训练不是简单格式对齐，而是让模型把第一轮里学会的“历史感知推理”迁移回真实接口，同时保住点击定位能力。

### 为什么这套设计有效？
从因果链看，HAR 的收益来自四步闭环：

1. **暖启动**让模型先拥有 GUI 感知、动作语义和基础 CoT；
2. **错误样本 + 纠错指南**把训练火力集中到真正困难的 episodic case；
3. **MAR + 动作奖励**让 RL 不只优化结果，还优化“是否调用历史线索”；
4. **Round-2 + TMTS**把在反思场景中学到的能力迁回正常推理格式，并抑制 grounding 退化。

所以 HAR 的关键不是“prompt 更会提醒”，而是**训练目标真的变了**。

### 战略权衡

| 设计选择 | 改变的瓶颈 | 收益 | 代价 / 风险 |
|---|---|---|---|
| 文本历史而非图像历史 | 长程历史输入成本过高 | 更省算力，易扩到长 horizon | 依赖历史摘要质量，可能丢失视觉细节 |
| 错误样本上的反思训练 | 普通 RL 只优化 action，难改 CoT 模式 | 把学习集中到 episodic reasoning 真难点 | 需要额外挖掘 hard samples |
| 教师生成纠错指南 | 纯探索难补足 GUI 特定知识 | 给模型“如何改错”的方向信号 | 依赖强教师、成本高 |
| MAR 记忆奖励 | 动作答对不代表真的用了历史 | 促使短期记忆显性化进入 CoT | 奖励判别器可能有偏差 |
| TMTS 混合 grounding | reasoning RL 伤害点击定位 | 同时保留 reasoning 与 grounding | 训练流程更复杂、任务配比需调 |

---

## Part III：证据与局限

### 关键实验信号

#### 1）多 benchmark 上，HAR 的长程 GUI 推理确实更强
最核心证据来自 GUI episodic reasoning 基准上的一致提升：

- **AITW**：HAR-GUI-3B 达到 **70.2 SSR**，高于 MP-GUI 的 69.2、InfiGUI-R1-3B 的 67.7；
- **Mind2Web**：Cross-Task / Cross-Website / Cross-Domain 的 SSR 达到 **42.2 / 41.2 / 44.0**，显著高于 UI-R1-3B、GUI-R1-3B 等同类 3B reasoner；
- **GUI-Odyssey**：总体 **62.31 SSR**，超过 Qwen2.5-VL-7B 的 58.39。

这说明 HAR 的收益不是某一个 benchmark 上的偶然 prompt trick，而是对长程 GUI episode 有普遍帮助。

#### 2）HAR 没有靠牺牲 grounding 换推理
作者专门验证了 reasoning RL 的常见副作用：点击定位退化。结果显示：

- **ScreenSpot**：HAR-GUI-3B 平均 **83.3**
- **ScreenSpot-V2**：HAR-GUI-3B 平均 **86.2**

而消融实验也表明：**只做 episodic RL 会削弱 grounding**，TMTS 正是在修这个问题。  
所以 HAR 的价值不只是“会想”，而是“会按对地方”。

#### 3）OOD 中文小程序评测支持“历史感知”不是英文 benchmark 特化
作者手工构建了中文支付宝小程序 benchmark。这里的信号很关键，因为很多 GUI agent 在分布外场景会暴露出：

- 指令跟随变差；
- 只做表面 screen understanding；
- 长序列越后面越失控。

HAR-GUI-3B 在多个类别上显著优于同尺度模型，并在部分场景可与更大模型竞争。论文中的 case study 也直接展示了：
- HAR 的 CoT 会明确回顾“我已经完成了哪些步骤”；
- 基线则常把每一步都当成新的独立页面理解问题。

#### 4）Ablation 支持“收益来自训练机制，不是提示词”
这是整篇论文最强的因果证据：

- **vanilla GRPO**（无 reflection scenario / MAR / TMTS）明显弱于 HAR；
- **在 prompt 中强制要求关注历史** 的版本，反而比 HAR 更差；
- 说明单靠提示词约束，并不能把“历史感知”变成模型的默认推理模式。

这也直接回答了论文的核心主张：  
**必须在训练阶段改变模型对历史的使用方式，而不是只在推理阶段提醒它。**

### 局限性

- **Fails when**: 关键状态变化没有被文本历史摘要保留下来时，模型可能沿着不完整历史继续推理；另外，涉及高精度坐标的点击动作本身更难，论文也专门为此设计了 reward shaping。  
- **Assumes**: 依赖强教师模型做 CoT 蒸馏和纠错指南合成（Qwen2.5-VL-72B-Instruct），还依赖外部大模型作为 MAR 判别器（Qwen3-235B-A22B）；训练使用 8×A100 80GB，并需要多阶段数据构造、hard sample 挖掘和任务混训。  
- **Not designed for**: 不是为无历史的单步 GUI 感知任务设计；也不是为开放式真实在线计算机使用、安全约束控制、或非 GUI agent 场景直接设计。

### 可复用组件

这篇论文最值得迁移到别的 agent 系统中的，不一定是 HAR-GUI-3B 本身，而是这些训练算子：

- **reflection-scenario construction**：把“之前错了什么”显式引入训练；
- **teacher-synthesized correction guidelines**：把 hard samples 变成有方向的自我纠错信号；
- **memory-augmented reward (MAR)**：奖励模型是否真的用到了历史；
- **coordinate-shaped action reward**：对 spatial action 做更细粒度优化；
- **reasoning + grounding task mixing**：防止 reasoning RL 破坏界面定位能力。

### 一句话结论
HAR 抓住的核心不是“GUI agent 不够会想”，而是**它不会把刚刚发生过的事稳定带进当前思考**。这篇论文的贡献，在于把这个问题从 prompt 层提醒，推进到了训练目标层改造，并用多 benchmark、消融和 OOD 证据说明：**history-aware reasoning 是 GUI agent 迈向长程稳定执行的关键一步。**

![[paperPDFs/Agentic_Reasoning_Applications_Autonomous_Web_Exploration_Research_Agents_Self_evolving_agentic_reasoning/arXiv_2025/2025_History_Aware_Reasoning_for_GUI_Agents.pdf]]