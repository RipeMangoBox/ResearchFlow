---
title: "Context-Aware Human Behavior Prediction Using Multimodal Large Language Models: Challenges and Insights"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/video-understanding
  - in-context-learning
  - autoregressive-prediction
  - scene-captioning
  - dataset/PROX
  - dataset/PROX-S
  - opensource/no
core_operator: "用视觉上下文、少样本 ICL 和可选中间步自回归，把通用预训练 MLLM 约束到第三人称人类行为预测的封闭交互标签空间。"
primary_logic: |
  第三人称场景图像/图像序列/场景描述 + 过去2秒交互标签 + 少量ICL示例
  → 在提示中显式定义任务、输出格式与动作词表，并利用视觉上下文完成场景落地，可选先预测1秒/2秒中间交互
  → 输出3秒后目标帧的人类交互标签集合
claims:
  - "在基于 PROX/PROX-S 构造的 329 条第三人称序列上，GPT-4o + caption + 15 个 ICL 示例 + 自回归配置达到 92.8% cosine similarity 和 66.1% accuracy score [evidence: comparison]"
  - "相较于仅用历史标签的 blind 文本输入，加入视觉上下文通常能提升预测质量，其中 image sequence 与 caption 在零样本下整体优于 blind 或 single image [evidence: comparison]"
  - "ICL 增益在大多数模型上随示例数增加而上升但在约 10 个示例后趋于饱和，而中间步自回归平均仅带来约 1% accuracy 改善，未形成显著能力跃迁 [evidence: ablation]"
related_work_position:
  extends: "AntGPT (Zhao et al. 2024)"
  competes_with: "PALM (Kim et al. 2024); AntGPT (Zhao et al. 2024)"
  complementary_to: "GG-LLM (Graule and Isler 2024); 3D Dynamic Scene Graphs (Gorlo et al. 2024)"
evidence_strength: moderate
pdf_ref: "paperPDFs/Other_Multimodal_Large_Language_Model_MLLM_based_related_Embodied_Learning_2025/arXiv_2025/2025_Context_Aware_Human_Behavior_Prediction_Using_Multimodal_Large_Language_Models_Challenges_and_Insights.pdf"
category: Embodied_AI
---

# Context-Aware Human Behavior Prediction Using Multimodal Large Language Models: Challenges and Insights

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2504.00839), [Project](https://cap-mllm.github.io/)
> - **Summary**: 这篇论文提出一个无需微调的模块化 MLLM 行为预测框架，系统比较视觉表示、ICL 与自回归中间步，评估通用多模态模型能否从第三人称视角预测 3 秒后的人类交互行为。
> - **Key Performance**: 最佳配置达到 **92.8% semantic similarity** 与 **66.1% accuracy score**；零样本最佳为 **GPT-4o + caption 的 61.1% accuracy**。

> [!info] **Agent Summary**
> - **task_path**: 第三人称室内 RGB 场景/过去交互标签 -> 3 秒后目标帧的人类交互标签集合
> - **bottleneck**: 通用 MLLM 很难把开放式视觉-语言理解稳定映射成封闭、多标签、短时未来行为预测；尤其受限于空间 grounding、时间上下文压缩与 prompt 约束
> - **mechanism_delta**: 用 caption/sequence 视觉上下文 + 少量 ICL 示例 + 可选 1/2/3 秒中间步预测，在推理时把冻结 MLLM “转接”为行为预测器
> - **evidence_signal**: 跨 6 个 MLLM、4 类视觉表示、0-15 个 ICL 示例的系统 ablation 显示 caption + ICL 最稳，最佳配置达 66.1% accuracy
> - **reusable_ops**: [caption 作为视觉到语言桥接, ICL 将开放式生成约束到领域动作词表]
> - **failure_modes**: [单图或强遮挡场景下未来意图歧义大, 部分开源模型在多图 ICL 与结构化输出时会错配或生成损坏 token]
> - **open_questions**: [跨数据集与真实机器人部署时 ICL 增益是否仍成立, 如何在不依赖闭源 API 的前提下提升空间推理与格式稳定性]

## Part I：问题与挑战

### 任务是什么
论文研究的是**第三人称视角的人类行为预测**：  
给定当前场景视觉上下文 \(V_0\)、过去 2 秒的人类交互标签历史 \(L_{-2:0}\)，预测 **3 秒后目标帧** 的交互标签集合 \(L_3\)。

这里的标签不是单一动作，而是 **verb-noun** 形式的交互，如 `touch-table`，并且**同一帧可以有多个标签**，例如 `[sit on-sofa, touch-table]`。这比传统“单标签动作 anticipation”更难，也更接近机器人实际需要的语义接口。

### 真正的瓶颈是什么
真正难点不只是“预测未来”，而是把**开放域 MLLM**变成一个**受约束、可落地、可多标签输出**的预测器：

1. **开放式生成 vs. 封闭动作空间**  
   MLLM 天然擅长自由描述，但这里需要稳定输出 42 个交互标签中的一个或多个。

2. **视觉 grounding 不稳**  
   纯文本 LLM 容易“脑补”场景；而通用 MLLM 即使看图，也未必能稳定提取对未来行为有用的空间/可供性线索。

3. **时间上下文不足**  
   单帧很难看出意图，行为预测需要历史标签、场景状态和短时动态共同支持。

4. **不能依赖昂贵微调**  
   机器人场景通常跨环境、跨任务，若每个域都专门训练一个预测器，迁移成本高。

### 为什么现在值得做
因为 MLLM 已经具备：
- 比传统监督模型更强的跨场景 commonsense；
- 多模态输入能力，可把当前视觉场景“落地”到现实；
- 通过 ICL 在推理时快速适配任务，而不必重新训练。

这使得“**不微调、直接用通用 MLLM 做人类行为预测**”第一次变得值得系统验证。

### 边界条件
- 数据来自 **PROX / PROX-S**，是**室内第三人称 RGB**场景；
- 预测目标是**3 秒后单帧交互标签集合**，不是连续轨迹；
- 标签空间固定为 PROX-S 的 **42 个交互**；
- 论文特意选 3 秒预测窗，是因为 1-2 秒过于容易；3 秒设置下，约 **47.4%** 样本目标标签不同于当前标签，约 **25.8%** 样本目标帧有多个标签，任务才足够“非平凡”。

## Part II：方法与洞察

### 方法框架
论文没有训练一个新模型，而是搭建了一个**模块化预测框架**，对不同 MLLM 配置进行系统比较。四个核心旋钮：

1. **MLLM backbone**
   - GPT-4o / GPT-4o-mini
   - Qwen2-VL-72B / 7B
   - LLaVA-NeXT-Video-34B / 7B

2. **视觉上下文表示**
   - `blind`：只给历史标签，不给视觉
   - `image`：当前单张图
   - `sequence`：多帧图像序列
   - `caption`：对场景、人、物体、动作和可供性交互的文本描述

3. **ICL（In-Context Learning）**
   - 在 prompt 中加入若干输入-输出示例
   - 作用不是“教知识”，而是把模型输出**约束到任务格式、标签集合和时间模式**

4. **自回归中间步预测**
   - 先预测 1 秒、2 秒，再给出 3 秒
   - 目的是把“大跳步”未来预测拆成更平滑的滚动推断

### 核心直觉
这篇论文最关键的变化不是网络结构，而是**把“预测问题”重新组织成 MLLM 更擅长的输入形式**。

**What changed**：  
从“训练一个专用行为预测器”，改成“在推理时用视觉表示 + ICL + 输出格式约束，临时把通用 MLLM 对齐到该任务”。

**Which bottleneck changed**：  
- `caption/sequence` 改善了**视觉 grounding 与时间上下文缺失**；
- `ICL` 把开放域生成收缩到**领域标签分布**；
- `autoregressive rollout` 试图缓解**3 秒跳跃预测的跨度问题**。

**What capability changed**：  
模型从“会看图、会描述”变成“能在固定动作词表内，对第三人称场景做多标签未来交互预测”，而且**不需要 task-specific fine-tuning**。

更细一点地说，`caption` 之所以常常优于纯图像序列，核心原因是：  
它把视觉信息**压缩成 LLM 原生更擅长处理的文本语义**，降低了从视觉 token 到离散动作标签之间的接口摩擦。

### 为什么这个设计有效
- **视觉输入**负责“当前场景是什么”
- **历史标签**负责“人刚刚在做什么”
- **ICL 示例**负责“在这个领域里答案应该长什么样”
- **自回归中间步**负责“把 3 秒后的预测拆成更连续的推演”

也就是说，它不是在让 MLLM凭空“预测未来”，而是在让它做一种**受限语义滚动预测**。

### 策略权衡表

| 设计选择 | 改变的约束/瓶颈 | 带来的能力 | 代价/风险 | 论文观察 |
|---|---|---|---|---|
| blind → image | 从纯语言猜测变成有场景 grounding | 减少幻觉 | 单帧缺少运动线索，可能引入歧义 | 提升有限，个别模型还会变差 |
| image → sequence | 增加短时动态线索 | 更容易判断意图与动作延续 | 更多视觉 token，成本更高 | 整体优于单图 |
| sequence → caption | 把感知压缩成语言友好表示 | 提高场景理解与标签对齐 | 依赖 caption 质量，可能冗余 | GPT-4o 上最有效 |
| 增加 ICL | 把开放生成收缩到领域格式/词表 | 明显提升结构化输出质量 | 上下文成本上升，示例选择敏感 | 大多在 10 个左右后饱和 |
| 加自回归中间步 | 缩短预测跨度 | 平滑最终输出 | 额外生成步骤，收益可能很小 | 平均仅约 +1% accuracy |

## Part III：证据与局限

### 关键证据信号

**信号 1：无需微调，MLLM 的确能做“非平凡”预测。**  
最佳配置（GPT-4o + caption + 15 ICL + autoregressive）达到 **92.8% cosine similarity** 和 **66.1% accuracy score**。这说明通用 MLLM 不只是做语义描述，而是能在 3 秒未来、可能多标签的设置下给出可用预测。

**信号 2：视觉上下文是必要的，但“怎么给视觉”比“有没有视觉”更重要。**  
零样本下，`caption` 与 `sequence` 整体优于 `blind` 和 `single image`。这意味着未来行为预测需要的不只是当前图像，而是**更强的时间/语义压缩**。单图有时反而会制造歧义。

**信号 3：ICL 是最有效的推理时对齐旋钮。**  
随着 ICL 示例数从 0 增加到 15，大多数模型整体变好，支持 H3。  
但增益在约 **10 个示例后趋于饱和**，说明 ICL 更像“任务对齐器”，不是无限扩展的知识库。

**信号 4：自回归中间步不是核心能力跃迁点。**  
中间 1 秒/2 秒预测平均只带来约 **1% accuracy** 改善，更多像微调输出而不是根本性增强，因此 H4 没有得到强支持。

**信号 5：模型家族差异揭示了真实瓶颈在“空间推理 + 结构化输出”。**  
- GPT-4o 最能利用 caption；
- Qwen 对视觉增益较弱，说明空间信息提取并不稳定；
- LLaVA 在 ICL 场景下由于 prompt 模板和图文对位问题，甚至难以稳定输出结构化标签。

一个很重要的“所以呢”是：  
这篇论文真正证明的，不是“所有 MLLM 都能很好预测行为”，而是**只要用对输入接口，强 MLLM 已经能在不微调条件下完成有用的第三人称行为预测；但能力瓶颈仍主要卡在空间 grounding、ICL 稳定性和输出格式控制上。**

### 局限性
- **Fails when**: 需要更强空间推理、严重遮挡、目标物体可见性差、仅靠单图难以判定未来意图时，模型容易退化；LLaVA 在多图 ICL + 结构化输出场景下还会出现图文错配和损坏 token。
- **Assumes**: 已有准确的过去交互标签历史；评测集中于室内第三人称 RGB 场景与固定 42 类交互标签；ICL 示例来自与评测样本同分布的 PROX/PROX-S 派生集合；最佳结果依赖闭源 GPT-4o API，而开源模型实验需要 **2×NVIDIA H200**。
- **Not designed for**: 长时开放世界预测、连续轨迹预测、跨数据集泛化验证、实时低算力部署、以及音频/深度等额外模态融合。

### 可复用组件
- **caption-as-bridge**：先把视觉场景压成文本，再交给语言模型做未来标签推断；
- **ICL label-space steering**：用少量示例把开放生成对齐到封闭动作词表；
- **intermediate label rollout**：用 1s→2s→3s 的中间标签预测做轻量滚动推理；
- **模块化评测框架**：适合复用到其他机器人视角的人类活动预测或 MLLM 行为分析任务。

## Local PDF reference

![[paperPDFs/Other_Multimodal_Large_Language_Model_MLLM_based_related_Embodied_Learning_2025/arXiv_2025/2025_Context_Aware_Human_Behavior_Prediction_Using_Multimodal_Large_Language_Models_Challenges_and_Insights.pdf]]