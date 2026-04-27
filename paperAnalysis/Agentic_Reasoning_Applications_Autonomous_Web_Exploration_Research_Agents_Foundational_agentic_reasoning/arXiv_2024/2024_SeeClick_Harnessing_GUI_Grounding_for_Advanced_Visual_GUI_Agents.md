---
title: "SeeClick: Harnessing GUI Grounding for Advanced Visual GUI Agents"
venue: arXiv
year: 2024
tags:
  - Embodied_AI
  - task/gui-grounding
  - task/gui-automation
  - grounding-pretraining
  - continual-pretraining
  - lora
  - dataset/ScreenSpot
  - dataset/MiniWob
  - dataset/AITW
  - dataset/Mind2Web
  - opensource/full
core_operator: "利用自动构造的跨平台 GUI grounding 数据对 Qwen-VL 做持续预训练，使模型能从截图和指令直接生成可操作元素坐标。"
primary_logic: |
  界面截图 + 自然语言指令（下游任务中再加入历史动作） → 用 Web/移动端自动构造的 grounding、OCR 与界面理解数据持续预训练 LVLM，学习“描述↔位置”的 GUI 对应关系 → 输出点击坐标或输入动作，在移动端、桌面和网页上执行操作。
claims:
  - "在 ScreenSpot 上，SeeClick 的平均点击准确率为 53.4%，高于 CogAgent 的 47.4%，且模型更小（9.6B vs 18B） [evidence: comparison]"
  - "在 MiniWob 上，SeeClick 用 2.8K 训练轨迹取得 67.0 分，超过使用 1.3M 轨迹的 Pix2Act 的 64.6 分 [evidence: comparison]"
  - "在多个 SeeClick 检查点上，ScreenSpot grounding 能力提升与 MiniWob、AITW、Mind2Web 性能提升保持同向变化 [evidence: analysis]"
related_work_position:
  extends: "Qwen-VL (Bai et al. 2023)"
  competes_with: "CogAgent (Hong et al. 2023); Pix2Act (Shaw et al. 2023)"
  complementary_to: "Synapse (Zheng et al. 2023)"
evidence_strength: moderate
pdf_ref: paperPDFs/Agentic_Reasoning_Applications_Autonomous_Web_Exploration_Research_Agents_Foundational_agentic_reasoning/arXiv_2024/2024_SeeClick_Harnessing_GUI_Grounding_for_Advanced_Visual_GUI_Agents.pdf
category: Embodied_AI
---

# SeeClick: Harnessing GUI Grounding for Advanced Visual GUI Agents

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2401.10935), [Code](https://github.com/njucckevin/SeeClick)
> - **Summary**: 这篇论文把视觉 GUI agent 的核心瓶颈明确为“GUI grounding”，并通过跨 Web/移动端的持续预训练，让模型仅凭截图就能更可靠地找到该点哪里、该输什么。
> - **Key Performance**: ScreenSpot 平均点击准确率 **53.4%**（CogAgent 47.4%）；MiniWob **67.0**，超过 Pix2Act 64.6，且只用 **2.8K** 训练轨迹（vs 1.3M）

> [!info] **Agent Summary**
> - **task_path**: GUI 截图 + 自然语言指令 + 最近若干步动作 -> 下一步 GUI 动作（点击/输入/选择）
> - **bottleneck**: 现有 LVLM 在自然图像上有 grounding 能力，但在高密度文本、图标密集、控件相似的 GUI 中无法稳定做精确定位
> - **mechanism_delta**: 用自动构造的 Web/移动端元素-位置配对数据，对 Qwen-VL 进行 GUI grounding 持续预训练，并直接生成归一化点击坐标
> - **evidence_signal**: 在 ScreenSpot 上显著优于 Qwen-VL/CogAgent，且 grounding 提升与 MiniWob、AITW、Mind2Web 的下游提升同步
> - **reusable_ops**: [HTML 自动抽取元素-指令对, 文本↔坐标双向预训练]
> - **failure_modes**: [icon/widget 的精细定位仍弱, 纯视觉网页代理在复杂真实网站上仍落后于基于 HTML 候选的方法]
> - **open_questions**: [如何支持拖拽/双击等复杂动作, 如何统一跨平台训练而不牺牲单任务性能]

## Part I：问题与挑战

这篇论文解决的不是“GUI agent 会不会规划”，而是一个更底层、但更决定成败的问题：**模型能不能在截图里精确找到要操作的元素**。

### 1. 真正的问题是什么
现有 GUI agent 大多依赖 HTML、DOM、Android View Hierarchy 等结构化文本来感知界面。这样做有三个硬伤：

1. **并不总能拿到结构化信息**：尤其是 iOS、桌面应用。
2. **结构化文本冗长且不完整**：会占据 LLM 上下文，还丢失布局、图标、图片等关键信息。
3. **平台不统一**：网页、移动端、桌面端各有不同观察空间和动作接口。

所以，如果想做一个真正通用的 GUI agent，更自然的路线是：**直接看截图操作**。

### 2. 为什么现在值得解这个问题
原因在于两股趋势刚好交汇：

- 一方面，LLM/LVLM 已经有了不错的通用理解与推理能力；
- 另一方面，纯文本 GUI agent 的平台可迁移性和可访问性问题越来越明显。

作者的判断是：**现在限制视觉 GUI agent 的主要短板，已经不是“不会理解任务”，而是“不会把语言指令精确对到屏幕位置”**。

### 3. 输入/输出接口
SeeClick 的下游接口很直接：

- **输入**：当前 GUI 截图 + 用户指令 + 最近几步历史动作
- **输出**：下一步动作  
  例如：
  - `click(x, y)`
  - `type(text)`
  - `select(value)`
  - 以及少量系统动作（back/home/enter/swipe）

### 4. 边界条件
它主要面向的是**低层 GUI 执行**，不是完整的高层任务规划系统。也就是说，论文重点不在“长期计划”，而在“当前这一步到底该点哪里”。

---

## Part II：方法与洞察

SeeClick 的关键思路很清晰：**先把 GUI grounding 训明白，再谈 GUI agent。**

### 方法骨架

#### 1. 把 GUI grounding 显式定义为训练目标
作者把任务写成：

- 给定截图和文字指令
- 预测目标元素的位置

具体实现上，没有另造 1000 个坐标 token，而是让模型像生成自然语言一样直接输出归一化坐标，例如：

- `click (0.49, 0.40)`

这一步很重要：它把 LVLM 的输出空间直接对齐到 GUI 操作需要的“点”。

#### 2. 自动构造 GUI grounding 数据
为了让模型学会 GUI 分布，而不是只会自然图像分布，作者专门构造了跨平台 GUI 数据。

**Web 侧**：
- 从 Common Crawl 抓取约 30 万网页；
- 自动从 HTML 中抽取两类可操作元素：
  - 可见文本元素
  - 带 `title` 属性的元素（很多 icon/按钮靠它描述）
- 形成：
  - 文本 -> 坐标
  - 坐标 -> 文本（类似 OCR/反向 grounding）

**移动端**：
- 把 widget captioning 数据“反过来用”，把元素描述变成指令，元素位置变成目标；
- 加入 RICO 自动收集的数据；
- 再加 UI summarization，增强整体界面理解。

**通用视觉语言数据**：
- 混入 LLaVA 数据，防止模型 GUI 化后丢掉通用视觉能力。

最终混成约 **100 万样本** 的持续预训练集。

#### 3. 在 Qwen-VL 上做持续预训练
作者不是从零训练，而是基于 **Qwen-VL**：

- 对视觉编码器和 LLM 都做 LoRA 微调
- 训练约 10k steps
- 8 张 A100，约 24 小时

这说明它的核心贡献不是“大模型堆料”，而是**把预训练分布改对**。

#### 4. 再适配到具体 GUI agent 任务
预训练后的 SeeClick 再分别适配到：

- MiniWob
- AITW
- Mind2Web

即从“会定位元素”进一步变成“会执行下一步操作”。

### 核心直觉

过去的 LVLM 虽然会 grounding，但那主要是**自然图像里的物体 grounding**。GUI 完全不是这个分布：

- 文本更密
- 图标更小
- 控件更相似
- 精细位置误差会直接变成动作失败

所以作者动的不是“更强的 planner”这个旋钮，而是：

**把训练分布从自然图像 grounding 改成 GUI grounding，把输出目标从泛化描述改成操作坐标。**

这会带来一条很明确的因果链：

**训练分布变了**  
→ 模型开始学习“指令语义 ↔ GUI 元素 ↔ 屏幕位置”的绑定  
→ 低层点击精度提升  
→ 下游 GUI agent 的动作成功率提升

这也是论文最重要的洞察：  
**对视觉 GUI agent 来说，grounding 不是附属能力，而是基座能力。**

### 为什么这个设计有效
1. **截图-only 统一了平台接口**：网页、手机、桌面都能用同一种观测方式。
2. **自动挖掘数据降低标注成本**：尤其 Web 的可见文本和 `title` 属性，天然提供弱监督。
3. **双向任务强化对齐**：不仅学“描述找位置”，也学“位置读文本”，让元素语义和空间位置绑定更紧。
4. **直接输出点坐标更贴近真实动作**：不必依赖 HTML 候选集。

### 战略权衡

| 设计选择 | 得到的能力 | 代价/风险 |
| --- | --- | --- |
| 只看截图，不依赖 HTML/DOM | 跨平台统一，适用于 iOS/桌面等无结构化元数据场景 | 失去结构化候选集，真实网页上精确点击更难 |
| GUI grounding 持续预训练 | 明显提升“知道该点哪里”的能力 | 需要构造大量元素-位置配对数据 |
| 直接生成归一化点坐标 | 动作接口简单、通用、与 GUI 执行自然对齐 | 对小图标、密集元素、细粒度误差更敏感 |
| 统一移动/桌面/Web 基座 | 有跨环境迁移能力 | 联合训练会略降单任务性能上限 |

---

## Part III：证据与局限

### 关键实验信号

#### 1. 比较信号：ScreenSpot 证明它确实更会“找”
作者新建了 **ScreenSpot**，覆盖 iOS、Android、macOS、Windows、Web，包含 600+ 截图、1200+ 指令，且刻意包含大量 icon/widget。

关键信号不是“SeeClick 能跑”，而是：

- **SeeClick 53.4% 平均点击准确率**
- **CogAgent 47.4%**
- GPT-4V 只有 16.2%

这说明通用 LVLM 的自然图像 grounding 不能直接迁移到 GUI，而 GUI-specific grounding 预训练是有效的。

#### 2. 样本效率信号：MiniWob 超过强视觉基线
在 MiniWob 上，SeeClick 作为**纯视觉方法**达到 **67.0**，超过 Pix2Act 的 **64.6**，但只用了 **2.8K** 训练轨迹，而 Pix2Act 用了 **1.3M**。

这个结果支持的不是“它更大”，而是：  
**先学会 GUI grounding，能显著降低下游模仿数据需求。**

#### 3. 泛化信号：AITW / Mind2Web 都跟着涨
- 在更严格的 **instruction-wise AITW split** 上，SeeClick overall **59.3**，高于 Qwen-VL 的 **54.3**；点击准确率 **66.4 vs 57.4**
- 在 Mind2Web 上，相比 Qwen-VL，**Step Success Rate 几乎翻倍**

这说明提升不是某个单一 benchmark 的偶然收益，而是跨移动端和网页任务的共性收益。

#### 4. 因果信号：grounding 提升与 agent 提升同步
作者用不同训练 checkpoint 分析后发现，**ScreenSpot 上的 grounding 提升，与 MiniWob/AITW/Mind2Web 的性能提升呈一致趋势**。

这点很关键：  
论文真正的主张不是“SeeClick 这个模型有效”，而是  
**“GUI grounding 是视觉 GUI agent 的关键因果瓶颈。”**

### 1-2 个最值得记住的指标
- **ScreenSpot: 53.4%** 平均点击准确率
- **MiniWob: 67.0**，且只用 **2.8K** 训练轨迹

### 局限性

- **Fails when**: 目标元素是小尺寸 icon/widget、界面元素极其密集、多个候选位置非常相似时，模型容易出现“认对了区域但点得不够准”的 near-miss；在复杂真实网页上，纯视觉坐标预测仍弱于 HTML 候选式方法。
- **Assumes**: 假设可以从 HTML、widget captioning、RICO 等来源自动获得大量元素级监督；依赖现成开放 LVLM（Qwen-VL）作为基座；下游多步 GUI 任务仍需要任务特定训练数据；持续预训练资源约为 8×A100、24 小时。
- **Not designed for**: 拖拽、双击等复杂操作；不经任务适配就完成长时程多步 GUI 任务；依赖屏幕外上下文或完整页面结构才能解决的网页任务。

### 可复用组件
1. **ScreenSpot**：可直接作为 GUI grounding 评测基准。
2. **Web 自动数据构造管线**：从可见文本和 `title` 属性抽元素监督。
3. **移动端“caption 反转”策略**：把 widget captioning 转成 grounding 数据。
4. **截图到坐标的统一动作接口**：适合作为更高层 planner 的低层执行器。

### 一句话结论
SeeClick 的价值不只是“又一个 GUI agent”，而是它把视觉 GUI agent 的主矛盾明确成了：  
**先解决 grounding，agent 才真正开始成立。**

![[paperPDFs/Agentic_Reasoning_Applications_Autonomous_Web_Exploration_Research_Agents_Foundational_agentic_reasoning/arXiv_2024/2024_SeeClick_Harnessing_GUI_Grounding_for_Advanced_Visual_GUI_Agents.pdf]]