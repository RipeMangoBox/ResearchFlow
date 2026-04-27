---
title: "VisualWebBench: How Far Have Multimodal LLMs Evolved in Web Page Understanding and Grounding"
venue: arXiv
year: 2024
tags:
  - Survey_Benchmark
  - task/MLLM-evaluation
  - multi-granularity-evaluation
  - qa-style-evaluation
  - grounding-diagnosis
  - dataset/VisualWebBench
  - opensource/partial
core_operator: 以真实网页截图为载体，将网页理解拆成网站级、元素级、动作级七个统一QA评测任务，细粒度诊断MLLM的OCR、理解、推理与grounding能力。
primary_logic: |
  网页理解与grounding评测需求 → 基于139个真实网站构建网站/元素/动作三级七任务并统一为QA输入输出 → 用ROUGE-L、F1与Accuracy进行可比评分 → 揭示MLLM在文本密集网页、低分辨率输入和精确grounding上的能力边界
claims:
  - "Claim 1: 在 VisualWebBench 上，Claude Sonnet 和 GPT-4V 的平均分分别仅为 65.8 和 64.6，说明真实网页细粒度理解与 grounding 对现有顶级 MLLM 仍然具有显著挑战性 [evidence: comparison]"
  - "Claim 2: 最佳开源模型 LLaVA-1.6-34B 平均分为 50.5，明显低于 Claude Sonnet 的 65.8，表明开源与闭源 MLLM 在网页理解与 grounding 上仍存在明显差距 [evidence: comparison]"
  - "Claim 3: 对 LLaVA-1.6 的分辨率消融显示更高输入分辨率会稳定提升 VisualWebBench 得分，而在直接预测 bbox/point 的 grounding 设定下，多数通用 MLLM 几乎失效 [evidence: ablation]"
related_work_position:
  extends: "WebSRC (Chen et al. 2021)"
  competes_with: "MMMU (Yue et al. 2023); Mind2Web (Deng et al. 2024)"
  complementary_to: "WebArena (Zhou et al. 2023); VisualWebArena (Koh et al. 2024)"
evidence_strength: moderate
pdf_ref: paperPDFs/Agentic_Reasoning_Benchmarks_Applications_of_Agentic_Reasoning_Web_Agents/arXiv_2024/2024_VisualWebBench_How_Far_Have_Multimodal_LLMs_Evolved_in_Web_Page_Understanding_and_Grounding.pdf
category: Survey_Benchmark
---

# VisualWebBench: How Far Have Multimodal LLMs Evolved in Web Page Understanding and Grounding

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2404.05955), [Project](https://visualwebbench.github.io/)
> - **Summary**: 该工作提出一个面向真实网页截图的多粒度评测基准，把网页能力拆成网站级、元素级、动作级七类任务，从而系统暴露出现有 MLLM 在文本密集网页理解、grounding 与低分辨率输入上的明显短板。
> - **Key Performance**: Claude Sonnet 平均分 65.8，GPT-4V 64.6；最佳开源 LLaVA-1.6-34B 为 50.5。

> [!info] **Agent Summary**
> - **task_path**: 网页截图/网页问题/候选元素框 → 文本答案、选项选择或元素定位判断
> - **bottleneck**: 现有评测要么过于通用，要么只看端到端 web agent 成败，无法分离诊断网页场景中的 OCR、布局理解、动作推理与 grounding 错误
> - **mechanism_delta**: 用真实网站截图 + DOM/点击结果构造网站级、元素级、动作级七个统一 QA 任务，替代仅靠通用 VQA 或端到端代理评测
> - **evidence_signal**: 20 个 MLLM 横评显示最好模型平均分仅 65.8，且自由坐标 grounding 与低分辨率设定下性能明显崩塌
> - **reusable_ops**: [多粒度任务拆解, Playwright加DOM辅助标注]
> - **failure_modes**: [文本密集页面中的元素定位失败, 低分辨率输入下OCR与理解显著退化]
> - **open_questions**: [如何评测无候选框的真实grounding能力, 如何把单步截图评测扩展到多步动态网页交互]

## Part I：问题与挑战

这篇论文解决的不是“通用图像问答够不够强”，而是一个更具体的瓶颈：**MLLM 是否真的理解网页这种特殊视觉载体**。

网页与普通自然图像不同，难点集中在三点：

1. **文本极密集**：网页里大量关键信息是小字号文字，不是大物体。
2. **结构强且层级深**：同一页面同时包含标题、导航、正文、按钮、卡片、搜索框、跳转入口等，理解依赖布局关系。
3. **交互性强**：不仅要“看懂”，还要知道“该点哪里、点了会发生什么”。

作者认为现有评测有两个错位：

- **通用 MLLM benchmark** 多是物体/场景中心，不能覆盖网页里的细粒度 OCR、布局理解、元素 grounding。
- **Web agent benchmark** 多看端到端成败，但把感知、理解、grounding、规划混在一起，无法诊断究竟哪一步坏了。

所以真正瓶颈是：**缺少一个能把网页理解拆开测、并且足够接近真实网页分布的评测基准**。

### 输入 / 输出接口与边界

VisualWebBench 把问题统一成 QA 风格，输入通常是：

- 网页截图；
- 可选地带问题、说明文字、红框目标或 8 个候选元素框。

输出则是：

- 文本生成答案；
- 或 A-H 的多选项；
- 或在附加分析中输出 bbox / point 坐标。

其边界条件也很明确：

- 主要是**单张网页截图**评测，不是长时序交互。
- action prediction 只看**单次点击后的页面标题变化**。
- grounding 默认采用**8 选 1 候选框**设定，而不是完全自由坐标定位。

这也解释了“为什么现在要做”：因为 MLLM 已经被用于 web agent，但如果连网页本身都没看懂，端到端代理能力就很难可靠提升。

## Part II：方法与洞察

这是一篇 benchmark/evaluation paper，核心贡献不是新模型，而是**把网页理解问题设计成一个更可诊断的测量系统**。

### 任务分解与数据构造

VisualWebBench 由 **139 个真实网站、87 个子领域、12 个大类、1.5K 样本**构成，覆盖 science、sports、travel、government、animals 等多域网站。网站由 SimilarWeb 挑选，使用 Playwright 自动渲染与保存，再结合 HTML/DOM 信息和人工校验构建标注。

| 层级 | 任务 | 主要评测能力 | 标签来源 |
|---|---|---|---|
| 网站级 | Captioning | 整页理解与概括 | meta description + GPT-4V 草拟 + 人工校验 |
| 网站级 | WebQA | 网页理解与轻度推理 | 人工撰写问题 |
| 网站级 | Heading OCR | 标题定位与识别 | HTML `<h1>` |
| 元素级 | Element OCR | 长文本元素 OCR | DOM 文本 + 自动框 |
| 元素级 | Element Grounding | 元素文本-区域对齐 | DOM 描述 + 候选框 |
| 动作级 | Action Prediction | 点击后果推断 | Playwright 点击后页面 title |
| 动作级 | Action Grounding | 指令到可操作元素映射 | 人工指令 + 候选框标注 |

评分也保持统一而简单：

- 开放生成：ROUGE-L
- WebQA：SQuAD-style F1
- 多选题：Accuracy

### 核心直觉

**它真正改变的不是模型，而是测量瓶颈。**

从因果上看，这个 benchmark 的关键变化是：

- **从“端到端成功率”改成“能力拆解”**  
  以前只能看到代理最终成败，现在能区分是 OCR 不行、网页语义不行、还是 grounding 不行。

- **从“自然图像分布”改成“真实网页分布”**  
  网页截图含有高密文本、小元素、复杂布局，测出来的能力更贴近真实 web 应用。

- **从“自由交互难评分”改成“统一 QA 可比评测”**  
  把七类任务都收敛到统一输入输出协议，降低评测噪声，方便横向比较不同 MLLM。

因此，这个设计带来的能力变化不是“模型更强了”，而是**研究者第一次能系统看见：模型到底卡在网页理解链条的哪一环**。

更具体地说：

> 评测形式改变 → 测量到的能力因子更可分离 → 能显式暴露 OCR / grounding / action reasoning 的单点短板 → benchmark 对 web 场景更有诊断价值。

### 战略取舍

| 设计选择 | 改善了什么 | 代价 / 风险 |
|---|---|---|
| 真实网站截图而非合成页面 | 更接近真实网页分布 | 页面样式受抓取时刻影响，动态性有限 |
| 网站/元素/动作三级拆解 | 能定位具体失败环节 | 仍未覆盖多步规划与长期交互 |
| 统一 QA 格式 | 易于比较不同 MLLM | 与真实 agent API/动作空间仍有距离 |
| grounding 默认 8 选 1 | 降低坐标输出噪声，提升可评测性 | 会高估真实无标注场景下的 grounding 能力 |
| DOM/Playwright 自动抽取 + 人工校验 | 兼顾规模与标注客观性 | 依赖 DOM、title、meta 等网页元信息质量 |

## Part III：证据与局限

### 关键证据

**信号 1：横评结果说明“网页理解”仍远未被解决。**  
最强模型也不高：Claude Sonnet 65.8，GPT-4V 64.6；最佳开源 LLaVA-1.6-34B 只有 50.5。尤其在 Action Prediction / Action Grounding 这类更接近交互的任务上，很多模型接近随机水平，说明网页上的“看懂后还能正确行动”是当前明显短板。

**信号 2：VisualWebBench 不等价于已有基准。**  
与 MMMU 的某些子任务存在相关性，尤其 WebQA 和 Action Prediction 这类更吃推理的任务；但与 Mind2Web 的整体相关性较低（论文给出的平均相关约 0.27），说明端到端 agent 成绩不能替代网页感知与 grounding 诊断。  
结论不是“通用能力没用”，而是：**通用推理只是必要条件，不是网页能力的充分条件。**

**信号 3：分辨率是网页场景里的硬约束。**  
作者对 LLaVA-1.6 做了输入分辨率消融，结果显示分辨率提高会稳定拉升 VisualWebBench 分数，且 336→448 的提升尤其明显。对网页这种 1280 宽、文字密集的输入来说，低分辨率天然吃亏。

**信号 4：多选 grounding 会掩盖真实定位问题。**  
在默认多选设定下，GPT-4V 的 grounding 分数很高；但一旦切换到直接输出 bbox/point 的 REC 设定，绝大多数通用 MLLM 几乎失效。这个差异非常重要：  
**会选框 ≠ 真的能在未标注截图上准确定位。**  
GUI 专项模型在坐标输出上更强，但整体网页理解并未同步提升，论文还指出可能存在对通用指令能力的灾难性遗忘。

### 1-2 个最值得记住的指标

- **Claude Sonnet 65.8 / GPT-4V 64.6 / LLaVA-1.6-34B 50.5**：说明真实网页 benchmark 仍很难。
- **Mind2Web 与 VisualWebBench 平均相关仅约 0.27**：说明细粒度网页能力评测与端到端 agent 成绩并不等价。

### 局限性

- **Fails when**: 需要多步浏览、滚动、表单填写、页面动态刷新或完全无候选框的自由坐标 grounding 时，这个 benchmark 不能完整反映真实 agent 难度；在文本极密且元素极小的页面上，当前模型尤其容易失败。
- **Assumes**: 假设网页能被 Playwright 稳定渲染，且可从 HTML/DOM 中抽取标题、文本、跳转信息；默认是单截图或单点击后果评测；部分强基线依赖闭源 API；高分辨率推理对显存和计算资源有要求。
- **Not designed for**: 不是完整 web agent 规划 benchmark，不评测长期任务分解、错误恢复、跨页面记忆，也不覆盖移动端 GUI 或实时在线网页漂移鲁棒性。

### 可复用组件

- **三级任务分解框架**：网站级 / 元素级 / 动作级。
- **Playwright + DOM 辅助标注流水线**：适合扩展新的网页评测集。
- **候选框多选 vs 自由坐标双设定**：非常适合做 grounding 诊断。
- **统一 QA 协议**：方便横向比较开源与闭源 MLLM。

![[paperPDFs/Agentic_Reasoning_Benchmarks_Applications_of_Agentic_Reasoning_Web_Agents/arXiv_2024/2024_VisualWebBench_How_Far_Have_Multimodal_LLMs_Evolved_in_Web_Page_Understanding_and_Grounding.pdf]]