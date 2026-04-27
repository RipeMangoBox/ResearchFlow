---
title: "WINELL: Wikipedia Never-Ending Updating with LLM Agents"
venue: arXiv
year: 2025
tags:
  - Others
  - task/knowledge-base-updating
  - task/document-editing
  - multi-agent
  - iterative-search
  - supervised-finetuning
  - dataset/Wikipedia
  - opensource/promised
core_operator: "用章节准则诱导、迭代式多代理网页聚合和历史编辑微调编辑器，为 Wikipedia 页面持续生成带引用的细粒度更新建议"
primary_logic: |
  Wikipedia旧版本页面 + 时间窗内网页来源 → 诱导章节内容准则并进行迭代式搜索/抽取/聚合，筛出非冗余且值得写入的新事实 → 在对应段落做最小改动式编辑并输出可供人工审核的更新建议
claims:
  - "在600+编辑实例上，Llama-3.1-8B-Editor 达到 91.7% key facts coverage、18.7% commentary coverage 和 49.8 token change，优于 GPT-4o 的 91.3% / 53.1% / 73.4 [evidence: comparison]"
  - "在45个高活跃 Wikipedia 页面、1400+事实性人工编辑上，完整 WINELL 达到 15.4% hard coverage 与 34.4% soft coverage；去掉 agentic search 后降至 9.5% 与 21.5%，说明迭代搜索是端到端覆盖提升的主要来源 [evidence: ablation]"
  - "对100条代理编辑的人类评审中，68%可直接接受、29%需修改后接受、3%被拒绝，说明系统已适合作为 human-in-the-loop 的编辑建议器但尚不适合无人审核自动写入 [evidence: analysis]"
related_work_position:
  extends: "INFOGENT (Reddy et al. 2025)"
  competes_with: "GPT-4o; Qwen2.5-7B-Instruct"
  complementary_to: "Citation Needed (Redi et al. 2019); Improving Wikipedia Verifiability with AI (Petroni et al. 2023)"
evidence_strength: moderate
pdf_ref: "paperPDFs/Agentic_Reasoning_Applications_Autonomous_Web_Exploration_Research_Agents_Self_evolving_agentic_reasoning/arXiv_2025/2025_WINELL_Wikipedia_Never_Ending_Updating_with_LLM_Agents.pdf"
category: Others
---

# WINELL: Wikipedia Never-Ending Updating with LLM Agents

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2508.03728), [Project/Code](https://github.com/gangiswag/AutoWikiUpdate)
> - **Summary**: WINELL 把 Wikipedia 更新拆成“按章节定义什么值得写 → 迭代式上网找新事实 → 以人类编辑风格做最小改写”三步，从而持续生成带引用、可人工审核的页面更新建议。
> - **Key Performance**: 编辑器在 600+ 实例上达到 91.7% key-facts coverage / 18.7% commentary coverage；端到端在 45 个页面上达到 15.4% hard coverage / 34.4% soft coverage，100 条建议中 68% 被资深编辑直接接受。

> [!info] **Agent Summary**
> - **task_path**: Wikipedia 旧页面 + 时间窗内网页新闻 -> 带引用的段落级更新建议
> - **bottleneck**: 真正难点不是搜到新闻，而是在高噪声、强冗余的网页流中筛出 Wikipedia-worthy 的事实，并把它最小改动地放入正确章节
> - **mechanism_delta**: 用“章节准则诱导 + Navigator/Extractor/Aggregator 迭代搜索聚合 + 历史编辑微调编辑器”把发现、筛选、落点、改写四个子问题解耦
> - **evidence_signal**: 去掉 agentic search 后 hard/soft coverage 从 15.4%/34.4% 降到 9.5%/21.5%，而去掉 section criteria 主要伤害 section accuracy（33.2% -> 28.6%）
> - **reusable_ops**: [section-criteria induction, iterative search-aggregate loop]
> - **failure_modes**: [事实找到了但落错 subsection, 新闻噪声导致改动不重要或文风不够中性]
> - **open_questions**: [如何在复杂页面层级中稳定做 section mapping, 如何把段落编辑安全扩展到 infobox 和表格]

## Part I：问题与挑战

WINELL 要解决的不是普通问答，而是**开放世界、持续发生、带结构约束的知识库维护**。  
对一个既有 Wikipedia 页面，系统需要不断监控外部世界的新事实，并把“真的值得写进去”的那部分，以**中立、可验证、最小改动**的方式放回页面。

### 真问题是什么
现有 Wikipedia 更新主要靠人工编辑，导致两个现实问题：

1. **时效性滞后**：新事实发表后，进入 Wikipedia 往往有明显延迟。
2. **覆盖不均衡**：热门页面有人盯，冷门页面可能长期无人更新。

而且，过去工作多半只做：
- infobox 更新；
- 或假设“要加入的事实已经给定”。

WINELL 关注的是更难的版本：**事实还没给你，需要自己从网上发现、筛选、定位并写成可审阅编辑建议。**

### 真正瓶颈
这篇论文最重要的判断是：瓶颈不是单点的“检索”或“生成”，而是四个环节串联后的复合瓶颈：

- **发现**：网页里有没有新事实、搜不搜得到；
- **筛选**：它是否足够重要、是否重复、是否适合百科体裁；
- **定位**：应该写到哪一节，而不是仅仅“页面里某处”；
- **改写**：怎样只做必要修改、保留原文结构、避免主观评论和新闻腔。

这也是为什么通用 LLM 直接“看新闻改段落”并不够：它容易写得太多、带入评论、或把事实放错地方。

### 为什么现在值得做
两个条件刚好成熟：

- **LLM agent** 已经能做多步网页搜索和信息聚合；
- **Wikipedia 历史编辑记录**提供了大规模“人类如何更新页面”的监督信号和自动评测代理。

### 输入 / 输出 / 边界
- **输入**：时刻 \(T\) 的 Wikipedia 页面版本 + 时间窗 \([T, T+\Delta t]\) 内的网页/新闻来源。
- **输出**：带引用的、定位到具体 section/paragraph 的编辑建议。
- **边界条件**：
  - 需要**人工审核**，不是自动写回线上百科；
  - 主要处理**段落文本**；
  - 当前**不覆盖 infobox、表格**；
  - 评测集中在 2024 年高活跃页面，偏向有足够人工编辑记录的场景。

## Part II：方法与洞察

WINELL 的设计哲学很明确：**不要让一个大模型端到端“重写 Wikipedia”，而是把更新过程拆成可控的决策链。**

### 方法拆解

#### 1) Section Criteria Induction：先定义“这一节该写什么”
系统先读取整篇文章及其章节层级，让 LLM 为每个 section 生成**内容纳入准则**。  
这一步的作用不是写内容，而是把页面原本隐式的编辑规范显式化。

例如：
- “Early Life” 应接受哪些事实；
- “Career” 应接受哪些里程碑；
- 某个小节适合写长期背景还是近期事件。

这相当于先生成一份**页内编辑 policy**。

#### 2) Agentic Update Aggregation：再去网上迭代式找更新
这一部分改编自 INFOGENT，包含三个角色：

- **Navigator**：搜索网页、找候选来源；
- **Extractor**：从来源里抽取可能的新事实，并判断相关 section；
- **Aggregator**：结合对应 section 当前内容和历史已收集更新，决定这条信息应当  
  **ignore / add / replace**。

关键不在“三个名字”，而在**带反馈的迭代**：
- 如果抽到的是重复、无关或不重要信息；
- Navigator 就调整查询，继续搜索别的方面；
- 直到把页面仍然缺失的更新逐步补齐。

所以它不是一次性 query，而是**coverage-driven** 的搜索闭环。

#### 3) Fine-Grained Editing：最后做“最小增量编辑”
作者利用 Wikipedia 历史人工编辑训练编辑器。训练样本三元组是：

- 原始段落
- 编辑后段落
- 可能触发这次修改的来源内容

模型学到的不是“重写文章”，而是：
- 保留原文主体；
- 只插入必要事实；
- 避免把新闻评论、主观表述一起搬进百科。

这一步很关键，因为论文发现通用 instruct 模型往往要么**抄太多**，要么**引入 commentary**。

#### 4) 自动评估：用历史人工编辑做代理真值
为了避免大规模人工比对，作者把任务设成回放：

- 给系统时刻 \(T\) 的页面；
- 限制它只能看 \(T\) 到 \(T+\Delta t\) 之间发表的来源；
- 再拿同时间窗内真实人类做出的 factual edits 作为对照。

评估分成两层：
- **Soft Coverage**：事实找到了就算，不要求放在同一节；
- **Hard Coverage**：事实不仅找到了，还得放到和人类相同 section。

这个设计很有价值，因为它把“找到了”和“放对了”拆开诊断。

### 核心直觉

WINELL 的关键旋钮可以概括成三条因果链：

1. **章节名 → 章节准则**  
   变化：把模糊的页面结构变成显式的 section-level 编辑约束  
   影响的瓶颈：降低 section mapping 的语义歧义  
   能力变化：更容易把新事实放到对的位置，提升 section accuracy

2. **单次检索 → 迭代式 agentic 检索聚合**  
   变化：从固定查询改成围绕“剩余信息空缺”继续搜索  
   影响的瓶颈：改变了检索分布，提高召回并减少冗余  
   能力变化：端到端更新覆盖更高，尤其对多来源、多事件页面更有效

3. **通用 instruct 生成 → 人类编辑行为微调**  
   变化：把生成先验从“尽量完整复述来源”改成“最小必要编辑”  
   影响的瓶颈：减少 commentary 泄漏和过度改写  
   能力变化：接近人类编辑的中性、节制风格

### 为什么这套设计有效
因为 Wikipedia 更新不是一个单目标问题。  
“找对事实”“判断是否值得写”“选 section”“写得像百科”这四件事依赖的信息不同、错误类型也不同。把它们塞进一个一步式生成器，模型很难同时做好；拆开后，每一步都能围绕对应瓶颈优化。

### 战略权衡

| 设计选择 | 主要解决的瓶颈 | 带来的能力 | 代价/风险 |
|---|---|---|---|
| 章节准则诱导 | section 语义不明确 | 更好的落点判断 | 依赖 GPT-4.1，准则质量不稳定 |
| 迭代式多代理搜索 | 单查询漏召回、结果冗余 | 更高 factual coverage | 搜索/API 成本更高，受网页噪声影响大 |
| 历史编辑微调编辑器 | 通用 LLM 过度改写、带评论 | 更少 token change、更中性 | 依赖高质量历史编辑三元组，且训练数据过滤后规模不大 |
| 原子事实覆盖评测 | 人工评测不可扩展 | 可大规模评测“发现”与“放置” | 依赖 GPT-4o 做原子事实分解与蕴含判断，存在代理误差 |

## Part III：证据与局限

相较于以往“只做 infobox”或“假设事实已知”的工作，WINELL 的能力跃迁在于：它首次把**开放网页中的新事实发现 + 结构化写回 Wikipedia**连成了一个端到端流程。  
不过，它的绝对 coverage 仍然不高，因此更像是**编辑助手**，不是自动编辑员。

### 关键证据

- **比较信号：编辑器确实学到了“像人一样少改但改对”**  
  在 600+ 编辑实例上，Llama-3.1-8B-Editor 达到 91.7% key facts coverage、18.7% commentary coverage、49.8 token change，优于 GPT-4o 的 91.3% / 53.1% / 73.4。  
  结论：增益不是来自更大模型，而是来自**历史编辑行为监督**。

- **消融信号：端到端提升主要来自 agentic search，不是 prompt 小修小补**  
  完整 WINELL 在 45 个页面上达到 15.4% hard coverage、34.4% soft coverage。  
  去掉 agentic search 后降到 9.5% / 21.5%；  
  去掉 section criteria 后 coverage 变化不大，但 section accuracy 从 33.2% 降到 28.6%。  
  结论：**搜索闭环负责“找到更多事实”**，**章节准则负责“放得更对”**。

- **上界信号：即使给人类引用源，section mapping 仍然难**  
  Oracle 使用人类真实引用源时，hard/soft coverage 到 30.6% / 62.2%，但 SAcc 也只有 41.4%。  
  结论：瓶颈不仅是找源，更是**把事实融入正确 section**。

- **人工评审信号：适合作为建议器，但仍需要人类把关**  
  在 100 条代理编辑上，5 位资深 Wikipedia 编辑给出：68% 直接接受、29% 修改后接受、3% 拒绝。  
  主要问题是风格/清晰度、主观内容、以及“改了但不够重要”。  
  结论：系统有实用性，但目前最合理的部署方式是 **human-in-the-loop**。

### 局限性
- **Fails when**: 页面新闻流过于密集且噪声高（如政治人物、名人）；同一事实可落多个 section 时容易放错位置；更新本身不够显著时，系统可能提出“ technically true 但编辑价值不高”的修改。
- **Assumes**: 依赖 English Wikipedia 历史编辑与新增 citation URL 作为监督/评测锚点；依赖 GPT-4.1、GPT-4.1 mini、GPT-4o 和 Google Search API；训练中部分来源内容由 GPT-4o 补写增强，意味着编辑器学习到的“新闻输入分布”并非完全真实网页原文。
- **Not designed for**: 无人工审核的自动发布；infobox、表格、列表等非段落结构更新；无引用、原创研究式内容；脱离 Wikipedia 规范的自由文本写作。

### 复现与扩展注意
- 这是一个**闭源 API 依赖较重**的系统：章节准则、搜索代理和部分评测都依赖 OpenAI 模型与外部搜索服务。
- 自动评测虽然可规模化，但其“原子事实分解 + entailment”也带有 LLM judge 偏差。
- 评测页面限定为高活跃页面；作者认为长尾页面更有价值，但论文并未直接验证。

### 可复用组件
- **section-criteria induction**：适合任何层级化文档的“该往哪一节写”问题。
- **Navigator / Extractor / Aggregator 循环**：适合开放网页环境中的持续更新发现。
- **edit-history supervised editor**：适合“最小增量、风格受限”的文档编辑任务。
- **hard/soft coverage 评测框架**：适合区分“事实发现能力”和“结构落点能力”。

![[paperPDFs/Agentic_Reasoning_Applications_Autonomous_Web_Exploration_Research_Agents_Self_evolving_agentic_reasoning/arXiv_2025/2025_WINELL_Wikipedia_Never_Ending_Updating_with_LLM_Agents.pdf]]