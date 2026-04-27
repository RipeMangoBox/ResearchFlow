---
title: "BrowseComp: A Simple Yet Challenging Benchmark for Browsing Agents"
venue: arXiv
year: 2025
tags:
  - Survey_Benchmark
  - task/web-browsing
  - task/open-domain-question-answering
  - inverted-question-construction
  - semantic-equivalence-grading
  - short-answer-benchmark
  - dataset/BrowseComp
  - opensource/full
core_operator: 以“难找但易验”的反向构题法构造短答案网页检索题，并用语义等价评分与计算扩展分析评估持续浏览能力
primary_logic: |
  评测持久网页浏览能力 → 人工从已知事实反向构造“难找但易验”的短答案问题并做模型/人工筛查 → 用语义等价判分、置信度与测试时计算分析评测代理 → 揭示搜索持久性、查询改写与校准边界
claims:
  - "Claim 1: 在 1,266 道 BrowseComp 题目上，GPT-4o、GPT-4.5、GPT-4o with browsing 和 OpenAI o1 的准确率分别为 0.6%、0.9%、1.9% 和 9.9%，而 Deep Research 达到 51.5%，说明该基准对当前模型仍未饱和 [evidence: comparison]"
  - "Claim 2: 在 1,255 个被人工尝试的问题中，人类训练员在 2 小时预算内仅解出 29.2%，且解出的答案只有 86.4% 与参考答案一致，表明该基准具有显著的人类难度 [evidence: analysis]"
  - "Claim 3: Deep Research 的表现会随测试时计算平滑提升，并且对每题进行最多 64 次采样后用 majority/weighted/best-of-N 聚合，可比单次尝试再提升约 15%–25% [evidence: analysis]"
related_work_position:
  extends: "SimpleQA (Wei et al. 2024)"
  competes_with: "GAIA (Mialon et al. 2023); BEARCUBS (Song et al. 2025)"
  complementary_to: "Humanity's Last Exam (Phan et al. 2025); GPQA (Rein et al. 2024)"
evidence_strength: moderate
pdf_ref: paperPDFs/Agentic_Reasoning_Benchmarks_Applications_of_Agentic_Reasoning_Web_Agents/arXiv_2025/2025_BrowseComp_A_Simple_Yet_Challenging_Benchmark_for_Browsing_Agents.pdf
category: Survey_Benchmark
---

# BrowseComp: A Simple Yet Challenging Benchmark for Browsing Agents

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2504.12516), [GitHub / simple-evals](https://github.com/openai/simple-evals)
> - **Summary**: 这篇工作提出一个专门评测网页浏览代理“持续搜索 + 策略改写”能力的短答案基准，用“难找但易验”的问题设计把评测重点从长答案生成转移到真实的搜索能力上。
> - **Key Performance**: Deep Research 在 1,266 题上达到 51.5%，显著高于 o1 的 9.9%；人类训练员在 2 小时预算内仅解出 29.2%。

> [!info] **Agent Summary**
> - **task_path**: 高难度、明确定义的网页事实查询 -> 短字符串答案（可附置信度）
> - **bottleneck**: 现有开放域 QA / 网页评测大多测“能否搜到现成答案”，难以单独测出持续浏览、查询改写与跨站点拼接线索的能力
> - **mechanism_delta**: 用人工反向构题生成“答案好验、路径难找”的问题，并配合语义等价评分与测试时计算分析来分离浏览能力
> - **evidence_signal**: Deep Research 51.5% vs o1 9.9%，且准确率随浏览 effort 与多样本聚合持续上升
> - **reusable_ops**: [反向构题, 语义等价判分]
> - **failure_modes**: [搜索路径陷入局部最优, 浏览后过度自信]
> - **open_questions**: [与真实用户浏览任务的相关性有多强, 如何扩展到多模态与交互网页而不牺牲易评测性]

## Part I：问题与挑战

BrowseComp要解决的不是一般的开放域问答，而是一个更“代理化”的核心问题：**当答案确实在开放网页上存在，但不会通过几次普通搜索直接出现时，模型能否持续、灵活、可靠地把它找出来。**

### 真正的难点是什么
过去很多检索/事实问答 benchmark 的难点，往往混着几种因素：
- 答案生成是否流畅；
- 长答案是否组织得好；
- 用户问题是否有歧义；
- 网页操作是否复杂。

这会让我们很难判断：模型到底是**不会浏览**，还是**不会表达**。

BrowseComp的切法更干净：  
它把任务收缩成**“短答案、易验证，但发现路径很长”**。这样一来，主要误差就更可能来自：
- 是否能持续搜索；
- 是否会改写查询；
- 是否能从多个站点拼装线索；
- 是否能在大搜索空间里避免盲目 brute force。

### 输入 / 输出接口
- **输入**：一个自包含的、高约束度问题。
- **输出**：一个短的最终答案；论文评测时还要求模型给 explanation、exact answer 和 confidence。
- **评分**：用 LLM judge 判断预测答案与参考答案是否语义等价。

### 边界条件
BrowseComp有意**不**覆盖以下能力：
- 真实用户查询中的歧义澄清；
- 长报告生成与引用组织；
- 图片、视频、音频证据检索；
- 复杂交互网页或事务型操作。

所以它不是“完整的网页助理评测”，而是一个**聚焦 persistent browsing 核心能力**的压力测试。

### 为什么现在值得做
2024–2025 开始，Deep Research、Operator、Gemini Deep Research 等系统把“能持续上网查资料”变成前台能力。此时社区需要一个：
1. 比普通 factual QA 更难；
2. 比完整 computer-use 环境更低噪、更易复现；
3. 能随 test-time compute 拉开差距的基准。

BrowseComp正好卡在这个空白处。

## Part II：方法与洞察

### 评测设计

#### 1）数据如何构造
数据由人工训练员纯手工创建，整体遵循 SimpleQA 风格：  
答案应当短、稳定、无争议，并且有证据支持。

但 BrowseComp 的关键额外要求是“**极难找**”。为此作者要求训练员做三重检查：
1. 当时的现有模型解不出来：GPT-4o（有/无 browsing）、o1、以及早期 Deep Research 都不能解；
2. 做 5 次简单 Google 搜索，答案不能轻易出现在搜索结果首页；
3. 题目应难到让另一个人 **10 分钟内解不出**；若第二位训练员能在短时间内较高比例解出，则要求修题。

#### 2）核心构题技巧：反向构题
训练员不是从问题出发找答案，而是从一个已知“seed fact”出发，收集多个特征，再把它们反写成问题。

这会产生一个非常关键的性质：

- **验证答案很便宜**：一旦猜到答案，确认它是否对通常很快；
- **发现答案很昂贵**：在不知道答案前，搜索空间极大。

这正好把 benchmark 压力集中到代理最关键的能力：**搜索策略**，而不是写作能力。

#### 3）唯一性问题如何缓解
作者也承认：这种“反向构题”无法严格证明参考答案一定是唯一正确答案。  
缓解方法包括：
- 让训练员只写自己足够熟悉的题；
- 若怀疑可能有别的答案，就继续加约束；
- 若第二位训练员在 10 分钟内找到别的合法答案，就要求返工；
- 后处理时，对 Deep Research 64 次都失败的题做复查，并删除 21 道格式不匹配、表述歧义或答案错误的题。

最终数据集从 1,287 题清洗为 **1,266 题**。

#### 4）评分与附加分析
- **评分**：短字符串答案用语义等价判分，使用与 Humanity’s Last Exam 相同的 grader prompt。
- **可靠性分析**：要求模型输出 confidence，用于做 calibration analysis。
- **能力诊断**：还分析 test-time compute scaling、多样本投票、任务难度分布等。

### 核心直觉

**what changed**：从“容易搜到的开放域问答”切换成“难找但易验的网页搜索题”。  

**which bottleneck changed**：  
短答案 + 语义等价评分，去掉了长答案写作和评审噪声；  
反向构题 + 对抗式筛题，则把难度集中到搜索空间爆炸、查询改写和多跳拼线索上。  

**what capability changed**：  
这个 benchmark 因而更能区分：
- 只是“有浏览工具”的模型；
- 和真正会持续搜索、会回溯、会改写查询、会自我验证的代理。

**为什么这个设计因果上有效**：  
如果给出正确答案后，模型大多能回过头在网上找到支持证据，那就说明问题不在“网上没有证据”，而在“此前没找到对的搜索路径”。  
论文对 0% pass-rate 任务的后续分析基本支持这一点：很多题不是不可解，而是**太难 crack**。

### 策略权衡

| 设计选择 | 带来的能力诊断收益 | 代价 / 风险 |
| --- | --- | --- |
| 反向构题 | 放大搜索难度，突出 persistence 与 creativity | 不能形式化保证答案唯一 |
| 短答案 + 语义等价判分 | 低噪、易大规模评测 | 不测长答案组织与引用质量 |
| 对抗式筛题（模型失败 + 简单搜索失败 + 人工复核） | 避免 benchmark 过快饱和 | 构题成本高，题目分布受作者兴趣影响 |
| 置信度与多样本聚合分析 | 能诊断 calibration 与 compute scaling | 依赖额外算力，且置信度本身未必可靠 |

## Part III：证据与局限

### 关键信号

1. **模型分层信号很强**  
   GPT-4o、GPT-4.5 基本接近 0；GPT-4o 加 browsing 也只有 1.9%；o1 到 9.9%；Deep Research 跳到 51.5%。  
   这说明瓶颈不是“有没有网页工具”这么简单，而是**是否能战略性地搜索**。

2. **人类也觉得难**  
   在 1,255 个被尝试的问题里，人类训练员在 2 小时预算内只解出 29.2%。  
   这说明 BrowseComp 不是靠表面花活“卡模型”，而是真的有较高检索难度。

3. **test-time compute 确实有用**  
   准确率会随浏览 effort 平滑上升；对同一题做多次采样并聚合后，还能再提升约 15%–25%。  
   这很重要，因为它表明该 benchmark 对 agent 类系统的一个核心特征敏感：**愿意花更多测试时计算，能力就继续涨**。

4. **best-of-N 有效，说明模型常常“知道自己答对了”**  
   虽然绝对 calibration 很差，但 best-of-N 始终最好，说明模型内部存在某种有用的正确性信号，只是它没法把这个信号校准成可靠概率。

5. **浏览会放大过度自信问题**  
   GPT-4o w/ browsing 和 Deep Research 的 calibration error 都很高，后者甚至达到 91%。  
   所以“能搜到更多东西”不自动等于“知道自己什么时候不确定”。

### 局限性
- **Fails when**: 问题存在多个合理答案、答案格式约定不清、网页内容动态变化，或者需要图片/视频/音频/交互式网页证据时，BrowseComp 的分数可能不能准确反映真实能力。
- **Assumes**: 依赖高成本人工构题、训练员对领域有足够熟悉度、参考答案“很可能唯一”但非严格可证；评分还依赖 LLM judge。论文中的最强结果来自闭源 Deep Research，且脚注明确说明其训练数据专门教会模型处理 BrowseComp 类任务，这会影响严格可比性与复现性。
- **Not designed for**: 不面向真实用户的长答案研究任务、歧义澄清、多轮需求确认、网页事务执行，也不直接衡量最终产品层面的 helpfulness 或交互体验。

### 可复用组件
- **反向构题范式**：先锁定答案，再设计大搜索空间约束，适合构造“难找但易验”的 benchmark。
- **短答案语义等价判分**：能显著降低 benchmark 的评测噪声。
- **test-time compute / aggregation 分析协议**：适合评估 agent 是否真正受益于更多搜索 budget。
- **数据污染缓解思路**：加入 canary string，降低被训练语料直接吸收的风险。

**一句话结论**：  
BrowseComp 不是“完整网页助理能力”的终局 benchmark，但它非常有效地测到了一个关键中间能力：**代理能否在开放网络里持续、创造性地把难找的信息找出来。**

## Local PDF reference

![[paperPDFs/Agentic_Reasoning_Benchmarks_Applications_of_Agentic_Reasoning_Web_Agents/arXiv_2025/2025_BrowseComp_A_Simple_Yet_Challenging_Benchmark_for_Browsing_Agents.pdf]]